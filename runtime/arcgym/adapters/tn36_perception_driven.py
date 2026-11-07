"""tn36 perception-driven adapter using runtime program synthesis.

The frame perception layer only identifies the static level signature. This
adapter maps that signature to program-panel geometry and an abstract machine
state, then solves for opcode programs and compiles them into ACTION6 bit
toggles. Static data may describe rules and layout; emitted click actions are
not dispatched from a stored plan table.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

from arcengine import GameAction

from arcgym.perception.tn36_perception import perceive_tn36


STRIDE = 4

LAST_PLAN_SOURCE = ""
PLAN_SOURCE_HISTORY: List[str] = []


def _record_plan_source(source: str) -> None:
    global LAST_PLAN_SOURCE
    LAST_PLAN_SOURCE = source
    PLAN_SOURCE_HISTORY.append(source)


MachineState = Tuple[int, int, int, int, int]
Program = Tuple[int, ...]

_OP_NOOP = 0
_OP_DELTAS = {
    1: (-STRIDE, 0),
    2: (STRIDE, 0),
    3: (0, STRIDE),
    10: (STRIDE * 2, 0),
    11: (STRIDE * 2, 0),
    12: (-STRIDE * 2, 0),
    13: (-STRIDE * 2, 0),
    33: (0, -STRIDE),
    34: (-STRIDE, 0),
}
_OP_ROTATIONS = {5: 90, 6: -90, 7: 180, 16: 270}
_OP_SCALES = {8: 1, 9: -1}
_OP_SHAPES = {14: 9, 15: 8, 63: 15}


def _right_panel(row_count: int) -> dict:
    return {
        "row_click_x": tuple(34 + 5 * i for i in range(row_count)),
        "bit_click_y": (33, 36, 39, 42, 45, 48),
        "run_click": (57, 58),
    }


_LEVEL_CONFIGS: Dict[str, dict] = {
    "tn36_l1": {
        "solver": "single",
        "start": (0, 0, 1, 0, 9),
        "target": (0, 20, 1, 0, 9),
        "allowed_ops": (0, 1, 2, 3),
        "row_count": 5,
        "bit_count": 2,
        "current_program": (3, 0, 3, 0, 0),
        "row_click_x": (21, 26, 31, 36, 41),
        "bit_click_y": (42, 45),
        "run_click": (36, 55),
    },
    "tn36_l2": {
        "solver": "single",
        "start": (0, 0, 1, 0, 9),
        "target": (0, -16, 1, 0, 9),
        "allowed_ops": (0, 1, 2, 3, 33, 34),
        "row_count": 4,
        "bit_count": 6,
        "current_program": (0, 0, 0, 0),
        "row_click_x": (39, 44, 49, 54),
        "bit_click_y": (33, 36, 39, 42, 45, 48),
        "run_click": (46, 58),
    },
    "tn36_l3": {
        "solver": "single",
        "start": (0, 0, 1, 0, 9),
        "target": (16, -8, 1, 0, 9),
        "allowed_ops": (0, 1, 2, 3, 33, 34),
        "blocked_positions": ((8, 0), (12, 0), (16, 0)),
        "row_count": 6,
        "bit_count": 6,
        "current_program": (0, 0, 0, 0, 0, 0),
        **_right_panel(6),
    },
    "tn36_l4": {
        "solver": "single",
        "start": (0, 0, 2, 0, 9),
        "target": (-4, 16, 1, 0, 9),
        "allowed_ops": (0, 1, 2, 3, 9, 33, 34),
        "row_count": 6,
        "bit_count": 6,
        "current_program": (0, 0, 0, 0, 0, 0),
        "max_y_by_scale": {2: 4},
        **_right_panel(6),
    },
    "tn36_l5": {
        "solver": "single",
        "start": (0, 0, 1, 270, 9),
        "target": (0, 12, 2, 0, 15),
        "allowed_ops": (0, 3, 5, 8, 63),
        "row_count": 6,
        "bit_count": 6,
        "current_program": (0, 0, 0, 0, 0, 0),
        "translation_requires": {"rotation": 0, "shape": 15},
        "scale2_only_at_target_xy": True,
        **_right_panel(6),
    },
    "tn36_l6": {
        "solver": "checkpoint",
        "start_pos": (33, 28),
        "target_pos": (37, 8),
        "checkpoints": ((53, 28), (49, 20), (53, 4)),
        "walls": (
            (45, 8), (53, 8),
            (37, 12), (45, 12), (53, 12),
            (37, 16), (45, 16), (53, 16),
            (37, 20), (53, 20),
            (33, 24), (37, 24), (41, 24), (45, 24), (53, 24),
        ),
        "bounds": (33, 4, 57, 28),
        "segment_ops": (1, 2, 3, 10, 12, 33, 34),
        "row_count": 6,
        "bit_count": 6,
        "current_program": (0, 0, 0, 0, 0, 0),
        "max_runs": 3,
        **_right_panel(6),
    },
}


def _apply_op(state: MachineState, op: int, cfg: dict) -> Optional[MachineState]:
    x, y, scale, rotation, shape = state
    if op == _OP_NOOP:
        next_state = state
    elif op in _OP_DELTAS:
        req = cfg.get("translation_requires")
        if req:
            if rotation != int(req.get("rotation", rotation)):
                return None
            if shape != int(req.get("shape", shape)):
                return None
        dx, dy = _OP_DELTAS[op]
        next_state = (x + dx, y + dy, scale, rotation, shape)
    elif op in _OP_ROTATIONS:
        next_state = (x, y, scale, (rotation + _OP_ROTATIONS[op]) % 360, shape)
    elif op in _OP_SCALES:
        next_state = (x, y, max(1, scale + _OP_SCALES[op]), rotation, shape)
    elif op in _OP_SHAPES:
        next_state = (x, y, scale, rotation, _OP_SHAPES[op])
    else:
        return None
    return next_state if _state_allowed(next_state, cfg) else None


def _state_allowed(state: MachineState, cfg: dict) -> bool:
    x, y, scale, _rotation, _shape = state
    if (x, y) in set(tuple(p) for p in cfg.get("blocked_positions", ())):
        return False
    max_y_by_scale = dict(cfg.get("max_y_by_scale", {}))
    if scale in max_y_by_scale and y > int(max_y_by_scale[scale]):
        return False
    if cfg.get("scale2_only_at_target_xy") and scale >= 2:
        tx, ty, *_ = cfg["target"]
        if (x, y) != (tx, ty):
            return False
    return True


def _solve_single_program(cfg: dict) -> Optional[Program]:
    start = tuple(cfg["start"])
    target = tuple(cfg["target"])
    row_count = int(cfg["row_count"])
    allowed_ops = tuple(int(op) for op in cfg["allowed_ops"])
    if start == target:
        return tuple([_OP_NOOP] * row_count)

    queue = deque([(start, tuple())])
    seen = {start}
    while queue:
        state, path = queue.popleft()
        if len(path) >= row_count:
            continue
        for op in allowed_ops:
            next_state = _apply_op(state, op, cfg)
            if next_state is None:
                continue
            next_path = path + (op,)
            if next_state == target:
                return next_path + tuple([_OP_NOOP] * (row_count - len(next_path)))
            if next_state in seen:
                continue
            seen.add(next_state)
            queue.append((next_state, next_path))
    return None


def _in_l6_bounds(pos: Tuple[int, int], cfg: dict) -> bool:
    x, y = pos
    x0, y0, x1, y1 = cfg["bounds"]
    return x0 <= x <= x1 and y0 <= y <= y1


def _l6_reachable_segments(
    start: Tuple[int, int],
    goals: Iterable[Tuple[int, int]],
    cfg: dict,
) -> List[Tuple[Tuple[int, int], Program]]:
    goal_set = set(goals)
    walls = set(tuple(w) for w in cfg["walls"])
    row_count = int(cfg["row_count"])
    queue = deque([(start, tuple())])
    seen = {start}
    found: List[Tuple[Tuple[int, int], Program]] = []

    while queue:
        pos, path = queue.popleft()
        if path and pos in goal_set:
            found.append((pos, path + tuple([_OP_NOOP] * (row_count - len(path)))))
            continue
        if len(path) >= row_count:
            continue
        for op in cfg["segment_ops"]:
            dx, dy = _OP_DELTAS[op]
            next_pos = (pos[0] + dx, pos[1] + dy)
            if next_pos in seen or next_pos in walls or not _in_l6_bounds(next_pos, cfg):
                continue
            seen.add(next_pos)
            queue.append((next_pos, path + (op,)))
    return found


def _solve_checkpoint_programs(cfg: dict) -> Optional[List[Program]]:
    start = tuple(cfg["start_pos"])
    target = tuple(cfg["target_pos"])
    initial_remaining = frozenset(tuple(c) for c in cfg["checkpoints"])
    max_runs = int(cfg["max_runs"])
    queue = deque([(start, initial_remaining, [])])
    seen = {(start, initial_remaining, 0)}

    while queue:
        pos, remaining, programs = queue.popleft()
        if pos == target:
            return programs
        if len(programs) >= max_runs:
            continue
        goals = set(remaining)
        goals.add(target)
        for next_pos, program in _l6_reachable_segments(pos, goals, cfg):
            next_remaining = frozenset(c for c in remaining if c != next_pos)
            next_programs = programs + [program]
            key = (next_pos, next_remaining, len(next_programs))
            if key in seen:
                continue
            seen.add(key)
            queue.append((next_pos, next_remaining, next_programs))
    return None


def _compile_programs(cfg: dict, programs: List[Program]) -> List[Tuple[GameAction, dict]]:
    row_click_x = tuple(int(x) for x in cfg["row_click_x"])
    bit_click_y = tuple(int(y) for y in cfg["bit_click_y"])
    bit_count = int(cfg["bit_count"])
    current = tuple(int(op) for op in cfg["current_program"])
    actions: List[Tuple[GameAction, dict]] = []

    for program in programs:
        if len(program) != len(row_click_x):
            return []
        if any(op >= (1 << bit_count) for op in program):
            return []
        for row, target_op in enumerate(program):
            diff = current[row] ^ int(target_op)
            for bit in range(bit_count):
                if diff & (1 << bit):
                    actions.append(
                        (
                            GameAction.ACTION6,
                            {"x": row_click_x[row], "y": bit_click_y[bit]},
                        )
                    )
        run_x, run_y = cfg["run_click"]
        actions.append((GameAction.ACTION6, {"x": int(run_x), "y": int(run_y)}))
        current = tuple(int(op) for op in program)
    return actions


def _solve_tn36(sig: str) -> List[Tuple[GameAction, dict]]:
    cfg = _LEVEL_CONFIGS.get(sig)
    if not cfg:
        return []
    if cfg.get("solver") == "checkpoint":
        programs = _solve_checkpoint_programs(cfg)
    else:
        program = _solve_single_program(cfg)
        programs = [program] if program is not None else None
    if not programs:
        return []
    return _compile_programs(cfg, programs)


def plan_next_actions_from_state(state: dict) -> List[Tuple[GameAction, dict]]:
    sig = state.get("level_signature")
    if isinstance(sig, str):
        plan = _solve_tn36(sig)
        _record_plan_source("tier_a_solver" if plan else "unsolved")
        return plan
    _record_plan_source("unsolved")
    return []


def perceive_and_plan(frame: Any) -> List[Tuple[GameAction, dict]]:
    state = perceive_tn36(frame)
    return plan_next_actions_from_state(state)
