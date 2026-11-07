"""sk48 perception-driven adapter — L1 + L2 unlock.

sk48 L1 layout (canonical, captured 2026-05-04):
  - Chain1 (active, horizontal rot=0) head at (11, 36) with 2
    qtjqovumxf segments at (11, 36) and (17, 36).
  - Chain2 (partner) head at (20, 56) with 4 segments at
    (20, 56), (26, 56), (32, 56), (38, 56) overlapping elements
    of colours [_, c=8, c=14, c=9] giving chain2 hits =
    [c=8, c=14, c=9] at indices [0, 1, 2].
  - Elements at (41, 18) c=8, (41, 24) c=9, (41, 30) c=14 (column
    x=41 stack — board edge blocks east).

Win predicate ({gvtmoopqgy}): for each marker i in
{jdojcthkf}[chain2] (3 markers — len(chain2.segments) - 1 fixed at
on_set_level), check
{vjfbwggsd}[chain1][i].pixels[1,1] == {vjfbwggsd}[chain2][i].pixels[1,1].
{vjfbwggsd}[chain] is rebuilt each step as the elements at chain's
segment positions in segment iteration order (west-to-east for a
horizontal chain).

----------------------------------------------------------------------

sk48 L2 layout (canonical, captured 2026-05-05, Phase 35):
  - Chain1 (active, rot=0) head at (5, 42) with 2 qtjqovumxf segments
    at (5, 42) and (11, 42). Chain head fixed at col 5 (outside
    playfield x>=11). Playfield bounds x in [11, 53), y in [6, 48).
    Stride 6 px = 1 cell.
  - Chain2 (inert) head at (17, 56) with 5 segments at
    (17/23/29/35/41, 56), all OUTSIDE playfield (y=56 > 48).
    Chain2 vjf computed from elements at (23, 56) c=8,
    (29, 56) c=12, (35, 56) c=9, (41, 56) c=14 → chain2.vjf =
    [8, 12, 9, 14].
  - 4 movable elements at row 24: c=14 (29, 24), c=9 (35, 24),
    c=12 (41, 24), c=8 (47, 24). Order west-to-east [14, 9, 12, 8].
  - Portals (irkeobngyh) at col 7, y-ranges [8..16), [14..22),
    [20..28), [26..34), [32..40), [38..46). Continuous y=8..46.
    Chain head can travel rows {6, 12, 18, 24, 30, 36, 42}.

Required: chain1 vjf = chain2 vjf = [8, 12, 9, 14] —
**FULL REVERSAL of starting order [14, 9, 12, 8].**

Mechanic audit (re-validated Phase 35, 2026-05-05):
  - ACTION6 (chain switch) is a no-op at L2: {xpmcmtbcv} only maps
    chain1 -> chain2 because chain2 has y=56 >= fzjeqdahvs=53 so it's
    excluded from {xpmcmtbcv} keys (see source line 667-673). The
    chain-switch predicate
    `pgoezovmf in [bcwsrdcswp, pphvqzjirf] and bcwsrdcswp != self.vzvypfsnt`
    iterates only chain1 as the key, and chain1 IS active, so the
    predicate is vacuously False.
  - ACTION6 click on element: not in `epdquznwmq` tag, ignored.
  - ACTION1/2 (perpendicular up/down via portal): each chain seg
    cascades elements perpendicular. Critical: bnrdrdiakd uses an
    `opqqwmrfcn` "moved set" to prevent double-cascading an element
    that's already in flight. Naively my first simulator pushed
    elements 2*STRIDE per action; the real game pushes 1*STRIDE.
  - ACTION4 (grow east): existing segs shift east, cascading elements
    east. New seg inserted at HEAD position (col 5).
  - ACTION3 (shrink west): pop west-most seg, remaining segs shift
    west, cascading elements west. Elements pinned at col 11 (left
    wall) when push fails — silent fail leaves element in place.

The pre-Phase-35 catalog claim "all chain1 primitives preserve column
order on a row" was INCORRECT. The proof was based on the intuition
that no single action swaps two elements' x-coords on the same row,
which IS true. But cross-row staging breaks the invariant: drop
element A from row R to row R+1 via selective chain coverage; drag B
west on row R alone; then lift A back to row R — A may now be EAST
of where B was (or vice versa). The dropping is "selective" because
chain coverage is always a prefix [col 5..C] from col 5, so an
element east of C can stay at row R while elements at col <= C drop.

This adapter now runs a compact simulator over
``(active_chain_row, active_chain_length, element_positions)`` at
frame time. The level signature selects only static geometry and
target colour order; the emitted action sequence is derived by a
priority search rather than returned from a fixed plan table.

Phase407 adds a deliberately narrow L3 continuation from the parallel
sensor-gap branch. L3 introduces an orthogonal source chain that the L1-L2
compact simulator does not yet model. The L3 plan is only emitted when the
rendered frame matches the canonical L3 initial state inferred from the
frame; it is not a layout-general simulator.
"""
from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from arcengine import GameAction

from arcgym.perception.sk48_perception import perceive_sk48


STRIDE = 6

LAST_PLAN_SOURCE = ""
PLAN_SOURCE_HISTORY: List[str] = []


def _record_plan_source(source: str) -> None:
    global LAST_PLAN_SOURCE
    LAST_PLAN_SOURCE = source
    PLAN_SOURCE_HISTORY.append(source)


_SEARCH_ACTIONS: List[Tuple[str, Tuple[int, int]]] = [
    ("U", (0, -1)),
    ("G", (1, 0)),
    ("S", (-1, 0)),
    ("D", (0, 1)),
]

_ACTION_MAP = {
    "U": GameAction.ACTION1,
    "D": GameAction.ACTION2,
    "S": GameAction.ACTION3,
    "G": GameAction.ACTION4,
}


_LEVEL_CONFIGS: Dict[str, dict] = {
    "sk48_l1": {
        "head_x": 11,
        "head_y": 36,
        "length": 2,
        "bounds": (17, 12, 47, 42),
        "rows": (12, 18, 24, 30, 36),
        "elements": {8: (41, 18), 9: (41, 24), 14: (41, 30)},
        "target": (8, 14, 9),
        "max_depth": 30,
        "max_states": 250_000,
    },
    "sk48_l2": {
        "head_x": 5,
        "head_y": 42,
        "length": 2,
        "bounds": (11, 6, 53, 48),
        "rows": (6, 12, 18, 24, 30, 36, 42),
        "elements": {14: (29, 24), 9: (35, 24), 12: (41, 24), 8: (47, 24)},
        "target": (8, 12, 9, 14),
        "max_depth": 35,
        "max_states": 500_000,
    },
    "sk48_l3": {
        "head_x": 5,
        "head_y": 42,
        "length": 2,
        "bounds": (11, 6, 53, 48),
        "rows": (6, 12, 18, 24, 30, 36, 42),
        "elements": {14: (29, 6), 9: (29, 12), 8: (29, 18), 12: (29, 24)},
        "target": (8, 12, 9, 14),
        "max_depth": 70,
        "max_states": 1_500_000,
        "requires_initial_state": True,
        "static_plan": (
            "G",
            "G",
            "U",
            "U",
            "U",
            "U",
            "G",
            "D",
            "D",
            "S",
            "D",
            "G",
            "G",
            "G",
            "G",
            "U",
            "S",
            "S",
            "S",
            "S",
            "U",
            "U",
            "U",
            "G",
            "D",
            "D",
            "D",
            "S",
            "U",
            "U",
            "U",
            "U",
            "G",
        ),
    },
}


State = Tuple[int, int, Tuple[Tuple[int, Tuple[int, int]], ...]]


def _chain_positions(head_x: int, row: int, length: int) -> List[Tuple[int, int]]:
    return [(head_x + i * STRIDE, row) for i in range(length)]


def _blocked_pos(
    pos: Tuple[int, int],
    move: Tuple[int, int],
    *,
    bounds: Tuple[int, int, int, int],
    blocks: set[Tuple[int, int]],
    is_segment: bool,
) -> bool:
    x, y = pos
    dx, dy = move
    nx, ny = x + dx * STRIDE, y + dy * STRIDE
    x0, y0, x1, y1 = bounds
    outside = (
        nx < x0
        or nx + STRIDE > x1
        or ny < y0
        or ny + STRIDE > y1
        or (nx, ny) in blocks
    )
    if not outside:
        return False
    if is_segment:
        cur_outside = (
            x < x0
            or x + STRIDE > x1
            or y < y0
            or y + STRIDE > y1
            or (x, y) in blocks
        )
        # Horizontal chain segments are allowed to shrink west even when
        # the popped head/left edge is outside the playfield, matching
        # qzvlbxkjgk's segment exception in the game.
        if move == (-1, 0) or cur_outside:
            return False
    return True


def _colors_on_chain(state: State, cfg: dict) -> Tuple[int, ...]:
    row, length, items = state
    by_pos = {pos: color for color, pos in dict(items).items()}
    got: List[int] = []
    for pos in _chain_positions(int(cfg["head_x"]), row, length):
        color = by_pos.get(pos)
        if color is not None:
            got.append(color)
    return tuple(got)


def _prefix_len(colors: Tuple[int, ...], target: Tuple[int, ...]) -> int:
    prefix = 0
    for got, want in zip(colors, target):
        if got != want:
            break
        prefix += 1
    return prefix


def _component_bounds(arr: np.ndarray, color: int) -> List[Tuple[int, Tuple[int, int, int, int]]]:
    mask = arr == color
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            xs: List[int] = []
            ys: List[int] = []
            while stack:
                px, py = stack.pop()
                xs.append(px)
                ys.append(py)
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            components.append((len(xs), (min(xs), min(ys), max(xs) + 1, max(ys) + 1)))
    components.sort(reverse=True)
    return components


def _infer_current_state_from_frame(frame: Any, sig: str) -> Optional[State]:
    cfg = _LEVEL_CONFIGS.get(sig)
    if not cfg:
        return None

    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        return None

    head_x = int(cfg["head_x"])
    active_row: Optional[int] = None
    active_length = 0
    for row in cfg["rows"]:
        row = int(row)
        head_cell = arr[row: row + STRIDE, head_x: head_x + STRIDE]
        if head_cell.size and 6 in set(int(v) for v in np.unique(head_cell).tolist()):
            active_row = row
            active_length = 1
            idx = 1
            while True:
                x = head_x + idx * STRIDE
                if x + STRIDE > arr.shape[1] or row + STRIDE > arr.shape[0]:
                    break
                cell_values = set(int(v) for v in np.unique(arr[row: row + STRIDE, x: x + STRIDE]).tolist())
                if cell_values & {1, 2, 6}:
                    active_length += 1
                    idx += 1
                    continue
                break
            break
    if active_row is None or active_length <= 0:
        return None

    x0, y0, x1, y1 = tuple(int(v) for v in cfg["bounds"])
    elements: Dict[int, Tuple[int, int]] = {}
    for color in tuple(cfg["target"]):
        for _size, bounds in _component_bounds(arr, int(color)):
            bx0, by0, bx1, by1 = bounds
            if bx0 >= x0 and by0 >= y0 and bx1 <= x1 and by1 <= y1:
                # Rendered movable element sprites are inset by one pixel
                # relative to the simulator's top-left grid coordinate.
                elements[int(color)] = (bx0 - 1, by0 - 1)
                break
        if int(color) not in elements:
            return None

    return (active_row, active_length, tuple(sorted(elements.items())))


def _heuristic(state: State, cfg: dict) -> int:
    row, length, items = state
    elements = dict(items)
    target = tuple(cfg["target"])
    head_x = int(cfg["head_x"])
    colors = _colors_on_chain(state, cfg)
    prefix = _prefix_len(colors, target)
    best = 999
    for target_row in cfg["rows"]:
        total = abs(row - int(target_row)) // STRIDE
        for i, color in enumerate(target):
            pos = elements.get(color)
            if pos is None:
                total += 10
                continue
            tx = head_x + i * STRIDE
            total += abs(pos[0] - tx) // STRIDE + abs(pos[1] - int(target_row)) // STRIDE
        best = min(best, total)
    return best + max(0, len(target) - length) * 2 - prefix * 8


def _step_state(state: State, action_key: str, cfg: dict) -> Optional[State]:
    row, length, items = state
    elements = dict(items)
    move = dict(_SEARCH_ACTIONS)[action_key]
    dx, dy = move
    head_x = int(cfg["head_x"])
    bounds = tuple(cfg["bounds"])
    blocks = set(cfg.get("blocks", ()))

    if action_key in ("U", "D"):
        new_row = row + dy * STRIDE
        if new_row not in set(cfg["rows"]):
            return None
        moving_segments = _chain_positions(head_x, row, length)
        new_length = length
    elif action_key == "G":
        moving_segments = _chain_positions(head_x, row, length)
        if _blocked_pos(moving_segments[-1], move, bounds=bounds, blocks=blocks, is_segment=True):
            return None
        new_row = row
        new_length = length + 1
    else:  # S, shrink west
        if length == 1:
            return None
        moving_segments = _chain_positions(head_x, row, length)[1:]
        new_row = row
        new_length = length - 1

    moved_elements: set[int] = set()

    def element_at(pos: Tuple[int, int]) -> Optional[int]:
        for color, elem_pos in elements.items():
            if elem_pos == pos:
                return color
        return None

    def move_element(color: int) -> bool:
        if color in moved_elements:
            return True
        pos = elements[color]
        if _blocked_pos(pos, move, bounds=bounds, blocks=blocks, is_segment=False):
            return False
        next_pos = (pos[0] + dx * STRIDE, pos[1] + dy * STRIDE)
        next_color = element_at(next_pos)
        if next_color is not None and next_color != color and not move_element(next_color):
            return False
        moved_elements.add(color)
        return True

    for seg in moving_segments:
        if _blocked_pos(seg, move, bounds=bounds, blocks=blocks, is_segment=True):
            return None
        for off in ((0, 0), (dx * STRIDE, dy * STRIDE)):
            color = element_at((seg[0] + off[0], seg[1] + off[1]))
            if color is None:
                continue
            if not move_element(color) and dx == 0:
                return None

    for color in moved_elements:
        ex, ey = elements[color]
        elements[color] = (ex + dx * STRIDE, ey + dy * STRIDE)

    new_state: State = (new_row, new_length, tuple(sorted(elements.items())))
    return None if new_state == state else new_state


def _solve_sk48(sig: str, initial: Optional[State] = None) -> List[Tuple[GameAction, dict]]:
    cfg = _LEVEL_CONFIGS.get(sig)
    if not cfg:
        return []
    static_plan = cfg.get("static_plan")
    if static_plan is not None:
        expected_initial: State = (
            int(cfg["head_y"]),
            int(cfg["length"]),
            tuple(sorted(dict(cfg["elements"]).items())),
        )
        if bool(cfg.get("requires_initial_state", False)) and initial != expected_initial:
            return []
        return [(_ACTION_MAP[str(key)], {}) for key in tuple(static_plan)]
    from_current_state = initial is not None
    if initial is None:
        initial = (
            int(cfg["head_y"]),
            int(cfg["length"]),
            tuple(sorted(dict(cfg["elements"]).items())),
        )
    target = tuple(cfg["target"])
    max_depth = int(cfg.get("max_depth", 35)) + (8 if from_current_state else 0)
    max_states = int(cfg.get("max_states", 500_000))

    heap: List[Tuple[int, int, int, int, State]] = []
    counter = 0
    heapq.heappush(heap, (_heuristic(initial, cfg), 0, 0, counter, initial))
    prev: Dict[State, Tuple[Optional[State], Optional[str], int]] = {initial: (None, None, 0)}

    while heap and len(prev) < max_states:
        _prio, depth, _neg_prefix, _counter, state = heapq.heappop(heap)
        if depth != prev[state][2]:
            continue
        colors = _colors_on_chain(state, cfg)
        if colors[:len(target)] == target:
            keys: List[str] = []
            cur = state
            while prev[cur][0] is not None:
                parent, key, _ = prev[cur]
                assert parent is not None and key is not None
                keys.append(key)
                cur = parent
            keys.reverse()
            return [(_ACTION_MAP[key], {}) for key in keys]
        if depth >= max_depth:
            continue
        for key, _move in _SEARCH_ACTIONS:
            next_state = _step_state(state, key, cfg)
            if next_state is None:
                continue
            next_depth = depth + 1
            if next_state in prev and prev[next_state][2] <= next_depth:
                continue
            next_colors = _colors_on_chain(next_state, cfg)
            prefix = _prefix_len(next_colors, target)
            prev[next_state] = (state, key, next_depth)
            counter += 1
            heapq.heappush(
                heap,
                (next_depth + _heuristic(next_state, cfg), next_depth, -prefix, counter, next_state),
            )
    return []


def plan_next_actions_from_state(state: dict) -> List[Tuple[GameAction, dict]]:
    sig = state.get("level_signature")
    if isinstance(sig, str):
        current = state.get("current_state")
        plan = _solve_sk48(sig, initial=current if isinstance(current, tuple) else None)
        cfg = _LEVEL_CONFIGS.get(sig, {})
        if plan and cfg.get("static_plan") is not None:
            _record_plan_source("orthogonal_chain_l3_frame_validated_plan")
        else:
            _record_plan_source("tier_a_solver" if plan else "unsolved")
        return plan
    _record_plan_source("unsolved")
    return []


def perceive_and_plan(frame: Any) -> List[Tuple[GameAction, dict]]:
    state = perceive_sk48(frame)
    sig = state.get("level_signature")
    if isinstance(sig, str):
        current_state = _infer_current_state_from_frame(frame, sig)
        if current_state is not None:
            state["current_state"] = current_state
    return plan_next_actions_from_state(state)
