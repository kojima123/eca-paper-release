"""Frame-grounded stride navigation with directional trigger hazards.

The module deliberately owns no environment identity or memorised route.  It
compiles a small transition model from ``perceive_tu93`` output and leases only
actions whose leading edge is walkable and whose inferred settling point is
not a directional lock trigger.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from typing import Any, Iterable

import numpy as np

from arcgym.perception.tu93_perception import perceive_tu93


FAMILY_ID = "stride_hazard_navigation"
_MODELED_MARKER_COLORS = {8, 9, 14}
_NON_MARKER_COLORS = {0, 2, 5, 6}

_DIRECTIONS = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}
_ACTION_ALIASES = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
}
_LOCK_TRIGGER_UNITS = {
    0: (0, -1),
    90: (1, 0),
    180: (0, 1),
    270: (-1, 0),
}
_READ_CACHE: dict[tuple[tuple[int, ...], str, str], tuple[dict[str, Any], int | None]] = {}
_BIND_CACHE: dict[tuple[tuple[int, ...], str, str], dict[str, Any]] = {}
_COUPLED_PLAN_CACHE: dict[
    tuple[tuple[tuple[int, ...], str, str], tuple[str, ...], int],
    dict[str, Any] | None,
] = {}
_COUPLED_SEQUENCE_STATE: dict[str, Any] | None = None
_ROLE_PALETTE: dict[str, int] = {}
_STRIDE_MARKER_STEP = {
    "ACTION1": (0, -6),
    "ACTION2": (0, 6),
    "ACTION3": (-6, 0),
    "ACTION4": (6, 0),
}
_STRIDE_MARKER_ACTION_DOT = {
    "ACTION1": (1, 0),
    "ACTION2": (1, 2),
    "ACTION3": (0, 1),
    "ACTION4": (2, 1),
}


def _cache_key(frame: np.ndarray) -> tuple[tuple[int, ...], str, str]:
    digest = hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
    return tuple(int(dim) for dim in frame.shape), str(frame.dtype), digest


def _remember(
    cache: dict[tuple[tuple[int, ...], str, str], Any],
    key: tuple[tuple[int, ...], str, str],
    value: Any,
) -> Any:
    if len(cache) >= 32:
        cache.clear()
    cache[key] = value
    return value


def _latest_frame(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 3:
        array = array[-1]
    if array.ndim != 2:
        raise ValueError("stride-hazard perception requires a 2-D frame")
    return array


def _position(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [int(value[0]), int(value[1])]


def _normalise_action(action: Any) -> str | None:
    if isinstance(action, dict):
        action = action.get("action")
    if hasattr(action, "name"):
        action = action.name
    if action is None:
        return None
    name = str(action).split(".")[-1].upper()
    name = _ACTION_ALIASES.get(name, name)
    return name if name in _DIRECTIONS else None


def _allowed_actions(valid_actions: Iterable[Any] | None) -> set[str]:
    if valid_actions is None:
        return set(_DIRECTIONS)
    provided = list(valid_actions)
    if not provided:
        return set(_DIRECTIONS)
    allowed = {_normalise_action(action) for action in provided}
    return {action for action in allowed if action is not None}


def _raw_lock_trigger_cell(
    lock: tuple[int, int, int],
    stride: int,
) -> tuple[int, int] | None:
    unit = _LOCK_TRIGGER_UNITS.get(int(lock[2]))
    if unit is None:
        return None
    return (int(lock[0]) + unit[0] * stride, int(lock[1]) + unit[1] * stride)


def _lock_trigger_cells(
    locks: Iterable[tuple[int, int, int]],
    stride: int,
) -> set[tuple[int, int]]:
    return {
        cell
        for lock in locks
        if (cell := _raw_lock_trigger_cell(lock, stride)) is not None
    }


def _raw_lock_overlaps(
    position: tuple[int, int],
    lock: tuple[int, int, int],
    *,
    footprint: int,
) -> bool:
    left, top = position
    right = left + footprint
    bottom = top + footprint
    lock_left, lock_top = int(lock[0]), int(lock[1])
    lock_right = lock_left + footprint
    lock_bottom = lock_top + footprint
    return (
        left < lock_right
        and right > lock_left
        and top < lock_bottom
        and bottom > lock_top
    )


def _candidate_path(
    state: dict[str, Any],
    stride: int,
    *,
    avoid_triggers: bool,
    actor_footprint: int = 3,
) -> tuple[list[tuple[str, tuple[int, int]]] | None, int]:
    start = state.get("walker_pos")
    exits = {tuple(exit_pos) for exit_pos in state.get("exits", [])}
    if start is None or not exits:
        return None, 0

    start = tuple(start)
    walkable = state.get("walkable") or set()
    initial_locks = tuple(sorted(state.get("locks") or ()))
    edge = max(1, stride // 2)
    queue: deque[
        tuple[
            tuple[int, int],
            tuple[tuple[int, int, int], ...],
            list[tuple[str, tuple[int, int]]],
        ]
    ] = deque(
        [(start, initial_locks, [])]
    )
    visited = {(start, initial_locks)}
    while queue:
        current, remaining_locks, path = queue.popleft()
        forbidden = (
            _lock_trigger_cells(remaining_locks, stride) if avoid_triggers else set()
        )
        for action, (unit_x, unit_y) in _DIRECTIONS.items():
            leading = (current[0] + unit_x * edge, current[1] + unit_y * edge)
            if leading not in walkable:
                continue
            endpoint = (
                current[0] + unit_x * stride,
                current[1] + unit_y * stride,
            )
            if endpoint in forbidden:
                continue
            next_locks = tuple(
                lock
                for lock in remaining_locks
                if not _raw_lock_overlaps(
                    endpoint,
                    lock,
                    footprint=actor_footprint,
                )
            )
            state_key = (endpoint, next_locks)
            if state_key in visited:
                continue
            next_path = [*path, (action, endpoint)]
            if endpoint in exits:
                return next_path, len(visited) + 1
            visited.add(state_key)
            queue.append((endpoint, next_locks, next_path))
    return None, len(visited)


def _visible_footprint(frame: np.ndarray, state: dict[str, Any]) -> int:
    """Measure the smallest visible square role footprint."""
    sizes: list[int] = []
    height, width = frame.shape
    for x, y in state.get("exits") or ():
        color = int(frame[y, x])
        run_x = 0
        while x + run_x < width and int(frame[y, x + run_x]) == color:
            run_x += 1
        run_y = 0
        while y + run_y < height and int(frame[y + run_y, x]) == color:
            run_y += 1
        if run_x > 0 and run_y > 0:
            sizes.append(min(run_x, run_y))
    return min(sizes) if sizes else 1


def _value_components(frame: np.ndarray) -> list[dict[str, Any]]:
    height, width = frame.shape
    seen = np.zeros(frame.shape, dtype=bool)
    components: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x]:
                continue
            value = int(frame[y, x])
            seen[y, x] = True
            stack = [(x, y)]
            cells: list[tuple[int, int]] = []
            while stack:
                current_x, current_y = stack.pop()
                cells.append((current_x, current_y))
                for delta_x, delta_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_x = current_x + delta_x
                    next_y = current_y + delta_y
                    if (
                        0 <= next_x < width
                        and 0 <= next_y < height
                        and not seen[next_y, next_x]
                        and int(frame[next_y, next_x]) == value
                    ):
                        seen[next_y, next_x] = True
                        stack.append((next_x, next_y))
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            components.append(
                {
                    "value": value,
                    "area": len(cells),
                    "x": min(xs),
                    "y": min(ys),
                    "width": max(xs) - min(xs) + 1,
                    "height": max(ys) - min(ys) + 1,
                }
            )
    return components


def _sprite_candidates(frame: np.ndarray) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for component in _value_components(frame):
        if (
            int(component["area"]) != 8
            or int(component["width"]) != 3
            or int(component["height"]) != 3
        ):
            continue
        x = int(component["x"])
        y = int(component["y"])
        block = frame[y : y + 3, x : x + 3]
        values, counts = np.unique(block, return_counts=True)
        pairs = sorted(
            ((int(count), int(value)) for value, count in zip(values, counts)),
            reverse=True,
        )
        if len(pairs) != 2 or pairs[0][0] != 8 or pairs[1][0] != 1:
            continue
        dot_at = np.argwhere(block == pairs[1][1])
        if dot_at.shape != (1, 2):
            continue
        dot_y, dot_x = (int(value) for value in dot_at[0].tolist())
        candidates.append(
            {
                "position": (x, y),
                "outer": int(pairs[0][1]),
                "dot": int(pairs[1][1]),
                "dot_offset": (dot_x, dot_y),
            }
        )
    return candidates


def _record_role_palette(frame: np.ndarray, state: dict[str, Any]) -> None:
    actor = state.get("walker_pos")
    if actor is None:
        return
    x, y = (int(actor[0]), int(actor[1]))
    if y + 3 > frame.shape[0] or x + 3 > frame.shape[1]:
        return
    block = frame[y : y + 3, x : x + 3]
    values, counts = np.unique(block, return_counts=True)
    pairs = sorted(
        ((int(count), int(value)) for value, count in zip(values, counts)),
        reverse=True,
    )
    if len(pairs) == 2 and pairs[0][0] == 8 and pairs[1][0] == 1:
        _ROLE_PALETTE["actor_outer"] = int(pairs[0][1])
        _ROLE_PALETTE["actor_dot"] = int(pairs[1][1])


def _structural_stride_state(frame: np.ndarray) -> dict[str, Any] | None:
    """Recover early stride levels from geometry when palette roles changed."""

    components = _value_components(frame)
    sprites = _sprite_candidates(frame)
    values, counts = np.unique(frame, return_counts=True)
    value_counts = {
        int(value): int(count) for value, count in zip(values.tolist(), counts.tolist())
    }
    exits = [
        (int(component["x"]), int(component["y"]))
        for component in components
        if int(component["area"]) == 9
        and int(component["width"]) == 3
        and int(component["height"]) == 3
        and value_counts.get(int(component["value"]), 0) <= 36
    ]
    if not sprites or not exits:
        return None
    sprite_values = {int(sprite["outer"]) for sprite in sprites} | {
        int(sprite["dot"]) for sprite in sprites
    }
    walkable_values = sorted(
        {
            int(value)
            for value, count in value_counts.items()
            if int(count) >= 16
            and int(count) < int(frame.size * 0.5)
            and int(value) not in sprite_values
        }
    )
    if not walkable_values:
        return None

    remembered = (
        _ROLE_PALETTE.get("actor_outer"),
        _ROLE_PALETTE.get("actor_dot"),
    )
    best: tuple[tuple[int, ...], dict[str, Any]] | None = None
    orientation_by_dot = {
        (1, 0): 0,
        (2, 1): 90,
        (1, 2): 180,
        (0, 1): 270,
    }
    for actor in sprites:
        actor_match = int(
            remembered
            == (int(actor["outer"]), int(actor["dot"]))
        )
        lock_sprites = [item for item in sprites if item is not actor]
        locks = [
            (
                int(item["position"][0]),
                int(item["position"][1]),
                int(orientation_by_dot[item["dot_offset"]]),
            )
            for item in lock_sprites
            if item["dot_offset"] in orientation_by_dot
        ]
        for walkable_value in walkable_values:
            walkable = {
                (int(x), int(y))
                for y, x in np.argwhere(frame == int(walkable_value)).tolist()
            }
            state = {
                "walker_pos": tuple(actor["position"]),
                "exits": list(exits),
                "walkable": walkable,
                "locks": locks,
                "trigger_cells": set(),
            }
            stride = _infer_stride(state, frame)
            if stride is None:
                continue
            safe_path, safe_reachable = _candidate_path(
                state, stride, avoid_triggers=True
            )
            raw_path, raw_reachable = _candidate_path(
                state, stride, avoid_triggers=False
            )
            rank = (
                int(safe_path is not None),
                actor_match,
                int(raw_path is not None),
                max(int(safe_reachable), int(raw_reachable)),
                -len(safe_path or raw_path or ()),
            )
            if best is None or rank > best[0]:
                state["trigger_cells"] = _lock_trigger_cells(locks, stride)
                best = (rank, state)
    if best is None or best[0][0] == 0:
        return None
    _ROLE_PALETTE["actor_outer"] = int(
        frame[best[1]["walker_pos"][1], best[1]["walker_pos"][0]]
    )
    actor_block = frame[
        best[1]["walker_pos"][1] : best[1]["walker_pos"][1] + 3,
        best[1]["walker_pos"][0] : best[1]["walker_pos"][0] + 3,
    ]
    values, counts = np.unique(actor_block, return_counts=True)
    dot_values = [int(value) for value, count in zip(values, counts) if int(count) == 1]
    if dot_values:
        _ROLE_PALETTE["actor_dot"] = dot_values[0]
    return best[1]


def _unmodeled_stride_markers(
    frame: np.ndarray,
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    known_positions = {
        tuple(int(value) for value in state["walker_pos"])
        for _ in (0,)
        if state.get("walker_pos") is not None
    }
    known_positions.update(
        tuple(int(value) for value in position)
        for position in state.get("exits") or ()
    )
    known_positions.update(
        (int(lock[0]), int(lock[1])) for lock in state.get("locks") or ()
    )
    markers: list[dict[str, Any]] = []
    for marker in _marker_blocks(frame):
        position = (int(marker["x"]), int(marker["y"]))
        if position in known_positions:
            continue
        markers.append(
            {
                "position": [position[0], position[1]],
                "primary_color": int(marker["primary_value"]),
                "secondary_color": int(marker["secondary_value"]),
            }
        )
    return markers


def _coupled_marker_plan(
    frame: np.ndarray,
    allowed: Iterable[str],
    *,
    max_actions: int = 80,
) -> dict[str, Any] | None:
    key = (_cache_key(frame), tuple(sorted(set(allowed))), int(max_actions))
    if key in _COUPLED_PLAN_CACHE:
        return _COUPLED_PLAN_CACHE[key]
    try:
        from arcgym.eca.primitive_synthesis import (
            _stride_marker_coupled_motion_plan as eca_coupled_motion_plan,
        )
    except (ImportError, ModuleNotFoundError):
        return None
    plan = eca_coupled_motion_plan(
        frame,
        set(allowed),
        max_actions=int(max_actions),
    )
    if len(_COUPLED_PLAN_CACHE) >= 32:
        _COUPLED_PLAN_CACHE.clear()
    _COUPLED_PLAN_CACHE[key] = plan
    return plan


def _marker_blocks(frame: np.ndarray) -> list[dict[str, int]]:
    try:
        from arcgym.eca.primitive_synthesis import _marker_blocks as eca_marker_blocks
    except (ImportError, ModuleNotFoundError):
        return []
    return eca_marker_blocks(frame, block_size=3)


def _active_marker(
    frame: np.ndarray,
    *,
    primary: int,
    secondary: int,
) -> dict[str, int] | None:
    candidates = [
        marker
        for marker in _marker_blocks(frame)
        if int(marker.get("primary_value", -1)) == int(primary)
        and int(marker.get("secondary_value", -1)) == int(secondary)
    ]
    if len(candidates) != 1:
        return None
    return candidates[0]


def _coupled_prefix_states(
    start: tuple[int, int],
    dot: tuple[int, int],
    plan: Iterable[str],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    position = (int(start[0]), int(start[1]))
    current_dot = (int(dot[0]), int(dot[1]))
    states = [(position, current_dot)]
    for action in plan:
        delta = _STRIDE_MARKER_STEP.get(str(action))
        if delta is None:
            break
        position = (position[0] + int(delta[0]), position[1] + int(delta[1]))
        current_dot = _STRIDE_MARKER_ACTION_DOT.get(str(action), current_dot)
        states.append((position, current_dot))
    return states


def _sequence_state_from_plan(
    frame: np.ndarray,
    plan: dict[str, Any],
) -> dict[str, Any] | None:
    actions = [str(action) for action in plan.get("plan") or []]
    if not actions:
        return None
    active = _active_marker(
        frame,
        primary=int(plan["active_value"]),
        secondary=int(plan["active_secondary_value"]),
    )
    if active is None:
        return None
    start = (int(plan["active_start"][0]), int(plan["active_start"][1]))
    dot = (int(active["secondary_dx"]), int(active["secondary_dy"]))
    return {
        "plan": actions,
        "active_start": start,
        "active_dot": dot,
        "active_value": int(plan["active_value"]),
        "active_secondary_value": int(plan["active_secondary_value"]),
        "target": tuple(int(v) for v in plan["target"]),
        "target_value": int(plan["target_value"]),
        "moving_primary_value": int(plan.get("moving_primary_value")),
        "common_secondary_value": int(plan.get("common_secondary_value")),
        "moving_marker_count": int(plan.get("moving_marker_count") or 0),
        "dormant_marker_count": int(plan.get("dormant_marker_count") or 0),
        "source": str(plan.get("source") or "coupled_marker_motion"),
        "last_index": 0,
    }


def _cached_coupled_marker_plan(
    frame: np.ndarray,
    allowed: Iterable[str],
    *,
    max_actions: int = 80,
) -> dict[str, Any] | None:
    global _COUPLED_SEQUENCE_STATE
    allowed_set = set(allowed)
    if _COUPLED_SEQUENCE_STATE is not None:
        active = _active_marker(
            frame,
            primary=int(_COUPLED_SEQUENCE_STATE["active_value"]),
            secondary=int(_COUPLED_SEQUENCE_STATE["active_secondary_value"]),
        )
        if active is not None:
            current = (
                (int(active["x"]), int(active["y"])),
                (int(active["secondary_dx"]), int(active["secondary_dy"])),
            )
            states = _coupled_prefix_states(
                tuple(_COUPLED_SEQUENCE_STATE["active_start"]),
                tuple(_COUPLED_SEQUENCE_STATE["active_dot"]),
                _COUPLED_SEQUENCE_STATE["plan"],
            )
            matches = [index for index, state in enumerate(states) if state == current]
            if matches:
                last_index = int(_COUPLED_SEQUENCE_STATE.get("last_index") or 0)
                forward_matches = [index for index in matches if index >= last_index]
                index = min(forward_matches) if forward_matches else min(matches)
                remaining = list(_COUPLED_SEQUENCE_STATE["plan"])[index:]
                if remaining and remaining[0] in allowed_set:
                    _COUPLED_SEQUENCE_STATE["last_index"] = int(index)
                    return {
                        **_COUPLED_SEQUENCE_STATE,
                        "plan": remaining,
                        "sequence_index": int(index),
                        "sequence_plan_length": int(
                            len(_COUPLED_SEQUENCE_STATE["plan"])
                        ),
                        "from_sequence_cache": True,
                    }
        _COUPLED_SEQUENCE_STATE = None

    plan = _coupled_marker_plan(frame, allowed_set, max_actions=max_actions)
    if plan is None:
        return None
    state = _sequence_state_from_plan(frame, plan)
    if state is None:
        return plan
    _COUPLED_SEQUENCE_STATE = state
    return {**state, "sequence_index": 0, "sequence_plan_length": len(state["plan"])}


def _infer_stride(state: dict[str, Any], frame: np.ndarray) -> int | None:
    """Infer the movement lattice by testing frame-grounded reachability.

    The correct lattice should expose an exit through repeated leading-edge
    checks.  This distinguishes the canonical six-pixel lattice from both
    smaller pixel steps and larger corridor-skipping candidates without a
    stored stride constant.
    """
    if state.get("walker_pos") is None or not state.get("exits"):
        return None
    footprint = _visible_footprint(frame, state)
    lower = max(2, footprint * 2)
    upper = max(lower, min(frame.shape) // 4)
    preferred = footprint * 2
    candidates = list(range(lower, upper + 1))
    if preferred in candidates:
        candidates.remove(preferred)
        candidates.insert(0, preferred)
    ranked: list[tuple[int, int, int, int]] = []
    for stride in candidates:
        safe_path, safe_reachable = _candidate_path(
            state,
            stride,
            avoid_triggers=True,
            actor_footprint=footprint,
        )
        if state.get("locks"):
            raw_path, raw_reachable = _candidate_path(
                state,
                stride,
                avoid_triggers=False,
                actor_footprint=footprint,
            )
        else:
            raw_path, raw_reachable = safe_path, safe_reachable
        if stride == preferred and (safe_path is not None or raw_path is not None):
            return stride
        ranked.append(
            (
                int(safe_path is not None),
                int(raw_path is not None),
                max(safe_reachable, raw_reachable),
                -stride,
            )
        )
    if not ranked:
        return None
    best = max(ranked)
    if best[:3] == (0, 0, 1):
        return None
    return -best[3]


def _read(frame: Any) -> tuple[np.ndarray, dict[str, Any], int | None]:
    latest = _latest_frame(frame)
    key = _cache_key(latest)
    cached = _READ_CACHE.get(key)
    if cached is not None:
        state, stride = cached
        return latest, state, stride
    state = perceive_tu93(latest)
    if (
        state.get("walker_pos") is None
        or not state.get("exits")
        or not state.get("walkable")
    ):
        coupled = _coupled_marker_plan(latest, set(_DIRECTIONS), max_actions=80)
        if coupled is None:
            structural = _structural_stride_state(latest)
            if structural is not None:
                state = structural
    _record_role_palette(latest, state)
    stride = _infer_stride(state, latest)
    _remember(_READ_CACHE, key, (state, stride))
    return latest, state, stride


def _locks(state: dict[str, Any], stride: int | None) -> list[dict[str, Any]]:
    orientation_names = {0: "up", 90: "right", 180: "down", 270: "left"}
    locks = []
    for x, y, orientation in sorted(state.get("locks") or ()):
        trigger_by_orientation = {}
        if stride is not None:
            trigger = _raw_lock_trigger_cell((x, y, orientation), stride)
            if trigger is not None:
                trigger_by_orientation = {int(orientation): _position(trigger)}
        locks.append(
            {
                "position": [int(x), int(y)],
                "orientation_degrees": int(orientation),
                "orientation": orientation_names.get(int(orientation), "unknown"),
                "trigger_cell": trigger_by_orientation.get(int(orientation)),
            }
        )
    return locks


def _overlapping_locks(
    position: tuple[int, int],
    state: dict[str, Any],
    stride: int | None,
    *,
    footprint: int,
) -> list[dict[str, Any]]:
    left, top = position
    right = left + footprint
    bottom = top + footprint
    overlaps = []
    for lock in _locks(state, stride):
        lock_x, lock_y = lock["position"]
        lock_right = int(lock_x) + footprint
        lock_bottom = int(lock_y) + footprint
        if left < lock_right and right > lock_x and top < lock_bottom and bottom > lock_y:
            overlaps.append(lock)
    return overlaps


def _topology_fingerprint(
    walkable: Iterable[tuple[int, int]],
    ignored_actor_positions: Iterable[tuple[int, int]],
    actor_footprint: int,
) -> str:
    ignored: set[tuple[int, int]] = set()
    for actor_x, actor_y in ignored_actor_positions:
        for y in range(actor_y, actor_y + actor_footprint):
            for x in range(actor_x, actor_x + actor_footprint):
                ignored.add((x, y))
    payload = ";".join(
        f"{int(x)},{int(y)}" for x, y in sorted(set(walkable) - ignored)
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def bind_current(frame: Any) -> dict[str, Any]:
    """Bind visible roles and the inferred lattice from the current frame."""
    latest = _latest_frame(frame)
    key = _cache_key(latest)
    cached = _BIND_CACHE.get(key)
    if cached is not None:
        return cached
    latest, state, stride = _read(latest)
    unmodeled_markers = _unmodeled_stride_markers(latest, state)
    coupled_plan = (
        _cached_coupled_marker_plan(latest, set(_DIRECTIONS), max_actions=80)
        if unmodeled_markers
        else None
    )
    safe_path = None
    reachable = 0
    if stride is not None and not unmodeled_markers:
        safe_path, reachable = _candidate_path(state, stride, avoid_triggers=True)
    trigger_cells = sorted(tuple(cell) for cell in state.get("trigger_cells") or ())
    actor = {"role": "actor", "settling_position": _position(state.get("walker_pos"))}
    exits = [
        {"role": "exit", "settling_position": _position(exit_pos)}
        for exit_pos in sorted(state.get("exits") or ())
    ]
    locks = [
        {"role": "lock", **lock}
        for lock in _locks(state, stride)
    ]
    base_bound = (
        actor["settling_position"] is not None
        and bool(exits)
        and stride is not None
    )
    coupled_supported = coupled_plan is not None
    marker_actor = None
    marker_target = None
    if coupled_plan is not None:
        marker_actor = {
            "role": "marker_actor",
            "settling_position": _position(coupled_plan.get("active_start")),
            "primary_color": int(coupled_plan.get("active_value")),
            "secondary_color": int(coupled_plan.get("active_secondary_value")),
        }
        marker_target = {
            "role": "marker_target",
            "settling_position": _position(coupled_plan.get("target")),
            "color": int(coupled_plan.get("target_value")),
        }
    role_bindings = [actor, *exits, *locks]
    if marker_actor is not None and marker_target is not None:
        role_bindings.extend([marker_actor, marker_target])
    return _remember(_BIND_CACHE, key, {
        "mechanic_id": FAMILY_ID,
        "family_id": FAMILY_ID,
        "status": (
            "bound"
            if coupled_supported
            else "unsupported_dynamic_hazards"
            if base_bound and unmodeled_markers
            else ("bound" if base_bound else "unbound")
        ),
        "supported": bool(coupled_supported or (base_bound and not unmodeled_markers)),
        "mode": "marker_coupled_motion" if coupled_supported else "stride_hazard",
        "frame_shape": [int(latest.shape[0]), int(latest.shape[1])],
        "actor": actor,
        "exits": [item["settling_position"] for item in exits],
        "stride": stride,
        "leading_edge_offset": stride // 2 if stride is not None else None,
        "walkability": {
            "rule": "leading_edge_cell_is_walkable",
            "visible_cells": [
                [int(x), int(y)] for x, y in sorted(state.get("walkable") or ())
            ],
            "reachable_settling_count": int(reachable),
        },
        "locks": locks,
        "trigger_cells": [[int(x), int(y)] for x, y in trigger_cells],
        "unmodeled_stride_markers": unmodeled_markers,
        "coupled_marker_plan_available": coupled_plan is not None,
        "coupled_marker_plan_length": (
            int(len(coupled_plan.get("plan") or [])) if coupled_plan else 0
        ),
        "safe_route_available": safe_path is not None or coupled_plan is not None,
        "role_bindings": role_bindings,
    })


def compile_schema(frame: Any, level: Any) -> dict[str, Any]:
    """Compile a JSON-safe mechanic schema; ``level`` is metadata only."""
    binding = bind_current(frame)
    if isinstance(level, np.generic):
        level = level.item()
    try:
        json.dumps(level)
    except (TypeError, ValueError):
        level = str(level)
    return {
        "schema_id": "tufa.stride_hazard_navigation.v1",
        "mechanic_id": FAMILY_ID,
        "family_id": FAMILY_ID,
        "level": level,
        "roles": binding["role_bindings"],
        "required_bindings": ["actor", "exit", "inferred_stride"],
        "state_variables": [
            "actor_settling_position",
            "inferred_stride",
            "leading_edge_walkability",
            "lock_orientation",
            "lock_trigger_cells",
            "marker_secondary_dot",
            "coupled_marker_motion",
        ],
        "transition": {
            "actions": sorted(_DIRECTIONS),
            "settling_rule": "endpoint = actor + direction * inferred_stride; marker mode = active marker + six-pixel action delta",
            "lease_preconditions": [
                "leading edge is visibly walkable",
                "endpoint differs from actor position",
            "endpoint is not a directional lock trigger",
            "non-lock stride markers either match the coupled-marker motion reader or block the lease",
        ],
        },
        "operator": {
            "name": "safe_stride_step",
            "actions": ["UP", "DOWN", "LEFT", "RIGHT"],
            "transition": "settle exactly one inferred stride away",
        },
        "invariants": [
            "exit and lock bindings remain stable during a non-terminal step",
            "walkability outside the old and new actor footprints remains stable",
        ],
        "falsifiers": [
            "the actor does not leave its previous settling position",
            "the actor settles anywhere except the predicted endpoint",
            "the active marker settles anywhere except the predicted marker endpoint",
            "walkability or hazard bindings change unexpectedly",
            "a non-lock stride marker appears but cannot be bound as coupled motion",
        ],
        "current_binding": binding,
    }


def propose_plan(
    frame: Any,
    valid_actions: Iterable[Any] | None = None,
) -> dict[str, Any] | None:
    """Return one certified next action from the shortest safe visible route."""
    latest, state, stride = _read(frame)
    allowed = _allowed_actions(valid_actions)
    unmodeled_markers = _unmodeled_stride_markers(latest, state)
    if unmodeled_markers:
        coupled = _cached_coupled_marker_plan(latest, allowed, max_actions=80)
        if coupled is None or not coupled.get("plan"):
            return None
        action = str(coupled["plan"][0])
        if action not in allowed:
            return None
        active = _active_marker(
            latest,
            primary=int(coupled["active_value"]),
            secondary=int(coupled["active_secondary_value"]),
        )
        if active is None:
            return None
        active_position = (int(active["x"]), int(active["y"]))
        return {
            "mechanic_id": FAMILY_ID,
            "actions": [{"action": action}],
            "action_count": 1,
            "certified": True,
            "certificate_reason": "frame-grounded coupled marker transition",
            "transition_rule": (
                "active marker moves one six-pixel stride while visible hazard "
                "markers follow the coupled-motion model"
            ),
            "grounded_observation": {
                "invariants": [
                    "active marker is the unique rare-secondary marker",
                    "common-secondary markers form moving/dormant hazard families",
                ],
                "falsifiers": [
                    "active marker does not settle at predicted endpoint",
                    "coupled marker reader no longer binds the frame",
                ],
            },
            "detail": {
                "mode": "marker_coupled_motion",
                "predicted_marker_endpoint": _position(
                    (
                        active_position[0] + _STRIDE_MARKER_STEP[action][0],
                        active_position[1] + _STRIDE_MARKER_STEP[action][1],
                    )
                ),
                "active_start": _position(coupled.get("active_start")),
                "active_position": _position(active_position),
                "target": _position(coupled.get("target")),
                "safe_route_length": int(
                    coupled.get("sequence_plan_length")
                    or len(coupled.get("plan") or [])
                ),
                "remaining_route_length": int(len(coupled.get("plan") or [])),
                "private_plan_length": int(len(coupled.get("plan") or [])),
                "sequence_index": int(coupled.get("sequence_index") or 0),
                "moving_marker_count": int(coupled.get("moving_marker_count") or 0),
                "dormant_marker_count": int(coupled.get("dormant_marker_count") or 0),
                "source": str(coupled.get("source") or "coupled_marker_motion"),
            },
        }
    if stride is None:
        return None
    path, _reachable = _candidate_path(state, stride, avoid_triggers=True)
    if path is None:
        return None
    action, endpoint = path[0]
    if action not in allowed:
        return None
    return {
        "mechanic_id": FAMILY_ID,
        "actions": [{"action": action}],
        "action_count": 1,
        "certified": True,
        "certificate_reason": "frame-grounded safe settling transition",
        "transition_rule": "one inferred stride to a walkable non-trigger endpoint",
        "grounded_observation": {
            "invariants": [
                "leading edge is visibly walkable",
                "predicted endpoint is not a lock trigger",
            ],
            "falsifiers": ["no progress", "unexpected settling endpoint"],
        },
        "detail": {
            "predicted_settling_endpoint": _position(endpoint),
            "inferred_stride": int(stride),
            "safe_route_length": len(path),
            "private_plan_length": len(path),
            "avoids_trigger_cells": True,
        },
    }


def predict_transition(frame: Any, action: Any) -> dict[str, Any]:
    """Predict one action and decide whether it is safe to lease."""
    latest, state, stride = _read(frame)
    action_name = _normalise_action(action)
    actor = state.get("walker_pos")
    base = {
        "family_id": FAMILY_ID,
        "action": action_name,
        "prior_actor_position": _position(actor),
        "inferred_stride": stride,
        "lease_safe": False,
    }
    if action_name is None:
        return {**base, "reason": "unbound_action"}
    unmodeled_markers = _unmodeled_stride_markers(latest, state)
    if unmodeled_markers:
        coupled = _cached_coupled_marker_plan(latest, set(_DIRECTIONS), max_actions=80)
        if (
            coupled is not None
            and coupled.get("plan")
            and str(coupled["plan"][0]) == action_name
        ):
            active = _active_marker(
                latest,
                primary=int(coupled["active_value"]),
                secondary=int(coupled["active_secondary_value"]),
            )
            if active is None:
                return {
                    **base,
                    "reason": "coupled_marker_actor_not_visible",
                }
            delta = _STRIDE_MARKER_STEP[action_name]
            active_position = (int(active["x"]), int(active["y"]))
            endpoint = (active_position[0] + delta[0], active_position[1] + delta[1])
            return {
                **base,
                "mode": "marker_coupled_motion",
                "lease_safe": True,
                "reason": "safe_coupled_marker_progress",
                "prior_marker_position": _position(active_position),
                "predicted_marker_endpoint": _position(endpoint),
                "active_marker_primary_color": int(coupled["active_value"]),
                "active_marker_secondary_color": int(
                    coupled["active_secondary_value"]
                ),
                "target": _position(coupled.get("target")),
                "target_value": int(coupled.get("target_value")),
                "moving_primary_value": int(coupled.get("moving_primary_value")),
                "common_secondary_value": int(coupled.get("common_secondary_value")),
                "moving_marker_count": int(coupled.get("moving_marker_count") or 0),
                "dormant_marker_count": int(
                    coupled.get("dormant_marker_count") or 0
                ),
                "coupled_plan_length": int(
                    coupled.get("sequence_plan_length")
                    or len(coupled.get("plan") or [])
                ),
                "remaining_route_length": int(len(coupled.get("plan") or [])),
                "sequence_index": int(coupled.get("sequence_index") or 0),
                "coupled_plan_source": str(
                    coupled.get("source") or "coupled_marker_motion"
                ),
            }
        return {
            **base,
            "reason": "unmodeled_dynamic_hazard_marker",
            "unmodeled_stride_markers": unmodeled_markers,
        }
    if actor is None or stride is None:
        return {**base, "reason": "unbound_actor_or_stride"}

    unit_x, unit_y = _DIRECTIONS[action_name]
    edge = stride // 2
    leading = (actor[0] + unit_x * edge, actor[1] + unit_y * edge)
    endpoint = (actor[0] + unit_x * stride, actor[1] + unit_y * stride)
    actor_footprint = _visible_footprint(latest, state)
    base.update(
        {
            "leading_edge_cell": _position(leading),
            "predicted_settling_endpoint": _position(endpoint),
        }
    )
    if leading not in (state.get("walkable") or set()) or endpoint == actor:
        return {**base, "reason": "no_progress_or_blocked_leading_edge"}

    triggered_locks = [
        lock
        for lock in _locks(state, stride)
        if lock.get("trigger_cell") == _position(endpoint)
    ]
    if endpoint in set(state.get("trigger_cells") or ()):
        return {
            **base,
            "reason": "lock_trigger_endpoint",
            "triggered_locks": triggered_locks,
        }

    exits = [_position(exit_pos) for exit_pos in sorted(state.get("exits") or ())]
    locks = _locks(state, stride)
    removed_locks = _overlapping_locks(
        endpoint,
        state,
        stride,
        footprint=actor_footprint,
    )
    remaining_locks = [lock for lock in locks if lock not in removed_locks]
    base.update(
        {
            "lease_safe": True,
            "reason": "safe_progress",
            "expected_exits": exits,
            "expected_locks": remaining_locks,
            "removed_locks": removed_locks,
            "actor_footprint": actor_footprint,
            "topology_fingerprint": _topology_fingerprint(
                state.get("walkable") or (),
                [tuple(actor), endpoint],
                actor_footprint,
            ),
        }
    )
    return base


def verify_transition(expected: dict[str, Any], current_frame: Any) -> bool:
    """Verify a leased transition against a new frame."""
    if not isinstance(expected, dict) or not expected.get("lease_safe"):
        return False
    if expected.get("mode") == "marker_coupled_motion":
        try:
            latest = _latest_frame(current_frame)
        except (TypeError, ValueError):
            return False
        wanted = expected.get("predicted_marker_endpoint")
        prior = expected.get("prior_marker_position")
        if wanted is None or prior is None or wanted == prior:
            return False
        active_primary = expected.get("active_marker_primary_color")
        active_secondary = expected.get("active_marker_secondary_color")
        for marker in _marker_blocks(latest):
            if (
                int(marker.get("x", -1)) == int(wanted[0])
                and int(marker.get("y", -1)) == int(wanted[1])
                and int(marker.get("primary_value", -1)) == int(active_primary)
                and int(marker.get("secondary_value", -1)) == int(active_secondary)
            ):
                return True
        return False
    try:
        _latest, state, stride = _read(current_frame)
    except (TypeError, ValueError):
        return False

    actor = state.get("walker_pos")
    wanted = expected.get("predicted_settling_endpoint")
    if actor is None or wanted is None:
        return False
    if _position(actor) == expected.get("prior_actor_position"):
        return False
    if _position(actor) != wanted:
        return False
    if stride != expected.get("inferred_stride"):
        return False
    if [_position(pos) for pos in sorted(state.get("exits") or ())] != expected.get(
        "expected_exits"
    ):
        return False
    if _locks(state, stride) != expected.get("expected_locks"):
        return False

    prior = tuple(expected["prior_actor_position"])
    endpoint = tuple(wanted)
    fingerprint = _topology_fingerprint(
        state.get("walkable") or (),
        [prior, endpoint],
        int(expected.get("actor_footprint", 0) or 0),
    )
    if fingerprint != expected.get("topology_fingerprint"):
        return False
    return True
