"""Visible-frame mechanic for reflection-linked cursor navigation.

The reader binds a pair of visually identical lattice cursors, derives the
board geometry from their repeated cell footprint, and treats blended cells
as unsafe unless their texture identifies them as outlined movable blockers.
The planner searches the two cursor poses jointly: vertical actions translate
both identities together while horizontal actions reflect their displacement.

Some boards add selectable blockers.  Their outlined-on-floor texture is
visible without a palette contract, so the existing generic frame planner can
search blocker configurations before the same joint cursor BFS is applied.
The resulting suffix stays private.  Only one action is leased, and every
non-terminal successor is rebound from the next frame before execution may
continue.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from typing import Any

import numpy as np

from arcgym.eca import primitive_synthesis as _mirror

from .base import MechanicContext, as_frame, stable_data, stable_digest


FAMILY_ID = "coupled_mirror_navigation"

_DIRECTIONS: dict[str, tuple[int, int]] = {
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
    "MOUSE": "ACTION6",
}
_ALL_ACTIONS = frozenset({*_DIRECTIONS, "ACTION6"})

_READ_CACHE: dict[tuple[tuple[int, ...], str, str], dict[str, Any] | None] = {}
_PLAN_CACHE: dict[tuple[Any, ...], tuple[dict[str, Any], ...]] = {}
_PLAN_DIAGNOSTICS: dict[tuple[Any, ...], dict[str, Any]] = {}
_IDENTITY_ORIENTATION: dict[
    tuple[Any, ...], tuple[tuple[int, int], tuple[int, int]]
] = {}
_ADMITTED_TOPOLOGIES: set[str] = set()


def _json(value: Any) -> Any:
    return stable_data(value)


def _safe_metadata(value: Any) -> Any:
    normalized = _json(value)
    try:
        json.dumps(normalized, ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)
    return normalized


def _frame_key(frame: np.ndarray) -> tuple[tuple[int, ...], str, str]:
    digest = hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
    return tuple(int(value) for value in frame.shape), str(frame.dtype), digest


def _remember_read(
    key: tuple[tuple[int, ...], str, str],
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if len(_READ_CACHE) >= 64:
        _READ_CACHE.clear()
    _READ_CACHE[key] = value
    return value


def _position(value: Sequence[int] | None) -> list[int] | None:
    if value is None:
        return None
    return [int(value[0]), int(value[1])]


def _positions(values: Iterable[tuple[int, int]]) -> list[list[int]]:
    return [[int(x), int(y)] for x, y in sorted(set(values))]


def _component_part(
    component: Mapping[str, int],
    *,
    min_x: int,
    min_y: int,
    size: int,
) -> dict[str, int]:
    return {
        "value": int(component["value"]),
        "area": int(size * size),
        "min_x": int(min_x),
        "max_x": int(min_x + size - 1),
        "min_y": int(min_y),
        "max_y": int(min_y + size - 1),
        "width": int(size),
        "height": int(size),
        "cx": int(min_x + size // 2),
        "cy": int(min_y + size // 2),
    }


def _split_joined_cursor_state(frame: np.ndarray) -> dict[str, Any] | None:
    """Recover two cell identities when equal solid cursors touch edgewise.

    A vertically joined pair is a single 1x2-cell component.  The generic
    reader already handles the horizontal case; checking both orientations
    here avoids losing identity evidence midway through an otherwise exact
    visible route.  Requiring the component colour to occur nowhere else in
    the frame keeps this fallback intentionally narrow.
    """

    candidates: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for component in _mirror._valued_components(frame):
        width = int(component["width"])
        height = int(component["height"])
        area = int(component["area"])
        short = min(width, height)
        long = max(width, height)
        if short < 2 or short > 16 or long != short * 2:
            continue
        if area != width * height:
            continue
        value = int(component["value"])
        if int(np.count_nonzero(frame == value)) != area:
            continue

        if width > height:
            first = _component_part(
                component,
                min_x=int(component["min_x"]),
                min_y=int(component["min_y"]),
                size=short,
            )
            second = _component_part(
                component,
                min_x=int(component["min_x"]) + short,
                min_y=int(component["min_y"]),
                size=short,
            )
        else:
            first = _component_part(
                component,
                min_x=int(component["min_x"]),
                min_y=int(component["min_y"]),
                size=short,
            )
            second = _component_part(
                component,
                min_x=int(component["min_x"]),
                min_y=int(component["min_y"]) + short,
                size=short,
            )
        state = _mirror._mirror_joint_state_from_pair(frame, first, second)
        if not isinstance(state, Mapping):
            continue
        try:
            cursor_a = tuple(int(value) for value in state["cursor_a"])
            cursor_b = tuple(int(value) for value in state["cursor_b"])
            grid_w, grid_h = (int(value) for value in state["grid_size"])
            cursor_count = int(state.get("cursor_cell_count", 0))
            walkable_count = len(state.get("walkable") or ())
        except (KeyError, TypeError, ValueError):
            continue
        if cursor_count != 2 or cursor_a == cursor_b:
            continue
        if abs(cursor_a[0] - cursor_b[0]) + abs(cursor_a[1] - cursor_b[1]) != 1:
            continue
        if grid_w < 3 or grid_h < 3 or walkable_count < 5:
            continue
        score = (abs(grid_w - grid_h), -short, grid_w * grid_h)
        candidates.append((score, dict(state)))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _cell_patch(
    frame: np.ndarray,
    raw: Mapping[str, Any],
    site: tuple[int, int],
) -> np.ndarray | None:
    try:
        origin_x, origin_y = (int(value) for value in raw["origin"])
        cell = int(raw["cell"])
    except (KeyError, TypeError, ValueError):
        return None
    x, y = site
    x0 = origin_x + int(x) * cell
    y0 = origin_y + int(y) * cell
    patch = frame[y0 : y0 + cell, x0 : x0 + cell]
    if patch.shape != (cell, cell):
        return None
    return patch


def _outlined_blockers(
    frame: np.ndarray,
    raw: Mapping[str, Any],
) -> tuple[frozenset[tuple[int, int]], dict[tuple[int, int], int]]:
    """Separate outlined movable cells from checker/blended death cells."""

    try:
        floor_value = int(raw["walkable_value"])
        cell = int(raw["cell"])
    except (KeyError, TypeError, ValueError):
        return frozenset(), {}
    if cell < 3:
        return frozenset(), {}

    blockers: set[tuple[int, int]] = set()
    visual_values: dict[tuple[int, int], int] = {}
    for raw_site in raw.get("hazards") or ():
        site = (int(raw_site[0]), int(raw_site[1]))
        patch = _cell_patch(frame, raw, site)
        if patch is None:
            continue
        border = np.concatenate(
            (patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1])
        )
        if border.size == 0 or not bool(np.all(border == floor_value)):
            continue
        interior = patch[1:-1, 1:-1]
        foreign = interior[interior != floor_value]
        if foreign.size == 0:
            continue
        values, counts = np.unique(foreign, return_counts=True)
        blockers.add(site)
        visual_values[site] = int(values[int(np.argmax(counts))])
    return frozenset(blockers), visual_values


def _selected_blocker(
    blockers: frozenset[tuple[int, int]],
    visual_values: Mapping[tuple[int, int], int],
) -> tuple[int, int] | None:
    """Infer selection from one minority outline state among peer blockers."""

    if len(blockers) < 3:
        return None
    value_counts = Counter(int(visual_values[site]) for site in blockers)
    majority = [value for value, count in value_counts.items() if count >= 2]
    minority = [value for value, count in value_counts.items() if count == 1]
    if not majority or len(minority) != 1:
        return None
    selected = [
        site for site in blockers if int(visual_values[site]) == int(minority[0])
    ]
    return selected[0] if len(selected) == 1 else None


def _read_state(frame: Any) -> dict[str, Any] | None:
    arr = as_frame(frame)
    if arr is None:
        return None
    key = _frame_key(arr)
    if key in _READ_CACHE:
        return _READ_CACHE[key]

    raw = _mirror._mirror_joint_cursor_state(arr)
    if raw is None:
        raw = _split_joined_cursor_state(arr)
    if not isinstance(raw, Mapping):
        return _remember_read(key, None)
    try:
        cursor_a = tuple(int(value) for value in raw["cursor_a"])
        cursor_b = tuple(int(value) for value in raw["cursor_b"])
        grid_size = tuple(int(value) for value in raw["grid_size"])
        origin = tuple(int(value) for value in raw["origin"])
        cell = int(raw["cell"])
        cursor_count = int(raw.get("cursor_cell_count", 0))
        walkable = frozenset(
            (int(site[0]), int(site[1])) for site in raw.get("walkable") or ()
        )
        hazards = frozenset(
            (int(site[0]), int(site[1])) for site in raw.get("hazards") or ()
        )
    except (KeyError, TypeError, ValueError):
        return _remember_read(key, None)
    if (
        len(grid_size) != 2
        or grid_size[0] < 3
        or grid_size[1] < 3
        or cell < 2
        or cursor_count != 2
        or cursor_a == cursor_b
        or cursor_a not in walkable
        or cursor_b not in walkable
        or len(walkable) < 5
    ):
        return _remember_read(key, None)
    blockers, blocker_visual_values = _outlined_blockers(arr, raw)
    death = frozenset(hazards - blockers)
    full_floor = frozenset(set(walkable) | set(blockers))
    selected = _selected_blocker(blockers, blocker_visual_values)
    topology_digest = stable_digest(
        {
            "grid_size": grid_size,
            "full_floor": sorted(full_floor),
            "death": sorted(death),
        }
    )
    symmetric_anchor = bool(
        grid_size[0] == grid_size[1]
        and grid_size[0] % 2 == 1
        and cursor_a[0] + cursor_b[0] == grid_size[0] - 1
    )
    if not symmetric_anchor and topology_digest not in _ADMITTED_TOPOLOGIES:
        return _remember_read(key, None)
    if symmetric_anchor:
        _ADMITTED_TOPOLOGIES.add(topology_digest)
    state = {
        "raw": dict(raw),
        "frame_shape": tuple(int(value) for value in arr.shape),
        "frame_digest": key[2],
        "cursor_a": cursor_a,
        "cursor_b": cursor_b,
        "grid_size": grid_size,
        "origin": origin,
        "cell": cell,
        "walkable": walkable,
        "full_floor": full_floor,
        "death": death,
        "blockers": blockers,
        "selected": selected,
        "topology_digest": topology_digest,
        "identity_visually_ambiguous": bool(cursor_a[0] == cursor_b[0]),
    }
    return _remember_read(key, state)


def _normalise_action_name(action: Any) -> str | None:
    if isinstance(action, Mapping):
        action = action.get("action")
    if hasattr(action, "name"):
        action = action.name
    if action is None:
        return None
    name = str(action).split(".")[-1].strip().upper()
    name = _ACTION_ALIASES.get(name, name)
    return name if name in _ALL_ACTIONS else None


def _allowed_actions(valid_actions: Iterable[Any] | None) -> set[str]:
    if valid_actions is None:
        return set(_ALL_ACTIONS)
    provided = list(valid_actions)
    if not provided:
        return set(_ALL_ACTIONS)
    return {
        name for item in provided if (name := _normalise_action_name(item)) is not None
    }


def _joint_step(
    cursor_a: tuple[int, int],
    cursor_b: tuple[int, int],
    action: str,
    walkable: frozenset[tuple[int, int]],
    death: frozenset[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], bool] | None:
    delta = _DIRECTIONS.get(action)
    if delta is None:
        return None
    dx, dy = delta
    intended_a = (cursor_a[0] + dx, cursor_a[1] + dy)
    intended_b = (cursor_b[0] - dx, cursor_b[1] + dy)
    if intended_a in death or intended_b in death:
        return None
    next_a = intended_a if intended_a in walkable else cursor_a
    next_b = intended_b if intended_b in walkable else cursor_b
    if next_a == cursor_a and next_b == cursor_b:
        return None
    adjacent_swap = (
        abs(cursor_a[0] - cursor_b[0]) == 1
        and cursor_a[1] == cursor_b[1]
        and (next_a == cursor_b or next_b == cursor_a)
    )
    return next_a, next_b, bool(adjacent_swap or next_a == next_b)


def _joint_bfs(
    state: Mapping[str, Any],
    allowed: set[str],
) -> tuple[list[str] | None, int]:
    directions = [action for action in _DIRECTIONS if action in allowed]
    if not directions:
        return None, 0
    blockers = set(state.get("blockers") or ())
    walkable = frozenset(set(state["full_floor"]) - blockers)
    death = frozenset(state.get("death") or ())
    initial = (tuple(state["cursor_a"]), tuple(state["cursor_b"]))
    queue: deque[tuple[tuple[tuple[int, int], tuple[int, int]], list[str]]] = deque(
        [(initial, [])]
    )
    visited = {initial}
    while queue:
        (cursor_a, cursor_b), path = queue.popleft()
        for action in directions:
            successor = _joint_step(cursor_a, cursor_b, action, walkable, death)
            if successor is None:
                continue
            next_a, next_b, terminal = successor
            next_path = path + [action]
            if terminal:
                return next_path, len(visited)
            pair = (next_a, next_b)
            if pair in visited:
                continue
            visited.add(pair)
            queue.append((pair, next_path))
    return None, len(visited)


def _state_key_parts(
    topology_digest: str,
    cursor_a: tuple[int, int],
    cursor_b: tuple[int, int],
    blockers: Iterable[tuple[int, int]],
    selected: tuple[int, int] | None,
) -> tuple[Any, ...]:
    return (
        str(topology_digest),
        tuple(int(value) for value in cursor_a),
        tuple(int(value) for value in cursor_b),
        tuple(sorted((int(x), int(y)) for x, y in blockers)),
        None if selected is None else tuple(int(value) for value in selected),
    )


def _orientation_key(
    topology_digest: str,
    cursor_a: tuple[int, int],
    cursor_b: tuple[int, int],
    blockers: Iterable[tuple[int, int]],
    selected: tuple[int, int] | None,
) -> tuple[Any, ...]:
    return (
        str(topology_digest),
        tuple(sorted((tuple(cursor_a), tuple(cursor_b)))),
        tuple(sorted((int(x), int(y)) for x, y in blockers)),
        None if selected is None else tuple(int(value) for value in selected),
    )


def _oriented_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Restore private identities after visually indistinguishable crossings."""

    if state is None:
        return None
    cursor_a = tuple(state["cursor_a"])
    cursor_b = tuple(state["cursor_b"])
    key = _orientation_key(
        str(state["topology_digest"]),
        cursor_a,
        cursor_b,
        state.get("blockers") or (),
        state.get("selected"),
    )
    oriented = _IDENTITY_ORIENTATION.get(key)
    if oriented is None or set(oriented) != {cursor_a, cursor_b}:
        return state
    if oriented == (cursor_a, cursor_b):
        return state
    rebound = dict(state)
    rebound["cursor_a"], rebound["cursor_b"] = oriented
    return rebound


def _state_key(state: Mapping[str, Any]) -> tuple[Any, ...]:
    return _state_key_parts(
        str(state["topology_digest"]),
        tuple(state["cursor_a"]),
        tuple(state["cursor_b"]),
        state.get("blockers") or (),
        state.get("selected"),
    )


def _click_site(state: Mapping[str, Any], x: int, y: int) -> tuple[int, int] | None:
    origin_x, origin_y = (int(value) for value in state["origin"])
    grid_w, grid_h = (int(value) for value in state["grid_size"])
    cell = int(state["cell"])
    rel_x, rel_y = int(x) - origin_x, int(y) - origin_y
    if rel_x < 0 or rel_y < 0:
        return None
    grid_x, grid_y = rel_x // cell, rel_y // cell
    if not (0 <= grid_x < grid_w and 0 <= grid_y < grid_h):
        return None
    return int(grid_x), int(grid_y)


def _click_payload(state: Mapping[str, Any], site: tuple[int, int]) -> dict[str, int]:
    origin_x, origin_y = (int(value) for value in state["origin"])
    cell = int(state["cell"])
    return {
        "x": int(origin_x + int(site[0]) * cell + cell // 2),
        "y": int(origin_y + int(site[1]) * cell + cell // 2),
    }


def _public_action(state: Mapping[str, Any], step: Mapping[str, Any]) -> dict[str, Any]:
    action = str(step["action"])
    if action != "ACTION6":
        return {"action": action}
    payload = _click_payload(state, tuple(step["site"]))
    return {"action": action, **payload}


def _internal_action(
    state: Mapping[str, Any],
    action: Any,
) -> dict[str, Any] | None:
    name = _normalise_action_name(action)
    if name is None:
        return None
    if name != "ACTION6":
        return {"action": name}
    if not isinstance(action, Mapping):
        return None
    try:
        x = int(action["x"] if "x" in action else action["col"])
        y = int(action["y"] if "y" in action else action["row"])
    except (KeyError, TypeError, ValueError):
        return None
    site = _click_site(state, x, y)
    if site is None:
        return None
    return {"action": name, "site": site}


def _apply_abstract_step(
    *,
    cursor_a: tuple[int, int],
    cursor_b: tuple[int, int],
    blockers: frozenset[tuple[int, int]],
    selected: tuple[int, int] | None,
    step: Mapping[str, Any],
    full_floor: frozenset[tuple[int, int]],
    death: frozenset[tuple[int, int]],
) -> (
    tuple[
        tuple[int, int],
        tuple[int, int],
        frozenset[tuple[int, int]],
        tuple[int, int] | None,
        bool,
    ]
    | None
):
    action = str(step.get("action", ""))
    if action == "ACTION6":
        site = tuple(int(value) for value in step.get("site", ()))
        if len(site) != 2 or site not in full_floor:
            return None
        next_selected = site if site in blockers else None
        return cursor_a, cursor_b, blockers, next_selected, False
    if action not in _DIRECTIONS:
        return None
    if selected is not None:
        dx, dy = _DIRECTIONS[action]
        destination = (selected[0] + dx, selected[1] + dy)
        occupied = set(blockers) - {selected}
        if (
            destination not in full_floor
            or destination in death
            or destination in occupied
            or destination in {cursor_a, cursor_b}
        ):
            return None
        next_blockers = frozenset(occupied | {destination})
        return cursor_a, cursor_b, next_blockers, destination, False

    walkable = frozenset(set(full_floor) - set(blockers))
    successor = _joint_step(cursor_a, cursor_b, action, walkable, death)
    if successor is None:
        return None
    next_a, next_b, terminal = successor
    return next_a, next_b, blockers, None, terminal


def _simplify_steps(
    state: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cursor_a = tuple(state["cursor_a"])
    cursor_b = tuple(state["cursor_b"])
    blockers = frozenset(state.get("blockers") or ())
    selected = state.get("selected")
    full_floor = frozenset(state["full_floor"])
    death = frozenset(state.get("death") or ())
    simplified: list[dict[str, Any]] = []
    for raw_step in steps:
        step = {
            "action": str(raw_step["action"]),
            **(
                {"site": tuple(int(value) for value in raw_step["site"])}
                if raw_step.get("site") is not None
                else {}
            ),
        }
        successor = _apply_abstract_step(
            cursor_a=cursor_a,
            cursor_b=cursor_b,
            blockers=blockers,
            selected=selected,
            step=step,
            full_floor=full_floor,
            death=death,
        )
        if successor is None:
            return []
        next_a, next_b, next_blockers, next_selected, terminal = successor
        changed = (next_a, next_b, next_blockers, next_selected) != (
            cursor_a,
            cursor_b,
            blockers,
            selected,
        )
        if changed or terminal:
            simplified.append(step)
        cursor_a, cursor_b = next_a, next_b
        blockers, selected = next_blockers, next_selected
    return simplified


def _remember_plan(
    state: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> bool:
    simplified = _simplify_steps(state, steps)
    if not simplified:
        return False
    if len(_PLAN_CACHE) >= 2048:
        _PLAN_CACHE.clear()
        _PLAN_DIAGNOSTICS.clear()
        _IDENTITY_ORIENTATION.clear()

    cursor_a = tuple(state["cursor_a"])
    cursor_b = tuple(state["cursor_b"])
    blockers = frozenset(state.get("blockers") or ())
    selected = state.get("selected")
    full_floor = frozenset(state["full_floor"])
    death = frozenset(state.get("death") or ())
    topology_digest = str(state["topology_digest"])
    terminal_reached = False
    for index, step in enumerate(simplified):
        orientation_key = _orientation_key(
            topology_digest, cursor_a, cursor_b, blockers, selected
        )
        _IDENTITY_ORIENTATION[orientation_key] = (cursor_a, cursor_b)
        key = _state_key_parts(topology_digest, cursor_a, cursor_b, blockers, selected)
        suffix = tuple(dict(item) for item in simplified[index:])
        _PLAN_CACHE[key] = suffix
        _PLAN_DIAGNOSTICS[key] = {
            **_json(dict(diagnostics)),
            "private_plan_length": int(len(suffix)),
        }
        successor = _apply_abstract_step(
            cursor_a=cursor_a,
            cursor_b=cursor_b,
            blockers=blockers,
            selected=selected,
            step=step,
            full_floor=full_floor,
            death=death,
        )
        if successor is None:
            return False
        cursor_a, cursor_b, blockers, selected, terminal_reached = successor
        if terminal_reached:
            return index == len(simplified) - 1
    return terminal_reached


def _blocker_steps(
    state: Mapping[str, Any],
    allowed: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_plan, diagnostics = _mirror._mirror_selectable_blocker_reposition_plan(
        state["raw"],
        set(allowed),
        max_expansions=20000,
    )
    steps: list[dict[str, Any]] = []
    for action, payload in raw_plan:
        name = _normalise_action_name(action)
        if name is None or name not in allowed:
            return [], {**dict(diagnostics), "blocked_reason": "unavailable_action"}
        if name != "ACTION6":
            steps.append({"action": name})
            continue
        if not isinstance(payload, Mapping):
            return [], {**dict(diagnostics), "blocked_reason": "ungrounded_click"}
        try:
            site = _click_site(state, int(payload["x"]), int(payload["y"]))
        except (KeyError, TypeError, ValueError):
            site = None
        if site is None:
            return [], {**dict(diagnostics), "blocked_reason": "click_outside_grid"}
        steps.append({"action": name, "site": site})
    return _simplify_steps(state, steps), dict(diagnostics)


def _plan_for_state(
    state: Mapping[str, Any],
    allowed: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    oriented = _oriented_state(dict(state))
    assert oriented is not None
    state = oriented
    key = _state_key(state)
    cached = _PLAN_CACHE.get(key)
    if cached and all(str(step["action"]) in allowed for step in cached):
        return [dict(step) for step in cached], dict(
            _PLAN_DIAGNOSTICS.get(key, {"planner": "private_suffix_cache"})
        )

    if state.get("selected") is None:
        direct, visited = _joint_bfs(state, allowed)
        if direct:
            steps = [{"action": action} for action in direct]
            diagnostics = {
                "planner": "joint_cursor_bfs",
                "joint_state_count": int(visited),
            }
            if _remember_plan(state, steps, diagnostics):
                return [dict(step) for step in _PLAN_CACHE[_state_key(state)]], dict(
                    _PLAN_DIAGNOSTICS[_state_key(state)]
                )

    if state.get("blockers") and "ACTION6" in allowed:
        steps, diagnostics = _blocker_steps(state, allowed)
        if steps and _remember_plan(state, steps, diagnostics):
            return [dict(step) for step in _PLAN_CACHE[_state_key(state)]], dict(
                _PLAN_DIAGNOSTICS[_state_key(state)]
            )
        return [], diagnostics
    return [], {"planner": "joint_cursor_bfs", "blocked_reason": "no_merge_route"}


def _public_projection(
    *,
    cursor_a: tuple[int, int],
    cursor_b: tuple[int, int],
    blockers: Iterable[tuple[int, int]],
    selected: tuple[int, int] | None,
) -> dict[str, Any]:
    return {
        "cursor_positions": _positions((cursor_a, cursor_b)),
        "blocker_positions": _positions(blockers),
        "selected_blocker": _position(selected),
        "control_mode": "blocker" if selected is not None else "cursor_pair",
    }


def _prediction_rejection(
    state: Mapping[str, Any] | None,
    action: Any,
    reason: str,
) -> dict[str, Any]:
    return _json(
        {
            "schema_version": "tufa.mirror_joint.transition.v1",
            "family_id": FAMILY_ID,
            "accepted": False,
            "lease_safe": False,
            "reason": str(reason),
            "action": _normalise_action_name(action),
            "topology_digest": state.get("topology_digest", "") if state else "",
        }
    )


def _predict_step(
    state: Mapping[str, Any],
    step: Mapping[str, Any],
    *,
    remaining_before: int,
) -> dict[str, Any]:
    cursor_a = tuple(state["cursor_a"])
    cursor_b = tuple(state["cursor_b"])
    blockers = frozenset(state.get("blockers") or ())
    selected = state.get("selected")
    successor = _apply_abstract_step(
        cursor_a=cursor_a,
        cursor_b=cursor_b,
        blockers=blockers,
        selected=selected,
        step=step,
        full_floor=frozenset(state["full_floor"]),
        death=frozenset(state.get("death") or ()),
    )
    if successor is None:
        return _prediction_rejection(state, step, "unsafe_or_noop_transition")
    next_a, next_b, next_blockers, next_selected, terminal = successor
    action = str(step["action"])
    if action == "ACTION6":
        transition_kind = (
            "select_blocker" if next_selected is not None else "deselect_blocker"
        )
    elif selected is not None:
        transition_kind = "move_selected_blocker"
    else:
        transition_kind = "move_reflected_cursor_pair"
    if terminal:
        merge_position = (
            (int(next_a[0]) + int(next_b[0])) // 2,
            (int(next_a[1]) + int(next_b[1])) // 2,
        )
    else:
        merge_position = None
    before_projection = _public_projection(
        cursor_a=cursor_a,
        cursor_b=cursor_b,
        blockers=blockers,
        selected=selected,
    )
    after_projection = _public_projection(
        cursor_a=next_a,
        cursor_b=next_b,
        blockers=next_blockers,
        selected=next_selected,
    )
    successor_digest = stable_digest(
        {
            "topology_digest": state["topology_digest"],
            **after_projection,
        }
    )
    return _json(
        {
            "schema_version": "tufa.mirror_joint.transition.v1",
            "family_id": FAMILY_ID,
            "accepted": True,
            "lease_safe": True,
            "reason": "exact_visible_joint_successor",
            "action": _public_action(state, step),
            "transition_kind": transition_kind,
            "topology_digest": str(state["topology_digest"]),
            "before_frame_digest": str(state["frame_digest"]),
            "before": before_projection,
            "expected_after": after_projection,
            "expected_successor_digest": successor_digest,
            "terminal_identity_merge": bool(terminal),
            "merge_position": _position(merge_position),
            "remaining_plan_before": int(remaining_before),
            "expected_remaining_plan_after": max(0, int(remaining_before) - 1),
            "verification_rule": (
                "score closure owns terminal level changes"
                if terminal
                else "exact cursor/blocker projection plus invariant topology"
            ),
        }
    )


def bind_current(
    frame: Any,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the visible lattice, cursor pair, hazards, and blocker mode."""

    del context
    state = _oriented_state(_read_state(frame))
    if state is None:
        return {
            "schema_version": "tufa.mirror_joint.binding.v1",
            "mechanic_id": FAMILY_ID,
            "family_id": FAMILY_ID,
            "status": "unbound",
            "supported": False,
            "reason": "no_certified_visible_cursor_pair",
            "role_bindings": [],
        }
    direct, reachable = _joint_bfs(state, set(_DIRECTIONS))
    cursor_a = tuple(state["cursor_a"])
    cursor_b = tuple(state["cursor_b"])
    blockers = frozenset(state["blockers"])
    selected = state.get("selected")
    grid_w, _grid_h = state["grid_size"]
    roles = [
        {"role": "cursor_a", "grid_position": _position(cursor_a)},
        {"role": "cursor_b", "grid_position": _position(cursor_b)},
        {
            "role": "walkable_lattice",
            "cell_count": int(len(state["full_floor"])),
        },
    ]
    if state["death"]:
        roles.append({"role": "death_cells", "cell_count": len(state["death"])})
    if blockers:
        roles.append({"role": "movable_blockers", "cell_count": len(blockers)})
    return _json(
        {
            "schema_version": "tufa.mirror_joint.binding.v1",
            "mechanic_id": FAMILY_ID,
            "family_id": FAMILY_ID,
            "status": "bound",
            "supported": True,
            "reason": "visible_equal-footprint_pair_on_inferred_lattice",
            "frame_shape": list(state["frame_shape"]),
            "grid_size": list(state["grid_size"]),
            "cell_size": int(state["cell"]),
            "cursor_pair": {
                "positions": _positions((cursor_a, cursor_b)),
                "identity_visually_ambiguous": bool(
                    state["identity_visually_ambiguous"]
                ),
                "initial_reflection_axis": float(grid_w - 1) / 2.0,
            },
            "paired_transform": {
                "vertical": "delta_b=(delta_a.x,delta_a.y)",
                "horizontal": "delta_b=(-delta_a.x,delta_a.y)",
                "matrix_on_cursor_a_delta": [[-1, 0], [0, 1]],
                "independent_collision_clipping": True,
            },
            "walkability": {
                "rule": "cell-dominant floor with independent cursor clipping",
                "visible_cells": _positions(state["full_floor"]),
                "reachable_joint_state_count": int(reachable),
            },
            "death_cells": _positions(state["death"]),
            "movable_blockers": {
                "positions": _positions(blockers),
                "selected": _position(selected),
                "selection_rule": "minority outlined-on-floor visual state",
            },
            "control_mode": "blocker" if selected is not None else "cursor_pair",
            "collision_goal": {
                "relation": "identity_merge",
                "predicate": "same-cell successor or adjacent horizontal identity swap",
            },
            "direct_merge_route_available": bool(direct),
            "blocker_reposition_required": bool(not direct and blockers),
            "topology_digest": str(state["topology_digest"]),
            "role_bindings": roles,
        }
    )


def compile_schema(
    frame: Any,
    level: Any,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a transferable schema; ``level`` remains metadata only."""

    binding = bind_current(frame, context=context)
    return _json(
        {
            "schema_id": "tufa.coupled_mirror_navigation.v1",
            "schema_version": "tufa.mirror_joint.schema.v1",
            "mechanic_id": FAMILY_ID,
            "family_id": FAMILY_ID,
            "level": _safe_metadata(level),
            "supported": bool(binding.get("supported")),
            "roles": [
                "paired_cursor_identities",
                "walkable_lattice",
                "death_cells",
                "outlined_movable_blockers",
            ],
            "required_bindings": [
                "paired_cursor_identities",
                "walkable_lattice",
                "paired_displacement_transform",
                "identity_merge_goal",
            ],
            "state_variables": [
                "cursor_a_position",
                "cursor_b_position",
                "blocker_positions",
                "selected_blocker",
                "static_topology_digest",
            ],
            "operator": {
                "name": "reflected_joint_step",
                "actions": sorted(_ALL_ACTIONS),
                "transition": (
                    "vertical deltas are shared; horizontal deltas are reflected; "
                    "each identity clips independently against visible topology"
                ),
            },
            "goal": {
                "name": "identity_merge",
                "condition": "same successor cell or adjacent horizontal swap",
            },
            "invariants": [
                "normalized walkability and death topology remain unchanged",
                "non-selected blockers and cursor identities remain present",
                "only the leased cursor pair or selected blocker may move",
            ],
            "falsifiers": [
                "a cursor enters a visible death cell",
                "the observed joint displacement differs from the reflected transform",
                "a blocker or static topology cell changes unexpectedly",
                "the frame does not match the exact one-step successor projection",
            ],
            "verification": {
                "lease": "one action only",
                "non_terminal": "exact rebound projection and topology digest",
                "terminal": "score closure; never infer success from a new-level frame",
                "suffix_privacy": "only private action count is exposed",
            },
            "current_binding": binding,
        }
    )


def _blocked_plan(reason: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    return _json(
        {
            "schema_version": "tufa.mirror_joint.plan.v1",
            "family_id": FAMILY_ID,
            "status": "blocked",
            "actions": [],
            "action_count": 0,
            "certified": False,
            "certificate_reason": str(reason),
            "suffix_exposed": False,
            "requires_transition_verification": True,
            "detail": {"topology_digest": binding.get("topology_digest", "")},
        }
    )


def propose_plan(
    frame: Any,
    valid_actions: Iterable[Any] | None = None,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one certified action from a private frame-rebound joint plan."""

    del context
    state = _oriented_state(_read_state(frame))
    binding = bind_current(frame)
    if state is None or not binding.get("supported"):
        return _blocked_plan("visible_mirror_binding_unavailable", binding)
    allowed = _allowed_actions(valid_actions)
    plan, diagnostics = _plan_for_state(state, allowed)
    if not plan:
        return _blocked_plan(
            str(diagnostics.get("blocked_reason") or "no_safe_identity_merge_route"),
            binding,
        )
    first = plan[0]
    prediction = _predict_step(state, first, remaining_before=len(plan))
    if not prediction.get("accepted"):
        return _blocked_plan(str(prediction.get("reason")), binding)
    public_action = _public_action(state, first)
    return _json(
        {
            "schema_version": "tufa.mirror_joint.plan.v1",
            "family_id": FAMILY_ID,
            "status": "proposed",
            "actions": [public_action],
            "action_count": 1,
            "certified": True,
            "certificate_reason": "frame-bound exact coupled successor",
            "transition_rule": str(prediction["transition_kind"]),
            "suffix_exposed": False,
            "requires_transition_verification": True,
            "first_prediction": prediction,
            "progress_measure": {
                "metric": "remaining_private_joint_actions",
                "before": int(len(plan)),
                "expected_after": max(0, int(len(plan)) - 1),
                "strict": True,
            },
            "grounded_observation": {
                "entities": binding.get("role_bindings", []),
                "invariants": [
                    "static normalized topology is unchanged",
                    "the exact cursor/blocker successor is visible",
                ],
                "falsifiers": [
                    "no projected state change",
                    "unexpected death, blocker, or cursor displacement",
                ],
            },
            "detail": {
                "private_plan_length": int(len(plan)),
                "planner": str(diagnostics.get("planner") or "joint_cursor_bfs"),
                "topology_digest": str(state["topology_digest"]),
                "control_mode": binding["control_mode"],
            },
        }
    )


def predict_transition(
    frame: Any,
    action: Any,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict only the currently rebound private-plan prefix."""

    del context
    state = _oriented_state(_read_state(frame))
    if state is None:
        return _prediction_rejection(None, action, "visible_mirror_binding_unavailable")
    internal = _internal_action(state, action)
    if internal is None:
        return _prediction_rejection(state, action, "unbound_action")
    plan, diagnostics = _plan_for_state(state, set(_ALL_ACTIONS))
    if not plan:
        return _prediction_rejection(
            state,
            action,
            str(diagnostics.get("blocked_reason") or "no_safe_identity_merge_route"),
        )
    if internal != plan[0]:
        return _prediction_rejection(state, action, "action_is_not_rebound_plan_prefix")
    return _predict_step(state, plan[0], remaining_before=len(plan))


def verify_transition(expected: dict[str, Any], current_frame: Any) -> bool:
    """Fail closed unless the exact non-terminal abstract successor rebinds."""

    if not isinstance(expected, Mapping):
        return False
    if (
        str(expected.get("family_id", "")) != FAMILY_ID
        or not expected.get("accepted")
        or not expected.get("lease_safe")
        or expected.get("terminal_identity_merge")
    ):
        return False
    state = _oriented_state(_read_state(current_frame))
    if state is None:
        return False
    if str(state["frame_digest"]) == str(expected.get("before_frame_digest", "")):
        return False
    if str(state["topology_digest"]) != str(expected.get("topology_digest", "")):
        return False

    observed = _public_projection(
        cursor_a=tuple(state["cursor_a"]),
        cursor_b=tuple(state["cursor_b"]),
        blockers=state.get("blockers") or (),
        selected=state.get("selected"),
    )
    wanted = expected.get("expected_after")
    if not isinstance(wanted, Mapping) or observed != _json(dict(wanted)):
        return False
    observed_digest = stable_digest(
        {
            "topology_digest": state["topology_digest"],
            **observed,
        }
    )
    if observed_digest != str(expected.get("expected_successor_digest", "")):
        return False
    return True


__all__ = [
    "FAMILY_ID",
    "bind_current",
    "compile_schema",
    "predict_transition",
    "propose_plan",
    "verify_transition",
]
