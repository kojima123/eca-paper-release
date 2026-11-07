"""Visible reflection-orbit coverage with one-step exact verification."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any

import numpy as np

from arcgym.eca.primitive_synthesis import (
    _axis_reflection_cell_size,
    _axis_reflection_coverage_plan,
    _axis_reflection_coverage_state,
    _axis_reflection_horizontal_axis,
    _axis_reflection_proximity_cell_components,
    _axis_reflection_target_anchor,
    _axis_reflection_value_cell_sets,
    _axis_reflection_vertical_axis,
)


FAMILY_ID = "reflection_orbit_coverage"
_ACTIONS = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}
_ALIASES = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
    "SPACE": "ACTION5",
}
_MOVE = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}
_MULTI_SEQUENCE_STATE: dict[str, Any] | None = None


def _frame2d(frame: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(frame)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


def _action_name(action: Any) -> str:
    if isinstance(action, Mapping):
        action = action.get("action")
    if hasattr(action, "name"):
        action = action.name
    name = str(action or "").split(".")[-1].upper()
    return _ALIASES.get(name, name)


def _allowed(valid_actions: Sequence[Any] | set[Any] | None) -> set[str]:
    if not valid_actions:
        return set(_ACTIONS)
    return {
        name
        for item in valid_actions
        if (name := _action_name(item)) in _ACTIONS
    }


def _state(frame: Any) -> dict[str, Any] | None:
    global _MULTI_SEQUENCE_STATE
    arr = _frame2d(frame)
    if arr is None:
        return None
    frame_digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    if (
        isinstance(_MULTI_SEQUENCE_STATE, Mapping)
        and str(_MULTI_SEQUENCE_STATE.get("frame_digest", "")) == frame_digest
        and isinstance(_MULTI_SEQUENCE_STATE.get("state"), Mapping)
    ):
        return dict(_MULTI_SEQUENCE_STATE["state"])
    state = _axis_reflection_coverage_state(arr)
    if isinstance(state, Mapping):
        current = dict(state)
        if bool(current.get("multi_cursor_axis_reflection")) and current.get(
            "precomputed_operation_plan"
        ):
            continued = _continue_multi_cursor_sequence(
                current,
                frame_digest=frame_digest,
            )
            if continued is not None:
                return continued
            recovered_move = _recover_multi_cursor_move_state(
                arr,
                frame_digest=frame_digest,
            )
            if recovered_move is not None:
                return recovered_move
            current["_selection_index"] = -1
            _MULTI_SEQUENCE_STATE = {
                "frame_digest": frame_digest,
                "state": dict(current),
            }
        return current
    recovered_move = _recover_multi_cursor_move_state(
        arr,
        frame_digest=frame_digest,
    )
    if recovered_move is not None:
        return recovered_move
    recovered = _recover_multi_cursor_axis_state(arr, frame_digest=frame_digest)
    if recovered is not None:
        return recovered
    return _movable_axis_state(arr)


def _continue_multi_cursor_sequence(
    parsed: dict[str, Any],
    *,
    frame_digest: str,
) -> dict[str, Any] | None:
    global _MULTI_SEQUENCE_STATE
    cached = _MULTI_SEQUENCE_STATE
    if not isinstance(cached, Mapping) or not isinstance(cached.get("state"), Mapping):
        return None
    prior = dict(cached["state"])
    plan = [str(action) for action in prior.get("precomputed_operation_plan", ())]
    if not plan or not bool(prior.get("multi_cursor_axis_reflection")):
        return None
    if _static_signature(parsed) != _static_signature(prior):
        return None
    action = plan[0]
    raw_selection_index = prior.get("_selection_index", -1)
    selection_index = int(
        raw_selection_index if raw_selection_index is not None else -1
    )
    cursor_plans = [
        dict(item) for item in parsed.get("cursor_plans", ()) if isinstance(item, Mapping)
    ]
    continued = dict(parsed)
    prior_shape = tuple(
        sorted(
            (int(point[0]), int(point[1]))
            for point in prior.get("cursor_offsets", ())
        )
    )
    active: dict[str, Any] | None = None
    if action == "ACTION5":
        selection_index += 1
        candidates = [
            item
            for item in cursor_plans
            if tuple(
                sorted(
                    (int(point[0]), int(point[1]))
                    for point in item.get("cursor_offsets", ())
                )
            )
            == prior_shape
        ] if selection_index == 0 else [
            item
            for item in cursor_plans
            if tuple(
                sorted(
                    (int(point[0]), int(point[1]))
                    for point in item.get("cursor_offsets", ())
                )
            )
            != prior_shape
        ]
        if len(candidates) != 1:
            return None
        active = candidates[0]
    elif selection_index < 0:
        orientation = str(prior.get("axis_orientation", "") or "")
        dx, dy = _MOVE.get(action, (0, 0))
        prior_axis = int(
            prior.get("axis_x", 0) if orientation == "vertical" else prior.get("axis_y", 0)
        )
        parsed_axis = int(
            parsed.get("axis_x", 0) if orientation == "vertical" else parsed.get("axis_y", 0)
        )
        if parsed_axis != prior_axis + (dx if orientation == "vertical" else dy):
            return None
    else:
        if action not in _MOVE:
            return None
        candidates = [
            item
            for item in cursor_plans
            if tuple(
                sorted(
                    (int(point[0]), int(point[1]))
                    for point in item.get("cursor_offsets", ())
                )
            )
            == prior_shape
        ]
        if len(candidates) != 1:
            return None
        active = candidates[0]
        prior_cursor = tuple(int(value) for value in prior.get("cursor_pos", ()))
        current_cursor = tuple(
            int(value) for value in active.get("cursor_pos", ())
        )
        dx, dy = _MOVE[action]
        if current_cursor != (prior_cursor[0] + dx, prior_cursor[1] + dy):
            return None
    if active is not None:
        continued["cursor_pos"] = tuple(active.get("cursor_pos", ()))
        continued["target_anchor"] = tuple(active.get("target_anchor", ()))
        continued["cursor_offsets"] = tuple(active.get("cursor_offsets", ()))
    continued["_selection_index"] = selection_index
    continued["precomputed_operation_plan"] = tuple(plan[1:])
    continued["operation_plan_len"] = len(plan) - 1
    continued["distance_to_anchor"] = len(plan) - 1
    _MULTI_SEQUENCE_STATE = {
        "frame_digest": frame_digest,
        "state": dict(continued),
    }
    return continued


def _recover_multi_cursor_axis_state(
    frame: np.ndarray,
    *,
    frame_digest: str,
) -> dict[str, Any] | None:
    """Bridge only a one-cell, visibly re-detected transient axis overlap."""

    global _MULTI_SEQUENCE_STATE
    cached = _MULTI_SEQUENCE_STATE
    if not isinstance(cached, Mapping):
        return None
    prior = cached.get("state")
    if not isinstance(prior, Mapping):
        return None
    if str(cached.get("frame_digest", "")) == frame_digest:
        return dict(prior)
    plan = [str(action) for action in prior.get("precomputed_operation_plan", ())]
    if not plan or plan[0] not in _MOVE:
        return None
    orientation = str(prior.get("axis_orientation", "") or "")
    if orientation not in {"vertical", "horizontal"}:
        return None
    cell = int(prior.get("cell", 0) or 0)
    if cell <= 0:
        return None
    cell_sets = _axis_reflection_value_cell_sets(frame, cell=cell)
    if not cell_sets:
        return None
    background_value = max(cell_sets, key=lambda value: len(cell_sets[value]))
    values, counts = np.unique(frame, return_counts=True)
    pixel_counts = {
        int(value): int(count) for value, count in zip(values.tolist(), counts.tolist())
    }
    grid_h = int(frame.shape[0] // cell)
    grid_w = int(frame.shape[1] // cell)
    axis = (
        _axis_reflection_vertical_axis(
            cell_sets,
            grid_h=grid_h,
            background_value=int(background_value),
            pixel_counts=pixel_counts,
        )
        if orientation == "vertical"
        else _axis_reflection_horizontal_axis(
            cell_sets,
            grid_w=grid_w,
            background_value=int(background_value),
            pixel_counts=pixel_counts,
        )
    )
    prior_axis = int(
        prior.get("axis_x", 0) if orientation == "vertical" else prior.get("axis_y", 0)
    )
    dx, dy = _MOVE[plan[0]]
    expected_axis = prior_axis + (dx if orientation == "vertical" else dy)
    if axis is None:
        learned_axis_value = prior.get("axis_value")
        learned_cells = cell_sets.get(int(learned_axis_value), frozenset()) if learned_axis_value is not None else frozenset()
        visible_count = sum(
            (int(x) == expected_axis if orientation == "vertical" else int(y) == expected_axis)
            for x, y in learned_cells
        )
        required = max(4, int((grid_h if orientation == "vertical" else grid_w) * 0.25))
        if visible_count < required:
            return None
        axis = (expected_axis, int(learned_axis_value))
    if int(axis[0]) != expected_axis:
        return None
    recovered = dict(prior)
    if orientation == "vertical":
        recovered["axis_x"] = int(axis[0])
    else:
        recovered["axis_y"] = int(axis[0])
    recovered["precomputed_operation_plan"] = tuple(plan[1:])
    recovered["operation_plan_len"] = len(plan) - 1
    recovered["distance_to_anchor"] = len(plan) - 1
    recovered["transient_axis_overlap_rebound"] = True
    _MULTI_SEQUENCE_STATE = {
        "frame_digest": frame_digest,
        "state": dict(recovered),
    }
    return recovered


def _recover_multi_cursor_move_state(
    frame: np.ndarray,
    *,
    frame_digest: str,
) -> dict[str, Any] | None:
    """Rebind an active cursor while its body is partially hidden by an axis."""

    global _MULTI_SEQUENCE_STATE
    cached = _MULTI_SEQUENCE_STATE
    if not isinstance(cached, Mapping) or not isinstance(cached.get("state"), Mapping):
        return None
    prior = dict(cached["state"])
    selection_index = int(prior.get("_selection_index", -1))
    plan = [str(action) for action in prior.get("precomputed_operation_plan", ())]
    if selection_index < 0 or not plan or plan[0] not in _MOVE:
        return None
    cursor = tuple(int(value) for value in prior.get("cursor_pos", ()))
    offsets = tuple(
        (int(point[0]), int(point[1])) for point in prior.get("cursor_offsets", ())
    )
    if len(cursor) != 2 or not offsets:
        return None
    dx, dy = _MOVE[plan[0]]
    expected_cursor = (cursor[0] + dx, cursor[1] + dy)
    cell = int(prior.get("cell", 0) or 0)
    if cell <= 0:
        return None
    cell_sets = _axis_reflection_value_cell_sets(frame, cell=cell)
    cursor_value = prior.get("cursor_value")
    visible_cursor_cells = cell_sets.get(int(cursor_value), frozenset()) if cursor_value is not None else frozenset()
    expected_cells = {
        (expected_cursor[0] + offset_x, expected_cursor[1] + offset_y)
        for offset_x, offset_y in offsets
    }
    if len(expected_cells & set(visible_cursor_cells)) < max(2, len(offsets) // 2):
        return None
    orientation = str(prior.get("axis_orientation", "") or "")
    axis_position = int(
        prior.get("axis_x", 0) if orientation == "vertical" else prior.get("axis_y", 0)
    )
    axis_value = prior.get("axis_value")
    visible_axis_cells = cell_sets.get(int(axis_value), frozenset()) if axis_value is not None else frozenset()
    axis_visible_count = sum(
        (int(x) == axis_position if orientation == "vertical" else int(y) == axis_position)
        for x, y in visible_axis_cells
    )
    grid_w, grid_h = (int(value) for value in prior.get("grid_size", (0, 0)))
    if axis_visible_count < max(4, int((grid_h if orientation == "vertical" else grid_w) * 0.25)):
        return None
    recovered = dict(prior)
    recovered["cursor_pos"] = expected_cursor
    cursor_plans = [
        dict(item) for item in prior.get("cursor_plans", ()) if isinstance(item, Mapping)
    ]
    prior_shape = tuple(sorted(offsets))
    for item in cursor_plans:
        item_shape = tuple(
            sorted(
                (int(point[0]), int(point[1]))
                for point in item.get("cursor_offsets", ())
            )
        )
        if item_shape == prior_shape:
            item["cursor_pos"] = expected_cursor
    recovered["cursor_plans"] = tuple(cursor_plans)
    recovered["precomputed_operation_plan"] = tuple(plan[1:])
    recovered["operation_plan_len"] = len(plan) - 1
    recovered["distance_to_anchor"] = len(plan) - 1
    recovered["transient_cursor_axis_overlap_rebound"] = True
    _MULTI_SEQUENCE_STATE = {
        "frame_digest": frame_digest,
        "state": dict(recovered),
    }
    return recovered


def _delta_plan(
    delta: int,
    *,
    decrement: str,
    increment: str,
) -> list[str]:
    if delta < 0:
        return [decrement] * abs(int(delta))
    if delta > 0:
        return [increment] * int(delta)
    return []


def _move_plan(start: tuple[int, int], target: tuple[int, int]) -> list[str]:
    return _delta_plan(
        int(target[0]) - int(start[0]),
        decrement="ACTION3",
        increment="ACTION4",
    ) + _delta_plan(
        int(target[1]) - int(start[1]),
        decrement="ACTION1",
        increment="ACTION2",
    )


def _reflect_cells(
    cells: Sequence[tuple[int, int]],
    *,
    orientation: str,
    axis_position: int,
) -> frozenset[tuple[int, int]]:
    if orientation == "horizontal":
        return frozenset(
            (int(x), 2 * int(axis_position) - int(y)) for x, y in cells
        )
    return frozenset(
        (2 * int(axis_position) - int(x), int(y)) for x, y in cells
    )


def _compact_shape_sets(
    cell_sets: Mapping[int, frozenset[tuple[int, int]]],
    *,
    axis_orientation: str,
    axis_position: int,
    background_value: int,
) -> list[frozenset[tuple[int, int]]]:
    shapes: list[frozenset[tuple[int, int]]] = []
    for value, cells in cell_sets.items():
        if int(value) == int(background_value) or not (2 <= len(cells) <= 64):
            continue
        if axis_orientation == "vertical" and sum(
            int(x) == int(axis_position) for x, _y in cells
        ) >= max(8, int(len(cells) * 0.8)):
            continue
        if axis_orientation == "horizontal" and sum(
            int(y) == int(axis_position) for _x, y in cells
        ) >= max(8, int(len(cells) * 0.8)):
            continue
        components = _axis_reflection_proximity_cell_components(cells)
        if len(components) != 1:
            continue
        component = components[0]
        if int(component.get("width", 0)) > 8 or int(component.get("height", 0)) > 8:
            continue
        shapes.append(frozenset((int(x), int(y)) for x, y in cells))
    return shapes


def _movable_axis_state(frame: np.ndarray) -> dict[str, Any] | None:
    """Bind a single movable axis before control cycles to the cursor."""

    cell = _axis_reflection_cell_size(frame)
    grid_h = int(frame.shape[0] // cell)
    grid_w = int(frame.shape[1] // cell)
    cell_sets = _axis_reflection_value_cell_sets(frame, cell=cell)
    if len(cell_sets) < 4:
        return None
    background_value = max(cell_sets, key=lambda value: len(cell_sets[value]))
    values, counts = np.unique(frame, return_counts=True)
    pixel_counts = {
        int(value): int(count) for value, count in zip(values.tolist(), counts.tolist())
    }
    vertical = _axis_reflection_vertical_axis(
        cell_sets,
        grid_h=grid_h,
        background_value=int(background_value),
        pixel_counts=pixel_counts,
    )
    horizontal = _axis_reflection_horizontal_axis(
        cell_sets,
        grid_w=grid_w,
        background_value=int(background_value),
        pixel_counts=pixel_counts,
    )
    if (vertical is None) == (horizontal is None):
        return None
    orientation = "vertical" if vertical is not None else "horizontal"
    axis_position = int((vertical or horizontal)[0])
    shapes = _compact_shape_sets(
        cell_sets,
        axis_orientation=orientation,
        axis_position=axis_position,
        background_value=int(background_value),
    )
    grouped: dict[int, list[frozenset[tuple[int, int]]]] = {}
    for shape in shapes:
        grouped.setdefault(len(shape), []).append(shape)

    candidates: list[tuple[tuple[int, ...], dict[str, Any]]] = []
    for shape_group in grouped.values():
        if len(shape_group) != 3:
            continue
        for cursor_cells in shape_group:
            reflected = _reflect_cells(
                tuple(cursor_cells),
                orientation=orientation,
                axis_position=axis_position,
            )
            if reflected not in shape_group or reflected == cursor_cells:
                continue
            targets = next(
                (
                    shape
                    for shape in shape_group
                    if shape != cursor_cells and shape != reflected
                ),
                None,
            )
            if targets is None:
                continue
            if orientation == "vertical":
                cursor_side = sum(x for x, _y in cursor_cells) / len(cursor_cells) - axis_position
                target_side = sum(x for x, _y in targets) / len(targets) - axis_position
            else:
                cursor_side = sum(y for _x, y in cursor_cells) / len(cursor_cells) - axis_position
                target_side = sum(y for _x, y in targets) / len(targets) - axis_position
            if cursor_side == 0 or target_side == 0 or cursor_side * target_side >= 0:
                continue
            cursor_x = min(x for x, _y in cursor_cells)
            cursor_y = min(y for _x, y in cursor_cells)
            offsets = tuple(
                sorted((int(x - cursor_x), int(y - cursor_y)) for x, y in cursor_cells)
            )
            axis_range = range(grid_w) if orientation == "vertical" else range(grid_h)
            for target_axis in axis_range:
                if orientation == "vertical":
                    target_anchor = _axis_reflection_target_anchor(
                        offsets,
                        axis_x=int(target_axis),
                        targets=targets,
                    )
                else:
                    swapped_offsets = tuple((y, x) for x, y in offsets)
                    swapped_targets = frozenset((y, x) for x, y in targets)
                    swapped_anchor = _axis_reflection_target_anchor(
                        swapped_offsets,
                        axis_x=int(target_axis),
                        targets=swapped_targets,
                    )
                    target_anchor = (
                        (int(swapped_anchor[1]), int(swapped_anchor[0]))
                        if swapped_anchor is not None
                        else None
                    )
                if target_anchor is None:
                    continue
                max_x = int(target_anchor[0]) + max(x for x, _y in offsets)
                max_y = int(target_anchor[1]) + max(y for _x, y in offsets)
                if not (
                    0 <= int(target_anchor[0]) <= max_x < grid_w
                    and 0 <= int(target_anchor[1]) <= max_y < grid_h
                ):
                    continue
                axis_delta = int(target_axis) - axis_position
                axis_plan = _delta_plan(
                    axis_delta,
                    decrement="ACTION3" if orientation == "vertical" else "ACTION1",
                    increment="ACTION4" if orientation == "vertical" else "ACTION2",
                )
                cursor_plan = _move_plan(
                    (int(cursor_x), int(cursor_y)),
                    (int(target_anchor[0]), int(target_anchor[1])),
                )
                plan = [*axis_plan, "ACTION5", *cursor_plan]
                rank = (
                    len(plan),
                    abs(axis_delta),
                    abs(int(target_anchor[0]) - cursor_x)
                    + abs(int(target_anchor[1]) - cursor_y),
                    int(target_axis),
                )
                candidates.append(
                    (
                        rank,
                        {
                            "mechanic_state_type": "axis_reflection_coverage",
                            "goal_link_type": "movable_axis_then_reflected_cursor_coverage",
                            "cell": int(cell),
                            "grid_size": (grid_w, grid_h),
                            "axis_x": axis_position if orientation == "vertical" else 0,
                            "axis_y": axis_position if orientation == "horizontal" else 0,
                            "axis_orientation": orientation,
                            "target_axis_position": int(target_axis),
                            "cursor_pos": (int(cursor_x), int(cursor_y)),
                            "target_anchor": (
                                int(target_anchor[0]),
                                int(target_anchor[1]),
                            ),
                            "cursor_offsets": offsets,
                            "target_cells": tuple(sorted(targets)),
                            "target_count": len(targets),
                            "distance_to_anchor": len(plan),
                            "reflected_target_distance": len(plan),
                            "axis_control_active": True,
                            "precomputed_operation_plan": tuple(plan),
                            "expected_goal_delta": 1.0,
                            "reader_confidence": 0.79,
                            "goal_link_confidence": 0.79,
                        },
                    )
                )
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _plan(state: Mapping[str, Any], allowed: set[str]) -> list[str]:
    if "_selection_index" in state and state.get("precomputed_operation_plan"):
        raw = list(state.get("precomputed_operation_plan", ()))
    else:
        raw = _axis_reflection_coverage_plan(state, allowed)
    if not isinstance(raw, Sequence):
        return []
    plan = [str(action) for action in raw]
    if not plan or any(action not in allowed for action in plan):
        return []
    return plan


def _points(value: Any) -> list[list[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    points: list[list[int]] = []
    for point in value:
        if isinstance(point, Sequence) and len(point) >= 2:
            points.append([int(point[0]), int(point[1])])
    return sorted(points)


def _static_signature(state: Mapping[str, Any]) -> dict[str, Any]:
    orientation = str(state.get("axis_orientation", "") or "")
    if not orientation:
        orientation = "horizontal" if state.get("axis_y") is not None else "vertical"
    cursor_plans = [
        item for item in state.get("cursor_plans", ()) if isinstance(item, Mapping)
    ]
    cursor_shapes = (
        sorted(
            (_points(item.get("cursor_offsets", ())) for item in cursor_plans),
            key=lambda points: (len(points), points),
        )
        if cursor_plans
        else [_points(state.get("cursor_offsets", ()))]
    )
    return {
        "cell": int(state.get("cell", 0) or 0),
        "grid_size": [int(value) for value in state.get("grid_size", ())],
        "axis_orientation": orientation,
        "cursor_shapes": cursor_shapes,
        "target_cells": _points(state.get("target_cells", ())),
        "target_count": int(state.get("target_count", 0) or 0),
        "multi_cursor": bool(state.get("multi_cursor_axis_reflection")),
        "multi_axis": bool(state.get("multi_axis_orbit_reflection")),
    }


def bind_current(frame: Any, **_kwargs: Any) -> dict[str, Any]:
    state = _state(frame)
    if state is None:
        return {
            "mechanic_id": FAMILY_ID,
            "family_id": FAMILY_ID,
            "status": "unbound",
            "supported": False,
            "roles": [],
            "role_bindings": [],
        }
    cursor = [int(value) for value in state.get("cursor_pos", ())]
    target = [int(value) for value in state.get("target_anchor", ())]
    axis = {
        "role": "reflection_axis",
        "x": int(state.get("axis_x", 0) or 0),
        "y": int(state.get("axis_y", 0) or 0),
        "orientation": str(state.get("axis_orientation", "vertical") or "vertical"),
    }
    roles = [
        axis,
        {"role": "active_cursor", "position": cursor},
        {"role": "target_anchor", "position": target},
    ]
    return {
        "mechanic_id": FAMILY_ID,
        "family_id": FAMILY_ID,
        "status": "bound",
        "supported": bool(cursor and target),
        "roles": roles,
        "role_bindings": roles,
        "distance_to_anchor": int(state.get("distance_to_anchor", 0) or 0),
        "reflected_target_distance": int(
            state.get("reflected_target_distance", 0) or 0
        ),
        "target_count": int(state.get("target_count", 0) or 0),
        "static_signature": _static_signature(state),
    }


def compile_schema(frame: Any, *, level: int, **_kwargs: Any) -> dict[str, Any]:
    binding = bind_current(frame)
    supported = bool(binding.get("supported"))
    return {
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "level": int(level),
        "supported": supported,
        "roles": list(binding.get("roles") or []),
        "required_bindings": ["reflection_axis", "active_cursor", "target_anchor"],
        "operator": {
            "name": "move_active_reflection_entity",
            "transition": "one visible grid-cell move followed by orbit recomputation",
        },
        "goal": "the reflected cursor orbit exactly covers the visible targets",
        "invariants": [
            "axis and target footprint remain visible",
            "the current-frame reflection plan loses exactly its executed prefix",
        ],
        "falsifiers": [
            "active cursor misses its predicted cell",
            "axis or target footprint changes unexpectedly",
            "fresh reflection plan is not the predicted suffix",
        ],
        "current_binding": binding,
    }


def propose_plan(
    frame: Any,
    *,
    valid_actions: Sequence[Any] | set[Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any] | None:
    state = _state(frame)
    if state is None:
        return None
    allowed = _allowed(valid_actions)
    plan = _plan(state, allowed)
    if not plan:
        return None
    prediction = predict_transition(frame, {"action": plan[0]})
    if not prediction.get("accepted"):
        return None
    return {
        "mechanic_id": FAMILY_ID,
        "actions": [{"action": plan[0]}],
        "certified": True,
        "certificate_reason": "visible reflection state and exact one-step suffix",
        "transition_rule": "move one active entity cell and reparse the orbit residual",
        "first_prediction": prediction,
        "grounded_observation": {
            "invariants": [
                "reflection axis is unchanged",
                "target footprint is unchanged",
            ],
            "falsifiers": ["fresh plan differs from the predicted suffix"],
        },
        "detail": {
            "private_plan_length": len(plan),
            "remaining_reflection_actions": len(plan),
            "target_count": int(state.get("target_count", 0) or 0),
        },
    }


def predict_transition(frame: Any, action: Any, **_kwargs: Any) -> dict[str, Any]:
    state = _state(frame)
    action_name = _action_name(action)
    if state is None or action_name not in _ACTIONS:
        return {"family_id": FAMILY_ID, "accepted": False, "reason": "unbound"}
    plan = _plan(state, set(_ACTIONS))
    cursor = state.get("cursor_pos")
    if not plan or plan[0] != action_name or not isinstance(cursor, Sequence):
        return {
            "family_id": FAMILY_ID,
            "accepted": False,
            "reason": "action_is_not_current_reflection_prefix",
        }
    axis_x = int(state.get("axis_x", 0) or 0)
    axis_y = int(state.get("axis_y", 0) or 0)
    orientation = str(state.get("axis_orientation", "") or "")
    current_axis_position = axis_x if orientation == "vertical" else axis_y
    target_axis_position = int(
        state.get("target_axis_position", current_axis_position)
        or current_axis_position
    )
    axis_control = bool(
        state.get("axis_control_active")
        or current_axis_position != target_axis_position
    )
    operation = "cycle_control" if action_name == "ACTION5" else "move_cursor"
    expected_cursor: list[int] | None = [int(cursor[0]), int(cursor[1])]
    expected_axis = [axis_x, axis_y]
    if axis_control and action_name != "ACTION5":
        operation = "move_axis"
        dx, dy = _MOVE[action_name]
        if orientation == "vertical":
            expected_axis[0] += dx
        elif orientation == "horizontal":
            expected_axis[1] += dy
        else:
            return {"family_id": FAMILY_ID, "accepted": False, "reason": "unbound_axis"}
    elif action_name == "ACTION5":
        expected_cursor = None
    elif action_name in _MOVE:
        dx, dy = _MOVE[action_name]
        expected_cursor[0] += dx
        expected_cursor[1] += dy
    return {
        "family_id": FAMILY_ID,
        "accepted": True,
        "reason": "exact_reflection_prefix",
        "action": action_name,
        "prior_cursor": [int(cursor[0]), int(cursor[1])],
        "expected_cursor": expected_cursor,
        "expected_axis": expected_axis,
        "operation": operation,
        "static_signature": _static_signature(state),
        "prior_plan_length": len(plan),
        "expected_remaining_plan": plan[1:],
    }


def verify_transition(expected: Any, current_frame: Any) -> bool:
    if not isinstance(expected, Mapping) or not bool(expected.get("accepted")):
        return False
    state = _state(current_frame)
    if state is None:
        return False
    if _static_signature(state) != expected.get("static_signature"):
        return False
    wanted_cursor = expected.get("expected_cursor")
    if wanted_cursor is not None:
        cursor = [int(value) for value in state.get("cursor_pos", ())]
        if cursor != list(wanted_cursor):
            return False
    current_axis = [
        int(state.get("axis_x", 0) or 0),
        int(state.get("axis_y", 0) or 0),
    ]
    if current_axis != list(expected.get("expected_axis") or []):
        return False
    fresh_plan = _plan(state, set(_ACTIONS))
    remaining = [str(action) for action in expected.get("expected_remaining_plan", [])]
    if fresh_plan != remaining:
        return False
    return int(expected.get("prior_plan_length", 0) or 0) == len(fresh_plan) + 1


__all__ = [
    "FAMILY_ID",
    "bind_current",
    "compile_schema",
    "predict_transition",
    "propose_plan",
    "verify_transition",
]
