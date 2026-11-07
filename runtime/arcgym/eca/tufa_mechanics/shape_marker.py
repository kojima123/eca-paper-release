"""Visible shape-marker coverage with exact one-step runtime verification.

The mechanic treats palette values as ephemeral equivalence classes.  Every
coordinate, mask, marker class, selector, recolor zone, and wall is rebound
from the current level's visible frames.  The imported synthesis helpers are
used only as generic perception and search oracles; coordinate-guarded route
branches are deliberately not imported or accepted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import permutations
import math
from typing import Any

import numpy as np

from arcgym.eca.primitive_synthesis import (
    _shape_coverage_marker_rings,
    _shape_coverage_mask_components,
    _shape_coverage_paste_plan,
    _shape_coverage_paste_state,
    _shape_coverage_selector_values,
    _shape_coverage_target_anchor,
    _shape_recolor_best_waypoint,
    _shape_recolor_candidate_shapes,
    _shape_recolor_nonbackground_components,
    _shape_recolor_palette_pads,
    _shape_recolor_task_plan,
    _shape_recolor_wall_morph_coverage_state,
    _shape_wall_morph_obstacles_from_frame,
)
from arcgym.eca.tufa_mechanics.base import (
    MechanicContext,
    as_frame,
    stable_data,
    stable_digest,
)


FAMILY_ID = "shape_marker_coverage"
_SCHEMA_VERSION = "tufa_shape_marker_coverage.v1"
_ACTIONS = frozenset({"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"})
_ALIASES = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
    "SPACE": "ACTION5",
}
_MOVE_SIGN = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}
_PASTE_GOAL = "same_color_shape_pixels_cover_target_markers"
_RECOLOR_GOAL = "palette_recolored_shape_pixels_cover_target_markers"
_WALL_GOAL = "wall_morphed_shape_pixels_cover_target_markers"


def compile_schema(
    frame: Any,
    *,
    level: Any,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a coordinate-free schema while rebinding the current roles."""

    binding = bind_current(frame, context=context)
    return {
        "schema_version": _SCHEMA_VERSION,
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "level": _json_scalar(level),
        "supported": bool(binding.get("supported")),
        "accepted": bool(binding.get("accepted")),
        "roles": [
            "movable_shape",
            "active_selector",
            "same_class_marker_map",
            "wall",
            "morph_or_recolor_zone",
        ],
        "required_bindings": [
            "movable_shape",
            "active_selector",
            "same_class_marker_map",
        ],
        "state_variables": [
            "shape_masks",
            "active_shape",
            "marker_deficit",
            "wall_footprints",
            "zone_footprints",
        ],
        "operator": {
            "name": "move_or_select_visible_shape",
            "transition": (
                "one inferred lattice move or one selector cycle; a visible "
                "zone collision may recolor or morph the active mask"
            ),
        },
        "goal_relation": (
            "every visible marker is covered by a movable mask of its "
            "frame-inferred equivalence class"
        ),
        "invariants": [
            "pixels outside the old and predicted movable footprints do not change",
            "the active selector occupies the predicted shape-relative site",
            "walls and zones are inferred from the visible frame, never from identity",
        ],
        "falsifiers": [
            "a predicted mask pixel is absent or has the wrong inferred class",
            "the selected shape differs from the predicted cycle successor",
            "an unrelated marker, wall, zone, or background pixel changes",
            "the observed marker deficit differs from the exact mask calculation",
        ],
        "current_binding": binding,
    }


def bind_current(
    frame: Any,
    *,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve(frame, context=context)
    if not resolved.get("accepted"):
        return _unbound(str(resolved.get("reason", "shape_marker_roles_not_bound")))
    return _public_binding(resolved["model"], resolved)


def propose_plan(
    frame: Any,
    *,
    valid_actions: Sequence[Any] | set[Any] | None = None,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Propose exactly one action from a freshly validated residual."""

    resolved = _resolve(frame, context=context)
    if not resolved.get("accepted"):
        return _blocked(str(resolved.get("reason", "shape_marker_roles_not_bound")))
    plan = list(resolved.get("remaining_plan") or [])
    if not plan:
        return _blocked("no_current_frame_shape_marker_residual")
    allowed = _allowed_actions(valid_actions)
    if plan[0] not in allowed:
        return _blocked("required_action_not_available")
    prediction = _predict_resolved(resolved, plan[0])
    if not prediction.get("accepted"):
        return _blocked(str(prediction.get("reason", "transition_not_predictable")))
    deficit = dict(prediction["marker_deficit"])
    return {
        "schema_version": f"{_SCHEMA_VERSION}.plan",
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "status": "proposed",
        "actions": [{"action": plan[0]}],
        "action_count": 1,
        "certified": True,
        "certificate_reason": (
            "visible roles rebound and exact one-step mask transition verified"
        ),
        "transition_rule": str(prediction["transition_kind"]),
        "progress_measure": {
            "metric": "shape_marker_deficit",
            "before": int(deficit["before"]),
            "expected_after": int(deficit["after"]),
            "strict": False,
            "residual_before": len(plan),
            "residual_expected_after": len(plan) - 1,
            "marker_deficit_before": int(deficit["before"]),
            "marker_deficit_expected_after": int(deficit["after"]),
        },
        "requires_transition_verification": True,
        "first_prediction": prediction,
        "grounded_observation": {
            "invariants": [
                "unrelated pixels remain exact",
                "the selector and active mask match the predicted successor",
            ],
            "falsifiers": [
                "mask, selection, zone effect, or marker deficit mismatch",
            ],
        },
        "detail": {
            "private_plan_length": len(plan),
            "suffix_exposed": False,
            "oracle_kind": str(resolved["model"]["mode"]),
            "history_rebound": bool(resolved.get("history_rebound")),
            "marker_class_count": len(resolved["model"]["marker_classes"]),
            "wall_count": len(resolved["model"]["walls"]),
            "zone_count": len(resolved["model"]["zones"]),
        },
    }


def predict_transition(
    frame: Any,
    action: Any,
    *,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve(frame, context=context)
    if not resolved.get("accepted"):
        return _prediction_rejection(
            _action_name(action),
            str(resolved.get("reason", "shape_marker_roles_not_bound")),
        )
    return _predict_resolved(resolved, _action_name(action))


def verify_transition(expected: Any, current_frame: Any) -> dict[str, Any]:
    """Fail closed unless the visible successor matches the exact role model."""

    if not isinstance(expected, Mapping) or not bool(expected.get("accepted")):
        return _verification(False, ["prediction_not_accepted"])
    if str(expected.get("family_id", "")) != FAMILY_ID:
        return _verification(False, ["family_mismatch"])
    arr = as_frame(current_frame)
    if arr is None:
        return _verification(False, ["invalid_current_frame"])
    before_raw = expected.get("before_frame")
    before_model = expected.get("before_runtime_model")
    after_model = expected.get("expected_runtime_model")
    if (
        before_raw is None
        or not isinstance(before_model, Mapping)
        or not isinstance(after_model, Mapping)
    ):
        return _verification(False, ["missing_exact_runtime_payload"])
    try:
        before = np.asarray(before_raw)
    except Exception:
        return _verification(False, ["invalid_before_frame"])
    reasons, evidence = _match_successor(
        before,
        dict(before_model),
        arr,
        dict(after_model),
    )
    recomputed_deficit = _marker_deficit(dict(after_model))
    deficit = expected.get("marker_deficit")
    if not isinstance(deficit, Mapping) or recomputed_deficit != int(
        deficit.get("after", -1)
    ):
        reasons.append("marker_deficit_mismatch")
    evidence["recomputed_marker_deficit"] = int(recomputed_deficit)
    evidence["transition_kind"] = str(expected.get("transition_kind", ""))
    evidence["visual_exact"] = not reasons
    return _verification(not reasons, reasons, evidence)


def _resolve(
    frame: Any,
    *,
    context: MechanicContext | Mapping[str, Any] | None,
) -> dict[str, Any]:
    arr = as_frame(frame)
    if arr is None:
        return {"accepted": False, "reason": "invalid_frame"}
    records, current_index = _history_records(arr, context)
    candidate_indices = _candidate_anchor_indices(records, current_index)
    candidates: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for index in candidate_indices:
        anchor = records[index]["frame"]
        packet = _oracle_packet(anchor)
        if packet is None:
            continue
        executed = [
            str(records[row]["action"]) for row in range(index + 1, current_index + 1)
        ]
        if any(not action for action in executed):
            continue
        plan = list(packet["plan"])
        if len(executed) >= len(plan) or executed != plan[: len(executed)]:
            continue
        model = _build_model(anchor, packet)
        if model is None:
            continue
        initial_model = deepcopy(model)
        valid = True
        for action in executed:
            simulated = _simulate_one(model, action)
            if simulated is None:
                valid = False
                break
            model, _transition = simulated
        if not valid:
            continue
        reasons, _evidence = _match_successor(
            anchor,
            initial_model,
            arr,
            model,
        )
        if reasons:
            continue
        remaining = plan[len(executed) :]
        result = {
            "accepted": True,
            "frame": arr,
            "anchor_frame": anchor,
            "anchor_model": initial_model,
            "model": model,
            "remaining_plan": remaining,
            "history_rebound": bool(executed),
            "executed_prefix_length": len(executed),
        }
        rank = (
            len(executed),
            int(model.get("initial_marker_count", 0)),
            -index,
        )
        candidates.append((rank, result))
    if not candidates:
        packet = _oracle_packet(arr)
        model = _build_model(arr, packet) if packet is not None else None
        if packet is None or model is None:
            return {"accepted": False, "reason": "shape_marker_oracle_rejected"}
        reasons, _evidence = _match_successor(arr, model, arr, model)
        if reasons:
            return {"accepted": False, "reason": "current_binding_not_exact"}
        return {
            "accepted": True,
            "frame": arr,
            "anchor_frame": arr,
            "anchor_model": deepcopy(model),
            "model": model,
            "remaining_plan": list(packet["plan"]),
            "history_rebound": False,
            "executed_prefix_length": 0,
        }
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _history_records(
    current: np.ndarray,
    context: MechanicContext | Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], int]:
    if isinstance(context, MechanicContext):
        rows = context.history
        level = int(context.level)
        previous_action = _action_name(context.previous_action)
    elif isinstance(context, Mapping):
        rows = tuple(
            row for row in context.get("history", ()) if isinstance(row, Mapping)
        )
        level = int(context.get("level", 0) or 0)
        previous_action = _action_name(context.get("previous_action"))
    else:
        rows = ()
        level = 0
        previous_action = ""
    records: list[dict[str, Any]] = []
    for row in rows[-64:]:
        row_level = int(row.get("level", level) or level)
        if level and row_level and row_level != level:
            continue
        row_frame = as_frame(row.get("frame"))
        if row_frame is None or row_frame.shape != current.shape:
            continue
        records.append(
            {
                "frame": row_frame,
                "action": _action_name(row.get("action")),
            }
        )
    current_index = next(
        (
            index
            for index in range(len(records) - 1, -1, -1)
            if np.array_equal(records[index]["frame"], current)
        ),
        -1,
    )
    if current_index < 0:
        records.append({"frame": current, "action": previous_action})
        current_index = len(records) - 1
    else:
        records = records[: current_index + 1]
        current_index = len(records) - 1
    if not records:
        records = [{"frame": current, "action": ""}]
        current_index = 0
    return records, current_index


def _candidate_anchor_indices(
    records: Sequence[Mapping[str, Any]], current_index: int
) -> list[int]:
    indices = {0, current_index}
    if current_index > 0:
        indices.add(current_index - 1)
    for index in range(1, current_index + 1):
        if str(records[index].get("action", "")) == "ACTION5":
            indices.add(index)
    return sorted(index for index in indices if 0 <= index <= current_index)


def _oracle_packet(frame: np.ndarray) -> dict[str, Any] | None:
    paste = _shape_coverage_paste_state(frame)
    if (
        isinstance(paste, Mapping)
        and str(paste.get("goal_link_type", "")) == _PASTE_GOAL
    ):
        plan = _shape_coverage_paste_plan(dict(paste), set(_ACTIONS))
        normalized = _normalized_plan(plan)
        if normalized and _state_has_one_active_shape(paste):
            return {"mode": "rigid", "state": dict(paste), "plan": normalized}
    recolor = _generic_recolor_state(frame)
    if recolor is None:
        return None
    plan = _normalized_plan(recolor.get("precomputed_operation_plan"))
    goal = str(recolor.get("goal_link_type", ""))
    if not plan or goal not in {_RECOLOR_GOAL, _WALL_GOAL}:
        return None
    mode = "wall_morph" if goal == _WALL_GOAL else "zone_recolor"
    return {"mode": mode, "state": recolor, "plan": plan}


def _generic_recolor_state(frame: np.ndarray) -> dict[str, Any] | None:
    """Compose only the relational recolor and BFS wall-morph oracles."""

    arr = np.asarray(frame)
    if arr.ndim != 2 or arr.size == 0 or min(arr.shape) < 24:
        return None
    counts = Counter(int(value) for value in arr.flatten().tolist())
    background = int(counts.most_common(1)[0][0])
    markers = _shape_coverage_marker_rings(arr, background_value=background)
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    ring_values: set[int] = set()
    for marker in markers:
        groups[int(marker["color"])].append((int(marker["x"]), int(marker["y"])))
        ring_values.add(int(marker["ring_value"]))
    if len(groups) < 2:
        return None
    marker_values = set(groups)
    selector_values = _shape_coverage_selector_values(
        counts,
        background_value=background,
        marker_colors=marker_values,
        ring_values=ring_values,
    )
    wall_state = _shape_recolor_wall_morph_coverage_state(
        arr,
        background_value=background,
        marker_groups=groups,
        ring_values=ring_values,
        marker_colors=marker_values,
        selector_values=selector_values,
    )
    if (
        isinstance(wall_state, Mapping)
        and str(wall_state.get("goal_link_type", "")) == _WALL_GOAL
    ):
        return dict(wall_state)
    if len(groups) != 2:
        return None
    pads = _shape_recolor_palette_pads(
        arr,
        marker_colors=marker_values,
        background_value=background,
    )
    if not marker_values.issubset(pads):
        return None
    shapes = _shape_recolor_candidate_shapes(
        arr,
        background_value=background,
        marker_colors=marker_values,
        ring_values=ring_values,
        selector_values=selector_values,
    )
    active = [shape for shape in shapes if bool(shape.get("active"))]
    inactive = [shape for shape in shapes if not bool(shape.get("active"))]
    if len(shapes) != 2 or len(active) != 1 or len(inactive) != 1:
        return None
    options: dict[tuple[int, int], dict[str, Any]] = {}
    for shape in shapes:
        offsets = tuple(
            sorted((int(x), int(y)) for x, y in shape.get("coverage_offsets", ()))
        )
        position = tuple(int(value) for value in shape.get("pos", ()))
        if not offsets or len(position) != 2:
            continue
        for target_value, points in groups.items():
            if int(shape.get("color", -1)) == int(target_value):
                continue
            marker_points = tuple(sorted((int(x), int(y)) for x, y in points))
            anchor = _shape_coverage_target_anchor(
                offsets,
                markers=marker_points,
                current_pos=position,
                stride=3,
            )
            if anchor is None:
                continue
            waypoint = _shape_recolor_best_waypoint(
                shape,
                pads[int(target_value)],
                anchor=anchor,
                frame_shape=arr.shape,
            )
            if waypoint is None:
                continue
            task_plan = _shape_recolor_task_plan(position, waypoint, anchor)
            if not task_plan:
                continue
            shape_index = int(shape["shape_index"])
            options[(shape_index, int(target_value))] = {
                "shape_index": shape_index,
                "source_color": int(shape["color"]),
                "target_color": int(target_value),
                "marker_count": len(marker_points),
                "current_pos": position,
                "target_anchor": (int(anchor[0]), int(anchor[1])),
                "recolor_waypoint": (int(waypoint[0]), int(waypoint[1])),
                "palette_pad_bbox": tuple(
                    int(value) for value in pads[int(target_value)]["bbox"]
                ),
                "operation_plan": tuple(task_plan),
            }
    best: (
        tuple[tuple[int, tuple[tuple[int, int], ...]], list[dict[str, Any]], list[str]]
        | None
    ) = None
    active_index = int(active[0]["shape_index"])
    inactive_index = int(inactive[0]["shape_index"])
    for first_value, second_value in permutations(sorted(groups), 2):
        first = options.get((active_index, int(first_value)))
        second = options.get((inactive_index, int(second_value)))
        if first is None or second is None:
            continue
        plan = [*first["operation_plan"], "ACTION5", *second["operation_plan"]]
        if len(plan) > 96:
            continue
        rank = (
            len(plan),
            tuple(
                sorted(
                    (
                        len(groups[int(value)]),
                        min(groups[int(value)]),
                    )
                    for value in (first_value, second_value)
                )
            ),
        )
        if best is None or rank < best[0]:
            best = (rank, [dict(first), dict(second)], list(plan))
    if best is None:
        return None
    _rank, tasks, plan = best
    return {
        "mechanic_state_type": "shape_recolor_coverage",
        "goal_link_type": _RECOLOR_GOAL,
        "background_value": background,
        "marker_count": sum(len(points) for points in groups.values()),
        "marker_color_count": len(groups),
        "piece_count": len(shapes),
        "palette_pad_count": len(pads),
        "tasks": tasks,
        "pieces": shapes,
        "active_piece": dict(tasks[0]),
        "precomputed_operation_plan": tuple(plan),
        "operation_plan_len": len(plan),
    }


def _build_model(
    frame: np.ndarray,
    packet: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(packet, Mapping):
        return None
    state = packet.get("state")
    if not isinstance(state, Mapping):
        return None
    background = int(
        state.get("background_value", Counter(frame.flat).most_common(1)[0][0])
    )
    marker_rows = _shape_coverage_marker_rings(frame, background_value=background)
    if len(marker_rows) < 2:
        return None
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    ring_values: set[int] = set()
    for marker in marker_rows:
        grouped[int(marker["color"])].append((int(marker["x"]), int(marker["y"])))
        ring_values.add(int(marker["ring_value"]))
    marker_classes: list[dict[str, Any]] = []
    for index, (value, points) in enumerate(
        sorted(grouped.items(), key=lambda item: (tuple(sorted(item[1])), len(item[1])))
    ):
        marker_classes.append(
            {
                "class_id": f"marker_class_{index}",
                "value": int(value),
                "positions": [list(point) for point in sorted(points)],
            }
        )
    class_by_value = {int(row["value"]): str(row["class_id"]) for row in marker_classes}
    counts = Counter(int(value) for value in frame.flatten().tolist())
    selectors = _shape_coverage_selector_values(
        counts,
        background_value=background,
        marker_colors=set(grouped),
        ring_values=ring_values,
    )
    mode = str(packet.get("mode", ""))
    pieces = _model_pieces(
        frame,
        state,
        mode=mode,
        selector_values=selectors,
        class_by_value=class_by_value,
    )
    if not pieces or sum(bool(piece["active"]) for piece in pieces) != 1:
        return None
    if bool(state.get("same_color_multi_piece_partition")):
        pieces = _decompose_same_class_overlaps(pieces)
    active_piece = next(piece for piece in pieces if bool(piece["active"]))
    selector_value = active_piece.get("selector_value")
    if selector_value is None:
        return None
    for piece in pieces:
        piece["selector_value"] = int(selector_value)
    pieces.sort(key=lambda piece: tuple(piece["cycle_key"]))
    for index, piece in enumerate(pieces):
        piece["piece_id"] = f"shape_{index}"
        piece["cycle_rank"] = index
    zones = _model_zones(
        frame,
        state,
        class_by_value,
        background=background,
    )
    walls = _model_walls(
        frame,
        state,
        mode=mode,
        background=background,
        marker_values=set(grouped),
        ring_values=ring_values,
        selector_values=selectors,
    )
    stride = _infer_stride(state, marker_rows)
    if stride <= 0:
        return None
    model: dict[str, Any] = {
        "schema_version": f"{_SCHEMA_VERSION}.runtime_model",
        "mode": mode,
        "frame_shape": [int(frame.shape[0]), int(frame.shape[1])],
        "background_value": background,
        "selector_value": int(selector_value),
        "stride": int(stride),
        "marker_classes": marker_classes,
        "pieces": pieces,
        "zones": zones,
        "walls": walls,
        "volatile_status_points": [
            list(point) for point in sorted(_volatile_status_points(frame, background))
        ],
        "initial_marker_count": sum(len(row["positions"]) for row in marker_classes),
    }
    model["marker_deficit"] = _marker_deficit(model)
    return model


def _model_pieces(
    frame: np.ndarray,
    state: Mapping[str, Any],
    *,
    mode: str,
    selector_values: set[int],
    class_by_value: Mapping[int, str],
) -> list[dict[str, Any]]:
    raw_pieces = [
        dict(piece) for piece in state.get("pieces", ()) if isinstance(piece, Mapping)
    ]
    tasks = {
        int(task.get("shape_index", task.get("piece_index", -1))): dict(task)
        for task in state.get("tasks", ())
        if isinstance(task, Mapping)
    }
    selector_positions = [
        (int(x), int(y), int(value))
        for value in selector_values
        for y, x in np.argwhere(frame == int(value)).tolist()
    ]
    pieces: list[dict[str, Any]] = []
    for raw_index, raw in enumerate(raw_pieces):
        position = tuple(int(value) for value in raw.get("pos", ()))
        if len(position) != 2:
            continue
        width = int(raw.get("width", 0) or 0)
        height = int(raw.get("height", 0) or 0)
        value = int(raw.get("color", -1))
        active = bool(raw.get("active"))
        if width <= 0 or height <= 0:
            continue
        if mode == "rigid":
            offsets = _paste_piece_offsets(
                frame,
                value=value,
                position=position,
                width=width,
                height=height,
                active=active,
                selector_values=selector_values,
            )
        else:
            offsets = {(int(x), int(y)) for x, y in raw.get("coverage_offsets", ())}
            if not offsets:
                offsets = _paste_piece_offsets(
                    frame,
                    value=value,
                    position=position,
                    width=width,
                    height=height,
                    active=active,
                    selector_values=selector_values,
                )
        if not offsets and active:
            offsets = {
                (x, y)
                for y in range(height)
                for x in range(width)
                if 0 <= position[1] + y < frame.shape[0]
                and 0 <= position[0] + x < frame.shape[1]
                and int(frame[position[1] + y, position[0] + x]) == value
            }
        if not offsets:
            continue
        selector = _selector_for_piece(
            selector_positions,
            position=position,
            width=width,
            height=height,
            active=active,
        )
        selector_offset = (
            (int(selector[0]) - position[0], int(selector[1]) - position[1])
            if selector is not None
            else _infer_selector_site(offsets, width=width, height=height)
        )
        selector_value = int(selector[2]) if selector is not None else None
        if active and selector is None:
            continue
        inferred_width = max(width, 2 * selector_offset[0] + 1)
        inferred_height = max(height, 2 * selector_offset[1] + 1)
        if inferred_width % 2 == 0:
            inferred_width += 1
        if inferred_height % 2 == 0:
            inferred_height += 1
        width = inferred_width
        height = inferred_height
        oracle_index = int(raw.get("shape_index", raw.get("piece_index", raw_index)))
        task = tasks.get(oracle_index, {})
        target_value = int(task.get("target_color", value))
        target_class = class_by_value.get(target_value)
        if target_class is None and mode == "rigid":
            target_class = class_by_value.get(value)
        if target_class is None:
            continue
        pieces.append(
            {
                "piece_id": "",
                "oracle_index": oracle_index,
                "value": value,
                "target_value": target_value,
                "target_class": str(target_class),
                "pos": [position[0], position[1]],
                "width": width,
                "height": height,
                "color_offsets": [list(point) for point in sorted(offsets)],
                "selector_offset": [selector_offset[0], selector_offset[1]],
                "selector_value": selector_value,
                "active": active,
                "kind": str(raw.get("kind", "rigid_mask") or "rigid_mask"),
                "cycle_key": [
                    position[1] + selector_offset[1],
                    position[0] + selector_offset[0],
                ],
            }
        )
    return pieces


def _decompose_same_class_overlaps(
    pieces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    active = next(
        (piece for piece in pieces if bool(piece.get("active"))),
        None,
    )
    if active is None:
        return pieces
    active_points = _piece_world_color_points(active)
    out: list[dict[str, Any]] = []
    for piece in pieces:
        row = deepcopy(piece)
        if row is active or bool(row.get("active")):
            out.append(row)
            continue
        x0, y0 = (int(value) for value in row["pos"])
        offsets = {
            (int(point[0]), int(point[1]))
            for point in row.get("color_offsets", ())
            if (x0 + int(point[0]), y0 + int(point[1])) not in active_points
        }
        selector = _infer_selector_site(
            offsets,
            width=int(row["width"]),
            height=int(row["height"]),
        )
        row["selector_offset"] = [selector[0], selector[1]]
        row["width"] = max(int(row["width"]), 2 * selector[0] + 1)
        row["height"] = max(int(row["height"]), 2 * selector[1] + 1)
        row["cycle_key"] = [y0 + selector[1], x0 + selector[0]]
        row["color_offsets"] = [list(point) for point in sorted(offsets)]
        out.append(row)
    return out


def _paste_piece_offsets(
    frame: np.ndarray,
    *,
    value: int,
    position: tuple[int, int],
    width: int,
    height: int,
    active: bool,
    selector_values: set[int],
) -> set[tuple[int, int]]:
    allowed = frame == int(value)
    used_selectors = selector_values if active else set()
    if used_selectors:
        selector_mask = np.zeros(frame.shape, dtype=bool)
        for selector in used_selectors:
            selector_mask |= frame == int(selector)
        allowed |= selector_mask
    components = _shape_coverage_mask_components(
        frame,
        allowed,
        color=int(value),
        selector_values=used_selectors,
    )
    matches = [
        comp
        for comp in components
        if int(comp.get("min_x", -1)) == position[0]
        and int(comp.get("min_y", -1)) == position[1]
        and int(comp.get("width", 0)) == width
        and int(comp.get("height", 0)) == height
    ]
    if len(matches) != 1:
        return set()
    return {(int(x), int(y)) for x, y in matches[0].get("color_offsets", ())}


def _selector_for_piece(
    selector_positions: Sequence[tuple[int, int, int]],
    *,
    position: tuple[int, int],
    width: int,
    height: int,
    active: bool,
) -> tuple[int, int, int] | None:
    if not active:
        return None
    inside = [
        point
        for point in selector_positions
        if position[0] <= point[0] < position[0] + width
        and position[1] <= point[1] < position[1] + height
    ]
    if not inside:
        return None
    center = (position[0] + width // 2, position[1] + height // 2)
    inside.sort(
        key=lambda point: (
            abs(point[0] - center[0]) + abs(point[1] - center[1]),
            point[1],
            point[0],
        )
    )
    return inside[0]


def _infer_selector_site(
    offsets: set[tuple[int, int]], *, width: int, height: int
) -> tuple[int, int]:
    if not offsets:
        return width // 2, height // 2

    def best_axis(axis: int, extent: int) -> int:
        maximum = max(point[axis] for point in offsets)
        center = maximum / 2.0
        candidates: list[tuple[int, float, int]] = []
        for candidate in range(max(extent, maximum + 3)):
            hits = 0
            for point in offsets:
                reflected = list(point)
                reflected[axis] = 2 * candidate - point[axis]
                if tuple(reflected) in offsets:
                    hits += 1
            candidates.append((hits, -abs(candidate - center), -candidate))
        best = max(candidates)
        if best[0] * 3 < len(offsets) * 2:
            return extent // 2
        return -best[2]

    return best_axis(0, width), best_axis(1, height)


def _model_zones(
    frame: np.ndarray,
    state: Mapping[str, Any],
    class_by_value: Mapping[int, str],
    *,
    background: int,
) -> list[dict[str, Any]]:
    known_bboxes = {
        tuple(int(value) for value in task.get("palette_pad_bbox", ()))
        for task in state.get("tasks", ())
        if isinstance(task, Mapping)
        and len(tuple(task.get("palette_pad_bbox", ()))) == 4
    }
    components = [
        comp
        for comp in _shape_recolor_nonbackground_components(
            frame,
            background_value=background,
        )
        if 5 <= int(comp.get("width", 0) or 0) <= 8
        and 5 <= int(comp.get("height", 0) or 0) <= 8
    ]
    prototype_signatures = {
        (
            int(comp["width"]),
            int(comp["height"]),
            int(comp["area"]),
        )
        for comp in components
        if (
            int(comp["min_x"]),
            int(comp["min_y"]),
            int(comp["max_x"]),
            int(comp["max_y"]),
        )
        in known_bboxes
    }
    candidates = [
        comp
        for comp in components
        if (
            int(comp["width"]),
            int(comp["height"]),
            int(comp["area"]),
        )
        in prototype_signatures
    ]
    common_values: set[int] = set()
    if candidates:
        common_values = set(int(value) for value in candidates[0].get("values", ()))
        for comp in candidates[1:]:
            common_values &= set(int(value) for value in comp.get("values", ()))
    inferred: list[tuple[tuple[int, ...], int]] = []
    for comp in candidates:
        counts = Counter(int(value) for value in comp.get("values", ()))
        payloads = [
            (count, value)
            for value, count in counts.items()
            if value not in common_values
        ]
        if not payloads:
            continue
        _count, target = max(payloads)
        inferred.append(
            (
                (
                    int(comp["min_x"]),
                    int(comp["min_y"]),
                    int(comp["max_x"]),
                    int(comp["max_y"]),
                ),
                int(target),
            )
        )
    transient_classes = {
        target: f"transient_class_{index}"
        for index, target in enumerate(
            target for _bbox, target in sorted(inferred) if target not in class_by_value
        )
    }
    zones: dict[tuple[tuple[int, ...], int], dict[str, Any]] = {}
    for bbox, target in inferred:
        target_class = class_by_value.get(target, transient_classes.get(target))
        if target_class is None:
            continue
        zones[(bbox, target)] = {
            "zone_id": "",
            "bbox": list(bbox),
            "target_value": target,
            "target_class": str(target_class),
            "effect": "recolor_to_visible_class",
        }
    for task in state.get("tasks", ()):
        if not isinstance(task, Mapping):
            continue
        bbox = tuple(int(value) for value in task.get("palette_pad_bbox", ()))
        target = int(task.get("target_color", -1))
        if len(bbox) != 4 or target not in class_by_value:
            continue
        zones[(bbox, target)] = {
            "zone_id": "",
            "bbox": list(bbox),
            "target_value": target,
            "target_class": str(class_by_value[target]),
            "effect": "recolor_to_visible_class",
        }
    out = sorted(zones.values(), key=lambda zone: tuple(zone["bbox"]))
    for index, zone in enumerate(out):
        zone["zone_id"] = f"zone_{index}"
    return out


def _model_walls(
    frame: np.ndarray,
    state: Mapping[str, Any],
    *,
    mode: str,
    background: int,
    marker_values: set[int],
    ring_values: set[int],
    selector_values: set[int],
) -> list[dict[str, Any]]:
    if mode != "wall_morph":
        return []
    obstacles = _shape_wall_morph_obstacles_from_frame(
        frame,
        background_value=background,
        marker_colors=marker_values,
        ring_values=ring_values,
        selector_values=selector_values,
    )
    walls: list[dict[str, Any]] = []
    for index, obstacle in enumerate(obstacles):
        coords = sorted((int(x), int(y)) for x, y in obstacle.get("coords", ()))
        walls.append(
            {
                "wall_id": f"wall_{index}",
                "bbox": [int(value) for value in obstacle.get("bbox", ())],
                "coords": [list(point) for point in coords],
                "coordinate_digest": stable_digest(coords),
            }
        )
    return walls


def _infer_stride(
    state: Mapping[str, Any], markers: Sequence[Mapping[str, Any]]
) -> int:
    piece_strides = {
        int(piece.get("stride", 0) or 0)
        for piece in state.get("pieces", ())
        if isinstance(piece, Mapping) and int(piece.get("stride", 0) or 0) > 0
    }
    if len(piece_strides) == 1:
        return next(iter(piece_strides))
    deltas: list[int] = []
    for task in state.get("tasks", ()):
        if not isinstance(task, Mapping):
            continue
        for left_key, right_key in (
            ("current_pos", "recolor_waypoint"),
            ("recolor_waypoint", "target_anchor"),
        ):
            left = tuple(task.get(left_key, ()))
            right = tuple(task.get(right_key, ()))
            if len(left) != 2 or len(right) != 2:
                continue
            deltas.extend(abs(int(right[index]) - int(left[index])) for index in (0, 1))
    nonzero = [value for value in deltas if value > 0]
    if nonzero:
        return math.gcd(*nonzero)
    coordinates = [(int(marker["x"]), int(marker["y"])) for marker in markers]
    differences: list[int] = []
    for axis in (0, 1):
        values = sorted({point[axis] for point in coordinates})
        differences.extend(
            right - left for left, right in zip(values, values[1:]) if right > left
        )
    return math.gcd(*differences) if differences else 0


def _simulate_one(
    model: Mapping[str, Any], action: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    action = _action_name(action)
    if action not in _ACTIONS:
        return None
    before = deepcopy(dict(model))
    after = deepcopy(dict(model))
    pieces = after.get("pieces")
    if not isinstance(pieces, list):
        return None
    active_index = next(
        (index for index, piece in enumerate(pieces) if bool(piece.get("active"))),
        None,
    )
    if active_index is None:
        return None
    active_before = deepcopy(pieces[active_index])
    transition_kind = "rigid_move"
    zone_effect: dict[str, Any] | None = None
    wall_effect: dict[str, Any] | None = None
    if action == "ACTION5":
        selector_site = tuple(
            int(value) for value in pieces[active_index]["selector_offset"]
        )
        pieces[active_index]["color_offsets"] = [
            list(point)
            for point in pieces[active_index]["color_offsets"]
            if tuple(int(value) for value in point) != selector_site
        ]
        pieces[active_index]["active"] = False
        next_index = (active_index + 1) % len(pieces)
        pieces[next_index]["active"] = True
        active_after = deepcopy(pieces[next_index])
        transition_kind = "select_next_shape"
    else:
        sign_x, sign_y = _MOVE_SIGN[action]
        stride = int(after.get("stride", 0) or 0)
        if stride <= 0:
            return None
        if str(after.get("mode", "")) == "wall_morph":
            moved, wall_effect = _apply_wall_move(
                pieces[active_index],
                dx=sign_x * stride,
                dy=sign_y * stride,
                walls=list(after.get("walls") or []),
                frame_shape=tuple(int(value) for value in after["frame_shape"]),
                stride=stride,
            )
            pieces[active_index] = moved
            transition_kind = (
                "wall_morph" if bool(wall_effect.get("morphed")) else "rigid_move"
            )
        else:
            prior_zones = _overlapping_zones(
                pieces[active_index], after.get("zones", ())
            )
            pieces[active_index]["pos"][0] += sign_x * stride
            pieces[active_index]["pos"][1] += sign_y * stride
            zones = _overlapping_zones(pieces[active_index], after.get("zones", ()))
            prior_ids = {str(zone.get("zone_id", "")) for zone in prior_zones}
            entered = [
                zone for zone in zones if str(zone.get("zone_id", "")) not in prior_ids
            ]
            effective_zones = entered or zones
            targets = {int(zone["target_value"]) for zone in effective_zones}
            if len(targets) > 1:
                return None
            if targets:
                target = next(iter(targets))
                if int(pieces[active_index]["value"]) != target:
                    prior_label = _piece_appearance(pieces[active_index], after)
                    pieces[active_index]["value"] = target
                    zone = effective_zones[0]
                    zone_effect = {
                        "zone_id": str(zone["zone_id"]),
                        "appearance_before": prior_label,
                        "appearance_after": str(zone["target_class"]),
                        "mask_preserved": True,
                    }
                    transition_kind = "zone_recolor"
        active_after = deepcopy(pieces[active_index])
    after["marker_deficit"] = _marker_deficit(after)
    transition = {
        "transition_kind": transition_kind,
        "active_before": active_before,
        "active_after": active_after,
        "zone_effect": zone_effect,
        "wall_effect": wall_effect,
        "deficit_before": int(before.get("marker_deficit", _marker_deficit(before))),
        "deficit_after": int(after["marker_deficit"]),
    }
    return after, transition


def _apply_wall_move(
    piece: Mapping[str, Any],
    *,
    dx: int,
    dy: int,
    walls: Sequence[Mapping[str, Any]],
    frame_shape: tuple[int, int],
    stride: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = deepcopy(dict(piece))
    moved = deepcopy(dict(piece))
    moved["pos"][0] += int(dx)
    moved["pos"][1] += int(dy)
    center = _piece_selector_world(moved)
    if not (0 <= center[0] < frame_shape[1] and 0 <= center[1] < frame_shape[0]):
        return before, {"collision": False, "morphed": False, "blocked": True}
    wall = _first_wall_collision(moved, walls)
    if wall is None:
        return moved, {"collision": False, "morphed": False, "blocked": False}
    kind = str(moved.get("kind", ""))
    if kind == "aspect_exchange_frame":
        transformed = _morph_frame(moved, dx=dx, dy=dy, stride=stride)
    elif kind == "line_shift_cross":
        transformed = _morph_cross(
            moved,
            before,
            wall,
            dx=dx,
            dy=dy,
            stride=stride,
        )
    else:
        transformed = None
    if transformed is None:
        return before, {"collision": True, "morphed": False, "blocked": True}
    return transformed, {
        "collision": True,
        "morphed": True,
        "blocked": False,
        "wall_id": str(wall.get("wall_id", "")),
        "geometry_before": [int(before["width"]), int(before["height"])],
        "geometry_after": [int(transformed["width"]), int(transformed["height"])],
    }


def _morph_frame(
    moved: Mapping[str, Any], *, dx: int, dy: int, stride: int
) -> dict[str, Any] | None:
    out = deepcopy(dict(moved))
    width = int(out["width"])
    height = int(out["height"])
    if dx:
        if width <= 2 * stride:
            return None
        new_width = width - stride
        new_height = height + stride
        out["pos"][1] += height // 2 - new_height // 2
        out["pos"][1] = round(out["pos"][1] / stride) * stride
        if dx < 0:
            out["pos"][0] -= dx
    else:
        if height <= 2 * stride:
            return None
        new_width = width + stride
        new_height = height - stride
        out["pos"][0] += width // 2 - new_width // 2
        out["pos"][0] = round(out["pos"][0] / stride) * stride
        if dy < 0:
            out["pos"][1] -= dy
    out["width"] = int(new_width)
    out["height"] = int(new_height)
    out["selector_offset"] = [new_width // 2, new_height // 2]
    offsets = {
        (x, y)
        for y in range(new_height)
        for x in range(new_width)
        if x in {0, new_width - 1} or y in {0, new_height - 1}
    }
    out["color_offsets"] = [list(point) for point in sorted(offsets)]
    return out


def _morph_cross(
    moved: Mapping[str, Any],
    previous: Mapping[str, Any],
    wall: Mapping[str, Any],
    *,
    dx: int,
    dy: int,
    stride: int,
) -> dict[str, Any] | None:
    out = deepcopy(dict(moved))
    offsets = {tuple(point) for point in out["color_offsets"]}
    selector = tuple(out["selector_offset"])
    occupied = offsets | {selector}
    full_cols = [
        x
        for x in range(int(out["width"]))
        if all((x, y) in occupied for y in range(int(out["height"])))
    ]
    full_rows = [
        y
        for y in range(int(out["height"]))
        if all((x, y) in occupied for x in range(int(out["width"])))
    ]
    if not full_cols or not full_rows:
        return None
    col, row = full_cols[0], full_rows[0]
    min_x, min_y, max_x, max_y = (int(value) for value in wall["bbox"])
    col_in = min_x <= int(out["pos"][0]) + col <= max_x
    row_in = min_y <= int(out["pos"][1]) + row <= max_y
    if col_in and row_in:
        return None
    if dx:
        if col_in:
            shift = -stride if dx > 0 else stride
            if not 0 <= col + shift < int(out["width"]):
                return None
            offsets = {(x, y) for x, y in offsets if x != col}
            offsets.update((col + shift, y) for y in range(int(out["height"])))
            offsets.add((col, row))
        elif row_in:
            out = deepcopy(dict(previous))
            offsets = {tuple(point) for point in out["color_offsets"]}
            shift = stride if dx > 0 else -stride
            if not 0 <= col + shift < int(out["width"]):
                return None
            offsets = {(x, y) for x, y in offsets if x != col}
            offsets.update((col + shift, y) for y in range(int(out["height"])))
            offsets.add((col, row))
        else:
            return out
    elif dy:
        if row_in:
            shift = -stride if dy > 0 else stride
            if not 0 <= row + shift < int(out["height"]):
                return None
            offsets = {(x, y) for x, y in offsets if y != row}
            offsets.update((x, row + shift) for x in range(int(out["width"])))
            offsets.add((col, row))
        elif col_in:
            out = deepcopy(dict(previous))
            offsets = {tuple(point) for point in out["color_offsets"]}
            shift = stride if dy > 0 else -stride
            if not 0 <= row + shift < int(out["height"]):
                return None
            offsets = {(x, y) for x, y in offsets if y != row}
            offsets.update((x, row + shift) for x in range(int(out["width"])))
            offsets.add((col, row))
        else:
            return out
    out["color_offsets"] = [list(point) for point in sorted(offsets)]
    return out


def _first_wall_collision(
    piece: Mapping[str, Any], walls: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    occupied = _piece_world_points(piece, include_selector=True)
    for wall in walls:
        coords = {tuple(point) for point in wall.get("coords", ())}
        if occupied & coords:
            return wall
    return None


def _overlapping_zones(
    piece: Mapping[str, Any], zones: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    points = _piece_world_points(piece, include_selector=True)
    matches: list[Mapping[str, Any]] = []
    for zone in zones:
        min_x, min_y, max_x, max_y = (int(value) for value in zone["bbox"])
        if any(min_x <= x <= max_x and min_y <= y <= max_y for x, y in points):
            matches.append(zone)
    return matches


def _predict_resolved(resolved: Mapping[str, Any], action: str) -> dict[str, Any]:
    action = _action_name(action)
    plan = list(resolved.get("remaining_plan") or [])
    if not plan or action != plan[0]:
        return _prediction_rejection(action, "action_is_not_current_residual_prefix")
    model = deepcopy(dict(resolved["model"]))
    simulated = _simulate_one(model, action)
    if simulated is None:
        return _prediction_rejection(action, "operator_simulation_failed")
    expected_model, transition = simulated
    active_before = transition["active_before"]
    active_after = transition["active_after"]
    after_points = sorted(_piece_world_color_points(active_after))
    exact_piece = {
        "piece_before": str(active_before.get("piece_id", "")),
        "piece_after": str(active_after.get("piece_id", "")),
        "position_before": list(active_before.get("pos", ())),
        "position_after": list(active_after.get("pos", ())),
        "geometry_before": [int(active_before["width"]), int(active_before["height"])],
        "geometry_after": [int(active_after["width"]), int(active_after["height"])],
        "mask_before_digest": stable_digest(active_before.get("color_offsets", ())),
        "mask_after_digest": stable_digest(active_after.get("color_offsets", ())),
        "appearance_before": _piece_appearance(active_before, model),
        "appearance_after": _piece_appearance(active_after, expected_model),
        "expected_class_pixels": [list(point) for point in after_points],
    }
    deficit_before = int(transition["deficit_before"])
    deficit_after = int(transition["deficit_after"])
    return {
        "schema_version": f"{_SCHEMA_VERSION}.transition",
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "accepted": True,
        "reason": "exact_visible_shape_operator_prefix",
        "action": action,
        "transition_kind": str(transition["transition_kind"]),
        "exact_piece_transition": exact_piece,
        "selection_transition": (
            {
                "from_piece": str(active_before.get("piece_id", "")),
                "to_piece": str(active_after.get("piece_id", "")),
            }
            if action == "ACTION5"
            else None
        ),
        "zone_effect": stable_data(transition.get("zone_effect")),
        "wall_effect": stable_data(transition.get("wall_effect")),
        "marker_deficit": {
            "before": deficit_before,
            "after": deficit_after,
            "delta": deficit_after - deficit_before,
        },
        "static_signature": _public_static_signature(model),
        "residual_before": len(plan),
        "residual_expected_after": len(plan) - 1,
        "residual_suffix_digest": stable_digest(plan[1:]),
        "dynamic_footprint": [
            list(point)
            for point in sorted(
                _all_piece_world_points(model) | _all_piece_world_points(expected_model)
            )
        ],
        "before_runtime_model": stable_data(model),
        "expected_runtime_model": stable_data(expected_model),
        "before_frame": np.asarray(resolved["frame"]).astype(int, copy=False).tolist(),
    }


def _match_successor(
    before_frame: np.ndarray,
    before_model: Mapping[str, Any],
    current_frame: np.ndarray,
    expected_model: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    expected_shape = tuple(
        int(value) for value in expected_model.get("frame_shape", ())
    )
    if (
        before_frame.shape != current_frame.shape
        or current_frame.shape != expected_shape
    ):
        return ["frame_shape_changed"], {"unchanged_pixel_count": 0}
    dynamic = _all_piece_world_points(before_model) | _all_piece_world_points(
        expected_model
    )
    unchanged_checked = 0
    for y in range(current_frame.shape[0]):
        for x in range(current_frame.shape[1]):
            if (x, y) in dynamic:
                continue
            unchanged_checked += 1
            if int(before_frame[y, x]) != int(current_frame[y, x]):
                reasons.append(f"unrelated_pixel_changed:{x},{y}")
                if len(reasons) >= 8:
                    break
        if len(reasons) >= 8:
            break
    active_count = 0
    for piece in expected_model.get("pieces", ()):
        if not isinstance(piece, Mapping):
            reasons.append("invalid_expected_piece")
            continue
        active_count += int(bool(piece.get("active")))
    expected_pixels = _expected_piece_pixels(expected_model)
    checked_class_pixels = 0
    occluded_pixels = 0
    for (x, y), row in expected_pixels.items():
        if not _in_bounds((x, y), current_frame.shape):
            continue
        if _piece_bbox_cover_count(expected_model, (x, y)) > 1:
            occluded_pixels += 1
            continue
        checked_class_pixels += 1
        if int(current_frame[y, x]) != int(row["value"]):
            reasons.append(f"shape_pixel_mismatch:{row['piece_id']}:{x},{y}")
            if len(reasons) >= 16:
                break
    if active_count != 1:
        reasons.append("active_shape_count_mismatch")
    evidence = {
        "unchanged_pixel_count": unchanged_checked,
        "checked_shape_pixel_count": checked_class_pixels,
        "occluded_shape_pixel_count": occluded_pixels,
        "active_shape_count": active_count,
    }
    return reasons, evidence


def _piece_bbox_cover_count(model: Mapping[str, Any], point: tuple[int, int]) -> int:
    count = 0
    for piece in model.get("pieces", ()):
        if not isinstance(piece, Mapping):
            continue
        x0, y0 = (int(value) for value in piece.get("pos", (0, 0)))
        if x0 <= point[0] < x0 + int(piece.get("width", 0) or 0) and y0 <= point[
            1
        ] < y0 + int(piece.get("height", 0) or 0):
            count += 1
    return count


def _expected_piece_pixels(
    model: Mapping[str, Any],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Apply the frame-inferred shape order to overlapping visible pixels."""

    rendered: dict[tuple[int, int], dict[str, Any]] = {}
    pieces = sorted(
        (piece for piece in model.get("pieces", ()) if isinstance(piece, Mapping)),
        key=lambda piece: int(piece.get("cycle_rank", 0) or 0),
    )
    selector_value = int(model.get("selector_value", -1))
    for piece in pieces:
        piece_id = str(piece.get("piece_id", ""))
        for point in _piece_world_color_points(piece):
            rendered[point] = {
                "value": int(piece.get("value", -1)),
                "piece_id": piece_id,
            }
        if bool(piece.get("active")):
            rendered[_piece_selector_world(piece)] = {
                "value": selector_value,
                "piece_id": piece_id,
            }
    return rendered


def _marker_deficit(model: Mapping[str, Any]) -> int:
    pieces = [piece for piece in model.get("pieces", ()) if isinstance(piece, Mapping)]
    deficit = 0
    for marker_class in model.get("marker_classes", ()):
        if not isinstance(marker_class, Mapping):
            continue
        value = int(marker_class.get("value", -1))
        matching_points: set[tuple[int, int]] = set()
        for piece in pieces:
            if int(piece.get("value", -2)) == value:
                matching_points.update(_piece_world_color_points(piece))
        for point in marker_class.get("positions", ()):
            if tuple(int(value) for value in point) not in matching_points:
                deficit += 1
    return deficit


def _piece_world_color_points(piece: Mapping[str, Any]) -> set[tuple[int, int]]:
    x0, y0 = (int(value) for value in piece.get("pos", (0, 0)))
    selector = tuple(int(value) for value in piece.get("selector_offset", ()))
    offsets = {
        (int(point[0]), int(point[1])) for point in piece.get("color_offsets", ())
    }
    if bool(piece.get("active")):
        offsets.discard(selector)
    return {(x0 + x, y0 + y) for x, y in offsets}


def _piece_world_points(
    piece: Mapping[str, Any], *, include_selector: bool
) -> set[tuple[int, int]]:
    points = _piece_world_color_points(piece)
    if include_selector and bool(piece.get("active")):
        points.add(_piece_selector_world(piece))
    return points


def _piece_selector_world(piece: Mapping[str, Any]) -> tuple[int, int]:
    x0, y0 = (int(value) for value in piece.get("pos", (0, 0)))
    sx, sy = (int(value) for value in piece.get("selector_offset", (0, 0)))
    return x0 + sx, y0 + sy


def _all_piece_world_points(model: Mapping[str, Any]) -> set[tuple[int, int]]:
    points = {
        (int(point[0]), int(point[1]))
        for point in model.get("volatile_status_points", ())
    }
    for piece in model.get("pieces", ()):
        if isinstance(piece, Mapping):
            x0, y0 = (int(value) for value in piece.get("pos", (0, 0)))
            points.update(
                (x0 + x, y0 + y)
                for y in range(int(piece.get("height", 0) or 0))
                for x in range(int(piece.get("width", 0) or 0))
            )
    return points


def _volatile_status_points(frame: np.ndarray, background: int) -> set[tuple[int, int]]:
    """Infer low-entropy boundary strips used for visible action telemetry."""

    height, width = frame.shape
    lines = [
        [(x, 0) for x in range(width)],
        [(x, height - 1) for x in range(width)],
        [(0, y) for y in range(height)],
        [(width - 1, y) for y in range(height)],
    ]
    points: set[tuple[int, int]] = set()
    for line in lines:
        values = [int(frame[y, x]) for x, y in line]
        counts = Counter(values)
        transitions = sum(left != right for left, right in zip(values, values[1:]))
        if int(background) not in counts and len(counts) <= 2 and transitions <= 2:
            points.update(line)
    return points


def _piece_appearance(piece: Mapping[str, Any], model: Mapping[str, Any]) -> str:
    value = int(piece.get("value", -1))
    for marker_class in model.get("marker_classes", ()):
        if (
            isinstance(marker_class, Mapping)
            and int(marker_class.get("value", -2)) == value
        ):
            return str(marker_class.get("class_id", ""))
    for zone in model.get("zones", ()):
        if isinstance(zone, Mapping) and int(zone.get("target_value", -2)) == value:
            return str(zone.get("target_class", ""))
    return f"source_class_{piece.get('piece_id', '')}"


def _public_binding(
    model: Mapping[str, Any], resolved: Mapping[str, Any]
) -> dict[str, Any]:
    pieces: list[dict[str, Any]] = []
    roles: list[dict[str, Any]] = []
    for piece in model.get("pieces", ()):
        if not isinstance(piece, Mapping):
            continue
        public = {
            "role": "movable_shape",
            "piece_id": str(piece["piece_id"]),
            "appearance_class": _piece_appearance(piece, model),
            "target_marker_class": str(piece["target_class"]),
            "position": list(piece["pos"]),
            "size": [int(piece["width"]), int(piece["height"])],
            "mask": [list(point) for point in piece["color_offsets"]],
            "active": bool(piece["active"]),
            "kind": str(piece.get("kind", "rigid_mask")),
        }
        pieces.append(public)
        roles.append(dict(public))
    marker_map = [
        {
            "role": "same_class_marker_map",
            "class_id": str(row["class_id"]),
            "positions": [list(point) for point in row["positions"]],
        }
        for row in model.get("marker_classes", ())
        if isinstance(row, Mapping)
    ]
    roles.extend(marker_map)
    walls = [
        {
            "role": "wall",
            "wall_id": str(wall["wall_id"]),
            "bbox": list(wall["bbox"]),
            "coordinate_digest": str(wall["coordinate_digest"]),
        }
        for wall in model.get("walls", ())
        if isinstance(wall, Mapping)
    ]
    roles.extend(walls)
    zones = [
        {
            "role": "morph_or_recolor_zone",
            "zone_id": str(zone["zone_id"]),
            "bbox": list(zone["bbox"]),
            "target_marker_class": str(zone["target_class"]),
            "effect": str(zone["effect"]),
        }
        for zone in model.get("zones", ())
        if isinstance(zone, Mapping)
    ]
    roles.extend(zones)
    active = next(piece for piece in pieces if bool(piece["active"]))
    return {
        "schema_version": f"{_SCHEMA_VERSION}.binding",
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "status": "bound",
        "supported": True,
        "accepted": True,
        "mode": str(model["mode"]),
        "roles": roles,
        "role_bindings": roles,
        "pieces": pieces,
        "active_shape": active,
        "same_class_marker_map": marker_map,
        "walls": walls,
        "morph_or_recolor_zones": zones,
        "marker_deficit": int(model["marker_deficit"]),
        "stride": int(model["stride"]),
        "history_rebound": bool(resolved.get("history_rebound")),
        "private_plan_length": len(resolved.get("remaining_plan") or []),
        "static_signature": _public_static_signature(model),
    }


def _public_static_signature(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frame_shape": [int(value) for value in model.get("frame_shape", ())],
        "stride": int(model.get("stride", 0) or 0),
        "marker_classes": [
            {
                "class_id": str(row["class_id"]),
                "positions": [list(point) for point in row["positions"]],
            }
            for row in model.get("marker_classes", ())
            if isinstance(row, Mapping)
        ],
        "walls": [
            {
                "bbox": list(wall["bbox"]),
                "coordinate_digest": str(wall["coordinate_digest"]),
            }
            for wall in model.get("walls", ())
            if isinstance(wall, Mapping)
        ],
        "zones": [
            {
                "bbox": list(zone["bbox"]),
                "target_marker_class": str(zone["target_class"]),
            }
            for zone in model.get("zones", ())
            if isinstance(zone, Mapping)
        ],
    }


def _state_has_one_active_shape(state: Mapping[str, Any]) -> bool:
    pieces = [piece for piece in state.get("pieces", ()) if isinstance(piece, Mapping)]
    return bool(pieces) and sum(bool(piece.get("active")) for piece in pieces) == 1


def _normalized_plan(plan: Any) -> list[str]:
    if not isinstance(plan, Sequence) or isinstance(plan, (str, bytes, bytearray)):
        return []
    normalized = [_action_name(action) for action in plan]
    if not normalized or any(action not in _ACTIONS for action in normalized):
        return []
    return normalized


def _action_name(action: Any) -> str:
    if isinstance(action, Mapping):
        action = action.get("action")
    if hasattr(action, "name"):
        action = action.name
    name = str(action or "").strip().upper().split(".")[-1]
    if "(" in name:
        name = name.split("(", 1)[0]
    return _ALIASES.get(name, name if name in _ACTIONS else "")


def _allowed_actions(value: Sequence[Any] | set[Any] | None) -> set[str]:
    if not value:
        return set(_ACTIONS)
    return {name for item in value if (name := _action_name(item)) in _ACTIONS}


def _in_bounds(point: tuple[int, int], shape: tuple[int, int]) -> bool:
    return 0 <= point[0] < shape[1] and 0 <= point[1] < shape[0]


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return type(value).__name__


def _unbound(reason: str) -> dict[str, Any]:
    return {
        "schema_version": f"{_SCHEMA_VERSION}.binding",
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "status": "unbound",
        "supported": False,
        "accepted": False,
        "reject_reason": str(reason),
        "roles": [],
        "role_bindings": [],
        "pieces": [],
        "same_class_marker_map": [],
        "walls": [],
        "morph_or_recolor_zones": [],
    }


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": f"{_SCHEMA_VERSION}.plan",
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "status": "blocked",
        "actions": [],
        "action_count": 0,
        "certified": False,
        "certificate_reason": str(reason),
        "detail": {},
    }


def _prediction_rejection(action: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": f"{_SCHEMA_VERSION}.transition",
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "accepted": False,
        "action": str(action),
        "reason": str(reason),
    }


def _verification(
    accepted: bool,
    reasons: Sequence[str],
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": f"{_SCHEMA_VERSION}.verification",
        "family_id": FAMILY_ID,
        "accepted": bool(accepted),
        "reasons": [str(reason) for reason in reasons],
        "evidence": stable_data(dict(evidence or {})),
    }


__all__ = [
    "FAMILY_ID",
    "bind_current",
    "compile_schema",
    "predict_transition",
    "propose_plan",
    "verify_transition",
]
