"""Frame-grounded bit-program mechanic for Tufa advisory planning.

The module deliberately has no game-id input.  It binds visible program bits,
RUN controls, and actor/target tuples, then exposes one freshly grounded click
at a time.  Transition verification is exact for bit toggles and fail-closed
for RUN unless visible translation or bit-decoded evidence certifies the
complete trace.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from arcgym.eca.program_row_semantics_sensor import (
    _program_translation_actor_pairs,
    _program_translation_playfield,
    _program_translation_stride,
    build_program_row_semantics_checkpoint_bfs_proposals,
    build_program_row_semantics_packet,
    build_program_row_semantics_polarity_fill_proposals,
    build_program_row_semantics_scale_translation_bfs_proposals,
    build_program_row_semantics_shape_transform_bfs_proposals,
    build_program_row_semantics_translation_bfs_proposals,
)
from arcgym.eca.tufa_mechanics.base import (
    MechanicContext,
    as_frame,
    json_safe_mapping,
    stable_digest,
)


FAMILY_ID = "bit_program_execution"
_SCHEMA_VERSION = "tufa_bit_program_execution.v1"
_BINDING_VERSION = "tufa_bit_program_binding.v1"
_PREDICTION_VERSION = "tufa_bit_program_transition.v1"
_VERIFICATION_VERSION = "tufa_bit_program_verification.v1"

_TRANSLATION_BUILDERS = (
    ("checkpoint_bfs", build_program_row_semantics_checkpoint_bfs_proposals),
    ("shape_transform_bfs", build_program_row_semantics_shape_transform_bfs_proposals),
    (
        "scale_translation_bfs",
        build_program_row_semantics_scale_translation_bfs_proposals,
    ),
    ("translation_bfs", build_program_row_semantics_translation_bfs_proposals),
)
_TRANSFORM_BUILDERS = (
    (
        "scale_translation_bfs",
        build_program_row_semantics_scale_translation_bfs_proposals,
    ),
    ("shape_transform_bfs", build_program_row_semantics_shape_transform_bfs_proposals),
)
_VISUAL_OPCODE_DELTAS = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, 1),
    10: (2, 0),
    11: (2, 0),
    12: (-2, 0),
    13: (-2, 0),
    33: (0, -1),
    34: (-1, 0),
}
_DETERMINISTIC_RUN_CERTAINTIES = {
    "deterministic_visible_translation",
    "deterministic_visible_bit_trace",
    "deterministic_visible_scale_translation",
    "deterministic_visible_shape_transform",
}


def compile_schema(
    frame: Any,
    level: Any,
    context: MechanicContext | None = None,
) -> dict[str, Any]:
    """Compile a JSON-safe mechanic schema from current visible evidence."""

    binding = bind_current(frame, context=context)
    level_value = level if isinstance(level, (str, int, float, bool)) else None
    schema = {
        "schema_version": _SCHEMA_VERSION,
        "family_id": FAMILY_ID,
        "supported": bool(binding.get("supported")),
        "level": level_value,
        "roles": [
            "program_row",
            "program_bit",
            "run_control",
            "actor",
            "target",
            "opcode_model",
        ],
        "required_bindings": ["program_row", "program_bit", "run_control"],
        "state_variables": [
            "visible_bit_values",
            "program_row_opcodes",
            "actor_tuple",
            "target_tuple",
            "actor_scale",
            "actor_rotation",
            "actor_shape",
        ],
        "operators": [
            {
                "name": "toggle_visible_bit",
                "effect": "exactly_one_visible_bit_changes_binary_polarity",
            },
            {
                "name": "run_program",
                "effect": "execute_visible_rows_in_display_order",
            },
        ],
        "operator": {
            "kind": "bit_program_execution",
            "transition": "toggle one visible bit or execute the visible rows",
        },
        "invariants": [
            "program row order remains visible",
            "a toggle changes exactly one bound bit",
        ],
        "falsifiers": [
            "an unrelated visible bit changes",
            "the rebound actor trace disagrees with the visible program",
        ],
        "goal_relation": "actor_tuple_matches_visible_target_tuple_after_run",
        "binding_digest": binding.get("binding_digest", ""),
        "evidence": {
            "visible_row_count": len(binding.get("visible_rows", [])),
            "visible_bit_count": len(binding.get("visible_bits", [])),
            "run_control_count": len(binding.get("run_controls", [])),
            "actor_target_tuple_count": len(binding.get("actor_target_tuples", [])),
            "transform_target_tuple_count": len(
                binding.get("transform_target_tuples", [])
            ),
            "verified_history": dict(binding.get("verified_transition_history", {})),
        },
        "verification": {
            "bit_toggle": "exact_pixel_delta",
            "run": "deterministic_trace_and_level_transition",
            "fail_closed": True,
        },
    }
    return _json(schema)


def bind_current(
    frame: Any,
    context: MechanicContext | None = None,
) -> dict[str, Any]:
    """Bind visible rows, bits, RUN controls, and actor/target tuple evidence."""

    arr = as_frame(frame)
    if arr is None:
        return _json(
            {
                "schema_version": _BINDING_VERSION,
                "family_id": FAMILY_ID,
                "supported": False,
                "reason": "invalid_frame",
                "visible_rows": [],
                "visible_bits": [],
                "run_controls": [],
                "actor_target_tuples": [],
                "transform_target_tuples": [],
                "verified_transition_history": _verified_transition_history(context),
            }
        )

    packet = build_program_row_semantics_packet(arr)
    cells = [
        dict(row)
        for row in packet.get("program_cells_before", [])
        if isinstance(row, Mapping)
    ]
    row_centers = [float(value) for value in packet.get("row_centers", [])]
    bit_centers = [float(value) for value in packet.get("bit_centers", [])]
    visible_bits = _visible_bits(cells, row_centers, bit_centers)
    visible_rows = _visible_rows(visible_bits, row_centers)
    run_controls = [
        _run_control(row) for row in packet.get("run_button_regions_before", [])
    ]
    run_controls = [row for row in run_controls if row is not None]
    cell_colors = {int(row["color"]) for row in visible_bits}
    # The source sensor defines 1/5 as the observable program polarity pair.
    # Keep that pair bound when one polarity is temporarily absent after edits.
    binary_colors = (
        [1, 5] if cell_colors and cell_colors <= {1, 5} else sorted(cell_colors)
    )
    active_color = (
        5 if 5 in binary_colors else (max(binary_colors) if binary_colors else -1)
    )
    actor_target_tuples = _actor_target_evidence(
        arr,
        row_centers=row_centers,
        bit_centers=bit_centers,
        run_controls=run_controls,
    )
    transform_target_tuples = _transform_target_evidence(
        arr,
        active_color=active_color,
    )
    verified_transition_history = _verified_transition_history(context)
    supported = bool(
        len(visible_rows) >= 2
        and len(bit_centers) >= 2
        and len(binary_colors) == 2
        and run_controls
        and (actor_target_tuples or transform_target_tuples)
    )
    core = {
        "frame_shape": [int(arr.shape[0]), int(arr.shape[1])],
        "visible_rows": visible_rows,
        "visible_bits": visible_bits,
        "run_controls": run_controls,
        "actor_target_tuples": actor_target_tuples,
        "transform_target_tuples": transform_target_tuples,
        "verified_transition_history": verified_transition_history,
        "binary_colors": binary_colors,
        "active_color": int(active_color),
    }
    return _json(
        {
            "schema_version": _BINDING_VERSION,
            "family_id": FAMILY_ID,
            "supported": supported,
            "reason": "visible_program_bound"
            if supported
            else "insufficient_visible_evidence",
            "frame_digest": stable_digest(arr),
            "binding_digest": stable_digest(core),
            **core,
        }
    )


def propose_plan(
    frame: Any,
    valid_actions: Any = None,
    context: MechanicContext | None = None,
) -> dict[str, Any]:
    """Return one freshly grounded action; never expose a stored action suffix."""

    arr = as_frame(frame)
    binding = bind_current(frame, context=context)
    if arr is None or not binding.get("supported"):
        return _blocked_plan(binding, "insufficient_visible_evidence")
    if not _mouse_allowed(valid_actions):
        return _blocked_plan(binding, "mouse_action_not_available")

    candidate = _preferred_candidate(arr, binding)
    if candidate is None:
        return _blocked_plan(binding, "no_frame_grounded_program_candidate")
    steps = [row for row in candidate.get("plan", []) if isinstance(row, Mapping)]
    if not steps:
        return _blocked_plan(binding, "candidate_has_no_actions")
    step = steps[0]
    try:
        action = {"action": "MOUSE", "row": int(step["y"]), "col": int(step["x"])}
    except (KeyError, TypeError, ValueError):
        return _blocked_plan(binding, "candidate_action_is_not_grounded")
    prediction = predict_transition(frame, action, context=context)
    certified = bool(prediction.get("supported"))
    progress_before = len(steps)
    return _json(
        {
            "schema_version": "tufa_bit_program_plan.v1",
            "family_id": FAMILY_ID,
            "status": "proposed",
            "binding_digest": binding.get("binding_digest", ""),
            "source": str(candidate.get("mechanic_source", "")),
            "candidate_kind": str(candidate.get("candidate_kind", "")),
            "action_role": str(step.get("role", "")),
            "actions": [action],
            "action_count": 1,
            "certified": certified,
            "certificate_reason": (
                "frame-bound exact bit transition"
                if str(prediction.get("kind", "")) == "one_bit_toggle"
                else "frame-bound deterministic visible program trace"
                if certified
                else str(prediction.get("reason", "transition_not_certified"))
            ),
            "transition_rule": str(prediction.get("kind", "bit_program_execution")),
            "progress_measure": {
                "metric": "remaining_program_actions",
                "before": int(progress_before),
                "expected_after": max(0, int(progress_before) - 1),
                "strict": True,
            },
            "suffix_exposed": False,
            "requires_transition_verification": True,
            "abstract_program_evidence": {
                "op_sequence": [
                    int(value) for value in candidate.get("op_sequence", [])
                ],
                "desired_opcodes": [
                    int(value) for value in candidate.get("desired_opcodes", [])
                ],
            },
            "detail": {"private_plan_length": int(progress_before)},
        }
    )


def predict_transition(
    frame: Any,
    action: Any,
    context: MechanicContext | None = None,
) -> dict[str, Any]:
    """Predict an exact bit delta or a certified deterministic RUN trace."""

    arr = as_frame(frame)
    normalized = _mouse_action(action)
    binding = bind_current(frame, context=context)
    if arr is None or normalized is None or not binding.get("supported"):
        return _unsupported_prediction(
            binding, normalized, "invalid_or_unbound_transition"
        )
    x, y = int(normalized["col"]), int(normalized["row"])
    bit = _selected_bit(binding.get("visible_bits", []), x=x, y=y)
    if bit is not None:
        colors = [int(value) for value in binding.get("binary_colors", [])]
        source_color = int(bit["color"])
        targets = [value for value in colors if value != source_color]
        if len(targets) != 1:
            return _unsupported_prediction(
                binding, normalized, "binary_polarity_not_bound"
            )
        bbox = [int(value) for value in bit["bbox"]]
        return _json(
            {
                "schema_version": _PREDICTION_VERSION,
                "family_id": FAMILY_ID,
                "supported": True,
                "kind": "one_bit_toggle",
                "action": normalized,
                "before_frame": arr.tolist(),
                "before_frame_digest": stable_digest(arr),
                "binding_digest": binding.get("binding_digest", ""),
                "bit": {
                    "row_index": int(bit["row_index"]),
                    "bit_index": int(bit["bit_index"]),
                    "bbox": bbox,
                    "source_color": source_color,
                    "target_color": int(targets[0]),
                    "pixel_count": int(
                        (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
                    ),
                },
            }
        )

    run = _selected_run(binding.get("run_controls", []), x=x, y=y)
    if run is None:
        return _unsupported_prediction(
            binding, normalized, "click_is_not_a_bound_control"
        )
    deterministic = _deterministic_run_evidence(arr, binding, run)
    if deterministic is None:
        return _unsupported_prediction(binding, normalized, "run_trace_not_determined")
    return _json(
        {
            "schema_version": _PREDICTION_VERSION,
            "family_id": FAMILY_ID,
            "supported": True,
            "kind": "deterministic_run",
            "action": normalized,
            "before_frame": arr.tolist(),
            "before_frame_digest": stable_digest(arr),
            "binding_digest": binding.get("binding_digest", ""),
            "before_geometry": _geometry_signature(binding),
            "run_control": run,
            **deterministic,
        }
    )


def verify_transition(expected: Any, current_frame: Any) -> dict[str, Any]:
    """Verify a predicted transition, rejecting every unmodelled frame delta."""

    if not isinstance(expected, Mapping) or not bool(expected.get("supported")):
        return _verification(False, "unsupported_prediction")
    current = as_frame(current_frame)
    before = as_frame(expected.get("before_frame"))
    action = expected.get("action")
    normalized = _mouse_action(action)
    if (
        current is None
        or before is None
        or normalized is None
        or current.shape != before.shape
    ):
        return _verification(False, "invalid_transition_frames")
    if stable_digest(before) != str(expected.get("before_frame_digest", "")):
        return _verification(False, "before_frame_digest_mismatch")

    kind = str(expected.get("kind", ""))
    if kind == "one_bit_toggle":
        return _verify_one_bit(expected, before, current, normalized)
    if kind == "deterministic_run":
        return _verify_deterministic_run(expected, before, current, normalized)
    return _verification(False, "unknown_prediction_kind")


def _visible_bits(
    cells: Sequence[Mapping[str, Any]],
    row_centers: Sequence[float],
    bit_centers: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        centroid = cell.get("centroid")
        bbox = cell.get("bbox")
        if not _xy(centroid) or not _bbox(bbox):
            continue
        cx, cy = float(centroid[0]), float(centroid[1])
        row_index = _nearest_index(row_centers, cx)
        bit_index = _nearest_index(bit_centers, cy)
        if row_index is None or bit_index is None:
            continue
        rows.append(
            {
                "row_index": int(row_index),
                "bit_index": int(bit_index),
                "row_center": round(float(row_centers[row_index]), 3),
                "bit_center": round(float(bit_centers[bit_index]), 3),
                "color": int(cell.get("color", -1)),
                "bbox": [int(value) for value in bbox[:4]],
                "centroid": [round(cx, 3), round(cy, 3)],
                "orientation": str(cell.get("orientation", "")),
            }
        )
    return sorted(rows, key=lambda row: (row["row_index"], row["bit_index"]))


def _visible_rows(
    visible_bits: Sequence[Mapping[str, Any]], row_centers: Sequence[float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, center in enumerate(row_centers):
        bits = [dict(bit) for bit in visible_bits if int(bit["row_index"]) == row_index]
        if bits:
            rows.append(
                {
                    "row_index": int(row_index),
                    "center": round(float(center), 3),
                    "bits": bits,
                }
            )
    return rows


def _run_control(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or not _bbox(value.get("bbox")):
        return None
    centroid = value.get("centroid")
    if not _xy(centroid):
        return None
    return {
        "bbox": [int(item) for item in value["bbox"][:4]],
        "centroid": [round(float(centroid[0]), 3), round(float(centroid[1]), 3)],
        "color": int(value.get("color", -1)),
        "area": int(value.get("area", 0)),
    }


def _actor_target_evidence(
    arr: np.ndarray,
    *,
    row_centers: Sequence[float],
    bit_centers: Sequence[float],
    run_controls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not row_centers or not bit_centers or not run_controls:
        return []
    largest_run = max(run_controls, key=lambda row: int(row.get("area", 0)))
    playfield = _program_translation_playfield(
        arr,
        columns=row_centers,
        bit_centers=bit_centers,
        run_region=largest_run,
    )
    pairs = _program_translation_actor_pairs(arr, playfield_bbox=playfield["bbox"])
    stride = _program_translation_stride(arr, playfield_bbox=playfield["bbox"])
    evidence: list[dict[str, Any]] = []
    for pair in pairs:
        start = pair.get("start_centroid")
        target = pair.get("target_centroid")
        if not _xy(start) or not _xy(target):
            continue
        evidence.append(
            {
                "actor": {
                    "color": int(pair.get("color", -1)),
                    "bbox": [int(value) for value in pair.get("start_bbox", [])[:4]],
                    "centroid": [round(float(start[0]), 3), round(float(start[1]), 3)],
                },
                "target": {
                    "color": int(pair.get("color", -1)),
                    "bbox": [int(value) for value in pair.get("target_bbox", [])[:4]],
                    "centroid": [
                        round(float(target[0]), 3),
                        round(float(target[1]), 3),
                    ],
                },
                "delta": [
                    round(float(target[0]) - float(start[0]), 3),
                    round(float(target[1]) - float(start[1]), 3),
                ],
                "stride": int(stride),
                "evidence": "same_color_visible_component_pair",
            }
        )
    return evidence


def _transform_target_evidence(
    arr: np.ndarray,
    *,
    active_color: int,
) -> list[dict[str, Any]]:
    """Project transform proposals into action-free actor/target bindings."""

    evidence: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for source, builder in _TRANSFORM_BUILDERS:
        try:
            output = builder(arr, active_color=active_color, max_plan_steps=96)
        except Exception:
            continue
        for proposal in output.get("proposals", []):
            if not isinstance(proposal, Mapping):
                continue
            start_bbox = _int_bbox(proposal.get("start_bbox"))
            target_bbox = _int_bbox(proposal.get("target_bbox"))
            opcodes = _int_sequence(proposal.get("op_sequence"))
            if start_bbox is None or target_bbox is None or not opcodes:
                continue
            key = (source, tuple(start_bbox), tuple(target_bbox), tuple(opcodes))
            if key in seen:
                continue
            seen.add(key)
            start_center = _bbox_center(start_bbox)
            target_center = _bbox_center(target_bbox)
            semantics: list[dict[str, Any]] = []
            scale_opcode = _safe_int(proposal.get("scale_opcode"), -1)
            rotation_opcode = _safe_int(proposal.get("rotation_opcode"), -1)
            shape_opcode = _safe_int(proposal.get("shape_opcode"), -1)
            if scale_opcode >= 0:
                semantics.append(
                    {
                        "opcode": scale_opcode,
                        "effect": str(proposal.get("scale_relation", "scale_change")),
                    }
                )
            if rotation_opcode >= 0:
                semantics.append(
                    {"opcode": rotation_opcode, "effect": "visible_quarter_turn"}
                )
            if shape_opcode >= 0:
                semantics.append(
                    {"opcode": shape_opcode, "effect": "visible_shape_role_change"}
                )
            for opcode in _int_sequence(proposal.get("translation_op_sequence")):
                delta = _VISUAL_OPCODE_DELTAS.get(opcode)
                if delta is not None:
                    semantics.append(
                        {
                            "opcode": opcode,
                            "effect": "visible_translation",
                            "lattice_delta": [int(delta[0]), int(delta[1])],
                        }
                    )
            evidence.append(
                {
                    "kind": source,
                    "actor": {
                        "bbox": start_bbox,
                        "centroid": start_center,
                        "extent": _bbox_extent(start_bbox),
                    },
                    "target": {
                        "bbox": target_bbox,
                        "centroid": target_center,
                        "extent": _bbox_extent(target_bbox),
                    },
                    "target_anchor": _int_pair(proposal.get("target_anchor")),
                    "stride": _safe_int(proposal.get("stride"), 0),
                    "op_sequence": opcodes,
                    "opcode_semantics": semantics,
                    "scale_relation": str(proposal.get("scale_relation", "")),
                    "evidence": "visible_actor_target_pose_relation",
                }
            )
    return evidence


def _verified_transition_history(
    context: MechanicContext | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(context, MechanicContext):
        history = context.history
        memory = context.memory
        current_level = int(context.level)
    elif isinstance(context, Mapping):
        history = tuple(
            row for row in context.get("history", ()) if isinstance(row, Mapping)
        )
        memory = context.get("memory", {})
        current_level = _safe_int(context.get("level"), 0)
    else:
        history = ()
        memory = {}
        current_level = 0

    exact_count = 0
    score_closure_count = 0
    verified_levels: set[int] = set()
    schemas = memory.get("schemas", []) if isinstance(memory, Mapping) else []
    for schema in schemas if isinstance(schemas, Sequence) else []:
        if not isinstance(schema, Mapping):
            continue
        operator = schema.get("operator")
        operator_kind = (
            str(operator.get("kind", "")) if isinstance(operator, Mapping) else ""
        )
        mechanic_id = str(schema.get("mechanic_id", schema.get("mechanic_family", "")))
        if FAMILY_ID not in {mechanic_id, operator_kind}:
            continue
        item = schema.get("evidence")
        if not isinstance(item, Mapping):
            continue
        exact_count += max(0, _safe_int(item.get("exact_transition_count"), 0))
        score_closure_count += max(0, _safe_int(item.get("score_closure_count"), 0))
        for level in item.get("levels", []):
            value = _safe_int(level, -1)
            if value >= 0:
                verified_levels.add(value)

    observed_levels = {
        _safe_int(row.get("level"), -1) for row in history if isinstance(row, Mapping)
    }
    observed_levels.discard(-1)
    return {
        "exact_transition_count": int(exact_count),
        "score_closure_count": int(score_closure_count),
        "verified_levels": sorted(verified_levels),
        "observed_history_count": len(history),
        "observed_levels": sorted(observed_levels),
        "current_level": int(current_level),
        "verified": bool(exact_count > 0),
    }


def _preferred_candidate(
    arr: np.ndarray, binding: Mapping[str, Any]
) -> dict[str, Any] | None:
    active_color = int(binding.get("active_color", -1))
    for source, builder in _TRANSLATION_BUILDERS:
        output = builder(arr, active_color=active_color, max_plan_steps=96)
        proposals = [
            row for row in output.get("proposals", []) if isinstance(row, Mapping)
        ]
        if proposals:
            candidate = dict(proposals[0])
            candidate["mechanic_source"] = source
            return candidate
    output = build_program_row_semantics_polarity_fill_proposals(
        arr,
        known_program_colors=set(binding.get("binary_colors", [])),
        max_plan_steps=96,
    )
    proposals = [row for row in output.get("proposals", []) if isinstance(row, Mapping)]
    matching = [
        row for row in proposals if int(row.get("target_color", -1)) == active_color
    ]
    if not matching:
        return None
    candidate = dict(matching[0])
    candidate["mechanic_source"] = "polarity_fill"
    return candidate


def _deterministic_run_evidence(
    arr: np.ndarray,
    binding: Mapping[str, Any],
    run: Mapping[str, Any],
) -> dict[str, Any] | None:
    active_color = int(binding.get("active_color", -1))
    output = build_program_row_semantics_translation_bfs_proposals(
        arr,
        active_color=active_color,
        max_plan_steps=96,
    )
    run_x, run_y = [int(round(float(value))) for value in run["centroid"][:2]]
    for proposal in output.get("proposals", []):
        if not isinstance(proposal, Mapping):
            continue
        site = proposal.get("run_site")
        steps = [row for row in proposal.get("plan", []) if isinstance(row, Mapping)]
        if not _xy(site) or (int(site[0]), int(site[1])) != (run_x, run_y):
            continue
        if len(steps) != 1 or str(steps[0].get("role", "")) != "run":
            continue
        opcodes = [int(value) for value in proposal.get("op_sequence", [])]
        if not opcodes or any(
            opcode not in _VISUAL_OPCODE_DELTAS for opcode in opcodes
        ):
            continue
        start = proposal.get("start_centroid")
        target = proposal.get("target_centroid")
        stride = int(proposal.get("stride", 0))
        if not _xy(start) or not _xy(target) or stride <= 0:
            continue
        trace = [[round(float(start[0]), 3), round(float(start[1]), 3)]]
        x, y = float(start[0]), float(start[1])
        for opcode in opcodes:
            dx, dy = _VISUAL_OPCODE_DELTAS[opcode]
            x += float(dx * stride)
            y += float(dy * stride)
            trace.append([round(x, 3), round(y, 3)])
        distance = math.hypot(x - float(target[0]), y - float(target[1]))
        if distance > max(2.0, float(stride) * 0.75):
            continue
        return {
            "trace_certainty": "deterministic_visible_translation",
            "op_sequence": opcodes,
            "trace": trace,
            "expected_endpoint": trace[-1],
            "target_centroid": [round(float(target[0]), 3), round(float(target[1]), 3)],
            "target_distance": round(distance, 3),
            "target_closure": True,
        }
    for source, builder in _TRANSFORM_BUILDERS:
        output = builder(arr, active_color=active_color, max_plan_steps=96)
        for proposal in output.get("proposals", []):
            if not isinstance(proposal, Mapping):
                continue
            site = proposal.get("run_site")
            steps = [
                row for row in proposal.get("plan", []) if isinstance(row, Mapping)
            ]
            if not _xy(site) or (int(site[0]), int(site[1])) != (run_x, run_y):
                continue
            if len(steps) != 1 or str(steps[0].get("role", "")) != "run":
                continue
            evidence = _deterministic_transform_evidence(
                binding,
                proposal,
                source=source,
            )
            if evidence is not None:
                return evidence
    return _visible_bit_trace_evidence(binding)


def _deterministic_transform_evidence(
    binding: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any] | None:
    opcodes = _int_sequence(proposal.get("op_sequence"))
    desired = _int_sequence(proposal.get("desired_opcodes"))
    decoded = _decoded_panel_opcodes(binding, proposal)
    start_bbox = _int_bbox(proposal.get("start_bbox"))
    target_bbox = _int_bbox(proposal.get("target_bbox"))
    stride = _safe_int(proposal.get("stride"), 0)
    if (
        not opcodes
        or not desired
        or decoded != desired
        or start_bbox is None
        or target_bbox is None
        or stride <= 0
    ):
        return None

    translation_opcodes = _int_sequence(proposal.get("translation_op_sequence"))
    if not translation_opcodes or any(
        opcode not in _VISUAL_OPCODE_DELTAS for opcode in translation_opcodes
    ):
        return None
    scale_opcode = _safe_int(proposal.get("scale_opcode"), -1)
    rotation_opcode = _safe_int(proposal.get("rotation_opcode"), -1)
    shape_opcode = _safe_int(proposal.get("shape_opcode"), -1)
    admitted = set(translation_opcodes)
    admitted.update(
        opcode
        for opcode in (scale_opcode, rotation_opcode, shape_opcode)
        if opcode >= 0
    )
    if any(opcode not in admitted for opcode in opcodes):
        return None

    x, y = int(start_bbox[0]), int(start_bbox[1])
    trace: list[dict[str, Any]] = [
        {"anchor": [x, y], "effect": "start_visible_actor_pose"}
    ]
    for opcode in opcodes:
        delta = _VISUAL_OPCODE_DELTAS.get(opcode)
        if delta is not None:
            x += int(delta[0]) * stride
            y += int(delta[1]) * stride
            effect = "translation"
        elif opcode == scale_opcode:
            effect = str(proposal.get("scale_relation", "scale_change"))
        elif opcode == rotation_opcode:
            effect = "visible_quarter_turn"
        elif opcode == shape_opcode:
            effect = "visible_shape_role_change"
        else:
            return None
        trace.append(
            {"opcode": int(opcode), "anchor": [int(x), int(y)], "effect": effect}
        )

    if source == "scale_translation_bfs":
        if scale_opcode < 0 or not opcodes or opcodes[0] != scale_opcode:
            return None
        relation = str(proposal.get("scale_relation", ""))
        start_area = _bbox_area(start_bbox)
        target_area = _bbox_area(target_bbox)
        if relation == "larger_component_to_smaller_component":
            scale_relation_closes = start_area > target_area
        elif relation == "smaller_component_to_larger_component":
            scale_relation_closes = start_area < target_area
        else:
            scale_relation_closes = False
        wanted = [int(target_bbox[0]), int(target_bbox[1])]
        certainty = "deterministic_visible_scale_translation"
        pose_relation = "position_and_scale_match_visible_target"
    elif source == "shape_transform_bfs":
        target_anchor = _int_pair(proposal.get("target_anchor"))
        if (
            rotation_opcode < 0
            or shape_opcode < 0
            or len(opcodes) < 3
            or opcodes[0] != rotation_opcode
            or opcodes[1] != shape_opcode
            or target_anchor is None
            or not _point_near_bbox(target_anchor, target_bbox, margin=stride)
        ):
            return None
        scale_relation_closes = _bbox_area(target_bbox) > _bbox_area(start_bbox)
        if scale_opcode >= 0:
            scale_relation_closes = bool(
                scale_relation_closes and opcodes[-1] == scale_opcode
            )
        wanted = target_anchor
        certainty = "deterministic_visible_shape_transform"
        pose_relation = "position_rotation_shape_and_scale_match_visible_target"
    else:
        return None

    distance = math.hypot(float(x - wanted[0]), float(y - wanted[1]))
    target_closure = bool(
        scale_relation_closes and distance <= max(2.0, float(stride) * 0.75)
    )
    if not target_closure:
        return None
    return {
        "trace_certainty": certainty,
        "op_sequence": opcodes,
        "decoded_program": decoded,
        "trace": trace,
        "expected_endpoint": [int(x), int(y)],
        "target_anchor": [int(wanted[0]), int(wanted[1])],
        "target_bbox": target_bbox,
        "target_distance": round(float(distance), 3),
        "target_pose_relation": pose_relation,
        "target_closure": True,
        "opcode_model_evidence": {
            "current_visible_pose_relation": True,
            "verified_prior_transition_count": _safe_int(
                binding.get("verified_transition_history", {}).get(
                    "exact_transition_count", 0
                )
                if isinstance(binding.get("verified_transition_history"), Mapping)
                else 0,
                0,
            ),
        },
    }


def _decoded_panel_opcodes(
    binding: Mapping[str, Any], proposal: Mapping[str, Any]
) -> list[int]:
    columns = [float(value) for value in proposal.get("panel_columns", [])]
    bit_centers = [float(value) for value in proposal.get("bit_centers", [])]
    bits = [row for row in binding.get("visible_bits", []) if isinstance(row, Mapping)]
    active_color = _safe_int(binding.get("active_color"), -1)
    if not columns or not bit_centers or active_color < 0:
        return []
    decoded: list[int] = []
    for column in columns:
        opcode = 0
        used: set[tuple[int, ...]] = set()
        for bit_index, bit_center in enumerate(bit_centers):
            candidates: list[tuple[float, Mapping[str, Any]]] = []
            for bit in bits:
                centroid = bit.get("centroid")
                bbox = _int_bbox(bit.get("bbox"))
                if not _xy(centroid) or bbox is None or tuple(bbox) in used:
                    continue
                distance = abs(float(centroid[0]) - column) + abs(
                    float(centroid[1]) - bit_center
                )
                candidates.append((distance, bit))
            if not candidates:
                return []
            distance, selected = min(candidates, key=lambda item: item[0])
            if distance > 3.5:
                return []
            selected_bbox = _int_bbox(selected.get("bbox"))
            if selected_bbox is None:
                return []
            used.add(tuple(selected_bbox))
            if _safe_int(selected.get("color"), -1) == active_color:
                opcode |= 1 << bit_index
        decoded.append(int(opcode))
    return decoded


def _visible_bit_trace_evidence(
    binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    active_color = int(binding.get("active_color", -1))
    opcodes: list[int] = []
    for row in binding.get("visible_rows", []):
        if not isinstance(row, Mapping):
            return None
        bits = [bit for bit in row.get("bits", []) if isinstance(bit, Mapping)]
        if not bits:
            return None
        indices = [int(bit.get("bit_index", -1)) for bit in bits]
        if min(indices, default=-1) < 0 or len(indices) != len(set(indices)):
            return None
        opcode = sum(
            1 << int(bit["bit_index"])
            for bit in bits
            if int(bit.get("color", -1)) == active_color
        )
        if opcode not in _VISUAL_OPCODE_DELTAS:
            return None
        opcodes.append(int(opcode))
    if not opcodes:
        return None

    candidates: list[tuple[float, list[list[float]], list[float]]] = []
    for pair in binding.get("actor_target_tuples", []):
        if not isinstance(pair, Mapping):
            continue
        actor = pair.get("actor")
        target = pair.get("target")
        if not isinstance(actor, Mapping) or not isinstance(target, Mapping):
            continue
        actor_center = actor.get("centroid")
        target_center = target.get("centroid")
        stride = int(pair.get("stride", 0) or 0)
        if not _xy(actor_center) or not _xy(target_center) or stride <= 0:
            continue
        endpoints = (
            (actor_center, target_center),
            (target_center, actor_center),
        )
        for start, wanted in endpoints:
            x, y = float(start[0]), float(start[1])
            trace = [[round(x, 3), round(y, 3)]]
            for opcode in opcodes:
                dx, dy = _VISUAL_OPCODE_DELTAS[opcode]
                x += float(dx * stride)
                y += float(dy * stride)
                trace.append([round(x, 3), round(y, 3)])
            distance = math.hypot(x - float(wanted[0]), y - float(wanted[1]))
            if distance <= max(2.0, float(stride) * 0.75):
                candidates.append(
                    (
                        distance,
                        trace,
                        [round(float(wanted[0]), 3), round(float(wanted[1]), 3)],
                    )
                )
    if not candidates:
        return None
    distance, trace, target_centroid = min(candidates, key=lambda item: item[0])
    return {
        "trace_certainty": "deterministic_visible_bit_trace",
        "op_sequence": opcodes,
        "trace": trace,
        "expected_endpoint": trace[-1],
        "target_centroid": target_centroid,
        "target_distance": round(float(distance), 3),
        "target_closure": True,
    }


def _verify_one_bit(
    expected: Mapping[str, Any],
    before: np.ndarray,
    current: np.ndarray,
    action: Mapping[str, Any],
) -> dict[str, Any]:
    bit = expected.get("bit")
    if not isinstance(bit, Mapping) or not _bbox(bit.get("bbox")):
        return _verification(False, "invalid_bit_prediction")
    bbox = [int(value) for value in bit["bbox"][:4]]
    x0, y0, x1, y1 = bbox
    source = int(bit.get("source_color", -1))
    target = int(bit.get("target_color", -1))
    if not np.all(before[y0 : y1 + 1, x0 : x1 + 1] == source):
        return _verification(False, "source_bit_pixels_do_not_match")
    if not np.all(current[y0 : y1 + 1, x0 : x1 + 1] == target):
        return _verification(False, "target_bit_pixels_do_not_match")

    changed = before != current
    allowed = np.zeros_like(changed, dtype=bool)
    allowed[y0 : y1 + 1, x0 : x1 + 1] = True
    top_limit = max(2, int(round(before.shape[0] * 0.06)))
    progress = np.zeros_like(changed, dtype=bool)
    progress[: top_limit + 1, :] = True
    outside = changed & ~(allowed | progress)
    progress_count = int(np.count_nonzero(changed & progress))
    if np.any(outside):
        return _verification(False, "unmodelled_frame_delta")
    if progress_count > 1:
        return _verification(False, "nonlocal_progress_delta")

    packet = build_program_row_semantics_packet(
        before,
        current,
        action_name="ACTION6",
        action_payload={"x": int(action["col"]), "y": int(action["row"])},
    )
    selected = packet.get("selected_program_cell")
    if not bool(packet.get("program_bit_toggle_candidate")) or not isinstance(
        selected, Mapping
    ):
        return _verification(False, "sensor_did_not_confirm_one_bit_toggle")
    if int(packet.get("program_cell_changed_cell_count", 0)) != int(
        bit.get("pixel_count", 0)
    ):
        return _verification(False, "bit_delta_size_mismatch")
    if int(packet.get("result_region_changed_cell_count", 0)) != 0:
        return _verification(False, "result_region_changed_during_bit_toggle")
    return _verification(
        True,
        "exact_one_bit_toggle",
        kind="one_bit_toggle",
        changed_pixel_count=int(np.count_nonzero(changed)),
    )


def _verify_deterministic_run(
    expected: Mapping[str, Any],
    before: np.ndarray,
    current: np.ndarray,
    action: Mapping[str, Any],
) -> dict[str, Any]:
    if str(expected.get("trace_certainty", "")) not in _DETERMINISTIC_RUN_CERTAINTIES:
        return _verification(False, "run_trace_not_determined")
    if not bool(expected.get("target_closure")):
        return _verification(False, "run_trace_does_not_close_target")
    packet = build_program_row_semantics_packet(
        before,
        current,
        action_name="ACTION6",
        action_payload={"x": int(action["col"]), "y": int(action["row"])},
    )
    if not bool(packet.get("run_button_candidate")):
        return _verification(False, "sensor_did_not_confirm_run_control")
    if int(packet.get("result_region_changed_cell_count", 0)) <= 0:
        return _verification(False, "run_has_no_visible_result_delta")
    current_binding = bind_current(current)
    if not _program_structure_bound(current_binding):
        return _verification(False, "post_run_binding_missing")
    if _geometry_signature(current_binding) == expected.get("before_geometry"):
        return _verification(False, "run_did_not_advance_visible_program_geometry")
    return _verification(
        True,
        "deterministic_run_target_closure",
        kind="deterministic_run",
        trace=list(expected.get("trace", [])),
    )


def _geometry_signature(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rows": [
            round(float(row.get("center", 0.0)), 3)
            for row in binding.get("visible_rows", [])
        ],
        "bits": sorted(
            {
                round(float(row.get("bit_center", 0.0)), 3)
                for row in binding.get("visible_bits", [])
            }
        ),
        "runs": [
            [round(float(value), 3) for value in row.get("centroid", [])[:2]]
            for row in binding.get("run_controls", [])
        ],
        "actor_targets": [
            {
                "actor": [
                    round(float(value), 3)
                    for value in row.get("actor", {}).get("centroid", [])[:2]
                ],
                "target": [
                    round(float(value), 3)
                    for value in row.get("target", {}).get("centroid", [])[:2]
                ],
                "stride": _safe_int(row.get("stride"), 0),
            }
            for row in binding.get("actor_target_tuples", [])
            if isinstance(row, Mapping)
            and isinstance(row.get("actor"), Mapping)
            and isinstance(row.get("target"), Mapping)
        ],
        "transform_targets": [
            {
                "kind": str(row.get("kind", "")),
                "actor_bbox": _int_bbox(row.get("actor", {}).get("bbox")) or [],
                "target_bbox": _int_bbox(row.get("target", {}).get("bbox")) or [],
                "op_sequence": _int_sequence(row.get("op_sequence")),
            }
            for row in binding.get("transform_target_tuples", [])
            if isinstance(row, Mapping)
            and isinstance(row.get("actor"), Mapping)
            and isinstance(row.get("target"), Mapping)
        ],
    }


def _program_structure_bound(binding: Mapping[str, Any]) -> bool:
    return bool(
        len(binding.get("visible_rows", [])) >= 2
        and len(binding.get("visible_bits", [])) >= 4
        and binding.get("run_controls")
    )


def _selected_bit(bits: Any, *, x: int, y: int) -> dict[str, Any] | None:
    for row in bits if isinstance(bits, Sequence) else []:
        if not isinstance(row, Mapping) or not _bbox(row.get("bbox")):
            continue
        x0, y0, x1, y1 = [int(value) for value in row["bbox"][:4]]
        if x0 <= x <= x1 and y0 <= y <= y1:
            return dict(row)
    return None


def _selected_run(runs: Any, *, x: int, y: int) -> dict[str, Any] | None:
    for row in runs if isinstance(runs, Sequence) else []:
        if not isinstance(row, Mapping) or not _bbox(row.get("bbox")):
            continue
        x0, y0, x1, y1 = [int(value) for value in row["bbox"][:4]]
        if x0 <= x <= x1 and y0 <= y <= y1:
            return dict(row)
    return None


def _mouse_action(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    name = str(value.get("action", "")).strip().upper()
    try:
        if name == "MOUSE":
            return {
                "action": "MOUSE",
                "row": int(value["row"]),
                "col": int(value["col"]),
            }
        if name == "ACTION6":
            return {"action": "MOUSE", "row": int(value["y"]), "col": int(value["x"])}
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _mouse_allowed(valid_actions: Any) -> bool:
    if valid_actions is None:
        return True
    if not isinstance(valid_actions, Sequence) or isinstance(
        valid_actions, (str, bytes, bytearray)
    ):
        return False
    names: set[str] = set()
    for value in valid_actions:
        if isinstance(value, Mapping):
            names.add(str(value.get("action", "")).strip().upper())
        else:
            names.add(str(getattr(value, "name", value)).strip().upper())
    return bool({"MOUSE", "ACTION6"} & names)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _int_sequence(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


def _int_bbox(value: Any) -> list[int] | None:
    if not _bbox(value):
        return None
    try:
        result = [int(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if result[2] < result[0] or result[3] < result[1]:
        return None
    return result


def _int_pair(value: Any) -> list[int] | None:
    if not _xy(value):
        return None
    try:
        return [int(round(float(value[0]))), int(round(float(value[1])))]
    except (TypeError, ValueError):
        return None


def _bbox_center(bbox: Sequence[int]) -> list[float]:
    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def _bbox_extent(bbox: Sequence[int]) -> list[int]:
    return [int(bbox[2] - bbox[0] + 1), int(bbox[3] - bbox[1] + 1)]


def _bbox_area(bbox: Sequence[int]) -> int:
    width, height = _bbox_extent(bbox)
    return max(0, int(width * height))


def _point_near_bbox(point: Sequence[int], bbox: Sequence[int], *, margin: int) -> bool:
    return bool(
        int(bbox[0]) - margin <= int(point[0]) <= int(bbox[2]) + margin
        and int(bbox[1]) - margin <= int(point[1]) <= int(bbox[3]) + margin
    )


def _nearest_index(values: Sequence[float], target: float) -> int | None:
    if not values:
        return None
    return min(range(len(values)), key=lambda index: abs(float(values[index]) - target))


def _xy(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) >= 2
    )


def _bbox(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) >= 4
    )


def _blocked_plan(binding: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return _json(
        {
            "schema_version": "tufa_bit_program_plan.v1",
            "family_id": FAMILY_ID,
            "status": "blocked",
            "reason": reason,
            "binding_digest": binding.get("binding_digest", ""),
            "actions": [],
            "action_count": 0,
            "suffix_exposed": False,
        }
    )


def _unsupported_prediction(
    binding: Mapping[str, Any], action: Mapping[str, Any] | None, reason: str
) -> dict[str, Any]:
    return _json(
        {
            "schema_version": _PREDICTION_VERSION,
            "family_id": FAMILY_ID,
            "supported": False,
            "reason": reason,
            "action": action,
            "binding_digest": binding.get("binding_digest", ""),
        }
    )


def _verification(accepted: bool, reason: str, **details: Any) -> dict[str, Any]:
    return _json(
        {
            "schema_version": _VERIFICATION_VERSION,
            "family_id": FAMILY_ID,
            "accepted": bool(accepted),
            "reason": reason,
            **details,
        }
    )


def _json(value: Mapping[str, Any]) -> dict[str, Any]:
    result = json_safe_mapping(value)
    if result is None:
        raise ValueError("program mechanic produced non-JSON-safe data")
    return result


__all__ = [
    "FAMILY_ID",
    "bind_current",
    "compile_schema",
    "predict_transition",
    "propose_plan",
    "verify_transition",
]
