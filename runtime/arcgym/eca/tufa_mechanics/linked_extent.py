"""Frame-grounded linked extent-growth mechanic.

The module exposes a small JSON-safe boundary around the generic slider
planner.  Roles are rebound from each frame; observed palette values are used
only transiently to associate controls and pieces and never become rules.
"""

from __future__ import annotations

from collections import Counter
import heapq
from itertools import permutations
import math
from typing import Any, Mapping

import numpy as np

from arcgym.eca import generic_slider_growth_bfs as _slider
from arcgym.eca.component_extent_growth_sensor import (
    build_component_extent_growth_packet,
)
from arcgym.eca.tufa_mechanics.base import MechanicContext, stable_digest


FAMILY_ID = "linked_extent_growth"
_SCHEMA_VERSION = "tufa_linked_extent_growth.v2"


def compile_schema(
    frame: Any,
    level: Any,
    context: MechanicContext | None = None,
) -> dict[str, Any]:
    """Compile a relational mechanic schema and bind it to ``frame``."""

    binding = bind_current(frame, context=context)
    inheritance = _inherited_schema(context)
    return {
        "schema_version": _SCHEMA_VERSION,
        "family_id": FAMILY_ID,
        "level": _json_scalar(level),
        "accepted": bool(binding.get("accepted")),
        "roles": [
            "growable_piece",
            "slider",
            "child_link",
            "co_motion_link",
            "carried_target",
            "indicator",
        ],
        "required_bindings": [
            "growable_piece",
            "slider",
            "carried_target",
            "indicator",
        ],
        "state_variables": [
            "piece_extents",
            "parent_child_graph",
            "target_positions",
            "slider_modes",
        ],
        "operator": {
            "name": "change_extent_and_translate_dependants",
            "action": "ACTION6",
            "extent_units": 1,
            "effects": [
                "selected piece grows or shrinks one axial extent unit",
                "linked descendants translate by that exact vector",
                "carried targets translate with their bound piece",
            ],
        },
        "goal_relation": (
            "the rebound residual reaches zero; carried-target distance is an "
            "advisory local measure because chain clearance may increase it"
        ),
        "invariants": [
            "linked descendants preserve their inferred co-motion relation",
            "extent changes use one frame-inferred axial unit",
            "all visible piece collisions are jointly constrained",
        ],
        "falsifiers": [
            "an unlinked component moves",
            "the rebound successor violates the predicted exact transition",
            "neither exact transition nor monotonic rebound progress verifies",
        ],
        "verification": {
            "extent": "one inferred axial unit",
            "links": "exact from/to bboxes for every linked entity",
            "progress": (
                "decreasing frame-rebound residual or an exact clearance "
                "transition"
            ),
            "privacy": "only a digest of the private successor suffix is retained",
        },
        "inherited_schema": inheritance,
        "binding": binding,
    }


def bind_current(
    frame: Any,
    context: MechanicContext | None = None,
) -> dict[str, Any]:
    """Bind pieces, controls, links, targets, and indicators from a frame."""

    perceived = _perceive(frame)
    if not perceived.get("accepted"):
        return {
            "family_id": FAMILY_ID,
            "accepted": False,
            "reject_reason": perceived.get("reject_reason", "unbound_frame"),
        }
    binding = _public_binding(perceived)
    binding["inherited_schema"] = _inherited_schema(context)
    return binding


def propose_plan(
    frame: Any,
    valid_actions: Any = None,
    context: MechanicContext | None = None,
) -> dict[str, Any]:
    """Return one rebound action and keep the full successor suffix private."""

    if valid_actions is not None and "ACTION6" not in _action_names(valid_actions):
        return _blocked_plan("action6_not_available", context=context)
    if isinstance(context, MechanicContext) and int(context.level) > 3:
        return _blocked_plan(
            "level_exceeds_verified_l1_l3_scope",
            context=context,
        )
    plan, diagnostics = _plan_linked_extent_actions(frame)
    if not diagnostics.get("accepted"):
        return _blocked_plan(
            str(diagnostics.get("reject_reason", "no_linked_extent_plan")),
            context=context,
        )
    actions = _canonical_plan(plan)
    if not actions:
        return _blocked_plan("no_action6_growth_plan", context=context)

    prediction = _predict_from_plan(
        frame,
        actions[0],
        actions,
        diagnostics,
        context=context,
    )
    binding = bind_current(frame, context=context)
    scope_reason = _certification_scope_reason(binding, actions[0], context)
    certified = bool(prediction.get("accepted") and not scope_reason)
    inherited = _inherited_schema(context)
    return {
        "schema_version": "tufa_linked_extent_plan.v3",
        "family_id": FAMILY_ID,
        "status": "proposed",
        "actions": [actions[0]],
        "action_count": 1,
        "certified": certified,
        "certificate_reason": (
            "frame-rebound exact transition with joint residual certificate"
            if certified
            else scope_reason
            or str(prediction.get("reject_reason", "transition_not_certified"))
        ),
        "transition_rule": "change_extent_and_translate_dependants",
        "progress_measure": {
            "metric": "remaining_linked_extent_actions",
            "before": len(actions),
            "expected_after": max(0, len(actions) - 1),
            "strict": True,
        },
        "suffix_exposed": False,
        "requires_transition_verification": True,
        "first_prediction": prediction,
        "detail": {
            "private_plan_length": len(actions),
            "planner": str(diagnostics.get("solver") or "linked_extent_growth_bfs"),
            "joint_state": bool(diagnostics.get("joint_state_search")),
            "mode_counts": dict(diagnostics.get("mode_counts") or {}),
            "inherited_schema": inherited,
            "transfer_authority": (
                "verified_prior_schema_plus_current_rebinding"
                if inherited.get("verified")
                else "current_level_exact_rebinding"
            ),
        },
    }


def predict_transition(
    frame: Any,
    action: Any,
    context: MechanicContext | None = None,
) -> dict[str, Any]:
    """Predict one exact growth transition from the current binding."""

    plan, diagnostics = _plan_linked_extent_actions(frame)
    actions = _canonical_plan(plan) if diagnostics.get("accepted") else []
    return _predict_from_plan(
        frame,
        action,
        actions,
        diagnostics,
        context=context,
    )


def _predict_from_plan(
    frame: Any,
    action: Any,
    plan: list[dict[str, Any]],
    diagnostics: Mapping[str, Any],
    *,
    context: MechanicContext | None,
) -> dict[str, Any]:
    """Predict one transition against a freshly rebound private plan."""

    perceived = _perceive(frame)
    payload = _action_payload(action)
    if not perceived.get("accepted"):
        return _prediction_rejection(
            payload,
            perceived.get("reject_reason", "unbound_frame"),
        )
    if payload.get("action") != "ACTION6":
        return _prediction_rejection(payload, "unsupported_action")
    if not diagnostics.get("accepted") or not plan:
        return _prediction_rejection(payload, "no_frame_rebound_plan")
    if payload != _action_payload(plan[0]):
        return _prediction_rejection(payload, "action_is_not_rebound_plan_prefix")

    slider_idx, mode = _selected_slider(perceived["sliders"], payload)
    if slider_idx is None:
        return _prediction_rejection(payload, "action_site_not_bound_to_slider")
    slider = perceived["sliders"][slider_idx]
    selected_piece_indices = {
        int(piece_idx) for piece_idx in slider.get("piece_indices", [])
    }
    if not selected_piece_indices and slider.get("piece_idx") is not None:
        selected_piece_indices.add(int(slider["piece_idx"]))
    if not selected_piece_indices:
        return _prediction_rejection(payload, "slider_has_no_bound_piece")

    unit = int(perceived["growth_unit"])
    sign = 1 if mode == "grow" else -1
    entities = list(perceived.get("entities") or perceived["pieces"])
    controlled_count = int(
        perceived.get("controlled_piece_count", len(perceived["pieces"]))
    )
    ancestors = _entity_ancestors(perceived["parent_to_child"], len(entities))

    def selected_translation(entity_idx: int) -> tuple[int, int]:
        dx = 0
        dy = 0
        for ancestor_idx in ancestors[entity_idx]:
            if ancestor_idx not in selected_piece_indices:
                continue
            direction = _growth_direction(
                int(entities[ancestor_idx]["rotation"])
            )
            dx += sign * direction[0] * unit
            dy += sign * direction[1] * unit
        return dx, dy

    extent_rows: list[dict[str, Any]] = []
    linked_pieces: list[dict[str, Any]] = []
    changed_indices: set[int] = set()
    for entity_idx, entity in enumerate(entities):
        translation = selected_translation(entity_idx)
        # Passive axial bodies constrain the joint search but are not
        # controlled dependants of a slider transition.
        if entity_idx >= controlled_count:
            continue
        if entity_idx in selected_piece_indices:
            resized = _resized_bbox(entity, unit, mode)
            after_bbox = _translate_bbox(resized, translation)
            direction = _growth_direction(int(entity["rotation"]))
            extent_rows.append(
                {
                    "entity": _entity_id(entity_idx, controlled_count),
                    "from_bbox": _bbox(entity),
                    "to_bbox": after_bbox,
                    "axis": "horizontal" if direction[0] else "vertical",
                    "side": _growth_side(direction),
                    "delta": [
                        int(sign * direction[0] * unit),
                        int(sign * direction[1] * unit),
                    ],
                    "rotation": int(entity["rotation"]),
                    "mode": mode,
                }
            )
            changed_indices.add(entity_idx)
            continue
        if translation == (0, 0):
            continue
        changed_indices.add(entity_idx)
        linked_pieces.append(
            {
                "entity": _entity_id(entity_idx, controlled_count),
                "from_bbox": _bbox(entity),
                "to_bbox": _translate_bbox(_bbox(entity), translation),
                "delta": [int(translation[0]), int(translation[1])],
                "rotation": (
                    None
                    if entity.get("rotation") is None
                    else int(entity["rotation"])
                ),
            }
        )

    if not extent_rows:
        return _prediction_rejection(payload, "missing_extent_change")

    # A passive axial body that is flush with a moving extent edge or a rigidly
    # translated dependant is a follower, not an unaffected obstacle. This relation is
    # inferred from current geometry on every frame; it does not depend on a
    # game id, palette value, or remembered coordinate.  The exact verifier
    # still requires the predicted rigid translation after the action.
    follower_rows, follower_indices = _contact_follower_rows(
        entities,
        controlled_count=controlled_count,
        extent_rows=extent_rows,
        linked_rows=linked_pieces,
        unit=unit,
    )
    linked_pieces.extend(follower_rows)
    changed_indices.update(follower_indices)

    linked_targets: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    for target_idx, target in enumerate(perceived["targets"]):
        parent_idx = int(target["parent_idx"])
        target_dx = 0
        target_dy = 0
        for moving_idx in [*ancestors[parent_idx], parent_idx]:
            if moving_idx not in selected_piece_indices:
                continue
            direction = _growth_direction(int(entities[moving_idx]["rotation"]))
            target_dx += sign * direction[0] * unit
            target_dy += sign * direction[1] * unit
        target_delta = (target_dx, target_dy)
        if target_delta == (0, 0):
            continue
        indicator_idx = target.get("indicator_idx")
        if indicator_idx is None:
            continue
        indicator = perceived["indicators"][int(indicator_idx)]
        before_center = _bbox_center(_bbox(target))
        after_bbox = _translate_bbox(_bbox(target), target_delta)
        after_center = _bbox_center(after_bbox)
        indicator_center = _bbox_center(_bbox(indicator))
        before_distance = _distance(before_center, indicator_center)
        after_distance = _distance(after_center, indicator_center)
        linked_targets.append(
            {
                "entity": f"target_{target_idx}",
                "parent": f"piece_{int(target['parent_idx'])}",
                "indicator": f"indicator_{int(indicator_idx)}",
                "from_bbox": _bbox(target),
                "to_bbox": after_bbox,
                "from_visible_bbox": list(target["visible_bbox"]),
                "to_visible_bbox": _translate_bbox(
                    target["visible_bbox"], target_delta
                ),
                "delta": [int(target_delta[0]), int(target_delta[1])],
            }
        )
        progress_rows.append(
            {
                "target": f"target_{target_idx}",
                "indicator": f"indicator_{int(indicator_idx)}",
                "distance_before": round(before_distance, 6),
                "distance_after": round(after_distance, 6),
                "decreases": bool(after_distance < before_distance),
            }
        )

    unaffected: list[dict[str, Any]] = []
    for other_idx, other in enumerate(entities):
        if other_idx not in changed_indices:
            unaffected.append(
                {
                    "entity": _entity_id(other_idx, controlled_count),
                    "bbox": _bbox(other),
                    "rotation": (
                        None
                        if other.get("rotation") is None
                        else int(other["rotation"])
                    ),
                }
            )

    arr = _as_frame(frame)
    return {
        "schema_version": f"{_SCHEMA_VERSION}.transition",
        "family_id": FAMILY_ID,
        "accepted": True,
        "action": payload,
        "frame_shape": [int(arr.shape[0]), int(arr.shape[1])],
        "inferred_growth_unit": unit,
        "extent_delta": extent_rows[0],
        "extent_deltas": extent_rows,
        "linked_piece_translations": linked_pieces,
        "linked_target_translations": linked_targets,
        "unaffected_pieces": unaffected,
        "target_distance_evidence": progress_rows,
        "abstract_progress": {
            "metric": "remaining_linked_extent_actions",
            "before": len(plan),
            "expected_after": max(0, len(plan) - 1),
            "expected_successor_digest": stable_digest(plan[1:]),
            "current_plan_digest": stable_digest(plan),
            "distance_trend": (
                "decreasing"
                if progress_rows and all(row["decreases"] for row in progress_rows)
                else "not_applicable"
                if not progress_rows
                else "temporary_clearance_or_alignment"
            ),
        },
        "inherited_schema": _inherited_schema(context),
        "schema_repairs": [
            {
                "kind": "moving_edge_follower",
                "entity": str(row["entity"]),
                "relation": "flush_contact_with_moving_extent_edge",
                "delta": list(row["delta"]),
            }
            for row in follower_rows
        ],
        "before_frame": arr.astype(int, copy=False).tolist(),
    }


def verify_transition(expected: Any, current_frame: Any) -> dict[str, Any]:
    """Verify a predicted transition against the rebound abstract successor."""

    reasons: list[str] = []
    if not isinstance(expected, Mapping) or not bool(expected.get("accepted")):
        return _verification(False, ["prediction_not_accepted"])
    if expected.get("family_id") != FAMILY_ID:
        return _verification(False, ["family_mismatch"])

    current = _perceive(current_frame)
    if not current.get("accepted"):
        return _verification(
            False,
            [f"current_binding:{current.get('reject_reason', 'failed')}"]
        )
    arr = _as_frame(current_frame)
    if list(expected.get("frame_shape") or []) != [int(arr.shape[0]), int(arr.shape[1])]:
        reasons.append("frame_shape_changed")

    entities = list(current.get("entities") or current["pieces"])
    extent_rows = list(expected.get("extent_deltas") or [])
    if not extent_rows and expected.get("extent_delta"):
        extent_rows = [expected["extent_delta"]]
    for row in extent_rows:
        if not isinstance(row, Mapping) or not _has_entity(
            entities, row.get("to_bbox"), row.get("rotation")
        ):
            reasons.append(f"extent_bbox_mismatch:{_entity_name(row)}")

    linked_piece_rows = list(expected.get("linked_piece_translations") or [])
    for row in linked_piece_rows:
        if not isinstance(row, Mapping) or not _has_entity(
            entities, row.get("to_bbox"), row.get("rotation")
        ):
            reasons.append(f"linked_piece_mismatch:{_entity_name(row)}")

    for row in list(expected.get("unaffected_pieces") or []):
        if not isinstance(row, Mapping) or not _has_entity(
            entities, row.get("bbox"), row.get("rotation")
        ):
            reasons.append(f"unaffected_piece_changed:{_entity_name(row)}")

    targets = current["targets"]
    linked_target_rows = list(expected.get("linked_target_translations") or [])
    for row in linked_target_rows:
        if not isinstance(row, Mapping) or not _has_bbox(targets, row.get("to_bbox")):
            reasons.append(f"linked_target_mismatch:{_entity_name(row)}")

    before_frame = expected.get("before_frame")
    packet: dict[str, Any] = {}
    if before_frame is None:
        reasons.append("missing_before_frame")
    else:
        extent_mode = str(
            (extent_rows[0] if extent_rows else {}).get("mode", "grow")
        )
        packet_before = before_frame if extent_mode == "grow" else arr
        packet_after = arr if extent_mode == "grow" else before_frame
        packet = build_component_extent_growth_packet(
            packet_before,
            packet_after,
            action_name="ACTION6",
            action_payload=dict(expected.get("action") or {}),
        )
        for row in extent_rows:
            if isinstance(row, Mapping) and not _has_exact_extent_observation(
                packet, expected, extent=row
            ):
                reasons.append(
                    f"missing_exact_unit_extent_evidence:{_entity_name(row)}"
                )
        for row in linked_target_rows:
            if isinstance(row, Mapping) and not _has_exact_target_translation(
                packet,
                row,
                reverse=extent_mode == "shrink",
            ):
                reasons.append(f"missing_exact_target_translation:{_entity_name(row)}")
        if (
            linked_target_rows
            and all(
                bool(row.get("decreases"))
                for row in expected.get("target_distance_evidence") or []
                if isinstance(row, Mapping)
            )
            and not _has_decreasing_distance_evidence(packet, expected)
        ):
            reasons.append("missing_decreasing_target_distance_evidence")

    evidence = {
        "growth_observation_count": len(packet.get("growth_observations") or []),
        "attached_translation_count": len(
            packet.get("attached_translation_observations") or []
        ),
        "target_distance_observation_count": len(
            packet.get("target_distance_observations") or []
        ),
        "linked_piece_count": len(linked_piece_rows),
        "linked_target_count": len(linked_target_rows),
        "extent_change_count": len(extent_rows),
    }
    abstract = _verify_abstract_successor(expected, current_frame)
    evidence.update(abstract.get("evidence") or {})
    visual_exact = not reasons
    evidence["visual_exact"] = visual_exact
    evidence["visual_diagnostics"] = list(reasons)
    evidence["schema_repair_count"] = len(expected.get("schema_repairs") or [])
    evidence["schema_repair_applied"] = bool(
        visual_exact and expected.get("schema_repairs")
    )
    if visual_exact:
        evidence["certificate_basis"] = (
            "exact_transition_and_monotonic_progress"
            if abstract.get("accepted")
            else "exact_transition"
        )
        return _verification(True, [], evidence)
    goal_overlap_rebound = _verified_goal_overlap_rebound(
        expected,
        reasons,
        abstract,
    )
    evidence["goal_overlap_rebound"] = goal_overlap_rebound
    if goal_overlap_rebound:
        evidence["certificate_basis"] = "goal_overlap_and_exact_rebound"
        return _verification(True, [], evidence)

    structural_rebound = _verified_structural_rebound(
        expected,
        reasons,
        abstract,
    )
    evidence["structural_rebound"] = structural_rebound
    if structural_rebound:
        evidence["certificate_basis"] = "structural_transition_and_exact_rebound"
        return _verification(True, [], evidence)

    abstract_reason = str(abstract.get("reject_reason", "") or "")
    reasons.append(abstract_reason or "visual_transition_not_verified")
    return _verification(False, list(dict.fromkeys(reasons)), evidence)


def _perceive(frame: Any) -> dict[str, Any]:
    arr = _as_frame(frame)
    roles = _slider._infer_roles(arr)
    if roles is None:
        return {"accepted": False, "reject_reason": "missing_slider_roles"}
    pieces = [dict(piece) for piece in _slider._detect_pieces(arr, roles)]
    visible_indicators = _slider._detect_indicators(
        arr, int(roles["background"]), int(roles["target"])
    )
    targets = _slider._detect_targets(
        arr, pieces, visible_indicators, int(roles["target"])
    )
    indicator_sites = _slider._augment_indicators_from_target_crosses(
        arr, visible_indicators, int(roles["target"])
    )
    sliders = [dict(slider) for slider in _slider._detect_sliders(arr, roles)]
    if not pieces or not sliders or not targets or not indicator_sites:
        return {"accepted": False, "reject_reason": "incomplete_linked_extent_roles"}

    unit = _infer_growth_unit(pieces)
    if unit is None:
        return {"accepted": False, "reject_reason": "unstable_growth_unit"}
    passive_pieces = _detect_passive_axial_bodies(
        arr,
        roles,
        pieces,
        sliders,
        unit,
    )
    indicator_sites = _recover_occluded_indicator_sites(
        indicator_sites,
        passive_pieces,
        required_count=len(targets),
        frame_shape=arr.shape,
        unit=unit,
    )
    passive_pieces = _extend_bodies_through_indicator_sites(
        passive_pieces,
        indicator_sites,
        frame=arr,
        unit=unit,
    )
    entities = [
        *[{**piece, "passive": False} for piece in pieces],
        *passive_pieces,
    ]
    parent_to_child = _build_visible_chain_links(entities)
    indicators = [
        {"x": int(x), "y": int(y), "w": unit, "h": unit}
        for x, y in indicator_sites
    ]
    for slider in sliders:
        piece_indices = [
            idx
            for idx, piece in enumerate(pieces)
            if int(piece["color"]) == int(slider["controls_color"])
        ]
        slider["piece_indices"] = piece_indices
        slider["piece_idx"] = piece_indices[0] if piece_indices else None
    for target in targets:
        target["indicator_idx"] = _nearest_indicator(target, indicators)
        target["visible_bbox"] = _visible_bbox(
            arr,
            _bbox(target),
            int(roles["target"]),
        )
    return {
        "accepted": True,
        "roles": roles,
        "frame_shape": [int(arr.shape[0]), int(arr.shape[1])],
        "growth_unit": int(unit),
        "pieces": pieces,
        "entities": entities,
        "controlled_piece_count": len(pieces),
        "passive_pieces": passive_pieces,
        "sliders": sliders,
        "parent_to_child": parent_to_child,
        "targets": targets,
        "indicators": indicators,
    }


def _detect_passive_axial_bodies(
    frame: np.ndarray,
    roles: Mapping[str, Any],
    pieces: list[Mapping[str, Any]],
    sliders: list[Mapping[str, Any]],
    unit: int,
) -> list[dict[str, Any]]:
    """Recover linked bars even when edge contact joins their color regions."""

    height, width = frame.shape
    excluded = {
        int(roles["background"]),
        int(roles["marker"]),
        int(roles["slider_outline"]),
        int(roles["slider_fill"]),
        int(roles["target"]),
    }
    known = np.zeros_like(frame, dtype=bool)
    for item in [*pieces, *sliders]:
        x, y, item_width, item_height = _bbox(item)
        known[
            max(0, y) : min(height, y + item_height),
            max(0, x) : min(width, x + item_width),
        ] = True
    known[frame == int(roles["target"])] = True
    known[frame == int(roles["marker"])] = True
    attachment_probes = [
        _growth_probe(piece)
        for piece in pieces
        if piece.get("rotation") is not None
    ]

    bodies: list[dict[str, Any]] = []
    for raw_color in np.unique(frame):
        color = int(raw_color)
        if color in excluded:
            continue
        color_mask = (frame == color) & ~known
        for component in _mask_components(color_mask):
            ys, xs = np.where(component)
            if not len(xs):
                continue
            component_width = int(xs.max() - xs.min() + 1)
            component_height = int(ys.max() - ys.min() + 1)
            component_area = int(len(xs))
            bbox_area = component_width * component_height
            if (
                min(component_width, component_height) > unit
                and component_area * 10 > bbox_area * 9
            ):
                continue
            for x, y, body_width, body_height in _peel_axial_rectangles(
                component,
                unit,
                attachment_probes=attachment_probes,
            ):
                bodies.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "w": int(body_width),
                        "h": int(body_height),
                        "color": color,
                        "rotation": None,
                        "passive": True,
                    }
                )
    bodies = _merge_marker_occluded_bodies(
        bodies,
        frame,
        marker_value=int(roles["target"]),
        unit=unit,
    )
    bodies = _extend_through_perpendicular_end(
        bodies,
        frame,
        marker_value=int(roles["target"]),
        unit=unit,
    )
    bodies = _resolve_passive_overlaps(
        bodies,
        attachment_probes=attachment_probes,
        unit=unit,
    )
    return sorted(
        bodies,
        key=lambda item: (item["y"], item["x"], item["w"], item["h"]),
    )


def _merge_marker_occluded_bodies(
    bodies: list[dict[str, Any]],
    frame: np.ndarray,
    *,
    marker_value: int,
    unit: int,
) -> list[dict[str, Any]]:
    merged = [dict(body) for body in bodies]
    changed = True
    while changed:
        changed = False
        for left_idx, left in enumerate(merged):
            for right_idx in range(left_idx + 1, len(merged)):
                right = merged[right_idx]
                if int(left["color"]) != int(right["color"]):
                    continue
                candidate: dict[str, Any] | None = None
                gap_bbox: tuple[int, int, int, int] | None = None
                if (
                    int(left["x"]) == int(right["x"])
                    and int(left["w"]) == int(right["w"])
                ):
                    top, bottom = sorted(
                        (left, right), key=lambda item: int(item["y"])
                    )
                    gap = int(bottom["y"]) - (
                        int(top["y"]) + int(top["h"])
                    )
                    if 0 < gap <= unit * 2:
                        gap_bbox = (
                            int(top["x"]),
                            int(top["y"]) + int(top["h"]),
                            int(top["w"]),
                            gap,
                        )
                        candidate = {
                            **top,
                            "h": int(bottom["y"]) + int(bottom["h"])
                            - int(top["y"]),
                        }
                elif (
                    int(left["y"]) == int(right["y"])
                    and int(left["h"]) == int(right["h"])
                ):
                    first, second = sorted(
                        (left, right), key=lambda item: int(item["x"])
                    )
                    gap = int(second["x"]) - (
                        int(first["x"]) + int(first["w"])
                    )
                    if 0 < gap <= unit * 2:
                        gap_bbox = (
                            int(first["x"]) + int(first["w"]),
                            int(first["y"]),
                            gap,
                            int(first["h"]),
                        )
                        candidate = {
                            **first,
                            "w": int(second["x"]) + int(second["w"])
                            - int(first["x"]),
                        }
                if candidate is None or gap_bbox is None:
                    continue
                gap_x, gap_y, gap_width, gap_height = gap_bbox
                gap_pixels = frame[
                    gap_y : gap_y + gap_height,
                    gap_x : gap_x + gap_width,
                ]
                body_color = int(left["color"])
                if (
                    not np.any(gap_pixels == marker_value)
                    or not np.all(
                        (gap_pixels == marker_value)
                        | (gap_pixels == body_color)
                    )
                ):
                    continue
                merged[left_idx] = candidate
                del merged[right_idx]
                changed = True
                break
            if changed:
                break
    return merged


def _extend_through_perpendicular_end(
    bodies: list[dict[str, Any]],
    frame: np.ndarray,
    *,
    marker_value: int,
    unit: int,
) -> list[dict[str, Any]]:
    extended = [dict(body) for body in bodies]
    for vertical_idx, vertical in enumerate(extended):
        vx, vy, vertical_width, vertical_height = _bbox(vertical)
        if vertical_width != unit or vertical_height <= unit:
            continue
        for horizontal in extended:
            if horizontal is vertical:
                continue
            hx, hy, horizontal_width, horizontal_height = _bbox(horizontal)
            if (
                int(vertical["color"]) != int(horizontal["color"])
                or horizontal_height != unit
                or horizontal_width <= unit
                or not (hx == vx or hx + horizontal_width == vx + unit)
            ):
                continue
            candidate = dict(vertical)
            gap_bbox: tuple[int, int, int, int] | None = None
            if 0 < hy - (vy + vertical_height) <= unit:
                gap_bbox = (vx, vy + vertical_height, unit, hy - (vy + vertical_height))
                candidate["h"] = hy + unit - vy
            elif 0 < vy - (hy + unit) <= unit:
                gap_bbox = (vx, hy + unit, unit, vy - (hy + unit))
                candidate["y"] = hy
                candidate["h"] = vy + vertical_height - hy
            if gap_bbox is None:
                continue
            gap_x, gap_y, gap_width, gap_height = gap_bbox
            gap_pixels = frame[
                gap_y : gap_y + gap_height,
                gap_x : gap_x + gap_width,
            ]
            body_color = int(vertical["color"])
            if gap_pixels.size and not np.all(
                (gap_pixels == marker_value) | (gap_pixels == body_color)
            ):
                continue
            extended[vertical_idx] = candidate
            vertical = candidate
            vx, vy, vertical_width, vertical_height = _bbox(vertical)
            break
    return extended


def _resolve_passive_overlaps(
    bodies: list[dict[str, Any]],
    *,
    attachment_probes: list[tuple[int, int]],
    unit: int,
) -> list[dict[str, Any]]:
    resolved = [dict(body) for body in bodies]
    changed = True
    while changed:
        changed = False
        for left_idx, left in enumerate(resolved):
            for right_idx in range(left_idx + 1, len(resolved)):
                right = resolved[right_idx]
                if int(left["color"]) != int(right["color"]):
                    continue
                left_bbox = tuple(_bbox(left))
                right_bbox = tuple(_bbox(right))
                if not _bbox_overlap_xywh(left_bbox, right_bbox):
                    continue
                candidates: list[list[dict[str, Any]]] = []
                for trim_idx in (left_idx, right_idx):
                    trimmed = _trim_overlap_at_axial_end(
                        resolved[trim_idx],
                        resolved[right_idx if trim_idx == left_idx else left_idx],
                        unit,
                    )
                    if trimmed is None:
                        continue
                    candidate = [dict(body) for body in resolved]
                    candidate[trim_idx] = trimmed
                    candidates.append(candidate)
                if not candidates:
                    continue
                resolved = max(
                    candidates,
                    key=lambda rows: (
                        _attachment_matching_count(
                            [tuple(_bbox(body)) for body in rows],
                            attachment_probes,
                        ),
                        _minimum_major_extent(
                            rows, color=int(left["color"]), unit=unit
                        ),
                        sum(int(body["w"]) * int(body["h"]) for body in rows),
                    ),
                )
                changed = True
                break
            if changed:
                break
    return resolved


def _minimum_major_extent(
    bodies: list[Mapping[str, Any]], *, color: int, unit: int
) -> int:
    extents = [
        max(int(body["w"]), int(body["h"]))
        for body in bodies
        if int(body["color"]) == color
        and max(int(body["w"]), int(body["h"])) > unit
    ]
    return min(extents) if extents else 0


def _trim_overlap_at_axial_end(
    body: Mapping[str, Any], other: Mapping[str, Any], unit: int
) -> dict[str, Any] | None:
    x, y, width, height = _bbox(body)
    ox, oy, other_width, other_height = _bbox(other)
    overlap_x0 = max(x, ox)
    overlap_y0 = max(y, oy)
    overlap_x1 = min(x + width, ox + other_width)
    overlap_y1 = min(y + height, oy + other_height)
    overlap_width = overlap_x1 - overlap_x0
    overlap_height = overlap_y1 - overlap_y0
    trimmed = dict(body)
    if height == unit and overlap_height == unit and overlap_width == unit:
        if overlap_x0 == x and width > unit:
            trimmed["x"] = x + unit
            trimmed["w"] = width - unit
            return trimmed
        if overlap_x1 == x + width and width > unit:
            trimmed["w"] = width - unit
            return trimmed
    if width == unit and overlap_width == unit and overlap_height == unit:
        if overlap_y0 == y and height > unit:
            trimmed["y"] = y + unit
            trimmed["h"] = height - unit
            return trimmed
        if overlap_y1 == y + height and height > unit:
            trimmed["h"] = height - unit
            return trimmed
    return None


def _recover_occluded_indicator_sites(
    sites: list[tuple[int, int]],
    passive_bodies: list[Mapping[str, Any]],
    *,
    required_count: int,
    frame_shape: tuple[int, ...],
    unit: int,
) -> list[tuple[int, int]]:
    recovered = set((int(x), int(y)) for x, y in sites)
    if len(recovered) >= required_count:
        return sorted(recovered)
    height, width = int(frame_shape[0]), int(frame_shape[1])
    candidates: list[tuple[int, int]] = []
    for body in passive_bodies:
        x, y, body_width, body_height = _bbox(body)
        if body_width != unit or body_height != unit:
            continue
        for site_x, site_y in list(recovered):
            if site_y == y and site_x == x + body_width:
                candidates.append((x - unit, y))
            if site_y == y and site_x + unit == x:
                candidates.append((x + body_width, y))
            if site_x == x and site_y == y + body_height:
                candidates.append((x, y - unit))
            if site_x == x and site_y + unit == y:
                candidates.append((x, y + body_height))
    for x, y in candidates:
        if 0 <= x <= width - unit and 0 <= y <= height - unit:
            recovered.add((x, y))
        if len(recovered) >= required_count:
            break
    return sorted(recovered)


def _extend_bodies_through_indicator_sites(
    bodies: list[dict[str, Any]],
    sites: list[tuple[int, int]],
    *,
    frame: np.ndarray,
    unit: int,
) -> list[dict[str, Any]]:
    height, width = int(frame.shape[0]), int(frame.shape[1])
    extended: list[dict[str, Any]] = []
    for body_idx, body in enumerate(bodies):
        row = dict(body)
        x, y, body_width, body_height = _bbox(row)
        if max(body_width, body_height) <= unit:
            extended.append(row)
            continue
        for site_x, site_y in sites:
            site_pixels = frame[
                site_y : site_y + unit,
                site_x : site_x + unit,
            ]
            if not np.any(site_pixels == int(body["color"])):
                continue
            occupied_by_other = any(
                other_idx != body_idx
                and int(other["x"]) <= site_x < int(other["x"]) + int(other["w"])
                and int(other["y"]) <= site_y < int(other["y"]) + int(other["h"])
                for other_idx, other in enumerate(bodies)
            )
            if occupied_by_other:
                continue
            if body_width == unit and site_x == x:
                if y + body_height == site_y and site_y + unit <= height:
                    row["h"] = body_height + unit
                elif site_y + unit == y and site_y >= 0:
                    row["y"] = site_y
                    row["h"] = body_height + unit
            elif body_height == unit and site_y == y:
                if x + body_width == site_x and site_x + unit <= width:
                    row["w"] = body_width + unit
                elif site_x + unit == x and site_x >= 0:
                    row["x"] = site_x
                    row["w"] = body_width + unit
        extended.append(row)
    return extended


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for start_y, start_x in zip(*np.where(mask & ~seen)):
        if seen[start_y, start_x]:
            continue
        component = np.zeros_like(mask, dtype=bool)
        stack = [(int(start_y), int(start_x))]
        while stack:
            y, x = stack.pop()
            if (
                y < 0
                or y >= height
                or x < 0
                or x >= width
                or seen[y, x]
                or not mask[y, x]
            ):
                continue
            seen[y, x] = True
            component[y, x] = True
            stack.extend(((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)))
        components.append(component)
    return components


def _peel_axial_rectangles(
    component: np.ndarray,
    unit: int,
    *,
    attachment_probes: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    initial_candidates = _axial_rectangle_candidates(component, unit)
    if not initial_candidates:
        return []
    covers: list[list[tuple[int, int, int, int]]] = []
    for first in initial_candidates:
        remaining = component.copy()
        x, y, width, height = first
        remaining[y : y + height, x : x + width] = False
        tail, _complete = _greedy_axial_cover(remaining, unit)
        covers.append([first, *tail])
    if not covers:
        fallback, _complete = _greedy_axial_cover(component.copy(), unit)
        return fallback
    return max(
        covers,
        key=lambda rectangles: (
            _attachment_matching_count(rectangles, attachment_probes),
            -len(rectangles),
            sum(width * height for _x, _y, width, height in rectangles),
        ),
    )


def _greedy_axial_cover(
    remaining: np.ndarray, unit: int
) -> tuple[list[tuple[int, int, int, int]], bool]:
    rectangles: list[tuple[int, int, int, int]] = []
    while np.any(remaining):
        candidates = _axial_rectangle_candidates(remaining, unit)
        if not candidates:
            break
        x, y, width, height = max(
            candidates,
            key=lambda box: (
                box[2] * box[3],
                max(box[2], box[3]),
                -box[1],
                -box[0],
            ),
        )
        rectangles.append((x, y, width, height))
        remaining[y : y + height, x : x + width] = False
    return rectangles, not bool(np.any(remaining))


def _attachment_matching_count(
    rectangles: list[tuple[int, int, int, int]],
    probes: list[tuple[int, int]],
) -> int:
    candidates: list[list[int]] = []
    for probe_x, probe_y in probes:
        candidates.append(
            [
                rectangle_idx
                for rectangle_idx, (x, y, width, height) in enumerate(rectangles)
                if x <= probe_x < x + width and y <= probe_y < y + height
            ]
        )
    body_to_probe: dict[int, int] = {}

    def assign(probe_idx: int, seen: set[int]) -> bool:
        for body_idx in candidates[probe_idx]:
            if body_idx in seen:
                continue
            seen.add(body_idx)
            previous_probe = body_to_probe.get(body_idx)
            if previous_probe is None or assign(previous_probe, seen):
                body_to_probe[body_idx] = probe_idx
                return True
        return False

    matched = 0
    for probe_idx in range(len(probes)):
        if assign(probe_idx, set()):
            matched += 1
    return matched


def _axial_rectangle_candidates(
    mask: np.ndarray, unit: int
) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    candidates: set[tuple[int, int, int, int]] = set()
    for y in range(max(0, height - unit + 1)):
        full_columns = np.all(mask[y : y + unit, :], axis=0)
        for x, run_width in _true_runs(full_columns):
            if run_width >= unit:
                candidates.add((x, y, run_width, unit))
    for x in range(max(0, width - unit + 1)):
        full_rows = np.all(mask[:, x : x + unit], axis=1)
        for y, run_height in _true_runs(full_rows):
            if run_height >= unit:
                candidates.add((x, y, unit, run_height))
    return list(candidates)


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(values).tolist()
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index == previous + 1:
            previous = index
            continue
        runs.append((start, previous - start + 1))
        start = previous = index
    runs.append((start, previous - start + 1))
    return runs


def _build_visible_chain_links(
    entities: list[Mapping[str, Any]],
) -> dict[int, int | None]:
    links: dict[int, int | None] = {}
    probes: dict[int, tuple[int, int]] = {}
    for entity_idx, entity in enumerate(entities):
        if entity.get("rotation") is None:
            links[entity_idx] = None
            continue
        probe = _growth_probe(entity)
        probes[entity_idx] = probe
        candidates: list[int] = []
        for other_idx, other in enumerate(entities):
            if other_idx == entity_idx:
                continue
            other_x, other_y, other_width, other_height = _bbox(other)
            if (
                other_x <= probe[0] < other_x + other_width
                and other_y <= probe[1] < other_y + other_height
            ):
                candidates.append(other_idx)
        links[entity_idx] = (
            min(
                candidates,
                key=lambda idx: (
                    int(entities[idx]["w"]) * int(entities[idx]["h"]),
                    idx,
                ),
            )
            if candidates
            else None
        )
    incoming: dict[int, list[int]] = {}
    for parent, child in links.items():
        if child is not None:
            incoming.setdefault(int(child), []).append(int(parent))
    for child_idx, parents in incoming.items():
        if len(parents) <= 1:
            continue
        child = entities[child_idx]
        if child.get("rotation") is None:
            keep = max(
                parents,
                key=lambda parent: (
                    _visible_parent_depth(parent, links),
                    -parent,
                ),
            )
        else:
            anchor = _marker_anchor(child)
            keep = min(
                parents,
                key=lambda parent: (
                    abs(probes[parent][0] - anchor[0])
                    + abs(probes[parent][1] - anchor[1]),
                    parent,
                ),
            )
        for parent in parents:
            if parent != keep:
                links[parent] = None
    return links


def _visible_parent_depth(
    entity_idx: int, links: Mapping[int, int | None]
) -> int:
    child_to_parents: dict[int, list[int]] = {}
    for parent, child in links.items():
        if child is not None:
            child_to_parents.setdefault(int(child), []).append(int(parent))
    depth = 0
    current = int(entity_idx)
    seen = {current}
    while current in child_to_parents:
        candidates = [
            parent for parent in child_to_parents[current] if parent not in seen
        ]
        if not candidates:
            break
        current = min(candidates)
        seen.add(current)
        depth += 1
    return depth


def _growth_probe(entity: Mapping[str, Any]) -> tuple[int, int]:
    x, y, width, height = _bbox(entity)
    direction = _growth_direction(int(entity["rotation"]))
    if direction == (1, 0):
        return x + width, y + height // 2
    if direction == (-1, 0):
        return x - 1, y + height // 2
    if direction == (0, 1):
        return x + width // 2, y + height
    return x + width // 2, y - 1


def _marker_anchor(entity: Mapping[str, Any]) -> tuple[int, int]:
    x, y, width, height = _bbox(entity)
    rotation = int(entity["rotation"])
    if rotation == 0:
        return x + width // 2, y + height - 1
    if rotation == 90:
        return x, y + height // 2
    if rotation == 180:
        return x + width // 2, y
    return x + width - 1, y + height // 2


def _entity_ancestors(
    links: Mapping[int, int | None], entity_count: int
) -> list[list[int]]:
    child_to_parent: dict[int, int] = {}
    for parent, child in links.items():
        if child is not None and int(child) not in child_to_parent:
            child_to_parent[int(child)] = int(parent)
    out: list[list[int]] = []
    for entity_idx in range(entity_count):
        chain: list[int] = []
        seen = {entity_idx}
        current = entity_idx
        while current in child_to_parent:
            parent = child_to_parent[current]
            if parent in seen:
                break
            chain.append(parent)
            seen.add(parent)
            current = parent
        chain.reverse()
        out.append(chain)
    return out


def _entity_id(entity_idx: int, controlled_count: int) -> str:
    if entity_idx < controlled_count:
        return f"piece_{entity_idx}"
    return f"passive_{entity_idx - controlled_count}"


def _public_binding(perceived: Mapping[str, Any]) -> dict[str, Any]:
    pieces = list(perceived["pieces"])
    entities = list(perceived.get("entities") or pieces)
    controlled_count = int(perceived.get("controlled_piece_count", len(pieces)))
    sliders = list(perceived["sliders"])
    links = dict(perceived["parent_to_child"])
    targets = list(perceived["targets"])
    indicators = list(perceived["indicators"])
    return {
        "family_id": FAMILY_ID,
        "accepted": True,
        "frame_shape": list(perceived["frame_shape"]),
        "inferred_growth_unit": int(perceived["growth_unit"]),
        "pieces": [
            {
                "id": f"piece_{idx}",
                "bbox": _bbox(piece),
                "rotation": int(piece["rotation"]),
                "extent_axis": (
                    "horizontal"
                    if int(piece["rotation"]) in {90, 270}
                    else "vertical"
                ),
            }
            for idx, piece in enumerate(pieces)
        ],
        "passive_pieces": [
            {
                "id": f"passive_{idx - controlled_count}",
                "bbox": _bbox(piece),
                "role": "linked_collision_body",
            }
            for idx, piece in enumerate(entities)
            if idx >= controlled_count
        ],
        "sliders": [
            {
                "id": f"slider_{idx}",
                "bbox": _bbox(slider),
                "grow_site": [int(value) for value in slider["click"]],
                "shrink_site": [int(value) for value in slider["shrink_click"]],
                "piece": (
                    None
                    if slider.get("piece_idx") is None
                    else f"piece_{int(slider['piece_idx'])}"
                ),
                "pieces": [
                    f"piece_{int(piece_idx)}"
                    for piece_idx in slider.get("piece_indices", [])
                ],
            }
            for idx, slider in enumerate(sliders)
        ],
        "child_links": [
            {
                "parent": _entity_id(parent, controlled_count),
                "child": _entity_id(int(child), controlled_count),
            }
            for parent, child in sorted(links.items())
            if child is not None
        ],
        "co_motion_links": [
            {
                "driver": _entity_id(parent, controlled_count),
                "dependant": _entity_id(int(child), controlled_count),
                "relation": "translates_with_parent_extent_change",
            }
            for parent, child in sorted(links.items())
            if child is not None
        ],
        "targets": [
            {
                "id": f"target_{idx}",
                "bbox": _bbox(target),
                "parent": f"piece_{int(target['parent_idx'])}",
                "indicator": (
                    None
                    if target.get("indicator_idx") is None
                    else f"indicator_{int(target['indicator_idx'])}"
                ),
            }
            for idx, target in enumerate(targets)
        ],
        "indicators": [
            {"id": f"indicator_{idx}", "bbox": _bbox(indicator)}
            for idx, indicator in enumerate(indicators)
        ],
    }


def _plan_linked_extent_actions(
    frame: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Plan from roles rebound in the current frame only."""

    perceived = _perceive(frame)
    if not perceived.get("accepted"):
        return [], {
            "accepted": False,
            "reject_reason": str(
                perceived.get("reject_reason", "unbound_linked_extent_frame")
            ),
        }
    if len(perceived["pieces"]) <= 4 and len(perceived["sliders"]) <= 4:
        return _slider.plan_slider_growth_actions(frame)
    return _solve_joint_chain_actions(_as_frame(frame), perceived)


def _solve_joint_chain_actions(
    frame: np.ndarray,
    perceived: Mapping[str, Any],
    *,
    max_expansions: int = 180_000,
    max_plan_len: int = 80,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search signed slider deltas with all visible chain bodies coupled."""

    pieces = list(perceived["pieces"])
    entities = list(perceived.get("entities") or pieces)
    sliders = list(perceived["sliders"])
    targets = list(perceived["targets"])
    indicators = list(perceived["indicators"])
    controlled_count = int(perceived.get("controlled_piece_count", len(pieces)))
    unit = int(perceived["growth_unit"])
    if (
        not pieces
        or not sliders
        or not targets
        or not indicators
        or unit <= 0
    ):
        return [], {
            "accepted": False,
            "reject_reason": "missing_joint_chain_inputs",
        }
    if (
        len(pieces) > 10
        or len(sliders) > 6
        or len(targets) > 3
        or len(entities) > 16
    ):
        return [], {
            "accepted": False,
            "reject_reason": "joint_chain_scope_limit",
        }

    control_colors: list[int] = []
    control_sliders: list[Mapping[str, Any]] = []
    for slider in sliders:
        color = int(slider["controls_color"])
        if color in control_colors:
            continue
        control_colors.append(color)
        control_sliders.append(slider)
    color_to_control = {
        color: control_idx for control_idx, color in enumerate(control_colors)
    }
    try:
        piece_to_control = [
            color_to_control[int(piece["color"])] for piece in pieces
        ]
    except KeyError:
        return [], {
            "accepted": False,
            "reject_reason": "missing_joint_slider_for_piece",
        }

    frame_height, frame_width = int(frame.shape[0]), int(frame.shape[1])
    min_counts = [-(max(frame_height, frame_width) // unit) for _ in control_colors]
    max_counts = [max(frame_height, frame_width) // unit for _ in control_colors]
    for piece_idx, piece in enumerate(pieces):
        control_idx = piece_to_control[piece_idx]
        rotation = int(piece["rotation"])
        extent = int(piece["h"] if rotation in {0, 180} else piece["w"])
        min_counts[control_idx] = max(
            min_counts[control_idx],
            -(max(1, extent // unit) - 1),
        )
        if rotation == 0:
            capacity = int(piece["y"]) // unit
        elif rotation == 90:
            capacity = (
                frame_width - int(piece["x"]) - int(piece["w"])
            ) // unit
        elif rotation == 180:
            capacity = (
                frame_height - int(piece["y"]) - int(piece["h"])
            ) // unit
        else:
            capacity = int(piece["x"]) // unit
        max_counts[control_idx] = min(
            max_counts[control_idx], max(0, int(capacity))
        )
    min_counts_t = tuple(int(value) for value in min_counts)
    max_counts_t = tuple(int(value) for value in max_counts)

    ancestors = _entity_ancestors(
        perceived["parent_to_child"], len(entities)
    )
    entity_terms: list[list[tuple[int, int, int, int, int]]] = []
    for entity_idx, entity in enumerate(entities):
        terms: dict[int, list[int]] = {}

        def add_term(
            control_idx: int,
            dx: int = 0,
            dy: int = 0,
            dw: int = 0,
            dh: int = 0,
        ) -> None:
            row = terms.setdefault(control_idx, [0, 0, 0, 0])
            row[0] += int(dx)
            row[1] += int(dy)
            row[2] += int(dw)
            row[3] += int(dh)

        if entity_idx < controlled_count:
            control_idx = piece_to_control[entity_idx]
            rotation = int(entity["rotation"])
            if rotation == 0:
                add_term(control_idx, dy=-unit, dh=unit)
            elif rotation == 90:
                add_term(control_idx, dw=unit)
            elif rotation == 180:
                add_term(control_idx, dh=unit)
            else:
                add_term(control_idx, dx=-unit, dw=unit)
        for ancestor_idx in ancestors[entity_idx]:
            if ancestor_idx >= controlled_count:
                continue
            control_idx = piece_to_control[ancestor_idx]
            direction = _growth_direction(
                int(entities[ancestor_idx]["rotation"])
            )
            add_term(
                control_idx,
                dx=direction[0] * unit,
                dy=direction[1] * unit,
            )
        entity_terms.append(
            [
                (control_idx, values[0], values[1], values[2], values[3])
                for control_idx, values in terms.items()
            ]
        )

    target_terms: list[list[tuple[int, int, int]]] = []
    for target in targets:
        parent_idx = int(target["parent_idx"])
        terms: dict[int, list[int]] = {}
        for moving_idx in [*ancestors[parent_idx], parent_idx]:
            if moving_idx >= controlled_count:
                continue
            control_idx = piece_to_control[moving_idx]
            direction = _growth_direction(int(entities[moving_idx]["rotation"]))
            row = terms.setdefault(control_idx, [0, 0])
            row[0] += direction[0] * unit
            row[1] += direction[1] * unit
        target_terms.append(
            [
                (control_idx, delta[0], delta[1])
                for control_idx, delta in terms.items()
            ]
        )

    assignment = _assign_joint_indicators(
        targets,
        indicators,
        target_terms,
        min_counts_t,
        max_counts_t,
    )
    if assignment is None:
        return [], {
            "accepted": False,
            "reject_reason": "missing_joint_target_assignment",
        }

    obstacle = _joint_obstacle_mask(frame, perceived)
    obstacle_integral = np.pad(
        obstacle.astype(np.int32).cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
    )

    def state_bboxes(
        state: tuple[int, ...],
    ) -> tuple[tuple[int, int, int, int], ...]:
        rows: list[tuple[int, int, int, int]] = []
        for entity, terms in zip(entities, entity_terms):
            x, y, width, height = _bbox(entity)
            for control_idx, dx, dy, dw, dh in terms:
                count = state[control_idx]
                x += dx * count
                y += dy * count
                width += dw * count
                height += dh * count
            rows.append((x, y, width, height))
        return tuple(rows)

    validity_cache: dict[tuple[int, ...], bool] = {}

    def valid(state: tuple[int, ...]) -> bool:
        cached = validity_cache.get(state)
        if cached is not None:
            return cached
        bboxes = state_bboxes(state)
        for x, y, width, height in bboxes:
            if (
                width < unit
                or height < unit
                or x < 0
                or y < 0
                or x + width > frame_width
                or y + height > frame_height
                or _integral_rect_sum(
                    obstacle_integral, x, y, width, height
                )
            ):
                validity_cache[state] = False
                return False
        for left_idx, left in enumerate(bboxes):
            for right in bboxes[left_idx + 1 :]:
                if _bbox_overlap_xywh(left, right):
                    validity_cache[state] = False
                    return False
        validity_cache[state] = True
        return True

    def target_position(
        target_idx: int, state: tuple[int, ...]
    ) -> tuple[int, int]:
        target = targets[target_idx]
        x, y = int(target["x"]), int(target["y"])
        for control_idx, dx, dy in target_terms[target_idx]:
            x += dx * state[control_idx]
            y += dy * state[control_idx]
        return x, y

    def heuristic(state: tuple[int, ...]) -> int:
        residual = 0
        for target_idx, indicator_idx in enumerate(assignment):
            x, y = target_position(target_idx, state)
            indicator = indicators[indicator_idx]
            residual += (
                abs(x - int(indicator["x"]))
                + abs(y - int(indicator["y"]))
            ) // unit
        return int(residual)

    def goal(state: tuple[int, ...]) -> bool:
        return all(
            target_position(target_idx, state)
            == (
                int(indicators[indicator_idx]["x"]),
                int(indicators[indicator_idx]["y"]),
            )
            for target_idx, indicator_idx in enumerate(assignment)
        )

    start = tuple(0 for _ in control_colors)
    if not valid(start):
        return [], {
            "accepted": False,
            "reject_reason": "joint_chain_start_state_invalid",
        }

    previous: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}
    previous_action: dict[tuple[int, ...], tuple[int, str]] = {}
    costs: dict[tuple[int, ...], int] = {start: 0}
    queue: list[tuple[int, int, int, int, tuple[int, ...]]] = []
    serial = 0
    start_h = heuristic(start)
    heapq.heappush(queue, (start_h, start_h, 0, serial, start))
    expanded = 0
    found: tuple[int, ...] | None = None
    while queue and expanded < max_expansions:
        _priority, _residual, cost, _serial, state = heapq.heappop(queue)
        if cost != costs.get(state):
            continue
        expanded += 1
        if goal(state):
            found = state
            break
        if cost >= max_plan_len:
            continue
        candidates: list[tuple[int, int, int, str, tuple[int, ...]]] = []
        for control_idx in range(len(control_colors)):
            for delta, mode in ((1, "grow"), (-1, "shrink")):
                next_value = state[control_idx] + delta
                if not (
                    min_counts_t[control_idx]
                    <= next_value
                    <= max_counts_t[control_idx]
                ):
                    continue
                next_state_l = list(state)
                next_state_l[control_idx] = next_value
                next_state = tuple(next_state_l)
                if not valid(next_state):
                    continue
                next_h = heuristic(next_state)
                candidates.append(
                    (next_h, abs(next_value), control_idx, mode, next_state)
                )
        candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
        for next_h, _magnitude, control_idx, mode, next_state in candidates:
            next_cost = cost + 1
            if next_cost >= costs.get(next_state, 10**9):
                continue
            costs[next_state] = next_cost
            previous[next_state] = state
            previous_action[next_state] = (control_idx, mode)
            serial += 1
            heapq.heappush(
                queue,
                (
                    next_cost + next_h,
                    next_h,
                    next_cost,
                    serial,
                    next_state,
                ),
            )

    if found is None:
        return [], {
            "accepted": False,
            "reject_reason": "no_joint_chain_plan",
            "solver": "joint_linked_extent_astar",
            "joint_state_search": True,
            "expanded_state_count": expanded,
        }

    path: list[tuple[int, str]] = []
    node = found
    while previous[node] is not None:
        path.append(previous_action[node])
        node = previous[node]  # type: ignore[assignment]
    path.reverse()
    mode_counts: Counter[str] = Counter()
    payloads: list[dict[str, Any]] = []
    for control_idx, mode in path:
        slider = control_sliders[control_idx]
        site = slider["click"] if mode == "grow" else slider["shrink_click"]
        mode_counts[mode] += 1
        payloads.append(
            {
                "action": "ACTION6",
                "x": int(site[0]),
                "y": int(site[1]),
                "mode": mode,
            }
        )
    return payloads, {
        "accepted": True,
        "reject_reason": "",
        "solver": "joint_linked_extent_astar",
        "joint_state_search": True,
        "plan_len": len(payloads),
        "expanded_state_count": expanded,
        "controlled_piece_count": controlled_count,
        "passive_piece_count": len(entities) - controlled_count,
        "control_group_count": len(control_colors),
        "mode_counts": dict(mode_counts),
        "final_joint_counts": [int(value) for value in found],
    }


def _assign_joint_indicators(
    targets: list[Mapping[str, Any]],
    indicators: list[Mapping[str, Any]],
    target_terms: list[list[tuple[int, int, int]]],
    min_counts: tuple[int, ...],
    max_counts: tuple[int, ...],
) -> tuple[int, ...] | None:
    if len(targets) > len(indicators):
        return None
    best: tuple[int, tuple[int, ...]] | None = None
    for assignment in permutations(range(len(indicators)), len(targets)):
        distance = 0
        reachable = True
        for target_idx, indicator_idx in enumerate(assignment):
            target = targets[target_idx]
            indicator = indicators[indicator_idx]
            needed_x = int(indicator["x"]) - int(target["x"])
            needed_y = int(indicator["y"]) - int(target["y"])
            terms = target_terms[target_idx]
            if not _bounded_axis_reachable(
                needed_x,
                [(control_idx, dx) for control_idx, dx, _dy in terms],
                min_counts,
                max_counts,
            ) or not _bounded_axis_reachable(
                needed_y,
                [(control_idx, dy) for control_idx, _dx, dy in terms],
                min_counts,
                max_counts,
            ):
                reachable = False
                break
            distance += abs(needed_x) + abs(needed_y)
        if reachable and (best is None or distance < best[0]):
            best = (distance, tuple(int(value) for value in assignment))
    return None if best is None else best[1]


def _bounded_axis_reachable(
    needed: int,
    terms: list[tuple[int, int]],
    min_counts: tuple[int, ...],
    max_counts: tuple[int, ...],
) -> bool:
    coefficients: dict[int, int] = {}
    for control_idx, coefficient in terms:
        coefficients[control_idx] = coefficients.get(control_idx, 0) + int(
            coefficient
        )
    coefficients = {
        control_idx: coefficient
        for control_idx, coefficient in coefficients.items()
        if coefficient
    }
    if not coefficients:
        return needed == 0
    low = 0
    high = 0
    divisor = 0
    for control_idx, coefficient in coefficients.items():
        values = (
            coefficient * min_counts[control_idx],
            coefficient * max_counts[control_idx],
        )
        low += min(values)
        high += max(values)
        divisor = math.gcd(divisor, abs(coefficient))
    return low <= needed <= high and (divisor == 0 or needed % divisor == 0)


def _joint_obstacle_mask(
    frame: np.ndarray, perceived: Mapping[str, Any]
) -> np.ndarray:
    roles = perceived["roles"]
    obstacle = (frame >= 0) & (frame != int(roles["background"]))
    for item in [
        *list(perceived.get("entities") or perceived["pieces"]),
        *list(perceived["sliders"]),
    ]:
        x, y, width, height = _bbox(item)
        obstacle[
            max(0, y) : min(frame.shape[0], y + height),
            max(0, x) : min(frame.shape[1], x + width),
        ] = False
    obstacle &= frame != int(roles["target"])
    obstacle &= frame != int(roles["marker"])
    for piece in perceived["pieces"]:
        obstacle &= frame != int(piece["color"])
    return obstacle


def _integral_rect_sum(
    integral: np.ndarray, x: int, y: int, width: int, height: int
) -> int:
    x1 = x + width
    y1 = y + height
    return int(
        integral[y1, x1]
        - integral[y, x1]
        - integral[y1, x]
        + integral[y, x]
    )


def _bbox_overlap_xywh(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return lx < rx + rw and rx < lx + lw and ly < ry + rh and ry < ly + lh


def _canonical_plan(plan: Any) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for item in plan or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("action", "")).upper().split(".")[-1] != "ACTION6":
            continue
        try:
            action: dict[str, Any] = {
                "action": "ACTION6",
                "x": int(item["x"]),
                "y": int(item["y"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        mode = str(item.get("mode", "") or "").strip().lower()
        if mode:
            action["mode"] = mode
        actions.append(action)
    return actions


def _blocked_plan(
    reason: str,
    *,
    context: MechanicContext | None,
) -> dict[str, Any]:
    return {
        "schema_version": "tufa_linked_extent_plan.v3",
        "family_id": FAMILY_ID,
        "status": "blocked",
        "actions": [],
        "action_count": 0,
        "certified": False,
        "certificate_reason": str(reason),
        "suffix_exposed": False,
        "detail": {"inherited_schema": _inherited_schema(context)},
    }


def _certification_scope_reason(
    binding: Mapping[str, Any],
    action: Mapping[str, Any],
    context: MechanicContext | None,
) -> str:
    if not binding.get("accepted"):
        return "current_roles_not_bound"
    if str(action.get("mode", "grow") or "grow") not in {"grow", "shrink"}:
        return "unsupported_extent_mode"
    if len(binding.get("pieces") or []) > 10 or len(binding.get("sliders") or []) > 6:
        return "linked_graph_exceeds_verified_l1_l3_scope"
    level = int(context.level) if isinstance(context, MechanicContext) else 0
    inherited = _inherited_schema(context)
    if level > 3:
        return "level_exceeds_verified_l1_l3_scope"
    if level == 2 and not inherited.get("verified"):
        return "l2_requires_verified_prior_linked_extent_schema"
    if level == 3 and (
        not inherited.get("verified")
        or int(inherited.get("verified_through_level", 0) or 0) < 2
    ):
        return "l3_requires_verified_l2_linked_extent_schema"
    return ""


def _inherited_schema(context: MechanicContext | None) -> dict[str, Any]:
    empty = {
        "available": False,
        "verified": False,
        "learned_level": 0,
        "verified_through_level": 0,
        "bound_level": (
            int(context.level) if isinstance(context, MechanicContext) else 0
        ),
    }
    if not isinstance(context, MechanicContext) or not isinstance(
        context.memory, Mapping
    ):
        return empty
    schemas = context.memory.get("schemas")
    if not isinstance(schemas, list):
        return empty
    candidates = [
        row
        for row in schemas
        if isinstance(row, Mapping)
        and (
            str(row.get("source_family", "")) == "linked_extent"
            or str(row.get("mechanic_id", "")) == FAMILY_ID
        )
        and int(row.get("learned_level", 0) or 0) < int(context.level)
    ]
    if not candidates:
        return empty
    selected = max(
        candidates,
        key=lambda row: (
            str(row.get("status", "")) == "verified",
            int(row.get("learned_level", 0) or 0),
            float(row.get("confidence", 0.0) or 0.0),
        ),
    )
    evidence = selected.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    learned_level = int(selected.get("learned_level", 0) or 0)
    verified_levels: list[int] = []
    for value in evidence.get("levels") or ():
        try:
            verified_levels.append(int(value))
        except (TypeError, ValueError):
            continue
    return {
        "available": True,
        "verified": str(selected.get("status", "")) == "verified",
        "status": str(selected.get("status", "observed")),
        "learned_level": learned_level,
        "verified_through_level": max([learned_level, *verified_levels]),
        "bound_level": int(context.level),
        "operator_kind": str(
            (selected.get("operator") or {}).get("kind", FAMILY_ID)
            if isinstance(selected.get("operator"), Mapping)
            else FAMILY_ID
        ),
        "exact_transition_count": int(
            evidence.get("exact_transition_count", 0) or 0
        ),
        "score_closure_count": int(evidence.get("score_closure_count", 0) or 0),
    }


def _verify_abstract_successor(
    expected: Mapping[str, Any],
    current_frame: Any,
) -> dict[str, Any]:
    progress = expected.get("abstract_progress")
    if not isinstance(progress, Mapping):
        return {"accepted": False, "reject_reason": "missing_abstract_progress"}
    try:
        expected_after = int(progress["expected_after"])
    except (KeyError, TypeError, ValueError):
        return {"accepted": False, "reject_reason": "invalid_abstract_progress"}
    if expected_after <= 0:
        return {
            "accepted": False,
            "reject_reason": "zero_residual_requires_level_transition",
        }
    plan, diagnostics = _plan_linked_extent_actions(current_frame)
    if not diagnostics.get("accepted"):
        return {
            "accepted": False,
            "reject_reason": "successor_rebinding_failed",
            "evidence": {"successor_plan_accepted": False},
        }
    actions = _canonical_plan(plan)
    actual_digest = stable_digest(actions)
    expected_digest = str(progress.get("expected_successor_digest", "") or "")
    expected_before = int(progress.get("before", expected_after + 1) or 0)
    accepted = bool(
        actions
        and len(actions) <= expected_after
        and len(actions) < expected_before
    )
    return {
        "accepted": accepted,
        "reject_reason": (
            "" if accepted else "abstract_successor_not_monotonic"
        ),
        "evidence": {
            "successor_plan_accepted": True,
            "successor_plan_length": len(actions),
            "expected_successor_plan_length": expected_after,
            "successor_digest_match": actual_digest == expected_digest,
            "strictly_decreasing": len(actions) < expected_before,
        },
    }


def _verified_goal_overlap_rebound(
    expected: Mapping[str, Any],
    visual_reasons: list[str],
    abstract: Mapping[str, Any],
) -> bool:
    """Accept exact rebound proof when a completed target is visually occluded."""

    if not visual_reasons or not _has_exact_rebound(abstract):
        return False

    completed_targets: set[str] = set()
    for row in list(expected.get("target_distance_evidence") or []):
        if not isinstance(row, Mapping) or not bool(row.get("decreases")):
            continue
        try:
            distance_after = float(row.get("distance_after", float("inf")))
        except (TypeError, ValueError):
            continue
        if abs(distance_after) <= 1e-9:
            completed_targets.add(str(row.get("target", "")))
    if not completed_targets:
        return False

    completed_rows = [
        row
        for row in list(expected.get("linked_target_translations") or [])
        if isinstance(row, Mapping)
        and str(row.get("entity", "")) in completed_targets
    ]
    if not completed_rows:
        return False
    completed_parents = {str(row.get("parent", "")) for row in completed_rows}

    target_reason_prefixes = {
        "linked_target_mismatch",
        "missing_exact_target_translation",
    }
    parent_reason_prefixes = {
        "extent_bbox_mismatch",
        "missing_exact_unit_extent_evidence",
    }
    for reason in visual_reasons:
        if reason == "missing_decreasing_target_distance_evidence":
            continue
        prefix, separator, entity = str(reason).partition(":")
        if separator and prefix in target_reason_prefixes:
            if entity in completed_targets:
                continue
            return False
        if separator and prefix in parent_reason_prefixes:
            if entity in completed_parents:
                continue
            return False
        return False
    return True


def _verified_structural_rebound(
    expected: Mapping[str, Any],
    visual_reasons: list[str],
    abstract: Mapping[str, Any],
) -> bool:
    """Back exact structural matches when the auxiliary motion sensor is blind."""

    if not visual_reasons or not _has_exact_rebound(abstract):
        return False
    linked_targets = {
        str(row.get("entity", ""))
        for row in list(expected.get("linked_target_translations") or [])
        if isinstance(row, Mapping)
    }
    tracked_targets = {
        str(row.get("target", ""))
        for row in list(expected.get("target_distance_evidence") or [])
        if isinstance(row, Mapping)
    }
    decreasing_targets = {
        str(row.get("target", ""))
        for row in list(expected.get("target_distance_evidence") or [])
        if isinstance(row, Mapping) and bool(row.get("decreases"))
    }
    if not linked_targets or not linked_targets.issubset(tracked_targets):
        return False

    for reason in visual_reasons:
        if reason == "missing_decreasing_target_distance_evidence":
            if not linked_targets.issubset(decreasing_targets):
                return False
            continue
        prefix, separator, entity = str(reason).partition(":")
        if (
            separator
            and prefix == "missing_exact_target_translation"
            and entity in linked_targets
        ):
            continue
        return False
    return True


def _has_exact_rebound(abstract: Mapping[str, Any]) -> bool:
    if not bool(abstract.get("accepted")):
        return False
    evidence = abstract.get("evidence")
    if not isinstance(evidence, Mapping):
        return False
    if evidence.get("successor_digest_match") is not True:
        return False
    try:
        successor_length = int(evidence["successor_plan_length"])
        expected_length = int(evidence["expected_successor_plan_length"])
    except (KeyError, TypeError, ValueError):
        return False
    return successor_length == expected_length


def _infer_growth_unit(pieces: list[Mapping[str, Any]]) -> int | None:
    widths = [min(int(piece["w"]), int(piece["h"])) for piece in pieces]
    positive = [value for value in widths if value > 0]
    if not positive:
        return None
    counts = Counter(positive)
    unit, frequency = counts.most_common(1)[0]
    if frequency * 2 < len(positive):
        return None
    return int(unit)


def _nearest_indicator(
    target: Mapping[str, Any],
    indicators: list[Mapping[str, Any]],
) -> int | None:
    if not indicators:
        return None
    center = _bbox_center(_bbox(target))
    return min(
        range(len(indicators)),
        key=lambda idx: _distance(center, _bbox_center(_bbox(indicators[idx]))),
    )


def _visible_bbox(
    frame: np.ndarray,
    bbox: list[int],
    role_value: int,
) -> list[int]:
    x, y, w, h = bbox
    sub = frame[y : y + h, x : x + w]
    ys, xs = np.where(sub == role_value)
    if len(xs) == 0:
        return list(bbox)
    x0, x1 = int(xs.min()) + x, int(xs.max()) + x
    y0, y1 = int(ys.min()) + y, int(ys.max()) + y
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _selected_slider(
    sliders: list[Mapping[str, Any]], payload: Mapping[str, Any]
) -> tuple[int | None, str]:
    site = (int(payload.get("x", -1)), int(payload.get("y", -1)))
    for idx, slider in enumerate(sliders):
        if site == tuple(int(value) for value in slider["click"]):
            return idx, "grow"
        if site == tuple(int(value) for value in slider["shrink_click"]):
            return idx, "shrink"
    return None, ""


def _descendants(start: int, links: Mapping[int, int | None]) -> list[int]:
    out: list[int] = []
    seen = {start}
    current = links.get(start)
    while current is not None and int(current) not in seen:
        current = int(current)
        out.append(current)
        seen.add(current)
        current = links.get(current)
    return out


def _growth_direction(rotation: int) -> tuple[int, int]:
    return {
        0: (0, -1),
        90: (1, 0),
        180: (0, 1),
        270: (-1, 0),
    }[rotation]


def _growth_side(direction: tuple[int, int]) -> str:
    return {
        (0, -1): "up",
        (1, 0): "right",
        (0, 1): "down",
        (-1, 0): "left",
    }[direction]


def _grown_bbox(piece: Mapping[str, Any], unit: int) -> list[int]:
    x, y, w, h = _bbox(piece)
    direction = _growth_direction(int(piece["rotation"]))
    if direction == (0, -1):
        return [x, y - unit, w, h + unit]
    if direction == (1, 0):
        return [x, y, w + unit, h]
    if direction == (0, 1):
        return [x, y, w, h + unit]
    return [x - unit, y, w + unit, h]


def _resized_bbox(
    piece: Mapping[str, Any], unit: int, mode: str
) -> list[int]:
    if mode == "grow":
        return _grown_bbox(piece, unit)
    x, y, width, height = _bbox(piece)
    direction = _growth_direction(int(piece["rotation"]))
    if direction == (0, -1):
        return [x, y + unit, width, height - unit]
    if direction == (1, 0):
        return [x, y, width - unit, height]
    if direction == (0, 1):
        return [x, y, width, height - unit]
    return [x + unit, y, width - unit, height]


def _translate_bbox(bbox: Any, delta: tuple[int, int]) -> list[int]:
    x, y, w, h = [int(value) for value in bbox]
    return [x + delta[0], y + delta[1], w, h]


def _contact_follower_rows(
    entities: list[Mapping[str, Any]],
    *,
    controlled_count: int,
    extent_rows: list[Mapping[str, Any]],
    linked_rows: list[Mapping[str, Any]],
    unit: int,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Bind passive bodies carried by a moving edge or rigid contact."""

    followers: list[dict[str, Any]] = []
    follower_indices: set[int] = set()
    rigid_sources = [dict(row) for row in linked_rows]
    for entity_idx in range(int(controlled_count), len(entities)):
        entity = entities[entity_idx]
        x, y, width, height = _bbox(entity)
        candidates: list[tuple[int, int]] = []
        for extent in extent_rows:
            from_x, from_y, from_width, from_height = [
                int(value) for value in extent.get("from_bbox", [])
            ]
            side = str(extent.get("side", ""))
            delta_values = [int(value) for value in extent.get("delta", [])]
            if len(delta_values) != 2:
                continue
            delta = (delta_values[0], delta_values[1])
            horizontal_overlap = min(x + width, from_x + from_width) - max(x, from_x)
            vertical_overlap = min(y + height, from_y + from_height) - max(y, from_y)
            touching = bool(
                (side == "up" and height == unit and y + height == from_y and horizontal_overlap >= unit)
                or (
                    side == "down"
                    and height == unit
                    and y == from_y + from_height
                    and horizontal_overlap >= unit
                )
                or (
                    side == "left"
                    and width == unit
                    and x + width == from_x
                    and vertical_overlap >= unit
                )
                or (
                    side == "right"
                    and width == unit
                    and x == from_x + from_width
                    and vertical_overlap >= unit
                )
            )
            if touching:
                candidates.append(delta)
        for source in rigid_sources:
            from_x, from_y, from_width, from_height = [
                int(value) for value in source.get("from_bbox", [])
            ]
            delta_values = [int(value) for value in source.get("delta", [])]
            if len(delta_values) != 2:
                continue
            horizontal_overlap = min(x + width, from_x + from_width) - max(x, from_x)
            vertical_overlap = min(y + height, from_y + from_height) - max(y, from_y)
            delta_x, delta_y = delta_values
            horizontal_endpoint_drag = bool(
                delta_x != 0
                and delta_y == 0
                and height == unit
                and from_height > unit
                and (
                    y + height == from_y
                    or y == from_y + from_height
                )
                and horizontal_overlap >= unit
                and (
                    x == from_x
                    or x + width == from_x + from_width
                )
            )
            vertical_endpoint_drag = bool(
                delta_y != 0
                and delta_x == 0
                and width == unit
                and from_width > unit
                and (
                    x + width == from_x
                    or x == from_x + from_width
                )
                and vertical_overlap >= unit
                and (
                    y == from_y
                    or y + height == from_y + from_height
                )
            )
            flush_contact = bool(
                (
                    delta_y < 0
                    and height == unit
                    and y + height == from_y
                    and horizontal_overlap >= unit
                )
                or (
                    delta_y > 0
                    and height == unit
                    and y == from_y + from_height
                    and horizontal_overlap >= unit
                )
                or (
                    delta_x < 0
                    and width == unit
                    and x + width == from_x
                    and vertical_overlap >= unit
                )
                or (
                    delta_x > 0
                    and width == unit
                    and x == from_x + from_width
                    and vertical_overlap >= unit
                )
                or horizontal_endpoint_drag
                or vertical_endpoint_drag
            )
            if flush_contact:
                candidates.append((delta_values[0], delta_values[1]))
        unique_deltas = sorted(set(candidates))
        if len(unique_deltas) != 1:
            continue
        delta = unique_deltas[0]
        followers.append(
            {
                "entity": _entity_id(entity_idx, controlled_count),
                "from_bbox": [x, y, width, height],
                "to_bbox": _translate_bbox([x, y, width, height], delta),
                "delta": [int(delta[0]), int(delta[1])],
                "rotation": (
                    None
                    if entity.get("rotation") is None
                    else int(entity["rotation"])
                ),
                "relation": "moving_edge_follower",
            }
        )
        follower_indices.add(entity_idx)
    return followers, follower_indices


def _bbox(item: Mapping[str, Any]) -> list[int]:
    return [int(item[key]) for key in ("x", "y", "w", "h")]


def _bbox_center(bbox: Any) -> tuple[float, float]:
    x, y, w, h = [float(value) for value in bbox]
    return (x + (w - 1.0) / 2.0, y + (h - 1.0) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _has_piece(pieces: list[Mapping[str, Any]], bbox: Any, rotation: Any) -> bool:
    expected_bbox = [int(value) for value in list(bbox or [])]
    return any(
        _bbox(piece) == expected_bbox and int(piece["rotation"]) == int(rotation)
        for piece in pieces
    )


def _has_entity(entities: list[Mapping[str, Any]], bbox: Any, rotation: Any) -> bool:
    expected_bbox = [int(value) for value in list(bbox or [])]
    for entity in entities:
        if _bbox(entity) != expected_bbox:
            continue
        observed_rotation = entity.get("rotation")
        if rotation is None and observed_rotation is None:
            return True
        if rotation is not None and observed_rotation is not None:
            if int(observed_rotation) == int(rotation):
                return True
    return False


def _has_bbox(items: list[Mapping[str, Any]], bbox: Any) -> bool:
    expected_bbox = [int(value) for value in list(bbox or [])]
    return any(_bbox(item) == expected_bbox for item in items)


def _has_exact_extent_observation(
    packet: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    extent: Mapping[str, Any] | None = None,
) -> bool:
    unit = int(expected.get("inferred_growth_unit", 0) or 0)
    extent_row = dict(extent or expected.get("extent_delta") or {})
    axis = str(extent_row.get("axis", ""))
    side = str(extent_row.get("side", ""))
    for row in list(packet.get("growth_observations") or []):
        if not isinstance(row, Mapping) or row.get("growth_axis") != axis:
            continue
        major = (
            int(row.get("width_delta", 0) or 0)
            if axis == "horizontal"
            else int(row.get("height_delta", 0) or 0)
        )
        orthogonal = (
            int(row.get("height_delta", 0) or 0)
            if axis == "horizontal"
            else int(row.get("width_delta", 0) or 0)
        )
        if major == unit and orthogonal == 0 and list(row.get("growth_sides") or []) == [side]:
            return True
    return False


def _has_exact_target_translation(
    packet: Mapping[str, Any],
    expected_target: Mapping[str, Any],
    *,
    reverse: bool = False,
) -> bool:
    before_key = "to_visible_bbox" if reverse else "from_visible_bbox"
    after_key = "from_visible_bbox" if reverse else "to_visible_bbox"
    before = _inclusive_bbox(expected_target.get(before_key))
    after = _inclusive_bbox(expected_target.get(after_key))
    delta = [float(value) for value in list(expected_target.get("delta") or [])]
    if reverse:
        delta = [-value for value in delta]
    for row in list(packet.get("attached_translation_observations") or []):
        if not isinstance(row, Mapping):
            continue
        observed_delta = [float(value) for value in list(row.get("delta") or [])]
        if (
            list(row.get("before_bbox") or []) == before
            and list(row.get("after_bbox") or []) == after
            and observed_delta == delta
        ):
            return True
    return False


def _has_decreasing_distance_evidence(
    packet: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    required = len(expected.get("target_distance_evidence") or [])
    improved = [
        row
        for row in list(packet.get("target_distance_observations") or [])
        if isinstance(row, Mapping)
        and bool(row.get("distance_improved"))
        and bool(row.get("movement_aligned_with_marker"))
    ]
    return required > 0 and len(improved) >= required


def _inclusive_bbox(bbox: Any) -> list[int]:
    x, y, w, h = [int(value) for value in list(bbox or [])]
    return [x, y, x + w - 1, y + h - 1]


def _action_payload(action: Any) -> dict[str, Any]:
    if isinstance(action, Mapping):
        name = str(action.get("action", "")).upper().split(".")[-1]
        return {
            "action": name,
            "x": int(action.get("x", -1)),
            "y": int(action.get("y", -1)),
        }
    return {"action": str(action).upper().split(".")[-1], "x": -1, "y": -1}


def _action_names(valid_actions: Any) -> set[str]:
    if isinstance(valid_actions, str):
        values = [valid_actions]
    else:
        try:
            values = list(valid_actions)
        except TypeError:
            values = [valid_actions]
    return {str(value).upper().split(".")[-1] for value in values}


def _prediction_rejection(action: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": f"{_SCHEMA_VERSION}.transition",
        "family_id": FAMILY_ID,
        "accepted": False,
        "action": dict(action),
        "reject_reason": str(reason),
    }


def _verification(
    accepted: bool,
    reasons: list[str],
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": f"{_SCHEMA_VERSION}.verification",
        "family_id": FAMILY_ID,
        "accepted": bool(accepted),
        "rejected_reasons": list(reasons),
        "evidence": dict(evidence or {}),
    }


def _entity_name(row: Any) -> str:
    return str(row.get("entity", "unknown")) if isinstance(row, Mapping) else "invalid"


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _as_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError("linked extent mechanics require a 2-D frame")
    return arr.astype(np.int16, copy=False)


__all__ = [
    "FAMILY_ID",
    "bind_current",
    "compile_schema",
    "predict_transition",
    "propose_plan",
    "verify_transition",
]
