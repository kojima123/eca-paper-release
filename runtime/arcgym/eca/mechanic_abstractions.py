"""Game-agnostic mechanic slots inferred from action-effect evidence.

These bindings are intentionally not solvers.  They are the intermediate
mechanic vocabulary that lets ECA say "this looks like transport" or "this
looks like a toggle" before any per-game primitive is chosen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


DIRECTION_ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


@dataclass(frozen=True)
class MechanicBinding:
    mechanic_id: str
    confidence: float
    source_family: str
    slots: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    required_bindings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _first_target(target_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(target_candidates[0]) if target_candidates else {}


def _operator_names(operator_candidates: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for item in operator_candidates:
        if not isinstance(item, Mapping):
            continue
        operator = str(item.get("operator", "") or "")
        if operator and operator not in out:
            out.append(operator)
    return out


def _target_slot(target: Mapping[str, Any]) -> dict[str, Any]:
    if not target:
        return {"role": "goal_or_score_relevant_target", "binding_status": "unbound"}
    slot = {
        "role": "goal_or_score_relevant_target",
        "binding_status": "candidate",
        "kind": str(target.get("kind", "") or ""),
    }
    if isinstance(target.get("site"), (list, tuple)):
        site = target.get("site")
        if len(site) >= 2:
            try:
                slot["site"] = [int(site[0]), int(site[1])]
            except (TypeError, ValueError):
                pass
    if target.get("action_name"):
        slot["observed_action"] = str(target.get("action_name"))
    if target.get("source_site"):
        source_site = target.get("source_site")
        if isinstance(source_site, (list, tuple)) and len(source_site) >= 2:
            try:
                slot["source_site"] = [int(source_site[0]), int(source_site[1])]
            except (TypeError, ValueError):
                pass
    for key in (
        "segment_color",
        "dx",
        "dy",
        "direction_match",
        "consistency_count",
        "direction_match_rate",
    ):
        if key in target:
            slot[key] = target[key]
    return slot


def build_mechanic_bindings(
    *,
    solver_family: str,
    primitive_id: str,
    classification_label: str,
    archetype_scores: Mapping[str, float],
    target_candidates: list[dict[str, Any]],
    operator_candidates: list[dict[str, Any]],
    next_probe_actions: list[str],
    segment_motion_groups: list[dict[str, Any]] | None = None,
) -> list[MechanicBinding]:
    """Infer abstract mechanic slots for ControlIR.

    The output deliberately avoids game ids and level signatures.  Each binding
    describes the slots that still need frame-only evidence before a concrete
    solver can execute.
    """

    family = str(solver_family or "unknown")
    primitive = str(primitive_id or "unknown")
    label = str(classification_label or "unknown")
    scores = {str(key): float(value) for key, value in dict(archetype_scores or {}).items()}
    operators = _operator_names(operator_candidates)
    target = _first_target(target_candidates)
    target_slot = _target_slot(target)
    motion_groups = [dict(item) for item in list(segment_motion_groups or []) if isinstance(item, Mapping)]
    primary_motion_group = dict(motion_groups[0]) if motion_groups else {}
    bindings: list[MechanicBinding] = []

    push_score = scores.get("push_object_candidate", 0.0)
    attach_score = scores.get("attach_transport_candidate", 0.0)
    grid_score = scores.get("grid_navigation", scores.get("grid_navigation_candidate", 0.0))
    click_score = scores.get("click_toggle_candidate", scores.get("click_toggle", 0.0))
    overlay_score = scores.get("broad_overlay_candidate", 0.0)

    if family in {"navigation", "segment_transport", "push_object", "attach_transport"} or max(push_score, attach_score, grid_score) >= 0.35:
        if family == "push_object":
            object_role = "pushable_object"
            interaction = "push"
            confidence = max(push_score, 0.45)
        elif family == "attach_transport":
            object_role = "attachable_or_carried_object"
            interaction = "attach_then_transport"
            confidence = max(attach_score, 0.45)
        elif family == "segment_transport":
            object_role = "observed_moving_segment"
            interaction = "repeat_directional_transport"
            confidence = max(grid_score, 0.50)
        else:
            object_role = "actor_or_cursor"
            interaction = "route_to"
            confidence = max(grid_score, 0.35)
        bindings.append(
            MechanicBinding(
                mechanic_id="transport",
                confidence=_clamp(confidence),
                source_family=family,
                slots={
                    "object": {"role": object_role, "binding_status": "needs_perception"},
                    "source": {"role": "current_object_position", "binding_status": "frame"},
                    "target": target_slot,
                    "allowed_actions": list(DIRECTION_ACTIONS),
                    "interaction": interaction,
                    "motion_group": primary_motion_group,
                },
                constraints={
                    "requires_collision_model": family in {"push_object", "attach_transport"},
                    "requires_frame_only_binding": True,
                    "game_id_allowed": False,
                },
                required_bindings=["object", "source", "target", "transition_model"],
                evidence={
                    "primitive_id": primitive,
                    "classification_label": label,
                    "push_score": push_score,
                    "attach_score": attach_score,
                    "grid_score": grid_score,
                    "segment_motion_groups": motion_groups[:4],
                },
            )
        )

    if family in {"click_affordance", "overlay", "score_frontier"} or click_score >= 0.30 or overlay_score >= 0.45:
        action_name = "ACTION6"
        if target.get("action_name"):
            action_name = str(target.get("action_name"))
        bindings.append(
            MechanicBinding(
                mechanic_id="toggle",
                confidence=_clamp(max(click_score, overlay_score, 0.40)),
                source_family=family,
                slots={
                    "state_var": {
                        "role": "visible_region_or_score_precondition",
                        "binding_status": "needs_effect_memory",
                    },
                    "target_value": {"role": "changed_or_score_increasing", "binding_status": "goal"},
                    "action_to_modify": action_name,
                    "payload": {"role": "click_site_or_action_payload", "binding_status": "candidate"},
                    "target": target_slot,
                },
                constraints={
                    "requires_reversibility_check": family != "score_frontier",
                    "requires_frame_only_binding": True,
                    "game_id_allowed": False,
                },
                required_bindings=["state_var", "action_to_modify", "target_value"],
                evidence={
                    "primitive_id": primitive,
                    "classification_label": label,
                    "click_score": click_score,
                    "overlay_score": overlay_score,
                },
            )
        )

    selector_like = any(operator in {"click", "attach", "push"} for operator in operators)
    modifier_actions = [action for action in next_probe_actions if str(action).startswith("ACTION")]
    has_directional_modifier = any(action in DIRECTION_ACTIONS for action in modifier_actions)
    if family in {"push_object", "attach_transport"} or (
        selector_like and bool(target) and has_directional_modifier
    ):
        bindings.append(
            MechanicBinding(
                mechanic_id="select_then_act",
                confidence=_clamp(max(push_score, attach_score, click_score, 0.35)),
                source_family=family,
                slots={
                    "selector": {
                        "role": "object_or_region_selector",
                        "candidate_actions": [action for action in ("ACTION5", "ACTION6") if action in modifier_actions]
                        or ["ACTION5", "ACTION6"],
                    },
                    "modifier": {
                        "role": "post_selection_effect_action",
                        "candidate_actions": modifier_actions or list(DIRECTION_ACTIONS),
                    },
                    "selected_object": {"role": "selected_or_attached_object", "binding_status": "unverified"},
                    "target": target_slot,
                },
                constraints={
                    "requires_selection_persistence": True,
                    "requires_frame_only_binding": True,
                    "game_id_allowed": False,
                },
                required_bindings=["selector", "selected_object", "modifier", "target"],
                evidence={
                    "primitive_id": primitive,
                    "classification_label": label,
                    "operators": operators,
                },
            )
        )

    bindings.sort(key=lambda item: (-item.confidence, item.mechanic_id))
    return bindings


__all__ = ["MechanicBinding", "build_mechanic_bindings"]
