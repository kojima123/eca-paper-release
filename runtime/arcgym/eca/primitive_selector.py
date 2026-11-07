"""Select generic solver/probe primitives from action-effect evidence."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .action_effect_model import ActionEffectSummary, PrimitiveClassification
from .action_effect_hypotheses import ActionEffectHypothesis
from .mechanic_abstractions import build_mechanic_bindings
from .primitive_scope_registry import primitive_scope


@dataclass(frozen=True)
class PrimitiveCandidate:
    primitive_id: str
    priority: int
    confidence: float
    reason: str
    source_labels: list[str] = field(default_factory=list)
    next_probe_actions: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrimitiveSelection:
    primary_primitive: str
    confidence: float
    should_probe: bool
    candidates: list[PrimitiveCandidate]
    uncertainty_reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    control_target: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return payload


class SolverPrimitiveSelector:
    """Turn transition classification into ordered primitive hints.

    This is the first layer where ECA starts choosing tools from evidence. It
    still does not solve games; it decides which generic primitive or probe
    should be attempted next.
    """

    def select(
        self,
        classification: PrimitiveClassification,
        *,
        summary: ActionEffectSummary | None = None,
        hypotheses: list[ActionEffectHypothesis] | None = None,
        score_transition_kinds: dict[str, int] | None = None,
        rod_support: dict[str, Any] | None = None,
        outcome_prior_bias: Any | None = None,
        outcome_prior_context: Any | None = None,
    ) -> PrimitiveSelection:
        label = str(classification.primary_label or "unknown_transition_game")
        scores = dict(classification.scores or {})
        evidence = dict(classification.evidence or {})
        score_transition_kinds = {
            str(kind): int(count)
            for kind, count in sorted(dict(score_transition_kinds or {}).items())
            if int(count) > 0
        }
        changed_actions = list(evidence.get("changed_actions") or [])
        score_actions = list(evidence.get("score_actions") or [])
        directional_changed_rate = float(evidence.get("directional_changed_rate", 0.0) or 0.0)
        click_changed_rate = float(evidence.get("click_changed_rate", 0.0) or 0.0)
        click_fraction = float(evidence.get("click_avg_changed_fraction", 0.0) or 0.0)
        no_op_rate = float(evidence.get("overall_no_op_rate", 0.0) or 0.0)
        object_continuity = dict(evidence.get("object_continuity") or {})
        segment_motion_score = float(object_continuity.get("segment_motion_score", 0.0) or 0.0)
        consistent_segment_motion_score = float(
            object_continuity.get("consistent_segment_motion_score", 0.0) or 0.0
        )
        segment_motion_groups = _consistent_segment_motion_groups(object_continuity)
        target_candidates = [dict(candidate) for candidate in list(evidence.get("target_candidates") or [])]
        archetype_scores = {
            str(key): float(value)
            for key, value in dict(evidence.get("solver_archetype_scores") or {}).items()
        }
        click_transition_context = (
            float(archetype_scores.get("click_toggle", 0.0) or 0.0) >= 0.45
            or click_changed_rate >= 0.35
        )
        hypothesis_kinds = [hypothesis.kind for hypothesis in list(hypotheses or [])]

        candidates: list[PrimitiveCandidate] = []
        uncertainty: list[str] = []

        for hypothesis in list(hypotheses or []):
            if hypothesis.kind != "score_bearing_action":
                continue
            candidates.append(
                PrimitiveCandidate(
                    primitive_id="score_delta_frontier_search",
                    priority=5,
                    confidence=confidence_from_hypothesis(hypothesis),
                    reason="score-bearing action hypothesis has priority over structural exploration",
                    source_labels=[label, hypothesis.kind],
                    next_probe_actions=list(hypothesis.action_names),
                    required_evidence=_score_required_evidence(score_transition_kinds),
                )
            )

        if segment_motion_groups:
            first_group = segment_motion_groups[0]
            action_name = _normalize_action_name(first_group.get("action_name"))
            segment_priority = (
                45
                if click_transition_context
                else
                28
                if _strong_grid_navigation_context(
                    label,
                    changed_actions=changed_actions,
                    archetype_scores=archetype_scores,
                )
                else 8
            )
            candidates.append(
                PrimitiveCandidate(
                    primitive_id="segment_transport_probe",
                    priority=segment_priority,
                    confidence=min(0.90, max(0.55, consistent_segment_motion_score, segment_motion_score)),
                    reason="repeated action-aligned segment motion can bind a generic transport slot",
                    source_labels=[label, "repeated_segment_motion"],
                    next_probe_actions=[action_name] if action_name else [],
                    required_evidence=[
                        "consistent_segment_motion",
                        "current_segment_site",
                        "target_or_goal_relation",
                        "obstacle_map",
                    ],
                )
            )

        if label == "movement_game":
            candidates.append(
                PrimitiveCandidate(
                    primitive_id="grid_bfs",
                    priority=10,
                    confidence=max(scores.get("movement_game", 0.0), classification.confidence),
                    reason="directional actions consistently produce localized frame changes",
                    source_labels=[label, "localized_transition_game"],
                    next_probe_actions=_missing_directional_probes(changed_actions),
                    required_evidence=["actor_location", "goal_like_change_or_score"],
                )
            )
            if click_changed_rate > 0.0:
                candidates.append(
                    PrimitiveCandidate(
                        primitive_id="click_affordance_probe",
                        priority=40,
                        confidence=min(0.75, click_changed_rate),
                        reason="click action also changed the frame; probe clickable affordances before planning",
                        source_labels=[label, "click_toggle_game"],
                        next_probe_actions=["ACTION6"],
                        required_evidence=["click_target_bbox", "effect_reversibility"],
                    )
                )
        elif label == "click_toggle_game":
            candidates.append(
                PrimitiveCandidate(
                    primitive_id="click_toggle_search",
                    priority=10,
                    confidence=max(scores.get("click_toggle_game", 0.0), classification.confidence),
                    reason="click action changes visible state or score",
                    source_labels=[label],
                    next_probe_actions=["ACTION6"],
                    required_evidence=["clickable_regions", "toggle_graph", "score_delta_targets"],
                )
            )
            if click_fraction >= 0.08:
                candidates.append(
                    PrimitiveCandidate(
                        primitive_id="overlay_matcher",
                        priority=30,
                        confidence=min(0.80, click_fraction * 4.0),
                        reason="click effect is broad enough to look like canvas or overlay manipulation",
                        source_labels=[label],
                        next_probe_actions=["ACTION6"],
                        required_evidence=["changed_region_shape", "target_template"],
                    )
                )
        elif label == "multi_action_transition_game":
            candidates.append(
                PrimitiveCandidate(
                    primitive_id="finite_state_action_search",
                    priority=20,
                    confidence=max(scores.get("multi_action_transition_game", 0.0), classification.confidence),
                    reason="multiple action families alter visible state",
                    source_labels=[label],
                    next_probe_actions=_untried_core_actions(changed_actions),
                    required_evidence=["state_hash_graph", "goal_or_score_frontier"],
                )
            )
            if directional_changed_rate > 0.35:
                candidates.append(
                    PrimitiveCandidate(
                        primitive_id="grid_bfs",
                        priority=25,
                        confidence=min(0.85, directional_changed_rate),
                        reason="directional transitions are present inside a mixed-action game",
                        source_labels=[label, "movement_game"],
                        next_probe_actions=_missing_directional_probes(changed_actions),
                        required_evidence=["actor_location", "obstacle_map"],
                    )
                )
            if click_changed_rate > 0.35:
                candidates.append(
                    PrimitiveCandidate(
                        primitive_id="click_toggle_search",
                        priority=30,
                        confidence=min(0.85, click_changed_rate),
                        reason="click transitions are present inside a mixed-action game",
                        source_labels=[label, "click_toggle_game"],
                        next_probe_actions=["ACTION6"],
                        required_evidence=["clickable_regions", "toggle_effects"],
                    )
                )
        elif label == "score_action_game":
            candidates.append(
                PrimitiveCandidate(
                    primitive_id="score_delta_frontier_search",
                    priority=10,
                    confidence=max(scores.get("score_action_game", 0.0), classification.confidence),
                    reason="one or more probed actions directly changed score",
                    source_labels=[label],
                    next_probe_actions=score_actions,
                    required_evidence=_score_required_evidence(score_transition_kinds),
                )
            )
        elif label == "low_effect_or_noop":
            candidates.append(
                PrimitiveCandidate(
                    primitive_id="action_loop_probe",
                    priority=50,
                    confidence=max(scores.get("low_effect_or_noop", 0.0), classification.confidence),
                    reason="observed transitions are mostly no-ops; gather broader evidence before selecting a solver",
                    source_labels=[label],
                    next_probe_actions=_untried_core_actions(changed_actions),
                    required_evidence=["valid_action_set", "alternative_click_sites"],
                )
            )
            uncertainty.append("insufficient visible transition signal")
        else:
            candidates.append(
                PrimitiveCandidate(
                    primitive_id="action_loop_probe",
                    priority=60,
                    confidence=max(0.10, min(0.40, classification.confidence)),
                    reason="classification confidence is too low for solver selection",
                    source_labels=[label],
                    next_probe_actions=_untried_core_actions(changed_actions),
                    required_evidence=["additional_action_effect_records"],
                )
            )
            uncertainty.append("unknown transition family")

        _append_archetype_candidates(candidates, archetype_scores, label)

        if not candidates:
            candidates.append(
                PrimitiveCandidate(
                    primitive_id="action_loop_probe",
                    priority=99,
                    confidence=0.10,
                    reason="no candidate matched the observed evidence",
                    source_labels=[label],
                    next_probe_actions=_untried_core_actions(changed_actions),
                    required_evidence=["additional_action_effect_records"],
                )
            )
            uncertainty.append("empty primitive candidate set")

        if summary is not None and summary.records_count < 4:
            uncertainty.append("fewer than four action-effect records")
        if no_op_rate >= 0.60 and label != "low_effect_or_noop":
            uncertainty.append("high no-op rate weakens primitive confidence")

        strong_grid_context = _strong_grid_navigation_context(
            label,
            changed_actions=changed_actions,
            archetype_scores=archetype_scores,
        )
        control_target_candidates = (
            _prioritize_click_transition_targets(target_candidates)
            if click_transition_context
            else
            _demote_segment_targets(target_candidates)
            if strong_grid_context
            else target_candidates
        )
        candidates, rod_adjustment = _apply_rod_support(candidates, rod_support)
        outcome_prior_bias_diag: dict[str, Any] = {"enabled": False}
        if outcome_prior_bias is not None:
            candidates, _diag = outcome_prior_bias.apply(
                candidates, outcome_prior_context
            )
            outcome_prior_bias_diag = _diag.to_dict()
        ordered = sorted(candidates, key=lambda item: (item.priority, -item.confidence, item.primitive_id))
        primary = ordered[0]
        should_probe = bool(uncertainty or primary.primitive_id.endswith("_probe") or primary.primitive_id == "action_loop_probe")
        return PrimitiveSelection(
            primary_primitive=primary.primitive_id,
            confidence=float(primary.confidence),
            should_probe=should_probe,
            candidates=ordered,
            uncertainty_reasons=uncertainty,
            evidence={
                "classification_label": label,
                "classification_confidence": float(classification.confidence),
                "changed_actions": changed_actions,
                "score_actions": score_actions,
                "score_transition_kinds": score_transition_kinds,
                "overall_no_op_rate": no_op_rate,
                "object_continuity": object_continuity,
                "segment_motion_score": segment_motion_score,
                "consistent_segment_motion_score": consistent_segment_motion_score,
                "segment_motion_consistency_groups": segment_motion_groups,
                "solver_archetype_label": evidence.get("solver_archetype_label"),
                "solver_archetype_scores": archetype_scores,
                "target_candidates": target_candidates,
                "hypothesis_kinds": hypothesis_kinds,
                "rod_support": dict(rod_support or {}),
                "rod_adjustment": rod_adjustment,
                "outcome_prior_bias": outcome_prior_bias_diag,
            },
            control_target=_control_target_from_candidate(
                primary,
                label=label,
                archetype_scores=archetype_scores,
                target_candidates=control_target_candidates,
                segment_motion_groups=segment_motion_groups,
                uncertainty=uncertainty,
            ),
        )


def _missing_directional_probes(changed_actions: list[str]) -> list[str]:
    changed = {str(action).upper() for action in changed_actions}
    missing = [action for action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") if action not in changed]
    return missing


def _untried_core_actions(changed_actions: list[str]) -> list[str]:
    changed = {str(action).upper() for action in changed_actions}
    return [action for action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6") if action not in changed]


def _apply_rod_support(
    candidates: list[PrimitiveCandidate],
    rod_support: dict[str, Any] | None,
) -> tuple[list[PrimitiveCandidate], dict[str, Any]]:
    empty_adjustment = {
        "enabled": bool((rod_support or {}).get("enable_selector_bias")),
        "inserted_count": 0,
        "boosted_count": 0,
        "suppressed_count": 0,
        "runtime_mechanic_template_count": int(
            (rod_support or {}).get("runtime_mechanic_template_count", 0) or 0
        ),
        "mechanic_template_inserted_count": 0,
        "mechanic_template_boosted_count": 0,
        "inserted_primitives": [],
        "suppressed_primitives": [],
        "mechanic_template_inserted_primitives": [],
        "mechanic_template_boosted_primitives": [],
    }
    if not candidates or not rod_support:
        return candidates, empty_adjustment
    if not bool(rod_support.get("enable_selector_bias")):
        return candidates, empty_adjustment
    by_primitive = dict(rod_support.get("by_primitive") or {})
    by_solver_family = dict(rod_support.get("by_solver_family") or {})
    by_effect_type = dict(rod_support.get("by_effect_type_verdict") or {})
    if not (by_primitive or by_solver_family or by_effect_type):
        return candidates, empty_adjustment
    inserted = _rod_candidate_insertions(candidates, rod_support)
    inserted_ids = [candidate.primitive_id for candidate in inserted]
    working = list(candidates) + inserted
    out: list[PrimitiveCandidate] = []
    boosted_ids: list[str] = []
    suppressed_ids: list[str] = []
    mechanic_boosted_ids: list[str] = []
    for candidate in working:
        primitive_counts = dict(by_primitive.get(candidate.primitive_id) or {})
        family = _solver_family_for_primitive(candidate.primitive_id)
        family_counts = dict(by_solver_family.get(family) or {})
        effect_counts = _rod_effect_counts_for_candidate(candidate.primitive_id, by_effect_type)
        direct_supported = (
            int(primitive_counts.get("supported", 0) or 0)
            + int(family_counts.get("supported", 0) or 0)
        )
        direct_contradicted = (
            int(primitive_counts.get("contradicted", 0) or 0)
            + int(family_counts.get("contradicted", 0) or 0)
        )
        supported = (
            direct_supported
            + int(effect_counts.get("supported", 0) or 0)
        )
        contradicted = (
            direct_contradicted
            + int(effect_counts.get("contradicted", 0) or 0)
        )
        net_support = supported - contradicted
        if net_support == 0:
            out.append(candidate)
            continue
        anxiety = float(rod_support.get("expected_anxiety_delta", 0.0) or 0.0)
        goal_delta = float(rod_support.get("expected_goal_delta", 0.0) or 0.0)
        direct_net = direct_supported - direct_contradicted
        strong_contradiction = bool(net_support < 0 and direct_net <= -6)
        priority_delta = (
            min(12, max(3, abs(net_support) * 2))
            if strong_contradiction
            else min(3, max(1, abs(net_support) // 2))
            if net_support < 0 or direct_net > 0
            else 1
        )
        priority_delta = (
            min(1, priority_delta)
            if net_support > 0 and direct_net <= 0
            else priority_delta
        )
        confidence_bonus = min(0.10, 0.015 * float(max(0, supported)) + 0.02 * max(0.0, anxiety + goal_delta))
        confidence_penalty = (
            min(0.35, 0.05 * float(max(0, contradicted)) + 0.04 * max(0.0, -anxiety))
            if strong_contradiction
            else min(0.12, 0.02 * float(max(0, contradicted)) + 0.02 * max(0.0, -anxiety))
        )
        mechanic_sources = _runtime_mechanic_sources_for_candidate(
            candidate.primitive_id,
            rod_support,
        )
        source_labels = list(candidate.source_labels)
        required_evidence = list(candidate.required_evidence)
        if mechanic_sources and net_support > 0:
            source_labels = _dedupe_strings(
                [*source_labels, "runtime_mechanic_template", *mechanic_sources]
            )
            required_evidence = _dedupe_strings(
                [*required_evidence, "runtime_mechanic_template"]
            )
            mechanic_boosted_ids.append(candidate.primitive_id)
        if net_support > 0:
            priority = max(0, int(candidate.priority) - priority_delta)
            confidence = min(0.99, float(candidate.confidence) + confidence_bonus)
            reason = f"{candidate.reason}; rod_support={net_support}"
            if mechanic_sources:
                reason = f"{reason}; mechanic_template"
            boosted_ids.append(candidate.primitive_id)
        else:
            priority = int(candidate.priority) + priority_delta
            confidence = max(0.01, float(candidate.confidence) - confidence_penalty)
            suffix = "rod_family_suppression" if strong_contradiction else "rod_contradiction"
            reason = f"{candidate.reason}; {suffix}={abs(net_support)}"
            suppressed_ids.append(candidate.primitive_id)
        out.append(
            replace(
                candidate,
                priority=priority,
                confidence=confidence,
                reason=reason,
                source_labels=source_labels,
                required_evidence=required_evidence,
            )
        )
    mechanic_inserted_ids = [
        candidate.primitive_id
        for candidate in inserted
        if "runtime_mechanic_template" in set(candidate.source_labels)
    ]
    adjustment = {
        "enabled": True,
        "inserted_count": len(inserted_ids),
        "boosted_count": len(boosted_ids),
        "suppressed_count": len(suppressed_ids),
        "runtime_mechanic_template_count": int(
            rod_support.get("runtime_mechanic_template_count", 0) or 0
        ),
        "mechanic_template_inserted_count": len(mechanic_inserted_ids),
        "mechanic_template_boosted_count": len(mechanic_boosted_ids),
        "inserted_primitives": inserted_ids,
        "boosted_primitives": boosted_ids,
        "suppressed_primitives": suppressed_ids,
        "mechanic_template_inserted_primitives": mechanic_inserted_ids,
        "mechanic_template_boosted_primitives": mechanic_boosted_ids,
    }
    return out, adjustment


def _rod_candidate_insertions(
    candidates: list[PrimitiveCandidate],
    rod_support: dict[str, Any],
) -> list[PrimitiveCandidate]:
    existing = {str(candidate.primitive_id or "") for candidate in candidates}
    ordered_existing = sorted(candidates, key=lambda item: (item.priority, -item.confidence, item.primitive_id))
    current_primary = ordered_existing[0] if ordered_existing else None
    current_priority = int(current_primary.priority) if current_primary is not None else 99
    current_confidence = float(current_primary.confidence) if current_primary is not None else 0.0
    current_primitive = str(current_primary.primitive_id or "") if current_primary is not None else ""
    allow_preempt = bool(
        current_primitive in {"action_loop_probe", "finite_state_action_search"}
        or current_confidence < 0.35
    )
    by_solver_family = dict(rod_support.get("by_solver_family") or {})
    by_effect_type = dict(rod_support.get("by_effect_type_verdict") or {})
    requested: dict[str, tuple[int, float, str, list[str], list[str]]] = {}

    def add(
        primitive_id: str,
        *,
        net_support: int,
        reason: str,
        source_labels: list[str],
        actions: list[str] | None = None,
    ) -> None:
        if primitive_id in existing or primitive_id in requested or net_support < 2:
            return
        mechanic_sources = _runtime_mechanic_sources_for_candidate(primitive_id, rod_support)
        if mechanic_sources:
            source_labels = _dedupe_strings(
                [*source_labels, "runtime_mechanic_template", *mechanic_sources]
            )
        raw_priority = max(2, 18 - min(12, int(net_support) * 2))
        priority = raw_priority if allow_preempt else max(current_priority + 1, raw_priority)
        confidence = min(0.86, 0.42 + 0.06 * float(net_support))
        required_evidence = ["rod_supported_mechanic", "current_state_binding"]
        if mechanic_sources:
            required_evidence.append("runtime_mechanic_template")
            reason = f"{reason}; mechanic_template"
        requested[primitive_id] = (
            priority,
            confidence,
            reason,
            list(actions or []),
            source_labels,
            required_evidence,
        )

    for family, raw_counts in sorted(by_solver_family.items()):
        if not isinstance(raw_counts, dict):
            continue
        net = int(raw_counts.get("supported", 0) or 0) - int(raw_counts.get("contradicted", 0) or 0)
        primitive = _default_primitive_for_solver_family(str(family))
        if primitive:
            add(
                primitive,
                net_support=net,
                reason=f"rod-supported solver family inserted candidate: {family}",
                source_labels=["rod_support", f"solver_family:{family}"],
            )

    for effect_type, raw_counts in sorted(by_effect_type.items()):
        if not isinstance(raw_counts, dict):
            continue
        net = int(raw_counts.get("supported", 0) or 0) - int(raw_counts.get("contradicted", 0) or 0)
        for primitive in _default_primitives_for_effect_type(str(effect_type)):
            add(
                primitive,
                net_support=net,
                reason=f"rod-supported effect inserted candidate: {effect_type}",
                source_labels=["rod_support", f"effect:{effect_type}"],
                actions=_actions_from_rod_support(rod_support),
            )

    out = [
        PrimitiveCandidate(
            primitive_id=primitive_id,
            priority=priority,
            confidence=confidence,
            reason=reason,
            source_labels=source_labels,
            next_probe_actions=actions,
            required_evidence=required_evidence,
        )
        for primitive_id, (
            priority,
            confidence,
            reason,
            actions,
            source_labels,
            required_evidence,
        ) in requested.items()
    ]
    out.sort(key=lambda item: (item.priority, -item.confidence, item.primitive_id))
    return out


def _default_primitive_for_solver_family(solver_family: str) -> str:
    family = str(solver_family or "")
    return {
        "navigation": "grid_bfs",
        "segment_transport": "segment_transport_probe",
        "click_affordance": "click_toggle_search",
        "transition_probe": "finite_state_action_search",
        "score_frontier": "score_delta_frontier_search",
        "push_object": "select_push_probe",
        "attach_transport": "attach_transport_probe",
        "overlay": "overlay_matcher",
    }.get(family, "")


def _default_primitives_for_effect_type(effect_type: str) -> tuple[str, ...]:
    effect = str(effect_type or "")
    if effect == "segment_transport":
        return ("segment_transport_probe",)
    if effect in {"click_transition", "toggle"}:
        return ("click_toggle_search",)
    if effect == "route_connectivity_delta":
        return ("grid_bfs",)
    if effect == "score_delta":
        return ("score_delta_frontier_search",)
    if effect == "unexplained_effect":
        return ("finite_state_action_search",)
    return ()


def _actions_from_rod_support(rod_support: dict[str, Any]) -> list[str]:
    by_action = dict(rod_support.get("by_action") or {})
    scored: list[tuple[int, str]] = []
    for action, raw_counts in by_action.items():
        if not isinstance(raw_counts, dict):
            continue
        net = int(raw_counts.get("supported", 0) or 0) - int(raw_counts.get("contradicted", 0) or 0)
        if net > 0:
            scored.append((net, str(action)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [action for _net, action in scored[:4]]


def _runtime_mechanic_template_keys(rod_support: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for item in list(rod_support.get("episode_mechanic_candidates") or []):
        if not isinstance(item, dict):
            continue
        if not bool(item.get("runtime_selection_eligible")):
            continue
        key_kind = str(item.get("key_kind", "") or "")
        key = str(item.get("key", "") or "")
        if not key:
            continue
        if key_kind == "generic_key":
            keys.add(key)
        elif key_kind == "effect_type":
            keys.add(f"effect:{key}")
        elif key_kind == "solver_family":
            keys.add(f"solver_family:{key}")
    return keys


def _runtime_mechanic_sources_for_candidate(
    primitive_id: str,
    rod_support: dict[str, Any],
) -> list[str]:
    template_keys = _runtime_mechanic_template_keys(rod_support)
    if not template_keys:
        return []
    labels: list[str] = []
    family = _solver_family_for_primitive(primitive_id)
    if family and f"solver_family:{family}" in template_keys:
        labels.append(f"mechanic_template:solver_family:{family}")
    for effect_type in _expected_effect_types_for_primitive(primitive_id):
        if f"effect:{effect_type}" in template_keys:
            labels.append(f"mechanic_template:effect:{effect_type}")
    return _dedupe_strings(labels)


def _dedupe_strings(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "")
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _rod_effect_counts_for_candidate(
    primitive_id: str,
    by_effect_type: dict[str, Any],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for effect_type in _expected_effect_types_for_primitive(primitive_id):
        raw = by_effect_type.get(effect_type)
        if not isinstance(raw, dict):
            continue
        for verdict, value in raw.items():
            counts[str(verdict)] += int(value or 0)
    return counts


def _expected_effect_types_for_primitive(primitive_id: str) -> tuple[str, ...]:
    primitive = str(primitive_id or "")
    if primitive == "segment_transport_probe":
        return ("segment_transport", "route_connectivity_delta")
    if primitive in {"click_toggle_search", "click_affordance_probe", "click_context_recovery"}:
        return ("click_transition", "toggle", "score_delta")
    if primitive in {"grid_bfs", "click_opened_region_navigation"}:
        return ("route_connectivity_delta", "score_delta")
    if primitive == "score_delta_frontier_search":
        return ("score_delta",)
    if primitive in {"select_push_probe", "select_push_cover_bfs", "generic_select_push_cover_bfs"}:
        return ("segment_transport", "route_connectivity_delta")
    if primitive in {"attach_transport_probe", "attach_pair_fill_bfs", "generic_attach_pair_transport_bfs"}:
        return ("segment_transport", "route_connectivity_delta")
    if primitive == "overlay_matcher":
        return ("click_transition", "unexplained_effect")
    return ("unexplained_effect", "noop")


def _strong_grid_navigation_context(
    label: str,
    *,
    changed_actions: list[str],
    archetype_scores: dict[str, float],
) -> bool:
    if str(label or "") != "movement_game":
        return False
    directional_changed = {
        _normalize_action_name(action)
        for action in changed_actions
        if _normalize_action_name(action) in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
    }
    if len(directional_changed) < 2:
        return False
    return float(archetype_scores.get("grid_navigation", 0.0) or 0.0) >= 0.85


def _demote_segment_targets(target_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for candidate in target_candidates:
        item = dict(candidate)
        if str(item.get("kind", "") or "") in {
            "segment_displacement",
            "score_segment_displacement",
        }:
            item["grid_navigation_demoted"] = True
        candidates.append(item)
    return sorted(
        candidates,
        key=lambda candidate: (
            1
            if str(candidate.get("kind", "") or "") in {
                "segment_displacement",
                "score_segment_displacement",
            }
            else 0
        ),
    )


def _prioritize_click_transition_targets(target_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [dict(candidate) for candidate in target_candidates]
    return sorted(
        candidates,
        key=lambda candidate: (
            0 if str(candidate.get("kind", "") or "") == "click_point" else 1,
            1
            if str(candidate.get("kind", "") or "") in {
                "segment_displacement",
                "score_segment_displacement",
            }
            else 0,
            -float(candidate.get("goal_likelihood", 0.0) or 0.0),
        ),
    )


def _normalize_action_name(action_name: Any) -> str:
    value = str(action_name or "").upper()
    if value.startswith("ACTION"):
        return value
    if value.isdigit():
        return f"ACTION{value}"
    return value


def _score_required_evidence(score_transition_kinds: dict[str, int]) -> list[str]:
    if score_transition_kinds.get("broad_level_transition", 0) > 0:
        return ["repeatability", "level_boundary_context", "post_level_replan"]
    if score_transition_kinds.get("local_score_transition", 0) > 0:
        return ["repeatability", "local_precondition_hashes", "target_object_context"]
    if score_transition_kinds.get("score_counter_only", 0) > 0:
        return ["repeatability", "score_counter_preconditions"]
    return ["repeatability", "precondition_hashes"]


def _consistent_segment_motion_groups(object_continuity: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for raw in list(object_continuity.get("segment_motion_consistency_groups") or []):
        if not isinstance(raw, dict):
            continue
        if int(raw.get("direction_match_count", 0) or 0) < 2:
            continue
        if float(raw.get("direction_match_rate", 0.0) or 0.0) < 0.5:
            continue
        groups.append(dict(raw))
    groups.sort(
        key=lambda item: (
            int(item.get("direction_match_count", 0) or 0),
            float(item.get("direction_match_rate", 0.0) or 0.0),
            float(item.get("mean_action_alignment", 0.0) or 0.0),
            int(item.get("last_step", 0) or 0),
        ),
        reverse=True,
    )
    return groups[:8]


def _append_archetype_candidates(
    candidates: list[PrimitiveCandidate],
    archetype_scores: dict[str, float],
    label: str,
) -> None:
    existing = {candidate.primitive_id for candidate in candidates}
    push_score = float(archetype_scores.get("push_object_candidate", 0.0) or 0.0)
    attach_score = float(archetype_scores.get("attach_transport_candidate", 0.0) or 0.0)
    overlay_score = float(archetype_scores.get("broad_overlay_candidate", 0.0) or 0.0)

    if push_score >= 0.45 and "select_push_probe" not in existing:
        candidates.append(
            PrimitiveCandidate(
                primitive_id="select_push_probe",
                priority=35,
                confidence=min(0.80, push_score),
                reason="action-effect archetype suggests localized movement plus object-manipulation transitions",
                source_labels=[label, "push_object_candidate"],
                next_probe_actions=["ACTION5", "ACTION6"],
                required_evidence=["movable_object_delta", "target_or_cover_relation", "push_feasibility"],
            )
        )

    if attach_score >= 0.45 and "attach_transport_probe" not in existing:
        candidates.append(
            PrimitiveCandidate(
                primitive_id="attach_transport_probe",
                priority=36,
                confidence=min(0.80, attach_score),
                reason="action-effect archetype suggests non-directional attachment plus directional transport",
                source_labels=[label, "attach_transport_candidate"],
                next_probe_actions=["ACTION5", "ACTION6"],
                required_evidence=["attachable_pair_delta", "carried_object_continuity", "drop_target_relation"],
            )
        )

    if overlay_score >= 0.45 and "overlay_matcher" not in existing:
        candidates.append(
            PrimitiveCandidate(
                primitive_id="overlay_matcher",
                priority=38,
                confidence=min(0.80, overlay_score),
                reason="action-effect archetype suggests broad visual overlay changes",
                source_labels=[label, "broad_overlay_candidate"],
                next_probe_actions=["ACTION6"],
                required_evidence=["changed_region_shape", "target_template"],
            )
        )


def _control_target_from_candidate(
    candidate: PrimitiveCandidate,
    *,
    label: str,
    archetype_scores: dict[str, float],
    target_candidates: list[dict[str, Any]],
    segment_motion_groups: list[dict[str, Any]],
    uncertainty: list[str],
) -> dict[str, Any]:
    solver_family = _solver_family_for_primitive(candidate.primitive_id)
    operator_candidates = _operator_candidates_for_target(
        solver_family=solver_family,
        primitive_id=candidate.primitive_id,
        target_candidates=target_candidates,
        archetype_scores=archetype_scores,
    )
    probe_site_candidates = _probe_site_candidates_for_target(candidate, solver_family, target_candidates)
    mechanic_bindings = [
        binding.to_dict()
        for binding in build_mechanic_bindings(
            solver_family=solver_family,
            primitive_id=candidate.primitive_id,
            classification_label=label,
            archetype_scores=archetype_scores,
            target_candidates=target_candidates,
            operator_candidates=operator_candidates,
            next_probe_actions=list(candidate.next_probe_actions),
            segment_motion_groups=segment_motion_groups,
        )
    ]
    return {
        "schema_version": "eca_primitive_control_target.v1",
        "primitive_id": candidate.primitive_id,
        "primitive_scope": primitive_scope(candidate.primitive_id),
        "solver_family": solver_family,
        "classification_label": label,
        "confidence": float(candidate.confidence),
        "requires_probe": bool(candidate.primitive_id.endswith("_probe")),
        "next_probe_actions": list(candidate.next_probe_actions),
        "required_evidence": list(candidate.required_evidence),
        "source_labels": list(candidate.source_labels),
        "archetype_scores": dict(sorted(archetype_scores.items())),
        "target_candidates": target_candidates[:8],
        "primary_target": dict(target_candidates[0]) if target_candidates else {},
        "segment_motion_consistency_groups": segment_motion_groups[:8],
        "primary_segment_motion_group": dict(segment_motion_groups[0]) if segment_motion_groups else {},
        "probe_site_candidates": probe_site_candidates,
        "primary_probe_site": dict(probe_site_candidates[0]) if probe_site_candidates else {},
        "operator_candidates": operator_candidates,
        "primary_operator": operator_candidates[0]["operator"] if operator_candidates else "probe",
        "mechanic_bindings": mechanic_bindings,
        "primary_mechanic": mechanic_bindings[0]["mechanic_id"] if mechanic_bindings else "",
        "uncertainty_reasons": list(uncertainty),
    }


def _probe_site_candidates_for_target(
    candidate: PrimitiveCandidate,
    solver_family: str,
    target_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build concrete probe candidates without taking over execution.

    This keeps ControlIR actionable enough to audit while avoiding the earlier
    regression from directly forcing click payloads.
    """

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int, str]] = set()

    def add(
        *,
        operator: str,
        action_name: str,
        site: list[int] | tuple[int, int] | None,
        reason: str,
        source: dict[str, Any] | None = None,
        confidence: float = 0.30,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if site is not None:
            try:
                x = int(site[0])
                y = int(site[1])
            except (TypeError, ValueError, IndexError):
                return
        else:
            x = y = -1
        action = _normalize_action_name(action_name)
        key = (str(operator), action, x, y, str((source or {}).get("kind", "")))
        if key in seen:
            return
        seen.add(key)
        item: dict[str, Any] = {
            "operator": str(operator),
            "action_name": action,
            "confidence": round(max(0.0, min(1.0, float(confidence))), 6),
            "reason": reason,
        }
        if site is not None:
            item["site"] = [x, y]
        if payload:
            item["payload"] = dict(payload)
        if source:
            item["source_kind"] = str(source.get("kind", "") or "")
            item["goal_likelihood"] = float(source.get("goal_likelihood", 0.0) or 0.0)
            item["changed_fraction"] = float(source.get("changed_fraction", 0.0) or 0.0)
            if source.get("relation_kind"):
                item["relation_kind"] = str(source.get("relation_kind", "") or "")
            if "score_delta" in source:
                item["score_delta"] = int(source.get("score_delta", 0) or 0)
            item["actionable"] = _probe_site_actionable(item)
        out.append(item)

    for target in target_candidates[:8]:
        if not isinstance(target, dict):
            continue
        site = target.get("site")
        kind = str(target.get("kind", "") or "")
        action_name = _normalize_action_name(target.get("action_name"))
        goal_likelihood = float(target.get("goal_likelihood", 0.0) or 0.0)
        confidence = 0.35 + max(-0.20, min(0.40, goal_likelihood * 0.25))

        if kind == "click_point" or solver_family in {"click_affordance", "overlay"}:
            payload = None
            if isinstance(site, (list, tuple)) and len(site) >= 2:
                try:
                    payload = {"x": int(site[0]), "y": int(site[1])}
                except (TypeError, ValueError):
                    payload = None
            add(
                operator="click",
                action_name="ACTION6",
                site=site if isinstance(site, (list, tuple)) else None,
                payload=payload,
                source=target,
                confidence=max(0.45, confidence + (0.15 if kind == "click_point" else 0.0)),
                reason="target candidate can be tested as an explicit click payload",
            )

        if action_name:
            add(
                operator="repeat_action",
                action_name=action_name,
                site=site if isinstance(site, (list, tuple)) else None,
                source=target,
                confidence=max(0.30, confidence),
                reason="target candidate carries an observed action-effect action",
            )

        if isinstance(site, (list, tuple)) and solver_family in {
            "navigation",
            "segment_transport",
            "push_object",
            "attach_transport",
            "transition_probe",
            "score_frontier",
        }:
            operator = "push" if solver_family == "push_object" else "route_to"
            add(
                operator=operator,
                action_name=action_name or "ACTION1",
                site=site,
                source=target,
                confidence=max(0.25, confidence - 0.05),
                reason="localized target can be routed toward before operator verification",
            )

    for action_name in candidate.next_probe_actions:
        action = _normalize_action_name(action_name)
        if not action:
            continue
        add(
            operator="probe",
            action_name=action,
            site=None,
            confidence=max(0.20, float(candidate.confidence) * 0.50),
            reason="primitive candidate requested this probe action",
        )

    out.sort(
        key=lambda item: (
            0 if bool(item.get("actionable")) else 1,
            -float(item.get("confidence", 0.0) or 0.0),
            0 if item.get("site") else 1,
            str(item.get("operator", "")),
            str(item.get("action_name", "")),
        )
    )
    return out[:12]


def _probe_site_actionable(item: Mapping[str, Any]) -> bool:
    operator = str(item.get("operator", "") or "")
    if operator != "click":
        return True
    return (
        int(item.get("score_delta", 0) or 0) > 0
        or float(item.get("goal_likelihood", 0.0) or 0.0) > 0.0
        or float(item.get("changed_fraction", 0.0) or 0.0) > 0.0
    )


def _operator_candidates_for_target(
    *,
    solver_family: str,
    primitive_id: str,
    target_candidates: list[dict[str, Any]],
    archetype_scores: dict[str, float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    target_kind = str((target_candidates[0] if target_candidates else {}).get("kind", "") or "")

    def add(operator: str, confidence: float, reason: str) -> None:
        candidates.append(
            {
                "operator": operator,
                "confidence": round(max(0.0, min(1.0, float(confidence))), 6),
                "reason": reason,
            }
        )

    if solver_family == "click_affordance":
        add("click", 0.80, "click-affordance solver family selected")
        add("route_to", 0.35, "route toward click-effect context if actor is displaced")
    elif solver_family == "push_object":
        add("push", max(0.45, archetype_scores.get("push_object_candidate", 0.0)), "push archetype selected")
        add("route_to", 0.45, "route adjacent before push")
        add("click", 0.25, "click may select pushable object")
    elif solver_family == "attach_transport":
        add("attach", max(0.45, archetype_scores.get("attach_transport_candidate", 0.0)), "attach archetype selected")
        add("drop", 0.40, "drop/settle after carrying target")
        add("route_to", 0.35, "route to attach/drop site")
    elif solver_family == "segment_transport":
        add("route_to", 0.78, "repeated segment motion exposes a current transport target")
        add("repeat_action", 0.70, "repeat observed directional action to verify transport continuity")
        add("probe", 0.35, "target still needs goal relation verification")
    elif solver_family == "overlay":
        add("click", 0.65, "overlay-like target usually needs click/toggle")
    elif solver_family == "score_frontier":
        add("repeat_action", 0.80, "score-bearing action should be retested")
        add("route_to", 0.30, "route back to score precondition if visible")
    elif solver_family == "navigation":
        add("route_to", 0.70, "navigation family selected")
    elif solver_family == "transition_probe":
        if float(archetype_scores.get("click_toggle", 0.0) or 0.0) >= 0.45:
            add("click", 0.76, "click-transition evidence should not be hidden by route targets")
        elif target_kind == "click_point":
            add("click", 0.45, "target candidate is a click point")
        add("route_to", 0.40, "transition probe has localized target candidate")
        add("probe", 0.30, "operator remains uncertain")
    else:
        add("probe", 0.30, "unknown solver family")

    if target_kind in {"segment_displacement", "score_segment_displacement"}:
        add("route_to", 0.74, "target candidate is a repeated moved segment")
        add("repeat_action", 0.68, "repeat action can verify segment transport")
    if target_kind == "score_changed_bbox":
        add("repeat_action", 0.70, "target candidate is tied to score transition")
    candidates.sort(key=lambda item: (-float(item["confidence"]), str(item["operator"])))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        operator = str(item["operator"])
        if operator in seen:
            continue
        seen.add(operator)
        deduped.append(item)
    return deduped


def _solver_family_for_primitive(primitive_id: str) -> str:
    primitive = str(primitive_id or "")
    if primitive in {"grid_bfs", "mirror_joint_cursor_bfs", "stride_marker_maze_bfs"}:
        return "navigation"
    if primitive == "segment_transport_probe":
        return "segment_transport"
    if primitive in {"click_toggle_search", "click_affordance_probe", "click_context_recovery"}:
        return "click_affordance"
    if primitive in {"select_push_probe", "select_push_cover_bfs", "generic_select_push_cover_bfs"}:
        return "push_object"
    if primitive in {"attach_transport_probe", "attach_pair_fill_bfs", "generic_attach_pair_transport_bfs"}:
        return "attach_transport"
    if primitive == "overlay_matcher":
        return "overlay"
    if primitive in {"finite_state_action_search", "action_loop_probe"}:
        return "transition_probe"
    if primitive == "score_delta_frontier_search":
        return "score_frontier"
    return "unknown"


def confidence_from_hypothesis(hypothesis: ActionEffectHypothesis) -> float:
    return max(0.0, min(1.0, float(hypothesis.confidence)))


__all__ = [
    "PrimitiveCandidate",
    "PrimitiveSelection",
    "SolverPrimitiveSelector",
]
