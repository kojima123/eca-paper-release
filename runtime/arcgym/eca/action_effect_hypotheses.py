"""Build explicit ECA hypotheses from frame-only action-effect evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .action_effect_model import (
    CLICK_ACTIONS,
    DIRECTIONAL_ACTIONS,
    ActionEffectRecord,
    ActionEffectSummary,
    ActionEffectTracker,
)


@dataclass(frozen=True)
class ActionEffectHypothesis:
    hypothesis_id: str
    kind: str
    confidence: float
    action_names: list[str]
    status: str
    description: str
    supporting_records: int
    evidence: dict[str, Any] = field(default_factory=dict)
    next_probe_actions: list[str] = field(default_factory=list)
    primitive_implications: list[str] = field(default_factory=list)
    falsifiers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionEffectHypothesisBuilder:
    """Induce simple action-effect hypotheses from observed transitions."""

    def build(
        self,
        records: list[ActionEffectRecord],
        *,
        summary: ActionEffectSummary | None = None,
        rod_support: Mapping[str, Any] | None = None,
    ) -> list[ActionEffectHypothesis]:
        if summary is None:
            tracker = ActionEffectTracker(max_records=max(1, len(records)))
            tracker._records = list(records)
            tracker.record_count = len(records)
            summary = tracker.summary()

        hypotheses: list[ActionEffectHypothesis] = []
        hypotheses.extend(self._movement_hypotheses(summary))
        hypotheses.extend(self._click_hypotheses(summary))
        hypotheses.extend(self._score_hypotheses(summary))
        hypotheses.extend(self._noop_hypotheses(summary))
        hypotheses.extend(self._mixed_transition_hypotheses(summary))
        hypotheses.extend(self._rod_supported_hypotheses(rod_support or {}))
        hypotheses.sort(key=lambda item: (-item.confidence, item.kind, item.hypothesis_id))
        return hypotheses

    def _movement_hypotheses(self, summary: ActionEffectSummary) -> list[ActionEffectHypothesis]:
        action_stats = summary.action_stats
        directional = [action_stats[action] for action in sorted(DIRECTIONAL_ACTIONS & set(action_stats))]
        if not directional:
            return []
        attempts = sum(stats.attempts for stats in directional)
        changed = sum(stats.changed_count for stats in directional)
        localized = sum(stats.localized_change_rate * stats.attempts for stats in directional)
        changed_rate = float(changed) / float(attempts) if attempts else 0.0
        localized_rate = float(localized) / float(attempts) if attempts else 0.0
        confidence = min(1.0, 0.50 * changed_rate + 0.50 * localized_rate)
        if confidence < 0.35:
            return []
        observed_actions = [stats.action_name for stats in directional if stats.changed_count > 0]
        missing = [action for action in sorted(DIRECTIONAL_ACTIONS) if action not in action_stats]
        return [
            ActionEffectHypothesis(
                hypothesis_id="directional_actions_move_visible_actor",
                kind="movement_effect",
                confidence=confidence,
                action_names=observed_actions,
                status="active" if confidence >= 0.70 and len(observed_actions) >= 2 else "tentative",
                description="directional actions produce localized visible changes",
                supporting_records=changed,
                evidence={
                    "directional_attempts": attempts,
                    "directional_changed_rate": changed_rate,
                    "directional_localized_rate": localized_rate,
                },
                next_probe_actions=missing,
                primitive_implications=["grid_bfs", "finite_state_action_search"],
                falsifiers=["directional actions become high no-op under repeated probes"],
            )
        ]

    def _click_hypotheses(self, summary: ActionEffectSummary) -> list[ActionEffectHypothesis]:
        action_stats = summary.action_stats
        stats = action_stats.get("ACTION6")
        if stats is None:
            return []
        confidence = min(
            1.0,
            0.55 * stats.changed_rate
            + 0.25 * min(1.0, stats.avg_changed_fraction / 0.08)
            + 0.20 * min(1.0, float(stats.score_event_count) / float(max(1, stats.attempts))),
        )
        if confidence < 0.30:
            return []
        return [
            ActionEffectHypothesis(
                hypothesis_id="click_action_has_visible_affordance",
                kind="click_affordance",
                confidence=confidence,
                action_names=["ACTION6"],
                status="active" if confidence >= 0.65 else "tentative",
                description="click action changes visible state or score",
                supporting_records=stats.changed_count + stats.score_event_count,
                evidence={
                    "click_attempts": stats.attempts,
                    "click_changed_rate": stats.changed_rate,
                    "click_avg_changed_fraction": stats.avg_changed_fraction,
                    "click_score_events": stats.score_event_count,
                },
                next_probe_actions=["ACTION6"],
                primitive_implications=["click_toggle_search", "click_affordance_probe"],
                falsifiers=["click probes at varied sites remain no-op"],
            )
        ]

    def _score_hypotheses(self, summary: ActionEffectSummary) -> list[ActionEffectHypothesis]:
        hypotheses: list[ActionEffectHypothesis] = []
        for action in summary.score_actions:
            stats = summary.action_stats.get(action)
            if stats is None:
                continue
            score_rate = float(stats.score_event_count) / float(max(1, stats.attempts))
            confidence = min(1.0, 0.70 * score_rate + 0.30 * (1.0 if stats.score_delta_total > 0 else 0.0))
            hypotheses.append(
                ActionEffectHypothesis(
                    hypothesis_id=f"{action.lower()}_score_bearing",
                    kind="score_bearing_action",
                    confidence=confidence,
                    action_names=[action],
                    status="active" if confidence >= 0.60 else "tentative",
                    description="action changed public score",
                    supporting_records=stats.score_event_count,
                    evidence={
                        "score_event_count": stats.score_event_count,
                        "score_delta_total": stats.score_delta_total,
                    },
                    next_probe_actions=[action],
                    primitive_implications=["score_delta_frontier_search"],
                    falsifiers=["same action from matching state hash fails to repeat score delta"],
                )
            )
        return hypotheses

    def _noop_hypotheses(self, summary: ActionEffectSummary) -> list[ActionEffectHypothesis]:
        hypotheses: list[ActionEffectHypothesis] = []
        for action, stats in sorted(summary.action_stats.items()):
            if stats.attempts <= 0 or stats.no_op_rate < 0.60:
                continue
            hypotheses.append(
                ActionEffectHypothesis(
                    hypothesis_id=f"{action.lower()}_mostly_blocked_or_noop",
                    kind="blocked_or_noop_action",
                    confidence=min(1.0, stats.no_op_rate),
                    action_names=[action],
                    status="active" if stats.no_op_rate >= 0.80 else "tentative",
                    description="action usually has no visible effect in observed states",
                    supporting_records=stats.no_op_count,
                    evidence={"attempts": stats.attempts, "no_op_rate": stats.no_op_rate},
                    next_probe_actions=[candidate for candidate in sorted(DIRECTIONAL_ACTIONS | CLICK_ACTIONS) if candidate != action],
                    primitive_implications=["action_loop_probe"],
                    falsifiers=["same action changes frame from a nearby state hash"],
                )
            )
        return hypotheses

    def _mixed_transition_hypotheses(self, summary: ActionEffectSummary) -> list[ActionEffectHypothesis]:
        changed_actions = list(summary.changed_actions)
        if len(changed_actions) < 2:
            return []
        confidence = min(1.0, float(len(changed_actions)) / float(max(1, len(summary.action_stats))))
        if confidence < 0.35:
            return []
        return [
            ActionEffectHypothesis(
                hypothesis_id="multiple_action_families_change_state",
                kind="mixed_transition_system",
                confidence=confidence,
                action_names=changed_actions,
                status="active" if confidence >= 0.60 else "tentative",
                description="multiple action families produce visible transitions",
                supporting_records=summary.changed_count,
                evidence={
                    "changed_actions": changed_actions,
                    "changed_count": summary.changed_count,
                    "records_count": summary.records_count,
                },
                next_probe_actions=[],
                primitive_implications=["finite_state_action_search"],
                falsifiers=["additional actions collapse into one dominant action family"],
            )
        ]

    def _rod_supported_hypotheses(self, rod_support: Mapping[str, Any]) -> list[ActionEffectHypothesis]:
        by_effect = rod_support.get("by_effect_type_verdict")
        if not isinstance(by_effect, Mapping):
            return []
        out: list[ActionEffectHypothesis] = []
        for effect_type, raw_counts in sorted(by_effect.items()):
            if not isinstance(raw_counts, Mapping):
                continue
            supported = int(raw_counts.get("supported", 0) or 0)
            contradicted = int(raw_counts.get("contradicted", 0) or 0)
            net = supported - contradicted
            if net <= 0:
                continue
            effect = str(effect_type or "")
            primitive_implications = _primitive_implications_for_rod_effect(effect)
            out.append(
                ActionEffectHypothesis(
                    hypothesis_id=f"rod_supported_{effect}",
                    kind="rod_supported_mechanic",
                    confidence=min(0.85, 0.42 + 0.08 * float(net)),
                    action_names=_actions_for_rod_support(rod_support, effect),
                    status="active" if net >= 2 else "tentative",
                    description="structural rod evidence supports a reusable mechanic",
                    supporting_records=supported,
                    evidence={
                        "effect_type": effect,
                        "supported": supported,
                        "contradicted": contradicted,
                        "net_support": net,
                    },
                    primitive_implications=primitive_implications,
                    falsifiers=["future matching rods contradict this mechanic"],
                )
            )
        return out


def _primitive_implications_for_rod_effect(effect_type: str) -> list[str]:
    effect = str(effect_type or "")
    if effect == "segment_transport":
        return ["segment_transport_probe"]
    if effect in {"click_transition", "toggle"}:
        return ["click_toggle_search", "click_affordance_probe"]
    if effect == "route_connectivity_delta":
        return ["grid_bfs", "segment_transport_probe"]
    if effect == "score_delta":
        return ["score_delta_frontier_search"]
    if effect == "noop":
        return ["action_loop_probe"]
    return ["finite_state_action_search"]


def _actions_for_rod_support(rod_support: Mapping[str, Any], effect_type: str) -> list[str]:
    by_action = rod_support.get("by_action")
    if not isinstance(by_action, Mapping):
        return []
    actions: list[str] = []
    for action, raw_counts in sorted(by_action.items()):
        if not isinstance(raw_counts, Mapping):
            continue
        if int(raw_counts.get("supported", 0) or 0) > int(raw_counts.get("contradicted", 0) or 0):
            actions.append(str(action))
    return actions[:6]


__all__ = [
    "ActionEffectHypothesis",
    "ActionEffectHypothesisBuilder",
]
