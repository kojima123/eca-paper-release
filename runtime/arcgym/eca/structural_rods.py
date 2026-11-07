"""Reusable causal evidence units for ECA primitive synthesis.

Structural rods intentionally avoid game ids, level signatures, and adapter
advice. They compress one observe -> act -> verify cycle into generic evidence
that can be reused by the next primitive-selection cycle.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class EffectEvent:
    effect_type: str
    action_name: str
    score_delta: int = 0
    changed_pixel_count: int = 0
    segment_motion: list[dict[str, Any]] = field(default_factory=list)
    route_connectivity_delta: dict[str, Any] = field(default_factory=dict)
    no_op: bool = False
    binding_source: str = "action_effect"
    objective_progress: bool = False
    objective_progress_kind: str = "none"
    goal_progress: bool = False
    goal_progress_kind: str = "none"
    goal_aligned_progress: bool = False
    goal_aligned_progress_kind: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalTraceRod:
    primitive_id: str
    solver_family: str
    operator: str
    action_name: str
    target_kind: str
    effect: EffectEvent
    verdict: str
    generic_keys: tuple[str, ...]
    before_sig: dict[str, Any] = field(default_factory=dict)
    after_sig: dict[str, Any] = field(default_factory=dict)
    target_binding: dict[str, Any] = field(default_factory=dict)
    predicted_effect: dict[str, Any] | None = None
    observed_effect: dict[str, Any] = field(default_factory=dict)
    goal_delta: float = 0.0
    support_count: int = 1
    contradiction_count: int = 0
    anxiety_delta: float = 0.0
    prediction_error: float = 0.0
    target_key: tuple[str, int, int, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["effect"] = self.effect.to_dict()
        payload["generic_keys"] = list(self.generic_keys)
        payload["target_key"] = list(self.target_key) if self.target_key else None
        payload["before_sig"] = _sanitize_payload(self.before_sig)
        payload["after_sig"] = _sanitize_payload(self.after_sig)
        payload["target_binding"] = _sanitize_payload(self.target_binding)
        payload["predicted_effect"] = _sanitize_payload(self.predicted_effect or {})
        payload["observed_effect"] = _sanitize_payload(self.observed_effect or self.effect.to_dict())
        payload["metadata"] = _sanitize_payload(self.metadata)
        return payload


@dataclass(frozen=True)
class HypothesisRod:
    """Unverified causal-pass hypothesis kept separate from evidence rods.

    Hypothesis rods can guide bounded probe bookkeeping, but they are not
    selector support and are never eligible for macro action preemption. A
    hypothesis only becomes executable evidence after the normal action-effect
    verifier creates a CausalTraceRod.
    """

    hypothesis_id: str
    source: str = "causal_pass"
    status: str = "unverified"
    effect_type: str = "hypothesis_update"
    proposed_cause: str = ""
    predicted_effect: str = ""
    candidate_site: dict[str, Any] = field(default_factory=dict)
    probe_actions: tuple[str, ...] = ()
    confidence_delta: float = 0.0
    created_step: int = -1
    generic_keys: tuple[str, ...] = ()
    commitment_level: str = "observe_only"
    execution_eligible: bool = False
    max_probe_steps: int = 0
    bailout_conditions: tuple[str, ...] = ()
    spatial_prior: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    selector_eligible: bool = False
    action_preemption_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": str(self.hypothesis_id or ""),
            "source": str(self.source or "causal_pass"),
            "status": str(self.status or "unverified"),
            "effect_type": str(self.effect_type or "hypothesis_update"),
            "proposed_cause": str(self.proposed_cause or ""),
            "predicted_effect": str(self.predicted_effect or ""),
            "candidate_site": _sanitize_payload(self.candidate_site),
            "probe_actions": [str(action) for action in self.probe_actions],
            "confidence_delta": float(self.confidence_delta),
            "created_step": int(self.created_step),
            "generic_keys": list(self.generic_keys),
            "commitment_level": str(self.commitment_level or "observe_only"),
            "execution_eligible": bool(self.execution_eligible),
            "max_probe_steps": int(self.max_probe_steps),
            "bailout_conditions": [str(item) for item in self.bailout_conditions],
            "spatial_prior": _sanitize_payload(self.spatial_prior),
            "metadata": _sanitize_payload(self.metadata),
            "selector_eligible": False,
            "action_preemption_eligible": False,
        }


@dataclass(frozen=True)
class RodPromotionPolicy:
    """Quantitative gates for hypothesis/evidence/mechanic promotion."""

    evidence_min_supporting_observations: int = 1
    evidence_max_contradictions: int = 0
    mechanic_min_support_count: int = 3
    mechanic_min_net_support: int = 2
    mechanic_min_distinct_episodes: int = 2
    mechanic_min_distinct_variants: int = 2
    mechanic_max_contradiction_rate: float = 0.25
    mechanic_min_delayed_positive_rate: float = 0.50
    mechanic_max_adapter_leakage_score: float = 0.0
    demote_on_contradiction_rate: float = 0.50
    retire_on_contradiction_count: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def evaluate_mechanic_summary(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        support = _as_int(summary.get("support_count", 0), 0)
        contradictions = _as_int(summary.get("contradiction_count", 0), 0)
        episodes = _as_int(summary.get("distinct_episode_count", 0), 0)
        variants = _as_int(summary.get("distinct_variant_count", 0), 0)
        delayed_rate = _as_float(summary.get("delayed_positive_rate", 0.0), 0.0)
        leakage = _as_float(summary.get("adapter_leakage_score", 0.0), 0.0)
        total = max(1, support + contradictions)
        contradiction_rate = float(contradictions) / float(total)
        net_support = support - contradictions
        eligible = (
            support >= self.mechanic_min_support_count
            and net_support >= self.mechanic_min_net_support
            and episodes >= self.mechanic_min_distinct_episodes
            and variants >= self.mechanic_min_distinct_variants
            and contradiction_rate <= self.mechanic_max_contradiction_rate
            and delayed_rate >= self.mechanic_min_delayed_positive_rate
            and leakage <= self.mechanic_max_adapter_leakage_score
        )
        retired = contradictions >= self.retire_on_contradiction_count
        demoted = retired or contradiction_rate >= self.demote_on_contradiction_rate
        return {
            "eligible_for_mechanic_promotion": bool(eligible),
            "demote": bool(demoted and not eligible),
            "retire": bool(retired and not eligible),
            "support_count": support,
            "contradiction_count": contradictions,
            "net_support": net_support,
            "contradiction_rate": round(contradiction_rate, 6),
            "distinct_episode_count": episodes,
            "distinct_variant_count": variants,
            "delayed_positive_rate": round(delayed_rate, 6),
            "adapter_leakage_score": round(leakage, 6),
        }


@dataclass(frozen=True)
class MechanicRod:
    """Episode-local mechanic template inferred from repeated evidence rods."""

    key_kind: str
    key: str
    support_count: int
    contradiction_count: int
    inconclusive_count: int = 0
    net_support: int = 0
    contradiction_rate: float = 0.0
    delayed_positive_rate: float = 0.0
    delayed_positive_rate_source: str = "support_verdict_proxy"
    runtime_selection_eligible: bool = False
    eligible_for_mechanic_promotion: bool = False
    demote: bool = False
    retire: bool = False
    promotion_block_reasons: tuple[str, ...] = ()
    generic_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_kind": str(self.key_kind or ""),
            "key": str(self.key or ""),
            "support_count": int(self.support_count),
            "contradiction_count": int(self.contradiction_count),
            "inconclusive_count": int(self.inconclusive_count),
            "net_support": int(self.net_support),
            "contradiction_rate": round(float(self.contradiction_rate), 6),
            "delayed_positive_rate": round(float(self.delayed_positive_rate), 6),
            "delayed_positive_rate_source": str(self.delayed_positive_rate_source or ""),
            "runtime_selection_eligible": bool(self.runtime_selection_eligible),
            "eligible_for_mechanic_promotion": bool(self.eligible_for_mechanic_promotion),
            "demote": bool(self.demote),
            "retire": bool(self.retire),
            "promotion_block_reasons": [str(item) for item in self.promotion_block_reasons],
            "generic_keys": [str(item) for item in self.generic_keys],
        }


@dataclass(frozen=True)
class SpatialRodPosterior:
    row: int
    col: int
    prior_score: float = 0.5
    posterior_score: float = 0.5
    support_count: int = 0
    contradiction_count: int = 0
    neutral_count: int = 0
    last_updated_step: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": int(self.row),
            "col": int(self.col),
            "d_a_prior": round(float(self.prior_score), 6),
            "d_a_posterior": round(float(self.posterior_score), 6),
            "d_a_posterior_delta": round(float(self.posterior_score - self.prior_score), 6),
            "posterior_support_count": int(self.support_count),
            "posterior_contradiction_count": int(self.contradiction_count),
            "posterior_neutral_count": int(self.neutral_count),
            "last_updated_step": int(self.last_updated_step),
            "metadata": _sanitize_payload(self.metadata),
        }


class SpatialRodPosteriorMemory:
    """Episode-local spatial posterior coupling rods to D_A-style priors."""

    def __init__(self, *, max_cells: int = 256) -> None:
        self.max_cells = max(1, int(max_cells or 1))
        self._cells: dict[tuple[int, int], dict[str, Any]] = {}

    def clear(self) -> None:
        self._cells.clear()

    def update_prior(
        self,
        *,
        row: int,
        col: int,
        prior: Mapping[str, Any] | None,
        step: int = -1,
    ) -> dict[str, Any]:
        key = (int(row), int(col))
        cell = self._ensure_cell(key)
        prior_payload = _sanitize_payload(dict(prior or {}))
        prior_score = _extract_spatial_prior_score(prior_payload, default=float(cell["prior_score"]))
        cell["prior_score"] = max(float(cell["prior_score"]), float(prior_score))
        cell["last_updated_step"] = max(int(cell.get("last_updated_step", -1) or -1), int(step))
        metadata = dict(cell.get("metadata", {}) or {})
        metadata.update(prior_payload)
        cell["metadata"] = _sanitize_payload(metadata)
        self._trim()
        return self.prior_for(row=int(row), col=int(col))

    def observe(
        self,
        *,
        row: int,
        col: int,
        outcome: str,
        step: int,
    ) -> dict[str, Any]:
        key = (int(row), int(col))
        cell = self._ensure_cell(key)
        normalized = str(outcome or "").strip().lower()
        if normalized in {"visited_state_change", "supported", "score_delta", "state_changed"}:
            cell["support_count"] = int(cell.get("support_count", 0) or 0) + 1
        elif normalized in {"visited_no_effect", "noop", "contradicted", "no_effect"}:
            cell["contradiction_count"] = int(cell.get("contradiction_count", 0) or 0) + 1
        else:
            cell["neutral_count"] = int(cell.get("neutral_count", 0) or 0) + 1
        cell["last_updated_step"] = int(step)
        self._trim()
        return self.prior_for(row=int(row), col=int(col))

    def prior_for(self, *, row: int, col: int) -> dict[str, Any]:
        cell = self._cells.get((int(row), int(col)))
        if not cell:
            return {}
        posterior = self._cell_to_posterior(int(row), int(col), cell)
        payload = posterior.to_dict()
        payload["prior_source"] = "spatial_rod_posterior"
        return payload

    def summary(self) -> dict[str, Any]:
        rows = [
            self._cell_to_posterior(row, col, cell).to_dict()
            for (row, col), cell in sorted(self._cells.items())
        ]
        rows.sort(
            key=lambda item: (
                -float(item.get("d_a_posterior", 0.0) or 0.0),
                int(item.get("row", 0) or 0),
                int(item.get("col", 0) or 0),
            )
        )
        supported = sum(1 for row in rows if int(row.get("posterior_support_count", 0) or 0) > 0)
        contradicted = sum(
            1 for row in rows if int(row.get("posterior_contradiction_count", 0) or 0) > 0
        )
        return {
            "cell_count": len(rows),
            "supported_cell_count": supported,
            "contradicted_cell_count": contradicted,
            "top_cells": rows[:8],
        }

    def _ensure_cell(self, key: tuple[int, int]) -> dict[str, Any]:
        if key not in self._cells:
            self._cells[key] = {
                "prior_score": 0.5,
                "support_count": 0,
                "contradiction_count": 0,
                "neutral_count": 0,
                "last_updated_step": -1,
                "metadata": {},
            }
        return self._cells[key]

    def _trim(self) -> None:
        if len(self._cells) <= self.max_cells:
            return
        ranked = sorted(
            self._cells.items(),
            key=lambda item: int(item[1].get("last_updated_step", -1) or -1),
        )
        for key, _cell in ranked[: max(0, len(self._cells) - self.max_cells)]:
            self._cells.pop(key, None)

    @staticmethod
    def _cell_to_posterior(row: int, col: int, cell: Mapping[str, Any]) -> SpatialRodPosterior:
        prior = _clamp01(_as_float(cell.get("prior_score", 0.5), 0.5))
        support = _as_int(cell.get("support_count", 0), 0)
        contradiction = _as_int(cell.get("contradiction_count", 0), 0)
        neutral = _as_int(cell.get("neutral_count", 0), 0)
        posterior = _clamp01(prior + 0.22 * support - 0.28 * contradiction + 0.02 * neutral)
        return SpatialRodPosterior(
            row=int(row),
            col=int(col),
            prior_score=prior,
            posterior_score=posterior,
            support_count=support,
            contradiction_count=contradiction,
            neutral_count=neutral,
            last_updated_step=_as_int(cell.get("last_updated_step", -1), -1),
            metadata=_sanitize_payload(dict(cell.get("metadata", {}) or {})),
        )


class HypothesisRodMemory:
    """Bounded episode-local store for unverified causal-pass hypotheses."""

    def __init__(self, *, max_rods: int = 128) -> None:
        self.max_rods = max(1, int(max_rods or 1))
        self._rods: deque[HypothesisRod] = deque(maxlen=self.max_rods)

    def clear(self) -> None:
        self._rods.clear()

    @property
    def rods(self) -> tuple[HypothesisRod, ...]:
        return tuple(self._rods)

    def add(self, rod: HypothesisRod | None) -> None:
        if rod is None:
            return
        self._rods.append(rod)

    def extend(self, rods: Iterable[HypothesisRod]) -> None:
        for rod in rods:
            self.add(rod)

    def summary(self) -> dict[str, Any]:
        by_effect_type: Counter[str] = Counter()
        by_source: Counter[str] = Counter()
        by_status: Counter[str] = Counter()
        by_generic_key: Counter[str] = Counter()
        by_commitment_level: Counter[str] = Counter()
        for rod in self._rods:
            by_effect_type[str(rod.effect_type or "unknown")] += 1
            by_source[str(rod.source or "unknown")] += 1
            by_status[str(rod.status or "unknown")] += 1
            by_commitment_level[str(rod.commitment_level or "observe_only")] += 1
            for key in rod.generic_keys:
                by_generic_key[str(key)] += 1
        return {
            "hypothesis_rod_count": len(self._rods),
            "unverified_count": sum(1 for rod in self._rods if rod.status == "unverified"),
            "execution_eligible_count": sum(1 for rod in self._rods if rod.execution_eligible),
            "low_commitment_execution_eligible_count": sum(
                1
                for rod in self._rods
                if rod.execution_eligible and rod.commitment_level == "probe_low"
            ),
            "selector_eligible_count": 0,
            "action_preemption_eligible_count": 0,
            "by_effect_type": dict(sorted(by_effect_type.items())),
            "by_source": dict(sorted(by_source.items())),
            "by_status": dict(sorted(by_status.items())),
            "by_commitment_level": dict(sorted(by_commitment_level.items())),
            "by_generic_key": dict(sorted(by_generic_key.items())),
            "recent": [rod.to_dict() for rod in list(self._rods)[-8:]],
        }


class StructuralRodMemory:
    """Bounded episode-local store of generic causal evidence."""

    def __init__(self, *, max_rods: int = 256) -> None:
        self.max_rods = max(1, int(max_rods or 1))
        self._rods: deque[CausalTraceRod] = deque(maxlen=self.max_rods)
        # Per-rod insertion-step tracker (research branch
        # score-bearing-rod-tagging, 2026-05-22). Parallel maxlen
        # deque keeps the two structures aligned. -1 sentinel means
        # the caller didn't pass step (legacy usage).
        self._rod_steps: deque[int] = deque(maxlen=self.max_rods)
        self._reuse_count = 0
        self._selector_query_count = 0

    def clear(self) -> None:
        self._rods.clear()
        self._rod_steps.clear()
        self._reuse_count = 0
        self._selector_query_count = 0

    @property
    def rods(self) -> tuple[CausalTraceRod, ...]:
        return tuple(self._rods)

    @property
    def rod_steps(self) -> tuple[int, ...]:
        """Insertion-step indices parallel to .rods. Returns -1 for rods
        added without a step (legacy callers)."""
        return tuple(self._rod_steps)

    def iter_rods_with_steps(self):
        """Yield (rod, step) pairs in insertion order. step is -1 for
        legacy entries lacking explicit step tag."""
        return zip(self._rods, self._rod_steps)

    def add(self, rod: CausalTraceRod | None, *, step: int = -1) -> None:
        if rod is None:
            return
        self._rods.append(rod)
        self._rod_steps.append(int(step))

    def retrieve(
        self,
        *,
        primitive_id: str | None = None,
        solver_family: str | None = None,
        effect_types: Iterable[str] | None = None,
        generic_keys: Iterable[str] | None = None,
        record_reuse: bool = True,
    ) -> dict[str, Any]:
        """Summarize reusable rods for selection without game/layout keys."""

        wanted_effects = {str(item) for item in list(effect_types or []) if str(item)}
        wanted_keys = {str(item) for item in list(generic_keys or []) if str(item)}
        wanted_primitive = str(primitive_id or "")
        wanted_family = str(solver_family or "")
        by_primitive: dict[str, Counter[str]] = {}
        by_solver_family: dict[str, Counter[str]] = {}
        by_effect_type_verdict: dict[str, Counter[str]] = {}
        by_action: dict[str, Counter[str]] = {}
        matched_by_action: dict[str, Counter[str]] = {}
        matched_target_bindings: dict[str, dict[str, Any]] = {}
        by_generic_key: Counter[str] = Counter()
        by_generic_key_verdict: dict[str, Counter[str]] = {}
        by_effect_type: Counter[str] = Counter()
        matched_count = 0
        anxiety_total = 0.0
        goal_delta_total = 0.0
        for rod in self._rods:
            primitive = str(rod.primitive_id or "")
            family = str(rod.solver_family or "")
            effect_type = str(rod.effect.effect_type or "unknown")
            rod_keys = tuple(str(key) for key in rod.generic_keys)
            matched = (
                not (wanted_primitive or wanted_family or wanted_effects or wanted_keys)
                or (wanted_primitive and primitive == wanted_primitive)
                or (wanted_family and family == wanted_family)
                or (wanted_effects and effect_type in wanted_effects)
                or (wanted_keys and bool(wanted_keys.intersection(rod_keys)))
            )
            if matched:
                matched_count += 1
                anxiety_total += float(rod.anxiety_delta)
                goal_delta_total += float(rod.goal_delta)
                action = str(rod.action_name or rod.effect.action_name or "")
                if action:
                    matched_by_action.setdefault(action, Counter())[rod.verdict] += 1
                target_entry = _target_binding_entry_from_rod(rod)
                if target_entry is not None:
                    target_key = str(target_entry["target_key"])
                    bucket = matched_target_bindings.setdefault(
                        target_key,
                        {
                            "target_key": target_key,
                            "target_kind": str(target_entry.get("target_kind", "") or ""),
                            "operator": str(target_entry.get("operator", "") or ""),
                            "action_name": str(target_entry.get("action_name", "") or ""),
                            "site": list(target_entry.get("site") or []),
                            "effect_type": str(target_entry.get("effect_type", "") or ""),
                            "binding_source": str(target_entry.get("binding_source", "") or ""),
                            "supported": 0,
                            "contradicted": 0,
                            "inconclusive": 0,
                        },
                    )
                    verdict = str(rod.verdict or "inconclusive")
                    bucket[verdict] = int(bucket.get(verdict, 0) or 0) + 1
            by_effect_type[effect_type] += 1
            by_effect_type_verdict.setdefault(effect_type, Counter())[rod.verdict] += 1
            action = str(rod.action_name or rod.effect.action_name or "")
            if action:
                by_action.setdefault(action, Counter())[rod.verdict] += 1
            if primitive:
                bucket = by_primitive.setdefault(primitive, Counter())
                bucket[rod.verdict] += 1
            if family:
                bucket = by_solver_family.setdefault(family, Counter())
                bucket[rod.verdict] += 1
            for key in rod_keys:
                by_generic_key[key] += 1
                by_generic_key_verdict.setdefault(key, Counter())[rod.verdict] += 1
        support = {
            "rod_count": len(self._rods),
            "matched_rod_count": matched_count,
            "by_primitive": _counter_map_to_dict(by_primitive),
            "by_solver_family": _counter_map_to_dict(by_solver_family),
            "by_effect_type": dict(sorted(by_effect_type.items())),
            "by_effect_type_verdict": _counter_map_to_dict(by_effect_type_verdict),
            "by_action": _counter_map_to_dict(by_action),
            "matched_by_action": _counter_map_to_dict(matched_by_action),
            "matched_target_bindings": _target_binding_entries_to_list(matched_target_bindings),
            "by_generic_key": dict(sorted(by_generic_key.items())),
            "by_generic_key_verdict": _counter_map_to_dict(by_generic_key_verdict),
            "expected_anxiety_delta": (
                float(anxiety_total) / float(max(1, matched_count))
                if matched_count
                else 0.0
            ),
            "expected_goal_delta": (
                float(goal_delta_total) / float(max(1, matched_count))
                if matched_count
                else 0.0
            ),
        }
        mechanic_candidates = _episode_mechanic_rods(
            by_generic_key_verdict=by_generic_key_verdict,
            by_effect_type_verdict=by_effect_type_verdict,
            by_solver_family=by_solver_family,
        )
        support["episode_mechanic_candidate_count"] = len(mechanic_candidates)
        support["runtime_mechanic_template_count"] = sum(
            1 for candidate in mechanic_candidates if candidate.runtime_selection_eligible
        )
        support["mechanic_promotion_candidate_count"] = sum(
            1 for candidate in mechanic_candidates if candidate.eligible_for_mechanic_promotion
        )
        support["mechanic_promotion_blocked_count"] = sum(
            1
            for candidate in mechanic_candidates
            if candidate.runtime_selection_eligible and not candidate.eligible_for_mechanic_promotion
        )
        support["episode_mechanic_candidates"] = [
            candidate.to_dict() for candidate in mechanic_candidates[:20]
        ]
        if record_reuse:
            self._selector_query_count += 1
            if self._rods:
                self._reuse_count += 1
        support["reuse_count"] = self._reuse_count
        support["selector_query_count"] = self._selector_query_count
        support["reuse_rate"] = float(self._reuse_count) / float(max(1, self._selector_query_count))
        return support

    def selector_support(self, *, record_reuse: bool = True) -> dict[str, Any]:
        """Backward-compatible selector summary."""

        return self.retrieve(record_reuse=record_reuse)

    def summary(self) -> dict[str, Any]:
        support = self.selector_support(record_reuse=False)
        support["recent"] = [rod.to_dict() for rod in list(self._rods)[-8:]]
        support["contradiction_count"] = sum(rod.contradiction_count for rod in self._rods)
        support["support_count"] = sum(rod.support_count for rod in self._rods)
        support["auto_binding_count"] = sum(
            1
            for rod in self._rods
            if bool(rod.target_binding)
            and bool(rod.target_binding.get("target_kind") or rod.target_binding.get("target_key"))
            and str(rod.effect.binding_source or "") not in {"", "manual", "unknown"}
        )
        support["unexplained_effect_count"] = sum(
            1 for rod in self._rods if rod.effect.effect_type == "unexplained_effect"
        )
        support["unexplained_effect_rate"] = (
            float(support["unexplained_effect_count"]) / float(max(1, len(self._rods)))
        )
        return support


def build_effect_event(
    *,
    action_name: str,
    score_delta: int,
    changed_pixel_count: int,
    no_op: bool,
    target_kind: str,
    segment_motion: Iterable[Mapping[str, Any]] | None = None,
    route_connectivity_delta: Mapping[str, Any] | None = None,
    binding_source: str = "action_effect",
    objective_progress: bool = False,
    objective_progress_kind: str = "none",
    goal_progress: bool = False,
    goal_progress_kind: str = "none",
    goal_aligned_progress: bool = False,
    goal_aligned_progress_kind: str = "none",
) -> EffectEvent:
    effect_type = infer_effect_type(
        action_name=action_name,
        score_delta=score_delta,
        changed_pixel_count=changed_pixel_count,
        no_op=no_op,
        target_kind=target_kind,
        objective_progress=objective_progress,
        objective_progress_kind=objective_progress_kind,
        goal_progress=goal_progress,
        goal_progress_kind=goal_progress_kind,
    )
    return EffectEvent(
        effect_type=effect_type,
        action_name=str(action_name or ""),
        score_delta=int(score_delta),
        changed_pixel_count=int(changed_pixel_count),
        segment_motion=[dict(item) for item in list(segment_motion or []) if isinstance(item, Mapping)],
        route_connectivity_delta=dict(route_connectivity_delta or {}),
        no_op=bool(no_op),
        binding_source=str(binding_source or "action_effect"),
        objective_progress=bool(objective_progress),
        objective_progress_kind=str(objective_progress_kind or "none"),
        goal_progress=bool(goal_progress),
        goal_progress_kind=str(goal_progress_kind or "none"),
        goal_aligned_progress=bool(goal_aligned_progress),
        goal_aligned_progress_kind=str(goal_aligned_progress_kind or "none"),
    )


def build_causal_trace_rod(
    *,
    primitive_id: str,
    solver_family: str,
    operator: str,
    action_name: str,
    target_kind: str,
    target_key: Iterable[Any] | None,
    effect: EffectEvent,
    progressed: bool,
    before_sig: Mapping[str, Any] | None = None,
    after_sig: Mapping[str, Any] | None = None,
    target_binding: Mapping[str, Any] | None = None,
    predicted_effect: Mapping[str, Any] | None = None,
    observed_effect: Mapping[str, Any] | None = None,
    goal_delta: float | None = None,
) -> CausalTraceRod:
    target_binding_payload = dict(target_binding or {})
    success = bool(
        progressed
        or effect.objective_progress
        or effect.goal_progress
        or effect.goal_aligned_progress
        or effect.score_delta > 0
    )
    contradiction = bool(
        not success
        and (
            effect.no_op
            or effect.changed_pixel_count <= 0
            or bool(target_binding_payload.get("direct_score_probe_without_goal"))
        )
    )
    verdict = "supported" if success else "contradicted" if contradiction else "inconclusive"
    support_count = 1 if success else 0
    contradiction_count = 1 if contradiction else 0
    primitive = str(primitive_id or "")
    family = str(solver_family or "")
    op = str(operator or "")
    kind = str(target_kind or "")
    keys = tuple(
        key
        for key in (
            f"primitive:{primitive}" if primitive else "",
            f"solver_family:{family}" if family else "",
            f"operator:{op}" if op else "",
            f"target_kind:{kind}" if kind else "",
            f"effect:{effect.effect_type}",
            f"action:{effect.action_name}" if effect.action_name else "",
            f"binding_source:{effect.binding_source}" if effect.binding_source else "",
        )
        if key
    )
    parsed_key = _parse_target_key(target_key)
    computed_goal_delta = (
        float(goal_delta)
        if goal_delta is not None
        else 1.0
        if effect.score_delta > 0 or effect.goal_progress
        else 0.25
        if effect.goal_aligned_progress or effect.objective_progress or progressed
        else 0.0
    )
    return CausalTraceRod(
        primitive_id=primitive,
        solver_family=family,
        operator=op,
        action_name=str(action_name or ""),
        target_kind=kind,
        effect=effect,
        verdict=verdict,
        generic_keys=keys,
        before_sig=_sanitize_payload(dict(before_sig or {})),
        after_sig=_sanitize_payload(dict(after_sig or {})),
        target_binding=_sanitize_payload(target_binding_payload),
        predicted_effect=_sanitize_payload(dict(predicted_effect or {})),
        observed_effect=_sanitize_payload(dict(observed_effect or effect.to_dict())),
        goal_delta=computed_goal_delta,
        support_count=support_count,
        contradiction_count=contradiction_count,
        anxiety_delta=_anxiety_delta(success=success, contradiction=contradiction, effect=effect),
        prediction_error=0.0 if success else 1.0 if contradiction else 0.5,
        target_key=parsed_key,
    )


def build_hypothesis_rods_from_causal_pass_result(
    result: Any,
    *,
    step: int,
    site_prior_by_xy: Mapping[tuple[int, int], Mapping[str, Any]] | None = None,
    low_commitment_probe_steps: int = 2,
    low_commitment_bailout_conditions: Iterable[str] | None = None,
) -> list[HypothesisRod]:
    """Convert a causal-pass delta into unverified hypothesis rods.

    The output is deliberately not compatible with StructuralRodMemory selector
    support. It captures "what should be probed" while leaving "what is known"
    to CausalTraceRod after action-effect verification.
    """

    rods: list[HypothesisRod] = []
    prior_by_xy = dict(site_prior_by_xy or {})
    max_probe_steps = max(1, int(low_commitment_probe_steps or 1))
    bailout_conditions = tuple(
        str(item)
        for item in list(
            low_commitment_bailout_conditions
            or ("blocked", "no_state_change_after_ttl", "score_regression")
        )
        if str(item)
    )
    for idx, update in enumerate(list(getattr(result, "hypothesis_updates", []) or [])):
        if not isinstance(update, Mapping):
            continue
        action = str(update.get("action", "") or "")
        hid = str(update.get("id", "") or f"causal_pass_hypothesis:{step}:{idx}")
        cause = str(update.get("cause", "") or "")
        effect = str(update.get("effect", "") or "")
        effect_type = _hypothesis_effect_type(
            default="hypothesis_update",
            cause=cause,
            predicted_effect=effect,
            action=action,
        )
        metadata = {
            "update_action": action,
            "evidence_reason": str(update.get("evidence_reason", "") or ""),
        }
        rods.append(
            HypothesisRod(
                hypothesis_id=hid,
                effect_type=effect_type,
                proposed_cause=cause,
                predicted_effect=effect,
                confidence_delta=_as_float(update.get("confidence_delta", 0.0), 0.0),
                created_step=int(step),
                generic_keys=_hypothesis_generic_keys(
                    effect_type=effect_type,
                    source="causal_pass",
                    action=action,
                    kind="hypothesis_update",
                ),
                metadata=metadata,
            )
        )

    for idx, site in enumerate(list(getattr(result, "candidate_changer_sites", []) or [])):
        if not isinstance(site, Mapping):
            continue
        try:
            row = int(site.get("row", -1))
            col = int(site.get("col", -1))
        except (TypeError, ValueError):
            continue
        if row < 0 or col < 0:
            continue
        spatial_prior = _candidate_spatial_prior(site, prior_by_xy.get((row, col)))
        predicted = str(site.get("predicted_effect", "") or "")
        effect_type = _hypothesis_effect_type(
            default="candidate_changer_site",
            cause=str(site.get("rationale", "") or ""),
            predicted_effect=predicted,
            action="",
        )
        hid = str(site.get("hypothesis_id", "") or site.get("id", "") or f"candidate_site:{step}:{idx}")
        rods.append(
            HypothesisRod(
                hypothesis_id=hid,
                effect_type=effect_type,
                proposed_cause=str(site.get("rationale", "") or ""),
                predicted_effect=predicted,
                candidate_site={
                    "row": row,
                    "col": col,
                    "target_kind": "candidate_changer_site",
                },
                spatial_prior=spatial_prior,
                created_step=int(step),
                generic_keys=_hypothesis_generic_keys(
                    effect_type=effect_type,
                    source="causal_pass",
                    action="",
                    kind="candidate_changer_site",
                ),
                metadata={
                    "falsification": str(site.get("falsification", "") or ""),
                    "proposal_source": "causal_pass",
                },
            )
        )

    next_probe = getattr(result, "next_probe", {}) or {}
    if isinstance(next_probe, Mapping):
        actions = tuple(str(action) for action in list(next_probe.get("actions", []) or []) if str(action))
        if actions:
            hid = str(next_probe.get("hypothesis_id", "") or f"probe_intent:{step}")
            target_site = next_probe.get("target_site", {}) or {}
            candidate_site: dict[str, Any] = {}
            spatial_prior: dict[str, Any] = {}
            if isinstance(target_site, Mapping):
                candidate_site = _sanitize_payload(dict(target_site))
                try:
                    row = int(target_site.get("row", target_site.get("r", -1)))
                    col = int(target_site.get("col", target_site.get("c", -1)))
                    if row >= 0 and col >= 0:
                        spatial_prior = _candidate_spatial_prior(
                            target_site,
                            prior_by_xy.get((row, col)),
                        )
                except (TypeError, ValueError):
                    spatial_prior = {}
            effect_type = "probe_intent"
            rods.append(
                HypothesisRod(
                    hypothesis_id=hid,
                    effect_type=effect_type,
                    proposed_cause=str(next_probe.get("intent", "") or ""),
                    predicted_effect=str(next_probe.get("expected_support", "") or ""),
                    candidate_site=candidate_site,
                    probe_actions=actions,
                    commitment_level="probe_low",
                    execution_eligible=True,
                    max_probe_steps=min(max_probe_steps, len(actions)),
                    bailout_conditions=bailout_conditions,
                    spatial_prior=spatial_prior,
                    created_step=int(step),
                    generic_keys=_hypothesis_generic_keys(
                        effect_type=effect_type,
                        source="causal_pass",
                        action="probe",
                        kind="probe_intent",
                    ),
                    metadata={
                        "expected_refute": str(next_probe.get("expected_refute", "") or ""),
                    },
                )
            )
    return rods


def infer_effect_type(
    *,
    action_name: str,
    score_delta: int,
    changed_pixel_count: int,
    no_op: bool,
    target_kind: str,
    objective_progress: bool,
    objective_progress_kind: str,
    goal_progress: bool,
    goal_progress_kind: str,
) -> str:
    if int(score_delta) > 0:
        return "score_delta"
    objective_kind = str(objective_progress_kind or "")
    goal_kind = str(goal_progress_kind or "")
    kind = str(target_kind or "")
    action = str(action_name or "")
    if goal_progress:
        return _normalize_effect_kind(goal_kind or "goal_progress", target_kind=kind, action_name=action)
    if objective_progress and objective_kind:
        return _normalize_effect_kind(objective_kind, target_kind=kind, action_name=action)
    if kind in {"segment_displacement", "score_segment_displacement"}:
        return "segment_transport"
    if action == "ACTION6" and changed_pixel_count > 0:
        return "click_transition"
    if no_op or changed_pixel_count <= 0:
        return "noop"
    return "unexplained_effect"


def _normalize_effect_kind(kind: str, *, target_kind: str, action_name: str) -> str:
    text = str(kind or "")
    target = str(target_kind or "")
    action = str(action_name or "")
    if "segment_transport" in text or target in {"segment_displacement", "score_segment_displacement"}:
        return "segment_transport"
    if "click" in text or action == "ACTION6":
        return "click_transition"
    if "route" in text and ("open" in text or "connect" in text):
        return "route_connectivity_delta"
    if "toggle" in text:
        return "toggle"
    if text and text != "none":
        return text
    return "goal_progress"


def _hypothesis_effect_type(
    *,
    default: str,
    cause: str,
    predicted_effect: str,
    action: str,
) -> str:
    text = " ".join(
        part.lower()
        for part in (cause, predicted_effect, action)
        if str(part or "").strip()
    )
    if "score" in text or "goal" in text:
        return "score_precondition_hypothesis"
    if "route" in text or "path" in text or "block" in text or "connect" in text:
        return "route_hypothesis"
    if "toggle" in text or "switch" in text or "state" in text or "hidden" in text:
        return "state_transition_hypothesis"
    if "move" in text or "transport" in text or "push" in text:
        return "transport_hypothesis"
    if str(action or ""):
        return "probe_intent"
    return str(default or "hypothesis_update")


def _hypothesis_generic_keys(
    *,
    effect_type: str,
    source: str,
    action: str,
    kind: str,
) -> tuple[str, ...]:
    return tuple(
        key
        for key in (
            f"source:{source}" if source else "",
            "status:unverified",
            f"kind:{kind}" if kind else "",
            f"effect:{effect_type}" if effect_type else "",
            f"probe_action:{action}" if action else "",
        )
        if key
    )


def _candidate_spatial_prior(
    candidate_site: Mapping[str, Any],
    external_prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prior: dict[str, Any] = {}
    for source in (external_prior or {}, candidate_site):
        if not isinstance(source, Mapping):
            continue
        for key in (
            "route_quality",
            "path_len",
            "affordance_prior",
            "d_a_prior",
            "d_a_posterior",
            "d_a_posterior_delta",
            "d_a_posterior_confidence",
            "spatial_confidence",
            "posterior_support_count",
            "posterior_contradiction_count",
            "posterior_neutral_count",
        ):
            if key in source:
                prior[key] = source[key]
    if prior:
        prior.setdefault("prior_source", "spatial_prior")
    return _sanitize_payload(prior)


def default_rod_promotion_policy() -> RodPromotionPolicy:
    return RodPromotionPolicy()


_MECHANIC_RUNTIME_GENERIC_KEY_PREFIXES = (
    "effect:",
    "solver_family:",
    "target_kind:",
    "operator:",
    "binding_source:",
)
_MECHANIC_RUNTIME_EXCLUDED_EFFECTS = {"", "noop", "failure", "unexplained_effect"}


def _episode_mechanic_rods(
    *,
    by_generic_key_verdict: Mapping[str, Counter[str]],
    by_effect_type_verdict: Mapping[str, Counter[str]],
    by_solver_family: Mapping[str, Counter[str]],
    promotion_policy: RodPromotionPolicy | None = None,
) -> list[MechanicRod]:
    policy = promotion_policy or default_rod_promotion_policy()
    candidates: list[MechanicRod] = []
    candidates.extend(
        _episode_mechanic_rods_from_counters(
            by_generic_key_verdict,
            key_kind="generic_key",
            policy=policy,
        )
    )
    candidates.extend(
        _episode_mechanic_rods_from_counters(
            by_effect_type_verdict,
            key_kind="effect_type",
            policy=policy,
        )
    )
    candidates.extend(
        _episode_mechanic_rods_from_counters(
            by_solver_family,
            key_kind="solver_family",
            policy=policy,
        )
    )
    candidates.sort(
        key=lambda item: (
            not item.runtime_selection_eligible,
            not item.eligible_for_mechanic_promotion,
            -int(item.net_support),
            -int(item.support_count),
            str(item.key_kind),
            str(item.key),
        )
    )
    return candidates


def _episode_mechanic_rods_from_counters(
    counters: Mapping[str, Counter[str]],
    *,
    key_kind: str,
    policy: RodPromotionPolicy,
) -> list[MechanicRod]:
    out: list[MechanicRod] = []
    for raw_key, counter in counters.items():
        key = str(raw_key or "")
        if not _mechanic_runtime_key_allowed(key_kind=key_kind, key=key):
            continue
        supported = _as_int(counter.get("supported"), 0)
        if supported <= 0:
            continue
        contradicted = _as_int(counter.get("contradicted"), 0)
        inconclusive = _as_int(counter.get("inconclusive"), 0)
        total = max(1, supported + contradicted)
        net_support = supported - contradicted
        contradiction_rate = float(contradicted) / float(total)
        delayed_positive_rate = float(max(0, net_support)) / float(total)
        runtime_selection_eligible = (
            supported >= policy.mechanic_min_support_count
            and net_support >= policy.mechanic_min_net_support
            and contradiction_rate <= policy.mechanic_max_contradiction_rate
            and delayed_positive_rate >= policy.mechanic_min_delayed_positive_rate
        )
        promotion_eval = policy.evaluate_mechanic_summary(
            {
                "support_count": supported,
                "contradiction_count": contradicted,
                "distinct_episode_count": 1,
                "distinct_variant_count": 1,
                "delayed_positive_rate": delayed_positive_rate,
                "adapter_leakage_score": 0.0,
            }
        )
        out.append(
            MechanicRod(
                key_kind=str(key_kind),
                key=key,
                support_count=supported,
                contradiction_count=contradicted,
                inconclusive_count=inconclusive,
                net_support=net_support,
                contradiction_rate=contradiction_rate,
                delayed_positive_rate=delayed_positive_rate,
                runtime_selection_eligible=runtime_selection_eligible,
                eligible_for_mechanic_promotion=bool(
                    promotion_eval.get("eligible_for_mechanic_promotion")
                ),
                demote=bool(promotion_eval.get("demote")),
                retire=bool(promotion_eval.get("retire")),
                promotion_block_reasons=tuple(
                    _mechanic_promotion_block_reasons(promotion_eval, policy)
                ),
                generic_keys=(key,) if key_kind == "generic_key" else (f"{key_kind}:{key}",),
            )
        )
    return out


def _mechanic_runtime_key_allowed(*, key_kind: str, key: str) -> bool:
    if key_kind == "generic_key":
        if key.startswith("action:"):
            return False
        if key in {"effect:noop", "effect:failure", "effect:unexplained_effect"}:
            return False
        return key.startswith(_MECHANIC_RUNTIME_GENERIC_KEY_PREFIXES)
    if key_kind == "effect_type":
        return key not in _MECHANIC_RUNTIME_EXCLUDED_EFFECTS
    return bool(key)


def _mechanic_promotion_block_reasons(
    evaluation: Mapping[str, Any],
    policy: RodPromotionPolicy,
) -> list[str]:
    reasons: list[str] = []
    if _as_int(evaluation.get("support_count"), 0) < policy.mechanic_min_support_count:
        reasons.append("support_count")
    if _as_int(evaluation.get("net_support"), 0) < policy.mechanic_min_net_support:
        reasons.append("net_support")
    if _as_int(evaluation.get("distinct_episode_count"), 0) < policy.mechanic_min_distinct_episodes:
        reasons.append("distinct_episode_count")
    if _as_int(evaluation.get("distinct_variant_count"), 0) < policy.mechanic_min_distinct_variants:
        reasons.append("distinct_variant_count")
    if _as_float(evaluation.get("contradiction_rate"), 0.0) > policy.mechanic_max_contradiction_rate:
        reasons.append("contradiction_rate")
    if _as_float(evaluation.get("delayed_positive_rate"), 0.0) < policy.mechanic_min_delayed_positive_rate:
        reasons.append("delayed_positive_rate")
    if _as_float(evaluation.get("adapter_leakage_score"), 0.0) > policy.mechanic_max_adapter_leakage_score:
        reasons.append("adapter_leakage_score")
    return reasons


def _extract_spatial_prior_score(prior: Mapping[str, Any], *, default: float = 0.5) -> float:
    scores: list[float] = []
    for key in (
        "d_a_posterior",
        "d_a_prior",
        "affordance_prior",
        "spatial_confidence",
        "route_quality",
    ):
        if key not in prior:
            continue
        scores.append(_clamp01(_as_float(prior.get(key), default)))
    if not scores:
        return _clamp01(default)
    return _clamp01(sum(scores) / float(len(scores)))


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _as_float(value, 0.0)))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _counter_map_to_dict(data: Mapping[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {
        key: dict(sorted(counter.items()))
        for key, counter in sorted(data.items())
    }


def _target_binding_entry_from_rod(rod: CausalTraceRod) -> dict[str, Any] | None:
    binding = dict(rod.target_binding or {})
    site = binding.get("site")
    target_kind = str(binding.get("target_kind", "") or rod.target_kind or "")
    operator = str(binding.get("operator", "") or rod.operator or "")
    action_name = str(binding.get("action_name", "") or rod.action_name or rod.effect.action_name or "")
    if not isinstance(site, (list, tuple)) or len(site) < 2:
        if rod.target_key is not None and len(rod.target_key) >= 4:
            target_kind = target_kind or str(rod.target_key[0])
            operator = operator or str(rod.target_key[3])
            try:
                site = [int(rod.target_key[1]), int(rod.target_key[2])]
            except (TypeError, ValueError):
                site = None
    if not isinstance(site, (list, tuple)) or len(site) < 2:
        return None
    try:
        x = int(site[0])
        y = int(site[1])
    except (TypeError, ValueError):
        return None
    if not target_kind:
        target_kind = "target"
    if not operator:
        operator = "act"
    return {
        "target_key": f"{target_kind}:{x}:{y}:{operator}:{action_name}",
        "target_kind": target_kind,
        "operator": operator,
        "action_name": action_name,
        "site": [x, y],
        "effect_type": str(rod.effect.effect_type or ""),
        "binding_source": str(binding.get("binding_source", "") or rod.effect.binding_source or ""),
    }


def _target_binding_entries_to_list(entries: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in entries.values():
        item = dict(raw)
        supported = int(item.get("supported", 0) or 0)
        contradicted = int(item.get("contradicted", 0) or 0)
        inconclusive = int(item.get("inconclusive", 0) or 0)
        item["net_support"] = supported - contradicted
        item["score"] = round(float(item["net_support"]) + 0.10 * float(inconclusive), 3)
        out.append(item)
    out.sort(
        key=lambda item: (
            -float(item.get("score", 0.0) or 0.0),
            -int(item.get("supported", 0) or 0),
            int(item.get("contradicted", 0) or 0),
            str(item.get("target_key", "") or ""),
        )
    )
    return out


def _parse_target_key(raw: Iterable[Any] | None) -> tuple[str, int, int, str] | None:
    if raw is None:
        return None
    items = list(raw)
    if len(items) < 4:
        return None
    try:
        return str(items[0]), int(items[1]), int(items[2]), str(items[3])
    except (TypeError, ValueError):
        return None


def _anxiety_delta(*, success: bool, contradiction: bool, effect: EffectEvent) -> float:
    if effect.score_delta > 0 or effect.goal_progress:
        return 1.0
    if success:
        return 0.45
    if contradiction:
        return -0.35
    return 0.0


_FORBIDDEN_PAYLOAD_KEY_PARTS = (
    "game_id",
    "game_hash",
    "layout",
    "canonical",
    "adapter",
    "oracle",
    "frozen_plan",
    "solver_advice",
)


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            lowered = text_key.lower()
            if any(part in lowered for part in _FORBIDDEN_PAYLOAD_KEY_PARTS):
                continue
            out[text_key] = _sanitize_payload(item)
        return out
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_payload(item) for item in value]
    return value


__all__ = [
    "CausalTraceRod",
    "EffectEvent",
    "HypothesisRod",
    "HypothesisRodMemory",
    "MechanicRod",
    "RodPromotionPolicy",
    "SpatialRodPosterior",
    "SpatialRodPosteriorMemory",
    "StructuralRodMemory",
    "build_causal_trace_rod",
    "build_effect_event",
    "build_hypothesis_rods_from_causal_pass_result",
    "default_rod_promotion_policy",
    "infer_effect_type",
]
