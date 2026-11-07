"""Candidate Resolver: Score-based target selection driven by ControlIR.

The core problem this solves: "candidates exist but selected_target is empty."
Instead of asking the LLM to pick a target via prose, the Resolver fills
selected_target numerically using a hand-tuned scoring formula.

Scoring formula (Section 4 of the design):

    resolver_score
      = base_admission_score
      + ir_boost
      + recent_live_bonus
      + hedge_bonus
      + stability_bonus
      - dead_revisit_penalty
      - churn_penalty
      - family_conflict_penalty

The Resolver does NOT override the ECA semantic layer.  It only fills
selected_target when IR instructions say "promote_target" or "retain_route".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from arcgym.eca.control_ir import ControlIREvidence
from arcgym.eca.control_ir_compiler import CompiledIR

log = logging.getLogger(__name__)


# ─── Resolver Configuration ────────────────────────────────────────────────

@dataclass
class ResolverConfig:
    """Tunable parameters for the Candidate Resolver."""
    # Base IR boost (validated at 0.10 in ControlIR experiments)
    ir_boost: float = 0.10
    # Bonus for candidates from recently-live anchor families
    recent_live_bonus: float = 0.08
    # Bonus for protected hedge candidates
    hedge_bonus: float = 0.06
    # Bonus scaled by anchor_stability (0.0 - 1.0)
    stability_bonus_scale: float = 0.05
    # Penalty for preterminal_dead_revisit source
    dead_revisit_penalty: float = 0.15
    # Penalty for high churn (frequent target changes)
    churn_penalty_per_change: float = 0.02
    churn_penalty_max: float = 0.10
    # Penalty for family conflict (multiple families competing)
    family_conflict_penalty: float = 0.05
    # Minimum resolver score to accept a candidate
    min_resolver_score: float = 0.15
    # Phase 4: no_info_gain penalty (candidate from saturated/stale family)
    no_info_gain_penalty: float = 0.12
    # Phase 4: stale_target penalty (target health is STALE)
    stale_target_penalty: float = 0.10
    # Phase 4: saturation_family_penalty (candidate from dominant saturated family)
    saturation_family_penalty: float = 0.08
    # Phase 4: health_bonus for LIVE target candidates
    health_live_bonus: float = 0.07
    saturated_same_component_penalty: float = 0.12
    saturated_alternative_component_bonus: float = 0.08
    saturated_same_family_penalty: float = 0.08
    saturated_alternative_family_bonus: float = 0.06
    # Steering: when force_retarget is active, down-rank the current target
    retarget_same_component_penalty: float = 0.22
    retarget_alternative_bonus: float = 0.10
    # Steering: when diversify_family is active, prefer a different family
    diversify_same_family_penalty: float = 0.18
    diversify_alternative_bonus: float = 0.12
    # Steering: when relief_pool is active, prefer non-dominant families
    relief_alternative_bonus: float = 0.08
    diversify_alt_exists_extra_same_family_penalty: float = 0.10
    diversify_alt_exists_extra_alternative_bonus: float = 0.06
    relief_dominant_family_penalty: float = 0.10
    relief_alt_exists_extra_alternative_bonus: float = 0.06
    synthetic_family_variant_penalty: float = 0.08
    stale_recent_shift_hold_component_bonus: float = 0.14
    stale_recent_shift_hold_family_bonus: float = 0.08
    stale_recent_shift_revert_component_penalty: float = 0.16
    stale_recent_shift_revert_family_penalty: float = 0.10
    # Retained route bias: when anchor exists, boost matching candidates
    retained_route_component_bonus: float = 0.12
    retained_route_family_bonus: float = 0.06

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "ResolverConfig":
        if not raw:
            return cls()
        return cls(
            ir_boost=float(raw.get("ir_boost", 0.10)),
            recent_live_bonus=float(raw.get("recent_live_bonus", 0.08)),
            hedge_bonus=float(raw.get("hedge_bonus", 0.06)),
            stability_bonus_scale=float(raw.get("stability_bonus_scale", 0.05)),
            dead_revisit_penalty=float(raw.get("dead_revisit_penalty", 0.15)),
            churn_penalty_per_change=float(raw.get("churn_penalty_per_change", 0.02)),
            churn_penalty_max=float(raw.get("churn_penalty_max", 0.10)),
            family_conflict_penalty=float(raw.get("family_conflict_penalty", 0.05)),
            min_resolver_score=float(raw.get("min_resolver_score", 0.15)),
            # Phase 4
            no_info_gain_penalty=float(raw.get("no_info_gain_penalty", 0.12)),
            stale_target_penalty=float(raw.get("stale_target_penalty", 0.10)),
            saturation_family_penalty=float(raw.get("saturation_family_penalty", 0.08)),
            health_live_bonus=float(raw.get("health_live_bonus", 0.07)),
            saturated_same_component_penalty=float(raw.get("saturated_same_component_penalty", 0.12)),
            saturated_alternative_component_bonus=float(raw.get("saturated_alternative_component_bonus", 0.08)),
            saturated_same_family_penalty=float(raw.get("saturated_same_family_penalty", 0.08)),
            saturated_alternative_family_bonus=float(raw.get("saturated_alternative_family_bonus", 0.06)),
            retarget_same_component_penalty=float(raw.get("retarget_same_component_penalty", 0.22)),
            retarget_alternative_bonus=float(raw.get("retarget_alternative_bonus", 0.10)),
            diversify_same_family_penalty=float(raw.get("diversify_same_family_penalty", 0.18)),
            diversify_alternative_bonus=float(raw.get("diversify_alternative_bonus", 0.12)),
            relief_alternative_bonus=float(raw.get("relief_alternative_bonus", 0.08)),
            diversify_alt_exists_extra_same_family_penalty=float(
                raw.get("diversify_alt_exists_extra_same_family_penalty", 0.10)
            ),
            diversify_alt_exists_extra_alternative_bonus=float(
                raw.get("diversify_alt_exists_extra_alternative_bonus", 0.06)
            ),
            relief_dominant_family_penalty=float(raw.get("relief_dominant_family_penalty", 0.10)),
            relief_alt_exists_extra_alternative_bonus=float(
                raw.get("relief_alt_exists_extra_alternative_bonus", 0.06)
            ),
            synthetic_family_variant_penalty=float(raw.get("synthetic_family_variant_penalty", 0.08)),
            stale_recent_shift_hold_component_bonus=float(
                raw.get("stale_recent_shift_hold_component_bonus", 0.14)
            ),
            stale_recent_shift_hold_family_bonus=float(
                raw.get("stale_recent_shift_hold_family_bonus", 0.08)
            ),
            stale_recent_shift_revert_component_penalty=float(
                raw.get("stale_recent_shift_revert_component_penalty", 0.16)
            ),
            stale_recent_shift_revert_family_penalty=float(
                raw.get("stale_recent_shift_revert_family_penalty", 0.10)
            ),
            retained_route_component_bonus=float(
                raw.get("retained_route_component_bonus", 0.12)
            ),
            retained_route_family_bonus=float(
                raw.get("retained_route_family_bonus", 0.06)
            ),
        )


# ─── Scored Candidate ──────────────────────────────────────────────────────

@dataclass
class ScoredCandidate:
    """A candidate with its resolver score breakdown."""
    component_id: str = ""
    action_family: str = ""
    source: str = ""
    action_name: str = ""
    anchor_key: str = ""
    center_row: int = 0
    center_col: int = 0
    base_admission_score: float = 0.0
    ir_boost: float = 0.0
    recent_live_bonus: float = 0.0
    hedge_bonus: float = 0.0
    stability_bonus: float = 0.0
    dead_revisit_penalty: float = 0.0
    churn_penalty: float = 0.0
    family_conflict_penalty: float = 0.0
    # Phase 4
    no_info_gain_penalty: float = 0.0
    stale_target_penalty: float = 0.0
    saturation_family_penalty: float = 0.0
    health_live_bonus: float = 0.0
    saturated_same_component_penalty: float = 0.0
    saturated_alternative_component_bonus: float = 0.0
    saturated_same_family_penalty: float = 0.0
    saturated_alternative_family_bonus: float = 0.0
    retarget_same_component_penalty: float = 0.0
    retarget_alternative_bonus: float = 0.0
    diversify_same_family_penalty: float = 0.0
    diversify_alternative_bonus: float = 0.0
    relief_alternative_bonus: float = 0.0
    synthetic_family_variant_penalty: float = 0.0
    stale_recent_shift_hold_component_bonus: float = 0.0
    stale_recent_shift_hold_family_bonus: float = 0.0
    stale_recent_shift_revert_component_penalty: float = 0.0
    stale_recent_shift_revert_family_penalty: float = 0.0
    retained_route_component_bonus: float = 0.0
    retained_route_family_bonus: float = 0.0
    resolver_score: float = 0.0
    selected: bool = False
    selection_reason: str = ""
    stage: str = ""
    closure_status: str = ""
    hypothesis_id: str = ""
    hypothesis_kind: str = ""
    linked_effect_id: str = ""
    window_id: str = ""
    continuation_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "action_family": self.action_family,
            "source": self.source,
            "action_name": self.action_name,
            "anchor_key": self.anchor_key,
            "center_row": self.center_row,
            "center_col": self.center_col,
            "base_admission_score": round(self.base_admission_score, 4),
            "ir_boost": round(self.ir_boost, 4),
            "recent_live_bonus": round(self.recent_live_bonus, 4),
            "hedge_bonus": round(self.hedge_bonus, 4),
            "stability_bonus": round(self.stability_bonus, 4),
            "dead_revisit_penalty": round(self.dead_revisit_penalty, 4),
            "churn_penalty": round(self.churn_penalty, 4),
            "family_conflict_penalty": round(self.family_conflict_penalty, 4),
            "no_info_gain_penalty": round(self.no_info_gain_penalty, 4),
            "stale_target_penalty": round(self.stale_target_penalty, 4),
            "saturation_family_penalty": round(self.saturation_family_penalty, 4),
            "health_live_bonus": round(self.health_live_bonus, 4),
            "saturated_same_component_penalty": round(self.saturated_same_component_penalty, 4),
            "saturated_alternative_component_bonus": round(self.saturated_alternative_component_bonus, 4),
            "saturated_same_family_penalty": round(self.saturated_same_family_penalty, 4),
            "saturated_alternative_family_bonus": round(self.saturated_alternative_family_bonus, 4),
            "retarget_same_component_penalty": round(self.retarget_same_component_penalty, 4),
            "retarget_alternative_bonus": round(self.retarget_alternative_bonus, 4),
            "diversify_same_family_penalty": round(self.diversify_same_family_penalty, 4),
            "diversify_alternative_bonus": round(self.diversify_alternative_bonus, 4),
            "relief_alternative_bonus": round(self.relief_alternative_bonus, 4),
            "synthetic_family_variant_penalty": round(self.synthetic_family_variant_penalty, 4),
            "stale_recent_shift_hold_component_bonus": round(
                self.stale_recent_shift_hold_component_bonus, 4
            ),
            "stale_recent_shift_hold_family_bonus": round(
                self.stale_recent_shift_hold_family_bonus, 4
            ),
            "retained_route_component_bonus": round(self.retained_route_component_bonus, 4),
            "retained_route_family_bonus": round(self.retained_route_family_bonus, 4),
            "stale_recent_shift_revert_component_penalty": round(
                self.stale_recent_shift_revert_component_penalty, 4
            ),
            "stale_recent_shift_revert_family_penalty": round(
                self.stale_recent_shift_revert_family_penalty, 4
            ),
            "resolver_score": round(self.resolver_score, 4),
            "selected": self.selected,
            "selection_reason": self.selection_reason,
            "stage": self.stage,
            "closure_status": self.closure_status,
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_kind": self.hypothesis_kind,
            "linked_effect_id": self.linked_effect_id,
            "window_id": self.window_id,
            "continuation_kind": self.continuation_kind,
        }


# ─── Resolution Result ─────────────────────────────────────────────────────

@dataclass
class ResolutionResult:
    """Result of a single resolution pass."""
    selected: ScoredCandidate | None = None
    all_scored: list[ScoredCandidate] = field(default_factory=list)
    resolved: bool = False
    resolution_source: str = ""  # "ir_promotion" | "ir_retention" | "ir_second_chance" | ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved,
            "resolution_source": self.resolution_source,
            "selected": self.selected.to_dict() if self.selected else None,
            "candidates_scored": len(self.all_scored),
            "all_scored": [c.to_dict() for c in self.all_scored],
        }


# ─── Candidate Resolver ────────────────────────────────────────────────────

class CandidateResolver:
    """Score-based target selection driven by ControlIR instructions.

    Usage:
        resolver = CandidateResolver(config)
        result = resolver.resolve(
            instructions=compilation_result.instructions,
            evidence=evidence,
            candidate_pool=runtime.last_candidate_pool,
            target_change_count=runtime.target_change_count,
        )
        if result.resolved:
            # Use result.selected to fill selected_target
    """

    def __init__(self, config: ResolverConfig | None = None) -> None:
        self.config = config or ResolverConfig()
        self._consecutive_failures: int = 0

    def reset(self) -> None:
        """Reset for a new episode."""
        self._consecutive_failures = 0

    def resolve(
        self,
        *,
        instructions: list[CompiledIR],
        evidence: ControlIREvidence | None,
        candidate_pool: list[dict[str, Any]] | None = None,
        target_change_count: int = 0,
    ) -> ResolutionResult:
        """Run one resolution pass.

        Args:
            instructions: Compiled IR instructions from ControlIRCompiler.
            evidence: Current ECA evidence snapshot.
            candidate_pool: Raw candidate pool from runtime (list of dicts).
            target_change_count: Number of target changes so far in the episode.

        Returns:
            ResolutionResult with the selected candidate (if any).
        """
        result = ResolutionResult()

        if not instructions or evidence is None:
            return result

        # Filter for actionable instructions
        actionable = [
            ir for ir in instructions
            if ir.action in (
                "promote_target",
                "retain_route",
                "second_chance",
                "force_retarget",
                "diversify_family",
                "relief_pool",
            )
        ]
        if not actionable:
            return result

        # Build candidate list from pool + IR targets
        candidates = self._build_candidate_list(
            actionable, candidate_pool or [], evidence
        )
        if not candidates:
            self._consecutive_failures += 1
            return result

        # Score all candidates
        scored = self._score_candidates(
            candidates, actionable, evidence, target_change_count
        )
        result.all_scored = scored

        # Select the best
        best = self._select_best(scored)
        if best is not None:
            best.selected = True
            best.selection_reason = self._determine_selection_reason(actionable)
            result.selected = best
            result.resolved = True
            result.resolution_source = best.selection_reason
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

        return result

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    # ── Private: Build candidate list ───────────────────────────────────

    def _build_candidate_list(
        self,
        instructions: list[CompiledIR],
        candidate_pool: list[dict[str, Any]],
        evidence: ControlIREvidence,
    ) -> list[dict[str, Any]]:
        """Build unified candidate list from IR targets + pool."""
        candidates: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()

        # First: add IR-recommended targets
        for ir in instructions:
            component_id = str(ir.target_component_id or "")
            family = str(ir.target_action_family or "")
            key = (component_id, family)
            if component_id and key not in seen_keys:
                candidates.append({
                    "component_id": component_id,
                    "admission_score": ir.target_score,
                    "source": ir.target_source,
                    "family": family,
                    "ir_recommended": True,
                    "rejected": False,
                })
                seen_keys.add(key)

        # Second: add non-rejected pool candidates
        for c in candidate_pool:
            cid = str(c.get("component_id", "") or c.get("comp_id", "") or "")
            family = str(c.get("family", "") or c.get("action_family", "") or "")
            key = (cid, family)
            if cid and key not in seen_keys and not bool(c.get("rejected", False)):
                candidates.append({
                    "component_id": cid,
                    "admission_score": float(
                        c.get("admission_score", c.get("score", 0.0)) or 0.0
                    ),
                    "source": str(c.get("source", "") or ""),
                    "family": family,
                    "ir_recommended": False,
                    "rejected": False,
                })
                seen_keys.add(key)

        self._append_synthetic_family_variant(
            candidates=candidates,
            seen_keys=seen_keys,
            instructions=instructions,
            evidence=evidence,
        )

        return candidates

    def _append_synthetic_family_variant(
        self,
        *,
        candidates: list[dict[str, Any]],
        seen_keys: set[tuple[str, str]],
        instructions: list[CompiledIR],
        evidence: ControlIREvidence,
    ) -> None:
        actions = {ir.action for ir in instructions}
        if not actions.intersection({"diversify_family", "relief_pool"}):
            return
        current_family = str(evidence.current_action_family or "").strip().lower()
        if current_family not in {"click", "move"}:
            return
        if any(
            str(c.get("family", "") or "").strip().lower()
            and str(c.get("family", "") or "").strip().lower() != current_family
            for c in candidates
        ):
            return
        if not candidates:
            return
        alt_family = "move" if current_family == "click" else "click"
        current_component = str(evidence.current_target_component_id or "").strip()
        base = None
        if current_component:
            base = next(
                (c for c in candidates if str(c.get("component_id", "") or "") == current_component),
                None,
            )
        if base is None:
            base = max(candidates, key=lambda c: float(c.get("admission_score", 0.0) or 0.0))
        component_id = str(base.get("component_id", "") or "").strip()
        key = (component_id, alt_family)
        if not component_id or key in seen_keys:
            return
        synthetic = dict(base)
        synthetic["family"] = alt_family
        synthetic["source"] = f"{str(base.get('source', '') or '')}_synthetic_family_alt"
        synthetic["admission_score"] = max(
            0.0,
            float(base.get("admission_score", 0.0) or 0.0) - self.config.synthetic_family_variant_penalty,
        )
        synthetic["synthetic_family_variant"] = True
        candidates.append(synthetic)
        seen_keys.add(key)

    # ── Private: Score candidates ───────────────────────────────────────

    def _score_candidates(
        self,
        candidates: list[dict[str, Any]],
        instructions: list[CompiledIR],
        evidence: ControlIREvidence,
        target_change_count: int,
    ) -> list[ScoredCandidate]:
        """Score all candidates using the resolver formula."""
        # Collect IR-recommended IDs
        ir_recommended_ids = {
            ir.target_component_id
            for ir in instructions
            if ir.target_component_id
        }

        # Collect suppressed dispositions
        suppressed = set()
        actions = {ir.action for ir in instructions}
        for ir in instructions:
            suppressed.update(ir.suppress_dispositions)
        current_target_component_id = str(evidence.current_target_component_id or "").strip()
        current_action_family = str(evidence.current_action_family or "").strip()
        target_shift_from_component_id = str(
            evidence.target_shift_from_component_id or ""
        ).strip()
        target_shift_from_family = str(
            evidence.target_shift_from_family or ""
        ).strip()
        try:
            turns_since_target_shift = int(evidence.turns_since_target_shift)
        except Exception:
            turns_since_target_shift = -1
        stale_recent_shift_active = bool(
            evidence.target_health == "stale"
            and evidence.recent_target_shift
            and turns_since_target_shift >= 0
            and turns_since_target_shift <= 1
            and current_target_component_id
            and target_shift_from_component_id
            and current_target_component_id != target_shift_from_component_id
        )

        # Count distinct families for conflict detection
        families = {
            c.get("family", "") for c in candidates if c.get("family")
        }
        has_family_conflict = len(families) > 2
        diversify_alt_exists = bool(
            "diversify_family" in actions
            and current_action_family
            and any(
                str(c.get("family", "") or "").strip()
                and str(c.get("family", "") or "").strip() != current_action_family
                for c in candidates
            )
        )
        relief_alt_exists = bool(
            "relief_pool" in actions
            and evidence.candidate_pool_dominant_family
            and any(
                str(c.get("family", "") or "").strip()
                and str(c.get("family", "") or "").strip() != evidence.candidate_pool_dominant_family
                for c in candidates
            )
        )

        scored: list[ScoredCandidate] = []
        for c in candidates:
            sc = ScoredCandidate(
                component_id=str(c.get("component_id", "")),
                action_family=str(c.get("family", "")),
                source=str(c.get("source", "")),
                action_name=str(c.get("action_name", "") or ""),
                base_admission_score=float(c.get("admission_score", 0.0)),
                stage=str(c.get("stage", "") or ""),
                closure_status=str(c.get("closure_status", "") or ""),
                linked_effect_id=str(c.get("linked_effect_id", "") or ""),
                window_id=str(c.get("window_id", "") or ""),
                continuation_kind=str(c.get("continuation_kind", "") or ""),
            )

            # IR boost
            if c.get("ir_recommended") or sc.component_id in ir_recommended_ids:
                sc.ir_boost = self.config.ir_boost

            # Recent live bonus
            if (
                evidence.recently_live_anchor
                and evidence.recently_live_anchor_family
                and sc.action_family == evidence.recently_live_anchor_family
            ):
                sc.recent_live_bonus = self.config.recent_live_bonus

            # Retained route bias (source-aware: selected_target > protected_hedge)
            retained_route_cid = str(evidence.retained_route_component_id or "").strip()
            retained_route_fam = str(evidence.retained_route_family or "").strip()
            retained_route_src = str(getattr(evidence, "retained_route_source", "") or "").strip()
            retained_route_scale = (
                1.0 if retained_route_src in ("selected_target", "")
                else 0.5 if retained_route_src == "protected_hedge"
                else 0.75
            )
            if retained_route_cid and sc.component_id == retained_route_cid:
                sc.retained_route_component_bonus = self.config.retained_route_component_bonus * retained_route_scale
            if retained_route_fam and sc.action_family == retained_route_fam and not sc.retained_route_component_bonus:
                sc.retained_route_family_bonus = self.config.retained_route_family_bonus * retained_route_scale

            # Hedge bonus
            if sc.source in ("protected_hedge", "hedge"):
                sc.hedge_bonus = self.config.hedge_bonus

            # Stability bonus
            if evidence.anchor_stability > 0:
                sc.stability_bonus = (
                    evidence.anchor_stability * self.config.stability_bonus_scale
                )

            # Dead revisit penalty
            if sc.source in ("revisit", "dead_revisit", "preterminal_dead_revisit"):
                sc.dead_revisit_penalty = self.config.dead_revisit_penalty

            # Churn penalty
            churn = min(
                target_change_count * self.config.churn_penalty_per_change,
                self.config.churn_penalty_max,
            )
            sc.churn_penalty = churn

            # Family conflict penalty
            if has_family_conflict:
                sc.family_conflict_penalty = self.config.family_conflict_penalty

            # Phase 4: no_info_gain penalty (suppressed disposition)
            if sc.action_family and sc.action_family in suppressed:
                sc.no_info_gain_penalty = self.config.no_info_gain_penalty

            # Phase 4: stale_target penalty (target health is STALE)
            if evidence.target_health == "stale":
                sc.stale_target_penalty = self.config.stale_target_penalty
                if stale_recent_shift_active:
                    if sc.component_id == current_target_component_id:
                        sc.stale_recent_shift_hold_component_bonus = (
                            self.config.stale_recent_shift_hold_component_bonus
                        )
                    elif sc.component_id == target_shift_from_component_id:
                        sc.stale_recent_shift_revert_component_penalty = (
                            self.config.stale_recent_shift_revert_component_penalty
                        )
                    if current_action_family:
                        if sc.action_family == current_action_family:
                            sc.stale_recent_shift_hold_family_bonus = (
                                self.config.stale_recent_shift_hold_family_bonus
                            )
                        elif (
                            target_shift_from_family
                            and sc.action_family == target_shift_from_family
                        ):
                            sc.stale_recent_shift_revert_family_penalty = (
                                self.config.stale_recent_shift_revert_family_penalty
                            )

            # Phase 4: saturation_family_penalty
            if (
                evidence.candidate_pool_dominant_family
                and sc.action_family == evidence.candidate_pool_dominant_family
                and evidence.candidate_pool_dominant_family_ratio
                >= 0.8
            ):
                sc.saturation_family_penalty = self.config.saturation_family_penalty

            # Phase 4: health_live_bonus
            if evidence.target_health == "live":
                sc.health_live_bonus = self.config.health_live_bonus

            # Quality gate: under saturated health, reward escape from the
            # current component/family and modestly penalize staying in place.
            if evidence.target_health == "saturated":
                if current_target_component_id:
                    if sc.component_id == current_target_component_id:
                        sc.saturated_same_component_penalty = (
                            self.config.saturated_same_component_penalty
                        )
                    else:
                        sc.saturated_alternative_component_bonus = (
                            self.config.saturated_alternative_component_bonus
                        )
                if current_action_family:
                    if sc.action_family == current_action_family:
                        sc.saturated_same_family_penalty = (
                            self.config.saturated_same_family_penalty
                        )
                    elif sc.action_family:
                        sc.saturated_alternative_family_bonus = (
                            self.config.saturated_alternative_family_bonus
                        )

            # Steering: force_retarget should move away from the current component.
            if "force_retarget" in actions and current_target_component_id:
                if sc.component_id == current_target_component_id:
                    if not stale_recent_shift_active:
                        sc.retarget_same_component_penalty = (
                            self.config.retarget_same_component_penalty
                        )
                else:
                    if not (
                        stale_recent_shift_active
                        and target_shift_from_component_id
                        and sc.component_id == target_shift_from_component_id
                    ):
                        sc.retarget_alternative_bonus = (
                            self.config.retarget_alternative_bonus
                        )

            # Steering: diversify_family should prefer alternatives to the repeated family.
            if "diversify_family" in actions and current_action_family:
                if sc.action_family == current_action_family:
                    sc.diversify_same_family_penalty = (
                        self.config.diversify_same_family_penalty
                    )
                    if diversify_alt_exists:
                        sc.diversify_same_family_penalty += (
                            self.config.diversify_alt_exists_extra_same_family_penalty
                        )
                elif sc.action_family:
                    sc.diversify_alternative_bonus = (
                        self.config.diversify_alternative_bonus
                    )
                    if diversify_alt_exists:
                        sc.diversify_alternative_bonus += (
                            self.config.diversify_alt_exists_extra_alternative_bonus
                        )

            # Steering: relief_pool should slightly prefer non-dominant families.
            if (
                "relief_pool" in actions
                and evidence.candidate_pool_dominant_family
            ):
                if sc.action_family == evidence.candidate_pool_dominant_family:
                    sc.saturation_family_penalty += (
                        self.config.relief_dominant_family_penalty
                    )
                elif sc.action_family:
                    sc.relief_alternative_bonus = (
                        self.config.relief_alternative_bonus
                    )
                    if relief_alt_exists:
                        sc.relief_alternative_bonus += (
                            self.config.relief_alt_exists_extra_alternative_bonus
                        )

            if bool(c.get("synthetic_family_variant", False)):
                sc.synthetic_family_variant_penalty = (
                    self.config.synthetic_family_variant_penalty
                )

            # Final score
            sc.resolver_score = (
                sc.base_admission_score
                + sc.ir_boost
                + sc.recent_live_bonus
                + sc.hedge_bonus
                + sc.stability_bonus
                + sc.health_live_bonus
                + sc.saturated_alternative_component_bonus
                + sc.saturated_alternative_family_bonus
                + sc.retarget_alternative_bonus
                + sc.diversify_alternative_bonus
                + sc.relief_alternative_bonus
                + sc.stale_recent_shift_hold_component_bonus
                + sc.stale_recent_shift_hold_family_bonus
                + sc.retained_route_component_bonus
                + sc.retained_route_family_bonus
                - sc.dead_revisit_penalty
                - sc.churn_penalty
                - sc.family_conflict_penalty
                - sc.no_info_gain_penalty
                - sc.stale_target_penalty
                - sc.saturation_family_penalty
                - sc.saturated_same_component_penalty
                - sc.saturated_same_family_penalty
                - sc.retarget_same_component_penalty
                - sc.diversify_same_family_penalty
                - sc.synthetic_family_variant_penalty
                - sc.stale_recent_shift_revert_component_penalty
                - sc.stale_recent_shift_revert_family_penalty
            )

            scored.append(sc)

        # Sort by resolver_score descending
        scored.sort(key=lambda s: s.resolver_score, reverse=True)
        return scored

    # ── Private: Select best ────────────────────────────────────────────

    def _select_best(self, scored: list[ScoredCandidate]) -> ScoredCandidate | None:
        """Select the best candidate above minimum threshold."""
        for sc in scored:
            if sc.resolver_score >= self.config.min_resolver_score:
                return sc
        return None

    # ── Private: Determine selection reason ─────────────────────────────

    def _determine_selection_reason(self, instructions: list[CompiledIR]) -> str:
        """Determine the resolution source from the dominant IR action."""
        actions = [ir.action for ir in instructions]
        if "promote_target" in actions:
            return "ir_promotion"
        if "retain_route" in actions:
            return "ir_retention"
        if "second_chance" in actions:
            return "ir_second_chance"
        if "force_retarget" in actions:
            return "ir_stale_retarget"
        if "diversify_family" in actions:
            return "ir_low_gain_div"
        if "relief_pool" in actions:
            return "ir_saturation_relief"
        return "ir_unknown"
