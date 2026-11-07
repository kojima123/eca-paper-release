"""Inject emergent rod-mining candidates into the MechanicHypothesis pipeline.

This is the Gap 6 -> emergence bridge. It takes a candidate flagged as
novel by `rod_pattern_mining.propose_emergent_candidates` and converts
it into a `MechanicHypothesis` with `source_kind="emergent_pattern_induced"`
so the existing HypothesisPlanCompiler / executor / OutcomeLedger flow
can pick it up.

Design rules (matching docs/eca_psa_internal_llm_role_design_2026-05-16.md):

- LLM NOT consulted here; this is purely symbolic.
- The injector does NOT decide success or score; it only proposes a
  typed hypothesis. Promotion/falsification stays with Ledger/Rod.
- source_kind is a *new* string value not in the existing enum
  (`reader_fixed | ledger_induced | psa_llm_proposed | ...`).
  This is deliberate: a downstream `claim_boundary` audit can
  distinguish emergent_pattern_induced contributions from legacy
  source kinds without changing existing enum constants.

The injection is non-invasive: this module does NOT modify
`MechanicHypothesisInducer`; instead it builds a `MechanicHypothesis`
dataclass directly. Production callers can opt-in to consume it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .affordance_hypotheses import AffordanceHypothesis
from .mechanic_hypotheses import MechanicHypothesis
from .rod_pattern_mining import EmergentMechanicCandidate


SOURCE_KIND_EMERGENT_PATTERN = "emergent_pattern_induced"


@dataclass(frozen=True)
class InjectionResult:
    """Outcome of injecting one candidate."""

    hypothesis: MechanicHypothesis
    candidate_signature: str
    novel: bool
    injection_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis.to_dict(),
            "candidate_signature": self.candidate_signature,
            "novel": bool(self.novel),
            "injection_reason": str(self.injection_reason),
        }


def candidate_to_hypothesis(
    candidate: EmergentMechanicCandidate,
    *,
    family: str = "emergent",
    novel: bool = False,
    novel_primitive_ids: tuple[str, ...] = (),
) -> MechanicHypothesis:
    """Map an EmergentMechanicCandidate to a MechanicHypothesis.

    Conservative mapping:
    - hypothesis_id uniquely identifies the source signature.
    - source_kind = "emergent_pattern_induced".
    - induced_variables = primitive identifiers seen in the cluster.
    - goal_link_confidence / effect_confidence derived from
      cluster support density.
    - falsification_conditions enumerate what would invalidate the
      hypothesis (rod_count drops, contradiction_rate rises).
    """
    primitive_seed = "|".join(novel_primitive_ids[:3])
    hypothesis_id = (
        f"emergent_pattern:{family}:{primitive_seed}:"
        f"{abs(hash(candidate.signature)) % 10**10}"
    )
    confidence = float(candidate.confidence_score)
    source_event_ids: tuple[str, ...] = tuple(
        _sample_event_id(sample, idx)
        for idx, sample in enumerate(candidate.sample_rod_dicts[:3])
    )
    proposed_plan = _proposed_plan_from_samples(candidate.sample_rod_dicts)
    return MechanicHypothesis(
        hypothesis_id=hypothesis_id,
        family=family,
        source_kind=SOURCE_KIND_EMERGENT_PATTERN,
        source_event_ids=source_event_ids,
        induced_variables=tuple(novel_primitive_ids),
        controlled_objects=[],
        control_sites=[],
        current_state={"emergent_signature": candidate.signature},
        target_state=None,
        predicted_operator=None,
        goal_link_confidence=confidence,
        effect_confidence=confidence,
        contradiction_count=0,
        proposed_plan=proposed_plan,
        expected_goal_delta=0.0,
        source="emergent_candidate_injector",
        status="tentative",
        abort_conditions=[
            "rod_cluster_dissolved",
            "contradiction_rate_exceeds_threshold",
        ],
        falsification_conditions=(
            "rod_count_drops_below_min",
            "no_score_delta_after_protected_probe",
        ),
        compiler_family="emergent_pattern",
        executor_family="protected_probe_executor",
        llm_role="none",
        llm_direct_action=False,
        evidence={
            "rod_count": int(candidate.rod_count),
            "net_support": int(candidate.net_support),
            "contradiction_rate": float(candidate.contradiction_rate),
            "distinct_episodes": int(candidate.distinct_episodes),
            "distinct_variants": int(candidate.distinct_variants),
            "distinct_games": int(candidate.distinct_games),
            "candidate_signature": candidate.signature,
            "novel": bool(novel),
        },
    )


def _sample_event_id(sample: Mapping[str, Any], idx: int) -> str:
    pid = str(sample.get("primitive_id") or "?")
    action = str(sample.get("action_name") or "?")
    return f"emergent:{pid}:{action}:s{idx}"


def candidate_to_affordance_hypothesis(
    candidate: EmergentMechanicCandidate,
    *,
    pre_state_hash: str = "",
    expected_post_state_hash: str = "",
    risk_class: str = "low",
) -> AffordanceHypothesis | None:
    """Map a rod-mined candidate to a protected affordance hypothesis."""

    if not candidate.sample_rod_dicts:
        return None
    first = candidate.sample_rod_dicts[0]
    action_name = str(first.get("action_name") or "").strip()
    if not action_name:
        return None

    x_val: int | None = None
    y_val: int | None = None
    target_key = first.get("target_key")
    if isinstance(target_key, (list, tuple)) and len(target_key) >= 3:
        if isinstance(target_key[1], int):
            x_val = int(target_key[1])
        if isinstance(target_key[2], int):
            y_val = int(target_key[2])
    if x_val is None or y_val is None:
        before_sig = first.get("before_sig") or {}
        if isinstance(before_sig, Mapping):
            x_candidate = before_sig.get("x")
            y_candidate = before_sig.get("y")
            if isinstance(x_candidate, int):
                x_val = int(x_candidate)
            if isinstance(y_candidate, int):
                y_val = int(y_candidate)
    if x_val is None or y_val is None:
        return None

    source_event_id = _sample_event_id(first, 0)
    evidence_ids = tuple(
        _sample_event_id(sample, idx)
        for idx, sample in enumerate(candidate.sample_rod_dicts[:3])
    )
    metadata: dict[str, Any] = {
        "rod_count": int(candidate.rod_count),
        "net_support": int(candidate.net_support),
        "contradiction_rate": float(candidate.contradiction_rate),
        "distinct_episodes": int(candidate.distinct_episodes),
        "distinct_variants": int(candidate.distinct_variants),
        "distinct_games": int(candidate.distinct_games),
        "candidate_signature": candidate.signature,
        "origin": "emergent_candidate_injector.candidate_to_affordance_hypothesis",
    }
    return AffordanceHypothesis(
        source_event_id=source_event_id,
        action_name=action_name,
        x=int(x_val),
        y=int(y_val),
        pre_state_hash=str(pre_state_hash),
        expected_delta_signature=str(candidate.signature),
        evidence_ids=evidence_ids,
        novelty_key=f"emergent_rod_cluster:{candidate.signature}",
        precondition_hint="rod_cluster_evidence",
        risk_class=str(risk_class),
        confidence=float(candidate.confidence_score),
        expected_post_state_hash=str(expected_post_state_hash),
        source_kind=SOURCE_KIND_EMERGENT_PATTERN,
        llm_role="none",
        metadata=metadata,
    )


def inject_novel_affordance_candidates(
    candidates: list[EmergentMechanicCandidate],
    novelty_payloads: list[Mapping[str, Any]],
    *,
    pre_state_hash: str = "",
) -> list[AffordanceHypothesis]:
    """Filter novel rod-mined candidates into affordance hypotheses."""

    if len(candidates) != len(novelty_payloads):
        raise ValueError(
            f"candidates ({len(candidates)}) and novelty_payloads "
            f"({len(novelty_payloads)}) must align 1:1"
        )
    out: list[AffordanceHypothesis] = []
    for cand, novelty in zip(candidates, novelty_payloads):
        if not bool(novelty.get("is_novel")):
            continue
        affordance = candidate_to_affordance_hypothesis(
            cand,
            pre_state_hash=pre_state_hash,
        )
        if affordance is not None:
            out.append(affordance)
    return out


def inject_all_affordance_candidates(
    candidates: list[EmergentMechanicCandidate],
    *,
    pre_state_hash: str = "",
) -> list[AffordanceHypothesis]:
    """Emit affordance hypotheses for all candidates with action/site evidence."""

    out: list[AffordanceHypothesis] = []
    for cand in candidates:
        affordance = candidate_to_affordance_hypothesis(
            cand,
            pre_state_hash=pre_state_hash,
        )
        if affordance is not None:
            out.append(affordance)
    return out


def _proposed_plan_from_samples(
    samples: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a one-step protected probe plan from sample rod action/site data."""

    if not samples:
        return []
    first = samples[0]
    action_name = str(first.get("action_name") or "").strip()
    if not action_name:
        return []
    payload: dict[str, int] = {}
    target_key = first.get("target_key")
    if isinstance(target_key, (list, tuple)) and len(target_key) >= 3:
        if isinstance(target_key[1], int):
            payload["x"] = int(target_key[1])
        if isinstance(target_key[2], int):
            payload["y"] = int(target_key[2])
    if not payload:
        before_sig = first.get("before_sig") or {}
        if isinstance(before_sig, Mapping):
            x_val = before_sig.get("x")
            y_val = before_sig.get("y")
            if isinstance(x_val, int):
                payload["x"] = int(x_val)
            if isinstance(y_val, int):
                payload["y"] = int(y_val)
    return [{"action_name": action_name, "payload": payload, "step_index": 0}]


def inject_novel_candidates(
    candidates: list[EmergentMechanicCandidate],
    novelty_payloads: list[Mapping[str, Any]],
    *,
    family: str = "emergent",
) -> list[InjectionResult]:
    """Filter to novel candidates (per novelty payload) and inject each.

    `novelty_payloads` must be aligned 1:1 with `candidates`. Each entry
    is the dict returned by
    `rod_pattern_mining_cli._classify_candidate_novelty(samples)`.
    """
    if len(candidates) != len(novelty_payloads):
        raise ValueError(
            f"candidates ({len(candidates)}) and novelty_payloads "
            f"({len(novelty_payloads)}) must align 1:1"
        )
    results: list[InjectionResult] = []
    for cand, novelty in zip(candidates, novelty_payloads):
        is_novel = bool(novelty.get("is_novel"))
        is_mixed = bool(novelty.get("is_mixed"))
        if is_mixed:
            reason = "mixed_known_and_unknown_primitives"
        elif is_novel:
            reason = "all_samples_unregistered_primitive"
        else:
            reason = "all_samples_registered_primitive"
        # Only emit injection result for genuinely novel candidates.
        if not is_novel:
            continue
        novel_ids = tuple(
            sorted(str(x) for x in novelty.get("unregistered_primitive_ids", []))
        )
        hypothesis = candidate_to_hypothesis(
            cand,
            family=family,
            novel=True,
            novel_primitive_ids=novel_ids,
        )
        results.append(
            InjectionResult(
                hypothesis=hypothesis,
                candidate_signature=cand.signature,
                novel=True,
                injection_reason=reason,
            )
        )
    return results
