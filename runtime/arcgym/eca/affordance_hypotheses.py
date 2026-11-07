"""Unknown executable affordance hypotheses.

This layer is intentionally narrower than PSA and MechanicHypothesis.  It
represents a legal primitive action that was selected from weak evidence but is
not already owned by a fixed reader path.  The claim is only that the action is
worth a protected probe; later PSA repair / goal linking decides whether it
becomes score bearing.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from .perceptual_state_abstraction import PSAAbstraction


@dataclass(frozen=True)
class AffordanceHypothesis:
    source_event_id: str
    action_name: str
    x: int
    y: int
    pre_state_hash: str
    expected_delta_signature: str
    evidence_ids: tuple[str, ...] = ()
    novelty_key: str = ""
    precondition_hint: str = ""
    risk_class: str = "low"
    confidence: float = 0.0
    expected_post_state_hash: str = ""
    source_kind: str = "psa_llm_proposed"
    llm_role: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(max(0.0, min(1.0, float(self.confidence))), 6)
        return _sanitize(payload)


@dataclass(frozen=True)
class AffordanceInductionResult:
    hypotheses: tuple[AffordanceHypothesis, ...]
    diagnostics: dict[str, Any]


class AffordanceHypothesisInducer:
    """Materialize protected-probe candidates from weak PSA evidence."""

    def build(
        self,
        *,
        psa_abstractions: Iterable[PSAAbstraction] = (),
        records: Iterable[Any] = (),
        current_state_hash: str = "",
        known_reader_keys: Iterable[tuple[str, int, int]] = (),
        require_current_state: bool = True,
        external_emergent_affordances: Iterable["AffordanceHypothesis"] = (),
    ) -> AffordanceInductionResult:
        record_list = list(records)
        known = {
            (str(action or ""), int(x), int(y))
            for action, x, y in known_reader_keys
            if str(action or "")
        }
        rejected: Counter[str] = Counter()
        materialized: list[AffordanceHypothesis] = []
        candidate_count = 0
        from_gemma_count = 0
        for abstraction in psa_abstractions:
            candidate_count += 1
            metadata = dict(abstraction.metadata or {})
            if _is_gemma_or_llm_seed(abstraction):
                from_gemma_count += 1
            else:
                rejected["not_weak_evidence_source"] += 1
                continue
            if bool(metadata.get("grounding_repair_applied", False)):
                rejected["grounding_repair_rejected"] += 1
                continue
            source_event_id = _first_source_event_id(abstraction)
            if not source_event_id:
                rejected["missing_source_event_id"] += 1
                continue
            action_name, x, y = _action_site_from_abstraction(abstraction)
            if not action_name or x is None or y is None:
                rejected["missing_action_site"] += 1
                continue
            if record_list:
                record = _matching_record_for_event_id(record_list, source_event_id)
                if record is None:
                    rejected["source_event_record_missing"] += 1
                    continue
                record_action = str(getattr(record, "action_name", "") or "")
                record_site = _record_click_site(record)
                if record_action != action_name:
                    rejected["wrong_action_family"] += 1
                    continue
                if record_site is not None and record_site != (int(x), int(y)):
                    rejected["wrong_click_site"] += 1
                    continue
            if (action_name, int(x), int(y)) in known:
                rejected["known_reader_masked"] += 1
                continue
            pre_hash, post_hash = _state_hashes_from_abstraction(abstraction)
            if not pre_hash:
                rejected["missing_pre_state_hash"] += 1
                continue
            if not post_hash or post_hash == pre_hash:
                rejected["missing_expected_delta"] += 1
                continue
            state_mode = "predictive_pre_state"
            expected_post_hash = post_hash
            effective_pre_hash = pre_hash
            if require_current_state and current_state_hash:
                current = str(current_state_hash)
                if pre_hash == current:
                    state_mode = "predictive_pre_state"
                elif post_hash == current:
                    state_mode = "repeat_from_observed_post_state"
                    effective_pre_hash = post_hash
                    expected_post_hash = ""
                else:
                    rejected["stale_current_state"] += 1
                    continue
            signature = (
                f"repeat_visible_effect:{source_event_id}"
                if state_mode == "repeat_from_observed_post_state"
                else f"{pre_hash[:8]}->{post_hash[:8]}"
            )
            materialized.append(
                AffordanceHypothesis(
                    source_event_id=source_event_id,
                    action_name=action_name,
                    x=int(x),
                    y=int(y),
                    pre_state_hash=effective_pre_hash,
                    expected_delta_signature=signature,
                    evidence_ids=tuple(abstraction.source_event_ids),
                    novelty_key=f"{action_name}:{int(x)},{int(y)}:{effective_pre_hash[:8]}",
                    precondition_hint="current_state_hash_matches",
                    risk_class="low",
                    confidence=max(0.0, min(1.0, float(abstraction.confidence))),
                    expected_post_state_hash=expected_post_hash,
                    source_kind=str(abstraction.source_kind or "psa_llm_proposed"),
                    llm_role=str(abstraction.llm_role or "none"),
                    metadata={
                        "abstraction_id": abstraction.abstraction_id,
                        "state_precondition_mode": state_mode,
                        "field_copy_materialized_by_provider": bool(
                            metadata.get("field_copy_materialized_by_provider", False)
                        ),
                        "field_copy_site_source": str(metadata.get("field_copy_site_source", "") or ""),
                        "direct_slot_fill_mode": str(metadata.get("direct_slot_fill_mode", "") or ""),
                    },
                )
            )
        external_count = 0
        external_masked = 0
        for ext in external_emergent_affordances:
            external_count += 1
            ext_key = (str(ext.action_name or ""), int(ext.x), int(ext.y))
            if ext_key in known:
                external_masked += 1
                rejected["known_reader_masked"] += 1
                continue
            materialized.append(ext)
        materialized.sort(key=lambda item: (-item.confidence, item.novelty_key))
        diagnostics = {
            "unknown_action_candidate_count": int(candidate_count),
            "unknown_action_candidate_from_gemma_count": int(from_gemma_count),
            "unknown_action_candidate_from_external_emergent_count": int(external_count),
            "unknown_action_candidate_external_emergent_masked_count": int(external_masked),
            "unknown_action_candidate_rejected_count": int(sum(rejected.values())),
            "unknown_action_candidate_materialized_count": int(len(materialized)),
            "unknown_action_candidate_executable_count": 0,
            "unknown_action_probe_generated_count": 0,
            "unknown_action_expected_delta_match_count": 0,
            "unknown_action_goal_linked_count": 0,
            "unknown_action_score_delta_count": 0,
            "unknown_action_false_positive_count": 0,
            "known_reader_masking_count": int(rejected.get("known_reader_masked", 0)),
            "rejected_reasons": dict(sorted(rejected.items())),
            "selected": materialized[0].to_dict() if materialized else None,
        }
        return AffordanceInductionResult(tuple(materialized[:4]), diagnostics)


class AffordanceProbeCompiler:
    """Compile an affordance hypothesis into a single protected probe step."""

    def compile(
        self,
        hypotheses: Iterable[AffordanceHypothesis],
        *,
        available_actions: Iterable[str],
        min_confidence: float = 0.45,
        emergent_min_confidence: float | None = None,
    ) -> dict[str, Any] | None:
        """Pick the highest-confidence available-action hypothesis.

        `min_confidence` is the global floor applied to every hypothesis.
        `emergent_min_confidence` is an OPTIONAL stricter floor applied
        only to hypotheses with `source_kind == "emergent_pattern_induced"`.
        Pass None to apply only the global floor. Per
        docs/method_a_postmortem_2026-05-21.md section 6, raising the
        emergent floor to >= 0.85 is the prescribed defensive knob for
        Method A's weak-evidence probes.
        """
        available = {str(action or "") for action in available_actions}
        emergent_floor = (
            float(emergent_min_confidence)
            if emergent_min_confidence is not None
            else None
        )
        for hypothesis in sorted(hypotheses, key=lambda item: (-item.confidence, item.novelty_key)):
            if hypothesis.confidence < float(min_confidence):
                continue
            if emergent_floor is not None and str(
                hypothesis.source_kind or ""
            ) == "emergent_pattern_induced":
                if hypothesis.confidence < emergent_floor:
                    continue
            if hypothesis.action_name not in available:
                continue
            return {
                "primitive_id": "unknown_affordance_probe",
                "hypothesis": hypothesis,
                "step": (
                    hypothesis.action_name,
                    {"x": int(hypothesis.x), "y": int(hypothesis.y)}
                    if hypothesis.action_name == "ACTION6"
                    else {},
                ),
                "reason": (
                    "unknown_affordance_probe compiled weak-evidence "
                    f"{hypothesis.novelty_key}"
                ),
            }
        return None


def _is_gemma_or_llm_seed(abstraction: PSAAbstraction) -> bool:
    source_kind = str(abstraction.source_kind or "")
    llm_role = str(abstraction.llm_role or "none")
    metadata = dict(abstraction.metadata or {})
    return bool(
        source_kind in {"psa_llm_proposed", "mixed"}
        or llm_role != "none"
        or metadata.get("field_copy_materialized_by_provider")
        or metadata.get("weak_seed_materialized_by_provider")
        or metadata.get("contract_repair_candidate_seed")
    )


def _first_source_event_id(abstraction: PSAAbstraction) -> str:
    for event_id in abstraction.source_event_ids:
        if str(event_id or ""):
            return str(event_id)
    metadata = dict(abstraction.metadata or {})
    return str(metadata.get("field_copy_source_event_id", "") or "")


def _action_site_from_abstraction(abstraction: PSAAbstraction) -> tuple[str, int | None, int | None]:
    for site in abstraction.control_sites:
        if not isinstance(site, Mapping):
            continue
        action = str(site.get("action_name", "") or "")
        x = _optional_int(site.get("x"))
        y = _optional_int(site.get("y"))
        if action and x is not None and y is not None:
            return action, x, y
    for step in abstraction.proposed_plan:
        if not isinstance(step, Mapping):
            continue
        action = str(step.get("action_name", "") or "")
        payload = step.get("payload") if isinstance(step.get("payload"), Mapping) else {}
        x = _optional_int(payload.get("x"))
        y = _optional_int(payload.get("y"))
        if action and x is not None and y is not None:
            return action, x, y
    for operator in abstraction.operator_candidates:
        if not isinstance(operator, Mapping):
            continue
        action = str(operator.get("action_name", "") or "")
        payload = operator.get("payload") if isinstance(operator.get("payload"), Mapping) else {}
        x = _optional_int(payload.get("x"))
        y = _optional_int(payload.get("y"))
        if action and x is not None and y is not None:
            return action, x, y
    return "", None, None


def _state_hashes_from_abstraction(abstraction: PSAAbstraction) -> tuple[str, str]:
    for variable in abstraction.state_variables:
        if not isinstance(variable, Mapping):
            continue
        pre_hash = str(variable.get("pre_state_hash", "") or "")
        post_hash = str(variable.get("post_state_hash", "") or "")
        if pre_hash or post_hash:
            return pre_hash, post_hash
    return "", ""


def _matching_record_for_event_id(records: Iterable[Any], event_id: str) -> Any | None:
    wanted = str(event_id or "")
    if not wanted:
        return None
    for record in records:
        action_name = str(getattr(record, "action_name", "") or "")
        try:
            step_index = int(getattr(record, "step_index"))
        except (TypeError, ValueError):
            step_index = -1
        candidates = {
            f"action_effect:{step_index}:{action_name}",
            f"action_effect:{step_index}",
        }
        if wanted in candidates or wanted.endswith(f":{step_index}:{action_name}"):
            return record
    return None


def _record_click_site(record: Any) -> tuple[int, int] | None:
    payload = getattr(record, "action_payload", {}) or {}
    if not isinstance(payload, Mapping):
        return None
    x = _optional_int(payload.get("x"))
    y = _optional_int(payload.get("y"))
    if x is None or y is None:
        return None
    return int(x), int(y)


def _optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


__all__ = [
    "AffordanceHypothesis",
    "AffordanceHypothesisInducer",
    "AffordanceInductionResult",
    "AffordanceProbeCompiler",
]
