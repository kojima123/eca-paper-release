"""Typed bridge for PSA/ECA-internal LLM cognition.

This module deliberately does not call any LLM backend.  It defines the
source-swappable boundary where a PSA/ECA-internal LLM provider may return
typed candidates.  The rest of Lane C still compiles those candidates through
MechanicHypothesis and protected execution.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import hashlib
import importlib
import json
from typing import Any, Callable, Iterable, Mapping

from .action_effect_model import ActionEffectRecord
from .mechanic_hypotheses import MechanicHypothesis
from .perceptual_state_abstraction import PSAAbstraction


InternalLLMProvider = Callable[..., Any]
PSA_PROVIDER_SCHEMA_VERSION = "internal_llm_psa_candidate.v1"
ECA_INTERPRETER_SCHEMA_VERSION = "internal_llm_mechanic_hypothesis.v1"
PSA_PROVIDER_SOURCE_KINDS = {"psa_llm_proposed"}
ECA_INTERPRETER_SOURCE_KINDS = {"eca_llm_interpreted"}
PSA_PROVIDER_LLM_ROLES = {"candidate_generation", "labeling"}
ECA_INTERPRETER_LLM_ROLES = {"hypothesis_interpretation", "probe_design"}


@dataclass(frozen=True)
class InternalLLMCognitionResult:
    psa_abstractions: list[PSAAbstraction] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InternalLLMInterpretationResult:
    mechanic_hypotheses: list[MechanicHypothesis] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InternalLLMPromptPacket:
    role: str
    expected_schema_version: str
    system_prompt: str
    user_payload: dict[str, Any]
    response_schema: dict[str, Any]
    forbidden_fields: tuple[str, ...] = ("natural_language_plan", "action_plan")

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_payload(asdict(self))

    def audit(self) -> dict[str, Any]:
        return internal_llm_prompt_packet_audit(self)


class InternalLLMCognitionBridge:
    """Normalize internal LLM provider output into typed PSA abstractions."""

    def __init__(
        self,
        provider: InternalLLMProvider | None = None,
        *,
        provider_source: str = "",
        provider_path: str = "",
    ) -> None:
        self.provider = provider
        self.provider_source = str(provider_source or ("constructor" if provider is not None else "none"))
        self.provider_path = str(provider_path or "")

    def build(
        self,
        *,
        enabled: bool,
        frame: Any,
        records: Iterable[ActionEffectRecord],
        frame_index: int | None = None,
        available_actions: Iterable[str] = (),
        current_state_hash: str = "",
    ) -> InternalLLMCognitionResult:
        record_list = list(records)
        diagnostics: dict[str, Any] = {
            "enabled": bool(enabled),
            "provider_available": self.provider is not None,
            "provider_called": False,
            "provider_source": self.provider_source,
            "provider_path": self.provider_path,
            "raw_candidate_count": 0,
            "psa_abstraction_count": 0,
            "rejected_candidate_count": 0,
            "llm_direct_action_candidate_count": 0,
            "source_kind_histogram": {},
            "llm_role_histogram": {},
            "error": "",
            "schema_enforced": True,
            "accepted_schema_versions": [PSA_PROVIDER_SCHEMA_VERSION],
            "rejection_reasons": {},
            "grounding_repair_count": 0,
            "grounding_repair_reasons": {},
            "prompt_packet_role": "psa_candidate_generation",
            "prompt_packet_schema_version": PSA_PROVIDER_SCHEMA_VERSION,
            "prompt_packet_hash": "",
            "prompt_packet_audit": {},
            "raw_response_kind": "",
            "raw_response_preview": None,
            "raw_candidate_previews": [],
            "rejected_candidate_previews": [],
            "json_response_parsed": False,
            "json_response_parse_error": "",
            "parsed_response_keys": [],
        }
        if not enabled or self.provider is None:
            return InternalLLMCognitionResult(diagnostics=diagnostics)

        diagnostics["provider_called"] = True
        prompt_packet = build_psa_llm_prompt_packet(
            frame=frame,
            records=record_list,
            frame_index=frame_index,
            available_actions=available_actions,
            current_state_hash=current_state_hash,
        )
        prompt_audit = internal_llm_prompt_packet_audit(prompt_packet)
        diagnostics["prompt_packet_hash"] = prompt_audit["prompt_packet_hash"]
        diagnostics["prompt_packet_audit"] = prompt_audit
        try:
            raw = self.provider(
                frame=frame,
                records=record_list,
                frame_index=frame_index,
                available_actions=tuple(str(action) for action in available_actions),
                current_state_hash=str(current_state_hash or ""),
                prompt_packet=prompt_packet.to_dict(),
            )
        except Exception as exc:  # pragma: no cover - defensive diagnostic path.
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            return InternalLLMCognitionResult(diagnostics=diagnostics)

        diagnostics["raw_response_preview"] = _payload_preview(raw)
        raw = _parse_json_provider_response_if_needed(raw, diagnostics)
        items = _extract_candidate_items(raw)
        diagnostics["raw_candidate_count"] = int(len(items))
        diagnostics["raw_candidate_previews"] = [
            _candidate_preview(item, index=index) for index, item in enumerate(items[:4])
        ]
        abstractions: list[PSAAbstraction] = []
        rejection_reasons: Counter[str] = Counter()
        grounding_repair_reasons: Counter[str] = Counter()
        rejected_previews: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            abstraction, reason = _coerce_psa_abstraction_with_reason(item)
            if abstraction is None:
                diagnostics["rejected_candidate_count"] = int(diagnostics["rejected_candidate_count"]) + 1
                rejection_reasons[reason or "invalid_candidate"] += 1
                if len(rejected_previews) < 4:
                    rejected_previews.append(
                        _candidate_preview(
                            item,
                            index=index,
                            reason=reason or "invalid_candidate",
                        )
                    )
                continue
            abstraction, grounding_reason, repaired = _ground_psa_abstraction_in_evidence(
                abstraction,
                records=record_list,
                available_actions=available_actions,
            )
            if abstraction is None:
                diagnostics["rejected_candidate_count"] = int(diagnostics["rejected_candidate_count"]) + 1
                rejection_reasons[grounding_reason or "ungrounded_candidate"] += 1
                if len(rejected_previews) < 4:
                    rejected_previews.append(
                        _candidate_preview(
                            item,
                            index=index,
                            reason=grounding_reason or "ungrounded_candidate",
                        )
                    )
                continue
            if repaired:
                diagnostics["grounding_repair_count"] = int(diagnostics["grounding_repair_count"]) + 1
                grounding_repair_reasons[grounding_reason or "grounding_repaired"] += 1
            abstractions.append(abstraction)

        diagnostics["psa_abstraction_count"] = int(len(abstractions))
        diagnostics["llm_direct_action_candidate_count"] = int(
            sum(1 for abstraction in abstractions if abstraction.llm_direct_action)
        )
        source_histogram = Counter(str(abstraction.source_kind or "unknown") for abstraction in abstractions)
        role_histogram = Counter(str(abstraction.llm_role or "none") for abstraction in abstractions)
        diagnostics["source_kind_histogram"] = dict(sorted(source_histogram.items()))
        diagnostics["llm_role_histogram"] = dict(sorted(role_histogram.items()))
        diagnostics["rejection_reasons"] = dict(sorted(rejection_reasons.items()))
        diagnostics["grounding_repair_reasons"] = dict(sorted(grounding_repair_reasons.items()))
        diagnostics["rejected_candidate_previews"] = rejected_previews
        return InternalLLMCognitionResult(
            psa_abstractions=abstractions[:4],
            diagnostics=diagnostics,
        )


class InternalLLMInterpreterBridge:
    """Normalize ECA-internal LLM output into typed mechanic hypotheses."""

    def __init__(
        self,
        provider: InternalLLMProvider | None = None,
        *,
        provider_source: str = "",
        provider_path: str = "",
    ) -> None:
        self.provider = provider
        self.provider_source = str(provider_source or ("constructor" if provider is not None else "none"))
        self.provider_path = str(provider_path or "")

    def build(
        self,
        *,
        enabled: bool,
        frame: Any,
        records: Iterable[ActionEffectRecord],
        frame_index: int | None = None,
        available_actions: Iterable[str] = (),
        current_state_hash: str = "",
        psa_abstractions: Iterable[PSAAbstraction] = (),
        mechanic_hypotheses: Iterable[MechanicHypothesis] = (),
    ) -> InternalLLMInterpretationResult:
        record_list = list(records)
        psa_list = list(psa_abstractions)
        base_hypotheses = list(mechanic_hypotheses)
        diagnostics: dict[str, Any] = {
            "enabled": bool(enabled),
            "provider_available": self.provider is not None,
            "provider_called": False,
            "provider_source": self.provider_source,
            "provider_path": self.provider_path,
            "input_psa_abstraction_count": int(len(psa_list)),
            "input_mechanic_hypothesis_count": int(len(base_hypotheses)),
            "raw_hypothesis_count": 0,
            "mechanic_hypothesis_count": 0,
            "rejected_hypothesis_count": 0,
            "llm_direct_action_hypothesis_count": 0,
            "source_kind_histogram": {},
            "llm_role_histogram": {},
            "error": "",
            "schema_enforced": True,
            "accepted_schema_versions": [ECA_INTERPRETER_SCHEMA_VERSION],
            "rejection_reasons": {},
            "prompt_packet_role": "eca_hypothesis_interpretation",
            "prompt_packet_schema_version": ECA_INTERPRETER_SCHEMA_VERSION,
            "prompt_packet_hash": "",
            "prompt_packet_audit": {},
            "raw_response_kind": "",
            "json_response_parsed": False,
            "json_response_parse_error": "",
            "parsed_response_keys": [],
        }
        if not enabled or self.provider is None:
            return InternalLLMInterpretationResult(diagnostics=diagnostics)

        diagnostics["provider_called"] = True
        prompt_packet = build_eca_llm_interpreter_prompt_packet(
            frame=frame,
            records=record_list,
            frame_index=frame_index,
            available_actions=available_actions,
            current_state_hash=current_state_hash,
            psa_abstractions=psa_list,
            mechanic_hypotheses=base_hypotheses,
        )
        prompt_audit = internal_llm_prompt_packet_audit(prompt_packet)
        diagnostics["prompt_packet_hash"] = prompt_audit["prompt_packet_hash"]
        diagnostics["prompt_packet_audit"] = prompt_audit
        try:
            raw = self.provider(
                frame=frame,
                records=record_list,
                frame_index=frame_index,
                available_actions=tuple(str(action) for action in available_actions),
                current_state_hash=str(current_state_hash or ""),
                psa_abstractions=tuple(psa_list),
                mechanic_hypotheses=tuple(base_hypotheses),
                prompt_packet=prompt_packet.to_dict(),
            )
        except Exception as exc:  # pragma: no cover - defensive diagnostic path.
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            return InternalLLMInterpretationResult(diagnostics=diagnostics)

        raw = _parse_json_provider_response_if_needed(raw, diagnostics)
        items = _extract_hypothesis_items(raw)
        diagnostics["raw_hypothesis_count"] = int(len(items))
        hypotheses: list[MechanicHypothesis] = []
        rejection_reasons: Counter[str] = Counter()
        for item in items:
            hypothesis, reason = _coerce_mechanic_hypothesis_with_reason(item)
            if hypothesis is None:
                diagnostics["rejected_hypothesis_count"] = int(diagnostics["rejected_hypothesis_count"]) + 1
                rejection_reasons[reason or "invalid_hypothesis"] += 1
                continue
            hypotheses.append(hypothesis)

        diagnostics["mechanic_hypothesis_count"] = int(len(hypotheses))
        diagnostics["llm_direct_action_hypothesis_count"] = int(
            sum(1 for hypothesis in hypotheses if hypothesis.llm_direct_action)
        )
        source_histogram = Counter(str(hypothesis.source_kind or "unknown") for hypothesis in hypotheses)
        role_histogram = Counter(str(hypothesis.llm_role or "none") for hypothesis in hypotheses)
        diagnostics["source_kind_histogram"] = dict(sorted(source_histogram.items()))
        diagnostics["llm_role_histogram"] = dict(sorted(role_histogram.items()))
        diagnostics["rejection_reasons"] = dict(sorted(rejection_reasons.items()))
        return InternalLLMInterpretationResult(
            mechanic_hypotheses=hypotheses[:4],
            diagnostics=diagnostics,
        )


def build_psa_llm_prompt_packet(
    *,
    frame: Any,
    records: Iterable[ActionEffectRecord],
    frame_index: int | None = None,
    available_actions: Iterable[str] = (),
    current_state_hash: str = "",
    max_records: int = 8,
) -> InternalLLMPromptPacket:
    return InternalLLMPromptPacket(
        role="psa_candidate_generation",
        expected_schema_version=PSA_PROVIDER_SCHEMA_VERSION,
        system_prompt=(
            "You are the PSA-LLM inside ECA/PSA. Do not solve the game and "
            "do not emit a natural-language action plan. Return only typed "
            "state-variable, predicate, operator, and goal-link candidates "
            "with family exactly 'cycle'. Do not invent game semantics; use "
            "only recent_action_effect_records. Return at most one candidate; "
            "each required candidate array should contain exactly one grounded "
            "object. source_event_ids must be copied from valid_source_event_ids; "
            "do not enumerate placeholder IDs. If evidence is insufficient, "
            "return an empty psa_abstractions array. "
            f"using schema_version={PSA_PROVIDER_SCHEMA_VERSION}. /no_think"
        ),
        user_payload={
            "frame_shape": _frame_shape(frame),
            "frame_value_summary": _frame_value_summary(frame),
            "frame_index": None if frame_index is None else int(frame_index),
            "current_state_hash": str(current_state_hash or ""),
            "available_actions": [str(action) for action in available_actions],
            "recent_action_effect_records": _compact_records(records, max_records=max_records),
            "valid_source_event_ids": _valid_source_event_ids(records, max_records=max_records),
            "visible_effect_grammar_summary": _visible_effect_grammar_summary(
                records,
                frame=frame,
                available_actions=available_actions,
                max_records=max_records,
            ),
            "required_source_kind": "psa_llm_proposed",
            "allowed_llm_roles": sorted(PSA_PROVIDER_LLM_ROLES),
            "allowed_families": ["cycle"],
            "required_family": "cycle",
            "forbidden_fields": ["natural_language_plan", "action_plan"],
            "empty_when_no_supporting_effect_record": True,
            "response_cardinality_contract": {
                "psa_abstractions_max_items": 1,
                "source_event_ids_max_items": 4,
                "candidate_section_items": 1,
            },
            "minimum_valid_response_template": _psa_response_template(),
        },
        response_schema=_psa_response_schema(),
    )


def build_eca_llm_interpreter_prompt_packet(
    *,
    frame: Any,
    records: Iterable[ActionEffectRecord],
    frame_index: int | None = None,
    available_actions: Iterable[str] = (),
    current_state_hash: str = "",
    psa_abstractions: Iterable[PSAAbstraction] = (),
    mechanic_hypotheses: Iterable[MechanicHypothesis] = (),
    max_records: int = 8,
) -> InternalLLMPromptPacket:
    return InternalLLMPromptPacket(
        role="eca_hypothesis_interpretation",
        expected_schema_version=ECA_INTERPRETER_SCHEMA_VERSION,
        system_prompt=(
            "You are the ECA-LLM inside ECA/PSA. Do not control actions "
            "directly. Interpret typed PSA candidates into structural "
            "deficits, falsification conditions, and testable mechanic "
            "hypotheses with family exactly 'cycle'. Do not invent game "
            "semantics; use only PSA candidates and recent action-effect "
            "records. If evidence is insufficient, return an empty "
            "mechanic_hypotheses array. "
            f"hypotheses using schema_version={ECA_INTERPRETER_SCHEMA_VERSION}. /no_think"
        ),
        user_payload={
            "frame_shape": _frame_shape(frame),
            "frame_index": None if frame_index is None else int(frame_index),
            "current_state_hash": str(current_state_hash or ""),
            "available_actions": [str(action) for action in available_actions],
            "recent_action_effect_records": _compact_records(records, max_records=max_records),
            "psa_abstractions": [_to_payload(item) for item in psa_abstractions][:4],
            "existing_mechanic_hypotheses": [_to_payload(item) for item in mechanic_hypotheses][:4],
            "required_source_kind": "eca_llm_interpreted",
            "allowed_llm_roles": sorted(ECA_INTERPRETER_LLM_ROLES),
            "allowed_families": ["cycle"],
            "required_family": "cycle",
            "forbidden_fields": ["natural_language_plan", "action_plan"],
            "empty_when_no_psa_candidate": True,
            "minimum_valid_response_template": _eca_response_template(),
        },
        response_schema=_eca_response_schema(),
    )


def parse_internal_llm_json_response(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError(f"unsupported internal LLM response type: {type(raw).__name__}")


def internal_llm_prompt_packet_audit(
    packet: InternalLLMPromptPacket | Mapping[str, Any],
) -> dict[str, Any]:
    payload = packet.to_dict() if isinstance(packet, InternalLLMPromptPacket) else _sanitize_payload(packet)
    if not isinstance(payload, Mapping):
        payload = {}
    user_payload = payload.get("user_payload", {})
    if not isinstance(user_payload, Mapping):
        user_payload = {}
    response_schema = payload.get("response_schema", {})
    if not isinstance(response_schema, Mapping):
        response_schema = {}
    packet_json = _canonical_json(payload)
    user_payload_json = _canonical_json(user_payload)
    response_schema_json = _canonical_json(response_schema)
    recent_records = user_payload.get("recent_action_effect_records", [])
    available_actions = user_payload.get("available_actions", [])
    forbidden_fields = payload.get("forbidden_fields", [])
    if isinstance(forbidden_fields, (str, bytes)) or not isinstance(forbidden_fields, Iterable):
        forbidden_fields = []
    return {
        "prompt_packet_hash": "sha256:" + hashlib.sha256(packet_json.encode("utf-8")).hexdigest(),
        "role": str(payload.get("role", "") or ""),
        "expected_schema_version": str(payload.get("expected_schema_version", "") or ""),
        "system_prompt_chars": len(str(payload.get("system_prompt", "") or "")),
        "packet_json_chars": len(packet_json),
        "user_payload_json_chars": len(user_payload_json),
        "response_schema_json_chars": len(response_schema_json),
        "user_payload_keys": sorted(str(key) for key in user_payload.keys()),
        "frame_shape": _sanitize_payload(user_payload.get("frame_shape", [])),
        "available_action_count": len(list(available_actions)) if isinstance(available_actions, Iterable) and not isinstance(available_actions, (str, bytes, Mapping)) else 0,
        "recent_action_effect_record_count": len(list(recent_records)) if isinstance(recent_records, Iterable) and not isinstance(recent_records, (str, bytes, Mapping)) else 0,
        "forbidden_fields": [str(item) for item in forbidden_fields],
    }


def resolve_internal_llm_provider_path(path: str | None) -> InternalLLMProvider | None:
    normalized = str(path or "").strip()
    if not normalized:
        return None
    if ":" in normalized:
        module_name, attr_path = normalized.split(":", 1)
    else:
        module_name, sep, attr_path = normalized.rpartition(".")
        if not sep:
            raise ValueError(
                "internal LLM provider path must be 'module:callable' or 'module.callable'"
            )
    module_name = module_name.strip()
    attr_path = attr_path.strip()
    if not module_name or not attr_path:
        raise ValueError(
            "internal LLM provider path must include both module and callable"
        )
    module = importlib.import_module(module_name)
    target: Any = module
    for attr in attr_path.split("."):
        attr = attr.strip()
        if not attr:
            raise ValueError(f"invalid internal LLM provider attribute path: {normalized}")
        target = getattr(target, attr)
    if not callable(target):
        raise TypeError(f"internal LLM provider is not callable: {normalized}")
    return target


def _payload_preview(value: Any, *, max_chars: int = 1200) -> Any:
    sanitized = _sanitize_payload(value)
    try:
        text = json.dumps(sanitized, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError):
        text = str(sanitized)
    if len(text) <= int(max_chars):
        return sanitized
    return {
        "truncated": True,
        "char_count": len(text),
        "preview": text[: int(max_chars)],
    }


def _candidate_preview(
    item: Any,
    *,
    index: int,
    reason: str = "",
) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {
            "index": int(index),
            "kind": type(item).__name__,
            "reason": str(reason or ""),
            "preview": _payload_preview(item, max_chars=600),
        }
    missing_sections = [
        section
        for section, keys in {
            "state_variables": (
                "state_variables",
                "state_variable_candidates",
                "current_state",
                "current_state_candidate",
            ),
            "predicate_candidates": (
                "predicate_candidates",
                "predicates",
                "predicate",
                "goal_predicate",
                "target_state",
                "target_state_candidate",
            ),
            "operator_candidates": (
                "operator_candidates",
                "operators",
                "operator",
                "predicted_operator",
            ),
            "goal_link_candidates": (
                "goal_link_candidates",
                "goal_links",
                "goal_link",
                "goal_link_candidate",
            ),
            "control_sites": ("control_sites", "control_site"),
            "proposed_plan": ("proposed_plan", "probe_plan", "verification_plan"),
        }.items()
        if not _tuple_of_dicts_for_keys(item, *keys)
    ]
    return {
        "index": int(index),
        "kind": "mapping",
        "reason": str(reason or ""),
        "keys": sorted(str(key) for key in item.keys()),
        "schema_version": str(item.get("schema_version", "") or ""),
        "family": str(item.get("family", "") or ""),
        "source_kind": str(item.get("source_kind", "") or ""),
        "llm_role": str(item.get("llm_role", "") or ""),
        "missing_sections": missing_sections,
        "preview": _payload_preview(item, max_chars=600),
    }


def _parse_json_provider_response_if_needed(
    raw: Any,
    diagnostics: dict[str, Any],
) -> Any:
    diagnostics["raw_response_kind"] = type(raw).__name__
    if not isinstance(raw, (str, bytes, bytearray)):
        return raw
    try:
        parsed = parse_internal_llm_json_response(raw)
    except Exception as exc:
        diagnostics["json_response_parse_error"] = f"{type(exc).__name__}: {exc}"
        return None
    diagnostics["json_response_parsed"] = True
    diagnostics["parsed_response_keys"] = sorted(str(key) for key in parsed.keys())
    return parsed


def _extract_candidate_items(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, PSAAbstraction):
        return [raw]
    if isinstance(raw, Mapping):
        for key in ("psa_abstractions", "state_abstractions", "candidates"):
            value = raw.get(key)
            if isinstance(value, PSAAbstraction):
                return [value]
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
                return list(value)
        return [raw]
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        return list(raw)
    return []


def _extract_hypothesis_items(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, MechanicHypothesis):
        return [raw]
    if isinstance(raw, Mapping):
        for key in ("mechanic_hypotheses", "hypotheses", "candidates"):
            value = raw.get(key)
            if isinstance(value, MechanicHypothesis):
                return [value]
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
                return list(value)
        return [raw]
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        return list(raw)
    return []


def _coerce_psa_abstraction(item: Any) -> PSAAbstraction | None:
    abstraction, _reason = _coerce_psa_abstraction_with_reason(item)
    return abstraction


def _coerce_psa_abstraction_with_reason(item: Any) -> tuple[PSAAbstraction | None, str]:
    if isinstance(item, PSAAbstraction):
        reason = _validate_psa_abstraction(item)
        return (None, reason) if reason else (item, "")
    if not isinstance(item, Mapping):
        return None, "not_mapping"
    if str(item.get("schema_version", "") or "") != PSA_PROVIDER_SCHEMA_VERSION:
        return None, "schema_version_mismatch"
    if item.get("natural_language_plan") or item.get("action_plan"):
        return None, "natural_language_plan_present"
    family = str(item.get("family", "") or "")
    if family != "cycle":
        return None, "unsupported_family"
    state_variables = _tuple_of_dicts_for_keys(
        item,
        "state_variables",
        "state_variable_candidates",
        "current_state",
        "current_state_candidate",
    )
    predicate_candidates = _tuple_of_dicts_for_keys(
        item,
        "predicate_candidates",
        "predicates",
        "predicate",
        "goal_predicate",
        "target_state",
        "target_state_candidate",
    )
    operator_candidates = _tuple_of_dicts_for_keys(
        item,
        "operator_candidates",
        "operators",
        "operator",
        "predicted_operator",
    )
    goal_link_candidates = _tuple_of_dicts_for_keys(
        item,
        "goal_link_candidates",
        "goal_links",
        "goal_link",
        "goal_link_candidate",
    )
    control_sites = _tuple_of_dicts_for_keys(item, "control_sites", "control_site")
    proposed_plan = _tuple_of_dicts_for_keys(item, "proposed_plan", "probe_plan", "verification_plan")
    if not state_variables or not predicate_candidates or not operator_candidates:
        return None, "missing_state_predicate_or_operator"
    if not goal_link_candidates or not control_sites or not proposed_plan:
        return None, "missing_goal_link_control_or_plan"

    source_kind = str(item.get("source_kind", "") or "psa_llm_proposed")
    llm_role = str(item.get("llm_role", "") or "candidate_generation")
    abstraction = PSAAbstraction(
        abstraction_id=str(item.get("abstraction_id", "") or f"{source_kind}:candidate"),
        family=family,
        source_kind=source_kind,
        source_event_ids=tuple(str(value) for value in _as_sequence(item.get("source_event_ids"))),
        state_variables=state_variables,
        predicate_candidates=predicate_candidates,
        operator_candidates=operator_candidates,
        goal_link_candidates=goal_link_candidates,
        control_sites=control_sites,
        controlled_objects=_tuple_of_dicts(item.get("controlled_objects")),
        proposed_plan=proposed_plan,
        evidence_frames=tuple(int(value) for value in _as_sequence(item.get("evidence_frames"))),
        effect_confidence=float(item.get("effect_confidence", 0.0) or 0.0),
        predicate_confidence=float(item.get("predicate_confidence", 0.0) or 0.0),
        contradiction_count=int(item.get("contradiction_count", 0) or 0),
        llm_role=llm_role,
        llm_direct_action=bool(item.get("llm_direct_action", False)),
        metadata=dict(item.get("metadata", {}) or {}),
    )
    reason = _validate_psa_abstraction(abstraction)
    return (None, reason) if reason else (abstraction, "")


def _coerce_mechanic_hypothesis(item: Any) -> MechanicHypothesis | None:
    hypothesis, _reason = _coerce_mechanic_hypothesis_with_reason(item)
    return hypothesis


def _coerce_mechanic_hypothesis_with_reason(item: Any) -> tuple[MechanicHypothesis | None, str]:
    if isinstance(item, MechanicHypothesis):
        reason = _validate_mechanic_hypothesis(item)
        return (None, reason) if reason else (item, "")
    if not isinstance(item, Mapping):
        return None, "not_mapping"
    if str(item.get("schema_version", "") or "") != ECA_INTERPRETER_SCHEMA_VERSION:
        return None, "schema_version_mismatch"
    if item.get("natural_language_plan") or item.get("action_plan"):
        return None, "natural_language_plan_present"
    family = str(item.get("family", "") or "")
    if family != "cycle":
        return None, "unsupported_family"
    proposed_plan = [dict(step) for step in _tuple_of_dicts(item.get("proposed_plan"))]
    control_sites = [dict(site) for site in _tuple_of_dicts(item.get("control_sites"))]
    if not proposed_plan or not control_sites:
        return None, "missing_control_or_plan"
    source_kind = str(item.get("source_kind", "") or "eca_llm_interpreted")
    llm_role = str(item.get("llm_role", "") or "hypothesis_interpretation")
    evidence = dict(item.get("evidence", {}) or {})
    evidence.setdefault("primitive_id", "eca_llm_interpreted_hypothesis")
    evidence.setdefault("source_kind", source_kind)
    evidence.setdefault("llm_role", llm_role)
    evidence.setdefault("llm_direct_action", bool(item.get("llm_direct_action", False)))
    hypothesis = MechanicHypothesis(
        hypothesis_id=str(item.get("hypothesis_id", "") or f"{source_kind}:hypothesis"),
        family=family,
        source_kind=source_kind,
        source_event_ids=tuple(str(value) for value in _as_sequence(item.get("source_event_ids"))),
        induced_variables=tuple(
            str(value)
            for value in (
                _as_sequence(item.get("induced_variables"))
                or ("eca_structural_deficit", "eca_verification_task", "eca_mechanic_hypothesis")
            )
        ),
        evidence_frames=[int(value) for value in _as_sequence(item.get("evidence_frames"))],
        controlled_objects=[dict(obj) for obj in _tuple_of_dicts(item.get("controlled_objects"))],
        control_sites=control_sites,
        current_state=dict(item.get("current_state", {}) or {}),
        target_state=dict(item.get("target_state", {}) or {}) or None,
        predicted_operator=dict(item.get("predicted_operator", {}) or {}) or None,
        goal_link_confidence=float(item.get("goal_link_confidence", 0.0) or 0.0),
        effect_confidence=float(item.get("effect_confidence", 0.0) or 0.0),
        contradiction_count=int(item.get("contradiction_count", 0) or 0),
        proposed_plan=proposed_plan,
        expected_goal_delta=float(item.get("expected_goal_delta", 0.0) or 0.0),
        source=str(item.get("source", "") or "internal_llm_interpreter_bridge"),
        status=str(item.get("status", "") or "tentative"),
        abort_conditions=[str(value) for value in _as_sequence(item.get("abort_conditions"))],
        falsification_conditions=tuple(
            str(value) for value in _as_sequence(item.get("falsification_conditions"))
        ),
        compiler_family=str(item.get("compiler_family", "") or "eca_llm_interpreted_plan_compiler"),
        executor_family=str(item.get("executor_family", "") or "mechanic_hypothesis_cycle_executor"),
        llm_role=llm_role,
        llm_direct_action=bool(item.get("llm_direct_action", False)),
        evidence=evidence,
    )
    reason = _validate_mechanic_hypothesis(hypothesis)
    return (None, reason) if reason else (hypothesis, "")


def _tuple_of_dicts(value: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(value, Mapping):
        return (dict(value),)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return tuple(dict(item) for item in value if isinstance(item, Mapping))
    return ()


def _tuple_of_dicts_for_keys(item: Mapping[str, Any], *keys: str) -> tuple[dict[str, Any], ...]:
    for key in keys:
        values = _tuple_of_dicts(item.get(key))
        if values:
            return values
    return ()


def _ground_psa_abstraction_in_evidence(
    abstraction: PSAAbstraction,
    *,
    records: Iterable[ActionEffectRecord],
    available_actions: Iterable[str],
) -> tuple[PSAAbstraction | None, str, bool]:
    record_list = list(records)
    available = {str(action or "") for action in available_actions if str(action or "")}
    valid_event_ids = {_source_event_id(record) for record in record_list}
    action_name, x, y = _action_site_from_abstraction(abstraction)
    source_ids_valid = bool(abstraction.source_event_ids) and (
        not valid_event_ids or all(str(event_id) in valid_event_ids for event_id in abstraction.source_event_ids)
    )
    action_valid = bool(action_name) and (not available or action_name in available)
    site_valid = action_name != "ACTION6" or (x is not None and y is not None)
    if source_ids_valid and action_valid and site_valid:
        return abstraction, "", False

    grounding = _latest_grounding_record(record_list, available_actions=available)
    if grounding is None:
        return None, "ungrounded_source_or_action", False

    record = grounding
    payload = dict(record.action_payload or {})
    gx = _optional_int(payload.get("x"))
    gy = _optional_int(payload.get("y"))
    if str(record.action_name) == "ACTION6" and (gx is None or gy is None):
        return None, "ungrounded_source_or_action", False
    bbox = record.changed_bbox.to_dict() if record.changed_bbox is not None else {}
    action_payload = {"x": int(gx), "y": int(gy)} if gx is not None and gy is not None else {}
    site = {"action_name": str(record.action_name), **action_payload}
    source_event_ids = (_source_event_id(record),)
    state_variable = _merge_first_dict(
        abstraction.state_variables,
        {
            "variable_id": f"psa_llm:grounded_state:{record.action_name}:{gx},{gy}",
            "variable_type": "visible_state_hash_transition",
            "pre_state_hash": str(record.pre_state_hash),
            "post_state_hash": str(record.post_state_hash),
            "control_site": dict(site),
            "changed_bbox": bbox,
            "confidence": float(abstraction.effect_confidence),
        },
        replace_placeholder_keys=("variable_id", "variable_type", "pre_state_hash", "post_state_hash"),
    )
    predicate = _merge_first_dict(
        abstraction.predicate_candidates,
        {
            "predicate_id": f"psa_llm:grounded_predicate:visible_state_advance:{record.action_name}:{gx},{gy}",
            "predicate_kind": "visible_state_advance",
            "score_delta": int(record.score_delta),
            "post_state_hash": str(record.post_state_hash),
            "confidence": float(abstraction.predicate_confidence),
        },
        replace_placeholder_keys=("predicate_id", "predicate_kind", "post_state_hash"),
    )
    operator = _merge_first_dict(
        abstraction.operator_candidates,
        {
            "operator_id": f"psa_llm:grounded_operator:repeat_effective_click:{record.action_name}:{gx},{gy}",
            "operator_family": "click_operator",
            "operation_kind": "repeat_effective_click",
            "action_name": str(record.action_name),
            "payload": dict(action_payload),
            "support_count": 1,
        },
        replace_placeholder_keys=("operator_id", "operator_family", "operation_kind", "action_name", "payload"),
    )
    goal_link = _merge_first_dict(
        abstraction.goal_link_candidates,
        {
            "goal_link_id": f"psa_llm:grounded_goal_link:visible_state_advance:{record.action_name}:{gx},{gy}",
            "goal_link_type": "visible_state_advance",
            "predicted_goal_delta": float(max(0, int(record.score_delta))),
            "predicate_id": str(predicate.get("predicate_id", "")),
            "operator_id": str(operator.get("operator_id", "")),
            "confidence": float(abstraction.predicate_confidence),
        },
        replace_placeholder_keys=("goal_link_id", "goal_link_type", "predicate_id", "operator_id"),
    )
    controlled_object = {
        "kind": "changed_region",
        "x": int(bbox.get("min_x", gx if gx is not None else 0) or 0),
        "y": int(bbox.get("min_y", gy if gy is not None else 0) or 0),
    }
    metadata = dict(abstraction.metadata or {})
    metadata.update(
        {
            "grounding_repair_applied": True,
            "grounding_repair_reason": _grounding_repair_reason(
                source_ids_valid=source_ids_valid,
                action_valid=action_valid,
                site_valid=site_valid,
            ),
            "grounding_record_step": int(record.step_index),
        }
    )
    return (
        replace(
            abstraction,
            source_event_ids=source_event_ids,
            state_variables=(state_variable,),
            predicate_candidates=(predicate,),
            operator_candidates=(operator,),
            goal_link_candidates=(goal_link,),
            control_sites=(site,),
            controlled_objects=(controlled_object,),
            proposed_plan=({"action_name": str(record.action_name), "payload": dict(action_payload)},),
            metadata=metadata,
        ),
        str(metadata["grounding_repair_reason"]),
        True,
    )


def _source_event_id(record: ActionEffectRecord) -> str:
    return f"action_effect:{int(record.step_index)}:{str(record.action_name)}"


def _action_site_from_abstraction(abstraction: PSAAbstraction) -> tuple[str, int | None, int | None]:
    step = abstraction.proposed_plan[0] if abstraction.proposed_plan else {}
    site = abstraction.control_sites[0] if abstraction.control_sites else {}
    action_name = str(
        step.get("action_name", "")
        or site.get("action_name", "")
        or ""
    )
    payload = step.get("payload") if isinstance(step.get("payload"), Mapping) else {}
    x = _optional_int(payload.get("x"))
    y = _optional_int(payload.get("y"))
    if x is None:
        x = _optional_int(site.get("x"))
    if y is None:
        y = _optional_int(site.get("y"))
    return action_name, x, y


def _latest_grounding_record(
    records: Iterable[ActionEffectRecord],
    *,
    available_actions: set[str],
) -> ActionEffectRecord | None:
    candidates: list[ActionEffectRecord] = []
    for record in records:
        if bool(record.no_op) or int(record.changed_pixel_count) <= 0:
            continue
        action_name = str(record.action_name)
        if available_actions and action_name not in available_actions:
            continue
        if action_name == "ACTION6":
            payload = dict(record.action_payload or {})
            if _optional_int(payload.get("x")) is None or _optional_int(payload.get("y")) is None:
                continue
        candidates.append(record)
    if not candidates:
        return None
    return max(candidates, key=lambda record: int(record.step_index))


def _merge_first_dict(
    values: Iterable[Mapping[str, Any]],
    fallback: Mapping[str, Any],
    *,
    replace_placeholder_keys: Iterable[str],
) -> dict[str, Any]:
    merged = dict(next(iter(values), {}) or {})
    for key, value in fallback.items():
        if key not in merged or str(merged.get(key, "") or "") in {"", "1", "N/A", "none", "None"}:
            merged[key] = value
    for key in replace_placeholder_keys:
        if str(merged.get(key, "") or "") in {"", "1", "N/A", "none", "None"}:
            merged[key] = fallback.get(key, merged.get(key))
    return merged


def _grounding_repair_reason(
    *,
    source_ids_valid: bool,
    action_valid: bool,
    site_valid: bool,
) -> str:
    missing: list[str] = []
    if not source_ids_valid:
        missing.append("source_event_ids")
    if not action_valid:
        missing.append("action_name")
    if not site_valid:
        missing.append("action_site")
    return "grounding_repaired:" + "+".join(missing or ["unknown"])


def _validate_psa_abstraction(abstraction: PSAAbstraction) -> str:
    if abstraction.family != "cycle":
        return "unsupported_family"
    if str(abstraction.source_kind or "") not in PSA_PROVIDER_SOURCE_KINDS:
        return "unsupported_source_kind"
    if str(abstraction.llm_role or "") not in PSA_PROVIDER_LLM_ROLES:
        return "unsupported_llm_role"
    if not _confidence_in_range(abstraction.effect_confidence):
        return "effect_confidence_out_of_range"
    if not _confidence_in_range(abstraction.predicate_confidence):
        return "predicate_confidence_out_of_range"
    if not abstraction.source_event_ids:
        return "missing_source_event_ids"
    if not abstraction.state_variables:
        return "missing_state_variables"
    if not abstraction.predicate_candidates:
        return "missing_predicate_candidates"
    if not abstraction.operator_candidates:
        return "missing_operator_candidates"
    if not abstraction.goal_link_candidates:
        return "missing_goal_link_candidates"
    if not abstraction.control_sites:
        return "missing_control_sites"
    if not abstraction.proposed_plan:
        return "missing_proposed_plan"
    for site in abstraction.control_sites:
        if _action_site_reason(site):
            return "invalid_control_site"
    for step in abstraction.proposed_plan:
        reason = _action_step_reason(step)
        if reason:
            return f"invalid_proposed_plan:{reason}"
    return ""


def _validate_mechanic_hypothesis(hypothesis: MechanicHypothesis) -> str:
    if hypothesis.family != "cycle":
        return "unsupported_family"
    if str(hypothesis.source_kind or "") not in ECA_INTERPRETER_SOURCE_KINDS:
        return "unsupported_source_kind"
    if str(hypothesis.llm_role or "") not in ECA_INTERPRETER_LLM_ROLES:
        return "unsupported_llm_role"
    if not _confidence_in_range(hypothesis.goal_link_confidence):
        return "goal_link_confidence_out_of_range"
    if not _confidence_in_range(hypothesis.effect_confidence):
        return "effect_confidence_out_of_range"
    if not hypothesis.source_event_ids:
        return "missing_source_event_ids"
    if not hypothesis.induced_variables:
        return "missing_induced_variables"
    if not hypothesis.current_state:
        return "missing_current_state"
    if not hypothesis.target_state:
        return "missing_target_state"
    if not hypothesis.predicted_operator:
        return "missing_predicted_operator"
    if not hypothesis.control_sites:
        return "missing_control_sites"
    if not hypothesis.proposed_plan:
        return "missing_proposed_plan"
    if not hypothesis.abort_conditions:
        return "missing_abort_conditions"
    if not hypothesis.falsification_conditions:
        return "missing_falsification_conditions"
    for site in hypothesis.control_sites:
        if _action_site_reason(site):
            return "invalid_control_site"
    for step in hypothesis.proposed_plan:
        reason = _action_step_reason(step)
        if reason:
            return f"invalid_proposed_plan:{reason}"
    return ""


def _confidence_in_range(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= numeric <= 1.0


def _action_site_reason(site: Mapping[str, Any]) -> str:
    action_name = str(site.get("action_name", "") or "")
    if not action_name:
        return "missing_action_name"
    if action_name == "ACTION6":
        try:
            int(site.get("x"))
            int(site.get("y"))
        except (TypeError, ValueError):
            return "missing_xy"
    return ""


def _action_step_reason(step: Mapping[str, Any]) -> str:
    action_name = str(step.get("action_name", "") or "")
    if not action_name:
        return "missing_action_name"
    payload = step.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    if action_name == "ACTION6":
        try:
            int(payload.get("x"))
            int(payload.get("y"))
        except (TypeError, ValueError):
            return "missing_xy"
    return ""


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def _frame_shape(frame: Any) -> list[int]:
    shape = getattr(frame, "shape", None)
    if shape is not None:
        try:
            return [int(value) for value in shape]
        except (TypeError, ValueError):
            return []
    if isinstance(frame, list):
        if frame and isinstance(frame[0], list):
            return [len(frame), len(frame[0])]
        return [len(frame)]
    return []


def _scalar_payload(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return str(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _frame_cell_count(frame: Any) -> int:
    shape = _frame_shape(frame)
    if len(shape) >= 2:
        return max(1, int(shape[0]) * int(shape[1]))
    if len(shape) == 1:
        return max(1, int(shape[0]))
    return 1


def _bbox_area(bbox: Mapping[str, Any] | None) -> int:
    if not isinstance(bbox, Mapping):
        return 0
    try:
        min_x = int(bbox.get("min_x"))
        min_y = int(bbox.get("min_y"))
        max_x = int(bbox.get("max_x"))
        max_y = int(bbox.get("max_y"))
    except (TypeError, ValueError):
        return 0
    return max(0, max_x - min_x + 1) * max(0, max_y - min_y + 1)


def _frame_value_summary(frame: Any, *, max_values: int = 16, max_cells: int = 50000) -> dict[str, Any]:
    shape = _frame_shape(frame)
    values: Iterable[Any]
    if hasattr(frame, "ravel"):
        try:
            values = frame.ravel()
        except Exception:
            values = ()
    elif isinstance(frame, list):
        if frame and isinstance(frame[0], list):
            values = (cell for row in frame for cell in row)
        else:
            values = frame
    else:
        values = ()

    counts: Counter[str] = Counter()
    sampled = 0
    for sampled, value in enumerate(values, start=1):
        if sampled > int(max_cells):
            break
        counts[str(_scalar_payload(value))] += 1
    top = [
        {"value": key, "count": int(count)}
        for key, count in counts.most_common(max(0, int(max_values)))
    ]
    return {
        "shape": shape,
        "sampled_cell_count": int(min(sampled, int(max_cells))) if sampled else 0,
        "unique_value_count": int(len(counts)),
        "value_counts_top": top,
        "nonzero_count": int(sum(count for key, count in counts.items() if key not in {"0", "0.0"})),
        "truncated": bool(sampled > int(max_cells)),
    }


def _compact_records(
    records: Iterable[ActionEffectRecord],
    *,
    max_records: int,
) -> list[dict[str, Any]]:
    items = list(records)[-max(0, int(max_records)) :]
    out: list[dict[str, Any]] = []
    for record in items:
        out.append(
            {
                "event_id": f"action_effect:{int(record.step_index)}:{str(record.action_name)}",
                "step_index": int(record.step_index),
                "action_name": str(record.action_name),
                "action_payload": dict(record.action_payload or {}),
                "pre_state_hash": str(record.pre_state_hash),
                "post_state_hash": str(record.post_state_hash),
                "score_delta": int(record.score_delta),
                "changed_pixel_count": int(record.changed_pixel_count),
                "changed_bbox": record.changed_bbox.to_dict() if record.changed_bbox else None,
                "no_op": bool(record.no_op),
            }
        )
    return out


def _valid_source_event_ids(
    records: Iterable[ActionEffectRecord],
    *,
    max_records: int,
) -> list[str]:
    return [
        str(item.get("event_id", ""))
        for item in _compact_records(records, max_records=max_records)
        if str(item.get("event_id", "") or "")
    ]


def _visible_effect_grammar_summary(
    records: Iterable[ActionEffectRecord],
    *,
    frame: Any,
    available_actions: Iterable[str],
    max_records: int,
) -> dict[str, Any]:
    items = list(records)[-max(0, int(max_records)) :]
    actions = [str(action) for action in available_actions]
    grouped: dict[tuple[str, int | None, int | None], list[ActionEffectRecord]] = {}
    for record in items:
        payload = dict(record.action_payload or {})
        x = _optional_int(payload.get("x"))
        y = _optional_int(payload.get("y"))
        grouped.setdefault((str(record.action_name), x, y), []).append(record)

    effective_sites: list[dict[str, Any]] = []
    for (action_name, x, y), group in sorted(
        grouped.items(),
        key=lambda item: (
            str(item[0][0]),
            -sum(int(record.changed_pixel_count) for record in item[1]),
            item[0][1] is None,
            item[0][1] or 0,
            item[0][2] or 0,
        ),
    ):
        latest = max(group, key=lambda record: int(record.step_index))
        changed = [record for record in group if int(record.changed_pixel_count) > 0]
        if not changed and int(latest.score_delta) == 0:
            continue
        bbox = latest.changed_bbox.to_dict() if latest.changed_bbox is not None else {}
        site: dict[str, Any] = {
            "action_name": action_name,
            "attempt_count": int(len(group)),
            "effective_count": int(len(changed)),
            "no_op_count": int(sum(1 for record in group if bool(record.no_op))),
            "score_delta_total": int(sum(int(record.score_delta) for record in group)),
            "changed_pixel_total": int(sum(int(record.changed_pixel_count) for record in group)),
            "latest_step_index": int(latest.step_index),
            "latest_pre_state_hash": str(latest.pre_state_hash),
            "latest_post_state_hash": str(latest.post_state_hash),
            "latest_changed_bbox": bbox,
            "latest_changed_bbox_area": int(_bbox_area(bbox)),
            "latest_segment_displacements": _sanitize_payload(latest.segment_displacements[:4]),
        }
        if x is not None and y is not None:
            site["x"] = int(x)
            site["y"] = int(y)
            site["control_site"] = {"action_name": action_name, "x": int(x), "y": int(y)}
        effective_sites.append(site)

    action_summaries: list[dict[str, Any]] = []
    for action in sorted({str(record.action_name) for record in items} | set(actions)):
        action_records = [record for record in items if str(record.action_name) == action]
        action_summaries.append(
            {
                "action_name": action,
                "attempt_count": int(len(action_records)),
                "effective_count": int(
                    sum(1 for record in action_records if int(record.changed_pixel_count) > 0)
                ),
                "no_op_count": int(sum(1 for record in action_records if bool(record.no_op))),
                "score_event_count": int(
                    sum(1 for record in action_records if int(record.score_delta) != 0)
                ),
            }
        )

    localized_sites = [
        site
        for site in effective_sites
        if int(site.get("latest_changed_bbox_area", 0) or 0) > 0
        and int(site.get("latest_changed_bbox_area", 0) or 0)
        <= max(4, int(0.20 * _frame_cell_count(frame)))
    ]
    return {
        "schema_version": "visible_effect_grammar_summary.v1",
        "purpose": "candidate_generation_only_not_solver",
        "available_actions": actions,
        "record_count": int(len(items)),
        "effective_site_count": int(len(effective_sites)),
        "localized_effect_site_count": int(len(localized_sites)),
        "repeated_effect_site_count": int(
            sum(1 for site in effective_sites if int(site.get("effective_count", 0) or 0) >= 2)
        ),
        "score_event_count": int(sum(1 for record in items if int(record.score_delta) != 0)),
        "action_summaries": action_summaries[:8],
        "effective_control_sites": effective_sites[:8],
        "cycle_candidate_requirements": {
            "state_variable": "visible quantity or region whose value changes after a control action",
            "predicate": "testable condition that the changed state satisfies or moves toward",
            "operator": "action_name plus payload that caused the observed transition",
            "goal_link": "why the predicate could reduce mismatch or enable score; use visible evidence only",
        },
        "empty_response_when": [
            "no effective control site",
            "no state variable candidate",
            "no predicate candidate",
            "no operator candidate",
        ],
    }


def _psa_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["psa_abstractions"],
        "additionalProperties": False,
        "properties": {
            "psa_abstractions": {
                "type": "array",
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "required": [
                        "schema_version",
                        "abstraction_id",
                        "family",
                        "source_kind",
                        "source_event_ids",
                        "state_variables",
                        "predicate_candidates",
                        "operator_candidates",
                        "goal_link_candidates",
                        "control_sites",
                        "proposed_plan",
                        "effect_confidence",
                        "predicate_confidence",
                        "llm_role",
                    ],
                    "properties": {
                        "schema_version": {"const": PSA_PROVIDER_SCHEMA_VERSION},
                        "abstraction_id": {"type": "string", "minLength": 1},
                        "family": {"const": "cycle"},
                        "source_kind": {"enum": sorted(PSA_PROVIDER_SOURCE_KINDS)},
                        "source_event_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 4,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "state_variables": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["variable_id", "variable_type"],
                                "properties": {
                                    "variable_id": {"type": "string", "minLength": 1},
                                    "variable_type": {"type": "string", "minLength": 1},
                                },
                                "additionalProperties": True,
                            },
                        },
                        "predicate_candidates": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["predicate_id", "predicate_kind"],
                                "properties": {
                                    "predicate_id": {"type": "string", "minLength": 1},
                                    "predicate_kind": {"type": "string", "minLength": 1},
                                },
                                "additionalProperties": True,
                            },
                        },
                        "operator_candidates": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["operator_id", "operator_family", "operation_kind"],
                                "properties": {
                                    "operator_id": {"type": "string", "minLength": 1},
                                    "operator_family": {"type": "string", "minLength": 1},
                                    "operation_kind": {"type": "string", "minLength": 1},
                                    "action_name": {"type": "string", "minLength": 1},
                                },
                                "additionalProperties": True,
                            },
                        },
                        "goal_link_candidates": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["goal_link_id", "goal_link_type"],
                                "properties": {
                                    "goal_link_id": {"type": "string", "minLength": 1},
                                    "goal_link_type": {"type": "string", "minLength": 1},
                                    "predicted_goal_delta": {"type": "number"},
                                },
                                "additionalProperties": True,
                            },
                        },
                        "control_sites": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["action_name", "x", "y"],
                                "properties": {
                                    "action_name": {"type": "string", "minLength": 1},
                                    "x": {"type": "integer"},
                                    "y": {"type": "integer"},
                                },
                                "additionalProperties": True,
                            },
                        },
                        "controlled_objects": {
                            "type": "array",
                            "maxItems": 1,
                            "items": {"type": "object", "additionalProperties": True},
                        },
                        "proposed_plan": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["action_name", "payload"],
                                "properties": {
                                    "action_name": {"type": "string", "minLength": 1},
                                    "payload": {
                                        "type": "object",
                                        "required": ["x", "y"],
                                        "properties": {
                                            "x": {"type": "integer"},
                                            "y": {"type": "integer"},
                                        },
                                        "additionalProperties": True,
                                    },
                                },
                                "additionalProperties": True,
                            },
                        },
                        "evidence_frames": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "effect_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "predicate_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "llm_role": {"enum": sorted(PSA_PROVIDER_LLM_ROLES)},
                        "llm_direct_action": {"type": "boolean"},
                    },
                    "additionalProperties": True,
                },
            }
        },
    }


def _eca_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["mechanic_hypotheses"],
        "properties": {
            "mechanic_hypotheses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "schema_version",
                        "hypothesis_id",
                        "family",
                        "source_kind",
                        "source_event_ids",
                        "induced_variables",
                        "current_state",
                        "target_state",
                        "predicted_operator",
                        "control_sites",
                        "proposed_plan",
                        "goal_link_confidence",
                        "effect_confidence",
                        "abort_conditions",
                        "falsification_conditions",
                        "llm_role",
                    ],
                    "properties": {
                        "schema_version": {"const": ECA_INTERPRETER_SCHEMA_VERSION},
                        "family": {"const": "cycle"},
                        "source_kind": {"enum": sorted(ECA_INTERPRETER_SOURCE_KINDS)},
                        "llm_role": {"enum": sorted(ECA_INTERPRETER_LLM_ROLES)},
                    },
                    "additionalProperties": True,
                },
            }
        },
    }


def _psa_response_template() -> dict[str, Any]:
    return {
        "psa_abstractions": [
            {
                "schema_version": PSA_PROVIDER_SCHEMA_VERSION,
                "abstraction_id": "psa_llm:cycle:ACTION6:x:y",
                "family": "cycle",
                "source_kind": "psa_llm_proposed",
                "source_event_ids": ["action_effect:step:ACTION6"],
                "state_variables": [
                    {
                        "variable_id": "psa_llm:visible_state_hash:ACTION6:x,y",
                        "variable_type": "visible_state_hash_transition",
                        "pre_state_hash": "record.pre_state_hash",
                        "post_state_hash": "record.post_state_hash",
                        "control_site": {"action_name": "ACTION6", "x": 0, "y": 0},
                        "changed_bbox": {
                            "min_x": 0,
                            "min_y": 0,
                            "max_x": 0,
                            "max_y": 0,
                        },
                        "confidence": 0.5,
                    }
                ],
                "predicate_candidates": [
                    {
                        "predicate_id": "psa_llm:predicate:visible_state_advance",
                        "predicate_kind": "visible_state_advance",
                        "post_state_hash": "record.post_state_hash",
                        "confidence": 0.5,
                    }
                ],
                "operator_candidates": [
                    {
                        "operator_id": "psa_llm:operator:repeat_effective_click:ACTION6:x,y",
                        "operator_family": "click_operator",
                        "operation_kind": "repeat_effective_click",
                        "action_name": "ACTION6",
                        "payload": {"x": 0, "y": 0},
                        "support_count": 1,
                    }
                ],
                "goal_link_candidates": [
                    {
                        "goal_link_id": "psa_llm:goal_link:visible_state_advance",
                        "goal_link_type": "visible_state_advance",
                        "predicted_goal_delta": 0.0,
                        "predicate_id": "psa_llm:predicate:visible_state_advance",
                        "operator_id": "psa_llm:operator:repeat_effective_click:ACTION6:x,y",
                        "confidence": 0.5,
                    }
                ],
                "control_sites": [{"action_name": "ACTION6", "x": 0, "y": 0}],
                "controlled_objects": [{"kind": "changed_region", "x": 0, "y": 0}],
                "proposed_plan": [{"action_name": "ACTION6", "payload": {"x": 0, "y": 0}}],
                "evidence_frames": [0],
                "effect_confidence": 0.5,
                "predicate_confidence": 0.5,
                "llm_role": "candidate_generation",
                "llm_direct_action": False,
            }
        ]
    }


def _eca_response_template() -> dict[str, Any]:
    return {
        "mechanic_hypotheses": [
            {
                "schema_version": ECA_INTERPRETER_SCHEMA_VERSION,
                "hypothesis_id": "eca_llm:cycle:ACTION6:x:y",
                "family": "cycle",
                "source_kind": "eca_llm_interpreted",
                "source_event_ids": ["psa_llm:proposal:0", "eca_llm:interpretation:0"],
                "induced_variables": [
                    "eca_structural_deficit",
                    "eca_verification_task",
                    "eca_mechanic_hypothesis",
                ],
                "evidence_frames": [0],
                "controlled_objects": [{"kind": "changed_region", "x": 0, "y": 0}],
                "control_sites": [{"action_name": "ACTION6", "x": 0, "y": 0}],
                "current_state": {
                    "mechanic_state_type": "eca_llm_interpreted_visible_state_variable",
                    "state_variable_id": "psa_llm:visible_state_hash:ACTION6:x,y",
                    "action_name": "ACTION6",
                    "control_site": [0, 0],
                },
                "target_state": {
                    "goal_link_type": "visible_state_advance",
                    "post_state_hash": "record.post_state_hash",
                    "score_delta": 0,
                },
                "predicted_operator": {
                    "operator_family": "click_operator",
                    "operation_kind": "repeat_effective_click",
                    "action_names": ["ACTION6"],
                    "operation_count": 1,
                },
                "goal_link_confidence": 0.5,
                "effect_confidence": 0.5,
                "contradiction_count": 0,
                "proposed_plan": [{"action_name": "ACTION6", "payload": {"x": 0, "y": 0}}],
                "expected_goal_delta": 0.0,
                "abort_conditions": ["unexpected_effect"],
                "falsification_conditions": ["no_visible_delta_after_probe"],
                "compiler_family": "eca_llm_interpreted_plan_compiler",
                "executor_family": "mechanic_hypothesis_cycle_executor",
                "llm_role": "hypothesis_interpretation",
                "llm_direct_action": False,
            }
        ]
    }


def _to_payload(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            return str(value)
    return _sanitize_payload(value)


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_payload(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _sanitize_payload(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


__all__ = [
    "ECA_INTERPRETER_SCHEMA_VERSION",
    "InternalLLMPromptPacket",
    "InternalLLMCognitionBridge",
    "InternalLLMCognitionResult",
    "InternalLLMInterpretationResult",
    "InternalLLMInterpreterBridge",
    "InternalLLMProvider",
    "PSA_PROVIDER_SCHEMA_VERSION",
    "build_eca_llm_interpreter_prompt_packet",
    "build_psa_llm_prompt_packet",
    "internal_llm_prompt_packet_audit",
    "parse_internal_llm_json_response",
    "resolve_internal_llm_provider_path",
]
