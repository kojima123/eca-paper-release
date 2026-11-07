"""Ollama/Gemma E4B providers for PSA/ECA-internal LLM cognition.

The provider functions in this module are intentionally thin backend adapters.
They consume the Phase 180 prompt-packet contract, call Ollama's
OpenAI-compatible chat endpoint, parse a JSON object, and return it to the
existing PSA/ECA bridges for schema enforcement.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from arcgym.utils import llm_token_tracker

from .internal_llm_cognition import parse_internal_llm_json_response


DEFAULT_GEMMA_E4B_MODEL = "gemma4:e4b"
PSA_GEMMA_E4B_CALL_SITE = "internal_llm_psa"
ECA_GEMMA_E4B_CALL_SITE = "internal_llm_eca"
COORDINATE_HYPOTHESIS_GEMMA_E4B_CALL_SITE = "internal_llm_coordinate_hypothesis"


def psa_gemma_e4b_provider(**kwargs: Any) -> Any:
    """Return PSA candidate JSON from Gemma E4B through Ollama."""

    return _call_ollama_prompt_packet(
        prompt_packet=kwargs.get("prompt_packet"),
        call_site=PSA_GEMMA_E4B_CALL_SITE,
        workdir=_workdir_from_kwargs(kwargs),
    )


def eca_gemma_e4b_provider(**kwargs: Any) -> Any:
    """Return ECA mechanic-hypothesis JSON from Gemma E4B through Ollama."""

    return _call_ollama_prompt_packet(
        prompt_packet=kwargs.get("prompt_packet"),
        call_site=ECA_GEMMA_E4B_CALL_SITE,
        workdir=_workdir_from_kwargs(kwargs),
    )


def coordinate_hypothesis_gemma_e4b_provider(**kwargs: Any) -> Any:
    """Return enum-only coordinate-hypothesis JSON from Gemma E4B."""

    return _call_ollama_prompt_packet(
        prompt_packet=kwargs.get("prompt_packet"),
        call_site=COORDINATE_HYPOTHESIS_GEMMA_E4B_CALL_SITE,
        workdir=_workdir_from_kwargs(kwargs),
    )


def _call_ollama_prompt_packet(
    *,
    prompt_packet: Any,
    call_site: str,
    workdir: Path,
) -> Any:
    import requests as _requests

    packet = _coerce_prompt_packet(prompt_packet)
    direct_slot_fill = _direct_slot_fill_enabled(packet)
    if direct_slot_fill:
        system_prompt = _direct_slot_fill_system_prompt()
        user_content = _direct_slot_fill_user_content(packet)
    else:
        system_prompt = str(packet.get("system_prompt", "") or "")
        user_content = _prompt_packet_user_content(packet)
    model = _internal_llm_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    temperature = _env_float("ECA_INTERNAL_LLM_TEMPERATURE", 0.1)
    json_response_format = _env_bool(
        "ECA_INTERNAL_LLM_JSON_RESPONSE_FORMAT",
        fallback_name="OLLAMA_JSON_RESPONSE_FORMAT",
        default=True,
    )
    format_mode = _env_first(
        "ECA_INTERNAL_LLM_OLLAMA_FORMAT_MODE",
        "OLLAMA_FORMAT_MODE",
        default="schema",
        ignore_values={""},
    ).strip().lower()
    think = _env_bool(
        "ECA_INTERNAL_LLM_THINK",
        fallback_name="OLLAMA_THINK",
        default=False,
    )
    max_tokens = _env_int(
        "ECA_INTERNAL_LLM_MAX_TOKENS",
        fallback_name="OLLAMA_MAX_TOKENS",
        default=1024,
    )

    base_url = _env_first(
        "ECA_INTERNAL_LLM_OLLAMA_BASE_URL",
        "OLLAMA_BASE_URL",
        default="http://localhost:11434",
    ).rstrip("/")
    endpoint = _env_first(
        "ECA_INTERNAL_LLM_OLLAMA_ENDPOINT",
        "OLLAMA_ENDPOINT",
        default="openai_chat",
        ignore_values={"", "none", "disabled"},
    ).strip().lower()
    use_native_chat = endpoint in {"native", "native_chat", "api", "api_chat", "ollama_chat"}
    if use_native_chat:
        api_url = f"{base_url}/api/chat"
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {
                "temperature": temperature,
                "num_predict": max(1, int(max_tokens)) if max_tokens is not None else 1024,
            },
        }
        num_ctx = _env_int("ECA_INTERNAL_LLM_NUM_CTX", fallback_name="OLLAMA_NUM_CTX", default=None)
        if num_ctx is not None:
            body["options"]["num_ctx"] = max(128, int(num_ctx))
        for option_key, env_name, fallback_name in (
            ("num_gpu", "ECA_INTERNAL_LLM_NUM_GPU", "OLLAMA_NUM_GPU"),
            ("num_thread", "ECA_INTERNAL_LLM_NUM_THREAD", "OLLAMA_NUM_THREAD"),
            ("num_batch", "ECA_INTERNAL_LLM_NUM_BATCH", "OLLAMA_NUM_BATCH"),
            ("seed", "ECA_INTERNAL_LLM_SEED", "OLLAMA_SEED"),
            ("top_k", "ECA_INTERNAL_LLM_TOP_K", "OLLAMA_TOP_K"),
        ):
            option_value = _env_int(env_name, fallback_name=fallback_name, default=None)
            if option_value is not None:
                body["options"][option_key] = int(option_value)
        for option_key, env_name, fallback_name in (
            ("top_p", "ECA_INTERNAL_LLM_TOP_P", "OLLAMA_TOP_P"),
            ("min_p", "ECA_INTERNAL_LLM_MIN_P", "OLLAMA_MIN_P"),
            ("repeat_penalty", "ECA_INTERNAL_LLM_REPEAT_PENALTY", "OLLAMA_REPEAT_PENALTY"),
        ):
            option_value = _env_float_optional(env_name, fallback_name=fallback_name)
            if option_value is not None:
                body["options"][option_key] = float(option_value)
        keep_alive = _env_first(
            "ECA_INTERNAL_LLM_KEEP_ALIVE",
            "OLLAMA_KEEP_ALIVE",
            default="",
            ignore_values={""},
        ).strip()
        if keep_alive:
            body["keep_alive"] = keep_alive
        if json_response_format and format_mode not in {"none", "off", "disabled"}:
            schema = (
                _direct_slot_fill_response_schema(packet)
                if direct_slot_fill
                else _response_schema_for_prompt_packet(packet)
            )
            if format_mode in {"json", "json_object", "simple_json"}:
                body["format"] = "json"
            else:
                body["format"] = schema if isinstance(schema, Mapping) and schema else "json"
    else:
        api_url = f"{base_url}/v1/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "think": think,
        }
        if json_response_format and format_mode not in {"none", "off", "disabled"}:
            body["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            body["max_tokens"] = max(1, int(max_tokens))
    timeout = _env_int("ECA_INTERNAL_LLM_TIMEOUT_SECONDS", default=180) or 180
    attempt_bodies = _ollama_attempt_bodies(
        body,
        use_native_chat=use_native_chat,
        json_response_format=json_response_format,
    )

    prompt_chars = len(system_prompt) + len(user_content)
    started = time.perf_counter()
    status = "ok"
    response_chars = 0
    prompt_tokens = completion_tokens = total_tokens = None
    try:
        last_error: Exception | None = None
        for attempt_index, attempt_body in enumerate(attempt_bodies):
            try:
                response = _requests.post(api_url, json=attempt_body, timeout=timeout)
                _raise_for_status_with_ollama_diagnostics(response, api_url=api_url, request_body=attempt_body)
                data = response.json()
                usage = data.get("usage") if isinstance(data, Mapping) else None
                if isinstance(usage, Mapping):
                    prompt_tokens = _optional_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
                    completion_tokens = _optional_int(
                        usage.get("completion_tokens") or usage.get("output_tokens")
                    )
                    total_tokens = _optional_int(usage.get("total_tokens"))
                if use_native_chat and isinstance(data, Mapping):
                    prompt_tokens = _optional_int(data.get("prompt_eval_count")) or prompt_tokens
                    completion_tokens = _optional_int(data.get("eval_count")) or completion_tokens
                    if prompt_tokens is not None and completion_tokens is not None:
                        total_tokens = int(prompt_tokens) + int(completion_tokens)
                content = (
                    _extract_native_chat_content(data) if use_native_chat else _extract_chat_content(data)
                )
                response_chars = len(content)
                if direct_slot_fill and _direct_slot_fill_mode() == "line_seed":
                    materialized = _direct_slot_fill_line_seed_response(packet, content)
                    if materialized is not None:
                        return materialized
                try:
                    parsed = _parse_json_object_from_content(content)
                except Exception:
                    if _env_bool("ECA_INTERNAL_LLM_RETURN_RAW_ON_PARSE_ERROR", default=True):
                        return content
                    raise
                if direct_slot_fill and _direct_slot_fill_mode() in {
                    "field_copy",
                    "seed_selector",
                    "seed_selector_first",
                    "weak_seed",
                }:
                    materialized = _direct_slot_fill_selector_response(packet, parsed)
                    if materialized is not None:
                        return materialized
                if _should_run_contract_repair_retry(packet, parsed):
                    repair_body = _ollama_contract_repair_body(
                        attempt_body,
                        packet=packet,
                        raw_response=parsed,
                        use_native_chat=use_native_chat,
                    )
                    prompt_chars += _message_chars(repair_body.get("messages"))
                    repair_response = _requests.post(api_url, json=repair_body, timeout=timeout)
                    _raise_for_status_with_ollama_diagnostics(
                        repair_response,
                        api_url=api_url,
                        request_body=repair_body,
                    )
                    repair_data = repair_response.json()
                    repair_usage = repair_data.get("usage") if isinstance(repair_data, Mapping) else None
                    if isinstance(repair_usage, Mapping):
                        prompt_tokens = _sum_optional_ints(
                            prompt_tokens,
                            _optional_int(repair_usage.get("prompt_tokens") or repair_usage.get("input_tokens")),
                        )
                        completion_tokens = _sum_optional_ints(
                            completion_tokens,
                            _optional_int(
                                repair_usage.get("completion_tokens") or repair_usage.get("output_tokens")
                            ),
                        )
                        repair_total = _optional_int(repair_usage.get("total_tokens"))
                        total_tokens = _sum_optional_ints(total_tokens, repair_total)
                    if use_native_chat and isinstance(repair_data, Mapping):
                        repair_prompt_tokens = _optional_int(repair_data.get("prompt_eval_count"))
                        repair_completion_tokens = _optional_int(repair_data.get("eval_count"))
                        prompt_tokens = _sum_optional_ints(prompt_tokens, repair_prompt_tokens)
                        completion_tokens = _sum_optional_ints(completion_tokens, repair_completion_tokens)
                        if prompt_tokens is not None and completion_tokens is not None:
                            total_tokens = int(prompt_tokens) + int(completion_tokens)
                    repair_content = (
                        _extract_native_chat_content(repair_data)
                        if use_native_chat
                        else _extract_chat_content(repair_data)
                    )
                    response_chars += len(repair_content)
                    try:
                        repaired = _parse_json_object_from_content(repair_content)
                    except Exception:
                        if _contract_repair_empty_on_failure():
                            return _empty_contract_response(packet)
                        if _env_bool("ECA_INTERNAL_LLM_RETURN_RAW_ON_PARSE_ERROR", default=True):
                            return content
                        raise
                    if _contract_response_matches_expected(packet, repaired):
                        return repaired
                    if _contract_repair_empty_on_failure():
                        return _empty_contract_response(packet)
                if direct_slot_fill and not _contract_response_matches_expected(packet, parsed):
                    if _contract_repair_empty_on_failure():
                        return _empty_contract_response(packet)
                return parsed
            except Exception as exc:
                last_error = exc
                if attempt_index + 1 >= len(attempt_bodies) or not _should_retry_ollama_attempt(exc):
                    raise
                sleep_seconds = _env_float("ECA_INTERNAL_LLM_OLLAMA_RETRY_SLEEP_SECONDS", 0.0)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("Ollama provider had no request attempts")
    except Exception:
        status = "error"
        raise
    finally:
        try:
            llm_token_tracker.record(
                workdir,
                provider="ollama",
                model=model,
                call_site=call_site,
                prompt_chars=prompt_chars,
                response_chars=response_chars,
                wall_seconds=time.perf_counter() - started,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                status=status,
            )
        except Exception:
            pass


def _coerce_prompt_packet(prompt_packet: Any) -> dict[str, Any]:
    if not isinstance(prompt_packet, Mapping):
        raise ValueError("Gemma E4B internal LLM provider requires a prompt_packet mapping")
    return dict(prompt_packet)


def _direct_slot_fill_enabled(packet: Mapping[str, Any]) -> bool:
    if str(packet.get("expected_schema_version", "") or "") != "internal_llm_psa_candidate.v1":
        return False
    return _env_bool(
        "ECA_INTERNAL_LLM_DIRECT_SLOT_FILL",
        fallback_name="OLLAMA_DIRECT_SLOT_FILL",
        default=False,
    )


def _direct_slot_fill_system_prompt() -> str:
    mode = _direct_slot_fill_mode()
    if mode == "line_seed":
        return (
            "Return exactly one line: FAMILY=cycle OPERATOR=click "
            "TARGET_HINT=button EFFECT_HINT=state_change. "
            "Do not solve the game. Do not explain. /no_think"
        )
    if mode == "weak_seed":
        return (
            "Return exactly one JSON object with weak mechanic seed labels. "
            "Do not solve the game. Do not explain. Do not emit a schema. /no_think"
        )
    if mode == "field_copy":
        return (
            "Return exactly one JSON object copying visible-effect binding fields. "
            "Do not solve the game. Do not explain. Do not emit a schema. /no_think"
        )
    if mode == "seed_selector_first":
        return (
            "Return exactly one JSON object selecting the first valid candidate seed id. "
            "Do not solve the game. Do not explain. Do not emit a schema. /no_think"
        )
    if mode == "seed_selector":
        return (
            "Return exactly one JSON object selecting a candidate_seed_id. "
            "Do not solve the game. Do not explain. Do not emit a schema. /no_think"
        )
    return (
        "Return exactly the JSON candidate_seed from the user message. "
        "Do not solve the game. Do not explain. Do not emit a schema. /no_think"
    )


def _direct_slot_fill_mode() -> str:
    mode = _env_first(
        "ECA_INTERNAL_LLM_DIRECT_SLOT_FILL_MODE",
        "OLLAMA_DIRECT_SLOT_FILL_MODE",
        default="echo",
        ignore_values={""},
    ).strip().lower()
    if mode in {"selector", "seed_selector", "select", "id_select", "candidate_selector"}:
        return "seed_selector"
    if mode in {"selector_first", "seed_selector_first", "select_first", "first_valid"}:
        return "seed_selector_first"
    if mode in {"field_copy", "fieldcopy", "copy_fields", "evidence_field_copy"}:
        return "field_copy"
    if mode in {"weak_seed", "weak_seed_json", "seed_labels", "mechanic_seed"}:
        return "weak_seed"
    if mode in {"line_seed", "weak_seed_line", "line_protocol", "seed_line"}:
        return "line_seed"
    return "echo"


def _direct_slot_fill_user_content(packet: Mapping[str, Any]) -> str:
    user_payload = packet.get("user_payload", {})
    user_payload = user_payload if isinstance(user_payload, Mapping) else {}
    expected_schema = str(packet.get("expected_schema_version", "") or "")
    candidate_seed = _psa_slot_fill_response_from_payload(
        user_payload,
        expected_schema=expected_schema,
    )
    mode = _direct_slot_fill_mode()
    if mode == "line_seed":
        contract = _weak_seed_contract(candidate_seed)
        payload = {
            "task": "direct_slot_fill_line_seed",
            "allowed_line": "FAMILY=cycle OPERATOR=click TARGET_HINT=button EFFECT_HINT=state_change",
            "candidate_seed_summary": _candidate_seed_summary(candidate_seed),
            "weak_seed_contract": contract,
            "instruction": (
                "Return exactly allowed_line and nothing else if "
                "weak_seed_contract.has_candidate is true. Otherwise return REJECT."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if mode == "weak_seed":
        contract = _weak_seed_contract(candidate_seed)
        payload = {
            "task": "direct_slot_fill_weak_seed",
            "allowed_top_level_keys": ["family", "operator", "target_hint", "effect_hint"],
            "allowed_values": contract.get("expected", {}),
            "candidate_seed_summary": _candidate_seed_summary(candidate_seed),
            "instruction": (
                "Return exactly the four allowed top-level keys by copying "
                "allowed_values when weak_seed_contract.has_candidate is true. "
                "If false, return empty strings for all four keys. No markdown, no prose."
            ),
            "weak_seed_contract": contract,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if mode == "field_copy":
        contract = _field_copy_contract(candidate_seed)
        payload = {
            "task": "direct_slot_fill_field_copy",
            "allowed_top_level_keys": [
                "source_event_id",
                "action_name",
                "x",
                "y",
                "pre_state_hash",
                "post_state_hash",
            ],
            "selection_policy": "copy_exact_visible_effect_binding_fields",
            "field_contract": contract,
            "instruction": (
                "Return exactly the six allowed top-level keys by copying the "
                "values from field_contract.expected. If field_contract.has_candidate "
                "is false, return empty strings for string fields and 0 for x/y. "
                "No markdown, no prose."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if mode.startswith("seed_selector"):
        candidate_seed_id = _candidate_seed_id(candidate_seed)
        instruction = (
            "Return {\"selected_candidate_seed_id\": candidate_seed_id} if "
            "candidate_seed_summary.has_candidate is true and the candidate is "
            "grounded by the listed source_event_ids. Otherwise return "
            "{\"selected_candidate_seed_id\":\"\"}. No markdown, no prose."
        )
        if mode == "seed_selector_first":
            instruction = (
                "If valid_selected_candidate_seed_ids is non-empty, return exactly "
                "{\"selected_candidate_seed_id\": valid_selected_candidate_seed_ids[0]}. "
                "Only return an empty selected_candidate_seed_id when the valid list is empty. "
                "No markdown, no prose."
            )
        payload = {
            "task": "direct_slot_fill_candidate_seed_selector",
            "allowed_top_level_keys": ["selected_candidate_seed_id"],
            "selection_policy": (
                "select_first_valid_candidate_seed_id"
                if mode == "seed_selector_first"
                else "select_grounded_candidate_seed_id"
            ),
            "candidate_seed_id": candidate_seed_id,
            "valid_selected_candidate_seed_ids": [candidate_seed_id] if candidate_seed_id else [],
            "candidate_seed_summary": _candidate_seed_summary(candidate_seed),
            "instruction": instruction,
        }
        if mode == "seed_selector":
            payload["empty_response"] = {"selected_candidate_seed_id": ""}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    payload = {
        "task": "direct_slot_fill_candidate_echo",
        "allowed_top_level_keys": ["psa_abstractions"],
        "candidate_seed": candidate_seed,
        "instruction": (
            "Return candidate_seed exactly as the entire JSON response. "
            "If candidate_seed.psa_abstractions is empty, return it exactly. "
            "No markdown, no prose, no JSON Schema."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _direct_slot_fill_response_schema(packet: Mapping[str, Any]) -> Mapping[str, Any] | None:
    mode = _direct_slot_fill_mode()
    if mode == "weak_seed":
        user_payload = packet.get("user_payload", {})
        user_payload = user_payload if isinstance(user_payload, Mapping) else {}
        expected_schema = str(packet.get("expected_schema_version", "") or "")
        candidate_seed = _psa_slot_fill_response_from_payload(
            user_payload,
            expected_schema=expected_schema,
        )
        expected = _weak_seed_contract(candidate_seed).get("expected", {})
        expected = expected if isinstance(expected, Mapping) else {}
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["family", "operator", "target_hint", "effect_hint"],
            "properties": {
                "family": {"type": "string", "enum": [str(expected.get("family", "") or "")]},
                "operator": {"type": "string", "enum": [str(expected.get("operator", "") or "")]},
                "target_hint": {
                    "type": "string",
                    "enum": [str(expected.get("target_hint", "") or "")],
                },
                "effect_hint": {
                    "type": "string",
                    "enum": [str(expected.get("effect_hint", "") or "")],
                },
            },
        }
    if mode == "field_copy":
        user_payload = packet.get("user_payload", {})
        user_payload = user_payload if isinstance(user_payload, Mapping) else {}
        expected_schema = str(packet.get("expected_schema_version", "") or "")
        candidate_seed = _psa_slot_fill_response_from_payload(
            user_payload,
            expected_schema=expected_schema,
        )
        expected = _field_copy_contract(candidate_seed).get("expected", {})
        expected = expected if isinstance(expected, Mapping) else {}
        source_event_id = str(expected.get("source_event_id", "") or "")
        action_name = str(expected.get("action_name", "") or "")
        pre_state_hash = str(expected.get("pre_state_hash", "") or "")
        post_state_hash = str(expected.get("post_state_hash", "") or "")
        x = _optional_int(expected.get("x"))
        y = _optional_int(expected.get("y"))
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "source_event_id",
                "action_name",
                "x",
                "y",
                "pre_state_hash",
                "post_state_hash",
            ],
            "properties": {
                "source_event_id": {
                    "type": "string",
                    "enum": [source_event_id] if source_event_id else [""],
                },
                "action_name": {"type": "string", "enum": [action_name] if action_name else [""]},
                "x": {"type": "integer", "enum": [int(x)] if x is not None else [0]},
                "y": {"type": "integer", "enum": [int(y)] if y is not None else [0]},
                "pre_state_hash": {
                    "type": "string",
                    "enum": [pre_state_hash] if pre_state_hash else [""],
                },
                "post_state_hash": {
                    "type": "string",
                    "enum": [post_state_hash] if post_state_hash else [""],
                },
            },
        }
    if mode.startswith("seed_selector"):
        user_payload = packet.get("user_payload", {})
        user_payload = user_payload if isinstance(user_payload, Mapping) else {}
        expected_schema = str(packet.get("expected_schema_version", "") or "")
        candidate_seed = _psa_slot_fill_response_from_payload(
            user_payload,
            expected_schema=expected_schema,
        )
        candidate_seed_id = _candidate_seed_id(candidate_seed)
        if mode == "seed_selector_first":
            allowed_ids = [candidate_seed_id] if candidate_seed_id else [""]
        else:
            allowed_ids = [""]
            if candidate_seed_id:
                allowed_ids.append(candidate_seed_id)
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["selected_candidate_seed_id"],
            "properties": {
                "selected_candidate_seed_id": {
                    "type": "string",
                    "enum": allowed_ids,
                }
            },
        }
    schema = packet.get("response_schema")
    return schema if isinstance(schema, Mapping) else None


def _response_schema_for_prompt_packet(packet: Mapping[str, Any]) -> Mapping[str, Any] | None:
    mode = _env_first(
        "ECA_INTERNAL_LLM_RESPONSE_SCHEMA_MODE",
        "OLLAMA_RESPONSE_SCHEMA_MODE",
        default="full",
        ignore_values={""},
    ).strip().lower()
    if (
        mode in {"minimal", "compact", "psa_minimal", "minimal_psa"}
        and str(packet.get("expected_schema_version", "") or "") == "internal_llm_psa_candidate.v1"
    ):
        return _minimal_psa_candidate_response_schema()
    schema = packet.get("response_schema")
    return schema if isinstance(schema, Mapping) else None


def _minimal_psa_candidate_response_schema() -> dict[str, Any]:
    object_array = {
        "type": "array",
        "minItems": 1,
        "maxItems": 1,
        "items": {
            "type": "object",
            "additionalProperties": True,
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["psa_abstractions"],
        "properties": {
            "psa_abstractions": {
                "type": "array",
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": True,
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
                    ],
                    "properties": {
                        "schema_version": {
                            "type": "string",
                            "enum": ["internal_llm_psa_candidate.v1"],
                        },
                        "abstraction_id": {"type": "string"},
                        "family": {"type": "string", "enum": ["cycle"]},
                        "source_kind": {"type": "string", "enum": ["psa_llm_proposed"]},
                        "source_event_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 4,
                            "items": {"type": "string"},
                        },
                        "state_variables": object_array,
                        "predicate_candidates": object_array,
                        "operator_candidates": object_array,
                        "goal_link_candidates": object_array,
                        "control_sites": object_array,
                        "proposed_plan": object_array,
                        "controlled_objects": {
                            "type": "array",
                            "maxItems": 1,
                            "items": {"type": "object", "additionalProperties": True},
                        },
                        "evidence_frames": {
                            "type": "array",
                            "maxItems": 4,
                            "items": {"type": "integer"},
                        },
                        "effect_confidence": {"type": "number"},
                        "predicate_confidence": {"type": "number"},
                        "contradiction_count": {"type": "integer"},
                        "llm_role": {
                            "type": "string",
                            "enum": ["candidate_generation", "labeling"],
                        },
                        "llm_direct_action": {"type": "boolean"},
                        "metadata": {"type": "object", "additionalProperties": True},
                    },
                },
            }
        },
    }


def _direct_slot_fill_selector_response(
    packet: Mapping[str, Any],
    parsed: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(parsed, Mapping):
        return None
    if _direct_slot_fill_mode() == "weak_seed":
        return _direct_slot_fill_weak_seed_response(packet, parsed)
    if _direct_slot_fill_mode() == "field_copy":
        return _direct_slot_fill_field_copy_response(packet, parsed)
    selector_keys = ("selected_candidate_seed_id", "candidate_seed_id", "selected_id")
    if not any(key in parsed for key in selector_keys):
        return None
    selected = ""
    for key in selector_keys:
        value = parsed.get(key)
        if value is not None:
            selected = str(value).strip()
            break
    user_payload = packet.get("user_payload", {})
    user_payload = user_payload if isinstance(user_payload, Mapping) else {}
    expected_schema = str(packet.get("expected_schema_version", "") or "")
    candidate_seed = _psa_slot_fill_response_from_payload(
        user_payload,
        expected_schema=expected_schema,
    )
    expected_id = _candidate_seed_id(candidate_seed)
    if not selected or selected.lower() in {"none", "null", "reject", "empty"}:
        return {"psa_abstractions": []}
    if selected != expected_id:
        return {"psa_abstractions": []}
    candidate_seed = json.loads(json.dumps(candidate_seed, ensure_ascii=False))
    items = candidate_seed.get("psa_abstractions")
    if isinstance(items, list) and items:
        metadata = items[0].setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["direct_slot_fill_mode"] = _direct_slot_fill_mode()
            metadata["selected_candidate_seed_id"] = selected
            metadata["candidate_seed_materialized_by_provider"] = True
    return candidate_seed


def _direct_slot_fill_weak_seed_response(
    packet: Mapping[str, Any],
    parsed: Mapping[str, Any],
) -> dict[str, Any]:
    expected_schema = str(packet.get("expected_schema_version", "") or "")
    user_payload = packet.get("user_payload", {})
    user_payload = user_payload if isinstance(user_payload, Mapping) else {}
    candidate_seed = _psa_slot_fill_response_from_payload(
        user_payload,
        expected_schema=expected_schema,
    )
    expected = _weak_seed_contract(candidate_seed).get("expected", {})
    expected = expected if isinstance(expected, Mapping) else {}
    for key in ("family", "operator", "target_hint", "effect_hint"):
        if str(parsed.get(key, "") or "").strip().lower() != str(expected.get(key, "") or ""):
            return {"psa_abstractions": []}
    candidate_seed = json.loads(json.dumps(candidate_seed, ensure_ascii=False))
    _annotate_weak_seed_candidate(
        candidate_seed,
        mode="weak_seed",
        fields={key: str(parsed.get(key, "") or "") for key in ("family", "operator", "target_hint", "effect_hint")},
    )
    return candidate_seed


def _direct_slot_fill_line_seed_response(
    packet: Mapping[str, Any],
    content: str,
) -> dict[str, Any] | None:
    expected_schema = str(packet.get("expected_schema_version", "") or "")
    user_payload = packet.get("user_payload", {})
    user_payload = user_payload if isinstance(user_payload, Mapping) else {}
    candidate_seed = _psa_slot_fill_response_from_payload(
        user_payload,
        expected_schema=expected_schema,
    )
    expected = _weak_seed_contract(candidate_seed).get("expected", {})
    expected = expected if isinstance(expected, Mapping) else {}
    parsed = _parse_line_seed_fields(content)
    if not parsed:
        return {"psa_abstractions": []}
    for key in ("family", "operator", "target_hint", "effect_hint"):
        if str(parsed.get(key, "") or "").strip().lower() != str(expected.get(key, "") or ""):
            return {"psa_abstractions": []}
    candidate_seed = json.loads(json.dumps(candidate_seed, ensure_ascii=False))
    _annotate_weak_seed_candidate(candidate_seed, mode="line_seed", fields=parsed)
    return candidate_seed


def _direct_slot_fill_field_copy_response(
    packet: Mapping[str, Any],
    parsed: Mapping[str, Any],
) -> dict[str, Any]:
    expected_schema = str(packet.get("expected_schema_version", "") or "")
    user_payload = packet.get("user_payload", {})
    user_payload = user_payload if isinstance(user_payload, Mapping) else {}
    candidate_seed = _psa_slot_fill_response_from_payload(
        user_payload,
        expected_schema=expected_schema,
    )
    expected = _field_copy_contract(candidate_seed).get("expected", {})
    expected = expected if isinstance(expected, Mapping) else {}
    required = ("source_event_id", "action_name", "x", "y", "pre_state_hash", "post_state_hash")
    for key in required:
        if key not in parsed:
            return {"psa_abstractions": []}
    for key in ("source_event_id", "action_name", "pre_state_hash", "post_state_hash"):
        if str(parsed.get(key, "") or "").strip() != str(expected.get(key, "") or ""):
            return {"psa_abstractions": []}
    if _optional_int(parsed.get("x")) != _optional_int(expected.get("x")):
        return {"psa_abstractions": []}
    if _optional_int(parsed.get("y")) != _optional_int(expected.get("y")):
        return {"psa_abstractions": []}
    candidate_seed = json.loads(json.dumps(candidate_seed, ensure_ascii=False))
    items = candidate_seed.get("psa_abstractions")
    if isinstance(items, list) and items:
        metadata = items[0].setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["direct_slot_fill_mode"] = "field_copy"
            metadata["field_copy_materialized_by_provider"] = True
            metadata["field_copy_source_event_id"] = str(parsed.get("source_event_id", ""))
            metadata["field_copy_action_name"] = str(parsed.get("action_name", ""))
            metadata["field_copy_site"] = {
                "x": _optional_int(parsed.get("x")),
                "y": _optional_int(parsed.get("y")),
            }
    return candidate_seed


def _annotate_weak_seed_candidate(
    candidate_seed: dict[str, Any],
    *,
    mode: str,
    fields: Mapping[str, Any],
) -> None:
    items = candidate_seed.get("psa_abstractions")
    if isinstance(items, list) and items:
        metadata = items[0].setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["direct_slot_fill_mode"] = mode
            metadata["weak_seed_materialized_by_provider"] = True
            metadata["weak_seed_fields"] = {
                "family": str(fields.get("family", "") or ""),
                "operator": str(fields.get("operator", "") or ""),
                "target_hint": str(fields.get("target_hint", "") or ""),
                "effect_hint": str(fields.get("effect_hint", "") or ""),
            }


def _parse_line_seed_fields(content: str) -> dict[str, str]:
    line = str(content or "").strip()
    if not line or line.upper() == "REJECT":
        return {}
    fields: dict[str, str] = {}
    for part in line.replace("\n", " ").split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip().lower()
        if key == "family":
            fields["family"] = value
        elif key == "operator":
            fields["operator"] = value
        elif key == "target_hint":
            fields["target_hint"] = value
        elif key == "effect_hint":
            fields["effect_hint"] = value
    return fields


def _prompt_packet_user_content(packet: Mapping[str, Any]) -> str:
    prompt_schema_mode = _env_first(
        "ECA_INTERNAL_LLM_PROMPT_SCHEMA_MODE",
        "OLLAMA_PROMPT_SCHEMA_MODE",
        default="full",
        ignore_values={""},
    ).strip().lower()
    prompt_payload_mode = _env_first(
        "ECA_INTERNAL_LLM_PROMPT_PAYLOAD_MODE",
        "OLLAMA_PROMPT_PAYLOAD_MODE",
        default="full",
        ignore_values={""},
    ).strip().lower()
    raw_user_payload = packet.get("user_payload", {})
    if not isinstance(raw_user_payload, Mapping):
        raw_user_payload = {}
    user_payload = (
        _compact_prompt_user_payload(
            raw_user_payload,
            expected_schema_version=str(packet.get("expected_schema_version", "") or ""),
        )
        if prompt_payload_mode in {"compact", "minimal", "evidence_only"}
        else dict(raw_user_payload)
    )
    instruction = (
        "Return exactly one JSON object matching response_schema. "
        "Do not include markdown or natural-language plans."
    )
    payload = {
        "role": packet.get("role", ""),
        "expected_schema_version": packet.get("expected_schema_version", ""),
        "user_payload": user_payload,
        "forbidden_fields": packet.get("forbidden_fields", []),
        "instruction": instruction,
    }
    if prompt_schema_mode in {"template", "minimal", "no_schema"}:
        payload["instruction"] = (
            "Return exactly one JSON object. The only allowed top-level key is "
            "psa_abstractions. If user_payload.visible_effect_summary."
            "effective_control_sites is empty, return {\"psa_abstractions\": []}. "
            "Otherwise fill one candidate by copying x, y, action_name, event_id, "
            "pre_state_hash, and post_state_hash from user_payload. Do not copy, "
            "describe, or emit a JSON Schema. Do not return top-level keys like "
            "type, properties, required, name, description, concepts, analogy, or "
            "example_usage."
        )
        if prompt_payload_mode not in {"compact", "minimal", "evidence_only"}:
            payload["response_template"] = user_payload.get(
                "minimum_valid_response_template",
                {},
            )
    elif prompt_schema_mode in {"none", "off", "disabled"}:
        payload["instruction"] = (
            "Return exactly one JSON object. Use the minimum_valid_response_template "
            "inside user_payload as the shape example. Do not copy or describe a JSON "
            "Schema and do not include markdown or natural-language plans."
        )
    else:
        payload["response_schema"] = packet.get("response_schema", {})
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    max_chars = _env_int("ECA_INTERNAL_LLM_MAX_PROMPT_CHARS", default=24000)
    if max_chars is None or len(content) <= max_chars:
        return content
    keep = max(1000, int(max_chars) // 2)
    return content[:keep] + "\n...<prompt_packet_truncated>...\n" + content[-keep:]


def _compact_prompt_user_payload(
    user_payload: Mapping[str, Any],
    *,
    expected_schema_version: str,
) -> dict[str, Any]:
    visible = user_payload.get("visible_effect_grammar_summary")
    visible = visible if isinstance(visible, Mapping) else {}
    records = user_payload.get("recent_action_effect_records")
    records = records if isinstance(records, list) else []
    effective_sites = visible.get("effective_control_sites")
    effective_sites = effective_sites if isinstance(effective_sites, list) else []
    action_summaries = visible.get("action_summaries")
    action_summaries = action_summaries if isinstance(action_summaries, list) else []

    return {
        "frame_shape": user_payload.get("frame_shape", []),
        "frame_index": user_payload.get("frame_index"),
        "current_state_hash": user_payload.get("current_state_hash", ""),
        "available_actions": user_payload.get("available_actions", []),
        "required_source_kind": user_payload.get("required_source_kind", ""),
        "required_family": user_payload.get("required_family", ""),
        "allowed_llm_roles": user_payload.get("allowed_llm_roles", []),
        "valid_source_event_ids": user_payload.get("valid_source_event_ids", []),
        "recent_action_effect_records": [
            _compact_prompt_record(record) for record in records[-4:] if isinstance(record, Mapping)
        ],
        "visible_effect_summary": {
            "record_count": int(visible.get("record_count", 0) or 0),
            "effective_site_count": int(visible.get("effective_site_count", 0) or 0),
            "localized_effect_site_count": int(visible.get("localized_effect_site_count", 0) or 0),
            "repeated_effect_site_count": int(visible.get("repeated_effect_site_count", 0) or 0),
            "score_event_count": int(visible.get("score_event_count", 0) or 0),
            "action_summaries": action_summaries[:4],
            "effective_control_sites": [
                _compact_effective_site(site)
                for site in _prioritized_effective_sites(effective_sites)[:4]
                if isinstance(site, Mapping)
            ],
        },
        "response_cardinality_contract": user_payload.get("response_cardinality_contract", {}),
        "empty_when_no_supporting_effect_record": bool(
            user_payload.get("empty_when_no_supporting_effect_record", True)
        ),
        "answer_must_not_be_json_schema": True,
        "allowed_top_level_keys": ["psa_abstractions"],
        "forbidden_top_level_keys": [
            "type",
            "properties",
            "required",
            "name",
            "description",
            "concepts",
            "analogy",
            "example_usage",
        ],
        "field_fill_instructions": {
            "source_event_ids": "copy the event_id matching effective_control_sites[0]",
            "action_name": "copy effective_control_sites[0].action_name",
            "x": "copy effective_control_sites[0].x",
            "y": "copy effective_control_sites[0].y",
            "pre_state_hash": "copy effective_control_sites[0].latest_pre_state_hash",
            "post_state_hash": "copy effective_control_sites[0].latest_post_state_hash",
        },
        "minimum_valid_response_template": _minimal_psa_response_template(expected_schema_version)
        if expected_schema_version == "internal_llm_psa_candidate.v1"
        else user_payload.get("minimum_valid_response_template", {}),
    }


def _prioritized_effective_sites(sites: list[Any]) -> list[Mapping[str, Any]]:
    indexed = [(index, dict(site)) for index, site in enumerate(sites) if isinstance(site, Mapping)]

    def rank(item: tuple[int, Mapping[str, Any]]) -> tuple[int, int, int]:
        index, site = item
        action = str(site.get("action_name", "") or "")
        has_xy = _optional_int(site.get("x")) is not None and _optional_int(site.get("y")) is not None
        is_action6 = action == "ACTION6"
        return (0 if is_action6 and has_xy else 1 if has_xy else 2, index, 0)

    return [site for _index, site in sorted(indexed, key=rank)]


def _compact_prompt_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": record.get("event_id", ""),
        "step_index": record.get("step_index", 0),
        "action_name": record.get("action_name", ""),
        "action_payload": record.get("action_payload", {}),
        "pre_state_hash": record.get("pre_state_hash", ""),
        "post_state_hash": record.get("post_state_hash", ""),
        "score_delta": record.get("score_delta", 0),
        "changed_pixel_count": record.get("changed_pixel_count", 0),
        "changed_bbox": record.get("changed_bbox", {}),
        "no_op": bool(record.get("no_op", False)),
    }


def _compact_effective_site(site: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action_name": site.get("action_name", ""),
        "x": site.get("x"),
        "y": site.get("y"),
        "control_site": site.get("control_site", {}),
        "effective_count": site.get("effective_count", 0),
        "changed_pixel_total": site.get("changed_pixel_total", 0),
        "latest_step_index": site.get("latest_step_index", 0),
        "latest_pre_state_hash": site.get("latest_pre_state_hash", ""),
        "latest_post_state_hash": site.get("latest_post_state_hash", ""),
        "latest_changed_bbox": site.get("latest_changed_bbox", {}),
    }


def _minimal_psa_response_template(expected_schema_version: str) -> dict[str, Any]:
    schema_version = expected_schema_version or "internal_llm_psa_candidate.v1"
    return {
        "psa_abstractions": [
            {
                "schema_version": schema_version,
                "abstraction_id": "psa_llm:cycle:ACTION6:x,y",
                "family": "cycle",
                "source_kind": "psa_llm_proposed",
                "source_event_ids": ["COPY_VALID_SOURCE_EVENT_ID"],
                "state_variables": [
                    {
                        "variable_id": "psa_llm:visible_state_hash:ACTION6:x,y",
                        "variable_type": "visible_state_hash_transition",
                        "pre_state_hash": "COPY_RECORD_PRE_STATE_HASH",
                        "post_state_hash": "COPY_RECORD_POST_STATE_HASH",
                        "control_site": {"action_name": "ACTION6", "x": 0, "y": 0},
                    }
                ],
                "predicate_candidates": [
                    {
                        "predicate_id": "psa_llm:predicate:visible_state_advance",
                        "predicate_kind": "visible_state_advance",
                        "post_state_hash": "COPY_RECORD_POST_STATE_HASH",
                    }
                ],
                "operator_candidates": [
                    {
                        "operator_id": "psa_llm:operator:repeat_effective_click:ACTION6:x,y",
                        "operator_family": "click_operator",
                        "operation_kind": "repeat_effective_click",
                        "action_name": "ACTION6",
                        "payload": {"x": 0, "y": 0},
                    }
                ],
                "goal_link_candidates": [
                    {
                        "goal_link_id": "psa_llm:goal_link:visible_state_advance",
                        "goal_link_type": "visible_state_advance",
                        "predicted_goal_delta": 0.0,
                    }
                ],
                "control_sites": [{"action_name": "ACTION6", "x": 0, "y": 0}],
                "proposed_plan": [{"action_name": "ACTION6", "payload": {"x": 0, "y": 0}}],
                "effect_confidence": 0.7,
                "predicate_confidence": 0.7,
                "llm_role": "candidate_generation",
                "llm_direct_action": False,
            }
        ]
    }


def _should_run_contract_repair_retry(
    packet: Mapping[str, Any],
    parsed: Mapping[str, Any],
) -> bool:
    if not _env_bool(
        "ECA_INTERNAL_LLM_CONTRACT_REPAIR_RETRY",
        fallback_name="OLLAMA_CONTRACT_REPAIR_RETRY",
        default=False,
    ):
        return False
    if not isinstance(parsed, Mapping):
        return False
    return not _contract_response_matches_expected(packet, parsed)


def _contract_response_matches_expected(
    packet: Mapping[str, Any],
    parsed: Mapping[str, Any],
) -> bool:
    expected_schema = str(packet.get("expected_schema_version", "") or "")
    if expected_schema == "internal_llm_psa_candidate.v1":
        items = parsed.get("psa_abstractions")
        if not isinstance(items, list):
            return False
        return all(
            isinstance(item, Mapping)
            and str(item.get("schema_version", "") or "") == expected_schema
            for item in items
        )
    if expected_schema == "internal_llm_mechanic_hypothesis.v1":
        items = parsed.get("mechanic_hypotheses")
        if not isinstance(items, list):
            return False
        return all(
            isinstance(item, Mapping)
            and str(item.get("schema_version", "") or "") == expected_schema
            for item in items
        )
    return True


def _contract_repair_empty_on_failure() -> bool:
    return _env_bool(
        "ECA_INTERNAL_LLM_CONTRACT_REPAIR_EMPTY_ON_FAILURE",
        fallback_name="OLLAMA_CONTRACT_REPAIR_EMPTY_ON_FAILURE",
        default=True,
    )


def _contract_repair_mode() -> str:
    mode = _env_first(
        "ECA_INTERNAL_LLM_CONTRACT_REPAIR_MODE",
        "OLLAMA_CONTRACT_REPAIR_MODE",
        default="envelope",
        ignore_values={""},
    ).strip().lower()
    if mode in {"slot_fill", "field_fill", "candidate_fill"}:
        return "slot_fill"
    return "envelope"


def _empty_contract_response(packet: Mapping[str, Any]) -> dict[str, Any]:
    expected_schema = str(packet.get("expected_schema_version", "") or "")
    if expected_schema == "internal_llm_mechanic_hypothesis.v1":
        return {"mechanic_hypotheses": []}
    return {"psa_abstractions": []}


def _psa_slot_fill_response_from_payload(
    user_payload: Mapping[str, Any],
    *,
    expected_schema: str,
) -> dict[str, Any]:
    compact = _compact_prompt_user_payload(
        user_payload,
        expected_schema_version=expected_schema,
    )
    visible = compact.get("visible_effect_summary")
    visible = visible if isinstance(visible, Mapping) else {}
    sites = visible.get("effective_control_sites")
    sites = sites if isinstance(sites, list) else []
    records = compact.get("recent_action_effect_records")
    records = records if isinstance(records, list) else []
    if not sites or not records:
        return {"psa_abstractions": []}

    site = next(
        (
            dict(item)
            for item in sites
            if isinstance(item, Mapping)
            and str(item.get("action_name", "") or "") == "ACTION6"
            and _optional_int(item.get("x")) is not None
            and _optional_int(item.get("y")) is not None
        ),
        {},
    )
    if not site:
        site = next(
            (
                dict(item)
                for item in sites
                if isinstance(item, Mapping)
                and _optional_int(item.get("x")) is not None
                and _optional_int(item.get("y")) is not None
            ),
            {},
        )
    if not site:
        site = next((dict(item) for item in sites if isinstance(item, Mapping)), {})
    action_name = str(site.get("action_name", "") or "")
    x = _optional_int(site.get("x"))
    y = _optional_int(site.get("y"))
    site_source = "control_site"
    if (x is None or y is None) and _field_copy_allow_bbox_anchor_site():
        anchor = _bbox_anchor_from_mapping(site.get("latest_changed_bbox"))
        if anchor is not None:
            x, y = anchor
            site_source = "latest_changed_bbox_anchor"
    if not action_name or x is None or y is None:
        return {"psa_abstractions": []}

    record = _matching_compact_record(records, action_name=action_name, x=x, y=y)
    if not record:
        return {"psa_abstractions": []}
    if site_source != "control_site":
        record_anchor = _bbox_anchor_from_mapping(record.get("changed_bbox"))
        if record_anchor is not None:
            x, y = record_anchor
    event_id = str(record.get("event_id", "") or "")
    pre_state_hash = str(
        site.get("latest_pre_state_hash", "")
        or record.get("pre_state_hash", "")
        or ""
    )
    post_state_hash = str(
        site.get("latest_post_state_hash", "")
        or record.get("post_state_hash", "")
        or ""
    )
    if not event_id or not pre_state_hash or not post_state_hash:
        return {"psa_abstractions": []}

    site_payload = {"action_name": action_name, "x": int(x), "y": int(y)}
    payload = {"x": int(x), "y": int(y)}
    suffix = f"{action_name}:{int(x)},{int(y)}"
    return {
        "psa_abstractions": [
            {
                "schema_version": expected_schema or "internal_llm_psa_candidate.v1",
                "abstraction_id": f"psa_llm:slot_fill:cycle:{suffix}",
                "family": "cycle",
                "source_kind": "psa_llm_proposed",
                "source_event_ids": [event_id],
                "state_variables": [
                    {
                        "variable_id": f"psa_llm:slot_fill_state:{suffix}",
                        "variable_type": "visible_state_hash_transition",
                        "pre_state_hash": pre_state_hash,
                        "post_state_hash": post_state_hash,
                        "control_site": dict(site_payload),
                        "changed_bbox": record.get("changed_bbox", {}),
                        "confidence": 0.7,
                    }
                ],
                "predicate_candidates": [
                    {
                        "predicate_id": f"psa_llm:slot_fill_predicate:visible_state_advance:{suffix}",
                        "predicate_kind": "visible_state_advance",
                        "post_state_hash": post_state_hash,
                        "confidence": 0.65,
                    }
                ],
                "operator_candidates": [
                    {
                        "operator_id": f"psa_llm:slot_fill_operator:repeat_effective_click:{suffix}",
                        "operator_family": "click_operator",
                        "operation_kind": "repeat_effective_click",
                        "action_name": action_name,
                        "payload": dict(payload),
                        "support_count": int(site.get("effective_count", 1) or 1),
                    }
                ],
                "goal_link_candidates": [
                    {
                        "goal_link_id": f"psa_llm:slot_fill_goal_link:visible_state_advance:{suffix}",
                        "goal_link_type": "visible_state_advance",
                        "predicted_goal_delta": 0.0,
                        "confidence": 0.6,
                    }
                ],
                "control_sites": [dict(site_payload)],
                "controlled_objects": [
                    {
                        "kind": "changed_region",
                        "changed_bbox": record.get("changed_bbox", {}),
                    }
                ],
                "proposed_plan": [{"action_name": action_name, "payload": dict(payload)}],
                "evidence_frames": [int(record.get("step_index", 0) or 0)],
                "effect_confidence": 0.7,
                "predicate_confidence": 0.65,
                "llm_role": "candidate_generation",
                "llm_direct_action": False,
                "metadata": {
                    "contract_repair_mode": "slot_fill",
                    "contract_repair_candidate_seed": True,
                    "field_copy_site_source": site_source,
                },
            }
        ]
    }


def _field_copy_allow_bbox_anchor_site() -> bool:
    raw = _env_first(
        "ECA_INTERNAL_LLM_FIELD_COPY_ALLOW_BBOX_ANCHOR_SITE",
        "OLLAMA_FIELD_COPY_ALLOW_BBOX_ANCHOR_SITE",
        default="",
        ignore_values=set(),
    ).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _bbox_anchor_from_mapping(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        min_x = int(value.get("min_x"))
        min_y = int(value.get("min_y"))
        max_x = int(value.get("max_x", min_x))
        max_y = int(value.get("max_y", min_y))
    except (TypeError, ValueError):
        return None
    return int(round((min_x + max_x) / 2.0)), int(round((min_y + max_y) / 2.0))


def _candidate_seed_id(candidate_seed: Mapping[str, Any]) -> str:
    items = candidate_seed.get("psa_abstractions")
    if not isinstance(items, list) or not items:
        return ""
    return "candidate_0"


def _field_copy_contract(candidate_seed: Mapping[str, Any]) -> dict[str, Any]:
    items = candidate_seed.get("psa_abstractions")
    items = items if isinstance(items, list) else []
    if not items or not isinstance(items[0], Mapping):
        return {
            "has_candidate": False,
            "expected": {
                "source_event_id": "",
                "action_name": "",
                "x": 0,
                "y": 0,
                "pre_state_hash": "",
                "post_state_hash": "",
            },
        }
    candidate = items[0]
    source_event_ids = candidate.get("source_event_ids")
    source_event_ids = source_event_ids if isinstance(source_event_ids, list) else []
    control_sites = candidate.get("control_sites")
    control_sites = control_sites if isinstance(control_sites, list) else []
    state_variables = candidate.get("state_variables")
    state_variables = state_variables if isinstance(state_variables, list) else []
    site = control_sites[0] if control_sites and isinstance(control_sites[0], Mapping) else {}
    state = state_variables[0] if state_variables and isinstance(state_variables[0], Mapping) else {}
    return {
        "has_candidate": True,
        "expected": {
            "source_event_id": str(source_event_ids[0]) if source_event_ids else "",
            "action_name": str(site.get("action_name", "") or ""),
            "x": int(_optional_int(site.get("x")) or 0),
            "y": int(_optional_int(site.get("y")) or 0),
            "pre_state_hash": str(state.get("pre_state_hash", "") or ""),
            "post_state_hash": str(state.get("post_state_hash", "") or ""),
        },
        "seed_hash": _canonical_hash(candidate_seed)[:16],
    }


def _weak_seed_contract(candidate_seed: Mapping[str, Any]) -> dict[str, Any]:
    has_candidate = bool(_candidate_seed_id(candidate_seed))
    expected = {
        "family": "cycle" if has_candidate else "",
        "operator": "click" if has_candidate else "",
        "target_hint": "button" if has_candidate else "",
        "effect_hint": "state_change" if has_candidate else "",
    }
    return {
        "has_candidate": has_candidate,
        "expected": expected,
        "seed_hash": _canonical_hash(candidate_seed)[:16] if has_candidate else "",
    }


def _candidate_seed_summary(candidate_seed: Mapping[str, Any]) -> dict[str, Any]:
    items = candidate_seed.get("psa_abstractions")
    items = items if isinstance(items, list) else []
    if not items or not isinstance(items[0], Mapping):
        return {
            "has_candidate": False,
            "candidate_seed_id": "",
            "seed_hash": "",
            "family": "",
            "source_event_ids": [],
            "control_site": {},
            "operator": {},
        }
    candidate = items[0]
    control_sites = candidate.get("control_sites")
    control_sites = control_sites if isinstance(control_sites, list) else []
    operators = candidate.get("operator_candidates")
    operators = operators if isinstance(operators, list) else []
    return {
        "has_candidate": True,
        "candidate_seed_id": _candidate_seed_id(candidate_seed),
        "seed_hash": _canonical_hash(candidate_seed)[:16],
        "schema_version": candidate.get("schema_version", ""),
        "family": candidate.get("family", ""),
        "source_kind": candidate.get("source_kind", ""),
        "source_event_ids": candidate.get("source_event_ids", []),
        "control_site": control_sites[0] if control_sites and isinstance(control_sites[0], Mapping) else {},
        "operator": operators[0] if operators and isinstance(operators[0], Mapping) else {},
        "llm_role": candidate.get("llm_role", ""),
    }


def _canonical_hash(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = str(value)
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _matching_compact_record(
    records: list[Any],
    *,
    action_name: str,
    x: int,
    y: int,
) -> dict[str, Any]:
    first_valid: dict[str, Any] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if not first_valid:
            first_valid = dict(record)
        payload = record.get("action_payload")
        payload = payload if isinstance(payload, Mapping) else {}
        if (
            str(record.get("action_name", "") or "") == action_name
            and _optional_int(payload.get("x")) == int(x)
            and _optional_int(payload.get("y")) == int(y)
        ):
            return dict(record)
        if (
            _field_copy_allow_bbox_anchor_site()
            and str(record.get("action_name", "") or "") == action_name
            and _optional_int(payload.get("x")) is None
            and _optional_int(payload.get("y")) is None
        ):
            return dict(record)
    return first_valid


def _ollama_contract_repair_body(
    base_body: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    raw_response: Mapping[str, Any],
    use_native_chat: bool,
) -> dict[str, Any]:
    body = dict(base_body)
    expected_schema = str(packet.get("expected_schema_version", "") or "")
    user_payload = packet.get("user_payload", {})
    user_payload = user_payload if isinstance(user_payload, Mapping) else {}
    compact_payload = _contract_repair_payload(user_payload, expected_schema=expected_schema)
    repair_mode = _contract_repair_mode()
    target_key = (
        "mechanic_hypotheses"
        if expected_schema == "internal_llm_mechanic_hypothesis.v1"
        else "psa_abstractions"
    )
    candidate_seed = (
        _psa_slot_fill_response_from_payload(user_payload, expected_schema=expected_schema)
        if expected_schema == "internal_llm_psa_candidate.v1"
        and repair_mode in {"slot_fill", "field_fill", "candidate_fill"}
        else None
    )
    repair_packet = {
        "task": "contract_repair_slot_fill" if candidate_seed is not None else "contract_repair_only",
        "repair_mode": repair_mode,
        "expected_schema_version": expected_schema,
        "allowed_top_level_keys": [target_key],
        "raw_response": _truncate_for_prompt(raw_response, max_chars=1400),
        "evidence_payload": compact_payload,
        "candidate_seed": candidate_seed,
        "instruction": (
            (
                "Return exactly candidate_seed as one JSON object with no extra keys. "
                "If candidate_seed has an empty array, return it exactly. Do not "
                "rewrite, summarize, explain, or emit a JSON Schema. Do not include "
                "markdown or prose."
            )
            if candidate_seed is not None
            else (
                "Return exactly one JSON object with only the allowed top-level key. "
                "If raw_response is a JSON Schema, story, explanation, title, or lacks "
                "grounded candidate fields, return an empty array for the allowed key. "
                "Do not invent source_event_ids, actions, coordinates, hashes, or game "
                "semantics. Do not include markdown or prose."
            )
        ),
    }
    repair_messages = [
        {
            "role": "system",
            "content": (
                "You repair an internal ECA/PSA LLM response into the required "
                "JSON envelope. You do not solve the game. /no_think"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(repair_packet, ensure_ascii=False, sort_keys=True),
        },
    ]
    body["messages"] = repair_messages
    repair_max_tokens = _env_int(
        "ECA_INTERNAL_LLM_CONTRACT_REPAIR_MAX_TOKENS",
        fallback_name="OLLAMA_CONTRACT_REPAIR_MAX_TOKENS",
        default=512,
    )
    if use_native_chat:
        options = dict(body.get("options", {}) or {})
        options["temperature"] = min(float(options.get("temperature", 0.1) or 0.1), 0.1)
        if repair_max_tokens is not None:
            options["num_predict"] = max(1, int(repair_max_tokens))
        body["options"] = options
        body["format"] = "json"
    else:
        body["temperature"] = min(float(body.get("temperature", 0.1) or 0.1), 0.1)
        if repair_max_tokens is not None:
            body["max_tokens"] = max(1, int(repair_max_tokens))
        body["response_format"] = {"type": "json_object"}
    return body


def _contract_repair_payload(
    user_payload: Mapping[str, Any],
    *,
    expected_schema: str,
) -> dict[str, Any]:
    if expected_schema == "internal_llm_psa_candidate.v1":
        return _compact_prompt_user_payload(
            user_payload,
            expected_schema_version=expected_schema,
        )
    return {
        "frame_shape": user_payload.get("frame_shape", []),
        "frame_index": user_payload.get("frame_index"),
        "current_state_hash": user_payload.get("current_state_hash", ""),
        "available_actions": user_payload.get("available_actions", []),
        "recent_action_effect_records": user_payload.get("recent_action_effect_records", [])[:4]
        if isinstance(user_payload.get("recent_action_effect_records"), list)
        else [],
        "minimum_valid_response_template": user_payload.get("minimum_valid_response_template", {}),
    }


def _truncate_for_prompt(value: Any, *, max_chars: int) -> Any:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    if len(text) <= int(max_chars):
        return value
    keep = max(100, int(max_chars) // 2)
    return {
        "truncated": True,
        "char_count": len(text),
        "prefix": text[:keep],
        "suffix": text[-keep:],
    }


def _message_chars(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    return sum(
        len(str(message.get("content", "") or ""))
        for message in messages
        if isinstance(message, Mapping)
    )


def _sum_optional_ints(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return int(a) + int(b)


def _ollama_attempt_bodies(
    body: Mapping[str, Any],
    *,
    use_native_chat: bool,
    json_response_format: bool,
) -> list[dict[str, Any]]:
    primary = dict(body)
    attempts = [primary]
    if not json_response_format or not _env_bool(
        "ECA_INTERNAL_LLM_OLLAMA_FORMAT_FALLBACK",
        default=False,
    ):
        return attempts

    if use_native_chat:
        simple = dict(primary)
        if simple.get("format") != "json":
            simple["format"] = "json"
            attempts.append(simple)
        plain = dict(primary)
        if "format" in plain:
            plain.pop("format", None)
            attempts.append(plain)
    else:
        plain = dict(primary)
        if "response_format" in plain:
            plain.pop("response_format", None)
            attempts.append(plain)

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for attempt in attempts:
        key = json.dumps(attempt, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(attempt)
    return unique


def _should_retry_ollama_attempt(exc: Exception) -> bool:
    names = " ".join(type(item).__name__ for item in type(exc).mro())
    text = f"{names} {exc}".lower()
    if "timeout" in text or "timed out" in text:
        return _env_bool("ECA_INTERNAL_LLM_OLLAMA_RETRY_ON_TIMEOUT", default=False)
    return True


def _raise_for_status_with_ollama_diagnostics(
    response: Any,
    *,
    api_url: str,
    request_body: Mapping[str, Any],
) -> None:
    try:
        response.raise_for_status()
    except Exception as exc:
        status_code = getattr(response, "status_code", None)
        response_body = _response_error_preview(response)
        request_summary = _ollama_request_error_summary(request_body)
        raise RuntimeError(
            "Ollama request failed"
            f" status_code={status_code!r}"
            f" url={api_url!r}"
            f" response_body={response_body!r}"
            f" request_summary={request_summary!r}"
        ) from exc


def _response_error_preview(response: Any, *, max_chars: int = 500) -> str:
    text = getattr(response, "text", None)
    if text is None:
        try:
            data = response.json()
        except Exception:
            return ""
        if isinstance(data, Mapping):
            for key in ("error", "message", "detail"):
                value = data.get(key)
                if value:
                    text = str(value)
                    break
        if text is None:
            text = str(data)
    return str(text).replace("\n", "\\n")[:max_chars]


def _ollama_request_error_summary(request_body: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("model", "format", "keep_alive", "stream", "think"):
        if key in request_body:
            summary[key] = request_body.get(key)
    options = request_body.get("options")
    if isinstance(options, Mapping):
        summary["options"] = {
            str(key): value
            for key, value in options.items()
            if key
            in {
                "temperature",
                "num_predict",
                "num_ctx",
                "num_gpu",
                "num_thread",
                "num_batch",
                "seed",
                "top_k",
                "top_p",
                "min_p",
                "repeat_penalty",
            }
        }
    return summary


def _parse_json_object_from_content(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return _ensure_json_object(parse_internal_llm_json_response(text))
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            preview = text[:300].replace("\n", "\\n")
            raise ValueError(f"Gemma E4B response did not contain a JSON object; preview={preview!r}")
        return _ensure_json_object(parse_internal_llm_json_response(text[start : end + 1]))


def _ensure_json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Gemma E4B response JSON is not an object: {type(value).__name__}")
    return dict(value)


def _extract_chat_content(data: Any) -> str:
    if not isinstance(data, Mapping):
        raise ValueError(f"Ollama response is not a mapping: {type(data).__name__}")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Ollama response does not include choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("Ollama response choice is not a mapping")
    message = first.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
    else:
        content = first.get("text")
    if content is None:
        raise ValueError("Ollama response does not include message content")
    return str(content)


def _extract_native_chat_content(data: Any) -> str:
    if not isinstance(data, Mapping):
        raise ValueError(f"Ollama native response is not a mapping: {type(data).__name__}")
    message = data.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if content is not None:
            return str(content)
    content = data.get("response")
    if content is not None:
        return str(content)
    raise ValueError("Ollama native response does not include message content")


def _internal_llm_model() -> str:
    return _env_first(
        "ECA_INTERNAL_LLM_OLLAMA_MODEL",
        "ECA_INTERNAL_LLM_MODEL",
        "V2_LLM_MODEL",
        default=DEFAULT_GEMMA_E4B_MODEL,
        ignore_values={"", "none", "disabled"},
    )


def _workdir_from_kwargs(kwargs: Mapping[str, Any]) -> Path:
    raw = (
        kwargs.get("workdir")
        or os.environ.get("ECA_INTERNAL_LLM_WORKDIR")
        or os.environ.get("V2_RUN_WORKDIR")
        or "."
    )
    return Path(str(raw))


def _env_first(
    *names: str,
    default: str,
    ignore_values: set[str] | None = None,
) -> str:
    ignored = {item.lower() for item in (ignore_values or set())}
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized.lower() in ignored:
            continue
        return normalized
    return default


def _env_bool(
    name: str,
    *,
    fallback_name: str | None = None,
    default: bool,
) -> bool:
    raw = os.environ.get(name)
    if raw is None and fallback_name:
        raw = os.environ.get(fallback_name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(
    name: str,
    *,
    fallback_name: str | None = None,
    default: int | None,
) -> int | None:
    raw = os.environ.get(name)
    if raw is None and fallback_name:
        raw = os.environ.get(fallback_name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _env_float_optional(name: str, *, fallback_name: str | None = None) -> float | None:
    raw = os.environ.get(name)
    if raw is None and fallback_name:
        raw = os.environ.get(fallback_name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


psa_provider = psa_gemma_e4b_provider
eca_provider = eca_gemma_e4b_provider


__all__ = [
    "DEFAULT_GEMMA_E4B_MODEL",
    "ECA_GEMMA_E4B_CALL_SITE",
    "PSA_GEMMA_E4B_CALL_SITE",
    "eca_gemma_e4b_provider",
    "eca_provider",
    "psa_gemma_e4b_provider",
    "psa_provider",
]
