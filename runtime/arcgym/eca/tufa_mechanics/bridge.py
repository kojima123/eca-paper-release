"""Normalize independent mechanic modules into the Tufa toolbox contract."""
from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from .base import (
    MechanicContext,
    coerce_mechanic_context,
    json_safe_mapping,
    normalize_plan,
    stable_digest,
)
from .registry import load_runtime_mechanic, runtime_mechanic_spec


_ENGINE_TO_MODEL = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "ACTION6": "MOUSE",
}


def _model_actions(value: Any) -> list[dict[str, Any]]:
    normalized = normalize_plan({"actions": value})
    if normalized is None:
        return []
    actions: list[dict[str, Any]] = []
    for action in normalized["actions"]:
        item = copy.deepcopy(action)
        item["action"] = _ENGINE_TO_MODEL.get(item["action"], item["action"])
        if item["action"] == "MOUSE" and "row" not in item:
            if "y" in item and "x" in item:
                item["row"] = int(item.pop("y"))
                item["col"] = int(item.pop("x"))
            else:
                return []
        actions.append(item)
    return actions


def _binding_entities(binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = binding.get("role_bindings")
    if not isinstance(candidates, list):
        candidates = binding.get("roles")
    entities: list[dict[str, Any]] = []
    for item in candidates or []:
        if isinstance(item, Mapping):
            row = json_safe_mapping(item)
            if row is not None and row.get("role"):
                entities.append(row)
    return entities[:16]


def _call_with_context(
    function: Any,
    *args: Any,
    context: MechanicContext,
    **kwargs: Any,
) -> Any:
    """Pass context only to modules that declare the optional parameter."""

    try:
        parameters = inspect.signature(function).parameters.values()
        accepts_context = any(
            item.name == "context" or item.kind == inspect.Parameter.VAR_KEYWORD
            for item in parameters
        )
    except (TypeError, ValueError):
        accepts_context = False
    if accepts_context:
        kwargs["context"] = context
    return function(*args, **kwargs)


def compute_runtime_artifact(
    toolbox_family: str,
    frame: Any,
    *,
    level: int,
    valid_actions: set[str] | None = None,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    spec = runtime_mechanic_spec(toolbox_family)
    module = load_runtime_mechanic(toolbox_family)
    if spec is None or module is None:
        return None
    mechanic_context = coerce_mechanic_context(
        frame,
        level=int(level),
        context=context,
    )
    try:
        schema = json_safe_mapping(
            _call_with_context(
                module.compile_schema,
                frame,
                level=int(level),
                context=mechanic_context,
            )
        )
        binding = json_safe_mapping(
            _call_with_context(
                module.bind_current,
                frame,
                context=mechanic_context,
            )
        )
    except Exception:
        return None
    if schema is None or binding is None:
        return None

    try:
        proposed = _call_with_context(
            module.propose_plan,
            frame,
            valid_actions=sorted(valid_actions or ()),
            context=mechanic_context,
        )
    except Exception:
        proposed = None
    if isinstance(proposed, Mapping):
        plan_candidate: Any = proposed
    elif isinstance(proposed, Sequence) and not isinstance(
        proposed, (str, bytes, bytearray)
    ):
        plan_candidate = {"actions": proposed}
    else:
        plan_candidate = None
    plan = normalize_plan(plan_candidate)
    engine_actions = copy.deepcopy((plan or {}).get("actions") or [])
    actions = _model_actions(engine_actions)
    prediction = json_safe_mapping((plan or {}).get("first_prediction"))
    if prediction is None and engine_actions:
        try:
            prediction = json_safe_mapping(
                _call_with_context(
                    module.predict_transition,
                    frame,
                    copy.deepcopy(engine_actions[0]),
                    context=mechanic_context,
                )
            )
        except Exception:
            try:
                prediction = json_safe_mapping(
                    _call_with_context(
                        module.predict_transition,
                        frame,
                        copy.deepcopy(actions[0]),
                        context=mechanic_context,
                    )
                )
            except Exception:
                prediction = None

    binding_status = str(binding.get("status", "bound") or "bound")
    binding_accepted = binding.get("accepted")
    binding_supported = binding.get("supported")
    schema_supported = schema.get("supported")
    required_bindings = schema.get("required_bindings") or schema.get("roles")
    supported = bool(
        schema.get("roles")
        and required_bindings
        and schema_supported is not False
        and binding_supported is not False
        and binding_status not in {"unbound", "absent", "insufficient_rewrite"}
        and binding_accepted is not False
    )
    prediction_accepted = prediction is not None and prediction.get("accepted") is not False
    certified = bool(
        actions
        and prediction_accepted
        and supported
        and (plan or {}).get("certified", False)
        and spec.rollout == "repair"
    )
    runtime_row = {
        "toolbox_family": spec.toolbox_family,
        "mechanic_family": spec.mechanic_family,
        "rollout": spec.rollout,
        "supported": supported,
        "certified": certified,
        "schema": schema,
        "current_binding": binding,
        "context": mechanic_context.public_summary(),
    }

    entry = None
    if actions:
        expected_change: dict[str, Any] = {
            "first_action": copy.deepcopy(actions[0]),
            "rule": str(
                (plan or {}).get("transition_rule")
                or schema.get("operator", {}).get("transition")
                or schema.get("operator", {}).get("effect")
                or spec.mechanic_family
            ),
        }
        if prediction is not None:
            expected_change["runtime_mechanic_exact"] = {
                "verifier": "runtime_mechanic_transition_v1",
                "toolbox_family": spec.toolbox_family,
                "mechanic_family": spec.mechanic_family,
                "prediction": prediction,
            }
        verification_state = {
            "family": spec.toolbox_family,
            "mechanic_family": spec.mechanic_family,
            "binding_signature": stable_digest(binding),
            "plan_hash": stable_digest(actions),
            "plan_len": len(actions),
        }
        grounding = json_safe_mapping((plan or {}).get("grounded_observation")) or {}
        grounding.setdefault("entities", _binding_entities(binding))
        grounding.setdefault("invariants", list(schema.get("invariants") or []))
        grounding.setdefault("falsifiers", list(schema.get("falsifiers") or []))
        progress_measure = json_safe_mapping(
            (plan or {}).get("progress_measure")
        )
        if progress_measure is not None:
            grounding["progress_measure"] = progress_measure
            expected_change["progress_measure"] = copy.deepcopy(progress_measure)
        grounding["expected_change"] = expected_change
        grounding["verification_state"] = verification_state
        detail = json_safe_mapping((plan or {}).get("detail")) or {}
        try:
            private_plan_length = max(
                len(actions),
                int(detail.get("private_plan_length", 0) or 0),
            )
        except (TypeError, ValueError):
            private_plan_length = len(actions)
        entry = {
            "family": spec.toolbox_family,
            "mechanic_family": spec.mechanic_family,
            "certified": certified,
            "certificate_reason": str(
                (plan or {}).get("certificate_reason")
                or (
                    "current binding and exact runtime verifier available"
                    if prediction is not None
                    else "advisory binding only; exact prediction unavailable"
                )
            ),
            "actions": actions,
            "private_plan_length": private_plan_length,
            "grounded_observation": grounding,
            "runtime_mechanic": runtime_row,
            "detail": detail,
        }
    runtime_row["plan_available"] = bool(entry)
    runtime_row["prediction_available"] = prediction is not None
    return {"runtime_mechanic": runtime_row, "plan": entry}


def verify_runtime_exact_diagnostic(
    exact: Mapping[str, Any], current_frame: Any
) -> dict[str, Any]:
    """Run a runtime verifier while preserving its repair/reject evidence."""

    if str(exact.get("verifier", "")) != "runtime_mechanic_transition_v1":
        return {"accepted": False, "classification": "invalid_verifier"}
    toolbox_family = str(exact.get("toolbox_family", "") or "")
    spec = runtime_mechanic_spec(toolbox_family)
    module = load_runtime_mechanic(toolbox_family)
    if spec is None or module is None:
        return {
            "accepted": False,
            "classification": "runtime_mechanic_unavailable",
            "toolbox_family": toolbox_family,
        }
    if str(exact.get("mechanic_family", "")) != spec.mechanic_family:
        return {
            "accepted": False,
            "classification": "mechanic_family_mismatch",
            "toolbox_family": toolbox_family,
        }
    prediction = exact.get("prediction")
    if not isinstance(prediction, Mapping):
        return {
            "accepted": False,
            "classification": "missing_prediction",
            "toolbox_family": toolbox_family,
        }
    try:
        verified = module.verify_transition(dict(prediction), current_frame)
        if isinstance(verified, Mapping):
            result = json_safe_mapping(verified) or {
                "accepted": bool(verified.get("accepted"))
            }
        else:
            result = {"accepted": bool(verified)}
    except Exception as exc:
        return {
            "accepted": False,
            "classification": "verifier_exception",
            "toolbox_family": toolbox_family,
            "error": f"{type(exc).__name__}: {exc}",
        }
    result["toolbox_family"] = toolbox_family
    evidence = result.get("evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}
    rejected = [str(value) for value in result.get("rejected_reasons", [])]
    if bool(result.get("accepted")):
        result["classification"] = (
            "model_repaired_match"
            if evidence.get("schema_repair_applied")
            else "exact_match"
        )
    elif rejected and all(
        value.startswith("unaffected_piece_changed:")
        or value == "visual_transition_not_verified"
        for value in rejected
    ) and bool(evidence.get("successor_digest_match")):
        result["classification"] = "world_model_residual"
    else:
        result["classification"] = "true_reject"
    return result


def verify_runtime_exact(exact: Mapping[str, Any], current_frame: Any) -> bool:
    return bool(verify_runtime_exact_diagnostic(exact, current_frame).get("accepted"))


__all__ = [
    "compute_runtime_artifact",
    "verify_runtime_exact",
    "verify_runtime_exact_diagnostic",
]
