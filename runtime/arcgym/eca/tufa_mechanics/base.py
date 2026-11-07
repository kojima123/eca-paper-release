"""Shared fail-closed protocol helpers for Tufa mechanic modules."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np


REQUIRED_MODULE_CALLABLES = (
    "compile_schema",
    "bind_current",
    "propose_plan",
    "predict_transition",
    "verify_transition",
)


@dataclass(frozen=True)
class MechanicContext:
    """Episode context available to frame-grounded mechanic modules.

    ``frame`` remains the only source of current coordinates. History and
    memory carry transition evidence and coordinate-free schemas so a module
    can learn an operator on one level and rebind it on the next.
    """

    frame: Any
    level: int = 0
    history: tuple[Mapping[str, Any], ...] = ()
    previous_action: str = ""
    previous_frame: Any = None
    score: int = 0
    memory: Mapping[str, Any] = field(default_factory=dict)

    def module_input(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "level": int(self.level),
            "history": list(self.history),
            "previous_action": str(self.previous_action),
            "previous_frame": self.previous_frame,
            "score": int(self.score),
            "memory": dict(self.memory),
        }

    def public_summary(self) -> dict[str, Any]:
        schemas = self.memory.get("schemas", []) if isinstance(self.memory, Mapping) else []
        return {
            "level": int(self.level),
            "score": int(self.score),
            "history_count": len(self.history),
            "previous_action": str(self.previous_action),
            "memory_schema_count": len(schemas) if isinstance(schemas, Sequence) else 0,
        }

    def cache_payload(self) -> dict[str, Any]:
        history_tail: list[dict[str, Any]] = []
        for item in self.history[-8:]:
            if not isinstance(item, Mapping):
                continue
            row: dict[str, Any] = {"action": str(item.get("action", ""))}
            if item.get("frame") is not None:
                history_frame = as_frame(item.get("frame"))
                if history_frame is not None:
                    row["frame_digest"] = hashlib.sha256(
                        np.ascontiguousarray(history_frame).tobytes()
                    ).hexdigest()
            history_tail.append(row)
        return {
            "level": int(self.level),
            "score": int(self.score),
            "previous_action": str(self.previous_action),
            "history_tail": history_tail,
            "memory": stable_data(dict(self.memory)),
        }


def coerce_mechanic_context(
    frame: Any,
    *,
    level: int = 0,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> MechanicContext:
    if isinstance(context, MechanicContext):
        return MechanicContext(
            frame=frame,
            level=int(level if level is not None else context.level),
            history=context.history,
            previous_action=context.previous_action,
            previous_frame=context.previous_frame,
            score=context.score,
            memory=context.memory,
        )
    raw = context if isinstance(context, Mapping) else {}
    history = tuple(
        item for item in raw.get("history", ()) if isinstance(item, Mapping)
    )[-64:]
    return MechanicContext(
        frame=frame,
        level=int(raw.get("level", level) or 0),
        history=history,
        previous_action=str(raw.get("previous_action", "") or ""),
        previous_frame=raw.get("previous_frame"),
        score=int(raw.get("score", 0) or 0),
        memory=(
            dict(raw.get("memory", {}))
            if isinstance(raw.get("memory", {}), Mapping)
            else {}
        ),
    )


def stable_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): stable_data(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, np.ndarray):
        return stable_data(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        rows = [stable_data(item) for item in value]
        return sorted(rows, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [stable_data(item) for item in value]
    return value


def stable_digest(value: Any) -> str:
    payload = json.dumps(
        stable_data(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def as_frame(value: Any) -> np.ndarray | None:
    if isinstance(value, MechanicContext):
        value = value.frame
    elif isinstance(value, Mapping) and value.get("frame") is not None:
        value = value.get("frame")
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


def module_protocol_errors(module: ModuleType) -> list[str]:
    errors: list[str] = []
    family_id = str(getattr(module, "FAMILY_ID", "") or "").strip()
    if not family_id:
        errors.append("missing FAMILY_ID")
    for name in REQUIRED_MODULE_CALLABLES:
        if not callable(getattr(module, name, None)):
            errors.append(f"missing callable {name}")
    return errors


def json_safe_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    normalized = stable_data(dict(value))
    try:
        json.dumps(normalized, ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError):
        return None
    return normalized


def normalize_actions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    actions: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            action = {"action": str(item).strip().upper()}
        elif isinstance(item, Mapping):
            action = stable_data(dict(item))
            action["action"] = str(action.get("action", "")).strip().upper()
        else:
            return []
        if not action.get("action"):
            return []
        if action["action"] == "MOUSE":
            try:
                action["row"] = int(action["row"])
                action["col"] = int(action["col"])
            except (KeyError, TypeError, ValueError):
                return []
        actions.append(action)
    return actions


def normalize_plan(value: Any) -> dict[str, Any] | None:
    plan = json_safe_mapping(value)
    if plan is None:
        return None
    actions = normalize_actions(plan.get("actions"))
    if not actions:
        return None
    plan["actions"] = actions
    plan["action_count"] = len(actions)
    return plan


__all__ = [
    "MechanicContext",
    "REQUIRED_MODULE_CALLABLES",
    "as_frame",
    "coerce_mechanic_context",
    "json_safe_mapping",
    "module_protocol_errors",
    "normalize_actions",
    "normalize_plan",
    "stable_data",
    "stable_digest",
]
