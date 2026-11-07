"""Family-neutral descriptor repair contract helpers.

The first runtime client is the slot-pool descriptor repair loop, but the
posterior key shape is intentionally not slot-pool specific. Keeping this
contract in one module makes future coverage/rule/cycle repair clients use the
same evidence accounting instead of growing separate posterior formats.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


REPAIR_CONTRACT_KEY_FIELDS = (
    "source_kind",
    "family",
    "mechanic_state_type",
    "goal_link_type",
    "reject_reason",
    "abstraction_gap",
    "repair_target",
)


@dataclass(frozen=True)
class DescriptorRepairContractKey:
    source_kind: str
    family: str
    mechanic_state_type: str
    goal_link_type: str
    reject_reason: str
    abstraction_gap: str
    repair_target: str

    def to_key(self) -> str:
        return "|".join(
            [
                self.source_kind,
                self.family,
                self.mechanic_state_type,
                self.goal_link_type,
                self.reject_reason,
                self.abstraction_gap,
                self.repair_target,
            ]
        )

    def to_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in REPAIR_CONTRACT_KEY_FIELDS}


def _clean(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else str(default or "")


def build_descriptor_repair_key(
    *,
    source_kind: Any = "",
    family: Any = "",
    mechanic_state_type: Any = "",
    goal_link_type: Any = "",
    reject_reason: Any = "",
    abstraction_gap: Any = "",
    repair_target: Any = "",
    defaults: Mapping[str, Any] | None = None,
) -> str:
    defaults = defaults if isinstance(defaults, Mapping) else {}
    contract = DescriptorRepairContractKey(
        source_kind=_clean(source_kind, _clean(defaults.get("source_kind"), "unknown")),
        family=_clean(family, _clean(defaults.get("family"), "unknown")),
        mechanic_state_type=_clean(
            mechanic_state_type, _clean(defaults.get("mechanic_state_type"), "unknown")
        ),
        goal_link_type=_clean(goal_link_type, _clean(defaults.get("goal_link_type"), "")),
        reject_reason=_clean(reject_reason, _clean(defaults.get("reject_reason"), "")),
        abstraction_gap=_clean(
            abstraction_gap, _clean(defaults.get("abstraction_gap"), "")
        ),
        repair_target=_clean(repair_target, _clean(defaults.get("repair_target"), "")),
    )
    return contract.to_key()


def parse_descriptor_repair_key(key: str) -> DescriptorRepairContractKey | None:
    parts = str(key or "").split("|")
    if len(parts) != len(REPAIR_CONTRACT_KEY_FIELDS):
        return None
    return DescriptorRepairContractKey(**dict(zip(REPAIR_CONTRACT_KEY_FIELDS, parts)))


def descriptor_repair_key_mismatches(
    key: str,
    expected: Mapping[str, Any],
    *,
    fields: Iterable[str] = REPAIR_CONTRACT_KEY_FIELDS,
) -> tuple[str, ...]:
    parsed = parse_descriptor_repair_key(key)
    if parsed is None:
        return ("invalid_key",)
    mismatches: list[str] = []
    for field in fields:
        if field not in REPAIR_CONTRACT_KEY_FIELDS:
            continue
        expected_value = _clean(expected.get(field), "")
        if expected_value and getattr(parsed, field) != expected_value:
            mismatches.append(field)
    return tuple(mismatches)


def descriptor_repair_key_matches(
    key: str,
    expected: Mapping[str, Any],
    *,
    fields: Iterable[str] = REPAIR_CONTRACT_KEY_FIELDS,
) -> bool:
    return not descriptor_repair_key_mismatches(key, expected, fields=fields)


def descriptor_repair_confidence(counter: Mapping[str, Any]) -> float:
    attempts = int(counter.get("attempts", 0) or 0)
    repair_successes = int(counter.get("repair_successes", 0) or 0)
    validation_successes = int(counter.get("validation_successes", 0) or 0)
    execution_successes = int(counter.get("execution_successes", 0) or 0)
    score_successes = int(counter.get("score_successes", 0) or 0)
    no_score_events = int(counter.get("no_score_events", 0) or 0)
    repair_failures = max(0, attempts - repair_successes)
    validation_failures = max(0, attempts - validation_successes)
    return max(
        0.0,
        min(
            0.99,
            0.35
            + 0.10 * min(3, validation_successes)
            + 0.10 * min(3, execution_successes)
            + 0.15 * min(3, score_successes)
            - 0.08 * min(3, repair_failures)
            - 0.08 * min(3, validation_failures)
            - 0.05 * min(3, no_score_events),
        ),
    )


def descriptor_repair_suppression_reason(counter: Mapping[str, Any]) -> str:
    attempts = int(counter.get("attempts", 0) or 0)
    repair_successes = int(counter.get("repair_successes", 0) or 0)
    validation_successes = int(counter.get("validation_successes", 0) or 0)
    execution_events = int(counter.get("execution_events", 0) or 0)
    score_successes = int(counter.get("score_successes", 0) or 0)
    no_score_events = int(counter.get("no_score_events", 0) or 0)
    if attempts >= 2 and repair_successes == 0:
        return "repair_failed"
    if attempts >= 2 and validation_successes == 0:
        return "validation_failed"
    if execution_events >= 3 and no_score_events >= 3 and score_successes == 0:
        return "score_stagnant"
    return ""
