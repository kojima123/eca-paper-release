"""Night bundle evolution from stored ECA snapshots."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from arcgym.eca.types import DriveWeights, NightBundle

_SHARED_TAIL_TRANSITION_KEY_FIELDS = (
    "tail_phase",
    "rearm_state",
    "visible_window",
    "churn_class",
    "contract_owner",
    "queue_health",
    "repeat_depth",
    "psa_risk",
    "candidate_family",
)
_SHARED_TAIL_TRANSITION_BACKOFF_ORDER = (
    "exact",
    "drop_repeat_depth",
    "drop_queue_health",
    "drop_visible_window",
    "candidate_family_global",
)
_SHARED_TAIL_TRANSITION_BACKOFF_DROPS = {
    "exact": (),
    "drop_repeat_depth": ("repeat_depth",),
    "drop_queue_health": ("repeat_depth", "queue_health"),
    "drop_visible_window": ("repeat_depth", "queue_health", "visible_window"),
    "candidate_family_global": tuple(
        field
        for field in _SHARED_TAIL_TRANSITION_KEY_FIELDS
        if field != "candidate_family"
    ),
}
_SHARED_TAIL_TRANSITION_COUNT_FIELDS = (
    "contract_survive_2",
    "frontier_gain_2",
    "turnover_gain_2",
    "stable_churn_return_2",
    "slip",
)
_SHARED_TAIL_TRANSITION_LAPLACE_ALPHA = 1.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _normalize_transition_count(value: Any) -> int:
    return max(0, int(value or 0))


def _canonicalize_transition_key(raw: dict[str, Any] | None) -> dict[str, str]:
    payload = dict(raw or {})
    return {
        field: str(payload.get(field, "*") or "*")
        for field in _SHARED_TAIL_TRANSITION_KEY_FIELDS
    }


def _serialize_transition_key(key: dict[str, Any] | None) -> str:
    return json.dumps(
        _canonicalize_transition_key(key),
        ensure_ascii=False,
        sort_keys=True,
    )


def _backoff_transition_key(*, key: dict[str, Any], level: str) -> dict[str, str]:
    normalized = _canonicalize_transition_key(key)
    for field in _SHARED_TAIL_TRANSITION_BACKOFF_DROPS.get(level, ()):
        normalized[field] = "*"
    return normalized


def _empty_transition_entry(*, key: dict[str, str]) -> dict[str, Any]:
    return {
        "key": dict(key),
        "support": 0,
        "counts": {
            field: 0 for field in _SHARED_TAIL_TRANSITION_COUNT_FIELDS
        },
        "outcome_class_histogram": {},
    }


def _coerce_transition_tables(
    previous_memory: dict[str, Any] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    tables: dict[str, dict[str, dict[str, Any]]] = {
        level: {}
        for level in _SHARED_TAIL_TRANSITION_BACKOFF_ORDER
    }
    raw_tables = dict((previous_memory or {}).get("tables") or {})
    for level in _SHARED_TAIL_TRANSITION_BACKOFF_ORDER:
        raw_table = raw_tables.get(level) or {}
        if isinstance(raw_table, dict):
            items = raw_table.items()
        elif isinstance(raw_table, list):
            items = [
                (
                    str(row.get("lookup_key") or ""),
                    row,
                )
                for row in raw_table
                if isinstance(row, dict)
            ]
        else:
            items = []
        for lookup_key, raw_entry in items:
            if not isinstance(raw_entry, dict):
                continue
            key = _canonicalize_transition_key(raw_entry.get("key"))
            entry_lookup_key = str(lookup_key or _serialize_transition_key(key))
            entry = _empty_transition_entry(key=key)
            entry["support"] = _normalize_transition_count(raw_entry.get("support"))
            raw_counts = dict(raw_entry.get("counts") or {})
            entry["counts"] = {
                field: _normalize_transition_count(raw_counts.get(field))
                for field in _SHARED_TAIL_TRANSITION_COUNT_FIELDS
            }
            entry["outcome_class_histogram"] = {
                str(name): _normalize_transition_count(count)
                for name, count in dict(
                    raw_entry.get("outcome_class_histogram") or {}
                ).items()
                if _normalize_transition_count(count) > 0
            }
            tables[level][entry_lookup_key] = entry
    return tables


def _transition_counts_from_sample(sample: dict[str, Any]) -> tuple[dict[str, int], str]:
    outcome = dict(sample.get("outcome") or {})
    slip = bool(outcome.get("no_target_slip_1")) or bool(
        outcome.get("psa_invalidation_1")
    )
    counts = {
        "contract_survive_2": int(bool(outcome.get("contract_survive_2"))),
        "frontier_gain_2": int(bool(outcome.get("frontier_gain_2"))),
        "turnover_gain_2": int(bool(outcome.get("turnover_gain_2"))),
        "stable_churn_return_2": int(bool(outcome.get("stable_churn_return_2"))),
        "slip": int(slip),
    }
    outcome_class = str(
        sample.get("post_final_shared_after_exit_outcome_class")
        or outcome.get("post_final_shared_after_exit_outcome_class")
        or "unobserved"
    )
    return counts, outcome_class


def _smoothed_probability(*, count: int, support: int) -> float:
    alpha = float(_SHARED_TAIL_TRANSITION_LAPLACE_ALPHA)
    return round(float((count + alpha) / (support + (2.0 * alpha))), 6)


def _finalize_transition_entry(entry: dict[str, Any]) -> dict[str, Any]:
    support = _normalize_transition_count(entry.get("support"))
    counts = {
        field: _normalize_transition_count(
            dict(entry.get("counts") or {}).get(field)
        )
        for field in _SHARED_TAIL_TRANSITION_COUNT_FIELDS
    }
    probabilities = {
        "p_contract": _smoothed_probability(
            count=counts["contract_survive_2"],
            support=support,
        ),
        "p_frontier": _smoothed_probability(
            count=counts["frontier_gain_2"],
            support=support,
        ),
        "p_turnover": _smoothed_probability(
            count=counts["turnover_gain_2"],
            support=support,
        ),
        "p_dead_churn": _smoothed_probability(
            count=counts["stable_churn_return_2"],
            support=support,
        ),
        "p_slip": _smoothed_probability(
            count=counts["slip"],
            support=support,
        ),
    }
    quality_gain = round(
        max(
            float(probabilities["p_frontier"]),
            float(probabilities["p_turnover"]),
        ),
        6,
    )
    predicted_anxiety = round(
        (0.45 * (1.0 - float(probabilities["p_contract"])))
        + (0.30 * float(probabilities["p_dead_churn"]))
        + (0.15 * (1.0 - quality_gain))
        + (0.10 * float(probabilities["p_slip"])),
        6,
    )
    return {
        "key": _canonicalize_transition_key(entry.get("key")),
        "support": support,
        "counts": counts,
        "outcome_class_histogram": {
            str(name): _normalize_transition_count(count)
            for name, count in dict(
                entry.get("outcome_class_histogram") or {}
            ).items()
            if _normalize_transition_count(count) > 0
        },
        "probabilities": probabilities,
        "quality_gain": quality_gain,
        "predicted_anxiety": predicted_anxiety,
    }


def _transition_rule_record(entry: dict[str, Any], *, level: str) -> dict[str, Any]:
    return {
        "level": level,
        "key": dict(entry.get("key") or {}),
        "support": _normalize_transition_count(entry.get("support")),
        "probabilities": dict(entry.get("probabilities") or {}),
        "quality_gain": round(float(entry.get("quality_gain", 0.0) or 0.0), 6),
        "predicted_anxiety": round(
            float(entry.get("predicted_anxiety", 0.0) or 0.0),
            6,
        ),
        "outcome_class_histogram": dict(entry.get("outcome_class_histogram") or {}),
    }


def _build_transition_top_rules(
    exact_table: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    entries = [entry for entry in exact_table.values() if int(entry.get("support", 0) or 0) > 0]
    good_entries = sorted(
        entries,
        key=lambda entry: (
            float(entry.get("predicted_anxiety", 1.0) or 1.0),
            -int(entry.get("support", 0) or 0),
            -float(dict(entry.get("probabilities") or {}).get("p_contract", 0.0) or 0.0),
            -float(entry.get("quality_gain", 0.0) or 0.0),
            str(dict(entry.get("key") or {}).get("candidate_family", "")),
        ),
    )[:5]
    bad_entries = sorted(
        entries,
        key=lambda entry: (
            -float(entry.get("predicted_anxiety", 0.0) or 0.0),
            -int(entry.get("support", 0) or 0),
            float(dict(entry.get("probabilities") or {}).get("p_contract", 0.0) or 0.0),
            float(entry.get("quality_gain", 0.0) or 0.0),
            str(dict(entry.get("key") or {}).get("candidate_family", "")),
        ),
    )[:5]
    return {
        "good": [
            _transition_rule_record(entry, level="exact")
            for entry in good_entries
        ],
        "bad": [
            _transition_rule_record(entry, level="exact")
            for entry in bad_entries
        ],
    }


def build_shared_tail_transition_memory(
    *,
    previous_memory: dict[str, Any] | None,
    transition_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    tables = _coerce_transition_tables(previous_memory)
    for sample in transition_samples:
        key = _canonicalize_transition_key(sample.get("key"))
        counts, outcome_class = _transition_counts_from_sample(sample)
        for level in _SHARED_TAIL_TRANSITION_BACKOFF_ORDER:
            backoff_key = _backoff_transition_key(key=key, level=level)
            lookup_key = _serialize_transition_key(backoff_key)
            table = tables[level]
            entry = table.setdefault(
                lookup_key,
                _empty_transition_entry(key=backoff_key),
            )
            entry["support"] = int(entry.get("support", 0) or 0) + 1
            count_map = dict(entry.get("counts") or {})
            for field, increment in counts.items():
                count_map[field] = int(count_map.get(field, 0) or 0) + int(
                    increment or 0
                )
            entry["counts"] = count_map
            outcome_histogram = dict(entry.get("outcome_class_histogram") or {})
            outcome_histogram[outcome_class] = int(
                outcome_histogram.get(outcome_class, 0) or 0
            ) + 1
            entry["outcome_class_histogram"] = outcome_histogram

    finalized_tables = {
        level: {
            lookup_key: _finalize_transition_entry(entry)
            for lookup_key, entry in sorted(table.items())
        }
        for level, table in tables.items()
    }
    exact_table = finalized_tables.get("exact") or {}
    total_samples = sum(
        int(entry.get("support", 0) or 0)
        for entry in exact_table.values()
    )
    return {
        "schema_version": "shared_tail_transition_memory_v1",
        "laplace_alpha": float(_SHARED_TAIL_TRANSITION_LAPLACE_ALPHA),
        "samples_added_this_run": int(len(transition_samples)),
        "total_samples": int(total_samples),
        "backoff_order": list(_SHARED_TAIL_TRANSITION_BACKOFF_ORDER),
        "table_rule_counts": {
            level: int(len(table))
            for level, table in finalized_tables.items()
        },
        "tables": finalized_tables,
        "top_rules": _build_transition_top_rules(exact_table),
    }


def evolve_bundle(
    *,
    previous_bundle: NightBundle,
    snapshots: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    transition_samples: list[dict[str, Any]],
    game_id: str | None,
    thresholds: dict[str, float],
) -> NightBundle:
    if not snapshots:
        return NightBundle.from_mapping(previous_bundle.to_dict(), game_id=game_id)

    top_drive_counts = Counter(str(row.get("top_drive", "unknown")) for row in snapshots)
    reason_counts = Counter(str(row.get("reason", "")) for row in episodes if row.get("reason"))
    avg_total = sum(float(row.get("total", 0.0) or 0.0) for row in snapshots) / len(snapshots)
    high_anxiety_count = sum(1 for row in snapshots if row.get("high_anxiety"))
    interrupt_count = sum(1 for row in episodes if row.get("interrupt"))

    base_weights = DriveWeights.from_mapping(previous_bundle.drive_weights).to_dict()
    if top_drive_counts:
        dominant, dominant_count = top_drive_counts.most_common(1)[0]
        total = max(1, sum(top_drive_counts.values()))
        if dominant in base_weights:
            base_weights[dominant] = _clamp(base_weights[dominant] + (dominant_count / total) * 0.10, 0.05, 0.70)
            base_weights = DriveWeights.from_mapping(base_weights).to_dict()

    new_thresholds = dict(thresholds)
    new_thresholds.update(previous_bundle.thresholds or {})
    if interrupt_count >= max(2, len(snapshots) // 3):
        new_thresholds["threshold_high"] = _clamp(float(new_thresholds.get("threshold_high", 0.62)) + 0.03, 0.20, 0.95)
    elif high_anxiety_count and interrupt_count == 0:
        new_thresholds["threshold_high"] = _clamp(float(new_thresholds.get("threshold_high", 0.62)) - 0.03, 0.20, 0.95)

    world_rows = [row for row in snapshots if row.get("top_drive") == "world"]
    structure_rows = [row for row in snapshots if row.get("top_drive") == "structure"]
    self_rows = [row for row in snapshots if row.get("top_drive") == "self"]
    shared_tail_transition_memory = build_shared_tail_transition_memory(
        previous_memory=dict(
            (previous_bundle.world_model_bundle or {}).get(
                "shared_tail_transition_memory"
            )
            or {}
        ),
        transition_samples=transition_samples,
    )

    return NightBundle(
        game_id=game_id,
        version=previous_bundle.version,
        updated_at=datetime.now(timezone.utc).isoformat(),
        drive_weights=base_weights,
        thresholds=new_thresholds,
        world_model_bundle={
            "high_world_events": len(world_rows),
            "repeated_state_hashes": Counter(str(row.get("state_hash", "")) for row in world_rows).most_common(5),
            "avg_no_change_streak": (
                sum(int(row.get("no_change_streak", 0) or 0) for row in world_rows) / max(1, len(world_rows))
            ),
            "shared_tail_transition_memory": shared_tail_transition_memory,
        },
        structure_memory_bundle={
            "high_structure_events": len(structure_rows),
            "avg_unresolved_components": (
                sum(int(row.get("unresolved_components", 0) or 0) for row in structure_rows)
                / max(1, len(structure_rows))
            ),
            "top_drive_counts": dict(top_drive_counts),
        },
        self_model_bundle={
            "high_self_events": len(self_rows),
            "interrupt_count": interrupt_count,
            "recurring_failure_modes": [reason for reason, _ in reason_counts.most_common(5)],
        },
        night_summary={
            "snapshots": len(snapshots),
            "episodes": len(episodes),
            "avg_total_anxiety": round(float(avg_total), 6),
            "high_anxiety_snapshots": int(high_anxiety_count),
            "interrupts": int(interrupt_count),
            "top_drive_histogram": dict(top_drive_counts),
            "shared_tail_transition_samples_this_run": int(len(transition_samples)),
            "shared_tail_transition_samples_total": int(
                shared_tail_transition_memory.get("total_samples", 0) or 0
            ),
        },
    )
