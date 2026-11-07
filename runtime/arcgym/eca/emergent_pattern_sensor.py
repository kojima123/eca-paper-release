"""Diagnostic sensor for cross-episode emergent rod pattern mining.

This wraps the parallel Gap-6 rod mining work as an ECA sensor. It consumes
already-observed CausalTraceRod dictionaries from benchmark rows and reports
whether any coordinate-free signatures have enough repeated support to be
considered mechanistic hypotheses. It does not inject hypotheses, rank
candidates, or authorize actions.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from .rod_pattern_mining import (
    cluster_rod_dicts,
    propose_emergent_candidates,
    summarize_pattern_mining,
)


EMERGENT_ROD_PATTERN_SENSOR_SCHEMA_VERSION = (
    "eca_emergent_rod_pattern_mining_sensor.v1"
)


def summarize_emergent_rod_pattern_mining(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_rod_count: int = 3,
    min_net_support: int = 2,
    max_contradiction_rate: float = 0.25,
    sample_limit: int = 8,
) -> dict[str, Any]:
    """Summarize repeated structural rod signatures across benchmark rows."""

    rod_dicts: list[dict[str, Any]] = []
    episode_ids: list[str] = []
    variants: list[str] = []
    game_ids: list[str] = []
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[str] = set()

    for row_index, row in enumerate(rows):
        game_id = str(row.get("game_id", "") or "unknown")
        variant = str(row.get("variant", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for step_index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            for rod in _extract_rod_dicts(item):
                key = _dedupe_key(game_id, variant, rod)
                if key in seen:
                    continue
                seen.add(key)
                rod_dicts.append(rod)
                episode_ids.append(f"row={row_index}|step={step_index}")
                variants.append(variant)
                game_ids.append(game_id)
                by_game[game_id]["rod_count"] += 1
                effect = rod.get("effect")
                effect_type = ""
                if isinstance(effect, Mapping):
                    effect_type = str(effect.get("effect_type", "") or "")
                if effect_type:
                    by_game[game_id][f"effect:{effect_type}"] += 1

    clusters = cluster_rod_dicts(
        rod_dicts,
        episode_ids=episode_ids,
        variants=variants,
        game_ids=game_ids,
    )
    candidates = propose_emergent_candidates(
        clusters,
        min_rod_count=min_rod_count,
        min_net_support=min_net_support,
        max_contradiction_rate=max_contradiction_rate,
        min_distinct_episodes=1,
        min_distinct_variants=0,
        cross_game_bonus=True,
    )
    mining = summarize_pattern_mining(clusters, candidates)
    cross_game_candidates = [
        candidate for candidate in candidates if int(candidate.distinct_games) >= 2
    ]
    single_game_candidates = [
        candidate for candidate in candidates if int(candidate.distinct_games) < 2
    ]
    return {
        "schema_version": EMERGENT_ROD_PATTERN_SENSOR_SCHEMA_VERSION,
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
        "mechanic_hypothesis_injection_enabled": False,
        "source": "parallel_gap6_rod_pattern_mining_intake",
        "input_rod_count": int(len(rod_dicts)),
        "cluster_count": int(mining.get("cluster_count", 0)),
        "candidate_count": int(mining.get("candidate_count", 0)),
        "cross_game_candidate_count": int(len(cross_game_candidates)),
        "single_game_candidate_count": int(len(single_game_candidates)),
        "top_candidate_confidence": float(mining.get("top_candidate_confidence", 0.0)),
        "candidate_signatures": list(mining.get("candidate_signatures") or [])[
            :sample_limit
        ],
        "cluster_size_histogram": dict(mining.get("cluster_size_histogram") or {}),
        "candidate_samples": [
            candidate.to_dict() for candidate in candidates[:sample_limit]
        ],
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "next_planner_contract": {
            "required_before_injection": (
                "real protected probes with score/control margin; no shadow-only "
                "confirmation"
            ),
            "allowed_connection": "diagnostic_only_until_real_probe_outcome_ledger",
            "action_generation_allowed": False,
            "llm_role": "none",
        },
    }


def _extract_rod_dicts(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    rods: list[dict[str, Any]] = []
    for key in ("causal_trace_rod", "rod"):
        payload = item.get(key)
        if isinstance(payload, Mapping):
            rods.append(dict(payload))
    payload = item.get("recent_rods")
    if isinstance(payload, list):
        rods.extend(dict(row) for row in payload if isinstance(row, Mapping))

    structural = item.get("structural_rods")
    if not isinstance(structural, Mapping):
        diagnostics = item.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            structural = diagnostics.get("structural_rods")
    if isinstance(structural, Mapping):
        recent = structural.get("recent")
        if isinstance(recent, list):
            rods.extend(dict(row) for row in recent if isinstance(row, Mapping))
    return [rod for rod in rods if _is_rod_like(rod)]


def _is_rod_like(rod: Mapping[str, Any]) -> bool:
    effect = rod.get("effect")
    return isinstance(effect, Mapping) or bool(
        rod.get("action_name") or rod.get("primitive_id") or rod.get("generic_keys")
    )


def _dedupe_key(game_id: str, variant: str, rod: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(rod), sort_keys=True, separators=(",", ":"), default=str)
    return f"{game_id}|{variant}|{payload}"


__all__ = [
    "EMERGENT_ROD_PATTERN_SENSOR_SCHEMA_VERSION",
    "summarize_emergent_rod_pattern_mining",
]
