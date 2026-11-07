"""Offline candidate-ranker baseline for ECA modeling logs.

This is not a learned controller.  It scores exported
`candidate_ranking_example.v2` rows so we can measure whether the current
candidate features carry enough signal for P2 ranker training.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


SOURCE_PRIORS = {
    "pending_contact": 0.55,
    "symmetric_seed": 0.50,
    "selected_target": 0.35,
    "candidate_extractor": 0.32,
    "object_table": 0.26,
    "local_controller": 0.22,
}


def load_candidate_ranking_examples(path: Path) -> list[dict[str, Any]]:
    """Load candidate ranking examples from a JSONL file or run directory."""
    path = Path(path)
    if path.is_dir():
        path = path / "eca_candidate_ranking_examples.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    examples: list[dict[str, Any]] = []
    for line in lines:
        text = line.strip().lstrip("\ufeff")
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            examples.append(item)
    return examples


def score_candidate_example(
    example: Mapping[str, Any],
    *,
    profile: str = "deployable",
) -> float:
    """Score one candidate without reading its label.

    `deployable` uses features that are close to creation-time signals.
    `trace` additionally uses queue-lifecycle features for offline diagnosis.
    """
    features = dict(example.get("features") or {})
    source = str(features.get("source", "") or "").strip().lower()
    score = SOURCE_PRIORS.get(source, 0.10)
    if _as_bool(features.get("has_site")):
        score += 0.06
    if str(features.get("branch_direction", "") or "") in {"west", "east"}:
        score += 0.04

    created_step = max(0, _as_int(features.get("created_step"), 0))
    budget_used = max(1, _as_int(features.get("trajectory_budget_used"), 1))
    relative_creation = min(1.0, created_step / budget_used)
    score += 0.08 * (1.0 - relative_creation)

    # Row-progress bias is intentionally weak.  In ls20-like traces, lower rows
    # often mean a deeper corridor, but this must not dominate cross-game data.
    site_row = _as_int(features.get("site_row"), 0)
    if site_row:
        score += min(0.08, max(0.0, site_row / 800.0))

    if profile == "trace":
        if _as_bool(features.get("was_advanced")):
            score += 0.18
        if _as_bool(features.get("was_queued")):
            score += 0.08
        score += min(0.08, max(0, _as_int(features.get("status_count"), 0)) * 0.01)

    return round(score, 6)


def rank_candidate_examples(
    examples: Iterable[Mapping[str, Any]],
    *,
    profile: str = "deployable",
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for example in examples:
        row = dict(example)
        row["ranker_score"] = score_candidate_example(row, profile=profile)
        ranked.append(row)
    return sorted(
        ranked,
        key=lambda item: (
            -_as_float(item.get("ranker_score"), 0.0),
            str(item.get("candidate_id", "") or ""),
        ),
    )


def _label_value(example: Mapping[str, Any]) -> float:
    label = dict(example.get("label") or {})
    return _as_float(label.get("candidate_value"), 0.0)


def _dcg(values: list[float]) -> float:
    total = 0.0
    for idx, value in enumerate(values):
        relevance = max(0.0, value)
        total += (2.0**relevance - 1.0) / math.log2(idx + 2.0)
    return total


def _ndcg(ranked: list[Mapping[str, Any]]) -> float:
    values = [_label_value(item) for item in ranked]
    ideal = sorted(values, reverse=True)
    ideal_dcg = _dcg(ideal)
    if ideal_dcg <= 0.0:
        return 0.0
    return _dcg(values) / ideal_dcg


def _pairwise_accuracy(ranked: list[Mapping[str, Any]]) -> tuple[int, int]:
    by_id = {str(item.get("candidate_id", "") or ""): idx for idx, item in enumerate(ranked)}
    comparable = 0
    correct = 0
    for left_idx, left in enumerate(ranked):
        for right in ranked[left_idx + 1 :]:
            left_value = _label_value(left)
            right_value = _label_value(right)
            if abs(left_value - right_value) <= 1e-9:
                continue
            comparable += 1
            left_should_rank_first = left_value > right_value
            left_ranked_first = by_id[str(left.get("candidate_id", "") or "")] < by_id[
                str(right.get("candidate_id", "") or "")
            ]
            if left_should_rank_first == left_ranked_first:
                correct += 1
    return correct, comparable


def evaluate_candidate_ranker(
    examples: Iterable[Mapping[str, Any]],
    *,
    profile: str = "deployable",
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in examples:
        group_id = str(raw.get("group_id", "") or raw.get("run_id", "") or "default")
        groups[group_id].append(dict(raw))

    group_metrics: list[dict[str, Any]] = []
    pairwise_correct = 0
    pairwise_total = 0
    ndcg_total = 0.0
    top1_promotions = 0
    top1_positive = 0
    evaluated_groups = 0

    for group_id, group in sorted(groups.items()):
        if not group:
            continue
        ranked = rank_candidate_examples(group, profile=profile)
        top1 = ranked[0]
        correct, total = _pairwise_accuracy(ranked)
        pairwise_correct += correct
        pairwise_total += total
        ndcg = _ndcg(ranked)
        ndcg_total += ndcg
        evaluated_groups += 1
        if _as_bool((top1.get("label") or {}).get("promotion")):
            top1_promotions += 1
        if _label_value(top1) > 0.0:
            top1_positive += 1
        group_metrics.append(
            {
                "group_id": group_id,
                "candidate_count": len(group),
                "ndcg": round(ndcg, 6),
                "pairwise_accuracy": round(correct / total, 6) if total else 0.0,
                "pairwise_comparable": total,
                "top1_candidate_id": str(top1.get("candidate_id", "") or ""),
                "top1_score": _as_float(top1.get("ranker_score"), 0.0),
                "top1_label_value": _label_value(top1),
                "top1_promotion": _as_bool((top1.get("label") or {}).get("promotion")),
            }
        )

    return {
        "schema_version": "candidate_ranker_eval.v1",
        "profile": profile,
        "group_count": evaluated_groups,
        "example_count": sum(len(group) for group in groups.values()),
        "mean_ndcg": round(ndcg_total / evaluated_groups, 6) if evaluated_groups else 0.0,
        "pairwise_accuracy": (
            round(pairwise_correct / pairwise_total, 6) if pairwise_total else 0.0
        ),
        "pairwise_comparable": pairwise_total,
        "top1_promotion_rate": (
            round(top1_promotions / evaluated_groups, 6) if evaluated_groups else 0.0
        ),
        "top1_positive_value_rate": (
            round(top1_positive / evaluated_groups, 6) if evaluated_groups else 0.0
        ),
        "groups": group_metrics,
    }
