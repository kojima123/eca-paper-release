"""Offline risk-predictor baseline for ECA modeling logs.

The functions here evaluate exported `risk_prediction_example.v2` rows.  They
are deliberately heuristic: the goal is to expose metrics and feature gaps
before wiring a learned predictor into the controller.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


BINARY_RISK_TARGETS = (
    "p_queue_exhaustion_5",
    "p_no_target_collapse_5",
    "p_dead_revisit_5",
    "p_selected_target_empty_5",
    "p_wrong_branch_lockin_5",
    "p_budget_exhaustion_before_test",
    "p_progress_5",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _clip01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def load_risk_prediction_examples(path: Path) -> list[dict[str, Any]]:
    """Load risk prediction examples from a JSONL file or run directory."""
    path = Path(path)
    if path.is_dir():
        path = path / "eca_risk_prediction_examples.jsonl"
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


def predict_risk_example(example: Mapping[str, Any]) -> dict[str, float]:
    """Predict risk probabilities without reading the label."""
    features = dict(example.get("features") or {})
    remaining = _as_int(features.get("remaining_actions"), 0)
    pool_size = _as_int(features.get("candidate_pool_size"), 0)
    no_change = _as_int(features.get("no_change_streak"), 0)
    repeated = _as_int(features.get("repeated_state_count"), 0)
    queue_exhaustions = _as_int(features.get("queue_exhaustions"), 0)
    dead_tail = _as_float(features.get("candidate_dead_tail_ratio"), 0.0)
    uncertainty = _as_float(features.get("structural_time_uncertainty"), 0.0)
    goal_progress = _as_float(features.get("structural_time_goal_progress"), 0.0)
    target_dead_ratio = _as_float(features.get("target_health_dead_ratio"), 0.0)
    target_no_score = _as_int(features.get("target_health_no_score_streak"), 0)
    selected_present = _as_bool(features.get("selected_target_present"))
    selected_empty = _as_bool(features.get("selected_target_empty_with_candidate_pool"))
    no_target_disposition = str(features.get("no_target_disposition", "") or "")
    control_ticket_count = _as_int(features.get("control_ir_live_ticket_count"), 0)
    live_escape_count = _as_int(features.get("live_escape_candidate_count"), 0)

    low_remaining = 1.0 if remaining <= 5 else (0.5 if remaining <= 15 else 0.0)
    no_target_signal = 1.0 if no_target_disposition else 0.0
    terminal_no_target = 1.0 if no_target_disposition in {
        "preterminal_dead_revisit",
        "hard_terminal",
        "collapse",
    } else 0.0

    p_progress = (
        0.12
        + (0.22 if selected_present else 0.0)
        + min(0.18, live_escape_count * 0.04)
        + min(0.12, control_ticket_count * 0.03)
        + min(0.18, max(0.0, goal_progress) * 0.4)
        - min(0.28, no_change * 0.025)
        - min(0.22, dead_tail * 0.22)
        - (0.18 if selected_empty else 0.0)
    )
    p_queue_exhaustion = (
        0.04
        + min(0.24, queue_exhaustions * 0.015)
        + min(0.24, no_change * 0.035)
        + (0.12 if pool_size == 0 else 0.0)
        + 0.10 * low_remaining
    )
    p_no_target = (
        0.04
        + (0.55 if selected_empty else 0.0)
        + 0.36 * no_target_signal
        + 0.25 * terminal_no_target
        + min(0.20, dead_tail * 0.2)
        + (0.08 if pool_size == 0 else 0.0)
    )
    p_dead_revisit = (
        0.02
        + (0.70 if no_target_disposition == "preterminal_dead_revisit" else 0.0)
        + min(0.16, repeated * 0.025)
        + min(0.18, target_dead_ratio * 0.18)
        + min(0.12, no_change * 0.015)
    )
    p_selected_empty = (
        0.03
        + (0.72 if selected_empty else 0.0)
        + (0.16 if pool_size > 0 and not selected_present else 0.0)
        + min(0.12, target_no_score * 0.015)
    )
    p_wrong_branch = (
        0.03
        + 0.12 * no_target_signal
        + 0.14 * low_remaining
        + min(0.12, no_change * 0.015)
        + min(0.12, uncertainty * 0.2)
    )
    p_budget = (
        0.02
        + 0.38 * low_remaining
        + (0.16 if pool_size > 0 else 0.0)
        + (0.18 if selected_empty else 0.0)
        + min(0.10, no_change * 0.01)
    )
    expected_recovery_steps = 0.0
    if p_wrong_branch > 0.25 or p_no_target > 0.35:
        expected_recovery_steps = min(20.0, 2.0 + no_change + pool_size * 1.5)

    return {
        "p_queue_exhaustion_5": round(_clip01(p_queue_exhaustion), 6),
        "p_no_target_collapse_5": round(_clip01(p_no_target), 6),
        "p_dead_revisit_5": round(_clip01(p_dead_revisit), 6),
        "p_selected_target_empty_5": round(_clip01(p_selected_empty), 6),
        "p_wrong_branch_lockin_5": round(_clip01(p_wrong_branch), 6),
        "p_budget_exhaustion_before_test": round(_clip01(p_budget), 6),
        "p_progress_5": round(_clip01(p_progress), 6),
        "expected_recovery_steps": round(expected_recovery_steps, 6),
    }


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _average_precision(scores: list[float], labels: list[bool]) -> float | None:
    positive_count = sum(1 for label in labels if label)
    if positive_count == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hits = 0
    precision_sum = 0.0
    for idx, (_, label) in enumerate(ranked, start=1):
        if not label:
            continue
        hits += 1
        precision_sum += hits / idx
    return precision_sum / positive_count


def _binary_metrics(predictions: list[dict[str, float]], examples: list[Mapping[str, Any]], target: str) -> dict[str, Any]:
    labels = [_as_bool((example.get("label") or {}).get(target)) for example in examples]
    scores = [_as_float(prediction.get(target), 0.0) for prediction in predictions]
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    brier = (
        sum((score - (1.0 if label else 0.0)) ** 2 for score, label in zip(scores, labels))
        / len(labels)
        if labels
        else 0.0
    )
    auc = _auc(scores, labels)
    ap = _average_precision(scores, labels)
    return {
        "target": target,
        "positive_count": positives,
        "negative_count": negatives,
        "auroc": round(auc, 6) if auc is not None else None,
        "pr_auc": round(ap, 6) if ap is not None else None,
        "brier": round(brier, 6),
    }


def evaluate_risk_predictor(examples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(example) for example in examples]
    predictions = [predict_risk_example(example) for example in rows]
    target_metrics = [
        _binary_metrics(predictions, rows, target)
        for target in BINARY_RISK_TARGETS
    ]
    if rows:
        recovery_mae = sum(
            abs(
                _as_float(prediction.get("expected_recovery_steps"), 0.0)
                - _as_float((example.get("label") or {}).get("expected_recovery_steps"), 0.0)
            )
            for prediction, example in zip(predictions, rows)
        ) / len(rows)
    else:
        recovery_mae = 0.0
    return {
        "schema_version": "risk_predictor_eval.v1",
        "example_count": len(rows),
        "target_metrics": target_metrics,
        "expected_recovery_steps_mae": round(recovery_mae, 6),
    }
