"""Trainable offline ECA policy bundle.

This module is the first dependency-free learned baseline over the v0.2 ECA
modeling examples.  It deliberately stays outside controller execution unless
called through the dry-run runtime bridge at the bottom of the file.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from arcgym.eca.action_prior import (
    build_action_prior_examples_from_bundle,
    normalize_action_name,
)
from arcgym.eca.mode_selector import build_mode_selection_examples_from_bundle
from arcgym.eca.modeling_batch import find_modeling_run_dirs
from arcgym.eca.modeling_logs import build_modeling_logs_from_run_dir
from arcgym.eca.risk_predictor import BINARY_RISK_TARGETS


POLICY_BUNDLE_SCHEMA_VERSION = "eca_policy_bundle.v1"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


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


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _mean_bucket(values: Iterable[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    return {
        "count": len(rows),
        "mean": round(sum(rows) / len(rows), 6) if rows else 0.0,
    }


def _count_bucket(values: Iterable[str]) -> dict[str, Any]:
    counts = Counter(str(value or "") for value in values if str(value or ""))
    total = sum(counts.values())
    return {
        "count": total,
        "counts": dict(sorted(counts.items())),
    }


def _top_count(counts: Mapping[str, Any]) -> tuple[str, float]:
    items = [(str(key), _as_int(value, 0)) for key, value in dict(counts or {}).items()]
    items = [item for item in items if item[0]]
    if not items:
        return "", 0.0
    items.sort(key=lambda item: (-item[1], item[0]))
    total = sum(value for _key, value in items)
    return items[0][0], _safe_div(items[0][1], total)


def _remaining_bucket(value: Any) -> str:
    remaining = _as_int(value, 0)
    if remaining <= 5:
        return "r<=5"
    if remaining <= 15:
        return "r<=15"
    if remaining <= 50:
        return "r<=50"
    return "r>50"


def _pool_bucket(value: Any) -> str:
    size = _as_int(value, 0)
    if size <= 0:
        return "pool0"
    if size == 1:
        return "pool1"
    if size <= 3:
        return "pool2_3"
    return "pool4p"


def _candidate_keys(example: Mapping[str, Any]) -> list[str]:
    features = dict(example.get("features") or {})
    source = str(features.get("source", "") or "unknown")
    branch = str(features.get("branch_direction", "") or "none")
    has_site = "site" if _as_bool(features.get("has_site")) else "no_site"
    advanced = "adv" if _as_bool(features.get("was_advanced")) else "not_adv"
    queued = "queued" if _as_bool(features.get("was_queued")) else "not_queued"
    return [
        f"source={source}|branch={branch}|{has_site}|{advanced}|{queued}",
        f"source={source}|branch={branch}",
        f"source={source}",
        has_site,
        "__global__",
    ]


def _risk_keys(example: Mapping[str, Any]) -> list[str]:
    features = dict(example.get("features") or {})
    no_target = str(features.get("no_target_disposition", "") or "none")
    selected = "selected" if _as_bool(features.get("selected_target_present")) else "unselected"
    selected_empty = "selected_empty" if _as_bool(features.get("selected_target_empty_with_candidate_pool")) else "selected_ok"
    remaining = _remaining_bucket(features.get("remaining_actions"))
    pool = _pool_bucket(features.get("candidate_pool_size"))
    return [
        f"{remaining}|{pool}|{selected}|{selected_empty}|no_target={no_target}",
        f"{remaining}|{pool}|{selected_empty}",
        f"{remaining}|{pool}",
        "__global__",
    ]


def _mode_keys(example: Mapping[str, Any]) -> list[str]:
    features = dict(example.get("features") or {})
    selected = "selected" if _as_bool(features.get("selected_target_present")) else "unselected"
    selected_empty = "selected_empty" if _as_bool(features.get("selected_target_empty_with_candidate_pool")) else "selected_ok"
    recovery = "recovery_window" if _as_bool(features.get("recovery_in_window")) else "no_recovery_window"
    no_target = str(features.get("no_target_disposition", "") or "none")
    remaining = _remaining_bucket(features.get("remaining_actions"))
    pool = _pool_bucket(features.get("candidate_pool_size"))
    return [
        f"{remaining}|{pool}|{selected}|{selected_empty}|{recovery}|no_target={no_target}",
        f"{remaining}|{pool}|{selected_empty}|{recovery}",
        f"{remaining}|{pool}|{selected}",
        "__global__",
    ]


def _action_keys(example: Mapping[str, Any]) -> list[str]:
    features = dict(example.get("features") or {})
    mode = str(features.get("mode", "") or "unknown")
    family = str(features.get("candidate_family_top", "") or "none")
    transition = str(features.get("candidate_transition_family_top", "") or "none")
    selected = "selected" if _as_bool(features.get("selected_target_present")) else "unselected"
    remaining = _remaining_bucket(features.get("remaining_actions"))
    return [
        f"mode={mode}|family={family}|transition={transition}|{selected}|{remaining}",
        f"mode={mode}|family={family}|{selected}",
        f"mode={mode}",
        "__global__",
    ]


def _label_value(example: Mapping[str, Any]) -> float:
    return _as_float((example.get("label") or {}).get("candidate_value"), 0.0)


def _group_candidate_examples(examples: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in examples:
        item = dict(raw)
        group_id = str(item.get("group_id", "") or item.get("run_id", "") or "default")
        groups[group_id].append(item)
    return groups


def collect_policy_examples(
    roots: Iterable[Path],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Collect P2-P5 examples from one or more modeling run roots."""
    run_dirs = find_modeling_run_dirs([Path(root) for root in roots], limit=limit)
    candidate_examples: list[dict[str, Any]] = []
    risk_examples: list[dict[str, Any]] = []
    mode_examples: list[dict[str, Any]] = []
    action_examples: list[dict[str, Any]] = []
    games: Counter[str] = Counter()

    for run_dir in run_dirs:
        bundle = build_modeling_logs_from_run_dir(run_dir)
        modes = build_mode_selection_examples_from_bundle(bundle)
        actions = build_action_prior_examples_from_bundle(
            bundle,
            mode_selection_examples=modes,
        )
        steps = list(bundle.get("step_logs") or [])
        game_id = ""
        if steps:
            game_id = str((steps[0].get("observation") or {}).get("game_id", "") or "")
        if not game_id:
            game_id = str((bundle.get("trajectory_summary") or {}).get("run_id", "") or "").split(":", 1)[0]
        games[game_id or "unknown"] += 1
        candidate_examples.extend(dict(row, _run_dir=str(run_dir), _game_id=game_id) for row in bundle.get("candidate_ranking_examples") or [])
        risk_examples.extend(dict(row, _run_dir=str(run_dir), _game_id=game_id) for row in bundle.get("risk_prediction_examples") or [])
        mode_examples.extend(dict(row, _run_dir=str(run_dir), _game_id=game_id) for row in modes)
        action_examples.extend(dict(row, _run_dir=str(run_dir), _game_id=game_id) for row in actions)

    return {
        "run_dirs": [str(path) for path in run_dirs],
        "game_counts": dict(sorted(games.items())),
        "candidate_examples": candidate_examples,
        "risk_examples": risk_examples,
        "mode_examples": mode_examples,
        "action_examples": action_examples,
    }


def train_policy_bundle(
    *,
    candidate_examples: Iterable[Mapping[str, Any]],
    risk_examples: Iterable[Mapping[str, Any]],
    mode_examples: Iterable[Mapping[str, Any]],
    action_examples: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = [dict(row) for row in candidate_examples]
    risks = [dict(row) for row in risk_examples]
    modes = [dict(row) for row in mode_examples]
    actions = [dict(row) for row in action_examples]

    candidate_values_by_key: dict[str, list[float]] = defaultdict(list)
    for example in candidates:
        value = _label_value(example)
        for key in _candidate_keys(example):
            candidate_values_by_key[key].append(value)

    risk_values_by_target_key: dict[str, dict[str, list[float]]] = {
        target: defaultdict(list) for target in BINARY_RISK_TARGETS
    }
    recovery_values_by_key: dict[str, list[float]] = defaultdict(list)
    for example in risks:
        label = dict(example.get("label") or {})
        for key in _risk_keys(example):
            recovery_values_by_key[key].append(_as_float(label.get("expected_recovery_steps"), 0.0))
            for target in BINARY_RISK_TARGETS:
                risk_values_by_target_key[target][key].append(
                    1.0 if _as_bool(label.get(target)) else 0.0
                )

    mode_values_by_key: dict[str, list[str]] = defaultdict(list)
    for example in modes:
        mode = str((example.get("label") or {}).get("selected_mode", "") or "")
        for key in _mode_keys(example):
            mode_values_by_key[key].append(mode)

    action_family_by_key: dict[str, list[str]] = defaultdict(list)
    action_name_by_key: dict[str, list[str]] = defaultdict(list)
    for example in actions:
        label = dict(example.get("label") or {})
        family = str(label.get("action_family", "") or "")
        action = normalize_action_name(str(label.get("action_name", "") or ""))
        for key in _action_keys(example):
            action_family_by_key[key].append(family)
            action_name_by_key[key].append(action)

    return _json_safe(
        {
            "schema_version": POLICY_BUNDLE_SCHEMA_VERSION,
            "metadata": dict(metadata or {}),
            "train_counts": {
                "candidate_examples": len(candidates),
                "risk_examples": len(risks),
                "mode_examples": len(modes),
                "action_examples": len(actions),
            },
            "candidate_model": {
                "buckets": {
                    key: _mean_bucket(values)
                    for key, values in sorted(candidate_values_by_key.items())
                }
            },
            "risk_model": {
                "targets": {
                    target: {
                        "buckets": {
                            key: _mean_bucket(values)
                            for key, values in sorted(by_key.items())
                        }
                    }
                    for target, by_key in risk_values_by_target_key.items()
                },
                "expected_recovery_steps": {
                    "buckets": {
                        key: _mean_bucket(values)
                        for key, values in sorted(recovery_values_by_key.items())
                    }
                },
            },
            "mode_model": {
                "buckets": {
                    key: _count_bucket(values)
                    for key, values in sorted(mode_values_by_key.items())
                }
            },
            "action_model": {
                "family_buckets": {
                    key: _count_bucket(values)
                    for key, values in sorted(action_family_by_key.items())
                },
                "action_buckets": {
                    key: _count_bucket(values)
                    for key, values in sorted(action_name_by_key.items())
                },
            },
        }
    )


def train_policy_bundle_from_roots(
    roots: Iterable[Path],
    *,
    limit: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    examples = collect_policy_examples(roots, limit=limit)
    bundle = train_policy_bundle(
        candidate_examples=examples["candidate_examples"],
        risk_examples=examples["risk_examples"],
        mode_examples=examples["mode_examples"],
        action_examples=examples["action_examples"],
        metadata={
            **dict(metadata or {}),
            "run_dirs": examples["run_dirs"],
            "game_counts": examples["game_counts"],
        },
    )
    return bundle, examples


def _bucket_mean(model: Mapping[str, Any], keys: list[str], *, default: float = 0.0) -> float:
    buckets = dict(model.get("buckets") or {})
    for key in keys:
        bucket = dict(buckets.get(key) or {})
        if _as_int(bucket.get("count"), 0) > 0:
            return _as_float(bucket.get("mean"), default)
    return default


def predict_candidate_score(policy_bundle: Mapping[str, Any], example: Mapping[str, Any]) -> float:
    model = dict(policy_bundle.get("candidate_model") or {})
    score = _bucket_mean(model, _candidate_keys(example), default=0.0)
    return round(score, 6)


def predict_risk(policy_bundle: Mapping[str, Any], example: Mapping[str, Any]) -> dict[str, float]:
    risk_model = dict(policy_bundle.get("risk_model") or {})
    targets = dict(risk_model.get("targets") or {})
    out: dict[str, float] = {}
    keys = _risk_keys(example)
    for target in BINARY_RISK_TARGETS:
        out[target] = round(
            _bucket_mean(dict(targets.get(target) or {}), keys, default=0.0),
            6,
        )
    out["expected_recovery_steps"] = round(
        _bucket_mean(dict(risk_model.get("expected_recovery_steps") or {}), keys, default=0.0),
        6,
    )
    return out


def predict_mode(policy_bundle: Mapping[str, Any], example: Mapping[str, Any]) -> dict[str, Any]:
    buckets = dict((policy_bundle.get("mode_model") or {}).get("buckets") or {})
    selected = ""
    confidence = 0.0
    scores: dict[str, float] = {}
    for key in _mode_keys(example):
        bucket = dict(buckets.get(key) or {})
        counts = dict(bucket.get("counts") or {})
        if _as_int(bucket.get("count"), 0) <= 0:
            continue
        selected, confidence = _top_count(counts)
        total = max(1, sum(_as_int(value, 0) for value in counts.values()))
        scores = {
            str(mode): round(_as_int(count, 0) / total, 6)
            for mode, count in sorted(counts.items())
        }
        break
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {
        "selected_mode": selected,
        "confidence": confidence,
        "scores": scores,
        "top2_modes": [mode for mode, _score in ranked[:2]],
    }


def predict_action(policy_bundle: Mapping[str, Any], example: Mapping[str, Any]) -> dict[str, Any]:
    action_model = dict(policy_bundle.get("action_model") or {})
    family_buckets = dict(action_model.get("family_buckets") or {})
    action_buckets = dict(action_model.get("action_buckets") or {})
    family = ""
    action = ""
    confidence = 0.0
    for key in _action_keys(example):
        family_bucket = dict(family_buckets.get(key) or {})
        action_bucket = dict(action_buckets.get(key) or {})
        if _as_int(family_bucket.get("count"), 0) > 0:
            family, confidence = _top_count(family_bucket.get("counts") or {})
        if _as_int(action_bucket.get("count"), 0) > 0:
            action, _action_conf = _top_count(action_bucket.get("counts") or {})
        if family or action:
            break
    features = dict(example.get("features") or {})
    target = str(features.get("selected_target_id", "") or "") or str(
        features.get("top_candidate_id", "") or ""
    )
    return {
        "action_family": family,
        "action_name": normalize_action_name(action),
        "target_candidate_id": target,
        "confidence": confidence,
    }


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
    return _dcg(values) / ideal_dcg if ideal_dcg > 0 else 0.0


def _pairwise_accuracy(ranked: list[Mapping[str, Any]]) -> tuple[int, int]:
    correct = 0
    total = 0
    for idx, left in enumerate(ranked):
        for right in ranked[idx + 1 :]:
            left_value = _label_value(left)
            right_value = _label_value(right)
            if abs(left_value - right_value) <= 1e-9:
                continue
            total += 1
            if left_value > right_value:
                correct += 1
    return correct, total


def _candidate_eval(policy_bundle: Mapping[str, Any], examples: list[Mapping[str, Any]]) -> dict[str, Any]:
    groups = _group_candidate_examples(examples)
    ndcg_sum = 0.0
    pairwise_correct = 0
    pairwise_total = 0
    top1_promotion = 0
    group_count = 0
    for group in groups.values():
        ranked = sorted(
            [dict(row, predicted_score=predict_candidate_score(policy_bundle, row)) for row in group],
            key=lambda item: (-_as_float(item.get("predicted_score"), 0.0), str(item.get("candidate_id", "") or "")),
        )
        if not ranked:
            continue
        group_count += 1
        ndcg_sum += _ndcg(ranked)
        correct, total = _pairwise_accuracy(ranked)
        pairwise_correct += correct
        pairwise_total += total
        if _as_bool((ranked[0].get("label") or {}).get("promotion")):
            top1_promotion += 1
    return {
        "schema_version": "eca_policy_candidate_eval.v1",
        "group_count": group_count,
        "example_count": len(examples),
        "mean_ndcg": round(ndcg_sum / group_count, 6) if group_count else 0.0,
        "pairwise_accuracy": _safe_div(pairwise_correct, pairwise_total),
        "pairwise_comparable": pairwise_total,
        "top1_promotion_rate": _safe_div(top1_promotion, group_count),
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


def _risk_eval(policy_bundle: Mapping[str, Any], examples: list[Mapping[str, Any]]) -> dict[str, Any]:
    predictions = [predict_risk(policy_bundle, row) for row in examples]
    metrics: list[dict[str, Any]] = []
    for target in BINARY_RISK_TARGETS:
        labels = [_as_bool((row.get("label") or {}).get(target)) for row in examples]
        scores = [_as_float(pred.get(target), 0.0) for pred in predictions]
        positives = sum(1 for label in labels if label)
        negatives = len(labels) - positives
        brier = (
            sum((score - (1.0 if label else 0.0)) ** 2 for score, label in zip(scores, labels))
            / len(labels)
            if labels
            else 0.0
        )
        auc = _auc(scores, labels)
        metrics.append(
            {
                "target": target,
                "positive_count": positives,
                "negative_count": negatives,
                "auroc": round(auc, 6) if auc is not None else None,
                "brier": round(brier, 6),
            }
        )
    mae = (
        sum(
            abs(
                _as_float(pred.get("expected_recovery_steps"), 0.0)
                - _as_float((row.get("label") or {}).get("expected_recovery_steps"), 0.0)
            )
            for pred, row in zip(predictions, examples)
        )
        / len(examples)
        if examples
        else 0.0
    )
    return {
        "schema_version": "eca_policy_risk_eval.v1",
        "example_count": len(examples),
        "target_metrics": metrics,
        "expected_recovery_steps_mae": round(mae, 6),
    }


def _mode_eval(policy_bundle: Mapping[str, Any], examples: list[Mapping[str, Any]]) -> dict[str, Any]:
    correct = 0
    top2 = 0
    for row in examples:
        gold = str((row.get("label") or {}).get("selected_mode", "") or "")
        pred = predict_mode(policy_bundle, row)
        if pred.get("selected_mode") == gold:
            correct += 1
        if gold in set(pred.get("top2_modes") or []):
            top2 += 1
    return {
        "schema_version": "eca_policy_mode_eval.v1",
        "example_count": len(examples),
        "accuracy": _safe_div(correct, len(examples)),
        "top2_accuracy": _safe_div(top2, len(examples)),
    }


def _action_eval(policy_bundle: Mapping[str, Any], examples: list[Mapping[str, Any]]) -> dict[str, Any]:
    family_total = family_correct = 0
    action_total = action_correct = 0
    target_total = target_correct = 0
    for row in examples:
        label = dict(row.get("label") or {})
        pred = predict_action(policy_bundle, row)
        gold_family = str(label.get("action_family", "") or "")
        if gold_family and gold_family != "unknown":
            family_total += 1
            if str(pred.get("action_family", "") or "") == gold_family:
                family_correct += 1
        gold_action = normalize_action_name(str(label.get("action_name", "") or ""))
        if gold_action:
            action_total += 1
            if normalize_action_name(str(pred.get("action_name", "") or "")) == gold_action:
                action_correct += 1
        gold_target = str(label.get("target_candidate_id", "") or "")
        if gold_target:
            target_total += 1
            if str(pred.get("target_candidate_id", "") or "") == gold_target:
                target_correct += 1
    return {
        "schema_version": "eca_policy_action_eval.v1",
        "example_count": len(examples),
        "family_accuracy": _safe_div(family_correct, family_total),
        "family_total": family_total,
        "action_accuracy": _safe_div(action_correct, action_total),
        "action_total": action_total,
        "target_accuracy": _safe_div(target_correct, target_total),
        "target_total": target_total,
    }


def evaluate_policy_bundle(
    policy_bundle: Mapping[str, Any],
    *,
    candidate_examples: Iterable[Mapping[str, Any]],
    risk_examples: Iterable[Mapping[str, Any]],
    mode_examples: Iterable[Mapping[str, Any]],
    action_examples: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = [dict(row) for row in candidate_examples]
    risks = [dict(row) for row in risk_examples]
    modes = [dict(row) for row in mode_examples]
    actions = [dict(row) for row in action_examples]
    return {
        "schema_version": "eca_policy_bundle_eval.v1",
        "train_counts": dict(policy_bundle.get("train_counts") or {}),
        "eval_counts": {
            "candidate_examples": len(candidates),
            "risk_examples": len(risks),
            "mode_examples": len(modes),
            "action_examples": len(actions),
        },
        "candidate_ranker": _candidate_eval(policy_bundle, candidates),
        "risk_predictor": _risk_eval(policy_bundle, risks),
        "mode_selector": _mode_eval(policy_bundle, modes),
        "action_prior": _action_eval(policy_bundle, actions),
    }


def build_runtime_policy_packet(
    policy_bundle: Mapping[str, Any],
    *,
    candidate_examples: Iterable[Mapping[str, Any]] | None = None,
    risk_example: Mapping[str, Any] | None = None,
    mode_example: Mapping[str, Any] | None = None,
    action_example: Mapping[str, Any] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Build a safe advisory packet for future runtime integration.

    The packet carries learned predictions only.  It never executes an action
    and defaults to dry-run so it can be logged beside the current controller.
    """
    candidate_rows = [dict(row) for row in candidate_examples or []]
    ranked_candidates = sorted(
        [
            {
                "candidate_id": str(row.get("candidate_id", "") or ""),
                "predicted_score": predict_candidate_score(policy_bundle, row),
            }
            for row in candidate_rows
        ],
        key=lambda item: (-_as_float(item.get("predicted_score"), 0.0), item["candidate_id"]),
    )
    risk = predict_risk(policy_bundle, risk_example or {}) if risk_example else {}
    mode = predict_mode(policy_bundle, mode_example or {}) if mode_example else {}
    action = predict_action(policy_bundle, action_example or {}) if action_example else {}
    return _json_safe(
        {
            "schema_version": "eca_runtime_policy_packet.v1",
            "dry_run": bool(dry_run),
            "policy_bundle_schema_version": str(policy_bundle.get("schema_version", "") or ""),
            "top_candidates": ranked_candidates[:5],
            "risk": risk,
            "mode_decision": mode,
            "action_prior": action,
            "execute_allowed": False,
        }
    )


def write_policy_bundle(path: Path, policy_bundle: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(policy_bundle), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def load_policy_bundle(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
