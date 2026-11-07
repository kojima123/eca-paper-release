"""Offline mode-selection baseline for ECA modeling logs.

This module creates and evaluates examples for the v0.2 ControlIR mode
selector task:

    state + risk + candidate value -> mode

It is intentionally offline-only.  The baseline selector does not control the
runtime; it gives us a measurable target before replacing hand-authored control
rules with learned predictors.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from arcgym.eca.candidate_ranker import rank_candidate_examples


CANONICAL_MODES = (
    "explore",
    "verify",
    "hedge",
    "reclaim",
    "rotate",
    "recover_branch",
    "budget_conserve",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        text = line.strip().lstrip("\ufeff")
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _branch_recovery_step(hindsight_labels: Mapping[str, Any]) -> int | None:
    steps: list[int] = []
    for branch in hindsight_labels.get("branch_labels") or []:
        if not isinstance(branch, Mapping):
            continue
        if not _as_bool(branch.get("recovery_success")):
            continue
        step = branch.get("recovery_trigger_step")
        if step is not None:
            steps.append(_as_int(step, 0))
    return min(steps) if steps else None


def _risk_by_step(risk_examples: Iterable[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for example in risk_examples:
        step_id = _as_int(example.get("step_id"), -1)
        if step_id >= 0:
            out[step_id] = dict(example)
    return out


def _top_candidate_features(
    candidate_examples: Iterable[Mapping[str, Any]],
    *,
    profile: str = "deployable",
) -> dict[str, Any]:
    ranked = rank_candidate_examples(candidate_examples, profile=profile)
    if not ranked:
        return {
            "top_candidate_id": "",
            "top_candidate_score": 0.0,
            "top_candidate_source": "",
            "top_candidate_has_site": False,
            "positive_candidate_count": 0,
        }
    top = ranked[0]
    features = dict(top.get("features") or {})
    positive_count = sum(
        1
        for item in ranked
        if _as_float((item.get("label") or {}).get("candidate_value"), 0.0) > 0.0
    )
    return {
        "top_candidate_id": str(top.get("candidate_id", "") or ""),
        "top_candidate_score": _as_float(top.get("ranker_score"), 0.0),
        "top_candidate_source": str(features.get("source", "") or ""),
        "top_candidate_has_site": _as_bool(features.get("has_site")),
        "positive_candidate_count": positive_count,
    }


def _features_from_step(
    step_log: Mapping[str, Any],
    *,
    risk_example: Mapping[str, Any] | None,
    top_candidate: Mapping[str, Any],
    recovery_step: int | None,
) -> dict[str, Any]:
    observation = dict(step_log.get("observation") or {})
    runtime = dict(step_log.get("runtime_packet") or {})
    candidate_pool = dict(step_log.get("candidate_pool") or {})
    hindsight = dict(step_log.get("hindsight_labels") or {})
    control_ir = dict(step_log.get("control_ir") or {})
    risk_label = dict((risk_example or {}).get("label") or {})
    step_id = _as_int(step_log.get("step_id"), 0)
    remaining = _as_int(observation.get("remaining_actions"), 0)
    recovery_in_window = False
    if recovery_step is not None:
        recovery_in_window = bool(step_id <= recovery_step <= step_id + 8)
    return {
        "step_id": step_id,
        "remaining_actions": remaining,
        "level": _as_int(observation.get("level"), 0),
        "score": _as_float(observation.get("score"), 0.0),
        "selected_target_present": _as_bool(hindsight.get("selected_target_present")),
        "progress_within_5": _as_bool(hindsight.get("progress_within_5")),
        "no_change_streak": _as_int(runtime.get("no_change_streak"), 0),
        "repeated_state_count": _as_int(runtime.get("repeated_state_count"), 0),
        "queue_exhaustions": _as_int(runtime.get("queue_exhaustions"), 0),
        "candidate_pool_size": _as_int(candidate_pool.get("size"), 0),
        "candidate_dead_tail_ratio": _as_float(candidate_pool.get("dead_tail_ratio"), 0.0),
        "live_escape_candidate_count": _as_int(
            candidate_pool.get("live_escape_candidate_count"), 0
        ),
        "selected_target_empty_with_candidate_pool": _as_bool(
            candidate_pool.get("selected_target_empty_with_candidate_pool")
        ),
        "candidate_escape_pending": _as_bool(candidate_pool.get("escape_pending")),
        "no_target_disposition": str(candidate_pool.get("no_target_disposition", "") or ""),
        "control_ir_live_ticket_count": _as_int(control_ir.get("control_ir_live_ticket_count"), 0),
        "risk_queue_exhaustion_5": _as_bool(risk_label.get("p_queue_exhaustion_5")),
        "risk_no_target_collapse_5": _as_bool(risk_label.get("p_no_target_collapse_5")),
        "risk_selected_target_empty_5": _as_bool(risk_label.get("p_selected_target_empty_5")),
        "risk_budget_exhaustion_before_test": _as_bool(
            risk_label.get("p_budget_exhaustion_before_test")
        ),
        "risk_progress_5": _as_bool(risk_label.get("p_progress_5")),
        "recovery_in_window": recovery_in_window,
        **dict(top_candidate),
    }


def _derive_mode_label(features: Mapping[str, Any]) -> tuple[str, str]:
    remaining = _as_int(features.get("remaining_actions"), 0)
    pool_size = _as_int(features.get("candidate_pool_size"), 0)
    no_change = _as_int(features.get("no_change_streak"), 0)
    no_target_disposition = str(features.get("no_target_disposition", "") or "")
    selected_present = _as_bool(features.get("selected_target_present"))
    selected_empty = _as_bool(features.get("selected_target_empty_with_candidate_pool"))
    live_escape_count = _as_int(features.get("live_escape_candidate_count"), 0)

    if _as_bool(features.get("risk_budget_exhaustion_before_test")) or (
        remaining <= 5 and pool_size > 0
    ):
        return "budget_conserve", "low_budget_with_candidates"
    if _as_bool(features.get("recovery_in_window")):
        return "recover_branch", "observed_branch_recovery_window"
    if (
        selected_empty
        or _as_bool(features.get("risk_no_target_collapse_5"))
        or no_target_disposition in {"preterminal_dead_revisit", "hard_terminal", "collapse"}
    ):
        return "reclaim", "target_collapse_or_dead_revisit"
    if no_change >= 5 and pool_size > 1:
        return "rotate", "stale_target_with_alternatives"
    if pool_size > 1 and (live_escape_count > 0 or _as_bool(features.get("candidate_escape_pending"))):
        return "hedge", "alternatives_with_escape_signal"
    if selected_present:
        return "verify", "selected_target_available"
    return "explore", "no_selected_target"


def build_mode_selection_examples(
    *,
    step_logs: Iterable[Mapping[str, Any]],
    candidate_ranking_examples: Iterable[Mapping[str, Any]] | None = None,
    hindsight_labels: Mapping[str, Any] | None = None,
    risk_prediction_examples: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    steps = [dict(row) for row in step_logs]
    candidate_examples = [dict(row) for row in candidate_ranking_examples or []]
    risk_map = _risk_by_step(risk_prediction_examples or [])
    recovery_step = _branch_recovery_step(hindsight_labels or {})
    top_candidate = _top_candidate_features(candidate_examples)
    examples: list[dict[str, Any]] = []
    run_id = str(steps[0].get("run_id", "") or "") if steps else ""
    for step in steps:
        features = _features_from_step(
            step,
            risk_example=risk_map.get(_as_int(step.get("step_id"), 0)),
            top_candidate=top_candidate,
            recovery_step=recovery_step,
        )
        mode, reason = _derive_mode_label(features)
        examples.append(
            _json_safe(
                {
                    "schema_version": "mode_selection_example.v2",
                    "run_id": run_id,
                    "group_id": f"{run_id}:mode",
                    "step_id": features["step_id"],
                    "features": features,
                    "label": {
                        "selected_mode": mode,
                        "reason": reason,
                    },
                }
            )
        )
    return examples


def build_mode_selection_examples_from_bundle(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    return build_mode_selection_examples(
        step_logs=bundle.get("step_logs") or [],
        candidate_ranking_examples=bundle.get("candidate_ranking_examples") or [],
        hindsight_labels=bundle.get("hindsight_labels") or {},
        risk_prediction_examples=bundle.get("risk_prediction_examples") or [],
    )


def load_mode_selection_examples(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        path = path / "eca_mode_selection_examples.jsonl"
    return _read_jsonl(path)


def score_modes(example: Mapping[str, Any]) -> dict[str, float]:
    features = dict(example.get("features") or {})
    remaining = _as_int(features.get("remaining_actions"), 0)
    pool_size = _as_int(features.get("candidate_pool_size"), 0)
    no_change = _as_int(features.get("no_change_streak"), 0)
    no_target = str(features.get("no_target_disposition", "") or "")
    selected_present = _as_bool(features.get("selected_target_present"))
    selected_empty = _as_bool(features.get("selected_target_empty_with_candidate_pool"))
    live_escape_count = _as_int(features.get("live_escape_candidate_count"), 0)
    top_score = _as_float(features.get("top_candidate_score"), 0.0)

    scores = {
        "explore": 0.08,
        "verify": 0.08,
        "hedge": 0.06,
        "reclaim": 0.05,
        "rotate": 0.05,
        "recover_branch": 0.04,
        "budget_conserve": 0.04,
    }
    if remaining <= 5:
        scores["budget_conserve"] += 0.55
    elif remaining <= 15:
        scores["budget_conserve"] += 0.20
    if _as_bool(features.get("recovery_in_window")):
        scores["recover_branch"] += 0.70
    if selected_empty or no_target:
        scores["reclaim"] += 0.58
    if no_change >= 5 and pool_size > 1:
        scores["rotate"] += 0.46
    if pool_size > 1 and (live_escape_count > 0 or _as_bool(features.get("candidate_escape_pending"))):
        scores["hedge"] += 0.40
    if selected_present:
        scores["verify"] += 0.33 + min(0.12, top_score * 0.1)
    else:
        scores["explore"] += 0.28
    if pool_size == 0:
        scores["explore"] += 0.16
    return {mode: round(value, 6) for mode, value in scores.items()}


def select_mode(example: Mapping[str, Any]) -> dict[str, Any]:
    scores = score_modes(example)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {
        "selected_mode": ranked[0][0],
        "confidence": round(ranked[0][1], 6),
        "scores": scores,
        "top2_modes": [mode for mode, _ in ranked[:2]],
    }


def evaluate_mode_selector(examples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in examples]
    correct = 0
    top2 = 0
    label_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    confusion: dict[str, dict[str, int]] = {}
    for row in rows:
        gold = str((row.get("label") or {}).get("selected_mode", "") or "")
        pred = select_mode(row)
        selected = str(pred.get("selected_mode", "") or "")
        label_counts[gold] += 1
        prediction_counts[selected] += 1
        confusion.setdefault(gold, {})
        confusion[gold][selected] = int(confusion[gold].get(selected, 0) or 0) + 1
        if selected == gold:
            correct += 1
        if gold in set(pred.get("top2_modes") or []):
            top2 += 1
    total = len(rows)
    return {
        "schema_version": "mode_selector_eval.v1",
        "example_count": total,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "top2_accuracy": round(top2 / total, 6) if total else 0.0,
        "label_counts": dict(sorted(label_counts.items())),
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "confusion": {key: dict(sorted(value.items())) for key, value in sorted(confusion.items())},
    }


def write_mode_selection_examples(path: Path, examples: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "eca_mode_selection_examples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")
    return path
