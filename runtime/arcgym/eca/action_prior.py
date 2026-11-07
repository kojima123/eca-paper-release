"""Offline action-prior baseline for ECA modeling logs.

This is the final v0.2 modeling slice before runtime integration:

    mode + candidate ranking -> action prior

The module builds examples from StepLog v2 plus mode-selection examples and
evaluates a simple heuristic prior.  It does not execute actions.
"""
from __future__ import annotations

from collections import Counter
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from arcgym.eca.candidate_ranker import rank_candidate_examples


ACTION_PRIOR_SCHEMA_VERSION = "action_prior_example.v2"


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


def normalize_action_name(action_name: str) -> str:
    text = str(action_name or "").strip().upper()
    if not text:
        return ""
    match = re.search(r"ACTION\s*([0-9]+)", text)
    if match:
        return f"ACTION{int(match.group(1))}"
    if text.isdigit():
        return f"ACTION{int(text)}"
    if text in {"RESET", "NOOP", "WAIT"}:
        return text
    return text


def action_family(action_name: str) -> str:
    action = normalize_action_name(action_name)
    if action in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
        return "move"
    if action == "ACTION5":
        return "wait"
    if action == "ACTION6":
        return "click"
    if action == "RESET":
        return "reset"
    if action == "NOOP":
        return "noop"
    return "unknown" if not action else action.lower()


def _label_action_from_step(step_log: Mapping[str, Any]) -> str:
    action = dict(step_log.get("action") or {})
    action_name = normalize_action_name(str(action.get("name", "") or ""))
    if action_name:
        return action_name
    preview = action.get("preview") or []
    if isinstance(preview, list) and preview:
        return normalize_action_name(str(preview[0] or ""))
    return ""


def _selected_target_id(step_log: Mapping[str, Any]) -> str:
    for item in step_log.get("evidence_items") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("kind", "") or "") != "selected_target":
            continue
        return str(item.get("candidate_id", "") or "")
    return ""


def _top_histogram_key(histogram: Mapping[str, Any]) -> str:
    if not isinstance(histogram, Mapping) or not histogram:
        return ""
    ranked = sorted(
        ((str(key or ""), _as_float(value, 0.0)) for key, value in histogram.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[0][0]


def _top_candidate_features(candidate_examples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    ranked = rank_candidate_examples(candidate_examples, profile="deployable")
    if not ranked:
        return {
            "top_candidate_id": "",
            "top_candidate_score": 0.0,
            "top_candidate_source": "",
            "top_candidate_has_site": False,
        }
    top = ranked[0]
    features = dict(top.get("features") or {})
    return {
        "top_candidate_id": str(top.get("candidate_id", "") or ""),
        "top_candidate_score": _as_float(top.get("ranker_score"), 0.0),
        "top_candidate_source": str(features.get("source", "") or ""),
        "top_candidate_has_site": _as_bool(features.get("has_site")),
    }


def _mode_by_step(mode_examples: Iterable[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for example in mode_examples:
        step_id = _as_int(example.get("step_id"), -1)
        if step_id >= 0:
            out[step_id] = dict(example)
    return out


def _features_from_step(
    step_log: Mapping[str, Any],
    *,
    mode_example: Mapping[str, Any] | None,
    top_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    observation = dict(step_log.get("observation") or {})
    candidate_pool = dict(step_log.get("candidate_pool") or {})
    hindsight = dict(step_log.get("hindsight_labels") or {})
    mode_label = dict((mode_example or {}).get("label") or {})
    mode_features = dict((mode_example or {}).get("features") or {})
    family_hist = dict(candidate_pool.get("family_histogram") or {})
    transition_family_hist = dict(candidate_pool.get("transition_family_histogram") or {})
    return {
        "step_id": _as_int(step_log.get("step_id"), 0),
        "mode": str(mode_label.get("selected_mode", "") or ""),
        "mode_reason": str(mode_label.get("reason", "") or ""),
        "remaining_actions": _as_int(observation.get("remaining_actions"), 0),
        "selected_target_present": _as_bool(hindsight.get("selected_target_present")),
        "selected_target_id": _selected_target_id(step_log),
        "candidate_pool_size": _as_int(candidate_pool.get("size"), 0),
        "candidate_family_top": _top_histogram_key(family_hist),
        "candidate_transition_family_top": _top_histogram_key(transition_family_hist),
        "candidate_escape_pending": _as_bool(candidate_pool.get("escape_pending")),
        "live_escape_candidate_count": _as_int(candidate_pool.get("live_escape_candidate_count"), 0),
        "no_change_streak": _as_int((step_log.get("runtime_packet") or {}).get("no_change_streak"), 0),
        "top_candidate_id": str(top_candidate.get("top_candidate_id", "") or ""),
        "top_candidate_score": _as_float(top_candidate.get("top_candidate_score"), 0.0),
        "top_candidate_source": str(top_candidate.get("top_candidate_source", "") or ""),
        "top_candidate_has_site": _as_bool(top_candidate.get("top_candidate_has_site")),
        "mode_top_candidate_id": str(mode_features.get("top_candidate_id", "") or ""),
    }


def build_action_prior_examples(
    *,
    step_logs: Iterable[Mapping[str, Any]],
    mode_selection_examples: Iterable[Mapping[str, Any]],
    candidate_ranking_examples: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    steps = [dict(row) for row in step_logs]
    mode_map = _mode_by_step(mode_selection_examples)
    top_candidate = _top_candidate_features(candidate_ranking_examples or [])
    run_id = str(steps[0].get("run_id", "") or "") if steps else ""
    examples: list[dict[str, Any]] = []
    for step in steps:
        step_id = _as_int(step.get("step_id"), 0)
        features = _features_from_step(
            step,
            mode_example=mode_map.get(step_id),
            top_candidate=top_candidate,
        )
        action = _label_action_from_step(step)
        examples.append(
            _json_safe(
                {
                    "schema_version": ACTION_PRIOR_SCHEMA_VERSION,
                    "run_id": run_id,
                    "group_id": f"{run_id}:action_prior",
                    "step_id": step_id,
                    "features": features,
                    "label": {
                        "action_name": action,
                        "action_family": action_family(action),
                        "target_candidate_id": _selected_target_id(step),
                    },
                }
            )
        )
    return examples


def build_action_prior_examples_from_bundle(
    bundle: Mapping[str, Any],
    *,
    mode_selection_examples: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return build_action_prior_examples(
        step_logs=bundle.get("step_logs") or [],
        mode_selection_examples=mode_selection_examples,
        candidate_ranking_examples=bundle.get("candidate_ranking_examples") or [],
    )


def load_action_prior_examples(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        path = path / "eca_action_prior_examples.jsonl"
    return _read_jsonl(path)


def predict_action_prior(example: Mapping[str, Any]) -> dict[str, Any]:
    """Predict an action prior without reading the label."""
    features = dict(example.get("features") or {})
    mode = str(features.get("mode", "") or "")
    family_top = str(features.get("candidate_family_top", "") or "").lower()
    transition_top = str(features.get("candidate_transition_family_top", "") or "").lower()
    top_source = str(features.get("top_candidate_source", "") or "").lower()
    selected_present = _as_bool(features.get("selected_target_present"))

    family = ""
    if family_top in {"click", "move", "wait", "reset"}:
        family = family_top
    elif transition_top in {"click", "move", "wait", "reset"}:
        family = transition_top
    elif mode in {"explore", "recover_branch"}:
        family = "move"
    elif mode in {"verify", "reclaim", "rotate"} and selected_present:
        family = "click"
    elif top_source in {"pending_contact", "symmetric_seed"}:
        family = "move"
    else:
        family = "click" if selected_present else "move"

    action_by_family = {
        "move": "ACTION1",
        "click": "ACTION6",
        "wait": "ACTION5",
        "reset": "RESET",
        "noop": "NOOP",
    }
    target = ""
    if mode in {"verify", "reclaim", "hedge", "rotate", "budget_conserve"}:
        target = str(features.get("selected_target_id", "") or "") or str(
            features.get("top_candidate_id", "") or ""
        )
    elif mode == "recover_branch":
        target = str(features.get("top_candidate_id", "") or "")
    return {
        "action_family": family,
        "action_name": action_by_family.get(family, ""),
        "target_candidate_id": target,
    }


def evaluate_action_prior(examples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in examples]
    family_total = 0
    family_correct = 0
    action_total = 0
    action_correct = 0
    target_total = 0
    target_correct = 0
    label_families: Counter[str] = Counter()
    pred_families: Counter[str] = Counter()
    for row in rows:
        label = dict(row.get("label") or {})
        pred = predict_action_prior(row)
        gold_family = str(label.get("action_family", "") or "")
        pred_family = str(pred.get("action_family", "") or "")
        if gold_family and gold_family != "unknown":
            family_total += 1
            label_families[gold_family] += 1
            pred_families[pred_family] += 1
            if pred_family == gold_family:
                family_correct += 1
        gold_action = normalize_action_name(str(label.get("action_name", "") or ""))
        pred_action = normalize_action_name(str(pred.get("action_name", "") or ""))
        if gold_action:
            action_total += 1
            if pred_action == gold_action:
                action_correct += 1
        gold_target = str(label.get("target_candidate_id", "") or "")
        pred_target = str(pred.get("target_candidate_id", "") or "")
        if gold_target:
            target_total += 1
            if pred_target == gold_target:
                target_correct += 1
    return {
        "schema_version": "action_prior_eval.v1",
        "example_count": len(rows),
        "family_accuracy": round(family_correct / family_total, 6) if family_total else 0.0,
        "family_total": family_total,
        "action_accuracy": round(action_correct / action_total, 6) if action_total else 0.0,
        "action_total": action_total,
        "target_accuracy": round(target_correct / target_total, 6) if target_total else 0.0,
        "target_total": target_total,
        "label_family_counts": dict(sorted(label_families.items())),
        "prediction_family_counts": dict(sorted(pred_families.items())),
    }


def write_action_prior_examples(path: Path, examples: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "eca_action_prior_examples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")
    return path
