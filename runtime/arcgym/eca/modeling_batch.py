"""Batch aggregation for offline ECA modeling pipeline reports.

This stays outside the controller.  It lets us answer whether the v0.2
modeling artefacts have enough coverage across many runs before treating the
heuristic baselines as training targets.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from arcgym.eca.modeling_pipeline import build_modeling_pipeline_report


RUN_SIGNATURE_FILES = (
    "eca_step_metrics.jsonl",
    "eca_anxiety.jsonl",
)

COVERAGE_COUNT_KEYS = (
    "step_count",
    "candidate_trace_count",
    "branch_trace_count",
    "candidate_ranking_example_count",
    "risk_prediction_example_count",
    "mode_selection_example_count",
    "action_prior_example_count",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in {float("inf"), float("-inf")}:
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
    return round(numerator / denominator, 6)


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def _has_run_signature(path: Path) -> bool:
    return path.is_dir() and any((path / name).exists() for name in RUN_SIGNATURE_FILES)


def find_modeling_run_dirs(
    roots: Iterable[Path],
    *,
    limit: int | None = None,
) -> list[Path]:
    """Find run directories containing ECA modeling source logs."""
    found: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> bool:
        if not _has_run_signature(path):
            return False
        key = _path_key(path)
        if key in seen:
            return False
        seen.add(key)
        found.append(path)
        return limit is not None and len(found) >= limit

    for raw_root in roots:
        root = Path(raw_root)
        if root.is_file():
            if root.name in RUN_SIGNATURE_FILES and add(root.parent):
                break
            continue
        if not root.exists():
            continue
        if add(root):
            break
        for signature in RUN_SIGNATURE_FILES:
            for file_path in root.rglob(signature):
                if add(file_path.parent):
                    break
            if limit is not None and len(found) >= limit:
                break
        if limit is not None and len(found) >= limit:
            break
    return sorted(found, key=lambda item: str(item))


def _merge_counts(*mappings: Mapping[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for mapping in mappings:
        for key, value in (mapping or {}).items():
            counts[str(key)] += _as_int(value, 0)
    return dict(sorted(counts.items()))


def _merge_confusion(reports: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    merged: dict[str, Counter[str]] = {}
    for report in reports:
        confusion = dict((report.get("mode_selector") or {}).get("confusion") or {})
        for gold, predictions in confusion.items():
            bucket = merged.setdefault(str(gold), Counter())
            for pred, value in dict(predictions or {}).items():
                bucket[str(pred)] += _as_int(value, 0)
    return {gold: dict(sorted(bucket.items())) for gold, bucket in sorted(merged.items())}


def _gap_flags(report: Mapping[str, Any]) -> dict[str, bool]:
    coverage = dict(report.get("coverage") or {})
    candidate = dict(report.get("candidate_ranker") or {})
    action = dict(report.get("action_prior") or {})
    risk = dict(report.get("risk_predictor") or {})
    return {
        "missing_step_logs": _as_int(coverage.get("step_count"), 0) == 0,
        "missing_candidate_ranking_examples": _as_int(
            coverage.get("candidate_ranking_example_count"), 0
        )
        == 0,
        "missing_risk_prediction_examples": _as_int(
            coverage.get("risk_prediction_example_count"), 0
        )
        == 0,
        "missing_branch_traces": _as_int(coverage.get("branch_trace_count"), 0) == 0,
        "missing_action_labels": (
            _as_int(coverage.get("action_prior_example_count"), 0) > 0
            and _as_int(action.get("family_total"), 0) == 0
        ),
        "no_candidate_pairwise_labels": (
            _as_int(candidate.get("example_count"), 0) > 0
            and _as_int(candidate.get("pairwise_comparable"), 0) == 0
        ),
        "risk_predictor_skipped": bool(risk.get("skipped")),
    }


def _compact_run_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(report.get("trajectory_summary") or {})
    return {
        "run_dir": str(report.get("run_dir", "") or ""),
        "run_id": str(summary.get("run_id", "") or ""),
        "game_id": str(summary.get("game_id", "") or ""),
        "score": _as_float(summary.get("score"), 0.0),
        "highest_level": _as_int(summary.get("highest_level"), 0),
        "coverage": dict(report.get("coverage") or {}),
        "branch_recovery": dict(report.get("branch_recovery") or {}),
        "gaps": _gap_flags(report),
    }


def _aggregate_coverage(reports: list[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {
        key: sum(_as_int((report.get("coverage") or {}).get(key), 0) for report in reports)
        for key in COVERAGE_COUNT_KEYS
    }
    step_count = totals["step_count"]
    mode_count = totals["mode_selection_example_count"]
    action_count = totals["action_prior_example_count"]
    return {
        **totals,
        "overall_mode_step_coverage": _safe_div(mode_count, step_count),
        "overall_action_prior_step_coverage": _safe_div(action_count, step_count),
        "mean_mode_step_coverage": _safe_div(
            sum(_as_float((report.get("coverage") or {}).get("mode_step_coverage")) for report in reports),
            len(reports),
        ),
        "mean_action_prior_step_coverage": _safe_div(
            sum(
                _as_float((report.get("coverage") or {}).get("action_prior_step_coverage"))
                for report in reports
            ),
            len(reports),
        ),
    }


def _aggregate_candidate_ranker(reports: list[Mapping[str, Any]]) -> dict[str, Any]:
    sections = [dict(report.get("candidate_ranker") or {}) for report in reports]
    group_count = sum(_as_int(section.get("group_count"), 0) for section in sections)
    comparable = sum(_as_int(section.get("pairwise_comparable"), 0) for section in sections)
    pairwise_correct = sum(
        _as_float(section.get("pairwise_accuracy"), 0.0)
        * _as_int(section.get("pairwise_comparable"), 0)
        for section in sections
    )
    return {
        "schema_version": "candidate_ranker_batch_eval.v1",
        "group_count": group_count,
        "example_count": sum(_as_int(section.get("example_count"), 0) for section in sections),
        "mean_ndcg": _safe_div(
            sum(
                _as_float(section.get("mean_ndcg"), 0.0)
                * _as_int(section.get("group_count"), 0)
                for section in sections
            ),
            group_count,
        ),
        "pairwise_accuracy": _safe_div(pairwise_correct, comparable),
        "pairwise_comparable": comparable,
        "top1_promotion_rate": _safe_div(
            sum(
                _as_float(section.get("top1_promotion_rate"), 0.0)
                * _as_int(section.get("group_count"), 0)
                for section in sections
            ),
            group_count,
        ),
        "top1_positive_value_rate": _safe_div(
            sum(
                _as_float(section.get("top1_positive_value_rate"), 0.0)
                * _as_int(section.get("group_count"), 0)
                for section in sections
            ),
            group_count,
        ),
    }


def _aggregate_risk_predictor(reports: list[Mapping[str, Any]]) -> dict[str, Any]:
    sections = [
        dict(report.get("risk_predictor") or {})
        for report in reports
        if not (report.get("risk_predictor") or {}).get("skipped")
    ]
    example_count = sum(_as_int(section.get("example_count"), 0) for section in sections)
    target_accum: dict[str, dict[str, float]] = {}
    for section in sections:
        for metric in section.get("target_metrics") or []:
            if not isinstance(metric, Mapping):
                continue
            target = str(metric.get("target", "") or "")
            if not target:
                continue
            bucket = target_accum.setdefault(
                target,
                {
                    "positive_count": 0.0,
                    "negative_count": 0.0,
                    "brier_weighted_sum": 0.0,
                    "brier_weight": 0.0,
                    "auroc_weighted_sum": 0.0,
                    "auroc_weight": 0.0,
                    "pr_auc_weighted_sum": 0.0,
                    "pr_auc_weight": 0.0,
                },
            )
            positive = _as_int(metric.get("positive_count"), 0)
            negative = _as_int(metric.get("negative_count"), 0)
            weight = positive + negative
            bucket["positive_count"] += positive
            bucket["negative_count"] += negative
            bucket["brier_weighted_sum"] += _as_float(metric.get("brier"), 0.0) * weight
            bucket["brier_weight"] += weight
            if metric.get("auroc") is not None:
                bucket["auroc_weighted_sum"] += _as_float(metric.get("auroc"), 0.0) * weight
                bucket["auroc_weight"] += weight
            if metric.get("pr_auc") is not None:
                bucket["pr_auc_weighted_sum"] += _as_float(metric.get("pr_auc"), 0.0) * weight
                bucket["pr_auc_weight"] += weight

    target_metrics = []
    for target, bucket in sorted(target_accum.items()):
        target_metrics.append(
            {
                "target": target,
                "positive_count": int(bucket["positive_count"]),
                "negative_count": int(bucket["negative_count"]),
                "mean_auroc": (
                    _safe_div(bucket["auroc_weighted_sum"], bucket["auroc_weight"])
                    if bucket["auroc_weight"]
                    else None
                ),
                "mean_pr_auc": (
                    _safe_div(bucket["pr_auc_weighted_sum"], bucket["pr_auc_weight"])
                    if bucket["pr_auc_weight"]
                    else None
                ),
                "mean_brier": _safe_div(bucket["brier_weighted_sum"], bucket["brier_weight"]),
            }
        )
    return {
        "schema_version": "risk_predictor_batch_eval.v1",
        "evaluated_run_count": len(sections),
        "skipped_run_count": len(reports) - len(sections),
        "example_count": example_count,
        "target_metrics": target_metrics,
        "expected_recovery_steps_mae": _safe_div(
            sum(
                _as_float(section.get("expected_recovery_steps_mae"), 0.0)
                * _as_int(section.get("example_count"), 0)
                for section in sections
            ),
            example_count,
        ),
    }


def _aggregate_mode_selector(reports: list[Mapping[str, Any]]) -> dict[str, Any]:
    sections = [dict(report.get("mode_selector") or {}) for report in reports]
    total = sum(_as_int(section.get("example_count"), 0) for section in sections)
    return {
        "schema_version": "mode_selector_batch_eval.v1",
        "example_count": total,
        "accuracy": _safe_div(
            sum(
                _as_float(section.get("accuracy"), 0.0)
                * _as_int(section.get("example_count"), 0)
                for section in sections
            ),
            total,
        ),
        "top2_accuracy": _safe_div(
            sum(
                _as_float(section.get("top2_accuracy"), 0.0)
                * _as_int(section.get("example_count"), 0)
                for section in sections
            ),
            total,
        ),
        "label_counts": _merge_counts(*(section.get("label_counts") or {} for section in sections)),
        "prediction_counts": _merge_counts(
            *(section.get("prediction_counts") or {} for section in sections)
        ),
        "confusion": _merge_confusion(reports),
    }


def _aggregate_action_prior(reports: list[Mapping[str, Any]]) -> dict[str, Any]:
    sections = [dict(report.get("action_prior") or {}) for report in reports]
    family_total = sum(_as_int(section.get("family_total"), 0) for section in sections)
    action_total = sum(_as_int(section.get("action_total"), 0) for section in sections)
    target_total = sum(_as_int(section.get("target_total"), 0) for section in sections)
    return {
        "schema_version": "action_prior_batch_eval.v1",
        "example_count": sum(_as_int(section.get("example_count"), 0) for section in sections),
        "family_accuracy": _safe_div(
            sum(
                _as_float(section.get("family_accuracy"), 0.0)
                * _as_int(section.get("family_total"), 0)
                for section in sections
            ),
            family_total,
        ),
        "family_total": family_total,
        "action_accuracy": _safe_div(
            sum(
                _as_float(section.get("action_accuracy"), 0.0)
                * _as_int(section.get("action_total"), 0)
                for section in sections
            ),
            action_total,
        ),
        "action_total": action_total,
        "target_accuracy": _safe_div(
            sum(
                _as_float(section.get("target_accuracy"), 0.0)
                * _as_int(section.get("target_total"), 0)
                for section in sections
            ),
            target_total,
        ),
        "target_total": target_total,
        "label_family_counts": _merge_counts(
            *(section.get("label_family_counts") or {} for section in sections)
        ),
        "prediction_family_counts": _merge_counts(
            *(section.get("prediction_family_counts") or {} for section in sections)
        ),
    }


def _aggregate_branch_recovery(reports: list[Mapping[str, Any]]) -> dict[str, Any]:
    sections = [dict(report.get("branch_recovery") or {}) for report in reports]
    branch_trace_count = sum(_as_int(section.get("branch_trace_count"), 0) for section in sections)
    branch_label_count = sum(_as_int(section.get("branch_label_count"), 0) for section in sections)
    wrong_branch_count = sum(_as_int(section.get("wrong_branch_count"), 0) for section in sections)
    recoverable_count = sum(
        _as_int(section.get("wrong_branch_recoverable_count"), 0) for section in sections
    )
    success_count = sum(_as_int(section.get("recovery_success_count"), 0) for section in sections)
    return {
        "schema_version": "eca_branch_recovery_batch_summary.v1",
        "run_count": len(sections),
        "runs_with_branch_traces": sum(
            1 for section in sections if _as_int(section.get("branch_trace_count"), 0) > 0
        ),
        "runs_with_wrong_branch_recovery": sum(
            1 for section in sections if bool(section.get("trajectory_wrong_branch_recovery"))
        ),
        "branch_trace_count": branch_trace_count,
        "branch_label_count": branch_label_count,
        "wrong_branch_count": wrong_branch_count,
        "wrong_branch_recoverable_count": recoverable_count,
        "recovery_success_count": success_count,
        "wrong_branch_recovery_rate": _safe_div(success_count, wrong_branch_count),
        "recoverable_recovery_rate": _safe_div(success_count, recoverable_count),
        "initial_direction_counts": _merge_counts(
            *(section.get("initial_direction_counts") or {} for section in sections)
        ),
        "branch_status_counts": _merge_counts(
            *(section.get("branch_status_counts") or {} for section in sections)
        ),
        "recovery_target_direction_counts": _merge_counts(
            *(section.get("recovery_target_direction_counts") or {} for section in sections)
        ),
    }


def build_modeling_batch_report(
    roots: Iterable[Path],
    *,
    limit: int | None = None,
    include_reports: bool = False,
    write_exports: bool = False,
) -> dict[str, Any]:
    """Build a multi-run aggregate report from one or more roots."""
    run_dirs = find_modeling_run_dirs(roots, limit=limit)
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for run_dir in run_dirs:
        try:
            reports.append(
                build_modeling_pipeline_report(run_dir, write_exports=write_exports)
            )
        except Exception as exc:  # pragma: no cover - defensive batch isolation.
            failures.append({"run_dir": str(run_dir), "error": repr(exc)})

    gap_counts: Counter[str] = Counter()
    for report in reports:
        for key, active in _gap_flags(report).items():
            if active:
                gap_counts[key] += 1

    payload: dict[str, Any] = {
        "schema_version": "eca_modeling_batch_report.v1",
        "input_roots": [str(Path(root)) for root in roots],
        "discovered_run_count": len(run_dirs),
        "evaluated_run_count": len(reports),
        "failed_run_count": len(failures),
        "failures": failures,
        "coverage": _aggregate_coverage(reports),
        "gap_counts": dict(sorted(gap_counts.items())),
        "branch_recovery": _aggregate_branch_recovery(reports),
        "candidate_ranker": _aggregate_candidate_ranker(reports),
        "risk_predictor": _aggregate_risk_predictor(reports),
        "mode_selector": _aggregate_mode_selector(reports),
        "action_prior": _aggregate_action_prior(reports),
        "run_summaries": [_compact_run_summary(report) for report in reports],
    }
    if include_reports:
        payload["reports"] = reports
    return payload


def write_modeling_batch_report(
    roots: Iterable[Path],
    output_path: Path,
    *,
    limit: int | None = None,
    include_reports: bool = False,
    write_exports: bool = False,
) -> Path:
    report = build_modeling_batch_report(
        roots,
        limit=limit,
        include_reports=include_reports,
        write_exports=write_exports,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
