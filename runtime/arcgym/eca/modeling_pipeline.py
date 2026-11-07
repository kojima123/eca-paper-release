"""End-to-end offline report for ECA modeling artefacts.

The report stitches together the P0-P5 exporters/evaluators:

* modeling logs
* candidate ranking
* risk prediction, when the optional module is present
* mode selection
* action prior

It is deliberately offline-only and does not mutate controller behavior.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from arcgym.eca.action_prior import (
    build_action_prior_examples_from_bundle,
    evaluate_action_prior,
    write_action_prior_examples,
)
from arcgym.eca.candidate_ranker import evaluate_candidate_ranker
from arcgym.eca.mode_selector import (
    build_mode_selection_examples_from_bundle,
    evaluate_mode_selector,
    write_mode_selection_examples,
)
from arcgym.eca.modeling_logs import build_modeling_logs_from_run_dir, write_modeling_logs


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _count_values(rows: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        text = str(value)
        counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items()))


def _risk_eval(bundle: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        from arcgym.eca.risk_predictor import evaluate_risk_predictor
    except ModuleNotFoundError:
        return None
    risk_examples = list(bundle.get("risk_prediction_examples") or [])
    if not risk_examples:
        return None
    return evaluate_risk_predictor(risk_examples)


def _coverage(bundle: Mapping[str, Any], mode_examples: list[dict[str, Any]], action_examples: list[dict[str, Any]]) -> dict[str, Any]:
    step_count = len(bundle.get("step_logs") or [])
    return {
        "step_count": step_count,
        "candidate_trace_count": len(bundle.get("candidate_traces") or []),
        "branch_trace_count": len(bundle.get("branch_traces") or []),
        "candidate_ranking_example_count": len(bundle.get("candidate_ranking_examples") or []),
        "risk_prediction_example_count": len(bundle.get("risk_prediction_examples") or []),
        "mode_selection_example_count": len(mode_examples),
        "action_prior_example_count": len(action_examples),
        "mode_step_coverage": round(len(mode_examples) / step_count, 6) if step_count else 0.0,
        "action_prior_step_coverage": round(len(action_examples) / step_count, 6) if step_count else 0.0,
    }


def _branch_recovery_summary(bundle: Mapping[str, Any]) -> dict[str, Any]:
    branch_traces = [dict(row) for row in bundle.get("branch_traces") or []]
    hindsight = dict(bundle.get("hindsight_labels") or {})
    branch_labels = [dict(row) for row in hindsight.get("branch_labels") or []]
    trajectory = dict(hindsight.get("trajectory") or {})

    wrong_branch_count = sum(1 for row in branch_labels if _as_bool(row.get("wrong_branch")))
    recoverable_count = sum(
        1 for row in branch_labels if _as_bool(row.get("wrong_branch_recoverable"))
    )
    success_count = sum(1 for row in branch_labels if _as_bool(row.get("recovery_success")))
    return {
        "schema_version": "eca_branch_recovery_summary.v1",
        "branch_trace_count": len(branch_traces),
        "branch_label_count": len(branch_labels),
        "wrong_branch_count": wrong_branch_count,
        "wrong_branch_recoverable_count": recoverable_count,
        "recovery_success_count": success_count,
        "wrong_branch_recovery_rate": _safe_div(success_count, wrong_branch_count),
        "recoverable_recovery_rate": _safe_div(success_count, recoverable_count),
        "initial_direction_counts": _count_values(branch_traces, "initial_direction"),
        "branch_status_counts": _count_values(branch_traces, "branch_status"),
        "recovery_target_direction_counts": _count_values(
            branch_labels, "recovery_target_direction"
        ),
        "trajectory_wrong_branch_recovery": _as_bool(
            trajectory.get("wrong_branch_recovery")
        ),
    }


def build_modeling_pipeline_report(
    run_dir: Path,
    *,
    write_exports: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if write_exports:
        write_modeling_logs(run_dir)
    bundle = build_modeling_logs_from_run_dir(run_dir)
    mode_examples = build_mode_selection_examples_from_bundle(bundle)
    action_examples = build_action_prior_examples_from_bundle(
        bundle,
        mode_selection_examples=mode_examples,
    )
    if write_exports:
        write_mode_selection_examples(run_dir, mode_examples)
        write_action_prior_examples(run_dir, action_examples)

    report: dict[str, Any] = {
        "schema_version": "eca_modeling_pipeline_report.v1",
        "run_dir": str(run_dir),
        "trajectory_summary": dict(bundle.get("trajectory_summary") or {}),
        "coverage": _coverage(bundle, mode_examples, action_examples),
        "branch_recovery": _branch_recovery_summary(bundle),
        "candidate_ranker": evaluate_candidate_ranker(
            bundle.get("candidate_ranking_examples") or []
        ),
        "mode_selector": evaluate_mode_selector(mode_examples),
        "action_prior": evaluate_action_prior(action_examples),
    }
    risk_metrics = _risk_eval(bundle)
    if risk_metrics is not None:
        report["risk_predictor"] = risk_metrics
    else:
        report["risk_predictor"] = {
            "skipped": True,
            "reason": "risk_predictor_module_or_examples_unavailable",
        }
    return report


def write_modeling_pipeline_report(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    write_exports: bool = False,
) -> Path:
    run_dir = Path(run_dir)
    report = build_modeling_pipeline_report(run_dir, write_exports=write_exports)
    path = output_path or (run_dir / "eca_modeling_pipeline_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
