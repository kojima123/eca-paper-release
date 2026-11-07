"""Research-log projection for ECA / GCSTM / ControlIR modeling.

This module is intentionally an exporter, not a new controller.  It projects
the existing runtime artefacts into stable StepLog v2, CandidateTrace,
BranchTrace, and TrajectorySummary v2 records so downstream predictors can be
trained without depending on the current ad-hoc metric field names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping


STEPLOG_SCHEMA_VERSION = "steplog.v2"
CANDIDATE_TRACE_SCHEMA_VERSION = "candidate_trace.v2"
BRANCH_TRACE_SCHEMA_VERSION = "branch_trace.v2"
TRAJECTORY_SUMMARY_SCHEMA_VERSION = "trajectory_summary.v2"
HINDSIGHT_LABELS_SCHEMA_VERSION = "eca_hindsight_labels.v2"
CANDIDATE_RANKING_EXAMPLE_SCHEMA_VERSION = "candidate_ranking_example.v2"
RISK_PREDICTION_EXAMPLE_SCHEMA_VERSION = "risk_prediction_example.v2"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


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


def _step_id(row: Mapping[str, Any], fallback: int) -> int:
    for key in ("step_id", "action_idx", "step"):
        if key in row:
            return _as_int(row.get(key), fallback)
    return fallback


def _score(row: Mapping[str, Any]) -> float:
    return _as_float(row.get("score"), 0.0)


def _level(row: Mapping[str, Any]) -> int:
    return _as_int(row.get("level", row.get("level_number")), 0)


def _score_delta(rows: list[Mapping[str, Any]], index: int) -> float:
    row = rows[index]
    if "score_delta" in row:
        return _as_float(row.get("score_delta"), 0.0)
    if index <= 0:
        return 0.0
    return _score(row) - _score(rows[index - 1])


def _level_delta(rows: list[Mapping[str, Any]], index: int) -> int:
    row = rows[index]
    if "level_delta" in row:
        return _as_int(row.get("level_delta"), 0)
    if index <= 0:
        return 0
    return _level(row) - _level(rows[index - 1])


def _progress_within(rows: list[Mapping[str, Any]], index: int, horizon: int = 5) -> bool:
    base_score = _score(rows[index])
    base_level = _level(rows[index])
    end = min(len(rows), index + horizon + 1)
    for future in rows[index + 1 : end]:
        if _score(future) > base_score or _level(future) > base_level:
            return True
    return False


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _model_from_context(
    *,
    rows: list[Mapping[str, Any]],
    episode_summary: Mapping[str, Any] | None,
    run_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    first_row = rows[0] if rows else {}
    episode_summary = episode_summary or {}
    run_metadata = run_metadata or {}
    model_name = _first_nonempty(
        run_metadata.get("model"),
        run_metadata.get("model_name"),
        episode_summary.get("model"),
        episode_summary.get("llm_model"),
        first_row.get("model"),
        first_row.get("llm_model"),
        first_row.get("V2_LLM_MODEL"),
    )
    backend = _first_nonempty(
        run_metadata.get("backend"),
        episode_summary.get("backend"),
        first_row.get("backend"),
        first_row.get("llm_backend"),
    )
    local_hint = " ".join([model_name, backend]).lower()
    return {
        "name": model_name,
        "backend": backend,
        "local": bool(
            run_metadata.get("local", False)
            or episode_summary.get("local_model", False)
            or any(token in local_hint for token in ("gemma", "qwen", "vulkan", "local"))
        ),
    }


def _selected_target_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = _first_nonempty(
        row.get("selected_target_id"),
        row.get("local_controller_selected_target_id"),
        row.get("selected_target_anchor"),
    )
    return {
        "candidate_id": candidate_id,
        "anchor": str(row.get("selected_target_anchor", "") or ""),
        "source": str(row.get("selected_target_source", "") or ""),
        "admission_score": _as_float(row.get("selected_target_admission_score"), 0.0),
        "reason": str(row.get("selected_target_reason", "") or ""),
        "confidence": str(row.get("selected_target_confidence", "") or ""),
        "local_controller_selected": str(row.get("local_controller_selected_target_id", "") or ""),
    }


def _candidate_pool_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "size": _as_int(row.get("candidate_pool_size"), 0),
        "family_histogram": _json_safe(row.get("candidate_pool_family_histogram") or {}),
        "transition_family_histogram": _json_safe(
            row.get("candidate_pool_transition_family_histogram") or {}
        ),
        "reject_reason_histogram": _json_safe(
            row.get("candidate_pool_reject_reason_histogram") or {}
        ),
        "no_target_disposition": str(row.get("candidate_pool_no_target_disposition", "") or ""),
        "escape_pending": _as_bool(row.get("candidate_pool_escape_pending")),
        "dead_tail_ratio": _as_float(row.get("candidate_pool_dead_tail_ratio"), 0.0),
        "live_escape_candidate_count": _as_int(
            row.get("candidate_pool_live_escape_candidate_count"), 0
        ),
        "selected_target_empty_with_candidate_pool": _as_bool(
            row.get("candidate_pool_selected_target_empty_with_candidate_pool")
        ),
    }


def _control_ir_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "control_ir_enabled",
        "control_ir_mode",
        "control_ir_live_ticket_count",
        "control_ir_live_tickets_by_kind",
        "control_ir_active_suppression_dispositions",
        "control_ir_active_promote_from_sources",
        "control_ir_last_issue_decisions",
        "control_ir_live_tickets",
    )
    return {key: _json_safe(row.get(key)) for key in keys if key in row}


def _runtime_packet(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "no_change_streak",
        "repeated_state_count",
        "queue_exhaustions",
        "analyzer_calls",
        "target_health_reason",
        "target_health_no_score_streak",
        "target_health_no_info_gain_streak",
        "target_health_dead_ratio",
        "local_controller_locally_planned",
        "local_controller_action_preview",
        "selected_target_promotion_count",
        "selected_target_promotion_success_count",
        "structural_time_tau",
        "structural_time_goal_progress",
        "structural_time_uncertainty",
        "structural_guidance_mode",
        "candidate_pool_no_target_disposition",
    )
    return {key: _json_safe(row.get(key)) for key in keys if key in row}


def _evidence_items(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    selected = _selected_target_payload(row)
    if selected["candidate_id"] or selected["anchor"]:
        evidence.append({"kind": "selected_target", **selected})
    pool = _candidate_pool_payload(row)
    if pool["size"] or pool["no_target_disposition"]:
        evidence.append({"kind": "candidate_pool", **pool})
    if "target_health_reason" in row:
        evidence.append(
            {
                "kind": "target_health",
                "reason": str(row.get("target_health_reason", "") or ""),
                "no_score_streak": _as_int(row.get("target_health_no_score_streak"), 0),
                "no_info_gain_streak": _as_int(
                    row.get("target_health_no_info_gain_streak"), 0
                ),
                "dead_ratio": _as_float(row.get("target_health_dead_ratio"), 0.0),
            }
        )
    control_ir = _control_ir_payload(row)
    if control_ir:
        evidence.append({"kind": "control_ir", **control_ir})
    if any(key in row for key in ("structural_time_tau", "structural_time_goal_progress")):
        evidence.append(
            {
                "kind": "structural_time",
                "tau": _as_float(row.get("structural_time_tau"), 0.0),
                "goal_progress": _as_float(row.get("structural_time_goal_progress"), 0.0),
                "uncertainty": _as_float(row.get("structural_time_uncertainty"), 0.0),
                "q_flow": _as_float(row.get("structural_time_q_flow"), 0.0),
                "d_flow": _as_float(row.get("structural_time_d_flow"), 0.0),
            }
        )
    return evidence


def _action_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    action = _first_nonempty(
        row.get("action"),
        row.get("issued_action"),
        row.get("current_action"),
        row.get("last_action"),
    )
    preview = row.get("local_controller_action_preview") or []
    if not action and isinstance(preview, list) and preview:
        action = str(preview[0] or "")
    payload = row.get("action_payload") if isinstance(row.get("action_payload"), Mapping) else {}
    return {
        "name": action,
        "preview": _json_safe(preview),
        "source": str(row.get("action_source", "") or ""),
        "payload": _json_safe(payload),
        "locally_planned": _as_bool(row.get("local_controller_locally_planned")),
    }


@dataclass
class StepLogV2:
    run_id: str
    episode_id: str
    step_id: int
    model: dict[str, Any]
    observation: dict[str, Any]
    evidence_items: list[dict[str, Any]]
    runtime_packet: dict[str, Any]
    candidate_pool: dict[str, Any]
    queue_state: dict[str, Any]
    control_ir: dict[str, Any]
    action: dict[str, Any]
    outcome: dict[str, Any]
    hindsight_labels: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "schema_version": STEPLOG_SCHEMA_VERSION,
                "run_id": self.run_id,
                "episode_id": self.episode_id,
                "step_id": self.step_id,
                "model": self.model,
                "observation": self.observation,
                "evidence_items": self.evidence_items,
                "runtime_packet": self.runtime_packet,
                "candidate_pool": self.candidate_pool,
                "queue_state": self.queue_state,
                "control_ir": self.control_ir,
                "action": self.action,
                "outcome": self.outcome,
                "hindsight_labels": self.hindsight_labels,
            }
        )


@dataclass
class CandidateTrace:
    candidate_id: str
    site: list[int] | None
    source: str
    created_step: int
    first_tried_step: int | None = None
    status_history: list[list[Any]] = field(default_factory=list)
    final_outcome: str = "open"
    promotion: bool = False
    contributed_to_score: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "schema_version": CANDIDATE_TRACE_SCHEMA_VERSION,
                "candidate_id": self.candidate_id,
                "site": self.site,
                "source": self.source,
                "created_step": self.created_step,
                "first_tried_step": self.first_tried_step,
                "status_history": self.status_history,
                "final_outcome": self.final_outcome,
                "promotion": self.promotion,
                "contributed_to_score": self.contributed_to_score,
                "metadata": self.metadata,
            }
        )


@dataclass
class BranchTrace:
    branch_id: str
    initial_direction: str
    branch_status: str
    tried_candidates: list[str]
    opposite_branch_available: bool
    recovery_trigger_step: int | None = None
    recovered_to: str | None = None
    recovery_success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "schema_version": BRANCH_TRACE_SCHEMA_VERSION,
                "branch_id": self.branch_id,
                "initial_direction": self.initial_direction,
                "branch_status": self.branch_status,
                "tried_candidates": self.tried_candidates,
                "opposite_branch_available": self.opposite_branch_available,
                "recovery_trigger_step": self.recovery_trigger_step,
                "recovered_to": self.recovered_to,
                "recovery_success": self.recovery_success,
            }
        )


@dataclass
class TrajectorySummaryV2:
    run_id: str
    model: str
    score: float
    highest_level: int
    promotions: int
    wrong_branch_recovery: bool
    tried_site_count: int
    promotion_sites: list[list[int]]
    failure_cluster: str | None
    budget_used: int
    local_model: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "schema_version": TRAJECTORY_SUMMARY_SCHEMA_VERSION,
                "run_id": self.run_id,
                "model": self.model,
                "score": self.score,
                "highest_level": self.highest_level,
                "promotions": self.promotions,
                "wrong_branch_recovery": self.wrong_branch_recovery,
                "tried_site_count": self.tried_site_count,
                "promotion_sites": self.promotion_sites,
                "failure_cluster": self.failure_cluster,
                "budget_used": self.budget_used,
                "local_model": self.local_model,
                "metadata": self.metadata,
            }
        )


def build_step_logs_v2(
    step_rows: Iterable[Mapping[str, Any]],
    *,
    episode_summary: Mapping[str, Any] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in step_rows]
    model = _model_from_context(
        rows=rows,
        episode_summary=episode_summary,
        run_metadata=run_metadata,
    )
    run_id = _first_nonempty(
        (run_metadata or {}).get("run_id"),
        (episode_summary or {}).get("run_id"),
        rows[0].get("run_id") if rows else "",
    )
    episode_id = _first_nonempty(
        (run_metadata or {}).get("episode_id"),
        (episode_summary or {}).get("episode_id"),
        run_id,
    )
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        step_id = _step_id(row, idx)
        score_delta = _score_delta(rows, idx)
        level_delta = _level_delta(rows, idx)
        item = StepLogV2(
            run_id=run_id,
            episode_id=episode_id,
            step_id=step_id,
            model=dict(model),
            observation={
                "game_id": str(row.get("game_id", "") or ""),
                "level": _level(row),
                "score": _score(row),
                "state_hash": str(row.get("state_hash", "") or ""),
                "arc_state_name": str(row.get("arc_state_name", "") or ""),
                "arc_game_over": _as_bool(row.get("arc_game_over")),
                "remaining_actions": _as_int(row.get("remaining_actions"), 0),
            },
            evidence_items=_evidence_items(row),
            runtime_packet=_runtime_packet(row),
            candidate_pool=_candidate_pool_payload(row),
            queue_state={
                "queue_exhaustions": _as_int(row.get("queue_exhaustions"), 0),
                "analyzer_calls": _as_int(row.get("analyzer_calls"), 0),
                "local_controller_action_preview": _json_safe(
                    row.get("local_controller_action_preview") or []
                ),
                "candidate_pool_size": _as_int(row.get("candidate_pool_size"), 0),
            },
            control_ir=_control_ir_payload(row),
            action=_action_payload(row),
            outcome={
                "score_delta": score_delta,
                "level_delta": level_delta,
                "no_change": _as_bool(row.get("no_change")),
                "repeated_state": _as_bool(row.get("repeated_state")),
                "board_changed": (
                    _as_bool(row.get("board_changed"))
                    if "board_changed" in row
                    else not _as_bool(row.get("no_change"))
                ),
                "arc_game_over": _as_bool(row.get("arc_game_over")),
            },
            hindsight_labels={
                "progress_within_5": _progress_within(rows, idx, horizon=5),
                "score_progress": bool(score_delta > 0.0),
                "level_progress": bool(level_delta > 0),
                "selected_target_present": bool(
                    _selected_target_payload(row).get("candidate_id")
                    or _selected_target_payload(row).get("anchor")
                ),
            },
        )
        out.append(item.to_dict())
    return out


def _site_from_anchor(anchor: str) -> list[int] | None:
    nums = re.findall(r"-?\d+", str(anchor or ""))
    if len(nums) < 2:
        return None
    return [_as_int(nums[0]), _as_int(nums[1])]


def _site_from_trace(trace: Mapping[str, Any]) -> list[int] | None:
    if "site" in trace and isinstance(trace.get("site"), (list, tuple)):
        site = list(trace.get("site") or [])
        if len(site) >= 2:
            return [_as_int(site[0]), _as_int(site[1])]
    if "site_row" in trace and "site_col" in trace:
        return [_as_int(trace.get("site_row")), _as_int(trace.get("site_col"))]
    events = trace.get("events") or []
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, Mapping):
                continue
            if "site_row" in event and "site_col" in event:
                return [_as_int(event.get("site_row")), _as_int(event.get("site_col"))]
    return None


def _first_event_step(events: list[Any], default: int) -> int:
    for event in events:
        if isinstance(event, Mapping) and "step" in event:
            return _as_int(event.get("step"), default)
    return default


def _append_status(trace: CandidateTrace, status: str, step: int) -> None:
    if trace.status_history and trace.status_history[-1] == [status, step]:
        return
    trace.status_history.append([status, step])


def _status_from_event_phase(event: Mapping[str, Any]) -> str:
    phase = str(event.get("phase", "") or "").strip()
    outcome = str(event.get("outcome", "") or "").strip()
    if phase == "contact_registered":
        return "created"
    if phase == "contact_queued":
        return "queued"
    if phase == "queue_advanced":
        return "advanced"
    if phase == "retest_step":
        return "tested"
    if phase == "pending_contact_suppressed":
        return "suppressed"
    if phase == "finalize" and outcome:
        return outcome
    return phase or outcome or "observed"


def _candidate_trace_from_retest_trace(trace: Mapping[str, Any], fallback_index: int) -> CandidateTrace:
    site = _site_from_trace(trace)
    events = list(trace.get("events") or [])
    created_step = _first_event_step(events, _as_int(trace.get("step"), 0))
    site_text = f"{site[0]},{site[1]}" if site else f"unknown,{fallback_index}"
    candidate = CandidateTrace(
        candidate_id=f"pending_contact:{site_text}:{created_step}",
        site=site,
        source="pending_contact",
        created_step=created_step,
        first_tried_step=None,
        final_outcome=str(trace.get("outcome", "") or "open"),
        metadata={
            "reason": str(trace.get("reason", "") or ""),
            "raw_outcome": str(trace.get("outcome", "") or ""),
        },
    )
    for event in events:
        if not isinstance(event, Mapping):
            continue
        step = _as_int(event.get("step"), created_step)
        status = _status_from_event_phase(event)
        _append_status(candidate, status, step)
        if status in {"advanced", "tested", "promoted", "inconclusive"}:
            candidate.first_tried_step = (
                step if candidate.first_tried_step is None else min(candidate.first_tried_step, step)
            )
    if not candidate.status_history:
        _append_status(candidate, candidate.final_outcome, created_step)
    if candidate.first_tried_step is None:
        candidate.first_tried_step = created_step
    candidate.promotion = candidate.final_outcome == "promoted"
    candidate.contributed_to_score = candidate.promotion
    return candidate


def build_candidate_traces(
    step_rows: Iterable[Mapping[str, Any]],
    *,
    retest_traces: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in step_rows]
    traces: dict[str, CandidateTrace] = {}
    last_candidate_id = ""
    for idx, row in enumerate(rows):
        selected = _selected_target_payload(row)
        candidate_id = str(selected.get("candidate_id", "") or "")
        anchor = str(selected.get("anchor", "") or "")
        if not candidate_id and anchor:
            candidate_id = f"anchor:{anchor}"
        if not candidate_id:
            if _score_delta(rows, idx) > 0.0 and last_candidate_id in traces:
                trace = traces[last_candidate_id]
                step = _step_id(row, idx)
                _append_status(trace, "promoted", step)
                trace.final_outcome = "promoted"
                trace.promotion = True
                trace.contributed_to_score = True
            continue
        last_candidate_id = candidate_id
        step = _step_id(row, idx)
        if candidate_id not in traces:
            traces[candidate_id] = CandidateTrace(
                candidate_id=candidate_id,
                site=_site_from_anchor(anchor),
                source=str(selected.get("source", "") or "selected_target"),
                created_step=step,
                first_tried_step=step,
                status_history=[],
                final_outcome="selected",
                metadata={
                    "anchor": anchor,
                    "admission_score": selected.get("admission_score", 0.0),
                    "reason": selected.get("reason", ""),
                    "confidence": selected.get("confidence", ""),
                },
            )
            _append_status(traces[candidate_id], "created", step)
        trace = traces[candidate_id]
        _append_status(trace, "selected", step)
        trace.first_tried_step = min(trace.first_tried_step or step, step)
        if _score_delta(rows, idx) > 0.0:
            _append_status(trace, "promoted", step)
            trace.final_outcome = "promoted"
            trace.promotion = True
            trace.contributed_to_score = True

    for idx, raw_trace in enumerate(retest_traces or []):
        if not isinstance(raw_trace, Mapping):
            continue
        candidate = _candidate_trace_from_retest_trace(raw_trace, idx)
        traces[candidate.candidate_id] = candidate

    return [trace.to_dict() for trace in sorted(traces.values(), key=lambda t: (t.created_step, t.candidate_id))]


def _direction_for_site(site: list[int] | None, center_col: float) -> str:
    if not site or len(site) < 2:
        return "unknown"
    col = float(site[1])
    if col < center_col:
        return "west"
    if col > center_col:
        return "east"
    return "center"


def build_branch_traces(candidate_traces: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    traces = [dict(trace) for trace in candidate_traces]
    sites = [trace.get("site") for trace in traces if isinstance(trace.get("site"), list)]
    cols = [float(site[1]) for site in sites if len(site) >= 2]
    if len(set(cols)) < 2:
        return []
    center_col = float(median(cols))
    by_direction: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        direction = _direction_for_site(trace.get("site"), center_col)
        if direction in {"west", "east"}:
            by_direction.setdefault(direction, []).append(trace)
    if len(by_direction) < 2:
        return []

    ordered = sorted(
        (
            (
                direction,
                sorted(
                    items,
                    key=lambda item: _as_int(
                        item.get("first_tried_step", item.get("created_step")), 0
                    ),
                ),
            )
            for direction, items in by_direction.items()
        ),
        key=lambda pair: _as_int(
            pair[1][0].get("first_tried_step", pair[1][0].get("created_step")), 0
        ),
    )
    initial_direction, initial_items = ordered[0]
    opposite_direction = "west" if initial_direction == "east" else "east"
    opposite_items = by_direction.get(opposite_direction, [])
    initial_promoted = any(item.get("final_outcome") == "promoted" for item in initial_items)
    initial_closed = all(
        str(item.get("final_outcome", "") or "") in {"inconclusive", "rejected", "suppressed", "selected"}
        for item in initial_items
    )
    opposite_promoted_items = [
        item for item in opposite_items if str(item.get("final_outcome", "") or "") == "promoted"
    ]
    recovery_success = bool((not initial_promoted) and opposite_promoted_items)
    branch_status = "promoted" if initial_promoted else ("inconclusive" if initial_closed else "open")
    recovery_trigger_step = None
    if recovery_success:
        recovery_trigger_step = min(
            _as_int(item.get("first_tried_step", item.get("created_step")), 0)
            for item in opposite_promoted_items
        )
    branch = BranchTrace(
        branch_id=f"{initial_direction}_initial",
        initial_direction=initial_direction,
        branch_status=branch_status,
        tried_candidates=[str(item.get("candidate_id", "") or "") for item in initial_items],
        opposite_branch_available=bool(opposite_items),
        recovery_trigger_step=recovery_trigger_step,
        recovered_to=opposite_direction if recovery_success else None,
        recovery_success=recovery_success,
    )
    return [branch.to_dict()]


def build_trajectory_summary_v2(
    step_rows: Iterable[Mapping[str, Any]],
    *,
    candidate_traces: Iterable[Mapping[str, Any]],
    branch_traces: Iterable[Mapping[str, Any]],
    episode_summary: Mapping[str, Any] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [dict(row) for row in step_rows]
    candidates = [dict(trace) for trace in candidate_traces]
    branches = [dict(trace) for trace in branch_traces]
    model = _model_from_context(
        rows=rows,
        episode_summary=episode_summary,
        run_metadata=run_metadata,
    )
    run_id = _first_nonempty(
        (run_metadata or {}).get("run_id"),
        (episode_summary or {}).get("run_id"),
        rows[0].get("run_id") if rows else "",
    )
    final_score = _score(rows[-1]) if rows else _as_float((episode_summary or {}).get("max_score"), 0.0)
    max_score = max([_score(row) for row in rows], default=final_score)
    highest_level = max([_level(row) for row in rows], default=_as_int((episode_summary or {}).get("highest_level"), 0))
    promoted = [trace for trace in candidates if _as_bool(trace.get("promotion"))]
    promotion_sites = [
        list(trace.get("site") or [])
        for trace in promoted
        if isinstance(trace.get("site"), list) and len(trace.get("site") or []) >= 2
    ]
    tried_sites = {
        tuple(trace.get("site") or [])
        for trace in candidates
        if isinstance(trace.get("site"), list) and len(trace.get("site") or []) >= 2
    }
    summary = TrajectorySummaryV2(
        run_id=run_id,
        model=str(model.get("name", "") or ""),
        score=max_score,
        highest_level=highest_level,
        promotions=len(promoted),
        wrong_branch_recovery=any(_as_bool(trace.get("recovery_success")) for trace in branches),
        tried_site_count=len(tried_sites),
        promotion_sites=promotion_sites,
        failure_cluster=(
            str((episode_summary or {}).get("failure_cluster") or "")
            or str((episode_summary or {}).get("failure_cluster_id") or "")
            or None
        ),
        budget_used=len(rows),
        local_model=_as_bool(model.get("local")),
        metadata={
            "final_score": final_score,
            "max_score": max_score,
            "arc_game_over": _as_bool(rows[-1].get("arc_game_over")) if rows else False,
        },
    )
    return summary.to_dict()


def _candidate_status_steps(trace: Mapping[str, Any], status: str) -> list[int]:
    steps: list[int] = []
    for item in trace.get("status_history") or []:
        if not isinstance(item, list) or len(item) < 2:
            continue
        if str(item[0] or "") == status:
            steps.append(_as_int(item[1], 0))
    return steps


def _candidate_terminal_step(trace: Mapping[str, Any]) -> int:
    history = trace.get("status_history") or []
    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, list) and len(last) >= 2:
            return _as_int(last[1], _as_int(trace.get("created_step"), 0))
    return _as_int(trace.get("first_tried_step", trace.get("created_step")), 0)


def _candidate_value_label(trace: Mapping[str, Any]) -> dict[str, Any]:
    outcome = str(trace.get("final_outcome", "") or "")
    promoted = _as_bool(trace.get("promotion"))
    created_step = _as_int(trace.get("created_step"), 0)
    first_tried_step = trace.get("first_tried_step")
    tried_step = _as_int(first_tried_step, created_step) if first_tried_step is not None else None
    terminal_step = _candidate_terminal_step(trace)
    test_cost = max(0, terminal_step - (tried_step if tried_step is not None else created_step))
    if promoted:
        value = 1.0
    elif outcome == "inconclusive":
        value = 0.15
    elif outcome in {"suppressed", "rejected"}:
        value = -0.25
    elif outcome in {"queued", "advanced", "selected", "open"}:
        value = 0.05
    else:
        value = 0.0
    value -= min(0.35, test_cost / 120.0)
    return {
        "candidate_id": str(trace.get("candidate_id", "") or ""),
        "site": _json_safe(trace.get("site")),
        "source": str(trace.get("source", "") or ""),
        "final_outcome": outcome,
        "candidate_value": round(value, 4),
        "promotion": promoted,
        "led_to_progress": _as_bool(trace.get("contributed_to_score")),
        "later_used": bool(tried_step is not None),
        "first_tried_step": tried_step,
        "test_cost": test_cost,
        "wasted_budget": bool((not promoted) and test_cost > 0),
        "promotion_step": min(_candidate_status_steps(trace, "promoted"), default=None),
    }


def _open_or_untried_candidates(candidate_traces: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for trace in candidate_traces:
        outcome = str(trace.get("final_outcome", "") or "")
        if outcome in {"open", "queued", "selected"} or trace.get("first_tried_step") is None:
            out.append(trace)
    return out


def _budget_exhaustion_before_truth(
    *,
    step_logs: list[Mapping[str, Any]],
    candidate_traces: list[Mapping[str, Any]],
    trajectory_summary: Mapping[str, Any],
) -> bool:
    if str(trajectory_summary.get("failure_cluster", "") or "").find("budget") >= 0:
        return True
    if not step_logs:
        return False
    last = step_logs[-1]
    remaining = _as_int((last.get("observation") or {}).get("remaining_actions"), 0)
    if remaining > 0:
        return False
    return bool(_open_or_untried_candidates(candidate_traces))


def build_hindsight_labels_v2(
    *,
    step_logs: Iterable[Mapping[str, Any]],
    candidate_traces: Iterable[Mapping[str, Any]],
    branch_traces: Iterable[Mapping[str, Any]],
    trajectory_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build first-pass hindsight labels for model training.

    These labels are conservative projections from observed outcomes.  They do
    not assert hidden truth where the run never tested a candidate.
    """
    steps = [dict(row) for row in step_logs]
    candidates = [dict(trace) for trace in candidate_traces]
    branches = [dict(trace) for trace in branch_traces]
    candidate_labels = [_candidate_value_label(trace) for trace in candidates]
    branch_labels = []
    for branch in branches:
        wrong_branch = bool(
            branch.get("opposite_branch_available")
            and str(branch.get("branch_status", "") or "") != "promoted"
        )
        branch_labels.append(
            {
                "branch_id": str(branch.get("branch_id", "") or ""),
                "wrong_branch": wrong_branch,
                "wrong_branch_recoverable": bool(
                    wrong_branch and branch.get("opposite_branch_available")
                ),
                "recovery_success": _as_bool(branch.get("recovery_success")),
                "recovery_target_direction": branch.get("recovered_to"),
                "recovery_trigger_step": branch.get("recovery_trigger_step"),
            }
        )

    promoted_count = sum(1 for item in candidate_labels if item["promotion"])
    untried_open = _open_or_untried_candidates(candidates)
    labels = {
        "schema_version": HINDSIGHT_LABELS_SCHEMA_VERSION,
        "trajectory": {
            "wrong_branch_recovery": _as_bool(
                trajectory_summary.get("wrong_branch_recovery")
            ),
            "budget_exhaustion_before_truth": _budget_exhaustion_before_truth(
                step_logs=steps,
                candidate_traces=candidates,
                trajectory_summary=trajectory_summary,
            ),
            "candidate_loss": bool(untried_open and promoted_count == 0),
            "promotion_miss": bool(
                promoted_count == 0
                and any(item["later_used"] and item["wasted_budget"] for item in candidate_labels)
            ),
            "failure_cluster": trajectory_summary.get("failure_cluster"),
        },
        "candidate_labels": candidate_labels,
        "branch_labels": branch_labels,
    }
    return _json_safe(labels)


def _candidate_label_by_id(hindsight_labels: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in hindsight_labels.get("candidate_labels") or []:
        if not isinstance(item, Mapping):
            continue
        candidate_id = str(item.get("candidate_id", "") or "")
        if candidate_id:
            out[candidate_id] = dict(item)
    return out


def _candidate_direction_from_branch(
    *,
    candidate_id: str,
    branch_traces: list[Mapping[str, Any]],
) -> str:
    for branch in branch_traces:
        if candidate_id in set(str(item or "") for item in branch.get("tried_candidates") or []):
            return str(branch.get("initial_direction", "") or "")
    return ""


def build_candidate_ranking_examples_v2(
    *,
    candidate_traces: Iterable[Mapping[str, Any]],
    hindsight_labels: Mapping[str, Any],
    branch_traces: Iterable[Mapping[str, Any]],
    trajectory_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build listwise/pairwise-ready candidate ranking examples.

    The output is intentionally flat JSONL-friendly data.  It can feed a
    baseline ranker now and a learned ranker later.
    """
    labels_by_id = _candidate_label_by_id(hindsight_labels)
    branches = [dict(branch) for branch in branch_traces]
    examples: list[dict[str, Any]] = []
    run_id = str(trajectory_summary.get("run_id", "") or "")
    for trace in candidate_traces:
        candidate_id = str(trace.get("candidate_id", "") or "")
        if not candidate_id:
            continue
        label = labels_by_id.get(candidate_id, {})
        site = trace.get("site") if isinstance(trace.get("site"), list) else None
        status_history = list(trace.get("status_history") or [])
        features = {
            "source": str(trace.get("source", "") or ""),
            "created_step": _as_int(trace.get("created_step"), 0),
            "first_tried_step": _as_int(trace.get("first_tried_step"), -1)
            if trace.get("first_tried_step") is not None
            else -1,
            "has_site": bool(site and len(site) >= 2),
            "site_row": _as_int(site[0], 0) if site and len(site) >= 2 else 0,
            "site_col": _as_int(site[1], 0) if site and len(site) >= 2 else 0,
            "status_count": len(status_history),
            "was_queued": any(
                isinstance(item, list) and item and item[0] == "queued"
                for item in status_history
            ),
            "was_advanced": any(
                isinstance(item, list) and item and item[0] == "advanced"
                for item in status_history
            ),
            "branch_direction": _candidate_direction_from_branch(
                candidate_id=candidate_id,
                branch_traces=branches,
            ),
            "trajectory_wrong_branch_recovery": _as_bool(
                trajectory_summary.get("wrong_branch_recovery")
            ),
            "trajectory_budget_used": _as_int(trajectory_summary.get("budget_used"), 0),
        }
        examples.append(
            _json_safe(
                {
                    "schema_version": CANDIDATE_RANKING_EXAMPLE_SCHEMA_VERSION,
                    "run_id": run_id,
                    "group_id": f"{run_id}:candidates",
                    "candidate_id": candidate_id,
                    "features": features,
                    "label": {
                        "candidate_value": _as_float(label.get("candidate_value"), 0.0),
                        "promotion": _as_bool(label.get("promotion")),
                        "led_to_progress": _as_bool(label.get("led_to_progress")),
                        "wasted_budget": _as_bool(label.get("wasted_budget")),
                    },
                }
            )
        )
    return examples


def _window_any(
    rows: list[Mapping[str, Any]],
    index: int,
    *,
    horizon: int,
    predicate: Any,
) -> bool:
    end = min(len(rows), index + horizon + 1)
    for row in rows[index + 1 : end]:
        if predicate(row):
            return True
    return False


def _risk_no_target_collapse(row: Mapping[str, Any]) -> bool:
    candidate_pool = dict(row.get("candidate_pool") or {})
    disposition = str(candidate_pool.get("no_target_disposition", "") or "")
    return bool(
        disposition in {"preterminal_dead_revisit", "hard_terminal", "collapse"}
        or candidate_pool.get("selected_target_empty_with_candidate_pool")
    )


def _risk_dead_revisit(row: Mapping[str, Any]) -> bool:
    candidate_pool = dict(row.get("candidate_pool") or {})
    return str(candidate_pool.get("no_target_disposition", "") or "") == "preterminal_dead_revisit"


def _risk_selected_target_empty(row: Mapping[str, Any]) -> bool:
    candidate_pool = dict(row.get("candidate_pool") or {})
    return _as_bool(candidate_pool.get("selected_target_empty_with_candidate_pool"))


def _risk_queue_exhaustion_delta(rows: list[Mapping[str, Any]], index: int) -> bool:
    current = _as_int((rows[index].get("queue_state") or {}).get("queue_exhaustions"), 0)
    return _window_any(
        rows,
        index,
        horizon=5,
        predicate=lambda row: _as_int((row.get("queue_state") or {}).get("queue_exhaustions"), 0) > current,
    )


def _recovery_trigger_step(hindsight_labels: Mapping[str, Any]) -> int | None:
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


def _wrong_branch_lockin_label(hindsight_labels: Mapping[str, Any]) -> bool:
    for branch in hindsight_labels.get("branch_labels") or []:
        if not isinstance(branch, Mapping):
            continue
        if _as_bool(branch.get("wrong_branch")) and not _as_bool(branch.get("recovery_success")):
            return True
    return False


def _risk_features_from_step(row: Mapping[str, Any], *, max_step: int) -> dict[str, Any]:
    observation = dict(row.get("observation") or {})
    runtime_packet = dict(row.get("runtime_packet") or {})
    candidate_pool = dict(row.get("candidate_pool") or {})
    control_ir = dict(row.get("control_ir") or {})
    hindsight = dict(row.get("hindsight_labels") or {})
    remaining = _as_int(observation.get("remaining_actions"), 0)
    step_id = _as_int(row.get("step_id"), 0)
    return {
        "step_id": step_id,
        "progress_ratio": round(step_id / max(1, max_step), 6),
        "level": _as_int(observation.get("level"), 0),
        "score": _as_float(observation.get("score"), 0.0),
        "remaining_actions": remaining,
        "remaining_ratio": round(remaining / max(1, remaining + step_id), 6),
        "no_change_streak": _as_int(runtime_packet.get("no_change_streak"), 0),
        "repeated_state_count": _as_int(runtime_packet.get("repeated_state_count"), 0),
        "queue_exhaustions": _as_int(runtime_packet.get("queue_exhaustions"), 0),
        "analyzer_calls": _as_int(runtime_packet.get("analyzer_calls"), 0),
        "candidate_pool_size": _as_int(candidate_pool.get("size"), 0),
        "candidate_dead_tail_ratio": _as_float(candidate_pool.get("dead_tail_ratio"), 0.0),
        "live_escape_candidate_count": _as_int(
            candidate_pool.get("live_escape_candidate_count"), 0
        ),
        "selected_target_present": _as_bool(hindsight.get("selected_target_present")),
        "selected_target_empty_with_candidate_pool": _as_bool(
            candidate_pool.get("selected_target_empty_with_candidate_pool")
        ),
        "candidate_escape_pending": _as_bool(candidate_pool.get("escape_pending")),
        "no_target_disposition": str(candidate_pool.get("no_target_disposition", "") or ""),
        "control_ir_live_ticket_count": _as_int(control_ir.get("control_ir_live_ticket_count"), 0),
        "target_health_dead_ratio": _as_float(runtime_packet.get("target_health_dead_ratio"), 0.0),
        "target_health_no_score_streak": _as_int(
            runtime_packet.get("target_health_no_score_streak"), 0
        ),
        "target_health_no_info_gain_streak": _as_int(
            runtime_packet.get("target_health_no_info_gain_streak"), 0
        ),
        "structural_time_goal_progress": _as_float(
            runtime_packet.get("structural_time_goal_progress"), 0.0
        ),
        "structural_time_uncertainty": _as_float(
            runtime_packet.get("structural_time_uncertainty"), 0.0
        ),
    }


def build_risk_prediction_examples_v2(
    *,
    step_logs: Iterable[Mapping[str, Any]],
    hindsight_labels: Mapping[str, Any],
    trajectory_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build horizon-labelled risk examples from StepLog v2 rows."""
    steps = [dict(row) for row in step_logs]
    if not steps:
        return []
    run_id = str(trajectory_summary.get("run_id", "") or "")
    max_step = max(_as_int(row.get("step_id"), idx) for idx, row in enumerate(steps))
    trajectory_labels = dict(hindsight_labels.get("trajectory") or {})
    recovery_step = _recovery_trigger_step(hindsight_labels)
    wrong_branch_lockin = _wrong_branch_lockin_label(hindsight_labels)
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(steps):
        step_id = _as_int(row.get("step_id"), idx)
        expected_recovery_steps = 0.0
        if recovery_step is not None and recovery_step >= step_id:
            expected_recovery_steps = float(recovery_step - step_id)
        label = {
            "p_queue_exhaustion_5": _risk_queue_exhaustion_delta(steps, idx),
            "p_no_target_collapse_5": _window_any(
                steps,
                idx,
                horizon=5,
                predicate=_risk_no_target_collapse,
            ),
            "p_dead_revisit_5": _window_any(
                steps,
                idx,
                horizon=5,
                predicate=_risk_dead_revisit,
            ),
            "p_selected_target_empty_5": _window_any(
                steps,
                idx,
                horizon=5,
                predicate=_risk_selected_target_empty,
            ),
            "p_wrong_branch_lockin_5": wrong_branch_lockin,
            "p_budget_exhaustion_before_test": bool(
                trajectory_labels.get("budget_exhaustion_before_truth")
                and _as_int((row.get("observation") or {}).get("remaining_actions"), 0) <= 5
            ),
            "p_progress_5": _as_bool((row.get("hindsight_labels") or {}).get("progress_within_5")),
            "expected_recovery_steps": expected_recovery_steps,
        }
        out.append(
            _json_safe(
                {
                    "schema_version": RISK_PREDICTION_EXAMPLE_SCHEMA_VERSION,
                    "run_id": run_id,
                    "group_id": f"{run_id}:risk",
                    "step_id": step_id,
                    "features": _risk_features_from_step(row, max_step=max_step),
                    "label": label,
                }
            )
        )
    return out


def build_modeling_logs(
    step_rows: Iterable[Mapping[str, Any]],
    *,
    episode_summary: Mapping[str, Any] | None = None,
    retest_traces: Iterable[Mapping[str, Any]] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [dict(row) for row in step_rows]
    step_logs = build_step_logs_v2(
        rows,
        episode_summary=episode_summary,
        run_metadata=run_metadata,
    )
    candidate_traces = build_candidate_traces(rows, retest_traces=retest_traces)
    branch_traces = build_branch_traces(candidate_traces)
    trajectory_summary = build_trajectory_summary_v2(
        rows,
        candidate_traces=candidate_traces,
        branch_traces=branch_traces,
        episode_summary=episode_summary,
        run_metadata=run_metadata,
    )
    hindsight_labels = build_hindsight_labels_v2(
        step_logs=step_logs,
        candidate_traces=candidate_traces,
        branch_traces=branch_traces,
        trajectory_summary=trajectory_summary,
    )
    candidate_ranking_examples = build_candidate_ranking_examples_v2(
        candidate_traces=candidate_traces,
        hindsight_labels=hindsight_labels,
        branch_traces=branch_traces,
        trajectory_summary=trajectory_summary,
    )
    risk_prediction_examples = build_risk_prediction_examples_v2(
        step_logs=step_logs,
        hindsight_labels=hindsight_labels,
        trajectory_summary=trajectory_summary,
    )
    return {
        "schema_version": "eca_modeling_logs.v2",
        "step_logs": step_logs,
        "candidate_traces": candidate_traces,
        "branch_traces": branch_traces,
        "trajectory_summary": trajectory_summary,
        "hindsight_labels": hindsight_labels,
        "candidate_ranking_examples": candidate_ranking_examples,
        "risk_prediction_examples": risk_prediction_examples,
    }


def build_modeling_logs_from_run_dir(run_dir: Path) -> dict[str, Any]:
    step_rows = _read_jsonl(run_dir / "eca_step_metrics.jsonl")
    if not step_rows:
        step_rows = _read_jsonl(run_dir / "eca_anxiety.jsonl")
    episode_summary = _read_json(run_dir / "eca_episode_summary.json")
    retest_traces = _read_jsonl(run_dir / "retest_traces.jsonl")
    return build_modeling_logs(
        step_rows,
        episode_summary=episode_summary if isinstance(episode_summary, Mapping) else None,
        retest_traces=retest_traces,
    )


def write_modeling_logs(
    run_dir: Path,
    *,
    step_rows: Iterable[Mapping[str, Any]] | None = None,
    episode_summary: Mapping[str, Any] | None = None,
    retest_traces: Iterable[Mapping[str, Any]] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if step_rows is None:
        bundle = build_modeling_logs_from_run_dir(run_dir)
    else:
        if retest_traces is None:
            retest_traces = _read_jsonl(run_dir / "retest_traces.jsonl")
        bundle = build_modeling_logs(
            step_rows,
            episode_summary=episode_summary,
            retest_traces=retest_traces,
            run_metadata=run_metadata,
        )

    step_path = run_dir / "eca_step_logs_v2.jsonl"
    with step_path.open("w", encoding="utf-8") as handle:
        for row in bundle["step_logs"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    candidate_path = run_dir / "eca_candidate_traces.json"
    candidate_path.write_text(
        json.dumps(bundle["candidate_traces"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    branch_path = run_dir / "eca_branch_traces.json"
    branch_path.write_text(
        json.dumps(bundle["branch_traces"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path = run_dir / "eca_trajectory_summary_v2.json"
    summary_path.write_text(
        json.dumps(bundle["trajectory_summary"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    labels_path = run_dir / "eca_hindsight_labels_v2.json"
    labels_path.write_text(
        json.dumps(bundle["hindsight_labels"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    ranking_path = run_dir / "eca_candidate_ranking_examples.jsonl"
    with ranking_path.open("w", encoding="utf-8") as handle:
        for row in bundle["candidate_ranking_examples"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    risk_path = run_dir / "eca_risk_prediction_examples.jsonl"
    with risk_path.open("w", encoding="utf-8") as handle:
        for row in bundle["risk_prediction_examples"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "eca_modeling_manifest.v2",
        "step_logs": step_path.name,
        "candidate_traces": candidate_path.name,
        "branch_traces": branch_path.name,
        "trajectory_summary": summary_path.name,
        "hindsight_labels": labels_path.name,
        "candidate_ranking_examples": ranking_path.name,
        "risk_prediction_examples": risk_path.name,
    }
    manifest_path = run_dir / "eca_modeling_logs_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {key: str(value) for key, value in manifest.items()}
