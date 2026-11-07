"""Frame-only transition state variable diagnostics.

The sensor is intentionally observational. It summarizes before/after frame
deltas as toggle, selector, cycle, or rule-entry hypotheses. It does not choose
actions, emit plans, or authorize a solver.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable, Mapping

import numpy as np


TRANSITION_STATE_VARIABLE_PACKET_SCHEMA_VERSION = (
    "eca_transition_state_variable_sensor_packet.v1"
)
TRANSITION_STATE_VARIABLE_POSTERIOR_SCHEMA_VERSION = (
    "eca_transition_state_variable_posterior_packet.v1"
)
TRANSITION_STATE_VARIABLE_TRACE_SENSOR_SCHEMA_VERSION = (
    "eca_transition_state_variable_diagnostic_sensor.v1"
)
TRANSITION_STATE_VARIABLE_SCORE_CLOSURE_POSTERIOR_SCHEMA_VERSION = (
    "eca_transition_state_variable_score_closure_posterior.v1"
)
TRANSITION_STATE_VARIABLE_SCORE_RETEST_CANDIDATE_MINER_SCHEMA_VERSION = (
    "eca_transition_state_variable_score_retest_candidate_miner.v1"
)
TRANSITION_STATE_VARIABLE_SCORE_RETEST_PAIRED_DIFFERENTIAL_SCHEMA_VERSION = (
    "eca_transition_state_variable_score_retest_paired_differential.v1"
)
TRANSITION_STATE_VARIABLE_SKIP_PROBE_COUNTERFACTUAL_SCHEMA_VERSION = (
    "eca_transition_state_variable_skip_probe_counterfactual.v1"
)
TRANSITION_STATE_VARIABLE_TRANSITION_KEY_ABLATION_SCHEMA_VERSION = (
    "eca_transition_state_variable_transition_key_ablation.v1"
)
TRANSITION_STATE_VARIABLE_TRANSITION_SIGNATURE_ABLATION_SCHEMA_VERSION = (
    "eca_transition_state_variable_transition_signature_ablation.v1"
)
TRANSITION_STATE_VARIABLE_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION = (
    "eca_transition_state_variable_guarded_planner_connection_sensor.v1"
)


def build_transition_state_variable_packet(
    before_frame: Any,
    after_frame: Any | None = None,
    *,
    game_id: str = "",
    state_hash: str = "",
    step: int = 0,
    action_name: str = "",
    action_payload: Mapping[str, Any] | None = None,
    score_delta: int = 0,
) -> dict[str, Any]:
    before = _as_frame(before_frame)
    after = _as_frame(after_frame) if after_frame is not None else None
    delta = _frame_delta(before, after)
    candidate_types = set(str(row) for row in delta["candidate_types"] if row)
    if bool(delta["toggle_candidate"]):
        candidate_types.add("toggle")
    if bool(delta["selector_candidate"]):
        candidate_types.add("selector")
    if bool(delta["rule_entry_candidate"]):
        candidate_types.add("rule_entry")
    if bool(delta["cycle_phase_candidate"].get("detected")):
        candidate_types.add("cycle_phase")
    if bool(delta["period_candidate"].get("detected")):
        candidate_types.add("period")
    return _sanitize(
        {
            "schema_version": TRANSITION_STATE_VARIABLE_PACKET_SCHEMA_VERSION,
            "sensor_family": "transition_state_variable_sensor",
            "game_id": str(game_id or "").strip().lower(),
            "state_hash": str(state_hash or _frame_hash(before)),
            "step": int(step),
            "changed_bbox_signature": str(delta["changed_bbox_signature"]),
            "changed_cell_count": int(delta["changed_cell_count"]),
            "changed_value_histogram": dict(delta["changed_value_histogram"]),
            "before_value_histogram": dict(delta["before_value_histogram"]),
            "after_value_histogram": dict(delta["after_value_histogram"]),
            "action_to_state_delta": {
                "action_name": str(action_name or ""),
                "payload": dict(action_payload or {}),
                "changed_cell_count": int(delta["changed_cell_count"]),
                "changed_bbox_signature": str(delta["changed_bbox_signature"]),
                "changed_value_histogram": dict(delta["changed_value_histogram"]),
            },
            "period_candidate": dict(delta["period_candidate"]),
            "cycle_phase_candidate": dict(delta["cycle_phase_candidate"]),
            "toggle_candidate": bool(delta["toggle_candidate"]),
            "selector_candidate": bool(delta["selector_candidate"]),
            "rule_entry_candidate": bool(delta["rule_entry_candidate"]),
            "terminal_positive_observed": bool(int(score_delta) > 0),
            "score_delta": int(score_delta),
            "candidate_types": sorted(candidate_types),
        }
    )


def validate_transition_state_variable_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if str(packet.get("schema_version", "") or "") != TRANSITION_STATE_VARIABLE_PACKET_SCHEMA_VERSION:
        reasons.append("schema_version_mismatch")
    if str(packet.get("sensor_family", "") or "") != "transition_state_variable_sensor":
        reasons.append("sensor_family_mismatch")
    candidate_types = [str(row) for row in list(packet.get("candidate_types") or []) if row]
    changed_cell_count = _safe_int(packet.get("changed_cell_count"))
    if changed_cell_count < 0:
        reasons.append("invalid_changed_cell_count")
    if changed_cell_count <= 0 and not candidate_types:
        reasons.append("missing_transition_evidence")
    if bool(packet.get("toggle_candidate", False)) and "toggle" not in candidate_types:
        reasons.append("toggle_candidate_missing_type")
    if bool(packet.get("selector_candidate", False)) and "selector" not in candidate_types:
        reasons.append("selector_candidate_missing_type")
    if bool(packet.get("rule_entry_candidate", False)) and "rule_entry" not in candidate_types:
        reasons.append("rule_entry_candidate_missing_type")
    return {
        "schema_version": "eca_transition_state_variable_validation.v1",
        "accepted": not reasons,
        "rejected_reasons": reasons,
        "family": "transition_state_variable",
        "mechanic_state_type": _mechanic_state_type(packet),
        "changed_cell_count": changed_cell_count,
        "candidate_types": candidate_types,
        "goal_link_type": "state_variable_progress",
    }


def build_transition_state_variable_posterior_packet(
    packet: Mapping[str, Any],
    *,
    plan_completed: bool = False,
    score_delta: int | None = None,
) -> dict[str, Any]:
    validation = validate_transition_state_variable_packet(packet)
    changed = _safe_int(packet.get("changed_cell_count")) > 0
    delta = _safe_int(packet.get("score_delta")) if score_delta is None else int(score_delta)
    score_positive = delta > 0
    if not bool(validation.get("accepted", False)):
        posterior = "rejected"
        falsified_reason = "descriptor_validation_failed"
    elif score_positive or changed:
        posterior = "confirmed"
        falsified_reason = ""
    elif bool(plan_completed):
        posterior = "falsified"
        falsified_reason = "no_state_variable_transition_after_completed_plan"
    else:
        posterior = "pending"
        falsified_reason = ""
    return {
        "schema_version": TRANSITION_STATE_VARIABLE_POSTERIOR_SCHEMA_VERSION,
        "sensor_family": "transition_state_variable_sensor",
        "family": "transition_state_variable",
        "mechanic_state_type": str(validation.get("mechanic_state_type", "")),
        "goal_link_type": "state_variable_progress",
        "score_delta": int(delta),
        "plan_completed": bool(plan_completed),
        "validation": validation,
        "objective_progress": bool(score_positive or changed),
        "scoreless_objective_progress": bool(changed and not score_positive),
        "posterior": posterior,
        "falsified_reason": falsified_reason,
    }


def summarize_transition_state_variable_packets(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 12,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    validation_counts: Counter[str] = Counter()
    candidate_type_counts: Counter[str] = Counter()
    posterior_counts: Counter[str] = Counter()
    changed_cell_sum = 0
    score_delta_sum = 0
    samples: list[dict[str, Any]] = []
    for packet in packet_rows:
        validation = validate_transition_state_variable_packet(packet)
        accepted = bool(validation.get("accepted"))
        validation_counts["accepted" if accepted else "rejected"] += 1
        changed_cell_sum += _safe_int(packet.get("changed_cell_count"))
        score_delta_sum += _safe_int(packet.get("score_delta"))
        for candidate_type in list(packet.get("candidate_types") or []):
            candidate_type_counts[str(candidate_type)] += 1
        posterior = build_transition_state_variable_posterior_packet(packet)
        posterior_counts[str(posterior.get("posterior", "") or "unknown")] += 1
        if len(samples) < sample_limit:
            samples.append(
                {
                    "game_id": str(packet.get("game_id", "") or ""),
                    "step": _safe_int(packet.get("step")),
                    "changed_cell_count": _safe_int(packet.get("changed_cell_count")),
                    "candidate_types": list(packet.get("candidate_types") or []),
                    "mechanic_state_type": str(validation.get("mechanic_state_type", "")),
                    "posterior": str(posterior.get("posterior", "")),
                    "score_delta": _safe_int(packet.get("score_delta")),
                    "changed_bbox_signature": str(
                        packet.get("changed_bbox_signature", "") or ""
                    ),
                }
            )
    return {
        "schema_version": TRANSITION_STATE_VARIABLE_TRACE_SENSOR_SCHEMA_VERSION,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
        "packet_count": int(len(packet_rows)),
        "accepted_packet_count": int(validation_counts.get("accepted", 0)),
        "rejected_packet_count": int(validation_counts.get("rejected", 0)),
        "changed_cell_sum": int(changed_cell_sum),
        "score_delta_sum": int(score_delta_sum),
        "candidate_type_counts": dict(sorted(candidate_type_counts.items())),
        "posterior_counts": dict(sorted(posterior_counts.items())),
        "samples": samples,
    }


def summarize_transition_state_variable_score_closure_packets(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    state_counts: Counter[str] = Counter()
    candidate_type_counts: Counter[str] = Counter()
    by_transition_key: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    accepted_packet_count = 0
    rejected_packet_count = 0
    score_closure_packet_count = 0
    scoreless_transition_packet_count = 0
    no_transition_packet_count = 0
    score_delta_sum = 0

    for packet in packet_rows:
        validation = validate_transition_state_variable_packet(packet)
        accepted = bool(validation.get("accepted"))
        if accepted:
            accepted_packet_count += 1
        else:
            rejected_packet_count += 1
        mechanic_state_type = str(validation.get("mechanic_state_type", "") or "")
        candidate_types = sorted(
            str(row) for row in list(packet.get("candidate_types") or []) if row
        )
        for candidate_type in candidate_types:
            candidate_type_counts[candidate_type] += 1
        changed_cell_count = _safe_int(packet.get("changed_cell_count"))
        score_delta = _safe_int(packet.get("score_delta"))
        score_delta_sum += score_delta
        score_positive = bool(score_delta > 0 or packet.get("terminal_positive_observed"))
        if not accepted:
            closure_state = "rejected_descriptor"
        elif score_positive:
            closure_state = "score_closure_confirmed"
            score_closure_packet_count += 1
        elif changed_cell_count > 0:
            closure_state = "state_transition_score_closure_unconfirmed"
            scoreless_transition_packet_count += 1
        else:
            closure_state = "no_transition_evidence"
            no_transition_packet_count += 1
        state_counts[closure_state] += 1

        transition_key = _transition_closure_key(
            packet,
            mechanic_state_type=mechanic_state_type,
            candidate_types=candidate_types,
        )
        entry = by_transition_key.setdefault(
            transition_key,
            {
                "transition_key": transition_key,
                "game_id": str(packet.get("game_id", "") or ""),
                "mechanic_state_type": mechanic_state_type,
                "candidate_types": candidate_types,
                "action_name": str(
                    dict(packet.get("action_to_state_delta") or {}).get(
                        "action_name", ""
                    )
                    or ""
                ),
                "observation_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "score_closure_count": 0,
                "scoreless_transition_count": 0,
                "no_transition_count": 0,
                "score_delta_sum": 0,
                "changed_cell_sum": 0,
            },
        )
        entry["observation_count"] += 1
        entry["accepted_count"] += int(accepted)
        entry["rejected_count"] += int(not accepted)
        entry["score_closure_count"] += int(accepted and score_positive)
        entry["scoreless_transition_count"] += int(
            accepted and not score_positive and changed_cell_count > 0
        )
        entry["no_transition_count"] += int(accepted and changed_cell_count <= 0)
        entry["score_delta_sum"] += int(score_delta)
        entry["changed_cell_sum"] += int(changed_cell_count)

        if len(samples) < sample_limit:
            samples.append(
                {
                    "transition_key": transition_key,
                    "game_id": str(packet.get("game_id", "") or ""),
                    "step": _safe_int(packet.get("step")),
                    "mechanic_state_type": mechanic_state_type,
                    "candidate_types": candidate_types,
                    "closure_state": closure_state,
                    "score_delta": int(score_delta),
                    "changed_cell_count": int(changed_cell_count),
                }
            )

    posterior_state_counts: Counter[str] = Counter()
    confirmed_entry_count = 0
    unconfirmed_entry_count = 0
    rejected_entry_count = 0
    for entry in by_transition_key.values():
        if int(entry.get("score_closure_count", 0)) > 0:
            posterior_state = "confirmed_score_closure"
            confirmed_entry_count += 1
        elif int(entry.get("accepted_count", 0)) > 0 and int(
            entry.get("scoreless_transition_count", 0)
        ) > 0:
            posterior_state = "unconfirmed_scoreless_transition"
            unconfirmed_entry_count += 1
        elif int(entry.get("rejected_count", 0)) >= int(entry.get("observation_count", 0)):
            posterior_state = "rejected_descriptor"
            rejected_entry_count += 1
        else:
            posterior_state = "insufficient_evidence"
            unconfirmed_entry_count += 1
        entry["posterior_state"] = posterior_state
        posterior_state_counts[posterior_state] += 1

    return {
        "schema_version": TRANSITION_STATE_VARIABLE_SCORE_CLOSURE_POSTERIOR_SCHEMA_VERSION,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
        "packet_count": int(len(packet_rows)),
        "accepted_packet_count": int(accepted_packet_count),
        "rejected_packet_count": int(rejected_packet_count),
        "score_closure_confirmed_packet_count": int(score_closure_packet_count),
        "scoreless_transition_packet_count": int(scoreless_transition_packet_count),
        "no_transition_packet_count": int(no_transition_packet_count),
        "score_delta_sum": int(score_delta_sum),
        "closure_state_counts": dict(sorted(state_counts.items())),
        "candidate_type_counts": dict(sorted(candidate_type_counts.items())),
        "posterior_entry_count": int(len(by_transition_key)),
        "confirmed_score_closure_entry_count": int(confirmed_entry_count),
        "unconfirmed_scoreless_transition_entry_count": int(unconfirmed_entry_count),
        "rejected_entry_count": int(rejected_entry_count),
        "posterior_state_counts": dict(sorted(posterior_state_counts.items())),
        "by_transition_key": {
            key: value for key, value in sorted(by_transition_key.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "confirmed_score_closure for the same transition key",
            "allowed_connection": "diagnostic_to_guarded_ranking_only",
            "action_generation_allowed": False,
        },
    }


def summarize_transition_state_variable_score_retest_candidates(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    posterior = summarize_transition_state_variable_score_closure_packets(
        packet_rows,
        sample_limit=sample_limit,
    )
    candidates: list[dict[str, Any]] = []
    blocked_keys: list[str] = []
    by_game: dict[str, Counter[str]] = {}
    for transition_key, entry_raw in sorted(
        dict(posterior.get("by_transition_key") or {}).items()
    ):
        if not isinstance(entry_raw, Mapping):
            continue
        entry = dict(entry_raw)
        state = str(entry.get("posterior_state", "") or "")
        game_id = str(entry.get("game_id", "") or "")
        by_game.setdefault(game_id, Counter())["transition_key_count"] += 1
        if state == "confirmed_score_closure":
            by_game[game_id]["already_score_closed_count"] += 1
            continue
        if state != "unconfirmed_scoreless_transition":
            by_game[game_id]["blocked_or_rejected_count"] += 1
            blocked_keys.append(str(transition_key))
            continue
        source_action = str(entry.get("action_name", "") or "")
        candidate = {
            "candidate_key": f"transition_score_retest:{transition_key}:ACTION5",
            "source_transition_key": str(transition_key),
            "game_id": game_id,
            "mechanic_state_type": str(entry.get("mechanic_state_type", "") or ""),
            "candidate_types": list(entry.get("candidate_types") or []),
            "source_action_name": source_action,
            "retest_action_name": "ACTION5",
            "candidate_kind": "post_transition_score_retest_probe",
            "probe_policy": "one_shot_protected_score_retest",
            "observation_count": _safe_int(entry.get("observation_count")),
            "scoreless_transition_count": _safe_int(
                entry.get("scoreless_transition_count")
            ),
            "score_closure_count": _safe_int(entry.get("score_closure_count")),
            "changed_cell_sum": _safe_int(entry.get("changed_cell_sum")),
            "admission_state": "diagnostic_candidate_only",
            "required_before_execution": (
                "focused opt-in sandbox must replay the transition key and then run "
                "one protected ACTION5 retest"
            ),
            "action_generation_allowed": False,
            "ranking_enabled": False,
        }
        candidates.append(candidate)
        by_game[game_id]["retest_candidate_count"] += 1

    candidates.sort(
        key=lambda row: (
            -_safe_int(row.get("observation_count")),
            str(row.get("game_id", "")),
            str(row.get("source_transition_key", "")),
        )
    )
    sampled_candidates = candidates[:sample_limit]
    return {
        "schema_version": (
            TRANSITION_STATE_VARIABLE_SCORE_RETEST_CANDIDATE_MINER_SCHEMA_VERSION
        ),
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
        "packet_count": int(len(packet_rows)),
        "input_posterior_schema_version": str(posterior.get("schema_version", "") or ""),
        "input_posterior_entry_count": _safe_int(
            posterior.get("posterior_entry_count")
        ),
        "confirmed_score_closure_entry_count": _safe_int(
            posterior.get("confirmed_score_closure_entry_count")
        ),
        "unconfirmed_scoreless_transition_entry_count": _safe_int(
            posterior.get("unconfirmed_scoreless_transition_entry_count")
        ),
        "retest_candidate_count": int(len(candidates)),
        "blocked_or_rejected_transition_key_count": int(len(blocked_keys)),
        "retest_action_name_counts": (
            {"ACTION5": int(len(candidates))} if candidates else {}
        ),
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "candidates": sampled_candidates,
        "blocked_or_rejected_transition_keys": blocked_keys[:sample_limit],
        "next_runtime_contract": {
            "env_flag": "ECA_ENABLE_TRANSITION_STATE_VARIABLE_SCORE_RETEST_PROBE",
            "required_before_execution": (
                "focused opt-in sandbox with max one ACTION5 retest per source "
                "transition key; no positive ranking until score closure is observed"
            ),
            "allowed_connection": "diagnostic_candidate_to_protected_probe_only",
            "action_generation_allowed": False,
            "positive_ranking_allowed": False,
        },
    }


def summarize_transition_state_variable_score_retest_paired_differential(
    probe_on_rows: Iterable[Mapping[str, Any]],
    probe_off_rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    on_rows = [dict(row) for row in probe_on_rows if isinstance(row, Mapping)]
    off_rows = [dict(row) for row in probe_off_rows if isinstance(row, Mapping)]
    on_by_game = _rows_by_game(on_rows)
    off_by_game = _rows_by_game(off_rows)
    runtime_on = _score_retest_runtime_summary(on_rows, sample_limit=sample_limit)
    runtime_off = _score_retest_runtime_summary(off_rows, sample_limit=sample_limit)
    game_ids = sorted(set(on_by_game) | set(off_by_game))
    by_game: dict[str, dict[str, Any]] = {}
    score_delta_sum = 0
    adapter_free_score_delta_sum = 0
    regression_count = 0
    positive_delta_count = 0
    for game_id in game_ids:
        on_row = on_by_game.get(game_id) or {}
        off_row = off_by_game.get(game_id) or {}
        on_score = _safe_int(on_row.get("final_score"))
        off_score = _safe_int(off_row.get("final_score"))
        delta = on_score - off_score
        score_delta_sum += delta
        on_adapter_free = not bool(on_row.get("uses_game_conditioned_primitive"))
        off_adapter_free = not bool(off_row.get("uses_game_conditioned_primitive"))
        if on_adapter_free and off_adapter_free:
            adapter_free_score_delta_sum += delta
        if delta < 0:
            regression_count += 1
        if delta > 0:
            positive_delta_count += 1
        by_game[game_id] = {
            "probe_on_final_score": on_score,
            "probe_off_final_score": off_score,
            "score_delta": delta,
            "probe_on_actions_taken": _safe_int(on_row.get("actions_taken")),
            "probe_off_actions_taken": _safe_int(off_row.get("actions_taken")),
            "probe_on_retest_execution_count": int(
                runtime_on.get("by_game", {}).get(game_id, {}).get("execution_count", 0)
            ),
            "probe_off_retest_execution_count": int(
                runtime_off.get("by_game", {}).get(game_id, {}).get("execution_count", 0)
            ),
            "adapter_free_pair": bool(on_adapter_free and off_adapter_free),
        }

    if not game_ids:
        promotion_state = "no_paired_rows"
    elif score_delta_sum > 0 and regression_count == 0 and runtime_on["execution_count"] > 0:
        promotion_state = "positive_paired_differential_no_regression"
    elif score_delta_sum > 0:
        promotion_state = "mixed_paired_differential"
    elif regression_count > 0:
        promotion_state = "negative_paired_differential"
    else:
        promotion_state = "no_paired_score_differential"

    return {
        "schema_version": (
            TRANSITION_STATE_VARIABLE_SCORE_RETEST_PAIRED_DIFFERENTIAL_SCHEMA_VERSION
        ),
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "probe_on_run_count": int(len(on_rows)),
        "probe_off_run_count": int(len(off_rows)),
        "paired_game_count": int(len(game_ids)),
        "probe_on_score_sum": sum(_safe_int(row.get("final_score")) for row in on_rows),
        "probe_off_score_sum": sum(_safe_int(row.get("final_score")) for row in off_rows),
        "score_sum_delta": int(score_delta_sum),
        "adapter_free_score_sum_delta": int(adapter_free_score_delta_sum),
        "positive_delta_game_count": int(positive_delta_count),
        "regression_game_count": int(regression_count),
        "runtime_on": runtime_on,
        "runtime_off": runtime_off,
        "by_game": by_game,
        "promotion_state": promotion_state,
        "next_planner_contract": {
            "required_before_ranking": (
                "repeat focused paired differential, breadth preservation, and all25 "
                "non-regression before using transition-state retests for ranking"
            ),
            "allowed_connection": "diagnostic_to_guarded_probe_admission_only",
            "action_generation_allowed": False,
        },
    }


def summarize_transition_state_variable_guarded_planner_connection(
    delayed_attribution: Mapping[str, Any] | None,
    admission_guard: Mapping[str, Any] | None = None,
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Shadow admission guard for transition-state planner connection.

    A transition key is shadow-admitted only when it has positive delayed score
    margin over controls and the retest admission path is non-displacing
    (`repeat_source_action_retest` only). The summary never enables ranking or
    action generation.
    """

    delayed = dict(delayed_attribution or {})
    guard = dict(admission_guard or {})
    by_transition_key = {
        str(key): dict(value)
        for key, value in dict(delayed.get("by_transition_key") or {}).items()
        if isinstance(value, Mapping)
    }
    guard_enabled = bool(guard.get("enabled"))
    guard_preflight_count = _safe_int(guard.get("preflight_count"))
    guard_execution_count = _safe_int(guard.get("execution_count"))
    action5_execution_count = _safe_int(
        guard.get("action5_score_retest_execution_count")
    )
    repeat_execution_count = _safe_int(
        guard.get("repeat_source_action_retest_execution_count")
    )

    repeat_signatures: Counter[str] = Counter()
    action5_signatures: Counter[str] = Counter()
    for sample in list(guard.get("samples") or []):
        if not isinstance(sample, Mapping) or not bool(sample.get("executed")):
            continue
        transition_key = str(sample.get("source_transition_key", "") or "")
        signature = str(sample.get("transition_signature", "") or "") or (
            transition_state_variable_transition_key_signature(transition_key)
            if transition_key
            else ""
        )
        if not signature:
            continue
        strategy = str(sample.get("retest_strategy", "") or "")
        if strategy == "repeat_source_action_retest":
            repeat_signatures[signature] += 1
        elif strategy == "action5_score_retest":
            action5_signatures[signature] += 1

    rows: list[dict[str, Any]] = []
    admitted_count = 0
    blocked_count = 0
    blocked_missing_positive_control_margin_count = 0
    blocked_control_matched_delayed_positive_count = 0
    blocked_missing_admission_guard_count = 0
    blocked_action5_risk_count = 0
    blocked_missing_non_displacing_admission_count = 0

    for transition_key, stats in sorted(by_transition_key.items()):
        signature = transition_state_variable_transition_key_signature(transition_key)
        positive_margin = _safe_int(stats.get("positive_control_margin_count")) > 0
        delayed_positive = (
            _safe_int(stats.get("delayed_positive_execution_count")) > 0
            or _safe_int(stats.get("delayed_score_delta_sum")) > 0
        )
        control_matched = (
            _safe_int(stats.get("control_matched_delayed_positive_count")) > 0
            or (
                delayed_positive
                and _safe_int(stats.get("positive_control_margin_count")) <= 0
                and _safe_int(stats.get("negative_control_max_score_delta_sum")) > 0
            )
        )
        signature_repeat_count = _safe_int(repeat_signatures.get(signature))
        signature_action5_count = _safe_int(action5_signatures.get(signature))
        if not signature_repeat_count and guard_execution_count == repeat_execution_count:
            signature_repeat_count = int(
                repeat_execution_count > 0
                and bool(guard.get("executed_transition_signature_counts"))
                and signature
                in dict(guard.get("executed_transition_signature_counts") or {})
            )
        non_displacing_admission = bool(
            guard_enabled
            and signature_repeat_count > 0
            and signature_action5_count == 0
            and action5_execution_count == 0
        )

        blocked_reasons: list[str] = []
        if not positive_margin:
            if control_matched:
                blocked_reasons.append("delayed_score_control_matched")
                blocked_control_matched_delayed_positive_count += 1
            else:
                blocked_reasons.append("missing_positive_control_margin")
                blocked_missing_positive_control_margin_count += 1
        if not guard_enabled or guard_preflight_count <= 0:
            blocked_reasons.append("missing_admission_guard_summary")
            blocked_missing_admission_guard_count += 1
        elif signature_action5_count > 0 or action5_execution_count > 0:
            blocked_reasons.append("action5_retest_risk")
            blocked_action5_risk_count += 1
        elif not non_displacing_admission:
            blocked_reasons.append("missing_non_displacing_repeat_source_admission")
            blocked_missing_non_displacing_admission_count += 1

        admitted = bool(positive_margin and non_displacing_admission and not blocked_reasons)
        if admitted:
            admitted_count += 1
            connection_state = "planner_connection_admitted_shadow"
        else:
            blocked_count += 1
            if "delayed_score_control_matched" in blocked_reasons:
                connection_state = (
                    "planner_connection_blocked_control_matched_delayed_score"
                )
            elif "missing_positive_control_margin" in blocked_reasons:
                connection_state = (
                    "planner_connection_blocked_missing_positive_control_margin"
                )
            elif "missing_admission_guard_summary" in blocked_reasons:
                connection_state = "planner_connection_blocked_missing_admission_guard"
            elif "action5_retest_risk" in blocked_reasons:
                connection_state = "planner_connection_blocked_action5_retest_risk"
            else:
                connection_state = (
                    "planner_connection_blocked_missing_non_displacing_admission"
                )

        rows.append(
            {
                "source_transition_key": transition_key,
                "transition_signature": signature,
                "connection_state": connection_state,
                "admitted_shadow": admitted,
                "blocked_reasons": blocked_reasons,
                "positive_control_margin_count": _safe_int(
                    stats.get("positive_control_margin_count")
                ),
                "delayed_positive_execution_count": _safe_int(
                    stats.get("delayed_positive_execution_count")
                ),
                "control_matched_delayed_positive_count": _safe_int(
                    stats.get("control_matched_delayed_positive_count")
                ),
                "delayed_score_delta_sum": _safe_int(
                    stats.get("delayed_score_delta_sum")
                ),
                "negative_control_max_score_delta_sum": _safe_int(
                    stats.get("negative_control_max_score_delta_sum")
                ),
                "join_margin_sum": _safe_int(stats.get("join_margin_sum")),
                "repeat_source_action_retest_admission_count": signature_repeat_count,
                "action5_score_retest_execution_count": signature_action5_count,
            }
        )

    reviewed_count = len(rows)
    if admitted_count:
        promotion_state = "transition_state_variable_guarded_planner_connection_admitted_shadow"
    elif blocked_control_matched_delayed_positive_count:
        promotion_state = "transition_state_variable_guarded_planner_connection_blocked_control_matched_delayed_score"
    elif blocked_missing_positive_control_margin_count:
        promotion_state = "transition_state_variable_guarded_planner_connection_blocked_missing_positive_control_margin"
    elif blocked_missing_admission_guard_count:
        promotion_state = "transition_state_variable_guarded_planner_connection_blocked_missing_admission_guard"
    elif blocked_action5_risk_count:
        promotion_state = "transition_state_variable_guarded_planner_connection_blocked_action5_retest_risk"
    elif reviewed_count:
        promotion_state = "transition_state_variable_guarded_planner_connection_blocked"
    elif delayed:
        promotion_state = "transition_state_variable_guarded_planner_connection_no_transition_keys"
    else:
        promotion_state = "transition_state_variable_guarded_planner_connection_no_evidence"

    return _sanitize(
        {
            "schema_version": (
                TRANSITION_STATE_VARIABLE_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "shadow_only": True,
            "default_off": True,
            "planner_connection_reviewed_transition_count": int(reviewed_count),
            "planner_connection_admitted_transition_count": int(admitted_count),
            "planner_connection_blocked_transition_count": int(blocked_count),
            "positive_control_margin_count": _safe_int(
                delayed.get("positive_control_margin_count")
            ),
            "delayed_positive_execution_count": _safe_int(
                delayed.get("delayed_positive_execution_count")
            ),
            "control_matched_delayed_positive_count": _safe_int(
                delayed.get("control_matched_delayed_positive_count")
            ),
            "blocked_missing_positive_control_margin_count": int(
                blocked_missing_positive_control_margin_count
            ),
            "blocked_control_matched_delayed_positive_count": int(
                blocked_control_matched_delayed_positive_count
            ),
            "blocked_missing_admission_guard_count": int(
                blocked_missing_admission_guard_count
            ),
            "blocked_action5_retest_risk_count": int(blocked_action5_risk_count),
            "blocked_missing_non_displacing_admission_count": int(
                blocked_missing_non_displacing_admission_count
            ),
            "admission_guard_enabled": bool(guard_enabled),
            "admission_guard_preflight_count": int(guard_preflight_count),
            "admission_guard_execution_count": int(guard_execution_count),
            "repeat_source_action_retest_execution_count": int(
                repeat_execution_count
            ),
            "action5_score_retest_execution_count": int(action5_execution_count),
            "promotion_state": promotion_state,
            "by_transition_key": rows[:sample_limit],
            "next_planner_contract": {
                "required_before_runtime_connection": (
                    "positive control margin, non-displacing repeat-source "
                    "admission, focused preservation, and all25 non-regression"
                ),
                "allowed_connection": (
                    "shadow_admission_only_until_breadth_and_all25_non_regression"
                ),
                "action_generation_allowed": False,
            },
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_transition_state_variable_skip_probe_counterfactual(
    probe_on_rows: Iterable[Mapping[str, Any]],
    skip_probe_rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize score deltas for same-prefix runs with and without retest probes."""

    on_rows = [dict(row) for row in probe_on_rows if isinstance(row, Mapping)]
    skip_rows = [dict(row) for row in skip_probe_rows if isinstance(row, Mapping)]
    on_by_key = _rows_by_counterfactual_key(on_rows)
    skip_by_key = _rows_by_counterfactual_key(skip_rows)
    runtime_on = _score_retest_runtime_summary(on_rows, sample_limit=sample_limit)
    runtime_skip = _score_retest_runtime_summary(skip_rows, sample_limit=sample_limit)

    pair_keys = sorted(set(on_by_key) | set(skip_by_key))
    by_pair: dict[str, dict[str, Any]] = {}
    by_transition_key: dict[str, Counter[str]] = {}
    samples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    score_delta_sum = 0
    adapter_free_score_delta_sum = 0

    for pair_key in pair_keys:
        on_row = on_by_key.get(pair_key) or {}
        skip_row = skip_by_key.get(pair_key) or {}
        game_id = _row_game_id(on_row or skip_row)
        variant = str((on_row or skip_row).get("variant", "") or "canonical_full_plan")
        seed = _safe_int((on_row or skip_row).get("seed"))
        prefix_length = _safe_int((on_row or skip_row).get("prefix_length"))
        on_score = _safe_int(on_row.get("final_score"))
        skip_score = _safe_int(skip_row.get("final_score"))
        score_delta = on_score - skip_score
        score_delta_sum += score_delta
        on_adapter_free = not bool(on_row.get("uses_game_conditioned_primitive"))
        skip_adapter_free = not bool(skip_row.get("uses_game_conditioned_primitive"))
        adapter_free_pair = bool(on_adapter_free and skip_adapter_free)
        if adapter_free_pair:
            adapter_free_score_delta_sum += score_delta
        on_execution_count = int(
            runtime_on.get("by_game", {}).get(game_id, {}).get("execution_count", 0)
        )
        skip_execution_count = int(
            runtime_skip.get("by_game", {}).get(game_id, {}).get("execution_count", 0)
        )
        if pair_key not in on_by_key:
            state = "missing_probe_on_row"
        elif pair_key not in skip_by_key:
            state = "missing_skip_probe_row"
        elif score_delta > 0 and on_execution_count > 0 and skip_execution_count == 0:
            state = "score_improves_with_probe"
        elif score_delta < 0:
            state = "score_regresses_with_probe"
        elif on_execution_count > 0:
            state = "probe_execution_without_score_delta"
        else:
            state = "no_probe_execution_or_no_score_difference"
        counts[f"counterfactual_state:{state}"] += 1
        if score_delta > 0:
            counts["positive_counterfactual_pair_count"] += 1
        if score_delta < 0:
            counts["regression_pair_count"] += 1
        if adapter_free_pair:
            counts["adapter_free_pair_count"] += 1
        if on_execution_count > 0:
            counts["probe_on_execution_pair_count"] += 1
        if skip_execution_count > 0:
            counts["skip_probe_execution_pair_count"] += 1
        if state == "score_improves_with_probe":
            counts["score_supported_execution_count"] += on_execution_count
        elif on_execution_count > 0:
            counts["unsupported_execution_count"] += on_execution_count

        executed_keys = _score_retest_executed_transition_keys(on_row)
        for transition_key in executed_keys:
            counter = by_transition_key.setdefault(transition_key, Counter())
            counter["execution_count"] += 1
            counter[f"counterfactual_state:{state}"] += 1
            counter["score_delta_sum"] += score_delta
            if state == "score_improves_with_probe":
                counter["game_level_supported_execution_count"] += 1
            elif on_execution_count > 0:
                counter["unsupported_execution_count"] += 1

        by_pair[pair_key] = {
            "game_id": game_id,
            "variant": variant,
            "seed": int(seed),
            "prefix_length": int(prefix_length),
            "probe_on_final_score": int(on_score),
            "skip_probe_final_score": int(skip_score),
            "score_delta": int(score_delta),
            "probe_on_actions_taken": _safe_int(on_row.get("actions_taken")),
            "skip_probe_actions_taken": _safe_int(skip_row.get("actions_taken")),
            "probe_on_retest_execution_count": int(on_execution_count),
            "skip_probe_retest_execution_count": int(skip_execution_count),
            "adapter_free_pair": adapter_free_pair,
            "counterfactual_state": state,
            "executed_transition_keys": executed_keys[:sample_limit],
        }
        if len(samples) < sample_limit:
            samples.append(by_pair[pair_key])

    paired_count = int(len(pair_keys))
    positive_pairs = int(counts.get("positive_counterfactual_pair_count", 0))
    regressions = int(counts.get("regression_pair_count", 0))
    supported_exec = int(counts.get("score_supported_execution_count", 0))
    if paired_count <= 0:
        promotion_state = "no_counterfactual_pairs"
    elif supported_exec > 0 and score_delta_sum > 0 and regressions == 0:
        promotion_state = "positive_skip_probe_counterfactual_no_regression"
    elif score_delta_sum > 0 and regressions > 0:
        promotion_state = "positive_skip_probe_counterfactual_with_regressions"
    elif regressions > 0:
        promotion_state = "negative_skip_probe_counterfactual"
    else:
        promotion_state = "no_skip_probe_score_differential"

    return {
        "schema_version": TRANSITION_STATE_VARIABLE_SKIP_PROBE_COUNTERFACTUAL_SCHEMA_VERSION,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "probe_on_run_count": int(len(on_rows)),
        "skip_probe_run_count": int(len(skip_rows)),
        "paired_key_count": paired_count,
        "adapter_free_pair_count": int(counts.get("adapter_free_pair_count", 0)),
        "probe_on_score_sum": sum(_safe_int(row.get("final_score")) for row in on_rows),
        "skip_probe_score_sum": sum(
            _safe_int(row.get("final_score")) for row in skip_rows
        ),
        "score_sum_delta": int(score_delta_sum),
        "adapter_free_score_sum_delta": int(adapter_free_score_delta_sum),
        "positive_counterfactual_pair_count": positive_pairs,
        "regression_pair_count": regressions,
        "score_supported_execution_count": supported_exec,
        "unsupported_execution_count": int(counts.get("unsupported_execution_count", 0)),
        "skip_probe_execution_pair_count": int(
            counts.get("skip_probe_execution_pair_count", 0)
        ),
        "state_counts": {
            key.split("counterfactual_state:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("counterfactual_state:")
        },
        "runtime_on": runtime_on,
        "runtime_skip_probe": runtime_skip,
        "by_pair": by_pair,
        "by_transition_key": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(by_transition_key.items())
        },
        "samples": samples,
        "promotion_state": promotion_state,
        "next_planner_contract": {
            "required_before_ranking": (
                "same-prefix skip-probe support plus per-transition ablation, "
                "breadth preservation, and all25 non-regression"
            ),
            "allowed_connection": "diagnostic_to_guarded_admission_only",
            "action_generation_allowed": False,
        },
    }


def summarize_transition_state_variable_transition_key_ablation(
    baseline_rows: Iterable[Mapping[str, Any]],
    ablation_rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize score loss when specific transition retest keys are denylisted."""

    baseline = [dict(row) for row in baseline_rows if isinstance(row, Mapping)]
    ablations = [dict(row) for row in ablation_rows if isinstance(row, Mapping)]
    baseline_by_key = _rows_by_counterfactual_key(baseline)
    runtime_baseline = _score_retest_runtime_summary(
        baseline, sample_limit=sample_limit
    )
    runtime_ablation = _score_retest_runtime_summary(
        ablations, sample_limit=sample_limit
    )
    by_transition_key: dict[str, Counter[str]] = {}
    samples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    score_loss_sum = 0
    regression_loss_sum = 0

    for row in ablations:
        pair_key = _counterfactual_pair_key(row)
        baseline_row = baseline_by_key.get(pair_key) or {}
        game_id = _row_game_id(row or baseline_row)
        ablated_key = str(
            row.get("ablation_transition_key")
            or row.get("ablated_transition_key")
            or row.get("transition_key_ablation")
            or ""
        ).strip()
        if not ablated_key:
            ablated_key = "unknown"
        baseline_score = _safe_int(baseline_row.get("final_score"))
        ablation_score = _safe_int(row.get("final_score"))
        score_loss = baseline_score - ablation_score
        score_loss_sum += score_loss
        if score_loss < 0:
            regression_loss_sum += score_loss
        baseline_adapter_free = not bool(
            baseline_row.get("uses_game_conditioned_primitive")
        )
        ablation_adapter_free = not bool(row.get("uses_game_conditioned_primitive"))
        adapter_free_pair = bool(baseline_adapter_free and ablation_adapter_free)
        baseline_exec = len(_score_retest_executed_transition_keys(baseline_row))
        ablation_exec = len(_score_retest_executed_transition_keys(row))
        if pair_key not in baseline_by_key:
            state = "missing_baseline_row"
        elif score_loss > 0:
            state = "score_drops_when_key_ablated"
        elif score_loss < 0:
            state = "score_improves_when_key_ablated"
        elif baseline_exec > ablation_exec:
            state = "key_blocked_without_score_loss"
        else:
            state = "no_score_or_execution_difference"
        counts[f"ablation_state:{state}"] += 1
        if adapter_free_pair:
            counts["adapter_free_pair_count"] += 1
        if score_loss > 0:
            counts["positive_score_loss_pair_count"] += 1
        if score_loss < 0:
            counts["regression_pair_count"] += 1
        if state == "score_drops_when_key_ablated":
            counts["causal_supported_transition_key_count"] += 1
        counter = by_transition_key.setdefault(ablated_key, Counter())
        counter["ablation_pair_count"] += 1
        counter[f"ablation_state:{state}"] += 1
        counter["score_loss_sum"] += score_loss
        counter["baseline_score_sum"] += baseline_score
        counter["ablation_score_sum"] += ablation_score
        counter["baseline_execution_count_sum"] += baseline_exec
        counter["ablation_execution_count_sum"] += ablation_exec
        if adapter_free_pair:
            counter["adapter_free_pair_count"] += 1
        if state == "score_drops_when_key_ablated":
            counter["causal_supported_count"] += 1
        sample = {
            "game_id": game_id,
            "pair_key": pair_key,
            "ablation_transition_key": ablated_key,
            "baseline_final_score": int(baseline_score),
            "ablation_final_score": int(ablation_score),
            "score_loss": int(score_loss),
            "baseline_actions_taken": _safe_int(baseline_row.get("actions_taken")),
            "ablation_actions_taken": _safe_int(row.get("actions_taken")),
            "baseline_retest_execution_count": int(baseline_exec),
            "ablation_retest_execution_count": int(ablation_exec),
            "adapter_free_pair": adapter_free_pair,
            "ablation_state": state,
        }
        if len(samples) < sample_limit:
            samples.append(sample)

    positive_loss = int(counts.get("positive_score_loss_pair_count", 0))
    regressions = int(counts.get("regression_pair_count", 0))
    if not ablations:
        promotion_state = "no_ablation_rows"
    elif positive_loss > 0 and regressions == 0:
        promotion_state = "positive_transition_key_ablation_no_regression"
    elif positive_loss > 0:
        promotion_state = "mixed_transition_key_ablation"
    elif regressions > 0:
        promotion_state = "negative_transition_key_ablation"
    else:
        promotion_state = "no_transition_key_ablation_score_loss"

    return {
        "schema_version": TRANSITION_STATE_VARIABLE_TRANSITION_KEY_ABLATION_SCHEMA_VERSION,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "baseline_run_count": int(len(baseline)),
        "ablation_run_count": int(len(ablations)),
        "adapter_free_pair_count": int(counts.get("adapter_free_pair_count", 0)),
        "baseline_score_sum": sum(_safe_int(row.get("final_score")) for row in baseline),
        "ablation_score_sum": sum(_safe_int(row.get("final_score")) for row in ablations),
        "score_loss_sum": int(score_loss_sum),
        "regression_loss_sum": int(regression_loss_sum),
        "positive_score_loss_pair_count": positive_loss,
        "regression_pair_count": regressions,
        "causal_supported_transition_key_count": int(
            counts.get("causal_supported_transition_key_count", 0)
        ),
        "state_counts": {
            key.split("ablation_state:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("ablation_state:")
        },
        "runtime_baseline": runtime_baseline,
        "runtime_ablation": runtime_ablation,
        "by_transition_key": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(by_transition_key.items())
        },
        "samples": samples,
        "promotion_state": promotion_state,
        "next_planner_contract": {
            "required_before_ranking": (
                "transition-key ablation support, breadth preservation, and all25 "
                "non-regression before using transition-state retests for ranking"
            ),
            "allowed_connection": "diagnostic_to_guarded_admission_only",
            "action_generation_allowed": False,
        },
    }


def summarize_transition_state_variable_transition_signature_ablation(
    baseline_rows: Iterable[Mapping[str, Any]],
    ablation_rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize transition-key ablations after stripping game identity.

    Phase623 localized useful retests to full keys such as
    ``r11l|cycle_phase_transition|cycle_phase|ACTION6`` and
    ``vc33|cycle_phase_transition|cycle_phase|ACTION6``. That is still too
    game-specific for a general sensor. This summarizer groups the same
    ablation evidence by the generic signature
    ``cycle_phase_transition|cycle_phase|ACTION6``.
    """

    baseline = [dict(row) for row in baseline_rows if isinstance(row, Mapping)]
    ablations = [dict(row) for row in ablation_rows if isinstance(row, Mapping)]
    baseline_by_key = _rows_by_counterfactual_key(baseline)
    by_signature: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    score_loss_sum = 0
    regression_loss_sum = 0

    for row in ablations:
        pair_key = _counterfactual_pair_key(row)
        baseline_row = baseline_by_key.get(pair_key) or {}
        game_id = _row_game_id(row or baseline_row)
        ablated_key = str(
            row.get("ablation_transition_key")
            or row.get("ablated_transition_key")
            or row.get("transition_key_ablation")
            or ""
        ).strip()
        if not ablated_key:
            ablated_key = "unknown"
        signature = transition_state_variable_transition_key_signature(ablated_key)
        baseline_score = _safe_int(baseline_row.get("final_score"))
        ablation_score = _safe_int(row.get("final_score"))
        score_loss = baseline_score - ablation_score
        score_loss_sum += score_loss
        if score_loss < 0:
            regression_loss_sum += score_loss
        baseline_adapter_free = not bool(
            baseline_row.get("uses_game_conditioned_primitive")
        )
        ablation_adapter_free = not bool(row.get("uses_game_conditioned_primitive"))
        adapter_free_pair = bool(baseline_adapter_free and ablation_adapter_free)
        baseline_exec = len(_score_retest_executed_transition_keys(baseline_row))
        ablation_exec = len(_score_retest_executed_transition_keys(row))
        if pair_key not in baseline_by_key:
            state = "missing_baseline_row"
        elif score_loss > 0:
            state = "score_drops_when_signature_ablated"
        elif score_loss < 0:
            state = "score_improves_when_signature_ablated"
        elif baseline_exec > ablation_exec:
            state = "signature_blocked_without_score_loss"
        else:
            state = "no_score_or_execution_difference"

        counts[f"ablation_state:{state}"] += 1
        if adapter_free_pair:
            counts["adapter_free_pair_count"] += 1
        if score_loss > 0:
            counts["positive_score_loss_pair_count"] += 1
        if score_loss < 0:
            counts["regression_pair_count"] += 1

        entry = by_signature.setdefault(
            signature,
            {
                "transition_signature": signature,
                "ablation_pair_count": 0,
                "adapter_free_pair_count": 0,
                "score_loss_sum": 0,
                "baseline_score_sum": 0,
                "ablation_score_sum": 0,
                "baseline_execution_count_sum": 0,
                "ablation_execution_count_sum": 0,
                "positive_score_loss_pair_count": 0,
                "no_loss_pair_count": 0,
                "regression_pair_count": 0,
                "positive_score_loss_game_ids": set(),
                "observed_game_ids": set(),
                "source_transition_keys": set(),
            },
        )
        entry["ablation_pair_count"] += 1
        entry["score_loss_sum"] += score_loss
        entry["baseline_score_sum"] += baseline_score
        entry["ablation_score_sum"] += ablation_score
        entry["baseline_execution_count_sum"] += baseline_exec
        entry["ablation_execution_count_sum"] += ablation_exec
        entry["observed_game_ids"].add(game_id)
        entry["source_transition_keys"].add(ablated_key)
        if adapter_free_pair:
            entry["adapter_free_pair_count"] += 1
        if score_loss > 0:
            entry["positive_score_loss_pair_count"] += 1
            entry["positive_score_loss_game_ids"].add(game_id)
        elif score_loss < 0:
            entry["regression_pair_count"] += 1
        else:
            entry["no_loss_pair_count"] += 1

        sample = {
            "game_id": game_id,
            "pair_key": pair_key,
            "ablation_transition_key": ablated_key,
            "transition_signature": signature,
            "baseline_final_score": int(baseline_score),
            "ablation_final_score": int(ablation_score),
            "score_loss": int(score_loss),
            "baseline_retest_execution_count": int(baseline_exec),
            "ablation_retest_execution_count": int(ablation_exec),
            "adapter_free_pair": adapter_free_pair,
            "ablation_state": state,
        }
        if len(samples) < sample_limit:
            samples.append(sample)

    serialized_by_signature: dict[str, dict[str, Any]] = {}
    supported_signatures: list[str] = []
    no_loss_signatures: list[str] = []
    regression_signatures: list[str] = []
    for signature, raw_entry in sorted(by_signature.items()):
        entry = dict(raw_entry)
        positive_games = sorted(str(v) for v in entry.pop("positive_score_loss_game_ids"))
        observed_games = sorted(str(v) for v in entry.pop("observed_game_ids"))
        source_keys = sorted(str(v) for v in entry.pop("source_transition_keys"))
        entry["positive_score_loss_game_ids"] = positive_games
        entry["positive_score_loss_game_count"] = int(len(positive_games))
        entry["observed_game_ids"] = observed_games
        entry["observed_game_count"] = int(len(observed_games))
        entry["source_transition_keys"] = source_keys[:sample_limit]
        supported = bool(
            int(entry.get("positive_score_loss_pair_count", 0)) > 0
            and int(entry.get("positive_score_loss_game_count", 0)) >= 2
            and int(entry.get("regression_pair_count", 0)) == 0
        )
        entry["cross_game_causal_supported"] = supported
        if supported:
            supported_signatures.append(signature)
        elif int(entry.get("regression_pair_count", 0)) > 0:
            regression_signatures.append(signature)
        elif int(entry.get("no_loss_pair_count", 0)) > 0:
            no_loss_signatures.append(signature)
        serialized_by_signature[signature] = entry

    positive_loss = int(counts.get("positive_score_loss_pair_count", 0))
    regressions = int(counts.get("regression_pair_count", 0))
    if not ablations:
        promotion_state = "no_ablation_rows"
    elif supported_signatures and regressions == 0:
        promotion_state = "cross_game_transition_signature_supported_no_regression"
    elif supported_signatures:
        promotion_state = "mixed_transition_signature_ablation"
    elif positive_loss > 0:
        promotion_state = "single_game_transition_signature_support_only"
    elif regressions > 0:
        promotion_state = "negative_transition_signature_ablation"
    else:
        promotion_state = "no_transition_signature_ablation_score_loss"

    return {
        "schema_version": (
            TRANSITION_STATE_VARIABLE_TRANSITION_SIGNATURE_ABLATION_SCHEMA_VERSION
        ),
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "game_identity_used_for_signature": False,
        "baseline_run_count": int(len(baseline)),
        "ablation_run_count": int(len(ablations)),
        "adapter_free_pair_count": int(counts.get("adapter_free_pair_count", 0)),
        "baseline_score_sum": sum(_safe_int(row.get("final_score")) for row in baseline),
        "ablation_score_sum": sum(_safe_int(row.get("final_score")) for row in ablations),
        "score_loss_sum": int(score_loss_sum),
        "regression_loss_sum": int(regression_loss_sum),
        "positive_score_loss_pair_count": positive_loss,
        "regression_pair_count": regressions,
        "transition_signature_count": int(len(serialized_by_signature)),
        "cross_game_causal_supported_signature_count": int(len(supported_signatures)),
        "cross_game_causal_supported_signatures": supported_signatures,
        "no_loss_signatures": no_loss_signatures,
        "regression_signatures": regression_signatures,
        "state_counts": {
            key.split("ablation_state:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("ablation_state:")
        },
        "by_transition_signature": serialized_by_signature,
        "samples": samples,
        "promotion_state": promotion_state,
        "next_planner_contract": {
            "required_before_ranking": (
                "cross-game signature ablation support plus breadth/all25 "
                "non-regression; full game-id keys stay diagnostic controls only"
            ),
            "allowed_connection": "diagnostic_to_guarded_admission_only",
            "action_generation_allowed": False,
        },
    }


def _frame_delta(before: np.ndarray, after: np.ndarray | None) -> dict[str, Any]:
    if after is None or before.shape != after.shape:
        return _empty_delta("no_after_frame")
    changed = before != after
    ys, xs = np.where(changed)
    changed_count = int(len(xs))
    if changed_count <= 0:
        return _empty_delta("no_change")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    before_values = [int(v) for v in before[changed].tolist()]
    after_values = [int(v) for v in after[changed].tolist()]
    pair_hist = Counter(f"{b}->{a}" for b, a in zip(before_values, after_values))
    before_hist = Counter(str(v) for v in before_values)
    after_hist = Counter(str(v) for v in after_values)
    colors = set(before_values) | set(after_values)
    reciprocal_toggle = _has_reciprocal_binary_swap(pair_hist)
    toggle_like = bool(reciprocal_toggle or (len(pair_hist) == 1 and 1 <= len(colors) <= 2))
    cycle_like = bool(len(colors) >= 3 and len(pair_hist) >= 2)
    bbox_w = x1 - x0 + 1
    bbox_h = y1 - y0 + 1
    selector_like = bool(changed_count <= 64 and (bbox_w <= 8 or bbox_h <= 8))
    rule_like = bool(changed_count <= 128 and (toggle_like or cycle_like))
    candidate_types: list[str] = []
    if toggle_like:
        candidate_types.append("toggle")
    if cycle_like:
        candidate_types.append("cycle_phase")
    if selector_like:
        candidate_types.append("selector")
    if rule_like:
        candidate_types.append("rule_entry")
    return {
        "changed_bbox_signature": (
            f"x0:{x0},y0:{y0},x1:{x1},y1:{y1},w:{bbox_w},h:{bbox_h},area:{changed_count}"
        ),
        "changed_cell_count": changed_count,
        "changed_value_histogram": dict(sorted(pair_hist.items())),
        "before_value_histogram": dict(sorted(before_hist.items())),
        "after_value_histogram": dict(sorted(after_hist.items())),
        "period_candidate": {
            "detected": bool(toggle_like),
            "period": 2 if toggle_like else 0,
            "reason": "binary_state_transition" if toggle_like else "not_binary",
        },
        "cycle_phase_candidate": {
            "detected": bool(cycle_like),
            "phase_transition_count": int(len(pair_hist)),
        },
        "toggle_candidate": bool(toggle_like),
        "selector_candidate": bool(selector_like),
        "rule_entry_candidate": bool(rule_like),
        "candidate_types": candidate_types,
    }


def _empty_delta(reason: str) -> dict[str, Any]:
    return {
        "changed_bbox_signature": "",
        "changed_cell_count": 0,
        "changed_value_histogram": {},
        "before_value_histogram": {},
        "after_value_histogram": {},
        "period_candidate": {"detected": False, "period": 0, "reason": reason},
        "cycle_phase_candidate": {"detected": False, "phase_transition_count": 0},
        "toggle_candidate": False,
        "selector_candidate": False,
        "rule_entry_candidate": False,
        "candidate_types": [],
    }


def _has_reciprocal_binary_swap(pair_hist: Mapping[str, int]) -> bool:
    if len(pair_hist) != 2:
        return False
    pairs: list[tuple[str, str]] = []
    for key in pair_hist:
        if "->" not in key:
            return False
        left, right = key.split("->", 1)
        pairs.append((left, right))
    return pairs[0] == (pairs[1][1], pairs[1][0])


def _mechanic_state_type(packet: Mapping[str, Any]) -> str:
    types = [str(row) for row in list(packet.get("candidate_types") or []) if row]
    if "toggle" in types:
        return "binary_toggle_transition"
    if "cycle_phase" in types:
        return "cycle_phase_transition"
    if "rule_entry" in types:
        return "rule_entry_transition"
    if "selector" in types:
        return "selector_transition"
    return "transition_state_variable"


def _transition_closure_key(
    packet: Mapping[str, Any],
    *,
    mechanic_state_type: str,
    candidate_types: Iterable[str],
) -> str:
    action = dict(packet.get("action_to_state_delta") or {})
    action_name = str(action.get("action_name", "") or "unknown")
    candidate_sig = "+".join(sorted(str(row) for row in candidate_types if row)) or "none"
    game_id = str(packet.get("game_id", "") or "unknown")
    return f"{game_id}|{mechanic_state_type}|{candidate_sig}|{action_name}"


def transition_state_variable_transition_key_signature(transition_key: Any) -> str:
    """Return the game-identity-free transition signature for a key.

    Runtime transition keys are currently emitted as
    ``game|mechanic_state_type|candidate_signature|action``. Older focused
    tests may use ``game|mechanic|action``. In both cases the leading game
    component is a diagnostic namespace, not part of the generalizable
    sensor signature.
    """

    key = str(transition_key or "").strip()
    if not key:
        return "unknown"
    parts = [part.strip() for part in key.split("|")]
    if len(parts) >= 4:
        signature = "|".join(parts[1:])
    elif len(parts) == 3 and parts[2].upper().startswith("ACTION"):
        signature = "|".join(parts[1:])
    else:
        signature = key
    return signature or "unknown"


def _as_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError("expected a 2D frame or frame stack")
    return arr.astype(int, copy=False)


def _frame_hash(frame: np.ndarray) -> str:
    arr = np.ascontiguousarray(frame.astype(np.int16, copy=False))
    digest = hashlib.sha1(arr.tobytes()).hexdigest()
    return digest[:16]


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _rows_by_game(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_game: dict[str, dict[str, Any]] = {}
    for row in rows:
        game_id = str(row.get("game_id") or row.get("game") or "").strip()
        if game_id:
            by_game[game_id] = dict(row)
    return by_game


def _row_game_id(row: Mapping[str, Any]) -> str:
    return str(row.get("game_id") or row.get("game") or "unknown").strip() or "unknown"


def _counterfactual_pair_key(row: Mapping[str, Any]) -> str:
    game_id = _row_game_id(row)
    variant = str(row.get("variant", "") or "canonical_full_plan").strip()
    seed = _safe_int(row.get("seed"))
    prefix_length = _safe_int(row.get("prefix_length"))
    return f"{game_id}|{variant}|seed:{seed}|prefix:{prefix_length}"


def _rows_by_counterfactual_key(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping):
            by_key[_counterfactual_pair_key(row)] = dict(row)
    return by_key


def _score_retest_executed_transition_keys(row: Mapping[str, Any]) -> list[str]:
    trace = row.get("trace")
    if not isinstance(trace, list):
        return []
    keys: list[str] = []
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        unsolved = item.get("unsolved_replan")
        if not isinstance(unsolved, Mapping):
            continue
        if str(unsolved.get("schema_version", "") or "") != (
            "eca_transition_state_variable_score_retest_probe.v1"
        ):
            continue
        executed = bool(
            unsolved.get("executed")
            or unsolved.get("used")
            or str(item.get("phase", "") or "") == "transition_state_score_retest_probe"
        )
        if not executed:
            continue
        transition_key = str(unsolved.get("source_transition_key", "") or "").strip()
        if transition_key:
            keys.append(transition_key)
    return keys


def _score_retest_runtime_summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    samples: list[dict[str, Any]] = []
    for row in rows:
        game_id = str(row.get("game_id") or row.get("game") or "unknown")
        by_game.setdefault(game_id, Counter())
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            if str(unsolved.get("schema_version", "") or "") != (
                "eca_transition_state_variable_score_retest_probe.v1"
            ):
                continue
            executed = bool(
                unsolved.get("executed")
                or unsolved.get("used")
                or str(item.get("phase", "") or "")
                == "transition_state_score_retest_probe"
            )
            reason = str(unsolved.get("reason", "") or "unknown")
            retest_strategy = str(unsolved.get("retest_strategy", "") or "")
            selected_action = str(
                unsolved.get("selected_action") or item.get("action") or ""
            )
            counts["preflight_count"] += 1
            by_game[game_id]["preflight_count"] += 1
            counts[f"reason:{reason}"] += 1
            by_game[game_id][f"reason:{reason}"] += 1
            if executed:
                counts["execution_count"] += 1
                by_game[game_id]["execution_count"] += 1
                if retest_strategy:
                    counts[f"execution_strategy:{retest_strategy}"] += 1
                    by_game[game_id][f"execution_strategy:{retest_strategy}"] += 1
                if selected_action:
                    counts[f"selected_action:{selected_action}"] += 1
                    by_game[game_id][f"selected_action:{selected_action}"] += 1
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "step": _safe_int(item.get("step")),
                        "phase": str(item.get("phase", "") or ""),
                        "action": str(item.get("action", "") or ""),
                        "reason": reason,
                        "retest_strategy": retest_strategy,
                        "selected_action": selected_action,
                        "source_transition_key": str(
                            unsolved.get("source_transition_key", "") or ""
                        ),
                        "executed": bool(executed),
                    }
                )
    return {
        "preflight_count": int(counts.get("preflight_count", 0)),
        "execution_count": int(counts.get("execution_count", 0)),
        "reason_counts": {
            key.split("reason:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("reason:")
        },
        "execution_strategy_counts": {
            key.split("execution_strategy:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("execution_strategy:")
        },
        "selected_action_counts": {
            key.split("selected_action:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("selected_action:")
        },
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "samples": samples,
    }


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


__all__ = [
    "TRANSITION_STATE_VARIABLE_PACKET_SCHEMA_VERSION",
    "TRANSITION_STATE_VARIABLE_POSTERIOR_SCHEMA_VERSION",
    "TRANSITION_STATE_VARIABLE_SCORE_CLOSURE_POSTERIOR_SCHEMA_VERSION",
    "TRANSITION_STATE_VARIABLE_SCORE_RETEST_CANDIDATE_MINER_SCHEMA_VERSION",
    "TRANSITION_STATE_VARIABLE_SCORE_RETEST_PAIRED_DIFFERENTIAL_SCHEMA_VERSION",
    "TRANSITION_STATE_VARIABLE_SKIP_PROBE_COUNTERFACTUAL_SCHEMA_VERSION",
    "TRANSITION_STATE_VARIABLE_TRANSITION_SIGNATURE_ABLATION_SCHEMA_VERSION",
    "TRANSITION_STATE_VARIABLE_TRANSITION_KEY_ABLATION_SCHEMA_VERSION",
    "TRANSITION_STATE_VARIABLE_TRACE_SENSOR_SCHEMA_VERSION",
    "build_transition_state_variable_packet",
    "build_transition_state_variable_posterior_packet",
    "summarize_transition_state_variable_guarded_planner_connection",
    "summarize_transition_state_variable_score_closure_packets",
    "summarize_transition_state_variable_skip_probe_counterfactual",
    "summarize_transition_state_variable_transition_signature_ablation",
    "summarize_transition_state_variable_transition_key_ablation",
    "summarize_transition_state_variable_score_retest_paired_differential",
    "summarize_transition_state_variable_score_retest_candidates",
    "summarize_transition_state_variable_packets",
    "transition_state_variable_transition_key_signature",
    "validate_transition_state_variable_packet",
]
