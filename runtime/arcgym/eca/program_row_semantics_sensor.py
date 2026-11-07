"""Frame-only program row / run-button semantics diagnostics.

The sensor summarizes observed program-panel edits from public before/after
frames. It does not synthesize programs, choose rows, or authorize planner
ranking.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, deque
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROGRAM_ROW_SEMANTICS_PACKET_SCHEMA_VERSION = (
    "eca_program_row_semantics_packet.v1"
)
PROGRAM_ROW_SEMANTICS_DIAGNOSTIC_SENSOR_SCHEMA_VERSION = (
    "eca_program_row_semantics_diagnostic_sensor.v1"
)
PROGRAM_ROW_SEMANTICS_POSTERIOR_SCHEMA_VERSION = (
    "eca_program_row_semantics_posterior.v1"
)
PROGRAM_ROW_SEMANTICS_CANDIDATE_MINER_SCHEMA_VERSION = (
    "eca_program_row_semantics_candidate_miner.v1"
)
PROGRAM_ROW_SEMANTICS_GUARDED_PROBE_OUTCOME_SCHEMA_VERSION = (
    "eca_program_row_semantics_guarded_probe_outcome_sensor.v1"
)
PROGRAM_ROW_SEMANTICS_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION = (
    "eca_program_row_semantics_counterfactual_control_sensor.v1"
)
PROGRAM_ROW_SEMANTICS_POLARITY_FILL_PROPOSAL_SCHEMA_VERSION = (
    "eca_program_row_semantics_polarity_fill_proposal.v1"
)
PROGRAM_ROW_SEMANTICS_EDGE_BIT_PAIR_PROPOSAL_SCHEMA_VERSION = (
    "eca_program_row_semantics_edge_bit_pair_proposal.v1"
)
PROGRAM_ROW_SEMANTICS_TRANSLATION_BFS_PROPOSAL_SCHEMA_VERSION = (
    "eca_program_row_semantics_translation_bfs_proposal.v1"
)
PROGRAM_ROW_SEMANTICS_SCALE_TRANSLATION_BFS_PROPOSAL_SCHEMA_VERSION = (
    "eca_program_row_semantics_scale_translation_bfs_proposal.v1"
)
PROGRAM_ROW_SEMANTICS_SHAPE_TRANSFORM_BFS_PROPOSAL_SCHEMA_VERSION = (
    "eca_program_row_semantics_shape_transform_bfs_proposal.v1"
)
PROGRAM_ROW_SEMANTICS_CHECKPOINT_BFS_PROPOSAL_SCHEMA_VERSION = (
    "eca_program_row_semantics_checkpoint_bfs_proposal.v1"
)
PROGRAM_ROW_SEMANTICS_ALL25_NON_REGRESSION_SCHEMA_VERSION = (
    "eca_program_row_semantics_all25_non_regression_sensor.v1"
)


def build_program_row_semantics_packet(
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
    """Build an observational program-row semantics packet."""

    before = _as_frame(before_frame)
    after = _as_frame(after_frame) if after_frame is not None else before
    action_payload = dict(action_payload or {})
    selected_site = _selected_site(action_payload)
    action = _normalize_action_name(action_name)
    program_before = _program_panel_observation(before)
    program_after = _program_panel_observation(after)
    change = _change_observation(before, after)
    row_centers = _merge_centers(
        [
            *[float(row["centroid"][0]) for row in program_before["program_cells"]],
            *[float(row["centroid"][0]) for row in program_after["program_cells"]],
            *[float(row["centroid"][0]) for row in change["program_change_components"]],
        ]
    )
    bit_centers = _merge_centers(
        [
            *[float(row["centroid"][1]) for row in program_before["program_cells"]],
            *[float(row["centroid"][1]) for row in program_after["program_cells"]],
            *[float(row["centroid"][1]) for row in change["program_change_components"]],
        ]
    )
    selected_program_cell = _selected_program_cell(
        before,
        after,
        selected_site=selected_site,
        row_centers=row_centers,
        bit_centers=bit_centers,
        program_change_components=change["program_change_components"],
    )
    selected_run_region = _selected_run_button_region(
        selected_site,
        program_before["run_button_regions"],
        program_after["run_button_regions"],
    )
    program_bit_toggle_candidate = bool(
        action == "ACTION6"
        and selected_program_cell is not None
        and bool(selected_program_cell.get("changed"))
        and bool(selected_program_cell.get("program_color_pair"))
    )
    run_button_candidate = bool(action == "ACTION6" and selected_run_region is not None)
    program_execution_candidate = bool(
        run_button_candidate
        and (
            int(change["result_region_changed_cell_count"]) > 0
            or int(score_delta) > 0
        )
    )
    program_state_delta_candidate = bool(
        program_bit_toggle_candidate or program_execution_candidate
    )
    score_closure_candidate = bool(
        int(score_delta) > 0
        and (program_bit_toggle_candidate or run_button_candidate or program_execution_candidate)
    )
    candidate_types: set[str] = set()
    if program_bit_toggle_candidate:
        candidate_types.add("program_bit_toggle")
    if run_button_candidate:
        candidate_types.add("run_button")
    if program_execution_candidate:
        candidate_types.add("program_execution")
    if program_state_delta_candidate:
        candidate_types.add("program_state_delta")
    if score_closure_candidate:
        candidate_types.add("score_closure")

    panel_bbox = _union_bbox(
        [
            *[row["bbox"] for row in program_before["program_cells"]],
            *[row["bbox"] for row in program_after["program_cells"]],
            *[row["bbox"] for row in program_before["run_button_regions"]],
            *[row["bbox"] for row in program_after["run_button_regions"]],
            *[row["bbox"] for row in change["program_change_components"]],
        ]
    )

    return _sanitize(
        {
            "schema_version": PROGRAM_ROW_SEMANTICS_PACKET_SCHEMA_VERSION,
            "sensor_family": "program_row_semantics_sensor",
            "game_id": str(game_id or "").strip().lower(),
            "state_hash": str(state_hash or _frame_hash(before)),
            "step": int(step),
            "action_name": action,
            "action_payload": action_payload,
            "selected_site": list(selected_site) if selected_site is not None else None,
            "score_delta": int(score_delta),
            "frame_shape": [int(before.shape[0]), int(before.shape[1])],
            "program_panel_bbox": panel_bbox,
            "row_centers": [round(float(v), 3) for v in row_centers],
            "bit_centers": [round(float(v), 3) for v in bit_centers],
            "row_count_estimate": int(len(row_centers)),
            "bit_count_estimate": int(len(bit_centers)),
            "program_cells_before": program_before["program_cells"][:96],
            "program_cells_after": program_after["program_cells"][:96],
            "program_cell_count_before": int(len(program_before["program_cells"])),
            "program_cell_count_after": int(len(program_after["program_cells"])),
            "active_program_bit_count_before": int(
                program_before["active_program_bit_count"]
            ),
            "active_program_bit_count_after": int(
                program_after["active_program_bit_count"]
            ),
            "active_program_bit_count_delta": int(
                program_after["active_program_bit_count"]
                - program_before["active_program_bit_count"]
            ),
            "run_button_regions_before": program_before["run_button_regions"][:8],
            "run_button_regions_after": program_after["run_button_regions"][:8],
            "selected_run_button_region": selected_run_region,
            "selected_program_cell": selected_program_cell,
            "changed_cell_count": int(change["changed_cell_count"]),
            "top_progress_changed_cell_count": int(
                change["top_progress_changed_cell_count"]
            ),
            "program_cell_changed_cell_count": int(
                change["program_cell_changed_cell_count"]
            ),
            "result_region_changed_cell_count": int(
                change["result_region_changed_cell_count"]
            ),
            "program_change_components": change["program_change_components"][:16],
            "program_bit_toggle_candidate": program_bit_toggle_candidate,
            "run_button_candidate": run_button_candidate,
            "program_execution_candidate": program_execution_candidate,
            "program_state_delta_candidate": program_state_delta_candidate,
            "score_closure_candidate": score_closure_candidate,
            "candidate_types": sorted(candidate_types),
            "diagnostic_only": True,
        }
    )


def validate_program_row_semantics_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if (
        str(packet.get("schema_version", "") or "")
        != PROGRAM_ROW_SEMANTICS_PACKET_SCHEMA_VERSION
    ):
        reasons.append("schema_version_mismatch")
    if str(packet.get("sensor_family", "") or "") != "program_row_semantics_sensor":
        reasons.append("sensor_family_mismatch")
    selected_site = packet.get("selected_site")
    if selected_site is not None and not _is_xy(selected_site):
        reasons.append("invalid_selected_site")
    row_count = _safe_int(packet.get("row_count_estimate"))
    bit_count = _safe_int(packet.get("bit_count_estimate"))
    if row_count < 0 or bit_count < 0:
        reasons.append("invalid_program_dimension_estimate")
    return {
        "schema_version": "eca_program_row_semantics_validation.v1",
        "accepted": not reasons,
        "rejected_reasons": reasons,
        "family": "program_row_semantics",
        "mechanic_state_type": _mechanic_state_type(packet),
        "goal_link_type": "program_panel_state_to_result",
        "candidate_types": list(packet.get("candidate_types") or []),
    }


def summarize_program_row_semantics_packets(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 12,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    candidate_type_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    changed_cell_count_sum = 0
    top_progress_changed_cell_count_sum = 0
    program_cell_changed_cell_count_sum = 0
    result_region_changed_cell_count_sum = 0
    active_program_bit_count_delta_sum = 0
    score_delta_sum = 0
    max_row_count = 0
    max_bit_count = 0

    for packet in packet_rows:
        validation = validate_program_row_semantics_packet(packet)
        accepted = bool(validation.get("accepted"))
        counts["accepted_packet_count" if accepted else "rejected_packet_count"] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["packet_count"] += 1
        action = str(packet.get("action_name", "") or "")
        if action == "ACTION6":
            counts["action6_packet_count"] += 1
            by_game[game_id]["action6_packet_count"] += 1
        for field, name in (
            ("program_bit_toggle_candidate", "program_bit_toggle_candidate_count"),
            ("run_button_candidate", "run_button_candidate_count"),
            ("program_execution_candidate", "program_execution_candidate_count"),
            ("program_state_delta_candidate", "program_state_delta_candidate_count"),
            ("score_closure_candidate", "score_closure_candidate_count"),
        ):
            if bool(packet.get(field)):
                counts[name] += 1
                by_game[game_id][name] += 1
        for candidate_type in list(packet.get("candidate_types") or []):
            candidate_type_counts[str(candidate_type)] += 1

        changed = _safe_int(packet.get("changed_cell_count"))
        top_changed = _safe_int(packet.get("top_progress_changed_cell_count"))
        program_changed = _safe_int(packet.get("program_cell_changed_cell_count"))
        result_changed = _safe_int(packet.get("result_region_changed_cell_count"))
        bit_delta = _safe_int(packet.get("active_program_bit_count_delta"))
        score_delta = _safe_int(packet.get("score_delta"))
        changed_cell_count_sum += changed
        top_progress_changed_cell_count_sum += top_changed
        program_cell_changed_cell_count_sum += program_changed
        result_region_changed_cell_count_sum += result_changed
        active_program_bit_count_delta_sum += bit_delta
        score_delta_sum += score_delta
        max_row_count = max(max_row_count, _safe_int(packet.get("row_count_estimate")))
        max_bit_count = max(max_bit_count, _safe_int(packet.get("bit_count_estimate")))

        if len(samples) < sample_limit:
            selected_cell = packet.get("selected_program_cell")
            selected_cell = selected_cell if isinstance(selected_cell, Mapping) else {}
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "action_name": action,
                    "selected_site": packet.get("selected_site"),
                    "candidate_types": list(packet.get("candidate_types") or []),
                    "row_index": selected_cell.get("row_index"),
                    "bit_index": selected_cell.get("bit_index"),
                    "changed_cell_count": changed,
                    "program_cell_changed_cell_count": program_changed,
                    "result_region_changed_cell_count": result_changed,
                    "active_program_bit_count_delta": bit_delta,
                    "score_delta": score_delta,
                }
            )

    return _sanitize(
        {
            "schema_version": PROGRAM_ROW_SEMANTICS_DIAGNOSTIC_SENSOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "accepted_packet_count": int(counts.get("accepted_packet_count", 0)),
            "rejected_packet_count": int(counts.get("rejected_packet_count", 0)),
            "action6_packet_count": int(counts.get("action6_packet_count", 0)),
            "program_bit_toggle_candidate_count": int(
                counts.get("program_bit_toggle_candidate_count", 0)
            ),
            "run_button_candidate_count": int(
                counts.get("run_button_candidate_count", 0)
            ),
            "program_execution_candidate_count": int(
                counts.get("program_execution_candidate_count", 0)
            ),
            "program_state_delta_candidate_count": int(
                counts.get("program_state_delta_candidate_count", 0)
            ),
            "score_closure_candidate_count": int(
                counts.get("score_closure_candidate_count", 0)
            ),
            "changed_cell_count_sum": int(changed_cell_count_sum),
            "top_progress_changed_cell_count_sum": int(
                top_progress_changed_cell_count_sum
            ),
            "program_cell_changed_cell_count_sum": int(
                program_cell_changed_cell_count_sum
            ),
            "result_region_changed_cell_count_sum": int(
                result_region_changed_cell_count_sum
            ),
            "active_program_bit_count_delta_sum": int(
                active_program_bit_count_delta_sum
            ),
            "score_delta_sum": int(score_delta_sum),
            "max_row_count_estimate": int(max_row_count),
            "max_bit_count_estimate": int(max_bit_count),
            "candidate_type_counts": dict(sorted(candidate_type_counts.items())),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "samples": samples,
            "next_planner_contract": {
                "required_before_ranking": (
                    "program row posterior with run-result closure attribution"
                ),
                "allowed_connection": (
                    "diagnostic_only_until_program_semantics_posterior_controls"
                ),
                "action_generation_allowed": False,
            },
        }
    )


def summarize_program_row_semantics_posterior(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize observed program edits and run-result closures.

    This is an attribution posterior only. It does not synthesize program rows,
    rank candidates, execute actions, or authorize planner promotion.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_row_bit: dict[str, Counter[str]] = {}
    by_run_button: dict[str, Counter[str]] = {}
    hypotheses: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    score_closure_delta_sum = 0
    program_execution_result_delta_sum = 0

    for index, packet in enumerate(packet_rows):
        validation = validate_program_row_semantics_packet(packet)
        game_id = str(packet.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["packet_count"] += 1
        if not bool(validation.get("accepted")):
            counts["rejected_packet_count"] += 1
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "packet_index": int(index),
                        "game_id": game_id,
                        "rejected_reasons": list(
                            validation.get("rejected_reasons") or []
                        ),
                    }
                )
            continue

        counts["accepted_packet_count"] += 1
        candidate_types = set(str(row) for row in list(packet.get("candidate_types") or []))
        score_delta = _safe_int(packet.get("score_delta"))
        result_delta = _safe_int(packet.get("result_region_changed_cell_count"))
        selected_cell = packet.get("selected_program_cell")
        selected_cell = selected_cell if isinstance(selected_cell, Mapping) else {}
        selected_run = packet.get("selected_run_button_region")
        selected_run = selected_run if isinstance(selected_run, Mapping) else {}

        if bool(packet.get("program_bit_toggle_candidate")):
            row_index = selected_cell.get("row_index")
            bit_index = selected_cell.get("bit_index")
            row_key = f"row{row_index}:bit{bit_index}"
            by_row_bit.setdefault(row_key, Counter())
            by_row_bit[row_key]["toggle_observation_count"] += 1
            by_game[game_id]["toggle_observation_count"] += 1
            counts["toggle_observation_count"] += 1
            if len(hypotheses) < sample_limit:
                hypotheses.append(
                    {
                        "hypothesis_type": "program_bit_toggle_relation",
                        "game_id": game_id,
                        "packet_index": int(index),
                        "step": _safe_int(packet.get("step")),
                        "row_index": row_index,
                        "bit_index": bit_index,
                        "selected_site": packet.get("selected_site"),
                        "active_program_bit_count_delta": _safe_int(
                            packet.get("active_program_bit_count_delta")
                        ),
                        "program_cell_changed_cell_count": _safe_int(
                            packet.get("program_cell_changed_cell_count")
                        ),
                        "posterior_state": "confirmed_observed_toggle",
                        "planner_connection_enabled": False,
                        "ranking_enabled": False,
                        "action_generation_allowed": False,
                    }
                )

        if bool(packet.get("run_button_candidate")):
            bbox = selected_run.get("bbox")
            run_key = "run_button:" + (
                ",".join(str(int(v)) for v in bbox[:4])
                if _is_bbox(bbox)
                else "unknown"
            )
            by_run_button.setdefault(run_key, Counter())
            by_run_button[run_key]["run_button_observation_count"] += 1
            by_game[game_id]["run_button_observation_count"] += 1
            counts["run_button_observation_count"] += 1
            posterior_state = (
                "confirmed_result_region_change"
                if bool(packet.get("program_execution_candidate"))
                else "observed_run_button_no_result_change"
            )
            if bool(packet.get("program_execution_candidate")):
                counts["program_execution_observation_count"] += 1
                by_run_button[run_key]["program_execution_observation_count"] += 1
                by_game[game_id]["program_execution_observation_count"] += 1
                program_execution_result_delta_sum += int(result_delta)
            if len(hypotheses) < sample_limit:
                hypotheses.append(
                    {
                        "hypothesis_type": "program_run_button_relation",
                        "game_id": game_id,
                        "packet_index": int(index),
                        "step": _safe_int(packet.get("step")),
                        "selected_site": packet.get("selected_site"),
                        "run_button_bbox": bbox if _is_bbox(bbox) else None,
                        "result_region_changed_cell_count": int(result_delta),
                        "posterior_state": posterior_state,
                        "planner_connection_enabled": False,
                        "ranking_enabled": False,
                        "action_generation_allowed": False,
                    }
                )

        if "score_closure" in candidate_types or bool(
            packet.get("score_closure_candidate")
        ):
            counts["score_closure_observation_count"] += 1
            by_game[game_id]["score_closure_observation_count"] += 1
            score_closure_delta_sum += int(score_delta)
            if len(hypotheses) < sample_limit:
                hypotheses.append(
                    {
                        "hypothesis_type": "program_score_closure_relation",
                        "game_id": game_id,
                        "packet_index": int(index),
                        "step": _safe_int(packet.get("step")),
                        "score_delta": int(score_delta),
                        "result_region_changed_cell_count": int(result_delta),
                        "posterior_state": (
                            "confirmed_score_closure"
                            if score_delta > 0
                            else "score_closure_candidate_no_score_delta"
                        ),
                        "planner_connection_enabled": False,
                        "ranking_enabled": False,
                        "action_generation_allowed": False,
                    }
                )

    toggle_count = int(counts.get("toggle_observation_count", 0))
    run_button_count = int(counts.get("run_button_observation_count", 0))
    execution_count = int(counts.get("program_execution_observation_count", 0))
    score_closure_count = int(counts.get("score_closure_observation_count", 0))
    if score_closure_count > 0:
        promotion_state = "program_row_score_closure_posterior"
    elif execution_count > 0:
        promotion_state = "program_row_execution_posterior"
    elif toggle_count > 0 or run_button_count > 0:
        promotion_state = "program_row_semantics_observed_posterior"
    elif packet_rows:
        promotion_state = "program_row_semantics_no_relation_observed"
    else:
        promotion_state = "program_row_semantics_no_packets"

    return _sanitize(
        {
            "schema_version": PROGRAM_ROW_SEMANTICS_POSTERIOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "default_off": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "accepted_packet_count": int(counts.get("accepted_packet_count", 0)),
            "rejected_packet_count": int(counts.get("rejected_packet_count", 0)),
            "toggle_observation_count": int(toggle_count),
            "run_button_observation_count": int(run_button_count),
            "program_execution_observation_count": int(execution_count),
            "score_closure_observation_count": int(score_closure_count),
            "score_closure_delta_sum": int(score_closure_delta_sum),
            "program_execution_result_delta_sum": int(
                program_execution_result_delta_sum
            ),
            "posterior_hypothesis_count": int(len(hypotheses)),
            "promotion_state": promotion_state,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_row_bit": {
                key: dict(sorted(counter.items()))
                for key, counter in sorted(by_row_bit.items())
            },
            "by_run_button": {
                key: dict(sorted(counter.items()))
                for key, counter in sorted(by_run_button.items())
            },
            "hypotheses": hypotheses[:sample_limit],
            "rejected": rejected[:sample_limit],
            "next_runtime_contract": {
                "required_before_candidate_mining": (
                    "candidate mining requires posterior hypotheses plus "
                    "counterfactual controls that distinguish score-bearing "
                    "program executions from scoreless row edits"
                ),
                "allowed_connection": "program_row_posterior_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_program_row_semantics_candidate_miner(
    packets: Iterable[Mapping[str, Any]],
    *,
    posterior_summary: Mapping[str, Any] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Mine guarded-probe candidate descriptions from program-row posterior."""

    posterior = (
        dict(posterior_summary)
        if isinstance(posterior_summary, Mapping)
        else summarize_program_row_semantics_posterior(
            packets,
            sample_limit=max(int(sample_limit), 128),
        )
    )
    hypotheses = [
        dict(row)
        for row in list(posterior.get("hypotheses") or [])
        if isinstance(row, Mapping)
    ]
    by_game: dict[str, Counter[str]] = {}
    by_kind: dict[str, Counter[str]] = {}
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for index, hypothesis in enumerate(hypotheses):
        game_id = str(hypothesis.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        kind = str(hypothesis.get("hypothesis_type", "") or "")
        candidate_kind = ""
        evidence_role = ""
        if kind == "program_bit_toggle_relation":
            candidate_kind = "program_bit_toggle_retest_candidate"
            evidence_role = "program_bit_toggle"
        elif kind == "program_run_button_relation":
            candidate_kind = "program_run_button_result_probe_candidate"
            evidence_role = "program_run_button"
        elif kind == "program_score_closure_relation":
            candidate_kind = "program_score_closure_replay_candidate"
            evidence_role = "program_score_closure"
        else:
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "posterior_hypothesis_index": int(index),
                        "game_id": game_id,
                        "rejected_reason": "unsupported_hypothesis_type",
                        "hypothesis_type": kind,
                    }
                )
            continue

        candidate_key = (
            f"program_row_semantics:{game_id}:{candidate_kind}:"
            f"step{_safe_int(hypothesis.get('step'))}:idx{int(index)}"
        )
        row = {
            "candidate_key": candidate_key,
            "candidate_kind": candidate_kind,
            "game_id": game_id,
            "step": _safe_int(hypothesis.get("step")),
            "source_action_name": "ACTION6",
            "selected_site": hypothesis.get("selected_site"),
            "evidence_role": evidence_role,
            "posterior_state": str(hypothesis.get("posterior_state", "") or ""),
            "row_index": hypothesis.get("row_index"),
            "bit_index": hypothesis.get("bit_index"),
            "run_button_bbox": hypothesis.get("run_button_bbox"),
            "score_delta": _safe_int(hypothesis.get("score_delta")),
            "requires_guarded_probe": True,
            "runtime_execution_enabled": False,
            "planner_connection_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
        candidates.append(row)
        by_game[game_id]["candidate_count"] += 1
        by_game[game_id][f"candidate_kind:{candidate_kind}"] += 1
        by_kind.setdefault(candidate_kind, Counter())
        by_kind[candidate_kind]["candidate_count"] += 1

    candidate_count = int(len(candidates))
    if candidate_count > 0:
        promotion_state = "program_row_semantics_candidate_miner_supported"
    elif hypotheses:
        promotion_state = "program_row_semantics_candidate_miner_blocked"
    else:
        promotion_state = "program_row_semantics_candidate_miner_no_posterior"

    return _sanitize(
        {
            "schema_version": PROGRAM_ROW_SEMANTICS_CANDIDATE_MINER_SCHEMA_VERSION,
            "diagnostic_only": True,
            "default_off": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "posterior_promotion_state": str(
                posterior.get("promotion_state", "") or ""
            ),
            "posterior_hypothesis_count": _safe_int(
                posterior.get("posterior_hypothesis_count")
            ),
            "candidate_count": int(candidate_count),
            "toggle_retest_candidate_count": int(
                by_kind.get("program_bit_toggle_retest_candidate", Counter()).get(
                    "candidate_count", 0
                )
            ),
            "run_button_probe_candidate_count": int(
                by_kind.get(
                    "program_run_button_result_probe_candidate", Counter()
                ).get("candidate_count", 0)
            ),
            "score_closure_replay_candidate_count": int(
                by_kind.get("program_score_closure_replay_candidate", Counter()).get(
                    "candidate_count", 0
                )
            ),
            "rejected_candidate_count": int(len(rejected)),
            "promotion_state": promotion_state,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_candidate_kind": {
                kind: dict(sorted(counter.items()))
                for kind, counter in sorted(by_kind.items())
            },
            "candidates": candidates[:sample_limit],
            "rejected": rejected[:sample_limit],
            "next_runtime_contract": {
                "required_before_guarded_probe": (
                    "candidate miner output must be validated by a default-off "
                    "guarded probe with scoreless-edit and wrong-row controls"
                ),
                "allowed_connection": "program_row_candidate_design_only",
                "action_generation_allowed": False,
            },
        }
    )


def build_program_row_semantics_polarity_fill_proposals(
    frame: Any,
    *,
    game_id: str = "",
    state_hash: str = "",
    known_program_colors: Iterable[int] | None = None,
    max_plan_steps: int = 24,
    sample_limit: int = 4,
) -> dict[str, Any]:
    """Build guarded all-one-polarity program-fill proposals from a frame.

    The proposal enumerates visible program-cell colors as candidate polarities
    and compiles each into setup toggles plus a run click. It does not infer a
    game-specific target program or embed a known solution.
    """

    arr = _as_frame(frame)
    packet = build_program_row_semantics_packet(
        arr,
        arr,
        game_id=game_id,
        state_hash=state_hash,
    )
    program_cells = [
        dict(row)
        for row in list(packet.get("program_cells_before") or [])
        if isinstance(row, Mapping)
    ]
    run_regions = [
        dict(row)
        for row in list(packet.get("run_button_regions_before") or [])
        if isinstance(row, Mapping)
    ]
    colors = {
        _safe_int(row.get("color"), -1)
        for row in program_cells
        if _safe_int(row.get("color"), -1) >= 0
    }
    if known_program_colors is not None:
        colors.update(
            _safe_int(color, -1)
            for color in known_program_colors
            if _safe_int(color, -1) >= 0
        )
    reasons: list[str] = []
    if not program_cells:
        reasons.append("no_program_cells")
    if not run_regions:
        reasons.append("no_run_region")
    if len(colors) < 2:
        reasons.append("single_program_color")
    if max_plan_steps <= 0:
        reasons.append("max_plan_steps_exhausted")

    proposals: list[dict[str, Any]] = []
    if not reasons:
        run_region = run_regions[0]
        run_site = _centroid_site(run_region.get("centroid"))
        if run_site is None:
            reasons.append("invalid_run_region_centroid")
        else:
            for target_color in sorted(colors):
                setup_steps: list[dict[str, Any]] = []
                for cell_index, cell in enumerate(program_cells):
                    color = _safe_int(cell.get("color"), -1)
                    if color == int(target_color):
                        continue
                    site = _centroid_site(cell.get("centroid"))
                    if site is None:
                        continue
                    setup_steps.append(
                        {
                            "action": "ACTION6",
                            "x": int(site[0]),
                            "y": int(site[1]),
                            "role": "setup_toggle",
                            "candidate_kind": "program_polarity_fill_setup_toggle",
                            "cell_index": int(cell_index),
                            "source_color": int(color),
                            "target_color": int(target_color),
                        }
                    )
                plan_steps = setup_steps + [
                    {
                        "action": "ACTION6",
                        "x": int(run_site[0]),
                        "y": int(run_site[1]),
                        "role": "run",
                        "candidate_kind": "program_polarity_fill_run_candidate",
                        "target_color": int(target_color),
                    }
                ]
                if len(plan_steps) > int(max_plan_steps):
                    continue
                proposals.append(
                    {
                        "candidate_key": (
                            "program_row_semantics_polarity_fill:"
                            f"{state_hash or _frame_hash(arr)}:"
                            f"target_color:{int(target_color)}"
                        ),
                        "candidate_kind": "program_polarity_fill_plan_candidate",
                        "game_id": str(game_id or "").strip().lower(),
                        "state_hash": str(state_hash or _frame_hash(arr)),
                        "target_color": int(target_color),
                        "program_cell_count": int(len(program_cells)),
                        "setup_toggle_count": int(len(setup_steps)),
                        "run_site": [int(run_site[0]), int(run_site[1])],
                        "plan_step_count": int(len(plan_steps)),
                        "plan": plan_steps,
                        "guarded_probe_required": True,
                        "planner_connection_enabled": False,
                        "ranking_enabled": False,
                        "positive_boost_enabled": False,
                        "action_generation_allowed": False,
                    }
                )

    proposals.sort(
        key=lambda row: (
            _safe_int(row.get("setup_toggle_count")),
            _safe_int(row.get("target_color")),
        )
    )
    if proposals:
        promotion_state = "program_row_polarity_fill_proposals_supported"
    elif reasons:
        promotion_state = "program_row_polarity_fill_proposals_blocked"
    else:
        promotion_state = "program_row_polarity_fill_proposals_empty"

    return _sanitize(
        {
            "schema_version": PROGRAM_ROW_SEMANTICS_POLARITY_FILL_PROPOSAL_SCHEMA_VERSION,
            "default_off": True,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "program_cell_count": int(len(program_cells)),
            "run_region_count": int(len(run_regions)),
            "candidate_color_count": int(len(colors)),
            "proposal_count": int(len(proposals)),
            "promotion_state": promotion_state,
            "rejected_reasons": reasons,
            "proposals": proposals[:sample_limit],
        }
    )


def build_program_row_semantics_edge_bit_pair_proposals(
    frame: Any,
    *,
    game_id: str = "",
    state_hash: str = "",
    active_color: int | None = None,
    max_plan_steps: int = 32,
    sample_limit: int = 4,
) -> dict[str, Any]:
    """Build guarded proposals that set panel edge bits across all rows.

    This is a frame-derived program-row hypothesis for small bit-vector program
    panels: if a panel has several columns and at least four bit positions, set
    the first and last bit active for every column, clear middle bits that are
    already active, then click the panel's run button. It does not use game IDs,
    level IDs, or a stored target program.
    """

    arr = _as_frame(frame)
    panel = _program_panel_observation(arr)
    program_cells = [
        dict(row)
        for row in list(panel.get("program_cells") or [])
        if isinstance(row, Mapping)
    ]
    run_regions = [
        dict(row)
        for row in list(panel.get("run_button_regions") or [])
        if isinstance(row, Mapping)
    ]
    colors = {
        _safe_int(row.get("color"), -1)
        for row in program_cells
        if _safe_int(row.get("color"), -1) >= 0
    }
    reasons: list[str] = []
    if not program_cells:
        reasons.append("no_program_cells")
    if not run_regions:
        reasons.append("no_run_region")
    if len(colors) < 2:
        reasons.append("single_program_color")
    if max_plan_steps <= 0:
        reasons.append("max_plan_steps_exhausted")

    active = _safe_int(active_color, -1)
    if active < 0 or active not in colors:
        if 5 in colors:
            active = 5
        else:
            active = max(colors) if colors else -1
    inactive_candidates = sorted(color for color in colors if color != active)
    inactive = inactive_candidates[0] if inactive_candidates else -1
    if active < 0 or inactive < 0:
        reasons.append("missing_binary_program_colors")

    x_centers = _merge_centers(
        [float(row["centroid"][0]) for row in program_cells],
        tolerance=2.0,
    )
    y_centers = _merge_centers(
        [float(row["centroid"][1]) for row in program_cells],
        tolerance=2.0,
    )
    panels = _split_program_x_panels(x_centers)
    if not panels:
        reasons.append("no_program_x_panel")
    if len(y_centers) < 4:
        reasons.append("insufficient_bit_axis")

    proposals: list[dict[str, Any]] = []
    if not reasons:
        state_key = str(state_hash or _frame_hash(arr))
        candidate_panels = _program_panels_with_run_regions(panels, run_regions)
        for panel_index, panel_row in enumerate(candidate_panels):
            columns = [float(v) for v in list(panel_row.get("columns") or [])]
            run_region = panel_row.get("run_region")
            run_region = run_region if isinstance(run_region, Mapping) else {}
            run_site = _centroid_site(run_region.get("centroid"))
            if len(columns) < 3 or run_site is None:
                continue
            plan_steps: list[dict[str, Any]] = []
            desired_by_bit = {
                0: int(active),
                len(y_centers) - 1: int(active),
            }
            for bit_index in range(1, len(y_centers) - 1):
                desired_by_bit[bit_index] = int(inactive)
            for col_index, x_center in enumerate(columns):
                for bit_index, y_center in enumerate(y_centers):
                    cell = _nearest_program_cell(
                        program_cells,
                        x_center=x_center,
                        y_center=float(y_center),
                    )
                    if cell is None:
                        continue
                    current_color = _safe_int(cell.get("color"), -1)
                    desired_color = int(desired_by_bit[bit_index])
                    if current_color == desired_color:
                        continue
                    site = _centroid_site(cell.get("centroid"))
                    if site is None:
                        continue
                    plan_steps.append(
                        {
                            "action": "ACTION6",
                            "x": int(site[0]),
                            "y": int(site[1]),
                            "role": "edge_bit_pair_setup_toggle",
                            "candidate_kind": "program_edge_bit_pair_setup_toggle",
                            "panel_index": int(panel_row.get("source_panel_index", panel_index)),
                            "column_index": int(col_index),
                            "bit_index": int(bit_index),
                            "source_color": int(current_color),
                            "target_color": int(desired_color),
                            "active_color": int(active),
                            "inactive_color": int(inactive),
                        }
                    )
            if not plan_steps:
                continue
            plan_steps.append(
                {
                    "action": "ACTION6",
                    "x": int(run_site[0]),
                    "y": int(run_site[1]),
                    "role": "run",
                    "candidate_kind": "program_edge_bit_pair_run_candidate",
                    "panel_index": int(panel_row.get("source_panel_index", panel_index)),
                    "active_color": int(active),
                }
            )
            if len(plan_steps) > int(max_plan_steps):
                continue
            panel_columns = [round(float(value), 3) for value in columns]
            proposals.append(
                {
                    "candidate_key": (
                        "program_row_semantics_edge_bit_pair:"
                        f"{state_key}:panel:{int(panel_row.get('source_panel_index', panel_index))}:"
                        f"active_color:{int(active)}"
                    ),
                    "candidate_kind": "program_edge_bit_pair_plan_candidate",
                    "game_id": str(game_id or "").strip().lower(),
                    "state_hash": state_key,
                    "active_color": int(active),
                    "inactive_color": int(inactive),
                    "panel_index": int(panel_row.get("source_panel_index", panel_index)),
                    "panel_columns": panel_columns,
                    "bit_centers": [round(float(value), 3) for value in y_centers],
                    "setup_toggle_count": int(len(plan_steps) - 1),
                    "run_site": [int(run_site[0]), int(run_site[1])],
                    "run_region_area": _safe_int(run_region.get("area")),
                    "plan_step_count": int(len(plan_steps)),
                    "plan": plan_steps,
                    "guarded_probe_required": True,
                    "planner_connection_enabled": False,
                    "ranking_enabled": False,
                    "positive_boost_enabled": False,
                    "action_generation_allowed": False,
                }
            )

    proposals.sort(
        key=lambda row: (
            -_safe_int(row.get("run_region_area")),
            _safe_int(row.get("setup_toggle_count")),
            _safe_int(row.get("panel_index")),
        )
    )
    if proposals:
        promotion_state = "program_row_edge_bit_pair_proposals_supported"
    elif reasons:
        promotion_state = "program_row_edge_bit_pair_proposals_blocked"
    else:
        promotion_state = "program_row_edge_bit_pair_proposals_empty"

    return _sanitize(
        {
            "schema_version": PROGRAM_ROW_SEMANTICS_EDGE_BIT_PAIR_PROPOSAL_SCHEMA_VERSION,
            "default_off": True,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "program_cell_count": int(len(program_cells)),
            "run_region_count": int(len(run_regions)),
            "candidate_panel_count": int(len(proposals)),
            "candidate_color_count": int(len(colors)),
            "active_color": int(active),
            "inactive_color": int(inactive),
            "promotion_state": promotion_state,
            "rejected_reasons": reasons,
            "proposals": proposals[:sample_limit],
        }
    )


def build_program_row_semantics_translation_bfs_proposals(
    frame: Any,
    *,
    game_id: str = "",
    state_hash: str = "",
    active_color: int | None = None,
    max_plan_steps: int = 32,
    sample_limit: int = 4,
) -> dict[str, Any]:
    """Build guarded program proposals from visible actor/target displacement.

    The proposal stays at the program-row abstraction: it reads the visible
    program panel, finds a same-colour actor/target pair in the associated
    playfield, runs a small grid BFS around visible blockers, compiles the
    resulting translation opcodes into bit toggles, and appends the panel run
    button. It does not use game IDs, level IDs, stored programs, or adapter
    state.
    """

    arr = _as_frame(frame)
    panel = _program_panel_observation(arr)
    program_cells = [
        dict(row)
        for row in list(panel.get("program_cells") or [])
        if isinstance(row, Mapping)
    ]
    run_regions = [
        dict(row)
        for row in list(panel.get("run_button_regions") or [])
        if isinstance(row, Mapping)
    ]
    colors = {
        _safe_int(row.get("color"), -1)
        for row in program_cells
        if _safe_int(row.get("color"), -1) >= 0
    }
    reasons: list[str] = []
    if not program_cells:
        reasons.append("no_program_cells")
    if not run_regions:
        reasons.append("no_run_region")
    if max_plan_steps <= 0:
        reasons.append("max_plan_steps_exhausted")

    active = _safe_int(active_color, -1)
    active_supplied = active >= 0
    if active < 0 or (active not in colors and not active_supplied):
        active = 5 if 5 in colors else max(colors) if colors else -1
    if active_supplied and active >= 0:
        colors.add(active)
    inactive_candidates = sorted(color for color in colors if color != active)
    inactive = inactive_candidates[0] if inactive_candidates else -1
    if active < 0 or inactive < 0:
        reasons.append("missing_binary_program_colors")

    x_centers = _merge_centers(
        [float(row["centroid"][0]) for row in program_cells],
        tolerance=2.0,
    )
    y_centers = _merge_centers(
        [float(row["centroid"][1]) for row in program_cells],
        tolerance=2.0,
    )
    panels = _split_program_x_panels(x_centers)
    if not panels:
        reasons.append("no_program_x_panel")
    if len(y_centers) < 4:
        reasons.append("insufficient_bit_axis")

    proposals: list[dict[str, Any]] = []
    actor_pair_count = 0
    blocker_count = 0
    if not reasons:
        state_key = str(state_hash or _frame_hash(arr))
        candidate_panels = _program_panels_with_run_regions(panels, run_regions)
        for panel_index, panel_row in enumerate(candidate_panels):
            columns = [float(v) for v in list(panel_row.get("columns") or [])]
            run_region = panel_row.get("run_region")
            run_region = run_region if isinstance(run_region, Mapping) else {}
            run_site = _centroid_site(run_region.get("centroid"))
            if len(columns) < 3 or run_site is None:
                continue
            playfield = _program_translation_playfield(
                arr,
                columns=columns,
                bit_centers=y_centers,
                run_region=run_region,
            )
            actor_pairs = _program_translation_actor_pairs(
                arr,
                playfield_bbox=playfield["bbox"],
            )
            actor_pair_count += len(actor_pairs)
            blockers = _program_translation_blockers(
                arr,
                playfield_bbox=playfield["bbox"],
                ignored_colors={0, 1, 4, 5, 9},
            )
            blocker_count += len(blockers)
            stride = _program_translation_stride(arr, playfield_bbox=playfield["bbox"])
            for pair_index, pair in enumerate(actor_pairs):
                op_sequence = _program_translation_bfs_ops(
                    start=pair["start_centroid"],
                    target=pair["target_centroid"],
                    actor_bbox=pair["start_bbox"],
                    blockers=blockers,
                    stride=stride,
                    max_ops=len(columns),
                )
                if not op_sequence:
                    continue
                desired = list(op_sequence)
                desired.extend([0] * (len(columns) - len(desired)))
                plan_steps = _compile_program_row_opcode_plan(
                    program_cells,
                    columns=columns,
                    bit_centers=y_centers,
                    desired_opcodes=desired,
                    active_color=active,
                    inactive_color=inactive,
                )
                plan_steps.append(
                    {
                        "action": "ACTION6",
                        "x": int(run_site[0]),
                        "y": int(run_site[1]),
                        "role": "run",
                        "candidate_kind": "program_translation_bfs_run_candidate",
                        "panel_index": int(panel_row.get("source_panel_index", panel_index)),
                        "active_color": int(active),
                    }
                )
                if len(plan_steps) > int(max_plan_steps):
                    continue
                proposals.append(
                    {
                        "candidate_key": (
                            "program_row_semantics_translation_bfs:"
                            f"{state_key}:panel:{int(panel_row.get('source_panel_index', panel_index))}:"
                            f"pair:{int(pair_index)}:ops:{','.join(str(int(op)) for op in desired)}"
                        ),
                        "candidate_kind": "program_translation_bfs_plan_candidate",
                        "game_id": str(game_id or "").strip().lower(),
                        "state_hash": state_key,
                        "active_color": int(active),
                        "inactive_color": int(inactive),
                        "panel_index": int(panel_row.get("source_panel_index", panel_index)),
                        "panel_columns": [round(float(value), 3) for value in columns],
                        "bit_centers": [round(float(value), 3) for value in y_centers],
                        "run_site": [int(run_site[0]), int(run_site[1])],
                        "run_region_area": _safe_int(run_region.get("area")),
                        "stride": int(stride),
                        "actor_color": int(pair["color"]),
                        "start_centroid": [
                            round(float(pair["start_centroid"][0]), 3),
                            round(float(pair["start_centroid"][1]), 3),
                        ],
                        "target_centroid": [
                            round(float(pair["target_centroid"][0]), 3),
                            round(float(pair["target_centroid"][1]), 3),
                        ],
                        "op_sequence": [int(op) for op in op_sequence],
                        "desired_opcodes": [int(op) for op in desired],
                        "setup_toggle_count": int(len(plan_steps) - 1),
                        "plan_step_count": int(len(plan_steps)),
                        "plan": plan_steps,
                        "guarded_probe_required": True,
                        "planner_connection_enabled": False,
                        "ranking_enabled": False,
                        "positive_boost_enabled": False,
                        "action_generation_allowed": False,
                    }
                )

    proposals.sort(
        key=lambda row: (
            len(list(row.get("op_sequence") or [])),
            _safe_int(row.get("setup_toggle_count")),
            -_safe_int(row.get("run_region_area")),
            _safe_int(row.get("panel_index")),
        )
    )
    if proposals:
        promotion_state = "program_row_translation_bfs_proposals_supported"
    elif reasons:
        promotion_state = "program_row_translation_bfs_proposals_blocked"
    else:
        promotion_state = "program_row_translation_bfs_proposals_empty"

    return _sanitize(
        {
            "schema_version": PROGRAM_ROW_SEMANTICS_TRANSLATION_BFS_PROPOSAL_SCHEMA_VERSION,
            "default_off": True,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "program_cell_count": int(len(program_cells)),
            "run_region_count": int(len(run_regions)),
            "candidate_panel_count": int(len(proposals)),
            "candidate_color_count": int(len(colors)),
            "actor_pair_count": int(actor_pair_count),
            "blocker_count": int(blocker_count),
            "active_color": int(active),
            "inactive_color": int(inactive),
            "proposal_count": int(len(proposals)),
            "promotion_state": promotion_state,
            "rejected_reasons": reasons,
            "proposals": proposals[:sample_limit],
        }
    )


def build_program_row_semantics_scale_translation_bfs_proposals(
    frame: Any,
    *,
    game_id: str = "",
    state_hash: str = "",
    active_color: int | None = None,
    max_plan_steps: int = 32,
    sample_limit: int = 4,
) -> dict[str, Any]:
    """Build guarded proposals for scale-then-translate program rows."""

    arr = _as_frame(frame)
    panel = _program_panel_observation(arr)
    program_cells = [
        dict(row)
        for row in list(panel.get("program_cells") or [])
        if isinstance(row, Mapping)
    ]
    run_regions = [
        dict(row)
        for row in list(panel.get("run_button_regions") or [])
        if isinstance(row, Mapping)
    ]
    colors = {
        _safe_int(row.get("color"), -1)
        for row in program_cells
        if _safe_int(row.get("color"), -1) >= 0
    }
    reasons: list[str] = []
    if not program_cells:
        reasons.append("no_program_cells")
    if not run_regions:
        reasons.append("no_run_region")
    if max_plan_steps <= 0:
        reasons.append("max_plan_steps_exhausted")

    active = _safe_int(active_color, -1)
    active_supplied = active >= 0
    if active < 0 or (active not in colors and not active_supplied):
        active = 5 if 5 in colors else max(colors) if colors else -1
    if active_supplied and active >= 0:
        colors.add(active)
    inactive_candidates = sorted(color for color in colors if color != active)
    inactive = inactive_candidates[0] if inactive_candidates else -1
    if active < 0 or inactive < 0:
        reasons.append("missing_binary_program_colors")

    x_centers = _merge_centers(
        [float(row["centroid"][0]) for row in program_cells],
        tolerance=2.0,
    )
    y_centers = _merge_centers(
        [float(row["centroid"][1]) for row in program_cells],
        tolerance=2.0,
    )
    panels = _split_program_x_panels(x_centers)
    if not panels:
        reasons.append("no_program_x_panel")
    if len(y_centers) < 4:
        reasons.append("insufficient_bit_axis")

    proposals: list[dict[str, Any]] = []
    actor_pair_count = 0
    if not reasons:
        state_key = str(state_hash or _frame_hash(arr))
        candidate_panels = _program_panels_with_run_regions(panels, run_regions)
        for panel_index, panel_row in enumerate(candidate_panels):
            columns = [float(v) for v in list(panel_row.get("columns") or [])]
            run_region = panel_row.get("run_region")
            run_region = run_region if isinstance(run_region, Mapping) else {}
            run_site = _centroid_site(run_region.get("centroid"))
            if len(columns) < 2 or run_site is None:
                continue
            playfield = _program_translation_playfield(
                arr,
                columns=columns,
                bit_centers=y_centers,
                run_region=run_region,
            )
            actor_pairs = _program_scale_translation_actor_pairs(
                arr,
                playfield_bbox=playfield["bbox"],
            )
            actor_pair_count += len(actor_pairs)
            stride = _program_translation_stride(arr, playfield_bbox=playfield["bbox"])
            for pair_index, pair in enumerate(actor_pairs):
                translation_ops = _program_scale_translation_bfs_ops(
                    start_bbox=pair["start_bbox"],
                    target_bbox=pair["target_bbox"],
                    stride=stride,
                    max_ops=max(0, len(columns) - 1),
                )
                if not translation_ops:
                    continue
                op_sequence = [int(pair["scale_opcode"])] + translation_ops
                desired = list(op_sequence)
                desired.extend([0] * (len(columns) - len(desired)))
                plan_steps = _compile_program_row_opcode_plan(
                    program_cells,
                    columns=columns,
                    bit_centers=y_centers,
                    desired_opcodes=desired,
                    active_color=active,
                    inactive_color=inactive,
                )
                for step in plan_steps:
                    step["role"] = "scale_translation_bfs_setup_toggle"
                    step["candidate_kind"] = "program_scale_translation_bfs_setup_toggle"
                plan_steps.append(
                    {
                        "action": "ACTION6",
                        "x": int(run_site[0]),
                        "y": int(run_site[1]),
                        "role": "run",
                        "candidate_kind": "program_scale_translation_bfs_run_candidate",
                        "panel_index": int(
                            panel_row.get("source_panel_index", panel_index)
                        ),
                        "active_color": int(active),
                    }
                )
                if len(plan_steps) > int(max_plan_steps):
                    continue
                proposals.append(
                    {
                        "candidate_key": (
                            "program_row_semantics_scale_translation_bfs:"
                            f"{state_key}:panel:{int(panel_row.get('source_panel_index', panel_index))}:"
                            f"pair:{int(pair_index)}:ops:{','.join(str(int(op)) for op in desired)}"
                        ),
                        "candidate_kind": "program_scale_translation_bfs_plan_candidate",
                        "game_id": str(game_id or "").strip().lower(),
                        "state_hash": state_key,
                        "active_color": int(active),
                        "inactive_color": int(inactive),
                        "panel_index": int(
                            panel_row.get("source_panel_index", panel_index)
                        ),
                        "panel_columns": [round(float(value), 3) for value in columns],
                        "bit_centers": [round(float(value), 3) for value in y_centers],
                        "run_site": [int(run_site[0]), int(run_site[1])],
                        "run_region_area": _safe_int(run_region.get("area")),
                        "stride": int(stride),
                        "actor_color": int(pair["color"]),
                        "scale_opcode": int(pair["scale_opcode"]),
                        "scale_relation": str(pair.get("scale_relation", "")),
                        "area_ratio": round(float(pair.get("area_ratio", 0.0) or 0.0), 3),
                        "vertical_delta": round(
                            float(pair.get("vertical_delta", 0.0) or 0.0), 3
                        ),
                        "start_bbox": [int(value) for value in pair["start_bbox"]],
                        "target_bbox": [int(value) for value in pair["target_bbox"]],
                        "start_centroid": [
                            round(float(pair["start_centroid"][0]), 3),
                            round(float(pair["start_centroid"][1]), 3),
                        ],
                        "target_centroid": [
                            round(float(pair["target_centroid"][0]), 3),
                            round(float(pair["target_centroid"][1]), 3),
                        ],
                        "translation_op_sequence": [int(op) for op in translation_ops],
                        "op_sequence": [int(op) for op in op_sequence],
                        "desired_opcodes": [int(op) for op in desired],
                        "setup_toggle_count": int(len(plan_steps) - 1),
                        "plan_step_count": int(len(plan_steps)),
                        "plan": plan_steps,
                        "guarded_probe_required": True,
                        "planner_connection_enabled": False,
                        "ranking_enabled": False,
                        "positive_boost_enabled": False,
                        "action_generation_allowed": False,
                    }
                )

    proposals.sort(
        key=lambda row: (
            len(list(row.get("op_sequence") or [])),
            _safe_int(row.get("setup_toggle_count")),
            -_safe_int(row.get("run_region_area")),
            _safe_int(row.get("panel_index")),
        )
    )
    if proposals:
        promotion_state = "program_row_scale_translation_bfs_proposals_supported"
    elif reasons:
        promotion_state = "program_row_scale_translation_bfs_proposals_blocked"
    else:
        promotion_state = "program_row_scale_translation_bfs_proposals_empty"

    return _sanitize(
        {
            "schema_version": PROGRAM_ROW_SEMANTICS_SCALE_TRANSLATION_BFS_PROPOSAL_SCHEMA_VERSION,
            "default_off": True,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "program_cell_count": int(len(program_cells)),
            "run_region_count": int(len(run_regions)),
            "candidate_panel_count": int(len(proposals)),
            "candidate_color_count": int(len(colors)),
            "actor_pair_count": int(actor_pair_count),
            "active_color": int(active),
            "inactive_color": int(inactive),
            "proposal_count": int(len(proposals)),
            "promotion_state": promotion_state,
            "rejected_reasons": reasons,
            "proposals": proposals[:sample_limit],
        }
    )


def build_program_row_semantics_shape_transform_bfs_proposals(
    frame: Any,
    *,
    game_id: str = "",
    state_hash: str = "",
    active_color: int | None = None,
    max_plan_steps: int = 48,
    sample_limit: int = 4,
) -> dict[str, Any]:
    """Build guarded proposals for shape/rotation/scale program rows."""

    arr = _as_frame(frame)
    panel = _program_panel_observation(arr)
    program_cells = [
        dict(row)
        for row in list(panel.get("program_cells") or [])
        if isinstance(row, Mapping)
    ]
    run_regions = [
        dict(row)
        for row in list(panel.get("run_button_regions") or [])
        if isinstance(row, Mapping)
    ]
    colors = {
        _safe_int(row.get("color"), -1)
        for row in program_cells
        if _safe_int(row.get("color"), -1) >= 0
    }
    reasons: list[str] = []
    if not program_cells:
        reasons.append("no_program_cells")
    if not run_regions:
        reasons.append("no_run_region")
    if max_plan_steps <= 0:
        reasons.append("max_plan_steps_exhausted")

    active = _safe_int(active_color, -1)
    active_supplied = active >= 0
    if active < 0 or (active not in colors and not active_supplied):
        active = 5 if 5 in colors else max(colors) if colors else -1
    if active_supplied and active >= 0:
        colors.add(active)
    inactive_candidates = sorted(color for color in colors if color != active)
    inactive = inactive_candidates[0] if inactive_candidates else -1
    if active < 0 or inactive < 0:
        reasons.append("missing_binary_program_colors")

    x_centers = _merge_centers(
        [float(row["centroid"][0]) for row in program_cells],
        tolerance=2.0,
    )
    y_centers = _merge_centers(
        [float(row["centroid"][1]) for row in program_cells],
        tolerance=2.0,
    )
    panels = _split_program_x_panels(x_centers)
    if not panels:
        reasons.append("no_program_x_panel")

    proposals: list[dict[str, Any]] = []
    actor_pair_count = 0
    if not reasons:
        state_key = str(state_hash or _frame_hash(arr))
        candidate_panels = _program_panels_with_run_regions(panels, run_regions)
        for panel_index, panel_row in enumerate(candidate_panels):
            columns = [float(v) for v in list(panel_row.get("columns") or [])]
            run_region = panel_row.get("run_region")
            run_region = run_region if isinstance(run_region, Mapping) else {}
            run_site = _centroid_site(run_region.get("centroid"))
            bit_axis = _supported_program_bit_centers(
                program_cells,
                columns=columns,
                bit_centers=y_centers,
            )
            if len(columns) < 3 or len(bit_axis) < 4 or run_site is None:
                continue
            playfield = _program_translation_playfield(
                arr,
                columns=columns,
                bit_centers=bit_axis,
                run_region=run_region,
            )
            actor_pairs = _program_shape_transform_actor_pairs(
                arr,
                playfield_bbox=playfield["bbox"],
            )
            actor_pair_count += len(actor_pairs)
            stride = _program_translation_stride(arr, playfield_bbox=playfield["bbox"])
            for pair_index, pair in enumerate(actor_pairs):
                translation_ops = _program_shape_transform_bfs_ops(
                    start_bbox=pair["start_bbox"],
                    target_anchor=pair["target_anchor"],
                    stride=stride,
                    max_ops=max(0, len(columns) - 3),
                )
                if not translation_ops:
                    continue
                op_sequence = [
                    int(pair["rotation_opcode"]),
                    int(pair["shape_opcode"]),
                    *[int(op) for op in translation_ops],
                ]
                scale_opcode = _safe_int(pair.get("scale_opcode"), -1)
                if scale_opcode >= 0:
                    op_sequence.append(int(scale_opcode))
                desired = list(op_sequence)
                desired.extend([0] * max(0, len(columns) - len(desired)))
                if len(desired) > len(columns):
                    continue
                if any(int(op) >= (1 << len(bit_axis)) for op in desired):
                    continue
                plan_steps = _compile_program_row_opcode_plan(
                    program_cells,
                    columns=columns,
                    bit_centers=bit_axis,
                    desired_opcodes=desired,
                    active_color=active,
                    inactive_color=inactive,
                )
                for step in plan_steps:
                    step["role"] = "shape_transform_bfs_setup_toggle"
                    step["candidate_kind"] = "program_shape_transform_bfs_setup_toggle"
                plan_steps.append(
                    {
                        "action": "ACTION6",
                        "x": int(run_site[0]),
                        "y": int(run_site[1]),
                        "role": "run",
                        "candidate_kind": "program_shape_transform_bfs_run_candidate",
                        "panel_index": int(
                            panel_row.get("source_panel_index", panel_index)
                        ),
                        "active_color": int(active),
                    }
                )
                if len(plan_steps) > int(max_plan_steps):
                    continue
                proposals.append(
                    {
                        "candidate_key": (
                            "program_row_semantics_shape_transform_bfs:"
                            f"{state_key}:panel:{int(panel_row.get('source_panel_index', panel_index))}:"
                            f"pair:{int(pair_index)}:ops:{','.join(str(int(op)) for op in desired)}"
                        ),
                        "candidate_kind": "program_shape_transform_bfs_plan_candidate",
                        "game_id": str(game_id or "").strip().lower(),
                        "state_hash": state_key,
                        "active_color": int(active),
                        "inactive_color": int(inactive),
                        "panel_index": int(
                            panel_row.get("source_panel_index", panel_index)
                        ),
                        "panel_columns": [round(float(value), 3) for value in columns],
                        "bit_centers": [round(float(value), 3) for value in bit_axis],
                        "run_site": [int(run_site[0]), int(run_site[1])],
                        "run_region_area": _safe_int(run_region.get("area")),
                        "stride": int(stride),
                        "actor_color": int(pair["actor_color"]),
                        "target_color": int(pair["target_color"]),
                        "rotation_opcode": int(pair["rotation_opcode"]),
                        "shape_opcode": int(pair["shape_opcode"]),
                        "scale_opcode": int(scale_opcode),
                        "scale_relation": str(pair.get("scale_relation", "")),
                        "start_bbox": [int(value) for value in pair["start_bbox"]],
                        "target_bbox": [int(value) for value in pair["target_bbox"]],
                        "target_anchor": [int(value) for value in pair["target_anchor"]],
                        "translation_op_sequence": [int(op) for op in translation_ops],
                        "op_sequence": [int(op) for op in op_sequence],
                        "desired_opcodes": [int(op) for op in desired],
                        "setup_toggle_count": int(len(plan_steps) - 1),
                        "plan_step_count": int(len(plan_steps)),
                        "plan": plan_steps,
                        "guarded_probe_required": True,
                        "planner_connection_enabled": False,
                        "ranking_enabled": False,
                        "positive_boost_enabled": False,
                        "action_generation_allowed": False,
                    }
                )

    proposals.sort(
        key=lambda row: (
            len(list(row.get("op_sequence") or [])),
            _safe_int(row.get("setup_toggle_count")),
            _safe_int(row.get("rotation_opcode")),
            -_safe_int(row.get("shape_opcode")),
            -_safe_int(row.get("run_region_area")),
            _safe_int(row.get("panel_index")),
        )
    )
    if proposals:
        promotion_state = "program_row_shape_transform_bfs_proposals_supported"
    elif reasons:
        promotion_state = "program_row_shape_transform_bfs_proposals_blocked"
    else:
        promotion_state = "program_row_shape_transform_bfs_proposals_empty"

    return _sanitize(
        {
            "schema_version": PROGRAM_ROW_SEMANTICS_SHAPE_TRANSFORM_BFS_PROPOSAL_SCHEMA_VERSION,
            "default_off": True,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "program_cell_count": int(len(program_cells)),
            "run_region_count": int(len(run_regions)),
            "candidate_panel_count": int(len(proposals)),
            "candidate_color_count": int(len(colors)),
            "actor_pair_count": int(actor_pair_count),
            "active_color": int(active),
            "inactive_color": int(inactive),
            "proposal_count": int(len(proposals)),
            "promotion_state": promotion_state,
            "rejected_reasons": reasons,
            "proposals": proposals[:sample_limit],
        }
    )


def build_program_row_semantics_checkpoint_bfs_proposals(
    frame: Any,
    *,
    game_id: str = "",
    state_hash: str = "",
    active_color: int | None = None,
    max_plan_steps: int = 96,
    max_runs: int = 3,
    sample_limit: int = 4,
) -> dict[str, Any]:
    """Build guarded multi-run checkpoint proposals from a visible lattice.

    This proposal is still at the program-row abstraction: it reads the
    displayed program cells and a compact actor/target pair, infers 4 px
    lattice walls/checkpoints from the public frame, searches short programs
    that can save progress at checkpoint-like cells, then compiles row toggles
    plus RUN clicks. It does not use game IDs, stored level data, adapter
    state, or direct future score information.
    """

    arr = _as_frame(frame)
    panel = _program_panel_observation(arr)
    program_cells = [
        dict(row)
        for row in list(panel.get("program_cells") or [])
        if isinstance(row, Mapping)
    ]
    run_regions = [
        dict(row)
        for row in list(panel.get("run_button_regions") or [])
        if isinstance(row, Mapping)
    ]
    colors = {
        _safe_int(row.get("color"), -1)
        for row in program_cells
        if _safe_int(row.get("color"), -1) >= 0
    }
    reasons: list[str] = []
    if not program_cells:
        reasons.append("no_program_cells")
    if not run_regions:
        reasons.append("no_run_region")
    if max_plan_steps <= 0:
        reasons.append("max_plan_steps_exhausted")

    active = _safe_int(active_color, -1)
    active_supplied = active >= 0
    if active < 0 or (active not in colors and not active_supplied):
        active = 5 if 5 in colors else max(colors) if colors else -1
    if active_supplied and active >= 0:
        colors.add(active)
    inactive_candidates = sorted(color for color in colors if color != active)
    inactive = inactive_candidates[0] if inactive_candidates else -1
    if active < 0 or inactive < 0:
        reasons.append("missing_binary_program_colors")

    x_centers = _merge_centers(
        [float(row["centroid"][0]) for row in program_cells],
        tolerance=2.0,
    )
    y_centers = _merge_centers(
        [float(row["centroid"][1]) for row in program_cells],
        tolerance=2.0,
    )
    panels = _split_program_x_panels(x_centers)
    if not panels:
        reasons.append("no_program_x_panel")

    proposals: list[dict[str, Any]] = []
    route_state = "not_evaluated"
    checkpoint_count = 0
    wall_count = 0
    if not reasons:
        state_key = str(state_hash or _frame_hash(arr))
        candidate_panels = _program_panels_with_run_regions(panels, run_regions)
        for panel_index, panel_row in enumerate(candidate_panels):
            columns = [float(v) for v in list(panel_row.get("columns") or [])]
            run_region = panel_row.get("run_region")
            run_region = run_region if isinstance(run_region, Mapping) else {}
            run_site = _centroid_site(run_region.get("centroid"))
            bit_axis = _supported_program_bit_centers(
                program_cells,
                columns=columns,
                bit_centers=y_centers,
            )
            if len(columns) < 3 or len(bit_axis) < 5 or run_site is None:
                route_state = "insufficient_program_geometry"
                continue
            playfield = _program_translation_playfield(
                arr,
                columns=columns,
                bit_centers=bit_axis,
                run_region=run_region,
            )
            route = _program_checkpoint_route_model(
                arr,
                playfield_bbox=playfield["bbox"],
                row_count=len(columns),
                bit_count=len(bit_axis),
                max_runs=max_runs,
            )
            candidate_route_state = str(route.get("route_state", "") or "")
            if route_state == "not_evaluated":
                route_state = candidate_route_state
            checkpoint_count += len(list(route.get("checkpoints") or []))
            wall_count += len(list(route.get("walls") or []))
            programs = [
                [int(op) for op in row]
                for row in list(route.get("programs") or [])
                if isinstance(row, Sequence)
            ]
            if not programs:
                continue
            route_state = candidate_route_state
            if any(len(program) != len(columns) for program in programs):
                continue
            if any(
                int(op) >= (1 << len(bit_axis))
                for program in programs
                for op in program
            ):
                continue
            plan_steps = _compile_program_row_multirun_opcode_plan(
                program_cells,
                columns=columns,
                bit_centers=bit_axis,
                desired_programs=programs,
                active_color=active,
                inactive_color=inactive,
                run_site=run_site,
                panel_index=int(panel_row.get("source_panel_index", panel_index)),
            )
            if not plan_steps or len(plan_steps) > int(max_plan_steps):
                continue
            proposals.append(
                {
                    "candidate_key": (
                        "program_row_semantics_checkpoint_bfs:"
                        f"{state_key}:panel:{int(panel_row.get('source_panel_index', panel_index))}:"
                        f"programs:{';'.join(','.join(str(int(op)) for op in program) for program in programs)}"
                    ),
                    "candidate_kind": "program_checkpoint_bfs_plan_candidate",
                    "game_id": str(game_id or "").strip().lower(),
                    "state_hash": state_key,
                    "active_color": int(active),
                    "inactive_color": int(inactive),
                    "panel_index": int(panel_row.get("source_panel_index", panel_index)),
                    "panel_columns": [round(float(value), 3) for value in columns],
                    "bit_centers": [round(float(value), 3) for value in bit_axis],
                    "run_site": [int(run_site[0]), int(run_site[1])],
                    "run_region_area": _safe_int(run_region.get("area")),
                    "start_site": [int(v) for v in list(route.get("start_site") or [])[:2]],
                    "target_site": [int(v) for v in list(route.get("target_site") or [])[:2]],
                    "checkpoint_sites": [
                        [int(v) for v in list(site)[:2]]
                        for site in list(route.get("checkpoints") or [])
                    ],
                    "wall_sites": [
                        [int(v) for v in list(site)[:2]]
                        for site in list(route.get("walls") or [])[:32]
                    ],
                    "programs": programs,
                    "desired_opcodes": [int(op) for program in programs for op in program],
                    "run_count": int(len(programs)),
                    "setup_toggle_count": int(
                        sum(1 for step in plan_steps if step.get("role") != "run")
                    ),
                    "plan_step_count": int(len(plan_steps)),
                    "plan": plan_steps,
                    "guarded_probe_required": True,
                    "planner_connection_enabled": False,
                    "ranking_enabled": False,
                    "positive_boost_enabled": False,
                    "action_generation_allowed": False,
                }
            )

    proposals.sort(
        key=lambda row: (
            _safe_int(row.get("run_count")),
            _safe_int(row.get("setup_toggle_count")),
            -_safe_int(row.get("run_region_area")),
            _safe_int(row.get("panel_index")),
        )
    )
    if proposals:
        promotion_state = "program_row_checkpoint_bfs_proposals_supported"
    elif reasons:
        promotion_state = "program_row_checkpoint_bfs_proposals_blocked"
    else:
        promotion_state = "program_row_checkpoint_bfs_proposals_empty"

    return _sanitize(
        {
            "schema_version": PROGRAM_ROW_SEMANTICS_CHECKPOINT_BFS_PROPOSAL_SCHEMA_VERSION,
            "default_off": True,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "program_cell_count": int(len(program_cells)),
            "run_region_count": int(len(run_regions)),
            "candidate_panel_count": int(len(proposals)),
            "candidate_color_count": int(len(colors)),
            "active_color": int(active),
            "inactive_color": int(inactive),
            "checkpoint_count": int(checkpoint_count),
            "wall_count": int(wall_count),
            "route_state": route_state,
            "proposal_count": int(len(proposals)),
            "promotion_state": promotion_state,
            "rejected_reasons": reasons,
            "proposals": proposals[:sample_limit],
        }
    )


def summarize_program_row_semantics_guarded_probe_outcomes(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize observed outcomes from default-off program-row guarded probes."""

    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_candidate_kind: dict[str, Counter[str]] = {}
    samples: list[dict[str, Any]] = []

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if isinstance(trace, list):
            trace_items = [item for item in trace if isinstance(item, Mapping)]
        elif isinstance(row.get("unsolved_replan"), Mapping):
            trace_items = [row]
        else:
            trace_items = []
        by_game.setdefault(game_id, Counter())

        for item in trace_items:
            unsolved = item.get("unsolved_replan")
            unsolved = unsolved if isinstance(unsolved, Mapping) else {}
            preflight = unsolved.get("program_row_semantics_guarded_probe_preflight")
            if isinstance(preflight, Mapping) and bool(preflight.get("enabled")):
                counts["preflight_count"] += 1
                by_game[game_id]["preflight_count"] += 1
            is_guarded = str(item.get("phase", "") or "") == (
                "program_row_semantics_guarded_probe"
            )
            is_guarded = is_guarded or str(unsolved.get("source", "") or "") == (
                "program_row_semantics_guarded_probe"
            )
            if not is_guarded:
                continue

            packet = item.get("program_row_semantics_packet")
            packet = packet if isinstance(packet, Mapping) else {}
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            candidate_kind = str(
                unsolved.get("selected_candidate_kind")
                or unsolved.get("candidate_kind")
                or "unknown"
            )
            selected_site = unsolved.get("selected_site")
            selected_site = list(selected_site) if isinstance(selected_site, Sequence) else []
            score_delta = _safe_int(
                packet.get("score_delta"),
                _safe_int(outcome.get("score_delta")),
            )
            result_delta = _safe_int(packet.get("result_region_changed_cell_count"))
            program_delta = _safe_int(packet.get("program_cell_changed_cell_count"))
            changed_pixel_count = _safe_int(
                outcome.get("changed_pixel_count"),
                _safe_int(packet.get("changed_cell_count")),
            )

            counts["execution_count"] += 1
            by_game[game_id]["execution_count"] += 1
            by_candidate_kind.setdefault(candidate_kind, Counter())
            by_candidate_kind[candidate_kind]["execution_count"] += 1
            if bool(unsolved.get("admitted")):
                counts["admitted_guarded_probe_count"] += 1
                by_game[game_id]["admitted_guarded_probe_count"] += 1
                by_candidate_kind[candidate_kind]["admitted_guarded_probe_count"] += 1
            if bool(outcome):
                counts["runtime_outcome_count"] += 1
                by_game[game_id]["runtime_outcome_count"] += 1
            if bool(unsolved.get("direct_llm_action_allowed")):
                counts["direct_llm_action_allowed_count"] += 1
                by_game[game_id]["direct_llm_action_allowed_count"] += 1

            counts["score_delta_sum"] += int(score_delta)
            by_game[game_id]["score_delta_sum"] += int(score_delta)
            counts["result_region_changed_cell_count_sum"] += int(result_delta)
            counts["program_cell_changed_cell_count_sum"] += int(program_delta)
            if score_delta > 0:
                outcome_state = "score_positive"
                counts["score_positive_execution_count"] += 1
                by_game[game_id]["score_positive_execution_count"] += 1
                by_candidate_kind[candidate_kind]["score_positive_execution_count"] += 1
            elif result_delta > 0:
                outcome_state = "result_region_changed"
                counts["result_region_changed_execution_count"] += 1
                by_game[game_id]["result_region_changed_execution_count"] += 1
                by_candidate_kind[candidate_kind][
                    "result_region_changed_execution_count"
                ] += 1
            elif program_delta > 0 or changed_pixel_count > 0:
                outcome_state = "scoreless_edit"
                counts["scoreless_edit_execution_count"] += 1
                by_game[game_id]["scoreless_edit_execution_count"] += 1
                by_candidate_kind[candidate_kind]["scoreless_edit_execution_count"] += 1
            else:
                outcome_state = "scoreless_no_effect"
                counts["scoreless_no_effect_execution_count"] += 1
                by_game[game_id]["scoreless_no_effect_execution_count"] += 1
                by_candidate_kind[candidate_kind][
                    "scoreless_no_effect_execution_count"
                ] += 1
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _safe_int(item.get("step")),
                        "candidate_key": str(unsolved.get("candidate_key", "") or ""),
                        "candidate_kind": candidate_kind,
                        "selected_site": selected_site,
                        "outcome_state": outcome_state,
                        "score_delta": int(score_delta),
                        "result_region_changed_cell_count": int(result_delta),
                        "program_cell_changed_cell_count": int(program_delta),
                        "changed_pixel_count": int(changed_pixel_count),
                    }
                )

    execution_count = int(counts.get("execution_count", 0))
    if execution_count <= 0 and int(counts.get("preflight_count", 0)) > 0:
        promotion_state = "program_row_guarded_probe_preflight_only"
    elif execution_count <= 0:
        promotion_state = "program_row_guarded_probe_no_execution"
    elif int(counts.get("score_positive_execution_count", 0)) > 0:
        promotion_state = "program_row_guarded_probe_score_positive"
    elif int(counts.get("result_region_changed_execution_count", 0)) > 0:
        promotion_state = "program_row_guarded_probe_result_region_change"
    elif int(counts.get("scoreless_edit_execution_count", 0)) > 0:
        promotion_state = "program_row_guarded_probe_scoreless_edit"
    else:
        promotion_state = "program_row_guarded_probe_scoreless_no_effect"

    return _sanitize(
        {
            "schema_version": PROGRAM_ROW_SEMANTICS_GUARDED_PROBE_OUTCOME_SCHEMA_VERSION,
            "default_off": True,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": bool(execution_count > 0),
            "guarded_probe_execution_observed": bool(execution_count > 0),
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "direct_llm_action_allowed_count": int(
                counts.get("direct_llm_action_allowed_count", 0)
            ),
            "preflight_count": int(counts.get("preflight_count", 0)),
            "execution_count": execution_count,
            "admitted_guarded_probe_count": int(
                counts.get("admitted_guarded_probe_count", 0)
            ),
            "runtime_outcome_count": int(counts.get("runtime_outcome_count", 0)),
            "score_positive_execution_count": int(
                counts.get("score_positive_execution_count", 0)
            ),
            "result_region_changed_execution_count": int(
                counts.get("result_region_changed_execution_count", 0)
            ),
            "scoreless_edit_execution_count": int(
                counts.get("scoreless_edit_execution_count", 0)
            ),
            "scoreless_no_effect_execution_count": int(
                counts.get("scoreless_no_effect_execution_count", 0)
            ),
            "score_delta_sum": int(counts.get("score_delta_sum", 0)),
            "result_region_changed_cell_count_sum": int(
                counts.get("result_region_changed_cell_count_sum", 0)
            ),
            "program_cell_changed_cell_count_sum": int(
                counts.get("program_cell_changed_cell_count_sum", 0)
            ),
            "promotion_state": promotion_state,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_candidate_kind": {
                kind: dict(sorted(counter.items()))
                for kind, counter in sorted(by_candidate_kind.items())
            },
            "samples": samples,
            "next_runtime_contract": {
                "required_before_planner_promotion": (
                    "counterfactual controls must separate score-bearing run "
                    "executions from scoreless row edits before ranking changes"
                ),
                "allowed_connection": "guarded_probe_outcome_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_program_row_semantics_counterfactual_controls(
    rows: Iterable[Mapping[str, Any]],
    *,
    outcome_summary: Mapping[str, Any] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Check whether guarded outcomes have positive evidence and controls."""

    outcome = (
        dict(outcome_summary)
        if isinstance(outcome_summary, Mapping)
        else summarize_program_row_semantics_guarded_probe_outcomes(
            rows,
            sample_limit=sample_limit,
        )
    )
    score_positive = _safe_int(outcome.get("score_positive_execution_count"))
    result_region_changed = _safe_int(
        outcome.get("result_region_changed_execution_count")
    )
    scoreless_edit = _safe_int(outcome.get("scoreless_edit_execution_count"))
    scoreless_no_effect = _safe_int(
        outcome.get("scoreless_no_effect_execution_count")
    )
    by_kind = outcome.get("by_candidate_kind")
    by_kind = by_kind if isinstance(by_kind, Mapping) else {}
    protected_setup_scoreless = 0
    run_candidate_scoreless = 0
    for kind, counter in by_kind.items():
        if not isinstance(counter, Mapping):
            continue
        kind_text = str(kind or "")
        scoreless_for_kind = _safe_int(
            counter.get("scoreless_edit_execution_count")
        ) + _safe_int(counter.get("scoreless_no_effect_execution_count"))
        if kind_text == "program_polarity_fill_setup_toggle":
            protected_setup_scoreless += int(scoreless_for_kind)
        else:
            run_candidate_scoreless += int(scoreless_for_kind)
    raw_scoreless_control = int(scoreless_edit + scoreless_no_effect)
    scoreless_control = (
        int(run_candidate_scoreless)
        if by_kind
        else int(raw_scoreless_control)
    )
    scoreless_execution_control = int(scoreless_control + result_region_changed)
    control_pair = int(score_positive > 0 and scoreless_control > 0)
    margin = int(score_positive - scoreless_control)
    positive_control_margin = int(control_pair > 0 and margin >= 0)

    if score_positive <= 0 and scoreless_execution_control > 0:
        promotion_state = "program_row_counterfactual_control_missing_positive"
    elif score_positive <= 0:
        promotion_state = "program_row_counterfactual_control_no_guarded_outcome"
    elif scoreless_control <= 0:
        promotion_state = "program_row_counterfactual_control_no_scoreless_control"
    elif positive_control_margin > 0:
        promotion_state = "program_row_counterfactual_control_supported"
    else:
        promotion_state = "program_row_counterfactual_control_dominated"

    samples = [
        dict(row)
        for row in list(outcome.get("samples") or [])
        if isinstance(row, Mapping)
    ][:sample_limit]

    return _sanitize(
        {
            "schema_version": PROGRAM_ROW_SEMANTICS_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION,
            "default_off": True,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "guarded_probe_execution_count": _safe_int(
                outcome.get("execution_count")
            ),
            "score_positive_execution_count": int(score_positive),
            "result_region_changed_execution_count": int(result_region_changed),
            "scoreless_edit_execution_count": int(scoreless_edit),
            "scoreless_no_effect_execution_count": int(scoreless_no_effect),
            "protected_setup_scoreless_execution_count": int(
                protected_setup_scoreless
            ),
            "raw_scoreless_control_execution_count": int(raw_scoreless_control),
            "run_candidate_scoreless_control_execution_count": int(
                run_candidate_scoreless
            ),
            "scoreless_control_execution_count": int(scoreless_control),
            "scoreless_execution_control_count": int(scoreless_execution_control),
            "counterfactual_control_pair_count": int(control_pair),
            "positive_control_margin": int(margin),
            "positive_control_margin_count": int(positive_control_margin),
            "planner_promotion_blocked": bool(positive_control_margin <= 0),
            "promotion_state": promotion_state,
            "outcome_promotion_state": str(outcome.get("promotion_state", "") or ""),
            "by_candidate_kind": outcome.get("by_candidate_kind", {}),
            "samples": samples,
            "next_runtime_contract": {
                "required_before_ranking": (
                    "positive score-bearing program execution must survive "
                    "scoreless run-button and row-edit controls"
                ),
                "allowed_connection": "counterfactual_control_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_program_row_semantics_all25_non_regression(
    run_rows: Iterable[Mapping[str, Any]],
    *,
    expected_score_sum: int | None = None,
    expected_by_game: Mapping[str, Any] | None = None,
    require_no_game_conditioned_primitives: bool = True,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Check all25 preservation while program-row guarded diagnostics are enabled."""

    rows = [dict(row) for row in run_rows if isinstance(row, Mapping)]
    expected_map = {
        str(game_id): _safe_int(score)
        for game_id, score in dict(expected_by_game or {}).items()
    }
    if expected_score_sum is None and expected_map:
        expected_score_sum = int(sum(expected_map.values()))

    by_game: dict[str, int] = {}
    status_counts: Counter[str] = Counter()
    score_sum = 0
    adapter_free_score_sum = 0
    game_conditioned_count = 0
    error_count = 0

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        score = _safe_int(row.get("final_score"))
        by_game[game_id] = int(score)
        score_sum += int(score)
        status = str(row.get("status", "") or "unknown")
        status_counts[status] += 1
        if status != "ok":
            error_count += 1
        uses_game_conditioned = bool(row.get("uses_game_conditioned_primitive"))
        if uses_game_conditioned:
            game_conditioned_count += 1
        else:
            adapter_free_score_sum += int(score)

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    missing_games: list[str] = []
    for game_id, expected in sorted(expected_map.items()):
        if game_id not in by_game:
            missing_games.append(game_id)
            continue
        actual = int(by_game[game_id])
        delta = int(actual - int(expected))
        if delta < 0 and len(regressions) < sample_limit:
            regressions.append(
                {
                    "game_id": game_id,
                    "expected_score": int(expected),
                    "actual_score": int(actual),
                    "delta": int(delta),
                }
            )
        elif delta > 0 and len(improvements) < sample_limit:
            improvements.append(
                {
                    "game_id": game_id,
                    "expected_score": int(expected),
                    "actual_score": int(actual),
                    "delta": int(delta),
                }
            )

    score_regression = (
        expected_score_sum is not None and int(score_sum) < int(expected_score_sum)
    )
    game_regression_count = int(
        sum(
            1
            for game_id, expected in expected_map.items()
            if game_id in by_game and int(by_game[game_id]) < int(expected)
        )
    )
    improved_game_count = int(
        sum(
            1
            for game_id, expected in expected_map.items()
            if game_id in by_game and int(by_game[game_id]) > int(expected)
        )
    )
    reason_counts: Counter[str] = Counter()
    if not rows:
        reason_counts["no_run_rows"] += 1
    if expected_score_sum is None and not expected_map:
        reason_counts["missing_reference_score_profile"] += 1
    if missing_games:
        reason_counts["missing_expected_games"] += len(missing_games)
    if error_count > 0:
        reason_counts["run_errors_present"] += int(error_count)
    if require_no_game_conditioned_primitives and game_conditioned_count > 0:
        reason_counts["game_conditioned_primitive_present"] += int(
            game_conditioned_count
        )
    if score_regression:
        reason_counts["score_sum_regression"] += 1
    if game_regression_count > 0:
        reason_counts["by_game_regression"] += int(game_regression_count)

    supported = not bool(reason_counts)
    if supported:
        reason_counts["program_row_semantics_all25_non_regression_supported"] += 1
        promotion_state = "program_row_semantics_all25_non_regression_supported"
    elif rows:
        promotion_state = "program_row_semantics_all25_non_regression_blocked"
    else:
        promotion_state = "program_row_semantics_all25_non_regression_no_rows"

    return _sanitize(
        {
            "schema_version": PROGRAM_ROW_SEMANTICS_ALL25_NON_REGRESSION_SCHEMA_VERSION,
            "diagnostic_only": True,
            "default_off": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "run_count": int(len(rows)),
            "ok_count": int(status_counts.get("ok", 0)),
            "error_count": int(error_count),
            "score_sum": int(score_sum),
            "adapter_free_score_sum": int(adapter_free_score_sum),
            "uses_game_conditioned_primitive_count": int(game_conditioned_count),
            "expected_score_sum": (
                None if expected_score_sum is None else int(expected_score_sum)
            ),
            "score_sum_delta_vs_reference": (
                None
                if expected_score_sum is None
                else int(score_sum - int(expected_score_sum))
            ),
            "expected_game_count": int(len(expected_map)),
            "observed_expected_game_count": int(
                sum(1 for game_id in expected_map if game_id in by_game)
            ),
            "regressed_game_count": int(game_regression_count),
            "improved_game_count": int(improved_game_count),
            "missing_expected_game_count": int(len(missing_games)),
            "missing_expected_games": missing_games[:sample_limit],
            "regressions": regressions,
            "improvements": improvements,
            "score_by_game": dict(sorted(by_game.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "promotion_state": promotion_state,
            "next_runtime_contract": {
                "required_before_planner_or_ranking_promotion": (
                    "focused score-bearing program-row evidence must preserve "
                    "the all25 reference profile before any ranking or planner "
                    "connection is enabled"
                ),
                "allowed_connection": "all25_non_regression_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def _program_panel_observation(arr: np.ndarray) -> dict[str, Any]:
    cells = _program_cell_components(arr)
    run_regions = _run_button_regions(arr)
    active_count = sum(1 for row in cells if int(row.get("color", -1)) == 1)
    return {
        "program_cells": cells,
        "run_button_regions": run_regions,
        "active_program_bit_count": int(active_count),
    }


def _program_cell_components(arr: np.ndarray) -> list[dict[str, Any]]:
    height = int(arr.shape[0])
    y_min = max(4, int(round(height * 0.42)))
    out: list[dict[str, Any]] = []
    for component in _connected_components(arr):
        color = int(component.get("color", -1))
        if color not in (1, 5):
            continue
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]
        if y0 < y_min:
            continue
        width = x1 - x0 + 1
        height_box = y1 - y0 + 1
        area = int(component.get("area", 0))
        if area > 9 or width > 5 or height_box > 5:
            continue
        if width < 2 and height_box < 2:
            continue
        row = dict(component)
        row["orientation"] = "horizontal" if width >= height_box else "vertical"
        out.append(row)
    return sorted(out, key=lambda row: (float(row["centroid"][1]), float(row["centroid"][0])))


def _run_button_regions(arr: np.ndarray) -> list[dict[str, Any]]:
    height = int(arr.shape[0])
    y_min = max(8, int(round(height * 0.55)))
    out: list[dict[str, Any]] = []
    for component in _connected_components(arr):
        if int(component.get("color", -1)) != 9:
            continue
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]
        if y0 < y_min:
            continue
        width = x1 - x0 + 1
        height_box = y1 - y0 + 1
        area = int(component.get("area", 0))
        if area < 8 or width < 4 or height_box < 4:
            continue
        out.append(dict(component))
    return sorted(out, key=lambda row: (float(row["centroid"][1]), float(row["centroid"][0])))


def _change_observation(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    changed = before != after
    height = int(before.shape[0])
    y_indices = np.arange(height)[:, None]
    top_limit = max(2, int(round(height * 0.06)))
    top_mask = changed & (y_indices <= top_limit)
    lower_mask = changed & (y_indices >= max(4, int(round(height * 0.42))))
    program_color_mask = lower_mask & (
        np.isin(before, [1, 5]) | np.isin(after, [1, 5])
    )
    program_components = _components_from_mask(
        program_color_mask,
        before=before,
        after=after,
    )
    program_component_mask = np.zeros_like(changed, dtype=bool)
    for row in program_components:
        for x, y in row.get("cells", []):
            if 0 <= int(y) < height and 0 <= int(x) < int(before.shape[1]):
                program_component_mask[int(y), int(x)] = True
    result_mask = changed & ~top_mask & ~program_component_mask
    visible_components = []
    for row in program_components:
        item = dict(row)
        item.pop("cells", None)
        visible_components.append(item)
    return {
        "changed_cell_count": int(np.count_nonzero(changed)),
        "top_progress_changed_cell_count": int(np.count_nonzero(top_mask)),
        "program_cell_changed_cell_count": int(np.count_nonzero(program_component_mask)),
        "result_region_changed_cell_count": int(np.count_nonzero(result_mask)),
        "program_change_components": visible_components,
    }


def _components_from_mask(
    mask: np.ndarray,
    *,
    before: np.ndarray,
    after: np.ndarray,
) -> list[dict[str, Any]]:
    height, width = mask.shape[:2]
    seen = np.zeros((height, width), dtype=bool)
    out: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x] or not bool(mask[y, x]):
                continue
            stack = [(x, y)]
            seen[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            before_colors: Counter[str] = Counter()
            after_colors: Counter[str] = Counter()
            pair_counts: Counter[str] = Counter()
            while stack:
                px, py = stack.pop()
                xs.append(px)
                ys.append(py)
                b = int(before[py, px])
                a = int(after[py, px])
                before_colors[str(b)] += 1
                after_colors[str(a)] += 1
                pair_counts[f"{b}->{a}"] += 1
                for nx, ny in (
                    (px + 1, py),
                    (px - 1, py),
                    (px, py + 1),
                    (px, py - 1),
                ):
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and not seen[ny, nx]
                        and bool(mask[ny, nx])
                    ):
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            area = len(xs)
            out.append(
                {
                    "id": f"pc{len(out)}",
                    "area": int(area),
                    "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    "centroid": [
                        round(float(sum(xs)) / float(area), 3),
                        round(float(sum(ys)) / float(area), 3),
                    ],
                    "before_colors": dict(sorted(before_colors.items())),
                    "after_colors": dict(sorted(after_colors.items())),
                    "color_pair_counts": dict(sorted(pair_counts.items())),
                    "program_color_pair": bool(
                        any(pair in pair_counts for pair in ("1->5", "5->1"))
                    ),
                    "cells": [(int(px), int(py)) for px, py in zip(xs, ys)],
                }
            )
    return out


def _selected_program_cell(
    before: np.ndarray,
    after: np.ndarray,
    *,
    selected_site: tuple[int, int] | None,
    row_centers: Sequence[float],
    bit_centers: Sequence[float],
    program_change_components: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if selected_site is None:
        return None
    sx, sy = selected_site
    if not (0 <= sx < int(before.shape[1]) and 0 <= sy < int(before.shape[0])):
        return None
    nearest_change = _nearest_component_to_site(program_change_components, selected_site)
    near_changed_program_cell = bool(
        nearest_change is not None and float(nearest_change.get("distance", 999.0)) <= 2.5
    )
    row_index = _nearest_index(row_centers, float(sx), max_distance=3.5)
    bit_index = _nearest_index(bit_centers, float(sy), max_distance=3.5)
    if row_index is None and near_changed_program_cell:
        row_index = _nearest_index(row_centers, float(sx), max_distance=8.0)
    if bit_index is None and near_changed_program_cell:
        bit_index = _nearest_index(bit_centers, float(sy), max_distance=8.0)
    if row_index is None and bit_index is None and not near_changed_program_cell:
        return None
    before_color = int(before[sy, sx])
    after_color = int(after[sy, sx])
    color_pair = f"{before_color}->{after_color}"
    changed = before_color != after_color or near_changed_program_cell
    return {
        "site": [int(sx), int(sy)],
        "row_index": row_index,
        "bit_index": bit_index,
        "row_center": (
            round(float(row_centers[row_index]), 3) if row_index is not None else None
        ),
        "bit_center": (
            round(float(bit_centers[bit_index]), 3) if bit_index is not None else None
        ),
        "before_color": before_color,
        "after_color": after_color,
        "changed": bool(changed),
        "program_color_pair": bool(
            color_pair in ("1->5", "5->1")
            or (
                nearest_change is not None
                and bool(nearest_change.get("program_color_pair"))
            )
        ),
        "nearest_change": nearest_change,
    }


def _selected_run_button_region(
    selected_site: tuple[int, int] | None,
    before_regions: Sequence[Mapping[str, Any]],
    after_regions: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if selected_site is None:
        return None
    sx, sy = selected_site
    for region in list(before_regions) + list(after_regions):
        bbox = region.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]
        if x0 - 2 <= sx <= x1 + 2 and y0 - 2 <= sy <= y1 + 2:
            out = dict(region)
            out["selected_inside_expanded_bbox"] = True
            return out
    return None


def _nearest_component_to_site(
    components: Sequence[Mapping[str, Any]],
    site: tuple[int, int],
) -> dict[str, Any] | None:
    sx, sy = site
    best: dict[str, Any] | None = None
    best_distance = 1e9
    for component in components:
        centroid = component.get("centroid")
        if not isinstance(centroid, Sequence) or len(centroid) < 2:
            continue
        distance = math.hypot(float(centroid[0]) - float(sx), float(centroid[1]) - float(sy))
        if distance < best_distance:
            best_distance = distance
            best = dict(component)
    if best is None:
        return None
    best.pop("cells", None)
    best["distance"] = round(float(best_distance), 3)
    return best


def _centroid_site(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Sequence) or len(value) < 2:
        return None
    try:
        return int(round(float(value[0]))), int(round(float(value[1])))
    except (TypeError, ValueError):
        return None


def _merge_centers(values: Sequence[float], *, tolerance: float = 2.0) -> list[float]:
    centers: list[float] = []
    for value in sorted(float(v) for v in values):
        if not centers or abs(value - centers[-1]) > tolerance:
            centers.append(value)
        else:
            centers[-1] = (centers[-1] + value) / 2.0
    return centers


def _split_program_x_panels(x_centers: Sequence[float]) -> list[list[float]]:
    values = [float(value) for value in sorted(x_centers)]
    if not values:
        return []
    panels: list[list[float]] = [[values[0]]]
    gaps = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    small_gaps = [gap for gap in gaps if gap > 0]
    median_gap = sorted(small_gaps)[len(small_gaps) // 2] if small_gaps else 0.0
    split_gap = max(8.0, median_gap * 1.75)
    for previous, value in zip(values, values[1:]):
        if float(value) - float(previous) > split_gap:
            panels.append([])
        panels[-1].append(float(value))
    return [panel for panel in panels if panel]


def _program_panels_with_run_regions(
    panels: Sequence[Sequence[float]],
    run_regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel_index, panel in enumerate(panels):
        columns = [float(value) for value in panel]
        if not columns:
            continue
        x_min = min(columns)
        x_max = max(columns)
        best_region: Mapping[str, Any] | None = None
        best_distance = float("inf")
        for region in run_regions:
            centroid = region.get("centroid")
            if not isinstance(centroid, Sequence) or len(centroid) < 2:
                continue
            try:
                cx = float(centroid[0])
            except (TypeError, ValueError):
                continue
            inside_distance = 0.0 if x_min - 4.0 <= cx <= x_max + 4.0 else min(
                abs(cx - x_min),
                abs(cx - x_max),
            )
            if inside_distance < best_distance:
                best_distance = inside_distance
                best_region = region
        if best_region is None:
            continue
        rows.append(
            {
                "source_panel_index": int(panel_index),
                "columns": columns,
                "run_region": dict(best_region),
                "run_region_area": _safe_int(best_region.get("area")),
                "run_region_distance": round(float(best_distance), 3),
            }
        )
    rows.sort(
        key=lambda row: (
            -_safe_int(row.get("run_region_area")),
            float(row.get("run_region_distance", 0.0) or 0.0),
            _safe_int(row.get("source_panel_index")),
        )
    )
    return rows


def _nearest_program_cell(
    cells: Sequence[Mapping[str, Any]],
    *,
    x_center: float,
    y_center: float,
) -> Mapping[str, Any] | None:
    best: Mapping[str, Any] | None = None
    best_distance = float("inf")
    for cell in cells:
        centroid = cell.get("centroid")
        if not isinstance(centroid, Sequence) or len(centroid) < 2:
            continue
        try:
            cx = float(centroid[0])
            cy = float(centroid[1])
        except (TypeError, ValueError):
            continue
        distance = abs(cx - float(x_center)) + abs(cy - float(y_center))
        if distance < best_distance:
            best_distance = distance
            best = cell
    if best_distance > 4.0:
        return None
    return best


def _program_translation_playfield(
    arr: np.ndarray,
    *,
    columns: Sequence[float],
    bit_centers: Sequence[float],
    run_region: Mapping[str, Any],
) -> dict[str, Any]:
    height, width = arr.shape[:2]
    x_min = max(0, int(math.floor(min(columns) - 4.0))) if columns else 0
    x_max = min(width - 1, int(math.ceil(max(columns) + 4.0))) if columns else width - 1
    bbox = run_region.get("bbox")
    if _is_bbox(bbox):
        rb = [int(v) for v in bbox]
        x_min = min(x_min, max(0, rb[0] - 2))
        x_max = max(x_max, min(width - 1, rb[2] + 2))
    y_min = max(0, int(round(height * 0.04)))
    first_bit_y = min(float(value) for value in bit_centers) if bit_centers else height * 0.5
    y_max = max(y_min, min(height - 1, int(math.floor(first_bit_y - 3.0))))
    return {"bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]}


def _program_translation_actor_pairs(
    arr: np.ndarray,
    *,
    playfield_bbox: Sequence[int],
) -> list[dict[str, Any]]:
    if not _is_bbox(playfield_bbox):
        return []
    x0, y0, x1, y1 = [int(v) for v in playfield_bbox]
    by_color: dict[int, list[dict[str, Any]]] = {}
    for component in _connected_components(arr):
        color = _safe_int(component.get("color"), -1)
        if color in {0, 1, 4, 5, 9} or color < 0:
            continue
        bbox = component.get("bbox")
        centroid = component.get("centroid")
        if not _is_bbox(bbox) or not isinstance(centroid, Sequence) or len(centroid) < 2:
            continue
        cx, cy = float(centroid[0]), float(centroid[1])
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        bx0, by0, bx1, by1 = [int(v) for v in bbox]
        box_w = bx1 - bx0 + 1
        box_h = by1 - by0 + 1
        area = _safe_int(component.get("area"))
        if area < 4 or box_w > 10 or box_h > 10:
            continue
        by_color.setdefault(color, []).append(dict(component))

    pairs: list[dict[str, Any]] = []
    for color, components in sorted(by_color.items()):
        if len(components) < 2:
            continue
        rows = sorted(
            components,
            key=lambda row: (
                -_safe_int(row.get("area")),
                float(row.get("centroid", [0, 0])[1]),
                float(row.get("centroid", [0, 0])[0]),
            ),
        )[:4]
        for a_index, first in enumerate(rows):
            for second in rows[a_index + 1 :]:
                c1 = first.get("centroid")
                c2 = second.get("centroid")
                if not (
                    isinstance(c1, Sequence)
                    and len(c1) >= 2
                    and isinstance(c2, Sequence)
                    and len(c2) >= 2
                ):
                    continue
                distance = math.hypot(float(c1[0]) - float(c2[0]), float(c1[1]) - float(c2[1]))
                if distance < 6.0:
                    continue
                # Program goals normally show the desired pose above the current actor.
                if float(c1[1]) >= float(c2[1]):
                    start, target = first, second
                else:
                    start, target = second, first
                pairs.append(
                    {
                        "color": int(color),
                        "start_bbox": list(start.get("bbox") or []),
                        "target_bbox": list(target.get("bbox") or []),
                        "start_centroid": list(start.get("centroid") or []),
                        "target_centroid": list(target.get("centroid") or []),
                        "distance": round(float(distance), 3),
                    }
                )
    pairs.sort(
        key=lambda row: (
            -float(row.get("distance", 0.0) or 0.0),
            float(row.get("target_centroid", [0, 0])[1]),
            float(row.get("target_centroid", [0, 0])[0]),
        )
    )
    return pairs[:8]


def _program_scale_translation_actor_pairs(
    arr: np.ndarray,
    *,
    playfield_bbox: Sequence[int],
) -> list[dict[str, Any]]:
    if not _is_bbox(playfield_bbox):
        return []
    x0, y0, x1, y1 = [int(v) for v in playfield_bbox]
    by_color: dict[int, list[dict[str, Any]]] = {}
    for component in _connected_components(arr):
        color = _safe_int(component.get("color"), -1)
        if color in {0, 1, 4, 5, 9} or color < 0:
            continue
        bbox = component.get("bbox")
        centroid = component.get("centroid")
        if not _is_bbox(bbox) or not isinstance(centroid, Sequence) or len(centroid) < 2:
            continue
        cx, cy = float(centroid[0]), float(centroid[1])
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        bx0, by0, bx1, by1 = [int(v) for v in bbox]
        box_w = bx1 - bx0 + 1
        box_h = by1 - by0 + 1
        area = _safe_int(component.get("area"))
        if area < 8 or box_w > 16 or box_h > 16:
            continue
        by_color.setdefault(color, []).append(dict(component))

    pairs: list[dict[str, Any]] = []
    for color, components in sorted(by_color.items()):
        if len(components) < 2:
            continue
        rows = sorted(
            components,
            key=lambda row: (
                -_safe_int(row.get("area")),
                float(row.get("centroid", [0, 0])[1]),
                float(row.get("centroid", [0, 0])[0]),
            ),
        )[:6]
        for first_index, first in enumerate(rows):
            for second in rows[first_index + 1 :]:
                first_area = _safe_int(first.get("area"))
                second_area = _safe_int(second.get("area"))
                if min(first_area, second_area) <= 0:
                    continue
                larger, smaller = (
                    (first, second) if first_area >= second_area else (second, first)
                )
                larger_area = max(first_area, second_area)
                smaller_area = min(first_area, second_area)
                if larger_area < smaller_area * 1.75:
                    continue
                larger_bbox = larger.get("bbox")
                smaller_bbox = smaller.get("bbox")
                larger_centroid = larger.get("centroid")
                smaller_centroid = smaller.get("centroid")
                if (
                    not _is_bbox(larger_bbox)
                    or not _is_bbox(smaller_bbox)
                    or not isinstance(larger_centroid, Sequence)
                    or len(larger_centroid) < 2
                    or not isinstance(smaller_centroid, Sequence)
                    or len(smaller_centroid) < 2
                ):
                    continue
                distance = math.hypot(
                    float(larger_centroid[0]) - float(smaller_centroid[0]),
                    float(larger_centroid[1]) - float(smaller_centroid[1]),
                )
                if distance < 6.0:
                    continue
                vertical_delta = abs(
                    float(larger_centroid[1]) - float(smaller_centroid[1])
                )
                if vertical_delta < 6.0:
                    continue
                pairs.append(
                    {
                        "color": int(color),
                        "scale_opcode": 9,
                        "scale_relation": "larger_component_to_smaller_component",
                        "start_bbox": [int(v) for v in list(larger_bbox)[:4]],
                        "target_bbox": [int(v) for v in list(smaller_bbox)[:4]],
                        "start_centroid": list(larger_centroid),
                        "target_centroid": list(smaller_centroid),
                        "area_ratio": round(float(larger_area) / float(smaller_area), 3),
                        "distance": round(float(distance), 3),
                        "vertical_delta": round(float(vertical_delta), 3),
                    }
                )
    pairs.sort(
        key=lambda row: (
            len(_program_scale_translation_bfs_ops(
                start_bbox=row["start_bbox"],
                target_bbox=row["target_bbox"],
                stride=4,
                max_ops=8,
            ))
            or 999,
            -float(row.get("area_ratio", 0.0) or 0.0),
            float(row.get("target_centroid", [0, 0])[1]),
            float(row.get("target_centroid", [0, 0])[0]),
        )
    )
    return pairs[:8]


def _supported_program_bit_centers(
    cells: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[float],
    bit_centers: Sequence[float],
) -> list[float]:
    if not columns or not bit_centers:
        return []
    threshold = max(2, min(3, int(math.ceil(len(columns) * 0.5))))
    supported: list[float] = []
    for y_center in bit_centers:
        seen_columns: set[int] = set()
        for cell in cells:
            centroid = cell.get("centroid")
            if not isinstance(centroid, Sequence) or len(centroid) < 2:
                continue
            try:
                cx = float(centroid[0])
                cy = float(centroid[1])
            except (TypeError, ValueError):
                continue
            if abs(cy - float(y_center)) > 1.75:
                continue
            col_index = _nearest_index(columns, cx, max_distance=3.0)
            if col_index is not None:
                seen_columns.add(int(col_index))
        if len(seen_columns) >= threshold:
            supported.append(float(y_center))
    return supported


def _program_shape_transform_actor_pairs(
    arr: np.ndarray,
    *,
    playfield_bbox: Sequence[int],
) -> list[dict[str, Any]]:
    if not _is_bbox(playfield_bbox):
        return []
    x0, y0, x1, y1 = [int(v) for v in playfield_bbox]
    components: list[dict[str, Any]] = []
    for component in _connected_components(arr):
        color = _safe_int(component.get("color"), -1)
        if color in {0, 1, 2, 3, 4, 5, 6, 9} or color < 0:
            continue
        bbox = component.get("bbox")
        centroid = component.get("centroid")
        if not _is_bbox(bbox) or not isinstance(centroid, Sequence) or len(centroid) < 2:
            continue
        cx, cy = float(centroid[0]), float(centroid[1])
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        bx0, by0, bx1, by1 = [int(v) for v in bbox]
        box_w = bx1 - bx0 + 1
        box_h = by1 - by0 + 1
        area = _safe_int(component.get("area"))
        if area < 4 or box_w > 20 or box_h > 20:
            continue
        components.append(dict(component))

    pairs: list[dict[str, Any]] = []
    for start in components:
        start_area = _safe_int(start.get("area"))
        start_bbox = start.get("bbox")
        start_centroid = start.get("centroid")
        if not _is_bbox(start_bbox) or not isinstance(start_centroid, Sequence):
            continue
        sx0, sy0, sx1, sy1 = [int(v) for v in list(start_bbox)[:4]]
        start_w = sx1 - sx0 + 1
        start_h = sy1 - sy0 + 1
        if start_area > 40 or start_w > 8 or start_h > 8:
            continue
        for target in components:
            if target is start:
                continue
            target_area = _safe_int(target.get("area"))
            target_bbox = target.get("bbox")
            target_centroid = target.get("centroid")
            if not _is_bbox(target_bbox) or not isinstance(target_centroid, Sequence):
                continue
            tx0, ty0, tx1, ty1 = [int(v) for v in list(target_bbox)[:4]]
            target_w = tx1 - tx0 + 1
            target_h = ty1 - ty0 + 1
            if target_area < max(16, int(start_area * 1.75)):
                continue
            if float(target_centroid[1]) <= float(start_centroid[1]) + 4.0:
                continue
            anchors = _program_shape_transform_target_anchors(
                start_bbox=[sx0, sy0, sx1, sy1],
                target_bbox=[tx0, ty0, tx1, ty1],
            )
            for anchor in anchors:
                pairs.append(
                    {
                        "actor_color": _safe_int(start.get("color"), -1),
                        "target_color": _safe_int(target.get("color"), -1),
                        "rotation_opcode": 5,
                        "shape_opcode": _program_shape_transform_shape_opcode(
                            _safe_int(target.get("color"), -1)
                        ),
                        "scale_opcode": 8 if target_area > start_area * 1.75 else -1,
                        "scale_relation": (
                            "smaller_component_to_larger_component"
                            if target_area > start_area * 1.75
                            else ""
                        ),
                        "start_bbox": [sx0, sy0, sx1, sy1],
                        "target_bbox": [tx0, ty0, tx1, ty1],
                        "target_anchor": [int(anchor[0]), int(anchor[1])],
                        "start_centroid": list(start_centroid),
                        "target_centroid": list(target_centroid),
                        "area_ratio": round(float(target_area) / float(start_area), 3),
                        "target_width": int(target_w),
                        "target_height": int(target_h),
                    }
                )
    pairs.sort(
        key=lambda row: (
            len(_program_shape_transform_bfs_ops(
                start_bbox=row["start_bbox"],
                target_anchor=row["target_anchor"],
                stride=4,
                max_ops=8,
            ))
            or 999,
            _safe_int(row.get("rotation_opcode")),
            -_safe_int(row.get("shape_opcode")),
            float(row.get("target_centroid", [0, 0])[1]),
            float(row.get("target_centroid", [0, 0])[0]),
        )
    )
    return pairs[:8]


def _program_shape_transform_target_anchors(
    *,
    start_bbox: Sequence[int],
    target_bbox: Sequence[int],
) -> list[tuple[int, int]]:
    sx0, _sy0, sx1, _sy1 = [int(v) for v in list(start_bbox)[:4]]
    tx0, ty0, tx1, _ty1 = [int(v) for v in list(target_bbox)[:4]]
    start_w = sx1 - sx0 + 1
    raw: list[tuple[int, int]] = []
    if tx0 - 4 <= sx0 <= tx1 + 4:
        raw.append((sx0, ty0))
    raw.extend(
        [
            (tx0, ty0),
            (int(round((tx0 + tx1 + 1 - start_w) / 2.0)), ty0),
            (tx1 - start_w + 1, ty0),
        ]
    )
    anchors: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in raw:
        key = (int(item[0]), int(item[1]))
        if key in seen:
            continue
        seen.add(key)
        anchors.append(key)
    return anchors


def _program_shape_transform_shape_opcode(target_color: int) -> int:
    if int(target_color) == 15:
        return 63
    if int(target_color) == 8:
        return 15
    if int(target_color) == 9:
        return 14
    return 63


def _program_translation_blockers(
    arr: np.ndarray,
    *,
    playfield_bbox: Sequence[int],
    ignored_colors: set[int],
) -> list[dict[str, Any]]:
    if not _is_bbox(playfield_bbox):
        return []
    x0, y0, x1, y1 = [int(v) for v in playfield_bbox]
    blockers: list[dict[str, Any]] = []
    for component in _connected_components(arr):
        color = _safe_int(component.get("color"), -1)
        if color in ignored_colors or color < 0:
            continue
        bbox = component.get("bbox")
        centroid = component.get("centroid")
        if not _is_bbox(bbox) or not isinstance(centroid, Sequence) or len(centroid) < 2:
            continue
        cx, cy = float(centroid[0]), float(centroid[1])
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        bx0, by0, bx1, by1 = [int(v) for v in bbox]
        box_w = bx1 - bx0 + 1
        box_h = by1 - by0 + 1
        area = _safe_int(component.get("area"))
        if area < 12:
            continue
        if box_w <= 10 and box_h <= 10:
            continue
        item = dict(component)
        item.pop("cells", None)
        blockers.append(item)
    return blockers


def _program_translation_stride(
    arr: np.ndarray,
    *,
    playfield_bbox: Sequence[int],
) -> int:
    if not _is_bbox(playfield_bbox):
        return 4
    x0, y0, x1, y1 = [int(v) for v in playfield_bbox]
    centers_by_y: dict[int, list[float]] = {}
    for component in _connected_components(arr):
        color = _safe_int(component.get("color"), -1)
        if color not in {4, 5}:
            continue
        bbox = component.get("bbox")
        centroid = component.get("centroid")
        if not _is_bbox(bbox) or not isinstance(centroid, Sequence) or len(centroid) < 2:
            continue
        cx, cy = float(centroid[0]), float(centroid[1])
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        bx0, by0, bx1, by1 = [int(v) for v in bbox]
        if bx1 - bx0 + 1 > 6 or by1 - by0 + 1 > 6:
            continue
        centers_by_y.setdefault(int(round(cy)), []).append(cx)
    gaps: list[float] = []
    for xs in centers_by_y.values():
        values = sorted(xs)
        for left, right in zip(values, values[1:]):
            gap = float(right) - float(left)
            if 2.0 <= gap <= 8.0:
                gaps.append(gap)
    if not gaps:
        return 4
    return max(1, int(round(sorted(gaps)[len(gaps) // 2])))


def _program_translation_bfs_ops(
    *,
    start: Sequence[float],
    target: Sequence[float],
    actor_bbox: Sequence[int],
    blockers: Sequence[Mapping[str, Any]],
    stride: int,
    max_ops: int,
) -> list[int]:
    if len(start) < 2 or len(target) < 2 or max_ops <= 0:
        return []
    stride = max(1, int(stride))
    sx, sy = float(start[0]), float(start[1])
    tx, ty = float(target[0]), float(target[1])
    gx = int(round((tx - sx) / float(stride)))
    gy = int(round((ty - sy) / float(stride)))
    if abs((sx + gx * stride) - tx) > stride * 0.75:
        return []
    if abs((sy + gy * stride) - ty) > stride * 0.75:
        return []
    target_state = (gx, gy)
    if target_state == (0, 0):
        return []
    op_deltas = {
        1: (-1, 0),
        2: (1, 0),
        3: (0, 1),
        33: (0, -1),
        34: (-1, 0),
    }
    queue: deque[tuple[tuple[int, int], list[int]]] = deque([((0, 0), [])])
    seen = {(0, 0)}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_ops:
            continue
        for op, delta in op_deltas.items():
            next_state = (state[0] + delta[0], state[1] + delta[1])
            if next_state in seen:
                continue
            if _program_translation_state_blocked(
                start=start,
                grid_state=next_state,
                actor_bbox=actor_bbox,
                blockers=blockers,
                stride=stride,
            ):
                continue
            next_path = path + [int(op)]
            if next_state == target_state:
                return next_path
            seen.add(next_state)
            if (
                abs(next_state[0]) <= abs(gx) + 4
                and abs(next_state[1]) <= abs(gy) + 4
            ):
                queue.append((next_state, next_path))
    return []


def _program_scale_translation_bfs_ops(
    *,
    start_bbox: Sequence[int],
    target_bbox: Sequence[int],
    stride: int,
    max_ops: int,
) -> list[int]:
    if not _is_bbox(start_bbox) or not _is_bbox(target_bbox) or max_ops <= 0:
        return []
    stride = max(1, int(stride))
    sx0, sy0, _sx1, _sy1 = [int(v) for v in start_bbox[:4]]
    tx0, ty0, _tx1, _ty1 = [int(v) for v in target_bbox[:4]]
    gx = int(round((float(tx0) - float(sx0)) / float(stride)))
    gy = int(round((float(ty0) - float(sy0)) / float(stride)))
    if abs((sx0 + gx * stride) - tx0) > stride * 0.75:
        return []
    if abs((sy0 + gy * stride) - ty0) > stride * 0.75:
        return []
    target_state = (gx, gy)
    if target_state == (0, 0):
        return []
    op_deltas = {
        1: (-1, 0),
        2: (1, 0),
        3: (0, 1),
        33: (0, -1),
        34: (-1, 0),
    }
    queue: deque[tuple[tuple[int, int], list[int]]] = deque([((0, 0), [])])
    seen = {(0, 0)}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_ops:
            continue
        for op, delta in op_deltas.items():
            next_state = (state[0] + delta[0], state[1] + delta[1])
            if next_state in seen:
                continue
            next_path = path + [int(op)]
            if next_state == target_state:
                return next_path
            seen.add(next_state)
            if (
                abs(next_state[0]) <= abs(gx) + 4
                and abs(next_state[1]) <= abs(gy) + 4
            ):
                queue.append((next_state, next_path))
    return []


def _program_shape_transform_bfs_ops(
    *,
    start_bbox: Sequence[int],
    target_anchor: Sequence[int],
    stride: int,
    max_ops: int,
) -> list[int]:
    if not _is_bbox(start_bbox) or len(target_anchor) < 2 or max_ops <= 0:
        return []
    stride = max(1, int(stride))
    sx0, sy0, _sx1, _sy1 = [int(v) for v in start_bbox[:4]]
    tx0, ty0 = int(target_anchor[0]), int(target_anchor[1])
    gx = int(round((float(tx0) - float(sx0)) / float(stride)))
    gy = int(round((float(ty0) - float(sy0)) / float(stride)))
    if abs((sx0 + gx * stride) - tx0) > stride * 0.75:
        return []
    if abs((sy0 + gy * stride) - ty0) > stride * 0.75:
        return []
    target_state = (gx, gy)
    if target_state == (0, 0):
        return []
    op_deltas = {
        1: (-1, 0),
        2: (1, 0),
        3: (0, 1),
        33: (0, -1),
        34: (-1, 0),
    }
    queue: deque[tuple[tuple[int, int], list[int]]] = deque([((0, 0), [])])
    seen = {(0, 0)}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_ops:
            continue
        for op, delta in op_deltas.items():
            next_state = (state[0] + delta[0], state[1] + delta[1])
            if next_state in seen:
                continue
            next_path = path + [int(op)]
            if next_state == target_state:
                return next_path
            seen.add(next_state)
            if (
                abs(next_state[0]) <= abs(gx) + 4
                and abs(next_state[1]) <= abs(gy) + 4
            ):
                queue.append((next_state, next_path))
    return []


def _program_checkpoint_route_model(
    arr: np.ndarray,
    *,
    playfield_bbox: Sequence[int],
    row_count: int,
    bit_count: int,
    max_runs: int,
) -> dict[str, Any]:
    pair = _program_checkpoint_actor_target_pair(arr, playfield_bbox=playfield_bbox)
    if not pair:
        return {"route_state": "no_compact_actor_target_pair", "programs": []}
    start = tuple(int(v) for v in list(pair.get("start_site") or [])[:2])
    target = tuple(int(v) for v in list(pair.get("target_site") or [])[:2])
    actor_color = _safe_int(pair.get("actor_color"), -1)
    if len(start) != 2 or len(target) != 2 or actor_color < 0:
        return {"route_state": "invalid_actor_target_pair", "programs": []}
    lattice = _program_checkpoint_lattice(
        arr,
        playfield_bbox=playfield_bbox,
        start=start,
        target=target,
        actor_color=actor_color,
    )
    checkpoints = set(tuple(row) for row in list(lattice.get("checkpoints") or []))
    walls = set(tuple(row) for row in list(lattice.get("walls") or []))
    open_cells = set(tuple(row) for row in list(lattice.get("open_cells") or []))
    if len(checkpoints) < 1:
        return {
            "route_state": "no_checkpoint_like_cells",
            "start_site": list(start),
            "target_site": list(target),
            "checkpoints": [],
            "walls": sorted(walls),
            "programs": [],
        }
    programs = _program_checkpoint_multi_run_programs(
        start=start,
        target=target,
        checkpoints=checkpoints,
        open_cells=open_cells,
        walls=walls,
        row_count=row_count,
        bit_count=bit_count,
        max_runs=max_runs,
    )
    route_state = (
        "checkpoint_route_supported" if programs else "checkpoint_route_not_found"
    )
    return {
        "route_state": route_state,
        "start_site": list(start),
        "target_site": list(target),
        "actor_color": int(actor_color),
        "checkpoints": [list(row) for row in sorted(checkpoints)],
        "walls": [list(row) for row in sorted(walls)],
        "programs": [list(program) for program in programs],
    }


def _program_checkpoint_actor_target_pair(
    arr: np.ndarray,
    *,
    playfield_bbox: Sequence[int],
) -> dict[str, Any]:
    if not _is_bbox(playfield_bbox):
        return {}
    x0, y0, x1, y1 = [int(v) for v in playfield_bbox[:4]]
    by_color: dict[int, list[dict[str, Any]]] = {}
    for component in _connected_components(arr):
        color = _safe_int(component.get("color"), -1)
        if color in {0, 1, 2, 3, 5, 9} or color < 0:
            continue
        bbox = component.get("bbox")
        centroid = component.get("centroid")
        if not _is_bbox(bbox) or not isinstance(centroid, Sequence):
            continue
        cx = float(centroid[0])
        cy = float(centroid[1])
        if not (x0 <= cx <= x1 and y0 <= cy <= y1 + 4):
            continue
        bx0, by0, bx1, by1 = [int(v) for v in list(bbox)[:4]]
        width = bx1 - bx0 + 1
        height = by1 - by0 + 1
        area = _safe_int(component.get("area"))
        if area < 6 or area > 48 or width > 8 or height > 8:
            continue
        by_color.setdefault(color, []).append(dict(component))

    pair_candidates: list[dict[str, Any]] = []
    for color, components in sorted(by_color.items()):
        if len(components) != 2:
            continue
        ordered = sorted(
            components,
            key=lambda row: (
                float(row.get("centroid", [0.0, 0.0])[1]),
                float(row.get("centroid", [0.0, 0.0])[0]),
            ),
        )
        target_component = ordered[0]
        start_component = ordered[-1]
        start_bbox = list(start_component.get("bbox") or [])
        target_bbox = list(target_component.get("bbox") or [])
        if not _is_bbox(start_bbox) or not _is_bbox(target_bbox):
            continue
        vertical_span = abs(int(start_bbox[1]) - int(target_bbox[1]))
        if vertical_span < 8:
            continue
        target_site = _program_checkpoint_target_site(
            start_bbox=start_bbox,
            target_bbox=target_bbox,
        )
        pair_candidates.append(
            {
                "actor_color": int(color),
                "start_site": [int(start_bbox[0]), int(start_bbox[1])],
                "target_site": [int(target_site[0]), int(target_site[1])],
                "start_bbox": [int(v) for v in start_bbox[:4]],
                "target_bbox": [int(v) for v in target_bbox[:4]],
                "vertical_span": int(vertical_span),
            }
        )
    pair_candidates.sort(
        key=lambda row: (
            -_safe_int(row.get("vertical_span")),
            _safe_int(row.get("target_site", [0, 0])[1]),
            _safe_int(row.get("target_site", [0, 0])[0]),
        )
    )
    return pair_candidates[0] if pair_candidates else {}


def _program_checkpoint_target_site(
    *,
    start_bbox: Sequence[int],
    target_bbox: Sequence[int],
) -> tuple[int, int]:
    sx0, sy0, sx1, sy1 = [int(v) for v in list(start_bbox)[:4]]
    tx0, ty0, tx1, ty1 = [int(v) for v in list(target_bbox)[:4]]
    start_w = sx1 - sx0 + 1
    start_h = sy1 - sy0 + 1
    target_w = tx1 - tx0 + 1
    target_h = ty1 - ty0 + 1
    if target_w > start_w or target_h > start_h:
        return (tx0 + 1, ty0 + 1)
    return (tx0, ty0)


def _program_checkpoint_lattice(
    arr: np.ndarray,
    *,
    playfield_bbox: Sequence[int],
    start: tuple[int, int],
    target: tuple[int, int],
    actor_color: int,
    stride: int = 4,
) -> dict[str, Any]:
    height, width = arr.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in playfield_bbox[:4]]
    x_min = max(0, x0)
    x_max = min(width - stride, max(x1, start[0], target[0]))
    y_min = max(0, y0)
    y_max = min(height - stride, max(y1, start[1], target[1]))
    x_residue = int(start[0]) % int(stride)
    y_residue = int(start[1]) % int(stride)
    x_values = [
        int(x)
        for x in range(x_min, x_max + 1)
        if int(x) % int(stride) == x_residue
    ]
    y_values = [
        int(y)
        for y in range(y_min, y_max + 1)
        if int(y) % int(stride) == y_residue
    ]

    blocker_colors = _program_checkpoint_blocker_colors(
        arr,
        playfield_bbox=playfield_bbox,
        actor_color=actor_color,
    )
    open_cells: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()
    checkpoints: set[tuple[int, int]] = set()
    for y in y_values:
        for x in x_values:
            block = arr[y : y + stride, x : x + stride]
            if block.shape[0] < stride or block.shape[1] < stride:
                continue
            counts = Counter(int(v) for v in block.reshape(-1).tolist())
            dominant_color, dominant_count = counts.most_common(1)[0]
            cell = (int(x), int(y))
            if (
                dominant_color in blocker_colors
                and dominant_count >= int(stride * stride * 0.6)
            ):
                walls.add(cell)
            else:
                open_cells.add(cell)
            if (
                int(counts.get(actor_color, 0)) >= 2
                and cell not in {start, target}
                and abs(cell[0] - target[0]) + abs(cell[1] - target[1]) > stride
            ):
                checkpoints.add(cell)
                open_cells.add(cell)
                walls.discard(cell)
    open_cells.add(start)
    open_cells.add(target)
    walls.discard(start)
    walls.discard(target)
    return {
        "open_cells": [list(row) for row in sorted(open_cells)],
        "walls": [list(row) for row in sorted(walls)],
        "checkpoints": [list(row) for row in sorted(checkpoints)],
        "blocker_colors": sorted(int(color) for color in blocker_colors),
    }


def _program_checkpoint_blocker_colors(
    arr: np.ndarray,
    *,
    playfield_bbox: Sequence[int],
    actor_color: int,
) -> set[int]:
    if not _is_bbox(playfield_bbox):
        return set()
    x0, y0, x1, y1 = [int(v) for v in playfield_bbox[:4]]
    scores: Counter[int] = Counter()
    for component in _connected_components(arr):
        color = _safe_int(component.get("color"), -1)
        if color in {0, 1, 2, 3, 5, 9, actor_color} or color < 0:
            continue
        bbox = component.get("bbox")
        centroid = component.get("centroid")
        if not _is_bbox(bbox) or not isinstance(centroid, Sequence):
            continue
        cx = float(centroid[0])
        cy = float(centroid[1])
        if not (x0 <= cx <= x1 and y0 <= cy <= y1 + 4):
            continue
        area = _safe_int(component.get("area"))
        if area >= 40:
            scores[color] += area
    if not scores:
        return set()
    max_score = max(scores.values())
    return {int(color) for color, score in scores.items() if score >= max_score * 0.5}


def _program_checkpoint_multi_run_programs(
    *,
    start: tuple[int, int],
    target: tuple[int, int],
    checkpoints: set[tuple[int, int]],
    open_cells: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    row_count: int,
    bit_count: int,
    max_runs: int,
) -> list[list[int]]:
    if row_count <= 0 or bit_count <= 0 or max_runs <= 0:
        return []
    complete: list[tuple[list[list[int]], frozenset[tuple[int, int]]]] = []
    queue: deque[
        tuple[tuple[int, int], frozenset[tuple[int, int]], list[list[int]]]
    ] = deque([(start, frozenset(checkpoints), [])])
    seen = {(start, frozenset(checkpoints), 0)}
    while queue:
        pos, remaining, programs = queue.popleft()
        if pos == target:
            complete.append((programs, remaining))
            continue
        if len(programs) >= max_runs:
            continue
        goals = set(remaining)
        goals.add(target)
        for next_pos, program in _program_checkpoint_reachable_segments(
            start=pos,
            goals=goals,
            open_cells=open_cells,
            walls=walls,
            row_count=row_count,
            bit_count=bit_count,
        ):
            next_remaining = frozenset(cell for cell in remaining if cell != next_pos)
            next_programs = programs + [program]
            key = (next_pos, next_remaining, len(next_programs))
            if key in seen:
                continue
            seen.add(key)
            queue.append((next_pos, next_remaining, next_programs))
    if not complete:
        return []

    has_direct_decoy = any(
        len(checkpoints) - len(remaining) <= 0 for _, remaining in complete
    )

    def _route_rank(
        row: tuple[list[list[int]], frozenset[tuple[int, int]]],
    ) -> tuple[int, int, int, int, list[list[int]]]:
        programs, remaining = row
        checkpoint_save_count = max(0, len(checkpoints) - len(remaining))
        nonzero_count = sum(
            1 for program in programs for op in program if int(op) != 0
        )
        if has_direct_decoy:
            return (
                0 if checkpoint_save_count > 0 else 1,
                -int(checkpoint_save_count),
                int(len(programs)),
                int(nonzero_count),
                programs,
            )
        return (
            0 if checkpoint_save_count > 0 else 1,
            int(len(programs)),
            int(nonzero_count),
            -int(checkpoint_save_count),
            programs,
        )

    complete.sort(key=_route_rank)
    return complete[0][0]


def _program_checkpoint_reachable_segments(
    *,
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    open_cells: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    row_count: int,
    bit_count: int,
) -> list[tuple[tuple[int, int], list[int]]]:
    op_deltas = {
        1: (-4, 0),
        2: (4, 0),
        3: (0, 4),
        33: (0, -4),
        10: (8, 0),
        11: (8, 0),
        12: (-8, 0),
        13: (-8, 0),
        34: (-4, 0),
    }
    max_opcode = 1 << max(0, int(bit_count))
    ops = [
        (int(op), delta)
        for op, delta in op_deltas.items()
        if 0 <= int(op) < max_opcode
    ]
    queue: deque[tuple[tuple[int, int], list[int]]] = deque([(start, [])])
    seen = {start}
    found: list[tuple[tuple[int, int], list[int]]] = []
    while queue:
        pos, path = queue.popleft()
        if path and pos in goals:
            padded = path + [0] * max(0, int(row_count) - len(path))
            found.append((pos, padded[: int(row_count)]))
            continue
        if len(path) >= int(row_count):
            continue
        for op, (dx, dy) in ops:
            next_pos = (int(pos[0] + dx), int(pos[1] + dy))
            if next_pos in seen or next_pos in walls:
                continue
            if next_pos not in open_cells and next_pos not in goals:
                continue
            seen.add(next_pos)
            queue.append((next_pos, path + [int(op)]))
    found.sort(
        key=lambda row: (
            sum(1 for op in row[1] if int(op) != 0),
            row[0][1],
            row[0][0],
        )
    )
    return found


def _program_translation_state_blocked(
    *,
    start: Sequence[float],
    grid_state: tuple[int, int],
    actor_bbox: Sequence[int],
    blockers: Sequence[Mapping[str, Any]],
    stride: int,
) -> bool:
    if len(start) < 2:
        return True
    cx = float(start[0]) + float(grid_state[0] * stride)
    cy = float(start[1]) + float(grid_state[1] * stride)
    expansion = 0.0
    if _is_bbox(actor_bbox):
        ax0, ay0, ax1, ay1 = [float(v) for v in actor_bbox]
        half_w = max(1.0, (ax1 - ax0 + 1.0) / 2.0)
        half_h = max(1.0, (ay1 - ay0 + 1.0) / 2.0)
    else:
        half_w = half_h = expansion
    actor = [cx - half_w, cy - half_h, cx + half_w, cy + half_h]
    for blocker in blockers:
        bbox = blocker.get("bbox")
        if not _is_bbox(bbox):
            continue
        bx0, by0, bx1, by1 = [float(v) for v in bbox]
        expanded = [bx0 - expansion, by0 - expansion, bx1 + expansion, by1 + expansion]
        if not (
            actor[2] <= expanded[0]
            or actor[0] >= expanded[2]
            or actor[3] <= expanded[1]
            or actor[1] >= expanded[3]
        ):
            return True
    return False


def _program_row_current_opcodes(
    cells: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[float],
    bit_centers: Sequence[float],
    active_color: int,
) -> list[int]:
    current: list[int] = []
    for x_center in columns:
        opcode = 0
        for bit_index, y_center in enumerate(bit_centers):
            cell = _nearest_program_cell(
                cells,
                x_center=float(x_center),
                y_center=float(y_center),
            )
            if cell is None:
                continue
            if _safe_int(cell.get("color"), -1) == int(active_color):
                opcode |= 1 << int(bit_index)
        current.append(int(opcode))
    return current


def _compile_program_row_multirun_opcode_plan(
    cells: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[float],
    bit_centers: Sequence[float],
    desired_programs: Sequence[Sequence[int]],
    active_color: int,
    inactive_color: int,
    run_site: Sequence[int],
    panel_index: int,
) -> list[dict[str, Any]]:
    current = _program_row_current_opcodes(
        cells,
        columns=columns,
        bit_centers=bit_centers,
        active_color=active_color,
    )
    steps: list[dict[str, Any]] = []
    for run_index, desired in enumerate(desired_programs):
        desired_values = [
            int(desired[col_index]) if col_index < len(desired) else 0
            for col_index in range(len(columns))
        ]
        for col_index, x_center in enumerate(columns):
            target_opcode = int(desired_values[col_index])
            current_opcode = int(current[col_index]) if col_index < len(current) else 0
            for bit_index, y_center in enumerate(bit_centers):
                if ((current_opcode >> bit_index) & 1) == (
                    (target_opcode >> bit_index) & 1
                ):
                    continue
                cell = _nearest_program_cell(
                    cells,
                    x_center=float(x_center),
                    y_center=float(y_center),
                )
                if cell is None:
                    continue
                site = _centroid_site(cell.get("centroid"))
                if site is None:
                    continue
                target_color = (
                    int(active_color)
                    if target_opcode & (1 << bit_index)
                    else int(inactive_color)
                )
                steps.append(
                    {
                        "action": "ACTION6",
                        "x": int(site[0]),
                        "y": int(site[1]),
                        "role": "checkpoint_bfs_setup_toggle",
                        "candidate_kind": "program_checkpoint_bfs_setup_toggle",
                        "run_index": int(run_index),
                        "column_index": int(col_index),
                        "bit_index": int(bit_index),
                        "opcode": int(target_opcode),
                        "source_opcode": int(current_opcode),
                        "target_color": int(target_color),
                        "active_color": int(active_color),
                        "inactive_color": int(inactive_color),
                    }
                )
        steps.append(
            {
                "action": "ACTION6",
                "x": int(run_site[0]),
                "y": int(run_site[1]),
                "role": "run",
                "candidate_kind": "program_checkpoint_bfs_run_candidate",
                "run_index": int(run_index),
                "panel_index": int(panel_index),
                "active_color": int(active_color),
            }
        )
        current = list(desired_values)
    return steps


def _compile_program_row_opcode_plan(
    cells: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[float],
    bit_centers: Sequence[float],
    desired_opcodes: Sequence[int],
    active_color: int,
    inactive_color: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for col_index, x_center in enumerate(columns):
        opcode = int(desired_opcodes[col_index]) if col_index < len(desired_opcodes) else 0
        for bit_index, y_center in enumerate(bit_centers):
            desired_color = int(active_color) if opcode & (1 << bit_index) else int(inactive_color)
            cell = _nearest_program_cell(
                cells,
                x_center=float(x_center),
                y_center=float(y_center),
            )
            if cell is None:
                continue
            current_color = _safe_int(cell.get("color"), -1)
            if current_color == desired_color:
                continue
            site = _centroid_site(cell.get("centroid"))
            if site is None:
                continue
            steps.append(
                {
                    "action": "ACTION6",
                    "x": int(site[0]),
                    "y": int(site[1]),
                    "role": "translation_bfs_setup_toggle",
                    "candidate_kind": "program_translation_bfs_setup_toggle",
                    "column_index": int(col_index),
                    "bit_index": int(bit_index),
                    "opcode": int(opcode),
                    "source_color": int(current_color),
                    "target_color": int(desired_color),
                    "active_color": int(active_color),
                    "inactive_color": int(inactive_color),
                }
            )
    return steps


def _nearest_index(
    values: Sequence[float],
    target: float,
    *,
    max_distance: float,
) -> int | None:
    best_index: int | None = None
    best_distance = float(max_distance)
    for index, value in enumerate(values):
        distance = abs(float(value) - float(target))
        if distance <= best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _union_bbox(bboxes: Sequence[Any]) -> list[int] | None:
    valid = [[int(v) for v in bbox[:4]] for bbox in bboxes if _is_bbox(bbox)]
    if not valid:
        return None
    return [
        min(row[0] for row in valid),
        min(row[1] for row in valid),
        max(row[2] for row in valid),
        max(row[3] for row in valid),
    ]


def _connected_components(arr: np.ndarray) -> list[dict[str, Any]]:
    height, width = arr.shape[:2]
    seen = np.zeros((height, width), dtype=bool)
    out: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x]:
                continue
            color = int(arr[y, x])
            stack = [(x, y)]
            seen[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                px, py = stack.pop()
                xs.append(px)
                ys.append(py)
                for nx, ny in (
                    (px + 1, py),
                    (px - 1, py),
                    (px, py + 1),
                    (px, py - 1),
                ):
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and not seen[ny, nx]
                        and int(arr[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            area = len(xs)
            out.append(
                {
                    "id": f"c{len(out)}",
                    "color": int(color),
                    "area": int(area),
                    "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    "centroid": [
                        round(float(sum(xs)) / float(area), 3),
                        round(float(sum(ys)) / float(area), 3),
                    ],
                }
            )
    return out


def _mechanic_state_type(packet: Mapping[str, Any]) -> str:
    types = set(str(row) for row in list(packet.get("candidate_types") or []))
    if "score_closure" in types:
        return "program_score_closure"
    if "program_execution" in types:
        return "program_execution_relation"
    if "program_bit_toggle" in types:
        return "program_bit_toggle_relation"
    if "run_button" in types:
        return "program_run_button_relation"
    return "program_panel_observation"


def _selected_site(action_payload: Mapping[str, Any]) -> tuple[int, int] | None:
    if "x" not in action_payload or "y" not in action_payload:
        return None
    try:
        return int(action_payload["x"]), int(action_payload["y"])
    except (TypeError, ValueError):
        return None


def _normalize_action_name(action: Any) -> str:
    name = getattr(action, "name", action)
    text = str(name or "").strip().upper()
    if text.startswith("GAMEACTION."):
        text = text.split(".", 1)[1]
    if text.isdigit():
        return f"ACTION{int(text)}"
    return text


def _as_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.int16)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError("frame must be a 2D or stacked 3D array")
    return arr


def _frame_hash(frame: np.ndarray) -> str:
    arr = np.ascontiguousarray(frame)
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    return f"sha256:{digest[:16]}"


def _is_xy(value: Any) -> bool:
    if not isinstance(value, Sequence) or len(value) < 2:
        return False
    try:
        int(value[0])
        int(value[1])
    except (TypeError, ValueError):
        return False
    return True


def _is_bbox(value: Any) -> bool:
    if not isinstance(value, Sequence) or len(value) < 4:
        return False
    try:
        int(value[0])
        int(value[1])
        int(value[2])
        int(value[3])
    except (TypeError, ValueError):
        return False
    return True


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, set):
        return sorted(_sanitize(item) for item in value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
    return value


__all__ = [
    "PROGRAM_ROW_SEMANTICS_ALL25_NON_REGRESSION_SCHEMA_VERSION",
    "PROGRAM_ROW_SEMANTICS_CANDIDATE_MINER_SCHEMA_VERSION",
    "PROGRAM_ROW_SEMANTICS_CHECKPOINT_BFS_PROPOSAL_SCHEMA_VERSION",
    "PROGRAM_ROW_SEMANTICS_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION",
    "PROGRAM_ROW_SEMANTICS_DIAGNOSTIC_SENSOR_SCHEMA_VERSION",
    "PROGRAM_ROW_SEMANTICS_EDGE_BIT_PAIR_PROPOSAL_SCHEMA_VERSION",
    "PROGRAM_ROW_SEMANTICS_GUARDED_PROBE_OUTCOME_SCHEMA_VERSION",
    "PROGRAM_ROW_SEMANTICS_PACKET_SCHEMA_VERSION",
    "PROGRAM_ROW_SEMANTICS_POLARITY_FILL_PROPOSAL_SCHEMA_VERSION",
    "PROGRAM_ROW_SEMANTICS_POSTERIOR_SCHEMA_VERSION",
    "PROGRAM_ROW_SEMANTICS_SCALE_TRANSLATION_BFS_PROPOSAL_SCHEMA_VERSION",
    "PROGRAM_ROW_SEMANTICS_SHAPE_TRANSFORM_BFS_PROPOSAL_SCHEMA_VERSION",
    "PROGRAM_ROW_SEMANTICS_TRANSLATION_BFS_PROPOSAL_SCHEMA_VERSION",
    "build_program_row_semantics_checkpoint_bfs_proposals",
    "build_program_row_semantics_edge_bit_pair_proposals",
    "build_program_row_semantics_polarity_fill_proposals",
    "build_program_row_semantics_packet",
    "build_program_row_semantics_scale_translation_bfs_proposals",
    "build_program_row_semantics_shape_transform_bfs_proposals",
    "build_program_row_semantics_translation_bfs_proposals",
    "summarize_program_row_semantics_all25_non_regression",
    "summarize_program_row_semantics_candidate_miner",
    "summarize_program_row_semantics_counterfactual_controls",
    "summarize_program_row_semantics_guarded_probe_outcomes",
    "summarize_program_row_semantics_posterior",
    "summarize_program_row_semantics_packets",
    "validate_program_row_semantics_packet",
]
