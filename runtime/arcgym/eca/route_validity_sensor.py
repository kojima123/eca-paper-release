"""Frame-only route validity diagnostics.

The sensor is intentionally observational. It summarizes passable-region
topology and blocked movement evidence from public frames. It does not choose
actions, emit plans, or authorize a solver.
"""
from __future__ import annotations

import hashlib
from collections import Counter, deque
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROUTE_VALIDITY_PACKET_SCHEMA_VERSION = "eca_route_validity_sensor_packet.v1"
ROUTE_VALIDITY_TRACE_SENSOR_SCHEMA_VERSION = "eca_route_validity_diagnostic_sensor.v1"
ROUTE_VALIDITY_PROGRESS_POSTERIOR_SCHEMA_VERSION = (
    "eca_route_validity_progress_posterior.v1"
)
ROUTE_VALIDITY_RECOVERY_CANDIDATE_MINER_SCHEMA_VERSION = (
    "eca_route_validity_recovery_candidate_miner.v1"
)
ROUTE_VALIDITY_ALL25_NON_REGRESSION_SCHEMA_VERSION = (
    "eca_route_validity_all25_non_regression_sensor.v1"
)

_MOVE_ACTION_DELTAS = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}


def build_route_validity_packet(
    frame: Any,
    *,
    before_frame: Any | None = None,
    game_id: str = "",
    state_hash: str = "",
    step: int = 0,
    source_site: Sequence[Any] | None = None,
    target_site: Sequence[Any] | None = None,
    action_name: str = "",
    action_payload: Mapping[str, Any] | None = None,
    score_delta: int = 0,
) -> dict[str, Any]:
    """Build a compact route-validity packet from a public frame."""

    arr = _as_frame(frame)
    before = _as_frame(before_frame) if before_frame is not None else None
    topology = _route_topology(arr, source_site=source_site, target_site=target_site)
    before_topology = (
        _route_topology(before, source_site=source_site, target_site=target_site)
        if before is not None and before.shape == arr.shape
        else None
    )
    topology_delta = _route_topology_delta(before_topology, topology)
    action = _normalize_action_name(action_name)
    changed_cell_count = (
        int(np.count_nonzero(before != arr))
        if before is not None and before.shape == arr.shape
        else -1
    )
    blocked_action = bool(
        action in _MOVE_ACTION_DELTAS
        and before is not None
        and before.shape == arr.shape
        and changed_cell_count == 0
    )
    return _sanitize(
        {
            "schema_version": ROUTE_VALIDITY_PACKET_SCHEMA_VERSION,
            "sensor_family": "route_validity_sensor",
            "game_id": str(game_id or "").strip().lower(),
            "state_hash": str(state_hash or _frame_hash(arr)),
            "step": int(step),
            "source_site": list(source_site) if _is_xy(source_site) else None,
            "target_site": list(target_site) if _is_xy(target_site) else None,
            "action_to_route_delta": {
                "action_name": action,
                "payload": dict(action_payload or {}),
                "changed_cell_count": int(changed_cell_count),
                "blocked_action_candidate": blocked_action,
                "move_delta": list(_MOVE_ACTION_DELTAS.get(action, (0, 0))),
            },
            "passable_color_candidates": topology["passable_color_candidates"],
            "passable_candidate_count": topology["passable_candidate_count"],
            "passable_region_count": topology["passable_region_count"],
            "largest_passable_region_area": topology["largest_passable_region_area"],
            "largest_passable_region_bbox": topology["largest_passable_region_bbox"],
            "narrow_corridor_cell_count": topology["narrow_corridor_cell_count"],
            "dead_end_cell_count": topology["dead_end_cell_count"],
            "obstacle_component_count": topology["obstacle_component_count"],
            "source_target_route_state": topology["source_target_route_state"],
            "source_target_connected": topology["source_target_connected"],
            "source_target_blocked": topology["source_target_blocked"],
            "site_color_mismatch": topology["site_color_mismatch"],
            "route_ambiguity": topology["route_ambiguity"],
            "route_topology_delta": topology_delta,
            "route_progress_candidate": _route_progress_candidate(
                topology_delta,
                topology,
                score_delta=int(score_delta),
            ),
            "route_regression_candidate": _route_regression_candidate(
                topology_delta,
                topology,
            ),
            "score_delta": int(score_delta),
            "diagnostic_only": True,
        }
    )


def validate_route_validity_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if str(packet.get("schema_version", "") or "") != ROUTE_VALIDITY_PACKET_SCHEMA_VERSION:
        reasons.append("schema_version_mismatch")
    if str(packet.get("sensor_family", "") or "") != "route_validity_sensor":
        reasons.append("sensor_family_mismatch")
    passable_count = _safe_int(packet.get("passable_candidate_count"))
    region_count = _safe_int(packet.get("passable_region_count"))
    largest_area = _safe_int(packet.get("largest_passable_region_area"))
    if passable_count <= 0:
        reasons.append("missing_passable_color_candidates")
    if region_count <= 0 or largest_area <= 0:
        reasons.append("missing_passable_region_topology")
    route_state = str(packet.get("source_target_route_state", "") or "")
    if bool(packet.get("source_target_connected")) and route_state != "source_target_connected":
        reasons.append("connected_state_mismatch")
    if bool(packet.get("source_target_blocked")) and route_state != "source_target_blocked":
        reasons.append("blocked_state_mismatch")
    return {
        "schema_version": "eca_route_validity_validation.v1",
        "accepted": not reasons,
        "rejected_reasons": reasons,
        "family": "route_validity",
        "mechanic_state_type": _mechanic_state_type(packet),
        "goal_link_type": "route_validity_progress",
        "passable_candidate_count": passable_count,
        "passable_region_count": region_count,
        "largest_passable_region_area": largest_area,
        "blocked_action_candidate": bool(
            (packet.get("action_to_route_delta") or {}).get("blocked_action_candidate")
            if isinstance(packet.get("action_to_route_delta"), Mapping)
            else False
        ),
    }


def summarize_route_validity_packets(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 12,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    validation_counts: Counter[str] = Counter()
    route_state_counts: Counter[str] = Counter()
    mechanic_state_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    samples: list[dict[str, Any]] = []
    bottleneck_sum = 0
    dead_end_sum = 0
    obstacle_sum = 0
    blocked_action_count = 0
    connected_count = 0
    blocked_count = 0

    for packet in packet_rows:
        validation = validate_route_validity_packet(packet)
        accepted = bool(validation.get("accepted"))
        validation_counts["accepted" if accepted else "rejected"] += 1
        route_state = str(packet.get("source_target_route_state", "") or "unknown")
        mechanic_state = str(validation.get("mechanic_state_type", "") or "unknown")
        route_state_counts[route_state] += 1
        mechanic_state_counts[mechanic_state] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["packet_count"] += 1
        by_game[game_id][f"route_state:{route_state}"] += 1
        bottleneck_sum += _safe_int(packet.get("narrow_corridor_cell_count"))
        dead_end_sum += _safe_int(packet.get("dead_end_cell_count"))
        obstacle_sum += _safe_int(packet.get("obstacle_component_count"))
        action_delta = packet.get("action_to_route_delta")
        action_delta = action_delta if isinstance(action_delta, Mapping) else {}
        if bool(action_delta.get("blocked_action_candidate")):
            blocked_action_count += 1
            by_game[game_id]["blocked_action_candidate_count"] += 1
        if bool(packet.get("source_target_connected")):
            connected_count += 1
        if bool(packet.get("source_target_blocked")):
            blocked_count += 1
        if len(samples) < sample_limit:
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "route_state": route_state,
                    "mechanic_state_type": mechanic_state,
                    "passable_region_count": _safe_int(packet.get("passable_region_count")),
                    "largest_passable_region_area": _safe_int(
                        packet.get("largest_passable_region_area")
                    ),
                    "narrow_corridor_cell_count": _safe_int(
                        packet.get("narrow_corridor_cell_count")
                    ),
                    "blocked_action_candidate": bool(
                        action_delta.get("blocked_action_candidate")
                    ),
                    "action_name": str(action_delta.get("action_name", "") or ""),
                }
            )

    return {
        "schema_version": ROUTE_VALIDITY_TRACE_SENSOR_SCHEMA_VERSION,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
        "packet_count": int(len(packet_rows)),
        "accepted_packet_count": int(validation_counts.get("accepted", 0)),
        "rejected_packet_count": int(validation_counts.get("rejected", 0)),
        "route_state_counts": dict(sorted(route_state_counts.items())),
        "mechanic_state_counts": dict(sorted(mechanic_state_counts.items())),
        "blocked_action_candidate_count": int(blocked_action_count),
        "source_target_connected_count": int(connected_count),
        "source_target_blocked_count": int(blocked_count),
        "narrow_corridor_cell_count_sum": int(bottleneck_sum),
        "dead_end_cell_count_sum": int(dead_end_sum),
        "obstacle_component_count_sum": int(obstacle_sum),
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "route_validity_posterior with positive counterfactual margin",
            "allowed_connection": "diagnostic_only_until_route_posterior_and_controls",
            "action_generation_allowed": False,
        },
    }


def summarize_route_validity_progress_posterior(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_action: dict[str, Counter[str]] = {}
    samples: list[dict[str, Any]] = []
    route_progress_score_delta_sum = 0
    blocked_action_score_delta_sum = 0
    no_progress_score_delta_sum = 0
    score_delta_sum = 0

    for packet in packet_rows:
        validation = validate_route_validity_packet(packet)
        accepted = bool(validation.get("accepted"))
        counts["accepted_packet_count" if accepted else "rejected_packet_count"] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        action_delta = packet.get("action_to_route_delta")
        action_delta = action_delta if isinstance(action_delta, Mapping) else {}
        action_name = str(action_delta.get("action_name", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_action.setdefault(action_name, Counter())
        by_game[game_id]["packet_count"] += 1
        by_action[action_name]["packet_count"] += 1
        score_delta = _safe_int(packet.get("score_delta"))
        score_delta_sum += score_delta
        progress = bool(packet.get("route_progress_candidate"))
        regression = bool(packet.get("route_regression_candidate"))
        blocked_action = bool(action_delta.get("blocked_action_candidate"))
        if not accepted:
            state = "rejected_descriptor"
        elif progress and score_delta > 0:
            state = "route_progress_score_closure_confirmed"
            route_progress_score_delta_sum += score_delta
            counts["confirmed_score_closure_count"] += 1
        elif progress:
            state = "scoreless_route_progress_unconfirmed"
            counts["scoreless_route_progress_count"] += 1
        elif blocked_action:
            state = "blocked_action_negative_control"
            blocked_action_score_delta_sum += max(0, score_delta)
            counts["blocked_action_candidate_count"] += 1
        elif regression:
            state = "route_regression_observed"
            no_progress_score_delta_sum += max(0, score_delta)
            counts["route_regression_candidate_count"] += 1
        else:
            state = "route_observed_no_progress"
            no_progress_score_delta_sum += max(0, score_delta)
        counts[f"posterior_state:{state}"] += 1
        by_game[game_id][f"posterior_state:{state}"] += 1
        by_action[action_name][f"posterior_state:{state}"] += 1
        if progress:
            counts["route_progress_candidate_count"] += 1
            by_game[game_id]["route_progress_candidate_count"] += 1
            by_action[action_name]["route_progress_candidate_count"] += 1
        if score_delta > 0:
            counts["score_positive_packet_count"] += 1
            by_game[game_id]["score_positive_packet_count"] += 1
            by_action[action_name]["score_positive_packet_count"] += 1
        if len(samples) < sample_limit:
            delta = packet.get("route_topology_delta")
            delta = delta if isinstance(delta, Mapping) else {}
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "action_name": action_name,
                    "posterior_state": state,
                    "score_delta": score_delta,
                    "route_progress_candidate": progress,
                    "blocked_action_candidate": blocked_action,
                    "route_regression_candidate": regression,
                    "largest_area_delta": _safe_int(
                        delta.get("largest_passable_region_area_delta")
                    ),
                    "passable_region_count_delta": _safe_int(
                        delta.get("passable_region_count_delta")
                    ),
                    "obstacle_component_count_delta": _safe_int(
                        delta.get("obstacle_component_count_delta")
                    ),
                }
            )

    negative_control = max(blocked_action_score_delta_sum, no_progress_score_delta_sum)
    control_margin = int(route_progress_score_delta_sum - negative_control)
    if counts.get("confirmed_score_closure_count", 0) > 0 and control_margin > 0:
        promotion_state = "route_progress_counterfactual_margin"
    elif route_progress_score_delta_sum > 0:
        promotion_state = "route_progress_control_matched"
    elif counts.get("scoreless_route_progress_count", 0) > 0:
        promotion_state = "route_progress_scoreless_unconfirmed"
    elif counts.get("blocked_action_candidate_count", 0) > 0:
        promotion_state = "blocked_route_action_observed"
    else:
        promotion_state = "no_route_progress_evidence"
    return {
        "schema_version": ROUTE_VALIDITY_PROGRESS_POSTERIOR_SCHEMA_VERSION,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "packet_count": int(len(packet_rows)),
        "accepted_packet_count": int(counts.get("accepted_packet_count", 0)),
        "rejected_packet_count": int(counts.get("rejected_packet_count", 0)),
        "score_delta_sum": int(score_delta_sum),
        "score_positive_packet_count": int(counts.get("score_positive_packet_count", 0)),
        "route_progress_candidate_count": int(
            counts.get("route_progress_candidate_count", 0)
        ),
        "scoreless_route_progress_count": int(
            counts.get("scoreless_route_progress_count", 0)
        ),
        "blocked_action_candidate_count": int(
            counts.get("blocked_action_candidate_count", 0)
        ),
        "route_regression_candidate_count": int(
            counts.get("route_regression_candidate_count", 0)
        ),
        "confirmed_score_closure_count": int(
            counts.get("confirmed_score_closure_count", 0)
        ),
        "route_progress_score_delta_sum": int(route_progress_score_delta_sum),
        "blocked_action_negative_control_score_delta_sum": int(
            blocked_action_score_delta_sum
        ),
        "no_progress_negative_control_score_delta_sum": int(no_progress_score_delta_sum),
        "negative_control_max_score_delta_sum": int(negative_control),
        "route_progress_counterfactual_margin": int(control_margin),
        "positive_control_margin_count": int(control_margin > 0),
        "promotion_state": promotion_state,
        "posterior_state_counts": {
            key.split(":", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("posterior_state:")
        },
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "by_action": {
            action: dict(sorted(counter.items()))
            for action, counter in sorted(by_action.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "positive_control_margin_count > 0 with focused, breadth, and all25 non-regression",
            "allowed_connection": "diagnostic_only_until_route_counterfactual_margin",
            "action_generation_allowed": False,
        },
    }


def summarize_route_validity_recovery_candidates(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    candidate_rows: list[dict[str, Any]] = []
    by_game: dict[str, Counter[str]] = {}
    by_source_action: dict[str, Counter[str]] = {}
    rejected_packet_count = 0
    accepted_packet_count = 0
    blocked_action_packet_count = 0

    for packet in packet_rows:
        validation = validate_route_validity_packet(packet)
        if not bool(validation.get("accepted")):
            rejected_packet_count += 1
            continue
        accepted_packet_count += 1
        action_delta = packet.get("action_to_route_delta")
        action_delta = action_delta if isinstance(action_delta, Mapping) else {}
        if not bool(action_delta.get("blocked_action_candidate")):
            continue
        blocked_action_packet_count += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        source_action = _normalize_action_name(action_delta.get("action_name", ""))
        recovery_actions = _route_recovery_action_candidates(source_action)
        if not recovery_actions:
            continue
        candidate_key = (
            f"route_recovery:{game_id}:blocked_move_transition:"
            f"{source_action}:step{_safe_int(packet.get('step'))}"
        )
        by_game.setdefault(game_id, Counter())
        by_source_action.setdefault(source_action, Counter())
        by_game[game_id]["candidate_count"] += 1
        by_game[game_id][f"source_action:{source_action}"] += 1
        by_source_action[source_action]["candidate_count"] += 1
        candidate_rows.append(
            {
                "candidate_key": candidate_key,
                "candidate_kind": "blocked_route_alternate_action_probe",
                "game_id": game_id,
                "step": _safe_int(packet.get("step")),
                "source_action_name": source_action,
                "recovery_action_names": recovery_actions,
                "evidence": {
                    "mechanic_state_type": str(validation.get("mechanic_state_type", "")),
                    "blocked_action_candidate": True,
                    "changed_cell_count": _safe_int(action_delta.get("changed_cell_count")),
                    "route_state": str(packet.get("source_target_route_state", "") or ""),
                    "passable_region_count": _safe_int(packet.get("passable_region_count")),
                    "largest_passable_region_area": _safe_int(
                        packet.get("largest_passable_region_area")
                    ),
                },
                "runtime_execution_enabled": False,
                "requires_guarded_probe": True,
            }
        )

    return {
        "schema_version": ROUTE_VALIDITY_RECOVERY_CANDIDATE_MINER_SCHEMA_VERSION,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "packet_count": int(len(packet_rows)),
        "accepted_packet_count": int(accepted_packet_count),
        "rejected_packet_count": int(rejected_packet_count),
        "blocked_action_packet_count": int(blocked_action_packet_count),
        "recovery_candidate_count": int(len(candidate_rows)),
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "by_source_action": {
            action: dict(sorted(counter.items()))
            for action, counter in sorted(by_source_action.items())
        },
        "candidates": candidate_rows[:sample_limit],
        "next_runtime_contract": {
            "required_before_execution": "default-off guarded route recovery probe with same-context controls",
            "allowed_connection": "candidate_design_only_until_runtime_probe_enabled",
            "action_generation_allowed": False,
        },
    }


def summarize_route_validity_all25_non_regression(
    run_rows: Iterable[Mapping[str, Any]],
    *,
    expected_score_sum: int | None = None,
    expected_by_game: Mapping[str, Any] | None = None,
    require_no_game_conditioned_primitives: bool = True,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Check all25 preservation while route-validity diagnostics are enabled."""

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
        reason_counts["route_validity_all25_non_regression_supported"] += 1
        promotion_state = "route_validity_all25_non_regression_supported"
    elif rows:
        promotion_state = "route_validity_all25_non_regression_blocked"
    else:
        promotion_state = "route_validity_all25_non_regression_no_rows"

    return _sanitize(
        {
            "schema_version": ROUTE_VALIDITY_ALL25_NON_REGRESSION_SCHEMA_VERSION,
            "diagnostic_only": True,
            "default_off": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
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
            "improved_game_count": int(
                sum(
                    1
                    for game_id, expected in expected_map.items()
                    if game_id in by_game and int(by_game[game_id]) > int(expected)
                )
            ),
            "missing_expected_game_count": int(len(missing_games)),
            "missing_expected_games": missing_games[:sample_limit],
            "regressions": regressions,
            "improvements": improvements,
            "by_game": dict(sorted(by_game.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "planner_promotion_blocked": bool(not supported),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "next_runtime_contract": {
                "required_before_promotion": (
                    "all25 non-regression is a preservation gate only; route "
                    "validity planner promotion still needs a separate review"
                ),
                "allowed_connection": "all25_non_regression_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def _route_topology(
    arr: np.ndarray,
    *,
    source_site: Sequence[Any] | None,
    target_site: Sequence[Any] | None,
) -> dict[str, Any]:
    height, width = arr.shape[:2]
    total = max(1, int(height * width))
    hist = Counter(int(value) for value in arr.reshape(-1) if int(value) >= 0)
    if not hist:
        return _empty_topology()
    min_area = max(8, int(total * 0.04))
    colors = [
        int(color)
        for color, count in hist.most_common()
        if int(count) >= min_area
    ][:4]
    if not colors:
        colors = [int(hist.most_common(1)[0][0])]

    source = _xy_tuple(source_site)
    target = _xy_tuple(target_site)
    candidate_rows: list[dict[str, Any]] = []
    region_count = 0
    largest_area = 0
    largest_bbox: list[int] = []
    narrow_sum = 0
    dead_end_sum = 0
    obstacle_mask = np.ones(arr.shape[:2], dtype=bool)
    any_source_target_same_candidate = False
    any_connected = False
    any_blocked = False
    site_color_mismatch = False

    for color in colors:
        mask = arr == int(color)
        obstacle_mask &= ~mask
        regions, component_ids = _components_for_mask(mask)
        if not regions:
            continue
        region_count += len(regions)
        largest = max(regions, key=lambda row: int(row["area"]))
        if int(largest["area"]) > largest_area:
            largest_area = int(largest["area"])
            largest_bbox = list(largest["bbox"])
        degree = _degree_counts(mask)
        narrow = int(np.count_nonzero((degree > 0) & (degree <= 2)))
        dead_ends = int(np.count_nonzero(degree == 1))
        narrow_sum += narrow
        dead_end_sum += dead_ends
        connected = False
        blocked = False
        both_on_color = False
        if source is not None and target is not None:
            sx, sy = source
            tx, ty = target
            if _in_bounds(sx, sy, width, height) and _in_bounds(tx, ty, width, height):
                source_component = int(component_ids[sy, sx])
                target_component = int(component_ids[ty, tx])
                both_on_color = source_component >= 0 and target_component >= 0
                if both_on_color:
                    any_source_target_same_candidate = True
                    connected = source_component == target_component
                    blocked = source_component != target_component
                    any_connected = any_connected or connected
                    any_blocked = any_blocked or blocked
        candidate_rows.append(
            {
                "color": int(color),
                "cell_count": int(hist.get(color, 0)),
                "region_count": int(len(regions)),
                "largest_region_area": int(largest["area"]),
                "largest_region_bbox": list(largest["bbox"]),
                "narrow_corridor_cell_count": narrow,
                "dead_end_cell_count": dead_ends,
                "source_target_both_on_candidate": both_on_color,
                "source_target_connected": connected,
                "source_target_blocked": blocked,
            }
        )

    if source is not None and target is not None and not any_source_target_same_candidate:
        site_color_mismatch = True
    obstacle_components = _components_for_mask(obstacle_mask)[0]
    obstacle_count = sum(1 for region in obstacle_components if int(region["area"]) >= 2)
    route_ambiguity = bool(region_count > len(candidate_rows) or len(candidate_rows) > 1)
    if any_connected:
        route_state = "source_target_connected"
    elif any_blocked:
        route_state = "source_target_blocked"
    elif site_color_mismatch:
        route_state = "site_color_mismatch"
    elif route_ambiguity:
        route_state = "multi_region_route_topology"
    elif region_count > 0:
        route_state = "route_topology_observed"
    else:
        route_state = "insufficient_route_evidence"

    return {
        "passable_color_candidates": candidate_rows,
        "passable_candidate_count": len(candidate_rows),
        "passable_region_count": int(region_count),
        "largest_passable_region_area": int(largest_area),
        "largest_passable_region_bbox": largest_bbox,
        "narrow_corridor_cell_count": int(narrow_sum),
        "dead_end_cell_count": int(dead_end_sum),
        "obstacle_component_count": int(obstacle_count),
        "source_target_route_state": route_state,
        "source_target_connected": bool(any_connected),
        "source_target_blocked": bool(any_blocked),
        "site_color_mismatch": bool(site_color_mismatch),
        "route_ambiguity": bool(route_ambiguity),
    }


def _route_topology_delta(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any],
) -> dict[str, Any]:
    if before is None:
        return {
            "observed": False,
            "passable_region_count_delta": 0,
            "largest_passable_region_area_delta": 0,
            "narrow_corridor_cell_count_delta": 0,
            "dead_end_cell_count_delta": 0,
            "obstacle_component_count_delta": 0,
        }
    return {
        "observed": True,
        "passable_region_count_delta": _safe_int(after.get("passable_region_count"))
        - _safe_int(before.get("passable_region_count")),
        "largest_passable_region_area_delta": _safe_int(
            after.get("largest_passable_region_area")
        )
        - _safe_int(before.get("largest_passable_region_area")),
        "narrow_corridor_cell_count_delta": _safe_int(
            after.get("narrow_corridor_cell_count")
        )
        - _safe_int(before.get("narrow_corridor_cell_count")),
        "dead_end_cell_count_delta": _safe_int(after.get("dead_end_cell_count"))
        - _safe_int(before.get("dead_end_cell_count")),
        "obstacle_component_count_delta": _safe_int(after.get("obstacle_component_count"))
        - _safe_int(before.get("obstacle_component_count")),
    }


def _route_progress_candidate(
    delta: Mapping[str, Any],
    topology: Mapping[str, Any],
    *,
    score_delta: int,
) -> bool:
    if int(score_delta) > 0 or bool(topology.get("source_target_connected")):
        return True
    if not bool(delta.get("observed")):
        return False
    region_delta = _safe_int(delta.get("passable_region_count_delta"))
    area_delta = _safe_int(delta.get("largest_passable_region_area_delta"))
    narrow_delta = _safe_int(delta.get("narrow_corridor_cell_count_delta"))
    dead_end_delta = _safe_int(delta.get("dead_end_cell_count_delta"))
    obstacle_delta = _safe_int(delta.get("obstacle_component_count_delta"))
    return bool(
        (area_delta > 0 and region_delta <= 0)
        or obstacle_delta < 0
        or (narrow_delta < 0 and area_delta >= 0)
        or (dead_end_delta < 0 and area_delta >= 0)
    )


def _route_regression_candidate(
    delta: Mapping[str, Any],
    topology: Mapping[str, Any],
) -> bool:
    if bool(topology.get("source_target_blocked")):
        return True
    if not bool(delta.get("observed")):
        return False
    region_delta = _safe_int(delta.get("passable_region_count_delta"))
    area_delta = _safe_int(delta.get("largest_passable_region_area_delta"))
    narrow_delta = _safe_int(delta.get("narrow_corridor_cell_count_delta"))
    dead_end_delta = _safe_int(delta.get("dead_end_cell_count_delta"))
    obstacle_delta = _safe_int(delta.get("obstacle_component_count_delta"))
    return bool(
        obstacle_delta > 0
        or (area_delta < 0 and region_delta >= 0)
        or (dead_end_delta > 0 and narrow_delta > 0)
    )


def _route_recovery_action_candidates(source_action: str) -> list[str]:
    source = _normalize_action_name(source_action)
    if source not in _MOVE_ACTION_DELTAS:
        return []
    opposite = {
        "ACTION1": "ACTION2",
        "ACTION2": "ACTION1",
        "ACTION3": "ACTION4",
        "ACTION4": "ACTION3",
    }.get(source)
    perpendicular = {
        "ACTION1": ["ACTION3", "ACTION4"],
        "ACTION2": ["ACTION3", "ACTION4"],
        "ACTION3": ["ACTION1", "ACTION2"],
        "ACTION4": ["ACTION1", "ACTION2"],
    }.get(source, [])
    out = [action for action in perpendicular if action != source]
    if opposite and opposite not in out:
        out.append(opposite)
    return out


def _empty_topology() -> dict[str, Any]:
    return {
        "passable_color_candidates": [],
        "passable_candidate_count": 0,
        "passable_region_count": 0,
        "largest_passable_region_area": 0,
        "largest_passable_region_bbox": [],
        "narrow_corridor_cell_count": 0,
        "dead_end_cell_count": 0,
        "obstacle_component_count": 0,
        "source_target_route_state": "insufficient_route_evidence",
        "source_target_connected": False,
        "source_target_blocked": False,
        "site_color_mismatch": False,
        "route_ambiguity": False,
    }


def _components_for_mask(mask: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    height, width = mask.shape[:2]
    seen = np.zeros((height, width), dtype=bool)
    component_ids = np.full((height, width), -1, dtype=np.int32)
    components: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x] or not bool(mask[y, x]):
                continue
            component_id = len(components)
            queue: deque[tuple[int, int]] = deque([(x, y)])
            seen[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while queue:
                px, py = queue.popleft()
                xs.append(px)
                ys.append(py)
                component_ids[py, px] = component_id
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and not seen[ny, nx]
                        and bool(mask[ny, nx])
                    ):
                        seen[ny, nx] = True
                        queue.append((nx, ny))
            area = len(xs)
            components.append(
                {
                    "area": int(area),
                    "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    "centroid": [
                        int(round(sum(xs) / area)),
                        int(round(sum(ys) / area)),
                    ],
                }
            )
    return components, component_ids


def _degree_counts(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    degree = np.zeros((height, width), dtype=np.int16)
    for y in range(height):
        for x in range(width):
            if not bool(mask[y, x]):
                continue
            count = 0
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < width and 0 <= ny < height and bool(mask[ny, nx]):
                    count += 1
            degree[y, x] = count
    return degree


def _mechanic_state_type(packet: Mapping[str, Any]) -> str:
    action_delta = packet.get("action_to_route_delta")
    action_delta = action_delta if isinstance(action_delta, Mapping) else {}
    if bool(packet.get("source_target_connected")):
        return "connected_route_candidate"
    if bool(packet.get("source_target_blocked")):
        return "blocked_route_candidate"
    if bool(action_delta.get("blocked_action_candidate")):
        return "blocked_move_transition"
    if bool(packet.get("route_ambiguity")):
        return "ambiguous_route_topology"
    if _safe_int(packet.get("passable_region_count")) > 0:
        return "route_topology"
    return "insufficient_route_evidence"


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


def _normalize_action_name(action: Any) -> str:
    text = str(getattr(action, "name", action) or "").strip().upper()
    if text.startswith("GAMEACTION."):
        text = text.split(".", 1)[1]
    if text.isdigit():
        return f"ACTION{int(text)}"
    return text


def _xy_tuple(value: Sequence[Any] | None) -> tuple[int, int] | None:
    if not _is_xy(value):
        return None
    assert value is not None
    return int(value[0]), int(value[1])


def _is_xy(value: Sequence[Any] | None) -> bool:
    if not isinstance(value, Sequence) or len(value) < 2:
        return False
    try:
        int(value[0])
        int(value[1])
    except (TypeError, ValueError):
        return False
    return True


def _in_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= int(x) < int(width) and 0 <= int(y) < int(height)


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
    return value


__all__ = [
    "ROUTE_VALIDITY_PACKET_SCHEMA_VERSION",
    "ROUTE_VALIDITY_ALL25_NON_REGRESSION_SCHEMA_VERSION",
    "ROUTE_VALIDITY_PROGRESS_POSTERIOR_SCHEMA_VERSION",
    "ROUTE_VALIDITY_RECOVERY_CANDIDATE_MINER_SCHEMA_VERSION",
    "ROUTE_VALIDITY_TRACE_SENSOR_SCHEMA_VERSION",
    "build_route_validity_packet",
    "summarize_route_validity_all25_non_regression",
    "summarize_route_validity_progress_posterior",
    "summarize_route_validity_recovery_candidates",
    "summarize_route_validity_packets",
    "validate_route_validity_packet",
]
