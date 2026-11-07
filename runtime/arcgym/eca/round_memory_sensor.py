"""Frame-only round-memory diagnostics and guarded proposals.

The sensor observes public before/after frames for multi-round puzzles where a
submit action records prior motion and later actions replay or preserve that
recorded state. The packet builders remain observational; guarded proposals use
only frame-derived roles and stay behind explicit opt-in flags.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROUND_MEMORY_PACKET_SCHEMA_VERSION = "eca_round_memory_packet.v1"
ROUND_MEMORY_DIAGNOSTIC_SENSOR_SCHEMA_VERSION = (
    "eca_round_memory_diagnostic_sensor.v1"
)
ROUND_MEMORY_POSTERIOR_SCHEMA_VERSION = "eca_round_memory_posterior.v1"
ROUND_MEMORY_PROTECTED_PROPOSAL_SCHEMA_VERSION = (
    "eca_round_memory_protected_proposal.v1"
)


def build_round_memory_packet(
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
    """Build an observational round-memory packet from public frames."""

    before = _as_frame(before_frame)
    after = _as_frame(after_frame) if after_frame is not None else before
    action_payload = dict(action_payload or {})
    action = _normalize_action_name(action_name)
    before_state = _round_memory_topology(before)
    after_state = _round_memory_topology(after)
    marker_delta = _round_marker_delta(before_state, after_state)
    player_motion = _player_motion(before_state, after_state)
    color8_delta = _color8_delta(before_state, after_state)
    goal_relation = _goal_relation(before_state, after_state)
    changed = _change_observation(before, after)

    action_is_submit = action == "ACTION5"
    action_is_motion = action in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
    player_reset_candidate = bool(
        action_is_submit
        and bool(player_motion.get("moved_player"))
        and (
            abs(float(player_motion.get("delta", [0.0, 0.0])[0])) >= 6.0
            or abs(float(player_motion.get("delta", [0.0, 0.0])[1])) >= 6.0
        )
    )
    round_progress_visible = bool(after_state.get("round_progress_visible"))
    round_progress_candidate = bool(
        int(marker_delta["color1_count_delta"]) != 0
        or int(marker_delta["color2_count_delta"]) != 0
        or bool(marker_delta["progress_visibility_changed"])
    )
    round_submit_candidate = bool(
        action_is_submit
        and (
            round_progress_candidate
            or player_reset_candidate
            or bool(color8_delta.get("door_switch_state_delta_candidate"))
        )
    )
    memory_replay_candidate = bool(
        action_is_motion
        and round_progress_visible
        and (
            bool(player_motion.get("moved_player"))
            or round_progress_candidate
            or bool(color8_delta.get("door_switch_state_delta_candidate"))
        )
    )
    switch_memory_candidate = bool(
        action_is_motion
        and round_progress_visible
        and bool(color8_delta.get("door_switch_state_delta_candidate"))
    )
    goal_route_progress_candidate = bool(
        goal_relation.get("distance_delta") is not None
        and float(goal_relation.get("distance_delta") or 0.0) < -0.5
    )
    score_closure_candidate = bool(
        int(score_delta) > 0
        and (
            round_submit_candidate
            or memory_replay_candidate
            or switch_memory_candidate
            or goal_route_progress_candidate
            or bool(player_motion.get("moved_player"))
        )
    )

    candidate_types: set[str] = set()
    if round_submit_candidate:
        candidate_types.add("round_submit")
    if round_progress_candidate:
        candidate_types.add("round_progress")
    if player_reset_candidate:
        candidate_types.add("player_reset_after_submit")
    if memory_replay_candidate:
        candidate_types.add("memory_replay")
    if switch_memory_candidate:
        candidate_types.add("switch_memory")
    if bool(color8_delta.get("door_switch_state_delta_candidate")):
        candidate_types.add("door_switch_state_delta")
    if goal_route_progress_candidate:
        candidate_types.add("goal_route_progress")
    if score_closure_candidate:
        candidate_types.add("score_closure")

    return _sanitize(
        {
            "schema_version": ROUND_MEMORY_PACKET_SCHEMA_VERSION,
            "sensor_family": "round_memory_sensor",
            "game_id": str(game_id or "").strip().lower(),
            "state_hash": str(state_hash or _frame_hash(before)),
            "step": int(step),
            "action_name": action,
            "action_payload": action_payload,
            "score_delta": int(score_delta),
            "before_counts": _compact_counts(before_state),
            "after_counts": _compact_counts(after_state),
            "round_marker_delta": marker_delta,
            "player_motion": player_motion,
            "color8_delta": color8_delta,
            "goal_relation": goal_relation,
            "changed_cell_count": int(changed["changed_cell_count"]),
            "round_marker_changed_cell_count": int(
                changed["round_marker_changed_cell_count"]
            ),
            "player_changed_cell_count": int(changed["player_changed_cell_count"]),
            "door_switch_changed_cell_count": int(
                changed["door_switch_changed_cell_count"]
            ),
            "round_submit_candidate": round_submit_candidate,
            "round_progress_candidate": round_progress_candidate,
            "player_reset_candidate": player_reset_candidate,
            "memory_replay_candidate": memory_replay_candidate,
            "switch_memory_candidate": switch_memory_candidate,
            "door_switch_state_delta_candidate": bool(
                color8_delta.get("door_switch_state_delta_candidate")
            ),
            "goal_route_progress_candidate": goal_route_progress_candidate,
            "score_closure_candidate": score_closure_candidate,
            "candidate_types": sorted(candidate_types),
            "diagnostic_only": True,
        }
    )


def validate_round_memory_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if str(packet.get("schema_version", "") or "") != ROUND_MEMORY_PACKET_SCHEMA_VERSION:
        reasons.append("schema_version_mismatch")
    if str(packet.get("sensor_family", "") or "") != "round_memory_sensor":
        reasons.append("sensor_family_mismatch")
    action = str(packet.get("action_name", "") or "")
    if action and action not in {
        "ACTION1",
        "ACTION2",
        "ACTION3",
        "ACTION4",
        "ACTION5",
        "ACTION6",
    }:
        reasons.append("unknown_action_name")
    return {
        "schema_version": "eca_round_memory_validation.v1",
        "accepted": not reasons,
        "rejected_reasons": reasons,
        "family": "round_memory",
        "mechanic_state_type": _mechanic_state_type(packet),
        "goal_link_type": "round_submit_memory_to_delayed_route_progress",
        "candidate_types": list(packet.get("candidate_types") or []),
    }


def summarize_round_memory_packets(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 12,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    candidate_type_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    color1_count_delta_sum = 0
    color2_count_delta_sum = 0
    color8_area_delta_sum = 0
    changed_cell_count_sum = 0
    score_delta_sum = 0

    for packet in packet_rows:
        validation = validate_round_memory_packet(packet)
        accepted = bool(validation.get("accepted"))
        counts["accepted_packet_count" if accepted else "rejected_packet_count"] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["packet_count"] += 1
        action = str(packet.get("action_name", "") or "")
        if action == "ACTION5":
            counts["action5_packet_count"] += 1
            by_game[game_id]["action5_packet_count"] += 1
        for field, name in (
            ("round_submit_candidate", "round_submit_candidate_count"),
            ("round_progress_candidate", "round_progress_candidate_count"),
            ("player_reset_candidate", "player_reset_candidate_count"),
            ("memory_replay_candidate", "memory_replay_candidate_count"),
            ("switch_memory_candidate", "switch_memory_candidate_count"),
            (
                "door_switch_state_delta_candidate",
                "door_switch_state_delta_candidate_count",
            ),
            ("goal_route_progress_candidate", "goal_route_progress_candidate_count"),
            ("score_closure_candidate", "score_closure_candidate_count"),
        ):
            if bool(packet.get(field)):
                counts[name] += 1
                by_game[game_id][name] += 1
        player_motion = packet.get("player_motion")
        player_motion = player_motion if isinstance(player_motion, Mapping) else {}
        if bool(player_motion.get("moved_player")):
            counts["player_motion_count"] += 1
            by_game[game_id]["player_motion_count"] += 1
        color8_delta = packet.get("color8_delta")
        color8_delta = color8_delta if isinstance(color8_delta, Mapping) else {}
        if bool(color8_delta.get("door_switch_state_delta_candidate")):
            counts["color8_changed_packet_count"] += 1
            by_game[game_id]["color8_changed_packet_count"] += 1
        for candidate_type in list(packet.get("candidate_types") or []):
            candidate_type_counts[str(candidate_type)] += 1

        marker_delta = packet.get("round_marker_delta")
        marker_delta = marker_delta if isinstance(marker_delta, Mapping) else {}
        color1_delta = _safe_int(marker_delta.get("color1_count_delta"))
        color2_delta = _safe_int(marker_delta.get("color2_count_delta"))
        color8_area_delta = _safe_int(color8_delta.get("color8_area_delta"))
        changed = _safe_int(packet.get("changed_cell_count"))
        score_delta = _safe_int(packet.get("score_delta"))
        color1_count_delta_sum += color1_delta
        color2_count_delta_sum += color2_delta
        color8_area_delta_sum += color8_area_delta
        changed_cell_count_sum += changed
        score_delta_sum += score_delta
        if len(samples) < sample_limit:
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "action_name": action,
                    "candidate_types": list(packet.get("candidate_types") or []),
                    "color1_count_delta": color1_delta,
                    "color2_count_delta": color2_delta,
                    "color8_area_delta": color8_area_delta,
                    "player_delta": player_motion.get("delta"),
                    "goal_distance_delta": (
                        packet.get("goal_relation")
                        if isinstance(packet.get("goal_relation"), Mapping)
                        else {}
                    ).get("distance_delta"),
                    "score_delta": score_delta,
                }
            )

    return _sanitize(
        {
            "schema_version": ROUND_MEMORY_DIAGNOSTIC_SENSOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "accepted_packet_count": int(counts.get("accepted_packet_count", 0)),
            "rejected_packet_count": int(counts.get("rejected_packet_count", 0)),
            "action5_packet_count": int(counts.get("action5_packet_count", 0)),
            "round_submit_candidate_count": int(
                counts.get("round_submit_candidate_count", 0)
            ),
            "round_progress_candidate_count": int(
                counts.get("round_progress_candidate_count", 0)
            ),
            "player_reset_candidate_count": int(
                counts.get("player_reset_candidate_count", 0)
            ),
            "memory_replay_candidate_count": int(
                counts.get("memory_replay_candidate_count", 0)
            ),
            "switch_memory_candidate_count": int(
                counts.get("switch_memory_candidate_count", 0)
            ),
            "door_switch_state_delta_candidate_count": int(
                counts.get("door_switch_state_delta_candidate_count", 0)
            ),
            "goal_route_progress_candidate_count": int(
                counts.get("goal_route_progress_candidate_count", 0)
            ),
            "score_closure_candidate_count": int(
                counts.get("score_closure_candidate_count", 0)
            ),
            "player_motion_count": int(counts.get("player_motion_count", 0)),
            "color8_changed_packet_count": int(
                counts.get("color8_changed_packet_count", 0)
            ),
            "color1_count_delta_sum": int(color1_count_delta_sum),
            "color2_count_delta_sum": int(color2_count_delta_sum),
            "color8_area_delta_sum": int(color8_area_delta_sum),
            "changed_cell_count_sum": int(changed_cell_count_sum),
            "score_delta_sum": int(score_delta_sum),
            "candidate_type_counts": dict(sorted(candidate_type_counts.items())),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "samples": samples,
            "next_planner_contract": {
                "required_before_ranking": (
                    "round-memory posterior with submit/replay controls"
                ),
                "allowed_connection": (
                    "diagnostic_only_until_round_memory_posterior_controls"
                ),
                "action_generation_allowed": False,
            },
        }
    )


def summarize_round_memory_posterior(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    by_key: dict[str, dict[str, Any]] = {}
    state_counts: Counter[str] = Counter()
    candidate_type_counts: Counter[str] = Counter()
    accepted_packet_count = 0
    rejected_packet_count = 0
    samples: list[dict[str, Any]] = []

    for packet in packet_rows:
        validation = validate_round_memory_packet(packet)
        accepted = bool(validation.get("accepted"))
        accepted_packet_count += int(accepted)
        rejected_packet_count += int(not accepted)
        candidate_types = sorted(
            str(row) for row in list(packet.get("candidate_types") or []) if row
        )
        for candidate_type in candidate_types:
            candidate_type_counts[candidate_type] += 1
        key = _round_memory_posterior_key(packet, validation=validation)
        entry = by_key.setdefault(
            key,
            {
                "round_memory_key": key,
                "game_id": str(packet.get("game_id", "") or ""),
                "mechanic_state_type": str(validation.get("mechanic_state_type", "")),
                "goal_link_type": str(validation.get("goal_link_type", "")),
                "observation_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "round_submit_count": 0,
                "round_progress_count": 0,
                "memory_replay_count": 0,
                "switch_memory_count": 0,
                "door_switch_delta_count": 0,
                "goal_route_progress_count": 0,
                "score_closure_count": 0,
                "score_delta_sum": 0,
                "candidate_types": [],
            },
        )
        entry["observation_count"] += 1
        entry["accepted_count"] += int(accepted)
        entry["rejected_count"] += int(not accepted)
        entry["round_submit_count"] += int(bool(packet.get("round_submit_candidate")))
        entry["round_progress_count"] += int(bool(packet.get("round_progress_candidate")))
        entry["memory_replay_count"] += int(bool(packet.get("memory_replay_candidate")))
        entry["switch_memory_count"] += int(bool(packet.get("switch_memory_candidate")))
        entry["door_switch_delta_count"] += int(
            bool(packet.get("door_switch_state_delta_candidate"))
        )
        entry["goal_route_progress_count"] += int(
            bool(packet.get("goal_route_progress_candidate"))
        )
        entry["score_closure_count"] += int(bool(packet.get("score_closure_candidate")))
        entry["score_delta_sum"] += _safe_int(packet.get("score_delta"))
        entry["candidate_types"] = sorted(
            set(list(entry.get("candidate_types") or []) + candidate_types)
        )
        if len(samples) < sample_limit:
            samples.append(
                {
                    "round_memory_key": key,
                    "game_id": str(packet.get("game_id", "") or ""),
                    "step": _safe_int(packet.get("step")),
                    "action_name": str(packet.get("action_name", "") or ""),
                    "candidate_types": candidate_types,
                    "score_delta": _safe_int(packet.get("score_delta")),
                }
            )

    confirmed_score_closure_entry_count = 0
    confirmed_switch_memory_entry_count = 0
    unconfirmed_submit_replay_entry_count = 0
    insufficient_evidence_entry_count = 0
    rejected_entry_count = 0
    for entry in by_key.values():
        if int(entry.get("score_closure_count", 0)) > 0:
            posterior_state = "confirmed_score_closure"
            confirmed_score_closure_entry_count += 1
        elif int(entry.get("switch_memory_count", 0)) > 0 and int(
            entry.get("memory_replay_count", 0)
        ) > 0:
            posterior_state = "confirmed_switch_memory"
            confirmed_switch_memory_entry_count += 1
        elif int(entry.get("round_submit_count", 0)) > 0 and int(
            entry.get("memory_replay_count", 0)
        ) > 0:
            posterior_state = "unconfirmed_submit_replay"
            unconfirmed_submit_replay_entry_count += 1
        elif int(entry.get("rejected_count", 0)) >= int(entry.get("observation_count", 0)):
            posterior_state = "rejected_descriptor"
            rejected_entry_count += 1
        else:
            posterior_state = "insufficient_evidence"
            insufficient_evidence_entry_count += 1
        entry["posterior_state"] = posterior_state
        state_counts[posterior_state] += 1

    return _sanitize(
        {
            "schema_version": ROUND_MEMORY_POSTERIOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "accepted_packet_count": int(accepted_packet_count),
            "rejected_packet_count": int(rejected_packet_count),
            "posterior_entry_count": int(len(by_key)),
            "confirmed_score_closure_entry_count": int(
                confirmed_score_closure_entry_count
            ),
            "confirmed_switch_memory_entry_count": int(
                confirmed_switch_memory_entry_count
            ),
            "unconfirmed_submit_replay_entry_count": int(
                unconfirmed_submit_replay_entry_count
            ),
            "insufficient_evidence_entry_count": int(
                insufficient_evidence_entry_count
            ),
            "rejected_entry_count": int(rejected_entry_count),
            "posterior_state_counts": dict(sorted(state_counts.items())),
            "candidate_type_counts": dict(sorted(candidate_type_counts.items())),
            "by_round_memory_key": {
                key: value for key, value in sorted(by_key.items())
            },
            "samples": samples,
            "next_runtime_contract": {
                "env_flag": "ECA_ENABLE_ROUND_MEMORY_GUARDED_PROBE",
                "required_before_execution": (
                    "protected frame-derived proposal; max one active plan; "
                    "no positive ranking until score closure is observed"
                ),
                "allowed_connection": "posterior_to_guarded_probe_only",
                "action_generation_allowed": False,
                "positive_ranking_allowed": False,
            },
        }
    )


def build_round_memory_protected_proposal(
    frame: Any,
    *,
    game_id: str = "",
    state_hash: str = "",
    max_plan_steps: int = 32,
    role_discovery: bool = False,
    post_score_replan: bool = False,
) -> dict[str, Any]:
    """Compile a guarded round-memory proposal from public frame structure."""

    arr = _as_frame(frame)
    state = _round_memory_topology(arr)
    if bool(post_score_replan):
        post_score = _build_round_memory_post_score_replan_proposal(
            arr,
            state=state,
            game_id=game_id,
            state_hash=state_hash,
            max_plan_steps=max_plan_steps,
            role_discovery=role_discovery,
        )
        if post_score is not None:
            return post_score
    reasons: list[str] = []
    role_discovery = bool(role_discovery)
    role_summary: dict[str, Any] = {"enabled": role_discovery}
    if role_discovery:
        roles = _discover_round_memory_proposal_roles(arr)
        reasons.extend(str(row) for row in list(roles.get("rejected_reasons") or []))
        player = roles.get("player_component")
        goal = roles.get("goal_component")
        color8 = roles.get("door_switch_component")
        start = tuple(roles.get("player_anchor") or ()) or None
        goal_site = tuple(roles.get("goal_anchor") or ()) or None
        switch = tuple(roles.get("switch_anchor") or ()) or None
        stride = _safe_int(roles.get("stride"), 6)
        color1_count = _safe_int(roles.get("round_marker_count"))
        role_summary.update(
            {
                "background_color": roles.get("background_color"),
                "actor_goal_color": roles.get("actor_goal_color"),
                "door_switch_color": roles.get("door_switch_color"),
                "round_marker_color": roles.get("round_marker_color"),
                "round_progress_visible": bool(roles.get("round_progress_visible")),
                "candidate_score": roles.get("candidate_score"),
            }
        )
    else:
        player = _largest_component(state.get("player_components") or [])
        goal = _largest_component(state.get("goal_components") or [])
        color8 = _largest_component(state.get("color8_components") or [])
        if not player:
            reasons.append("missing_player_component")
        if not goal:
            reasons.append("missing_goal_component")
        if not color8:
            reasons.append("missing_door_switch_component")
        if bool(state.get("round_progress_visible")):
            reasons.append("round_memory_already_submitted")
        color1_count = _safe_int(state.get("color1_count"))
        if color1_count > 12:
            reasons.append("unsupported_round_counter_count")

        stride = 6
        start = _player_anchor(player) if player else None
        goal_site = _goal_anchor(goal) if goal else None
        switch = (
            _switch_anchor(color8, start_y=start[1])
            if color8 is not None and start is not None
            else None
        )
    if start and switch and abs(start[0] - switch[0]) % stride != 0:
        reasons.append("switch_not_stride_aligned")
    if start and goal_site and abs(start[1] - goal_site[1]) % stride != 0:
        reasons.append("door_entry_not_stride_aligned")
    if start and goal_site and abs(start[0] - goal_site[0]) % stride != 0:
        reasons.append("goal_not_stride_aligned")

    plan: list[dict[str, Any]] = []
    if not reasons and start and switch and goal_site:
        door_entry = (int(start[0]), int(goal_site[1]))
        round1 = _axis_path(start, switch, stride)
        round2 = _axis_path(start, door_entry, stride) + _axis_path(
            door_entry,
            goal_site,
            stride,
        )
        if not round1 or not round2:
            reasons.append("no_stride_consistent_round_memory_plan")
        else:
            plan = [
                *_plan_steps(round1, role="record_switch_path"),
                {
                    "action_name": "ACTION5",
                    "payload": {},
                    "role": "submit_round_memory",
                },
                *_plan_steps(round2, role="replay_opened_route_to_goal"),
            ]
    if len(plan) > int(max_plan_steps):
        reasons.append("proposal_plan_too_long")
        plan = []
    if reasons:
        plan = []
    accepted = not reasons
    return _sanitize(
        {
            "schema_version": ROUND_MEMORY_PROTECTED_PROPOSAL_SCHEMA_VERSION,
            "accepted": bool(accepted),
            "rejected_reasons": sorted(set(str(reason) for reason in reasons)),
            "descriptor_source": "round_memory_sensor",
            "sensor_family": "round_memory_sensor",
            "family": "round_memory",
            "game_id": str(game_id or "").strip().lower(),
            "state_hash": str(state_hash or _frame_hash(arr)),
            "mechanic_state_type": "round_memory_replay",
            "goal_link_type": "round_memory_switch_hold_route",
            "protected_probe": True,
            "direct_llm_action_allowed": False,
            "candidate_generation_owner": "eca_verifier",
            "plan": plan,
            "plan_length": int(len(plan)),
            "expected_score_delta": 1 if accepted else 0,
            "round_memory": {
                "color1_count": int(color1_count),
                "color2_count": int(state.get("color2_count", 0)),
                "round_progress_visible": bool(state.get("round_progress_visible")),
                "post_score_replan": False,
                "player_anchor": list(start) if start else [],
                "switch_anchor": list(switch) if switch else [],
                "goal_anchor": list(goal_site) if goal_site else [],
                "stride": int(stride),
                "role_discovery": role_summary,
                "plan_action_counts": dict(
                    Counter(str(step.get("action_name", "")) for step in plan)
                ),
            },
        }
    )


def _build_round_memory_post_score_replan_proposal(
    arr: np.ndarray,
    *,
    state: Mapping[str, Any],
    game_id: str,
    state_hash: str,
    max_plan_steps: int,
    role_discovery: bool,
) -> dict[str, Any] | None:
    roles = _discover_round_memory_multi_switch_post_score_replan_roles(arr, state=state)
    if "round_memory_multi_switch_post_score_replan_no_candidate" in set(
        str(row) for row in list(roles.get("rejected_reasons") or [])
    ):
        roles = _discover_round_memory_portal_switch_post_score_replan_roles(
            arr,
            state=state,
        )
    if "round_memory_portal_switch_post_score_replan_no_candidate" in set(
        str(row) for row in list(roles.get("rejected_reasons") or [])
    ):
        roles = _discover_round_memory_portal_hold_post_score_replan_roles(
            arr,
            state=state,
        )
    portal_hold_reasons = set(str(row) for row in list(roles.get("rejected_reasons") or []))
    if (
        "round_memory_portal_hold_post_score_replan_no_candidate" in portal_hold_reasons
        or "missing_portal_hold_component" in portal_hold_reasons
    ):
        roles = _discover_round_memory_top_pair_bottom_door_post_score_replan_roles(
            arr,
            state=state,
        )
    top_pair_reasons = set(str(row) for row in list(roles.get("rejected_reasons") or []))
    if (
        "round_memory_top_pair_bottom_door_post_score_replan_no_candidate" in top_pair_reasons
        or "missing_top_pair_vertical_door_components" in top_pair_reasons
    ):
        roles = _discover_round_memory_dual_portal_final_post_score_replan_roles(
            arr,
            state=state,
        )
    if "round_memory_dual_portal_final_post_score_replan_no_candidate" in set(
        str(row) for row in list(roles.get("rejected_reasons") or [])
    ):
        roles = _discover_round_memory_post_score_replan_roles(arr, state=state)
    reasons = [str(row) for row in list(roles.get("rejected_reasons") or []) if row]
    plan: list[dict[str, Any]] = []
    if not reasons:
        plan.extend(_plan_steps(list(roles.get("round1_actions") or []), role="record_primary_switch_path"))
        plan.append({"action_name": "ACTION5", "payload": {}, "role": "submit_primary_switch_memory"})
        plan.extend(_plan_steps(list(roles.get("round2_actions") or []), role="record_secondary_switch_path"))
        plan.append({"action_name": "ACTION5", "payload": {}, "role": "submit_secondary_switch_memory"})
        plan.extend(_plan_steps(list(roles.get("round3_actions") or []), role="replay_dual_switch_route_to_goal"))
        if not plan:
            reasons.append("post_score_replan_empty_plan")
    if len(plan) > int(max_plan_steps):
        reasons.append("proposal_plan_too_long")
        plan = []
    if reasons:
        plan = []
    accepted = not reasons
    if not accepted and "round_memory_post_score_replan_no_candidate" in set(reasons):
        return None
    return _sanitize(
        {
            "schema_version": ROUND_MEMORY_PROTECTED_PROPOSAL_SCHEMA_VERSION,
            "accepted": bool(accepted),
            "rejected_reasons": sorted(set(str(reason) for reason in reasons)),
            "descriptor_source": "round_memory_sensor",
            "sensor_family": "round_memory_sensor",
            "family": "round_memory",
            "game_id": str(game_id or "").strip().lower(),
            "state_hash": str(state_hash or _frame_hash(arr)),
            "mechanic_state_type": "round_memory_replay",
            "goal_link_type": str(
                roles.get("goal_link_type", "round_memory_dual_switch_post_score_route")
                or "round_memory_dual_switch_post_score_route"
            ),
            "protected_probe": True,
            "direct_llm_action_allowed": False,
            "candidate_generation_owner": "eca_verifier",
            "plan": plan,
            "plan_length": int(len(plan)),
            "expected_score_delta": 1 if accepted else 0,
            "round_memory": {
                "color1_count": int(state.get("color1_count", 0)),
                "color2_count": int(state.get("color2_count", 0)),
                "round_progress_visible": bool(state.get("round_progress_visible")),
                "post_score_replan": True,
                "player_anchor": list(roles.get("player_anchor") or []),
                "goal_anchor": list(roles.get("goal_anchor") or []),
                "switch_anchor": list(roles.get("primary_switch_anchor") or []),
                "primary_switch_anchor": list(roles.get("primary_switch_anchor") or []),
                "secondary_switch_anchor": list(roles.get("secondary_switch_anchor") or []),
                "tertiary_switch_anchor": list(roles.get("tertiary_switch_anchor") or []),
                "mid_switch_anchor": list(roles.get("mid_switch_anchor") or []),
                "portal_upper_pad_anchor": list(roles.get("portal_upper_pad_anchor") or []),
                "portal_lower_pad_anchor": list(roles.get("portal_lower_pad_anchor") or []),
                "stride": int(roles.get("stride", 0) or 0),
                "role_discovery": {
                    "enabled": bool(role_discovery),
                    "post_score_replan": True,
                    "candidate_score": roles.get("candidate_score"),
                    "actor_goal_color": roles.get("actor_goal_color"),
                    "door_switch_color": roles.get("door_switch_color"),
                    "background_color": roles.get("background_color"),
                },
                "plan_action_counts": dict(
                    Counter(str(step.get("action_name", "")) for step in plan)
                ),
            },
        }
    )


def _discover_round_memory_multi_switch_post_score_replan_roles(
    arr: np.ndarray,
    *,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    components = [dict(row) for row in list(state.get("components") or [])]
    background_colors = _background_like_colors(arr)
    background_color = _dominant_color(arr)
    small_by_color: dict[int, list[dict[str, Any]]] = {}
    for component in components:
        color = int(component.get("color", -1))
        if color in background_colors:
            continue
        if _round_memory_actor_goal_shape(component):
            bbox = component.get("bbox")
            if _is_bbox(bbox) and int(bbox[1]) >= 7:
                small_by_color.setdefault(color, []).append(dict(component))

    best_pair: tuple[dict[str, Any], dict[str, Any], int] | None = None
    best_pair_score = float("-inf")
    for actor_color, rows in sorted(small_by_color.items()):
        if len(rows) < 2:
            continue
        for player in rows:
            start = _player_anchor(player)
            for goal in rows:
                if goal is player:
                    continue
                goal_site = _goal_anchor(goal)
                dx = int(goal_site[0]) - int(start[0])
                dy = abs(int(goal_site[1]) - int(start[1]))
                if dx <= 0 or dx > 24 or dy > 4:
                    continue
                score = float(dx) - float(dy) * 2.0
                if score <= best_pair_score:
                    continue
                best_pair_score = score
                best_pair = (dict(player), dict(goal), int(actor_color))
    if best_pair is None:
        return {"rejected_reasons": ["round_memory_multi_switch_post_score_replan_no_candidate"]}

    player, goal, actor_color = best_pair
    start = _player_anchor(player)
    goal_site = _goal_anchor(goal)

    vertical_switch_components: list[dict[str, Any]] = []
    for component in components:
        color = int(component.get("color", -1))
        if color in background_colors or color == int(actor_color):
            continue
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        area = _safe_int(component.get("area"))
        if area < 32 or area > 128:
            continue
        if width < 3 or width > 8 or height < 15 or height > 32:
            continue
        if y1 <= start[1] or x0 <= start[0]:
            continue
        vertical_switch_components.append(dict(component))
    if len(vertical_switch_components) < 2:
        return {"rejected_reasons": ["missing_multi_switch_vertical_components"]}

    vertical_switch_components.sort(
        key=lambda row: (int((row.get("bbox") or [0, 0, 0, 0])[0]), int((row.get("bbox") or [0, 0, 0, 0])[1]))
    )
    left = vertical_switch_components[0]
    right = vertical_switch_components[-1]
    lbox = [int(v) for v in list(left.get("bbox") or [])[:4]]
    rbox = [int(v) for v in list(right.get("bbox") or [])[:4]]
    if len(lbox) < 4 or len(rbox) < 4:
        return {"rejected_reasons": ["multi_switch_bbox_not_inferred"]}

    secondary_switch = (int(rbox[0]) - 1, int(rbox[1]) - 2)
    primary_switch = (int(lbox[0]) - 1, int(lbox[3]) - 4)
    stride_candidates = [
        abs(int(goal_site[0]) - int(start[0])),
        abs(int(secondary_switch[0]) - int(start[0])),
        abs(int(secondary_switch[1]) - int(start[1])),
        abs(int(primary_switch[1]) - int(start[1])),
    ]
    nonzero = [value for value in stride_candidates if value > 0]
    if not nonzero:
        return {"rejected_reasons": ["multi_switch_stride_not_inferred"]}
    stride = int(nonzero[0])
    for value in nonzero[1:]:
        stride = math.gcd(stride, int(value))
    if stride < 3 or stride > 8:
        return {"rejected_reasons": ["multi_switch_stride_not_inferred"]}

    top_y = int(start[1]) - 2 * int(stride)
    right_x = int(secondary_switch[0]) + 2 * int(stride)
    turn_x = int(secondary_switch[0]) - int(stride)
    detour_y = int(primary_switch[1]) - 3 * int(stride)
    if min(top_y, right_x, turn_x, detour_y) < 0:
        return {"rejected_reasons": ["multi_switch_route_out_of_bounds"]}

    round1 = (
        _axis_path(start, (start[0], top_y), stride)
        + _axis_path((start[0], top_y), (turn_x, top_y), stride)
        + _axis_path((turn_x, top_y), (turn_x, secondary_switch[1]), stride)
        + _axis_path((turn_x, secondary_switch[1]), secondary_switch, stride)
    )
    round2 = (
        _axis_path(start, (start[0], top_y), stride)
        + _axis_path((start[0], top_y), (right_x, top_y), stride)
        + _axis_path((right_x, top_y), (right_x, primary_switch[1]), stride)
        + _axis_path((right_x, primary_switch[1]), primary_switch, stride)
    )
    round3 = (
        list(round2)
        + _axis_path(primary_switch, (start[0], primary_switch[1]), stride)
        + _axis_path((start[0], primary_switch[1]), (start[0], detour_y), stride)
        + _axis_path((start[0], detour_y), (goal_site[0], detour_y), stride)
        + _axis_path((goal_site[0], detour_y), goal_site, stride)
    )
    if not round1 or not round2 or not round3:
        return {"rejected_reasons": ["multi_switch_axis_path_not_compiled"]}

    return {
        "rejected_reasons": [],
        "goal_link_type": "round_memory_multi_switch_post_score_route",
        "player_anchor": list(start),
        "goal_anchor": list(goal_site),
        "primary_switch_anchor": list(primary_switch),
        "secondary_switch_anchor": list(secondary_switch),
        "tertiary_switch_anchor": [],
        "stride": int(stride),
        "round1_actions": round1,
        "round2_actions": round2,
        "round3_actions": round3,
        "actor_goal_color": int(actor_color),
        "door_switch_color": int(left.get("color", -1)),
        "background_color": int(background_color),
        "candidate_score": round(float(best_pair_score), 3),
    }


def _discover_round_memory_portal_hold_post_score_replan_roles(
    arr: np.ndarray,
    *,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    components = [dict(row) for row in list(state.get("components") or [])]
    background_colors = _background_like_colors(arr)
    background_color = _dominant_color(arr)
    small_by_color: dict[int, list[dict[str, Any]]] = {}
    for component in components:
        color = int(component.get("color", -1))
        if color in background_colors:
            continue
        if _round_memory_actor_goal_shape(component):
            small_by_color.setdefault(color, []).append(dict(component))

    best_pair: tuple[dict[str, Any], dict[str, Any], int] | None = None
    best_pair_score = float("-inf")
    for actor_color, rows in sorted(small_by_color.items()):
        if len(rows) < 2:
            continue
        for player in rows:
            start = _player_anchor(player)
            for goal in rows:
                if goal is player:
                    continue
                goal_site = _goal_anchor(goal)
                dy = int(goal_site[1]) - int(start[1])
                dx = abs(int(goal_site[0]) - int(start[0]))
                if dy < 24 or dx > 12 or int(goal_site[0]) < int(start[0]):
                    continue
                score = float(dy - dx)
                if score <= best_pair_score:
                    continue
                best_pair_score = score
                best_pair = (dict(player), dict(goal), int(actor_color))
    if best_pair is None:
        return {
            "rejected_reasons": [
                "round_memory_portal_hold_post_score_replan_no_candidate"
            ]
        }

    player, goal, actor_color = best_pair
    start = _player_anchor(player)
    goal_site = _goal_anchor(goal)

    horizontal_bars: list[dict[str, Any]] = []
    vertical_doors: list[dict[str, Any]] = []
    portal_component: dict[str, Any] | None = None
    for component in components:
        color = int(component.get("color", -1))
        if color in background_colors or color == int(actor_color):
            continue
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        area = _safe_int(component.get("area"))
        if area < 20:
            continue
        if height <= 8 and width >= 12 and y0 < int(goal_site[1]):
            horizontal_bars.append(dict(component))
            continue
        if (
            width <= 8
            and height >= 12
            and x0 > int(start[0])
            and y0 > int(start[1])
            and y1 >= int(goal_site[1])
        ):
            vertical_doors.append(dict(component))
            continue
        if (
            width >= 12
            and height >= 16
            and x0 > int(start[0])
            and y0 > int(start[1])
            and y1 >= int(goal_site[1])
        ):
            if portal_component is None or area > _safe_int(
                portal_component.get("area")
            ):
                portal_component = dict(component)

    top_bar: dict[str, Any] | None = None
    mid_bar: dict[str, Any] | None = None
    for component in sorted(
        horizontal_bars,
        key=lambda row: (
            int((row.get("bbox") or [0, 0, 0, 0])[1]),
            int((row.get("bbox") or [0, 0, 0, 0])[0]),
        ),
    ):
        bbox = [int(v) for v in list(component.get("bbox") or [])[:4]]
        if len(bbox) < 4:
            continue
        if bbox[1] < int(start[1]) and top_bar is None:
            top_bar = dict(component)
            continue
        if int(start[1]) < bbox[1] < int(goal_site[1]):
            mid_bar = dict(component)
            break
    if top_bar is None or mid_bar is None:
        return {"rejected_reasons": ["missing_portal_hold_special_door_bars"]}
    if portal_component is None:
        return {"rejected_reasons": ["missing_portal_hold_component"]}
    if not vertical_doors:
        return {"rejected_reasons": ["missing_portal_hold_lower_door_component"]}

    lower_door = max(
        vertical_doors,
        key=lambda row: (
            int((row.get("bbox") or [0, 0, 0, 0])[3]),
            _safe_int(row.get("area")),
        ),
    )
    tbox = [int(v) for v in list(top_bar.get("bbox") or [])[:4]]
    mbox = [int(v) for v in list(mid_bar.get("bbox") or [])[:4]]
    lbox = [int(v) for v in list(lower_door.get("bbox") or [])[:4]]
    pbox = [int(v) for v in list(portal_component.get("bbox") or [])[:4]]
    if len(tbox) < 4 or len(mbox) < 4 or len(lbox) < 4 or len(pbox) < 4:
        return {"rejected_reasons": ["portal_hold_bbox_not_inferred"]}

    top_switch_y = int(tbox[1]) - 1
    stride_candidates = [
        abs(int(goal_site[0]) - int(start[0])),
        abs(int(goal_site[1]) - int(start[1])),
        abs(int(start[1]) - int(top_switch_y)),
        abs(int(mbox[1]) - int(tbox[1])),
    ]
    nonzero = [value for value in stride_candidates if value > 0]
    if not nonzero:
        return {"rejected_reasons": ["portal_hold_stride_not_inferred"]}
    stride = int(nonzero[0])
    for value in nonzero[1:]:
        stride = math.gcd(stride, int(value))
    if stride < 3 or stride > 8:
        return {"rejected_reasons": ["portal_hold_stride_not_inferred"]}

    switch_margin = max(1, int(stride) // 3)
    top_switch = (int(start[0]), int(top_switch_y))
    mid_switch = (int(mbox[0]) - int(switch_margin), int(mbox[1]) - 1)
    lower_switch = (int(lbox[0]) - 1, int(lbox[1]) - int(switch_margin))
    portal_switch = (int(pbox[2]) - 4, int(mid_switch[1]))
    upper_pad = (int(pbox[0]), int(lower_switch[1]))
    lower_pad = (int(pbox[0]), int(lower_switch[1]) + 2 * int(stride))
    bottom_lane = (int(upper_pad[0]) - int(stride), int(lower_pad[1]) + int(stride))
    upper_pre = (int(upper_pad[0]) - int(stride), int(top_switch[1]))
    current_low = (int(start[0]), int(start[1]) + int(stride))
    mid_low = (int(mid_switch[0]), int(start[1]) + int(stride))
    derived_points = [
        top_switch,
        mid_switch,
        lower_switch,
        portal_switch,
        upper_pad,
        lower_pad,
        bottom_lane,
        upper_pre,
        current_low,
        mid_low,
    ]
    if any(x < 0 or y < 0 for x, y in derived_points):
        return {"rejected_reasons": ["portal_hold_route_out_of_bounds"]}

    round1 = (
        _axis_path(start, top_switch, stride)
        + _axis_path(top_switch, current_low, stride)
        + _axis_path(current_low, mid_low, stride)
        + _axis_path(mid_low, (mid_switch[0], top_switch[1]), stride)
        + _axis_path((mid_switch[0], top_switch[1]), (portal_switch[0], top_switch[1]), stride)
        + _axis_path((portal_switch[0], top_switch[1]), portal_switch, stride)
    )
    round2 = (
        _axis_path(start, current_low, stride)
        + _axis_path(current_low, mid_low, stride)
        + _axis_path(mid_low, lower_switch, stride)
    )
    round3_to_pad_base = (
        _axis_path(start, current_low, stride)
        + _axis_path(current_low, mid_low, stride)
        + _axis_path(mid_low, (mid_switch[0], top_switch[1]), stride)
        + _axis_path((mid_switch[0], top_switch[1]), upper_pre, stride)
        + _axis_path(upper_pre, (upper_pre[0], upper_pad[1]), stride)
        + _axis_path((upper_pre[0], upper_pad[1]), upper_pad, stride)
    )
    sync_gap = int(len(round1) - len(round3_to_pad_base))
    if sync_gap < 0 or sync_gap % 2:
        return {"rejected_reasons": ["portal_hold_sync_length_mismatch"]}
    sync_prefix: list[str] = []
    for _ in range(sync_gap // 2):
        sync_prefix.extend(["ACTION2", "ACTION1"])
    round3_to_pad = sync_prefix + round3_to_pad_base
    lower_exit = (int(lower_pad[0]) - int(stride), int(lower_pad[1]))
    round3_to_goal = (
        _axis_path(lower_pad, lower_exit, stride)
        + _axis_path(lower_exit, bottom_lane, stride)
        + _axis_path(bottom_lane, (goal_site[0], bottom_lane[1]), stride)
        + _axis_path((goal_site[0], bottom_lane[1]), goal_site, stride)
    )
    round3 = round3_to_pad + round3_to_goal
    if not round1 or not round2 or not round3_to_pad or not round3_to_goal:
        return {"rejected_reasons": ["portal_hold_axis_path_not_compiled"]}
    if len(round1) != len(round3_to_pad):
        return {"rejected_reasons": ["portal_hold_sync_length_mismatch"]}

    return {
        "rejected_reasons": [],
        "goal_link_type": "round_memory_portal_hold_post_score_route",
        "player_anchor": list(start),
        "goal_anchor": list(goal_site),
        "primary_switch_anchor": list(top_switch),
        "secondary_switch_anchor": list(portal_switch),
        "tertiary_switch_anchor": list(lower_switch),
        "mid_switch_anchor": list(mid_switch),
        "portal_upper_pad_anchor": list(upper_pad),
        "portal_lower_pad_anchor": list(lower_pad),
        "stride": int(stride),
        "round1_actions": round1,
        "round2_actions": round2,
        "round3_actions": round3,
        "actor_goal_color": int(actor_color),
        "door_switch_color": int(top_bar.get("color", -1)),
        "portal_component_color": int(portal_component.get("color", -1)),
        "background_color": int(background_color),
        "candidate_score": round(float(best_pair_score), 3),
    }


def _discover_round_memory_top_pair_bottom_door_post_score_replan_roles(
    arr: np.ndarray,
    *,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    components = [dict(row) for row in list(state.get("components") or [])]
    background_colors = _background_like_colors(arr)
    background_color = _dominant_color(arr)
    small_by_color: dict[int, list[dict[str, Any]]] = {}
    for component in components:
        color = int(component.get("color", -1))
        if color in background_colors:
            continue
        if _round_memory_actor_goal_shape(component):
            small_by_color.setdefault(color, []).append(dict(component))

    best_pair: tuple[dict[str, Any], dict[str, Any], int] | None = None
    best_pair_score = float("-inf")
    for actor_color, rows in sorted(small_by_color.items()):
        if len(rows) < 2:
            continue
        for player in rows:
            start = _player_anchor(player)
            for goal in rows:
                if goal is player:
                    continue
                goal_site = _goal_anchor(goal)
                dx = int(start[0]) - int(goal_site[0])
                dy = int(goal_site[1]) - int(start[1])
                if dx < 6 or dx > 24 or dy < 12 or dy > 30:
                    continue
                score = float(dy + dx)
                if score <= best_pair_score:
                    continue
                best_pair_score = score
                best_pair = (dict(player), dict(goal), int(actor_color))
    if best_pair is None:
        return {
            "rejected_reasons": [
                "round_memory_top_pair_bottom_door_post_score_replan_no_candidate"
            ]
        }

    player, goal, actor_color = best_pair
    start = _player_anchor(player)
    goal_site = _goal_anchor(goal)

    vertical_bars: list[dict[str, Any]] = []
    horizontal_bars: list[dict[str, Any]] = []
    for component in components:
        color = int(component.get("color", -1))
        if color in background_colors or color == int(actor_color):
            continue
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        area = _safe_int(component.get("area"))
        if area < 20:
            continue
        if width <= 8 and height >= 16 and y0 < int(start[1]) and y1 <= int(start[1]):
            vertical_bars.append(dict(component))
            continue
        if height <= 8 and width >= 12 and y0 >= int(goal_site[1]):
            horizontal_bars.append(dict(component))

    if len(vertical_bars) < 4:
        return {"rejected_reasons": ["missing_top_pair_vertical_door_components"]}
    if not horizontal_bars:
        return {"rejected_reasons": ["missing_bottom_special_door_component"]}

    vertical_bars = sorted(
        vertical_bars,
        key=lambda row: int((row.get("bbox") or [0, 0, 0, 0])[0]),
    )
    left_special, right_special = vertical_bars[0], vertical_bars[1]
    left_top, right_top = vertical_bars[-2], vertical_bars[-1]
    bottom_bar = max(
        horizontal_bars,
        key=lambda row: (
            int((row.get("bbox") or [0, 0, 0, 0])[2])
            - int((row.get("bbox") or [0, 0, 0, 0])[0]),
            _safe_int(row.get("area")),
        ),
    )
    lsbox = [int(v) for v in list(left_special.get("bbox") or [])[:4]]
    rsbox = [int(v) for v in list(right_special.get("bbox") or [])[:4]]
    ltbox = [int(v) for v in list(left_top.get("bbox") or [])[:4]]
    rtbox = [int(v) for v in list(right_top.get("bbox") or [])[:4]]
    bbox = [int(v) for v in list(bottom_bar.get("bbox") or [])[:4]]
    if any(len(row) < 4 for row in [lsbox, rsbox, ltbox, rtbox, bbox]):
        return {"rejected_reasons": ["top_pair_bottom_door_bbox_not_inferred"]}

    stride_candidates = [
        abs(int(start[0]) - int(goal_site[0])),
        abs(int(goal_site[1]) - int(start[1])),
        abs(int(rtbox[0]) - int(ltbox[0])),
        abs(int(rsbox[0]) - int(lsbox[0])),
    ]
    nonzero = [value for value in stride_candidates if value > 0]
    if not nonzero:
        return {"rejected_reasons": ["top_pair_bottom_door_stride_not_inferred"]}
    stride = int(nonzero[0])
    for value in nonzero[1:]:
        stride = math.gcd(stride, int(value))
    if stride < 3 or stride > 8:
        return {"rejected_reasons": ["top_pair_bottom_door_stride_not_inferred"]}

    switch_margin = max(1, int(stride) // 3)
    top_switch_y = int(max(ltbox[3], rtbox[3])) - max(1, int(stride) - 2)
    left_top_switch = (int(ltbox[0]) - 1, int(top_switch_y))
    right_top_switch = (int(rtbox[0]) - 1, int(top_switch_y))
    left_special_x = int(lsbox[0]) - 1
    right_special_x = int(rsbox[0]) - 1
    bottom_switch = (int(bbox[0]) - int(switch_margin), int(bbox[1]) - 1)
    lower_lane_y = int(start[1]) + int(stride)
    bottom_lane_y = int(bottom_switch[1]) + int(stride)
    derived_points = [
        left_top_switch,
        right_top_switch,
        (left_special_x, top_switch_y),
        (right_special_x, top_switch_y),
        bottom_switch,
        (left_special_x, bottom_lane_y),
        (right_special_x, lower_lane_y),
        (left_top_switch[0], lower_lane_y),
        (left_top_switch[0], goal_site[1]),
    ]
    if any(x < 0 or y < 0 for x, y in derived_points):
        return {"rejected_reasons": ["top_pair_bottom_door_route_out_of_bounds"]}

    round1 = (
        _axis_path(start, (right_top_switch[0], start[1]), stride)
        + _axis_path((right_top_switch[0], start[1]), right_top_switch, stride)
    )
    round2 = (
        _axis_path(start, (right_top_switch[0], start[1]), stride)
        + _axis_path((right_top_switch[0], start[1]), right_top_switch, stride)
        + _axis_path(right_top_switch, left_top_switch, stride)
    )
    round3 = (
        _axis_path(start, (right_top_switch[0], lower_lane_y), stride)
        + _axis_path((right_top_switch[0], lower_lane_y), (right_special_x, lower_lane_y), stride)
        + _axis_path((right_special_x, lower_lane_y), (right_special_x, top_switch_y), stride)
        + _axis_path((right_special_x, top_switch_y), (left_special_x, top_switch_y), stride)
        + _axis_path((left_special_x, top_switch_y), (left_special_x, bottom_lane_y), stride)
        + _axis_path((left_special_x, bottom_lane_y), (bottom_switch[0], bottom_lane_y), stride)
        + _axis_path((bottom_switch[0], bottom_lane_y), bottom_switch, stride)
        + _axis_path(bottom_switch, (bottom_switch[0], bottom_lane_y), stride)
        + _axis_path((bottom_switch[0], bottom_lane_y), (left_special_x, bottom_lane_y), stride)
        + _axis_path((left_special_x, bottom_lane_y), (left_special_x, top_switch_y), stride)
        + _axis_path((left_special_x, top_switch_y), (right_special_x, top_switch_y), stride)
        + _axis_path((right_special_x, top_switch_y), (right_special_x, lower_lane_y), stride)
        + _axis_path((right_special_x, lower_lane_y), (left_top_switch[0], lower_lane_y), stride)
        + _axis_path((left_top_switch[0], lower_lane_y), (left_top_switch[0], goal_site[1]), stride)
        + _axis_path((left_top_switch[0], goal_site[1]), goal_site, stride)
    )
    if not round1 or not round2 or not round3:
        return {"rejected_reasons": ["top_pair_bottom_door_axis_path_not_compiled"]}

    return {
        "rejected_reasons": [],
        "goal_link_type": "round_memory_top_pair_bottom_door_post_score_route",
        "player_anchor": list(start),
        "goal_anchor": list(goal_site),
        "primary_switch_anchor": list(right_top_switch),
        "secondary_switch_anchor": list(left_top_switch),
        "tertiary_switch_anchor": list(bottom_switch),
        "mid_switch_anchor": [int(right_special_x), int(top_switch_y)],
        "portal_upper_pad_anchor": [],
        "portal_lower_pad_anchor": [],
        "stride": int(stride),
        "round1_actions": round1,
        "round2_actions": round2,
        "round3_actions": round3,
        "actor_goal_color": int(actor_color),
        "door_switch_color": int(right_top.get("color", -1)),
        "portal_component_color": int(bottom_bar.get("color", -1)),
        "background_color": int(background_color),
        "candidate_score": round(float(best_pair_score), 3),
    }


def _discover_round_memory_dual_portal_final_post_score_replan_roles(
    arr: np.ndarray,
    *,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    components = [dict(row) for row in list(state.get("components") or [])]
    background_colors = _background_like_colors(arr)
    background_color = _dominant_color(arr)
    small_by_color: dict[int, list[dict[str, Any]]] = {}
    for component in components:
        color = int(component.get("color", -1))
        if color in background_colors:
            continue
        if _round_memory_actor_goal_shape(component):
            small_by_color.setdefault(color, []).append(dict(component))

    best_pair: tuple[dict[str, Any], dict[str, Any], int] | None = None
    best_pair_score = float("-inf")
    for actor_color, rows in sorted(small_by_color.items()):
        if len(rows) < 2:
            continue
        for player in rows:
            start = _player_anchor(player)
            for goal in rows:
                if goal is player:
                    continue
                goal_site = _goal_anchor(goal)
                dx = int(goal_site[0]) - int(start[0])
                dy = int(goal_site[1]) - int(start[1])
                if dx <= 0 or dx > 12 or dy < 18 or dy > 36:
                    continue
                score = float(dy * 2 + dx)
                if score <= best_pair_score:
                    continue
                best_pair_score = score
                best_pair = (dict(player), dict(goal), int(actor_color))
    if best_pair is None:
        return {
            "rejected_reasons": [
                "round_memory_dual_portal_final_post_score_replan_no_candidate"
            ]
        }

    player, goal, actor_color = best_pair
    start = _player_anchor(player)
    goal_site = _goal_anchor(goal)
    stride_candidates = [
        abs(int(goal_site[0]) - int(start[0])),
        abs(int(goal_site[1]) - int(start[1])),
    ]
    nonzero = [value for value in stride_candidates if value > 0]
    if not nonzero:
        return {"rejected_reasons": ["dual_portal_final_stride_not_inferred"]}
    stride = int(nonzero[0])
    for value in nonzero[1:]:
        stride = math.gcd(stride, int(value))
    if stride < 3 or stride > 8:
        return {"rejected_reasons": ["dual_portal_final_stride_not_inferred"]}
    if int(goal_site[0]) != int(start[0]) + int(stride):
        return {"rejected_reasons": ["dual_portal_final_goal_x_not_stride_plus_one"]}
    if int(goal_site[1]) != int(start[1]) + 4 * int(stride):
        return {"rejected_reasons": ["dual_portal_final_goal_y_not_four_strides"]}

    sx, sy = int(start[0]), int(start[1])
    s = int(stride)
    special_switch = (sx - s, sy + 2 * s)
    left_portal_switch = (sx - 4 * s, sy)
    left_upper_pad = (sx - 2 * s, sy - 2 * s)
    left_lower_pad = (sx - 2 * s, sy)
    top_switch = (sx - 2 * s, sy - 4 * s)
    normal_door = (sx + s, sy - 4 * s)
    right_switch = (sx + 4 * s, sy - 4 * s)
    right_upper_pad = (sx + 4 * s, sy + 2 * s)
    right_lower_pad = (sx + 4 * s, sy + 4 * s)
    special_door = (sx - 4 * s, sy + 2 * s)
    derived_points = [
        special_switch,
        left_portal_switch,
        left_upper_pad,
        left_lower_pad,
        top_switch,
        normal_door,
        right_switch,
        right_upper_pad,
        right_lower_pad,
        special_door,
    ]
    height, width = arr.shape[:2]
    if any(x < 0 or y < 0 or x >= width or y >= height for x, y in derived_points):
        return {"rejected_reasons": ["dual_portal_final_route_out_of_bounds"]}

    center_offset = max(1, s // 2)

    def center(anchor: tuple[int, int]) -> tuple[int, int]:
        return int(anchor[0]) + center_offset, int(anchor[1]) + center_offset

    def contains(component: Mapping[str, Any], anchor: tuple[int, int]) -> bool:
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            return False
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        cx, cy = center(anchor)
        return x0 <= cx <= x1 and y0 <= cy <= y1

    def non_actor_components() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for component in components:
            color = int(component.get("color", -1))
            if color in background_colors or color == int(actor_color):
                continue
            rows.append(dict(component))
        return rows

    non_actor = non_actor_components()

    def find_network(
        anchors: Sequence[tuple[int, int]],
        *,
        min_area: int,
    ) -> dict[str, Any] | None:
        matches: list[dict[str, Any]] = []
        for component in non_actor:
            if _safe_int(component.get("area")) < int(min_area):
                continue
            if all(contains(component, anchor) for anchor in anchors):
                matches.append(dict(component))
        if not matches:
            return None
        return max(matches, key=lambda row: _safe_int(row.get("area")))

    left_portal_network = find_network(
        [left_portal_switch, left_upper_pad, left_lower_pad],
        min_area=24,
    )
    right_portal_network = find_network(
        [right_switch, right_upper_pad, right_lower_pad],
        min_area=24,
    )
    top_door_network = find_network([top_switch, normal_door], min_area=18)
    special_door_network = find_network([special_switch, special_door], min_area=18)
    if left_portal_network is None:
        return {"rejected_reasons": ["missing_dual_portal_left_network"]}
    if right_portal_network is None:
        return {"rejected_reasons": ["missing_dual_portal_right_network"]}
    if top_door_network is None:
        return {"rejected_reasons": ["missing_dual_portal_top_door_network"]}
    if special_door_network is None:
        return {"rejected_reasons": ["missing_dual_portal_special_door_network"]}

    round1 = (
        _axis_path(start, (sx, sy + 2 * s), s)
        + _axis_path((sx, sy + 2 * s), special_switch, s)
        + _axis_path(special_switch, (sx, sy + 2 * s), s)
        + _axis_path((sx, sy + 2 * s), start, s)
        + _axis_path(start, left_lower_pad, s)
        + _axis_path(left_upper_pad, top_switch, s)
    )
    round2 = (
        _axis_path(start, (sx, sy + 2 * s), s)
        + _axis_path((sx, sy + 2 * s), (sx + 2 * s, sy + 2 * s), s)
        + _axis_path((sx + 2 * s, sy + 2 * s), (sx + 2 * s, sy - 2 * s), s)
        + _axis_path((sx + 2 * s, sy - 2 * s), (sx + s, sy - 2 * s), s)
        + _axis_path((sx + s, sy - 2 * s), normal_door, s)
        + _axis_path(normal_door, right_switch, s)
    )
    round3 = (
        _axis_path(start, (sx, sy + 2 * s), s)
        + _axis_path((sx, sy + 2 * s), (sx + 2 * s, sy + 2 * s), s)
        + _axis_path((sx + 2 * s, sy + 2 * s), (sx + 2 * s, sy - 2 * s), s)
        + _axis_path((sx + 2 * s, sy - 2 * s), (sx + 4 * s, sy - 2 * s), s)
        + _axis_path((sx + 4 * s, sy - 2 * s), right_upper_pad, s)
        + _axis_path(right_lower_pad, goal_site, s)
    )
    if not round1 or not round2 or not round3:
        return {"rejected_reasons": ["dual_portal_final_axis_path_not_compiled"]}

    return {
        "rejected_reasons": [],
        "goal_link_type": "round_memory_dual_portal_final_post_score_route",
        "player_anchor": list(start),
        "goal_anchor": list(goal_site),
        "primary_switch_anchor": list(special_switch),
        "secondary_switch_anchor": list(top_switch),
        "tertiary_switch_anchor": list(right_switch),
        "mid_switch_anchor": list(left_portal_switch),
        "portal_upper_pad_anchor": list(right_upper_pad),
        "portal_lower_pad_anchor": list(right_lower_pad),
        "stride": int(stride),
        "round1_actions": round1,
        "round2_actions": round2,
        "round3_actions": round3,
        "actor_goal_color": int(actor_color),
        "door_switch_color": int(top_door_network.get("color", -1)),
        "portal_component_color": int(right_portal_network.get("color", -1)),
        "background_color": int(background_color),
        "candidate_score": round(float(best_pair_score), 3),
    }


def _discover_round_memory_portal_switch_post_score_replan_roles(
    arr: np.ndarray,
    *,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    components = [dict(row) for row in list(state.get("components") or [])]
    background_colors = _background_like_colors(arr)
    background_color = _dominant_color(arr)
    small_by_color: dict[int, list[dict[str, Any]]] = {}
    for component in components:
        color = int(component.get("color", -1))
        if color in background_colors:
            continue
        if _round_memory_actor_goal_shape(component):
            bbox = component.get("bbox")
            if _is_bbox(bbox) and int(bbox[1]) >= 7:
                small_by_color.setdefault(color, []).append(dict(component))

    best_pair: tuple[dict[str, Any], dict[str, Any], int] | None = None
    best_pair_score = float("-inf")
    for actor_color, rows in sorted(small_by_color.items()):
        if len(rows) < 2:
            continue
        for player in rows:
            start = _player_anchor(player)
            for goal in rows:
                if goal is player:
                    continue
                goal_site = _goal_anchor(goal)
                dx = int(start[0]) - int(goal_site[0])
                dy = int(goal_site[1]) - int(start[1])
                if dx <= 0 or dy <= 0:
                    continue
                if dx < 12 or dy < 24:
                    continue
                score = float(dx + dy)
                if score <= best_pair_score:
                    continue
                best_pair_score = score
                best_pair = (dict(player), dict(goal), int(actor_color))
    if best_pair is None:
        return {
            "rejected_reasons": [
                "round_memory_portal_switch_post_score_replan_no_candidate"
            ]
        }

    player, goal, actor_color = best_pair
    start = _player_anchor(player)
    goal_site = _goal_anchor(goal)

    door_component: dict[str, Any] | None = None
    portal_component: dict[str, Any] | None = None
    for component in components:
        color = int(component.get("color", -1))
        if color in background_colors or color == int(actor_color):
            continue
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        area = _safe_int(component.get("area"))
        if area < 20:
            continue
        if (
            height <= 8
            and width >= 12
            and start[1] < y0 < goal_site[1]
            and x0 <= goal_site[0] + 2
            and x1 > start[0]
        ):
            if door_component is None or width > (
                int((door_component.get("bbox") or [0, 0, -1, 0])[2])
                - int((door_component.get("bbox") or [0, 0, 0, 0])[0])
                + 1
            ):
                door_component = dict(component)
            continue
        if (
            width >= 16
            and height >= 16
            and x0 >= min(start[0], goal_site[0])
            and x1 > start[0]
            and y0 > start[1]
            and y1 >= goal_site[1]
        ):
            if portal_component is None or area > _safe_int(
                portal_component.get("area")
            ):
                portal_component = dict(component)

    if door_component is None:
        return {"rejected_reasons": ["missing_portal_route_door_component"]}
    if portal_component is None:
        return {"rejected_reasons": ["missing_portal_route_component"]}

    dbox = [int(v) for v in list(door_component.get("bbox") or [])[:4]]
    pbox = [int(v) for v in list(portal_component.get("bbox") or [])[:4]]
    if len(dbox) < 4 or len(pbox) < 4:
        return {"rejected_reasons": ["portal_route_bbox_not_inferred"]}

    door_switch = (int(dbox[2]) - 4, int(dbox[1]) - 1)
    stride_candidates = [
        abs(int(door_switch[0]) - int(start[0])),
        abs(int(door_switch[1]) - int(start[1])),
        abs(int(start[0]) - int(goal_site[0])),
        abs(int(goal_site[1]) - int(start[1])),
    ]
    nonzero = [value for value in stride_candidates if value > 0]
    if not nonzero:
        return {"rejected_reasons": ["portal_route_stride_not_inferred"]}
    stride = int(nonzero[0])
    for value in nonzero[1:]:
        stride = math.gcd(stride, int(value))
    if stride < 3 or stride > 8:
        return {"rejected_reasons": ["portal_route_stride_not_inferred"]}

    portal_switch = (int(pbox[2]) - 4, int(door_switch[1]))
    lower_pad = (int(pbox[0]), int(goal_site[1]))
    upper_pad = (int(lower_pad[0]), int(lower_pad[1]) - 2 * int(stride))
    door_pre = (int(start[0]), int(door_switch[1]) - int(stride))
    portal_mid = (int(door_switch[0]) + int(stride), int(door_pre[1]))
    portal_top = (int(portal_mid[0]), int(start[1]))
    left_top = (int(goal_site[0]), int(start[1]))
    left_upper = (int(goal_site[0]), int(upper_pad[1]))

    derived_points = [
        door_switch,
        portal_switch,
        lower_pad,
        upper_pad,
        door_pre,
        portal_mid,
        portal_top,
        left_top,
        left_upper,
    ]
    if any(x < 0 or y < 0 for x, y in derived_points):
        return {"rejected_reasons": ["portal_route_out_of_bounds"]}

    round1 = (
        _axis_path(start, door_pre, stride)
        + _axis_path(door_pre, (door_switch[0], door_pre[1]), stride)
        + _axis_path((door_switch[0], door_pre[1]), door_switch, stride)
    )
    round2 = (
        _axis_path(start, door_pre, stride)
        + _axis_path(door_pre, portal_mid, stride)
        + _axis_path(portal_mid, portal_top, stride)
        + _axis_path(portal_top, (portal_switch[0], portal_top[1]), stride)
        + _axis_path((portal_switch[0], portal_top[1]), portal_switch, stride)
    )
    round3_to_pad = (
        _axis_path(start, left_top, stride)
        + _axis_path(left_top, left_upper, stride)
        + _axis_path(left_upper, upper_pad, stride)
    )
    round3 = round3_to_pad + _axis_path(lower_pad, goal_site, stride)
    if not round1 or not round2 or not round3:
        return {"rejected_reasons": ["portal_route_axis_path_not_compiled"]}
    if len(round2) != len(round3_to_pad):
        return {"rejected_reasons": ["portal_route_sync_length_mismatch"]}

    return {
        "rejected_reasons": [],
        "goal_link_type": "round_memory_portal_switch_post_score_route",
        "player_anchor": list(start),
        "goal_anchor": list(goal_site),
        "primary_switch_anchor": list(door_switch),
        "secondary_switch_anchor": list(portal_switch),
        "tertiary_switch_anchor": [],
        "portal_upper_pad_anchor": list(upper_pad),
        "portal_lower_pad_anchor": list(lower_pad),
        "stride": int(stride),
        "round1_actions": round1,
        "round2_actions": round2,
        "round3_actions": round3,
        "actor_goal_color": int(actor_color),
        "door_switch_color": int(door_component.get("color", -1)),
        "portal_component_color": int(portal_component.get("color", -1)),
        "background_color": int(background_color),
        "candidate_score": round(float(best_pair_score), 3),
    }


def _discover_round_memory_post_score_replan_roles(
    arr: np.ndarray,
    *,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    components = [dict(row) for row in list(state.get("components") or [])]
    background_colors = _background_like_colors(arr)
    background_color = _dominant_color(arr)
    small_by_color: dict[int, list[dict[str, Any]]] = {}
    for component in components:
        color = int(component.get("color", -1))
        if color in background_colors:
            continue
        if _round_memory_actor_goal_shape(component):
            bbox = component.get("bbox")
            if _is_bbox(bbox) and int(bbox[1]) >= 7:
                small_by_color.setdefault(color, []).append(dict(component))
    best_pair: tuple[dict[str, Any], dict[str, Any], int] | None = None
    best_pair_score = float("-inf")
    for actor_color, rows in sorted(small_by_color.items()):
        if len(rows) < 2:
            continue
        for player in rows:
            start = _player_anchor(player)
            for goal in rows:
                if goal is player:
                    continue
                goal_site = _goal_anchor(goal)
                if goal_site[0] >= start[0] or goal_site[1] > start[1]:
                    continue
                separation = abs(start[0] - goal_site[0]) + abs(start[1] - goal_site[1])
                if separation <= best_pair_score:
                    continue
                best_pair_score = float(separation)
                best_pair = (dict(player), dict(goal), int(actor_color))
    if best_pair is None:
        return {"rejected_reasons": ["round_memory_post_score_replan_no_candidate"]}

    player, goal, actor_color = best_pair
    start = _player_anchor(player)
    goal_site = _goal_anchor(goal)
    door_rows_by_color: dict[int, list[dict[str, Any]]] = {}
    for component in components:
        color = int(component.get("color", -1))
        if color in background_colors or color == int(actor_color):
            continue
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        area = _safe_int(component.get("area"))
        if area < 24 or area > 128 or width < 3 or width > 12 or height < 10:
            continue
        door_rows_by_color.setdefault(color, []).append(dict(component))
    door_color = None
    door_components: list[dict[str, Any]] = []
    for color, rows in sorted(door_rows_by_color.items()):
        if len(rows) < 2:
            continue
        score = sum(_safe_int(row.get("area")) for row in rows)
        if door_color is None or score > sum(_safe_int(row.get("area")) for row in door_components):
            door_color = int(color)
            door_components = list(rows)
    if door_color is None or len(door_components) < 2:
        return {"rejected_reasons": ["missing_post_score_dual_switch_components"]}

    stride_candidates: list[int] = [
        abs(int(start[0]) - int(goal_site[0])),
        abs(int(start[1]) - int(goal_site[1])),
    ]
    for component in door_components:
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        for value in (
            abs((x0 - 1) - start[0]),
            abs((x0 - 1) - goal_site[0]),
        ):
            if value > 0:
                stride_candidates.append(int(value))
    stride = stride_candidates[0] if stride_candidates else 0
    for value in stride_candidates[1:]:
        stride = math.gcd(stride, int(value))
    if stride < 3 or stride > 8:
        return {"rejected_reasons": ["post_score_stride_not_inferred"]}

    primary: dict[str, Any] | None = None
    primary_distance = 1e9
    secondary: dict[str, Any] | None = None
    for component in door_components:
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        same_row_anchor = (x0 - 1, start[1])
        distance = abs(same_row_anchor[1] - start[1]) + abs(same_row_anchor[0] - start[0])
        if same_row_anchor[0] < start[0] and distance < primary_distance:
            primary_distance = float(distance)
            primary = dict(component)
    if primary is None:
        return {"rejected_reasons": ["missing_primary_same_row_switch"]}
    primary_bbox = list(primary.get("bbox") or [])
    for component in sorted(door_components, key=lambda row: int((row.get("bbox") or [999])[0])):
        if list(component.get("bbox") or []) == primary_bbox:
            continue
        secondary = dict(component)
        break
    if secondary is None:
        return {"rejected_reasons": ["missing_secondary_switch"]}

    pbox = [int(v) for v in list(primary.get("bbox") or [])[:4]]
    sbox = [int(v) for v in list(secondary.get("bbox") or [])[:4]]
    primary_switch = (int(pbox[0]) - 1, int(start[1]))
    secondary_switch = (int(sbox[0]) - 1, int(sbox[3]) - 4)
    bottom_y = int(pbox[3]) - 5
    top_y = int(goal_site[1]) - 2 * int(stride)
    left_x = int(secondary_switch[0]) - int(stride)
    if top_y < 0 or left_x < 0:
        return {"rejected_reasons": ["post_score_route_out_of_bounds"]}

    round1 = _axis_path(start, primary_switch, stride)
    round2 = (
        _axis_path(start, (start[0], bottom_y), stride)
        + _axis_path((start[0], bottom_y), (goal_site[0], bottom_y), stride)
        + _axis_path((goal_site[0], bottom_y), (goal_site[0], secondary_switch[1]), stride)
        + _axis_path((goal_site[0], secondary_switch[1]), secondary_switch, stride)
    )
    round3 = (
        _axis_path(start, (start[0], top_y), stride)
        + _axis_path((start[0], top_y), (left_x, top_y), stride)
        + _axis_path((left_x, top_y), (left_x, goal_site[1]), stride)
        + _axis_path((left_x, goal_site[1]), goal_site, stride)
    )
    if not round1 or not round2 or not round3:
        return {"rejected_reasons": ["post_score_axis_path_not_compiled"]}
    return {
        "rejected_reasons": [],
        "player_anchor": list(start),
        "goal_anchor": list(goal_site),
        "primary_switch_anchor": list(primary_switch),
        "secondary_switch_anchor": list(secondary_switch),
        "stride": int(stride),
        "round1_actions": round1,
        "round2_actions": round2,
        "round3_actions": round3,
        "actor_goal_color": int(actor_color),
        "door_switch_color": int(door_color),
        "background_color": int(background_color),
        "candidate_score": round(float(best_pair_score), 3),
    }


def _round_memory_topology(arr: np.ndarray) -> dict[str, Any]:
    components = _connected_components(arr)
    by_color: dict[int, list[dict[str, Any]]] = {}
    for component in components:
        by_color.setdefault(int(component.get("color", -1)), []).append(component)
    color_counts = {
        int(color): int(np.count_nonzero(arr == color))
        for color in np.unique(arr).tolist()
    }
    color1_count = int(color_counts.get(1, 0))
    color2_count = int(color_counts.get(2, 0))
    player_components = _player_like_components(by_color.get(9, []))
    goal_components = _goal_like_components(by_color.get(9, []))
    return {
        "components": components,
        "color_counts": color_counts,
        "color1_count": color1_count,
        "color2_count": color2_count,
        "round_progress_visible": bool(color2_count > 0),
        "player_components": player_components,
        "goal_components": goal_components,
        "color8_components": by_color.get(8, []),
        "color8_area": _area_sum(by_color.get(8, [])),
        "color8_component_count": len(by_color.get(8, [])),
        "player_component_count": len(player_components),
        "goal_component_count": len(goal_components),
    }


def _discover_round_memory_proposal_roles(arr: np.ndarray) -> dict[str, Any]:
    components = _connected_components(arr)
    background_color = _dominant_color(arr)
    background_colors = _background_like_colors(arr)
    height, width = arr.shape[:2]
    small_by_color: dict[int, list[dict[str, Any]]] = {}
    for component in components:
        color = int(component.get("color", -1))
        if color in background_colors:
            continue
        if _round_memory_actor_goal_shape(component):
            small_by_color.setdefault(color, []).append(dict(component))

    rejection_counts: Counter[str] = Counter()
    best: dict[str, Any] | None = None
    best_score = -1.0
    for actor_color, actor_components in sorted(small_by_color.items()):
        if len(actor_components) < 2:
            continue
        for player in actor_components:
            player_anchor = _player_anchor(player)
            for goal in actor_components:
                if goal is player:
                    continue
                goal_anchor = _goal_anchor(goal)
                pair_reasons = _round_memory_actor_goal_pair_rejections(
                    player_anchor,
                    goal_anchor,
                    width=width,
                    height=height,
                )
                if pair_reasons:
                    rejection_counts.update(pair_reasons)
                    continue
                door = _best_round_memory_door_component(
                    components,
                    background_colors=background_colors,
                    actor_color=actor_color,
                    player_anchor=player_anchor,
                    goal_anchor=goal_anchor,
                )
                if door is None:
                    rejection_counts["missing_door_switch_component"] += 1
                    continue
                switch_anchor = _switch_anchor(door, start_y=player_anchor[1])
                stride = _infer_round_memory_stride(
                    player_anchor,
                    switch_anchor,
                    goal_anchor,
                )
                if stride <= 0:
                    rejection_counts["stride_not_inferred"] += 1
                    continue
                marker = _discover_round_memory_marker_state(
                    components,
                    background_colors=background_colors,
                    excluded_colors={actor_color, int(door.get("color", -1))},
                    width=width,
                    height=height,
                )
                marker_reasons = list(marker.get("rejected_reasons") or [])
                if marker_reasons:
                    rejection_counts.update(str(row) for row in marker_reasons)
                    continue
                plan_score = (
                    abs(switch_anchor[0] - player_anchor[0])
                    + abs(goal_anchor[1] - player_anchor[1])
                    + abs(goal_anchor[0] - player_anchor[0])
                ) / max(1, stride)
                separation = math.hypot(
                    goal_anchor[0] - player_anchor[0],
                    goal_anchor[1] - player_anchor[1],
                )
                score = float(separation) - float(plan_score)
                if score <= best_score:
                    continue
                best_score = score
                best = {
                    "player_component": dict(player),
                    "goal_component": dict(goal),
                    "door_switch_component": dict(door),
                    "player_anchor": list(player_anchor),
                    "goal_anchor": list(goal_anchor),
                    "switch_anchor": list(switch_anchor),
                    "stride": int(stride),
                    "background_color": int(background_color),
                    "background_colors": sorted(int(color) for color in background_colors),
                    "actor_goal_color": int(actor_color),
                    "door_switch_color": int(door.get("color", -1)),
                    "round_marker_color": marker.get("round_marker_color"),
                    "round_marker_count": int(marker.get("round_marker_count", 0)),
                    "round_progress_visible": bool(marker.get("round_progress_visible")),
                    "candidate_score": round(float(score), 3),
                    "rejected_reasons": [],
                }
    if best is not None:
        return best

    reasons = [
        reason
        for reason, _count in rejection_counts.most_common()
        if str(reason)
    ]
    if not reasons:
        if not any(len(rows) >= 2 for rows in small_by_color.values()):
            reasons.append("missing_actor_goal_pair")
        else:
            reasons.append("round_memory_role_discovery_no_candidate")
    return {
        "background_color": int(background_color),
        "background_colors": sorted(int(color) for color in background_colors),
        "rejected_reasons": reasons,
    }


def _dominant_color(arr: np.ndarray) -> int:
    values, counts = np.unique(arr, return_counts=True)
    if len(values) == 0:
        return 0
    index = int(np.argmax(counts))
    return int(values[index])


def _background_like_colors(arr: np.ndarray) -> set[int]:
    total = int(arr.size)
    if total <= 0:
        return {0}
    colors = {_dominant_color(arr)}
    border = np.concatenate(
        [
            arr[0, :].reshape(-1),
            arr[-1, :].reshape(-1),
            arr[:, 0].reshape(-1),
            arr[:, -1].reshape(-1),
        ]
    )
    border_total = int(border.size)
    if border_total > 0:
        total_counts = {
            int(value): int(count)
            for value, count in zip(*np.unique(arr, return_counts=True))
        }
        values, counts = np.unique(border, return_counts=True)
        threshold = max(16, int(border_total * 0.20))
        total_threshold = max(128, int(total * 0.05))
        for value, count in zip(values.tolist(), counts.tolist()):
            color = int(value)
            if int(count) >= threshold and int(total_counts.get(color, 0)) >= total_threshold:
                colors.add(color)
    return colors


def _round_memory_actor_goal_shape(component: Mapping[str, Any]) -> bool:
    bbox = component.get("bbox")
    if not _is_bbox(bbox):
        return False
    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    area = _safe_int(component.get("area"))
    if area < 12 or area > 48:
        return False
    if width < 3 or height < 3 or width > 9 or height > 9:
        return False
    # Goal sprites can be hollow or framed, so density must be permissive.
    # Geometry and route/marker/door constraints below provide the real guard.
    return float(area) / float(max(1, width * height)) >= 0.25


def _round_memory_actor_goal_pair_rejections(
    player_anchor: tuple[int, int],
    goal_anchor: tuple[int, int],
    *,
    width: int,
    height: int,
) -> list[str]:
    reasons: list[str] = []
    if goal_anchor[0] <= player_anchor[0] + max(8, width // 8):
        reasons.append("goal_not_right_of_player")
    if goal_anchor[1] <= player_anchor[1] + max(12, height // 5):
        reasons.append("goal_not_below_player")
    if goal_anchor[0] < int(width * 0.55) or goal_anchor[1] < int(height * 0.65):
        reasons.append("goal_not_lower_right")
    if player_anchor[1] < 4:
        reasons.append("player_too_close_to_top")
    return reasons


def _best_round_memory_door_component(
    components: Sequence[Mapping[str, Any]],
    *,
    background_colors: set[int],
    actor_color: int,
    player_anchor: tuple[int, int],
    goal_anchor: tuple[int, int],
) -> Mapping[str, Any] | None:
    best: Mapping[str, Any] | None = None
    best_score = float("-inf")
    px, py = player_anchor
    gx, gy = goal_anchor
    for component in components:
        color = int(component.get("color", -1))
        if color in set(background_colors) | {int(actor_color)}:
            continue
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        area = _safe_int(component.get("area"))
        if area < 64 or width < 3 or height < 12:
            continue
        if not (x0 - 2 <= px <= x1 + 2):
            continue
        if not (y0 - 3 <= py <= y1 + 3):
            continue
        if x1 <= px + 6 or x1 > gx + 8:
            continue
        if y1 <= py + 8 or y0 >= gy:
            continue
        switch_anchor = _switch_anchor(component, start_y=py)
        stride = _infer_round_memory_stride(player_anchor, switch_anchor, goal_anchor)
        if stride <= 0:
            continue
        door_entry = (px, gy)
        estimated_plan_length = (
            len(_axis_path(player_anchor, switch_anchor, stride))
            + 1
            + len(_axis_path(player_anchor, door_entry, stride))
            + len(_axis_path(door_entry, goal_anchor, stride))
        )
        if estimated_plan_length <= 1:
            continue
        density = float(area) / float(max(1, width * height))
        sparse_structure_bonus = 0.5 if density <= 0.35 else 0.0
        score = (
            -float(estimated_plan_length)
            + sparse_structure_bonus
            + min(float(area), 200.0) / 1000.0
        )
        if score > best_score:
            best_score = score
            best = component
    return best


def _infer_round_memory_stride(
    player_anchor: tuple[int, int],
    switch_anchor: tuple[int, int],
    goal_anchor: tuple[int, int],
) -> int:
    values = [
        abs(int(switch_anchor[0]) - int(player_anchor[0])),
        abs(int(goal_anchor[1]) - int(player_anchor[1])),
        abs(int(goal_anchor[0]) - int(player_anchor[0])),
    ]
    nonzero = [value for value in values if value > 0]
    if not nonzero:
        return 0
    stride = nonzero[0]
    for value in nonzero[1:]:
        stride = math.gcd(stride, value)
    if stride < 3 or stride > 8:
        return 0
    if any(value % stride for value in nonzero):
        return 0
    return int(stride)


def _discover_round_memory_marker_state(
    components: Sequence[Mapping[str, Any]],
    *,
    background_colors: set[int],
    excluded_colors: set[int],
    width: int,
    height: int,
) -> dict[str, Any]:
    x_limit = max(8, int(width * 0.2))
    y_limit = max(8, int(height * 0.15))
    by_color: Counter[int] = Counter()
    for component in components:
        color = int(component.get("color", -1))
        if color in set(background_colors) or color in excluded_colors:
            continue
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        area = _safe_int(component.get("area"))
        if area <= 0 or area > 16:
            continue
        if x0 < x_limit and y0 < y_limit and x1 < x_limit and y1 < y_limit:
            by_color[color] += area
    if not by_color:
        return {"rejected_reasons": ["missing_round_marker_candidate"]}
    marker_color, marker_count = by_color.most_common(1)[0]
    reasons: list[str] = []
    if marker_count > 12:
        reasons.append("unsupported_round_counter_count")
    if len(by_color) > 1:
        reasons.append("round_memory_already_submitted")
    return {
        "round_marker_color": int(marker_color),
        "round_marker_count": int(marker_count),
        "round_progress_visible": bool(len(by_color) > 1),
        "rejected_reasons": reasons,
    }


def _round_marker_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    before_progress = bool(before.get("round_progress_visible"))
    after_progress = bool(after.get("round_progress_visible"))
    return {
        "color1_count_before": int(before.get("color1_count", 0)),
        "color1_count_after": int(after.get("color1_count", 0)),
        "color1_count_delta": int(after.get("color1_count", 0))
        - int(before.get("color1_count", 0)),
        "color2_count_before": int(before.get("color2_count", 0)),
        "color2_count_after": int(after.get("color2_count", 0)),
        "color2_count_delta": int(after.get("color2_count", 0))
        - int(before.get("color2_count", 0)),
        "round_progress_visible_before": before_progress,
        "round_progress_visible_after": after_progress,
        "progress_visibility_changed": bool(before_progress != after_progress),
    }


def _player_motion(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_players = list(before.get("player_components") or [])
    after_players = list(after.get("player_components") or [])
    left = _largest_component(before_players)
    if not left:
        return {
            "moved_player": False,
            "delta": [0.0, 0.0],
            "before_centroid": None,
            "after_centroid": None,
        }
    right = _nearest_component(left, after_players)
    if not right:
        return {
            "moved_player": False,
            "delta": [0.0, 0.0],
            "before_centroid": _rounded_centroid(left),
            "after_centroid": None,
        }
    bx, by = _centroid_tuple(left)
    ax, ay = _centroid_tuple(right)
    dx = ax - bx
    dy = ay - by
    return {
        "moved_player": bool(math.hypot(dx, dy) >= 0.5),
        "delta": [round(dx, 3), round(dy, 3)],
        "before_centroid": [round(bx, 3), round(by, 3)],
        "after_centroid": [round(ax, 3), round(ay, 3)],
        "before_component": _compact_component(left),
        "after_component": _compact_component(right),
    }


def _color8_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_components = list(before.get("color8_components") or [])
    after_components = list(after.get("color8_components") or [])
    moved: list[dict[str, Any]] = []
    for left, right in _match_components(before_components, after_components, color=8):
        bx, by = _centroid_tuple(left)
        ax, ay = _centroid_tuple(right)
        dx = ax - bx
        dy = ay - by
        if math.hypot(dx, dy) < 0.5:
            continue
        moved.append(
            {
                "before_centroid": [round(bx, 3), round(by, 3)],
                "after_centroid": [round(ax, 3), round(ay, 3)],
                "delta": [round(dx, 3), round(dy, 3)],
                "before_area": int(left.get("area", 0)),
                "after_area": int(right.get("area", 0)),
            }
        )
    area_delta = int(after.get("color8_area", 0)) - int(before.get("color8_area", 0))
    count_delta = int(after.get("color8_component_count", 0)) - int(
        before.get("color8_component_count", 0)
    )
    return {
        "color8_area_before": int(before.get("color8_area", 0)),
        "color8_area_after": int(after.get("color8_area", 0)),
        "color8_area_delta": int(area_delta),
        "color8_component_count_before": int(before.get("color8_component_count", 0)),
        "color8_component_count_after": int(after.get("color8_component_count", 0)),
        "color8_component_count_delta": int(count_delta),
        "moved_color8_component_count": int(len(moved)),
        "moved_color8_components": moved[:8],
        "door_switch_state_delta_candidate": bool(
            area_delta != 0 or count_delta != 0 or moved
        ),
    }


def _goal_relation(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_player = _largest_component(before.get("player_components") or [])
    after_player = _largest_component(after.get("player_components") or [])
    before_goal = _largest_component(before.get("goal_components") or [])
    after_goal = _largest_component(after.get("goal_components") or [])
    if not before_player or not after_player or not before_goal or not after_goal:
        return {
            "distance_before": None,
            "distance_after": None,
            "distance_delta": None,
        }
    bpx, bpy = _centroid_tuple(before_player)
    apx, apy = _centroid_tuple(after_player)
    bgx, bgy = _centroid_tuple(before_goal)
    agx, agy = _centroid_tuple(after_goal)
    before_distance = math.hypot(bpx - bgx, bpy - bgy)
    after_distance = math.hypot(apx - agx, apy - agy)
    return {
        "distance_before": round(before_distance, 3),
        "distance_after": round(after_distance, 3),
        "distance_delta": round(after_distance - before_distance, 3),
        "before_player_centroid": [round(bpx, 3), round(bpy, 3)],
        "after_player_centroid": [round(apx, 3), round(apy, 3)],
        "before_goal_centroid": [round(bgx, 3), round(bgy, 3)],
        "after_goal_centroid": [round(agx, 3), round(agy, 3)],
    }


def _change_observation(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    changed = before != after
    round_marker_mask = changed & (np.isin(before, [1, 2]) | np.isin(after, [1, 2]))
    player_mask = changed & (np.isin(before, [9]) | np.isin(after, [9]))
    door_switch_mask = changed & (np.isin(before, [8]) | np.isin(after, [8]))
    return {
        "changed_cell_count": int(np.count_nonzero(changed)),
        "round_marker_changed_cell_count": int(np.count_nonzero(round_marker_mask)),
        "player_changed_cell_count": int(np.count_nonzero(player_mask)),
        "door_switch_changed_cell_count": int(np.count_nonzero(door_switch_mask)),
    }


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


def _player_like_components(
    components: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for component in components:
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        area = _safe_int(component.get("area"))
        if area < 16 or area > 36:
            continue
        if width > 9 or height > 9:
            continue
        if x0 >= 42 and y0 >= 48:
            continue
        if y0 < 7:
            continue
        out.append(dict(component))
    return sorted(out, key=lambda row: -_safe_int(row.get("area")))


def _goal_like_components(
    components: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for component in components:
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        area = _safe_int(component.get("area"))
        if area < 12 or area > 48:
            continue
        if x0 >= 40 and y0 >= 45:
            out.append(dict(component))
    return sorted(out, key=lambda row: -_safe_int(row.get("area")))


def _match_components(
    before_components: Sequence[Mapping[str, Any]],
    after_components: Sequence[Mapping[str, Any]],
    *,
    color: int,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    remaining = [row for row in after_components if int(row.get("color", -1)) == color]
    matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for before in before_components:
        if int(before.get("color", -1)) != color:
            continue
        best = _nearest_component(before, remaining)
        if best is None:
            continue
        remaining.remove(best)
        matches.append((before, best))
    return matches


def _nearest_component(
    source: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    sx, sy = _centroid_tuple(source)
    best: Mapping[str, Any] | None = None
    best_distance = 1e9
    for candidate in candidates:
        cx, cy = _centroid_tuple(candidate)
        distance = math.hypot(cx - sx, cy - sy)
        if distance < best_distance:
            best_distance = distance
            best = candidate
    if best is None or best_distance > 64.0:
        return None
    return best


def _compact_counts(topology: Mapping[str, Any]) -> dict[str, int | bool]:
    return {
        "color1_count": int(topology.get("color1_count", 0)),
        "color2_count": int(topology.get("color2_count", 0)),
        "round_progress_visible": bool(topology.get("round_progress_visible")),
        "player_component_count": int(topology.get("player_component_count", 0)),
        "goal_component_count": int(topology.get("goal_component_count", 0)),
        "color8_area": int(topology.get("color8_area", 0)),
        "color8_component_count": int(topology.get("color8_component_count", 0)),
    }


def _compact_component(component: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "color": int(component.get("color", -1)),
        "area": _safe_int(component.get("area")),
        "bbox": list(component.get("bbox") or []),
        "centroid": list(component.get("centroid") or []),
    }


def _largest_component(components: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not components:
        return None
    return max(components, key=lambda row: _safe_int(row.get("area")))


def _area_sum(components: Sequence[Mapping[str, Any]]) -> int:
    return int(sum(_safe_int(row.get("area")) for row in components))


def _centroid_tuple(component: Mapping[str, Any]) -> tuple[float, float]:
    centroid = component.get("centroid")
    if isinstance(centroid, Sequence) and len(centroid) >= 2:
        return float(centroid[0]), float(centroid[1])
    bbox = component.get("bbox")
    if _is_bbox(bbox):
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return 0.0, 0.0


def _rounded_centroid(component: Mapping[str, Any]) -> list[float]:
    x, y = _centroid_tuple(component)
    return [round(x, 3), round(y, 3)]


def _mechanic_state_type(packet: Mapping[str, Any]) -> str:
    types = set(str(row) for row in list(packet.get("candidate_types") or []))
    if "score_closure" in types:
        return "round_memory_score_closure"
    if "switch_memory" in types:
        return "round_memory_switch_replay_relation"
    if "memory_replay" in types:
        return "round_memory_replay_relation"
    if "round_submit" in types:
        return "round_submit_memory_recording"
    if "round_progress" in types:
        return "round_progress_marker_relation"
    return "round_memory_observation"


def _round_memory_posterior_key(
    packet: Mapping[str, Any],
    *,
    validation: Mapping[str, Any],
) -> str:
    game_id = str(packet.get("game_id", "") or "unknown")
    mechanic = str(validation.get("mechanic_state_type", "") or "round_memory")
    action = _normalize_action_name(packet.get("action_name", "") or "unknown")
    types = "+".join(
        sorted(str(row) for row in list(packet.get("candidate_types") or []) if row)
    ) or "none"
    return f"{game_id}|{mechanic}|{types}|{action}"


def _player_anchor(component: Mapping[str, Any]) -> tuple[int, int]:
    bbox = component.get("bbox")
    if _is_bbox(bbox):
        x0, y0 = int(bbox[0]), int(bbox[1])
        return x0 - 1, y0 - 1
    x, y = _centroid_tuple(component)
    return int(round(x)), int(round(y))


def _goal_anchor(component: Mapping[str, Any]) -> tuple[int, int]:
    bbox = component.get("bbox")
    if _is_bbox(bbox):
        return int(bbox[0]), int(bbox[1])
    x, y = _centroid_tuple(component)
    return int(round(x)), int(round(y))


def _switch_anchor(component: Mapping[str, Any], *, start_y: int) -> tuple[int, int]:
    bbox = component.get("bbox")
    if _is_bbox(bbox):
        # In the observed round-memory frame, the right edge of the color-8
        # switch/door relation sits four pixels to the right of the switch
        # anchor. This keeps the proposal frame-derived instead of using a
        # fixed x-coordinate.
        return int(bbox[2]) - 4, int(start_y)
    x, _y = _centroid_tuple(component)
    return int(round(x)), int(start_y)


def _axis_path(
    start: tuple[int, int],
    target: tuple[int, int],
    stride: int,
) -> list[str]:
    sx, sy = int(start[0]), int(start[1])
    tx, ty = int(target[0]), int(target[1])
    dx = tx - sx
    dy = ty - sy
    if int(stride) <= 0 or dx % int(stride) or dy % int(stride):
        return []
    plan: list[str] = []
    if dx > 0:
        plan.extend(["ACTION4"] * (dx // int(stride)))
    elif dx < 0:
        plan.extend(["ACTION3"] * ((-dx) // int(stride)))
    if dy > 0:
        plan.extend(["ACTION2"] * (dy // int(stride)))
    elif dy < 0:
        plan.extend(["ACTION1"] * ((-dy) // int(stride)))
    return plan


def _plan_steps(actions: Sequence[str], *, role: str) -> list[dict[str, Any]]:
    return [
        {
            "action_name": str(action_name),
            "payload": {},
            "role": str(role),
        }
        for action_name in actions
    ]


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
    "ROUND_MEMORY_DIAGNOSTIC_SENSOR_SCHEMA_VERSION",
    "ROUND_MEMORY_PACKET_SCHEMA_VERSION",
    "ROUND_MEMORY_POSTERIOR_SCHEMA_VERSION",
    "ROUND_MEMORY_PROTECTED_PROPOSAL_SCHEMA_VERSION",
    "build_round_memory_packet",
    "build_round_memory_protected_proposal",
    "summarize_round_memory_posterior",
    "summarize_round_memory_packets",
    "validate_round_memory_packet",
]
