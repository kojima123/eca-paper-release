"""Frame-only multi-body contact topology diagnostics.

The sensor summarizes visible component/contact topology and before/after
contact deltas. It is observational only: no action selection, no planning, and
no ranking authorization.
"""
from __future__ import annotations

import hashlib
from collections import Counter, deque
from typing import Any, Iterable, Mapping

import numpy as np


MULTI_BODY_CONTACT_TOPOLOGY_PACKET_SCHEMA_VERSION = (
    "eca_multi_body_contact_topology_packet.v1"
)
MULTI_BODY_CONTACT_TOPOLOGY_SENSOR_SCHEMA_VERSION = (
    "eca_multi_body_contact_topology_diagnostic_sensor.v1"
)
MULTI_BODY_CONTACT_TOPOLOGY_POSTERIOR_SCHEMA_VERSION = (
    "eca_multi_body_contact_topology_posterior.v1"
)
MULTI_BODY_CONTACT_TOPOLOGY_CANDIDATE_MINER_SCHEMA_VERSION = (
    "eca_multi_body_contact_topology_candidate_miner.v1"
)
MULTI_BODY_CONTACT_TOPOLOGY_GUARDED_PROBE_SCHEMA_VERSION = (
    "eca_multi_body_contact_topology_guarded_probe_outcome_sensor.v1"
)
MULTI_BODY_CONTACT_TOPOLOGY_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION = (
    "eca_multi_body_contact_topology_counterfactual_control_sensor.v1"
)


def build_multi_body_contact_topology_packet(
    frame: Any,
    *,
    before_frame: Any | None = None,
    game_id: str = "",
    state_hash: str = "",
    step: int = 0,
    action_name: str = "",
    score_delta: int = 0,
) -> dict[str, Any]:
    """Build a compact contact-topology packet from public frames."""

    arr = _as_frame(frame)
    before = _as_frame(before_frame) if before_frame is not None else None
    topology = _component_topology(arr)
    before_topology = (
        _component_topology(before)
        if before is not None and before.shape == arr.shape
        else None
    )
    delta = _topology_delta(before_topology, topology)
    moved_count = int(delta.get("moved_component_count", 0))
    contact_edge_delta = int(delta.get("contact_edge_count_delta", 0))
    added_edges = int(delta.get("contact_edges_added_count", 0))
    removed_edges = int(delta.get("contact_edges_removed_count", 0))
    shared_delta_groups = [
        row
        for row in delta.get("shared_delta_groups", [])
        if isinstance(row, Mapping) and int(row.get("component_count", 0)) >= 2
    ]

    return _sanitize(
        {
            "schema_version": MULTI_BODY_CONTACT_TOPOLOGY_PACKET_SCHEMA_VERSION,
            "sensor_family": "multi_body_contact_topology_sensor",
            "game_id": str(game_id or "").strip().lower(),
            "state_hash": str(state_hash or _frame_hash(arr)),
            "step": int(step),
            "action_name": str(action_name or ""),
            "score_delta": int(score_delta),
            "background_color": topology["background_color"],
            "component_count": topology["component_count"],
            "component_color_histogram": topology["component_color_histogram"],
            "component_area_histogram": topology["component_area_histogram"],
            "contact_edge_count": topology["contact_edge_count"],
            "contact_color_pair_histogram": topology["contact_color_pair_histogram"],
            "components": topology["components"][:16],
            "contact_edges": topology["contact_edges"][:16],
            "contact_topology_delta": delta,
            "multi_body_state_candidate": int(topology["component_count"]) >= 2,
            "contact_event_candidate": bool(added_edges or removed_edges),
            "multi_component_motion_candidate": moved_count >= 2,
            "shared_motion_coupling_candidate": bool(shared_delta_groups),
            "collision_transfer_candidate": bool((added_edges or removed_edges) and moved_count),
            "score_closure_candidate": bool(
                int(score_delta) > 0 and (added_edges or removed_edges or moved_count >= 2)
            ),
            "diagnostic_only": True,
        }
    )


def validate_multi_body_contact_topology_packet(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if (
        str(packet.get("schema_version", "") or "")
        != MULTI_BODY_CONTACT_TOPOLOGY_PACKET_SCHEMA_VERSION
    ):
        reasons.append("schema_version_mismatch")
    if str(packet.get("sensor_family", "") or "") != (
        "multi_body_contact_topology_sensor"
    ):
        reasons.append("sensor_family_mismatch")
    component_count = _safe_int(packet.get("component_count"))
    if component_count <= 0:
        reasons.append("missing_visible_components")
    return {
        "schema_version": "eca_multi_body_contact_topology_validation.v1",
        "accepted": not reasons,
        "rejected_reasons": reasons,
        "family": "multi_body_contact_topology",
        "mechanic_state_type": _mechanic_state_type(packet),
        "component_count": component_count,
        "contact_edge_count": _safe_int(packet.get("contact_edge_count")),
    }


def summarize_multi_body_contact_topology_packets(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    color_pair_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    moved_sum = 0
    score_delta_sum = 0

    for packet in packet_rows:
        validation = validate_multi_body_contact_topology_packet(packet)
        accepted = bool(validation.get("accepted"))
        counts["accepted_packet_count" if accepted else "rejected_packet_count"] += 1
        mechanic = str(validation.get("mechanic_state_type", "") or "unknown")
        counts[f"mechanic_state:{mechanic}"] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["packet_count"] += 1
        by_game[game_id][f"mechanic_state:{mechanic}"] += 1
        delta = packet.get("contact_topology_delta")
        delta = delta if isinstance(delta, Mapping) else {}
        moved = _safe_int(delta.get("moved_component_count"))
        moved_sum += moved
        score_delta = _safe_int(packet.get("score_delta"))
        score_delta_sum += score_delta
        if bool(packet.get("multi_body_state_candidate")):
            counts["multi_body_state_candidate_count"] += 1
        if bool(packet.get("contact_event_candidate")):
            counts["contact_event_candidate_count"] += 1
            by_game[game_id]["contact_event_candidate_count"] += 1
        if bool(packet.get("multi_component_motion_candidate")):
            counts["multi_component_motion_candidate_count"] += 1
        if bool(packet.get("shared_motion_coupling_candidate")):
            counts["shared_motion_coupling_candidate_count"] += 1
        if bool(packet.get("collision_transfer_candidate")):
            counts["collision_transfer_candidate_count"] += 1
        if bool(packet.get("score_closure_candidate")):
            counts["score_closure_candidate_count"] += 1
        for pair, value in (packet.get("contact_color_pair_histogram") or {}).items():
            color_pair_counts[str(pair)] += _safe_int(value)
        if len(samples) < sample_limit:
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "action_name": str(packet.get("action_name", "") or ""),
                    "mechanic_state_type": mechanic,
                    "component_count": _safe_int(packet.get("component_count")),
                    "contact_edge_count": _safe_int(packet.get("contact_edge_count")),
                    "moved_component_count": moved,
                    "contact_edges_added_count": _safe_int(
                        delta.get("contact_edges_added_count")
                    ),
                    "contact_edges_removed_count": _safe_int(
                        delta.get("contact_edges_removed_count")
                    ),
                    "score_delta": score_delta,
                }
            )

    return {
        "schema_version": MULTI_BODY_CONTACT_TOPOLOGY_SENSOR_SCHEMA_VERSION,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "packet_count": int(len(packet_rows)),
        "accepted_packet_count": int(counts.get("accepted_packet_count", 0)),
        "rejected_packet_count": int(counts.get("rejected_packet_count", 0)),
        "multi_body_state_candidate_count": int(
            counts.get("multi_body_state_candidate_count", 0)
        ),
        "contact_event_candidate_count": int(
            counts.get("contact_event_candidate_count", 0)
        ),
        "multi_component_motion_candidate_count": int(
            counts.get("multi_component_motion_candidate_count", 0)
        ),
        "shared_motion_coupling_candidate_count": int(
            counts.get("shared_motion_coupling_candidate_count", 0)
        ),
        "collision_transfer_candidate_count": int(
            counts.get("collision_transfer_candidate_count", 0)
        ),
        "score_closure_candidate_count": int(
            counts.get("score_closure_candidate_count", 0)
        ),
        "moved_component_count_sum": int(moved_sum),
        "score_delta_sum": int(score_delta_sum),
        "mechanic_state_counts": {
            key.split(":", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("mechanic_state:")
        },
        "contact_color_pair_histogram": dict(sorted(color_pair_counts.items())),
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "contact topology posterior with score/control margin",
            "allowed_connection": "diagnostic_only_until_contact_posterior_controls",
            "action_generation_allowed": False,
        },
    }


def summarize_multi_body_contact_topology_posterior(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_action: dict[str, Counter[str]] = {}
    samples: list[dict[str, Any]] = []
    contact_signal_score_delta_sum = 0
    shared_motion_score_delta_sum = 0
    collision_transfer_score_delta_sum = 0
    score_closure_signal_score_delta_sum = 0
    no_contact_control_score_delta_sum = 0
    static_control_score_delta_sum = 0
    score_delta_sum = 0

    for packet in packet_rows:
        validation = validate_multi_body_contact_topology_packet(packet)
        accepted = bool(validation.get("accepted"))
        counts["accepted_packet_count" if accepted else "rejected_packet_count"] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        action_name = str(packet.get("action_name", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_action.setdefault(action_name, Counter())
        by_game[game_id]["packet_count"] += 1
        by_action[action_name]["packet_count"] += 1
        score_delta = _safe_int(packet.get("score_delta"))
        score_delta_sum += score_delta
        contact_event = bool(packet.get("contact_event_candidate"))
        shared_motion = bool(packet.get("shared_motion_coupling_candidate"))
        collision_transfer = bool(packet.get("collision_transfer_candidate"))
        score_closure = bool(packet.get("score_closure_candidate"))
        multi_motion = bool(packet.get("multi_component_motion_candidate"))
        contact_signal = bool(contact_event or shared_motion or collision_transfer)
        mechanic = str(validation.get("mechanic_state_type", "") or "unknown")
        if contact_event:
            counts["contact_event_candidate_count"] += 1
        if shared_motion:
            counts["shared_motion_coupling_candidate_count"] += 1
        if collision_transfer:
            counts["collision_transfer_candidate_count"] += 1

        if not accepted:
            state = "rejected_descriptor"
        elif score_closure and contact_signal:
            state = "contact_topology_score_closure_confirmed"
            counts["confirmed_score_closure_count"] += 1
            score_closure_signal_score_delta_sum += max(0, score_delta)
        elif collision_transfer:
            state = "collision_transfer_scoreless_or_unconfirmed"
        elif shared_motion:
            state = "shared_motion_scoreless_or_unconfirmed"
        elif contact_event:
            state = "contact_event_scoreless_or_unconfirmed"
        elif multi_motion:
            state = "multi_component_motion_control"
            no_contact_control_score_delta_sum += max(0, score_delta)
            counts["multi_component_motion_control_count"] += 1
            counts["negative_control_packet_count"] += 1
        else:
            state = "static_or_no_contact_control"
            no_contact_control_score_delta_sum += max(0, score_delta)
            static_control_score_delta_sum += max(0, score_delta)
            counts["static_or_no_contact_control_count"] += 1
            counts["negative_control_packet_count"] += 1

        if contact_signal:
            counts["contact_signal_packet_count"] += 1
            contact_signal_score_delta_sum += max(0, score_delta)
        if shared_motion:
            shared_motion_score_delta_sum += max(0, score_delta)
        if collision_transfer:
            collision_transfer_score_delta_sum += max(0, score_delta)
        if score_delta > 0:
            counts["score_positive_packet_count"] += 1
            by_game[game_id]["score_positive_packet_count"] += 1
            by_action[action_name]["score_positive_packet_count"] += 1
        counts[f"posterior_state:{state}"] += 1
        by_game[game_id][f"posterior_state:{state}"] += 1
        by_action[action_name][f"posterior_state:{state}"] += 1
        if contact_signal:
            by_game[game_id]["contact_signal_packet_count"] += 1
            by_action[action_name]["contact_signal_packet_count"] += 1
        if len(samples) < sample_limit:
            delta = packet.get("contact_topology_delta")
            delta = delta if isinstance(delta, Mapping) else {}
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "action_name": action_name,
                    "posterior_state": state,
                    "mechanic_state_type": mechanic,
                    "score_delta": score_delta,
                    "contact_event_candidate": contact_event,
                    "shared_motion_coupling_candidate": shared_motion,
                    "collision_transfer_candidate": collision_transfer,
                    "score_closure_candidate": score_closure,
                    "moved_component_count": _safe_int(
                        delta.get("moved_component_count")
                    ),
                    "contact_edges_added_count": _safe_int(
                        delta.get("contact_edges_added_count")
                    ),
                    "contact_edges_removed_count": _safe_int(
                        delta.get("contact_edges_removed_count")
                    ),
                }
            )

    negative_control = max(no_contact_control_score_delta_sum, static_control_score_delta_sum)
    control_margin = int(score_closure_signal_score_delta_sum - negative_control)
    has_control = int(counts.get("negative_control_packet_count", 0)) > 0
    if (
        counts.get("confirmed_score_closure_count", 0) > 0
        and has_control
        and control_margin > 0
    ):
        promotion_state = "contact_topology_positive_control_margin"
    elif score_closure_signal_score_delta_sum > 0:
        promotion_state = "contact_topology_score_closure_control_matched"
    elif counts.get("contact_signal_packet_count", 0) > 0:
        promotion_state = "contact_topology_signal_scoreless_unconfirmed"
    else:
        promotion_state = "no_contact_topology_signal_evidence"

    return {
        "schema_version": MULTI_BODY_CONTACT_TOPOLOGY_POSTERIOR_SCHEMA_VERSION,
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
        "contact_signal_packet_count": int(counts.get("contact_signal_packet_count", 0)),
        "contact_event_candidate_count": int(
            counts.get("contact_event_candidate_count", 0)
        ),
        "shared_motion_coupling_candidate_count": int(
            counts.get("shared_motion_coupling_candidate_count", 0)
        ),
        "collision_transfer_candidate_count": int(
            counts.get("collision_transfer_candidate_count", 0)
        ),
        "confirmed_score_closure_count": int(
            counts.get("confirmed_score_closure_count", 0)
        ),
        "negative_control_packet_count": int(
            counts.get("negative_control_packet_count", 0)
        ),
        "contact_signal_score_delta_sum": int(contact_signal_score_delta_sum),
        "shared_motion_score_delta_sum": int(shared_motion_score_delta_sum),
        "collision_transfer_score_delta_sum": int(collision_transfer_score_delta_sum),
        "score_closure_signal_score_delta_sum": int(
            score_closure_signal_score_delta_sum
        ),
        "no_contact_negative_control_score_delta_sum": int(
            no_contact_control_score_delta_sum
        ),
        "static_negative_control_score_delta_sum": int(static_control_score_delta_sum),
        "negative_control_max_score_delta_sum": int(negative_control),
        "contact_topology_counterfactual_margin": int(control_margin),
        "positive_control_margin_count": int(has_control and control_margin > 0),
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
            "allowed_connection": "diagnostic_only_until_contact_topology_counterfactual_margin",
            "action_generation_allowed": False,
        },
    }


def summarize_multi_body_contact_topology_candidates(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    candidates: list[dict[str, Any]] = []
    by_game: dict[str, Counter[str]] = {}
    by_action: dict[str, Counter[str]] = {}
    accepted_packet_count = 0
    rejected_packet_count = 0
    confirmed_closure_packet_count = 0

    for packet in packet_rows:
        validation = validate_multi_body_contact_topology_packet(packet)
        if not bool(validation.get("accepted")):
            rejected_packet_count += 1
            continue
        accepted_packet_count += 1
        score_delta = _safe_int(packet.get("score_delta"))
        contact_signal = bool(
            packet.get("contact_event_candidate")
            or packet.get("shared_motion_coupling_candidate")
            or packet.get("collision_transfer_candidate")
        )
        if not bool(packet.get("score_closure_candidate")):
            continue
        if score_delta <= 0 or not contact_signal:
            continue
        confirmed_closure_packet_count += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        action_name = _normalize_action_name(packet.get("action_name", ""))
        if not action_name:
            continue
        delta = packet.get("contact_topology_delta")
        delta = delta if isinstance(delta, Mapping) else {}
        signature = _contact_topology_signature(packet)
        candidate_key = (
            f"contact_topology_replay:{game_id}:{action_name}:"
            f"{signature}:step{_safe_int(packet.get('step'))}"
        )
        by_game.setdefault(game_id, Counter())
        by_action.setdefault(action_name, Counter())
        by_game[game_id]["candidate_count"] += 1
        by_game[game_id][f"source_action:{action_name}"] += 1
        by_action[action_name]["candidate_count"] += 1
        candidates.append(
            {
                "candidate_key": candidate_key,
                "candidate_kind": "contact_topology_source_action_replay_probe",
                "game_id": game_id,
                "step": _safe_int(packet.get("step")),
                "source_action_name": action_name,
                "replay_action_names": [action_name],
                "contact_topology_signature": signature,
                "evidence": {
                    "mechanic_state_type": str(
                        validation.get("mechanic_state_type", "")
                    ),
                    "score_delta": score_delta,
                    "contact_event_candidate": bool(
                        packet.get("contact_event_candidate")
                    ),
                    "shared_motion_coupling_candidate": bool(
                        packet.get("shared_motion_coupling_candidate")
                    ),
                    "collision_transfer_candidate": bool(
                        packet.get("collision_transfer_candidate")
                    ),
                    "moved_component_count": _safe_int(
                        delta.get("moved_component_count")
                    ),
                    "contact_edges_added_count": _safe_int(
                        delta.get("contact_edges_added_count")
                    ),
                    "contact_edges_removed_count": _safe_int(
                        delta.get("contact_edges_removed_count")
                    ),
                    "contact_color_pair_histogram": dict(
                        packet.get("contact_color_pair_histogram") or {}
                    ),
                },
                "runtime_execution_enabled": False,
                "requires_guarded_probe": True,
                "requires_positive_control_margin": True,
            }
        )

    return {
        "schema_version": MULTI_BODY_CONTACT_TOPOLOGY_CANDIDATE_MINER_SCHEMA_VERSION,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "packet_count": int(len(packet_rows)),
        "accepted_packet_count": int(accepted_packet_count),
        "rejected_packet_count": int(rejected_packet_count),
        "confirmed_closure_packet_count": int(confirmed_closure_packet_count),
        "candidate_count": int(len(candidates)),
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "by_action": {
            action: dict(sorted(counter.items()))
            for action, counter in sorted(by_action.items())
        },
        "candidates": candidates[:sample_limit],
        "next_runtime_contract": {
            "required_before_execution": "default-off guarded contact-topology replay probe with paired controls",
            "allowed_connection": "candidate_design_only_until_runtime_probe_enabled",
            "action_generation_allowed": False,
        },
    }


def summarize_multi_body_contact_topology_guarded_probe_outcomes(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize default-off guarded-probe admission for contact candidates.

    This does not execute an action. It only says whether existing
    contact-topology candidate evidence has the posterior controls required to
    become a protected probe in a later runtime phase.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    posterior = summarize_multi_body_contact_topology_posterior(packet_rows)
    miner = summarize_multi_body_contact_topology_candidates(packet_rows)
    has_positive_margin = _safe_int(posterior.get("positive_control_margin_count")) > 0
    reason_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_action: dict[str, Counter[str]] = {}
    admitted: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for candidate in list(miner.get("candidates") or []):
        if not isinstance(candidate, Mapping):
            continue
        game_id = str(candidate.get("game_id", "") or "unknown")
        action = _normalize_action_name(candidate.get("source_action_name", ""))
        if not action:
            action = "unknown"
        by_game.setdefault(game_id, Counter())
        by_action.setdefault(action, Counter())
        by_game[game_id]["candidate_count"] += 1
        by_action[action]["candidate_count"] += 1
        reasons: list[str] = []
        if not has_positive_margin:
            reasons.append("missing_positive_control_margin")
        if str(candidate.get("candidate_kind", "") or "") != (
            "contact_topology_source_action_replay_probe"
        ):
            reasons.append("unsupported_candidate_kind")
        if not _normalize_action_name(candidate.get("source_action_name", "")):
            reasons.append("missing_source_action")
        signature = str(candidate.get("contact_topology_signature", "") or "")
        if not signature:
            reasons.append("missing_contact_topology_signature")

        if reasons:
            reason = reasons[0]
            reason_counts[reason] += 1
            by_game[game_id][f"blocked_reason:{reason}"] += 1
            by_action[action][f"blocked_reason:{reason}"] += 1
            if len(blocked) < sample_limit:
                blocked.append(
                    {
                        "candidate_key": str(candidate.get("candidate_key", "") or ""),
                        "game_id": game_id,
                        "source_action_name": action,
                        "blocked_reason": reason,
                        "blocked_reasons": reasons,
                        "contact_topology_signature": signature,
                    }
                )
            continue

        reason_counts["admitted_guarded_contact_topology_probe_design"] += 1
        by_game[game_id]["admitted_guarded_probe_count"] += 1
        by_action[action]["admitted_guarded_probe_count"] += 1
        if len(admitted) < sample_limit:
            admitted.append(
                {
                    "candidate_key": str(candidate.get("candidate_key", "") or ""),
                    "game_id": game_id,
                    "source_action_name": action,
                    "replay_action_names": list(candidate.get("replay_action_names") or []),
                    "contact_topology_signature": signature,
                    "guard_state": "admitted_guarded_probe_design",
                    "runtime_execution_enabled": False,
                    "requires_current_signature_match": True,
                    "requires_state_hash_match": True,
                }
            )

    admitted_count = int(
        reason_counts.get("admitted_guarded_contact_topology_probe_design", 0)
    )
    blocked_count = sum(
        int(value)
        for key, value in reason_counts.items()
        if key != "admitted_guarded_contact_topology_probe_design"
    )
    return {
        "schema_version": MULTI_BODY_CONTACT_TOPOLOGY_GUARDED_PROBE_SCHEMA_VERSION,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "packet_count": int(len(packet_rows)),
        "candidate_count": _safe_int(miner.get("candidate_count")),
        "positive_control_margin_count": _safe_int(
            posterior.get("positive_control_margin_count")
        ),
        "contact_topology_counterfactual_margin": _safe_int(
            posterior.get("contact_topology_counterfactual_margin")
        ),
        "admitted_guarded_probe_count": admitted_count,
        "blocked_guarded_probe_count": int(blocked_count),
        "reason_counts": dict(sorted(reason_counts.items())),
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "by_action": {
            action: dict(sorted(counter.items()))
            for action, counter in sorted(by_action.items())
        },
        "guarded_probe_plans": admitted,
        "blocked_samples": blocked,
        "next_runtime_contract": {
            "required_before_execution": "runtime executor must re-check current state hash and contact_topology_signature before one-shot protected probe",
            "allowed_connection": "guarded_probe_design_only_until_runtime_execution_phase",
            "action_generation_allowed": False,
        },
    }


def summarize_multi_body_contact_topology_counterfactual_controls(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize counterfactual controls for contact-topology probe designs.

    The control is observational: a contact-topology score-closure candidate is
    considered supported only when the same game also contains scoreless
    no-contact/static evidence. This keeps the signal in the
    sensor->posterior->guard lane and does not execute or rank actions.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    posterior = summarize_multi_body_contact_topology_posterior(packet_rows)
    miner = summarize_multi_body_contact_topology_candidates(
        packet_rows,
        sample_limit=max(sample_limit, len(packet_rows), 16),
    )
    guarded = summarize_multi_body_contact_topology_guarded_probe_outcomes(
        packet_rows,
        sample_limit=max(sample_limit, len(packet_rows), 16),
    )

    admitted_keys = {
        str(plan.get("candidate_key", "") or "")
        for plan in list(guarded.get("guarded_probe_plans") or [])
        if isinstance(plan, Mapping)
    }
    negative_controls_by_game: Counter[str] = Counter()
    negative_controls_by_game_action: Counter[str] = Counter()
    scoreless_contact_by_signature: Counter[str] = Counter()
    accepted_packet_count = 0
    rejected_packet_count = 0

    for packet in packet_rows:
        validation = validate_multi_body_contact_topology_packet(packet)
        if not bool(validation.get("accepted")):
            rejected_packet_count += 1
            continue
        accepted_packet_count += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        action = _normalize_action_name(packet.get("action_name", "")) or "unknown"
        score_delta = _safe_int(packet.get("score_delta"))
        contact_signal = bool(
            packet.get("contact_event_candidate")
            or packet.get("shared_motion_coupling_candidate")
            or packet.get("collision_transfer_candidate")
        )
        if score_delta <= 0 and not contact_signal:
            negative_controls_by_game[game_id] += 1
            negative_controls_by_game_action[f"{game_id}|{action}"] += 1
        if score_delta <= 0 and contact_signal:
            signature = _contact_topology_signature(packet)
            scoreless_contact_by_signature[f"{game_id}|{signature}"] += 1

    has_positive_margin = _safe_int(posterior.get("positive_control_margin_count")) > 0
    reason_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_action: dict[str, Counter[str]] = {}
    supported_samples: list[dict[str, Any]] = []
    blocked_samples: list[dict[str, Any]] = []
    counterfactual_control_pair_count = 0
    same_action_negative_control_pair_count = 0
    game_negative_control_pair_count = 0
    same_signature_scoreless_contact_control_count = 0

    for candidate in list(miner.get("candidates") or []):
        if not isinstance(candidate, Mapping):
            continue
        game_id = str(candidate.get("game_id", "") or "unknown")
        action = _normalize_action_name(candidate.get("source_action_name", "")) or "unknown"
        candidate_key = str(candidate.get("candidate_key", "") or "")
        signature = str(candidate.get("contact_topology_signature", "") or "")
        by_game.setdefault(game_id, Counter())
        by_action.setdefault(action, Counter())
        by_game[game_id]["candidate_count"] += 1
        by_action[action]["candidate_count"] += 1

        same_action_controls = int(
            negative_controls_by_game_action.get(f"{game_id}|{action}", 0)
        )
        game_controls = int(negative_controls_by_game.get(game_id, 0))
        same_signature_scoreless = int(
            scoreless_contact_by_signature.get(f"{game_id}|{signature}", 0)
        )
        if same_signature_scoreless > 0:
            same_signature_scoreless_contact_control_count += 1

        reasons: list[str] = []
        if candidate_key not in admitted_keys:
            reasons.append("missing_guarded_probe_admission")
        if not has_positive_margin:
            reasons.append("missing_positive_control_margin")
        if game_controls <= 0:
            reasons.append("missing_same_game_negative_control")

        if reasons:
            reason = reasons[0]
            reason_counts[reason] += 1
            by_game[game_id][f"blocked_reason:{reason}"] += 1
            by_action[action][f"blocked_reason:{reason}"] += 1
            if len(blocked_samples) < sample_limit:
                blocked_samples.append(
                    {
                        "candidate_key": candidate_key,
                        "game_id": game_id,
                        "source_action_name": action,
                        "blocked_reason": reason,
                        "blocked_reasons": reasons,
                        "same_game_negative_control_count": game_controls,
                        "same_action_negative_control_count": same_action_controls,
                        "same_signature_scoreless_contact_control_count": (
                            same_signature_scoreless
                        ),
                        "contact_topology_signature": signature,
                    }
                )
            continue

        counterfactual_control_pair_count += 1
        if same_action_controls > 0:
            same_action_negative_control_pair_count += 1
            control_kind = "same_action_negative_control"
        else:
            game_negative_control_pair_count += 1
            control_kind = "same_game_negative_control"
        reason_counts["counterfactual_control_supported"] += 1
        by_game[game_id]["counterfactual_control_supported_count"] += 1
        by_action[action]["counterfactual_control_supported_count"] += 1
        if len(supported_samples) < sample_limit:
            supported_samples.append(
                {
                    "candidate_key": candidate_key,
                    "game_id": game_id,
                    "source_action_name": action,
                    "control_kind": control_kind,
                    "same_game_negative_control_count": game_controls,
                    "same_action_negative_control_count": same_action_controls,
                    "same_signature_scoreless_contact_control_count": (
                        same_signature_scoreless
                    ),
                    "contact_topology_signature": signature,
                    "runtime_execution_enabled": False,
                }
            )

    if counterfactual_control_pair_count > 0 and has_positive_margin:
        promotion_state = "contact_topology_counterfactual_control_supported"
    elif _safe_int(guarded.get("admitted_guarded_probe_count")) > 0:
        promotion_state = "guarded_probe_without_counterfactual_control"
    elif _safe_int(miner.get("candidate_count")) > 0:
        promotion_state = "candidate_without_guarded_control"
    else:
        promotion_state = "no_contact_topology_control_candidate"

    return {
        "schema_version": MULTI_BODY_CONTACT_TOPOLOGY_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "packet_count": int(len(packet_rows)),
        "accepted_packet_count": int(accepted_packet_count),
        "rejected_packet_count": int(rejected_packet_count),
        "candidate_count": _safe_int(miner.get("candidate_count")),
        "admitted_guarded_probe_count": _safe_int(
            guarded.get("admitted_guarded_probe_count")
        ),
        "positive_control_margin_count": _safe_int(
            posterior.get("positive_control_margin_count")
        ),
        "contact_topology_counterfactual_margin": _safe_int(
            posterior.get("contact_topology_counterfactual_margin")
        ),
        "counterfactual_control_pair_count": int(counterfactual_control_pair_count),
        "same_action_negative_control_pair_count": int(
            same_action_negative_control_pair_count
        ),
        "same_game_negative_control_pair_count": int(game_negative_control_pair_count),
        "same_signature_scoreless_contact_control_count": int(
            same_signature_scoreless_contact_control_count
        ),
        "blocked_candidate_count": sum(
            int(value)
            for key, value in reason_counts.items()
            if key != "counterfactual_control_supported"
        ),
        "negative_control_packet_count": int(sum(negative_controls_by_game.values())),
        "promotion_state": promotion_state,
        "reason_counts": dict(sorted(reason_counts.items())),
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "by_action": {
            action: dict(sorted(counter.items()))
            for action, counter in sorted(by_action.items())
        },
        "supported_samples": supported_samples,
        "blocked_samples": blocked_samples,
        "next_runtime_contract": {
            "required_before_execution": "focused runtime guarded probe plus breadth/all25 preservation",
            "allowed_connection": "counterfactual_control_diagnostic_only_until_runtime_guard_phase",
            "action_generation_allowed": False,
        },
    }


def _component_topology(arr: np.ndarray) -> dict[str, Any]:
    background = _background_color(arr)
    components = _connected_components(arr, background_color=background)
    label = np.full(arr.shape, -1, dtype=int)
    for component in components:
        for x, y in component["cells"]:
            label[int(y), int(x)] = int(component["id"])
    edges = _contact_edges(arr, label, components)
    color_hist = Counter(str(component["color"]) for component in components)
    area_hist = Counter(str(component["area"]) for component in components)
    pair_hist = Counter(str(edge["color_pair"]) for edge in edges)
    public_components = [
        {key: value for key, value in component.items() if key != "cells"}
        for component in components
    ]
    return {
        "background_color": int(background),
        "component_count": int(len(components)),
        "component_color_histogram": dict(sorted(color_hist.items())),
        "component_area_histogram": dict(sorted(area_hist.items())),
        "contact_edge_count": int(len(edges)),
        "contact_color_pair_histogram": dict(sorted(pair_hist.items())),
        "components": public_components,
        "contact_edges": edges,
    }


def _connected_components(arr: np.ndarray, *, background_color: int) -> list[dict[str, Any]]:
    h, w = arr.shape
    visited = np.zeros(arr.shape, dtype=bool)
    components: list[dict[str, Any]] = []
    for y in range(h):
        for x in range(w):
            color = int(arr[y, x])
            if visited[y, x] or color == background_color:
                continue
            visited[y, x] = True
            queue: deque[tuple[int, int]] = deque([(x, y)])
            cells: list[tuple[int, int]] = []
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue
                    if visited[ny, nx] or int(arr[ny, nx]) != color:
                        continue
                    visited[ny, nx] = True
                    queue.append((nx, ny))
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            area = len(cells)
            components.append(
                {
                    "id": len(components),
                    "color": int(color),
                    "area": int(area),
                    "bbox": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
                    "centroid": [
                        round(sum(xs) / max(1, area), 3),
                        round(sum(ys) / max(1, area), 3),
                    ],
                    "cells": cells,
                }
            )
    return components


def _contact_edges(
    arr: np.ndarray,
    label: np.ndarray,
    components: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    edges: dict[tuple[int, int], dict[str, Any]] = {}
    h, w = arr.shape
    for y in range(h):
        for x in range(w):
            a = int(label[y, x])
            if a < 0:
                continue
            for nx, ny in ((x + 1, y), (x, y + 1)):
                if nx >= w or ny >= h:
                    continue
                b = int(label[ny, nx])
                if b < 0 or b == a:
                    continue
                key = tuple(sorted((a, b)))
                ca = int(components[key[0]]["color"])
                cb = int(components[key[1]]["color"])
                color_pair = tuple(sorted((ca, cb)))
                edge = edges.setdefault(
                    key,
                    {
                        "component_ids": [int(key[0]), int(key[1])],
                        "color_pair": f"{color_pair[0]}:{color_pair[1]}",
                        "contact_cell_count": 0,
                    },
                )
                edge["contact_cell_count"] += 1
    return [dict(edge) for _key, edge in sorted(edges.items())]


def _topology_delta(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any],
) -> dict[str, Any]:
    if before is None:
        return {
            "observed": False,
            "component_count_delta": 0,
            "contact_edge_count_delta": 0,
            "contact_edges_added_count": 0,
            "contact_edges_removed_count": 0,
            "moved_component_count": 0,
            "moved_components": [],
            "shared_delta_groups": [],
        }
    before_edges = _color_pair_edge_set(before)
    after_edges = _color_pair_edge_set(after)
    moved = _moved_components(before, after)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in moved:
        dx, dy = row.get("delta", [0, 0])
        if float(dx) == 0.0 and float(dy) == 0.0:
            continue
        groups.setdefault(f"{dx}:{dy}", []).append(row)
    shared_groups = [
        {
            "delta": [float(key.split(":")[0]), float(key.split(":")[1])],
            "component_count": len(rows),
            "component_ids": [row["after_component_id"] for row in rows],
            "colors": sorted({int(row["color"]) for row in rows}),
        }
        for key, rows in sorted(groups.items())
        if len(rows) >= 2
    ]
    return {
        "observed": True,
        "component_count_delta": int(after["component_count"]) - int(before["component_count"]),
        "contact_edge_count_delta": int(after["contact_edge_count"])
        - int(before["contact_edge_count"]),
        "contact_edges_added_count": len(after_edges - before_edges),
        "contact_edges_removed_count": len(before_edges - after_edges),
        "contact_edges_added": sorted(after_edges - before_edges)[:16],
        "contact_edges_removed": sorted(before_edges - after_edges)[:16],
        "moved_component_count": len(moved),
        "moved_components": moved[:16],
        "shared_delta_groups": shared_groups[:16],
    }


def _moved_components(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[dict[str, Any]]:
    before_components = [
        c for c in before.get("components", []) if isinstance(c, Mapping)
    ]
    after_components = [c for c in after.get("components", []) if isinstance(c, Mapping)]
    used_before: set[int] = set()
    moved: list[dict[str, Any]] = []
    for after_component in after_components:
        color = _safe_int(after_component.get("color"))
        best_index = -1
        best_distance = float("inf")
        ax, ay = _centroid(after_component)
        for index, before_component in enumerate(before_components):
            if index in used_before or _safe_int(before_component.get("color")) != color:
                continue
            bx, by = _centroid(before_component)
            distance = (ax - bx) ** 2 + (ay - by) ** 2
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index < 0:
            continue
        used_before.add(best_index)
        before_component = before_components[best_index]
        bx, by = _centroid(before_component)
        dx = round(ax - bx, 3)
        dy = round(ay - by, 3)
        if dx == 0 and dy == 0 and before_component.get("bbox") == after_component.get("bbox"):
            continue
        moved.append(
            {
                "before_component_id": _safe_int(before_component.get("id")),
                "after_component_id": _safe_int(after_component.get("id")),
                "color": int(color),
                "area_before": _safe_int(before_component.get("area")),
                "area_after": _safe_int(after_component.get("area")),
                "delta": [dx, dy],
            }
        )
    return moved


def _color_pair_edge_set(topology: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for edge in topology.get("contact_edges", []):
        if not isinstance(edge, Mapping):
            continue
        pair = str(edge.get("color_pair", "") or "")
        if pair:
            out.add(pair)
    return out


def _mechanic_state_type(packet: Mapping[str, Any]) -> str:
    if bool(packet.get("score_closure_candidate")):
        return "score_closure_contact_topology"
    if bool(packet.get("collision_transfer_candidate")):
        return "collision_or_contact_transfer"
    if bool(packet.get("shared_motion_coupling_candidate")):
        return "shared_motion_coupling"
    if bool(packet.get("multi_component_motion_candidate")):
        return "multi_component_motion"
    if bool(packet.get("contact_event_candidate")):
        return "contact_topology_transition"
    if bool(packet.get("multi_body_state_candidate")):
        return "multi_body_static_topology"
    return "insufficient_multi_body_evidence"


def _contact_topology_signature(packet: Mapping[str, Any]) -> str:
    delta = packet.get("contact_topology_delta")
    delta = delta if isinstance(delta, Mapping) else {}
    added = _safe_int(delta.get("contact_edges_added_count"))
    removed = _safe_int(delta.get("contact_edges_removed_count"))
    moved = _safe_int(delta.get("moved_component_count"))
    shared = len(
        [
            row
            for row in list(delta.get("shared_delta_groups") or [])
            if isinstance(row, Mapping)
        ]
    )
    pairs = ",".join(
        f"{key}:{_safe_int(value)}"
        for key, value in sorted((packet.get("contact_color_pair_histogram") or {}).items())
    )
    return f"added{added}|removed{removed}|moved{moved}|shared{shared}|pairs:{pairs}"


def _normalize_action_name(value: Any) -> str:
    raw = value.name if hasattr(value, "name") else value
    action = str(raw or "").strip().upper()
    if action.startswith("GAMEACTION."):
        action = action.split(".", 1)[1]
    if action in {f"ACTION{i}" for i in range(1, 7)}:
        return action
    return ""


def _as_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError("expected 2D frame or animation stack")
    return arr.astype(int, copy=False)


def _background_color(arr: np.ndarray) -> int:
    values, counts = np.unique(arr, return_counts=True)
    if len(values) == 0:
        return 0
    return int(values[int(np.argmax(counts))])


def _centroid(component: Mapping[str, Any]) -> tuple[float, float]:
    raw = component.get("centroid")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return float(raw[0]), float(raw[1])
    bbox = component.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return (float(bbox[0]) + float(bbox[2])) / 2.0, (
            float(bbox[1]) + float(bbox[3])
        ) / 2.0
    return 0.0, 0.0


def _frame_hash(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "MULTI_BODY_CONTACT_TOPOLOGY_CANDIDATE_MINER_SCHEMA_VERSION",
    "MULTI_BODY_CONTACT_TOPOLOGY_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION",
    "MULTI_BODY_CONTACT_TOPOLOGY_GUARDED_PROBE_SCHEMA_VERSION",
    "MULTI_BODY_CONTACT_TOPOLOGY_PACKET_SCHEMA_VERSION",
    "MULTI_BODY_CONTACT_TOPOLOGY_POSTERIOR_SCHEMA_VERSION",
    "MULTI_BODY_CONTACT_TOPOLOGY_SENSOR_SCHEMA_VERSION",
    "build_multi_body_contact_topology_packet",
    "summarize_multi_body_contact_topology_candidates",
    "summarize_multi_body_contact_topology_counterfactual_controls",
    "summarize_multi_body_contact_topology_guarded_probe_outcomes",
    "summarize_multi_body_contact_topology_posterior",
    "summarize_multi_body_contact_topology_packets",
    "validate_multi_body_contact_topology_packet",
]
