"""Frame-only destructible platform topology diagnostics.

The sensor observes click-to-terrain deltas, player motion, hazard proximity,
and goal-distance changes from public before/after frames. It does not plan a
platform route or authorize candidate ranking.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PLATFORM_DESTRUCTIBLE_TOPOLOGY_PACKET_SCHEMA_VERSION = (
    "eca_platform_destructible_topology_packet.v1"
)
PLATFORM_DESTRUCTIBLE_TOPOLOGY_DIAGNOSTIC_SENSOR_SCHEMA_VERSION = (
    "eca_platform_destructible_topology_diagnostic_sensor.v1"
)
PLATFORM_DESTRUCTIBLE_TOPOLOGY_POSTERIOR_SCHEMA_VERSION = (
    "eca_platform_destructible_topology_posterior.v1"
)
PLATFORM_DESTRUCTIBLE_TOPOLOGY_CANDIDATE_MINER_SCHEMA_VERSION = (
    "eca_platform_destructible_topology_candidate_miner.v1"
)
PLATFORM_DESTRUCTIBLE_TOPOLOGY_GUARDED_PROBE_OUTCOME_SCHEMA_VERSION = (
    "eca_platform_destructible_topology_guarded_probe_outcome.v1"
)
PLATFORM_DESTRUCTIBLE_TOPOLOGY_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION = (
    "eca_platform_destructible_topology_counterfactual_control.v1"
)
PLATFORM_DESTRUCTIBLE_TOPOLOGY_ALL25_NON_REGRESSION_SCHEMA_VERSION = (
    "eca_platform_destructible_topology_all25_non_regression.v1"
)


def build_platform_destructible_topology_packet(
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
    """Build an observational destructible-platform topology packet."""

    before = _as_frame(before_frame)
    after = _as_frame(after_frame) if after_frame is not None else before
    action_payload = dict(action_payload or {})
    selected_site = _selected_site(action_payload)
    action = _normalize_action_name(action_name)
    before_topology = _platform_topology(before)
    after_topology = _platform_topology(after)
    selected = _selected_component_summary(before, after, selected_site=selected_site)
    terrain_delta = _terrain_delta(before_topology, after_topology)
    player_motion = _player_motion(before_topology, after_topology)
    goal_relation = _goal_relation(before_topology, after_topology)
    change = _change_observation(before, after)

    selected_color = selected.get("before_color")
    click_destructible_candidate = bool(
        action == "ACTION6"
        and selected_site is not None
        and (
            selected_color == 14
            or int(terrain_delta["destructible_area_delta"]) < 0
            or int(terrain_delta["destructible_component_count_delta"]) < 0
        )
    )
    destructible_removed_candidate = bool(
        int(terrain_delta["destructible_area_delta"]) < 0
        or int(terrain_delta["destructible_component_count_delta"]) < 0
    )
    toggle_pair_candidate = bool(
        action == "ACTION6"
        and selected_site is not None
        and (
            selected_color == 12
            or int(terrain_delta["toggle_area_delta"]) != 0
            or int(terrain_delta["toggle_component_count_delta"]) != 0
        )
    )
    platform_gap_opened_candidate = bool(
        click_destructible_candidate and destructible_removed_candidate
    )
    gravity_fall_candidate = bool(
        abs(float(player_motion.get("delta", [0.0, 0.0])[1])) >= 2.0
        and int(player_motion.get("moved_player_count", 0)) > 0
    )
    hazard_relation_candidate = bool(
        int(before_topology["hazard_component_count"]) > 0
        and (
            platform_gap_opened_candidate
            or toggle_pair_candidate
            or gravity_fall_candidate
            or selected_color == 15
        )
    )
    goal_route_progress_candidate = bool(
        goal_relation.get("distance_delta") is not None
        and float(goal_relation.get("distance_delta") or 0.0) < -0.5
    )
    score_closure_candidate = bool(
        int(score_delta) > 0
        and (
            platform_gap_opened_candidate
            or toggle_pair_candidate
            or gravity_fall_candidate
            or goal_route_progress_candidate
        )
    )
    candidate_types: set[str] = set()
    if click_destructible_candidate:
        candidate_types.add("click_destructible")
    if destructible_removed_candidate:
        candidate_types.add("destructible_removed")
    if platform_gap_opened_candidate:
        candidate_types.add("platform_gap_opened")
    if toggle_pair_candidate:
        candidate_types.add("toggle_pair")
    if gravity_fall_candidate:
        candidate_types.add("gravity_fall")
    if hazard_relation_candidate:
        candidate_types.add("hazard_relation")
    if goal_route_progress_candidate:
        candidate_types.add("goal_route_progress")
    if score_closure_candidate:
        candidate_types.add("score_closure")

    return _sanitize(
        {
            "schema_version": PLATFORM_DESTRUCTIBLE_TOPOLOGY_PACKET_SCHEMA_VERSION,
            "sensor_family": "platform_destructible_topology_sensor",
            "game_id": str(game_id or "").strip().lower(),
            "state_hash": str(state_hash or _frame_hash(before)),
            "step": int(step),
            "action_name": action,
            "action_payload": action_payload,
            "selected_site": list(selected_site) if selected_site is not None else None,
            "score_delta": int(score_delta),
            "selected_component": selected,
            "before_counts": _compact_counts(before_topology),
            "after_counts": _compact_counts(after_topology),
            "terrain_delta": terrain_delta,
            "player_motion": player_motion,
            "goal_relation": goal_relation,
            "changed_cell_count": int(change["changed_cell_count"]),
            "terrain_changed_cell_count": int(change["terrain_changed_cell_count"]),
            "player_changed_cell_count": int(change["player_changed_cell_count"]),
            "hazard_changed_cell_count": int(change["hazard_changed_cell_count"]),
            "terrain_change_components": change["terrain_change_components"][:16],
            "click_destructible_candidate": click_destructible_candidate,
            "destructible_removed_candidate": destructible_removed_candidate,
            "platform_gap_opened_candidate": platform_gap_opened_candidate,
            "toggle_pair_candidate": toggle_pair_candidate,
            "gravity_fall_candidate": gravity_fall_candidate,
            "hazard_relation_candidate": hazard_relation_candidate,
            "goal_route_progress_candidate": goal_route_progress_candidate,
            "score_closure_candidate": score_closure_candidate,
            "candidate_types": sorted(candidate_types),
            "diagnostic_only": True,
        }
    )


def validate_platform_destructible_topology_packet(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if (
        str(packet.get("schema_version", "") or "")
        != PLATFORM_DESTRUCTIBLE_TOPOLOGY_PACKET_SCHEMA_VERSION
    ):
        reasons.append("schema_version_mismatch")
    if (
        str(packet.get("sensor_family", "") or "")
        != "platform_destructible_topology_sensor"
    ):
        reasons.append("sensor_family_mismatch")
    selected_site = packet.get("selected_site")
    if selected_site is not None and not _is_xy(selected_site):
        reasons.append("invalid_selected_site")
    return {
        "schema_version": "eca_platform_destructible_topology_validation.v1",
        "accepted": not reasons,
        "rejected_reasons": reasons,
        "family": "platform_destructible_topology",
        "mechanic_state_type": _mechanic_state_type(packet),
        "goal_link_type": "terrain_delta_to_player_goal_progress",
        "candidate_types": list(packet.get("candidate_types") or []),
    }


def summarize_platform_destructible_topology_packets(
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
    terrain_changed_cell_count_sum = 0
    destructible_area_delta_sum = 0
    toggle_area_delta_sum = 0
    score_delta_sum = 0

    for packet in packet_rows:
        validation = validate_platform_destructible_topology_packet(packet)
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
            ("click_destructible_candidate", "click_destructible_candidate_count"),
            ("destructible_removed_candidate", "destructible_removed_candidate_count"),
            ("platform_gap_opened_candidate", "platform_gap_opened_candidate_count"),
            ("toggle_pair_candidate", "toggle_pair_candidate_count"),
            ("gravity_fall_candidate", "gravity_fall_candidate_count"),
            ("hazard_relation_candidate", "hazard_relation_candidate_count"),
            ("goal_route_progress_candidate", "goal_route_progress_candidate_count"),
            ("score_closure_candidate", "score_closure_candidate_count"),
        ):
            if bool(packet.get(field)):
                counts[name] += 1
                by_game[game_id][name] += 1
        for candidate_type in list(packet.get("candidate_types") or []):
            candidate_type_counts[str(candidate_type)] += 1
        terrain_delta = packet.get("terrain_delta")
        terrain_delta = terrain_delta if isinstance(terrain_delta, Mapping) else {}
        changed = _safe_int(packet.get("changed_cell_count"))
        terrain_changed = _safe_int(packet.get("terrain_changed_cell_count"))
        destructible_delta = _safe_int(terrain_delta.get("destructible_area_delta"))
        toggle_delta = _safe_int(terrain_delta.get("toggle_area_delta"))
        score_delta = _safe_int(packet.get("score_delta"))
        changed_cell_count_sum += changed
        terrain_changed_cell_count_sum += terrain_changed
        destructible_area_delta_sum += destructible_delta
        toggle_area_delta_sum += toggle_delta
        score_delta_sum += score_delta
        if len(samples) < sample_limit:
            selected = packet.get("selected_component")
            selected = selected if isinstance(selected, Mapping) else {}
            motion = packet.get("player_motion")
            motion = motion if isinstance(motion, Mapping) else {}
            goal = packet.get("goal_relation")
            goal = goal if isinstance(goal, Mapping) else {}
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "action_name": action,
                    "selected_site": packet.get("selected_site"),
                    "selected_before_color": selected.get("before_color"),
                    "candidate_types": list(packet.get("candidate_types") or []),
                    "destructible_area_delta": destructible_delta,
                    "toggle_area_delta": toggle_delta,
                    "player_delta": motion.get("delta"),
                    "goal_distance_delta": goal.get("distance_delta"),
                    "score_delta": score_delta,
                }
            )

    return _sanitize(
        {
            "schema_version": (
                PLATFORM_DESTRUCTIBLE_TOPOLOGY_DIAGNOSTIC_SENSOR_SCHEMA_VERSION
            ),
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
            "click_destructible_candidate_count": int(
                counts.get("click_destructible_candidate_count", 0)
            ),
            "destructible_removed_candidate_count": int(
                counts.get("destructible_removed_candidate_count", 0)
            ),
            "platform_gap_opened_candidate_count": int(
                counts.get("platform_gap_opened_candidate_count", 0)
            ),
            "toggle_pair_candidate_count": int(
                counts.get("toggle_pair_candidate_count", 0)
            ),
            "gravity_fall_candidate_count": int(
                counts.get("gravity_fall_candidate_count", 0)
            ),
            "hazard_relation_candidate_count": int(
                counts.get("hazard_relation_candidate_count", 0)
            ),
            "goal_route_progress_candidate_count": int(
                counts.get("goal_route_progress_candidate_count", 0)
            ),
            "score_closure_candidate_count": int(
                counts.get("score_closure_candidate_count", 0)
            ),
            "changed_cell_count_sum": int(changed_cell_count_sum),
            "terrain_changed_cell_count_sum": int(terrain_changed_cell_count_sum),
            "destructible_area_delta_sum": int(destructible_area_delta_sum),
            "toggle_area_delta_sum": int(toggle_area_delta_sum),
            "score_delta_sum": int(score_delta_sum),
            "candidate_type_counts": dict(sorted(candidate_type_counts.items())),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "samples": samples,
            "next_planner_contract": {
                "required_before_ranking": (
                    "platform topology posterior with fall/hazard controls"
                ),
                "allowed_connection": (
                    "diagnostic_only_until_platform_topology_posterior_controls"
                ),
                "action_generation_allowed": False,
            },
        }
    )


def summarize_platform_destructible_topology_posterior(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Aggregate platform topology observations into causal evidence rows.

    This posterior is intentionally observational. It groups packet-level
    terrain/player/goal deltas into reusable hypotheses, but it does not
    generate actions, rank candidates, or authorize planner connections.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    candidate_type_counts: Counter[str] = Counter()
    hypothesis_aggs: dict[str, dict[str, Any]] = {}
    score_delta_sum = 0
    destructible_area_delta_sum = 0
    toggle_area_delta_sum = 0
    goal_distance_delta_sum = 0.0
    player_vertical_delta_sum = 0.0

    for packet in packet_rows:
        validation = validate_platform_destructible_topology_packet(packet)
        accepted = bool(validation.get("accepted"))
        counts["accepted_packet_count" if accepted else "rejected_packet_count"] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["packet_count"] += 1
        if not accepted:
            continue

        action_name = str(packet.get("action_name", "") or "unknown")
        candidate_types = sorted(str(row) for row in list(packet.get("candidate_types") or []))
        if not candidate_types:
            candidate_types = ["observation"]
        for candidate_type in candidate_types:
            candidate_type_counts[candidate_type] += 1
            by_game[game_id][f"candidate_type:{candidate_type}"] += 1

        selected = packet.get("selected_component")
        selected = selected if isinstance(selected, Mapping) else {}
        selected_color = selected.get("before_color")
        hypothesis_type = _platform_posterior_hypothesis_type(packet)
        hypothesis_key = _platform_posterior_hypothesis_key(
            hypothesis_type=hypothesis_type,
            action_name=action_name,
            selected_color=selected_color,
            candidate_types=candidate_types,
        )
        terrain_delta = packet.get("terrain_delta")
        terrain_delta = terrain_delta if isinstance(terrain_delta, Mapping) else {}
        player_motion = packet.get("player_motion")
        player_motion = player_motion if isinstance(player_motion, Mapping) else {}
        goal_relation = packet.get("goal_relation")
        goal_relation = goal_relation if isinstance(goal_relation, Mapping) else {}
        player_delta = player_motion.get("delta")
        if not isinstance(player_delta, Sequence) or len(player_delta) < 2:
            player_delta = [0.0, 0.0]

        score_delta = _safe_int(packet.get("score_delta"))
        destructible_delta = _safe_int(terrain_delta.get("destructible_area_delta"))
        toggle_delta = _safe_int(terrain_delta.get("toggle_area_delta"))
        goal_delta = _safe_float(goal_relation.get("distance_delta"))
        player_vertical_delta = _safe_float(player_delta[1])
        score_delta_sum += score_delta
        destructible_area_delta_sum += destructible_delta
        toggle_area_delta_sum += toggle_delta
        goal_distance_delta_sum += goal_delta
        player_vertical_delta_sum += player_vertical_delta

        agg = hypothesis_aggs.setdefault(
            hypothesis_key,
            {
                "hypothesis_key": hypothesis_key,
                "hypothesis_type": hypothesis_type,
                "action_name": action_name,
                "selected_before_color": selected_color,
                "candidate_types": candidate_types,
                "observation_count": 0,
                "score_delta_sum": 0,
                "destructible_area_delta_sum": 0,
                "toggle_area_delta_sum": 0,
                "terrain_changed_cell_count_sum": 0,
                "changed_cell_count_sum": 0,
                "goal_distance_delta_sum": 0.0,
                "player_vertical_delta_sum": 0.0,
                "destructible_removed_observation_count": 0,
                "platform_gap_opened_observation_count": 0,
                "toggle_pair_observation_count": 0,
                "gravity_fall_observation_count": 0,
                "goal_route_progress_observation_count": 0,
                "score_closure_observation_count": 0,
                "positive_score_observation_count": 0,
                "game_ids": set(),
                "example_steps": [],
            },
        )
        agg["observation_count"] += 1
        agg["score_delta_sum"] += score_delta
        agg["destructible_area_delta_sum"] += destructible_delta
        agg["toggle_area_delta_sum"] += toggle_delta
        agg["terrain_changed_cell_count_sum"] += _safe_int(
            packet.get("terrain_changed_cell_count")
        )
        agg["changed_cell_count_sum"] += _safe_int(packet.get("changed_cell_count"))
        agg["goal_distance_delta_sum"] += goal_delta
        agg["player_vertical_delta_sum"] += player_vertical_delta
        agg["game_ids"].add(game_id)
        if len(agg["example_steps"]) < 4:
            agg["example_steps"].append(_safe_int(packet.get("step")))
        if destructible_delta < 0 or bool(packet.get("destructible_removed_candidate")):
            agg["destructible_removed_observation_count"] += 1
            counts["destructible_removal_support_count"] += 1
            by_game[game_id]["destructible_removal_support_count"] += 1
        if bool(packet.get("platform_gap_opened_candidate")):
            agg["platform_gap_opened_observation_count"] += 1
            counts["platform_gap_support_count"] += 1
            by_game[game_id]["platform_gap_support_count"] += 1
        if bool(packet.get("toggle_pair_candidate")):
            agg["toggle_pair_observation_count"] += 1
            counts["toggle_support_count"] += 1
            by_game[game_id]["toggle_support_count"] += 1
        if bool(packet.get("gravity_fall_candidate")):
            agg["gravity_fall_observation_count"] += 1
            counts["gravity_support_count"] += 1
            by_game[game_id]["gravity_support_count"] += 1
        if bool(packet.get("goal_route_progress_candidate")):
            agg["goal_route_progress_observation_count"] += 1
            counts["goal_progress_support_count"] += 1
            by_game[game_id]["goal_progress_support_count"] += 1
        if bool(packet.get("score_closure_candidate")):
            agg["score_closure_observation_count"] += 1
            counts["score_closure_support_count"] += 1
            by_game[game_id]["score_closure_support_count"] += 1
        if score_delta > 0:
            agg["positive_score_observation_count"] += 1
            counts["positive_score_packet_count"] += 1
            by_game[game_id]["positive_score_packet_count"] += 1

    hypotheses: list[dict[str, Any]] = []
    support_state_counts: Counter[str] = Counter()
    supported_hypothesis_count = 0
    score_closure_hypothesis_count = 0
    goal_progress_hypothesis_count = 0
    terrain_delta_hypothesis_count = 0
    for agg in hypothesis_aggs.values():
        support_state = _platform_posterior_support_state(agg)
        agg["support_state"] = support_state
        agg["planner_disposition"] = "posterior_only"
        agg["ranking_allowed"] = False
        agg["action_generation_allowed"] = False
        agg["game_ids"] = sorted(str(row) for row in agg["game_ids"])
        agg["goal_distance_delta_sum"] = round(float(agg["goal_distance_delta_sum"]), 3)
        agg["player_vertical_delta_sum"] = round(
            float(agg["player_vertical_delta_sum"]), 3
        )
        support_state_counts[support_state] += 1
        if support_state not in {"observed_no_platform_effect", "insufficient_evidence"}:
            supported_hypothesis_count += 1
        if support_state == "score_closure_supported":
            score_closure_hypothesis_count += 1
        if support_state == "goal_progress_supported":
            goal_progress_hypothesis_count += 1
        if support_state in {
            "terrain_to_gravity_supported",
            "terrain_delta_supported",
        }:
            terrain_delta_hypothesis_count += 1
        hypotheses.append(_sanitize(agg))

    hypotheses.sort(
        key=lambda row: (
            _platform_support_rank(str(row.get("support_state", ""))),
            -_safe_int(row.get("score_delta_sum")),
            -_safe_int(row.get("observation_count")),
            str(row.get("hypothesis_key", "")),
        )
    )

    if score_closure_hypothesis_count > 0:
        promotion_state = "platform_topology_score_closure_posterior"
    elif goal_progress_hypothesis_count > 0:
        promotion_state = "platform_topology_goal_progress_posterior"
    elif terrain_delta_hypothesis_count > 0:
        promotion_state = "platform_topology_terrain_delta_posterior"
    elif supported_hypothesis_count > 0:
        promotion_state = "platform_topology_observed_effect_posterior"
    else:
        promotion_state = "no_platform_topology_posterior_evidence"

    return _sanitize(
        {
            "schema_version": PLATFORM_DESTRUCTIBLE_TOPOLOGY_POSTERIOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "accepted_packet_count": int(counts.get("accepted_packet_count", 0)),
            "rejected_packet_count": int(counts.get("rejected_packet_count", 0)),
            "posterior_hypothesis_count": int(len(hypotheses)),
            "supported_hypothesis_count": int(supported_hypothesis_count),
            "score_closure_hypothesis_count": int(score_closure_hypothesis_count),
            "goal_progress_hypothesis_count": int(goal_progress_hypothesis_count),
            "terrain_delta_hypothesis_count": int(terrain_delta_hypothesis_count),
            "score_delta_sum": int(score_delta_sum),
            "destructible_area_delta_sum": int(destructible_area_delta_sum),
            "toggle_area_delta_sum": int(toggle_area_delta_sum),
            "goal_distance_delta_sum": round(float(goal_distance_delta_sum), 3),
            "player_vertical_delta_sum": round(float(player_vertical_delta_sum), 3),
            "destructible_removal_support_count": int(
                counts.get("destructible_removal_support_count", 0)
            ),
            "platform_gap_support_count": int(counts.get("platform_gap_support_count", 0)),
            "toggle_support_count": int(counts.get("toggle_support_count", 0)),
            "gravity_support_count": int(counts.get("gravity_support_count", 0)),
            "goal_progress_support_count": int(
                counts.get("goal_progress_support_count", 0)
            ),
            "score_closure_support_count": int(
                counts.get("score_closure_support_count", 0)
            ),
            "positive_score_packet_count": int(
                counts.get("positive_score_packet_count", 0)
            ),
            "candidate_type_counts": dict(sorted(candidate_type_counts.items())),
            "support_state_counts": dict(sorted(support_state_counts.items())),
            "promotion_state": promotion_state,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "posterior_hypotheses": hypotheses[:sample_limit],
            "next_planner_contract": {
                "required_before_ranking": (
                    "counterfactual controls plus focused, breadth, and all25 non-regression"
                ),
                "allowed_connection": (
                    "diagnostic_only_until_platform_topology_posterior_controls"
                ),
                "action_generation_allowed": False,
            },
        }
    )


def summarize_platform_destructible_topology_candidate_miner(
    packets: Iterable[Mapping[str, Any]],
    *,
    posterior_summary: Mapping[str, Any] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Mine guarded-probe candidate templates from topology posterior rows."""

    posterior = (
        dict(posterior_summary)
        if isinstance(posterior_summary, Mapping)
        else summarize_platform_destructible_topology_posterior(
            packets,
            sample_limit=max(int(sample_limit), 128),
        )
    )
    hypotheses = [
        dict(row)
        for row in list(posterior.get("posterior_hypotheses") or [])
        if isinstance(row, Mapping)
    ]
    by_game: dict[str, Counter[str]] = {}
    by_kind: dict[str, Counter[str]] = {}
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for index, hypothesis in enumerate(hypotheses):
        support_state = str(hypothesis.get("support_state", "") or "")
        hypothesis_type = str(hypothesis.get("hypothesis_type", "") or "")
        candidate_types = [
            str(row) for row in list(hypothesis.get("candidate_types") or [])
        ]
        candidate_type_set = set(candidate_types)
        game_ids = [
            str(game_id)
            for game_id in list(hypothesis.get("game_ids") or [])
            if str(game_id)
        ]
        game_id = game_ids[0] if game_ids else "unknown"
        by_game.setdefault(game_id, Counter())
        candidate_kind = ""
        evidence_role = ""
        rejected_reason = ""

        if support_state == "score_closure_supported":
            candidate_kind = "platform_score_closure_replay_candidate"
            evidence_role = "platform_score_closure"
        elif support_state == "goal_progress_supported":
            candidate_kind = "platform_goal_progress_probe_candidate"
            evidence_role = "platform_goal_progress"
        elif support_state == "terrain_to_gravity_supported":
            candidate_kind = "platform_gap_gravity_probe_candidate"
            evidence_role = "platform_gap_to_gravity"
        elif support_state == "terrain_delta_supported" and (
            "platform_gap_opened" in candidate_type_set
            or "click_destructible" in candidate_type_set
            or hypothesis_type
            in {
                "platform_gap_opened_relation",
                "platform_destructible_click_relation",
            }
        ):
            candidate_kind = "platform_destructible_gap_retest_candidate"
            evidence_role = "platform_destructible_gap"
        elif support_state == "terrain_delta_supported" and (
            "toggle_pair" in candidate_type_set
            or hypothesis_type == "platform_toggle_pair_relation"
        ):
            candidate_kind = "platform_toggle_pair_retest_candidate"
            evidence_role = "platform_toggle_pair"
        elif support_state in {"observed_no_platform_effect", "insufficient_evidence"}:
            rejected_reason = "insufficient_supported_platform_effect"
        else:
            rejected_reason = "unsupported_platform_hypothesis"

        if rejected_reason:
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "posterior_hypothesis_index": int(index),
                        "game_id": game_id,
                        "rejected_reason": rejected_reason,
                        "hypothesis_type": hypothesis_type,
                        "support_state": support_state,
                        "candidate_types": candidate_types,
                    }
                )
            continue

        candidate_key = (
            f"platform_destructible_topology:{game_id}:{candidate_kind}:"
            f"{_text_hash(str(hypothesis.get('hypothesis_key', '')))}:"
            f"idx{int(index)}"
        )
        row = {
            "candidate_key": candidate_key,
            "candidate_kind": candidate_kind,
            "game_id": game_id,
            "game_ids": game_ids,
            "source_action_name": str(hypothesis.get("action_name", "") or ""),
            "selected_before_color": hypothesis.get("selected_before_color"),
            "evidence_role": evidence_role,
            "support_state": support_state,
            "hypothesis_type": hypothesis_type,
            "candidate_types": candidate_types,
            "observation_count": _safe_int(hypothesis.get("observation_count")),
            "score_delta_sum": _safe_int(hypothesis.get("score_delta_sum")),
            "destructible_area_delta_sum": _safe_int(
                hypothesis.get("destructible_area_delta_sum")
            ),
            "toggle_area_delta_sum": _safe_int(
                hypothesis.get("toggle_area_delta_sum")
            ),
            "goal_distance_delta_sum": _safe_float(
                hypothesis.get("goal_distance_delta_sum")
            ),
            "player_vertical_delta_sum": _safe_float(
                hypothesis.get("player_vertical_delta_sum")
            ),
            "sample_steps": list(hypothesis.get("example_steps") or [])[:4],
            "requires_guarded_probe": True,
            "scoreless_terrain_control_required": True,
            "wrong_effect_control_required": True,
            "hazard_and_gravity_control_required": True,
            "runtime_execution_enabled": False,
            "planner_connection_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
        }
        candidates.append(row)
        by_game[game_id]["candidate_count"] += 1
        by_game[game_id][f"candidate_kind:{candidate_kind}"] += 1
        by_kind.setdefault(candidate_kind, Counter())
        by_kind[candidate_kind]["candidate_count"] += 1

    candidate_count = int(len(candidates))
    if candidate_count > 0:
        promotion_state = "platform_destructible_topology_candidate_miner_supported"
    elif hypotheses:
        promotion_state = "platform_destructible_topology_candidate_miner_blocked"
    else:
        promotion_state = "platform_destructible_topology_candidate_miner_no_posterior"

    return _sanitize(
        {
            "schema_version": (
                PLATFORM_DESTRUCTIBLE_TOPOLOGY_CANDIDATE_MINER_SCHEMA_VERSION
            ),
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
            "supported_hypothesis_count": _safe_int(
                posterior.get("supported_hypothesis_count")
            ),
            "candidate_count": int(candidate_count),
            "platform_destructible_gap_retest_candidate_count": int(
                by_kind.get(
                    "platform_destructible_gap_retest_candidate", Counter()
                ).get("candidate_count", 0)
            ),
            "platform_toggle_pair_retest_candidate_count": int(
                by_kind.get("platform_toggle_pair_retest_candidate", Counter()).get(
                    "candidate_count", 0
                )
            ),
            "platform_gap_gravity_probe_candidate_count": int(
                by_kind.get("platform_gap_gravity_probe_candidate", Counter()).get(
                    "candidate_count", 0
                )
            ),
            "platform_goal_progress_probe_candidate_count": int(
                by_kind.get("platform_goal_progress_probe_candidate", Counter()).get(
                    "candidate_count", 0
                )
            ),
            "platform_score_closure_replay_candidate_count": int(
                by_kind.get("platform_score_closure_replay_candidate", Counter()).get(
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
                    "guarded probe with scoreless terrain, wrong-effect, "
                    "hazard, and gravity controls"
                ),
                "allowed_connection": "platform_topology_candidate_design_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_platform_destructible_topology_guarded_probe_outcomes(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize outcomes from default-off platform topology guarded probes."""

    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_candidate_kind: dict[str, Counter[str]] = {}
    rejected_reasons: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    preflight_samples: list[dict[str, Any]] = []

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
            preflight = unsolved.get(
                "platform_destructible_topology_guarded_probe_preflight"
            )
            if isinstance(preflight, Mapping) and bool(preflight.get("enabled")):
                counts["preflight_count"] += 1
                by_game[game_id]["preflight_count"] += 1
                for reason, value in dict(
                    preflight.get("rejected_candidate_reasons") or {}
                ).items():
                    amount = _safe_int(value)
                    if amount > 0:
                        rejected_reasons[str(reason)] += int(amount)
                if len(preflight_samples) < sample_limit:
                    preflight_samples.append(
                        {
                            "game_id": game_id,
                            "variant": str(row.get("variant", "") or "unknown"),
                            "step": _safe_int(item.get("step")),
                            "reason": str(preflight.get("reason", "") or ""),
                            "candidate_count": _safe_int(
                                preflight.get("candidate_count")
                            ),
                            "admissible_candidate_count": _safe_int(
                                preflight.get("admissible_candidate_count")
                            ),
                            "rejected_candidate_count": _safe_int(
                                preflight.get("rejected_candidate_count")
                            ),
                        }
                    )

            is_guarded = str(item.get("phase", "") or "") == (
                "platform_destructible_topology_guarded_probe"
            )
            is_guarded = is_guarded or str(unsolved.get("source", "") or "") == (
                "platform_destructible_topology_guarded_probe"
            )
            if not is_guarded:
                continue

            packet = item.get("platform_destructible_topology_packet")
            packet = packet if isinstance(packet, Mapping) else {}
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            terrain_delta = packet.get("terrain_delta")
            terrain_delta = terrain_delta if isinstance(terrain_delta, Mapping) else {}
            candidate_kind = str(
                unsolved.get("selected_candidate_kind")
                or unsolved.get("candidate_kind")
                or "unknown"
            )
            selected_site = unsolved.get("selected_site")
            selected_site = list(selected_site) if _is_xy(selected_site) else []
            score_delta = _safe_int(
                packet.get("score_delta"),
                _safe_int(outcome.get("score_delta")),
            )
            destructible_delta = _safe_int(
                terrain_delta.get("destructible_area_delta")
            )
            toggle_delta = _safe_int(terrain_delta.get("toggle_area_delta"))
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
            counts["destructible_area_delta_sum"] += int(destructible_delta)
            counts["toggle_area_delta_sum"] += int(toggle_delta)
            counts["changed_pixel_count_sum"] += int(changed_pixel_count)
            if score_delta > 0:
                outcome_state = "score_positive"
                counts["score_positive_execution_count"] += 1
                by_game[game_id]["score_positive_execution_count"] += 1
                by_candidate_kind[candidate_kind]["score_positive_execution_count"] += 1
            elif destructible_delta < 0 or bool(
                packet.get("platform_gap_opened_candidate")
            ):
                outcome_state = "scoreless_platform_gap"
                counts["scoreless_platform_gap_execution_count"] += 1
                by_game[game_id]["scoreless_platform_gap_execution_count"] += 1
                by_candidate_kind[candidate_kind][
                    "scoreless_platform_gap_execution_count"
                ] += 1
            elif toggle_delta != 0 or bool(packet.get("toggle_pair_candidate")):
                outcome_state = "scoreless_toggle_delta"
                counts["scoreless_toggle_delta_execution_count"] += 1
                by_game[game_id]["scoreless_toggle_delta_execution_count"] += 1
                by_candidate_kind[candidate_kind][
                    "scoreless_toggle_delta_execution_count"
                ] += 1
            elif changed_pixel_count > 0:
                outcome_state = "scoreless_visible_change"
                counts["scoreless_visible_change_execution_count"] += 1
                by_game[game_id]["scoreless_visible_change_execution_count"] += 1
                by_candidate_kind[candidate_kind][
                    "scoreless_visible_change_execution_count"
                ] += 1
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
                        "stable_candidate_key": str(
                            unsolved.get("stable_candidate_key", "") or ""
                        ),
                        "site_candidate_key": str(
                            unsolved.get("site_candidate_key", "") or ""
                        ),
                        "candidate_kind": candidate_kind,
                        "selected_site": selected_site,
                        "outcome_state": outcome_state,
                        "score_delta": int(score_delta),
                        "destructible_area_delta": int(destructible_delta),
                        "toggle_area_delta": int(toggle_delta),
                        "changed_pixel_count": int(changed_pixel_count),
                    }
                )

    execution_count = int(counts.get("execution_count", 0))
    if execution_count <= 0 and int(counts.get("preflight_count", 0)) > 0:
        promotion_state = "platform_destructible_topology_guarded_probe_preflight_only"
    elif execution_count <= 0:
        promotion_state = "platform_destructible_topology_guarded_probe_no_execution"
    elif int(counts.get("score_positive_execution_count", 0)) > 0:
        promotion_state = "platform_destructible_topology_guarded_probe_score_positive"
    elif int(counts.get("scoreless_platform_gap_execution_count", 0)) > 0:
        promotion_state = "platform_destructible_topology_guarded_probe_scoreless_gap"
    elif int(counts.get("scoreless_toggle_delta_execution_count", 0)) > 0:
        promotion_state = "platform_destructible_topology_guarded_probe_scoreless_toggle"
    elif int(counts.get("scoreless_visible_change_execution_count", 0)) > 0:
        promotion_state = "platform_destructible_topology_guarded_probe_visible_change"
    else:
        promotion_state = "platform_destructible_topology_guarded_probe_scoreless_no_effect"

    return _sanitize(
        {
            "schema_version": (
                PLATFORM_DESTRUCTIBLE_TOPOLOGY_GUARDED_PROBE_OUTCOME_SCHEMA_VERSION
            ),
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
            "scoreless_platform_gap_execution_count": int(
                counts.get("scoreless_platform_gap_execution_count", 0)
            ),
            "scoreless_toggle_delta_execution_count": int(
                counts.get("scoreless_toggle_delta_execution_count", 0)
            ),
            "scoreless_visible_change_execution_count": int(
                counts.get("scoreless_visible_change_execution_count", 0)
            ),
            "scoreless_no_effect_execution_count": int(
                counts.get("scoreless_no_effect_execution_count", 0)
            ),
            "score_delta_sum": int(counts.get("score_delta_sum", 0)),
            "destructible_area_delta_sum": int(
                counts.get("destructible_area_delta_sum", 0)
            ),
            "toggle_area_delta_sum": int(counts.get("toggle_area_delta_sum", 0)),
            "changed_pixel_count_sum": int(counts.get("changed_pixel_count_sum", 0)),
            "rejected_candidate_reasons": dict(sorted(rejected_reasons.items())),
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
            "preflight_samples": preflight_samples,
            "next_runtime_contract": {
                "required_before_planner_promotion": (
                    "counterfactual controls must distinguish score-bearing "
                    "platform closure from scoreless terrain, wrong-effect, "
                    "hazard, and gravity outcomes before ranking changes"
                ),
                "allowed_connection": "guarded_probe_outcome_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_platform_destructible_topology_counterfactual_controls(
    rows: Iterable[Mapping[str, Any]],
    *,
    outcome_summary: Mapping[str, Any] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Check whether platform guarded outcomes have positive controls."""

    outcome = (
        dict(outcome_summary)
        if isinstance(outcome_summary, Mapping)
        else summarize_platform_destructible_topology_guarded_probe_outcomes(
            rows,
            sample_limit=sample_limit,
        )
    )
    score_positive = _safe_int(outcome.get("score_positive_execution_count"))
    scoreless_gap = _safe_int(outcome.get("scoreless_platform_gap_execution_count"))
    scoreless_toggle = _safe_int(
        outcome.get("scoreless_toggle_delta_execution_count")
    )
    scoreless_visible = _safe_int(
        outcome.get("scoreless_visible_change_execution_count")
    )
    scoreless_no_effect = _safe_int(outcome.get("scoreless_no_effect_execution_count"))
    scoreless_control = int(
        scoreless_gap + scoreless_toggle + scoreless_visible + scoreless_no_effect
    )
    control_pair = int(score_positive > 0 and scoreless_control > 0)
    margin = int(score_positive - scoreless_control)
    positive_control_margin = int(control_pair > 0 and margin >= 0)

    if score_positive <= 0 and scoreless_control > 0:
        promotion_state = (
            "platform_destructible_topology_counterfactual_control_missing_positive"
        )
    elif score_positive <= 0:
        promotion_state = (
            "platform_destructible_topology_counterfactual_control_no_guarded_outcome"
        )
    elif scoreless_control <= 0:
        promotion_state = (
            "platform_destructible_topology_counterfactual_control_no_scoreless_control"
        )
    elif positive_control_margin > 0:
        promotion_state = (
            "platform_destructible_topology_counterfactual_control_supported"
        )
    else:
        promotion_state = (
            "platform_destructible_topology_counterfactual_control_dominated"
        )

    samples = [
        dict(row)
        for row in list(outcome.get("samples") or [])
        if isinstance(row, Mapping)
    ][:sample_limit]

    return _sanitize(
        {
            "schema_version": (
                PLATFORM_DESTRUCTIBLE_TOPOLOGY_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION
            ),
            "default_off": True,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "outcome_promotion_state": str(outcome.get("promotion_state", "") or ""),
            "execution_count": _safe_int(outcome.get("execution_count")),
            "runtime_outcome_count": _safe_int(outcome.get("runtime_outcome_count")),
            "score_positive_execution_count": int(score_positive),
            "scoreless_platform_gap_execution_count": int(scoreless_gap),
            "scoreless_toggle_delta_execution_count": int(scoreless_toggle),
            "scoreless_visible_change_execution_count": int(scoreless_visible),
            "scoreless_no_effect_execution_count": int(scoreless_no_effect),
            "scoreless_control_execution_count": int(scoreless_control),
            "counterfactual_pair_count": int(control_pair),
            "counterfactual_margin": int(margin),
            "positive_control_margin_count": int(positive_control_margin),
            "planner_promotion_blocked": bool(positive_control_margin <= 0),
            "promotion_state": promotion_state,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in dict(outcome.get("by_game") or {}).items()
                if isinstance(counter, Mapping)
            },
            "by_candidate_kind": {
                kind: dict(sorted(counter.items()))
                for kind, counter in dict(
                    outcome.get("by_candidate_kind") or {}
                ).items()
                if isinstance(counter, Mapping)
            },
            "samples": samples,
            "next_runtime_contract": {
                "required_before_all25": (
                    "positive score-bearing platform controls or an explicit "
                    "blocked record must be preserved before all25 promotion"
                ),
                "allowed_connection": "counterfactual_control_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_platform_destructible_topology_all25_non_regression(
    run_rows: Iterable[Mapping[str, Any]],
    *,
    expected_score_sum: int | None = None,
    expected_by_game: Mapping[str, Any] | None = None,
    require_no_game_conditioned_primitives: bool = True,
    counterfactual_summary: Mapping[str, Any] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Check all25 preservation while platform topology probes are enabled."""

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
        reason_counts["platform_topology_all25_non_regression_supported"] += 1
        promotion_state = "platform_destructible_topology_all25_non_regression_supported"
    elif rows:
        promotion_state = "platform_destructible_topology_all25_non_regression_blocked"
    else:
        promotion_state = "platform_destructible_topology_all25_non_regression_no_rows"

    counterfactual = (
        dict(counterfactual_summary)
        if isinstance(counterfactual_summary, Mapping)
        else {}
    )
    return _sanitize(
        {
            "schema_version": (
                PLATFORM_DESTRUCTIBLE_TOPOLOGY_ALL25_NON_REGRESSION_SCHEMA_VERSION
            ),
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
            "expected_by_game": dict(sorted(expected_map.items())),
            "actual_by_game": dict(sorted(by_game.items())),
            "missing_game_count": int(len(missing_games)),
            "missing_games": missing_games[:sample_limit],
            "regressed_game_count": int(game_regression_count),
            "improved_game_count": int(
                sum(
                    1
                    for game_id, expected in expected_map.items()
                    if game_id in by_game and int(by_game[game_id]) > int(expected)
                )
            ),
            "regressions": regressions,
            "improvements": improvements,
            "status_counts": dict(sorted(status_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "counterfactual_promotion_state": str(
                counterfactual.get("promotion_state", "") or ""
            ),
            "counterfactual_planner_promotion_blocked": bool(
                counterfactual.get("planner_promotion_blocked", False)
            ),
            "counterfactual_pair_count": _safe_int(
                counterfactual.get("counterfactual_pair_count")
            ),
            "ready_for_planner_promotion_review": bool(
                supported
                and not bool(counterfactual.get("planner_promotion_blocked", True))
            ),
            "promotion_state": promotion_state,
            "next_runtime_contract": {
                "required_before_planner_promotion": (
                    "all25 non-regression plus non-blocked counterfactual controls"
                ),
                "allowed_connection": "all25_non_regression_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def _platform_topology(arr: np.ndarray) -> dict[str, Any]:
    components = _connected_components(arr)
    by_color: dict[int, list[dict[str, Any]]] = {}
    for component in components:
        by_color.setdefault(int(component.get("color", -1)), []).append(component)
    return {
        "components": components,
        "destructible_components": by_color.get(14, []),
        "toggle_components": by_color.get(12, []),
        "hazard_components": by_color.get(15, []),
        "goal_components": by_color.get(11, []),
        "player_components": by_color.get(9, []),
        "wall_components": by_color.get(5, []),
        "destructible_area": _area_sum(by_color.get(14, [])),
        "toggle_area": _area_sum(by_color.get(12, [])),
        "hazard_area": _area_sum(by_color.get(15, [])),
        "goal_area": _area_sum(by_color.get(11, [])),
        "player_area": _area_sum(by_color.get(9, [])),
        "destructible_component_count": len(by_color.get(14, [])),
        "toggle_component_count": len(by_color.get(12, [])),
        "hazard_component_count": len(by_color.get(15, [])),
        "goal_component_count": len(by_color.get(11, [])),
        "player_component_count": len(by_color.get(9, [])),
    }


def _selected_component_summary(
    before: np.ndarray,
    after: np.ndarray,
    *,
    selected_site: tuple[int, int] | None,
) -> dict[str, Any]:
    if selected_site is None:
        return {"selected": False}
    sx, sy = selected_site
    if not (0 <= sx < int(before.shape[1]) and 0 <= sy < int(before.shape[0])):
        return {"selected": False, "out_of_bounds": True}
    return {
        "selected": True,
        "site": [int(sx), int(sy)],
        "before_color": int(before[sy, sx]),
        "after_color": int(after[sy, sx]),
        "changed": bool(int(before[sy, sx]) != int(after[sy, sx])),
        "before_component": _component_at(before, selected_site),
        "after_component": _component_at(after, selected_site),
    }


def _terrain_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "destructible_area_delta": int(after["destructible_area"])
        - int(before["destructible_area"]),
        "destructible_component_count_delta": int(after["destructible_component_count"])
        - int(before["destructible_component_count"]),
        "toggle_area_delta": int(after["toggle_area"]) - int(before["toggle_area"]),
        "toggle_component_count_delta": int(after["toggle_component_count"])
        - int(before["toggle_component_count"]),
        "hazard_area_delta": int(after["hazard_area"]) - int(before["hazard_area"]),
        "hazard_component_count_delta": int(after["hazard_component_count"])
        - int(before["hazard_component_count"]),
        "goal_area_delta": int(after["goal_area"]) - int(before["goal_area"]),
        "player_area_delta": int(after["player_area"]) - int(before["player_area"]),
    }


def _player_motion(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_players = list(before.get("player_components") or [])
    after_players = list(after.get("player_components") or [])
    matches = _match_components(before_players, after_players, color=9)
    moved: list[dict[str, Any]] = []
    aggregate = [0.0, 0.0]
    for left, right in matches:
        bx, by = _centroid_tuple(left)
        ax, ay = _centroid_tuple(right)
        dx = ax - bx
        dy = ay - by
        if math.hypot(dx, dy) < 0.5:
            continue
        aggregate[0] += dx
        aggregate[1] += dy
        moved.append(
            {
                "before_centroid": [round(bx, 3), round(by, 3)],
                "after_centroid": [round(ax, 3), round(ay, 3)],
                "delta": [round(dx, 3), round(dy, 3)],
            }
        )
    return {
        "moved_player_count": int(len(moved)),
        "delta": [round(aggregate[0], 3), round(aggregate[1], 3)],
        "moved_players": moved[:8],
    }


def _goal_relation(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_player = _largest_component(before.get("player_components") or [])
    after_player = _largest_component(after.get("player_components") or [])
    before_goal = _largest_component(before.get("goal_components") or [])
    after_goal = _largest_component(after.get("goal_components") or [])
    if not before_player or not before_goal or not after_player or not after_goal:
        return {
            "distance_before": None,
            "distance_after": None,
            "distance_delta": None,
        }
    bpx, bpy = _centroid_tuple(before_player)
    bgx, bgy = _centroid_tuple(before_goal)
    apx, apy = _centroid_tuple(after_player)
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


def _change_observation(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    changed = before != after
    terrain_mask = changed & (
        np.isin(before, [12, 14]) | np.isin(after, [12, 14])
    )
    player_mask = changed & (np.isin(before, [9]) | np.isin(after, [9]))
    hazard_mask = changed & (np.isin(before, [15]) | np.isin(after, [15]))
    return {
        "changed_cell_count": int(np.count_nonzero(changed)),
        "terrain_changed_cell_count": int(np.count_nonzero(terrain_mask)),
        "player_changed_cell_count": int(np.count_nonzero(player_mask)),
        "hazard_changed_cell_count": int(np.count_nonzero(hazard_mask)),
        "terrain_change_components": _components_from_mask(
            terrain_mask,
            before=before,
            after=after,
        ),
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
            pair_counts: Counter[str] = Counter()
            while stack:
                px, py = stack.pop()
                xs.append(px)
                ys.append(py)
                pair_counts[f"{int(before[py, px])}->{int(after[py, px])}"] += 1
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
                    "id": f"tc{len(out)}",
                    "area": int(area),
                    "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    "centroid": [
                        round(float(sum(xs)) / float(area), 3),
                        round(float(sum(ys)) / float(area), 3),
                    ],
                    "color_pair_counts": dict(sorted(pair_counts.items())),
                }
            )
    return out


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


def _component_at(arr: np.ndarray, site: tuple[int, int]) -> dict[str, Any] | None:
    sx, sy = site
    color = int(arr[sy, sx])
    for component in _connected_components(arr):
        if int(component.get("color", -1)) != color:
            continue
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        if x0 <= sx <= x1 and y0 <= sy <= y1:
            return dict(component)
    return None


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
        bx, by = _centroid_tuple(before)
        best_index = -1
        best_distance = 1e9
        for index, after in enumerate(remaining):
            ax, ay = _centroid_tuple(after)
            distance = math.hypot(ax - bx, ay - by)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index >= 0 and best_distance <= 48.0:
            matches.append((before, remaining.pop(best_index)))
    return matches


def _compact_counts(topology: Mapping[str, Any]) -> dict[str, int]:
    return {
        "destructible_area": int(topology.get("destructible_area", 0)),
        "destructible_component_count": int(
            topology.get("destructible_component_count", 0)
        ),
        "toggle_area": int(topology.get("toggle_area", 0)),
        "toggle_component_count": int(topology.get("toggle_component_count", 0)),
        "hazard_area": int(topology.get("hazard_area", 0)),
        "hazard_component_count": int(topology.get("hazard_component_count", 0)),
        "goal_area": int(topology.get("goal_area", 0)),
        "goal_component_count": int(topology.get("goal_component_count", 0)),
        "player_area": int(topology.get("player_area", 0)),
        "player_component_count": int(topology.get("player_component_count", 0)),
    }


def _area_sum(components: Sequence[Mapping[str, Any]]) -> int:
    return int(sum(_safe_int(row.get("area")) for row in components))


def _largest_component(components: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not components:
        return None
    return max(components, key=lambda row: _safe_int(row.get("area")))


def _centroid_tuple(component: Mapping[str, Any]) -> tuple[float, float]:
    centroid = component.get("centroid")
    if isinstance(centroid, Sequence) and len(centroid) >= 2:
        return float(centroid[0]), float(centroid[1])
    bbox = component.get("bbox")
    if _is_bbox(bbox):
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return 0.0, 0.0


def _mechanic_state_type(packet: Mapping[str, Any]) -> str:
    types = set(str(row) for row in list(packet.get("candidate_types") or []))
    if "score_closure" in types:
        return "platform_topology_score_closure"
    if "goal_route_progress" in types:
        return "platform_goal_route_progress"
    if "gravity_fall" in types:
        return "platform_gravity_fall_relation"
    if "platform_gap_opened" in types:
        return "platform_gap_opened_relation"
    if "destructible_removed" in types:
        return "platform_destructible_removed_relation"
    if "toggle_pair" in types:
        return "platform_toggle_pair_relation"
    if "click_destructible" in types:
        return "platform_destructible_click_relation"
    return "platform_topology_observation"


def _platform_posterior_hypothesis_type(packet: Mapping[str, Any]) -> str:
    types = set(str(row) for row in list(packet.get("candidate_types") or []))
    if "score_closure" in types:
        return "platform_score_closure_relation"
    if "goal_route_progress" in types:
        return "platform_goal_route_progress_relation"
    if "gravity_fall" in types and "platform_gap_opened" in types:
        return "platform_gap_to_gravity_route_relation"
    if "gravity_fall" in types:
        return "platform_gravity_fall_relation"
    if "platform_gap_opened" in types:
        return "platform_gap_opened_relation"
    if "toggle_pair" in types:
        return "platform_toggle_pair_relation"
    if "click_destructible" in types:
        return "platform_destructible_click_relation"
    return "platform_topology_observation"


def _platform_posterior_hypothesis_key(
    *,
    hypothesis_type: str,
    action_name: str,
    selected_color: Any,
    candidate_types: Sequence[str],
) -> str:
    color_text = "none" if selected_color is None else str(selected_color)
    type_text = "+".join(str(row) for row in candidate_types) or "observation"
    return (
        f"{hypothesis_type}:action={str(action_name or 'unknown')}:"
        f"selected_color={color_text}:types={type_text}"
    )


def _platform_posterior_support_state(agg: Mapping[str, Any]) -> str:
    if _safe_int(agg.get("score_closure_observation_count")) > 0:
        return "score_closure_supported"
    if _safe_int(agg.get("goal_route_progress_observation_count")) > 0 and (
        _safe_int(agg.get("gravity_fall_observation_count")) > 0
        or _safe_int(agg.get("platform_gap_opened_observation_count")) > 0
        or _safe_int(agg.get("destructible_removed_observation_count")) > 0
        or _safe_int(agg.get("toggle_pair_observation_count")) > 0
    ):
        return "goal_progress_supported"
    if (
        _safe_int(agg.get("gravity_fall_observation_count")) > 0
        and (
            _safe_int(agg.get("destructible_removed_observation_count")) > 0
            or _safe_int(agg.get("platform_gap_opened_observation_count")) > 0
        )
    ):
        return "terrain_to_gravity_supported"
    if (
        _safe_int(agg.get("destructible_removed_observation_count")) > 0
        or _safe_int(agg.get("platform_gap_opened_observation_count")) > 0
        or _safe_int(agg.get("toggle_pair_observation_count")) > 0
    ):
        return "terrain_delta_supported"
    if _safe_int(agg.get("observation_count")) > 0:
        return "observed_no_platform_effect"
    return "insufficient_evidence"


def _platform_support_rank(state: str) -> int:
    order = {
        "score_closure_supported": 0,
        "goal_progress_supported": 1,
        "terrain_to_gravity_supported": 2,
        "terrain_delta_supported": 3,
        "observed_no_platform_effect": 4,
        "insufficient_evidence": 5,
    }
    return int(order.get(state, 99))


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


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
    "PLATFORM_DESTRUCTIBLE_TOPOLOGY_DIAGNOSTIC_SENSOR_SCHEMA_VERSION",
    "PLATFORM_DESTRUCTIBLE_TOPOLOGY_PACKET_SCHEMA_VERSION",
    "PLATFORM_DESTRUCTIBLE_TOPOLOGY_POSTERIOR_SCHEMA_VERSION",
    "PLATFORM_DESTRUCTIBLE_TOPOLOGY_CANDIDATE_MINER_SCHEMA_VERSION",
    "PLATFORM_DESTRUCTIBLE_TOPOLOGY_GUARDED_PROBE_OUTCOME_SCHEMA_VERSION",
    "PLATFORM_DESTRUCTIBLE_TOPOLOGY_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION",
    "PLATFORM_DESTRUCTIBLE_TOPOLOGY_ALL25_NON_REGRESSION_SCHEMA_VERSION",
    "build_platform_destructible_topology_packet",
    "summarize_platform_destructible_topology_all25_non_regression",
    "summarize_platform_destructible_topology_candidate_miner",
    "summarize_platform_destructible_topology_counterfactual_controls",
    "summarize_platform_destructible_topology_guarded_probe_outcomes",
    "summarize_platform_destructible_topology_packets",
    "summarize_platform_destructible_topology_posterior",
    "validate_platform_destructible_topology_packet",
]
