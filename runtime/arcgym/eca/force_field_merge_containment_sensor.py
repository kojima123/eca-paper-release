"""Frame-only force-field / merge / containment diagnostics.

The sensor summarizes observed area-effect clicks from public before/after
frames. It does not choose click sites, emit plans, or authorize ranking.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FORCE_FIELD_MERGE_CONTAINMENT_PACKET_SCHEMA_VERSION = (
    "eca_force_field_merge_containment_packet.v1"
)
FORCE_FIELD_MERGE_CONTAINMENT_DIAGNOSTIC_SENSOR_SCHEMA_VERSION = (
    "eca_force_field_merge_containment_diagnostic_sensor.v1"
)
FORCE_FIELD_MERGE_CONTAINMENT_POSTERIOR_SCHEMA_VERSION = (
    "eca_force_field_merge_containment_posterior.v1"
)
FORCE_FIELD_MERGE_CONTAINMENT_CANDIDATE_MINER_SCHEMA_VERSION = (
    "eca_force_field_merge_containment_candidate_miner.v1"
)
FORCE_FIELD_MERGE_CONTAINMENT_GUARDED_PROBE_OUTCOME_SCHEMA_VERSION = (
    "eca_force_field_merge_containment_guarded_probe_outcome.v1"
)
FORCE_FIELD_MERGE_CONTAINMENT_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION = (
    "eca_force_field_merge_containment_counterfactual_control.v1"
)
FORCE_FIELD_MERGE_CONTAINMENT_ALL25_NON_REGRESSION_SCHEMA_VERSION = (
    "eca_force_field_merge_containment_all25_non_regression.v1"
)
FORCE_FIELD_MERGE_CONTAINMENT_SCORE_TARGET_GAP_SCHEMA_VERSION = (
    "eca_force_field_merge_containment_score_target_gap.v1"
)
FORCE_FIELD_CONTAINMENT_SCORE_TARGET_RELATION_SCHEMA_VERSION = (
    "eca_force_field_containment_score_target_relation.v1"
)
FORCE_FIELD_SCORE_POSITIVE_COUNTERFACTUAL_PAIR_SEARCH_SCHEMA_VERSION = (
    "eca_force_field_score_positive_counterfactual_pair_search.v1"
)


def build_force_field_merge_containment_packet(
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
    """Build an observational force/merge/containment packet."""

    before = _as_frame(before_frame)
    after = _as_frame(after_frame) if after_frame is not None else before
    action_payload = dict(action_payload or {})
    selected_site = _selected_site(action_payload)
    before_topology = _component_topology(before)
    after_topology = _component_topology(after)
    movement = _component_movement(
        before_topology["components"],
        after_topology["components"],
        selected_site=selected_site,
    )
    containment = _containment_delta(before_topology, after_topology)
    component_count_delta = int(after_topology["target_component_count"]) - int(
        before_topology["target_component_count"]
    )
    color_count_delta = _counter_delta(
        before_topology["target_component_color_counts"],
        after_topology["target_component_color_counts"],
    )
    area_delta = _counter_delta(
        before_topology["target_component_area_by_color"],
        after_topology["target_component_area_by_color"],
    )
    force_drag_candidate = bool(
        _normalize_action_name(action_name) == "ACTION6"
        and selected_site is not None
        and int(movement["moved_towards_site_count"]) > 0
    )
    merge_candidate = bool(
        component_count_delta < 0
        or any(int(value) < 0 for value in color_count_delta.values())
        or _safe_int(area_delta.get("removed_area_total")) > 0
    )
    containment_progress_candidate = bool(
        int(containment["contained_component_count_delta"]) > 0
        or int(containment["moved_component_contained_after_count"]) > 0
    )
    score_closure_candidate = bool(
        int(score_delta) > 0
        and (force_drag_candidate or merge_candidate or containment_progress_candidate)
    )
    candidate_types: set[str] = set()
    if force_drag_candidate:
        candidate_types.add("force_drag")
    if merge_candidate:
        candidate_types.add("merge")
    if containment_progress_candidate:
        candidate_types.add("containment_progress")
    if score_closure_candidate:
        candidate_types.add("score_closure")

    return _sanitize(
        {
            "schema_version": FORCE_FIELD_MERGE_CONTAINMENT_PACKET_SCHEMA_VERSION,
            "sensor_family": "force_field_merge_containment_sensor",
            "game_id": str(game_id or "").strip().lower(),
            "state_hash": str(state_hash or _frame_hash(before)),
            "step": int(step),
            "action_name": _normalize_action_name(action_name),
            "action_payload": action_payload,
            "selected_site": list(selected_site) if selected_site is not None else None,
            "score_delta": int(score_delta),
            "background_color": int(before_topology["background_color"]),
            "before_target_component_count": int(
                before_topology["target_component_count"]
            ),
            "after_target_component_count": int(after_topology["target_component_count"]),
            "target_component_count_delta": int(component_count_delta),
            "target_component_color_count_delta": color_count_delta,
            "target_component_area_delta": area_delta,
            "containment_regions_before": before_topology["containment_regions"][:12],
            "containment_regions_after": after_topology["containment_regions"][:12],
            "containment_region_count_before": int(
                before_topology["containment_region_count"]
            ),
            "containment_region_count_after": int(
                after_topology["containment_region_count"]
            ),
            "component_movement": movement,
            "containment_delta": containment,
            "force_drag_candidate": force_drag_candidate,
            "merge_candidate": merge_candidate,
            "containment_progress_candidate": containment_progress_candidate,
            "score_closure_candidate": score_closure_candidate,
            "candidate_types": sorted(candidate_types),
            "diagnostic_only": True,
        }
    )


def validate_force_field_merge_containment_packet(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if (
        str(packet.get("schema_version", "") or "")
        != FORCE_FIELD_MERGE_CONTAINMENT_PACKET_SCHEMA_VERSION
    ):
        reasons.append("schema_version_mismatch")
    if str(packet.get("sensor_family", "") or "") != (
        "force_field_merge_containment_sensor"
    ):
        reasons.append("sensor_family_mismatch")
    before_count = _safe_int(packet.get("before_target_component_count"))
    after_count = _safe_int(packet.get("after_target_component_count"))
    if before_count < 0 or after_count < 0:
        reasons.append("invalid_component_count")
    selected_site = packet.get("selected_site")
    if (
        str(packet.get("action_name", "") or "") == "ACTION6"
        and selected_site is not None
        and not _is_xy(selected_site)
    ):
        reasons.append("invalid_selected_site")
    return {
        "schema_version": "eca_force_field_merge_containment_validation.v1",
        "accepted": not reasons,
        "rejected_reasons": reasons,
        "family": "force_field_merge_containment",
        "mechanic_state_type": _mechanic_state_type(packet),
        "goal_link_type": "force_merge_containment_progress",
        "candidate_types": list(packet.get("candidate_types") or []),
    }


def summarize_force_field_merge_containment_packets(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 12,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    candidate_type_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    moved_component_count_sum = 0
    moved_towards_site_count_sum = 0
    containment_delta_sum = 0
    score_delta_sum = 0

    for packet in packet_rows:
        validation = validate_force_field_merge_containment_packet(packet)
        accepted = bool(validation.get("accepted"))
        counts["accepted_packet_count" if accepted else "rejected_packet_count"] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["packet_count"] += 1
        movement = packet.get("component_movement")
        movement = movement if isinstance(movement, Mapping) else {}
        containment = packet.get("containment_delta")
        containment = containment if isinstance(containment, Mapping) else {}
        moved_count = _safe_int(movement.get("moved_component_count"))
        moved_towards = _safe_int(movement.get("moved_towards_site_count"))
        contained_delta = _safe_int(containment.get("contained_component_count_delta"))
        score_delta = _safe_int(packet.get("score_delta"))
        moved_component_count_sum += moved_count
        moved_towards_site_count_sum += moved_towards
        containment_delta_sum += contained_delta
        score_delta_sum += score_delta
        for candidate_type in list(packet.get("candidate_types") or []):
            candidate_type_counts[str(candidate_type)] += 1
        for field, count_name in (
            ("force_drag_candidate", "force_drag_candidate_count"),
            ("merge_candidate", "merge_candidate_count"),
            ("containment_progress_candidate", "containment_progress_candidate_count"),
            ("score_closure_candidate", "score_closure_candidate_count"),
        ):
            if bool(packet.get(field)):
                counts[count_name] += 1
                by_game[game_id][count_name] += 1
        if str(packet.get("action_name", "") or "") == "ACTION6":
            counts["action6_packet_count"] += 1
            by_game[game_id]["action6_packet_count"] += 1
        if len(samples) < sample_limit:
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "action_name": str(packet.get("action_name", "") or ""),
                    "selected_site": packet.get("selected_site"),
                    "candidate_types": list(packet.get("candidate_types") or []),
                    "moved_component_count": moved_count,
                    "moved_towards_site_count": moved_towards,
                    "target_component_count_delta": _safe_int(
                        packet.get("target_component_count_delta")
                    ),
                    "contained_component_count_delta": contained_delta,
                    "score_delta": score_delta,
                }
            )

    return _sanitize(
        {
            "schema_version": (
                FORCE_FIELD_MERGE_CONTAINMENT_DIAGNOSTIC_SENSOR_SCHEMA_VERSION
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
            "force_drag_candidate_count": int(
                counts.get("force_drag_candidate_count", 0)
            ),
            "merge_candidate_count": int(counts.get("merge_candidate_count", 0)),
            "containment_progress_candidate_count": int(
                counts.get("containment_progress_candidate_count", 0)
            ),
            "score_closure_candidate_count": int(
                counts.get("score_closure_candidate_count", 0)
            ),
            "moved_component_count_sum": int(moved_component_count_sum),
            "moved_towards_site_count_sum": int(moved_towards_site_count_sum),
            "contained_component_count_delta_sum": int(containment_delta_sum),
            "score_delta_sum": int(score_delta_sum),
            "candidate_type_counts": dict(sorted(candidate_type_counts.items())),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "samples": samples,
            "next_planner_contract": {
                "required_before_ranking": (
                    "force/merge/containment posterior with paired controls"
                ),
                "allowed_connection": (
                    "diagnostic_only_until_force_field_posterior_controls"
                ),
                "action_generation_allowed": False,
            },
        }
    )


def summarize_force_field_merge_containment_posterior(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Aggregate force/merge/containment packets into posterior hypotheses.

    This summary is diagnostic only. It records observed relations and the
    evidence level behind them, but does not mine candidates, rank actions, or
    authorize planner promotion.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    candidate_type_counts: Counter[str] = Counter()
    hypotheses: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    score_delta_sum = 0
    moved_component_count_sum = 0
    moved_towards_site_count_sum = 0
    contained_component_count_delta_sum = 0

    for index, packet in enumerate(packet_rows):
        validation = validate_force_field_merge_containment_packet(packet)
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
        by_game[game_id]["accepted_packet_count"] += 1
        candidate_types = sorted(
            str(row) for row in list(packet.get("candidate_types") or [])
        )
        candidate_type_set = set(candidate_types)
        for candidate_type in candidate_types:
            candidate_type_counts[candidate_type] += 1
            by_game[game_id][f"{candidate_type}_observation_count"] += 1

        movement = packet.get("component_movement")
        movement = movement if isinstance(movement, Mapping) else {}
        containment = packet.get("containment_delta")
        containment = containment if isinstance(containment, Mapping) else {}
        moved_count = _safe_int(movement.get("moved_component_count"))
        moved_towards = _safe_int(movement.get("moved_towards_site_count"))
        contained_delta = _safe_int(
            containment.get("contained_component_count_delta")
        )
        count_delta = _safe_int(packet.get("target_component_count_delta"))
        score_delta = _safe_int(packet.get("score_delta"))
        score_delta_sum += score_delta
        moved_component_count_sum += moved_count
        moved_towards_site_count_sum += moved_towards
        contained_component_count_delta_sum += contained_delta

        if "force_drag" in candidate_type_set:
            counts["force_drag_observation_count"] += 1
        if "merge" in candidate_type_set:
            counts["merge_observation_count"] += 1
        if "containment_progress" in candidate_type_set:
            counts["containment_progress_observation_count"] += 1
        if "score_closure" in candidate_type_set:
            counts["score_closure_observation_count"] += 1
        if score_delta > 0:
            counts["score_positive_observation_count"] += 1

        if candidate_types or score_delta > 0:
            mechanic_state = _mechanic_state_type(packet)
            hypothesis_key = _posterior_hypothesis_key(packet, mechanic_state)
            if hypothesis_key not in hypotheses:
                hypotheses[hypothesis_key] = {
                    "hypothesis_key": hypothesis_key,
                    "hypothesis_type": mechanic_state,
                    "candidate_types": [],
                    "game_ids": [],
                    "observation_count": 0,
                    "score_positive_observation_count": 0,
                    "score_delta_sum": 0,
                    "moved_component_count_sum": 0,
                    "moved_towards_site_count_sum": 0,
                    "target_component_count_delta_sum": 0,
                    "contained_component_count_delta_sum": 0,
                    "sample_selected_sites": [],
                    "sample_steps": [],
                }
            hypothesis = hypotheses[hypothesis_key]
            hypothesis["candidate_types"] = sorted(
                set(list(hypothesis.get("candidate_types") or []) + candidate_types)
            )
            hypothesis["game_ids"] = sorted(
                set(list(hypothesis.get("game_ids") or []) + [game_id])
            )
            hypothesis["observation_count"] = _safe_int(
                hypothesis.get("observation_count")
            ) + 1
            if score_delta > 0:
                hypothesis["score_positive_observation_count"] = _safe_int(
                    hypothesis.get("score_positive_observation_count")
                ) + 1
            hypothesis["score_delta_sum"] = _safe_int(
                hypothesis.get("score_delta_sum")
            ) + score_delta
            hypothesis["moved_component_count_sum"] = _safe_int(
                hypothesis.get("moved_component_count_sum")
            ) + moved_count
            hypothesis["moved_towards_site_count_sum"] = _safe_int(
                hypothesis.get("moved_towards_site_count_sum")
            ) + moved_towards
            hypothesis["target_component_count_delta_sum"] = _safe_int(
                hypothesis.get("target_component_count_delta_sum")
            ) + count_delta
            hypothesis["contained_component_count_delta_sum"] = _safe_int(
                hypothesis.get("contained_component_count_delta_sum")
            ) + contained_delta
            selected_site = packet.get("selected_site")
            if _is_xy(selected_site) and len(
                list(hypothesis.get("sample_selected_sites") or [])
            ) < 4:
                hypothesis["sample_selected_sites"] = list(
                    hypothesis.get("sample_selected_sites") or []
                ) + [[int(selected_site[0]), int(selected_site[1])]]
            if len(list(hypothesis.get("sample_steps") or [])) < 4:
                hypothesis["sample_steps"] = list(
                    hypothesis.get("sample_steps") or []
                ) + [_safe_int(packet.get("step"))]

        if len(samples) < sample_limit and (candidate_types or score_delta > 0):
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "action_name": str(packet.get("action_name", "") or ""),
                    "selected_site": packet.get("selected_site"),
                    "candidate_types": candidate_types,
                    "moved_component_count": moved_count,
                    "moved_towards_site_count": moved_towards,
                    "target_component_count_delta": count_delta,
                    "contained_component_count_delta": contained_delta,
                    "score_delta": score_delta,
                }
            )

    for hypothesis in hypotheses.values():
        candidate_type_set = set(str(row) for row in hypothesis.get("candidate_types") or [])
        if _safe_int(hypothesis.get("score_positive_observation_count")) > 0:
            posterior_state = "confirmed_score_closure"
        elif "containment_progress" in candidate_type_set:
            posterior_state = "supported_containment_relation"
        elif "force_drag" in candidate_type_set and "merge" in candidate_type_set:
            posterior_state = "supported_force_merge_relation"
        elif "merge" in candidate_type_set:
            posterior_state = "supported_merge_relation"
        elif "force_drag" in candidate_type_set:
            posterior_state = "supported_force_drag_relation"
        else:
            posterior_state = "insufficient_relation_evidence"
        hypothesis["posterior_state"] = posterior_state
        hypothesis["planner_connection_enabled"] = False
        hypothesis["ranking_enabled"] = False
        hypothesis["action_generation_allowed"] = False

    hypothesis_rows = sorted(
        hypotheses.values(),
        key=lambda row: (
            -_safe_int(row.get("score_delta_sum")),
            -_safe_int(row.get("score_positive_observation_count")),
            -_safe_int(row.get("contained_component_count_delta_sum")),
            -_safe_int(row.get("moved_towards_site_count_sum")),
            -_safe_int(row.get("observation_count")),
            str(row.get("hypothesis_key", "")),
        ),
    )
    supported_hypothesis_count = sum(
        1
        for row in hypothesis_rows
        if str(row.get("posterior_state", "")).startswith("supported_")
        or str(row.get("posterior_state", "")) == "confirmed_score_closure"
    )
    score_closure_hypothesis_count = sum(
        1
        for row in hypothesis_rows
        if str(row.get("posterior_state", "")) == "confirmed_score_closure"
    )
    containment_hypothesis_count = sum(
        1
        for row in hypothesis_rows
        if str(row.get("posterior_state", ""))
        == "supported_containment_relation"
    )
    force_merge_hypothesis_count = sum(
        1
        for row in hypothesis_rows
        if str(row.get("posterior_state", ""))
        == "supported_force_merge_relation"
    )
    if score_closure_hypothesis_count > 0:
        promotion_state = "force_field_merge_containment_score_closure_posterior"
    elif containment_hypothesis_count > 0:
        promotion_state = "force_field_merge_containment_containment_posterior"
    elif force_merge_hypothesis_count > 0:
        promotion_state = "force_field_merge_containment_force_merge_posterior"
    elif supported_hypothesis_count > 0:
        promotion_state = "force_field_merge_containment_effect_posterior"
    elif int(counts.get("accepted_packet_count", 0)) > 0:
        promotion_state = "force_field_merge_containment_insufficient_effect_evidence"
    else:
        promotion_state = "force_field_merge_containment_no_evidence"

    return _sanitize(
        {
            "schema_version": FORCE_FIELD_MERGE_CONTAINMENT_POSTERIOR_SCHEMA_VERSION,
            "default_off": True,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "accepted_packet_count": int(counts.get("accepted_packet_count", 0)),
            "rejected_packet_count": int(counts.get("rejected_packet_count", 0)),
            "force_drag_observation_count": int(
                counts.get("force_drag_observation_count", 0)
            ),
            "merge_observation_count": int(counts.get("merge_observation_count", 0)),
            "containment_progress_observation_count": int(
                counts.get("containment_progress_observation_count", 0)
            ),
            "score_closure_observation_count": int(
                counts.get("score_closure_observation_count", 0)
            ),
            "score_positive_observation_count": int(
                counts.get("score_positive_observation_count", 0)
            ),
            "score_delta_sum": int(score_delta_sum),
            "moved_component_count_sum": int(moved_component_count_sum),
            "moved_towards_site_count_sum": int(moved_towards_site_count_sum),
            "contained_component_count_delta_sum": int(
                contained_component_count_delta_sum
            ),
            "posterior_hypothesis_count": int(len(hypothesis_rows)),
            "supported_hypothesis_count": int(supported_hypothesis_count),
            "score_closure_hypothesis_count": int(score_closure_hypothesis_count),
            "containment_hypothesis_count": int(containment_hypothesis_count),
            "force_merge_hypothesis_count": int(force_merge_hypothesis_count),
            "candidate_type_counts": dict(sorted(candidate_type_counts.items())),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "hypotheses": hypothesis_rows[:sample_limit],
            "rejected": rejected[:sample_limit],
            "samples": samples,
            "promotion_state": promotion_state,
            "next_runtime_contract": {
                "required_before_candidate_miner": (
                    "candidate mining requires posterior hypotheses plus "
                    "scoreless controls that distinguish useful force/merge "
                    "effects from harmless area motion"
                ),
                "allowed_connection": "posterior_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_force_field_merge_containment_candidate_miner(
    packets: Iterable[Mapping[str, Any]],
    *,
    posterior_summary: Mapping[str, Any] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Mine guarded-probe candidate templates from force/merge posterior.

    The miner emits relation templates only. It deliberately omits direct
    action sites and leaves runtime execution/ranking disabled.
    """

    posterior = (
        dict(posterior_summary)
        if isinstance(posterior_summary, Mapping)
        else summarize_force_field_merge_containment_posterior(
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
        posterior_state = str(hypothesis.get("posterior_state", "") or "")
        candidate_types = [
            str(row) for row in list(hypothesis.get("candidate_types") or [])
        ]
        game_ids = [
            str(game_id)
            for game_id in list(hypothesis.get("game_ids") or [])
            if str(game_id)
        ]
        game_id = game_ids[0] if game_ids else "unknown"
        by_game.setdefault(game_id, Counter())
        if posterior_state == "confirmed_score_closure":
            candidate_kind = "force_field_score_closure_replay_candidate"
        elif posterior_state == "supported_containment_relation":
            candidate_kind = "force_field_containment_progress_retest_candidate"
        elif posterior_state == "supported_force_merge_relation":
            candidate_kind = "force_field_force_merge_retest_candidate"
        elif posterior_state == "supported_force_drag_relation":
            candidate_kind = "force_field_force_drag_retest_candidate"
        elif posterior_state == "supported_merge_relation":
            candidate_kind = "force_field_merge_relation_control_candidate"
        else:
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "hypothesis_index": int(index),
                        "hypothesis_key": hypothesis.get("hypothesis_key"),
                        "rejected_reason": "unsupported_posterior_state",
                        "posterior_state": posterior_state,
                    }
                )
            continue

        source_key = str(hypothesis.get("hypothesis_key", "") or "")
        source_bucket = _source_site_bucket_from_hypothesis_key(source_key)
        candidate_key = _stable_candidate_key(
            "force_field_merge_containment_candidate",
            source_key,
            candidate_kind,
        )
        candidate = {
            "candidate_key": candidate_key,
            "candidate_kind": candidate_kind,
            "source_hypothesis_key": source_key,
            "source_posterior_state": posterior_state,
            "source_relation_bucket": source_bucket,
            "candidate_types": candidate_types,
            "game_ids": game_ids,
            "observation_count": _safe_int(hypothesis.get("observation_count")),
            "score_delta_sum": _safe_int(hypothesis.get("score_delta_sum")),
            "score_positive_observation_count": _safe_int(
                hypothesis.get("score_positive_observation_count")
            ),
            "moved_towards_site_count_sum": _safe_int(
                hypothesis.get("moved_towards_site_count_sum")
            ),
            "target_component_count_delta_sum": _safe_int(
                hypothesis.get("target_component_count_delta_sum")
            ),
            "contained_component_count_delta_sum": _safe_int(
                hypothesis.get("contained_component_count_delta_sum")
            ),
            "requires_guarded_probe": True,
            "required_controls": [
                "scoreless_no_effect_control",
                "wrong_relation_bucket_control",
                "unchanged_component_count_control",
            ],
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
        candidates.append(candidate)
        by_game[game_id]["candidate_count"] += 1
        by_game[game_id][f"candidate_kind:{candidate_kind}"] += 1
        by_kind.setdefault(candidate_kind, Counter())
        by_kind[candidate_kind]["candidate_count"] += 1
        if _safe_int(candidate.get("score_positive_observation_count")) > 0:
            by_kind[candidate_kind]["score_positive_candidate_count"] += 1

    candidates = sorted(
        candidates,
        key=lambda row: (
            -_safe_int(row.get("score_positive_observation_count")),
            -_safe_int(row.get("moved_towards_site_count_sum")),
            -abs(_safe_int(row.get("target_component_count_delta_sum"))),
            str(row.get("candidate_key", "")),
        ),
    )
    kind_counts = Counter(str(row.get("candidate_kind", "")) for row in candidates)
    score_positive_candidate_count = sum(
        1 for row in candidates if _safe_int(row.get("score_positive_observation_count")) > 0
    )
    if score_positive_candidate_count > 0:
        promotion_state = (
            "force_field_merge_containment_candidate_miner_score_closure_supported"
        )
    elif candidates:
        promotion_state = "force_field_merge_containment_candidate_miner_supported"
    else:
        promotion_state = "force_field_merge_containment_candidate_miner_no_candidates"

    return _sanitize(
        {
            "schema_version": (
                FORCE_FIELD_MERGE_CONTAINMENT_CANDIDATE_MINER_SCHEMA_VERSION
            ),
            "default_off": True,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "posterior_promotion_state": posterior.get("promotion_state"),
            "posterior_hypothesis_count": _safe_int(
                posterior.get("posterior_hypothesis_count")
            ),
            "supported_hypothesis_count": _safe_int(
                posterior.get("supported_hypothesis_count")
            ),
            "candidate_count": int(len(candidates)),
            "score_positive_candidate_count": int(score_positive_candidate_count),
            "force_field_score_closure_replay_candidate_count": int(
                kind_counts.get("force_field_score_closure_replay_candidate", 0)
            ),
            "force_field_containment_progress_retest_candidate_count": int(
                kind_counts.get("force_field_containment_progress_retest_candidate", 0)
            ),
            "force_field_force_merge_retest_candidate_count": int(
                kind_counts.get("force_field_force_merge_retest_candidate", 0)
            ),
            "force_field_force_drag_retest_candidate_count": int(
                kind_counts.get("force_field_force_drag_retest_candidate", 0)
            ),
            "force_field_merge_relation_control_candidate_count": int(
                kind_counts.get("force_field_merge_relation_control_candidate", 0)
            ),
            "rejected_candidate_count": int(len(rejected)),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_candidate_kind": {
                key: dict(sorted(counter.items()))
                for key, counter in sorted(by_kind.items())
            },
            "candidates": candidates[:sample_limit],
            "rejected": rejected[:sample_limit],
            "promotion_state": promotion_state,
            "next_runtime_contract": {
                "required_before_guarded_probe": (
                    "candidate templates require guarded runtime admission and "
                    "paired scoreless controls before execution"
                ),
                "allowed_connection": "candidate_templates_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_force_field_merge_containment_guarded_probe_outcomes(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize outcomes from default-off force-field guarded probes."""

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
                "force_field_merge_containment_guarded_probe_preflight"
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
                "force_field_merge_containment_guarded_probe"
            )
            is_guarded = is_guarded or str(unsolved.get("source", "") or "") == (
                "force_field_merge_containment_guarded_probe"
            )
            if not is_guarded:
                continue

            packet = item.get("force_field_merge_containment_packet")
            packet = packet if isinstance(packet, Mapping) else {}
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            movement = packet.get("component_movement")
            movement = movement if isinstance(movement, Mapping) else {}
            containment = packet.get("containment_delta")
            containment = containment if isinstance(containment, Mapping) else {}
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
            changed_pixel_count = _safe_int(
                outcome.get("changed_pixel_count"),
                _safe_int(packet.get("changed_cell_count")),
            )
            moved_count = _safe_int(movement.get("moved_component_count"))
            moved_towards = _safe_int(movement.get("moved_towards_site_count"))
            count_delta = _safe_int(packet.get("target_component_count_delta"))
            contained_delta = _safe_int(
                containment.get("contained_component_count_delta")
            )
            candidate_types = [str(row) for row in list(packet.get("candidate_types") or [])]

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
            counts["changed_pixel_count_sum"] += int(changed_pixel_count)
            counts["moved_component_count_sum"] += int(moved_count)
            counts["moved_towards_site_count_sum"] += int(moved_towards)
            counts["target_component_count_delta_sum"] += int(count_delta)
            counts["contained_component_count_delta_sum"] += int(contained_delta)
            if score_delta > 0:
                outcome_state = "score_positive"
                counts["score_positive_execution_count"] += 1
                by_game[game_id]["score_positive_execution_count"] += 1
                by_candidate_kind[candidate_kind]["score_positive_execution_count"] += 1
            elif contained_delta > 0 or "containment_progress" in candidate_types:
                outcome_state = "scoreless_containment_progress"
                counts["scoreless_containment_progress_execution_count"] += 1
                by_game[game_id][
                    "scoreless_containment_progress_execution_count"
                ] += 1
                by_candidate_kind[candidate_kind][
                    "scoreless_containment_progress_execution_count"
                ] += 1
            elif moved_towards > 0 or "force_drag" in candidate_types:
                outcome_state = "scoreless_force_drag"
                counts["scoreless_force_drag_execution_count"] += 1
                by_game[game_id]["scoreless_force_drag_execution_count"] += 1
                by_candidate_kind[candidate_kind][
                    "scoreless_force_drag_execution_count"
                ] += 1
            elif count_delta < 0 or "merge" in candidate_types:
                outcome_state = "scoreless_merge"
                counts["scoreless_merge_execution_count"] += 1
                by_game[game_id]["scoreless_merge_execution_count"] += 1
                by_candidate_kind[candidate_kind][
                    "scoreless_merge_execution_count"
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
                        "source_relation_bucket": str(
                            unsolved.get("source_relation_bucket", "") or ""
                        ),
                        "outcome_state": outcome_state,
                        "candidate_types": candidate_types,
                        "score_delta": int(score_delta),
                        "changed_pixel_count": int(changed_pixel_count),
                        "moved_component_count": int(moved_count),
                        "moved_towards_site_count": int(moved_towards),
                        "target_component_count_delta": int(count_delta),
                        "contained_component_count_delta": int(contained_delta),
                    }
                )

    execution_count = int(counts.get("execution_count", 0))
    if execution_count <= 0 and int(counts.get("preflight_count", 0)) > 0:
        promotion_state = "force_field_merge_containment_guarded_probe_preflight_only"
    elif execution_count <= 0:
        promotion_state = "force_field_merge_containment_guarded_probe_no_execution"
    elif int(counts.get("score_positive_execution_count", 0)) > 0:
        promotion_state = "force_field_merge_containment_guarded_probe_score_positive"
    elif int(counts.get("scoreless_containment_progress_execution_count", 0)) > 0:
        promotion_state = (
            "force_field_merge_containment_guarded_probe_scoreless_containment"
        )
    elif int(counts.get("scoreless_force_drag_execution_count", 0)) > 0:
        promotion_state = (
            "force_field_merge_containment_guarded_probe_scoreless_force_drag"
        )
    elif int(counts.get("scoreless_merge_execution_count", 0)) > 0:
        promotion_state = "force_field_merge_containment_guarded_probe_scoreless_merge"
    elif int(counts.get("scoreless_visible_change_execution_count", 0)) > 0:
        promotion_state = "force_field_merge_containment_guarded_probe_visible_change"
    else:
        promotion_state = "force_field_merge_containment_guarded_probe_scoreless_no_effect"

    return _sanitize(
        {
            "schema_version": (
                FORCE_FIELD_MERGE_CONTAINMENT_GUARDED_PROBE_OUTCOME_SCHEMA_VERSION
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
            "scoreless_containment_progress_execution_count": int(
                counts.get("scoreless_containment_progress_execution_count", 0)
            ),
            "scoreless_force_drag_execution_count": int(
                counts.get("scoreless_force_drag_execution_count", 0)
            ),
            "scoreless_merge_execution_count": int(
                counts.get("scoreless_merge_execution_count", 0)
            ),
            "scoreless_visible_change_execution_count": int(
                counts.get("scoreless_visible_change_execution_count", 0)
            ),
            "scoreless_no_effect_execution_count": int(
                counts.get("scoreless_no_effect_execution_count", 0)
            ),
            "score_delta_sum": int(counts.get("score_delta_sum", 0)),
            "changed_pixel_count_sum": int(counts.get("changed_pixel_count_sum", 0)),
            "moved_component_count_sum": int(
                counts.get("moved_component_count_sum", 0)
            ),
            "moved_towards_site_count_sum": int(
                counts.get("moved_towards_site_count_sum", 0)
            ),
            "target_component_count_delta_sum": int(
                counts.get("target_component_count_delta_sum", 0)
            ),
            "contained_component_count_delta_sum": int(
                counts.get("contained_component_count_delta_sum", 0)
            ),
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
                    "force/merge/containment closure from scoreless force, "
                    "merge, containment, and no-effect probes before ranking changes"
                ),
                "allowed_connection": "guarded_probe_outcome_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_force_field_merge_containment_counterfactual_controls(
    rows: Iterable[Mapping[str, Any]],
    *,
    outcome_summary: Mapping[str, Any] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Check whether force-field guarded outcomes have positive controls."""

    outcome = (
        dict(outcome_summary)
        if isinstance(outcome_summary, Mapping)
        else summarize_force_field_merge_containment_guarded_probe_outcomes(
            rows,
            sample_limit=sample_limit,
        )
    )
    score_positive = _safe_int(outcome.get("score_positive_execution_count"))
    scoreless_containment = _safe_int(
        outcome.get("scoreless_containment_progress_execution_count")
    )
    scoreless_force = _safe_int(outcome.get("scoreless_force_drag_execution_count"))
    scoreless_merge = _safe_int(outcome.get("scoreless_merge_execution_count"))
    scoreless_visible = _safe_int(
        outcome.get("scoreless_visible_change_execution_count")
    )
    scoreless_no_effect = _safe_int(outcome.get("scoreless_no_effect_execution_count"))
    scoreless_control = int(
        scoreless_containment
        + scoreless_force
        + scoreless_merge
        + scoreless_visible
        + scoreless_no_effect
    )
    counterfactual_pair = int(score_positive > 0 and scoreless_control > 0)
    margin = int(score_positive - scoreless_control)
    positive_control_margin = int(counterfactual_pair > 0 and margin >= 0)

    if score_positive <= 0 and scoreless_control > 0:
        promotion_state = (
            "force_field_merge_containment_counterfactual_control_missing_positive"
        )
        blocked = True
    elif score_positive > 0 and scoreless_control > 0:
        promotion_state = (
            "force_field_merge_containment_counterfactual_control_supported"
        )
        blocked = False
    elif score_positive > 0:
        promotion_state = (
            "force_field_merge_containment_counterfactual_control_missing_control"
        )
        blocked = True
    elif _safe_int(outcome.get("execution_count")) > 0:
        promotion_state = (
            "force_field_merge_containment_counterfactual_control_no_effect_evidence"
        )
        blocked = True
    else:
        promotion_state = (
            "force_field_merge_containment_counterfactual_control_no_execution"
        )
        blocked = True

    return _sanitize(
        {
            "schema_version": (
                FORCE_FIELD_MERGE_CONTAINMENT_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION
            ),
            "default_off": True,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "outcome_promotion_state": outcome.get("promotion_state"),
            "execution_count": _safe_int(outcome.get("execution_count")),
            "runtime_outcome_count": _safe_int(outcome.get("runtime_outcome_count")),
            "score_positive_execution_count": int(score_positive),
            "scoreless_containment_progress_execution_count": int(
                scoreless_containment
            ),
            "scoreless_force_drag_execution_count": int(scoreless_force),
            "scoreless_merge_execution_count": int(scoreless_merge),
            "scoreless_visible_change_execution_count": int(scoreless_visible),
            "scoreless_no_effect_execution_count": int(scoreless_no_effect),
            "scoreless_control_execution_count": int(scoreless_control),
            "counterfactual_pair_count": int(counterfactual_pair),
            "counterfactual_margin": int(margin),
            "positive_control_margin_count": int(positive_control_margin),
            "planner_promotion_blocked": bool(blocked),
            "promotion_state": promotion_state,
            "by_game": dict(outcome.get("by_game") or {}),
            "by_candidate_kind": dict(outcome.get("by_candidate_kind") or {}),
            "samples": list(outcome.get("samples") or [])[:sample_limit],
            "next_runtime_contract": {
                "required_before_planner_promotion": (
                    "score-positive and scoreless force-field controls must "
                    "coexist before any ranking or planner promotion"
                ),
                "allowed_connection": "counterfactual_control_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_force_field_merge_containment_all25_non_regression(
    run_rows: Iterable[Mapping[str, Any]],
    *,
    expected_score_sum: int | None = None,
    expected_by_game: Mapping[str, Any] | None = None,
    require_no_game_conditioned_primitives: bool = True,
    counterfactual_summary: Mapping[str, Any] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Check all25 preservation while force-field diagnostics are enabled."""

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
        reason_counts["force_field_all25_non_regression_supported"] += 1
        promotion_state = "force_field_merge_containment_all25_non_regression_supported"
    elif rows:
        promotion_state = "force_field_merge_containment_all25_non_regression_blocked"
    else:
        promotion_state = "force_field_merge_containment_all25_non_regression_no_rows"

    counterfactual = (
        dict(counterfactual_summary)
        if isinstance(counterfactual_summary, Mapping)
        else {}
    )
    return _sanitize(
        {
            "schema_version": (
                FORCE_FIELD_MERGE_CONTAINMENT_ALL25_NON_REGRESSION_SCHEMA_VERSION
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


def summarize_force_field_merge_containment_score_target_gap(
    *,
    diagnostic_summary: Mapping[str, Any] | None = None,
    posterior_summary: Mapping[str, Any] | None = None,
    candidate_miner_summary: Mapping[str, Any] | None = None,
    relation_summary: Mapping[str, Any] | None = None,
    outcome_summary: Mapping[str, Any] | None = None,
    counterfactual_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize the missing bridge from force/merge evidence to score targets."""

    diagnostic = (
        dict(diagnostic_summary) if isinstance(diagnostic_summary, Mapping) else {}
    )
    posterior = (
        dict(posterior_summary) if isinstance(posterior_summary, Mapping) else {}
    )
    miner = (
        dict(candidate_miner_summary)
        if isinstance(candidate_miner_summary, Mapping)
        else {}
    )
    relation = (
        dict(relation_summary) if isinstance(relation_summary, Mapping) else {}
    )
    outcome = dict(outcome_summary) if isinstance(outcome_summary, Mapping) else {}
    control = (
        dict(counterfactual_summary)
        if isinstance(counterfactual_summary, Mapping)
        else {}
    )

    force_drag_candidate_count = _safe_int(
        diagnostic.get("force_drag_candidate_count")
    )
    merge_candidate_count = _safe_int(diagnostic.get("merge_candidate_count"))
    containment_candidate_count = _safe_int(
        diagnostic.get("containment_progress_candidate_count")
    ) + _safe_int(
        miner.get("force_field_containment_progress_retest_candidate_count")
    )
    score_closure_candidate_count = _safe_int(
        diagnostic.get("score_closure_candidate_count")
    ) + _safe_int(miner.get("force_field_score_closure_replay_candidate_count"))
    force_merge_hypothesis_count = _safe_int(
        posterior.get("force_merge_hypothesis_count")
    )
    score_closure_hypothesis_count = _safe_int(
        posterior.get("score_closure_hypothesis_count")
    )
    relation_score_target_count = _safe_int(
        relation.get("score_target_relation_packet_count")
    )
    relation_force_to_target_count = _safe_int(
        relation.get("force_to_score_target_relation_packet_count")
    )
    score_positive_count = max(
        _safe_int(posterior.get("score_positive_observation_count")),
        _safe_int(miner.get("score_positive_candidate_count")),
        _safe_int(relation.get("score_positive_relation_packet_count")),
        _safe_int(outcome.get("score_positive_execution_count")),
        _safe_int(control.get("score_positive_execution_count")),
    )
    execution_count = _safe_int(outcome.get("execution_count"))
    scoreless_control_count = _safe_int(
        control.get("scoreless_control_execution_count")
    )
    force_or_merge_evidence_count = (
        force_drag_candidate_count
        + merge_candidate_count
        + force_merge_hypothesis_count
    )
    score_target_evidence_count = (
        containment_candidate_count
        + score_closure_candidate_count
        + score_closure_hypothesis_count
        + relation_score_target_count
        + relation_force_to_target_count
        + score_positive_count
    )

    if score_positive_count > 0:
        promotion_state = "force_field_score_target_gap_score_positive_supported"
        recommended_next_sensor = "score_positive_counterfactual_pairing"
    elif force_or_merge_evidence_count <= 0:
        promotion_state = "force_field_score_target_gap_insufficient_force_merge_evidence"
        recommended_next_sensor = "force_merge_relation_evidence"
    elif score_target_evidence_count <= 0:
        promotion_state = (
            "force_field_score_target_gap_missing_containment_score_target_relation"
        )
        recommended_next_sensor = "force_field_containment_score_target_relation_sensor"
    elif execution_count > 0 or scoreless_control_count > 0:
        promotion_state = "force_field_score_target_gap_scoreless_probe_only"
        recommended_next_sensor = "score_positive_counterfactual_pair_search"
    elif relation_force_to_target_count > 0:
        promotion_state = (
            "force_field_score_target_gap_relation_supported_missing_score_positive"
        )
        recommended_next_sensor = "score_positive_counterfactual_pair_search"
    else:
        promotion_state = "force_field_score_target_gap_unexecuted_target_candidates"
        recommended_next_sensor = "guarded_target_candidate_execution"

    planner_blocked = bool(score_positive_count <= 0)
    return _sanitize(
        {
            "schema_version": (
                FORCE_FIELD_MERGE_CONTAINMENT_SCORE_TARGET_GAP_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "default_off": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "force_drag_candidate_count": int(force_drag_candidate_count),
            "merge_candidate_count": int(merge_candidate_count),
            "force_merge_hypothesis_count": int(force_merge_hypothesis_count),
            "force_or_merge_evidence_count": int(force_or_merge_evidence_count),
            "containment_candidate_count": int(containment_candidate_count),
            "score_closure_candidate_count": int(score_closure_candidate_count),
            "score_closure_hypothesis_count": int(score_closure_hypothesis_count),
            "relation_score_target_packet_count": int(relation_score_target_count),
            "relation_force_to_score_target_packet_count": int(
                relation_force_to_target_count
            ),
            "score_target_evidence_count": int(score_target_evidence_count),
            "candidate_count": _safe_int(miner.get("candidate_count")),
            "execution_count": int(execution_count),
            "score_positive_execution_count": _safe_int(
                outcome.get("score_positive_execution_count")
            ),
            "score_positive_observation_or_candidate_count": int(
                score_positive_count
            ),
            "scoreless_control_execution_count": int(scoreless_control_count),
            "counterfactual_pair_count": _safe_int(
                control.get("counterfactual_pair_count")
            ),
            "planner_promotion_blocked": bool(planner_blocked),
            "promotion_state": promotion_state,
            "recommended_next_sensor": recommended_next_sensor,
            "claim_boundary": {
                "score_lift_claim": False,
                "planner_connection_enabled": False,
                "candidate_generation_enabled": False,
                "ranking_enabled": False,
                "action_generation_allowed": False,
            },
        }
    )


def summarize_force_field_containment_score_target_relations(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Relate force-field observations to visible containment target regions."""

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    relation_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    for packet in packet_rows:
        validation = validate_force_field_merge_containment_packet(packet)
        game_id = str(packet.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["packet_count"] += 1
        if not bool(validation.get("accepted")):
            counts["rejected_packet_count"] += 1
            by_game[game_id]["rejected_packet_count"] += 1
            continue
        counts["accepted_packet_count"] += 1
        by_game[game_id]["accepted_packet_count"] += 1

        candidate_types = set(str(row) for row in list(packet.get("candidate_types") or []))
        force_or_merge = bool(
            "force_drag" in candidate_types
            or "merge" in candidate_types
            or bool(packet.get("force_drag_candidate"))
            or bool(packet.get("merge_candidate"))
        )
        if force_or_merge:
            counts["force_or_merge_packet_count"] += 1
            by_game[game_id]["force_or_merge_packet_count"] += 1
        score_delta = _safe_int(packet.get("score_delta"))
        if score_delta > 0:
            counts["score_positive_packet_count"] += 1
            by_game[game_id]["score_positive_packet_count"] += 1

        regions = [
            dict(row)
            for row in list(packet.get("containment_regions_after") or [])
            if isinstance(row, Mapping) and _is_bbox(row.get("bbox"))
        ]
        if not regions:
            regions = [
                dict(row)
                for row in list(packet.get("containment_regions_before") or [])
                if isinstance(row, Mapping) and _is_bbox(row.get("bbox"))
            ]
        if regions:
            counts["packet_with_containment_region_count"] += 1
            by_game[game_id]["packet_with_containment_region_count"] += 1

        selected_site = packet.get("selected_site")
        selected_relation = _nearest_region_relation(selected_site, regions)
        if selected_relation:
            relation = str(selected_relation.get("relation", "") or "")
            relation_counts[f"selected_site_{relation}"] += 1
            by_game[game_id][f"selected_site_{relation}_count"] += 1
            if relation in {"inside", "near"}:
                counts["selected_site_target_relation_count"] += 1
                by_game[game_id]["selected_site_target_relation_count"] += 1
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "step": _safe_int(packet.get("step")),
                        "relation_source": "selected_site_to_region",
                        "selected_site": list(selected_site) if _is_xy(selected_site) else [],
                        "relation": relation,
                        "region_bucket": selected_relation.get("region_bucket"),
                        "distance_to_region": selected_relation.get("distance"),
                        "candidate_types": sorted(candidate_types),
                        "score_delta": score_delta,
                    }
                )

        movement = packet.get("component_movement")
        movement = movement if isinstance(movement, Mapping) else {}
        moved_components = [
            dict(row)
            for row in list(movement.get("moved_components") or [])
            if isinstance(row, Mapping)
        ]
        moved_towards_region = 0
        moved_into_region = 0
        for component in moved_components:
            before = component.get("before_centroid")
            after = component.get("after_centroid")
            before_rel = _nearest_region_relation(before, regions)
            after_rel = _nearest_region_relation(after, regions)
            if not before_rel or not after_rel:
                continue
            before_distance = float(before_rel.get("distance", 0.0) or 0.0)
            after_distance = float(after_rel.get("distance", 0.0) or 0.0)
            if after_distance + 0.01 < before_distance:
                moved_towards_region += 1
            if before_distance > 0.01 and after_distance <= 0.01:
                moved_into_region += 1
            if len(relation_rows) < sample_limit and (
                after_distance + 0.01 < before_distance or after_distance <= 0.01
            ):
                relation_rows.append(
                    {
                        "game_id": game_id,
                        "step": _safe_int(packet.get("step")),
                        "component_color": _safe_int(component.get("color")),
                        "before_centroid": list(before) if _is_xy(before) else [],
                        "after_centroid": list(after) if _is_xy(after) else [],
                        "before_distance_to_region": round(before_distance, 3),
                        "after_distance_to_region": round(after_distance, 3),
                        "region_bucket": after_rel.get("region_bucket"),
                        "candidate_types": sorted(candidate_types),
                        "score_delta": score_delta,
                    }
                )
        if moved_towards_region > 0:
            counts["moved_component_towards_region_packet_count"] += 1
            counts["moved_component_towards_region_count"] += int(moved_towards_region)
            by_game[game_id]["moved_component_towards_region_packet_count"] += 1
            by_game[game_id]["moved_component_towards_region_count"] += int(
                moved_towards_region
            )
        if moved_into_region > 0:
            counts["moved_component_into_region_packet_count"] += 1
            counts["moved_component_into_region_count"] += int(moved_into_region)
            by_game[game_id]["moved_component_into_region_packet_count"] += 1
            by_game[game_id]["moved_component_into_region_count"] += int(
                moved_into_region
            )

        containment = packet.get("containment_delta")
        containment = containment if isinstance(containment, Mapping) else {}
        contained_delta = _safe_int(containment.get("contained_component_count_delta"))
        contained_after = _safe_int(
            containment.get("moved_component_contained_after_count")
        )
        if contained_delta > 0 or contained_after > 0:
            counts["direct_containment_progress_packet_count"] += 1
            by_game[game_id]["direct_containment_progress_packet_count"] += 1

        score_target_relation = bool(
            selected_relation
            and str(selected_relation.get("relation", "") or "") in {"inside", "near"}
        ) or moved_towards_region > 0 or moved_into_region > 0 or contained_delta > 0 or contained_after > 0
        if score_target_relation:
            counts["score_target_relation_packet_count"] += 1
            by_game[game_id]["score_target_relation_packet_count"] += 1
            if force_or_merge:
                counts["force_to_score_target_relation_packet_count"] += 1
                by_game[game_id]["force_to_score_target_relation_packet_count"] += 1
            if score_delta > 0:
                counts["score_positive_relation_packet_count"] += 1
                by_game[game_id]["score_positive_relation_packet_count"] += 1

    if int(counts.get("score_positive_relation_packet_count", 0)) > 0:
        promotion_state = (
            "force_field_containment_score_target_relation_score_positive_supported"
        )
    elif int(counts.get("force_to_score_target_relation_packet_count", 0)) > 0:
        promotion_state = "force_field_containment_score_target_relation_supported"
    elif int(counts.get("packet_with_containment_region_count", 0)) > 0:
        promotion_state = (
            "force_field_containment_score_target_relation_regions_unlinked"
        )
    elif int(counts.get("accepted_packet_count", 0)) > 0:
        promotion_state = (
            "force_field_containment_score_target_relation_no_regions"
        )
    else:
        promotion_state = "force_field_containment_score_target_relation_no_evidence"

    return _sanitize(
        {
            "schema_version": (
                FORCE_FIELD_CONTAINMENT_SCORE_TARGET_RELATION_SCHEMA_VERSION
            ),
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
            "force_or_merge_packet_count": int(
                counts.get("force_or_merge_packet_count", 0)
            ),
            "packet_with_containment_region_count": int(
                counts.get("packet_with_containment_region_count", 0)
            ),
            "selected_site_target_relation_count": int(
                counts.get("selected_site_target_relation_count", 0)
            ),
            "moved_component_towards_region_packet_count": int(
                counts.get("moved_component_towards_region_packet_count", 0)
            ),
            "moved_component_towards_region_count": int(
                counts.get("moved_component_towards_region_count", 0)
            ),
            "moved_component_into_region_packet_count": int(
                counts.get("moved_component_into_region_packet_count", 0)
            ),
            "moved_component_into_region_count": int(
                counts.get("moved_component_into_region_count", 0)
            ),
            "direct_containment_progress_packet_count": int(
                counts.get("direct_containment_progress_packet_count", 0)
            ),
            "score_target_relation_packet_count": int(
                counts.get("score_target_relation_packet_count", 0)
            ),
            "force_to_score_target_relation_packet_count": int(
                counts.get("force_to_score_target_relation_packet_count", 0)
            ),
            "score_positive_relation_packet_count": int(
                counts.get("score_positive_relation_packet_count", 0)
            ),
            "selected_site_relation_counts": dict(sorted(relation_counts.items())),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "relation_samples": relation_rows[:sample_limit],
            "samples": samples[:sample_limit],
            "promotion_state": promotion_state,
            "next_runtime_contract": {
                "required_before_planner_promotion": (
                    "score-positive relation packets plus paired scoreless controls"
                ),
                "allowed_connection": "containment_score_target_relation_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_force_field_score_positive_counterfactual_pair_search_candidates(
    packets: Iterable[Mapping[str, Any]],
    *,
    relation_summary: Mapping[str, Any] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Mine bucket-level force-field candidates that advance toward score targets."""

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    relation = (
        dict(relation_summary)
        if isinstance(relation_summary, Mapping)
        else summarize_force_field_containment_score_target_relations(
            packet_rows,
            sample_limit=max(int(sample_limit), 128),
        )
    )
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_kind: dict[str, Counter[str]] = {}
    candidates_by_key: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []

    for packet in packet_rows:
        validation = validate_force_field_merge_containment_packet(packet)
        game_id = str(packet.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["packet_count"] += 1
        if not bool(validation.get("accepted")):
            counts["rejected_packet_count"] += 1
            by_game[game_id]["rejected_packet_count"] += 1
            continue
        counts["accepted_packet_count"] += 1
        by_game[game_id]["accepted_packet_count"] += 1

        candidate_types = set(str(row) for row in list(packet.get("candidate_types") or []))
        force_or_merge = bool(
            "force_drag" in candidate_types
            or "merge" in candidate_types
            or bool(packet.get("force_drag_candidate"))
            or bool(packet.get("merge_candidate"))
        )
        if not force_or_merge:
            continue
        if "force_drag" not in candidate_types:
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "game_id": game_id,
                        "step": _safe_int(packet.get("step")),
                        "rejected_reason": "missing_force_drag_source_relation",
                        "candidate_types": sorted(candidate_types),
                    }
                )
            continue
        if "containment_progress" in candidate_types:
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "game_id": game_id,
                        "step": _safe_int(packet.get("step")),
                        "rejected_reason": "direct_containment_progress_source_relation",
                        "candidate_types": sorted(candidate_types),
                    }
                )
            continue

        regions = [
            dict(row)
            for row in list(packet.get("containment_regions_after") or [])
            if isinstance(row, Mapping) and _is_bbox(row.get("bbox"))
        ]
        if not regions:
            regions = [
                dict(row)
                for row in list(packet.get("containment_regions_before") or [])
                if isinstance(row, Mapping) and _is_bbox(row.get("bbox"))
            ]
        if not regions:
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "game_id": game_id,
                        "step": _safe_int(packet.get("step")),
                        "rejected_reason": "missing_containment_regions",
                    }
                )
            continue

        movement = packet.get("component_movement")
        movement = movement if isinstance(movement, Mapping) else {}
        moved_components = [
            dict(row)
            for row in list(movement.get("moved_components") or [])
            if isinstance(row, Mapping)
        ]
        if not moved_components:
            continue

        for component in moved_components:
            before = component.get("before_centroid")
            after = component.get("after_centroid")
            before_rel = _nearest_region_relation(before, regions)
            after_rel = _nearest_region_relation(after, regions)
            if not before_rel or not after_rel:
                continue
            before_distance = float(before_rel.get("distance", 0.0) or 0.0)
            after_distance = float(after_rel.get("distance", 0.0) or 0.0)
            distance_improvement = before_distance - after_distance
            if after_distance + 0.01 >= before_distance:
                continue
            if distance_improvement < 4.0:
                if len(rejected) < sample_limit:
                    rejected.append(
                        {
                            "game_id": game_id,
                            "step": _safe_int(packet.get("step")),
                            "rejected_reason": "weak_region_distance_improvement",
                            "distance_improvement": round(distance_improvement, 3),
                            "candidate_types": sorted(candidate_types),
                        }
                    )
                continue
            component_color = _safe_int(component.get("color"), -1)
            if component_color == _safe_int(packet.get("background_color"), -999):
                if len(rejected) < sample_limit:
                    rejected.append(
                        {
                            "game_id": game_id,
                            "step": _safe_int(packet.get("step")),
                            "rejected_reason": "background_component_source_relation",
                            "component_color": int(component_color),
                            "candidate_types": sorted(candidate_types),
                        }
                    )
                continue
            if "merge" in candidate_types and distance_improvement < 8.0:
                if len(rejected) < sample_limit:
                    rejected.append(
                        {
                            "game_id": game_id,
                            "step": _safe_int(packet.get("step")),
                            "rejected_reason": "weak_merge_coupled_region_progress",
                            "distance_improvement": round(distance_improvement, 3),
                            "candidate_types": sorted(candidate_types),
                        }
                    )
                continue
            region_bbox = after_rel.get("region_bbox")
            if not _is_bbox(region_bbox):
                continue
            if _safe_int(packet.get("score_delta")) > 0:
                candidate_kind = "force_field_score_positive_relation_replay_candidate"
                bucket = _site_bucket_from_point(after, bucket_size=4)
            elif after_distance <= 8.0:
                candidate_kind = "force_field_relation_closure_retest_candidate"
                bucket = _advance_site_bucket_towards_bbox(after, region_bbox)
            else:
                candidate_kind = "force_field_relation_advance_retest_candidate"
                bucket = _advance_site_bucket_towards_bbox(after, region_bbox)
            if not bucket:
                continue

            region_bucket = str(after_rel.get("region_bucket", "") or "")
            candidate_key = _stable_candidate_key(
                "force_field_score_positive_counterfactual_pair_search",
                game_id,
                candidate_kind,
                bucket,
                region_bucket,
                _safe_int(component.get("color"), -1),
            )
            row = candidates_by_key.setdefault(
                candidate_key,
                {
                    "candidate_key": candidate_key,
                    "candidate_kind": candidate_kind,
                    "source_relation_bucket": bucket,
                    "target_region_bucket": region_bucket,
                    "source_after_component_bucket": _site_bucket_from_point(
                        after,
                        bucket_size=4,
                    ),
                    "component_color": int(component_color),
                    "candidate_types": sorted(candidate_types),
                    "game_ids": [],
                    "observation_count": 0,
                    "score_positive_observation_count": 0,
                    "distance_improvement_sum": 0.0,
                    "after_distance_to_region_min": round(after_distance, 3),
                    "sample_steps": [],
                    "requires_guarded_probe": True,
                    "required_controls": [
                        "scoreless_force_drag_control",
                        "scoreless_merge_control",
                        "wrong_relation_bucket_control",
                    ],
                    "planner_connection_enabled": False,
                    "runtime_execution_enabled": False,
                    "ranking_enabled": False,
                    "action_generation_allowed": False,
                },
            )
            row["game_ids"] = sorted(set(list(row.get("game_ids") or []) + [game_id]))
            row["candidate_types"] = sorted(
                set(list(row.get("candidate_types") or []) + sorted(candidate_types))
            )
            row["observation_count"] = _safe_int(row.get("observation_count")) + 1
            if _safe_int(packet.get("score_delta")) > 0:
                row["score_positive_observation_count"] = _safe_int(
                    row.get("score_positive_observation_count")
                ) + 1
            row["distance_improvement_sum"] = round(
                float(row.get("distance_improvement_sum", 0.0) or 0.0)
                + max(0.0, before_distance - after_distance),
                3,
            )
            row["after_distance_to_region_min"] = round(
                min(
                    float(row.get("after_distance_to_region_min", after_distance) or 0.0),
                    after_distance,
                ),
                3,
            )
            if len(list(row.get("sample_steps") or [])) < 4:
                row["sample_steps"] = list(row.get("sample_steps") or []) + [
                    _safe_int(packet.get("step"))
                ]
            counts["candidate_observation_count"] += 1
            by_game[game_id]["candidate_observation_count"] += 1

    candidates = sorted(
        candidates_by_key.values(),
        key=lambda row: (
            -_safe_int(row.get("score_positive_observation_count")),
            float(row.get("after_distance_to_region_min", 1e9) or 1e9),
            -float(row.get("distance_improvement_sum", 0.0) or 0.0),
            str(row.get("candidate_key", "")),
        ),
    )
    for candidate in candidates:
        kind = str(candidate.get("candidate_kind", "") or "")
        game_ids = [str(row) for row in list(candidate.get("game_ids") or [])]
        by_kind.setdefault(kind, Counter())
        by_kind[kind]["candidate_count"] += 1
        if _safe_int(candidate.get("score_positive_observation_count")) > 0:
            by_kind[kind]["score_positive_candidate_count"] += 1
        for game_id in game_ids or ["unknown"]:
            by_game.setdefault(game_id, Counter())
            by_game[game_id]["candidate_count"] += 1
            by_game[game_id][f"candidate_kind:{kind}"] += 1

    kind_counts = Counter(str(row.get("candidate_kind", "")) for row in candidates)
    score_positive_candidate_count = sum(
        1 for row in candidates if _safe_int(row.get("score_positive_observation_count")) > 0
    )
    if score_positive_candidate_count > 0:
        promotion_state = (
            "force_field_score_positive_counterfactual_pair_search_score_positive_supported"
        )
    elif candidates:
        promotion_state = (
            "force_field_score_positive_counterfactual_pair_search_candidates_supported"
        )
    elif _safe_int(relation.get("force_to_score_target_relation_packet_count")) > 0:
        promotion_state = (
            "force_field_score_positive_counterfactual_pair_search_relation_without_candidate"
        )
    else:
        promotion_state = (
            "force_field_score_positive_counterfactual_pair_search_no_relation"
        )

    return _sanitize(
        {
            "schema_version": (
                FORCE_FIELD_SCORE_POSITIVE_COUNTERFACTUAL_PAIR_SEARCH_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "default_off": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "relation_promotion_state": relation.get("promotion_state"),
            "relation_force_to_score_target_packet_count": _safe_int(
                relation.get("force_to_score_target_relation_packet_count")
            ),
            "packet_count": int(len(packet_rows)),
            "accepted_packet_count": int(counts.get("accepted_packet_count", 0)),
            "rejected_packet_count": int(counts.get("rejected_packet_count", 0)),
            "candidate_observation_count": int(
                counts.get("candidate_observation_count", 0)
            ),
            "candidate_count": int(len(candidates)),
            "score_positive_candidate_count": int(score_positive_candidate_count),
            "force_field_relation_advance_retest_candidate_count": int(
                kind_counts.get("force_field_relation_advance_retest_candidate", 0)
            ),
            "force_field_relation_closure_retest_candidate_count": int(
                kind_counts.get("force_field_relation_closure_retest_candidate", 0)
            ),
            "force_field_score_positive_relation_replay_candidate_count": int(
                kind_counts.get("force_field_score_positive_relation_replay_candidate", 0)
            ),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_candidate_kind": {
                key: dict(sorted(counter.items()))
                for key, counter in sorted(by_kind.items())
            },
            "candidates": candidates[:sample_limit],
            "rejected": rejected[:sample_limit],
            "promotion_state": promotion_state,
            "next_runtime_contract": {
                "required_before_guarded_probe": (
                    "relation-advance templates must be admitted only through "
                    "the default-off guarded probe and paired with scoreless "
                    "controls before planner promotion"
                ),
                "allowed_connection": "score_positive_counterfactual_pair_search_templates_only",
                "action_generation_allowed": False,
            },
        }
    )


def _component_topology(arr: np.ndarray) -> dict[str, Any]:
    background = _background_color(arr)
    components = _connected_components(arr)
    target_components = [
        row
        for row in components
        if int(row["color"]) != background
        and int(row["area"]) <= 256
        and not _is_ui_or_border(row, arr.shape)
    ]
    containment_regions = [
        row
        for row in target_components
        if _bbox_width(row["bbox"]) >= 5
        and _bbox_height(row["bbox"]) >= 5
        and int(row["area"]) >= 16
    ]
    color_counts: Counter[str] = Counter()
    area_by_color: Counter[str] = Counter()
    for row in target_components:
        color = str(int(row["color"]))
        color_counts[color] += 1
        area_by_color[color] += int(row["area"])
    return {
        "background_color": int(background),
        "components": target_components,
        "target_component_count": int(len(target_components)),
        "containment_regions": containment_regions,
        "containment_region_count": int(len(containment_regions)),
        "target_component_color_counts": dict(sorted(color_counts.items())),
        "target_component_area_by_color": dict(sorted(area_by_color.items())),
    }


def _component_movement(
    before_components: Sequence[Mapping[str, Any]],
    after_components: Sequence[Mapping[str, Any]],
    *,
    selected_site: tuple[int, int] | None,
) -> dict[str, Any]:
    matches = _match_components(before_components, after_components)
    moved: list[dict[str, Any]] = []
    moved_towards = 0
    moved_to_site = 0
    for before, after in matches:
        bx, by = _centroid_tuple(before)
        ax, ay = _centroid_tuple(after)
        dx = ax - bx
        dy = ay - by
        distance = math.hypot(dx, dy)
        if distance < 0.5:
            continue
        before_to_site = None
        after_to_site = None
        towards_site = False
        if selected_site is not None:
            sx, sy = selected_site
            before_to_site = math.hypot(float(sx) - bx, float(sy) - by)
            after_to_site = math.hypot(float(sx) - ax, float(sy) - ay)
            towards_site = after_to_site + 0.01 < before_to_site
            if towards_site:
                moved_towards += 1
            if after_to_site <= 1.5:
                moved_to_site += 1
        moved.append(
            {
                "color": int(before.get("color", -1)),
                "area": int(before.get("area", 0)),
                "before_centroid": [round(bx, 3), round(by, 3)],
                "after_centroid": [round(ax, 3), round(ay, 3)],
                "delta": [round(dx, 3), round(dy, 3)],
                "distance": round(distance, 3),
                "distance_to_selected_site_before": (
                    round(before_to_site, 3) if before_to_site is not None else None
                ),
                "distance_to_selected_site_after": (
                    round(after_to_site, 3) if after_to_site is not None else None
                ),
                "moved_towards_selected_site": bool(towards_site),
            }
        )
    return {
        "moved_component_count": int(len(moved)),
        "moved_towards_site_count": int(moved_towards),
        "moved_to_selected_site_count": int(moved_to_site),
        "moved_components": moved[:12],
    }


def _containment_delta(
    before_topology: Mapping[str, Any],
    after_topology: Mapping[str, Any],
) -> dict[str, Any]:
    before_rel = _contained_components(
        before_topology.get("components") or [],
        before_topology.get("containment_regions") or [],
    )
    after_rel = _contained_components(
        after_topology.get("components") or [],
        after_topology.get("containment_regions") or [],
    )
    moved_after = 0
    after_region_keys = {str(row.get("component_key", "")) for row in after_rel}
    for before, after in _match_components(
        before_topology.get("components") or [],
        after_topology.get("components") or [],
    ):
        if _component_key(after) in after_region_keys and _centroid_tuple(before) != _centroid_tuple(after):
            moved_after += 1
    return {
        "contained_component_count_before": int(len(before_rel)),
        "contained_component_count_after": int(len(after_rel)),
        "contained_component_count_delta": int(len(after_rel) - len(before_rel)),
        "moved_component_contained_after_count": int(moved_after),
        "contained_components_after": after_rel[:12],
    }


def _contained_components(
    components: Sequence[Mapping[str, Any]],
    regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    region_keys = {_component_key(region) for region in regions}
    for component in components:
        key = _component_key(component)
        if key in region_keys:
            continue
        cx, cy = _centroid_tuple(component)
        for region in regions:
            bbox = region.get("bbox")
            if not _is_bbox(bbox):
                continue
            x0, y0, x1, y1 = (int(v) for v in bbox)
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                out.append(
                    {
                        "component_key": key,
                        "component_color": int(component.get("color", -1)),
                        "component_centroid": [round(cx, 3), round(cy, 3)],
                        "region_key": _component_key(region),
                        "region_color": int(region.get("color", -1)),
                        "region_bbox": [x0, y0, x1, y1],
                    }
                )
                break
    return out


def _match_components(
    before_components: Sequence[Mapping[str, Any]],
    after_components: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    remaining = list(after_components)
    matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for before in before_components:
        before_color = int(before.get("color", -999))
        before_area = int(before.get("area", -1))
        bx, by = _centroid_tuple(before)
        best_index = -1
        best_distance = 1e9
        for index, after in enumerate(remaining):
            if int(after.get("color", -998)) != before_color:
                continue
            if int(after.get("area", -2)) != before_area:
                continue
            ax, ay = _centroid_tuple(after)
            distance = math.hypot(ax - bx, ay - by)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index >= 0 and best_distance <= 48.0:
            matches.append((before, remaining.pop(best_index)))
    return matches


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


def _counter_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, int]:
    keys = sorted(set(str(key) for key in before) | set(str(key) for key in after))
    out = {
        key: _safe_int(after.get(key), 0) - _safe_int(before.get(key), 0)
        for key in keys
    }
    removed = sum(-value for value in out.values() if value < 0)
    added = sum(value for value in out.values() if value > 0)
    out["removed_area_total"] = int(removed)
    out["added_area_total"] = int(added)
    return out


def _nearest_region_relation(
    point: Any,
    regions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not _is_xy(point) or not regions:
        return {}
    px = float(point[0])
    py = float(point[1])
    best: dict[str, Any] = {}
    best_distance = 1e9
    for region in regions:
        bbox = region.get("bbox")
        if not _is_bbox(bbox):
            continue
        x0, y0, x1, y1 = (int(v) for v in bbox)
        distance = _point_bbox_distance(px, py, [x0, y0, x1, y1])
        if distance < best_distance:
            best_distance = distance
            cx = (float(x0) + float(x1)) / 2.0
            cy = (float(y0) + float(y1)) / 2.0
            relation = "inside" if distance <= 0.01 else "near" if distance <= 8.0 else "far"
            best = {
                "relation": relation,
                "distance": round(distance, 3),
                "region_bbox": [x0, y0, x1, y1],
                "region_color": _safe_int(region.get("color"), -1),
                "region_bucket": f"region_bucket:{int(cx) // 8}:{int(cy) // 8}",
            }
    return best


def _point_bbox_distance(px: float, py: float, bbox: Sequence[int]) -> float:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    dx = 0.0 if x0 <= px <= x1 else min(abs(px - x0), abs(px - x1))
    dy = 0.0 if y0 <= py <= y1 else min(abs(py - y0), abs(py - y1))
    return math.hypot(dx, dy)


def _mechanic_state_type(packet: Mapping[str, Any]) -> str:
    types = set(str(row) for row in list(packet.get("candidate_types") or []))
    if "score_closure" in types:
        return "force_merge_containment_score_closure"
    if "merge" in types and "containment_progress" in types:
        return "merge_containment_progress"
    if "merge" in types:
        return "merge_relation"
    if "containment_progress" in types:
        return "containment_relation"
    if "force_drag" in types:
        return "force_drag_relation"
    return "force_field_observation"


def _posterior_hypothesis_key(
    packet: Mapping[str, Any],
    mechanic_state: str,
) -> str:
    candidate_types = ",".join(
        sorted(str(row) for row in list(packet.get("candidate_types") or []))
    )
    action_name = str(packet.get("action_name", "") or "")
    selected_site = packet.get("selected_site")
    site_bucket = "site:none"
    if _is_xy(selected_site):
        # Bucket to preserve locality without turning the posterior into an
        # exact-coordinate action proposal.
        sx = int(selected_site[0]) // 8
        sy = int(selected_site[1]) // 8
        site_bucket = f"site_bucket:{sx}:{sy}"
    return f"{mechanic_state}:{action_name}:{candidate_types}:{site_bucket}"


def _source_site_bucket_from_hypothesis_key(hypothesis_key: str) -> str | None:
    for part in str(hypothesis_key or "").split(":"):
        if part == "site_bucket":
            pieces = str(hypothesis_key or "").split(":")
            try:
                index = pieces.index("site_bucket")
                return ":".join(pieces[index : index + 3])
            except (ValueError, IndexError):
                return None
    return None


def _site_bucket_from_point(point: Any, *, bucket_size: int = 8) -> str:
    if not _is_xy(point):
        return ""
    size = 4 if int(bucket_size) == 4 else 8
    prefix = "site4_bucket" if size == 4 else "site_bucket"
    x = max(0, int(float(point[0])))
    y = max(0, int(float(point[1])))
    return f"{prefix}:{x // size}:{y // size}"


def _advance_site_bucket_towards_bbox(point: Any, bbox: Any) -> str:
    if not _is_xy(point) or not _is_bbox(bbox):
        return ""
    px = float(point[0])
    py = float(point[1])
    x0, y0, x1, y1 = (float(value) for value in bbox)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    dx = cx - px
    dy = cy - py
    distance = math.hypot(dx, dy)
    if distance <= 0.01:
        return _site_bucket_from_point([px, py])
    step = min(5.5, distance)
    return _site_bucket_from_point(
        [px + dx / distance * step, py + dy / distance * step],
        bucket_size=4,
    )


def _stable_candidate_key(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"force_field_merge_containment_candidate:{digest[:16]}"


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


def _background_color(arr: np.ndarray) -> int:
    values, counts = np.unique(arr, return_counts=True)
    index = int(np.argmax(counts))
    return int(values[index])


def _is_ui_or_border(component: Mapping[str, Any], shape: Sequence[int]) -> bool:
    bbox = component.get("bbox")
    if not _is_bbox(bbox):
        return False
    x0, y0, x1, y1 = (int(v) for v in bbox)
    height = int(shape[0])
    width = int(shape[1])
    if y0 == 0 and _bbox_width(bbox) >= width // 2 and _bbox_height(bbox) <= 12:
        return True
    if y1 == height - 1 and _bbox_width(bbox) >= width // 2 and _bbox_height(bbox) <= 2:
        return True
    return False


def _component_key(component: Mapping[str, Any]) -> str:
    return (
        f"{int(component.get('color', -1))}:"
        f"{int(component.get('area', 0))}:"
        f"{','.join(str(int(v)) for v in component.get('bbox', [])[:4])}"
    )


def _centroid_tuple(component: Mapping[str, Any]) -> tuple[float, float]:
    centroid = component.get("centroid")
    if isinstance(centroid, Sequence) and len(centroid) >= 2:
        return float(centroid[0]), float(centroid[1])
    bbox = component.get("bbox")
    if _is_bbox(bbox):
        x0, y0, x1, y1 = (int(v) for v in bbox)
        return (float(x0 + x1) / 2.0), (float(y0 + y1) / 2.0)
    return 0.0, 0.0


def _bbox_width(bbox: Any) -> int:
    if not _is_bbox(bbox):
        return 0
    return int(bbox[2]) - int(bbox[0]) + 1


def _bbox_height(bbox: Any) -> int:
    if not _is_bbox(bbox):
        return 0
    return int(bbox[3]) - int(bbox[1]) + 1


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
    "FORCE_FIELD_MERGE_CONTAINMENT_ALL25_NON_REGRESSION_SCHEMA_VERSION",
    "FORCE_FIELD_MERGE_CONTAINMENT_CANDIDATE_MINER_SCHEMA_VERSION",
    "FORCE_FIELD_MERGE_CONTAINMENT_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION",
    "FORCE_FIELD_MERGE_CONTAINMENT_DIAGNOSTIC_SENSOR_SCHEMA_VERSION",
    "FORCE_FIELD_CONTAINMENT_SCORE_TARGET_RELATION_SCHEMA_VERSION",
    "FORCE_FIELD_MERGE_CONTAINMENT_GUARDED_PROBE_OUTCOME_SCHEMA_VERSION",
    "FORCE_FIELD_MERGE_CONTAINMENT_PACKET_SCHEMA_VERSION",
    "FORCE_FIELD_MERGE_CONTAINMENT_POSTERIOR_SCHEMA_VERSION",
    "FORCE_FIELD_MERGE_CONTAINMENT_SCORE_TARGET_GAP_SCHEMA_VERSION",
    "FORCE_FIELD_SCORE_POSITIVE_COUNTERFACTUAL_PAIR_SEARCH_SCHEMA_VERSION",
    "build_force_field_merge_containment_packet",
    "summarize_force_field_containment_score_target_relations",
    "summarize_force_field_score_positive_counterfactual_pair_search_candidates",
    "summarize_force_field_merge_containment_all25_non_regression",
    "summarize_force_field_merge_containment_candidate_miner",
    "summarize_force_field_merge_containment_counterfactual_controls",
    "summarize_force_field_merge_containment_guarded_probe_outcomes",
    "summarize_force_field_merge_containment_packets",
    "summarize_force_field_merge_containment_posterior",
    "summarize_force_field_merge_containment_score_target_gap",
    "validate_force_field_merge_containment_packet",
]
