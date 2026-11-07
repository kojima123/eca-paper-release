"""Frame-only component extent-growth diagnostics.

The sensor observes bbox/area growth and small attached-component translations
from public before/after frames. It does not choose sites, emit candidates,
rank actions, or authorize planner execution.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


COMPONENT_EXTENT_GROWTH_PACKET_SCHEMA_VERSION = (
    "eca_component_extent_growth_packet.v1"
)
COMPONENT_EXTENT_GROWTH_DIAGNOSTIC_SENSOR_SCHEMA_VERSION = (
    "eca_component_extent_growth_diagnostic_sensor.v1"
)
COMPONENT_EXTENT_GROWTH_PRECISION_POSTERIOR_SCHEMA_VERSION = (
    "eca_component_extent_growth_precision_posterior.v1"
)
COMPONENT_EXTENT_GROWTH_SLIDER_STRIDE_POSTERIOR_SCHEMA_VERSION = (
    "eca_component_extent_growth_slider_stride_posterior.v1"
)
COMPONENT_EXTENT_GROWTH_TARGET_DISTANCE_POSTERIOR_SCHEMA_VERSION = (
    "eca_component_extent_growth_target_distance_posterior.v1"
)


def build_component_extent_growth_packet(
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
    """Build an observational extent-growth packet."""

    before = _as_frame(before_frame)
    after = _as_frame(after_frame) if after_frame is not None else before
    action_payload = dict(action_payload or {})
    selected_site = _selected_site(action_payload)
    action = _normalize_action_name(action_name)
    before_topology = _component_topology(before)
    after_topology = _component_topology(after)
    growths = _component_growth_observations(
        before_topology["components"],
        after_topology["components"],
        selected_site=selected_site,
    )
    translations = _component_translation_observations(
        before_topology["components"],
        after_topology["components"],
    )
    attached_translations = _attached_translations_for_growth(
        growths,
        translations,
    )
    target_distance_observations = _target_distance_observations(
        before,
        after,
        attached_translations,
    )
    color_area_delta = _counter_delta(
        before_topology["component_area_by_color"],
        after_topology["component_area_by_color"],
    )
    action6_click = action == "ACTION6"
    extent_growth_candidate = bool(action6_click and growths)
    endpoint_growth_candidate = bool(
        action6_click
        and any(
            str(item.get("growth_axis", "") or "") in {"horizontal", "vertical"}
            for item in growths
        )
    )
    attached_translation_candidate = bool(action6_click and attached_translations)
    score_closure_candidate = bool(
        int(score_delta) > 0
        and (extent_growth_candidate or attached_translation_candidate)
    )
    candidate_types: set[str] = set()
    if extent_growth_candidate:
        candidate_types.add("component_extent_growth")
    if endpoint_growth_candidate:
        candidate_types.add("component_endpoint_growth")
    if attached_translation_candidate:
        candidate_types.add("attached_component_translation")
    if score_closure_candidate:
        candidate_types.add("score_closure")

    return _sanitize(
        {
            "schema_version": COMPONENT_EXTENT_GROWTH_PACKET_SCHEMA_VERSION,
            "sensor_family": "component_extent_growth_sensor",
            "game_id": str(game_id or "").strip().lower(),
            "state_hash": str(state_hash or _frame_hash(before)),
            "step": int(step),
            "action_name": action,
            "action_payload": action_payload,
            "selected_site": list(selected_site) if selected_site is not None else None,
            "score_delta": int(score_delta),
            "background_color": int(before_topology["background_color"]),
            "before_component_count": int(before_topology["component_count"]),
            "after_component_count": int(after_topology["component_count"]),
            "component_count_delta": int(
                after_topology["component_count"] - before_topology["component_count"]
            ),
            "component_area_delta_by_color": color_area_delta,
            "growth_observations": growths[:16],
            "translation_observations": translations[:16],
            "attached_translation_observations": attached_translations[:16],
            "target_distance_observations": target_distance_observations[:16],
            "extent_growth_candidate": extent_growth_candidate,
            "endpoint_growth_candidate": endpoint_growth_candidate,
            "attached_translation_candidate": attached_translation_candidate,
            "score_closure_candidate": score_closure_candidate,
            "candidate_types": sorted(candidate_types),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
        }
    )


def validate_component_extent_growth_packet(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if (
        str(packet.get("schema_version", "") or "")
        != COMPONENT_EXTENT_GROWTH_PACKET_SCHEMA_VERSION
    ):
        reasons.append("schema_version_mismatch")
    if str(packet.get("sensor_family", "") or "") != "component_extent_growth_sensor":
        reasons.append("sensor_family_mismatch")
    selected_site = packet.get("selected_site")
    if selected_site is not None and not _is_xy(selected_site):
        reasons.append("invalid_selected_site")
    return {
        "schema_version": "eca_component_extent_growth_validation.v1",
        "accepted": not reasons,
        "rejected_reasons": reasons,
        "family": "component_extent_growth",
        "mechanic_state_type": _mechanic_state_type(packet),
        "goal_link_type": "extent_growth_to_attached_motion",
        "candidate_types": list(packet.get("candidate_types") or []),
    }


def summarize_component_extent_growth_packets(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 12,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    candidate_type_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    score_delta_sum = 0
    growth_area_delta_sum = 0
    attached_translation_count_sum = 0
    target_distance_count_sum = 0
    target_distance_improved_count = 0

    for packet in packet_rows:
        validation = validate_component_extent_growth_packet(packet)
        accepted = bool(validation.get("accepted"))
        counts["accepted_packet_count" if accepted else "rejected_packet_count"] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["packet_count"] += 1
        action_name = str(packet.get("action_name", "") or "")
        if action_name == "ACTION6":
            counts["action6_packet_count"] += 1
            by_game[game_id]["action6_packet_count"] += 1
        score_delta = _safe_int(packet.get("score_delta"))
        score_delta_sum += score_delta
        growths = list(packet.get("growth_observations") or [])
        attached = list(packet.get("attached_translation_observations") or [])
        target_distances = list(packet.get("target_distance_observations") or [])
        attached_translation_count_sum += len(attached)
        target_distance_count_sum += len(target_distances)
        improved_count = sum(
            1
            for item in target_distances
            if isinstance(item, Mapping) and bool(item.get("distance_improved"))
        )
        target_distance_improved_count += improved_count
        if target_distances:
            by_game[game_id]["target_distance_observation_count"] += len(
                target_distances
            )
        if improved_count:
            by_game[game_id]["target_distance_improved_count"] += improved_count
        for growth in growths:
            if not isinstance(growth, Mapping):
                continue
            growth_area_delta_sum += _safe_int(growth.get("area_delta"))
            axis = str(growth.get("growth_axis", "") or "")
            if axis:
                by_game[game_id][f"growth_axis:{axis}"] += 1
        for candidate_type in list(packet.get("candidate_types") or []):
            key = str(candidate_type)
            candidate_type_counts[key] += 1
            by_game[game_id][f"candidate_type:{key}"] += 1
        if bool(packet.get("extent_growth_candidate")):
            counts["extent_growth_candidate_count"] += 1
            by_game[game_id]["extent_growth_candidate_count"] += 1
        if bool(packet.get("endpoint_growth_candidate")):
            counts["endpoint_growth_candidate_count"] += 1
            by_game[game_id]["endpoint_growth_candidate_count"] += 1
        if bool(packet.get("attached_translation_candidate")):
            counts["attached_translation_candidate_count"] += 1
            by_game[game_id]["attached_translation_candidate_count"] += 1
        if bool(packet.get("score_closure_candidate")):
            counts["score_closure_candidate_count"] += 1
            by_game[game_id]["score_closure_candidate_count"] += 1
        if len(samples) < sample_limit:
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "action_name": action_name,
                    "selected_site": packet.get("selected_site"),
                    "candidate_types": list(packet.get("candidate_types") or []),
                    "growth_observation_count": len(growths),
                    "attached_translation_observation_count": len(attached),
                    "target_distance_observation_count": len(target_distances),
                    "growth_sample": growths[:2],
                    "attached_translation_sample": attached[:2],
                    "target_distance_sample": target_distances[:2],
                    "score_delta": score_delta,
                }
            )

    return _sanitize(
        {
            "schema_version": COMPONENT_EXTENT_GROWTH_DIAGNOSTIC_SENSOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": len(packet_rows),
            "accepted_packet_count": int(counts["accepted_packet_count"]),
            "rejected_packet_count": int(counts["rejected_packet_count"]),
            "action6_packet_count": int(counts["action6_packet_count"]),
            "extent_growth_candidate_count": int(
                counts["extent_growth_candidate_count"]
            ),
            "endpoint_growth_candidate_count": int(
                counts["endpoint_growth_candidate_count"]
            ),
            "attached_translation_candidate_count": int(
                counts["attached_translation_candidate_count"]
            ),
            "score_closure_candidate_count": int(
                counts["score_closure_candidate_count"]
            ),
            "score_delta_sum": int(score_delta_sum),
            "growth_area_delta_sum": int(growth_area_delta_sum),
            "attached_translation_count_sum": int(attached_translation_count_sum),
            "target_distance_observation_count": int(target_distance_count_sum),
            "target_distance_improved_count": int(target_distance_improved_count),
            "candidate_type_counts": dict(sorted(candidate_type_counts.items())),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "samples": samples,
        }
    )


def summarize_component_extent_growth_precision_posterior(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    by_game: dict[str, Counter[str]] = {}
    group_counts: Counter[str] = Counter()
    group_attached_counts: Counter[str] = Counter()
    group_score_counts: Counter[str] = Counter()
    group_examples: dict[str, dict[str, Any]] = {}
    candidate_packet_count = 0
    score_closure_packet_count = 0
    attached_candidate_packet_count = 0

    for packet in packet_rows:
        validation = validate_component_extent_growth_packet(packet)
        if not bool(validation.get("accepted")):
            continue
        game_id = str(packet.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["packet_count"] += 1
        if not bool(packet.get("extent_growth_candidate")):
            continue
        candidate_packet_count += 1
        by_game[game_id]["extent_growth_candidate_count"] += 1
        if bool(packet.get("attached_translation_candidate")):
            attached_candidate_packet_count += 1
            by_game[game_id]["attached_translation_candidate_count"] += 1
        if bool(packet.get("score_closure_candidate")):
            score_closure_packet_count += 1
            by_game[game_id]["score_closure_candidate_count"] += 1
        for growth in list(packet.get("growth_observations") or []):
            if not isinstance(growth, Mapping):
                continue
            key = _growth_group_key(game_id, growth)
            group_counts[key] += 1
            if bool(packet.get("attached_translation_candidate")):
                group_attached_counts[key] += 1
            if bool(packet.get("score_closure_candidate")):
                group_score_counts[key] += 1
            group_examples.setdefault(
                key,
                {
                    "group_key": key,
                    "game_id": game_id,
                    "color": _safe_int(growth.get("color")),
                    "growth_axis": str(growth.get("growth_axis", "") or ""),
                    "growth_sides": list(growth.get("growth_sides") or []),
                    "area_delta": _safe_int(growth.get("area_delta")),
                    "width_delta": _safe_int(growth.get("width_delta")),
                    "height_delta": _safe_int(growth.get("height_delta")),
                    "first_step": _safe_int(packet.get("step")),
                    "selected_site_relation": str(
                        growth.get("selected_site_relation", "") or ""
                    ),
                },
            )

    repeated_groups: list[dict[str, Any]] = []
    repeated_packet_sum = 0
    repeated_game_set: set[str] = set()
    for key, count in group_counts.items():
        if count < 3:
            continue
        example = dict(group_examples.get(key) or {})
        game_id = str(example.get("game_id", "") or "unknown")
        repeated_game_set.add(game_id)
        repeated_packet_sum += int(count)
        attached_count = int(group_attached_counts.get(key, 0))
        score_count = int(group_score_counts.get(key, 0))
        example.update(
            {
                "observation_count": int(count),
                "attached_translation_count": attached_count,
                "score_closure_count": score_count,
                "attached_translation_rate": _safe_div(attached_count, int(count)),
                "score_closure_rate": _safe_div(score_count, int(count)),
            }
        )
        repeated_groups.append(example)
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["repeated_stride_growth_group_count"] += 1
        by_game[game_id]["repeated_stride_growth_observation_count"] += int(count)
        if attached_count > 0:
            by_game[game_id]["repeated_stride_attached_group_count"] += 1

    repeated_groups.sort(
        key=lambda item: (
            -_safe_int(item.get("observation_count")),
            str(item.get("game_id", "")),
            str(item.get("group_key", "")),
        )
    )
    candidate_game_count = sum(
        1
        for counter in by_game.values()
        if _safe_int(counter.get("extent_growth_candidate_count")) > 0
    )
    repeated_game_count = len(repeated_game_set)
    ready_for_guarded_probe_review = bool(
        score_closure_packet_count > 0
        and repeated_game_count == 1
        and candidate_game_count == 1
    )
    if candidate_packet_count == 0:
        promotion_state = "component_extent_growth_no_candidate_evidence"
    elif score_closure_packet_count == 0:
        promotion_state = "component_extent_growth_missing_score_closure"
    elif not ready_for_guarded_probe_review:
        promotion_state = "component_extent_growth_precision_review_required"
    else:
        promotion_state = "component_extent_growth_precision_supported"

    return _sanitize(
        {
            "schema_version": COMPONENT_EXTENT_GROWTH_PRECISION_POSTERIOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "default_off": True,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": len(packet_rows),
            "extent_growth_candidate_packet_count": int(candidate_packet_count),
            "attached_translation_candidate_packet_count": int(
                attached_candidate_packet_count
            ),
            "score_closure_candidate_packet_count": int(score_closure_packet_count),
            "candidate_game_count": int(candidate_game_count),
            "repeated_stride_growth_group_count": len(repeated_groups),
            "repeated_stride_growth_observation_count": int(repeated_packet_sum),
            "repeated_stride_growth_game_count": int(repeated_game_count),
            "ready_for_guarded_probe_review": bool(ready_for_guarded_probe_review),
            "planner_promotion_blocked": not ready_for_guarded_probe_review,
            "promotion_state": promotion_state,
            "precision_blockers": _precision_blockers(
                candidate_packet_count=candidate_packet_count,
                score_closure_packet_count=score_closure_packet_count,
                candidate_game_count=candidate_game_count,
                repeated_game_count=repeated_game_count,
            ),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "repeated_stride_growth_groups": repeated_groups[:sample_limit],
            "next_planner_contract": {
                "required_before_guarded_probe": (
                    "score-positive or target-distance counterfactual evidence, "
                    "single-family precision, and all25 non-regression"
                ),
                "allowed_connection": "diagnostic_precision_review_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_component_extent_growth_slider_stride_posterior(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Split repeated extent growth into slider-like stride evidence."""

    precision = summarize_component_extent_growth_precision_posterior(
        packets,
        sample_limit=max(sample_limit, 32),
    )
    groups = [
        dict(item)
        for item in list(precision.get("repeated_stride_growth_groups") or [])
        if isinstance(item, Mapping)
    ]
    slider_groups: list[dict[str, Any]] = []
    rejected_groups: list[dict[str, Any]] = []
    by_game: dict[str, Counter[str]] = {}
    score_closure_count = 0
    slider_observation_count = 0
    attached_observation_count = 0

    for group in groups:
        game_id = str(group.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        decision = _slider_stride_group_decision(group)
        row = dict(group)
        row["slider_stride_decision"] = decision
        if bool(decision.get("accepted")):
            slider_groups.append(row)
            observations = _safe_int(group.get("observation_count"))
            attached = _safe_int(group.get("attached_translation_count"))
            scores = _safe_int(group.get("score_closure_count"))
            slider_observation_count += observations
            attached_observation_count += attached
            score_closure_count += scores
            by_game[game_id]["slider_stride_group_count"] += 1
            by_game[game_id]["slider_stride_observation_count"] += observations
            by_game[game_id]["slider_stride_attached_translation_count"] += attached
            by_game[game_id]["slider_stride_score_closure_count"] += scores
        else:
            rejected_groups.append(row)
            by_game[game_id]["rejected_repeated_stride_group_count"] += 1
            for reason in list(decision.get("rejected_reasons") or []):
                by_game[game_id][f"rejected_reason:{reason}"] += 1

    slider_groups.sort(
        key=lambda item: (
            -_safe_int(item.get("observation_count")),
            str(item.get("game_id", "")),
            str(item.get("group_key", "")),
        )
    )
    rejected_groups.sort(
        key=lambda item: (
            str(item.get("game_id", "")),
            str(item.get("group_key", "")),
        )
    )
    slider_game_count = len({str(item.get("game_id", "") or "unknown") for item in slider_groups})
    ready_for_guarded_probe_review = bool(
        slider_groups and score_closure_count > 0 and slider_game_count == 1
    )
    if not slider_groups:
        promotion_state = "component_extent_growth_no_slider_stride_evidence"
    elif score_closure_count <= 0:
        promotion_state = (
            "component_extent_growth_slider_stride_missing_score_or_target_counterfactual"
        )
    elif slider_game_count > 1:
        promotion_state = "component_extent_growth_slider_stride_breadth_precision_required"
    else:
        promotion_state = "component_extent_growth_slider_stride_precision_supported"

    return _sanitize(
        {
            "schema_version": COMPONENT_EXTENT_GROWTH_SLIDER_STRIDE_POSTERIOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "default_off": True,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "input_precision_schema_version": str(
                precision.get("schema_version", "") or ""
            ),
            "input_repeated_stride_growth_group_count": len(groups),
            "slider_stride_group_count": len(slider_groups),
            "slider_stride_observation_count": int(slider_observation_count),
            "slider_stride_attached_translation_count": int(attached_observation_count),
            "slider_stride_score_closure_count": int(score_closure_count),
            "slider_stride_game_count": int(slider_game_count),
            "rejected_repeated_stride_group_count": len(rejected_groups),
            "ready_for_guarded_probe_review": bool(ready_for_guarded_probe_review),
            "planner_promotion_blocked": not ready_for_guarded_probe_review,
            "promotion_state": promotion_state,
            "precision_blockers": _slider_stride_precision_blockers(
                slider_group_count=len(slider_groups),
                score_closure_count=score_closure_count,
                slider_game_count=slider_game_count,
            ),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "slider_stride_groups": slider_groups[:sample_limit],
            "rejected_repeated_stride_groups": rejected_groups[:sample_limit],
            "next_planner_contract": {
                "required_before_guarded_probe": (
                    "slider-stride evidence plus score-positive or target-distance "
                    "counterfactual evidence and all25 non-regression"
                ),
                "allowed_connection": "diagnostic_slider_stride_precision_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_component_extent_growth_target_distance_posterior(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize whether attached slider motion moves targets toward markers."""

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    slider = summarize_component_extent_growth_slider_stride_posterior(
        packet_rows,
        sample_limit=max(sample_limit, 32),
    )
    by_game: dict[str, Counter[str]] = {}
    observations: list[dict[str, Any]] = []
    packet_count_with_distance = 0
    improved_count = 0
    worsened_count = 0
    unchanged_count = 0
    aligned_count = 0

    for packet in packet_rows:
        validation = validate_component_extent_growth_packet(packet)
        if not bool(validation.get("accepted")):
            continue
        game_id = str(packet.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        target_rows = [
            dict(item)
            for item in list(packet.get("target_distance_observations") or [])
            if isinstance(item, Mapping)
        ]
        if not target_rows:
            continue
        packet_count_with_distance += 1
        by_game[game_id]["packet_with_target_distance_count"] += 1
        for item in target_rows:
            relation = str(item.get("distance_relation", "") or "unknown")
            by_game[game_id]["target_distance_observation_count"] += 1
            by_game[game_id][f"target_distance_relation:{relation}"] += 1
            if bool(item.get("distance_improved")):
                improved_count += 1
                by_game[game_id]["target_distance_improved_count"] += 1
            elif relation == "worsened":
                worsened_count += 1
                by_game[game_id]["target_distance_worsened_count"] += 1
            else:
                unchanged_count += 1
                by_game[game_id]["target_distance_unchanged_count"] += 1
            if bool(item.get("movement_aligned_with_marker")):
                aligned_count += 1
                by_game[game_id]["movement_aligned_with_marker_count"] += 1
            if len(observations) < sample_limit:
                row = dict(item)
                row["game_id"] = game_id
                row["step"] = _safe_int(packet.get("step"))
                row["action_name"] = str(packet.get("action_name", "") or "")
                row["selected_site"] = packet.get("selected_site")
                observations.append(row)

    target_distance_game_count = sum(
        1
        for counter in by_game.values()
        if _safe_int(counter.get("target_distance_observation_count")) > 0
    )
    improved_game_count = sum(
        1
        for counter in by_game.values()
        if _safe_int(counter.get("target_distance_improved_count")) > 0
    )
    slider_group_count = _safe_int(slider.get("slider_stride_group_count"))
    slider_game_count = _safe_int(slider.get("slider_stride_game_count"))
    ready_for_guarded_probe_review = bool(
        slider_group_count > 0
        and improved_count > 0
        and improved_game_count == 1
        and slider_game_count == 1
    )
    if slider_group_count <= 0:
        promotion_state = "component_extent_growth_target_distance_no_slider_stride"
    elif improved_count <= 0:
        promotion_state = "component_extent_growth_target_distance_missing_improvement"
    elif improved_game_count > 1 or slider_game_count > 1:
        promotion_state = (
            "component_extent_growth_target_distance_breadth_precision_required"
        )
    else:
        promotion_state = "component_extent_growth_target_distance_supported"

    return _sanitize(
        {
            "schema_version": (
                COMPONENT_EXTENT_GROWTH_TARGET_DISTANCE_POSTERIOR_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "default_off": True,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "input_slider_stride_schema_version": str(
                slider.get("schema_version", "") or ""
            ),
            "packet_count": len(packet_rows),
            "packet_with_target_distance_count": int(packet_count_with_distance),
            "target_distance_observation_count": int(
                improved_count + worsened_count + unchanged_count
            ),
            "target_distance_improved_count": int(improved_count),
            "target_distance_worsened_count": int(worsened_count),
            "target_distance_unchanged_count": int(unchanged_count),
            "movement_aligned_with_marker_count": int(aligned_count),
            "target_distance_game_count": int(target_distance_game_count),
            "target_distance_improved_game_count": int(improved_game_count),
            "slider_stride_group_count": slider_group_count,
            "slider_stride_game_count": slider_game_count,
            "ready_for_guarded_probe_review": bool(ready_for_guarded_probe_review),
            "planner_promotion_blocked": not ready_for_guarded_probe_review,
            "promotion_state": promotion_state,
            "precision_blockers": _target_distance_precision_blockers(
                slider_group_count=slider_group_count,
                improved_count=improved_count,
                improved_game_count=improved_game_count,
                slider_game_count=slider_game_count,
            ),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "target_distance_samples": observations[:sample_limit],
            "next_planner_contract": {
                "required_before_guarded_probe": (
                    "counterfactual target-distance support, guarded action "
                    "candidate audit, and all25 non-regression"
                ),
                "allowed_connection": "diagnostic_target_distance_counterfactual_only",
                "action_generation_allowed": False,
            },
        }
    )


def _component_growth_observations(
    before_components: Sequence[Mapping[str, Any]],
    after_components: Sequence[Mapping[str, Any]],
    *,
    selected_site: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    used_after: set[str] = set()
    for before in before_components:
        match = _best_component_match(before, after_components, used_after)
        if match is None:
            continue
        used_after.add(str(match.get("id", "")))
        before_bbox = _bbox_tuple(before.get("bbox"))
        after_bbox = _bbox_tuple(match.get("bbox"))
        before_area = _safe_int(before.get("area"))
        after_area = _safe_int(match.get("area"))
        area_delta = after_area - before_area
        width_delta = _bbox_width(after_bbox) - _bbox_width(before_bbox)
        height_delta = _bbox_height(after_bbox) - _bbox_height(before_bbox)
        if area_delta <= 0:
            continue
        if width_delta <= 0 and height_delta <= 0:
            continue
        axis = _growth_axis(width_delta, height_delta)
        sides = _growth_sides(before_bbox, after_bbox)
        observations.append(
            {
                "before_component_id": str(before.get("id", "")),
                "after_component_id": str(match.get("id", "")),
                "color": _safe_int(before.get("color")),
                "before_bbox": list(before_bbox),
                "after_bbox": list(after_bbox),
                "before_area": before_area,
                "after_area": after_area,
                "area_delta": int(area_delta),
                "width_delta": int(width_delta),
                "height_delta": int(height_delta),
                "growth_axis": axis,
                "growth_sides": sides,
                "selected_site_relation": _selected_site_relation(
                    selected_site,
                    before_bbox,
                    after_bbox,
                ),
            }
        )
    observations.sort(
        key=lambda item: (
            -_safe_int(item.get("area_delta")),
            str(item.get("color", "")),
            str(item.get("before_bbox", "")),
        )
    )
    return observations


def _component_translation_observations(
    before_components: Sequence[Mapping[str, Any]],
    after_components: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    used_after: set[str] = set()
    for before in before_components:
        match = _best_component_match(before, after_components, used_after)
        if match is None:
            continue
        used_after.add(str(match.get("id", "")))
        before_area = _safe_int(before.get("area"))
        after_area = _safe_int(match.get("area"))
        if before_area != after_area:
            continue
        before_centroid = _xy_tuple(before.get("centroid"))
        after_centroid = _xy_tuple(match.get("centroid"))
        dx = round(after_centroid[0] - before_centroid[0], 3)
        dy = round(after_centroid[1] - before_centroid[1], 3)
        distance = round(math.hypot(dx, dy), 3)
        if distance < 0.5:
            continue
        observations.append(
            {
                "before_component_id": str(before.get("id", "")),
                "after_component_id": str(match.get("id", "")),
                "color": _safe_int(before.get("color")),
                "area": before_area,
                "before_bbox": list(_bbox_tuple(before.get("bbox"))),
                "after_bbox": list(_bbox_tuple(match.get("bbox"))),
                "before_centroid": [round(before_centroid[0], 3), round(before_centroid[1], 3)],
                "after_centroid": [round(after_centroid[0], 3), round(after_centroid[1], 3)],
                "delta": [dx, dy],
                "distance": distance,
                "dominant_axis": _translation_axis(dx, dy),
            }
        )
    observations.sort(
        key=lambda item: (
            -float(item.get("distance", 0.0) or 0.0),
            str(item.get("color", "")),
        )
    )
    return observations


def _attached_translations_for_growth(
    growths: Sequence[Mapping[str, Any]],
    translations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    for translation in translations:
        t_axis = str(translation.get("dominant_axis", "") or "")
        t_delta = _xy_tuple(translation.get("delta"))
        for growth in growths:
            g_axis = str(growth.get("growth_axis", "") or "")
            if g_axis == "mixed" or t_axis == "mixed":
                continue
            if g_axis == "horizontal" and t_axis != "horizontal":
                continue
            if g_axis == "vertical" and t_axis != "vertical":
                continue
            before_bbox = _bbox_tuple(growth.get("before_bbox"))
            after_bbox = _bbox_tuple(growth.get("after_bbox"))
            gx = float(_bbox_width(after_bbox) - _bbox_width(before_bbox))
            gy = float(_bbox_height(after_bbox) - _bbox_height(before_bbox))
            alignment = _alignment_sign(t_delta, (gx, gy))
            if alignment <= 0:
                continue
            record = dict(translation)
            record["growth_component_color"] = _safe_int(growth.get("color"))
            record["growth_axis"] = g_axis
            record["alignment_sign"] = int(alignment)
            attached.append(record)
            break
    return attached


def _target_distance_observations(
    before_frame: np.ndarray,
    after_frame: np.ndarray,
    attached_translations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    markers = _sparse_diamond_markers(before_frame) + _sparse_diamond_markers(
        after_frame
    )
    markers_by_color: dict[int, list[dict[str, Any]]] = {}
    seen_marker_keys: set[tuple[int, tuple[int, int, int, int]]] = set()
    for marker in markers:
        color = _safe_int(marker.get("color"))
        bbox = _bbox_tuple(marker.get("bbox"))
        key = (color, bbox)
        if key in seen_marker_keys:
            continue
        seen_marker_keys.add(key)
        markers_by_color.setdefault(color, []).append(marker)

    observations: list[dict[str, Any]] = []
    for translation in attached_translations:
        color = _safe_int(translation.get("color"))
        same_color_markers = markers_by_color.get(color) or []
        if not same_color_markers:
            continue
        before_centroid = _xy_tuple(translation.get("before_centroid"))
        after_centroid = _xy_tuple(translation.get("after_centroid"))
        marker = min(
            same_color_markers,
            key=lambda item: _point_distance(
                before_centroid,
                _xy_tuple(item.get("centroid")),
            ),
        )
        marker_centroid = _xy_tuple(marker.get("centroid"))
        before_distance = _point_distance(before_centroid, marker_centroid)
        after_distance = _point_distance(after_centroid, marker_centroid)
        delta = round(after_distance - before_distance, 3)
        movement = (
            after_centroid[0] - before_centroid[0],
            after_centroid[1] - before_centroid[1],
        )
        marker_vector = (
            marker_centroid[0] - before_centroid[0],
            marker_centroid[1] - before_centroid[1],
        )
        dot = movement[0] * marker_vector[0] + movement[1] * marker_vector[1]
        relation = _target_distance_relation(delta)
        observations.append(
            {
                "translation_component_color": color,
                "translation_before_component_id": str(
                    translation.get("before_component_id", "") or ""
                ),
                "translation_after_component_id": str(
                    translation.get("after_component_id", "") or ""
                ),
                "translation_before_centroid": [
                    round(before_centroid[0], 3),
                    round(before_centroid[1], 3),
                ],
                "translation_after_centroid": [
                    round(after_centroid[0], 3),
                    round(after_centroid[1], 3),
                ],
                "translation_delta": list(translation.get("delta") or []),
                "growth_component_color": _safe_int(
                    translation.get("growth_component_color")
                ),
                "growth_axis": str(translation.get("growth_axis", "") or ""),
                "marker_pattern": str(marker.get("pattern", "") or ""),
                "marker_bbox": list(_bbox_tuple(marker.get("bbox"))),
                "marker_centroid": [
                    round(marker_centroid[0], 3),
                    round(marker_centroid[1], 3),
                ],
                "distance_before": round(before_distance, 3),
                "distance_after": round(after_distance, 3),
                "distance_delta": delta,
                "distance_relation": relation,
                "distance_improved": relation == "improved",
                "movement_aligned_with_marker": dot > 0.0,
            }
        )
    observations.sort(
        key=lambda item: (
            str(item.get("distance_relation", "")) != "improved",
            float(item.get("distance_delta", 0.0) or 0.0),
            str(item.get("translation_before_component_id", "")),
        )
    )
    return observations


def _sparse_diamond_markers(frame: np.ndarray) -> list[dict[str, Any]]:
    arr = _as_frame(frame)
    height, width = arr.shape
    if height < 3 or width < 3:
        return []
    background = _background_color(arr)
    foreground_colors = [
        int(color)
        for color in np.unique(arr).tolist()
        if int(color) != int(background)
    ]
    mask = np.array(
        [
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ],
        dtype=bool,
    )
    markers: list[dict[str, Any]] = []
    for color in foreground_colors:
        ys, xs = np.where(arr == int(color))
        candidates: set[tuple[int, int]] = set()
        for cy, cx in zip(ys.tolist(), xs.tolist()):
            for dy in (-2, -1, 0):
                for dx in (-2, -1, 0):
                    x0 = int(cx + dx)
                    y0 = int(cy + dy)
                    if 0 <= x0 <= width - 3 and 0 <= y0 <= height - 3:
                        candidates.add((x0, y0))
        for x0, y0 in sorted(candidates):
            block = arr[y0 : y0 + 3, x0 : x0 + 3]
            if not bool(np.all(block[mask] == int(color))):
                continue
            if not bool(np.all(block[~mask] == int(background))):
                continue
            markers.append(
                {
                    "pattern": "sparse_diamond_3x3",
                    "color": int(color),
                    "bbox": [int(x0), int(y0), int(x0 + 2), int(y0 + 2)],
                    "centroid": [round(float(x0 + 1), 3), round(float(y0 + 1), 3)],
                }
            )
    markers.sort(
        key=lambda item: (
            _safe_int(item.get("color")),
            str(item.get("bbox", "")),
        )
    )
    return markers


def _component_topology(frame: np.ndarray) -> dict[str, Any]:
    arr = _as_frame(frame)
    background = _background_color(arr)
    components = _connected_components(arr, background)
    area_by_color: Counter[int] = Counter()
    for comp in components:
        area_by_color[_safe_int(comp.get("color"))] += _safe_int(comp.get("area"))
    return {
        "background_color": int(background),
        "component_count": len(components),
        "component_area_by_color": dict(sorted(area_by_color.items())),
        "components": components,
    }


def _connected_components(frame: np.ndarray, background_color: int) -> list[dict[str, Any]]:
    arr = _as_frame(frame)
    height, width = arr.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x]:
                continue
            color = int(arr[y, x])
            if color == int(background_color):
                seen[y, x] = True
                continue
            stack = [(x, y)]
            seen[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                cx, cy = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if seen[ny, nx] or int(arr[ny, nx]) != color:
                        continue
                    seen[ny, nx] = True
                    stack.append((nx, ny))
            area = len(xs)
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            components.append(
                {
                    "id": f"c{len(components)}",
                    "color": color,
                    "area": area,
                    "bbox": [x0, y0, x1, y1],
                    "centroid": [round(sum(xs) / area, 3), round(sum(ys) / area, 3)],
                }
            )
    return components


def _best_component_match(
    before: Mapping[str, Any],
    after_components: Sequence[Mapping[str, Any]],
    used_after: set[str],
) -> Mapping[str, Any] | None:
    before_color = _safe_int(before.get("color"))
    before_bbox = _bbox_tuple(before.get("bbox"))
    best: tuple[float, Mapping[str, Any]] | None = None
    for after in after_components:
        after_id = str(after.get("id", ""))
        if after_id in used_after or _safe_int(after.get("color")) != before_color:
            continue
        after_bbox = _bbox_tuple(after.get("bbox"))
        overlap = _bbox_overlap_area(before_bbox, after_bbox)
        expanded_overlap = _bbox_overlap_area(_expand_bbox(before_bbox, 4), after_bbox)
        distance = _bbox_distance(before_bbox, after_bbox)
        if overlap <= 0 and expanded_overlap <= 0 and distance > 8.0:
            continue
        score = float(overlap * 16 + expanded_overlap * 4) - distance
        if best is None or score > best[0]:
            best = (score, after)
    return None if best is None else best[1]


def _mechanic_state_type(packet: Mapping[str, Any]) -> str:
    if bool(packet.get("score_closure_candidate")):
        return "component_extent_growth_score_closure"
    if bool(packet.get("extent_growth_candidate")) and bool(
        packet.get("attached_translation_candidate")
    ):
        return "extent_growth_with_attached_translation"
    if bool(packet.get("extent_growth_candidate")):
        return "component_extent_growth"
    if bool(packet.get("attached_translation_candidate")):
        return "attached_translation_without_growth"
    return "no_extent_growth"


def _growth_group_key(game_id: str, growth: Mapping[str, Any]) -> str:
    sides = "+".join(str(item) for item in list(growth.get("growth_sides") or []))
    return "|".join(
        [
            str(game_id or "unknown"),
            str(_safe_int(growth.get("color"))),
            str(growth.get("growth_axis", "") or ""),
            sides,
            str(_safe_int(growth.get("area_delta"))),
            str(_safe_int(growth.get("width_delta"))),
            str(_safe_int(growth.get("height_delta"))),
        ]
    )


def _precision_blockers(
    *,
    candidate_packet_count: int,
    score_closure_packet_count: int,
    candidate_game_count: int,
    repeated_game_count: int,
) -> list[str]:
    blockers: list[str] = []
    if candidate_packet_count <= 0:
        blockers.append("missing_extent_growth_candidate")
    if score_closure_packet_count <= 0:
        blockers.append("missing_score_closure")
    if candidate_game_count > 1:
        blockers.append("candidate_evidence_spans_multiple_games")
    if repeated_game_count > 1:
        blockers.append("repeated_stride_evidence_spans_multiple_games")
    return blockers


def _slider_stride_group_decision(group: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    axis = str(group.get("growth_axis", "") or "")
    sides = [str(item) for item in list(group.get("growth_sides") or [])]
    width_delta = abs(_safe_int(group.get("width_delta")))
    height_delta = abs(_safe_int(group.get("height_delta")))
    area_delta = abs(_safe_int(group.get("area_delta")))
    observation_count = _safe_int(group.get("observation_count"))
    attached_rate = float(group.get("attached_translation_rate", 0.0) or 0.0)
    if observation_count < 3:
        reasons.append("insufficient_repeated_observations")
    if axis not in {"horizontal", "vertical"}:
        reasons.append("not_single_axis_growth")
    if len(sides) != 1:
        reasons.append("not_single_side_growth")
    major_delta = width_delta if axis == "horizontal" else height_delta
    orthogonal_delta = height_delta if axis == "horizontal" else width_delta
    if major_delta < 2:
        reasons.append("stride_delta_too_small")
    if orthogonal_delta != 0:
        reasons.append("orthogonal_extent_changed")
    if area_delta < max(3, major_delta * 2):
        reasons.append("growth_area_too_small_for_slider")
    if attached_rate < 0.75:
        reasons.append("attached_translation_rate_too_low")
    return {
        "accepted": not reasons,
        "rejected_reasons": reasons,
        "rule": "single_axis_single_side_repeated_stride_with_attached_translation",
        "min_observation_count": 3,
        "min_major_delta": 2,
        "min_attached_translation_rate": 0.75,
    }


def _slider_stride_precision_blockers(
    *,
    slider_group_count: int,
    score_closure_count: int,
    slider_game_count: int,
) -> list[str]:
    blockers: list[str] = []
    if slider_group_count <= 0:
        blockers.append("missing_slider_stride_group")
    if score_closure_count <= 0:
        blockers.append("missing_score_or_target_distance_counterfactual")
    if slider_game_count > 1:
        blockers.append("slider_stride_evidence_spans_multiple_games")
    return blockers


def _target_distance_precision_blockers(
    *,
    slider_group_count: int,
    improved_count: int,
    improved_game_count: int,
    slider_game_count: int,
) -> list[str]:
    blockers: list[str] = []
    if slider_group_count <= 0:
        blockers.append("missing_slider_stride_group")
    if improved_count <= 0:
        blockers.append("missing_target_distance_improvement")
    if improved_game_count > 1:
        blockers.append("target_distance_improvement_spans_multiple_games")
    if slider_game_count > 1:
        blockers.append("slider_stride_evidence_spans_multiple_games")
    return blockers


def _growth_axis(width_delta: int, height_delta: int) -> str:
    if width_delta > 0 and height_delta <= 0:
        return "horizontal"
    if height_delta > 0 and width_delta <= 0:
        return "vertical"
    if width_delta > height_delta:
        return "horizontal"
    if height_delta > width_delta:
        return "vertical"
    return "mixed"


def _translation_axis(dx: float, dy: float) -> str:
    if abs(dx) > abs(dy) * 1.5:
        return "horizontal"
    if abs(dy) > abs(dx) * 1.5:
        return "vertical"
    return "mixed"


def _growth_sides(before_bbox: tuple[int, int, int, int], after_bbox: tuple[int, int, int, int]) -> list[str]:
    sides: list[str] = []
    if after_bbox[0] < before_bbox[0]:
        sides.append("left")
    if after_bbox[2] > before_bbox[2]:
        sides.append("right")
    if after_bbox[1] < before_bbox[1]:
        sides.append("up")
    if after_bbox[3] > before_bbox[3]:
        sides.append("down")
    return sides


def _selected_site_relation(
    selected_site: tuple[int, int] | None,
    before_bbox: tuple[int, int, int, int],
    after_bbox: tuple[int, int, int, int],
) -> str:
    if selected_site is None:
        return "no_selected_site"
    if _point_in_bbox(selected_site, before_bbox):
        return "inside_before_component"
    if _point_in_bbox(selected_site, after_bbox):
        return "inside_after_component"
    if _point_in_bbox(selected_site, _expand_bbox(before_bbox, 2)):
        return "near_before_component"
    if _point_in_bbox(selected_site, _expand_bbox(after_bbox, 2)):
        return "near_after_component"
    return "far"


def _alignment_sign(delta: tuple[float, float], growth_delta: tuple[float, float]) -> int:
    dot = delta[0] * growth_delta[0] + delta[1] * growth_delta[1]
    if dot > 0:
        return 1
    if dot < 0:
        return -1
    return 0


def _target_distance_relation(distance_delta: float) -> str:
    if distance_delta <= -0.5:
        return "improved"
    if distance_delta >= 0.5:
        return "worsened"
    return "unchanged"


def _point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _counter_delta(before: Mapping[Any, Any], after: Mapping[Any, Any]) -> dict[str, int]:
    keys = set(before) | set(after)
    out: dict[str, int] = {}
    added_total = 0
    removed_total = 0
    for key in sorted(keys, key=lambda item: str(item)):
        delta = _safe_int(after.get(key)) - _safe_int(before.get(key))
        out[str(key)] = int(delta)
        if delta > 0:
            added_total += delta
        elif delta < 0:
            removed_total += -delta
    out["added_area_total"] = int(added_total)
    out["removed_area_total"] = int(removed_total)
    return out


def _as_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError(f"expected 2D frame, got shape={arr.shape}")
    return arr.astype(int, copy=False)


def _background_color(frame: np.ndarray) -> int:
    values, counts = np.unique(frame, return_counts=True)
    if len(values) == 0:
        return 0
    return int(values[int(np.argmax(counts))])


def _frame_hash(frame: np.ndarray) -> str:
    arr = _as_frame(frame)
    digest = hashlib.sha1(arr.astype(np.int16, copy=False).tobytes()).hexdigest()
    return digest[:16]


def _selected_site(action_payload: Mapping[str, Any]) -> tuple[int, int] | None:
    if "x" not in action_payload or "y" not in action_payload:
        site = action_payload.get("site") or action_payload.get("selected_site")
        if _is_xy(site):
            return (int(site[0]), int(site[1]))
        return None
    try:
        return (int(action_payload["x"]), int(action_payload["y"]))
    except (TypeError, ValueError):
        return None


def _normalize_action_name(action_name: str) -> str:
    return str(action_name or "").strip().upper()


def _is_xy(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if len(value) != 2:
        return False
    try:
        int(value[0])
        int(value[1])
    except (TypeError, ValueError):
        return False
    return True


def _xy_tuple(value: Any) -> tuple[float, float]:
    if _is_xy(value):
        return (float(value[0]), float(value[1]))
    return (0.0, 0.0)


def _bbox_tuple(value: Any) -> tuple[int, int, int, int]:
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 4
    ):
        try:
            x0, y0, x1, y1 = [int(v) for v in value]
            return (x0, y0, x1, y1)
        except (TypeError, ValueError):
            pass
    return (0, 0, 0, 0)


def _bbox_width(bbox: tuple[int, int, int, int]) -> int:
    return max(0, int(bbox[2]) - int(bbox[0]) + 1)


def _bbox_height(bbox: tuple[int, int, int, int]) -> int:
    return max(0, int(bbox[3]) - int(bbox[1]) + 1)


def _bbox_overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 < x0 or y1 < y0:
        return 0
    return (x1 - x0 + 1) * (y1 - y0 + 1)


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax = 0.0 if a[0] <= b[2] and b[0] <= a[2] else min(abs(a[0] - b[2]), abs(b[0] - a[2]))
    ay = 0.0 if a[1] <= b[3] and b[1] <= a[3] else min(abs(a[1] - b[3]), abs(b[1] - a[3]))
    return math.hypot(ax, ay)


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)


def _point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_div(num: int | float, den: int | float) -> float:
    try:
        denominator = float(den)
        if denominator == 0.0:
            return 0.0
        return round(float(num) / denominator, 6)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _sanitize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize(v) for v in value]
    return value
