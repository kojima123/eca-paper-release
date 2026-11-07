"""Frame-only spell pattern diagnostics.

The sensor observes spell-grid edits and cast-like effects from public
before/after frames. It does not infer a spell plan, select spell icons, or
authorize candidate ranking.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROGRAM_SPELL_PATTERN_PACKET_SCHEMA_VERSION = (
    "eca_program_spell_pattern_packet.v1"
)
PROGRAM_SPELL_PATTERN_DIAGNOSTIC_SENSOR_SCHEMA_VERSION = (
    "eca_program_spell_pattern_diagnostic_sensor.v1"
)
PROGRAM_SPELL_PATTERN_POSTERIOR_SCHEMA_VERSION = (
    "eca_program_spell_pattern_posterior.v1"
)
PROGRAM_SPELL_PATTERN_CANDIDATE_MINER_SCHEMA_VERSION = (
    "eca_program_spell_pattern_candidate_miner.v1"
)
PROGRAM_SPELL_PATTERN_GUARDED_PROBE_OUTCOME_SCHEMA_VERSION = (
    "eca_program_spell_pattern_guarded_probe_outcome_sensor.v1"
)
PROGRAM_SPELL_PATTERN_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION = (
    "eca_program_spell_pattern_counterfactual_control_sensor.v1"
)
PROGRAM_SPELL_PATTERN_ALL25_NON_REGRESSION_SCHEMA_VERSION = (
    "eca_program_spell_pattern_all25_non_regression_sensor.v1"
)


def build_program_spell_pattern_packet(
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
    """Build an observational spell-pattern packet."""

    before = _as_frame(before_frame)
    after = _as_frame(after_frame) if after_frame is not None else before
    action_payload = dict(action_payload or {})
    selected_site = _selected_site(action_payload)
    action = _normalize_action_name(action_name)
    before_grid = _spell_grid_observation(before)
    after_grid = _spell_grid_observation(after)
    grid_bbox = _union_bbox(
        [
            *[row["bbox"] for row in before_grid["cells"]],
            *[row["bbox"] for row in after_grid["cells"]],
        ]
    )
    selected_grid_cell = _selected_grid_cell(
        before,
        after,
        selected_site=selected_site,
        before_grid=before_grid,
        after_grid=after_grid,
    )
    change = _change_observation(before, after, grid_bbox=grid_bbox)
    selected_icon = _selected_spell_icon_region(
        selected_site=selected_site,
        grid_bbox=grid_bbox,
        frame_shape=before.shape,
    )
    active_delta = int(after_grid["active_cell_count"]) - int(
        before_grid["active_cell_count"]
    )
    hint_delta = int(after_grid["hint_required_cell_count"]) - int(
        before_grid["hint_required_cell_count"]
    )
    grid_changed = int(change["grid_changed_cell_count"]) > 0
    result_changed = int(change["effect_region_changed_cell_count"]) > 0
    on_grid = selected_grid_cell is not None
    spell_grid_toggle_candidate = bool(
        action == "ACTION6"
        and on_grid
        and grid_changed
        and (
            bool(selected_grid_cell.get("changed"))
            or active_delta != 0
            or hint_delta != 0
        )
    )
    spell_selection_candidate = bool(
        action == "ACTION6" and selected_icon is not None and not on_grid
    )
    spell_cast_candidate = bool(
        action == "ACTION6"
        and on_grid
        and grid_changed
        and (
            result_changed
            or (
                int(before_grid["active_cell_count"]) > 0
                and int(after_grid["active_cell_count"]) == 0
                and int(change["grid_changed_cell_count"]) >= 9
            )
            or (
                int(before_grid["hint_required_cell_count"]) <= 1
                and active_delta <= 0
                and int(change["grid_changed_cell_count"]) >= 9
            )
        )
    )
    spell_effect_candidate = bool(spell_cast_candidate and result_changed)
    spell_state_delta_candidate = bool(
        spell_grid_toggle_candidate or spell_selection_candidate or spell_cast_candidate
    )
    score_closure_candidate = bool(
        int(score_delta) > 0
        and (spell_cast_candidate or spell_effect_candidate or spell_state_delta_candidate)
    )
    candidate_types: set[str] = set()
    if spell_grid_toggle_candidate:
        candidate_types.add("spell_grid_toggle")
    if spell_selection_candidate:
        candidate_types.add("spell_selection")
    if spell_cast_candidate:
        candidate_types.add("spell_cast")
    if spell_effect_candidate:
        candidate_types.add("spell_effect")
    if spell_state_delta_candidate:
        candidate_types.add("spell_state_delta")
    if score_closure_candidate:
        candidate_types.add("score_closure")

    return _sanitize(
        {
            "schema_version": PROGRAM_SPELL_PATTERN_PACKET_SCHEMA_VERSION,
            "sensor_family": "program_spell_pattern_sensor",
            "game_id": str(game_id or "").strip().lower(),
            "state_hash": str(state_hash or _frame_hash(before)),
            "step": int(step),
            "action_name": action,
            "action_payload": action_payload,
            "selected_site": list(selected_site) if selected_site is not None else None,
            "score_delta": int(score_delta),
            "frame_shape": [int(before.shape[0]), int(before.shape[1])],
            "spell_grid_bbox": grid_bbox,
            "grid_x_centers": [round(float(v), 3) for v in before_grid["x_centers"]],
            "grid_y_centers": [round(float(v), 3) for v in before_grid["y_centers"]],
            "grid_cell_count_before": int(len(before_grid["cells"])),
            "grid_cell_count_after": int(len(after_grid["cells"])),
            "grid_cells_before": before_grid["cells"][:16],
            "grid_cells_after": after_grid["cells"][:16],
            "active_cell_count_before": int(before_grid["active_cell_count"]),
            "active_cell_count_after": int(after_grid["active_cell_count"]),
            "active_cell_count_delta": int(active_delta),
            "hint_required_cell_count_before": int(
                before_grid["hint_required_cell_count"]
            ),
            "hint_required_cell_count_after": int(
                after_grid["hint_required_cell_count"]
            ),
            "hint_required_cell_count_delta": int(hint_delta),
            "pattern_signature_before": before_grid["pattern_signature"],
            "pattern_signature_after": after_grid["pattern_signature"],
            "selected_grid_cell": selected_grid_cell,
            "selected_spell_icon_region": selected_icon,
            "changed_cell_count": int(change["changed_cell_count"]),
            "top_progress_changed_cell_count": int(
                change["top_progress_changed_cell_count"]
            ),
            "grid_changed_cell_count": int(change["grid_changed_cell_count"]),
            "effect_region_changed_cell_count": int(
                change["effect_region_changed_cell_count"]
            ),
            "grid_change_components": change["grid_change_components"][:16],
            "effect_change_components": change["effect_change_components"][:16],
            "spell_grid_toggle_candidate": spell_grid_toggle_candidate,
            "spell_selection_candidate": spell_selection_candidate,
            "spell_cast_candidate": spell_cast_candidate,
            "spell_effect_candidate": spell_effect_candidate,
            "spell_state_delta_candidate": spell_state_delta_candidate,
            "score_closure_candidate": score_closure_candidate,
            "candidate_types": sorted(candidate_types),
            "diagnostic_only": True,
        }
    )


def validate_program_spell_pattern_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if (
        str(packet.get("schema_version", "") or "")
        != PROGRAM_SPELL_PATTERN_PACKET_SCHEMA_VERSION
    ):
        reasons.append("schema_version_mismatch")
    if str(packet.get("sensor_family", "") or "") != "program_spell_pattern_sensor":
        reasons.append("sensor_family_mismatch")
    selected_site = packet.get("selected_site")
    if selected_site is not None and not _is_xy(selected_site):
        reasons.append("invalid_selected_site")
    before_count = _safe_int(packet.get("grid_cell_count_before"))
    after_count = _safe_int(packet.get("grid_cell_count_after"))
    if before_count < 0 or after_count < 0:
        reasons.append("invalid_grid_cell_count")
    return {
        "schema_version": "eca_program_spell_pattern_validation.v1",
        "accepted": not reasons,
        "rejected_reasons": reasons,
        "family": "program_spell_pattern",
        "mechanic_state_type": _mechanic_state_type(packet),
        "goal_link_type": "spell_pattern_state_to_effect",
        "candidate_types": list(packet.get("candidate_types") or []),
    }


def summarize_program_spell_pattern_packets(
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
    grid_changed_cell_count_sum = 0
    effect_region_changed_cell_count_sum = 0
    active_cell_count_delta_sum = 0
    hint_required_cell_count_delta_sum = 0
    score_delta_sum = 0
    max_grid_cell_count = 0

    for packet in packet_rows:
        validation = validate_program_spell_pattern_packet(packet)
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
            ("spell_grid_toggle_candidate", "spell_grid_toggle_candidate_count"),
            ("spell_selection_candidate", "spell_selection_candidate_count"),
            ("spell_cast_candidate", "spell_cast_candidate_count"),
            ("spell_effect_candidate", "spell_effect_candidate_count"),
            ("spell_state_delta_candidate", "spell_state_delta_candidate_count"),
            ("score_closure_candidate", "score_closure_candidate_count"),
        ):
            if bool(packet.get(field)):
                counts[name] += 1
                by_game[game_id][name] += 1
        for candidate_type in list(packet.get("candidate_types") or []):
            candidate_type_counts[str(candidate_type)] += 1

        changed = _safe_int(packet.get("changed_cell_count"))
        grid_changed = _safe_int(packet.get("grid_changed_cell_count"))
        effect_changed = _safe_int(packet.get("effect_region_changed_cell_count"))
        active_delta = _safe_int(packet.get("active_cell_count_delta"))
        hint_delta = _safe_int(packet.get("hint_required_cell_count_delta"))
        score_delta = _safe_int(packet.get("score_delta"))
        changed_cell_count_sum += changed
        grid_changed_cell_count_sum += grid_changed
        effect_region_changed_cell_count_sum += effect_changed
        active_cell_count_delta_sum += active_delta
        hint_required_cell_count_delta_sum += hint_delta
        score_delta_sum += score_delta
        max_grid_cell_count = max(
            max_grid_cell_count,
            _safe_int(packet.get("grid_cell_count_before")),
            _safe_int(packet.get("grid_cell_count_after")),
        )

        if len(samples) < sample_limit:
            selected_grid = packet.get("selected_grid_cell")
            selected_grid = selected_grid if isinstance(selected_grid, Mapping) else {}
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "action_name": action,
                    "selected_site": packet.get("selected_site"),
                    "candidate_types": list(packet.get("candidate_types") or []),
                    "row_index": selected_grid.get("row_index"),
                    "col_index": selected_grid.get("col_index"),
                    "pattern_signature_before": packet.get("pattern_signature_before"),
                    "pattern_signature_after": packet.get("pattern_signature_after"),
                    "grid_changed_cell_count": grid_changed,
                    "effect_region_changed_cell_count": effect_changed,
                    "active_cell_count_delta": active_delta,
                    "score_delta": score_delta,
                }
            )

    return _sanitize(
        {
            "schema_version": PROGRAM_SPELL_PATTERN_DIAGNOSTIC_SENSOR_SCHEMA_VERSION,
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
            "spell_grid_toggle_candidate_count": int(
                counts.get("spell_grid_toggle_candidate_count", 0)
            ),
            "spell_selection_candidate_count": int(
                counts.get("spell_selection_candidate_count", 0)
            ),
            "spell_cast_candidate_count": int(
                counts.get("spell_cast_candidate_count", 0)
            ),
            "spell_effect_candidate_count": int(
                counts.get("spell_effect_candidate_count", 0)
            ),
            "spell_state_delta_candidate_count": int(
                counts.get("spell_state_delta_candidate_count", 0)
            ),
            "score_closure_candidate_count": int(
                counts.get("score_closure_candidate_count", 0)
            ),
            "changed_cell_count_sum": int(changed_cell_count_sum),
            "grid_changed_cell_count_sum": int(grid_changed_cell_count_sum),
            "effect_region_changed_cell_count_sum": int(
                effect_region_changed_cell_count_sum
            ),
            "active_cell_count_delta_sum": int(active_cell_count_delta_sum),
            "hint_required_cell_count_delta_sum": int(
                hint_required_cell_count_delta_sum
            ),
            "score_delta_sum": int(score_delta_sum),
            "max_grid_cell_count": int(max_grid_cell_count),
            "candidate_type_counts": dict(sorted(candidate_type_counts.items())),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "samples": samples,
            "next_planner_contract": {
                "required_before_ranking": (
                    "spell pattern posterior with cast-effect and score controls"
                ),
                "allowed_connection": (
                    "diagnostic_only_until_spell_pattern_posterior_controls"
                ),
                "action_generation_allowed": False,
            },
        }
    )


def summarize_program_spell_pattern_posterior(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Aggregate observed spell-grid transitions into posterior hypotheses."""

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    candidate_type_counts: Counter[str] = Counter()
    pattern_transition_counts: Counter[str] = Counter()
    cast_effect_counts: Counter[str] = Counter()
    score_closure_counts: Counter[str] = Counter()
    before_patterns: set[str] = set()
    after_patterns: set[str] = set()
    hypotheses: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []

    for packet in packet_rows:
        validation = validate_program_spell_pattern_packet(packet)
        accepted = bool(validation.get("accepted"))
        counts["accepted_packet_count" if accepted else "rejected_packet_count"] += 1
        if not accepted:
            continue
        game_id = str(packet.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["accepted_packet_count"] += 1
        candidate_types = [str(row) for row in list(packet.get("candidate_types") or [])]
        for candidate_type in candidate_types:
            candidate_type_counts[candidate_type] += 1
            by_game[game_id][f"{candidate_type}_observation_count"] += 1

        before_sig = _signature_key(packet.get("pattern_signature_before"))
        after_sig = _signature_key(packet.get("pattern_signature_after"))
        if before_sig:
            before_patterns.add(before_sig)
        if after_sig:
            after_patterns.add(after_sig)
        transition_key = f"{before_sig}->{after_sig}"
        if before_sig or after_sig:
            pattern_transition_counts[transition_key] += 1
            if transition_key not in hypotheses:
                hypotheses[transition_key] = {
                    "hypothesis_key": transition_key,
                    "hypothesis_type": "spell_pattern_transition",
                    "pattern_signature_before": _signature_rows(before_sig),
                    "pattern_signature_after": _signature_rows(after_sig),
                    "observation_count": 0,
                    "score_delta_sum": 0,
                    "effect_region_changed_cell_count_sum": 0,
                    "candidate_types": [],
                    "game_ids": [],
                    "sample_selected_sites": [],
                    "sample_steps": [],
                }
            hypothesis = hypotheses[transition_key]
            hypothesis["observation_count"] = _safe_int(
                hypothesis.get("observation_count")
            ) + 1
            hypothesis["score_delta_sum"] = _safe_int(
                hypothesis.get("score_delta_sum")
            ) + _safe_int(packet.get("score_delta"))
            hypothesis["effect_region_changed_cell_count_sum"] = _safe_int(
                hypothesis.get("effect_region_changed_cell_count_sum")
            ) + _safe_int(packet.get("effect_region_changed_cell_count"))
            hypothesis["candidate_types"] = sorted(
                set(list(hypothesis.get("candidate_types") or []) + candidate_types)
            )
            hypothesis["game_ids"] = sorted(
                set(list(hypothesis.get("game_ids") or []) + [game_id])
            )
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

        if bool(packet.get("spell_grid_toggle_candidate")):
            counts["spell_grid_toggle_observation_count"] += 1
        if bool(packet.get("spell_selection_candidate")):
            counts["spell_selection_observation_count"] += 1
        if bool(packet.get("spell_cast_candidate")):
            counts["spell_cast_observation_count"] += 1
        if bool(packet.get("spell_effect_candidate")):
            counts["spell_effect_observation_count"] += 1
        if bool(packet.get("score_closure_candidate")):
            counts["score_closure_observation_count"] += 1
        if bool(packet.get("spell_cast_candidate")) and bool(
            packet.get("spell_effect_candidate")
        ):
            effect_key = f"{after_sig}:effect"
            cast_effect_counts[effect_key] += 1
        if bool(packet.get("score_closure_candidate")):
            score_key = f"{after_sig}:score"
            score_closure_counts[score_key] += 1

        if len(samples) < sample_limit and (
            bool(packet.get("spell_grid_toggle_candidate"))
            or bool(packet.get("spell_cast_candidate"))
            or bool(packet.get("score_closure_candidate"))
        ):
            selected_grid = packet.get("selected_grid_cell")
            selected_grid = selected_grid if isinstance(selected_grid, Mapping) else {}
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "selected_site": packet.get("selected_site"),
                    "candidate_types": candidate_types,
                    "row_index": selected_grid.get("row_index"),
                    "col_index": selected_grid.get("col_index"),
                    "pattern_signature_before": packet.get(
                        "pattern_signature_before"
                    ),
                    "pattern_signature_after": packet.get("pattern_signature_after"),
                    "grid_changed_cell_count": _safe_int(
                        packet.get("grid_changed_cell_count")
                    ),
                    "effect_region_changed_cell_count": _safe_int(
                        packet.get("effect_region_changed_cell_count")
                    ),
                    "score_delta": _safe_int(packet.get("score_delta")),
                }
            )

    accepted_count = int(counts.get("accepted_packet_count", 0))
    if int(counts.get("score_closure_observation_count", 0)) > 0:
        promotion_state = "program_spell_pattern_score_closure_posterior"
    elif int(counts.get("spell_effect_observation_count", 0)) > 0:
        promotion_state = "program_spell_pattern_cast_effect_posterior"
    elif int(counts.get("spell_cast_observation_count", 0)) > 0 or int(
        counts.get("spell_grid_toggle_observation_count", 0)
    ) > 0:
        promotion_state = "program_spell_pattern_observed_posterior"
    elif accepted_count > 0:
        promotion_state = "program_spell_pattern_insufficient_effect_evidence"
    else:
        promotion_state = "program_spell_pattern_no_evidence"

    hypothesis_rows = sorted(
        hypotheses.values(),
        key=lambda row: (
            -_safe_int(row.get("score_delta_sum")),
            -_safe_int(row.get("effect_region_changed_cell_count_sum")),
            -_safe_int(row.get("observation_count")),
            str(row.get("hypothesis_key", "")),
        ),
    )[:sample_limit]

    return _sanitize(
        {
            "schema_version": PROGRAM_SPELL_PATTERN_POSTERIOR_SCHEMA_VERSION,
            "default_off": True,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "accepted_packet_count": accepted_count,
            "rejected_packet_count": int(counts.get("rejected_packet_count", 0)),
            "spell_grid_toggle_observation_count": int(
                counts.get("spell_grid_toggle_observation_count", 0)
            ),
            "spell_selection_observation_count": int(
                counts.get("spell_selection_observation_count", 0)
            ),
            "spell_cast_observation_count": int(
                counts.get("spell_cast_observation_count", 0)
            ),
            "spell_effect_observation_count": int(
                counts.get("spell_effect_observation_count", 0)
            ),
            "score_closure_observation_count": int(
                counts.get("score_closure_observation_count", 0)
            ),
            "pattern_transition_count": int(sum(pattern_transition_counts.values())),
            "unique_pattern_transition_count": int(len(pattern_transition_counts)),
            "unique_pattern_before_count": int(len(before_patterns)),
            "unique_pattern_after_count": int(len(after_patterns)),
            "cast_effect_relation_count": int(sum(cast_effect_counts.values())),
            "score_closure_relation_count": int(sum(score_closure_counts.values())),
            "posterior_hypothesis_count": int(len(hypotheses)),
            "candidate_type_counts": dict(sorted(candidate_type_counts.items())),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "hypotheses": hypothesis_rows,
            "samples": samples,
            "promotion_state": promotion_state,
            "next_runtime_contract": {
                "required_before_candidate_miner": (
                    "posterior hypotheses must be converted into guarded spell "
                    "toggle/cast candidates with scoreless and wrong-pattern controls"
                ),
                "allowed_connection": "posterior_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_program_spell_pattern_candidate_miner(
    packets: Iterable[Mapping[str, Any]],
    *,
    posterior_summary: Mapping[str, Any] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Mine guarded-probe candidate descriptions from spell-pattern posterior."""

    posterior = (
        dict(posterior_summary)
        if isinstance(posterior_summary, Mapping)
        else summarize_program_spell_pattern_posterior(
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
        kind = str(hypothesis.get("hypothesis_type", "") or "")
        candidate_types = [
            str(row) for row in list(hypothesis.get("candidate_types") or [])
        ]
        candidate_type_set = set(candidate_types)
        before_signature = list(hypothesis.get("pattern_signature_before") or [])
        after_signature = list(hypothesis.get("pattern_signature_after") or [])
        game_ids = [
            str(game_id)
            for game_id in list(hypothesis.get("game_ids") or [])
            if str(game_id)
        ]
        game_id = game_ids[0] if game_ids else "unknown"
        by_game.setdefault(game_id, Counter())
        score_delta_sum = _safe_int(hypothesis.get("score_delta_sum"))
        effect_sum = _safe_int(
            hypothesis.get("effect_region_changed_cell_count_sum")
        )
        candidate_kind = ""
        evidence_role = ""
        if kind != "spell_pattern_transition":
            rejected_reason = "unsupported_hypothesis_type"
        elif "score_closure" in candidate_type_set or score_delta_sum > 0:
            candidate_kind = "spell_score_closure_replay_candidate"
            evidence_role = "spell_score_closure"
            rejected_reason = ""
        elif "spell_cast" in candidate_type_set and (
            "spell_effect" in candidate_type_set or effect_sum > 0
        ):
            candidate_kind = "spell_cast_effect_probe_candidate"
            evidence_role = "spell_cast_effect"
            rejected_reason = ""
        elif (
            "spell_grid_toggle" in candidate_type_set
            and before_signature != after_signature
        ):
            candidate_kind = "spell_grid_toggle_retest_candidate"
            evidence_role = "spell_grid_toggle"
            rejected_reason = ""
        else:
            rejected_reason = "non_intervention_pattern_transition"

        if rejected_reason:
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "posterior_hypothesis_index": int(index),
                        "game_id": game_id,
                        "rejected_reason": rejected_reason,
                        "hypothesis_type": kind,
                        "candidate_types": candidate_types,
                        "pattern_signature_before": before_signature,
                        "pattern_signature_after": after_signature,
                    }
                )
            continue

        selected_sites = [
            list(site)
            for site in list(hypothesis.get("sample_selected_sites") or [])
            if _is_xy(site)
        ]
        candidate_key = (
            f"program_spell_pattern:{game_id}:{candidate_kind}:"
            f"{_text_hash(str(hypothesis.get('hypothesis_key', '')))}:"
            f"idx{int(index)}"
        )
        row = {
            "candidate_key": candidate_key,
            "candidate_kind": candidate_kind,
            "game_id": game_id,
            "game_ids": game_ids,
            "source_action_name": "ACTION6",
            "selected_sites_observed": selected_sites[:4],
            "evidence_role": evidence_role,
            "pattern_signature_before": before_signature,
            "pattern_signature_after": after_signature,
            "candidate_types": candidate_types,
            "observation_count": _safe_int(hypothesis.get("observation_count")),
            "effect_region_changed_cell_count_sum": int(effect_sum),
            "score_delta_sum": int(score_delta_sum),
            "sample_steps": list(hypothesis.get("sample_steps") or [])[:4],
            "requires_guarded_probe": True,
            "wrong_pattern_control_required": True,
            "scoreless_effect_control_required": True,
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
        promotion_state = "program_spell_pattern_candidate_miner_supported"
    elif hypotheses:
        promotion_state = "program_spell_pattern_candidate_miner_blocked"
    else:
        promotion_state = "program_spell_pattern_candidate_miner_no_posterior"

    return _sanitize(
        {
            "schema_version": PROGRAM_SPELL_PATTERN_CANDIDATE_MINER_SCHEMA_VERSION,
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
            "spell_grid_toggle_retest_candidate_count": int(
                by_kind.get("spell_grid_toggle_retest_candidate", Counter()).get(
                    "candidate_count", 0
                )
            ),
            "spell_cast_effect_probe_candidate_count": int(
                by_kind.get("spell_cast_effect_probe_candidate", Counter()).get(
                    "candidate_count", 0
                )
            ),
            "spell_score_closure_replay_candidate_count": int(
                by_kind.get("spell_score_closure_replay_candidate", Counter()).get(
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
                    "guarded probe with scoreless-effect and wrong-pattern controls"
                ),
                "allowed_connection": "program_spell_candidate_design_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_program_spell_pattern_guarded_probe_outcomes(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize observed outcomes from default-off spell-pattern probes."""

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
            preflight = unsolved.get("program_spell_pattern_guarded_probe_preflight")
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
                "program_spell_pattern_guarded_probe"
            )
            is_guarded = is_guarded or str(unsolved.get("source", "") or "") == (
                "program_spell_pattern_guarded_probe"
            )
            if not is_guarded:
                continue

            packet = item.get("program_spell_pattern_packet")
            packet = packet if isinstance(packet, Mapping) else {}
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
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
            effect_delta = _safe_int(packet.get("effect_region_changed_cell_count"))
            grid_delta = _safe_int(packet.get("grid_changed_cell_count"))
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
            counts["effect_region_changed_cell_count_sum"] += int(effect_delta)
            counts["grid_changed_cell_count_sum"] += int(grid_delta)
            if score_delta > 0:
                outcome_state = "score_positive"
                counts["score_positive_execution_count"] += 1
                by_game[game_id]["score_positive_execution_count"] += 1
                by_candidate_kind[candidate_kind]["score_positive_execution_count"] += 1
            elif effect_delta > 0:
                outcome_state = "scoreless_effect_change"
                counts["scoreless_effect_change_execution_count"] += 1
                by_game[game_id]["scoreless_effect_change_execution_count"] += 1
                by_candidate_kind[candidate_kind][
                    "scoreless_effect_change_execution_count"
                ] += 1
            elif grid_delta > 0 or changed_pixel_count > 0:
                outcome_state = "scoreless_pattern_edit"
                counts["scoreless_pattern_edit_execution_count"] += 1
                by_game[game_id]["scoreless_pattern_edit_execution_count"] += 1
                by_candidate_kind[candidate_kind][
                    "scoreless_pattern_edit_execution_count"
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
                        "pattern_admission_mode": str(
                            unsolved.get("pattern_admission_mode", "") or ""
                        ),
                        "outcome_state": outcome_state,
                        "score_delta": int(score_delta),
                        "effect_region_changed_cell_count": int(effect_delta),
                        "grid_changed_cell_count": int(grid_delta),
                        "changed_pixel_count": int(changed_pixel_count),
                    }
                )

    execution_count = int(counts.get("execution_count", 0))
    if execution_count <= 0 and int(counts.get("preflight_count", 0)) > 0:
        promotion_state = "program_spell_pattern_guarded_probe_preflight_only"
    elif execution_count <= 0:
        promotion_state = "program_spell_pattern_guarded_probe_no_execution"
    elif int(counts.get("score_positive_execution_count", 0)) > 0:
        promotion_state = "program_spell_pattern_guarded_probe_score_positive"
    elif int(counts.get("scoreless_effect_change_execution_count", 0)) > 0:
        promotion_state = "program_spell_pattern_guarded_probe_scoreless_effect"
    elif int(counts.get("scoreless_pattern_edit_execution_count", 0)) > 0:
        promotion_state = "program_spell_pattern_guarded_probe_scoreless_pattern_edit"
    else:
        promotion_state = "program_spell_pattern_guarded_probe_scoreless_no_effect"

    return _sanitize(
        {
            "schema_version": PROGRAM_SPELL_PATTERN_GUARDED_PROBE_OUTCOME_SCHEMA_VERSION,
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
            "scoreless_effect_change_execution_count": int(
                counts.get("scoreless_effect_change_execution_count", 0)
            ),
            "scoreless_pattern_edit_execution_count": int(
                counts.get("scoreless_pattern_edit_execution_count", 0)
            ),
            "scoreless_no_effect_execution_count": int(
                counts.get("scoreless_no_effect_execution_count", 0)
            ),
            "score_delta_sum": int(counts.get("score_delta_sum", 0)),
            "effect_region_changed_cell_count_sum": int(
                counts.get("effect_region_changed_cell_count_sum", 0)
            ),
            "grid_changed_cell_count_sum": int(
                counts.get("grid_changed_cell_count_sum", 0)
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
                    "spell closure from scoreless effect and wrong-pattern "
                    "probe outcomes before ranking changes"
                ),
                "allowed_connection": "guarded_probe_outcome_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_program_spell_pattern_counterfactual_controls(
    rows: Iterable[Mapping[str, Any]],
    *,
    outcome_summary: Mapping[str, Any] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Check whether spell-pattern guarded outcomes have positive controls."""

    outcome = (
        dict(outcome_summary)
        if isinstance(outcome_summary, Mapping)
        else summarize_program_spell_pattern_guarded_probe_outcomes(
            rows,
            sample_limit=sample_limit,
        )
    )
    score_positive = _safe_int(outcome.get("score_positive_execution_count"))
    scoreless_effect = _safe_int(
        outcome.get("scoreless_effect_change_execution_count")
    )
    scoreless_pattern = _safe_int(
        outcome.get("scoreless_pattern_edit_execution_count")
    )
    scoreless_no_effect = _safe_int(
        outcome.get("scoreless_no_effect_execution_count")
    )
    scoreless_control = int(
        scoreless_effect + scoreless_pattern + scoreless_no_effect
    )
    scoreless_execution_control = int(scoreless_control)
    control_pair = int(score_positive > 0 and scoreless_control > 0)
    margin = int(score_positive - scoreless_control)
    positive_control_margin = int(control_pair > 0 and margin >= 0)

    if score_positive <= 0 and scoreless_execution_control > 0:
        promotion_state = "program_spell_pattern_counterfactual_control_missing_positive"
    elif score_positive <= 0:
        promotion_state = "program_spell_pattern_counterfactual_control_no_guarded_outcome"
    elif scoreless_control <= 0:
        promotion_state = "program_spell_pattern_counterfactual_control_no_scoreless_control"
    elif positive_control_margin > 0:
        promotion_state = "program_spell_pattern_counterfactual_control_supported"
    else:
        promotion_state = "program_spell_pattern_counterfactual_control_dominated"

    samples = [
        dict(row)
        for row in list(outcome.get("samples") or [])
        if isinstance(row, Mapping)
    ][:sample_limit]

    return _sanitize(
        {
            "schema_version": PROGRAM_SPELL_PATTERN_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION,
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
            "scoreless_effect_change_execution_count": int(scoreless_effect),
            "scoreless_pattern_edit_execution_count": int(scoreless_pattern),
            "scoreless_no_effect_execution_count": int(scoreless_no_effect),
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
                    "positive score-bearing spell closure must survive "
                    "scoreless effect and wrong-pattern controls"
                ),
                "allowed_connection": "counterfactual_control_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_program_spell_pattern_all25_non_regression(
    run_rows: Iterable[Mapping[str, Any]],
    *,
    expected_score_sum: int | None = None,
    expected_by_game: Mapping[str, Any] | None = None,
    require_no_game_conditioned_primitives: bool = True,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Check all25 preservation while spell-pattern diagnostics are enabled."""

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
        reason_counts["program_spell_pattern_all25_non_regression_supported"] += 1
        promotion_state = "program_spell_pattern_all25_non_regression_supported"
    elif rows:
        promotion_state = "program_spell_pattern_all25_non_regression_blocked"
    else:
        promotion_state = "program_spell_pattern_all25_non_regression_no_rows"

    return _sanitize(
        {
            "schema_version": PROGRAM_SPELL_PATTERN_ALL25_NON_REGRESSION_SCHEMA_VERSION,
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
                    "spell-pattern guarded and counterfactual diagnostics must "
                    "preserve the all25 reference profile before any ranking or "
                    "planner connection is enabled"
                ),
                "allowed_connection": "all25_non_regression_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def _spell_grid_observation(arr: np.ndarray) -> dict[str, Any]:
    components = _grid_cell_components(arr)
    x_centers = _merge_centers([float(row["centroid"][0]) for row in components])
    y_centers = _merge_centers([float(row["centroid"][1]) for row in components])
    cells: list[dict[str, Any]] = []
    active = 0
    hints = 0
    for component in components:
        cx = float(component["centroid"][0])
        cy = float(component["centroid"][1])
        col = _nearest_index(x_centers, cx, max_distance=2.5)
        row = _nearest_index(y_centers, cy, max_distance=2.5)
        color = int(component.get("color", -1))
        if color == 14:
            active += 1
        if color == 0:
            hints += 1
        item = dict(component)
        item["row_index"] = row
        item["col_index"] = col
        item["cell_state"] = _grid_cell_state(color)
        cells.append(item)
    cells = sorted(
        cells,
        key=lambda item: (
            99 if item.get("row_index") is None else int(item["row_index"]),
            99 if item.get("col_index") is None else int(item["col_index"]),
            float(item["centroid"][1]),
            float(item["centroid"][0]),
        ),
    )
    return {
        "cells": cells,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "active_cell_count": int(active),
        "hint_required_cell_count": int(hints),
        "pattern_signature": _pattern_signature(cells),
    }


def _grid_cell_components(arr: np.ndarray) -> list[dict[str, Any]]:
    height = int(arr.shape[0])
    y_min = max(8, int(round(height * 0.68)))
    candidates: list[dict[str, Any]] = []
    for component in _connected_components(arr):
        color = int(component.get("color", -1))
        if color not in (0, 2, 9, 14):
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
        if area != 9 or width != 3 or height_box != 3:
            continue
        candidates.append(dict(component))
    if len(candidates) <= 9:
        return sorted(candidates, key=lambda row: (float(row["centroid"][1]), float(row["centroid"][0])))
    xs = _merge_centers([float(row["centroid"][0]) for row in candidates])
    ys = _merge_centers([float(row["centroid"][1]) for row in candidates])
    # Prefer the densest 3x3 cluster in the lower panel.
    best: list[dict[str, Any]] = []
    for xi in range(max(1, len(xs) - 2)):
        for yi in range(max(1, len(ys) - 2)):
            x_window = xs[xi : xi + 3]
            y_window = ys[yi : yi + 3]
            if len(x_window) < 3 or len(y_window) < 3:
                continue
            rows = [
                row
                for row in candidates
                if any(abs(float(row["centroid"][0]) - x) <= 2.5 for x in x_window)
                and any(abs(float(row["centroid"][1]) - y) <= 2.5 for y in y_window)
            ]
            if len(rows) > len(best):
                best = rows
    return sorted(best or candidates[:9], key=lambda row: (float(row["centroid"][1]), float(row["centroid"][0])))


def _selected_grid_cell(
    before: np.ndarray,
    after: np.ndarray,
    *,
    selected_site: tuple[int, int] | None,
    before_grid: Mapping[str, Any],
    after_grid: Mapping[str, Any],
) -> dict[str, Any] | None:
    if selected_site is None:
        return None
    sx, sy = selected_site
    if not (0 <= sx < int(before.shape[1]) and 0 <= sy < int(before.shape[0])):
        return None
    cells = list(before_grid.get("cells") or []) + list(after_grid.get("cells") or [])
    best: Mapping[str, Any] | None = None
    best_distance = 1e9
    for cell in cells:
        centroid = cell.get("centroid")
        if not isinstance(centroid, Sequence) or len(centroid) < 2:
            continue
        distance = math.hypot(float(centroid[0]) - sx, float(centroid[1]) - sy)
        if distance < best_distance:
            best_distance = distance
            best = cell
    if best is None or best_distance > 3.5:
        return None
    before_color = int(before[sy, sx])
    after_color = int(after[sy, sx])
    return {
        "site": [int(sx), int(sy)],
        "row_index": best.get("row_index"),
        "col_index": best.get("col_index"),
        "cell_centroid": list(best.get("centroid") or []),
        "before_color": before_color,
        "after_color": after_color,
        "before_state": _grid_cell_state(before_color),
        "after_state": _grid_cell_state(after_color),
        "changed": bool(before_color != after_color),
        "distance_to_cell": round(float(best_distance), 3),
    }


def _selected_spell_icon_region(
    *,
    selected_site: tuple[int, int] | None,
    grid_bbox: Sequence[int] | None,
    frame_shape: Sequence[int],
) -> dict[str, Any] | None:
    if selected_site is None or grid_bbox is None:
        return None
    sx, sy = selected_site
    x0, y0, _x1, _y1 = [int(v) for v in grid_bbox[:4]]
    if sx < max(0, x0 - 1) and sy < y0:
        return {
            "site": [int(sx), int(sy)],
            "region": "spell_icon_panel_left_of_grid",
            "relative_to_grid": [int(sx - x0), int(sy - y0)],
            "frame_shape": [int(frame_shape[0]), int(frame_shape[1])],
        }
    return None


def _change_observation(
    before: np.ndarray,
    after: np.ndarray,
    *,
    grid_bbox: Sequence[int] | None,
) -> dict[str, Any]:
    changed = before != after
    height = int(before.shape[0])
    y_indices = np.arange(height)[:, None]
    top_limit = max(2, int(round(height * 0.08)))
    top_mask = changed & (y_indices <= top_limit)
    grid_mask = np.zeros_like(changed, dtype=bool)
    if grid_bbox is not None and _is_bbox(grid_bbox):
        x0, y0, x1, y1 = [int(v) for v in grid_bbox[:4]]
        x0 = max(0, x0 - 1)
        y0 = max(0, y0 - 1)
        x1 = min(int(before.shape[1]) - 1, x1 + 1)
        y1 = min(int(before.shape[0]) - 1, y1 + 1)
        grid_mask[y0 : y1 + 1, x0 : x1 + 1] = True
    grid_change_mask = changed & grid_mask
    effect_mask = changed & ~top_mask & ~grid_change_mask
    return {
        "changed_cell_count": int(np.count_nonzero(changed)),
        "top_progress_changed_cell_count": int(np.count_nonzero(top_mask)),
        "grid_changed_cell_count": int(np.count_nonzero(grid_change_mask)),
        "effect_region_changed_cell_count": int(np.count_nonzero(effect_mask)),
        "grid_change_components": _components_from_mask(grid_change_mask, before=before, after=after),
        "effect_change_components": _components_from_mask(effect_mask, before=before, after=after)[:16],
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
                    "id": f"ch{len(out)}",
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


def _pattern_signature(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    grid = [["?" for _ in range(3)] for _ in range(3)]
    for cell in cells:
        row = cell.get("row_index")
        col = cell.get("col_index")
        if row is None or col is None:
            continue
        try:
            r = int(row)
            c = int(col)
        except (TypeError, ValueError):
            continue
        if 0 <= r < 3 and 0 <= c < 3:
            state = str(cell.get("cell_state", "") or "")
            grid[r][c] = {
                "active": "X",
                "required_hint": "h",
                "inactive": ".",
                "cast_colored": "c",
            }.get(state, "?")
    return ["".join(row) for row in grid]


def _signature_key(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ""
    rows = [str(row) for row in value]
    return "|".join(rows)


def _signature_rows(value: str) -> list[str]:
    if not value:
        return []
    return [str(row) for row in str(value).split("|")]


def _text_hash(value: str) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return digest[:12]


def _grid_cell_state(color: int) -> str:
    if int(color) == 14:
        return "active"
    if int(color) == 0:
        return "required_hint"
    if int(color) == 2:
        return "inactive"
    if int(color) == 9:
        return "cast_colored"
    return "other"


def _merge_centers(values: Sequence[float], *, tolerance: float = 2.0) -> list[float]:
    centers: list[float] = []
    for value in sorted(float(v) for v in values):
        if not centers or abs(value - centers[-1]) > tolerance:
            centers.append(value)
        else:
            centers[-1] = (centers[-1] + value) / 2.0
    return centers


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
        return "spell_pattern_score_closure"
    if "spell_effect" in types:
        return "spell_cast_effect_relation"
    if "spell_cast" in types:
        return "spell_pattern_cast_relation"
    if "spell_grid_toggle" in types:
        return "spell_grid_toggle_relation"
    if "spell_selection" in types:
        return "spell_selection_relation"
    return "spell_grid_observation"


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
    "PROGRAM_SPELL_PATTERN_ALL25_NON_REGRESSION_SCHEMA_VERSION",
    "PROGRAM_SPELL_PATTERN_CANDIDATE_MINER_SCHEMA_VERSION",
    "PROGRAM_SPELL_PATTERN_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION",
    "PROGRAM_SPELL_PATTERN_DIAGNOSTIC_SENSOR_SCHEMA_VERSION",
    "PROGRAM_SPELL_PATTERN_GUARDED_PROBE_OUTCOME_SCHEMA_VERSION",
    "PROGRAM_SPELL_PATTERN_PACKET_SCHEMA_VERSION",
    "PROGRAM_SPELL_PATTERN_POSTERIOR_SCHEMA_VERSION",
    "build_program_spell_pattern_packet",
    "summarize_program_spell_pattern_all25_non_regression",
    "summarize_program_spell_pattern_candidate_miner",
    "summarize_program_spell_pattern_counterfactual_controls",
    "summarize_program_spell_pattern_guarded_probe_outcomes",
    "summarize_program_spell_pattern_posterior",
    "summarize_program_spell_pattern_packets",
    "validate_program_spell_pattern_packet",
]
