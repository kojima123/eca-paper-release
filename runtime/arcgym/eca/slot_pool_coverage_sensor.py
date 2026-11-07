"""Frame-only slot/pool/coverage sensor packets.

The packet is intentionally observational. It does not choose actions or
authorize a solver; it exposes slot, pool, and coverage deficits that later ECA
posterior logic can validate.
"""
from __future__ import annotations

import hashlib
import heapq
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SLOT_POOL_COVERAGE_PACKET_SCHEMA_VERSION = "eca_slot_pool_coverage_sensor_packet.v1"
SLOT_POOL_COVERAGE_PROTECTED_PROPOSAL_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_protected_proposal.v1"
)
SLOT_POOL_COVERAGE_DIAGNOSTIC_SENSOR_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_diagnostic_sensor.v1"
)
SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_posterior_packet.v1"
)
SLOT_POOL_COVERAGE_POSTERIOR_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_posterior.v1"
)
SLOT_POOL_COVERAGE_CANDIDATE_MINER_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_candidate_miner.v1"
)
SLOT_POOL_COVERAGE_GUARDED_PROBE_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_guarded_probe_outcome_sensor.v1"
)
SLOT_POOL_COVERAGE_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_counterfactual_control_sensor.v1"
)
SLOT_POOL_COVERAGE_SCORELESS_PROGRESS_CONTROL_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_scoreless_progress_control_sensor.v1"
)
SLOT_POOL_COVERAGE_SCOPED_SCORELESS_CONTROL_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_scoped_scoreless_control_sensor.v1"
)
SLOT_POOL_COVERAGE_TARGET_SLOT_SCORELESS_CONTROL_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_target_slot_scoreless_control_sensor.v1"
)
SLOT_POOL_COVERAGE_EXACT_SLOT_SCORELESS_CONTROL_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_exact_slot_scoreless_control_sensor.v1"
)
SLOT_POOL_COVERAGE_REGION_IDENTITY_CONTROL_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_region_identity_control_sensor.v1"
)
SLOT_POOL_COVERAGE_EFFECT_INTENT_CONTROL_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_effect_intent_control_sensor.v1"
)
SLOT_POOL_COVERAGE_TERMINAL_EFFECT_SCORE_CLOSURE_ATTRIBUTION_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_terminal_effect_score_closure_attribution_sensor.v1"
)
SLOT_POOL_COVERAGE_SCORE_EVENT_SCOPE_ATTRIBUTION_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_score_event_scope_attribution_sensor.v1"
)
SLOT_POOL_COVERAGE_INTERVENTION_CONTRAST_ATTRIBUTION_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_intervention_contrast_attribution_sensor.v1"
)
SLOT_POOL_COVERAGE_INTERVENTION_TEMPORAL_COUNTERFACTUAL_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_intervention_temporal_counterfactual_sensor.v1"
)
SLOT_POOL_COVERAGE_ALREADY_SCORED_EVENT_EXCLUSION_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_already_scored_event_exclusion_sensor.v1"
)
SLOT_POOL_COVERAGE_NEGATIVE_MEMORY_GUARDED_REVIEW_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_negative_memory_guarded_review_sensor.v1"
)
SLOT_POOL_COVERAGE_NEGATIVE_MEMORY_RANKING_SHADOW_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_negative_memory_ranking_shadow_sensor.v1"
)
SLOT_POOL_COVERAGE_FOCUSED_REPLAY_READINESS_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_focused_replay_readiness_sensor.v1"
)
SLOT_POOL_COVERAGE_FOCUSED_REPLAY_EXECUTION_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_focused_replay_execution_sensor.v1"
)
SLOT_POOL_COVERAGE_FOCUSED_REPLAY_OUTCOME_ADMISSION_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_focused_replay_outcome_admission_guard_sensor.v1"
)
SLOT_POOL_COVERAGE_FOCUSED_SCORE_BEARING_CANDIDATE_DISCOVERY_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_focused_score_bearing_candidate_discovery_sensor.v1"
)
SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_SCORE_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_submit_precondition_score_sensor.v1"
)
SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RUNTIME_BRIDGE_GUARD_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_submit_precondition_runtime_bridge_guard_sensor.v1"
)
SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RUNTIME_BRIDGE_OUTCOME_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_submit_precondition_runtime_bridge_outcome_sensor.v1"
)
SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_BREADTH_NON_REGRESSION_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_submit_precondition_breadth_non_regression_sensor.v1"
)
SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RANK_GUARD_SHADOW_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_submit_precondition_rank_guard_shadow_sensor.v1"
)
SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RANK_GUARD_RUNTIME_GUARD_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_submit_precondition_rank_guard_runtime_guard_sensor.v1"
)
SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_LIVE_RANK_GUARD_FOCUSED_RUNTIME_SCHEMA_VERSION = (
    "eca_slot_pool_coverage_submit_precondition_live_rank_guard_focused_runtime.v1"
)


def build_slot_pool_coverage_packet(
    frame: Any,
    *,
    game_id: str = "",
    state_hash: str = "",
    step: int = 0,
) -> dict[str, Any]:
    """Build a compact slot/pool/coverage packet from a public frame."""

    arr = _as_frame(frame)
    game = str(game_id or "").strip().lower()
    if game.startswith("sb26"):
        packet = _build_sb26_packet(arr)
    elif game.startswith("cd82"):
        packet = _build_cd82_packet(arr)
    elif game.startswith("sp80"):
        packet = _build_sp80_packet(arr)
    elif game.startswith("ar25"):
        packet = _build_ar25_packet(arr)
    elif game.startswith("re86"):
        packet = _build_re86_packet(arr)
    else:
        packet = _build_generic_packet(arr)
    packet.update(
        {
            "schema_version": SLOT_POOL_COVERAGE_PACKET_SCHEMA_VERSION,
            "sensor_family": "slot_pool_coverage_sensor",
            "game_id": game,
            "state_hash": str(state_hash or _frame_hash(arr)),
            "step": int(step),
        }
    )
    return _sanitize(packet)


def validate_slot_pool_coverage_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a slot/pool/coverage packet as a descriptor source."""

    reasons: list[str] = []
    if str(packet.get("schema_version", "") or "") != SLOT_POOL_COVERAGE_PACKET_SCHEMA_VERSION:
        reasons.append("schema_version_mismatch")
    if str(packet.get("sensor_family", "") or "") != "slot_pool_coverage_sensor":
        reasons.append("sensor_family_mismatch")
    slot_count = _safe_int(packet.get("slot_count"))
    required = _safe_int(packet.get("coverage_required_cell_count"))
    observed = _safe_int(packet.get("coverage_observed_cell_count"))
    deficit = _safe_int(packet.get("coverage_deficit_cell_count"))
    pool_count = _safe_int(packet.get("pool_object_count"))
    binding_count = _safe_int(packet.get("slot_candidate_binding_count"))
    if slot_count <= 0:
        reasons.append("missing_slots")
    if required <= 0:
        reasons.append("missing_coverage_requirement")
    if observed < 0 or observed > required:
        reasons.append("invalid_observed_coverage")
    if deficit != max(0, required - observed):
        reasons.append("coverage_deficit_mismatch")
    if deficit > 0 and pool_count <= 0:
        reasons.append("missing_pool_objects")
    if deficit > 0 and binding_count <= 0:
        reasons.append("missing_slot_candidate_bindings")
    coverage = packet.get("coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    accepted = not reasons
    return _sanitize(
        {
            "schema_version": "eca_slot_pool_coverage_validation.v1",
            "accepted": bool(accepted),
            "rejected_reasons": reasons,
            "family": "slot_pool_coverage",
            "mechanic_state_type": str(coverage.get("kind", "") or "slot_pool_coverage"),
            "goal_link_type": "coverage_deficit_reduction",
            "slot_count": slot_count,
            "pool_count": pool_count,
            "coverage_required_cell_count": required,
            "coverage_observed_cell_count": observed,
            "coverage_deficit_cell_count": deficit,
            "ambiguous_binding_count": _safe_int(packet.get("ambiguous_binding_count")),
            "pool_exhausted": bool(packet.get("pool_exhausted", False)),
        }
    )


def summarize_slot_pool_coverage_packets(
    packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 12,
) -> dict[str, Any]:
    """Summarize slot/pool/coverage packets without authorizing execution."""

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    counts: Counter[str] = Counter()
    coverage_kind_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    samples: list[dict[str, Any]] = []
    slot_count_sum = 0
    unfilled_slot_count_sum = 0
    pool_object_count_sum = 0
    binding_count_sum = 0
    required_cell_count_sum = 0
    observed_cell_count_sum = 0
    deficit_cell_count_sum = 0

    for packet in packet_rows:
        validation = validate_slot_pool_coverage_packet(packet)
        accepted = bool(validation.get("accepted"))
        counts["accepted_packet_count" if accepted else "rejected_packet_count"] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        coverage = packet.get("coverage")
        coverage = coverage if isinstance(coverage, Mapping) else {}
        kind = str(coverage.get("kind", "") or "unknown")
        coverage_kind_counts[kind] += 1
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["packet_count"] += 1
        by_game[game_id][f"coverage_kind:{kind}"] += 1

        slot_count = _safe_int(packet.get("slot_count"))
        unfilled_slot_count = _safe_int(packet.get("unfilled_slot_count"))
        pool_count = _safe_int(packet.get("pool_object_count"))
        binding_count = _safe_int(packet.get("slot_candidate_binding_count"))
        required = _safe_int(packet.get("coverage_required_cell_count"))
        observed = _safe_int(packet.get("coverage_observed_cell_count"))
        deficit = _safe_int(packet.get("coverage_deficit_cell_count"))
        slot_count_sum += slot_count
        unfilled_slot_count_sum += unfilled_slot_count
        pool_object_count_sum += pool_count
        binding_count_sum += binding_count
        required_cell_count_sum += required
        observed_cell_count_sum += observed
        deficit_cell_count_sum += deficit
        if unfilled_slot_count > 0:
            counts["unfilled_slot_detected_count"] += 1
            by_game[game_id]["unfilled_slot_detected_count"] += 1
        if binding_count > 0:
            counts["slot_candidate_binding_packet_count"] += 1
        if deficit > 0:
            counts["coverage_deficit_detected_count"] += 1
            by_game[game_id]["coverage_deficit_detected_count"] += 1
        if bool(packet.get("pool_exhausted")):
            counts["pool_exhausted_packet_count"] += 1

        if len(samples) < sample_limit:
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "coverage_kind": kind,
                    "slot_count": slot_count,
                    "unfilled_slot_count": unfilled_slot_count,
                    "pool_object_count": pool_count,
                    "slot_candidate_binding_count": binding_count,
                    "coverage_required_cell_count": required,
                    "coverage_observed_cell_count": observed,
                    "coverage_deficit_cell_count": deficit,
                    "accepted": accepted,
                }
            )

    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_DIAGNOSTIC_SENSOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "accepted_packet_count": int(counts.get("accepted_packet_count", 0)),
            "rejected_packet_count": int(counts.get("rejected_packet_count", 0)),
            "unfilled_slot_detected_count": int(
                counts.get("unfilled_slot_detected_count", 0)
            ),
            "slot_candidate_binding_packet_count": int(
                counts.get("slot_candidate_binding_packet_count", 0)
            ),
            "coverage_deficit_detected_count": int(
                counts.get("coverage_deficit_detected_count", 0)
            ),
            "pool_exhausted_packet_count": int(
                counts.get("pool_exhausted_packet_count", 0)
            ),
            "slot_count_sum": int(slot_count_sum),
            "unfilled_slot_count_sum": int(unfilled_slot_count_sum),
            "pool_object_count_sum": int(pool_object_count_sum),
            "slot_candidate_binding_count_sum": int(binding_count_sum),
            "coverage_required_cell_count_sum": int(required_cell_count_sum),
            "coverage_observed_cell_count_sum": int(observed_cell_count_sum),
            "coverage_deficit_cell_count_sum": int(deficit_cell_count_sum),
            "coverage_kind_counts": dict(sorted(coverage_kind_counts.items())),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "samples": samples,
            "next_planner_contract": {
                "required_before_ranking": (
                    "coverage-deficit posterior with score/control attribution"
                ),
                "allowed_connection": (
                    "diagnostic_only_until_slot_pool_posterior_controls"
                ),
                "action_generation_allowed": False,
            },
        }
    )


def build_slot_pool_coverage_posterior_packet(
    before_packet: Mapping[str, Any],
    after_packet: Mapping[str, Any],
    *,
    score_delta: int = 0,
    plan_completed: bool = False,
    repair_target: str = "",
    action_name: str = "",
    action_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare two slot/pool/coverage packets and emit a progress posterior."""

    before_validation = validate_slot_pool_coverage_packet(before_packet)
    after_validation = validate_slot_pool_coverage_packet(after_packet)
    before_deficit = _safe_int(before_packet.get("coverage_deficit_cell_count"))
    after_deficit = _safe_int(after_packet.get("coverage_deficit_cell_count"))
    before_filled = _safe_int(before_packet.get("filled_slot_count"))
    after_filled = _safe_int(after_packet.get("filled_slot_count"))
    deficit_delta = int(after_deficit - before_deficit)
    filled_delta = int(after_filled - before_filled)
    score_positive = int(score_delta) > 0
    objective_progress = bool(score_positive or deficit_delta < 0 or filled_delta > 0)
    if not bool(before_validation.get("accepted")) or not bool(after_validation.get("accepted")):
        posterior = "rejected"
        falsified_reason = "descriptor_validation_failed"
    elif objective_progress:
        posterior = "confirmed"
        falsified_reason = ""
    elif bool(plan_completed):
        posterior = "falsified"
        falsified_reason = "no_slot_pool_coverage_progress_after_completed_plan"
    else:
        posterior = "ambiguous"
        falsified_reason = ""
    slot_scope = _slot_pool_posterior_slot_scope(before_packet, after_packet)
    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION,
            "sensor_family": "slot_pool_coverage_sensor",
            "family": "slot_pool_coverage",
            "game_id": str(
                after_packet.get("game_id", "")
                or before_packet.get("game_id", "")
                or ""
            ),
            "state_hash_before": str(before_packet.get("state_hash", "") or ""),
            "state_hash_after": str(after_packet.get("state_hash", "") or ""),
            "step_before": _safe_int(before_packet.get("step")),
            "step_after": _safe_int(after_packet.get("step")),
            "action_name": str(action_name or ""),
            "action_payload": dict(action_payload or {}),
            "mechanic_state_type": str(
                before_validation.get("mechanic_state_type", "")
                or after_validation.get("mechanic_state_type", "")
                or "slot_pool_coverage"
            ),
            "goal_link_type": "coverage_deficit_reduction",
            "repair_target": str(repair_target or ""),
            "score_delta": int(score_delta),
            "plan_completed": bool(plan_completed),
            "before_validation": before_validation,
            "after_validation": after_validation,
            "before": {
                "slot_count": _safe_int(before_packet.get("slot_count")),
                "filled_slot_count": before_filled,
                "coverage_deficit_cell_count": before_deficit,
            },
            "after": {
                "slot_count": _safe_int(after_packet.get("slot_count")),
                "filled_slot_count": after_filled,
                "coverage_deficit_cell_count": after_deficit,
            },
            "deltas": {
                "coverage_deficit_delta": int(deficit_delta),
                "filled_slot_delta": int(filled_delta),
            },
            "slot_scope": slot_scope,
            "objective_progress": bool(objective_progress),
            "scoreless_objective_progress": bool(objective_progress and not score_positive),
            "posterior": posterior,
            "falsified_reason": falsified_reason,
        }
    )


def summarize_slot_pool_coverage_posterior(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize slot/pool coverage progress posteriors without planner effects."""

    packet_rows = [
        dict(packet)
        for packet in posterior_packets
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_kind: dict[str, Counter[str]] = {}
    samples: list[dict[str, Any]] = []
    score_delta_sum = 0
    confirmed_score_delta_sum = 0
    scoreless_objective_progress_count = 0
    falsified_score_delta_sum = 0
    no_progress_score_delta_sum = 0
    deficit_improvement_sum = 0
    filled_improvement_sum = 0

    for packet in packet_rows:
        posterior = str(packet.get("posterior", "") or "unknown")
        game_id = str(packet.get("game_id", "") or "unknown")
        mechanic = str(packet.get("mechanic_state_type", "") or "slot_pool_coverage")
        score_delta = _safe_int(packet.get("score_delta"))
        deltas = packet.get("deltas")
        deltas = deltas if isinstance(deltas, Mapping) else {}
        deficit_delta = _safe_int(deltas.get("coverage_deficit_delta"))
        filled_delta = _safe_int(deltas.get("filled_slot_delta"))
        score_delta_sum += score_delta
        counts[f"posterior_state:{posterior}"] += 1
        by_game.setdefault(game_id, Counter())
        by_kind.setdefault(mechanic, Counter())
        by_game[game_id]["packet_count"] += 1
        by_kind[mechanic]["packet_count"] += 1
        by_game[game_id][f"posterior_state:{posterior}"] += 1
        by_kind[mechanic][f"posterior_state:{posterior}"] += 1
        if posterior == "confirmed":
            counts["confirmed_count"] += 1
            confirmed_score_delta_sum += max(0, score_delta)
            if score_delta > 0:
                counts["confirmed_score_closure_count"] += 1
            if bool(packet.get("scoreless_objective_progress")):
                scoreless_objective_progress_count += 1
        elif posterior == "falsified":
            counts["falsified_count"] += 1
            falsified_score_delta_sum += max(0, score_delta)
        elif posterior == "ambiguous":
            counts["ambiguous_count"] += 1
            no_progress_score_delta_sum += max(0, score_delta)
        else:
            counts["insufficient_evidence_count"] += 1
        if deficit_delta < 0:
            deficit_improvement_sum += abs(deficit_delta)
        if filled_delta > 0:
            filled_improvement_sum += filled_delta
        if len(samples) < sample_limit:
            samples.append(
                {
                    "game_id": game_id,
                    "step_before": _safe_int(packet.get("step_before")),
                    "step_after": _safe_int(packet.get("step_after")),
                    "action_name": str(packet.get("action_name", "") or ""),
                    "mechanic_state_type": mechanic,
                    "posterior": posterior,
                    "score_delta": score_delta,
                    "coverage_deficit_delta": deficit_delta,
                    "filled_slot_delta": filled_delta,
                    "falsified_reason": str(packet.get("falsified_reason", "") or ""),
                }
            )

    negative_control = max(falsified_score_delta_sum, no_progress_score_delta_sum)
    margin = int(confirmed_score_delta_sum - negative_control)
    if counts.get("confirmed_score_closure_count", 0) > 0 and margin > 0:
        promotion_state = "slot_pool_score_closure_control_margin"
    elif counts.get("confirmed_count", 0) > 0:
        promotion_state = "slot_pool_objective_progress_confirmed"
    elif counts.get("falsified_count", 0) > 0:
        promotion_state = "slot_pool_falsified_controls_observed"
    elif counts.get("ambiguous_count", 0) > 0:
        promotion_state = "slot_pool_ambiguous"
    else:
        promotion_state = "no_slot_pool_posterior_evidence"

    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_POSTERIOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "confirmed_count": int(counts.get("confirmed_count", 0)),
            "confirmed_score_closure_count": int(
                counts.get("confirmed_score_closure_count", 0)
            ),
            "scoreless_objective_progress_count": int(
                scoreless_objective_progress_count
            ),
            "falsified_count": int(counts.get("falsified_count", 0)),
            "ambiguous_count": int(counts.get("ambiguous_count", 0)),
            "insufficient_evidence_count": int(
                counts.get("insufficient_evidence_count", 0)
            ),
            "score_delta_sum": int(score_delta_sum),
            "confirmed_score_delta_sum": int(confirmed_score_delta_sum),
            "falsified_negative_control_score_delta_sum": int(
                falsified_score_delta_sum
            ),
            "no_progress_negative_control_score_delta_sum": int(
                no_progress_score_delta_sum
            ),
            "negative_control_max_score_delta_sum": int(negative_control),
            "slot_pool_counterfactual_margin": int(margin),
            "positive_control_margin_count": int(margin > 0),
            "coverage_deficit_improvement_sum": int(deficit_improvement_sum),
            "filled_slot_improvement_sum": int(filled_improvement_sum),
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
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_kind.items())
            },
            "samples": samples,
            "next_planner_contract": {
                "required_before_ranking": (
                    "candidate miner plus guarded probe controls after slot-pool posterior"
                ),
                "allowed_connection": (
                    "diagnostic_only_until_slot_pool_candidate_miner_and_controls"
                ),
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_candidates(
    packets: Iterable[Mapping[str, Any]],
    *,
    posterior_packets: Iterable[Mapping[str, Any]] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Compile slot-pool descriptors into protected-probe candidate designs.

    The miner is design-only. It reuses the verifier-owned protected proposal
    builder and records whether posterior evidence supports the same
    game/mechanic family, but it does not execute, rank, or authorize actions.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    posterior_rows = [
        dict(packet)
        for packet in list(posterior_packets or [])
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    supported_keys = {
        _posterior_support_key(packet)
        for packet in posterior_rows
        if str(packet.get("posterior", "") or "") == "confirmed"
    }
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for packet in packet_rows:
        validation = validate_slot_pool_coverage_packet(packet)
        accepted_packet = bool(validation.get("accepted"))
        counts["accepted_packet_count" if accepted_packet else "rejected_packet_count"] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        mechanic = str(validation.get("mechanic_state_type", "") or "slot_pool_coverage")
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["packet_count"] += 1
        by_mechanic[mechanic]["packet_count"] += 1
        if not accepted_packet:
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "game_id": game_id,
                        "step": _safe_int(packet.get("step")),
                        "rejected_reasons": list(validation.get("rejected_reasons") or []),
                    }
                )
            continue

        proposal = build_slot_pool_coverage_protected_proposal(packet)
        counts["proposal_attempt_count"] += 1
        proposal_accepted = bool(proposal.get("accepted")) and _safe_int(
            proposal.get("plan_length")
        ) > 0
        if not proposal_accepted:
            counts["rejected_proposal_count"] += 1
            reason = str(
                (list(proposal.get("rejected_reasons") or []) or ["empty_or_rejected_plan"])[0]
            )
            by_game[game_id][f"rejected_reason:{reason}"] += 1
            by_mechanic[mechanic][f"rejected_reason:{reason}"] += 1
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "game_id": game_id,
                        "step": _safe_int(packet.get("step")),
                        "mechanic_state_type": mechanic,
                        "rejected_reasons": list(proposal.get("rejected_reasons") or [])
                        or ["empty_or_rejected_plan"],
                        "plan_length": _safe_int(proposal.get("plan_length")),
                    }
                )
            continue

        support_key = _packet_support_key(packet, validation)
        posterior_supported = support_key in supported_keys
        if posterior_supported:
            counts["posterior_supported_candidate_count"] += 1
            by_game[game_id]["posterior_supported_candidate_count"] += 1
            by_mechanic[mechanic]["posterior_supported_candidate_count"] += 1
        counts["accepted_proposal_count"] += 1
        by_game[game_id]["candidate_count"] += 1
        by_mechanic[mechanic]["candidate_count"] += 1
        candidate = {
            "candidate_key": _slot_pool_candidate_key(packet, validation, proposal),
            "candidate_kind": "slot_pool_coverage_protected_probe_design",
            "game_id": game_id,
            "step": _safe_int(packet.get("step")),
            "mechanic_state_type": mechanic,
            "coverage_kind": mechanic,
            "posterior_supported": bool(posterior_supported),
            "plan_length": _safe_int(proposal.get("plan_length")),
            "slot_fill_count": _safe_int(proposal.get("slot_fill_count")),
            "plan_roles": [
                str(row.get("role", "") or "")
                for row in list(proposal.get("plan") or [])
                if isinstance(row, Mapping)
            ],
            "evidence": {
                "coverage_required_cell_count": _safe_int(
                    packet.get("coverage_required_cell_count")
                ),
                "coverage_observed_cell_count": _safe_int(
                    packet.get("coverage_observed_cell_count")
                ),
                "coverage_deficit_cell_count": _safe_int(
                    packet.get("coverage_deficit_cell_count")
                ),
                "slot_candidate_binding_count": _safe_int(
                    packet.get("slot_candidate_binding_count")
                ),
                "posterior_support_key": support_key,
            },
            "runtime_execution_enabled": False,
            "requires_guarded_probe": True,
            "requires_positive_control_margin": True,
            "direct_llm_action_allowed": False,
            "candidate_generation_owner": "eca_verifier",
        }
        if len(candidates) < sample_limit:
            candidates.append(candidate)

    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_CANDIDATE_MINER_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "posterior_packet_count": int(len(posterior_rows)),
            "posterior_confirmed_packet_count": int(len(supported_keys)),
            "accepted_packet_count": int(counts.get("accepted_packet_count", 0)),
            "rejected_packet_count": int(counts.get("rejected_packet_count", 0)),
            "proposal_attempt_count": int(counts.get("proposal_attempt_count", 0)),
            "accepted_proposal_count": int(counts.get("accepted_proposal_count", 0)),
            "rejected_proposal_count": int(counts.get("rejected_proposal_count", 0)),
            "candidate_count": int(counts.get("accepted_proposal_count", 0)),
            "posterior_supported_candidate_count": int(
                counts.get("posterior_supported_candidate_count", 0)
            ),
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "candidates": candidates,
            "rejected_samples": rejected,
            "next_runtime_contract": {
                "required_before_execution": (
                    "default-off guarded slot-pool probe with posterior and paired controls"
                ),
                "allowed_connection": "candidate_design_only_until_guarded_probe_enabled",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_guarded_probe_outcomes(
    packets: Iterable[Mapping[str, Any]],
    *,
    posterior_packets: Iterable[Mapping[str, Any]] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize guarded-probe admission for slot-pool candidate designs.

    This is still design-only. It admits metadata only when the candidate is a
    protected slot-pool proposal and the posterior has positive control margin
    for the same game/mechanic family.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    posterior_rows = [
        dict(packet)
        for packet in list(posterior_packets or [])
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    posterior = summarize_slot_pool_coverage_posterior(posterior_rows)
    miner = summarize_slot_pool_coverage_candidates(
        packet_rows,
        posterior_packets=posterior_rows,
        sample_limit=max(sample_limit, 32),
    )
    has_positive_margin = _safe_int(posterior.get("positive_control_margin_count")) > 0
    reason_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    admitted: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for candidate in list(miner.get("candidates") or []):
        if not isinstance(candidate, Mapping):
            continue
        game_id = str(candidate.get("game_id", "") or "unknown")
        mechanic = str(candidate.get("mechanic_state_type", "") or "slot_pool_coverage")
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["candidate_count"] += 1
        by_mechanic[mechanic]["candidate_count"] += 1
        reasons: list[str] = []
        if str(candidate.get("candidate_kind", "") or "") != (
            "slot_pool_coverage_protected_probe_design"
        ):
            reasons.append("unsupported_candidate_kind")
        if _safe_int(candidate.get("plan_length")) <= 0:
            reasons.append("missing_protected_plan")
        if not bool(candidate.get("posterior_supported")):
            reasons.append("missing_posterior_supported_candidate")
        if not has_positive_margin:
            reasons.append("missing_positive_control_margin")

        if reasons:
            reason = reasons[0]
            reason_counts[reason] += 1
            by_game[game_id][f"blocked_reason:{reason}"] += 1
            by_mechanic[mechanic][f"blocked_reason:{reason}"] += 1
            if len(blocked) < sample_limit:
                blocked.append(
                    {
                        "candidate_key": str(candidate.get("candidate_key", "") or ""),
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "blocked_reason": reason,
                        "blocked_reasons": reasons,
                        "plan_length": _safe_int(candidate.get("plan_length")),
                    }
                )
            continue

        reason_counts["admitted_guarded_slot_pool_probe_design"] += 1
        by_game[game_id]["admitted_guarded_probe_count"] += 1
        by_mechanic[mechanic]["admitted_guarded_probe_count"] += 1
        if len(admitted) < sample_limit:
            admitted.append(
                {
                    "candidate_key": str(candidate.get("candidate_key", "") or ""),
                    "game_id": game_id,
                    "mechanic_state_type": mechanic,
                    "guard_state": "admitted_guarded_probe_design",
                    "plan_length": _safe_int(candidate.get("plan_length")),
                    "slot_fill_count": _safe_int(candidate.get("slot_fill_count")),
                    "runtime_execution_enabled": False,
                    "requires_current_descriptor_match": True,
                    "requires_state_hash_match": True,
                    "requires_positive_control_margin": True,
                }
            )

    admitted_count = int(
        reason_counts.get("admitted_guarded_slot_pool_probe_design", 0)
    )
    blocked_count = sum(
        int(value)
        for key, value in reason_counts.items()
        if key != "admitted_guarded_slot_pool_probe_design"
    )
    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_GUARDED_PROBE_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "posterior_packet_count": int(len(posterior_rows)),
            "positive_control_margin_count": _safe_int(
                posterior.get("positive_control_margin_count")
            ),
            "candidate_count": _safe_int(miner.get("candidate_count")),
            "posterior_supported_candidate_count": _safe_int(
                miner.get("posterior_supported_candidate_count")
            ),
            "admitted_guarded_probe_count": admitted_count,
            "blocked_guarded_probe_count": int(blocked_count),
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "admitted": admitted,
            "blocked": blocked,
            "next_runtime_contract": {
                "required_before_execution": (
                    "counterfactual-control sensor plus focused/breadth/all25 preservation"
                ),
                "allowed_connection": "guarded_probe_design_only_until_runtime_phase",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_counterfactual_controls(
    packets: Iterable[Mapping[str, Any]],
    *,
    posterior_packets: Iterable[Mapping[str, Any]] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize counterfactual controls for admitted slot-pool designs."""

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    posterior_rows = [
        dict(packet)
        for packet in list(posterior_packets or [])
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    posterior = summarize_slot_pool_coverage_posterior(posterior_rows)
    guarded = summarize_slot_pool_coverage_guarded_probe_outcomes(
        packet_rows,
        posterior_packets=posterior_rows,
        sample_limit=max(sample_limit, 32),
    )
    negative_control_count = int(
        _safe_int(posterior.get("falsified_count"))
        + _safe_int(posterior.get("ambiguous_count"))
        + _safe_int(posterior.get("insufficient_evidence_count"))
    )
    positive_margin = _safe_int(posterior.get("positive_control_margin_count"))
    reason_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    supported: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for candidate in list(guarded.get("admitted") or []):
        if not isinstance(candidate, Mapping):
            continue
        game_id = str(candidate.get("game_id", "") or "unknown")
        mechanic = str(candidate.get("mechanic_state_type", "") or "slot_pool_coverage")
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["admitted_guarded_probe_count"] += 1
        by_mechanic[mechanic]["admitted_guarded_probe_count"] += 1
        reasons: list[str] = []
        if positive_margin <= 0:
            reasons.append("missing_positive_control_margin")
        if negative_control_count <= 0:
            reasons.append("missing_negative_control_packet")
        if str(candidate.get("guard_state", "") or "") != "admitted_guarded_probe_design":
            reasons.append("missing_guarded_probe_admission")

        if reasons:
            reason = reasons[0]
            reason_counts[reason] += 1
            by_game[game_id][f"blocked_reason:{reason}"] += 1
            by_mechanic[mechanic][f"blocked_reason:{reason}"] += 1
            if len(blocked) < sample_limit:
                blocked.append(
                    {
                        "candidate_key": str(candidate.get("candidate_key", "") or ""),
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "blocked_reason": reason,
                        "blocked_reasons": reasons,
                    }
                )
            continue

        reason_counts["counterfactual_control_supported"] += 1
        by_game[game_id]["counterfactual_control_pair_count"] += 1
        by_mechanic[mechanic]["counterfactual_control_pair_count"] += 1
        if len(supported) < sample_limit:
            supported.append(
                {
                    "candidate_key": str(candidate.get("candidate_key", "") or ""),
                    "game_id": game_id,
                    "mechanic_state_type": mechanic,
                    "control_state": "counterfactual_control_supported",
                    "positive_control_margin_count": int(positive_margin),
                    "negative_control_packet_count": int(negative_control_count),
                    "runtime_execution_enabled": False,
                }
            )

    supported_count = int(reason_counts.get("counterfactual_control_supported", 0))
    blocked_count = sum(
        int(value)
        for key, value in reason_counts.items()
        if key != "counterfactual_control_supported"
    )
    if supported_count > 0:
        promotion_state = "slot_pool_counterfactual_control_supported"
    elif _safe_int(guarded.get("admitted_guarded_probe_count")) > 0:
        promotion_state = "slot_pool_counterfactual_control_missing_controls"
    else:
        promotion_state = "slot_pool_counterfactual_control_no_admitted_probe"

    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "posterior_packet_count": int(len(posterior_rows)),
            "positive_control_margin_count": int(positive_margin),
            "negative_control_packet_count": int(negative_control_count),
            "candidate_count": _safe_int(guarded.get("candidate_count")),
            "admitted_guarded_probe_count": _safe_int(
                guarded.get("admitted_guarded_probe_count")
            ),
            "counterfactual_control_pair_count": int(supported_count),
            "blocked_candidate_count": int(blocked_count),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "supported": supported,
            "blocked": blocked,
            "next_runtime_contract": {
                "required_before_execution": (
                    "focused/breadth/all25 non-regression before any runtime promotion"
                ),
                "allowed_connection": "counterfactual_control_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_scoreless_progress_controls(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Separate scoreless slot-pool progress from no-progress controls."""

    packet_rows = [
        dict(packet)
        for packet in posterior_packets
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    samples: list[dict[str, Any]] = []
    deficit_improvement_sum = 0
    filled_improvement_sum = 0

    for packet in packet_rows:
        posterior = str(packet.get("posterior", "") or "unknown")
        game_id = str(packet.get("game_id", "") or "unknown")
        mechanic = str(packet.get("mechanic_state_type", "") or "slot_pool_coverage")
        score_delta = _safe_int(packet.get("score_delta"))
        deltas = packet.get("deltas")
        deltas = deltas if isinstance(deltas, Mapping) else {}
        deficit_delta = _safe_int(deltas.get("coverage_deficit_delta"))
        filled_delta = _safe_int(deltas.get("filled_slot_delta"))
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["posterior_packet_count"] += 1
        by_mechanic[mechanic]["posterior_packet_count"] += 1

        if posterior == "confirmed" and score_delta > 0:
            state = "score_closure_progress"
        elif posterior == "confirmed" and bool(packet.get("scoreless_objective_progress")):
            state = "scoreless_objective_progress"
            if deficit_delta < 0:
                deficit_improvement_sum += abs(deficit_delta)
            if filled_delta > 0:
                filled_improvement_sum += filled_delta
        elif posterior in {"falsified", "ambiguous"}:
            state = posterior
        else:
            state = "insufficient_evidence"

        counts[state] += 1
        by_game[game_id][state] += 1
        by_mechanic[mechanic][state] += 1
        if len(samples) < sample_limit:
            samples.append(
                {
                    "game_id": game_id,
                    "step_before": _safe_int(packet.get("step_before")),
                    "step_after": _safe_int(packet.get("step_after")),
                    "mechanic_state_type": mechanic,
                    "posterior": posterior,
                    "control_state": state,
                    "score_delta": score_delta,
                    "coverage_deficit_delta": deficit_delta,
                    "filled_slot_delta": filled_delta,
                }
            )

    scoreless_count = int(counts.get("scoreless_objective_progress", 0))
    score_closure_count = int(counts.get("score_closure_progress", 0))
    insufficient_count = int(counts.get("insufficient_evidence", 0))
    falsified_count = int(counts.get("falsified", 0))
    ambiguous_count = int(counts.get("ambiguous", 0))
    negative_count = int(falsified_count + ambiguous_count + insufficient_count)
    control_margin = int(scoreless_count - negative_count)
    scoreless_supported = int(scoreless_count > 0 and negative_count > 0 and control_margin > 0)
    if score_closure_count > 0:
        promotion_state = "score_closure_progress_present"
    elif scoreless_supported:
        promotion_state = "scoreless_objective_progress_control_supported"
    elif scoreless_count > 0 and negative_count > 0:
        promotion_state = "scoreless_objective_progress_control_dominated"
    elif scoreless_count > 0 and negative_count == 0:
        promotion_state = "scoreless_objective_progress_no_controls"
    elif negative_count > 0:
        promotion_state = "negative_controls_without_scoreless_progress"
    else:
        promotion_state = "no_scoreless_progress_evidence"

    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_SCORELESS_PROGRESS_CONTROL_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "posterior_packet_count": int(len(packet_rows)),
            "score_closure_progress_count": int(score_closure_count),
            "scoreless_objective_progress_count": int(scoreless_count),
            "negative_control_packet_count": int(negative_count),
            "falsified_count": int(falsified_count),
            "ambiguous_count": int(ambiguous_count),
            "insufficient_evidence_count": int(insufficient_count),
            "scoreless_progress_control_margin": int(control_margin),
            "scoreless_positive_control_margin_count": int(scoreless_supported),
            "scoreless_progress_control_supported_count": int(scoreless_supported),
            "planner_promotion_blocked": bool(
                score_closure_count <= 0 and scoreless_supported <= 0
            ),
            "coverage_deficit_improvement_sum": int(deficit_improvement_sum),
            "filled_slot_improvement_sum": int(filled_improvement_sum),
            "deficit_improvement_sum": int(deficit_improvement_sum),
            "filled_slot_improvement_sum": int(filled_improvement_sum),
            "promotion_state": promotion_state,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "samples": samples,
            "next_runtime_contract": {
                "required_before_execution": (
                    "positive scoreless-progress margin plus focused/breadth/all25 safety"
                ),
                "allowed_connection": "diagnostic_only_until_scoreless_controls_supported",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_scoped_scoreless_controls(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Compare scoreless progress only against same game/mechanic controls.

    The coarse scoreless-control sensor intentionally counts every rejected or
    insufficient posterior as negative evidence. That is useful as a safety
    blocker, but too noisy for promotion triage. This scoped variant only uses
    ambiguous/falsified packets from keys that also have scoreless objective
    progress, keeping unrelated generic-component noise out of the margin.
    """

    packet_rows = [
        dict(packet)
        for packet in posterior_packets
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    by_key: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    excluded_insufficient_count = 0
    excluded_unpaired_negative_count = 0
    score_closure_count = 0

    for packet in packet_rows:
        posterior = str(packet.get("posterior", "") or "unknown")
        game_id = str(packet.get("game_id", "") or "unknown")
        mechanic = str(packet.get("mechanic_state_type", "") or "slot_pool_coverage")
        key = _posterior_support_key(packet)
        score_delta = _safe_int(packet.get("score_delta"))
        deltas = packet.get("deltas")
        deltas = deltas if isinstance(deltas, Mapping) else {}
        deficit_delta = _safe_int(deltas.get("coverage_deficit_delta"))
        filled_delta = _safe_int(deltas.get("filled_slot_delta"))
        row = by_key.setdefault(
            key,
            {
                "scope_key": key,
                "game_id": game_id,
                "mechanic_state_type": mechanic,
                "posterior_packet_count": 0,
                "score_closure_progress_count": 0,
                "scoreless_objective_progress_count": 0,
                "ambiguous_count": 0,
                "falsified_count": 0,
                "insufficient_evidence_count": 0,
                "coverage_deficit_improvement_sum": 0,
                "filled_slot_improvement_sum": 0,
            },
        )
        row["posterior_packet_count"] = _safe_int(row["posterior_packet_count"]) + 1

        if posterior == "confirmed" and score_delta > 0:
            row["score_closure_progress_count"] = (
                _safe_int(row["score_closure_progress_count"]) + 1
            )
            score_closure_count += 1
            control_state = "score_closure_progress"
        elif posterior == "confirmed" and bool(packet.get("scoreless_objective_progress")):
            row["scoreless_objective_progress_count"] = (
                _safe_int(row["scoreless_objective_progress_count"]) + 1
            )
            if deficit_delta < 0:
                row["coverage_deficit_improvement_sum"] = (
                    _safe_int(row["coverage_deficit_improvement_sum"])
                    + abs(deficit_delta)
                )
            if filled_delta > 0:
                row["filled_slot_improvement_sum"] = (
                    _safe_int(row["filled_slot_improvement_sum"]) + filled_delta
                )
            control_state = "scoreless_objective_progress"
        elif posterior == "ambiguous":
            row["ambiguous_count"] = _safe_int(row["ambiguous_count"]) + 1
            control_state = "ambiguous"
        elif posterior == "falsified":
            row["falsified_count"] = _safe_int(row["falsified_count"]) + 1
            control_state = "falsified"
        else:
            row["insufficient_evidence_count"] = (
                _safe_int(row["insufficient_evidence_count"]) + 1
            )
            excluded_insufficient_count += 1
            control_state = "insufficient_evidence"

        if len(samples) < sample_limit:
            samples.append(
                {
                    "scope_key": key,
                    "game_id": game_id,
                    "step_before": _safe_int(packet.get("step_before")),
                    "step_after": _safe_int(packet.get("step_after")),
                    "mechanic_state_type": mechanic,
                    "posterior": posterior,
                    "control_state": control_state,
                    "score_delta": score_delta,
                    "coverage_deficit_delta": deficit_delta,
                    "filled_slot_delta": filled_delta,
                }
            )

    scoped_rows: list[dict[str, Any]] = []
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    positive_key_count = 0
    negative_key_count = 0
    scoped_positive_count = 0
    scoped_negative_count = 0
    improvement_sum = 0
    filled_sum = 0

    for row in by_key.values():
        positive = _safe_int(row["scoreless_objective_progress_count"])
        negatives = _safe_int(row["ambiguous_count"]) + _safe_int(row["falsified_count"])
        if positive <= 0:
            excluded_unpaired_negative_count += negatives
            continue
        positive_key_count += 1
        if negatives > 0:
            negative_key_count += 1
        margin = int(positive - negatives)
        row = dict(row)
        row["scoped_negative_control_count"] = int(negatives)
        row["scoped_scoreless_progress_control_margin"] = int(margin)
        row["scoped_positive_control_margin_count"] = int(negatives > 0 and margin > 0)
        scoped_rows.append(row)
        scoped_positive_count += positive
        scoped_negative_count += negatives
        improvement_sum += _safe_int(row["coverage_deficit_improvement_sum"])
        filled_sum += _safe_int(row["filled_slot_improvement_sum"])
        game_id = str(row["game_id"])
        mechanic = str(row["mechanic_state_type"])
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["scoped_positive_key_count"] += 1
        by_game[game_id]["scoped_scoreless_objective_progress_count"] += positive
        by_game[game_id]["scoped_negative_control_count"] += negatives
        by_game[game_id]["scoped_scoreless_progress_control_margin"] += margin
        by_mechanic[mechanic]["scoped_positive_key_count"] += 1
        by_mechanic[mechanic]["scoped_scoreless_objective_progress_count"] += positive
        by_mechanic[mechanic]["scoped_negative_control_count"] += negatives
        by_mechanic[mechanic]["scoped_scoreless_progress_control_margin"] += margin

    scoped_rows.sort(
        key=lambda row: (
            -_safe_int(row["scoreless_objective_progress_count"]),
            _safe_int(row["scoped_scoreless_progress_control_margin"]),
            str(row["scope_key"]),
        )
    )
    scoped_margin = int(scoped_positive_count - scoped_negative_count)
    scoped_supported = int(
        scoped_positive_count > 0 and scoped_negative_count > 0 and scoped_margin > 0
    )
    if score_closure_count > 0:
        promotion_state = "score_closure_progress_present"
    elif scoped_supported:
        promotion_state = "scoped_scoreless_control_supported"
    elif scoped_positive_count > 0 and scoped_negative_count > 0:
        promotion_state = "scoped_scoreless_control_dominated"
    elif scoped_positive_count > 0:
        promotion_state = "scoped_scoreless_progress_no_controls"
    else:
        promotion_state = "no_scoped_scoreless_progress"

    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_SCOPED_SCORELESS_CONTROL_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "posterior_packet_count": int(len(packet_rows)),
            "scope": "game_id|mechanic_state_type",
            "scoped_positive_key_count": int(positive_key_count),
            "scoped_negative_control_key_count": int(negative_key_count),
            "scoped_scoreless_objective_progress_count": int(scoped_positive_count),
            "scoped_negative_control_count": int(scoped_negative_count),
            "scoped_scoreless_progress_control_margin": int(scoped_margin),
            "scoped_positive_control_margin_count": int(scoped_supported),
            "score_closure_progress_count": int(score_closure_count),
            "excluded_insufficient_evidence_count": int(excluded_insufficient_count),
            "excluded_unpaired_negative_control_count": int(
                excluded_unpaired_negative_count
            ),
            "planner_promotion_blocked": bool(
                score_closure_count <= 0 and scoped_supported <= 0
            ),
            "coverage_deficit_improvement_sum": int(improvement_sum),
            "filled_slot_improvement_sum": int(filled_sum),
            "promotion_state": promotion_state,
            "scoped_rows": scoped_rows[:sample_limit],
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "samples": samples,
            "next_runtime_contract": {
                "required_before_execution": (
                    "positive scoped scoreless-control margin plus all25 safety"
                ),
                "allowed_connection": "diagnostic_only_until_scoped_controls_supported",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_target_slot_scoreless_controls(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Compare scoreless progress to controls that overlap the same slots."""

    packet_rows = [
        dict(packet)
        for packet in posterior_packets
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    positives: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    missing_slot_scope_count = 0
    score_closure_count = 0

    for index, packet in enumerate(packet_rows):
        posterior = str(packet.get("posterior", "") or "unknown")
        scope = packet.get("slot_scope")
        if not isinstance(scope, Mapping):
            missing_slot_scope_count += 1
            continue
        game_id = str(packet.get("game_id", "") or "unknown")
        mechanic = str(packet.get("mechanic_state_type", "") or "slot_pool_coverage")
        mechanic_key = _posterior_support_key(packet)
        resolved_tokens = [
            str(token)
            for token in list(scope.get("resolved_slot_tokens") or [])
            if str(token)
        ]
        active_tokens = [
            str(token)
            for token in list(scope.get("before_active_slot_tokens") or [])
            if str(token)
        ]
        if posterior == "confirmed" and _safe_int(packet.get("score_delta")) > 0:
            score_closure_count += 1
        elif posterior == "confirmed" and bool(packet.get("scoreless_objective_progress")):
            tokens = resolved_tokens or active_tokens
            if tokens:
                positives.append(
                    {
                        "packet_index": index,
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "mechanic_key": mechanic_key,
                        "tokens": sorted(set(tokens)),
                        "step_before": _safe_int(packet.get("step_before")),
                        "step_after": _safe_int(packet.get("step_after")),
                    }
                )
        elif posterior in {"ambiguous", "falsified"} and active_tokens:
            controls.append(
                {
                    "packet_index": index,
                    "game_id": game_id,
                    "mechanic_state_type": mechanic,
                    "mechanic_key": mechanic_key,
                    "posterior": posterior,
                    "tokens": sorted(set(active_tokens)),
                    "step_before": _safe_int(packet.get("step_before")),
                    "step_after": _safe_int(packet.get("step_after")),
                }
            )

    controls_by_mechanic: dict[str, list[dict[str, Any]]] = {}
    for control in controls:
        controls_by_mechanic.setdefault(str(control["mechanic_key"]), []).append(control)

    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    target_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    positive_token_count = 0
    overlap_pair_count = 0
    unique_control_indices: set[int] = set()

    for positive in positives:
        positive_tokens = set(str(token) for token in positive["tokens"])
        positive_token_count += len(positive_tokens)
        matched_controls = 0
        matched_token_overlap = 0
        for control in controls_by_mechanic.get(str(positive["mechanic_key"]), []):
            overlap = sorted(positive_tokens & set(str(token) for token in control["tokens"]))
            if not overlap:
                continue
            matched_controls += 1
            matched_token_overlap += len(overlap)
            overlap_pair_count += len(overlap)
            unique_control_indices.add(int(control["packet_index"]))
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "positive_packet_index": int(positive["packet_index"]),
                        "control_packet_index": int(control["packet_index"]),
                        "game_id": str(positive["game_id"]),
                        "mechanic_state_type": str(positive["mechanic_state_type"]),
                        "control_posterior": str(control["posterior"]),
                        "overlap_tokens": overlap[:8],
                        "overlap_token_count": len(overlap),
                    }
                )
        margin = int(len(positive_tokens) - matched_token_overlap)
        game_id = str(positive["game_id"])
        mechanic = str(positive["mechanic_state_type"])
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["target_slot_positive_packet_count"] += 1
        by_game[game_id]["target_slot_positive_token_count"] += len(positive_tokens)
        by_game[game_id]["target_slot_control_overlap_pair_count"] += matched_token_overlap
        by_game[game_id]["target_slot_control_margin"] += margin
        by_mechanic[mechanic]["target_slot_positive_packet_count"] += 1
        by_mechanic[mechanic]["target_slot_positive_token_count"] += len(positive_tokens)
        by_mechanic[mechanic]["target_slot_control_overlap_pair_count"] += matched_token_overlap
        by_mechanic[mechanic]["target_slot_control_margin"] += margin
        target_rows.append(
            {
                "positive_packet_index": int(positive["packet_index"]),
                "game_id": game_id,
                "mechanic_state_type": mechanic,
                "mechanic_key": str(positive["mechanic_key"]),
                "positive_token_count": len(positive_tokens),
                "matched_control_packet_count": int(matched_controls),
                "control_overlap_pair_count": int(matched_token_overlap),
                "target_slot_control_margin": int(margin),
                "sample_tokens": sorted(positive_tokens)[:8],
            }
        )

    target_rows.sort(
        key=lambda row: (
            _safe_int(row["target_slot_control_margin"]),
            -_safe_int(row["control_overlap_pair_count"]),
            str(row["mechanic_key"]),
        )
    )
    margin = int(positive_token_count - overlap_pair_count)
    positive_margin_count = int(
        positive_token_count > 0 and overlap_pair_count > 0 and margin > 0
    )
    if score_closure_count > 0:
        promotion_state = "score_closure_progress_present"
    elif positive_margin_count:
        promotion_state = "target_slot_scoreless_control_supported"
    elif positive_token_count > 0 and overlap_pair_count > 0:
        promotion_state = "target_slot_scoreless_control_dominated"
    elif positive_token_count > 0:
        promotion_state = "target_slot_scoreless_progress_no_controls"
    else:
        promotion_state = "no_target_slot_scoreless_progress"

    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_TARGET_SLOT_SCORELESS_CONTROL_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "posterior_packet_count": int(len(packet_rows)),
            "scope": "game_id|mechanic_state_type|target_slot_token_overlap",
            "missing_slot_scope_count": int(missing_slot_scope_count),
            "score_closure_progress_count": int(score_closure_count),
            "target_slot_positive_packet_count": int(len(positives)),
            "target_slot_positive_token_count": int(positive_token_count),
            "target_slot_control_packet_count": int(len(unique_control_indices)),
            "target_slot_control_overlap_pair_count": int(overlap_pair_count),
            "target_slot_scoreless_control_margin": int(margin),
            "target_slot_positive_control_margin_count": int(positive_margin_count),
            "planner_promotion_blocked": bool(
                score_closure_count <= 0 and positive_margin_count <= 0
            ),
            "promotion_state": promotion_state,
            "target_rows": target_rows[:sample_limit],
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "samples": samples,
            "next_runtime_contract": {
                "required_before_execution": (
                    "positive target-slot control margin plus all25 safety"
                ),
                "allowed_connection": "diagnostic_only_until_target_slot_controls_supported",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_exact_slot_scoreless_controls(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Compare scoreless progress using exact signatures and unique controls."""

    packet_rows = [
        dict(packet)
        for packet in posterior_packets
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    positives: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    missing_slot_scope_count = 0
    score_closure_count = 0

    for index, packet in enumerate(packet_rows):
        posterior = str(packet.get("posterior", "") or "unknown")
        scope = packet.get("slot_scope")
        if not isinstance(scope, Mapping):
            missing_slot_scope_count += 1
            continue
        game_id = str(packet.get("game_id", "") or "unknown")
        mechanic = str(packet.get("mechanic_state_type", "") or "slot_pool_coverage")
        mechanic_key = _posterior_support_key(packet)
        resolved_tokens = sorted(
            {
                str(token)
                for token in list(scope.get("resolved_slot_tokens") or [])
                if str(token)
            }
        )
        active_tokens = sorted(
            {
                str(token)
                for token in list(scope.get("before_active_slot_tokens") or [])
                if str(token)
            }
        )
        before_signature = str(
            scope.get("before_active_slot_signature", "") or _slot_token_signature(active_tokens)
        )
        resolved_signature = str(
            scope.get("resolved_slot_signature", "") or _slot_token_signature(resolved_tokens)
        )
        if posterior == "confirmed" and _safe_int(packet.get("score_delta")) > 0:
            score_closure_count += 1
        elif posterior == "confirmed" and bool(packet.get("scoreless_objective_progress")):
            tokens = resolved_tokens or active_tokens
            if tokens:
                positives.append(
                    {
                        "packet_index": index,
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "mechanic_key": mechanic_key,
                        "before_active_slot_signature": before_signature,
                        "resolved_slot_signature": resolved_signature,
                        "tokens": tokens,
                        "step_before": _safe_int(packet.get("step_before")),
                        "step_after": _safe_int(packet.get("step_after")),
                    }
                )
        elif posterior in {"ambiguous", "falsified"} and active_tokens:
            controls.append(
                {
                    "packet_index": index,
                    "game_id": game_id,
                    "mechanic_state_type": mechanic,
                    "mechanic_key": mechanic_key,
                    "posterior": posterior,
                    "before_active_slot_signature": before_signature,
                    "tokens": active_tokens,
                    "step_before": _safe_int(packet.get("step_before")),
                    "step_after": _safe_int(packet.get("step_after")),
                }
            )

    controls_by_mechanic: dict[str, list[dict[str, Any]]] = {}
    for control in controls:
        controls_by_mechanic.setdefault(str(control["mechanic_key"]), []).append(control)

    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    signature_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    exact_same_before_pair_count = 0
    unique_overlap_pair_count = 0
    unique_exact_control_indices: set[int] = set()
    unique_overlap_control_indices: set[int] = set()
    positive_token_count = 0
    positive_resolved_signatures: set[str] = set()
    positive_before_signatures: set[str] = set()

    for positive in positives:
        mechanic_key = str(positive["mechanic_key"])
        positive_tokens = set(str(token) for token in positive["tokens"])
        positive_token_count += len(positive_tokens)
        positive_resolved_signatures.add(str(positive["resolved_slot_signature"]))
        positive_before_signatures.add(str(positive["before_active_slot_signature"]))
        exact_matches = 0
        unique_overlaps = 0
        for control in controls_by_mechanic.get(mechanic_key, []):
            exact_same_before = (
                str(control["before_active_slot_signature"])
                == str(positive["before_active_slot_signature"])
            )
            overlap = sorted(positive_tokens & set(str(token) for token in control["tokens"]))
            if exact_same_before:
                exact_matches += 1
                exact_same_before_pair_count += 1
                unique_exact_control_indices.add(int(control["packet_index"]))
            if overlap:
                unique_overlaps += 1
                unique_overlap_pair_count += 1
                unique_overlap_control_indices.add(int(control["packet_index"]))
            if (exact_same_before or overlap) and len(samples) < sample_limit:
                samples.append(
                    {
                        "positive_packet_index": int(positive["packet_index"]),
                        "control_packet_index": int(control["packet_index"]),
                        "game_id": str(positive["game_id"]),
                        "mechanic_state_type": str(positive["mechanic_state_type"]),
                        "control_posterior": str(control["posterior"]),
                        "exact_same_before_signature": bool(exact_same_before),
                        "overlap_token_count": len(overlap),
                        "overlap_tokens": overlap[:8],
                    }
                )

        exact_margin = int(1 - exact_matches)
        unique_margin = int(1 - unique_overlaps)
        game_id = str(positive["game_id"])
        mechanic = str(positive["mechanic_state_type"])
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["exact_scoreless_positive_packet_count"] += 1
        by_game[game_id]["exact_same_before_signature_control_count"] += exact_matches
        by_game[game_id]["unique_overlap_control_pair_count"] += unique_overlaps
        by_game[game_id]["exact_signature_margin"] += exact_margin
        by_game[game_id]["unique_control_margin"] += unique_margin
        by_mechanic[mechanic]["exact_scoreless_positive_packet_count"] += 1
        by_mechanic[mechanic]["exact_same_before_signature_control_count"] += exact_matches
        by_mechanic[mechanic]["unique_overlap_control_pair_count"] += unique_overlaps
        by_mechanic[mechanic]["exact_signature_margin"] += exact_margin
        by_mechanic[mechanic]["unique_control_margin"] += unique_margin
        signature_rows.append(
            {
                "positive_packet_index": int(positive["packet_index"]),
                "game_id": game_id,
                "mechanic_state_type": mechanic,
                "mechanic_key": mechanic_key,
                "before_active_slot_signature": str(
                    positive["before_active_slot_signature"]
                ),
                "resolved_slot_signature": str(positive["resolved_slot_signature"]),
                "positive_token_count": len(positive_tokens),
                "exact_same_before_signature_control_count": int(exact_matches),
                "unique_overlap_control_pair_count": int(unique_overlaps),
                "exact_signature_margin": int(exact_margin),
                "unique_control_margin": int(unique_margin),
                "sample_tokens": sorted(positive_tokens)[:8],
            }
        )

    signature_rows.sort(
        key=lambda row: (
            _safe_int(row["unique_control_margin"]),
            _safe_int(row["exact_signature_margin"]),
            str(row["mechanic_key"]),
        )
    )
    positive_count = int(len(positives))
    exact_signature_margin = int(positive_count - exact_same_before_pair_count)
    unique_control_margin = int(positive_count - unique_overlap_pair_count)
    exact_supported = int(
        positive_count > 0
        and exact_same_before_pair_count > 0
        and exact_signature_margin > 0
    )
    unique_supported = int(
        positive_count > 0
        and unique_overlap_pair_count > 0
        and unique_control_margin > 0
    )
    positive_margin_count = int(exact_supported > 0 or unique_supported > 0)
    if score_closure_count > 0:
        promotion_state = "score_closure_progress_present"
    elif positive_margin_count:
        promotion_state = "exact_or_unique_control_supported"
    elif positive_count > 0 and (
        exact_same_before_pair_count > 0 or unique_overlap_pair_count > 0
    ):
        promotion_state = "exact_or_unique_control_dominated"
    elif positive_count > 0:
        promotion_state = "exact_or_unique_progress_no_controls"
    else:
        promotion_state = "no_exact_slot_scoreless_progress"

    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_EXACT_SLOT_SCORELESS_CONTROL_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "posterior_packet_count": int(len(packet_rows)),
            "scope": (
                "game_id|mechanic_state_type|exact_before_signature|"
                "unique_target_slot_overlap"
            ),
            "missing_slot_scope_count": int(missing_slot_scope_count),
            "score_closure_progress_count": int(score_closure_count),
            "exact_positive_signature_count": int(len(positive_resolved_signatures)),
            "exact_positive_before_signature_count": int(len(positive_before_signatures)),
            "exact_scoreless_positive_packet_count": int(positive_count),
            "exact_scoreless_positive_token_count": int(positive_token_count),
            "exact_same_before_signature_control_count": int(
                exact_same_before_pair_count
            ),
            "exact_same_before_signature_control_packet_count": int(
                len(unique_exact_control_indices)
            ),
            "unique_overlap_control_packet_count": int(
                len(unique_overlap_control_indices)
            ),
            "unique_overlap_control_pair_count": int(unique_overlap_pair_count),
            "exact_signature_margin": int(exact_signature_margin),
            "unique_control_margin": int(unique_control_margin),
            "exact_positive_control_margin_count": int(exact_supported),
            "unique_positive_control_margin_count": int(unique_supported),
            "positive_margin_count": int(positive_margin_count),
            "planner_promotion_blocked": bool(
                score_closure_count <= 0 and positive_margin_count <= 0
            ),
            "promotion_state": promotion_state,
            "signature_rows": signature_rows[:sample_limit],
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "samples": samples,
            "next_runtime_contract": {
                "required_before_execution": (
                    "positive exact/unique control margin plus all25 safety"
                ),
                "allowed_connection": "diagnostic_only_until_exact_slot_controls_supported",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_region_identity_controls(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Compare scoreless progress by exact slot-region identities.

    This is stricter than token overlap: a broad active deficit that merely
    contains the positive cells does not match unless the connected region
    identity itself is the same.
    """

    packet_rows = [
        dict(packet)
        for packet in posterior_packets
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    positives: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    missing_slot_scope_count = 0
    score_closure_count = 0

    for index, packet in enumerate(packet_rows):
        posterior = str(packet.get("posterior", "") or "unknown")
        scope = packet.get("slot_scope")
        if not isinstance(scope, Mapping):
            missing_slot_scope_count += 1
            continue
        game_id = str(packet.get("game_id", "") or "unknown")
        mechanic = str(packet.get("mechanic_state_type", "") or "slot_pool_coverage")
        mechanic_key = _posterior_support_key(packet)
        resolved_tokens = [
            str(token)
            for token in list(scope.get("resolved_slot_tokens") or [])
            if str(token)
        ]
        active_tokens = [
            str(token)
            for token in list(scope.get("before_active_slot_tokens") or [])
            if str(token)
        ]
        if posterior == "confirmed" and _safe_int(packet.get("score_delta")) > 0:
            score_closure_count += 1
        elif posterior == "confirmed" and bool(packet.get("scoreless_objective_progress")):
            identities = _slot_pool_region_identities(resolved_tokens or active_tokens)
            if identities:
                positives.append(
                    {
                        "packet_index": index,
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "mechanic_key": mechanic_key,
                        "identities": identities,
                        "step_before": _safe_int(packet.get("step_before")),
                        "step_after": _safe_int(packet.get("step_after")),
                    }
                )
        elif posterior in {"ambiguous", "falsified"} and active_tokens:
            identities = _slot_pool_region_identities(active_tokens)
            if identities:
                controls.append(
                    {
                        "packet_index": index,
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "mechanic_key": mechanic_key,
                        "posterior": posterior,
                        "identities": identities,
                        "step_before": _safe_int(packet.get("step_before")),
                        "step_after": _safe_int(packet.get("step_after")),
                    }
                )

    controls_by_mechanic: dict[str, list[dict[str, Any]]] = {}
    for control in controls:
        controls_by_mechanic.setdefault(str(control["mechanic_key"]), []).append(control)

    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    region_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    positive_identity_count = 0
    region_control_pair_count = 0
    unique_control_indices: set[int] = set()
    positive_region_signatures: set[str] = set()

    for positive in positives:
        positive_identities = set(str(item) for item in positive["identities"])
        positive_identity_count += len(positive_identities)
        positive_region_signatures.update(positive_identities)
        matched_controls = 0
        matched_identity_count = 0
        for control in controls_by_mechanic.get(str(positive["mechanic_key"]), []):
            overlap = sorted(
                positive_identities & set(str(item) for item in control["identities"])
            )
            if not overlap:
                continue
            matched_controls += 1
            matched_identity_count += len(overlap)
            region_control_pair_count += 1
            unique_control_indices.add(int(control["packet_index"]))
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "positive_packet_index": int(positive["packet_index"]),
                        "control_packet_index": int(control["packet_index"]),
                        "game_id": str(positive["game_id"]),
                        "mechanic_state_type": str(positive["mechanic_state_type"]),
                        "control_posterior": str(control["posterior"]),
                        "matched_region_identities": overlap[:8],
                        "matched_region_identity_count": len(overlap),
                    }
                )
        margin = int(len(positive_identities) - matched_controls)
        game_id = str(positive["game_id"])
        mechanic = str(positive["mechanic_state_type"])
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["region_positive_packet_count"] += 1
        by_game[game_id]["region_positive_identity_count"] += len(positive_identities)
        by_game[game_id]["region_control_pair_count"] += matched_controls
        by_game[game_id]["region_matched_identity_count"] += matched_identity_count
        by_game[game_id]["region_identity_margin"] += margin
        by_mechanic[mechanic]["region_positive_packet_count"] += 1
        by_mechanic[mechanic]["region_positive_identity_count"] += len(positive_identities)
        by_mechanic[mechanic]["region_control_pair_count"] += matched_controls
        by_mechanic[mechanic]["region_matched_identity_count"] += matched_identity_count
        by_mechanic[mechanic]["region_identity_margin"] += margin
        region_rows.append(
            {
                "positive_packet_index": int(positive["packet_index"]),
                "game_id": game_id,
                "mechanic_state_type": mechanic,
                "mechanic_key": str(positive["mechanic_key"]),
                "positive_region_identity_count": len(positive_identities),
                "matched_control_packet_count": int(matched_controls),
                "matched_region_identity_count": int(matched_identity_count),
                "region_identity_margin": int(margin),
                "sample_region_identities": sorted(positive_identities)[:8],
            }
        )

    region_rows.sort(
        key=lambda row: (
            _safe_int(row["region_identity_margin"]),
            -_safe_int(row["matched_control_packet_count"]),
            str(row["mechanic_key"]),
        )
    )
    region_identity_margin = int(positive_identity_count - region_control_pair_count)
    region_supported = int(
        positive_identity_count > 0
        and region_control_pair_count > 0
        and region_identity_margin > 0
    )
    if score_closure_count > 0:
        promotion_state = "score_closure_progress_present"
    elif region_supported:
        promotion_state = "region_identity_control_supported"
    elif positive_identity_count > 0 and region_control_pair_count > 0:
        promotion_state = "region_identity_control_dominated"
    elif positive_identity_count > 0:
        promotion_state = "region_identity_progress_no_controls"
    else:
        promotion_state = "no_region_identity_scoreless_progress"

    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_REGION_IDENTITY_CONTROL_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "posterior_packet_count": int(len(packet_rows)),
            "scope": "game_id|mechanic_state_type|connected_region_identity",
            "missing_slot_scope_count": int(missing_slot_scope_count),
            "score_closure_progress_count": int(score_closure_count),
            "region_positive_packet_count": int(len(positives)),
            "region_positive_signature_count": int(len(positive_region_signatures)),
            "region_positive_identity_count": int(positive_identity_count),
            "region_control_packet_count": int(len(unique_control_indices)),
            "region_control_pair_count": int(region_control_pair_count),
            "region_identity_margin": int(region_identity_margin),
            "region_positive_control_margin_count": int(region_supported),
            "planner_promotion_blocked": bool(
                score_closure_count <= 0 and region_supported <= 0
            ),
            "promotion_state": promotion_state,
            "region_rows": region_rows[:sample_limit],
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "samples": samples,
            "next_runtime_contract": {
                "required_before_execution": (
                    "positive region-identity control margin plus all25 safety"
                ),
                "allowed_connection": "diagnostic_only_until_region_identity_controls_supported",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_effect_intent_controls(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Separate terminal slot-pool intent from partial scoreless progress."""

    packet_rows = [
        dict(packet)
        for packet in posterior_packets
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    terminal_positives: list[dict[str, Any]] = []
    score_closures: list[dict[str, Any]] = []
    partial_positives: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    missing_slot_scope_count = 0
    effect_intent_counts: Counter[str] = Counter()

    for index, packet in enumerate(packet_rows):
        scope = packet.get("slot_scope")
        if not isinstance(scope, Mapping):
            missing_slot_scope_count += 1
            continue
        intent = _slot_pool_effect_intent(packet)
        effect_intent = str(intent.get("effect_intent", "") or "unknown")
        effect_intent_counts[effect_intent] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        mechanic = str(packet.get("mechanic_state_type", "") or "slot_pool_coverage")
        mechanic_key = _posterior_support_key(packet)
        identities = _slot_pool_region_identities(
            _slot_pool_scope_tokens(scope, resolved_first=True)
        )
        active_identities = _slot_pool_region_identities(
            _slot_pool_scope_tokens(scope, resolved_first=False)
        )
        row = {
            "packet_index": index,
            "game_id": game_id,
            "mechanic_state_type": mechanic,
            "mechanic_key": mechanic_key,
            "effect_intent": effect_intent,
            "identities": identities,
            "active_identities": active_identities,
            "step_before": _safe_int(packet.get("step_before")),
            "step_after": _safe_int(packet.get("step_after")),
            "score_delta": _safe_int(packet.get("score_delta")),
            "coverage_deficit_delta": _safe_int(intent.get("coverage_deficit_delta")),
            "filled_slot_delta": _safe_int(intent.get("filled_slot_delta")),
            "after_deficit": _safe_int(intent.get("after_deficit")),
            "after_filled_slot_count": _safe_int(intent.get("after_filled_slot_count")),
            "after_slot_count": _safe_int(intent.get("after_slot_count")),
        }
        if effect_intent == "score_closure":
            if identities:
                score_closures.append(row)
        elif effect_intent == "terminal_scoreless_slot_closure":
            if identities:
                terminal_positives.append(row)
        elif effect_intent.startswith("partial_") or effect_intent == (
            "scoreless_objective_progress_unclassified"
        ):
            if identities:
                partial_positives.append(row)
        elif effect_intent in {"ambiguous_no_progress_control", "falsified_no_progress_control"}:
            control_identities = active_identities or identities
            if control_identities:
                row["identities"] = control_identities
                controls.append(row)

    controls_by_mechanic: dict[str, list[dict[str, Any]]] = {}
    for control in controls:
        controls_by_mechanic.setdefault(str(control["mechanic_key"]), []).append(control)

    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    effect_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    terminal_identity_count = 0
    terminal_control_pair_count = 0
    score_closure_identity_count = 0
    partial_identity_count = 0
    unique_control_indices: set[int] = set()

    for positive in terminal_positives:
        positive_identities = set(str(item) for item in positive["identities"])
        terminal_identity_count += len(positive_identities)
        matched_controls = 0
        matched_identity_count = 0
        for control in controls_by_mechanic.get(str(positive["mechanic_key"]), []):
            overlap = sorted(
                positive_identities & set(str(item) for item in control["identities"])
            )
            if not overlap:
                continue
            matched_controls += 1
            matched_identity_count += len(overlap)
            terminal_control_pair_count += 1
            unique_control_indices.add(int(control["packet_index"]))
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "positive_packet_index": int(positive["packet_index"]),
                        "control_packet_index": int(control["packet_index"]),
                        "game_id": str(positive["game_id"]),
                        "mechanic_state_type": str(positive["mechanic_state_type"]),
                        "effect_intent": str(positive["effect_intent"]),
                        "control_effect_intent": str(control["effect_intent"]),
                        "matched_region_identities": overlap[:8],
                        "matched_region_identity_count": len(overlap),
                    }
                )
        margin = int(len(positive_identities) - matched_controls)
        game_id = str(positive["game_id"])
        mechanic = str(positive["mechanic_state_type"])
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["terminal_scoreless_slot_closure_packet_count"] += 1
        by_game[game_id]["terminal_effect_intent_identity_count"] += len(
            positive_identities
        )
        by_game[game_id]["terminal_effect_intent_control_pair_count"] += matched_controls
        by_game[game_id]["terminal_effect_intent_margin"] += margin
        by_mechanic[mechanic]["terminal_scoreless_slot_closure_packet_count"] += 1
        by_mechanic[mechanic]["terminal_effect_intent_identity_count"] += len(
            positive_identities
        )
        by_mechanic[mechanic]["terminal_effect_intent_control_pair_count"] += (
            matched_controls
        )
        by_mechanic[mechanic]["terminal_effect_intent_margin"] += margin
        effect_rows.append(
            {
                "positive_packet_index": int(positive["packet_index"]),
                "game_id": game_id,
                "mechanic_state_type": mechanic,
                "mechanic_key": str(positive["mechanic_key"]),
                "effect_intent": str(positive["effect_intent"]),
                "positive_region_identity_count": len(positive_identities),
                "matched_control_packet_count": int(matched_controls),
                "matched_region_identity_count": int(matched_identity_count),
                "terminal_effect_intent_margin": int(margin),
                "after_deficit": _safe_int(positive.get("after_deficit")),
                "after_filled_slot_count": _safe_int(
                    positive.get("after_filled_slot_count")
                ),
                "after_slot_count": _safe_int(positive.get("after_slot_count")),
                "sample_region_identities": sorted(positive_identities)[:8],
            }
        )

    for positive in score_closures:
        identities = set(str(item) for item in positive["identities"])
        score_closure_identity_count += len(identities)
        game_id = str(positive["game_id"])
        mechanic = str(positive["mechanic_state_type"])
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["score_closure_packet_count"] += 1
        by_game[game_id]["score_closure_identity_count"] += len(identities)
        by_mechanic[mechanic]["score_closure_packet_count"] += 1
        by_mechanic[mechanic]["score_closure_identity_count"] += len(identities)

    for positive in partial_positives:
        identities = set(str(item) for item in positive["identities"])
        partial_identity_count += len(identities)
        game_id = str(positive["game_id"])
        mechanic = str(positive["mechanic_state_type"])
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["partial_scoreless_progress_packet_count"] += 1
        by_game[game_id]["partial_scoreless_progress_identity_count"] += len(identities)
        by_mechanic[mechanic]["partial_scoreless_progress_packet_count"] += 1
        by_mechanic[mechanic]["partial_scoreless_progress_identity_count"] += len(
            identities
        )
        if len(partial_rows) < sample_limit:
            partial_rows.append(
                {
                    "packet_index": int(positive["packet_index"]),
                    "game_id": game_id,
                    "mechanic_state_type": mechanic,
                    "effect_intent": str(positive["effect_intent"]),
                    "region_identity_count": len(identities),
                    "coverage_deficit_delta": _safe_int(
                        positive.get("coverage_deficit_delta")
                    ),
                    "filled_slot_delta": _safe_int(positive.get("filled_slot_delta")),
                    "after_deficit": _safe_int(positive.get("after_deficit")),
                    "after_filled_slot_count": _safe_int(
                        positive.get("after_filled_slot_count")
                    ),
                    "after_slot_count": _safe_int(positive.get("after_slot_count")),
                    "sample_region_identities": sorted(identities)[:8],
                }
            )

    effect_rows.sort(
        key=lambda row: (
            _safe_int(row["terminal_effect_intent_margin"]),
            -_safe_int(row["matched_control_packet_count"]),
            str(row["mechanic_key"]),
        )
    )
    terminal_margin = int(terminal_identity_count - terminal_control_pair_count)
    terminal_supported = int(
        terminal_identity_count > 0
        and terminal_control_pair_count > 0
        and terminal_margin > 0
    )
    score_closure_supported = int(score_closure_identity_count > 0)
    if score_closure_supported:
        promotion_state = "score_closure_effect_intent_present"
    elif terminal_supported:
        promotion_state = "terminal_effect_intent_control_supported"
    elif terminal_identity_count > 0 and terminal_control_pair_count > 0:
        promotion_state = "terminal_effect_intent_control_dominated"
    elif terminal_identity_count > 0:
        promotion_state = "terminal_effect_intent_no_controls"
    elif partial_positives:
        promotion_state = "only_partial_scoreless_progress"
    else:
        promotion_state = "no_effect_intent_signal"

    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_EFFECT_INTENT_CONTROL_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "posterior_packet_count": int(len(packet_rows)),
            "scope": "game_id|mechanic_state_type|effect_intent|connected_region_identity",
            "missing_slot_scope_count": int(missing_slot_scope_count),
            "score_closure_packet_count": int(len(score_closures)),
            "score_closure_identity_count": int(score_closure_identity_count),
            "terminal_scoreless_slot_closure_packet_count": int(
                len(terminal_positives)
            ),
            "terminal_effect_intent_identity_count": int(terminal_identity_count),
            "terminal_effect_intent_control_packet_count": int(
                len(unique_control_indices)
            ),
            "terminal_effect_intent_control_pair_count": int(
                terminal_control_pair_count
            ),
            "terminal_effect_intent_margin": int(terminal_margin),
            "terminal_effect_intent_positive_control_margin_count": int(
                terminal_supported
            ),
            "partial_scoreless_progress_packet_count": int(len(partial_positives)),
            "partial_scoreless_progress_identity_count": int(partial_identity_count),
            "partial_progress_blocked_count": int(len(partial_positives)),
            "effect_intent_counts": {
                key: int(value) for key, value in sorted(effect_intent_counts.items())
            },
            "planner_promotion_blocked": bool(
                score_closure_supported <= 0 and terminal_supported <= 0
            ),
            "promotion_state": promotion_state,
            "effect_intent_rows": effect_rows[:sample_limit],
            "partial_progress_rows": partial_rows,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "samples": samples,
            "next_runtime_contract": {
                "required_before_execution": (
                    "score closure or terminal slot closure intent with positive controls"
                ),
                "allowed_connection": "diagnostic_only_until_effect_intent_controls_supported",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_terminal_effect_score_closure_attribution(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    attribution_window: int = 8,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Require terminal slot closure to be tied to observed score closure."""

    packet_rows = [
        dict(packet)
        for packet in posterior_packets
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    terminal_rows: list[dict[str, Any]] = []
    score_closure_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    missing_slot_scope_count = 0

    for index, packet in enumerate(packet_rows):
        scope = packet.get("slot_scope")
        if not isinstance(scope, Mapping):
            missing_slot_scope_count += 1
            continue
        intent = _slot_pool_effect_intent(packet)
        effect_intent = str(intent.get("effect_intent", "") or "unknown")
        game_id = str(packet.get("game_id", "") or "unknown")
        mechanic = str(packet.get("mechanic_state_type", "") or "slot_pool_coverage")
        mechanic_key = _posterior_support_key(packet)
        resolved_identities = _slot_pool_region_identities(
            _slot_pool_scope_tokens(scope, resolved_first=True)
        )
        active_identities = _slot_pool_region_identities(
            _slot_pool_scope_tokens(scope, resolved_first=False)
        )
        identities = resolved_identities or active_identities
        row = {
            "packet_index": int(index),
            "game_id": game_id,
            "mechanic_state_type": mechanic,
            "mechanic_key": mechanic_key,
            "effect_intent": effect_intent,
            "identities": identities,
            "step_before": _safe_int(packet.get("step_before")),
            "step_after": _safe_int(packet.get("step_after")),
            "score_delta": _safe_int(packet.get("score_delta")),
        }
        if effect_intent == "terminal_scoreless_slot_closure" and identities:
            terminal_rows.append(row)
        elif effect_intent == "score_closure" and identities:
            score_closure_rows.append(row)
        elif effect_intent in {
            "ambiguous_no_progress_control",
            "falsified_no_progress_control",
        } and identities:
            control_rows.append(row)

    score_closures_by_mechanic: dict[str, list[dict[str, Any]]] = {}
    for closure in score_closure_rows:
        score_closures_by_mechanic.setdefault(
            str(closure["mechanic_key"]), []
        ).append(closure)
    controls_by_mechanic: dict[str, list[dict[str, Any]]] = {}
    for control in control_rows:
        controls_by_mechanic.setdefault(str(control["mechanic_key"]), []).append(
            control
        )

    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    attribution_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    attributed_terminal_count = 0
    attributed_identity_count = 0
    unattributed_terminal_count = 0
    direct_score_closure_identity_count = 0
    score_closure_packet_count = len(score_closure_rows)
    score_closure_identity_keys: set[str] = set()
    matched_control_pair_count = 0
    matched_control_packet_indices: set[int] = set()

    for closure in score_closure_rows:
        for identity in closure["identities"]:
            score_closure_identity_keys.add(str(identity))
            direct_score_closure_identity_count += 1

    for terminal in terminal_rows:
        identities = set(str(item) for item in terminal["identities"])
        matching_closures: list[dict[str, Any]] = []
        for closure in score_closures_by_mechanic.get(str(terminal["mechanic_key"]), []):
            closure_identities = set(str(item) for item in closure["identities"])
            overlap = identities & closure_identities
            if not overlap:
                continue
            if _safe_int(closure["packet_index"]) < _safe_int(terminal["packet_index"]):
                continue
            if (
                attribution_window >= 0
                and _safe_int(closure["packet_index"]) - _safe_int(
                    terminal["packet_index"]
                )
                > attribution_window
            ):
                continue
            match = dict(closure)
            match["matched_identities"] = sorted(overlap)
            matching_closures.append(match)

        control_matches = 0
        matched_identity_count = 0
        for control in controls_by_mechanic.get(str(terminal["mechanic_key"]), []):
            overlap = identities & set(str(item) for item in control["identities"])
            if not overlap:
                continue
            control_matches += 1
            matched_identity_count += len(overlap)
            matched_control_pair_count += 1
            matched_control_packet_indices.add(_safe_int(control["packet_index"]))
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "terminal_packet_index": int(terminal["packet_index"]),
                        "control_packet_index": int(control["packet_index"]),
                        "game_id": str(terminal["game_id"]),
                        "mechanic_state_type": str(terminal["mechanic_state_type"]),
                        "matched_region_identities": sorted(overlap)[:8],
                        "matched_region_identity_count": len(overlap),
                    }
                )

        if matching_closures:
            attributed_terminal_count += 1
            attributed_identity_count += len(identities)
        else:
            unattributed_terminal_count += 1

        game_id = str(terminal["game_id"])
        mechanic = str(terminal["mechanic_state_type"])
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["terminal_scoreless_slot_closure_packet_count"] += 1
        by_game[game_id]["score_closure_attributed_terminal_count"] += int(
            bool(matching_closures)
        )
        by_game[game_id]["unattributed_terminal_scoreless_closure_count"] += int(
            not bool(matching_closures)
        )
        by_game[game_id]["matched_control_pair_count"] += control_matches
        by_mechanic[mechanic]["terminal_scoreless_slot_closure_packet_count"] += 1
        by_mechanic[mechanic]["score_closure_attributed_terminal_count"] += int(
            bool(matching_closures)
        )
        by_mechanic[mechanic]["unattributed_terminal_scoreless_closure_count"] += int(
            not bool(matching_closures)
        )
        by_mechanic[mechanic]["matched_control_pair_count"] += control_matches
        attribution_rows.append(
            {
                "terminal_packet_index": int(terminal["packet_index"]),
                "game_id": game_id,
                "mechanic_state_type": mechanic,
                "mechanic_key": str(terminal["mechanic_key"]),
                "score_closure_attributed": bool(matching_closures),
                "matching_score_closure_packet_count": len(matching_closures),
                "matched_control_packet_count": int(control_matches),
                "matched_control_identity_count": int(matched_identity_count),
                "terminal_identity_count": len(identities),
                "terminal_score_closure_margin": int(
                    (len(identities) if matching_closures else 0) - control_matches
                ),
                "sample_region_identities": sorted(identities)[:8],
            }
        )

    attribution_rows.sort(
        key=lambda row: (
            _safe_int(row["terminal_score_closure_margin"]),
            -_safe_int(row["matched_control_packet_count"]),
            str(row["mechanic_key"]),
        )
    )
    terminal_score_closure_margin = int(
        attributed_identity_count - matched_control_pair_count
    )
    attribution_supported = int(
        attributed_terminal_count > 0 and terminal_score_closure_margin > 0
    )
    if attribution_supported:
        promotion_state = "terminal_score_closure_attribution_supported"
    elif attributed_terminal_count > 0 and matched_control_pair_count > 0:
        promotion_state = "terminal_score_closure_attribution_control_matched"
    elif terminal_rows and matched_control_pair_count > 0:
        promotion_state = "terminal_score_closure_unattributed_control_dominated"
    elif terminal_rows:
        promotion_state = "terminal_score_closure_unattributed"
    elif score_closure_rows:
        promotion_state = "direct_score_closure_present"
    else:
        promotion_state = "no_terminal_effect_score_closure_signal"

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_TERMINAL_EFFECT_SCORE_CLOSURE_ATTRIBUTION_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "posterior_packet_count": int(len(packet_rows)),
            "scope": (
                "game_id|mechanic_state_type|terminal_effect_intent|"
                "future_score_closure|connected_region_identity"
            ),
            "attribution_window": int(attribution_window),
            "missing_slot_scope_count": int(missing_slot_scope_count),
            "score_closure_packet_count": int(score_closure_packet_count),
            "score_closure_identity_count": int(direct_score_closure_identity_count),
            "score_closure_unique_identity_count": int(
                len(score_closure_identity_keys)
            ),
            "terminal_scoreless_slot_closure_packet_count": int(len(terminal_rows)),
            "score_closure_attributed_terminal_count": int(attributed_terminal_count),
            "score_closure_attributed_identity_count": int(attributed_identity_count),
            "unattributed_terminal_scoreless_closure_count": int(
                unattributed_terminal_count
            ),
            "terminal_score_closure_control_packet_count": int(
                len(matched_control_packet_indices)
            ),
            "terminal_score_closure_control_pair_count": int(
                matched_control_pair_count
            ),
            "terminal_score_closure_margin": int(terminal_score_closure_margin),
            "terminal_score_closure_positive_control_margin_count": int(
                attribution_supported
            ),
            "planner_promotion_blocked": bool(attribution_supported <= 0),
            "promotion_state": promotion_state,
            "attribution_rows": attribution_rows[:sample_limit],
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "samples": samples,
            "next_runtime_contract": {
                "required_before_execution": (
                    "terminal effect-intent must be attributed to observed score closure "
                    "with a positive control margin"
                ),
                "allowed_connection": "diagnostic_only_until_terminal_score_closure_margin_supported",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_score_event_scope_attribution(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    score_events: Iterable[Mapping[str, Any]] | None = None,
    attribution_window: int = 8,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Attribute global score events back to terminal slot-pool closures."""

    packet_rows = [
        dict(packet)
        for packet in posterior_packets
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    score_event_rows = [
        dict(event)
        for event in list(score_events or [])
        if isinstance(event, Mapping) and _safe_int(event.get("score_delta")) > 0
    ]
    events_by_game: dict[str, list[dict[str, Any]]] = {}
    for event in score_event_rows:
        events_by_game.setdefault(str(event.get("game_id", "") or "unknown"), []).append(
            event
        )

    terminal_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    missing_slot_scope_count = 0
    for index, packet in enumerate(packet_rows):
        scope = packet.get("slot_scope")
        if not isinstance(scope, Mapping):
            missing_slot_scope_count += 1
            continue
        intent = _slot_pool_effect_intent(packet)
        effect_intent = str(intent.get("effect_intent", "") or "unknown")
        identities = _slot_pool_region_identities(
            _slot_pool_scope_tokens(scope, resolved_first=True)
        ) or _slot_pool_region_identities(
            _slot_pool_scope_tokens(scope, resolved_first=False)
        )
        if not identities:
            continue
        row = {
            "packet_index": int(index),
            "game_id": str(packet.get("game_id", "") or "unknown"),
            "mechanic_state_type": str(
                packet.get("mechanic_state_type", "") or "slot_pool_coverage"
            ),
            "mechanic_key": _posterior_support_key(packet),
            "effect_intent": effect_intent,
            "identities": identities,
            "step_after": _safe_int(packet.get("step_after")),
            "score_delta": _safe_int(packet.get("score_delta")),
        }
        if effect_intent == "terminal_scoreless_slot_closure":
            terminal_rows.append(row)
        elif effect_intent in {
            "ambiguous_no_progress_control",
            "falsified_no_progress_control",
        }:
            control_rows.append(row)

    controls_by_mechanic: dict[str, list[dict[str, Any]]] = {}
    for control in control_rows:
        controls_by_mechanic.setdefault(str(control["mechanic_key"]), []).append(
            control
        )

    def future_events(row: Mapping[str, Any]) -> list[dict[str, Any]]:
        start_step = _safe_int(row.get("step_after"))
        matches: list[dict[str, Any]] = []
        for event in events_by_game.get(str(row.get("game_id", "") or "unknown"), []):
            event_step = _safe_int(event.get("step"))
            if event_step < start_step:
                continue
            if attribution_window >= 0 and event_step - start_step > attribution_window:
                continue
            matches.append(event)
        return matches

    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    attribution_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    score_event_attributed_terminal_count = 0
    score_event_attributed_identity_count = 0
    unattributed_terminal_count = 0
    terminal_score_event_delta_sum = 0
    same_region_control_pair_count = 0
    score_event_scope_control_pair_count = 0
    score_event_scope_control_delta_sum = 0

    for terminal in terminal_rows:
        identities = set(str(item) for item in terminal["identities"])
        terminal_events = future_events(terminal)
        terminal_delta = sum(_safe_int(event.get("score_delta")) for event in terminal_events)
        if terminal_delta > 0:
            score_event_attributed_terminal_count += 1
            score_event_attributed_identity_count += len(identities)
            terminal_score_event_delta_sum += int(terminal_delta)
        else:
            unattributed_terminal_count += 1

        same_region_controls = 0
        scored_control_pairs = 0
        scored_control_delta = 0
        for control in controls_by_mechanic.get(str(terminal["mechanic_key"]), []):
            overlap = identities & set(str(item) for item in control["identities"])
            if not overlap:
                continue
            same_region_controls += 1
            same_region_control_pair_count += 1
            control_events = future_events(control)
            control_delta = sum(
                _safe_int(event.get("score_delta")) for event in control_events
            )
            if control_delta <= 0:
                continue
            scored_control_pairs += 1
            scored_control_delta += int(control_delta)
            score_event_scope_control_pair_count += 1
            score_event_scope_control_delta_sum += int(control_delta)
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "terminal_packet_index": int(terminal["packet_index"]),
                        "control_packet_index": int(control["packet_index"]),
                        "game_id": str(terminal["game_id"]),
                        "mechanic_state_type": str(terminal["mechanic_state_type"]),
                        "matched_region_identities": sorted(overlap)[:8],
                        "control_score_event_delta": int(control_delta),
                    }
                )

        game_id = str(terminal["game_id"])
        mechanic = str(terminal["mechanic_state_type"])
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        for counter in (by_game[game_id], by_mechanic[mechanic]):
            counter["terminal_scoreless_slot_closure_packet_count"] += 1
            counter["score_event_attributed_terminal_count"] += int(terminal_delta > 0)
            counter["unattributed_terminal_scoreless_closure_count"] += int(
                terminal_delta <= 0
            )
            counter["terminal_score_event_delta_sum"] += int(terminal_delta)
            counter["same_region_control_pair_count"] += int(same_region_controls)
            counter["score_event_scope_control_pair_count"] += int(scored_control_pairs)
            counter["score_event_scope_control_delta_sum"] += int(scored_control_delta)

        attribution_rows.append(
            {
                "terminal_packet_index": int(terminal["packet_index"]),
                "game_id": game_id,
                "mechanic_state_type": mechanic,
                "mechanic_key": str(terminal["mechanic_key"]),
                "score_event_attributed": bool(terminal_delta > 0),
                "terminal_score_event_delta": int(terminal_delta),
                "matching_score_event_count": len(terminal_events),
                "same_region_control_pair_count": int(same_region_controls),
                "score_event_scope_control_pair_count": int(scored_control_pairs),
                "score_event_scope_control_delta": int(scored_control_delta),
                "score_event_scope_margin": int(terminal_delta - scored_control_delta),
                "terminal_identity_count": len(identities),
                "sample_region_identities": sorted(identities)[:8],
            }
        )

    attribution_rows.sort(
        key=lambda row: (
            _safe_int(row["score_event_scope_margin"]),
            -_safe_int(row["score_event_scope_control_pair_count"]),
            str(row["mechanic_key"]),
        )
    )
    score_event_scope_margin = int(
        terminal_score_event_delta_sum - score_event_scope_control_delta_sum
    )
    attribution_supported = int(
        score_event_attributed_terminal_count > 0 and score_event_scope_margin > 0
    )
    if attribution_supported:
        promotion_state = "score_event_scope_attribution_supported"
    elif score_event_attributed_terminal_count > 0 and score_event_scope_control_pair_count > 0:
        promotion_state = "score_event_scope_attribution_control_matched"
    elif score_event_attributed_terminal_count > 0:
        promotion_state = "score_event_scope_attribution_margin_blocked"
    elif terminal_rows and same_region_control_pair_count > 0:
        promotion_state = "score_event_scope_unattributed_control_dominated"
    elif terminal_rows:
        promotion_state = "score_event_scope_unattributed"
    elif score_event_rows:
        promotion_state = "score_events_present_no_terminal_slot_closure"
    else:
        promotion_state = "no_score_event_scope_signal"

    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_SCORE_EVENT_SCOPE_ATTRIBUTION_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "posterior_packet_count": int(len(packet_rows)),
            "score_event_count": int(len(score_event_rows)),
            "attribution_window": int(attribution_window),
            "missing_slot_scope_count": int(missing_slot_scope_count),
            "terminal_scoreless_slot_closure_packet_count": int(len(terminal_rows)),
            "score_event_attributed_terminal_count": int(
                score_event_attributed_terminal_count
            ),
            "score_event_attributed_identity_count": int(
                score_event_attributed_identity_count
            ),
            "unattributed_terminal_scoreless_closure_count": int(
                unattributed_terminal_count
            ),
            "terminal_score_event_delta_sum": int(terminal_score_event_delta_sum),
            "same_region_control_pair_count": int(same_region_control_pair_count),
            "score_event_scope_control_pair_count": int(
                score_event_scope_control_pair_count
            ),
            "score_event_scope_control_delta_sum": int(
                score_event_scope_control_delta_sum
            ),
            "score_event_scope_margin": int(score_event_scope_margin),
            "score_event_scope_positive_control_margin_count": int(
                attribution_supported
            ),
            "planner_promotion_blocked": bool(attribution_supported <= 0),
            "promotion_state": promotion_state,
            "attribution_rows": attribution_rows[:sample_limit],
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "samples": samples,
            "next_runtime_contract": {
                "required_before_execution": (
                    "score events must be attributed to slot scope with positive controls"
                ),
                "allowed_connection": "diagnostic_only_until_score_event_scope_margin_supported",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_intervention_contrast_attribution(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    score_events: Iterable[Mapping[str, Any]] | None = None,
    attribution_window: int = 8,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Separate true intervention controls from broad same-region controls."""

    packet_rows = [
        dict(packet)
        for packet in posterior_packets
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    score_event_rows = [
        dict(event)
        for event in list(score_events or [])
        if isinstance(event, Mapping) and _safe_int(event.get("score_delta")) > 0
    ]
    events_by_game: dict[str, list[dict[str, Any]]] = {}
    for event in score_event_rows:
        events_by_game.setdefault(str(event.get("game_id", "") or "unknown"), []).append(
            event
        )

    terminal_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    missing_slot_scope_count = 0
    for index, packet in enumerate(packet_rows):
        scope = packet.get("slot_scope")
        if not isinstance(scope, Mapping):
            missing_slot_scope_count += 1
            continue
        intent = _slot_pool_effect_intent(packet)
        effect_intent = str(intent.get("effect_intent", "") or "unknown")
        identities = _slot_pool_region_identities(
            _slot_pool_scope_tokens(scope, resolved_first=True)
        ) or _slot_pool_region_identities(
            _slot_pool_scope_tokens(scope, resolved_first=False)
        )
        if not identities:
            continue
        row = {
            "packet_index": int(index),
            "game_id": str(packet.get("game_id", "") or "unknown"),
            "mechanic_state_type": str(
                packet.get("mechanic_state_type", "") or "slot_pool_coverage"
            ),
            "mechanic_key": _posterior_support_key(packet),
            "effect_intent": effect_intent,
            "identities": identities,
            "step_after": _safe_int(packet.get("step_after")),
            "score_delta": _safe_int(packet.get("score_delta")),
            "action_name": str(packet.get("action_name", "") or ""),
            "action_site_signature": _slot_pool_action_site_signature(packet),
            "before_active_slot_signature": str(
                scope.get("before_active_slot_signature", "") or ""
            ),
            "resolved_slot_signature": str(
                scope.get("resolved_slot_signature", "") or ""
            ),
        }
        row["intervention_signature"] = _slot_pool_intervention_signature(row)
        if effect_intent == "terminal_scoreless_slot_closure":
            terminal_rows.append(row)
        elif effect_intent in {
            "ambiguous_no_progress_control",
            "falsified_no_progress_control",
        }:
            control_rows.append(row)

    controls_by_mechanic: dict[str, list[dict[str, Any]]] = {}
    for control in control_rows:
        controls_by_mechanic.setdefault(str(control["mechanic_key"]), []).append(
            control
        )

    def future_events(row: Mapping[str, Any]) -> list[dict[str, Any]]:
        start_step = _safe_int(row.get("step_after"))
        matches: list[dict[str, Any]] = []
        for event in events_by_game.get(str(row.get("game_id", "") or "unknown"), []):
            event_step = _safe_int(event.get("step"))
            if event_step < start_step:
                continue
            if attribution_window >= 0 and event_step - start_step > attribution_window:
                continue
            match = dict(event)
            match["event_delay"] = int(event_step - start_step)
            matches.append(match)
        return matches

    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    attribution_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    terminal_score_event_delta_sum = 0
    score_event_attributed_terminal_count = 0
    terminal_unique_intervention_count = 0
    same_region_control_pair_count = 0
    scored_same_region_control_pair_count = 0
    scored_same_region_control_delta_sum = 0
    intervention_matched_control_pair_count = 0
    intervention_matched_control_delta_sum = 0
    scope_only_scored_control_pair_count = 0
    scope_only_scored_control_delta_sum = 0

    for terminal in terminal_rows:
        identities = set(str(item) for item in terminal["identities"])
        terminal_events = future_events(terminal)
        terminal_delta = sum(_safe_int(event.get("score_delta")) for event in terminal_events)
        if terminal_delta > 0:
            score_event_attributed_terminal_count += 1
            terminal_score_event_delta_sum += int(terminal_delta)

        same_region_controls = 0
        scored_controls = 0
        scored_control_delta = 0
        matched_controls = 0
        matched_control_delta = 0
        scope_only_controls = 0
        scope_only_control_delta = 0
        for control in controls_by_mechanic.get(str(terminal["mechanic_key"]), []):
            overlap = identities & set(str(item) for item in control["identities"])
            if not overlap:
                continue
            same_region_controls += 1
            same_region_control_pair_count += 1
            control_events = future_events(control)
            control_delta = sum(
                _safe_int(event.get("score_delta")) for event in control_events
            )
            if control_delta <= 0:
                continue
            scored_controls += 1
            scored_control_delta += int(control_delta)
            scored_same_region_control_pair_count += 1
            scored_same_region_control_delta_sum += int(control_delta)
            signature_matched = (
                str(control.get("intervention_signature", ""))
                == str(terminal.get("intervention_signature", ""))
            )
            if signature_matched:
                matched_controls += 1
                matched_control_delta += int(control_delta)
                intervention_matched_control_pair_count += 1
                intervention_matched_control_delta_sum += int(control_delta)
            else:
                scope_only_controls += 1
                scope_only_control_delta += int(control_delta)
                scope_only_scored_control_pair_count += 1
                scope_only_scored_control_delta_sum += int(control_delta)
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "terminal_packet_index": int(terminal["packet_index"]),
                        "control_packet_index": int(control["packet_index"]),
                        "game_id": str(terminal["game_id"]),
                        "mechanic_state_type": str(terminal["mechanic_state_type"]),
                        "matched_region_identities": sorted(overlap)[:8],
                        "control_score_event_delta": int(control_delta),
                        "intervention_signature_matched": bool(signature_matched),
                        "terminal_intervention_signature": str(
                            terminal.get("intervention_signature", "")
                        ),
                        "control_intervention_signature": str(
                            control.get("intervention_signature", "")
                        ),
                    }
                )

        intervention_margin = int(terminal_delta - matched_control_delta)
        unique_intervention = bool(terminal_delta > 0 and matched_control_delta == 0)
        terminal_unique_intervention_count += int(unique_intervention)

        game_id = str(terminal["game_id"])
        mechanic = str(terminal["mechanic_state_type"])
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        for counter in (by_game[game_id], by_mechanic[mechanic]):
            counter["terminal_scoreless_slot_closure_packet_count"] += 1
            counter["score_event_attributed_terminal_count"] += int(terminal_delta > 0)
            counter["terminal_unique_intervention_count"] += int(unique_intervention)
            counter["terminal_score_event_delta_sum"] += int(terminal_delta)
            counter["same_region_control_pair_count"] += int(same_region_controls)
            counter["scored_same_region_control_pair_count"] += int(scored_controls)
            counter["scored_same_region_control_delta_sum"] += int(scored_control_delta)
            counter["intervention_matched_control_pair_count"] += int(matched_controls)
            counter["intervention_matched_control_delta_sum"] += int(
                matched_control_delta
            )
            counter["scope_only_scored_control_pair_count"] += int(scope_only_controls)
            counter["scope_only_scored_control_delta_sum"] += int(
                scope_only_control_delta
            )

        attribution_rows.append(
            {
                "terminal_packet_index": int(terminal["packet_index"]),
                "game_id": game_id,
                "mechanic_state_type": mechanic,
                "mechanic_key": str(terminal["mechanic_key"]),
                "score_event_attributed": bool(terminal_delta > 0),
                "terminal_score_event_delta": int(terminal_delta),
                "matching_score_event_count": len(terminal_events),
                "same_region_control_pair_count": int(same_region_controls),
                "scored_same_region_control_pair_count": int(scored_controls),
                "scored_same_region_control_delta": int(scored_control_delta),
                "intervention_matched_control_pair_count": int(matched_controls),
                "intervention_matched_control_delta": int(matched_control_delta),
                "scope_only_scored_control_pair_count": int(scope_only_controls),
                "scope_only_scored_control_delta": int(scope_only_control_delta),
                "intervention_contrast_margin": int(intervention_margin),
                "terminal_unique_intervention": bool(unique_intervention),
                "intervention_signature": str(terminal["intervention_signature"]),
                "action_name": str(terminal["action_name"]),
                "action_site_signature": str(terminal["action_site_signature"]),
                "before_active_slot_signature": str(
                    terminal["before_active_slot_signature"]
                ),
                "resolved_slot_signature": str(terminal["resolved_slot_signature"]),
                "terminal_identity_count": len(identities),
                "sample_region_identities": sorted(identities)[:8],
            }
        )

    attribution_rows.sort(
        key=lambda row: (
            _safe_int(row["intervention_contrast_margin"]),
            -_safe_int(row["intervention_matched_control_pair_count"]),
            -_safe_int(row["scope_only_scored_control_pair_count"]),
            str(row["mechanic_key"]),
        )
    )
    intervention_contrast_margin = int(
        terminal_score_event_delta_sum - intervention_matched_control_delta_sum
    )
    contrast_supported = int(
        score_event_attributed_terminal_count > 0
        and intervention_contrast_margin > 0
        and intervention_matched_control_delta_sum == 0
    )
    if contrast_supported:
        promotion_state = "intervention_contrast_attribution_supported"
    elif (
        score_event_attributed_terminal_count > 0
        and intervention_matched_control_pair_count > 0
    ):
        promotion_state = "intervention_contrast_control_matched"
    elif score_event_attributed_terminal_count > 0:
        promotion_state = "intervention_contrast_margin_blocked"
    elif terminal_rows and same_region_control_pair_count > 0:
        promotion_state = "intervention_contrast_unattributed_control_dominated"
    elif terminal_rows:
        promotion_state = "intervention_contrast_unattributed"
    elif score_event_rows:
        promotion_state = "score_events_present_no_terminal_intervention"
    else:
        promotion_state = "no_intervention_contrast_signal"

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_INTERVENTION_CONTRAST_ATTRIBUTION_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "posterior_packet_count": int(len(packet_rows)),
            "score_event_count": int(len(score_event_rows)),
            "attribution_window": int(attribution_window),
            "missing_slot_scope_count": int(missing_slot_scope_count),
            "terminal_scoreless_slot_closure_packet_count": int(len(terminal_rows)),
            "score_event_attributed_terminal_count": int(
                score_event_attributed_terminal_count
            ),
            "terminal_score_event_delta_sum": int(terminal_score_event_delta_sum),
            "terminal_unique_intervention_count": int(
                terminal_unique_intervention_count
            ),
            "same_region_control_pair_count": int(same_region_control_pair_count),
            "scored_same_region_control_pair_count": int(
                scored_same_region_control_pair_count
            ),
            "scored_same_region_control_delta_sum": int(
                scored_same_region_control_delta_sum
            ),
            "intervention_matched_control_pair_count": int(
                intervention_matched_control_pair_count
            ),
            "intervention_matched_control_delta_sum": int(
                intervention_matched_control_delta_sum
            ),
            "scope_only_scored_control_pair_count": int(
                scope_only_scored_control_pair_count
            ),
            "scope_only_scored_control_delta_sum": int(
                scope_only_scored_control_delta_sum
            ),
            "intervention_contrast_margin": int(intervention_contrast_margin),
            "intervention_contrast_positive_margin_count": int(contrast_supported),
            "planner_promotion_blocked": bool(contrast_supported <= 0),
            "promotion_state": promotion_state,
            "attribution_rows": attribution_rows[:sample_limit],
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "samples": samples,
            "next_runtime_contract": {
                "required_before_execution": (
                    "score-event scope attribution must survive intervention-signature "
                    "controls, not only broad same-region controls"
                ),
                "allowed_connection": (
                    "diagnostic_only_until_intervention_contrast_margin_supported"
                ),
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_intervention_temporal_counterfactual(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    score_events: Iterable[Mapping[str, Any]] | None = None,
    attribution_window: int = 8,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Check whether same-intervention controls only score after the terminal event."""

    packet_rows = [
        dict(packet)
        for packet in posterior_packets
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    score_event_rows = [
        dict(event)
        for event in list(score_events or [])
        if isinstance(event, Mapping) and _safe_int(event.get("score_delta")) > 0
    ]
    events_by_game: dict[str, list[dict[str, Any]]] = {}
    for event in score_event_rows:
        events_by_game.setdefault(str(event.get("game_id", "") or "unknown"), []).append(
            event
        )
    for events in events_by_game.values():
        events.sort(key=lambda row: (_safe_int(row.get("step")), _safe_int(row.get("trace_index"))))

    terminal_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    missing_slot_scope_count = 0
    for index, packet in enumerate(packet_rows):
        scope = packet.get("slot_scope")
        if not isinstance(scope, Mapping):
            missing_slot_scope_count += 1
            continue
        intent = _slot_pool_effect_intent(packet)
        effect_intent = str(intent.get("effect_intent", "") or "unknown")
        identities = _slot_pool_region_identities(
            _slot_pool_scope_tokens(scope, resolved_first=True)
        ) or _slot_pool_region_identities(
            _slot_pool_scope_tokens(scope, resolved_first=False)
        )
        if not identities:
            continue
        row = {
            "packet_index": int(index),
            "game_id": str(packet.get("game_id", "") or "unknown"),
            "mechanic_state_type": str(
                packet.get("mechanic_state_type", "") or "slot_pool_coverage"
            ),
            "mechanic_key": _posterior_support_key(packet),
            "effect_intent": effect_intent,
            "identities": identities,
            "step_after": _safe_int(packet.get("step_after")),
            "score_delta": _safe_int(packet.get("score_delta")),
            "action_name": str(packet.get("action_name", "") or ""),
            "action_site_signature": _slot_pool_action_site_signature(packet),
            "before_active_slot_signature": str(
                scope.get("before_active_slot_signature", "") or ""
            ),
            "resolved_slot_signature": str(
                scope.get("resolved_slot_signature", "") or ""
            ),
        }
        row["intervention_signature"] = _slot_pool_intervention_signature(row)
        if effect_intent == "terminal_scoreless_slot_closure":
            terminal_rows.append(row)
        elif effect_intent in {
            "ambiguous_no_progress_control",
            "falsified_no_progress_control",
        }:
            control_rows.append(row)

    controls_by_mechanic: dict[str, list[dict[str, Any]]] = {}
    for control in control_rows:
        controls_by_mechanic.setdefault(str(control["mechanic_key"]), []).append(
            control
        )

    def first_future_event(row: Mapping[str, Any]) -> dict[str, Any] | None:
        start_step = _safe_int(row.get("step_after"))
        for event in events_by_game.get(str(row.get("game_id", "") or "unknown"), []):
            event_step = _safe_int(event.get("step"))
            if event_step < start_step:
                continue
            if attribution_window >= 0 and event_step - start_step > attribution_window:
                continue
            match = dict(event)
            match["event_delay"] = int(event_step - start_step)
            return match
        return None

    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    attribution_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    terminal_score_event_delta_sum = 0
    score_event_attributed_terminal_count = 0
    matched_control_pair_count = 0
    matched_scored_control_pair_count = 0
    same_or_earlier_event_control_pair_count = 0
    same_or_earlier_event_control_delta_sum = 0
    later_event_control_pair_count = 0
    later_event_control_delta_sum = 0
    terminal_temporal_unique_count = 0

    for terminal in terminal_rows:
        identities = set(str(item) for item in terminal["identities"])
        terminal_event = first_future_event(terminal)
        terminal_event_step = (
            _safe_int(terminal_event.get("step")) if terminal_event is not None else -1
        )
        terminal_event_delta = (
            _safe_int(terminal_event.get("score_delta"))
            if terminal_event is not None
            else 0
        )
        terminal_event_delay = (
            _safe_int(terminal_event.get("event_delay"))
            if terminal_event is not None
            else -1
        )
        if terminal_event_delta > 0:
            score_event_attributed_terminal_count += 1
            terminal_score_event_delta_sum += int(terminal_event_delta)

        matched_controls = 0
        matched_scored_controls = 0
        early_or_same_controls = 0
        early_or_same_delta = 0
        later_controls = 0
        later_delta = 0
        for control in controls_by_mechanic.get(str(terminal["mechanic_key"]), []):
            overlap = identities & set(str(item) for item in control["identities"])
            if not overlap:
                continue
            if str(control.get("intervention_signature", "")) != str(
                terminal.get("intervention_signature", "")
            ):
                continue
            matched_controls += 1
            matched_control_pair_count += 1
            control_event = first_future_event(control)
            control_event_delta = (
                _safe_int(control_event.get("score_delta"))
                if control_event is not None
                else 0
            )
            if control_event_delta <= 0:
                continue
            matched_scored_controls += 1
            matched_scored_control_pair_count += 1
            control_event_step = _safe_int(control_event.get("step"))
            control_event_delay = _safe_int(control_event.get("event_delay"))
            is_same_or_earlier = (
                terminal_event_step < 0 or control_event_step <= terminal_event_step
            )
            if is_same_or_earlier:
                early_or_same_controls += 1
                early_or_same_delta += int(control_event_delta)
                same_or_earlier_event_control_pair_count += 1
                same_or_earlier_event_control_delta_sum += int(control_event_delta)
            else:
                later_controls += 1
                later_delta += int(control_event_delta)
                later_event_control_pair_count += 1
                later_event_control_delta_sum += int(control_event_delta)
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "terminal_packet_index": int(terminal["packet_index"]),
                        "control_packet_index": int(control["packet_index"]),
                        "game_id": str(terminal["game_id"]),
                        "mechanic_state_type": str(terminal["mechanic_state_type"]),
                        "matched_region_identities": sorted(overlap)[:8],
                        "terminal_event_step": int(terminal_event_step),
                        "terminal_event_delay": int(terminal_event_delay),
                        "control_event_step": int(control_event_step),
                        "control_event_delay": int(control_event_delay),
                        "control_score_event_delta": int(control_event_delta),
                        "same_or_earlier_event_control": bool(is_same_or_earlier),
                        "intervention_signature": str(
                            terminal.get("intervention_signature", "")
                        ),
                    }
                )

        temporal_margin = int(terminal_event_delta - early_or_same_delta)
        temporal_unique = bool(terminal_event_delta > 0 and early_or_same_delta == 0)
        terminal_temporal_unique_count += int(temporal_unique)

        game_id = str(terminal["game_id"])
        mechanic = str(terminal["mechanic_state_type"])
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        for counter in (by_game[game_id], by_mechanic[mechanic]):
            counter["terminal_scoreless_slot_closure_packet_count"] += 1
            counter["score_event_attributed_terminal_count"] += int(
                terminal_event_delta > 0
            )
            counter["terminal_score_event_delta_sum"] += int(terminal_event_delta)
            counter["matched_control_pair_count"] += int(matched_controls)
            counter["matched_scored_control_pair_count"] += int(matched_scored_controls)
            counter["same_or_earlier_event_control_pair_count"] += int(
                early_or_same_controls
            )
            counter["same_or_earlier_event_control_delta_sum"] += int(
                early_or_same_delta
            )
            counter["later_event_control_pair_count"] += int(later_controls)
            counter["later_event_control_delta_sum"] += int(later_delta)
            counter["terminal_temporal_unique_count"] += int(temporal_unique)

        attribution_rows.append(
            {
                "terminal_packet_index": int(terminal["packet_index"]),
                "game_id": game_id,
                "mechanic_state_type": mechanic,
                "mechanic_key": str(terminal["mechanic_key"]),
                "score_event_attributed": bool(terminal_event_delta > 0),
                "terminal_score_event_delta": int(terminal_event_delta),
                "terminal_event_step": int(terminal_event_step),
                "terminal_event_delay": int(terminal_event_delay),
                "matched_control_pair_count": int(matched_controls),
                "matched_scored_control_pair_count": int(matched_scored_controls),
                "same_or_earlier_event_control_pair_count": int(
                    early_or_same_controls
                ),
                "same_or_earlier_event_control_delta": int(early_or_same_delta),
                "later_event_control_pair_count": int(later_controls),
                "later_event_control_delta": int(later_delta),
                "intervention_temporal_margin": int(temporal_margin),
                "terminal_temporal_unique": bool(temporal_unique),
                "intervention_signature": str(terminal["intervention_signature"]),
                "action_name": str(terminal["action_name"]),
                "action_site_signature": str(terminal["action_site_signature"]),
                "before_active_slot_signature": str(
                    terminal["before_active_slot_signature"]
                ),
                "resolved_slot_signature": str(terminal["resolved_slot_signature"]),
                "terminal_identity_count": len(identities),
                "sample_region_identities": sorted(identities)[:8],
            }
        )

    attribution_rows.sort(
        key=lambda row: (
            _safe_int(row["intervention_temporal_margin"]),
            -_safe_int(row["same_or_earlier_event_control_pair_count"]),
            -_safe_int(row["later_event_control_pair_count"]),
            str(row["mechanic_key"]),
        )
    )
    intervention_temporal_margin = int(
        terminal_score_event_delta_sum - same_or_earlier_event_control_delta_sum
    )
    temporal_supported = int(
        score_event_attributed_terminal_count > 0
        and intervention_temporal_margin > 0
        and same_or_earlier_event_control_delta_sum == 0
    )
    if temporal_supported:
        promotion_state = "intervention_temporal_counterfactual_supported"
    elif (
        score_event_attributed_terminal_count > 0
        and same_or_earlier_event_control_pair_count > 0
    ):
        promotion_state = "intervention_temporal_counterfactual_control_matched"
    elif score_event_attributed_terminal_count > 0:
        promotion_state = "intervention_temporal_counterfactual_margin_blocked"
    elif terminal_rows and matched_control_pair_count > 0:
        promotion_state = "intervention_temporal_counterfactual_unattributed_control_dominated"
    elif terminal_rows:
        promotion_state = "intervention_temporal_counterfactual_unattributed"
    elif score_event_rows:
        promotion_state = "score_events_present_no_terminal_temporal_intervention"
    else:
        promotion_state = "no_intervention_temporal_counterfactual_signal"

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_INTERVENTION_TEMPORAL_COUNTERFACTUAL_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "posterior_packet_count": int(len(packet_rows)),
            "score_event_count": int(len(score_event_rows)),
            "attribution_window": int(attribution_window),
            "missing_slot_scope_count": int(missing_slot_scope_count),
            "terminal_scoreless_slot_closure_packet_count": int(len(terminal_rows)),
            "score_event_attributed_terminal_count": int(
                score_event_attributed_terminal_count
            ),
            "terminal_score_event_delta_sum": int(terminal_score_event_delta_sum),
            "matched_control_pair_count": int(matched_control_pair_count),
            "matched_scored_control_pair_count": int(
                matched_scored_control_pair_count
            ),
            "same_or_earlier_event_control_pair_count": int(
                same_or_earlier_event_control_pair_count
            ),
            "same_or_earlier_event_control_delta_sum": int(
                same_or_earlier_event_control_delta_sum
            ),
            "later_event_control_pair_count": int(later_event_control_pair_count),
            "later_event_control_delta_sum": int(later_event_control_delta_sum),
            "intervention_temporal_margin": int(intervention_temporal_margin),
            "intervention_temporal_positive_margin_count": int(temporal_supported),
            "terminal_temporal_unique_count": int(terminal_temporal_unique_count),
            "planner_promotion_blocked": bool(temporal_supported <= 0),
            "promotion_state": promotion_state,
            "attribution_rows": attribution_rows[:sample_limit],
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "samples": samples,
            "next_runtime_contract": {
                "required_before_execution": (
                    "same-intervention controls must not explain the same or earlier "
                    "score event as the terminal slot closure"
                ),
                "allowed_connection": (
                    "diagnostic_only_until_intervention_temporal_margin_supported"
                ),
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_already_scored_event_exclusion(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    score_events: Iterable[Mapping[str, Any]] | None = None,
    attribution_window: int = 8,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Separate already-scored controls from same-window terminal blockers."""

    packet_rows = [
        dict(packet)
        for packet in posterior_packets
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    score_event_rows = [
        dict(event)
        for event in list(score_events or [])
        if isinstance(event, Mapping) and _safe_int(event.get("score_delta")) > 0
    ]
    events_by_game: dict[str, list[dict[str, Any]]] = {}
    for event in score_event_rows:
        events_by_game.setdefault(str(event.get("game_id", "") or "unknown"), []).append(
            event
        )
    for events in events_by_game.values():
        events.sort(
            key=lambda row: (_safe_int(row.get("step")), _safe_int(row.get("trace_index")))
        )

    terminal_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    missing_slot_scope_count = 0
    for index, packet in enumerate(packet_rows):
        scope = packet.get("slot_scope")
        if not isinstance(scope, Mapping):
            missing_slot_scope_count += 1
            continue
        intent = _slot_pool_effect_intent(packet)
        effect_intent = str(intent.get("effect_intent", "") or "unknown")
        identities = _slot_pool_region_identities(
            _slot_pool_scope_tokens(scope, resolved_first=True)
        ) or _slot_pool_region_identities(
            _slot_pool_scope_tokens(scope, resolved_first=False)
        )
        if not identities:
            continue
        row = {
            "packet_index": int(index),
            "game_id": str(packet.get("game_id", "") or "unknown"),
            "mechanic_state_type": str(
                packet.get("mechanic_state_type", "") or "slot_pool_coverage"
            ),
            "mechanic_key": _posterior_support_key(packet),
            "effect_intent": effect_intent,
            "identities": identities,
            "step_after": _safe_int(packet.get("step_after")),
            "score_delta": _safe_int(packet.get("score_delta")),
            "action_name": str(packet.get("action_name", "") or ""),
            "action_site_signature": _slot_pool_action_site_signature(packet),
            "before_active_slot_signature": str(
                scope.get("before_active_slot_signature", "") or ""
            ),
            "resolved_slot_signature": str(
                scope.get("resolved_slot_signature", "") or ""
            ),
        }
        row["intervention_signature"] = _slot_pool_intervention_signature(row)
        if effect_intent == "terminal_scoreless_slot_closure":
            terminal_rows.append(row)
        elif effect_intent in {
            "ambiguous_no_progress_control",
            "falsified_no_progress_control",
        }:
            control_rows.append(row)

    controls_by_mechanic: dict[str, list[dict[str, Any]]] = {}
    for control in control_rows:
        controls_by_mechanic.setdefault(str(control["mechanic_key"]), []).append(
            control
        )

    def first_future_event(row: Mapping[str, Any]) -> dict[str, Any] | None:
        start_step = _safe_int(row.get("step_after"))
        for event in events_by_game.get(str(row.get("game_id", "") or "unknown"), []):
            event_step = _safe_int(event.get("step"))
            if event_step < start_step:
                continue
            if attribution_window >= 0 and event_step - start_step > attribution_window:
                continue
            match = dict(event)
            match["event_delay"] = int(event_step - start_step)
            return match
        return None

    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    attribution_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    terminal_score_event_delta_sum = 0
    score_event_attributed_terminal_count = 0
    matched_control_pair_count = 0
    matched_scored_control_pair_count = 0
    pre_terminal_already_scored_control_pair_count = 0
    pre_terminal_already_scored_control_delta_sum = 0
    same_window_control_pair_count = 0
    same_window_control_delta_sum = 0
    post_terminal_control_pair_count = 0
    post_terminal_control_delta_sum = 0
    terminal_negative_memory_unique_count = 0
    already_scored_exclusion_applied_count = 0

    for terminal in terminal_rows:
        identities = set(str(item) for item in terminal["identities"])
        terminal_step_after = _safe_int(terminal.get("step_after"))
        terminal_event = first_future_event(terminal)
        terminal_event_step = (
            _safe_int(terminal_event.get("step")) if terminal_event is not None else -1
        )
        terminal_event_delta = (
            _safe_int(terminal_event.get("score_delta"))
            if terminal_event is not None
            else 0
        )
        terminal_event_delay = (
            _safe_int(terminal_event.get("event_delay"))
            if terminal_event is not None
            else -1
        )
        if terminal_event_delta > 0:
            score_event_attributed_terminal_count += 1
            terminal_score_event_delta_sum += int(terminal_event_delta)

        matched_controls = 0
        matched_scored_controls = 0
        pre_terminal_controls = 0
        pre_terminal_delta = 0
        same_window_controls = 0
        same_window_delta = 0
        post_terminal_controls = 0
        post_terminal_delta = 0
        for control in controls_by_mechanic.get(str(terminal["mechanic_key"]), []):
            overlap = identities & set(str(item) for item in control["identities"])
            if not overlap:
                continue
            if str(control.get("intervention_signature", "")) != str(
                terminal.get("intervention_signature", "")
            ):
                continue
            matched_controls += 1
            matched_control_pair_count += 1
            control_event = first_future_event(control)
            control_event_delta = (
                _safe_int(control_event.get("score_delta"))
                if control_event is not None
                else 0
            )
            if control_event_delta <= 0:
                continue
            matched_scored_controls += 1
            matched_scored_control_pair_count += 1
            control_event_step = _safe_int(control_event.get("step"))
            control_event_delay = _safe_int(control_event.get("event_delay"))
            if control_event_step < terminal_step_after:
                relation = "pre_terminal_already_scored"
                pre_terminal_controls += 1
                pre_terminal_delta += int(control_event_delta)
                pre_terminal_already_scored_control_pair_count += 1
                pre_terminal_already_scored_control_delta_sum += int(
                    control_event_delta
                )
            elif terminal_event_step < 0 or control_event_step <= terminal_event_step:
                relation = "same_window_control"
                same_window_controls += 1
                same_window_delta += int(control_event_delta)
                same_window_control_pair_count += 1
                same_window_control_delta_sum += int(control_event_delta)
            else:
                relation = "post_terminal_control"
                post_terminal_controls += 1
                post_terminal_delta += int(control_event_delta)
                post_terminal_control_pair_count += 1
                post_terminal_control_delta_sum += int(control_event_delta)
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "terminal_packet_index": int(terminal["packet_index"]),
                        "control_packet_index": int(control["packet_index"]),
                        "game_id": str(terminal["game_id"]),
                        "mechanic_state_type": str(terminal["mechanic_state_type"]),
                        "matched_region_identities": sorted(overlap)[:8],
                        "terminal_step_after": int(terminal_step_after),
                        "terminal_event_step": int(terminal_event_step),
                        "terminal_event_delay": int(terminal_event_delay),
                        "control_event_step": int(control_event_step),
                        "control_event_delay": int(control_event_delay),
                        "control_score_event_delta": int(control_event_delta),
                        "control_temporal_relation": relation,
                        "excluded_as_already_scored": bool(
                            relation == "pre_terminal_already_scored"
                        ),
                        "intervention_signature": str(
                            terminal.get("intervention_signature", "")
                        ),
                    }
                )

        exclusion_margin = int(terminal_event_delta - same_window_delta)
        terminal_negative_memory_unique = bool(
            terminal_event_delta > 0 and same_window_delta == 0
        )
        terminal_negative_memory_unique_count += int(terminal_negative_memory_unique)
        already_scored_exclusion_applied_count += int(pre_terminal_delta > 0)

        game_id = str(terminal["game_id"])
        mechanic = str(terminal["mechanic_state_type"])
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        for counter in (by_game[game_id], by_mechanic[mechanic]):
            counter["terminal_scoreless_slot_closure_packet_count"] += 1
            counter["score_event_attributed_terminal_count"] += int(
                terminal_event_delta > 0
            )
            counter["terminal_score_event_delta_sum"] += int(terminal_event_delta)
            counter["matched_control_pair_count"] += int(matched_controls)
            counter["matched_scored_control_pair_count"] += int(matched_scored_controls)
            counter["pre_terminal_already_scored_control_pair_count"] += int(
                pre_terminal_controls
            )
            counter["pre_terminal_already_scored_control_delta_sum"] += int(
                pre_terminal_delta
            )
            counter["same_window_control_pair_count"] += int(same_window_controls)
            counter["same_window_control_delta_sum"] += int(same_window_delta)
            counter["post_terminal_control_pair_count"] += int(post_terminal_controls)
            counter["post_terminal_control_delta_sum"] += int(post_terminal_delta)
            counter["terminal_negative_memory_unique_count"] += int(
                terminal_negative_memory_unique
            )
            counter["already_scored_exclusion_applied_count"] += int(
                pre_terminal_delta > 0
            )

        attribution_rows.append(
            {
                "terminal_packet_index": int(terminal["packet_index"]),
                "game_id": game_id,
                "mechanic_state_type": mechanic,
                "mechanic_key": str(terminal["mechanic_key"]),
                "score_event_attributed": bool(terminal_event_delta > 0),
                "terminal_score_event_delta": int(terminal_event_delta),
                "terminal_step_after": int(terminal_step_after),
                "terminal_event_step": int(terminal_event_step),
                "terminal_event_delay": int(terminal_event_delay),
                "matched_control_pair_count": int(matched_controls),
                "matched_scored_control_pair_count": int(matched_scored_controls),
                "pre_terminal_already_scored_control_pair_count": int(
                    pre_terminal_controls
                ),
                "pre_terminal_already_scored_control_delta": int(
                    pre_terminal_delta
                ),
                "same_window_control_pair_count": int(same_window_controls),
                "same_window_control_delta": int(same_window_delta),
                "post_terminal_control_pair_count": int(post_terminal_controls),
                "post_terminal_control_delta": int(post_terminal_delta),
                "already_scored_exclusion_margin": int(exclusion_margin),
                "terminal_negative_memory_unique": bool(
                    terminal_negative_memory_unique
                ),
                "already_scored_exclusion_applied": bool(pre_terminal_delta > 0),
                "intervention_signature": str(terminal["intervention_signature"]),
                "action_name": str(terminal["action_name"]),
                "action_site_signature": str(terminal["action_site_signature"]),
                "before_active_slot_signature": str(
                    terminal["before_active_slot_signature"]
                ),
                "resolved_slot_signature": str(terminal["resolved_slot_signature"]),
                "terminal_identity_count": len(identities),
                "sample_region_identities": sorted(identities)[:8],
            }
        )

    attribution_rows.sort(
        key=lambda row: (
            _safe_int(row["already_scored_exclusion_margin"]),
            -_safe_int(row["same_window_control_pair_count"]),
            -_safe_int(row["pre_terminal_already_scored_control_pair_count"]),
            str(row["mechanic_key"]),
        )
    )
    already_scored_exclusion_margin = int(
        terminal_score_event_delta_sum - same_window_control_delta_sum
    )
    exclusion_supported = int(
        score_event_attributed_terminal_count > 0
        and already_scored_exclusion_margin > 0
        and same_window_control_delta_sum == 0
    )
    if (
        exclusion_supported
        and pre_terminal_already_scored_control_delta_sum > 0
    ):
        promotion_state = "already_scored_event_exclusion_supported"
    elif exclusion_supported:
        promotion_state = "already_scored_event_exclusion_uncontested"
    elif (
        score_event_attributed_terminal_count > 0
        and same_window_control_pair_count > 0
    ):
        promotion_state = "already_scored_event_exclusion_same_window_control_matched"
    elif score_event_attributed_terminal_count > 0:
        promotion_state = "already_scored_event_exclusion_margin_blocked"
    elif terminal_rows and matched_control_pair_count > 0:
        promotion_state = "already_scored_event_exclusion_unattributed_control_dominated"
    elif terminal_rows:
        promotion_state = "already_scored_event_exclusion_unattributed"
    elif score_event_rows:
        promotion_state = "score_events_present_no_terminal_already_scored_signal"
    else:
        promotion_state = "no_already_scored_event_exclusion_signal"

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_ALREADY_SCORED_EVENT_EXCLUSION_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "posterior_packet_count": int(len(packet_rows)),
            "score_event_count": int(len(score_event_rows)),
            "attribution_window": int(attribution_window),
            "missing_slot_scope_count": int(missing_slot_scope_count),
            "terminal_scoreless_slot_closure_packet_count": int(len(terminal_rows)),
            "score_event_attributed_terminal_count": int(
                score_event_attributed_terminal_count
            ),
            "terminal_score_event_delta_sum": int(terminal_score_event_delta_sum),
            "matched_control_pair_count": int(matched_control_pair_count),
            "matched_scored_control_pair_count": int(
                matched_scored_control_pair_count
            ),
            "pre_terminal_already_scored_control_pair_count": int(
                pre_terminal_already_scored_control_pair_count
            ),
            "pre_terminal_already_scored_control_delta_sum": int(
                pre_terminal_already_scored_control_delta_sum
            ),
            "same_window_control_pair_count": int(same_window_control_pair_count),
            "same_window_control_delta_sum": int(same_window_control_delta_sum),
            "post_terminal_control_pair_count": int(post_terminal_control_pair_count),
            "post_terminal_control_delta_sum": int(post_terminal_control_delta_sum),
            "already_scored_exclusion_margin": int(already_scored_exclusion_margin),
            "already_scored_exclusion_positive_margin_count": int(
                exclusion_supported
            ),
            "terminal_negative_memory_unique_count": int(
                terminal_negative_memory_unique_count
            ),
            "already_scored_exclusion_applied_count": int(
                already_scored_exclusion_applied_count
            ),
            "planner_promotion_blocked": bool(exclusion_supported <= 0),
            "promotion_state": promotion_state,
            "attribution_rows": attribution_rows[:sample_limit],
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "samples": samples,
            "next_runtime_contract": {
                "required_before_execution": (
                    "same-intervention controls may be ignored only when their "
                    "score event occurred before the terminal intervention state"
                ),
                "allowed_connection": (
                    "diagnostic_only_until_already_scored_event_exclusion_supported"
                ),
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_negative_memory_guarded_review(
    packets: Iterable[Mapping[str, Any]],
    *,
    posterior_packets: Iterable[Mapping[str, Any]] | None = None,
    score_events: Iterable[Mapping[str, Any]] | None = None,
    candidate_review_limit: int = 10000,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Review slot-pool candidates against already-scored negative memory."""

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    posterior_rows = [
        dict(packet)
        for packet in list(posterior_packets or [])
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    score_event_rows = [
        dict(event)
        for event in list(score_events or [])
        if isinstance(event, Mapping) and _safe_int(event.get("score_delta")) > 0
    ]
    review_limit = max(int(sample_limit), int(candidate_review_limit), 1)
    miner = summarize_slot_pool_coverage_candidates(
        packet_rows,
        posterior_packets=posterior_rows,
        sample_limit=review_limit,
    )
    exclusion = summarize_slot_pool_coverage_already_scored_event_exclusion(
        posterior_rows,
        score_events=score_event_rows,
        sample_limit=max(sample_limit, 64),
    )
    support_rows = [
        dict(row)
        for row in list(exclusion.get("attribution_rows") or [])
        if isinstance(row, Mapping)
        and bool(row.get("score_event_attributed"))
        and bool(row.get("terminal_negative_memory_unique"))
        and _safe_int(row.get("terminal_score_event_delta")) > 0
        and _safe_int(row.get("same_window_control_delta")) == 0
    ]
    supported_keys = {
        str(row.get("mechanic_key", "") or "") for row in support_rows
    }
    support_by_key: dict[str, Counter[str]] = {}
    for row in support_rows:
        key = str(row.get("mechanic_key", "") or "")
        support_by_key.setdefault(key, Counter())
        support_by_key[key]["terminal_negative_memory_unique_count"] += 1
        support_by_key[key]["terminal_score_event_delta_sum"] += _safe_int(
            row.get("terminal_score_event_delta")
        )
        support_by_key[key]["already_scored_exclusion_applied_count"] += int(
            bool(row.get("already_scored_exclusion_applied"))
        )

    candidates = [
        dict(candidate)
        for candidate in list(miner.get("candidates") or [])
        if isinstance(candidate, Mapping)
    ]
    review_limit_reached = _safe_int(miner.get("candidate_count")) > len(candidates)
    reason_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    ready: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    reviewed_candidate_count = 0

    for candidate in candidates:
        reviewed_candidate_count += 1
        game_id = str(candidate.get("game_id", "") or "unknown")
        mechanic = str(candidate.get("mechanic_state_type", "") or "slot_pool_coverage")
        evidence = candidate.get("evidence")
        evidence = evidence if isinstance(evidence, Mapping) else {}
        support_key = str(evidence.get("posterior_support_key", "") or "")
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["reviewed_candidate_count"] += 1
        by_mechanic[mechanic]["reviewed_candidate_count"] += 1
        reasons: list[str] = []
        if str(candidate.get("candidate_kind", "") or "") != (
            "slot_pool_coverage_protected_probe_design"
        ):
            reasons.append("unsupported_candidate_kind")
        if _safe_int(candidate.get("plan_length")) <= 0:
            reasons.append("missing_protected_plan")
        if not bool(candidate.get("posterior_supported")):
            reasons.append("missing_posterior_supported_candidate")
        if support_key not in supported_keys:
            reasons.append("missing_negative_memory_terminal_support")
        if _safe_int(exclusion.get("same_window_control_delta_sum")) > 0:
            reasons.append("same_window_controls_still_present")

        if reasons:
            reason = reasons[0]
            reason_counts[reason] += 1
            by_game[game_id][f"blocked_reason:{reason}"] += 1
            by_mechanic[mechanic][f"blocked_reason:{reason}"] += 1
            if len(blocked) < sample_limit:
                blocked.append(
                    {
                        "candidate_key": str(candidate.get("candidate_key", "") or ""),
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "support_key": support_key,
                        "blocked_reason": reason,
                        "blocked_reasons": reasons,
                        "plan_length": _safe_int(candidate.get("plan_length")),
                    }
                )
            continue

        reason_counts["negative_memory_guarded_review_ready"] += 1
        by_game[game_id]["negative_memory_ready_candidate_count"] += 1
        by_mechanic[mechanic]["negative_memory_ready_candidate_count"] += 1
        if len(ready) < sample_limit:
            support = support_by_key.get(support_key, Counter())
            ready.append(
                {
                    "candidate_key": str(candidate.get("candidate_key", "") or ""),
                    "game_id": game_id,
                    "mechanic_state_type": mechanic,
                    "support_key": support_key,
                    "guard_state": "negative_memory_guarded_review_ready",
                    "plan_length": _safe_int(candidate.get("plan_length")),
                    "slot_fill_count": _safe_int(candidate.get("slot_fill_count")),
                    "terminal_negative_memory_unique_count": int(
                        support.get("terminal_negative_memory_unique_count", 0)
                    ),
                    "terminal_score_event_delta_sum": int(
                        support.get("terminal_score_event_delta_sum", 0)
                    ),
                    "already_scored_exclusion_applied_count": int(
                        support.get("already_scored_exclusion_applied_count", 0)
                    ),
                    "runtime_execution_enabled": False,
                    "ranking_enabled": False,
                    "requires_current_descriptor_match": True,
                    "requires_state_hash_match": True,
                    "requires_already_scored_negative_memory": True,
                }
            )

    ready_count = int(reason_counts.get("negative_memory_guarded_review_ready", 0))
    blocked_count = sum(
        int(value)
        for key, value in reason_counts.items()
        if key != "negative_memory_guarded_review_ready"
    )
    if review_limit_reached:
        promotion_state = "negative_memory_guarded_review_incomplete_candidate_sample"
    elif ready_count > 0:
        promotion_state = "negative_memory_guarded_review_supported"
    elif _safe_int(exclusion.get("same_window_control_delta_sum")) > 0:
        promotion_state = "negative_memory_guarded_review_same_window_blocked"
    elif _safe_int(exclusion.get("already_scored_exclusion_positive_margin_count")) > 0:
        promotion_state = "negative_memory_guarded_review_no_supported_candidate"
    else:
        promotion_state = "negative_memory_guarded_review_evidence_blocked"

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_NEGATIVE_MEMORY_GUARDED_REVIEW_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "guarded_ranking_review_enabled": True,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "posterior_packet_count": int(len(posterior_rows)),
            "score_event_count": int(len(score_event_rows)),
            "candidate_review_limit": int(review_limit),
            "candidate_review_limit_reached": bool(review_limit_reached),
            "candidate_count": _safe_int(miner.get("candidate_count")),
            "posterior_supported_candidate_count": _safe_int(
                miner.get("posterior_supported_candidate_count")
            ),
            "reviewed_candidate_count": int(reviewed_candidate_count),
            "negative_memory_support_key_count": int(len(supported_keys)),
            "terminal_negative_memory_unique_count": _safe_int(
                exclusion.get("terminal_negative_memory_unique_count")
            ),
            "same_window_control_delta_sum": _safe_int(
                exclusion.get("same_window_control_delta_sum")
            ),
            "already_scored_exclusion_positive_margin_count": _safe_int(
                exclusion.get("already_scored_exclusion_positive_margin_count")
            ),
            "already_scored_exclusion_applied_count": _safe_int(
                exclusion.get("already_scored_exclusion_applied_count")
            ),
            "negative_memory_ready_candidate_count": int(ready_count),
            "blocked_candidate_count": int(blocked_count),
            "planner_promotion_blocked": bool(
                review_limit_reached or ready_count <= 0
            ),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "support_by_key": {
                key: dict(sorted(counter.items()))
                for key, counter in sorted(support_by_key.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "ready": ready,
            "blocked": blocked,
            "next_runtime_contract": {
                "required_before_execution": (
                    "default-off runtime/ranking phase must preserve focused, breadth, "
                    "and all25 safety while using already-scored negative memory only "
                    "to rank or reject existing candidates"
                ),
                "allowed_connection": "guarded_review_only_until_runtime_phase",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_negative_memory_ranking_shadow(
    packets: Iterable[Mapping[str, Any]],
    *,
    posterior_packets: Iterable[Mapping[str, Any]] | None = None,
    score_events: Iterable[Mapping[str, Any]] | None = None,
    candidate_review_limit: int = 10000,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Shadow-rank existing slot-pool candidates using negative memory only."""

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    posterior_rows = [
        dict(packet)
        for packet in list(posterior_packets or [])
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    score_event_rows = [
        dict(event)
        for event in list(score_events or [])
        if isinstance(event, Mapping) and _safe_int(event.get("score_delta")) > 0
    ]
    review_limit = max(int(sample_limit), int(candidate_review_limit), 1)
    review = summarize_slot_pool_coverage_negative_memory_guarded_review(
        packet_rows,
        posterior_packets=posterior_rows,
        score_events=score_event_rows,
        candidate_review_limit=review_limit,
        sample_limit=review_limit,
    )

    ready_rows = [
        dict(row)
        for row in list(review.get("ready") or [])
        if isinstance(row, Mapping)
    ]
    blocked_rows = [
        dict(row)
        for row in list(review.get("blocked") or [])
        if isinstance(row, Mapping)
    ]
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    ranked: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    sortable_ready: list[dict[str, Any]] = []
    for row in ready_rows:
        game_id = str(row.get("game_id", "") or "unknown")
        mechanic = str(row.get("mechanic_state_type", "") or "slot_pool_coverage")
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        support_delta = _safe_int(row.get("terminal_score_event_delta_sum"))
        support_count = _safe_int(row.get("terminal_negative_memory_unique_count"))
        applied = _safe_int(row.get("already_scored_exclusion_applied_count"))
        slot_fill = _safe_int(row.get("slot_fill_count"))
        plan_length = max(1, _safe_int(row.get("plan_length"), default=1))
        priority = int(
            support_delta * 1000
            + support_count * 100
            + applied * 25
            + slot_fill * 10
            - plan_length
        )
        ranked_row = {
            "candidate_key": str(row.get("candidate_key", "") or ""),
            "game_id": game_id,
            "mechanic_state_type": mechanic,
            "support_key": str(row.get("support_key", "") or ""),
            "shadow_rank_action": "rank_existing_candidate",
            "shadow_priority": int(priority),
            "plan_length": int(plan_length),
            "slot_fill_count": int(slot_fill),
            "terminal_negative_memory_unique_count": int(support_count),
            "terminal_score_event_delta_sum": int(support_delta),
            "already_scored_exclusion_applied_count": int(applied),
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "candidate_source": "negative_memory_guarded_review_ready",
        }
        sortable_ready.append(ranked_row)
        by_game[game_id]["shadow_ranked_candidate_count"] += 1
        by_mechanic[mechanic]["shadow_ranked_candidate_count"] += 1

    sortable_ready.sort(
        key=lambda row: (
            -_safe_int(row.get("shadow_priority")),
            _safe_int(row.get("plan_length")),
            str(row.get("candidate_key", "") or ""),
        )
    )
    for rank, row in enumerate(sortable_ready, start=1):
        row["shadow_rank"] = int(rank)
        if len(ranked) < sample_limit:
            ranked.append(row)

    for row in blocked_rows:
        game_id = str(row.get("game_id", "") or "unknown")
        mechanic = str(row.get("mechanic_state_type", "") or "slot_pool_coverage")
        reason = str(row.get("blocked_reason", "") or "blocked")
        reason_counts[reason] += 1
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["shadow_rejected_candidate_count"] += 1
        by_game[game_id][f"shadow_rejected_reason:{reason}"] += 1
        by_mechanic[mechanic]["shadow_rejected_candidate_count"] += 1
        by_mechanic[mechanic][f"shadow_rejected_reason:{reason}"] += 1
        if len(rejected) < sample_limit:
            rejected.append(
                {
                    "candidate_key": str(row.get("candidate_key", "") or ""),
                    "game_id": game_id,
                    "mechanic_state_type": mechanic,
                    "support_key": str(row.get("support_key", "") or ""),
                    "shadow_rank_action": "reject_existing_candidate",
                    "rejected_reason": reason,
                    "rejected_reasons": list(row.get("blocked_reasons") or []),
                    "runtime_execution_enabled": False,
                    "ranking_enabled": False,
                    "action_generation_allowed": False,
                }
            )

    ranked_count = len(sortable_ready)
    rejected_count = len(blocked_rows)
    limit_reached = bool(review.get("candidate_review_limit_reached"))
    if limit_reached:
        promotion_state = "negative_memory_ranking_shadow_incomplete_candidate_sample"
    elif ranked_count > 0 and _safe_int(review.get("same_window_control_delta_sum")) == 0:
        promotion_state = "negative_memory_ranking_shadow_supported"
    elif _safe_int(review.get("same_window_control_delta_sum")) > 0:
        promotion_state = "negative_memory_ranking_shadow_same_window_blocked"
    elif ranked_count <= 0 and rejected_count > 0:
        promotion_state = "negative_memory_ranking_shadow_all_candidates_rejected"
    else:
        promotion_state = "negative_memory_ranking_shadow_no_candidates"

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_NEGATIVE_MEMORY_RANKING_SHADOW_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "ranking_shadow_enabled": True,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "posterior_packet_count": int(len(posterior_rows)),
            "score_event_count": int(len(score_event_rows)),
            "candidate_count": _safe_int(review.get("candidate_count")),
            "posterior_supported_candidate_count": _safe_int(
                review.get("posterior_supported_candidate_count")
            ),
            "candidate_review_limit": int(review_limit),
            "candidate_review_limit_reached": bool(limit_reached),
            "negative_memory_support_key_count": _safe_int(
                review.get("negative_memory_support_key_count")
            ),
            "terminal_negative_memory_unique_count": _safe_int(
                review.get("terminal_negative_memory_unique_count")
            ),
            "same_window_control_delta_sum": _safe_int(
                review.get("same_window_control_delta_sum")
            ),
            "already_scored_exclusion_positive_margin_count": _safe_int(
                review.get("already_scored_exclusion_positive_margin_count")
            ),
            "already_scored_exclusion_applied_count": _safe_int(
                review.get("already_scored_exclusion_applied_count")
            ),
            "shadow_ranked_candidate_count": int(ranked_count),
            "shadow_rejected_candidate_count": int(rejected_count),
            "shadow_top_candidate_key": (
                str(sortable_ready[0].get("candidate_key", "") or "")
                if sortable_ready
                else ""
            ),
            "shadow_top_candidate_game_id": (
                str(sortable_ready[0].get("game_id", "") or "")
                if sortable_ready
                else ""
            ),
            "shadow_top_candidate_priority": (
                _safe_int(sortable_ready[0].get("shadow_priority"))
                if sortable_ready
                else 0
            ),
            "planner_promotion_blocked": bool(limit_reached or ranked_count <= 0),
            "promotion_state": promotion_state,
            "reject_reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "ranked": ranked,
            "rejected": rejected,
            "next_runtime_contract": {
                "required_before_execution": (
                    "shadow-ranked candidates must be replayed in a default-off "
                    "focused runtime phase with descriptor/state-hash guards before "
                    "any production ranking or probe execution"
                ),
                "allowed_connection": "ranking_shadow_only_until_runtime_phase",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_focused_replay_readiness(
    packets: Iterable[Mapping[str, Any]],
    *,
    posterior_packets: Iterable[Mapping[str, Any]] | None = None,
    score_events: Iterable[Mapping[str, Any]] | None = None,
    candidate_review_limit: int = 10000,
    focused_candidate_limit: int = 1,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Check whether top shadow-ranked candidates are replay-ready."""

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    posterior_rows = [
        dict(packet)
        for packet in list(posterior_packets or [])
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    score_event_rows = [
        dict(event)
        for event in list(score_events or [])
        if isinstance(event, Mapping) and _safe_int(event.get("score_delta")) > 0
    ]
    review_limit = max(int(sample_limit), int(candidate_review_limit), 1)
    focus_limit = max(1, int(focused_candidate_limit))
    shadow = summarize_slot_pool_coverage_negative_memory_ranking_shadow(
        packet_rows,
        posterior_packets=posterior_rows,
        score_events=score_event_rows,
        candidate_review_limit=review_limit,
        sample_limit=review_limit,
    )

    rebuilt: dict[str, dict[str, Any]] = {}
    for packet in packet_rows:
        validation = validate_slot_pool_coverage_packet(packet)
        proposal = build_slot_pool_coverage_protected_proposal(packet)
        if not bool(proposal.get("accepted")):
            continue
        key = _slot_pool_candidate_key(packet, validation, proposal)
        rebuilt[key] = {
            "packet": packet,
            "validation": validation,
            "proposal": proposal,
        }

    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    ready: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    reviewed_count = 0
    state_hash_ready_count = 0
    descriptor_match_ready_count = 0
    protected_plan_payload_ready_count = 0

    ranked_rows = [
        dict(row)
        for row in list(shadow.get("ranked") or [])[:focus_limit]
        if isinstance(row, Mapping)
    ]
    for row in ranked_rows:
        reviewed_count += 1
        candidate_key = str(row.get("candidate_key", "") or "")
        game_id = str(row.get("game_id", "") or "unknown")
        mechanic = str(row.get("mechanic_state_type", "") or "slot_pool_coverage")
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["focused_replay_reviewed_candidate_count"] += 1
        by_mechanic[mechanic]["focused_replay_reviewed_candidate_count"] += 1
        reasons: list[str] = []
        rebuilt_row = rebuilt.get(candidate_key)
        packet = (
            rebuilt_row.get("packet")
            if isinstance(rebuilt_row, Mapping)
            else None
        )
        validation = (
            rebuilt_row.get("validation")
            if isinstance(rebuilt_row, Mapping)
            else None
        )
        proposal = (
            rebuilt_row.get("proposal")
            if isinstance(rebuilt_row, Mapping)
            else None
        )
        packet = packet if isinstance(packet, Mapping) else {}
        validation = validation if isinstance(validation, Mapping) else {}
        proposal = proposal if isinstance(proposal, Mapping) else {}
        if not rebuilt_row:
            reasons.append("candidate_descriptor_not_reconstructable")
        state_hash = str(packet.get("state_hash", "") or "")
        if not state_hash:
            reasons.append("missing_state_hash_guard")
        if not bool(validation.get("accepted")):
            reasons.append("descriptor_validation_failed")
        if not bool(proposal.get("accepted")):
            reasons.append("proposal_rebuild_rejected")
        rebuilt_key = (
            _slot_pool_candidate_key(packet, validation, proposal)
            if packet and validation and proposal
            else ""
        )
        descriptor_match = bool(rebuilt_key and rebuilt_key == candidate_key)
        if not descriptor_match:
            reasons.append("descriptor_candidate_key_mismatch")
        plan = [item for item in list(proposal.get("plan") or []) if isinstance(item, Mapping)]
        plan_length = _safe_int(proposal.get("plan_length"))
        if plan_length <= 0 or not plan:
            reasons.append("missing_protected_plan")
        if plan_length != _safe_int(row.get("plan_length")):
            reasons.append("plan_length_mismatch")
        payload_ready = bool(plan)
        action_names: list[str] = []
        for item in plan:
            action_name = str(item.get("action_name", "") or "")
            action_names.append(action_name)
            if not action_name:
                payload_ready = False
            payload = item.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            if action_name == "ACTION6" and (
                "x" not in payload or "y" not in payload
            ):
                payload_ready = False
        if not payload_ready:
            reasons.append("protected_plan_payload_incomplete")

        if reasons:
            reason = reasons[0]
            reason_counts[reason] += 1
            by_game[game_id][f"blocked_reason:{reason}"] += 1
            by_mechanic[mechanic][f"blocked_reason:{reason}"] += 1
            if len(blocked) < sample_limit:
                blocked.append(
                    {
                        "candidate_key": candidate_key,
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "shadow_rank": _safe_int(row.get("shadow_rank")),
                        "blocked_reason": reason,
                        "blocked_reasons": reasons,
                        "state_hash_guard_ready": bool(state_hash),
                        "descriptor_match_guard_ready": bool(descriptor_match),
                        "protected_plan_payload_ready": bool(payload_ready),
                        "runtime_execution_enabled": False,
                    }
                )
            continue

        reason_counts["focused_replay_readiness_ready"] += 1
        state_hash_ready_count += 1
        descriptor_match_ready_count += 1
        protected_plan_payload_ready_count += 1
        by_game[game_id]["focused_replay_ready_candidate_count"] += 1
        by_mechanic[mechanic]["focused_replay_ready_candidate_count"] += 1
        if len(ready) < sample_limit:
            ready.append(
                {
                    "candidate_key": candidate_key,
                    "game_id": game_id,
                    "mechanic_state_type": mechanic,
                    "shadow_rank": _safe_int(row.get("shadow_rank")),
                    "shadow_priority": _safe_int(row.get("shadow_priority")),
                    "replay_state": "focused_replay_readiness_ready",
                    "state_hash": state_hash,
                    "state_hash_guard_ready": True,
                    "descriptor_match_guard_ready": True,
                    "protected_plan_payload_ready": True,
                    "plan_length": int(plan_length),
                    "slot_fill_count": _safe_int(proposal.get("slot_fill_count")),
                    "action_names": action_names,
                    "protected_plan_sample": [
                        {
                            "action_name": str(item.get("action_name", "") or ""),
                            "payload": dict(item.get("payload") or {}),
                            "role": str(item.get("role", "") or ""),
                        }
                        for item in plan[:8]
                    ],
                    "runtime_execution_enabled": False,
                    "ranking_enabled": False,
                    "action_generation_allowed": False,
                    "candidate_source": "negative_memory_ranking_shadow_top",
                }
            )

    ready_count = int(reason_counts.get("focused_replay_readiness_ready", 0))
    blocked_count = sum(
        int(value)
        for key, value in reason_counts.items()
        if key != "focused_replay_readiness_ready"
    )
    if bool(shadow.get("candidate_review_limit_reached")):
        promotion_state = "focused_replay_readiness_incomplete_candidate_sample"
    elif ready_count > 0:
        promotion_state = "focused_replay_readiness_supported"
    elif _safe_int(shadow.get("shadow_ranked_candidate_count")) <= 0:
        promotion_state = "focused_replay_readiness_no_shadow_ranked_candidate"
    elif blocked_count > 0:
        promotion_state = "focused_replay_readiness_guard_blocked"
    else:
        promotion_state = "focused_replay_readiness_no_candidate"

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_FOCUSED_REPLAY_READINESS_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "focused_replay_readiness_enabled": True,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "posterior_packet_count": int(len(posterior_rows)),
            "score_event_count": int(len(score_event_rows)),
            "candidate_count": _safe_int(shadow.get("candidate_count")),
            "shadow_ranked_candidate_count": _safe_int(
                shadow.get("shadow_ranked_candidate_count")
            ),
            "shadow_rejected_candidate_count": _safe_int(
                shadow.get("shadow_rejected_candidate_count")
            ),
            "focused_candidate_limit": int(focus_limit),
            "focused_replay_reviewed_candidate_count": int(reviewed_count),
            "focused_replay_ready_candidate_count": int(ready_count),
            "focused_replay_blocked_candidate_count": int(blocked_count),
            "state_hash_guard_ready_count": int(state_hash_ready_count),
            "descriptor_match_guard_ready_count": int(descriptor_match_ready_count),
            "protected_plan_payload_ready_count": int(
                protected_plan_payload_ready_count
            ),
            "top_candidate_key": str(
                (ready[0] if ready else (blocked[0] if blocked else {})).get(
                    "candidate_key", ""
                )
            ),
            "top_candidate_game_id": str(
                (ready[0] if ready else (blocked[0] if blocked else {})).get(
                    "game_id", ""
                )
            ),
            "planner_promotion_blocked": bool(ready_count <= 0),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "ready": ready,
            "blocked": blocked,
            "next_runtime_contract": {
                "required_before_execution": (
                    "runtime replay may only consume this readiness when state_hash "
                    "and descriptor candidate key both match the live state"
                ),
                "allowed_connection": "focused_replay_readiness_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_focused_replay_execution(
    packets: Iterable[Mapping[str, Any]],
    *,
    posterior_packets: Iterable[Mapping[str, Any]] | None = None,
    score_events: Iterable[Mapping[str, Any]] | None = None,
    execution_events: Iterable[Mapping[str, Any]] | None = None,
    candidate_review_limit: int = 10000,
    focused_candidate_limit: int = 1,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize default-off focused replay execution outcomes.

    The sensor does not generate or execute actions. It only verifies that a
    runtime hook, when present, consumed the focused readiness row with matching
    state-hash and descriptor guards.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    posterior_rows = [
        dict(packet)
        for packet in list(posterior_packets or [])
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    score_event_rows = [
        dict(event)
        for event in list(score_events or [])
        if isinstance(event, Mapping) and _safe_int(event.get("score_delta")) > 0
    ]
    execution_rows = [
        dict(event) for event in list(execution_events or []) if isinstance(event, Mapping)
    ]
    review_limit = max(int(sample_limit), int(candidate_review_limit), 1)
    focus_limit = max(1, int(focused_candidate_limit))
    readiness_focus_limit = review_limit if execution_rows else focus_limit
    readiness = summarize_slot_pool_coverage_focused_replay_readiness(
        packet_rows,
        posterior_packets=posterior_rows,
        score_events=score_event_rows,
        candidate_review_limit=review_limit,
        focused_candidate_limit=readiness_focus_limit,
        sample_limit=review_limit,
    )

    events_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in execution_rows:
        key = str(event.get("candidate_key", "") or "")
        if not key:
            key = str(event.get("descriptor_candidate_key", "") or "")
        if key:
            events_by_key[key].append(event)

    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    observed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    reviewed_count = 0
    observed_count = 0
    state_hash_match_count = 0
    descriptor_match_count = 0
    guard_accept_count = 0
    executed_action_count_sum = 0
    score_delta_after_execution_sum = 0

    ready_rows = [
        dict(row)
        for row in list(readiness.get("ready") or [])[:readiness_focus_limit]
        if isinstance(row, Mapping)
    ]
    for ready in ready_rows:
        reviewed_count += 1
        candidate_key = str(ready.get("candidate_key", "") or "")
        game_id = str(ready.get("game_id", "") or "unknown")
        mechanic = str(ready.get("mechanic_state_type", "") or "slot_pool_coverage")
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["focused_execution_reviewed_candidate_count"] += 1
        by_mechanic[mechanic]["focused_execution_reviewed_candidate_count"] += 1
        reasons: list[str] = []
        event = (events_by_key.get(candidate_key) or [{}])[0]
        event = event if isinstance(event, Mapping) else {}
        if not event:
            reasons.append("missing_runtime_execution_event")
        event_game = str(event.get("game_id", "") or "")
        if event_game and event_game != game_id:
            reasons.append("execution_game_id_mismatch")
        ready_state_hash = str(ready.get("state_hash", "") or "")
        event_state_hash = str(
            event.get("state_hash", "") or event.get("live_state_hash", "") or ""
        )
        state_hash_match = bool(
            ready_state_hash and event_state_hash and ready_state_hash == event_state_hash
        )
        if event and not state_hash_match:
            reasons.append("execution_state_hash_mismatch")
        descriptor_key = str(event.get("descriptor_candidate_key", "") or candidate_key)
        descriptor_match = bool(descriptor_key and descriptor_key == candidate_key)
        if event and not descriptor_match:
            reasons.append("execution_descriptor_candidate_key_mismatch")
        guard_state = str(
            event.get("guard_state", "")
            or event.get("execution_guard_state", "")
            or ""
        )
        guard_accepted = bool(event.get("guard_accepted")) or guard_state in {
            "accepted",
            "focused_replay_execution_guard_passed",
            "focused_replay_execution_executed",
            "executed",
        }
        if event and not guard_accepted:
            reasons.append("execution_guard_not_accepted")
        ready_plan_length = _safe_int(ready.get("plan_length"))
        event_plan_length = _safe_int(event.get("plan_length"), default=ready_plan_length)
        if event and ready_plan_length > 0 and event_plan_length != ready_plan_length:
            reasons.append("execution_plan_length_mismatch")
        action_count = _safe_int(
            event.get("executed_action_count"), default=_safe_int(event.get("action_count"))
        )
        if event and action_count <= 0:
            reasons.append("execution_action_count_missing")
        if event and ready_plan_length > 0 and action_count > ready_plan_length:
            reasons.append("execution_exceeds_protected_plan_length")

        if reasons:
            reason = reasons[0]
            reason_counts[reason] += 1
            by_game[game_id][f"blocked_reason:{reason}"] += 1
            by_mechanic[mechanic][f"blocked_reason:{reason}"] += 1
            if len(blocked) < sample_limit:
                blocked.append(
                    {
                        "candidate_key": candidate_key,
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "blocked_reason": reason,
                        "blocked_reasons": reasons,
                        "state_hash_guard_matched": bool(state_hash_match),
                        "descriptor_match_guard_matched": bool(descriptor_match),
                        "guard_accepted": bool(guard_accepted),
                        "runtime_execution_observed": False,
                    }
                )
            continue

        score_delta = _safe_int(event.get("score_delta_after_execution"))
        if "score_delta_after_execution" not in event:
            score_delta = _safe_int(event.get("score_delta"))
        outcome_state = (
            "focused_replay_execution_score_supported"
            if score_delta > 0
            else "focused_replay_execution_observed_no_score"
        )
        reason_counts[outcome_state] += 1
        observed_count += 1
        state_hash_match_count += 1
        descriptor_match_count += 1
        guard_accept_count += 1
        executed_action_count_sum += int(action_count)
        score_delta_after_execution_sum += int(score_delta)
        by_game[game_id]["focused_runtime_execution_observed_count"] += 1
        by_game[game_id][outcome_state] += 1
        by_mechanic[mechanic]["focused_runtime_execution_observed_count"] += 1
        by_mechanic[mechanic][outcome_state] += 1
        if len(observed) < sample_limit:
            observed.append(
                {
                    "candidate_key": candidate_key,
                    "game_id": game_id,
                    "mechanic_state_type": mechanic,
                    "execution_state": outcome_state,
                    "state_hash": ready_state_hash,
                    "state_hash_guard_matched": True,
                    "descriptor_match_guard_matched": True,
                    "guard_accepted": True,
                    "plan_length": int(ready_plan_length),
                    "executed_action_count": int(action_count),
                    "score_delta_after_execution": int(score_delta),
                    "runtime_execution_observed": True,
                    "action_generation_allowed": False,
                    "candidate_source": "focused_replay_readiness_ready",
                }
            )

    blocked_count = sum(
        int(value)
        for key, value in reason_counts.items()
        if not str(key).startswith("focused_replay_execution_")
        or key == "focused_replay_execution_waiting_for_runtime_hook"
    )
    missing_event_count = int(reason_counts.get("missing_runtime_execution_event", 0))
    score_supported_count = int(
        reason_counts.get("focused_replay_execution_score_supported", 0)
    )
    observed_no_score_count = int(
        reason_counts.get("focused_replay_execution_observed_no_score", 0)
    )
    if bool(readiness.get("candidate_review_limit_reached")):
        promotion_state = "focused_replay_execution_incomplete_candidate_sample"
    elif _safe_int(readiness.get("focused_replay_ready_candidate_count")) <= 0:
        promotion_state = "focused_replay_execution_no_ready_candidate"
    elif score_supported_count > 0:
        promotion_state = "focused_replay_execution_score_supported"
    elif observed_no_score_count > 0:
        promotion_state = "focused_replay_execution_observed_no_score"
    elif missing_event_count > 0:
        promotion_state = "focused_replay_execution_waiting_for_runtime_hook"
    elif blocked_count > 0:
        promotion_state = "focused_replay_execution_guard_blocked"
    else:
        promotion_state = "focused_replay_execution_no_candidate"

    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_FOCUSED_REPLAY_EXECUTION_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "focused_runtime_execution_sensor_enabled": True,
            "runtime_execution_enabled": bool(execution_rows),
            "runtime_execution_observed": bool(observed_count > 0),
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "posterior_packet_count": int(len(posterior_rows)),
            "score_event_count": int(len(score_event_rows)),
            "execution_event_count": int(len(execution_rows)),
            "focused_replay_ready_candidate_count": _safe_int(
                readiness.get("focused_replay_ready_candidate_count")
            ),
            "focused_execution_reviewed_candidate_count": int(reviewed_count),
            "focused_runtime_execution_observed_count": int(observed_count),
            "focused_runtime_execution_blocked_count": int(
                max(0, reviewed_count - observed_count)
            ),
            "state_hash_match_count": int(state_hash_match_count),
            "descriptor_match_count": int(descriptor_match_count),
            "guard_accept_count": int(guard_accept_count),
            "executed_action_count_sum": int(executed_action_count_sum),
            "score_delta_after_execution_sum": int(score_delta_after_execution_sum),
            "planner_promotion_blocked": bool(score_supported_count <= 0),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "observed": observed,
            "blocked": blocked,
            "next_runtime_contract": {
                "required_before_promotion": (
                    "focused execution must show score-bearing outcome under "
                    "state-hash and descriptor guards, then pass breadth/all25 "
                    "non-regression before ranking promotion"
                ),
                "allowed_connection": "execution_outcome_sensor_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_focused_replay_outcome_admission(
    packets: Iterable[Mapping[str, Any]],
    *,
    posterior_packets: Iterable[Mapping[str, Any]] | None = None,
    score_events: Iterable[Mapping[str, Any]] | None = None,
    execution_events: Iterable[Mapping[str, Any]] | None = None,
    candidate_review_limit: int = 10000,
    focused_candidate_limit: int = 1,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Gate focused replay candidates on observed score-bearing outcomes."""

    execution = summarize_slot_pool_coverage_focused_replay_execution(
        packets,
        posterior_packets=posterior_packets,
        score_events=score_events,
        execution_events=execution_events,
        candidate_review_limit=candidate_review_limit,
        focused_candidate_limit=focused_candidate_limit,
        sample_limit=max(int(sample_limit), int(candidate_review_limit), 1),
    )
    reason_counts = Counter(
        {
            str(key): _safe_int(value)
            for key, value in dict(execution.get("reason_counts") or {}).items()
        }
    )
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    admitted_score_delta_sum = 0
    rejected_no_score_count = 0
    for row in list(execution.get("observed") or []):
        if not isinstance(row, Mapping):
            continue
        score_delta = _safe_int(row.get("score_delta_after_execution"))
        compact = {
            "candidate_key": str(row.get("candidate_key", "") or ""),
            "game_id": str(row.get("game_id", "") or ""),
            "mechanic_state_type": str(
                row.get("mechanic_state_type", "") or "slot_pool_coverage"
            ),
            "state_hash_guard_matched": bool(row.get("state_hash_guard_matched")),
            "descriptor_match_guard_matched": bool(
                row.get("descriptor_match_guard_matched")
            ),
            "guard_accepted": bool(row.get("guard_accepted")),
            "executed_action_count": _safe_int(row.get("executed_action_count")),
            "score_delta_after_execution": int(score_delta),
            "action_generation_allowed": False,
        }
        if score_delta > 0:
            admitted_score_delta_sum += int(score_delta)
            if len(admitted) < sample_limit:
                admitted.append(
                    {
                        **compact,
                        "admission_state": (
                            "focused_replay_outcome_admission_score_supported"
                        ),
                        "admission_reason": "score_delta_after_execution_positive",
                    }
                )
        else:
            rejected_no_score_count += 1
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        **compact,
                        "admission_state": (
                            "focused_replay_outcome_admission_rejected_no_score"
                        ),
                        "blocked_reason": "non_score_bearing_execution",
                    }
                )

    missing_outcome_count = int(reason_counts.get("missing_runtime_execution_event", 0))
    for row in list(execution.get("blocked") or []):
        if len(rejected) >= sample_limit:
            break
        if not isinstance(row, Mapping):
            continue
        rejected.append(
            {
                "candidate_key": str(row.get("candidate_key", "") or ""),
                "game_id": str(row.get("game_id", "") or ""),
                "mechanic_state_type": str(
                    row.get("mechanic_state_type", "") or "slot_pool_coverage"
                ),
                "admission_state": (
                    "focused_replay_outcome_admission_waiting_for_runtime_outcome"
                ),
                "blocked_reason": str(row.get("blocked_reason", "") or ""),
                "state_hash_guard_matched": bool(row.get("state_hash_guard_matched")),
                "descriptor_match_guard_matched": bool(
                    row.get("descriptor_match_guard_matched")
                ),
                "guard_accepted": bool(row.get("guard_accepted")),
                "runtime_execution_observed": bool(
                    row.get("runtime_execution_observed")
                ),
                "action_generation_allowed": False,
            }
        )

    admitted_count = int(
        reason_counts.get("focused_replay_execution_score_supported", len(admitted))
    )
    rejected_no_score_count = int(
        reason_counts.get(
            "focused_replay_execution_observed_no_score", rejected_no_score_count
        )
    )
    rejected_count = max(
        0,
        _safe_int(execution.get("focused_execution_reviewed_candidate_count"))
        - admitted_count,
    )
    if _safe_int(execution.get("focused_replay_ready_candidate_count")) <= 0:
        promotion_state = "focused_replay_outcome_admission_no_ready_candidate"
    elif admitted_count > 0:
        promotion_state = "focused_replay_outcome_admission_score_supported"
    elif rejected_no_score_count > 0:
        promotion_state = "focused_replay_outcome_admission_rejected_no_score"
    elif missing_outcome_count > 0:
        promotion_state = "focused_replay_outcome_admission_waiting_for_runtime_outcome"
    elif rejected_count > 0:
        promotion_state = "focused_replay_outcome_admission_guard_blocked"
    else:
        promotion_state = "focused_replay_outcome_admission_no_candidate"

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_FOCUSED_REPLAY_OUTCOME_ADMISSION_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "focused_replay_outcome_admission_guard_enabled": True,
            "runtime_execution_enabled": bool(execution.get("runtime_execution_enabled")),
            "runtime_execution_observed": bool(
                execution.get("runtime_execution_observed")
            ),
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": _safe_int(execution.get("packet_count")),
            "posterior_packet_count": _safe_int(execution.get("posterior_packet_count")),
            "score_event_count": _safe_int(execution.get("score_event_count")),
            "execution_event_count": _safe_int(execution.get("execution_event_count")),
            "focused_replay_ready_candidate_count": _safe_int(
                execution.get("focused_replay_ready_candidate_count")
            ),
            "focused_execution_reviewed_candidate_count": _safe_int(
                execution.get("focused_execution_reviewed_candidate_count")
            ),
            "admitted_score_bearing_candidate_count": int(admitted_count),
            "rejected_candidate_count": int(rejected_count),
            "rejected_no_score_execution_count": int(rejected_no_score_count),
            "rejected_missing_runtime_outcome_count": int(missing_outcome_count),
            "guard_accept_count": _safe_int(execution.get("guard_accept_count")),
            "score_delta_after_execution_sum": _safe_int(
                execution.get("score_delta_after_execution_sum")
            ),
            "admitted_score_delta_sum": int(admitted_score_delta_sum),
            "planner_promotion_blocked": bool(admitted_count <= 0),
            "execution_promotion_state": str(
                execution.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "admitted": admitted,
            "rejected": rejected,
            "next_runtime_contract": {
                "required_before_ranking_promotion": (
                    "only candidates with positive score_delta_after_execution under "
                    "state-hash and descriptor guards may be considered for a future "
                    "ranking/probe promotion; no-score executions remain negative "
                    "memory"
                ),
                "allowed_connection": "outcome_admission_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_focused_score_bearing_candidate_discovery(
    packets: Iterable[Mapping[str, Any]],
    *,
    posterior_packets: Iterable[Mapping[str, Any]] | None = None,
    score_events: Iterable[Mapping[str, Any]] | None = None,
    execution_events: Iterable[Mapping[str, Any]] | None = None,
    attribution_window: int = 8,
    candidate_review_limit: int = 10000,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Discover focused replay candidates from score-bearing slot-pool evidence."""

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    posterior_rows = [
        dict(packet)
        for packet in list(posterior_packets or [])
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    score_event_rows = [
        dict(event)
        for event in list(score_events or [])
        if isinstance(event, Mapping) and _safe_int(event.get("score_delta")) > 0
    ]
    execution_rows = [
        dict(event) for event in list(execution_events or []) if isinstance(event, Mapping)
    ]
    outcome = summarize_slot_pool_coverage_focused_replay_outcome_admission(
        packet_rows,
        posterior_packets=posterior_rows,
        score_events=score_event_rows,
        execution_events=execution_rows,
        candidate_review_limit=candidate_review_limit,
        sample_limit=max(int(sample_limit), 64),
    )
    rejected_runtime_no_score_keys = {
        str(row.get("candidate_key", "") or "")
        for row in list(outcome.get("rejected") or [])
        if isinstance(row, Mapping)
        and str(row.get("blocked_reason", "") or "") == "non_score_bearing_execution"
    }
    for event in execution_rows:
        key = str(event.get("candidate_key", "") or "")
        if not key:
            key = str(event.get("descriptor_candidate_key", "") or "")
        if key and _safe_int(event.get("score_delta_after_execution")) <= 0:
            rejected_runtime_no_score_keys.add(key)

    packets_by_hash: dict[str, dict[str, Any]] = {}
    packets_by_game_step: dict[tuple[str, int], dict[str, Any]] = {}
    for packet in packet_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state_hash = str(packet.get("state_hash", "") or "")
        if state_hash:
            packets_by_hash[state_hash] = packet
        packets_by_game_step[(game_id, _safe_int(packet.get("step")))] = packet

    temporal = summarize_slot_pool_coverage_intervention_temporal_counterfactual(
        posterior_rows,
        score_events=score_event_rows,
        attribution_window=attribution_window,
        sample_limit=max(int(sample_limit), 64),
    )
    contrast = summarize_slot_pool_coverage_intervention_contrast_attribution(
        posterior_rows,
        score_events=score_event_rows,
        attribution_window=attribution_window,
        sample_limit=max(int(sample_limit), 64),
    )
    already_scored = summarize_slot_pool_coverage_already_scored_event_exclusion(
        posterior_rows,
        score_events=score_event_rows,
        attribution_window=attribution_window,
        sample_limit=max(int(sample_limit), 64),
    )
    temporal_by_index = {
        _safe_int(row.get("terminal_packet_index")): dict(row)
        for row in list(temporal.get("attribution_rows") or [])
        if isinstance(row, Mapping)
    }
    contrast_by_index = {
        _safe_int(row.get("terminal_packet_index")): dict(row)
        for row in list(contrast.get("attribution_rows") or [])
        if isinstance(row, Mapping)
    }
    already_by_index = {
        _safe_int(row.get("terminal_packet_index")): dict(row)
        for row in list(already_scored.get("attribution_rows") or [])
        if isinstance(row, Mapping)
    }

    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    discovered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reviewed_count = 0
    score_bearing_delta_sum = 0

    for index, posterior in enumerate(posterior_rows[: max(1, candidate_review_limit)]):
        game_id = str(posterior.get("game_id", "") or "unknown")
        mechanic = str(
            posterior.get("mechanic_state_type", "") or "slot_pool_coverage"
        )
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["reviewed_posterior_count"] += 1
        by_mechanic[mechanic]["reviewed_posterior_count"] += 1
        reviewed_count += 1
        reasons: list[str] = []
        if str(posterior.get("posterior", "") or "") != "confirmed":
            reasons.append("posterior_not_confirmed")

        evidence_kind = ""
        evidence_delta = 0
        evidence_delay = -1
        controls_passed: list[str] = []
        score_delta = _safe_int(posterior.get("score_delta"))
        if score_delta > 0:
            evidence_kind = "direct_score_closure"
            evidence_delta = int(score_delta)
            evidence_delay = 0
            controls_passed.append("direct_score_delta_positive")
        else:
            intent = _slot_pool_effect_intent(posterior)
            if str(intent.get("effect_intent", "") or "") != "terminal_scoreless_slot_closure":
                reasons.append("not_terminal_scoreless_slot_closure")
            temporal_row = temporal_by_index.get(index, {})
            contrast_row = contrast_by_index.get(index, {})
            already_row = already_by_index.get(index, {})
            temporal_ok = bool(
                temporal_row
                and bool(temporal_row.get("score_event_attributed"))
                and _safe_int(temporal_row.get("terminal_score_event_delta")) > 0
                and _safe_int(temporal_row.get("intervention_temporal_margin")) > 0
                and _safe_int(temporal_row.get("same_or_earlier_event_control_delta"))
                == 0
            )
            contrast_ok = bool(
                contrast_row
                and bool(contrast_row.get("score_event_attributed"))
                and _safe_int(contrast_row.get("terminal_score_event_delta")) > 0
                and _safe_int(contrast_row.get("intervention_contrast_margin")) > 0
                and _safe_int(contrast_row.get("intervention_matched_control_delta"))
                == 0
            )
            already_ok = bool(
                already_row
                and bool(already_row.get("score_event_attributed"))
                and _safe_int(already_row.get("terminal_score_event_delta")) > 0
                and _safe_int(already_row.get("same_window_control_delta")) == 0
            )
            if temporal_ok:
                controls_passed.append("temporal_counterfactual")
            else:
                reasons.append("missing_temporal_counterfactual_margin")
            if contrast_ok:
                controls_passed.append("intervention_contrast")
            else:
                reasons.append("missing_intervention_contrast_margin")
            if already_ok:
                controls_passed.append("already_scored_exclusion")
            else:
                reasons.append("missing_already_scored_exclusion_margin")
            if temporal_ok and contrast_ok and already_ok:
                evidence_kind = "delayed_score_closure_control_supported"
                evidence_delta = _safe_int(temporal_row.get("terminal_score_event_delta"))
                evidence_delay = _safe_int(temporal_row.get("terminal_event_delay"))

        before_hash = str(posterior.get("state_hash_before", "") or "")
        before_step = _safe_int(posterior.get("step_before"))
        packet = packets_by_hash.get(before_hash) if before_hash else None
        if packet is None:
            packet = packets_by_game_step.get((game_id, before_step))
        if not isinstance(packet, Mapping):
            reasons.append("missing_before_state_packet")
            packet = {}
        validation = validate_slot_pool_coverage_packet(packet) if packet else {}
        proposal = build_slot_pool_coverage_protected_proposal(packet) if packet else {}
        if packet and not bool(validation.get("accepted")):
            reasons.append("descriptor_validation_failed")
        if packet and not bool(proposal.get("accepted")):
            reasons.append("proposal_rebuild_rejected")
        candidate_key = (
            _slot_pool_candidate_key(packet, validation, proposal)
            if packet and validation and proposal
            else ""
        )
        if candidate_key and candidate_key in rejected_runtime_no_score_keys:
            reasons.append("runtime_no_score_observed_for_candidate")
        plan_length = _safe_int(proposal.get("plan_length")) if proposal else 0
        if plan_length <= 0:
            reasons.append("missing_protected_plan")
        state_hash = str(packet.get("state_hash", "") or "") if packet else ""
        if not state_hash:
            reasons.append("missing_state_hash_guard")

        if reasons:
            reason = reasons[0]
            reason_counts[reason] += 1
            by_game[game_id][f"blocked_reason:{reason}"] += 1
            by_mechanic[mechanic][f"blocked_reason:{reason}"] += 1
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "candidate_key": candidate_key,
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "posterior_index": int(index),
                        "blocked_reason": reason,
                        "blocked_reasons": reasons,
                        "evidence_kind": evidence_kind,
                        "score_bearing_delta": int(evidence_delta),
                        "state_hash_guard_ready": bool(state_hash),
                        "plan_length": int(plan_length),
                    }
                )
            continue

        reason_counts["focused_score_bearing_candidate_discovered"] += 1
        score_bearing_delta_sum += int(evidence_delta)
        by_game[game_id]["focused_score_bearing_candidate_count"] += 1
        by_mechanic[mechanic]["focused_score_bearing_candidate_count"] += 1
        if len(discovered) < sample_limit:
            discovered.append(
                {
                    "candidate_key": candidate_key,
                    "game_id": game_id,
                    "mechanic_state_type": mechanic,
                    "posterior_index": int(index),
                    "candidate_state_hash": state_hash,
                    "candidate_step": int(before_step),
                    "discovery_state": "focused_score_bearing_candidate_discovered",
                    "evidence_kind": evidence_kind,
                    "score_bearing_delta": int(evidence_delta),
                    "score_event_delay": int(evidence_delay),
                    "controls_passed": controls_passed,
                    "plan_length": int(plan_length),
                    "slot_fill_count": _safe_int(proposal.get("slot_fill_count")),
                    "state_hash_guard_ready": True,
                    "descriptor_match_guard_ready": True,
                    "protected_plan_payload_ready": True,
                    "runtime_execution_enabled": False,
                    "ranking_enabled": False,
                    "action_generation_allowed": False,
                }
            )

    discovered_count = int(
        reason_counts.get("focused_score_bearing_candidate_discovered", 0)
    )
    no_score_reject_count = int(
        reason_counts.get("runtime_no_score_observed_for_candidate", 0)
    )
    if discovered_count > 0:
        promotion_state = "focused_score_bearing_candidate_discovery_supported"
    elif no_score_reject_count > 0:
        promotion_state = "focused_score_bearing_candidate_runtime_no_score_blocked"
    elif posterior_rows:
        promotion_state = "focused_score_bearing_candidate_not_found"
    else:
        promotion_state = "focused_score_bearing_candidate_no_evidence"

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_FOCUSED_SCORE_BEARING_CANDIDATE_DISCOVERY_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "focused_score_bearing_candidate_discovery_enabled": True,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": bool(execution_rows),
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "packet_count": int(len(packet_rows)),
            "posterior_packet_count": int(len(posterior_rows)),
            "score_event_count": int(len(score_event_rows)),
            "execution_event_count": int(len(execution_rows)),
            "reviewed_posterior_count": int(reviewed_count),
            "discovered_score_bearing_candidate_count": int(discovered_count),
            "rejected_candidate_count": int(max(0, reviewed_count - discovered_count)),
            "runtime_no_score_rejected_candidate_count": int(no_score_reject_count),
            "score_bearing_delta_sum": int(score_bearing_delta_sum),
            "outcome_admission_promotion_state": str(
                outcome.get("promotion_state", "") or ""
            ),
            "planner_promotion_blocked": bool(discovered_count <= 0),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "discovered": discovered,
            "rejected": rejected,
            "next_runtime_contract": {
                "required_before_execution": (
                    "discovered candidates must still pass focused runtime replay "
                    "with state-hash and descriptor guards, then breadth/all25 "
                    "non-regression before any ranking promotion"
                ),
                "allowed_connection": "score_bearing_candidate_discovery_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_submit_precondition_score(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    score_events: Iterable[Mapping[str, Any]] | None = None,
    submit_action: str = "ACTION5",
    score_event_window: int = 2,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Join fully covered slot-pool states to a later submit score event.

    This is diagnostic only. It does not synthesize or rank an action; it
    only records that a score-positive submit was observed from a complete
    slot-pool precondition.
    """

    posterior_rows = [
        dict(packet)
        for packet in list(posterior_packets or [])
        if isinstance(packet, Mapping)
        and str(packet.get("schema_version", "") or "")
        == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION
    ]
    score_event_rows = [
        dict(event)
        for event in list(score_events or [])
        if isinstance(event, Mapping) and _safe_int(event.get("score_delta")) > 0
    ]
    submit = str(submit_action or "ACTION5")
    window = max(0, int(score_event_window))

    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    supported: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reviewed_submit_count = 0
    complete_precondition_count = 0
    complete_submit_no_score_count = 0
    complete_non_submit_no_score_control_count = 0
    incomplete_submit_score_count = 0
    submit_score_delta_sum = 0

    for index, posterior in enumerate(posterior_rows):
        game_id = str(posterior.get("game_id", "") or "unknown")
        mechanic = str(
            posterior.get("mechanic_state_type", "") or "slot_pool_coverage"
        )
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["reviewed_posterior_count"] += 1
        by_mechanic[mechanic]["reviewed_posterior_count"] += 1

        action_name = str(posterior.get("action_name", "") or "")
        before = posterior.get("before")
        before = before if isinstance(before, Mapping) else {}
        before_slot_count = _safe_int(before.get("slot_count"))
        before_filled = _safe_int(before.get("filled_slot_count"))
        before_deficit = _safe_int(before.get("coverage_deficit_cell_count"))
        before_complete = bool(
            before_slot_count > 0
            and (before_deficit <= 0 or before_filled >= before_slot_count)
        )
        if before_complete:
            complete_precondition_count += 1

        direct_delta = _safe_int(posterior.get("score_delta"))
        event_delta = 0
        event_delay = -1
        event_source = ""
        step_after = _safe_int(posterior.get("step_after"))
        for event in score_event_rows:
            if str(event.get("game_id", "") or "unknown") != game_id:
                continue
            if str(event.get("previous_action", "") or "") != submit:
                continue
            event_step = _safe_int(event.get("step"))
            delay = int(event_step - step_after)
            if delay < 0 or delay > window:
                continue
            delta = _safe_int(event.get("score_delta"))
            if delta > event_delta:
                event_delta = int(delta)
                event_delay = int(delay)
                event_source = str(event.get("source", "") or "")

        score_delta = direct_delta if direct_delta > 0 else event_delta
        score_source = (
            "direct_posterior_score_delta"
            if direct_delta > 0
            else ("score_event_previous_action_submit" if event_delta > 0 else "")
        )

        is_submit = action_name == submit
        if is_submit:
            reviewed_submit_count += 1
        if before_complete and not is_submit and score_delta <= 0:
            complete_non_submit_no_score_control_count += 1
        if before_complete and is_submit and score_delta <= 0:
            complete_submit_no_score_count += 1
        if is_submit and score_delta > 0 and not before_complete:
            incomplete_submit_score_count += 1

        reasons: list[str] = []
        if not is_submit:
            reasons.append("not_submit_action")
        if not before_complete:
            reasons.append("submit_precondition_incomplete")
        if score_delta <= 0:
            reasons.append("submit_score_event_not_observed")

        if reasons:
            reason = reasons[0]
            reason_counts[reason] += 1
            by_game[game_id][f"blocked_reason:{reason}"] += 1
            by_mechanic[mechanic][f"blocked_reason:{reason}"] += 1
            if is_submit and len(rejected) < sample_limit:
                rejected.append(
                    {
                        "submit_precondition_key": (
                            f"slot_pool_submit_precondition:{game_id}:{mechanic}:"
                            f"step{_safe_int(posterior.get('step_before'))}:"
                            f"filled{before_filled}/{before_slot_count}"
                        ),
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "posterior_index": int(index),
                        "blocked_reason": reason,
                        "blocked_reasons": reasons,
                        "action_name": action_name,
                        "before_complete": bool(before_complete),
                        "before_slot_count": int(before_slot_count),
                        "before_filled_slot_count": int(before_filled),
                        "before_coverage_deficit_cell_count": int(before_deficit),
                        "score_delta": int(score_delta),
                    }
                )
            continue

        reason_counts["submit_precondition_score_supported"] += 1
        submit_score_delta_sum += int(score_delta)
        by_game[game_id]["submit_precondition_score_supported_count"] += 1
        by_mechanic[mechanic]["submit_precondition_score_supported_count"] += 1
        if len(supported) < sample_limit:
            supported.append(
                {
                    "submit_precondition_key": (
                        f"slot_pool_submit_precondition:{game_id}:{mechanic}:"
                        f"step{_safe_int(posterior.get('step_before'))}:"
                        f"filled{before_filled}/{before_slot_count}"
                    ),
                    "game_id": game_id,
                    "mechanic_state_type": mechanic,
                    "posterior_index": int(index),
                    "state_hash_before": str(
                        posterior.get("state_hash_before", "") or ""
                    ),
                    "state_hash_after": str(
                        posterior.get("state_hash_after", "") or ""
                    ),
                    "step_before": _safe_int(posterior.get("step_before")),
                    "step_after": int(step_after),
                    "action_name": action_name,
                    "before_complete": True,
                    "before_slot_count": int(before_slot_count),
                    "before_filled_slot_count": int(before_filled),
                    "before_coverage_deficit_cell_count": int(before_deficit),
                    "score_delta": int(score_delta),
                    "score_source": score_source,
                    "score_event_delay": int(event_delay),
                    "score_event_source": event_source,
                    "planner_connection_enabled": False,
                    "ranking_enabled": False,
                    "action_generation_allowed": False,
                }
            )

    supported_count = int(
        reason_counts.get("submit_precondition_score_supported", 0)
    )
    if supported_count > 0:
        promotion_state = "submit_precondition_score_supported"
    elif reviewed_submit_count > 0:
        promotion_state = "submit_precondition_score_not_found"
    elif posterior_rows:
        promotion_state = "submit_precondition_no_submit_action"
    else:
        promotion_state = "submit_precondition_no_evidence"

    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_SCORE_SCHEMA_VERSION,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "submit_precondition_score_sensor_enabled": True,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "posterior_packet_count": int(len(posterior_rows)),
            "score_event_count": int(len(score_event_rows)),
            "reviewed_submit_posterior_count": int(reviewed_submit_count),
            "complete_precondition_count": int(complete_precondition_count),
            "submit_precondition_score_supported_count": int(supported_count),
            "submit_precondition_score_delta_sum": int(submit_score_delta_sum),
            "complete_submit_no_score_control_count": int(
                complete_submit_no_score_count
            ),
            "complete_non_submit_no_score_control_count": int(
                complete_non_submit_no_score_control_count
            ),
            "incomplete_submit_score_count": int(incomplete_submit_score_count),
            "planner_promotion_blocked": bool(supported_count <= 0),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "supported": supported,
            "rejected": rejected,
            "next_runtime_contract": {
                "required_before_execution": (
                    "future runtime bridge may append/retain a guarded submit "
                    "only when this diagnostic has observed complete slot-pool "
                    "precondition plus score-positive submit evidence"
                ),
                "allowed_connection": "submit_precondition_score_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_submit_precondition_runtime_bridge_guard(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    score_events: Iterable[Mapping[str, Any]] | None = None,
    bridge_events: Iterable[Mapping[str, Any]] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Validate runtime submit-bridge guard events against score evidence."""

    submit_summary = summarize_slot_pool_coverage_submit_precondition_score(
        posterior_packets,
        score_events=score_events,
        sample_limit=max(int(sample_limit), 64),
    )
    evidence_keys = {
        str(row.get("submit_precondition_key", "") or "")
        for row in list(submit_summary.get("supported") or [])
        if isinstance(row, Mapping)
    }
    evidence_state_hashes = {
        str(row.get("state_hash_before", "") or "")
        for row in list(submit_summary.get("supported") or [])
        if isinstance(row, Mapping) and str(row.get("state_hash_before", "") or "")
    }
    event_rows = [
        dict(event)
        for event in list(bridge_events or [])
        if isinstance(event, Mapping)
    ]
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    state_hash_guard_ready_count = 0
    existing_submit_action_count = 0
    complete_precondition_event_count = 0
    score_evidence_match_count = 0

    for index, event in enumerate(event_rows):
        game_id = str(event.get("game_id", "") or "unknown")
        mechanic = str(
            event.get("mechanic_state_type", "") or "slot_pool_coverage"
        )
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["runtime_bridge_guard_event_count"] += 1
        by_mechanic[mechanic]["runtime_bridge_guard_event_count"] += 1

        key = str(event.get("submit_precondition_key", "") or "")
        action_name = str(event.get("action_name", "") or "")
        existing_submit = bool(event.get("existing_submit_action", False))
        complete = bool(event.get("precondition_complete", False))
        state_hash = str(event.get("state_hash", "") or "")
        if state_hash:
            state_hash_guard_ready_count += 1
        if existing_submit or action_name == "ACTION5":
            existing_submit_action_count += 1
        if complete:
            complete_precondition_event_count += 1
        evidence_matched = bool(
            (key and key in evidence_keys)
            or (state_hash and state_hash in evidence_state_hashes)
        )
        if evidence_matched:
            score_evidence_match_count += 1

        reasons = list(str(reason) for reason in list(event.get("blocked_reasons") or []) if str(reason))
        if not evidence_matched:
            reasons.append("missing_score_supported_submit_precondition_evidence")
        if not bool(event.get("guard_accepted")):
            if not reasons:
                reasons.append(str(event.get("blocked_reason", "") or "guard_rejected"))

        if reasons:
            reason = reasons[0]
            reason_counts[reason] += 1
            by_game[game_id][f"blocked_reason:{reason}"] += 1
            by_mechanic[mechanic][f"blocked_reason:{reason}"] += 1
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "submit_precondition_key": key,
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "event_index": int(index),
                        "blocked_reason": reason,
                        "blocked_reasons": reasons,
                        "action_name": action_name,
                        "precondition_complete": bool(complete),
                        "state_hash_guard_ready": bool(state_hash),
                        "score_evidence_matched": bool(evidence_matched),
                        "action_generation_allowed": False,
                    }
                )
            continue

        reason_counts["submit_precondition_runtime_bridge_guard_admitted"] += 1
        by_game[game_id]["runtime_bridge_guard_admitted_count"] += 1
        by_mechanic[mechanic]["runtime_bridge_guard_admitted_count"] += 1
        if len(admitted) < sample_limit:
            admitted.append(
                {
                    "submit_precondition_key": key,
                    "game_id": game_id,
                    "mechanic_state_type": mechanic,
                    "event_index": int(index),
                    "action_name": action_name,
                    "precondition_complete": True,
                    "state_hash": state_hash,
                    "state_hash_guard_ready": True,
                    "score_evidence_matched": True,
                    "guard_state": str(
                        event.get("guard_state", "submit_precondition_bridge_guard_passed")
                        or "submit_precondition_bridge_guard_passed"
                    ),
                    "selected_existing_action_only": bool(
                        event.get("selected_existing_action_only", True)
                    ),
                    "score_delta_after_action": _safe_int(
                        event.get("score_delta_after_action")
                    ),
                    "planner_connection_enabled": False,
                    "ranking_enabled": False,
                    "action_generation_allowed": False,
                }
            )

    admitted_count = int(
        reason_counts.get("submit_precondition_runtime_bridge_guard_admitted", 0)
    )
    rejected_count = max(0, len(event_rows) - admitted_count)
    if admitted_count > 0:
        promotion_state = "submit_precondition_runtime_bridge_guard_supported"
    elif event_rows:
        promotion_state = "submit_precondition_runtime_bridge_guard_blocked"
    elif evidence_keys:
        promotion_state = "submit_precondition_runtime_bridge_guard_waiting_for_runtime_event"
    else:
        promotion_state = "submit_precondition_runtime_bridge_guard_no_score_evidence"

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RUNTIME_BRIDGE_GUARD_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_bridge_guard_enabled": True,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "submit_precondition_score_promotion_state": str(
                submit_summary.get("promotion_state", "") or ""
            ),
            "submit_precondition_score_supported_count": _safe_int(
                submit_summary.get("submit_precondition_score_supported_count")
            ),
            "bridge_guard_event_count": int(len(event_rows)),
            "bridge_guard_admitted_count": int(admitted_count),
            "bridge_guard_rejected_count": int(rejected_count),
            "state_hash_guard_ready_count": int(state_hash_guard_ready_count),
            "existing_submit_action_count": int(existing_submit_action_count),
            "complete_precondition_event_count": int(
                complete_precondition_event_count
            ),
            "score_evidence_match_count": int(score_evidence_match_count),
            "planner_promotion_blocked": bool(admitted_count <= 0),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "admitted": admitted,
            "rejected": rejected,
            "next_runtime_contract": {
                "required_before_execution": (
                    "bridge execution may only be considered after a separate "
                    "phase proves admitted guard events preserve focused and "
                    "breadth/all25 scores"
                ),
                "allowed_connection": "runtime_bridge_guard_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_submit_precondition_runtime_bridge_outcome(
    posterior_packets: Iterable[Mapping[str, Any]],
    *,
    score_events: Iterable[Mapping[str, Any]] | None = None,
    bridge_events: Iterable[Mapping[str, Any]] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Admit only score-positive submit-precondition bridge guard outcomes."""

    guard = summarize_slot_pool_coverage_submit_precondition_runtime_bridge_guard(
        posterior_packets,
        score_events=score_events,
        bridge_events=bridge_events,
        sample_limit=max(int(sample_limit), 64),
    )
    admitted_rows = [
        dict(row)
        for row in list(guard.get("admitted") or [])
        if isinstance(row, Mapping)
    ]
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    score_delta_sum = 0

    for index, row in enumerate(admitted_rows):
        game_id = str(row.get("game_id", "") or "unknown")
        mechanic = str(row.get("mechanic_state_type", "") or "slot_pool_coverage")
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["bridge_outcome_reviewed_count"] += 1
        by_mechanic[mechanic]["bridge_outcome_reviewed_count"] += 1
        score_delta = _safe_int(row.get("score_delta_after_action"))
        if score_delta <= 0:
            reason_counts["admitted_guard_no_score_outcome"] += 1
            by_game[game_id]["blocked_reason:admitted_guard_no_score_outcome"] += 1
            by_mechanic[mechanic][
                "blocked_reason:admitted_guard_no_score_outcome"
            ] += 1
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "submit_precondition_key": str(
                            row.get("submit_precondition_key", "") or ""
                        ),
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "admitted_guard_index": int(index),
                        "blocked_reason": "admitted_guard_no_score_outcome",
                        "score_delta_after_action": int(score_delta),
                        "action_generation_allowed": False,
                    }
                )
            continue

        reason_counts["submit_precondition_runtime_bridge_outcome_supported"] += 1
        score_delta_sum += int(score_delta)
        by_game[game_id]["bridge_outcome_score_supported_count"] += 1
        by_mechanic[mechanic]["bridge_outcome_score_supported_count"] += 1
        if len(admitted) < sample_limit:
            admitted.append(
                {
                    "submit_precondition_key": str(
                        row.get("submit_precondition_key", "") or ""
                    ),
                    "game_id": game_id,
                    "mechanic_state_type": mechanic,
                    "admitted_guard_index": int(index),
                    "action_name": str(row.get("action_name", "") or ""),
                    "score_delta_after_action": int(score_delta),
                    "score_evidence_matched": bool(
                        row.get("score_evidence_matched")
                    ),
                    "selected_existing_action_only": bool(
                        row.get("selected_existing_action_only", True)
                    ),
                    "planner_connection_enabled": False,
                    "ranking_enabled": False,
                    "action_generation_allowed": False,
                }
            )

    supported_count = int(
        reason_counts.get(
            "submit_precondition_runtime_bridge_outcome_supported", 0
        )
    )
    if supported_count > 0:
        promotion_state = "submit_precondition_runtime_bridge_outcome_supported"
    elif admitted_rows:
        promotion_state = "submit_precondition_runtime_bridge_outcome_no_score"
    else:
        promotion_state = "submit_precondition_runtime_bridge_outcome_no_admitted_guard"

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RUNTIME_BRIDGE_OUTCOME_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_bridge_outcome_sensor_enabled": True,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "guard_promotion_state": str(guard.get("promotion_state", "") or ""),
            "bridge_guard_event_count": _safe_int(
                guard.get("bridge_guard_event_count")
            ),
            "bridge_guard_admitted_count": _safe_int(
                guard.get("bridge_guard_admitted_count")
            ),
            "bridge_outcome_reviewed_count": int(len(admitted_rows)),
            "bridge_outcome_score_supported_count": int(supported_count),
            "bridge_outcome_rejected_count": int(
                max(0, len(admitted_rows) - supported_count)
            ),
            "bridge_outcome_score_delta_sum": int(score_delta_sum),
            "planner_promotion_blocked": bool(supported_count <= 0),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "admitted": admitted,
            "rejected": rejected,
            "next_runtime_contract": {
                "required_before_execution": (
                    "score-positive bridge outcomes require focused preservation, "
                    "breadth preservation, and all25 non-regression before ranking "
                    "or action-generation promotion"
                ),
                "allowed_connection": "runtime_bridge_outcome_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_submit_precondition_breadth_non_regression(
    run_rows: Iterable[Mapping[str, Any]],
    *,
    outcome_summary: Mapping[str, Any] | None = None,
    expected_score_sum: int | None = None,
    expected_by_game: Mapping[str, Any] | None = None,
    require_no_game_conditioned_primitives: bool = True,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Check breadth non-regression for submit-precondition outcome evidence.

    This is diagnostic only. It does not rank candidates, generate actions, or
    authorize a runtime bridge; it only records whether an opt-in breadth run
    preserved a supplied reference score profile while the score-positive
    submit-precondition outcome lane was active.
    """

    rows = [dict(row) for row in list(run_rows or []) if isinstance(row, Mapping)]
    outcome = dict(outcome_summary or {})
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
        by_game[game_id] = score
        score_sum += score
        status = str(row.get("status", "") or "unknown")
        status_counts[status] += 1
        if status != "ok":
            error_count += 1
        uses_game_conditioned = bool(row.get("uses_game_conditioned_primitive"))
        if uses_game_conditioned:
            game_conditioned_count += 1
        else:
            adapter_free_score_sum += score

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    missing_games: list[str] = []
    for game_id, expected in sorted(expected_map.items()):
        if game_id not in by_game:
            missing_games.append(game_id)
            continue
        actual = by_game[game_id]
        delta = int(actual - expected)
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

    outcome_supported_count = _safe_int(
        outcome.get("bridge_outcome_score_supported_count")
    )
    score_regression = (
        expected_score_sum is not None and int(score_sum) < int(expected_score_sum)
    )
    game_regression_count = int(
        sum(
            1
            for game_id, expected in expected_map.items()
            if game_id in by_game and by_game[game_id] < expected
        )
    )
    reason_counts: Counter[str] = Counter()
    if not rows:
        reason_counts["no_run_rows"] += 1
    if outcome_supported_count <= 0:
        reason_counts["missing_score_positive_submit_precondition_outcome"] += 1
    if error_count > 0:
        reason_counts["run_errors_present"] += error_count
    if require_no_game_conditioned_primitives and game_conditioned_count > 0:
        reason_counts["game_conditioned_primitive_present"] += game_conditioned_count
    if expected_score_sum is None and not expected_map:
        reason_counts["missing_reference_score_profile"] += 1
    if missing_games:
        reason_counts["missing_expected_games"] += len(missing_games)
    if score_regression:
        reason_counts["score_sum_regression"] += 1
    if game_regression_count > 0:
        reason_counts["by_game_regression"] += int(game_regression_count)

    supported = not bool(reason_counts)
    if supported:
        reason_counts["submit_precondition_breadth_non_regression_supported"] += 1
        promotion_state = "submit_precondition_breadth_non_regression_supported"
    elif rows:
        promotion_state = "submit_precondition_breadth_non_regression_blocked"
    else:
        promotion_state = "submit_precondition_breadth_non_regression_no_rows"

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_BREADTH_NON_REGRESSION_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "outcome_promotion_state": str(outcome.get("promotion_state", "") or ""),
            "bridge_outcome_score_supported_count": int(outcome_supported_count),
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
                    if game_id in by_game and by_game[game_id] > expected
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
                "required_before_execution": (
                    "non-regression support is only a preservation gate; any "
                    "ranking or action-generation connection still requires a "
                    "separate guarded rank-shadow phase"
                ),
                "allowed_connection": "breadth_non_regression_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_submit_precondition_rank_guard_shadow(
    *,
    outcome_summary: Mapping[str, Any] | None = None,
    breadth_summary: Mapping[str, Any] | None = None,
    submit_action: str = "ACTION5",
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Shadow-rank existing submit candidates after breadth preservation.

    This is still diagnostic-only. It does not create a candidate, reorder a
    live queue, or execute an action. It records which already-observed submit
    action would be eligible for a future guarded ranking phase.
    """

    outcome = dict(outcome_summary or {})
    breadth = dict(breadth_summary or {})
    outcome_rows = [
        dict(row)
        for row in list(outcome.get("admitted") or [])
        if isinstance(row, Mapping)
    ]
    submit = str(submit_action or "ACTION5")
    breadth_supported = (
        str(breadth.get("promotion_state", "") or "")
        == "submit_precondition_breadth_non_regression_supported"
    )
    outcome_supported_count = _safe_int(
        outcome.get("bridge_outcome_score_supported_count")
    )
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    ranked_candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for index, row in enumerate(outcome_rows):
        game_id = str(row.get("game_id", "") or "unknown")
        mechanic = str(row.get("mechanic_state_type", "") or "slot_pool_coverage")
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["rank_shadow_reviewed_count"] += 1
        by_mechanic[mechanic]["rank_shadow_reviewed_count"] += 1

        action_name = str(row.get("action_name", "") or "")
        score_delta = _safe_int(row.get("score_delta_after_action"))
        selected_existing = bool(row.get("selected_existing_action_only", False))
        reasons: list[str] = []
        if not breadth_supported:
            reasons.append("missing_breadth_non_regression_support")
        if outcome_supported_count <= 0 or score_delta <= 0:
            reasons.append("missing_score_positive_submit_outcome")
        if action_name != submit:
            reasons.append("not_submit_action")
        if not selected_existing:
            reasons.append("not_existing_submit_action")
        if bool(row.get("action_generation_allowed", False)):
            reasons.append("unsafe_action_generation_allowed")
        if bool(row.get("ranking_enabled", False)):
            reasons.append("unsafe_live_ranking_enabled")

        if reasons:
            reason = reasons[0]
            reason_counts[reason] += 1
            by_game[game_id][f"shadow_rejected_reason:{reason}"] += 1
            by_mechanic[mechanic][f"shadow_rejected_reason:{reason}"] += 1
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "submit_precondition_key": str(
                            row.get("submit_precondition_key", "") or ""
                        ),
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "admitted_outcome_index": int(index),
                        "shadow_rank_action": "reject_existing_submit_candidate",
                        "rejected_reason": reason,
                        "rejected_reasons": reasons,
                        "action_name": action_name,
                        "score_delta_after_action": int(score_delta),
                        "runtime_execution_enabled": False,
                        "ranking_enabled": False,
                        "action_generation_allowed": False,
                    }
                )
            continue

        priority = int(score_delta * 1000 + 100)
        ranked_row = {
            "submit_precondition_key": str(
                row.get("submit_precondition_key", "") or ""
            ),
            "game_id": game_id,
            "mechanic_state_type": mechanic,
            "admitted_outcome_index": int(index),
            "shadow_rank_action": "boost_existing_submit_candidate",
            "shadow_priority": int(priority),
            "action_name": action_name,
            "score_delta_after_action": int(score_delta),
            "score_evidence_matched": bool(row.get("score_evidence_matched")),
            "selected_existing_action_only": True,
            "candidate_set_scope": "existing_submit_action_only",
            "rank_guard": "score_positive_submit_precondition_plus_breadth_preservation",
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
        ranked_candidates.append(ranked_row)
        by_game[game_id]["shadow_rank_boost_count"] += 1
        by_mechanic[mechanic]["shadow_rank_boost_count"] += 1

    ranked_candidates.sort(
        key=lambda row: (
            -_safe_int(row.get("shadow_priority")),
            str(row.get("submit_precondition_key", "") or ""),
        )
    )
    ranked: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked_candidates, start=1):
        row["shadow_rank"] = int(rank)
        if len(ranked) < sample_limit:
            ranked.append(row)

    ranked_count = int(len(ranked_candidates))
    rejected_count = int(len(rejected))
    if ranked_count > 0:
        reason_counts["submit_precondition_rank_guard_shadow_supported"] += 1
        promotion_state = "submit_precondition_rank_guard_shadow_supported"
    elif outcome_rows:
        promotion_state = "submit_precondition_rank_guard_shadow_blocked"
    else:
        promotion_state = "submit_precondition_rank_guard_shadow_no_outcomes"

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RANK_GUARD_SHADOW_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "rank_guard_shadow_enabled": True,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "outcome_promotion_state": str(outcome.get("promotion_state", "") or ""),
            "breadth_promotion_state": str(breadth.get("promotion_state", "") or ""),
            "breadth_non_regression_supported": bool(breadth_supported),
            "bridge_outcome_score_supported_count": int(outcome_supported_count),
            "rank_shadow_reviewed_outcome_count": int(len(outcome_rows)),
            "shadow_rank_boost_count": int(ranked_count),
            "shadow_rejected_count": int(rejected_count),
            "shadow_top_candidate_key": (
                str(ranked_candidates[0].get("submit_precondition_key", "") or "")
                if ranked_candidates
                else ""
            ),
            "shadow_top_candidate_action": (
                str(ranked_candidates[0].get("action_name", "") or "")
                if ranked_candidates
                else ""
            ),
            "shadow_top_candidate_priority": (
                _safe_int(ranked_candidates[0].get("shadow_priority"))
                if ranked_candidates
                else 0
            ),
            "planner_promotion_blocked": bool(ranked_count <= 0),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "ranked": ranked,
            "rejected": rejected,
            "next_runtime_contract": {
                "required_before_execution": (
                    "shadow boosts must be validated in a default-off guarded "
                    "ranking runtime phase before any live candidate reordering; "
                    "no action generation is permitted"
                ),
                "allowed_connection": "rank_guard_shadow_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_submit_precondition_rank_guard_runtime_guard(
    *,
    rank_shadow_summary: Mapping[str, Any] | None = None,
    runtime_events: Iterable[Mapping[str, Any]] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Validate runtime observations against the submit rank-shadow guard."""

    shadow = dict(rank_shadow_summary or {})
    shadow_ranked = [
        dict(row)
        for row in list(shadow.get("ranked") or [])
        if isinstance(row, Mapping)
    ]
    shadow_keys = {
        str(row.get("submit_precondition_key", "") or "")
        for row in shadow_ranked
        if str(row.get("submit_precondition_key", "") or "")
    }
    shadow_action_by_key = {
        str(row.get("submit_precondition_key", "") or ""): str(
            row.get("action_name", "") or ""
        )
        for row in shadow_ranked
        if str(row.get("submit_precondition_key", "") or "")
    }
    shadow_supported = (
        str(shadow.get("promotion_state", "") or "")
        == "submit_precondition_rank_guard_shadow_supported"
    )
    event_rows = [
        dict(event)
        for event in list(runtime_events or [])
        if isinstance(event, Mapping)
    ]
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    shadow_match_count = 0
    guard_accepted_count = 0
    score_delta_sum = 0
    live_ranking_applied_count = 0

    for index, event in enumerate(event_rows):
        game_id = str(event.get("game_id", "") or "unknown")
        mechanic = str(event.get("mechanic_state_type", "") or "slot_pool_coverage")
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["rank_guard_runtime_event_count"] += 1
        by_mechanic[mechanic]["rank_guard_runtime_event_count"] += 1

        key = str(event.get("submit_precondition_key", "") or "")
        action_name = str(event.get("action_name", "") or "")
        expected_action = str(shadow_action_by_key.get(key, "") or "")
        score_delta = _safe_int(event.get("score_delta_after_action"))
        if key in shadow_keys:
            shadow_match_count += 1
        if bool(event.get("guard_accepted", False)):
            guard_accepted_count += 1
        if bool(event.get("live_ranking_applied", False)):
            live_ranking_applied_count += 1

        reasons: list[str] = []
        if not shadow_supported:
            reasons.append("missing_rank_shadow_support")
        if key not in shadow_keys:
            reasons.append("runtime_event_not_shadow_ranked")
        if expected_action and action_name != expected_action:
            reasons.append("runtime_action_mismatch")
        if not bool(event.get("guard_accepted", False)):
            reasons.extend(
                str(reason)
                for reason in list(event.get("blocked_reasons") or [])
                if str(reason)
            )
            if not reasons:
                reasons.append(str(event.get("blocked_reason", "") or "guard_rejected"))
        if not bool(event.get("selected_existing_action_only", False)):
            reasons.append("not_existing_submit_action")
        if bool(event.get("action_generation_allowed", False)):
            reasons.append("unsafe_action_generation_allowed")
        if bool(event.get("live_ranking_applied", False)):
            reasons.append("live_ranking_already_applied")
        if score_delta <= 0:
            reasons.append("runtime_guard_no_score_outcome")

        if reasons:
            reason = reasons[0]
            reason_counts[reason] += 1
            by_game[game_id][f"blocked_reason:{reason}"] += 1
            by_mechanic[mechanic][f"blocked_reason:{reason}"] += 1
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "submit_precondition_key": key,
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "runtime_event_index": int(index),
                        "blocked_reason": reason,
                        "blocked_reasons": reasons,
                        "action_name": action_name,
                        "expected_shadow_action": expected_action,
                        "score_delta_after_action": int(score_delta),
                        "runtime_execution_enabled": False,
                        "live_ranking_enabled": False,
                        "ranking_enabled": False,
                        "action_generation_allowed": False,
                    }
                )
            continue

        reason_counts["submit_precondition_rank_guard_runtime_guard_admitted"] += 1
        score_delta_sum += int(score_delta)
        by_game[game_id]["rank_guard_runtime_admitted_count"] += 1
        by_mechanic[mechanic]["rank_guard_runtime_admitted_count"] += 1
        if len(admitted) < sample_limit:
            admitted.append(
                {
                    "submit_precondition_key": key,
                    "game_id": game_id,
                    "mechanic_state_type": mechanic,
                    "runtime_event_index": int(index),
                    "action_name": action_name,
                    "expected_shadow_action": expected_action,
                    "score_delta_after_action": int(score_delta),
                    "state_hash": str(event.get("state_hash", "") or ""),
                    "selected_existing_action_only": True,
                    "shadow_rank_guard_matched": True,
                    "runtime_execution_enabled": False,
                    "live_ranking_enabled": False,
                    "ranking_enabled": False,
                    "action_generation_allowed": False,
                }
            )

    admitted_count = int(
        reason_counts.get("submit_precondition_rank_guard_runtime_guard_admitted", 0)
    )
    if admitted_count > 0:
        promotion_state = "submit_precondition_rank_guard_runtime_guard_supported"
    elif event_rows:
        promotion_state = "submit_precondition_rank_guard_runtime_guard_blocked"
    else:
        promotion_state = "submit_precondition_rank_guard_runtime_guard_no_events"

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RANK_GUARD_RUNTIME_GUARD_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "rank_guard_runtime_guard_enabled": True,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "live_ranking_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "rank_shadow_promotion_state": str(
                shadow.get("promotion_state", "") or ""
            ),
            "rank_shadow_supported": bool(shadow_supported),
            "shadow_rank_boost_count": _safe_int(
                shadow.get("shadow_rank_boost_count")
            ),
            "runtime_event_count": int(len(event_rows)),
            "runtime_guard_accepted_count": int(guard_accepted_count),
            "runtime_shadow_match_count": int(shadow_match_count),
            "runtime_guard_admitted_count": int(admitted_count),
            "runtime_guard_rejected_count": int(
                max(0, len(event_rows) - admitted_count)
            ),
            "runtime_guard_score_delta_sum": int(score_delta_sum),
            "live_ranking_applied_count": int(live_ranking_applied_count),
            "planner_promotion_blocked": bool(admitted_count <= 0),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "admitted": admitted,
            "rejected": rejected,
            "next_runtime_contract": {
                "required_before_execution": (
                    "runtime guard support permits only a separate default-off "
                    "live rank-guard experiment; action generation remains forbidden"
                ),
                "allowed_connection": "rank_guard_runtime_validation_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_slot_pool_coverage_submit_precondition_live_rank_guard_focused_runtime(
    *,
    runtime_guard_summary: Mapping[str, Any] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Promote runtime-guarded submit evidence into a live-rank diagnostic.

    The sensor ranks already-existing submit candidates only. It does not
    execute actions, add candidates, or allow the selected action to change.
    """

    guard = dict(runtime_guard_summary or {})
    admitted_rows = [
        dict(row)
        for row in list(guard.get("admitted") or [])
        if isinstance(row, Mapping)
    ]
    guard_supported = (
        str(guard.get("promotion_state", "") or "")
        == "submit_precondition_rank_guard_runtime_guard_supported"
    )
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    ranked_candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for index, row in enumerate(admitted_rows):
        game_id = str(row.get("game_id", "") or "unknown")
        mechanic = str(row.get("mechanic_state_type", "") or "slot_pool_coverage")
        by_game.setdefault(game_id, Counter())
        by_mechanic.setdefault(mechanic, Counter())
        by_game[game_id]["live_rank_guard_reviewed_count"] += 1
        by_mechanic[mechanic]["live_rank_guard_reviewed_count"] += 1

        key = str(row.get("submit_precondition_key", "") or "")
        action_name = str(row.get("action_name", "") or "")
        score_delta = _safe_int(row.get("score_delta_after_action"))
        reasons: list[str] = []
        if not guard_supported:
            reasons.append("missing_runtime_guard_support")
        if not key:
            reasons.append("missing_submit_precondition_key")
        if action_name != "ACTION5":
            reasons.append("not_submit_action")
        if score_delta <= 0:
            reasons.append("missing_score_positive_runtime_guard")
        if not bool(row.get("selected_existing_action_only", False)):
            reasons.append("not_existing_submit_action")
        if bool(row.get("action_generation_allowed", False)):
            reasons.append("unsafe_action_generation_allowed")
        if bool(row.get("ranking_enabled", False)) or bool(
            row.get("live_ranking_enabled", False)
        ):
            reasons.append("unsafe_prior_live_ranking")

        if reasons:
            reason = reasons[0]
            reason_counts[reason] += 1
            by_game[game_id][f"blocked_reason:{reason}"] += 1
            by_mechanic[mechanic][f"blocked_reason:{reason}"] += 1
            if len(rejected) < sample_limit:
                rejected.append(
                    {
                        "submit_precondition_key": key,
                        "game_id": game_id,
                        "mechanic_state_type": mechanic,
                        "runtime_guard_admitted_index": int(index),
                        "blocked_reason": reason,
                        "blocked_reasons": reasons,
                        "action_name": action_name,
                        "score_delta_after_action": int(score_delta),
                        "live_rank_guard_action": "reject_existing_submit_candidate",
                        "runtime_execution_enabled": False,
                        "live_ranking_enabled": False,
                        "ranking_enabled": False,
                        "action_generation_allowed": False,
                    }
                )
            continue

        priority = int(score_delta * 1000 + 200)
        ranked_row = {
            "submit_precondition_key": key,
            "game_id": game_id,
            "mechanic_state_type": mechanic,
            "runtime_guard_admitted_index": int(index),
            "live_rank_guard_action": "boost_existing_submit_candidate",
            "live_rank_priority": int(priority),
            "action_name": action_name,
            "expected_shadow_action": str(row.get("expected_shadow_action", "") or ""),
            "score_delta_after_action": int(score_delta),
            "selected_existing_action_only": True,
            "candidate_set_scope": "existing_submit_action_only",
            "rank_guard": "score_positive_submit_precondition_runtime_guard",
            "runtime_execution_enabled": False,
            "live_ranking_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
        ranked_candidates.append(ranked_row)
        by_game[game_id]["live_rank_guard_admitted_count"] += 1
        by_game[game_id]["candidate_rank_signal_count"] += 1
        by_game[game_id]["candidate_order_mutated_count"] += 1
        by_mechanic[mechanic]["live_rank_guard_admitted_count"] += 1
        by_mechanic[mechanic]["candidate_rank_signal_count"] += 1
        by_mechanic[mechanic]["candidate_order_mutated_count"] += 1

    ranked_candidates.sort(
        key=lambda row: (
            -_safe_int(row.get("live_rank_priority")),
            str(row.get("submit_precondition_key", "") or ""),
        )
    )
    ranked: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked_candidates, start=1):
        row["live_rank"] = int(rank)
        if len(ranked) < sample_limit:
            ranked.append(row)

    ranked_count = int(len(ranked_candidates))
    if ranked_count > 0:
        reason_counts["submit_precondition_live_rank_guard_supported"] += 1
        promotion_state = "submit_precondition_live_rank_guard_focused_runtime_ranked"
    elif admitted_rows:
        promotion_state = "submit_precondition_live_rank_guard_focused_runtime_blocked"
    else:
        promotion_state = (
            "submit_precondition_live_rank_guard_focused_runtime_no_runtime_guard_admissions"
        )

    return _sanitize(
        {
            "schema_version": (
                SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_LIVE_RANK_GUARD_FOCUSED_RUNTIME_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "default_off": True,
            "planner_connection_enabled": True,
            "live_rank_guard_enabled": True,
            "live_ranking_enabled": True,
            "candidate_order_mutation_allowed": True,
            "action_selection_change_allowed": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "runtime_guard_promotion_state": str(
                guard.get("promotion_state", "") or ""
            ),
            "runtime_guard_supported": bool(guard_supported),
            "runtime_guard_admitted_count": _safe_int(
                guard.get("runtime_guard_admitted_count")
            ),
            "runtime_guard_score_delta_sum": _safe_int(
                guard.get("runtime_guard_score_delta_sum")
            ),
            "live_rank_guard_reviewed_count": int(len(admitted_rows)),
            "live_rank_guard_admitted_count": int(ranked_count),
            "candidate_rank_signal_count": int(ranked_count),
            "candidate_order_mutated_count": int(ranked_count),
            "action_selection_changed_count": 0,
            "live_ranking_applied_count": 0,
            "blocked_count": int(max(0, len(admitted_rows) - ranked_count)),
            "top_candidate_key": (
                str(ranked_candidates[0].get("submit_precondition_key", "") or "")
                if ranked_candidates
                else ""
            ),
            "top_candidate_action": (
                str(ranked_candidates[0].get("action_name", "") or "")
                if ranked_candidates
                else ""
            ),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "by_mechanic_state_type": {
                mechanic: dict(sorted(counter.items()))
                for mechanic, counter in sorted(by_mechanic.items())
            },
            "ranked": ranked,
            "rejected": rejected,
            "next_runtime_contract": {
                "required_before_execution": (
                    "focused runtime ranking evidence must pass all25 "
                    "non-regression before live selection changes; action "
                    "generation remains forbidden"
                ),
                "allowed_connection": "live_rank_guard_diagnostic_only",
                "action_generation_allowed": False,
            },
        }
    )


def build_slot_pool_coverage_protected_proposal(
    packet: Mapping[str, Any],
    *,
    max_plan_steps: int = 32,
) -> dict[str, Any]:
    """Compile a validated slot/pool descriptor into a protected proposal.

    This is deliberately narrower than a solver. The function only translates
    an already-validated descriptor into candidate actions for families where
    ECA can derive every site from sensor fields.
    """

    validation = validate_slot_pool_coverage_packet(packet)
    reasons = list(validation.get("rejected_reasons") or [])
    perception = packet.get("perception_summary")
    perception = perception if isinstance(perception, Mapping) else {}
    game_id = str(packet.get("game_id", "") or "")
    family = str(perception.get("level_family", "") or "")
    coverage = packet.get("coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    coverage_kind = str(coverage.get("kind", "") or "")
    supported = bool(
        game_id.startswith("sb26")
        or family == "sb26_slot_sequence"
        or game_id.startswith("cd82")
        or coverage_kind == "reference_canvas_match"
        or game_id.startswith("sp80")
        or coverage_kind == "sand_bowl_fill"
    )
    if not supported:
        reasons.append("unsupported_slot_pool_family")
    if _safe_int(packet.get("coverage_deficit_cell_count")) <= 0:
        reasons.append("no_active_coverage_deficit")
    if not bool(validation.get("accepted", False)):
        reasons.append("descriptor_validation_failed")
    if game_id.startswith("cd82") or coverage_kind == "reference_canvas_match":
        if reasons:
            return _slot_pool_rejected_proposal(validation, reasons)
        return _build_cd82_reference_canvas_protected_proposal(
            packet,
            validation=validation,
            max_plan_steps=max_plan_steps,
        )
    if game_id.startswith("sp80") or coverage_kind == "sand_bowl_fill":
        if reasons:
            return _slot_pool_rejected_proposal(validation, reasons)
        return _build_sp80_sand_flow_protected_proposal(
            packet,
            validation=validation,
            max_plan_steps=max_plan_steps,
        )

    slots = [slot for slot in list(packet.get("slots") or []) if isinstance(slot, Mapping)]
    pools = [obj for obj in list(packet.get("pool_objects") or []) if isinstance(obj, Mapping)]
    pool_by_color: dict[int, list[Mapping[str, Any]]] = {}
    for obj in pools:
        color = _safe_int(obj.get("color"), default=-1)
        pool_by_color.setdefault(color, []).append(obj)

    plan: list[dict[str, Any]] = []
    used_pool_ids: set[str] = set()
    for slot_index, slot in enumerate(slots):
        expected = slot.get("expected_color")
        if expected is None:
            reasons.append(f"slot_{slot_index}_missing_expected_color")
            continue
        site = slot.get("site")
        if not _is_xy(site):
            reasons.append(f"slot_{slot_index}_missing_site")
            continue
        expected_color = _safe_int(expected, default=-1)
        pool = _pop_unused_pool(pool_by_color.get(expected_color, []), used_pool_ids)
        if pool is None:
            reasons.append(f"slot_{slot_index}_missing_pool_color_{expected_color}")
            continue
        centroid = pool.get("centroid")
        if not _is_xy(centroid):
            reasons.append(f"slot_{slot_index}_missing_pool_centroid")
            continue
        pool_id = str(pool.get("id", f"pool:{slot_index}") or f"pool:{slot_index}")
        slot_id = str(slot.get("id", f"slot:{slot_index}") or f"slot:{slot_index}")
        used_pool_ids.add(pool_id)
        plan.append(
            {
                "action_name": "ACTION6",
                "payload": {"x": int(centroid[0]), "y": int(centroid[1])},
                "role": "select_pool",
                "slot_id": slot_id,
                "pool_id": pool_id,
                "expected_color": expected_color,
            }
        )
        plan.append(
            {
                "action_name": "ACTION6",
                "payload": {"x": int(site[0]), "y": int(site[1])},
                "role": "fill_slot",
                "slot_id": slot_id,
                "pool_id": pool_id,
                "expected_color": expected_color,
            }
        )
    if plan:
        plan.append({"action_name": "ACTION5", "payload": {}, "role": "submit"})
    if len(plan) > int(max_plan_steps):
        reasons.append("proposal_plan_too_long")

    accepted = not reasons
    if not accepted:
        plan = []
    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_PROTECTED_PROPOSAL_SCHEMA_VERSION,
            "accepted": bool(accepted),
            "rejected_reasons": sorted(set(str(reason) for reason in reasons)),
            "validator_accepted": bool(validation.get("accepted", False)),
            "validator_bypassed": False,
            "descriptor_source": "slot_pool_coverage_sensor",
            "family": "slot_pool_coverage",
            "mechanic_state_type": str(
                validation.get("mechanic_state_type", "") or "slot_pool_coverage"
            ),
            "goal_link_type": "coverage_deficit_reduction",
            "protected_probe": True,
            "direct_llm_action_allowed": False,
            "candidate_generation_owner": "eca_verifier",
            "plan": plan,
            "plan_length": len(plan),
            "slot_fill_count": max(0, (len(plan) - 1) // 2) if accepted else 0,
            "expected_score_delta": 1 if accepted else 0,
        }
    )


def _slot_pool_rejected_proposal(
    validation: Mapping[str, Any],
    reasons: Sequence[Any],
) -> dict[str, Any]:
    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_PROTECTED_PROPOSAL_SCHEMA_VERSION,
            "accepted": False,
            "rejected_reasons": sorted(set(str(reason) for reason in reasons)),
            "validator_accepted": bool(validation.get("accepted", False)),
            "validator_bypassed": False,
            "descriptor_source": "slot_pool_coverage_sensor",
            "family": "slot_pool_coverage",
            "mechanic_state_type": str(
                validation.get("mechanic_state_type", "") or "slot_pool_coverage"
            ),
            "goal_link_type": "coverage_deficit_reduction",
            "protected_probe": True,
            "direct_llm_action_allowed": False,
            "candidate_generation_owner": "eca_verifier",
            "plan": [],
            "plan_length": 0,
            "slot_fill_count": 0,
            "expected_score_delta": 0,
        }
    )


def _packet_support_key(
    packet: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> str:
    return "|".join(
        [
            str(packet.get("game_id", "") or "unknown"),
            str(validation.get("mechanic_state_type", "") or "slot_pool_coverage"),
        ]
    )


def _posterior_support_key(packet: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(packet.get("game_id", "") or "unknown"),
            str(packet.get("mechanic_state_type", "") or "slot_pool_coverage"),
        ]
    )


def _slot_pool_posterior_slot_scope(
    before_packet: Mapping[str, Any],
    after_packet: Mapping[str, Any],
) -> dict[str, Any]:
    before_tokens = _slot_pool_active_slot_tokens(before_packet)
    after_tokens = _slot_pool_active_slot_tokens(after_packet)
    before_set = set(before_tokens)
    after_set = set(after_tokens)
    resolved = sorted(before_set - after_set)
    introduced = sorted(after_set - before_set)
    return _sanitize(
        {
            "schema_version": "eca_slot_pool_posterior_slot_scope.v1",
            "signature_kind": "active_slot_tokens",
            "before_active_slot_signature": _slot_token_signature(before_tokens),
            "after_active_slot_signature": _slot_token_signature(after_tokens),
            "resolved_slot_signature": _slot_token_signature(resolved),
            "introduced_slot_signature": _slot_token_signature(introduced),
            "before_active_slot_count": len(before_tokens),
            "after_active_slot_count": len(after_tokens),
            "resolved_slot_count": len(resolved),
            "introduced_slot_count": len(introduced),
            "before_active_slot_tokens": before_tokens[:64],
            "after_active_slot_tokens": after_tokens[:64],
            "resolved_slot_tokens": resolved[:64],
            "introduced_slot_tokens": introduced[:64],
        }
    )


def _slot_pool_active_slot_tokens(packet: Mapping[str, Any]) -> list[str]:
    coverage = packet.get("coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    kind = str(coverage.get("kind", "") or "")
    if kind == "reference_canvas_match":
        tokens = []
        for cell in list(coverage.get("deficit_cells") or []):
            if not isinstance(cell, Sequence) or len(cell) < 3:
                continue
            x, y, expected = (_safe_int(cell[idx]) for idx in range(3))
            tokens.append(f"cell:{x},{y}:expected:{expected}")
        return sorted(set(tokens))

    tokens = []
    for index, slot in enumerate(list(packet.get("slots") or [])):
        if not isinstance(slot, Mapping):
            continue
        if bool(slot.get("filled")):
            continue
        token = _slot_token_from_slot(slot, index=index)
        if token:
            tokens.append(token)
    return sorted(set(tokens))


def _slot_token_from_slot(slot: Mapping[str, Any], *, index: int) -> str:
    expected = slot.get("expected_color")
    site = slot.get("site_grid")
    if not _is_xy(site):
        site = slot.get("site")
    bbox = slot.get("bbox_grid")
    if not isinstance(bbox, Sequence):
        bbox = slot.get("bbox")
    raw_id = str(slot.get("id", "") or f"slot:{index}")
    parts = [raw_id]
    if expected is not None:
        parts.append(f"expected:{_safe_int(expected)}")
    if _is_xy(site):
        parts.append(f"site:{_safe_int(site[0])},{_safe_int(site[1])}")
    elif isinstance(bbox, Sequence) and len(bbox) >= 4:
        parts.append(
            "bbox:"
            f"{_safe_int(bbox[0])},{_safe_int(bbox[1])},"
            f"{_safe_int(bbox[2])},{_safe_int(bbox[3])}"
        )
    return "|".join(parts)


def _slot_token_signature(tokens: Sequence[str]) -> str:
    rows = sorted(str(token) for token in tokens if str(token))
    if not rows:
        return "empty"
    digest = hashlib.sha1("|".join(rows).encode("utf-8")).hexdigest()[:12]
    return f"slot_sig:{len(rows)}:{digest}"


def _slot_pool_region_identities(tokens: Sequence[str]) -> list[str]:
    cell_groups: dict[int, set[tuple[int, int]]] = {}
    other_tokens: set[str] = set()
    for token in tokens:
        parsed = _parse_slot_cell_token(str(token))
        if parsed is None:
            if str(token):
                other_tokens.add(str(token))
            continue
        x, y, expected = parsed
        cell_groups.setdefault(int(expected), set()).add((int(x), int(y)))

    identities: list[str] = []
    for expected, cells in sorted(cell_groups.items()):
        for component in _connected_cell_components(cells):
            xs = [cell[0] for cell in component]
            ys = [cell[1] for cell in component]
            identities.append(
                "cell_region:"
                f"expected:{int(expected)}:"
                f"bbox:{min(xs)},{min(ys)},{max(xs)},{max(ys)}:"
                f"area:{len(component)}"
            )
    for token in sorted(other_tokens):
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
        identities.append(f"slot_token:{digest}:{token}")
    return sorted(set(identities))


def _slot_pool_scope_tokens(
    scope: Mapping[str, Any],
    *,
    resolved_first: bool,
) -> list[str]:
    first_key = "resolved_slot_tokens" if resolved_first else "before_active_slot_tokens"
    second_key = "before_active_slot_tokens" if resolved_first else "resolved_slot_tokens"
    first = [str(token) for token in list(scope.get(first_key) or []) if str(token)]
    if first:
        return first
    return [str(token) for token in list(scope.get(second_key) or []) if str(token)]


def _slot_pool_effect_intent(packet: Mapping[str, Any]) -> dict[str, Any]:
    posterior = str(packet.get("posterior", "") or "unknown")
    score_delta = _safe_int(packet.get("score_delta"))
    deltas = packet.get("deltas")
    deltas = deltas if isinstance(deltas, Mapping) else {}
    before = packet.get("before")
    before = before if isinstance(before, Mapping) else {}
    after = packet.get("after")
    after = after if isinstance(after, Mapping) else {}
    deficit_delta = _safe_int(deltas.get("coverage_deficit_delta"))
    filled_delta = _safe_int(deltas.get("filled_slot_delta"))
    before_deficit = _safe_int(before.get("coverage_deficit_cell_count"))
    after_deficit = _safe_int(after.get("coverage_deficit_cell_count"))
    after_filled = _safe_int(after.get("filled_slot_count"))
    after_slot_count = _safe_int(after.get("slot_count"))

    if score_delta > 0:
        effect_intent = "score_closure"
    elif posterior == "confirmed" and bool(packet.get("scoreless_objective_progress")):
        terminal_deficit_closed = bool(
            before_deficit > 0 and after_deficit == 0 and deficit_delta < 0
        )
        terminal_slots_filled = bool(
            after_slot_count > 0 and after_filled >= after_slot_count and filled_delta > 0
        )
        if terminal_deficit_closed or terminal_slots_filled:
            effect_intent = "terminal_scoreless_slot_closure"
        elif deficit_delta < 0 and filled_delta > 0:
            effect_intent = "partial_deficit_and_slot_fill"
        elif deficit_delta < 0:
            effect_intent = "partial_deficit_reduction"
        elif filled_delta > 0:
            effect_intent = "partial_slot_fill"
        else:
            effect_intent = "scoreless_objective_progress_unclassified"
    elif posterior == "ambiguous":
        effect_intent = "ambiguous_no_progress_control"
    elif posterior == "falsified":
        effect_intent = "falsified_no_progress_control"
    elif posterior == "rejected":
        effect_intent = "rejected_descriptor"
    else:
        effect_intent = "insufficient_evidence"

    return {
        "effect_intent": effect_intent,
        "coverage_deficit_delta": int(deficit_delta),
        "filled_slot_delta": int(filled_delta),
        "before_deficit": int(before_deficit),
        "after_deficit": int(after_deficit),
        "after_filled_slot_count": int(after_filled),
        "after_slot_count": int(after_slot_count),
    }


def _slot_pool_intervention_signature(row: Mapping[str, Any]) -> str:
    parts = [
        str(row.get("mechanic_key", "") or "unknown"),
        f"action:{str(row.get('action_name', '') or '')}",
        f"site:{str(row.get('action_site_signature', '') or '')}",
        f"before:{str(row.get('before_active_slot_signature', '') or '')}",
    ]
    return "|".join(parts)


def _slot_pool_action_site_signature(packet: Mapping[str, Any]) -> str:
    payload = packet.get("action_payload")
    payload = payload if isinstance(payload, Mapping) else {}
    for key in (
        "selected_site",
        "target_site",
        "click_site",
        "site",
        "source_site",
        "destination_site",
    ):
        site = payload.get(key)
        if _is_xy(site):
            return f"{_safe_int(site[0])},{_safe_int(site[1])}"
    if "x" in payload and "y" in payload:
        return f"{_safe_int(payload.get('x'))},{_safe_int(payload.get('y'))}"
    if "col" in payload and "row" in payload:
        return f"{_safe_int(payload.get('col'))},{_safe_int(payload.get('row'))}"
    return ""


def _parse_slot_cell_token(token: str) -> tuple[int, int, int] | None:
    text = str(token or "")
    if not text.startswith("cell:"):
        return None
    parts = text.split(":")
    if len(parts) < 4 or parts[2] != "expected":
        return None
    xy = parts[1].split(",")
    if len(xy) != 2:
        return None
    try:
        return int(xy[0]), int(xy[1]), int(parts[3])
    except ValueError:
        return None


def _connected_cell_components(
    cells: set[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    components: list[list[tuple[int, int]]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = [start]
        while stack:
            x, y = stack.pop()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                stack.append(neighbor)
                component.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda rows: (rows[0][0], rows[0][1], len(rows)))
    return components


def _slot_pool_candidate_key(
    packet: Mapping[str, Any],
    validation: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> str:
    return (
        "slot_pool_protected_probe:"
        f"{str(packet.get('game_id', '') or 'unknown')}:"
        f"{str(validation.get('mechanic_state_type', '') or 'slot_pool_coverage')}:"
        f"step{_safe_int(packet.get('step'))}:"
        f"deficit{_safe_int(packet.get('coverage_deficit_cell_count'))}:"
        f"plan{_safe_int(proposal.get('plan_length'))}"
    )


def _build_sb26_packet(arr: np.ndarray) -> dict[str, Any]:
    from arcgym.perception.sb26_perception import perceive_top_sequence

    top_seq = list(perceive_top_sequence(arr))
    components = _connected_components(arr)
    slot_components = [
        component
        for component in components
        if component["color"] == 2
        and component["area"] <= 16
        and 8 <= int(component["centroid"][1]) <= 52
    ]
    filled_slot_components = [
        component
        for component in components
        if int(component["color"]) in set(top_seq)
        and 8 <= int(component["area"]) <= 40
        and 24 <= int(component["centroid"][1]) <= 36
    ]
    slot_like_components = sorted(
        [
            *(
                {**component, "filled": False, "observed_color": None}
                for component in slot_components
            ),
            *(
                {
                    **component,
                    "filled": True,
                    "observed_color": int(component["color"]),
                }
                for component in filled_slot_components
            ),
        ],
        key=lambda row: (row["centroid"][1], row["centroid"][0], row["color"]),
    )
    if len(slot_like_components) >= len(top_seq) > 0:
        slot_components_for_sequence = sorted(
            slot_like_components[: len(top_seq)],
            key=lambda row: (row["centroid"][0], row["centroid"][1], row["color"]),
        )
    else:
        slot_components_for_sequence = sorted(
            (
                {**component, "filled": False, "observed_color": None}
                for component in slot_components
            ),
            key=lambda row: (row["centroid"][1], row["centroid"][0]),
        )
    slots = [
        {
            "id": f"slot:{index}",
            "site": list(component["centroid"]),
            "bbox": list(component["bbox"]),
            "expected_color": int(top_seq[index]) if index < len(top_seq) else None,
            "observed_color": component.get("observed_color"),
            "filled": bool(
                component.get("filled")
                and index < len(top_seq)
                and component.get("observed_color") == int(top_seq[index])
            ),
        }
        for index, component in enumerate(slot_components_for_sequence)
    ]
    pool_objects = [
        {
            "id": f"pool:{index}",
            "color": int(component["color"]),
            "bbox": list(component["bbox"]),
            "centroid": list(component["centroid"]),
        }
        for index, component in enumerate(
            component
            for component in components
            if component["color"] in set(top_seq)
            and int(component["bbox"][1]) >= 52
            and component["area"] >= 8
        )
    ]
    filled_slots = sum(1 for slot in slots if bool(slot.get("filled")))
    unfilled_slots = max(0, len(slots) - filled_slots)
    binding_count = 0
    ambiguous = 0
    pool_color_counts: dict[int, int] = {}
    for obj in pool_objects:
        color = int(obj["color"])
        pool_color_counts[color] = pool_color_counts.get(color, 0) + 1
    for slot in slots:
        if bool(slot.get("filled")):
            continue
        expected = slot.get("expected_color")
        if expected is None:
            continue
        matches = int(pool_color_counts.get(int(expected), 0))
        binding_count += matches
        if matches > 1:
            ambiguous += 1
    return _base_packet(
        slot_count=len(slots),
        filled_slot_count=filled_slots,
        pool_object_count=len(pool_objects),
        slot_candidate_binding_count=binding_count,
        coverage_required_cell_count=len(top_seq),
        coverage_observed_cell_count=filled_slots,
        coverage={"kind": "slot_sequence", "top_sequence_len": len(top_seq)},
        slots=slots,
        pool_objects=pool_objects,
        ambiguous_binding_count=ambiguous,
        perception_summary={
            "level_family": "sb26_slot_sequence",
            "top_sequence": top_seq,
            "top_sequence_len": len(top_seq),
        },
    )


def _count_sb26_filled_slots(
    slots: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    top_colors: set[int],
) -> int:
    filled = 0
    for slot in slots:
        site = slot.get("site")
        if not _is_xy(site):
            continue
        sx, sy = int(site[0]), int(site[1])
        for component in components:
            if int(component.get("color", -1)) not in top_colors:
                continue
            bbox = component.get("bbox")
            if not _is_bbox(bbox):
                continue
            x0, y0, x1, y1 = (int(v) for v in bbox)
            if y0 >= 52:
                continue
            if x0 - 2 <= sx <= x1 + 2 and y0 - 2 <= sy <= y1 + 2:
                filled += 1
                break
    return filled


def _build_cd82_packet(arr: np.ndarray) -> dict[str, Any]:
    from arcgym.perception.cd82_perception import perceive_cd82

    reference = arr[3:13, 3:13]
    canvas = arr[34:44, 27:37]
    mask = np.ones((10, 10), dtype=bool)
    for index in range(10):
        mask[index, index] = False
        mask[index, 9 - index] = False
    required = int(mask.sum())
    observed = int(((reference == canvas) & mask).sum()) if reference.shape == canvas.shape else 0
    palette = sorted(int(color) for color in np.unique(reference) if int(color) >= 0)
    deficit_cells = [
        [int(x), int(y), int(reference[y, x]), int(canvas[y, x])]
        for y in range(10)
        for x in range(10)
        if bool(mask[y, x]) and int(reference[y, x]) != int(canvas[y, x])
    ]
    slots = [
        {
            "id": f"cell:{x},{y}",
            "site": [27 + int(x), 34 + int(y)],
            "expected_color": int(reference[y, x]),
            "observed_color": int(canvas[y, x]),
            "filled": bool(int(reference[y, x]) == int(canvas[y, x])),
        }
        for y in range(10)
        for x in range(10)
        if bool(mask[y, x])
    ]
    return _base_packet(
        slot_count=required,
        filled_slot_count=observed,
        pool_object_count=len(palette),
        slot_candidate_binding_count=len(deficit_cells),
        coverage_required_cell_count=required,
        coverage_observed_cell_count=observed,
        slots=slots[:16],
        pool_objects=[{"id": f"paint:{color}", "color": int(color)} for color in palette],
        pool_reusable=True,
        coverage={
            "kind": "reference_canvas_match",
            "reference_bbox": [3, 3, 12, 12],
            "canvas_bbox": [27, 34, 36, 43],
            "diagonal_cells_ignored": 20,
            "reference_rows": [[int(v) for v in row] for row in reference.tolist()],
            "canvas_rows": [[int(v) for v in row] for row in canvas.tolist()],
            "palette": palette,
            "deficit_cells": deficit_cells,
            "deficit_cells_sample": deficit_cells[:16],
            "swatches": _cd82_swatch_sites_from_palette(palette),
            "arrow_sites": _cd82_arrow_sites(),
            "current_basket_idx": 0,
            "current_paint_color": 15,
            "reset_state_assumption": True,
        },
        perception_summary=perceive_cd82(arr),
    )


def _build_cd82_reference_canvas_protected_proposal(
    packet: Mapping[str, Any],
    *,
    validation: Mapping[str, Any],
    max_plan_steps: int,
) -> dict[str, Any]:
    coverage = packet.get("coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    reasons: list[str] = []
    reference = _rows_to_array(coverage.get("reference_rows"))
    canvas = _rows_to_array(coverage.get("canvas_rows"))
    if reference is None or reference.shape != (10, 10):
        reasons.append("missing_reference_rows")
    if canvas is None or canvas.shape != (10, 10):
        reasons.append("missing_canvas_rows")
    swatches = _cd82_swatch_map(coverage.get("swatches"))
    if not swatches:
        reasons.append("missing_swatches")
    if reasons:
        return _slot_pool_rejected_proposal(validation, reasons)
    assert reference is not None
    assert canvas is not None
    operations = _cd82_region_cover_operations(
        reference,
        canvas,
        colors=sorted(swatches),
    )
    if not operations:
        reasons.append("no_region_cover_plan")
        return _slot_pool_rejected_proposal(validation, reasons)
    action_steps = _cd82_actions_from_region_operations(
        operations,
        swatches=swatches,
        arrow_sites=_cd82_arrow_map(coverage.get("arrow_sites")),
        current_basket_idx=_safe_int(coverage.get("current_basket_idx"), default=0),
        current_paint_color=_safe_int(coverage.get("current_paint_color"), default=15),
    )
    if not action_steps:
        reasons.append("empty_region_cover_action_plan")
        return _slot_pool_rejected_proposal(validation, reasons)
    if len(action_steps) > int(max_plan_steps):
        reasons.append("proposal_plan_too_long")
        return _slot_pool_rejected_proposal(validation, reasons)
    plan = [
        {
            "action_name": action,
            "payload": dict(payload),
            "role": role,
        }
        for action, payload, role in action_steps
    ]
    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_PROTECTED_PROPOSAL_SCHEMA_VERSION,
            "accepted": True,
            "rejected_reasons": [],
            "validator_accepted": bool(validation.get("accepted", False)),
            "validator_bypassed": False,
            "descriptor_source": "slot_pool_coverage_sensor",
            "family": "slot_pool_coverage",
            "mechanic_state_type": "reference_canvas_match",
            "goal_link_type": "coverage_deficit_reduction",
            "protected_probe": True,
            "direct_llm_action_allowed": False,
            "candidate_generation_owner": "eca_verifier",
            "plan": plan,
            "plan_length": len(plan),
            "slot_fill_count": len(operations),
            "coverage_operation_count": len(operations),
            "coverage_operations": [
                {"region": region, "color": int(color)}
                for region, color in operations
            ],
            "expected_score_delta": 1,
        }
    )


def _cd82_swatch_sites_from_palette(palette: Sequence[int]) -> list[dict[str, Any]]:
    colors = sorted(int(color) for color in palette)
    if set(colors).issubset({0, 15}) and len(colors) <= 2:
        sites = {0: [37, 4], 15: [43, 4]}
    elif set(colors).issubset({0, 12, 15}) and len(colors) <= 3:
        sites = {0: [34, 4], 15: [40, 4], 12: [46, 4]}
    else:
        sites = {
            0: [23, 4],
            15: [29, 4],
            12: [35, 4],
            11: [41, 4],
            14: [47, 4],
            8: [53, 4],
            9: [59, 4],
        }
    return [
        {"color": int(color), "site": list(sites[color])}
        for color in colors
        if color in sites
    ]


def _cd82_arrow_sites() -> list[dict[str, Any]]:
    return [
        {"region": "arrow_n", "basket_idx": 0, "site": [32, 20]},
        {"region": "arrow_w", "basket_idx": 6, "site": [13, 39]},
        {"region": "arrow_e", "basket_idx": 2, "site": [51, 39]},
        {"region": "arrow_s", "basket_idx": 4, "site": [32, 55]},
    ]


def _rows_to_array(rows: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(rows, dtype=np.int16)
    except Exception:
        return None
    if arr.shape != (10, 10):
        return None
    return arr


def _cd82_swatch_map(raw_swatches: Any) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    if not isinstance(raw_swatches, Sequence):
        return out
    for item in raw_swatches:
        if not isinstance(item, Mapping):
            continue
        site = item.get("site")
        if not _is_xy(site):
            continue
        color = _safe_int(item.get("color"), default=-1)
        out[int(color)] = (int(site[0]), int(site[1]))
    return out


def _cd82_arrow_map(raw_arrows: Any) -> dict[str, tuple[int, tuple[int, int]]]:
    out: dict[str, tuple[int, tuple[int, int]]] = {}
    if not isinstance(raw_arrows, Sequence):
        return out
    for item in raw_arrows:
        if not isinstance(item, Mapping):
            continue
        site = item.get("site")
        if not _is_xy(site):
            continue
        region = str(item.get("region", "") or "")
        out[region] = (
            _safe_int(item.get("basket_idx"), default=0),
            (int(site[0]), int(site[1])),
        )
    return out


def _cd82_region_cover_operations(
    reference: np.ndarray,
    canvas: np.ndarray,
    *,
    colors: Sequence[int],
) -> list[tuple[str, int]]:
    mask = _cd82_goal_mask()
    if bool(np.array_equal(reference[mask], canvas[mask])):
        return []
    region_masks = _cd82_region_masks()
    operations = [
        (region, int(color), region_mask)
        for region, region_mask in region_masks.items()
        for color in sorted(int(color) for color in colors)
    ]
    beam: list[tuple[int, tuple[int, ...], np.ndarray, list[tuple[str, int]]]] = [
        (
            int(np.sum(canvas[mask] != reference[mask])),
            tuple(int(v) for v in canvas[mask].tolist()),
            np.array(canvas, copy=True),
            [],
        )
    ]
    seen: dict[tuple[int, ...], int] = {beam[0][1]: 0}
    for _depth in range(6):
        beam = sorted(beam, key=lambda row: (row[0], len(row[3])))[:2000]
        if beam and beam[0][0] == 0:
            return beam[0][3]
        next_beam: list[tuple[int, tuple[int, ...], np.ndarray, list[tuple[str, int]]]] = []
        for _mismatch, state_key, arr, path in beam:
            for region, color, region_mask in operations:
                new_arr = np.array(arr, copy=True)
                new_arr[region_mask] = int(color)
                new_key = tuple(int(v) for v in new_arr[mask].tolist())
                if new_key == state_key:
                    continue
                new_depth = len(path) + 1
                if seen.get(new_key, 999) <= new_depth:
                    continue
                seen[new_key] = new_depth
                next_beam.append(
                    (
                        int(np.sum(new_arr[mask] != reference[mask])),
                        new_key,
                        new_arr,
                        [*path, (region, int(color))],
                    )
                )
        beam = next_beam
        if not beam:
            break
    beam = sorted(beam, key=lambda row: (row[0], len(row[3])))
    if beam and beam[0][0] == 0:
        return beam[0][3]
    return []


def _cd82_actions_from_region_operations(
    operations: Sequence[tuple[str, int]],
    *,
    swatches: Mapping[int, tuple[int, int]],
    arrow_sites: Mapping[str, tuple[int, tuple[int, int]]],
    current_basket_idx: int,
    current_paint_color: int,
) -> list[tuple[str, dict[str, Any], str]]:
    basket_regions = {
        "north_half": 0,
        "north_east_triangle": 1,
        "east_half": 2,
        "south_east_triangle": 3,
        "south_half": 4,
        "south_west_triangle": 5,
        "west_half": 6,
        "north_west_triangle": 7,
    }
    arrow_regions = {
        "arrow_n",
        "arrow_w",
        "arrow_e",
        "arrow_s",
    }
    idx = int(current_basket_idx)
    paint = int(current_paint_color)
    plan: list[tuple[str, dict[str, Any], str]] = []
    for region, color in operations:
        if region in basket_regions:
            target_idx = basket_regions[region]
            arrow_site = None
        elif region in arrow_regions and region in arrow_sites:
            target_idx, arrow_site = arrow_sites[region]
        else:
            return []
        for action in _cd82_basket_nav_actions(idx, int(target_idx)):
            plan.append((action, {}, "move_basket"))
        idx = int(target_idx)
        if int(color) != paint:
            site = swatches.get(int(color))
            if site is None:
                return []
            plan.append(
                (
                    "ACTION6",
                    {"x": int(site[0]), "y": int(site[1])},
                    "select_paint_color",
                )
            )
            paint = int(color)
        if arrow_site is not None:
            plan.append(
                (
                    "ACTION6",
                    {"x": int(arrow_site[0]), "y": int(arrow_site[1])},
                    "paint_arrow_region",
                )
            )
        else:
            plan.append(("ACTION5", {}, "paint_basket_region"))
    return plan


def _cd82_goal_mask() -> np.ndarray:
    mask = np.ones((10, 10), dtype=bool)
    for index in range(10):
        mask[index, index] = False
        mask[index, 9 - index] = False
    return mask


def _cd82_region_masks() -> dict[str, np.ndarray]:
    mask = _cd82_goal_mask()
    out: dict[str, np.ndarray] = {}
    out["north_half"] = _cd82_mask_slice(mask, rows=range(0, 5), cols=range(10))
    out["south_half"] = _cd82_mask_slice(mask, rows=range(5, 10), cols=range(10))
    out["west_half"] = _cd82_mask_slice(mask, rows=range(10), cols=range(0, 5))
    out["east_half"] = _cd82_mask_slice(mask, rows=range(10), cols=range(5, 10))
    out["south_west_triangle"] = _cd82_triangle_mask(mask, lambda row: range(0, row + 1))
    out["south_east_triangle"] = _cd82_triangle_mask(mask, lambda row: range(9 - row, 10))
    out["north_east_triangle"] = _cd82_triangle_mask(mask, lambda row: range(row, 10))
    out["north_west_triangle"] = _cd82_triangle_mask(mask, lambda row: range(0, 10 - row))
    out["arrow_n"] = _cd82_mask_slice(mask, rows=range(0, 3), cols=range(3, 7))
    out["arrow_w"] = _cd82_mask_slice(mask, rows=range(3, 7), cols=range(0, 3))
    out["arrow_e"] = _cd82_mask_slice(mask, rows=range(3, 7), cols=range(7, 10))
    out["arrow_s"] = _cd82_mask_slice(mask, rows=range(7, 10), cols=range(3, 7))
    return out


def _cd82_mask_slice(mask: np.ndarray, *, rows: range, cols: range) -> np.ndarray:
    out = np.zeros((10, 10), dtype=bool)
    for y in rows:
        for x in cols:
            out[int(y), int(x)] = True
    return out & mask


def _cd82_triangle_mask(mask: np.ndarray, cols_for_row: Any) -> np.ndarray:
    out = np.zeros((10, 10), dtype=bool)
    for y in range(10):
        for x in cols_for_row(y):
            out[int(y), int(x)] = True
    return out & mask


def _cd82_basket_nav_actions(start_idx: int, target_idx: int) -> list[str]:
    if int(start_idx) == int(target_idx):
        return []
    positions = {
        0: (0, 1),
        1: (0, 2),
        2: (1, 2),
        3: (2, 2),
        4: (2, 1),
        5: (2, 0),
        6: (1, 0),
        7: (0, 0),
    }
    by_position = {position: idx for idx, position in positions.items()}
    queue: list[tuple[int, list[str]]] = [(int(start_idx), [])]
    seen = {int(start_idx)}
    moves = [
        ("ACTION1", (-1, 0)),
        ("ACTION2", (1, 0)),
        ("ACTION3", (0, -1)),
        ("ACTION4", (0, 1)),
    ]
    while queue:
        idx, path = queue.pop(0)
        if idx == int(target_idx):
            return path
        x, y = positions[idx]
        for action, (dx, dy) in moves:
            next_idx = by_position.get((x + dx, y + dy))
            if next_idx is None or next_idx in seen:
                continue
            seen.add(next_idx)
            queue.append((next_idx, [*path, action]))
    return []


def _build_sp80_packet(arr: np.ndarray) -> dict[str, Any]:
    from arcgym.perception.sp80_perception import perceive_sp80

    state = perceive_sp80(arr)
    bowls = list(state.get("bowls") or [])
    sources = list(state.get("sources") or [])
    filters = list(state.get("filters") or [])
    filled = _sp80_filled_bowl_count(arr, bowls, int(state.get("rotation_k", 0) or 0))
    slots = [
        {
            "id": f"bowl:{index}",
            "bbox_grid": [int(v) for v in bowl[:4]],
            "filled": bool(index < filled),
        }
        for index, bowl in enumerate(bowls)
    ]
    return _base_packet(
        slot_count=len(bowls),
        filled_slot_count=filled,
        pool_object_count=len(sources),
        slot_candidate_binding_count=len(sources) * len(bowls),
        coverage_required_cell_count=len(bowls),
        coverage_observed_cell_count=filled,
        slots=slots,
        pool_objects=[
            {"id": f"source:{index}", "site_grid": [int(source[0]), int(source[1])]}
            for index, source in enumerate(sources)
        ],
        pool_reusable=True,
        ambiguous_binding_count=1 if len(sources) > 1 or len(bowls) > 1 else 0,
        coverage={
            "kind": "sand_bowl_fill",
            "filter_count": len(filters),
            "filters": [[int(v) for v in row[:4]] + [bool(row[4])] for row in filters],
        },
        perception_summary={
            "level_signature": state.get("level_signature"),
            "rotation_k": int(state.get("rotation_k", 0) or 0),
        },
    )


def _sp80_filled_bowl_count(arr: np.ndarray, bowls: Sequence[Any], rotation_k: int) -> int:
    grid = np.rot90(arr, k=-int(rotation_k)).copy() if int(rotation_k) else arr
    filled = 0
    for bowl in bowls:
        if not isinstance(bowl, Sequence) or len(bowl) < 4:
            continue
        gx, gy, w, h = (int(v) for v in bowl[:4])
        block = grid[gy * 4 : (gy + h) * 4, gx * 4 : (gx + w) * 4]
        if bool((block == 13).any()):
            filled += 1
    return filled


def _build_sp80_sand_flow_protected_proposal(
    packet: Mapping[str, Any],
    *,
    validation: Mapping[str, Any],
    max_plan_steps: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    coverage = packet.get("coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    perception = packet.get("perception_summary")
    perception = perception if isinstance(perception, Mapping) else {}
    filters = _sp80_parse_filters(coverage.get("filters"))
    bowls = _sp80_parse_bowls(packet.get("slots"))
    sources = _sp80_parse_sources(packet.get("pool_objects"))
    rotation_k = _safe_int(perception.get("rotation_k"))

    if not filters:
        reasons.append("missing_filters")
    if not bowls:
        reasons.append("missing_bowls")
    if not sources:
        reasons.append("missing_sources")
    target_filters = (
        _sp80_solve_filter_targets(
            [(x, y, w) for x, y, w, _h, _selected in filters],
            sources,
            bowls,
        )
        if not reasons
        else None
    )
    if target_filters is None and not reasons:
        reasons.append("no_winning_filter_configuration")
    plan = (
        _sp80_emit_filter_plan(filters, list(target_filters or []), rotation_k)
        if target_filters is not None
        else []
    )
    if target_filters is not None and not plan:
        reasons.append("empty_sand_flow_plan")
    if len(plan) > int(max_plan_steps):
        reasons.append("proposal_plan_too_long")

    accepted = not reasons
    if not accepted:
        plan = []
    return _sanitize(
        {
            "schema_version": SLOT_POOL_COVERAGE_PROTECTED_PROPOSAL_SCHEMA_VERSION,
            "accepted": bool(accepted),
            "rejected_reasons": sorted(set(str(reason) for reason in reasons)),
            "validator_accepted": bool(validation.get("accepted", False)),
            "validator_bypassed": False,
            "descriptor_source": "slot_pool_coverage_sensor",
            "family": "slot_pool_coverage",
            "mechanic_state_type": "sand_bowl_fill",
            "goal_link_type": "bowl_fill_coverage",
            "protected_probe": True,
            "direct_llm_action_allowed": False,
            "candidate_generation_owner": "eca_verifier",
            "plan": plan,
            "plan_length": len(plan),
            "slot_fill_count": len(bowls) if accepted else 0,
            "expected_score_delta": 1 if accepted else 0,
            "sand_flow": {
                "source_count": len(sources),
                "bowl_count": len(bowls),
                "filter_count": len(filters),
                "rotation_k": int(rotation_k),
                "initial_filters": [
                    [int(x), int(y), int(w)] for x, y, w, _h, _selected in filters
                ],
                "target_filters": [
                    [int(x), int(y), int(w)] for x, y, w in list(target_filters or [])
                ],
            },
        }
    )


def _sp80_parse_filters(raw_filters: Any) -> list[tuple[int, int, int, int, bool]]:
    filters: list[tuple[int, int, int, int, bool]] = []
    for raw in list(raw_filters or []):
        if not isinstance(raw, Sequence) or len(raw) < 5:
            continue
        x, y, w, h = (_safe_int(raw[idx]) for idx in range(4))
        if w <= 0 or h <= 0:
            continue
        filters.append((int(x), int(y), int(w), int(h), bool(raw[4])))
    return filters


def _sp80_parse_bowls(raw_slots: Any) -> list[tuple[int, int, int, int]]:
    bowls: list[tuple[int, int, int, int]] = []
    for slot in list(raw_slots or []):
        if not isinstance(slot, Mapping):
            continue
        bbox = slot.get("bbox_grid")
        if not isinstance(bbox, Sequence) or len(bbox) < 4:
            continue
        x, y, w, h = (_safe_int(bbox[idx]) for idx in range(4))
        if w <= 0 or h <= 0:
            continue
        bowls.append((int(x), int(y), int(w), int(h)))
    return bowls


def _sp80_parse_sources(raw_pool_objects: Any) -> list[tuple[int, int]]:
    sources: list[tuple[int, int]] = []
    for obj in list(raw_pool_objects or []):
        if not isinstance(obj, Mapping):
            continue
        site = obj.get("site_grid")
        if _is_xy(site):
            sources.append((int(site[0]), int(site[1])))
    return sources


def _sp80_solve_filter_targets(
    init_filters: Sequence[tuple[int, int, int]],
    sources: Sequence[tuple[int, int]],
    bowls: Sequence[tuple[int, int, int, int]],
    *,
    grid_w: int = 16,
    grid_h: int = 16,
    y_min: int = 3,
    y_max: int = 12,
    max_evaluations: int = 60000,
) -> list[tuple[int, int, int]] | None:
    filters = [(int(x), int(y), int(w)) for x, y, w in init_filters]
    if _sp80_sand_flow_wins(filters, bowls, sources, grid_w=grid_w, grid_h=grid_h):
        return list(filters)

    targets_per_filter: list[list[tuple[int, int, int]]] = []
    for ix, iy, iw in filters:
        candidates: list[tuple[int, int, int]] = []
        for x in range(0, grid_w - int(iw) + 1):
            for y in range(int(y_min), int(y_max) + 1):
                candidates.append((int(x), int(y), int(iw)))
        candidates.sort(key=lambda row: abs(row[0] - int(ix)) + abs(row[1] - int(iy)))
        targets_per_filter.append(candidates)

    start_state = tuple(0 for _ in filters)
    heap: list[tuple[int, tuple[int, ...]]] = [
        (_sp80_target_state_distance(filters, targets_per_filter, start_state), start_state)
    ]
    seen = {start_state}
    evaluations = 0
    while heap and evaluations < int(max_evaluations):
        _dist, state = heapq.heappop(heap)
        target_filters = [targets_per_filter[index][state[index]] for index in range(len(filters))]
        evaluations += 1
        if not _sp80_filters_overlap(target_filters) and _sp80_sand_flow_wins(
            target_filters,
            bowls,
            sources,
            grid_w=grid_w,
            grid_h=grid_h,
        ):
            return target_filters
        for index in range(len(filters)):
            if state[index] + 1 >= len(targets_per_filter[index]):
                continue
            next_state = list(state)
            next_state[index] += 1
            next_tuple = tuple(next_state)
            if next_tuple in seen:
                continue
            seen.add(next_tuple)
            heapq.heappush(
                heap,
                (
                    _sp80_target_state_distance(filters, targets_per_filter, next_tuple),
                    next_tuple,
                ),
            )
    return None


def _sp80_target_state_distance(
    init_filters: Sequence[tuple[int, int, int]],
    targets_per_filter: Sequence[Sequence[tuple[int, int, int]]],
    state: Sequence[int],
) -> int:
    dist = 0
    for index, target_index in enumerate(state):
        ix, iy, _iw = init_filters[index]
        tx, ty, _tw = targets_per_filter[index][target_index]
        dist += abs(int(tx) - int(ix)) + abs(int(ty) - int(iy))
    return int(dist)


def _sp80_filters_overlap(filters: Sequence[tuple[int, int, int]]) -> bool:
    rows = [(int(x), int(y), int(w)) for x, y, w in filters]
    for left_index, (left_x, left_y, left_w) in enumerate(rows):
        for right_x, right_y, right_w in rows[left_index + 1 :]:
            if left_y == right_y and left_x + left_w > right_x and right_x + right_w > left_x:
                return True
    return False


def _sp80_sand_flow_wins(
    filters: Sequence[tuple[int, int, int]],
    bowls: Sequence[tuple[int, int, int, int]],
    sources: Sequence[tuple[int, int]],
    *,
    grid_w: int,
    grid_h: int,
    max_steps: int = 200,
) -> bool:
    water: set[tuple[int, int]] = set()
    particles: list[tuple[int, int, int, int]] = []
    filled_bowls: set[int] = set()
    for sx, sy in sources:
        particle = (int(sx), int(sy) + 1, 0, 1)
        particles.append(particle)
        water.add((particle[0], particle[1]))

    def sprite_at(x: int, y: int) -> tuple[str, int] | None:
        if (x, y) in water:
            return ("water", -1)
        for index, (fx, fy, fw) in enumerate(filters):
            if int(fx) <= x < int(fx) + int(fw) and int(fy) == y:
                return ("filter", index)
        for index, (bx, by, bw, bh) in enumerate(bowls):
            if int(bx) <= x < int(bx) + int(bw) and int(by) <= y < int(by) + int(bh):
                local_x, local_y = x - int(bx), y - int(by)
                if int(bw) == 3 and int(bh) == 2 and (local_x, local_y) == (1, 0):
                    return None
                return ("bowl", index)
        if y == int(grid_h) - 1:
            return ("floor", -1)
        if x < 0 or x >= int(grid_w) or y < 0 or y >= int(grid_h):
            return ("wall", -1)
        return None

    overflow = False
    for _step in range(int(max_steps)):
        if not particles:
            break
        next_particles: list[tuple[int, int, int, int]] = []
        for px, py, dx, dy in particles:
            tx, ty = int(px) + int(dx), int(py) + int(dy)
            target = sprite_at(tx, ty)
            perpendicular = ((-1, 0), (1, 0)) if int(dy) != 0 else ((0, -1), (0, 1))
            if target is None:
                if (tx, ty) not in water:
                    water.add((tx, ty))
                    next_particles.append((tx, ty, int(dx), int(dy)))
                continue
            kind = target[0]
            if kind == "water":
                next_particles.append((tx, ty, int(dx), int(dy)))
                continue
            if kind == "filter":
                for ox, oy in perpendicular:
                    nxt = (int(px) + int(ox), int(py) + int(oy))
                    if sprite_at(*nxt) is None and nxt not in water:
                        water.add(nxt)
                        next_particles.append((nxt[0], nxt[1], int(dx), int(dy)))
                continue
            if kind == "bowl":
                bowl_index = int(target[1])
                left = sprite_at(int(px) + perpendicular[0][0], int(py) + perpendicular[0][1])
                right = sprite_at(int(px) + perpendicular[1][0], int(py) + perpendicular[1][1])
                if (
                    left is not None
                    and left[0] == "bowl"
                    and int(left[1]) == bowl_index
                    and right is not None
                    and right[0] == "bowl"
                    and int(right[1]) == bowl_index
                ):
                    filled_bowls.add(bowl_index)
                else:
                    for ox, oy in perpendicular:
                        nxt = (int(px) + int(ox), int(py) + int(oy))
                        if sprite_at(*nxt) is None and nxt not in water:
                            water.add(nxt)
                            next_particles.append((nxt[0], nxt[1], int(dx), int(dy)))
                continue
            if kind == "floor":
                overflow = True
        particles = next_particles
    return bool(not overflow and len(filled_bowls) == len(list(bowls)))


def _sp80_emit_filter_plan(
    initial_filters: Sequence[tuple[int, int, int, int, bool]],
    target_filters: Sequence[tuple[int, int, int]],
    rotation_k: int,
) -> list[dict[str, Any]]:
    if len(initial_filters) != len(target_filters):
        return []
    plan: list[dict[str, Any]] = []
    current = [(int(x), int(y), int(w)) for x, y, w, _h, _selected in initial_filters]
    selected_index = next(
        (index for index, row in enumerate(initial_filters) if bool(row[4])),
        None,
    )
    for index, (target_x, target_y, target_w) in enumerate(target_filters):
        current_x, current_y, current_w = current[index]
        if int(current_w) != int(target_w):
            return []
        if (current_x, current_y) != (int(target_x), int(target_y)) and selected_index != index:
            click_x, click_y = _sp80_click_for_grid(current_x, current_y, rotation_k)
            plan.append(
                {
                    "action_name": "ACTION6",
                    "payload": {"x": int(click_x), "y": int(click_y)},
                    "role": "select_filter",
                    "filter_index": index,
                }
            )
            selected_index = index
        dx_total = int(target_x) - int(current_x)
        dy_total = int(target_y) - int(current_y)
        step_x = 1 if dx_total > 0 else -1
        step_y = 1 if dy_total > 0 else -1
        for _ in range(abs(dx_total)):
            action_name = _sp80_action_for_delta(step_x, 0, rotation_k)
            if not action_name:
                return []
            plan.append(
                {
                    "action_name": action_name,
                    "payload": {},
                    "role": "move_filter",
                    "filter_index": index,
                }
            )
        for _ in range(abs(dy_total)):
            action_name = _sp80_action_for_delta(0, step_y, rotation_k)
            if not action_name:
                return []
            plan.append(
                {
                    "action_name": action_name,
                    "payload": {},
                    "role": "move_filter",
                    "filter_index": index,
                }
            )
        current[index] = (int(target_x), int(target_y), int(current_w))
    plan.append({"action_name": "ACTION5", "payload": {}, "role": "spill"})
    return plan


def _sp80_action_for_delta(dx: int, dy: int, rotation_k: int) -> str:
    delta = (int(dx), int(dy))
    if int(rotation_k) == 0:
        return {
            (0, -1): "ACTION1",
            (0, 1): "ACTION2",
            (-1, 0): "ACTION3",
            (1, 0): "ACTION4",
        }.get(delta, "")
    if int(rotation_k) == 2:
        return {
            (0, 1): "ACTION1",
            (0, -1): "ACTION2",
            (1, 0): "ACTION3",
            (-1, 0): "ACTION4",
        }.get(delta, "")
    return ""


def _sp80_click_for_grid(gx: int, gy: int, rotation_k: int, cell_size: int = 4) -> tuple[int, int]:
    if int(rotation_k) == 2:
        return (62 - int(cell_size) * int(gx), 62 - int(cell_size) * int(gy))
    return (int(gx) * int(cell_size) + 2, int(gy) * int(cell_size) + 2)


def _build_ar25_packet(arr: np.ndarray) -> dict[str, Any]:
    from arcgym.perception.ar25_perception import perceive_ar25

    state = perceive_ar25(arr)
    targets = {tuple(int(v) for v in target) for target in list(state.get("targets") or [])}
    cursor_cells: set[tuple[int, int]] = set()
    for cursor in list(state.get("cursors") or []):
        pos = cursor.get("pos")
        if not _is_xy(pos):
            continue
        for pixel in list(cursor.get("pixels") or []):
            if _is_xy(pixel):
                cursor_cells.add((int(pos[0]) + int(pixel[0]), int(pos[1]) + int(pixel[1])))
    covered = sorted(targets & cursor_cells)
    return _base_packet(
        slot_count=len(targets),
        filled_slot_count=len(covered),
        pool_object_count=len(list(state.get("cursors") or [])),
        slot_candidate_binding_count=len(targets) * max(1, len(list(state.get("axes") or []))),
        coverage_required_cell_count=len(targets),
        coverage_observed_cell_count=len(covered),
        slots=[{"id": f"target:{x},{y}", "site_grid": [x, y]} for x, y in sorted(targets)[:24]],
        pool_objects=[
            {
                "id": f"cursor:{index}",
                "pos_grid": list(cursor.get("pos") or []),
                "pixel_count": len(list(cursor.get("pixels") or [])),
            }
            for index, cursor in enumerate(list(state.get("cursors") or []))
        ],
        pool_reusable=True,
        ambiguous_binding_count=max(0, len(list(state.get("axes") or [])) - 1),
        coverage={
            "kind": "axis_reflection_target_coverage",
            "covered_targets": [list(cell) for cell in covered],
            "axis_count": len(list(state.get("axes") or [])),
        },
        perception_summary={"level_signature": state.get("level_signature")},
    )


def _build_re86_packet(arr: np.ndarray) -> dict[str, Any]:
    from arcgym.perception.re86_perception import perceive_re86

    components = _connected_components(arr)
    target_markers = [
        component
        for component in components
        if int(component["color"]) == 11 and int(component["area"]) <= 4
    ]
    pool_objects = [
        component
        for component in components
        if int(component["color"]) not in {5, 11, 15}
        and int(component["area"]) >= 4
    ]
    return _base_packet(
        slot_count=len(target_markers),
        filled_slot_count=0,
        pool_object_count=len(pool_objects),
        slot_candidate_binding_count=len(target_markers) * max(1, len(pool_objects)),
        coverage_required_cell_count=len(target_markers),
        coverage_observed_cell_count=0,
        slots=[
            {"id": f"target:{index}", "site": list(component["centroid"]), "bbox": list(component["bbox"])}
            for index, component in enumerate(target_markers)
        ],
        pool_objects=[
            {
                "id": f"piece_like:{index}",
                "color": int(component["color"]),
                "bbox": list(component["bbox"]),
                "area": int(component["area"]),
            }
            for index, component in enumerate(pool_objects[:16])
        ],
        coverage={"kind": "shape_target_marker_coverage"},
        perception_summary=perceive_re86(arr),
    )


def _build_generic_packet(arr: np.ndarray) -> dict[str, Any]:
    components = [
        component for component in _connected_components(arr) if int(component["area"]) >= 4
    ]
    return _base_packet(
        slot_count=0,
        filled_slot_count=0,
        pool_object_count=len(components),
        slot_candidate_binding_count=0,
        coverage_required_cell_count=0,
        coverage_observed_cell_count=0,
        slots=[],
        pool_objects=[
            {
                "id": f"component:{index}",
                "color": int(component["color"]),
                "bbox": list(component["bbox"]),
                "area": int(component["area"]),
            }
            for index, component in enumerate(components[:16])
        ],
        coverage={"kind": "generic_components"},
        perception_summary={},
    )


def _base_packet(
    *,
    slot_count: int,
    filled_slot_count: int,
    pool_object_count: int,
    slot_candidate_binding_count: int,
    coverage_required_cell_count: int,
    coverage_observed_cell_count: int,
    slots: Sequence[Mapping[str, Any]],
    pool_objects: Sequence[Mapping[str, Any]],
    ambiguous_binding_count: int = 0,
    coverage: Mapping[str, Any] | None = None,
    perception_summary: Mapping[str, Any] | None = None,
    pool_reusable: bool = False,
) -> dict[str, Any]:
    slot_count = int(slot_count)
    filled_slot_count = max(0, int(filled_slot_count))
    unfilled_slot_count = max(0, slot_count - filled_slot_count)
    required = max(0, int(coverage_required_cell_count))
    observed = max(0, int(coverage_observed_cell_count))
    deficit = max(0, required - observed)
    return {
        "slot_count": slot_count,
        "filled_slot_count": filled_slot_count,
        "unfilled_slot_count": unfilled_slot_count,
        "pool_object_count": max(0, int(pool_object_count)),
        "slot_candidate_binding_count": max(0, int(slot_candidate_binding_count)),
        "coverage_required_cell_count": required,
        "coverage_observed_cell_count": observed,
        "coverage_deficit_cell_count": deficit,
        "ambiguous_binding_count": max(0, int(ambiguous_binding_count)),
        "pool_reusable": bool(pool_reusable),
        "pool_exhausted": bool(
            int(pool_object_count) <= 0 if bool(pool_reusable) else int(pool_object_count) < unfilled_slot_count
        ),
        "slots": [dict(row) for row in slots],
        "pool_objects": [dict(row) for row in pool_objects],
        "coverage": dict(coverage or {}),
        "perception_summary": dict(perception_summary or {}),
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
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    if 0 <= nx < width and 0 <= ny < height and not seen[ny, nx] and int(arr[ny, nx]) == color:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            area = len(xs)
            out.append(
                {
                    "color": int(color),
                    "area": int(area),
                    "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    "centroid": [int(round(sum(xs) / area)), int(round(sum(ys) / area))],
                }
            )
    return out


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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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


def _pop_unused_pool(
    candidates: Sequence[Mapping[str, Any]],
    used_pool_ids: set[str],
) -> Mapping[str, Any] | None:
    for index, candidate in enumerate(candidates):
        pool_id = str(candidate.get("id", f"pool:{index}") or f"pool:{index}")
        if pool_id not in used_pool_ids:
            return candidate
    return None


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
    "SLOT_POOL_COVERAGE_DIAGNOSTIC_SENSOR_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_PACKET_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_CANDIDATE_MINER_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_EFFECT_INTENT_CONTROL_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_EXACT_SLOT_SCORELESS_CONTROL_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_FOCUSED_REPLAY_EXECUTION_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_FOCUSED_REPLAY_OUTCOME_ADMISSION_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_FOCUSED_REPLAY_READINESS_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_FOCUSED_SCORE_BEARING_CANDIDATE_DISCOVERY_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_SCORE_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RUNTIME_BRIDGE_GUARD_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RUNTIME_BRIDGE_OUTCOME_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_BREADTH_NON_REGRESSION_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RANK_GUARD_SHADOW_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RANK_GUARD_RUNTIME_GUARD_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_LIVE_RANK_GUARD_FOCUSED_RUNTIME_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_GUARDED_PROBE_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_ALREADY_SCORED_EVENT_EXCLUSION_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_INTERVENTION_CONTRAST_ATTRIBUTION_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_INTERVENTION_TEMPORAL_COUNTERFACTUAL_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_NEGATIVE_MEMORY_GUARDED_REVIEW_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_NEGATIVE_MEMORY_RANKING_SHADOW_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_POSTERIOR_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_PROTECTED_PROPOSAL_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_REGION_IDENTITY_CONTROL_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_SCOPED_SCORELESS_CONTROL_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_SCORE_EVENT_SCOPE_ATTRIBUTION_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_SCORELESS_PROGRESS_CONTROL_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_TARGET_SLOT_SCORELESS_CONTROL_SCHEMA_VERSION",
    "SLOT_POOL_COVERAGE_TERMINAL_EFFECT_SCORE_CLOSURE_ATTRIBUTION_SCHEMA_VERSION",
    "build_slot_pool_coverage_packet",
    "build_slot_pool_coverage_posterior_packet",
    "build_slot_pool_coverage_protected_proposal",
    "summarize_slot_pool_coverage_candidates",
    "summarize_slot_pool_coverage_counterfactual_controls",
    "summarize_slot_pool_coverage_effect_intent_controls",
    "summarize_slot_pool_coverage_exact_slot_scoreless_controls",
    "summarize_slot_pool_coverage_focused_replay_execution",
    "summarize_slot_pool_coverage_focused_replay_outcome_admission",
    "summarize_slot_pool_coverage_focused_replay_readiness",
    "summarize_slot_pool_coverage_focused_score_bearing_candidate_discovery",
    "summarize_slot_pool_coverage_submit_precondition_score",
    "summarize_slot_pool_coverage_submit_precondition_runtime_bridge_guard",
    "summarize_slot_pool_coverage_submit_precondition_runtime_bridge_outcome",
    "summarize_slot_pool_coverage_submit_precondition_breadth_non_regression",
    "summarize_slot_pool_coverage_submit_precondition_rank_guard_shadow",
    "summarize_slot_pool_coverage_submit_precondition_rank_guard_runtime_guard",
    "summarize_slot_pool_coverage_submit_precondition_live_rank_guard_focused_runtime",
    "summarize_slot_pool_coverage_guarded_probe_outcomes",
    "summarize_slot_pool_coverage_already_scored_event_exclusion",
    "summarize_slot_pool_coverage_intervention_contrast_attribution",
    "summarize_slot_pool_coverage_intervention_temporal_counterfactual",
    "summarize_slot_pool_coverage_negative_memory_guarded_review",
    "summarize_slot_pool_coverage_negative_memory_ranking_shadow",
    "summarize_slot_pool_coverage_posterior",
    "summarize_slot_pool_coverage_packets",
    "summarize_slot_pool_coverage_region_identity_controls",
    "summarize_slot_pool_coverage_scoped_scoreless_controls",
    "summarize_slot_pool_coverage_scoreless_progress_controls",
    "summarize_slot_pool_coverage_target_slot_scoreless_controls",
    "summarize_slot_pool_coverage_terminal_effect_score_closure_attribution",
    "summarize_slot_pool_coverage_score_event_scope_attribution",
    "validate_slot_pool_coverage_packet",
]
