"""A/C-axis sensor gap inventory.

This module is diagnostic only. It compares a high-ceiling axis, such as the
Phase37 A-axis adapter run, with a current adapter-free run and groups the
remaining score gap by ECA sensor families. It does not select actions or
authorize planner promotion.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping


SENSOR_GAP_INVENTORY_SCHEMA_VERSION = "eca_sensor_gap_inventory.v1"
SENSOR_PROMOTION_GAP_INVENTORY_SCHEMA_VERSION = (
    "eca_sensor_promotion_gap_inventory.v1"
)
PARALLEL_SENSOR_SUBSTRATE_GAP_INVENTORY_SCHEMA_VERSION = (
    "eca_parallel_sensor_substrate_gap_inventory.v1"
)
SENSOR_PROMOTION_BLOCKER_INVENTORY_SCHEMA_VERSION = (
    "eca_sensor_promotion_blocker_inventory.v1"
)


A_AXIS_PHASE37_ADAPTER_SCORE_BY_GAME: dict[str, int] = {
    "ar25": 8,
    "bp35": 4,
    "cd82": 6,
    "cn04": 6,
    "dc22": 3,
    "ft09": 6,
    "g50t": 1,
    "ka59": 5,
    "lf52": 5,
    "lp85": 8,
    "ls20": 7,
    "m0r0": 2,
    "r11l": 6,
    "re86": 3,
    "s5i5": 3,
    "sb26": 8,
    "sc25": 6,
    "sk48": 2,
    "sp80": 3,
    "su15": 3,
    "tn36": 6,
    "tr87": 6,
    "tu93": 2,
    "vc33": 7,
    "wa30": 3,
}


SENSOR_FAMILIES_BY_GAME: dict[str, tuple[str, ...]] = {
    "ar25": ("coordinate_objective_posterior", "slot_pool_coverage"),
    "bp35": ("transition_state_variable", "platform_destructible_topology"),
    "cd82": ("slot_pool_coverage",),
    "cn04": ("transition_state_variable", "multi_body_contact_topology"),
    "dc22": ("route_validity", "coordinate_objective_posterior"),
    "g50t": ("transition_state_variable", "round_memory"),
    "ka59": (
        "route_validity",
        "coordinate_objective_posterior",
        "multi_body_contact_topology",
    ),
    "lf52": ("transition_state_variable", "multi_body_contact_topology"),
    "lp85": ("coordinate_objective_posterior", "transition_state_variable"),
    "ls20": ("transition_state_variable", "coordinate_objective_posterior"),
    "m0r0": ("route_validity",),
    "r11l": ("coordinate_objective_posterior", "kinematic_membership"),
    "re86": ("slot_pool_coverage", "shape_coverage_posterior"),
    "s5i5": ("transition_state_variable",),
    "sb26": ("slot_pool_coverage",),
    "sc25": ("transition_state_variable", "program_spell_pattern"),
    "sk48": ("transition_state_variable", "multi_body_contact_topology"),
    "sp80": ("slot_pool_coverage",),
    "su15": ("force_field_merge_containment",),
    "tn36": ("transition_state_variable", "program_row_semantics"),
    "tu93": ("route_validity",),
    "vc33": ("transition_state_variable",),
    "wa30": ("route_validity", "multi_body_contact_topology"),
}


SENSOR_FAMILY_STATUS: dict[str, str] = {
    "coordinate_objective_posterior": "partial_h_axis_current",
    "force_field_merge_containment": "partial_h_axis_current",
    "kinematic_membership": "partial_h_axis_current",
    "multi_body_contact_topology": "partial_h_axis_current",
    "platform_destructible_topology": "partial_h_axis_current",
    "program_row_semantics": "partial_h_axis_current",
    "program_spell_pattern": "partial_h_axis_current",
    "route_validity": "partial_h_axis_current",
    "round_memory": "partial_h_axis_current",
    "shape_coverage_posterior": "partial_c_axis_current",
    "slot_pool_coverage": "partial_h_axis_current",
    "transition_state_variable": "partial_h_axis_current",
}


SENSOR_PROMOTION_REQUIRED_STAGES: tuple[str, ...] = (
    "diagnostic_sensor",
    "posterior",
    "candidate_miner",
    "guarded_probe",
    "counterfactual_control",
    "all25_non_regression",
)


SENSOR_FAMILY_REQUIRED_STAGE_OVERRIDES: dict[str, tuple[str, ...]] = {
    "slot_pool_coverage": SENSOR_PROMOTION_REQUIRED_STAGES
    + (
        "negative_memory_guarded_review",
        "negative_memory_ranking_shadow",
        "focused_replay_readiness",
        "focused_runtime_execution",
        "focused_replay_outcome_admission_guard",
        "focused_score_bearing_candidate_discovery",
        "submit_precondition_score",
        "submit_precondition_runtime_bridge_guard",
        "submit_precondition_runtime_bridge_outcome",
        "submit_precondition_breadth_non_regression",
        "submit_precondition_rank_guard_shadow",
        "submit_precondition_rank_guard_runtime_guard",
        "submit_precondition_live_rank_guard_focused_runtime",
    ),
}


SENSOR_FAMILY_COMPLETED_STAGES: dict[str, tuple[str, ...]] = {
    "coordinate_objective_posterior": (
        "diagnostic_sensor",
        "posterior",
        "candidate_miner",
        "guarded_probe",
        "counterfactual_control",
        "all25_non_regression",
    ),
    "force_field_merge_containment": (
        "diagnostic_sensor",
        "posterior",
        "candidate_miner",
        "guarded_probe",
        "counterfactual_control",
        "all25_non_regression",
    ),
    "kinematic_membership": (
        "diagnostic_sensor",
        "posterior",
        "candidate_miner",
        "guarded_probe",
        "counterfactual_control",
        "all25_non_regression",
    ),
    "multi_body_contact_topology": (
        "diagnostic_sensor",
        "posterior",
        "candidate_miner",
        "guarded_probe",
        "counterfactual_control",
        "all25_non_regression",
    ),
    "platform_destructible_topology": (
        "diagnostic_sensor",
        "posterior",
        "candidate_miner",
        "guarded_probe",
        "counterfactual_control",
        "all25_non_regression",
    ),
    "program_row_semantics": (
        "diagnostic_sensor",
        "posterior",
        "candidate_miner",
        "guarded_probe",
        "counterfactual_control",
        "all25_non_regression",
    ),
    "program_spell_pattern": (
        "diagnostic_sensor",
        "posterior",
        "candidate_miner",
        "guarded_probe",
        "counterfactual_control",
        "all25_non_regression",
    ),
    "route_validity": (
        "diagnostic_sensor",
        "posterior",
        "candidate_miner",
        "guarded_probe",
        "counterfactual_control",
        "all25_non_regression",
    ),
    "round_memory": (
        "diagnostic_sensor",
        "posterior",
        "candidate_miner",
        "guarded_probe",
        "counterfactual_control",
        "all25_non_regression",
    ),
    "shape_coverage_posterior": (
        "diagnostic_sensor",
        "posterior",
        "candidate_miner",
        "guarded_probe",
        "counterfactual_control",
        "all25_non_regression",
    ),
    "slot_pool_coverage": (
        "diagnostic_sensor",
        "posterior",
        "candidate_miner",
        "guarded_probe",
        "counterfactual_control",
        "all25_non_regression",
        "negative_memory_guarded_review",
        "negative_memory_ranking_shadow",
        "focused_replay_readiness",
        "focused_runtime_execution",
        "focused_replay_outcome_admission_guard",
        "focused_score_bearing_candidate_discovery",
        "submit_precondition_score",
        "submit_precondition_runtime_bridge_guard",
        "submit_precondition_runtime_bridge_outcome",
        "submit_precondition_breadth_non_regression",
        "submit_precondition_rank_guard_shadow",
        "submit_precondition_rank_guard_runtime_guard",
        "submit_precondition_live_rank_guard_focused_runtime",
    ),
    "transition_state_variable": (
        "diagnostic_sensor",
        "posterior",
        "candidate_miner",
        "guarded_probe",
        "counterfactual_control",
        "all25_non_regression",
    ),
}


SENSOR_PROMOTION_BLOCKER_CONTRACT_KEYS_BY_FAMILY: dict[
    str, dict[str, str]
] = {
    "force_field_merge_containment": {
        "all25": "force_field_merge_containment_all25_non_regression_sensor",
        "counterfactual": "force_field_merge_containment_counterfactual_control_sensor",
        "outcome": "force_field_merge_containment_guarded_probe_outcome_sensor",
    },
    "kinematic_membership": {
        "all25": "kinematic_membership_all25_non_regression_sensor",
        "counterfactual": "kinematic_membership_counterfactual_control_sensor",
    },
    "platform_destructible_topology": {
        "all25": "platform_destructible_topology_all25_non_regression_sensor",
        "counterfactual": "platform_destructible_topology_counterfactual_control_sensor",
        "outcome": "platform_destructible_topology_guarded_probe_outcome_sensor",
    },
    "program_row_semantics": {
        "all25": "program_row_semantics_all25_non_regression_sensor",
        "counterfactual": "program_row_semantics_counterfactual_control_sensor",
        "outcome": "program_row_semantics_guarded_probe_outcome_sensor",
    },
    "program_spell_pattern": {
        "all25": "program_spell_pattern_all25_non_regression_sensor",
        "counterfactual": "program_spell_pattern_counterfactual_control_sensor",
        "outcome": "program_spell_pattern_guarded_probe_outcome_sensor",
    },
    "route_validity": {
        "all25": "route_validity_all25_non_regression_sensor",
        "outcome": "route_validity_recovery_probe_outcome_sensor",
    },
}


PARALLEL_SENSOR_SUBSTRATE_FAMILIES_BY_GAME: dict[str, tuple[str, ...]] = {
    # Parallel L/M work found that palette-shift transfer and layout robustness
    # now mostly fail below the planner: actor identity, structural role
    # perception, or interaction-learned effects are missing.
    "ar25": ("coordinate_objective_posterior", "slot_pool_coverage"),
    "bp35": (
        "moving_sprite_tracker",
        "actor_identity_anchor",
        "platform_destructible_topology",
    ),
    "cd82": ("slot_pool_coverage",),
    "cn04": ("moving_sprite_tracker", "multi_body_contact_topology"),
    "dc22": ("invisible_effect_position_memory", "route_validity"),
    "ft09": ("mechanic_certificate_dispatch", "palette_invariant_perception"),
    "g50t": ("round_memory", "transport_effect_self_repair"),
    "ka59": (
        "moving_sprite_tracker",
        "actor_identity_anchor",
        "multi_body_contact_topology",
    ),
    "lf52": (
        "moving_sprite_tracker",
        "actor_identity_anchor",
        "transport_effect_self_repair",
    ),
    "lp85": ("coordinate_objective_posterior", "transition_state_variable"),
    "ls20": (
        "moving_sprite_tracker",
        "actor_identity_anchor",
        "transport_effect_self_repair",
        "invisible_effect_position_memory",
    ),
    "m0r0": (
        "mechanic_certificate_dispatch",
        "tile_effect_self_repair",
        "palette_invariant_perception",
        "invisible_effect_position_memory",
    ),
    "r11l": (
        "coordinate_objective_posterior",
        "kinematic_membership",
        "moving_sprite_tracker",
        "actor_identity_anchor",
    ),
    "re86": ("palette_invariant_perception", "slot_pool_coverage"),
    "s5i5": ("transition_state_variable",),
    "sb26": ("slot_pool_coverage",),
    "sc25": (
        "moving_sprite_tracker",
        "actor_identity_anchor",
        "program_spell_pattern",
    ),
    "sk48": (
        "moving_sprite_tracker",
        "actor_identity_anchor",
        "multi_body_contact_topology",
    ),
    "sp80": ("mechanic_certificate_dispatch", "palette_invariant_perception"),
    "su15": ("force_field_merge_containment",),
    "tn36": ("program_row_semantics", "transition_state_variable"),
    "tr87": ("transition_state_variable",),
    "tu93": ("route_validity",),
    "vc33": ("transition_state_variable",),
    "wa30": (
        "moving_sprite_tracker",
        "actor_identity_anchor",
        "transport_effect_self_repair",
        "multi_body_contact_topology",
    ),
}


PARALLEL_SENSOR_SUBSTRATE_STATUS: dict[str, str] = {
    "actor_identity_anchor": "parallel_intake_unit_verified_observation_only",
    "invisible_effect_position_memory": "missing_parallel_frontier",
    "mechanic_certificate_dispatch": "parallel_intake_unit_verified_observation_only",
    "moving_sprite_tracker": "parallel_intake_unit_verified_observation_only",
    "multi_body_contact_topology": "partial_h_axis_current",
    "palette_invariant_perception": "parallel_intake_unit_verified_observation_only",
    "platform_destructible_topology": "partial_h_axis_current",
    "program_spell_pattern": "partial_h_axis_current",
    "route_validity": "partial_h_axis_current",
    "round_memory": "partial_h_axis_current",
    "slot_pool_coverage": "partial_h_axis_current",
    "tile_effect_self_repair": "parallel_intake_unit_verified_observation_only",
    "transport_effect_self_repair": "parallel_intake_unit_verified_observation_only",
}


PARALLEL_SENSOR_SUBSTRATE_REQUIRED_STAGES: tuple[str, ...] = (
    "diagnostic_sensor",
    "unit_fixture",
    "runtime_trace_connection",
    "posterior",
    "guarded_planner_connection",
    "all25_non_regression",
)


PARALLEL_SENSOR_SUBSTRATE_REQUIRED_STAGE_OVERRIDES: dict[str, tuple[str, ...]] = {
    "moving_sprite_tracker": (
        "diagnostic_sensor",
        "unit_fixture",
        "runtime_trace_connection",
        "actor_identity_anchor_connection",
        "motion_posterior",
        "guarded_planner_readiness",
        "actor_ambiguity_splitter",
        "guarded_actor_anchor_posterior",
        "soft_anchor_motion_posterior",
        "coupled_motion_symmetry",
        "coupled_pair_goal_effect_bridge",
        "coupled_pair_goal_score_attribution_guard",
        "target_goal_counterfactual_attribution",
        "effect_component_spatial_stability",
        "sparse_chain_rail_state_sensor",
        "target_goal_binding_state_sensor",
        "goal_anchor_perception_sensor",
        "guarded_planner_connection",
        "all25_non_regression",
    ),
    "tile_effect_self_repair": (
        "diagnostic_sensor",
        "unit_fixture",
        "runtime_trace_connection",
        "segment_action_effect_trace_ingestion",
        "posterior",
        "guarded_planner_connection",
        "all25_non_regression",
        "guard_precision_review",
        "guard_precision_all25_non_regression",
        "counterfactual_hazard_control",
        "hazard_source_trace_connection",
        "return_anchor_prior_trace_connection",
        "family_scope_negative_control",
        "hazard_source_repair_bridge",
        "model_repair_proposal",
        "model_repair_write_guard",
        "model_repair_write_audit",
        "model_repair_runtime_probe_proposal",
        "model_repair_protected_probe_execution_gate",
        "model_repair_protected_probe_runtime_bridge_guard",
        "model_repair_protected_probe_runtime_executor",
        "model_repair_protected_probe_runtime_executor_all25_non_regression",
        "model_repair_protected_probe_runtime_outcome_execution",
        "model_repair_protected_probe_runtime_bridge_outcome",
        "model_repair_protected_probe_outcome_posterior",
        "model_repair_protected_probe_rank_guard_shadow",
        "model_repair_protected_probe_rank_guard_runtime_guard",
        "model_repair_protected_probe_live_rank_guard_focused_runtime",
        "model_repair_protected_probe_live_rank_guard_all25_non_regression",
        "model_repair_protected_probe_live_rank_guard_action_selection_guard",
        "model_repair_protected_probe_live_rank_guard_positive_action_selection_admission",
        "model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution",
        "model_repair_protected_probe_live_rank_guard_score_attributed_positive_rank_promotion_guard",
        "model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery",
        "model_repair_protected_probe_live_rank_guard_score_attributed_candidate_focused_runtime_replay",
    ),
}


PARALLEL_SENSOR_SUBSTRATE_COMPLETED_STAGES: dict[str, tuple[str, ...]] = {
    "actor_identity_anchor": (
        "diagnostic_sensor",
        "unit_fixture",
        "runtime_trace_connection",
        "posterior",
        "guarded_planner_connection",
        "all25_non_regression",
    ),
    "mechanic_certificate_dispatch": (
        "diagnostic_sensor",
        "unit_fixture",
        "runtime_trace_connection",
        "posterior",
        "guarded_planner_connection",
        "all25_non_regression",
    ),
    "coordinate_objective_posterior": (
        "diagnostic_sensor",
        "unit_fixture",
        "runtime_trace_connection",
        "posterior",
        "guarded_planner_connection",
        "all25_non_regression",
    ),
    "invisible_effect_position_memory": (
        "diagnostic_sensor",
        "unit_fixture",
        "runtime_trace_connection",
        "posterior",
        "guarded_planner_connection",
        "all25_non_regression",
    ),
    "moving_sprite_tracker": (
        "diagnostic_sensor",
        "unit_fixture",
        "runtime_trace_connection",
        "actor_identity_anchor_connection",
        "motion_posterior",
        "guarded_planner_readiness",
        "actor_ambiguity_splitter",
        "guarded_actor_anchor_posterior",
        "soft_anchor_motion_posterior",
        "coupled_motion_symmetry",
        "coupled_pair_goal_effect_bridge",
        "coupled_pair_goal_score_attribution_guard",
        "target_goal_counterfactual_attribution",
        "effect_component_spatial_stability",
        "sparse_chain_rail_state_sensor",
        "target_goal_binding_state_sensor",
        "goal_anchor_perception_sensor",
        "guarded_planner_connection",
        "all25_non_regression",
    ),
    "palette_invariant_perception": (
        "diagnostic_sensor",
        "unit_fixture",
        "runtime_trace_connection",
        "posterior",
        "guarded_planner_connection",
        "all25_non_regression",
    ),
    "tile_effect_self_repair": (
        "diagnostic_sensor",
        "unit_fixture",
        "runtime_trace_connection",
        "segment_action_effect_trace_ingestion",
        "posterior",
        "guarded_planner_connection",
        "all25_non_regression",
        "guard_precision_review",
        "guard_precision_all25_non_regression",
        "counterfactual_hazard_control",
        "hazard_source_trace_connection",
        "return_anchor_prior_trace_connection",
        "family_scope_negative_control",
        "hazard_source_repair_bridge",
        "model_repair_proposal",
        "model_repair_write_guard",
        "model_repair_write_audit",
        "model_repair_runtime_probe_proposal",
        "model_repair_protected_probe_execution_gate",
        "model_repair_protected_probe_runtime_bridge_guard",
        "model_repair_protected_probe_runtime_executor",
        "model_repair_protected_probe_runtime_executor_all25_non_regression",
        "model_repair_protected_probe_runtime_outcome_execution",
        "model_repair_protected_probe_runtime_bridge_outcome",
        "model_repair_protected_probe_outcome_posterior",
        "model_repair_protected_probe_rank_guard_shadow",
        "model_repair_protected_probe_rank_guard_runtime_guard",
        "model_repair_protected_probe_live_rank_guard_focused_runtime",
        "model_repair_protected_probe_live_rank_guard_all25_non_regression",
        "model_repair_protected_probe_live_rank_guard_action_selection_guard",
        "model_repair_protected_probe_live_rank_guard_positive_action_selection_admission",
        "model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution",
        "model_repair_protected_probe_live_rank_guard_score_attributed_positive_rank_promotion_guard",
        "model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery",
        "model_repair_protected_probe_live_rank_guard_score_attributed_candidate_focused_runtime_replay",
    ),
    "transition_state_variable": (
        "diagnostic_sensor",
        "unit_fixture",
        "runtime_trace_connection",
        "posterior",
        "guarded_planner_connection",
        "all25_non_regression",
    ),
    "transport_effect_self_repair": (
        "diagnostic_sensor",
        "unit_fixture",
        "runtime_trace_connection",
        "posterior",
        "guarded_planner_connection",
        "all25_non_regression",
    ),
}


def build_sensor_gap_inventory(
    high_ceiling_scores: Mapping[str, Any] | None,
    current_scores: Mapping[str, Any] | None,
    *,
    sensor_families_by_game: Mapping[str, tuple[str, ...]] | None = None,
    sensor_family_status: Mapping[str, str] | None = None,
    high_ceiling_axis: str = "A_phase37_adapter",
    current_axis: str = "C_current_adapter_free",
    include_zero_gap_games: bool = False,
) -> dict[str, Any]:
    """Build a diagnostic inventory of remaining score gap by sensor family."""

    high = _score_map(high_ceiling_scores or {})
    current = _score_map(current_scores or {})
    family_map = dict(sensor_families_by_game or SENSOR_FAMILIES_BY_GAME)
    status_map = dict(sensor_family_status or SENSOR_FAMILY_STATUS)
    games = sorted(set(high) | set(current) | set(family_map))

    per_game: list[dict[str, Any]] = []
    primary_gap_by_family: Counter[str] = Counter()
    support_gap_by_family: Counter[str] = Counter()
    game_count_by_family: Counter[str] = Counter()
    status_by_family: dict[str, str] = {}
    missing_gap_sum = 0

    for game in games:
        ceiling = int(high.get(game, 0))
        score = int(current.get(game, 0))
        gap = max(0, ceiling - score)
        families = tuple(str(v) for v in family_map.get(game, ("unclassified",)) if str(v))
        primary = families[0] if families else "unclassified"
        if gap > 0:
            primary_gap_by_family[primary] += gap
            for family in families:
                support_gap_by_family[family] += gap
                game_count_by_family[family] += 1
                status_by_family[family] = str(
                    status_map.get(family, "unclassified_sensor_status")
                )
            if any(_status_is_missing_or_partial(status_by_family.get(f, "")) for f in families):
                missing_gap_sum += gap
        if gap > 0 or include_zero_gap_games:
            per_game.append(
                {
                    "game_id": game,
                    "high_ceiling_score": ceiling,
                    "current_score": score,
                    "positive_gap": gap,
                    "primary_sensor_family": primary,
                    "sensor_families": list(families),
                    "sensor_family_status": {
                        family: str(status_map.get(family, "unclassified_sensor_status"))
                        for family in families
                    },
                }
            )

    family_rows = []
    all_families = sorted(set(primary_gap_by_family) | set(support_gap_by_family))
    for family in all_families:
        family_rows.append(
            {
                "sensor_family": family,
                "status": str(status_map.get(family, "unclassified_sensor_status")),
                "primary_gap_score_sum": int(primary_gap_by_family.get(family, 0)),
                "supporting_gap_score_sum": int(support_gap_by_family.get(family, 0)),
                "positive_gap_game_count": int(game_count_by_family.get(family, 0)),
                "missing_or_partial": _status_is_missing_or_partial(
                    str(status_map.get(family, "unclassified_sensor_status"))
                ),
            }
        )
    family_rows.sort(
        key=lambda row: (
            -int(row["primary_gap_score_sum"]),
            -int(row["supporting_gap_score_sum"]),
            str(row["sensor_family"]),
        )
    )
    per_game.sort(
        key=lambda row: (-int(row["positive_gap"]), str(row["primary_sensor_family"]), str(row["game_id"]))
    )

    gap_sum = sum(max(0, int(high.get(game, 0)) - int(current.get(game, 0))) for game in games)
    return {
        "schema_version": SENSOR_GAP_INVENTORY_SCHEMA_VERSION,
        "diagnostic_only": True,
        "high_ceiling_axis": str(high_ceiling_axis),
        "current_axis": str(current_axis),
        "high_ceiling_score_sum": sum(int(high.get(game, 0)) for game in games),
        "current_score_sum": sum(int(current.get(game, 0)) for game in games),
        "positive_gap_score_sum": int(gap_sum),
        "missing_or_partial_gap_score_sum": int(missing_gap_sum),
        "positive_gap_game_count": sum(1 for row in per_game if int(row["positive_gap"]) > 0),
        "top_missing_sensor_family": family_rows[0]["sensor_family"] if family_rows else "",
        "ranked_sensor_families": family_rows,
        "per_game": per_game,
        "claim_boundary": {
            "score_lift_claim": False,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "axis_comparison_only": True,
        },
    }


def build_phase37_to_current_sensor_gap_inventory(
    current_scores: Mapping[str, Any],
    *,
    current_axis: str = "C_current_adapter_free",
) -> dict[str, Any]:
    return build_sensor_gap_inventory(
        A_AXIS_PHASE37_ADAPTER_SCORE_BY_GAME,
        current_scores,
        high_ceiling_axis="A_phase37_adapter_119",
        current_axis=current_axis,
    )


def build_sensor_promotion_gap_inventory(
    high_ceiling_scores: Mapping[str, Any] | None,
    current_scores: Mapping[str, Any] | None,
    *,
    sensor_families_by_game: Mapping[str, tuple[str, ...]] | None = None,
    sensor_family_status: Mapping[str, str] | None = None,
    completed_stages_by_family: Mapping[str, tuple[str, ...]] | None = None,
    required_stages: tuple[str, ...] = SENSOR_PROMOTION_REQUIRED_STAGES,
    required_stage_overrides_by_family: Mapping[str, tuple[str, ...]] | None = None,
    high_ceiling_axis: str = "A_phase37_adapter",
    current_axis: str = "C_current_adapter_free",
) -> dict[str, Any]:
    """Rank remaining sensor families by their next missing safe-promotion stage."""

    base = build_sensor_gap_inventory(
        high_ceiling_scores,
        current_scores,
        sensor_families_by_game=sensor_families_by_game,
        sensor_family_status=sensor_family_status,
        high_ceiling_axis=high_ceiling_axis,
        current_axis=current_axis,
    )
    completed_map = dict(completed_stages_by_family or SENSOR_FAMILY_COMPLETED_STAGES)
    override_map = dict(
        required_stage_overrides_by_family or SENSOR_FAMILY_REQUIRED_STAGE_OVERRIDES
    )
    stage_rows: list[dict[str, Any]] = []
    for family_row in list(base.get("ranked_sensor_families") or []):
        family = str(family_row.get("sensor_family", "") or "")
        if not family:
            continue
        family_required_stages = tuple(
            str(stage) for stage in override_map.get(family, required_stages)
        )
        completed_set = set(str(v) for v in completed_map.get(family, ()))
        completed = tuple(
            stage for stage in family_required_stages if stage in completed_set
        )
        missing = tuple(
            stage for stage in family_required_stages if stage not in completed_set
        )
        next_stage = missing[0] if missing else "ready_for_planner_promotion_review"
        recommended_next_sensor = "" if not missing else f"{family}_{next_stage}"
        stage_rows.append(
            {
                "sensor_family": family,
                "status": str(family_row.get("status", "") or ""),
                "primary_gap_score_sum": _safe_int(
                    family_row.get("primary_gap_score_sum")
                ),
                "supporting_gap_score_sum": _safe_int(
                    family_row.get("supporting_gap_score_sum")
                ),
                "positive_gap_game_count": _safe_int(
                    family_row.get("positive_gap_game_count")
                ),
                "completed_stages": list(completed),
                "missing_stages": list(missing),
                "next_missing_stage": next_stage,
                "recommended_next_sensor": recommended_next_sensor,
                "ready_for_planner_promotion_review": not bool(missing),
            }
        )

    actionable_rows = [
        row
        for row in stage_rows
        if row["next_missing_stage"] != "ready_for_planner_promotion_review"
        and int(row["supporting_gap_score_sum"]) > 0
    ]
    actionable_rows.sort(
        key=lambda row: (
            -int(row["supporting_gap_score_sum"]),
            -int(row["primary_gap_score_sum"]),
            str(row["next_missing_stage"]),
            str(row["sensor_family"]),
        )
    )
    stage_rows.sort(
        key=lambda row: (
            row["next_missing_stage"] == "ready_for_planner_promotion_review",
            -int(row["supporting_gap_score_sum"]),
            -int(row["primary_gap_score_sum"]),
            str(row["sensor_family"]),
        )
    )
    top = actionable_rows[0] if actionable_rows else {}
    return {
        "schema_version": SENSOR_PROMOTION_GAP_INVENTORY_SCHEMA_VERSION,
        "diagnostic_only": True,
        "high_ceiling_axis": str(high_ceiling_axis),
        "current_axis": str(current_axis),
        "positive_gap_score_sum": _safe_int(base.get("positive_gap_score_sum")),
        "top_missing_sensor_family": str(top.get("sensor_family", "") or ""),
        "top_missing_stage": str(top.get("next_missing_stage", "") or ""),
        "top_recommended_next_sensor": str(
            top.get("recommended_next_sensor", "") or ""
        ),
        "ranked_stage_gaps": stage_rows,
        "claim_boundary": {
            "score_lift_claim": False,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "axis_comparison_only": True,
            "stage_inventory_only": True,
        },
    }


def build_phase37_to_current_sensor_promotion_gap_inventory(
    current_scores: Mapping[str, Any],
    *,
    current_axis: str = "C_current_adapter_free",
) -> dict[str, Any]:
    return build_sensor_promotion_gap_inventory(
        A_AXIS_PHASE37_ADAPTER_SCORE_BY_GAME,
        current_scores,
        high_ceiling_axis="A_phase37_adapter_119",
        current_axis=current_axis,
    )


def build_sensor_promotion_blocker_inventory(
    contract: Mapping[str, Any] | None,
    *,
    stage_inventory: Mapping[str, Any] | None = None,
    contract_keys_by_family: Mapping[str, Mapping[str, str]]
    | None = None,
) -> dict[str, Any]:
    """Rank stage-complete sensor families by remaining promotion blockers."""

    contract = contract if isinstance(contract, Mapping) else {}
    key_map = dict(
        contract_keys_by_family or SENSOR_PROMOTION_BLOCKER_CONTRACT_KEYS_BY_FAMILY
    )
    stage_inventory = (
        stage_inventory
        if isinstance(stage_inventory, Mapping)
        else contract.get("a_axis_sensor_promotion_gap_inventory_sensor")
    )
    if not isinstance(stage_inventory, Mapping):
        stage_inventory = {}
    stage_rows = [
        dict(row)
        for row in list(stage_inventory.get("ranked_stage_gaps") or [])
        if isinstance(row, Mapping)
    ]

    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for stage_row in stage_rows:
        family = str(stage_row.get("sensor_family", "") or "")
        if not family or family not in key_map:
            continue
        keys = dict(key_map.get(family) or {})
        all25 = _contract_summary(contract, keys.get("all25"))
        counterfactual = _contract_summary(contract, keys.get("counterfactual"))
        outcome = _contract_summary(contract, keys.get("outcome"))

        stage_missing = [
            str(stage) for stage in list(stage_row.get("missing_stages") or [])
        ]
        stage_ready = not bool(stage_missing)
        all25_present = bool(all25) and _summary_enabled_or_implicit(all25)
        all25_error_count = _safe_int(all25.get("error_count"))
        regressed_count = _safe_int(
            all25.get("regressed_game_count"),
        )
        score_delta = _safe_int(all25.get("score_sum_delta_vs_reference"))
        score_delta_available = "score_sum_delta_vs_reference" in all25
        all25_preserved = bool(
            all25_present
            and all25_error_count == 0
            and regressed_count == 0
            and (not score_delta_available or score_delta >= 0)
        )
        score_positive = max(
            _safe_int(outcome.get("score_positive_execution_count")),
            _safe_int(counterfactual.get("score_positive_execution_count")),
            _safe_int(all25.get("score_positive_execution_count")),
        )
        scoreless_control = max(
            _safe_int(outcome.get("scoreless_control_execution_count")),
            _safe_int(counterfactual.get("scoreless_control_execution_count")),
            _safe_int(all25.get("scoreless_control_execution_count")),
        )
        counterfactual_pair_count = max(
            _safe_int(counterfactual.get("counterfactual_pair_count")),
            _safe_int(all25.get("counterfactual_pair_count")),
        )
        planner_blocked = bool(counterfactual.get("planner_promotion_blocked")) or bool(
            all25.get("counterfactual_planner_promotion_blocked")
        )
        if "ready_for_planner_promotion_review" in all25:
            all25_ready = bool(all25.get("ready_for_planner_promotion_review"))
        else:
            all25_ready = all25_preserved and not planner_blocked

        blockers: list[str] = []
        if not stage_ready:
            blockers.append("stage_inventory_incomplete")
        if not all25_present:
            blockers.append("missing_all25_non_regression_evidence")
        elif not all25_preserved:
            blockers.append("all25_non_regression_not_preserved")
        if score_positive <= 0:
            blockers.append("missing_score_positive_counterfactual")
        if counterfactual_pair_count <= 0:
            blockers.append("missing_counterfactual_pair")
        if planner_blocked:
            blockers.append("counterfactual_planner_blocked")
        if not all25_ready:
            blockers.append("all25_ready_for_planner_review_false")
        for blocker in blockers:
            reason_counts[blocker] += 1

        rows.append(
            {
                "sensor_family": family,
                "status": str(stage_row.get("status", "") or ""),
                "supporting_gap_score_sum": _safe_int(
                    stage_row.get("supporting_gap_score_sum")
                ),
                "primary_gap_score_sum": _safe_int(
                    stage_row.get("primary_gap_score_sum")
                ),
                "stage_ready_for_planner_promotion_review": bool(stage_ready),
                "completed_stages": list(stage_row.get("completed_stages") or []),
                "missing_stages": stage_missing,
                "all25_contract_key": str(keys.get("all25", "") or ""),
                "counterfactual_contract_key": str(
                    keys.get("counterfactual", "") or ""
                ),
                "outcome_contract_key": str(keys.get("outcome", "") or ""),
                "all25_evidence_present": bool(all25_present),
                "all25_preserved": bool(all25_preserved),
                "all25_promotion_state": str(
                    all25.get("promotion_state", "") or ""
                ),
                "score_sum_delta_vs_reference": (
                    score_delta if score_delta_available else None
                ),
                "regressed_game_count": int(regressed_count),
                "score_positive_execution_count": int(score_positive),
                "scoreless_control_execution_count": int(scoreless_control),
                "counterfactual_pair_count": int(counterfactual_pair_count),
                "counterfactual_promotion_state": str(
                    counterfactual.get("promotion_state")
                    or all25.get("counterfactual_promotion_state")
                    or ""
                ),
                "planner_promotion_blocked": bool(planner_blocked),
                "ready_for_planner_promotion_review": bool(
                    stage_ready
                    and all25_preserved
                    and score_positive > 0
                    and counterfactual_pair_count > 0
                    and not planner_blocked
                    and all25_ready
                ),
                "promotion_blockers": blockers,
                "recommended_next_evidence": _recommended_next_evidence(blockers),
            }
        )

    blocked_rows = [
        row
        for row in rows
        if not bool(row.get("ready_for_planner_promotion_review"))
        and _safe_int(row.get("supporting_gap_score_sum")) > 0
    ]
    blocked_rows.sort(
        key=lambda row: (
            -_safe_int(row.get("supporting_gap_score_sum")),
            -_safe_int(row.get("primary_gap_score_sum")),
            str(row.get("recommended_next_evidence", "")),
            str(row.get("sensor_family", "")),
        )
    )
    rows.sort(
        key=lambda row: (
            bool(row.get("ready_for_planner_promotion_review")),
            -_safe_int(row.get("supporting_gap_score_sum")),
            -_safe_int(row.get("primary_gap_score_sum")),
            str(row.get("sensor_family", "")),
        )
    )
    top = blocked_rows[0] if blocked_rows else {}
    return {
        "schema_version": SENSOR_PROMOTION_BLOCKER_INVENTORY_SCHEMA_VERSION,
        "diagnostic_only": True,
        "stage_inventory_schema_version": str(
            stage_inventory.get("schema_version", "") or ""
        ),
        "evaluated_sensor_family_count": int(len(rows)),
        "blocked_sensor_family_count": int(len(blocked_rows)),
        "ready_sensor_family_count": int(len(rows) - len(blocked_rows)),
        "top_blocked_sensor_family": str(top.get("sensor_family", "") or ""),
        "top_promotion_blocker": str(
            (top.get("promotion_blockers") or [""])[0]
            if top
            else ""
        ),
        "top_recommended_next_evidence": str(
            top.get("recommended_next_evidence", "") or ""
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "ranked_promotion_blockers": rows,
        "claim_boundary": {
            "score_lift_claim": False,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "diagnostic_inventory_only": True,
        },
    }


def build_parallel_sensor_substrate_gap_inventory(
    high_ceiling_scores: Mapping[str, Any] | None,
    current_scores: Mapping[str, Any] | None,
    *,
    high_ceiling_axis: str = "A_phase37_adapter",
    current_axis: str = "C_current_adapter_free",
) -> dict[str, Any]:
    """Inventory the parallel L/M substrate sensors against the A-axis gap.

    This is a separate diagnostic lens from the main promotion inventory. It
    asks where the newly imported parallel sensors are still missing runtime
    connection: palette-invariant role perception, moving-sprite tracking,
    actor identity anchoring, and interaction-learned tile/transport effects.
    """

    base = build_sensor_gap_inventory(
        high_ceiling_scores,
        current_scores,
        sensor_families_by_game=PARALLEL_SENSOR_SUBSTRATE_FAMILIES_BY_GAME,
        sensor_family_status={
            **SENSOR_FAMILY_STATUS,
            **PARALLEL_SENSOR_SUBSTRATE_STATUS,
        },
        high_ceiling_axis=high_ceiling_axis,
        current_axis=current_axis,
    )
    enriched: list[dict[str, Any]] = []
    for row in list(base.get("ranked_sensor_families") or []):
        family = str(row.get("sensor_family", "") or "")
        completed_set = set(
            str(stage)
            for stage in PARALLEL_SENSOR_SUBSTRATE_COMPLETED_STAGES.get(family, ())
        )
        required_stages = PARALLEL_SENSOR_SUBSTRATE_REQUIRED_STAGE_OVERRIDES.get(
            family, PARALLEL_SENSOR_SUBSTRATE_REQUIRED_STAGES
        )
        completed = [
            stage
            for stage in required_stages
            if stage in completed_set
        ]
        missing = [
            stage
            for stage in required_stages
            if stage not in completed_set
        ]
        next_stage = missing[0] if missing else "ready_for_planner_promotion_review"
        item = dict(row)
        item.update(
            {
                "completed_stages": completed,
                "missing_stages": missing,
                "next_missing_stage": next_stage,
                "recommended_next_sensor": (
                    "" if not missing else f"{family}_{next_stage}"
                ),
            }
        )
        enriched.append(item)

    actionable = [
        row
        for row in enriched
        if row["next_missing_stage"] != "ready_for_planner_promotion_review"
        and _safe_int(row.get("supporting_gap_score_sum")) > 0
        and str(row.get("status", "")).startswith(
            ("parallel_intake", "missing_parallel")
        )
    ]
    actionable.sort(
        key=lambda row: (
            -_safe_int(row.get("supporting_gap_score_sum")),
            -_safe_int(row.get("primary_gap_score_sum")),
            str(row.get("next_missing_stage", "")),
            str(row.get("sensor_family", "")),
        )
    )
    top = actionable[0] if actionable else {}
    base["schema_version"] = PARALLEL_SENSOR_SUBSTRATE_GAP_INVENTORY_SCHEMA_VERSION
    base["parallel_source"] = "merge/phase246-into-phase217:L/M sensor substrate"
    base["top_missing_sensor_family"] = str(top.get("sensor_family", "") or "")
    base["top_missing_stage"] = str(top.get("next_missing_stage", "") or "")
    base["top_recommended_next_sensor"] = str(
        top.get("recommended_next_sensor", "") or ""
    )
    base["ranked_sensor_families"] = enriched
    base["claim_boundary"].update(
        {
            "parallel_intake_only": True,
            "runtime_trace_connection_enabled": False,
            "planner_connection_enabled": False,
            "score_lift_claim": False,
        }
    )
    return base


def build_phase37_to_current_parallel_sensor_substrate_gap_inventory(
    current_scores: Mapping[str, Any],
    *,
    current_axis: str = "C_current_adapter_free",
) -> dict[str, Any]:
    return build_parallel_sensor_substrate_gap_inventory(
        A_AXIS_PHASE37_ADAPTER_SCORE_BY_GAME,
        current_scores,
        high_ceiling_axis="A_phase37_adapter_119",
        current_axis=current_axis,
    )


def _score_map(values: Mapping[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in values.items():
        game = str(key or "").strip().lower()
        if not game:
            continue
        out[game] = max(0, _safe_int(value))
    return out


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _contract_summary(
    contract: Mapping[str, Any],
    key: str | None,
) -> dict[str, Any]:
    if not key:
        return {}
    value = contract.get(str(key))
    return dict(value) if isinstance(value, Mapping) else {}


def _summary_enabled_or_implicit(summary: Mapping[str, Any]) -> bool:
    if "enabled" not in summary:
        return True
    return bool(summary.get("enabled"))


def _recommended_next_evidence(blockers: list[str]) -> str:
    if "stage_inventory_incomplete" in blockers:
        return "complete_missing_safe_promotion_stage"
    if "missing_all25_non_regression_evidence" in blockers:
        return "run_all25_non_regression_with_reference_profile"
    if "all25_non_regression_not_preserved" in blockers:
        return "repair_all25_regression_before_promotion"
    if (
        "missing_score_positive_counterfactual" in blockers
        or "missing_counterfactual_pair" in blockers
        or "counterfactual_planner_blocked" in blockers
    ):
        return "collect_score_positive_counterfactual_pair"
    if "all25_ready_for_planner_review_false" in blockers:
        return "repair_planner_review_readiness"
    return ""


def _status_is_missing_or_partial(status: str) -> bool:
    return str(status or "").startswith(("missing", "partial", "unclassified"))


__all__ = [
    "A_AXIS_PHASE37_ADAPTER_SCORE_BY_GAME",
    "SENSOR_FAMILIES_BY_GAME",
    "SENSOR_FAMILY_COMPLETED_STAGES",
    "SENSOR_FAMILY_REQUIRED_STAGE_OVERRIDES",
    "SENSOR_FAMILY_STATUS",
    "SENSOR_GAP_INVENTORY_SCHEMA_VERSION",
    "SENSOR_PROMOTION_BLOCKER_CONTRACT_KEYS_BY_FAMILY",
    "SENSOR_PROMOTION_BLOCKER_INVENTORY_SCHEMA_VERSION",
    "SENSOR_PROMOTION_GAP_INVENTORY_SCHEMA_VERSION",
    "SENSOR_PROMOTION_REQUIRED_STAGES",
    "PARALLEL_SENSOR_SUBSTRATE_COMPLETED_STAGES",
    "PARALLEL_SENSOR_SUBSTRATE_FAMILIES_BY_GAME",
    "PARALLEL_SENSOR_SUBSTRATE_GAP_INVENTORY_SCHEMA_VERSION",
    "PARALLEL_SENSOR_SUBSTRATE_REQUIRED_STAGE_OVERRIDES",
    "PARALLEL_SENSOR_SUBSTRATE_REQUIRED_STAGES",
    "PARALLEL_SENSOR_SUBSTRATE_STATUS",
    "build_phase37_to_current_sensor_gap_inventory",
    "build_phase37_to_current_parallel_sensor_substrate_gap_inventory",
    "build_phase37_to_current_sensor_promotion_gap_inventory",
    "build_parallel_sensor_substrate_gap_inventory",
    "build_sensor_gap_inventory",
    "build_sensor_promotion_blocker_inventory",
    "build_sensor_promotion_gap_inventory",
]
