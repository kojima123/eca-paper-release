"""Primitive scope registry for ECA generalisation accounting.

The registry is deliberately separate from primitive execution. It records how
much prior game/layout knowledge a primitive is allowed to carry so benchmark
summaries can distinguish score-through from adapter-free generalisation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable


SCOPE_GENERIC = "generic"
SCOPE_FRAME_CONDITIONED_GENERIC = "frame_conditioned_generic"
SCOPE_FRAME_CONDITIONED_GAME = "frame_conditioned_game"
SCOPE_PERCEPTION_DRIVEN_SOLVER = "perception_driven_solver"
SCOPE_HARDCODED_ADAPTER = "hardcoded_adapter"
SCOPE_UNKNOWN = "unknown"

GAME_CONDITIONED_SCOPES = frozenset(
    {
        SCOPE_FRAME_CONDITIONED_GAME,
        SCOPE_PERCEPTION_DRIVEN_SOLVER,
        SCOPE_HARDCODED_ADAPTER,
    }
)
ADAPTER_FREE_SCOPES = frozenset({SCOPE_GENERIC, SCOPE_FRAME_CONDITIONED_GENERIC})


@dataclass(frozen=True)
class PrimitiveScopeRecord:
    primitive_id: str
    scope: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRIMITIVE_SCOPE_REGISTRY: dict[str, PrimitiveScopeRecord] = {
    "action_loop_probe": PrimitiveScopeRecord(
        "action_loop_probe",
        SCOPE_GENERIC,
        "blind action-effect evidence collection",
    ),
    "grid_bfs": PrimitiveScopeRecord(
        "grid_bfs",
        SCOPE_GENERIC,
        "requires perceived actor, obstacle, and target-like evidence only",
    ),
    "click_toggle_search": PrimitiveScopeRecord(
        "click_toggle_search",
        SCOPE_GENERIC,
        "requires clickable regions and visible toggle effects only",
    ),
    "click_affordance_probe": PrimitiveScopeRecord(
        "click_affordance_probe",
        SCOPE_GENERIC,
        "probes visible click effects without game id dispatch",
    ),
    "generic_force_field_merge_containment": PrimitiveScopeRecord(
        "generic_force_field_merge_containment",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible small-marker merge products and compact containment geometry without game id dispatch",
    ),
    "finite_state_action_search": PrimitiveScopeRecord(
        "finite_state_action_search",
        SCOPE_GENERIC,
        "searches visible state hash transitions",
    ),
    "score_delta_frontier_search": PrimitiveScopeRecord(
        "score_delta_frontier_search",
        SCOPE_GENERIC,
        "follows observed score-bearing transitions",
    ),
    "goal_bearing_shortplan": PrimitiveScopeRecord(
        "goal_bearing_shortplan",
        SCOPE_GENERIC,
        "continues a locally inferred goal-bearing route without game id dispatch",
    ),
    "overlay_matcher": PrimitiveScopeRecord(
        "overlay_matcher",
        SCOPE_GENERIC,
        "requires visible overlay/template evidence",
    ),
    "select_push_probe": PrimitiveScopeRecord(
        "select_push_probe",
        SCOPE_GENERIC,
        "action-effect hint for push-like manipulation, not a game solver",
    ),
    "segment_transport_probe": PrimitiveScopeRecord(
        "segment_transport_probe",
        SCOPE_GENERIC,
        "uses repeated frame-only segment motion to bind a transport candidate",
    ),
    "attach_transport_probe": PrimitiveScopeRecord(
        "attach_transport_probe",
        SCOPE_GENERIC,
        "action-effect hint for attach/transport mechanics, not a game solver",
    ),
    "mirror_joint_cursor_bfs": PrimitiveScopeRecord(
        "mirror_joint_cursor_bfs",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible joint-cursor symmetry without game id dispatch",
    ),
    "mirror_selectable_blocker_reposition": PrimitiveScopeRecord(
        "mirror_selectable_blocker_reposition",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible selectable blocker cells to restore joint-cursor reachability without game id dispatch",
    ),
    "mirror_color_lock_target_planner": PrimitiveScopeRecord(
        "mirror_color_lock_target_planner",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible same-colour lock bars and target cells to route joint cursors without game id dispatch",
    ),
    "mirror_color_lock_blocker_planner": PrimitiveScopeRecord(
        "mirror_color_lock_blocker_planner",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible same-colour temporary lock gates and selectable blockers to route joint cursors without game id dispatch",
    ),
    "axis_reflection_coverage_search": PrimitiveScopeRecord(
        "axis_reflection_coverage_search",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible axis reflection, cursor footprint, and target coverage without game id dispatch",
    ),
    "shape_coverage_paste_solver": PrimitiveScopeRecord(
        "shape_coverage_paste_solver",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible shape pixels and same-colour target-marker coverage without game id dispatch",
    ),
    "shape_recolor_coverage_solver": PrimitiveScopeRecord(
        "shape_recolor_coverage_solver",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible shape pixels, palette pads, and target-marker coverage without game id dispatch",
    ),
    "sand_filter_flow_solver": PrimitiveScopeRecord(
        "sand_filter_flow_solver",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible source/filter/bowl grammar and bounded flow simulation without game id dispatch",
    ),
    "paint_region_reference_solver": PrimitiveScopeRecord(
        "paint_region_reference_solver",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible reference canvas, swatches, and region paint masks without game id dispatch",
    ),
    "slot_pool_sequence_solver": PrimitiveScopeRecord(
        "slot_pool_sequence_solver",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible target sequence, slot row, and bottom color pool without game id dispatch",
    ),
    "mechanic_hypothesis_cycle_executor": PrimitiveScopeRecord(
        "mechanic_hypothesis_cycle_executor",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "executes a short plan compiled from an induced visible cycle-state hypothesis",
    ),
    "mechanic_hypothesis_goal_candidate_plan": PrimitiveScopeRecord(
        "mechanic_hypothesis_goal_candidate_plan",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "executes bounded interventions compiled from confirmed mechanic-hypothesis goal candidates",
    ),
    "coordinate_hypothesis_choice": PrimitiveScopeRecord(
        "coordinate_hypothesis_choice",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "executes an ECA-verified coordinate-relation candidate derived from sensor packets",
    ),
    "coordinate_hypothesis_plan_lease": PrimitiveScopeRecord(
        "coordinate_hypothesis_plan_lease",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "continues an ECA-owned coordinate-relation protected plan without direct LLM actions",
    ),
    "coordinate_hypothesis_target_commitment": PrimitiveScopeRecord(
        "coordinate_hypothesis_target_commitment",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "continues an ECA-observed coordinate target after verified progress without direct LLM actions",
    ),
    "coordinate_hypothesis_terminal_local_verifier": PrimitiveScopeRecord(
        "coordinate_hypothesis_terminal_local_verifier",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses the ECA coordinate verifier after provider budget is exhausted without another LLM call",
    ),
    "stride_marker_maze_bfs": PrimitiveScopeRecord(
        "stride_marker_maze_bfs",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible stride/marker relation without game id dispatch",
    ),
    "generic_stride_marker_absorber_bfs": PrimitiveScopeRecord(
        "generic_stride_marker_absorber_bfs",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible active-marker, blocker-marker dots, and target geometry without game id dispatch",
    ),
    "generic_stride_marker_coupled_motion_bfs": PrimitiveScopeRecord(
        "generic_stride_marker_coupled_motion_bfs",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible rare-dot marker motion and moving/dormant marker hazards without game id dispatch",
    ),
    "click_context_recovery": PrimitiveScopeRecord(
        "click_context_recovery",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "returns to a visible deferred click-effect context",
    ),
    "button_cycle_rotation_solver": PrimitiveScopeRecord(
        "button_cycle_rotation_solver",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible button-cycle slots, current token, target token, and click directions without game id dispatch",
    ),
    "click_opened_region_navigation": PrimitiveScopeRecord(
        "click_opened_region_navigation",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "routes through visible regions opened by prior click effects without game id dispatch",
    ),
    "color_cycle_constraint_solver": PrimitiveScopeRecord(
        "color_cycle_constraint_solver",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "solves visible palette-cycle template constraints without game id dispatch",
    ),
    "button_cycle_rotation_solver": PrimitiveScopeRecord(
        "button_cycle_rotation_solver",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "solves visible single-loop button-cycle target relations without game id dispatch",
    ),
    "rule_table_rotation_solver": PrimitiveScopeRecord(
        "rule_table_rotation_solver",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "solves visible rule-table glyph rotation constraints without game id dispatch",
    ),
    "generic_attach_pair_transport_bfs": PrimitiveScopeRecord(
        "generic_attach_pair_transport_bfs",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible attach-pair continuity without per-game dispatch",
    ),
    "generic_select_push_cover_bfs": PrimitiveScopeRecord(
        "generic_select_push_cover_bfs",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible select-push-cover relation without per-game dispatch",
    ),
    "generic_slider_growth_bfs": PrimitiveScopeRecord(
        "generic_slider_growth_bfs",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible slider boxes, marker stripes, carried targets, and bounded grow-count BFS without game id dispatch",
    ),
    "generic_shape_transform_tile_bfs": PrimitiveScopeRecord(
        "generic_shape_transform_tile_bfs",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible carried shape mask, goal shape mask, transform tiles, and bounded route-phase BFS without game id dispatch",
    ),
    "generic_chain_sequence_bfs": PrimitiveScopeRecord(
        "generic_chain_sequence_bfs",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible active chain, movable colour blocks, target hit sequence, and bounded sequence-matching BFS without game id dispatch",
    ),
    "generic_peg_solitaire_bfs": PrimitiveScopeRecord(
        "generic_peg_solitaire_bfs",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible peg/empty/pivot/rail lattice structure and bounded peg-count BFS without game id dispatch",
    ),
    "generic_side_fall_bfs": PrimitiveScopeRecord(
        "generic_side_fall_bfs",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses source-free reset/replay side-fall discovery plus visible destructible/hazard lattice follow-up without game id dispatch",
    ),
    "generic_configure_cast_bfs": PrimitiveScopeRecord(
        "generic_configure_cast_bfs",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses source-free configure-then-cast discovery and drains a score-confirmed prefix without game id dispatch",
    ),
    "generic_button_press_bfs": PrimitiveScopeRecord(
        "generic_button_press_bfs",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses source-free button-press discovery and bounded post-score follow-up without game id dispatch",
    ),
    "generic_canvas_centroid_bootstrap": PrimitiveScopeRecord(
        "generic_canvas_centroid_bootstrap",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible canvas/piece/target centroid grammar and RESET+replay score confirmation without adapter dispatch",
    ),
    "sprite_anchor_overlap_solver": PrimitiveScopeRecord(
        "sprite_anchor_overlap_solver",
        SCOPE_FRAME_CONDITIONED_GENERIC,
        "uses visible sprite anchor pairs and bounded rotation/translation search without game id dispatch",
    ),
    "room_gate_policy": PrimitiveScopeRecord(
        "room_gate_policy",
        SCOPE_FRAME_CONDITIONED_GAME,
        "room/gate policy currently tuned to dc22-like layouts",
    ),
    "attach_pair_fill_bfs": PrimitiveScopeRecord(
        "attach_pair_fill_bfs",
        SCOPE_FRAME_CONDITIONED_GAME,
        "attach/fill policy currently tuned to wa30-like mechanics",
    ),
    "select_push_cover_bfs": PrimitiveScopeRecord(
        "select_push_cover_bfs",
        SCOPE_FRAME_CONDITIONED_GAME,
        "select/push/cover policy currently tuned to ka59-like mechanics",
    ),
}


def primitive_scope(primitive_id: Any) -> str:
    primitive = str(primitive_id or "").strip()
    if not primitive:
        return SCOPE_UNKNOWN
    record = PRIMITIVE_SCOPE_REGISTRY.get(primitive)
    return record.scope if record is not None else SCOPE_UNKNOWN


def primitive_scope_records(primitive_ids: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for primitive_id in primitive_ids:
        primitive = str(primitive_id or "").strip()
        if not primitive:
            continue
        record = PRIMITIVE_SCOPE_REGISTRY.get(primitive)
        if record is None:
            out.append(
                PrimitiveScopeRecord(
                    primitive,
                    SCOPE_UNKNOWN,
                    "primitive is not registered; treat as unknown for claims",
                ).to_dict()
            )
        else:
            out.append(record.to_dict())
    return out


def primitive_scope_counts(primitive_ids: Iterable[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for primitive_id in primitive_ids:
        scope = primitive_scope(primitive_id)
        if scope:
            counts[scope] += 1
    return dict(sorted(counts.items()))


def uses_game_conditioned_scope(primitive_ids: Iterable[Any]) -> bool:
    return any(primitive_scope(primitive_id) in GAME_CONDITIONED_SCOPES for primitive_id in primitive_ids)


def is_adapter_free_scope(primitive_ids: Iterable[Any]) -> bool:
    scopes = [primitive_scope(primitive_id) for primitive_id in primitive_ids if str(primitive_id or "").strip()]
    return bool(scopes) and all(scope in ADAPTER_FREE_SCOPES for scope in scopes)


def generalisation_tier_for_primitives(primitive_ids: Iterable[Any]) -> str:
    primitive_list = [str(primitive_id or "").strip() for primitive_id in primitive_ids]
    primitive_list = [primitive_id for primitive_id in primitive_list if primitive_id]
    if not primitive_list:
        return "action_effect_probe_only"
    scopes = {primitive_scope(primitive_id) for primitive_id in primitive_list}
    if scopes & GAME_CONDITIONED_SCOPES:
        return "frame_conditioned_game_primitive"
    if scopes <= {SCOPE_GENERIC}:
        return "adapter_free_generic_primitive"
    if scopes <= ADAPTER_FREE_SCOPES:
        return "adapter_free_frame_conditioned_primitive"
    return "unknown"


__all__ = [
    "ADAPTER_FREE_SCOPES",
    "GAME_CONDITIONED_SCOPES",
    "PRIMITIVE_SCOPE_REGISTRY",
    "PrimitiveScopeRecord",
    "SCOPE_FRAME_CONDITIONED_GAME",
    "SCOPE_FRAME_CONDITIONED_GENERIC",
    "SCOPE_GENERIC",
    "SCOPE_HARDCODED_ADAPTER",
    "SCOPE_PERCEPTION_DRIVEN_SOLVER",
    "SCOPE_UNKNOWN",
    "generalisation_tier_for_primitives",
    "is_adapter_free_scope",
    "primitive_scope",
    "primitive_scope_counts",
    "primitive_scope_records",
    "uses_game_conditioned_scope",
]
