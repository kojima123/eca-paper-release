"""Real-environment perturbation benchmark for frame-conditioned adapters.

The benchmark executes perception adapters only from public observations. It
never falls back to ``env._env._game`` for frames, so a missing frame is a
benchmark failure rather than an internal-state escape hatch.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import importlib
import json
import math
import os
import random
import time
from typing import Any, Iterable, Mapping

import numpy as np

from .primitive_scope_registry import (
    generalisation_tier_for_primitives,
    primitive_scope_counts,
    primitive_scope_records,
    uses_game_conditioned_scope,
)
from .actor_identity_anchor import (
    summarize_actor_identity_anchor_packets,
    summarize_actor_identity_anchor_guarded_planner_connection,
    summarize_actor_identity_anchor_posterior,
)
from .component_extent_growth_sensor import (
    COMPONENT_EXTENT_GROWTH_PACKET_SCHEMA_VERSION,
    summarize_component_extent_growth_packets,
    summarize_component_extent_growth_precision_posterior,
    summarize_component_extent_growth_slider_stride_posterior,
    summarize_component_extent_growth_target_distance_posterior,
)
from .sensor_gap_inventory import (
    build_phase37_to_current_parallel_sensor_substrate_gap_inventory,
    build_phase37_to_current_sensor_gap_inventory,
    build_phase37_to_current_sensor_promotion_gap_inventory,
    build_sensor_promotion_blocker_inventory,
)
from .force_field_merge_containment_sensor import (
    FORCE_FIELD_MERGE_CONTAINMENT_PACKET_SCHEMA_VERSION,
    summarize_force_field_containment_score_target_relations,
    summarize_force_field_score_positive_counterfactual_pair_search_candidates,
    summarize_force_field_merge_containment_all25_non_regression,
    summarize_force_field_merge_containment_candidate_miner,
    summarize_force_field_merge_containment_counterfactual_controls,
    summarize_force_field_merge_containment_guarded_probe_outcomes,
    summarize_force_field_merge_containment_packets,
    summarize_force_field_merge_containment_posterior,
    summarize_force_field_merge_containment_score_target_gap,
)
from .invisible_effect_position_memory import (
    summarize_invisible_effect_position_memory_guarded_planner_connection,
    summarize_invisible_effect_position_memory_packets,
    summarize_invisible_effect_position_memory_posterior,
)
from .multi_body_contact_topology_sensor import (
    MULTI_BODY_CONTACT_TOPOLOGY_PACKET_SCHEMA_VERSION,
    summarize_multi_body_contact_topology_candidates,
    summarize_multi_body_contact_topology_counterfactual_controls,
    summarize_multi_body_contact_topology_guarded_probe_outcomes,
    summarize_multi_body_contact_topology_posterior,
    summarize_multi_body_contact_topology_packets,
)
from .moving_sprite_tracker import (
    summarize_moving_sprite_coupled_pair_goal_effect_bridge,
    summarize_moving_sprite_coupled_pair_goal_score_attribution_guard,
    summarize_moving_sprite_coupled_motion_symmetry_sensor,
    summarize_moving_sprite_effect_component_spatial_stability_sensor,
    summarize_moving_sprite_goal_anchor_perception_sensor,
    summarize_moving_sprite_guarded_planner_connection_sensor,
    summarize_moving_sprite_target_bound_effect_component_sensor,
    summarize_moving_sprite_target_goal_counterfactual_attribution,
    summarize_moving_sprite_target_goal_binding_state_sensor,
    summarize_moving_sprite_actor_ambiguity_splitter,
    summarize_moving_sprite_guarded_actor_anchor_posterior,
    summarize_moving_sprite_guarded_planner_readiness,
    summarize_moving_sprite_motion_posterior,
    summarize_moving_sprite_sparse_chain_rail_state_sensor,
    summarize_moving_sprite_soft_anchor_motion_posterior,
    MOVING_SPRITE_RUNTIME_TRACE_PACKET_SCHEMA_VERSION,
    summarize_moving_sprite_runtime_trace_packets,
)
from .mechanic_certificate import (
    MECHANIC_CERTIFICATE_RUNTIME_TRACE_PACKET_SCHEMA_VERSION,
    summarize_mechanic_certificate_guarded_planner_connection,
    summarize_mechanic_certificate_posterior,
    summarize_mechanic_certificate_runtime_trace_packets,
)
from .palette_invariant_perception import (
    PALETTE_INVARIANT_RUNTIME_TRACE_PACKET_SCHEMA_VERSION,
    summarize_palette_invariant_guarded_planner_connection,
    summarize_palette_invariant_posterior,
    summarize_palette_invariant_runtime_trace_packets,
)
from .platform_destructible_topology_sensor import (
    PLATFORM_DESTRUCTIBLE_TOPOLOGY_PACKET_SCHEMA_VERSION,
    summarize_platform_destructible_topology_all25_non_regression,
    summarize_platform_destructible_topology_candidate_miner,
    summarize_platform_destructible_topology_counterfactual_controls,
    summarize_platform_destructible_topology_guarded_probe_outcomes,
    summarize_platform_destructible_topology_packets,
    summarize_platform_destructible_topology_posterior,
)
from .program_row_semantics_sensor import (
    PROGRAM_ROW_SEMANTICS_PACKET_SCHEMA_VERSION,
    summarize_program_row_semantics_all25_non_regression,
    summarize_program_row_semantics_candidate_miner,
    summarize_program_row_semantics_counterfactual_controls,
    summarize_program_row_semantics_guarded_probe_outcomes,
    summarize_program_row_semantics_posterior,
    summarize_program_row_semantics_packets,
)
from .program_spell_pattern_sensor import (
    PROGRAM_SPELL_PATTERN_PACKET_SCHEMA_VERSION,
    summarize_program_spell_pattern_all25_non_regression,
    summarize_program_spell_pattern_candidate_miner,
    summarize_program_spell_pattern_counterfactual_controls,
    summarize_program_spell_pattern_guarded_probe_outcomes,
    summarize_program_spell_pattern_posterior,
    summarize_program_spell_pattern_packets,
)
from .round_memory_sensor import (
    ROUND_MEMORY_PACKET_SCHEMA_VERSION,
    summarize_round_memory_posterior,
    summarize_round_memory_packets,
)
from .slot_pool_coverage_sensor import (
    SLOT_POOL_COVERAGE_PACKET_SCHEMA_VERSION,
    SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION,
    build_slot_pool_coverage_posterior_packet,
    summarize_slot_pool_coverage_already_scored_event_exclusion,
    summarize_slot_pool_coverage_candidates,
    summarize_slot_pool_coverage_counterfactual_controls,
    summarize_slot_pool_coverage_effect_intent_controls,
    summarize_slot_pool_coverage_exact_slot_scoreless_controls,
    summarize_slot_pool_coverage_focused_replay_execution,
    summarize_slot_pool_coverage_focused_replay_outcome_admission,
    summarize_slot_pool_coverage_focused_replay_readiness,
    summarize_slot_pool_coverage_focused_score_bearing_candidate_discovery,
    summarize_slot_pool_coverage_guarded_probe_outcomes,
    summarize_slot_pool_coverage_intervention_contrast_attribution,
    summarize_slot_pool_coverage_intervention_temporal_counterfactual,
    summarize_slot_pool_coverage_negative_memory_guarded_review,
    summarize_slot_pool_coverage_negative_memory_ranking_shadow,
    summarize_slot_pool_coverage_posterior,
    summarize_slot_pool_coverage_packets,
    summarize_slot_pool_coverage_region_identity_controls,
    summarize_slot_pool_coverage_scoped_scoreless_controls,
    summarize_slot_pool_coverage_score_event_scope_attribution,
    summarize_slot_pool_coverage_scoreless_progress_controls,
    summarize_slot_pool_coverage_submit_precondition_score,
    summarize_slot_pool_coverage_submit_precondition_breadth_non_regression,
    summarize_slot_pool_coverage_submit_precondition_live_rank_guard_focused_runtime,
    summarize_slot_pool_coverage_submit_precondition_rank_guard_runtime_guard,
    summarize_slot_pool_coverage_submit_precondition_rank_guard_shadow,
    summarize_slot_pool_coverage_submit_precondition_runtime_bridge_guard,
    summarize_slot_pool_coverage_submit_precondition_runtime_bridge_outcome,
    summarize_slot_pool_coverage_target_slot_scoreless_controls,
    summarize_slot_pool_coverage_terminal_effect_score_closure_attribution,
)
from .route_validity_sensor import (
    ROUTE_VALIDITY_PACKET_SCHEMA_VERSION,
    summarize_route_validity_all25_non_regression,
    summarize_route_validity_progress_posterior,
    summarize_route_validity_recovery_candidates,
    summarize_route_validity_packets,
)
from .structural_rods import RodPromotionPolicy, default_rod_promotion_policy
from .emergent_pattern_sensor import summarize_emergent_rod_pattern_mining
from .transition_state_variable_sensor import (
    TRANSITION_STATE_VARIABLE_PACKET_SCHEMA_VERSION,
    summarize_transition_state_variable_guarded_planner_connection,
    summarize_transition_state_variable_score_closure_packets,
    summarize_transition_state_variable_score_retest_candidates,
    summarize_transition_state_variable_packets,
    transition_state_variable_transition_key_signature,
)
from .tile_effect_learner import (
    summarize_tile_effect_family_scope_negative_control,
    summarize_tile_effect_hazard_source_repair_bridge,
    summarize_tile_effect_model_repair_proposal,
    summarize_tile_effect_model_repair_protected_probe_live_rank_guard_action_selection_guard,
    summarize_tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime,
    summarize_tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_admission,
    summarize_tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution,
    summarize_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery,
    summarize_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_focused_runtime_replay,
    summarize_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_positive_rank_promotion_guard,
    summarize_tile_effect_model_repair_protected_probe_execution_gate,
    summarize_tile_effect_model_repair_protected_probe_outcome_posterior,
    summarize_tile_effect_model_repair_protected_probe_rank_guard_runtime_guard,
    summarize_tile_effect_model_repair_protected_probe_rank_guard_shadow,
    summarize_tile_effect_model_repair_protected_probe_runtime_bridge_guard,
    summarize_tile_effect_model_repair_protected_probe_runtime_bridge_outcome,
    summarize_tile_effect_model_repair_protected_probe_runtime_executor,
    summarize_tile_effect_model_repair_runtime_probe_proposal,
    summarize_tile_effect_model_repair_write_audit,
    summarize_tile_effect_model_repair_write_guard,
    TILE_EFFECT_RUNTIME_TRACE_PACKET_SCHEMA_VERSION,
    summarize_tile_effect_guard_counterfactual_control,
    summarize_tile_effect_guard_precision_review,
    summarize_tile_effect_guarded_planner_connection,
    summarize_tile_effect_posterior,
    summarize_tile_effect_runtime_trace_packets,
)
from .transport_effect_learner import (
    summarize_transport_effect_guarded_planner_connection,
    summarize_transport_effect_posterior,
    summarize_transport_effect_runtime_trace_packets,
)


PERCEPTION_ADAPTER_REGISTRY = {
    "wa30": "arcgym.adapters.wa30_perception_driven",
    "tu93": "arcgym.adapters.tu93_perception_driven",
    "m0r0": "arcgym.adapters.m0r0_perception_driven",
    "ka59": "arcgym.adapters.ka59_perception_driven",
    "r11l": "arcgym.adapters.r11l_perception_driven",
    "s5i5": "arcgym.adapters.s5i5_perception_driven",
    "tn36": "arcgym.adapters.tn36_perception_driven",
    "cn04": "arcgym.adapters.cn04_perception_driven",
    "ar25": "arcgym.adapters.ar25_perception_driven",
    "su15": "arcgym.adapters.su15_perception_driven",
    "tr87": "arcgym.adapters.tr87_perception_driven",
    "sc25": "arcgym.adapters.sc25_perception_driven",
    "sp80": "arcgym.adapters.sp80_perception_driven",
    "re86": "arcgym.adapters.re86_perception_driven",
    "dc22": "arcgym.adapters.dc22_perception_driven",
    "lf52": "arcgym.adapters.lf52_perception_driven",
    "bp35": "arcgym.adapters.bp35_perception_driven",
    "g50t": "arcgym.adapters.g50t_perception_driven",
    "sk48": "arcgym.adapters.sk48_perception_driven",
    "cd82": "arcgym.adapters.cd82_perception_driven",
    "ls20": "arcgym.adapters.ls20_perception_driven",
    "ft09": "arcgym.adapters.ft09_perception_driven",
    "vc33": "arcgym.adapters.vc33_perception_driven",
    "lp85": "arcgym.adapters.lp85_perception_driven",
    "sb26": "arcgym.adapters.sb26_perception_driven",
}

CORE_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6")
DEFAULT_VARIANTS = (
    "canonical_full_plan",
    "mid_state_replan",
    "random_prefix_recovery",
    "off_plan_injection",
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PerturbationSpec:
    variant: str
    prefix_length: int = 0
    injection_step: int = 1
    seed: int = 0


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _env_score_map(name: str) -> dict[str, int]:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): _as_int(value) for key, value in payload.items()}


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _normalize_action_name(action: Any) -> str:
    name = getattr(action, "name", action)
    text = str(name or "").strip().upper()
    if text.startswith("GAMEACTION."):
        text = text.split(".", 1)[1]
    if text.isdigit():
        return f"ACTION{int(text)}"
    return text


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def frame_from_observation(observation: Mapping[str, Any]) -> np.ndarray:
    frame = observation.get("frame")
    if frame is None:
        raise RuntimeError("frame-only benchmark requires observation['frame']")
    if isinstance(frame, list) and frame:
        frame = frame[-1]
    return np.asarray(frame)


def available_action_names(observation: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for raw in list(observation.get("available_actions") or []):
        name = _normalize_action_name(raw)
        if name and name not in names:
            names.append(name)
    if not names:
        names = list(CORE_ACTIONS)
    return [name for name in names if name in CORE_ACTIONS]


def _payload_for_action_name(action_name: str, frame: np.ndarray, *, reason: str) -> dict[str, Any]:
    from arcengine import GameAction

    action_name = _normalize_action_name(action_name)
    payload: dict[str, Any] = {
        "action": GameAction.from_name(action_name),
        "reasoning": reason,
    }
    if action_name == "ACTION6":
        height = int(frame.shape[0]) if frame.ndim >= 1 else 1
        width = int(frame.shape[1]) if frame.ndim >= 2 else height
        payload.update({"x": width // 2, "y": height // 2})
    return payload


def _plan_entry_payload(entry: Any, frame: np.ndarray, *, reason: str) -> dict[str, Any]:
    from arcengine import GameAction

    if isinstance(entry, tuple):
        action, data = entry
    else:
        action, data = entry, {}
    action_name = _normalize_action_name(action)
    if isinstance(action, str):
        action = GameAction.from_name(action_name)
    payload = {"action": action, "reasoning": reason, **dict(data or {})}
    if action_name == "ACTION6" and ("x" not in payload or "y" not in payload):
        height = int(frame.shape[0]) if frame.ndim >= 1 else 1
        width = int(frame.shape[1]) if frame.ndim >= 2 else height
        payload.update({"x": width // 2, "y": height // 2})
    return payload


def _payload_action_name(payload: Mapping[str, Any]) -> str:
    return _normalize_action_name(payload.get("action"))


def choose_off_plan_action(
    available: Iterable[str],
    *,
    suggested_action: str,
    rng: random.Random,
) -> str:
    ordered = [_normalize_action_name(action) for action in available]
    ordered = [action for action in ordered if action in CORE_ACTIONS]
    alternatives = [action for action in ordered if action != _normalize_action_name(suggested_action)]
    if alternatives:
        return rng.choice(alternatives)
    if ordered:
        return ordered[0]
    return "ACTION5"


class _PerceptionPlanner:
    def __init__(
        self,
        game_id: str,
        *,
        enable_plan_state_cache: bool | None = None,
    ) -> None:
        self.game_id = game_id
        module_name = PERCEPTION_ADAPTER_REGISTRY[game_id]
        self.module = importlib.import_module(module_name)
        if hasattr(self.module, "PLAN_SOURCE_HISTORY"):
            self.module.PLAN_SOURCE_HISTORY.clear()
        if hasattr(self.module, "LAST_PLAN_SOURCE"):
            self.module.LAST_PLAN_SOURCE = ""
        if hasattr(self.module, "TARGET_CACHE"):
            self.module.TARGET_CACHE.clear()
        if hasattr(self.module, "PIECE_CACHE"):
            self.module.PIECE_CACHE.clear()
        if hasattr(self.module, "PLAN_QUEUE"):
            self.module.PLAN_QUEUE.clear()
        if hasattr(self.module, "PLAN_QUEUE_KEY"):
            self.module.PLAN_QUEUE_KEY = None
        self.fz_cache = None
        self.last_level_proxy = -1
        self.plan_state_cache_enabled = (
            _env_flag("ECA_ENABLE_ADAPTER_PLAN_STATE_CACHE", False)
            if enable_plan_state_cache is None
            else bool(enable_plan_state_cache)
        )
        # Plan-state cache: when adapter returns empty (level_signature
        # changed mid-execution), fall back to remaining actions from
        # the LAST non-empty plan. Mirrors the LocalController
        # ADAPTER_HANDOFF / plan-state cache mechanism (CLAUDE.md
        # Phase 49-65 "ECA-driven progression").
        self._cached_plan: list[Any] = []
        self._cached_plan_source: str = ""
        self._cached_plan_consumed: int = 0
        self._last_observed_score: int = -1
        self._cache_sequence_id: int = 0
        self._pending_cache_invalidated: bool = False
        self._pending_invalidation_reason: str = ""
        self._shadow_game: Any | None = None
        self._shadow_plan_source: str = ""
        self._shadow_replay_error: str = ""
        self._shadow_tail_start_hint: int = -1
        self.last_plan_cache_event: dict[str, Any] = self._plan_cache_event(
            reason="initial"
        )

    def _plan_cache_event(self, **updates: Any) -> dict[str, Any]:
        event: dict[str, Any] = {
            "enabled": bool(self.plan_state_cache_enabled),
            "reason": "",
            "source": self.last_plan_source,
            "fresh_plan_len": 0,
            "cached_plan_len": len(self._cached_plan),
            "cached_plan_index": int(self._cached_plan_consumed),
            "cache_sequence_id": int(self._cache_sequence_id),
            "cache_hit": False,
            "cache_refreshed": False,
            "cache_invalidated": False,
            "cache_action_available": False,
            "cache_rejected": False,
            "cache_peek": False,
            "cache_continuation_length": max(
                0, len(self._cached_plan) - int(self._cached_plan_consumed)
            ),
            "invalidation_reason": "",
            "shadow_replay_available": bool(self._shadow_game is not None),
            "shadow_replay_error": self._shadow_replay_error,
        }
        event.update(_json_safe(updates))
        return event

    def _maybe_refresh_shadow_from_fresh_plan(self, fresh_plan: list[Any]) -> None:
        if not fresh_plan:
            return
        factory = getattr(self.module, "make_shadow_game_from_last_perception", None)
        if not callable(factory):
            return
        try:
            shadow_game = factory()
        except Exception as exc:
            self._shadow_replay_error = repr(exc)
            return
        if shadow_game is None:
            return
        self._shadow_game = shadow_game
        self._shadow_plan_source = str(self.last_plan_source or "")
        self._shadow_replay_error = ""

    def _replay_shadow_action(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        replay = getattr(self.module, "replay_action_on_shadow_game", None)
        if not callable(replay):
            return {}
        if self._shadow_game is None:
            return {"applied": False, "reason": "missing_shadow_game"}
        action_name = _payload_action_name(dict(payload or {}))
        try:
            event = dict(replay(self._shadow_game, action_name, dict(payload or {})) or {})
        except Exception as exc:
            event = {"applied": False, "reason": "replay_error", "error": repr(exc)}
        if not bool(event.get("applied")):
            self._shadow_replay_error = str(event.get("error") or event.get("reason") or "")
        else:
            self._shadow_replay_error = ""
        return event

    def _plan_from_shadow(self) -> list[Any]:
        if self._shadow_game is None:
            return []
        planner = getattr(self.module, "plan_next_actions_from_shadow_game", None)
        if not callable(planner):
            return []
        try:
            setattr(self._shadow_game, "_eca_tail_start_hint", int(self._shadow_tail_start_hint))
        except Exception:
            pass
        try:
            plan = list(planner(self._shadow_game) or [])
        except Exception as exc:
            self._shadow_replay_error = repr(exc)
            return []
        if plan:
            self._shadow_plan_source = self.last_plan_source
            self._shadow_replay_error = ""
        return plan

    def _refresh_cache_from_shadow(
        self,
        *,
        invalidated: bool,
        invalidation_reason: str,
    ) -> list[Any]:
        shadow_plan = self._plan_from_shadow()
        shadow_source = self.last_plan_source
        if not shadow_plan:
            return []
        self._cache_sequence_id += 1
        self._cached_plan = list(shadow_plan)
        self._cached_plan_consumed = 0
        self._cached_plan_source = shadow_source
        self.last_plan_cache_event = self._plan_cache_event(
            reason="shadow_replay_refresh",
            source=shadow_source,
            fresh_plan_len=len(shadow_plan),
            cached_plan_index=0,
            cache_refreshed=True,
            cache_action_available=True,
            cache_peek=True,
            cache_continuation_length=len(shadow_plan),
            cache_invalidated=invalidated,
            invalidation_reason=invalidation_reason,
            shadow_replay_available=True,
        )
        return list(self._cached_plan)

    def commit_cached_plan_action(
        self,
        *,
        cache_sequence_id: Any,
        cached_plan_index: Any,
        executed_match: bool,
        executed_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "enabled": bool(self.plan_state_cache_enabled),
            "requested": True,
            "committed": False,
            "reason": "",
            "cache_sequence_id": int(self._cache_sequence_id),
            "requested_cache_sequence_id": _as_int(cache_sequence_id, -1),
            "cached_plan_index_before": int(self._cached_plan_consumed),
            "requested_cached_plan_index": _as_int(cached_plan_index, -1),
            "cached_plan_index_after": int(self._cached_plan_consumed),
            "shadow_replay": {},
        }
        if not self.plan_state_cache_enabled:
            event["reason"] = "disabled"
            return event
        event["shadow_replay"] = self._replay_shadow_action(executed_payload)
        if not bool(executed_match):
            self._shadow_tail_start_hint = _as_int(cached_plan_index, -1)
            event["reason"] = "executed_action_mismatch"
            return event
        requested_sequence = _as_int(cache_sequence_id, -1)
        requested_index = _as_int(cached_plan_index, -1)
        if requested_sequence != int(self._cache_sequence_id):
            event["reason"] = "cache_sequence_mismatch"
            return event
        if requested_index < 0:
            event["reason"] = "missing_cache_index"
            return event
        current_index = int(self._cached_plan_consumed)
        if requested_index < current_index:
            event["reason"] = "already_committed"
            event["committed"] = True
            return event
        if requested_index != current_index:
            event["reason"] = "stale_cache_index"
            return event
        self._cached_plan_consumed = min(len(self._cached_plan), requested_index + 1)
        self._shadow_tail_start_hint = int(self._cached_plan_consumed)
        event["committed"] = True
        event["reason"] = "committed"
        event["cached_plan_index_after"] = int(self._cached_plan_consumed)
        return event

    def invalidate_plan_cache(self, reason: str = "external_boundary") -> dict[str, Any]:
        self._cached_plan = []
        self._cached_plan_source = ""
        self._cached_plan_consumed = 0
        self._pending_cache_invalidated = True
        self._pending_invalidation_reason = str(reason or "external_boundary")
        self.last_plan_cache_event = self._plan_cache_event(
            reason="manual_invalidation",
            cache_invalidated=True,
            cache_rejected=True,
            invalidation_reason=self._pending_invalidation_reason,
        )
        return dict(self.last_plan_cache_event)

    def plan(self, frame: np.ndarray, observation: Mapping[str, Any]) -> list[Any]:
        if self.game_id == "wa30":
            level_proxy = _as_int(observation.get("score"), 0)
            if level_proxy != self.last_level_proxy:
                from arcgym.perception.wa30_perception import perceive_wa30

                state = perceive_wa30(frame)
                self.fz_cache = state["fill_zone_cells"]
                self.last_level_proxy = level_proxy
            fresh_plan = list(self.module.perceive_and_plan(frame, fill_zones_cache=self.fz_cache) or [])
        else:
            fresh_plan = list(self.module.perceive_and_plan(frame) or [])
        fresh_source = self.last_plan_source
        if not self.plan_state_cache_enabled:
            self.last_plan_cache_event = self._plan_cache_event(
                reason="disabled",
                source=fresh_source,
                fresh_plan_len=len(fresh_plan),
            )
            return fresh_plan
        # Reset cache when score changes (new level reached → adapter
        # should produce a new canonical plan for that level).
        current_score = _as_int(observation.get("score"), 0)
        invalidated = False
        invalidation_reason = ""
        if self._last_observed_score >= 0 and current_score != self._last_observed_score:
            self._cached_plan = []
            self._cached_plan_consumed = 0
            self._cached_plan_source = ""
            self._shadow_tail_start_hint = -1
            invalidated = True
            invalidation_reason = "score_changed"
        if self._pending_cache_invalidated:
            invalidated = True
            invalidation_reason = self._pending_invalidation_reason or "external_boundary"
            self._pending_cache_invalidated = False
            self._pending_invalidation_reason = ""
        self._last_observed_score = current_score
        if invalidation_reason == "off_plan_injection":
            shadow_plan = self._refresh_cache_from_shadow(
                invalidated=invalidated,
                invalidation_reason=invalidation_reason,
            )
            if shadow_plan:
                return shadow_plan
        shadow_remaining = self._cached_plan[self._cached_plan_consumed:]
        if self._cached_plan_source.startswith("shadow_repair_") and shadow_remaining:
            self.last_plan_cache_event = self._plan_cache_event(
                reason="shadow_repair_cache_hit",
                source=self._cached_plan_source,
                fresh_plan_len=len(fresh_plan),
                cached_plan_index=int(self._cached_plan_consumed),
                cache_hit=True,
                cache_action_available=True,
                cache_peek=True,
                cache_continuation_length=len(shadow_remaining),
                cache_invalidated=invalidated,
                invalidation_reason=invalidation_reason,
                shadow_replay_available=True,
            )
            return shadow_remaining
        # Fresh non-empty plan: refresh cache on a new source, otherwise
        # serve the unconsumed tail. Consumption is committed only after
        # the caller actually executes the advised action.
        if fresh_plan:
            if fresh_source != self._cached_plan_source or not self._cached_plan:
                self._cache_sequence_id += 1
                self._cached_plan = list(fresh_plan)
                self._cached_plan_consumed = 0
                self._cached_plan_source = fresh_source
                self._shadow_tail_start_hint = 0
                self._maybe_refresh_shadow_from_fresh_plan(fresh_plan)
                reason = "fresh_plan_refresh"
                refreshed = True
            else:
                reason = "fresh_plan_cache_peek"
                refreshed = False
            remaining = self._cached_plan[self._cached_plan_consumed :]
            if remaining:
                self.last_plan_cache_event = self._plan_cache_event(
                    reason=reason,
                    source=fresh_source,
                    fresh_plan_len=len(fresh_plan),
                    cached_plan_index=int(self._cached_plan_consumed),
                    cache_hit=not refreshed,
                    cache_refreshed=refreshed,
                    cache_action_available=True,
                    cache_peek=True,
                    cache_continuation_length=len(remaining),
                    cache_invalidated=invalidated,
                    invalidation_reason=invalidation_reason,
                )
                return remaining
            self.last_plan_cache_event = self._plan_cache_event(
                reason="fresh_plan_cache_exhausted",
                source=fresh_source,
                fresh_plan_len=len(fresh_plan),
                cache_rejected=True,
                cache_invalidated=invalidated,
                invalidation_reason=invalidation_reason,
            )
            return []
        # Fresh plan is empty: serve from cache if available.
        remaining = self._cached_plan[self._cached_plan_consumed:]
        if remaining:
            index_before = int(self._cached_plan_consumed)
            self.last_plan_cache_event = self._plan_cache_event(
                reason="cache_hit",
                source=self._cached_plan_source,
                fresh_plan_len=0,
                cached_plan_index=index_before,
                cache_hit=True,
                cache_action_available=True,
                cache_peek=True,
                cache_continuation_length=len(remaining),
                cache_invalidated=invalidated,
                invalidation_reason=invalidation_reason,
            )
            return remaining
        shadow_plan = self._refresh_cache_from_shadow(
            invalidated=invalidated,
            invalidation_reason=invalidation_reason,
        )
        if shadow_plan:
            return shadow_plan
        self.last_plan_cache_event = self._plan_cache_event(
            reason="cache_empty",
            source=fresh_source or self._cached_plan_source,
            fresh_plan_len=0,
            cache_rejected=True,
            cache_invalidated=invalidated,
            invalidation_reason=invalidation_reason,
        )
        return []

    @property
    def last_plan_source(self) -> str:
        return str(getattr(self.module, "LAST_PLAN_SOURCE", "") or "")


def _step_env(env: Any, payload: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    observation, _reward, done = env.step(dict(payload))
    return dict(observation), bool(done)


def _compact_trace_item(
    *,
    phase: str,
    step: int,
    observation: Mapping[str, Any],
    payload: Mapping[str, Any],
    plan_source: str = "",
) -> dict[str, Any]:
    return {
        "phase": phase,
        "step": step,
        "score_before": _as_int(observation.get("score"), 0),
        "state_before": str(observation.get("state", "") or ""),
        "action": _payload_action_name(payload),
        "plan_source": plan_source,
    }


def _run_prefix(
    *,
    env: Any,
    observation: dict[str, Any],
    rng: random.Random,
    prefix_length: int,
    max_actions: int,
    trace: list[dict[str, Any]],
) -> tuple[dict[str, Any], int, bool]:
    actions = 0
    done = False
    for _idx in range(max(0, prefix_length)):
        if actions >= max_actions:
            break
        frame = frame_from_observation(observation)
        action_name = rng.choice(available_action_names(observation))
        payload = _payload_for_action_name(
            action_name,
            frame,
            reason="eca_perturbation_random_prefix",
        )
        trace.append(
            _compact_trace_item(
                phase="random_prefix",
                step=actions,
                observation=observation,
                payload=payload,
            )
        )
        observation, done = _step_env(env, payload)
        actions += 1
        if done:
            break
    return observation, actions, done


def run_perception_perturbation_episode(
    game_id: str,
    spec: PerturbationSpec,
    *,
    max_actions: int = 80,
) -> dict[str, Any]:
    import arc_agi
    from arcgym.environments import ArcAgi3Env

    start = time.time()
    trace: list[dict[str, Any]] = []
    errors: list[str] = []
    rng = random.Random(spec.seed)
    planner = None
    env = None
    actions_taken = 0
    perturbation_score = 0
    injected = False
    done = False
    observation: dict[str, Any] = {}
    try:
        planner = _PerceptionPlanner(game_id)
        env = ArcAgi3Env(
            game_id=game_id,
            max_actions=max_actions,
            operation_mode=arc_agi.OperationMode.OFFLINE,
        )
        observation = dict(env.reset())

        if spec.variant == "random_prefix_recovery":
            observation, prefix_actions, done = _run_prefix(
                env=env,
                observation=observation,
                rng=rng,
                prefix_length=spec.prefix_length,
                max_actions=max_actions,
                trace=trace,
            )
            actions_taken += prefix_actions
            perturbation_score = _as_int(observation.get("score"), 0)
        else:
            perturbation_score = _as_int(observation.get("score"), 0)

        solver_steps = 0
        while actions_taken < max_actions and not done:
            frame = frame_from_observation(observation)
            plan = planner.plan(frame, observation)
            plan_source = planner.last_plan_source
            if not plan:
                break
            payloads = [
                _plan_entry_payload(
                    entry,
                    frame,
                    reason=f"eca_perturbation:{spec.variant}",
                )
                for entry in plan
            ]
            if spec.variant == "canonical_full_plan":
                selected_payloads = payloads
                phase = "canonical_plan"
            else:
                selected_payloads = [payloads[0]]
                phase = "step_replan"
                if (
                    spec.variant == "off_plan_injection"
                    and not injected
                    and solver_steps >= max(0, spec.injection_step)
                ):
                    suggested = _payload_action_name(payloads[0])
                    action_name = choose_off_plan_action(
                        available_action_names(observation),
                        suggested_action=suggested,
                        rng=rng,
                    )
                    selected_payloads = [
                        _payload_for_action_name(
                            action_name,
                            frame,
                            reason="eca_perturbation_off_plan_injection",
                        )
                    ]
                    phase = "off_plan_injection"
                    injected = True

            for payload in selected_payloads:
                if actions_taken >= max_actions or done:
                    break
                trace.append(
                    _compact_trace_item(
                        phase=phase,
                        step=actions_taken,
                        observation=observation,
                        payload=payload,
                        plan_source=plan_source,
                    )
                )
                observation, done = _step_env(env, payload)
                actions_taken += 1
                if phase == "off_plan_injection":
                    perturbation_score = _as_int(observation.get("score"), 0)
            solver_steps += 1
    except Exception as exc:  # pragma: no cover - exercised by integration smoke.
        errors.append(repr(exc))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup.
                errors.append(f"close:{exc!r}")

    final_score = _as_int(observation.get("score"), 0)
    final_state = str(observation.get("state", "") or "")
    recovery_delta = final_score - perturbation_score
    perturbation_applied = spec.variant != "canonical_full_plan"
    plan_sources = Counter(str(item.get("plan_source", "") or "") for item in trace if item.get("plan_source"))
    return _json_safe(
        {
            "schema_version": "eca_perception_perturbation_episode.v1",
            "game_id": game_id,
            "variant": spec.variant,
            "seed": spec.seed,
            "prefix_length": spec.prefix_length,
            "injection_step": spec.injection_step,
            "status": "ok" if not errors else "error",
            "errors": errors,
            "frame_only": not any("observation['frame']" in error for error in errors),
            "actions_taken": actions_taken,
            "final_score": final_score,
            "highest_level": final_score,
            "final_state": final_state,
            "perturbation_score": perturbation_score,
            "recovery_score_delta": recovery_delta,
            "perturbation_applied": perturbation_applied,
            "recovered_after_perturbation": bool(
                perturbation_applied and (recovery_delta > 0 or final_state == "WIN")
            ),
            "off_plan_injected": injected,
            "plan_source_counts": dict(sorted(plan_sources.items())),
            "trace": trace,
            "elapsed_sec": round(time.time() - start, 3),
        }
    )


def build_perturbation_specs(
    variants: Iterable[str],
    *,
    seeds: Iterable[int],
    prefix_lengths: Iterable[int],
    injection_step: int,
) -> list[PerturbationSpec]:
    specs: list[PerturbationSpec] = []
    for variant in variants:
        variant = str(variant or "").strip()
        if not variant:
            continue
        if variant == "random_prefix_recovery":
            for seed in seeds:
                for prefix_length in prefix_lengths:
                    specs.append(
                        PerturbationSpec(
                            variant=variant,
                            prefix_length=max(0, int(prefix_length)),
                            injection_step=injection_step,
                            seed=int(seed),
                        )
                    )
        else:
            for seed in seeds:
                specs.append(
                    PerturbationSpec(
                        variant=variant,
                        prefix_length=0,
                        injection_step=injection_step,
                        seed=int(seed),
                    )
                )
    return specs


def _canonical_floor_scores(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    floors: dict[str, int] = {}
    for row in rows:
        if str(row.get("variant", "") or "") != "canonical_full_plan":
            continue
        if row.get("status") != "ok":
            continue
        game_id = str(row.get("game_id", "") or "")
        if not game_id:
            continue
        floors[game_id] = max(floors.get(game_id, 0), _as_int(row.get("final_score"), 0))
    return floors


def _annotate_canonical_floor_recovery(rows: list[dict[str, Any]]) -> None:
    floors = _canonical_floor_scores(rows)
    for row in rows:
        game_id = str(row.get("game_id", "") or "")
        floor = floors.get(game_id)
        row["canonical_floor_score"] = floor
        row["canonical_floor_recovered"] = bool(
            floor is not None
            and bool(row.get("perturbation_applied"))
            and _as_int(row.get("final_score"), 0) >= floor
        )
        row["positive_canonical_floor_recovered"] = bool(
            floor is not None
            and floor > 0
            and bool(row.get("perturbation_applied"))
            and _as_int(row.get("final_score"), 0) >= floor
        )


def _generalisation_tier_name(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("generalisation_tier", "") or "").strip()
    if explicit:
        return explicit
    primitive_ids = row.get("primitive_ids")
    if isinstance(primitive_ids, list):
        return generalisation_tier_for_primitives(primitive_ids)
    return "unknown"


def _has_game_conditioned_primitive_field(row: Mapping[str, Any]) -> bool:
    return "uses_game_conditioned_primitive" in row or isinstance(row.get("primitive_ids"), list)


def _uses_game_conditioned_primitive(row: Mapping[str, Any]) -> bool:
    if "uses_game_conditioned_primitive" in row:
        return bool(row.get("uses_game_conditioned_primitive"))
    primitive_ids = row.get("primitive_ids")
    if isinstance(primitive_ids, list):
        return uses_game_conditioned_scope(primitive_ids)
    return False


def _is_adapter_free_row(row: Mapping[str, Any]) -> bool:
    if bool(row.get("uses_per_game_adapter")):
        return False
    if "adapter_free" in row and not bool(row.get("adapter_free")):
        return False
    if not _has_game_conditioned_primitive_field(row):
        return False
    return not _uses_game_conditioned_primitive(row)


def _summarise_group(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    item_list = list(items)
    run_count = len(item_list)
    perturbation_count = sum(1 for row in item_list if bool(row.get("perturbation_applied")))
    recovered_count = sum(1 for row in item_list if bool(row.get("recovered_after_perturbation")))
    floor_recovered_count = sum(1 for row in item_list if bool(row.get("canonical_floor_recovered")))
    positive_floor_perturbation_count = sum(
        1
        for row in item_list
        if bool(row.get("perturbation_applied"))
        and _as_int(row.get("canonical_floor_score"), 0) > 0
    )
    positive_floor_recovered_count = sum(
        1 for row in item_list if bool(row.get("positive_canonical_floor_recovered"))
    )
    score_sum = sum(_as_int(row.get("final_score"), 0) for row in item_list)
    game_conditioned_count = sum(1 for row in item_list if _uses_game_conditioned_primitive(row))
    adapter_free_count = sum(1 for row in item_list if _is_adapter_free_row(row))
    return {
        "run_count": run_count,
        "perturbation_count": perturbation_count,
        "ok_count": sum(1 for row in item_list if row.get("status") == "ok"),
        "error_count": sum(1 for row in item_list if row.get("status") != "ok"),
        "score_sum": score_sum,
        "mean_score": _safe_div(
            sum(_as_float(row.get("final_score"), 0.0) for row in item_list),
            run_count,
        ),
        "win_count": sum(1 for row in item_list if str(row.get("final_state", "") or "") == "WIN"),
        "recovered_count": recovered_count,
        "recovery_rate": _safe_div(recovered_count, perturbation_count),
        "canonical_floor_recovered_count": floor_recovered_count,
        "canonical_floor_recovery_rate": _safe_div(floor_recovered_count, perturbation_count),
        "positive_canonical_floor_perturbation_count": positive_floor_perturbation_count,
        "positive_canonical_floor_recovered_count": positive_floor_recovered_count,
        "positive_canonical_floor_recovery_rate": _safe_div(
            positive_floor_recovered_count,
            positive_floor_perturbation_count,
        ),
        "frame_only_failure_count": sum(1 for row in item_list if not bool(row.get("frame_only"))),
        "strict_generic_ablation_count": sum(
            1 for row in item_list if bool(row.get("strict_generic_ablation"))
        ),
        "uses_game_conditioned_primitive_count": game_conditioned_count,
        "adapter_free_count": adapter_free_count,
        "adapter_free_score_sum": sum(
            _as_int(row.get("final_score"), 0)
            for row in item_list
            if _is_adapter_free_row(row)
        ),
        "uses_game_conditioned_primitive_score_sum": sum(
            _as_int(row.get("final_score"), 0)
            for row in item_list
            if _uses_game_conditioned_primitive(row)
        ),
    }


def _primitive_usage_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        primitive_ids = row.get("primitive_ids")
        if not isinstance(primitive_ids, list):
            continue
        for primitive_id in primitive_ids:
            normalized = str(primitive_id or "").strip()
            if normalized:
                counts[normalized] += 1
    return dict(sorted(counts.items()))


def _trace_counter(row: Mapping[str, Any], key: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    trace = row.get("trace")
    if not isinstance(trace, list):
        return counts
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        value = str(item.get(key, "") or "").strip()
        if value:
            counts[value] += 1
    return counts


def _top_counter_item(counts: Counter[str]) -> tuple[str, int]:
    if not counts:
        return ("", 0)
    key, value = counts.most_common(1)[0]
    return (str(key), int(value))


def _row_trace_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    trace = row.get("trace")
    trace_len = len(trace) if isinstance(trace, list) else 0
    primitive_counts = _trace_counter(row, "primitive")
    action_counts = _trace_counter(row, "action")
    solver_family_counts = _trace_counter(row, "solver_family")
    dominant_primitive, dominant_primitive_count = _top_counter_item(primitive_counts)
    dominant_action, dominant_action_count = _top_counter_item(action_counts)
    dominant_family, dominant_family_count = _top_counter_item(solver_family_counts)
    primitive_repetition = _safe_div(dominant_primitive_count, trace_len)
    action_repetition = _safe_div(dominant_action_count, trace_len)
    family_repetition = _safe_div(dominant_family_count, trace_len)
    return {
        "trace_len": trace_len,
        "dominant_primitive": dominant_primitive,
        "dominant_primitive_repetition_rate": primitive_repetition,
        "dominant_action": dominant_action,
        "dominant_action_repetition_rate": action_repetition,
        "dominant_solver_family": dominant_family,
        "dominant_solver_family_repetition_rate": family_repetition,
        "primitive_counts": dict(sorted(primitive_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "solver_family_counts": dict(sorted(solver_family_counts.items())),
    }


def _classify_failure_signature(row: Mapping[str, Any], signature: Mapping[str, Any]) -> str:
    if _as_int(row.get("final_score"), 0) > 0:
        return "scored"
    dominant_primitive = str(signature.get("dominant_primitive", "") or "")
    dominant_action = str(signature.get("dominant_action", "") or "")
    primitive_repetition = _as_float(signature.get("dominant_primitive_repetition_rate"), 0.0)
    action_repetition = _as_float(signature.get("dominant_action_repetition_rate"), 0.0)
    if dominant_primitive == "action_loop_probe":
        return "insufficient_transition_family_signal"
    if dominant_primitive in {"grid_bfs", "mirror_joint_cursor_bfs", "stride_marker_maze_bfs"} and action_repetition >= 0.85:
        return "navigation_target_lockin"
    if dominant_primitive in {"finite_state_action_search", "generic_attach_pair_transport_bfs"} and primitive_repetition >= 0.70:
        return "solver_family_without_target_translation"
    if dominant_action == "ACTION6" and action_repetition >= 0.85:
        return "click_affordance_target_lockin"
    if action_repetition >= 0.85:
        return "single_action_lockin"
    return "no_score_after_generic_search"


def _failure_signatures(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("status", "") or "") != "ok":
            continue
        if _as_int(row.get("final_score"), 0) > 0:
            continue
        signature = _row_trace_signature(row)
        failures.append(
            {
                "game_id": str(row.get("game_id", "") or "unknown"),
                "variant": str(row.get("variant", "") or "unknown"),
                "failure_signature": _classify_failure_signature(row, signature),
                **signature,
            }
        )
    return failures


def _operator_counts_from_trace(row: Mapping[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    trace = row.get("trace")
    if not isinstance(trace, list):
        return counts
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        control_target = item.get("control_target")
        if not isinstance(control_target, Mapping):
            continue
        operator = str(control_target.get("primary_operator", "") or "").strip()
        if operator:
            counts[operator] += 1
    return counts


def _last_target_verification(row: Mapping[str, Any]) -> dict[str, Any]:
    trace = row.get("trace")
    if not isinstance(trace, list):
        return {}
    for item in reversed(trace):
        if not isinstance(item, Mapping):
            continue
        payload = item.get("target_verification")
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _target_verification_direct_score_probe_counts(row: Mapping[str, Any]) -> tuple[int, int]:
    trace = row.get("trace")
    if not isinstance(trace, list):
        return 0, 0
    max_direct = 0
    max_failure = 0
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        payload = item.get("target_verification")
        if not isinstance(payload, Mapping):
            continue
        direct = _as_int(payload.get("direct_score_probe_without_goal_count"), 0)
        failure = _as_int(
            payload.get("direct_score_probe_without_goal_failure_count"),
            0,
        )
        recent = payload.get("recent")
        if direct <= 0 and isinstance(recent, list):
            direct = sum(
                1
                for entry in recent
                if isinstance(entry, Mapping)
                and bool(entry.get("direct_score_probe_without_goal"))
            )
            failure = sum(
                1
                for entry in recent
                if isinstance(entry, Mapping)
                and bool(entry.get("direct_score_probe_without_goal"))
                and not bool(entry.get("target_success"))
            )
        max_direct = max(max_direct, direct)
        max_failure = max(max_failure, failure)
    return max_direct, max_failure


def _target_key_tuple(value: Any) -> tuple[str, int, int, str] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return (str(value[0]), int(value[1]), int(value[2]), str(value[3]))
    except (TypeError, ValueError):
        return None


def _direct_score_probe_reentry_counts(row: Mapping[str, Any]) -> tuple[int, int]:
    trace = row.get("trace")
    if not isinstance(trace, list):
        return 0, 0
    contradicted_keys: set[tuple[str, int, int, str]] = set()
    observed_keys: set[tuple[str, int, int, str]] = set()
    reentry_count = 0
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        decision = _rod_score_precondition_action_from_trace_item(item)
        decision_key = _target_key_tuple(decision.get("target_key")) if decision else None
        if (
            decision_key is not None
            and decision_key in contradicted_keys
            and (bool(decision.get("applied")) or str(decision.get("target_action", "") or ""))
        ):
            reentry_count += 1
        payload = item.get("target_verification")
        if not isinstance(payload, Mapping):
            continue
        recent = payload.get("recent")
        if not isinstance(recent, list):
            continue
        for entry in recent:
            if not isinstance(entry, Mapping):
                continue
            if not bool(entry.get("direct_score_probe_without_goal")):
                continue
            if bool(entry.get("target_success")):
                continue
            key = _target_key_tuple(entry.get("target_key"))
            if key is None:
                continue
            contradicted_keys.add(key)
            observed_keys.add(key)
    return len(observed_keys), reentry_count


def _operator_trace_alignment(row: Mapping[str, Any]) -> dict[str, int]:
    trace = row.get("trace")
    if not isinstance(trace, list):
        return {
            "control_target_count": 0,
            "target_candidate_trace_count": 0,
            "probe_site_candidate_trace_count": 0,
            "probe_site_with_site_candidate_trace_count": 0,
            "click_probe_site_candidate_trace_count": 0,
            "actionable_click_probe_site_candidate_trace_count": 0,
            "executable_operator_action_count": 0,
            "probe_site_action_match_count": 0,
            "probe_site_action_match_with_site_count": 0,
        }
    control_target_count = 0
    target_candidate_trace_count = 0
    probe_site_candidate_trace_count = 0
    probe_site_with_site_candidate_trace_count = 0
    click_probe_site_candidate_trace_count = 0
    actionable_click_probe_site_candidate_trace_count = 0
    executable_operator_action_count = 0
    probe_site_action_match_count = 0
    probe_site_action_match_with_site_count = 0
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        control_target = item.get("control_target")
        if not isinstance(control_target, Mapping):
            continue
        operator = str(control_target.get("primary_operator", "") or "").strip()
        if not operator:
            continue
        control_target_count += 1
        candidates = control_target.get("target_candidates")
        if isinstance(candidates, list):
            target_candidate_trace_count += len(candidates)
        probe_sites = control_target.get("probe_site_candidates")
        if isinstance(probe_sites, list):
            probe_site_candidate_trace_count += len(probe_sites)
            probe_site_with_site_candidate_trace_count += sum(
                1 for probe_site in probe_sites if _probe_site_has_concrete_site(probe_site)
            )
            click_probe_site_candidate_trace_count += sum(
                1
                for probe_site in probe_sites
                if isinstance(probe_site, Mapping)
                and str(probe_site.get("operator", "") or "") == "click"
                and _normalize_action_name(probe_site.get("action_name")) == "ACTION6"
                and _probe_site_has_concrete_site(probe_site)
            )
            actionable_click_probe_site_candidate_trace_count += sum(
                1
                for probe_site in probe_sites
                if isinstance(probe_site, Mapping)
                and str(probe_site.get("operator", "") or "") == "click"
                and _normalize_action_name(probe_site.get("action_name")) == "ACTION6"
                and _probe_site_has_concrete_site(probe_site)
                and bool(probe_site.get("actionable"))
            )
        action = str(item.get("action", "") or "").strip()
        if operator == "click" and action == "ACTION6":
            executable_operator_action_count += 1
        elif operator in {"route_to", "push"} and action in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
            executable_operator_action_count += 1
        elif operator in {"attach", "drop"} and action in {"ACTION5", "ACTION6"}:
            executable_operator_action_count += 1
        elif operator == "repeat_action" and action:
            executable_operator_action_count += 1
        if isinstance(probe_sites, list):
            for probe_site in probe_sites:
                if not isinstance(probe_site, Mapping):
                    continue
                if action and action == str(probe_site.get("action_name", "") or "").strip():
                    probe_site_action_match_count += 1
                    if _probe_site_has_concrete_site(probe_site):
                        probe_site_action_match_with_site_count += 1
                    break
    return {
        "control_target_count": control_target_count,
        "target_candidate_trace_count": target_candidate_trace_count,
        "probe_site_candidate_trace_count": probe_site_candidate_trace_count,
        "probe_site_with_site_candidate_trace_count": probe_site_with_site_candidate_trace_count,
        "click_probe_site_candidate_trace_count": click_probe_site_candidate_trace_count,
        "actionable_click_probe_site_candidate_trace_count": actionable_click_probe_site_candidate_trace_count,
        "executable_operator_action_count": executable_operator_action_count,
        "probe_site_action_match_count": probe_site_action_match_count,
        "probe_site_action_match_with_site_count": probe_site_action_match_with_site_count,
    }


def _probe_site_has_concrete_site(probe_site: Any) -> bool:
    if not isinstance(probe_site, Mapping):
        return False
    site = probe_site.get("site")
    if isinstance(site, (list, tuple)) and len(site) >= 2:
        try:
            int(site[0])
            int(site[1])
            return True
        except (TypeError, ValueError):
            return False
    payload = probe_site.get("payload")
    if isinstance(payload, Mapping) and "x" in payload and "y" in payload:
        try:
            int(payload.get("x"))
            int(payload.get("y"))
            return True
        except (TypeError, ValueError):
            return False
    return False


def _operator_failure_signature(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if str(row.get("status", "") or "") != "ok":
        return None
    if _as_int(row.get("final_score"), 0) > 0:
        return None
    operator_counts = _operator_counts_from_trace(row)
    dominant_operator, dominant_operator_count = _top_counter_item(operator_counts)
    trace = row.get("trace")
    trace_len = len(trace) if isinstance(trace, list) else 0
    verification = _last_target_verification(row)
    attempt_count = _as_int(verification.get("attempt_count"), 0)
    success_count = _as_int(verification.get("success_count"), 0)
    failure_count = _as_int(verification.get("failure_count"), 0)
    alignment = _operator_trace_alignment(row)
    recent = verification.get("recent")
    recent_failures = 0
    recent_successes = 0
    if isinstance(recent, list):
        for item in recent:
            if not isinstance(item, Mapping):
                continue
            if bool(item.get("progressed")):
                recent_successes += 1
            else:
                recent_failures += 1

    if not dominant_operator:
        signature = "no_operator_signal"
    elif attempt_count <= 0:
        if int(alignment.get("click_probe_site_candidate_trace_count", 0)) > 0:
            if int(alignment.get("actionable_click_probe_site_candidate_trace_count", 0)) > 0:
                signature = "click_probe_site_payload_not_connected"
            else:
                signature = "click_probe_sites_not_actionable"
        elif int(alignment.get("target_candidate_trace_count", 0)) <= 0:
            if int(alignment.get("probe_site_candidate_trace_count", 0)) > 0:
                if int(alignment.get("probe_site_with_site_candidate_trace_count", 0)) > 0:
                    signature = "operator_has_probe_sites_but_no_targets"
                else:
                    signature = "operator_has_action_probes_but_no_sites"
            else:
                signature = "operator_without_target_candidates"
        elif dominant_operator == "probe" and int(alignment.get("probe_site_candidate_trace_count", 0)) > 0:
            if int(alignment.get("probe_site_action_match_with_site_count", 0)) > 0:
                signature = "probe_site_candidates_executed_but_unverified"
            elif int(alignment.get("probe_site_action_match_count", 0)) > 0:
                signature = "probe_action_candidates_executed_without_site"
            else:
                signature = "probe_site_candidates_not_executed"
        elif dominant_operator == "probe":
            signature = "probe_operator_not_actionable"
        elif int(alignment.get("executable_operator_action_count", 0)) <= 0:
            if int(alignment.get("probe_site_candidate_trace_count", 0)) > 0:
                signature = "operator_probe_site_action_mismatch"
            else:
                signature = "operator_action_mismatch"
        elif dominant_operator == "click":
            if int(alignment.get("probe_site_candidate_trace_count", 0)) > 0:
                signature = "click_probe_site_payload_not_connected"
            else:
                signature = "click_payload_target_mismatch"
        else:
            if int(alignment.get("probe_site_candidate_trace_count", 0)) > 0:
                signature = "operator_probe_site_alignment_mismatch"
            else:
                signature = "operator_target_alignment_mismatch"
    elif dominant_operator == "click" and failure_count > success_count:
        signature = "click_target_repeat_failure"
    elif dominant_operator in {"attach", "drop", "push"} and failure_count > success_count:
        signature = f"{dominant_operator}_operator_no_progress"
    elif success_count > 0 and _as_int(row.get("final_score"), 0) <= 0:
        if dominant_operator == "route_to" and success_count > failure_count:
            signature = "route_progress_only_no_score"
        elif dominant_operator == "repeat_action" and success_count > failure_count:
            signature = "repeat_progress_only_no_score"
        else:
            signature = "target_progress_no_score"
    elif failure_count > success_count:
        signature = "operator_noop_or_no_progress"
    else:
        signature = "operator_unresolved_no_score"

    return {
        "game_id": str(row.get("game_id", "") or "unknown"),
        "variant": str(row.get("variant", "") or "unknown"),
        "operator_failure_signature": signature,
        "dominant_operator": dominant_operator,
        "dominant_operator_repetition_rate": _safe_div(dominant_operator_count, trace_len),
        "operator_counts": dict(sorted(operator_counts.items())),
        "target_attempt_count": attempt_count,
        "target_success_count": success_count,
        "target_failure_count": failure_count,
        "recent_target_success_count": recent_successes,
        "recent_target_failure_count": recent_failures,
        **alignment,
    }


def _operator_failure_signatures(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = _operator_failure_signature(row)
        if item is not None:
            out.append(item)
    return out


_PROGRESS_TO_SCORE_GAP_SIGNATURES = {
    "target_progress_no_score",
    "route_progress_only_no_score",
    "repeat_progress_only_no_score",
}


def _progress_to_score_closure_state(
    *,
    signature: str,
    target_success_count: int,
    target_failure_count: int,
    recent_target_success_count: int,
    recent_target_failure_count: int,
) -> str:
    if target_success_count <= 0:
        return "insufficient_evidence"
    if signature == "route_progress_only_no_score":
        return "route_progress_not_score_bearing"
    if signature == "repeat_progress_only_no_score":
        return "repeat_progress_not_score_bearing"
    if recent_target_failure_count > 0 and recent_target_success_count <= 0:
        return "stale_progress_no_recent_score_closure"
    if target_failure_count > target_success_count:
        return "mixed_progress_unreliable"
    return "progress_not_score_bearing"


def _progress_to_score_causal_closure_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    signature_counts: Counter[str] = Counter()
    closure_state_counts: Counter[str] = Counter()
    dominant_operator_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []

    for row in rows:
        operator_signature = _operator_failure_signature(row)
        if not operator_signature:
            continue
        signature = str(operator_signature.get("operator_failure_signature", "") or "")
        if signature not in _PROGRESS_TO_SCORE_GAP_SIGNATURES:
            continue

        game_id = str(operator_signature.get("game_id", "") or "unknown")
        target_success_count = _as_int(operator_signature.get("target_success_count"), 0)
        target_failure_count = _as_int(operator_signature.get("target_failure_count"), 0)
        recent_target_success_count = _as_int(
            operator_signature.get("recent_target_success_count"), 0
        )
        recent_target_failure_count = _as_int(
            operator_signature.get("recent_target_failure_count"), 0
        )
        closure_state = _progress_to_score_closure_state(
            signature=signature,
            target_success_count=target_success_count,
            target_failure_count=target_failure_count,
            recent_target_success_count=recent_target_success_count,
            recent_target_failure_count=recent_target_failure_count,
        )
        dominant_operator = str(operator_signature.get("dominant_operator", "") or "unknown")

        signature_counts[signature] += 1
        closure_state_counts[closure_state] += 1
        dominant_operator_counts[dominant_operator] += 1
        by_game[game_id]["row_count"] += 1
        by_game[game_id]["target_attempt_count"] += _as_int(
            operator_signature.get("target_attempt_count"), 0
        )
        by_game[game_id]["target_success_count"] += target_success_count
        by_game[game_id]["target_failure_count"] += target_failure_count
        by_game[game_id]["recent_target_success_count"] += recent_target_success_count
        by_game[game_id]["recent_target_failure_count"] += recent_target_failure_count
        by_game[game_id][f"signature:{signature}"] += 1
        by_game[game_id][f"closure_state:{closure_state}"] += 1

        if len(samples) < sample_limit:
            samples.append(
                {
                    "game_id": game_id,
                    "variant": str(operator_signature.get("variant", "") or "unknown"),
                    "operator_failure_signature": signature,
                    "closure_state": closure_state,
                    "dominant_operator": dominant_operator,
                    "target_attempt_count": _as_int(
                        operator_signature.get("target_attempt_count"), 0
                    ),
                    "target_success_count": target_success_count,
                    "target_failure_count": target_failure_count,
                    "recent_target_success_count": recent_target_success_count,
                    "recent_target_failure_count": recent_target_failure_count,
                    "probe_site_candidate_trace_count": _as_int(
                        operator_signature.get("probe_site_candidate_trace_count"), 0
                    ),
                    "executable_operator_action_count": _as_int(
                        operator_signature.get("executable_operator_action_count"), 0
                    ),
                }
            )

    row_count = sum(signature_counts.values())
    return {
        "schema_version": "eca_progress_to_score_causal_closure_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "row_count": int(row_count),
        "progress_only_no_score_count": int(row_count),
        "signature_counts": dict(sorted(signature_counts.items())),
        "closure_state_counts": dict(sorted(closure_state_counts.items())),
        "dominant_operator_counts": dict(sorted(dominant_operator_counts.items())),
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "validated score-bearing closure posterior",
            "allowed_connection": "guarded_ranking_only",
            "action_generation_allowed": False,
        },
    }


def _local_progress_closure_verifier_outcome_posterior(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    state_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    effect_type_counts: Counter[str] = Counter()
    by_target_key: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    preflight_reason_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    initial_execution_count = 0
    continuation_execution_count = 0
    score_delta_sum = 0

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            preflight = unsolved.get("local_progress_closure_verifier_preflight")
            if isinstance(preflight, Mapping):
                reason = str(preflight.get("reason", "") or "unknown")
                preflight_reason_counts[reason] += 1
                by_game[game_id][f"preflight:{reason}"] += 1
            if not bool(unsolved.get("local_progress_closure_verifier")):
                continue
            if not bool(unsolved.get("executed") or unsolved.get("used")):
                continue

            request = unsolved.get("local_progress_request")
            request = request if isinstance(request, Mapping) else {}
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            observed = unsolved.get("observed_effect")
            observed = observed if isinstance(observed, Mapping) else {}
            target_key = str(request.get("target_key", "") or "unknown")
            phase = str(item.get("phase", "") or "")
            score_delta = _as_int(
                outcome.get("score_delta", observed.get("score_delta", 0)),
                0,
            )
            score_delta_sum += score_delta
            goal_bearing = bool(outcome.get("goal_bearing") or score_delta > 0)
            objective_progress = bool(
                outcome.get("objective_progress") or observed.get("objective_progress")
            )
            goal_aligned_progress = bool(
                outcome.get("goal_aligned_progress")
                or observed.get("goal_aligned_progress")
            )
            structural_progress = bool(
                outcome.get("structural_progress")
                or observed.get("structural_progress")
            )
            no_op = bool(outcome.get("no_op") or observed.get("no_op"))
            verdict = str(unsolved.get("verdict", "") or outcome.get("verdict", "") or "unknown")
            effect_type = str(
                observed.get("effect_type", "") or outcome.get("effect_type", "") or "unknown"
            )

            if goal_bearing:
                state = "score_bearing_confirmed"
            elif objective_progress or goal_aligned_progress:
                state = "progress_no_score_unconfirmed"
            elif no_op:
                state = "noop_falsified"
            elif structural_progress:
                state = "visible_effect_no_score_falsified"
            else:
                state = "no_progress_falsified"

            state_counts[state] += 1
            verdict_counts[verdict] += 1
            effect_type_counts[effect_type] += 1
            by_target_key[target_key]["row_count"] += 1
            by_target_key[target_key][f"state:{state}"] += 1
            by_target_key[target_key]["score_delta_sum"] += score_delta
            by_game[game_id]["row_count"] += 1
            by_game[game_id][f"state:{state}"] += 1
            by_game[game_id]["score_delta_sum"] += score_delta
            if phase == "local_progress_closure_verifier":
                initial_execution_count += 1
            else:
                continuation_execution_count += 1

            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), 0),
                        "phase": phase,
                        "action": str(item.get("action", "") or ""),
                        "target_key": target_key,
                        "verifier_type": str(request.get("verifier_type", "") or ""),
                        "dominant_goal_link_type": str(
                            request.get("dominant_goal_link_type", "") or ""
                        ),
                        "gap_count": _as_int(request.get("gap_count"), 0),
                        "state": state,
                        "verdict": verdict,
                        "effect_type": effect_type,
                        "score_delta": score_delta,
                        "structural_progress": structural_progress,
                        "objective_progress": objective_progress,
                        "goal_aligned_progress": goal_aligned_progress,
                        "no_op": no_op,
                    }
                )

    row_count = sum(state_counts.values())
    score_bearing_count = _as_int(state_counts.get("score_bearing_confirmed"), 0)
    if row_count <= 0:
        promotion_state = "no_execution_evidence"
    elif score_bearing_count > 0:
        promotion_state = "score_bearing_candidate"
    else:
        promotion_state = "ranking_blocked_no_score_closure"

    return {
        "schema_version": "eca_local_progress_closure_verifier_outcome_posterior.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "row_count": int(row_count),
        "initial_execution_count": int(initial_execution_count),
        "continuation_execution_count": int(continuation_execution_count),
        "score_delta_sum": int(score_delta_sum),
        "score_bearing_count": int(score_bearing_count),
        "promotion_state": promotion_state,
        "state_counts": dict(sorted(state_counts.items())),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "effect_type_counts": dict(sorted(effect_type_counts.items())),
        "preflight_reason_counts": dict(sorted(preflight_reason_counts.items())),
        "by_target_key": {
            target_key: dict(sorted(stats.items()))
            for target_key, stats in sorted(by_target_key.items())
        },
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "score_bearing_count > 0 for the target key",
            "allowed_connection": "negative_ranking_guard_only_until_score_closure",
            "action_generation_allowed": False,
        },
    }


def _score_relation_candidate_verifier_outcome_posterior(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    state_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    effect_type_counts: Counter[str] = Counter()
    candidate_relation_counts: Counter[str] = Counter()
    probe_kind_counts: Counter[str] = Counter()
    by_target_key: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    preflight_reason_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    execution_count = 0
    score_delta_sum = 0

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            preflight = unsolved.get("score_relation_candidate_verifier_preflight")
            if isinstance(preflight, Mapping):
                reason = str(preflight.get("reason", "") or "unknown")
                preflight_reason_counts[reason] += 1
                by_game[game_id][f"preflight:{reason}"] += 1
            if not bool(unsolved.get("score_relation_candidate_verifier")):
                continue
            if not bool(unsolved.get("executed") or unsolved.get("used")):
                continue

            candidate = unsolved.get("score_relation_candidate")
            candidate = candidate if isinstance(candidate, Mapping) else {}
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            observed = unsolved.get("observed_effect")
            observed = observed if isinstance(observed, Mapping) else {}
            target_key = str(candidate.get("target_key", "") or "unknown")
            candidate_relation = str(
                candidate.get("candidate_relation", "") or "unknown"
            )
            probe_kind = str(candidate.get("probe_kind", "") or "unknown")
            score_delta = _as_int(
                outcome.get("score_delta", observed.get("score_delta", 0)),
                0,
            )
            score_delta_sum += score_delta
            goal_bearing = bool(outcome.get("goal_bearing") or score_delta > 0)
            objective_progress = bool(
                outcome.get("objective_progress") or observed.get("objective_progress")
            )
            goal_aligned_progress = bool(
                outcome.get("goal_aligned_progress")
                or observed.get("goal_aligned_progress")
            )
            structural_progress = bool(
                outcome.get("structural_progress")
                or observed.get("structural_progress")
            )
            no_op = bool(outcome.get("no_op") or observed.get("no_op"))
            verdict = str(unsolved.get("verdict", "") or outcome.get("verdict", "") or "unknown")
            effect_type = str(
                observed.get("effect_type", "") or outcome.get("effect_type", "") or "unknown"
            )

            if goal_bearing:
                state = "score_relation_confirmed"
            elif objective_progress or goal_aligned_progress:
                state = "progress_no_score_unconfirmed"
            elif no_op:
                state = "noop_falsified"
            elif structural_progress or effect_type == "visible_effect":
                state = "visible_effect_no_score_falsified"
            else:
                state = "no_progress_falsified"

            execution_count += 1
            state_counts[state] += 1
            verdict_counts[verdict] += 1
            effect_type_counts[effect_type] += 1
            candidate_relation_counts[candidate_relation] += 1
            probe_kind_counts[probe_kind] += 1
            by_target_key[target_key]["row_count"] += 1
            by_target_key[target_key][f"state:{state}"] += 1
            by_target_key[target_key]["score_delta_sum"] += score_delta
            by_game[game_id]["row_count"] += 1
            by_game[game_id][f"state:{state}"] += 1
            by_game[game_id]["score_delta_sum"] += score_delta
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), 0),
                        "phase": str(item.get("phase", "") or ""),
                        "action": str(item.get("action", "") or ""),
                        "target_key": target_key,
                        "candidate_relation": candidate_relation,
                        "probe_kind": probe_kind,
                        "gap_count": _as_int(candidate.get("gap_count"), 0),
                        "state": state,
                        "verdict": verdict,
                        "effect_type": effect_type,
                        "score_delta": score_delta,
                        "structural_progress": structural_progress,
                        "objective_progress": objective_progress,
                        "goal_aligned_progress": goal_aligned_progress,
                        "no_op": no_op,
                    }
                )

    confirmed_count = _as_int(state_counts.get("score_relation_confirmed"), 0)
    if execution_count <= 0:
        promotion_state = "no_execution_evidence"
    elif confirmed_count > 0:
        promotion_state = "score_relation_confirmed"
    else:
        promotion_state = "ranking_blocked_no_score_relation"

    return {
        "schema_version": "eca_score_relation_candidate_verifier_outcome_posterior.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "row_count": int(execution_count),
        "score_delta_sum": int(score_delta_sum),
        "score_relation_confirmed_count": int(confirmed_count),
        "promotion_state": promotion_state,
        "state_counts": dict(sorted(state_counts.items())),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "effect_type_counts": dict(sorted(effect_type_counts.items())),
        "candidate_relation_counts": dict(sorted(candidate_relation_counts.items())),
        "probe_kind_counts": dict(sorted(probe_kind_counts.items())),
        "preflight_reason_counts": dict(sorted(preflight_reason_counts.items())),
        "by_target_key": {
            target_key: dict(sorted(stats.items()))
            for target_key, stats in sorted(by_target_key.items())
        },
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "score_relation_confirmed_count > 0 for the candidate relation",
            "allowed_connection": "negative_ranking_guard_only_until_score_relation_confirmed",
            "action_generation_allowed": False,
        },
    }


def _score_relation_falsified_target_action_signature_memory_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    execution_count = 0
    falsified_execution_count = 0
    visible_effect_no_score_falsified_count = 0
    score_relation_confirmed_count = 0
    state_counts: Counter[str] = Counter()
    signature_counts: Counter[str] = Counter()
    signature_stats: dict[str, dict[str, Any]] = {}
    by_game: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            if not bool(unsolved.get("score_relation_candidate_verifier")):
                continue
            if not bool(unsolved.get("executed") or unsolved.get("used")):
                continue
            candidate = unsolved.get("score_relation_candidate")
            candidate = candidate if isinstance(candidate, Mapping) else {}
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            observed = unsolved.get("observed_effect")
            observed = observed if isinstance(observed, Mapping) else {}
            target_key = str(candidate.get("target_key", "") or "unknown")
            candidate_relation = str(
                candidate.get("candidate_relation", "") or "unknown"
            )
            probe_kind = str(candidate.get("probe_kind", "") or "unknown")
            action = _normalize_action_name(
                candidate.get(
                    "dominant_action",
                    unsolved.get("selected_action", item.get("action", "")),
                )
            )
            score_delta = _as_int(
                outcome.get("score_delta", observed.get("score_delta", 0)),
                0,
            )
            goal_bearing = bool(outcome.get("goal_bearing") or score_delta > 0)
            objective_progress = bool(
                outcome.get("objective_progress") or observed.get("objective_progress")
            )
            goal_aligned_progress = bool(
                outcome.get("goal_aligned_progress")
                or observed.get("goal_aligned_progress")
            )
            structural_progress = bool(
                outcome.get("structural_progress")
                or observed.get("structural_progress")
            )
            no_op = bool(outcome.get("no_op") or observed.get("no_op"))
            effect_type = str(
                observed.get("effect_type", "")
                or outcome.get("effect_type", "")
                or "unknown"
            )
            if goal_bearing:
                state = "score_relation_confirmed"
            elif objective_progress or goal_aligned_progress:
                state = "progress_no_score_unconfirmed"
            elif no_op:
                state = "noop_falsified"
            elif structural_progress or effect_type == "visible_effect":
                state = "visible_effect_no_score_falsified"
            else:
                state = "no_progress_falsified"

            execution_count += 1
            state_counts[state] += 1
            by_game[game_id]["execution_count"] += 1
            by_game[game_id][f"state:{state}"] += 1
            if state == "score_relation_confirmed":
                score_relation_confirmed_count += 1
                continue
            if state not in {
                "visible_effect_no_score_falsified",
                "noop_falsified",
                "no_progress_falsified",
            }:
                continue

            falsified_execution_count += 1
            if state == "visible_effect_no_score_falsified":
                visible_effect_no_score_falsified_count += 1
            signature = _score_relation_target_action_signature(
                target_key=target_key,
                action=action,
                candidate_relation=candidate_relation,
                probe_kind=probe_kind,
            )
            signature_counts[signature] += 1
            by_game[game_id]["falsified_execution_count"] += 1
            by_game[game_id][f"falsified_signature:{signature}"] += 1
            site = _site_effect_target_site_from_key(target_key)
            stats = signature_stats.setdefault(
                signature,
                {
                    "signature": signature,
                    "target_key": target_key,
                    "action": action,
                    "candidate_relation": candidate_relation,
                    "probe_kind": probe_kind,
                    "target_site": list(site) if site is not None else [],
                    "falsified_count": 0,
                    "visible_effect_no_score_falsified_count": 0,
                    "noop_falsified_count": 0,
                    "no_progress_falsified_count": 0,
                    "score_delta_sum": 0,
                    "game_ids": set(),
                    "first_seen_step": _as_int(item.get("step"), 0),
                    "last_seen_step": _as_int(item.get("step"), 0),
                },
            )
            stats["falsified_count"] = _as_int(stats.get("falsified_count"), 0) + 1
            stats[f"{state}_count"] = _as_int(stats.get(f"{state}_count"), 0) + 1
            stats["score_delta_sum"] = _as_int(stats.get("score_delta_sum"), 0) + score_delta
            stats["first_seen_step"] = min(
                _as_int(stats.get("first_seen_step"), 0),
                _as_int(item.get("step"), 0),
            )
            stats["last_seen_step"] = max(
                _as_int(stats.get("last_seen_step"), 0),
                _as_int(item.get("step"), 0),
            )
            if isinstance(stats.get("game_ids"), set):
                stats["game_ids"].add(game_id)

    signature_rows: list[dict[str, Any]] = []
    for stats in signature_stats.values():
        row = {
            key: value
            for key, value in stats.items()
            if key != "game_ids"
        }
        row["game_ids"] = (
            sorted(stats.get("game_ids", set()))
            if isinstance(stats.get("game_ids"), set)
            else []
        )
        signature_rows.append(row)
    signature_rows.sort(
        key=lambda row: (
            -_as_int(row.get("falsified_count"), 0),
            str(row.get("target_key", "")),
            str(row.get("action", "")),
        )
    )

    return {
        "schema_version": (
            "eca_score_relation_falsified_target_action_signature_memory_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "execution_count": int(execution_count),
        "score_relation_confirmed_count": int(score_relation_confirmed_count),
        "falsified_execution_count": int(falsified_execution_count),
        "visible_effect_no_score_falsified_count": int(
            visible_effect_no_score_falsified_count
        ),
        "falsified_signature_count": int(len(signature_rows)),
        "state_counts": dict(sorted(state_counts.items())),
        "signature_counts": dict(sorted(signature_counts.items())),
        "signature_rows": signature_rows[:sample_limit],
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "promotion_state": (
            "falsified_signatures_present_block_reexecution"
            if falsified_execution_count > 0
            else (
                "confirmed_without_falsified_signatures"
                if score_relation_confirmed_count > 0
                else "no_falsified_signature_memory"
            )
        ),
        "next_planner_contract": {
            "required_before_execution": "candidate target/action/relation/probe signature must not match a falsified signature",
            "allowed_connection": "negative_admission_guard_only",
            "action_generation_allowed": False,
        },
    }


def _score_relation_family_scope_negative_memory_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    row_list = list(rows)
    exact_memory = _score_relation_falsified_target_action_signature_memory_sensor(
        row_list,
        sample_limit=4096,
    )
    direction = _score_relation_progress_goal_binding_action_effect_direction_sensor(
        row_list,
        sample_limit=4096,
    )

    exact_falsified_signatures = {
        str(row.get("signature", "") or "")
        for row in list(exact_memory.get("signature_rows") or [])
        if isinstance(row, Mapping) and str(row.get("signature", "") or "")
    }

    family_stats: dict[str, dict[str, Any]] = {}
    for row in list(exact_memory.get("signature_rows") or []):
        if not isinstance(row, Mapping):
            continue
        family_signature = _score_relation_target_action_family_signature(
            target_key=str(row.get("target_key", "") or ""),
            action=str(row.get("action", "") or ""),
            candidate_relation=str(row.get("candidate_relation", "") or ""),
            probe_kind=str(row.get("probe_kind", "") or ""),
        )
        stats = family_stats.setdefault(
            family_signature,
            {
                "family_signature": family_signature,
                "action_family": _normalize_action_name(row.get("action", "")),
                "candidate_relation_family": str(
                    row.get("candidate_relation", "") or ""
                ),
                "probe_kind_family": str(row.get("probe_kind", "") or ""),
                "target_geometry_family": _score_relation_target_action_family_signature(
                    target_key=str(row.get("target_key", "") or ""),
                    action="",
                    candidate_relation="",
                    probe_kind="",
                ).split("\t")[3:],
                "falsified_count": 0,
                "visible_effect_no_score_falsified_count": 0,
                "noop_falsified_count": 0,
                "no_progress_falsified_count": 0,
                "example_exact_signatures": [],
                "example_target_keys": [],
                "game_ids": set(),
            },
        )
        falsified_count = _as_int(row.get("falsified_count"), 0)
        stats["falsified_count"] = _as_int(stats.get("falsified_count"), 0) + falsified_count
        for state_key in (
            "visible_effect_no_score_falsified_count",
            "noop_falsified_count",
            "no_progress_falsified_count",
        ):
            stats[state_key] = _as_int(stats.get(state_key), 0) + _as_int(
                row.get(state_key), 0
            )
        examples = stats.get("example_exact_signatures")
        if isinstance(examples, list) and len(examples) < 4:
            examples.append(str(row.get("signature", "") or ""))
        targets = stats.get("example_target_keys")
        if isinstance(targets, list) and len(targets) < 4:
            targets.append(str(row.get("target_key", "") or ""))
        if isinstance(stats.get("game_ids"), set):
            for game_id in list(row.get("game_ids") or []):
                stats["game_ids"].add(str(game_id))

    family_rows: list[dict[str, Any]] = []
    for stats in family_stats.values():
        out = {key: value for key, value in stats.items() if key != "game_ids"}
        out["game_ids"] = (
            sorted(stats.get("game_ids", set()))
            if isinstance(stats.get("game_ids"), set)
            else []
        )
        family_rows.append(out)
    family_rows.sort(
        key=lambda row: (
            -_as_int(row.get("falsified_count"), 0),
            str(row.get("family_signature", "")),
        )
    )

    candidate_count = 0
    direction_confirmed_candidate_count = 0
    blocked_exact_signature_count = 0
    blocked_exact_signature_observation_count = 0
    blocked_family_scope_count = 0
    blocked_family_scope_observation_count = 0
    family_clear_count = 0
    family_clear_observation_count = 0
    admission_state_counts: Counter[str] = Counter()
    admission_rows: list[dict[str, Any]] = []

    for row in list(direction.get("target_rows") or []):
        if not isinstance(row, Mapping):
            continue
        if (
            str(row.get("direction_state", "") or "")
            != "effect_toward_objective_confirmed"
        ):
            continue
        count = max(1, _as_int(row.get("observation_count"), 1))
        target_key = str(row.get("target_key", "") or "")
        action = _normalize_action_name(row.get("dominant_action", ""))
        candidate_relation = str(row.get("candidate_relation", "") or "")
        probe_kind = str(row.get("probe_kind", "") or "")
        exact_signature = _score_relation_target_action_signature(
            target_key=target_key,
            action=action,
            candidate_relation=candidate_relation,
            probe_kind=probe_kind,
        )
        family_signature = _score_relation_target_action_family_signature(
            target_key=target_key,
            action=action,
            candidate_relation=candidate_relation,
            probe_kind=probe_kind,
        )
        candidate_count += count
        direction_confirmed_candidate_count += count
        if exact_signature in exact_falsified_signatures:
            admission_state = "blocked_exact_falsified_signature"
            blocked_exact_signature_count += 1
            blocked_exact_signature_observation_count += count
        elif family_signature in family_stats:
            admission_state = "blocked_family_scope_negative_memory"
            blocked_family_scope_count += 1
            blocked_family_scope_observation_count += count
        else:
            admission_state = "family_scope_clear_pending_other_admission_controls"
            family_clear_count += 1
            family_clear_observation_count += count
        admission_state_counts[admission_state] += count
        if len(admission_rows) < sample_limit:
            family = family_stats.get(family_signature, {})
            admission_rows.append(
                {
                    "target_key": target_key,
                    "exact_signature": exact_signature,
                    "family_signature": family_signature,
                    "candidate_site": list(row.get("candidate_site") or []),
                    "observation_count": int(count),
                    "admission_state": admission_state,
                    "exact_falsified_signature_match": bool(
                        exact_signature in exact_falsified_signatures
                    ),
                    "family_negative_memory_match": bool(
                        family_signature in family_stats
                    ),
                    "family_falsified_count": _as_int(
                        family.get("falsified_count"), 0
                    ),
                    "family_example_target_keys": list(
                        family.get("example_target_keys") or []
                    ),
                    "game_ids": list(row.get("game_ids") or []),
                }
            )

    return {
        "schema_version": "eca_score_relation_family_scope_negative_memory_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "source_falsified_memory_sensor": str(
            exact_memory.get("schema_version", "") or ""
        ),
        "source_direction_sensor": str(direction.get("schema_version", "") or ""),
        "family_key_fields": [
            "action_family",
            "candidate_relation_family",
            "probe_kind_family",
            "target_kind",
            "target_operator",
            "target_site_edge_class",
            "target_site_horizontal_half",
            "target_site_x_band",
            "target_site_y_band",
        ],
        "exact_falsified_signature_count": int(len(exact_falsified_signatures)),
        "family_falsified_signature_count": int(len(family_rows)),
        "candidate_count": int(candidate_count),
        "direction_confirmed_candidate_count": int(direction_confirmed_candidate_count),
        "blocked_exact_signature_count": int(blocked_exact_signature_count),
        "blocked_exact_signature_observation_count": int(
            blocked_exact_signature_observation_count
        ),
        "blocked_family_scope_count": int(blocked_family_scope_count),
        "blocked_family_scope_observation_count": int(
            blocked_family_scope_observation_count
        ),
        "family_clear_count": int(family_clear_count),
        "family_clear_observation_count": int(family_clear_observation_count),
        "admission_state_counts": dict(sorted(admission_state_counts.items())),
        "family_rows": family_rows[:sample_limit],
        "admission_rows": admission_rows,
        "promotion_state": (
            "family_scope_negative_memory_blocks_direction_confirmed_candidates"
            if blocked_exact_signature_count > 0 or blocked_family_scope_count > 0
            else (
                "family_scope_clear_pending_other_admission_controls"
                if family_clear_count > 0
                else (
                    "family_negative_memory_present_no_direction_candidates"
                    if family_rows
                    else "no_family_scope_negative_memory"
                )
            )
        ),
        "next_planner_contract": {
            "required_before_execution": "candidate action/relation/geometry family must not match prior falsified family memory",
            "allowed_connection": "diagnostic_only_negative_admission_guard",
            "action_generation_allowed": False,
        },
    }


def _score_relation_site_effect_intervention_delayed_outcome_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    window: int = 40,
    sample_limit: int = 16,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_target_key: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        final_score = _as_int(row.get("final_score"), 0)
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            if not bool(unsolved.get("score_relation_candidate_verifier")):
                continue
            if not bool(unsolved.get("executed") or unsolved.get("used")):
                continue
            if str(unsolved.get("candidate_kind", "") or "") != (
                "score_relation_site_effect_action6_reprobe"
            ):
                continue
            if str(unsolved.get("verifier_schema", "") or "") != (
                "site_effect_action6_reprobe_schema"
            ):
                continue

            candidate = unsolved.get("score_relation_candidate")
            candidate = candidate if isinstance(candidate, Mapping) else {}
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            target_key = str(candidate.get("target_key", "") or "unknown")
            candidate_relation = str(
                candidate.get("candidate_relation", "") or "unknown"
            )
            probe_kind = str(candidate.get("probe_kind", "") or "unknown")
            current_score = _trace_score_before(item, final_score)
            if index + 1 < len(trace):
                next_score = _trace_score_before(trace[index + 1], current_score)
            else:
                next_score = final_score
            immediate_delta = max(
                0,
                _as_int(outcome.get("score_delta"), 0),
                int(next_score) - int(current_score),
            )
            delayed_delta, delay_steps = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=index,
                window=window,
                default_score=current_score,
                final_score=final_score,
            )
            shifted_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=min(index + 1, max(0, len(trace) - 1)),
                window=window,
                default_score=next_score,
                final_score=final_score,
            )
            pre_action_delta = _pre_action_score_delta(
                trace,
                index=index,
                window=window,
                current_score=current_score,
            )
            control_index = _deterministic_control_index(index, len(trace))
            random_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=control_index,
                window=window,
                default_score=_trace_score_before(trace[control_index], current_score)
                if trace
                else current_score,
                final_score=final_score,
            )
            control_max = max(shifted_delta, pre_action_delta, random_delta)
            join_margin = int(delayed_delta) - int(control_max)
            immediate_state = str(
                "score_relation_confirmed"
                if immediate_delta > 0
                else "immediate_unconfirmed"
            )
            if delayed_delta > 0 and immediate_delta <= 0:
                delayed_state = "delayed_score_associated_no_immediate_score"
            elif delayed_delta > 0:
                delayed_state = "delayed_score_associated_with_immediate_score"
            else:
                delayed_state = "no_delayed_score_association"
            if delayed_delta > 0 and join_margin > 0:
                control_state = "positive_control_margin"
            elif delayed_delta > 0:
                control_state = "control_matched_or_ambiguous"
            else:
                control_state = "no_delayed_score"

            counts["execution_count"] += 1
            counts[f"immediate_state:{immediate_state}"] += 1
            counts[f"delayed_state:{delayed_state}"] += 1
            counts[f"control_state:{control_state}"] += 1
            counts["immediate_score_delta_sum"] += int(immediate_delta)
            counts["delayed_score_delta_sum"] += int(delayed_delta)
            counts["shifted_delayed_score_delta_sum"] += int(shifted_delta)
            counts["pre_action_score_delta_sum"] += int(pre_action_delta)
            counts["random_step_score_delta_sum"] += int(random_delta)
            counts["negative_control_max_score_delta_sum"] += int(control_max)
            counts["join_margin_sum"] += int(join_margin)
            if bool(unsolved.get("schema_guarded_execution")):
                counts["schema_guarded_execution_count"] += 1
            if bool(unsolved.get("guarded_execution")):
                counts["guarded_execution_count"] += 1
            if delayed_delta > 0:
                counts["delayed_positive_execution_count"] += 1
            if delayed_delta > 0 and immediate_delta <= 0:
                counts["delayed_positive_no_immediate_count"] += 1
            if delayed_delta > 0 and join_margin > 0:
                counts["positive_control_margin_count"] += 1
            if delayed_delta > 0 and join_margin <= 0:
                counts["control_matched_delayed_positive_count"] += 1

            for counter in (by_target_key[target_key], by_game[game_id]):
                counter["execution_count"] += 1
                counter[f"immediate_state:{immediate_state}"] += 1
                counter[f"delayed_state:{delayed_state}"] += 1
                counter[f"control_state:{control_state}"] += 1
                counter["immediate_score_delta_sum"] += int(immediate_delta)
                counter["delayed_score_delta_sum"] += int(delayed_delta)
                counter["negative_control_max_score_delta_sum"] += int(control_max)
                counter["join_margin_sum"] += int(join_margin)
                if delayed_delta > 0:
                    counter["delayed_positive_execution_count"] += 1
                if delayed_delta > 0 and immediate_delta <= 0:
                    counter["delayed_positive_no_immediate_count"] += 1
                if delayed_delta > 0 and join_margin > 0:
                    counter["positive_control_margin_count"] += 1

            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), index),
                        "phase": str(item.get("phase", "") or ""),
                        "primitive": str(item.get("primitive", "") or ""),
                        "action": str(item.get("action", "") or ""),
                        "target_key": target_key,
                        "candidate_relation": candidate_relation,
                        "probe_kind": probe_kind,
                        "target_site": list(unsolved.get("target_site") or []),
                        "selected_payload": dict(unsolved.get("selected_payload") or {}),
                        "score_before": int(current_score),
                        "immediate_score_delta": int(immediate_delta),
                        "delayed_score_delta": int(delayed_delta),
                        "delay_steps": int(delay_steps),
                        "shifted_delayed_score_delta": int(shifted_delta),
                        "pre_action_score_delta": int(pre_action_delta),
                        "random_step_score_delta": int(random_delta),
                        "negative_control_max_score_delta": int(control_max),
                        "join_margin": int(join_margin),
                        "immediate_state": immediate_state,
                        "delayed_state": delayed_state,
                        "control_state": control_state,
                        "runtime_verdict": str(outcome.get("verdict", "") or ""),
                        "runtime_effect_type": str(
                            outcome.get("effect_type", "") or ""
                        ),
                    }
                )

    execution_count = int(counts.get("execution_count", 0))
    delayed_no_immediate = int(counts.get("delayed_positive_no_immediate_count", 0))
    positive_control_margin = int(counts.get("positive_control_margin_count", 0))
    if execution_count <= 0:
        promotion_state = "no_site_effect_execution_evidence"
    elif positive_control_margin > 0:
        promotion_state = "delayed_intervention_candidate"
    elif delayed_no_immediate > 0:
        promotion_state = "delayed_association_control_matched"
    else:
        promotion_state = "no_delayed_score_association"

    return {
        "schema_version": (
            "eca_score_relation_site_effect_intervention_delayed_outcome_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "window": int(window),
        "execution_count": execution_count,
        "schema_guarded_execution_count": int(
            counts.get("schema_guarded_execution_count", 0)
        ),
        "guarded_execution_count": int(counts.get("guarded_execution_count", 0)),
        "immediate_score_delta_sum": int(counts.get("immediate_score_delta_sum", 0)),
        "delayed_score_delta_sum": int(counts.get("delayed_score_delta_sum", 0)),
        "negative_control_max_score_delta_sum": int(
            counts.get("negative_control_max_score_delta_sum", 0)
        ),
        "join_margin_sum": int(counts.get("join_margin_sum", 0)),
        "delayed_positive_execution_count": int(
            counts.get("delayed_positive_execution_count", 0)
        ),
        "delayed_positive_no_immediate_count": delayed_no_immediate,
        "positive_control_margin_count": positive_control_margin,
        "control_matched_delayed_positive_count": int(
            counts.get("control_matched_delayed_positive_count", 0)
        ),
        "promotion_state": promotion_state,
        "state_counts": {
            key: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("immediate_state:")
            or key.startswith("delayed_state:")
            or key.startswith("control_state:")
        },
        "by_target_key": {
            target_key: dict(sorted(stats.items()))
            for target_key, stats in sorted(by_target_key.items())
        },
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "positive_control_margin_count > 0 with focused preservation and broader non-regression",
            "allowed_connection": "diagnostic_only_until_delayed_causal_margin",
            "action_generation_allowed": False,
        },
    }


def _score_relation_site_effect_target_action_signature_promotion_guard_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    window: int = 40,
    sample_limit: int = 16,
) -> dict[str, Any]:
    by_signature: dict[str, Counter[str]] = defaultdict(Counter)
    signature_meta: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        final_score = _as_int(row.get("final_score"), 0)
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            if not bool(unsolved.get("score_relation_candidate_verifier")):
                continue
            if not bool(unsolved.get("executed") or unsolved.get("used")):
                continue
            if str(unsolved.get("candidate_kind", "") or "") != (
                "score_relation_site_effect_action6_reprobe"
            ):
                continue
            if str(unsolved.get("verifier_schema", "") or "") != (
                "site_effect_action6_reprobe_schema"
            ):
                continue

            candidate = unsolved.get("score_relation_candidate")
            candidate = candidate if isinstance(candidate, Mapping) else {}
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            target_key = str(candidate.get("target_key", "") or "unknown")
            candidate_relation = str(
                candidate.get("candidate_relation", "") or "unknown"
            )
            probe_kind = str(candidate.get("probe_kind", "") or "unknown")
            selected_action = str(
                unsolved.get("selected_action")
                or candidate.get("dominant_action")
                or item.get("action")
                or ""
            )
            preserved = unsolved.get("score_bearing_path_preservation_guard_bypass")
            preserved = preserved if isinstance(preserved, Mapping) else {}
            preserved_primitive = str(
                preserved.get("preserved_primitive_id")
                or item.get("primitive")
                or ""
            )
            preserved_action = str(item.get("action", "") or "")
            preserved_scope = str(
                preserved.get("preserved_primitive_scope")
                or row.get("generalisation_tier")
                or ""
            )
            signature_key = "\t".join(
                [
                    candidate_relation,
                    probe_kind,
                    target_key,
                    selected_action,
                    preserved_primitive,
                    preserved_action,
                ]
            )
            current_score = _trace_score_before(item, final_score)
            if index + 1 < len(trace):
                next_score = _trace_score_before(trace[index + 1], current_score)
            else:
                next_score = final_score
            immediate_delta = max(
                0,
                _as_int(outcome.get("score_delta"), 0),
                int(next_score) - int(current_score),
            )
            delayed_delta, delay_steps = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=index,
                window=window,
                default_score=current_score,
                final_score=final_score,
            )
            shifted_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=min(index + 1, max(0, len(trace) - 1)),
                window=window,
                default_score=next_score,
                final_score=final_score,
            )
            pre_action_delta = _pre_action_score_delta(
                trace,
                index=index,
                window=window,
                current_score=current_score,
            )
            control_index = _deterministic_control_index(index, len(trace))
            random_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=control_index,
                window=window,
                default_score=_trace_score_before(trace[control_index], current_score)
                if trace
                else current_score,
                final_score=final_score,
            )
            control_max = max(shifted_delta, pre_action_delta, random_delta)
            join_margin = int(delayed_delta) - int(control_max)

            stats = by_signature[signature_key]
            stats["execution_count"] += 1
            stats["immediate_score_delta_sum"] += int(immediate_delta)
            stats["delayed_score_delta_sum"] += int(delayed_delta)
            stats["negative_control_max_score_delta_sum"] += int(control_max)
            stats["join_margin_sum"] += int(join_margin)
            if immediate_delta > 0:
                stats["score_relation_confirmed_count"] += 1
            if delayed_delta > 0 and immediate_delta <= 0:
                stats["delayed_positive_no_immediate_count"] += 1
            if delayed_delta > 0 and join_margin > 0:
                stats["positive_control_margin_count"] += 1
            if delayed_delta > 0 and join_margin <= 0:
                stats["control_matched_delayed_positive_count"] += 1

            signature_meta.setdefault(
                signature_key,
                {
                    "signature_key": signature_key,
                    "candidate_relation": candidate_relation,
                    "probe_kind": probe_kind,
                    "target_key": target_key,
                    "dominant_action": selected_action,
                    "selected_action": selected_action,
                    "verifier_schema": str(unsolved.get("verifier_schema", "") or ""),
                    "candidate_kind": str(unsolved.get("candidate_kind", "") or ""),
                    "preserved_primitive_id": preserved_primitive,
                    "preserved_action": preserved_action,
                    "preserved_primitive_scope": preserved_scope,
                    "target_site": list(unsolved.get("target_site") or []),
                },
            )
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), index),
                        "signature_key": signature_key,
                        "target_key": target_key,
                        "dominant_action": selected_action,
                        "preserved_primitive_id": preserved_primitive,
                        "preserved_action": preserved_action,
                        "score_before": int(current_score),
                        "immediate_score_delta": int(immediate_delta),
                        "delayed_score_delta": int(delayed_delta),
                        "delay_steps": int(delay_steps),
                        "negative_control_max_score_delta": int(control_max),
                        "join_margin": int(join_margin),
                    }
                )

    rows_out: list[dict[str, Any]] = []
    for signature_key, stats in sorted(by_signature.items()):
        meta = dict(signature_meta.get(signature_key) or {})
        immediate_sum = int(stats.get("immediate_score_delta_sum", 0))
        delayed_sum = int(stats.get("delayed_score_delta_sum", 0))
        control_max_sum = int(stats.get("negative_control_max_score_delta_sum", 0))
        join_margin_sum = int(stats.get("join_margin_sum", 0))
        positive_margin_count = int(stats.get("positive_control_margin_count", 0))
        confirmed_count = int(stats.get("score_relation_confirmed_count", 0))
        causal_confirmed = confirmed_count > 0
        robust_margin = bool(
            positive_margin_count > 0
            and join_margin_sum > 0
            and delayed_sum > control_max_sum
        )
        if causal_confirmed:
            promotion_state = "score_relation_confirmed"
            guard_reason = "causal_margin_confirmed"
            guard_decision = "allow"
        elif robust_margin:
            promotion_state = "positive_control_margin"
            guard_reason = "causal_margin_confirmed"
            guard_decision = "allow"
        elif delayed_sum > 0:
            promotion_state = "delayed_association_control_matched"
            guard_reason = "causal_margin_guard_blocked"
            guard_decision = "block"
        else:
            promotion_state = "no_delayed_score_association"
            guard_reason = "no_causal_margin_evidence"
            guard_decision = "block"
        rows_out.append(
            {
                **meta,
                "execution_count": int(stats.get("execution_count", 0)),
                "immediate_score_delta_sum": immediate_sum,
                "delayed_score_delta_sum": delayed_sum,
                "negative_control_max_score_delta_sum": control_max_sum,
                "join_margin_sum": join_margin_sum,
                "positive_control_margin_count": positive_margin_count,
                "score_relation_confirmed_count": confirmed_count,
                "delayed_positive_no_immediate_count": int(
                    stats.get("delayed_positive_no_immediate_count", 0)
                ),
                "control_matched_delayed_positive_count": int(
                    stats.get("control_matched_delayed_positive_count", 0)
                ),
                "causal_score_attribution_confirmed": bool(causal_confirmed),
                "robust_positive_control_margin": bool(robust_margin),
                "promotion_state": promotion_state,
                "guard_decision": guard_decision,
                "guard_reason": guard_reason,
            }
        )

    allowed_rows = [row for row in rows_out if row["guard_decision"] == "allow"]
    blocked_rows = [row for row in rows_out if row["guard_decision"] == "block"]
    return {
        "schema_version": (
            "eca_score_relation_site_effect_target_action_signature_promotion_guard_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "window": int(window),
        "input_posterior_key": (
            "score_relation_site_effect_target_action_signature_promotion_guard"
        ),
        "row_count": int(len(rows_out)),
        "allowed_row_count": int(len(allowed_rows)),
        "blocked_row_count": int(len(blocked_rows)),
        "rows": rows_out,
        "allowed_rows": allowed_rows[:20],
        "blocked_rows": blocked_rows[:20],
        "samples": samples,
        "next_planner_contract": {
            "allowed_connection": "diagnostic_only_until_focused_broader_all25_gates",
            "required_before_guard_enablement": "allowed_row_count > 0 with focused lift/objective progress and breadth/all25 non-regression",
            "action_generation_allowed": False,
        },
    }


def _score_relation_site_effect_target_action_signature_promotion_guard_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    window: int = 40,
    sample_limit: int = 16,
) -> dict[str, Any]:
    by_signature: dict[str, Counter[str]] = defaultdict(Counter)
    signature_meta: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        final_score = _as_int(row.get("final_score"), 0)
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            if not bool(unsolved.get("score_relation_candidate_verifier")):
                continue
            if not bool(unsolved.get("executed") or unsolved.get("used")):
                continue
            if str(unsolved.get("candidate_kind", "") or "") != (
                "score_relation_site_effect_action6_reprobe"
            ):
                continue
            if str(unsolved.get("verifier_schema", "") or "") != (
                "site_effect_action6_reprobe_schema"
            ):
                continue

            candidate = unsolved.get("score_relation_candidate")
            candidate = candidate if isinstance(candidate, Mapping) else {}
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            target_key = str(candidate.get("target_key", "") or "unknown")
            candidate_relation = str(
                candidate.get("candidate_relation", "") or "unknown"
            )
            probe_kind = str(candidate.get("probe_kind", "") or "unknown")
            selected_action = str(
                unsolved.get("selected_action")
                or candidate.get("dominant_action")
                or item.get("action")
                or ""
            )
            preserved = unsolved.get("score_bearing_path_preservation_guard_bypass")
            preserved = preserved if isinstance(preserved, Mapping) else {}
            preserved_primitive = str(
                preserved.get("preserved_primitive_id")
                or item.get("primitive")
                or ""
            )
            preserved_action = str(item.get("action", "") or "")
            preserved_scope = str(
                preserved.get("preserved_primitive_scope")
                or row.get("generalisation_tier")
                or ""
            )
            signature_key = "\t".join(
                [
                    candidate_relation,
                    probe_kind,
                    target_key,
                    selected_action,
                    preserved_primitive,
                    preserved_action,
                ]
            )
            current_score = _trace_score_before(item, final_score)
            if index + 1 < len(trace):
                next_score = _trace_score_before(trace[index + 1], current_score)
            else:
                next_score = final_score
            immediate_delta = max(
                0,
                _as_int(outcome.get("score_delta"), 0),
                int(next_score) - int(current_score),
            )
            delayed_delta, delay_steps = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=index,
                window=window,
                default_score=current_score,
                final_score=final_score,
            )
            shifted_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=min(index + 1, max(0, len(trace) - 1)),
                window=window,
                default_score=next_score,
                final_score=final_score,
            )
            pre_action_delta = _pre_action_score_delta(
                trace,
                index=index,
                window=window,
                current_score=current_score,
            )
            control_index = _deterministic_control_index(index, len(trace))
            random_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=control_index,
                window=window,
                default_score=_trace_score_before(trace[control_index], current_score)
                if trace
                else current_score,
                final_score=final_score,
            )
            control_max = max(shifted_delta, pre_action_delta, random_delta)
            join_margin = int(delayed_delta) - int(control_max)

            stats = by_signature[signature_key]
            stats["execution_count"] += 1
            stats["immediate_score_delta_sum"] += int(immediate_delta)
            stats["delayed_score_delta_sum"] += int(delayed_delta)
            stats["negative_control_max_score_delta_sum"] += int(control_max)
            stats["join_margin_sum"] += int(join_margin)
            if immediate_delta > 0:
                stats["score_relation_confirmed_count"] += 1
            if delayed_delta > 0 and immediate_delta <= 0:
                stats["delayed_positive_no_immediate_count"] += 1
            if delayed_delta > 0 and join_margin > 0:
                stats["positive_control_margin_count"] += 1
            if delayed_delta > 0 and join_margin <= 0:
                stats["control_matched_delayed_positive_count"] += 1

            signature_meta.setdefault(
                signature_key,
                {
                    "signature_key": signature_key,
                    "candidate_relation": candidate_relation,
                    "probe_kind": probe_kind,
                    "target_key": target_key,
                    "dominant_action": selected_action,
                    "selected_action": selected_action,
                    "verifier_schema": str(unsolved.get("verifier_schema", "") or ""),
                    "candidate_kind": str(unsolved.get("candidate_kind", "") or ""),
                    "preserved_primitive_id": preserved_primitive,
                    "preserved_action": preserved_action,
                    "preserved_primitive_scope": preserved_scope,
                    "target_site": list(unsolved.get("target_site") or []),
                },
            )
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), index),
                        "signature_key": signature_key,
                        "target_key": target_key,
                        "dominant_action": selected_action,
                        "preserved_primitive_id": preserved_primitive,
                        "preserved_action": preserved_action,
                        "score_before": int(current_score),
                        "immediate_score_delta": int(immediate_delta),
                        "delayed_score_delta": int(delayed_delta),
                        "delay_steps": int(delay_steps),
                        "negative_control_max_score_delta": int(control_max),
                        "join_margin": int(join_margin),
                    }
                )

    rows_out: list[dict[str, Any]] = []
    for signature_key, stats in sorted(by_signature.items()):
        meta = dict(signature_meta.get(signature_key) or {})
        immediate_sum = int(stats.get("immediate_score_delta_sum", 0))
        delayed_sum = int(stats.get("delayed_score_delta_sum", 0))
        control_max_sum = int(stats.get("negative_control_max_score_delta_sum", 0))
        join_margin_sum = int(stats.get("join_margin_sum", 0))
        positive_margin_count = int(stats.get("positive_control_margin_count", 0))
        confirmed_count = int(stats.get("score_relation_confirmed_count", 0))
        causal_confirmed = confirmed_count > 0
        robust_margin = bool(
            positive_margin_count > 0
            and join_margin_sum > 0
            and delayed_sum > control_max_sum
        )
        if causal_confirmed:
            promotion_state = "score_relation_confirmed"
            guard_reason = "causal_margin_confirmed"
            guard_decision = "allow"
        elif robust_margin:
            promotion_state = "positive_control_margin"
            guard_reason = "causal_margin_confirmed"
            guard_decision = "allow"
        elif delayed_sum > 0:
            promotion_state = "delayed_association_control_matched"
            guard_reason = "causal_margin_guard_blocked"
            guard_decision = "block"
        else:
            promotion_state = "no_delayed_score_association"
            guard_reason = "no_causal_margin_evidence"
            guard_decision = "block"
        rows_out.append(
            {
                **meta,
                "execution_count": int(stats.get("execution_count", 0)),
                "immediate_score_delta_sum": immediate_sum,
                "delayed_score_delta_sum": delayed_sum,
                "negative_control_max_score_delta_sum": control_max_sum,
                "join_margin_sum": join_margin_sum,
                "positive_control_margin_count": positive_margin_count,
                "score_relation_confirmed_count": confirmed_count,
                "delayed_positive_no_immediate_count": int(
                    stats.get("delayed_positive_no_immediate_count", 0)
                ),
                "control_matched_delayed_positive_count": int(
                    stats.get("control_matched_delayed_positive_count", 0)
                ),
                "causal_score_attribution_confirmed": bool(causal_confirmed),
                "robust_positive_control_margin": bool(robust_margin),
                "promotion_state": promotion_state,
                "guard_decision": guard_decision,
                "guard_reason": guard_reason,
            }
        )

    allowed_rows = [row for row in rows_out if row["guard_decision"] == "allow"]
    blocked_rows = [row for row in rows_out if row["guard_decision"] == "block"]
    return {
        "schema_version": (
            "eca_score_relation_site_effect_target_action_signature_promotion_guard_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "window": int(window),
        "input_posterior_key": (
            "score_relation_site_effect_target_action_signature_promotion_guard"
        ),
        "row_count": int(len(rows_out)),
        "allowed_row_count": int(len(allowed_rows)),
        "blocked_row_count": int(len(blocked_rows)),
        "rows": rows_out,
        "allowed_rows": allowed_rows[:20],
        "blocked_rows": blocked_rows[:20],
        "samples": samples,
        "next_planner_contract": {
            "allowed_connection": "diagnostic_only_until_focused_broader_all25_gates",
            "required_before_guard_enablement": "allowed_row_count > 0 with focused lift/objective progress and breadth/all25 non-regression",
            "action_generation_allowed": False,
        },
    }


def _site_effect_selector_first_seen_bucket(step: int) -> str:
    value = int(step)
    if value <= 18:
        return "early_le18"
    if value <= 60:
        return "mid_le60"
    return "late_gt60"


def _site_effect_selector_count_bucket(count: int) -> str:
    value = int(count)
    if value >= 18:
        return "very_high_ge18"
    if value >= 15:
        return "high_ge15"
    if value >= 10:
        return "medium_ge10"
    return "low_lt10"


def _site_effect_selector_shadow_state(
    *,
    dominant_operator: str,
    gap_count: int,
    first_seen_step: int,
    observation_count: int,
) -> str:
    first_bucket = _site_effect_selector_first_seen_bucket(first_seen_step)
    count_bucket = _site_effect_selector_count_bucket(observation_count)
    if (
        int(gap_count) == 2
        and first_bucket == "early_le18"
        and count_bucket == "medium_ge10"
        and dominant_operator in {"click", "probe"}
    ):
        return "phase551_in_sample_score3_feature_match"
    return "phase551_feature_nonmatch"


def _site_effect_target_site_from_key(target_key: str) -> tuple[int, int] | None:
    parts = str(target_key or "").split("|")
    if len(parts) < 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except (TypeError, ValueError):
        return None


def _site_effect_target_geometry_features(target_key: str) -> dict[str, str]:
    site = _site_effect_target_site_from_key(target_key)
    if site is None:
        return {
            "target_site_edge_class": "unknown",
            "target_site_horizontal_half": "unknown",
        }
    x, y = site
    if x <= 4 or y <= 4 or x >= 59 or y >= 59:
        edge_class = "edge_touching"
    elif x <= 12 or y <= 12 or x >= 51 or y >= 51:
        edge_class = "near_edge"
    else:
        edge_class = "interior"
    return {
        "target_site_edge_class": edge_class,
        "target_site_horizontal_half": "right_half" if y >= 32 else "left_half",
    }


def _score_relation_target_action_signature(
    *,
    target_key: str,
    action: str,
    candidate_relation: str,
    probe_kind: str,
) -> str:
    return "\t".join(
        [
            str(target_key or ""),
            _normalize_action_name(action),
            str(candidate_relation or ""),
            str(probe_kind or ""),
        ]
    )


def _score_relation_target_action_family_signature(
    *,
    target_key: str,
    action: str,
    candidate_relation: str,
    probe_kind: str,
) -> str:
    parts = str(target_key or "").split("|")
    target_kind = parts[0] if parts else "unknown"
    target_operator = parts[3] if len(parts) >= 4 else "unknown"
    site = _site_effect_target_site_from_key(target_key)
    if site is None:
        x_band = "x_unknown"
        y_band = "y_unknown"
    else:
        x, y = site
        if x <= 21:
            x_band = "x_left"
        elif x <= 42:
            x_band = "x_middle"
        else:
            x_band = "x_right"
        if y <= 21:
            y_band = "y_top"
        elif y <= 42:
            y_band = "y_middle"
        else:
            y_band = "y_bottom"
    geometry = _site_effect_target_geometry_features(target_key)
    return "\t".join(
        [
            _normalize_action_name(action),
            str(candidate_relation or ""),
            str(probe_kind or ""),
            str(target_kind or "unknown"),
            str(target_operator or "unknown"),
            str(geometry.get("target_site_edge_class", "unknown")),
            str(geometry.get("target_site_horizontal_half", "unknown")),
            x_band,
            y_band,
        ]
    )


def _objective_bbox_from_focus_key(value: Any) -> tuple[int, int, int, int] | None:
    text = str(value or "").strip()
    if not text.startswith("target:"):
        return None
    parts = text.split(":")
    if len(parts) < 3:
        return None
    try:
        values = [int(part.strip()) for part in parts[-1].split(",")]
    except (TypeError, ValueError):
        return None
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = values
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _bbox_centroid(bbox: tuple[int, int, int, int] | None) -> list[float]:
    if bbox is None:
        return []
    x1, y1, x2, y2 = bbox
    return [round((float(x1) + float(x2)) / 2.0, 6), round((float(y1) + float(y2)) / 2.0, 6)]


def _state_hashes_from_value(
    value: Any,
    *,
    limit: int = 8,
    _depth: int = 0,
) -> list[str]:
    if _depth > 6 or limit <= 0:
        return []
    found: list[str] = []
    if isinstance(value, Mapping):
        direct = value.get("state_hash")
        if direct is not None:
            text = str(direct or "").strip()
            if text:
                found.append(text)
        for child_key, child_value in value.items():
            if len(found) >= limit:
                break
            if child_key == "state_hash":
                continue
            for item in _state_hashes_from_value(
                child_value,
                limit=limit - len(found),
                _depth=_depth + 1,
            ):
                if item not in found:
                    found.append(item)
                    if len(found) >= limit:
                        break
    elif isinstance(value, list):
        for child_value in value:
            if len(found) >= limit:
                break
            for item in _state_hashes_from_value(
                child_value,
                limit=limit - len(found),
                _depth=_depth + 1,
            ):
                if item not in found:
                    found.append(item)
                    if len(found) >= limit:
                        break
    return found[:limit]


def _objective_bbox_from_value(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        values = [int(part) for part in value]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = values
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _objective_bbox_records_from_trace_item(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    roots: list[Any] = []
    for key in ("coordinate_hypothesis", "control_target", "target_verification"):
        payload = item.get(key)
        if isinstance(payload, (Mapping, list)):
            roots.append(payload)
    records: dict[tuple[int, int, int, int], dict[str, Any]] = {}

    def add_record(
        bbox: tuple[int, int, int, int] | None,
        *,
        source: str,
        target_key: str = "",
    ) -> None:
        if bbox is None:
            return
        existing = records.setdefault(
            bbox,
            {
                "target_key": target_key,
                "bbox": list(bbox),
                "source": source,
            },
        )
        if target_key and not str(existing.get("target_key", "") or ""):
            existing["target_key"] = target_key

    def visit(value: Any, source: str) -> None:
        if isinstance(value, Mapping):
            focus_key = str(value.get("target_focus_key", "") or "")
            add_record(
                _objective_bbox_from_focus_key(focus_key),
                source=source,
                target_key=focus_key,
            )
            target_key = str(value.get("target_key", "") or "")
            add_record(
                _objective_bbox_from_focus_key(target_key),
                source=source,
                target_key=target_key,
            )
            add_record(
                _objective_bbox_from_value(value.get("target_bbox")),
                source=source,
                target_key=focus_key or target_key,
            )
            preferred = value.get("preferred_target_keys")
            if isinstance(preferred, list):
                for entry in preferred:
                    entry_key = str(entry or "")
                    add_record(
                        _objective_bbox_from_focus_key(entry_key),
                        source=source,
                        target_key=entry_key,
                    )
            for child in value.values():
                if isinstance(child, (Mapping, list)):
                    visit(child, source)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (Mapping, list)):
                    visit(child, source)

    for root in roots:
        visit(root, "trace_item")
    return list(records.values())


def _site_to_bbox_metrics(
    site: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> dict[str, Any]:
    x, y = site
    x1, y1, x2, y2 = bbox
    cx = (float(x1) + float(x2)) / 2.0
    cy = (float(y1) + float(y2)) / 2.0
    inside = bool(x1 <= x <= x2 and y1 <= y <= y2)
    dx_edge = 0.0 if x1 <= x <= x2 else min(abs(float(x) - x1), abs(float(x) - x2))
    dy_edge = 0.0 if y1 <= y <= y2 else min(abs(float(y) - y1), abs(float(y) - y2))
    edge_distance = math.sqrt(dx_edge * dx_edge + dy_edge * dy_edge)
    centroid_distance = math.sqrt((float(x) - cx) ** 2 + (float(y) - cy) ** 2)
    return {
        "inside": inside,
        "edge_distance": round(edge_distance, 6),
        "centroid_distance": round(centroid_distance, 6),
    }


def _xy_from_value(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _kinematic_canvas_coupling_rows_from_trace_item(
    item: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_row(raw: Any) -> None:
        if not isinstance(raw, Mapping):
            return
        selected_site = _xy_from_value(raw.get("selected_site"))
        if selected_site is None:
            return
        target_key = str(
            raw.get("target_key", "") or raw.get("target_focus_key", "") or ""
        )
        status = str(raw.get("status", "") or "insufficient_evidence")
        rows.append(
            {
                "status": status,
                "target_key": target_key,
                "target_focus_key": target_key,
                "selected_site": [int(selected_site[0]), int(selected_site[1])],
                "canvas_delta": [
                    int(value)
                    for value in list(raw.get("canvas_delta") or [])
                    if isinstance(value, int)
                    or (isinstance(value, str) and value.strip().lstrip("-").isdigit())
                ][:2],
                "overlap_delta": _as_int(raw.get("overlap_delta"), 0),
                "distance_delta": _as_int(raw.get("distance_delta"), 0),
                "score_delta": _as_int(raw.get("score_delta"), 0),
                "posterior_key": str(raw.get("posterior_key", "") or ""),
            }
        )

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            sensor = value.get("kinematic_canvas_coupling_sensor")
            if isinstance(sensor, Mapping):
                for row in list(sensor.get("rows") or []):
                    add_row(row)
            evidence = value.get("kinematic_canvas_coupling_evidence")
            if isinstance(evidence, list):
                for row in evidence:
                    add_row(row)
            posterior = value.get("kinematic_canvas_coupling_posterior")
            if isinstance(posterior, Mapping):
                for row in list(posterior.get("recent") or []):
                    add_row(row)
            for child in value.values():
                if isinstance(child, (Mapping, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (Mapping, list)):
                    visit(child)

    for key in ("coordinate_hypothesis", "control_target", "target_verification"):
        payload = item.get(key)
        if isinstance(payload, (Mapping, list)):
            visit(payload)

    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        site = tuple(row.get("selected_site") or [])
        key = (
            site,
            str(row.get("target_key", "") or ""),
            str(row.get("status", "") or ""),
            _as_int(row.get("overlap_delta"), 0),
            _as_int(row.get("distance_delta"), 0),
            _as_int(row.get("score_delta"), 0),
        )
        unique.setdefault(key, row)
    return list(unique.values())


KINEMATIC_MEMBERSHIP_CANDIDATE_MINER_SCHEMA_VERSION = (
    "eca_kinematic_membership_candidate_miner.v1"
)
KINEMATIC_MEMBERSHIP_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION = (
    "eca_kinematic_membership_counterfactual_control_sensor.v1"
)
KINEMATIC_MEMBERSHIP_ALL25_NON_REGRESSION_SCHEMA_VERSION = (
    "eca_kinematic_membership_all25_non_regression_sensor.v1"
)


def _structured_group_completion_sensor_rows_from_trace_item(
    item: Mapping[str, Any],
    sensor_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            sensor = value.get(sensor_name)
            if isinstance(sensor, Mapping):
                for row in list(sensor.get("rows") or []):
                    if isinstance(row, Mapping):
                        rows.append(dict(row))
            for child in value.values():
                if isinstance(child, (Mapping, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (Mapping, list)):
                    visit(child)

    for key in ("coordinate_hypothesis", "control_target", "target_verification"):
        payload = item.get(key)
        if isinstance(payload, (Mapping, list)):
            visit(payload)
    return rows


def _first_sensor_row_by_target(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        target_key = str(
            row.get("target_focus_key", "") or row.get("target_key", "") or ""
        )
        if target_key and target_key not in out:
            out[target_key] = dict(row)
    return out


def _rank_guard_row_for_candidate(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_key: str,
    selected_site: list[int],
) -> dict[str, Any]:
    for row in rows:
        row_key = str(
            row.get("target_focus_key", "") or row.get("target_key", "") or ""
        )
        row_site = _xy_from_value(row.get("site") or row.get("selected_site"))
        if row_key != target_key or row_site is None:
            continue
        if [int(row_site[0]), int(row_site[1])] == list(selected_site[:2]):
            return dict(row)
    for row in rows:
        row_key = str(
            row.get("target_focus_key", "") or row.get("target_key", "") or ""
        )
        if row_key == target_key:
            return dict(row)
    return {}


def summarize_kinematic_membership_candidate_miner(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Mine guarded candidate descriptions from observed kinematic membership."""

    by_game: dict[str, Counter[str]] = {}
    by_candidate_kind: dict[str, Counter[str]] = {}
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, tuple[int, int]]] = set()
    coupling_evidence_row_count = 0
    readiness_row_count = 0
    posterior_row_count = 0
    rank_guard_row_count = 0

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        variant = str(row.get("variant", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            step = _as_int(item.get("step"), 0)
            coupling_rows = _kinematic_canvas_coupling_rows_from_trace_item(item)
            if not coupling_rows:
                continue
            readiness_rows = _structured_group_completion_sensor_rows_from_trace_item(
                item,
                "target_color_support_readiness_sensor",
            )
            posterior_rows = _structured_group_completion_sensor_rows_from_trace_item(
                item,
                "target_membership_causal_posterior_sensor",
            )
            rank_rows = (
                _structured_group_completion_sensor_rows_from_trace_item(
                    item,
                    "kinematic_membership_rank_guard",
                )
                + _structured_group_completion_sensor_rows_from_trace_item(
                    item,
                    "target_membership_causal_posterior_rank_guard",
                )
            )
            coupling_evidence_row_count += int(len(coupling_rows))
            readiness_row_count += int(len(readiness_rows))
            posterior_row_count += int(len(posterior_rows))
            rank_guard_row_count += int(len(rank_rows))
            readiness_by_target = _first_sensor_row_by_target(readiness_rows)
            posterior_by_target = _first_sensor_row_by_target(posterior_rows)

            for evidence in coupling_rows:
                target_key = str(
                    evidence.get("target_focus_key", "")
                    or evidence.get("target_key", "")
                    or ""
                )
                selected_site = [
                    int(value)
                    for value in list(evidence.get("selected_site") or [])
                    if isinstance(value, int)
                    or (
                        isinstance(value, str)
                        and value.strip().lstrip("-").isdigit()
                    )
                ][:2]
                status = str(
                    evidence.get("status", "") or "insufficient_evidence"
                )
                if len(selected_site) < 2 or not target_key:
                    if len(rejected) < sample_limit:
                        rejected.append(
                            {
                                "game_id": game_id,
                                "step": int(step),
                                "rejected_reason": "missing_target_or_site",
                                "coupling_status": status,
                                "target_focus_key": target_key,
                                "selected_site": selected_site,
                            }
                        )
                    continue
                key = (game_id, target_key, status, tuple(selected_site[:2]))
                if key in seen:
                    continue
                seen.add(key)

                candidate_kind = ""
                evidence_role = ""
                if status == "confirmed":
                    candidate_kind = (
                        "kinematic_confirmed_membership_replay_candidate"
                    )
                    evidence_role = "confirmed_canvas_coupling"
                elif status == "falsified":
                    candidate_kind = (
                        "kinematic_falsified_membership_control_candidate"
                    )
                    evidence_role = "falsified_canvas_coupling"
                elif status == "ambiguous":
                    candidate_kind = (
                        "kinematic_ambiguous_membership_control_candidate"
                    )
                    evidence_role = "ambiguous_canvas_coupling"
                else:
                    if len(rejected) < sample_limit:
                        rejected.append(
                            {
                                "game_id": game_id,
                                "variant": variant,
                                "step": int(step),
                                "rejected_reason": "insufficient_coupling_evidence",
                                "coupling_status": status,
                                "target_focus_key": target_key,
                                "selected_site": selected_site,
                            }
                        )
                    continue

                readiness = readiness_by_target.get(target_key, {})
                posterior = posterior_by_target.get(target_key, {})
                rank_guard = _rank_guard_row_for_candidate(
                    rank_rows,
                    target_key=target_key,
                    selected_site=selected_site,
                )
                candidate_key = (
                    "kinematic_membership:"
                    f"{game_id}:{candidate_kind}:{target_key}:"
                    f"{selected_site[0]},{selected_site[1]}"
                )
                candidate = {
                    "candidate_key": candidate_key,
                    "candidate_kind": candidate_kind,
                    "game_id": game_id,
                    "variant": variant,
                    "step": int(step),
                    "target_focus_key": target_key,
                    "selected_site_observed": selected_site,
                    "source_sensor": "kinematic_canvas_coupling_sensor",
                    "evidence_role": evidence_role,
                    "coupling_status": status,
                    "canvas_delta": [
                        int(value)
                        for value in list(evidence.get("canvas_delta") or [])
                    ][:2],
                    "overlap_delta": _as_int(evidence.get("overlap_delta"), 0),
                    "distance_delta": _as_int(evidence.get("distance_delta"), 0),
                    "score_delta": _as_int(evidence.get("score_delta"), 0),
                    "readiness_strength": str(
                        readiness.get("readiness_strength", "") or ""
                    ),
                    "kinematic_membership_required": bool(
                        readiness.get("kinematic_membership_required")
                    ),
                    "causal_state": str(posterior.get("causal_state", "") or ""),
                    "planner_disposition": str(
                        posterior.get("planner_disposition", "") or ""
                    ),
                    "rank_signal": str(posterior.get("rank_signal", "") or ""),
                    "rank_guard_reason": str(
                        rank_guard.get("rank_reason", "")
                        or rank_guard.get("base_rank_reason", "")
                        or ""
                    ),
                    "rank_guard_score": _as_int(
                        rank_guard.get("rank_score", rank_guard.get("base_rank_score")),
                        0,
                    ),
                    "requires_guarded_probe": True,
                    "counterfactual_control_required": True,
                    "existing_candidate_set_only": True,
                    "runtime_execution_enabled": False,
                    "planner_connection_enabled": False,
                    "candidate_generation_enabled": False,
                    "ranking_enabled": False,
                    "positive_boost_enabled": False,
                    "action_generation_allowed": False,
                }
                candidates.append(candidate)
                by_game[game_id]["candidate_count"] += 1
                by_game[game_id][f"candidate_kind:{candidate_kind}"] += 1
                by_candidate_kind.setdefault(candidate_kind, Counter())
                by_candidate_kind[candidate_kind]["candidate_count"] += 1

    candidate_count = int(len(candidates))
    confirmed_count = int(
        by_candidate_kind.get(
            "kinematic_confirmed_membership_replay_candidate",
            Counter(),
        ).get("candidate_count", 0)
    )
    falsified_count = int(
        by_candidate_kind.get(
            "kinematic_falsified_membership_control_candidate",
            Counter(),
        ).get("candidate_count", 0)
    )
    ambiguous_count = int(
        by_candidate_kind.get(
            "kinematic_ambiguous_membership_control_candidate",
            Counter(),
        ).get("candidate_count", 0)
    )
    if candidate_count > 0:
        promotion_state = "kinematic_membership_candidate_miner_supported"
    elif coupling_evidence_row_count > 0:
        promotion_state = "kinematic_membership_candidate_miner_blocked"
    else:
        promotion_state = "kinematic_membership_candidate_miner_no_evidence"

    return {
        "schema_version": KINEMATIC_MEMBERSHIP_CANDIDATE_MINER_SCHEMA_VERSION,
        "diagnostic_only": True,
        "default_off": True,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "coupling_evidence_row_count": int(coupling_evidence_row_count),
        "readiness_row_count": int(readiness_row_count),
        "posterior_row_count": int(posterior_row_count),
        "rank_guard_row_count": int(rank_guard_row_count),
        "candidate_count": int(candidate_count),
        "confirmed_replay_candidate_count": int(confirmed_count),
        "falsified_control_candidate_count": int(falsified_count),
        "ambiguous_control_candidate_count": int(ambiguous_count),
        "rejected_candidate_count": int(len(rejected)),
        "promotion_state": promotion_state,
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "by_candidate_kind": {
            kind: dict(sorted(counter.items()))
            for kind, counter in sorted(by_candidate_kind.items())
        },
        "candidates": candidates[:sample_limit],
        "rejected": rejected[:sample_limit],
        "next_runtime_contract": {
            "required_before_counterfactual_control": (
                "confirmed, ambiguous, and falsified kinematic membership "
                "candidate descriptions must be paired with score/effect controls"
            ),
            "allowed_connection": "diagnostic_candidate_description_only",
            "existing_candidate_set_only": True,
            "action_generation_allowed": False,
        },
    }


def summarize_kinematic_membership_counterfactual_controls(
    rows: Iterable[Mapping[str, Any]],
    *,
    candidate_summary: Mapping[str, Any] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Pair confirmed membership candidates with falsified/ambiguous controls."""

    candidate_sensor = (
        dict(candidate_summary)
        if isinstance(candidate_summary, Mapping)
        else summarize_kinematic_membership_candidate_miner(
            rows,
            sample_limit=max(int(sample_limit), 128),
        )
    )
    candidates = [
        dict(row)
        for row in list(candidate_sensor.get("candidates") or [])
        if isinstance(row, Mapping)
    ]
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    by_game: dict[str, Counter[str]] = {}
    for candidate in candidates:
        game_id = str(candidate.get("game_id", "") or "unknown")
        target_key = str(candidate.get("target_focus_key", "") or "")
        kind = str(candidate.get("candidate_kind", "") or "")
        if not target_key:
            continue
        by_game.setdefault(game_id, Counter())
        key = (game_id, target_key)
        bucket = grouped.setdefault(key, {"confirmed": [], "control": []})
        if kind == "kinematic_confirmed_membership_replay_candidate":
            bucket["confirmed"].append(candidate)
            by_game[game_id]["confirmed_candidate_count"] += 1
        elif kind in {
            "kinematic_falsified_membership_control_candidate",
            "kinematic_ambiguous_membership_control_candidate",
        }:
            bucket["control"].append(candidate)
            by_game[game_id]["control_candidate_count"] += 1
            by_game[game_id][f"control_kind:{kind}"] += 1

    pair_rows: list[dict[str, Any]] = []
    pair_count = 0
    confirmed_without_control = 0
    control_without_confirmed = 0
    for (game_id, target_key), bucket in sorted(grouped.items()):
        confirmed = bucket["confirmed"]
        controls = bucket["control"]
        if confirmed and controls:
            pair_count += 1
            by_game.setdefault(game_id, Counter())
            by_game[game_id]["counterfactual_pair_count"] += 1
            if len(pair_rows) < sample_limit:
                pair_rows.append(
                    {
                        "game_id": game_id,
                        "target_focus_key": target_key,
                        "confirmed_candidate_count": int(len(confirmed)),
                        "control_candidate_count": int(len(controls)),
                        "confirmed_sites": [
                            list(row.get("selected_site_observed") or [])[:2]
                            for row in confirmed[:4]
                        ],
                        "control_sites": [
                            list(row.get("selected_site_observed") or [])[:2]
                            for row in controls[:4]
                        ],
                        "control_kinds": sorted(
                            {
                                str(row.get("candidate_kind", "") or "")
                                for row in controls
                            }
                        ),
                        "counterfactual_state": (
                            "confirmed_membership_with_local_controls"
                        ),
                    }
                )
        elif confirmed:
            confirmed_without_control += 1
        elif controls:
            control_without_confirmed += 1

    confirmed_count = _as_int(
        candidate_sensor.get("confirmed_replay_candidate_count"),
        0,
    )
    falsified_count = _as_int(
        candidate_sensor.get("falsified_control_candidate_count"),
        0,
    )
    ambiguous_count = _as_int(
        candidate_sensor.get("ambiguous_control_candidate_count"),
        0,
    )
    control_count = int(falsified_count) + int(ambiguous_count)
    if pair_count > 0:
        promotion_state = "kinematic_membership_counterfactual_control_supported"
    elif confirmed_count > 0 and control_count <= 0:
        promotion_state = "kinematic_membership_counterfactual_control_missing_controls"
    elif control_count > 0 and confirmed_count <= 0:
        promotion_state = "kinematic_membership_counterfactual_control_missing_confirmed"
    elif _as_int(candidate_sensor.get("candidate_count"), 0) > 0:
        promotion_state = "kinematic_membership_counterfactual_control_unpaired"
    else:
        promotion_state = "kinematic_membership_counterfactual_control_no_candidates"

    return {
        "schema_version": KINEMATIC_MEMBERSHIP_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION,
        "diagnostic_only": True,
        "default_off": True,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "candidate_miner_promotion_state": str(
            candidate_sensor.get("promotion_state", "") or ""
        ),
        "candidate_count": _as_int(candidate_sensor.get("candidate_count"), 0),
        "confirmed_replay_candidate_count": int(confirmed_count),
        "control_candidate_count": int(control_count),
        "falsified_control_candidate_count": int(falsified_count),
        "ambiguous_control_candidate_count": int(ambiguous_count),
        "counterfactual_pair_count": int(pair_count),
        "confirmed_target_without_control_count": int(confirmed_without_control),
        "control_target_without_confirmed_count": int(control_without_confirmed),
        "promotion_state": promotion_state,
        "counterfactual_stage_supported": promotion_state
        == "kinematic_membership_counterfactual_control_supported",
        "planner_promotion_blocked": True,
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "counterfactual_pairs": pair_rows[:sample_limit],
        "next_runtime_contract": {
            "required_before_all25_non_regression": (
                "counterfactual pairs must preserve focused behavior and then "
                "pass an all25 opt-in non-regression gate before planner review"
            ),
            "allowed_connection": "diagnostic_counterfactual_pairing_only",
            "existing_candidate_set_only": True,
            "action_generation_allowed": False,
        },
    }


def summarize_kinematic_membership_all25_non_regression(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_score_sum: int | None = None,
    expected_by_game: Mapping[str, Any] | None = None,
    counterfactual_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize all25 non-regression for kinematic membership diagnostics."""

    row_list = [dict(row) for row in rows if isinstance(row, Mapping)]
    expected_map = {
        str(game_id): _as_int(score, 0)
        for game_id, score in dict(expected_by_game or {}).items()
    }
    actual_by_game: dict[str, int] = {}
    ok_count = 0
    error_count = 0
    adapter_free_score_sum = 0
    uses_game_conditioned_count = 0
    for row in row_list:
        game_id = str(row.get("game_id", "") or "unknown")
        score = _as_int(row.get("final_score"), 0)
        actual_by_game[game_id] = actual_by_game.get(game_id, 0) + int(score)
        if str(row.get("status", "") or "") == "ok":
            ok_count += 1
        elif str(row.get("status", "") or ""):
            error_count += 1
        if _is_adapter_free_row(row):
            adapter_free_score_sum += int(score)
        if _uses_game_conditioned_primitive(row):
            uses_game_conditioned_count += 1

    score_sum = int(sum(actual_by_game.values()))
    regressions: dict[str, dict[str, int]] = {}
    improvements: dict[str, dict[str, int]] = {}
    if expected_map:
        for game_id in sorted(set(expected_map) | set(actual_by_game)):
            expected = int(expected_map.get(game_id, 0))
            actual = int(actual_by_game.get(game_id, 0))
            if actual < expected:
                regressions[game_id] = {
                    "expected": expected,
                    "actual": actual,
                    "delta": actual - expected,
                }
            elif actual > expected:
                improvements[game_id] = {
                    "expected": expected,
                    "actual": actual,
                    "delta": actual - expected,
                }
    score_sum_matches = (
        True
        if expected_score_sum is None
        else int(score_sum) == int(expected_score_sum)
    )
    if row_list and score_sum_matches and not regressions:
        promotion_state = "kinematic_membership_all25_non_regression_supported"
    elif row_list and regressions:
        promotion_state = "kinematic_membership_all25_non_regression_regressed"
    elif row_list:
        promotion_state = "kinematic_membership_all25_non_regression_score_sum_mismatch"
    else:
        promotion_state = "kinematic_membership_all25_non_regression_no_rows"

    counterfactual = (
        dict(counterfactual_summary)
        if isinstance(counterfactual_summary, Mapping)
        else {}
    )
    return {
        "schema_version": KINEMATIC_MEMBERSHIP_ALL25_NON_REGRESSION_SCHEMA_VERSION,
        "diagnostic_only": True,
        "default_off": True,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "run_count": int(len(row_list)),
        "ok_count": int(ok_count),
        "error_count": int(error_count),
        "score_sum": int(score_sum),
        "adapter_free_score_sum": int(adapter_free_score_sum),
        "uses_game_conditioned_primitive_count": int(uses_game_conditioned_count),
        "expected_score_sum": expected_score_sum,
        "score_sum_delta_vs_reference": (
            None
            if expected_score_sum is None
            else int(score_sum) - int(expected_score_sum)
        ),
        "actual_by_game": dict(sorted(actual_by_game.items())),
        "expected_by_game": dict(sorted(expected_map.items())),
        "regressed_game_count": int(len(regressions)),
        "improved_game_count": int(len(improvements)),
        "regressions": regressions,
        "improvements": improvements,
        "counterfactual_promotion_state": str(
            counterfactual.get("promotion_state", "") or ""
        ),
        "counterfactual_pair_count": _as_int(
            counterfactual.get("counterfactual_pair_count"),
            0,
        ),
        "promotion_state": promotion_state,
        "ready_for_planner_promotion_review": bool(
            promotion_state == "kinematic_membership_all25_non_regression_supported"
            and str(counterfactual.get("promotion_state", "") or "")
            == "kinematic_membership_counterfactual_control_supported"
        ),
        "next_runtime_contract": {
            "required_before_planner_review": (
                "all25 non-regression must preserve the reference profile with "
                "no game-conditioned primitives and no runtime/action generation"
            ),
            "allowed_connection": "promotion_review_only",
            "existing_candidate_set_only": True,
            "action_generation_allowed": False,
        },
    }


def _site_effect_selector_shadow_v2_state(
    *,
    target_key: str,
    dominant_operator: str,
    gap_count: int,
    first_seen_step: int,
    observation_count: int,
) -> str:
    first_bucket = _site_effect_selector_first_seen_bucket(first_seen_step)
    count_bucket = _site_effect_selector_count_bucket(observation_count)
    geometry = _site_effect_target_geometry_features(target_key)
    if int(gap_count) != 2 or first_bucket != "early_le18":
        return "phase554_feature_nonmatch"
    if (
        dominant_operator == "probe"
        and count_bucket == "medium_ge10"
        and geometry["target_site_edge_class"] == "edge_touching"
    ):
        return "phase554_score3_feature_match"
    if (
        dominant_operator == "click"
        and count_bucket == "high_ge15"
        and geometry["target_site_horizontal_half"] == "right_half"
    ):
        return "phase554_score3_feature_match"
    return "phase554_feature_nonmatch"


def _score_relation_site_effect_selector_shadow_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    target_stats: dict[str, dict[str, Any]] = {}
    feature_signature_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    event_count = 0
    raw_candidate_count = 0

    def observe(
        *,
        game_id: str,
        target_key: str,
        candidate_relation: str,
        probe_kind: str,
        dominant_operator: str,
        dominant_action: str,
        gap_count: int,
        verifier_schema: str,
        verifier_schema_role: str,
        step: int,
        count: int,
    ) -> None:
        nonlocal raw_candidate_count
        if (
            candidate_relation != "site_effect_to_score_delta_relation"
            or probe_kind != "site_reprobe_score_relation_probe"
            or dominant_action != "ACTION6"
            or verifier_schema != "site_effect_action6_reprobe_schema"
            or verifier_schema_role != "positive_action"
        ):
            return
        raw_candidate_count += int(count)
        stats = target_stats.setdefault(
            target_key,
            {
                "target_key": target_key,
                "candidate_relation": candidate_relation,
                "probe_kind": probe_kind,
                "dominant_operator": dominant_operator,
                "dominant_action": dominant_action,
                "gap_count": int(gap_count),
                "verifier_schema": verifier_schema,
                "verifier_schema_role": verifier_schema_role,
                "observation_count": 0,
                "first_seen_step": int(step),
                "last_seen_step": int(step),
                "game_ids": set(),
            },
        )
        stats["observation_count"] = int(stats.get("observation_count", 0)) + int(
            count
        )
        stats["first_seen_step"] = min(int(stats.get("first_seen_step", step)), int(step))
        stats["last_seen_step"] = max(int(stats.get("last_seen_step", step)), int(step))
        stats["gap_count"] = min(int(stats.get("gap_count", gap_count)), int(gap_count))
        if isinstance(stats.get("game_ids"), set):
            stats["game_ids"].add(game_id)
        feature_key = "\t".join(
            [
                dominant_operator,
                str(int(gap_count)),
                verifier_schema,
                verifier_schema_role,
            ]
        )
        feature_signature_counts[feature_key] += int(count)
        by_game[game_id]["site_effect_selector_candidate_count"] += int(count)

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            preflight = unsolved.get("score_relation_candidate_verifier_preflight")
            if not isinstance(preflight, Mapping):
                continue
            shadow = preflight.get("score_relation_shadow_evidence")
            if not isinstance(shadow, Mapping) or not bool(shadow.get("enabled")):
                continue
            step = _as_int(item.get("step"), 0)
            target_feature_counts = shadow.get(
                "candidate_selector_target_feature_counts"
            )
            observed_from_counts = False
            if isinstance(target_feature_counts, Mapping):
                for key, count_value in target_feature_counts.items():
                    parts = str(key).split("\t")
                    if len(parts) != 8:
                        continue
                    (
                        target_key,
                        candidate_relation,
                        probe_kind,
                        dominant_operator,
                        dominant_action,
                        gap_text,
                        verifier_schema,
                        verifier_schema_role,
                    ) = parts
                    observe(
                        game_id=game_id,
                        target_key=target_key,
                        candidate_relation=candidate_relation,
                        probe_kind=probe_kind,
                        dominant_operator=dominant_operator,
                        dominant_action=dominant_action,
                        gap_count=_as_int(gap_text, 0),
                        verifier_schema=verifier_schema,
                        verifier_schema_role=verifier_schema_role,
                        step=step,
                        count=_as_int(count_value, 0),
                    )
                    observed_from_counts = True
            if not observed_from_counts:
                for sample in list(shadow.get("samples") or []):
                    if not isinstance(sample, Mapping):
                        continue
                    observe(
                        game_id=game_id,
                        target_key=str(sample.get("target_key", "") or ""),
                        candidate_relation=str(
                            sample.get("candidate_relation", "") or ""
                        ),
                        probe_kind=str(sample.get("probe_kind", "") or ""),
                        dominant_operator=str(
                            sample.get("dominant_operator", "") or ""
                        ),
                        dominant_action=str(
                            sample.get("dominant_action", "") or ""
                        ),
                        gap_count=_as_int(sample.get("gap_count"), 0),
                        verifier_schema=str(
                            sample.get("verifier_schema", "") or ""
                        ),
                        verifier_schema_role=str(
                            sample.get("verifier_schema_role", "") or ""
                        ),
                        step=step,
                        count=1,
                    )
            event_count += 1

    target_rows: list[dict[str, Any]] = []
    for stats in target_stats.values():
        observation_count = int(stats.get("observation_count", 0))
        first_seen_step = int(stats.get("first_seen_step", 0))
        state = _site_effect_selector_shadow_state(
            dominant_operator=str(stats.get("dominant_operator", "") or ""),
            gap_count=_as_int(stats.get("gap_count"), 0),
            first_seen_step=first_seen_step,
            observation_count=observation_count,
        )
        first_bucket = _site_effect_selector_first_seen_bucket(first_seen_step)
        count_bucket = _site_effect_selector_count_bucket(observation_count)
        geometry = _site_effect_target_geometry_features(
            str(stats.get("target_key", "") or "")
        )
        v2_state = _site_effect_selector_shadow_v2_state(
            target_key=str(stats.get("target_key", "") or ""),
            dominant_operator=str(stats.get("dominant_operator", "") or ""),
            gap_count=_as_int(stats.get("gap_count"), 0),
            first_seen_step=first_seen_step,
            observation_count=observation_count,
        )
        feature_signature = "\t".join(
            [
                str(stats.get("dominant_operator", "") or ""),
                str(_as_int(stats.get("gap_count"), 0)),
                first_bucket,
                count_bucket,
            ]
        )
        target_rows.append(
            {
                "target_key": str(stats.get("target_key", "") or ""),
                "candidate_relation": str(stats.get("candidate_relation", "") or ""),
                "probe_kind": str(stats.get("probe_kind", "") or ""),
                "dominant_operator": str(stats.get("dominant_operator", "") or ""),
                "dominant_action": str(stats.get("dominant_action", "") or ""),
                "gap_count": _as_int(stats.get("gap_count"), 0),
                "verifier_schema": str(stats.get("verifier_schema", "") or ""),
                "observation_count": observation_count,
                "first_seen_step": first_seen_step,
                "last_seen_step": _as_int(stats.get("last_seen_step"), first_seen_step),
                "first_seen_step_bucket": first_bucket,
                "observation_count_bucket": count_bucket,
                **geometry,
                "selector_feature_signature": feature_signature,
                "selector_shadow_state": state,
                "selector_shadow_rank": 1
                if state == "phase551_in_sample_score3_feature_match"
                else 0,
                "selector_shadow_v2_state": v2_state,
                "selector_shadow_v2_rank": 1
                if v2_state == "phase554_score3_feature_match"
                else 0,
                "game_ids": sorted(stats.get("game_ids", set()))
                if isinstance(stats.get("game_ids"), set)
                else [],
            }
        )
    target_rows.sort(
        key=lambda row: (
            -_as_int(row.get("selector_shadow_rank"), 0),
            -_as_int(row.get("observation_count"), 0),
            _as_int(row.get("first_seen_step"), 0),
            str(row.get("target_key", "")),
        )
    )
    selected = [
        row
        for row in target_rows
        if str(row.get("selector_shadow_state", ""))
        == "phase551_in_sample_score3_feature_match"
    ]
    selected_v2 = [
        row
        for row in target_rows
        if str(row.get("selector_shadow_v2_state", ""))
        == "phase554_score3_feature_match"
    ]
    return {
        "schema_version": "eca_score_relation_site_effect_selector_shadow_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "source_feature_rule": "phase551_pre_execution_shadow_coarse",
        "shadow_event_count": int(event_count),
        "raw_site_effect_candidate_count": int(raw_candidate_count),
        "target_candidate_count": int(len(target_rows)),
        "selector_shadow_candidate_count": int(len(selected)),
        "selector_shadow_target_keys": [row["target_key"] for row in selected],
        "selector_shadow_v2_source_feature_rule": (
            "phase554_pre_execution_shadow_geometry_v2"
        ),
        "selector_shadow_v2_candidate_count": int(len(selected_v2)),
        "selector_shadow_v2_target_keys": [row["target_key"] for row in selected_v2],
        "feature_signature_counts": dict(sorted(feature_signature_counts.items())),
        "target_rows": target_rows[:sample_limit],
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "promotion_state": (
            "selector_shadow_candidates_present_no_causal_margin"
            if selected
            else "no_selector_shadow_candidates"
        ),
        "next_planner_contract": {
            "required_before_ranking": "selector-shadow targets must beat paired/counterfactual controls and broader non-regression",
            "allowed_connection": "diagnostic_only_until_counterfactual_margin",
            "action_generation_allowed": False,
        },
    }


def _score_relation_progress_goal_binding_candidate_mining_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
    min_gap_count: int = 2,
) -> dict[str, Any]:
    event_count = 0
    raw_candidate_count = 0
    min_gap_ready_candidate_count = 0
    min_gap_blocked_candidate_count = 0
    missing_target_site_count = 0
    unsupported_action_count = 0
    schema_not_ready_count = 0
    gap_count_histogram: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    operator_counts: Counter[str] = Counter()
    verifier_schema_counts: Counter[str] = Counter()
    verifier_schema_role_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    target_stats: dict[str, dict[str, Any]] = {}

    progress_schemas = {
        "progress_goal_binding_action6_reprobe_schema",
        "progress_goal_binding_action_replay_observation_schema",
    }

    def observe(
        *,
        game_id: str,
        target_key: str,
        candidate_relation: str,
        probe_kind: str,
        dominant_operator: str,
        dominant_action: str,
        gap_count: int,
        verifier_schema: str,
        verifier_schema_role: str,
        step: int,
        count: int,
    ) -> int:
        nonlocal raw_candidate_count
        nonlocal min_gap_ready_candidate_count
        nonlocal min_gap_blocked_candidate_count
        nonlocal missing_target_site_count
        nonlocal unsupported_action_count
        nonlocal schema_not_ready_count
        if (
            candidate_relation != "progress_to_goal_binding_relation"
            or probe_kind != "goal_binding_relation_probe"
        ):
            return 0
        count = max(0, int(count))
        if count <= 0:
            return 0

        gap = max(0, int(gap_count))
        target_site = _site_effect_target_site_from_key(target_key)
        has_target_site = target_site is not None
        action_supported = dominant_action in CORE_ACTIONS
        schema_ready = (
            verifier_schema in progress_schemas
            and verifier_schema_role == "positive_action"
        )
        ready = bool(
            gap >= int(min_gap_count)
            and has_target_site
            and action_supported
            and schema_ready
        )

        raw_candidate_count += count
        gap_count_histogram[str(gap)] += count
        action_counts[dominant_action or ""] += count
        operator_counts[dominant_operator or ""] += count
        verifier_schema_counts[verifier_schema or ""] += count
        verifier_schema_role_counts[verifier_schema_role or ""] += count
        by_game[game_id]["progress_goal_binding_candidate_count"] += count
        by_game[game_id][f"gap:{gap}"] += count
        by_game[game_id][f"action:{dominant_action or ''}"] += count
        if ready:
            min_gap_ready_candidate_count += count
            by_game[game_id]["min_gap_ready_candidate_count"] += count
        else:
            if gap < int(min_gap_count):
                min_gap_blocked_candidate_count += count
                by_game[game_id]["min_gap_blocked_candidate_count"] += count
            if not has_target_site:
                missing_target_site_count += count
                by_game[game_id]["missing_target_site_count"] += count
            if not action_supported:
                unsupported_action_count += count
                by_game[game_id]["unsupported_action_count"] += count
            if not schema_ready:
                schema_not_ready_count += count
                by_game[game_id]["schema_not_ready_count"] += count

        key = "\t".join([target_key, dominant_action, str(gap), verifier_schema])
        stats = target_stats.setdefault(
            key,
            {
                "target_key": target_key,
                "candidate_relation": candidate_relation,
                "probe_kind": probe_kind,
                "dominant_operator": dominant_operator,
                "dominant_action": dominant_action,
                "gap_count": int(gap),
                "verifier_schema": verifier_schema,
                "verifier_schema_role": verifier_schema_role,
                "has_target_site": bool(has_target_site),
                "target_site": list(target_site) if target_site is not None else [],
                "action_supported": bool(action_supported),
                "schema_ready": bool(schema_ready),
                "min_gap_ready": bool(ready),
                "observation_count": 0,
                "first_seen_step": int(step),
                "last_seen_step": int(step),
                "game_ids": set(),
            },
        )
        stats["observation_count"] = int(stats.get("observation_count", 0)) + count
        stats["first_seen_step"] = min(int(stats.get("first_seen_step", step)), int(step))
        stats["last_seen_step"] = max(int(stats.get("last_seen_step", step)), int(step))
        if isinstance(stats.get("game_ids"), set):
            stats["game_ids"].add(game_id)
        return count

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            preflight = unsolved.get("score_relation_candidate_verifier_preflight")
            if not isinstance(preflight, Mapping):
                continue
            shadow = preflight.get("score_relation_shadow_evidence")
            if not isinstance(shadow, Mapping) or not bool(shadow.get("enabled")):
                continue
            step = _as_int(item.get("step"), 0)
            observed_count = 0
            target_feature_counts = shadow.get(
                "candidate_selector_target_feature_counts"
            )
            observed_from_counts = False
            if isinstance(target_feature_counts, Mapping):
                for key, count_value in target_feature_counts.items():
                    parts = str(key).split("\t")
                    if len(parts) != 8:
                        continue
                    (
                        target_key,
                        candidate_relation,
                        probe_kind,
                        dominant_operator,
                        dominant_action,
                        gap_text,
                        verifier_schema,
                        verifier_schema_role,
                    ) = parts
                    observed_count += observe(
                        game_id=game_id,
                        target_key=target_key,
                        candidate_relation=candidate_relation,
                        probe_kind=probe_kind,
                        dominant_operator=dominant_operator,
                        dominant_action=dominant_action,
                        gap_count=_as_int(gap_text, 0),
                        verifier_schema=verifier_schema,
                        verifier_schema_role=verifier_schema_role,
                        step=step,
                        count=_as_int(count_value, 0),
                    )
                    observed_from_counts = True
            if not observed_from_counts:
                for sample in list(shadow.get("samples") or []):
                    if not isinstance(sample, Mapping):
                        continue
                    observed_count += observe(
                        game_id=game_id,
                        target_key=str(sample.get("target_key", "") or ""),
                        candidate_relation=str(
                            sample.get("candidate_relation", "") or ""
                        ),
                        probe_kind=str(sample.get("probe_kind", "") or ""),
                        dominant_operator=str(
                            sample.get("dominant_operator", "") or ""
                        ),
                        dominant_action=str(
                            sample.get("dominant_action", "") or ""
                        ),
                        gap_count=_as_int(sample.get("gap_count"), 0),
                        verifier_schema=str(
                            sample.get("verifier_schema", "") or ""
                        ),
                        verifier_schema_role=str(
                            sample.get("verifier_schema_role", "") or ""
                        ),
                        step=step,
                        count=1,
                    )
            if observed_count > 0:
                event_count += 1
                by_game[game_id]["progress_goal_binding_event_count"] += 1

    target_rows: list[dict[str, Any]] = []
    for stats in target_stats.values():
        row = {
            key: value
            for key, value in stats.items()
            if key != "game_ids"
        }
        row["game_ids"] = (
            sorted(stats.get("game_ids", set()))
            if isinstance(stats.get("game_ids"), set)
            else []
        )
        target_rows.append(row)
    target_rows.sort(
        key=lambda row: (
            0 if bool(row.get("min_gap_ready")) else 1,
            -_as_int(row.get("observation_count"), 0),
            _as_int(row.get("gap_count"), 0),
            str(row.get("target_key", "")),
            str(row.get("dominant_action", "")),
        )
    )

    return {
        "schema_version": (
            "eca_score_relation_progress_goal_binding_candidate_mining_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "source_shadow_field": "candidate_selector_target_feature_counts",
        "min_gap_count": int(min_gap_count),
        "shadow_event_count": int(event_count),
        "raw_progress_goal_binding_candidate_count": int(raw_candidate_count),
        "target_candidate_count": int(len(target_rows)),
        "min_gap_ready_candidate_count": int(min_gap_ready_candidate_count),
        "min_gap_blocked_candidate_count": int(min_gap_blocked_candidate_count),
        "missing_target_site_count": int(missing_target_site_count),
        "unsupported_action_count": int(unsupported_action_count),
        "schema_not_ready_count": int(schema_not_ready_count),
        "gap_count_histogram": dict(sorted(gap_count_histogram.items())),
        "dominant_action_counts": dict(sorted(action_counts.items())),
        "dominant_operator_counts": dict(sorted(operator_counts.items())),
        "verifier_schema_counts": dict(sorted(verifier_schema_counts.items())),
        "verifier_schema_role_counts": dict(sorted(verifier_schema_role_counts.items())),
        "target_rows": target_rows[:sample_limit],
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "promotion_state": (
            "min_gap_ready_candidates_present_diagnostic_only"
            if min_gap_ready_candidate_count > 0
            else (
                "blocked_by_min_gap_or_schema"
                if raw_candidate_count > 0
                else "no_progress_goal_binding_candidates"
            )
        ),
        "next_planner_contract": {
            "required_before_execution": "focused target allowlist plus one-step objective-progress controls; keep ranking disconnected",
            "allowed_connection": "diagnostic_only_until_objective_progress_controls",
            "action_generation_allowed": False,
        },
    }


def _score_relation_progress_goal_binding_objective_binding_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
    near_target_distance: float = 2.0,
) -> dict[str, Any]:
    event_count = 0
    candidate_count = 0
    bound_candidate_count = 0
    unbound_candidate_count = 0
    no_objective_context_count = 0
    missing_site_count = 0
    binding_state_counts: Counter[str] = Counter()
    gap_count_histogram: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    target_stats: dict[str, dict[str, Any]] = {}

    def observe(
        *,
        game_id: str,
        item: Mapping[str, Any],
        target_key: str,
        candidate_relation: str,
        probe_kind: str,
        dominant_operator: str,
        dominant_action: str,
        gap_count: int,
        verifier_schema: str,
        verifier_schema_role: str,
        step: int,
        count: int,
    ) -> int:
        nonlocal candidate_count
        nonlocal bound_candidate_count
        nonlocal unbound_candidate_count
        nonlocal no_objective_context_count
        nonlocal missing_site_count
        if (
            candidate_relation != "progress_to_goal_binding_relation"
            or probe_kind != "goal_binding_relation_probe"
        ):
            return 0
        count = max(0, int(count))
        if count <= 0:
            return 0
        candidate_count += count
        gap = max(0, int(gap_count))
        gap_count_histogram[str(gap)] += count

        site = _site_effect_target_site_from_key(target_key)
        objective_records = _objective_bbox_records_from_trace_item(item)
        if site is None:
            binding_state = "missing_candidate_site"
            missing_site_count += count
            best: dict[str, Any] = {}
        elif not objective_records:
            binding_state = "no_objective_target_context"
            no_objective_context_count += count
            best = {}
        else:
            best_record: dict[str, Any] | None = None
            best_metrics: dict[str, Any] | None = None
            for record in objective_records:
                bbox = _objective_bbox_from_value(record.get("bbox"))
                if bbox is None:
                    continue
                metrics = _site_to_bbox_metrics(site, bbox)
                if best_metrics is None or float(metrics["edge_distance"]) < float(
                    best_metrics["edge_distance"]
                ):
                    best_record = record
                    best_metrics = metrics
            if best_record is None or best_metrics is None:
                binding_state = "no_objective_target_context"
                no_objective_context_count += count
                best = {}
            elif bool(best_metrics.get("inside")):
                binding_state = "candidate_site_inside_objective_target"
                bound_candidate_count += count
                best = {**dict(best_record), **dict(best_metrics)}
            elif float(best_metrics.get("edge_distance", 0.0)) <= float(
                near_target_distance
            ):
                binding_state = "candidate_site_near_objective_target"
                bound_candidate_count += count
                best = {**dict(best_record), **dict(best_metrics)}
            else:
                binding_state = "candidate_site_far_from_objective_target"
                unbound_candidate_count += count
                best = {**dict(best_record), **dict(best_metrics)}

        binding_state_counts[binding_state] += count
        by_game[game_id]["progress_goal_binding_objective_binding_candidate_count"] += count
        by_game[game_id][f"binding_state:{binding_state}"] += count
        key = "\t".join([target_key, dominant_action, str(gap), binding_state])
        stats = target_stats.setdefault(
            key,
            {
                "target_key": target_key,
                "candidate_relation": candidate_relation,
                "probe_kind": probe_kind,
                "dominant_operator": dominant_operator,
                "dominant_action": dominant_action,
                "gap_count": int(gap),
                "verifier_schema": verifier_schema,
                "verifier_schema_role": verifier_schema_role,
                "binding_state": binding_state,
                "candidate_site": list(site) if site is not None else [],
                "nearest_objective_target": best,
                "observation_count": 0,
                "first_seen_step": int(step),
                "last_seen_step": int(step),
                "game_ids": set(),
            },
        )
        stats["observation_count"] = int(stats.get("observation_count", 0)) + count
        stats["first_seen_step"] = min(int(stats.get("first_seen_step", step)), int(step))
        stats["last_seen_step"] = max(int(stats.get("last_seen_step", step)), int(step))
        if isinstance(stats.get("game_ids"), set):
            stats["game_ids"].add(game_id)
        return count

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            preflight = unsolved.get("score_relation_candidate_verifier_preflight")
            if not isinstance(preflight, Mapping):
                continue
            shadow = preflight.get("score_relation_shadow_evidence")
            if not isinstance(shadow, Mapping) or not bool(shadow.get("enabled")):
                continue
            step = _as_int(item.get("step"), 0)
            observed_count = 0
            target_feature_counts = shadow.get(
                "candidate_selector_target_feature_counts"
            )
            observed_from_counts = False
            if isinstance(target_feature_counts, Mapping):
                for key, count_value in target_feature_counts.items():
                    parts = str(key).split("\t")
                    if len(parts) != 8:
                        continue
                    (
                        target_key,
                        candidate_relation,
                        probe_kind,
                        dominant_operator,
                        dominant_action,
                        gap_text,
                        verifier_schema,
                        verifier_schema_role,
                    ) = parts
                    observed_count += observe(
                        game_id=game_id,
                        item=item,
                        target_key=target_key,
                        candidate_relation=candidate_relation,
                        probe_kind=probe_kind,
                        dominant_operator=dominant_operator,
                        dominant_action=dominant_action,
                        gap_count=_as_int(gap_text, 0),
                        verifier_schema=verifier_schema,
                        verifier_schema_role=verifier_schema_role,
                        step=step,
                        count=_as_int(count_value, 0),
                    )
                    observed_from_counts = True
            if not observed_from_counts:
                for sample in list(shadow.get("samples") or []):
                    if not isinstance(sample, Mapping):
                        continue
                    observed_count += observe(
                        game_id=game_id,
                        item=item,
                        target_key=str(sample.get("target_key", "") or ""),
                        candidate_relation=str(
                            sample.get("candidate_relation", "") or ""
                        ),
                        probe_kind=str(sample.get("probe_kind", "") or ""),
                        dominant_operator=str(
                            sample.get("dominant_operator", "") or ""
                        ),
                        dominant_action=str(
                            sample.get("dominant_action", "") or ""
                        ),
                        gap_count=_as_int(sample.get("gap_count"), 0),
                        verifier_schema=str(
                            sample.get("verifier_schema", "") or ""
                        ),
                        verifier_schema_role=str(
                            sample.get("verifier_schema_role", "") or ""
                        ),
                        step=step,
                        count=1,
                    )
            if observed_count > 0:
                event_count += 1
                by_game[game_id]["progress_goal_binding_objective_binding_event_count"] += 1

    target_rows: list[dict[str, Any]] = []
    for stats in target_stats.values():
        row = {
            key: value
            for key, value in stats.items()
            if key != "game_ids"
        }
        row["game_ids"] = (
            sorted(stats.get("game_ids", set()))
            if isinstance(stats.get("game_ids"), set)
            else []
        )
        target_rows.append(row)
    target_rows.sort(
        key=lambda row: (
            0
            if str(row.get("binding_state", "")).startswith("candidate_site_inside")
            else 1
            if str(row.get("binding_state", "")).startswith("candidate_site_near")
            else 2,
            -_as_int(row.get("observation_count"), 0),
            str(row.get("target_key", "")),
        )
    )

    return {
        "schema_version": (
            "eca_score_relation_progress_goal_binding_objective_binding_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "near_target_distance": float(near_target_distance),
        "shadow_event_count": int(event_count),
        "candidate_count": int(candidate_count),
        "target_candidate_count": int(len(target_rows)),
        "bound_candidate_count": int(bound_candidate_count),
        "unbound_candidate_count": int(unbound_candidate_count),
        "no_objective_context_count": int(no_objective_context_count),
        "missing_site_count": int(missing_site_count),
        "gap_count_histogram": dict(sorted(gap_count_histogram.items())),
        "binding_state_counts": dict(sorted(binding_state_counts.items())),
        "target_rows": target_rows[:sample_limit],
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "promotion_state": (
            "objective_bound_candidates_present_diagnostic_only"
            if bound_candidate_count > 0
            else (
                "objective_unbound_candidates"
                if candidate_count > 0
                else "no_progress_goal_binding_candidates"
            )
        ),
        "next_planner_contract": {
            "required_before_execution": "candidate target site must be objective-bound and still pass one-step objective-progress controls",
            "allowed_connection": "diagnostic_only_until_objective_binding_and_controls",
            "action_generation_allowed": False,
        },
    }


def _score_relation_progress_goal_binding_action_effect_direction_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
    near_target_distance: float = 2.0,
) -> dict[str, Any]:
    event_count = 0
    candidate_count = 0
    objective_bound_candidate_count = 0
    objective_unbound_candidate_count = 0
    direction_evidence_candidate_count = 0
    direction_confirmed_candidate_count = 0
    direction_falsified_candidate_count = 0
    direction_ambiguous_candidate_count = 0
    no_direction_evidence_candidate_count = 0
    binding_state_counts: Counter[str] = Counter()
    direction_state_counts: Counter[str] = Counter()
    kinematic_status_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    target_stats: dict[str, dict[str, Any]] = {}

    def best_objective_binding(
        *,
        site: tuple[int, int] | None,
        objective_records: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        if site is None:
            return "missing_candidate_site", {}
        if not objective_records:
            return "no_objective_target_context", {}
        best_record: dict[str, Any] | None = None
        best_metrics: dict[str, Any] | None = None
        for record in objective_records:
            bbox = _objective_bbox_from_value(record.get("bbox"))
            if bbox is None:
                continue
            metrics = _site_to_bbox_metrics(site, bbox)
            if best_metrics is None or float(metrics["edge_distance"]) < float(
                best_metrics["edge_distance"]
            ):
                best_record = record
                best_metrics = metrics
        if best_record is None or best_metrics is None:
            return "no_objective_target_context", {}
        best = {**dict(best_record), **dict(best_metrics)}
        if bool(best_metrics.get("inside")):
            return "candidate_site_inside_objective_target", best
        if float(best_metrics.get("edge_distance", 0.0)) <= float(near_target_distance):
            return "candidate_site_near_objective_target", best
        return "candidate_site_far_from_objective_target", best

    def kinematic_target_matches_objective(
        row: Mapping[str, Any],
        objective: Mapping[str, Any],
    ) -> bool:
        target_key = str(row.get("target_key", "") or row.get("target_focus_key", "") or "")
        if not target_key:
            return True
        objective_key = str(objective.get("target_key", "") or "")
        if target_key and objective_key and target_key == objective_key:
            return True
        target_bbox = _objective_bbox_from_focus_key(target_key)
        objective_bbox = _objective_bbox_from_value(objective.get("bbox"))
        if target_bbox is not None and objective_bbox is not None:
            return target_bbox == objective_bbox
        return False

    def best_kinematic_match(
        *,
        site: tuple[int, int] | None,
        objective: Mapping[str, Any],
        kinematic_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if site is None:
            return {}
        matches: list[dict[str, Any]] = []
        for row in kinematic_rows:
            row_site = _xy_from_value(row.get("selected_site"))
            if row_site != site:
                continue
            row_copy = dict(row)
            row_copy["target_matches_objective"] = bool(
                kinematic_target_matches_objective(row_copy, objective)
            )
            matches.append(row_copy)
        if not matches:
            return {}

        def rank(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
            status = str(row.get("status", "") or "")
            target_match_rank = 0 if bool(row.get("target_matches_objective")) else 1
            positive = int(
                _as_int(row.get("score_delta"), 0) > 0
                or _as_int(row.get("overlap_delta"), 0) > 0
                or _as_int(row.get("distance_delta"), 0) > 0
            )
            status_rank = {
                "confirmed": 0,
                "ambiguous": 1,
                "falsified": 2,
                "insufficient_evidence": 3,
            }.get(status, 4)
            return (
                target_match_rank,
                -positive,
                status_rank,
                -_as_int(row.get("overlap_delta"), 0),
                -_as_int(row.get("distance_delta"), 0),
            )

        return dict(sorted(matches, key=rank)[0])

    def direction_state_for(
        *,
        binding_state: str,
        kinematic: Mapping[str, Any],
    ) -> str:
        if not kinematic:
            return "no_action_effect_direction_evidence"
        if binding_state not in {
            "candidate_site_inside_objective_target",
            "candidate_site_near_objective_target",
        }:
            return "candidate_unbound_even_with_motion_evidence"
        if not bool(kinematic.get("target_matches_objective", True)):
            return "motion_target_mismatch"
        status = str(kinematic.get("status", "") or "insufficient_evidence")
        overlap_delta = _as_int(kinematic.get("overlap_delta"), 0)
        distance_delta = _as_int(kinematic.get("distance_delta"), 0)
        score_delta = _as_int(kinematic.get("score_delta"), 0)
        canvas_delta = [
            _as_int(value, 0)
            for value in list(kinematic.get("canvas_delta") or [])
        ][:2]
        positive_direction = bool(
            score_delta > 0 or overlap_delta > 0 or distance_delta > 0
        )
        canvas_moved = bool(canvas_delta and any(value != 0 for value in canvas_delta))
        if status == "confirmed" and positive_direction:
            return "effect_toward_objective_confirmed"
        if status == "falsified" or not canvas_moved:
            return "effect_falsified_no_canvas_motion"
        if positive_direction:
            return "effect_toward_objective_ambiguous"
        return "effect_visible_not_toward_objective"

    def observe(
        *,
        game_id: str,
        item: Mapping[str, Any],
        target_key: str,
        candidate_relation: str,
        probe_kind: str,
        dominant_operator: str,
        dominant_action: str,
        gap_count: int,
        verifier_schema: str,
        verifier_schema_role: str,
        step: int,
        count: int,
    ) -> int:
        nonlocal candidate_count
        nonlocal objective_bound_candidate_count
        nonlocal objective_unbound_candidate_count
        nonlocal direction_evidence_candidate_count
        nonlocal direction_confirmed_candidate_count
        nonlocal direction_falsified_candidate_count
        nonlocal direction_ambiguous_candidate_count
        nonlocal no_direction_evidence_candidate_count
        if (
            candidate_relation != "progress_to_goal_binding_relation"
            or probe_kind != "goal_binding_relation_probe"
        ):
            return 0
        count = max(0, int(count))
        if count <= 0:
            return 0
        candidate_count += count
        gap = max(0, int(gap_count))
        site = _site_effect_target_site_from_key(target_key)
        objective_records = _objective_bbox_records_from_trace_item(item)
        binding_state, best_objective = best_objective_binding(
            site=site,
            objective_records=objective_records,
        )
        bound = binding_state in {
            "candidate_site_inside_objective_target",
            "candidate_site_near_objective_target",
        }
        if bound:
            objective_bound_candidate_count += count
        else:
            objective_unbound_candidate_count += count
        kinematic_rows = _kinematic_canvas_coupling_rows_from_trace_item(item)
        kinematic = best_kinematic_match(
            site=site,
            objective=best_objective,
            kinematic_rows=kinematic_rows,
        )
        direction_state = direction_state_for(
            binding_state=binding_state,
            kinematic=kinematic,
        )
        if kinematic:
            direction_evidence_candidate_count += count
            kinematic_status_counts[str(kinematic.get("status", "") or "")] += count
        else:
            no_direction_evidence_candidate_count += count
        if direction_state == "effect_toward_objective_confirmed":
            direction_confirmed_candidate_count += count
        elif direction_state in {
            "effect_falsified_no_canvas_motion",
            "effect_visible_not_toward_objective",
            "motion_target_mismatch",
            "candidate_unbound_even_with_motion_evidence",
        }:
            direction_falsified_candidate_count += count
        elif direction_state == "effect_toward_objective_ambiguous":
            direction_ambiguous_candidate_count += count

        binding_state_counts[binding_state] += count
        direction_state_counts[direction_state] += count
        by_game[game_id]["progress_goal_binding_action_effect_direction_candidate_count"] += count
        by_game[game_id][f"binding_state:{binding_state}"] += count
        by_game[game_id][f"direction_state:{direction_state}"] += count
        key = "\t".join([target_key, dominant_action, str(gap), direction_state])
        stats = target_stats.setdefault(
            key,
            {
                "target_key": target_key,
                "candidate_relation": candidate_relation,
                "probe_kind": probe_kind,
                "dominant_operator": dominant_operator,
                "dominant_action": dominant_action,
                "gap_count": int(gap),
                "verifier_schema": verifier_schema,
                "verifier_schema_role": verifier_schema_role,
                "binding_state": binding_state,
                "direction_state": direction_state,
                "candidate_site": list(site) if site is not None else [],
                "nearest_objective_target": best_objective,
                "matched_kinematic_evidence": kinematic,
                "observation_count": 0,
                "first_seen_step": int(step),
                "last_seen_step": int(step),
                "game_ids": set(),
                "state_hashes": set(),
            },
        )
        stats["observation_count"] = int(stats.get("observation_count", 0)) + count
        stats["first_seen_step"] = min(int(stats.get("first_seen_step", step)), int(step))
        stats["last_seen_step"] = max(int(stats.get("last_seen_step", step)), int(step))
        if isinstance(stats.get("game_ids"), set):
            stats["game_ids"].add(game_id)
        if isinstance(stats.get("state_hashes"), set):
            for state_hash in _state_hashes_from_value(item):
                stats["state_hashes"].add(str(state_hash))
        return count

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            preflight = unsolved.get("score_relation_candidate_verifier_preflight")
            if not isinstance(preflight, Mapping):
                continue
            shadow = preflight.get("score_relation_shadow_evidence")
            if not isinstance(shadow, Mapping) or not bool(shadow.get("enabled")):
                continue
            step = _as_int(item.get("step"), 0)
            observed_count = 0
            target_feature_counts = shadow.get(
                "candidate_selector_target_feature_counts"
            )
            observed_from_counts = False
            if isinstance(target_feature_counts, Mapping):
                for key, count_value in target_feature_counts.items():
                    parts = str(key).split("\t")
                    if len(parts) != 8:
                        continue
                    (
                        target_key,
                        candidate_relation,
                        probe_kind,
                        dominant_operator,
                        dominant_action,
                        gap_text,
                        verifier_schema,
                        verifier_schema_role,
                    ) = parts
                    observed_count += observe(
                        game_id=game_id,
                        item=item,
                        target_key=target_key,
                        candidate_relation=candidate_relation,
                        probe_kind=probe_kind,
                        dominant_operator=dominant_operator,
                        dominant_action=dominant_action,
                        gap_count=_as_int(gap_text, 0),
                        verifier_schema=verifier_schema,
                        verifier_schema_role=verifier_schema_role,
                        step=step,
                        count=_as_int(count_value, 0),
                    )
                    observed_from_counts = True
            if not observed_from_counts:
                for sample in list(shadow.get("samples") or []):
                    if not isinstance(sample, Mapping):
                        continue
                    observed_count += observe(
                        game_id=game_id,
                        item=item,
                        target_key=str(sample.get("target_key", "") or ""),
                        candidate_relation=str(
                            sample.get("candidate_relation", "") or ""
                        ),
                        probe_kind=str(sample.get("probe_kind", "") or ""),
                        dominant_operator=str(
                            sample.get("dominant_operator", "") or ""
                        ),
                        dominant_action=str(sample.get("dominant_action", "") or ""),
                        gap_count=_as_int(sample.get("gap_count"), 0),
                        verifier_schema=str(sample.get("verifier_schema", "") or ""),
                        verifier_schema_role=str(
                            sample.get("verifier_schema_role", "") or ""
                        ),
                        step=step,
                        count=1,
                    )
            if observed_count > 0:
                event_count += 1
                by_game[game_id][
                    "progress_goal_binding_action_effect_direction_event_count"
                ] += 1

    target_rows: list[dict[str, Any]] = []
    for stats in target_stats.values():
        row = {
            key: value
            for key, value in stats.items()
            if key not in {"game_ids", "state_hashes"}
        }
        row["game_ids"] = (
            sorted(stats.get("game_ids", set()))
            if isinstance(stats.get("game_ids"), set)
            else []
        )
        all_state_hashes = (
            sorted(stats.get("state_hashes", set()))
            if isinstance(stats.get("state_hashes"), set)
            else []
        )
        row["state_hashes"] = all_state_hashes[:16]
        row["state_hash_count"] = len(all_state_hashes)
        target_rows.append(row)
    target_rows.sort(
        key=lambda row: (
            0
            if row.get("direction_state") == "effect_toward_objective_confirmed"
            else 1
            if row.get("direction_state") == "effect_toward_objective_ambiguous"
            else 2
            if str(row.get("binding_state", "")).startswith("candidate_site_inside")
            else 3
            if str(row.get("binding_state", "")).startswith("candidate_site_near")
            else 4,
            -_as_int(row.get("observation_count"), 0),
            str(row.get("target_key", "")),
        )
    )

    return {
        "schema_version": (
            "eca_score_relation_progress_goal_binding_action_effect_direction_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "near_target_distance": float(near_target_distance),
        "shadow_event_count": int(event_count),
        "candidate_count": int(candidate_count),
        "target_candidate_count": int(len(target_rows)),
        "objective_bound_candidate_count": int(objective_bound_candidate_count),
        "objective_unbound_candidate_count": int(objective_unbound_candidate_count),
        "direction_evidence_candidate_count": int(direction_evidence_candidate_count),
        "direction_confirmed_candidate_count": int(direction_confirmed_candidate_count),
        "direction_falsified_candidate_count": int(direction_falsified_candidate_count),
        "direction_ambiguous_candidate_count": int(direction_ambiguous_candidate_count),
        "no_direction_evidence_candidate_count": int(no_direction_evidence_candidate_count),
        "binding_state_counts": dict(sorted(binding_state_counts.items())),
        "direction_state_counts": dict(sorted(direction_state_counts.items())),
        "kinematic_status_counts": dict(sorted(kinematic_status_counts.items())),
        "target_rows": target_rows[:sample_limit],
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "promotion_state": (
            "direction_confirmed_candidates_present_diagnostic_only"
            if direction_confirmed_candidate_count > 0
            else (
                "objective_bound_without_direction_confirmation"
                if objective_bound_candidate_count > 0
                else (
                    "progress_goal_candidates_unbound"
                    if candidate_count > 0
                    else "no_progress_goal_binding_candidates"
                )
            )
        ),
        "next_planner_contract": {
            "required_before_execution": "objective-bound candidate must have same-site kinematic evidence with overlap or distance improvement, then pass target-specific controls",
            "allowed_connection": "diagnostic_only_until_direction_controls",
            "action_generation_allowed": False,
        },
    }


def _score_relation_progress_goal_binding_admission_guard_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    row_list = list(rows)
    direction = _score_relation_progress_goal_binding_action_effect_direction_sensor(
        row_list,
        sample_limit=256,
    )
    memory = _score_relation_falsified_target_action_signature_memory_sensor(
        row_list,
        sample_limit=256,
    )
    falsified_signatures = {
        str(row.get("signature", "") or "")
        for row in list(memory.get("signature_rows") or [])
        if str(row.get("signature", "") or "")
    }

    candidate_count = 0
    direction_confirmed_candidate_count = 0
    blocked_falsified_signature_count = 0
    blocked_no_direction_confirmation_count = 0
    provisionally_admissible_count = 0
    admission_state_counts: Counter[str] = Counter()
    by_signature: dict[str, Counter[str]] = defaultdict(Counter)
    rows_out: list[dict[str, Any]] = []

    for row in list(direction.get("target_rows") or []):
        if not isinstance(row, Mapping):
            continue
        count = max(1, _as_int(row.get("observation_count"), 1))
        target_key = str(row.get("target_key", "") or "")
        dominant_action = _normalize_action_name(row.get("dominant_action", ""))
        candidate_relation = str(row.get("candidate_relation", "") or "")
        probe_kind = str(row.get("probe_kind", "") or "")
        signature = _score_relation_target_action_signature(
            target_key=target_key,
            action=dominant_action,
            candidate_relation=candidate_relation,
            probe_kind=probe_kind,
        )
        direction_state = str(row.get("direction_state", "") or "")
        candidate_count += count
        if direction_state == "effect_toward_objective_confirmed":
            direction_confirmed_candidate_count += count
            if signature in falsified_signatures:
                admission_state = "blocked_falsified_signature"
                blocked_falsified_signature_count += count
            else:
                admission_state = "provisionally_admissible_pending_controls"
                provisionally_admissible_count += count
        else:
            admission_state = "blocked_no_direction_confirmation"
            blocked_no_direction_confirmation_count += count
        admission_state_counts[admission_state] += count
        by_signature[signature]["candidate_count"] += count
        by_signature[signature][f"admission_state:{admission_state}"] += count
        if len(rows_out) < sample_limit:
            rows_out.append(
                {
                    "target_key": target_key,
                    "signature": signature,
                    "dominant_action": dominant_action,
                    "candidate_relation": candidate_relation,
                    "probe_kind": probe_kind,
                    "direction_state": direction_state,
                    "binding_state": str(row.get("binding_state", "") or ""),
                    "observation_count": count,
                    "admission_state": admission_state,
                    "falsified_signature_match": bool(signature in falsified_signatures),
                    "candidate_site": list(row.get("candidate_site") or []),
                    "game_ids": list(row.get("game_ids") or []),
                }
            )

    return {
        "schema_version": (
            "eca_score_relation_progress_goal_binding_admission_guard_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "source_direction_sensor": str(direction.get("schema_version", "") or ""),
        "source_falsified_memory_sensor": str(memory.get("schema_version", "") or ""),
        "candidate_count": int(candidate_count),
        "direction_confirmed_candidate_count": int(direction_confirmed_candidate_count),
        "blocked_falsified_signature_count": int(blocked_falsified_signature_count),
        "blocked_no_direction_confirmation_count": int(
            blocked_no_direction_confirmation_count
        ),
        "provisionally_admissible_count": int(provisionally_admissible_count),
        "known_falsified_signature_count": int(len(falsified_signatures)),
        "admission_state_counts": dict(sorted(admission_state_counts.items())),
        "by_signature": {
            signature: dict(sorted(stats.items()))
            for signature, stats in sorted(by_signature.items())
        },
        "rows": rows_out,
        "promotion_state": (
            "admission_blocked_falsified_signatures"
            if blocked_falsified_signature_count > 0
            else (
                "provisionally_admissible_pending_controls"
                if provisionally_admissible_count > 0
                else (
                    "blocked_no_direction_confirmation"
                    if candidate_count > 0
                    else "no_direction_candidates"
                )
            )
        ),
        "next_planner_contract": {
            "required_before_execution": "candidate must be direction-confirmed, avoid falsified target/action signatures, and beat paired controls",
            "allowed_connection": "diagnostic_only_until_control_margin",
            "action_generation_allowed": False,
        },
    }


def _score_relation_progress_goal_binding_control_margin_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    row_list = list(rows)
    direction = _score_relation_progress_goal_binding_action_effect_direction_sensor(
        row_list,
        sample_limit=256,
    )
    memory = _score_relation_falsified_target_action_signature_memory_sensor(
        row_list,
        sample_limit=256,
    )
    falsified_signatures = {
        str(row.get("signature", "") or "")
        for row in list(memory.get("signature_rows") or [])
        if str(row.get("signature", "") or "")
    }
    grouped: dict[str, dict[str, Any]] = {}
    for row in list(direction.get("target_rows") or []):
        if not isinstance(row, Mapping):
            continue
        target_key = str(row.get("target_key", "") or "")
        dominant_action = _normalize_action_name(row.get("dominant_action", ""))
        candidate_relation = str(row.get("candidate_relation", "") or "")
        probe_kind = str(row.get("probe_kind", "") or "")
        signature = _score_relation_target_action_signature(
            target_key=target_key,
            action=dominant_action,
            candidate_relation=candidate_relation,
            probe_kind=probe_kind,
        )
        count = max(1, _as_int(row.get("observation_count"), 1))
        direction_state = str(row.get("direction_state", "") or "")
        bucket = grouped.setdefault(
            signature,
            {
                "signature": signature,
                "target_key": target_key,
                "action": dominant_action,
                "candidate_relation": candidate_relation,
                "probe_kind": probe_kind,
                "candidate_site": list(row.get("candidate_site") or []),
                "direction_state_counts": Counter(),
                "game_ids": set(),
            },
        )
        states = bucket.get("direction_state_counts")
        if isinstance(states, Counter):
            states[direction_state] += count
        if isinstance(bucket.get("game_ids"), set):
            for game_id in list(row.get("game_ids") or []):
                bucket["game_ids"].add(str(game_id))

    rows_out: list[dict[str, Any]] = []
    positive_signature_count = 0
    blocked_falsified_signature_count = 0
    blocked_control_margin_signature_count = 0
    positive_margin_signature_count = 0
    no_control_signature_count = 0
    positive_observation_count = 0
    blocked_falsified_observation_count = 0
    blocked_control_margin_observation_count = 0
    positive_margin_observation_count = 0
    no_control_observation_count = 0
    admission_state_counts: Counter[str] = Counter()

    for bucket in grouped.values():
        states = bucket.get("direction_state_counts")
        states = states if isinstance(states, Counter) else Counter()
        positive = int(states.get("effect_toward_objective_confirmed", 0))
        if positive <= 0:
            continue
        same_signature_control = int(sum(states.values()) - positive)
        margin = int(positive - same_signature_control)
        signature = str(bucket.get("signature", "") or "")
        positive_signature_count += 1
        positive_observation_count += positive
        if signature in falsified_signatures:
            admission_state = "blocked_falsified_signature"
            blocked_falsified_signature_count += 1
            blocked_falsified_observation_count += positive
        elif same_signature_control >= positive:
            admission_state = "blocked_same_signature_control_margin"
            blocked_control_margin_signature_count += 1
            blocked_control_margin_observation_count += positive
        elif same_signature_control > 0:
            admission_state = "positive_margin_pending_counterfactual_controls"
            positive_margin_signature_count += 1
            positive_margin_observation_count += positive
        else:
            admission_state = "no_same_signature_controls_pending_counterfactual"
            no_control_signature_count += 1
            no_control_observation_count += positive
        admission_state_counts[admission_state] += positive
        rows_out.append(
            {
                "signature": signature,
                "target_key": str(bucket.get("target_key", "") or ""),
                "action": str(bucket.get("action", "") or ""),
                "candidate_relation": str(bucket.get("candidate_relation", "") or ""),
                "probe_kind": str(bucket.get("probe_kind", "") or ""),
                "candidate_site": list(bucket.get("candidate_site") or []),
                "positive_count": int(positive),
                "same_signature_control_count": int(same_signature_control),
                "control_margin": int(margin),
                "direction_state_counts": dict(sorted(states.items())),
                "admission_state": admission_state,
                "falsified_signature_match": bool(signature in falsified_signatures),
                "game_ids": (
                    sorted(bucket.get("game_ids", set()))
                    if isinstance(bucket.get("game_ids"), set)
                    else []
                ),
            }
        )

    rows_out.sort(
        key=lambda row: (
            0
            if row.get("admission_state") == "blocked_falsified_signature"
            else 1
            if row.get("admission_state")
            == "blocked_same_signature_control_margin"
            else 2
            if row.get("admission_state")
            == "positive_margin_pending_counterfactual_controls"
            else 3,
            _as_int(row.get("control_margin"), 0),
            -_as_int(row.get("positive_count"), 0),
            str(row.get("target_key", "")),
        )
    )

    return {
        "schema_version": (
            "eca_score_relation_progress_goal_binding_control_margin_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "source_direction_sensor": str(direction.get("schema_version", "") or ""),
        "source_falsified_memory_sensor": str(memory.get("schema_version", "") or ""),
        "positive_signature_count": int(positive_signature_count),
        "positive_observation_count": int(positive_observation_count),
        "blocked_falsified_signature_count": int(blocked_falsified_signature_count),
        "blocked_falsified_observation_count": int(
            blocked_falsified_observation_count
        ),
        "blocked_control_margin_signature_count": int(
            blocked_control_margin_signature_count
        ),
        "blocked_control_margin_observation_count": int(
            blocked_control_margin_observation_count
        ),
        "positive_margin_signature_count": int(positive_margin_signature_count),
        "positive_margin_observation_count": int(positive_margin_observation_count),
        "no_control_signature_count": int(no_control_signature_count),
        "no_control_observation_count": int(no_control_observation_count),
        "admission_state_counts": dict(sorted(admission_state_counts.items())),
        "rows": rows_out[:sample_limit],
        "promotion_state": (
            "control_margin_blocks_some_direction_confirmed_candidates"
            if blocked_falsified_signature_count > 0
            or blocked_control_margin_signature_count > 0
            else (
                "positive_margin_candidates_pending_counterfactual_controls"
                if positive_margin_signature_count > 0 or no_control_signature_count > 0
                else "no_direction_confirmed_candidates"
            )
        ),
        "next_planner_contract": {
            "required_before_execution": "direction-confirmed candidates must beat same-signature controls and then paired nearby/no-evidence controls",
            "allowed_connection": "diagnostic_only_until_paired_control_margin",
            "action_generation_allowed": False,
        },
    }


def _score_relation_progress_goal_binding_paired_control_margin_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
    pair_site_radius: int = 6,
) -> dict[str, Any]:
    row_list = list(rows)
    direction = _score_relation_progress_goal_binding_action_effect_direction_sensor(
        row_list,
        sample_limit=4096,
    )
    same_signature_margin = _score_relation_progress_goal_binding_control_margin_sensor(
        row_list,
        sample_limit=4096,
    )
    same_margin_by_signature = {
        str(row.get("signature", "") or ""): row
        for row in list(same_signature_margin.get("rows") or [])
        if isinstance(row, Mapping) and str(row.get("signature", "") or "")
    }

    control_states = {
        "no_action_effect_direction_evidence",
        "motion_target_mismatch",
        "candidate_unbound_even_with_motion_evidence",
        "effect_falsified_no_canvas_motion",
        "effect_visible_not_toward_objective",
    }

    def signature_for(row: Mapping[str, Any]) -> str:
        return _score_relation_target_action_signature(
            target_key=str(row.get("target_key", "") or ""),
            action=_normalize_action_name(row.get("dominant_action", "")),
            candidate_relation=str(row.get("candidate_relation", "") or ""),
            probe_kind=str(row.get("probe_kind", "") or ""),
        )

    def objective_key_for(row: Mapping[str, Any]) -> str:
        objective = row.get("nearest_objective_target")
        if isinstance(objective, Mapping):
            return str(objective.get("target_key", "") or "")
        return ""

    def objective_color_for(row: Mapping[str, Any]) -> str:
        objective_key = objective_key_for(row)
        parts = objective_key.split(":")
        if len(parts) >= 3 and parts[0] == "target":
            return parts[1]
        return ""

    def candidate_site_for(row: Mapping[str, Any]) -> tuple[int, int] | None:
        return _xy_from_value(row.get("candidate_site"))

    def same_selector_family(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        return (
            _normalize_action_name(a.get("dominant_action", ""))
            == _normalize_action_name(b.get("dominant_action", ""))
            and str(a.get("candidate_relation", "") or "")
            == str(b.get("candidate_relation", "") or "")
            and str(a.get("probe_kind", "") or "") == str(b.get("probe_kind", "") or "")
            and _as_int(a.get("gap_count"), 0) == _as_int(b.get("gap_count"), 0)
        )

    def chebyshev_distance(
        a: tuple[int, int] | None,
        b: tuple[int, int] | None,
    ) -> int | None:
        if a is None or b is None:
            return None
        return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))

    target_rows = [
        row
        for row in list(direction.get("target_rows") or [])
        if isinstance(row, Mapping)
    ]
    positive_rows = [
        row
        for row in target_rows
        if str(row.get("direction_state", "") or "")
        == "effect_toward_objective_confirmed"
    ]
    control_rows = [
        row
        for row in target_rows
        if str(row.get("direction_state", "") or "") in control_states
    ]

    rows_out: list[dict[str, Any]] = []
    positive_signature_count = 0
    positive_observation_count = 0
    blocked_prior_guard_signature_count = 0
    blocked_prior_guard_observation_count = 0
    blocked_paired_control_margin_signature_count = 0
    blocked_paired_control_margin_observation_count = 0
    positive_margin_signature_count = 0
    positive_margin_observation_count = 0
    no_paired_control_signature_count = 0
    no_paired_control_observation_count = 0
    paired_control_observation_count = 0
    admission_state_counts: Counter[str] = Counter()

    for positive in positive_rows:
        signature = signature_for(positive)
        same_margin_row = same_margin_by_signature.get(signature, {})
        prior_state = str(same_margin_row.get("admission_state", "") or "")
        positive_count = _as_int(
            same_margin_row.get("positive_count", positive.get("observation_count")),
            _as_int(positive.get("observation_count"), 0),
        )
        positive_signature_count += 1
        positive_observation_count += positive_count

        positive_site = candidate_site_for(positive)
        positive_objective_key = objective_key_for(positive)
        positive_objective_color = objective_color_for(positive)

        paired_controls: list[dict[str, Any]] = []
        same_objective_control_count = 0
        nearby_control_count = 0
        same_color_control_count = 0
        paired_control_count = 0
        for control in control_rows:
            if signature_for(control) == signature:
                continue
            if not same_selector_family(positive, control):
                continue
            control_site = candidate_site_for(control)
            distance = chebyshev_distance(positive_site, control_site)
            if distance is None or int(distance) > int(pair_site_radius):
                continue
            control_objective_key = objective_key_for(control)
            control_objective_color = objective_color_for(control)
            same_objective = bool(
                positive_objective_key
                and control_objective_key
                and positive_objective_key == control_objective_key
            )
            same_objective_color = bool(
                positive_objective_color
                and control_objective_color
                and positive_objective_color == control_objective_color
            )
            if not (same_objective or same_objective_color):
                continue
            count = max(1, _as_int(control.get("observation_count"), 1))
            paired_control_count += count
            if same_objective:
                same_objective_control_count += count
            if same_objective_color:
                same_color_control_count += count
            if int(distance) <= int(pair_site_radius):
                nearby_control_count += count
            if len(paired_controls) < 8:
                paired_controls.append(
                    {
                        "target_key": str(control.get("target_key", "") or ""),
                        "candidate_site": list(control.get("candidate_site") or []),
                        "direction_state": str(
                            control.get("direction_state", "") or ""
                        ),
                        "binding_state": str(control.get("binding_state", "") or ""),
                        "nearest_objective_target_key": control_objective_key,
                        "same_objective": bool(same_objective),
                        "same_objective_color": bool(same_objective_color),
                        "site_distance": int(distance),
                        "observation_count": int(count),
                    }
                )

        paired_control_observation_count += int(paired_control_count)
        paired_margin = int(positive_count - paired_control_count)
        total_control_count = _as_int(
            same_margin_row.get("same_signature_control_count"), 0
        ) + int(paired_control_count)
        total_margin = int(positive_count - total_control_count)

        if prior_state in {
            "blocked_falsified_signature",
            "blocked_same_signature_control_margin",
        }:
            admission_state = prior_state
            blocked_prior_guard_signature_count += 1
            blocked_prior_guard_observation_count += positive_count
        elif paired_control_count >= positive_count:
            admission_state = "blocked_paired_control_margin"
            blocked_paired_control_margin_signature_count += 1
            blocked_paired_control_margin_observation_count += positive_count
        elif paired_control_count > 0:
            admission_state = "positive_margin_pending_temporal_instance_controls"
            positive_margin_signature_count += 1
            positive_margin_observation_count += positive_count
        else:
            admission_state = "no_paired_controls_pending_temporal_instance_controls"
            no_paired_control_signature_count += 1
            no_paired_control_observation_count += positive_count
        admission_state_counts[admission_state] += positive_count

        rows_out.append(
            {
                "signature": signature,
                "target_key": str(positive.get("target_key", "") or ""),
                "action": _normalize_action_name(
                    positive.get("dominant_action", "")
                ),
                "candidate_relation": str(
                    positive.get("candidate_relation", "") or ""
                ),
                "probe_kind": str(positive.get("probe_kind", "") or ""),
                "gap_count": _as_int(positive.get("gap_count"), 0),
                "candidate_site": list(positive.get("candidate_site") or []),
                "nearest_objective_target_key": positive_objective_key,
                "nearest_objective_target": (
                    dict(positive.get("nearest_objective_target") or {})
                    if isinstance(positive.get("nearest_objective_target"), Mapping)
                    else {}
                ),
                "positive_count": int(positive_count),
                "same_signature_control_count": _as_int(
                    same_margin_row.get("same_signature_control_count"), 0
                ),
                "paired_control_count": int(paired_control_count),
                "same_objective_control_count": int(same_objective_control_count),
                "same_color_control_count": int(same_color_control_count),
                "nearby_control_count": int(nearby_control_count),
                "paired_control_margin": int(paired_margin),
                "total_control_count": int(total_control_count),
                "total_control_margin": int(total_margin),
                "prior_control_margin_state": prior_state,
                "admission_state": admission_state,
                "paired_control_samples": paired_controls,
                "game_ids": list(positive.get("game_ids") or []),
            }
        )

    rows_out.sort(
        key=lambda row: (
            0
            if row.get("admission_state") in {
                "blocked_falsified_signature",
                "blocked_same_signature_control_margin",
            }
            else 1
            if row.get("admission_state") == "blocked_paired_control_margin"
            else 2
            if row.get("admission_state")
            == "positive_margin_pending_temporal_instance_controls"
            else 3,
            _as_int(row.get("paired_control_margin"), 0),
            -_as_int(row.get("positive_count"), 0),
            str(row.get("target_key", "")),
        )
    )

    return {
        "schema_version": (
            "eca_score_relation_progress_goal_binding_paired_control_margin_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "source_direction_sensor": str(direction.get("schema_version", "") or ""),
        "source_same_signature_control_margin_sensor": str(
            same_signature_margin.get("schema_version", "") or ""
        ),
        "pair_site_radius": int(pair_site_radius),
        "control_direction_states": sorted(control_states),
        "positive_signature_count": int(positive_signature_count),
        "positive_observation_count": int(positive_observation_count),
        "blocked_prior_guard_signature_count": int(
            blocked_prior_guard_signature_count
        ),
        "blocked_prior_guard_observation_count": int(
            blocked_prior_guard_observation_count
        ),
        "blocked_paired_control_margin_signature_count": int(
            blocked_paired_control_margin_signature_count
        ),
        "blocked_paired_control_margin_observation_count": int(
            blocked_paired_control_margin_observation_count
        ),
        "positive_margin_signature_count": int(positive_margin_signature_count),
        "positive_margin_observation_count": int(positive_margin_observation_count),
        "no_paired_control_signature_count": int(no_paired_control_signature_count),
        "no_paired_control_observation_count": int(
            no_paired_control_observation_count
        ),
        "paired_control_observation_count": int(paired_control_observation_count),
        "admission_state_counts": dict(sorted(admission_state_counts.items())),
        "rows": rows_out[:sample_limit],
        "promotion_state": (
            "paired_control_margin_blocks_some_direction_confirmed_candidates"
            if blocked_paired_control_margin_signature_count > 0
            else (
                "prior_guards_block_direction_confirmed_candidates"
                if blocked_prior_guard_signature_count > 0
                else (
                    "positive_margin_candidates_pending_temporal_instance_controls"
                    if positive_margin_signature_count > 0
                    or no_paired_control_signature_count > 0
                    else "no_direction_confirmed_candidates"
                )
            )
        ),
        "next_planner_contract": {
            "required_before_execution": "direction-confirmed candidates must beat falsified, same-signature, paired nearby/objective controls, then pass temporal instance and score-closure attribution",
            "allowed_connection": "diagnostic_only_until_temporal_instance_controls",
            "action_generation_allowed": False,
        },
    }


def _score_relation_progress_goal_binding_target_instance_consistency_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    row_list = list(rows)
    direction = _score_relation_progress_goal_binding_action_effect_direction_sensor(
        row_list,
        sample_limit=4096,
    )
    paired_margin = _score_relation_progress_goal_binding_paired_control_margin_sensor(
        row_list,
        sample_limit=4096,
    )
    paired_by_signature = {
        str(row.get("signature", "") or ""): row
        for row in list(paired_margin.get("rows") or [])
        if isinstance(row, Mapping) and str(row.get("signature", "") or "")
    }

    def signature_for(row: Mapping[str, Any]) -> str:
        return _score_relation_target_action_signature(
            target_key=str(row.get("target_key", "") or ""),
            action=_normalize_action_name(row.get("dominant_action", "")),
            candidate_relation=str(row.get("candidate_relation", "") or ""),
            probe_kind=str(row.get("probe_kind", "") or ""),
        )

    def objective_key_for(row: Mapping[str, Any]) -> str:
        objective = row.get("nearest_objective_target")
        if isinstance(objective, Mapping):
            return str(objective.get("target_key", "") or "")
        return ""

    def target_color_for_key(target_key: str) -> str:
        parts = str(target_key or "").split(":")
        if len(parts) >= 3 and parts[0] == "target":
            return parts[1]
        return ""

    def kinematic_target_key_for(row: Mapping[str, Any]) -> str:
        kinematic = row.get("matched_kinematic_evidence")
        if isinstance(kinematic, Mapping):
            return str(
                kinematic.get("target_key", "")
                or kinematic.get("target_focus_key", "")
                or ""
            )
        return ""

    def target_instance_evidence_for(row: Mapping[str, Any]) -> dict[str, Any]:
        objective_key = objective_key_for(row)
        kinematic_key = kinematic_target_key_for(row)
        objective_bbox = _objective_bbox_from_focus_key(objective_key)
        kinematic_bbox = _objective_bbox_from_focus_key(kinematic_key)
        state_hashes = [
            str(value)
            for value in list(row.get("state_hashes") or [])
            if str(value or "").strip()
        ]
        state_hash_count = _as_int(row.get("state_hash_count"), len(state_hashes))
        return {
            "objective_target_instance_id": objective_key,
            "kinematic_target_instance_id": kinematic_key,
            "objective_target_bbox": list(objective_bbox) if objective_bbox else [],
            "kinematic_target_bbox": list(kinematic_bbox) if kinematic_bbox else [],
            "objective_target_centroid": _bbox_centroid(objective_bbox),
            "kinematic_target_centroid": _bbox_centroid(kinematic_bbox),
            "state_hashes": state_hashes,
            "state_hash_count": int(state_hash_count),
        }

    def instance_state_for(row: Mapping[str, Any]) -> str:
        objective_key = objective_key_for(row)
        kinematic_key = kinematic_target_key_for(row)
        if not objective_key:
            return "missing_objective_target_instance"
        if not kinematic_key:
            return "no_kinematic_target_instance_evidence"
        if objective_key == kinematic_key:
            return "target_instance_confirmed"
        objective_bbox = _objective_bbox_from_focus_key(objective_key)
        kinematic_bbox = _objective_bbox_from_focus_key(kinematic_key)
        if (
            objective_bbox is not None
            and kinematic_bbox is not None
            and objective_bbox == kinematic_bbox
        ):
            return "target_instance_confirmed"
        objective_color = target_color_for_key(objective_key)
        kinematic_color = target_color_for_key(kinematic_key)
        if objective_color and kinematic_color and objective_color == kinematic_color:
            return "target_instance_same_color_mismatch"
        return "target_instance_cross_color_mismatch"

    target_rows = [
        row
        for row in list(direction.get("target_rows") or [])
        if isinstance(row, Mapping)
    ]
    objective_instances_by_color: dict[str, set[str]] = defaultdict(set)
    for row in target_rows:
        objective_key = objective_key_for(row)
        color = target_color_for_key(objective_key)
        if color and objective_key:
            objective_instances_by_color[color].add(objective_key)

    target_instance_candidate_count = 0
    confirmed_instance_candidate_count = 0
    mismatched_instance_candidate_count = 0
    no_kinematic_instance_evidence_count = 0
    multi_instance_color_candidate_count = 0
    state_hash_bound_candidate_count = 0
    target_instance_state_counts: Counter[str] = Counter()
    direction_state_counts: Counter[str] = Counter()
    mismatch_rows: list[dict[str, Any]] = []

    for row in target_rows:
        count = max(1, _as_int(row.get("observation_count"), 1))
        state = instance_state_for(row)
        target_instance_candidate_count += count
        target_instance_state_counts[state] += count
        direction_state_counts[str(row.get("direction_state", "") or "")] += count
        objective_key = objective_key_for(row)
        color = target_color_for_key(objective_key)
        if color and len(objective_instances_by_color.get(color, set())) > 1:
            multi_instance_color_candidate_count += count
        evidence = target_instance_evidence_for(row)
        if _as_int(evidence.get("state_hash_count"), 0) > 0:
            state_hash_bound_candidate_count += count
        if state == "target_instance_confirmed":
            confirmed_instance_candidate_count += count
        elif state == "no_kinematic_target_instance_evidence":
            no_kinematic_instance_evidence_count += count
        else:
            mismatched_instance_candidate_count += count
            if len(mismatch_rows) < sample_limit:
                mismatch_rows.append(
                    {
                        "target_key": str(row.get("target_key", "") or ""),
                        "candidate_site": list(row.get("candidate_site") or []),
                        "direction_state": str(row.get("direction_state", "") or ""),
                        "objective_target_key": objective_key,
                        "kinematic_target_key": kinematic_target_key_for(row),
                        **target_instance_evidence_for(row),
                        "target_instance_state": state,
                        "observation_count": int(count),
                        "game_ids": list(row.get("game_ids") or []),
                    }
                )

    admission_rows: list[dict[str, Any]] = []
    admission_state_counts: Counter[str] = Counter()
    positive_signature_count = 0
    positive_observation_count = 0
    blocked_prior_guard_signature_count = 0
    blocked_prior_guard_observation_count = 0
    blocked_target_instance_mismatch_signature_count = 0
    blocked_target_instance_mismatch_observation_count = 0
    confirmed_pending_temporal_signature_count = 0
    confirmed_pending_temporal_observation_count = 0

    for row in target_rows:
        if (
            str(row.get("direction_state", "") or "")
            != "effect_toward_objective_confirmed"
        ):
            continue
        signature = signature_for(row)
        paired_row = paired_by_signature.get(signature, {})
        prior_state = str(paired_row.get("admission_state", "") or "")
        count = _as_int(
            paired_row.get("positive_count", row.get("observation_count")),
            _as_int(row.get("observation_count"), 0),
        )
        positive_signature_count += 1
        positive_observation_count += count
        state = instance_state_for(row)
        if prior_state in {
            "blocked_falsified_signature",
            "blocked_same_signature_control_margin",
            "blocked_paired_control_margin",
        }:
            admission_state = prior_state
            blocked_prior_guard_signature_count += 1
            blocked_prior_guard_observation_count += count
        elif state != "target_instance_confirmed":
            admission_state = "blocked_target_instance_mismatch"
            blocked_target_instance_mismatch_signature_count += 1
            blocked_target_instance_mismatch_observation_count += count
        else:
            admission_state = "target_instance_confirmed_pending_temporal_precondition"
            confirmed_pending_temporal_signature_count += 1
            confirmed_pending_temporal_observation_count += count
        admission_state_counts[admission_state] += count
        if len(admission_rows) < sample_limit:
            objective_key = objective_key_for(row)
            color = target_color_for_key(objective_key)
            admission_rows.append(
                {
                    "signature": signature,
                    "target_key": str(row.get("target_key", "") or ""),
                    "candidate_site": list(row.get("candidate_site") or []),
                    "objective_target_key": objective_key,
                    "kinematic_target_key": kinematic_target_key_for(row),
                    **target_instance_evidence_for(row),
                    "target_instance_state": state,
                    "same_color_objective_instance_count": int(
                        len(objective_instances_by_color.get(color, set()))
                    ),
                    "positive_count": int(count),
                    "prior_admission_state": prior_state,
                    "admission_state": admission_state,
                    "game_ids": list(row.get("game_ids") or []),
                }
            )

    return {
        "schema_version": (
            "eca_score_relation_progress_goal_binding_target_instance_consistency_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "source_direction_sensor": str(direction.get("schema_version", "") or ""),
        "source_paired_control_margin_sensor": str(
            paired_margin.get("schema_version", "") or ""
        ),
        "target_instance_candidate_count": int(target_instance_candidate_count),
        "confirmed_instance_candidate_count": int(confirmed_instance_candidate_count),
        "mismatched_instance_candidate_count": int(
            mismatched_instance_candidate_count
        ),
        "no_kinematic_instance_evidence_count": int(
            no_kinematic_instance_evidence_count
        ),
        "multi_instance_color_candidate_count": int(
            multi_instance_color_candidate_count
        ),
        "state_hash_bound_candidate_count": int(state_hash_bound_candidate_count),
        "objective_instance_count_by_color": {
            color: int(len(instances))
            for color, instances in sorted(objective_instances_by_color.items())
        },
        "target_instance_state_counts": dict(
            sorted(target_instance_state_counts.items())
        ),
        "direction_state_counts": dict(sorted(direction_state_counts.items())),
        "positive_signature_count": int(positive_signature_count),
        "positive_observation_count": int(positive_observation_count),
        "blocked_prior_guard_signature_count": int(
            blocked_prior_guard_signature_count
        ),
        "blocked_prior_guard_observation_count": int(
            blocked_prior_guard_observation_count
        ),
        "blocked_target_instance_mismatch_signature_count": int(
            blocked_target_instance_mismatch_signature_count
        ),
        "blocked_target_instance_mismatch_observation_count": int(
            blocked_target_instance_mismatch_observation_count
        ),
        "confirmed_pending_temporal_signature_count": int(
            confirmed_pending_temporal_signature_count
        ),
        "confirmed_pending_temporal_observation_count": int(
            confirmed_pending_temporal_observation_count
        ),
        "admission_state_counts": dict(sorted(admission_state_counts.items())),
        "mismatch_rows": mismatch_rows,
        "admission_rows": admission_rows,
        "promotion_state": (
            "target_instance_mismatch_blocks_direction_confirmed_candidates"
            if blocked_target_instance_mismatch_signature_count > 0
            else (
                "target_instance_confirmed_candidates_pending_temporal_precondition"
                if confirmed_pending_temporal_signature_count > 0
                else (
                    "prior_guards_block_direction_confirmed_candidates"
                    if blocked_prior_guard_signature_count > 0
                    else "no_direction_confirmed_candidates"
                )
            )
        ),
        "next_planner_contract": {
            "required_before_execution": "candidate must pass falsified, same-signature, paired-control, target-instance, temporal precondition, and score-closure attribution gates",
            "allowed_connection": "diagnostic_only_until_temporal_precondition",
            "action_generation_allowed": False,
        },
    }


def _score_relation_progress_goal_binding_temporal_readiness_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    row_list = list(rows)
    target_instance = (
        _score_relation_progress_goal_binding_target_instance_consistency_sensor(
            row_list,
            sample_limit=4096,
        )
    )

    def signature_for_parts(
        *,
        target_key: str,
        action: str,
        candidate_relation: str,
        probe_kind: str,
    ) -> str:
        return _score_relation_target_action_signature(
            target_key=target_key,
            action=action,
            candidate_relation=candidate_relation,
            probe_kind=probe_kind,
        )

    temporal_stats: dict[str, dict[str, Any]] = {}

    def observe(
        *,
        game_id: str,
        item: Mapping[str, Any],
        target_key: str,
        candidate_relation: str,
        probe_kind: str,
        dominant_action: str,
        count: int,
    ) -> None:
        if (
            candidate_relation != "progress_to_goal_binding_relation"
            or probe_kind != "goal_binding_relation_probe"
        ):
            return
        count = max(0, int(count))
        if count <= 0:
            return
        signature = signature_for_parts(
            target_key=target_key,
            action=dominant_action,
            candidate_relation=candidate_relation,
            probe_kind=probe_kind,
        )
        score_before = _as_int(item.get("score_before"), -1)
        step = _as_int(item.get("step"), 0)
        stats = temporal_stats.setdefault(
            signature,
            {
                "signature": signature,
                "target_key": target_key,
                "dominant_action": _normalize_action_name(dominant_action),
                "candidate_relation": candidate_relation,
                "probe_kind": probe_kind,
                "observation_count": 0,
                "score_before_counts": Counter(),
                "first_seen_step": step,
                "last_seen_step": step,
                "game_ids": set(),
            },
        )
        stats["observation_count"] = _as_int(stats.get("observation_count"), 0) + count
        score_counts = stats.get("score_before_counts")
        if isinstance(score_counts, Counter):
            score_counts[str(score_before)] += count
        stats["first_seen_step"] = min(_as_int(stats.get("first_seen_step"), step), step)
        stats["last_seen_step"] = max(_as_int(stats.get("last_seen_step"), step), step)
        if isinstance(stats.get("game_ids"), set):
            stats["game_ids"].add(game_id)

    for row in row_list:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            preflight = unsolved.get("score_relation_candidate_verifier_preflight")
            if not isinstance(preflight, Mapping):
                continue
            shadow = preflight.get("score_relation_shadow_evidence")
            if not isinstance(shadow, Mapping) or not bool(shadow.get("enabled")):
                continue
            target_feature_counts = shadow.get(
                "candidate_selector_target_feature_counts"
            )
            observed_from_counts = False
            if isinstance(target_feature_counts, Mapping):
                for key, count_value in target_feature_counts.items():
                    parts = str(key).split("\t")
                    if len(parts) != 8:
                        continue
                    (
                        target_key,
                        candidate_relation,
                        probe_kind,
                        _dominant_operator,
                        dominant_action,
                        _gap_text,
                        _verifier_schema,
                        _verifier_schema_role,
                    ) = parts
                    observe(
                        game_id=game_id,
                        item=item,
                        target_key=target_key,
                        candidate_relation=candidate_relation,
                        probe_kind=probe_kind,
                        dominant_action=dominant_action,
                        count=_as_int(count_value, 0),
                    )
                    observed_from_counts = True
            if observed_from_counts:
                continue
            for sample in list(shadow.get("samples") or []):
                if not isinstance(sample, Mapping):
                    continue
                observe(
                    game_id=game_id,
                    item=item,
                    target_key=str(sample.get("target_key", "") or ""),
                    candidate_relation=str(sample.get("candidate_relation", "") or ""),
                    probe_kind=str(sample.get("probe_kind", "") or ""),
                    dominant_action=str(sample.get("dominant_action", "") or ""),
                    count=1,
                )

    admission_rows: list[dict[str, Any]] = []
    temporal_state_counts: Counter[str] = Counter()
    positive_signature_count = 0
    positive_observation_count = 0
    blocked_prior_guard_signature_count = 0
    blocked_prior_guard_observation_count = 0
    blocked_temporal_ambiguous_signature_count = 0
    blocked_temporal_ambiguous_observation_count = 0
    single_band_pending_score_closure_signature_count = 0
    single_band_pending_score_closure_observation_count = 0
    missing_temporal_evidence_signature_count = 0
    missing_temporal_evidence_observation_count = 0

    for row in list(target_instance.get("admission_rows") or []):
        if not isinstance(row, Mapping):
            continue
        signature = str(row.get("signature", "") or "")
        count = max(1, _as_int(row.get("positive_count"), 1))
        prior_state = str(row.get("admission_state", "") or "")
        stats = temporal_stats.get(signature, {})
        score_counts = stats.get("score_before_counts")
        score_counts = score_counts if isinstance(score_counts, Counter) else Counter()
        score_band_count = len(score_counts)
        positive_signature_count += 1
        positive_observation_count += count
        if prior_state in {
            "blocked_falsified_signature",
            "blocked_same_signature_control_margin",
            "blocked_paired_control_margin",
            "blocked_target_instance_mismatch",
        }:
            temporal_state = prior_state
            blocked_prior_guard_signature_count += 1
            blocked_prior_guard_observation_count += count
        elif not score_counts:
            temporal_state = "missing_temporal_readiness_evidence"
            missing_temporal_evidence_signature_count += 1
            missing_temporal_evidence_observation_count += count
        elif score_band_count > 1:
            temporal_state = "blocked_temporal_readiness_ambiguous_score_band"
            blocked_temporal_ambiguous_signature_count += 1
            blocked_temporal_ambiguous_observation_count += count
        else:
            temporal_state = "single_score_band_pending_score_closure_attribution"
            single_band_pending_score_closure_signature_count += 1
            single_band_pending_score_closure_observation_count += count
        temporal_state_counts[temporal_state] += count
        if len(admission_rows) < sample_limit:
            admission_rows.append(
                {
                    "signature": signature,
                    "target_key": str(row.get("target_key", "") or ""),
                    "candidate_site": list(row.get("candidate_site") or []),
                    "positive_count": int(count),
                    "prior_admission_state": prior_state,
                    "temporal_readiness_state": temporal_state,
                    "score_before_counts": dict(sorted(score_counts.items())),
                    "score_band_count": int(score_band_count),
                    "first_seen_step": _as_int(stats.get("first_seen_step"), 0),
                    "last_seen_step": _as_int(stats.get("last_seen_step"), 0),
                    "game_ids": (
                        sorted(stats.get("game_ids", set()))
                        if isinstance(stats.get("game_ids"), set)
                        else []
                    ),
                }
            )

    return {
        "schema_version": (
            "eca_score_relation_progress_goal_binding_temporal_readiness_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "source_target_instance_consistency_sensor": str(
            target_instance.get("schema_version", "") or ""
        ),
        "positive_signature_count": int(positive_signature_count),
        "positive_observation_count": int(positive_observation_count),
        "blocked_prior_guard_signature_count": int(
            blocked_prior_guard_signature_count
        ),
        "blocked_prior_guard_observation_count": int(
            blocked_prior_guard_observation_count
        ),
        "blocked_temporal_ambiguous_signature_count": int(
            blocked_temporal_ambiguous_signature_count
        ),
        "blocked_temporal_ambiguous_observation_count": int(
            blocked_temporal_ambiguous_observation_count
        ),
        "single_band_pending_score_closure_signature_count": int(
            single_band_pending_score_closure_signature_count
        ),
        "single_band_pending_score_closure_observation_count": int(
            single_band_pending_score_closure_observation_count
        ),
        "missing_temporal_evidence_signature_count": int(
            missing_temporal_evidence_signature_count
        ),
        "missing_temporal_evidence_observation_count": int(
            missing_temporal_evidence_observation_count
        ),
        "temporal_readiness_state_counts": dict(sorted(temporal_state_counts.items())),
        "admission_rows": admission_rows,
        "promotion_state": (
            "temporal_readiness_blocks_all_remaining_direction_confirmed_candidates"
            if blocked_temporal_ambiguous_signature_count > 0
            and single_band_pending_score_closure_signature_count == 0
            and missing_temporal_evidence_signature_count == 0
            else (
                "single_score_band_candidates_pending_score_closure_attribution"
                if single_band_pending_score_closure_signature_count > 0
                else (
                    "prior_guards_block_direction_confirmed_candidates"
                    if blocked_prior_guard_signature_count > 0
                    else "no_direction_confirmed_candidates"
                )
            )
        ),
        "next_planner_contract": {
            "required_before_execution": "candidate must be admitted for the current score/readiness band and then pass score-closure attribution",
            "allowed_connection": "diagnostic_only_until_score_closure_attribution",
            "action_generation_allowed": False,
        },
    }


def _score_relation_progress_goal_binding_score_closure_attribution_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
    closure_window_steps: int = 4,
) -> dict[str, Any]:
    row_list = list(rows)
    temporal = _score_relation_progress_goal_binding_temporal_readiness_sensor(
        row_list,
        sample_limit=4096,
    )

    def signature_for_parts(
        *,
        target_key: str,
        action: str,
        candidate_relation: str,
        probe_kind: str,
    ) -> str:
        return _score_relation_target_action_signature(
            target_key=target_key,
            action=action,
            candidate_relation=candidate_relation,
            probe_kind=probe_kind,
        )

    closure_stats: dict[str, dict[str, Any]] = {}

    def observe(
        *,
        game_id: str,
        item: Mapping[str, Any],
        score_events: list[tuple[int, int, int]],
        target_key: str,
        candidate_relation: str,
        probe_kind: str,
        dominant_action: str,
        count: int,
    ) -> None:
        if (
            candidate_relation != "progress_to_goal_binding_relation"
            or probe_kind != "goal_binding_relation_probe"
        ):
            return
        count = max(0, int(count))
        if count <= 0:
            return
        signature = signature_for_parts(
            target_key=target_key,
            action=dominant_action,
            candidate_relation=candidate_relation,
            probe_kind=probe_kind,
        )
        score_before = _as_int(item.get("score_before"), -1)
        step = _as_int(item.get("step"), 0)
        next_events = [event for event in score_events if int(event[0]) > step]
        next_event = next_events[0] if next_events else None
        steps_to_next_score = (
            int(next_event[0]) - step if next_event is not None else None
        )
        closure_attributed = bool(
            steps_to_next_score is not None
            and int(steps_to_next_score) <= int(closure_window_steps)
        )
        stats = closure_stats.setdefault(
            signature,
            {
                "signature": signature,
                "target_key": target_key,
                "dominant_action": _normalize_action_name(dominant_action),
                "candidate_relation": candidate_relation,
                "probe_kind": probe_kind,
                "observation_count": 0,
                "closure_attributed_count": 0,
                "immediate_closure_count": 0,
                "delayed_closure_count": 0,
                "stale_or_unclosed_count": 0,
                "score_before_counts": Counter(),
                "closure_score_before_counts": Counter(),
                "stale_score_before_counts": Counter(),
                "closure_step_distance_counts": Counter(),
                "first_seen_step": step,
                "last_seen_step": step,
                "nearest_closure_step_distance_min": None,
                "game_ids": set(),
            },
        )
        stats["observation_count"] = _as_int(stats.get("observation_count"), 0) + count
        score_counts = stats.get("score_before_counts")
        if isinstance(score_counts, Counter):
            score_counts[str(score_before)] += count
        if closure_attributed:
            stats["closure_attributed_count"] = _as_int(
                stats.get("closure_attributed_count"), 0
            ) + count
            closure_counts = stats.get("closure_score_before_counts")
            if isinstance(closure_counts, Counter):
                closure_counts[str(score_before)] += count
            distance_counts = stats.get("closure_step_distance_counts")
            if isinstance(distance_counts, Counter):
                distance_counts[str(int(steps_to_next_score))] += count
            if int(steps_to_next_score) <= 1:
                stats["immediate_closure_count"] = _as_int(
                    stats.get("immediate_closure_count"), 0
                ) + count
            else:
                stats["delayed_closure_count"] = _as_int(
                    stats.get("delayed_closure_count"), 0
                ) + count
            current_min = stats.get("nearest_closure_step_distance_min")
            if current_min is None:
                stats["nearest_closure_step_distance_min"] = int(steps_to_next_score)
            else:
                stats["nearest_closure_step_distance_min"] = min(
                    _as_int(current_min, int(steps_to_next_score)),
                    int(steps_to_next_score),
                )
        else:
            stats["stale_or_unclosed_count"] = _as_int(
                stats.get("stale_or_unclosed_count"), 0
            ) + count
            stale_counts = stats.get("stale_score_before_counts")
            if isinstance(stale_counts, Counter):
                stale_counts[str(score_before)] += count
        stats["first_seen_step"] = min(_as_int(stats.get("first_seen_step"), step), step)
        stats["last_seen_step"] = max(_as_int(stats.get("last_seen_step"), step), step)
        if isinstance(stats.get("game_ids"), set):
            stats["game_ids"].add(game_id)

    for row in row_list:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        score_events: list[tuple[int, int, int]] = []
        previous_score: int | None = None
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            score_before = item.get("score_before")
            if isinstance(score_before, int):
                if previous_score is not None and score_before > previous_score:
                    score_events.append(
                        (
                            _as_int(item.get("step"), 0),
                            int(previous_score),
                            int(score_before),
                        )
                    )
                previous_score = int(score_before)
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            preflight = unsolved.get("score_relation_candidate_verifier_preflight")
            if not isinstance(preflight, Mapping):
                continue
            shadow = preflight.get("score_relation_shadow_evidence")
            if not isinstance(shadow, Mapping) or not bool(shadow.get("enabled")):
                continue
            target_feature_counts = shadow.get(
                "candidate_selector_target_feature_counts"
            )
            observed_from_counts = False
            if isinstance(target_feature_counts, Mapping):
                for key, count_value in target_feature_counts.items():
                    parts = str(key).split("\t")
                    if len(parts) != 8:
                        continue
                    (
                        target_key,
                        candidate_relation,
                        probe_kind,
                        _dominant_operator,
                        dominant_action,
                        _gap_text,
                        _verifier_schema,
                        _verifier_schema_role,
                    ) = parts
                    observe(
                        game_id=game_id,
                        item=item,
                        score_events=score_events,
                        target_key=target_key,
                        candidate_relation=candidate_relation,
                        probe_kind=probe_kind,
                        dominant_action=dominant_action,
                        count=_as_int(count_value, 0),
                    )
                    observed_from_counts = True
            if observed_from_counts:
                continue
            for sample in list(shadow.get("samples") or []):
                if not isinstance(sample, Mapping):
                    continue
                observe(
                    game_id=game_id,
                    item=item,
                    score_events=score_events,
                    target_key=str(sample.get("target_key", "") or ""),
                    candidate_relation=str(sample.get("candidate_relation", "") or ""),
                    probe_kind=str(sample.get("probe_kind", "") or ""),
                    dominant_action=str(sample.get("dominant_action", "") or ""),
                    count=1,
                )

    admission_rows: list[dict[str, Any]] = []
    closure_state_counts: Counter[str] = Counter()
    positive_signature_count = 0
    positive_observation_count = 0
    blocked_prior_guard_signature_count = 0
    blocked_prior_guard_observation_count = 0
    blocked_score_closure_ambiguous_signature_count = 0
    blocked_score_closure_ambiguous_observation_count = 0
    score_closure_confirmed_pending_sandbox_signature_count = 0
    score_closure_confirmed_pending_sandbox_observation_count = 0
    no_score_closure_signature_count = 0
    no_score_closure_observation_count = 0

    for row in list(temporal.get("admission_rows") or []):
        if not isinstance(row, Mapping):
            continue
        signature = str(row.get("signature", "") or "")
        count = max(1, _as_int(row.get("positive_count"), 1))
        prior_state = str(row.get("temporal_readiness_state", "") or "")
        stats = closure_stats.get(signature, {})
        closure_counts = stats.get("closure_score_before_counts")
        closure_counts = closure_counts if isinstance(closure_counts, Counter) else Counter()
        stale_counts = stats.get("stale_score_before_counts")
        stale_counts = stale_counts if isinstance(stale_counts, Counter) else Counter()
        closure_attributed_count = _as_int(
            stats.get("closure_attributed_count"), 0
        )
        stale_or_unclosed_count = _as_int(stats.get("stale_or_unclosed_count"), 0)
        closure_band_count = len(closure_counts)
        positive_signature_count += 1
        positive_observation_count += count
        if prior_state in {
            "blocked_falsified_signature",
            "blocked_same_signature_control_margin",
            "blocked_paired_control_margin",
            "blocked_target_instance_mismatch",
        }:
            closure_state = prior_state
            blocked_prior_guard_signature_count += 1
            blocked_prior_guard_observation_count += count
        elif closure_attributed_count <= 0:
            closure_state = "blocked_no_score_closure_attribution"
            no_score_closure_signature_count += 1
            no_score_closure_observation_count += count
        elif closure_band_count != 1 or stale_or_unclosed_count > 0:
            closure_state = "blocked_score_closure_ambiguous_multi_band_or_stale"
            blocked_score_closure_ambiguous_signature_count += 1
            blocked_score_closure_ambiguous_observation_count += count
        else:
            closure_state = "score_closure_attributed_single_band_pending_sandbox"
            score_closure_confirmed_pending_sandbox_signature_count += 1
            score_closure_confirmed_pending_sandbox_observation_count += count
        closure_state_counts[closure_state] += count
        if len(admission_rows) < sample_limit:
            admission_rows.append(
                {
                    "signature": signature,
                    "target_key": str(row.get("target_key", "") or ""),
                    "candidate_site": list(row.get("candidate_site") or []),
                    "positive_count": int(count),
                    "prior_temporal_readiness_state": prior_state,
                    "score_closure_attribution_state": closure_state,
                    "closure_window_steps": int(closure_window_steps),
                    "score_before_counts": dict(
                        sorted(
                            (
                                stats.get("score_before_counts")
                                if isinstance(stats.get("score_before_counts"), Counter)
                                else Counter()
                            ).items()
                        )
                    ),
                    "closure_score_before_counts": dict(
                        sorted(closure_counts.items())
                    ),
                    "stale_score_before_counts": dict(sorted(stale_counts.items())),
                    "closure_attributed_count": int(closure_attributed_count),
                    "immediate_closure_count": _as_int(
                        stats.get("immediate_closure_count"), 0
                    ),
                    "delayed_closure_count": _as_int(
                        stats.get("delayed_closure_count"), 0
                    ),
                    "stale_or_unclosed_count": int(stale_or_unclosed_count),
                    "closure_step_distance_counts": dict(
                        sorted(
                            (
                                stats.get("closure_step_distance_counts")
                                if isinstance(
                                    stats.get("closure_step_distance_counts"), Counter
                                )
                                else Counter()
                            ).items()
                        )
                    ),
                    "closure_score_band_count": int(closure_band_count),
                    "nearest_closure_step_distance_min": stats.get(
                        "nearest_closure_step_distance_min"
                    ),
                    "first_seen_step": _as_int(stats.get("first_seen_step"), 0),
                    "last_seen_step": _as_int(stats.get("last_seen_step"), 0),
                    "game_ids": (
                        sorted(stats.get("game_ids", set()))
                        if isinstance(stats.get("game_ids"), set)
                        else []
                    ),
                }
            )

    return {
        "schema_version": (
            "eca_score_relation_progress_goal_binding_score_closure_attribution_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "source_temporal_readiness_sensor": str(
            temporal.get("schema_version", "") or ""
        ),
        "closure_window_steps": int(closure_window_steps),
        "positive_signature_count": int(positive_signature_count),
        "positive_observation_count": int(positive_observation_count),
        "blocked_prior_guard_signature_count": int(
            blocked_prior_guard_signature_count
        ),
        "blocked_prior_guard_observation_count": int(
            blocked_prior_guard_observation_count
        ),
        "blocked_score_closure_ambiguous_signature_count": int(
            blocked_score_closure_ambiguous_signature_count
        ),
        "blocked_score_closure_ambiguous_observation_count": int(
            blocked_score_closure_ambiguous_observation_count
        ),
        "score_closure_confirmed_pending_sandbox_signature_count": int(
            score_closure_confirmed_pending_sandbox_signature_count
        ),
        "score_closure_confirmed_pending_sandbox_observation_count": int(
            score_closure_confirmed_pending_sandbox_observation_count
        ),
        "no_score_closure_signature_count": int(no_score_closure_signature_count),
        "no_score_closure_observation_count": int(no_score_closure_observation_count),
        "score_closure_attribution_state_counts": dict(
            sorted(closure_state_counts.items())
        ),
        "admission_rows": admission_rows,
        "promotion_state": (
            "score_closure_attribution_blocks_all_remaining_direction_confirmed_candidates"
            if blocked_score_closure_ambiguous_signature_count > 0
            and score_closure_confirmed_pending_sandbox_signature_count == 0
            and no_score_closure_signature_count == 0
            else (
                "score_closure_attributed_candidates_pending_sandbox"
                if score_closure_confirmed_pending_sandbox_signature_count > 0
                else (
                    "prior_guards_block_direction_confirmed_candidates"
                    if blocked_prior_guard_signature_count > 0
                    else "no_direction_confirmed_candidates"
                )
            )
        ),
        "next_planner_contract": {
            "required_before_execution": "candidate must have single-band score-closure attribution before sandbox execution or ranking promotion",
            "allowed_connection": "diagnostic_only_until_single_band_score_closure_sandbox",
            "action_generation_allowed": False,
        },
    }


def _score_relation_remote_status_runtime_guard_outcome_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    preflight_reason_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    blocked_transition_counts: Counter[str] = Counter()
    blocked_target_counts: Counter[str] = Counter()
    skip_event_samples: list[dict[str, Any]] = []
    summary_samples: list[dict[str, Any]] = []
    score_relation_used_samples: list[dict[str, Any]] = []
    preflight_count = 0
    guard_enabled_preflight_count = 0
    runtime_skip_event_count = 0
    runtime_skipped_candidate_count_sum = 0
    all_candidates_blocked_event_count = 0
    partial_skip_event_count = 0
    target_verification_summary_count = 0
    target_verification_blocked_summary_count = 0
    score_relation_verifier_used_count = 0
    score_relation_verifier_used_with_guard_enabled_count = 0

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            step = _as_int(item.get("step"), 0)
            target_verification = item.get("target_verification")
            if isinstance(target_verification, Mapping):
                guard_summary = target_verification.get(
                    "score_relation_remote_status_negative_rank_guard"
                )
                if isinstance(guard_summary, Mapping):
                    target_verification_summary_count += 1
                    blocked_count = _as_int(
                        guard_summary.get("blocked_transition_count"),
                        0,
                    )
                    if blocked_count > 0:
                        target_verification_blocked_summary_count += 1
                        by_game[game_id]["target_verification_blocked_summary_count"] += 1
                        for key in list(guard_summary.get("blocked_transition_keys") or []):
                            blocked_transition_counts[str(key)] += 1
                        by_transition = guard_summary.get("by_transition_key")
                        if isinstance(by_transition, Mapping):
                            for transition_key, entry_raw in by_transition.items():
                                if not isinstance(entry_raw, Mapping):
                                    continue
                                if (
                                    str(entry_raw.get("guard_state", "") or "")
                                    != "suppress_repeated_remote_status_transition"
                                ):
                                    continue
                                target_counts = entry_raw.get("target_key_counts")
                                if isinstance(target_counts, Mapping):
                                    for target_key, count in target_counts.items():
                                        blocked_target_counts[str(target_key)] += _as_int(
                                            count,
                                            0,
                                        )
                        if len(summary_samples) < sample_limit:
                            summary_samples.append(
                                {
                                    "game_id": game_id,
                                    "variant": str(row.get("variant", "") or "unknown"),
                                    "step": step,
                                    "blocked_transition_count": blocked_count,
                                    "blocked_transition_keys": list(
                                        guard_summary.get("blocked_transition_keys") or []
                                    ),
                                }
                            )

            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            preflight = unsolved.get("score_relation_candidate_verifier_preflight")
            if isinstance(preflight, Mapping):
                preflight_count += 1
                reason = str(preflight.get("reason", "") or "unknown")
                preflight_reason_counts[reason] += 1
                by_game[game_id]["preflight_count"] += 1
                by_game[game_id][f"preflight:{reason}"] += 1
                if bool(preflight.get("remote_status_negative_rank_guard_enabled")):
                    guard_enabled_preflight_count += 1
                    by_game[game_id]["guard_enabled_preflight_count"] += 1
                skipped_count = _as_int(
                    preflight.get("skipped_remote_status_negative_guard_count"),
                    0,
                )
                if skipped_count > 0:
                    runtime_skip_event_count += 1
                    runtime_skipped_candidate_count_sum += skipped_count
                    by_game[game_id]["runtime_skip_event_count"] += 1
                    by_game[game_id]["runtime_skipped_candidate_count_sum"] += skipped_count
                    if reason == "all_candidates_blocked_by_remote_status_negative_guard":
                        all_candidates_blocked_event_count += 1
                        by_game[game_id]["all_candidates_blocked_event_count"] += 1
                    else:
                        partial_skip_event_count += 1
                        by_game[game_id]["partial_skip_event_count"] += 1
                    samples = [
                        dict(sample)
                        for sample in list(
                            preflight.get("remote_status_negative_guard_samples") or []
                        )
                        if isinstance(sample, Mapping)
                    ]
                    for sample in samples:
                        for key in list(sample.get("matched_transition_keys") or []):
                            blocked_transition_counts[str(key)] += 1
                        target_key = str(sample.get("target_key", "") or "")
                        if target_key:
                            blocked_target_counts[target_key] += 1
                    if len(skip_event_samples) < sample_limit:
                        skip_event_samples.append(
                            {
                                "game_id": game_id,
                                "variant": str(row.get("variant", "") or "unknown"),
                                "step": step,
                                "phase": str(item.get("phase", "") or ""),
                                "reason": reason,
                                "candidate_count": _as_int(
                                    preflight.get("candidate_count"),
                                    0,
                                ),
                                "skipped_remote_status_negative_guard_count": skipped_count,
                                "remote_status_negative_guard_samples": samples[:4],
                            }
                        )

            if bool(unsolved.get("score_relation_candidate_verifier")) and bool(
                unsolved.get("executed") or unsolved.get("used")
            ):
                score_relation_verifier_used_count += 1
                by_game[game_id]["score_relation_verifier_used_count"] += 1
                if bool(unsolved.get("remote_status_negative_rank_guard_enabled")):
                    score_relation_verifier_used_with_guard_enabled_count += 1
                    by_game[game_id][
                        "score_relation_verifier_used_with_guard_enabled_count"
                    ] += 1
                candidate = unsolved.get("score_relation_candidate")
                candidate = candidate if isinstance(candidate, Mapping) else {}
                if len(score_relation_used_samples) < sample_limit:
                    score_relation_used_samples.append(
                        {
                            "game_id": game_id,
                            "variant": str(row.get("variant", "") or "unknown"),
                            "step": step,
                            "phase": str(item.get("phase", "") or ""),
                            "reason": str(unsolved.get("reason", "") or ""),
                            "target_key": str(candidate.get("target_key", "") or ""),
                            "candidate_relation": str(
                                candidate.get("candidate_relation", "") or ""
                            ),
                            "probe_kind": str(candidate.get("probe_kind", "") or ""),
                            "remote_status_negative_rank_guard_enabled": bool(
                                unsolved.get("remote_status_negative_rank_guard_enabled")
                            ),
                        }
                    )

    return {
        "schema_version": "eca_score_relation_remote_status_runtime_guard_outcome_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "preflight_count": int(preflight_count),
        "guard_enabled_preflight_count": int(guard_enabled_preflight_count),
        "runtime_skip_event_count": int(runtime_skip_event_count),
        "runtime_skipped_candidate_count_sum": int(
            runtime_skipped_candidate_count_sum
        ),
        "all_candidates_blocked_event_count": int(all_candidates_blocked_event_count),
        "partial_skip_event_count": int(partial_skip_event_count),
        "target_verification_summary_count": int(target_verification_summary_count),
        "target_verification_blocked_summary_count": int(
            target_verification_blocked_summary_count
        ),
        "score_relation_verifier_used_count": int(score_relation_verifier_used_count),
        "score_relation_verifier_used_with_guard_enabled_count": int(
            score_relation_verifier_used_with_guard_enabled_count
        ),
        "preflight_reason_counts": dict(sorted(preflight_reason_counts.items())),
        "blocked_transition_counts": dict(sorted(blocked_transition_counts.items())),
        "blocked_target_counts": dict(sorted(blocked_target_counts.items())),
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "skip_event_samples": skip_event_samples,
        "target_verification_summary_samples": summary_samples,
        "score_relation_verifier_used_samples": score_relation_used_samples,
        "next_planner_contract": {
            "required_before_default_on": "all25 non-regression plus no positive-candidate suppression",
            "allowed_connection": "runtime_demotion_or_skip_metrics_only",
            "action_generation_allowed": False,
            "positive_ranking_allowed": False,
        },
    }


def _score_relation_shadow_evidence_lane_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    trigger_reason_counts: Counter[str] = Counter()
    candidate_relation_counts: Counter[str] = Counter()
    probe_kind_counts: Counter[str] = Counter()
    preserved_primitive_id_counts: Counter[str] = Counter()
    preserved_primitive_scope_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    shadow_event_count = 0
    candidate_count_sum = 0
    eligible_candidate_count_sum = 0

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            preflight = unsolved.get("score_relation_candidate_verifier_preflight")
            if not isinstance(preflight, Mapping):
                continue
            shadow = preflight.get("score_relation_shadow_evidence")
            if not isinstance(shadow, Mapping):
                continue
            if not bool(shadow.get("enabled")):
                continue

            shadow_event_count += 1
            reason = str(shadow.get("reason", "") or "unknown")
            trigger_reason = str(preflight.get("reason", "") or "unknown")
            primitive_id = str(shadow.get("current_primitive_id", "") or "")
            primitive_scope = str(shadow.get("current_primitive_scope", "") or "")
            candidate_count = _as_int(shadow.get("candidate_count"), 0)
            eligible_count = _as_int(
                shadow.get("eligible_shadow_candidate_count"),
                0,
            )
            candidate_count_sum += candidate_count
            eligible_candidate_count_sum += eligible_count
            reason_counts[reason] += 1
            trigger_reason_counts[trigger_reason] += 1
            if primitive_id:
                preserved_primitive_id_counts[primitive_id] += 1
            if primitive_scope:
                preserved_primitive_scope_counts[primitive_scope] += 1
            by_game[game_id]["shadow_event_count"] += 1
            by_game[game_id]["candidate_count_sum"] += candidate_count
            by_game[game_id]["eligible_shadow_candidate_count_sum"] += eligible_count
            by_game[game_id][f"trigger:{trigger_reason}"] += 1

            for relation, count in dict(
                shadow.get("candidate_relation_counts") or {}
            ).items():
                value = _as_int(count, 0)
                candidate_relation_counts[str(relation)] += value
                by_game[game_id][f"candidate_relation:{relation}"] += value
            for probe_kind, count in dict(shadow.get("probe_kind_counts") or {}).items():
                value = _as_int(count, 0)
                probe_kind_counts[str(probe_kind)] += value
                by_game[game_id][f"probe_kind:{probe_kind}"] += value

            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), 0),
                        "phase": str(item.get("phase", "") or ""),
                        "trigger_reason": trigger_reason,
                        "current_primitive_id": primitive_id,
                        "current_primitive_scope": primitive_scope,
                        "candidate_count": candidate_count,
                        "eligible_shadow_candidate_count": eligible_count,
                        "samples": [
                            dict(sample)
                            for sample in list(shadow.get("samples") or [])[:4]
                            if isinstance(sample, Mapping)
                        ],
                    }
                )

    return {
        "schema_version": "eca_score_relation_shadow_evidence_lane_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "shadow_event_count": int(shadow_event_count),
        "candidate_count_sum": int(candidate_count_sum),
        "eligible_shadow_candidate_count_sum": int(eligible_candidate_count_sum),
        "reason_counts": dict(sorted(reason_counts.items())),
        "trigger_reason_counts": dict(sorted(trigger_reason_counts.items())),
        "candidate_relation_counts": dict(sorted(candidate_relation_counts.items())),
        "probe_kind_counts": dict(sorted(probe_kind_counts.items())),
        "preserved_primitive_id_counts": dict(
            sorted(preserved_primitive_id_counts.items())
        ),
        "preserved_primitive_scope_counts": dict(
            sorted(preserved_primitive_scope_counts.items())
        ),
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "shadow evidence joined with later score/outcome posterior",
            "allowed_connection": "diagnostic_only_until_confirmed",
            "action_generation_allowed": False,
        },
    }


def _score_relation_shadow_verifier_schema_translation_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    schema_counts: Counter[str] = Counter()
    schema_relation_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    shadow_event_count = 0
    candidate_count_sum = 0
    active_verifier_eligible_count_sum = 0
    positive_action_schema_candidate_count = 0
    suppression_schema_candidate_count = 0
    unsupported_schema_candidate_count = 0

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            preflight = unsolved.get("score_relation_candidate_verifier_preflight")
            if not isinstance(preflight, Mapping):
                continue
            shadow = preflight.get("score_relation_shadow_evidence")
            if not isinstance(shadow, Mapping) or not bool(shadow.get("enabled")):
                continue

            schema_map = dict(shadow.get("candidate_verifier_schema_counts") or {})
            if not schema_map:
                continue
            shadow_event_count += 1
            event_candidate_count = _as_int(shadow.get("candidate_count"), 0)
            event_active_eligible = _as_int(
                shadow.get("eligible_shadow_candidate_count"),
                0,
            )
            event_positive = _as_int(
                shadow.get("positive_action_verifier_schema_candidate_count"),
                0,
            )
            event_suppression = _as_int(
                shadow.get("suppression_verifier_schema_candidate_count"),
                0,
            )
            event_unsupported = _as_int(
                shadow.get("unsupported_verifier_schema_candidate_count"),
                0,
            )
            candidate_count_sum += event_candidate_count
            active_verifier_eligible_count_sum += event_active_eligible
            positive_action_schema_candidate_count += event_positive
            suppression_schema_candidate_count += event_suppression
            unsupported_schema_candidate_count += event_unsupported

            by_game[game_id]["shadow_event_count"] += 1
            by_game[game_id]["candidate_count_sum"] += event_candidate_count
            by_game[game_id]["active_verifier_eligible_candidate_count_sum"] += (
                event_active_eligible
            )
            by_game[game_id]["positive_action_schema_candidate_count"] += event_positive
            by_game[game_id]["suppression_schema_candidate_count"] += event_suppression
            by_game[game_id]["unsupported_schema_candidate_count"] += event_unsupported

            for schema, count in schema_map.items():
                value = _as_int(count, 0)
                schema_text = str(schema)
                schema_counts[schema_text] += value
                by_game[game_id][f"schema:{schema_text}"] += value
            for key, count in dict(
                shadow.get("candidate_verifier_schema_relation_counts") or {}
            ).items():
                value = _as_int(count, 0)
                schema_relation_counts[str(key)] += value

            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), 0),
                        "phase": str(item.get("phase", "") or ""),
                        "current_primitive_id": str(
                            shadow.get("current_primitive_id", "") or ""
                        ),
                        "current_primitive_scope": str(
                            shadow.get("current_primitive_scope", "") or ""
                        ),
                        "current_action": str(shadow.get("current_action", "") or ""),
                        "candidate_count": event_candidate_count,
                        "active_verifier_eligible_candidate_count": event_active_eligible,
                        "positive_action_schema_candidate_count": event_positive,
                        "suppression_schema_candidate_count": event_suppression,
                        "unsupported_schema_candidate_count": event_unsupported,
                        "candidate_verifier_schema_counts": dict(
                            sorted(
                                (str(schema), _as_int(count, 0))
                                for schema, count in schema_map.items()
                            )
                        ),
                        "samples": [
                            dict(sample)
                            for sample in list(shadow.get("samples") or [])[:4]
                            if isinstance(sample, Mapping)
                        ],
                    }
                )

    return {
        "schema_version": (
            "eca_score_relation_shadow_verifier_schema_translation_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "shadow_event_count": int(shadow_event_count),
        "candidate_count_sum": int(candidate_count_sum),
        "active_verifier_eligible_candidate_count_sum": int(
            active_verifier_eligible_count_sum
        ),
        "positive_action_schema_candidate_count": int(
            positive_action_schema_candidate_count
        ),
        "suppression_schema_candidate_count": int(suppression_schema_candidate_count),
        "unsupported_schema_candidate_count": int(unsupported_schema_candidate_count),
        "schema_counts": dict(sorted(schema_counts.items())),
        "schema_relation_counts": dict(sorted(schema_relation_counts.items())),
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_execution": "schema-specific verifier must remain default-off, protected, and prove focused/broader/all25 safety",
            "allowed_connection": "diagnostic_only_until_schema_verified",
            "action_generation_allowed": False,
        },
    }


def _score_relation_shadow_evidence_outcome_join_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    window: int = 8,
    sample_limit: int = 16,
) -> dict[str, Any]:
    by_relation: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    trigger_reason_counts: Counter[str] = Counter()
    preserved_primitive_scope_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    shadow_event_count = 0
    candidate_count_sum = 0
    relations_observed: set[str] = set()

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        final_score = _as_int(row.get("final_score"), 0)
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            preflight = unsolved.get("score_relation_candidate_verifier_preflight")
            if not isinstance(preflight, Mapping):
                continue
            shadow = preflight.get("score_relation_shadow_evidence")
            if not isinstance(shadow, Mapping) or not bool(shadow.get("enabled")):
                continue

            relation_counts = {
                str(relation): _as_int(count, 0)
                for relation, count in dict(
                    shadow.get("candidate_relation_counts") or {}
                ).items()
                if str(relation)
            }
            if not relation_counts:
                continue

            shadow_event_count += 1
            event_candidate_count = _as_int(shadow.get("candidate_count"), 0)
            candidate_count_sum += event_candidate_count
            current_score = _trace_score_before(item, final_score)
            if index + 1 < len(trace):
                next_score = _trace_score_before(trace[index + 1], current_score)
            else:
                next_score = final_score
            immediate_delta = max(0, int(next_score) - int(current_score))
            delayed_delta, delay_steps = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=index,
                window=window,
                default_score=current_score,
                final_score=final_score,
            )
            shifted_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=min(index + 1, max(0, len(trace) - 1)),
                window=window,
                default_score=next_score,
                final_score=final_score,
            )
            pre_action_delta = _pre_action_score_delta(
                trace,
                index=index,
                window=window,
                current_score=current_score,
            )
            control_index = _deterministic_control_index(index, len(trace))
            random_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=control_index,
                window=window,
                default_score=_trace_score_before(trace[control_index], current_score)
                if trace
                else current_score,
                final_score=final_score,
            )
            control_max = max(shifted_delta, pre_action_delta, random_delta)
            join_margin = int(delayed_delta) - int(control_max)
            trigger_reason = str(preflight.get("reason", "") or "unknown")
            primitive_scope = str(shadow.get("current_primitive_scope", "") or "")
            trigger_reason_counts[trigger_reason] += 1
            if primitive_scope:
                preserved_primitive_scope_counts[primitive_scope] += 1

            by_game[game_id]["shadow_event_count"] += 1
            by_game[game_id]["candidate_count_sum"] += event_candidate_count
            by_game[game_id]["delayed_score_delta_sum"] += int(delayed_delta)
            by_game[game_id]["immediate_score_delta_sum"] += int(immediate_delta)
            by_game[game_id]["negative_control_max_score_delta_sum"] += int(control_max)
            if delayed_delta > 0:
                by_game[game_id]["delayed_positive_event_count"] += 1
            if join_margin > 0:
                by_game[game_id]["positive_join_margin_event_count"] += 1

            for relation, count in sorted(relation_counts.items()):
                relations_observed.add(relation)
                stats = by_relation[relation]
                stats["shadow_event_count"] += 1
                stats["candidate_count_sum"] += int(count)
                stats["immediate_score_delta_sum"] += int(immediate_delta)
                stats["delayed_score_delta_sum"] += int(delayed_delta)
                stats["shifted_delayed_score_delta_sum"] += int(shifted_delta)
                stats["pre_action_score_delta_sum"] += int(pre_action_delta)
                stats["random_step_score_delta_sum"] += int(random_delta)
                stats["negative_control_max_score_delta_sum"] += int(control_max)
                stats["join_margin_sum"] += int(join_margin)
                if immediate_delta > 0:
                    stats["immediate_positive_event_count"] += 1
                if delayed_delta > 0:
                    stats["delayed_positive_event_count"] += 1
                if join_margin > 0:
                    stats["positive_join_margin_event_count"] += 1
                by_game[game_id][f"relation:{relation}:event_count"] += 1
                by_game[game_id][f"relation:{relation}:candidate_count_sum"] += int(
                    count
                )
                by_game[game_id][
                    f"relation:{relation}:delayed_score_delta_sum"
                ] += int(delayed_delta)

            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), index),
                        "phase": str(item.get("phase", "") or ""),
                        "trigger_reason": trigger_reason,
                        "current_primitive_id": str(
                            shadow.get("current_primitive_id", "") or ""
                        ),
                        "current_primitive_scope": primitive_scope,
                        "candidate_relation_counts": dict(sorted(relation_counts.items())),
                        "score_before": int(current_score),
                        "immediate_score_delta": int(immediate_delta),
                        "delayed_score_delta": int(delayed_delta),
                        "delay_steps": int(delay_steps),
                        "shifted_delayed_score_delta": int(shifted_delta),
                        "pre_action_score_delta": int(pre_action_delta),
                        "random_step_score_delta": int(random_delta),
                        "join_margin": int(join_margin),
                    }
                )

    def convert(counter: Counter[str]) -> dict[str, Any]:
        delayed = int(counter.get("delayed_score_delta_sum", 0))
        control_max = int(counter.get("negative_control_max_score_delta_sum", 0))
        event_count = int(counter.get("shadow_event_count", 0))
        positive_events = int(counter.get("delayed_positive_event_count", 0))
        return {
            **{key: int(value) for key, value in sorted(counter.items())},
            "delayed_positive_rate": _safe_div(positive_events, event_count),
            "outcome_join_margin": delayed - control_max,
            "promotion_hint": (
                "score_correlated_shadow_relation"
                if delayed > 0 and delayed > control_max
                else "no_score_correlation"
            ),
        }

    by_relation_summary = {
        relation: convert(counter) for relation, counter in sorted(by_relation.items())
    }
    score_correlated_relations = sorted(
        relation
        for relation, stats in by_relation_summary.items()
        if str(stats.get("promotion_hint", "")) == "score_correlated_shadow_relation"
    )
    return {
        "schema_version": "eca_score_relation_shadow_evidence_outcome_join_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "window": int(window),
        "shadow_event_count": int(shadow_event_count),
        "candidate_count_sum": int(candidate_count_sum),
        "relation_count": int(len(relations_observed)),
        "score_correlated_relation_count": int(len(score_correlated_relations)),
        "score_correlated_relations": score_correlated_relations,
        "trigger_reason_counts": dict(sorted(trigger_reason_counts.items())),
        "preserved_primitive_scope_counts": dict(
            sorted(preserved_primitive_scope_counts.items())
        ),
        "by_relation": by_relation_summary,
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "relation has positive outcome_join_margin on focused and broader non-regression runs",
            "allowed_connection": "diagnostic_only_until_breadth_confirmed",
            "action_generation_allowed": False,
        },
    }


def _score_relation_shadow_signature_records(
    shadow: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw_counts = shadow.get("candidate_signature_counts")
    if isinstance(raw_counts, Mapping):
        for key, count in sorted(raw_counts.items()):
            parts = str(key).split("\t")
            if len(parts) != 4:
                continue
            candidate_relation, probe_kind, target_key, dominant_action = parts
            records.append(
                {
                    "signature_key": str(key),
                    "candidate_relation": candidate_relation,
                    "probe_kind": probe_kind,
                    "target_key": target_key,
                    "dominant_action": dominant_action,
                    "count": _as_int(count, 0),
                }
            )
    if records:
        return records
    for sample in list(shadow.get("samples") or []):
        if not isinstance(sample, Mapping):
            continue
        candidate_relation = str(sample.get("candidate_relation", "") or "")
        probe_kind = str(sample.get("probe_kind", "") or "")
        target_key = str(sample.get("target_key", "") or "")
        dominant_action = str(sample.get("dominant_action", "") or "")
        signature_key = "\t".join(
            [candidate_relation, probe_kind, target_key, dominant_action]
        )
        records.append(
            {
                "signature_key": signature_key,
                "candidate_relation": candidate_relation,
                "probe_kind": probe_kind,
                "target_key": target_key,
                "dominant_action": dominant_action,
                "count": 1,
            }
        )
    return records


def _score_relation_shadow_signature_outcome_join_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    window: int = 8,
    sample_limit: int = 16,
) -> dict[str, Any]:
    by_signature: dict[str, Counter[str]] = defaultdict(Counter)
    by_relation: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    shadow_event_count = 0
    signature_observation_count = 0
    candidate_count_sum = 0

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        final_score = _as_int(row.get("final_score"), 0)
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            preflight = unsolved.get("score_relation_candidate_verifier_preflight")
            if not isinstance(preflight, Mapping):
                continue
            shadow = preflight.get("score_relation_shadow_evidence")
            if not isinstance(shadow, Mapping) or not bool(shadow.get("enabled")):
                continue
            signature_records = _score_relation_shadow_signature_records(shadow)
            if not signature_records:
                continue

            shadow_event_count += 1
            current_score = _trace_score_before(item, final_score)
            if index + 1 < len(trace):
                next_score = _trace_score_before(trace[index + 1], current_score)
            else:
                next_score = final_score
            immediate_delta = max(0, int(next_score) - int(current_score))
            delayed_delta, delay_steps = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=index,
                window=window,
                default_score=current_score,
                final_score=final_score,
            )
            shifted_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=min(index + 1, max(0, len(trace) - 1)),
                window=window,
                default_score=next_score,
                final_score=final_score,
            )
            pre_action_delta = _pre_action_score_delta(
                trace,
                index=index,
                window=window,
                current_score=current_score,
            )
            control_index = _deterministic_control_index(index, len(trace))
            random_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=control_index,
                window=window,
                default_score=_trace_score_before(trace[control_index], current_score)
                if trace
                else current_score,
                final_score=final_score,
            )
            control_max = max(shifted_delta, pre_action_delta, random_delta)
            join_margin = int(delayed_delta) - int(control_max)

            for record in signature_records:
                signature_key = str(record.get("signature_key", "") or "")
                relation = str(record.get("candidate_relation", "") or "")
                count = _as_int(record.get("count"), 0)
                if not signature_key or count <= 0:
                    continue
                signature_observation_count += 1
                candidate_count_sum += count
                stats = by_signature[signature_key]
                stats["shadow_event_count"] += 1
                stats["candidate_count_sum"] += count
                stats["immediate_score_delta_sum"] += int(immediate_delta)
                stats["delayed_score_delta_sum"] += int(delayed_delta)
                stats["shifted_delayed_score_delta_sum"] += int(shifted_delta)
                stats["pre_action_score_delta_sum"] += int(pre_action_delta)
                stats["random_step_score_delta_sum"] += int(random_delta)
                stats["negative_control_max_score_delta_sum"] += int(control_max)
                stats["join_margin_sum"] += int(join_margin)
                if immediate_delta > 0:
                    stats["immediate_positive_event_count"] += 1
                if delayed_delta > 0:
                    stats["delayed_positive_event_count"] += 1
                if join_margin > 0:
                    stats["positive_join_margin_event_count"] += 1

                relation_stats = by_relation[relation]
                relation_stats["signature_observation_count"] += 1
                relation_stats["candidate_count_sum"] += count
                relation_stats["delayed_score_delta_sum"] += int(delayed_delta)
                relation_stats["negative_control_max_score_delta_sum"] += int(
                    control_max
                )
                relation_stats["join_margin_sum"] += int(join_margin)
                if join_margin > 0:
                    relation_stats["positive_join_margin_event_count"] += 1

                by_game[game_id]["signature_observation_count"] += 1
                by_game[game_id]["candidate_count_sum"] += count
                by_game[game_id]["delayed_score_delta_sum"] += int(delayed_delta)
                by_game[game_id]["negative_control_max_score_delta_sum"] += int(
                    control_max
                )
                by_game[game_id]["join_margin_sum"] += int(join_margin)

                if len(samples) < sample_limit:
                    samples.append(
                        {
                            "game_id": game_id,
                            "variant": str(row.get("variant", "") or "unknown"),
                            "step": _as_int(item.get("step"), index),
                            "phase": str(item.get("phase", "") or ""),
                            "signature_key": signature_key,
                            "candidate_relation": relation,
                            "probe_kind": str(record.get("probe_kind", "") or ""),
                            "target_key": str(record.get("target_key", "") or ""),
                            "dominant_action": str(
                                record.get("dominant_action", "") or ""
                            ),
                            "candidate_count": count,
                            "score_before": int(current_score),
                            "immediate_score_delta": int(immediate_delta),
                            "delayed_score_delta": int(delayed_delta),
                            "delay_steps": int(delay_steps),
                            "negative_control_max_score_delta": int(control_max),
                            "join_margin": int(join_margin),
                        }
                    )

    def convert_signature(key: str, counter: Counter[str]) -> dict[str, Any]:
        parts = str(key).split("\t")
        candidate_relation, probe_kind, target_key, dominant_action = (
            parts if len(parts) == 4 else ["", "", "", ""]
        )
        delayed = int(counter.get("delayed_score_delta_sum", 0))
        control_max = int(counter.get("negative_control_max_score_delta_sum", 0))
        event_count = int(counter.get("shadow_event_count", 0))
        positive_events = int(counter.get("delayed_positive_event_count", 0))
        margin = delayed - control_max
        return {
            "signature_key": key,
            "candidate_relation": candidate_relation,
            "probe_kind": probe_kind,
            "target_key": target_key,
            "dominant_action": dominant_action,
            **{name: int(value) for name, value in sorted(counter.items())},
            "delayed_positive_rate": _safe_div(positive_events, event_count),
            "outcome_join_margin": int(margin),
            "promotion_hint": (
                "score_correlated_shadow_signature"
                if delayed > 0 and margin > 0
                else "no_score_correlation"
            ),
        }

    by_signature_summary = {
        key: convert_signature(key, counter)
        for key, counter in sorted(by_signature.items())
    }
    score_correlated = [
        stats
        for stats in by_signature_summary.values()
        if str(stats.get("promotion_hint", "")) == "score_correlated_shadow_signature"
    ]
    score_correlated_sorted = sorted(
        score_correlated,
        key=lambda item: (
            -int(item.get("outcome_join_margin", 0)),
            -int(item.get("delayed_score_delta_sum", 0)),
            str(item.get("signature_key", "")),
        ),
    )
    top_delayed = sorted(
        by_signature_summary.values(),
        key=lambda item: (
            -int(item.get("delayed_score_delta_sum", 0)),
            -int(item.get("outcome_join_margin", 0)),
            str(item.get("signature_key", "")),
        ),
    )[:20]
    return {
        "schema_version": (
            "eca_score_relation_shadow_signature_outcome_join_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "window": int(window),
        "shadow_event_count": int(shadow_event_count),
        "signature_observation_count": int(signature_observation_count),
        "candidate_count_sum": int(candidate_count_sum),
        "signature_count": int(len(by_signature_summary)),
        "score_correlated_signature_count": int(len(score_correlated_sorted)),
        "score_correlated_signatures": score_correlated_sorted[:20],
        "top_delayed_score_signatures": top_delayed,
        "by_signature": by_signature_summary,
        "by_relation": {
            relation: dict(sorted(stats.items()))
            for relation, stats in sorted(by_relation.items())
        },
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "signature has positive outcome_join_margin on focused and broader non-regression runs",
            "allowed_connection": "diagnostic_only_until_signature_confirmed",
            "action_generation_allowed": False,
        },
    }


def _score_relation_shadow_context_signature_records(
    shadow: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw_counts = shadow.get("candidate_context_signature_counts")
    if isinstance(raw_counts, Mapping):
        for key, count in sorted(raw_counts.items()):
            parts = str(key).split("\t")
            if len(parts) != 7:
                continue
            (
                candidate_relation,
                probe_kind,
                target_key,
                dominant_action,
                current_primitive_id,
                current_primitive_scope,
                current_action,
            ) = parts
            records.append(
                {
                    "context_signature_key": str(key),
                    "candidate_relation": candidate_relation,
                    "probe_kind": probe_kind,
                    "target_key": target_key,
                    "dominant_action": dominant_action,
                    "current_primitive_id": current_primitive_id,
                    "current_primitive_scope": current_primitive_scope,
                    "current_action": current_action,
                    "count": _as_int(count, 0),
                }
            )
    if records:
        return records

    current_primitive_id = str(shadow.get("current_primitive_id", "") or "")
    current_primitive_scope = str(shadow.get("current_primitive_scope", "") or "")
    current_action = str(shadow.get("current_action", "") or "")
    for record in _score_relation_shadow_signature_records(shadow):
        candidate_relation = str(record.get("candidate_relation", "") or "")
        probe_kind = str(record.get("probe_kind", "") or "")
        target_key = str(record.get("target_key", "") or "")
        dominant_action = str(record.get("dominant_action", "") or "")
        context_signature_key = "\t".join(
            [
                candidate_relation,
                probe_kind,
                target_key,
                dominant_action,
                current_primitive_id,
                current_primitive_scope,
                current_action,
            ]
        )
        records.append(
            {
                "context_signature_key": context_signature_key,
                "candidate_relation": candidate_relation,
                "probe_kind": probe_kind,
                "target_key": target_key,
                "dominant_action": dominant_action,
                "current_primitive_id": current_primitive_id,
                "current_primitive_scope": current_primitive_scope,
                "current_action": current_action,
                "count": _as_int(record.get("count"), 0),
            }
        )
    return records


def _score_relation_shadow_context_signature_outcome_join_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    window: int = 8,
    sample_limit: int = 16,
) -> dict[str, Any]:
    by_context_signature: dict[str, Counter[str]] = defaultdict(Counter)
    by_preserved_path: dict[str, Counter[str]] = defaultdict(Counter)
    by_relation: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    shadow_event_count = 0
    context_signature_observation_count = 0
    candidate_count_sum = 0

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        final_score = _as_int(row.get("final_score"), 0)
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            preflight = unsolved.get("score_relation_candidate_verifier_preflight")
            if not isinstance(preflight, Mapping):
                continue
            shadow = preflight.get("score_relation_shadow_evidence")
            if not isinstance(shadow, Mapping) or not bool(shadow.get("enabled")):
                continue
            context_records = _score_relation_shadow_context_signature_records(shadow)
            if not context_records:
                continue

            shadow_event_count += 1
            current_score = _trace_score_before(item, final_score)
            if index + 1 < len(trace):
                next_score = _trace_score_before(trace[index + 1], current_score)
            else:
                next_score = final_score
            immediate_delta = max(0, int(next_score) - int(current_score))
            delayed_delta, delay_steps = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=index,
                window=window,
                default_score=current_score,
                final_score=final_score,
            )
            shifted_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=min(index + 1, max(0, len(trace) - 1)),
                window=window,
                default_score=next_score,
                final_score=final_score,
            )
            pre_action_delta = _pre_action_score_delta(
                trace,
                index=index,
                window=window,
                current_score=current_score,
            )
            control_index = _deterministic_control_index(index, len(trace))
            random_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=control_index,
                window=window,
                default_score=_trace_score_before(trace[control_index], current_score)
                if trace
                else current_score,
                final_score=final_score,
            )
            control_max = max(shifted_delta, pre_action_delta, random_delta)
            join_margin = int(delayed_delta) - int(control_max)

            for record in context_records:
                key = str(record.get("context_signature_key", "") or "")
                relation = str(record.get("candidate_relation", "") or "")
                primitive_id = str(record.get("current_primitive_id", "") or "")
                primitive_scope = str(record.get("current_primitive_scope", "") or "")
                current_action = str(record.get("current_action", "") or "")
                preserved_path_key = "\t".join(
                    [primitive_id, primitive_scope, current_action]
                )
                count = _as_int(record.get("count"), 0)
                if not key or count <= 0:
                    continue
                context_signature_observation_count += 1
                candidate_count_sum += count

                stats = by_context_signature[key]
                stats["shadow_event_count"] += 1
                stats["candidate_count_sum"] += count
                stats["immediate_score_delta_sum"] += int(immediate_delta)
                stats["delayed_score_delta_sum"] += int(delayed_delta)
                stats["shifted_delayed_score_delta_sum"] += int(shifted_delta)
                stats["pre_action_score_delta_sum"] += int(pre_action_delta)
                stats["random_step_score_delta_sum"] += int(random_delta)
                stats["negative_control_max_score_delta_sum"] += int(control_max)
                stats["join_margin_sum"] += int(join_margin)
                if immediate_delta > 0:
                    stats["immediate_positive_event_count"] += 1
                if delayed_delta > 0:
                    stats["delayed_positive_event_count"] += 1
                if join_margin > 0:
                    stats["positive_join_margin_event_count"] += 1

                path_stats = by_preserved_path[preserved_path_key]
                path_stats["context_signature_observation_count"] += 1
                path_stats["candidate_count_sum"] += count
                path_stats["delayed_score_delta_sum"] += int(delayed_delta)
                path_stats["negative_control_max_score_delta_sum"] += int(control_max)
                path_stats["join_margin_sum"] += int(join_margin)
                if join_margin > 0:
                    path_stats["positive_join_margin_event_count"] += 1

                relation_stats = by_relation[relation]
                relation_stats["context_signature_observation_count"] += 1
                relation_stats["candidate_count_sum"] += count
                relation_stats["delayed_score_delta_sum"] += int(delayed_delta)
                relation_stats["negative_control_max_score_delta_sum"] += int(
                    control_max
                )
                relation_stats["join_margin_sum"] += int(join_margin)
                if join_margin > 0:
                    relation_stats["positive_join_margin_event_count"] += 1

                if len(samples) < sample_limit:
                    samples.append(
                        {
                            "game_id": game_id,
                            "variant": str(row.get("variant", "") or "unknown"),
                            "step": _as_int(item.get("step"), index),
                            "phase": str(item.get("phase", "") or ""),
                            "context_signature_key": key,
                            "candidate_relation": relation,
                            "probe_kind": str(record.get("probe_kind", "") or ""),
                            "target_key": str(record.get("target_key", "") or ""),
                            "dominant_action": str(
                                record.get("dominant_action", "") or ""
                            ),
                            "current_primitive_id": primitive_id,
                            "current_primitive_scope": primitive_scope,
                            "current_action": current_action,
                            "candidate_count": count,
                            "score_before": int(current_score),
                            "immediate_score_delta": int(immediate_delta),
                            "delayed_score_delta": int(delayed_delta),
                            "delay_steps": int(delay_steps),
                            "negative_control_max_score_delta": int(control_max),
                            "join_margin": int(join_margin),
                        }
                    )

    def convert_context(key: str, counter: Counter[str]) -> dict[str, Any]:
        parts = str(key).split("\t")
        (
            candidate_relation,
            probe_kind,
            target_key,
            dominant_action,
            current_primitive_id,
            current_primitive_scope,
            current_action,
        ) = parts if len(parts) == 7 else ["", "", "", "", "", "", ""]
        delayed = int(counter.get("delayed_score_delta_sum", 0))
        control_max = int(counter.get("negative_control_max_score_delta_sum", 0))
        event_count = int(counter.get("shadow_event_count", 0))
        positive_events = int(counter.get("delayed_positive_event_count", 0))
        margin = delayed - control_max
        return {
            "context_signature_key": key,
            "candidate_relation": candidate_relation,
            "probe_kind": probe_kind,
            "target_key": target_key,
            "dominant_action": dominant_action,
            "current_primitive_id": current_primitive_id,
            "current_primitive_scope": current_primitive_scope,
            "current_action": current_action,
            **{name: int(value) for name, value in sorted(counter.items())},
            "delayed_positive_rate": _safe_div(positive_events, event_count),
            "outcome_join_margin": int(margin),
            "promotion_hint": (
                "score_correlated_shadow_context_signature"
                if delayed > 0 and margin > 0
                else "no_score_correlation"
            ),
        }

    by_context_signature_summary = {
        key: convert_context(key, counter)
        for key, counter in sorted(by_context_signature.items())
    }
    score_correlated = [
        stats
        for stats in by_context_signature_summary.values()
        if str(stats.get("promotion_hint", ""))
        == "score_correlated_shadow_context_signature"
    ]
    score_correlated_sorted = sorted(
        score_correlated,
        key=lambda item: (
            -int(item.get("outcome_join_margin", 0)),
            -int(item.get("delayed_score_delta_sum", 0)),
            str(item.get("context_signature_key", "")),
        ),
    )
    top_delayed = sorted(
        by_context_signature_summary.values(),
        key=lambda item: (
            -int(item.get("delayed_score_delta_sum", 0)),
            -int(item.get("outcome_join_margin", 0)),
            str(item.get("context_signature_key", "")),
        ),
    )[:20]
    return {
        "schema_version": (
            "eca_score_relation_shadow_context_signature_outcome_join_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "window": int(window),
        "shadow_event_count": int(shadow_event_count),
        "context_signature_observation_count": int(
            context_signature_observation_count
        ),
        "candidate_count_sum": int(candidate_count_sum),
        "context_signature_count": int(len(by_context_signature_summary)),
        "score_correlated_context_signature_count": int(
            len(score_correlated_sorted)
        ),
        "score_correlated_context_signatures": score_correlated_sorted[:20],
        "top_delayed_score_context_signatures": top_delayed,
        "by_context_signature": by_context_signature_summary,
        "by_preserved_path": {
            key: dict(sorted(stats.items()))
            for key, stats in sorted(by_preserved_path.items())
        },
        "by_relation": {
            relation: dict(sorted(stats.items()))
            for relation, stats in sorted(by_relation.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "context signature has positive outcome_join_margin on focused and broader non-regression runs",
            "allowed_connection": "diagnostic_only_until_context_confirmed",
            "action_generation_allowed": False,
        },
    }


def _score_relation_shadow_signature_credit_split_sensor(
    signature_outcome_sensor: Mapping[str, Any],
) -> dict[str, Any]:
    skip_credit: list[dict[str, Any]] = []
    action_credit_candidates: list[dict[str, Any]] = []
    ambiguous_credit: list[dict[str, Any]] = []

    for item in list(signature_outcome_sensor.get("score_correlated_signatures") or []):
        if not isinstance(item, Mapping):
            continue
        record = dict(item)
        probe_kind = str(record.get("probe_kind", "") or "")
        if "do_not_repeat" in probe_kind or "falsified" in probe_kind:
            record["credit_kind"] = "skip_credit"
            record["positive_ranking_allowed"] = False
            record["reason"] = "score followed non-execution of a falsified probe"
            skip_credit.append(record)
        elif probe_kind in {
            "site_reprobe_score_relation_probe",
            "terminal_route_score_relation_probe",
            "bounded_repeat_score_relation_probe",
            "goal_binding_relation_probe",
        }:
            record["credit_kind"] = "action_credit_candidate"
            record["positive_ranking_allowed"] = True
            record["reason"] = "positive signature uses an executable score-relation probe kind"
            action_credit_candidates.append(record)
        else:
            record["credit_kind"] = "ambiguous_credit"
            record["positive_ranking_allowed"] = False
            record["reason"] = "probe kind is not yet mapped to skip or action credit"
            ambiguous_credit.append(record)

    positive_count = _as_int(
        signature_outcome_sensor.get("score_correlated_signature_count"),
        0,
    )
    return {
        "schema_version": (
            "eca_score_relation_shadow_signature_credit_split_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "source_sensor": str(
            signature_outcome_sensor.get("schema_version", "") or ""
        ),
        "positive_signature_count": int(positive_count),
        "skip_credit_signature_count": int(len(skip_credit)),
        "action_credit_candidate_signature_count": int(
            len(action_credit_candidates)
        ),
        "ambiguous_credit_signature_count": int(len(ambiguous_credit)),
        "positive_ranking_ready": bool(action_credit_candidates),
        "skip_credit_signatures": skip_credit[:20],
        "action_credit_candidate_signatures": action_credit_candidates[:20],
        "ambiguous_credit_signatures": ambiguous_credit[:20],
        "next_planner_contract": {
            "required_before_positive_ranking": "non-skip action-credit signatures reproduce under focused and broader controls",
            "skip_credit_allowed_connection": "negative_or_suppression_posterior_only",
            "action_generation_allowed": False,
        },
    }


def _score_relation_shadow_context_signature_credit_split_sensor(
    context_outcome_sensor: Mapping[str, Any],
) -> dict[str, Any]:
    skip_credit: list[dict[str, Any]] = []
    action_credit_candidates: list[dict[str, Any]] = []
    ambiguous_credit: list[dict[str, Any]] = []

    for item in list(
        context_outcome_sensor.get("score_correlated_context_signatures") or []
    ):
        if not isinstance(item, Mapping):
            continue
        record = dict(item)
        probe_kind = str(record.get("probe_kind", "") or "")
        if "do_not_repeat" in probe_kind or "falsified" in probe_kind:
            record["credit_kind"] = "skip_credit"
            record["positive_ranking_allowed"] = False
            record["reason"] = (
                "score followed preserved-path execution while a falsified probe remained unexecuted"
            )
            skip_credit.append(record)
        elif probe_kind in {
            "site_reprobe_score_relation_probe",
            "terminal_route_score_relation_probe",
            "bounded_repeat_score_relation_probe",
            "goal_binding_relation_probe",
        }:
            record["credit_kind"] = "action_credit_candidate"
            record["positive_ranking_allowed"] = True
            record["reason"] = (
                "score-correlated context uses an executable score-relation probe kind"
            )
            action_credit_candidates.append(record)
        else:
            record["credit_kind"] = "ambiguous_credit"
            record["positive_ranking_allowed"] = False
            record["reason"] = "context probe kind is not yet mapped to skip or action credit"
            ambiguous_credit.append(record)

    positive_count = _as_int(
        context_outcome_sensor.get("score_correlated_context_signature_count"),
        0,
    )
    return {
        "schema_version": (
            "eca_score_relation_shadow_context_signature_credit_split_sensor.v1"
        ),
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_allowed": False,
        "source_sensor": str(
            context_outcome_sensor.get("schema_version", "") or ""
        ),
        "positive_context_signature_count": int(positive_count),
        "skip_credit_context_signature_count": int(len(skip_credit)),
        "action_credit_candidate_context_signature_count": int(
            len(action_credit_candidates)
        ),
        "ambiguous_credit_context_signature_count": int(len(ambiguous_credit)),
        "positive_ranking_ready": bool(action_credit_candidates),
        "skip_credit_context_signatures": skip_credit[:20],
        "action_credit_candidate_context_signatures": action_credit_candidates[:20],
        "ambiguous_credit_context_signatures": ambiguous_credit[:20],
        "next_planner_contract": {
            "required_before_positive_ranking": "non-skip action-credit context signatures reproduce under focused and broader controls",
            "skip_credit_allowed_connection": "negative_or_suppression_posterior_only",
            "action_generation_allowed": False,
        },
    }


def _score_relation_terminal_condition_gap_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    terminal_states = {"WIN", "WON", "SUCCESS", "SOLVED"}
    gap_state_counts: Counter[str] = Counter()
    candidate_relation_counts: Counter[str] = Counter()
    effect_type_counts: Counter[str] = Counter()
    post_state_counts: Counter[str] = Counter()
    by_candidate_relation: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    row_count = 0
    score_delta_sum = 0

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            if not bool(unsolved.get("score_relation_candidate_verifier")):
                continue
            if not bool(unsolved.get("executed") or unsolved.get("used")):
                continue

            candidate = unsolved.get("score_relation_candidate")
            candidate = candidate if isinstance(candidate, Mapping) else {}
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            observed = unsolved.get("observed_effect")
            observed = observed if isinstance(observed, Mapping) else {}
            candidate_relation = str(
                candidate.get("candidate_relation", "") or "unknown"
            )
            target_key = str(candidate.get("target_key", "") or "unknown")
            score_delta = _as_int(
                outcome.get("score_delta", observed.get("score_delta", 0)),
                0,
            )
            goal_bearing = bool(outcome.get("goal_bearing") or score_delta > 0)
            post_state = str(outcome.get("post_state", "") or "unknown")
            post_terminal = post_state in terminal_states
            objective_progress = bool(
                outcome.get("objective_progress") or observed.get("objective_progress")
            )
            goal_aligned_progress = bool(
                outcome.get("goal_aligned_progress")
                or observed.get("goal_aligned_progress")
            )
            structural_progress = bool(
                outcome.get("structural_progress")
                or observed.get("structural_progress")
            )
            no_op = bool(outcome.get("no_op") or observed.get("no_op"))
            state_changed = bool(outcome.get("state_changed") or observed.get("state_changed"))
            proximity_delta = _as_float(outcome.get("objective_proximity_delta"), 0.0)
            effect_type = str(
                observed.get("effect_type", "") or outcome.get("effect_type", "") or "unknown"
            )

            if goal_bearing or post_terminal:
                gap_state = "score_or_terminal_confirmed"
            elif objective_progress or goal_aligned_progress:
                gap_state = "objective_progress_without_terminal_score"
            elif no_op:
                gap_state = "noop_without_terminal_score"
            elif structural_progress and proximity_delta < 0.0:
                gap_state = "visible_effect_regresses_terminal_distance"
            elif structural_progress or effect_type == "visible_effect":
                gap_state = "visible_effect_without_terminal_progress"
            elif state_changed:
                gap_state = "state_changed_without_terminal_score"
            else:
                gap_state = "no_terminal_progress"

            row_count += 1
            score_delta_sum += score_delta
            gap_state_counts[gap_state] += 1
            candidate_relation_counts[candidate_relation] += 1
            effect_type_counts[effect_type] += 1
            post_state_counts[post_state] += 1
            by_candidate_relation[candidate_relation]["row_count"] += 1
            by_candidate_relation[candidate_relation][f"gap_state:{gap_state}"] += 1
            by_candidate_relation[candidate_relation]["score_delta_sum"] += score_delta
            by_game[game_id]["row_count"] += 1
            by_game[game_id][f"gap_state:{gap_state}"] += 1
            by_game[game_id]["score_delta_sum"] += score_delta

            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), 0),
                        "phase": str(item.get("phase", "") or ""),
                        "action": str(item.get("action", "") or ""),
                        "target_key": target_key,
                        "candidate_relation": candidate_relation,
                        "gap_state": gap_state,
                        "effect_type": effect_type,
                        "score_delta": score_delta,
                        "pre_state": str(outcome.get("pre_state", "") or "unknown"),
                        "post_state": post_state,
                        "post_terminal": bool(post_terminal),
                        "state_changed": bool(state_changed),
                        "objective_progress": bool(objective_progress),
                        "goal_aligned_progress": bool(goal_aligned_progress),
                        "objective_proximity_delta": round(float(proximity_delta), 6),
                        "objective_distance_before": round(
                            _as_float(outcome.get("objective_distance_before"), 0.0),
                            6,
                        ),
                        "objective_distance_after": round(
                            _as_float(outcome.get("objective_distance_after"), 0.0),
                            6,
                        ),
                    }
                )

    score_or_terminal_confirmed_count = _as_int(
        gap_state_counts.get("score_or_terminal_confirmed"),
        0,
    )
    return {
        "schema_version": "eca_score_relation_terminal_condition_gap_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "row_count": int(row_count),
        "score_delta_sum": int(score_delta_sum),
        "score_or_terminal_confirmed_count": int(score_or_terminal_confirmed_count),
        "gap_state_counts": dict(sorted(gap_state_counts.items())),
        "candidate_relation_counts": dict(sorted(candidate_relation_counts.items())),
        "effect_type_counts": dict(sorted(effect_type_counts.items())),
        "post_state_counts": dict(sorted(post_state_counts.items())),
        "by_candidate_relation": {
            relation: dict(sorted(stats.items()))
            for relation, stats in sorted(by_candidate_relation.items())
        },
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "score_or_terminal_confirmed_count > 0",
            "allowed_connection": "diagnostic_to_score_predicate_sensor_only",
            "action_generation_allowed": False,
        },
    }


def _score_relation_terminal_requirement_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    terminal_states = {"WIN", "WON", "SUCCESS", "SOLVED"}
    requirement_counts: Counter[str] = Counter()
    gap_state_counts: Counter[str] = Counter()
    candidate_relation_counts: Counter[str] = Counter()
    by_candidate_relation: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    requirements: list[dict[str, Any]] = []
    row_count = 0
    score_delta_sum = 0

    def classify_requirement(
        *,
        goal_bearing: bool,
        post_terminal: bool,
        objective_progress: bool,
        goal_aligned_progress: bool,
        structural_progress: bool,
        no_op: bool,
        state_changed: bool,
        proximity_delta: float,
        distance_before: float,
        distance_after: float,
        effect_type: str,
    ) -> tuple[str, str]:
        if goal_bearing or post_terminal:
            return "none_score_or_terminal_confirmed", "score_or_terminal_confirmed"
        if objective_progress or goal_aligned_progress:
            return (
                "terminal_predicate_after_objective_progress_required",
                "objective_progress_without_terminal_score",
            )
        if no_op:
            return "affordance_precondition_required", "noop_without_terminal_score"
        if structural_progress and proximity_delta < 0.0:
            return (
                "candidate_direction_or_target_relation_repair_required",
                "visible_effect_regresses_terminal_distance",
            )
        if structural_progress or effect_type == "visible_effect":
            if distance_before == 0.0 and distance_after == 0.0:
                return (
                    "objective_target_binding_required",
                    "visible_effect_without_terminal_progress",
                )
            return (
                "terminal_progress_predicate_required",
                "visible_effect_without_terminal_progress",
            )
        if state_changed:
            return (
                "state_transition_score_predicate_required",
                "state_changed_without_terminal_score",
            )
        return "terminal_condition_sensor_required", "no_terminal_progress"

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            if not bool(unsolved.get("score_relation_candidate_verifier")):
                continue
            if not bool(unsolved.get("executed") or unsolved.get("used")):
                continue

            candidate = unsolved.get("score_relation_candidate")
            candidate = candidate if isinstance(candidate, Mapping) else {}
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            observed = unsolved.get("observed_effect")
            observed = observed if isinstance(observed, Mapping) else {}
            candidate_relation = str(
                candidate.get("candidate_relation", "") or "unknown"
            )
            target_key = str(candidate.get("target_key", "") or "unknown")
            score_delta = _as_int(
                outcome.get("score_delta", observed.get("score_delta", 0)),
                0,
            )
            post_state = str(outcome.get("post_state", "") or "unknown")
            post_terminal = post_state in terminal_states
            goal_bearing = bool(outcome.get("goal_bearing") or score_delta > 0)
            objective_progress = bool(
                outcome.get("objective_progress") or observed.get("objective_progress")
            )
            goal_aligned_progress = bool(
                outcome.get("goal_aligned_progress")
                or observed.get("goal_aligned_progress")
            )
            structural_progress = bool(
                outcome.get("structural_progress")
                or observed.get("structural_progress")
            )
            no_op = bool(outcome.get("no_op") or observed.get("no_op"))
            state_changed = bool(outcome.get("state_changed") or observed.get("state_changed"))
            proximity_delta = _as_float(outcome.get("objective_proximity_delta"), 0.0)
            distance_before = _as_float(outcome.get("objective_distance_before"), 0.0)
            distance_after = _as_float(outcome.get("objective_distance_after"), 0.0)
            effect_type = str(
                observed.get("effect_type", "") or outcome.get("effect_type", "") or "unknown"
            )

            requirement, gap_state = classify_requirement(
                goal_bearing=goal_bearing,
                post_terminal=post_terminal,
                objective_progress=objective_progress,
                goal_aligned_progress=goal_aligned_progress,
                structural_progress=structural_progress,
                no_op=no_op,
                state_changed=state_changed,
                proximity_delta=proximity_delta,
                distance_before=distance_before,
                distance_after=distance_after,
                effect_type=effect_type,
            )

            row_count += 1
            score_delta_sum += score_delta
            requirement_counts[requirement] += 1
            gap_state_counts[gap_state] += 1
            candidate_relation_counts[candidate_relation] += 1
            by_candidate_relation[candidate_relation]["row_count"] += 1
            by_candidate_relation[candidate_relation][f"requirement:{requirement}"] += 1
            by_candidate_relation[candidate_relation][f"gap_state:{gap_state}"] += 1
            by_candidate_relation[candidate_relation]["score_delta_sum"] += score_delta
            by_game[game_id]["row_count"] += 1
            by_game[game_id][f"requirement:{requirement}"] += 1
            by_game[game_id][f"gap_state:{gap_state}"] += 1
            by_game[game_id]["score_delta_sum"] += score_delta

            if len(requirements) < sample_limit:
                requirements.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), 0),
                        "target_key": target_key,
                        "candidate_relation": candidate_relation,
                        "requirement": requirement,
                        "gap_state": gap_state,
                        "effect_type": effect_type,
                        "score_delta": score_delta,
                        "post_state": post_state,
                        "post_terminal": bool(post_terminal),
                        "objective_progress": bool(objective_progress),
                        "goal_aligned_progress": bool(goal_aligned_progress),
                        "objective_proximity_delta": round(float(proximity_delta), 6),
                        "objective_distance_before": round(float(distance_before), 6),
                        "objective_distance_after": round(float(distance_after), 6),
                    }
                )

    return {
        "schema_version": "eca_score_relation_terminal_requirement_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "row_count": int(row_count),
        "score_delta_sum": int(score_delta_sum),
        "score_or_terminal_confirmed_count": int(
            requirement_counts.get("none_score_or_terminal_confirmed", 0)
        ),
        "requirement_counts": dict(sorted(requirement_counts.items())),
        "gap_state_counts": dict(sorted(gap_state_counts.items())),
        "candidate_relation_counts": dict(sorted(candidate_relation_counts.items())),
        "by_candidate_relation": {
            relation: dict(sorted(stats.items()))
            for relation, stats in sorted(by_candidate_relation.items())
        },
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "requirements": requirements,
        "next_planner_contract": {
            "required_before_ranking": "terminal requirement resolved by score or terminal evidence",
            "allowed_connection": "diagnostic_to_terminal_condition_sensor_only",
            "action_generation_allowed": False,
        },
    }


def _score_relation_objective_target_binding_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    terminal_states = {"WIN", "WON", "SUCCESS", "SOLVED"}
    binding_state_counts: Counter[str] = Counter()
    candidate_relation_counts: Counter[str] = Counter()
    by_candidate_relation: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    row_count = 0
    score_delta_sum = 0

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            if not bool(unsolved.get("score_relation_candidate_verifier")):
                continue
            if not bool(unsolved.get("executed") or unsolved.get("used")):
                continue

            candidate = unsolved.get("score_relation_candidate")
            candidate = candidate if isinstance(candidate, Mapping) else {}
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            observed = unsolved.get("observed_effect")
            observed = observed if isinstance(observed, Mapping) else {}
            candidate_relation = str(
                candidate.get("candidate_relation", "") or "unknown"
            )
            target_key = str(candidate.get("target_key", "") or "unknown")
            score_delta = _as_int(
                outcome.get("score_delta", observed.get("score_delta", 0)),
                0,
            )
            post_state = str(outcome.get("post_state", "") or "unknown")
            post_terminal = post_state in terminal_states
            goal_bearing = bool(outcome.get("goal_bearing") or score_delta > 0)
            objective_progress = bool(
                outcome.get("objective_progress") or observed.get("objective_progress")
            )
            goal_aligned_progress = bool(
                outcome.get("goal_aligned_progress")
                or observed.get("goal_aligned_progress")
            )
            structural_progress = bool(
                outcome.get("structural_progress")
                or observed.get("structural_progress")
            )
            effect_type = str(
                observed.get("effect_type", "") or outcome.get("effect_type", "") or "unknown"
            )
            proximity_delta = _as_float(outcome.get("objective_proximity_delta"), 0.0)
            distance_before = _as_float(outcome.get("objective_distance_before"), 0.0)
            distance_after = _as_float(outcome.get("objective_distance_after"), 0.0)
            has_distance_signal = bool(distance_before > 0.0 or distance_after > 0.0)

            if goal_bearing or post_terminal:
                binding_state = "score_or_terminal_confirmed"
            elif objective_progress or goal_aligned_progress:
                binding_state = "objective_bound_progress_without_terminal"
            elif structural_progress and proximity_delta < 0.0:
                binding_state = "objective_bound_regressed"
            elif structural_progress and not has_distance_signal:
                binding_state = "objective_target_unbound_visible_effect"
            elif structural_progress or effect_type == "visible_effect":
                binding_state = "objective_bound_no_progress"
            else:
                binding_state = "no_objective_binding_evidence"

            row_count += 1
            score_delta_sum += score_delta
            binding_state_counts[binding_state] += 1
            candidate_relation_counts[candidate_relation] += 1
            by_candidate_relation[candidate_relation]["row_count"] += 1
            by_candidate_relation[candidate_relation][f"binding_state:{binding_state}"] += 1
            by_candidate_relation[candidate_relation]["score_delta_sum"] += score_delta
            by_game[game_id]["row_count"] += 1
            by_game[game_id][f"binding_state:{binding_state}"] += 1
            by_game[game_id]["score_delta_sum"] += score_delta

            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), 0),
                        "target_key": target_key,
                        "candidate_relation": candidate_relation,
                        "binding_state": binding_state,
                        "effect_type": effect_type,
                        "score_delta": score_delta,
                        "post_state": post_state,
                        "post_terminal": bool(post_terminal),
                        "objective_progress": bool(objective_progress),
                        "goal_aligned_progress": bool(goal_aligned_progress),
                        "objective_proximity_delta": round(float(proximity_delta), 6),
                        "objective_distance_before": round(float(distance_before), 6),
                        "objective_distance_after": round(float(distance_after), 6),
                        "has_distance_signal": bool(has_distance_signal),
                    }
                )

    return {
        "schema_version": "eca_score_relation_objective_target_binding_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "row_count": int(row_count),
        "score_delta_sum": int(score_delta_sum),
        "binding_state_counts": dict(sorted(binding_state_counts.items())),
        "candidate_relation_counts": dict(sorted(candidate_relation_counts.items())),
        "by_candidate_relation": {
            relation: dict(sorted(stats.items()))
            for relation, stats in sorted(by_candidate_relation.items())
        },
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "objective binding plus terminal or score evidence",
            "allowed_connection": "diagnostic_to_objective_target_binding_sensor_only",
            "action_generation_allowed": False,
        },
    }


def _score_relation_terminal_readiness_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    terminal_states = {"WIN", "WON", "SUCCESS", "SOLVED"}
    readiness_state_counts: Counter[str] = Counter()
    candidate_relation_counts: Counter[str] = Counter()
    by_candidate_relation: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    row_count = 0
    score_delta_sum = 0

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            if not bool(unsolved.get("score_relation_candidate_verifier")):
                continue
            if not bool(unsolved.get("executed") or unsolved.get("used")):
                continue

            candidate = unsolved.get("score_relation_candidate")
            candidate = candidate if isinstance(candidate, Mapping) else {}
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            observed = unsolved.get("observed_effect")
            observed = observed if isinstance(observed, Mapping) else {}
            candidate_relation = str(
                candidate.get("candidate_relation", "") or "unknown"
            )
            target_key = str(candidate.get("target_key", "") or "unknown")
            score_delta = _as_int(
                outcome.get("score_delta", observed.get("score_delta", 0)),
                0,
            )
            post_state = str(outcome.get("post_state", "") or "unknown")
            post_terminal = post_state in terminal_states
            goal_bearing = bool(outcome.get("goal_bearing") or score_delta > 0)
            objective_progress = bool(
                outcome.get("objective_progress") or observed.get("objective_progress")
            )
            goal_aligned_progress = bool(
                outcome.get("goal_aligned_progress")
                or observed.get("goal_aligned_progress")
            )
            structural_progress = bool(
                outcome.get("structural_progress")
                or observed.get("structural_progress")
            )
            effect_type = str(
                observed.get("effect_type", "") or outcome.get("effect_type", "") or "unknown"
            )
            proximity_delta = _as_float(outcome.get("objective_proximity_delta"), 0.0)
            distance_before = _as_float(outcome.get("objective_distance_before"), 0.0)
            distance_after = _as_float(outcome.get("objective_distance_after"), 0.0)
            has_distance_signal = bool(distance_before > 0.0 or distance_after > 0.0)

            if goal_bearing or post_terminal:
                readiness_state = "score_or_terminal_confirmed"
            elif not has_distance_signal:
                readiness_state = "objective_target_unbound_no_terminal_readiness"
            elif distance_after <= 0.0 and (objective_progress or proximity_delta > 0.0):
                readiness_state = "objective_target_reached_terminal_predicate_missing"
            elif objective_progress and proximity_delta > 0.0 and distance_after > 0.0:
                readiness_state = "objective_target_closer_continue_required"
            elif structural_progress and proximity_delta < 0.0:
                readiness_state = "objective_target_regressed"
            elif structural_progress or effect_type == "visible_effect":
                readiness_state = "objective_target_bound_no_closure_progress"
            else:
                readiness_state = "terminal_readiness_unknown"

            row_count += 1
            score_delta_sum += score_delta
            readiness_state_counts[readiness_state] += 1
            candidate_relation_counts[candidate_relation] += 1
            by_candidate_relation[candidate_relation]["row_count"] += 1
            by_candidate_relation[candidate_relation][
                f"readiness_state:{readiness_state}"
            ] += 1
            by_candidate_relation[candidate_relation]["score_delta_sum"] += score_delta
            by_game[game_id]["row_count"] += 1
            by_game[game_id][f"readiness_state:{readiness_state}"] += 1
            by_game[game_id]["score_delta_sum"] += score_delta

            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), 0),
                        "target_key": target_key,
                        "candidate_relation": candidate_relation,
                        "readiness_state": readiness_state,
                        "effect_type": effect_type,
                        "score_delta": score_delta,
                        "post_state": post_state,
                        "post_terminal": bool(post_terminal),
                        "objective_progress": bool(objective_progress),
                        "goal_aligned_progress": bool(goal_aligned_progress),
                        "objective_proximity_delta": round(float(proximity_delta), 6),
                        "objective_distance_before": round(float(distance_before), 6),
                        "objective_distance_after": round(float(distance_after), 6),
                        "has_distance_signal": bool(has_distance_signal),
                    }
                )

    score_or_terminal_confirmed_count = _as_int(
        readiness_state_counts.get("score_or_terminal_confirmed"),
        0,
    )
    terminal_predicate_probe_ready_count = _as_int(
        readiness_state_counts.get(
            "objective_target_reached_terminal_predicate_missing"
        ),
        0,
    )
    continue_required_count = _as_int(
        readiness_state_counts.get("objective_target_closer_continue_required"),
        0,
    )
    return {
        "schema_version": "eca_score_relation_terminal_readiness_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "row_count": int(row_count),
        "score_delta_sum": int(score_delta_sum),
        "score_or_terminal_confirmed_count": int(score_or_terminal_confirmed_count),
        "terminal_predicate_probe_ready_count": int(
            terminal_predicate_probe_ready_count
        ),
        "continue_required_count": int(continue_required_count),
        "readiness_state_counts": dict(sorted(readiness_state_counts.items())),
        "candidate_relation_counts": dict(sorted(candidate_relation_counts.items())),
        "by_candidate_relation": {
            relation: dict(sorted(stats.items()))
            for relation, stats in sorted(by_candidate_relation.items())
        },
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "score_or_terminal_confirmed_count > 0",
            "allowed_connection": "diagnostic_to_terminal_predicate_probe_design_only",
            "action_generation_allowed": False,
        },
    }


def _score_relation_bounded_repeat_continuation_outcome_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    continuation_state_counts: Counter[str] = Counter()
    closure_result_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    continuation_count = 0
    route_refit_count = 0
    objective_progress_count = 0
    continuation_score_delta_sum = 0
    closure_count = 0
    closure_score_delta_sum = 0
    closure_goal_bearing_count = 0
    closure_action_mode_counts: Counter[str] = Counter()
    target_click_preferred_count = 0
    post_transition_count = 0
    post_transition_score_delta_sum = 0
    post_transition_goal_bearing_count = 0
    post_transition_action_source_counts: Counter[str] = Counter()
    post_transition_state_counts: Counter[str] = Counter()

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            if (
                bool(unsolved.get("continuation"))
                and str(unsolved.get("candidate_kind", "") or "")
                == "score_relation_bounded_repeat_probe"
            ):
                score_delta = _as_int(outcome.get("score_delta"), 0)
                objective_progress = bool(outcome.get("objective_progress"))
                proximity_delta = _as_float(
                    outcome.get("objective_proximity_delta"),
                    0.0,
                )
                route_refit = bool(unsolved.get("route_refit"))
                if score_delta > 0 or bool(outcome.get("goal_bearing")):
                    state = "score_or_terminal_confirmed"
                elif objective_progress:
                    state = "objective_progress_continued"
                elif proximity_delta < 0.0:
                    state = "objective_regressed_after_continuation"
                elif bool(outcome.get("structural_progress")):
                    state = "visible_effect_after_continuation"
                elif bool(outcome.get("no_op")):
                    state = "noop_after_continuation"
                else:
                    state = "unknown_after_continuation"

                continuation_count += 1
                route_refit_count += 1 if route_refit else 0
                objective_progress_count += 1 if objective_progress else 0
                continuation_score_delta_sum += score_delta
                continuation_state_counts[state] += 1
                by_game[game_id]["continuation_count"] += 1
                by_game[game_id][f"continuation_state:{state}"] += 1
                by_game[game_id]["continuation_score_delta_sum"] += score_delta
                if route_refit:
                    by_game[game_id]["route_refit_count"] += 1
                if len(samples) < sample_limit:
                    samples.append(
                        {
                            "game_id": game_id,
                            "variant": str(row.get("variant", "") or "unknown"),
                            "step": _as_int(item.get("step"), 0),
                            "phase": str(item.get("phase", "") or ""),
                            "action": str(item.get("action", "") or ""),
                            "event": "continuation",
                            "state": state,
                            "selected_action": str(
                                unsolved.get("selected_action", "") or ""
                            ),
                            "target_site": list(unsolved.get("target_site") or []),
                            "continuation_trigger": str(
                                unsolved.get("continuation_trigger", "") or ""
                            ),
                            "route_refit": route_refit,
                            "route_refit_source": str(
                                unsolved.get("route_refit_source", "") or ""
                            ),
                            "score_delta": score_delta,
                            "objective_progress": objective_progress,
                            "objective_proximity_delta": round(
                                float(proximity_delta),
                                6,
                            ),
                            "objective_distance_before": round(
                                _as_float(outcome.get("objective_distance_before"), 0.0),
                                6,
                            ),
                            "objective_distance_after": round(
                                _as_float(outcome.get("objective_distance_after"), 0.0),
                                6,
                            ),
                        }
                    )

            if (
                bool(unsolved.get("closure"))
                and str(unsolved.get("closure_parent_candidate_kind", "") or "")
                == "score_relation_bounded_repeat_probe"
            ):
                closure_result = unsolved.get("closure_result")
                closure_result = closure_result if isinstance(closure_result, Mapping) else {}
                result = str(closure_result.get("result", "") or "unknown")
                action_mode = str(unsolved.get("closure_action_mode", "") or "unknown")
                score_delta = _as_int(outcome.get("score_delta"), 0)
                goal_bearing = bool(
                    outcome.get("goal_bearing") or closure_result.get("goal_bearing")
                )
                target_click_preferred = bool(
                    unsolved.get("score_relation_target_click_preferred")
                )
                closure_count += 1
                closure_score_delta_sum += score_delta
                closure_goal_bearing_count += 1 if goal_bearing else 0
                target_click_preferred_count += 1 if target_click_preferred else 0
                closure_result_counts[result] += 1
                closure_action_mode_counts[action_mode] += 1
                by_game[game_id]["closure_count"] += 1
                by_game[game_id][f"closure_result:{result}"] += 1
                by_game[game_id][f"closure_action_mode:{action_mode}"] += 1
                by_game[game_id]["closure_score_delta_sum"] += score_delta
                if target_click_preferred:
                    by_game[game_id]["target_click_preferred_count"] += 1
                if len(samples) < sample_limit:
                    samples.append(
                        {
                            "game_id": game_id,
                            "variant": str(row.get("variant", "") or "unknown"),
                            "step": _as_int(item.get("step"), 0),
                            "phase": str(item.get("phase", "") or ""),
                            "action": str(item.get("action", "") or ""),
                            "event": "closure",
                            "closure_result": result,
                            "closure_action_mode": str(
                                unsolved.get("closure_action_mode", "") or ""
                            ),
                            "target_click_preferred": target_click_preferred,
                            "target_site": list(unsolved.get("target_site") or []),
                            "score_delta": score_delta,
                            "goal_bearing": goal_bearing,
                            "effect_type": str(outcome.get("effect_type", "") or ""),
                        }
                    )

            if (
                bool(unsolved.get("post_transition_replanner"))
                and str(unsolved.get("parent_candidate_kind", "") or "")
                == "score_relation_bounded_repeat_probe"
            ):
                score_delta = _as_int(outcome.get("score_delta"), 0)
                goal_bearing = bool(outcome.get("goal_bearing") or score_delta > 0)
                action_source = str(
                    unsolved.get("post_transition_action_source", "") or "unknown"
                )
                if goal_bearing:
                    state = "score_or_terminal_confirmed"
                elif bool(outcome.get("objective_progress")):
                    state = "objective_progress_after_post_transition"
                elif bool(outcome.get("structural_progress")):
                    state = "visible_effect_after_post_transition"
                elif bool(outcome.get("no_op")):
                    state = "noop_after_post_transition"
                else:
                    state = "no_score_after_post_transition"

                post_transition_count += 1
                post_transition_score_delta_sum += score_delta
                post_transition_goal_bearing_count += 1 if goal_bearing else 0
                post_transition_action_source_counts[action_source] += 1
                post_transition_state_counts[state] += 1
                by_game[game_id]["post_transition_count"] += 1
                by_game[game_id]["post_transition_score_delta_sum"] += score_delta
                by_game[game_id][f"post_transition_action_source:{action_source}"] += 1
                by_game[game_id][f"post_transition_state:{state}"] += 1
                if len(samples) < sample_limit:
                    samples.append(
                        {
                            "game_id": game_id,
                            "variant": str(row.get("variant", "") or "unknown"),
                            "step": _as_int(item.get("step"), 0),
                            "phase": str(item.get("phase", "") or ""),
                            "action": str(item.get("action", "") or ""),
                            "event": "post_transition",
                            "state": state,
                            "selected_action": str(
                                unsolved.get("selected_action", "") or ""
                            ),
                            "post_transition_action_source": action_source,
                            "target_site": list(unsolved.get("target_site") or []),
                            "score_delta": score_delta,
                            "goal_bearing": goal_bearing,
                            "effect_type": str(outcome.get("effect_type", "") or ""),
                            "effect_supported": bool(outcome.get("effect_supported")),
                        }
                    )

    return {
        "schema_version": "eca_score_relation_bounded_repeat_continuation_outcome_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "continuation_count": int(continuation_count),
        "route_refit_count": int(route_refit_count),
        "objective_progress_count": int(objective_progress_count),
        "continuation_score_delta_sum": int(continuation_score_delta_sum),
        "continuation_state_counts": dict(sorted(continuation_state_counts.items())),
        "closure_count": int(closure_count),
        "closure_score_delta_sum": int(closure_score_delta_sum),
        "closure_goal_bearing_count": int(closure_goal_bearing_count),
        "target_click_preferred_count": int(target_click_preferred_count),
        "closure_action_mode_counts": dict(sorted(closure_action_mode_counts.items())),
        "closure_result_counts": dict(sorted(closure_result_counts.items())),
        "post_transition_count": int(post_transition_count),
        "post_transition_score_delta_sum": int(post_transition_score_delta_sum),
        "post_transition_goal_bearing_count": int(post_transition_goal_bearing_count),
        "post_transition_action_source_counts": dict(
            sorted(post_transition_action_source_counts.items())
        ),
        "post_transition_state_counts": dict(sorted(post_transition_state_counts.items())),
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "closure_goal_bearing_count > 0 or continuation_score_delta_sum > 0",
            "allowed_connection": "diagnostic_to_closure_action_design_only",
            "action_generation_allowed": False,
        },
    }


def _score_relation_post_closure_transition_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    closure_state_counts: Counter[str] = Counter()
    closure_result_counts: Counter[str] = Counter()
    closure_action_mode_counts: Counter[str] = Counter()
    post_transition_state_counts: Counter[str] = Counter()
    post_transition_action_source_counts: Counter[str] = Counter()
    required_sensor_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    closure_transition_count = 0
    closure_visible_no_goal_count = 0
    closure_score_delta_sum = 0
    closure_goal_bearing_count = 0
    post_transition_count = 0
    post_transition_visible_no_goal_count = 0
    post_transition_score_delta_sum = 0
    post_transition_goal_bearing_count = 0

    def _score_delta_from(
        outcome: Mapping[str, Any],
        fallback: Mapping[str, Any],
    ) -> int:
        if "score_delta" in outcome:
            return _as_int(outcome.get("score_delta"), 0)
        return _as_int(fallback.get("score_delta"), 0)

    def _visible_transition(outcome: Mapping[str, Any]) -> bool:
        return bool(
            outcome.get("state_changed")
            or outcome.get("structural_progress")
            or _as_int(outcome.get("changed_pixel_count"), 0) > 0
        )

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}

            if (
                bool(unsolved.get("closure"))
                and str(unsolved.get("closure_parent_candidate_kind", "") or "")
                == "score_relation_bounded_repeat_probe"
            ):
                closure_result = unsolved.get("closure_result")
                closure_result = closure_result if isinstance(closure_result, Mapping) else {}
                result = str(closure_result.get("result", "") or "unknown")
                action_mode = str(unsolved.get("closure_action_mode", "") or "unknown")
                score_delta = _score_delta_from(outcome, closure_result)
                goal_bearing = bool(
                    outcome.get("goal_bearing")
                    or closure_result.get("goal_bearing")
                    or score_delta > 0
                )
                state_transition_candidate = bool(
                    closure_result.get("state_transition_candidate")
                    or _visible_transition(outcome)
                    or result
                    in {
                        "objective_progress_no_goal",
                        "state_changed_no_goal",
                        "visible_effect_no_goal",
                        "effect_supported_no_goal",
                    }
                )
                if goal_bearing:
                    state = "score_or_terminal_confirmed"
                elif bool(outcome.get("objective_progress")) or result == "objective_progress_no_goal":
                    state = "post_closure_objective_progress_no_goal"
                    required_sensor_counts["post_closure_state_delta_interpreter_required"] += 1
                    required_sensor_counts["terminal_predicate_unresolved"] += 1
                elif bool(outcome.get("state_changed")) or result == "state_changed_no_goal":
                    state = "post_closure_state_changed_no_goal"
                    required_sensor_counts["post_closure_state_delta_interpreter_required"] += 1
                    required_sensor_counts["changed_region_followup_relation_required"] += 1
                    required_sensor_counts["terminal_predicate_unresolved"] += 1
                elif state_transition_candidate:
                    state = "post_closure_visible_transition_no_goal"
                    required_sensor_counts["post_closure_state_delta_interpreter_required"] += 1
                    required_sensor_counts["changed_region_followup_relation_required"] += 1
                    required_sensor_counts["terminal_predicate_unresolved"] += 1
                elif bool(outcome.get("no_op")) or result == "noop_no_goal":
                    state = "post_closure_noop_or_dead_site"
                    required_sensor_counts["negative_evidence_posterior_required"] += 1
                else:
                    state = "post_closure_unclassified_no_goal"
                    required_sensor_counts["post_closure_state_delta_interpreter_required"] += 1

                closure_transition_count += 1 if state_transition_candidate else 0
                closure_visible_no_goal_count += (
                    1
                    if state
                    in {
                        "post_closure_objective_progress_no_goal",
                        "post_closure_state_changed_no_goal",
                        "post_closure_visible_transition_no_goal",
                    }
                    else 0
                )
                closure_score_delta_sum += score_delta
                closure_goal_bearing_count += 1 if goal_bearing else 0
                closure_state_counts[state] += 1
                closure_result_counts[result] += 1
                closure_action_mode_counts[action_mode] += 1
                by_game[game_id]["closure_event_count"] += 1
                by_game[game_id]["closure_transition_count"] += (
                    1 if state_transition_candidate else 0
                )
                by_game[game_id][f"closure_state:{state}"] += 1
                by_game[game_id][f"closure_result:{result}"] += 1
                by_game[game_id][f"closure_action_mode:{action_mode}"] += 1
                if len(samples) < sample_limit:
                    samples.append(
                        {
                            "game_id": game_id,
                            "variant": str(row.get("variant", "") or "unknown"),
                            "step": _as_int(item.get("step"), 0),
                            "phase": str(item.get("phase", "") or ""),
                            "event": "closure",
                            "state": state,
                            "closure_result": result,
                            "closure_action_mode": action_mode,
                            "state_transition_candidate": state_transition_candidate,
                            "selected_action": str(
                                unsolved.get("selected_action")
                                or closure_result.get("action")
                                or item.get("action")
                                or ""
                            ),
                            "target_site": list(unsolved.get("target_site") or []),
                            "score_delta": score_delta,
                            "goal_bearing": goal_bearing,
                            "effect_type": str(outcome.get("effect_type", "") or ""),
                            "changed_pixel_count": _as_int(
                                outcome.get("changed_pixel_count"),
                                0,
                            ),
                        }
                    )

            if (
                bool(unsolved.get("post_transition_replanner"))
                and str(unsolved.get("parent_candidate_kind", "") or "")
                == "score_relation_bounded_repeat_probe"
            ):
                closure_result = unsolved.get("closure_result")
                closure_result = closure_result if isinstance(closure_result, Mapping) else {}
                score_delta = _score_delta_from(outcome, closure_result)
                goal_bearing = bool(outcome.get("goal_bearing") or score_delta > 0)
                action_source = str(
                    unsolved.get("post_transition_action_source", "") or "unknown"
                )
                visible_transition = _visible_transition(outcome)
                if goal_bearing:
                    state = "score_retest_confirmed"
                elif bool(outcome.get("objective_progress")):
                    state = "score_retest_objective_progress_no_goal"
                    required_sensor_counts["post_transition_visible_effect_interpreter_required"] += 1
                    required_sensor_counts["terminal_predicate_unresolved"] += 1
                elif visible_transition:
                    state = "score_retest_visible_effect_no_goal"
                    required_sensor_counts["post_transition_visible_effect_interpreter_required"] += 1
                    required_sensor_counts["changed_region_followup_relation_required"] += 1
                    required_sensor_counts["terminal_predicate_unresolved"] += 1
                elif bool(outcome.get("no_op")):
                    state = "score_retest_noop_no_goal"
                    required_sensor_counts["negative_evidence_posterior_required"] += 1
                else:
                    state = "score_retest_unclassified_no_goal"
                    required_sensor_counts["post_transition_visible_effect_interpreter_required"] += 1

                post_transition_count += 1
                post_transition_visible_no_goal_count += 1 if visible_transition and not goal_bearing else 0
                post_transition_score_delta_sum += score_delta
                post_transition_goal_bearing_count += 1 if goal_bearing else 0
                post_transition_state_counts[state] += 1
                post_transition_action_source_counts[action_source] += 1
                by_game[game_id]["post_transition_count"] += 1
                by_game[game_id]["post_transition_score_delta_sum"] += score_delta
                by_game[game_id][f"post_transition_state:{state}"] += 1
                by_game[game_id][f"post_transition_action_source:{action_source}"] += 1
                if len(samples) < sample_limit:
                    samples.append(
                        {
                            "game_id": game_id,
                            "variant": str(row.get("variant", "") or "unknown"),
                            "step": _as_int(item.get("step"), 0),
                            "phase": str(item.get("phase", "") or ""),
                            "event": "post_transition",
                            "state": state,
                            "selected_action": str(
                                unsolved.get("selected_action") or item.get("action") or ""
                            ),
                            "post_transition_action_source": action_source,
                            "target_site": list(unsolved.get("target_site") or []),
                            "score_delta": score_delta,
                            "goal_bearing": goal_bearing,
                            "effect_type": str(outcome.get("effect_type", "") or ""),
                            "changed_pixel_count": _as_int(
                                outcome.get("changed_pixel_count"),
                                0,
                            ),
                            "closure_result": str(
                                closure_result.get("result", "") or ""
                            ),
                        }
                    )

    return {
        "schema_version": "eca_score_relation_post_closure_transition_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "closure_transition_count": int(closure_transition_count),
        "closure_visible_no_goal_count": int(closure_visible_no_goal_count),
        "closure_score_delta_sum": int(closure_score_delta_sum),
        "closure_goal_bearing_count": int(closure_goal_bearing_count),
        "closure_state_counts": dict(sorted(closure_state_counts.items())),
        "closure_result_counts": dict(sorted(closure_result_counts.items())),
        "closure_action_mode_counts": dict(sorted(closure_action_mode_counts.items())),
        "post_transition_count": int(post_transition_count),
        "post_transition_visible_no_goal_count": int(
            post_transition_visible_no_goal_count
        ),
        "post_transition_score_delta_sum": int(post_transition_score_delta_sum),
        "post_transition_goal_bearing_count": int(post_transition_goal_bearing_count),
        "post_transition_state_counts": dict(sorted(post_transition_state_counts.items())),
        "post_transition_action_source_counts": dict(
            sorted(post_transition_action_source_counts.items())
        ),
        "required_sensor_counts": dict(sorted(required_sensor_counts.items())),
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "score_retest_confirmed or score_or_terminal_confirmed evidence",
            "allowed_connection": "diagnostic_to_post_closure_state_transition_interpreter_only",
            "action_generation_allowed": False,
        },
    }


def _score_relation_post_closure_state_delta_interpreter_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    state_delta_counts: Counter[str] = Counter()
    required_followup_counts: Counter[str] = Counter()
    action_source_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    row_count = 0
    interpreted_count = 0
    target_site_reached_count = 0
    visible_no_goal_count = 0
    post_transition_retest_count = 0
    score_delta_sum = 0
    goal_bearing_count = 0

    def _motion(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        value = payload.get(key)
        return value if isinstance(value, Mapping) else {}

    def _distance_after(motion: Mapping[str, Any], outcome: Mapping[str, Any]) -> float:
        if "target_distance_after" in motion:
            return _as_float(motion.get("target_distance_after"), 0.0)
        return _as_float(outcome.get("objective_distance_after"), 0.0)

    def _compact_motion(motion: Mapping[str, Any]) -> dict[str, Any]:
        if not motion:
            return {}
        return {
            key: motion.get(key)
            for key in (
                "color_key",
                "moved_pixel_count",
                "pre_centroid",
                "post_centroid",
                "delta_row",
                "delta_col",
                "target_distance_delta",
                "target_distance_before",
                "target_distance_after",
                "target_progress",
                "action_aligned",
            )
            if key in motion
        }

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            closure_result = unsolved.get("closure_result")
            closure_result = closure_result if isinstance(closure_result, Mapping) else {}
            is_closure = bool(unsolved.get("closure")) and str(
                unsolved.get("closure_parent_candidate_kind", "") or ""
            ) == "score_relation_bounded_repeat_probe"
            is_post_transition = bool(unsolved.get("post_transition_replanner")) and str(
                unsolved.get("parent_candidate_kind", "") or ""
            ) == "score_relation_bounded_repeat_probe"
            if not is_closure and not is_post_transition:
                continue

            event = "closure" if is_closure else "post_transition"
            result = str(closure_result.get("result", "") or "")
            action_mode = str(unsolved.get("closure_action_mode", "") or "")
            action_source = str(
                unsolved.get("post_transition_action_source", "")
                or action_mode
                or "unknown"
            )
            current_motion = _motion(outcome, "dominant_motion")
            parent_motion = _motion(unsolved, "closure_parent_dominant_motion")
            if not parent_motion:
                parent_motion = _motion(closure_result, "parent_dominant_motion")
            score_delta = _as_int(outcome.get("score_delta"), _as_int(closure_result.get("score_delta"), 0))
            goal_bearing = bool(
                outcome.get("goal_bearing")
                or closure_result.get("goal_bearing")
                or score_delta > 0
            )
            changed_pixels = _as_int(outcome.get("changed_pixel_count"), 0)
            visible_transition = bool(
                outcome.get("state_changed")
                or outcome.get("structural_progress")
                or changed_pixels > 0
                or result
                in {
                    "objective_progress_no_goal",
                    "state_changed_no_goal",
                    "visible_effect_no_goal",
                    "effect_supported_no_goal",
                }
            )
            if not visible_transition and not goal_bearing:
                continue

            parent_distance_after = _distance_after(parent_motion, outcome)
            current_distance_delta = _as_float(
                current_motion.get("target_distance_delta"),
                _as_float(outcome.get("objective_proximity_delta"), 0.0),
            )
            parent_reached_target = bool(
                parent_motion and 0.0 < parent_distance_after <= 1.0
            )
            if goal_bearing:
                state = "score_or_terminal_confirmed"
            elif event == "closure" and parent_reached_target and action_mode == "target_click":
                state = "target_click_after_reached_site_visible_no_goal"
                required_followup_counts["terminal_predicate_sensor_required"] += 1
                required_followup_counts["state_transition_semantics_required"] += 1
            elif event == "closure" and visible_transition:
                state = "post_closure_visible_delta_uninterpreted"
                required_followup_counts["state_transition_semantics_required"] += 1
                required_followup_counts["changed_region_followup_relation_required"] += 1
            elif event == "post_transition" and current_distance_delta > 0.0:
                state = "post_transition_moves_toward_target_no_goal"
                required_followup_counts["terminal_predicate_sensor_required"] += 1
                required_followup_counts["post_transition_visible_effect_interpreter_required"] += 1
            elif event == "post_transition" and visible_transition:
                state = "post_transition_visible_delta_uninterpreted"
                required_followup_counts["post_transition_visible_effect_interpreter_required"] += 1
                required_followup_counts["changed_region_followup_relation_required"] += 1
            else:
                state = "state_delta_unclassified"
                required_followup_counts["state_transition_semantics_required"] += 1

            row_count += 1
            interpreted_count += 1 if state != "state_delta_unclassified" else 0
            target_site_reached_count += 1 if parent_reached_target else 0
            visible_no_goal_count += 1 if visible_transition and not goal_bearing else 0
            post_transition_retest_count += 1 if is_post_transition else 0
            score_delta_sum += score_delta
            goal_bearing_count += 1 if goal_bearing else 0
            state_delta_counts[state] += 1
            action_source_counts[action_source] += 1
            by_game[game_id]["row_count"] += 1
            by_game[game_id][f"state_delta:{state}"] += 1
            by_game[game_id][f"action_source:{action_source}"] += 1
            by_game[game_id]["visible_no_goal_count"] += (
                1 if visible_transition and not goal_bearing else 0
            )
            if parent_reached_target:
                by_game[game_id]["target_site_reached_count"] += 1
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), 0),
                        "phase": str(item.get("phase", "") or ""),
                        "event": event,
                        "state_delta": state,
                        "selected_action": str(
                            unsolved.get("selected_action") or item.get("action") or ""
                        ),
                        "action_source": action_source,
                        "target_site": list(unsolved.get("target_site") or []),
                        "closure_result": result,
                        "score_delta": score_delta,
                        "goal_bearing": goal_bearing,
                        "visible_transition": visible_transition,
                        "changed_pixel_count": changed_pixels,
                        "parent_reached_target": parent_reached_target,
                        "parent_distance_after": round(parent_distance_after, 6),
                        "current_distance_delta": round(current_distance_delta, 6),
                        "current_motion": _compact_motion(current_motion),
                        "parent_motion": _compact_motion(parent_motion),
                    }
                )

    return {
        "schema_version": "eca_score_relation_post_closure_state_delta_interpreter_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "row_count": int(row_count),
        "interpreted_count": int(interpreted_count),
        "target_site_reached_count": int(target_site_reached_count),
        "visible_no_goal_count": int(visible_no_goal_count),
        "post_transition_retest_count": int(post_transition_retest_count),
        "score_delta_sum": int(score_delta_sum),
        "goal_bearing_count": int(goal_bearing_count),
        "state_delta_counts": dict(sorted(state_delta_counts.items())),
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "required_followup_counts": dict(sorted(required_followup_counts.items())),
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "terminal_predicate_sensor or changed-region follow-up relation must confirm score-bearing semantics",
            "allowed_connection": "diagnostic_to_changed_region_followup_relation_design_only",
            "action_generation_allowed": False,
        },
    }


def _score_relation_changed_region_followup_relation_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    relation_counts: Counter[str] = Counter()
    required_followup_counts: Counter[str] = Counter()
    action_source_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    row_count = 0
    changed_region_available_count = 0
    localized_followup_candidate_count = 0
    target_site_inside_changed_region_count = 0
    score_delta_sum = 0
    goal_bearing_count = 0

    def _changed_region(outcome: Mapping[str, Any]) -> Mapping[str, Any]:
        region = outcome.get("changed_region")
        if isinstance(region, Mapping):
            return region
        bbox = outcome.get("changed_bbox")
        centroid = outcome.get("changed_centroid")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return {
                "available": True,
                "reason": "changed_bbox_only",
                "changed_pixel_count": _as_int(outcome.get("changed_pixel_count"), 0),
                "bbox": list(bbox),
                "centroid": list(centroid) if isinstance(centroid, (list, tuple)) else [],
            }
        return {}

    def _bbox_contains_site(bbox: Any, site: Any) -> bool:
        if not (
            isinstance(bbox, (list, tuple))
            and len(bbox) >= 4
            and isinstance(site, (list, tuple))
            and len(site) >= 2
        ):
            return False
        row = _as_float(site[0], 0.0)
        col = _as_float(site[1], 0.0)
        return bool(
            _as_float(bbox[0], 0.0) <= row <= _as_float(bbox[2], 0.0)
            and _as_float(bbox[1], 0.0) <= col <= _as_float(bbox[3], 0.0)
        )

    def _centroid_distance(centroid: Any, site: Any) -> float:
        if not (
            isinstance(centroid, (list, tuple))
            and len(centroid) >= 2
            and isinstance(site, (list, tuple))
            and len(site) >= 2
        ):
            return 0.0
        return abs(_as_float(centroid[0], 0.0) - _as_float(site[0], 0.0)) + abs(
            _as_float(centroid[1], 0.0) - _as_float(site[1], 0.0)
        )

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            closure_result = unsolved.get("closure_result")
            closure_result = closure_result if isinstance(closure_result, Mapping) else {}
            is_closure = bool(unsolved.get("closure")) and str(
                unsolved.get("closure_parent_candidate_kind", "") or ""
            ) == "score_relation_bounded_repeat_probe"
            is_post_transition = bool(unsolved.get("post_transition_replanner")) and str(
                unsolved.get("parent_candidate_kind", "") or ""
            ) == "score_relation_bounded_repeat_probe"
            if not is_closure and not is_post_transition:
                continue

            score_delta = _as_int(outcome.get("score_delta"), _as_int(closure_result.get("score_delta"), 0))
            goal_bearing = bool(
                outcome.get("goal_bearing")
                or closure_result.get("goal_bearing")
                or score_delta > 0
            )
            changed_pixels = _as_int(outcome.get("changed_pixel_count"), 0)
            visible_no_goal = bool(changed_pixels > 0 and not goal_bearing)
            if not visible_no_goal:
                continue

            event = "closure" if is_closure else "post_transition"
            action_source = str(
                unsolved.get("post_transition_action_source", "")
                or unsolved.get("closure_action_mode", "")
                or "unknown"
            )
            target_site = list(unsolved.get("target_site") or [])
            region = _changed_region(outcome)
            region_available = bool(region.get("available"))
            bbox = list(region.get("bbox") or [])
            centroid = list(region.get("centroid") or [])
            target_inside = _bbox_contains_site(bbox, target_site)
            distance = _centroid_distance(centroid, target_site)
            if not region_available:
                relation = "changed_region_packet_missing_no_goal"
                required_followup_counts["changed_region_packet_required"] += 1
            elif target_inside:
                relation = "target_site_inside_changed_region_no_goal"
                required_followup_counts["terminal_predicate_sensor_required"] += 1
                required_followup_counts["changed_region_semantics_required"] += 1
            elif centroid and target_site and distance <= 2.0:
                relation = "changed_region_adjacent_to_target_no_goal"
                required_followup_counts["changed_region_followup_relation_required"] += 1
                required_followup_counts["terminal_predicate_sensor_required"] += 1
            else:
                relation = "changed_region_remote_followup_required"
                required_followup_counts["changed_region_followup_relation_required"] += 1
            if event == "post_transition":
                required_followup_counts[
                    "post_transition_visible_effect_interpreter_required"
                ] += 1

            row_count += 1
            changed_region_available_count += 1 if region_available else 0
            localized_followup_candidate_count += 1 if centroid else 0
            target_site_inside_changed_region_count += 1 if target_inside else 0
            score_delta_sum += score_delta
            goal_bearing_count += 1 if goal_bearing else 0
            relation_counts[relation] += 1
            action_source_counts[action_source] += 1
            by_game[game_id]["row_count"] += 1
            by_game[game_id][f"relation:{relation}"] += 1
            by_game[game_id][f"action_source:{action_source}"] += 1
            if region_available:
                by_game[game_id]["changed_region_available_count"] += 1
            if centroid:
                by_game[game_id]["localized_followup_candidate_count"] += 1
            if target_inside:
                by_game[game_id]["target_site_inside_changed_region_count"] += 1
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), 0),
                        "phase": str(item.get("phase", "") or ""),
                        "event": event,
                        "relation": relation,
                        "selected_action": str(
                            unsolved.get("selected_action") or item.get("action") or ""
                        ),
                        "action_source": action_source,
                        "target_site": target_site,
                        "score_delta": score_delta,
                        "goal_bearing": goal_bearing,
                        "changed_pixel_count": changed_pixels,
                        "changed_region_available": region_available,
                        "changed_bbox": bbox,
                        "changed_centroid": centroid,
                        "target_site_inside_changed_region": target_inside,
                        "changed_centroid_target_distance": round(distance, 6),
                        "changed_region_reason": str(region.get("reason", "") or ""),
                        "pre_colors_top": list(region.get("pre_colors_top") or []),
                        "post_colors_top": list(region.get("post_colors_top") or []),
                    }
                )

    return {
        "schema_version": "eca_score_relation_changed_region_followup_relation_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "row_count": int(row_count),
        "changed_region_available_count": int(changed_region_available_count),
        "localized_followup_candidate_count": int(localized_followup_candidate_count),
        "target_site_inside_changed_region_count": int(
            target_site_inside_changed_region_count
        ),
        "score_delta_sum": int(score_delta_sum),
        "goal_bearing_count": int(goal_bearing_count),
        "relation_counts": dict(sorted(relation_counts.items())),
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "required_followup_counts": dict(sorted(required_followup_counts.items())),
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "changed-region semantics or terminal predicate must be confirmed by score-bearing evidence",
            "allowed_connection": "diagnostic_to_terminal_predicate_or_changed_region_probe_design_only",
            "action_generation_allowed": False,
        },
    }


def _score_relation_remote_changed_region_semantics_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    semantics_counts: Counter[str] = Counter()
    action_source_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    required_followup_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    row_count = 0
    remote_changed_region_count = 0
    border_region_count = 0
    compact_region_count = 0
    repeated_transition_count = 0
    status_indicator_candidate_count = 0
    score_delta_sum = 0
    goal_bearing_count = 0

    def _changed_region(outcome: Mapping[str, Any]) -> Mapping[str, Any]:
        region = outcome.get("changed_region")
        if isinstance(region, Mapping):
            return region
        bbox = outcome.get("changed_bbox")
        centroid = outcome.get("changed_centroid")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return {
                "available": True,
                "changed_pixel_count": _as_int(outcome.get("changed_pixel_count"), 0),
                "bbox": list(bbox),
                "centroid": list(centroid) if isinstance(centroid, (list, tuple)) else [],
            }
        return {}

    def _bbox_contains_site(bbox: Any, site: Any) -> bool:
        if not (
            isinstance(bbox, (list, tuple))
            and len(bbox) >= 4
            and isinstance(site, (list, tuple))
            and len(site) >= 2
        ):
            return False
        row = _as_float(site[0], 0.0)
        col = _as_float(site[1], 0.0)
        return bool(
            _as_float(bbox[0], 0.0) <= row <= _as_float(bbox[2], 0.0)
            and _as_float(bbox[1], 0.0) <= col <= _as_float(bbox[3], 0.0)
        )

    def _top_color_key(rows: Any) -> str:
        if not isinstance(rows, list) or not rows:
            return "unknown"
        first = rows[0]
        if not isinstance(first, Mapping):
            return "unknown"
        return str(first.get("color_key", "") or "unknown")

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            closure_result = unsolved.get("closure_result")
            closure_result = closure_result if isinstance(closure_result, Mapping) else {}
            is_closure = bool(unsolved.get("closure")) and str(
                unsolved.get("closure_parent_candidate_kind", "") or ""
            ) == "score_relation_bounded_repeat_probe"
            is_post_transition = bool(unsolved.get("post_transition_replanner")) and str(
                unsolved.get("parent_candidate_kind", "") or ""
            ) == "score_relation_bounded_repeat_probe"
            if not is_closure and not is_post_transition:
                continue

            score_delta = _as_int(outcome.get("score_delta"), _as_int(closure_result.get("score_delta"), 0))
            goal_bearing = bool(
                outcome.get("goal_bearing")
                or closure_result.get("goal_bearing")
                or score_delta > 0
            )
            changed_pixels = _as_int(outcome.get("changed_pixel_count"), 0)
            if changed_pixels <= 0 or goal_bearing:
                continue
            region = _changed_region(outcome)
            if not bool(region.get("available")):
                continue
            bbox = list(region.get("bbox") or [])
            if len(bbox) < 4:
                continue
            target_site = list(unsolved.get("target_site") or [])
            target_inside = _bbox_contains_site(bbox, target_site)
            min_row = _as_int(bbox[0], 0)
            min_col = _as_int(bbox[1], 0)
            max_row = _as_int(bbox[2], 0)
            max_col = _as_int(bbox[3], 0)
            height = max(0, max_row - min_row + 1)
            width = max(0, max_col - min_col + 1)
            area = max(1, height * width)
            compact_region = bool(area <= 6 and changed_pixels <= 6)
            border_region = bool(min_row == 0 or min_col == 0)
            remote_region = bool(not target_inside)
            pre_color = _top_color_key(region.get("pre_colors_top"))
            post_color = _top_color_key(region.get("post_colors_top"))
            transition_key = f"{pre_color}->{post_color}"
            transition_counts[transition_key] += 1
            repeated_transition = bool(transition_counts[transition_key] >= 2)
            if remote_region and border_region and compact_region:
                semantics = "remote_border_status_indicator_candidate_no_goal"
                status_indicator_candidate = True
                required_followup_counts["terminal_or_status_predicate_sensor_required"] += 1
                required_followup_counts["remote_changed_region_semantics_required"] += 1
            elif remote_region and compact_region:
                semantics = "remote_compact_indicator_candidate_no_goal"
                status_indicator_candidate = True
                required_followup_counts["remote_changed_region_semantics_required"] += 1
            elif remote_region:
                semantics = "remote_changed_region_unclassified_no_goal"
                status_indicator_candidate = False
                required_followup_counts["remote_changed_region_semantics_required"] += 1
            else:
                semantics = "target_region_changed_no_goal"
                status_indicator_candidate = False
                required_followup_counts["terminal_predicate_sensor_required"] += 1

            row_count += 1
            remote_changed_region_count += 1 if remote_region else 0
            border_region_count += 1 if border_region else 0
            compact_region_count += 1 if compact_region else 0
            repeated_transition_count += 1 if repeated_transition else 0
            status_indicator_candidate_count += 1 if status_indicator_candidate else 0
            score_delta_sum += score_delta
            goal_bearing_count += 1 if goal_bearing else 0
            semantics_counts[semantics] += 1
            action_source = str(
                unsolved.get("post_transition_action_source", "")
                or unsolved.get("closure_action_mode", "")
                or "unknown"
            )
            action_source_counts[action_source] += 1
            by_game[game_id]["row_count"] += 1
            by_game[game_id][f"semantics:{semantics}"] += 1
            by_game[game_id][f"transition:{transition_key}"] += 1
            if remote_region:
                by_game[game_id]["remote_changed_region_count"] += 1
            if status_indicator_candidate:
                by_game[game_id]["status_indicator_candidate_count"] += 1
            if repeated_transition:
                by_game[game_id]["repeated_transition_count"] += 1
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), 0),
                        "phase": str(item.get("phase", "") or ""),
                        "event": "closure" if is_closure else "post_transition",
                        "semantics": semantics,
                        "selected_action": str(
                            unsolved.get("selected_action") or item.get("action") or ""
                        ),
                        "action_source": action_source,
                        "target_site": target_site,
                        "changed_bbox": bbox,
                        "changed_centroid": list(region.get("centroid") or []),
                        "changed_pixel_count": changed_pixels,
                        "pre_color_key": pre_color,
                        "post_color_key": post_color,
                        "transition_key": transition_key,
                        "remote_region": remote_region,
                        "border_region": border_region,
                        "compact_region": compact_region,
                        "status_indicator_candidate": status_indicator_candidate,
                        "repeated_transition": repeated_transition,
                        "score_delta": score_delta,
                        "goal_bearing": goal_bearing,
                    }
                )

    return {
        "schema_version": "eca_score_relation_remote_changed_region_semantics_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "row_count": int(row_count),
        "remote_changed_region_count": int(remote_changed_region_count),
        "border_region_count": int(border_region_count),
        "compact_region_count": int(compact_region_count),
        "repeated_transition_count": int(repeated_transition_count),
        "status_indicator_candidate_count": int(status_indicator_candidate_count),
        "score_delta_sum": int(score_delta_sum),
        "goal_bearing_count": int(goal_bearing_count),
        "semantics_counts": dict(sorted(semantics_counts.items())),
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
        "required_followup_counts": dict(sorted(required_followup_counts.items())),
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "remote status-indicator semantics must be tied to score or terminal evidence",
            "allowed_connection": "diagnostic_to_terminal_status_predicate_design_only",
            "action_generation_allowed": False,
        },
    }


def _score_relation_terminal_status_predicate_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    predicate_state_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    action_source_counts: Counter[str] = Counter()
    required_followup_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    row_count = 0
    status_predicate_candidate_count = 0
    remote_status_candidate_count = 0
    repeated_status_transition_count = 0
    terminal_or_score_confirmed_count = 0
    immediate_terminal_not_confirmed_count = 0
    score_delta_sum = 0
    goal_bearing_count = 0

    def _changed_region(outcome: Mapping[str, Any]) -> Mapping[str, Any]:
        region = outcome.get("changed_region")
        if isinstance(region, Mapping):
            return region
        bbox = outcome.get("changed_bbox")
        centroid = outcome.get("changed_centroid")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return {
                "available": True,
                "changed_pixel_count": _as_int(outcome.get("changed_pixel_count"), 0),
                "bbox": list(bbox),
                "centroid": list(centroid) if isinstance(centroid, (list, tuple)) else [],
            }
        return {}

    def _bbox_contains_site(bbox: Any, site: Any) -> bool:
        if not (
            isinstance(bbox, (list, tuple))
            and len(bbox) >= 4
            and isinstance(site, (list, tuple))
            and len(site) >= 2
        ):
            return False
        row = _as_float(site[0], 0.0)
        col = _as_float(site[1], 0.0)
        return bool(
            _as_float(bbox[0], 0.0) <= row <= _as_float(bbox[2], 0.0)
            and _as_float(bbox[1], 0.0) <= col <= _as_float(bbox[3], 0.0)
        )

    def _top_color_key(rows: Any) -> str:
        if not isinstance(rows, list) or not rows:
            return "unknown"
        first = rows[0]
        if not isinstance(first, Mapping):
            return "unknown"
        return str(first.get("color_key", "") or "unknown")

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            closure_result = unsolved.get("closure_result")
            closure_result = closure_result if isinstance(closure_result, Mapping) else {}
            is_closure = bool(unsolved.get("closure")) and str(
                unsolved.get("closure_parent_candidate_kind", "") or ""
            ) == "score_relation_bounded_repeat_probe"
            is_post_transition = bool(unsolved.get("post_transition_replanner")) and str(
                unsolved.get("parent_candidate_kind", "") or ""
            ) == "score_relation_bounded_repeat_probe"
            if not is_closure and not is_post_transition:
                continue

            region = _changed_region(outcome)
            if not bool(region.get("available")):
                continue
            bbox = list(region.get("bbox") or [])
            if len(bbox) < 4:
                continue
            changed_pixels = _as_int(
                region.get("changed_pixel_count"),
                _as_int(outcome.get("changed_pixel_count"), 0),
            )
            if changed_pixels <= 0:
                continue
            target_site = list(unsolved.get("target_site") or [])
            target_inside = _bbox_contains_site(bbox, target_site)
            min_row = _as_int(bbox[0], 0)
            min_col = _as_int(bbox[1], 0)
            max_row = _as_int(bbox[2], 0)
            max_col = _as_int(bbox[3], 0)
            area = max(1, (max_row - min_row + 1) * (max_col - min_col + 1))
            compact_region = bool(area <= 6 and changed_pixels <= 6)
            border_region = bool(min_row == 0 or min_col == 0)
            remote_region = bool(not target_inside)
            status_candidate = bool(remote_region and compact_region and border_region)
            if not status_candidate:
                continue
            pre_color = _top_color_key(region.get("pre_colors_top"))
            post_color = _top_color_key(region.get("post_colors_top"))
            transition_key = f"{pre_color}->{post_color}"
            transition_counts[transition_key] += 1
            repeated_transition = bool(transition_counts[transition_key] >= 2)
            score_delta = _as_int(outcome.get("score_delta"), _as_int(closure_result.get("score_delta"), 0))
            goal_bearing = bool(
                outcome.get("goal_bearing")
                or closure_result.get("goal_bearing")
                or score_delta > 0
            )
            if goal_bearing:
                predicate_state = "remote_status_candidate_terminal_confirmed"
                terminal_or_score_confirmed_count += 1
            elif repeated_transition:
                predicate_state = "repeated_remote_status_candidate_immediate_terminal_unconfirmed"
                immediate_terminal_not_confirmed_count += 1
                required_followup_counts["remote_status_transition_posterior_required"] += 1
                required_followup_counts["terminal_confirmation_probe_required"] += 1
                required_followup_counts["negative_evidence_posterior_required"] += 1
            else:
                predicate_state = "remote_status_candidate_immediate_terminal_unconfirmed"
                immediate_terminal_not_confirmed_count += 1
                required_followup_counts["remote_status_transition_posterior_required"] += 1
                required_followup_counts["terminal_confirmation_probe_required"] += 1
                required_followup_counts["negative_evidence_posterior_required"] += 1

            row_count += 1
            status_predicate_candidate_count += 1
            remote_status_candidate_count += 1
            repeated_status_transition_count += 1 if repeated_transition else 0
            score_delta_sum += score_delta
            goal_bearing_count += 1 if goal_bearing else 0
            predicate_state_counts[predicate_state] += 1
            action_source = str(
                unsolved.get("post_transition_action_source", "")
                or unsolved.get("closure_action_mode", "")
                or "unknown"
            )
            action_source_counts[action_source] += 1
            by_game[game_id]["row_count"] += 1
            by_game[game_id][f"predicate_state:{predicate_state}"] += 1
            by_game[game_id][f"transition:{transition_key}"] += 1
            by_game[game_id]["remote_status_candidate_count"] += 1
            if repeated_transition:
                by_game[game_id]["repeated_status_transition_count"] += 1
            if goal_bearing:
                by_game[game_id]["terminal_or_score_confirmed_count"] += 1
            else:
                by_game[game_id]["immediate_terminal_not_confirmed_count"] += 1
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), 0),
                        "phase": str(item.get("phase", "") or ""),
                        "event": "closure" if is_closure else "post_transition",
                        "predicate_state": predicate_state,
                        "selected_action": str(
                            unsolved.get("selected_action") or item.get("action") or ""
                        ),
                        "action_source": action_source,
                        "transition_key": transition_key,
                        "changed_bbox": bbox,
                        "changed_centroid": list(region.get("centroid") or []),
                        "changed_pixel_count": changed_pixels,
                        "remote_status_candidate": True,
                        "repeated_transition": repeated_transition,
                        "terminal_or_score_confirmed": goal_bearing,
                        "score_delta": score_delta,
                        "goal_bearing": goal_bearing,
                    }
                )

    return {
        "schema_version": "eca_score_relation_terminal_status_predicate_sensor.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "row_count": int(row_count),
        "status_predicate_candidate_count": int(status_predicate_candidate_count),
        "remote_status_candidate_count": int(remote_status_candidate_count),
        "repeated_status_transition_count": int(repeated_status_transition_count),
        "terminal_or_score_confirmed_count": int(terminal_or_score_confirmed_count),
        "immediate_terminal_not_confirmed_count": int(
            immediate_terminal_not_confirmed_count
        ),
        "score_delta_sum": int(score_delta_sum),
        "goal_bearing_count": int(goal_bearing_count),
        "predicate_state_counts": dict(sorted(predicate_state_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
        "action_source_counts": dict(sorted(action_source_counts.items())),
        "required_followup_counts": dict(sorted(required_followup_counts.items())),
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "terminal_or_score_confirmed_count > 0 or posterior confirms status transition semantics",
            "allowed_connection": "diagnostic_to_remote_status_transition_posterior_only",
            "action_generation_allowed": False,
        },
    }


def _score_relation_remote_status_transition_posterior(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    transition_stats: dict[str, Counter[str]] = defaultdict(Counter)
    action_source_by_transition: dict[str, Counter[str]] = defaultdict(Counter)
    target_key_by_transition: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    observation_count = 0
    status_candidate_observation_count = 0
    terminal_or_score_confirmed_observation_count = 0
    no_goal_observation_count = 0
    negative_evidence_observation_count = 0
    score_delta_sum = 0

    def _changed_region(outcome: Mapping[str, Any]) -> Mapping[str, Any]:
        region = outcome.get("changed_region")
        if isinstance(region, Mapping):
            return region
        bbox = outcome.get("changed_bbox")
        centroid = outcome.get("changed_centroid")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return {
                "available": True,
                "changed_pixel_count": _as_int(outcome.get("changed_pixel_count"), 0),
                "bbox": list(bbox),
                "centroid": list(centroid) if isinstance(centroid, (list, tuple)) else [],
            }
        return {}

    def _bbox_contains_site(bbox: Any, site: Any) -> bool:
        if not (
            isinstance(bbox, (list, tuple))
            and len(bbox) >= 4
            and isinstance(site, (list, tuple))
            and len(site) >= 2
        ):
            return False
        row = _as_float(site[0], 0.0)
        col = _as_float(site[1], 0.0)
        return bool(
            _as_float(bbox[0], 0.0) <= row <= _as_float(bbox[2], 0.0)
            and _as_float(bbox[1], 0.0) <= col <= _as_float(bbox[3], 0.0)
        )

    def _top_color_key(rows: Any) -> str:
        if not isinstance(rows, list) or not rows:
            return "unknown"
        first = rows[0]
        if not isinstance(first, Mapping):
            return "unknown"
        return str(first.get("color_key", "") or "unknown")

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            closure_result = unsolved.get("closure_result")
            closure_result = closure_result if isinstance(closure_result, Mapping) else {}
            is_closure = bool(unsolved.get("closure")) and str(
                unsolved.get("closure_parent_candidate_kind", "") or ""
            ) == "score_relation_bounded_repeat_probe"
            is_post_transition = bool(unsolved.get("post_transition_replanner")) and str(
                unsolved.get("parent_candidate_kind", "") or ""
            ) == "score_relation_bounded_repeat_probe"
            if not is_closure and not is_post_transition:
                continue

            region = _changed_region(outcome)
            if not bool(region.get("available")):
                continue
            bbox = list(region.get("bbox") or [])
            if len(bbox) < 4:
                continue
            changed_pixels = _as_int(
                region.get("changed_pixel_count"),
                _as_int(outcome.get("changed_pixel_count"), 0),
            )
            if changed_pixels <= 0:
                continue
            target_site = list(unsolved.get("target_site") or [])
            min_row = _as_int(bbox[0], 0)
            min_col = _as_int(bbox[1], 0)
            max_row = _as_int(bbox[2], 0)
            max_col = _as_int(bbox[3], 0)
            area = max(1, (max_row - min_row + 1) * (max_col - min_col + 1))
            remote_region = bool(not _bbox_contains_site(bbox, target_site))
            compact_region = bool(area <= 6 and changed_pixels <= 6)
            border_region = bool(min_row == 0 or min_col == 0)
            if not (remote_region and compact_region and border_region):
                continue
            pre_color = _top_color_key(region.get("pre_colors_top"))
            post_color = _top_color_key(region.get("post_colors_top"))
            transition_key = f"{pre_color}->{post_color}"
            score_delta = _as_int(outcome.get("score_delta"), _as_int(closure_result.get("score_delta"), 0))
            terminal_or_score = bool(
                outcome.get("goal_bearing")
                or closure_result.get("goal_bearing")
                or score_delta > 0
            )
            no_goal = bool(not terminal_or_score)
            action_source = str(
                unsolved.get("post_transition_action_source", "")
                or unsolved.get("closure_action_mode", "")
                or "unknown"
            )
            score_relation_candidate = unsolved.get("score_relation_candidate")
            score_relation_candidate = (
                score_relation_candidate
                if isinstance(score_relation_candidate, Mapping)
                else {}
            )
            target_key_text = str(
                score_relation_candidate.get("target_key", "")
                or unsolved.get("target_key", "")
                or (
                    "site|" + "|".join(str(part) for part in target_site[:2])
                    if len(target_site) >= 2
                    else "unknown"
                )
            )

            observation_count += 1
            status_candidate_observation_count += 1
            terminal_or_score_confirmed_observation_count += 1 if terminal_or_score else 0
            no_goal_observation_count += 1 if no_goal else 0
            negative_evidence_observation_count += 1 if no_goal else 0
            score_delta_sum += score_delta
            transition_stats[transition_key]["observation_count"] += 1
            transition_stats[transition_key]["score_delta_sum"] += score_delta
            transition_stats[transition_key]["terminal_or_score_confirmed_count"] += (
                1 if terminal_or_score else 0
            )
            transition_stats[transition_key]["no_goal_count"] += 1 if no_goal else 0
            transition_stats[transition_key]["negative_evidence_count"] += (
                1 if no_goal else 0
            )
            action_source_by_transition[transition_key][action_source] += 1
            target_key_by_transition[transition_key][target_key_text] += 1
            by_game[game_id]["observation_count"] += 1
            by_game[game_id][f"transition:{transition_key}"] += 1
            if no_goal:
                by_game[game_id]["negative_evidence_count"] += 1
            if terminal_or_score:
                by_game[game_id]["terminal_or_score_confirmed_count"] += 1
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), 0),
                        "phase": str(item.get("phase", "") or ""),
                        "event": "closure" if is_closure else "post_transition",
                        "transition_key": transition_key,
                        "selected_action": str(
                            unsolved.get("selected_action") or item.get("action") or ""
                        ),
                        "action_source": action_source,
                        "changed_bbox": bbox,
                        "changed_centroid": list(region.get("centroid") or []),
                        "score_delta": score_delta,
                        "terminal_or_score_confirmed": terminal_or_score,
                        "negative_evidence": no_goal,
                    }
                )

    by_transition_key: dict[str, dict[str, Any]] = {}
    posterior_state_counts: Counter[str] = Counter()
    confirmed_transition_count = 0
    falsified_immediate_terminal_count = 0
    unconfirmed_transition_count = 0
    negative_evidence_transition_count = 0
    for transition_key, stats in sorted(transition_stats.items()):
        confirmed = _as_int(stats.get("terminal_or_score_confirmed_count"), 0)
        no_goal = _as_int(stats.get("no_goal_count"), 0)
        observations = _as_int(stats.get("observation_count"), 0)
        if confirmed > 0:
            posterior_state = "confirmed_terminal_or_score"
            confirmed_transition_count += 1
        elif no_goal >= 2:
            posterior_state = "falsified_immediate_terminal_no_goal"
            falsified_immediate_terminal_count += 1
            negative_evidence_transition_count += 1
        elif no_goal > 0:
            posterior_state = "unconfirmed_negative_single_observation"
            unconfirmed_transition_count += 1
            negative_evidence_transition_count += 1
        else:
            posterior_state = "insufficient_evidence"
            unconfirmed_transition_count += 1
        posterior_state_counts[posterior_state] += 1
        by_transition_key[transition_key] = {
            "posterior_state": posterior_state,
            "observation_count": observations,
            "terminal_or_score_confirmed_count": confirmed,
            "no_goal_count": no_goal,
            "negative_evidence_count": _as_int(stats.get("negative_evidence_count"), 0),
            "score_delta_sum": _as_int(stats.get("score_delta_sum"), 0),
            "action_source_counts": dict(
                sorted(action_source_by_transition[transition_key].items())
            ),
            "target_key_counts": dict(
                sorted(target_key_by_transition[transition_key].items())
            ),
        }

    return {
        "schema_version": "eca_score_relation_remote_status_transition_posterior.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "posterior_entry_count": int(len(by_transition_key)),
        "observation_count": int(observation_count),
        "status_candidate_observation_count": int(status_candidate_observation_count),
        "terminal_or_score_confirmed_observation_count": int(
            terminal_or_score_confirmed_observation_count
        ),
        "no_goal_observation_count": int(no_goal_observation_count),
        "negative_evidence_observation_count": int(
            negative_evidence_observation_count
        ),
        "confirmed_transition_count": int(confirmed_transition_count),
        "falsified_immediate_terminal_count": int(
            falsified_immediate_terminal_count
        ),
        "unconfirmed_transition_count": int(unconfirmed_transition_count),
        "negative_evidence_transition_count": int(
            negative_evidence_transition_count
        ),
        "score_delta_sum": int(score_delta_sum),
        "posterior_state_counts": dict(sorted(posterior_state_counts.items())),
        "by_transition_key": by_transition_key,
        "by_game": {
            game_id: dict(sorted(stats.items())) for game_id, stats in sorted(by_game.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "confirmed_transition_count > 0; negative-only transitions may only feed demotion/suppression design",
            "allowed_connection": "diagnostic_to_negative_evidence_rank_guard_design_only",
            "action_generation_allowed": False,
        },
    }


def _score_relation_remote_status_negative_rank_guard(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    posterior = _score_relation_remote_status_transition_posterior(
        rows,
        sample_limit=sample_limit,
    )
    blocked_transition_keys: list[str] = []
    weak_negative_transition_keys: list[str] = []
    by_transition_key: dict[str, dict[str, Any]] = {}
    blocked_observation_count = 0
    weak_negative_observation_count = 0
    confirmed_transition_count = _as_int(
        posterior.get("confirmed_transition_count"),
        0,
    )

    for transition_key, entry_raw in sorted(
        dict(posterior.get("by_transition_key") or {}).items()
    ):
        if not isinstance(entry_raw, Mapping):
            continue
        entry = dict(entry_raw)
        state = str(entry.get("posterior_state", "") or "")
        negative_count = _as_int(entry.get("negative_evidence_count"), 0)
        observation_count = _as_int(entry.get("observation_count"), 0)
        terminal_count = _as_int(entry.get("terminal_or_score_confirmed_count"), 0)
        guard_state = "not_guarded"
        rank_effect = "none"
        if state == "falsified_immediate_terminal_no_goal" and terminal_count <= 0:
            guard_state = "suppress_repeated_remote_status_transition"
            rank_effect = "demote_or_skip"
            blocked_transition_keys.append(str(transition_key))
            blocked_observation_count += observation_count
        elif (
            state == "unconfirmed_negative_single_observation"
            and terminal_count <= 0
        ):
            guard_state = "monitor_only_single_negative_observation"
            rank_effect = "none_until_repeated_negative"
            weak_negative_transition_keys.append(str(transition_key))
            weak_negative_observation_count += observation_count
        elif state == "confirmed_terminal_or_score":
            guard_state = "confirmed_not_blocked"
            rank_effect = "none"

        by_transition_key[str(transition_key)] = {
            "posterior_state": state,
            "guard_state": guard_state,
            "rank_effect": rank_effect,
            "observation_count": observation_count,
            "negative_evidence_count": negative_count,
            "terminal_or_score_confirmed_count": terminal_count,
            "score_delta_sum": _as_int(entry.get("score_delta_sum"), 0),
            "action_source_counts": dict(entry.get("action_source_counts") or {}),
            "target_key_counts": dict(entry.get("target_key_counts") or {}),
        }

    samples: list[dict[str, Any]] = []
    blocked_set = set(blocked_transition_keys)
    for sample_raw in list(posterior.get("samples") or []):
        if not isinstance(sample_raw, Mapping):
            continue
        transition_key = str(sample_raw.get("transition_key", "") or "")
        if transition_key not in blocked_set:
            continue
        sample = dict(sample_raw)
        sample["guard_state"] = "suppress_repeated_remote_status_transition"
        sample["rank_effect"] = "demote_or_skip"
        samples.append(sample)
        if len(samples) >= sample_limit:
            break

    return {
        "schema_version": "eca_score_relation_remote_status_negative_rank_guard.v1",
        "enabled": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "posterior_source_schema_version": str(
            posterior.get("schema_version", "") or ""
        ),
        "posterior_entry_count": _as_int(posterior.get("posterior_entry_count"), 0),
        "confirmed_transition_count": int(confirmed_transition_count),
        "blocked_transition_count": int(len(blocked_transition_keys)),
        "weak_negative_transition_count": int(len(weak_negative_transition_keys)),
        "blocked_observation_count": int(blocked_observation_count),
        "weak_negative_observation_count": int(weak_negative_observation_count),
        "negative_evidence_observation_count": _as_int(
            posterior.get("negative_evidence_observation_count"),
            0,
        ),
        "blocked_transition_keys": blocked_transition_keys,
        "weak_negative_transition_keys": weak_negative_transition_keys,
        "by_transition_key": by_transition_key,
        "samples": samples,
        "rank_guard_policy": {
            "minimum_posterior_state": "falsified_immediate_terminal_no_goal",
            "allowed_effect": "demote_or_skip_matching_score_relation_status_transition_candidate",
            "disallowed_effects": [
                "generate_new_action",
                "positive_boost_from_negative_evidence",
                "override_confirmed_transition",
            ],
            "candidate_relation_scope": [
                "bounded_repeat_to_score_delta_relation",
            ],
            "candidate_kind_scope": [
                "score_relation_bounded_repeat_probe",
                "score_relation_target_click_closure",
                "score_retest_after_score_relation_target_click_transition",
            ],
        },
        "next_runtime_contract": {
            "env_flag": "ECA_ENABLE_SCORE_RELATION_REMOTE_STATUS_NEGATIVE_RANK_GUARD",
            "default": "off",
            "required_input": "score_relation_remote_status_transition_posterior",
            "allowed_connection": "negative_posterior_to_demote_or_skip_only",
            "action_generation_allowed": False,
            "positive_ranking_allowed": False,
        },
    }


def _click_discovery_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    enabled_rows = 0
    max_attempts = 0
    max_successes = 0
    max_failures = 0
    rows_with_attempts = 0
    source_totals: dict[str, Counter[str]] = defaultdict(Counter)
    source_rows_with_attempts: Counter[str] = Counter()
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        row_attempts = 0
        row_enabled = False
        row_source_max: dict[str, Counter[str]] = defaultdict(Counter)
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            payload = item.get("click_site_discovery")
            if not isinstance(payload, Mapping):
                continue
            row_enabled = row_enabled or bool(payload.get("enabled"))
            attempts = _as_int(payload.get("attempt_count"), 0)
            successes = _as_int(payload.get("success_count"), 0)
            failures = _as_int(payload.get("failure_count"), 0)
            row_attempts = max(row_attempts, attempts)
            max_attempts = max(max_attempts, attempts)
            max_successes = max(max_successes, successes)
            max_failures = max(max_failures, failures)
            source_stats = payload.get("source_stats")
            if isinstance(source_stats, Mapping):
                for raw_source, raw_stats in source_stats.items():
                    if not isinstance(raw_stats, Mapping):
                        continue
                    source = str(raw_source)
                    row_source_max[source]["attempts"] = max(
                        row_source_max[source]["attempts"],
                        _as_int(raw_stats.get("attempts"), 0),
                    )
                    row_source_max[source]["successes"] = max(
                        row_source_max[source]["successes"],
                        _as_int(raw_stats.get("successes"), 0),
                    )
                    row_source_max[source]["failures"] = max(
                        row_source_max[source]["failures"],
                        _as_int(raw_stats.get("failures"), 0),
                    )
        if row_enabled:
            enabled_rows += 1
        if row_attempts > 0:
            rows_with_attempts += 1
        for source, stats in row_source_max.items():
            for key in ("attempts", "successes", "failures"):
                source_totals[source][key] += int(stats[key])
            if int(stats["attempts"]) > 0:
                source_rows_with_attempts[source] += 1
    return {
        "enabled_row_count": enabled_rows,
        "rows_with_attempts": rows_with_attempts,
        "max_attempt_count": max_attempts,
        "max_success_count": max_successes,
        "max_failure_count": max_failures,
        "source_stats": {
            source: {
                "attempts": int(stats["attempts"]),
                "successes": int(stats["successes"]),
                "failures": int(stats["failures"]),
                "rows_with_attempts": int(source_rows_with_attempts[source]),
            }
            for source, stats in sorted(source_totals.items())
        },
    }


def _click_affordance_induction_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    enabled_rows = 0
    rows_with_observations = 0
    max_attempts = 0
    max_successes = 0
    max_failures = 0
    relation_totals: dict[str, Counter[str]] = defaultdict(Counter)
    verified_relation_totals: dict[str, Counter[str]] = defaultdict(Counter)
    chain_totals: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        row_enabled = False
        row_observations = 0
        row_attempts = 0
        row_successes = 0
        row_failures = 0
        row_relations: dict[str, Counter[str]] = defaultdict(Counter)
        row_verified_relations: dict[str, Counter[str]] = defaultdict(Counter)
        row_chains: dict[str, Counter[str]] = defaultdict(Counter)
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            payload = item.get("click_affordance_induction")
            if not isinstance(payload, Mapping):
                continue
            row_enabled = row_enabled or bool(payload.get("enabled"))
            row_observations = max(row_observations, _as_int(payload.get("observation_count"), 0))
            row_attempts = max(row_attempts, _as_int(payload.get("attempt_count"), 0))
            row_successes = max(row_successes, _as_int(payload.get("success_count"), 0))
            row_failures = max(row_failures, _as_int(payload.get("failure_count"), 0))
            relation_stats = payload.get("relation_stats")
            if isinstance(relation_stats, Mapping):
                for raw_kind, raw_stats in relation_stats.items():
                    if not isinstance(raw_stats, Mapping):
                        continue
                    kind = str(raw_kind)
                    row_relations[kind]["attempts"] = max(
                        row_relations[kind]["attempts"],
                        _as_int(raw_stats.get("attempts"), 0),
                    )
                    row_relations[kind]["successes"] = max(
                        row_relations[kind]["successes"],
                        _as_int(raw_stats.get("successes"), 0),
                    )
                    row_relations[kind]["failures"] = max(
                        row_relations[kind]["failures"],
                        _as_int(raw_stats.get("failures"), 0),
                    )
            verified_relation_stats = payload.get("verified_relation_stats")
            if isinstance(verified_relation_stats, Mapping):
                for raw_kind, raw_stats in verified_relation_stats.items():
                    if not isinstance(raw_stats, Mapping):
                        continue
                    kind = str(raw_kind)
                    row_verified_relations[kind]["attempts"] = max(
                        row_verified_relations[kind]["attempts"],
                        _as_int(raw_stats.get("attempts"), 0),
                    )
                    row_verified_relations[kind]["successes"] = max(
                        row_verified_relations[kind]["successes"],
                        _as_int(raw_stats.get("successes"), 0),
                    )
                    row_verified_relations[kind]["failures"] = max(
                        row_verified_relations[kind]["failures"],
                        _as_int(raw_stats.get("failures"), 0),
                    )
            chain_stats = payload.get("chain_stats")
            if isinstance(chain_stats, Mapping):
                for raw_action, raw_stats in chain_stats.items():
                    if not isinstance(raw_stats, Mapping):
                        continue
                    action = str(raw_action)
                    row_chains[action]["attempts"] = max(
                        row_chains[action]["attempts"],
                        _as_int(raw_stats.get("attempts"), 0),
                    )
                    row_chains[action]["successes"] = max(
                        row_chains[action]["successes"],
                        _as_int(raw_stats.get("successes"), 0),
                    )
                    row_chains[action]["failures"] = max(
                        row_chains[action]["failures"],
                        _as_int(raw_stats.get("failures"), 0),
                    )
        if row_enabled:
            enabled_rows += 1
        if row_observations > 0:
            rows_with_observations += 1
        max_attempts = max(max_attempts, row_attempts)
        max_successes = max(max_successes, row_successes)
        max_failures = max(max_failures, row_failures)
        for kind, stats in row_relations.items():
            for key in ("attempts", "successes", "failures"):
                relation_totals[kind][key] += int(stats[key])
        for kind, stats in row_verified_relations.items():
            for key in ("attempts", "successes", "failures"):
                verified_relation_totals[kind][key] += int(stats[key])
        for action, stats in row_chains.items():
            for key in ("attempts", "successes", "failures"):
                chain_totals[action][key] += int(stats[key])
    return {
        "enabled_row_count": enabled_rows,
        "rows_with_observations": rows_with_observations,
        "max_attempt_count": max_attempts,
        "max_success_count": max_successes,
        "max_failure_count": max_failures,
        "relation_stats": {
            kind: {
                "attempts": int(stats["attempts"]),
                "successes": int(stats["successes"]),
                "failures": int(stats["failures"]),
            }
            for kind, stats in sorted(relation_totals.items())
        },
        "verified_relation_stats": {
            kind: {
                "attempts": int(stats["attempts"]),
                "successes": int(stats["successes"]),
                "failures": int(stats["failures"]),
            }
            for kind, stats in sorted(verified_relation_totals.items())
        },
        "chain_stats": {
            action: {
                "attempts": int(stats["attempts"]),
                "successes": int(stats["successes"]),
                "failures": int(stats["failures"]),
            }
            for action, stats in sorted(chain_totals.items())
        },
    }


def _canonical_instability_games(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_game: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        by_game[game_id].append(row)

    unstable: list[dict[str, Any]] = []
    for game_id, items in sorted(by_game.items()):
        canonical_scores = [
            _as_int(row.get("final_score"), 0)
            for row in items
            if str(row.get("variant", "") or "") == "canonical_full_plan"
        ]
        perturbation_scores = [
            _as_int(row.get("final_score"), 0)
            for row in items
            if bool(row.get("perturbation_applied"))
        ]
        if not canonical_scores or not perturbation_scores:
            continue
        canonical_best = max(canonical_scores)
        perturbation_best = max(perturbation_scores)
        if perturbation_best > canonical_best:
            unstable.append(
                {
                    "game_id": game_id,
                    "canonical_best_score": canonical_best,
                    "perturbation_best_score": perturbation_best,
                    "score_gap": perturbation_best - canonical_best,
                }
            )
    return unstable


def _solver_family_budget_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "trace_count": 0,
            "exhausted_trace_count": 0,
            "rows_with_exhaustion": 0,
            "max_attempts": 0,
            "max_score_successes": 0,
            "max_goal_successes": 0,
            "max_progress_only": 0,
            "max_failures": 0,
        }
    )
    rows_with_family: Counter[str] = Counter()
    for row in rows:
        row_exhausted: set[str] = set()
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            budget = item.get("solver_family_budget")
            if not isinstance(budget, Mapping):
                continue
            for family, raw_stats in budget.items():
                if not isinstance(raw_stats, Mapping):
                    continue
                key = str(family or "")
                if not key:
                    continue
                stats = summary[key]
                stats["trace_count"] += 1
                attempts = _as_int(raw_stats.get("attempts"), 0)
                score_successes = _as_int(raw_stats.get("score_successes"), 0)
                goal_successes = _as_int(raw_stats.get("goal_successes"), score_successes)
                progress_only = _as_int(raw_stats.get("progress_only"), 0)
                failures = _as_int(raw_stats.get("failures"), 0)
                stats["max_attempts"] = max(stats["max_attempts"], attempts)
                stats["max_score_successes"] = max(stats["max_score_successes"], score_successes)
                stats["max_goal_successes"] = max(stats["max_goal_successes"], goal_successes)
                stats["max_progress_only"] = max(stats["max_progress_only"], progress_only)
                stats["max_failures"] = max(stats["max_failures"], failures)
                if bool(raw_stats.get("exhausted")):
                    stats["exhausted_trace_count"] += 1
                    row_exhausted.add(key)
        for family in row_exhausted:
            rows_with_family[family] += 1
    for family, count in rows_with_family.items():
        summary[family]["rows_with_exhaustion"] = int(count)
    return {family: dict(stats) for family, stats in sorted(summary.items())}


def _goal_target_contract_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    row_status_counts: Counter[str] = Counter()
    by_family: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "trace_count": 0,
            "missing_objective": 0,
            "progress_objective_only": 0,
            "goal_objective_bound": 0,
            "max_objective_candidate_count": 0,
            "max_goal_objective_candidate_count": 0,
        }
    )
    trace_count = 0
    for row in rows:
        row_statuses: set[str] = set()
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            control_target = item.get("control_target")
            if not isinstance(control_target, Mapping):
                continue
            contract = control_target.get("goal_target_contract")
            if not isinstance(contract, Mapping):
                continue
            trace_count += 1
            status = str(contract.get("target_translation_status", "") or "unknown")
            family = str(contract.get("solver_family", "") or "unknown")
            status_counts[status] += 1
            row_statuses.add(status)
            stats = by_family[family]
            stats["trace_count"] += 1
            if status in stats:
                stats[status] += 1
            stats["max_objective_candidate_count"] = max(
                stats["max_objective_candidate_count"],
                _as_int(contract.get("objective_candidate_count"), 0),
            )
            stats["max_goal_objective_candidate_count"] = max(
                stats["max_goal_objective_candidate_count"],
                _as_int(contract.get("goal_objective_candidate_count"), 0),
            )
        for status in row_statuses:
            row_status_counts[status] += 1
    return {
        "trace_count": int(trace_count),
        "status_counts": dict(sorted(status_counts.items())),
        "row_status_counts": dict(sorted(row_status_counts.items())),
        "by_solver_family": {family: dict(stats) for family, stats in sorted(by_family.items())},
    }


def _adapter_agreement_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "trace_step_count": 0,
        "comparison_count": 0,
        "advice_coverage_rate": 0.0,
        "agreement_count": 0,
        "action_agreement_rate": 0.0,
        "payload_comparison_count": 0,
        "payload_agreement_count": 0,
        "payload_agreement_rate": 0.0,
        "advice_available_count": 0,
        "advice_used_count": 0,
        "advice_usage_rate": 0.0,
        "by_mode": {},
        "by_game": {},
    }
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        mode = str(row.get("adapter_advice_mode", "") or "none")
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            totals["trace_step_count"] += 1
            by_mode[mode]["trace_step_count"] += 1
            by_game[game_id]["trace_step_count"] += 1
            advice = item.get("adapter_advice")
            if not isinstance(advice, Mapping):
                continue
            if not bool(advice.get("available")):
                continue
            totals["advice_available_count"] += 1
            by_mode[mode]["advice_available_count"] += 1
            by_game[game_id]["advice_available_count"] += 1
            if bool(advice.get("used")):
                totals["advice_used_count"] += 1
                by_mode[mode]["advice_used_count"] += 1
                by_game[game_id]["advice_used_count"] += 1
            if "action_agreed" not in advice:
                continue
            totals["comparison_count"] += 1
            by_mode[mode]["comparison_count"] += 1
            by_game[game_id]["comparison_count"] += 1
            if bool(advice.get("action_agreed")):
                totals["agreement_count"] += 1
                by_mode[mode]["agreement_count"] += 1
                by_game[game_id]["agreement_count"] += 1
            if "payload_agreed" in advice:
                totals["payload_comparison_count"] += 1
                by_mode[mode]["payload_comparison_count"] += 1
                by_game[game_id]["payload_comparison_count"] += 1
                if bool(advice.get("payload_agreed")):
                    totals["payload_agreement_count"] += 1
                    by_mode[mode]["payload_agreement_count"] += 1
                    by_game[game_id]["payload_agreement_count"] += 1

    totals["action_agreement_rate"] = _safe_div(
        totals["agreement_count"], totals["comparison_count"]
    )
    totals["advice_coverage_rate"] = _safe_div(
        totals["advice_available_count"], totals["trace_step_count"]
    )
    totals["payload_agreement_rate"] = _safe_div(
        totals["payload_agreement_count"], totals["payload_comparison_count"]
    )
    totals["advice_usage_rate"] = _safe_div(
        totals["advice_used_count"], totals["advice_available_count"]
    )

    def convert(counter: Counter[str]) -> dict[str, Any]:
        comparisons = int(counter.get("comparison_count", 0))
        trace_steps = int(counter.get("trace_step_count", 0))
        agreements = int(counter.get("agreement_count", 0))
        payload_comparisons = int(counter.get("payload_comparison_count", 0))
        payload_agreements = int(counter.get("payload_agreement_count", 0))
        available = int(counter.get("advice_available_count", 0))
        used = int(counter.get("advice_used_count", 0))
        return {
            "trace_step_count": trace_steps,
            "comparison_count": comparisons,
            "advice_coverage_rate": _safe_div(available, trace_steps),
            "agreement_count": agreements,
            "action_agreement_rate": _safe_div(agreements, comparisons),
            "payload_comparison_count": payload_comparisons,
            "payload_agreement_count": payload_agreements,
            "payload_agreement_rate": _safe_div(payload_agreements, payload_comparisons),
            "advice_available_count": available,
            "advice_used_count": used,
            "advice_usage_rate": _safe_div(used, available),
        }

    totals["by_mode"] = {key: convert(value) for key, value in sorted(by_mode.items())}
    totals["by_game"] = {key: convert(value) for key, value in sorted(by_game.items())}
    return totals


def _adapter_plan_cache_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)

    def bump(counter: Counter[str], key: str, amount: int = 1) -> None:
        counter[key] += int(amount)

    def observe(counter: Counter[str], advice: Mapping[str, Any], cache: Mapping[str, Any]) -> None:
        bump(counter, "event_count")
        if bool(cache.get("enabled")):
            bump(counter, "enabled_event_count")
        if bool(cache.get("cache_refreshed")):
            bump(counter, "cache_refresh_count")
        if bool(cache.get("cache_hit")):
            bump(counter, "cache_hit_count")
        if bool(cache.get("cache_action_available")):
            bump(counter, "cache_action_available_count")
            if bool(advice.get("used")):
                bump(counter, "cache_action_used_count")
                payload = advice.get("payload")
                if isinstance(payload, Mapping) and len(payload) > 1:
                    bump(counter, "cache_payload_used_count")
            elif bool(advice.get("available")):
                bump(counter, "cache_action_rejected_count")
                reason = str(advice.get("decision_reason", "") or "unknown")
                bump(counter, f"rejection_reason:{reason}")
        if bool(cache.get("cache_rejected")):
            bump(counter, "cache_rejected_count")
        if bool(cache.get("cache_invalidated")):
            bump(counter, "cache_invalidated_count")
            reason = str(cache.get("invalidation_reason", "") or "unknown")
            bump(counter, f"invalidation_reason:{reason}")
        reason = str(cache.get("reason", "") or "unknown")
        bump(counter, f"reason:{reason}")
        source = str(cache.get("source", "") or "unknown")
        bump(counter, f"source:{source}")
        if reason == "cache_empty":
            bump(counter, f"cache_empty_source:{source}")

    for row in rows:
        mode = str(row.get("adapter_advice_mode", "") or "none")
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            advice = item.get("adapter_advice")
            if not isinstance(advice, Mapping):
                continue
            cache = advice.get("plan_cache")
            if not isinstance(cache, Mapping):
                continue
            observe(counts, advice, cache)
            observe(by_mode[mode], advice, cache)
            observe(by_game[game_id], advice, cache)

    def prefixed(counter: Counter[str], prefix: str) -> dict[str, int]:
        out: dict[str, int] = {}
        marker = f"{prefix}:"
        for key, value in sorted(counter.items()):
            if key.startswith(marker):
                out[key[len(marker) :]] = int(value)
        return out

    def convert(counter: Counter[str]) -> dict[str, Any]:
        event_count = int(counter.get("event_count", 0))
        hit_count = int(counter.get("cache_hit_count", 0))
        action_available = int(counter.get("cache_action_available_count", 0))
        action_used = int(counter.get("cache_action_used_count", 0))
        return {
            "event_count": event_count,
            "enabled_event_count": int(counter.get("enabled_event_count", 0)),
            "cache_refresh_count": int(counter.get("cache_refresh_count", 0)),
            "cache_hit_count": hit_count,
            "cache_hit_rate": _safe_div(hit_count, event_count),
            "cache_action_available_count": action_available,
            "cache_action_used_count": action_used,
            "cache_action_rejected_count": int(
                counter.get("cache_action_rejected_count", 0)
            ),
            "cache_action_usage_rate": _safe_div(action_used, action_available),
            "cache_payload_used_count": int(counter.get("cache_payload_used_count", 0)),
            "cache_rejected_count": int(counter.get("cache_rejected_count", 0)),
            "cache_invalidated_count": int(counter.get("cache_invalidated_count", 0)),
            "reason_counts": prefixed(counter, "reason"),
            "source_counts": prefixed(counter, "source"),
            "cache_empty_source_counts": prefixed(counter, "cache_empty_source"),
            "invalidation_reasons": prefixed(counter, "invalidation_reason"),
            "rejection_reasons": prefixed(counter, "rejection_reason"),
        }

    summary = convert(counts)
    summary["by_mode"] = {key: convert(value) for key, value in sorted(by_mode.items())}
    summary["by_game"] = {key: convert(value) for key, value in sorted(by_game.items())}
    return summary


def _unsolved_to_replan_summary(
    rows: Iterable[Mapping[str, Any]],
    sample_limit: int = 20,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_signature_counts: Counter[str] = Counter()
    eligible_signature_counts: Counter[str] = Counter()
    probe_candidate_counts: Counter[str] = Counter()
    probe_effect_counts: Counter[str] = Counter()
    probe_to_rod_verdict_counts: Counter[str] = Counter()
    probe_to_rod_effect_counts: Counter[str] = Counter()
    short_plan_kind_counts: Counter[str] = Counter()
    short_plan_effect_counts: Counter[str] = Counter()
    short_plan_admission_reason_counts: Counter[str] = Counter()
    short_plan_admission_kind_counts: Counter[str] = Counter()
    eligible_steps_with_probe_candidates = 0
    eligible_steps_with_short_plan_candidates = 0
    eligible_steps_with_admitted_short_plan = 0
    probe_candidate_total = 0
    probe_to_rod_preview_total = 0
    short_plan_candidate_total = 0
    short_plan_admission_eval_total = 0
    short_plan_admission_accepted_total = 0
    probe_to_rod_payload_leakage_count = 0
    short_plan_payload_leakage_count = 0
    short_plan_admission_payload_leakage_count = 0
    signature_leakage_count = 0
    samples: list[dict[str, Any]] = []

    def bump(counter: Counter[str], key: str, amount: int = 1) -> None:
        counter[key] += int(amount)

    def goal_status(item: Mapping[str, Any]) -> str:
        control = item.get("control_target")
        if not isinstance(control, Mapping):
            return ""
        contract = control.get("goal_target_contract")
        if not isinstance(contract, Mapping):
            return ""
        return str(contract.get("target_translation_status", "") or "")

    def is_unsolved_source(source: str) -> bool:
        normalized = str(source or "").strip().lower()
        return normalized == "unsolved" or normalized.endswith("_unsolved")

    def count_band(count: int) -> str:
        value = max(0, int(count or 0))
        if value <= 0:
            return "0"
        if value == 1:
            return "1"
        if value <= 3:
            return "2-3"
        if value <= 7:
            return "4-7"
        return "8+"

    def score_delta_band(delta: int) -> str:
        value = int(delta or 0)
        if value > 0:
            return "positive"
        if value < 0:
            return "negative"
        return "zero"

    def coarse_operator(value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return "none"
        if "click" in text:
            return "click"
        if "route" in text or "grid" in text or "nav" in text or "maze" in text:
            return "route"
        if "push" in text or "transport" in text:
            return "transport"
        if "attach" in text or "select" in text or "drop" in text:
            return "select_attach"
        if "probe" in text or "frontier" in text:
            return "probe"
        if "repeat" in text:
            return "repeat"
        if "score" in text or "objective" in text or "goal" in text:
            return "objective"
        return "other"

    def coarse_target_kind(value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return "none"
        if "score" in text:
            return "score"
        if "objective" in text or "goal" in text:
            return "objective"
        if "segment" in text or "transport" in text:
            return "segment"
        if "click" in text or "button" in text:
            return "click"
        if "route" in text or "path" in text or "cell" in text:
            return "route"
        if "actor" in text or "avatar" in text:
            return "actor"
        if "object" in text or "component" in text:
            return "object"
        return "other"

    def recent_score_delta(trace: list[Any], index: int, window: int = 6) -> int:
        if not trace or index < 0:
            return 0
        current = _trace_score_before(trace[index], 0)
        start = max(0, int(index) - max(1, int(window)))
        baseline = _trace_score_before(trace[start], current)
        return int(current - baseline)

    def recent_action_pressure(trace: list[Any], index: int, window: int = 6) -> tuple[str, str]:
        recent = [
            _normalize_action_name(item.get("action"))
            for item in trace[max(0, int(index) - max(1, int(window))) : int(index)]
            if isinstance(item, Mapping)
        ]
        recent = [action for action in recent if action]
        if not recent:
            return "0", "none"
        counts = Counter(recent)
        dominant = max(counts.values(), default=0)
        pressure = "high" if dominant >= 4 else "medium" if dominant >= 2 else "low"
        return count_band(len(counts)), pressure

    def verification_counts(item: Mapping[str, Any]) -> Counter[str]:
        counter: Counter[str] = Counter()
        payload = item.get("target_verification")
        if not isinstance(payload, Mapping):
            return counter
        recent = payload.get("recent")
        if isinstance(recent, list):
            entries = [entry for entry in recent if isinstance(entry, Mapping)]
        else:
            entries = []
        counter["attempt"] = max(_as_int(payload.get("attempt_count"), 0), len(entries))
        counter["success"] = max(
            _as_int(payload.get("success_count"), 0),
            sum(1 for entry in entries if bool(entry.get("target_success"))),
        )
        counter["failure"] = max(
            _as_int(payload.get("failure_count"), 0),
            sum(1 for entry in entries if entry.get("target_success") is False),
        )
        counter["noop"] = sum(1 for entry in entries if bool(entry.get("no_op")))
        counter["visible_effect"] = sum(
            1 for entry in entries if _as_int(entry.get("changed_pixel_count"), 0) > 0
        )
        counter["goal_progress"] = sum(1 for entry in entries if bool(entry.get("goal_progress")))
        counter["objective_progress"] = sum(
            1 for entry in entries if bool(entry.get("objective_progress"))
        )
        counter["direct_score_probe_failure"] = max(
            _as_int(payload.get("direct_score_probe_without_goal_failure_count"), 0),
            sum(
                1
                for entry in entries
                if bool(entry.get("direct_score_probe_without_goal"))
                and not bool(entry.get("target_success"))
            ),
        )
        return counter

    def target_verification_entries(item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        payload = item.get("target_verification")
        if not isinstance(payload, Mapping):
            return []
        recent = payload.get("recent")
        if not isinstance(recent, list):
            return []
        return [entry for entry in recent if isinstance(entry, Mapping)]

    def observed_probe_effect(item: Mapping[str, Any]) -> dict[str, Any]:
        verify = verification_counts(item)
        entries = target_verification_entries(item)
        score_delta = max(
            [_as_int(item.get("score_delta"), 0)]
            + [_as_int(entry.get("score_delta"), 0) for entry in entries],
            default=0,
        )
        changed_pixels = max(
            [_as_int(item.get("changed_pixel_count"), 0)]
            + [_as_int(entry.get("changed_pixel_count"), 0) for entry in entries],
            default=0,
        )
        goal_progress = bool(
            int(verify.get("goal_progress", 0)) > 0
            or any(bool(entry.get("goal_progress")) for entry in entries)
        )
        objective_progress = bool(
            int(verify.get("objective_progress", 0)) > 0
            or any(bool(entry.get("objective_progress")) for entry in entries)
        )
        no_op = bool(int(verify.get("noop", 0)) > 0 or bool(item.get("no_op")))
        failure_count = int(verify.get("failure", 0))
        direct_failure_count = int(verify.get("direct_score_probe_failure", 0))
        if score_delta > 0:
            effect_type = "score_delta"
        elif goal_progress:
            effect_type = "goal_progress"
        elif objective_progress:
            effect_type = "objective_progress"
        elif changed_pixels > 0:
            effect_type = "visible_effect"
        elif no_op:
            effect_type = "noop"
        elif failure_count > 0 or direct_failure_count > 0:
            effect_type = "failure"
        else:
            effect_type = "unexplained_effect"
        return {
            "effect_type": effect_type,
            "score_delta": int(score_delta),
            "changed_pixel_count": int(changed_pixels),
            "goal_progress": bool(goal_progress),
            "objective_progress": bool(objective_progress),
            "no_op": bool(no_op),
            "failure_count": int(failure_count),
            "direct_score_probe_failure_count": int(direct_failure_count),
        }

    def current_state_signature(
        item: Mapping[str, Any],
        trace: list[Any],
        index: int,
    ) -> dict[str, Any]:
        control = item.get("control_target")
        control = control if isinstance(control, Mapping) else {}
        contract = control.get("goal_target_contract")
        contract = contract if isinstance(contract, Mapping) else {}
        primary = control.get("primary_target")
        primary = primary if isinstance(primary, Mapping) else {}
        candidates = [
            candidate
            for candidate in list(control.get("target_candidates") or [])
            if isinstance(candidate, Mapping)
        ]
        probe_sites = [
            candidate
            for candidate in list(control.get("probe_site_candidates") or [])
            if isinstance(candidate, Mapping)
        ]
        actionable_probe_sites = [
            candidate
            for candidate in probe_sites
            if bool(candidate.get("actionable")) or bool(candidate.get("action_name"))
        ]
        click_probe_sites = [
            candidate
            for candidate in probe_sites
            if str(candidate.get("operator", "") or "").lower() == "click"
            or _normalize_action_name(candidate.get("action_name")) == "ACTION6"
        ]
        verify = verification_counts(item)
        rods = _structural_rods_from_trace_item(item)
        rod_effect_types = [
            str(effect)
            for effect, _count in sorted(
                dict(rods.get("by_effect_type") or {}).items(),
                key=lambda pair: (-_as_int(pair[1], 0), str(pair[0])),
            )
            if str(effect or "")
        ][:3]
        recent_action_diversity, recent_repeat_pressure = recent_action_pressure(trace, index)
        fields = {
            "goal_status": str(contract.get("target_translation_status", "") or ""),
            "operator": coarse_operator(control.get("primary_operator")),
            "solver_family": coarse_operator(
                contract.get("solver_family") or control.get("solver_family")
            ),
            "primary_target_kind": coarse_target_kind(primary.get("kind")),
            "primary_has_site": "yes" if _site_from_mapping(primary) is not None else "no",
            "target_candidate_band": count_band(len(candidates)),
            "probe_site_band": count_band(len(probe_sites)),
            "actionable_probe_band": count_band(len(actionable_probe_sites)),
            "click_probe_band": count_band(len(click_probe_sites)),
            "recent_action_diversity_band": recent_action_diversity,
            "recent_repeat_pressure": recent_repeat_pressure,
            "recent_score_delta": score_delta_band(recent_score_delta(trace, index)),
            "target_attempt_band": count_band(int(verify.get("attempt", 0))),
            "target_failure_band": count_band(int(verify.get("failure", 0))),
            "target_success_band": count_band(int(verify.get("success", 0))),
            "noop_band": count_band(int(verify.get("noop", 0))),
            "visible_effect_band": count_band(int(verify.get("visible_effect", 0))),
            "goal_progress_band": count_band(int(verify.get("goal_progress", 0))),
            "objective_progress_band": count_band(int(verify.get("objective_progress", 0))),
            "direct_score_probe_failure_band": count_band(
                int(verify.get("direct_score_probe_failure", 0))
            ),
            "rod_count_band": count_band(_as_int(rods.get("rod_count"), 0)),
            "rod_contradiction_band": count_band(_as_int(rods.get("contradiction_count"), 0)),
            "rod_effect_profile": "+".join(rod_effect_types) if rod_effect_types else "none",
        }
        key = "|".join(f"{name}={fields[name]}" for name in sorted(fields))
        return {
            "schema_version": "eca_unsolved_current_state_signature.v1",
            "key": key,
            **fields,
        }

    def signature_has_leakage(signature: Mapping[str, Any]) -> bool:
        text = str(signature.get("key", "") or "").lower()
        forbidden = set(PERCEPTION_ADAPTER_REGISTRY.keys()) | {
            "adapter",
            "canonical",
            "fixed_plan",
            "layout",
        }
        return any(token and token in text for token in forbidden)

    def band_has_any(value: Any) -> bool:
        return str(value or "0") not in {"", "0", "none", "zero"}

    def probe_candidates_from_signature(signature: Mapping[str, Any]) -> list[dict[str, Any]]:
        state_key = str(signature.get("key", "") or "")
        operator = str(signature.get("operator", "") or "none")
        target_kind = str(signature.get("primary_target_kind", "") or "none")
        candidates: list[dict[str, Any]] = []

        def add(
            kind: str,
            *,
            expected_effect_type: str,
            max_probe_length: int,
            bailout_condition: str,
            evidence_keys: Iterable[str],
            priority: float,
        ) -> None:
            generic_keys = [str(key) for key in evidence_keys if str(key)]
            candidates.append(
                {
                    "schema_version": "eca_unsolved_probe_candidate.v1",
                    "candidate_kind": str(kind),
                    "expected_effect_type": str(expected_effect_type),
                    "commitment_level": "probe_low",
                    "bailout_condition": str(bailout_condition),
                    "max_probe_length": max(1, int(max_probe_length)),
                    "generic_evidence_keys": sorted(set(generic_keys)),
                    "source_state_key": state_key,
                    "priority": round(float(priority), 6),
                }
            )

        if band_has_any(signature.get("click_probe_band")):
            add(
                "click_probe_site",
                expected_effect_type="click_transition",
                max_probe_length=1,
                bailout_condition="stop if click has no visible effect, score delta, or goal progress",
                evidence_keys=[
                    "operator:click",
                    "probe_site:click",
                    f"goal_status:{signature.get('goal_status', '')}",
                ],
                priority=0.72,
            )
        if band_has_any(signature.get("target_candidate_band")):
            if operator in {"route", "objective", "transport"} or target_kind in {
                "objective",
                "score",
                "route",
                "segment",
            }:
                add(
                    "route_or_objective_frontier_probe",
                    expected_effect_type="route_connectivity_delta",
                    max_probe_length=3,
                    bailout_condition="stop on no route progress after bounded movement probe",
                    evidence_keys=[
                        f"operator:{operator}",
                        f"target_kind:{target_kind}",
                        f"goal_status:{signature.get('goal_status', '')}",
                    ],
                    priority=0.66,
                )
            else:
                add(
                    "visible_target_effect_probe",
                    expected_effect_type="visible_effect",
                    max_probe_length=2,
                    bailout_condition="stop if target attempt causes no visible or objective progress",
                    evidence_keys=[
                        f"operator:{operator}",
                        f"target_kind:{target_kind}",
                    ],
                    priority=0.58,
                )
        if (
            str(signature.get("recent_repeat_pressure", "") or "") == "high"
            or band_has_any(signature.get("noop_band"))
        ):
            add(
                "escape_noop_loop_probe",
                expected_effect_type="unblocked_or_visible_effect",
                max_probe_length=2,
                bailout_condition="stop after first repeated no-op or unchanged state",
                evidence_keys=[
                    f"recent_repeat_pressure:{signature.get('recent_repeat_pressure', '')}",
                    f"noop_band:{signature.get('noop_band', '')}",
                ],
                priority=0.52,
            )
        if band_has_any(signature.get("direct_score_probe_failure_band")):
            add(
                "indirect_score_precondition_probe",
                expected_effect_type="objective_progress",
                max_probe_length=3,
                bailout_condition="stop if direct-score failure repeats without intermediate progress",
                evidence_keys=[
                    "direct_score_probe_failure:true",
                    f"goal_status:{signature.get('goal_status', '')}",
                ],
                priority=0.70,
            )
        if (
            str(signature.get("rod_effect_profile", "") or "none") != "none"
            and not band_has_any(signature.get("rod_contradiction_band"))
        ):
            add(
                "rod_effect_reprobe",
                expected_effect_type=str(signature.get("rod_effect_profile", "") or "visible_effect"),
                max_probe_length=2,
                bailout_condition="stop if expected rod effect is not reproduced locally",
                evidence_keys=[
                    f"rod_effect_profile:{signature.get('rod_effect_profile', '')}",
                    "rod_contradiction:false",
                ],
                priority=0.62,
            )
        if not candidates:
            add(
                "generic_frontier_probe",
                expected_effect_type="unexplained_effect",
                max_probe_length=1,
                bailout_condition="stop after one action if no new visible effect appears",
                evidence_keys=[
                    f"operator:{operator}",
                    f"goal_status:{signature.get('goal_status', '')}",
                ],
                priority=0.40,
            )
        candidates.sort(
            key=lambda item: (
                -_as_float(item.get("priority"), 0.0),
                str(item.get("candidate_kind", "")),
            )
        )
        return candidates[:4]

    def expected_effect_supported(expected: str, observed: Mapping[str, Any]) -> bool:
        expected_norm = str(expected or "").strip()
        observed_effect = str(observed.get("effect_type", "") or "")
        score_or_goal = bool(
            _as_int(observed.get("score_delta"), 0) > 0
            or bool(observed.get("goal_progress"))
            or bool(observed.get("objective_progress"))
        )
        visible = _as_int(observed.get("changed_pixel_count"), 0) > 0
        if expected_norm == observed_effect:
            return True
        if expected_norm == "click_transition":
            return bool(visible or score_or_goal)
        if expected_norm == "route_connectivity_delta":
            return bool(score_or_goal)
        if expected_norm == "visible_effect":
            return bool(visible)
        if expected_norm == "unblocked_or_visible_effect":
            return bool(visible or score_or_goal)
        if expected_norm == "objective_progress":
            return bool(score_or_goal)
        if expected_norm == "unexplained_effect":
            return bool(visible and not score_or_goal)
        if "+" in expected_norm:
            expected_parts = {part for part in expected_norm.split("+") if part}
            return observed_effect in expected_parts or (
                "visible_effect" in expected_parts and visible
            )
        return False

    def probe_to_rod_preview(
        probe_candidate: Mapping[str, Any],
        *,
        observed: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected = str(probe_candidate.get("expected_effect_type", "") or "")
        supported = expected_effect_supported(expected, observed)
        no_op = bool(observed.get("no_op"))
        changed = _as_int(observed.get("changed_pixel_count"), 0)
        failure_count = _as_int(observed.get("failure_count"), 0)
        direct_failure_count = _as_int(
            observed.get("direct_score_probe_failure_count"),
            0,
        )
        if supported:
            verdict = "support"
        elif no_op and changed <= 0:
            verdict = "noop"
        elif direct_failure_count > 0:
            verdict = "contradiction"
        elif changed > 0:
            verdict = "unexplained_effect"
        elif failure_count > 0:
            verdict = "failure"
        else:
            verdict = "failure"
        candidate_kind = str(probe_candidate.get("candidate_kind", "") or "")
        observed_effect = str(observed.get("effect_type", "") or "unexplained_effect")
        generic_keys = sorted(
            set(
                [
                    f"probe:{candidate_kind}" if candidate_kind else "",
                    f"expected_effect:{expected}" if expected else "",
                    f"observed_effect:{observed_effect}" if observed_effect else "",
                    *[
                        str(key)
                        for key in list(probe_candidate.get("generic_evidence_keys") or [])
                        if str(key)
                    ],
                ]
            )
            - {""}
        )
        return {
            "schema_version": "eca_unsolved_probe_to_rod.v1",
            "source": "unsolved_probe_candidate",
            "candidate_kind": candidate_kind,
            "expected_effect_type": expected,
            "observed_effect_type": observed_effect,
            "verdict": verdict,
            "support_count": 1 if verdict == "support" else 0,
            "contradiction_count": 1 if verdict == "contradiction" else 0,
            "generic_keys": generic_keys,
            "target_binding": {
                "binding_source": "unsolved_probe_candidate",
                "commitment_level": str(
                    probe_candidate.get("commitment_level", "probe_low") or "probe_low"
                ),
                "candidate_kind": candidate_kind,
            },
            "observed_effect": {
                "effect_type": observed_effect,
                "score_delta": _as_int(observed.get("score_delta"), 0),
                "changed_pixel_count": _as_int(observed.get("changed_pixel_count"), 0),
                "goal_progress": bool(observed.get("goal_progress")),
                "objective_progress": bool(observed.get("objective_progress")),
                "no_op": bool(observed.get("no_op")),
                "direct_score_probe_failure_count": _as_int(
                    observed.get("direct_score_probe_failure_count"),
                    0,
                ),
            },
        }

    def payload_has_leakage(payload: Mapping[str, Any]) -> bool:
        text = str(_json_safe(payload)).lower()
        forbidden = set(PERCEPTION_ADAPTER_REGISTRY.keys()) | {
            "adapter",
            "canonical",
            "fixed_plan",
            "layout",
            "game_id",
        }
        return any(token and token in text for token in forbidden)

    def plan_steps_for_probe_candidate(probe_candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
        kind = str(probe_candidate.get("candidate_kind", "") or "")
        if kind == "click_probe_site":
            return [{"operator": "click", "action_template": "ACTION6", "max_repeats": 1}]
        if kind == "route_or_objective_frontier_probe":
            return [
                {
                    "operator": "route_to",
                    "action_template": "MOVE_TOWARD_TARGET_CANDIDATE",
                    "max_repeats": min(
                        3,
                        max(1, _as_int(probe_candidate.get("max_probe_length"), 3)),
                    ),
                }
            ]
        if kind == "visible_target_effect_probe":
            return [
                {
                    "operator": "interact",
                    "action_template": "TRY_VISIBLE_TARGET_ACTION",
                    "max_repeats": min(
                        2,
                        max(1, _as_int(probe_candidate.get("max_probe_length"), 2)),
                    ),
                }
            ]
        if kind == "escape_noop_loop_probe":
            return [
                {"operator": "escape", "action_template": "ALTERNATE_RECENT_ACTION", "max_repeats": 1},
                {"operator": "verify", "action_template": "CHECK_VISIBLE_OR_ROUTE_EFFECT", "max_repeats": 1},
            ]
        if kind == "indirect_score_precondition_probe":
            return [
                {
                    "operator": "probe_precondition",
                    "action_template": "MOVE_OR_INTERACT_NEAR_OBJECTIVE_CONTEXT",
                    "max_repeats": min(
                        3,
                        max(1, _as_int(probe_candidate.get("max_probe_length"), 3)),
                    ),
                }
            ]
        if kind == "rod_effect_reprobe":
            return [
                {
                    "operator": "reprobe",
                    "action_template": "REPRODUCE_SUPPORTED_EFFECT",
                    "max_repeats": min(
                        2,
                        max(1, _as_int(probe_candidate.get("max_probe_length"), 2)),
                    ),
                }
            ]
        return [{"operator": "frontier", "action_template": "TRY_LOW_COMMITMENT_ACTION", "max_repeats": 1}]

    def short_plan_confidence(preview: Mapping[str, Any], observed: Mapping[str, Any]) -> float:
        confidence = 0.45
        if str(preview.get("verdict", "") or "") == "support":
            confidence += 0.20
        if _as_int(observed.get("score_delta"), 0) > 0:
            confidence += 0.15
        if bool(observed.get("goal_progress")) or bool(observed.get("objective_progress")):
            confidence += 0.10
        if bool(observed.get("no_op")):
            confidence -= 0.10
        return round(max(0.0, min(0.90, confidence)), 6)

    def short_plan_candidates_from_previews(
        signature: Mapping[str, Any],
        probe_candidates: list[Mapping[str, Any]],
        previews: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        plans: list[dict[str, Any]] = []
        for probe_candidate, preview in zip(probe_candidates, previews):
            if str(preview.get("verdict", "") or "") != "support":
                continue
            if _as_int(preview.get("contradiction_count"), 0) > 0:
                continue
            observed = (
                preview.get("observed_effect")
                if isinstance(preview.get("observed_effect"), Mapping)
                else {}
            )
            plan_steps = plan_steps_for_probe_candidate(probe_candidate)
            max_plan_length = min(
                6,
                max(1, sum(_as_int(step.get("max_repeats"), 1) for step in plan_steps)),
            )
            candidate_kind = str(probe_candidate.get("candidate_kind", "") or "")
            observed_effect = str(preview.get("observed_effect_type", "") or "")
            expected_effect = str(probe_candidate.get("expected_effect_type", "") or observed_effect)
            supporting_keys = sorted(
                {
                    str(key)
                    for key in list(preview.get("generic_keys") or [])
                    if str(key)
                }
            )[:10]
            plans.append(
                {
                    "schema_version": "eca_unsolved_short_plan_candidate.v1",
                    "source": "eca_unsolved_replan",
                    "status": "candidate",
                    "plan_kind": candidate_kind or "generic_frontier_probe",
                    "commitment_level": "probe_low",
                    "source_state_key": str(signature.get("key", "") or ""),
                    "short_plan": plan_steps,
                    "max_plan_length": int(max_plan_length),
                    "expected_effect": expected_effect,
                    "observed_support_effect": observed_effect,
                    "supporting_rod_keys": supporting_keys,
                    "bailout_guard": str(probe_candidate.get("bailout_condition", "") or ""),
                    "validation_required": True,
                    "executable_authority": "diagnostic_only",
                    "confidence": short_plan_confidence(preview, observed),
                }
            )
        plans.sort(
            key=lambda item: (
                -_as_float(item.get("confidence"), 0.0),
                str(item.get("plan_kind", "")),
            )
        )
        return plans[:3]

    def short_plan_admission(plan: Mapping[str, Any]) -> dict[str, Any]:
        leakage = payload_has_leakage(plan)
        plan_steps = plan.get("short_plan")
        step_count = len(plan_steps) if isinstance(plan_steps, list) else 0
        max_length = _as_int(plan.get("max_plan_length"), 0)
        confidence = _as_float(plan.get("confidence"), 0.0)
        supporting_keys = [
            str(key)
            for key in list(plan.get("supporting_rod_keys") or [])
            if str(key)
        ]
        reason = "accepted"
        accepted = True
        if str(plan.get("schema_version", "") or "") != "eca_unsolved_short_plan_candidate.v1":
            reason = "bad_schema"
            accepted = False
        elif str(plan.get("source", "") or "") != "eca_unsolved_replan":
            reason = "wrong_source"
            accepted = False
        elif str(plan.get("executable_authority", "") or "") != "diagnostic_only":
            reason = "already_executable_or_unknown_authority"
            accepted = False
        elif str(plan.get("commitment_level", "") or "") != "probe_low":
            reason = "commitment_too_high"
            accepted = False
        elif not bool(plan.get("validation_required")):
            reason = "missing_validation_guard"
            accepted = False
        elif not str(plan.get("source_state_key", "") or ""):
            reason = "missing_state_key"
            accepted = False
        elif not supporting_keys:
            reason = "missing_supporting_rod_keys"
            accepted = False
        elif not str(plan.get("bailout_guard", "") or ""):
            reason = "missing_bailout_guard"
            accepted = False
        elif step_count <= 0 or max_length <= 0 or max_length > 6:
            reason = "invalid_plan_length"
            accepted = False
        elif confidence < 0.70:
            reason = "low_confidence"
            accepted = False
        elif leakage:
            reason = "payload_leakage"
            accepted = False
        return {
            "schema_version": "eca_unsolved_short_plan_admission.v1",
            "plan_kind": str(plan.get("plan_kind", "") or ""),
            "accepted": bool(accepted),
            "reason": reason,
            "confidence": confidence,
            "max_plan_length": int(max_length),
            "step_count": int(step_count),
            "commitment_level": str(plan.get("commitment_level", "") or ""),
            "next_authority": "guarded_candidate" if accepted else "diagnostic_only",
            "payload_leakage": bool(leakage),
        }

    def prefixed(counter: Counter[str], prefix: str) -> dict[str, int]:
        out: dict[str, int] = {}
        marker = f"{prefix}:"
        for key, value in sorted(counter.items()):
            if key.startswith(marker):
                out[key[len(marker) :]] = int(value)
        return out

    def observe(
        counter: Counter[str],
        *,
        item: Mapping[str, Any],
        advice: Mapping[str, Any],
        cache: Mapping[str, Any],
    ) -> tuple[bool, str]:
        bump(counter, "trace_step_count")
        reason = str(cache.get("reason", "") or "")
        source = str(cache.get("source", "") or "")
        source_unsolved = is_unsolved_source(source)
        if source_unsolved:
            bump(counter, "adapter_unsolved_step_count")
        if reason == "cache_empty":
            bump(counter, "cache_empty_step_count")
        if reason == "fresh_plan_cache_exhausted":
            bump(counter, "cache_exhausted_step_count")
        cache_disabled_no_plan = reason == "disabled" and source_unsolved
        if cache_disabled_no_plan:
            bump(counter, "cache_disabled_no_plan_step_count")
        no_plan_reason = (
            reason in {"cache_empty", "fresh_plan_cache_exhausted"}
            or cache_disabled_no_plan
        )
        if not no_plan_reason:
            return False, "has_plan_or_unrelated_cache_state"
        bump(counter, "candidate_count")
        if bool(cache.get("cache_action_available")) or bool(advice.get("available")):
            bump(counter, "blocked_count")
            bump(counter, "blocked_reason:valid_continuation_available")
            return False, "valid_continuation_available"
        if not source_unsolved:
            bump(counter, "blocked_count")
            bump(counter, "blocked_reason:adapter_source_not_unsolved")
            return False, "adapter_source_not_unsolved"
        status = goal_status(item)
        if status not in {"", "missing_objective", "progress_objective_only"}:
            bump(counter, "blocked_count")
            bump(counter, "blocked_reason:confident_goal_target")
            return False, "confident_goal_target"
        bump(counter, "eligible_count")
        return True, "eligible"

    def observe_runtime_guarded_execution(
        counter: Counter[str],
        *,
        item: Mapping[str, Any],
    ) -> None:
        payload = item.get("unsolved_replan")
        if not isinstance(payload, Mapping):
            return
        bump(counter, "guarded_runtime_trace_count")
        if bool(payload.get("enabled")):
            bump(counter, "guarded_runtime_enabled_count")
        if bool(payload.get("eligible")):
            bump(counter, "guarded_runtime_eligible_count")
        if bool(payload.get("admitted")):
            bump(counter, "guarded_runtime_admitted_count")
        if bool(payload.get("executed") or payload.get("used")):
            bump(counter, "guarded_runtime_executed_count")
        is_continuation = bool(payload.get("continuation"))
        if is_continuation:
            bump(counter, "guarded_runtime_effect_continuation_trace_count")
            trigger = str(payload.get("continuation_trigger", "") or "unknown")
            bump(counter, f"guarded_runtime_effect_continuation_trigger:{trigger}")
            if bool(payload.get("executed") or payload.get("used")):
                bump(counter, "guarded_runtime_effect_continuation_executed_count")
        is_closure = bool(payload.get("closure"))
        if is_closure:
            bump(counter, "guarded_runtime_objective_closure_trace_count")
            closure_mode = str(payload.get("closure_action_mode", "") or "unknown")
            closure_trigger = str(payload.get("closure_trigger", "") or "unknown")
            bump(counter, f"guarded_runtime_objective_closure_action_mode:{closure_mode}")
            bump(counter, f"guarded_runtime_objective_closure_trigger:{closure_trigger}")
            if bool(payload.get("closure_stall")):
                bump(counter, "guarded_runtime_objective_closure_stall_count")
            if bool(payload.get("executed") or payload.get("used")):
                bump(counter, "guarded_runtime_objective_closure_executed_count")
        is_post_transition_plan = bool(payload.get("post_transition_replanner"))
        if is_post_transition_plan:
            bump(counter, "guarded_runtime_post_transition_replanner_invoked_count")
            if isinstance(payload.get("current_state_plan_hypothesis"), Mapping):
                bump(counter, "guarded_runtime_post_transition_plan_generated_count")
                hypothesis = payload.get("current_state_plan_hypothesis")
                hypothesis = hypothesis if isinstance(hypothesis, Mapping) else {}
                goal_link_type = str(hypothesis.get("goal_link_type", "") or "")
                if goal_link_type:
                    bump(counter, "guarded_runtime_post_transition_goal_linked_plan_count")
                    bump(
                        counter,
                        f"guarded_runtime_post_transition_goal_link:{goal_link_type}",
                    )
                action_family = str(hypothesis.get("action_family", "") or "")
                if action_family:
                    bump(
                        counter,
                        f"guarded_runtime_post_transition_action_family:{action_family}",
                    )
            if bool(payload.get("executed") or payload.get("used")):
                bump(counter, "guarded_runtime_post_transition_plan_executed_count")
        is_state_transition_reprobe = bool(payload.get("state_transition_reprobe"))
        if is_state_transition_reprobe:
            bump(counter, "guarded_runtime_state_transition_reprobe_trace_count")
            if bool(payload.get("executed") or payload.get("used")):
                bump(counter, "guarded_runtime_state_transition_reprobe_executed_count")
        is_inverse_reprobe = bool(payload.get("state_transition_inverse_reprobe"))
        if is_inverse_reprobe:
            bump(counter, "guarded_runtime_inverse_reprobe_trace_count")
            inverse_source = str(payload.get("inverse_reprobe_source", "") or "unknown")
            bump(counter, f"guarded_runtime_inverse_reprobe_source:{inverse_source}")
            if bool(payload.get("executed") or payload.get("used")):
                bump(counter, "guarded_runtime_inverse_reprobe_executed_count")
        reason = str(payload.get("reason", "") or "unknown")
        bump(counter, f"guarded_runtime_reason:{reason}")
        if reason in {
            "repeated_guarded_probe_suppressed",
            "all_guarded_probe_candidates_suppressed",
            "all_guarded_probe_candidates_outcome_suppressed",
        }:
            bump(counter, "guarded_runtime_suppressed_count")
        if reason == "all_guarded_probe_candidates_outcome_suppressed":
            bump(counter, "guarded_runtime_outcome_suppressed_count")
        skipped = _as_int(payload.get("skipped_attempt_signature_count"), 0)
        if skipped > 0:
            bump(counter, "guarded_runtime_skipped_signature_count", skipped)
        outcome_suppressed = _as_int(payload.get("outcome_suppressed_candidate_count"), 0)
        if outcome_suppressed > 0:
            bump(
                counter,
                "guarded_runtime_outcome_suppressed_candidate_count",
                outcome_suppressed,
            )
        outcome_deprioritized = _as_int(
            payload.get("outcome_deprioritized_candidate_count"),
            0,
        )
        if outcome_deprioritized > 0:
            bump(
                counter,
                "guarded_runtime_outcome_deprioritized_candidate_count",
                outcome_deprioritized,
            )
        action_cooldown_candidates = _as_int(
            payload.get("action_cooldown_candidate_count"),
            0,
        )
        if action_cooldown_candidates > 0:
            bump(
                counter,
                "guarded_runtime_action_cooldown_candidate_count",
                action_cooldown_candidates,
            )
        if reason == "all_guarded_probe_candidates_direction_cooled_down":
            bump(counter, "guarded_runtime_action_cooldown_blocked_count")
        cooldown_after = payload.get("action_cooldown_after")
        cooldown_after = cooldown_after if isinstance(cooldown_after, Mapping) else {}
        if cooldown_after:
            bump(counter, "guarded_runtime_action_cooldown_set_count")
            action = _normalize_action_name(cooldown_after.get("action"))
            if action:
                bump(counter, f"guarded_runtime_action_cooldown_action:{action}")
            reason_after = str(cooldown_after.get("reason", "") or "unknown")
            bump(counter, f"guarded_runtime_action_cooldown_reason:{reason_after}")
        deprioritization_reason = str(
            payload.get("outcome_deprioritization_reason", "") or ""
        )
        if deprioritization_reason:
            bump(counter, "guarded_runtime_outcome_deprioritized_execution_count")
            bump(
                counter,
                f"guarded_runtime_outcome_deprioritization_reason:{deprioritization_reason}",
            )
        if isinstance(payload.get("outcome_ledger_prior"), Mapping) and payload.get(
            "outcome_ledger_prior"
        ):
            bump(counter, "guarded_runtime_ledger_prior_count")
        if isinstance(payload.get("outcome_ledger_after"), Mapping) and payload.get(
            "outcome_ledger_after"
        ):
            bump(counter, "guarded_runtime_ledger_update_count")
        if str(payload.get("outcome_feedback_key", "") or ""):
            bump(counter, "guarded_runtime_feedback_key_count")
        candidate_kind = str(payload.get("candidate_kind", "") or "")
        if candidate_kind:
            bump(counter, f"guarded_runtime_candidate_kind:{candidate_kind}")
        action = _normalize_action_name(payload.get("selected_action") or item.get("action"))
        if action and bool(payload.get("executed") or payload.get("used")):
            bump(counter, f"guarded_runtime_action:{action}")
        outcome = payload.get("runtime_outcome")
        outcome = outcome if isinstance(outcome, Mapping) else {}
        if outcome:
            bump(counter, "guarded_runtime_outcome_count")
            verdict = str(outcome.get("verdict", "") or payload.get("verdict") or "unknown")
            effect_type = str(outcome.get("effect_type", "") or "unknown")
            bump(counter, f"guarded_runtime_verdict:{verdict}")
            bump(counter, f"guarded_runtime_effect:{effect_type}")
            bump(
                counter,
                "guarded_runtime_score_delta_sum",
                _as_int(outcome.get("score_delta"), 0),
            )
            bump(
                counter,
                "guarded_runtime_changed_pixel_count_sum",
                _as_int(outcome.get("changed_pixel_count"), 0),
            )
            if bool(outcome.get("no_op")):
                bump(counter, "guarded_runtime_noop_count")
            if bool(outcome.get("state_changed")):
                bump(counter, "guarded_runtime_state_changed_count")
            if bool(outcome.get("structural_progress")):
                bump(counter, "guarded_runtime_structural_progress_count")
            if bool(outcome.get("objective_progress")):
                bump(counter, "guarded_runtime_objective_progress_count")
            if bool(outcome.get("goal_aligned_progress")):
                bump(counter, "guarded_runtime_goal_aligned_progress_count")
                reason = str(outcome.get("goal_aligned_progress_reason", "") or "unknown")
                bump(counter, f"guarded_runtime_goal_aligned_progress_reason:{reason}")
            if bool(outcome.get("motion_action_aligned")):
                bump(counter, "guarded_runtime_motion_action_aligned_count")
            proximity_delta = _as_float(outcome.get("objective_proximity_delta"), 0.0)
            if proximity_delta > 0:
                bump(counter, "guarded_runtime_objective_proximity_positive_count")
                counter["guarded_runtime_objective_proximity_delta_sum"] += proximity_delta
            elif proximity_delta < 0:
                bump(counter, "guarded_runtime_objective_proximity_negative_count")
                counter["guarded_runtime_objective_proximity_delta_sum"] += proximity_delta
            if bool(outcome.get("effect_supported")):
                bump(counter, "guarded_runtime_effect_supported_count")
                if is_continuation:
                    bump(counter, "guarded_runtime_effect_continuation_effect_supported_count")
                if is_closure:
                    bump(counter, "guarded_runtime_objective_closure_effect_supported_count")
                if is_post_transition_plan:
                    bump(counter, "guarded_runtime_post_transition_plan_validated_count")
                if is_state_transition_reprobe:
                    bump(counter, "guarded_runtime_state_transition_reprobe_effect_supported_count")
                if is_inverse_reprobe:
                    bump(counter, "guarded_runtime_inverse_reprobe_effect_supported_count")
            if bool(outcome.get("goal_bearing")):
                bump(counter, "guarded_runtime_goal_bearing_count")
                if is_continuation:
                    bump(counter, "guarded_runtime_effect_continuation_goal_bearing_count")
                if is_closure:
                    bump(counter, "guarded_runtime_objective_closure_goal_bearing_count")
                if is_post_transition_plan:
                    bump(counter, "guarded_runtime_post_transition_goal_bearing_count")
                if is_state_transition_reprobe:
                    bump(counter, "guarded_runtime_state_transition_reprobe_goal_bearing_count")
                if is_inverse_reprobe:
                    bump(counter, "guarded_runtime_inverse_reprobe_goal_bearing_count")
            if is_continuation:
                bump(counter, "guarded_runtime_effect_continuation_outcome_count")
                bump(
                    counter,
                    "guarded_runtime_effect_continuation_score_delta_sum",
                    _as_int(outcome.get("score_delta"), 0),
                )
            if is_closure:
                bump(counter, "guarded_runtime_objective_closure_outcome_count")
                bump(
                    counter,
                    "guarded_runtime_objective_closure_score_delta_sum",
                    _as_int(outcome.get("score_delta"), 0),
                )
            if is_post_transition_plan:
                bump(counter, "guarded_runtime_post_transition_plan_outcome_count")
                bump(
                    counter,
                    "guarded_runtime_post_transition_score_delta_sum",
                    _as_int(outcome.get("score_delta"), 0),
                )
            if is_state_transition_reprobe:
                bump(counter, "guarded_runtime_state_transition_reprobe_outcome_count")
                bump(
                    counter,
                    "guarded_runtime_state_transition_reprobe_score_delta_sum",
                    _as_int(outcome.get("score_delta"), 0),
                )
            if is_inverse_reprobe:
                bump(counter, "guarded_runtime_inverse_reprobe_outcome_count")
                bump(
                    counter,
                    "guarded_runtime_inverse_reprobe_score_delta_sum",
                    _as_int(outcome.get("score_delta"), 0),
                )
        closure_result = payload.get("closure_result")
        closure_result = closure_result if isinstance(closure_result, Mapping) else {}
        if closure_result:
            bump(counter, "guarded_runtime_objective_closure_result_count")
            bump(
                counter,
                f"guarded_runtime_objective_closure_result:{closure_result.get('result') or 'unknown'}",
            )
            bump(
                counter,
                f"guarded_runtime_objective_closure_result_mode:{closure_result.get('action_mode') or 'unknown'}",
            )
            bump(
                counter,
                f"guarded_runtime_objective_closure_semantic_role:{closure_result.get('semantic_role') or 'unknown'}",
            )
            if bool(closure_result.get("state_transition_candidate")):
                bump(counter, "guarded_runtime_objective_closure_state_transition_candidate_count")

    for row in rows:
        mode = str(row.get("adapter_advice_mode", "") or "none")
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            observe_runtime_guarded_execution(counts, item=item)
            observe_runtime_guarded_execution(by_mode[mode], item=item)
            observe_runtime_guarded_execution(by_game[game_id], item=item)
            advice = item.get("adapter_advice")
            if not isinstance(advice, Mapping):
                continue
            cache = advice.get("plan_cache")
            if not isinstance(cache, Mapping):
                continue
            eligible, outcome = observe(counts, item=item, advice=advice, cache=cache)
            observe(by_mode[mode], item=item, advice=advice, cache=cache)
            observe(by_game[game_id], item=item, advice=advice, cache=cache)
            signature: dict[str, Any] | None = None
            if outcome in {
                "eligible",
                "valid_continuation_available",
                "adapter_source_not_unsolved",
                "confident_goal_target",
            }:
                signature = current_state_signature(item, trace, index)
                candidate_signature_counts[str(signature.get("key", ""))] += 1
                if eligible:
                    eligible_signature_counts[str(signature.get("key", ""))] += 1
                    probe_candidates = probe_candidates_from_signature(signature)
                    if probe_candidates:
                        eligible_steps_with_probe_candidates += 1
                    probe_candidate_total += len(probe_candidates)
                    observed = observed_probe_effect(item)
                    previews: list[dict[str, Any]] = []
                    for probe_candidate in probe_candidates:
                        probe_candidate_counts[
                            str(probe_candidate.get("candidate_kind", "") or "unknown")
                        ] += 1
                        probe_effect_counts[
                            str(probe_candidate.get("expected_effect_type", "") or "unknown")
                        ] += 1
                        preview = probe_to_rod_preview(
                            probe_candidate,
                            observed=observed,
                        )
                        previews.append(preview)
                        probe_to_rod_preview_total += 1
                        probe_to_rod_verdict_counts[
                            str(preview.get("verdict", "") or "unknown")
                        ] += 1
                        probe_to_rod_effect_counts[
                            str(preview.get("observed_effect_type", "") or "unknown")
                        ] += 1
                        if payload_has_leakage(preview):
                            probe_to_rod_payload_leakage_count += 1
                    short_plans = short_plan_candidates_from_previews(
                        signature,
                        probe_candidates,
                        previews,
                    )
                    if short_plans:
                        eligible_steps_with_short_plan_candidates += 1
                    short_plan_candidate_total += len(short_plans)
                    for plan in short_plans:
                        short_plan_kind_counts[
                            str(plan.get("plan_kind", "") or "unknown")
                        ] += 1
                        short_plan_effect_counts[
                            str(plan.get("expected_effect", "") or "unknown")
                        ] += 1
                        if payload_has_leakage(plan):
                            short_plan_payload_leakage_count += 1
                    admissions = [short_plan_admission(plan) for plan in short_plans]
                    if any(bool(item.get("accepted")) for item in admissions):
                        eligible_steps_with_admitted_short_plan += 1
                    short_plan_admission_eval_total += len(admissions)
                    short_plan_admission_accepted_total += sum(
                        1 for item in admissions if bool(item.get("accepted"))
                    )
                    for admission in admissions:
                        reason = str(admission.get("reason", "") or "unknown")
                        short_plan_admission_reason_counts[reason] += 1
                        if bool(admission.get("accepted")):
                            short_plan_admission_kind_counts[
                                str(admission.get("plan_kind", "") or "unknown")
                            ] += 1
                        if bool(admission.get("payload_leakage")):
                            short_plan_admission_payload_leakage_count += 1
                if signature_has_leakage(signature):
                    signature_leakage_count += 1
            if outcome in {"eligible", "valid_continuation_available", "adapter_source_not_unsolved", "confident_goal_target"}:
                if len(samples) < sample_limit:
                    sample_probe_candidates = (
                        probe_candidates_from_signature(signature or {})
                        if eligible and signature is not None
                        else []
                    )
                    sample_observed = observed_probe_effect(item) if eligible else {}
                    sample_probe_to_rod_previews = [
                        probe_to_rod_preview(candidate, observed=sample_observed)
                        for candidate in sample_probe_candidates
                    ]
                    sample_short_plan_candidates = (
                        short_plan_candidates_from_previews(
                            signature or {},
                            sample_probe_candidates,
                            sample_probe_to_rod_previews,
                        )
                        if eligible and signature is not None
                        else []
                    )
                    sample_short_plan_admissions = [
                        short_plan_admission(plan)
                        for plan in sample_short_plan_candidates
                    ]
                    samples.append(
                        {
                            "game_id": game_id,
                            "variant": str(row.get("variant", "") or ""),
                            "mode": mode,
                            "step": _as_int(item.get("step"), -1),
                            "eligible": bool(eligible),
                            "outcome": outcome,
                            "cache_reason": str(cache.get("reason", "") or ""),
                            "adapter_source": str(cache.get("source", "") or ""),
                            "goal_status": goal_status(item),
                            "action": str(item.get("action", "") or ""),
                            "current_state_key": str(
                                (signature or {}).get("key", "")
                            ),
                            "current_state_signature": dict(signature or {}),
                            "probe_candidates": sample_probe_candidates,
                            "probe_to_rod_previews": sample_probe_to_rod_previews,
                            "short_plan_candidates": sample_short_plan_candidates,
                            "short_plan_admissions": sample_short_plan_admissions,
                        }
                    )

    def convert(counter: Counter[str]) -> dict[str, Any]:
        trace_steps = int(counter.get("trace_step_count", 0))
        candidates = int(counter.get("candidate_count", 0))
        eligible = int(counter.get("eligible_count", 0))
        blocked = int(counter.get("blocked_count", 0))
        return {
            "trace_step_count": trace_steps,
            "adapter_unsolved_step_count": int(counter.get("adapter_unsolved_step_count", 0)),
            "cache_empty_step_count": int(counter.get("cache_empty_step_count", 0)),
            "cache_exhausted_step_count": int(counter.get("cache_exhausted_step_count", 0)),
            "cache_disabled_no_plan_step_count": int(
                counter.get("cache_disabled_no_plan_step_count", 0)
            ),
            "candidate_count": candidates,
            "eligible_count": eligible,
            "blocked_count": blocked,
            "eligible_rate": _safe_div(eligible, candidates),
            "blocked_reasons": prefixed(counter, "blocked_reason"),
            "guarded_runtime_trace_count": int(
                counter.get("guarded_runtime_trace_count", 0)
            ),
            "guarded_runtime_enabled_count": int(
                counter.get("guarded_runtime_enabled_count", 0)
            ),
            "guarded_runtime_eligible_count": int(
                counter.get("guarded_runtime_eligible_count", 0)
            ),
            "guarded_runtime_admitted_count": int(
                counter.get("guarded_runtime_admitted_count", 0)
            ),
            "guarded_runtime_executed_count": int(
                counter.get("guarded_runtime_executed_count", 0)
            ),
            "guarded_runtime_effect_continuation_trace_count": int(
                counter.get("guarded_runtime_effect_continuation_trace_count", 0)
            ),
            "guarded_runtime_effect_continuation_executed_count": int(
                counter.get("guarded_runtime_effect_continuation_executed_count", 0)
            ),
            "guarded_runtime_effect_continuation_outcome_count": int(
                counter.get("guarded_runtime_effect_continuation_outcome_count", 0)
            ),
            "guarded_runtime_effect_continuation_effect_supported_count": int(
                counter.get("guarded_runtime_effect_continuation_effect_supported_count", 0)
            ),
            "guarded_runtime_effect_continuation_goal_bearing_count": int(
                counter.get("guarded_runtime_effect_continuation_goal_bearing_count", 0)
            ),
            "guarded_runtime_effect_continuation_score_delta_sum": int(
                counter.get("guarded_runtime_effect_continuation_score_delta_sum", 0)
            ),
            "guarded_runtime_effect_continuation_trigger_counts": prefixed(
                counter,
                "guarded_runtime_effect_continuation_trigger",
            ),
            "guarded_runtime_objective_closure_trace_count": int(
                counter.get("guarded_runtime_objective_closure_trace_count", 0)
            ),
            "guarded_runtime_objective_closure_executed_count": int(
                counter.get("guarded_runtime_objective_closure_executed_count", 0)
            ),
            "guarded_runtime_objective_closure_outcome_count": int(
                counter.get("guarded_runtime_objective_closure_outcome_count", 0)
            ),
            "guarded_runtime_objective_closure_effect_supported_count": int(
                counter.get("guarded_runtime_objective_closure_effect_supported_count", 0)
            ),
            "guarded_runtime_objective_closure_goal_bearing_count": int(
                counter.get("guarded_runtime_objective_closure_goal_bearing_count", 0)
            ),
            "guarded_runtime_objective_closure_score_delta_sum": int(
                counter.get("guarded_runtime_objective_closure_score_delta_sum", 0)
            ),
            "guarded_runtime_objective_closure_action_mode_counts": prefixed(
                counter,
                "guarded_runtime_objective_closure_action_mode",
            ),
            "guarded_runtime_objective_closure_trigger_counts": prefixed(
                counter,
                "guarded_runtime_objective_closure_trigger",
            ),
            "guarded_runtime_objective_closure_stall_count": int(
                counter.get("guarded_runtime_objective_closure_stall_count", 0)
            ),
            "guarded_runtime_objective_closure_result_count": int(
                counter.get("guarded_runtime_objective_closure_result_count", 0)
            ),
            "guarded_runtime_objective_closure_result_counts": prefixed(
                counter,
                "guarded_runtime_objective_closure_result",
            ),
            "guarded_runtime_objective_closure_result_mode_counts": prefixed(
                counter,
                "guarded_runtime_objective_closure_result_mode",
            ),
            "guarded_runtime_objective_closure_semantic_role_counts": prefixed(
                counter,
                "guarded_runtime_objective_closure_semantic_role",
            ),
            "guarded_runtime_objective_closure_state_transition_candidate_count": int(
                counter.get(
                    "guarded_runtime_objective_closure_state_transition_candidate_count",
                    0,
                )
            ),
            "guarded_runtime_post_transition_replanner_invoked_count": int(
                counter.get("guarded_runtime_post_transition_replanner_invoked_count", 0)
            ),
            "guarded_runtime_post_transition_plan_generated_count": int(
                counter.get("guarded_runtime_post_transition_plan_generated_count", 0)
            ),
            "guarded_runtime_post_transition_plan_executed_count": int(
                counter.get("guarded_runtime_post_transition_plan_executed_count", 0)
            ),
            "guarded_runtime_post_transition_plan_outcome_count": int(
                counter.get("guarded_runtime_post_transition_plan_outcome_count", 0)
            ),
            "guarded_runtime_post_transition_plan_validated_count": int(
                counter.get("guarded_runtime_post_transition_plan_validated_count", 0)
            ),
            "guarded_runtime_post_transition_goal_linked_plan_count": int(
                counter.get("guarded_runtime_post_transition_goal_linked_plan_count", 0)
            ),
            "guarded_runtime_post_transition_goal_bearing_count": int(
                counter.get("guarded_runtime_post_transition_goal_bearing_count", 0)
            ),
            "guarded_runtime_post_transition_score_delta_sum": int(
                counter.get("guarded_runtime_post_transition_score_delta_sum", 0)
            ),
            "guarded_runtime_post_transition_goal_link_counts": prefixed(
                counter,
                "guarded_runtime_post_transition_goal_link",
            ),
            "guarded_runtime_post_transition_action_family_counts": prefixed(
                counter,
                "guarded_runtime_post_transition_action_family",
            ),
            "guarded_runtime_state_transition_reprobe_trace_count": int(
                counter.get("guarded_runtime_state_transition_reprobe_trace_count", 0)
            ),
            "guarded_runtime_state_transition_reprobe_executed_count": int(
                counter.get("guarded_runtime_state_transition_reprobe_executed_count", 0)
            ),
            "guarded_runtime_state_transition_reprobe_outcome_count": int(
                counter.get("guarded_runtime_state_transition_reprobe_outcome_count", 0)
            ),
            "guarded_runtime_state_transition_reprobe_effect_supported_count": int(
                counter.get(
                    "guarded_runtime_state_transition_reprobe_effect_supported_count",
                    0,
                )
            ),
            "guarded_runtime_state_transition_reprobe_goal_bearing_count": int(
                counter.get(
                    "guarded_runtime_state_transition_reprobe_goal_bearing_count",
                    0,
                )
            ),
            "guarded_runtime_state_transition_reprobe_score_delta_sum": int(
                counter.get("guarded_runtime_state_transition_reprobe_score_delta_sum", 0)
            ),
            "guarded_runtime_inverse_reprobe_trace_count": int(
                counter.get("guarded_runtime_inverse_reprobe_trace_count", 0)
            ),
            "guarded_runtime_inverse_reprobe_executed_count": int(
                counter.get("guarded_runtime_inverse_reprobe_executed_count", 0)
            ),
            "guarded_runtime_inverse_reprobe_outcome_count": int(
                counter.get("guarded_runtime_inverse_reprobe_outcome_count", 0)
            ),
            "guarded_runtime_inverse_reprobe_effect_supported_count": int(
                counter.get("guarded_runtime_inverse_reprobe_effect_supported_count", 0)
            ),
            "guarded_runtime_inverse_reprobe_goal_bearing_count": int(
                counter.get("guarded_runtime_inverse_reprobe_goal_bearing_count", 0)
            ),
            "guarded_runtime_inverse_reprobe_score_delta_sum": int(
                counter.get("guarded_runtime_inverse_reprobe_score_delta_sum", 0)
            ),
            "guarded_runtime_inverse_reprobe_source_counts": prefixed(
                counter,
                "guarded_runtime_inverse_reprobe_source",
            ),
            "guarded_runtime_suppressed_count": int(
                counter.get("guarded_runtime_suppressed_count", 0)
            ),
            "guarded_runtime_skipped_signature_count": int(
                counter.get("guarded_runtime_skipped_signature_count", 0)
            ),
            "guarded_runtime_outcome_suppressed_count": int(
                counter.get("guarded_runtime_outcome_suppressed_count", 0)
            ),
            "guarded_runtime_outcome_suppressed_candidate_count": int(
                counter.get("guarded_runtime_outcome_suppressed_candidate_count", 0)
            ),
            "guarded_runtime_outcome_deprioritized_candidate_count": int(
                counter.get("guarded_runtime_outcome_deprioritized_candidate_count", 0)
            ),
            "guarded_runtime_outcome_deprioritized_execution_count": int(
                counter.get("guarded_runtime_outcome_deprioritized_execution_count", 0)
            ),
            "guarded_runtime_action_cooldown_candidate_count": int(
                counter.get("guarded_runtime_action_cooldown_candidate_count", 0)
            ),
            "guarded_runtime_action_cooldown_blocked_count": int(
                counter.get("guarded_runtime_action_cooldown_blocked_count", 0)
            ),
            "guarded_runtime_action_cooldown_set_count": int(
                counter.get("guarded_runtime_action_cooldown_set_count", 0)
            ),
            "guarded_runtime_action_cooldown_action_counts": prefixed(
                counter,
                "guarded_runtime_action_cooldown_action",
            ),
            "guarded_runtime_action_cooldown_reason_counts": prefixed(
                counter,
                "guarded_runtime_action_cooldown_reason",
            ),
            "guarded_runtime_outcome_deprioritization_reason_counts": prefixed(
                counter,
                "guarded_runtime_outcome_deprioritization_reason",
            ),
            "guarded_runtime_ledger_prior_count": int(
                counter.get("guarded_runtime_ledger_prior_count", 0)
            ),
            "guarded_runtime_ledger_update_count": int(
                counter.get("guarded_runtime_ledger_update_count", 0)
            ),
            "guarded_runtime_feedback_key_count": int(
                counter.get("guarded_runtime_feedback_key_count", 0)
            ),
            "guarded_runtime_outcome_count": int(
                counter.get("guarded_runtime_outcome_count", 0)
            ),
            "guarded_runtime_score_delta_sum": int(
                counter.get("guarded_runtime_score_delta_sum", 0)
            ),
            "guarded_runtime_changed_pixel_count_sum": int(
                counter.get("guarded_runtime_changed_pixel_count_sum", 0)
            ),
            "guarded_runtime_noop_count": int(
                counter.get("guarded_runtime_noop_count", 0)
            ),
            "guarded_runtime_state_changed_count": int(
                counter.get("guarded_runtime_state_changed_count", 0)
            ),
            "guarded_runtime_structural_progress_count": int(
                counter.get("guarded_runtime_structural_progress_count", 0)
            ),
            "guarded_runtime_objective_progress_count": int(
                counter.get("guarded_runtime_objective_progress_count", 0)
            ),
            "guarded_runtime_goal_aligned_progress_count": int(
                counter.get("guarded_runtime_goal_aligned_progress_count", 0)
            ),
            "guarded_runtime_goal_aligned_progress_reason_counts": prefixed(
                counter,
                "guarded_runtime_goal_aligned_progress_reason",
            ),
            "guarded_runtime_motion_action_aligned_count": int(
                counter.get("guarded_runtime_motion_action_aligned_count", 0)
            ),
            "guarded_runtime_objective_proximity_positive_count": int(
                counter.get("guarded_runtime_objective_proximity_positive_count", 0)
            ),
            "guarded_runtime_objective_proximity_negative_count": int(
                counter.get("guarded_runtime_objective_proximity_negative_count", 0)
            ),
            "guarded_runtime_objective_proximity_delta_sum": round(
                float(counter.get("guarded_runtime_objective_proximity_delta_sum", 0.0)),
                6,
            ),
            "guarded_runtime_effect_supported_count": int(
                counter.get("guarded_runtime_effect_supported_count", 0)
            ),
            "guarded_runtime_goal_bearing_count": int(
                counter.get("guarded_runtime_goal_bearing_count", 0)
            ),
            "guarded_runtime_reason_counts": prefixed(counter, "guarded_runtime_reason"),
            "guarded_runtime_candidate_kind_counts": prefixed(
                counter,
                "guarded_runtime_candidate_kind",
            ),
            "guarded_runtime_action_counts": prefixed(counter, "guarded_runtime_action"),
            "guarded_runtime_verdict_counts": prefixed(counter, "guarded_runtime_verdict"),
            "guarded_runtime_effect_counts": prefixed(counter, "guarded_runtime_effect"),
        }

    summary = convert(counts)
    summary["schema_version"] = "eca_unsolved_to_replan_summary.v1"
    summary["current_state_signature_schema_version"] = (
        "eca_unsolved_current_state_signature.v1"
    )
    summary["candidate_current_state_signature_counts"] = dict(
        candidate_signature_counts.most_common(20)
    )
    summary["eligible_current_state_signature_counts"] = dict(
        eligible_signature_counts.most_common(20)
    )
    summary["current_state_signature_leakage_count"] = int(signature_leakage_count)
    summary["probe_candidate_schema_version"] = "eca_unsolved_probe_candidate.v1"
    summary["probe_candidate_count"] = int(probe_candidate_total)
    summary["eligible_step_with_probe_candidate_count"] = int(
        eligible_steps_with_probe_candidates
    )
    summary["probe_candidate_counts"] = dict(sorted(probe_candidate_counts.items()))
    summary["probe_expected_effect_counts"] = dict(sorted(probe_effect_counts.items()))
    summary["probe_to_rod_schema_version"] = "eca_unsolved_probe_to_rod.v1"
    summary["probe_to_rod_preview_count"] = int(probe_to_rod_preview_total)
    summary["probe_to_rod_verdict_counts"] = dict(
        sorted(probe_to_rod_verdict_counts.items())
    )
    summary["probe_to_rod_observed_effect_counts"] = dict(
        sorted(probe_to_rod_effect_counts.items())
    )
    summary["probe_to_rod_payload_leakage_count"] = int(
        probe_to_rod_payload_leakage_count
    )
    summary["short_plan_candidate_schema_version"] = (
        "eca_unsolved_short_plan_candidate.v1"
    )
    summary["short_plan_candidate_count"] = int(short_plan_candidate_total)
    summary["eligible_step_with_short_plan_candidate_count"] = int(
        eligible_steps_with_short_plan_candidates
    )
    summary["short_plan_kind_counts"] = dict(sorted(short_plan_kind_counts.items()))
    summary["short_plan_expected_effect_counts"] = dict(
        sorted(short_plan_effect_counts.items())
    )
    summary["short_plan_payload_leakage_count"] = int(short_plan_payload_leakage_count)
    summary["short_plan_admission_schema_version"] = (
        "eca_unsolved_short_plan_admission.v1"
    )
    summary["short_plan_admission_eval_count"] = int(short_plan_admission_eval_total)
    summary["short_plan_admission_accepted_count"] = int(
        short_plan_admission_accepted_total
    )
    summary["short_plan_admission_blocked_count"] = int(
        short_plan_admission_eval_total - short_plan_admission_accepted_total
    )
    summary["eligible_step_with_admitted_short_plan_count"] = int(
        eligible_steps_with_admitted_short_plan
    )
    summary["short_plan_admission_reason_counts"] = dict(
        sorted(short_plan_admission_reason_counts.items())
    )
    summary["short_plan_admission_accepted_kind_counts"] = dict(
        sorted(short_plan_admission_kind_counts.items())
    )
    summary["short_plan_admission_payload_leakage_count"] = int(
        short_plan_admission_payload_leakage_count
    )
    summary["samples"] = samples
    summary["by_mode"] = {key: convert(value) for key, value in sorted(by_mode.items())}
    summary["by_game"] = {key: convert(value) for key, value in sorted(by_game.items())}
    return summary


def _adapter_advice_sequence_replay_audit(
    rows: Iterable[Mapping[str, Any]],
    sample_limit: int = 40,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []

    def bump(counter: Counter[str], key: str, amount: int = 1) -> None:
        counter[key] += int(amount)

    def observe(counter: Counter[str], advice: Mapping[str, Any], replay: Mapping[str, Any]) -> None:
        if bool(advice.get("available")):
            bump(counter, "advice_available_count")
        sequence = advice.get("sequence")
        if isinstance(sequence, list) and sequence:
            bump(counter, "sequence_available_count")
        tail = advice.get("sequence_tail")
        if isinstance(tail, list) and tail:
            bump(counter, "sequence_tail_available_count")
        status = str(replay.get("status", "") or "unknown")
        bump(counter, f"status:{status}")
        if status in {
            "matched",
            "action_mismatch",
            "payload_mismatch",
            "current_advice_unavailable",
        }:
            bump(counter, "replay_comparison_count")
        if status == "matched":
            bump(counter, "replay_match_count")
        elif status == "action_mismatch":
            bump(counter, "action_mismatch_count")
            bump(counter, "replay_mismatch_count")
        elif status == "payload_mismatch":
            bump(counter, "payload_mismatch_count")
            bump(counter, "replay_mismatch_count")
        elif status == "current_advice_unavailable":
            bump(counter, "current_advice_unavailable_count")
        elif status == "injection_interrupted":
            bump(counter, "injection_interrupted_count")
        elif status == "cache_boundary":
            bump(counter, "cache_boundary_count")

    for row in rows:
        mode = str(row.get("adapter_advice_mode", "") or "none")
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            counts["trace_step_count"] += 1
            by_mode[mode]["trace_step_count"] += 1
            by_game[game_id]["trace_step_count"] += 1
            advice = item.get("adapter_advice")
            if not isinstance(advice, Mapping):
                continue
            replay = advice.get("sequence_replay")
            if not isinstance(replay, Mapping):
                continue
            observe(counts, advice, replay)
            observe(by_mode[mode], advice, replay)
            observe(by_game[game_id], advice, replay)
            status = str(replay.get("status", "") or "unknown")
            if (
                len(samples) < sample_limit
                and status
                not in {
                    "matched",
                    "no_prior_advice",
                    "prior_tail_empty",
                    "cache_boundary",
                }
            ):
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "mode": mode,
                        "step": _as_int(item.get("step"), 0),
                        "phase": str(item.get("phase", "") or ""),
                        "status": status,
                        "expected_action": str(replay.get("expected_action", "") or ""),
                        "actual_action": str(replay.get("actual_action", "") or ""),
                        "expected_payload": _json_safe(replay.get("expected_payload")),
                        "actual_payload": _json_safe(replay.get("actual_payload")),
                        "advice_used": bool(advice.get("used")),
                        "decision_reason": str(advice.get("decision_reason", "") or ""),
                        "plan_cache": _json_safe(advice.get("plan_cache")),
                    }
                )

    def status_counts(counter: Counter[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, value in sorted(counter.items()):
            if key.startswith("status:"):
                out[key.split(":", 1)[1]] = int(value)
        return out

    def convert(counter: Counter[str]) -> dict[str, Any]:
        comparisons = int(counter.get("replay_comparison_count", 0))
        matches = int(counter.get("replay_match_count", 0))
        return {
            "trace_step_count": int(counter.get("trace_step_count", 0)),
            "advice_available_count": int(counter.get("advice_available_count", 0)),
            "sequence_available_count": int(counter.get("sequence_available_count", 0)),
            "sequence_tail_available_count": int(
                counter.get("sequence_tail_available_count", 0)
            ),
            "replay_comparison_count": comparisons,
            "replay_match_count": matches,
            "replay_mismatch_count": int(counter.get("replay_mismatch_count", 0)),
            "replay_match_rate": _safe_div(matches, comparisons),
            "action_mismatch_count": int(counter.get("action_mismatch_count", 0)),
            "payload_mismatch_count": int(counter.get("payload_mismatch_count", 0)),
            "injection_interrupted_count": int(
                counter.get("injection_interrupted_count", 0)
            ),
            "cache_boundary_count": int(counter.get("cache_boundary_count", 0)),
            "current_advice_unavailable_count": int(
                counter.get("current_advice_unavailable_count", 0)
            ),
            "by_status": status_counts(counter),
        }

    summary = convert(counts)
    summary["by_mode"] = {key: convert(value) for key, value in sorted(by_mode.items())}
    summary["by_game"] = {key: convert(value) for key, value in sorted(by_game.items())}
    summary["samples"] = samples
    return summary


def _adapter_advice_execution_replay_audit(
    rows: Iterable[Mapping[str, Any]],
    sample_limit: int = 40,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    by_decision_reason: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []

    def bump(counter: Counter[str], key: str, amount: int = 1) -> None:
        counter[key] += int(amount)

    def observe(counter: Counter[str], advice: Mapping[str, Any]) -> None:
        if bool(advice.get("available")):
            bump(counter, "advice_available_count")
        replay = advice.get("execution_replay")
        if not isinstance(replay, Mapping):
            return
        status = str(replay.get("status", "") or "unknown")
        bump(counter, f"status:{status}")
        if status in {"matched", "action_mismatch", "payload_mismatch"}:
            bump(counter, "execution_comparison_count")
        if status == "matched":
            bump(counter, "execution_match_count")
        elif status == "action_mismatch":
            bump(counter, "execution_action_mismatch_count")
            bump(counter, "execution_mismatch_count")
        elif status == "payload_mismatch":
            bump(counter, "execution_payload_mismatch_count")
            bump(counter, "execution_mismatch_count")
        elif status == "injection_interrupted":
            bump(counter, "injection_interrupted_count")
        commit = advice.get("plan_commit")
        if isinstance(commit, Mapping):
            bump(counter, "plan_commit_event_count")
            if bool(commit.get("committed")):
                bump(counter, "plan_committed_count")
            reason = str(commit.get("reason", "") or "unknown")
            bump(counter, f"commit_reason:{reason}")

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        mode = str(row.get("adapter_advice_mode", "") or "none")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            counts["trace_step_count"] += 1
            by_game[game_id]["trace_step_count"] += 1
            by_mode[mode]["trace_step_count"] += 1
            advice = item.get("adapter_advice")
            if not isinstance(advice, Mapping):
                continue
            reason = str(advice.get("decision_reason", "") or "unknown")
            observe(counts, advice)
            observe(by_game[game_id], advice)
            observe(by_mode[mode], advice)
            observe(by_decision_reason[reason], advice)
            replay = advice.get("execution_replay")
            status = str(replay.get("status", "") or "") if isinstance(replay, Mapping) else ""
            if (
                len(samples) < sample_limit
                and status in {"action_mismatch", "payload_mismatch"}
            ):
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "mode": mode,
                        "step": _as_int(item.get("step"), 0),
                        "phase": str(item.get("phase", "") or ""),
                        "status": status,
                        "decision_reason": reason,
                        "expected_action": str(replay.get("expected_action", "") or ""),
                        "actual_action": str(replay.get("actual_action", "") or ""),
                        "expected_payload": _json_safe(replay.get("expected_payload")),
                        "actual_payload": _json_safe(replay.get("actual_payload")),
                        "plan_cache": _json_safe(advice.get("plan_cache")),
                        "plan_commit": _json_safe(advice.get("plan_commit")),
                    }
                )

    def prefixed(counter: Counter[str], prefix: str) -> dict[str, int]:
        out: dict[str, int] = {}
        marker = f"{prefix}:"
        for key, value in sorted(counter.items()):
            if key.startswith(marker):
                out[key[len(marker) :]] = int(value)
        return out

    def convert(counter: Counter[str]) -> dict[str, Any]:
        comparisons = int(counter.get("execution_comparison_count", 0))
        matches = int(counter.get("execution_match_count", 0))
        return {
            "trace_step_count": int(counter.get("trace_step_count", 0)),
            "advice_available_count": int(counter.get("advice_available_count", 0)),
            "execution_comparison_count": comparisons,
            "execution_match_count": matches,
            "execution_mismatch_count": int(counter.get("execution_mismatch_count", 0)),
            "execution_match_rate": _safe_div(matches, comparisons),
            "execution_action_mismatch_count": int(
                counter.get("execution_action_mismatch_count", 0)
            ),
            "execution_payload_mismatch_count": int(
                counter.get("execution_payload_mismatch_count", 0)
            ),
            "injection_interrupted_count": int(
                counter.get("injection_interrupted_count", 0)
            ),
            "plan_commit_event_count": int(counter.get("plan_commit_event_count", 0)),
            "plan_committed_count": int(counter.get("plan_committed_count", 0)),
            "plan_commit_rate": _safe_div(
                int(counter.get("plan_committed_count", 0)),
                int(counter.get("plan_commit_event_count", 0)),
            ),
            "by_status": prefixed(counter, "status"),
            "commit_reasons": prefixed(counter, "commit_reason"),
        }

    summary = convert(counts)
    summary["by_game"] = {key: convert(value) for key, value in sorted(by_game.items())}
    summary["by_mode"] = {key: convert(value) for key, value in sorted(by_mode.items())}
    summary["by_decision_reason"] = {
        key: convert(value) for key, value in sorted(by_decision_reason.items())
    }
    summary["samples"] = samples
    return summary


def _window_score_delta_and_delay_with_final_score(
    trace: list[Any],
    *,
    baseline_index: int,
    window: int,
    default_score: int,
    final_score: int,
) -> tuple[int, int]:
    if not trace:
        return 0, 0
    baseline_index = max(0, min(int(baseline_index), len(trace) - 1))
    baseline_score = _trace_score_before(trace[baseline_index], default_score)
    scores: list[tuple[int, int]] = []
    for offset, future in enumerate(
        trace[baseline_index + 1 : baseline_index + 1 + max(0, int(window))],
        start=1,
    ):
        if isinstance(future, Mapping):
            scores.append((offset, _trace_score_before(future, baseline_score)))
    final_offset = len(trace) - baseline_index
    if 0 < final_offset <= max(0, int(window)):
        scores.append((final_offset, int(final_score)))
    delta = max([baseline_score, *(score for _, score in scores)]) - baseline_score
    delay = 0
    if delta > 0:
        for offset, score in scores:
            if score > baseline_score:
                delay = offset
                break
    return delta, delay


def _adapter_advice_score_attribution(
    rows: Iterable[Mapping[str, Any]],
    *,
    window: int = 8,
    sample_limit: int = 40,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    by_decision_reason: dict[str, Counter[str]] = defaultdict(Counter)
    by_action: dict[str, Counter[str]] = defaultdict(Counter)
    by_cache_reason: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []

    def bump(counter: Counter[str], key: str, amount: int = 1) -> None:
        counter[key] += int(amount)

    def observe_available(counter: Counter[str], advice: Mapping[str, Any]) -> None:
        bump(counter, "advice_available_count")
        if bool(advice.get("used")):
            bump(counter, "advice_used_count")
        else:
            bump(counter, "advice_rejected_count")

    def observe_used(
        counter: Counter[str],
        *,
        immediate_delta: int,
        delayed_delta: int,
        shifted_delta: int,
        pre_action_delta: int,
        random_delta: int,
    ) -> None:
        bump(counter, "used_immediate_score_delta_sum", immediate_delta)
        bump(counter, "used_delayed_score_delta_sum", delayed_delta)
        bump(counter, "used_shifted_delayed_score_delta_sum", shifted_delta)
        bump(counter, "used_pre_action_score_delta_sum", pre_action_delta)
        bump(counter, "used_random_step_score_delta_sum", random_delta)
        if immediate_delta > 0:
            bump(counter, "used_immediate_positive_count")
        if delayed_delta > 0:
            bump(counter, "used_delayed_positive_count")
        else:
            bump(counter, "used_no_delayed_score_count")

    def cache_reason(advice: Mapping[str, Any]) -> str:
        cache = advice.get("plan_cache")
        if isinstance(cache, Mapping):
            return str(cache.get("reason", "") or "unknown")
        return "none"

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        mode = str(row.get("adapter_advice_mode", "") or "none")
        final_score = _as_int(row.get("final_score"), 0)
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            counts["trace_step_count"] += 1
            by_game[game_id]["trace_step_count"] += 1
            by_mode[mode]["trace_step_count"] += 1
            advice = item.get("adapter_advice")
            if not isinstance(advice, Mapping) or not bool(advice.get("available")):
                continue

            reason = str(advice.get("decision_reason", "") or "unknown")
            action = str(advice.get("action", "") or item.get("action", "") or "unknown")
            cache = cache_reason(advice)
            for counter in (
                counts,
                by_game[game_id],
                by_mode[mode],
                by_decision_reason[reason],
                by_action[action],
                by_cache_reason[cache],
            ):
                observe_available(counter, advice)

            if not bool(advice.get("used")):
                continue

            current_score = _trace_score_before(item, final_score)
            if index + 1 < len(trace):
                next_score = _trace_score_before(trace[index + 1], current_score)
            else:
                next_score = final_score
            immediate_delta = max(0, next_score - current_score)
            delayed_delta, delay_steps = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=index,
                window=window,
                default_score=current_score,
                final_score=final_score,
            )
            if index + 1 < len(trace):
                shifted_delta, _ = _window_score_delta_and_delay_with_final_score(
                    trace,
                    baseline_index=index + 1,
                    window=window,
                    default_score=current_score,
                    final_score=final_score,
                )
            else:
                shifted_delta = 0
            pre_action_delta = _pre_action_score_delta(
                trace,
                index=index,
                window=window,
                current_score=current_score,
            )
            random_index = _deterministic_control_index(index, len(trace))
            random_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=random_index,
                window=window,
                default_score=current_score,
                final_score=final_score,
            )
            for counter in (
                counts,
                by_game[game_id],
                by_mode[mode],
                by_decision_reason[reason],
                by_action[action],
                by_cache_reason[cache],
            ):
                observe_used(
                    counter,
                    immediate_delta=immediate_delta,
                    delayed_delta=delayed_delta,
                    shifted_delta=shifted_delta,
                    pre_action_delta=pre_action_delta,
                    random_delta=random_delta,
                )
            if len(samples) < sample_limit and (
                delayed_delta > 0 or immediate_delta > 0 or delayed_delta <= 0
            ):
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "mode": mode,
                        "step": _as_int(item.get("step"), index),
                        "action": action,
                        "decision_reason": reason,
                        "cache_reason": cache,
                        "score_before": current_score,
                        "immediate_score_delta": immediate_delta,
                        "delayed_score_delta": delayed_delta,
                        "delay_steps": delay_steps,
                        "shifted_delayed_score_delta": shifted_delta,
                        "pre_action_score_delta": pre_action_delta,
                        "random_step_score_delta": random_delta,
                    }
                )

    def convert(counter: Counter[str]) -> dict[str, Any]:
        available = int(counter.get("advice_available_count", 0))
        used = int(counter.get("advice_used_count", 0))
        delayed_sum = int(counter.get("used_delayed_score_delta_sum", 0))
        shifted_sum = int(counter.get("used_shifted_delayed_score_delta_sum", 0))
        pre_action_sum = int(counter.get("used_pre_action_score_delta_sum", 0))
        random_sum = int(counter.get("used_random_step_score_delta_sum", 0))
        negative_control = max(shifted_sum, pre_action_sum, random_sum)
        margin = delayed_sum - negative_control
        return {
            "trace_step_count": int(counter.get("trace_step_count", 0)),
            "advice_available_count": available,
            "advice_used_count": used,
            "advice_rejected_count": int(counter.get("advice_rejected_count", 0)),
            "advice_usage_rate": _safe_div(used, available),
            "used_immediate_score_delta_sum": int(
                counter.get("used_immediate_score_delta_sum", 0)
            ),
            "used_immediate_positive_count": int(
                counter.get("used_immediate_positive_count", 0)
            ),
            "used_immediate_positive_rate": _safe_div(
                int(counter.get("used_immediate_positive_count", 0)),
                used,
            ),
            "used_delayed_score_delta_sum": delayed_sum,
            "used_delayed_positive_count": int(
                counter.get("used_delayed_positive_count", 0)
            ),
            "used_delayed_positive_rate": _safe_div(
                int(counter.get("used_delayed_positive_count", 0)),
                used,
            ),
            "used_no_delayed_score_count": int(
                counter.get("used_no_delayed_score_count", 0)
            ),
            "used_shifted_delayed_score_delta_sum": shifted_sum,
            "used_pre_action_score_delta_sum": pre_action_sum,
            "used_random_step_score_delta_sum": random_sum,
            "negative_control_max_score_delta_sum": negative_control,
            "causal_credit_margin": margin,
            "causal_credit_confidence": (
                0.0
                if delayed_sum <= 0 or margin <= 0
                else round(min(1.0, _safe_div(margin, delayed_sum)), 6)
            ),
        }

    summary = convert(counts)
    summary["window"] = int(window)
    summary["by_game"] = {key: convert(value) for key, value in sorted(by_game.items())}
    summary["by_mode"] = {key: convert(value) for key, value in sorted(by_mode.items())}
    summary["by_decision_reason"] = {
        key: convert(value) for key, value in sorted(by_decision_reason.items())
    }
    summary["by_action"] = {
        key: convert(value) for key, value in sorted(by_action.items())
    }
    summary["by_cache_reason"] = {
        key: convert(value) for key, value in sorted(by_cache_reason.items())
    }
    summary["samples"] = samples
    return summary


def _outcome_source_attribution_summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    unsolved_to_replan: Mapping[str, Any] | None = None,
    window: int = 8,
) -> dict[str, Any]:
    row_list = list(rows)
    source_names = (
        "adapter_advisory",
        "validated_continuation",
        "shadow_repair",
        "unsolved_replan",
        "rod_generic",
        "runtime_llm_hypothesis",
    )
    source_counters: dict[str, Counter[str]] = {
        name: Counter() for name in source_names
    }
    source_rows: dict[str, set[tuple[str, str]]] = {name: set() for name in source_names}
    source_score_sums: dict[str, int] = {name: 0 for name in source_names}
    score_by_row: dict[tuple[str, str], int] = {}
    samples: list[dict[str, Any]] = []

    def row_key(row: Mapping[str, Any]) -> tuple[str, str]:
        return (
            str(row.get("game_id", "") or "unknown"),
            str(row.get("variant", "") or "unknown"),
        )

    def add_event(
        source: str,
        *,
        row: Mapping[str, Any],
        trace: list[Any],
        index: int,
        item: Mapping[str, Any],
        reason: str,
        final_score: int,
    ) -> None:
        if source not in source_counters:
            return
        key = row_key(row)
        current_score = _trace_score_before(item, final_score)
        if index + 1 < len(trace):
            next_score = _trace_score_before(trace[index + 1], current_score)
        else:
            next_score = final_score
        immediate_delta = max(0, next_score - current_score)
        delayed_delta, delay_steps = _window_score_delta_and_delay_with_final_score(
            trace,
            baseline_index=index,
            window=window,
            default_score=current_score,
            final_score=final_score,
        )
        if index + 1 < len(trace):
            shifted_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=index + 1,
                window=window,
                default_score=current_score,
                final_score=final_score,
            )
        else:
            shifted_delta = 0
        pre_action_delta = _pre_action_score_delta(
            trace,
            index=index,
            window=window,
            current_score=current_score,
        )
        random_index = _deterministic_control_index(index, len(trace))
        random_delta, _ = _window_score_delta_and_delay_with_final_score(
            trace,
            baseline_index=random_index,
            window=window,
            default_score=current_score,
            final_score=final_score,
        )
        counter = source_counters[source]
        counter["evidence_step_count"] += 1
        counter["immediate_score_delta_sum"] += int(immediate_delta)
        counter["delayed_score_delta_sum"] += int(delayed_delta)
        counter["shifted_delayed_score_delta_sum"] += int(shifted_delta)
        counter["pre_action_score_delta_sum"] += int(pre_action_delta)
        counter["random_step_score_delta_sum"] += int(random_delta)
        if immediate_delta > 0:
            counter["immediate_positive_count"] += 1
        if delayed_delta > 0:
            counter["delayed_positive_count"] += 1
        counter[f"reason:{reason or 'unknown'}"] += 1
        if key not in source_rows[source]:
            source_rows[source].add(key)
            score = int(score_by_row.get(key, final_score))
            source_score_sums[source] += score
        if len(samples) < 40:
            samples.append(
                {
                    "source": source,
                    "game_id": str(row.get("game_id", "") or "unknown"),
                    "variant": str(row.get("variant", "") or "unknown"),
                    "step": _as_int(item.get("step"), index),
                    "reason": reason,
                    "score_before": current_score,
                    "immediate_score_delta": immediate_delta,
                    "delayed_score_delta": delayed_delta,
                    "delay_steps": delay_steps,
                    "negative_control_max_score_delta": max(
                        shifted_delta,
                        pre_action_delta,
                        random_delta,
                    ),
                }
            )

    def prefixed(counter: Counter[str], prefix: str) -> dict[str, int]:
        out: dict[str, int] = {}
        marker = f"{prefix}:"
        for key, value in sorted(counter.items()):
            if key.startswith(marker):
                out[key[len(marker) :]] = int(value)
        return out

    def convert(source: str) -> dict[str, Any]:
        counter = source_counters[source]
        delayed = int(counter.get("delayed_score_delta_sum", 0))
        negative_control = max(
            int(counter.get("shifted_delayed_score_delta_sum", 0)),
            int(counter.get("pre_action_score_delta_sum", 0)),
            int(counter.get("random_step_score_delta_sum", 0)),
        )
        margin = delayed - negative_control
        return {
            "evidence_step_count": int(counter.get("evidence_step_count", 0)),
            "row_count": len(source_rows[source]),
            "associated_score_sum": int(source_score_sums[source]),
            "immediate_score_delta_sum": int(counter.get("immediate_score_delta_sum", 0)),
            "immediate_positive_count": int(counter.get("immediate_positive_count", 0)),
            "delayed_score_delta_sum": delayed,
            "delayed_positive_count": int(counter.get("delayed_positive_count", 0)),
            "negative_control_max_score_delta_sum": negative_control,
            "causal_credit_margin": margin,
            "causal_credit_confidence": (
                0.0
                if delayed <= 0 or margin <= 0
                else round(min(1.0, _safe_div(margin, delayed)), 6)
            ),
            "reason_counts": prefixed(counter, "reason"),
        }

    for row in row_list:
        key = row_key(row)
        final_score = _as_int(row.get("final_score"), 0)
        score_by_row[key] = final_score
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            advice = item.get("adapter_advice")
            if isinstance(advice, Mapping):
                cache = advice.get("plan_cache")
                cache = cache if isinstance(cache, Mapping) else {}
                cache_source = str(cache.get("source", "") or "")
                cache_reason = str(cache.get("reason", "") or "")
                decision_reason = str(advice.get("decision_reason", "") or "")
                if bool(advice.get("used")):
                    add_event(
                        "adapter_advisory",
                        row=row,
                        trace=trace,
                        index=index,
                        item=item,
                        reason=decision_reason or "used",
                        final_score=final_score,
                    )
                if bool(cache.get("cache_action_used")) or (
                    bool(advice.get("used")) and cache_reason in {"cache_hit", "fresh_plan_cache_hit"}
                ):
                    add_event(
                        "validated_continuation",
                        row=row,
                        trace=trace,
                        index=index,
                        item=item,
                        reason=cache_reason or "cache_action_used",
                        final_score=final_score,
                    )
                if cache_source.startswith("shadow_repair_") or cache_reason.startswith("shadow_repair_"):
                    add_event(
                        "shadow_repair",
                        row=row,
                        trace=trace,
                        index=index,
                        item=item,
                        reason=cache_source or cache_reason,
                        final_score=final_score,
                    )
                if cache_source == "eca_unsolved_replan" or cache_reason == "eca_unsolved_replan":
                    add_event(
                        "unsolved_replan",
                        row=row,
                        trace=trace,
                        index=index,
                        item=item,
                        reason="plan_cache",
                        final_score=final_score,
                    )
            unsolved_payload = item.get("unsolved_replan")
            if isinstance(unsolved_payload, Mapping) and bool(
                unsolved_payload.get("executed") or unsolved_payload.get("used")
            ):
                add_event(
                    "unsolved_replan",
                    row=row,
                    trace=trace,
                    index=index,
                    item=item,
                    reason=str(unsolved_payload.get("reason", "") or "executed"),
                    final_score=final_score,
                )
            rod_action = _rod_score_precondition_action_from_trace_item(item)
            if rod_action and bool(rod_action.get("applied")):
                add_event(
                    "rod_generic",
                    row=row,
                    trace=trace,
                    index=index,
                    item=item,
                    reason=str(rod_action.get("mode", "") or "rod_score_precondition_action"),
                    final_score=final_score,
                )
            diagnostics = item.get("diagnostics")
            diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
            causal_pass = item.get("causal_pass") or diagnostics.get("causal_pass")
            hypothesis_rods = diagnostics.get("hypothesis_rods")
            if (
                isinstance(causal_pass, Mapping)
                and bool(causal_pass.get("used") or causal_pass.get("called"))
            ) or (
                isinstance(hypothesis_rods, Mapping)
                and _as_int(hypothesis_rods.get("hypothesis_rod_count"), 0) > 0
            ):
                add_event(
                    "runtime_llm_hypothesis",
                    row=row,
                    trace=trace,
                    index=index,
                    item=item,
                    reason="causal_pass_or_hypothesis_rod",
                    final_score=final_score,
                )

    unsolved_summary = dict(unsolved_to_replan or {})
    payload: dict[str, Any] = {
        "schema_version": "eca_outcome_source_attribution.v1",
        "score_sum": sum(_as_int(row.get("final_score"), 0) for row in row_list),
        "window": int(window),
        "sources": {source: convert(source) for source in source_names},
        "samples": samples,
    }
    payload["sources"]["unsolved_replan"]["diagnostic_eligible_count"] = _as_int(
        unsolved_summary.get("eligible_count"),
        0,
    )
    payload["sources"]["unsolved_replan"]["diagnostic_probe_candidate_count"] = _as_int(
        unsolved_summary.get("probe_candidate_count"),
        0,
    )
    payload["sources"]["unsolved_replan"]["diagnostic_short_plan_candidate_count"] = _as_int(
        unsolved_summary.get("short_plan_candidate_count"),
        0,
    )
    return payload


def _site_from_mapping(payload: Any) -> list[int] | None:
    if not isinstance(payload, Mapping):
        return None
    site = payload.get("site")
    if isinstance(site, (list, tuple)) and len(site) >= 2:
        try:
            return [int(site[0]), int(site[1])]
        except (TypeError, ValueError):
            return None
    if "x" in payload and "y" in payload:
        try:
            return [int(payload.get("x")), int(payload.get("y"))]
        except (TypeError, ValueError):
            return None
    return None


def _teacher_objective_from_advice(advice: Mapping[str, Any]) -> dict[str, Any]:
    explicit = advice.get("teacher_objective")
    if isinstance(explicit, Mapping):
        return _json_safe(explicit)
    action = _normalize_action_name(advice.get("action"))
    payload = advice.get("payload")
    if not isinstance(payload, Mapping):
        payload = {}
    if action == "ACTION6":
        objective: dict[str, Any] = {"operator": "click", "action": action}
        site = _site_from_mapping(payload)
        if site is not None:
            objective["site"] = site
        return objective
    if action in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
        return {"operator": "move", "direction_action": action, "action": action}
    if action == "ACTION5":
        return {"operator": "select_or_attach", "action": action}
    if action:
        return {"operator": "act", "action": action}
    return {}


def _eca_objective_from_control_target(control_target: Any, action: Any) -> dict[str, Any]:
    action_name = _normalize_action_name(action)
    if not isinstance(control_target, Mapping):
        if action_name:
            return {"operator": "act", "action": action_name}
        return {}
    operator = str(control_target.get("primary_operator", "") or "").strip()
    if not operator:
        if action_name == "ACTION6":
            operator = "click"
        elif action_name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
            operator = "move"
        elif action_name == "ACTION5":
            operator = "select_or_attach"
        elif action_name:
            operator = "act"
    objective: dict[str, Any] = {"operator": operator, "action": action_name}
    primary_target = control_target.get("primary_target")
    if isinstance(primary_target, Mapping):
        site = _site_from_mapping(primary_target)
        if site is not None:
            objective["site"] = site
    if "site" not in objective:
        candidates = control_target.get("target_candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                site = _site_from_mapping(candidate)
                if site is not None:
                    objective["site"] = site
                    break
    contract = control_target.get("goal_target_contract")
    if isinstance(contract, Mapping):
        objective["target_translation_status"] = str(
            contract.get("target_translation_status", "") or ""
        )
        objective["solver_family"] = str(contract.get("solver_family", "") or "")
    elif control_target.get("solver_family"):
        objective["solver_family"] = str(control_target.get("solver_family", "") or "")
    return objective


def _objective_operator_matches(eca_objective: Mapping[str, Any], teacher_objective: Mapping[str, Any]) -> bool:
    eca_operator = str(eca_objective.get("operator", "") or "")
    teacher_operator = str(teacher_objective.get("operator", "") or "")
    if not eca_operator or not teacher_operator:
        return False
    if eca_operator == teacher_operator:
        return True
    if teacher_operator == "move" and eca_operator in {"route_to", "push"}:
        return True
    if teacher_operator == "click" and eca_operator in {"click", "probe"}:
        return True
    if teacher_operator == "select_or_attach" and eca_operator in {"select", "attach", "drop"}:
        return True
    return False


def _adapter_objective_alignment_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        mode = str(row.get("adapter_advice_mode", "") or "none")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            advice = item.get("adapter_advice")
            if not isinstance(advice, Mapping) or not bool(advice.get("available")):
                continue
            teacher = _teacher_objective_from_advice(advice)
            if not teacher:
                continue
            eca_action = advice.get("eca_action") or item.get("action")
            eca = _eca_objective_from_control_target(item.get("control_target"), eca_action)
            totals["comparison_count"] += 1
            by_game[game_id]["comparison_count"] += 1
            by_mode[mode]["comparison_count"] += 1
            if _objective_operator_matches(eca, teacher):
                totals["operator_agreement_count"] += 1
                by_game[game_id]["operator_agreement_count"] += 1
                by_mode[mode]["operator_agreement_count"] += 1
            eca_site = eca.get("site")
            teacher_site = teacher.get("site")
            if isinstance(eca_site, list) and isinstance(teacher_site, list):
                totals["site_comparison_count"] += 1
                by_game[game_id]["site_comparison_count"] += 1
                by_mode[mode]["site_comparison_count"] += 1
                if eca_site == teacher_site:
                    totals["site_agreement_count"] += 1
                    by_game[game_id]["site_agreement_count"] += 1
                    by_mode[mode]["site_agreement_count"] += 1

    def convert(counter: Counter[str]) -> dict[str, Any]:
        comparisons = int(counter.get("comparison_count", 0))
        operator_agreements = int(counter.get("operator_agreement_count", 0))
        site_comparisons = int(counter.get("site_comparison_count", 0))
        site_agreements = int(counter.get("site_agreement_count", 0))
        return {
            "comparison_count": comparisons,
            "operator_agreement_count": operator_agreements,
            "operator_agreement_rate": _safe_div(operator_agreements, comparisons),
            "site_comparison_count": site_comparisons,
            "site_agreement_count": site_agreements,
            "site_agreement_rate": _safe_div(site_agreements, site_comparisons),
        }

    payload = convert(totals)
    payload["by_game"] = {key: convert(value) for key, value in sorted(by_game.items())}
    payload["by_mode"] = {key: convert(value) for key, value in sorted(by_mode.items())}
    return payload


def _adapter_disagreement_replay(rows: Iterable[Mapping[str, Any]], sample_limit: int = 40) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        mode = str(row.get("adapter_advice_mode", "") or "none")
        final_score = _as_int(row.get("final_score"), 0)
        row_scoreless = final_score <= 0
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            advice = item.get("adapter_advice")
            if not isinstance(advice, Mapping) or not bool(advice.get("available")):
                continue
            action_disagreed = "action_agreed" in advice and not bool(advice.get("action_agreed"))
            payload_disagreed = "payload_agreed" in advice and not bool(advice.get("payload_agreed"))
            if not action_disagreed and not payload_disagreed:
                continue
            teacher = _teacher_objective_from_advice(advice)
            eca_action = advice.get("eca_action") or item.get("action")
            eca = _eca_objective_from_control_target(item.get("control_target"), eca_action)
            objective_disagreed = bool(teacher and not _objective_operator_matches(eca, teacher))
            if not objective_disagreed:
                eca_site = eca.get("site")
                teacher_site = teacher.get("site") if teacher else None
                objective_disagreed = isinstance(eca_site, list) and isinstance(teacher_site, list) and eca_site != teacher_site

            counts["disagreement_count"] += 1
            by_game[game_id]["disagreement_count"] += 1
            if action_disagreed:
                counts["action_disagreement_count"] += 1
                by_game[game_id]["action_disagreement_count"] += 1
            if payload_disagreed:
                counts["payload_disagreement_count"] += 1
                by_game[game_id]["payload_disagreement_count"] += 1
            if objective_disagreed:
                counts["objective_disagreement_count"] += 1
                by_game[game_id]["objective_disagreement_count"] += 1
            if row_scoreless:
                counts["scoreless_disagreement_count"] += 1
                by_game[game_id]["scoreless_disagreement_count"] += 1
            if len(samples) < sample_limit and (row_scoreless or objective_disagreed or action_disagreed):
                control_target = item.get("control_target")
                contract_status = ""
                solver_family = ""
                if isinstance(control_target, Mapping):
                    solver_family = str(control_target.get("solver_family", "") or "")
                    contract = control_target.get("goal_target_contract")
                    if isinstance(contract, Mapping):
                        contract_status = str(contract.get("target_translation_status", "") or "")
                        solver_family = str(contract.get("solver_family", "") or solver_family)
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "mode": mode,
                        "step": _as_int(item.get("step"), 0),
                        "phase": str(item.get("phase", "") or ""),
                        "final_score": final_score,
                        "score_before": _as_int(item.get("score_before"), 0),
                        "eca_action": _normalize_action_name(eca_action),
                        "executed_action": _normalize_action_name(item.get("action")),
                        "adapter_action": _normalize_action_name(advice.get("action")),
                        "action_disagreed": action_disagreed,
                        "payload_disagreed": payload_disagreed,
                        "objective_disagreed": objective_disagreed,
                        "advice_used": bool(advice.get("used")),
                        "solver_family": solver_family,
                        "target_translation_status": contract_status,
                        "eca_objective": eca,
                        "teacher_objective": teacher,
                        "plan_source": str(advice.get("plan_source", "") or ""),
                    }
                )

    def convert(counter: Counter[str]) -> dict[str, Any]:
        return {
            "disagreement_count": int(counter.get("disagreement_count", 0)),
            "action_disagreement_count": int(counter.get("action_disagreement_count", 0)),
            "payload_disagreement_count": int(counter.get("payload_disagreement_count", 0)),
            "objective_disagreement_count": int(counter.get("objective_disagreement_count", 0)),
            "scoreless_disagreement_count": int(counter.get("scoreless_disagreement_count", 0)),
        }

    payload = convert(counts)
    payload["sample_count"] = len(samples)
    payload["samples"] = samples
    payload["by_game"] = {key: convert(value) for key, value in sorted(by_game.items())}
    return payload


def _structural_rods_from_trace_item(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = item.get("structural_rods")
    if isinstance(payload, Mapping):
        return dict(payload)
    diagnostics = item.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        payload = diagnostics.get("structural_rods")
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _rod_selector_bias_from_trace_item(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = item.get("rod_selector_bias")
    if isinstance(payload, Mapping):
        return dict(payload)
    diagnostics = item.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        payload = diagnostics.get("rod_selector_bias")
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _rod_action_binding_from_trace_item(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = item.get("rod_action_binding")
    if isinstance(payload, Mapping):
        return dict(payload)
    diagnostics = item.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        payload = diagnostics.get("rod_action_binding")
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _rod_target_binding_from_trace_item(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = item.get("rod_target_binding")
    if isinstance(payload, Mapping):
        return dict(payload)
    diagnostics = item.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        payload = diagnostics.get("rod_target_binding")
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _rod_score_precondition_action_from_trace_item(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = item.get("rod_score_precondition_action")
    if isinstance(payload, Mapping):
        return dict(payload)
    diagnostics = item.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        payload = diagnostics.get("rod_score_precondition_action")
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _diagnose_rod_score_precondition_action_path(counter: Counter[str]) -> dict[str, Any]:
    applied = int(counter.get("applied_step_count", 0))
    causal = int(counter.get("causally_supported_applied_step_count", 0))
    associated = int(counter.get("associated_only_applied_step_count", 0))
    no_delayed = int(counter.get("no_delayed_score_applied_step_count", 0))
    unverified = int(counter.get("blocked_reason:score_target_unverified_locally", 0))
    direct_without_goal = int(counter.get("direct_score_probe_without_goal_count", 0))
    post_direct_reentry = int(
        counter.get("post_direct_score_probe_contradiction_reentry_count", 0)
    )
    reasons: dict[str, int] = {}
    if unverified:
        reasons["blocked_unverified_score_target"] = unverified
    if direct_without_goal:
        reasons["direct_score_probe_without_goal"] = direct_without_goal
    if post_direct_reentry:
        reasons["post_direct_score_probe_contradiction_reentry"] = post_direct_reentry
    if no_delayed:
        reasons["applied_no_delayed_score"] = no_delayed
    if associated:
        reasons["delayed_score_without_causal_path"] = associated
    if applied <= 0:
        status = "blocked_unverified_score_targets" if unverified else "not_used"
    elif causal > 0:
        status = "causal_support_present"
    elif post_direct_reentry > 0:
        status = "contradicted_score_target_reentered"
    elif direct_without_goal > 0:
        status = "direct_score_probe_contradicted"
    elif associated > 0:
        status = "score_associated_but_causal_path_unproven"
    elif no_delayed > 0:
        status = "applied_without_delayed_score"
    else:
        status = "applied_without_classification"
    return {
        "status": status,
        "reason_counts": dict(sorted(reasons.items())),
    }


def _trace_score_before(item: Any, default: int = 0) -> int:
    if not isinstance(item, Mapping):
        return int(default)
    return _as_int(item.get("score_before"), default)


def _window_score_delta_from_index(
    trace: list[Any],
    *,
    baseline_index: int,
    window: int,
    default_score: int,
) -> int:
    delta, _delay = _window_score_delta_and_delay_from_index(
        trace,
        baseline_index=baseline_index,
        window=window,
        default_score=default_score,
    )
    return delta


def _window_score_delta_and_delay_from_index(
    trace: list[Any],
    *,
    baseline_index: int,
    window: int,
    default_score: int,
) -> tuple[int, int]:
    if not trace:
        return 0, 0
    baseline_index = max(0, min(int(baseline_index), len(trace) - 1))
    baseline_score = _trace_score_before(trace[baseline_index], default_score)
    scores = [
        _trace_score_before(future, baseline_score)
        for future in trace[baseline_index + 1 : baseline_index + 1 + max(0, int(window))]
        if isinstance(future, Mapping)
    ]
    delta = max([baseline_score, *scores]) - baseline_score
    delay = 0
    if delta > 0:
        for offset, score in enumerate(scores, start=1):
            if score > baseline_score:
                delay = offset
                break
    return delta, delay


def _pre_action_score_delta(
    trace: list[Any],
    *,
    index: int,
    window: int,
    current_score: int,
) -> int:
    if not trace or index <= 0:
        return 0
    start_index = max(0, int(index) - max(1, int(window)))
    baseline_score = _trace_score_before(trace[start_index], current_score)
    scores = [
        _trace_score_before(item, baseline_score)
        for item in trace[start_index + 1 : int(index) + 1]
        if isinstance(item, Mapping)
    ]
    return max([baseline_score, *scores]) - baseline_score


def _deterministic_control_index(index: int, trace_len: int) -> int:
    if trace_len <= 0:
        return 0
    control_index = (int(index) * 7 + trace_len // 2 + 3) % trace_len
    if trace_len > 1 and control_index == index:
        control_index = (control_index + 1) % trace_len
    return control_index


def _wrong_action_name(action_name: str) -> str:
    normalized = _normalize_action_name(action_name)
    if normalized not in CORE_ACTIONS:
        return ""
    index = CORE_ACTIONS.index(normalized)
    return CORE_ACTIONS[(index + 1) % len(CORE_ACTIONS)]


def _trace_item_site_signature(item: Any) -> str:
    if not isinstance(item, Mapping):
        return ""
    for key in ("target_site", "site"):
        site = _site_signature(item.get(key))
        if site:
            return site
    payload = item.get("payload")
    site_from_payload = _site_from_mapping(payload)
    if site_from_payload is not None:
        return _site_signature(site_from_payload)
    control_target = item.get("control_target")
    if isinstance(control_target, Mapping):
        primary_target = control_target.get("primary_target")
        site = _site_signature(primary_target.get("site")) if isinstance(primary_target, Mapping) else ""
        if site:
            return site
        candidates = control_target.get("target_candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                site = _site_signature(candidate.get("site"))
                if site:
                    return site
    return ""


def _runtime_mechanic_target_site_signature(item: Mapping[str, Any]) -> str:
    decision = _rod_score_precondition_action_from_trace_item(item)
    if decision:
        site = _site_signature(decision.get("target_site"))
        if site:
            return site
    binding = _rod_target_binding_from_trace_item(item)
    if binding:
        for key in ("target_site", "selected_site"):
            site = _site_signature(binding.get(key))
            if site:
                return site
    site = _site_signature(item.get("target_site"))
    if site:
        return site
    control_target = item.get("control_target")
    if isinstance(control_target, Mapping):
        primary_target = control_target.get("primary_target")
        if isinstance(primary_target, Mapping):
            site = _site_signature(primary_target.get("site"))
            if site:
                return site
    return ""


def _wrong_target_site_before_score(
    trace: list[Any],
    *,
    index: int,
    target_site: str,
    delayed_delay_steps: int,
) -> tuple[bool, int]:
    if not target_site or delayed_delay_steps <= 0:
        return False, 0
    item = trace[index] if 0 <= index < len(trace) else {}
    current_site = _trace_item_site_signature(item)
    if current_site and current_site != target_site:
        return True, 0
    lookahead_items = [
        future
        for future in trace[index + 1 : index + 1 + max(0, int(delayed_delay_steps))]
        if isinstance(future, Mapping)
    ]
    for offset, future in enumerate(lookahead_items, start=1):
        if offset >= delayed_delay_steps:
            break
        future_site = _trace_item_site_signature(future)
        if future_site and future_site != target_site:
            return True, offset
    return False, 0


def _rod_selector_bias_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    trace_step_count = 0
    enabled_step_count = 0
    compared_step_count = 0
    rod_backed_step_count = 0
    primary_change_count = 0
    confidence_delta_total = 0.0
    rows_with_primary_change = 0
    score_sum_with_primary_change = 0
    transitions: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    for row in row_list:
        game_id = str(row.get("game_id", "") or "unknown")
        row_changed = False
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            bias = _rod_selector_bias_from_trace_item(item)
            if not bias:
                continue
            trace_step_count += 1
            by_game[game_id]["trace_step_count"] += 1
            if bool(bias.get("enabled")):
                enabled_step_count += 1
                by_game[game_id]["enabled_step_count"] += 1
            if bool(bias.get("compared")):
                compared_step_count += 1
                by_game[game_id]["compared_step_count"] += 1
            if _as_int(bias.get("rod_count"), 0) > 0:
                rod_backed_step_count += 1
                by_game[game_id]["rod_backed_step_count"] += 1
            confidence_delta_total += _as_float(bias.get("confidence_delta"), 0.0)
            observer = str(bias.get("observer_primary_primitive", "") or "")
            biased = str(bias.get("biased_primary_primitive", "") or "")
            if bool(bias.get("changed_primary")):
                primary_change_count += 1
                by_game[game_id]["primary_change_count"] += 1
                row_changed = True
                if observer or biased:
                    transitions[f"{observer}->{biased}"] += 1
                if len(samples) < 20:
                    samples.append(
                        {
                            "game_id": game_id,
                            "variant": str(row.get("variant", "") or "unknown"),
                            "step": _as_int(item.get("step"), 0),
                            "phase": str(item.get("phase", "") or ""),
                            "observer_primary_primitive": observer,
                            "biased_primary_primitive": biased,
                            "final_primary_primitive": str(bias.get("final_primary_primitive", "") or ""),
                            "observer_rank_of_biased_primary": _as_int(
                                bias.get("observer_rank_of_biased_primary"),
                                0,
                            ),
                            "biased_rank_of_observer_primary": _as_int(
                                bias.get("biased_rank_of_observer_primary"),
                                0,
                            ),
                            "rod_count": _as_int(bias.get("rod_count"), 0),
                            "confidence_delta": _as_float(bias.get("confidence_delta"), 0.0),
                        }
                    )
        if row_changed:
            rows_with_primary_change += 1
            score_sum_with_primary_change += _as_int(row.get("final_score"), 0)

    def convert(counter: Counter[str]) -> dict[str, Any]:
        trace_steps = int(counter.get("trace_step_count", 0))
        primary_changes = int(counter.get("primary_change_count", 0))
        return {
            "trace_step_count": trace_steps,
            "enabled_step_count": int(counter.get("enabled_step_count", 0)),
            "compared_step_count": int(counter.get("compared_step_count", 0)),
            "rod_backed_step_count": int(counter.get("rod_backed_step_count", 0)),
            "primary_change_count": primary_changes,
            "primary_change_rate": _safe_div(primary_changes, trace_steps),
        }

    return {
        "schema_version": "eca_rod_selector_bias_summary.v1",
        "trace_step_count": trace_step_count,
        "enabled_step_count": enabled_step_count,
        "compared_step_count": compared_step_count,
        "rod_backed_step_count": rod_backed_step_count,
        "primary_change_count": primary_change_count,
        "primary_change_rate": _safe_div(primary_change_count, trace_step_count),
        "mean_confidence_delta": _safe_div(confidence_delta_total, compared_step_count),
        "rows_with_primary_change": rows_with_primary_change,
        "score_sum_with_primary_change": score_sum_with_primary_change,
        "transitions": dict(sorted(transitions.items())),
        "samples": samples,
        "by_game": {key: convert(value) for key, value in sorted(by_game.items())},
    }


def _rod_action_binding_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    trace_step_count = 0
    enabled_step_count = 0
    applied_step_count = 0
    rod_backed_step_count = 0
    blocked_step_count = 0
    rows_with_applied = 0
    score_sum_with_applied = 0
    transitions: Counter[str] = Counter()
    blocked_reasons: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    for row in row_list:
        game_id = str(row.get("game_id", "") or "unknown")
        row_applied = False
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            binding = _rod_action_binding_from_trace_item(item)
            if not binding:
                continue
            trace_step_count += 1
            by_game[game_id]["trace_step_count"] += 1
            if bool(binding.get("enabled")):
                enabled_step_count += 1
                by_game[game_id]["enabled_step_count"] += 1
            if _as_int(binding.get("matched_rod_count"), 0) > 0:
                rod_backed_step_count += 1
                by_game[game_id]["rod_backed_step_count"] += 1
            blocked = str(binding.get("blocked_reason", "") or "")
            if blocked:
                blocked_step_count += 1
                blocked_reasons[blocked] += 1
                by_game[game_id]["blocked_step_count"] += 1
                by_game[game_id][f"blocked_reason:{blocked}"] += 1
            if bool(binding.get("applied")):
                applied_step_count += 1
                by_game[game_id]["applied_step_count"] += 1
                row_applied = True
                previous = _normalize_action_name(binding.get("previous_action") or binding.get("current_action"))
                selected = _normalize_action_name(binding.get("selected_action"))
                if previous or selected:
                    transitions[f"{previous}->{selected}"] += 1
                if len(samples) < 20:
                    samples.append(
                        {
                            "game_id": game_id,
                            "variant": str(row.get("variant", "") or "unknown"),
                            "step": _as_int(item.get("step"), 0),
                            "phase": str(item.get("phase", "") or ""),
                            "primitive_id": str(binding.get("primitive_id", "") or ""),
                            "solver_family": str(binding.get("solver_family", "") or ""),
                            "previous_action": previous,
                            "selected_action": selected,
                            "matched_rod_count": _as_int(binding.get("matched_rod_count"), 0),
                            "recommended_net_support": _as_int(binding.get("recommended_net_support"), 0),
                            "current_net_support": _as_int(binding.get("current_net_support"), 0),
                        }
                    )
        if row_applied:
            rows_with_applied += 1
            score_sum_with_applied += _as_int(row.get("final_score"), 0)

    def convert(counter: Counter[str]) -> dict[str, Any]:
        trace_steps = int(counter.get("trace_step_count", 0))
        applied = int(counter.get("applied_step_count", 0))
        reason_counts = {
            key.split(":", 1)[1]: int(value)
            for key, value in counter.items()
            if key.startswith("blocked_reason:")
        }
        return {
            "trace_step_count": trace_steps,
            "enabled_step_count": int(counter.get("enabled_step_count", 0)),
            "rod_backed_step_count": int(counter.get("rod_backed_step_count", 0)),
            "applied_step_count": applied,
            "applied_rate": _safe_div(applied, trace_steps),
            "blocked_step_count": int(counter.get("blocked_step_count", 0)),
            "blocked_reasons": dict(sorted(reason_counts.items())),
            "click_action_requires_target_binding_blocked_count": int(
                counter.get("blocked_reason:click_action_requires_target_binding", 0)
            ),
        }

    return {
        "schema_version": "eca_rod_action_binding_summary.v1",
        "trace_step_count": trace_step_count,
        "enabled_step_count": enabled_step_count,
        "rod_backed_step_count": rod_backed_step_count,
        "applied_step_count": applied_step_count,
        "applied_rate": _safe_div(applied_step_count, trace_step_count),
        "blocked_step_count": blocked_step_count,
        "click_action_requires_target_binding_blocked_count": int(
            blocked_reasons.get("click_action_requires_target_binding", 0)
        ),
        "rows_with_applied": rows_with_applied,
        "score_sum_with_applied": score_sum_with_applied,
        "transitions": dict(sorted(transitions.items())),
        "blocked_reasons": dict(sorted(blocked_reasons.items())),
        "samples": samples,
        "by_game": {key: convert(value) for key, value in sorted(by_game.items())},
    }


def _rod_target_binding_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    trace_step_count = 0
    enabled_step_count = 0
    applied_step_count = 0
    rod_backed_step_count = 0
    rows_with_applied = 0
    score_sum_with_applied = 0
    transitions: Counter[str] = Counter()
    blocked_reasons: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    for row in row_list:
        game_id = str(row.get("game_id", "") or "unknown")
        row_applied = False
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            binding = _rod_target_binding_from_trace_item(item)
            if not binding:
                continue
            trace_step_count += 1
            by_game[game_id]["trace_step_count"] += 1
            if bool(binding.get("enabled")):
                enabled_step_count += 1
                by_game[game_id]["enabled_step_count"] += 1
            if _as_int(binding.get("matched_rod_count"), 0) > 0:
                rod_backed_step_count += 1
                by_game[game_id]["rod_backed_step_count"] += 1
            blocked = str(binding.get("blocked_reason", "") or "")
            if blocked:
                blocked_reasons[blocked] += 1
            if bool(binding.get("applied")):
                applied_step_count += 1
                by_game[game_id]["applied_step_count"] += 1
                row_applied = True
                previous = _site_signature(binding.get("previous_site") or binding.get("current_site"))
                selected = _site_signature(binding.get("selected_site"))
                if previous or selected:
                    transitions[f"{previous}->{selected}"] += 1
                if len(samples) < 20:
                    samples.append(
                        {
                            "game_id": game_id,
                            "variant": str(row.get("variant", "") or "unknown"),
                            "step": _as_int(item.get("step"), 0),
                            "phase": str(item.get("phase", "") or ""),
                            "primitive_id": str(binding.get("primitive_id", "") or ""),
                            "solver_family": str(binding.get("solver_family", "") or ""),
                            "previous_site": previous,
                            "selected_site": selected,
                            "matched_rod_count": _as_int(binding.get("matched_rod_count"), 0),
                            "recommended_net_support": _as_int(binding.get("recommended_net_support"), 0),
                        }
                    )
        if row_applied:
            rows_with_applied += 1
            score_sum_with_applied += _as_int(row.get("final_score"), 0)

    def convert(counter: Counter[str]) -> dict[str, Any]:
        trace_steps = int(counter.get("trace_step_count", 0))
        applied = int(counter.get("applied_step_count", 0))
        return {
            "trace_step_count": trace_steps,
            "enabled_step_count": int(counter.get("enabled_step_count", 0)),
            "rod_backed_step_count": int(counter.get("rod_backed_step_count", 0)),
            "applied_step_count": applied,
            "applied_rate": _safe_div(applied, trace_steps),
        }

    return {
        "schema_version": "eca_rod_target_binding_summary.v1",
        "trace_step_count": trace_step_count,
        "enabled_step_count": enabled_step_count,
        "rod_backed_step_count": rod_backed_step_count,
        "applied_step_count": applied_step_count,
        "applied_rate": _safe_div(applied_step_count, trace_step_count),
        "rows_with_applied": rows_with_applied,
        "score_sum_with_applied": score_sum_with_applied,
        "transitions": dict(sorted(transitions.items())),
        "blocked_reasons": dict(sorted(blocked_reasons.items())),
        "samples": samples,
        "by_game": {key: convert(value) for key, value in sorted(by_game.items())},
    }


def _rod_score_precondition_action_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    delayed_credit_window = 8
    shifted_credit_window_offset = delayed_credit_window
    trace_step_count = 0
    enabled_step_count = 0
    applied_step_count = 0
    macro_step_count = 0
    blocked_step_count = 0
    low_commitment_probe_eligible_count = 0
    low_commitment_probe_applied_step_count = 0
    locally_grounded_applied_step_count = 0
    score_grounded_applied_step_count = 0
    progress_only_grounded_applied_step_count = 0
    ungrounded_applied_step_count = 0
    direct_score_probe_without_goal_count = 0
    direct_score_probe_without_goal_failure_count = 0
    direct_score_probe_contradicted_target_count = 0
    post_direct_score_probe_contradiction_reentry_count = 0
    causally_supported_applied_step_count = 0
    associated_only_applied_step_count = 0
    no_delayed_score_applied_step_count = 0
    causally_supported_score_delta_sum = 0
    rows_with_applied = 0
    score_sum_with_applied = 0
    applied_score_delta_sum = 0
    applied_positive_score_step_count = 0
    delayed_score_delta_sum = 0
    delayed_positive_score_step_count = 0
    shifted_delayed_score_delta_sum = 0
    shifted_delayed_positive_score_step_count = 0
    pre_action_score_delta_sum = 0
    pre_action_positive_score_step_count = 0
    random_step_score_delta_sum = 0
    random_step_positive_score_step_count = 0
    wrong_target_action_credit_sum = 0
    wrong_target_action_before_score_count = 0
    wrong_target_site_credit_sum = 0
    wrong_target_site_before_score_count = 0
    target_action_executed_within_window_count = 0
    score_after_target_action_count = 0
    executed_action_observed_count = 0
    executed_action_match_count = 0
    executed_action_mismatch_count = 0
    modes: Counter[str] = Counter()
    blocked_reasons: Counter[str] = Counter()
    low_commitment_probe_blocked_reasons: Counter[str] = Counter()
    local_grounding_sources: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    by_target_action: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    execution_mismatch_samples: list[dict[str, Any]] = []

    for row in row_list:
        game_id = str(row.get("game_id", "") or "unknown")
        row_applied = False
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        (
            row_direct_score_probe_without_goal,
            row_direct_score_probe_without_goal_failure,
        ) = _target_verification_direct_score_probe_counts(row)
        (
            row_direct_score_probe_contradicted_targets,
            row_post_direct_score_probe_contradiction_reentry,
        ) = _direct_score_probe_reentry_counts(row)
        if row_direct_score_probe_without_goal:
            direct_score_probe_without_goal_count += row_direct_score_probe_without_goal
            by_game[game_id]["direct_score_probe_without_goal_count"] += (
                row_direct_score_probe_without_goal
            )
        if row_direct_score_probe_without_goal_failure:
            direct_score_probe_without_goal_failure_count += row_direct_score_probe_without_goal_failure
            by_game[game_id]["direct_score_probe_without_goal_failure_count"] += (
                row_direct_score_probe_without_goal_failure
            )
        if row_direct_score_probe_contradicted_targets:
            direct_score_probe_contradicted_target_count += (
                row_direct_score_probe_contradicted_targets
            )
            by_game[game_id]["direct_score_probe_contradicted_target_count"] += (
                row_direct_score_probe_contradicted_targets
            )
        if row_post_direct_score_probe_contradiction_reentry:
            post_direct_score_probe_contradiction_reentry_count += (
                row_post_direct_score_probe_contradiction_reentry
            )
            by_game[game_id][
                "post_direct_score_probe_contradiction_reentry_count"
            ] += row_post_direct_score_probe_contradiction_reentry
        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            decision = _rod_score_precondition_action_from_trace_item(item)
            if not decision:
                continue
            trace_step_count += 1
            by_game[game_id]["trace_step_count"] += 1
            if bool(decision.get("enabled")):
                enabled_step_count += 1
                by_game[game_id]["enabled_step_count"] += 1
            if bool(decision.get("macro_solver_step")):
                macro_step_count += 1
                by_game[game_id]["macro_step_count"] += 1
            if bool(decision.get("low_commitment_probe_eligible")):
                low_commitment_probe_eligible_count += 1
                by_game[game_id]["low_commitment_probe_eligible_count"] += 1
            low_probe_blocked = str(decision.get("low_commitment_probe_blocked_reason", "") or "")
            if low_probe_blocked:
                low_commitment_probe_blocked_reasons[low_probe_blocked] += 1
                by_game[game_id][f"low_commitment_probe_blocked_reason:{low_probe_blocked}"] += 1
            blocked = str(decision.get("blocked_reason", "") or "")
            if blocked:
                blocked_step_count += 1
                blocked_reasons[blocked] += 1
                by_game[game_id]["blocked_step_count"] += 1
                by_game[game_id][f"blocked_reason:{blocked}"] += 1
            mode = str(decision.get("mode", "") or "")
            mode_key = mode or "unknown"
            target_action = _normalize_action_name(decision.get("target_action", "") or "")
            target_action_key = target_action or "unknown"
            local_success_count = _as_int(decision.get("local_success_count"), 0)
            local_failure_count = _as_int(decision.get("local_failure_count"), 0)
            local_score_success_count = _as_int(decision.get("local_score_success_count"), 0)
            local_goal_success_count = _as_int(decision.get("local_goal_success_count"), 0)
            local_progress_only_success_count = _as_int(
                decision.get("local_progress_only_success_count"),
                0,
            )
            score_grounded_score_target = bool(
                decision.get("score_grounded_score_target")
                or local_score_success_count > 0
                or local_goal_success_count > 0
            )
            local_grounding_source = str(decision.get("local_grounding_source", "") or "")
            if not local_grounding_source:
                if local_score_success_count > 0:
                    local_grounding_source = "score_delta"
                elif local_goal_success_count > 0:
                    local_grounding_source = "goal_progress"
                elif local_progress_only_success_count > 0 or local_success_count > 0:
                    local_grounding_source = "progress_only"
                else:
                    local_grounding_source = "none"
            locally_grounded_score_target = bool(
                decision.get("locally_grounded_score_target")
                or local_success_count > 0
                or local_score_success_count > 0
                or local_goal_success_count > 0
            )
            if mode or bool(decision.get("applied")):
                by_mode[mode_key]["trace_step_count"] += 1
                if bool(decision.get("enabled")):
                    by_mode[mode_key]["enabled_step_count"] += 1
                if bool(decision.get("macro_solver_step")):
                    by_mode[mode_key]["macro_step_count"] += 1
                if bool(decision.get("low_commitment_probe_eligible")):
                    by_mode[mode_key]["low_commitment_probe_eligible_count"] += 1
            if target_action or bool(decision.get("applied")):
                by_target_action[target_action_key]["trace_step_count"] += 1
                if bool(decision.get("enabled")):
                    by_target_action[target_action_key]["enabled_step_count"] += 1
                if bool(decision.get("macro_solver_step")):
                    by_target_action[target_action_key]["macro_step_count"] += 1
                if bool(decision.get("low_commitment_probe_eligible")):
                    by_target_action[target_action_key]["low_commitment_probe_eligible_count"] += 1
            if bool(decision.get("applied")):
                applied_step_count += 1
                by_game[game_id]["applied_step_count"] += 1
                row_applied = True
                if bool(decision.get("low_commitment_probe_applied")):
                    low_commitment_probe_applied_step_count += 1
                    by_game[game_id]["low_commitment_probe_applied_step_count"] += 1
                    by_mode[mode_key]["low_commitment_probe_applied_step_count"] += 1
                    by_target_action[target_action_key]["low_commitment_probe_applied_step_count"] += 1
                if mode:
                    modes[mode] += 1
                by_mode[mode_key]["applied_step_count"] += 1
                by_target_action[target_action_key]["applied_step_count"] += 1
                if locally_grounded_score_target:
                    locally_grounded_applied_step_count += 1
                    by_game[game_id]["locally_grounded_applied_step_count"] += 1
                    by_mode[mode_key]["locally_grounded_applied_step_count"] += 1
                    by_target_action[target_action_key]["locally_grounded_applied_step_count"] += 1
                else:
                    ungrounded_applied_step_count += 1
                    by_game[game_id]["ungrounded_applied_step_count"] += 1
                    by_mode[mode_key]["ungrounded_applied_step_count"] += 1
                    by_target_action[target_action_key]["ungrounded_applied_step_count"] += 1
                if score_grounded_score_target:
                    score_grounded_applied_step_count += 1
                    by_game[game_id]["score_grounded_applied_step_count"] += 1
                    by_mode[mode_key]["score_grounded_applied_step_count"] += 1
                    by_target_action[target_action_key]["score_grounded_applied_step_count"] += 1
                elif locally_grounded_score_target and local_grounding_source == "progress_only":
                    progress_only_grounded_applied_step_count += 1
                    by_game[game_id]["progress_only_grounded_applied_step_count"] += 1
                    by_mode[mode_key]["progress_only_grounded_applied_step_count"] += 1
                    by_target_action[target_action_key]["progress_only_grounded_applied_step_count"] += 1
                if local_grounding_source:
                    local_grounding_sources[local_grounding_source] += 1
                    by_game[game_id][f"local_grounding_source:{local_grounding_source}"] += 1
                    by_mode[mode_key][f"local_grounding_source:{local_grounding_source}"] += 1
                    by_target_action[target_action_key][f"local_grounding_source:{local_grounding_source}"] += 1
                current_score = _as_int(item.get("score_before"), 0)
                next_score = _as_int(row.get("final_score"), current_score)
                if index + 1 < len(trace) and isinstance(trace[index + 1], Mapping):
                    next_score = _as_int(trace[index + 1].get("score_before"), current_score)
                score_delta_after_action = next_score - current_score
                applied_score_delta_sum += score_delta_after_action
                by_game[game_id]["applied_score_delta_sum"] += score_delta_after_action
                by_mode[mode_key]["applied_score_delta_sum"] += score_delta_after_action
                by_target_action[target_action_key]["applied_score_delta_sum"] += score_delta_after_action
                if score_delta_after_action > 0:
                    applied_positive_score_step_count += 1
                    by_game[game_id]["applied_positive_score_step_count"] += 1
                    by_mode[mode_key]["applied_positive_score_step_count"] += 1
                    by_target_action[target_action_key]["applied_positive_score_step_count"] += 1
                delayed_scores = [
                    _as_int(future.get("score_before"), current_score)
                    for future in trace[index + 1 : index + 1 + delayed_credit_window]
                    if isinstance(future, Mapping)
                ]
                delayed_delta = max([current_score, *delayed_scores]) - current_score
                delayed_score_delta_sum += delayed_delta
                by_game[game_id]["delayed_score_delta_sum"] += delayed_delta
                by_mode[mode_key]["delayed_score_delta_sum"] += delayed_delta
                by_target_action[target_action_key]["delayed_score_delta_sum"] += delayed_delta
                delayed_delay_steps = 0
                if delayed_delta > 0:
                    delayed_positive_score_step_count += 1
                    by_game[game_id]["delayed_positive_score_step_count"] += 1
                    by_mode[mode_key]["delayed_positive_score_step_count"] += 1
                    by_target_action[target_action_key]["delayed_positive_score_step_count"] += 1
                    for offset, score in enumerate(delayed_scores, start=1):
                        if score > current_score:
                            delayed_delay_steps = offset
                            break
                lookahead_items = [
                    future
                    for future in trace[index + 1 : index + 1 + delayed_credit_window]
                    if isinstance(future, Mapping)
                ]
                shifted_baseline_index = index + shifted_credit_window_offset
                shifted_baseline_score = current_score
                if shifted_baseline_index < len(trace) and isinstance(trace[shifted_baseline_index], Mapping):
                    shifted_baseline_score = _as_int(
                        trace[shifted_baseline_index].get("score_before"),
                        current_score,
                    )
                shifted_scores = [
                    _as_int(future.get("score_before"), shifted_baseline_score)
                    for future in trace[
                        shifted_baseline_index + 1 : shifted_baseline_index + 1 + delayed_credit_window
                    ]
                    if isinstance(future, Mapping)
                ]
                shifted_delta = max([shifted_baseline_score, *shifted_scores]) - shifted_baseline_score
                shifted_delayed_score_delta_sum += shifted_delta
                by_game[game_id]["shifted_delayed_score_delta_sum"] += shifted_delta
                by_mode[mode_key]["shifted_delayed_score_delta_sum"] += shifted_delta
                by_target_action[target_action_key]["shifted_delayed_score_delta_sum"] += shifted_delta
                if shifted_delta > 0:
                    shifted_delayed_positive_score_step_count += 1
                    by_game[game_id]["shifted_delayed_positive_score_step_count"] += 1
                    by_mode[mode_key]["shifted_delayed_positive_score_step_count"] += 1
                    by_target_action[target_action_key]["shifted_delayed_positive_score_step_count"] += 1
                pre_action_delta = _pre_action_score_delta(
                    trace,
                    index=index,
                    window=delayed_credit_window,
                    current_score=current_score,
                )
                pre_action_score_delta_sum += pre_action_delta
                by_game[game_id]["pre_action_score_delta_sum"] += pre_action_delta
                by_mode[mode_key]["pre_action_score_delta_sum"] += pre_action_delta
                by_target_action[target_action_key]["pre_action_score_delta_sum"] += pre_action_delta
                if pre_action_delta > 0:
                    pre_action_positive_score_step_count += 1
                    by_game[game_id]["pre_action_positive_score_step_count"] += 1
                    by_mode[mode_key]["pre_action_positive_score_step_count"] += 1
                    by_target_action[target_action_key]["pre_action_positive_score_step_count"] += 1
                random_control_index = _deterministic_control_index(index, len(trace))
                random_step_delta = _window_score_delta_from_index(
                    trace,
                    baseline_index=random_control_index,
                    window=delayed_credit_window,
                    default_score=current_score,
                )
                random_step_score_delta_sum += random_step_delta
                by_game[game_id]["random_step_score_delta_sum"] += random_step_delta
                by_mode[mode_key]["random_step_score_delta_sum"] += random_step_delta
                by_target_action[target_action_key]["random_step_score_delta_sum"] += random_step_delta
                if random_step_delta > 0:
                    random_step_positive_score_step_count += 1
                    by_game[game_id]["random_step_positive_score_step_count"] += 1
                    by_mode[mode_key]["random_step_positive_score_step_count"] += 1
                    by_target_action[target_action_key]["random_step_positive_score_step_count"] += 1
                previous = _normalize_action_name(
                    decision.get("previous_action", "") or decision.get("current_action", "") or ""
                )
                selected = _normalize_action_name(decision.get("selected_action", "") or "")
                executed_action = _normalize_action_name(item.get("action"))
                target_site = _site_signature(decision.get("target_site"))
                wrong_action = _wrong_action_name(target_action)
                wrong_target_action_before_score = False
                wrong_target_action_delay_steps = 0
                wrong_target_action_credit = 0
                if wrong_action and delayed_delta > 0 and delayed_delay_steps > 0:
                    if executed_action == wrong_action:
                        wrong_target_action_before_score = True
                        wrong_target_action_delay_steps = 0
                    else:
                        for offset, future in enumerate(lookahead_items, start=1):
                            if offset >= delayed_delay_steps:
                                break
                            if _normalize_action_name(future.get("action")) == wrong_action:
                                wrong_target_action_before_score = True
                                wrong_target_action_delay_steps = offset
                                break
                    if wrong_target_action_before_score:
                        wrong_target_action_credit = delayed_delta
                        wrong_target_action_credit_sum += wrong_target_action_credit
                        wrong_target_action_before_score_count += 1
                        by_game[game_id]["wrong_target_action_credit_sum"] += wrong_target_action_credit
                        by_game[game_id]["wrong_target_action_before_score_count"] += 1
                        by_mode[mode_key]["wrong_target_action_credit_sum"] += wrong_target_action_credit
                        by_mode[mode_key]["wrong_target_action_before_score_count"] += 1
                        by_target_action[target_action_key][
                            "wrong_target_action_credit_sum"
                        ] += wrong_target_action_credit
                        by_target_action[target_action_key]["wrong_target_action_before_score_count"] += 1
                wrong_target_site_before_score = False
                wrong_target_site_delay_steps = 0
                wrong_target_site_credit = 0
                if target_site and delayed_delta > 0 and delayed_delay_steps > 0:
                    current_site = _trace_item_site_signature(item)
                    if current_site and current_site != target_site:
                        wrong_target_site_before_score = True
                        wrong_target_site_delay_steps = 0
                    else:
                        for offset, future in enumerate(lookahead_items, start=1):
                            if offset >= delayed_delay_steps:
                                break
                            future_site = _trace_item_site_signature(future)
                            if future_site and future_site != target_site:
                                wrong_target_site_before_score = True
                                wrong_target_site_delay_steps = offset
                                break
                    if wrong_target_site_before_score:
                        wrong_target_site_credit = delayed_delta
                        wrong_target_site_credit_sum += wrong_target_site_credit
                        wrong_target_site_before_score_count += 1
                        by_game[game_id]["wrong_target_site_credit_sum"] += wrong_target_site_credit
                        by_game[game_id]["wrong_target_site_before_score_count"] += 1
                        by_mode[mode_key]["wrong_target_site_credit_sum"] += wrong_target_site_credit
                        by_mode[mode_key]["wrong_target_site_before_score_count"] += 1
                        by_target_action[target_action_key][
                            "wrong_target_site_credit_sum"
                        ] += wrong_target_site_credit
                        by_target_action[target_action_key]["wrong_target_site_before_score_count"] += 1
                target_action_delay_steps = 0
                target_action_executed_within_window = False
                score_after_target_action = False
                if target_action:
                    if executed_action == target_action:
                        target_action_delay_steps = 0
                        target_action_executed_within_window = True
                    else:
                        for offset, future in enumerate(lookahead_items, start=1):
                            if _normalize_action_name(future.get("action")) == target_action:
                                target_action_delay_steps = offset
                                target_action_executed_within_window = True
                                break
                    if target_action_executed_within_window:
                        target_action_executed_within_window_count += 1
                        by_game[game_id]["target_action_executed_within_window_count"] += 1
                        by_mode[mode_key]["target_action_executed_within_window_count"] += 1
                        by_target_action[target_action_key]["target_action_executed_within_window_count"] += 1
                        score_after_target_action = bool(
                            delayed_delta > 0
                            and delayed_delay_steps > 0
                            and target_action_delay_steps < delayed_delay_steps
                        )
                        if score_after_target_action:
                            score_after_target_action_count += 1
                            by_game[game_id]["score_after_target_action_count"] += 1
                            by_mode[mode_key]["score_after_target_action_count"] += 1
                            by_target_action[target_action_key]["score_after_target_action_count"] += 1
                execution_match = None
                if executed_action:
                    executed_action_observed_count += 1
                    by_game[game_id]["executed_action_observed_count"] += 1
                    by_mode[mode_key]["executed_action_observed_count"] += 1
                    by_target_action[target_action_key]["executed_action_observed_count"] += 1
                    execution_match = bool(selected and executed_action == selected)
                    if execution_match:
                        executed_action_match_count += 1
                        by_game[game_id]["executed_action_match_count"] += 1
                        by_mode[mode_key]["executed_action_match_count"] += 1
                        by_target_action[target_action_key]["executed_action_match_count"] += 1
                    else:
                        executed_action_mismatch_count += 1
                        by_game[game_id]["executed_action_mismatch_count"] += 1
                        by_mode[mode_key]["executed_action_mismatch_count"] += 1
                        by_target_action[target_action_key]["executed_action_mismatch_count"] += 1
                        if len(execution_mismatch_samples) < 8:
                            execution_mismatch_samples.append(
                                {
                                    "game_id": game_id,
                                    "variant": str(row.get("variant", "") or "unknown"),
                                    "step": _as_int(item.get("step"), 0),
                                    "phase": str(item.get("phase", "") or ""),
                                    "mode": mode,
                                    "target_site": _site_signature(decision.get("target_site")),
                                    "selected_action": selected,
                                    "executed_action": executed_action,
                                }
                            )
                applied_effect_class = "no_delayed_score"
                if delayed_delta > 0:
                    if bool(execution_match) and (
                        score_delta_after_action > 0 or score_after_target_action
                    ):
                        applied_effect_class = "causally_supported"
                    else:
                        applied_effect_class = "associated_only"
                if applied_effect_class == "causally_supported":
                    causally_supported_applied_step_count += 1
                    causally_supported_score_delta_sum += delayed_delta
                    by_game[game_id]["causally_supported_applied_step_count"] += 1
                    by_game[game_id]["causally_supported_score_delta_sum"] += delayed_delta
                    by_mode[mode_key]["causally_supported_applied_step_count"] += 1
                    by_mode[mode_key]["causally_supported_score_delta_sum"] += delayed_delta
                    by_target_action[target_action_key]["causally_supported_applied_step_count"] += 1
                    by_target_action[target_action_key]["causally_supported_score_delta_sum"] += delayed_delta
                elif applied_effect_class == "associated_only":
                    associated_only_applied_step_count += 1
                    by_game[game_id]["associated_only_applied_step_count"] += 1
                    by_mode[mode_key]["associated_only_applied_step_count"] += 1
                    by_target_action[target_action_key]["associated_only_applied_step_count"] += 1
                else:
                    no_delayed_score_applied_step_count += 1
                    by_game[game_id]["no_delayed_score_applied_step_count"] += 1
                    by_mode[mode_key]["no_delayed_score_applied_step_count"] += 1
                    by_target_action[target_action_key]["no_delayed_score_applied_step_count"] += 1
                if previous or selected:
                    transitions[f"{previous}->{selected}"] += 1
                if len(samples) < 20:
                    samples.append(
                        {
                            "game_id": game_id,
                            "variant": str(row.get("variant", "") or "unknown"),
                            "step": _as_int(item.get("step"), 0),
                            "phase": str(item.get("phase", "") or ""),
                            "previous_action": previous,
                            "selected_action": selected,
                            "executed_action": executed_action,
                            "execution_match": execution_match,
                            "mode": mode,
                            "target_site": _site_signature(decision.get("target_site")),
                            "target_action": target_action,
                            "rod_net_support": _as_int(decision.get("rod_net_support"), 0),
                            "local_success_count": local_success_count,
                            "local_failure_count": local_failure_count,
                            "local_score_success_count": local_score_success_count,
                            "local_goal_success_count": local_goal_success_count,
                            "local_progress_only_success_count": local_progress_only_success_count,
                            "locally_grounded_score_target": locally_grounded_score_target,
                            "score_grounded_score_target": score_grounded_score_target,
                            "local_grounding_source": local_grounding_source,
                            "applied_effect_class": applied_effect_class,
                            "score_delta_after_action": score_delta_after_action,
                            "delayed_score_delta": delayed_delta,
                            "delayed_delay_steps": delayed_delay_steps,
                            "shifted_delayed_score_delta": shifted_delta,
                            "pre_action_score_delta": pre_action_delta,
                            "random_control_step": _as_int(
                                trace[random_control_index].get("step"),
                                random_control_index,
                            )
                            if isinstance(trace[random_control_index], Mapping)
                            else random_control_index,
                            "random_step_score_delta": random_step_delta,
                            "target_action_executed_within_window": target_action_executed_within_window,
                            "target_action_delay_steps": target_action_delay_steps,
                            "score_after_target_action": score_after_target_action,
                            "wrong_target_action": wrong_action,
                            "wrong_target_action_before_score": wrong_target_action_before_score,
                            "wrong_target_action_delay_steps": wrong_target_action_delay_steps,
                            "wrong_target_action_credit": wrong_target_action_credit,
                            "wrong_target_site_before_score": wrong_target_site_before_score,
                            "wrong_target_site_delay_steps": wrong_target_site_delay_steps,
                            "wrong_target_site_credit": wrong_target_site_credit,
                        }
                    )
        if row_applied:
            rows_with_applied += 1
            score_sum_with_applied += _as_int(row.get("final_score"), 0)

    def convert(counter: Counter[str]) -> dict[str, Any]:
        trace_steps = int(counter.get("trace_step_count", 0))
        applied = int(counter.get("applied_step_count", 0))
        reason_counts = {
            key.split(":", 1)[1]: int(value)
            for key, value in counter.items()
            if key.startswith("blocked_reason:")
        }
        delayed_sum = int(counter.get("delayed_score_delta_sum", 0))
        shifted_sum = int(counter.get("shifted_delayed_score_delta_sum", 0))
        pre_action_sum = int(counter.get("pre_action_score_delta_sum", 0))
        random_sum = int(counter.get("random_step_score_delta_sum", 0))
        wrong_action_sum = int(counter.get("wrong_target_action_credit_sum", 0))
        wrong_site_sum = int(counter.get("wrong_target_site_credit_sum", 0))
        control_max = max(shifted_sum, pre_action_sum, random_sum, wrong_action_sum, wrong_site_sum)
        execution_match_rate = _safe_div(
            int(counter.get("executed_action_match_count", 0)),
            int(counter.get("executed_action_observed_count", 0)),
        )
        score_after_target_action_rate = _safe_div(
            int(counter.get("score_after_target_action_count", 0)),
            applied,
        )
        causal_margin = delayed_sum - control_max
        causal_credit_confidence = round(
            max(0.0, _safe_div(causal_margin, delayed_sum))
            * execution_match_rate
            * score_after_target_action_rate,
            6,
        )
        return {
            "trace_step_count": trace_steps,
            "enabled_step_count": int(counter.get("enabled_step_count", 0)),
            "macro_step_count": int(counter.get("macro_step_count", 0)),
            "applied_step_count": applied,
            "applied_rate": _safe_div(applied, trace_steps),
            "blocked_step_count": int(counter.get("blocked_step_count", 0)),
            "blocked_reasons": dict(sorted(reason_counts.items())),
            "low_commitment_probe_eligible_count": int(
                counter.get("low_commitment_probe_eligible_count", 0)
            ),
            "low_commitment_probe_applied_step_count": int(
                counter.get("low_commitment_probe_applied_step_count", 0)
            ),
            "low_commitment_probe_blocked_reasons": {
                key.split(":", 1)[1]: int(value)
                for key, value in sorted(counter.items())
                if key.startswith("low_commitment_probe_blocked_reason:")
            },
            "score_target_unverified_locally_blocked_count": int(
                counter.get("blocked_reason:score_target_unverified_locally", 0)
            ),
            "locally_grounded_applied_step_count": int(
                counter.get("locally_grounded_applied_step_count", 0)
            ),
            "score_grounded_applied_step_count": int(
                counter.get("score_grounded_applied_step_count", 0)
            ),
            "progress_only_grounded_applied_step_count": int(
                counter.get("progress_only_grounded_applied_step_count", 0)
            ),
            "ungrounded_applied_step_count": int(counter.get("ungrounded_applied_step_count", 0)),
            "locally_grounded_applied_rate": _safe_div(
                int(counter.get("locally_grounded_applied_step_count", 0)),
                applied,
            ),
            "score_grounded_applied_rate": _safe_div(
                int(counter.get("score_grounded_applied_step_count", 0)),
                applied,
            ),
            "direct_score_probe_without_goal_count": int(
                counter.get("direct_score_probe_without_goal_count", 0)
            ),
            "direct_score_probe_without_goal_failure_count": int(
                counter.get("direct_score_probe_without_goal_failure_count", 0)
            ),
            "direct_score_probe_contradicted_target_count": int(
                counter.get("direct_score_probe_contradicted_target_count", 0)
            ),
            "post_direct_score_probe_contradiction_reentry_count": int(
                counter.get("post_direct_score_probe_contradiction_reentry_count", 0)
            ),
            "local_grounding_sources": {
                key.split(":", 1)[1]: int(value)
                for key, value in sorted(counter.items())
                if key.startswith("local_grounding_source:")
            },
            "causally_supported_applied_step_count": int(
                counter.get("causally_supported_applied_step_count", 0)
            ),
            "associated_only_applied_step_count": int(
                counter.get("associated_only_applied_step_count", 0)
            ),
            "no_delayed_score_applied_step_count": int(
                counter.get("no_delayed_score_applied_step_count", 0)
            ),
            "causally_supported_rate": _safe_div(
                int(counter.get("causally_supported_applied_step_count", 0)),
                applied,
            ),
            "causally_supported_score_delta_sum": int(
                counter.get("causally_supported_score_delta_sum", 0)
            ),
            "score_path_diagnosis": _diagnose_rod_score_precondition_action_path(counter),
            "applied_score_delta_sum": int(counter.get("applied_score_delta_sum", 0)),
            "applied_positive_score_step_count": int(counter.get("applied_positive_score_step_count", 0)),
            "applied_positive_score_rate": _safe_div(
                int(counter.get("applied_positive_score_step_count", 0)),
                applied,
            ),
            "delayed_score_delta_sum": delayed_sum,
            "delayed_positive_score_step_count": int(counter.get("delayed_positive_score_step_count", 0)),
            "delayed_positive_score_rate": _safe_div(
                int(counter.get("delayed_positive_score_step_count", 0)),
                applied,
            ),
            "shifted_delayed_score_delta_sum": shifted_sum,
            "shifted_delayed_positive_score_step_count": int(
                counter.get("shifted_delayed_positive_score_step_count", 0)
            ),
            "shifted_delayed_positive_score_rate": _safe_div(
                int(counter.get("shifted_delayed_positive_score_step_count", 0)),
                applied,
            ),
            "delayed_minus_shifted_score_delta_sum": delayed_sum - shifted_sum,
            "pre_action_score_delta_sum": pre_action_sum,
            "pre_action_positive_score_step_count": int(
                counter.get("pre_action_positive_score_step_count", 0)
            ),
            "random_step_score_delta_sum": random_sum,
            "random_step_positive_score_step_count": int(
                counter.get("random_step_positive_score_step_count", 0)
            ),
            "wrong_target_action_credit_sum": wrong_action_sum,
            "wrong_target_action_before_score_count": int(
                counter.get("wrong_target_action_before_score_count", 0)
            ),
            "wrong_target_site_credit_sum": wrong_site_sum,
            "wrong_target_site_before_score_count": int(
                counter.get("wrong_target_site_before_score_count", 0)
            ),
            "negative_control_max_score_delta_sum": control_max,
            "real_vs_random_step_delta": delayed_sum - random_sum,
            "real_vs_pre_action_delta": delayed_sum - pre_action_sum,
            "real_vs_wrong_action_delta": delayed_sum - wrong_action_sum,
            "real_vs_wrong_site_delta": delayed_sum - wrong_site_sum,
            "causal_credit_margin": causal_margin,
            "causal_credit_confidence": causal_credit_confidence,
            "target_action_executed_within_window_count": int(
                counter.get("target_action_executed_within_window_count", 0)
            ),
            "target_action_executed_within_window_rate": _safe_div(
                int(counter.get("target_action_executed_within_window_count", 0)),
                applied,
            ),
            "score_after_target_action_count": int(counter.get("score_after_target_action_count", 0)),
            "score_after_target_action_rate": _safe_div(
                int(counter.get("score_after_target_action_count", 0)),
                applied,
            ),
            "executed_action_observed_count": int(counter.get("executed_action_observed_count", 0)),
            "executed_action_match_count": int(counter.get("executed_action_match_count", 0)),
            "executed_action_mismatch_count": int(counter.get("executed_action_mismatch_count", 0)),
            "executed_action_match_rate": execution_match_rate,
        }

    negative_control_max_score_delta_sum = max(
        shifted_delayed_score_delta_sum,
        pre_action_score_delta_sum,
        random_step_score_delta_sum,
        wrong_target_action_credit_sum,
        wrong_target_site_credit_sum,
    )
    causal_credit_margin = delayed_score_delta_sum - negative_control_max_score_delta_sum
    executed_action_match_rate = _safe_div(
        executed_action_match_count,
        executed_action_observed_count,
    )
    score_after_target_action_rate = _safe_div(
        score_after_target_action_count,
        applied_step_count,
    )
    causal_credit_confidence = round(
        max(0.0, _safe_div(causal_credit_margin, delayed_score_delta_sum))
        * executed_action_match_rate
        * score_after_target_action_rate,
        6,
    )
    by_game_summary = {key: convert(value) for key, value in sorted(by_game.items())}
    by_mode_summary = {key: convert(value) for key, value in sorted(by_mode.items())}
    by_target_action_summary = {
        key: convert(value) for key, value in sorted(by_target_action.items())
    }
    by_game_status_histogram: Counter[str] = Counter()
    by_game_reason_counts: Counter[str] = Counter()
    for game_summary in by_game_summary.values():
        diagnosis = game_summary.get("score_path_diagnosis")
        if not isinstance(diagnosis, Mapping):
            continue
        status = str(diagnosis.get("status", "") or "")
        if status:
            by_game_status_histogram[status] += 1
        reason_counts = diagnosis.get("reason_counts")
        if isinstance(reason_counts, Mapping):
            for reason, raw_count in reason_counts.items():
                count = _as_int(raw_count, 0)
                if count:
                    by_game_reason_counts[str(reason)] += count

    return {
        "schema_version": "eca_rod_score_precondition_action_summary.v6",
        "delayed_credit_window": delayed_credit_window,
        "shifted_credit_window_offset": shifted_credit_window_offset,
        "trace_step_count": trace_step_count,
        "enabled_step_count": enabled_step_count,
        "macro_step_count": macro_step_count,
        "applied_step_count": applied_step_count,
        "applied_rate": _safe_div(applied_step_count, trace_step_count),
        "blocked_step_count": blocked_step_count,
        "low_commitment_probe_eligible_count": low_commitment_probe_eligible_count,
        "low_commitment_probe_applied_step_count": low_commitment_probe_applied_step_count,
        "low_commitment_probe_blocked_reasons": dict(
            sorted(low_commitment_probe_blocked_reasons.items())
        ),
        "score_target_unverified_locally_blocked_count": int(
            blocked_reasons.get("score_target_unverified_locally", 0)
        ),
        "locally_grounded_applied_step_count": locally_grounded_applied_step_count,
        "score_grounded_applied_step_count": score_grounded_applied_step_count,
        "progress_only_grounded_applied_step_count": progress_only_grounded_applied_step_count,
        "ungrounded_applied_step_count": ungrounded_applied_step_count,
        "locally_grounded_applied_rate": _safe_div(
            locally_grounded_applied_step_count,
            applied_step_count,
        ),
        "score_grounded_applied_rate": _safe_div(
            score_grounded_applied_step_count,
            applied_step_count,
        ),
        "direct_score_probe_without_goal_count": direct_score_probe_without_goal_count,
        "direct_score_probe_without_goal_failure_count": (
            direct_score_probe_without_goal_failure_count
        ),
        "direct_score_probe_contradicted_target_count": (
            direct_score_probe_contradicted_target_count
        ),
        "post_direct_score_probe_contradiction_reentry_count": (
            post_direct_score_probe_contradiction_reentry_count
        ),
        "local_grounding_sources": dict(sorted(local_grounding_sources.items())),
        "causally_supported_applied_step_count": causally_supported_applied_step_count,
        "associated_only_applied_step_count": associated_only_applied_step_count,
        "no_delayed_score_applied_step_count": no_delayed_score_applied_step_count,
        "causally_supported_rate": _safe_div(
            causally_supported_applied_step_count,
            applied_step_count,
        ),
        "causally_supported_score_delta_sum": causally_supported_score_delta_sum,
        "score_path_diagnosis": _diagnose_rod_score_precondition_action_path(
            Counter(
                {
                    "applied_step_count": applied_step_count,
                    "causally_supported_applied_step_count": causally_supported_applied_step_count,
                    "associated_only_applied_step_count": associated_only_applied_step_count,
                    "no_delayed_score_applied_step_count": no_delayed_score_applied_step_count,
                    "direct_score_probe_without_goal_count": (
                        direct_score_probe_without_goal_count
                    ),
                    "post_direct_score_probe_contradiction_reentry_count": (
                        post_direct_score_probe_contradiction_reentry_count
                    ),
                    "blocked_reason:score_target_unverified_locally": int(
                        blocked_reasons.get("score_target_unverified_locally", 0)
                    ),
                }
            )
        ),
        "by_game_score_path_diagnosis_status_histogram": dict(
            sorted(by_game_status_histogram.items())
        ),
        "by_game_score_path_diagnosis_reason_counts": dict(
            sorted(by_game_reason_counts.items())
        ),
        "applied_score_delta_sum": applied_score_delta_sum,
        "applied_positive_score_step_count": applied_positive_score_step_count,
        "applied_positive_score_rate": _safe_div(applied_positive_score_step_count, applied_step_count),
        "delayed_score_delta_sum": delayed_score_delta_sum,
        "delayed_positive_score_step_count": delayed_positive_score_step_count,
        "delayed_positive_score_rate": _safe_div(delayed_positive_score_step_count, applied_step_count),
        "shifted_delayed_score_delta_sum": shifted_delayed_score_delta_sum,
        "shifted_delayed_positive_score_step_count": shifted_delayed_positive_score_step_count,
        "shifted_delayed_positive_score_rate": _safe_div(
            shifted_delayed_positive_score_step_count,
            applied_step_count,
        ),
        "delayed_minus_shifted_score_delta_sum": (
            delayed_score_delta_sum - shifted_delayed_score_delta_sum
        ),
        "pre_action_score_delta_sum": pre_action_score_delta_sum,
        "pre_action_positive_score_step_count": pre_action_positive_score_step_count,
        "random_step_score_delta_sum": random_step_score_delta_sum,
        "random_step_positive_score_step_count": random_step_positive_score_step_count,
        "wrong_target_action_credit_sum": wrong_target_action_credit_sum,
        "wrong_target_action_before_score_count": wrong_target_action_before_score_count,
        "wrong_target_site_credit_sum": wrong_target_site_credit_sum,
        "wrong_target_site_before_score_count": wrong_target_site_before_score_count,
        "negative_control_max_score_delta_sum": negative_control_max_score_delta_sum,
        "real_vs_shifted_delta": delayed_score_delta_sum - shifted_delayed_score_delta_sum,
        "real_vs_random_step_delta": delayed_score_delta_sum - random_step_score_delta_sum,
        "real_vs_pre_action_delta": delayed_score_delta_sum - pre_action_score_delta_sum,
        "real_vs_wrong_action_delta": delayed_score_delta_sum - wrong_target_action_credit_sum,
        "real_vs_wrong_site_delta": delayed_score_delta_sum - wrong_target_site_credit_sum,
        "causal_credit_margin": causal_credit_margin,
        "causal_credit_confidence": causal_credit_confidence,
        "target_action_executed_within_window_count": target_action_executed_within_window_count,
        "target_action_executed_within_window_rate": _safe_div(
            target_action_executed_within_window_count,
            applied_step_count,
        ),
        "score_after_target_action_count": score_after_target_action_count,
        "score_after_target_action_rate": score_after_target_action_rate,
        "executed_action_observed_count": executed_action_observed_count,
        "executed_action_match_count": executed_action_match_count,
        "executed_action_mismatch_count": executed_action_mismatch_count,
        "executed_action_match_rate": executed_action_match_rate,
        "rows_with_applied": rows_with_applied,
        "score_sum_with_applied": score_sum_with_applied,
        "modes": dict(sorted(modes.items())),
        "by_mode": by_mode_summary,
        "by_target_action": by_target_action_summary,
        "transitions": dict(sorted(transitions.items())),
        "blocked_reasons": dict(sorted(blocked_reasons.items())),
        "samples": samples,
        "execution_mismatch_samples": execution_mismatch_samples,
        "by_game": by_game_summary,
    }


def _site_signature(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return f"{int(value[0])},{int(value[1])}"
        except (TypeError, ValueError):
            return ""
    return ""


def _rod_score_precondition_context_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    trace_step_count = 0
    candidate_step_count = 0
    primary_step_count = 0
    promoted_step_count = 0
    rows_with_candidate = 0
    rows_with_primary = 0
    rows_with_promoted = 0
    score_sum_with_candidate = 0
    score_sum_with_primary = 0
    score_sum_with_promoted = 0
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []

    for row in row_list:
        game_id = str(row.get("game_id", "") or "unknown")
        row_candidate = False
        row_primary = False
        row_promoted = False
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            trace_step_count += 1
            by_game[game_id]["trace_step_count"] += 1
            control = item.get("control_target")
            if not isinstance(control, Mapping):
                continue
            primary_raw = control.get("primary_target")
            primary = dict(primary_raw) if isinstance(primary_raw, Mapping) else {}
            target_candidates = [
                candidate
                for candidate in list(control.get("target_candidates") or [])
                if isinstance(candidate, Mapping)
            ]
            candidate_count = _as_int(control.get("rod_score_precondition_candidate_count"), 0)
            has_candidate = bool(
                candidate_count > 0
                or any(bool(candidate.get("rod_supported")) for candidate in target_candidates)
            )
            has_primary = bool(
                str(primary.get("kind", "") or "") == "score_precondition"
                and bool(primary.get("rod_supported"))
            )
            uncertainty = {
                str(reason)
                for reason in list(control.get("uncertainty_reasons") or [])
            }
            promoted = bool(
                has_primary
                and str(item.get("primitive", "") or "") == "score_delta_frontier_search"
                and "rod_score_precondition_available" in uncertainty
            )
            if has_candidate:
                candidate_step_count += 1
                by_game[game_id]["candidate_step_count"] += 1
                row_candidate = True
            if has_primary:
                primary_step_count += 1
                by_game[game_id]["primary_step_count"] += 1
                row_primary = True
            if promoted:
                promoted_step_count += 1
                by_game[game_id]["promoted_step_count"] += 1
                row_promoted = True
            if (has_candidate or has_primary or promoted) and len(samples) < 8:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), 0),
                        "phase": str(item.get("phase", "") or ""),
                        "primitive": str(item.get("primitive", "") or ""),
                        "candidate_count": int(candidate_count),
                        "primary_site": _site_signature(primary.get("site")),
                        "action_name": str(primary.get("action_name", "") or ""),
                        "rod_net_support": _as_int(primary.get("rod_net_support"), 0),
                        "promoted": promoted,
                    }
                )
        if row_candidate:
            rows_with_candidate += 1
            score_sum_with_candidate += _as_int(row.get("final_score"), 0)
        if row_primary:
            rows_with_primary += 1
            score_sum_with_primary += _as_int(row.get("final_score"), 0)
        if row_promoted:
            rows_with_promoted += 1
            score_sum_with_promoted += _as_int(row.get("final_score"), 0)

    def convert(counter: Counter[str]) -> dict[str, Any]:
        trace_steps = int(counter.get("trace_step_count", 0))
        candidates = int(counter.get("candidate_step_count", 0))
        promoted = int(counter.get("promoted_step_count", 0))
        return {
            "trace_step_count": trace_steps,
            "candidate_step_count": candidates,
            "primary_step_count": int(counter.get("primary_step_count", 0)),
            "promoted_step_count": promoted,
            "candidate_rate": _safe_div(candidates, trace_steps),
            "promoted_rate": _safe_div(promoted, trace_steps),
        }

    return {
        "schema_version": "eca_rod_score_precondition_context_summary.v1",
        "trace_step_count": trace_step_count,
        "candidate_step_count": candidate_step_count,
        "primary_step_count": primary_step_count,
        "promoted_step_count": promoted_step_count,
        "candidate_rate": _safe_div(candidate_step_count, trace_step_count),
        "primary_rate": _safe_div(primary_step_count, trace_step_count),
        "promoted_rate": _safe_div(promoted_step_count, trace_step_count),
        "rows_with_candidate": rows_with_candidate,
        "rows_with_primary": rows_with_primary,
        "rows_with_promoted": rows_with_promoted,
        "score_sum_with_candidate": score_sum_with_candidate,
        "score_sum_with_primary": score_sum_with_primary,
        "score_sum_with_promoted": score_sum_with_promoted,
        "samples": samples,
        "by_game": {key: convert(value) for key, value in sorted(by_game.items())},
    }


def _last_structural_rods(row: Mapping[str, Any]) -> dict[str, Any]:
    trace = row.get("trace")
    if not isinstance(trace, list):
        return {}
    for item in reversed(trace):
        if not isinstance(item, Mapping):
            continue
        rods = _structural_rods_from_trace_item(item)
        if rods:
            return rods
    return {}


def _merge_nested_counter(target: dict[str, Counter[str]], payload: Any) -> None:
    if not isinstance(payload, Mapping):
        return
    for key, counts in payload.items():
        if not isinstance(counts, Mapping):
            continue
        bucket = target.setdefault(str(key), Counter())
        for label, value in counts.items():
            bucket[str(label)] += _as_int(value, 0)


def _runtime_mechanic_candidates_from_rods(rods: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for raw in list(rods.get("episode_mechanic_candidates") or []):
        if not isinstance(raw, Mapping):
            continue
        if not bool(raw.get("runtime_selection_eligible")):
            continue
        key = str(raw.get("key", "") or "")
        key_kind = str(raw.get("key_kind", "") or "")
        if not key or not key_kind:
            continue
        candidates.append(dict(raw))
    return candidates


def _runtime_mechanic_template_count(rods: Mapping[str, Any]) -> int:
    declared = _as_int(rods.get("runtime_mechanic_template_count"), 0)
    if declared > 0:
        return declared
    return len(_runtime_mechanic_candidates_from_rods(rods))


def _runtime_mechanic_credit_by_key_summary(
    data: Mapping[str, Counter[str]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    delayed_by_key = {
        str(key): _as_int(counter.get("delayed_score_delta_sum"), 0)
        for key, counter in data.items()
    }
    for key, counter in sorted(data.items()):
        key_str = str(key)
        delayed = _as_int(counter.get("delayed_score_delta_sum"), 0)
        shifted = _as_int(counter.get("shifted_delayed_score_delta_sum"), 0)
        random_delta = _as_int(counter.get("random_step_score_delta_sum"), 0)
        wrong_site = _as_int(counter.get("wrong_target_site_credit_sum"), 0)
        wrong_key = max(
            [0]
            + [
                value
                for other_key, value in delayed_by_key.items()
                if other_key != key_str
            ]
        )
        negative_control = max(shifted, random_delta, wrong_site, wrong_key)
        margin = delayed - negative_control
        credit_steps = _as_int(counter.get("credit_step_count"), 0)
        positive_steps = _as_int(counter.get("delayed_positive_score_step_count"), 0)
        wrong_site_count = _as_int(counter.get("wrong_target_site_before_score_count"), 0)
        out[key_str] = {
            "credit_step_count": credit_steps,
            "delayed_score_delta_sum": delayed,
            "delayed_positive_score_step_count": positive_steps,
            "delayed_positive_score_rate": _safe_div(positive_steps, credit_steps),
            "shifted_delayed_score_delta_sum": shifted,
            "random_step_score_delta_sum": random_delta,
            "wrong_target_site_credit_sum": wrong_site,
            "wrong_target_site_before_score_count": wrong_site_count,
            "wrong_key_delayed_score_delta_sum": wrong_key,
            "negative_control_max_score_delta_sum": negative_control,
            "real_vs_wrong_site_delta": delayed - wrong_site,
            "real_vs_wrong_key_delta": delayed - wrong_key,
            "causal_credit_margin": margin,
            "causal_credit_margin_rate": max(0.0, _safe_div(margin, delayed)),
        }
    return out


def _nested_counter_to_dict(data: Mapping[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {
        key: dict(sorted(counter.items()))
        for key, counter in sorted(data.items())
    }


_ROD_LEAKAGE_KEY_PARTS = (
    "game_id",
    "game_hash",
    "layout",
    "canonical",
    "adapter",
    "oracle",
    "frozen_plan",
    "solver_advice",
)


def _rod_payload_leak_count(payload: Any) -> int:
    if isinstance(payload, Mapping):
        count = 0
        for key, value in payload.items():
            if any(part in str(key).lower() for part in _ROD_LEAKAGE_KEY_PARTS):
                count += 1
            count += _rod_payload_leak_count(value)
        return count
    if isinstance(payload, list):
        return sum(_rod_payload_leak_count(item) for item in payload)
    return 0


def _record_verdict_label_support(
    target: dict[str, set[str]],
    payload: Any,
    *,
    label: str,
) -> None:
    if not isinstance(payload, Mapping):
        return
    for key, counts in payload.items():
        if not isinstance(counts, Mapping):
            continue
        if _as_int(counts.get("supported"), 0) > 0:
            target.setdefault(str(key), set()).add(str(label))


def _record_cross_game_verdict_support(
    target: dict[str, set[str]],
    payload: Any,
    *,
    game_id: str,
) -> None:
    _record_verdict_label_support(target, payload, label=game_id)


def _cross_game_support_summary(data: Mapping[str, set[str]]) -> dict[str, Any]:
    supported = {
        key: sorted(games)
        for key, games in sorted(data.items())
        if len(games) >= 2
    }
    return {
        "count": len(supported),
        "keys": list(supported.keys())[:20],
        "by_key": supported,
    }


def _mechanic_promotion_block_reasons(
    evaluation: Mapping[str, Any],
    policy: RodPromotionPolicy,
) -> list[str]:
    reasons: list[str] = []
    if _as_int(evaluation.get("support_count"), 0) < policy.mechanic_min_support_count:
        reasons.append("support_count")
    if _as_int(evaluation.get("net_support"), 0) < policy.mechanic_min_net_support:
        reasons.append("net_support")
    if _as_int(evaluation.get("distinct_episode_count"), 0) < policy.mechanic_min_distinct_episodes:
        reasons.append("distinct_episode_count")
    if _as_int(evaluation.get("distinct_variant_count"), 0) < policy.mechanic_min_distinct_variants:
        reasons.append("distinct_variant_count")
    if _as_float(evaluation.get("contradiction_rate"), 0.0) > policy.mechanic_max_contradiction_rate:
        reasons.append("contradiction_rate")
    if _as_float(evaluation.get("delayed_positive_rate"), 0.0) < policy.mechanic_min_delayed_positive_rate:
        reasons.append("delayed_positive_rate")
    if _as_float(evaluation.get("adapter_leakage_score"), 0.0) > policy.mechanic_max_adapter_leakage_score:
        reasons.append("adapter_leakage_score")
    return reasons


_MECHANIC_PROMOTION_GENERIC_KEY_PREFIXES = (
    "effect:",
    "solver_family:",
    "target_kind:",
    "operator:",
    "binding_source:",
)
_MECHANIC_PROMOTION_EXCLUDED_EFFECTS = {"", "noop", "failure", "unexplained_effect"}
_MECHANIC_PROMOTION_DELAYED_SCORE_WINDOW = 8


def _record_delayed_score_evidence(
    target: dict[str, Counter[str]],
    keys: Iterable[str],
    *,
    positive: bool,
) -> None:
    for key in keys:
        text_key = str(key)
        if not text_key:
            continue
        target[text_key]["sample_count"] += 1
        if positive:
            target[text_key]["positive_count"] += 1


def _record_structural_rod_delayed_score_evidence(
    *,
    trace: list[Any],
    index: int,
    item: Mapping[str, Any],
    structural_rods: Mapping[str, Any],
    default_score: int,
    generic_key_evidence: dict[str, Counter[str]],
    effect_type_evidence: dict[str, Counter[str]],
    solver_family_evidence: dict[str, Counter[str]],
) -> None:
    if "score_before" not in item:
        return
    recent = structural_rods.get("recent")
    if not isinstance(recent, list):
        return
    generic_keys: set[str] = set()
    effect_types: set[str] = set()
    solver_families: set[str] = set()
    for rod in recent:
        if not isinstance(rod, Mapping) or str(rod.get("verdict", "") or "") != "supported":
            continue
        for key in list(rod.get("generic_keys") or []):
            generic_keys.add(str(key))
        effect = rod.get("effect")
        if isinstance(effect, Mapping):
            effect_type = str(effect.get("effect_type", "") or "")
            if effect_type:
                effect_types.add(effect_type)
        family = str(rod.get("solver_family", "") or "")
        if family:
            solver_families.add(family)
    if not generic_keys and not effect_types and not solver_families:
        return
    delayed_delta = _window_score_delta_from_index(
        trace,
        baseline_index=index,
        window=_MECHANIC_PROMOTION_DELAYED_SCORE_WINDOW,
        default_score=default_score,
    )
    positive = delayed_delta > 0
    _record_delayed_score_evidence(generic_key_evidence, generic_keys, positive=positive)
    _record_delayed_score_evidence(effect_type_evidence, effect_types, positive=positive)
    _record_delayed_score_evidence(solver_family_evidence, solver_families, positive=positive)


def _mechanic_promotion_candidates(
    verdicts: Mapping[str, Counter[str]],
    cross_game_support: Mapping[str, set[str]],
    *,
    key_kind: str,
    variant_support: Mapping[str, set[str]] | None = None,
    episode_support: Mapping[str, set[str]] | None = None,
    delayed_score_evidence: Mapping[str, Counter[str]] | None = None,
    adapter_leakage_score: float = 0.0,
    promotion_policy: RodPromotionPolicy | None = None,
) -> list[dict[str, Any]]:
    policy = promotion_policy or default_rod_promotion_policy()
    candidates: list[dict[str, Any]] = []
    for key, counter in verdicts.items():
        text_key = str(key)
        if key_kind == "generic_key":
            if text_key.startswith("action:"):
                continue
            if not text_key.startswith(_MECHANIC_PROMOTION_GENERIC_KEY_PREFIXES):
                continue
            if text_key in {"effect:noop", "effect:failure", "effect:unexplained_effect"}:
                continue
        elif key_kind == "effect_type" and text_key in _MECHANIC_PROMOTION_EXCLUDED_EFFECTS:
            continue
        supported = _as_int(counter.get("supported"), 0)
        contradicted = _as_int(counter.get("contradicted"), 0)
        inconclusive = _as_int(counter.get("inconclusive"), 0)
        if supported <= 0:
            continue
        net_support = supported - contradicted
        total_verdicts = supported + contradicted
        contradiction_rate = _safe_div(contradicted, total_verdicts)
        games = sorted(cross_game_support.get(text_key, set()))
        variants = sorted((variant_support or {}).get(text_key, set()))
        episodes = sorted((episode_support or {}).get(text_key, set()))
        delayed_counts = (delayed_score_evidence or {}).get(text_key)
        delayed_sample_count = (
            _as_int(delayed_counts.get("sample_count"), 0)
            if isinstance(delayed_counts, Mapping)
            else 0
        )
        delayed_positive_count = (
            _as_int(delayed_counts.get("positive_count"), 0)
            if isinstance(delayed_counts, Mapping)
            else 0
        )
        delayed_positive_rate = (
            _safe_div(delayed_positive_count, delayed_sample_count)
            if delayed_sample_count > 0
            else _safe_div(max(0, net_support), total_verdicts)
        )
        delayed_positive_rate_source = (
            "trace_score_window" if delayed_sample_count > 0 else "support_verdict_proxy"
        )
        summary = {
            "support_count": supported,
            "contradiction_count": contradicted,
            "distinct_episode_count": len(episodes),
            "distinct_variant_count": len(variants),
            "delayed_positive_rate": delayed_positive_rate,
            "adapter_leakage_score": adapter_leakage_score,
        }
        evaluation = policy.evaluate_mechanic_summary(summary)
        candidates.append(
            {
                "key_kind": key_kind,
                "key": text_key,
                "support_count": supported,
                "contradiction_count": contradicted,
                "inconclusive_count": inconclusive,
                "net_support": net_support,
                "contradiction_rate": contradiction_rate,
                "cross_game_count": len(games),
                "distinct_episode_count": len(episodes),
                "distinct_variant_count": len(variants),
                "delayed_positive_rate": delayed_positive_rate,
                "delayed_positive_rate_source": delayed_positive_rate_source,
                "delayed_positive_sample_count": delayed_sample_count,
                "delayed_positive_count": delayed_positive_count,
                "adapter_leakage_score": adapter_leakage_score,
                "eligible_for_mechanic_promotion": bool(
                    evaluation.get("eligible_for_mechanic_promotion")
                ),
                "demote": bool(evaluation.get("demote")),
                "retire": bool(evaluation.get("retire")),
                "promotion_block_reasons": _mechanic_promotion_block_reasons(
                    evaluation,
                    policy,
                ),
                "games": games[:8],
                "variants": variants[:8],
                "episodes": episodes[:8],
            }
        )
    candidates.sort(
        key=lambda item: (
            -_as_int(item.get("cross_game_count"), 0),
            -_as_int(item.get("net_support"), 0),
            -_as_int(item.get("support_count"), 0),
            str(item.get("key_kind", "")),
            str(item.get("key", "")),
        )
    )
    return candidates


def _structural_rod_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    trace_step_count = 0
    trace_step_with_rods_count = 0
    trace_step_with_reuse_count = 0
    trace_step_with_runtime_mechanic_count = 0
    runtime_mechanic_template_credit_step_count = 0
    runtime_mechanic_template_delayed_score_delta_sum = 0
    runtime_mechanic_template_delayed_positive_score_step_count = 0
    runtime_mechanic_template_shifted_delayed_score_delta_sum = 0
    runtime_mechanic_template_random_step_score_delta_sum = 0
    runtime_mechanic_template_wrong_target_site_credit_sum = 0
    runtime_mechanic_template_wrong_target_site_before_score_count = 0
    runtime_mechanic_template_stale_episode_wrong_key_credit_sum = 0
    runtime_mechanic_template_stale_episode_wrong_key_event_count = 0
    rows_with_rods = 0
    rows_with_runtime_mechanic_templates = 0
    final_rod_count_sum = 0
    final_runtime_mechanic_template_count_sum = 0
    support_count = 0
    contradiction_count = 0
    unexplained_effect_count = 0
    auto_binding_count = 0
    adapter_leakage_count = 0
    rod_supported_score_sum = 0
    rod_reused_score_sum = 0
    runtime_mechanic_template_score_sum = 0
    max_rod_count = 0
    max_reuse_count = 0
    max_runtime_mechanic_template_count = 0
    by_effect_type: Counter[str] = Counter()
    by_primitive: dict[str, Counter[str]] = {}
    by_solver_family: dict[str, Counter[str]] = {}
    by_runtime_mechanic_key: Counter[str] = Counter()
    by_runtime_mechanic_key_kind: Counter[str] = Counter()
    runtime_mechanic_credit_by_key: dict[str, Counter[str]] = defaultdict(Counter)
    runtime_mechanic_samples: list[dict[str, Any]] = []
    previous_runtime_key_episodes: dict[str, set[str]] = {}
    by_generic_key_verdict: dict[str, Counter[str]] = {}
    by_effect_type_verdict: dict[str, Counter[str]] = {}
    cross_game_generic_keys: dict[str, set[str]] = {}
    cross_game_effect_types: dict[str, set[str]] = {}
    cross_game_solver_families: dict[str, set[str]] = {}
    variant_generic_keys: dict[str, set[str]] = {}
    variant_effect_types: dict[str, set[str]] = {}
    variant_solver_families: dict[str, set[str]] = {}
    episode_generic_keys: dict[str, set[str]] = {}
    episode_effect_types: dict[str, set[str]] = {}
    episode_solver_families: dict[str, set[str]] = {}
    delayed_score_generic_keys: dict[str, Counter[str]] = defaultdict(Counter)
    delayed_score_effect_types: dict[str, Counter[str]] = defaultdict(Counter)
    delayed_score_solver_families: dict[str, Counter[str]] = defaultdict(Counter)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    for row_index, row in enumerate(row_list):
        game_id = str(row.get("game_id", "") or "unknown")
        variant = str(row.get("variant", "") or "unknown")
        episode_id = str(row.get("episode_id", "") or f"{game_id}:{variant}:{row_index}")
        trace = row.get("trace")
        if isinstance(trace, list):
            default_score = _as_int(row.get("final_score"), 0)
            for trace_index, item in enumerate(trace):
                if not isinstance(item, Mapping):
                    continue
                trace_step_count += 1
                rods = _structural_rods_from_trace_item(item)
                if not rods:
                    continue
                rod_count = _as_int(rods.get("rod_count"), 0)
                reuse_count = _as_int(rods.get("reuse_count"), 0)
                max_rod_count = max(max_rod_count, rod_count)
                max_reuse_count = max(max_reuse_count, reuse_count)
                runtime_count = _runtime_mechanic_template_count(rods)
                max_runtime_mechanic_template_count = max(
                    max_runtime_mechanic_template_count,
                    runtime_count,
                )
                if rod_count > 0:
                    trace_step_with_rods_count += 1
                if reuse_count > 0:
                    trace_step_with_reuse_count += 1
                if runtime_count > 0:
                    trace_step_with_runtime_mechanic_count += 1
                    if "score_before" in item:
                        runtime_candidates = _runtime_mechanic_candidates_from_rods(rods)
                        runtime_candidate_keys = {
                            f"{str(candidate.get('key_kind', '') or '')}:{str(candidate.get('key', '') or '')}"
                            for candidate in runtime_candidates
                            if str(candidate.get("key_kind", "") or "")
                            and str(candidate.get("key", "") or "")
                        }
                        runtime_mechanic_template_credit_step_count += 1
                        current_score = _as_int(item.get("score_before"), default_score)
                        delayed_delta, delayed_delay_steps = _window_score_delta_and_delay_from_index(
                            trace,
                            baseline_index=trace_index,
                            window=_MECHANIC_PROMOTION_DELAYED_SCORE_WINDOW,
                            default_score=current_score,
                        )
                        shifted_delta = _window_score_delta_from_index(
                            trace,
                            baseline_index=trace_index + 1,
                            window=_MECHANIC_PROMOTION_DELAYED_SCORE_WINDOW,
                            default_score=current_score,
                        )
                        random_delta = _window_score_delta_from_index(
                            trace,
                            baseline_index=_deterministic_control_index(trace_index, len(trace)),
                            window=_MECHANIC_PROMOTION_DELAYED_SCORE_WINDOW,
                            default_score=current_score,
                        )
                        runtime_mechanic_template_delayed_score_delta_sum += delayed_delta
                        runtime_mechanic_template_shifted_delayed_score_delta_sum += shifted_delta
                        runtime_mechanic_template_random_step_score_delta_sum += random_delta
                        if delayed_delta > 0:
                            runtime_mechanic_template_delayed_positive_score_step_count += 1
                        stale_wrong_keys = [
                            key
                            for key, episodes in previous_runtime_key_episodes.items()
                            if key not in runtime_candidate_keys and episode_id not in episodes
                        ]
                        if delayed_delta > 0 and stale_wrong_keys:
                            runtime_mechanic_template_stale_episode_wrong_key_credit_sum += (
                                delayed_delta
                            )
                            runtime_mechanic_template_stale_episode_wrong_key_event_count += 1
                        wrong_target_site_credit = 0
                        wrong_target_site_before_score = False
                        target_site = _runtime_mechanic_target_site_signature(item)
                        if target_site and delayed_delta > 0 and delayed_delay_steps > 0:
                            wrong_target_site_before_score, _wrong_site_delay = (
                                _wrong_target_site_before_score(
                                    trace,
                                    index=trace_index,
                                    target_site=target_site,
                                    delayed_delay_steps=delayed_delay_steps,
                                )
                            )
                            if wrong_target_site_before_score:
                                wrong_target_site_credit = delayed_delta
                                runtime_mechanic_template_wrong_target_site_credit_sum += (
                                    wrong_target_site_credit
                                )
                                runtime_mechanic_template_wrong_target_site_before_score_count += 1
                        for candidate in runtime_candidates:
                            key_kind = str(candidate.get("key_kind", "") or "")
                            key = str(candidate.get("key", "") or "")
                            if not key_kind or not key:
                                continue
                            credit_key = f"{key_kind}:{key}"
                            bucket = runtime_mechanic_credit_by_key[credit_key]
                            bucket["credit_step_count"] += 1
                            bucket["delayed_score_delta_sum"] += delayed_delta
                            bucket["shifted_delayed_score_delta_sum"] += shifted_delta
                            bucket["random_step_score_delta_sum"] += random_delta
                            bucket["wrong_target_site_credit_sum"] += wrong_target_site_credit
                            if wrong_target_site_before_score:
                                bucket["wrong_target_site_before_score_count"] += 1
                            if delayed_delta > 0:
                                bucket["delayed_positive_score_step_count"] += 1
                _record_structural_rod_delayed_score_evidence(
                    trace=trace,
                    index=trace_index,
                    item=item,
                    structural_rods=rods,
                    default_score=default_score,
                    generic_key_evidence=delayed_score_generic_keys,
                    effect_type_evidence=delayed_score_effect_types,
                    solver_family_evidence=delayed_score_solver_families,
                )
        final_rods = _last_structural_rods(row)
        if not final_rods:
            continue
        final_count = _as_int(final_rods.get("rod_count"), 0)
        if final_count <= 0:
            continue
        rows_with_rods += 1
        final_rod_count_sum += final_count
        runtime_template_count = _runtime_mechanic_template_count(final_rods)
        final_runtime_mechanic_template_count_sum += runtime_template_count
        if runtime_template_count > 0:
            rows_with_runtime_mechanic_templates += 1
            by_game[game_id]["rows_with_runtime_mechanic_templates"] += 1
            by_game[game_id]["runtime_mechanic_template_count_sum"] += runtime_template_count
        support_count += _as_int(final_rods.get("support_count"), 0)
        contradiction_count += _as_int(final_rods.get("contradiction_count"), 0)
        unexplained_effect_count += _as_int(final_rods.get("unexplained_effect_count"), 0)
        auto_binding_count += _as_int(final_rods.get("auto_binding_count"), 0)
        if _as_int(final_rods.get("support_count"), 0) > 0:
            rod_supported_score_sum += _as_int(row.get("final_score"), 0)
        if _as_int(final_rods.get("reuse_count"), 0) > 0:
            rod_reused_score_sum += _as_int(row.get("final_score"), 0)
        if runtime_template_count > 0:
            runtime_mechanic_template_score_sum += _as_int(row.get("final_score"), 0)
            by_game[game_id]["runtime_mechanic_template_score_sum"] += _as_int(row.get("final_score"), 0)
        adapter_leakage_count += _rod_payload_leak_count(final_rods)
        by_game[game_id]["rows_with_rods"] += 1
        by_game[game_id]["final_rod_count_sum"] += final_count
        by_game[game_id]["support_count"] += _as_int(final_rods.get("support_count"), 0)
        by_game[game_id]["contradiction_count"] += _as_int(final_rods.get("contradiction_count"), 0)
        by_game[game_id]["auto_binding_count"] += _as_int(final_rods.get("auto_binding_count"), 0)
        effect_counts = final_rods.get("by_effect_type")
        if isinstance(effect_counts, Mapping):
            for effect_type, count in effect_counts.items():
                by_effect_type[str(effect_type)] += _as_int(count, 0)
        _merge_nested_counter(by_primitive, final_rods.get("by_primitive"))
        _merge_nested_counter(by_solver_family, final_rods.get("by_solver_family"))
        _merge_nested_counter(by_generic_key_verdict, final_rods.get("by_generic_key_verdict"))
        _merge_nested_counter(by_effect_type_verdict, final_rods.get("by_effect_type_verdict"))
        for candidate in _runtime_mechanic_candidates_from_rods(final_rods):
            key_kind = str(candidate.get("key_kind", "") or "")
            key = str(candidate.get("key", "") or "")
            runtime_key = f"{key_kind}:{key}"
            by_runtime_mechanic_key[f"{key_kind}:{key}"] += 1
            by_runtime_mechanic_key_kind[key_kind] += 1
            if key_kind and key:
                previous_runtime_key_episodes.setdefault(runtime_key, set()).add(episode_id)
            if len(runtime_mechanic_samples) < 8:
                runtime_mechanic_samples.append(
                    {
                        "game_id": game_id,
                        "variant": variant,
                        "key_kind": key_kind,
                        "key": key,
                        "support_count": _as_int(candidate.get("support_count"), 0),
                        "net_support": _as_int(candidate.get("net_support"), 0),
                    }
                )
        _record_cross_game_verdict_support(
            cross_game_generic_keys,
            final_rods.get("by_generic_key_verdict"),
            game_id=game_id,
        )
        _record_verdict_label_support(
            variant_generic_keys,
            final_rods.get("by_generic_key_verdict"),
            label=variant,
        )
        _record_verdict_label_support(
            episode_generic_keys,
            final_rods.get("by_generic_key_verdict"),
            label=episode_id,
        )
        _record_cross_game_verdict_support(
            cross_game_effect_types,
            final_rods.get("by_effect_type_verdict"),
            game_id=game_id,
        )
        _record_verdict_label_support(
            variant_effect_types,
            final_rods.get("by_effect_type_verdict"),
            label=variant,
        )
        _record_verdict_label_support(
            episode_effect_types,
            final_rods.get("by_effect_type_verdict"),
            label=episode_id,
        )
        _record_cross_game_verdict_support(
            cross_game_solver_families,
            final_rods.get("by_solver_family"),
            game_id=game_id,
        )
        _record_verdict_label_support(
            variant_solver_families,
            final_rods.get("by_solver_family"),
            label=variant,
        )
        _record_verdict_label_support(
            episode_solver_families,
            final_rods.get("by_solver_family"),
            label=episode_id,
        )
        recent = final_rods.get("recent")
        if isinstance(recent, list):
            for rod in recent:
                if not isinstance(rod, Mapping) or str(rod.get("verdict", "") or "") != "supported":
                    continue
                effect = rod.get("effect")
                if isinstance(effect, Mapping):
                    effect_type = str(effect.get("effect_type", "") or "")
                    if effect_type:
                        cross_game_effect_types.setdefault(effect_type, set()).add(game_id)
                        variant_effect_types.setdefault(effect_type, set()).add(variant)
                        episode_effect_types.setdefault(effect_type, set()).add(episode_id)
                family = str(rod.get("solver_family", "") or "")
                if family:
                    cross_game_solver_families.setdefault(family, set()).add(game_id)
                    variant_solver_families.setdefault(family, set()).add(variant)
                    episode_solver_families.setdefault(family, set()).add(episode_id)
                for key in list(rod.get("generic_keys") or []):
                    cross_game_generic_keys.setdefault(str(key), set()).add(game_id)
                    variant_generic_keys.setdefault(str(key), set()).add(variant)
                    episode_generic_keys.setdefault(str(key), set()).add(episode_id)
    adapter_leakage_score = _safe_div(adapter_leakage_count, final_rod_count_sum)
    promotion_policy = default_rod_promotion_policy()
    raw_mechanic_candidates = (
        _mechanic_promotion_candidates(
            by_generic_key_verdict,
            cross_game_generic_keys,
            key_kind="generic_key",
            variant_support=variant_generic_keys,
            episode_support=episode_generic_keys,
            delayed_score_evidence=delayed_score_generic_keys,
            adapter_leakage_score=adapter_leakage_score,
            promotion_policy=promotion_policy,
        )
        + _mechanic_promotion_candidates(
            by_effect_type_verdict,
            cross_game_effect_types,
            key_kind="effect_type",
            variant_support=variant_effect_types,
            episode_support=episode_effect_types,
            delayed_score_evidence=delayed_score_effect_types,
            adapter_leakage_score=adapter_leakage_score,
            promotion_policy=promotion_policy,
        )
        + _mechanic_promotion_candidates(
            by_solver_family,
            cross_game_solver_families,
            key_kind="solver_family",
            variant_support=variant_solver_families,
            episode_support=episode_solver_families,
            delayed_score_evidence=delayed_score_solver_families,
            adapter_leakage_score=adapter_leakage_score,
            promotion_policy=promotion_policy,
        )
    )
    raw_mechanic_candidates.sort(
        key=lambda item: (
            -_as_int(item.get("cross_game_count"), 0),
            -_as_int(item.get("net_support"), 0),
            -_as_int(item.get("support_count"), 0),
            str(item.get("key_kind", "")),
            str(item.get("key", "")),
        )
    )
    promotion_leakage_guard_passed = adapter_leakage_count == 0
    policy_eligible_candidates = [
        candidate
        for candidate in raw_mechanic_candidates
        if bool(candidate.get("eligible_for_mechanic_promotion"))
    ]
    mechanic_candidates = policy_eligible_candidates if promotion_leakage_guard_passed else []
    blocked_mechanic_candidates = [
        candidate
        for candidate in raw_mechanic_candidates
        if not bool(candidate.get("eligible_for_mechanic_promotion"))
    ]
    if not promotion_leakage_guard_passed:
        blocked_mechanic_candidates = raw_mechanic_candidates
    runtime_mechanic_template_credit_by_key = _runtime_mechanic_credit_by_key_summary(
        runtime_mechanic_credit_by_key
    )
    runtime_mechanic_template_wrong_key_max_score_delta_sum = max(
        [
            _as_int(item.get("wrong_key_delayed_score_delta_sum"), 0)
            for item in runtime_mechanic_template_credit_by_key.values()
            if isinstance(item, Mapping)
        ],
        default=0,
    )
    runtime_mechanic_template_negative_control_max_score_delta_sum = max(
        runtime_mechanic_template_shifted_delayed_score_delta_sum,
        runtime_mechanic_template_random_step_score_delta_sum,
        runtime_mechanic_template_wrong_target_site_credit_sum,
        runtime_mechanic_template_stale_episode_wrong_key_credit_sum,
    )
    runtime_mechanic_template_causal_credit_margin = (
        runtime_mechanic_template_delayed_score_delta_sum
        - runtime_mechanic_template_negative_control_max_score_delta_sum
    )
    return {
        "schema_version": "eca_structural_rod_summary.v1",
        "row_count": len(row_list),
        "rows_with_rods": rows_with_rods,
        "row_rod_coverage_rate": _safe_div(rows_with_rods, len(row_list)),
        "trace_step_count": trace_step_count,
        "trace_step_with_rods_count": trace_step_with_rods_count,
        "trace_step_rod_coverage_rate": _safe_div(trace_step_with_rods_count, trace_step_count),
        "trace_step_with_reuse_count": trace_step_with_reuse_count,
        "rod_reuse_rate": _safe_div(trace_step_with_reuse_count, trace_step_count),
        "trace_step_with_runtime_mechanic_count": trace_step_with_runtime_mechanic_count,
        "trace_step_runtime_mechanic_rate": _safe_div(
            trace_step_with_runtime_mechanic_count,
            trace_step_count,
        ),
        "final_rod_count_sum": final_rod_count_sum,
        "mean_final_rods_per_row": _safe_div(final_rod_count_sum, len(row_list)),
        "max_rod_count": max_rod_count,
        "max_reuse_count": max_reuse_count,
        "max_runtime_mechanic_template_count": max_runtime_mechanic_template_count,
        "support_count": support_count,
        "contradiction_count": contradiction_count,
        "auto_binding_count": auto_binding_count,
        "auto_binding_rate": _safe_div(auto_binding_count, final_rod_count_sum),
        "rod_supported_score_sum": rod_supported_score_sum,
        "rod_reused_score_sum": rod_reused_score_sum,
        "runtime_mechanic_template_score_sum": runtime_mechanic_template_score_sum,
        "runtime_mechanic_template_credit_step_count": (
            runtime_mechanic_template_credit_step_count
        ),
        "runtime_mechanic_template_delayed_score_delta_sum": (
            runtime_mechanic_template_delayed_score_delta_sum
        ),
        "runtime_mechanic_template_delayed_positive_score_step_count": (
            runtime_mechanic_template_delayed_positive_score_step_count
        ),
        "runtime_mechanic_template_delayed_positive_score_rate": _safe_div(
            runtime_mechanic_template_delayed_positive_score_step_count,
            runtime_mechanic_template_credit_step_count,
        ),
        "runtime_mechanic_template_shifted_delayed_score_delta_sum": (
            runtime_mechanic_template_shifted_delayed_score_delta_sum
        ),
        "runtime_mechanic_template_random_step_score_delta_sum": (
            runtime_mechanic_template_random_step_score_delta_sum
        ),
        "runtime_mechanic_template_wrong_target_site_credit_sum": (
            runtime_mechanic_template_wrong_target_site_credit_sum
        ),
        "runtime_mechanic_template_wrong_target_site_before_score_count": (
            runtime_mechanic_template_wrong_target_site_before_score_count
        ),
        "runtime_mechanic_template_real_vs_wrong_target_site_delta": (
            runtime_mechanic_template_delayed_score_delta_sum
            - runtime_mechanic_template_wrong_target_site_credit_sum
        ),
        "runtime_mechanic_template_stale_episode_wrong_key_credit_sum": (
            runtime_mechanic_template_stale_episode_wrong_key_credit_sum
        ),
        "runtime_mechanic_template_stale_episode_wrong_key_event_count": (
            runtime_mechanic_template_stale_episode_wrong_key_event_count
        ),
        "runtime_mechanic_template_real_vs_stale_episode_wrong_key_delta": (
            runtime_mechanic_template_delayed_score_delta_sum
            - runtime_mechanic_template_stale_episode_wrong_key_credit_sum
        ),
        "runtime_mechanic_template_wrong_key_max_score_delta_sum": (
            runtime_mechanic_template_wrong_key_max_score_delta_sum
        ),
        "runtime_mechanic_template_negative_control_max_score_delta_sum": (
            runtime_mechanic_template_negative_control_max_score_delta_sum
        ),
        "runtime_mechanic_template_real_vs_wrong_key_delta": (
            runtime_mechanic_template_delayed_score_delta_sum
            - runtime_mechanic_template_wrong_key_max_score_delta_sum
        ),
        "runtime_mechanic_template_causal_credit_margin": (
            runtime_mechanic_template_causal_credit_margin
        ),
        "runtime_mechanic_template_causal_credit_margin_rate": max(
            0.0,
            _safe_div(
                runtime_mechanic_template_causal_credit_margin,
                runtime_mechanic_template_delayed_score_delta_sum,
            ),
        ),
        "runtime_mechanic_template_credit_by_key": runtime_mechanic_template_credit_by_key,
        "rows_with_runtime_mechanic_templates": rows_with_runtime_mechanic_templates,
        "runtime_mechanic_template_row_rate": _safe_div(
            rows_with_runtime_mechanic_templates,
            len(row_list),
        ),
        "runtime_mechanic_template_count_sum": final_runtime_mechanic_template_count_sum,
        "mean_runtime_mechanic_templates_per_row": _safe_div(
            final_runtime_mechanic_template_count_sum,
            len(row_list),
        ),
        "by_runtime_mechanic_key": dict(sorted(by_runtime_mechanic_key.items())),
        "by_runtime_mechanic_key_kind": dict(sorted(by_runtime_mechanic_key_kind.items())),
        "runtime_mechanic_template_samples": runtime_mechanic_samples,
        "unexplained_effect_count": unexplained_effect_count,
        "contradiction_rate": _safe_div(contradiction_count, support_count + contradiction_count),
        "unexplained_effect_rate": _safe_div(unexplained_effect_count, final_rod_count_sum),
        "adapter_leakage_count": adapter_leakage_count,
        "adapter_leakage_score": adapter_leakage_score,
        "mechanic_promotion_policy": promotion_policy.to_dict(),
        "mechanic_promotion_evaluated_candidate_count": len(raw_mechanic_candidates),
        "mechanic_promotion_policy_blocked_count": len(
            [
                candidate
                for candidate in raw_mechanic_candidates
                if not bool(candidate.get("eligible_for_mechanic_promotion"))
            ]
        ),
        "mechanic_promotion_leakage_guard_passed": promotion_leakage_guard_passed,
        "mechanic_promotion_candidate_count": len(policy_eligible_candidates),
        "mechanic_promotion_count": len(mechanic_candidates),
        "cross_game_mechanic_promotion_count": len(
            [
                candidate
                for candidate in mechanic_candidates
                if _as_int(candidate.get("cross_game_count"), 0) >= 2
            ]
        ),
        "mechanic_promotion_candidates": mechanic_candidates[:20],
        "mechanic_promotion_blocked_candidates": blocked_mechanic_candidates[:20],
        "cross_game_rod_support": {
            "generic_keys": _cross_game_support_summary(cross_game_generic_keys),
            "effect_types": _cross_game_support_summary(cross_game_effect_types),
            "solver_families": _cross_game_support_summary(cross_game_solver_families),
        },
        "by_effect_type": dict(sorted(by_effect_type.items())),
        "by_primitive": _nested_counter_to_dict(by_primitive),
        "by_solver_family": _nested_counter_to_dict(by_solver_family),
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
    }


def _a_axis_sensor_gap_inventory_sensor(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    current_scores: dict[str, int] = {}
    for row in rows:
        game_id = str(row.get("game_id", "") or "").strip().lower()
        if not game_id:
            continue
        current_scores[game_id] = max(
            current_scores.get(game_id, 0),
            _as_int(row.get("final_score"), 0),
        )
    return build_phase37_to_current_sensor_gap_inventory(
        current_scores,
        current_axis="current_run_adapter_free_rows",
    )


def _a_axis_sensor_promotion_gap_inventory_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    current_scores: dict[str, int] = {}
    for row in rows:
        game_id = str(row.get("game_id", "") or "").strip().lower()
        if not game_id:
            continue
        current_scores[game_id] = max(
            current_scores.get(game_id, 0),
            _as_int(row.get("final_score"), 0),
        )
    return build_phase37_to_current_sensor_promotion_gap_inventory(
        current_scores,
        current_axis="current_run_adapter_free_rows",
    )


def _parallel_sensor_substrate_gap_inventory_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    current_scores: dict[str, int] = {}
    for row in rows:
        game_id = str(row.get("game_id", "") or "").strip().lower()
        if not game_id:
            continue
        current_scores[game_id] = max(
            current_scores.get(game_id, 0),
            _as_int(row.get("final_score"), 0),
        )
    return build_phase37_to_current_parallel_sensor_substrate_gap_inventory(
        current_scores,
        current_axis="current_run_adapter_free_rows",
    )


def _transition_state_variable_diagnostic_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TRANSITION_STATE_VARIABLE_DIAGNOSTIC_SENSOR")
    packets = _transition_state_variable_packets_from_rows(rows) if enabled else []
    summary = summarize_transition_state_variable_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TRANSITION_STATE_VARIABLE_DIAGNOSTIC_SENSOR=1"
        )
    return summary


def _moving_sprite_runtime_trace_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MOVING_SPRITE_RUNTIME_TRACE_SENSOR")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_runtime_trace_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_RUNTIME_TRACE_SENSOR=1"
        )
    return summary


def _mechanic_certificate_runtime_trace_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MECHANIC_CERTIFICATE_RUNTIME_TRACE_SENSOR")
    packets = _mechanic_certificate_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_mechanic_certificate_runtime_trace_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MECHANIC_CERTIFICATE_RUNTIME_TRACE_SENSOR=1"
        )
    return summary


def _mechanic_certificate_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MECHANIC_CERTIFICATE_POSTERIOR")
    packets = _mechanic_certificate_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_mechanic_certificate_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = "set ECA_ENABLE_MECHANIC_CERTIFICATE_POSTERIOR=1"
    return summary


def _mechanic_certificate_guarded_planner_connection(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MECHANIC_CERTIFICATE_GUARDED_PLANNER_CONNECTION")
    packets = _mechanic_certificate_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_mechanic_certificate_guarded_planner_connection(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MECHANIC_CERTIFICATE_GUARDED_PLANNER_CONNECTION=1"
        )
    return summary


def _palette_invariant_runtime_trace_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PALETTE_INVARIANT_RUNTIME_TRACE_SENSOR")
    packets = _palette_invariant_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_palette_invariant_runtime_trace_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PALETTE_INVARIANT_RUNTIME_TRACE_SENSOR=1"
        )
    return summary


def _palette_invariant_perception_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PALETTE_INVARIANT_POSTERIOR")
    packets = _palette_invariant_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_palette_invariant_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = "set ECA_ENABLE_PALETTE_INVARIANT_POSTERIOR=1"
    return summary


def _palette_invariant_guarded_planner_connection(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PALETTE_INVARIANT_GUARDED_PLANNER_CONNECTION")
    packets = _palette_invariant_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_palette_invariant_guarded_planner_connection(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PALETTE_INVARIANT_GUARDED_PLANNER_CONNECTION=1"
        )
    return summary


def _actor_identity_anchor_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_ACTOR_IDENTITY_ANCHOR_SENSOR")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_actor_identity_anchor_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_ACTOR_IDENTITY_ANCHOR_SENSOR=1"
        )
    return summary


def _actor_identity_anchor_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_ACTOR_IDENTITY_ANCHOR_POSTERIOR")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_actor_identity_anchor_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_ACTOR_IDENTITY_ANCHOR_POSTERIOR=1"
        )
    return summary


def _actor_identity_anchor_guarded_planner_connection(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_ACTOR_IDENTITY_ANCHOR_GUARDED_PLANNER_CONNECTION")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_actor_identity_anchor_guarded_planner_connection(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_ACTOR_IDENTITY_ANCHOR_GUARDED_PLANNER_CONNECTION=1"
        )
    return summary


def _transport_effect_runtime_trace_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TRANSPORT_EFFECT_RUNTIME_TRACE_SENSOR")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_transport_effect_runtime_trace_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TRANSPORT_EFFECT_RUNTIME_TRACE_SENSOR=1"
        )
    return summary


def _tile_effect_runtime_trace_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TILE_EFFECT_RUNTIME_TRACE_SENSOR")
    packets = _tile_effect_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_tile_effect_runtime_trace_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_RUNTIME_TRACE_SENSOR=1"
        )
    return summary


def _tile_effect_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TILE_EFFECT_POSTERIOR")
    packets = _tile_effect_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_tile_effect_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = "set ECA_ENABLE_TILE_EFFECT_POSTERIOR=1"
    return summary


def _tile_effect_guarded_planner_connection(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TILE_EFFECT_GUARDED_PLANNER_CONNECTION")
    packets = _tile_effect_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_tile_effect_guarded_planner_connection(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_GUARDED_PLANNER_CONNECTION=1"
        )
    return summary


def _tile_effect_guard_precision_review(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TILE_EFFECT_GUARD_PRECISION_REVIEW")
    packets = _tile_effect_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_tile_effect_guard_precision_review(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_GUARD_PRECISION_REVIEW=1"
        )
    return summary


def _tile_effect_guard_counterfactual_control(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TILE_EFFECT_GUARD_COUNTERFACTUAL_CONTROL")
    packets = _tile_effect_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_tile_effect_guard_counterfactual_control(
        packets,
        allow_return_anchor_prior_occupancy_trace=_env_flag(
            "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
        ),
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_GUARD_COUNTERFACTUAL_CONTROL=1"
        )
    return summary


def _tile_effect_family_scope_negative_control(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TILE_EFFECT_FAMILY_SCOPE_NEGATIVE_CONTROL")
    packets = _tile_effect_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_tile_effect_family_scope_negative_control(
        packets,
        allow_return_anchor_prior_occupancy_trace=_env_flag(
            "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
        ),
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_FAMILY_SCOPE_NEGATIVE_CONTROL=1"
        )
    return summary


def _tile_effect_hazard_source_repair_bridge(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TILE_EFFECT_HAZARD_SOURCE_REPAIR_BRIDGE")
    packets = _tile_effect_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_tile_effect_hazard_source_repair_bridge(
        packets,
        allow_return_anchor_prior_occupancy_trace=_env_flag(
            "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
        ),
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_HAZARD_SOURCE_REPAIR_BRIDGE=1"
        )
    return summary


def _tile_effect_model_repair_proposal(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROPOSAL")
    packets = _tile_effect_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_tile_effect_model_repair_proposal(
        packets,
        allow_return_anchor_prior_occupancy_trace=_env_flag(
            "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
        ),
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROPOSAL=1"
        )
    return summary


def _tile_effect_model_repair_write_guard(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_WRITE_GUARD")
    packets = _tile_effect_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_tile_effect_model_repair_write_guard(
        packets,
        allow_return_anchor_prior_occupancy_trace=_env_flag(
            "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
        ),
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_WRITE_GUARD=1"
        )
    return summary


def _tile_effect_model_repair_write_audit(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_WRITE_AUDIT")
    packets = _tile_effect_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_tile_effect_model_repair_write_audit(
        packets,
        allow_return_anchor_prior_occupancy_trace=_env_flag(
            "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
        ),
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_WRITE_AUDIT=1"
        )
    return summary


def _tile_effect_model_repair_runtime_probe_proposal(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_RUNTIME_PROBE_PROPOSAL")
    packets = _tile_effect_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_tile_effect_model_repair_runtime_probe_proposal(
        packets,
        allow_return_anchor_prior_occupancy_trace=_env_flag(
            "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
        ),
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_RUNTIME_PROBE_PROPOSAL=1"
        )
    return summary


def _tile_effect_model_repair_protected_probe_execution_gate(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_EXECUTION_GATE")
    packets = _tile_effect_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_tile_effect_model_repair_protected_probe_execution_gate(
        packets,
        allow_return_anchor_prior_occupancy_trace=_env_flag(
            "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
        ),
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_EXECUTION_GATE=1"
        )
    return summary


def _tile_effect_model_repair_protected_probe_runtime_bridge_guard(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_BRIDGE_GUARD"
    )
    packets = _tile_effect_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_tile_effect_model_repair_protected_probe_runtime_bridge_guard(
        packets,
        allow_return_anchor_prior_occupancy_trace=_env_flag(
            "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
        ),
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_BRIDGE_GUARD=1"
        )
    return summary


def _tile_effect_model_repair_protected_probe_outcome_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_OUTCOME_POSTERIOR"
    )
    row_list = list(rows)
    packets = _tile_effect_runtime_trace_packets_from_rows(row_list) if enabled else []
    probe_outcomes = (
        _tile_effect_model_repair_runtime_probe_outcomes_from_rows(row_list)
        if enabled
        else []
    )
    summary = summarize_tile_effect_model_repair_protected_probe_outcome_posterior(
        packets,
        probe_outcomes=probe_outcomes,
        allow_return_anchor_prior_occupancy_trace=_env_flag(
            "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
        ),
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_OUTCOME_POSTERIOR=1"
        )
    return summary


def _tile_effect_model_repair_protected_probe_rank_guard_shadow(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RANK_GUARD_SHADOW"
    )
    row_list = list(rows)
    packets = _tile_effect_runtime_trace_packets_from_rows(row_list) if enabled else []
    probe_outcomes = (
        _tile_effect_model_repair_runtime_probe_outcomes_from_rows(row_list)
        if enabled
        else []
    )
    summary = summarize_tile_effect_model_repair_protected_probe_rank_guard_shadow(
        packets,
        probe_outcomes=probe_outcomes,
        allow_return_anchor_prior_occupancy_trace=_env_flag(
            "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
        ),
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RANK_GUARD_SHADOW=1"
        )
    return summary


def _tile_effect_model_repair_protected_probe_rank_guard_runtime_guard(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RANK_GUARD_RUNTIME_GUARD"
    )
    row_list = list(rows)
    packets = _tile_effect_runtime_trace_packets_from_rows(row_list) if enabled else []
    probe_outcomes = (
        _tile_effect_model_repair_runtime_probe_outcomes_from_rows(row_list)
        if enabled
        else []
    )
    summary = (
        summarize_tile_effect_model_repair_protected_probe_rank_guard_runtime_guard(
            packets,
            probe_outcomes=probe_outcomes,
            allow_return_anchor_prior_occupancy_trace=_env_flag(
                "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
            ),
        )
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RANK_GUARD_RUNTIME_GUARD=1"
        )
    return summary


def _tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_FOCUSED_RUNTIME"
    )
    events = (
        _tile_effect_model_repair_live_rank_guard_focused_runtime_events_from_rows(
            rows
        )
        if enabled
        else []
    )
    summary = (
        summarize_tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime(
            events
        )
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_FOCUSED_RUNTIME=1"
        )
    return summary


def _tile_effect_model_repair_protected_probe_live_rank_guard_action_selection_guard(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_FOCUSED_RUNTIME"
    )
    events = (
        _tile_effect_model_repair_live_rank_guard_focused_runtime_events_from_rows(
            rows
        )
        if enabled
        else []
    )
    summary = (
        summarize_tile_effect_model_repair_protected_probe_live_rank_guard_action_selection_guard(
            events
        )
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_FOCUSED_RUNTIME=1"
        )
    return summary


def _tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_admission(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_FOCUSED_RUNTIME"
    )
    events = (
        _tile_effect_model_repair_live_rank_guard_focused_runtime_events_from_rows(
            rows
        )
        if enabled
        else []
    )
    summary = (
        summarize_tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_admission(
            events
        )
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_FOCUSED_RUNTIME=1"
        )
    return summary


def _tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_FOCUSED_RUNTIME"
    )
    events = (
        _tile_effect_positive_action_selection_score_attribution_events_from_rows(
            rows
        )
        if enabled
        else []
    )
    summary = (
        summarize_tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution(
            events
        )
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_FOCUSED_RUNTIME=1"
        )
    return summary


def _tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_positive_rank_promotion_guard(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    attribution = (
        _tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution(
            rows
        )
    )
    summary = (
        summarize_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_positive_rank_promotion_guard(
            attribution
        )
    )
    summary["enabled"] = bool(attribution.get("enabled"))
    summary["default_off"] = True
    if not bool(attribution.get("enabled")):
        summary["disabled_reason"] = str(attribution.get("disabled_reason", "") or "")
    return summary


def _tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    attribution = (
        _tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution(
            rows
        )
    )
    summary = (
        summarize_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery(
            attribution
        )
    )
    summary["enabled"] = bool(attribution.get("enabled"))
    summary["default_off"] = True
    if not bool(attribution.get("enabled")):
        summary["disabled_reason"] = str(attribution.get("disabled_reason", "") or "")
    return summary


def _tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_focused_runtime_replay(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    discovery = (
        _tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery(
            rows
        )
    )
    summary = (
        summarize_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_focused_runtime_replay(
            discovery
        )
    )
    summary["enabled"] = bool(discovery.get("enabled"))
    summary["default_off"] = True
    if not bool(discovery.get("enabled")):
        summary["disabled_reason"] = str(discovery.get("disabled_reason", "") or "")
    return summary


def _tile_effect_model_repair_protected_probe_runtime_bridge_outcome(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_BRIDGE_OUTCOME"
    )
    row_list = list(rows)
    packets = _tile_effect_runtime_trace_packets_from_rows(row_list) if enabled else []
    runtime_bridge_events = (
        _tile_effect_model_repair_runtime_bridge_events_from_rows(row_list)
        if enabled
        else []
    )
    probe_outcomes = (
        _tile_effect_model_repair_runtime_probe_outcomes_from_rows(row_list)
        if enabled
        else []
    )
    summary = summarize_tile_effect_model_repair_protected_probe_runtime_bridge_outcome(
        packets,
        runtime_bridge_events=runtime_bridge_events,
        probe_outcomes=probe_outcomes,
        allow_return_anchor_prior_occupancy_trace=_env_flag(
            "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
        ),
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_BRIDGE_OUTCOME=1"
        )
    return summary


def _tile_effect_model_repair_protected_probe_runtime_executor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_EXECUTOR"
    )
    row_list = list(rows)
    packets = _tile_effect_runtime_trace_packets_from_rows(row_list) if enabled else []
    runtime_bridge_events = (
        _tile_effect_model_repair_runtime_bridge_events_from_rows(row_list)
        if enabled
        else []
    )
    probe_outcomes = (
        _tile_effect_model_repair_runtime_probe_outcomes_from_rows(row_list)
        if enabled
        else []
    )
    summary = summarize_tile_effect_model_repair_protected_probe_runtime_executor(
        packets,
        runtime_bridge_events=runtime_bridge_events,
        probe_outcomes=probe_outcomes,
        allow_return_anchor_prior_occupancy_trace=_env_flag(
            "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
        ),
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_EXECUTOR=1"
        )
    return summary


def _tile_effect_model_repair_protected_probe_runtime_outcome_execution(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_OUTCOME_EXECUTION"
    )
    row_list = list(rows)
    packets = _tile_effect_runtime_trace_packets_from_rows(row_list) if enabled else []
    runtime_bridge_events = (
        _tile_effect_model_repair_runtime_bridge_events_from_rows(row_list)
        if enabled
        else []
    )
    probe_outcomes = (
        _tile_effect_model_repair_runtime_probe_outcomes_from_rows(row_list)
        if enabled
        else []
    )
    summary = summarize_tile_effect_model_repair_protected_probe_runtime_executor(
        packets,
        runtime_bridge_events=runtime_bridge_events,
        probe_outcomes=probe_outcomes,
        allow_return_anchor_prior_occupancy_trace=_env_flag(
            "ECA_ENABLE_TILE_EFFECT_RETURN_ANCHOR_PRIOR_TRACE_CONNECTION"
        ),
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    summary["runtime_outcome_execution_trace_event_count"] = int(len(probe_outcomes))
    summary["runtime_outcome_execution_event_count"] = int(
        summary.get("runtime_executor_attempted_count", 0) or 0
    )
    summary["runtime_outcome_execution_observed"] = bool(
        summary.get("runtime_executor_attempted_count", 0) or 0
    )
    summary["contract_key"] = (
        "tile_effect_model_repair_protected_probe_runtime_outcome_execution"
    )
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_OUTCOME_EXECUTION=1"
        )
    return summary


def _transport_effect_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TRANSPORT_EFFECT_POSTERIOR")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_transport_effect_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TRANSPORT_EFFECT_POSTERIOR=1"
        )
    return summary


def _transport_effect_guarded_planner_connection(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TRANSPORT_EFFECT_GUARDED_PLANNER_CONNECTION")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_transport_effect_guarded_planner_connection(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TRANSPORT_EFFECT_GUARDED_PLANNER_CONNECTION=1"
        )
    return summary


def _invisible_effect_position_memory_diagnostic_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_INVISIBLE_EFFECT_POSITION_MEMORY_DIAGNOSTIC_SENSOR")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_invisible_effect_position_memory_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_INVISIBLE_EFFECT_POSITION_MEMORY_DIAGNOSTIC_SENSOR=1"
        )
    return summary


def _invisible_effect_position_memory_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_INVISIBLE_EFFECT_POSITION_MEMORY_POSTERIOR")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_invisible_effect_position_memory_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_INVISIBLE_EFFECT_POSITION_MEMORY_POSTERIOR=1"
        )
    return summary


def _invisible_effect_position_memory_guarded_planner_connection(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_INVISIBLE_EFFECT_POSITION_MEMORY_GUARDED_PLANNER_CONNECTION"
    )
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_invisible_effect_position_memory_guarded_planner_connection(
        packets
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_INVISIBLE_EFFECT_POSITION_MEMORY_GUARDED_PLANNER_CONNECTION=1"
        )
    return summary


def _moving_sprite_motion_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MOVING_SPRITE_MOTION_POSTERIOR")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_motion_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_MOTION_POSTERIOR=1"
        )
    return summary


def _moving_sprite_guarded_planner_readiness(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MOVING_SPRITE_GUARDED_PLANNER_READINESS")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_guarded_planner_readiness(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_GUARDED_PLANNER_READINESS=1"
        )
    return summary


def _moving_sprite_actor_ambiguity_splitter(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MOVING_SPRITE_ACTOR_AMBIGUITY_SPLITTER")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_actor_ambiguity_splitter(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_ACTOR_AMBIGUITY_SPLITTER=1"
        )
    return summary


def _moving_sprite_guarded_actor_anchor_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MOVING_SPRITE_GUARDED_ACTOR_ANCHOR_POSTERIOR")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_guarded_actor_anchor_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_GUARDED_ACTOR_ANCHOR_POSTERIOR=1"
        )
    return summary


def _moving_sprite_soft_anchor_motion_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MOVING_SPRITE_SOFT_ANCHOR_MOTION_POSTERIOR")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_soft_anchor_motion_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_SOFT_ANCHOR_MOTION_POSTERIOR=1"
        )
    return summary


def _moving_sprite_coupled_motion_symmetry_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MOVING_SPRITE_COUPLED_MOTION_SYMMETRY_SENSOR")
    packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_coupled_motion_symmetry_sensor(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_COUPLED_MOTION_SYMMETRY_SENSOR=1"
        )
    return summary


def _moving_sprite_coupled_pair_goal_effect_bridge(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MOVING_SPRITE_COUPLED_PAIR_GOAL_EFFECT_BRIDGE")
    moving_packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    spell_packets = _program_spell_pattern_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_coupled_pair_goal_effect_bridge(
        moving_packets,
        spell_packets,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_COUPLED_PAIR_GOAL_EFFECT_BRIDGE=1"
        )
    return summary


def _moving_sprite_coupled_pair_goal_score_attribution_guard(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_MOVING_SPRITE_COUPLED_PAIR_GOAL_SCORE_ATTRIBUTION_GUARD"
    )
    moving_packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    spell_packets = _program_spell_pattern_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_coupled_pair_goal_score_attribution_guard(
        moving_packets,
        spell_packets,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_COUPLED_PAIR_GOAL_SCORE_ATTRIBUTION_GUARD=1"
        )
    return summary


def _moving_sprite_target_goal_counterfactual_attribution(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_MOVING_SPRITE_TARGET_GOAL_COUNTERFACTUAL_ATTRIBUTION"
    )
    moving_packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    spell_packets = _program_spell_pattern_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_target_goal_counterfactual_attribution(
        moving_packets,
        spell_packets,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_TARGET_GOAL_COUNTERFACTUAL_ATTRIBUTION=1"
        )
    return summary


def _moving_sprite_target_bound_effect_component_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_MOVING_SPRITE_TARGET_BOUND_EFFECT_COMPONENT_SENSOR"
    )
    moving_packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    spell_packets = _program_spell_pattern_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_target_bound_effect_component_sensor(
        moving_packets,
        spell_packets,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_TARGET_BOUND_EFFECT_COMPONENT_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_diagnostic_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_SLOT_POOL_COVERAGE_DIAGNOSTIC_SENSOR")
    packets = _slot_pool_coverage_packets_from_rows(rows) if enabled else []
    summary = summarize_slot_pool_coverage_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_DIAGNOSTIC_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_SLOT_POOL_COVERAGE_POSTERIOR")
    packets = _slot_pool_coverage_posterior_packets_from_rows(rows) if enabled else []
    summary = summarize_slot_pool_coverage_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = "set ECA_ENABLE_SLOT_POOL_COVERAGE_POSTERIOR=1"
    return summary


def _slot_pool_coverage_candidate_miner(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_SLOT_POOL_COVERAGE_CANDIDATE_MINER")
    packets = _slot_pool_coverage_packets_from_rows(rows) if enabled else []
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(rows) if enabled else []
    )
    summary = summarize_slot_pool_coverage_candidates(
        packets,
        posterior_packets=posterior_packets,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_CANDIDATE_MINER=1"
        )
    return summary


def _slot_pool_coverage_guarded_probe_outcome_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_SLOT_POOL_COVERAGE_GUARDED_PROBE_OUTCOME_SENSOR")
    packets = _slot_pool_coverage_packets_from_rows(rows) if enabled else []
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(rows) if enabled else []
    )
    summary = summarize_slot_pool_coverage_guarded_probe_outcomes(
        packets,
        posterior_packets=posterior_packets,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_GUARDED_PROBE_OUTCOME_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_counterfactual_control_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_SLOT_POOL_COVERAGE_COUNTERFACTUAL_CONTROL_SENSOR")
    packets = _slot_pool_coverage_packets_from_rows(rows) if enabled else []
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(rows) if enabled else []
    )
    summary = summarize_slot_pool_coverage_counterfactual_controls(
        packets,
        posterior_packets=posterior_packets,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_COUNTERFACTUAL_CONTROL_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_scoreless_progress_control_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_SCORELESS_PROGRESS_CONTROL_SENSOR"
    )
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(rows) if enabled else []
    )
    summary = summarize_slot_pool_coverage_scoreless_progress_controls(
        posterior_packets
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_SCORELESS_PROGRESS_CONTROL_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_scoped_scoreless_control_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_SCOPED_SCORELESS_CONTROL_SENSOR"
    )
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(rows) if enabled else []
    )
    summary = summarize_slot_pool_coverage_scoped_scoreless_controls(
        posterior_packets
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_SCOPED_SCORELESS_CONTROL_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_target_slot_scoreless_control_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_TARGET_SLOT_SCORELESS_CONTROL_SENSOR"
    )
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(rows) if enabled else []
    )
    summary = summarize_slot_pool_coverage_target_slot_scoreless_controls(
        posterior_packets
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_TARGET_SLOT_SCORELESS_CONTROL_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_exact_slot_scoreless_control_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_EXACT_SLOT_SCORELESS_CONTROL_SENSOR"
    )
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(rows) if enabled else []
    )
    summary = summarize_slot_pool_coverage_exact_slot_scoreless_controls(
        posterior_packets
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_EXACT_SLOT_SCORELESS_CONTROL_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_region_identity_control_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_SLOT_POOL_COVERAGE_REGION_IDENTITY_CONTROL_SENSOR")
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(rows) if enabled else []
    )
    summary = summarize_slot_pool_coverage_region_identity_controls(
        posterior_packets
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_REGION_IDENTITY_CONTROL_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_effect_intent_control_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_SLOT_POOL_COVERAGE_EFFECT_INTENT_CONTROL_SENSOR")
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(rows) if enabled else []
    )
    summary = summarize_slot_pool_coverage_effect_intent_controls(posterior_packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_EFFECT_INTENT_CONTROL_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_terminal_effect_score_closure_attribution_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_TERMINAL_EFFECT_SCORE_CLOSURE_ATTRIBUTION_SENSOR"
    )
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(rows) if enabled else []
    )
    summary = summarize_slot_pool_coverage_terminal_effect_score_closure_attribution(
        posterior_packets
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_SLOT_POOL_COVERAGE_TERMINAL_EFFECT_SCORE_CLOSURE_ATTRIBUTION_SENSOR=1"
        )
    return summary


def _slot_pool_trace_score_events_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        final_score = _as_int(row.get("final_score"), 0)
        trace = row.get("trace")
        if not isinstance(trace, list):
            if final_score > 0:
                events.append(
                    {
                        "game_id": game_id,
                        "step": 0,
                        "trace_index": 0,
                        "score_delta": int(final_score),
                        "score_before": 0,
                        "score_after": int(final_score),
                        "source": "final_score_without_trace",
                    }
                )
            continue
        previous_score: int | None = None
        previous_item: Mapping[str, Any] | None = None
        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            default_score = previous_score if previous_score is not None else 0
            current_score = _trace_score_before(item, default_score)
            if previous_score is not None and current_score > previous_score:
                events.append(
                    {
                        "game_id": game_id,
                        "step": _as_int(item.get("step"), index),
                        "trace_index": int(index),
                        "score_delta": int(current_score - previous_score),
                        "score_before": int(previous_score),
                        "score_after": int(current_score),
                        "source": "next_step_score_before",
                        "previous_action": str(
                            (previous_item or {}).get("action", "") or ""
                        ),
                    }
                )
            previous_score = current_score
            previous_item = item
        if trace:
            last_item = trace[-1]
            if isinstance(last_item, Mapping):
                last_score = _trace_score_before(last_item, previous_score or 0)
                if final_score > last_score:
                    last_step = _as_int(last_item.get("step"), max(0, len(trace) - 1))
                    events.append(
                        {
                            "game_id": game_id,
                            "step": int(last_step + 1),
                            "trace_index": int(len(trace)),
                            "score_delta": int(final_score - last_score),
                            "score_before": int(last_score),
                            "score_after": int(final_score),
                            "source": "final_score_after_last_trace_item",
                            "previous_action": str(last_item.get("action", "") or ""),
                        }
                    )
    return events


def _slot_pool_coverage_score_event_scope_attribution_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_SCORE_EVENT_SCOPE_ATTRIBUTION_SENSOR"
    )
    row_list = list(rows)
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(row_list) if enabled else []
    )
    score_events = _slot_pool_trace_score_events_from_rows(row_list) if enabled else []
    summary = summarize_slot_pool_coverage_score_event_scope_attribution(
        posterior_packets,
        score_events=score_events,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_SCORE_EVENT_SCOPE_ATTRIBUTION_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_intervention_contrast_attribution_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_INTERVENTION_CONTRAST_ATTRIBUTION_SENSOR"
    )
    row_list = list(rows)
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(row_list) if enabled else []
    )
    score_events = _slot_pool_trace_score_events_from_rows(row_list) if enabled else []
    summary = summarize_slot_pool_coverage_intervention_contrast_attribution(
        posterior_packets,
        score_events=score_events,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_SLOT_POOL_COVERAGE_INTERVENTION_CONTRAST_ATTRIBUTION_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_intervention_temporal_counterfactual_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_INTERVENTION_TEMPORAL_COUNTERFACTUAL_SENSOR"
    )
    row_list = list(rows)
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(row_list) if enabled else []
    )
    score_events = _slot_pool_trace_score_events_from_rows(row_list) if enabled else []
    summary = summarize_slot_pool_coverage_intervention_temporal_counterfactual(
        posterior_packets,
        score_events=score_events,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_SLOT_POOL_COVERAGE_INTERVENTION_TEMPORAL_COUNTERFACTUAL_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_already_scored_event_exclusion_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_ALREADY_SCORED_EVENT_EXCLUSION_SENSOR"
    )
    row_list = list(rows)
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(row_list) if enabled else []
    )
    score_events = _slot_pool_trace_score_events_from_rows(row_list) if enabled else []
    summary = summarize_slot_pool_coverage_already_scored_event_exclusion(
        posterior_packets,
        score_events=score_events,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_ALREADY_SCORED_EVENT_EXCLUSION_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_negative_memory_guarded_review_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_NEGATIVE_MEMORY_GUARDED_REVIEW_SENSOR"
    )
    row_list = list(rows)
    packets = _slot_pool_coverage_packets_from_rows(row_list) if enabled else []
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(row_list) if enabled else []
    )
    score_events = _slot_pool_trace_score_events_from_rows(row_list) if enabled else []
    summary = summarize_slot_pool_coverage_negative_memory_guarded_review(
        packets,
        posterior_packets=posterior_packets,
        score_events=score_events,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_NEGATIVE_MEMORY_GUARDED_REVIEW_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_negative_memory_ranking_shadow_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_NEGATIVE_MEMORY_RANKING_SHADOW_SENSOR"
    )
    row_list = list(rows)
    packets = _slot_pool_coverage_packets_from_rows(row_list) if enabled else []
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(row_list) if enabled else []
    )
    score_events = _slot_pool_trace_score_events_from_rows(row_list) if enabled else []
    summary = summarize_slot_pool_coverage_negative_memory_ranking_shadow(
        packets,
        posterior_packets=posterior_packets,
        score_events=score_events,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_NEGATIVE_MEMORY_RANKING_SHADOW_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_focused_replay_readiness_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_SLOT_POOL_COVERAGE_FOCUSED_REPLAY_READINESS_SENSOR")
    row_list = list(rows)
    packets = _slot_pool_coverage_packets_from_rows(row_list) if enabled else []
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(row_list) if enabled else []
    )
    score_events = _slot_pool_trace_score_events_from_rows(row_list) if enabled else []
    summary = summarize_slot_pool_coverage_focused_replay_readiness(
        packets,
        posterior_packets=posterior_packets,
        score_events=score_events,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_FOCUSED_REPLAY_READINESS_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_focused_replay_execution_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_SLOT_POOL_COVERAGE_FOCUSED_REPLAY_EXECUTION_SENSOR")
    row_list = list(rows)
    packets = _slot_pool_coverage_packets_from_rows(row_list) if enabled else []
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(row_list) if enabled else []
    )
    score_events = _slot_pool_trace_score_events_from_rows(row_list) if enabled else []
    execution_events = (
        _slot_pool_coverage_focused_replay_execution_events_from_rows(row_list)
        if enabled
        else []
    )
    summary = summarize_slot_pool_coverage_focused_replay_execution(
        packets,
        posterior_packets=posterior_packets,
        score_events=score_events,
        execution_events=execution_events,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_FOCUSED_REPLAY_EXECUTION_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_focused_replay_outcome_admission_guard_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_FOCUSED_REPLAY_OUTCOME_ADMISSION_GUARD_SENSOR"
    )
    row_list = list(rows)
    packets = _slot_pool_coverage_packets_from_rows(row_list) if enabled else []
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(row_list) if enabled else []
    )
    score_events = _slot_pool_trace_score_events_from_rows(row_list) if enabled else []
    execution_events = (
        _slot_pool_coverage_focused_replay_execution_events_from_rows(row_list)
        if enabled
        else []
    )
    summary = summarize_slot_pool_coverage_focused_replay_outcome_admission(
        packets,
        posterior_packets=posterior_packets,
        score_events=score_events,
        execution_events=execution_events,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_SLOT_POOL_COVERAGE_FOCUSED_REPLAY_OUTCOME_ADMISSION_GUARD_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_focused_score_bearing_candidate_discovery_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_FOCUSED_SCORE_BEARING_CANDIDATE_DISCOVERY_SENSOR"
    )
    row_list = list(rows)
    packets = _slot_pool_coverage_packets_from_rows(row_list) if enabled else []
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(row_list) if enabled else []
    )
    score_events = _slot_pool_trace_score_events_from_rows(row_list) if enabled else []
    execution_events = (
        _slot_pool_coverage_focused_replay_execution_events_from_rows(row_list)
        if enabled
        else []
    )
    summary = summarize_slot_pool_coverage_focused_score_bearing_candidate_discovery(
        packets,
        posterior_packets=posterior_packets,
        score_events=score_events,
        execution_events=execution_events,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_SLOT_POOL_COVERAGE_FOCUSED_SCORE_BEARING_CANDIDATE_DISCOVERY_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_submit_precondition_score_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_SCORE_SENSOR"
    )
    row_list = list(rows)
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(row_list) if enabled else []
    )
    score_events = _slot_pool_trace_score_events_from_rows(row_list) if enabled else []
    summary = summarize_slot_pool_coverage_submit_precondition_score(
        posterior_packets,
        score_events=score_events,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_SCORE_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_submit_precondition_runtime_bridge_guard_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RUNTIME_BRIDGE_GUARD_SENSOR"
    )
    row_list = list(rows)
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(row_list) if enabled else []
    )
    score_events = _slot_pool_trace_score_events_from_rows(row_list) if enabled else []
    bridge_events = (
        _slot_pool_coverage_submit_precondition_runtime_bridge_events_from_rows(
            row_list
        )
        if enabled
        else []
    )
    summary = summarize_slot_pool_coverage_submit_precondition_runtime_bridge_guard(
        posterior_packets,
        score_events=score_events,
        bridge_events=bridge_events,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RUNTIME_BRIDGE_GUARD_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_submit_precondition_runtime_bridge_outcome_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RUNTIME_BRIDGE_OUTCOME_SENSOR"
    )
    row_list = list(rows)
    posterior_packets = (
        _slot_pool_coverage_posterior_packets_from_rows(row_list) if enabled else []
    )
    score_events = _slot_pool_trace_score_events_from_rows(row_list) if enabled else []
    bridge_events = (
        _slot_pool_coverage_submit_precondition_runtime_bridge_events_from_rows(
            row_list
        )
        if enabled
        else []
    )
    summary = summarize_slot_pool_coverage_submit_precondition_runtime_bridge_outcome(
        posterior_packets,
        score_events=score_events,
        bridge_events=bridge_events,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RUNTIME_BRIDGE_OUTCOME_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_submit_precondition_breadth_non_regression_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_BREADTH_NON_REGRESSION_SENSOR"
    )
    row_list = list(rows)
    if enabled:
        outcome = _slot_pool_coverage_submit_precondition_runtime_bridge_outcome_sensor(
            row_list
        )
    else:
        outcome = {}
    expected_score_sum = _as_int(
        os.environ.get(
            "ECA_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_BREADTH_REFERENCE_SCORE_SUM",
            "",
        ),
        default=-1,
    )
    reference_by_game = _env_score_map(
        "ECA_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_BREADTH_REFERENCE_BY_GAME_JSON"
    )
    summary = summarize_slot_pool_coverage_submit_precondition_breadth_non_regression(
        row_list if enabled else [],
        outcome_summary=outcome,
        expected_score_sum=(
            None if expected_score_sum < 0 else int(expected_score_sum)
        ),
        expected_by_game=reference_by_game,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_BREADTH_NON_REGRESSION_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_submit_precondition_rank_guard_shadow_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RANK_GUARD_SHADOW_SENSOR"
    )
    row_list = list(rows)
    outcome = (
        _slot_pool_coverage_submit_precondition_runtime_bridge_outcome_sensor(row_list)
        if enabled
        else {}
    )
    breadth = (
        _slot_pool_coverage_submit_precondition_breadth_non_regression_sensor(row_list)
        if enabled
        else {}
    )
    summary = summarize_slot_pool_coverage_submit_precondition_rank_guard_shadow(
        outcome_summary=outcome,
        breadth_summary=breadth,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RANK_GUARD_SHADOW_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_submit_precondition_rank_guard_runtime_guard_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RANK_GUARD_RUNTIME_GUARD_SENSOR"
    )
    row_list = list(rows)
    rank_shadow = (
        _slot_pool_coverage_submit_precondition_rank_guard_shadow_sensor(row_list)
        if enabled
        else {}
    )
    runtime_events = (
        _slot_pool_coverage_submit_precondition_rank_guard_runtime_events_from_rows(
            row_list
        )
        if enabled
        else []
    )
    summary = summarize_slot_pool_coverage_submit_precondition_rank_guard_runtime_guard(
        rank_shadow_summary=rank_shadow,
        runtime_events=runtime_events,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_RANK_GUARD_RUNTIME_GUARD_SENSOR=1"
        )
    return summary


def _slot_pool_coverage_submit_precondition_live_rank_guard_focused_runtime_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_LIVE_RANK_GUARD_FOCUSED_RUNTIME_SENSOR"
    )
    row_list = list(rows)
    runtime_guard = (
        _slot_pool_coverage_submit_precondition_rank_guard_runtime_guard_sensor(
            row_list
        )
        if enabled
        else {}
    )
    summary = (
        summarize_slot_pool_coverage_submit_precondition_live_rank_guard_focused_runtime(
            runtime_guard_summary=runtime_guard
        )
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_SLOT_POOL_COVERAGE_SUBMIT_PRECONDITION_LIVE_RANK_GUARD_FOCUSED_RUNTIME_SENSOR=1"
        )
    return summary


def _force_field_merge_containment_diagnostic_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_FORCE_FIELD_MERGE_CONTAINMENT_DIAGNOSTIC_SENSOR")
    packets = _force_field_merge_containment_packets_from_rows(rows) if enabled else []
    summary = summarize_force_field_merge_containment_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_FORCE_FIELD_MERGE_CONTAINMENT_DIAGNOSTIC_SENSOR=1"
        )
    return summary


def _force_field_merge_containment_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_FORCE_FIELD_MERGE_CONTAINMENT_POSTERIOR")
    packets = _force_field_merge_containment_packets_from_rows(rows) if enabled else []
    summary = summarize_force_field_merge_containment_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_FORCE_FIELD_MERGE_CONTAINMENT_POSTERIOR=1"
        )
    return summary


def _force_field_merge_containment_candidate_miner(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_FORCE_FIELD_MERGE_CONTAINMENT_CANDIDATE_MINER")
    packets = _force_field_merge_containment_packets_from_rows(rows) if enabled else []
    posterior = (
        summarize_force_field_merge_containment_posterior(
            packets,
            sample_limit=128,
        )
        if enabled
        else {}
    )
    summary = summarize_force_field_merge_containment_candidate_miner(
        packets,
        posterior_summary=posterior,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_FORCE_FIELD_MERGE_CONTAINMENT_CANDIDATE_MINER=1"
        )
    return summary


def _force_field_merge_containment_guarded_probe_outcome_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_FORCE_FIELD_MERGE_CONTAINMENT_GUARDED_PROBE_OUTCOME_SENSOR"
    )
    summary = summarize_force_field_merge_containment_guarded_probe_outcomes(
        rows if enabled else []
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_FORCE_FIELD_MERGE_CONTAINMENT_GUARDED_PROBE_OUTCOME_SENSOR=1"
        )
    return summary


def _force_field_merge_containment_counterfactual_control_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_FORCE_FIELD_MERGE_CONTAINMENT_COUNTERFACTUAL_CONTROL_SENSOR"
    )
    outcome = (
        summarize_force_field_merge_containment_guarded_probe_outcomes(rows)
        if enabled
        else {}
    )
    summary = summarize_force_field_merge_containment_counterfactual_controls(
        rows if enabled else [],
        outcome_summary=outcome,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_FORCE_FIELD_MERGE_CONTAINMENT_COUNTERFACTUAL_CONTROL_SENSOR=1"
        )
    return summary


def _force_field_merge_containment_all25_non_regression_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    row_list = [dict(row) for row in rows if isinstance(row, Mapping)]
    enabled = _env_flag(
        "ECA_ENABLE_FORCE_FIELD_MERGE_CONTAINMENT_ALL25_NON_REGRESSION_SENSOR"
    )
    expected_score_sum = _as_int(
        os.environ.get(
            "ECA_FORCE_FIELD_MERGE_CONTAINMENT_ALL25_REFERENCE_SCORE_SUM",
            "-1",
        ),
        -1,
    )
    reference_by_game = _env_score_map(
        "ECA_FORCE_FIELD_MERGE_CONTAINMENT_ALL25_REFERENCE_BY_GAME_JSON"
    )
    counterfactual = (
        _force_field_merge_containment_counterfactual_control_sensor(row_list)
        if enabled
        else {}
    )
    summary = summarize_force_field_merge_containment_all25_non_regression(
        row_list if enabled else [],
        expected_score_sum=(
            None if expected_score_sum < 0 else int(expected_score_sum)
        ),
        expected_by_game=reference_by_game,
        counterfactual_summary=counterfactual,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_FORCE_FIELD_MERGE_CONTAINMENT_ALL25_NON_REGRESSION_SENSOR=1"
        )
    return summary


def _force_field_merge_containment_score_target_gap_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    row_list = [dict(row) for row in rows if isinstance(row, Mapping)]
    enabled = _env_flag(
        "ECA_ENABLE_FORCE_FIELD_MERGE_CONTAINMENT_SCORE_TARGET_GAP_SENSOR"
    )
    if enabled:
        packets = _force_field_merge_containment_packets_from_rows(row_list)
        diagnostic = _force_field_merge_containment_diagnostic_sensor(row_list)
        posterior = _force_field_merge_containment_posterior(row_list)
        candidate_miner = _force_field_merge_containment_candidate_miner(row_list)
        relation = summarize_force_field_containment_score_target_relations(packets)
        outcome = _force_field_merge_containment_guarded_probe_outcome_sensor(row_list)
        counterfactual = _force_field_merge_containment_counterfactual_control_sensor(
            row_list
        )
    else:
        diagnostic = {}
        posterior = {}
        candidate_miner = {}
        relation = {}
        outcome = {}
        counterfactual = {}
    summary = summarize_force_field_merge_containment_score_target_gap(
        diagnostic_summary=diagnostic,
        posterior_summary=posterior,
        candidate_miner_summary=candidate_miner,
        relation_summary=relation,
        outcome_summary=outcome,
        counterfactual_summary=counterfactual,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_FORCE_FIELD_MERGE_CONTAINMENT_SCORE_TARGET_GAP_SENSOR=1"
        )
    return summary


def _force_field_containment_score_target_relation_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_FORCE_FIELD_CONTAINMENT_SCORE_TARGET_RELATION_SENSOR"
    )
    packets = _force_field_merge_containment_packets_from_rows(rows) if enabled else []
    summary = summarize_force_field_containment_score_target_relations(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_FORCE_FIELD_CONTAINMENT_SCORE_TARGET_RELATION_SENSOR=1"
        )
    return summary


def _force_field_score_positive_counterfactual_pair_search_candidate_miner(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_FORCE_FIELD_SCORE_POSITIVE_COUNTERFACTUAL_PAIR_SEARCH_CANDIDATE_MINER"
    )
    packets = _force_field_merge_containment_packets_from_rows(rows) if enabled else []
    relation = (
        summarize_force_field_containment_score_target_relations(packets)
        if enabled
        else {}
    )
    summary = summarize_force_field_score_positive_counterfactual_pair_search_candidates(
        packets,
        relation_summary=relation,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set "
            "ECA_ENABLE_FORCE_FIELD_SCORE_POSITIVE_COUNTERFACTUAL_PAIR_SEARCH_CANDIDATE_MINER=1"
        )
    return summary


def _program_row_semantics_diagnostic_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PROGRAM_ROW_SEMANTICS_DIAGNOSTIC_SENSOR")
    packets = _program_row_semantics_packets_from_rows(rows) if enabled else []
    summary = summarize_program_row_semantics_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PROGRAM_ROW_SEMANTICS_DIAGNOSTIC_SENSOR=1"
        )
    return summary


def _program_row_semantics_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PROGRAM_ROW_SEMANTICS_POSTERIOR")
    packets = _program_row_semantics_packets_from_rows(rows) if enabled else []
    summary = summarize_program_row_semantics_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PROGRAM_ROW_SEMANTICS_POSTERIOR=1"
        )
    return summary


def _program_row_semantics_candidate_miner(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PROGRAM_ROW_SEMANTICS_CANDIDATE_MINER")
    packets = _program_row_semantics_packets_from_rows(rows) if enabled else []
    posterior = (
        summarize_program_row_semantics_posterior(packets, sample_limit=128)
        if enabled
        else {}
    )
    summary = summarize_program_row_semantics_candidate_miner(
        packets,
        posterior_summary=posterior,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PROGRAM_ROW_SEMANTICS_CANDIDATE_MINER=1"
        )
    return summary


def _program_row_semantics_guarded_probe_outcome_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PROGRAM_ROW_SEMANTICS_GUARDED_PROBE_OUTCOME_SENSOR")
    summary = summarize_program_row_semantics_guarded_probe_outcomes(
        rows if enabled else []
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PROGRAM_ROW_SEMANTICS_GUARDED_PROBE_OUTCOME_SENSOR=1"
        )
    return summary


def _program_row_semantics_counterfactual_control_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_PROGRAM_ROW_SEMANTICS_COUNTERFACTUAL_CONTROL_SENSOR"
    )
    row_list = list(rows) if enabled else []
    outcome = (
        summarize_program_row_semantics_guarded_probe_outcomes(
            row_list,
            sample_limit=128,
        )
        if enabled
        else {}
    )
    summary = summarize_program_row_semantics_counterfactual_controls(
        row_list,
        outcome_summary=outcome,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PROGRAM_ROW_SEMANTICS_COUNTERFACTUAL_CONTROL_SENSOR=1"
        )
    return summary


def _program_row_semantics_all25_non_regression_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PROGRAM_ROW_SEMANTICS_ALL25_NON_REGRESSION_SENSOR")
    expected_score_sum = _as_int(
        os.environ.get("ECA_PROGRAM_ROW_SEMANTICS_ALL25_REFERENCE_SCORE_SUM", ""),
        default=-1,
    )
    reference_by_game = _env_score_map(
        "ECA_PROGRAM_ROW_SEMANTICS_ALL25_REFERENCE_BY_GAME_JSON"
    )
    summary = summarize_program_row_semantics_all25_non_regression(
        list(rows) if enabled else [],
        expected_score_sum=(
            None if expected_score_sum < 0 else int(expected_score_sum)
        ),
        expected_by_game=reference_by_game,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PROGRAM_ROW_SEMANTICS_ALL25_NON_REGRESSION_SENSOR=1"
        )
    return summary


def _program_spell_pattern_diagnostic_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PROGRAM_SPELL_PATTERN_DIAGNOSTIC_SENSOR")
    packets = _program_spell_pattern_packets_from_rows(rows) if enabled else []
    summary = summarize_program_spell_pattern_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PROGRAM_SPELL_PATTERN_DIAGNOSTIC_SENSOR=1"
        )
    return summary


def _program_spell_pattern_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PROGRAM_SPELL_PATTERN_POSTERIOR")
    packets = _program_spell_pattern_packets_from_rows(rows) if enabled else []
    summary = summarize_program_spell_pattern_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PROGRAM_SPELL_PATTERN_POSTERIOR=1"
        )
    return summary


def _program_spell_pattern_candidate_miner(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PROGRAM_SPELL_PATTERN_CANDIDATE_MINER")
    packets = _program_spell_pattern_packets_from_rows(rows) if enabled else []
    posterior = (
        summarize_program_spell_pattern_posterior(
            packets,
            sample_limit=128,
        )
        if enabled
        else {}
    )
    summary = summarize_program_spell_pattern_candidate_miner(
        packets,
        posterior_summary=posterior,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PROGRAM_SPELL_PATTERN_CANDIDATE_MINER=1"
        )
    return summary


def _program_spell_pattern_guarded_probe_outcome_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PROGRAM_SPELL_PATTERN_GUARDED_PROBE_OUTCOME_SENSOR")
    summary = summarize_program_spell_pattern_guarded_probe_outcomes(
        list(rows) if enabled else []
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PROGRAM_SPELL_PATTERN_GUARDED_PROBE_OUTCOME_SENSOR=1"
        )
    return summary


def _program_spell_pattern_counterfactual_control_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PROGRAM_SPELL_PATTERN_COUNTERFACTUAL_CONTROL_SENSOR")
    outcome = (
        summarize_program_spell_pattern_guarded_probe_outcomes(
            list(rows),
            sample_limit=128,
        )
        if enabled
        else {}
    )
    summary = summarize_program_spell_pattern_counterfactual_controls(
        list(rows) if enabled else [],
        outcome_summary=outcome,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PROGRAM_SPELL_PATTERN_COUNTERFACTUAL_CONTROL_SENSOR=1"
        )
    return summary


def _program_spell_pattern_all25_non_regression_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PROGRAM_SPELL_PATTERN_ALL25_NON_REGRESSION_SENSOR")
    expected_score_sum = _as_int(
        os.environ.get("ECA_PROGRAM_SPELL_PATTERN_ALL25_REFERENCE_SCORE_SUM", ""),
        -1,
    )
    reference_by_game = _env_score_map(
        "ECA_PROGRAM_SPELL_PATTERN_ALL25_REFERENCE_BY_GAME_JSON"
    )
    summary = summarize_program_spell_pattern_all25_non_regression(
        list(rows) if enabled else [],
        expected_score_sum=(
            None if expected_score_sum < 0 else int(expected_score_sum)
        ),
        expected_by_game=reference_by_game,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PROGRAM_SPELL_PATTERN_ALL25_NON_REGRESSION_SENSOR=1"
        )
    return summary


def _kinematic_membership_candidate_miner_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_KINEMATIC_MEMBERSHIP_CANDIDATE_MINER_SENSOR")
    summary = summarize_kinematic_membership_candidate_miner(
        list(rows) if enabled else []
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_KINEMATIC_MEMBERSHIP_CANDIDATE_MINER_SENSOR=1"
        )
    return summary


def _kinematic_membership_counterfactual_control_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_KINEMATIC_MEMBERSHIP_COUNTERFACTUAL_CONTROL_SENSOR"
    )
    row_list = list(rows) if enabled else []
    candidate_summary = (
        summarize_kinematic_membership_candidate_miner(
            row_list,
            sample_limit=128,
        )
        if enabled
        else {}
    )
    summary = summarize_kinematic_membership_counterfactual_controls(
        row_list,
        candidate_summary=candidate_summary,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_KINEMATIC_MEMBERSHIP_COUNTERFACTUAL_CONTROL_SENSOR=1"
        )
    return summary


def _kinematic_membership_all25_non_regression_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_KINEMATIC_MEMBERSHIP_ALL25_NON_REGRESSION_SENSOR")
    row_list = list(rows) if enabled else []
    expected_score_sum = _as_int(
        os.environ.get("ECA_KINEMATIC_MEMBERSHIP_ALL25_REFERENCE_SCORE_SUM", ""),
        -1,
    )
    reference_by_game = _env_score_map(
        "ECA_KINEMATIC_MEMBERSHIP_ALL25_REFERENCE_BY_GAME_JSON"
    )
    counterfactual = (
        summarize_kinematic_membership_counterfactual_controls(
            row_list,
            candidate_summary=summarize_kinematic_membership_candidate_miner(
                row_list,
                sample_limit=128,
            ),
            sample_limit=128,
        )
        if enabled
        else {}
    )
    summary = summarize_kinematic_membership_all25_non_regression(
        row_list,
        expected_score_sum=(
            None if expected_score_sum < 0 else int(expected_score_sum)
        ),
        expected_by_game=reference_by_game,
        counterfactual_summary=counterfactual,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_KINEMATIC_MEMBERSHIP_ALL25_NON_REGRESSION_SENSOR=1"
        )
    return summary


def _platform_destructible_topology_diagnostic_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PLATFORM_DESTRUCTIBLE_TOPOLOGY_DIAGNOSTIC_SENSOR")
    packets = _platform_destructible_topology_packets_from_rows(rows) if enabled else []
    summary = summarize_platform_destructible_topology_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PLATFORM_DESTRUCTIBLE_TOPOLOGY_DIAGNOSTIC_SENSOR=1"
        )
    return summary


def _component_extent_growth_diagnostic_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_COMPONENT_EXTENT_GROWTH_DIAGNOSTIC_SENSOR")
    packets = _component_extent_growth_packets_from_rows(rows) if enabled else []
    summary = summarize_component_extent_growth_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_COMPONENT_EXTENT_GROWTH_DIAGNOSTIC_SENSOR=1"
        )
    return summary


def _component_extent_growth_precision_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_COMPONENT_EXTENT_GROWTH_PRECISION_POSTERIOR")
    packets = _component_extent_growth_packets_from_rows(rows) if enabled else []
    summary = summarize_component_extent_growth_precision_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_COMPONENT_EXTENT_GROWTH_PRECISION_POSTERIOR=1"
        )
    return summary


def _component_extent_growth_slider_stride_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_COMPONENT_EXTENT_GROWTH_SLIDER_STRIDE_POSTERIOR")
    packets = _component_extent_growth_packets_from_rows(rows) if enabled else []
    summary = summarize_component_extent_growth_slider_stride_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_COMPONENT_EXTENT_GROWTH_SLIDER_STRIDE_POSTERIOR=1"
        )
    return summary


def _component_extent_growth_target_distance_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_COMPONENT_EXTENT_GROWTH_TARGET_DISTANCE_POSTERIOR")
    packets = _component_extent_growth_packets_from_rows(rows) if enabled else []
    summary = summarize_component_extent_growth_target_distance_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_COMPONENT_EXTENT_GROWTH_TARGET_DISTANCE_POSTERIOR=1"
        )
    return summary


def _component_extent_growth_target_distance_rank_guard_focused_runtime(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_COMPONENT_EXTENT_GROWTH_TARGET_DISTANCE_RANK_GUARD")
    window = max(
        1,
        min(
            128,
            _as_int(
                os.environ.get(
                    "ECA_COMPONENT_EXTENT_GROWTH_TARGET_DISTANCE_RANK_GUARD_SCORE_WINDOW",
                    "40",
                ),
                40,
            ),
        ),
    )
    events = (
        _component_extent_growth_target_distance_rank_guard_events_from_rows(
            rows,
            score_window=window,
        )
        if enabled
        else []
    )

    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    blocked_reasons: Counter[str] = Counter()
    selected_site_histogram: Counter[str] = Counter()
    applied_count = 0
    action_changed_count = 0
    admitted_no_change_count = 0
    target_distance_improved_after_rank_guard_count = 0
    target_distance_worsened_after_rank_guard_count = 0
    delayed_positive_event_count = 0
    delayed_score_delta_sum = 0
    score_positive_games: set[str] = set()
    unsafe_action_generation_count = 0
    generated_candidate_count = 0

    for event in events:
        game_id = str(event.get("game_id", "") or "unknown")
        by_game[game_id]["reviewed_count"] += 1
        blocked_reason = str(event.get("blocked_reason", "") or "")
        if blocked_reason:
            blocked_reasons[blocked_reason] += 1
            by_game[game_id][f"blocked_reason:{blocked_reason}"] += 1
        if bool(event.get("action_generation_allowed")):
            unsafe_action_generation_count += 1
        if bool(event.get("candidate_generation_enabled")):
            generated_candidate_count += 1
        if bool(event.get("applied")):
            applied_count += 1
            by_game[game_id]["applied_count"] += 1
            site = event.get("selected_site")
            if isinstance(site, list) and len(site) == 2:
                selected_site_histogram[f"{site[0]},{site[1]}"] += 1
            improved_count = _as_int(
                event.get("target_distance_improved_after_rank_guard_count"),
                0,
            )
            worsened_count = _as_int(
                event.get("target_distance_worsened_after_rank_guard_count"),
                0,
            )
            target_distance_improved_after_rank_guard_count += improved_count
            target_distance_worsened_after_rank_guard_count += worsened_count
            by_game[game_id][
                "target_distance_improved_after_rank_guard_count"
            ] += improved_count
            by_game[game_id][
                "target_distance_worsened_after_rank_guard_count"
            ] += worsened_count
            delayed_delta = _as_int(event.get("delayed_score_delta"), 0)
            if delayed_delta > 0:
                delayed_positive_event_count += 1
                delayed_score_delta_sum += delayed_delta
                score_positive_games.add(game_id)
                by_game[game_id]["delayed_positive_event_count"] += 1
                by_game[game_id]["delayed_score_delta_sum"] += delayed_delta
        if bool(event.get("action_changed")):
            action_changed_count += 1
            by_game[game_id]["action_changed_count"] += 1
        elif bool(event.get("applied")):
            admitted_no_change_count += 1
            by_game[game_id]["admitted_no_change_count"] += 1

    promotion_state = "component_extent_growth_target_distance_rank_guard_disabled"
    if enabled and applied_count and delayed_positive_event_count:
        promotion_state = (
            "component_extent_growth_target_distance_rank_guard_focused_lift"
        )
    elif enabled and applied_count:
        promotion_state = (
            "component_extent_growth_target_distance_rank_guard_ranked_no_score_lift"
        )
    elif enabled:
        promotion_state = (
            "component_extent_growth_target_distance_rank_guard_no_admissions"
        )

    summary = {
        "schema_version": (
            "eca_component_extent_growth_target_distance_rank_guard_focused_runtime.v1"
        ),
        "enabled": bool(enabled),
        "default_off": True,
        "allowed_connection": (
            "observed_target_distance_improvement_to_existing_site_replay_rank"
        ),
        "planner_connection_enabled": True,
        "ranking_enabled": True,
        "candidate_generation_enabled": False,
        "action_generation_allowed": False,
        "reviewed_count": int(len(events)),
        "applied_count": int(applied_count),
        "action_changed_count": int(action_changed_count),
        "admitted_no_change_count": int(admitted_no_change_count),
        "blocked_reasons": dict(sorted(blocked_reasons.items())),
        "selected_site_histogram": dict(sorted(selected_site_histogram.items())),
        "target_distance_improved_after_rank_guard_count": int(
            target_distance_improved_after_rank_guard_count
        ),
        "target_distance_worsened_after_rank_guard_count": int(
            target_distance_worsened_after_rank_guard_count
        ),
        "score_window": int(window),
        "delayed_positive_event_count": int(delayed_positive_event_count),
        "delayed_score_delta_sum": int(delayed_score_delta_sum),
        "score_positive_game_count": int(len(score_positive_games)),
        "score_positive_games": sorted(score_positive_games),
        "unsafe_action_generation_count": int(unsafe_action_generation_count),
        "generated_candidate_count": int(generated_candidate_count),
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "promotion_state": promotion_state,
    }
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_COMPONENT_EXTENT_GROWTH_TARGET_DISTANCE_RANK_GUARD=1"
        )
    return summary


def _platform_destructible_topology_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PLATFORM_DESTRUCTIBLE_TOPOLOGY_POSTERIOR")
    packets = _platform_destructible_topology_packets_from_rows(rows) if enabled else []
    summary = summarize_platform_destructible_topology_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PLATFORM_DESTRUCTIBLE_TOPOLOGY_POSTERIOR=1"
        )
    return summary


def _platform_destructible_topology_candidate_miner(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_PLATFORM_DESTRUCTIBLE_TOPOLOGY_CANDIDATE_MINER")
    packets = _platform_destructible_topology_packets_from_rows(rows) if enabled else []
    posterior = summarize_platform_destructible_topology_posterior(packets)
    summary = summarize_platform_destructible_topology_candidate_miner(
        packets,
        posterior_summary=posterior,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PLATFORM_DESTRUCTIBLE_TOPOLOGY_CANDIDATE_MINER=1"
        )
    return summary


def _platform_destructible_topology_guarded_probe_outcome_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_PLATFORM_DESTRUCTIBLE_TOPOLOGY_GUARDED_PROBE_OUTCOME_SENSOR"
    )
    summary = summarize_platform_destructible_topology_guarded_probe_outcomes(
        rows if enabled else []
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PLATFORM_DESTRUCTIBLE_TOPOLOGY_GUARDED_PROBE_OUTCOME_SENSOR=1"
        )
    return summary


def _platform_destructible_topology_counterfactual_control_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_PLATFORM_DESTRUCTIBLE_TOPOLOGY_COUNTERFACTUAL_CONTROL_SENSOR"
    )
    outcome = summarize_platform_destructible_topology_guarded_probe_outcomes(
        rows if enabled else []
    )
    summary = summarize_platform_destructible_topology_counterfactual_controls(
        rows if enabled else [],
        outcome_summary=outcome,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PLATFORM_DESTRUCTIBLE_TOPOLOGY_COUNTERFACTUAL_CONTROL_SENSOR=1"
        )
    return summary


def _platform_destructible_topology_all25_non_regression_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    row_list = [dict(row) for row in rows if isinstance(row, Mapping)]
    enabled = _env_flag(
        "ECA_ENABLE_PLATFORM_DESTRUCTIBLE_TOPOLOGY_ALL25_NON_REGRESSION_SENSOR"
    )
    expected_score_sum = _as_int(
        os.environ.get(
            "ECA_PLATFORM_DESTRUCTIBLE_TOPOLOGY_ALL25_REFERENCE_SCORE_SUM",
            "-1",
        ),
        -1,
    )
    reference_by_game = _env_score_map(
        "ECA_PLATFORM_DESTRUCTIBLE_TOPOLOGY_ALL25_REFERENCE_BY_GAME_JSON"
    )
    counterfactual = (
        _platform_destructible_topology_counterfactual_control_sensor(row_list)
        if enabled
        else {}
    )
    summary = summarize_platform_destructible_topology_all25_non_regression(
        row_list if enabled else [],
        expected_score_sum=(
            None if expected_score_sum < 0 else int(expected_score_sum)
        ),
        expected_by_game=reference_by_game,
        counterfactual_summary=counterfactual,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_PLATFORM_DESTRUCTIBLE_TOPOLOGY_ALL25_NON_REGRESSION_SENSOR=1"
        )
    return summary


def _moving_sprite_effect_component_spatial_stability_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_MOVING_SPRITE_EFFECT_COMPONENT_SPATIAL_STABILITY_SENSOR"
    )
    moving_packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    spell_packets = _program_spell_pattern_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_effect_component_spatial_stability_sensor(
        moving_packets,
        spell_packets,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_EFFECT_COMPONENT_SPATIAL_STABILITY_SENSOR=1"
        )
    return summary


def _moving_sprite_sparse_chain_rail_state_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MOVING_SPRITE_SPARSE_CHAIN_RAIL_STATE_SENSOR")
    moving_packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    contact_packets = _multi_body_contact_topology_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_sparse_chain_rail_state_sensor(
        moving_packets,
        contact_packets,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_SPARSE_CHAIN_RAIL_STATE_SENSOR=1"
        )
    return summary


def _moving_sprite_target_goal_binding_state_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MOVING_SPRITE_TARGET_GOAL_BINDING_STATE_SENSOR")
    moving_packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    spell_packets = _program_spell_pattern_packets_from_rows(rows) if enabled else []
    platform_packets = _platform_destructible_topology_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_target_goal_binding_state_sensor(
        moving_packets,
        spell_packets,
        platform_packets,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_TARGET_GOAL_BINDING_STATE_SENSOR=1"
        )
    return summary


def _moving_sprite_goal_anchor_perception_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MOVING_SPRITE_GOAL_ANCHOR_PERCEPTION_SENSOR")
    moving_packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    spell_packets = _program_spell_pattern_packets_from_rows(rows) if enabled else []
    platform_packets = _platform_destructible_topology_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_goal_anchor_perception_sensor(
        moving_packets,
        spell_packets,
        platform_packets,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_GOAL_ANCHOR_PERCEPTION_SENSOR=1"
        )
    return summary


def _moving_sprite_guarded_planner_connection_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MOVING_SPRITE_GUARDED_PLANNER_CONNECTION_SENSOR")
    moving_packets = _moving_sprite_runtime_trace_packets_from_rows(rows) if enabled else []
    spell_packets = _program_spell_pattern_packets_from_rows(rows) if enabled else []
    platform_packets = _platform_destructible_topology_packets_from_rows(rows) if enabled else []
    summary = summarize_moving_sprite_guarded_planner_connection_sensor(
        moving_packets,
        spell_packets,
        platform_packets,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MOVING_SPRITE_GUARDED_PLANNER_CONNECTION_SENSOR=1"
        )
    return summary


def _round_memory_diagnostic_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_ROUND_MEMORY_DIAGNOSTIC_SENSOR")
    packets = _round_memory_packets_from_rows(rows) if enabled else []
    summary = summarize_round_memory_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = "set ECA_ENABLE_ROUND_MEMORY_DIAGNOSTIC_SENSOR=1"
    return summary


def _round_memory_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_ROUND_MEMORY_POSTERIOR")
    packets = _round_memory_packets_from_rows(rows) if enabled else []
    summary = summarize_round_memory_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = "set ECA_ENABLE_ROUND_MEMORY_POSTERIOR=1"
    return summary


def _round_memory_guarded_probe_outcome_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    window: int = 8,
    sample_limit: int = 12,
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_ROUND_MEMORY_GUARDED_PROBE")
    base: dict[str, Any] = {
        "schema_version": "eca_round_memory_guarded_probe_outcome_sensor.v1",
        "enabled": bool(enabled),
        "default_off": True,
        "diagnostic_only": False,
        "planner_connection_enabled": bool(enabled),
        "candidate_generation_enabled": bool(enabled),
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": bool(enabled),
        "direct_llm_action_allowed": False,
        "protected_probe_execution_enabled": bool(enabled),
        "window": int(window),
    }
    if not enabled:
        return {
            **base,
            "execution_count": 0,
            "admitted_plan_count": 0,
            "lease_execution_count": 0,
            "score_closure_plan_count": 0,
            "score_delta_after_guard_sum": 0,
            "promotion_state": "disabled",
            "samples": [],
            "disabled_reason": "set ECA_ENABLE_ROUND_MEMORY_GUARDED_PROBE=1",
        }

    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        final_score = _as_int(row.get("final_score"), 0)
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue

        current_block: list[int] = []

        def flush_block() -> None:
            if not current_block:
                return
            start = current_block[0]
            end = current_block[-1]
            first = trace[start] if start < len(trace) else {}
            last = trace[end] if end < len(trace) else {}
            score_before = _trace_score_before(first, 0)
            score_after = (
                _trace_score_before(trace[end + 1], score_before)
                if end + 1 < len(trace)
                else final_score
            )
            delayed_delta, delay_steps = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=end,
                window=window,
                default_score=_trace_score_before(last, score_before),
                final_score=final_score,
            )
            plan_delta = max(0, int(score_after) - int(score_before), int(delayed_delta))
            counts["plan_count"] += 1
            by_game[game_id]["plan_count"] += 1
            counts["score_delta_after_guard_sum"] += int(plan_delta)
            by_game[game_id]["score_delta_after_guard_sum"] += int(plan_delta)
            if plan_delta > 0:
                counts["score_closure_plan_count"] += 1
                by_game[game_id]["score_closure_plan_count"] += 1
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "start_step": _as_int(first.get("step"), start)
                        if isinstance(first, Mapping)
                        else start,
                        "end_step": _as_int(last.get("step"), end)
                        if isinstance(last, Mapping)
                        else end,
                        "action_sequence": [
                            str(trace[index].get("action", "") or "")
                            for index in current_block
                            if isinstance(trace[index], Mapping)
                        ],
                        "score_before": int(score_before),
                        "score_after": int(score_after),
                        "score_delta_after_guard": int(plan_delta),
                        "delayed_score_delta": int(delayed_delta),
                        "delay_steps": int(delay_steps),
                    }
                )
            current_block.clear()

        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                flush_block()
                continue
            unsolved = item.get("unsolved_replan")
            unsolved = unsolved if isinstance(unsolved, Mapping) else {}
            is_guard = str(item.get("phase", "") or "") == "round_memory_guarded_probe"
            is_guard = is_guard or str(unsolved.get("source", "") or "") == (
                "round_memory_guarded_probe"
            )
            if not is_guard:
                flush_block()
                continue
            current_block.append(index)
            counts["execution_count"] += 1
            by_game[game_id]["execution_count"] += 1
            reason = str(unsolved.get("reason", "") or "")
            admission_mode = str(unsolved.get("admission_mode", "") or "")
            if admission_mode:
                counts[f"admission_mode:{admission_mode}"] += 1
                by_game[game_id][f"admission_mode:{admission_mode}"] += 1
            if reason == "admitted_round_memory_guarded_probe_plan":
                counts["admitted_plan_count"] += 1
                by_game[game_id]["admitted_plan_count"] += 1
                if admission_mode == "frame_admission":
                    counts["frame_admission_admitted_plan_count"] += 1
                    by_game[game_id]["frame_admission_admitted_plan_count"] += 1
                if admission_mode == "game_allowlist":
                    counts["game_allowlist_admitted_plan_count"] += 1
                    by_game[game_id]["game_allowlist_admitted_plan_count"] += 1
            if reason == "continued_round_memory_guarded_probe_plan":
                counts["lease_execution_count"] += 1
                by_game[game_id]["lease_execution_count"] += 1
            proposal = unsolved.get("proposal")
            proposal = proposal if isinstance(proposal, Mapping) else {}
            if bool(proposal.get("accepted")):
                counts["accepted_proposal_count"] += 1
                by_game[game_id]["accepted_proposal_count"] += 1
            if bool(proposal.get("direct_llm_action_allowed")):
                counts["direct_llm_action_allowed_count"] += 1
                by_game[game_id]["direct_llm_action_allowed_count"] += 1
        flush_block()

    execution_count = int(counts.get("execution_count", 0))
    score_closure_plans = int(counts.get("score_closure_plan_count", 0))
    if execution_count <= 0:
        promotion_state = "no_guarded_probe_execution"
    elif score_closure_plans > 0:
        promotion_state = "focused_score_closure"
    else:
        promotion_state = "executed_no_score_closure"

    return {
        **base,
        "execution_count": execution_count,
        "admitted_plan_count": int(counts.get("admitted_plan_count", 0)),
        "frame_admission_admitted_plan_count": int(
            counts.get("frame_admission_admitted_plan_count", 0)
        ),
        "game_allowlist_admitted_plan_count": int(
            counts.get("game_allowlist_admitted_plan_count", 0)
        ),
        "accepted_proposal_count": int(counts.get("accepted_proposal_count", 0)),
        "lease_execution_count": int(counts.get("lease_execution_count", 0)),
        "plan_count": int(counts.get("plan_count", 0)),
        "score_closure_plan_count": score_closure_plans,
        "score_delta_after_guard_sum": int(
            counts.get("score_delta_after_guard_sum", 0)
        ),
        "direct_llm_action_allowed_count": int(
            counts.get("direct_llm_action_allowed_count", 0)
        ),
        "admission_mode_counts": {
            key.split("admission_mode:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("admission_mode:")
        },
        "promotion_state": promotion_state,
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "samples": samples,
    }


def _transition_state_variable_score_closure_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TRANSITION_STATE_VARIABLE_SCORE_CLOSURE_POSTERIOR")
    packets = _transition_state_variable_packets_from_rows(rows) if enabled else []
    summary = summarize_transition_state_variable_score_closure_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TRANSITION_STATE_VARIABLE_SCORE_CLOSURE_POSTERIOR=1"
        )
    return summary


def _transition_state_variable_score_retest_candidate_miner(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_TRANSITION_STATE_VARIABLE_SCORE_RETEST_CANDIDATE_MINER"
    )
    packets = _transition_state_variable_packets_from_rows(rows) if enabled else []
    summary = summarize_transition_state_variable_score_retest_candidates(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TRANSITION_STATE_VARIABLE_SCORE_RETEST_CANDIDATE_MINER=1"
        )
    return summary


def _transition_state_variable_delayed_score_closure_attribution(
    rows: Iterable[Mapping[str, Any]],
    *,
    window: int = 16,
    sample_limit: int = 16,
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_TRANSITION_STATE_VARIABLE_DELAYED_SCORE_CLOSURE_ATTRIBUTION"
    )
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    by_transition_key: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    if not enabled:
        return {
            "schema_version": (
                "eca_transition_state_variable_delayed_score_closure_attribution.v1"
            ),
            "enabled": False,
            "default_off": True,
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "candidate_generation_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "window": int(window),
            "execution_count": 0,
            "delayed_positive_execution_count": 0,
            "positive_control_margin_count": 0,
            "promotion_state": "disabled",
            "disabled_reason": (
                "set ECA_ENABLE_TRANSITION_STATE_VARIABLE_DELAYED_SCORE_CLOSURE_ATTRIBUTION=1"
            ),
        }

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        final_score = _as_int(row.get("final_score"), 0)
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            if str(item.get("phase", "") or "") != "transition_state_score_retest_probe":
                continue
            unsolved = item.get("unsolved_replan")
            if not isinstance(unsolved, Mapping):
                continue
            if not bool(unsolved.get("executed") or unsolved.get("used")):
                continue
            if str(unsolved.get("source", "") or "") != (
                "transition_state_variable_score_retest_probe"
            ):
                continue
            transition_key = str(
                unsolved.get("source_transition_key", "") or "unknown"
            )
            candidate_key = str(unsolved.get("candidate_key", "") or "")
            retest_strategy = str(unsolved.get("retest_strategy", "") or "")
            outcome = unsolved.get("runtime_outcome")
            outcome = outcome if isinstance(outcome, Mapping) else {}
            observed = unsolved.get("observed_effect")
            observed = observed if isinstance(observed, Mapping) else {}
            current_score = _trace_score_before(item, final_score)
            if index + 1 < len(trace):
                next_score = _trace_score_before(trace[index + 1], current_score)
            else:
                next_score = final_score
            runtime_score_delta = max(
                _as_int(outcome.get("score_delta"), 0),
                _as_int(observed.get("score_delta"), 0),
            )
            immediate_delta = max(
                0,
                runtime_score_delta,
                int(next_score) - int(current_score),
            )
            delayed_delta, delay_steps = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=index,
                window=window,
                default_score=current_score,
                final_score=final_score,
            )
            shifted_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=min(index + 1, max(0, len(trace) - 1)),
                window=window,
                default_score=next_score,
                final_score=final_score,
            )
            pre_action_delta = _pre_action_score_delta(
                trace,
                index=index,
                window=window,
                current_score=current_score,
            )
            control_index = _deterministic_control_index(index, len(trace))
            random_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=control_index,
                window=window,
                default_score=_trace_score_before(trace[control_index], current_score)
                if trace
                else current_score,
                final_score=final_score,
            )
            control_components = {
                "shifted_next_step": int(shifted_delta),
                "pre_action_window": int(pre_action_delta),
                "deterministic_random_step": int(random_delta),
            }
            control_max = max(control_components.values())
            dominant_control_sources = [
                key for key, value in sorted(control_components.items()) if value == control_max
            ]
            if control_max <= 0:
                dominant_control_sources = ["none"]
            dominant_control_key = (
                dominant_control_sources[0]
                if len(dominant_control_sources) == 1
                else "tie:" + "+".join(dominant_control_sources)
            )
            join_margin = int(delayed_delta) - int(control_max)
            if immediate_delta > 0:
                closure_state = "direct_probe_score_closure"
            elif delayed_delta > 0:
                closure_state = "delayed_score_closure_after_probe"
            else:
                closure_state = "no_score_closure_after_probe"
            if delayed_delta > 0 and join_margin > 0:
                margin_state = "positive_control_margin"
            elif delayed_delta > 0:
                margin_state = "control_matched_or_ambiguous"
            else:
                margin_state = "no_delayed_score"

            counts["execution_count"] += 1
            counts[f"closure_state:{closure_state}"] += 1
            counts[f"margin_state:{margin_state}"] += 1
            counts["immediate_score_delta_sum"] += int(immediate_delta)
            counts["delayed_score_delta_sum"] += int(delayed_delta)
            counts["shifted_delayed_score_delta_sum"] += int(shifted_delta)
            counts["pre_action_score_delta_sum"] += int(pre_action_delta)
            counts["random_step_score_delta_sum"] += int(random_delta)
            counts["negative_control_max_score_delta_sum"] += int(control_max)
            counts["join_margin_sum"] += int(join_margin)
            counts[f"dominant_negative_control:{dominant_control_key}"] += 1
            if delayed_delta > 0:
                counts["delayed_positive_execution_count"] += 1
                counts[
                    f"delayed_positive_dominant_negative_control:{dominant_control_key}"
                ] += 1
            if delayed_delta > 0 and immediate_delta <= 0:
                counts["delayed_positive_no_immediate_count"] += 1
            if delayed_delta > 0 and join_margin > 0:
                counts["positive_control_margin_count"] += 1
            if delayed_delta > 0 and join_margin <= 0:
                counts["control_matched_delayed_positive_count"] += 1

            for counter in (by_game[game_id], by_transition_key[transition_key]):
                counter["execution_count"] += 1
                counter[f"closure_state:{closure_state}"] += 1
                counter[f"margin_state:{margin_state}"] += 1
                counter["immediate_score_delta_sum"] += int(immediate_delta)
                counter["delayed_score_delta_sum"] += int(delayed_delta)
                counter["shifted_delayed_score_delta_sum"] += int(shifted_delta)
                counter["pre_action_score_delta_sum"] += int(pre_action_delta)
                counter["random_step_score_delta_sum"] += int(random_delta)
                counter["negative_control_max_score_delta_sum"] += int(control_max)
                counter["join_margin_sum"] += int(join_margin)
                counter[f"dominant_negative_control:{dominant_control_key}"] += 1
                if delayed_delta > 0:
                    counter["delayed_positive_execution_count"] += 1
                    counter[
                        f"delayed_positive_dominant_negative_control:{dominant_control_key}"
                    ] += 1
                if delayed_delta > 0 and immediate_delta <= 0:
                    counter["delayed_positive_no_immediate_count"] += 1
                if delayed_delta > 0 and join_margin > 0:
                    counter["positive_control_margin_count"] += 1
                if delayed_delta > 0 and join_margin <= 0:
                    counter["control_matched_delayed_positive_count"] += 1

            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or "unknown"),
                        "step": _as_int(item.get("step"), index),
                        "phase": str(item.get("phase", "") or ""),
                        "action": str(item.get("action", "") or ""),
                        "candidate_key": candidate_key,
                        "source_transition_key": transition_key,
                        "retest_strategy": retest_strategy,
                        "runtime_verdict": str(unsolved.get("verdict", "") or ""),
                        "runtime_effect_type": str(
                            observed.get("effect_type", "")
                            or outcome.get("effect_type", "")
                            or ""
                        ),
                        "score_before": int(current_score),
                        "immediate_score_delta": int(immediate_delta),
                        "delayed_score_delta": int(delayed_delta),
                        "delay_steps": int(delay_steps),
                        "shifted_delayed_score_delta": int(shifted_delta),
                        "pre_action_score_delta": int(pre_action_delta),
                        "random_step_score_delta": int(random_delta),
                        "negative_control_max_score_delta": int(control_max),
                        "dominant_negative_control_sources": dominant_control_sources,
                        "dominant_negative_control_key": dominant_control_key,
                        "join_margin": int(join_margin),
                        "closure_state": closure_state,
                        "margin_state": margin_state,
                    }
                )

    execution_count = int(counts.get("execution_count", 0))
    delayed_positive = int(counts.get("delayed_positive_execution_count", 0))
    positive_margin = int(counts.get("positive_control_margin_count", 0))
    if execution_count <= 0:
        promotion_state = "no_transition_score_retest_execution"
    elif positive_margin > 0:
        promotion_state = "delayed_score_closure_positive_control_margin"
    elif delayed_positive > 0:
        promotion_state = "delayed_score_closure_control_matched"
    else:
        promotion_state = "no_delayed_score_closure"

    return {
        "schema_version": (
            "eca_transition_state_variable_delayed_score_closure_attribution.v1"
        ),
        "enabled": True,
        "default_off": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "window": int(window),
        "execution_count": execution_count,
        "immediate_score_delta_sum": int(counts.get("immediate_score_delta_sum", 0)),
        "delayed_score_delta_sum": int(counts.get("delayed_score_delta_sum", 0)),
        "shifted_delayed_score_delta_sum": int(
            counts.get("shifted_delayed_score_delta_sum", 0)
        ),
        "pre_action_score_delta_sum": int(
            counts.get("pre_action_score_delta_sum", 0)
        ),
        "random_step_score_delta_sum": int(
            counts.get("random_step_score_delta_sum", 0)
        ),
        "negative_control_max_score_delta_sum": int(
            counts.get("negative_control_max_score_delta_sum", 0)
        ),
        "join_margin_sum": int(counts.get("join_margin_sum", 0)),
        "delayed_positive_execution_count": delayed_positive,
        "delayed_positive_no_immediate_count": int(
            counts.get("delayed_positive_no_immediate_count", 0)
        ),
        "positive_control_margin_count": positive_margin,
        "control_matched_delayed_positive_count": int(
            counts.get("control_matched_delayed_positive_count", 0)
        ),
        "promotion_state": promotion_state,
        "state_counts": {
            key: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("closure_state:")
            or key.startswith("margin_state:")
        },
        "negative_control_breakdown": {
            "dominant_control_counts": {
                key.removeprefix("dominant_negative_control:"): int(value)
                for key, value in sorted(counts.items())
                if key.startswith("dominant_negative_control:")
            },
            "delayed_positive_dominant_control_counts": {
                key.removeprefix(
                    "delayed_positive_dominant_negative_control:"
                ): int(value)
                for key, value in sorted(counts.items())
                if key.startswith("delayed_positive_dominant_negative_control:")
            },
        },
        "by_game": {
            game_id: dict(sorted(stats.items()))
            for game_id, stats in sorted(by_game.items())
        },
        "by_transition_key": {
            key: dict(sorted(stats.items()))
            for key, stats in sorted(by_transition_key.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "positive_control_margin_count > 0 with focused, breadth, and all25 non-regression",
            "allowed_connection": "diagnostic_only_until_delayed_causal_margin",
            "action_generation_allowed": False,
        },
    }


def _transition_state_variable_score_retest_admission_guard_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_TRANSITION_STATE_VARIABLE_SCORE_RETEST_PROBE")
    strategy_allowlist = [
        part.strip()
        for part in str(
            os.environ.get(
                "ECA_TRANSITION_STATE_VARIABLE_SCORE_RETEST_PROBE_STRATEGY_ALLOWLIST",
                "",
            )
            or ""
        ).split(",")
        if part.strip()
    ]
    source_action_allowlist = [
        part.strip().upper()
        for part in str(
            os.environ.get(
                "ECA_TRANSITION_STATE_VARIABLE_SCORE_RETEST_PROBE_SOURCE_ACTION_ALLOWLIST",
                "",
            )
            or ""
        ).split(",")
        if part.strip()
    ]
    transition_key_denylist = [
        part.strip()
        for part in str(
            os.environ.get(
                "ECA_TRANSITION_STATE_VARIABLE_SCORE_RETEST_PROBE_TRANSITION_KEY_DENYLIST",
                "",
            )
            or ""
        ).split(",")
        if part.strip()
    ]
    transition_key_denylist_signatures = sorted(
        {
            transition_state_variable_transition_key_signature(part)
            for part in transition_key_denylist
            if part
        }
    )
    transition_signature_allowlist = [
        part.strip()
        for part in str(
            os.environ.get(
                "ECA_TRANSITION_STATE_VARIABLE_SCORE_RETEST_PROBE_TRANSITION_SIGNATURE_ALLOWLIST",
                "",
            )
            or ""
        ).split(",")
        if part.strip()
    ]
    blocked_source_action_budget_active = _env_flag(
        "ECA_TRANSITION_STATE_VARIABLE_SCORE_RETEST_PROBE_COUNT_BLOCKED_SOURCE_ACTION_AS_BUDGET"
    )
    base: dict[str, Any] = {
        "schema_version": (
            "eca_transition_state_variable_score_retest_admission_guard_sensor.v1"
        ),
        "enabled": bool(enabled),
        "default_off": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "candidate_generation_enabled": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "strategy_allowlist": strategy_allowlist,
        "source_action_allowlist": source_action_allowlist,
        "transition_key_denylist": transition_key_denylist,
        "transition_key_denylist_signatures": transition_key_denylist_signatures,
        "transition_signature_allowlist": transition_signature_allowlist,
        "non_displacement_guard_active": bool(strategy_allowlist),
        "source_action_guard_active": bool(source_action_allowlist),
        "transition_key_denylist_active": bool(transition_key_denylist),
        "transition_signature_allowlist_active": bool(transition_signature_allowlist),
        "blocked_source_action_budget_active": bool(
            blocked_source_action_budget_active
        ),
        "preflight_count": 0,
        "admitted_count": 0,
        "execution_count": 0,
        "action5_score_retest_execution_count": 0,
        "repeat_source_action_retest_execution_count": 0,
        "blocked_by_strategy_allowlist_count": 0,
        "blocked_by_source_action_allowlist_count": 0,
        "blocked_by_transition_key_denylist_count": 0,
        "blocked_by_transition_signature_allowlist_count": 0,
        "blocked_source_action_counted_against_budget_count": 0,
        "promotion_state": "disabled" if not enabled else "no_score_retest_preflight",
    }
    if not enabled:
        base["disabled_reason"] = (
            "set ECA_ENABLE_TRANSITION_STATE_VARIABLE_SCORE_RETEST_PROBE=1"
        )
        return base

    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    schema_version = "eca_transition_state_variable_score_retest_probe.v1"
    diagnostic_reasons = {
        "retest_strategy_not_allowlisted",
        "source_action_not_allowlisted",
        "transition_key_denylisted",
        "transition_signature_not_allowlisted",
    }
    seen_preflights: set[tuple[str, int, str, str, str, str, bool]] = set()
    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        preflight_records: list[tuple[str, Mapping[str, Any], Mapping[str, Any], bool]] = []
        trace = row.get("trace")
        if isinstance(trace, list):
            for item in trace:
                if not isinstance(item, Mapping):
                    continue
                unsolved = item.get("unsolved_replan")
                unsolved = unsolved if isinstance(unsolved, Mapping) else {}
                if str(unsolved.get("schema_version", "") or "") == schema_version:
                    executed = bool(
                        unsolved.get("executed")
                        or unsolved.get("used")
                        or str(item.get("phase", "") or "")
                        == "transition_state_score_retest_probe"
                    )
                    preflight_records.append((game_id, item, unsolved, executed))
                nested = unsolved.get("transition_state_score_retest_probe_preflight")
                if isinstance(nested, Mapping) and (
                    str(nested.get("schema_version", "") or "") == schema_version
                    or bool(nested.get("enabled"))
                ):
                    nested_reason = str(nested.get("reason", "") or "")
                    if nested_reason in diagnostic_reasons or bool(
                        nested.get("admitted")
                    ):
                        preflight_records.append((game_id, item, nested, False))
        row_preflights = row.get(
            "transition_state_score_retest_preflight_diagnostics"
        )
        if isinstance(row_preflights, list):
            for raw in row_preflights:
                if not isinstance(raw, Mapping):
                    continue
                raw_reason = str(raw.get("reason", "") or "")
                if raw_reason not in diagnostic_reasons and not bool(
                    raw.get("admitted")
                ):
                    continue
                record_game_id = str(raw.get("game_id", "") or game_id)
                item = {
                    "step": raw.get("step", -1),
                    "phase": "transition_state_score_retest_probe_preflight",
                    "action": raw.get("selected_action")
                    or raw.get("blocked_retest_action")
                    or "",
                }
                preflight_records.append((record_game_id, item, raw, False))
        for record_game_id, item, unsolved, executed in preflight_records:
            reason = str(unsolved.get("reason", "") or "unknown")
            step = _as_int(item.get("step"), -1)
            candidate_strategy = str(
                unsolved.get("candidate_retest_strategy", "")
                or unsolved.get("blocked_retest_strategy", "")
                or ""
            )
            retest_strategy = str(unsolved.get("retest_strategy", "") or "")
            selected_action = str(
                unsolved.get("selected_action", "")
                or unsolved.get("blocked_retest_action", "")
                or item.get("action", "")
                or ""
            )
            source_action = str(unsolved.get("source_action_name", "") or "")
            transition_key = str(unsolved.get("source_transition_key", "") or "")
            transition_signature = (
                str(unsolved.get("source_transition_signature", "") or "")
                or (
                    transition_state_variable_transition_key_signature(transition_key)
                    if transition_key
                    else ""
                )
            )
            dedupe_key = (
                record_game_id,
                step,
                reason,
                transition_key,
                str(unsolved.get("blocked_retest_action", "") or ""),
                selected_action,
                bool(executed),
            )
            if dedupe_key in seen_preflights:
                continue
            seen_preflights.add(dedupe_key)
            counts["preflight_count"] += 1
            by_game[record_game_id]["preflight_count"] += 1
            counts[f"reason:{reason}"] += 1
            by_game[record_game_id][f"reason:{reason}"] += 1
            if candidate_strategy:
                counts[f"candidate_strategy:{candidate_strategy}"] += 1
                by_game[record_game_id][f"candidate_strategy:{candidate_strategy}"] += 1
            if transition_signature:
                counts[f"transition_signature:{transition_signature}"] += 1
                by_game[record_game_id][
                    f"transition_signature:{transition_signature}"
                ] += 1
            if reason == "retest_strategy_not_allowlisted":
                counts["blocked_by_strategy_allowlist_count"] += 1
                by_game[record_game_id]["blocked_by_strategy_allowlist_count"] += 1
                blocked = str(unsolved.get("blocked_retest_strategy", "") or "")
                if blocked:
                    counts[f"blocked_strategy:{blocked}"] += 1
                    by_game[record_game_id][f"blocked_strategy:{blocked}"] += 1
            if reason == "source_action_not_allowlisted":
                counts["blocked_by_source_action_allowlist_count"] += 1
                by_game[record_game_id]["blocked_by_source_action_allowlist_count"] += 1
                if bool(unsolved.get("blocked_source_action_counted_against_budget")):
                    counts[
                        "blocked_source_action_counted_against_budget_count"
                    ] += 1
                    by_game[record_game_id][
                        "blocked_source_action_counted_against_budget_count"
                    ] += 1
                blocked_action = str(unsolved.get("blocked_source_action", "") or "")
                if blocked_action:
                    counts[f"blocked_source_action:{blocked_action}"] += 1
                    by_game[record_game_id][f"blocked_source_action:{blocked_action}"] += 1
            if reason == "transition_key_denylisted":
                counts["blocked_by_transition_key_denylist_count"] += 1
                by_game[record_game_id][
                    "blocked_by_transition_key_denylist_count"
                ] += 1
                blocked_key = str(unsolved.get("blocked_transition_key", "") or "")
                if blocked_key:
                    counts[f"blocked_transition_key:{blocked_key}"] += 1
                    by_game[record_game_id][
                        f"blocked_transition_key:{blocked_key}"
                    ] += 1
                    blocked_signature = (
                        transition_state_variable_transition_key_signature(blocked_key)
                    )
                    counts[f"blocked_transition_signature:{blocked_signature}"] += 1
                    by_game[record_game_id][
                        f"blocked_transition_signature:{blocked_signature}"
                    ] += 1
            if reason == "transition_signature_not_allowlisted":
                counts["blocked_by_transition_signature_allowlist_count"] += 1
                by_game[record_game_id][
                    "blocked_by_transition_signature_allowlist_count"
                ] += 1
                blocked_signature = str(
                    unsolved.get("blocked_transition_signature", "") or ""
                )
                if not blocked_signature:
                    blocked_key = str(unsolved.get("blocked_transition_key", "") or "")
                    blocked_signature = (
                        transition_state_variable_transition_key_signature(blocked_key)
                        if blocked_key
                        else transition_signature
                    )
                if blocked_signature:
                    counts[f"blocked_transition_signature:{blocked_signature}"] += 1
                    by_game[record_game_id][
                        f"blocked_transition_signature:{blocked_signature}"
                    ] += 1
            if bool(unsolved.get("admitted")):
                counts["admitted_count"] += 1
                by_game[record_game_id]["admitted_count"] += 1
            if executed:
                counts["execution_count"] += 1
                by_game[record_game_id]["execution_count"] += 1
                if retest_strategy:
                    counts[f"execution_strategy:{retest_strategy}"] += 1
                    by_game[record_game_id][f"execution_strategy:{retest_strategy}"] += 1
                if selected_action:
                    counts[f"selected_action:{selected_action}"] += 1
                    by_game[record_game_id][f"selected_action:{selected_action}"] += 1
                if transition_signature:
                    counts[f"executed_transition_signature:{transition_signature}"] += 1
                    by_game[record_game_id][
                        f"executed_transition_signature:{transition_signature}"
                    ] += 1
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "game_id": record_game_id,
                        "step": step,
                        "phase": str(item.get("phase", "") or ""),
                        "reason": reason,
                        "candidate_retest_strategy": candidate_strategy,
                        "retest_strategy": retest_strategy,
                        "selected_action": selected_action,
                        "source_action_name": source_action,
                        "source_transition_key": transition_key,
                        "transition_signature": transition_signature,
                        "executed": bool(executed),
                    }
                )

    action5_count = int(counts.get("execution_strategy:action5_score_retest", 0))
    repeat_count = int(counts.get("execution_strategy:repeat_source_action_retest", 0))
    execution_count = int(counts.get("execution_count", 0))
    if int(counts.get("preflight_count", 0)) <= 0:
        promotion_state = "no_score_retest_preflight"
    elif action5_count > 0:
        promotion_state = "action5_retest_risk_observed"
    elif strategy_allowlist and repeat_count == execution_count and execution_count > 0:
        promotion_state = "repeat_source_only_non_displacing"
    elif execution_count > 0:
        promotion_state = "score_retest_execution_without_strategy_risk"
    else:
        promotion_state = "score_retest_preflight_no_execution"

    return {
        **base,
        "preflight_count": int(counts.get("preflight_count", 0)),
        "admitted_count": int(counts.get("admitted_count", 0)),
        "execution_count": execution_count,
        "action5_score_retest_execution_count": action5_count,
        "repeat_source_action_retest_execution_count": repeat_count,
        "blocked_by_strategy_allowlist_count": int(
            counts.get("blocked_by_strategy_allowlist_count", 0)
        ),
        "blocked_by_source_action_allowlist_count": int(
            counts.get("blocked_by_source_action_allowlist_count", 0)
        ),
        "blocked_by_transition_key_denylist_count": int(
            counts.get("blocked_by_transition_key_denylist_count", 0)
        ),
        "blocked_by_transition_signature_allowlist_count": int(
            counts.get("blocked_by_transition_signature_allowlist_count", 0)
        ),
        "blocked_source_action_counted_against_budget_count": int(
            counts.get("blocked_source_action_counted_against_budget_count", 0)
        ),
        "reason_counts": {
            key.split("reason:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("reason:")
        },
        "candidate_strategy_counts": {
            key.split("candidate_strategy:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("candidate_strategy:")
        },
        "execution_strategy_counts": {
            key.split("execution_strategy:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("execution_strategy:")
        },
        "selected_action_counts": {
            key.split("selected_action:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("selected_action:")
        },
        "transition_signature_counts": {
            key.split("transition_signature:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("transition_signature:")
        },
        "executed_transition_signature_counts": {
            key.split("executed_transition_signature:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("executed_transition_signature:")
        },
        "blocked_strategy_counts": {
            key.split("blocked_strategy:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("blocked_strategy:")
        },
        "blocked_source_action_counts": {
            key.split("blocked_source_action:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("blocked_source_action:")
        },
        "blocked_transition_key_counts": {
            key.split("blocked_transition_key:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("blocked_transition_key:")
        },
        "blocked_transition_signature_counts": {
            key.split("blocked_transition_signature:", 1)[1]: int(value)
            for key, value in sorted(counts.items())
            if key.startswith("blocked_transition_signature:")
        },
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "samples": samples,
        "promotion_state": promotion_state,
        "next_planner_contract": {
            "required_before_broad_execution": (
                "all25 non-regression and no action5_score_retest regression before "
                "allowing broad transition-state score retests"
            ),
            "allowed_connection": "diagnostic_to_guarded_probe_admission_only",
            "action_generation_allowed": False,
        },
    }


def _transition_state_variable_guarded_planner_connection_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_TRANSITION_STATE_VARIABLE_GUARDED_PLANNER_CONNECTION_SENSOR"
    )
    delayed = (
        _transition_state_variable_delayed_score_closure_attribution(rows)
        if enabled
        else {}
    )
    admission = (
        _transition_state_variable_score_retest_admission_guard_sensor(rows)
        if enabled
        else {}
    )
    summary = summarize_transition_state_variable_guarded_planner_connection(
        delayed,
        admission,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_TRANSITION_STATE_VARIABLE_GUARDED_PLANNER_CONNECTION_SENSOR=1"
        )
    return summary


def _route_validity_diagnostic_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_ROUTE_VALIDITY_DIAGNOSTIC_SENSOR")
    packets = _route_validity_packets_from_rows(rows) if enabled else []
    summary = summarize_route_validity_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = "set ECA_ENABLE_ROUTE_VALIDITY_DIAGNOSTIC_SENSOR=1"
    return summary


def _route_validity_progress_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_ROUTE_VALIDITY_PROGRESS_POSTERIOR")
    packets = _route_validity_packets_from_rows(rows) if enabled else []
    summary = summarize_route_validity_progress_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = "set ECA_ENABLE_ROUTE_VALIDITY_PROGRESS_POSTERIOR=1"
    return summary


def _route_validity_recovery_candidate_miner(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_ROUTE_VALIDITY_RECOVERY_CANDIDATE_MINER")
    packets = _route_validity_packets_from_rows(rows) if enabled else []
    summary = summarize_route_validity_recovery_candidates(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_ROUTE_VALIDITY_RECOVERY_CANDIDATE_MINER=1"
        )
    return summary


def _route_validity_all25_non_regression_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_ROUTE_VALIDITY_ALL25_NON_REGRESSION_SENSOR")
    expected_score_sum = _as_int(
        os.environ.get("ECA_ROUTE_VALIDITY_ALL25_REFERENCE_SCORE_SUM", ""),
        default=-1,
    )
    reference_by_game = _env_score_map(
        "ECA_ROUTE_VALIDITY_ALL25_REFERENCE_BY_GAME_JSON"
    )
    summary = summarize_route_validity_all25_non_regression(
        list(rows) if enabled else [],
        expected_score_sum=(
            None if expected_score_sum < 0 else int(expected_score_sum)
        ),
        expected_by_game=reference_by_game,
    )
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_ROUTE_VALIDITY_ALL25_NON_REGRESSION_SENSOR=1"
        )
    return summary


def _route_validity_recovery_probe_outcome_sensor(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_ROUTE_VALIDITY_RECOVERY_PROBE_OUTCOME")
    base: dict[str, Any] = {
        "schema_version": "eca_route_validity_recovery_probe_outcome_sensor.v1",
        "enabled": bool(enabled),
        "default_off": True,
        "diagnostic_only": True,
        "planner_connection_enabled": False,
        "runtime_execution_observed": False,
        "ranking_enabled": False,
        "positive_boost_enabled": False,
        "action_generation_allowed": False,
        "execution_count": 0,
        "score_delta_sum": 0,
        "route_progress_count": 0,
        "route_recovery_intervention_positive_count": 0,
        "promotion_state": "disabled" if not enabled else "no_route_recovery_execution",
    }
    if not enabled:
        base["disabled_reason"] = "set ECA_ENABLE_ROUTE_VALIDITY_RECOVERY_PROBE_OUTCOME=1"
        return base
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    by_source_action: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    score_delta_sum = 0
    source_score_delta_sum = 0
    route_progress_score_delta_sum = 0
    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            unsolved = item.get("unsolved_replan")
            unsolved = unsolved if isinstance(unsolved, Mapping) else {}
            preflight = unsolved.get("route_validity_recovery_probe_preflight")
            preflight = preflight if isinstance(preflight, Mapping) else {}
            if bool(preflight.get("enabled")):
                if bool(preflight.get("regression_guard_enabled")):
                    counts["regression_guard_preflight_count"] += 1
                blocked_actions = list(preflight.get("regression_guard_blocked_actions") or [])
                if blocked_actions:
                    counts["regression_guard_preflight_blocked_action_count"] += len(
                        blocked_actions
                    )
                    by_game[game_id]["regression_guard_preflight_blocked_action_count"] += len(
                        blocked_actions
                    )
                if (
                    str(preflight.get("reason", "") or "")
                    == "route_recovery_regression_guard_blocked_all"
                ):
                    counts["regression_guard_blocked_all_count"] += 1
                    by_game[game_id]["regression_guard_blocked_all_count"] += 1
            if (
                str(item.get("phase", "") or "") != "route_validity_recovery_probe"
                and str(unsolved.get("source", "") or "") != "route_validity_recovery_probe"
            ):
                continue
            if str(unsolved.get("candidate_kind", "") or "") != (
                "blocked_route_alternate_action_probe"
            ):
                continue
            counts["execution_count"] += 1
            by_game[game_id]["execution_count"] += 1
            source_action = str(unsolved.get("source_action_name", "") or "unknown")
            selected_action = str(unsolved.get("selected_action", "") or "unknown")
            by_source_action[source_action]["execution_count"] += 1
            packet = item.get("route_validity_packet")
            packet = packet if isinstance(packet, Mapping) else {}
            score_delta = _as_int(packet.get("score_delta"), 0)
            source_score_delta = _as_int(unsolved.get("source_score_delta"), 0)
            score_delta_sum += score_delta
            source_score_delta_sum += source_score_delta
            progress = bool(packet.get("route_progress_candidate"))
            regression = bool(packet.get("route_regression_candidate"))
            if score_delta > 0:
                counts["score_positive_count"] += 1
                by_game[game_id]["score_positive_count"] += 1
            if progress:
                counts["route_progress_count"] += 1
                by_game[game_id]["route_progress_count"] += 1
                route_progress_score_delta_sum += max(0, score_delta)
            if regression:
                counts["route_regression_count"] += 1
                by_game[game_id]["route_regression_count"] += 1
            intervention_positive = bool(score_delta > source_score_delta or progress)
            if intervention_positive:
                counts["route_recovery_intervention_positive_count"] += 1
                by_game[game_id]["route_recovery_intervention_positive_count"] += 1
            runtime = unsolved.get("runtime_outcome")
            runtime = runtime if isinstance(runtime, Mapping) else {}
            if bool(runtime.get("no_op")):
                counts["runtime_no_op_count"] += 1
            if bool(unsolved.get("regression_guard_enabled")):
                counts["regression_guard_execution_count"] += 1
            blocked_actions = list(unsolved.get("regression_guard_blocked_actions") or [])
            if blocked_actions:
                counts["regression_guard_blocked_action_count"] += len(blocked_actions)
                by_game[game_id]["regression_guard_blocked_action_count"] += len(
                    blocked_actions
                )
            if bool(unsolved.get("positive_history_ordering_enabled")) and list(
                unsolved.get("positive_history_actions") or []
            ):
                counts["positive_history_ordered_execution_count"] += 1
            if len(samples) < sample_limit:
                delta = packet.get("route_topology_delta")
                delta = delta if isinstance(delta, Mapping) else {}
                samples.append(
                    {
                        "game_id": game_id,
                        "step": _as_int(item.get("step"), -1),
                        "source_action_name": source_action,
                        "selected_action": selected_action,
                        "score_delta": score_delta,
                        "source_score_delta": source_score_delta,
                        "route_progress_candidate": progress,
                        "route_regression_candidate": regression,
                        "largest_area_delta": _as_int(
                            delta.get("largest_passable_region_area_delta"), 0
                        ),
                        "passable_region_count_delta": _as_int(
                            delta.get("passable_region_count_delta"), 0
                        ),
                        "runtime_effect_type": str(runtime.get("effect_type", "") or ""),
                        "runtime_no_op": bool(runtime.get("no_op", False)),
                        "regression_guard_blocked_actions": list(
                            unsolved.get("regression_guard_blocked_actions") or []
                        ),
                        "positive_history_actions": list(
                            unsolved.get("positive_history_actions") or []
                        ),
                    }
                )
    margin = int(score_delta_sum - source_score_delta_sum)
    positive = int(counts.get("route_recovery_intervention_positive_count", 0))
    if positive > 0 and margin >= 0:
        promotion_state = "route_recovery_intervention_positive"
    elif int(counts.get("execution_count", 0)) > 0:
        promotion_state = "route_recovery_executed_no_positive_margin"
    else:
        promotion_state = "no_route_recovery_execution"
    return {
        **base,
        "runtime_execution_observed": int(counts.get("execution_count", 0)) > 0,
        "execution_count": int(counts.get("execution_count", 0)),
        "score_delta_sum": int(score_delta_sum),
        "source_blocked_score_delta_sum": int(source_score_delta_sum),
        "score_delta_margin_vs_source_blocked": int(margin),
        "score_positive_count": int(counts.get("score_positive_count", 0)),
        "route_progress_count": int(counts.get("route_progress_count", 0)),
        "route_regression_count": int(counts.get("route_regression_count", 0)),
        "route_progress_score_delta_sum": int(route_progress_score_delta_sum),
        "route_recovery_intervention_positive_count": positive,
        "runtime_no_op_count": int(counts.get("runtime_no_op_count", 0)),
        "regression_guard_execution_count": int(
            counts.get("regression_guard_execution_count", 0)
        ),
        "regression_guard_blocked_action_count": int(
            counts.get("regression_guard_blocked_action_count", 0)
        ),
        "regression_guard_preflight_count": int(
            counts.get("regression_guard_preflight_count", 0)
        ),
        "regression_guard_preflight_blocked_action_count": int(
            counts.get("regression_guard_preflight_blocked_action_count", 0)
        ),
        "regression_guard_blocked_all_count": int(
            counts.get("regression_guard_blocked_all_count", 0)
        ),
        "positive_history_ordered_execution_count": int(
            counts.get("positive_history_ordered_execution_count", 0)
        ),
        "promotion_state": promotion_state,
        "by_game": {
            game_id: dict(sorted(counter.items()))
            for game_id, counter in sorted(by_game.items())
        },
        "by_source_action": {
            action: dict(sorted(counter.items()))
            for action, counter in sorted(by_source_action.items())
        },
        "samples": samples,
        "next_planner_contract": {
            "required_before_ranking": "positive route recovery intervention replicated with focused, breadth, and all25 non-regression",
            "allowed_connection": "diagnostic_only_until_replicated_intervention_margin",
            "action_generation_allowed": False,
        },
    }


def _multi_body_contact_topology_diagnostic_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MULTI_BODY_CONTACT_TOPOLOGY_DIAGNOSTIC_SENSOR")
    packets = _multi_body_contact_topology_packets_from_rows(rows) if enabled else []
    summary = summarize_multi_body_contact_topology_packets(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MULTI_BODY_CONTACT_TOPOLOGY_DIAGNOSTIC_SENSOR=1"
        )
    return summary


def _multi_body_contact_topology_posterior(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MULTI_BODY_CONTACT_TOPOLOGY_POSTERIOR")
    packets = _multi_body_contact_topology_packets_from_rows(rows) if enabled else []
    summary = summarize_multi_body_contact_topology_posterior(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MULTI_BODY_CONTACT_TOPOLOGY_POSTERIOR=1"
        )
    return summary


def _multi_body_contact_topology_candidate_miner(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag("ECA_ENABLE_MULTI_BODY_CONTACT_TOPOLOGY_CANDIDATE_MINER")
    packets = _multi_body_contact_topology_packets_from_rows(rows) if enabled else []
    summary = summarize_multi_body_contact_topology_candidates(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MULTI_BODY_CONTACT_TOPOLOGY_CANDIDATE_MINER=1"
        )
    return summary


def _multi_body_contact_topology_guarded_probe_outcome_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_MULTI_BODY_CONTACT_TOPOLOGY_GUARDED_PROBE_OUTCOME_SENSOR"
    )
    packets = _multi_body_contact_topology_packets_from_rows(rows) if enabled else []
    summary = summarize_multi_body_contact_topology_guarded_probe_outcomes(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MULTI_BODY_CONTACT_TOPOLOGY_GUARDED_PROBE_OUTCOME_SENSOR=1"
        )
    return summary


def _multi_body_contact_topology_counterfactual_control_sensor(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    enabled = _env_flag(
        "ECA_ENABLE_MULTI_BODY_CONTACT_TOPOLOGY_COUNTERFACTUAL_CONTROL_SENSOR"
    )
    packets = _multi_body_contact_topology_packets_from_rows(rows) if enabled else []
    summary = summarize_multi_body_contact_topology_counterfactual_controls(packets)
    summary["enabled"] = bool(enabled)
    summary["default_off"] = True
    if not enabled:
        summary["disabled_reason"] = (
            "set ECA_ENABLE_MULTI_BODY_CONTACT_TOPOLOGY_COUNTERFACTUAL_CONTROL_SENSOR=1"
        )
    return summary


def _transition_state_variable_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            packet = item.get("transition_state_variable_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("transition_state_variable_packet")
            if not isinstance(packet, Mapping):
                continue
            if (
                str(packet.get("schema_version", "") or "")
                != TRANSITION_STATE_VARIABLE_PACKET_SCHEMA_VERSION
            ):
                continue
            packets.append(packet)
    return packets


def _moving_sprite_runtime_trace_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            packet = item.get("moving_sprite_runtime_trace_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("moving_sprite_runtime_trace_packet")
            if not isinstance(packet, Mapping):
                continue
            if (
                str(packet.get("schema_version", "") or "")
                != MOVING_SPRITE_RUNTIME_TRACE_PACKET_SCHEMA_VERSION
            ):
                continue
            packets.append(packet)
    return packets


def _mechanic_certificate_runtime_trace_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            packet = item.get("mechanic_certificate_runtime_trace_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("mechanic_certificate_runtime_trace_packet")
            if not isinstance(packet, Mapping):
                continue
            if (
                str(packet.get("schema_version", "") or "")
                != MECHANIC_CERTIFICATE_RUNTIME_TRACE_PACKET_SCHEMA_VERSION
            ):
                continue
            packets.append(packet)
    return packets


def _palette_invariant_runtime_trace_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            packet = item.get("palette_invariant_runtime_trace_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("palette_invariant_runtime_trace_packet")
            if not isinstance(packet, Mapping):
                continue
            if (
                str(packet.get("schema_version", "") or "")
                != PALETTE_INVARIANT_RUNTIME_TRACE_PACKET_SCHEMA_VERSION
            ):
                continue
            packets.append(packet)
    return packets


def _tile_effect_runtime_trace_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    packets: list[Mapping[str, Any]] = []
    fallback_moving_packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            packet = item.get("tile_effect_runtime_trace_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("tile_effect_runtime_trace_packet")
            if isinstance(packet, Mapping) and (
                str(packet.get("schema_version", "") or "")
                == TILE_EFFECT_RUNTIME_TRACE_PACKET_SCHEMA_VERSION
            ):
                packets.append(packet)
                continue
            moving_packet = item.get("moving_sprite_runtime_trace_packet")
            if not isinstance(moving_packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                moving_packet = diagnostics.get("moving_sprite_runtime_trace_packet")
            if isinstance(moving_packet, Mapping) and (
                str(moving_packet.get("schema_version", "") or "")
                == MOVING_SPRITE_RUNTIME_TRACE_PACKET_SCHEMA_VERSION
            ):
                fallback_moving_packets.append(moving_packet)
    return packets if packets else fallback_moving_packets


def _tile_effect_model_repair_runtime_bridge_events_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            event = item.get("tile_effect_model_repair_runtime_bridge_event")
            if not isinstance(event, Mapping):
                event = item.get(
                    "tile_effect_model_repair_protected_probe_runtime_bridge_event"
                )
            if not isinstance(event, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                event = diagnostics.get("tile_effect_model_repair_runtime_bridge_event")
            if not isinstance(event, Mapping):
                continue
            events.append(event)
    return events


def _tile_effect_model_repair_runtime_probe_outcomes_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    outcomes: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            outcome_list = item.get("tile_effect_model_repair_runtime_probe_outcomes")
            if isinstance(outcome_list, list):
                outcomes.extend(
                    outcome
                    for outcome in outcome_list
                    if isinstance(outcome, Mapping)
                )
                continue
            outcome = item.get("tile_effect_model_repair_runtime_probe_outcome")
            if not isinstance(outcome, Mapping):
                outcome = item.get(
                    "tile_effect_model_repair_protected_probe_runtime_outcome_event"
                )
            if not isinstance(outcome, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                outcome = diagnostics.get(
                    "tile_effect_model_repair_runtime_probe_outcome"
                )
            if not isinstance(outcome, Mapping):
                continue
            outcomes.append(outcome)
    return outcomes


def _tile_effect_model_repair_live_rank_guard_focused_runtime_events_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            event = item.get(
                "tile_effect_model_repair_live_rank_guard_focused_runtime"
            )
            if not isinstance(event, Mapping):
                event = item.get(
                    "tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime_event"
                )
            if not isinstance(event, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                event = diagnostics.get(
                    "tile_effect_model_repair_live_rank_guard_focused_runtime"
                )
            if not isinstance(event, Mapping):
                continue
            events.append(event)
    return events


def _tile_effect_positive_action_selection_score_attribution_events_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    window: int = 8,
) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        variant = str(row.get("variant", "") or "unknown")
        final_score = _as_int(row.get("final_score"), 0)
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            event = item.get(
                "tile_effect_model_repair_live_rank_guard_focused_runtime"
            )
            if not isinstance(event, Mapping):
                event = item.get(
                    "tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime_event"
                )
            if not isinstance(event, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                event = diagnostics.get(
                    "tile_effect_model_repair_live_rank_guard_focused_runtime"
                )
            if not isinstance(event, Mapping):
                continue

            before_action = str(event.get("selected_action_before", "") or "")
            after_action = str(event.get("selected_action_after", "") or "")
            unsafe = (
                bool(event.get("action_generation_allowed"))
                or bool(event.get("model_repair_write_enabled"))
                or bool(event.get("generated_new_candidate"))
            )
            positive = bool(
                event.get("action_selection_change_requested")
                and event.get("action_selection_guard_admitted")
                and event.get("action_selection_changed")
                and event.get("selected_existing_candidate_only")
                and before_action
                and before_action == after_action
                and not unsafe
            )
            if not positive:
                continue

            current_score = _trace_score_before(item, final_score)
            if index + 1 < len(trace):
                next_score = _trace_score_before(trace[index + 1], current_score)
            else:
                next_score = final_score
            immediate_delta = max(0, int(next_score) - int(current_score))
            delayed_delta, delay_steps = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=index,
                window=window,
                default_score=current_score,
                final_score=final_score,
            )
            shifted_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=min(index + 1, max(0, len(trace) - 1)),
                window=window,
                default_score=next_score,
                final_score=final_score,
            )
            pre_action_delta = _pre_action_score_delta(
                trace,
                index=index,
                window=window,
                current_score=current_score,
            )
            control_index = _deterministic_control_index(index, len(trace))
            random_delta, _ = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=control_index,
                window=window,
                default_score=_trace_score_before(trace[control_index], current_score)
                if trace
                else current_score,
                final_score=final_score,
            )
            control_max = max(shifted_delta, pre_action_delta, random_delta)
            enriched = {
                **dict(event),
                "game_id": str(event.get("game_id", "") or game_id),
                "variant": variant,
                "step": _as_int(event.get("step"), index),
                "phase": str(item.get("phase", "") or ""),
                "primitive": str(item.get("primitive", "") or ""),
                "action": str(item.get("action", "") or ""),
                "positive_action_selection_admitted": True,
                "same_action_family": True,
                "score_before": int(current_score),
                "immediate_score_delta": int(immediate_delta),
                "delayed_score_delta": int(delayed_delta),
                "delay_steps": int(delay_steps),
                "shifted_delayed_score_delta": int(shifted_delta),
                "pre_action_score_delta": int(pre_action_delta),
                "random_step_score_delta": int(random_delta),
                "negative_control_max_score_delta": int(control_max),
                "join_margin": int(delayed_delta) - int(control_max),
            }
            events.append(enriched)
    return events


def _route_validity_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            packet = item.get("route_validity_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("route_validity_packet")
            if not isinstance(packet, Mapping):
                continue
            if (
                str(packet.get("schema_version", "") or "")
                != ROUTE_VALIDITY_PACKET_SCHEMA_VERSION
            ):
                continue
            packets.append(packet)
    return packets


def _multi_body_contact_topology_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            packet = item.get("multi_body_contact_topology_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("multi_body_contact_topology_packet")
            if not isinstance(packet, Mapping):
                continue
            if (
                str(packet.get("schema_version", "") or "")
                != MULTI_BODY_CONTACT_TOPOLOGY_PACKET_SCHEMA_VERSION
            ):
                continue
            packets.append(packet)
    return packets


def _slot_pool_coverage_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            packet = item.get("slot_pool_coverage_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("slot_pool_coverage_packet")
            if not isinstance(packet, Mapping):
                continue
            if (
                str(packet.get("schema_version", "") or "")
                != SLOT_POOL_COVERAGE_PACKET_SCHEMA_VERSION
            ):
                continue
            packets.append(packet)
    return packets


def _slot_pool_coverage_posterior_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    posterior_packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        previous_packet: Mapping[str, Any] | None = None
        previous_item: Mapping[str, Any] | None = None
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            explicit = item.get("slot_pool_coverage_posterior_packet")
            if isinstance(explicit, Mapping) and str(
                explicit.get("schema_version", "") or ""
            ) == SLOT_POOL_COVERAGE_POSTERIOR_PACKET_SCHEMA_VERSION:
                posterior_packets.append(explicit)
            packet = item.get("slot_pool_coverage_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("slot_pool_coverage_packet")
            if not isinstance(packet, Mapping):
                continue
            if (
                str(packet.get("schema_version", "") or "")
                != SLOT_POOL_COVERAGE_PACKET_SCHEMA_VERSION
            ):
                continue
            if previous_packet is not None:
                posterior_packets.append(
                    build_slot_pool_coverage_posterior_packet(
                        previous_packet,
                        packet,
                        score_delta=_trace_item_score_delta(item),
                        plan_completed=bool(item.get("slot_pool_plan_completed", False)),
                        action_name=str(item.get("action", "") or ""),
                        action_payload=_trace_item_action_payload(item),
                    )
                )
            previous_packet = packet
            previous_item = item
        if previous_item is None:
            continue
    return posterior_packets


def _slot_pool_coverage_focused_replay_execution_events_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            event = item.get("slot_pool_coverage_focused_replay_execution_event")
            if not isinstance(event, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                event = diagnostics.get(
                    "slot_pool_coverage_focused_replay_execution_event"
                )
            if not isinstance(event, Mapping):
                continue
            events.append(event)
    return events


def _slot_pool_coverage_submit_precondition_runtime_bridge_events_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            event = item.get(
                "slot_pool_coverage_submit_precondition_runtime_bridge_guard_event"
            )
            if not isinstance(event, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                event = diagnostics.get(
                    "slot_pool_coverage_submit_precondition_runtime_bridge_guard_event"
                )
            if not isinstance(event, Mapping):
                continue
            events.append(event)
    return events


def _slot_pool_coverage_submit_precondition_rank_guard_runtime_events_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            event = item.get(
                "slot_pool_coverage_submit_precondition_rank_guard_runtime_event"
            )
            if not isinstance(event, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                event = diagnostics.get(
                    "slot_pool_coverage_submit_precondition_rank_guard_runtime_event"
                )
            if not isinstance(event, Mapping):
                continue
            events.append(event)
    return events


def _trace_item_score_delta(item: Mapping[str, Any]) -> int:
    if "score_delta" in item:
        return _as_int(item.get("score_delta"), 0)
    unsolved = item.get("unsolved_replan")
    unsolved = unsolved if isinstance(unsolved, Mapping) else {}
    runtime = unsolved.get("runtime_outcome")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    observed = runtime.get("observed_effect")
    observed = observed if isinstance(observed, Mapping) else {}
    if "score_delta" in observed:
        return _as_int(observed.get("score_delta"), 0)
    if "score_delta" in runtime:
        return _as_int(runtime.get("score_delta"), 0)
    return 0


def _trace_item_action_payload(item: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = item.get("action_payload")
    if isinstance(payload, Mapping):
        return payload
    adapter = item.get("adapter_advice")
    adapter = adapter if isinstance(adapter, Mapping) else {}
    executed = adapter.get("executed_payload")
    executed = executed if isinstance(executed, Mapping) else {}
    if executed:
        return executed
    control_target = item.get("control_target")
    return control_target if isinstance(control_target, Mapping) else {}


def _force_field_merge_containment_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            packet = item.get("force_field_merge_containment_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("force_field_merge_containment_packet")
            if not isinstance(packet, Mapping):
                continue
            if (
                str(packet.get("schema_version", "") or "")
                != FORCE_FIELD_MERGE_CONTAINMENT_PACKET_SCHEMA_VERSION
            ):
                continue
            packets.append(packet)
    return packets


def _program_row_semantics_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            packet = item.get("program_row_semantics_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("program_row_semantics_packet")
            if not isinstance(packet, Mapping):
                continue
            if (
                str(packet.get("schema_version", "") or "")
                != PROGRAM_ROW_SEMANTICS_PACKET_SCHEMA_VERSION
            ):
                continue
            packets.append(packet)
    return packets


def _program_spell_pattern_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            packet = item.get("program_spell_pattern_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("program_spell_pattern_packet")
            if not isinstance(packet, Mapping):
                continue
            if (
                str(packet.get("schema_version", "") or "")
                != PROGRAM_SPELL_PATTERN_PACKET_SCHEMA_VERSION
            ):
                continue
            packets.append(packet)
    return packets


def _platform_destructible_topology_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            packet = item.get("platform_destructible_topology_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("platform_destructible_topology_packet")
            if not isinstance(packet, Mapping):
                continue
            if (
                str(packet.get("schema_version", "") or "")
                != PLATFORM_DESTRUCTIBLE_TOPOLOGY_PACKET_SCHEMA_VERSION
            ):
                continue
            packets.append(packet)
    return packets


def _component_extent_growth_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            packet = item.get("component_extent_growth_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("component_extent_growth_packet")
            if not isinstance(packet, Mapping):
                continue
            if (
                str(packet.get("schema_version", "") or "")
                != COMPONENT_EXTENT_GROWTH_PACKET_SCHEMA_VERSION
            ):
                continue
            packets.append(packet)
    return packets


def _component_extent_growth_target_distance_rank_guard_events_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    score_window: int = 40,
) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    window = max(1, int(score_window))
    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        variant = str(row.get("variant", "") or "unknown")
        final_score = _as_int(row.get("final_score"), 0)
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for index, item in enumerate(trace):
            if not isinstance(item, Mapping):
                continue
            event = item.get("component_extent_growth_target_distance_rank_guard")
            if not isinstance(event, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                event = diagnostics.get(
                    "component_extent_growth_target_distance_rank_guard"
                )
            if not isinstance(event, Mapping):
                continue

            packet = item.get("component_extent_growth_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("component_extent_growth_packet")
            observations = (
                list(packet.get("target_distance_observations") or [])
                if isinstance(packet, Mapping)
                else []
            )
            improved = [
                obs
                for obs in observations
                if isinstance(obs, Mapping) and bool(obs.get("distance_improved"))
            ]
            worsened = [
                obs
                for obs in observations
                if isinstance(obs, Mapping)
                and str(obs.get("distance_relation", "") or "") == "worsened"
            ]
            current_score = _trace_score_before(item, final_score)
            delayed_delta, delay_steps = _window_score_delta_and_delay_with_final_score(
                trace,
                baseline_index=index,
                window=window,
                default_score=current_score,
                final_score=final_score,
            )
            events.append(
                {
                    **dict(event),
                    "game_id": game_id,
                    "variant": variant,
                    "phase": str(item.get("phase", "") or ""),
                    "step": _as_int(item.get("step"), index),
                    "trace_index": int(index),
                    "primitive": str(item.get("primitive", "") or ""),
                    "action": str(item.get("action", "") or ""),
                    "score_before": int(current_score),
                    "final_score": int(final_score),
                    "target_distance_observation_after_rank_guard_count": int(
                        len(observations)
                    ),
                    "target_distance_improved_after_rank_guard_count": int(
                        len(improved)
                    ),
                    "target_distance_worsened_after_rank_guard_count": int(
                        len(worsened)
                    ),
                    "delayed_score_delta": int(delayed_delta),
                    "delay_steps": int(delay_steps),
                }
            )
    return events


def _round_memory_packets_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    packets: list[Mapping[str, Any]] = []
    for row in rows:
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            packet = item.get("round_memory_packet")
            if not isinstance(packet, Mapping):
                diagnostics = item.get("diagnostics")
                diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
                packet = diagnostics.get("round_memory_packet")
            if not isinstance(packet, Mapping):
                continue
            if (
                str(packet.get("schema_version", "") or "")
                != ROUND_MEMORY_PACKET_SCHEMA_VERSION
            ):
                continue
            packets.append(packet)
    return packets


def _generalisation_contract(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    unsolved_to_replan_summary = _unsolved_to_replan_summary(row_list)
    score_relation_shadow_signature_outcome_join = (
        _score_relation_shadow_signature_outcome_join_sensor(row_list)
    )
    score_relation_shadow_context_signature_outcome_join = (
        _score_relation_shadow_context_signature_outcome_join_sensor(row_list)
    )
    score_sum = sum(_as_int(row.get("final_score"), 0) for row in row_list)
    strict_rows = [row for row in row_list if bool(row.get("strict_generic_ablation"))]
    game_conditioned_rows = [row for row in row_list if _uses_game_conditioned_primitive(row)]
    adapter_free_rows = [row for row in row_list if _is_adapter_free_row(row)]
    game_conditioned_score = sum(
        _as_int(row.get("final_score"), 0) for row in game_conditioned_rows
    )
    adapter_free_score = sum(_as_int(row.get("final_score"), 0) for row in adapter_free_rows)
    strict_score = sum(_as_int(row.get("final_score"), 0) for row in strict_rows)

    game_conditioned_games = sorted(
        {
            str(row.get("game_id", "") or "unknown")
            for row in game_conditioned_rows
        }
    )
    adapter_free_games = sorted(
        {
            str(row.get("game_id", "") or "unknown")
            for row in adapter_free_rows
        }
    )
    return {
        "schema_version": "eca_generalisation_contract.v1",
        "score_sum": score_sum,
        "strict_generic_floor_score_sum": strict_score,
        "strict_generic_floor_run_count": len(strict_rows),
        "adapter_free_score_sum": adapter_free_score,
        "adapter_free_score_share": _safe_div(adapter_free_score, score_sum),
        "game_conditioned_score_sum": game_conditioned_score,
        "game_conditioned_score_share": _safe_div(game_conditioned_score, score_sum),
        "game_conditioned_run_share": _safe_div(len(game_conditioned_rows), len(row_list)),
        "game_conditioned_games": game_conditioned_games,
        "adapter_free_games": adapter_free_games,
        "primitive_usage": _primitive_usage_summary(row_list),
        "primitive_scope_counts": primitive_scope_counts(
            primitive_id
            for row in row_list
            for primitive_id in (
                row.get("primitive_ids") if isinstance(row.get("primitive_ids"), list) else []
            )
        ),
        "primitive_scope_records": primitive_scope_records(
            sorted(
                {
                    str(primitive_id or "").strip()
                    for row in row_list
                    for primitive_id in (
                        row.get("primitive_ids") if isinstance(row.get("primitive_ids"), list) else []
                    )
                    if str(primitive_id or "").strip()
                }
            )
        ),
        "failure_signatures": _failure_signatures(row_list),
        "operator_failure_signatures": _operator_failure_signatures(row_list),
        "a_axis_sensor_gap_inventory_sensor": _a_axis_sensor_gap_inventory_sensor(
            row_list
        ),
        "a_axis_sensor_promotion_gap_inventory_sensor": _a_axis_sensor_promotion_gap_inventory_sensor(
            row_list
        ),
        "parallel_sensor_substrate_gap_inventory_sensor": _parallel_sensor_substrate_gap_inventory_sensor(
            row_list
        ),
        "moving_sprite_runtime_trace_sensor": _moving_sprite_runtime_trace_sensor(
            row_list
        ),
        "mechanic_certificate_runtime_trace_sensor": _mechanic_certificate_runtime_trace_sensor(
            row_list
        ),
        "mechanic_certificate_posterior": _mechanic_certificate_posterior(
            row_list
        ),
        "mechanic_certificate_guarded_planner_connection": _mechanic_certificate_guarded_planner_connection(
            row_list
        ),
        "palette_invariant_runtime_trace_sensor": _palette_invariant_runtime_trace_sensor(
            row_list
        ),
        "palette_invariant_perception_posterior": _palette_invariant_perception_posterior(
            row_list
        ),
        "palette_invariant_guarded_planner_connection": _palette_invariant_guarded_planner_connection(
            row_list
        ),
        "actor_identity_anchor_sensor": _actor_identity_anchor_sensor(row_list),
        "actor_identity_anchor_posterior": _actor_identity_anchor_posterior(
            row_list
        ),
        "actor_identity_anchor_guarded_planner_connection": _actor_identity_anchor_guarded_planner_connection(
            row_list
        ),
        "transport_effect_runtime_trace_sensor": _transport_effect_runtime_trace_sensor(
            row_list
        ),
        "tile_effect_runtime_trace_sensor": _tile_effect_runtime_trace_sensor(
            row_list
        ),
        "tile_effect_posterior": _tile_effect_posterior(row_list),
        "tile_effect_guarded_planner_connection": _tile_effect_guarded_planner_connection(
            row_list
        ),
        "tile_effect_guard_precision_review": _tile_effect_guard_precision_review(
            row_list
        ),
        "tile_effect_guard_counterfactual_control": _tile_effect_guard_counterfactual_control(
            row_list
        ),
        "tile_effect_family_scope_negative_control": _tile_effect_family_scope_negative_control(
            row_list
        ),
        "tile_effect_hazard_source_repair_bridge": _tile_effect_hazard_source_repair_bridge(
            row_list
        ),
        "tile_effect_model_repair_proposal": _tile_effect_model_repair_proposal(
            row_list
        ),
        "tile_effect_model_repair_write_guard": _tile_effect_model_repair_write_guard(
            row_list
        ),
        "tile_effect_model_repair_write_audit": _tile_effect_model_repair_write_audit(
            row_list
        ),
        "tile_effect_model_repair_runtime_probe_proposal": _tile_effect_model_repair_runtime_probe_proposal(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_execution_gate": _tile_effect_model_repair_protected_probe_execution_gate(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_runtime_bridge_guard": _tile_effect_model_repair_protected_probe_runtime_bridge_guard(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_runtime_executor": _tile_effect_model_repair_protected_probe_runtime_executor(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_runtime_outcome_execution": _tile_effect_model_repair_protected_probe_runtime_outcome_execution(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_runtime_bridge_outcome": _tile_effect_model_repair_protected_probe_runtime_bridge_outcome(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_outcome_posterior": _tile_effect_model_repair_protected_probe_outcome_posterior(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_rank_guard_shadow": _tile_effect_model_repair_protected_probe_rank_guard_shadow(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_rank_guard_runtime_guard": _tile_effect_model_repair_protected_probe_rank_guard_runtime_guard(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime": _tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_live_rank_guard_action_selection_guard": _tile_effect_model_repair_protected_probe_live_rank_guard_action_selection_guard(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_admission": _tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_admission(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution": _tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_positive_rank_promotion_guard": _tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_positive_rank_promotion_guard(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery": _tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery(
            row_list
        ),
        "tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_focused_runtime_replay": _tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_focused_runtime_replay(
            row_list
        ),
        "transport_effect_posterior": _transport_effect_posterior(row_list),
        "transport_effect_guarded_planner_connection": _transport_effect_guarded_planner_connection(
            row_list
        ),
        "invisible_effect_position_memory_diagnostic_sensor": _invisible_effect_position_memory_diagnostic_sensor(
            row_list
        ),
        "invisible_effect_position_memory_posterior": _invisible_effect_position_memory_posterior(
            row_list
        ),
        "invisible_effect_position_memory_guarded_planner_connection": _invisible_effect_position_memory_guarded_planner_connection(
            row_list
        ),
        "moving_sprite_motion_posterior": _moving_sprite_motion_posterior(row_list),
        "moving_sprite_guarded_planner_readiness": _moving_sprite_guarded_planner_readiness(
            row_list
        ),
        "moving_sprite_actor_ambiguity_splitter": _moving_sprite_actor_ambiguity_splitter(
            row_list
        ),
        "moving_sprite_guarded_actor_anchor_posterior": _moving_sprite_guarded_actor_anchor_posterior(
            row_list
        ),
        "moving_sprite_soft_anchor_motion_posterior": _moving_sprite_soft_anchor_motion_posterior(
            row_list
        ),
        "moving_sprite_coupled_motion_symmetry_sensor": _moving_sprite_coupled_motion_symmetry_sensor(
            row_list
        ),
        "moving_sprite_coupled_pair_goal_effect_bridge": _moving_sprite_coupled_pair_goal_effect_bridge(
            row_list
        ),
        "moving_sprite_coupled_pair_goal_score_attribution_guard": _moving_sprite_coupled_pair_goal_score_attribution_guard(
            row_list
        ),
        "moving_sprite_target_goal_counterfactual_attribution": _moving_sprite_target_goal_counterfactual_attribution(
            row_list
        ),
        "moving_sprite_target_bound_effect_component_sensor": _moving_sprite_target_bound_effect_component_sensor(
            row_list
        ),
        "moving_sprite_effect_component_spatial_stability_sensor": _moving_sprite_effect_component_spatial_stability_sensor(
            row_list
        ),
        "moving_sprite_sparse_chain_rail_state_sensor": _moving_sprite_sparse_chain_rail_state_sensor(
            row_list
        ),
        "moving_sprite_target_goal_binding_state_sensor": _moving_sprite_target_goal_binding_state_sensor(
            row_list
        ),
        "moving_sprite_goal_anchor_perception_sensor": _moving_sprite_goal_anchor_perception_sensor(
            row_list
        ),
        "moving_sprite_guarded_planner_connection_sensor": _moving_sprite_guarded_planner_connection_sensor(
            row_list
        ),
        "slot_pool_coverage_diagnostic_sensor": _slot_pool_coverage_diagnostic_sensor(
            row_list
        ),
        "slot_pool_coverage_posterior": _slot_pool_coverage_posterior(row_list),
        "slot_pool_coverage_candidate_miner": _slot_pool_coverage_candidate_miner(
            row_list
        ),
        "slot_pool_coverage_guarded_probe_outcome_sensor": _slot_pool_coverage_guarded_probe_outcome_sensor(
            row_list
        ),
        "slot_pool_coverage_counterfactual_control_sensor": _slot_pool_coverage_counterfactual_control_sensor(
            row_list
        ),
        "slot_pool_coverage_scoreless_progress_control_sensor": _slot_pool_coverage_scoreless_progress_control_sensor(
            row_list
        ),
        "slot_pool_coverage_scoped_scoreless_control_sensor": _slot_pool_coverage_scoped_scoreless_control_sensor(
            row_list
        ),
        "slot_pool_coverage_target_slot_scoreless_control_sensor": _slot_pool_coverage_target_slot_scoreless_control_sensor(
            row_list
        ),
        "slot_pool_coverage_exact_slot_scoreless_control_sensor": _slot_pool_coverage_exact_slot_scoreless_control_sensor(
            row_list
        ),
        "slot_pool_coverage_region_identity_control_sensor": _slot_pool_coverage_region_identity_control_sensor(
            row_list
        ),
        "slot_pool_coverage_effect_intent_control_sensor": _slot_pool_coverage_effect_intent_control_sensor(
            row_list
        ),
        "slot_pool_coverage_terminal_effect_score_closure_attribution_sensor": _slot_pool_coverage_terminal_effect_score_closure_attribution_sensor(
            row_list
        ),
        "slot_pool_coverage_score_event_scope_attribution_sensor": _slot_pool_coverage_score_event_scope_attribution_sensor(
            row_list
        ),
        "slot_pool_coverage_intervention_contrast_attribution_sensor": _slot_pool_coverage_intervention_contrast_attribution_sensor(
            row_list
        ),
        "slot_pool_coverage_intervention_temporal_counterfactual_sensor": _slot_pool_coverage_intervention_temporal_counterfactual_sensor(
            row_list
        ),
        "slot_pool_coverage_already_scored_event_exclusion_sensor": _slot_pool_coverage_already_scored_event_exclusion_sensor(
            row_list
        ),
        "slot_pool_coverage_negative_memory_guarded_review_sensor": _slot_pool_coverage_negative_memory_guarded_review_sensor(
            row_list
        ),
        "slot_pool_coverage_negative_memory_ranking_shadow_sensor": _slot_pool_coverage_negative_memory_ranking_shadow_sensor(
            row_list
        ),
        "slot_pool_coverage_focused_replay_readiness_sensor": _slot_pool_coverage_focused_replay_readiness_sensor(
            row_list
        ),
        "slot_pool_coverage_focused_replay_execution_sensor": _slot_pool_coverage_focused_replay_execution_sensor(
            row_list
        ),
        "slot_pool_coverage_focused_replay_outcome_admission_guard_sensor": _slot_pool_coverage_focused_replay_outcome_admission_guard_sensor(
            row_list
        ),
        "slot_pool_coverage_focused_score_bearing_candidate_discovery_sensor": _slot_pool_coverage_focused_score_bearing_candidate_discovery_sensor(
            row_list
        ),
        "slot_pool_coverage_submit_precondition_score_sensor": _slot_pool_coverage_submit_precondition_score_sensor(
            row_list
        ),
        "slot_pool_coverage_submit_precondition_runtime_bridge_guard_sensor": _slot_pool_coverage_submit_precondition_runtime_bridge_guard_sensor(
            row_list
        ),
        "slot_pool_coverage_submit_precondition_runtime_bridge_outcome_sensor": _slot_pool_coverage_submit_precondition_runtime_bridge_outcome_sensor(
            row_list
        ),
        "slot_pool_coverage_submit_precondition_breadth_non_regression_sensor": _slot_pool_coverage_submit_precondition_breadth_non_regression_sensor(
            row_list
        ),
        "slot_pool_coverage_submit_precondition_rank_guard_shadow_sensor": _slot_pool_coverage_submit_precondition_rank_guard_shadow_sensor(
            row_list
        ),
        "slot_pool_coverage_submit_precondition_rank_guard_runtime_guard_sensor": _slot_pool_coverage_submit_precondition_rank_guard_runtime_guard_sensor(
            row_list
        ),
        "slot_pool_coverage_submit_precondition_live_rank_guard_focused_runtime_sensor": _slot_pool_coverage_submit_precondition_live_rank_guard_focused_runtime_sensor(
            row_list
        ),
        "force_field_merge_containment_diagnostic_sensor": _force_field_merge_containment_diagnostic_sensor(
            row_list
        ),
        "force_field_merge_containment_posterior": _force_field_merge_containment_posterior(
            row_list
        ),
        "force_field_merge_containment_candidate_miner": _force_field_merge_containment_candidate_miner(
            row_list
        ),
        "force_field_merge_containment_guarded_probe_outcome_sensor": _force_field_merge_containment_guarded_probe_outcome_sensor(
            row_list
        ),
        "force_field_merge_containment_counterfactual_control_sensor": _force_field_merge_containment_counterfactual_control_sensor(
            row_list
        ),
        "force_field_merge_containment_all25_non_regression_sensor": _force_field_merge_containment_all25_non_regression_sensor(
            row_list
        ),
        "force_field_merge_containment_score_target_gap_sensor": _force_field_merge_containment_score_target_gap_sensor(
            row_list
        ),
        "force_field_containment_score_target_relation_sensor": _force_field_containment_score_target_relation_sensor(
            row_list
        ),
        "force_field_score_positive_counterfactual_pair_search_candidate_miner": _force_field_score_positive_counterfactual_pair_search_candidate_miner(
            row_list
        ),
        "program_row_semantics_diagnostic_sensor": _program_row_semantics_diagnostic_sensor(
            row_list
        ),
        "program_row_semantics_posterior": _program_row_semantics_posterior(
            row_list
        ),
        "program_row_semantics_candidate_miner": _program_row_semantics_candidate_miner(
            row_list
        ),
        "program_row_semantics_guarded_probe_outcome_sensor": _program_row_semantics_guarded_probe_outcome_sensor(
            row_list
        ),
        "program_row_semantics_counterfactual_control_sensor": _program_row_semantics_counterfactual_control_sensor(
            row_list
        ),
        "program_row_semantics_all25_non_regression_sensor": _program_row_semantics_all25_non_regression_sensor(
            row_list
        ),
        "program_spell_pattern_diagnostic_sensor": _program_spell_pattern_diagnostic_sensor(
            row_list
        ),
        "program_spell_pattern_posterior": _program_spell_pattern_posterior(
            row_list
        ),
        "program_spell_pattern_candidate_miner": _program_spell_pattern_candidate_miner(
            row_list
        ),
        "program_spell_pattern_guarded_probe_outcome_sensor": _program_spell_pattern_guarded_probe_outcome_sensor(
            row_list
        ),
        "program_spell_pattern_counterfactual_control_sensor": _program_spell_pattern_counterfactual_control_sensor(
            row_list
        ),
        "program_spell_pattern_all25_non_regression_sensor": _program_spell_pattern_all25_non_regression_sensor(
            row_list
        ),
        "kinematic_membership_candidate_miner_sensor": _kinematic_membership_candidate_miner_sensor(
            row_list
        ),
        "kinematic_membership_counterfactual_control_sensor": _kinematic_membership_counterfactual_control_sensor(
            row_list
        ),
        "kinematic_membership_all25_non_regression_sensor": _kinematic_membership_all25_non_regression_sensor(
            row_list
        ),
        "platform_destructible_topology_diagnostic_sensor": _platform_destructible_topology_diagnostic_sensor(
            row_list
        ),
        "platform_destructible_topology_posterior": _platform_destructible_topology_posterior(
            row_list
        ),
        "platform_destructible_topology_candidate_miner": _platform_destructible_topology_candidate_miner(
            row_list
        ),
        "platform_destructible_topology_guarded_probe_outcome_sensor": _platform_destructible_topology_guarded_probe_outcome_sensor(
            row_list
        ),
        "platform_destructible_topology_counterfactual_control_sensor": _platform_destructible_topology_counterfactual_control_sensor(
            row_list
        ),
        "platform_destructible_topology_all25_non_regression_sensor": _platform_destructible_topology_all25_non_regression_sensor(
            row_list
        ),
        "component_extent_growth_diagnostic_sensor": _component_extent_growth_diagnostic_sensor(
            row_list
        ),
        "component_extent_growth_precision_posterior": _component_extent_growth_precision_posterior(
            row_list
        ),
        "component_extent_growth_slider_stride_posterior": _component_extent_growth_slider_stride_posterior(
            row_list
        ),
        "component_extent_growth_target_distance_posterior": _component_extent_growth_target_distance_posterior(
            row_list
        ),
        "component_extent_growth_target_distance_rank_guard_focused_runtime": _component_extent_growth_target_distance_rank_guard_focused_runtime(
            row_list
        ),
        "round_memory_diagnostic_sensor": _round_memory_diagnostic_sensor(row_list),
        "round_memory_posterior": _round_memory_posterior(row_list),
        "round_memory_guarded_probe_outcome_sensor": (
            _round_memory_guarded_probe_outcome_sensor(row_list)
        ),
        "transition_state_variable_diagnostic_sensor": _transition_state_variable_diagnostic_sensor(
            row_list
        ),
        "transition_state_variable_score_closure_posterior": _transition_state_variable_score_closure_posterior(
            row_list
        ),
        "transition_state_variable_score_retest_candidate_miner": _transition_state_variable_score_retest_candidate_miner(
            row_list
        ),
        "transition_state_variable_delayed_score_closure_attribution": _transition_state_variable_delayed_score_closure_attribution(
            row_list
        ),
        "transition_state_variable_score_retest_admission_guard_sensor": _transition_state_variable_score_retest_admission_guard_sensor(
            row_list
        ),
        "transition_state_variable_guarded_planner_connection_sensor": _transition_state_variable_guarded_planner_connection_sensor(
            row_list
        ),
        "route_validity_diagnostic_sensor": _route_validity_diagnostic_sensor(
            row_list
        ),
        "route_validity_progress_posterior": _route_validity_progress_posterior(
            row_list
        ),
        "route_validity_recovery_candidate_miner": _route_validity_recovery_candidate_miner(
            row_list
        ),
        "route_validity_all25_non_regression_sensor": _route_validity_all25_non_regression_sensor(
            row_list
        ),
        "route_validity_recovery_probe_outcome_sensor": _route_validity_recovery_probe_outcome_sensor(
            row_list
        ),
        "multi_body_contact_topology_diagnostic_sensor": _multi_body_contact_topology_diagnostic_sensor(
            row_list
        ),
        "multi_body_contact_topology_posterior": _multi_body_contact_topology_posterior(
            row_list
        ),
        "multi_body_contact_topology_candidate_miner": _multi_body_contact_topology_candidate_miner(
            row_list
        ),
        "multi_body_contact_topology_guarded_probe_outcome_sensor": _multi_body_contact_topology_guarded_probe_outcome_sensor(
            row_list
        ),
        "multi_body_contact_topology_counterfactual_control_sensor": _multi_body_contact_topology_counterfactual_control_sensor(
            row_list
        ),
        "progress_to_score_causal_closure_sensor": _progress_to_score_causal_closure_sensor(
            row_list
        ),
        "local_progress_closure_verifier_outcome_posterior": _local_progress_closure_verifier_outcome_posterior(
            row_list
        ),
        "score_relation_candidate_verifier_outcome_posterior": _score_relation_candidate_verifier_outcome_posterior(
            row_list
        ),
        "score_relation_falsified_target_action_signature_memory_sensor": _score_relation_falsified_target_action_signature_memory_sensor(
            row_list
        ),
        "score_relation_family_scope_negative_memory_sensor": _score_relation_family_scope_negative_memory_sensor(
            row_list
        ),
        "score_relation_site_effect_intervention_delayed_outcome_sensor": _score_relation_site_effect_intervention_delayed_outcome_sensor(
            row_list
        ),
        "score_relation_site_effect_target_action_signature_promotion_guard_sensor": _score_relation_site_effect_target_action_signature_promotion_guard_sensor(
            row_list
        ),
        "score_relation_site_effect_selector_shadow_sensor": _score_relation_site_effect_selector_shadow_sensor(
            row_list
        ),
        "score_relation_progress_goal_binding_candidate_mining_sensor": _score_relation_progress_goal_binding_candidate_mining_sensor(
            row_list
        ),
        "score_relation_progress_goal_binding_objective_binding_sensor": _score_relation_progress_goal_binding_objective_binding_sensor(
            row_list
        ),
        "score_relation_progress_goal_binding_action_effect_direction_sensor": _score_relation_progress_goal_binding_action_effect_direction_sensor(
            row_list
        ),
        "score_relation_progress_goal_binding_admission_guard_sensor": _score_relation_progress_goal_binding_admission_guard_sensor(
            row_list
        ),
        "score_relation_progress_goal_binding_control_margin_sensor": _score_relation_progress_goal_binding_control_margin_sensor(
            row_list
        ),
        "score_relation_progress_goal_binding_paired_control_margin_sensor": _score_relation_progress_goal_binding_paired_control_margin_sensor(
            row_list
        ),
        "score_relation_progress_goal_binding_target_instance_consistency_sensor": _score_relation_progress_goal_binding_target_instance_consistency_sensor(
            row_list
        ),
        "score_relation_progress_goal_binding_temporal_readiness_sensor": _score_relation_progress_goal_binding_temporal_readiness_sensor(
            row_list
        ),
        "score_relation_progress_goal_binding_score_closure_attribution_sensor": _score_relation_progress_goal_binding_score_closure_attribution_sensor(
            row_list
        ),
        "score_relation_remote_status_runtime_guard_outcome_sensor": _score_relation_remote_status_runtime_guard_outcome_sensor(
            row_list
        ),
        "score_relation_shadow_evidence_lane_sensor": _score_relation_shadow_evidence_lane_sensor(
            row_list
        ),
        "score_relation_shadow_verifier_schema_translation_sensor": _score_relation_shadow_verifier_schema_translation_sensor(
            row_list
        ),
        "score_relation_shadow_evidence_outcome_join_sensor": _score_relation_shadow_evidence_outcome_join_sensor(
            row_list
        ),
        "score_relation_shadow_signature_outcome_join_sensor": score_relation_shadow_signature_outcome_join,
        "score_relation_shadow_context_signature_outcome_join_sensor": score_relation_shadow_context_signature_outcome_join,
        "score_relation_shadow_context_signature_credit_split_sensor": _score_relation_shadow_context_signature_credit_split_sensor(
            score_relation_shadow_context_signature_outcome_join
        ),
        "score_relation_shadow_signature_credit_split_sensor": _score_relation_shadow_signature_credit_split_sensor(
            score_relation_shadow_signature_outcome_join
        ),
        "score_relation_terminal_condition_gap_sensor": _score_relation_terminal_condition_gap_sensor(
            row_list
        ),
        "score_relation_terminal_requirement_sensor": _score_relation_terminal_requirement_sensor(
            row_list
        ),
        "score_relation_objective_target_binding_sensor": _score_relation_objective_target_binding_sensor(
            row_list
        ),
        "score_relation_terminal_readiness_sensor": _score_relation_terminal_readiness_sensor(
            row_list
        ),
        "score_relation_bounded_repeat_continuation_outcome_sensor": _score_relation_bounded_repeat_continuation_outcome_sensor(
            row_list
        ),
        "score_relation_post_closure_transition_sensor": _score_relation_post_closure_transition_sensor(
            row_list
        ),
        "score_relation_post_closure_state_delta_interpreter_sensor": _score_relation_post_closure_state_delta_interpreter_sensor(
            row_list
        ),
        "score_relation_changed_region_followup_relation_sensor": _score_relation_changed_region_followup_relation_sensor(
            row_list
        ),
        "score_relation_remote_changed_region_semantics_sensor": _score_relation_remote_changed_region_semantics_sensor(
            row_list
        ),
        "score_relation_terminal_status_predicate_sensor": _score_relation_terminal_status_predicate_sensor(
            row_list
        ),
        "score_relation_remote_status_transition_posterior": _score_relation_remote_status_transition_posterior(
            row_list
        ),
        "score_relation_remote_status_negative_rank_guard": _score_relation_remote_status_negative_rank_guard(
            row_list
        ),
        "click_site_discovery": _click_discovery_summary(row_list),
        "click_affordance_induction": _click_affordance_induction_summary(row_list),
        "solver_family_budget": _solver_family_budget_summary(row_list),
        "goal_target_contract": _goal_target_contract_summary(row_list),
        "adapter_agreement": _adapter_agreement_summary(row_list),
        "adapter_plan_cache": _adapter_plan_cache_summary(row_list),
        "unsolved_to_replan": unsolved_to_replan_summary,
        "adapter_advice_sequence_replay": _adapter_advice_sequence_replay_audit(row_list),
        "adapter_advice_execution_replay": _adapter_advice_execution_replay_audit(row_list),
        "adapter_advice_score_attribution": _adapter_advice_score_attribution(row_list),
        "outcome_source_attribution": _outcome_source_attribution_summary(
            row_list,
            unsolved_to_replan=unsolved_to_replan_summary,
        ),
        "adapter_objective_alignment": _adapter_objective_alignment_summary(row_list),
        "adapter_disagreement_replay": _adapter_disagreement_replay(row_list),
        "structural_rods": _structural_rod_summary(row_list),
        "emergent_rod_pattern_mining_sensor": summarize_emergent_rod_pattern_mining(
            row_list
        ),
        "rod_selector_bias": _rod_selector_bias_summary(row_list),
        "rod_action_binding": _rod_action_binding_summary(row_list),
        "rod_target_binding": _rod_target_binding_summary(row_list),
        "rod_score_precondition_action": _rod_score_precondition_action_summary(row_list),
        "rod_score_precondition_context": _rod_score_precondition_context_summary(row_list),
        "canonical_instability_games": _canonical_instability_games(row_list),
    }


def summarise_perturbation_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    grouped_by_tier: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    grouped_by_game: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    all_rows = [dict(row) for row in rows]
    _annotate_canonical_floor_recovery(all_rows)
    for row in all_rows:
        grouped[str(row.get("variant", "") or "unknown")].append(row)
        grouped_by_tier[_generalisation_tier_name(row)].append(row)
        grouped_by_game[str(row.get("game_id", "") or "unknown")].append(row)

    by_variant: dict[str, dict[str, Any]] = {}
    for variant, items in sorted(grouped.items()):
        by_variant[variant] = _summarise_group(items)

    by_generalisation_tier = {
        tier: _summarise_group(items) for tier, items in sorted(grouped_by_tier.items())
    }
    by_game = {
        game_id: _summarise_group(items) for game_id, items in sorted(grouped_by_game.items())
    }
    generalisation_contract = _generalisation_contract(all_rows)
    generalisation_contract["a_axis_sensor_promotion_blocker_inventory_sensor"] = (
        build_sensor_promotion_blocker_inventory(
            generalisation_contract,
            stage_inventory=generalisation_contract.get(
                "a_axis_sensor_promotion_gap_inventory_sensor"
            ),
        )
    )

    total_perturbations = sum(1 for row in all_rows if bool(row.get("perturbation_applied")))
    total_recovered = sum(1 for row in all_rows if bool(row.get("recovered_after_perturbation")))
    total_floor_recovered = sum(1 for row in all_rows if bool(row.get("canonical_floor_recovered")))
    total_positive_floor_perturbations = sum(
        1
        for row in all_rows
        if bool(row.get("perturbation_applied"))
        and _as_int(row.get("canonical_floor_score"), 0) > 0
    )
    total_positive_floor_recovered = sum(
        1 for row in all_rows if bool(row.get("positive_canonical_floor_recovered"))
    )
    tier_counts = Counter(_generalisation_tier_name(row) for row in all_rows)
    return {
        "schema_version": "eca_perception_perturbation_summary.v1",
        "run_count": len(all_rows),
        "perturbation_count": total_perturbations,
        "ok_count": sum(1 for row in all_rows if row.get("status") == "ok"),
        "error_count": sum(1 for row in all_rows if row.get("status") != "ok"),
        "score_sum": sum(_as_int(row.get("final_score"), 0) for row in all_rows),
        "recovered_count": total_recovered,
        "recovery_rate": _safe_div(total_recovered, total_perturbations),
        "canonical_floor_recovered_count": total_floor_recovered,
        "canonical_floor_recovery_rate": _safe_div(total_floor_recovered, total_perturbations),
        "positive_canonical_floor_perturbation_count": total_positive_floor_perturbations,
        "positive_canonical_floor_recovered_count": total_positive_floor_recovered,
        "positive_canonical_floor_recovery_rate": _safe_div(
            total_positive_floor_recovered,
            total_positive_floor_perturbations,
        ),
        "frame_only_failure_count": sum(1 for row in all_rows if not bool(row.get("frame_only"))),
        "strict_generic_ablation_count": sum(
            1 for row in all_rows if bool(row.get("strict_generic_ablation"))
        ),
        "generalisation_tier_counts": dict(sorted(tier_counts.items())),
        "uses_game_conditioned_primitive_count": sum(
            1 for row in all_rows if _uses_game_conditioned_primitive(row)
        ),
        "adapter_free_count": sum(1 for row in all_rows if _is_adapter_free_row(row)),
        "adapter_free_score_sum": sum(
            _as_int(row.get("final_score"), 0)
            for row in all_rows
            if _is_adapter_free_row(row)
        ),
        "uses_game_conditioned_primitive_score_sum": sum(
            _as_int(row.get("final_score"), 0)
            for row in all_rows
            if _uses_game_conditioned_primitive(row)
        ),
        "by_variant": by_variant,
        "by_game": by_game,
        "by_generalisation_tier": by_generalisation_tier,
        "generalisation_contract": generalisation_contract,
    }


def run_perception_perturbation_benchmark(
    games: Iterable[str],
    *,
    variants: Iterable[str] = DEFAULT_VARIANTS,
    seeds: Iterable[int] = (0,),
    prefix_lengths: Iterable[int] = (1, 3),
    injection_step: int = 1,
    max_actions: int = 80,
) -> dict[str, Any]:
    specs = build_perturbation_specs(
        variants,
        seeds=seeds,
        prefix_lengths=prefix_lengths,
        injection_step=injection_step,
    )
    rows: list[dict[str, Any]] = []
    for game_id in games:
        game_id = str(game_id or "").strip()
        if not game_id:
            continue
        if game_id not in PERCEPTION_ADAPTER_REGISTRY:
            rows.append(
                {
                    "schema_version": "eca_perception_perturbation_episode.v1",
                    "game_id": game_id,
                    "variant": "registry_lookup",
                    "status": "error",
                    "errors": [f"no perception adapter registered for {game_id}"],
                    "frame_only": True,
                    "actions_taken": 0,
                    "final_score": 0,
                    "final_state": "",
                    "perturbation_score": 0,
                    "perturbation_applied": False,
                    "recovery_score_delta": 0,
                    "recovered_after_perturbation": False,
                    "trace": [],
                }
            )
            continue
        for spec in specs:
            rows.append(
                run_perception_perturbation_episode(
                    game_id,
                    spec,
                    max_actions=max_actions,
                )
            )
    _annotate_canonical_floor_recovery(rows)
    return {
        "schema_version": "eca_perception_perturbation_benchmark.v1",
        "description": "real environment perturbation benchmark for frame-conditioned perception adapters",
        "games": list(games),
        "variants": list(variants),
        "seeds": list(seeds),
        "prefix_lengths": list(prefix_lengths),
        "injection_step": injection_step,
        "max_actions": max_actions,
        "summary": summarise_perturbation_rows(rows),
        "rows": rows,
    }
