"""Local Controller: Orchestrate IR consumption and Analyzer fallback.

This is the top-level orchestrator that ties together:
    ControlIRCompiler → CandidateResolver → MicroPlanner → action queue

The Local Controller is called once per step from the harness/runtime.
It returns either a locally-planned action queue or a signal to fall back
to the Analyzer LLM.

Flow:
    env.step
     -> ECA.observe
     -> LocalController.step(runtime, step_index)
        -> ControlIRCompiler.compile(...)
        -> CandidateResolver.resolve(...)
        -> MicroPlanner.plan(...)
        -> if queue available: return LocalControllerResult(actions=...)
        -> else: return LocalControllerResult(needs_analyzer=True)

The harness checks `result.needs_analyzer` and only calls the Analyzer
when True.  This is the key mechanism for reducing LLM calls.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def _l1_quota_keyboard_click_enabled() -> bool:
    """Phase 8.B flag: let the L1 click-dominance quota fire even when
    move actions are available, IF the recent action window did not
    actually use them. Default off — preserves prior bail behaviour.

    Off: keep the legacy "move_family_available -> bail" gate.
    On:  bail only when the agent has emitted >= K move actions in the
         recent local-action window (i.e. rotation is actually
         happening). Targets the Group A `keyboard_click` games where
         keyboard is in the action surface but the controller still
         emits 100% click.
    """
    raw = os.environ.get("ENABLE_L1_QUOTA_KEYBOARD_CLICK", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}

from arcgym.utils.grid_utils import _ASCII_PALETTE
from arcgym.eca.control_ir import ControlIRConfig, ControlIREvidence
from arcgym.eca.control_ir_compiler import (
    CompiledIR,
    CompilationResult,
    ControlIRCompiler,
)
from arcgym.eca.candidate_resolver import (
    CandidateResolver,
    ResolverConfig,
    ResolutionResult,
    ScoredCandidate,
)
from arcgym.eca.candidate_extractor import StructuredCandidate
from arcgym.eca.micro_planner import (
    MicroPlanner,
    MicroPlannerConfig,
    MicroPlanResult,
    PlannedAction,
)
from arcgym.eca.escalation_manager import (
    EscalationManager,
    EscalationManagerConfig,
    EscalationDecision,
)
from arcgym.eca.hidden_state_executor import HiddenStateExecutor, HiddenStateExecutorLease
from arcgym.semantics.api import AdapterExecutionTicket

log = logging.getLogger(__name__)

_MOVE_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
_MOVE_ACTION_SET = set(_MOVE_ACTIONS)
_CLICK_ACTION = "ACTION6"
_ASCII_GRID_VALUE_BY_CHAR = {
    _ASCII_PALETTE[min(int((value / 16) * (len(_ASCII_PALETTE) - 1)), len(_ASCII_PALETTE) - 1)]: value
    for value in range(16)
}
_L1_LATERAL_CONTOUR_ACTIONS = (
    "ACTION3", "ACTION3",
    "ACTION1", "ACTION1", "ACTION1", "ACTION1",
    "ACTION4", "ACTION4", "ACTION4",
    "ACTION1", "ACTION1", "ACTION1", "ACTION1",
)
_GENERIC_KEYBOARD_ACTION_RE = re.compile(r"^ACTION(\d+)$")
_SEMANTIC_BRANCH_ENTRY_EVENTS = {
    "goal_slot_blocked",
    "state_mismatch",
    "state_changer_reachable",
    "state_transition_site_candidate",
}
_HIDDEN_STATE_ABORT_RECOVERY_REASONS = {
    "invalid_phase_transition",
    "stall_budget_exhausted",
    "ticket_canceled",
    "ttl_exhausted",
}
_NO_SCORE_EXHAUSTED_TARGET_DROP_REASONS = {
    "fresh_target_minimal_probe_exhausted",
    "target_click_probe_exhausted",
    "target_family_probe_exhausted",
    "target_move_probe_exhausted",
}


def _infer_auxiliary_action_name(*, action_name: Any = "", component_id: Any = "") -> str:
    normalized = str(action_name or "").strip().upper()
    if _GENERIC_KEYBOARD_ACTION_RE.match(normalized):
        if normalized not in _MOVE_ACTION_SET and normalized not in {_CLICK_ACTION, "RESET"}:
            return normalized
        return ""
    component = str(component_id or "").strip()
    match = re.search(r"(?:^|::)(?:ACTION)?(\d+)$", component, re.IGNORECASE)
    if not match:
        return ""
    inferred = f"ACTION{int(match.group(1))}"
    if inferred in _MOVE_ACTION_SET or inferred in {_CLICK_ACTION, "RESET"}:
        return ""
    return inferred
# ─── Controller Configuration ──────────────────────────────────────────────

@dataclass
class LocalControllerConfig:
    """Configuration for the Local Controller."""
    enabled: bool = True
    # ControlIR config (shared with compiler)
    control_ir_config: ControlIRConfig | None = None
    # Resolver config
    resolver_config: ResolverConfig | None = None
    # MicroPlanner config
    micro_planner_config: MicroPlannerConfig | None = None
    # Maximum consecutive local failures before forcing analyzer
    max_consecutive_local_failures: int = 2
    # EscalationManager config
    escalation_config: EscalationManagerConfig | None = None
    # Whether to log detailed step metrics
    verbose_metrics: bool = False

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "LocalControllerConfig":
        if not raw:
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", True)),
            control_ir_config=ControlIRConfig.from_mapping(raw.get("control_ir")),
            resolver_config=ResolverConfig.from_mapping(raw.get("resolver")),
            micro_planner_config=MicroPlannerConfig.from_mapping(raw.get("micro_planner")),
            escalation_config=EscalationManagerConfig.from_mapping(raw.get("escalation")),
            max_consecutive_local_failures=max(
                1, int(raw.get("max_consecutive_local_failures", 2))
            ),
            verbose_metrics=bool(raw.get("verbose_metrics", False)),
        )


# ─── Controller Result ─────────────────────────────────────────────────────

@dataclass
class LocalControllerResult:
    """Result of a single Local Controller step."""
    # Whether the controller produced a local action queue
    locally_planned: bool = False
    # The planned actions (empty if needs_analyzer)
    actions: list[PlannedAction] = field(default_factory=list)
    # Whether the Analyzer should be called
    needs_analyzer: bool = False
    # Reason for analyzer fallback
    analyzer_reason: str = ""
    # Compilation result (for metrics)
    compilation: CompilationResult | None = None
    # Resolution result (for metrics)
    resolution: ResolutionResult | None = None
    # MicroPlan result (for metrics)
    micro_plan: MicroPlanResult | None = None
    # Selected target info (for runtime to fill selected_target)
    selected_target: ScoredCandidate | None = None
    # Compact analyzer input (for sparse replanner mode)
    sparse_analyzer_input: dict[str, Any] | None = None
    # EscalationManager decision (Phase 2)
    escalation_decision: "EscalationDecision | None" = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "locally_planned": self.locally_planned,
            "action_count": len(self.actions),
            "needs_analyzer": self.needs_analyzer,
            "analyzer_reason": self.analyzer_reason,
        }
        if self.compilation:
            d["compilation"] = self.compilation.to_dict()
        if self.resolution:
            d["resolution"] = self.resolution.to_dict()
        if self.micro_plan:
            d["micro_plan"] = self.micro_plan.to_dict()
        if self.selected_target:
            d["selected_target"] = self.selected_target.to_dict()
        if self.sparse_analyzer_input:
            d["sparse_analyzer_input"] = self.sparse_analyzer_input
        if self.escalation_decision:
            d["escalation"] = self.escalation_decision.to_dict()
        return d


# ─── Local Controller ──────────────────────────────────────────────────────

class LocalController:
    """Top-level orchestrator for ControlIR-first runtime.

    Usage in harness.py:
        controller = LocalController(config)
        # Each step:
        result = controller.step(runtime, step_index, ...)
        if result.locally_planned:
            agent.set_action_plan(result.actions)
        elif result.needs_analyzer:
            # Call Analyzer with result.sparse_analyzer_input
            analyzer_output = call_analyzer(result.sparse_analyzer_input)
            agent.set_action_plan(analyzer_output)
    """

    def __init__(self, config: LocalControllerConfig | None = None) -> None:
        self.config = config or LocalControllerConfig()

        # Initialize sub-components
        self._compiler = ControlIRCompiler(self.config.control_ir_config)
        self._resolver = CandidateResolver(self.config.resolver_config)
        self._planner = MicroPlanner(self.config.micro_planner_config)
        self._escalation_manager = EscalationManager(self.config.escalation_config)
        self._hidden_state_executor = HiddenStateExecutor()

        # Tracking
        self._consecutive_local_failures: int = 0
        self._total_local_steps: int = 0
        self._total_base_local_steps: int = 0
        self._total_ticket_assisted_local_steps: int = 0
        self._total_semantic_assisted_local_steps: int = 0
        self._total_local_accepted_steps: int = 0
        self._total_local_rejected_steps: int = 0
        self._total_analyzer_calls: int = 0
        self._pending_local_plan_ack: bool = False
        self._step_history: list[dict[str, Any]] = []
        self._semantic_recommendation_turn_count: int = 0
        self._semantic_recommended_action_match_count: int = 0
        self._semantic_recommended_action_executed_count: int = 0
        self._semantic_branch_entry_attempt_count: int = 0
        self._semantic_branch_entry_success_count: int = 0
        self._semantic_branch_internal_progress_turns: int = 0
        self._semantic_branch_to_site_distance_min: int | None = None
        self._semantic_site_reach_attempt_count: int = 0
        self._semantic_site_reach_success_count: int = 0
        self._semantic_branch_exit_back_to_corridor_count: int = 0
        self._semantic_site_reaching_enter_count: int = 0
        self._semantic_site_reaching_exit_count: int = 0
        self._semantic_site_reaching_total_duration: int = 0
        self._semantic_site_actuation_enter_count: int = 0
        self._semantic_site_actuation_attempt_count: int = 0
        self._semantic_site_actuation_success_count: int = 0
        self._semantic_site_actuation_sequence_issue_count: int = 0
        self._semantic_site_actuation_sequence_consume_count: int = 0
        self._semantic_site_actuation_contact_success_count: int = 0
        self._semantic_site_actuation_display_change_count: int = 0
        self._semantic_site_actuation_no_contact_count: int = 0
        self._semantic_site_actuation_state_change_success_count: int = 0
        self._semantic_site_actuation_verify_success_count: int = 0
        self._semantic_site_exit_before_actuation_count: int = 0
        self._semantic_site_exit_before_verify_count: int = 0
        self._semantic_site_actuation_sequence_histogram: dict[str, int] = {}
        self._semantic_site_actuation_display_change_sequence_histogram: dict[str, int] = {}
        self._semantic_site_actuation_no_display_change_sequence_histogram: dict[str, int] = {}
        self._semantic_site_actuation_no_contact_sequence_histogram: dict[str, int] = {}
        self._semantic_site_actuation_contact_position_histogram: dict[str, int] = {}
        self._semantic_site_actuation_contact_pair_histogram: dict[str, int] = {}
        self._semantic_site_actuation_followup_window_count: int = 0
        self._semantic_site_actuation_followup_stamp_change_count: int = 0
        self._semantic_site_actuation_followup_no_change_count: int = 0
        self._semantic_site_actuation_followup_change_pair_histogram: dict[str, int] = {}
        self._semantic_site_actuation_followup_no_change_pair_histogram: dict[str, int] = {}
        self._semantic_site_actuation_reused_sequence_count: int = 0
        self._semantic_site_actuation_sequence_queue_load_count: int = 0
        self._semantic_site_actuation_stale_tail_queue_skip_count: int = 0
        self._semantic_site_actuation_sequence_execute_success_count: int = 0
        self._semantic_site_actuation_sequence_abort_count: int = 0
        self._semantic_site_actuation_sequence_abort_reason_histogram: dict[str, int] = {}
        self._semantic_site_actuation_far_tail_turn_count: int = 0
        self._semantic_site_actuation_far_tail_sequence_histogram: dict[str, int] = {}
        self._semantic_site_actuation_far_tail_site_histogram: dict[str, int] = {}
        self._semantic_site_actuation_reacquire_candidate_count: int = 0
        self._semantic_site_actuation_reacquire_count: int = 0
        self._semantic_site_actuation_controller_reset_count: int = 0
        self._semantic_site_actuation_controller_blocked_sequence_histogram: dict[str, int] = {}
        self._semantic_site_actuation_controller_blocked_sequence_count: int = 0
        self._semantic_site_actuation_controller_far_tail_turn_count: int = 0
        self._semantic_site_actuation_controller_far_tail_sequence_histogram: dict[str, int] = {}
        self._semantic_site_actuation_controller_reacquire_issue_count: int = 0
        self._semantic_site_actuation_controller_reacquire_cooldown_remaining: int = 0
        self._semantic_site_actuation_controller_reacquire_hold_remaining: int = 0
        self._semantic_site_actuation_controller_reacquire_last_distance: int | None = None
        self._semantic_site_actuation_controller_reacquire_last_action: str = ""
        self._semantic_site_actuation_controller_reacquire_lateral_recovery_used: bool = False
        self._semantic_site_actuation_controller_reacquire_blocked_sequences: set[str] = set()
        self._semantic_site_actuation_controller_route_reentry_count: int = 0
        self._semantic_site_actuation_controller_route_reentry_lateral_count: int = 0
        self._semantic_site_actuation_controller_adapter_notify_count: int = 0
        self._semantic_site_actuation_controller_adapter_notify_success_count: int = 0
        self._semantic_site_actuation_controller_adapter_notify_block_count: int = 0
        self._semantic_site_closer_outcome_record_count: int = 0
        self._semantic_site_closer_sequence_blocked_count: int = 0
        self._semantic_site_closer_sequence_rotated_count: int = 0
        self._semantic_site_closer_same_failed_suppressed_count: int = 0
        self._semantic_site_closer_promising_selected_count: int = 0
        self._semantic_site_closer_closed_count: int = 0
        self._semantic_site_closer_failed_tail_count: int = 0
        self._semantic_site_closer_active_block_count: int = 0
        self._semantic_site_closer_alternate_pending_count: int = 0
        self._semantic_site_closer_alternate_forced_count: int = 0
        self._semantic_site_closer_alternate_missing_count: int = 0
        self._semantic_site_closer_alternate_expired_count: int = 0
        self._semantic_site_closer_pending_alternate: dict[str, Any] = {}
        self._semantic_site_closer_outcome_histogram: dict[str, int] = {}
        self._semantic_site_closer_sequence_outcome_histogram: dict[str, int] = {}
        self._semantic_site_closer_blocked_sequence_histogram: dict[str, int] = {}
        self._semantic_site_actuation_controller_far_tail_streak_name: str = ""
        self._semantic_site_actuation_controller_far_tail_streak_count: int = 0
        self._semantic_site_actuation_phase_downgrade_while_active_count: int = 0
        self._semantic_site_actuation_sequence_cursor_max: int = 0
        self._semantic_site_actuation_verify_issue_count: int = 0
        self._semantic_exact_site_lock_issue_count: int = 0
        self._semantic_exact_site_lock_consume_count: int = 0
        self._semantic_exact_site_lock_ttl_expire_count: int = 0
        self._semantic_exact_site_lock_abort_count: int = 0
        self._semantic_exact_site_lock_abort_reason_histogram: dict[str, int] = {}
        self._semantic_phase_downgrade_while_exact_site_lock_count: int = 0
        self._semantic_site_actuation_verify_execute_success_count: int = 0
        self._semantic_site_actuation_verify_state_change_success_count: int = 0
        self._semantic_exact_site_tx_issue_count: int = 0
        self._semantic_exact_site_tx_complete_count: int = 0
        self._semantic_exact_site_tx_abort_count: int = 0
        self._semantic_exact_site_tx_abort_reason_histogram: dict[str, int] = {}
        self._semantic_phase_downgrade_while_tx_active_count: int = 0
        self._semantic_reseed_while_tx_active_count: int = 0
        self._semantic_followup_enter_count: int = 0
        self._semantic_followup_complete_count: int = 0
        self._last_semantic_recommendation_active: bool = False
        self._last_semantic_mechanics_mode: str = ""
        self._last_semantic_phase: str = ""
        self._last_semantic_site_mode: str = ""
        self._last_semantic_event_kind: str = ""
        self._last_semantic_event_confidence: float = 0.0
        self._last_semantic_lease_age: int = -1
        self._last_semantic_suggested_actions: list[str] = []
        self._current_semantic_route_source: str = ""
        self._current_semantic_route_target_anchor: list[int] = []
        self._current_semantic_route_preview: list[str] = []
        self._last_semantic_route_source: str = ""
        self._last_semantic_route_target_anchor: list[int] = []
        self._last_semantic_route_preview: list[str] = []
        self._last_semantic_progress_goal_route_target_anchor: list[int] = []
        self._last_semantic_progress_goal_route_preview: list[str] = []
        self._last_hidden_state_route_resync_from_progress: bool = False
        self._semantic_route_recommendation_count: int = 0
        self._semantic_route_source_histogram: dict[str, int] = {}
        self._hidden_state_queue_clear_count: int = 0
        self._hidden_state_cleared_action_count: int = 0
        self._hidden_state_queue_clear_reason_histogram: dict[str, int] = {}
        self._stabilize_verify_one_shot_seen_count: int = 0
        self._stabilize_verify_one_shot_selected_count: int = 0
        self._stabilize_verify_one_shot_normalized_count: int = 0
        self._stabilize_verify_one_shot_blocked_reason_counts: dict[str, int] = {}
        self._stabilize_verify_one_shot_window_keys: set[str] = set()
        self._closed_rule_one_shot_seen_count: int = 0
        self._closed_rule_one_shot_selected_count: int = 0
        self._closed_rule_one_shot_normalized_count: int = 0
        self._closed_rule_one_shot_blocked_reason_counts: dict[str, int] = {}
        self._closed_rule_one_shot_keys: set[str] = set()
        self._post_stabilize_no_target_continuation_step_active: bool = False
        self._move_action_available: bool = True
        self._click_only_grid_probe_packet_counts: dict[str, int] = {}
        self._click_only_grid_probe_focus_key: str = ""
        self._click_only_grid_probe_focus_score: int = 0
        self._click_only_grid_probe_focus_level: int = 0
        self._click_only_grid_probe_focus_attempt: int = 0
        self._l1_family_starved_count: int = 0
        self._l1_action_family_quota_issued_count: int = 0
        self._l1_action_family_quota_blocked_count: int = 0
        self._l1_action_family_quota_last_step: int = -10_000
        self._l1_action_family_quota_cursor: int = 0
        self._l1_action_family_quota_action_histogram: dict[str, int] = {}
        self._l1_action_family_quota_reason_histogram: dict[str, int] = {}
        self._l1_action_family_quota_blocked_reason_counts: dict[str, int] = {}
        self._l1_lateral_probe_remaining: int = 0
        self._l1_lateral_probe_component_id: str = ""
        self._l1_lateral_probe_queue: list[str] = []
        self._l1_lateral_probe_issued_count: int = 0
        self._l1_attempt_outcome_record_count: int = 0
        self._l1_attempt_failed_prefix_count: int = 0
        self._l1_attempt_relay_anchor_count: int = 0
        self._l1_attempt_relay_followup_issued_count: int = 0
        self._l1_attempt_relay_followup_score_count: int = 0
        self._l1_attempt_relay_followup_gameover_count: int = 0
        self._l1_attempt_prefix_suppressed_count: int = 0
        self._l1_attempt_last_recorded_attempt: int = -1
        self._l1_attempt_relay_followup_last_step: int = -10_000
        self._l1_attempt_failed_prefix_keys: set[str] = set()
        self._l1_attempt_relay_anchor_queue: list[dict[str, Any]] = []
        self._l1_attempt_relay_followup_keys: set[str] = set()
        self._l1_attempt_relay_followup_outcome_steps: set[int] = set()
        self._auxiliary_click_pair_outcome_ranked_count: int = 0
        self._auxiliary_click_pair_vertical_transfer_count: int = 0
        self._auxiliary_click_pair_delta_anchor_count: int = 0
        self._auxiliary_click_pair_deferred_unsettled_count: int = 0
        self._auxiliary_click_pair_settle_tick_count: int = 0
        self._auxiliary_visual_arrangement_candidate_count: int = 0
        self._auxiliary_visual_arrangement_plan_issued_count: int = 0
        self._auxiliary_visual_arrangement_validate_count: int = 0
        self._auxiliary_visual_arrangement_direct_issued_count: int = 0
        self._auxiliary_visual_arrangement_duplicate_suppressed_count: int = 0
        self._auxiliary_visual_arrangement_plan_keys: set[str] = set()
        self._auxiliary_visual_arrangement_active_until_step: int = -10_000
        self._auxiliary_visual_arrangement_active_level: int = 0
        self._auxiliary_visual_arrangement_active_score: int = 0
        self._visual_stencil_first_effect_candidate_count: int = 0
        self._visual_stencil_first_effect_plan_issued_count: int = 0
        self._visual_stencil_first_effect_duplicate_suppressed_count: int = 0
        self._visual_stencil_first_effect_plan_keys: set[str] = set()
        self._visual_stencil_first_effect_active_until_step: int = -10_000
        self._visual_stencil_first_effect_active_level: int = 0
        self._visual_stencil_first_effect_active_score: int = 0
        self._visual_stencil_composition_candidate_count: int = 0
        self._visual_stencil_composition_plan_issued_count: int = 0
        self._visual_stencil_composition_duplicate_suppressed_count: int = 0
        self._visual_stencil_composition_plan_keys: set[str] = set()
        self._visual_stencil_composition_active_until_step: int = -10_000
        self._visual_stencil_composition_active_level: int = 0
        self._visual_stencil_composition_active_score: int = 0
        self._auxiliary_click_pair_reverse_suppressed_count: int = 0
        self._auxiliary_click_pair_repeat_suppressed_count: int = 0
        self._auxiliary_click_pair_attempt_counts: dict[str, int] = {}
        self._toggle_search_active: bool = False
        self._toggle_search_targets: list[dict[str, Any]] = []
        self._toggle_search_gray_index: int = 0
        self._toggle_search_last_score: int | None = None
        self._toggle_search_last_level: int | None = None
        self._last_structural_suggested_action_key: str = ""
        self._last_structural_suggested_action_turn: int = -10_000
        self._last_semantic_do_not_repeat_actions: list[str] = []
        self._last_semantic_site_distance: int | None = None
        self._last_semantic_site_distance_delta: int | None = None
        self._last_semantic_stagnant_turns: int = 0
        self._last_semantic_site_reaching_score: float = 0.0
        self._last_semantic_site_zone_confidence: float = 0.0
        self._last_semantic_site_mode_age: int = 0
        self._last_semantic_site_reaching_current_duration: int = 0
        self._last_semantic_site_actuation_active: bool = False
        self._last_semantic_site_actuation_age: int = 0
        self._last_semantic_site_actuation_sequence_name: str = ""
        self._last_semantic_site_actuation_sequence_remaining: list[str] = []
        self._last_semantic_site_actuation_preferred_sequence_name: str = ""
        self._last_semantic_site_actuation_contact_examples: list[dict[str, Any]] = []
        self._last_semantic_site_actuation_followup_examples: list[dict[str, Any]] = []
        self._active_semantic_adapter = None
        self._last_semantic_selected_action: str = ""
        self._last_semantic_recommended_action_selected: bool = False
        self._last_semantic_action_source: str = ""
        self._semantic_bridge_choice_recorded_this_step: bool = False
        self._site_actuation_contract_id: int = 0
        self._site_actuation_contract_active: bool = False
        self._site_actuation_contract_armed: bool = False
        self._site_actuation_contract_sequence_name: str = ""
        self._site_actuation_contract_steps: list[str] = []
        self._site_actuation_contract_cursor: int = 0
        self._site_actuation_contract_started_turn: int = -1
        self._site_actuation_contract_last_queue_load_turn: int = -1
        self._site_actuation_contract_last_executed_turn: int = -1
        self._site_actuation_contract_last_executed_action: str = ""
        self._site_actuation_contract_abort_reason: str = ""
        self._site_actuation_contract_result_pending: bool = False
        self._site_actuation_contract_last_completed_sequence_name: str = ""
        self._site_actuation_contract_last_completed_actions: list[str] = []
        self._site_actuation_contract_last_completed_turn: int = -1
        self._site_actuation_contract_pending_verify_name: str = ""
        self._site_actuation_contract_pending_verify_steps: list[str] = []
        self._exact_site_lock_active: bool = False
        self._exact_site_lock_sequence_name: str = ""
        self._exact_site_lock_ttl_remaining: int = 0
        self._exact_site_lock_started_turn: int = -1
        self._exact_site_tx_id: int = 0
        self._exact_site_tx_active: bool = False
        self._exact_site_tx_phase: str = ""
        self._exact_site_tx_cursor: int = 0
        self._exact_site_tx_expected_actions: list[str] = []
        self._exact_site_tx_executed_actions: list[str] = []
        self._exact_site_tx_site_lock_active: bool = False
        self._exact_site_tx_ttl_remaining: int = 0
        self._exact_site_tx_retry_budget: int = 0
        self._exact_site_tx_abort_reason: str = ""
        self._exact_site_tx_started_turn: int = -1

    @property
    def compiler(self) -> ControlIRCompiler:
        return self._compiler

    @property
    def resolver(self) -> CandidateResolver:
        return self._resolver

    @property
    def planner(self) -> MicroPlanner:
        return self._planner

    def reset(self) -> None:
        """Reset for a new episode."""
        self._compiler.reset()
        self._resolver.reset()
        self._planner.reset()
        self._escalation_manager.reset()
        self._hidden_state_executor.reset()
        self._consecutive_local_failures = 0
        self._total_local_steps = 0
        self._total_base_local_steps = 0
        self._total_ticket_assisted_local_steps = 0
        self._total_semantic_assisted_local_steps = 0
        self._total_local_accepted_steps = 0
        self._total_local_rejected_steps = 0
        self._total_analyzer_calls = 0
        self._stabilize_verify_one_shot_seen_count = 0
        self._stabilize_verify_one_shot_selected_count = 0
        self._stabilize_verify_one_shot_normalized_count = 0
        self._stabilize_verify_one_shot_blocked_reason_counts = {}
        self._stabilize_verify_one_shot_window_keys = set()
        self._closed_rule_one_shot_seen_count = 0
        self._closed_rule_one_shot_selected_count = 0
        self._closed_rule_one_shot_normalized_count = 0
        self._closed_rule_one_shot_blocked_reason_counts = {}
        self._closed_rule_one_shot_keys = set()
        self._post_stabilize_no_target_continuation_step_active = False
        self._move_action_available = True
        self._click_only_grid_probe_packet_counts = {}
        self._click_only_grid_probe_focus_key = ""
        self._click_only_grid_probe_focus_score = 0
        self._click_only_grid_probe_focus_level = 0
        self._click_only_grid_probe_focus_attempt = 0
        self._l1_family_starved_count = 0
        self._l1_action_family_quota_issued_count = 0
        self._l1_action_family_quota_blocked_count = 0
        self._l1_action_family_quota_last_step = -10_000
        self._l1_action_family_quota_cursor = 0
        self._l1_action_family_quota_action_histogram = {}
        self._l1_action_family_quota_reason_histogram = {}
        self._l1_action_family_quota_blocked_reason_counts = {}
        self._l1_lateral_probe_remaining = 0
        self._l1_lateral_probe_component_id = ""
        self._l1_lateral_probe_queue = []
        self._l1_lateral_probe_issued_count = 0
        self._l1_attempt_outcome_record_count = 0
        self._l1_attempt_failed_prefix_count = 0
        self._l1_attempt_relay_anchor_count = 0
        self._l1_attempt_relay_followup_issued_count = 0
        self._l1_attempt_relay_followup_score_count = 0
        self._l1_attempt_relay_followup_gameover_count = 0
        self._l1_attempt_prefix_suppressed_count = 0
        self._l1_attempt_last_recorded_attempt = -1
        self._l1_attempt_relay_followup_last_step = -10_000
        self._l1_attempt_failed_prefix_keys = set()
        self._l1_attempt_relay_anchor_queue = []
        self._l1_attempt_relay_followup_keys = set()
        self._l1_attempt_relay_followup_outcome_steps = set()
        self._auxiliary_click_pair_outcome_ranked_count = 0
        self._auxiliary_click_pair_vertical_transfer_count = 0
        self._auxiliary_click_pair_delta_anchor_count = 0
        self._auxiliary_click_pair_deferred_unsettled_count = 0
        self._auxiliary_click_pair_settle_tick_count = 0
        self._auxiliary_visual_arrangement_candidate_count = 0
        self._auxiliary_visual_arrangement_plan_issued_count = 0
        self._auxiliary_visual_arrangement_validate_count = 0
        self._auxiliary_visual_arrangement_direct_issued_count = 0
        self._auxiliary_visual_arrangement_duplicate_suppressed_count = 0
        self._auxiliary_visual_arrangement_plan_keys = set()
        self._auxiliary_visual_arrangement_active_until_step = -10_000
        self._auxiliary_visual_arrangement_active_level = 0
        self._auxiliary_visual_arrangement_active_score = 0
        self._visual_stencil_first_effect_candidate_count = 0
        self._visual_stencil_first_effect_plan_issued_count = 0
        self._visual_stencil_first_effect_duplicate_suppressed_count = 0
        self._visual_stencil_first_effect_plan_keys = set()
        self._visual_stencil_first_effect_active_until_step = -10_000
        self._visual_stencil_first_effect_active_level = 0
        self._visual_stencil_first_effect_active_score = 0
        self._visual_stencil_composition_candidate_count = 0
        self._visual_stencil_composition_plan_issued_count = 0
        self._visual_stencil_composition_duplicate_suppressed_count = 0
        self._visual_stencil_composition_plan_keys = set()
        self._visual_stencil_composition_active_until_step = -10_000
        self._visual_stencil_composition_active_level = 0
        self._visual_stencil_composition_active_score = 0
        self._auxiliary_click_pair_reverse_suppressed_count = 0
        self._auxiliary_click_pair_repeat_suppressed_count = 0
        self._auxiliary_click_pair_attempt_counts = {}
        self._pending_local_plan_ack = False
        self._step_history.clear()
        self._semantic_recommendation_turn_count = 0
        self._semantic_recommended_action_match_count = 0
        self._semantic_recommended_action_executed_count = 0
        self._semantic_branch_entry_attempt_count = 0
        self._semantic_branch_entry_success_count = 0
        self._semantic_branch_internal_progress_turns = 0
        self._semantic_branch_to_site_distance_min = None
        self._semantic_site_reach_attempt_count = 0
        self._semantic_site_reach_success_count = 0
        self._semantic_branch_exit_back_to_corridor_count = 0
        self._semantic_site_reaching_enter_count = 0
        self._semantic_site_reaching_exit_count = 0
        self._semantic_site_reaching_total_duration = 0
        self._semantic_site_actuation_enter_count = 0
        self._semantic_site_actuation_attempt_count = 0
        self._semantic_site_actuation_success_count = 0
        self._semantic_site_actuation_sequence_issue_count = 0
        self._semantic_site_actuation_sequence_consume_count = 0
        self._semantic_site_actuation_contact_success_count = 0
        self._semantic_site_actuation_display_change_count = 0
        self._semantic_site_actuation_no_contact_count = 0
        self._semantic_site_actuation_state_change_success_count = 0
        self._semantic_site_actuation_verify_success_count = 0
        self._semantic_site_exit_before_actuation_count = 0
        self._semantic_site_exit_before_verify_count = 0
        self._semantic_site_actuation_sequence_histogram = {}
        self._semantic_site_actuation_display_change_sequence_histogram = {}
        self._semantic_site_actuation_no_display_change_sequence_histogram = {}
        self._semantic_site_actuation_no_contact_sequence_histogram = {}
        self._semantic_site_actuation_contact_position_histogram = {}
        self._semantic_site_actuation_contact_pair_histogram = {}
        self._semantic_site_actuation_followup_window_count = 0
        self._semantic_site_actuation_followup_stamp_change_count = 0
        self._semantic_site_actuation_followup_no_change_count = 0
        self._semantic_site_actuation_followup_change_pair_histogram = {}
        self._semantic_site_actuation_followup_no_change_pair_histogram = {}
        self._semantic_site_actuation_reused_sequence_count = 0
        self._semantic_site_actuation_sequence_queue_load_count = 0
        self._semantic_site_actuation_stale_tail_queue_skip_count = 0
        self._semantic_site_actuation_sequence_execute_success_count = 0
        self._semantic_site_actuation_sequence_abort_count = 0
        self._semantic_site_actuation_sequence_abort_reason_histogram = {}
        self._semantic_site_actuation_far_tail_turn_count = 0
        self._semantic_site_actuation_far_tail_sequence_histogram = {}
        self._semantic_site_actuation_far_tail_site_histogram = {}
        self._semantic_site_actuation_reacquire_candidate_count = 0
        self._semantic_site_actuation_reacquire_count = 0
        self._semantic_site_actuation_controller_reset_count = 0
        self._semantic_site_actuation_controller_blocked_sequence_histogram = {}
        self._semantic_site_actuation_controller_blocked_sequence_count = 0
        self._semantic_site_actuation_controller_far_tail_turn_count = 0
        self._semantic_site_actuation_controller_far_tail_sequence_histogram = {}
        self._semantic_site_actuation_controller_reacquire_issue_count = 0
        self._semantic_site_actuation_controller_reacquire_cooldown_remaining = 0
        self._semantic_site_actuation_controller_reacquire_hold_remaining = 0
        self._semantic_site_actuation_controller_reacquire_last_distance = None
        self._semantic_site_actuation_controller_reacquire_last_action = ""
        self._semantic_site_actuation_controller_reacquire_lateral_recovery_used = False
        self._semantic_site_actuation_controller_reacquire_blocked_sequences = set()
        self._semantic_site_actuation_controller_route_reentry_count = 0
        self._semantic_site_actuation_controller_route_reentry_lateral_count = 0
        self._semantic_site_actuation_controller_adapter_notify_count = 0
        self._semantic_site_actuation_controller_adapter_notify_success_count = 0
        self._semantic_site_actuation_controller_adapter_notify_block_count = 0
        self._semantic_site_closer_outcome_record_count = 0
        self._semantic_site_closer_sequence_blocked_count = 0
        self._semantic_site_closer_sequence_rotated_count = 0
        self._semantic_site_closer_same_failed_suppressed_count = 0
        self._semantic_site_closer_promising_selected_count = 0
        self._semantic_site_closer_closed_count = 0
        self._semantic_site_closer_failed_tail_count = 0
        self._semantic_site_closer_active_block_count = 0
        self._semantic_site_closer_alternate_pending_count = 0
        self._semantic_site_closer_alternate_forced_count = 0
        self._semantic_site_closer_alternate_missing_count = 0
        self._semantic_site_closer_alternate_expired_count = 0
        self._semantic_site_closer_pending_alternate = {}
        self._semantic_site_closer_outcome_histogram = {}
        self._semantic_site_closer_sequence_outcome_histogram = {}
        self._semantic_site_closer_blocked_sequence_histogram = {}
        self._semantic_site_actuation_controller_far_tail_streak_name = ""
        self._semantic_site_actuation_controller_far_tail_streak_count = 0
        self._semantic_site_actuation_phase_downgrade_while_active_count = 0
        self._semantic_site_actuation_sequence_cursor_max = 0
        self._semantic_site_actuation_verify_issue_count = 0
        self._semantic_exact_site_lock_issue_count = 0
        self._semantic_exact_site_lock_consume_count = 0
        self._semantic_exact_site_lock_ttl_expire_count = 0
        self._semantic_exact_site_lock_abort_count = 0
        self._semantic_exact_site_lock_abort_reason_histogram = {}
        self._semantic_phase_downgrade_while_exact_site_lock_count = 0
        self._semantic_site_actuation_verify_execute_success_count = 0
        self._semantic_site_actuation_verify_state_change_success_count = 0
        self._semantic_exact_site_tx_issue_count = 0
        self._semantic_exact_site_tx_complete_count = 0
        self._semantic_exact_site_tx_abort_count = 0
        self._semantic_exact_site_tx_abort_reason_histogram = {}
        self._semantic_phase_downgrade_while_tx_active_count = 0
        self._semantic_reseed_while_tx_active_count = 0
        self._semantic_followup_enter_count = 0
        self._semantic_followup_complete_count = 0
        self._last_semantic_recommendation_active = False
        self._last_semantic_mechanics_mode = ""
        self._last_semantic_phase = ""
        self._last_semantic_site_mode = ""
        self._last_semantic_event_kind = ""
        self._last_semantic_event_confidence = 0.0
        self._last_semantic_lease_age = -1
        self._last_semantic_suggested_actions = []
        self._current_semantic_route_source = ""
        self._current_semantic_route_target_anchor = []
        self._current_semantic_route_preview = []
        self._last_semantic_route_source = ""
        self._last_semantic_route_target_anchor = []
        self._last_semantic_route_preview = []
        self._last_semantic_progress_goal_route_target_anchor = []
        self._last_semantic_progress_goal_route_preview = []
        self._last_hidden_state_route_resync_from_progress = False
        self._semantic_route_recommendation_count = 0
        self._semantic_route_source_histogram = {}
        self._hidden_state_queue_clear_count = 0
        self._hidden_state_cleared_action_count = 0
        self._hidden_state_queue_clear_reason_histogram = {}
        self._reset_toggle_combination_search()
        self._last_structural_suggested_action_key = ""
        self._last_structural_suggested_action_turn = -10_000
        self._last_semantic_do_not_repeat_actions = []
        self._last_semantic_site_distance = None
        self._last_semantic_site_distance_delta = None
        self._last_semantic_stagnant_turns = 0
        self._last_semantic_site_reaching_score = 0.0
        self._last_semantic_site_zone_confidence = 0.0
        self._last_semantic_site_mode_age = 0
        self._last_semantic_site_reaching_current_duration = 0
        self._last_semantic_site_actuation_active = False
        self._last_semantic_site_actuation_age = 0
        self._last_semantic_site_actuation_sequence_name = ""
        self._last_semantic_site_actuation_sequence_remaining = []
        self._last_semantic_site_actuation_preferred_sequence_name = ""
        self._last_semantic_site_actuation_contact_examples = []
        self._last_semantic_site_actuation_followup_examples = []
        self._last_semantic_selected_action = ""
        self._last_semantic_recommended_action_selected = False
        self._last_semantic_action_source = ""
        self._semantic_bridge_choice_recorded_this_step = False
        self._site_actuation_contract_id = 0
        self._site_actuation_contract_active = False
        self._site_actuation_contract_armed = False
        self._site_actuation_contract_sequence_name = ""
        self._site_actuation_contract_steps = []
        self._site_actuation_contract_cursor = 0
        self._site_actuation_contract_started_turn = -1
        self._site_actuation_contract_last_queue_load_turn = -1
        self._site_actuation_contract_last_executed_turn = -1
        self._site_actuation_contract_last_executed_action = ""
        self._site_actuation_contract_abort_reason = ""
        self._site_actuation_contract_result_pending = False

    def _hidden_state_executor_metrics(self) -> dict[str, Any]:
        metrics = dict(self._hidden_state_executor.episode_summary())
        metrics["queue_clear_count"] = int(self._hidden_state_queue_clear_count)
        metrics["cleared_action_count"] = int(self._hidden_state_cleared_action_count)
        metrics["queue_clear_reason_histogram"] = dict(self._hidden_state_queue_clear_reason_histogram)
        return metrics

    def _reset_toggle_combination_search(self) -> None:
        self._toggle_search_active = False
        self._toggle_search_targets = []
        self._toggle_search_gray_index = 0
        self._toggle_search_last_score = None
        self._toggle_search_last_level = None

    def _handoff_site_actuation_contract_for_state_change(self) -> None:
        if self._exact_site_tx_active:
            self._complete_exact_site_transaction(reason="state_change_handoff")
        if self._exact_site_lock_active:
            self._release_exact_site_lock()
        self._site_actuation_contract_active = False
        self._site_actuation_contract_armed = False
        self._site_actuation_contract_abort_reason = ""
        self._site_actuation_contract_sequence_name = ""
        self._site_actuation_contract_steps = []
        self._site_actuation_contract_cursor = 0
        self._site_actuation_contract_started_turn = -1
        self._site_actuation_contract_last_queue_load_turn = -1
        self._site_actuation_contract_last_executed_turn = -1
        self._site_actuation_contract_last_executed_action = ""
        self._site_actuation_contract_result_pending = False
        self._site_actuation_contract_pending_verify_name = ""
        self._site_actuation_contract_pending_verify_steps = []
        self._site_actuation_contract_last_completed_sequence_name = ""
        self._site_actuation_contract_last_completed_actions = []
        self._site_actuation_contract_last_completed_turn = -1
        self._site_actuation_contract_pending_verify_name = ""
        self._site_actuation_contract_pending_verify_steps = []
        self._exact_site_lock_active = False
        self._exact_site_lock_sequence_name = ""
        self._exact_site_lock_ttl_remaining = 0
        self._exact_site_lock_started_turn = -1
        self._exact_site_tx_id = 0
        self._exact_site_tx_active = False
        self._exact_site_tx_phase = ""
        self._exact_site_tx_cursor = 0
        self._exact_site_tx_expected_actions = []
        self._exact_site_tx_executed_actions = []
        self._exact_site_tx_site_lock_active = False
        self._exact_site_tx_ttl_remaining = 0
        self._exact_site_tx_retry_budget = 0
        self._exact_site_tx_abort_reason = ""
        self._exact_site_tx_started_turn = -1

    def step(
        self,
        runtime: Any,
        step_index: int,
        *,
        candidate_pool: list[dict[str, Any]] | None = None,
        target_change_count: int = 0,
        agent: Any | None = None,
        allowed_actions: list[str] | None = None,
        blocked_actions: set[str] | None = None,
        board_delta: dict[str, Any] | None = None,
    ) -> LocalControllerResult:
        """Execute one Local Controller step.

        This is the main entry point called from the harness each step.

        Args:
            runtime: The ECARuntime instance.
            step_index: Current step index.
            candidate_pool: Raw candidate pool from runtime.
            target_change_count: Number of target changes so far.
            allowed_actions: List of action names the environment accepts.
            blocked_actions: Set of action names currently blocked.
            board_delta: Optional dict describing recent board changes.

        Returns:
            LocalControllerResult with either planned actions or analyzer flag.
        """
        result = LocalControllerResult()

        if not self.config.enabled:
            result.needs_analyzer = True
            result.analyzer_reason = "local_controller_disabled"
            self._total_analyzer_calls += 1
            return result

        # ── Step 1: Compile IR ──────────────────────────────────────────
        semantic_action_lease = self._build_semantic_action_lease(
            runtime=runtime,
            agent=agent,
            step_index=step_index,
        )
        hidden_state_ticket = self._build_hidden_state_execution_ticket(runtime)
        hidden_state_lease = self._update_hidden_state_executor(
            runtime=runtime,
            execution_ticket=hidden_state_ticket,
            step_index=step_index,
        )
        semantic_action_lease = self._merge_hidden_state_executor_lease(
            semantic_action_lease=semantic_action_lease,
            hidden_state_lease=hidden_state_lease,
        )
        self._maybe_apply_controller_site_reacquire(
            semantic_action_lease,
            runtime=runtime,
            step_index=step_index,
        )
        self._begin_semantic_bridge_step(semantic_action_lease)
        move_action_available = (
            allowed_actions is None
            or any(
                str(action_name or "").strip().upper() in _MOVE_ACTION_SET
                for action_name in list(allowed_actions or [])
            )
        )
        self._move_action_available = bool(move_action_available)
        self._update_l1_attempt_outcome_ledger(runtime=runtime, step_index=step_index)
        post_stabilize_no_target_active = (
            move_action_available
            and self._post_stabilize_no_target_continuation_active(runtime)
        )
        stabilize_one_shot_plan = (
            self._build_stabilize_one_shot_plan(
                runtime=runtime,
                agent=agent,
                semantic_action_lease=semantic_action_lease,
                allowed_actions=allowed_actions,
                blocked_actions=blocked_actions,
                allow_fresh_unstarved=True,
            )
            if hidden_state_lease.active and not post_stabilize_no_target_active
            else None
        )
        closed_rule_one_shot_plan = (
            self._build_closed_rule_one_shot_plan(
                runtime=runtime,
                agent=agent,
                semantic_action_lease=semantic_action_lease,
                allowed_actions=allowed_actions,
                blocked_actions=blocked_actions,
            )
            if (
                hidden_state_lease.active
                and stabilize_one_shot_plan is None
                and not post_stabilize_no_target_active
            )
            else None
        )
        if stabilize_one_shot_plan is not None:
            stabilize_actions, stabilize_target = stabilize_one_shot_plan
            result.locally_planned = True
            result.actions = stabilize_actions
            result.selected_target = stabilize_target
            result.resolution = ResolutionResult(
                selected=stabilize_target,
                all_scored=[stabilize_target],
                resolved=True,
                resolution_source="stabilize_verify_one_shot",
            )
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._total_base_local_steps += 1
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result
        if closed_rule_one_shot_plan is not None:
            closed_actions, closed_target = closed_rule_one_shot_plan
            result.locally_planned = True
            result.actions = closed_actions
            result.selected_target = closed_target
            result.resolution = ResolutionResult(
                selected=closed_target,
                all_scored=[closed_target],
                resolved=True,
                resolution_source="closed_rule_one_shot",
            )
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._total_base_local_steps += 1
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result
        if hidden_state_lease.active:
            hidden_state_actions = self._execute_hidden_state_executor(
                agent=agent,
                semantic_action_lease=semantic_action_lease,
            )
            if hidden_state_actions:
                result.locally_planned = True
                result.actions = hidden_state_actions
                self._consecutive_local_failures = 0
                self._total_local_steps += 1
                self._note_semantic_assisted_local_step()
                self._pending_local_plan_ack = True
                self._escalation_manager.record_local_success()
                self._record_step(step_index, result)
                return result
            self._hidden_state_executor.note_generic_preemption()
        compilation = self._compiler.compile(runtime, step_index)
        self._post_stabilize_no_target_continuation_step_active = bool(
            post_stabilize_no_target_active
        )
        if post_stabilize_no_target_active and self._should_suppress_post_stabilize_stale_retarget(runtime, compilation):
            compilation.instructions = [
                instruction
                for instruction in list(compilation.instructions or [])
                if str(getattr(instruction, "action", "") or "") != "force_retarget"
            ]
        result.compilation = compilation
        site_actuation_preempt = bool(
            semantic_action_lease.get("active", False)
            and (
                str(semantic_action_lease.get("semantic_phase", "") or "").strip().lower().startswith("site_actuation")
                or bool(semantic_action_lease.get("site_actuation_active", False))
                or bool(str(semantic_action_lease.get("site_actuation_sequence_name", "") or "").strip())
                or bool(str(semantic_action_lease.get("site_actuation_pending_verify_name", "") or "").strip())
            )
            and list(semantic_action_lease.get("site_actuation_sequence_remaining", []) or [])
        )
        # Hard preempt: semantic_site_commit mode with priority >= 0.99
        # overrides all contract/analyzer decisions for the commit horizon.
        _site_commit_mode = str(semantic_action_lease.get("site_mode", "") or "").strip().lower()
        _site_commit_priority = float(semantic_action_lease.get("priority", 0.0) or 0.0)
        site_commit_preempt = bool(
            semantic_action_lease.get("active", False)
            and (_site_commit_mode == "semantic_site_commit" or _site_commit_priority >= 0.99)
            and semantic_action_lease.get("suggested_actions")
        )
        if site_commit_preempt:
            site_actuation_preempt = True

        # ── Step 1b: EscalationManager pre-check (BaseLocalLoop) ───────
        # Determine TargetHealth and base pool from runtime (Phase 1 observers)
        structural_suggested_actions = self._build_structural_suggested_local_test_actions(
            runtime=runtime,
            agent=agent,
            semantic_action_lease=semantic_action_lease,
            allowed_actions=allowed_actions,
            blocked_actions=blocked_actions,
            step_index=step_index,
        )
        if structural_suggested_actions:
            result.locally_planned = True
            result.actions = structural_suggested_actions
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._total_base_local_steps += 1
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result

        click_only_grid_focus_actions = self._build_click_only_grid_probe_focus_actions(
            runtime=runtime,
            allowed_actions=allowed_actions,
            blocked_actions=blocked_actions,
        )
        if click_only_grid_focus_actions:
            result.locally_planned = True
            result.actions = click_only_grid_focus_actions
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._total_base_local_steps += 1
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result

        l1_lateral_probe_actions = self._build_l1_lateral_probe_lease_actions(
            runtime=runtime,
            allowed_actions=allowed_actions,
            blocked_actions=blocked_actions,
            semantic_action_lease=semantic_action_lease,
            hidden_state_lease=hidden_state_lease,
        )
        if l1_lateral_probe_actions:
            result.locally_planned = True
            result.actions = l1_lateral_probe_actions
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._total_base_local_steps += 1
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result

        l1_attempt_relay_plan = self._build_l1_attempt_relay_followup_plan(
            runtime=runtime,
            step_index=step_index,
            allowed_actions=allowed_actions,
            blocked_actions=blocked_actions,
            semantic_action_lease=semantic_action_lease,
            hidden_state_lease=hidden_state_lease,
        )
        if l1_attempt_relay_plan is not None:
            relay_actions, relay_target = l1_attempt_relay_plan
            result.locally_planned = True
            result.actions = relay_actions
            result.selected_target = relay_target
            result.resolution = ResolutionResult(
                selected=relay_target,
                all_scored=[relay_target],
                resolved=True,
                resolution_source="l1_attempt_relay_followup",
            )
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._total_base_local_steps += 1
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result

        visual_stencil_plan = (
            self._build_visual_stencil_first_effect_direct_plan(
                runtime=runtime,
                step_index=step_index,
                agent=agent,
                semantic_action_lease=semantic_action_lease,
                allowed_actions=allowed_actions,
                blocked_actions=blocked_actions,
            )
            if not bool(semantic_action_lease.get("active", False))
            and not bool(hidden_state_lease.active)
            else None
        )
        if visual_stencil_plan is not None:
            stencil_actions, stencil_target = visual_stencil_plan
            result.locally_planned = True
            result.actions = stencil_actions
            result.selected_target = stencil_target
            result.resolution = ResolutionResult(
                selected=stencil_target,
                all_scored=[stencil_target],
                resolved=True,
                resolution_source="visual_stencil_first_effect",
            )
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._total_base_local_steps += 1
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result

        visual_stencil_composition_plan = (
            self._build_visual_stencil_composition_direct_plan(
                runtime=runtime,
                step_index=step_index,
                agent=agent,
                semantic_action_lease=semantic_action_lease,
                allowed_actions=allowed_actions,
                blocked_actions=blocked_actions,
            )
            if not bool(semantic_action_lease.get("active", False))
            and not bool(hidden_state_lease.active)
            else None
        )
        if visual_stencil_composition_plan is not None:
            composition_actions, composition_target = visual_stencil_composition_plan
            result.locally_planned = True
            result.actions = composition_actions
            result.selected_target = composition_target
            result.resolution = ResolutionResult(
                selected=composition_target,
                all_scored=[composition_target],
                resolved=True,
                resolution_source="visual_stencil_composition",
            )
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._total_base_local_steps += 1
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result

        visual_arrangement_plan = (
            self._build_auxiliary_visual_arrangement_direct_plan(
                runtime=runtime,
                step_index=step_index,
                agent=agent,
                semantic_action_lease=semantic_action_lease,
                allowed_actions=allowed_actions,
                blocked_actions=blocked_actions,
            )
            if not bool(semantic_action_lease.get("active", False))
            and not bool(hidden_state_lease.active)
            else None
        )
        if visual_arrangement_plan is not None:
            visual_actions, visual_target = visual_arrangement_plan
            result.locally_planned = True
            result.actions = visual_actions
            result.selected_target = visual_target
            result.resolution = ResolutionResult(
                selected=visual_target,
                all_scored=[visual_target],
                resolved=True,
                resolution_source="auxiliary_visual_arrangement",
            )
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._total_base_local_steps += 1
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result

        base_stabilize_one_shot_plan = (
            self._build_stabilize_one_shot_plan(
                runtime=runtime,
                agent=agent,
                semantic_action_lease=semantic_action_lease,
                allowed_actions=allowed_actions,
                blocked_actions=blocked_actions,
                allow_fresh_unstarved=True,
            )
            if not bool(semantic_action_lease.get("active", False))
            and not bool(hidden_state_lease.active)
            else None
        )
        if base_stabilize_one_shot_plan is not None:
            stabilize_actions, stabilize_target = base_stabilize_one_shot_plan
            result.locally_planned = True
            result.actions = stabilize_actions
            result.selected_target = stabilize_target
            result.resolution = ResolutionResult(
                selected=stabilize_target,
                all_scored=[stabilize_target],
                resolved=True,
                resolution_source="stabilize_verify_one_shot",
            )
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._total_base_local_steps += 1
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result

        l1_quota_actions = self._build_l1_action_family_quota_actions(
            runtime=runtime,
            step_index=step_index,
            allowed_actions=allowed_actions,
            blocked_actions=blocked_actions,
            semantic_action_lease=semantic_action_lease,
            hidden_state_lease=hidden_state_lease,
        )
        if l1_quota_actions:
            quota_action_name = str(l1_quota_actions[0].action_name or "").strip().upper()
            visual_arrangement_actions = self._build_auxiliary_visual_arrangement_actions(
                runtime=runtime,
                validate_action_name=quota_action_name,
            )
            if visual_arrangement_actions:
                direct_visual_actions: list[PlannedAction] = []
                for action in visual_arrangement_actions:
                    if str(action.action_name or "").strip().lower() == "click":
                        position = dict(action.parameters.get("position") or {})
                        direct_visual_actions.append(
                            PlannedAction(
                                action_name=_CLICK_ACTION,
                                target_component_id=action.target_component_id,
                                parameters={
                                    "x": int(position.get("row", 0) or 0),
                                    "y": int(position.get("col", 0) or 0),
                                },
                                reason=action.reason,
                            )
                        )
                    else:
                        direct_visual_actions.append(action)
                l1_quota_actions = direct_visual_actions
            quota_component_id = f"auxiliary::{quota_action_name.lower()}"
            quota_target = ScoredCandidate(
                component_id=quota_component_id,
                action_family="auxiliary",
                source="l1_action_family_quota",
                action_name=quota_action_name,
                anchor_key=f"{quota_component_id}||1||1",
                center_row=1,
                center_col=1,
                base_admission_score=1.0,
                resolver_score=1.0,
                selected=True,
                selection_reason="l1_action_family_quota",
            )
            result.locally_planned = True
            result.actions = l1_quota_actions
            result.selected_target = quota_target
            result.resolution = ResolutionResult(
                selected=quota_target,
                all_scored=[quota_target],
                resolved=True,
                resolution_source="l1_action_family_quota",
            )
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._total_base_local_steps += 1
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result

        toggle_search_actions = self._build_toggle_combination_search_actions(
            runtime=runtime,
            semantic_action_lease=semantic_action_lease,
            allowed_actions=allowed_actions,
            blocked_actions=blocked_actions,
        )
        if toggle_search_actions:
            result.locally_planned = True
            result.actions = toggle_search_actions
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._total_base_local_steps += 1
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result

        _th_result = getattr(runtime, "_last_target_health_result", None)
        _base_pool = getattr(runtime, "_last_base_pool", None)
        _health = _th_result.health if _th_result is not None else None
        _pool_size = len(_base_pool.candidates) if _base_pool is not None else 0
        _current_family = self._infer_current_family(runtime, compilation, _base_pool)
        _ir_override = bool(
            compilation.needs_analyzer_fallback
            and compilation.fallback_reason not in ("", "no_ir_no_target")
        )
        aftercare_signal = self._runtime_aftercare_signal(runtime)

        # Import TargetHealth here to avoid circular imports at module level
        from arcgym.eca.target_health import TargetHealth
        if _health is None:
            # Phase 1 observers not yet run; fall through to legacy logic
            pass
        else:
            escalation = self._escalation_manager.decide(
                health=_health,
                base_pool_size=_pool_size,
                current_family=_current_family,
                ir_override=_ir_override,
                step_index=step_index,
                protected_hedge_visible_turn_count=int(
                    aftercare_signal.get("protected_hedge_visible_turn_count", 0) or 0
                ),
                post_reclaim_hold_count=int(
                    aftercare_signal.get("post_reclaim_hold_count", 0) or 0
                ),
                post_final_shared_entry_count=int(
                    aftercare_signal.get("post_final_shared_entry_count", 0) or 0
                ),
            )
            result.escalation_decision = escalation
            if escalation.should_escalate and not site_actuation_preempt:
                hidden_abort_recovery_actions = self._build_hidden_abort_recovery_actions(
                    agent=agent,
                    semantic_action_lease=semantic_action_lease,
                    allowed_actions=allowed_actions,
                    blocked_actions=blocked_actions,
                )
                if hidden_abort_recovery_actions:
                    result.locally_planned = True
                    result.actions = hidden_abort_recovery_actions
                    self._consecutive_local_failures = 0
                    self._total_local_steps += 1
                    self._pending_local_plan_ack = True
                    self._escalation_manager.record_local_success()
                    self._record_step(step_index, result)
                    return result
                escalation_recovery_resolution = self._build_base_resolution(
                    runtime=runtime,
                    evidence=compilation.evidence,
                    target_change_count=target_change_count,
                )
                escalation_recovery_target = (
                    escalation_recovery_resolution.selected
                    if escalation_recovery_resolution is not None
                    and escalation_recovery_resolution.resolved
                    else None
                )
                generic_fallback_actions = self._build_generic_exploration_fallback_actions(
                    runtime=runtime,
                    selected_target=escalation_recovery_target,
                    agent=agent,
                    allowed_actions=allowed_actions,
                    blocked_actions=blocked_actions,
                    reason=f"escalation:{escalation.reason}",
                )
                if generic_fallback_actions:
                    result.locally_planned = True
                    result.actions = generic_fallback_actions
                    result.resolution = escalation_recovery_resolution
                    result.selected_target = escalation_recovery_target
                    self._consecutive_local_failures = 0
                    self._total_local_steps += 1
                    self._pending_local_plan_ack = True
                    self._escalation_manager.record_local_success()
                    self._record_step(step_index, result)
                    return result
                result.needs_analyzer = True
                result.analyzer_reason = f"escalation:{escalation.reason}"
                result.sparse_analyzer_input = self._build_sparse_input(
                    compilation, None, None, runtime=runtime
                )
                self._total_analyzer_calls += 1
                self._record_step(step_index, result)
                return result

        if site_actuation_preempt:
            semantic_fallback_actions = self._build_semantic_fallback_actions(
                agent=agent,
                semantic_action_lease=semantic_action_lease,
            )
            if semantic_fallback_actions:
                result.locally_planned = True
                result.actions = semantic_fallback_actions
                self._consecutive_local_failures = 0
                self._total_local_steps += 1
                self._note_semantic_assisted_local_step()
                self._pending_local_plan_ack = True
                self._escalation_manager.record_local_success()
                self._record_step(step_index, result)
                return result

        # If compiler says fallback needed (legacy path)
        if compilation.needs_analyzer_fallback and compilation.fallback_reason not in ("", "no_ir_no_target"):
            result.needs_analyzer = True
            result.analyzer_reason = f"compiler_fallback:{compilation.fallback_reason}"
            result.sparse_analyzer_input = self._build_sparse_input(
                compilation, None, None, runtime=runtime
            )
            self._total_analyzer_calls += 1
            self._record_step(step_index, result)
            return result

        # If no instructions, check if we even need to act
        if not compilation.instructions:
            base_resolution = self._build_base_resolution(
                runtime=runtime,
                evidence=compilation.evidence,
                target_change_count=target_change_count,
            )
            if base_resolution is not None and base_resolution.resolved and base_resolution.selected:
                result.resolution = base_resolution
                result.selected_target = base_resolution.selected
                micro_plan = self._planner.plan(
                    resolution=base_resolution,
                    instructions=[],
                    allowed_actions=allowed_actions,
                    blocked_actions=blocked_actions,
                    board_delta=board_delta,
                )
                result.micro_plan = micro_plan
                if micro_plan.planned and micro_plan.actions:
                    normalized_actions = self._normalize_planned_actions(
                        runtime=runtime,
                        actions=micro_plan.actions,
                        selected_target=base_resolution.selected,
                        agent=agent,
                        semantic_action_lease=semantic_action_lease,
                    )
                    if not normalized_actions:
                        semantic_fallback_actions = self._build_semantic_fallback_actions(
                            agent=agent,
                            semantic_action_lease=semantic_action_lease,
                        )
                        if semantic_fallback_actions:
                            result.locally_planned = True
                            result.actions = semantic_fallback_actions
                            self._consecutive_local_failures = 0
                            self._total_local_steps += 1
                            self._note_semantic_assisted_local_step()
                            self._pending_local_plan_ack = True
                            self._escalation_manager.record_local_success()
                            self._record_step(step_index, result)
                            return result
                        hidden_abort_recovery_actions = self._build_hidden_abort_recovery_actions(
                            agent=agent,
                            semantic_action_lease=semantic_action_lease,
                            allowed_actions=allowed_actions,
                            blocked_actions=blocked_actions,
                        )
                        if hidden_abort_recovery_actions:
                            result.locally_planned = True
                            result.actions = hidden_abort_recovery_actions
                            self._consecutive_local_failures = 0
                            self._total_local_steps += 1
                            self._pending_local_plan_ack = True
                            self._escalation_manager.record_local_success()
                            self._record_step(step_index, result)
                            return result
                        generic_fallback_actions = self._build_generic_exploration_fallback_actions(
                            runtime=runtime,
                            selected_target=base_resolution.selected,
                            agent=agent,
                            allowed_actions=allowed_actions,
                            blocked_actions=blocked_actions,
                            reason="local_action_normalization_failed",
                        )
                        if generic_fallback_actions:
                            result.locally_planned = True
                            result.actions = generic_fallback_actions
                            self._consecutive_local_failures = 0
                            self._total_local_steps += 1
                            self._pending_local_plan_ack = True
                            self._escalation_manager.record_local_success()
                            self._record_step(step_index, result)
                            return result
                        self._consecutive_local_failures += 1
                        self._escalation_manager.record_local_failure()
                        result.needs_analyzer = True
                        result.analyzer_reason = "local_action_normalization_failed"
                        result.sparse_analyzer_input = self._build_sparse_input(
                            compilation, base_resolution, micro_plan, runtime=runtime
                        )
                        self._total_analyzer_calls += 1
                        self._record_step(step_index, result)
                        return result
                    result.locally_planned = True
                    result.actions = normalized_actions
                    self._consecutive_local_failures = 0
                    self._total_local_steps += 1
                    self._total_base_local_steps += 1
                    self._note_semantic_assisted_local_step()
                    self._pending_local_plan_ack = True
                    self._escalation_manager.record_local_success()
                    self._record_step(step_index, result)
                    return result

                semantic_fallback_actions = self._build_semantic_fallback_actions(
                    agent=agent,
                    semantic_action_lease=semantic_action_lease,
                )
                if semantic_fallback_actions:
                    result.locally_planned = True
                    result.actions = semantic_fallback_actions
                    self._consecutive_local_failures = 0
                    self._total_local_steps += 1
                    self._note_semantic_assisted_local_step()
                    self._pending_local_plan_ack = True
                    self._escalation_manager.record_local_success()
                    self._record_step(step_index, result)
                    return result
                hidden_abort_recovery_actions = self._build_hidden_abort_recovery_actions(
                    agent=agent,
                    semantic_action_lease=semantic_action_lease,
                    allowed_actions=allowed_actions,
                    blocked_actions=blocked_actions,
                )
                if hidden_abort_recovery_actions:
                    result.locally_planned = True
                    result.actions = hidden_abort_recovery_actions
                    self._consecutive_local_failures = 0
                    self._total_local_steps += 1
                    self._pending_local_plan_ack = True
                    self._escalation_manager.record_local_success()
                    self._record_step(step_index, result)
                    return result
                generic_fallback_actions = self._build_generic_exploration_fallback_actions(
                    runtime=runtime,
                    selected_target=base_resolution.selected,
                    agent=agent,
                    allowed_actions=allowed_actions,
                    blocked_actions=blocked_actions,
                    reason=micro_plan.analyzer_reason or "base_local_plan_failed",
                )
                if generic_fallback_actions:
                    result.locally_planned = True
                    result.actions = generic_fallback_actions
                    self._consecutive_local_failures = 0
                    self._total_local_steps += 1
                    self._pending_local_plan_ack = True
                    self._escalation_manager.record_local_success()
                    self._record_step(step_index, result)
                    return result
                self._consecutive_local_failures += 1
                self._escalation_manager.record_local_failure()
                result.needs_analyzer = True
                result.analyzer_reason = micro_plan.analyzer_reason or "base_local_plan_failed"
                result.sparse_analyzer_input = self._build_sparse_input(
                    compilation, base_resolution, micro_plan, runtime=runtime
                )
                self._total_analyzer_calls += 1
                self._record_step(step_index, result)
                return result

            semantic_fallback_actions = self._build_semantic_fallback_actions(
                agent=agent,
                semantic_action_lease=semantic_action_lease,
            )
            if semantic_fallback_actions:
                result.locally_planned = True
                result.actions = semantic_fallback_actions
                self._consecutive_local_failures = 0
                self._total_local_steps += 1
                self._note_semantic_assisted_local_step()
                self._pending_local_plan_ack = True
                self._escalation_manager.record_local_success()
                self._record_step(step_index, result)
                return result
            hidden_abort_recovery_actions = self._build_hidden_abort_recovery_actions(
                agent=agent,
                semantic_action_lease=semantic_action_lease,
                allowed_actions=allowed_actions,
                blocked_actions=blocked_actions,
            )
            if hidden_abort_recovery_actions:
                result.locally_planned = True
                result.actions = hidden_abort_recovery_actions
                self._consecutive_local_failures = 0
                self._total_local_steps += 1
                self._pending_local_plan_ack = True
                self._escalation_manager.record_local_success()
                self._record_step(step_index, result)
                return result
            generic_fallback_actions = self._build_generic_exploration_fallback_actions(
                runtime=runtime,
                selected_target=None,
                agent=agent,
                allowed_actions=allowed_actions,
                blocked_actions=blocked_actions,
                reason="no_ir_no_base_candidate",
            )
            if generic_fallback_actions:
                result.locally_planned = True
                result.actions = generic_fallback_actions
                self._consecutive_local_failures = 0
                self._total_local_steps += 1
                self._pending_local_plan_ack = True
                self._escalation_manager.record_local_success()
                self._record_step(step_index, result)
                return result
            result.needs_analyzer = True
            result.analyzer_reason = "no_ir_no_base_candidate"
            result.sparse_analyzer_input = self._build_sparse_input(
                compilation, None, None, runtime=runtime
            )
            self._total_analyzer_calls += 1
            self._record_step(step_index, result)
            return result

        # ── Step 2: Resolve target ──────────────────────────────────────
        resolution = self._resolver.resolve(
            instructions=compilation.instructions,
            evidence=compilation.evidence,
            candidate_pool=candidate_pool,
            target_change_count=target_change_count,
        )
        if not resolution.resolved or resolution.selected is None:
            base_resolution = self._build_base_resolution(
                runtime=runtime,
                evidence=compilation.evidence,
                target_change_count=target_change_count,
            )
            if base_resolution is not None and base_resolution.resolved and base_resolution.selected:
                resolution = base_resolution
        if resolution.resolved and resolution.selected:
            resolution = self._maybe_replace_exhausted_resolution(
                runtime=runtime,
                resolution=resolution,
                evidence=compilation.evidence,
                target_change_count=target_change_count,
            )
        result.resolution = resolution

        if resolution.resolved and resolution.selected:
            result.selected_target = resolution.selected

        # ── Step 3: MicroPlan ───────────────────────────────────────────
        planning_instructions = (
            []
            if str(resolution.resolution_source or "") == "exhausted_target_alternative"
            else compilation.instructions
        )
        micro_plan = self._planner.plan(
            resolution=resolution,
            instructions=planning_instructions,
            allowed_actions=allowed_actions,
            blocked_actions=blocked_actions,
            board_delta=board_delta,
        )
        result.micro_plan = micro_plan

        if micro_plan.planned and micro_plan.actions:
            normalized_actions = self._normalize_planned_actions(
                runtime=runtime,
                actions=micro_plan.actions,
                selected_target=resolution.selected,
                agent=agent,
                semantic_action_lease=semantic_action_lease,
            )
            if not normalized_actions:
                semantic_fallback_actions = self._build_semantic_fallback_actions(
                    agent=agent,
                    semantic_action_lease=semantic_action_lease,
                )
                if semantic_fallback_actions:
                    result.locally_planned = True
                    result.actions = semantic_fallback_actions
                    self._consecutive_local_failures = 0
                    self._total_local_steps += 1
                    self._note_semantic_assisted_local_step()
                    self._pending_local_plan_ack = True
                    self._escalation_manager.record_local_success()
                    self._record_step(step_index, result)
                    return result
                hidden_abort_recovery_actions = self._build_hidden_abort_recovery_actions(
                    agent=agent,
                    semantic_action_lease=semantic_action_lease,
                    allowed_actions=allowed_actions,
                    blocked_actions=blocked_actions,
                )
                if hidden_abort_recovery_actions:
                    result.locally_planned = True
                    result.actions = hidden_abort_recovery_actions
                    self._consecutive_local_failures = 0
                    self._total_local_steps += 1
                    self._pending_local_plan_ack = True
                    self._escalation_manager.record_local_success()
                    self._record_step(step_index, result)
                    return result
                generic_fallback_actions = self._build_generic_exploration_fallback_actions(
                    runtime=runtime,
                    selected_target=resolution.selected,
                    agent=agent,
                    allowed_actions=allowed_actions,
                    blocked_actions=blocked_actions,
                    reason="local_action_normalization_failed",
                )
                if generic_fallback_actions:
                    result.locally_planned = True
                    result.actions = generic_fallback_actions
                    self._consecutive_local_failures = 0
                    self._total_local_steps += 1
                    self._pending_local_plan_ack = True
                    self._escalation_manager.record_local_success()
                    self._record_step(step_index, result)
                    return result
                self._consecutive_local_failures += 1
                self._escalation_manager.record_local_failure()
                result.needs_analyzer = True
                result.analyzer_reason = "local_action_normalization_failed"
                result.sparse_analyzer_input = self._build_sparse_input(
                    compilation, resolution, micro_plan, runtime=runtime
                )
                self._total_analyzer_calls += 1
                self._record_step(step_index, result)
                return result
            # Success: locally planned
            result.locally_planned = True
            result.actions = normalized_actions
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._total_ticket_assisted_local_steps += 1
            self._note_semantic_assisted_local_step()
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result

        # ── Step 4: Fallback to Analyzer ────────────────────────────────
        semantic_fallback_actions = self._build_semantic_fallback_actions(
            agent=agent,
            semantic_action_lease=semantic_action_lease,
        )
        if semantic_fallback_actions:
            result.locally_planned = True
            result.actions = semantic_fallback_actions
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._note_semantic_assisted_local_step()
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result

        hidden_abort_recovery_actions = self._build_hidden_abort_recovery_actions(
            agent=agent,
            semantic_action_lease=semantic_action_lease,
            allowed_actions=allowed_actions,
            blocked_actions=blocked_actions,
        )
        if hidden_abort_recovery_actions:
            result.locally_planned = True
            result.actions = hidden_abort_recovery_actions
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result

        generic_fallback_actions = self._build_generic_exploration_fallback_actions(
            runtime=runtime,
            selected_target=resolution.selected,
            agent=agent,
            allowed_actions=allowed_actions,
            blocked_actions=blocked_actions,
            reason=micro_plan.analyzer_reason or "micro_plan_failed",
        )
        if generic_fallback_actions:
            result.locally_planned = True
            result.actions = generic_fallback_actions
            self._consecutive_local_failures = 0
            self._total_local_steps += 1
            self._pending_local_plan_ack = True
            self._escalation_manager.record_local_success()
            self._record_step(step_index, result)
            return result

        self._consecutive_local_failures += 1
        self._escalation_manager.record_local_failure()

        # Check if we've hit the consecutive failure threshold
        if self._consecutive_local_failures >= self.config.max_consecutive_local_failures:
            analyzer_reason = (
                f"consecutive_local_failures={self._consecutive_local_failures}"
            )
        else:
            analyzer_reason = micro_plan.analyzer_reason or "micro_plan_failed"

        result.needs_analyzer = True
        result.analyzer_reason = analyzer_reason
        result.sparse_analyzer_input = self._build_sparse_input(
            compilation, resolution, micro_plan, runtime=runtime
        )
        self._total_analyzer_calls += 1
        self._record_step(step_index, result)
        return result

    def record_outcome(self, success: bool) -> None:
        """Record the outcome of the last action (local or analyzer).

        Call this after the environment processes the action.
        """
        self._planner.record_outcome(success)
        if self._pending_local_plan_ack:
            if success:
                self._total_local_accepted_steps += 1
            else:
                self._total_local_rejected_steps += 1
            self._pending_local_plan_ack = False
        if success:
            self._consecutive_local_failures = 0

    def _site_actuation_contract_remaining(self) -> list[str]:
        if not self._site_actuation_contract_steps:
            return []
        cursor = max(0, min(self._site_actuation_contract_cursor, len(self._site_actuation_contract_steps)))
        return [
            action_name
            for action_name in self._site_actuation_contract_steps[cursor:]
            if action_name in _MOVE_ACTION_SET
        ]

    def _site_actuation_pending_verify_remaining(self) -> list[str]:
        return [
            str(action_name or "").strip().upper()
            for action_name in list(self._site_actuation_contract_pending_verify_steps or [])
            if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
        ]

    def _exact_site_tx_remaining(self) -> list[str]:
        if not self._exact_site_tx_expected_actions:
            return []
        cursor = max(0, min(self._exact_site_tx_cursor, len(self._exact_site_tx_expected_actions)))
        return [
            str(action_name or "").strip().upper()
            for action_name in self._exact_site_tx_expected_actions[cursor:]
            if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
        ]

    def _site_actuation_contract_has_protected_work(self) -> bool:
        return bool(
            self._site_actuation_contract_remaining()
            or self._site_actuation_pending_verify_remaining()
            or self._exact_site_tx_remaining()
            or self._site_actuation_contract_result_pending
            or self._exact_site_lock_active
        )

    def _site_actuation_contract_has_executable_work(self) -> bool:
        return bool(
            self._site_actuation_contract_remaining()
            or self._site_actuation_pending_verify_remaining()
            or self._exact_site_tx_remaining()
        )

    def _apply_site_actuation_contract_lease(
        self,
        lease: dict[str, Any],
        *,
        event_kind: str,
        priority: float = 1.35,
    ) -> bool:
        remaining = (
            self._site_actuation_contract_remaining()
            or self._site_actuation_pending_verify_remaining()
            or self._exact_site_tx_remaining()
        )
        if not remaining:
            return False
        tx_phase = self._exact_site_tx_semantic_phase() if self._exact_site_tx_active else ""
        sequence_name = str(
            self._site_actuation_contract_sequence_name
            or self._site_actuation_contract_pending_verify_name
            or self._exact_site_lock_sequence_name
            or self._site_actuation_contract_last_completed_sequence_name
            or ""
        ).strip().lower()
        lease["active"] = True
        lease["mechanics_mode"] = lease.get("mechanics_mode") or "semantic"
        lease["semantic_phase"] = tx_phase or "site_actuation"
        lease["site_actuation_active"] = True
        lease["site_actuation_sequence_name"] = sequence_name
        lease["site_actuation_sequence_index"] = int(self._site_actuation_contract_cursor)
        lease["site_actuation_sequence_remaining"] = list(remaining)
        lease["suggested_actions"] = list(remaining)
        lease["route_preview"] = list(remaining)
        lease["lease_horizon"] = max(1, min(3, len(remaining) or 1))
        lease["priority"] = max(float(lease.get("priority", 0.0) or 0.0), float(priority))
        lease["event_kind"] = str(event_kind or "site_actuation_contract_deferred_handoff").strip()
        lease["site_actuation_contract_handoff_deferred"] = True
        lease["site_actuation_contract_active"] = bool(remaining)
        lease["site_actuation_contract_sequence_name"] = str(self._site_actuation_contract_sequence_name or "")
        lease["site_actuation_contract_cursor"] = int(self._site_actuation_contract_cursor)
        lease["site_actuation_contract_remaining"] = list(self._site_actuation_contract_remaining())
        lease["site_actuation_pending_verify_name"] = str(self._site_actuation_contract_pending_verify_name or "")
        lease["site_actuation_pending_verify_remaining"] = list(self._site_actuation_pending_verify_remaining())
        lease["exact_site_tx_active"] = bool(self._exact_site_tx_active)
        lease["exact_site_tx_id"] = int(self._exact_site_tx_id)
        lease["exact_site_tx_phase"] = str(self._exact_site_tx_phase or "")
        lease["exact_site_tx_cursor"] = int(self._exact_site_tx_cursor)
        lease["exact_site_tx_expected_actions"] = list(self._exact_site_tx_expected_actions)
        lease["exact_site_tx_executed_actions"] = list(self._exact_site_tx_executed_actions)
        lease["exact_site_tx_ttl_remaining"] = int(self._exact_site_tx_ttl_remaining)
        lease["exact_site_tx_retry_budget"] = int(self._exact_site_tx_retry_budget)
        return True

    def _exact_site_tx_semantic_phase(self) -> str:
        phase = str(self._exact_site_tx_phase or "").strip().lower()
        if phase == "verify":
            return "site_actuation_verify"
        if phase == "followup":
            return "site_actuation_followup"
        if phase == "contact":
            return "site_actuation_contact"
        return "site_actuation"

    def _activate_exact_site_transaction(
        self,
        *,
        step_index: int,
        sequence_name: str,
        expected_actions: list[str] | None,
        phase: str,
    ) -> None:
        normalized_name = str(sequence_name or "").strip().lower() or "site_actuation_contract"
        normalized_phase = str(phase or "").strip().lower() or "contact"
        normalized_actions = [
            str(action_name or "").strip().upper()
            for action_name in list(expected_actions or [])
            if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
        ]
        is_new_tx = not self._exact_site_tx_active
        if is_new_tx:
            self._exact_site_tx_id += 1
            self._semantic_exact_site_tx_issue_count += 1
            self._exact_site_tx_started_turn = int(step_index)
        elif normalized_phase == "followup" and self._exact_site_tx_phase != "followup":
            self._semantic_followup_enter_count += 1
        self._exact_site_tx_active = True
        self._exact_site_tx_phase = normalized_phase
        self._exact_site_tx_expected_actions = list(normalized_actions)
        self._exact_site_tx_executed_actions = []
        self._exact_site_tx_cursor = 0
        self._exact_site_tx_site_lock_active = bool(self._exact_site_lock_active)
        self._exact_site_tx_ttl_remaining = max(
            self._exact_site_tx_ttl_remaining,
            max(2, len(normalized_actions) + 1),
        )
        self._exact_site_tx_retry_budget = max(self._exact_site_tx_retry_budget, 1)
        self._exact_site_tx_abort_reason = ""
        if normalized_name:
            self._site_actuation_contract_sequence_name = normalized_name

    def _complete_exact_site_transaction(self, *, reason: str = "") -> None:
        if self._exact_site_tx_active:
            self._semantic_exact_site_tx_complete_count += 1
            if self._exact_site_tx_phase == "followup":
                self._semantic_followup_complete_count += 1
        self._exact_site_tx_active = False
        self._exact_site_tx_phase = ""
        self._exact_site_tx_cursor = 0
        self._exact_site_tx_expected_actions = []
        self._exact_site_tx_executed_actions = []
        self._exact_site_tx_site_lock_active = False
        self._exact_site_tx_ttl_remaining = 0
        self._exact_site_tx_retry_budget = 0
        self._exact_site_tx_abort_reason = str(reason or "").strip().lower()
        self._exact_site_tx_started_turn = -1

    def _abort_exact_site_transaction(self, reason: str) -> None:
        normalized_reason = str(reason or "unknown").strip().lower() or "unknown"
        if self._exact_site_tx_active:
            self._semantic_exact_site_tx_abort_count += 1
            self._semantic_exact_site_tx_abort_reason_histogram[normalized_reason] = (
                self._semantic_exact_site_tx_abort_reason_histogram.get(normalized_reason, 0) + 1
            )
        self._exact_site_tx_active = False
        self._exact_site_tx_phase = ""
        self._exact_site_tx_cursor = 0
        self._exact_site_tx_expected_actions = []
        self._exact_site_tx_executed_actions = []
        self._exact_site_tx_site_lock_active = False
        self._exact_site_tx_ttl_remaining = 0
        self._exact_site_tx_retry_budget = 0
        self._exact_site_tx_abort_reason = normalized_reason
        self._exact_site_tx_started_turn = -1

    def _normalize_move_action_name(self, action_name: Any) -> str:
        normalized = str(action_name or "").strip().upper()
        if normalized in _MOVE_ACTION_SET:
            return normalized
        if "." in normalized:
            normalized = normalized.rsplit(".", 1)[-1].strip().upper()
        return normalized if normalized in _MOVE_ACTION_SET else ""

    def _abort_site_actuation_contract(self, reason: str) -> None:
        normalized_reason = str(reason or "unknown").strip().lower() or "unknown"
        self._abort_exact_site_transaction(normalized_reason)
        if self._site_actuation_contract_active or self._site_actuation_contract_armed:
            self._semantic_site_actuation_sequence_abort_count += 1
            self._semantic_site_actuation_sequence_abort_reason_histogram[normalized_reason] = (
                self._semantic_site_actuation_sequence_abort_reason_histogram.get(normalized_reason, 0) + 1
            )
        if self._exact_site_lock_active:
            self._semantic_exact_site_lock_abort_count += 1
            self._semantic_exact_site_lock_abort_reason_histogram[normalized_reason] = (
                self._semantic_exact_site_lock_abort_reason_histogram.get(normalized_reason, 0) + 1
            )
            self._exact_site_lock_active = False
            self._exact_site_lock_sequence_name = ""
            self._exact_site_lock_ttl_remaining = 0
            self._exact_site_lock_started_turn = -1
        self._site_actuation_contract_active = False
        self._site_actuation_contract_armed = False
        self._site_actuation_contract_abort_reason = normalized_reason
        self._site_actuation_contract_sequence_name = ""
        self._site_actuation_contract_steps = []
        self._site_actuation_contract_cursor = 0
        self._site_actuation_contract_started_turn = -1
        self._site_actuation_contract_last_queue_load_turn = -1
        self._site_actuation_contract_last_executed_turn = -1
        self._site_actuation_contract_last_executed_action = ""
        self._site_actuation_contract_result_pending = False

    def _issue_exact_site_lock(
        self,
        *,
        step_index: int,
        sequence_name: str,
        ttl: int,
    ) -> None:
        normalized_name = str(sequence_name or "").strip().lower()
        if not normalized_name:
            return
        ttl = max(1, int(ttl or 0))
        if not self._exact_site_lock_active or self._exact_site_lock_sequence_name != normalized_name:
            self._semantic_exact_site_lock_issue_count += 1
        self._exact_site_lock_active = True
        self._exact_site_lock_sequence_name = normalized_name
        self._exact_site_lock_ttl_remaining = max(self._exact_site_lock_ttl_remaining, ttl)
        self._exact_site_lock_started_turn = int(step_index)
        if self._exact_site_tx_active:
            self._exact_site_tx_site_lock_active = True
            self._exact_site_tx_ttl_remaining = max(self._exact_site_tx_ttl_remaining, ttl)

    def _release_exact_site_lock(
        self,
        *,
        reason: str = "",
        ttl_expired: bool = False,
    ) -> None:
        normalized_reason = str(reason or "").strip().lower()
        if self._exact_site_lock_active and ttl_expired:
            self._semantic_exact_site_lock_ttl_expire_count += 1
        elif self._exact_site_lock_active and normalized_reason:
            self._semantic_exact_site_lock_abort_count += 1
            self._semantic_exact_site_lock_abort_reason_histogram[normalized_reason] = (
                self._semantic_exact_site_lock_abort_reason_histogram.get(normalized_reason, 0) + 1
            )
        self._exact_site_lock_active = False
        self._exact_site_lock_sequence_name = ""
        self._exact_site_lock_ttl_remaining = 0
        self._exact_site_lock_started_turn = -1
        if self._exact_site_tx_active:
            self._exact_site_tx_site_lock_active = False

    def record_site_actuation_queue_load(
        self,
        *,
        step_index: int,
        queued_actions: list[str] | None,
    ) -> None:
        normalized_actions = [
            self._normalize_move_action_name(action_name)
            for action_name in list(queued_actions or [])
            if self._normalize_move_action_name(action_name) in _MOVE_ACTION_SET
        ]
        if not normalized_actions:
            return

        sequence_name = str(self._site_actuation_contract_pending_verify_name or "").strip().lower()
        pending_verify_steps = list(self._site_actuation_contract_pending_verify_steps)
        if not sequence_name:
            sequence_name = str(self._last_semantic_site_actuation_sequence_name or "").strip().lower()
        if not sequence_name and not (
            str(self._last_semantic_phase or "").strip().lower().startswith("site_actuation")
            or self._last_semantic_site_actuation_active
        ):
            return

        if pending_verify_steps and normalized_actions[: len(pending_verify_steps)] == pending_verify_steps[: len(normalized_actions)]:
            sequence_name = str(self._site_actuation_contract_pending_verify_name or sequence_name).strip().lower()

        try:
            last_site_distance = (
                None
                if self._last_semantic_site_distance is None
                else int(self._last_semantic_site_distance)
            )
        except (TypeError, ValueError):
            last_site_distance = None
        stale_verify_tail = bool(
            len(normalized_actions) == 1
            and sequence_name
            and not pending_verify_steps
            and not self._site_actuation_contract_active
            and not self._site_actuation_contract_armed
            and not self._site_actuation_contract_result_pending
            and str(self._last_semantic_phase or "").strip().lower().startswith("site_actuation")
            and last_site_distance is not None
            and last_site_distance >= 12
            and (
                "verify" in sequence_name
                or "probe" in sequence_name
                or "followup" in str(self._exact_site_tx_phase or "").strip().lower()
            )
        )
        if stale_verify_tail:
            self._semantic_site_actuation_stale_tail_queue_skip_count += 1
            return

        current_remaining = self._site_actuation_contract_remaining()
        if (
            self._site_actuation_contract_active
            and self._site_actuation_contract_sequence_name == sequence_name
            and current_remaining
            and (
                normalized_actions == current_remaining[: len(normalized_actions)]
                or normalized_actions == self._site_actuation_contract_steps[: len(normalized_actions)]
            )
        ):
            self._site_actuation_contract_armed = True
            self._site_actuation_contract_last_queue_load_turn = int(step_index)
            self._semantic_site_actuation_sequence_queue_load_count += 1
            if self._exact_site_tx_active:
                self._exact_site_tx_ttl_remaining = max(
                    self._exact_site_tx_ttl_remaining,
                    max(2, len(current_remaining) + 1),
                )
            return

        self._site_actuation_contract_id += 1
        self._site_actuation_contract_active = True
        self._site_actuation_contract_armed = True
        self._site_actuation_contract_sequence_name = sequence_name or "site_actuation_contract"
        self._site_actuation_contract_steps = list(normalized_actions)
        self._site_actuation_contract_cursor = 0
        self._site_actuation_contract_started_turn = int(step_index)
        self._site_actuation_contract_last_queue_load_turn = int(step_index)
        self._site_actuation_contract_last_executed_turn = -1
        self._site_actuation_contract_last_executed_action = ""
        self._site_actuation_contract_abort_reason = ""
        self._site_actuation_contract_result_pending = False
        self._semantic_site_actuation_sequence_queue_load_count += 1
        self._activate_exact_site_transaction(
            step_index=step_index,
            sequence_name=self._site_actuation_contract_sequence_name,
            expected_actions=self._site_actuation_contract_steps,
            phase="verify" if "verify" in self._site_actuation_contract_sequence_name else "contact",
        )

        if "verify" in self._site_actuation_contract_sequence_name:
            self._semantic_site_actuation_verify_issue_count += 1
        if (
            self._site_actuation_contract_pending_verify_name
            and self._site_actuation_contract_sequence_name == self._site_actuation_contract_pending_verify_name
        ):
            self._site_actuation_contract_pending_verify_name = ""
            self._site_actuation_contract_pending_verify_steps = []

    def record_site_actuation_executed_action(
        self,
        *,
        step_index: int,
        action_name: str,
    ) -> None:
        normalized_action = self._normalize_move_action_name(action_name)
        if normalized_action not in _MOVE_ACTION_SET:
            return
        if not (self._site_actuation_contract_active and self._site_actuation_contract_armed):
            return
        if self._site_actuation_contract_cursor >= len(self._site_actuation_contract_steps):
            return

        expected_action = str(
            self._site_actuation_contract_steps[self._site_actuation_contract_cursor] or ""
        ).strip().upper()
        if normalized_action != expected_action:
            self._abort_site_actuation_contract("action_mismatch")
            return

        self._site_actuation_contract_last_executed_turn = int(step_index)
        self._site_actuation_contract_last_executed_action = normalized_action
        self._site_actuation_contract_cursor += 1
        if self._exact_site_tx_active:
            self._exact_site_tx_executed_actions.append(normalized_action)
            self._exact_site_tx_cursor = len(self._exact_site_tx_executed_actions)
            self._exact_site_tx_ttl_remaining = max(0, self._exact_site_tx_ttl_remaining - 1)
        if self._exact_site_lock_active:
            self._semantic_exact_site_lock_consume_count += 1
            self._exact_site_lock_ttl_remaining = max(0, self._exact_site_lock_ttl_remaining - 1)
        self._semantic_site_actuation_sequence_cursor_max = max(
            self._semantic_site_actuation_sequence_cursor_max,
            self._site_actuation_contract_cursor,
        )
        if self._site_actuation_contract_cursor >= len(self._site_actuation_contract_steps):
            self._semantic_site_actuation_sequence_execute_success_count += 1
            if "verify" in str(self._site_actuation_contract_sequence_name or "").strip().lower():
                self._semantic_site_actuation_verify_execute_success_count += 1
            self._site_actuation_contract_active = False
            self._site_actuation_contract_armed = False
            self._site_actuation_contract_result_pending = True
            self._site_actuation_contract_last_completed_sequence_name = (
                self._site_actuation_contract_sequence_name
            )
            self._site_actuation_contract_last_completed_actions = list(
                self._site_actuation_contract_steps
            )
            self._site_actuation_contract_last_completed_turn = int(step_index)
            self._activate_exact_site_transaction(
                step_index=step_index,
                sequence_name=self._site_actuation_contract_last_completed_sequence_name,
                expected_actions=[],
                phase="followup",
            )
            if self._exact_site_lock_active:
                self._exact_site_lock_ttl_remaining = max(self._exact_site_lock_ttl_remaining, 2)
        elif (
            self._exact_site_lock_active
            and self._exact_site_lock_ttl_remaining <= 0
            and not (self._exact_site_tx_active or self._site_actuation_contract_result_pending)
        ):
            self._release_exact_site_lock(ttl_expired=True)

    def _resolve_adapter_sequence_spec(
        self,
        *,
        name: str,
        actions: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        adapter = self._active_semantic_adapter
        if adapter is None or not hasattr(adapter, "resolve_sequence"):
            return None
        try:
            spec = adapter.resolve_sequence(
                name=name,
                actions=actions,
                context=context,
            )
        except Exception:
            return None
        if spec is None:
            return None
        return {
            "name": str(getattr(spec, "name", "") or "").strip().lower(),
            "actions": [
                str(action or "").strip().upper()
                for action in list(getattr(spec, "actions", []) or [])
                if str(action or "").strip().upper() in _MOVE_ACTION_SET
            ],
            "verify_followup_name": str(
                getattr(spec, "verify_followup_name", "") or ""
            ).strip().lower(),
            "verify_followup_actions": [
                str(action or "").strip().upper()
                for action in list(getattr(spec, "verify_followup_actions", []) or [])
                if str(action or "").strip().upper() in _MOVE_ACTION_SET
            ],
            "action_bias": {
                str(action_name or "").strip().upper(): float(weight or 0.0)
                for action_name, weight in dict(getattr(spec, "action_bias", {}) or {}).items()
                if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
            },
            "metadata": dict(getattr(spec, "metadata", {}) or {}),
        }

    def _merge_sequence_route_metadata(
        self,
        lease: dict[str, Any],
        *,
        sequence_spec: dict[str, Any] | None,
        preview_actions: list[str] | None = None,
    ) -> None:
        if not sequence_spec:
            return
        metadata = dict(sequence_spec.get("metadata", {}) or {})
        route_source = str(metadata.get("route_source", "") or "").strip()
        if route_source and not str(lease.get("route_source", "") or "").strip():
            lease["route_source"] = route_source

        if not list(lease.get("route_target_anchor", []) or []):
            route_target_anchor = metadata.get("route_target_anchor")
            if not isinstance(route_target_anchor, (list, tuple)):
                sequence_context = dict(metadata.get("sequence_context", {}) or {})
                route_target_anchor = [
                    sequence_context.get("site_row"),
                    sequence_context.get("site_col"),
                ]
            normalized_target_anchor = [
                int(value)
                for value in list(route_target_anchor or [])
                if str(value).strip("-").isdigit()
            ]
            if len(normalized_target_anchor) == 2:
                lease["route_target_anchor"] = normalized_target_anchor

        if not list(lease.get("route_preview", []) or []):
            route_preview = [
                str(action or "").strip().upper()
                for action in list(metadata.get("route_preview", []) or [])
                if str(action or "").strip().upper() in _MOVE_ACTION_SET
            ]
            if not route_preview:
                route_preview = [
                    str(action or "").strip().upper()
                    for action in list(preview_actions or [])
                    if str(action or "").strip().upper() in _MOVE_ACTION_SET
                ]
            if not route_preview:
                route_preview = [
                    str(action or "").strip().upper()
                    for action in list(sequence_spec.get("actions", []) or [])
                    if str(action or "").strip().upper() in _MOVE_ACTION_SET
                ]
            if route_preview:
                lease["route_preview"] = route_preview

    def _update_site_actuation_contract_result(
        self,
        *,
        step_index: int,
        semantic_phase: str,
        semantic_progress: dict[str, Any],
    ) -> None:
        semantic_phase = str(semantic_phase or "").strip().lower()
        if self._site_actuation_contract_active and not semantic_phase.startswith("site_actuation"):
            self._semantic_site_actuation_phase_downgrade_while_active_count += 1
        if self._exact_site_tx_active and not semantic_phase.startswith("site_actuation"):
            self._semantic_phase_downgrade_while_tx_active_count += 1
        if (
            self._exact_site_tx_active
            and not self._site_actuation_contract_active
            and not self._site_actuation_contract_armed
            and not self._site_actuation_contract_result_pending
            and not self._site_actuation_contract_pending_verify_steps
            and semantic_phase in {"pre_entry", "branch_entry", "site_reaching"}
        ):
            raw_site_distance = semantic_progress.get("site_distance")
            try:
                site_distance = None if raw_site_distance is None else int(raw_site_distance)
            except (TypeError, ValueError):
                site_distance = None
            if semantic_phase == "pre_entry" or site_distance is None or site_distance > 8:
                self._abort_site_actuation_contract("phase_downgrade_far_from_site")
                return

        if not self._site_actuation_contract_result_pending:
            if (
                self._exact_site_tx_active
                and str(self._exact_site_tx_phase or "").strip().lower() == "followup"
                and not self._exact_site_tx_expected_actions
                and not self._site_actuation_contract_pending_verify_steps
            ):
                self._exact_site_tx_ttl_remaining = max(
                    0,
                    int(self._exact_site_tx_ttl_remaining or 0) - 1,
                )
                if self._exact_site_tx_ttl_remaining <= 0:
                    self._record_adapter_site_actuation_outcome(
                        sequence_name=(
                            self._site_actuation_contract_sequence_name
                            or self._last_semantic_site_actuation_sequence_name
                        ),
                        outcome="empty_followup",
                        reason="empty_followup_ttl_expired",
                        site_anchor=list(self._last_semantic_route_target_anchor or []),
                        step_index=int(step_index),
                        block_ttl=30,
                    )
                    self._complete_exact_site_transaction(reason="empty_followup_ttl_expired")
                    self._release_exact_site_lock(ttl_expired=True)
            return

        no_contact = bool(semantic_progress.get("site_actuation_no_contact", False))
        contact = bool(semantic_progress.get("site_actuation_contact_success", False))
        display_change = bool(semantic_progress.get("site_actuation_display_change", False))
        state_change = bool(semantic_progress.get("site_actuation_state_change_success", False))
        verify_success = bool(semantic_progress.get("site_actuation_verify_success", False))

        if not any((no_contact, contact, display_change, state_change, verify_success)):
            if int(step_index) <= int(self._site_actuation_contract_last_completed_turn):
                return
            no_contact = True

        if no_contact and not any((contact, display_change, state_change, verify_success)):
            followup_spec = self._resolve_adapter_sequence_spec(
                name=str(self._site_actuation_contract_last_completed_sequence_name or "").strip().lower(),
                actions=self._site_actuation_contract_last_completed_actions,
                context={"reason": "no_contact_verify_followup"},
            )
            if followup_spec and followup_spec.get("verify_followup_actions"):
                forced_name = str(followup_spec.get("verify_followup_name", "") or "").strip().lower()
                forced_steps = list(followup_spec.get("verify_followup_actions", []) or [])
                self._site_actuation_contract_pending_verify_name = str(forced_name or "").strip().lower()
                self._site_actuation_contract_pending_verify_steps = [
                    str(queued_action or "").strip().upper()
                    for queued_action in list(forced_steps)
                    if str(queued_action or "").strip().upper() in _MOVE_ACTION_SET
                ]
                verify_steps = list(self._site_actuation_contract_pending_verify_steps)
                verify_ttl = max(2, len(verify_steps) + 1)
                verify_lock_ttl = min(3, max(1, len(verify_steps)))
                if verify_steps:
                    self._activate_exact_site_transaction(
                        step_index=step_index,
                        sequence_name=self._site_actuation_contract_pending_verify_name,
                        expected_actions=verify_steps,
                        phase="verify",
                    )
                    self._exact_site_tx_ttl_remaining = max(
                        self._exact_site_tx_ttl_remaining,
                        verify_ttl,
                    )
                    self._exact_site_tx_retry_budget = max(self._exact_site_tx_retry_budget, 1)
                    self._issue_exact_site_lock(
                        step_index=step_index,
                        sequence_name=self._site_actuation_contract_pending_verify_name,
                        ttl=verify_lock_ttl,
                    )
            else:
                self._site_actuation_contract_pending_verify_name = ""
                self._site_actuation_contract_pending_verify_steps = []
                self._complete_exact_site_transaction(reason="no_followup")
                self._release_exact_site_lock()
        elif state_change or verify_success:
            self._site_actuation_contract_pending_verify_name = ""
            self._site_actuation_contract_pending_verify_steps = []
            self._complete_exact_site_transaction(reason="state_change")
            self._release_exact_site_lock()
        elif contact or display_change:
            self._site_actuation_contract_pending_verify_name = ""
            self._site_actuation_contract_pending_verify_steps = []
            if self._exact_site_tx_active:
                self._activate_exact_site_transaction(
                    step_index=step_index,
                    sequence_name=self._site_actuation_contract_last_completed_sequence_name,
                    expected_actions=[],
                    phase="followup",
                )
            if self._exact_site_lock_active:
                self._exact_site_lock_ttl_remaining = max(self._exact_site_lock_ttl_remaining, 1)
        else:
            self._site_actuation_contract_pending_verify_name = ""
            self._site_actuation_contract_pending_verify_steps = []
            self._complete_exact_site_transaction(reason="resolved_without_followup")
            self._release_exact_site_lock()

        if (
            (state_change or verify_success)
            and "verify" in str(self._site_actuation_contract_last_completed_sequence_name or "").strip().lower()
        ):
            self._semantic_site_actuation_verify_state_change_success_count += 1

        self._site_actuation_contract_result_pending = False

    # ── Metrics ─────────────────────────────────────────────────────────

    def step_metrics(self) -> dict[str, Any]:
        """Return Local Controller metrics for the current step."""
        base = self._compiler.step_metrics()
        semantic_match_rate = (
            self._semantic_recommended_action_match_count / self._semantic_recommendation_turn_count
            if self._semantic_recommendation_turn_count > 0 else 0.0
        )
        site_reaching_avg_duration = (
            self._semantic_site_reaching_total_duration / self._semantic_site_reaching_enter_count
            if self._semantic_site_reaching_enter_count > 0 else 0.0
        )
        base["local_controller"] = {
            "enabled": self.config.enabled,
            "consecutive_local_failures": self._consecutive_local_failures,
            "total_local_steps": self._total_local_steps,
            "total_base_local_steps": self._total_base_local_steps,
            "total_ticket_assisted_local_steps": self._total_ticket_assisted_local_steps,
            "total_semantic_assisted_local_steps": self._total_semantic_assisted_local_steps,
            "total_local_accepted_steps": self._total_local_accepted_steps,
            "total_local_rejected_steps": self._total_local_rejected_steps,
            "total_analyzer_calls": self._total_analyzer_calls,
            "stabilize_verify_one_shot_seen_count": self._stabilize_verify_one_shot_seen_count,
            "stabilize_verify_one_shot_selected_count": self._stabilize_verify_one_shot_selected_count,
            "stabilize_verify_one_shot_normalized_count": self._stabilize_verify_one_shot_normalized_count,
            "stabilize_verify_one_shot_blocked_reason_counts": dict(
                self._stabilize_verify_one_shot_blocked_reason_counts
            ),
            "closed_rule_one_shot_seen_count": self._closed_rule_one_shot_seen_count,
            "closed_rule_one_shot_selected_count": self._closed_rule_one_shot_selected_count,
            "closed_rule_one_shot_normalized_count": self._closed_rule_one_shot_normalized_count,
            "closed_rule_one_shot_blocked_reason_counts": dict(
                self._closed_rule_one_shot_blocked_reason_counts
            ),
            "l1_family_starved_count": self._l1_family_starved_count,
            "l1_action_family_quota_issued_count": self._l1_action_family_quota_issued_count,
            "l1_action_family_quota_blocked_count": self._l1_action_family_quota_blocked_count,
            "l1_action_family_quota_action_histogram": dict(
                self._l1_action_family_quota_action_histogram
            ),
            "l1_action_family_quota_reason_histogram": dict(
                self._l1_action_family_quota_reason_histogram
            ),
            "l1_action_family_quota_blocked_reason_counts": dict(
                self._l1_action_family_quota_blocked_reason_counts
            ),
            "l1_lateral_probe_issued_count": self._l1_lateral_probe_issued_count,
            "l1_lateral_probe_remaining": self._l1_lateral_probe_remaining,
            "l1_attempt_outcome_record_count": self._l1_attempt_outcome_record_count,
            "l1_attempt_failed_prefix_count": self._l1_attempt_failed_prefix_count,
            "l1_attempt_relay_anchor_count": self._l1_attempt_relay_anchor_count,
            "l1_attempt_relay_followup_issued_count": (
                self._l1_attempt_relay_followup_issued_count
            ),
            "l1_attempt_relay_followup_score_count": (
                self._l1_attempt_relay_followup_score_count
            ),
            "l1_attempt_relay_followup_gameover_count": (
                self._l1_attempt_relay_followup_gameover_count
            ),
            "l1_attempt_prefix_suppressed_count": self._l1_attempt_prefix_suppressed_count,
            "auxiliary_click_pair_outcome_ranked_count": self._auxiliary_click_pair_outcome_ranked_count,
            "auxiliary_click_pair_vertical_transfer_count": (
                self._auxiliary_click_pair_vertical_transfer_count
            ),
            "auxiliary_click_pair_delta_anchor_count": (
                self._auxiliary_click_pair_delta_anchor_count
            ),
            "auxiliary_click_pair_deferred_unsettled_count": (
                self._auxiliary_click_pair_deferred_unsettled_count
            ),
            "auxiliary_click_pair_settle_tick_count": self._auxiliary_click_pair_settle_tick_count,
            "auxiliary_visual_arrangement_candidate_count": (
                self._auxiliary_visual_arrangement_candidate_count
            ),
            "auxiliary_visual_arrangement_plan_issued_count": (
                self._auxiliary_visual_arrangement_plan_issued_count
            ),
            "auxiliary_visual_arrangement_validate_count": (
                self._auxiliary_visual_arrangement_validate_count
            ),
            "auxiliary_visual_arrangement_direct_issued_count": (
                self._auxiliary_visual_arrangement_direct_issued_count
            ),
            "auxiliary_visual_arrangement_duplicate_suppressed_count": (
                self._auxiliary_visual_arrangement_duplicate_suppressed_count
            ),
            "visual_stencil_first_effect_candidate_count": (
                self._visual_stencil_first_effect_candidate_count
            ),
            "visual_stencil_first_effect_plan_issued_count": (
                self._visual_stencil_first_effect_plan_issued_count
            ),
            "visual_stencil_first_effect_duplicate_suppressed_count": (
                self._visual_stencil_first_effect_duplicate_suppressed_count
            ),
            "visual_stencil_composition_candidate_count": (
                self._visual_stencil_composition_candidate_count
            ),
            "visual_stencil_composition_plan_issued_count": (
                self._visual_stencil_composition_plan_issued_count
            ),
            "visual_stencil_composition_duplicate_suppressed_count": (
                self._visual_stencil_composition_duplicate_suppressed_count
            ),
            "auxiliary_click_pair_reverse_suppressed_count": (
                self._auxiliary_click_pair_reverse_suppressed_count
            ),
            "auxiliary_click_pair_repeat_suppressed_count": (
                self._auxiliary_click_pair_repeat_suppressed_count
            ),
            "pending_local_plan_ack": self._pending_local_plan_ack,
            "micro_planner_stats": self._planner.stats,
            "semantic_bridge": {
                "recommendation_active": self._last_semantic_recommendation_active,
                "mechanics_mode": self._last_semantic_mechanics_mode,
                "phase": self._last_semantic_phase,
                "site_mode": self._last_semantic_site_mode,
                "event_kind": self._last_semantic_event_kind,
                "event_confidence": round(self._last_semantic_event_confidence, 3),
                "lease_age": self._last_semantic_lease_age,
                "suggested_actions": list(self._last_semantic_suggested_actions),
                "route_source": self._current_semantic_route_source,
                "route_target_anchor": list(self._current_semantic_route_target_anchor),
                "route_preview": list(self._current_semantic_route_preview),
                "progress_goal_route_target_anchor": list(
                    self._last_semantic_progress_goal_route_target_anchor
                ),
                "progress_goal_route_preview": list(
                    self._last_semantic_progress_goal_route_preview
                ),
                "hidden_state_route_resync_from_progress": self._last_hidden_state_route_resync_from_progress,
                "do_not_repeat_actions": list(self._last_semantic_do_not_repeat_actions),
                "site_distance": self._last_semantic_site_distance,
                "site_distance_delta": self._last_semantic_site_distance_delta,
                "stagnant_turns": self._last_semantic_stagnant_turns,
                "site_reaching_score": round(self._last_semantic_site_reaching_score, 4),
                "site_zone_confidence": round(self._last_semantic_site_zone_confidence, 4),
                "site_mode_age": self._last_semantic_site_mode_age,
                "site_reaching_current_duration": self._last_semantic_site_reaching_current_duration,
                "site_actuation_active": self._last_semantic_site_actuation_active,
                "site_actuation_age": self._last_semantic_site_actuation_age,
                "site_actuation_sequence_name": self._last_semantic_site_actuation_sequence_name,
                "site_actuation_sequence_remaining": list(self._last_semantic_site_actuation_sequence_remaining),
                "site_actuation_preferred_sequence_name": self._last_semantic_site_actuation_preferred_sequence_name,
                "site_actuation_contact_examples": [dict(example) for example in list(self._last_semantic_site_actuation_contact_examples)],
                "site_actuation_followup_examples": [dict(example) for example in list(self._last_semantic_site_actuation_followup_examples)],
                "selected_action": self._last_semantic_selected_action,
                "recommended_action_selected": self._last_semantic_recommended_action_selected,
                "action_source": self._last_semantic_action_source,
                "recommendation_turn_count": self._semantic_recommendation_turn_count,
                "recommended_action_match_count": self._semantic_recommended_action_match_count,
                "recommended_action_executed_count": self._semantic_recommended_action_executed_count,
                "branch_entry_attempt_count": self._semantic_branch_entry_attempt_count,
                "branch_entry_success_count": self._semantic_branch_entry_success_count,
                "branch_internal_progress_turns": self._semantic_branch_internal_progress_turns,
                "branch_to_site_distance_min": self._semantic_branch_to_site_distance_min,
                "site_reach_attempt_count": self._semantic_site_reach_attempt_count,
                "site_reach_success_count": self._semantic_site_reach_success_count,
                "branch_exit_back_to_corridor_count": self._semantic_branch_exit_back_to_corridor_count,
                "site_reaching_enter_count": self._semantic_site_reaching_enter_count,
                "site_reaching_exit_count": self._semantic_site_reaching_exit_count,
                "site_reaching_avg_duration": round(site_reaching_avg_duration, 3),
                "site_actuation_enter_count": self._semantic_site_actuation_enter_count,
                "site_actuation_attempt_count": self._semantic_site_actuation_attempt_count,
                "site_actuation_success_count": self._semantic_site_actuation_success_count,
                "site_actuation_sequence_issue_count": self._semantic_site_actuation_sequence_issue_count,
                "site_actuation_sequence_consume_count": self._semantic_site_actuation_sequence_consume_count,
                "site_actuation_contact_success_count": self._semantic_site_actuation_contact_success_count,
                "site_actuation_display_change_count": self._semantic_site_actuation_display_change_count,
                "site_actuation_no_contact_count": self._semantic_site_actuation_no_contact_count,
                "site_actuation_state_change_success_count": self._semantic_site_actuation_state_change_success_count,
                "site_actuation_verify_success_count": self._semantic_site_actuation_verify_success_count,
                "site_exit_before_actuation_count": self._semantic_site_exit_before_actuation_count,
                "site_exit_before_verify_count": self._semantic_site_exit_before_verify_count,
                "site_actuation_sequence_histogram": dict(self._semantic_site_actuation_sequence_histogram),
                "site_actuation_display_change_sequence_histogram": dict(self._semantic_site_actuation_display_change_sequence_histogram),
                "site_actuation_no_display_change_sequence_histogram": dict(self._semantic_site_actuation_no_display_change_sequence_histogram),
                "site_actuation_no_contact_sequence_histogram": dict(self._semantic_site_actuation_no_contact_sequence_histogram),
                "site_actuation_contact_position_histogram": dict(self._semantic_site_actuation_contact_position_histogram),
                "site_actuation_contact_pair_histogram": dict(self._semantic_site_actuation_contact_pair_histogram),
                "site_actuation_followup_window_count": self._semantic_site_actuation_followup_window_count,
                "site_actuation_followup_stamp_change_count": self._semantic_site_actuation_followup_stamp_change_count,
                "site_actuation_followup_no_change_count": self._semantic_site_actuation_followup_no_change_count,
                "site_actuation_followup_change_pair_histogram": dict(self._semantic_site_actuation_followup_change_pair_histogram),
                "site_actuation_followup_no_change_pair_histogram": dict(self._semantic_site_actuation_followup_no_change_pair_histogram),
                "site_actuation_reused_sequence_count": self._semantic_site_actuation_reused_sequence_count,
                "site_actuation_sequence_queue_load_count": self._semantic_site_actuation_sequence_queue_load_count,
                "site_actuation_stale_tail_queue_skip_count": self._semantic_site_actuation_stale_tail_queue_skip_count,
                "site_actuation_sequence_execute_success_count": self._semantic_site_actuation_sequence_execute_success_count,
                "site_actuation_controller_adapter_consume_gap_count": max(
                    0,
                    self._semantic_site_actuation_sequence_execute_success_count
                    - self._semantic_site_actuation_sequence_consume_count,
                ),
                "site_actuation_sequence_abort_count": self._semantic_site_actuation_sequence_abort_count,
                "site_actuation_sequence_abort_reason_histogram": dict(self._semantic_site_actuation_sequence_abort_reason_histogram),
                "site_actuation_far_tail_turn_count": self._semantic_site_actuation_far_tail_turn_count,
                "site_actuation_far_tail_sequence_histogram": dict(
                    self._semantic_site_actuation_far_tail_sequence_histogram
                ),
                "site_actuation_far_tail_site_histogram": dict(
                    self._semantic_site_actuation_far_tail_site_histogram
                ),
                "site_actuation_reacquire_candidate_count": (
                    self._semantic_site_actuation_reacquire_candidate_count
                ),
                "site_actuation_reacquire_count": self._semantic_site_actuation_reacquire_count,
                "site_actuation_controller_reset_count": (
                    self._semantic_site_actuation_controller_reset_count
                ),
                "site_actuation_controller_blocked_sequence_histogram": (
                    self._site_actuation_controller_blocked_sequence_histogram()
                ),
                "site_actuation_controller_blocked_sequence_count": (
                    max(
                        self._semantic_site_actuation_controller_blocked_sequence_count,
                        len(self._semantic_site_actuation_controller_reacquire_blocked_sequences),
                    )
                ),
                "site_actuation_controller_far_tail_turn_count": (
                    self._semantic_site_actuation_controller_far_tail_turn_count
                ),
                "site_actuation_controller_far_tail_sequence_histogram": dict(
                    self._semantic_site_actuation_controller_far_tail_sequence_histogram
                ),
                "site_actuation_controller_reacquire_issue_count": (
                    self._semantic_site_actuation_controller_reacquire_issue_count
                ),
                "site_actuation_controller_route_reentry_count": (
                    self._semantic_site_actuation_controller_route_reentry_count
                ),
                "site_actuation_controller_route_reentry_lateral_count": (
                    self._semantic_site_actuation_controller_route_reentry_lateral_count
                ),
                "site_actuation_controller_adapter_notify_count": (
                    self._semantic_site_actuation_controller_adapter_notify_count
                ),
                "site_actuation_controller_adapter_notify_success_count": (
                    self._semantic_site_actuation_controller_adapter_notify_success_count
                ),
                "site_actuation_controller_adapter_notify_block_count": (
                    self._semantic_site_actuation_controller_adapter_notify_block_count
                ),
                "site_closer_outcome_record_count": self._semantic_site_closer_outcome_record_count,
                "site_closer_sequence_blocked_count": self._semantic_site_closer_sequence_blocked_count,
                "site_closer_sequence_rotated_count": self._semantic_site_closer_sequence_rotated_count,
                "site_closer_same_failed_suppressed_count": (
                    self._semantic_site_closer_same_failed_suppressed_count
                ),
                "site_closer_promising_selected_count": self._semantic_site_closer_promising_selected_count,
                "site_closer_closed_count": self._semantic_site_closer_closed_count,
                "site_closer_failed_tail_count": self._semantic_site_closer_failed_tail_count,
                "site_closer_active_block_count": self._semantic_site_closer_active_block_count,
                "site_closer_alternate_pending_count": self._semantic_site_closer_alternate_pending_count,
                "site_closer_alternate_forced_count": self._semantic_site_closer_alternate_forced_count,
                "site_closer_alternate_missing_count": self._semantic_site_closer_alternate_missing_count,
                "site_closer_alternate_expired_count": self._semantic_site_closer_alternate_expired_count,
                "site_closer_outcome_histogram": dict(self._semantic_site_closer_outcome_histogram),
                "site_closer_sequence_outcome_histogram": dict(
                    self._semantic_site_closer_sequence_outcome_histogram
                ),
                "site_closer_blocked_sequence_histogram": dict(
                    self._semantic_site_closer_blocked_sequence_histogram
                ),
                "site_actuation_phase_downgrade_while_active_count": self._semantic_site_actuation_phase_downgrade_while_active_count,
                "site_actuation_sequence_cursor_max": self._semantic_site_actuation_sequence_cursor_max,
                "site_actuation_verify_issue_count": self._semantic_site_actuation_verify_issue_count,
                "site_actuation_contract_active": self._site_actuation_contract_active,
                "site_actuation_contract_sequence_name": self._site_actuation_contract_sequence_name,
                "site_actuation_contract_cursor": self._site_actuation_contract_cursor,
                "site_actuation_contract_remaining": list(self._site_actuation_contract_remaining()),
                "site_actuation_contract_pending_verify_name": self._site_actuation_contract_pending_verify_name,
                "exact_site_tx_active": self._exact_site_tx_active,
                "exact_site_tx_id": self._exact_site_tx_id,
                "exact_site_tx_phase": self._exact_site_tx_phase,
                "exact_site_tx_cursor": self._exact_site_tx_cursor,
                "exact_site_tx_expected_actions": list(self._exact_site_tx_expected_actions),
                "exact_site_tx_executed_actions": list(self._exact_site_tx_executed_actions),
                "exact_site_tx_ttl_remaining": self._exact_site_tx_ttl_remaining,
                "exact_site_tx_retry_budget": self._exact_site_tx_retry_budget,
                "exact_site_tx_abort_reason": self._exact_site_tx_abort_reason,
                "exact_site_tx_issue_count": self._semantic_exact_site_tx_issue_count,
                "exact_site_tx_complete_count": self._semantic_exact_site_tx_complete_count,
                "exact_site_tx_abort_count": self._semantic_exact_site_tx_abort_count,
                "exact_site_tx_abort_reason_histogram": dict(self._semantic_exact_site_tx_abort_reason_histogram),
                "phase_downgrade_while_tx_active_count": self._semantic_phase_downgrade_while_tx_active_count,
                "reseed_while_tx_active_count": self._semantic_reseed_while_tx_active_count,
                "followup_enter_count": self._semantic_followup_enter_count,
                "followup_complete_count": self._semantic_followup_complete_count,
                "exact_site_lock_active": self._exact_site_lock_active,
                "exact_site_lock_sequence_name": self._exact_site_lock_sequence_name,
                "exact_site_lock_ttl_remaining": self._exact_site_lock_ttl_remaining,
                "exact_site_lock_issue_count": self._semantic_exact_site_lock_issue_count,
                "exact_site_lock_consume_count": self._semantic_exact_site_lock_consume_count,
                "exact_site_lock_ttl_expire_count": self._semantic_exact_site_lock_ttl_expire_count,
                "exact_site_lock_abort_count": self._semantic_exact_site_lock_abort_count,
                "exact_site_lock_abort_reason_histogram": dict(self._semantic_exact_site_lock_abort_reason_histogram),
                "phase_downgrade_while_exact_site_lock_count": self._semantic_phase_downgrade_while_exact_site_lock_count,
                "site_actuation_verify_execute_success_count": self._semantic_site_actuation_verify_execute_success_count,
                "site_actuation_verify_state_change_success_count": self._semantic_site_actuation_verify_state_change_success_count,
                "route_recommendation_count": self._semantic_route_recommendation_count,
                "route_source_histogram": dict(self._semantic_route_source_histogram),
                "last_route_source": self._last_semantic_route_source,
                "last_route_target_anchor": list(self._last_semantic_route_target_anchor),
                "last_route_preview": list(self._last_semantic_route_preview),
                "suggested_action_match_rate": round(semantic_match_rate, 4),
            },
            "hidden_state_executor": self._hidden_state_executor_metrics(),
        }
        base["escalation"] = self._escalation_manager.step_metrics()
        return base

    def episode_summary(self) -> dict[str, Any]:
        """Return Local Controller section for episode summary."""
        base = self._compiler.episode_summary()
        total = self._total_local_steps + self._total_analyzer_calls
        semantic_match_rate = (
            self._semantic_recommended_action_match_count / self._semantic_recommendation_turn_count
            if self._semantic_recommendation_turn_count > 0 else 0.0
        )
        site_reaching_avg_duration = (
            self._semantic_site_reaching_total_duration / self._semantic_site_reaching_enter_count
            if self._semantic_site_reaching_enter_count > 0 else 0.0
        )
        base["local_controller_summary"] = {
            "total_local_steps": self._total_local_steps,
            "total_base_local_steps": self._total_base_local_steps,
            "total_ticket_assisted_local_steps": self._total_ticket_assisted_local_steps,
            "total_semantic_assisted_local_steps": self._total_semantic_assisted_local_steps,
            "total_local_accepted_steps": self._total_local_accepted_steps,
            "total_local_rejected_steps": self._total_local_rejected_steps,
            "total_analyzer_calls": self._total_analyzer_calls,
            "stabilize_verify_one_shot_seen_count": self._stabilize_verify_one_shot_seen_count,
            "stabilize_verify_one_shot_selected_count": self._stabilize_verify_one_shot_selected_count,
            "stabilize_verify_one_shot_normalized_count": self._stabilize_verify_one_shot_normalized_count,
            "stabilize_verify_one_shot_blocked_reason_counts": dict(
                self._stabilize_verify_one_shot_blocked_reason_counts
            ),
            "closed_rule_one_shot_seen_count": self._closed_rule_one_shot_seen_count,
            "closed_rule_one_shot_selected_count": self._closed_rule_one_shot_selected_count,
            "closed_rule_one_shot_normalized_count": self._closed_rule_one_shot_normalized_count,
            "closed_rule_one_shot_blocked_reason_counts": dict(
                self._closed_rule_one_shot_blocked_reason_counts
            ),
            "l1_family_starved_count": self._l1_family_starved_count,
            "l1_action_family_quota_issued_count": self._l1_action_family_quota_issued_count,
            "l1_action_family_quota_blocked_count": self._l1_action_family_quota_blocked_count,
            "l1_action_family_quota_action_histogram": dict(
                self._l1_action_family_quota_action_histogram
            ),
            "l1_action_family_quota_reason_histogram": dict(
                self._l1_action_family_quota_reason_histogram
            ),
            "l1_action_family_quota_blocked_reason_counts": dict(
                self._l1_action_family_quota_blocked_reason_counts
            ),
            "l1_lateral_probe_issued_count": self._l1_lateral_probe_issued_count,
            "l1_lateral_probe_remaining": self._l1_lateral_probe_remaining,
            "l1_attempt_outcome_record_count": self._l1_attempt_outcome_record_count,
            "l1_attempt_failed_prefix_count": self._l1_attempt_failed_prefix_count,
            "l1_attempt_relay_anchor_count": self._l1_attempt_relay_anchor_count,
            "l1_attempt_relay_followup_issued_count": (
                self._l1_attempt_relay_followup_issued_count
            ),
            "l1_attempt_relay_followup_score_count": (
                self._l1_attempt_relay_followup_score_count
            ),
            "l1_attempt_relay_followup_gameover_count": (
                self._l1_attempt_relay_followup_gameover_count
            ),
            "l1_attempt_prefix_suppressed_count": self._l1_attempt_prefix_suppressed_count,
            "auxiliary_click_pair_outcome_ranked_count": self._auxiliary_click_pair_outcome_ranked_count,
            "auxiliary_click_pair_vertical_transfer_count": (
                self._auxiliary_click_pair_vertical_transfer_count
            ),
            "auxiliary_click_pair_delta_anchor_count": (
                self._auxiliary_click_pair_delta_anchor_count
            ),
            "auxiliary_click_pair_deferred_unsettled_count": (
                self._auxiliary_click_pair_deferred_unsettled_count
            ),
            "auxiliary_click_pair_settle_tick_count": self._auxiliary_click_pair_settle_tick_count,
            "auxiliary_visual_arrangement_candidate_count": (
                self._auxiliary_visual_arrangement_candidate_count
            ),
            "auxiliary_visual_arrangement_plan_issued_count": (
                self._auxiliary_visual_arrangement_plan_issued_count
            ),
            "auxiliary_visual_arrangement_validate_count": (
                self._auxiliary_visual_arrangement_validate_count
            ),
            "auxiliary_visual_arrangement_direct_issued_count": (
                self._auxiliary_visual_arrangement_direct_issued_count
            ),
            "auxiliary_visual_arrangement_duplicate_suppressed_count": (
                self._auxiliary_visual_arrangement_duplicate_suppressed_count
            ),
            "visual_stencil_first_effect_candidate_count": (
                self._visual_stencil_first_effect_candidate_count
            ),
            "visual_stencil_first_effect_plan_issued_count": (
                self._visual_stencil_first_effect_plan_issued_count
            ),
            "visual_stencil_first_effect_duplicate_suppressed_count": (
                self._visual_stencil_first_effect_duplicate_suppressed_count
            ),
            "visual_stencil_composition_candidate_count": (
                self._visual_stencil_composition_candidate_count
            ),
            "visual_stencil_composition_plan_issued_count": (
                self._visual_stencil_composition_plan_issued_count
            ),
            "visual_stencil_composition_duplicate_suppressed_count": (
                self._visual_stencil_composition_duplicate_suppressed_count
            ),
            "auxiliary_click_pair_reverse_suppressed_count": (
                self._auxiliary_click_pair_reverse_suppressed_count
            ),
            "auxiliary_click_pair_repeat_suppressed_count": (
                self._auxiliary_click_pair_repeat_suppressed_count
            ),
            "local_step_rate": (
                self._total_local_steps / total if total > 0 else 0.0
            ),
            "ticketless_local_step_rate": (
                self._total_base_local_steps / total if total > 0 else 0.0
            ),
            "ticket_assisted_local_step_rate": (
                self._total_ticket_assisted_local_steps / total if total > 0 else 0.0
            ),
            "semantic_assisted_local_step_rate": (
                self._total_semantic_assisted_local_steps / total if total > 0 else 0.0
            ),
            "accepted_local_step_rate": (
                self._total_local_accepted_steps / total if total > 0 else 0.0
            ),
            "analyzer_call_rate": (
                self._total_analyzer_calls / total if total > 0 else 0.0
            ),
            "micro_planner_stats": self._planner.stats,
            "semantic_bridge_summary": {
                "recommendation_turn_count": self._semantic_recommendation_turn_count,
                "recommended_action_match_count": self._semantic_recommended_action_match_count,
                "recommended_action_executed_count": self._semantic_recommended_action_executed_count,
                "branch_entry_attempt_count": self._semantic_branch_entry_attempt_count,
                "branch_entry_success_count": self._semantic_branch_entry_success_count,
                "branch_internal_progress_turns": self._semantic_branch_internal_progress_turns,
                "branch_to_site_distance_min": self._semantic_branch_to_site_distance_min,
                "site_reach_attempt_count": self._semantic_site_reach_attempt_count,
                "site_reach_success_count": self._semantic_site_reach_success_count,
                "branch_exit_back_to_corridor_count": self._semantic_branch_exit_back_to_corridor_count,
                "site_reaching_enter_count": self._semantic_site_reaching_enter_count,
                "site_reaching_exit_count": self._semantic_site_reaching_exit_count,
                "site_reaching_avg_duration": round(site_reaching_avg_duration, 3),
                "site_actuation_enter_count": self._semantic_site_actuation_enter_count,
                "site_actuation_attempt_count": self._semantic_site_actuation_attempt_count,
                "site_actuation_success_count": self._semantic_site_actuation_success_count,
                "site_actuation_sequence_issue_count": self._semantic_site_actuation_sequence_issue_count,
                "site_actuation_sequence_consume_count": self._semantic_site_actuation_sequence_consume_count,
                "site_actuation_contact_success_count": self._semantic_site_actuation_contact_success_count,
                "site_actuation_display_change_count": self._semantic_site_actuation_display_change_count,
                "site_actuation_no_contact_count": self._semantic_site_actuation_no_contact_count,
                "site_actuation_state_change_success_count": self._semantic_site_actuation_state_change_success_count,
                "site_actuation_verify_success_count": self._semantic_site_actuation_verify_success_count,
                "site_exit_before_actuation_count": self._semantic_site_exit_before_actuation_count,
                "site_exit_before_verify_count": self._semantic_site_exit_before_verify_count,
                "site_actuation_sequence_histogram": dict(self._semantic_site_actuation_sequence_histogram),
                "site_actuation_display_change_sequence_histogram": dict(self._semantic_site_actuation_display_change_sequence_histogram),
                "site_actuation_no_display_change_sequence_histogram": dict(self._semantic_site_actuation_no_display_change_sequence_histogram),
                "site_actuation_no_contact_sequence_histogram": dict(self._semantic_site_actuation_no_contact_sequence_histogram),
                "site_actuation_contact_position_histogram": dict(self._semantic_site_actuation_contact_position_histogram),
                "site_actuation_contact_pair_histogram": dict(self._semantic_site_actuation_contact_pair_histogram),
                "site_actuation_followup_window_count": self._semantic_site_actuation_followup_window_count,
                "site_actuation_followup_stamp_change_count": self._semantic_site_actuation_followup_stamp_change_count,
                "site_actuation_followup_no_change_count": self._semantic_site_actuation_followup_no_change_count,
                "site_actuation_followup_change_pair_histogram": dict(self._semantic_site_actuation_followup_change_pair_histogram),
                "site_actuation_followup_no_change_pair_histogram": dict(self._semantic_site_actuation_followup_no_change_pair_histogram),
                "site_actuation_reused_sequence_count": self._semantic_site_actuation_reused_sequence_count,
                "site_actuation_sequence_queue_load_count": self._semantic_site_actuation_sequence_queue_load_count,
                "site_actuation_stale_tail_queue_skip_count": self._semantic_site_actuation_stale_tail_queue_skip_count,
                "site_actuation_sequence_execute_success_count": self._semantic_site_actuation_sequence_execute_success_count,
                "site_actuation_controller_adapter_consume_gap_count": max(
                    0,
                    self._semantic_site_actuation_sequence_execute_success_count
                    - self._semantic_site_actuation_sequence_consume_count,
                ),
                "site_actuation_sequence_abort_count": self._semantic_site_actuation_sequence_abort_count,
                "site_actuation_sequence_abort_reason_histogram": dict(self._semantic_site_actuation_sequence_abort_reason_histogram),
                "site_actuation_far_tail_turn_count": self._semantic_site_actuation_far_tail_turn_count,
                "site_actuation_far_tail_sequence_histogram": dict(
                    self._semantic_site_actuation_far_tail_sequence_histogram
                ),
                "site_actuation_far_tail_site_histogram": dict(
                    self._semantic_site_actuation_far_tail_site_histogram
                ),
                "site_actuation_reacquire_candidate_count": (
                    self._semantic_site_actuation_reacquire_candidate_count
                ),
                "site_actuation_reacquire_count": self._semantic_site_actuation_reacquire_count,
                "site_actuation_controller_reset_count": (
                    self._semantic_site_actuation_controller_reset_count
                ),
                "site_actuation_controller_blocked_sequence_histogram": (
                    self._site_actuation_controller_blocked_sequence_histogram()
                ),
                "site_actuation_controller_blocked_sequence_count": (
                    max(
                        self._semantic_site_actuation_controller_blocked_sequence_count,
                        len(self._semantic_site_actuation_controller_reacquire_blocked_sequences),
                    )
                ),
                "site_actuation_controller_far_tail_turn_count": (
                    self._semantic_site_actuation_controller_far_tail_turn_count
                ),
                "site_actuation_controller_far_tail_sequence_histogram": dict(
                    self._semantic_site_actuation_controller_far_tail_sequence_histogram
                ),
                "site_actuation_controller_reacquire_issue_count": (
                    self._semantic_site_actuation_controller_reacquire_issue_count
                ),
                "site_actuation_controller_route_reentry_count": (
                    self._semantic_site_actuation_controller_route_reentry_count
                ),
                "site_actuation_controller_route_reentry_lateral_count": (
                    self._semantic_site_actuation_controller_route_reentry_lateral_count
                ),
                "site_actuation_controller_adapter_notify_count": (
                    self._semantic_site_actuation_controller_adapter_notify_count
                ),
                "site_actuation_controller_adapter_notify_success_count": (
                    self._semantic_site_actuation_controller_adapter_notify_success_count
                ),
                "site_actuation_controller_adapter_notify_block_count": (
                    self._semantic_site_actuation_controller_adapter_notify_block_count
                ),
                "site_closer_outcome_record_count": self._semantic_site_closer_outcome_record_count,
                "site_closer_sequence_blocked_count": self._semantic_site_closer_sequence_blocked_count,
                "site_closer_sequence_rotated_count": self._semantic_site_closer_sequence_rotated_count,
                "site_closer_same_failed_suppressed_count": (
                    self._semantic_site_closer_same_failed_suppressed_count
                ),
                "site_closer_promising_selected_count": self._semantic_site_closer_promising_selected_count,
                "site_closer_closed_count": self._semantic_site_closer_closed_count,
                "site_closer_failed_tail_count": self._semantic_site_closer_failed_tail_count,
                "site_closer_active_block_count": self._semantic_site_closer_active_block_count,
                "site_closer_alternate_pending_count": self._semantic_site_closer_alternate_pending_count,
                "site_closer_alternate_forced_count": self._semantic_site_closer_alternate_forced_count,
                "site_closer_alternate_missing_count": self._semantic_site_closer_alternate_missing_count,
                "site_closer_alternate_expired_count": self._semantic_site_closer_alternate_expired_count,
                "site_closer_outcome_histogram": dict(self._semantic_site_closer_outcome_histogram),
                "site_closer_sequence_outcome_histogram": dict(
                    self._semantic_site_closer_sequence_outcome_histogram
                ),
                "site_closer_blocked_sequence_histogram": dict(
                    self._semantic_site_closer_blocked_sequence_histogram
                ),
                "site_actuation_phase_downgrade_while_active_count": self._semantic_site_actuation_phase_downgrade_while_active_count,
                "site_actuation_sequence_cursor_max": self._semantic_site_actuation_sequence_cursor_max,
                "site_actuation_verify_issue_count": self._semantic_site_actuation_verify_issue_count,
                "exact_site_tx_issue_count": self._semantic_exact_site_tx_issue_count,
                "exact_site_tx_complete_count": self._semantic_exact_site_tx_complete_count,
                "exact_site_tx_abort_count": self._semantic_exact_site_tx_abort_count,
                "exact_site_tx_abort_reason_histogram": dict(self._semantic_exact_site_tx_abort_reason_histogram),
                "phase_downgrade_while_tx_active_count": self._semantic_phase_downgrade_while_tx_active_count,
                "reseed_while_tx_active_count": self._semantic_reseed_while_tx_active_count,
                "followup_enter_count": self._semantic_followup_enter_count,
                "followup_complete_count": self._semantic_followup_complete_count,
                "exact_site_lock_issue_count": self._semantic_exact_site_lock_issue_count,
                "exact_site_lock_consume_count": self._semantic_exact_site_lock_consume_count,
                "exact_site_lock_ttl_expire_count": self._semantic_exact_site_lock_ttl_expire_count,
                "exact_site_lock_abort_count": self._semantic_exact_site_lock_abort_count,
                "exact_site_lock_abort_reason_histogram": dict(self._semantic_exact_site_lock_abort_reason_histogram),
                "phase_downgrade_while_exact_site_lock_count": self._semantic_phase_downgrade_while_exact_site_lock_count,
                "site_actuation_verify_execute_success_count": self._semantic_site_actuation_verify_execute_success_count,
                "site_actuation_verify_state_change_success_count": self._semantic_site_actuation_verify_state_change_success_count,
                "route_recommendation_count": self._semantic_route_recommendation_count,
                "route_source_histogram": dict(self._semantic_route_source_histogram),
                "last_route_source": self._last_semantic_route_source,
                "last_route_target_anchor": list(self._last_semantic_route_target_anchor),
                "last_route_preview": list(self._last_semantic_route_preview),
                "suggested_action_match_rate": round(semantic_match_rate, 4),
            },
            "hidden_state_executor_summary": self._hidden_state_executor_metrics(),
        }
        base["escalation_summary"] = self._escalation_manager.episode_summary()
        return base

    def advisory_text(self) -> str:
        """Return advisory text (backward compat)."""
        return self._compiler.advisory_text()

    def get_recommended_components(self) -> list[str]:
        """Return recommended component IDs (backward compat)."""
        return self._compiler.get_recommended_components()

    # ── Private: Build sparse analyzer input ────────────────────────────

    def _build_sparse_input(
        self,
        compilation: CompilationResult,
        resolution: ResolutionResult | None,
        micro_plan: MicroPlanResult | None,
        *,
        runtime: Any | None = None,
    ) -> dict[str, Any]:
        """Build compact input for the Analyzer (sparse replanner mode).

        Instead of the full prose advisory, we send:
        - active tickets (compact)
        - top candidate summary
        - last failure reason
        - GCSTM one-way evidence
        - PSA subtype tag
        """
        sparse: dict[str, Any] = {}

        # Active tickets
        if compilation.instructions:
            sparse["active_tickets"] = [
                ir.to_dict() for ir in compilation.instructions
            ]

        # Top candidate
        if resolution and resolution.selected:
            sparse["top_candidate"] = resolution.selected.to_dict()
        elif resolution and resolution.all_scored:
            sparse["top_candidate"] = resolution.all_scored[0].to_dict()

        # Evidence snapshot
        ev = compilation.evidence
        if ev:
            sparse["evidence_snapshot"] = {
                "selected_target_empty": ev.selected_target_empty,
                "candidate_pool_size": ev.candidate_pool_size,
                "recently_live_anchor": ev.recently_live_anchor,
                "recently_live_anchor_family": ev.recently_live_anchor_family,
                "anchor_stability": round(ev.anchor_stability, 4),
                "shared_rearm_live": ev.shared_rearm_live,
                "queue_exhaustion_detected": ev.queue_exhaustion_detected,
                "psa_subtype": ev.psa_subtype,
            }

        if runtime is not None:
            aftercare_signal = self._runtime_aftercare_signal(runtime)
            aftercare_snapshot = {
                "protected_hedge_visible_signal": bool(
                    aftercare_signal.get("protected_hedge_visible_turn_count", 0)
                ),
                "protected_hedge_visible_turn_count": int(
                    aftercare_signal.get("protected_hedge_visible_turn_count", 0) or 0
                ),
                "protected_hedge_recent_visible": bool(
                    aftercare_signal.get("protected_hedge_recent_visible", False)
                ),
                "protected_hedge_claim_state": str(
                    aftercare_signal.get("protected_hedge_claim_state", "") or ""
                ),
                "protected_hedge_claim_active": bool(
                    aftercare_signal.get("protected_hedge_claim_active", False)
                ),
                "post_reclaim_hold_signal": bool(
                    aftercare_signal.get("post_reclaim_hold_count", 0)
                ),
                "post_reclaim_hold_count": int(
                    aftercare_signal.get("post_reclaim_hold_count", 0) or 0
                ),
                "post_reclaim_hold_ready_source": str(
                    aftercare_signal.get("post_reclaim_hold_ready_source", "") or ""
                ),
                "post_final_shared_entry_count": int(
                    aftercare_signal.get("post_final_shared_entry_count", 0) or 0
                ),
            }
            if any(aftercare_snapshot.values()):
                sparse["aftercare_snapshot"] = aftercare_snapshot

        # Failure reason
        if micro_plan and micro_plan.analyzer_reason:
            sparse["last_failure_reason"] = micro_plan.analyzer_reason
        elif compilation.fallback_reason:
            sparse["last_failure_reason"] = compilation.fallback_reason

        return sparse

    # ── Private: Step recording ─────────────────────────────────────────

    def _record_step(self, step_index: int, result: LocalControllerResult) -> None:
        """Record step for history (limited to last 20 steps)."""
        self._maybe_expand_final_click_only_grid_probe_result(result)
        if self.config.verbose_metrics:
            entry = {
                "step": step_index,
                "locally_planned": result.locally_planned,
                "needs_analyzer": result.needs_analyzer,
                "analyzer_reason": result.analyzer_reason,
                "action_count": len(result.actions),
            }
            self._step_history.append(entry)
            # Keep only last 20
            if len(self._step_history) > 20:
                self._step_history = self._step_history[-20:]

    def _maybe_expand_final_click_only_grid_probe_result(
        self,
        result: LocalControllerResult,
    ) -> None:
        if self._move_action_available or len(result.actions) != 1:
            return
        selected = result.selected_target
        if selected is None and result.resolution is not None:
            selected = result.resolution.selected
        if selected is None:
            return
        source = str(getattr(selected, "source", "") or "").strip()
        if source not in {"grid_scan", "level_rule_symbol_scout"}:
            return
        action = result.actions[0]
        if str(action.action_name or "").strip().upper() != "ACTION6":
            return
        key = str(getattr(selected, "anchor_key", "") or getattr(selected, "component_id", "") or "").strip()
        if not key:
            return
        issued_count = int(self._click_only_grid_probe_packet_counts.get(key, 0) or 0)
        if issued_count >= 5:
            if self._click_only_grid_probe_focus_key == key:
                self._click_only_grid_probe_focus_key = ""
            return
        self._click_only_grid_probe_packet_counts[key] = issued_count + 1
        self._click_only_grid_probe_focus_key = key
        self._click_only_grid_probe_focus_score = 0
        self._click_only_grid_probe_focus_level = 0
        self._click_only_grid_probe_focus_attempt = 0
        result.actions = [
            action,
            PlannedAction(
                action_name=action.action_name,
                target_component_id=action.target_component_id,
                parameters=dict(action.parameters),
                reason=(action.reason or "click_only_grid_probe_packet") + ":repeat",
            ),
        ]

    def _build_click_only_grid_probe_focus_actions(
        self,
        *,
        runtime: Any,
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
    ) -> list[PlannedAction]:
        if self._move_action_available or not self._click_only_grid_probe_focus_key:
            return []
        available = self._normalized_available_actions(allowed_actions, blocked_actions)
        if "ACTION6" not in available:
            return []
        key = str(self._click_only_grid_probe_focus_key or "")
        issued_count = int(self._click_only_grid_probe_packet_counts.get(key, 0) or 0)
        if issued_count >= 5:
            self._click_only_grid_probe_focus_key = ""
            return []
        score = self._runtime_score(runtime)
        level = self._runtime_level_number(runtime)
        attempt = int(getattr(runtime, "last_arc_attempt_number", 0) or 0)
        if self._click_only_grid_probe_focus_score and score != self._click_only_grid_probe_focus_score:
            self._click_only_grid_probe_focus_key = ""
            return []
        if self._click_only_grid_probe_focus_level and level != self._click_only_grid_probe_focus_level:
            self._click_only_grid_probe_focus_key = ""
            return []
        if self._click_only_grid_probe_focus_attempt and attempt != self._click_only_grid_probe_focus_attempt:
            self._click_only_grid_probe_focus_key = ""
            return []
        parts = key.split("||")
        if len(parts) != 3 or parts[0] != "grid":
            self._click_only_grid_probe_focus_key = ""
            return []
        try:
            row = int(parts[1])
            col = int(parts[2])
        except (TypeError, ValueError):
            self._click_only_grid_probe_focus_key = ""
            return []
        self._click_only_grid_probe_packet_counts[key] = issued_count + 1
        return [
            PlannedAction(
                action_name="ACTION6",
                parameters={"x": row, "y": col},
                reason="click_only_grid_probe_focus",
            ),
            PlannedAction(
                action_name="ACTION6",
                parameters={"x": row, "y": col},
                reason="click_only_grid_probe_focus:repeat",
            ),
        ]

    def _infer_current_family(
        self,
        runtime: Any,
        compilation: CompilationResult,
        base_pool: Any | None,
    ) -> str:
        if compilation.instructions:
            family = str(compilation.instructions[0].target_action_family or "").strip()
            if family:
                return family

        selected = getattr(runtime, "last_selected_target_summary", None)
        if isinstance(selected, dict):
            family = str(
                selected.get("action_family", "")
                or selected.get("family", "")
                or ""
            ).strip()
            if family:
                return family

        candidates = getattr(base_pool, "candidates", None) or []
        for candidate in candidates:
            family = str(getattr(candidate, "family", "") or "").strip()
            if family:
                return family
        return ""

    def _runtime_aftercare_signal(self, runtime: Any) -> dict[str, Any]:
        current_step_index = int(getattr(getattr(runtime, "last_state", None), "step_index", -10_000) or -10_000)
        visible_turn_count = int(
            getattr(runtime, "protected_hedge_visible_turn_count", 0) or 0
        )
        post_reclaim_hold_count = int(
            getattr(runtime, "post_reclaim_hold_count", 0) or 0
        )
        post_final_shared_entry_count = int(
            getattr(runtime, "post_final_shared_entry_count", 0) or 0
        )

        claim_state = ""
        contract_visible = False
        claim_active = False
        try:
            claim_state = str(getattr(runtime, "_protected_hedge_runtime_claim_state")() or "")
        except Exception:
            claim_state = ""
        try:
            contract_visible = bool(getattr(runtime, "_protected_hedge_contract_visible")())
        except Exception:
            contract_visible = False
        try:
            claim_active = bool(getattr(runtime, "_protected_hedge_runtime_claim_active")())
        except Exception:
            claim_active = False
        recent_last_visible_turn = int(
            getattr(runtime, "protected_hedge_last_visible_turn", -10_000) or -10_000
        )
        recent_hedge_visible = bool(
            current_step_index >= 0
            and recent_last_visible_turn >= current_step_index - 24
        )

        live_hedge_signal = bool(
            visible_turn_count > 0
            or getattr(runtime, "last_protected_hedge_visible", False)
            or contract_visible
            or claim_active
            or recent_hedge_visible
            or claim_state
            in {
                "CLAIMED",
                "RECLAIMABLE",
                "POST_RECLAIM_HOLD",
            }
        )
        if visible_turn_count <= 0 and live_hedge_signal:
            visible_turn_count = 1

        live_reclaim_signal = bool(
            post_reclaim_hold_count > 0
            or claim_state == "POST_RECLAIM_HOLD"
            or bool(getattr(runtime, "last_post_reclaim_hold_ready_source", ""))
        )
        if post_reclaim_hold_count <= 0 and live_reclaim_signal:
            post_reclaim_hold_count = 1

        live_shared_signal = bool(
            post_final_shared_entry_count > 0
            or claim_state == "SHARED"
        )
        if post_final_shared_entry_count <= 0 and live_shared_signal:
            post_final_shared_entry_count = 1

        return {
            "protected_hedge_visible_turn_count": int(visible_turn_count),
            "protected_hedge_recent_visible": bool(recent_hedge_visible),
            "protected_hedge_claim_state": str(claim_state or ""),
            "protected_hedge_claim_active": bool(claim_active),
            "post_reclaim_hold_count": int(post_reclaim_hold_count),
            "post_reclaim_hold_ready_source": str(
                getattr(runtime, "last_post_reclaim_hold_ready_source", "") or ""
            ),
            "post_final_shared_entry_count": int(post_final_shared_entry_count),
        }

    def _build_base_resolution(
        self,
        *,
        runtime: Any,
        evidence: ControlIREvidence | None,
        target_change_count: int,
        avoid_component_id: str = "",
    ) -> ResolutionResult | None:
        _ = evidence
        _ = target_change_count
        normalized_avoid_component_id = str(avoid_component_id or "").strip()

        from arcgym.eca.target_health import TargetHealth

        base_pool = getattr(runtime, "_last_base_pool", None)
        if self._post_stabilize_no_target_continuation_step_active:
            return None
        raw_candidates: list[StructuredCandidate] = list(
            getattr(base_pool, "candidates", None) or []
        )
        candidates = self._filter_suppressed_family_candidates(
            runtime=runtime,
            candidates=raw_candidates,
        )
        if self._rule_discovery_scoreless_cooldown_active(runtime):
            raw_candidates = self._filter_rule_discovery_cooldown_candidates(raw_candidates)
            candidates = self._filter_rule_discovery_cooldown_candidates(candidates)
        game_over_target = self._recent_game_over_target(runtime)
        if game_over_target:
            raw_candidates = self._filter_recent_game_over_candidates(
                raw_candidates,
                game_over_target=game_over_target,
            )
            candidates = self._filter_recent_game_over_candidates(
                candidates,
                game_over_target=game_over_target,
            )
        raw_candidates = self._filter_exhausted_local_effect_candidates(
            runtime=runtime,
            candidates=raw_candidates,
        )
        candidates = self._filter_exhausted_local_effect_candidates(
            runtime=runtime,
            candidates=candidates,
        )
        raw_candidates = self._filter_l1_attempt_failed_prefix_candidates(
            runtime=runtime,
            candidates=raw_candidates,
        )
        candidates = self._filter_l1_attempt_failed_prefix_candidates(
            runtime=runtime,
            candidates=candidates,
        )
        selected_summary = getattr(runtime, "last_selected_target_summary", None)
        selected_component_id = ""
        selected_family = ""
        if isinstance(selected_summary, dict):
            selected_component_id = str(
                selected_summary.get("component_id", "")
                or selected_summary.get("id", "")
                or ""
            ).strip()
            selected_family = str(
                selected_summary.get("action_family", "")
                or selected_summary.get("family", "")
                or selected_summary.get("transition_model_candidate_family", "")
                or ""
            ).strip()
        selected_is_auxiliary_marker = bool(
            str(selected_component_id or "").strip().lower().startswith("auxiliary::")
            or str((selected_summary or {}).get("source", "") if isinstance(selected_summary, dict) else "").strip()
            == "l1_action_family_quota"
        )

        health = getattr(getattr(runtime, "_last_target_health_result", None), "health", None)
        frontier_preempts_selected = False
        frontier_selected_consumed = False
        selected_effect: dict[str, Any] = {}
        if isinstance(selected_summary, dict) and selected_component_id:
            selected_anchor_key = str(selected_summary.get("anchor_key", "") or "").strip()
            selected_effect = self._runtime_local_effect_entry(
                runtime=runtime,
                component_id=selected_component_id,
                anchor_key=selected_anchor_key,
            )
            frontier_preempts_selected = bool(
                int(selected_effect.get("score_change_count", 0) or 0) <= 0
                and int(selected_effect.get("structural_delta_count", 0) or 0) > 0
                and self._has_transition_frontier_alternative(
                    runtime=runtime,
                    component_id=selected_component_id,
                    anchor_key=selected_anchor_key,
                )
            )
            frontier_selected_consumed = bool(
                str(selected_summary.get("source", "") or "").strip() == "transition_frontier"
                and int(selected_effect.get("score_change_count", 0) or 0) <= 0
                and (
                    int(selected_effect.get("weak_effect_count", 0) or 0) > 0
                    or int(selected_effect.get("no_change_count", 0) or 0) > 0
                    or int(selected_effect.get("probe_count", 0) or 0) >= 2
                )
            )
        selected_effect_saturated = self._local_effect_no_score_saturated(selected_effect)
        selected_click_no_change = bool(
            selected_family == "click"
            and int(selected_effect.get("score_change_count", 0) or 0) <= 0
            and int(selected_effect.get("board_change_no_score_count", 0) or 0) <= 0
            and int(selected_effect.get("no_change_count", 0) or 0) > 0
        )
        selected_no_score_exhausted = self._target_no_score_exhausted(
            runtime=runtime,
            component_id=selected_component_id,
            effect_entry=selected_effect,
        ) or bool(selected_is_auxiliary_marker) or bool(
            normalized_avoid_component_id
            and selected_component_id
            and normalized_avoid_component_id == selected_component_id
        )
        resolved: ScoredCandidate | None = None
        local_effect_retained = False
        stabilize_one_shot = self._pick_pending_stabilize_verify_candidate(
            runtime=runtime,
            candidates=raw_candidates,
        )
        starved_goal_probe = self._pick_starved_goal_probe_candidate(
            runtime=runtime,
            candidates=raw_candidates,
        )
        post_score_source_transfer = self._pick_post_score_source_transfer_candidate(
            runtime=runtime,
            candidates=candidates,
        )
        level_rule_grid_scan = self._pick_level_rule_grid_scan_candidate(
            runtime=runtime,
            candidates=candidates,
        )
        retained_candidate = (
            None
            if selected_no_score_exhausted
            or stabilize_one_shot is not None
            or starved_goal_probe is not None
            or post_score_source_transfer is not None
            or level_rule_grid_scan is not None
            else self._maybe_retain_local_effect_candidate(
                runtime=runtime,
                selected_summary=selected_summary,
                selected_component_id=selected_component_id,
                selected_family=selected_family,
            )
        )
        if stabilize_one_shot is not None:
            self._stabilize_verify_one_shot_selected_count += 1
            resolved = self._structured_candidate_to_scored_candidate(stabilize_one_shot)
        elif starved_goal_probe is not None:
            resolved = self._structured_candidate_to_scored_candidate(starved_goal_probe)
        elif post_score_source_transfer is not None:
            resolved = self._structured_candidate_to_scored_candidate(post_score_source_transfer)
        elif level_rule_grid_scan is not None:
            resolved = self._structured_candidate_to_scored_candidate(level_rule_grid_scan)
        elif retained_candidate is not None:
            resolved = retained_candidate
            local_effect_retained = True
        elif (
            health == TargetHealth.LIVE
            and not frontier_preempts_selected
            and not frontier_selected_consumed
            and not selected_effect_saturated
            and not selected_click_no_change
            and not selected_no_score_exhausted
            and selected_component_id
            and selected_family
        ):
            resolved = self._selected_summary_to_scored_candidate(selected_summary)
        else:
            preferred = None
            if (
                (
                    health in (TargetHealth.STALE, TargetHealth.SATURATED)
                    or selected_click_no_change
                )
                and selected_component_id
                and selected_family == "click"
                and not selected_no_score_exhausted
                and not self._local_effect_family_suppressed(runtime=runtime, family="move")
            ):
                preferred = self._pick_same_component_family_variant(
                    candidates=candidates,
                    component_id=selected_component_id,
                    current_family=selected_family,
                )
                if preferred is None:
                    preferred = self._pick_family_candidate(
                        candidates=candidates,
                        family="move",
                    )
            if preferred is None:
                preferred = self._pick_base_candidate(
                    candidates=candidates,
                    avoid_component_id=(
                        normalized_avoid_component_id
                        or (
                            selected_component_id
                            if health in (TargetHealth.STALE, TargetHealth.SATURATED)
                            or frontier_preempts_selected
                            or frontier_selected_consumed
                            or selected_effect_saturated
                            or selected_no_score_exhausted
                            else ""
                        )
                    ),
                    strict_avoid_component=bool(
                        normalized_avoid_component_id
                        or health in (TargetHealth.STALE, TargetHealth.SATURATED)
                        or frontier_preempts_selected
                        or frontier_selected_consumed
                        or selected_effect_saturated
                        or selected_no_score_exhausted
                    ),
                    prefer_family=(
                        self._preferred_local_effect_family(runtime=runtime)
                        if health in (TargetHealth.EMPTY, TargetHealth.STALE, TargetHealth.SATURATED)
                        or frontier_preempts_selected
                        or frontier_selected_consumed
                        or selected_effect_saturated
                        or selected_no_score_exhausted
                        else selected_family
                    ),
                )
            if preferred is not None:
                resolved = self._structured_candidate_to_scored_candidate(preferred)
            elif selected_component_id and selected_family and health not in (
                TargetHealth.STALE,
                TargetHealth.SATURATED,
            ) and not selected_no_score_exhausted:
                resolved = self._selected_summary_to_scored_candidate(selected_summary)

        if resolved is None:
            return None

        resolved.selected = True
        resolved.selection_reason = (
            "local_effect_retention"
            if local_effect_retained
            else "stabilize_verify_one_shot"
            if stabilize_one_shot is not None
            else "post_score_source_transfer"
            if post_score_source_transfer is not None
            else "level_rule_grid_scan"
            if level_rule_grid_scan is not None
            else "base_local_loop"
        )
        return ResolutionResult(
            selected=resolved,
            all_scored=[resolved],
            resolved=True,
            resolution_source=(
                "local_effect_retention"
                if local_effect_retained
                else "stabilize_verify_one_shot"
                if stabilize_one_shot is not None
                else "post_score_source_transfer"
                if post_score_source_transfer is not None
                else "level_rule_grid_scan"
                if level_rule_grid_scan is not None
                else "base_local_loop"
            ),
        )

    @staticmethod
    def _recent_game_over_target(runtime: Any) -> dict[str, str]:
        selected_summary = getattr(runtime, "last_selected_target_summary", None)
        if getattr(runtime, "last_arc_game_over", False) is True and isinstance(selected_summary, dict):
            return {
                "component_id": str(
                    selected_summary.get("component_id", "")
                    or selected_summary.get("id", "")
                    or selected_summary.get("candidate_id", "")
                    or ""
                ).strip(),
                "anchor_key": str(selected_summary.get("anchor_key", "") or "").strip(),
                "source": str(selected_summary.get("source", "") or "").strip(),
            }
        rows = [
            dict(row or {})
            for row in list(getattr(runtime, "step_metrics", None) or [])[-10:]
            if isinstance(row, dict)
        ]
        for row in reversed(rows):
            if not bool(row.get("arc_game_over", False)):
                continue
            return {
                "component_id": str(row.get("selected_target_id", "") or "").strip(),
                "anchor_key": str(row.get("selected_target_anchor", "") or "").strip(),
                "source": str(row.get("selected_target_source", "") or "").strip(),
            }
        return {}

    @staticmethod
    def _should_suppress_post_stabilize_stale_retarget(
        runtime: Any,
        compilation: Any,
        *,
        grace_steps: int = 48,
    ) -> bool:
        instructions = list(getattr(compilation, "instructions", None) or [])
        if not any(str(getattr(item, "action", "") or "") == "force_retarget" for item in instructions):
            return False
        return LocalController._post_stabilize_no_target_continuation_active(
            runtime,
            grace_steps=grace_steps,
        )

    @staticmethod
    def _post_stabilize_no_target_continuation_active(
        runtime: Any,
        *,
        grace_steps: int = 48,
    ) -> bool:
        rows = [
            dict(row or {})
            for row in list(getattr(runtime, "step_metrics", None) or [])[-max(1, int(grace_steps)) :]
            if isinstance(row, dict)
        ]
        if not rows:
            return False

        latest_score = None
        latest_level = None
        for row in reversed(rows):
            try:
                latest_score = float(row.get("score", 0.0) or 0.0)
                latest_level = int(row.get("level", row.get("level_number", 0)) or 0)
                break
            except (TypeError, ValueError):
                continue

        for row in reversed(rows):
            if str(row.get("selected_target_source", "") or "").strip() != "stabilize_verify":
                continue
            if (
                str(row.get("local_controller_resolution_source", "") or "").strip()
                != "stabilize_verify_one_shot"
            ):
                continue
            try:
                stabilize_score = float(row.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                stabilize_score = latest_score
            try:
                stabilize_level = int(row.get("level", row.get("level_number", 0)) or 0)
            except (TypeError, ValueError):
                stabilize_level = latest_level
            if latest_score is not None and stabilize_score is not None and latest_score > stabilize_score:
                return False
            if latest_level is not None and stabilize_level is not None and latest_level > stabilize_level:
                return False
            return True
        return False

    @staticmethod
    def _post_stabilize_metric_dominant_move(runtime: Any, *, recent_steps: int = 6) -> str:
        rows = [
            dict(row or {})
            for row in list(getattr(runtime, "step_metrics", None) or [])[-max(1, int(recent_steps)) :]
            if isinstance(row, dict)
        ]
        if not rows:
            return ""
        last = rows[-1]
        if str(last.get("selected_target_source", "") or "").strip() != "stabilize_verify":
            return ""
        if (
            str(last.get("local_controller_resolution_source", "") or "").strip()
            != "stabilize_verify_one_shot"
        ):
            return ""
        counts: dict[str, int] = {}
        order: list[str] = []
        for row in rows[:-1]:
            actions = row.get("local_controller_action_preview", []) or []
            if isinstance(actions, str):
                actions = [actions]
            for action_name in actions:
                normalized = str(action_name or "").strip().upper()
                if normalized not in _MOVE_ACTION_SET:
                    continue
                counts[normalized] = counts.get(normalized, 0) + 1
                order.append(normalized)
        if not counts:
            return ""
        return max(
            counts,
            key=lambda action_name: (
                int(counts.get(action_name, 0) or 0),
                max((idx for idx, value in enumerate(order) if value == action_name), default=-1),
            ),
        )

    @staticmethod
    def _rule_discovery_scoreless_cooldown_active(
        runtime: Any,
        *,
        recent_steps: int = 8,
        min_rule_steps: int = 6,
    ) -> bool:
        rows = [
            dict(row or {})
            for row in list(getattr(runtime, "step_metrics", None) or [])[-max(1, int(recent_steps)) :]
            if isinstance(row, dict)
        ]
        if len(rows) < min_rule_steps:
            return False
        scores: list[float] = []
        levels: list[int] = []
        for row in rows:
            try:
                scores.append(float(row.get("score", 0.0) or 0.0))
            except (TypeError, ValueError):
                scores.append(0.0)
            try:
                levels.append(int(row.get("level", row.get("level_number", 0)) or 0))
            except (TypeError, ValueError):
                levels.append(0)
        if scores and max(scores) > min(scores):
            return False
        if levels and max(levels) > min(levels):
            return False
        rule_steps = sum(
            1
            for row in rows
            if str(row.get("selected_target_source", "") or "").strip()
            in {"rule_hypothesis", "goal_probe", "stabilize_verify"}
        )
        if rule_steps < min_rule_steps:
            return False
        churn_max = 0
        for row in rows:
            try:
                churn_max = max(
                    churn_max,
                    int(row.get("selected_target_recent_churn_no_score_change_count", 0) or 0),
                )
            except (TypeError, ValueError):
                continue
        return churn_max >= 4

    @staticmethod
    def _filter_rule_discovery_cooldown_candidates(
        candidates: list[StructuredCandidate],
    ) -> list[StructuredCandidate]:
        filtered: list[StructuredCandidate] = []
        for candidate in candidates:
            source = str(candidate.source or "").strip()
            stage = str(getattr(candidate, "stage", "") or "").strip()
            if source == "stabilize_verify" and stage == "lease_followup":
                filtered.append(candidate)
                continue
            if source in {"rule_hypothesis", "goal_probe", "stabilize_verify"}:
                continue
            filtered.append(candidate)
        return filtered

    @staticmethod
    def _filter_recent_game_over_candidates(
        candidates: list[StructuredCandidate],
        *,
        game_over_target: dict[str, str],
    ) -> list[StructuredCandidate]:
        component_id = str(game_over_target.get("component_id", "") or "").strip()
        anchor_key = str(game_over_target.get("anchor_key", "") or "").strip()
        if not component_id and not anchor_key:
            return candidates
        filtered = [
            candidate
            for candidate in candidates
            if (
                not component_id
                or str(candidate.component_id or "").strip() != component_id
            )
            and (
                not anchor_key
                or str(candidate.anchor_key or "").strip() != anchor_key
            )
        ]
        return filtered or candidates

    @staticmethod
    def _l1_attempt_target_key(component_id: str, anchor_key: str) -> str:
        component = str(component_id or "").strip()
        anchor = str(anchor_key or "").strip()
        if component or anchor:
            return f"{component}|{anchor}"
        return ""

    @staticmethod
    def _anchor_key_position(anchor_key: str) -> tuple[int, int] | None:
        parts = [part.strip() for part in str(anchor_key or "").split("||") if part.strip()]
        if len(parts) < 2:
            return None
        try:
            return int(parts[-2]), int(parts[-1])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _step_row_index(row: dict[str, Any], fallback: int) -> int:
        for key in ("action_idx", "step", "step_index"):
            try:
                value = row.get(key)
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                continue
        return int(fallback)

    @staticmethod
    def _step_row_board_changed(row: dict[str, Any]) -> bool:
        if "board_changed" in row:
            return bool(row.get("board_changed", False))
        if "no_change" in row:
            return not bool(row.get("no_change", False))
        return False

    def _update_l1_attempt_outcome_ledger(self, *, runtime: Any, step_index: int) -> None:
        rows = [
            dict(row or {})
            for row in list(getattr(runtime, "step_metrics", None) or [])
            if isinstance(row, dict)
        ]
        if not rows:
            return

        last_row = rows[-1]
        last_step = self._step_row_index(last_row, int(step_index) - 1)
        if (
            str(last_row.get("selected_target_source", "") or "").strip()
            == "l1_attempt_relay_followup"
            and last_step not in self._l1_attempt_relay_followup_outcome_steps
        ):
            self._l1_attempt_relay_followup_outcome_steps.add(last_step)
            try:
                followup_score_delta = float(last_row.get("score_delta", 0.0) or 0.0)
            except (TypeError, ValueError):
                followup_score_delta = 0.0
            if followup_score_delta > 0.0:
                self._l1_attempt_relay_followup_score_count += 1
            if bool(last_row.get("arc_game_over", False)):
                self._l1_attempt_relay_followup_gameover_count += 1

        if self._move_action_available:
            return
        if self._runtime_score(runtime) > 0 or self._runtime_level_number(runtime) > 1:
            return
        if not (
            bool(getattr(runtime, "last_arc_game_over", False))
            or bool(last_row.get("arc_game_over", False))
        ):
            return

        attempt = 0
        for raw_attempt in (
            getattr(runtime, "last_arc_attempt_number", None),
            last_row.get("arc_attempt_number"),
            last_row.get("arc_attempt"),
        ):
            try:
                if raw_attempt is not None:
                    attempt = int(raw_attempt)
                    break
            except (TypeError, ValueError):
                continue
        if attempt <= 0:
            attempt = last_step
        if attempt <= self._l1_attempt_last_recorded_attempt:
            return

        attempt_rows = []
        for row in rows[-32:]:
            try:
                row_attempt = int(row.get("arc_attempt_number", row.get("arc_attempt", -1)) or -1)
            except (TypeError, ValueError):
                row_attempt = -1
            if row_attempt == attempt:
                attempt_rows.append(row)
        if not attempt_rows:
            attempt_rows = rows[-16:]
        if any(float(row.get("score", 0.0) or 0.0) > 0.0 for row in attempt_rows):
            self._l1_attempt_last_recorded_attempt = attempt
            return

        self._l1_attempt_last_recorded_attempt = attempt
        self._l1_attempt_outcome_record_count += 1

        prefix_added = 0
        relay_added = 0
        seen_relay_keys = {
            self._l1_attempt_target_key(
                str(item.get("component_id", "") or ""),
                str(item.get("anchor_key", "") or ""),
            )
            for item in self._l1_attempt_relay_anchor_queue
        }
        for row in attempt_rows:
            component_id = str(row.get("selected_target_id", "") or "").strip()
            anchor_key = str(row.get("selected_target_anchor", "") or "").strip()
            source = str(row.get("selected_target_source", "") or "").strip()
            key = self._l1_attempt_target_key(component_id, anchor_key)
            if key and key not in self._l1_attempt_failed_prefix_keys:
                self._l1_attempt_failed_prefix_keys.add(key)
                prefix_added += 1
            try:
                score_delta = float(row.get("score_delta", 0.0) or 0.0)
            except (TypeError, ValueError):
                score_delta = 0.0
            if bool(row.get("arc_game_over", False)):
                continue
            if not self._step_row_board_changed(row) or abs(score_delta) > 1e-9:
                continue
            if source not in {
                "grid_scan",
                "level_rule_symbol_scout",
                "object_escape",
                "local_click_fanout",
                "local_reversible_click",
                "aligned_object_sweep",
                "candidate_pool",
                "object_table",
                "successor_slate",
            }:
                continue
            position = self._anchor_key_position(anchor_key)
            if position is None:
                continue
            relay_key = key or f"{source}|{anchor_key}"
            if relay_key in seen_relay_keys:
                continue
            row_idx, col_idx = position
            self._l1_attempt_relay_anchor_queue.append(
                {
                    "component_id": component_id,
                    "anchor_key": anchor_key,
                    "source": source,
                    "row": int(row_idx),
                    "col": int(col_idx),
                    "attempt": int(attempt),
                    "step": self._step_row_index(row, last_step),
                }
            )
            seen_relay_keys.add(relay_key)
            relay_added += 1

        self._l1_attempt_relay_anchor_queue = self._l1_attempt_relay_anchor_queue[-12:]
        self._l1_attempt_failed_prefix_count += prefix_added
        self._l1_attempt_relay_anchor_count += relay_added

    def _build_l1_attempt_relay_followup_plan(
        self,
        *,
        runtime: Any,
        step_index: int,
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
        semantic_action_lease: dict[str, Any],
        hidden_state_lease: HiddenStateExecutorLease,
    ) -> tuple[list[PlannedAction], ScoredCandidate] | None:
        if self._move_action_available:
            return None
        if bool(semantic_action_lease.get("active", False)) or bool(hidden_state_lease.active):
            return None
        if self._runtime_score(runtime) > 0 or self._runtime_level_number(runtime) > 1:
            return None
        if self._l1_attempt_outcome_record_count <= 0 or not self._l1_attempt_relay_anchor_queue:
            return None
        relay_budget = max(2, int(self._l1_attempt_outcome_record_count) * 2)
        if self._l1_attempt_relay_followup_issued_count >= relay_budget:
            return None
        if int(step_index) - int(self._l1_attempt_relay_followup_last_step) < 4:
            return None
        available = self._normalized_available_actions(allowed_actions, blocked_actions)
        if _CLICK_ACTION not in available:
            return None

        base_pool = getattr(runtime, "_last_base_pool", None)
        candidates = [
            candidate
            for candidate in list(getattr(base_pool, "candidates", None) or [])
            if not getattr(candidate, "reject_reason", "")
            and str(getattr(candidate, "family", "") or "").strip().lower()
            in {"", "click", "select"}
            and str(getattr(candidate, "source", "") or "").strip()
            in {
                "object_escape",
                "aligned_object_sweep",
                "local_click_fanout",
                "local_reversible_click",
                "object_table",
                "successor_slate",
                "candidate_pool",
            }
        ]
        if not candidates:
            return None

        anchors = list(reversed(self._l1_attempt_relay_anchor_queue[-8:]))

        def candidate_key(candidate: StructuredCandidate) -> str:
            return self._l1_attempt_target_key(
                str(candidate.component_id or ""),
                str(candidate.anchor_key or ""),
            )

        scored: list[tuple[float, int, int, StructuredCandidate, dict[str, Any]]] = []
        for candidate in candidates:
            key = candidate_key(candidate)
            if key and key in self._l1_attempt_failed_prefix_keys:
                self._l1_attempt_prefix_suppressed_count += 1
                continue
            if key and key in self._l1_attempt_relay_followup_keys:
                continue
            source = str(candidate.source or "").strip()
            source_bonus = {
                "object_escape": 2.0,
                "aligned_object_sweep": 1.55,
                "local_reversible_click": 1.20,
                "local_click_fanout": 1.05,
                "successor_slate": 0.85,
                "object_table": 0.70,
                "candidate_pool": 0.45,
            }.get(source, 0.0)
            best_anchor_score = -1000.0
            best_anchor: dict[str, Any] = {}
            for anchor in anchors:
                row_delta = abs(int(candidate.center_row or 0) - int(anchor.get("row", 0) or 0))
                col_delta = abs(int(candidate.center_col or 0) - int(anchor.get("col", 0) or 0))
                same_line_bonus = 1.10 if row_delta == 0 or col_delta == 0 else 0.0
                near_bonus = max(0.0, 0.8 - 0.08 * float(row_delta + col_delta))
                anchor_score = same_line_bonus + near_bonus
                if anchor_score > best_anchor_score:
                    best_anchor_score = anchor_score
                    best_anchor = anchor
            if best_anchor_score < 0.15:
                continue
            scored.append(
                (
                    float(candidate.base_score or 0.0) + float(candidate.effect_bonus or 0.0)
                    + source_bonus
                    + best_anchor_score,
                    int(candidate.center_row or 0),
                    int(candidate.center_col or 0),
                    candidate,
                    best_anchor,
                )
            )
        if not scored:
            return None
        _, _, _, selected, anchor = max(scored, key=lambda item: item[:3])
        key = candidate_key(selected)
        self._l1_attempt_relay_followup_keys.add(key)
        self._l1_attempt_relay_followup_last_step = int(step_index)
        self._l1_attempt_relay_followup_issued_count += 1
        scored_candidate = ScoredCandidate(
            component_id=str(selected.component_id or ""),
            action_family="click",
            source="l1_attempt_relay_followup",
            action_name=_CLICK_ACTION,
            anchor_key=str(selected.anchor_key or ""),
            center_row=int(selected.center_row or 0),
            center_col=int(selected.center_col or 0),
            base_admission_score=float(selected.base_score or 0.0),
            resolver_score=max(0.01, float(selected.base_score or 0.0) + float(selected.effect_bonus or 0.0)),
            selected=True,
            selection_reason="l1_attempt_relay_followup",
            stage="attempt_relay",
            linked_effect_id=str(anchor.get("anchor_key", "") or ""),
        )
        actions = [
            PlannedAction(
                action_name=_CLICK_ACTION,
                target_component_id=str(selected.component_id or ""),
                parameters={
                    "x": int(selected.center_row or 0),
                    "y": int(selected.center_col or 0),
                },
                reason="l1_attempt_relay_followup",
            )
        ]
        return actions, scored_candidate

    def _filter_l1_attempt_failed_prefix_candidates(
        self,
        *,
        runtime: Any,
        candidates: list[StructuredCandidate],
    ) -> list[StructuredCandidate]:
        if not candidates or not self._l1_attempt_failed_prefix_keys:
            return candidates
        if self._move_action_available:
            return candidates
        if self._runtime_score(runtime) > 0 or self._runtime_level_number(runtime) > 1:
            return candidates
        filtered: list[StructuredCandidate] = []
        suppressed = 0
        for candidate in candidates:
            key = self._l1_attempt_target_key(
                str(candidate.component_id or ""),
                str(candidate.anchor_key or ""),
            )
            if key and key in self._l1_attempt_failed_prefix_keys:
                suppressed += 1
                continue
            filtered.append(candidate)
        if not filtered:
            return candidates
        self._l1_attempt_prefix_suppressed_count += suppressed
        return filtered

    def _filter_exhausted_local_effect_candidates(
        self,
        *,
        runtime: Any,
        candidates: list[StructuredCandidate],
    ) -> list[StructuredCandidate]:
        if not candidates:
            return []
        filtered: list[StructuredCandidate] = []
        for candidate in candidates:
            source = str(candidate.source or "").strip()
            if source == "stabilize_verify":
                stage = str(getattr(candidate, "stage", "") or "").strip()
                if stage == "lease_followup":
                    filtered.append(candidate)
                    continue
                entry = self._runtime_local_effect_entry(
                    runtime=runtime,
                    component_id=str(candidate.component_id or ""),
                    anchor_key=str(candidate.anchor_key or ""),
                )
                entry_component = str(entry.get("component_id", "") or "").strip()
                if entry_component and entry_component != str(candidate.component_id or "").strip():
                    entry = {}
                if self._local_effect_no_score_saturated(entry):
                    continue
                filtered.append(candidate)
                continue
            if source in {
                "grid_scan",
                "level_rule_symbol_scout",
                "local_click_fanout",
                "local_reversible_click",
                "object_escape",
                "goal_probe",
                "rule_hypothesis",
            }:
                filtered.append(candidate)
                continue
            entry = self._runtime_local_effect_entry(
                runtime=runtime,
                component_id=str(candidate.component_id or ""),
                anchor_key=str(candidate.anchor_key or ""),
            )
            entry_component = str(entry.get("component_id", "") or "").strip()
            if entry_component and entry_component != str(candidate.component_id or "").strip():
                entry = {}
            if self._local_effect_no_score_saturated(entry):
                continue
            filtered.append(candidate)
        return filtered

    def _runtime_local_effect_entry(
        self,
        *,
        runtime: Any,
        component_id: str,
        anchor_key: str = "",
    ) -> dict[str, Any]:
        resolver = getattr(runtime, "_local_effect_entry_for", None)
        if callable(resolver):
            try:
                return dict(resolver(component_id=component_id, anchor_key=anchor_key) or {})
            except Exception:
                return {}
        ledger = getattr(runtime, "local_effect_ledger", None)
        if not isinstance(ledger, dict):
            return {}
        key = str(component_id or "").strip()
        if key and key in ledger:
            return dict(ledger.get(key) or {})
        anchor = str(anchor_key or "").strip()
        if anchor:
            for entry in ledger.values():
                item = dict(entry or {})
                if str(item.get("anchor_key", "") or "") == anchor:
                    return item
        return {}

    def _local_effect_retention_family(
        self,
        *,
        selected_family: str,
        effect_entry: dict[str, Any],
    ) -> str:
        normalized_selected = str(selected_family or "").strip().lower()
        last_family = str(effect_entry.get("last_family", "") or "").strip().lower()
        if last_family in {"click", "move"}:
            return last_family
        if normalized_selected in {"click", "move"}:
            return normalized_selected
        family_counts = dict(effect_entry.get("family_counts") or {})
        ranked = sorted(
            (
                (str(family or "").strip().lower(), int(count or 0))
                for family, count in family_counts.items()
                if str(family or "").strip().lower() in {"click", "move"}
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[0][0] if ranked else ""

    @staticmethod
    def _local_effect_family_entry(runtime: Any, *, family: str) -> dict[str, Any]:
        ledger = getattr(runtime, "local_effect_family_ledger", None)
        if not isinstance(ledger, dict):
            return {}
        return dict(ledger.get(str(family or "").strip().lower()) or {})

    @staticmethod
    def _local_effect_no_score_saturated(effect_entry: dict[str, Any]) -> bool:
        if not effect_entry:
            return False
        score_change_count = int(effect_entry.get("score_change_count", 0) or 0)
        if score_change_count > 0:
            return False
        probe_count = int(effect_entry.get("probe_count", 0) or 0)
        board_change_count = int(effect_entry.get("board_change_no_score_count", 0) or 0)
        no_change_count = int(effect_entry.get("no_change_count", 0) or 0)
        progress_effect_count = int(effect_entry.get("progress_effect_count", 0) or 0)
        weak_effect_count = int(effect_entry.get("weak_effect_count", 0) or 0)
        structural_delta_count = int(effect_entry.get("structural_delta_count", 0) or 0)
        if no_change_count >= 2:
            return True
        if probe_count >= 5 and board_change_count >= 4:
            return True
        if probe_count >= 5 and progress_effect_count >= 2:
            return True
        if weak_effect_count >= 2 and structural_delta_count <= weak_effect_count:
            return True
        return False

    def _target_no_score_exhausted(
        self,
        *,
        runtime: Any,
        component_id: str,
        effect_entry: dict[str, Any] | None = None,
    ) -> bool:
        normalized_component = str(component_id or "").strip()
        if not normalized_component:
            return False

        effect = dict(effect_entry or {})
        if effect:
            effect_component = str(effect.get("component_id", "") or "").strip()
            if effect_component and effect_component != normalized_component:
                effect = {}
        if int(effect.get("score_change_count", 0) or 0) > 0:
            return False

        drop_reason = str(
            getattr(runtime, "last_current_target_viability_drop_reason", "") or ""
        ).strip()
        if drop_reason in _NO_SCORE_EXHAUSTED_TARGET_DROP_REASONS:
            return True

        if self._local_effect_no_score_saturated(effect):
            return True

        return False

    def _maybe_replace_exhausted_resolution(
        self,
        *,
        runtime: Any,
        resolution: ResolutionResult,
        evidence: ControlIREvidence | None,
        target_change_count: int,
    ) -> ResolutionResult:
        selected = resolution.selected if resolution.resolved else None
        if selected is None:
            return resolution
        selected_component_id = str(selected.component_id or "").strip()
        selected_anchor_key = str(selected.anchor_key or "").strip()
        effect_entry = self._runtime_local_effect_entry(
            runtime=runtime,
            component_id=selected_component_id,
            anchor_key=selected_anchor_key,
        )
        if not self._target_no_score_exhausted(
            runtime=runtime,
            component_id=selected_component_id,
            effect_entry=effect_entry,
        ):
            return resolution

        alternative = self._build_base_resolution(
            runtime=runtime,
            evidence=evidence,
            target_change_count=target_change_count,
            avoid_component_id=selected_component_id,
        )
        if (
            alternative is None
            or not alternative.resolved
            or alternative.selected is None
            or str(alternative.selected.component_id or "").strip() == selected_component_id
        ):
            return resolution
        alternative.selected.selection_reason = "exhausted_target_alternative"
        alternative.resolution_source = "exhausted_target_alternative"
        return alternative

    def _local_effect_family_suppressed(self, *, runtime: Any, family: str) -> bool:
        entry = self._local_effect_family_entry(runtime, family=family)
        if not entry:
            return False
        progress_count = int(entry.get("progress_effect_count", 0) or 0)
        score_change_count = int(entry.get("score_change_count", 0) or 0)
        weak_count = int(entry.get("weak_effect_count", 0) or 0)
        no_change_count = int(entry.get("no_change_count", 0) or 0)
        score = float(entry.get("score", 0.0) or 0.0)
        if score_change_count > 0:
            return False
        if progress_count <= 0 and no_change_count >= 2:
            return True
        if progress_count <= 0 and score <= -1.0:
            return True
        if score_change_count <= 0 and progress_count <= 3 and weak_count >= progress_count + 6:
            return True
        return bool(weak_count >= max(3, progress_count + 3) and score <= 0.0)

    def _filter_suppressed_family_candidates(
        self,
        *,
        runtime: Any,
        candidates: list[StructuredCandidate],
    ) -> list[StructuredCandidate]:
        if not candidates:
            return []
        filtered = [
            candidate
            for candidate in candidates
            if not self._local_effect_family_suppressed(
                runtime=runtime,
                family=str(candidate.family or "").strip(),
            )
        ]
        return filtered if filtered else list(candidates)

    def _preferred_local_effect_family(self, *, runtime: Any) -> str:
        ledger = getattr(runtime, "local_effect_family_ledger", None)
        if not isinstance(ledger, dict):
            return ""
        candidates: list[tuple[float, str]] = []
        for family in ("click", "move"):
            if self._local_effect_family_suppressed(runtime=runtime, family=family):
                continue
            entry = dict(ledger.get(family) or {})
            if not entry:
                continue
            score = float(entry.get("score", 0.0) or 0.0)
            progress_count = int(entry.get("progress_effect_count", 0) or 0)
            structural_count = int(entry.get("structural_delta_count", 0) or 0)
            score_change_count = int(entry.get("score_change_count", 0) or 0)
            candidates.append(
                (
                    score
                    + 0.35 * float(score_change_count)
                    + 0.20 * float(progress_count)
                    + 0.10 * float(structural_count),
                    family,
                )
            )
        if not candidates:
            return ""
        candidates.sort(reverse=True)
        return candidates[0][1]

    @staticmethod
    def _has_transition_frontier_alternative(
        *,
        runtime: Any,
        component_id: str,
        anchor_key: str = "",
    ) -> bool:
        raw = getattr(runtime, "local_transition_frontier", None)
        if not isinstance(raw, list):
            return False
        state = getattr(runtime, "last_state", None)
        current_step = int(getattr(state, "step_index", -10_000) or -10_000)
        selected_component = str(component_id or "").strip()
        selected_anchor = str(anchor_key or "").strip()
        for item in raw:
            if not isinstance(item, dict):
                continue
            expires_at = int(item.get("expires_at", -10_000) or -10_000)
            if expires_at < current_step:
                continue
            frontier_component = str(item.get("component_id", "") or "").strip()
            frontier_anchor = str(item.get("anchor_key", "") or "").strip()
            if frontier_component and frontier_component != selected_component:
                return True
            if frontier_anchor and frontier_anchor != selected_anchor:
                return True
        return False

    def _maybe_retain_local_effect_candidate(
        self,
        *,
        runtime: Any,
        selected_summary: Any,
        selected_component_id: str,
        selected_family: str,
    ) -> ScoredCandidate | None:
        if not isinstance(selected_summary, dict):
            return None
        component_id = str(selected_component_id or "").strip()
        if not component_id:
            return None
        anchor_key = str(selected_summary.get("anchor_key", "") or "").strip()
        effect_entry = self._runtime_local_effect_entry(
            runtime=runtime,
            component_id=component_id,
            anchor_key=anchor_key,
        )
        if not effect_entry:
            return None
        state = getattr(runtime, "last_state", None)
        current_step = int(getattr(state, "step_index", -10_000) or -10_000)
        last_effect_step = int(effect_entry.get("last_step_idx", -10_000) or -10_000)
        if current_step - last_effect_step > 3:
            return None
        effect_score = float(effect_entry.get("score", 0.0) or 0.0)
        probe_count = int(effect_entry.get("probe_count", 0) or 0)
        board_change_count = int(effect_entry.get("board_change_no_score_count", 0) or 0)
        score_change_count = int(effect_entry.get("score_change_count", 0) or 0)
        no_change_count = int(effect_entry.get("no_change_count", 0) or 0)
        progress_effect_count = int(effect_entry.get("progress_effect_count", 0) or 0)
        weak_effect_count = int(effect_entry.get("weak_effect_count", 0) or 0)
        structural_delta_count = int(effect_entry.get("structural_delta_count", 0) or 0)
        if score_change_count <= 0 and board_change_count <= 0:
            return None
        if effect_score < 0.75:
            return None
        max_no_score_effect_probes = 4
        max_no_score_board_changes = 3
        if (
            score_change_count <= 0
            and probe_count >= max_no_score_effect_probes
            and board_change_count >= max_no_score_board_changes
            and structural_delta_count <= 0
        ):
            return None
        if (
            score_change_count <= 0
            and probe_count >= 5
            and progress_effect_count >= 2
            and board_change_count >= 4
        ):
            return None
        if self._local_effect_no_score_saturated(effect_entry):
            return None
        if score_change_count <= 0 and weak_effect_count >= 2 and structural_delta_count <= weak_effect_count:
            return None
        if score_change_count <= 0 and no_change_count >= board_change_count + 4:
            return None
        if (
            score_change_count <= 0
            and structural_delta_count > 0
            and self._has_transition_frontier_alternative(
                runtime=runtime,
                component_id=component_id,
                anchor_key=anchor_key,
            )
        ):
            return None
        retained_family = self._local_effect_retention_family(
            selected_family=selected_family,
            effect_entry=effect_entry,
        )
        if not retained_family:
            return None
        retained_summary = dict(selected_summary)
        retained_summary["action_family"] = retained_family
        retained_summary["family"] = retained_family
        retained_summary["transition_model_candidate_family"] = retained_family
        retained_summary["admission_score"] = max(
            self._coerce_summary_score(retained_summary.get("admission_score", 0.0)),
            min(0.95, 0.45 + 0.08 * effect_score),
        )
        retained = self._selected_summary_to_scored_candidate(
            retained_summary,
            action_family_override=retained_family,
            resolver_score_bonus=min(0.35, 0.06 * effect_score),
        )
        if retained is not None:
            retained.selection_reason = "local_effect_retention"
        return retained

    def _build_hidden_state_execution_ticket(
        self,
        runtime: Any,
    ) -> AdapterExecutionTicket | None:
        raw_ticket = getattr(runtime, "last_semantic_execution_ticket", None)
        if isinstance(raw_ticket, AdapterExecutionTicket):
            return raw_ticket
        if isinstance(raw_ticket, dict):
            return AdapterExecutionTicket.from_mapping(raw_ticket)
        return None

    def _runtime_board_rows(self, runtime: Any) -> list[str]:
        board_rows = getattr(runtime, "last_board_rows", None)
        if not isinstance(board_rows, list):
            return []
        return [str(row or "") for row in board_rows]

    def _runtime_level_number(self, runtime: Any) -> int:
        level_number = getattr(runtime, "last_level_number", None)
        try:
            return int(level_number if level_number is not None else 0)
        except (TypeError, ValueError):
            return 0

    def _runtime_score(self, runtime: Any) -> int:
        score = getattr(runtime, "last_arc_score", None)
        if score is None:
            score = getattr(getattr(runtime, "last_state", None), "score", None)
        try:
            return int(score if score is not None else 0)
        except (TypeError, ValueError):
            return 0

    def _runtime_semantic_player_bbox(
        self,
        runtime: Any,
    ) -> list[int]:
        raw_observation = getattr(runtime, "last_semantic_observation", None)
        if isinstance(raw_observation, dict):
            control_state = raw_observation.get("control_state")
            if isinstance(control_state, dict) and isinstance(control_state.get("player_bbox"), list):
                return [
                    int(value)
                    for value in list(control_state.get("player_bbox", []) or [])
                    if str(value).strip("-").isdigit()
                ]
        else:
            control_state = getattr(raw_observation, "control_state", None)
            raw_control_bbox = getattr(control_state, "player_bbox", None)
            if isinstance(raw_control_bbox, (list, tuple)):
                return [
                    int(value)
                    for value in list(raw_control_bbox)
                    if str(value).strip("-").isdigit()
                ]
        raw_bbox = getattr(runtime, "last_semantic_player_bbox", None)
        if isinstance(raw_bbox, (list, tuple)):
            return [
                int(value)
                for value in list(raw_bbox)
                if str(value).strip("-").isdigit()
            ]
        return []

    def _update_hidden_state_executor(
        self,
        *,
        runtime: Any,
        execution_ticket: AdapterExecutionTicket | None,
        step_index: int,
    ) -> HiddenStateExecutorLease:
        return self._hidden_state_executor.step(
            ticket=execution_ticket,
            player_bbox=self._runtime_semantic_player_bbox(runtime),
            board_rows=self._runtime_board_rows(runtime),
            step_index=step_index,
            score=self._runtime_score(runtime),
            level_number=self._runtime_level_number(runtime),
        )

    def _merge_hidden_state_executor_lease(
        self,
        *,
        semantic_action_lease: dict[str, Any] | None,
        hidden_state_lease: HiddenStateExecutorLease,
    ) -> dict[str, Any]:
        lease = dict(semantic_action_lease or {})
        progress_goal_route_sync = bool(lease.get("semantic_progress_goal_route_sync", False))
        progress_goal_route = [
            str(action or "").strip().upper()
            for action in list(
                lease.get("semantic_progress_goal_route_preview")
                or lease.get("route_preview", [])
                or []
            )
            if str(action or "").strip().upper() in _MOVE_ACTION_SET
        ]
        progress_goal_target = [
            int(value)
            for value in list(
                lease.get("semantic_progress_goal_route_target_anchor")
                or lease.get("route_target_anchor", [])
                or []
            )[:2]
            if str(value).strip("-").isdigit()
        ]
        progress_goal_source = str(
            lease.get("semantic_progress_goal_route_source")
            or lease.get("route_source", "")
            or ""
        ).strip()
        metadata = dict(hidden_state_lease.metadata or {})
        if not hidden_state_lease.active:
            lease["hidden_state_executor"] = hidden_state_lease.to_dict()
            return lease
        try:
            level_number = int(lease.get("level_number", 0) or 0)
        except (TypeError, ValueError):
            level_number = 0
        resource_cycle_verify_active = bool(
            metadata.get("resource_cycle_verify_active", False)
            or lease.get("resource_cycle_verify_active", False)
        )
        alternate_site_route_active = bool(
            metadata.get("alternate_site_route_active", False)
            or lease.get("alternate_site_route_active", False)
        )

        route_preview = [
            str(action or "").strip().upper()
            for action in list(hidden_state_lease.route_preview or [])
            if str(action or "").strip().upper() in _MOVE_ACTION_SET
        ]
        adapter_sequence_remaining = [
            str(action or "").strip().upper()
            for action in list(lease.get("site_actuation_sequence_remaining", []) or [])
            if str(action or "").strip().upper() in _MOVE_ACTION_SET
        ]
        adapter_sequence_name = str(lease.get("site_actuation_sequence_name", "") or "").strip().lower()
        adapter_site_sequence_active = bool(
            adapter_sequence_remaining
            and (
                str(lease.get("semantic_phase", "") or "").strip().lower().startswith("site_actuation")
                or bool(lease.get("site_actuation_active", False))
                or bool(adapter_sequence_name)
            )
        )
        if route_preview:
            lease["suggested_actions"] = list(route_preview)
            lease["route_preview"] = list(route_preview)
        lease["active"] = True
        lease["mechanics_mode"] = lease.get("mechanics_mode") or "semantic"
        lease["semantic_phase"] = str(hidden_state_lease.phase or lease.get("semantic_phase", "") or "").strip().lower()
        lease["priority"] = max(float(lease.get("priority", 0.0) or 0.0), 1.25)
        # Hidden-state execution must resynchronize with a fresh observation every
        # turn. Multi-action queues can consume stale route segments after the
        # board/player state has already changed.
        lease["lease_horizon"] = 1
        lease["route_source"] = str(hidden_state_lease.route_source or lease.get("route_source", "") or "").strip()
        lease["route_target_anchor"] = list(
            hidden_state_lease.target_anchor
            or lease.get("route_target_anchor", [])
            or []
        )
        hidden_target = [
            int(value)
            for value in list(hidden_state_lease.target_anchor or [])[:2]
            if str(value).strip("-").isdigit()
        ]
        hidden_phase = str(hidden_state_lease.phase or lease.get("semantic_phase", "") or "").strip().lower()
        if (
            progress_goal_route_sync
            and not resource_cycle_verify_active
            and hidden_phase in {"state_change_verification", "goal_route_execution"}
            and progress_goal_target
            and hidden_target
            and progress_goal_target != hidden_target
        ):
            lease["route_target_anchor"] = list(progress_goal_target)
            if progress_goal_route:
                lease["suggested_actions"] = list(progress_goal_route)
                lease["route_preview"] = list(progress_goal_route)
            if progress_goal_source:
                lease["route_source"] = progress_goal_source
            lease["hidden_state_executor_route_resync_from_progress"] = True
        lease["hidden_state_executor_active"] = True
        lease["hidden_state_executor_phase"] = hidden_state_lease.phase
        lease["hidden_state_executor_activation_id"] = int(hidden_state_lease.activation_id)
        lease["hidden_state_executor_ttl_remaining"] = int(hidden_state_lease.ttl_remaining)
        lease["hidden_state_executor_stall_remaining"] = int(hidden_state_lease.stall_remaining)
        lease["hidden_state_executor_missing_compatible_ticket_count"] = int(
            hidden_state_lease.missing_compatible_ticket_count
        )
        lease["hidden_state_executor_ownership"] = str(hidden_state_lease.ownership or "")
        lease["hidden_state_executor_kind"] = str(hidden_state_lease.kind or "")
        lease["hidden_state_executor_target_anchor"] = list(hidden_state_lease.target_anchor)
        lease["hidden_state_executor_site_anchor"] = list(hidden_state_lease.site_anchor)
        lease["hidden_state_executor_goal_anchor"] = list(hidden_state_lease.goal_anchor)
        lease["hidden_state_executor_provenance"] = str(hidden_state_lease.provenance or "")
        lease["hidden_state_executor_route_source"] = str(hidden_state_lease.route_source or "")
        lease["hidden_state_executor_metadata"] = metadata
        event_kind = str(lease.get("event_kind", "") or "").strip()
        if not event_kind:
            lease["event_kind"] = str(metadata.get("event_kind", "") or "hidden_state_executor").strip()
        if alternate_site_route_active:
            if level_number >= 5 and self._apply_site_actuation_contract_lease(
                lease,
                event_kind="site_actuation_contract_alternate_route_deferred",
                priority=1.45,
            ):
                lease["hidden_state_executor_alternate_route_deferred"] = True
            elif level_number >= 5 and adapter_site_sequence_active:
                lease["hidden_state_executor_alternate_route_deferred"] = True
                lease["semantic_phase"] = "site_actuation"
                lease["site_actuation_active"] = True
                lease["site_actuation_sequence_name"] = adapter_sequence_name
                lease["site_actuation_sequence_index"] = int(lease.get("site_actuation_sequence_index", 0) or 0)
                lease["site_actuation_sequence_remaining"] = list(adapter_sequence_remaining)
                lease["suggested_actions"] = list(adapter_sequence_remaining)
                lease["route_preview"] = list(adapter_sequence_remaining)
                lease["lease_horizon"] = max(1, min(3, len(adapter_sequence_remaining)))
                lease["priority"] = max(float(lease.get("priority", 0.0) or 0.0), 1.45)
                lease["event_kind"] = "site_actuation_adapter_sequence_alternate_route_deferred"
            else:
                self._abort_site_actuation_contract("alternate_route_handoff")
                lease["semantic_phase"] = "site_reaching"
                lease["site_actuation_active"] = False
                lease["site_actuation_sequence_name"] = ""
                lease["site_actuation_sequence_index"] = 0
                lease["site_actuation_sequence_remaining"] = []
                lease["event_kind"] = "alternate_site_route_handoff"
        if route_preview and not lease.get("site_actuation_sequence_remaining"):
            if lease["semantic_phase"].startswith("site_actuation"):
                lease["site_actuation_sequence_remaining"] = list(route_preview)
        lease["hidden_state_executor"] = hidden_state_lease.to_dict()
        return lease

    def _notify_adapter_site_reacquire(
        self,
        *,
        sequence_name: str,
        reason: str,
        block_sequence: bool = False,
        site_anchor: list[int] | None = None,
        step_index: int | None = None,
    ) -> bool:
        self._semantic_site_actuation_controller_adapter_notify_count += 1
        if block_sequence:
            self._semantic_site_actuation_controller_adapter_notify_block_count += 1
            self._note_site_closer_outcome(
                sequence_name=sequence_name,
                outcome=(
                    "no_distance_gain"
                    if "distance" in str(reason or "").strip().lower()
                    else "far_tail_failed"
                ),
                block_sequence=True,
            )
        adapter = self._active_semantic_adapter
        if adapter is None:
            return False
        callback = getattr(adapter, "force_site_reacquire", None)
        if not callable(callback):
            return False
        try:
            callback(
                sequence_name=str(sequence_name or "").strip().lower(),
                reason=str(reason or "").strip().lower(),
                block_sequence=bool(block_sequence),
                site_anchor=list(site_anchor or []),
                step_index=step_index,
            )
        except TypeError:
            return False
        self._semantic_site_actuation_controller_adapter_notify_success_count += 1
        return True

    def _note_site_closer_outcome(
        self,
        *,
        sequence_name: str,
        outcome: str,
        block_sequence: bool = False,
    ) -> None:
        normalized_outcome = str(outcome or "").strip().lower()
        normalized_name = str(sequence_name or "").strip().lower()
        if not normalized_outcome:
            return
        self._semantic_site_closer_outcome_record_count += 1
        self._semantic_site_closer_outcome_histogram[normalized_outcome] = (
            self._semantic_site_closer_outcome_histogram.get(normalized_outcome, 0) + 1
        )
        if normalized_name:
            key = f"{normalized_name}:{normalized_outcome}"
            self._semantic_site_closer_sequence_outcome_histogram[key] = (
                self._semantic_site_closer_sequence_outcome_histogram.get(key, 0) + 1
            )
        if normalized_outcome in {
            "far_tail",
            "far_tail_failed",
            "no_distance_gain",
            "empty_followup",
            "no_contact",
            "exit_before_verify",
        }:
            self._semantic_site_closer_failed_tail_count += 1
        if normalized_outcome in {"closed", "score_delta", "level_delta"}:
            self._semantic_site_closer_closed_count += 1
        if block_sequence and normalized_name:
            self._semantic_site_closer_sequence_blocked_count += 1
            self._semantic_site_closer_blocked_sequence_histogram[normalized_name] = (
                self._semantic_site_closer_blocked_sequence_histogram.get(normalized_name, 0) + 1
            )
            self._semantic_site_closer_active_block_count = max(
                self._semantic_site_closer_active_block_count,
                1,
            )

    def _record_adapter_site_actuation_outcome(
        self,
        *,
        sequence_name: str,
        outcome: str,
        reason: str = "",
        site_anchor: list[int] | None = None,
        step_index: int | None = None,
        block_ttl: int = 0,
    ) -> None:
        self._note_site_closer_outcome(
            sequence_name=sequence_name,
            outcome=outcome,
            block_sequence=block_ttl > 0,
        )
        adapter = self._active_semantic_adapter
        if adapter is None:
            return
        callback = getattr(adapter, "record_site_actuation_outcome", None)
        if not callable(callback):
            return
        try:
            callback(
                sequence_name=str(sequence_name or "").strip().lower(),
                outcome=str(outcome or "").strip().lower(),
                reason=str(reason or "").strip().lower(),
                site_anchor=list(site_anchor or []),
                step_index=step_index,
                block_ttl=int(block_ttl or 0),
            )
        except TypeError:
            return

    @staticmethod
    def _controller_anchor_from_raw(raw_anchor: Any) -> list[int]:
        if not isinstance(raw_anchor, (list, tuple)):
            return []
        anchor = [
            int(value)
            for value in list(raw_anchor)[:2]
            if str(value).strip("-").isdigit()
        ]
        return anchor if len(anchor) >= 2 else []

    def _controller_reacquire_target_anchor(self, lease: dict[str, Any]) -> list[int]:
        for key in (
            "hidden_state_executor_site_anchor",
            "route_target_anchor",
            "hidden_state_executor_target_anchor",
        ):
            anchor = self._controller_anchor_from_raw(lease.get(key))
            if anchor:
                return anchor
        metadata = lease.get("hidden_state_executor_metadata")
        if isinstance(metadata, dict):
            for key in ("site_anchor", "route_target_anchor", "target_anchor"):
                anchor = self._controller_anchor_from_raw(metadata.get(key))
                if anchor:
                    return anchor
        anchor = self._controller_anchor_from_raw(self._last_semantic_route_target_anchor)
        if anchor:
            return anchor
        return []

    def _controller_reacquire_route_actions(
        self,
        lease: dict[str, Any],
        *,
        runtime: Any | None = None,
        prefer_lateral: bool = False,
    ) -> tuple[list[str], bool]:
        player_bbox = self._runtime_semantic_player_bbox(runtime) if runtime is not None else []
        target_anchor = self._controller_reacquire_target_anchor(lease)
        if len(player_bbox) < 4 or len(target_anchor) < 2:
            return ["ACTION2"], False

        player_row = int(player_bbox[0])
        player_col = int(player_bbox[2])
        target_row = int(target_anchor[0])
        target_col = int(target_anchor[1])
        row_gap = target_row - player_row
        col_gap = target_col - player_col
        vertical = ""
        lateral = ""
        if row_gap > 0:
            vertical = "ACTION2"
        elif row_gap < 0:
            vertical = "ACTION1"
        if col_gap > 0:
            lateral = "ACTION4"
        elif col_gap < 0:
            lateral = "ACTION3"

        preferred: list[str] = []
        if prefer_lateral and lateral:
            preferred = [lateral, vertical]
        elif abs(row_gap) >= abs(col_gap):
            preferred = [vertical, lateral]
        else:
            preferred = [lateral, vertical]

        actions: list[str] = []
        for action in preferred:
            if action and action not in actions:
                actions.append(action)
        if actions and len(actions) < 3:
            primary = actions[0]
            if primary in {"ACTION1", "ACTION2"} and abs(row_gap) > 5:
                actions.append(primary)
            elif primary in {"ACTION3", "ACTION4"} and abs(col_gap) > 5:
                actions.append(primary)
        if not actions:
            return ["ACTION2"], False
        lateral_first = actions[0] in {"ACTION3", "ACTION4"}
        return actions[:3], lateral_first

    def _controller_site_closer_context(
        self,
        lease: dict[str, Any],
        *,
        runtime: Any | None = None,
        site_anchor: list[int] | None = None,
        reason: str = "controller_site_closer_alternate",
    ) -> dict[str, Any]:
        context: dict[str, Any] = {"reason": str(reason or "").strip().lower()}
        player_bbox = self._runtime_semantic_player_bbox(runtime) if runtime is not None else []
        if len(player_bbox) >= 4:
            context["player_bbox"] = list(player_bbox[:4])
            context["player_row"] = int(player_bbox[0])
            context["player_col"] = int(player_bbox[2])
        anchor = self._controller_anchor_from_raw(site_anchor)
        if not anchor:
            anchor = self._controller_reacquire_target_anchor(lease)
        if anchor:
            context["site_row"] = int(anchor[0])
            context["site_col"] = int(anchor[1])
        return context

    @staticmethod
    def _controller_alternate_site_closer_names(sequence_name: str) -> list[str]:
        current = str(sequence_name or "").strip().lower()
        ordered: list[str] = []

        def _append(names: list[str]) -> None:
            for name in names:
                normalized = str(name or "").strip().lower()
                if normalized and normalized not in ordered:
                    ordered.append(normalized)

        if "left" in current:
            _append([
                "right_probe_then_return_verify_alt",
                "right_probe_then_return_verify",
                "lateral_adjust_right_then_verify_alt",
                "lateral_adjust_right_then_verify",
            ])
        if "right" in current:
            _append([
                "left_probe_then_return_verify_alt",
                "left_probe_then_return_verify",
                "lateral_adjust_left_then_verify_alt",
                "lateral_adjust_left_then_verify",
            ])
        if "contact" in current or "step_on" in current or "enter" in current:
            _append([
                "left_probe_then_return_verify_alt",
                "right_probe_then_return_verify_alt",
            ])
        _append([
            "contact_then_verify_hold",
            "contact_then_verify",
            "step_on_then_retest_up",
            "enter_then_return",
            "left_probe_then_return_verify_alt",
            "right_probe_then_return_verify_alt",
            "left_probe_then_return_verify",
            "right_probe_then_return_verify",
            "lateral_adjust_left_then_verify_alt",
            "lateral_adjust_right_then_verify_alt",
        ])
        return [name for name in ordered if name != current]

    def _resolve_controller_site_closer_alternate(
        self,
        *,
        sequence_name: str,
        lease: dict[str, Any],
        runtime: Any | None = None,
        site_anchor: list[int] | None = None,
    ) -> dict[str, Any] | None:
        blocked_names = {
            str(name or "").strip().lower()
            for name in set(self._semantic_site_actuation_controller_reacquire_blocked_sequences)
            if str(name or "").strip()
        }
        current = str(sequence_name or "").strip().lower()
        context = self._controller_site_closer_context(
            lease,
            runtime=runtime,
            site_anchor=site_anchor,
            reason="controller_site_closer_alternate",
        )
        for alternate_name in self._controller_alternate_site_closer_names(current):
            if alternate_name in blocked_names:
                continue
            spec = self._resolve_adapter_sequence_spec(
                name=alternate_name,
                context=context,
            )
            if not spec:
                continue
            actions = [
                str(action or "").strip().upper()
                for action in list(spec.get("actions", []) or [])
                if str(action or "").strip().upper() in _MOVE_ACTION_SET
            ]
            resolved_name = str(spec.get("name", "") or alternate_name).strip().lower()
            if actions and resolved_name and resolved_name not in blocked_names:
                spec["actions"] = actions
                return spec
        return None

    def _schedule_controller_site_closer_alternate(
        self,
        *,
        sequence_name: str,
        lease: dict[str, Any],
        runtime: Any | None = None,
        site_anchor: list[int] | None = None,
        step_index: int | None = None,
    ) -> bool:
        normalized_name = str(sequence_name or "").strip().lower()
        if not normalized_name:
            return False
        resolved_anchor = self._controller_anchor_from_raw(site_anchor)
        if not resolved_anchor:
            resolved_anchor = self._controller_reacquire_target_anchor(lease)
        spec = self._resolve_controller_site_closer_alternate(
            sequence_name=normalized_name,
            lease=lease,
            runtime=runtime,
            site_anchor=resolved_anchor,
        )
        if spec is None:
            self._semantic_site_closer_alternate_missing_count += 1
            self._semantic_site_closer_pending_alternate = {}
            return False
        self._semantic_site_closer_pending_alternate = {
            "source_sequence_name": normalized_name,
            "site_anchor": list(resolved_anchor or []),
            "ttl_remaining": 120,
            "opened_step": -1 if step_index is None else int(step_index),
        }
        self._semantic_site_closer_alternate_pending_count += 1
        return True

    def _maybe_apply_pending_controller_site_closer_alternate(
        self,
        lease: dict[str, Any],
        *,
        runtime: Any | None = None,
        step_index: int | None = None,
        site_distance: int | None = None,
        semantic_phase: str = "",
    ) -> bool:
        pending = dict(self._semantic_site_closer_pending_alternate or {})
        if not pending:
            return False
        ttl_remaining = int(pending.get("ttl_remaining", 0) or 0)
        if ttl_remaining <= 0:
            self._semantic_site_closer_alternate_expired_count += 1
            self._semantic_site_closer_pending_alternate = {}
            return False

        ready = bool(site_distance is not None and int(site_distance) <= 14)
        if not ready:
            ttl_remaining -= 1
            if ttl_remaining <= 0:
                self._semantic_site_closer_alternate_expired_count += 1
                self._semantic_site_closer_pending_alternate = {}
            else:
                pending["ttl_remaining"] = ttl_remaining
                self._semantic_site_closer_pending_alternate = pending
            return False

        source_name = str(pending.get("source_sequence_name", "") or "").strip().lower()
        site_anchor = self._controller_anchor_from_raw(pending.get("site_anchor"))
        spec = self._resolve_controller_site_closer_alternate(
            sequence_name=source_name,
            lease=lease,
            runtime=runtime,
            site_anchor=site_anchor,
        )
        if spec is None:
            self._semantic_site_closer_alternate_missing_count += 1
            pending["ttl_remaining"] = ttl_remaining - 1
            self._semantic_site_closer_pending_alternate = pending if pending["ttl_remaining"] > 0 else {}
            return False

        alternate_name = str(spec.get("name", "") or "").strip().lower()
        alternate_actions = [
            str(action or "").strip().upper()
            for action in list(spec.get("actions", []) or [])
            if str(action or "").strip().upper() in _MOVE_ACTION_SET
        ]
        if not alternate_name or not alternate_actions:
            self._semantic_site_closer_alternate_missing_count += 1
            self._semantic_site_closer_pending_alternate = {}
            return False

        self._semantic_site_closer_pending_alternate = {}
        self._semantic_site_closer_alternate_forced_count += 1
        self._semantic_site_closer_sequence_rotated_count += 1
        lease["active"] = True
        lease["mechanics_mode"] = lease.get("mechanics_mode") or "semantic"
        lease["semantic_phase"] = "site_actuation"
        lease["site_actuation_active"] = True
        lease["site_actuation_sequence_name"] = alternate_name
        lease["site_actuation_sequence_index"] = 0
        lease["site_actuation_sequence_remaining"] = list(alternate_actions)
        lease["suggested_actions"] = list(alternate_actions)
        lease["route_preview"] = list(alternate_actions)
        lease["route_source"] = "controller_site_closer_alternate"
        lease["route_target_anchor"] = list(site_anchor or self._controller_reacquire_target_anchor(lease))
        lease["lease_horizon"] = max(1, min(3, len(alternate_actions)))
        lease["priority"] = max(float(lease.get("priority", 0.0) or 0.0), 1.42)
        lease["event_kind"] = "controller_site_closer_alternate"
        lease["site_actuation_controller_alternate_forced"] = True
        lease["hidden_state_executor_active"] = False
        lease["hidden_state_executor_phase"] = ""
        lease["hidden_state_executor_kind"] = ""
        lease["site_closer_sequence_rotated_count"] = self._semantic_site_closer_sequence_rotated_count
        lease["site_closer_alternate_forced_count"] = self._semantic_site_closer_alternate_forced_count
        lease["site_closer_alternate_source_sequence_name"] = source_name
        lease["site_closer_alternate_sequence_name"] = alternate_name
        if step_index is not None:
            self._issue_exact_site_lock(
                step_index=int(step_index),
                sequence_name=alternate_name,
                ttl=min(3, max(1, len(alternate_actions))),
            )
        return True

    def _maybe_apply_controller_site_reacquire(
        self,
        lease: dict[str, Any],
        *,
        runtime: Any | None = None,
        step_index: int | None = None,
    ) -> None:
        """Break a stale site-actuation verify tail with one bounded re-entry step."""
        if self._semantic_site_actuation_controller_reacquire_cooldown_remaining > 0:
            self._semantic_site_actuation_controller_reacquire_cooldown_remaining = max(
                0,
                int(self._semantic_site_actuation_controller_reacquire_cooldown_remaining) - 1,
            )

        sequence_name = str(lease.get("site_actuation_sequence_name", "") or "").strip().lower()
        semantic_phase = str(lease.get("semantic_phase", "") or "").strip().lower()
        try:
            site_distance = None if lease.get("site_distance") is None else int(lease.get("site_distance"))
        except (TypeError, ValueError):
            site_distance = None
        try:
            site_actuation_age = int(lease.get("site_actuation_age", 0) or 0)
        except (TypeError, ValueError):
            site_actuation_age = 0
        tail_actions = [
            str(action or "").strip().upper()
            for action in list(
                lease.get("site_actuation_sequence_remaining")
                or lease.get("suggested_actions")
                or []
            )
            if str(action or "").strip().upper() in _MOVE_ACTION_SET
        ]

        def _apply_reacquire_route(event_kind: str, *, prefer_lateral: bool = False) -> None:
            route_actions, lateral_first = self._controller_reacquire_route_actions(
                lease,
                runtime=runtime,
                prefer_lateral=prefer_lateral,
            )
            if route_actions:
                self._semantic_site_actuation_controller_reacquire_last_action = route_actions[0]
                self._semantic_site_actuation_controller_route_reentry_count += 1
                if lateral_first:
                    self._semantic_site_actuation_controller_route_reentry_lateral_count += 1
            lease["active"] = True
            lease["mechanics_mode"] = lease.get("mechanics_mode") or "semantic"
            lease["semantic_phase"] = "site_reaching"
            lease["site_actuation_active"] = False
            lease["site_actuation_sequence_remaining"] = []
            lease["suggested_actions"] = list(route_actions)
            lease["route_preview"] = list(route_actions)
            lease["route_source"] = "controller_site_reacquire"
            lease["route_target_anchor"] = self._controller_reacquire_target_anchor(lease)
            lease["lease_horizon"] = 1
            lease["priority"] = max(float(lease.get("priority", 0.0) or 0.0), 1.35)
            lease["event_kind"] = event_kind
            lease["exact_site_tx_active"] = False
            lease["exact_site_tx_phase"] = ""
            lease["exact_site_tx_expected_actions"] = []
            lease["exact_site_tx_ttl_remaining"] = 0
            lease["site_actuation_controller_reacquire"] = True

        if self._maybe_apply_pending_controller_site_closer_alternate(
            lease,
            runtime=runtime,
            step_index=step_index,
            site_distance=site_distance,
            semantic_phase=semantic_phase,
        ):
            return

        tail_like = bool(
            semantic_phase.startswith("site_actuation")
            and sequence_name
            and site_distance is not None
            and site_distance >= 15
            and site_actuation_age >= 7
            and (
                not tail_actions
                or (
                    len(tail_actions) == 1
                    and tail_actions[0] == "ACTION1"
                )
            )
        )
        if not tail_like and self._semantic_site_actuation_controller_reacquire_hold_remaining <= 0:
            self._semantic_site_actuation_controller_reacquire_hold_remaining = 0
            self._semantic_site_actuation_controller_reacquire_last_distance = None
            self._semantic_site_actuation_controller_reacquire_last_action = ""
            self._semantic_site_actuation_controller_reacquire_lateral_recovery_used = False
            self._semantic_site_actuation_controller_far_tail_streak_name = ""
            self._semantic_site_actuation_controller_far_tail_streak_count = 0
            return

        if self._semantic_site_actuation_controller_reacquire_hold_remaining > 0:
            last_distance = self._semantic_site_actuation_controller_reacquire_last_distance
            if last_distance is not None and site_distance is not None and site_distance >= last_distance:
                lateral_route, lateral_first = self._controller_reacquire_route_actions(
                    lease,
                    runtime=runtime,
                    prefer_lateral=True,
                )
                if (
                    not self._semantic_site_actuation_controller_reacquire_lateral_recovery_used
                    and lateral_first
                    and lateral_route
                    and lateral_route[0] != self._semantic_site_actuation_controller_reacquire_last_action
                ):
                    self._semantic_site_actuation_controller_reacquire_lateral_recovery_used = True
                    self._semantic_site_actuation_controller_reacquire_hold_remaining = max(
                        0,
                        int(self._semantic_site_actuation_controller_reacquire_hold_remaining) - 1,
                    )
                    self._semantic_site_actuation_controller_reacquire_last_distance = site_distance
                    if self._exact_site_tx_active:
                        self._complete_exact_site_transaction(
                            reason="controller_far_tail_reacquire_lateral_hold"
                        )
                    self._site_actuation_contract_active = False
                    self._site_actuation_contract_armed = False
                    self._site_actuation_contract_result_pending = False
                    self._site_actuation_contract_pending_verify_name = ""
                    self._site_actuation_contract_pending_verify_steps = []
                    _apply_reacquire_route(
                        "controller_site_reacquire_lateral_hold",
                        prefer_lateral=True,
                    )
                    return
                self._semantic_site_actuation_controller_reacquire_hold_remaining = 0
                self._semantic_site_actuation_controller_reacquire_last_distance = None
                self._semantic_site_actuation_controller_reacquire_last_action = ""
                self._semantic_site_actuation_controller_reacquire_lateral_recovery_used = False
                self._semantic_site_actuation_controller_reacquire_blocked_sequences.add(sequence_name)
                if self._exact_site_tx_active:
                    self._abort_exact_site_transaction(
                        "controller_far_tail_reacquire_no_distance_gain"
                    )
                self._release_exact_site_lock()
                self._site_actuation_contract_active = False
                self._site_actuation_contract_armed = False
                self._site_actuation_contract_result_pending = False
                self._site_actuation_contract_pending_verify_name = ""
                self._site_actuation_contract_pending_verify_steps = []
                site_anchor = self._controller_reacquire_target_anchor(lease)
                self._notify_adapter_site_reacquire(
                    sequence_name=sequence_name,
                    reason="controller_far_tail_reacquire_no_distance_gain",
                    block_sequence=True,
                    site_anchor=site_anchor,
                    step_index=step_index,
                )
                self._schedule_controller_site_closer_alternate(
                    sequence_name=sequence_name,
                    lease=lease,
                    runtime=runtime,
                    site_anchor=site_anchor,
                    step_index=step_index,
                )
                self._semantic_site_actuation_controller_reacquire_cooldown_remaining = max(
                    int(self._semantic_site_actuation_controller_reacquire_cooldown_remaining or 0),
                    12,
                )
                self._semantic_site_actuation_controller_far_tail_streak_name = ""
                self._semantic_site_actuation_controller_far_tail_streak_count = 0
                if self._maybe_apply_pending_controller_site_closer_alternate(
                    lease,
                    runtime=runtime,
                    step_index=step_index,
                    site_distance=site_distance,
                    semantic_phase=semantic_phase,
                ):
                    return
                _apply_reacquire_route("controller_site_reacquire_blocked_sequence")
                return
            self._semantic_site_actuation_controller_reacquire_hold_remaining = max(
                0,
                int(self._semantic_site_actuation_controller_reacquire_hold_remaining) - 1,
            )
            self._semantic_site_actuation_controller_reacquire_last_distance = site_distance
            if self._exact_site_tx_active:
                self._complete_exact_site_transaction(reason="controller_far_tail_reacquire_hold")
            self._site_actuation_contract_active = False
            self._site_actuation_contract_armed = False
            self._site_actuation_contract_result_pending = False
            self._site_actuation_contract_pending_verify_name = ""
            self._site_actuation_contract_pending_verify_steps = []
            _apply_reacquire_route("controller_site_reacquire_hold")
            return

        if sequence_name == self._semantic_site_actuation_controller_far_tail_streak_name:
            self._semantic_site_actuation_controller_far_tail_streak_count += 1
        else:
            self._semantic_site_actuation_controller_far_tail_streak_name = sequence_name
            self._semantic_site_actuation_controller_far_tail_streak_count = 1

        lease_tx_active = bool(lease.get("exact_site_tx_active", self._exact_site_tx_active))
        lease_tx_phase = str(lease.get("exact_site_tx_phase", self._exact_site_tx_phase) or "").strip().lower()
        lease_contract_remaining = [
            str(action or "").strip().upper()
            for action in list(lease.get("site_actuation_contract_remaining", []) or [])
            if str(action or "").strip().upper() in _MOVE_ACTION_SET
        ]
        lease_contract_active = bool(lease.get("site_actuation_contract_active", False)) or bool(
            lease_contract_remaining
        )
        lease_pending_verify_name = str(
            lease.get("site_actuation_pending_verify_name", self._site_actuation_contract_pending_verify_name) or ""
        ).strip()
        lease_pending_verify_remaining = [
            str(action or "").strip().upper()
            for action in list(lease.get("site_actuation_pending_verify_remaining", []) or [])
            if str(action or "").strip().upper() in _MOVE_ACTION_SET
        ]
        if (
            self._semantic_site_actuation_controller_far_tail_streak_count < 6
            or self._semantic_site_actuation_controller_reacquire_cooldown_remaining > 0
            or sequence_name in self._semantic_site_actuation_controller_reacquire_blocked_sequences
            or not lease_tx_active
            or lease_tx_phase != "followup"
            or lease_contract_active
            or lease_pending_verify_name
            or lease_pending_verify_remaining
        ):
            return

        if self._exact_site_tx_active:
            self._complete_exact_site_transaction(reason="controller_far_tail_reacquire")
        else:
            self._exact_site_tx_phase = ""
            self._exact_site_tx_expected_actions = []
            self._exact_site_tx_executed_actions = []
            self._exact_site_tx_cursor = 0
            self._exact_site_tx_ttl_remaining = 0
            self._exact_site_tx_retry_budget = 0
            self._exact_site_tx_abort_reason = "controller_far_tail_reacquire"
        self._site_actuation_contract_active = False
        self._site_actuation_contract_armed = False
        self._site_actuation_contract_result_pending = False
        self._site_actuation_contract_pending_verify_name = ""
        self._site_actuation_contract_pending_verify_steps = []
        self._release_exact_site_lock()
        self._semantic_site_actuation_controller_reacquire_issue_count += 1
        self._semantic_site_actuation_controller_reacquire_cooldown_remaining = 4
        self._semantic_site_actuation_controller_reacquire_hold_remaining = 4
        self._semantic_site_actuation_controller_reacquire_last_distance = site_distance
        self._semantic_site_actuation_controller_reacquire_last_action = ""
        self._semantic_site_actuation_controller_reacquire_lateral_recovery_used = False
        self._semantic_site_actuation_controller_far_tail_streak_count = 0

        _apply_reacquire_route("controller_site_reacquire")

    def _execute_hidden_state_executor(
        self,
        *,
        agent: Any | None,
        semantic_action_lease: dict[str, Any] | None,
    ) -> list[PlannedAction]:
        semantic_action_lease = semantic_action_lease or {}
        if not bool(semantic_action_lease.get("hidden_state_executor_active", False)):
            return []
        self._clear_agent_queue_for_hidden_state(
            agent=agent,
            semantic_action_lease=semantic_action_lease,
        )
        semantic_phase = str(semantic_action_lease.get("semantic_phase", "") or "").strip().lower()
        site_actuation_sequence_name = str(
            semantic_action_lease.get("site_actuation_sequence_name", "") or ""
        ).strip().lower()
        sequence_remaining = [
            str(action_name or "").strip().upper()
            for action_name in list(semantic_action_lease.get("site_actuation_sequence_remaining", []) or [])
            if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
        ]
        try:
            level_number = int(semantic_action_lease.get("level_number", 0) or 0)
        except (TypeError, ValueError):
            level_number = 0
        lease_horizon = max(1, int(semantic_action_lease.get("lease_horizon", 1) or 1))
        route_preview = [
            str(action_name or "").strip().upper()
            for action_name in list(semantic_action_lease.get("route_preview", []) or [])
            if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
        ]
        if sequence_remaining and (
            level_number >= 5
            and (
            semantic_phase.startswith("site_actuation")
            or bool(semantic_action_lease.get("site_actuation_active", False))
            )
        ):
            route_preview = list(sequence_remaining)
            lease_horizon = max(lease_horizon, min(3, len(sequence_remaining)))
        if not route_preview:
            route_preview = [
                str(action_name or "").strip().upper()
                for action_name in list(semantic_action_lease.get("suggested_actions", []) or [])
                if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
            ]
        if not route_preview:
            return []
        event_kind = str(semantic_action_lease.get("event_kind", "") or "").strip() or "hidden_state_executor"
        self._note_semantic_bridge_choice(
            semantic_action_lease=semantic_action_lease,
            action_name=route_preview[0],
            source="hidden_state_executor",
        )
        return [
            PlannedAction(
                action_name=action_name,
                reason=f"hidden_state_executor:{event_kind}",
            )
            for action_name in route_preview[: max(1, min(3, lease_horizon))]
        ]

    def _build_hidden_abort_recovery_actions(
        self,
        *,
        agent: Any | None,
        semantic_action_lease: dict[str, Any] | None,
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
    ) -> list[PlannedAction]:
        if self._post_stabilize_no_target_continuation_step_active:
            return []
        semantic_action_lease = semantic_action_lease or {}
        hidden_state = semantic_action_lease.get("hidden_state_executor")
        if not isinstance(hidden_state, dict):
            return []
        if bool(semantic_action_lease.get("hidden_state_executor_active", False)):
            return []
        if bool(hidden_state.get("active", False)):
            return []
        if str(hidden_state.get("completed_reason", "") or "").strip():
            return []
        abort_reason = str(hidden_state.get("abort_reason", "") or "").strip().lower()
        if abort_reason not in _HIDDEN_STATE_ABORT_RECOVERY_REASONS:
            return []

        allowed_set = set(_MOVE_ACTIONS)
        if allowed_actions is not None:
            allowed_set = {
                str(action_name or "").strip().upper()
                for action_name in list(allowed_actions or [])
                if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
            }
        blocked_set = {
            str(action_name or "").strip().upper()
            for action_name in set(blocked_actions or set())
            if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
        }
        if not allowed_set:
            return []

        candidates: list[str] = []

        def _append_move_actions(raw_actions: Any) -> None:
            for raw_action in list(raw_actions or []):
                action_name = str(raw_action or "").strip().upper()
                if (
                    action_name in _MOVE_ACTION_SET
                    and action_name in allowed_set
                    and action_name not in blocked_set
                    and action_name not in candidates
                ):
                    candidates.append(action_name)

        _append_move_actions(semantic_action_lease.get("route_preview", []))
        _append_move_actions(semantic_action_lease.get("suggested_actions", []))
        _append_move_actions(semantic_action_lease.get("site_actuation_sequence_remaining", []))
        _append_move_actions(hidden_state.get("route_preview", []))
        hidden_metadata = hidden_state.get("metadata", {})
        if isinstance(hidden_metadata, dict):
            _append_move_actions(hidden_metadata.get("route_preview", []))
            _append_move_actions(hidden_metadata.get("suggested_actions", []))
        _append_move_actions(_MOVE_ACTIONS)
        if not candidates:
            return []

        do_not_repeat_set = {
            str(action_name or "").strip().upper()
            for action_name in list(semantic_action_lease.get("do_not_repeat_actions", []) or [])
            if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
        }
        recovery_candidates = [
            action_name
            for action_name in candidates
            if action_name not in do_not_repeat_set
        ] or list(candidates)

        scoring_lease = dict(semantic_action_lease)
        scoring_lease["active"] = True
        scoring_lease["suggested_actions"] = list(recovery_candidates)
        selected_action = self._semantic_preferred_move_action(
            agent=agent,
            fallback_action=recovery_candidates[0],
            semantic_action_lease=scoring_lease,
        )
        if selected_action not in recovery_candidates:
            selected_action = recovery_candidates[0]

        note_lease = dict(semantic_action_lease)
        note_lease["event_kind"] = (
            str(note_lease.get("event_kind", "") or "").strip()
            or f"hidden_state_abort:{abort_reason}"
        )
        self._note_semantic_bridge_choice(
            semantic_action_lease=note_lease,
            action_name=selected_action,
            source="hidden_state_abort_recovery",
        )
        return [
            PlannedAction(
                action_name=selected_action,
                reason=f"hidden_state_abort_recovery:{abort_reason}",
            )
        ]

    @staticmethod
    def _normalized_available_actions(
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
    ) -> list[str]:
        raw_allowed = list(
            allowed_actions
            or ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
        )
        blocked = {
            str(action_name or "").strip().upper()
            for action_name in set(blocked_actions or set())
            if str(action_name or "").strip().upper()
        }
        normalized: list[str] = []
        seen: set[str] = set()
        for action_name in raw_allowed:
            candidate = str(action_name or "").strip().upper()
            if not candidate or candidate in blocked or candidate in seen:
                continue
            if candidate == "RESET" or _GENERIC_KEYBOARD_ACTION_RE.match(candidate):
                normalized.append(candidate)
                seen.add(candidate)
        return normalized

    @staticmethod
    def _recent_local_action_names(runtime: Any, *, limit: int = 18) -> list[str]:
        rows = getattr(runtime, "step_metrics", None)
        if not isinstance(rows, list):
            return []
        names: list[str] = []
        for row in rows[-max(1, int(limit or 1)):]:
            if not isinstance(row, dict):
                continue
            preview = row.get("local_controller_action_preview")
            if isinstance(preview, list) and preview:
                name = str(preview[0] or "").strip().upper()
            else:
                name = str(row.get("action", "") or "").strip().upper()
                if "(" in name:
                    name = name.split("(", 1)[0].strip()
            if name:
                names.append(name)
        return names

    @staticmethod
    def _recent_click_board_change_count(runtime: Any, *, limit: int = 10) -> int:
        rows = getattr(runtime, "step_metrics", None)
        if not isinstance(rows, list):
            return 0
        count = 0
        for row in rows[-max(1, int(limit or 1)):]:
            if not isinstance(row, dict):
                continue
            preview = row.get("local_controller_action_preview")
            if isinstance(preview, list):
                names = [str(item or "").strip().upper() for item in preview]
            else:
                name = str(row.get("action", "") or "").strip().upper()
                if "(" in name:
                    name = name.split("(", 1)[0].strip()
                names = [name] if name else []
            if _CLICK_ACTION not in names:
                continue
            if "no_change" in row or "no_state_change" in row:
                if bool(row.get("no_change", row.get("no_state_change", False))):
                    continue
            elif "board_changed" in row:
                if not bool(row.get("board_changed", False)):
                    continue
            else:
                continue
            count += 1
        return count

    def _note_l1_action_family_quota_blocked(self, reason: str) -> None:
        key = str(reason or "").strip() or "unknown"
        self._l1_action_family_quota_blocked_count += 1
        self._l1_action_family_quota_blocked_reason_counts[key] = (
            int(self._l1_action_family_quota_blocked_reason_counts.get(key, 0) or 0) + 1
        )

    @staticmethod
    def _l1_lateral_probe_anchor_allowed(selected_target: ScoredCandidate | None) -> bool:
        if selected_target is None:
            return False
        row = int(getattr(selected_target, "center_row", 0) or 0)
        if row <= 0:
            parts = [part.strip() for part in str(getattr(selected_target, "anchor_key", "") or "").split("||")]
            if len(parts) >= 3:
                try:
                    row = int(parts[-2])
                except (TypeError, ValueError):
                    row = 0
        return row >= 30

    def _build_l1_lateral_probe_lease_actions(
        self,
        *,
        runtime: Any,
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
        semantic_action_lease: dict[str, Any],
        hidden_state_lease: HiddenStateExecutorLease,
    ) -> list[PlannedAction]:
        if bool(semantic_action_lease.get("active", False)) or bool(hidden_state_lease.active):
            self._l1_lateral_probe_remaining = 0
            self._l1_lateral_probe_queue = []
            return []
        if self._runtime_level_number(runtime) > 1 or self._runtime_score(runtime) > 0:
            self._l1_lateral_probe_remaining = 0
            self._l1_lateral_probe_component_id = ""
            self._l1_lateral_probe_queue = []
            return []
        if not self._l1_lateral_probe_queue:
            self._l1_lateral_probe_remaining = 0
            return []
        available = self._normalized_available_actions(allowed_actions, blocked_actions)
        action_name = str(self._l1_lateral_probe_queue.pop(0) or "").strip().upper()
        self._l1_lateral_probe_remaining = len(self._l1_lateral_probe_queue)
        if action_name not in available:
            self._l1_lateral_probe_remaining = 0
            self._l1_lateral_probe_queue = []
            return []
        self._l1_lateral_probe_issued_count += 1
        return [
            PlannedAction(
                action_name=action_name,
                target_component_id=str(self._l1_lateral_probe_component_id or ""),
                parameters={},
                reason="l1_lateral_probe_lease",
            )
        ]

    def _build_l1_action_family_quota_actions(
        self,
        *,
        runtime: Any,
        step_index: int,
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
        semantic_action_lease: dict[str, Any],
        hidden_state_lease: HiddenStateExecutorLease,
    ) -> list[PlannedAction]:
        # Phase 8.B diagnostic: count early returns by reason so we can
        # see why the quota never fires on fresh games. Counters live on
        # `_l1_action_family_quota_blocked_reason_counts` (already part of
        # the dump) under `early_*` keys to keep them distinct from the
        # post-starve `_note_l1_action_family_quota_blocked` reasons.
        def _early_bail(tag: str) -> list[PlannedAction]:
            self._l1_action_family_quota_blocked_reason_counts[f"early_{tag}"] = (
                int(self._l1_action_family_quota_blocked_reason_counts.get(f"early_{tag}", 0) or 0) + 1
            )
            return []
        self._l1_action_family_quota_blocked_reason_counts["early_called"] = (
            int(self._l1_action_family_quota_blocked_reason_counts.get("early_called", 0) or 0) + 1
        )
        if bool(semantic_action_lease.get("active", False)) or bool(hidden_state_lease.active):
            return _early_bail("lease_active")
        if self._runtime_level_number(runtime) > 1:
            return _early_bail("level_gt_1")
        if self._runtime_score(runtime) > 0:
            return _early_bail("score_gt_0")
        if int(step_index) < 10:
            return _early_bail("step_lt_10")

        recent_actions = self._recent_local_action_names(runtime, limit=18)
        if len(recent_actions) < 8:
            return _early_bail("recent_lt_8")
        click_count = sum(1 for action_name in recent_actions if action_name == _CLICK_ACTION)
        non_click_count = sum(
            1
            for action_name in recent_actions
            if action_name and action_name != _CLICK_ACTION and action_name != "RESET"
        )
        if click_count < max(7, int(len(recent_actions) * 0.70)):
            return _early_bail(f"click_below_thresh:{click_count}/{len(recent_actions)}")

        self._l1_family_starved_count += 1
        attempt_override = bool(
            not self._move_action_available
            and self._l1_attempt_outcome_record_count >= 2
        )
        # Phase 8.B (2nd attempt): when the keyboard-click flag is on and
        # the agent has been L1 click-starved for many consecutive steps
        # without scoring, bypass the click_progress_active bail. The
        # 19-game breadth diagnostic showed bp35-style games triggering
        # this bail 391/401 steps because clicks DO change the board but
        # never score. Visual stir is not progress for an L1 clear gate.
        score_starved_override = bool(
            _l1_quota_keyboard_click_enabled()
            and int(self._l1_family_starved_count) >= 30
            and self._runtime_score(runtime) <= 0
        )
        if (
            self._recent_click_board_change_count(runtime, limit=4) >= 2
            and not attempt_override
            and not score_starved_override
        ):
            self._note_l1_action_family_quota_blocked("click_progress_active")
            return []
        if int(step_index) - int(self._l1_action_family_quota_last_step) < 7:
            self._note_l1_action_family_quota_blocked("cooldown")
            return []

        available = self._normalized_available_actions(allowed_actions, blocked_actions)
        move_in_surface = any(action_name in _MOVE_ACTION_SET for action_name in available)
        if move_in_surface:
            # Phase 8.B: when the keyboard-click flag is on, do not bail
            # purely on "move family is in the action surface" — the
            # 19-game breadth test showed that on `keyboard_click` games
            # (bp35, lf52, sp80, tn36, s5i5) the surface includes
            # ACTION1-4 but the controller still emits 100% ACTION6,
            # never giving keyboard a real chance. Instead, bail only
            # when the recent local-action window already shows ≥ 2
            # move actions (rotation is actually happening).
            if not _l1_quota_keyboard_click_enabled():
                self._note_l1_action_family_quota_blocked("move_family_available")
                return []
            recent_move_count = sum(
                1 for action_name in recent_actions if action_name in _MOVE_ACTION_SET
            )
            # Phase 8.G-2 follow-up: when LLM force-fire is also active,
            # LLM-driven keyboard rotation pushes recent_move_count above
            # the bail threshold and prevents the quota from EVER firing,
            # which keeps the L1 escape hatch
            # (`allow_l1_auxiliary_discovery` in rule_discovery.py) closed
            # forever. Override the bail when the quota has NEVER fired
            # this episode AND the agent is still score-starved — at
            # least one quota fire is needed to open the escape hatch
            # so rule_discovery can record on L1.
            never_fired_override = (
                int(self._l1_action_family_quota_issued_count) == 0
                and self._runtime_score(runtime) <= 0
                and int(self._l1_family_starved_count) >= 20
            )
            if recent_move_count >= 2 and not never_fired_override:
                self._note_l1_action_family_quota_blocked("move_family_recently_active")
                return []
        candidates = [
            action_name
            for action_name in available
            if action_name not in {_CLICK_ACTION, "RESET"}
        ]
        if not candidates:
            self._note_l1_action_family_quota_blocked("no_non_click_available")
            return []

        recent_hist = {
            action_name: sum(1 for recent in recent_actions if recent == action_name)
            for action_name in candidates
        }

        def rank(action_name: str) -> tuple[int, int, int]:
            match = _GENERIC_KEYBOARD_ACTION_RE.match(action_name)
            action_index = int(match.group(1)) if match else 99
            auxiliary = 0 if action_name not in _MOVE_ACTION_SET else 1
            rotated = (
                action_index - int(self._l1_action_family_quota_cursor)
            ) % 32
            return (int(recent_hist.get(action_name, 0) or 0), auxiliary, rotated)

        chosen = min(candidates, key=rank)
        self._l1_action_family_quota_cursor += 1
        self._l1_action_family_quota_last_step = int(step_index)
        self._l1_action_family_quota_issued_count += 1
        self._l1_action_family_quota_action_histogram[chosen] = (
            int(self._l1_action_family_quota_action_histogram.get(chosen, 0) or 0) + 1
        )
        reason = "click_dominant_l1_zero_score"
        if non_click_count <= 0:
            reason = "click_only_l1_zero_score"
        self._l1_action_family_quota_reason_histogram[reason] = (
            int(self._l1_action_family_quota_reason_histogram.get(reason, 0) or 0) + 1
        )
        return [
            PlannedAction(
                action_name=chosen,
                target_component_id="",
                parameters={},
                reason=f"l1_action_family_quota:{reason}",
            )
        ]

    def _build_generic_exploration_fallback_actions(
        self,
        *,
        runtime: Any,
        selected_target: ScoredCandidate | None,
        agent: Any | None,
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
        reason: str,
    ) -> list[PlannedAction]:
        allowed_set = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"}
        if allowed_actions is not None:
            allowed_set = {
                str(action_name or "").strip().upper()
                for action_name in list(allowed_actions or [])
                if str(action_name or "").strip().upper() in allowed_set
            }
        blocked_set = {
            str(action_name or "").strip().upper()
            for action_name in set(blocked_actions or set())
            if str(action_name or "").strip().upper()
        }
        if not allowed_set:
            return []

        normalized_reason = str(reason or "planner_failed").strip().lower()
        post_stabilize_no_target = self._post_stabilize_no_target_continuation_step_active
        if selected_target is None:
            fallback_candidate = None
            if not post_stabilize_no_target:
                base_pool = getattr(runtime, "_last_base_pool", None)
                fallback_candidate = self._pick_base_candidate(
                    candidates=list(getattr(base_pool, "candidates", None) or [])
                )
            if fallback_candidate is not None:
                family = str(fallback_candidate.family or "").strip().lower()
                component_id = str(fallback_candidate.component_id or "").strip()
                if family == "auxiliary":
                    auxiliary_action = _infer_auxiliary_action_name(
                        action_name=getattr(fallback_candidate, "action_name", ""),
                        component_id=component_id,
                    )
                    if auxiliary_action and auxiliary_action in allowed_set and auxiliary_action not in blocked_set:
                        return [
                            PlannedAction(
                                action_name=auxiliary_action,
                                target_component_id=component_id,
                                parameters={},
                                reason=f"generic_exploration_fallback:{normalized_reason}:pool_auxiliary",
                            )
                        ]
                if family in {"click", "select", ""} and "ACTION6" in allowed_set and "ACTION6" not in blocked_set:
                    row = int(fallback_candidate.center_row or 0)
                    col = int(fallback_candidate.center_col or 0)
                    return [
                        PlannedAction(
                            action_name="ACTION6",
                            target_component_id=component_id,
                            parameters={"x": row, "y": col},
                            reason=f"generic_exploration_fallback:{normalized_reason}:pool_click",
                        )
                    ]
                if family == "move":
                    move_candidates = [
                        action_name
                        for action_name in _MOVE_ACTIONS
                        if action_name in allowed_set and action_name not in blocked_set
                    ]
                    if move_candidates:
                        preferred_move = self._stale_move_redirect_action(agent) or self._dominant_move_action(agent)
                        if preferred_move not in move_candidates:
                            preferred_move = move_candidates[0]
                        return [
                            PlannedAction(
                                action_name=preferred_move,
                                target_component_id=component_id,
                                parameters={},
                                reason=f"generic_exploration_fallback:{normalized_reason}:pool_move",
                            )
                        ]
            move_candidates = [
                action_name
                for action_name in _MOVE_ACTIONS
                if action_name in allowed_set and action_name not in blocked_set
            ]
            if not move_candidates:
                return []
            preferred_move = (
                self._post_stabilize_metric_dominant_move(runtime)
                or self._dominant_move_action(agent)
                if post_stabilize_no_target
                else self._stale_move_redirect_action(agent) or self._dominant_move_action(agent)
            )
            if preferred_move not in move_candidates:
                preferred_move = move_candidates[0]
            return [
                PlannedAction(
                    action_name=preferred_move,
                    target_component_id="",
                    parameters={},
                    reason=f"generic_exploration_fallback:{normalized_reason}:move",
                )
            ]

        component_id = str(selected_target.component_id or "").strip()
        selected_family = str(selected_target.action_family or "").strip().lower()
        if selected_family == "auxiliary":
            auxiliary_action = _infer_auxiliary_action_name(
                action_name=str(selected_target.action_name or ""),
                component_id=component_id,
            )
            if auxiliary_action and auxiliary_action in allowed_set and auxiliary_action not in blocked_set:
                return [
                    PlannedAction(
                        action_name=auxiliary_action,
                        target_component_id=component_id,
                        parameters={},
                        reason=f"generic_exploration_fallback:{normalized_reason}:auxiliary",
                    )
                ]
            return []
        if "ACTION6" in allowed_set and "ACTION6" not in blocked_set:
            position = self._resolve_target_grid_position(
                runtime=runtime,
                selected_target=selected_target,
                target_component_id=component_id,
                parameters={},
            )
            if position is not None:
                row, col = position
                return [
                    PlannedAction(
                        action_name="ACTION6",
                        target_component_id=component_id,
                        parameters={"x": int(row), "y": int(col)},
                        reason=f"generic_exploration_fallback:{normalized_reason}:click",
                    )
                ]

        move_candidates = [
            action_name
            for action_name in _MOVE_ACTIONS
            if action_name in allowed_set and action_name not in blocked_set
        ]
        if not move_candidates:
            return []
        preferred_move = self._stale_move_redirect_action(agent) or self._dominant_move_action(agent)
        if preferred_move not in move_candidates:
            preferred_move = move_candidates[0]
        return [
            PlannedAction(
                action_name=preferred_move,
                target_component_id=component_id,
                parameters={},
                reason=f"generic_exploration_fallback:{normalized_reason}:move",
            )
        ]

    @staticmethod
    def _gray_changed_bit(previous_index: int, next_index: int) -> int:
        previous_gray = int(previous_index) ^ (int(previous_index) >> 1)
        next_gray = int(next_index) ^ (int(next_index) >> 1)
        diff = previous_gray ^ next_gray
        if diff <= 0:
            return 0
        return (diff & -diff).bit_length() - 1

    def _build_structural_suggested_local_test_actions(
        self,
        *,
        runtime: Any,
        agent: Any | None,
        semantic_action_lease: dict[str, Any],
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
        step_index: int,
    ) -> list[PlannedAction]:
        if self._post_stabilize_no_target_continuation_step_active:
            return []
        if bool(semantic_action_lease.get("active", False)) or agent is None:
            return []
        if not self._should_start_toggle_combination_search(runtime):
            return []
        allowed_set = {
            str(action_name or "").strip().upper()
            for action_name in list(allowed_actions or ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"])
            if str(action_name or "").strip().upper()
        }
        blocked_set = {
            str(action_name or "").strip().upper()
            for action_name in set(blocked_actions or set())
            if str(action_name or "").strip().upper()
        }
        build_memory = getattr(runtime, "build_stable_working_memory", None)
        if not callable(build_memory):
            return []
        try:
            working_memory = build_memory(
                agent=agent,
                level_number=self._runtime_level_number(runtime),
            )
        except Exception:
            return []
        hypothesis_card = dict((working_memory or {}).get("hypothesis_card") or {})
        for raw_action in list(hypothesis_card.get("suggested_local_test") or [])[:3]:
            if not isinstance(raw_action, dict):
                continue
            action_name = str(raw_action.get("action", "") or "").strip().upper()
            if action_name not in allowed_set or action_name in blocked_set:
                continue
            if action_name == "ACTION6":
                try:
                    row = int(raw_action.get("x"))
                    col = int(raw_action.get("y"))
                except (TypeError, ValueError):
                    continue
                key = f"ACTION6:{row}:{col}"
                recent_outcome = self._recent_action6_outcome(
                    agent,
                    row=row,
                    col=col,
                    limit=10,
                )
                if recent_outcome["score"] <= 0 and recent_outcome["weak"] > 0:
                    continue
                if (
                    key == self._last_structural_suggested_action_key
                    and int(step_index) - self._last_structural_suggested_action_turn < 6
                ):
                    continue
                self._last_structural_suggested_action_key = key
                self._last_structural_suggested_action_turn = int(step_index)
                return [
                    PlannedAction(
                        action_name="ACTION6",
                        parameters={"x": row, "y": col},
                        reason="structural_suggested_local_test",
                    )
                ]
            if action_name in _MOVE_ACTION_SET:
                key = action_name
                if (
                    key == self._last_structural_suggested_action_key
                    and int(step_index) - self._last_structural_suggested_action_turn < 4
                ):
                    continue
                self._last_structural_suggested_action_key = key
                self._last_structural_suggested_action_turn = int(step_index)
                return [
                    PlannedAction(
                        action_name=action_name,
                        reason="structural_suggested_local_test",
                    )
                ]
        return []

    def _build_toggle_combination_search_actions(
        self,
        *,
        runtime: Any,
        semantic_action_lease: dict[str, Any],
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
    ) -> list[PlannedAction]:
        if bool(semantic_action_lease.get("active", False)):
            return []
        allowed_set = {
            str(action_name or "").strip().upper()
            for action_name in list(allowed_actions or ["ACTION6"])
            if str(action_name or "").strip().upper()
        }
        blocked_set = {
            str(action_name or "").strip().upper()
            for action_name in set(blocked_actions or set())
            if str(action_name or "").strip().upper()
        }
        if "ACTION6" not in allowed_set or "ACTION6" in blocked_set:
            self._reset_toggle_combination_search()
            return []

        current_score = self._runtime_score(runtime)
        current_level = self._runtime_level_number(runtime)
        if (
            self._toggle_search_active
            and self._toggle_search_last_score is not None
            and (
                current_score != self._toggle_search_last_score
                or current_level != self._toggle_search_last_level
            )
        ):
            self._reset_toggle_combination_search()
            return []

        if not self._toggle_search_active:
            if not self._should_start_toggle_combination_search(runtime):
                return []
            targets = self._collect_toggle_combination_targets(runtime)
            if len(targets) < 4:
                return []
            self._toggle_search_active = True
            self._toggle_search_targets = targets[:10]
            self._toggle_search_gray_index = 0

        if not self._toggle_search_targets:
            self._reset_toggle_combination_search()
            return []

        max_index = 1 << len(self._toggle_search_targets)
        if self._toggle_search_gray_index + 1 >= max_index:
            self._reset_toggle_combination_search()
            return []

        next_index = self._toggle_search_gray_index + 1
        bit_index = self._gray_changed_bit(self._toggle_search_gray_index, next_index)
        self._toggle_search_gray_index = next_index
        self._toggle_search_last_score = current_score
        self._toggle_search_last_level = current_level

        if bit_index < 0 or bit_index >= len(self._toggle_search_targets):
            return []
        target = self._toggle_search_targets[bit_index]
        return [
            PlannedAction(
                action_name="ACTION6",
                target_component_id=str(target.get("component_id", "") or ""),
                parameters={
                    "x": int(target.get("row", 0) or 0),
                    "y": int(target.get("col", 0) or 0),
                },
                reason="toggle_combination_search:gray_code",
            )
        ]

    def _should_start_toggle_combination_search(self, runtime: Any) -> bool:
        health_result = getattr(runtime, "_last_target_health_result", None)
        raw_health = getattr(health_result, "health", "") if health_result is not None else ""
        health_name = str(getattr(raw_health, "value", raw_health) or "").strip().lower()
        attempt_forced = bool(
            not self._move_action_available
            and self._runtime_score(runtime) <= 0
            and self._runtime_level_number(runtime) <= 1
            and self._l1_attempt_outcome_record_count >= 2
        )
        if "live" in health_name and not attempt_forced:
            return False
        if self._base_pool_has_live_local_probe_candidates(runtime) and not attempt_forced:
            return False
        no_score_streak = int(getattr(health_result, "no_score_streak", 0) or 0)
        same_family_no_score = int(getattr(runtime, "_same_family_no_score_streak", 0) or 0)
        state = getattr(runtime, "last_state", None)
        no_change_streak = int(getattr(state, "no_change_streak", 0) or 0)
        board_change_no_score = int(getattr(runtime, "_board_change_no_score_streak", 0) or 0)
        stable_repeat = int(getattr(runtime, "repeated_stable_target_count", 0) or 0)
        if attempt_forced:
            return bool(
                self._l1_attempt_failed_prefix_count >= 4
                or self._l1_attempt_relay_anchor_count >= 2
                or board_change_no_score >= 1
            )
        return bool(
            max(no_score_streak, same_family_no_score, no_change_streak) >= 6
            and (board_change_no_score >= 2 or stable_repeat >= 3)
        )

    def _base_pool_has_live_local_probe_candidates(self, runtime: Any) -> bool:
        base_pool = getattr(runtime, "_last_base_pool", None)
        for candidate in list(getattr(base_pool, "candidates", None) or []):
            source = str(getattr(candidate, "source", "") or "").strip().lower()
            if source in {
                "local_click_fanout",
                "local_reversible_click",
                "object_escape",
                "level_rule_symbol_scout",
            }:
                return True
        return False

    def _collect_toggle_combination_targets(self, runtime: Any) -> list[dict[str, Any]]:
        base_pool = getattr(runtime, "_last_base_pool", None)
        raw_candidates = list(getattr(base_pool, "candidates", None) or [])
        scored: list[tuple[float, dict[str, Any]]] = []
        seen: set[tuple[int, int]] = set()
        for candidate in raw_candidates:
            component_id = str(getattr(candidate, "component_id", "") or "").strip()
            family = str(getattr(candidate, "family", "") or "").strip().lower()
            if family and family not in {"click", "select", "protective_hold", "switch_visible_hedge", "queue_refresh"}:
                continue
            selected = self._structured_candidate_to_scored_candidate(candidate)
            position = self._resolve_target_grid_position(
                runtime=runtime,
                selected_target=selected,
                target_component_id=component_id,
                parameters={},
            )
            if position is None:
                continue
            row, col = position
            key = (int(row), int(col))
            if key in seen:
                continue
            seen.add(key)
            source = str(getattr(candidate, "source", "") or "").strip()
            source_bonus = {
                "object_table": 0.40,
                "successor_slate": 0.35,
                "protected_hedge": 0.25,
                "local_click_fanout": 0.15,
                "local_reversible_click": 0.15,
                "level_rule_symbol_scout": 0.10,
                "grid_scan": -0.15,
            }.get(source, 0.0)
            family_bonus = 0.25 if family in {"click", "select"} else 0.0
            score = float(getattr(candidate, "base_score", 0.0) or 0.0) + source_bonus + family_bonus
            scored.append(
                (
                    score,
                    {
                        "component_id": component_id,
                        "row": int(row),
                        "col": int(col),
                        "source": source,
                    },
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        return [target for _, target in scored]

    def _clear_agent_queue_for_hidden_state(
        self,
        *,
        agent: Any | None,
        semantic_action_lease: dict[str, Any],
    ) -> int:
        if agent is None:
            return 0
        clear_action_queue = getattr(agent, "clear_action_queue", None)
        if not callable(clear_action_queue):
            return 0
        queue_remaining: int | None = None
        get_plan_status = getattr(agent, "get_plan_status", None)
        if callable(get_plan_status):
            try:
                plan_status = get_plan_status()
            except Exception:
                plan_status = None
            if isinstance(plan_status, dict):
                try:
                    queue_remaining = int(plan_status.get("queue_remaining", 0) or 0)
                except (TypeError, ValueError):
                    queue_remaining = 0
        if queue_remaining is None:
            action_queue = getattr(agent, "_action_queue", None)
            try:
                queue_remaining = len(action_queue) if action_queue is not None else 0
            except TypeError:
                queue_remaining = 0
        if queue_remaining <= 0:
            return 0
        phase = str(semantic_action_lease.get("hidden_state_executor_phase", "") or "").strip()
        if not phase:
            phase = str(semantic_action_lease.get("semantic_phase", "") or "").strip() or "active"
        activation_id = semantic_action_lease.get("hidden_state_executor_activation_id", "")
        reason = f"hidden_state_executor_active:{phase}"
        try:
            activation = int(activation_id)
        except (TypeError, ValueError):
            activation = 0
        if activation > 0:
            reason = f"{reason}:activation={activation}"
        cleared = clear_action_queue(reason=reason)
        try:
            cleared_count = int(cleared or 0)
        except (TypeError, ValueError):
            cleared_count = 0
        if cleared_count > 0:
            self._hidden_state_queue_clear_count += 1
            self._hidden_state_cleared_action_count += cleared_count
            self._hidden_state_queue_clear_reason_histogram[reason] = (
                self._hidden_state_queue_clear_reason_histogram.get(reason, 0) + 1
            )
        return cleared_count

    def _build_semantic_action_lease(
        self,
        *,
        runtime: Any,
        agent: Any | None,
        step_index: int,
    ) -> dict[str, Any]:
        self._active_semantic_adapter = getattr(runtime, "semantic_adapter", None)
        mechanics_mode = str(getattr(runtime, "last_semantic_mechanics_mode", "") or "").strip().lower()
        semantic_phase = str(getattr(runtime, "last_semantic_phase", "") or "").strip().lower()
        semantic_progress = getattr(runtime, "last_semantic_progress", None) or {}
        if not isinstance(semantic_progress, dict):
            semantic_progress = {}
        recommendation = getattr(runtime, "last_semantic_recommendation", None) or {}
        if not isinstance(recommendation, dict):
            recommendation = {}
        suggested_actions = [
            str(action or "").strip().upper()
            for action in list(
                recommendation.get("lease_sequence")
                or recommendation.get("suggested_actions")
                or semantic_progress.get("lease_sequence")
                or []
            )
            if str(action or "").strip().upper() in _MOVE_ACTION_SET
        ]
        avoid_actions = [
            str(action or "").strip().upper()
            for action in list(
                recommendation.get("avoid_actions")
                or semantic_progress.get("avoid_actions")
                or []
            )
            if str(action or "").strip().upper() in _MOVE_ACTION_SET
        ]
        issued_step = int(recommendation.get("issued_step", -10_000) or -10_000)
        lease_age = int(step_index) - issued_step
        lease_horizon = int(
            recommendation.get("lease_horizon", semantic_progress.get("lease_horizon", 2)) or 2
        )
        recommendation_mode = str(recommendation.get("mode", "") or "").strip().lower()
        discovery_retest_mode = recommendation_mode in {
            "discovery_changer_probe",
            "agenda_lock_approach",
            "post_contact_gate_retest",
            "post_contact_retest_sticky",
        }
        active = (
            (mechanics_mode == "semantic" or discovery_retest_mode)
            and bool(suggested_actions)
            and issued_step > -10_000
            and 0 <= lease_age <= lease_horizon
        )
        lease = {
            "active": active,
            "mechanics_mode": mechanics_mode,
            "level_number": self._runtime_level_number(runtime),
            "recommendation_mode": recommendation_mode,
            "semantic_phase": semantic_phase or str(recommendation.get("semantic_phase", "") or "").strip().lower(),
            "site_mode": str(
                recommendation.get("site_mode", semantic_progress.get("site_mode", ""))
                or ""
            ).strip().lower(),
            "site_mode_age": int(
                recommendation.get("site_mode_age", semantic_progress.get("site_mode_age", 0)) or 0
            ),
            "event_kind": str(recommendation.get("event_kind", "") or "").strip(),
            "event_confidence": float(recommendation.get("event_confidence", 0.0) or 0.0),
            "priority": float(recommendation.get("priority", 0.0) or 0.0),
            "suggested_actions": suggested_actions,
            "route_source": str(recommendation.get("route_source", "") or "").strip(),
            "route_target_anchor": (
                [
                    int(value)
                    for value in list(recommendation.get("route_target_anchor", []) or [])
                    if str(value).strip("-").isdigit()
                ]
                if isinstance(recommendation.get("route_target_anchor"), (list, tuple))
                else []
            ),
            "route_preview": [
                str(action or "").strip().upper()
                for action in list(recommendation.get("route_preview", []) or [])
                if str(action or "").strip()
            ],
            "avoid_actions": avoid_actions,
            "lease_horizon": lease_horizon,
            "lease_age": lease_age if issued_step > -10_000 else -1,
            "site_reaching_score": float(
                recommendation.get("site_reaching_score", semantic_progress.get("site_reaching_score", 0.0))
                or 0.0
            ),
            "site_zone_confidence": float(
                recommendation.get("site_zone_confidence", semantic_progress.get("site_zone_confidence", 0.0))
                or 0.0
            ),
            "site_reaching_entered": bool(
                recommendation.get("site_reaching_entered", semantic_progress.get("site_reaching_entered", False))
            ),
            "site_reaching_exited": bool(
                recommendation.get("site_reaching_exited", semantic_progress.get("site_reaching_exited", False))
            ),
            "site_reaching_current_duration": int(
                recommendation.get(
                    "site_reaching_current_duration",
                    semantic_progress.get("site_reaching_current_duration", 0),
                )
                or 0
            ),
            "site_reaching_enter_count": int(
                semantic_progress.get("site_reaching_enter_count", 0) or 0
            ),
            "site_reaching_exit_count": int(
                semantic_progress.get("site_reaching_exit_count", 0) or 0
            ),
            "site_reaching_total_duration": int(
                semantic_progress.get("site_reaching_total_duration", 0) or 0
            ),
            "site_reaching_avg_duration": float(
                semantic_progress.get("site_reaching_avg_duration", 0.0) or 0.0
            ),
            "site_actuation_active": bool(
                recommendation.get("site_actuation_active", semantic_progress.get("site_actuation_active", False))
            ),
            "site_actuation_age": int(
                recommendation.get("site_actuation_age", semantic_progress.get("site_actuation_age", 0)) or 0
            ),
            "site_actuation_entered": bool(
                recommendation.get("site_actuation_entered", semantic_progress.get("site_actuation_entered", False))
            ),
            "site_actuation_exited": bool(
                recommendation.get("site_actuation_exited", semantic_progress.get("site_actuation_exited", False))
            ),
            "site_actuation_enter_count": int(
                semantic_progress.get("site_actuation_enter_count", 0) or 0
            ),
            "site_actuation_preferred_sequence_name": str(
                recommendation.get(
                    "site_actuation_preferred_sequence_name",
                    semantic_progress.get("site_actuation_preferred_sequence_name", ""),
                )
                or ""
            ).strip().lower(),
            "site_actuation_sequence_name": str(
                recommendation.get(
                    "site_actuation_sequence_name",
                    semantic_progress.get("site_actuation_sequence_name", ""),
                )
                or ""
            ).strip().lower(),
            "site_actuation_sequence_index": int(
                recommendation.get(
                    "site_actuation_sequence_index",
                    semantic_progress.get("site_actuation_sequence_index", 0),
                )
                or 0
            ),
            "site_actuation_sequence_remaining": [
                str(action or "").strip().upper()
                for action in list(
                    recommendation.get(
                        "site_actuation_sequence_remaining",
                        semantic_progress.get("site_actuation_sequence_remaining", []),
                    )
                    or []
                )
                if str(action or "").strip().upper() in _MOVE_ACTION_SET
            ],
            "site_distance": recommendation.get("site_distance", semantic_progress.get("site_distance")),
            "site_distance_delta": recommendation.get("site_distance_delta", semantic_progress.get("site_distance_delta")),
            "site_progress_stagnant_turns": int(
                recommendation.get(
                    "site_progress_stagnant_turns",
                    semantic_progress.get("site_progress_stagnant_turns", 0),
                )
                or 0
            ),
            "branch_entry_success": bool(semantic_progress.get("branch_entry_success", False)),
            "branch_internal_progress": bool(semantic_progress.get("branch_internal_progress", False)),
            "branch_internal_cycle": bool(semantic_progress.get("branch_internal_cycle", False)),
            "branch_exit_regression": bool(semantic_progress.get("branch_exit_regression", False)),
            "site_reach_attempt": bool(semantic_progress.get("site_reach_attempt", False)),
            "site_reach_success": bool(semantic_progress.get("site_reach_success", False)),
            "site_actuation_sequence_issue": bool(semantic_progress.get("site_actuation_sequence_issue", False)),
            "site_actuation_sequence_issue_count": int(
                semantic_progress.get("site_actuation_sequence_issue_count", 0) or 0
            ),
            "site_actuation_sequence_consume": bool(semantic_progress.get("site_actuation_sequence_consume", False)),
            "site_actuation_sequence_consume_count": int(
                semantic_progress.get("site_actuation_sequence_consume_count", 0) or 0
            ),
            "site_actuation_contact_success": bool(semantic_progress.get("site_actuation_contact_success", False)),
            "site_actuation_contact_success_count": int(
                semantic_progress.get("site_actuation_contact_success_count", 0) or 0
            ),
            "site_actuation_display_change": bool(semantic_progress.get("site_actuation_display_change", False)),
            "site_actuation_display_change_count": int(
                semantic_progress.get("site_actuation_display_change_count", 0) or 0
            ),
            "site_actuation_no_display_change_count": int(
                semantic_progress.get("site_actuation_no_display_change_count", 0) or 0
            ),
            "site_actuation_no_contact": bool(semantic_progress.get("site_actuation_no_contact", False)),
            "site_actuation_no_contact_count": int(
                semantic_progress.get("site_actuation_no_contact_count", 0) or 0
            ),
            "site_actuation_state_change_success": bool(
                semantic_progress.get("site_actuation_state_change_success", False)
            ),
            "site_actuation_state_change_success_count": int(
                semantic_progress.get("site_actuation_state_change_success_count", 0) or 0
            ),
            "site_actuation_verify_success": bool(semantic_progress.get("site_actuation_verify_success", False)),
            "site_actuation_verify_success_count": int(
                semantic_progress.get("site_actuation_verify_success_count", 0) or 0
            ),
            "site_exit_before_actuation": bool(semantic_progress.get("site_exit_before_actuation", False)),
            "site_exit_before_actuation_count": int(
                semantic_progress.get("site_exit_before_actuation_count", 0) or 0
            ),
            "site_exit_before_verify": bool(semantic_progress.get("site_exit_before_verify", False)),
            "site_exit_before_verify_count": int(
                semantic_progress.get("site_exit_before_verify_count", 0) or 0
            ),
            "site_actuation_reused_sequence_count": int(
                semantic_progress.get("site_actuation_reused_sequence_count", 0) or 0
            ),
            "site_actuation_attempt": bool(semantic_progress.get("site_actuation_attempt", False)),
            "site_actuation_success": bool(semantic_progress.get("site_actuation_success", False)),
            "site_actuation_attempt_count": int(semantic_progress.get("site_actuation_attempt_count", 0) or 0),
            "site_actuation_success_count": int(semantic_progress.get("site_actuation_success_count", 0) or 0),
            "latent_transition_inferred": bool(semantic_progress.get("latent_transition_inferred", False)),
            "latent_transition_inferred_count": int(semantic_progress.get("latent_transition_inferred_count", 0) or 0),
            "site_actuation_sequence_histogram": dict(
                semantic_progress.get("site_actuation_sequence_histogram", {}) or {}
            ),
            "site_actuation_display_change_sequence_histogram": dict(
                semantic_progress.get("site_actuation_display_change_sequence_histogram", {}) or {}
            ),
            "site_actuation_no_display_change_sequence_histogram": dict(
                semantic_progress.get("site_actuation_no_display_change_sequence_histogram", {}) or {}
            ),
            "site_actuation_no_contact_sequence_histogram": dict(
                semantic_progress.get("site_actuation_no_contact_sequence_histogram", {}) or {}
            ),
            "site_actuation_contact_position_histogram": dict(
                semantic_progress.get("site_actuation_contact_position_histogram", {}) or {}
            ),
            "site_actuation_contact_pair_histogram": dict(
                semantic_progress.get("site_actuation_contact_pair_histogram", {}) or {}
            ),
            "site_actuation_contact_examples": [
                dict(example)
                for example in list(semantic_progress.get("site_actuation_contact_examples", []) or [])
            ],
            "site_actuation_followup_window_count": int(
                semantic_progress.get("site_actuation_followup_window_count", 0) or 0
            ),
            "site_actuation_followup_stamp_change_count": int(
                semantic_progress.get("site_actuation_followup_stamp_change_count", 0) or 0
            ),
            "site_actuation_followup_no_change_count": int(
                semantic_progress.get("site_actuation_followup_no_change_count", 0) or 0
            ),
            "site_actuation_followup_change_pair_histogram": dict(
                semantic_progress.get("site_actuation_followup_change_pair_histogram", {}) or {}
            ),
            "site_actuation_followup_no_change_pair_histogram": dict(
                semantic_progress.get("site_actuation_followup_no_change_pair_histogram", {}) or {}
            ),
            "site_actuation_followup_examples": [
                dict(example)
                for example in list(semantic_progress.get("site_actuation_followup_examples", []) or [])
            ],
            "site_actuation_far_tail_turn_count": int(
                semantic_progress.get("site_actuation_far_tail_turn_count", 0) or 0
            ),
            "site_actuation_far_tail_sequence_histogram": dict(
                semantic_progress.get("site_actuation_far_tail_sequence_histogram", {}) or {}
            ),
            "site_actuation_far_tail_site_histogram": dict(
                semantic_progress.get("site_actuation_far_tail_site_histogram", {}) or {}
            ),
            "site_actuation_reacquire_candidate": bool(
                semantic_progress.get("site_actuation_reacquire_candidate", False)
            ),
            "site_actuation_reacquire_candidate_count": int(
                semantic_progress.get("site_actuation_reacquire_candidate_count", 0) or 0
            ),
            "site_actuation_reacquire": bool(semantic_progress.get("site_actuation_reacquire", False)),
            "site_actuation_reacquire_count": int(
                semantic_progress.get("site_actuation_reacquire_count", 0) or 0
            ),
            "site_actuation_controller_reset_count": int(
                semantic_progress.get("site_actuation_controller_reset_count", 0) or 0
            ),
            "site_actuation_controller_blocked_sequence_histogram": dict(
                semantic_progress.get("site_actuation_controller_blocked_sequence_histogram", {}) or {}
            ),
            "site_actuation_controller_blocked_sequence_count": int(
                semantic_progress.get("site_actuation_controller_blocked_sequence_count", 0) or 0
            ),
            "site_closer_outcome_record_count": int(
                semantic_progress.get("site_closer_outcome_record_count", 0) or 0
            ),
            "site_closer_sequence_blocked_count": int(
                semantic_progress.get("site_closer_sequence_blocked_count", 0) or 0
            ),
            "site_closer_sequence_rotated_count": int(
                semantic_progress.get("site_closer_sequence_rotated_count", 0) or 0
            ),
            "site_closer_same_failed_suppressed_count": int(
                semantic_progress.get("site_closer_same_failed_suppressed_count", 0) or 0
            ),
            "site_closer_promising_selected_count": int(
                semantic_progress.get("site_closer_promising_selected_count", 0) or 0
            ),
            "site_closer_closed_count": int(semantic_progress.get("site_closer_closed_count", 0) or 0),
            "site_closer_failed_tail_count": int(
                semantic_progress.get("site_closer_failed_tail_count", 0) or 0
            ),
            "site_closer_active_block_count": int(
                semantic_progress.get("site_closer_active_block_count", 0) or 0
            ),
            "site_closer_outcome_histogram": dict(
                semantic_progress.get("site_closer_outcome_histogram", {}) or {}
            ),
            "site_closer_sequence_outcome_histogram": dict(
                semantic_progress.get("site_closer_sequence_outcome_histogram", {}) or {}
            ),
            "site_closer_blocked_sequence_histogram": dict(
                semantic_progress.get("site_closer_blocked_sequence_histogram", {}) or {}
            ),
            "branch_to_site_distance_min": semantic_progress.get("branch_to_site_distance_min"),
            "do_not_repeat_actions": [],
        }
        recommendation_metadata = recommendation.get("metadata", {})
        if not isinstance(recommendation_metadata, dict):
            recommendation_metadata = {}
        resource_cycle_verify_active = bool(
            semantic_progress.get("resource_cycle_verify_active", False)
            or recommendation_metadata.get("resource_cycle_verify_active", False)
        )
        progress_goal_route_phase = (
            str(lease.get("semantic_phase", "") or "").strip().lower()
            in {"state_change_verification", "goal_route_execution"}
            or semantic_phase in {"state_change_verification", "goal_route_execution"}
        )
        progress_goal_route = [
            str(action or "").strip().upper()
            for action in list(semantic_progress.get("goal_route_preview", []) or [])
            if str(action or "").strip().upper() in _MOVE_ACTION_SET
        ]
        progress_goal_target = (
            [
                int(value)
                for value in list(semantic_progress.get("goal_route_target_anchor", []) or [])[:2]
                if str(value).strip("-").isdigit()
            ]
            if isinstance(semantic_progress.get("goal_route_target_anchor"), (list, tuple))
            else []
        )
        progress_goal_source = str(semantic_progress.get("goal_route_source", "") or "").strip()
        if progress_goal_route or progress_goal_target:
            lease["semantic_progress_goal_route_sync"] = True
            lease["semantic_progress_goal_route_source"] = progress_goal_source
            lease["semantic_progress_goal_route_target_anchor"] = list(progress_goal_target)
            lease["semantic_progress_goal_route_preview"] = list(progress_goal_route)
        if (
            progress_goal_route_phase
            and not resource_cycle_verify_active
            and (progress_goal_route or progress_goal_target)
        ):
            if progress_goal_route:
                lease["suggested_actions"] = list(progress_goal_route)
                lease["route_preview"] = list(progress_goal_route)
            if progress_goal_target:
                lease["route_target_anchor"] = list(progress_goal_target)
            if progress_goal_source:
                lease["route_source"] = progress_goal_source
        self._update_site_actuation_contract_result(
            step_index=step_index,
            semantic_phase=semantic_phase,
            semantic_progress=semantic_progress,
        )
        resource_route_handoff = bool(
            mechanics_mode == "semantic"
            and semantic_phase in {"branch_entry", "site_reaching"}
            and (
                bool(semantic_progress.get("resource_route_active", False))
                or bool(recommendation_metadata.get("resource_route_active", False))
                or bool(semantic_progress.get("alternate_site_route_active", False))
                or bool(recommendation_metadata.get("alternate_site_route_active", False))
            )
        )
        if resource_route_handoff and (
            self._site_actuation_contract_active
            or self._site_actuation_contract_armed
            or self._site_actuation_contract_result_pending
            or self._exact_site_tx_active
            or self._exact_site_lock_active
            or bool(self._site_actuation_contract_pending_verify_name)
            or bool(self._site_actuation_contract_pending_verify_steps)
        ):
            if int(lease.get("level_number", 0) or 0) >= 5 and self._apply_site_actuation_contract_lease(
                lease,
                event_kind="site_actuation_contract_resource_route_deferred",
                priority=1.4,
            ):
                lease["resource_route_handoff_deferred"] = True
            else:
                self._abort_site_actuation_contract("resource_route_handoff")
                lease["site_actuation_active"] = False
                lease["site_actuation_sequence_name"] = ""
                lease["site_actuation_sequence_index"] = 0
                lease["site_actuation_sequence_remaining"] = []
        state_change_handoff = bool(
            mechanics_mode == "semantic"
            and (
                str(lease.get("semantic_phase", "") or "").strip().lower()
                in {"state_change_verification", "goal_route_execution"}
                or bool(lease.get("site_actuation_state_change_success", False))
                or bool(lease.get("site_actuation_verify_success", False))
                or bool(lease.get("latent_transition_inferred", False))
            )
        )
        if state_change_handoff and (
            self._site_actuation_contract_active
            or self._site_actuation_contract_armed
            or self._site_actuation_contract_result_pending
            or self._exact_site_tx_active
            or self._exact_site_lock_active
            or bool(self._site_actuation_contract_pending_verify_name)
            or bool(self._site_actuation_contract_pending_verify_steps)
        ):
            self._handoff_site_actuation_contract_for_state_change()
            lease["site_actuation_active"] = False
            lease["site_actuation_sequence_name"] = ""
            lease["site_actuation_sequence_index"] = 0
            lease["site_actuation_sequence_remaining"] = []
        contract_remaining = self._site_actuation_contract_remaining()
        adapter_sequence_remaining = list(lease.get("site_actuation_sequence_remaining", []) or [])
        pending_verify_remaining = [
            str(action_name or "").strip().upper()
            for action_name in list(self._site_actuation_contract_pending_verify_steps)
            if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
        ]
        tx_active = bool(self._exact_site_tx_active)
        tx_phase = self._exact_site_tx_semantic_phase() if tx_active else ""
        tx_sequence_name = str(
            self._site_actuation_contract_sequence_name
            or self._site_actuation_contract_pending_verify_name
            or self._exact_site_lock_sequence_name
            or self._site_actuation_contract_last_completed_sequence_name
            or ""
        ).strip().lower()
        branch_to_site_distance_min = lease.get("branch_to_site_distance_min")
        try:
            branch_to_site_distance_min = (
                None if branch_to_site_distance_min is None else int(branch_to_site_distance_min)
            )
        except (TypeError, ValueError):
            branch_to_site_distance_min = None
        if (
            pending_verify_remaining
            and branch_to_site_distance_min == 0
            and not self._exact_site_lock_active
        ):
            self._issue_exact_site_lock(
                step_index=step_index,
                sequence_name=self._site_actuation_contract_pending_verify_name,
                ttl=min(3, max(1, len(pending_verify_remaining))),
            )
        exact_lock_remaining = pending_verify_remaining if self._exact_site_lock_active else []
        adapter_sequence_name = str(lease.get("site_actuation_sequence_name", "") or "").strip().lower()
        if (
            tx_active
            and adapter_sequence_remaining
            and adapter_sequence_name
            and adapter_sequence_name != tx_sequence_name
        ):
            self._semantic_reseed_while_tx_active_count += 1
        if contract_remaining:
            lease["active"] = True
            lease["mechanics_mode"] = lease["mechanics_mode"] or "semantic"
            lease["semantic_phase"] = tx_phase or (
                "site_actuation_verify" if self._exact_site_lock_active else "site_actuation"
            )
            lease["site_actuation_active"] = True
            lease["site_actuation_sequence_name"] = self._site_actuation_contract_sequence_name
            lease["site_actuation_sequence_index"] = int(self._site_actuation_contract_cursor)
            lease["site_actuation_sequence_remaining"] = list(contract_remaining)
            lease["suggested_actions"] = list(contract_remaining)
            lease["lease_horizon"] = max(1, min(3, len(contract_remaining)))
            lease["priority"] = max(float(lease.get("priority", 0.0) or 0.0), 1.0)
            lease["event_kind"] = (
                "exact_site_lock" if self._exact_site_lock_active
                else str(lease.get("event_kind", "") or "").strip() or "site_actuation_contract"
            )
        elif mechanics_mode == "semantic" and exact_lock_remaining:
            lease["active"] = True
            lease["mechanics_mode"] = "semantic"
            lease["semantic_phase"] = tx_phase or "site_actuation_verify"
            lease["site_actuation_active"] = True
            lease["site_actuation_sequence_name"] = self._exact_site_lock_sequence_name or tx_sequence_name
            lease["site_actuation_sequence_index"] = int(self._exact_site_tx_cursor if tx_active else 0)
            lease["site_actuation_sequence_remaining"] = list(exact_lock_remaining)
            lease["suggested_actions"] = list(exact_lock_remaining)
            lease["lease_horizon"] = max(1, min(3, len(exact_lock_remaining)))
            lease["priority"] = max(float(lease.get("priority", 0.0) or 0.0), 1.2)
            lease["event_kind"] = "exact_site_transaction" if tx_active else "exact_site_lock"
        elif mechanics_mode == "semantic" and pending_verify_remaining:
            lease["active"] = True
            lease["mechanics_mode"] = "semantic"
            lease["semantic_phase"] = tx_phase or "site_actuation_verify"
            lease["site_actuation_active"] = True
            lease["site_actuation_sequence_name"] = self._site_actuation_contract_pending_verify_name
            lease["site_actuation_sequence_index"] = 0
            lease["site_actuation_sequence_remaining"] = list(pending_verify_remaining)
            lease["suggested_actions"] = list(pending_verify_remaining)
            lease["lease_horizon"] = max(1, min(3, len(pending_verify_remaining)))
            lease["priority"] = max(float(lease.get("priority", 0.0) or 0.0), 1.1)
            lease["event_kind"] = "exact_site_transaction_pending_verify" if tx_active else "site_actuation_verify"
        elif tx_active:
            lease["active"] = True
            lease["mechanics_mode"] = "semantic"
            lease["semantic_phase"] = tx_phase or "site_actuation_followup"
            lease["site_actuation_active"] = True
            lease["site_actuation_sequence_name"] = tx_sequence_name
            lease["site_actuation_sequence_index"] = int(self._exact_site_tx_cursor)
            lease["site_actuation_sequence_remaining"] = []
            lease["suggested_actions"] = []
            lease["lease_horizon"] = 1
            lease["priority"] = max(float(lease.get("priority", 0.0) or 0.0), 1.0)
            lease["event_kind"] = "exact_site_transaction"
        elif (
            mechanics_mode == "semantic"
            and adapter_sequence_remaining
            and not tx_active
            and (
                bool(lease.get("site_actuation_active", False))
                or str(lease.get("semantic_phase", "") or "").strip().lower().startswith("site_actuation")
                or bool(str(lease.get("site_actuation_sequence_name", "") or "").strip())
            )
        ):
            lease["active"] = True
            lease["suggested_actions"] = list(adapter_sequence_remaining)
            lease["lease_horizon"] = max(1, min(3, len(adapter_sequence_remaining)))
            lease["priority"] = max(float(lease.get("priority", 0.0) or 0.0), 0.9)
        lease["site_actuation_contract_active"] = bool(contract_remaining)
        lease["site_actuation_contract_sequence_name"] = str(self._site_actuation_contract_sequence_name or "")
        lease["site_actuation_contract_cursor"] = int(self._site_actuation_contract_cursor)
        lease["site_actuation_contract_remaining"] = list(contract_remaining)
        lease["site_actuation_contract_abort_reason"] = str(self._site_actuation_contract_abort_reason or "")
        lease["site_actuation_pending_verify_name"] = str(self._site_actuation_contract_pending_verify_name or "")
        lease["exact_site_tx_active"] = bool(self._exact_site_tx_active)
        lease["exact_site_tx_id"] = int(self._exact_site_tx_id)
        lease["exact_site_tx_phase"] = str(self._exact_site_tx_phase or "")
        lease["exact_site_tx_cursor"] = int(self._exact_site_tx_cursor)
        lease["exact_site_tx_expected_actions"] = list(self._exact_site_tx_expected_actions)
        lease["exact_site_tx_executed_actions"] = list(self._exact_site_tx_executed_actions)
        lease["exact_site_tx_ttl_remaining"] = int(self._exact_site_tx_ttl_remaining)
        lease["exact_site_tx_retry_budget"] = int(self._exact_site_tx_retry_budget)
        lease["exact_site_tx_abort_reason"] = str(self._exact_site_tx_abort_reason or "")
        lease["exact_site_lock_active"] = bool(self._exact_site_lock_active)
        lease["exact_site_lock_sequence_name"] = str(self._exact_site_lock_sequence_name or "")
        lease["exact_site_lock_ttl_remaining"] = int(self._exact_site_lock_ttl_remaining)
        if (
            self._exact_site_lock_active
            and not str(lease.get("semantic_phase", "") or "").strip().lower().startswith("site_actuation")
        ):
            self._semantic_phase_downgrade_while_exact_site_lock_count += 1

        current_sequence_name = str(lease.get("site_actuation_sequence_name", "") or "").strip().lower()
        current_sequence_actions = [
            str(action_name or "").strip().upper()
            for action_name in list(
                lease.get("site_actuation_sequence_remaining")
                or lease.get("suggested_actions")
                or []
            )
            if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
        ]
        if (
            mechanics_mode == "semantic"
            and (current_sequence_name or current_sequence_actions)
            and not (
                str(lease.get("route_source", "") or "").strip()
                and list(lease.get("route_preview", []) or [])
            )
        ):
            sequence_spec = self._resolve_adapter_sequence_spec(
                name=current_sequence_name,
                actions=current_sequence_actions,
                context={
                    "reason": "semantic_site_actuation_route_metadata",
                    "semantic_phase": str(lease.get("semantic_phase", "") or "").strip().lower(),
                },
            )
            self._merge_sequence_route_metadata(
                lease,
                sequence_spec=sequence_spec,
                preview_actions=current_sequence_actions,
            )
        if active:
            lease["do_not_repeat_actions"] = self._semantic_do_not_repeat_actions(
                agent=agent,
                suggested_actions=list(lease.get("suggested_actions", []) or []),
            )
            for action_name in avoid_actions:
                if action_name and action_name not in lease["do_not_repeat_actions"]:
                    lease["do_not_repeat_actions"].append(action_name)
        elif bool(lease.get("active", False)):
            lease["do_not_repeat_actions"] = self._semantic_do_not_repeat_actions(
                agent=agent,
                suggested_actions=list(lease.get("suggested_actions", []) or []),
            )
            for action_name in avoid_actions:
                if action_name and action_name not in lease["do_not_repeat_actions"]:
                    lease["do_not_repeat_actions"].append(action_name)
        return lease

    def _recent_move_rows(self, agent: Any | None, *, limit: int = 8) -> list[dict[str, Any]]:
        if agent is None or not hasattr(agent, "get_recent_history"):
            return []
        try:
            recent_history = list(agent.get_recent_history(limit=limit) or [])
        except Exception:
            return []
        return [
            row for row in recent_history
            if str(row.get("action", "") or "").strip().upper() in _MOVE_ACTION_SET
        ]

    @staticmethod
    def _parse_action6_grid_coords(action_text: Any) -> tuple[int, int] | None:
        match = re.search(r"ACTION6\(x=(\d+),\s*y=(\d+)", str(action_text or ""))
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def _recent_action6_rows(self, agent: Any | None, *, limit: int = 10) -> list[dict[str, Any]]:
        if agent is None or not hasattr(agent, "get_recent_history"):
            return []
        try:
            recent_history = list(agent.get_recent_history(limit=limit) or [])
        except Exception:
            return []
        rows: list[dict[str, Any]] = []
        for row in recent_history:
            action_name = str(row.get("action", "") or "").strip().upper()
            if not action_name.startswith("ACTION6"):
                continue
            rows.append(row)
        return rows

    def _recent_action6_outcome(
        self,
        agent: Any | None,
        *,
        row: int,
        col: int,
        limit: int = 10,
    ) -> dict[str, int]:
        outcome = {"seen": 0, "weak": 0, "score": 0}
        for item in self._recent_action6_rows(agent, limit=limit):
            coords = self._parse_action6_grid_coords(item.get("action"))
            if coords is None:
                raw_row = item.get("click_prompt_row", item.get("x"))
                raw_col = item.get("click_prompt_col", item.get("y"))
                try:
                    coords = (int(raw_row), int(raw_col))
                except (TypeError, ValueError):
                    coords = None
            if coords is None:
                continue
            if abs(int(coords[0]) - int(row)) > 1 or abs(int(coords[1]) - int(col)) > 1:
                continue
            outcome["seen"] += 1
            score_delta = float(item.get("score_delta", 0.0) or 0.0)
            if abs(score_delta) > 1e-9:
                outcome["score"] += 1
            elif bool(item.get("no_state_change")) or bool(item.get("contradiction")):
                outcome["weak"] += 1
        return outcome

    def _semantic_do_not_repeat_actions(
        self,
        *,
        agent: Any | None,
        suggested_actions: list[str],
    ) -> list[str]:
        recent_move_rows = self._recent_move_rows(agent, limit=8)
        if not recent_move_rows:
            return []

        penalties: list[str] = []
        latest = recent_move_rows[-1]
        latest_action = str(latest.get("action", "") or "").strip().upper()
        latest_failed = bool(latest.get("no_state_change")) or bool(latest.get("contradiction"))
        if latest_action in _MOVE_ACTION_SET and latest_failed:
            penalties.append(latest_action)

        if len(recent_move_rows) >= 2:
            previous = recent_move_rows[-2]
            previous_action = str(previous.get("action", "") or "").strip().upper()
            previous_failed = bool(previous.get("no_state_change")) or bool(previous.get("contradiction"))
            if latest_action == previous_action and latest_failed and previous_failed:
                penalties.append(latest_action)

        failed_counts: dict[str, int] = {}
        for row in recent_move_rows:
            action_name = str(row.get("action", "") or "").strip().upper()
            if action_name not in _MOVE_ACTION_SET:
                continue
            if bool(row.get("no_state_change")) or bool(row.get("contradiction")):
                failed_counts[action_name] = failed_counts.get(action_name, 0) + 1
        for action_name, count in failed_counts.items():
            if count >= 2 and action_name not in suggested_actions:
                penalties.append(action_name)

        ordered: list[str] = []
        for action_name in penalties:
            if action_name and action_name not in ordered:
                ordered.append(action_name)
        return ordered

    def _site_actuation_controller_blocked_sequence_histogram(self) -> dict[str, int]:
        histogram = dict(self._semantic_site_actuation_controller_blocked_sequence_histogram)
        for sequence_name in sorted(self._semantic_site_actuation_controller_reacquire_blocked_sequences):
            normalized_name = str(sequence_name or "").strip().lower()
            if normalized_name:
                histogram[normalized_name] = max(int(histogram.get(normalized_name, 0) or 0), 1)
        return histogram

    def _begin_semantic_bridge_step(self, semantic_action_lease: dict[str, Any] | None) -> None:
        semantic_action_lease = semantic_action_lease or {}
        self._semantic_bridge_choice_recorded_this_step = False
        self._last_semantic_recommendation_active = bool(semantic_action_lease.get("active", False))
        self._last_semantic_mechanics_mode = str(semantic_action_lease.get("mechanics_mode", "") or "")
        self._last_semantic_phase = str(semantic_action_lease.get("semantic_phase", "") or "")
        self._last_semantic_site_mode = str(semantic_action_lease.get("site_mode", "") or "")
        self._last_semantic_event_kind = str(semantic_action_lease.get("event_kind", "") or "")
        self._last_semantic_event_confidence = float(semantic_action_lease.get("event_confidence", 0.0) or 0.0)
        _lease_age = semantic_action_lease.get("lease_age", -1)
        self._last_semantic_lease_age = int(_lease_age) if _lease_age is not None else -1
        self._last_semantic_suggested_actions = list(semantic_action_lease.get("suggested_actions") or [])
        self._current_semantic_route_source = str(semantic_action_lease.get("route_source", "") or "")
        self._current_semantic_route_target_anchor = [
            int(value)
            for value in list(semantic_action_lease.get("route_target_anchor", []) or [])
            if str(value).strip("-").isdigit()
        ]
        self._current_semantic_route_preview = [
            str(action or "").strip().upper()
            for action in list(semantic_action_lease.get("route_preview", []) or [])
            if str(action or "").strip()
        ]
        self._last_semantic_progress_goal_route_target_anchor = [
            int(value)
            for value in list(
                semantic_action_lease.get("semantic_progress_goal_route_target_anchor", []) or []
            )[:2]
            if str(value).strip("-").isdigit()
        ]
        self._last_semantic_progress_goal_route_preview = [
            str(action or "").strip().upper()
            for action in list(semantic_action_lease.get("semantic_progress_goal_route_preview", []) or [])
            if str(action or "").strip()
        ]
        self._last_hidden_state_route_resync_from_progress = bool(
            semantic_action_lease.get("hidden_state_executor_route_resync_from_progress", False)
        )
        if self._current_semantic_route_source or self._current_semantic_route_preview:
            self._last_semantic_route_source = self._current_semantic_route_source
            self._last_semantic_route_target_anchor = list(self._current_semantic_route_target_anchor)
            self._last_semantic_route_preview = list(self._current_semantic_route_preview)
            self._semantic_route_recommendation_count += 1
            if self._current_semantic_route_source:
                self._semantic_route_source_histogram[self._current_semantic_route_source] = (
                    self._semantic_route_source_histogram.get(self._current_semantic_route_source, 0) + 1
                )
        self._last_semantic_do_not_repeat_actions = list(semantic_action_lease.get("do_not_repeat_actions") or [])
        self._last_semantic_site_reaching_score = float(
            semantic_action_lease.get("site_reaching_score", 0.0) or 0.0
        )
        self._last_semantic_site_zone_confidence = float(
            semantic_action_lease.get("site_zone_confidence", 0.0) or 0.0
        )
        self._last_semantic_site_mode_age = int(semantic_action_lease.get("site_mode_age", 0) or 0)
        self._last_semantic_site_reaching_current_duration = int(
            semantic_action_lease.get("site_reaching_current_duration", 0) or 0
        )
        self._last_semantic_site_actuation_active = bool(
            semantic_action_lease.get("site_actuation_active", False)
        )
        self._last_semantic_site_actuation_age = int(
            semantic_action_lease.get("site_actuation_age", 0) or 0
        )
        self._last_semantic_site_actuation_sequence_name = str(
            semantic_action_lease.get("site_actuation_sequence_name", "") or ""
        )
        self._last_semantic_site_actuation_sequence_remaining = list(
            semantic_action_lease.get("site_actuation_sequence_remaining") or []
        )
        self._last_semantic_site_actuation_preferred_sequence_name = str(
            semantic_action_lease.get("site_actuation_preferred_sequence_name", "") or ""
        )
        self._last_semantic_site_actuation_contact_examples = [
            dict(example)
            for example in list(semantic_action_lease.get("site_actuation_contact_examples", []) or [])
        ]
        self._last_semantic_site_actuation_followup_examples = [
            dict(example)
            for example in list(semantic_action_lease.get("site_actuation_followup_examples", []) or [])
        ]
        self._last_semantic_site_distance = (
            int(semantic_action_lease.get("site_distance"))
            if semantic_action_lease.get("site_distance") is not None else None
        )
        self._last_semantic_site_distance_delta = (
            int(semantic_action_lease.get("site_distance_delta"))
            if semantic_action_lease.get("site_distance_delta") is not None else None
        )
        self._last_semantic_stagnant_turns = int(semantic_action_lease.get("site_progress_stagnant_turns", 0) or 0)
        controller_tail_remaining = [
            str(action or "").strip().upper()
            for action in list(self._last_semantic_site_actuation_sequence_remaining or [])
            if str(action or "").strip().upper() in _MOVE_ACTION_SET
        ]
        controller_far_tail = bool(
            str(self._last_semantic_phase or "").strip().lower().startswith("site_actuation")
            and self._last_semantic_site_distance is not None
            and int(self._last_semantic_site_distance) >= 15
            and int(self._last_semantic_site_actuation_age) >= 7
            and str(self._last_semantic_site_actuation_sequence_name or "").strip()
            and (
                not controller_tail_remaining
                or (
                    len(controller_tail_remaining) == 1
                    and controller_tail_remaining[0] == "ACTION1"
                )
            )
        )
        if controller_far_tail:
            self._semantic_site_actuation_controller_far_tail_turn_count += 1
            controller_tail_name = str(
                self._last_semantic_site_actuation_sequence_name or "unknown"
            ).strip().lower()
            self._semantic_site_actuation_controller_far_tail_sequence_histogram[controller_tail_name] = (
                self._semantic_site_actuation_controller_far_tail_sequence_histogram.get(controller_tail_name, 0) + 1
            )
        self._last_semantic_selected_action = ""
        self._last_semantic_recommended_action_selected = False
        self._last_semantic_action_source = ""
        if bool(semantic_action_lease.get("branch_entry_success", False)):
            self._semantic_branch_entry_success_count += 1
        if bool(semantic_action_lease.get("branch_internal_progress", False)):
            self._semantic_branch_internal_progress_turns += 1
        if semantic_action_lease.get("branch_to_site_distance_min") is not None:
            _distance_min = int(semantic_action_lease.get("branch_to_site_distance_min"))
            if self._semantic_branch_to_site_distance_min is None:
                self._semantic_branch_to_site_distance_min = _distance_min
            else:
                self._semantic_branch_to_site_distance_min = min(
                    self._semantic_branch_to_site_distance_min,
                    _distance_min,
                )
        if bool(semantic_action_lease.get("site_reach_attempt", False)):
            self._semantic_site_reach_attempt_count += 1
        if bool(semantic_action_lease.get("site_reach_success", False)):
            self._semantic_site_reach_success_count += 1
        if bool(semantic_action_lease.get("branch_exit_regression", False)):
            self._semantic_branch_exit_back_to_corridor_count += 1
        self._semantic_site_reaching_enter_count = max(
            self._semantic_site_reaching_enter_count,
            int(semantic_action_lease.get("site_reaching_enter_count", 0) or 0),
        )
        self._semantic_site_reaching_exit_count = max(
            self._semantic_site_reaching_exit_count,
            int(semantic_action_lease.get("site_reaching_exit_count", 0) or 0),
        )
        self._semantic_site_reaching_total_duration = max(
            self._semantic_site_reaching_total_duration,
            int(semantic_action_lease.get("site_reaching_total_duration", 0) or 0),
        )
        self._semantic_site_actuation_enter_count = max(
            self._semantic_site_actuation_enter_count,
            int(semantic_action_lease.get("site_actuation_enter_count", 0) or 0),
        )
        self._semantic_site_actuation_attempt_count = max(
            self._semantic_site_actuation_attempt_count,
            int(semantic_action_lease.get("site_actuation_attempt_count", 0) or 0),
        )
        self._semantic_site_actuation_success_count = max(
            self._semantic_site_actuation_success_count,
            int(semantic_action_lease.get("site_actuation_success_count", 0) or 0),
        )
        self._semantic_site_actuation_sequence_issue_count = max(
            self._semantic_site_actuation_sequence_issue_count,
            int(semantic_action_lease.get("site_actuation_sequence_issue_count", 0) or 0),
        )
        self._semantic_site_actuation_sequence_consume_count = max(
            self._semantic_site_actuation_sequence_consume_count,
            int(semantic_action_lease.get("site_actuation_sequence_consume_count", 0) or 0),
        )
        self._semantic_site_actuation_contact_success_count = max(
            self._semantic_site_actuation_contact_success_count,
            int(semantic_action_lease.get("site_actuation_contact_success_count", 0) or 0),
        )
        self._semantic_site_actuation_display_change_count = max(
            self._semantic_site_actuation_display_change_count,
            int(semantic_action_lease.get("site_actuation_display_change_count", 0) or 0),
        )
        self._semantic_site_actuation_no_contact_count = max(
            self._semantic_site_actuation_no_contact_count,
            int(semantic_action_lease.get("site_actuation_no_contact_count", 0) or 0),
        )
        self._semantic_site_actuation_state_change_success_count = max(
            self._semantic_site_actuation_state_change_success_count,
            int(semantic_action_lease.get("site_actuation_state_change_success_count", 0) or 0),
        )
        self._semantic_site_actuation_verify_success_count = max(
            self._semantic_site_actuation_verify_success_count,
            int(semantic_action_lease.get("site_actuation_verify_success_count", 0) or 0),
        )
        self._semantic_site_exit_before_actuation_count = max(
            self._semantic_site_exit_before_actuation_count,
            int(semantic_action_lease.get("site_exit_before_actuation_count", 0) or 0),
        )
        self._semantic_site_exit_before_verify_count = max(
            self._semantic_site_exit_before_verify_count,
            int(semantic_action_lease.get("site_exit_before_verify_count", 0) or 0),
        )
        self._semantic_site_actuation_reused_sequence_count = max(
            self._semantic_site_actuation_reused_sequence_count,
            int(semantic_action_lease.get("site_actuation_reused_sequence_count", 0) or 0),
        )
        for sequence_name, count in dict(
            semantic_action_lease.get("site_actuation_sequence_histogram", {}) or {}
        ).items():
            normalized_name = str(sequence_name or "").strip().lower()
            if not normalized_name:
                continue
            self._semantic_site_actuation_sequence_histogram[normalized_name] = max(
                self._semantic_site_actuation_sequence_histogram.get(normalized_name, 0),
                int(count or 0),
            )
        for sequence_name, count in dict(
            semantic_action_lease.get("site_actuation_display_change_sequence_histogram", {}) or {}
        ).items():
            normalized_name = str(sequence_name or "").strip().lower()
            if not normalized_name:
                continue
            self._semantic_site_actuation_display_change_sequence_histogram[normalized_name] = max(
                self._semantic_site_actuation_display_change_sequence_histogram.get(normalized_name, 0),
                int(count or 0),
            )
        for sequence_name, count in dict(
            semantic_action_lease.get("site_actuation_no_display_change_sequence_histogram", {}) or {}
        ).items():
            normalized_name = str(sequence_name or "").strip().lower()
            if not normalized_name:
                continue
            self._semantic_site_actuation_no_display_change_sequence_histogram[normalized_name] = max(
                self._semantic_site_actuation_no_display_change_sequence_histogram.get(normalized_name, 0),
                int(count or 0),
            )
        for sequence_name, count in dict(
            semantic_action_lease.get("site_actuation_no_contact_sequence_histogram", {}) or {}
        ).items():
            normalized_name = str(sequence_name or "").strip().lower()
            if not normalized_name:
                continue
            self._semantic_site_actuation_no_contact_sequence_histogram[normalized_name] = max(
                self._semantic_site_actuation_no_contact_sequence_histogram.get(normalized_name, 0),
                int(count or 0),
            )
        for contact_key, count in dict(
            semantic_action_lease.get("site_actuation_contact_position_histogram", {}) or {}
        ).items():
            normalized_key = str(contact_key or "").strip().lower()
            if not normalized_key:
                continue
            self._semantic_site_actuation_contact_position_histogram[normalized_key] = max(
                self._semantic_site_actuation_contact_position_histogram.get(normalized_key, 0),
                int(count or 0),
            )
        for contact_key, count in dict(
            semantic_action_lease.get("site_actuation_contact_pair_histogram", {}) or {}
        ).items():
            normalized_key = str(contact_key or "").strip().lower()
            if not normalized_key:
                continue
            self._semantic_site_actuation_contact_pair_histogram[normalized_key] = max(
                self._semantic_site_actuation_contact_pair_histogram.get(normalized_key, 0),
                int(count or 0),
            )
        self._semantic_site_actuation_followup_window_count = max(
            self._semantic_site_actuation_followup_window_count,
            int(semantic_action_lease.get("site_actuation_followup_window_count", 0) or 0),
        )
        self._semantic_site_actuation_followup_stamp_change_count = max(
            self._semantic_site_actuation_followup_stamp_change_count,
            int(semantic_action_lease.get("site_actuation_followup_stamp_change_count", 0) or 0),
        )
        self._semantic_site_actuation_followup_no_change_count = max(
            self._semantic_site_actuation_followup_no_change_count,
            int(semantic_action_lease.get("site_actuation_followup_no_change_count", 0) or 0),
        )
        for pair_key, count in dict(
            semantic_action_lease.get("site_actuation_followup_change_pair_histogram", {}) or {}
        ).items():
            normalized_key = str(pair_key or "").strip().lower()
            if not normalized_key:
                continue
            self._semantic_site_actuation_followup_change_pair_histogram[normalized_key] = max(
                self._semantic_site_actuation_followup_change_pair_histogram.get(normalized_key, 0),
                int(count or 0),
            )
        for pair_key, count in dict(
            semantic_action_lease.get("site_actuation_followup_no_change_pair_histogram", {}) or {}
        ).items():
            normalized_key = str(pair_key or "").strip().lower()
            if not normalized_key:
                continue
            self._semantic_site_actuation_followup_no_change_pair_histogram[normalized_key] = max(
                self._semantic_site_actuation_followup_no_change_pair_histogram.get(normalized_key, 0),
                int(count or 0),
            )
        self._semantic_site_actuation_far_tail_turn_count = max(
            self._semantic_site_actuation_far_tail_turn_count,
            int(semantic_action_lease.get("site_actuation_far_tail_turn_count", 0) or 0),
        )
        for sequence_name, count in dict(
            semantic_action_lease.get("site_actuation_far_tail_sequence_histogram", {}) or {}
        ).items():
            normalized_name = str(sequence_name or "").strip().lower()
            if not normalized_name:
                continue
            self._semantic_site_actuation_far_tail_sequence_histogram[normalized_name] = max(
                self._semantic_site_actuation_far_tail_sequence_histogram.get(normalized_name, 0),
                int(count or 0),
            )
        for site_key, count in dict(
            semantic_action_lease.get("site_actuation_far_tail_site_histogram", {}) or {}
        ).items():
            normalized_key = str(site_key or "").strip().lower()
            if not normalized_key:
                continue
            self._semantic_site_actuation_far_tail_site_histogram[normalized_key] = max(
                self._semantic_site_actuation_far_tail_site_histogram.get(normalized_key, 0),
                int(count or 0),
            )
        self._semantic_site_actuation_reacquire_candidate_count = max(
            self._semantic_site_actuation_reacquire_candidate_count,
            int(semantic_action_lease.get("site_actuation_reacquire_candidate_count", 0) or 0),
        )
        self._semantic_site_actuation_reacquire_count = max(
            self._semantic_site_actuation_reacquire_count,
            int(semantic_action_lease.get("site_actuation_reacquire_count", 0) or 0),
        )
        self._semantic_site_actuation_controller_reset_count = max(
            self._semantic_site_actuation_controller_reset_count,
            int(semantic_action_lease.get("site_actuation_controller_reset_count", 0) or 0),
        )
        self._semantic_site_actuation_controller_blocked_sequence_count = max(
            self._semantic_site_actuation_controller_blocked_sequence_count,
            int(semantic_action_lease.get("site_actuation_controller_blocked_sequence_count", 0) or 0),
        )
        for name, count in dict(
            semantic_action_lease.get("site_actuation_controller_blocked_sequence_histogram", {}) or {}
        ).items():
            normalized_name = str(name or "").strip().lower()
            if not normalized_name:
                continue
            self._semantic_site_actuation_controller_blocked_sequence_histogram[normalized_name] = max(
                self._semantic_site_actuation_controller_blocked_sequence_histogram.get(normalized_name, 0),
                int(count or 0),
            )
        self._semantic_site_closer_outcome_record_count = max(
            self._semantic_site_closer_outcome_record_count,
            int(semantic_action_lease.get("site_closer_outcome_record_count", 0) or 0),
        )
        self._semantic_site_closer_sequence_blocked_count = max(
            self._semantic_site_closer_sequence_blocked_count,
            int(semantic_action_lease.get("site_closer_sequence_blocked_count", 0) or 0),
        )
        self._semantic_site_closer_sequence_rotated_count = max(
            self._semantic_site_closer_sequence_rotated_count,
            int(semantic_action_lease.get("site_closer_sequence_rotated_count", 0) or 0),
        )
        self._semantic_site_closer_same_failed_suppressed_count = max(
            self._semantic_site_closer_same_failed_suppressed_count,
            int(semantic_action_lease.get("site_closer_same_failed_suppressed_count", 0) or 0),
        )
        self._semantic_site_closer_promising_selected_count = max(
            self._semantic_site_closer_promising_selected_count,
            int(semantic_action_lease.get("site_closer_promising_selected_count", 0) or 0),
        )
        self._semantic_site_closer_closed_count = max(
            self._semantic_site_closer_closed_count,
            int(semantic_action_lease.get("site_closer_closed_count", 0) or 0),
        )
        self._semantic_site_closer_failed_tail_count = max(
            self._semantic_site_closer_failed_tail_count,
            int(semantic_action_lease.get("site_closer_failed_tail_count", 0) or 0),
        )
        self._semantic_site_closer_active_block_count = max(
            self._semantic_site_closer_active_block_count,
            int(semantic_action_lease.get("site_closer_active_block_count", 0) or 0),
        )
        self._semantic_site_closer_alternate_pending_count = max(
            self._semantic_site_closer_alternate_pending_count,
            int(semantic_action_lease.get("site_closer_alternate_pending_count", 0) or 0),
        )
        self._semantic_site_closer_alternate_forced_count = max(
            self._semantic_site_closer_alternate_forced_count,
            int(semantic_action_lease.get("site_closer_alternate_forced_count", 0) or 0),
        )
        self._semantic_site_closer_alternate_missing_count = max(
            self._semantic_site_closer_alternate_missing_count,
            int(semantic_action_lease.get("site_closer_alternate_missing_count", 0) or 0),
        )
        self._semantic_site_closer_alternate_expired_count = max(
            self._semantic_site_closer_alternate_expired_count,
            int(semantic_action_lease.get("site_closer_alternate_expired_count", 0) or 0),
        )
        for key, count in dict(semantic_action_lease.get("site_closer_outcome_histogram", {}) or {}).items():
            normalized_key = str(key or "").strip().lower()
            if normalized_key:
                self._semantic_site_closer_outcome_histogram[normalized_key] = max(
                    self._semantic_site_closer_outcome_histogram.get(normalized_key, 0),
                    int(count or 0),
                )
        for key, count in dict(
            semantic_action_lease.get("site_closer_sequence_outcome_histogram", {}) or {}
        ).items():
            normalized_key = str(key or "").strip().lower()
            if normalized_key:
                self._semantic_site_closer_sequence_outcome_histogram[normalized_key] = max(
                    self._semantic_site_closer_sequence_outcome_histogram.get(normalized_key, 0),
                    int(count or 0),
                )
        for key, count in dict(
            semantic_action_lease.get("site_closer_blocked_sequence_histogram", {}) or {}
        ).items():
            normalized_key = str(key or "").strip().lower()
            if normalized_key:
                self._semantic_site_closer_blocked_sequence_histogram[normalized_key] = max(
                    self._semantic_site_closer_blocked_sequence_histogram.get(normalized_key, 0),
                    int(count or 0),
                )
        if self._last_semantic_recommendation_active:
            self._semantic_recommendation_turn_count += 1

    def _note_semantic_bridge_choice(
        self,
        *,
        semantic_action_lease: dict[str, Any] | None,
        action_name: str,
        source: str,
    ) -> None:
        if self._semantic_bridge_choice_recorded_this_step:
            return
        action_name = str(action_name or "").strip().upper()
        if action_name not in _MOVE_ACTION_SET:
            return

        self._semantic_bridge_choice_recorded_this_step = True
        self._last_semantic_selected_action = action_name
        self._last_semantic_action_source = str(source or "")

        semantic_action_lease = semantic_action_lease or {}
        if not bool(semantic_action_lease.get("active", False)):
            self._last_semantic_recommended_action_selected = False
            return

        suggested_actions = list(semantic_action_lease.get("suggested_actions") or [])
        matched = action_name in suggested_actions
        self._last_semantic_recommended_action_selected = matched
        if matched:
            self._semantic_recommended_action_match_count += 1
            self._semantic_recommended_action_executed_count += 1
        if (
            action_name in {"ACTION3", "ACTION4"}
            and str(semantic_action_lease.get("event_kind", "") or "") in _SEMANTIC_BRANCH_ENTRY_EVENTS
        ):
            self._semantic_branch_entry_attempt_count += 1

    def _semantic_preferred_move_action(
        self,
        *,
        agent: Any | None,
        fallback_action: str,
        semantic_action_lease: dict[str, Any] | None,
    ) -> str:
        fallback_action = str(fallback_action or "").strip().upper()
        semantic_action_lease = semantic_action_lease or {}
        if not bool(semantic_action_lease.get("active", False)):
            return fallback_action or "ACTION1"

        suggested_actions = list(semantic_action_lease.get("suggested_actions") or [])
        do_not_repeat_actions = set(semantic_action_lease.get("do_not_repeat_actions") or [])
        event_kind = str(semantic_action_lease.get("event_kind", "") or "").strip()
        semantic_phase = str(semantic_action_lease.get("semantic_phase", "") or "").strip().lower()
        site_mode = str(semantic_action_lease.get("site_mode", "") or "").strip().lower()
        site_actuation_active = bool(semantic_action_lease.get("site_actuation_active", False))
        site_actuation_sequence_name = str(
            semantic_action_lease.get("site_actuation_sequence_name", "") or ""
        ).strip().lower()
        sequence_remaining = [
            str(action_name or "").strip().upper()
            for action_name in list(semantic_action_lease.get("site_actuation_sequence_remaining", []) or [])
            if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
        ]
        if sequence_remaining and (
            semantic_phase.startswith("site_actuation")
            or site_actuation_active
            or bool(site_actuation_sequence_name)
        ):
            suggested_actions = list(sequence_remaining)
        sequence_spec = self._resolve_adapter_sequence_spec(
            name=site_actuation_sequence_name,
            actions=sequence_remaining or suggested_actions,
            context={"reason": "semantic_move_scoring"},
        )
        sequence_action_bias = dict(sequence_spec.get("action_bias", {}) or {}) if sequence_spec else {}
        primary_suggested_action = suggested_actions[0] if suggested_actions else ""
        secondary_suggested_action = suggested_actions[1] if len(suggested_actions) > 1 else ""
        stagnant_turns = int(semantic_action_lease.get("site_progress_stagnant_turns", 0) or 0)
        site_reaching_score = float(semantic_action_lease.get("site_reaching_score", 0.0) or 0.0)
        site_zone_confidence = float(semantic_action_lease.get("site_zone_confidence", 0.0) or 0.0)
        site_distance = semantic_action_lease.get("site_distance")
        try:
            site_distance = None if site_distance is None else int(site_distance)
        except (TypeError, ValueError):
            site_distance = None
        site_distance_delta = semantic_action_lease.get("site_distance_delta")
        try:
            site_distance_delta = None if site_distance_delta is None else int(site_distance_delta)
        except (TypeError, ValueError):
            site_distance_delta = None
        branch_to_site_distance_min = semantic_action_lease.get("branch_to_site_distance_min")
        try:
            branch_to_site_distance_min = (
                None if branch_to_site_distance_min is None else int(branch_to_site_distance_min)
            )
        except (TypeError, ValueError):
            branch_to_site_distance_min = None
        site_reach_attempt = bool(semantic_action_lease.get("site_reach_attempt", False))
        recent_move_rows = self._recent_move_rows(agent, limit=8)

        effective_counts: dict[str, int] = {}
        recent_counts: dict[str, int] = {}
        no_progress_counts: dict[str, int] = {}
        last_failed_action = ""
        last_action_name = ""
        if recent_move_rows:
            latest = recent_move_rows[-1]
            last_action_name = str(latest.get("action", "") or "").strip().upper()
            if bool(latest.get("no_state_change")) or bool(latest.get("contradiction")):
                last_failed_action = str(latest.get("action", "") or "").strip().upper()
        for row in recent_move_rows:
            action_name = str(row.get("action", "") or "").strip().upper()
            if action_name not in _MOVE_ACTION_SET:
                continue
            recent_counts[action_name] = recent_counts.get(action_name, 0) + 1
            weak_outcome = bool(row.get("no_state_change")) or bool(row.get("contradiction"))
            if weak_outcome:
                no_progress_counts[action_name] = no_progress_counts.get(action_name, 0) + 1
            else:
                effective_counts[action_name] = effective_counts.get(action_name, 0) + 1

        candidates: list[str] = []
        for action_name in [*suggested_actions, fallback_action, *_MOVE_ACTIONS]:
            if action_name and action_name in _MOVE_ACTION_SET and action_name not in candidates:
                candidates.append(action_name)
        if not candidates:
            return fallback_action or "ACTION1"

        def _score(item: tuple[int, str]) -> tuple[float, int]:
            idx, action_name = item
            score = 0.0
            if action_name == fallback_action:
                score += 0.10
            if action_name in suggested_actions:
                suggested_index = suggested_actions.index(action_name)
                if suggested_index == 0:
                    score += 3.25
                elif suggested_index == 1:
                    score += 1.75
                else:
                    score += 0.9
            score += 0.40 * float(effective_counts.get(action_name, 0))
            score += 0.10 * float(recent_counts.get(action_name, 0))
            if (
                event_kind in _SEMANTIC_BRANCH_ENTRY_EVENTS
                and primary_suggested_action in {"ACTION3", "ACTION4"}
            ):
                if action_name == primary_suggested_action:
                    score += 1.5
                elif action_name == "ACTION1":
                    score -= 1.0
            if (
                semantic_phase == "branch_entry"
                and secondary_suggested_action
                and last_action_name == primary_suggested_action
            ):
                if action_name == primary_suggested_action:
                    score -= 2.5
                elif action_name == secondary_suggested_action:
                    score += 3.0
            if semantic_phase == "site_reaching":
                if action_name == primary_suggested_action:
                    score += 0.45
                if action_name == "ACTION1" and primary_suggested_action != "ACTION1":
                    score -= 1.0
                if no_progress_counts.get(action_name, 0) >= 2:
                    score -= 1.25 * float(min(2, no_progress_counts.get(action_name, 0) - 1))
                if (
                    secondary_suggested_action
                    and no_progress_counts.get(primary_suggested_action, 0) >= 2
                    and action_name == secondary_suggested_action
                ):
                    score += 1.35
                if (
                    event_kind == "state_transition_site_candidate"
                    and secondary_suggested_action
                    and last_action_name == primary_suggested_action
                ):
                    if action_name == primary_suggested_action:
                        score -= 3.0
                    elif action_name == secondary_suggested_action:
                        score += 4.0
                if event_kind == "branch_internal_cycle" and primary_suggested_action:
                    if action_name == primary_suggested_action:
                        score += 2.75
                    elif action_name == "ACTION1" and primary_suggested_action != "ACTION1":
                        score -= 2.5
                if (
                    secondary_suggested_action
                    and last_failed_action
                    and last_failed_action == primary_suggested_action
                ):
                    if action_name == primary_suggested_action:
                        score -= 2.5
                    elif action_name == secondary_suggested_action:
                        score += 3.0
                if (
                    secondary_suggested_action
                    and site_reach_attempt
                    and site_distance_delta is not None
                    and site_distance_delta >= 0
                ):
                    if action_name == primary_suggested_action:
                        score -= 1.25
                    elif action_name == secondary_suggested_action:
                        score += 1.75
                if (
                    secondary_suggested_action
                    and site_distance is not None
                    and branch_to_site_distance_min is not None
                    and site_distance > branch_to_site_distance_min
                ):
                    if action_name == primary_suggested_action:
                        score -= 1.75
                    elif action_name == secondary_suggested_action:
                        score += 2.5
                if stagnant_turns >= 2 and secondary_suggested_action:
                    if action_name == primary_suggested_action:
                        score -= 1.75
                    elif action_name == secondary_suggested_action:
                        score += 2.25
                if site_mode == "actuate" and action_name == "ACTION2":
                    score += 0.6
                if site_mode == "verify" and action_name == "ACTION1":
                    score += 0.6
                score += 0.25 * site_reaching_score
                score += 0.15 * site_zone_confidence
            if semantic_phase.startswith("site_actuation") or site_actuation_active:
                if action_name == primary_suggested_action:
                    score += 4.5
                elif action_name in suggested_actions:
                    score += 2.4
                else:
                    score -= 1.5
                if no_progress_counts.get(action_name, 0) >= 2:
                    score -= 1.5 * float(min(2, no_progress_counts.get(action_name, 0) - 1))
                if (
                    secondary_suggested_action
                    and last_action_name == primary_suggested_action
                ):
                    if action_name == primary_suggested_action:
                        score -= 1.75
                    elif action_name == secondary_suggested_action:
                        score += 2.5
                score += float(sequence_action_bias.get(action_name, 0.0) or 0.0)
                score += 0.35 * site_zone_confidence
                score += 0.20 * site_reaching_score
            if action_name in do_not_repeat_actions:
                score -= 2.0
            if last_failed_action and action_name == last_failed_action:
                score -= 1.0
            return score, -idx

        return max(enumerate(candidates), key=_score)[1]

    def _build_semantic_fallback_actions(
        self,
        *,
        agent: Any | None,
        semantic_action_lease: dict[str, Any] | None,
    ) -> list[PlannedAction]:
        if self._post_stabilize_no_target_continuation_step_active:
            return []
        semantic_action_lease = semantic_action_lease or {}
        if not bool(semantic_action_lease.get("active", False)):
            return []
        suggested_actions = list(semantic_action_lease.get("suggested_actions") or [])
        semantic_phase = str(semantic_action_lease.get("semantic_phase", "") or "").strip().lower()
        hidden_state_executor_active = bool(
            semantic_action_lease.get("hidden_state_executor_active", False)
        )
        hidden_state_goal_phase = hidden_state_executor_active and semantic_phase in {
            "state_change_verification",
            "goal_route_execution",
        }
        site_actuation_sequence_name = str(
            semantic_action_lease.get("site_actuation_sequence_name", "") or ""
        ).strip().lower()
        lease_horizon = max(1, int(semantic_action_lease.get("lease_horizon", 1) or 1))
        sequence_remaining = [
            str(action_name or "").strip().upper()
            for action_name in list(semantic_action_lease.get("site_actuation_sequence_remaining", []) or [])
            if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
        ]
        if sequence_remaining and not hidden_state_goal_phase and (
            semantic_phase.startswith("site_actuation")
            or bool(semantic_action_lease.get("site_actuation_active", False))
            or bool(site_actuation_sequence_name)
        ):
            suggested_actions = list(sequence_remaining)
        if semantic_phase.startswith("site_actuation") or (
            not hidden_state_goal_phase
            and
            sequence_remaining and (
                bool(semantic_action_lease.get("site_actuation_active", False))
                or bool(site_actuation_sequence_name)
            )
        ):
            queue_actions = [
                action_name
                for action_name in suggested_actions[: max(1, min(3, lease_horizon))]
                if action_name in _MOVE_ACTION_SET
            ]
            if not queue_actions:
                return []
            self._note_semantic_bridge_choice(
                semantic_action_lease=semantic_action_lease,
                action_name=queue_actions[0],
                source="semantic_site_actuation_sequence",
            )
            event_kind = str(semantic_action_lease.get("event_kind", "") or "").strip() or "semantic_hint"
            return [
                PlannedAction(
                    action_name=action_name,
                    reason=f"semantic_lease_fallback:{event_kind}",
                )
                for action_name in queue_actions
            ]
        if semantic_phase in {"state_change_verification", "goal_route_execution"}:
            queue_limit = max(1, min(8, lease_horizon, len(suggested_actions)))
            queue_actions = [
                action_name
                for action_name in suggested_actions[:queue_limit]
                if action_name in _MOVE_ACTION_SET
            ]
            if not queue_actions:
                return []
            self._note_semantic_bridge_choice(
                semantic_action_lease=semantic_action_lease,
                action_name=queue_actions[0],
                source="semantic_goal_route_sequence",
            )
            event_kind = str(semantic_action_lease.get("event_kind", "") or "").strip() or "semantic_hint"
            return [
                PlannedAction(
                    action_name=action_name,
                    reason=f"semantic_lease_fallback:{event_kind}",
                )
                for action_name in queue_actions
            ]
        fallback_action = suggested_actions[0] if suggested_actions else "ACTION1"
        selected_action = self._semantic_preferred_move_action(
            agent=agent,
            fallback_action=fallback_action,
            semantic_action_lease=semantic_action_lease,
        )
        if selected_action not in _MOVE_ACTION_SET:
            return []
        self._note_semantic_bridge_choice(
            semantic_action_lease=semantic_action_lease,
            action_name=selected_action,
            source="semantic_lease_fallback",
        )
        event_kind = str(semantic_action_lease.get("event_kind", "") or "").strip() or "semantic_hint"
        queue_actions: list[str] = [selected_action]
        queue_limit = lease_horizon
        if semantic_phase == "pre_entry" and event_kind == "state_mismatch":
            bootstrap_actions = [
                str(action_name or "").strip().upper()
                for action_name in list(suggested_actions or [])
                if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
            ]
            if bootstrap_actions:
                queue_actions = list(bootstrap_actions[: max(1, min(2, lease_horizon))])
                return [
                    PlannedAction(
                        action_name=action_name,
                        reason=f"semantic_lease_fallback:{event_kind}",
                    )
                    for action_name in queue_actions
                ]
        if semantic_phase in {"site_reaching", "branch_entry"}:
            if semantic_phase == "branch_entry":
                queue_limit = min(queue_limit, 2)
            for action_name in suggested_actions:
                action_name = str(action_name or "").strip().upper()
                if action_name in _MOVE_ACTION_SET and action_name not in queue_actions:
                    queue_actions.append(action_name)
                if len(queue_actions) >= min(3, queue_limit):
                    break
        return [
            PlannedAction(
                action_name=action_name,
                reason=f"semantic_lease_fallback:{event_kind}",
            )
            for action_name in queue_actions[:queue_limit]
        ]

    def _note_semantic_assisted_local_step(self) -> None:
        if self._last_semantic_recommendation_active and self._last_semantic_selected_action:
            self._total_semantic_assisted_local_steps += 1

    def _normalize_planned_actions(
        self,
        *,
        runtime: Any,
        actions: list[PlannedAction],
        selected_target: ScoredCandidate | None,
        agent: Any | None = None,
        semantic_action_lease: dict[str, Any] | None = None,
    ) -> list[PlannedAction]:
        normalized: list[PlannedAction] = []
        for action in actions:
            normalized_action = self._normalize_planned_action(
                runtime=runtime,
                action=action,
                selected_target=selected_target,
                agent=agent,
                semantic_action_lease=semantic_action_lease,
            )
            if normalized_action is None:
                return []
            normalized.append(normalized_action)
        normalized = self._maybe_expand_stale_move_packet(
            runtime=runtime,
            raw_actions=actions,
            normalized_actions=normalized,
            selected_target=selected_target,
            agent=agent,
            semantic_action_lease=semantic_action_lease,
        )
        normalized = self._maybe_expand_click_only_grid_probe_packet(
            runtime=runtime,
            normalized_actions=normalized,
            selected_target=selected_target,
        )
        return self._maybe_expand_runtime_followup_packet(
            runtime=runtime,
            raw_actions=actions,
            normalized_actions=normalized,
            selected_target=selected_target,
            agent=agent,
            semantic_action_lease=semantic_action_lease,
        )

    def _maybe_expand_click_only_grid_probe_packet(
        self,
        *,
        runtime: Any,
        normalized_actions: list[PlannedAction],
        selected_target: ScoredCandidate | None,
    ) -> list[PlannedAction]:
        if self._move_action_available:
            return normalized_actions
        if selected_target is None or len(normalized_actions) != 1:
            return normalized_actions
        action = normalized_actions[0]
        if str(action.action_name or "").strip().upper() != "ACTION6":
            return normalized_actions
        source = str(selected_target.source or "").strip()
        if source not in {"grid_scan", "level_rule_symbol_scout"}:
            return normalized_actions
        if self._runtime_level_number(runtime) < 2:
            return normalized_actions
        key = str(selected_target.anchor_key or selected_target.component_id or "").strip()
        if not key:
            return normalized_actions
        issued_count = int(self._click_only_grid_probe_packet_counts.get(key, 0) or 0)
        if issued_count >= 5:
            if self._click_only_grid_probe_focus_key == key:
                self._click_only_grid_probe_focus_key = ""
            return normalized_actions
        self._click_only_grid_probe_packet_counts[key] = issued_count + 1
        self._click_only_grid_probe_focus_key = key
        self._click_only_grid_probe_focus_score = self._runtime_score(runtime)
        self._click_only_grid_probe_focus_level = self._runtime_level_number(runtime)
        self._click_only_grid_probe_focus_attempt = int(
            getattr(runtime, "last_arc_attempt_number", 0) or 0
        )
        return [
            action,
            PlannedAction(
                action_name=action.action_name,
                target_component_id=action.target_component_id,
                parameters=dict(action.parameters),
                reason=(action.reason or "click_only_grid_probe_packet") + ":repeat",
            ),
        ]

    def _normalize_planned_action(
        self,
        *,
        runtime: Any,
        action: PlannedAction,
        selected_target: ScoredCandidate | None,
        agent: Any | None = None,
        semantic_action_lease: dict[str, Any] | None = None,
    ) -> PlannedAction | None:
        semantic_action_lease = semantic_action_lease or {}
        queue_oriented_phase = str(
            semantic_action_lease.get("semantic_phase", "") or ""
        ).strip().lower() in {"state_change_verification", "goal_route_execution"}
        queue_suggested_actions = [
            str(action_name or "").strip().upper()
            for action_name in list(semantic_action_lease.get("suggested_actions") or [])
            if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
        ]
        valid_name = str(action.action_name or "").strip().upper()
        if valid_name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7", "RESET"}:
            if valid_name in _MOVE_ACTION_SET:
                redirected = self._semantic_preferred_move_action(
                    agent=agent,
                    fallback_action=valid_name,
                    semantic_action_lease=semantic_action_lease,
                )
                if (
                    redirected != valid_name
                    and valid_name in set((semantic_action_lease or {}).get("do_not_repeat_actions") or [])
                    and valid_name not in set((semantic_action_lease or {}).get("suggested_actions") or [])
                ):
                    self._note_semantic_bridge_choice(
                        semantic_action_lease=semantic_action_lease,
                        action_name=redirected,
                        source="semantic_penalty_redirect",
                    )
                    return PlannedAction(
                        action_name=redirected,
                        target_component_id=action.target_component_id,
                        parameters=dict(action.parameters),
                        reason=action.reason or "semantic_penalty_redirect",
                    )
                self._note_semantic_bridge_choice(
                    semantic_action_lease=semantic_action_lease,
                    action_name=valid_name,
                    source="explicit_move_action",
                )
            return action

        raw_name = str(action.action_name or "").strip().lower()
        if raw_name == "reset":
            return PlannedAction(
                action_name="RESET",
                target_component_id=action.target_component_id,
                parameters={},
                reason=action.reason,
            )

        if raw_name == "move":
            direction = str(action.parameters.get("direction", "") or "").strip().lower()
            target_health = getattr(getattr(runtime, "_last_target_health_result", None), "health", None)
            target_health = getattr(target_health, "value", target_health)
            stale_move_redirect = bool(
                not bool(semantic_action_lease.get("active", False))
                and
                str(target_health or "").strip().lower() in {"stale", "saturated"}
                and selected_target is not None
                and str(selected_target.action_family or "").strip().lower() == "move"
            )
            click_nochange_lateral_probe = False
            if (
                not direction
                and not bool(semantic_action_lease.get("active", False))
                and selected_target is not None
                and str(selected_target.action_family or "").strip().lower() == "move"
                and self._l1_lateral_probe_anchor_allowed(selected_target)
                and self._runtime_level_number(runtime) <= 1
                and self._runtime_score(runtime) <= 0
            ):
                effect_entry = self._runtime_local_effect_entry(
                    runtime=runtime,
                    component_id=str(selected_target.component_id or ""),
                    anchor_key=str(selected_target.anchor_key or ""),
                )
                click_nochange_lateral_probe = bool(
                    str(effect_entry.get("last_family", "") or "").strip().lower() == "click"
                    and int(effect_entry.get("score_change_count", 0) or 0) <= 0
                    and int(effect_entry.get("board_change_no_score_count", 0) or 0) <= 0
                    and int(effect_entry.get("no_change_count", 0) or 0) > 0
                )
            fallback_move_name = {
                "up": "ACTION1",
                "down": "ACTION2",
                "left": "ACTION3",
                "right": "ACTION4",
                "action1": "ACTION1",
                "action2": "ACTION2",
                "action3": "ACTION3",
                "action4": "ACTION4",
            }.get(direction, self._dominant_move_action(agent, semantic_action_lease=semantic_action_lease))
            move_name = fallback_move_name
            move_source = "normalized_move_action"
            if not direction and queue_oriented_phase and queue_suggested_actions:
                move_name = queue_suggested_actions[0]
                move_source = "semantic_queue_move"
            elif not direction and click_nochange_lateral_probe:
                move_name = "ACTION3"
                move_source = "click_nochange_lateral_probe"
                if not self._l1_lateral_probe_queue:
                    self._l1_lateral_probe_queue = list(_L1_LATERAL_CONTOUR_ACTIONS)
                self._l1_lateral_probe_remaining = len(self._l1_lateral_probe_queue)
                self._l1_lateral_probe_component_id = str(selected_target.component_id or "")
            elif not direction and stale_move_redirect:
                redirected = self._stale_move_redirect_action(agent)
                if redirected:
                    move_name = redirected
                    move_source = "stale_move_redirect"
            elif not direction:
                move_name = self._semantic_preferred_move_action(
                    agent=agent,
                    fallback_action=fallback_move_name,
                    semantic_action_lease=semantic_action_lease,
                )
                if move_name != fallback_move_name:
                    move_source = "semantic_priority_move"
                elif bool((semantic_action_lease or {}).get("active", False)) and move_name in set((semantic_action_lease or {}).get("suggested_actions") or []):
                    move_source = "semantic_priority_move"
            elif (
                bool((semantic_action_lease or {}).get("active", False))
                and fallback_move_name in set((semantic_action_lease or {}).get("do_not_repeat_actions") or [])
                and fallback_move_name not in set((semantic_action_lease or {}).get("suggested_actions") or [])
            ):
                redirected = self._semantic_preferred_move_action(
                    agent=agent,
                    fallback_action=fallback_move_name,
                    semantic_action_lease=semantic_action_lease,
                )
                if redirected != fallback_move_name:
                    move_name = redirected
                    move_source = "semantic_penalty_redirect"
            self._note_semantic_bridge_choice(
                semantic_action_lease=semantic_action_lease,
                action_name=move_name,
                source=move_source,
            )
            return PlannedAction(
                action_name=move_name,
                target_component_id=action.target_component_id,
                parameters={},
                reason=action.reason or move_source,
            )

        if raw_name in {"click", "select"}:
            position = self._resolve_target_grid_position(
                runtime=runtime,
                selected_target=selected_target,
                target_component_id=str(action.target_component_id or ""),
                parameters=action.parameters,
            )
            if position is None:
                return None
            row, col = position
            return PlannedAction(
                action_name="ACTION6",
                target_component_id=action.target_component_id,
                # Analyzer/agent ACTION6 payloads use prompt-grid coordinates:
                # x=row and y=col. BaseArcAgent translates them to env x/y.
                parameters={"x": int(row), "y": int(col)},
                reason=action.reason or "normalized_click_action",
            )

        return None

    def _dominant_move_action(
        self,
        agent: Any | None,
        *,
        semantic_action_lease: dict[str, Any] | None = None,
    ) -> str:
        if bool((semantic_action_lease or {}).get("active", False)):
            semantic_move = self._semantic_preferred_move_action(
                agent=agent,
                fallback_action="ACTION1",
                semantic_action_lease=semantic_action_lease,
            )
            if semantic_move:
                return semantic_move
        recent_history = self._recent_move_rows(agent, limit=6)
        effective_moves = [
            str(row.get("action", "") or "")
            for row in recent_history
            if bool(row.get("no_state_change")) is False
            and str(row.get("action", "") or "") in _MOVE_ACTION_SET
        ]
        if effective_moves:
            return max(set(effective_moves), key=effective_moves.count)
        failed_counts: dict[str, int] = {}
        latest_failed_action = ""
        for row in recent_history:
            action_name = str(row.get("action", "") or "").strip().upper()
            if action_name not in _MOVE_ACTION_SET:
                continue
            if bool(row.get("no_state_change")) or bool(row.get("contradiction")):
                failed_counts[action_name] = failed_counts.get(action_name, 0) + 1
                latest_failed_action = action_name
        if failed_counts:
            rotated_actions = list(_MOVE_ACTIONS)
            if latest_failed_action in rotated_actions:
                pivot = rotated_actions.index(latest_failed_action)
                rotated_actions = rotated_actions[pivot + 1 :] + rotated_actions[: pivot + 1]
            return min(
                rotated_actions,
                key=lambda action_name: (
                    int(failed_counts.get(action_name, 0) or 0),
                    int(latest_failed_action == action_name),
                ),
            )
        recent_moves = [
            str(row.get("action", "") or "")
            for row in recent_history
            if str(row.get("action", "") or "") in _MOVE_ACTION_SET
        ]
        if recent_moves:
            return max(set(recent_moves), key=recent_moves.count)
        return "ACTION1"

    def _stale_move_redirect_action(
        self,
        agent: Any | None,
    ) -> str:
        actions = self._stale_move_redirect_actions(agent, limit=1)
        return actions[0] if actions else ""

    def _stale_move_redirect_actions(
        self,
        agent: Any | None,
        *,
        limit: int,
    ) -> list[str]:
        limit = max(1, int(limit or 1))
        recent_move_rows = self._recent_move_rows(agent, limit=6)
        if not recent_move_rows:
            return list(_MOVE_ACTIONS[:limit])
        latest_action = str(recent_move_rows[-1].get("action", "") or "").strip().upper()
        if latest_action not in _MOVE_ACTION_SET:
            return list(_MOVE_ACTIONS[:limit])
        move_counts: dict[str, int] = {}
        for row in recent_move_rows:
            action_name = str(row.get("action", "") or "").strip().upper()
            if action_name in _MOVE_ACTION_SET:
                move_counts[action_name] = move_counts.get(action_name, 0) + 1
        rotated_actions = list(_MOVE_ACTIONS)
        pivot = rotated_actions.index(latest_action)
        rotated_actions = rotated_actions[pivot + 1 :] + rotated_actions[: pivot + 1]
        ordered_actions = sorted(
            rotated_actions,
            key=lambda action_name: (
                int(move_counts.get(action_name, 0) or 0),
                int(action_name == latest_action),
                rotated_actions.index(action_name),
            ),
        )
        return ordered_actions[:limit]

    def _maybe_expand_stale_move_packet(
        self,
        *,
        runtime: Any,
        raw_actions: list[PlannedAction],
        normalized_actions: list[PlannedAction],
        selected_target: ScoredCandidate | None,
        agent: Any | None,
        semantic_action_lease: dict[str, Any] | None,
    ) -> list[PlannedAction]:
        semantic_action_lease = semantic_action_lease or {}
        if bool(semantic_action_lease.get("active", False)):
            return normalized_actions
        if len(raw_actions) != 1 or len(normalized_actions) != 1:
            return normalized_actions
        if str(normalized_actions[0].reason or "").strip() == "click_nochange_lateral_probe":
            return normalized_actions
        if (
            selected_target is not None
            and str(selected_target.action_family or "").strip().lower() == "move"
            and self._l1_lateral_probe_anchor_allowed(selected_target)
            and self._runtime_level_number(runtime) <= 1
            and self._runtime_score(runtime) <= 0
        ):
            effect_entry = self._runtime_local_effect_entry(
                runtime=runtime,
                component_id=str(selected_target.component_id or ""),
                anchor_key=str(selected_target.anchor_key or ""),
            )
            if (
                str(effect_entry.get("last_family", "") or "").strip().lower() == "click"
                and int(effect_entry.get("score_change_count", 0) or 0) <= 0
                and int(effect_entry.get("board_change_no_score_count", 0) or 0) <= 0
                and int(effect_entry.get("no_change_count", 0) or 0) > 0
            ):
                if not self._l1_lateral_probe_queue:
                    self._l1_lateral_probe_queue = list(_L1_LATERAL_CONTOUR_ACTIONS)
                self._l1_lateral_probe_remaining = len(self._l1_lateral_probe_queue)
                self._l1_lateral_probe_component_id = str(selected_target.component_id or "")
                return [
                    PlannedAction(
                        action_name="ACTION3",
                        target_component_id=str(normalized_actions[0].target_component_id or ""),
                        parameters={},
                        reason="click_nochange_lateral_probe",
                    )
                ]
        raw_action = raw_actions[0]
        if str(raw_action.action_name or "").strip().lower() != "move":
            return normalized_actions
        if str((raw_action.parameters or {}).get("direction", "") or "").strip():
            return normalized_actions
        target_health = getattr(getattr(runtime, "_last_target_health_result", None), "health", None)
        target_health = getattr(target_health, "value", target_health)
        if str(target_health or "").strip().lower() not in {"stale", "saturated"}:
            return normalized_actions
        if selected_target is None or str(selected_target.action_family or "").strip().lower() != "move":
            return normalized_actions
        packet_size = min(
            max(1, int(getattr(self.config.micro_planner_config, "max_actions_per_plan", 2) or 2)),
            2,
        )
        if packet_size <= 1:
            return normalized_actions
        packet_actions = self._stale_move_redirect_actions(agent, limit=packet_size)
        if len(packet_actions) <= 1:
            return normalized_actions
        return [
            PlannedAction(
                action_name=action_name,
                target_component_id=str(normalized_actions[0].target_component_id or ""),
                parameters={},
                reason="stale_move_packet",
            )
            for action_name in packet_actions
        ]

    def _maybe_expand_runtime_followup_packet(
        self,
        *,
        runtime: Any,
        raw_actions: list[PlannedAction],
        normalized_actions: list[PlannedAction],
        selected_target: ScoredCandidate | None,
        agent: Any | None,
        semantic_action_lease: dict[str, Any] | None,
    ) -> list[PlannedAction]:
        semantic_action_lease = semantic_action_lease or {}
        if bool(semantic_action_lease.get("active", False)):
            return normalized_actions
        rule_packet = self._maybe_expand_rule_hypothesis_followup_packet(
            runtime=runtime,
            raw_actions=raw_actions,
            normalized_actions=normalized_actions,
            selected_target=selected_target,
        )
        if rule_packet:
            return rule_packet
        if len(raw_actions) != 1 or len(normalized_actions) != 1:
            return normalized_actions
        raw_action = raw_actions[0]
        if str(raw_action.action_name or "").strip().lower() != "move":
            return normalized_actions
        if str((raw_action.parameters or {}).get("direction", "") or "").strip():
            return normalized_actions
        if selected_target is None or str(selected_target.action_family or "").strip().lower() != "move":
            return normalized_actions

        result = dict(getattr(runtime, "last_runtime_micro_probe_result", None) or {})
        if not result or bool(result.get("weak")) or bool(result.get("followup_packet_used")):
            return normalized_actions
        if bool(result.get("suppression_resolution_applied")):
            return normalized_actions

        runtime_followup_candidate = getattr(runtime, "_has_runtime_followup_candidate", None)
        if callable(runtime_followup_candidate):
            try:
                if not bool(runtime_followup_candidate(result)):
                    return normalized_actions
            except Exception:
                return normalized_actions

        continuation_score = float(result.get("continuation_score", 0.0) or 0.0)
        continuation_threshold = float(result.get("continuation_threshold", 0.0) or 0.0)
        if continuation_score < continuation_threshold:
            return normalized_actions

        selected_component_id = str(selected_target.component_id or "").strip()
        selected_anchor_key = str(selected_target.anchor_key or "").strip()
        result_component_id = str(result.get("selected_target_id", "") or "").strip()
        result_anchor_key = str(result.get("selected_target_anchor", "") or "").strip()
        if result_component_id and selected_component_id and result_component_id != selected_component_id:
            return normalized_actions
        if result_anchor_key and selected_anchor_key and result_anchor_key != selected_anchor_key:
            return normalized_actions

        current_state = getattr(runtime, "last_state", None)
        current_step_idx = int(getattr(current_state, "step_index", -10_000) or -10_000)
        result_step_idx = int(result.get("step_idx", -10_000) or -10_000)
        if current_step_idx not in {result_step_idx, result_step_idx + 1}:
            return normalized_actions

        primary_action = str(normalized_actions[0].action_name or "").strip().upper()
        if primary_action not in _MOVE_ACTION_SET:
            return normalized_actions
        packet_actions = [primary_action]
        for action_name in self._stale_move_redirect_actions(agent, limit=4):
            if action_name and action_name not in packet_actions:
                packet_actions.append(action_name)
            if len(packet_actions) >= 2:
                break
        if len(packet_actions) < 2:
            return normalized_actions

        packet_actions = packet_actions[:2]
        event = {
            "kind": "runtime_followup_packet",
            "selected_target_id": result_component_id or selected_component_id,
            "selected_target_anchor": result_anchor_key or selected_anchor_key,
            "continuation_score": continuation_score,
            "continuation_threshold": continuation_threshold,
            "source": str(result.get("source", "") or ""),
            "actions": list(packet_actions),
        }
        note_loaded = getattr(runtime, "note_runtime_followup_packet_loaded", None)
        if callable(note_loaded):
            try:
                note_loaded(event=event)
            except Exception:
                result["followup_packet_used"] = True
                result["followup_packet_reserved"] = False
                result["followup_packet_loaded_step_idx"] = current_step_idx
                result["followup_packet_actions"] = list(packet_actions)
                runtime.last_runtime_micro_probe_result = result
        else:
            result["followup_packet_used"] = True
            result["followup_packet_reserved"] = False
            result["followup_packet_loaded_step_idx"] = current_step_idx
            result["followup_packet_actions"] = list(packet_actions)
            runtime.last_runtime_micro_probe_result = result
        return [
            PlannedAction(
                action_name=action_name,
                target_component_id=str(normalized_actions[0].target_component_id or ""),
                parameters={},
                reason="runtime_followup_packet",
            )
            for action_name in packet_actions
        ]

    def _maybe_expand_rule_hypothesis_followup_packet(
        self,
        *,
        runtime: Any,
        raw_actions: list[PlannedAction],
        normalized_actions: list[PlannedAction],
        selected_target: ScoredCandidate | None,
    ) -> list[PlannedAction]:
        if len(raw_actions) != 1 or len(normalized_actions) != 1:
            return []
        if selected_target is None:
            return []
        if str(selected_target.source or "").strip() != "rule_hypothesis":
            return []
        if str(selected_target.action_family or "").strip().lower() not in {"click", "move"}:
            return []
        builder = getattr(runtime, "build_rule_hypothesis_followup_packet", None)
        if not callable(builder):
            return []
        current_state = getattr(runtime, "last_state", None)
        current_step = int(getattr(current_state, "step_index", 0) or 0)
        try:
            packet = list(
                builder(
                    selected_target={
                        "component_id": str(selected_target.component_id or ""),
                        "anchor_key": str(selected_target.anchor_key or ""),
                    },
                    current_step=current_step,
                )
                or []
            )
        except Exception:
            return []
        actions: list[PlannedAction] = []
        seen_components: set[str] = set()
        for item in packet[:2]:
            if not isinstance(item, dict):
                continue
            component_id = str(item.get("component_id", "") or "").strip()
            action_name = str(item.get("action_name", "") or "ACTION6").strip().upper()
            is_move_action = action_name in _MOVE_ACTION_SET
            if not component_id or component_id in seen_components:
                if str(item.get("stage", "") or "") != "source_repeat" and not is_move_action:
                    continue
            try:
                row = int(item.get("center_row", 0) or 0)
                col = int(item.get("center_col", 0) or 0)
            except (TypeError, ValueError):
                continue
            seen_components.add(component_id)
            actions.append(
                PlannedAction(
                    action_name=action_name if action_name in _MOVE_ACTION_SET else "ACTION6",
                    target_component_id=component_id,
                    parameters={} if is_move_action else {"x": int(row), "y": int(col)},
                    reason="rule_hypothesis_followup_packet",
                )
            )
        return actions if len(actions) >= 2 else []

    def _resolve_target_grid_position(
        self,
        *,
        runtime: Any,
        selected_target: ScoredCandidate | None,
        target_component_id: str,
        parameters: dict[str, Any],
    ) -> tuple[int, int] | None:
        position = dict(
            parameters.get("position")
            or parameters.get("coordinates")
            or parameters.get("center")
            or {}
        )
        if position:
            if "row" in position or "col" in position:
                row = int(position.get("row", 0) or 0)
                col = int(position.get("col", 0) or 0)
            else:
                row = int(position.get("y", 0) or 0)
                col = int(position.get("x", 0) or 0)
            return row, col

        component_id = str(target_component_id or "")
        anchor_key = ""
        if selected_target is not None:
            component_id = component_id or str(selected_target.component_id or "")
            anchor_key = str(selected_target.anchor_key or "")
            if int(selected_target.center_row or 0) or int(selected_target.center_col or 0):
                row = int(selected_target.center_row or 0)
                col = int(selected_target.center_col or 0)
                alternate = self._alternate_component_click_probe_position(
                    runtime=runtime,
                    component_id=component_id,
                    anchor_key=anchor_key,
                    fallback_row=row,
                    fallback_col=col,
                )
                return alternate or (row, col)

        for entry in list(getattr(runtime, "last_candidate_pool", None) or []):
            entry_component_id = str(entry.get("component_id", "") or entry.get("id", "") or "")
            if component_id and entry_component_id != component_id:
                continue
            row = int(entry.get("center_row", entry.get("row", 0)) or 0)
            col = int(entry.get("center_col", entry.get("col", 0)) or 0)
            if row or col:
                return row, col
            anchor_key = anchor_key or str(entry.get("anchor_key", "") or "")

        base_pool = getattr(runtime, "_last_base_pool", None)
        for candidate in list(getattr(base_pool, "candidates", None) or []):
            if component_id and str(getattr(candidate, "component_id", "") or "") != component_id:
                continue
            row = int(getattr(candidate, "center_row", 0) or 0)
            col = int(getattr(candidate, "center_col", 0) or 0)
            if row or col:
                return row, col
            anchor_key = anchor_key or str(getattr(candidate, "anchor_key", "") or "")

        selected_summary = getattr(runtime, "last_selected_target_summary", None)
        if isinstance(selected_summary, dict):
            summary_component_id = str(selected_summary.get("component_id", "") or selected_summary.get("id", "") or "")
            if not component_id or summary_component_id == component_id:
                anchor_key = anchor_key or str(selected_summary.get("anchor_key", "") or "")

        return self._parse_anchor_key_position(runtime, anchor_key)

    def _alternate_component_click_probe_position(
        self,
        *,
        runtime: Any,
        component_id: str,
        anchor_key: str,
        fallback_row: int,
        fallback_col: int,
    ) -> tuple[int, int] | None:
        component = str(component_id or "").strip()
        if not component:
            return None
        effect_entry = self._runtime_local_effect_entry(
            runtime=runtime,
            component_id=component,
            anchor_key=anchor_key,
        )
        if not effect_entry:
            return None
        if int(effect_entry.get("score_change_count", 0) or 0) > 0:
            return None
        probe_count = int(effect_entry.get("probe_count", 0) or 0)
        no_change_count = int(effect_entry.get("no_change_count", 0) or 0)
        weak_effect_count = int(effect_entry.get("weak_effect_count", 0) or 0)
        if probe_count <= 0 or (no_change_count <= 0 and weak_effect_count <= 0):
            return None

        entry = self._object_table_entry_for_component(runtime=runtime, component_id=component)
        bbox = dict((entry or {}).get("bbox") or {})
        if not bbox:
            return self._alternate_grid_scan_click_probe_position(
                runtime=runtime,
                component_id=component,
                anchor_key=anchor_key,
                fallback_row=fallback_row,
                fallback_col=fallback_col,
                probe_count=probe_count,
            )
        min_row = int(bbox.get("min_row", fallback_row) or fallback_row)
        max_row = int(bbox.get("max_row", min_row) or min_row)
        min_col = int(bbox.get("min_col", fallback_col) or fallback_col)
        max_col = int(bbox.get("max_col", min_col) or min_col)
        if min_row == max_row and min_col == max_col:
            return None
        mid_row = (min_row + max_row) // 2
        mid_col = (min_col + max_col) // 2
        raw_points = [
            (min_row, min_col),
            (max_row, max_col),
            (min_row, max_col),
            (max_row, min_col),
            (min_row, mid_col),
            (max_row, mid_col),
            (mid_row, min_col),
            (mid_row, max_col),
            (mid_row, mid_col),
        ]
        points: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        fallback = (int(fallback_row), int(fallback_col))
        for point in raw_points:
            row = max(min_row, min(max_row, int(point[0])))
            col = max(min_col, min(max_col, int(point[1])))
            normalized = (row, col)
            if normalized == fallback or normalized in seen:
                continue
            seen.add(normalized)
            points.append(normalized)
        if not points:
            return None
        return points[(max(1, probe_count) - 1) % len(points)]

    def _alternate_grid_scan_click_probe_position(
        self,
        *,
        runtime: Any,
        component_id: str,
        anchor_key: str,
        fallback_row: int,
        fallback_col: int,
        probe_count: int,
    ) -> tuple[int, int] | None:
        component = str(component_id or "").strip()
        anchor = str(anchor_key or "").strip()
        if not (component.startswith("grid_scan_") or anchor.startswith("grid||")):
            return None
        board_rows = self._runtime_board_rows(runtime)
        if not board_rows:
            return None
        max_row = len(board_rows) - 1
        max_col = max((len(row) for row in board_rows), default=0) - 1
        if max_row < 0 or max_col < 0:
            return None
        # Probe a small multi-scale ring around virtual grid points; this fixes
        # display/grid quantization misses without encoding any game rule.
        offsets = [
            (5, 0),
            (0, 5),
            (-5, 0),
            (0, -5),
            (5, 5),
            (-5, 5),
            (5, -5),
            (-5, -5),
            (2, 0),
            (0, 2),
            (-2, 0),
            (0, -2),
            (2, 6),
            (-2, 6),
            (6, 2),
            (6, -2),
            (-6, 2),
            (2, -6),
            (-2, -6),
            (-6, -2),
        ]
        points: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        fallback = (int(fallback_row), int(fallback_col))
        for row_delta, col_delta in offsets:
            row = max(0, min(max_row, int(fallback_row) + int(row_delta)))
            col = max(0, min(max_col, int(fallback_col) + int(col_delta)))
            normalized = (row, col)
            if normalized == fallback or normalized in seen:
                continue
            seen.add(normalized)
            points.append(normalized)
        if not points:
            return None
        return points[(max(1, int(probe_count or 1)) - 1) % len(points)]

    @staticmethod
    def _object_table_entry_for_component(*, runtime: Any, component_id: str) -> dict[str, Any]:
        component = str(component_id or "").strip()
        if not component:
            return {}
        state = getattr(runtime, "last_state", None)
        for entry in list(getattr(state, "object_table", None) or []):
            if not isinstance(entry, dict):
                continue
            if str(entry.get("component_id", "") or "").strip() == component:
                return dict(entry)
        return {}

    def _parse_anchor_key_position(self, runtime: Any, anchor_key: str) -> tuple[int, int] | None:
        normalized = str(anchor_key or "").strip()
        if not normalized:
            return None
        parser = getattr(runtime, "_parse_anchor_key", None)
        if callable(parser):
            try:
                parsed = parser(normalized)
            except Exception:
                parsed = None
            if parsed and len(parsed) == 3:
                return int(parsed[1]), int(parsed[2])

        parts = [item.strip() for item in normalized.split("||")]
        if len(parts) == 3:
            try:
                return int(parts[1]), int(parts[2])
            except ValueError:
                return None
        return None

    @staticmethod
    def _coerce_board_rows(raw: Any) -> list[str]:
        if isinstance(raw, str):
            return [str(row) for row in raw.splitlines()]
        if isinstance(raw, (list, tuple)):
            rows: list[str] = []
            for row in raw:
                if isinstance(row, str):
                    rows.append(row)
                elif isinstance(row, (list, tuple)):
                    rows.append("".join(str(cell) for cell in row))
            return rows
        return []

    @staticmethod
    def _normalize_history_action_name(raw: Any) -> str:
        match = re.search(r"\b(ACTION\d+|RESET)\b", str(raw or "").upper())
        return match.group(1) if match else ""

    @staticmethod
    def _coerce_board_value_grid(raw: Any) -> list[list[int]]:
        def parse_cell(cell: Any) -> int:
            if isinstance(cell, bool):
                return int(cell)
            if isinstance(cell, int):
                return max(0, min(15, int(cell)))
            text = str(cell or "")
            if not text:
                return 0
            if len(text) == 1:
                if text in _ASCII_GRID_VALUE_BY_CHAR:
                    return int(_ASCII_GRID_VALUE_BY_CHAR[text])
                if text in ".":
                    return 0
                if text.upper() in "0123456789ABCDEF":
                    return int(text, 16)
            try:
                return max(0, min(15, int(text)))
            except (TypeError, ValueError):
                return 0

        if isinstance(raw, str):
            return [[parse_cell(cell) for cell in row] for row in raw.splitlines() if row]
        if isinstance(raw, (list, tuple)):
            rows: list[list[int]] = []
            for row in raw:
                if isinstance(row, str):
                    rows.append([parse_cell(cell) for cell in row])
                elif isinstance(row, (list, tuple)):
                    rows.append([parse_cell(cell) for cell in row])
            width = min((len(row) for row in rows if row), default=0)
            if width <= 0:
                return []
            return [row[:width] for row in rows if len(row) >= width]
        return []

    def _recent_auxiliary_action_name(
        self,
        runtime: Any,
        *,
        history_limit: int = 18,
    ) -> str:
        state = getattr(runtime, "last_state", None)
        recent_history = list(getattr(state, "recent_history", None) or [])
        for step in reversed(recent_history[-max(1, int(history_limit or 1)) :]):
            if not isinstance(step, dict):
                continue
            action_name = self._normalize_history_action_name(step.get("action"))
            if (
                action_name
                and action_name not in _MOVE_ACTION_SET
                and action_name not in {_CLICK_ACTION, "RESET"}
            ):
                return action_name
        return ""

    def _compact_grid_components_from_rows(self, raw_rows: Any) -> list[dict[str, Any]]:
        grid = self._coerce_board_value_grid(raw_rows)
        if not grid:
            return []
        row_count = len(grid)
        col_count = min((len(row) for row in grid), default=0)
        if row_count <= 0 or col_count <= 0:
            return []
        values = [int(cell) for row in grid for cell in row[:col_count]]
        background = Counter(values).most_common(1)[0][0] if values else 0
        visited: set[tuple[int, int]] = set()
        components: list[dict[str, Any]] = []
        for start_row in range(row_count):
            for start_col in range(col_count):
                if (start_row, start_col) in visited:
                    continue
                value = int(grid[start_row][start_col])
                stack = [(start_row, start_col)]
                visited.add((start_row, start_col))
                cells: list[tuple[int, int]] = []
                while stack:
                    row, col = stack.pop()
                    cells.append((row, col))
                    for next_row, next_col in (
                        (row - 1, col),
                        (row + 1, col),
                        (row, col - 1),
                        (row, col + 1),
                    ):
                        if (
                            next_row < 0
                            or next_row >= row_count
                            or next_col < 0
                            or next_col >= col_count
                            or (next_row, next_col) in visited
                            or int(grid[next_row][next_col]) != value
                        ):
                            continue
                        visited.add((next_row, next_col))
                        stack.append((next_row, next_col))
                if value == background:
                    continue
                rows = [row for row, _col in cells]
                cols = [col for _row, col in cells]
                min_row = min(rows)
                max_row = max(rows)
                min_col = min(cols)
                max_col = max(cols)
                size = len(cells)
                components.append(
                    {
                        "component_id": f"grid_value_{value}_{min_row}_{min_col}_{max_row}_{max_col}",
                        "anchor_key": f"grid||{(min_row + max_row) // 2}||{(min_col + max_col) // 2}",
                        "row": int((min_row + max_row) // 2),
                        "col": int((min_col + max_col) // 2),
                        "value": int(value),
                        "size": int(size),
                        "bbox": {
                            "min_row": int(min_row),
                            "max_row": int(max_row),
                            "min_col": int(min_col),
                            "max_col": int(max_col),
                        },
                    }
                )
        return components

    def _build_auxiliary_visual_arrangement_actions(
        self,
        *,
        runtime: Any,
        validate_action_name: str,
        record: bool = True,
    ) -> list[PlannedAction]:
        validate_action = str(validate_action_name or "").strip().upper()
        if (
            not _GENERIC_KEYBOARD_ACTION_RE.match(validate_action)
            or validate_action in _MOVE_ACTION_SET
            or validate_action in {_CLICK_ACTION, "RESET"}
        ):
            return []
        raw_rows = getattr(runtime, "last_board_rows", None)
        components = self._compact_grid_components_from_rows(raw_rows)
        if not components:
            return []
        row_count = max((int(component["row"]) for component in components), default=0) + 1
        row_count = max(row_count, len(list(raw_rows or [])) if isinstance(raw_rows, list) else row_count)
        if row_count < 12:
            return []

        upper_limit = max(4, int(float(row_count) * 0.16))
        lower_min = max(upper_limit + 8, int(float(row_count) * 0.76))
        middle_min = max(upper_limit + 3, int(float(row_count) * 0.22))
        middle_max = max(middle_min + 1, int(float(row_count) * 0.68))
        compact_max = max(18, int(float(row_count) * 0.70))

        compact = [
            component
            for component in components
            if 3 <= int(component.get("size", 0) or 0) <= compact_max
        ]
        upper_components = [
            component
            for component in compact
            if int(component["row"]) <= upper_limit
        ]
        lower_components = [
            component
            for component in compact
            if int(component["row"]) >= lower_min
        ]
        upper_values = {int(component["value"]) for component in upper_components}
        lower_values = {int(component["value"]) for component in lower_components}
        middle_value_candidates = {
            int(component["value"])
            for component in compact
            if middle_min <= int(component["row"]) <= middle_max
            and int(component.get("size", 0) or 0) >= 3
            and int(component["value"]) != 0
        }
        arrangement_values = upper_values & (lower_values | middle_value_candidates)
        shared_values = upper_values & lower_values
        if len(shared_values) < 2:
            return []

        targets = [
            component
            for component in upper_components
            if int(component["value"]) in arrangement_values
        ]
        targets.sort(key=lambda component: (int(component["row"]), int(component["col"])))
        if len(targets) < 2:
            return []
        desired_values = [int(component["value"]) for component in targets[:12]]

        middle_components = [
            component
            for component in compact
            if middle_min <= int(component["row"]) <= middle_max
        ]
        neutral_slot_components = [
            component
            for component in middle_components
            if int(component["value"]) not in shared_values and int(component["value"]) != 0
        ]
        if len(neutral_slot_components) < min(len(desired_values), 2):
            return []

        grouped_slots: dict[int, list[dict[str, Any]]] = {}
        for component in neutral_slot_components:
            grouped_slots.setdefault(int(component["value"]), []).append(component)
        slot_value, slot_group = max(
            grouped_slots.items(),
            key=lambda item: (
                len(item[1]),
                -(
                    max(int(component["row"]) for component in item[1])
                    - min(int(component["row"]) for component in item[1])
                ),
                -abs(
                    (sum(int(component["row"]) for component in item[1]) / max(1, len(item[1])))
                    - (float(row_count) * 0.45)
                ),
            ),
        )
        del slot_value
        if len(slot_group) < min(len(desired_values), 2):
            return []
        base_slots = sorted(slot_group, key=lambda component: (int(component["row"]), int(component["col"])))

        def action_key(component: dict[str, Any]) -> str:
            return str(component.get("component_id", "") or component.get("anchor_key", ""))

        def clustered_centers(values: list[int], *, tolerance: int = 3) -> list[int]:
            if not values:
                return []
            clusters: list[list[int]] = []
            for value in sorted(values):
                if clusters and abs(value - clusters[-1][-1]) <= tolerance:
                    clusters[-1].append(value)
                else:
                    clusters.append([value])
            return [int(round(sum(cluster) / max(1, len(cluster)))) for cluster in clusters]

        slot_rows = clustered_centers([int(component["row"]) for component in base_slots])
        slot_gaps: list[int] = []
        row_slot_groups: list[tuple[int, list[dict[str, Any]]]] = []
        for row_center in slot_rows:
            row_slots = [
                component
                for component in base_slots
                if abs(int(component["row"]) - int(row_center)) <= 3
            ]
            if not row_slots:
                continue
            row_slot_groups.append((row_center, row_slots))
            row_cols = sorted({int(component["col"]) for component in row_slots})
            slot_gaps.extend(
                right - left
                for left, right in zip(row_cols, row_cols[1:])
                if right > left
            )
        slot_stride = 0
        if slot_gaps:
            slot_gaps.sort()
            slot_stride = int(next((gap for gap in slot_gaps if gap >= 3), slot_gaps[0]))
        col_tolerance = max(2, min(5, int(round(float(slot_stride or 6) * 0.55))))

        def aligned_with_slot_grid(component: dict[str, Any]) -> bool:
            row = int(component["row"])
            col = int(component["col"])
            for row_center, row_slots in row_slot_groups:
                if abs(row - int(row_center)) > 3:
                    continue
                row_cols = sorted({int(slot["col"]) for slot in row_slots})
                if not row_cols:
                    continue
                min_col = min(row_cols)
                max_col = max(row_cols)
                if col < min_col - col_tolerance or col > max_col + col_tolerance:
                    continue
                if slot_stride > 0:
                    nearest_col = min_col + round((col - min_col) / slot_stride) * slot_stride
                    if abs(col - nearest_col) > col_tolerance:
                        continue
                return True
            return False

        slot_by_key: dict[str, dict[str, Any]] = {}
        slot_current_values: dict[str, int | None] = {}
        for slot in base_slots:
            key = action_key(slot)
            slot_by_key[key] = slot
            slot_current_values[key] = None
        source_sizes = [
            int(component.get("size", 0) or 0)
            for component in lower_components
            if int(component["value"]) in shared_values
        ]
        normal_source_size = Counter(source_sizes).most_common(1)[0][0] if source_sizes else 0
        occupied_slot_min_size = (
            max(3, int(float(normal_source_size) * 0.85))
            if normal_source_size > 0
            else 3
        )
        for component in middle_components:
            if int(component["value"]) not in shared_values:
                continue
            if int(component.get("size", 0) or 0) < occupied_slot_min_size:
                continue
            if not aligned_with_slot_grid(component):
                continue
            key = action_key(component)
            slot_by_key[key] = component
            slot_current_values[key] = int(component["value"])

        def traversal_order(slot_components: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if len(slot_components) < 2 or slot_stride <= 0:
                return sorted(slot_components, key=lambda component: (int(component["row"]), int(component["col"])))
            rows = clustered_centers([int(component["row"]) for component in slot_components])
            row_groups_for_order: list[list[dict[str, Any]]] = []
            for row_center in rows:
                row_group = sorted(
                    [
                        component
                        for component in slot_components
                        if abs(int(component["row"]) - int(row_center)) <= 3
                    ],
                    key=lambda component: int(component["col"]),
                )
                if row_group:
                    row_groups_for_order.append(row_group)

            large_gap_threshold = max(slot_stride + col_tolerance, int(round(float(slot_stride) * 1.45)))

            def split_on_large_gaps(group: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
                if not group:
                    return []
                segments: list[list[dict[str, Any]]] = [[group[0]]]
                for left, right in zip(group, group[1:]):
                    if int(right["col"]) - int(left["col"]) > large_gap_threshold:
                        segments.append([right])
                    else:
                        segments[-1].append(right)
                return segments

            def order_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
                if not groups:
                    return []
                first = groups[0]
                if len(first) >= 2 and len(groups) > 1:
                    first_segments = split_on_large_gaps(first)
                    if len(first_segments) > 1:
                        rest_segments: list[list[dict[str, Any]]] = []
                        for group in groups[1:]:
                            rest_segments.extend(split_on_large_gaps(group))
                        if len(rest_segments) == len(first_segments) - 1:
                            ordered: list[dict[str, Any]] = []
                            for index, segment in enumerate(first_segments):
                                ordered.extend(segment)
                                if index < len(rest_segments):
                                    ordered.extend(rest_segments[index])
                            return ordered
                        return first_segments[0] + order_groups(groups[1:]) + [
                            component
                            for segment in first_segments[1:]
                            for component in segment
                        ]
                    rest_components = [
                        component
                        for group in groups[1:]
                        for component in group
                    ]
                    if rest_components:
                        first_cols = [int(component["col"]) for component in first]
                        rest_cols = [int(component["col"]) for component in rest_components]
                        if min(first_cols) < min(rest_cols) and max(rest_cols) < max(first_cols):
                            entry_col = min(rest_cols)
                            entry_index = min(
                                range(len(first)),
                                key=lambda index: abs(int(first[index]["col"]) - entry_col),
                            )
                            if abs(int(first[entry_index]["col"]) - entry_col) <= col_tolerance:
                                return (
                                    first[: entry_index + 1]
                                    + order_groups(groups[1:])
                                    + first[entry_index + 1 :]
                                )
                return first + order_groups(groups[1:])

            ordered = order_groups(row_groups_for_order)
            if len(ordered) == len(slot_components):
                return ordered
            return sorted(slot_components, key=lambda component: (int(component["row"]), int(component["col"])))

        slots = traversal_order(list(slot_by_key.values()))

        source_by_value: dict[int, list[dict[str, Any]]] = {}
        for component in lower_components:
            value = int(component["value"])
            if value not in shared_values:
                continue
            source_by_value.setdefault(value, []).append(component)
        for value in source_by_value:
            source_by_value[value].sort(key=lambda component: (-int(component["row"]), int(component["col"])))

        actions: list[PlannedAction] = []
        planned_pairs = 0

        def click(component: dict[str, Any], suffix: str) -> PlannedAction:
            return PlannedAction(
                action_name="click",
                target_component_id=str(component.get("component_id", "") or ""),
                parameters={
                    "position": {
                        "row": int(component.get("row", 0) or 0),
                        "col": int(component.get("col", 0) or 0),
                    }
                },
                reason=f"stabilize_verify_aux_visual_arrangement:{suffix}",
            )

        def component_bbox(component: dict[str, Any]) -> dict[str, int]:
            return dict(component.get("bbox") or {})

        def component_height(component: dict[str, Any]) -> int:
            bbox = component_bbox(component)
            return int(bbox.get("max_row", 0)) - int(bbox.get("min_row", 0)) + 1

        def component_width(component: dict[str, Any]) -> int:
            bbox = component_bbox(component)
            return int(bbox.get("max_col", 0)) - int(bbox.get("min_col", 0)) + 1

        frame_components = [
            component
            for component in components
            if middle_min <= int(component["row"]) <= middle_max
            and int(component.get("size", 0) or 0) >= max(24, int(float(normal_source_size or 16) * 1.8))
            and component_height(component) >= 7
            and component_width(component) >= max(8, int(float(slot_stride or 6) * 2.0))
        ]
        frame_components.sort(key=lambda component: (int(component["row"]), int(component["col"])))

        def frame_key(component: dict[str, Any]) -> str:
            return action_key(component) or (
                f"frame_{int(component.get('value', 0) or 0)}_"
                f"{int(component.get('row', 0) or 0)}_{int(component.get('col', 0) or 0)}"
            )

        def slot_inside_frame(slot: dict[str, Any], frame: dict[str, Any]) -> bool:
            bbox = component_bbox(frame)
            return (
                int(bbox.get("min_row", 0)) - 2 <= int(slot["row"]) <= int(bbox.get("max_row", 0)) + 2
                and int(bbox.get("min_col", 0)) - 2 <= int(slot["col"]) <= int(bbox.get("max_col", 0)) + 2
            )

        if frame_components:
            for component in middle_components:
                if int(component["value"]) not in arrangement_values:
                    continue
                if int(component.get("size", 0) or 0) < occupied_slot_min_size:
                    continue
                if not any(slot_inside_frame(component, frame) for frame in frame_components):
                    continue
                key = action_key(component)
                slot_by_key[key] = component
                slot_current_values[key] = int(component["value"])

        frame_slots: dict[str, list[dict[str, Any]]] = {}
        for frame in frame_components:
            grouped = [
                slot
                for slot in slot_by_key.values()
                if slot_inside_frame(slot, frame)
            ]
            grouped.sort(key=lambda component: (int(component["row"]), int(component["col"])))
            if grouped:
                frame_slots[frame_key(frame)] = grouped

        branch_source_by_value: dict[int, list[dict[str, Any]]] = {}
        branch_source_ids: set[str] = set()
        frame_values = {int(component["value"]) for component in frame_components}
        branch_size_limit = (
            max(3, int(float(normal_source_size) * 0.85))
            if normal_source_size > 0
            else 14
        )
        for component in lower_components:
            value = int(component["value"])
            if value not in frame_values:
                continue
            if int(component.get("size", 0) or 0) > branch_size_limit:
                continue
            branch_source_by_value.setdefault(value, []).append(component)
            branch_source_ids.add(action_key(component))
        for value in branch_source_by_value:
            branch_source_by_value[value].sort(
                key=lambda component: (-int(component["row"]), int(component["col"]))
            )

        if branch_source_ids:
            for value in list(source_by_value.keys()):
                source_by_value[value] = [
                    component
                    for component in source_by_value[value]
                    if action_key(component) not in branch_source_ids
                ]

        def build_branch_aware_assignments() -> dict[str, Any] | None:
            if len(frame_components) < 2 or not branch_source_by_value:
                return None
            root_key = frame_key(frame_components[0])
            if len(frame_slots.get(root_key, [])) < 2:
                return None
            frame_key_by_value: dict[int, str] = {}
            for frame in frame_components:
                key = frame_key(frame)
                if not frame_slots.get(key):
                    continue
                frame_key_by_value.setdefault(int(frame["value"]), key)
            branch_values = sorted(
                value
                for value in branch_source_by_value
                if value in frame_key_by_value
            )
            if not branch_values:
                return None
            branch_source_counts = {
                value: len(branch_source_by_value.get(value, []))
                for value in branch_values
            }

            regular_source_counts = Counter(
                int(component["value"])
                for values in source_by_value.values()
                for component in values
            )

            best: dict[str, Any] | None = None
            best_key: tuple[int, int, int] | None = None

            def validate_solution(
                frame_assignments: dict[str, tuple[tuple[str, int], ...]],
            ) -> bool:
                needed: Counter[int] = Counter()
                branch_needed: Counter[int] = Counter()
                for key, assignments_for_frame in frame_assignments.items():
                    slots_for_frame = frame_slots.get(key, [])
                    for slot, assignment in zip(slots_for_frame, assignments_for_frame):
                        kind, value = assignment
                        if kind == "branch":
                            branch_needed[int(value)] += 1
                            continue
                        current_value = slot_current_values.get(action_key(slot))
                        if current_value == int(value):
                            continue
                        needed[int(value)] += 1
                for value, count in needed.items():
                    if int(count) > int(regular_source_counts.get(int(value), 0)):
                        return False
                for value, count in branch_needed.items():
                    if int(count) > int(branch_source_counts.get(int(value), 0)):
                        return False
                return True

            def build_repeating_self_branch_cycle() -> dict[str, Any] | None:
                root_slots = frame_slots.get(root_key, [])
                if len(root_slots) < 2:
                    return None
                root_value = int(frame_components[0]["value"])
                if branch_source_counts.get(root_value, 0) <= 0:
                    return None

                root_item_count = len(root_slots) - 1
                if root_item_count <= 0:
                    return None

                for child_frame in frame_components[1:]:
                    child_key = frame_key(child_frame)
                    child_slots = frame_slots.get(child_key, [])
                    if len(child_slots) < 2:
                        continue
                    child_value = int(child_frame["value"])
                    if child_value == root_value:
                        continue
                    if branch_source_counts.get(child_value, 0) <= 0:
                        continue

                    child_item_count = len(child_slots) - 1
                    if child_item_count <= 0:
                        continue
                    unit_len = root_item_count + child_item_count
                    if unit_len <= 1 or len(desired_values) < unit_len * 2:
                        continue
                    if len(desired_values) % unit_len != 0:
                        continue

                    unit = [int(value) for value in desired_values[:unit_len]]
                    if any(
                        [int(value) for value in desired_values[index : index + unit_len]] != unit
                        for index in range(unit_len, len(desired_values), unit_len)
                    ):
                        continue

                    root_assignments = [
                        ("item", int(value))
                        for value in unit[:root_item_count]
                    ] + [("branch", child_value)]
                    child_assignments = [
                        ("item", int(value))
                        for value in unit[root_item_count:]
                    ] + [("branch", root_value)]
                    frame_assignments = {
                        root_key: tuple(root_assignments),
                        child_key: tuple(child_assignments),
                    }
                    if not validate_solution(frame_assignments):
                        continue
                    return {
                        "root_frame_key": root_key,
                        "frame_assignments": frame_assignments,
                    }
                return None

            def match_fixed_frame(
                *,
                key: str,
                target_index: int,
                frame_assignments: dict[str, tuple[tuple[str, int], ...]],
                path: tuple[str, ...],
            ) -> int | None:
                assignments_for_frame = frame_assignments.get(key)
                if assignments_for_frame is None:
                    return None
                consumed = 0
                for kind, value in assignments_for_frame:
                    if target_index + consumed >= len(desired_values):
                        return None
                    if kind == "item":
                        if int(desired_values[target_index + consumed]) != int(value):
                            return None
                        consumed += 1
                        continue
                    child_key = frame_key_by_value.get(int(value))
                    if child_key is None or child_key in path:
                        return None
                    child_consumed = match_fixed_frame(
                        key=child_key,
                        target_index=target_index + consumed,
                        frame_assignments=frame_assignments,
                        path=path + (key,),
                    )
                    if child_consumed is None:
                        return None
                    consumed += child_consumed
                return consumed

            def search_frame(
                *,
                key: str,
                target_index: int,
                frame_assignments: dict[str, tuple[tuple[str, int], ...]],
                used_branches: Counter[int],
                path: tuple[str, ...],
            ) -> list[tuple[int, dict[str, tuple[tuple[str, int], ...]], Counter[int]]]:
                if key in frame_assignments:
                    fixed_consumed = match_fixed_frame(
                        key=key,
                        target_index=target_index,
                        frame_assignments=frame_assignments,
                        path=path,
                    )
                    if fixed_consumed is None:
                        return []
                    return [(fixed_consumed, dict(frame_assignments), Counter(used_branches))]
                slots_for_frame = frame_slots.get(key, [])
                if not slots_for_frame or key in path:
                    return []
                results: list[tuple[int, dict[str, tuple[tuple[str, int], ...]], Counter[int]]] = []

                def search_slot(
                    slot_index: int,
                    current_target_index: int,
                    partial: list[tuple[str, int]],
                    current_assignments: dict[str, tuple[tuple[str, int], ...]],
                    current_used_branches: Counter[int],
                ) -> None:
                    if slot_index == len(slots_for_frame):
                        completed_assignments = dict(current_assignments)
                        completed_assignments[key] = tuple(partial)
                        results.append(
                            (
                                current_target_index - target_index,
                                completed_assignments,
                                Counter(current_used_branches),
                            )
                        )
                        return
                    if current_target_index >= len(desired_values):
                        return

                    direct_value = int(desired_values[current_target_index])
                    partial.append(("item", direct_value))
                    search_slot(
                        slot_index + 1,
                        current_target_index + 1,
                        partial,
                        current_assignments,
                        current_used_branches,
                    )
                    partial.pop()

                    for branch_value in branch_values:
                        child_key = frame_key_by_value.get(int(branch_value))
                        if child_key is None or child_key in path or child_key == key:
                            continue
                        if current_used_branches[branch_value] >= branch_source_counts.get(branch_value, 0):
                            continue
                        next_used = Counter(current_used_branches)
                        next_used[branch_value] += 1
                        child_results = search_frame(
                            key=child_key,
                            target_index=current_target_index,
                            frame_assignments=current_assignments,
                            used_branches=next_used,
                            path=path + (key,),
                        )
                        for consumed, child_assignments, child_used in child_results:
                            if consumed <= 0:
                                continue
                            partial.append(("branch", int(branch_value)))
                            search_slot(
                                slot_index + 1,
                                current_target_index + consumed,
                                partial,
                                child_assignments,
                                child_used,
                            )
                            partial.pop()

                search_slot(0, target_index, [], frame_assignments, used_branches)
                return results

            repeated_cycle = build_repeating_self_branch_cycle()
            if repeated_cycle is not None:
                return repeated_cycle

            for consumed, frame_assignments, used_branches in search_frame(
                key=root_key,
                target_index=0,
                frame_assignments={},
                used_branches=Counter(),
                path=(),
            ):
                if consumed != len(desired_values):
                    continue
                if not any(
                    kind == "branch"
                    for assignments_for_frame in frame_assignments.values()
                    for kind, _value in assignments_for_frame
                ):
                    continue
                if not validate_solution(frame_assignments):
                    continue
                branch_count = sum(
                    1
                    for assignments_for_frame in frame_assignments.values()
                    for kind, _value in assignments_for_frame
                    if kind == "branch"
                )
                source_count = sum(
                    1
                    for assignments_for_frame in frame_assignments.values()
                    for kind, _value in assignments_for_frame
                    if kind == "item"
                )
                key = (
                    -branch_count,
                    source_count,
                    sum(used_branches.values()),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = {
                        "root_frame_key": root_key,
                        "frame_assignments": dict(frame_assignments),
                    }
            return best

        def build_actions_from_assignments(assignments: dict[str, Any]) -> list[PlannedAction]:
            planned: list[PlannedAction] = []
            pair_count = 0
            regular_sources = {
                value: list(values)
                for value, values in source_by_value.items()
            }
            branch_sources = {
                value: list(values)
                for value, values in branch_source_by_value.items()
            }

            def add_assignment(
                *,
                slot: dict[str, Any],
                desired_value: int,
                source_kind: str,
            ) -> bool:
                nonlocal pair_count
                slot_key = action_key(slot)
                current_slot_value = slot_current_values.get(slot_key)
                if source_kind == "item" and current_slot_value == int(desired_value):
                    return True
                source_map = branch_sources if source_kind == "branch" else regular_sources
                candidates = source_map.get(int(desired_value), [])
                if not candidates:
                    return False
                source = candidates.pop(0)
                planned.append(click(source, f"{source_kind}_source"))
                planned.append(click(slot, f"{source_kind}_slot"))
                pair_count += 1
                if source_kind == "item" and current_slot_value in shared_values:
                    displaced_source = dict(source)
                    displaced_source["value"] = int(current_slot_value)
                    displaced_source["component_id"] = (
                        f"grid_visual_displaced_{int(current_slot_value)}_"
                        f"{int(source.get('row', 0) or 0)}_{int(source.get('col', 0) or 0)}_"
                        f"{pair_count}"
                    )
                    regular_sources.setdefault(int(current_slot_value), []).insert(0, displaced_source)
                return True

            root_frame_key = str(assignments.get("root_frame_key") or "")
            frame_assignments = dict(assignments.get("frame_assignments") or {})
            ordered_frame_keys: list[str] = []
            if root_frame_key and root_frame_key in frame_assignments:
                ordered_frame_keys.append(root_frame_key)
            for frame in frame_components:
                key = frame_key(frame)
                if key in frame_assignments and key not in ordered_frame_keys:
                    ordered_frame_keys.append(key)
            for key in ordered_frame_keys:
                slots_for_frame = frame_slots.get(key, [])
                assignments_for_frame = list(frame_assignments.get(key) or [])
                for slot, assignment in zip(slots_for_frame, assignments_for_frame):
                    kind, value = assignment
                    if not add_assignment(slot=slot, desired_value=int(value), source_kind=str(kind)):
                        return []
            if pair_count < min(len(desired_values), 2):
                return []
            planned.append(
                PlannedAction(
                    action_name=validate_action,
                    target_component_id="auxiliary_visual_arrangement",
                    parameters={},
                    reason="stabilize_verify_aux_visual_arrangement:validate",
                )
            )
            return planned

        branch_assignments = build_branch_aware_assignments()
        if branch_assignments is not None:
            branch_actions = build_actions_from_assignments(branch_assignments)
            if branch_actions:
                if record:
                    self._auxiliary_visual_arrangement_candidate_count += 1
                    self._auxiliary_visual_arrangement_plan_issued_count += 1
                    self._auxiliary_visual_arrangement_validate_count += 1
                return branch_actions

        simple_desired_values = [
            int(component["value"])
            for component in targets
            if int(component["value"]) in shared_values
        ][:8]
        if len(slots) < len(simple_desired_values):
            return []
        slots = slots[: len(simple_desired_values)]

        for desired_value, slot in zip(simple_desired_values, slots):
            slot_key = action_key(slot)
            current_slot_value = slot_current_values.get(slot_key)
            if current_slot_value == int(desired_value):
                continue
            candidates = source_by_value.get(int(desired_value), [])
            if not candidates:
                continue
            source = candidates.pop(0)
            actions.append(click(source, "source"))
            actions.append(click(slot, "slot"))
            planned_pairs += 1
            if current_slot_value in shared_values:
                displaced_source = dict(source)
                displaced_source["value"] = int(current_slot_value)
                displaced_source["component_id"] = (
                    f"grid_visual_displaced_{int(current_slot_value)}_"
                    f"{int(source.get('row', 0) or 0)}_{int(source.get('col', 0) or 0)}_"
                    f"{planned_pairs}"
                )
                source_by_value.setdefault(int(current_slot_value), []).insert(0, displaced_source)

        if planned_pairs < min(len(simple_desired_values), 2):
            return []
        if record:
            self._auxiliary_visual_arrangement_candidate_count += 1
        actions.append(
            PlannedAction(
                action_name=validate_action,
                target_component_id="auxiliary_visual_arrangement",
                parameters={},
                reason="stabilize_verify_aux_visual_arrangement:validate",
            )
        )
        if record:
            self._auxiliary_visual_arrangement_plan_issued_count += 1
            self._auxiliary_visual_arrangement_validate_count += 1
        return actions

    @staticmethod
    def _auxiliary_visual_arrangement_plan_key(actions: list[PlannedAction]) -> str:
        parts: list[str] = []
        for action in actions:
            name = str(action.action_name or "").strip().upper()
            position = dict(action.parameters.get("position") or {})
            if position:
                parts.append(
                    f"{name}:{int(position.get('row', 0) or 0)}:{int(position.get('col', 0) or 0)}"
                )
            else:
                parts.append(name)
        return "|".join(parts)

    def _select_auxiliary_visual_validate_action(
        self,
        *,
        runtime: Any,
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
    ) -> str:
        available = self._normalized_available_actions(allowed_actions, blocked_actions)
        recent = self._recent_auxiliary_action_name(runtime)
        if recent and recent in available:
            return recent
        auxiliary_actions = [
            action_name
            for action_name in available
            if _GENERIC_KEYBOARD_ACTION_RE.match(action_name)
            and action_name not in _MOVE_ACTION_SET
            and action_name not in {_CLICK_ACTION, "RESET"}
        ]
        return sorted(auxiliary_actions)[0] if auxiliary_actions else ""

    @staticmethod
    def _visual_stencil_ring_route(
        *,
        start_index: int,
        target_index: int,
        available: set[str],
    ) -> list[str]:
        if int(start_index) == int(target_index):
            return []
        ring_positions = {
            0: (0, 1),
            1: (0, 2),
            2: (1, 2),
            3: (2, 2),
            4: (2, 1),
            5: (2, 0),
            6: (1, 0),
            7: (0, 0),
        }
        index_by_position = {position: index for index, position in ring_positions.items()}
        deltas = [
            ("ACTION4", (0, 1)),
            ("ACTION2", (1, 0)),
            ("ACTION3", (0, -1)),
            ("ACTION1", (-1, 0)),
        ]
        queue: list[tuple[int, list[str]]] = [(int(start_index), [])]
        seen = {int(start_index)}
        while queue:
            current, route = queue.pop(0)
            row, col = ring_positions[current]
            for action_name, (dr, dc) in deltas:
                if action_name not in available:
                    continue
                next_position = (row + dr, col + dc)
                if next_position == (1, 1):
                    continue
                next_index = index_by_position.get(next_position)
                if next_index is None or next_index in seen:
                    continue
                next_route = route + [action_name]
                if next_index == int(target_index):
                    return next_route
                seen.add(next_index)
                queue.append((next_index, next_route))
        return []

    def _build_visual_stencil_first_effect_actions(
        self,
        *,
        runtime: Any,
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None = None,
    ) -> list[PlannedAction]:
        available = set(self._normalized_available_actions(allowed_actions, blocked_actions))
        if "ACTION5" not in available:
            return []
        raw_rows = getattr(runtime, "last_board_rows", None)
        grid = self._coerce_board_value_grid(raw_rows)
        if not grid:
            return []
        row_count = len(grid)
        col_count = min((len(row) for row in grid), default=0)
        if row_count < 24 or col_count < 24:
            return []
        components = self._compact_grid_components_from_rows(raw_rows)
        if not components:
            return []

        def bbox(component: dict[str, Any]) -> dict[str, int]:
            return dict(component.get("bbox") or {})

        def overlaps(a: dict[str, int], b: dict[str, int]) -> bool:
            return not (
                int(a.get("max_row", -1)) < int(b.get("min_row", 0))
                or int(b.get("max_row", -1)) < int(a.get("min_row", 0))
                or int(a.get("max_col", -1)) < int(b.get("min_col", 0))
                or int(b.get("max_col", -1)) < int(a.get("min_col", 0))
            )

        canvas_candidates: list[dict[str, Any]] = []
        for component in components:
            b = bbox(component)
            height = int(b.get("max_row", 0)) - int(b.get("min_row", 0)) + 1
            width = int(b.get("max_col", 0)) - int(b.get("min_col", 0)) + 1
            if (
                8 <= height <= 12
                and 8 <= width <= 12
                and int(component.get("size", 0) or 0) >= 64
                and int(component.get("row", 0) or 0) >= int(float(row_count) * 0.35)
            ):
                canvas_candidates.append(component)
        if not canvas_candidates:
            return []
        canvas = max(
            canvas_candidates,
            key=lambda component: (
                int(component.get("size", 0) or 0),
                int(component.get("row", 0) or 0),
            ),
        )
        canvas_bbox = bbox(canvas)
        canvas_value = int(canvas.get("value", 0) or 0)
        canvas_row = int(canvas.get("row", 0) or 0)
        canvas_col = int(canvas.get("col", 0) or 0)

        def common_value(cells: list[int]) -> tuple[int, float]:
            if not cells:
                return 0, 0.0
            value, count = Counter(cells).most_common(1)[0]
            return int(value), float(count) / float(len(cells))

        target: dict[str, Any] | None = None
        max_target_row = max(0, min(row_count - 10, int(canvas_bbox.get("min_row", 0)) - 6))
        for start_row in range(0, max_target_row + 1):
            for start_col in range(0, col_count - 9):
                window_bbox = {
                    "min_row": start_row,
                    "max_row": start_row + 9,
                    "min_col": start_col,
                    "max_col": start_col + 9,
                }
                if overlaps(window_bbox, canvas_bbox):
                    continue
                window = [row[start_col : start_col + 10] for row in grid[start_row : start_row + 10]]
                halves = [
                    (
                        "top",
                        [cell for row in window[:5] for cell in row],
                        "bottom",
                        [cell for row in window[5:] for cell in row],
                    ),
                    (
                        "left",
                        [cell for row in window for cell in row[:5]],
                        "right",
                        [cell for row in window for cell in row[5:]],
                    ),
                ]
                for first_region, first_cells, second_region, second_cells in halves:
                    first_value, first_purity = common_value(first_cells)
                    second_value, second_purity = common_value(second_cells)
                    if (
                        first_purity < 0.9
                        or second_purity < 0.9
                        or first_value == second_value
                    ):
                        continue
                    fill_region = ""
                    fill_value = 0
                    if first_value == canvas_value and second_value != canvas_value:
                        fill_region = second_region
                        fill_value = second_value
                    elif second_value == canvas_value and first_value != canvas_value:
                        fill_region = first_region
                        fill_value = first_value
                    if not fill_region:
                        continue
                    score = (
                        first_purity
                        + second_purity
                        + (0.1 if start_row <= int(float(row_count) * 0.25) else 0.0)
                    )
                    candidate = {
                        "bbox": window_bbox,
                        "region": fill_region,
                        "value": int(fill_value),
                        "score": float(score),
                    }
                    if target is None or float(candidate["score"]) > float(target["score"]):
                        target = candidate
        if target is None:
            return []

        target_bbox = dict(target["bbox"])
        fill_value = int(target["value"])
        brush_candidates: list[dict[str, Any]] = []
        for component in components:
            component_value = component.get("value", -1)
            if int(component_value if component_value is not None else -1) != fill_value:
                continue
            if int(component.get("size", 0) or 0) < 20:
                continue
            b = bbox(component)
            if overlaps(b, canvas_bbox) or overlaps(b, target_bbox):
                continue
            row = int(component.get("row", 0) or 0)
            col = int(component.get("col", 0) or 0)
            distance = abs(row - canvas_row) + abs(col - canvas_col)
            if distance > max(row_count, col_count) // 2:
                continue
            enriched = dict(component)
            enriched["_distance"] = int(distance)
            brush_candidates.append(enriched)
        if not brush_candidates:
            return []
        brush = min(
            brush_candidates,
            key=lambda component: (
                int(component.get("_distance", 10_000) or 10_000),
                -int(component.get("size", 0) or 0),
            ),
        )
        brush_row = int(brush.get("row", 0) or 0)
        brush_col = int(brush.get("col", 0) or 0)
        delta_row = brush_row - canvas_row
        delta_col = brush_col - canvas_col
        if abs(delta_row) >= abs(delta_col):
            current_region = "bottom" if delta_row > 0 else "top"
        else:
            current_region = "right" if delta_col > 0 else "left"

        region_to_index = {"top": 0, "right": 2, "bottom": 4, "left": 6}
        target_region = str(target["region"])
        if current_region not in region_to_index or target_region not in region_to_index:
            return []
        route = self._visual_stencil_ring_route(
            start_index=region_to_index[current_region],
            target_index=region_to_index[target_region],
            available=available,
        )
        if region_to_index[current_region] != region_to_index[target_region] and not route:
            return []
        route = route + ["ACTION5"]
        if any(action_name not in available for action_name in route):
            return []
        return [
            PlannedAction(
                action_name=action_name,
                reason=f"visual_stencil_first_effect:{target_region}",
            )
            for action_name in route
        ]

    def _build_visual_stencil_first_effect_direct_plan(
        self,
        *,
        runtime: Any,
        step_index: int,
        agent: Any | None,
        semantic_action_lease: dict[str, Any],
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
    ) -> tuple[list[PlannedAction], ScoredCandidate] | None:
        level_number = self._runtime_level_number(runtime)
        score = self._runtime_score(runtime)
        if score != 0:
            return None
        if (
            int(step_index) <= int(self._visual_stencil_first_effect_active_until_step)
            and int(level_number) <= int(self._visual_stencil_first_effect_active_level)
            and int(score) <= int(self._visual_stencil_first_effect_active_score)
        ):
            return None
        planned = self._build_visual_stencil_first_effect_actions(
            runtime=runtime,
            allowed_actions=allowed_actions,
            blocked_actions=blocked_actions,
        )
        if not planned:
            return None
        plan_key = self._auxiliary_visual_arrangement_plan_key(planned)
        keyed = f"{int(level_number)}|{plan_key}"
        if keyed in self._visual_stencil_first_effect_plan_keys:
            self._visual_stencil_first_effect_duplicate_suppressed_count += 1
            return None
        scored = ScoredCandidate(
            component_id="visual_stencil_first_effect",
            action_family="auxiliary",
            source="visual_stencil_first_effect",
            action_name=str(planned[0].action_name or "").strip().upper(),
            anchor_key=f"visual_stencil_first_effect||{int(level_number)}||0",
            center_row=0,
            center_col=0,
            base_admission_score=2.7,
            resolver_score=2.7,
            selected=True,
            selection_reason="visual_stencil_first_effect",
        )
        normalized = self._normalize_planned_actions(
            runtime=runtime,
            actions=planned,
            selected_target=scored,
            agent=agent,
            semantic_action_lease=semantic_action_lease,
        )
        if not normalized:
            return None
        self._visual_stencil_first_effect_plan_keys.add(keyed)
        self._visual_stencil_first_effect_candidate_count += 1
        self._visual_stencil_first_effect_plan_issued_count += 1
        self._visual_stencil_first_effect_active_until_step = int(step_index) + len(normalized) + 2
        self._visual_stencil_first_effect_active_level = int(level_number)
        self._visual_stencil_first_effect_active_score = int(score)
        return normalized, scored

    @staticmethod
    def _visual_stencil_mask(index: int) -> list[list[bool]]:
        mask = [[False for _col in range(10)] for _row in range(10)]
        for row in range(10):
            for col in range(10):
                if index == 0:
                    mask[row][col] = row < 5
                elif index == 4:
                    mask[row][col] = row >= 5
                elif index == 6:
                    mask[row][col] = col < 5
                elif index == 2:
                    mask[row][col] = col >= 5
                elif index == 7:
                    mask[row][col] = col <= 9 - row
                elif index == 3:
                    mask[row][col] = col >= 9 - row
                elif index == 1:
                    mask[row][col] = col >= row
                elif index == 5:
                    mask[row][col] = col <= row
        return mask

    @staticmethod
    def _visual_stencil_apply(
        canvas: list[list[int]],
        *,
        index: int,
        color: int,
    ) -> list[list[int]]:
        painted = [list(row) for row in canvas]
        mask = LocalController._visual_stencil_mask(index)
        for row in range(10):
            for col in range(10):
                if mask[row][col]:
                    painted[row][col] = int(color)
        return painted

    @staticmethod
    def _visual_stencil_arrow_mask(index: int) -> list[list[bool]]:
        mask = [[False for _col in range(10)] for _row in range(10)]
        if index == 0:
            rows = range(0, 3)
            cols = range(3, 7)
        elif index == 4:
            rows = range(7, 10)
            cols = range(3, 7)
        elif index == 6:
            rows = range(3, 7)
            cols = range(0, 3)
        elif index == 2:
            rows = range(3, 7)
            cols = range(7, 10)
        else:
            return mask
        for row in rows:
            for col in cols:
                mask[row][col] = True
        return mask

    @staticmethod
    def _visual_stencil_arrow_position(
        *,
        canvas_bbox: dict[str, int],
        index: int,
    ) -> tuple[int, int] | None:
        min_row = int(canvas_bbox.get("min_row", 0) or 0)
        max_row = int(canvas_bbox.get("max_row", 0) or 0)
        min_col = int(canvas_bbox.get("min_col", 0) or 0)
        max_col = int(canvas_bbox.get("max_col", 0) or 0)
        mid_row = (min_row + max_row) // 2
        mid_col = (min_col + max_col) // 2
        if int(index) == 0:
            return min_row - 14, mid_col
        if int(index) == 4:
            return max_row + 14, mid_col
        if int(index) == 6:
            return mid_row, min_col - 14
        if int(index) == 2:
            return mid_row, max_col + 14
        return None

    @staticmethod
    def _visual_stencil_mismatch(
        candidate: list[list[int]],
        target: list[list[int]],
    ) -> int:
        mismatch = 0
        for row in range(10):
            for col in range(10):
                if int(candidate[row][col]) != int(target[row][col]):
                    mismatch += 1
        return mismatch

    @staticmethod
    def _visual_stencil_infer_index(
        *,
        brush_row: int,
        brush_col: int,
        canvas_row: int,
        canvas_col: int,
    ) -> int:
        delta_row = int(brush_row) - int(canvas_row)
        delta_col = int(brush_col) - int(canvas_col)
        if abs(delta_row) >= abs(delta_col) * 2:
            return 4 if delta_row > 0 else 0
        if abs(delta_col) >= abs(delta_row) * 2:
            return 2 if delta_col > 0 else 6
        if delta_row < 0 and delta_col > 0:
            return 1
        if delta_row > 0 and delta_col > 0:
            return 3
        if delta_row > 0 and delta_col < 0:
            return 5
        return 7

    def _visual_stencil_composition_context(
        self,
        *,
        runtime: Any,
    ) -> dict[str, Any] | None:
        raw_rows = getattr(runtime, "last_board_rows", None)
        grid = self._coerce_board_value_grid(raw_rows)
        if not grid:
            return None
        row_count = len(grid)
        col_count = min((len(row) for row in grid), default=0)
        if row_count < 24 or col_count < 24:
            return None
        values = [int(cell) for row in grid for cell in row[:col_count]]
        background = Counter(values).most_common(1)[0][0] if values else 0
        components = self._compact_grid_components_from_rows(raw_rows)
        if not components:
            return None

        def bbox(component: dict[str, Any]) -> dict[str, int]:
            return dict(component.get("bbox") or {})

        def overlaps(a: dict[str, int], b: dict[str, int]) -> bool:
            return not (
                int(a.get("max_row", -1)) < int(b.get("min_row", 0))
                or int(b.get("max_row", -1)) < int(a.get("min_row", 0))
                or int(a.get("max_col", -1)) < int(b.get("min_col", 0))
                or int(b.get("max_col", -1)) < int(a.get("min_col", 0))
            )

        canvas_candidates: list[dict[str, Any]] = []
        for component in components:
            b = bbox(component)
            height = int(b.get("max_row", 0)) - int(b.get("min_row", 0)) + 1
            width = int(b.get("max_col", 0)) - int(b.get("min_col", 0)) + 1
            if (
                height == 10
                and width == 10
                and int(component.get("size", 0) or 0) >= 80
                and int(component.get("row", 0) or 0) >= int(float(row_count) * 0.35)
            ):
                canvas_candidates.append(component)
        if not canvas_candidates:
            return None
        canvas = max(
            canvas_candidates,
            key=lambda component: (
                int(component.get("size", 0) or 0),
                int(component.get("row", 0) or 0),
            ),
        )
        canvas_bbox = bbox(canvas)
        canvas_value = int(canvas.get("value", 0) or 0)
        canvas_row = int(canvas.get("row", 0) or 0)
        canvas_col = int(canvas.get("col", 0) or 0)
        canvas_matrix = [
            list(row[int(canvas_bbox["min_col"]) : int(canvas_bbox["max_col"]) + 1])
            for row in grid[int(canvas_bbox["min_row"]) : int(canvas_bbox["max_row"]) + 1]
        ]
        if len(canvas_matrix) != 10 or any(len(row) != 10 for row in canvas_matrix):
            return None

        target: dict[str, Any] | None = None
        max_target_row = max(0, min(row_count - 10, int(canvas_bbox.get("min_row", 0)) - 8))
        for start_row in range(0, max_target_row + 1):
            for start_col in range(0, col_count - 9):
                window_bbox = {
                    "min_row": start_row,
                    "max_row": start_row + 9,
                    "min_col": start_col,
                    "max_col": start_col + 9,
                }
                if overlaps(window_bbox, canvas_bbox):
                    continue
                window = [row[start_col : start_col + 10] for row in grid[start_row : start_row + 10]]
                cells = [int(cell) for row in window for cell in row]
                if len(cells) != 100:
                    continue
                non_background_ratio = (
                    sum(1 for cell in cells if cell != int(background)) / float(len(cells))
                )
                distinct = set(cells)
                if (
                    non_background_ratio < 0.95
                    or len(distinct) < 2
                    or len(distinct) > 6
                ):
                    continue
                score = (
                    non_background_ratio
                    + (0.2 if start_row <= int(float(row_count) * 0.25) else 0.0)
                    - (float(start_row) * 0.001)
                )
                candidate = {
                    "bbox": window_bbox,
                    "matrix": window,
                    "score": float(score),
                }
                if target is None or float(candidate["score"]) > float(target["score"]):
                    target = candidate
        if target is None:
            return None
        target_bbox = dict(target["bbox"])
        target_matrix = [list(row) for row in list(target["matrix"])]

        brush_candidates: list[dict[str, Any]] = []
        for component in components:
            if int(component.get("size", 0) or 0) < 20:
                continue
            b = bbox(component)
            if overlaps(b, canvas_bbox) or overlaps(b, target_bbox):
                continue
            row = int(component.get("row", 0) or 0)
            col = int(component.get("col", 0) or 0)
            distance = abs(row - canvas_row) + abs(col - canvas_col)
            if distance > max(row_count, col_count) // 2:
                continue
            enriched = dict(component)
            enriched["_distance"] = int(distance)
            brush_candidates.append(enriched)
        if not brush_candidates:
            return None
        brush = min(
            brush_candidates,
            key=lambda component: (
                int(component.get("_distance", 10_000) or 10_000),
                -int(component.get("size", 0) or 0),
            ),
        )
        brush_row = int(brush.get("row", 0) or 0)
        brush_col = int(brush.get("col", 0) or 0)
        current_index = self._visual_stencil_infer_index(
            brush_row=brush_row,
            brush_col=brush_col,
            canvas_row=canvas_row,
            canvas_col=canvas_col,
        )
        current_color = int(brush.get("value", 0) or 0)

        swatches: dict[int, tuple[int, int]] = {}
        for component in components:
            b = bbox(component)
            height = int(b.get("max_row", 0)) - int(b.get("min_row", 0)) + 1
            width = int(b.get("max_col", 0)) - int(b.get("min_col", 0)) + 1
            if (
                int(component.get("size", 0) or 0) < 5
                or int(component.get("size", 0) or 0) > 25
                or height > 5
                or width > 5
                or int(component.get("row", 0) or 0) >= int(canvas_bbox.get("min_row", 0))
                or overlaps(b, target_bbox)
            ):
                continue
            value = int(component.get("value", 0) or 0)
            existing = swatches.get(value)
            row = int(component.get("row", 0) or 0)
            col = int(component.get("col", 0) or 0)
            if existing is None or row < existing[0]:
                swatches[value] = (row, col)

        return {
            "available_grid": grid,
            "canvas": canvas_matrix,
            "target": target_matrix,
            "canvas_bbox": canvas_bbox,
            "target_bbox": target_bbox,
            "canvas_value": canvas_value,
            "current_index": current_index,
            "current_color": current_color,
            "swatches": swatches,
        }

    def _build_visual_stencil_composition_actions(
        self,
        *,
        runtime: Any,
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None = None,
    ) -> list[PlannedAction]:
        available = set(self._normalized_available_actions(allowed_actions, blocked_actions))
        if "ACTION5" not in available:
            return []
        context = self._visual_stencil_composition_context(runtime=runtime)
        if not context:
            return []
        canvas = [list(row) for row in list(context["canvas"])]
        target = [list(row) for row in list(context["target"])]
        if self._visual_stencil_mismatch(canvas, target) == 0:
            return []
        current_index = int(context["current_index"])
        current_color = int(context["current_color"])
        swatches = dict(context.get("swatches") or {})
        colors = sorted({int(cell) for row in target for cell in row})
        operation_options: list[dict[str, Any]] = []
        for color in colors:
            for index in range(8):
                operation_options.append(
                    {
                        "kind": "paint",
                        "index": int(index),
                        "color": int(color),
                        "mask": self._visual_stencil_mask(int(index)),
                    }
                )
            for index in (0, 2, 4, 6):
                operation_options.append(
                    {
                        "kind": "arrow",
                        "index": int(index),
                        "color": int(color),
                        "mask": self._visual_stencil_arrow_mask(int(index)),
                    }
                )

        def apply_operation(state: tuple[int, ...], operation: dict[str, Any]) -> tuple[int, ...]:
            painted = list(state)
            mask = list(operation.get("mask") or [])
            color = int(operation.get("color", 0) or 0)
            for row in range(10):
                for col in range(10):
                    if bool(mask[row][col]):
                        painted[row * 10 + col] = color
            return tuple(painted)

        def sequence_cost(
            sequence: list[dict[str, Any]]
        ) -> tuple[int, list[list[str]]] | None:
            index = current_index
            color = current_color
            routes: list[list[str]] = []
            cost = 0
            for operation in sequence:
                next_index = int(operation.get("index", 0) or 0)
                next_color = int(operation.get("color", 0) or 0)
                if int(next_color) != int(color):
                    if int(next_color) not in swatches:
                        return None
                    cost += 1
                route = self._visual_stencil_ring_route(
                    start_index=index,
                    target_index=int(next_index),
                    available=available,
                )
                if int(index) != int(next_index) and not route:
                    return None
                cost += len(route) + 1
                routes.append(route)
                index = int(next_index)
                color = int(next_color)
            return cost, routes

        start_state = tuple(int(cell) for row in canvas for cell in row)
        target_state = tuple(int(cell) for row in target for cell in row)
        best_sequence: list[dict[str, Any]] = []
        best_routes: list[list[str]] = []
        best_key: tuple[int, int, int] | None = None
        frontier: dict[tuple[int, ...], list[dict[str, Any]]] = {start_state: []}
        seen_states: set[tuple[int, ...]] = {start_state}
        for sequence_length in range(1, 5):
            next_frontier: dict[tuple[int, ...], list[dict[str, Any]]] = {}
            found_at_depth: list[tuple[tuple[int, int, int], list[dict[str, Any]], list[list[str]]]] = []
            for state, path in frontier.items():
                for operation in operation_options:
                    next_state = apply_operation(state, operation)
                    if next_state in seen_states:
                        continue
                    sequence = path + [operation]
                    seen_states.add(next_state)
                    next_frontier[next_state] = sequence
                    if next_state != target_state:
                        continue
                    cost_result = sequence_cost(sequence)
                    if cost_result is None:
                        continue
                    cost, routes = cost_result
                    key = (
                        sequence_length,
                        int(cost),
                        sum(int(op.get("index", 0) or 0) for op in sequence),
                    )
                    found_at_depth.append((key, sequence, routes))
            if found_at_depth:
                key, sequence, routes = min(found_at_depth, key=lambda item: item[0])
                best_key = key
                best_sequence = sequence
                best_routes = routes
                break
            frontier = next_frontier
        if not best_sequence or best_key is None:
            return []

        canvas_bbox = dict(context.get("canvas_bbox") or {})
        actions: list[PlannedAction] = []
        color = current_color
        for operation, route in zip(best_sequence, best_routes):
            index = int(operation.get("index", 0) or 0)
            next_color = int(operation.get("color", 0) or 0)
            if int(next_color) != int(color):
                swatch = swatches.get(int(next_color))
                if swatch is None:
                    return []
                actions.append(
                    PlannedAction(
                        action_name="click",
                        parameters={"position": {"row": int(swatch[0]), "col": int(swatch[1])}},
                        reason=f"visual_stencil_composition:select:{int(next_color)}",
                    )
                )
                color = int(next_color)
            for action_name in route:
                actions.append(
                    PlannedAction(
                        action_name=action_name,
                        reason=f"visual_stencil_composition:route:{int(index)}",
                    )
                )
            if str(operation.get("kind", "") or "") == "arrow":
                position = self._visual_stencil_arrow_position(
                    canvas_bbox=canvas_bbox,
                    index=int(index),
                )
                if position is None:
                    return []
                actions.append(
                    PlannedAction(
                        action_name="click",
                        parameters={
                            "position": {"row": int(position[0]), "col": int(position[1])}
                        },
                        reason=f"visual_stencil_composition:arrow:{int(index)}",
                    )
                )
            else:
                actions.append(
                    PlannedAction(
                        action_name="ACTION5",
                        reason=f"visual_stencil_composition:paint:{int(index)}",
                    )
                )
        return actions

    def _build_visual_stencil_composition_direct_plan(
        self,
        *,
        runtime: Any,
        step_index: int,
        agent: Any | None,
        semantic_action_lease: dict[str, Any],
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
    ) -> tuple[list[PlannedAction], ScoredCandidate] | None:
        level_number = self._runtime_level_number(runtime)
        score = self._runtime_score(runtime)
        if score <= 0:
            return None
        if (
            int(step_index) <= int(self._visual_stencil_composition_active_until_step)
            and int(level_number) <= int(self._visual_stencil_composition_active_level)
            and int(score) <= int(self._visual_stencil_composition_active_score)
        ):
            return None
        planned = self._build_visual_stencil_composition_actions(
            runtime=runtime,
            allowed_actions=allowed_actions,
            blocked_actions=blocked_actions,
        )
        if not planned:
            self._visual_stencil_composition_active_until_step = int(step_index) + 8
            self._visual_stencil_composition_active_level = int(level_number)
            self._visual_stencil_composition_active_score = int(score)
            return None
        plan_key = self._auxiliary_visual_arrangement_plan_key(planned)
        keyed = f"{int(level_number)}|{plan_key}"
        if keyed in self._visual_stencil_composition_plan_keys:
            self._visual_stencil_composition_duplicate_suppressed_count += 1
            return None
        first_action = str(planned[0].action_name or "").strip().upper()
        scored = ScoredCandidate(
            component_id="visual_stencil_composition",
            action_family="auxiliary",
            source="visual_stencil_composition",
            action_name=first_action if first_action else "ACTION5",
            anchor_key=f"visual_stencil_composition||{int(level_number)}||0",
            center_row=0,
            center_col=0,
            base_admission_score=2.8,
            resolver_score=2.8,
            selected=True,
            selection_reason="visual_stencil_composition",
        )
        normalized = self._normalize_planned_actions(
            runtime=runtime,
            actions=planned,
            selected_target=scored,
            agent=agent,
            semantic_action_lease=semantic_action_lease,
        )
        if not normalized:
            return None
        self._visual_stencil_composition_plan_keys.add(keyed)
        self._visual_stencil_composition_candidate_count += 1
        self._visual_stencil_composition_plan_issued_count += 1
        self._visual_stencil_composition_active_until_step = int(step_index) + len(normalized) + 2
        self._visual_stencil_composition_active_level = int(level_number)
        self._visual_stencil_composition_active_score = int(score)
        return normalized, scored

    def _build_auxiliary_visual_arrangement_direct_plan(
        self,
        *,
        runtime: Any,
        step_index: int,
        agent: Any | None,
        semantic_action_lease: dict[str, Any],
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
    ) -> tuple[list[PlannedAction], ScoredCandidate] | None:
        level_number = self._runtime_level_number(runtime)
        score = self._runtime_score(runtime)
        if (
            int(step_index) <= int(self._auxiliary_visual_arrangement_active_until_step)
            and int(level_number) <= int(self._auxiliary_visual_arrangement_active_level)
            and int(score) <= int(self._auxiliary_visual_arrangement_active_score)
        ):
            return None
        available = self._normalized_available_actions(allowed_actions, blocked_actions)
        if _CLICK_ACTION not in available:
            return None
        validate_action = self._select_auxiliary_visual_validate_action(
            runtime=runtime,
            allowed_actions=allowed_actions,
            blocked_actions=blocked_actions,
        )
        if not validate_action:
            return None
        planned = self._build_auxiliary_visual_arrangement_actions(
            runtime=runtime,
            validate_action_name=validate_action,
            record=False,
        )
        if not planned:
            return None
        plan_key = self._auxiliary_visual_arrangement_plan_key(planned)
        level_number = self._runtime_level_number(runtime)
        keyed = f"{int(level_number)}|{plan_key}"
        if keyed in self._auxiliary_visual_arrangement_plan_keys:
            self._auxiliary_visual_arrangement_duplicate_suppressed_count += 1
            return None
        scored = ScoredCandidate(
            component_id="auxiliary_visual_arrangement",
            action_family="click",
            source="auxiliary_visual_arrangement",
            action_name=_CLICK_ACTION,
            anchor_key=f"auxiliary_visual_arrangement||{int(level_number)}||0",
            center_row=0,
            center_col=0,
            base_admission_score=2.6,
            resolver_score=2.6,
            selected=True,
            selection_reason="auxiliary_visual_arrangement",
        )
        normalized = self._normalize_planned_actions(
            runtime=runtime,
            actions=planned,
            selected_target=scored,
            agent=agent,
            semantic_action_lease=semantic_action_lease,
        )
        if not normalized:
            return None
        self._auxiliary_visual_arrangement_plan_keys.add(keyed)
        self._auxiliary_visual_arrangement_candidate_count += 1
        self._auxiliary_visual_arrangement_plan_issued_count += 1
        self._auxiliary_visual_arrangement_validate_count += 1
        self._auxiliary_visual_arrangement_direct_issued_count += 1
        self._auxiliary_visual_arrangement_active_until_step = int(step_index) + len(normalized) + 2
        self._auxiliary_visual_arrangement_active_level = int(level_number)
        self._auxiliary_visual_arrangement_active_score = int(score)
        return normalized, scored

    def _recent_unsettled_auxiliary_action(
        self,
        runtime: Any,
        *,
        history_limit: int = 6,
        max_changed_ticks: int = 18,
    ) -> str:
        state = getattr(runtime, "last_state", None)
        recent_history = list(getattr(state, "recent_history", None) or [])
        if not recent_history:
            return ""
        changed_ticks = 0
        for step in reversed(recent_history[-max(1, int(history_limit or 1)) :]):
            if not isinstance(step, dict):
                continue
            action_name = self._normalize_history_action_name(step.get("action"))
            if not action_name:
                continue
            if action_name in _MOVE_ACTION_SET or action_name in {_CLICK_ACTION, "RESET"}:
                return ""
            if bool(step.get("no_state_change", False)):
                return ""
            changed_ticks += 1
            if changed_ticks >= max(1, int(max_changed_ticks or 1)):
                return ""
            return action_name
        return ""

    def _recent_auxiliary_board_delta_points(
        self,
        runtime: Any,
        *,
        limit: int = 32,
        history_limit: int = 8,
    ) -> list[dict[str, Any]]:
        current_rows = self._runtime_board_rows(runtime)
        if not current_rows:
            current_rows = self._coerce_board_rows(getattr(runtime, "last_board_rows", None))
        if not current_rows:
            return []
        state = getattr(runtime, "last_state", None)
        recent_history = list(getattr(state, "recent_history", None) or [])
        if not recent_history:
            return []

        for step in reversed(recent_history[-max(1, int(history_limit or 1)) :]):
            if not isinstance(step, dict):
                continue
            action_name = self._normalize_history_action_name(step.get("action"))
            if not action_name or action_name in _MOVE_ACTION_SET or action_name in {_CLICK_ACTION, "RESET"}:
                continue
            if bool(step.get("no_state_change", False)):
                continue
            previous_rows = self._coerce_board_rows(step.get("grid_raw") or step.get("grid") or [])
            if not previous_rows:
                continue
            after_rows = self._coerce_board_rows(
                step.get("post_grid_raw")
                or step.get("grid_after")
                or step.get("board_after")
                or []
            )
            if not after_rows:
                after_rows = current_rows
            row_limit = min(len(previous_rows), len(after_rows))
            delta_points: list[dict[str, Any]] = []
            for row_idx in range(row_limit):
                previous_row = str(previous_rows[row_idx] or "")
                current_row = str(after_rows[row_idx] or "")
                col_limit = min(len(previous_row), len(current_row))
                for col_idx in range(col_limit):
                    if previous_row[col_idx] == current_row[col_idx]:
                        continue
                    raw_value = current_row[col_idx]
                    try:
                        value = int(raw_value)
                    except (TypeError, ValueError):
                        value = 0
                    delta_points.append(
                        {
                            "component_id": f"aux_delta_{row_idx}_{col_idx}",
                            "anchor_key": f"delta||{row_idx}||{col_idx}",
                            "row": int(row_idx),
                            "col": int(col_idx),
                            "value": value,
                            "size": 1,
                            "delta_anchor": True,
                        }
                    )
                    if len(delta_points) >= max(1, int(limit or 1)):
                        return delta_points
            if delta_points:
                return delta_points
        return []

    def _build_auxiliary_click_pair_actions(
        self,
        *,
        runtime: Any,
        candidate: StructuredCandidate,
    ) -> list[PlannedAction]:
        seed_row = int(getattr(candidate, "center_row", 0) or 0)
        seed_col = int(getattr(candidate, "center_col", 0) or 0)
        if (not seed_row and not seed_col) and str(getattr(candidate, "anchor_key", "") or ""):
            parsed = self._parse_anchor_key_position(runtime, str(getattr(candidate, "anchor_key", "") or ""))
            if parsed is not None:
                seed_row, seed_col = parsed
        if not (seed_row or seed_col):
            return []

        rows = list(getattr(runtime, "last_board_rows", None) or [])
        row_count = len(rows)
        col_count = max((len(row) for row in rows), default=0)
        if row_count <= 0:
            state = getattr(runtime, "last_state", None)
            object_rows = [
                int((dict(item.get("center") or {}).get("row", 0) if isinstance(item, dict) else 0) or 0)
                for item in list(getattr(state, "object_table", None) or [])
                if isinstance(item, dict)
            ]
            row_count = max(object_rows, default=max(seed_row + 1, 64)) + 1
        lower_cutoff = max(int(float(row_count) * 0.55), seed_row + 2)
        upper_cutoff = max(2, int(float(row_count) * 0.18))

        state = getattr(runtime, "last_state", None)
        points: list[dict[str, Any]] = []
        for entry in list(getattr(state, "object_table", None) or []):
            if not isinstance(entry, dict):
                continue
            component_id = str(entry.get("component_id", "") or "").strip()
            if not component_id or component_id.lower().startswith(("auxiliary::", "anchor::auxiliary::")):
                continue
            center = dict(entry.get("center") or {})
            if center:
                row = int(center.get("row", 0) or 0)
                col = int(center.get("col", 0) or 0)
            else:
                bbox = dict(entry.get("bbox") or {})
                min_row = int(bbox.get("min_row", 0) or 0)
                max_row = int(bbox.get("max_row", min_row) or min_row)
                min_col = int(bbox.get("min_col", 0) or 0)
                max_col = int(bbox.get("max_col", min_col) or min_col)
                row = (min_row + max_row) // 2
                col = (min_col + max_col) // 2
            if row <= 0 or col <= 0:
                continue
            anchor_key = str(entry.get("anchor_key", "") or "")
            if not anchor_key:
                level_number = int(getattr(state, "level_number", 0) or 0)
                anchor_key = f"{level_number}||{row}||{col}"
            try:
                value = int(entry.get("value", 0) or 0)
            except (TypeError, ValueError):
                value = 0
            try:
                size = int(entry.get("size", 0) or 0)
            except (TypeError, ValueError):
                size = 0
            points.append(
                {
                    "component_id": component_id,
                    "anchor_key": anchor_key,
                    "row": int(row),
                    "col": int(col),
                    "value": value,
                    "size": size,
                }
            )

        def same_point(left: dict[str, Any], right: dict[str, Any]) -> bool:
            left_component = str(left.get("component_id", "") or "").strip()
            right_component = str(right.get("component_id", "") or "").strip()
            if left_component and right_component and left_component == right_component:
                return True
            left_anchor = str(left.get("anchor_key", "") or "").strip()
            right_anchor = str(right.get("anchor_key", "") or "").strip()
            if left_anchor and right_anchor and left_anchor == right_anchor:
                return True
            return int(left.get("row", 0) or 0) == int(right.get("row", 0) or 0) and int(
                left.get("col", 0) or 0
            ) == int(right.get("col", 0) or 0)

        def recent_click_board_change(point: dict[str, Any]) -> bool:
            entry = self._runtime_local_effect_entry(
                runtime=runtime,
                component_id=str(point.get("component_id", "") or ""),
                anchor_key=str(point.get("anchor_key", "") or ""),
            )
            if not entry:
                return False
            if str(entry.get("last_family", "") or "").strip().lower() != "click":
                return False
            current_step = int(getattr(state, "step_index", 0) or 0)
            last_step = int(entry.get("last_step_idx", -10_000) or -10_000)
            if current_step - last_step > 2:
                return False
            return (
                int(entry.get("board_change_no_score_count", 0) or 0) > 0
                or int(entry.get("structural_delta_count", 0) or 0) > 0
                or int(entry.get("progress_effect_count", 0) or 0) > 0
            )

        def point_key(point: dict[str, Any]) -> str:
            component = str(point.get("component_id", "") or "").strip()
            if component:
                return f"component:{component}"
            anchor = str(point.get("anchor_key", "") or "").strip()
            if anchor:
                return f"anchor:{anchor}"
            return f"pos:{int(point.get('row', 0) or 0)}:{int(point.get('col', 0) or 0)}"

        def click_pair_key(source: dict[str, Any], target: dict[str, Any]) -> str:
            return f"{point_key(source)}->{point_key(target)}"

        def point_effect_score(point: dict[str, Any]) -> float:
            entry = self._runtime_local_effect_entry(
                runtime=runtime,
                component_id=str(point.get("component_id", "") or ""),
                anchor_key=str(point.get("anchor_key", "") or ""),
            )
            if not entry:
                return 0.0
            if str(entry.get("last_family", "") or "").strip().lower() != "click":
                return 0.0
            board_change_count = int(entry.get("board_change_no_score_count", 0) or 0)
            score_change_count = int(entry.get("score_change_count", 0) or 0)
            progress_count = int(entry.get("progress_effect_count", 0) or 0)
            structural_count = int(entry.get("structural_delta_count", 0) or 0)
            weak_count = int(entry.get("weak_effect_count", 0) or 0)
            no_change_count = int(entry.get("no_change_count", 0) or 0)
            probe_count = int(entry.get("probe_count", 0) or 0)
            current_step = int(getattr(state, "step_index", 0) or 0)
            last_step = int(entry.get("last_step_idx", -10_000) or -10_000)
            age = max(0, current_step - last_step)
            recency_bonus = 0.0
            if age <= 4:
                recency_bonus = 0.70
            elif age <= 12:
                recency_bonus = 0.24
            raw_score = (
                1.10 * float(board_change_count)
                + 1.05 * float(structural_count)
                + 2.20 * float(progress_count)
                + 2.80 * float(score_change_count)
                + 0.20 * float(weak_count)
                + recency_bonus
                - 0.18 * float(no_change_count)
                - 0.04 * float(max(0, probe_count - 1))
            )
            return max(0.0, min(3.4, raw_score))

        seed_component = str(getattr(candidate, "component_id", "") or "").strip()
        seed_anchor = str(getattr(candidate, "anchor_key", "") or "").strip()
        if seed_component or seed_anchor:
            points.append(
                {
                    "component_id": seed_component,
                    "anchor_key": seed_anchor,
                    "row": int(seed_row),
                    "col": int(seed_col),
                    "value": 0,
                    "size": 0,
                    "synthetic_seed": True,
                }
            )

        auxiliary_family_entry = self._local_effect_family_entry(runtime, family="auxiliary")
        auxiliary_effect_active = any(
            int(auxiliary_family_entry.get(key, 0) or 0) > 0
            for key in (
                "score_change_count",
                "board_change_no_score_count",
                "progress_effect_count",
                "structural_delta_count",
                "weak_effect_count",
                "novel_hash_only_count",
            )
        )
        if auxiliary_effect_active:
            unsettled_auxiliary_action = self._recent_unsettled_auxiliary_action(runtime)
            if unsettled_auxiliary_action:
                self._auxiliary_click_pair_deferred_unsettled_count += 1
                self._auxiliary_click_pair_settle_tick_count += 1
                return [
                    PlannedAction(
                        action_name=unsettled_auxiliary_action,
                        target_component_id=str(seed_component or seed_anchor or "auxiliary"),
                        parameters={},
                        reason="stabilize_verify_aux_click_pair:settle_tick",
                    )
                ]
            visual_arrangement_actions = self._build_auxiliary_visual_arrangement_actions(
                runtime=runtime,
                validate_action_name=self._recent_auxiliary_action_name(runtime),
            )
            if visual_arrangement_actions:
                return visual_arrangement_actions
            for delta_point in self._recent_auxiliary_board_delta_points(runtime, limit=32):
                if any(same_point(existing, delta_point) for existing in points):
                    continue
                points.append(delta_point)

        delta_reference_points = [
            point for point in points if bool(point.get("delta_anchor"))
        ]
        source_points = [
            point
            for point in points
            if not bool(point.get("delta_anchor"))
            and (
                int(point["row"]) >= lower_cutoff
            or (
                int(point["row"]) >= int(float(row_count) * 0.50)
                and abs(int(point["col"]) - int(seed_col)) <= 8
            )
            )
        ]
        if not source_points and int(seed_row) >= int(float(row_count) * 0.50):
            source_points = [
                {
                    "component_id": seed_component,
                    "anchor_key": seed_anchor,
                    "row": int(seed_row),
                    "col": int(seed_col),
                }
            ]
        if not source_points:
            return []
        if auxiliary_effect_active:
            auxiliary_source_cutoff = max(1, int(float(row_count) * 0.50))
            lower_source_points = [
                point for point in source_points if int(point["row"]) >= auxiliary_source_cutoff
            ]
            if lower_source_points:
                source_points = lower_source_points
            real_source_points = [
                point for point in source_points if not bool(point.get("synthetic_seed"))
            ]
            if delta_reference_points and real_source_points:
                source_points = real_source_points

        same_band_target_points = [
            point
            for point in points
            if not bool(point.get("delta_anchor"))
            and int(point["row"]) >= int(float(row_count) * 0.55)
            and int(point["row"]) >= max(0, int(seed_row) - 12)
        ]
        upper_target_points = [
            point
            for point in points
            if not bool(point.get("delta_anchor"))
            and int(point["row"]) > upper_cutoff
            and int(point["row"]) < max(seed_row, int(float(row_count) * 0.55)) - 6
        ]
        if auxiliary_effect_active:
            lower_delta_points = [
                point
                for point in delta_reference_points
                if bool(point.get("delta_anchor")) and int(point["row"]) >= int(float(row_count) * 0.50)
            ]
            upper_rows = sorted(
                {
                    int(point["row"])
                    for point in upper_target_points
                    if int(point["row"]) > upper_cutoff
                },
                key=lambda row: abs(float(row) - float(row_count) * 0.45),
            )
            if lower_delta_points and upper_rows:
                projection_row = int(upper_rows[0])
                max_col = max((len(row) for row in rows), default=0) - 1
                projection_cols: list[int] = []
                for point in lower_delta_points[:4]:
                    col = int(point["col"])
                    if col not in projection_cols:
                        projection_cols.append(col)
                if int(seed_col) not in projection_cols:
                    projection_cols.append(int(seed_col))
                for col in projection_cols[:5]:
                    if max_col >= 0:
                        col = max(0, min(max_col, int(col)))
                    projection = {
                        "component_id": f"aux_projection_{projection_row}_{col}",
                        "anchor_key": f"projection||{projection_row}||{col}",
                        "row": int(projection_row),
                        "col": int(col),
                        "value": 0,
                        "size": 1,
                        "projection_anchor": True,
                    }
                    if any(same_point(existing, projection) for existing in upper_target_points):
                        continue
                    upper_target_points.append(projection)
                    points.append(projection)
        if not same_band_target_points and not upper_target_points:
            return []

        ranked: list[tuple[float, str, dict[str, Any], dict[str, Any], float, str]] = []

        def append_ranked(
            *,
            base_score: float,
            target_kind: str,
            source: dict[str, Any],
            target: dict[str, Any],
        ) -> None:
            pair_key = click_pair_key(source, target)
            reverse_key = click_pair_key(target, source)
            previous_attempts = int(self._auxiliary_click_pair_attempt_counts.get(pair_key, 0) or 0)
            reverse_attempts = int(self._auxiliary_click_pair_attempt_counts.get(reverse_key, 0) or 0)
            source_effect = point_effect_score(source)
            target_effect = point_effect_score(target)
            if previous_attempts >= 2:
                self._auxiliary_click_pair_repeat_suppressed_count += 1
                return
            if reverse_attempts > 0 and source_effect <= 0.0 and not recent_click_board_change(source):
                self._auxiliary_click_pair_reverse_suppressed_count += 1
                return
            ranked_score = (
                float(base_score)
                + float(source_effect)
                + 0.18 * float(target_effect)
                - 0.55 * float(previous_attempts)
                - 0.75 * float(reverse_attempts)
            )
            if auxiliary_effect_active:
                if target_kind == "upper_band":
                    ranked_score += 1.85
                elif target_kind == "same_band":
                    ranked_score -= 0.30
                if bool(source.get("delta_anchor")):
                    ranked_score += 3.10
                if bool(target.get("delta_anchor")):
                    ranked_score += 0.75
                if bool(target.get("projection_anchor")):
                    ranked_score -= 0.20
                if delta_reference_points:
                    source_delta_distance = min(
                        abs(int(source.get("row", 0) or 0) - int(delta.get("row", 0) or 0))
                        + abs(int(source.get("col", 0) or 0) - int(delta.get("col", 0) or 0))
                        for delta in delta_reference_points
                    )
                    if source_delta_distance <= 18:
                        ranked_score += 4.20 - 0.10 * float(source_delta_distance)
                    target_delta_distance = min(
                        abs(int(target.get("row", 0) or 0) - int(delta.get("row", 0) or 0))
                        + abs(int(target.get("col", 0) or 0) - int(delta.get("col", 0) or 0))
                        for delta in delta_reference_points
                    )
                    if target_delta_distance <= 18:
                        ranked_score += 0.50 - 0.025 * float(target_delta_distance)
            ranked.append((ranked_score, target_kind, source, target, source_effect, pair_key))

        for source in source_points:
            source_row = int(source["row"])
            source_col = int(source["col"])
            seed_distance = abs(source_row - int(seed_row)) + abs(source_col - int(seed_col))
            for target in same_band_target_points:
                if same_point(source, target):
                    continue
                target_row = int(target["row"])
                target_col = int(target["col"])
                row_distance = abs(target_row - source_row)
                col_distance = abs(target_col - source_col)
                if col_distance <= 1 and row_distance <= 1:
                    continue
                if row_distance > max(8, int(float(row_count) * 0.16)):
                    continue
                score = (
                    9.15
                    - 0.055 * float(seed_distance)
                    - 0.13 * float(row_distance)
                    - 0.045 * abs(float(col_distance) - 8.0)
                    + (0.32 if row_distance <= 1 else 0.0)
                )
                append_ranked(
                    base_score=score,
                    target_kind="same_band",
                    source=source,
                    target=target,
                )
            for target in upper_target_points:
                target_row = int(target["row"])
                target_col = int(target["col"])
                if target_row >= source_row - 6:
                    continue
                col_distance = abs(target_col - source_col)
                if col_distance > 18:
                    continue
                mid_row = float(row_count) * 0.45
                score = (
                    8.0
                    - 0.08 * float(seed_distance)
                    - 0.18 * float(col_distance)
                    - 0.035 * abs(float(target_row) - mid_row)
                )
                append_ranked(
                    base_score=score,
                    target_kind="upper_band",
                    source=source,
                    target=target,
                )
        if not ranked:
            return []
        ranked.sort(
            key=lambda item: (
                -item[0],
                0 if item[1] == "same_band" else 1,
                abs(int(item[3]["col"]) - int(item[2]["col"])),
                int(item[2]["row"]),
                int(item[3]["row"]),
                str(item[2]["component_id"]),
                str(item[3]["component_id"]),
            )
        )
        _score, _target_kind, source, target, source_effect, selected_pair_key = ranked[0]
        if source_effect > 0.0:
            self._auxiliary_click_pair_outcome_ranked_count += 1
        if auxiliary_effect_active and _target_kind == "upper_band":
            self._auxiliary_click_pair_vertical_transfer_count += 1
        selected_delta_guided = False
        if delta_reference_points:
            selected_delta_distance = min(
                abs(int(source.get("row", 0) or 0) - int(delta.get("row", 0) or 0))
                + abs(int(source.get("col", 0) or 0) - int(delta.get("col", 0) or 0))
                for delta in delta_reference_points
            )
            selected_delta_guided = selected_delta_distance <= 14
        if (
            selected_delta_guided
            or bool(source.get("delta_anchor"))
            or bool(target.get("delta_anchor"))
            or bool(target.get("projection_anchor"))
        ):
            self._auxiliary_click_pair_delta_anchor_count += 1
        self._auxiliary_click_pair_attempt_counts[selected_pair_key] = (
            int(self._auxiliary_click_pair_attempt_counts.get(selected_pair_key, 0) or 0) + 1
        )

        def click(point: dict[str, Any], suffix: str) -> PlannedAction:
            return PlannedAction(
                action_name="click",
                target_component_id=str(point.get("component_id", "") or ""),
                parameters={
                    "position": {
                        "row": int(point.get("row", 0) or 0),
                        "col": int(point.get("col", 0) or 0),
                    }
                },
                reason=f"stabilize_verify_aux_click_pair:{suffix}",
            )

        if recent_click_board_change(source):
            return [click(target, "target")]
        return [click(source, "source"), click(target, "target")]

    def _pick_base_candidate(
        self,
        *,
        candidates: list[StructuredCandidate],
        avoid_component_id: str = "",
        prefer_family: str = "",
        strict_avoid_component: bool = False,
    ) -> StructuredCandidate | None:
        valid = [candidate for candidate in candidates if not candidate.reject_reason]
        if avoid_component_id:
            alternatives = [
                candidate for candidate in valid if candidate.component_id != avoid_component_id
            ]
            if alternatives:
                valid = alternatives
            elif strict_avoid_component:
                return None
        valid = self._filter_saturated_no_score_effect_candidates(valid)
        if not valid:
            return None
        if not self._move_action_available and self._click_only_grid_probe_focus_key:
            focus_key = str(self._click_only_grid_probe_focus_key or "")
            focus_count = int(self._click_only_grid_probe_packet_counts.get(focus_key, 0) or 0)
            if focus_count < 5:
                focused = [
                    candidate
                    for candidate in valid
                    if str(candidate.anchor_key or candidate.component_id or "").strip() == focus_key
                    and str(candidate.source or "").strip() in {"grid_scan", "level_rule_symbol_scout"}
                ]
                if focused:
                    return max(
                        focused,
                        key=lambda candidate: (
                            float(candidate.base_score or 0.0) + float(candidate.effect_bonus or 0.0),
                            int(candidate.center_row or 0),
                            int(candidate.center_col or 0),
                            str(candidate.component_id or ""),
                        ),
                    )
            else:
                self._click_only_grid_probe_focus_key = ""

        def score(candidate: StructuredCandidate) -> tuple[float, float, float, float]:
            source = str(candidate.source or "")
            stage = str(getattr(candidate, "stage", "") or "")
            closure_status = str(getattr(candidate, "closure_status", "") or "")
            transition_frontier_bonus = 0.45 if source == "transition_frontier" else 0.0
            rule_hypothesis_bonus = 2.40 if source == "rule_hypothesis" else 0.0
            stabilize_verify_bonus = 0.0
            if source == "stabilize_verify":
                stabilize_verify_bonus = (
                    1.80
                    if stage == "lease_followup"
                    else 0.65
                    if closure_status == "closed"
                    else -0.35
                )
            stabilize_lease_bonus = 0.55 if stage == "lease_followup" else 0.0
            goal_probe_bonus = 2.20 if source == "goal_probe" else 0.0
            object_escape_bonus = 0.95 if source == "object_escape" else 0.0
            local_click_fanout_bonus = 0.85 if source == "local_click_fanout" else 0.0
            reversible_click_bonus = 0.95 if source == "local_reversible_click" else 0.0
            level_grid_scan_bonus = 0.0
            level_symbol_scout_bonus = (
                -1.95
                if source == "level_rule_symbol_scout"
                and stage.startswith("level_rule_discovery")
                else 0.0
            )
            return (
                float(candidate.base_score or 0.0)
                + transition_frontier_bonus
                + rule_hypothesis_bonus
                + stabilize_verify_bonus
                + stabilize_lease_bonus
                + goal_probe_bonus
                + object_escape_bonus
                + local_click_fanout_bonus
                + reversible_click_bonus
                + level_grid_scan_bonus
                + level_symbol_scout_bonus,
                1.0 if candidate.recent_live else 0.0,
                1.0 if candidate.is_hedge else 0.0,
                1.0 if prefer_family and candidate.family == prefer_family else 0.0,
            )

        return max(valid, key=score)

    def _note_stabilize_one_shot_blocked(self, reason: str) -> None:
        key = str(reason or "").strip() or "unknown"
        self._stabilize_verify_one_shot_blocked_reason_counts[key] = (
            int(self._stabilize_verify_one_shot_blocked_reason_counts.get(key, 0) or 0) + 1
        )

    def _note_closed_rule_one_shot_blocked(self, reason: str) -> None:
        key = str(reason or "").strip() or "unknown"
        self._closed_rule_one_shot_blocked_reason_counts[key] = (
            int(self._closed_rule_one_shot_blocked_reason_counts.get(key, 0) or 0) + 1
        )

    @staticmethod
    def _closed_rule_one_shot_key(candidate: StructuredCandidate) -> str:
        hypothesis_id = str(getattr(candidate, "hypothesis_id", "") or "").strip()
        if hypothesis_id:
            return hypothesis_id
        return "|".join(
            [
                str(candidate.source or ""),
                str(getattr(candidate, "closure_status", "") or ""),
                str(getattr(candidate, "stage", "") or ""),
                str(getattr(candidate, "action_name", "") or ""),
                str(candidate.component_id or ""),
                str(candidate.anchor_key or ""),
            ]
        )

    def _build_closed_rule_one_shot_plan(
        self,
        *,
        runtime: Any,
        agent: Any | None,
        semantic_action_lease: dict[str, Any],
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
    ) -> tuple[list[PlannedAction], ScoredCandidate] | None:
        base_pool = getattr(runtime, "_last_base_pool", None)
        candidates = list(getattr(base_pool, "candidates", None) or [])
        candidate = self._pick_closed_rule_one_shot_candidate(candidates=candidates)
        if candidate is None:
            return None
        family = str(candidate.family or "").strip().lower()
        action_name = str(getattr(candidate, "action_name", "") or "").strip().upper()
        if family not in {"click", "move", "auxiliary"}:
            self._note_closed_rule_one_shot_blocked("unsupported_family")
            return None
        key = self._closed_rule_one_shot_key(candidate)
        self._closed_rule_one_shot_selected_count += 1
        scored = self._structured_candidate_to_scored_candidate(candidate)
        if family == "auxiliary":
            if (
                not _GENERIC_KEYBOARD_ACTION_RE.match(action_name)
                or action_name in _MOVE_ACTION_SET
                or action_name in {_CLICK_ACTION, "RESET"}
            ):
                self._note_closed_rule_one_shot_blocked("missing_auxiliary_action")
                return None
            available = self._normalized_available_actions(allowed_actions, blocked_actions)
            if action_name not in available:
                self._note_closed_rule_one_shot_blocked("auxiliary_unavailable")
                return None
            planned = [
                PlannedAction(
                    action_name=action_name,
                    target_component_id=str(candidate.component_id or ""),
                    parameters={},
                    reason="closed_rule_one_shot",
                )
            ]
        elif family == "move":
            if action_name not in _MOVE_ACTION_SET:
                self._note_closed_rule_one_shot_blocked("missing_move_action")
                return None
            available = self._normalized_available_actions(allowed_actions, blocked_actions)
            if action_name not in available:
                self._note_closed_rule_one_shot_blocked("move_unavailable")
                return None
            planned = [
                PlannedAction(
                    action_name=action_name,
                    target_component_id=str(candidate.component_id or ""),
                    parameters={},
                    reason="closed_rule_one_shot",
                )
            ]
        else:
            planned = [
                PlannedAction(
                    action_name="click",
                    target_component_id=str(candidate.component_id or ""),
                    parameters={
                        "position": {
                            "row": int(candidate.center_row or 0),
                            "col": int(candidate.center_col or 0),
                        }
                    },
                    reason="closed_rule_one_shot",
                )
            ]
        normalized = self._normalize_planned_actions(
            runtime=runtime,
            actions=planned,
            selected_target=scored,
            agent=agent,
            semantic_action_lease=semantic_action_lease,
        )
        if not normalized:
            self._note_closed_rule_one_shot_blocked("normalization_failed")
            return None
        if key:
            self._closed_rule_one_shot_keys.add(key)
        self._closed_rule_one_shot_normalized_count += 1
        return normalized, scored

    def _build_stabilize_one_shot_plan(
        self,
        *,
        runtime: Any,
        agent: Any | None,
        semantic_action_lease: dict[str, Any],
        allowed_actions: list[str] | None,
        blocked_actions: set[str] | None,
        allow_fresh_unstarved: bool = False,
    ) -> tuple[list[PlannedAction], ScoredCandidate] | None:
        semantic_phase = str(
            semantic_action_lease.get("semantic_phase", "") or ""
        ).strip().lower()
        semantic_route_actions = [
            str(action_name or "").strip().upper()
            for action_name in list(
                semantic_action_lease.get("route_preview")
                or semantic_action_lease.get("suggested_actions")
                or []
            )
            if str(action_name or "").strip().upper() in _MOVE_ACTION_SET
        ]
        semantic_route_source = str(
            semantic_action_lease.get("route_source", "") or ""
        ).strip()
        short_resource_probe = bool(
            semantic_phase == "site_reaching"
            and semantic_route_source == "gameplay_resource_bfs"
            and 0 < len(semantic_route_actions) <= 2
        )
        if (
            bool(semantic_action_lease.get("active", False))
            and semantic_route_actions
            and not short_resource_probe
            and (
                bool(semantic_action_lease.get("hidden_state_executor_active", False))
                or semantic_phase in {
                    "branch_entry",
                    "site_reaching",
                    "state_change_verification",
                    "goal_route_execution",
                }
            )
        ):
            self._note_stabilize_one_shot_blocked("semantic_route_active")
            return None
        base_pool = getattr(runtime, "_last_base_pool", None)
        candidates = list(getattr(base_pool, "candidates", None) or [])
        candidate = self._pick_pending_stabilize_verify_candidate(
            runtime=runtime,
            candidates=candidates,
            allow_fresh_unstarved=allow_fresh_unstarved,
            allow_lease_followup=True,
        )
        if candidate is None:
            return None
        family = str(candidate.family or "").strip().lower()
        action_name = str(getattr(candidate, "action_name", "") or "").strip().upper()
        if family not in {"click", "move", "auxiliary"}:
            self._note_stabilize_one_shot_blocked("unsupported_family")
            return None
        self._stabilize_verify_one_shot_selected_count += 1
        scored = self._structured_candidate_to_scored_candidate(candidate)
        if family == "auxiliary":
            if (
                not _GENERIC_KEYBOARD_ACTION_RE.match(action_name)
                or action_name in _MOVE_ACTION_SET
                or action_name in {_CLICK_ACTION, "RESET"}
            ):
                self._note_stabilize_one_shot_blocked("missing_auxiliary_action")
                return None
            available = self._normalized_available_actions(allowed_actions, blocked_actions)
            if action_name not in available:
                self._note_stabilize_one_shot_blocked("auxiliary_unavailable")
                return None
            planned = self._build_auxiliary_visual_arrangement_actions(
                runtime=runtime,
                validate_action_name=action_name,
            )
            if not planned:
                planned = [
                    PlannedAction(
                        action_name=action_name,
                        target_component_id=str(candidate.component_id or ""),
                        parameters={},
                        reason="stabilize_verify_one_shot",
                    )
                ]
        elif family == "move":
            if action_name not in _MOVE_ACTION_SET:
                self._note_stabilize_one_shot_blocked("missing_move_action")
                return None
            available = self._normalized_available_actions(allowed_actions, blocked_actions)
            if action_name not in available:
                self._note_stabilize_one_shot_blocked("move_unavailable")
                return None
            planned = [
                PlannedAction(
                    action_name=action_name,
                    target_component_id=str(candidate.component_id or ""),
                    parameters={},
                    reason="stabilize_verify_one_shot",
                )
            ]
        else:
            continuation_kind = str(getattr(candidate, "continuation_kind", "") or "").strip()
            planned = []
            if continuation_kind == "aux_click_pair":
                planned = self._build_auxiliary_click_pair_actions(
                    runtime=runtime,
                    candidate=candidate,
                )
            if not planned:
                planned = [
                    PlannedAction(
                        action_name="click",
                        target_component_id=str(candidate.component_id or ""),
                        parameters={
                            "position": {
                                "row": int(candidate.center_row or 0),
                                "col": int(candidate.center_col or 0),
                            }
                        },
                        reason="stabilize_verify_one_shot",
                    )
                ]
        normalized = self._normalize_planned_actions(
            runtime=runtime,
            actions=planned,
            selected_target=scored,
            agent=agent,
            semantic_action_lease=semantic_action_lease,
        )
        if not normalized:
            self._note_stabilize_one_shot_blocked("normalization_failed")
            return None
        window_key = str(getattr(candidate, "window_id", "") or "").strip()
        if window_key:
            self._stabilize_verify_one_shot_window_keys.add(window_key)
        self._stabilize_verify_one_shot_normalized_count += 1
        return normalized, scored

    def _pick_closed_rule_one_shot_candidate(
        self,
        *,
        candidates: list[StructuredCandidate],
    ) -> StructuredCandidate | None:
        valid = [
            candidate
            for candidate in candidates
            if not candidate.reject_reason
            and str(getattr(candidate, "closure_status", "") or "").strip() == "closed"
            and self._closed_rule_one_shot_key(candidate) not in self._closed_rule_one_shot_keys
        ]
        if not valid:
            self._note_closed_rule_one_shot_blocked("no_closed_candidate")
            return None
        self._closed_rule_one_shot_seen_count += 1
        return max(
            valid,
            key=lambda candidate: (
                1.0 if str(getattr(candidate, "hypothesis_kind", "") or "") == "closed_goal_probe" else 0.0,
                1.0 if str(candidate.family or "").strip().lower() == "move" else 0.0,
                float(candidate.base_score or 0.0) + float(candidate.effect_bonus or 0.0),
                int(candidate.center_row or 0),
                int(candidate.center_col or 0),
                str(candidate.component_id or ""),
            ),
        )

    def _pick_starved_goal_probe_candidate(
        self,
        *,
        runtime: Any,
        candidates: list[StructuredCandidate],
    ) -> StructuredCandidate | None:
        discovery = getattr(runtime, "rule_discovery", None)
        goal_probe_candidate_count = int(
            getattr(discovery, "goal_probe_candidate_count", 0) or 0
        )
        goal_probe_issued_count = int(getattr(discovery, "goal_probe_issued_count", 0) or 0)
        if goal_probe_candidate_count < 12 or goal_probe_issued_count >= 4:
            return None
        if any(
            str(candidate.source or "").strip() == "stabilize_verify"
            or str(getattr(candidate, "closure_status", "") or "").strip() == "closed"
            for candidate in candidates
            if not candidate.reject_reason
        ):
            return None
        valid = [
            candidate
            for candidate in candidates
            if not candidate.reject_reason
            and str(candidate.source or "").strip() == "goal_probe"
        ]
        if not valid:
            return None
        return max(
            valid,
            key=lambda candidate: (
                float(candidate.base_score or 0.0),
                float(candidate.effect_bonus or 0.0),
                int(candidate.center_row or 0),
                int(candidate.center_col or 0),
                str(candidate.component_id or ""),
            ),
        )

    def _pick_level_rule_grid_scan_candidate(
        self,
        *,
        runtime: Any,
        candidates: list[StructuredCandidate],
    ) -> StructuredCandidate | None:
        valid = [
            candidate
            for candidate in candidates
            if not candidate.reject_reason
            and str(candidate.source or "").strip() in {"level_rule_symbol_scout", "grid_scan"}
            and str(getattr(candidate, "stage", "") or "").startswith("level_rule_discovery")
        ]
        if not valid:
            return None
        if any(
            not candidate.reject_reason
            and str(candidate.source or "").strip()
            in {"goal_probe", "rule_hypothesis", "stabilize_verify"}
            for candidate in candidates
        ):
            return None
        state = getattr(runtime, "last_state", None)
        level_number = 0
        for raw in (
            getattr(state, "level_number", None),
            getattr(runtime, "last_level_number", None),
        ):
            try:
                if raw is not None:
                    level_number = int(raw)
                    break
            except (TypeError, ValueError):
                continue
        if level_number < 2:
            return None
        try:
            score_delta = float(getattr(state, "score_delta", 0.0) or 0.0)
        except (TypeError, ValueError):
            score_delta = 0.0
        if abs(score_delta) > 1e-9:
            return None

        rows = list(getattr(runtime, "step_metrics", None) or [])
        current_level_rows = [
            dict(row or {})
            for row in rows
            if isinstance(row, dict)
            and int(row.get("level", row.get("level_number", 0)) or 0) == int(level_number)
        ]
        recent_rows = current_level_rows[-8:]
        if any(
            str(row.get("selected_target_source", "") or "").strip()
            in {"level_rule_symbol_scout", "grid_scan"}
            for row in recent_rows
        ):
            return None

        scout_selected_count = sum(
            1
            for row in current_level_rows
            if str(row.get("selected_target_source", "") or "").strip()
            in {"level_rule_symbol_scout", "grid_scan"}
        )
        rule_churn_count = sum(
            1
            for row in recent_rows
            if str(row.get("selected_target_source", "") or "").strip()
            in {"rule_hypothesis", "goal_probe", "stabilize_verify"}
        )
        if scout_selected_count > 0 and rule_churn_count < 4:
            return None

        return max(
            valid,
            key=lambda candidate: (
                float(candidate.base_score or 0.0)
                + float(candidate.effect_bonus or 0.0)
                + (
                    0.85
                    if str(candidate.source or "").strip() == "level_rule_symbol_scout"
                    else 0.0
                ),
                int(candidate.center_row or 0),
                int(candidate.center_col or 0),
                str(candidate.component_id or ""),
            ),
        )

    def _pick_post_score_source_transfer_candidate(
        self,
        *,
        runtime: Any,
        candidates: list[StructuredCandidate],
    ) -> StructuredCandidate | None:
        state = getattr(runtime, "last_state", None)
        level_number = 0
        for raw in (
            getattr(state, "level_number", None),
            getattr(runtime, "last_level_number", None),
        ):
            try:
                if raw is not None:
                    level_number = int(raw)
                    break
            except (TypeError, ValueError):
                continue
        if level_number < 2:
            return None
        try:
            score_delta = float(getattr(state, "score_delta", 0.0) or 0.0)
        except (TypeError, ValueError):
            score_delta = 0.0
        if abs(score_delta) > 1e-9:
            return None

        rows = [
            dict(row or {})
            for row in list(getattr(runtime, "step_metrics", None) or [])
            if isinstance(row, dict)
        ]
        if not rows:
            return None
        current_level_rows = [
            row
            for row in rows
            if int(row.get("level", row.get("level_number", 0)) or 0) == int(level_number)
        ]
        if len(current_level_rows) > 20:
            return None

        last_scoreful_source = ""
        previous_score: float | None = None
        for row in rows[-40:]:
            try:
                score = float(row.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if previous_score is not None and score > previous_score + 1e-9:
                source = str(row.get("selected_target_source", "") or "").strip()
                if source:
                    last_scoreful_source = source
            previous_score = score
        if last_scoreful_source not in {
            "object_escape",
            "transition_frontier",
            "aligned_object_sweep",
        }:
            return None
        selected_count = sum(
            1
            for row in current_level_rows
            if str(row.get("selected_target_source", "") or "").strip() == last_scoreful_source
        )
        if selected_count >= 2:
            return None
        valid = [
            candidate
            for candidate in candidates
            if not candidate.reject_reason
            and str(candidate.source or "").strip() == last_scoreful_source
        ]
        if not valid:
            return None
        return max(
            valid,
            key=lambda candidate: (
                float(candidate.base_score or 0.0) + float(candidate.effect_bonus or 0.0),
                1.0 if str(candidate.source or "").strip() == "object_escape" else 0.0,
                int(candidate.center_row or 0),
                int(candidate.center_col or 0),
                str(candidate.component_id or ""),
            ),
        )

    def _pick_pending_stabilize_verify_candidate(
        self,
        *,
        runtime: Any,
        candidates: list[StructuredCandidate],
        allow_fresh_unstarved: bool = False,
        allow_lease_followup: bool = False,
    ) -> StructuredCandidate | None:
        discovery = getattr(runtime, "rule_discovery", None)
        if discovery is None:
            self._note_stabilize_one_shot_blocked("missing_discovery")
            return None
        if int(getattr(discovery, "stabilize_verify_window_opened_count", 0) or 0) <= 0:
            self._note_stabilize_one_shot_blocked("no_open_window")
            return None
        allowed_stages = {"settled_retest", "relation_retest"}
        if bool(allow_lease_followup):
            allowed_stages.add("lease_followup")
        valid = [
            candidate
            for candidate in candidates
            if not candidate.reject_reason
            and str(candidate.source or "").strip() == "stabilize_verify"
            and str(getattr(candidate, "stage", "") or "").strip() in allowed_stages
        ]
        if not valid:
            self._note_stabilize_one_shot_blocked("no_base_candidate")
            return None
        goal_probe_candidate_count = int(
            getattr(discovery, "goal_probe_candidate_count", 0) or 0
        )
        issued_count = int(getattr(discovery, "stabilize_verify_issued_count", 0) or 0)
        fresh_unstarved_allowed = bool(allow_fresh_unstarved) and (
            issued_count <= 0 or not self._move_action_available
        )
        starved_allowed = goal_probe_candidate_count >= 128
        eligible = [
            candidate
            for candidate in valid
            if (
                (
                    (
                        bool(str(getattr(candidate, "window_id", "") or "").strip())
                        and str(getattr(candidate, "window_id", "") or "").strip()
                        not in self._stabilize_verify_one_shot_window_keys
                    )
                    or (
                        not bool(str(getattr(candidate, "window_id", "") or "").strip())
                        and issued_count <= 0
                    )
                )
                and (
                    fresh_unstarved_allowed
                    or starved_allowed
                )
            )
            or (
                bool(allow_lease_followup)
                and str(getattr(candidate, "stage", "") or "").strip() == "lease_followup"
            )
        ]
        if not eligible:
            unissued_window_exists = any(
                bool(str(getattr(candidate, "window_id", "") or "").strip())
                and str(getattr(candidate, "window_id", "") or "").strip()
                not in self._stabilize_verify_one_shot_window_keys
                for candidate in valid
            )
            if (issued_count <= 0 or unissued_window_exists) and goal_probe_candidate_count < 128:
                self._note_stabilize_one_shot_blocked("not_starved")
                return None
            self._note_stabilize_one_shot_blocked("already_issued")
            return None
        self._stabilize_verify_one_shot_seen_count += 1
        return max(
            eligible,
            key=lambda candidate: (
                1.0 if str(getattr(candidate, "stage", "") or "") == "lease_followup" else 0.0,
                1.0
                if str(candidate.family or "").strip().lower() == "auxiliary"
                else 0.82
                if str(candidate.family or "").strip().lower() == "click"
                or str(getattr(candidate, "action_name", "") or "").strip().upper()
                in {"ACTION6", "CLICK"}
                else 0.0,
                1.0 if str(getattr(candidate, "stage", "") or "") == "settled_retest" else 0.0,
                float(candidate.base_score or 0.0) + float(candidate.effect_bonus or 0.0),
                int(candidate.center_row or 0),
                int(candidate.center_col or 0),
                str(candidate.component_id or ""),
            ),
        )

    @staticmethod
    def _filter_saturated_no_score_effect_candidates(
        candidates: list[StructuredCandidate],
    ) -> list[StructuredCandidate]:
        if not candidates:
            return []
        non_saturated = [
            candidate
            for candidate in candidates
            if float(getattr(candidate, "effect_bonus", 0.0) or 0.0) > -0.15
        ]
        return non_saturated if non_saturated else list(candidates)

    def _pick_same_component_family_variant(
        self,
        *,
        candidates: list[StructuredCandidate],
        component_id: str,
        current_family: str = "",
    ) -> StructuredCandidate | None:
        normalized_component_id = str(component_id or "").strip()
        normalized_current_family = str(current_family or "").strip()
        if not normalized_component_id:
            return None
        valid = [
            candidate
            for candidate in candidates
            if not candidate.reject_reason
            and str(candidate.component_id or "").strip() == normalized_component_id
            and str(candidate.family or "").strip()
            and str(candidate.family or "").strip() != normalized_current_family
        ]
        valid = self._filter_saturated_no_score_effect_candidates(valid)
        if not valid:
            return None

        def score(candidate: StructuredCandidate) -> tuple[float, float, float, float]:
            transition_frontier_bonus = 0.35 if str(candidate.source or "") == "transition_frontier" else 0.0
            return (
                1.0 if str(candidate.family or "").strip() == "move" else 0.0,
                float(candidate.base_score or 0.0) + transition_frontier_bonus,
                1.0 if candidate.recent_live else 0.0,
                1.0 if candidate.is_hedge else 0.0,
            )

        return max(valid, key=score)

    def _pick_family_candidate(
        self,
        *,
        candidates: list[StructuredCandidate],
        family: str,
        avoid_component_id: str = "",
    ) -> StructuredCandidate | None:
        normalized_family = str(family or "").strip()
        if not normalized_family:
            return None
        valid = [
            candidate
            for candidate in candidates
            if not candidate.reject_reason
            and str(candidate.family or "").strip() == normalized_family
            and (not avoid_component_id or candidate.component_id != avoid_component_id)
        ]
        valid = self._filter_saturated_no_score_effect_candidates(valid)
        if not valid:
            return None

        def score(candidate: StructuredCandidate) -> tuple[float, float, float]:
            source = str(candidate.source or "")
            stage = str(getattr(candidate, "stage", "") or "")
            closure_status = str(getattr(candidate, "closure_status", "") or "")
            transition_frontier_bonus = 0.50 if source == "transition_frontier" else 0.0
            rule_hypothesis_bonus = 1.60 if source == "rule_hypothesis" else 0.0
            stabilize_verify_bonus = 0.0
            if source == "stabilize_verify":
                stabilize_verify_bonus = (
                    1.48
                    if stage == "lease_followup"
                    else 0.45
                    if closure_status == "closed"
                    else -0.35
                )
            stabilize_lease_bonus = 0.55 if stage == "lease_followup" else 0.0
            goal_probe_bonus = 1.35 if source == "goal_probe" else 0.0
            object_escape_bonus = 1.15 if source == "object_escape" else 0.0
            local_click_fanout_bonus = 0.90 if source == "local_click_fanout" else 0.0
            reversible_click_bonus = 1.05 if source == "local_reversible_click" else 0.0
            level_grid_scan_bonus = (
                3.10
                if source == "grid_scan"
                and stage.startswith("level_rule_discovery")
                else 0.0
            )
            level_symbol_scout_bonus = (
                -1.95
                if source == "level_rule_symbol_scout"
                and stage.startswith("level_rule_discovery")
                else 0.0
            )
            return (
                float(candidate.base_score or 0.0)
                + transition_frontier_bonus
                + rule_hypothesis_bonus
                + stabilize_verify_bonus
                + stabilize_lease_bonus
                + goal_probe_bonus
                + object_escape_bonus
                + local_click_fanout_bonus
                + reversible_click_bonus
                + level_grid_scan_bonus
                + level_symbol_scout_bonus,
                1.0 if candidate.recent_live else 0.0,
                1.0 if candidate.is_hedge else 0.0,
            )

        return max(valid, key=score)

    @staticmethod
    def _coerce_summary_score(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().lower()
        if not text:
            return 0.0
        confidence_map = {
            "low": 0.25,
            "medium": 0.5,
            "high": 0.75,
        }
        if text in confidence_map:
            return confidence_map[text]
        try:
            return float(text)
        except (TypeError, ValueError):
            return 0.0

    def _selected_summary_to_scored_candidate(
        self,
        summary: Any,
        *,
        action_family_override: str = "",
        resolver_score_bonus: float = 0.0,
    ) -> ScoredCandidate | None:
        if not isinstance(summary, dict):
            return None
        component_id = str(summary.get("component_id", "") or summary.get("id", "") or "").strip()
        family = str(
            action_family_override
            or summary.get("action_family", "")
            or summary.get("family", "")
            or summary.get("transition_model_candidate_family", "")
            or ""
        ).strip()
        if not component_id or not family:
            return None
        base_score = self._coerce_summary_score(
            summary.get("admission_score", 0.0)
            or summary.get("confidence", 0.0)
            or summary.get("score", 0.0)
            or 0.0
        )
        anchor_key = str(summary.get("anchor_key", "") or "")
        center_row = int(summary.get("center_row", 0) or 0)
        center_col = int(summary.get("center_col", 0) or 0)
        if (not center_row and not center_col) and anchor_key:
            parts = [part.strip() for part in anchor_key.split("||")]
            if len(parts) >= 3:
                try:
                    center_row = int(parts[-2])
                    center_col = int(parts[-1])
                except (TypeError, ValueError):
                    center_row = 0
                    center_col = 0
        bonus = max(0.0, float(resolver_score_bonus or 0.0))
        action_name = str(summary.get("action_name", "") or "").strip().upper()
        if family == "auxiliary":
            action_name = _infer_auxiliary_action_name(
                action_name=action_name,
                component_id=component_id,
            )
        return ScoredCandidate(
            component_id=component_id,
            action_family=family,
            source=str(summary.get("source", "") or summary.get("reason", "") or "selected_target_summary"),
            action_name=action_name,
            anchor_key=anchor_key,
            center_row=center_row,
            center_col=center_col,
            base_admission_score=base_score,
            resolver_score=max(base_score + bonus, 0.01),
            stage=str(summary.get("stage", "") or "").strip(),
            closure_status=str(summary.get("closure_status", "") or "").strip(),
            hypothesis_id=str(summary.get("hypothesis_id", "") or "").strip(),
            hypothesis_kind=str(summary.get("hypothesis_kind", "") or "").strip(),
            linked_effect_id=str(summary.get("linked_effect_id", "") or "").strip(),
            window_id=str(summary.get("window_id", "") or "").strip(),
            continuation_kind=str(summary.get("continuation_kind", "") or "").strip(),
        )

    def _structured_candidate_to_scored_candidate(self, candidate: StructuredCandidate) -> ScoredCandidate:
        recent_live_bonus = self._resolver.config.recent_live_bonus if candidate.recent_live else 0.0
        hedge_bonus = self._resolver.config.hedge_bonus if candidate.is_hedge else 0.0
        action_name = str(getattr(candidate, "action_name", "") or "").strip().upper()
        if str(candidate.family or "").strip().lower() == "auxiliary":
            action_name = _infer_auxiliary_action_name(
                action_name=action_name,
                component_id=candidate.component_id,
            )
        return ScoredCandidate(
            component_id=candidate.component_id,
            action_family=candidate.family,
            source=candidate.source,
            action_name=action_name,
            anchor_key=candidate.anchor_key,
            center_row=int(candidate.center_row or 0),
            center_col=int(candidate.center_col or 0),
            base_admission_score=float(candidate.base_score or 0.0),
            recent_live_bonus=recent_live_bonus,
            hedge_bonus=hedge_bonus,
            resolver_score=float(candidate.base_score or 0.0) + recent_live_bonus + hedge_bonus,
            stage=str(getattr(candidate, "stage", "") or ""),
            closure_status=str(getattr(candidate, "closure_status", "") or ""),
            hypothesis_id=str(getattr(candidate, "hypothesis_id", "") or ""),
            hypothesis_kind=str(getattr(candidate, "hypothesis_kind", "") or ""),
            linked_effect_id=str(getattr(candidate, "linked_effect_id", "") or ""),
            window_id=str(getattr(candidate, "window_id", "") or ""),
            continuation_kind=str(getattr(candidate, "continuation_kind", "") or ""),
        )
