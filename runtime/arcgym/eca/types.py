"""Types for the lightweight ECA integration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal


VALID_INTERRUPT_DRIVES = ("world", "structure", "self", "progress", "control")


class LockAdvisoryLevel(str, Enum):
    """v9: 4-level advisory contract strength (replaces binary pass/fail gate)."""
    BLOCKED = "blocked"       # Hard block: contradiction, dead anchor, invalid target, repeat mismatch
    HINT = "hint"             # Weak contract: observable fields present, semantic incomplete
    SOFT_LOCK = "soft_lock"   # Observable + partial semantic grounding
    FULL_LOCK = "full_lock"   # Full contract with delta match evidence


class ECAHintState(str, Enum):
    """v10.1: 3-tier confidence state — ECA never controls, only observes.

    HINT:   weak hypothesis / partial delta / board change no score — memo only
    FOCUS:  same region/family with 2-3 evidence turns, score still 0 — soft priority
    COMMIT: score_delta > 0 or level gain — exploit protection allowed
    """
    NONE = "none"
    HINT = "hint"
    FOCUS = "focus"
    COMMIT = "commit"


class HardGuardReason(str, Enum):
    """v10.1: reasons that justify a hard guard stop."""
    DEAD_ANCHOR_REVISIT = "dead_anchor_revisit"
    CLICK_FAMILY_EXHAUSTED_RETRY = "click_family_exhausted_retry"
    EXACT_REPEAT_STATE_LOOP = "exact_repeat_state_loop"
    IMPOSSIBLE_ACTION = "impossible_action"
    QUEUE_CHURN_NO_CHANGE = "queue_churn_no_change"


class ActionFamily(str, Enum):
    MOVE = "move"
    CLICK = "click"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class CandidateSource(str, Enum):
    FRESH = "fresh"
    REVISIT = "revisit"
    NEIGHBOR = "neighbor"
    CHANGE_FRONTIER = "change_frontier"
    SYMMETRY = "symmetry"
    TRANSFER = "transfer"


class CandidateRejectReason(str, Enum):
    ANCHOR_COOLDOWN = "anchor_cooldown"
    LOW_ADMISSION_SCORE = "low_admission_score"
    FAMILY_REDUNDANT = "family_redundant"
    TURNOVER_TOO_EXPENSIVE = "turnover_too_expensive"
    SAME_DEAD_ANCHOR = "same_dead_anchor"
    DEAD_REVISIT_SATURATED = "dead_revisit_saturated"
    NO_INFORMATION_GAIN = "no_information_gain"


class LatchReason(str, Enum):
    HARD_WORLD = "hard_world_contradiction"
    HARD_SELF = "hard_self_contradiction"
    STABLE_CLICK_CONTRADICTION = "stable_click_contradiction"
    JUDGE_FAMILY_MISMATCH = "judge_rejection_due_to_family_mismatch"


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _parse_drive_list(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return VALID_INTERRUPT_DRIVES
    if isinstance(raw, str):
        items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    else:
        items = [str(item).strip().lower() for item in raw if str(item).strip()]
    unique = []
    for item in items:
        if item in VALID_INTERRUPT_DRIVES and item not in unique:
            unique.append(item)
    return tuple(unique) if unique else VALID_INTERRUPT_DRIVES


@dataclass
class DriveWeights:
    world: float = 0.30
    structure: float = 0.25
    self_model: float = 0.20
    progress: float = 0.15
    control: float = 0.10

    def normalized(self) -> "DriveWeights":
        total = self.world + self.structure + self.self_model + self.progress + self.control
        if total <= 0:
            return DriveWeights()
        return DriveWeights(
            world=self.world / total,
            structure=self.structure / total,
            self_model=self.self_model / total,
            progress=self.progress / total,
            control=self.control / total,
        )

    def to_dict(self) -> dict[str, float]:
        normalized = self.normalized()
        return {
            "world": normalized.world,
            "structure": normalized.structure,
            "self": normalized.self_model,
            "progress": normalized.progress,
            "control": normalized.control,
        }

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "DriveWeights":
        if not raw:
            return cls()
        return cls(
            world=float(raw.get("world", raw.get("L_world", 0.30))),
            structure=float(raw.get("structure", raw.get("L_structure", 0.25))),
            self_model=float(raw.get("self", raw.get("self_model", raw.get("L_self", 0.20)))),
            progress=float(raw.get("progress", raw.get("L_progress", 0.15))),
            control=float(raw.get("control", raw.get("L_control", 0.10))),
        ).normalized()


@dataclass
class ECAConfig:
    enabled: bool = False
    mode: str = "online"
    threshold_high: float = 0.62
    threshold_spike: float = 0.18
    cooldown_steps: int = 2
    max_interrupts: int = 6
    max_interrupts_per_level: int = 2
    high_drive_threshold: float = 0.72
    stagnation_window: int = 3
    min_commit_steps: int = 3
    hard_world_threshold: float = 0.78
    hard_self_threshold: float = 0.72
    base_gate_threshold: float = 0.70
    soft_trigger_k: int = 99
    max_queue_waste_for_soft: float = 0.05
    stable_plan_size: int = 6
    repair_plan_size: int = 3
    stable_target_blacklist_steps: int = 8
    stable_target_repeat_limit: int = 2
    stable_action_family_blacklist_steps: int = 6
    stable_control_latch_steps: int = 10
    stable_control_latch_release_changed_moves: int = 0
    stable_control_latch_release_move_contradiction_streak: int = 2
    random_interrupt_probability: float = 0.0
    random_seed: int = 0
    interrupt_drives: tuple[str, ...] = VALID_INTERRUPT_DRIVES
    hard_interrupt_drives: tuple[str, ...] = ("world", "self")
    soft_watch_drives: tuple[str, ...] = ("structure", "progress")
    soft_interrupt_limit: int = 1
    judge_policy: str = "repair_or_high_anxiety"
    structural_guidance_enabled: bool = True
    structural_time_shadow_enabled: bool = False
    shadow_promotion_min_turns: int = 5
    shadow_promotion_no_progress_window: int = 3
    shadow_promotion_mid_late_min_turns: int = 3
    shadow_promotion_mid_late_no_progress_window: int = 2
    hedge_regeneration_none_run_threshold: int = 4
    bootstrap_regeneration_min_step: int = 8
    bootstrap_regeneration_threshold_multiplier: int = 2
    # Contract health based bootstrap trigger
    contract_health_window: int = 12
    contract_health_threshold: float = 0.25
    contract_health_min_step: int = 20
    # Phase handoff carry set
    phase_handoff_carry_size: int = 3
    phase_handoff_resurface_ttl: int = 25
    # Selected target → low-trust hedge promotion
    selected_target_promotion_min_repeats: int = 2
    selected_target_promotion_window: int = 6
    # Regen success detection (lagged)
    regen_success_window: int = 5
    regen_success_window_long: int = 10
    # Health debt based regen stabilization
    health_debt_threshold: float = 1.5
    health_debt_floor: float = 0.25
    health_debt_decay_on_recovery: float = 0.5
    health_debt_refractory_steps: int = 6
    # Provisional hedge (v4 legacy, superseded by support score in v5)
    provisional_hedge_ttl: int = 15
    provisional_hedge_confirm_repeats: int = 2
    provisional_hedge_confirm_window: int = 8
    # ── CH v5: support score & maturation ─────────────────────────────
    support_score_decay: float = 0.92
    support_score_max_ledger: int = 12
    support_score_recent_window: int = 10
    # Bonus weights
    support_bonus_visible: float = 1.0
    support_bonus_lagged5: float = 1.5
    support_bonus_lagged10: float = 0.7
    support_bonus_streak: float = 0.25
    support_bonus_ch_gain: float = 0.5
    # Penalty weights
    support_penalty_contradiction: float = 2.5
    support_penalty_expire: float = 1.0
    # Promotion thresholds (v6: relaxed for faster maturation)
    support_promote_to_provisional: float = 1.8
    support_promote_to_protected: float = 3.5
    support_provisional_hits_min: int = 2
    support_protected_hits_min: int = 2
    support_protected_lagged5_min: int = 1
    support_protected_streak_min: int = 2
    # Provisional TTL (v5 style — extensible)
    support_provisional_ttl_initial: int = 8
    support_provisional_ttl_extend: int = 3
    support_provisional_ttl_cap: int = 14
    # Carry evidence
    support_carry_top_k: int = 3
    support_carry_conf_coeff: float = 0.4
    support_carry_conf_cap: float = 1.5
    support_carry_auto_provisional_threshold: float = 4.0
    # Persistence KPI windows
    support_rehit_window: int = 8
    support_carry_survival_window: int = 20
    frontier_guidance_enabled: bool = False
    force_post_reclaim_hold_on_nonweak_reclaim: bool = False
    force_post_final_hold_runtime_packet_after_one_step: bool = False
    force_post_final_shared_rerank_on_first_visible_step: bool = False
    synthesize_pending_shared_tail_sample_on_first_visible_step: bool = False
    disable_psa_in_final_reclaimable: bool = False
    # v9: authority backoff
    authority_decay_per_no_value: float = 0.08
    authority_recovery_per_hit: float = 0.15
    authority_floor: float = 0.2
    authority_backoff_window: int = 8
    authority_min_steps_before_decay: int = 5
    # v10.1: observer/advisory/hard-guard architecture
    eca_observer_only: bool = True           # ECA has no execution authority
    eca_soft_advisory_enabled: bool = True   # compute advisory hints
    eca_soft_hint_inject_enabled: bool = True  # inject soft hints into analyzer prompt
    eca_observer_compact_advisory_enabled: bool = False  # inject typed compact advisory into default prompt
    eca_hard_guard_enabled: bool = True      # evaluate hard-guard conditions
    eca_hard_guard_live_mode: str = "off"    # off | deadloop_only | full
    eca_runtime_micro_probe_live_enabled: bool = False  # inject runtime micro-probe actions
    eca_runtime_protected_hedge_live_enabled: bool = False  # allow protected-hedge claim/reclaim runtime probes
    eca_soft_hint_max_items: int = 2
    eca_soft_hint_board_change_no_score_threshold: int = 4
    eca_soft_hint_same_family_threshold: int = 4
    eca_soft_hint_repeat_state_threshold: int = 2
    eca_soft_hint_no_change_threshold: int = 4
    eca_soft_hint_discomfort_threshold: float = 0.6
    eca_typed_lock_gate_enabled: bool = False    # typed lock gate OFF
    eca_require_example_grounding: bool = False  # example grounding OFF
    eca_no_target_from_gate_failure: bool = False  # no-target from gate OFF
    eca_protected_hedge_authoritative: bool = False  # hedge is observability only
    eca_shared_tail_rerank_authoritative: bool = False  # shared-tail rerank OFF
    eca_successor_authoritative: bool = False  # successor slate/handoff OFF
    # v10.1: HINT/FOCUS/COMMIT thresholds
    hint_focus_evidence_count: int = 2       # evidence turns to promote HINT→FOCUS
    focus_commit_requires_score: bool = True  # COMMIT only after score_delta > 0
    board_change_no_score_churn_limit: int = 5  # same family/region churn → demotion
    hard_guard_repeat_state_threshold: int = 2  # exact repeat states → hard stop
    hard_guard_dead_anchor_revisit: bool = True
    hard_guard_click_exhausted_retry: bool = True
    weights: DriveWeights = field(default_factory=DriveWeights)
    load_bundle_path: Path | None = None
    save_bundle_path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "ECAConfig":
        if not raw:
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            mode=str(raw.get("mode", "online") or "online"),
            threshold_high=float(raw.get("threshold_high", raw.get("threshold", 0.62))),
            threshold_spike=float(raw.get("threshold_spike", 0.18)),
            cooldown_steps=max(0, int(raw.get("cooldown_steps", raw.get("cooldown", 2)))),
            max_interrupts=max(0, int(raw.get("max_interrupts", 6))),
            max_interrupts_per_level=max(
                0,
                int(raw.get("max_interrupts_per_level", raw.get("max_interrupts", 2))),
            ),
            high_drive_threshold=float(raw.get("high_drive_threshold", 0.72)),
            stagnation_window=max(1, int(raw.get("stagnation_window", 3))),
            min_commit_steps=max(0, int(raw.get("min_commit_steps", 3))),
            hard_world_threshold=float(raw.get("hard_world_threshold", 0.78)),
            hard_self_threshold=float(raw.get("hard_self_threshold", 0.72)),
            base_gate_threshold=float(raw.get("base_gate_threshold", 0.70)),
            soft_trigger_k=max(0, int(raw.get("soft_trigger_k", 99))),
            max_queue_waste_for_soft=_clamp01(float(raw.get("max_queue_waste_for_soft", 0.05))),
            stable_plan_size=max(1, int(raw.get("stable_plan_size", 6))),
            repair_plan_size=max(1, int(raw.get("repair_plan_size", 3))),
            stable_target_blacklist_steps=max(0, int(raw.get("stable_target_blacklist_steps", 8))),
            stable_target_repeat_limit=max(1, int(raw.get("stable_target_repeat_limit", 2))),
            stable_action_family_blacklist_steps=max(0, int(raw.get("stable_action_family_blacklist_steps", 6))),
            stable_control_latch_steps=max(0, int(raw.get("stable_control_latch_steps", 10))),
            stable_control_latch_release_changed_moves=max(
                0,
                int(raw.get("stable_control_latch_release_changed_moves", 0)),
            ),
            stable_control_latch_release_move_contradiction_streak=max(
                0,
                int(raw.get("stable_control_latch_release_move_contradiction_streak", 2)),
            ),
            random_interrupt_probability=_clamp01(float(raw.get("random_interrupt_probability", 0.0))),
            random_seed=int(raw.get("random_seed", 0)),
            interrupt_drives=_parse_drive_list(raw.get("interrupt_drives")),
            hard_interrupt_drives=_parse_drive_list(raw.get("hard_interrupt_drives", ("world", "self"))),
            soft_watch_drives=_parse_drive_list(raw.get("soft_watch_drives", ("structure", "progress"))),
            soft_interrupt_limit=max(0, int(raw.get("soft_interrupt_limit", 1))),
            judge_policy=str(raw.get("judge_policy", "repair_or_high_anxiety") or "repair_or_high_anxiety"),
            structural_guidance_enabled=bool(raw.get("structural_guidance_enabled", True)),
            structural_time_shadow_enabled=bool(raw.get("structural_time_shadow_enabled", False)),
            shadow_promotion_min_turns=max(1, int(raw.get("shadow_promotion_min_turns", 5))),
            shadow_promotion_no_progress_window=max(1, int(raw.get("shadow_promotion_no_progress_window", 3))),
            shadow_promotion_mid_late_min_turns=max(1, int(raw.get("shadow_promotion_mid_late_min_turns", 3))),
            shadow_promotion_mid_late_no_progress_window=max(1, int(raw.get("shadow_promotion_mid_late_no_progress_window", 2))),
            hedge_regeneration_none_run_threshold=max(1, int(raw.get("hedge_regeneration_none_run_threshold", 4))),
            bootstrap_regeneration_min_step=max(1, int(raw.get("bootstrap_regeneration_min_step", 8))),
            bootstrap_regeneration_threshold_multiplier=max(1, int(raw.get("bootstrap_regeneration_threshold_multiplier", 2))),
            contract_health_window=max(4, int(raw.get("contract_health_window", 12))),
            contract_health_threshold=_clamp01(float(raw.get("contract_health_threshold", 0.25))),
            contract_health_min_step=max(1, int(raw.get("contract_health_min_step", 20))),
            phase_handoff_carry_size=max(1, int(raw.get("phase_handoff_carry_size", 3))),
            phase_handoff_resurface_ttl=max(1, int(raw.get("phase_handoff_resurface_ttl", 25))),
            selected_target_promotion_min_repeats=max(1, int(raw.get("selected_target_promotion_min_repeats", 2))),
            selected_target_promotion_window=max(2, int(raw.get("selected_target_promotion_window", 6))),
            regen_success_window=max(1, int(raw.get("regen_success_window", 5))),
            regen_success_window_long=max(1, int(raw.get("regen_success_window_long", 10))),
            health_debt_threshold=max(0.1, float(raw.get("health_debt_threshold", 1.5))),
            health_debt_floor=_clamp01(float(raw.get("health_debt_floor", 0.25))),
            health_debt_decay_on_recovery=max(0.0, float(raw.get("health_debt_decay_on_recovery", 0.5))),
            health_debt_refractory_steps=max(1, int(raw.get("health_debt_refractory_steps", 6))),
            provisional_hedge_ttl=max(1, int(raw.get("provisional_hedge_ttl", 15))),
            provisional_hedge_confirm_repeats=max(1, int(raw.get("provisional_hedge_confirm_repeats", 2))),
            provisional_hedge_confirm_window=max(2, int(raw.get("provisional_hedge_confirm_window", 8))),
            # CH v5 support score
            support_score_decay=max(0.5, min(0.99, float(raw.get("support_score_decay", 0.92)))),
            support_score_max_ledger=max(4, int(raw.get("support_score_max_ledger", 12))),
            support_score_recent_window=max(4, int(raw.get("support_score_recent_window", 10))),
            support_bonus_visible=max(0.0, float(raw.get("support_bonus_visible", 1.0))),
            support_bonus_lagged5=max(0.0, float(raw.get("support_bonus_lagged5", 1.5))),
            support_bonus_lagged10=max(0.0, float(raw.get("support_bonus_lagged10", 0.7))),
            support_bonus_streak=max(0.0, float(raw.get("support_bonus_streak", 0.25))),
            support_bonus_ch_gain=max(0.0, float(raw.get("support_bonus_ch_gain", 0.5))),
            support_penalty_contradiction=max(0.0, float(raw.get("support_penalty_contradiction", 2.5))),
            support_penalty_expire=max(0.0, float(raw.get("support_penalty_expire", 1.0))),
            support_promote_to_provisional=max(0.5, float(raw.get("support_promote_to_provisional", 1.8))),
            support_promote_to_protected=max(1.0, float(raw.get("support_promote_to_protected", 3.5))),
            support_provisional_hits_min=max(1, int(raw.get("support_provisional_hits_min", 2))),
            support_protected_hits_min=max(1, int(raw.get("support_protected_hits_min", 2))),
            support_protected_lagged5_min=max(0, int(raw.get("support_protected_lagged5_min", 1))),
            support_protected_streak_min=max(1, int(raw.get("support_protected_streak_min", 2))),
            support_provisional_ttl_initial=max(2, int(raw.get("support_provisional_ttl_initial", 8))),
            support_provisional_ttl_extend=max(1, int(raw.get("support_provisional_ttl_extend", 3))),
            support_provisional_ttl_cap=max(4, int(raw.get("support_provisional_ttl_cap", 14))),
            support_carry_top_k=max(1, int(raw.get("support_carry_top_k", 3))),
            support_carry_conf_coeff=max(0.0, float(raw.get("support_carry_conf_coeff", 0.4))),
            support_carry_conf_cap=max(0.0, float(raw.get("support_carry_conf_cap", 1.5))),
            support_carry_auto_provisional_threshold=max(0.0, float(raw.get("support_carry_auto_provisional_threshold", 4.0))),
            support_rehit_window=max(2, int(raw.get("support_rehit_window", 8))),
            support_carry_survival_window=max(4, int(raw.get("support_carry_survival_window", 20))),
            frontier_guidance_enabled=bool(raw.get("frontier_guidance_enabled", False)),
            force_post_reclaim_hold_on_nonweak_reclaim=bool(
                raw.get("force_post_reclaim_hold_on_nonweak_reclaim", False)
            ),
            force_post_final_hold_runtime_packet_after_one_step=bool(
                raw.get("force_post_final_hold_runtime_packet_after_one_step", False)
            ),
            force_post_final_shared_rerank_on_first_visible_step=bool(
                raw.get("force_post_final_shared_rerank_on_first_visible_step", False)
            ),
            synthesize_pending_shared_tail_sample_on_first_visible_step=bool(
                raw.get(
                    "synthesize_pending_shared_tail_sample_on_first_visible_step",
                    False,
                )
            ),
            disable_psa_in_final_reclaimable=bool(
                raw.get("disable_psa_in_final_reclaimable", False)
            ),
            weights=DriveWeights.from_mapping(raw.get("weights")),
            load_bundle_path=Path(raw["load_bundle_path"]) if raw.get("load_bundle_path") else None,
            save_bundle_path=Path(raw["save_bundle_path"]) if raw.get("save_bundle_path") else None,
            # v9: authority backoff
            authority_decay_per_no_value=float(raw.get("authority_decay_per_no_value", 0.08)),
            authority_recovery_per_hit=float(raw.get("authority_recovery_per_hit", 0.15)),
            authority_floor=max(0.0, float(raw.get("authority_floor", 0.2))),
            authority_backoff_window=max(1, int(raw.get("authority_backoff_window", 8))),
            authority_min_steps_before_decay=max(0, int(raw.get("authority_min_steps_before_decay", 5))),
            # v10.1: observer/advisory/hard-guard
            eca_observer_only=bool(raw.get("eca_observer_only", True)),
            eca_soft_advisory_enabled=bool(raw.get("eca_soft_advisory_enabled", True)),
            eca_soft_hint_inject_enabled=bool(raw.get("eca_soft_hint_inject_enabled", True)),
            eca_observer_compact_advisory_enabled=bool(
                raw.get("eca_observer_compact_advisory_enabled", False)
            ),
            eca_hard_guard_enabled=bool(raw.get("eca_hard_guard_enabled", True)),
            eca_hard_guard_live_mode=str(raw.get("eca_hard_guard_live_mode", "off") or "off"),
            eca_runtime_micro_probe_live_enabled=bool(raw.get("eca_runtime_micro_probe_live_enabled", False)),
            eca_runtime_protected_hedge_live_enabled=bool(
                raw.get(
                    "eca_runtime_protected_hedge_live_enabled",
                    raw.get("eca_runtime_micro_probe_live_enabled", False),
                )
            ),
            eca_soft_hint_max_items=max(1, int(raw.get("eca_soft_hint_max_items", 2))),
            eca_soft_hint_board_change_no_score_threshold=max(
                1, int(raw.get("eca_soft_hint_board_change_no_score_threshold", 4))
            ),
            eca_soft_hint_same_family_threshold=max(1, int(raw.get("eca_soft_hint_same_family_threshold", 4))),
            eca_soft_hint_repeat_state_threshold=max(1, int(raw.get("eca_soft_hint_repeat_state_threshold", 2))),
            eca_soft_hint_no_change_threshold=max(1, int(raw.get("eca_soft_hint_no_change_threshold", 4))),
            eca_soft_hint_discomfort_threshold=float(raw.get("eca_soft_hint_discomfort_threshold", 0.6)),
            eca_typed_lock_gate_enabled=bool(raw.get("eca_typed_lock_gate_enabled", False)),
            eca_require_example_grounding=bool(raw.get("eca_require_example_grounding", False)),
            eca_no_target_from_gate_failure=bool(raw.get("eca_no_target_from_gate_failure", False)),
            eca_protected_hedge_authoritative=bool(raw.get("eca_protected_hedge_authoritative", False)),
            eca_shared_tail_rerank_authoritative=bool(raw.get("eca_shared_tail_rerank_authoritative", False)),
            eca_successor_authoritative=bool(raw.get("eca_successor_authoritative", False)),
            hint_focus_evidence_count=max(1, int(raw.get("hint_focus_evidence_count", 2))),
            focus_commit_requires_score=bool(raw.get("focus_commit_requires_score", True)),
            board_change_no_score_churn_limit=max(1, int(raw.get("board_change_no_score_churn_limit", 5))),
            hard_guard_repeat_state_threshold=max(1, int(raw.get("hard_guard_repeat_state_threshold", 2))),
            hard_guard_dead_anchor_revisit=bool(raw.get("hard_guard_dead_anchor_revisit", True)),
            hard_guard_click_exhausted_retry=bool(raw.get("hard_guard_click_exhausted_retry", True)),
        )


@dataclass
class PlanContract:
    structure_question: str = ""
    local_target: str = ""
    keep_condition: str = ""
    drop_condition: str = ""
    why_this_plan: str = ""
    contradiction: str = ""
    local_hypothesis: str = ""
    confirm_if: str = ""
    reject_if: str = ""
    # Typed lock gate fields (v8)
    hypothesis_rule: str = ""
    target_object_or_region: str = ""
    expected_board_delta_within_2_steps: str = ""
    disconfirm_signal: str = ""
    rule_family: str = ""
    confidence: str = ""
    # Structured lock contract fields (v8.3)
    target_ref: str = ""  # selected_target | protected_hedge | component_id | anchor_key
    rule_payload: str = ""
    expected_delta_type: str = ""
    expected_delta_payload: str = ""
    # v8.4: grounded hypothesis generator fields
    target_slot: str = ""  # T0..T4 slot reference resolved by runtime
    rule_args: str = ""  # structured rule arguments (e.g. "direction=right, distance=1")
    expected_bbox: str = ""  # expected bounding box after delta
    expected_color_add: str = ""  # colors expected to appear
    expected_color_remove: str = ""  # colors expected to disappear
    expected_component_count_delta: str = ""  # e.g. "+1", "-1", "0"
    expected_cell_count_range: str = ""  # e.g. "3-5"
    supporting_example_ids: str = ""  # comma-separated train pair indices (e.g. "0,1")
    example_relation_type: str = ""  # enum: count_relation|color_mapping|relative_position|symmetry_relation|containment_or_bbox|copy_or_transfer
    why_this_target_matches_example: str = ""  # one sentence grounding
    # v8.6: task-grounded hypothesis slate
    task_hypothesis_id: str = ""  # ID from task_hypothesis_slate
    # v9: observable contract fields (minimum viable contract for HINT+)
    focus_region: str = ""                          # grid region the next actions target
    expected_observable_within_2_steps: str = ""     # what should visibly change
    disconfirm_cue: str = ""                        # observable that disproves hypothesis

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskHypothesis:
    """A single hypothesis from the task_hypothesis_slate (v8.6)."""
    task_hypothesis_id: str = ""
    support_pair_ids: str = ""  # comma-separated visual region indices
    rule_family: str = ""
    target_selector: str = ""  # description of what to target
    operator: str = ""  # what transformation to apply
    expected_test_delta: str = ""  # expected effect on the test board
    disconfirm_signal: str = ""  # what proves this wrong

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalyzerConstraints:
    required_action_family: ActionFamily = ActionFamily.UNKNOWN
    blacklisted_action_families: list[ActionFamily] = field(default_factory=list)
    blacklisted_targets: list[str] = field(default_factory=list)
    current_structure_question: str = ""
    current_local_target: str = ""
    keep_condition: str = ""
    drop_condition: str = ""
    suggested_local_test: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_action_family": self.required_action_family.value,
            "blacklisted_action_families": [item.value for item in self.blacklisted_action_families],
            "blacklisted_targets": list(self.blacklisted_targets),
            "current_structure_question": self.current_structure_question,
            "current_local_target": self.current_local_target,
            "keep_condition": self.keep_condition,
            "drop_condition": self.drop_condition,
            "suggested_local_test": self.suggested_local_test,
        }


@dataclass
class ControlHypothesisLatch:
    active: bool = False
    required_action_family: ActionFamily = ActionFamily.UNKNOWN
    activation_reason: str = ""
    ttl_steps: int = 0
    release_on_board_change: bool = True
    release_on_score_change: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": bool(self.active),
            "required_action_family": self.required_action_family.value,
            "activation_reason": self.activation_reason,
            "ttl_steps": int(self.ttl_steps),
            "release_on_board_change": bool(self.release_on_board_change),
            "release_on_score_change": bool(self.release_on_score_change),
        }


@dataclass
class CompactRecentAction:
    action: str
    changed: bool
    contradiction: bool
    score_delta: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompactWorkingMemory:
    level: int
    score: float
    top_drive: str
    stagnant_steps: int
    no_change_streak: int
    repeated_state_count: int
    current_structure_question: str
    current_local_target: str
    keep_condition: str
    drop_condition: str
    required_action_family: str
    blacklisted_action_families: list[str]
    blacklisted_targets: list[str]
    suggested_local_test: str
    structural_guidance_enabled: bool = True
    structural_guidance_mode: str = "active"
    structural_guidance_shadow_reason: str = ""
    frontier_guidance_enabled: bool = False
    active_frontier_id: str = ""
    active_frontier_question: str = ""
    active_frontier_status: str = ""
    active_frontier_family_bias: str = ""
    active_frontier_progress_score: float = 0.0
    active_frontier_uncertainty: float = 0.0
    active_frontier_linked_anchor_count: int = 0
    frontier_promotion_goal: str = ""
    frontier_open_count: int = 0
    frontier_saturated_count: int = 0
    frontier_last_promotion_outcome: str = ""
    structural_decomposition_summary: str = ""
    structural_plan_summary: str = ""
    structural_local_window_summary: str = ""
    structural_psa_dominant_type: str = ""
    structural_psa_route: str = ""
    structural_psa_urgency: float = 0.0
    structural_anxiety_total: float = 0.0
    structural_deficit_miss: float = 0.0
    structural_deficit_conf: float = 0.0
    structural_deficit_unk: float = 0.0
    structural_deficit_reach: float = 0.0
    structural_deficit_recur: float = 0.0
    structural_deficit_loc: float = 0.0
    self_location_mode: str = ""
    self_location_focus: str = ""
    self_location_reachability: float = 0.0
    self_location_resource_pressure: float = 0.0
    self_location_loc_conf: float = 0.0
    self_location_kuhcp_distribution: str = ""
    stable_target_repeat_count: int = 0
    target_rotation_reason: str = ""
    fresh_target_after_rotation: bool = False
    click_family_exhausted_current_target: bool = False
    current_target_click_failures: int = 0
    current_target_move_churn: int = 0
    current_target_weak_move_directions: int = 0
    target_click_probe_budget: int = 0
    target_move_probe_budget: int = 0
    target_anchor_locked: bool = False
    current_target_viability_score: float = 0.0
    current_target_viability_reason: str = ""
    current_target_psa_dead_end: bool = False
    current_target_psa_state: str = ""
    current_target_verify_budget: int = 0
    current_target_lifecycle_state: str = ""
    current_target_lifecycle_reason: str = ""
    target_transition_from: str = ""
    target_transition_reason: str = ""
    target_transition_to: str = ""
    selected_target_id: str = ""
    selected_target_anchor: str = ""
    selected_target_source: str = ""
    selected_target_admission_score: float = 0.0
    selected_target_reason: str = ""
    selected_target_click_budget: int = 0
    selected_target_move_probe_budget: int = 0
    selected_target_runtime_probe_ok: bool = False
    selected_target_confidence: str = ""
    candidate_pool_size: int = 0
    candidate_top2_summary: str = ""
    target_turnover_penalty: float = 0.0
    effective_local_contract_source: str = ""
    effective_local_contract_component_id: str = ""
    effective_local_contract_anchor: str = ""
    no_target_contract_allowed: bool = False
    preserved_alternative_visible: bool = False
    preserved_alternative_visible_kind: str = ""
    preserved_alternative_visible_summary: str = ""
    runtime_micro_probe_continuation_score: float = 0.0
    runtime_micro_probe_continuation_threshold: float = 0.0
    runtime_micro_probe_followup_open: bool = False
    runtime_micro_probe_followup_candidate: bool = False
    runtime_micro_probe_followup_suppressed: bool = False
    runtime_micro_probe_suppression_disposition: str = ""
    post_packet_resolution: str = ""
    successor_handoff_active: bool = False
    successor_handoff_state: str = ""
    successor_handoff_from_component: str = ""
    successor_handoff_from_anchor: str = ""
    successor_handoff_to_component: str = ""
    successor_handoff_to_anchor: str = ""
    successor_handoff_source: str = ""
    successor_handoff_family_bias: str = ""
    successor_handoff_probe_kind: str = ""
    successor_handoff_confidence: str = ""
    successor_handoff_reason: str = ""
    successor_handoff_ttl_remaining: int = 0
    successor_slate_size: int = 0
    successor_slate_top2_summary: str = ""
    successor_slate_late_guard_active: bool = False
    successor_slate_protected_hedge_active: bool = False
    successor_slate_protected_hedge_visible: bool = False
    successor_slate_protected_hedge_state: str = ""
    successor_slate_protected_hedge_summary: str = ""
    successor_slate_protected_hedge_runtime_claim_state: str = ""
    successor_slate_protected_hedge_runtime_claim_rearm_reason: str = ""
    successor_slate_protected_hedge_runtime_claim_active: bool = False
    selected_target_transfer_hint_bonus: float = 0.0
    recent_actions: list[CompactRecentAction] = field(default_factory=list)
    # Phase 5: compact IR fields for sparse Analyzer input
    control_ir_summary: str = ""          # human-readable summary of active IR tickets
    control_ir_escalation_reason: str = ""  # why LocalController escalated to Analyzer
    target_health_state: str = ""          # TargetHealth enum value (live/stale/saturated/blocked/empty)
    candidate_pool_base_size: int = 0      # CandidateExtractor base pool size

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["recent_actions"] = [item.to_dict() for item in self.recent_actions]
        return payload


@dataclass
class JudgeDecision:
    accepted: bool
    reject_reasons: list[str] = field(default_factory=list)
    family_mismatch: bool = False
    local_test_missing: bool = False
    multiple_questions: bool = False
    vague_target: bool = False
    missing_keep_drop: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LatchEvent:
    kind: Literal["activated", "extended", "released"]
    reason: str
    required_action_family: str
    ttl_steps: int
    step_idx: int
    level_idx: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Hypothesis:
    kind: str
    description: str
    confidence: float = 0.0
    status: str = "open"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuralGraphNode:
    node_id: str
    label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuralGraphEdge:
    edge_id: str
    source: str
    target: str
    relation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuralElementMeta:
    type: str
    conf: float
    status: str
    salience: float = 0.0
    last_tested: int = -1
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["conf"] = round(float(self.conf), 6)
        payload["salience"] = round(float(self.salience), 6)
        return payload


@dataclass
class StructuralDecomposition:
    nodes: list[StructuralGraphNode] = field(default_factory=list)
    edges: list[StructuralGraphEdge] = field(default_factory=list)
    meta: dict[str, StructuralElementMeta] = field(default_factory=dict)
    frontier_unknown_ids: list[str] = field(default_factory=list)

    def status_histogram(self) -> dict[str, int]:
        histogram = {"K": 0, "H": 0, "U": 0, "C": 0, "P": 0}
        for item in self.meta.values():
            key = str(item.status or "").upper()
            if key in histogram:
                histogram[key] += 1
        return histogram

    def summary(self) -> str:
        histogram = self.status_histogram()
        return (
            f"nodes={len(self.nodes)} edges={len(self.edges)} "
            f"frontier_unknown={len(self.frontier_unknown_ids)} "
            f"K/H/U/C/P={histogram['K']}/{histogram['H']}/{histogram['U']}/{histogram['C']}/{histogram['P']}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
            "meta": {key: value.to_dict() for key, value in self.meta.items()},
            "frontier_unknown_ids": list(self.frontier_unknown_ids),
            "summary": self.summary(),
        }


@dataclass
class StructuralRequirement:
    requirement_id: str
    scope: str
    predicate: str
    priority: float
    repair_ops: list[str] = field(default_factory=list)
    satisfied: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["priority"] = round(float(self.priority), 6)
        return payload


@dataclass
class StructuralPlan:
    requirements: list[StructuralRequirement] = field(default_factory=list)

    def summary(self) -> str:
        if not self.requirements:
            return "requirements=0"
        ordered = sorted(
            self.requirements,
            key=lambda item: (-float(item.priority or 0.0), str(item.requirement_id or "")),
        )
        top = []
        for item in ordered[:2]:
            state = "ok" if item.satisfied else ("miss" if item.satisfied is False else "open")
            top.append(f"{item.requirement_id}:{state}:{float(item.priority or 0.0):.2f}")
        return f"requirements={len(self.requirements)} top={'; '.join(top)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirements": [item.to_dict() for item in self.requirements],
            "summary": self.summary(),
        }


@dataclass
class SelfBodyCoordinate:
    mode: str = "execute"
    affordances: list[str] = field(default_factory=list)
    resource: dict[str, float] = field(default_factory=dict)
    risk: float = 0.0
    reachability: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "affordances": list(self.affordances),
            "resource": {key: round(float(value), 6) for key, value in self.resource.items()},
            "risk": round(float(self.risk), 6),
            "reachability": round(float(self.reachability), 6),
        }


@dataclass
class SelfRuleCoordinate:
    focus: str = ""
    kuhcp_distribution: dict[str, float] = field(default_factory=dict)
    frontier: list[str] = field(default_factory=list)
    active_hypothesis: str = ""
    loc_conf: float = 0.0

    def distribution_summary(self) -> str:
        keys = ("K", "H", "U", "C", "P")
        return "K/H/U/C/P=" + "/".join(
            f"{float(self.kuhcp_distribution.get(key, 0.0) or 0.0):.2f}" for key in keys
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus": self.focus,
            "kuhcp_distribution": {
                key: round(float(value), 6) for key, value in self.kuhcp_distribution.items()
            },
            "frontier": list(self.frontier),
            "active_hypothesis": self.active_hypothesis,
            "loc_conf": round(float(self.loc_conf), 6),
            "distribution_summary": self.distribution_summary(),
        }


@dataclass
class SelfLocation:
    body: SelfBodyCoordinate = field(default_factory=SelfBodyCoordinate)
    rule: SelfRuleCoordinate = field(default_factory=SelfRuleCoordinate)

    def summary(self) -> str:
        return (
            f"mode={self.body.mode} focus={self.rule.focus or '(none)'} "
            f"reach={float(self.body.reachability):.2f} loc_conf={float(self.rule.loc_conf):.2f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "body": self.body.to_dict(),
            "rule": self.rule.to_dict(),
            "summary": self.summary(),
        }


@dataclass
class StructuralLocalWindow:
    focus: str = ""
    scope_requirement_ids: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    reachable_node_ids: list[str] = field(default_factory=list)
    frontier_unknown_ids: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuralDeficitVector:
    miss: float = 0.0
    conf: float = 0.0
    unk: float = 0.0
    reach: float = 0.0
    recur: float = 0.0
    loc: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "miss": round(float(self.miss), 6),
            "conf": round(float(self.conf), 6),
            "unk": round(float(self.unk), 6),
            "reach": round(float(self.reach), 6),
            "recur": round(float(self.recur), 6),
            "loc": round(float(self.loc), 6),
        }


@dataclass
class StructuralAnxietySignal:
    deficits: StructuralDeficitVector = field(default_factory=StructuralDeficitVector)
    dominant_type: str = ""
    urgency: float = 0.0
    route: str = ""
    total: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    route_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "deficits": self.deficits.to_dict(),
            "dominant_type": self.dominant_type,
            "urgency": round(float(self.urgency), 6),
            "route": self.route,
            "total": round(float(self.total), 6),
            "weights": {key: round(float(value), 6) for key, value in self.weights.items()},
            "route_reason": self.route_reason,
        }


@dataclass
class PromotionEvent:
    anchor_key: str
    outcome: str
    family_used: str
    board_changed: bool = False
    score_changed: bool = False
    transfer_hint: str = ""
    psa_reason: str = ""
    lifecycle_state: str = ""
    frontier_id: str = ""
    question: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrontierNode:
    frontier_id: str
    region_key: str
    question: str
    family_bias: str
    status: str
    uncertainty: float
    progress_score: float
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    linked_anchors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrontierContract:
    frontier_id: str = ""
    question: str = ""
    local_target_hint: str = ""
    allowed_action_family: str = ""
    stop_condition: str = ""
    promotion_goal: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TargetCandidate:
    candidate_id: str
    level_idx: int
    component_id: str | None
    anchor_key: str
    center_row: int
    center_col: int
    source: CandidateSource
    family: str
    novelty_score: float = 0.0
    relation_potential: float = 0.0
    board_change_prior: float = 0.0
    contradiction_probe_value: float = 0.0
    move_transfer_score: float = 0.0
    dead_anchor_penalty: float = 0.0
    family_redundancy_penalty: float = 0.0
    turnover_penalty: float = 0.0
    low_yield_region_penalty: float = 0.0
    admission_score: float = 0.0
    rejected: bool = False
    rejected_reason: CandidateRejectReason | None = None
    suggested_probe_reason: str | None = None
    click_budget: int = 0
    move_probe_budget: int = 1
    one_action_runtime_probe_ok: bool = False
    transfer_hint_bonus: float = 0.0
    runtime_family_variant_kind: str = ""
    transition_model_candidate_family: str = ""
    transition_model_lookup_candidate_family: str = ""
    transition_model_backoff_level: str = ""
    transition_model_support: int = 0
    transition_model_p_contract: float = 0.0
    transition_model_p_frontier: float = 0.0
    transition_model_p_turnover: float = 0.0
    transition_model_p_dead_churn: float = 0.0
    transition_model_p_slip: float = 0.0
    transition_model_quality_gain: float = 0.0
    transition_model_predicted_anxiety: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"] = self.source.value
        payload["rejected_reason"] = self.rejected_reason.value if self.rejected_reason else None
        return payload


@dataclass
class SelectedTargetSummary:
    candidate_id: str
    anchor_key: str
    source: str
    admission_score: float
    suggested_probe_reason: str
    click_budget: int
    move_probe_budget: int
    rotate_cost: float
    confidence: str = ""
    runtime_probe_ok: bool = False
    component_id: str = ""
    transfer_hint_bonus: float = 0.0
    second_chance_kind: str = ""
    recovery_after_state: str = ""
    recovery_after_reason: str = ""
    transition_model_candidate_family: str = ""
    transition_model_backoff_level: str = ""
    transition_model_support: int = 0
    transition_model_p_contract: float = 0.0
    transition_model_p_frontier: float = 0.0
    transition_model_p_turnover: float = 0.0
    transition_model_p_dead_churn: float = 0.0
    transition_model_p_slip: float = 0.0
    transition_model_quality_gain: float = 0.0
    transition_model_predicted_anxiety: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EscapeReservation:
    candidate_id: str
    component_id: str
    anchor_key: str
    source: str
    family: str
    admission_score: float
    score_gap_to_normal_threshold: float
    observed_dead_tail_ratio: float
    reservation_reason: str
    reserved_step_idx: int
    expires_at: int
    click_budget: int = 0
    move_probe_budget: int = 1
    suggested_probe_reason: str = ""
    confidence: str = "low"
    level_idx: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuccessorHandoff:
    from_candidate_id: str
    from_component_id: str
    from_anchor_key: str
    to_candidate_id: str
    to_component_id: str
    to_anchor_key: str
    source: str
    family_bias: str
    probe_kind: str
    confidence: str
    score_gap_to_normal_threshold: float
    observed_dead_tail_ratio: float
    handoff_reason: str
    state: str = "ARMED"
    invalidators: list[str] = field(default_factory=list)
    created_step_idx: int = 0
    peek_used_step_idx: int = -1
    deferred_activated_step_idx: int = -1
    survived_one_turn: bool = False
    expires_at: int = 0
    move_probe_budget: int = 1
    click_budget: int = 0
    suggested_probe_reason: str = ""
    level_idx: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuccessorSlateEntry:
    frontier_id: str
    component_id: str
    anchor_key: str
    center_row: int
    center_col: int
    region_hint: str
    source: str
    family_bias: str
    admission_score: float
    preservation_score: float
    live_confidence: str
    slate_reason: str
    seed_source: str = ""
    preservation_value: float = 0.0
    slot_kind: str = "general"
    protected_hedge: bool = False
    diversity_from_selected: float = 0.0
    late_recovery_potential: float = 0.0
    distinct_from_selected: bool = True
    refresh_count: int = 0
    created_step_idx: int = 0
    last_seen_alive_step: int = 0
    expires_at: int = 0
    level_idx: int = 0
    invalid_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SharedTailTransitionKey:
    tail_phase: Literal["post_final_shared"] = "post_final_shared"
    rearm_state: Literal["reached_no_rearm", "reached_rearm_live"] = "reached_rearm_live"
    visible_window: Literal["shared_only", "hedge_only", "both", "none"] = "none"
    churn_class: Literal["stable", "mixed", "frontier"] = "stable"
    contract_owner: Literal["shared", "hedge", "none", "unstable"] = "none"
    queue_health: Literal["thin", "ok", "exhausting"] = "ok"
    repeat_depth: Literal["0", "1", "2plus"] = "0"
    psa_risk: Literal["low", "mid", "high"] = "low"
    candidate_family: Literal[
        "same_contract_revisit",
        "switch_visible_shared",
        "switch_visible_hedge",
        "queue_refresh",
        "stable_probe",
        "protective_hold",
    ] = "stable_probe"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SharedTailTransitionOutcome:
    contract_survive_1: bool = False
    contract_survive_2: bool = False
    no_target_slip_1: bool = False
    psa_invalidation_1: bool = False
    frontier_gain_2: bool = False
    turnover_gain_2: bool = False
    stable_churn_return_2: bool = False
    post_final_shared_after_exit_outcome_class: str = "unobserved"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PendingSharedTailTransitionSample:
    sample_id: str
    opened_level_idx: int
    opened_step_idx: int
    open_reason: str
    key: SharedTailTransitionKey
    open_source: str = ""
    open_effective_local_contract_source: str = ""
    open_selected_target_id: str = ""
    open_queue_exhaustions: int = 0
    open_analyzer_calls: int = 0
    open_state_hash: str = ""
    open_after_exit_same_route_anchor_event_count: int = 0
    open_probe_continuity_ticket_issue_count: int = 0
    open_probe_continuity_ticket_consume_count: int = 0
    open_probe_continuity_ticket_success_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["key"] = self.key.to_dict()
        return payload


@dataclass
class ECAState:
    game_id: str | None
    step_index: int
    level_number: int
    score: int
    score_delta: int
    arc_state: str
    state_hash: str
    component_count: int
    unresolved_components: int
    unique_colors: int
    object_table: list[dict[str, Any]] = field(default_factory=list)
    structural_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    queue_remaining: int = 0
    plan_index: int = 0
    plan_total: int = 0
    current_plan: str = ""
    plan_reasoning: str = ""
    analyzer_calls: int = 0
    interrupt_count: int = 0
    remaining_actions: int = 0
    max_actions: int = 0
    no_change_streak: int = 0
    recent_failures: int = 0
    stagnant_steps: int = 0
    repeated_state_count: int = 0
    same_action_streak: int = 0
    current_state_known_actions: int = 0
    current_state_changed_actions: int = 0
    click_coverage: float = 0.0
    known_effect_ratio: float = 0.0
    plan_active: bool = False
    recent_history: list[dict[str, Any]] = field(default_factory=list)
    current_state_actions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnxietySnapshot:
    step_index: int
    level_number: int
    score: int
    score_delta: int
    state_hash: str
    total: float
    top_drive: str
    breakdown: dict[str, float]
    weights: dict[str, float]
    plan_generation: int
    episode_id: str
    queue_remaining: int
    plan_index: int
    plan_total: int
    no_change_streak: int
    stagnant_steps: int
    repeated_state_count: int
    unresolved_components: int
    analyzer_calls: int
    interrupt_count: int
    plan_active: bool
    high_anxiety: bool = False
    plan_contract: PlanContract = field(default_factory=PlanContract)
    latch: ControlHypothesisLatch = field(default_factory=ControlHypothesisLatch)
    stable_click_contradiction: bool = False
    judge_rejection_due_to_family_mismatch: bool = False
    structural_decomposition: StructuralDecomposition = field(default_factory=StructuralDecomposition)
    structural_plan: StructuralPlan = field(default_factory=StructuralPlan)
    self_location: SelfLocation = field(default_factory=SelfLocation)
    local_window: StructuralLocalWindow = field(default_factory=StructuralLocalWindow)
    structural_signal: StructuralAnxietySignal = field(default_factory=StructuralAnxietySignal)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total"] = round(float(self.total), 6)
        data["breakdown"] = {k: round(float(v), 6) for k, v in self.breakdown.items()}
        data["weights"] = {k: round(float(v), 6) for k, v in self.weights.items()}
        if isinstance(self.latch, ControlHypothesisLatch):
            data["latch"] = self.latch.to_dict()
        if isinstance(self.plan_contract, PlanContract):
            data["plan_contract"] = self.plan_contract.to_dict()
        if isinstance(self.structural_decomposition, StructuralDecomposition):
            data["structural_decomposition"] = self.structural_decomposition.to_dict()
        if isinstance(self.structural_plan, StructuralPlan):
            data["structural_plan"] = self.structural_plan.to_dict()
        if isinstance(self.self_location, SelfLocation):
            data["self_location"] = self.self_location.to_dict()
        if isinstance(self.local_window, StructuralLocalWindow):
            data["local_window"] = self.local_window.to_dict()
        if isinstance(self.structural_signal, StructuralAnxietySignal):
            data["structural_signal"] = self.structural_signal.to_dict()
        return data


@dataclass
class InterruptDecision:
    interrupt: bool
    would_interrupt: bool
    trigger: str
    reason: str
    top_drive: str
    task: str
    preferred_plan_horizon: str
    snapshot_total: float
    delta_since_plan_start: float
    mode: str = "online"
    policy: str = "eca"
    analyzer_mode: str = "stable"
    hard: bool = False
    gate: float = 0.0
    inhibit: float = 0.0
    random_value: float | None = None
    breakdown: dict[str, float] = field(default_factory=dict)
    structural_guidance_mode: str = "active"
    structural_guidance_shadow_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "interrupt": bool(self.interrupt),
            "would_interrupt": bool(self.would_interrupt),
            "trigger": self.trigger,
            "reason": self.reason,
            "top_drive": self.top_drive,
            "task": self.task,
            "preferred_plan_horizon": self.preferred_plan_horizon,
            "snapshot_total": round(float(self.snapshot_total), 6),
            "delta_since_plan_start": round(float(self.delta_since_plan_start), 6),
            "mode": self.mode,
            "policy": self.policy,
            "analyzer_mode": self.analyzer_mode,
            "hard": bool(self.hard),
            "gate": round(float(self.gate), 6),
            "inhibit": round(float(self.inhibit), 6),
            "random_value": None if self.random_value is None else round(float(self.random_value), 6),
            "breakdown": {k: round(float(v), 6) for k, v in self.breakdown.items()},
            "structural_guidance_mode": self.structural_guidance_mode,
            "structural_guidance_shadow_reason": self.structural_guidance_shadow_reason,
        }


@dataclass
class NightBundle:
    game_id: str | None = None
    version: str = "eca-v1"
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    drive_weights: dict[str, float] = field(default_factory=lambda: DriveWeights().to_dict())
    thresholds: dict[str, float] = field(default_factory=dict)
    world_model_bundle: dict[str, Any] = field(default_factory=dict)
    structure_memory_bundle: dict[str, Any] = field(default_factory=dict)
    self_model_bundle: dict[str, Any] = field(default_factory=dict)
    night_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "version": self.version,
            "updated_at": self.updated_at,
            "drive_weights": {k: round(float(v), 6) for k, v in self.drive_weights.items()},
            "thresholds": {k: round(float(v), 6) for k, v in self.thresholds.items()},
            "world_model_bundle": self.world_model_bundle,
            "structure_memory_bundle": self.structure_memory_bundle,
            "self_model_bundle": self.self_model_bundle,
            "night_summary": self.night_summary,
        }

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None, *, game_id: str | None = None) -> "NightBundle":
        if not raw:
            return cls(game_id=game_id)
        return cls(
            game_id=raw.get("game_id", game_id),
            version=str(raw.get("version", "eca-v1")),
            updated_at=str(raw.get("updated_at", datetime.now(timezone.utc).isoformat())),
            drive_weights={k: _clamp01(float(v)) for k, v in (raw.get("drive_weights") or {}).items()},
            thresholds={k: float(v) for k, v in (raw.get("thresholds") or {}).items()},
            world_model_bundle=dict(raw.get("world_model_bundle") or {}),
            structure_memory_bundle=dict(raw.get("structure_memory_bundle") or {}),
            self_model_bundle=dict(raw.get("self_model_bundle") or {}),
            night_summary=dict(raw.get("night_summary") or {}),
        )


@dataclass
class AdvisoryClaim:
    claim_type: str
    subject: str | None = None
    scope: str = "local"
    ttl_turns: int = 1
    evidence_window: int = 1
    confidence: float = 0.0
    support_predicate: str = ""
    falsifier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidatedClaim:
    claim_type: str
    subject: str | None = None
    scope: str = "local"
    status: str = "weak"
    confidence: float = 0.0
    board_support: float = 0.0
    trajectory_support: float = 0.0
    structural_support: float = 0.0
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 4)
        payload["board_support"] = round(float(self.board_support), 4)
        payload["trajectory_support"] = round(float(self.trajectory_support), 4)
        payload["structural_support"] = round(float(self.structural_support), 4)
        return payload

    def prompt_label(self) -> str:
        if self.subject:
            return f"{self.claim_type}: {self.subject}"
        return self.claim_type

    def prompt_line(self) -> str:
        note = " ".join(str(self.note or "").split())
        if note:
            return f"- {self.prompt_label()} -- {note}"
        return f"- {self.prompt_label()}"


@dataclass
class SoftAdvisory:
    """v10.1: lightweight advisory emitted by ECA observer layer.

    These are soft hints — the baseline analyzer may use or ignore them.
    """
    hints: list[str] = field(default_factory=list)
    hint_state: ECAHintState = ECAHintState.NONE
    discomfort_score: float = 0.0
    hard_guard_reason: str = ""   # non-empty only if hard guard triggered
    board_change_no_score_streak: int = 0
    same_family_no_score_streak: int = 0
    dead_anchor_warning: str = ""
    stale_family_warning: str = ""
    click_exhausted_warning: str = ""
    repeat_state_warning: str = ""
    raw_claims: list[AdvisoryClaim] = field(default_factory=list)
    validated_claims: list[ValidatedClaim] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hints": list(self.hints),
            "hint_state": self.hint_state.value,
            "discomfort_score": round(float(self.discomfort_score), 4),
            "hard_guard_reason": self.hard_guard_reason,
            "board_change_no_score_streak": self.board_change_no_score_streak,
            "same_family_no_score_streak": self.same_family_no_score_streak,
            "dead_anchor_warning": self.dead_anchor_warning,
            "stale_family_warning": self.stale_family_warning,
            "click_exhausted_warning": self.click_exhausted_warning,
            "repeat_state_warning": self.repeat_state_warning,
            "raw_claims": [item.to_dict() for item in self.raw_claims],
            "validated_claims": [item.to_dict() for item in self.validated_claims],
        }

    def claim_status_histogram(self) -> dict[str, int]:
        histogram: dict[str, int] = {}
        for claim in self.validated_claims:
            key = str(claim.status or "weak")
            histogram[key] = histogram.get(key, 0) + 1
        return histogram

    def claim_type_status_histogram(self) -> dict[str, int]:
        histogram: dict[str, int] = {}
        for claim in self.validated_claims:
            key = f"{str(claim.claim_type or '')}:{str(claim.status or 'weak')}"
            histogram[key] = histogram.get(key, 0) + 1
        return histogram

    def prompt_claims(self) -> list[ValidatedClaim]:
        return [
            claim
            for claim in self.validated_claims
            if str(claim.status or "") in {"supported", "weak"}
        ]

    def format_verified_for_prompt(self) -> str:
        if not self.validated_claims:
            return ""

        supported = [
            claim
            for claim in self.validated_claims
            if str(claim.status or "") == "supported"
        ]
        weak = [
            claim
            for claim in self.validated_claims
            if str(claim.status or "") == "weak"
        ]
        suppressed = [
            claim
            for claim in self.validated_claims
            if str(claim.status or "") in {"contradicted", "stale"}
        ]
        lines = [
            "[ECA_VERIFIED_ADVISORY]",
            "Use only these validated local claims as advisory priors.",
            "Do not infer a larger shortest-path story from them.",
        ]
        if supported:
            lines.extend(["", "supported:"])
            lines.extend(claim.prompt_line() for claim in supported)
        if weak:
            lines.extend(["", "weak:"])
            lines.extend(claim.prompt_line() for claim in weak)
        if suppressed:
            lines.extend(["", "suppressed:"])
            for claim in suppressed:
                reason = " ".join(str(claim.note or "").split()) or str(claim.status or "")
                lines.append(f"- {claim.prompt_label()}: {claim.status} -- {reason}")
        return "\n".join(lines)

    def format_for_prompt(self) -> str:
        """Format as natural-language soft hints for the analyzer prompt."""
        if not self.hints:
            return ""
        lines = ["[ECA SOFT HINTS]"]
        for hint in self.hints:
            lines.append(f"- {hint}")
        lines.append("- These are advisory only. If board evidence suggests a better plan, ignore them.")
        return "\n".join(lines)

    def _build_signal_entries(self) -> list[dict[str, Any]]:
        def _signal(
            *,
            signal_id: str,
            kind: str,
            strength: float,
            scope: str,
            ttl_turns: int,
            bias: str,
            text: str,
            evidence: str = "",
        ) -> dict[str, Any]:
            return {
                "id": signal_id,
                "kind": kind,
                "strength": round(max(0.0, min(1.0, float(strength or 0.0))), 2),
                "scope": scope,
                "ttl_turns": max(1, int(ttl_turns or 1)),
                "bias": bias,
                "text": " ".join(str(text or "").split()),
                "evidence": " ".join(str(evidence or "").split()),
            }

        signals: list[dict[str, Any]] = []
        if self.dead_anchor_warning:
            signals.append(
                _signal(
                    signal_id="dead_anchor_warning",
                    kind="attention",
                    strength=0.92,
                    scope="local",
                    ttl_turns=2,
                    bias="prepare_handoff",
                    text="The current anchor has prior dead-end evidence and may not deserve renewed commitment.",
                    evidence=self.dead_anchor_warning,
                )
            )
        if self.click_exhausted_warning:
            signals.append(
                _signal(
                    signal_id="click_family_exhausted",
                    kind="tension",
                    strength=0.88,
                    scope="local",
                    ttl_turns=1,
                    bias="avoid_widening",
                    text="The current click-family line looks locally exhausted.",
                    evidence=self.click_exhausted_warning,
                )
            )
        if self.board_change_no_score_streak > 0:
            signals.append(
                _signal(
                    signal_id="board_change_no_score",
                    kind="tension",
                    strength=min(0.95, 0.55 + 0.05 * float(self.board_change_no_score_streak)),
                    scope="local",
                    ttl_turns=2,
                    bias="lower_confidence",
                    text="Recent local change may be churn rather than real progress.",
                    evidence=(
                        f"Board changed {int(self.board_change_no_score_streak)} time(s) without score."
                    ),
                )
            )
        if self.stale_family_warning:
            signals.append(
                _signal(
                    signal_id="same_family_flatline",
                    kind="tension",
                    strength=min(0.9, 0.5 + 0.05 * float(self.same_family_no_score_streak)),
                    scope="local",
                    ttl_turns=2,
                    bias="verify_before_commit",
                    text="The current action family may be flattening into a repeated local story.",
                    evidence=self.stale_family_warning,
                )
            )
        if self.repeat_state_warning:
            signals.append(
                _signal(
                    signal_id="repeat_state_warning",
                    kind="tension",
                    strength=0.72,
                    scope="boundary",
                    ttl_turns=1,
                    bias="prefer_local_probe",
                    text="The current line may be revisiting the same state without fresh structural evidence.",
                    evidence=self.repeat_state_warning,
                )
            )
        if self.hard_guard_reason:
            signals.append(
                _signal(
                    signal_id="hard_guard_reason",
                    kind="safety",
                    strength=0.98,
                    scope="boundary",
                    ttl_turns=1,
                    bias="verify_before_commit",
                    text="A stronger guard concern is active, so the next step should stay conservative and verifiable.",
                    evidence=f"hard_guard_reason={self.hard_guard_reason}",
                )
            )
        if not signals:
            for idx, hint in enumerate(self.hints[:3], start=1):
                signals.append(
                    _signal(
                        signal_id=f"advisory_hint_{idx}",
                        kind="tension",
                        strength=min(0.8, max(0.35, float(self.discomfort_score or 0.35))),
                        scope="local",
                        ttl_turns=1,
                        bias="verify_before_commit",
                        text=hint,
                        evidence="Derived from current ECA observer hints.",
                    )
                )
        return signals

    def format_signal_block(self) -> str:
        """Format as a structured advisory block for prompt-side integration."""
        signals = self._build_signal_entries()
        if not signals:
            return ""

        if self.hard_guard_reason:
            recommended_mode = "cautious"
        elif self.dead_anchor_warning:
            recommended_mode = "prepare_handoff"
        elif self.click_exhausted_warning or self.stale_family_warning or self.repeat_state_warning:
            recommended_mode = "probe_first"
        elif self.board_change_no_score_streak > 0:
            recommended_mode = "hold_local"
        else:
            recommended_mode = "normal"

        biases = {str(item.get("bias", "") or "") for item in signals}
        overall_tension = round(
            max(
                float(self.discomfort_score or 0.0),
                max(float(item.get("strength", 0.0) or 0.0) for item in signals),
            ),
            2,
        )
        overall_conflict = 0.0
        if len(signals) > 1:
            overall_conflict = round(min(1.0, 0.2 + 0.08 * max(0, len(biases) - 1)), 2)

        lines = [
            "[ECA_SIGNAL_BLOCK]",
            "",
            "ECA advisory state:",
            f"  overall_tension: {overall_tension:.2f}",
            f"  overall_conflict: {overall_conflict:.2f}",
            f"  recommended_mode: {recommended_mode}",
            "",
            "signals:",
        ]
        for signal in signals:
            lines.extend(
                [
                    f"  - id: {signal['id']}",
                    f"    kind: {signal['kind']}",
                    f"    strength: {signal['strength']:.2f}",
                    f"    scope: {signal['scope']}",
                    f"    ttl_turns: {signal['ttl_turns']}",
                    f"    bias: {signal['bias']}",
                    f"    text: {signal['text']}",
                    f"    evidence: {signal['evidence'] or '(none)'}",
                ]
            )
        lines.extend(
            [
                "",
                "notes:",
                "  - Treat all signals as advisory, not commands.",
                "  - If strong signals conflict, prefer a small reversible information-gathering step.",
                "  - Ignore expired or contradicted signals.",
            ]
        )
        return "\n".join(lines)
