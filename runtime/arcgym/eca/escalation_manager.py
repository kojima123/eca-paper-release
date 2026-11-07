"""EscalationManager: Decides when LocalController should escalate to Analyzer.

In ControlIR-first Runtime v2, LocalController runs every step (BaseLocalLoop).
EscalationManager determines whether the current step should be handled locally
or escalated to the LLM Analyzer.

Escalation triggers (any one is sufficient):
  1. HEALTH_BLOCKED   : TargetHealth == BLOCKED (anchor_cooldown / family_conflict)
  2. HEALTH_EMPTY     : TargetHealth == EMPTY and no candidates in base pool
  3. CONSECUTIVE_FAIL : N consecutive local planning failures
  4. IR_OVERRIDE      : An active IR ticket explicitly requests Analyzer
  5. UNKNOWN_FAMILY   : Current family not in MicroPlanner's known families

Non-escalation (LocalController handles locally):
  - TargetHealth in {LIVE, STALE, SATURATED} with non-empty base pool
  - Known family with available candidates

Design: EscalationManager is stateful and tracks consecutive failures.
It is reset at the start of each episode.

Usage in local_controller.py:
    self._escalation_manager = EscalationManager(config)
    # In step():
    decision = self._escalation_manager.decide(health, base_pool, ir_result, family)
    if decision.should_escalate:
        result.needs_analyzer = True
        result.analyzer_reason = decision.reason
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from arcgym.eca.target_health import TargetHealth
    from arcgym.eca.candidate_extractor import ExtractionResult


# ─── Escalation Trigger ───────────────────────────────────────────────────────

class EscalationTrigger(str, Enum):
    """Reason for escalating to Analyzer."""
    HEALTH_BLOCKED = "health_blocked"
    HEALTH_EMPTY_NO_POOL = "health_empty_no_pool"
    CONSECUTIVE_FAIL = "consecutive_fail"
    IR_OVERRIDE = "ir_override"
    UNKNOWN_FAMILY = "unknown_family"
    AFTERCARE_STALLED = "aftercare_stalled"
    NO_TRIGGER = "no_trigger"


# ─── Escalation Decision ─────────────────────────────────────────────────────

@dataclass
class EscalationDecision:
    """Result of an escalation decision."""

    should_escalate: bool = False
    trigger: EscalationTrigger = EscalationTrigger.NO_TRIGGER
    reason: str = ""
    step_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_escalate": self.should_escalate,
            "trigger": self.trigger.value,
            "reason": self.reason,
            "step_index": self.step_index,
        }


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class EscalationManagerConfig:
    """Configuration for EscalationManager."""

    # Number of consecutive local failures before escalating
    consecutive_fail_threshold: int = 3

    # Known action families that MicroPlanner can handle locally
    # Must match _HANDLER_MAP keys in micro_planner.py
    known_families: frozenset[str] = field(default_factory=lambda: frozenset({
        "click",
        "move",
        "color_fill",
        "drag",
        "copy",
        "flood_fill",
        "draw_line",
        "select",
        "reset",
    }))
    aftercare_stall_step_threshold: int = 48
    aftercare_stall_cooldown_steps: int = 32

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "EscalationManagerConfig":
        if not raw:
            return cls()
        known = raw.get("known_families")
        return cls(
            consecutive_fail_threshold=int(raw.get("consecutive_fail_threshold", 3)),
            known_families=frozenset(known) if known else cls().known_families,
            aftercare_stall_step_threshold=max(
                0, int(raw.get("aftercare_stall_step_threshold", 48))
            ),
            aftercare_stall_cooldown_steps=max(
                1, int(raw.get("aftercare_stall_cooldown_steps", 32))
            ),
        )


# ─── EscalationManager ───────────────────────────────────────────────────────

class EscalationManager:
    """Decides whether to escalate to Analyzer or handle locally.

    This is the central decision point for the BaseLocalLoop.
    """

    def __init__(self, config: EscalationManagerConfig | None = None) -> None:
        self.config = config or EscalationManagerConfig()
        self._consecutive_failures: int = 0
        self._history: list[EscalationDecision] = []
        self._last_aftercare_stall_step: int = -10_000

    def reset(self) -> None:
        """Reset for a new episode."""
        self._consecutive_failures = 0
        self._history.clear()
        self._last_aftercare_stall_step = -10_000

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def record_local_success(self) -> None:
        """Record that local planning succeeded this step."""
        self._consecutive_failures = 0

    def record_local_failure(self) -> None:
        """Record that local planning failed this step."""
        self._consecutive_failures += 1

    def decide(
        self,
        *,
        health: Any,  # TargetHealth enum value
        base_pool_size: int,
        current_family: str,
        ir_override: bool = False,
        step_index: int = -1,
        protected_hedge_visible_turn_count: int = 0,
        post_reclaim_hold_count: int = 0,
        post_final_shared_entry_count: int = 0,
    ) -> EscalationDecision:
        """Decide whether to escalate to Analyzer.

        Args:
            health: Current TargetHealth state.
            base_pool_size: Number of candidates in the base pool.
            current_family: Current action family being attempted.
            ir_override: Whether an active IR ticket requests Analyzer.
            step_index: Current step index.

        Returns:
            EscalationDecision with should_escalate flag and reason.
        """
        decision = EscalationDecision(step_index=step_index)

        # Import here to avoid circular imports
        from arcgym.eca.target_health import TargetHealth

        # ── Trigger 1: IR override ────────────────────────────────────
        if ir_override:
            decision.should_escalate = True
            decision.trigger = EscalationTrigger.IR_OVERRIDE
            decision.reason = "ir_ticket_requests_analyzer"
            self._history.append(decision)
            return decision

        # ── Trigger 2: BLOCKED health ─────────────────────────────────
        if health == TargetHealth.BLOCKED:
            decision.should_escalate = True
            decision.trigger = EscalationTrigger.HEALTH_BLOCKED
            decision.reason = "target_health_blocked"
            self._history.append(decision)
            return decision

        # ── Trigger 3: EMPTY health with no pool ─────────────────────
        if health == TargetHealth.EMPTY and base_pool_size == 0:
            decision.should_escalate = True
            decision.trigger = EscalationTrigger.HEALTH_EMPTY_NO_POOL
            decision.reason = "target_empty_no_candidates"
            self._history.append(decision)
            return decision

        # ── Trigger 4: Unknown family ─────────────────────────────────
        if current_family and current_family not in self.config.known_families:
            decision.should_escalate = True
            decision.trigger = EscalationTrigger.UNKNOWN_FAMILY
            decision.reason = f"unknown_family:{current_family}"
            self._history.append(decision)
            return decision

        # ── Trigger 5: Consecutive failures ───────────────────────────
        if self._consecutive_failures >= self.config.consecutive_fail_threshold:
            decision.should_escalate = True
            decision.trigger = EscalationTrigger.CONSECUTIVE_FAIL
            decision.reason = f"consecutive_failures:{self._consecutive_failures}"
            self._history.append(decision)
            return decision

        # ── No escalation ─────────────────────────────────────────────
        aftercare_cooldown_satisfied = (
            step_index - self._last_aftercare_stall_step
            >= self.config.aftercare_stall_cooldown_steps
        )
        if (
            health == TargetHealth.LIVE
            and base_pool_size > 0
            and step_index >= self.config.aftercare_stall_step_threshold
            and protected_hedge_visible_turn_count <= 0
            and post_reclaim_hold_count <= 0
            and post_final_shared_entry_count <= 0
            and aftercare_cooldown_satisfied
        ):
            decision.should_escalate = True
            decision.trigger = EscalationTrigger.AFTERCARE_STALLED
            decision.reason = (
                "aftercare_stalled:"
                f"step={step_index},"
                f"hedge={protected_hedge_visible_turn_count},"
                f"reclaim={post_reclaim_hold_count},"
                f"shared={post_final_shared_entry_count}"
            )
            self._last_aftercare_stall_step = step_index
            self._history.append(decision)
            return decision

        decision.should_escalate = False
        decision.trigger = EscalationTrigger.NO_TRIGGER
        decision.reason = "local_handling"
        self._history.append(decision)
        return decision

    def step_metrics(self) -> dict[str, Any]:
        """Return current step metrics."""
        if not self._history:
            return {
                "escalation_trigger": EscalationTrigger.NO_TRIGGER.value,
                "escalation_consecutive_failures": 0,
                "escalated": False,
            }
        last = self._history[-1]
        return {
            "escalation_trigger": last.trigger.value,
            "escalation_consecutive_failures": self._consecutive_failures,
            "escalated": last.should_escalate,
        }

    def episode_summary(self) -> dict[str, Any]:
        """Return episode-level summary."""
        if not self._history:
            return {
                "total_steps": 0,
                "escalated_steps": 0,
                "local_steps": 0,
                "escalation_rate": 0.0,
                "trigger_histogram": {},
            }
        escalated = [d for d in self._history if d.should_escalate]
        histogram: dict[str, int] = {}
        for d in self._history:
            histogram[d.trigger.value] = histogram.get(d.trigger.value, 0) + 1
        total = len(self._history)
        return {
            "total_steps": total,
            "escalated_steps": len(escalated),
            "local_steps": total - len(escalated),
            "escalation_rate": round(len(escalated) / total, 4) if total > 0 else 0.0,
            "trigger_histogram": histogram,
        }
