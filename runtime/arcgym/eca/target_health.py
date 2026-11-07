"""TargetHealth: 5-state target health evaluator for ControlIR-first Runtime v2.

Replaces the binary `selected_target_empty` flag with a richer 5-state
classification that allows LocalController to distinguish between:
  - LIVE       : target is valid and making progress
  - STALE      : target exists but progress has stalled
  - SATURATED  : target exists but is dominated by no_information_gain / dead_revisit
  - BLOCKED    : target exists but anchor_cooldown or family conflict prevents progress
  - EMPTY      : no target at all

This module is observer-only in Phase 1: it computes and records health state
but does not yet alter control flow.

Usage in runtime.py:
    self._target_health_evaluator = TargetHealthEvaluator()
    # In observe():
    health = self._target_health_evaluator.evaluate(self, step_index)
    self._last_target_health = health
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


# ─── TargetHealth State ───────────────────────────────────────────────────────

class TargetHealth(str, Enum):
    """5-state target health classification."""
    LIVE = "live"
    STALE = "stale"
    SATURATED = "saturated"
    BLOCKED = "blocked"
    EMPTY = "empty"


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TargetHealthConfig:
    """Thresholds for TargetHealth classification."""

    # STALE: number of consecutive steps without score improvement
    stale_no_score_streak_threshold: int = 3

    # SATURATED: fraction of candidate pool that is dead_revisit / no_information_gain
    saturated_dead_ratio_threshold: float = 0.60

    # SATURATED: consecutive no_information_gain steps
    saturated_no_info_gain_streak_threshold: int = 4
    saturated_no_info_gain_ratio_threshold: float = 0.35

    # BLOCKED: viability drop reasons that indicate blocking
    blocked_viability_drop_reasons: tuple[str, ...] = (
        "anchor_cooldown",
        "family_conflict",
        "same_dead_anchor",
        "no_information_gain_after_handoff",
    )
    # Route shifts should partially clear stale debt so newly recovered targets
    # can surface as LIVE before they accumulate a fresh no-score band.
    route_shift_stale_relief: int = 2
    # LIVE should not drop to STALE immediately on the first light no-score turn.
    live_stale_grace_steps: int = 1
    # After a stale/saturated route recovers to LIVE, keep that recovered route
    # alive for a short bounded window so old stale debt does not immediately
    # re-open the same steering loop.
    live_recovery_hold_steps: int = 1

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "TargetHealthConfig":
        if not raw:
            return cls()
        return cls(
            stale_no_score_streak_threshold=int(
                raw.get("stale_no_score_streak_threshold", 3)
            ),
            saturated_dead_ratio_threshold=float(
                raw.get("saturated_dead_ratio_threshold", 0.60)
            ),
            saturated_no_info_gain_streak_threshold=int(
                raw.get("saturated_no_info_gain_streak_threshold", 4)
            ),
            saturated_no_info_gain_ratio_threshold=float(
                raw.get("saturated_no_info_gain_ratio_threshold", 0.35)
            ),
            route_shift_stale_relief=max(
                0, int(raw.get("route_shift_stale_relief", 2))
            ),
            live_stale_grace_steps=max(
                0, int(raw.get("live_stale_grace_steps", 1))
            ),
            live_recovery_hold_steps=max(
                0, int(raw.get("live_recovery_hold_steps", 1))
            ),
            blocked_viability_drop_reasons=tuple(
                raw.get(
                    "blocked_viability_drop_reasons",
                    ["anchor_cooldown", "family_conflict",
                     "same_dead_anchor", "no_information_gain_after_handoff"],
                )
            ),
        )


# ─── Evaluation Result ────────────────────────────────────────────────────────

@dataclass
class TargetHealthResult:
    """Result of a single TargetHealth evaluation."""

    health: TargetHealth = TargetHealth.EMPTY
    reason: str = ""
    # Supporting signals
    selected_target_empty: bool = True
    no_score_streak: int = 0
    dead_ratio: float = 0.0
    no_info_gain_streak: int = 0
    viability_drop_reason: str = ""
    step_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "health": self.health.value,
            "reason": self.reason,
            "selected_target_empty": self.selected_target_empty,
            "no_score_streak": self.no_score_streak,
            "dead_ratio": round(self.dead_ratio, 4),
            "no_info_gain_streak": self.no_info_gain_streak,
            "viability_drop_reason": self.viability_drop_reason,
            "step_index": self.step_index,
        }


# ─── Evaluator ───────────────────────────────────────────────────────────────

class TargetHealthEvaluator:
    """Evaluates target health each step from ECARuntime state.

    This evaluator is stateful: it tracks streaks across steps to detect
    STALE and SATURATED conditions that require multiple consecutive steps.
    """

    def __init__(self, config: TargetHealthConfig | None = None) -> None:
        self.config = config or TargetHealthConfig()
        self._no_score_streak: int = 0
        self._no_info_gain_streak: int = 0
        self._last_health: TargetHealth = TargetHealth.EMPTY
        self._history: list[TargetHealthResult] = []
        self._last_target_component_id: str = ""
        self._last_target_family: str = ""
        self._live_recovery_hold_remaining: int = 0
        self._live_recovery_hold_component_id: str = ""
        self._live_recovery_hold_family: str = ""

    def reset(self) -> None:
        """Reset for a new episode."""
        self._no_score_streak = 0
        self._no_info_gain_streak = 0
        self._last_health = TargetHealth.EMPTY
        self._history.clear()
        self._last_target_component_id = ""
        self._last_target_family = ""
        self._live_recovery_hold_remaining = 0
        self._live_recovery_hold_component_id = ""
        self._live_recovery_hold_family = ""

    @property
    def last_health(self) -> TargetHealth:
        return self._last_health

    @property
    def history(self) -> list[TargetHealthResult]:
        return self._history

    def evaluate(self, runtime: Any, step_index: int) -> TargetHealthResult:
        """Evaluate target health from ECARuntime state.

        Args:
            runtime: ECARuntime instance (duck-typed).
            step_index: Current step index.

        Returns:
            TargetHealthResult with the current health state.
        """
        result = TargetHealthResult(step_index=step_index)

        # ── 1. Read raw signals from runtime ──────────────────────────────
        selected_target = getattr(runtime, "last_selected_target_summary", None)
        selected_target_empty = selected_target is None or not bool(selected_target)
        if hasattr(runtime, "_selected_target_summary_effectively_empty"):
            try:
                selected_target_empty = bool(
                    runtime._selected_target_summary_effectively_empty(selected_target)
                )
            except Exception:
                pass
        result.selected_target_empty = selected_target_empty

        dead_ratio = float(
            getattr(runtime, "last_candidate_pool_dead_tail_ratio", 0.0) or 0.0
        )
        result.dead_ratio = dead_ratio

        viability_drop_reason = str(
            getattr(runtime, "last_current_target_viability_drop_reason", "") or ""
        )
        result.viability_drop_reason = viability_drop_reason

        state = getattr(runtime, "last_state", None)
        no_change_streak = int(getattr(state, "no_change_streak", 0) or 0)
        repeated_state_count = int(getattr(state, "repeated_state_count", 0) or 0)
        same_family_no_score_streak = int(
            getattr(runtime, "_same_family_no_score_streak", 0) or 0
        )
        stable_target_repeat_count = int(
            getattr(runtime, "repeated_stable_target_count", 0) or 0
        )
        lifecycle_state = str(
            getattr(runtime, "last_target_lifecycle_state", "") or ""
        ).strip().upper()
        current_target_component_id = ""
        current_target_family = str(getattr(runtime, "last_stable_action_family", "") or "").strip().lower()
        if isinstance(selected_target, dict):
            current_target_component_id = str(
                selected_target.get("component_id", "")
                or selected_target.get("candidate_id", "")
                or ""
            ).strip()
            current_target_family = str(
                selected_target.get("action_family", "")
                or selected_target.get("family", "")
                or selected_target.get("transition_model_candidate_family", "")
                or current_target_family
                or ""
            ).strip().lower()
        route_shifted = bool(
            current_target_component_id
            and self._last_target_component_id
            and (
                current_target_component_id != self._last_target_component_id
                or (
                    current_target_family
                    and self._last_target_family
                    and current_target_family != self._last_target_family
                )
            )
        )
        live_recovery_hold_active = bool(
            not selected_target_empty
            and self._live_recovery_hold_remaining > 0
            and current_target_component_id
            and current_target_component_id == self._live_recovery_hold_component_id
        )

        # Infer no_information_gain streak from reject_reason histogram
        reject_histogram: dict[str, int] = dict(
            getattr(runtime, "last_candidate_pool_reject_reason_histogram", {}) or {}
        )
        candidate_pool_size = int(getattr(runtime, "last_candidate_pool_size", 0) or 0)
        if candidate_pool_size <= 0:
            candidate_pool_size = len(getattr(runtime, "last_candidate_pool", []) or [])
        no_info_gain_count = int(
            reject_histogram.get("no_information_gain", 0)
            + reject_histogram.get("no_information_gain_after_handoff", 0)
        )
        no_info_gain_ratio = (
            float(no_info_gain_count) / float(max(1, candidate_pool_size))
            if candidate_pool_size > 0
            else 0.0
        )

        # ── 2. Update streaks ─────────────────────────────────────────────
        if not selected_target_empty:
            adjusted_no_change_streak = no_change_streak
            if lifecycle_state == "FRESH" and adjusted_no_change_streak > 0:
                adjusted_no_change_streak = max(0, adjusted_no_change_streak - 1)
            same_family_signal = same_family_no_score_streak
            stable_repeat_signal = max(0, stable_target_repeat_count - 1)
            if route_shifted:
                same_family_signal = 0
                stable_repeat_signal = 0
            effective_no_score_streak = max(
                same_family_signal,
                adjusted_no_change_streak,
                stable_repeat_signal,
                max(0, repeated_state_count - 1),
            )
            if route_shifted and self.config.route_shift_stale_relief > 0:
                effective_no_score_streak = max(
                    0,
                    effective_no_score_streak - self.config.route_shift_stale_relief,
                )
            if effective_no_score_streak > 0:
                if route_shifted:
                    self._no_score_streak = effective_no_score_streak
                else:
                    self._no_score_streak = max(
                        effective_no_score_streak,
                        self._no_score_streak + 1,
                    )
            else:
                self._no_score_streak = 0

            if (
                no_info_gain_count > 0
                and no_info_gain_ratio >= self.config.saturated_no_info_gain_ratio_threshold
            ):
                self._no_info_gain_streak += 1
            else:
                self._no_info_gain_streak = 0
        else:
            # Target is empty — reset streaks
            self._no_score_streak = 0
            self._no_info_gain_streak = 0

        result.no_score_streak = self._no_score_streak
        result.no_info_gain_streak = self._no_info_gain_streak

        # ── 3. Classify health ────────────────────────────────────────────
        if selected_target_empty:
            result.health = TargetHealth.EMPTY
            result.reason = "selected_target_empty"

        elif viability_drop_reason in self.config.blocked_viability_drop_reasons:
            result.health = TargetHealth.BLOCKED
            result.reason = f"viability_drop:{viability_drop_reason}"

        elif (
            dead_ratio >= self.config.saturated_dead_ratio_threshold
            or self._no_info_gain_streak >= self.config.saturated_no_info_gain_streak_threshold
        ):
            result.health = TargetHealth.SATURATED
            if dead_ratio >= self.config.saturated_dead_ratio_threshold:
                result.reason = f"dead_ratio:{dead_ratio:.2f}"
            else:
                result.reason = f"no_info_gain_streak:{self._no_info_gain_streak}"

        elif self._no_score_streak >= self.config.stale_no_score_streak_threshold:
            if (
                self._last_health == TargetHealth.LIVE
                and self.config.live_stale_grace_steps > 0
                and self._no_info_gain_streak == 0
                and dead_ratio < self.config.saturated_dead_ratio_threshold
                and self._no_score_streak
                < self.config.stale_no_score_streak_threshold + self.config.live_stale_grace_steps
            ):
                result.health = TargetHealth.LIVE
                result.reason = "live_grace"
            elif (
                live_recovery_hold_active
                and self._no_info_gain_streak == 0
                and dead_ratio < self.config.saturated_dead_ratio_threshold
                and viability_drop_reason not in self.config.blocked_viability_drop_reasons
            ):
                result.health = TargetHealth.LIVE
                result.reason = "live_recovery_hold"
            else:
                result.health = TargetHealth.STALE
                if no_change_streak >= self.config.stale_no_score_streak_threshold:
                    result.reason = f"no_change_streak:{no_change_streak}"
                elif stable_target_repeat_count >= self.config.stale_no_score_streak_threshold:
                    result.reason = f"stable_target_repeat_count:{stable_target_repeat_count}"
                elif same_family_no_score_streak >= self.config.stale_no_score_streak_threshold:
                    result.reason = f"same_family_no_score_streak:{same_family_no_score_streak}"
                else:
                    result.reason = f"no_score_streak:{self._no_score_streak}"

        else:
            result.health = TargetHealth.LIVE
            result.reason = "target_live"

        previous_health = self._last_health
        recovered_to_live = bool(
            self.config.live_recovery_hold_steps > 0
            and not selected_target_empty
            and current_target_component_id
            and result.health == TargetHealth.LIVE
            and (
                route_shifted
                or previous_health in {
                    TargetHealth.STALE,
                    TargetHealth.SATURATED,
                    TargetHealth.BLOCKED,
                }
            )
        )
        if recovered_to_live:
            self._live_recovery_hold_remaining = self.config.live_recovery_hold_steps
            self._live_recovery_hold_component_id = current_target_component_id
            self._live_recovery_hold_family = current_target_family
        elif (
            live_recovery_hold_active
            and result.health == TargetHealth.LIVE
            and current_target_component_id == self._live_recovery_hold_component_id
        ):
            self._live_recovery_hold_remaining = max(0, self._live_recovery_hold_remaining - 1)
            if self._live_recovery_hold_remaining <= 0:
                self._live_recovery_hold_component_id = ""
                self._live_recovery_hold_family = ""
        elif (
            selected_target_empty
            or not current_target_component_id
            or current_target_component_id != self._live_recovery_hold_component_id
        ):
            self._live_recovery_hold_remaining = 0
            self._live_recovery_hold_component_id = ""
            self._live_recovery_hold_family = ""

        self._last_health = result.health
        self._history.append(result)
        if current_target_component_id:
            self._last_target_component_id = current_target_component_id
            self._last_target_family = current_target_family
        else:
            self._last_target_component_id = ""
            self._last_target_family = ""
        return result

    def step_metrics(self) -> dict[str, Any]:
        """Return current step metrics for inclusion in runtime step_metrics."""
        if not self._history:
            return {
                "target_health": TargetHealth.EMPTY.value,
                "target_health_reason": "",
                "target_health_no_score_streak": 0,
                "target_health_no_info_gain_streak": 0,
                "target_health_dead_ratio": 0.0,
                "target_health_live_recovery_hold_remaining": 0,
            }
        last = self._history[-1]
        return {
            "target_health": last.health.value,
            "target_health_reason": last.reason,
            "target_health_no_score_streak": last.no_score_streak,
            "target_health_no_info_gain_streak": last.no_info_gain_streak,
            "target_health_dead_ratio": round(last.dead_ratio, 4),
            "target_health_live_recovery_hold_remaining": int(self._live_recovery_hold_remaining),
        }

    def episode_summary(self) -> dict[str, Any]:
        """Return episode-level summary for inclusion in runtime episode_summary."""
        if not self._history:
            return {
                "total_steps": 0,
                "health_histogram": {},
                "selected_target_health_histogram": {},
                "stale_turns": 0,
                "saturated_turns": 0,
                "blocked_turns": 0,
                "empty_turns": 0,
                "live_turns": 0,
            }
        histogram: dict[str, int] = {}
        for r in self._history:
            histogram[r.health.value] = histogram.get(r.health.value, 0) + 1
        return {
            "total_steps": len(self._history),
            "health_histogram": histogram,
            "selected_target_health_histogram": histogram,
            "live_turns": histogram.get(TargetHealth.LIVE.value, 0),
            "stale_turns": histogram.get(TargetHealth.STALE.value, 0),
            "saturated_turns": histogram.get(TargetHealth.SATURATED.value, 0),
            "blocked_turns": histogram.get(TargetHealth.BLOCKED.value, 0),
            "empty_turns": histogram.get(TargetHealth.EMPTY.value, 0),
        }
