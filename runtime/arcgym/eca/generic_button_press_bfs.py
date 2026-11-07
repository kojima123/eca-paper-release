"""Runtime wrapper for the generic button-press bootstrap.

The bootstrap performs source-free disposable rollout discovery and returns a
score-confirmed ACTION6 click prefix. This wrapper drains that prefix in the
live episode, then allows one bounded post-score follow-up discovery using the
issued prefix. It is default-off and does not branch on game id.
"""
from __future__ import annotations

from typing import Any, Optional


ActionSpec = dict[str, Any]


class GenericButtonPressController:
    """Stateful wrapper that drains score-confirmed button-press plans."""

    def __init__(
        self,
        game_id: str,
        max_actions: int = 240,
        max_followup_plans: int = 2,
    ) -> None:
        self._game_id = str(game_id or "").strip().lower()
        self._max_actions = int(max_actions)
        self._max_followup_plans = max(0, int(max_followup_plans))
        self._plan: list[ActionSpec] = []
        self._attempted = False
        self._issued_prefix: list[ActionSpec] = []
        self._followup_plans = 0
        self._needs_followup_plan = False
        self.engaged = False
        self.fired = 0
        self.diagnostics: dict[str, Any] = {}

    @property
    def followup_plans(self) -> int:
        return int(self._followup_plans)

    def observe_transition(self, pre_score: int = 0, post_score: int = 0) -> None:
        pre = int(pre_score or 0)
        post = int(post_score or 0)
        if (
            self.engaged
            and post > pre
            and self._followup_plans < self._max_followup_plans
        ):
            self._needs_followup_plan = True
            self._plan.clear()

    def next_spec(self, available: set[str]) -> Optional[ActionSpec]:
        if not self._game_id or "ACTION6" not in available:
            return None
        if self._needs_followup_plan:
            self._needs_followup_plan = False
            self._followup_plans += 1
            try:
                from arcgym.eca import generic_button_press_bootstrap as button_press

                plan = button_press.plan_after_prefix(
                    self._game_id,
                    list(self._issued_prefix),
                    self._max_actions,
                )
                self.diagnostics = button_press.LAST_DIAGNOSTICS.to_dict()
            except Exception as exc:  # noqa: BLE001
                self.diagnostics = {"gate_reason": f"followup_exception:{type(exc).__name__}"}
                plan = None
            if plan:
                self._plan = list(plan)
        elif not self._attempted:
            self._attempted = True
            try:
                from arcgym.eca import generic_button_press_bootstrap as button_press

                plan = button_press.plan(self._game_id, self._max_actions)
                self.diagnostics = button_press.LAST_DIAGNOSTICS.to_dict()
            except Exception as exc:  # noqa: BLE001
                self.diagnostics = {"gate_reason": f"exception:{type(exc).__name__}"}
                plan = None
            if plan:
                self._plan = list(plan)
        while self._plan:
            spec = self._plan.pop(0)
            action_name = str(spec.get("action", "") or "")
            if action_name not in available:
                self._plan.clear()
                return None
            self.engaged = True
            self.fired += 1
            out = dict(spec)
            self._issued_prefix.append(out)
            return out
        return None
