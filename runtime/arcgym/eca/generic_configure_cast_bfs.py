"""Runtime wrapper for the generic configure-then-cast bootstrap.

The underlying bootstrap performs source-free disposable rollout discovery and
returns a score-confirmed prefix. This wrapper only drains that prefix into the
current C-axis live episode. It is default-off and does not branch on game id.
"""
from __future__ import annotations

from typing import Any, Optional


ActionSpec = dict[str, Any]


class GenericConfigureCastController:
    """Small stateful wrapper that drains a score-confirmed configure/cast prefix."""

    def __init__(
        self,
        game_id: str,
        max_actions: int = 240,
        max_followup_plans: int = 5,
    ) -> None:
        self._game_id = str(game_id or "").strip().lower()
        self._max_actions = int(max_actions)
        self._max_followup_plans = max(0, int(max_followup_plans))
        self._plan: list[ActionSpec] = []
        self._attempted = False
        self._issued_prefix: list[ActionSpec] = []
        self._followup_plans = 0
        self._needs_followup_plan = False
        self._last_score = 0
        self.engaged = False
        self.fired = 0
        self.diagnostics: dict[str, Any] = {}

    @property
    def followup_plans(self) -> int:
        return int(self._followup_plans)

    def observe_transition(self, pre_score: int = 0, post_score: int = 0) -> None:
        pre = int(pre_score or 0)
        post = int(post_score or 0)
        self._last_score = max(self._last_score, post)
        if (
            self.engaged
            and post > pre
            and self._followup_plans < self._max_followup_plans
        ):
            self._needs_followup_plan = True
            self._plan.clear()

    def next_spec(self, available: set[str]) -> Optional[ActionSpec]:
        if not self._game_id:
            return None
        if self._needs_followup_plan:
            self._needs_followup_plan = False
            self._followup_plans += 1
            try:
                from arcgym.eca import generic_configure_cast as configure_cast

                plan = configure_cast.plan_after_prefix(
                    self._game_id,
                    list(self._issued_prefix),
                    self._max_actions,
                )
                self.diagnostics = configure_cast.LAST_DIAGNOSTICS.to_dict()
            except Exception as exc:  # noqa: BLE001
                self.diagnostics = {"gate_reason": f"followup_exception:{type(exc).__name__}"}
                plan = None
            if plan:
                self._plan = list(plan)
        elif not self._attempted:
            self._attempted = True
            try:
                from arcgym.eca import generic_configure_cast as configure_cast

                plan = configure_cast.plan(self._game_id, self._max_actions)
                self.diagnostics = configure_cast.LAST_DIAGNOSTICS.to_dict()
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
