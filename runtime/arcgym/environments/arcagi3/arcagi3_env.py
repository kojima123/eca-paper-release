"""Thin ARC-AGI-3 environment on top of the official ``arc_agi`` toolkit.

Written for the ECA runtime release. Observations are plain dicts:
``game_id``, ``state`` (GameState name), ``score`` (levels completed), ``frame`` (list of 2-D lists),
``available_actions`` and ``guid``. ``step`` takes ``{"action": GameAction, "reasoning": ..., **data}``
and returns ``(observation, reward, done)``.
"""
from __future__ import annotations

from typing import Any, Mapping

import arc_agi
from arc_agi import OperationMode
from arcengine import GameState

from arcgym.core import BaseEnv

_MAX_REASONING_BYTES = 16000  # the toolkit accepts up to 16384 bytes of reasoning


def _as_lists(layer: Any) -> Any:
    return layer.tolist() if hasattr(layer, "tolist") else layer


class ArcAgi3Env(BaseEnv):
    """One ARC-AGI-3 game with an action budget."""

    def __init__(
        self,
        game_id: str,
        max_actions: int = 80,
        reward_mode: str = "binary",
        reward_scale: float = 1.0,
        arc_api_key: str = "",
        arc_base_url: str = "https://three.arcprize.org",
        operation_mode: OperationMode = OperationMode.NORMAL,
    ) -> None:
        self.game_id = game_id
        self.max_actions = max_actions
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self._arc = arc_agi.Arcade(arc_api_key=arc_api_key, arc_base_url=arc_base_url,
                                   operation_mode=operation_mode)
        self._owns_scorecard = True
        self._scorecard_id: str | None = None
        self._env = None
        self._last_obs = None
        self._actions_taken = 0

    @classmethod
    def from_arcade(cls, arcade: Any, game_id: str, scorecard_id: str, max_actions: int = 80,
                    reward_mode: str = "binary", reward_scale: float = 1.0) -> "ArcAgi3Env":
        """Use a caller-owned Arcade and scorecard (this env will not open or close it)."""
        env = cls.__new__(cls)
        env.game_id, env.max_actions = game_id, max_actions
        env.reward_mode, env.reward_scale = reward_mode, reward_scale
        env._arc, env._owns_scorecard, env._scorecard_id = arcade, False, scorecard_id
        env._env, env._last_obs, env._actions_taken = None, None, 0
        return env

    # -- episode -------------------------------------------------------------------------------
    def reset(self, task: dict | None = None) -> dict:
        task = task or {}
        if self._owns_scorecard:
            self._scorecard_id = self._arc.open_scorecard(tags=task.get("tags", []))
        self._env = self._arc.make(task.get("game_id", self.game_id), scorecard_id=self._scorecard_id)
        self._last_obs = self._env.reset()
        return self._format_observation(self._last_obs)

    def step(self, action_payload: Any) -> tuple[dict, float, bool]:
        if self._env is None or self._last_obs is None:
            raise RuntimeError("reset() must be called before step()")
        if not isinstance(action_payload, Mapping):
            raise TypeError(f"action payload must be a mapping, got {type(action_payload)}")
        action = action_payload.get("action")
        reasoning = action_payload.get("reasoning")
        if isinstance(reasoning, str):
            raw = reasoning.encode("utf-8")
            if len(raw) > _MAX_REASONING_BYTES:
                reasoning = raw[:_MAX_REASONING_BYTES].decode("utf-8", errors="ignore")
        data = {k: v for k, v in action_payload.items() if k not in ("action", "reasoning")}
        obs = self._env.step(action, data=data, reasoning=reasoning)
        if obs is None:
            raise ConnectionError("the toolkit returned no frame")
        self._actions_taken += 1
        self._last_obs = obs
        done = obs.state in (GameState.WIN, GameState.GAME_OVER) or self._actions_taken >= self.max_actions
        return self._format_observation(obs), self._reward(obs), done

    def close(self) -> None:
        if self._owns_scorecard:
            self._arc.close_scorecard(self._scorecard_id)
        self._env, self._last_obs, self._actions_taken = None, None, 0

    # -- scorecards ----------------------------------------------------------------------------
    def open_scorecard(self, tags: list[str] | None = None) -> str:
        return self._arc.open_scorecard(tags=tags)

    def close_scorecard(self, card_id: str | None = None) -> Any:
        return self._arc.close_scorecard(card_id)

    def get_scorecard(self) -> Any:
        return self._arc.get_scorecard(self._scorecard_id)

    # -- helpers -------------------------------------------------------------------------------
    def _format_observation(self, obs: Any) -> dict[str, Any]:
        return {
            "game_id": obs.game_id,
            "state": obs.state.name,
            "score": obs.levels_completed,
            "frame": [_as_lists(layer) for layer in obs.frame],
            "available_actions": obs.available_actions,
            "guid": obs.guid,
        }

    def _reward(self, obs: Any) -> float:
        if self.reward_mode == "binary":
            value = 1.0 if obs.state == GameState.WIN else 0.0
        elif self.reward_mode == "score":
            value = float(obs.levels_completed)
        else:
            raise ValueError(f"unknown reward_mode {self.reward_mode!r}")
        return value * float(self.reward_scale)
