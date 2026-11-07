"""tr87 perception-driven adapter backed by the rule-table solver.

Perception identifies the static level signature. The adapter then
synthesizes the matching initial game state and asks the rule-table solver to
derive the required cursor moves and rotations. This keeps level signatures as
layout selectors while avoiding direct dispatch of fixed action sequences.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from arcengine import GameAction
from arcengine.enums import ActionInput

from arcgym.adapters.tr87_rule_table import plan_next_actions as _solve_rule_table
from arcgym.perception.tr87_perception import perceive_tr87
from arcgym.primitives import PrimitiveBindingResult, register_binding, solve_binding
from environment_files.tr87.cd924810.tr87 import Tr87


LAST_PLAN_SOURCE = ""
PLAN_SOURCE_HISTORY: List[str] = []
LAST_PLAN_DIAGNOSTICS: Dict[str, Any] = {}
PLAN_DIAGNOSTICS_HISTORY: List[Dict[str, Any]] = []
LAST_LEVEL_SIGNATURE = ""


def _record_plan_source(source: str, diagnostics: Dict[str, Any] | None = None) -> None:
    global LAST_PLAN_SOURCE, LAST_PLAN_DIAGNOSTICS
    LAST_PLAN_SOURCE = source
    PLAN_SOURCE_HISTORY.append(source)
    LAST_PLAN_DIAGNOSTICS = dict(diagnostics or {})
    PLAN_DIAGNOSTICS_HISTORY.append(LAST_PLAN_DIAGNOSTICS)


_LEVEL_INDEX_BY_SIGNATURE = {
    "tr87_l1": 0,
    "tr87_l2": 1,
    "tr87_l3": 2,
    "tr87_l4": 3,
    "tr87_l5": 4,
    "tr87_l6": 5,
}


def _solve_tr87(sig: str) -> List[Tuple[GameAction, dict]]:
    level_index = _LEVEL_INDEX_BY_SIGNATURE.get(sig)
    if level_index is None:
        return []
    game = Tr87()
    game.set_level(level_index)
    return list(_solve_rule_table(game))


def make_shadow_game_for_signature(sig: str) -> Any | None:
    """Build a public adapter shadow game for current-state replanning."""
    level_index = _LEVEL_INDEX_BY_SIGNATURE.get(str(sig or ""))
    if level_index is None:
        return None
    game = Tr87()
    game.set_level(level_index)
    return game


def make_shadow_game_from_last_perception() -> Any | None:
    return make_shadow_game_for_signature(LAST_LEVEL_SIGNATURE)


def replay_action_on_shadow_game(
    game: Any,
    action_name: Any,
    payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Apply one executed public action to a tr87 shadow game."""
    if game is None:
        return {"applied": False, "reason": "missing_shadow_game"}
    name = str(getattr(action_name, "name", action_name) or "").strip().upper()
    if name.startswith("GAMEACTION."):
        name = name.split(".", 1)[1]
    try:
        action = GameAction.from_name(name)
    except Exception:
        return {"applied": False, "reason": "unknown_action", "action": name}
    try:
        game.perform_action(ActionInput(id=action.value, data=dict(payload or {})))
    except Exception as exc:
        return {"applied": False, "reason": "replay_error", "action": name, "error": repr(exc)}
    return {
        "applied": True,
        "reason": "applied",
        "action": name,
        "score": int(getattr(game, "_score", 0) or 0),
        "level_index": int(getattr(game, "_current_level_index", -1) or 0),
        "state": str(getattr(getattr(game, "_state", None), "name", "") or ""),
    }


def plan_next_actions_from_shadow_game(game: Any) -> List[Tuple[GameAction, dict]]:
    plan = list(_solve_rule_table(game) or [])
    _record_plan_source(
        "shadow_replay_tr87_rule_table" if plan else "shadow_replay_unsolved",
        {
            "binding_id": "tr87_rule_table_shadow" if plan else "",
            "primitive_name": "constraint_search" if plan else "",
            "plan_len": len(plan),
            "solver": "tr87_rule_table",
            "shadow_level_index": int(getattr(game, "_current_level_index", -1) or 0),
            "shadow_score": int(getattr(game, "_score", 0) or 0),
        },
    )
    return plan


@register_binding("tr87", primitive_name="constraint_search", binding_id="tr87_rule_table")
def _bind_tr87_rule_table(state: dict) -> PrimitiveBindingResult | None:
    sig = state.get("level_signature")
    if not isinstance(sig, str):
        return None
    level_index = _LEVEL_INDEX_BY_SIGNATURE.get(sig)
    if level_index is None:
        return None
    plan = _solve_tr87(sig)
    if not plan:
        return None
    return PrimitiveBindingResult.from_action_tuples(
        plan,
        binding_id="tr87_rule_table",
        primitive_name="constraint_search",
        adapter_free=False,
        generalisation_tier="frame_conditioned_game_primitive",
        metadata={
            "game_id": "tr87",
            "level_signature": sig,
            "level_index": level_index,
            "solver": "tr87_rule_table",
        },
    )


def plan_next_actions_from_state(state: dict) -> List[Tuple[GameAction, dict]]:
    global LAST_LEVEL_SIGNATURE
    LAST_LEVEL_SIGNATURE = str(state.get("level_signature", "") or "")
    result = solve_binding("tr87", state)
    if result is not None:
        _record_plan_source("primitive_binding_tr87_rule_table", result.diagnostics())
        return result.game_action_tuples(GameAction)
    _record_plan_source("unsolved", {"binding_id": "", "primitive_name": "", "plan_len": 0})
    return []


def perceive_and_plan(frame: Any) -> List[Tuple[GameAction, dict]]:
    state = perceive_tr87(frame)
    return plan_next_actions_from_state(state)
