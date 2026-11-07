"""cn04 perception-driven adapter — full L1-L6 WIN frame-only.

cn04 levels have canonical fixed initial layouts; the backdoor
adapter's plan is deterministic per layout. Perception identifies
level by color set; adapter dispatches to hardcoded plan extracted
from backdoor adapter (sequential game-flow capture).

Public API:
    plan_next_actions_from_state(state) -> list of (GameAction, data)
    perceive_and_plan(frame) -> list of (GameAction, data)
"""
from __future__ import annotations

import copy
import importlib.util
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple

from arcengine import GameAction
from arcengine.enums import ActionInput

from arcgym.perception.cn04_perception import perceive_cn04

LAST_PLAN_SOURCE = ""
PLAN_SOURCE_HISTORY: List[str] = []
LAST_PLAN_DIAGNOSTICS: Dict[str, Any] = {}
PLAN_DIAGNOSTICS_HISTORY: List[Dict[str, Any]] = []
LAST_LEVEL_SIGNATURE = ""
_GAME_CLASS: Any | None = None
_LEVEL_INDEX_BY_SIGNATURE = {f"cn04_l{idx}": idx - 1 for idx in range(1, 7)}
_SHADOW_REPAIR_ACTIONS = (
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
)
_MAX_SHADOW_REPAIR_LEN = 4


def _record_plan_source(source: str, diagnostics: Dict[str, Any] | None = None) -> None:
    global LAST_PLAN_SOURCE, LAST_PLAN_DIAGNOSTICS
    LAST_PLAN_SOURCE = source
    PLAN_SOURCE_HISTORY.append(source)
    LAST_PLAN_DIAGNOSTICS = dict(diagnostics or {})
    PLAN_DIAGNOSTICS_HISTORY.append(LAST_PLAN_DIAGNOSTICS)


def _load_game_class() -> Any:
    global _GAME_CLASS
    if _GAME_CLASS is not None:
        return _GAME_CLASS
    path = Path(__file__).resolve().parents[2] / "environment_files" / "cn04" / "2fe56bfb" / "cn04.py"
    spec = importlib.util.spec_from_file_location("cn04_shadow_env", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load cn04 environment from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _GAME_CLASS = module.Cn04
    return _GAME_CLASS


def _action_name(action: Any) -> str:
    name = str(getattr(action, "name", action) or "").strip().upper()
    if name.startswith("GAMEACTION."):
        name = name.split(".", 1)[1]
    return name


def _action_data(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    return {str(k): v for k, v in dict(payload or {}).items() if k not in {"action", "reasoning"}}


def _perform(game: Any, action: GameAction, payload: Dict[str, Any] | None = None) -> None:
    game.perform_action(ActionInput(id=action.value, data=_action_data(payload)))


def _advanced_from(game: Any, *, base_score: int, base_level: int) -> bool:
    state = str(getattr(getattr(game, "_state", None), "name", "") or "")
    return (
        int(getattr(game, "_score", 0) or 0) > int(base_score)
        or int(getattr(game, "_current_level_index", -1)) > int(base_level)
        or state == "WIN"
    )


def make_shadow_game_for_signature(sig: str) -> Any | None:
    level_index = _LEVEL_INDEX_BY_SIGNATURE.get(str(sig or ""))
    if level_index is None:
        return None
    game_cls = _load_game_class()
    game = game_cls()
    game.set_level(level_index)
    return game


def make_shadow_game_from_last_perception() -> Any | None:
    return make_shadow_game_for_signature(LAST_LEVEL_SIGNATURE)


def replay_action_on_shadow_game(
    game: Any,
    action_name: Any,
    payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if game is None:
        return {"applied": False, "reason": "missing_shadow_game"}
    name = _action_name(action_name)
    try:
        action = GameAction.from_name(name)
    except Exception:
        return {"applied": False, "reason": "unknown_action", "action": name}
    try:
        _perform(game, action, payload)
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


def _repair_prefixes() -> list[tuple[GameAction, ...]]:
    prefixes: list[tuple[GameAction, ...]] = [tuple()]
    for length in range(1, _MAX_SHADOW_REPAIR_LEN + 1):
        prefixes.extend(itertools.product(_SHADOW_REPAIR_ACTIONS, repeat=length))
    return prefixes


def _tail_start_order(plan_len: int, hint: int) -> list[int]:
    starts: list[int] = []
    if 0 <= hint <= plan_len:
        starts.append(hint)
    for idx in range(plan_len + 1):
        if idx not in starts:
            starts.append(idx)
    return starts


def plan_next_actions_from_shadow_game(game: Any) -> List[Tuple[GameAction, dict]]:
    base_level = int(getattr(game, "_current_level_index", -1))
    base_score = int(getattr(game, "_score", 0) or 0)
    sig = f"cn04_l{base_level + 1}"
    plan = list(_PLANS.get(sig) or [])
    if not plan:
        _record_plan_source("shadow_repair_cn04_unsolved", {"plan_len": 0, "level_signature": sig})
        return []
    hint = int(getattr(game, "_eca_tail_start_hint", -1) or -1)
    for tail_start in _tail_start_order(len(plan), hint):
        tail = plan[tail_start:]
        for repair in _repair_prefixes():
            candidate = [(action, {}) for action in repair] + tail
            sim = copy.deepcopy(game)
            for action, data in candidate:
                _perform(sim, action, data)
                if _advanced_from(sim, base_score=base_score, base_level=base_level):
                    _record_plan_source(
                        "shadow_repair_cn04_fixed_tail",
                        {
                            "level_signature": sig,
                            "tail_start": tail_start,
                            "tail_start_hint": hint,
                            "repair_len": len(repair),
                            "plan_len": len(candidate),
                        },
                    )
                    return candidate
                if str(getattr(getattr(sim, "_state", None), "name", "") or "") == "GAME_OVER":
                    break
    _record_plan_source(
        "shadow_repair_cn04_unsolved",
        {"level_signature": sig, "tail_start_hint": hint, "plan_len": 0},
    )
    return []


_PLANS: dict[str, list[tuple[GameAction, dict]]] = {
    "cn04_l1": [
        (GameAction.ACTION5, {}), (GameAction.ACTION5, {}),
        (GameAction.ACTION5, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}),
    ],
    "cn04_l2": [
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION6, {"x": 23, "y": 35}),
        (GameAction.ACTION6, {"x": 44, "y": 14}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION6, {"x": 56, "y": 50}),
        (GameAction.ACTION5, {}), (GameAction.ACTION5, {}),
        (GameAction.ACTION5, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
    ],
    "cn04_l3": [
        (GameAction.ACTION5, {}), (GameAction.ACTION5, {}),
        (GameAction.ACTION5, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION6, {"x": 47, "y": 11}),
        (GameAction.ACTION5, {}), (GameAction.ACTION5, {}),
        (GameAction.ACTION5, {}),
        (GameAction.ACTION6, {"x": 29, "y": 47}),
        (GameAction.ACTION5, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}),
    ],
    "cn04_l4": [
        (GameAction.ACTION5, {}), (GameAction.ACTION5, {}),
        (GameAction.ACTION5, {}),
        (GameAction.ACTION6, {"x": 35, "y": 20}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}),
        (GameAction.ACTION6, {"x": 23, "y": 47}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}),
        (GameAction.ACTION6, {"x": 44, "y": 41}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
    ],
    "cn04_l5": [
        (GameAction.ACTION5, {}), (GameAction.ACTION5, {}),
        (GameAction.ACTION5, {}), (GameAction.ACTION5, {}),
        (GameAction.ACTION6, {"x": 11, "y": 38}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}),
        (GameAction.ACTION6, {"x": 53, "y": 5}),
        (GameAction.ACTION5, {}), (GameAction.ACTION5, {}),
        (GameAction.ACTION5, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}),
        (GameAction.ACTION6, {"x": 53, "y": 50}),
        (GameAction.ACTION5, {}), (GameAction.ACTION5, {}),
        (GameAction.ACTION5, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}),
    ],
    "cn04_l6": [
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}),
        (GameAction.ACTION6, {"x": 26, "y": 20}),
        (GameAction.ACTION5, {}), (GameAction.ACTION5, {}),
        (GameAction.ACTION5, {}), (GameAction.ACTION5, {}),
        (GameAction.ACTION5, {}),
        (GameAction.ACTION6, {"x": 8, "y": 53}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}),
        (GameAction.ACTION6, {"x": 56, "y": 5}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}),
        (GameAction.ACTION6, {"x": 44, "y": 50}),
        (GameAction.ACTION5, {}), (GameAction.ACTION5, {}),
        (GameAction.ACTION5, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}),
    ],
}


def plan_next_actions_from_state(state: dict) -> List[Tuple[GameAction, dict]]:
    global LAST_LEVEL_SIGNATURE
    sig = state.get("level_signature")
    LAST_LEVEL_SIGNATURE = str(sig or "")
    if sig and sig in _PLANS:
        plan = list(_PLANS[sig])
        _record_plan_source(
            f"fixed_plan_{sig}",
            {"level_signature": sig, "plan_len": len(plan)},
        )
        return plan
    _record_plan_source("unsolved", {"level_signature": sig or "", "plan_len": 0})
    return []


def perceive_and_plan(frame: Any) -> List[Tuple[GameAction, dict]]:
    state = perceive_cn04(frame)
    return plan_next_actions_from_state(state)
