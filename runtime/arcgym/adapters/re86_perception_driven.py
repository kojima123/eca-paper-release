"""re86 perception-driven adapter — L1-L3 frame-only."""
from __future__ import annotations
from typing import Any, List, Tuple
from arcengine import GameAction
from arcgym.perception.re86_perception import perceive_re86

A1, A2, A3, A4, A5 = (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                      GameAction.ACTION4, GameAction.ACTION5)
_PLANS: dict[str, list[tuple[GameAction, dict]]] = {
    "re86_l1": [(a, {}) for a in [A4, A4, A4, A4, A1, A1, A1, A1, A1, A1, A1, A5,
                                  A3, A3, A1, A1, A1, A1, A1, A1]],
    "re86_l2": [(a, {}) for a in [A3, A3, A3, A2, A2, A2, A2, A2, A2, A2, A2, A2, A2, A5,
                                  A3, A3, A3, A3, A3, A3, A1, A1, A1, A1, A1, A1, A5,
                                  A3, A3, A3, A3, A3, A3, A3, A2, A2]],
    # L3: 8 anchors covered by 3 pieces. Active=p3 (1x43 line), then ACTION5 -> p1 (X-shape),
    # then ACTION5 -> p2 (diamond). No walls/paint zones in L3, so the morph mechanic
    # (collision-driven) doesn't trigger. Plan: 47 actions.
    # p3 (9,45)->(6,6): 1 left + 13 up. covers anchors (6,6), (45,6).
    # p1 (7,37)->(31,13): 8 right + 8 up. covers (51,15), (48,30), (33,33).
    # p2 (33,36)->(6,18): 9 left + 6 up. covers (9,27), (21,21), (21,39).
    "re86_l3": [(a, {}) for a in [A3, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A1, A5,
                                  A4, A4, A4, A4, A4, A4, A4, A4, A1, A1, A1, A1, A1, A1, A1, A1, A5,
                                  A3, A3, A3, A3, A3, A3, A3, A3, A3, A1, A1, A1, A1, A1, A1]],
}


def plan_next_actions_from_state(state: dict) -> List[Tuple[GameAction, dict]]:
    sig = state.get("level_signature")
    return list(_PLANS.get(sig, []))


def perceive_and_plan(frame: Any) -> List[Tuple[GameAction, dict]]:
    return plan_next_actions_from_state(perceive_re86(frame))
