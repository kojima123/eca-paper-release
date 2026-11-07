"""sb26 perception-driven adapter — L1-L8 (full WIN, 124 actions).

See ``arcgym/perception/sb26_perception.py`` for the level-signature
dispatch (top sequence colour list).

Mechanic recap
--------------

Each level is a sequence-matching puzzle. The validator (``dbfxrigdqx``,
sb26.py:915) walks frame slots in DFS order starting at frame[0]:

* Slot contains a basket (``lngftsryyw``) → match basket.colour against
  the next colour of the top-row sequence (``wcfyiodrx[pmygakdvy]``).
  Mismatch → ``sibihgzarf()`` (energy lose).
* Slot contains a transition (``vgszefyyyp``) → push the target frame
  (matched by outer colour) onto the buvfjfmpp stack at slot 0; does
  NOT consume from the top sequence.
* End of frame's slots → pop stack to parent frame, advance parent's
  next slot.
* Win when the final top frame is matched (``pmygakdvy == len-1``).

Pre-game player workflow:
* ACTION6 click on a bottom-row basket selects it (cursor moves there).
* ACTION6 click on a frame slot (susublrply at frame.x+2+i*6, frame.y+2)
  moves the selected basket to that slot. Costs 1 energy
  (``sjcuorclg -= 1``).
* ACTION5 submits and runs the DFS validator. Initial energy = 64.

Solution strategy: pre-computed assignment per level. The plan was
derived by an offline DFS solver
(``tests/_sb26_solver_test.py``) over slot/pool placements such that the
DFS-walk produces the top sequence exactly. Re-entering a frame via a
second ``vgszefyyyp`` transition reuses already-placed baskets — this is
how levels with top-sequence > slot count (e.g. L5 has 9 colours but
only 8 slots; L8 has 12 colours with 8 slots) achieve consumption.

Per-level action counts:
  L1=9, L2=15, L3=15, L4=15, L5=17, L6=19, L7=17, L8=17. Total = 124.

The sequence of clicks per plan alternates between BOTTOM-ROW basket
selection and SLOT placement. The final action is ``ACTION5`` to submit.
"""
from __future__ import annotations

from typing import Any, List, Tuple

from arcengine import GameAction

from arcgym.perception.sb26_perception import perceive_sb26


A5 = GameAction.ACTION5
A6 = GameAction.ACTION6


def _click(x: int, y: int) -> Tuple[GameAction, dict]:
    return (A6, {"x": x, "y": y})


# ──────────────────────────────────────────────────────────────────────
# Per-level action plans (derived offline via DFS solver).
# Each plan is a list of (selector, slot, ...) pairs ending in ACTION5.
# ──────────────────────────────────────────────────────────────────────

_PLANS: dict[str, list[tuple[GameAction, dict]]] = {
    # L1 — 4 slots in single qdmvvkvhaz4 frame at (18, 25).
    # top_seq = (9, 14, 11, 15)
    "sb26_l1": [
        _click(33, 56), _click(20, 27),  # basket c=9 -> slot 0
        _click(17, 56), _click(26, 27),  # basket c=14 -> slot 1
        _click(41, 56), _click(32, 27),  # basket c=11 -> slot 2
        _click(25, 56), _click(38, 27),  # basket c=15 -> slot 3
        (A5, {}),
    ],
    # L2 — 2 frames cap=4. Frame[0] outer=8 at (18,18), Frame[1] outer=14
    # at (18, 32). Pre-fixed vgszefyyyp(c=14) at slot 2 of frame[0]
    # transitions to frame[1].
    # top_seq = (12, 15, 8, 9, 14, 11, 6)
    # Plan: frame[0] slot0=12, slot1=15, slot2=FIXED-vgs->f1; f1 all 4
    # slots 8/9/14/11; back to frame[0] slot3=6.
    "sb26_l2": [
        _click(29, 56), _click(20, 20),  # basket c=12 -> f0 slot0
        _click(15, 56), _click(26, 20),  # basket c=15 -> f0 slot1
        _click(8, 56), _click(20, 34),   # basket c=8 -> f1 slot0
        _click(43, 56), _click(26, 34),  # basket c=9 -> f1 slot1
        _click(22, 56), _click(32, 34),  # basket c=14 -> f1 slot2
        _click(50, 56), _click(38, 34),  # basket c=11 -> f1 slot3
        _click(36, 56), _click(38, 20),  # basket c=6 -> f0 slot3
        (A5, {}),
    ],
    # L3 — 3 frames. Frame[0] outer=8 cap=5 at (15, 19). Frame[1] outer=14
    # cap=2 at (15, 31). Frame[2] outer=9 cap=2 at (33, 31). Pre-fixed
    # vgszefyyyp(c=14) at f0 slot1, vgszefyyyp(c=9) at f0 slot3.
    # top_seq = (8, 14, 15, 11, 6, 9, 12)
    "sb26_l3": [
        _click(50, 56), _click(17, 21),  # basket c=8 -> f0 slot0
        _click(15, 56), _click(17, 33),  # basket c=14 -> f1 slot0
        _click(22, 56), _click(23, 33),  # basket c=15 -> f1 slot1
        _click(29, 56), _click(29, 21),  # basket c=11 -> f0 slot2
        _click(43, 56), _click(35, 33),  # basket c=6 -> f2 slot0
        _click(36, 56), _click(41, 33),  # basket c=9 -> f2 slot1
        _click(8, 56), _click(41, 21),   # basket c=12 -> f0 slot4
        (A5, {}),
    ],
    # L4 — 2 frames. Frame[0] outer=8 cap=5 at (15, 18). Frame[1] outer=14
    # cap=3 at (21, 32). Frame[1] slot 0 has a fixed lngftsryyw c=14
    # (pre-placed). Pool has movable vgszefyyyp(c=14) which transitions
    # to frame[1].
    # top_seq = (11, 8, 14, 9, 6, 12, 15)
    "sb26_l4": [
        _click(8, 56), _click(17, 20),    # basket c=11 -> f0 slot0
        _click(29, 56), _click(23, 20),   # basket c=8 -> f0 slot1
        _click(50, 56), _click(29, 20),   # vgszefyyyp(c=14) -> f0 slot2 (transitions to f1)
        _click(43, 56), _click(29, 34),   # basket c=9 -> f1 slot1 (slot0 is FIXED c=14)
        _click(15, 56), _click(35, 34),   # basket c=6 -> f1 slot2
        _click(22, 56), _click(35, 20),   # basket c=12 -> f0 slot3
        _click(36, 56), _click(41, 20),   # basket c=15 -> f0 slot4
        (A5, {}),
    ],
    # L5 — 2 frames. Frame[0] outer=8 cap=5 at (15, 18). Frame[1] outer=9
    # cap=3 at (21, 32). Pool has 2 vgszefyyyp(c=9) — both target frame[1].
    # top_seq = (6, 14, 8, 8, 14, 8, 8, 11, 15) - 9 colours; only 8 slots.
    # Solution: re-enter frame[1] twice (positions 1, 3 of f0 are vgs(c=9)).
    "sb26_l5": [
        _click(11, 56), _click(17, 20),  # basket c=6 -> f0 slot0
        _click(46, 56), _click(23, 20),  # vgszefyyyp(c=9) -> f0 slot1 (transitions to f1)
        _click(39, 56), _click(23, 34),  # basket c=14 -> f1 slot0
        _click(25, 56), _click(29, 34),  # basket c=8 -> f1 slot1
        _click(18, 56), _click(35, 34),  # basket c=8 -> f1 slot2
        # Return to f0 slot2 = vgszefyyyp(c=9) again — re-enter f1 with same baskets
        # (c=14, c=8, c=8 again match top[4..6] = (14, 8, 8)).
        _click(53, 56), _click(29, 20),  # vgszefyyyp(c=9) -> f0 slot2
        _click(32, 56), _click(35, 20),  # basket c=11 -> f0 slot3
        _click(4, 56), _click(41, 20),   # basket c=15 -> f0 slot4
        (A5, {}),
    ],
    # L6 — 4 frames cap=3 each. Frames at (8,18) outer=8, (34,18) outer=14
    # (slot0 FIXED c=14), (8,32) outer=9 (slot0 FIXED c=9), (34,32) outer=12
    # (slot0 FIXED c=12). Pool has 3 movable vgszefyyyp targeting outer
    # 14/9/12 frames — they pre-bridge into different frames.
    # top_seq = (9, 11, 11, 12, 15, 15, 14, 6, 6)
    "sb26_l6": [
        _click(50, 56), _click(10, 20),  # vgszefyyyp(c=9) -> f0 slot0 (-> frame outer=9)
        _click(22, 56), _click(16, 34),  # basket c=11 -> frame_9 slot1 (slot0 FIXED c=9)
        _click(1, 56), _click(22, 34),   # basket c=11 -> frame_9 slot2
        _click(57, 56), _click(16, 20),  # vgszefyyyp(c=12) -> f0 slot1 (-> frame outer=12)
        _click(36, 56), _click(42, 34),  # basket c=15 -> frame_12 slot1 (slot0 FIXED c=12)
        _click(15, 56), _click(48, 34),  # basket c=15 -> frame_12 slot2
        _click(43, 56), _click(22, 20),  # vgszefyyyp(c=14) -> f0 slot2 (-> frame outer=14)
        _click(29, 56), _click(42, 20),  # basket c=6 -> frame_14 slot1 (slot0 FIXED c=14)
        _click(8, 56), _click(48, 20),   # basket c=6 -> frame_14 slot2
        (A5, {}),
    ],
    # L7 — 3 frames cap=3 each at (21,12)/(21,25)/(21,38). Frame[2] outer=14
    # has FIXED basket c=11 at slot 1. Pool has movable vgszefyyyp c=14 and c=9.
    # top_seq = (8, 9, 14, 11, 14, 9, 8)
    "sb26_l7": [
        _click(18, 56), _click(23, 14),  # basket c=8 -> f0 slot0
        _click(11, 56), _click(29, 14),  # basket c=9 -> f0 slot1
        _click(53, 56), _click(35, 14),  # vgszefyyyp(c=9) -> f0 slot2 (-> f1)
        _click(46, 56), _click(23, 27),  # vgszefyyyp(c=14) -> f1 slot0 (-> f2)
        _click(39, 56), _click(23, 40),  # basket c=14 -> f2 slot0
        # f2 slot1 is FIXED c=11 (matches top[3]=11)
        _click(25, 56), _click(35, 40),  # basket c=14 -> f2 slot2
        # back to f1 slot1
        _click(32, 56), _click(29, 27),  # basket c=9 -> f1 slot1
        _click(4, 56), _click(35, 27),   # basket c=8 -> f1 slot2
        (A5, {}),
    ],
    # L8 — 2 frames cap=4 each at (18,22) outer=8 and (18,36) outer=9.
    # Pool has vgszefyyyp(c=8) and vgszefyyyp(c=9). The top sequence
    # repeats [8,11,12,9,14,15] twice — needs a re-entry chain.
    # top_seq = (8, 11, 12, 9, 14, 15, 8, 11, 12, 9, 14, 15)
    # Plan: f0 [c=8, c=11, c=12, vgs(c=9)] -> f1 [c=9, c=14, c=15, vgs(c=8)]
    # -> back to f0 (re-entry) which matches second half of top via
    # already-placed baskets.
    "sb26_l8": [
        _click(25, 56), _click(20, 24),  # basket c=8 -> f0 slot0
        _click(11, 56), _click(26, 24),  # basket c=11 -> f0 slot1
        _click(18, 56), _click(32, 24),  # basket c=12 -> f0 slot2
        _click(53, 56), _click(38, 24),  # vgszefyyyp(c=9) -> f0 slot3 (-> f1)
        _click(32, 56), _click(20, 38),  # basket c=9 -> f1 slot0
        _click(39, 56), _click(26, 38),  # basket c=14 -> f1 slot1
        _click(4, 56), _click(32, 38),   # basket c=15 -> f1 slot2
        _click(46, 56), _click(38, 38),  # vgszefyyyp(c=8) -> f1 slot3 (-> f0 re-entry)
        # Re-enter f0: slot 0 (c=8 matches top[6]=8), slot1 (c=11 matches
        # top[7]=11), slot2 (c=12 matches top[8]=12), slot3 (vgs(c=9) ->
        # f1 re-entry: slot0 c=9 matches top[9]=9, slot1 c=14 matches
        # top[10]=14, slot2 c=15 matches top[11]=15 = WIN).
        (A5, {}),
    ],
}


def plan_next_actions_from_state(state: dict) -> List[Tuple[GameAction, dict]]:
    sig = state.get("level_signature")
    return list(_PLANS.get(sig, []))


def perceive_and_plan(frame: Any) -> List[Tuple[GameAction, dict]]:
    return plan_next_actions_from_state(perceive_sb26(frame))
