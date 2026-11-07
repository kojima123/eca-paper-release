"""cd82 perception-driven adapter — L1-L6 (full WIN).

Mechanic recap (see ``arcgym/perception/cd82_perception.py`` for level
signature dispatch). The 10x10 canvas (`xytrjjbyib` at world (27, 34))
must match the per-level reference sprite (`eoqnvkspoa-pqwme<N>-1`
at world (3, 3)) at every cell EXCEPT both diagonals.

Painting tools:

* 8 baskets indexed 0..7 on a 3x3 minus centre grid (`nfhykrqjp`).
  ACTION1/2/3/4 = move basket selector left/right/up/down on grid;
  blocked from settling on (1, 1) (centre).
  - idx 0 = N horizontal (paints rows 0:5 = top half)
  - idx 1 = NE diagonal (paints `[i, i:10]` upper-right triangle)
  - idx 2 = E horizontal (paints cols 5:10 = right half)
  - idx 3 = SE diagonal (paints `[i, 9-i:10]` lower-right triangle)
  - idx 4 = S horizontal (paints rows 5:10 = bottom half)
  - idx 5 = SW diagonal (paints `[i, 0:i+1]` lower-left triangle)
  - idx 6 = W horizontal (paints cols 0:5 = left half)
  - idx 7 = NW diagonal (paints `[i, 0:10-i]` upper-left triangle)
* ACTION5 fires the active basket with the current paint colour.
* ACTION6 click on a `pqkenviek` swatch sets the paint colour
  (knqmgavuh). Each level has its own swatch row at y=2..7.
  - L1 swatches: (35,2)→0, (41,2)→15. Centre clicks at (37,4), (43,4).
  - L2 swatches: (32,2)→0, (38,2)→15, (44,2)→12. Centres (34,4),
    (40,4), (46,4).
  - L3-L6 swatches: 7 swatches at x=21,27,33,39,45,51,57 → colours
    0,15,12,11,14,8,9. Centres (23,4), (29,4), (35,4), (41,4),
    (47,4), (53,4), (59,4).
* ACTION6 click on `ctwspzkygu` (arrow sprite, levels 3+) triggers
  arrow paint: an off-diagonal 3x4 (N/S) or 4x3 (E/W) block:
  - N arrow (rot 180 at world (29, 18) when basket idx == 0) paints
    canvas[0:3, 3:7] = paint colour. Click centre (32, 20).
  - W arrow (rot 90 at world (11, 36) when basket idx == 6) paints
    canvas[3:7, 0:3] = paint colour. Click centre (13, 39).
  - E arrow (basket idx 2): canvas[3:7, 7:10]. Click centre depends
    on idx-2 basket position (not used in shipped plans).
  - S arrow (basket idx 4): canvas[7:10, 3:7]. Click centre depends
    on idx-4 basket position (not used in shipped plans).

Per-level decompositions (verified end-to-end via the OFFLINE env
wrapper, score 0 -> 6 in 100 actions):

L1 (8 actions): top half = 0, bottom half = 15.
L2 (13 actions): LEFT half = 0, top half = 15, SE = 12. The SE basket
  paints exactly the upper-right of the anti-diagonal which matches
  the L2 ref's 12 region.
L3 (20 actions): top + bottom = 15 (background), SE = 14, LEFT half = 8,
  NW = 15 (restores upper-left of anti-diag without erasing SE),
  N-arrow = 12. The 8 region in ref covers (6,4), (7,3-4), (8,2-4),
  (9,1-4); LEFT half=8 paints cols 0-4 of all rows = 8 then NW=15
  trims back the upper-left of anti-diag.
L4 (16 actions): top + bottom = 15 (bg), NW = 12, LEFT half = 9,
  W-arrow = 11. The 12 region is upper-left of anti-diag (NW), the
  9 region is the LEFT half cols 0-4 (overwriting NW where they
  overlap), the 11 region is rows 3-6 cols 0-2 (W-arrow).
L5 (19 actions): bg = 9, SW = 14, SE = 12, N-arrow = 8. The L5 ref
  has 14 in the lower-left (col<=row) and 12 in the upper-right
  (col>=9-row); they overlap on rows 5-9 where SE wins (painted
  later). The 8 N-arrow block sits at rows 0-2 cols 3-6.
L6 (25 actions): bg = 8, SE = 14, LEFT half = 0, NW = 8 (restores
  upper-left of anti-diag), N-arrow = 15, W-arrow = 11. The 0
  triangle (rows 6-9 specific cols) is the LEFT-half-painted region
  not overwritten by NW.

The plans assume the canonical initial state per level:
``xwmfgtlso == 0`` (north basket selected) and
``knqmgavuh == 15`` (paint colour 15). Both are reset by
``on_set_level`` after every ``next_level`` call, so chained plans
across L1->L6 start from a known basket/colour state.
"""
from __future__ import annotations

from typing import Any, List, Tuple

from arcengine import GameAction

from arcgym.perception.cd82_perception import perceive_cd82


A1 = GameAction.ACTION1
A2 = GameAction.ACTION2
A3 = GameAction.ACTION3
A4 = GameAction.ACTION4
A5 = GameAction.ACTION5
A6 = GameAction.ACTION6


def _click(x: int, y: int) -> Tuple[GameAction, dict]:
    return (A6, {"x": x, "y": y})


# Swatch click centres.
_C0_L1 = _click(37, 4)   # L1 swatch 0
_C15_L1 = _click(43, 4)  # L1 swatch 15

_C0_L2 = _click(34, 4)   # L2 swatch 0
_C15_L2 = _click(40, 4)  # L2 swatch 15
_C12_L2 = _click(46, 4)  # L2 swatch 12

_C0 = _click(23, 4)      # L3-L6 swatch 0
_C15 = _click(29, 4)     # L3-L6 swatch 15
_C12 = _click(35, 4)     # L3-L6 swatch 12
_C11 = _click(41, 4)     # L3-L6 swatch 11
_C14 = _click(47, 4)     # L3-L6 swatch 14
_C8 = _click(53, 4)      # L3-L6 swatch 8
_C9 = _click(59, 4)      # L3-L6 swatch 9

# Arrow click centres (only for the basket idx that owns each arrow).
_N_ARROW = _click(32, 20)   # N arrow at idx 0
_W_ARROW = _click(13, 39)   # W arrow at idx 6


_PLANS: dict[str, list[tuple[GameAction, dict]]] = {
    # L1 — top half = 0, bottom half = 15. 8 actions.
    "cd82_l1": [
        _C0_L1,                       # set colour 0
        (A5, {}),                     # fire idx 0 (top half) = 0
        _C15_L1,                      # set colour 15
        (A4, {}), (A2, {}),           # idx 0 -> 1 -> 2
        (A2, {}), (A3, {}),           # -> 3 -> 4
        (A5, {}),                     # fire idx 4 (bottom half) = 15
    ],
    # L2 — LEFT half = 0, top half = 15, SE = 12. 13 actions.
    "cd82_l2": [
        # idx 0 -> 6 (LEFT half): A3 A2
        (A3, {}), (A2, {}),
        _C0_L2,                       # set colour 0
        (A5, {}),                     # fire LEFT half = 0
        # idx 6 -> 0: A1 A4
        (A1, {}), (A4, {}),
        _C15_L2,                      # set colour 15
        (A5, {}),                     # fire top half = 15
        # idx 0 -> 3 (SE): A4 A2 A2
        (A4, {}), (A2, {}), (A2, {}),
        _C12_L2,                      # set colour 12
        (A5, {}),                     # fire SE = 12
    ],
    # L3 — bg = 15, SE = 14, LEFT half = 8, NW = 15, N-arrow = 12. 20 actions.
    "cd82_l3": [
        (A5, {}),                     # fire idx 0 top half = 15 (initial)
        # idx 0 -> 4: A4 A2 A2 A3
        (A4, {}), (A2, {}), (A2, {}), (A3, {}),
        (A5, {}),                     # fire bottom half = 15
        # idx 4 -> 3 (SE): A4
        (A4, {}),
        _C14,                         # colour 14
        (A5, {}),                     # fire SE = 14
        # idx 3 -> 6 (LEFT): A3 A3 A1
        (A3, {}), (A3, {}), (A1, {}),
        _C8,                          # colour 8
        (A5, {}),                     # fire LEFT = 8
        # idx 6 -> 7 (NW): A1
        (A1, {}),
        _C15,                         # colour 15
        (A5, {}),                     # fire NW = 15
        # idx 7 -> 0 (N for arrow): A4
        (A4, {}),
        _C12,                         # colour 12
        _N_ARROW,                     # click N-arrow -> paint rows 0-2 cols 3-6 = 12
    ],
    # L4 — bg = 15, NW = 12, LEFT half = 9, W-arrow = 11. 16 actions.
    "cd82_l4": [
        (A5, {}),                     # fire top half = 15
        (A4, {}), (A2, {}), (A2, {}), (A3, {}),  # idx 0 -> 4
        (A5, {}),                     # fire bottom half = 15
        # idx 4 -> 7 (NW): A3 A1 A1
        (A3, {}), (A1, {}), (A1, {}),
        _C12,                         # colour 12
        (A5, {}),                     # fire NW = 12
        # idx 7 -> 6 (LEFT): A2
        (A2, {}),
        _C9,                          # colour 9
        (A5, {}),                     # fire LEFT = 9
        # stay at idx 6 for W-arrow
        _C11,                         # colour 11
        _W_ARROW,                     # click W-arrow -> paint rows 3-6 cols 0-2 = 11
    ],
    # L5 — bg = 9, SW = 14, SE = 12, N-arrow = 8. 19 actions.
    "cd82_l5": [
        _C9,                          # colour 9
        (A5, {}),                     # fire top half = 9
        (A4, {}), (A2, {}), (A2, {}), (A3, {}),  # idx 0 -> 4
        (A5, {}),                     # fire bottom half = 9
        (A3, {}),                     # idx 4 -> 5 (SW)
        _C14,                         # colour 14
        (A5, {}),                     # fire SW = 14
        (A4, {}), (A4, {}),           # idx 5 -> 4 -> 3 (SE)
        _C12,                         # colour 12
        (A5, {}),                     # fire SE = 12
        # idx 3 -> 0 (N): A1 A1 A3
        (A1, {}), (A1, {}), (A3, {}),
        _C8,                          # colour 8
        _N_ARROW,                     # click N-arrow = 8
    ],
    # L6 — bg = 8, SE = 14, LEFT half = 0, NW = 8, N-arrow = 15, W-arrow = 11.
    # 25 actions.
    "cd82_l6": [
        _C8,                          # colour 8
        (A5, {}),                     # fire top half = 8
        (A4, {}), (A2, {}), (A2, {}), (A3, {}),  # idx 0 -> 4
        (A5, {}),                     # fire bottom half = 8
        (A4, {}),                     # idx 4 -> 3 (SE)
        _C14,                         # colour 14
        (A5, {}),                     # fire SE = 14
        (A3, {}), (A3, {}), (A1, {}), # idx 3 -> 4 -> 5 -> 6 (LEFT)
        _C0,                          # colour 0
        (A5, {}),                     # fire LEFT = 0
        (A1, {}),                     # idx 6 -> 7 (NW)
        _C8,                          # colour 8
        (A5, {}),                     # fire NW = 8
        (A4, {}),                     # idx 7 -> 0 (N for arrow)
        _C15,                         # colour 15
        _N_ARROW,                     # click N-arrow = 15
        (A3, {}), (A2, {}),           # idx 0 -> 7 -> 6 (W for arrow)
        _C11,                         # colour 11
        _W_ARROW,                     # click W-arrow = 11
    ],
}


def plan_next_actions_from_state(state: dict) -> List[Tuple[GameAction, dict]]:
    sig = state.get("level_signature")
    return list(_PLANS.get(sig, []))


def perceive_and_plan(frame: Any) -> List[Tuple[GameAction, dict]]:
    return plan_next_actions_from_state(perceive_cd82(frame))
