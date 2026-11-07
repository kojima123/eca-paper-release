"""bp35 frame perception (L1+L2+L3+L4 signature dispatch).

bp35 is a click-destroy/walk platformer with gravity and a moving wall
threat. Player must walk left/right and click destructible platforms
to fall down toward the gem.

This perception module identifies the level via a fixed colour
signature (frozenset of colours present in the rendered frame).
L1, L2, L3 and L4 are mapped; higher levels return signature=None
and the adapter returns an empty plan.

Frame colour palette (bp35):
  - 0   : action-counter bar / background fill
  - 3   : pattern overlay on walls (decorative speckle)
  - 5   : walls (xcjjwqfzjfe)
  - 8   : `lrpkmzabbfa` gravity-flip switches — appears starting L4
  - 9   : player sprite (player_right)
  - 10  : viewport padding (mxsayyrckip)
  - 11  : gem (fjlzdjxhant)
  - 12  : yuuqpmlxorv / oonshderxef toggle pair — appears starting L3
  - 14  : destructible platforms (qclfkhjnaac)
  - 15  : spikes (ubhhgljbnpu) — appears starting L2

The level signature uses both the colour set and the colour-14 pixel
count to disambiguate L1 from L2/L3/L4. L3 has no spikes in the
initial frame because they sit behind walls in tight pockets that
the camera viewport doesn't expose at level start. L4's initial
viewport at player y=14 doesn't expose any of the four gravity-flip
switches (they sit at y=7, y=23, y=23, y=31) so colour 8 is also
ABSENT from the L4 initial frame; L4 ends up sharing the L2 colour
set, and is disambiguated by the much smaller colour-14 pixel
count (~42 in L4 vs ~294 in L2).

Public API:
    perceive_bp35(frame) -> dict with key "level_signature"
"""
from __future__ import annotations

from typing import Any

import numpy as np


# bp35 L1 has no spikes nor toggles.
_L1_COLORS = frozenset({0, 3, 5, 9, 10, 11, 14})
# L2 introduces colour-15 spikes but no toggle pairs. L4 also shares
# this colour set at level-start because none of the L4 gravity-flip
# switches (color 8) are inside the camera viewport at spawn.
_L2_COLORS = frozenset({0, 3, 5, 9, 10, 11, 14, 15})
# L3 introduces colour-12 toggles (yuuqpmlxorv / oonshderxef share
# colour 12). Camera viewport at L3 spawn doesn't expose the spike
# row (y=4) so the colour-15 marker is absent in the initial frame.
_L3_COLORS = frozenset({0, 3, 5, 9, 10, 11, 12, 14})


def perceive_bp35(frame: Any) -> dict:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    colors = frozenset(int(c) for c in np.unique(arr).tolist())

    # Count color-14 (destructible-platform) pixels as a coarse layout
    # signature. L1's initial frame has ~147 such pixels.
    c14 = int((arr == 14).sum())

    sig = None
    if colors == _L1_COLORS and 100 <= c14 <= 200:
        sig = "bp35_l1"
    elif colors == _L2_COLORS:
        # L2 ~294 c14, L4 ~42 c14 (only 7 plats × 6 px). Disambiguate
        # by count.
        if c14 >= 100:
            sig = "bp35_l2"
        else:
            sig = "bp35_l4"
    elif colors == _L3_COLORS:
        # L3 has only 3 destructible platforms (~63 colour-14 pixels)
        # plus a row of yuuqpmlxorv toggles and oonshderxef passables
        # all rendered with shared colour 12.
        sig = "bp35_l3"

    return {
        "level_signature": sig,
        "colors": colors,
        "c14": c14,
    }
