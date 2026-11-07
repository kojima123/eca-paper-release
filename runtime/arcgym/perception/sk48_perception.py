"""sk48 perception — L1-L3 (signature dispatch).

sk48 is a snake-chain alignment puzzle. The active chain (chain1) has a
fixed rotation (horizontal at L1-L3) and the player can grow east, shrink
west, or shift sideways through portals. The win predicate compares the
elements picked up by chain1's segments against chain2's element-color
sequence at marker positions.

This perception module identifies the level via a fixed colour
signature (frozenset of colours present in the rendered frame). Only
L1-L3 are mapped; higher levels return signature=None and the adapter
returns an empty plan.

L1 colour palette (frame): {0, 1, 2, 3, 4, 5, 6, 8, 9, 14}
L2 colour palette (frame): {0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 14}
  (the extra colour 12 is the L2-only element c=12; L2 has 4 movable
  elements [c=14, c=9, c=12, c=8] vs L1's 3 [c=8, c=14, c=9].)
L3 colour palette (frame): {0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14}
  (the extra colour 11 marks the L3-only orthogonal source chain cap.)
"""
from __future__ import annotations

from typing import Any

import numpy as np


_L1_COLORS = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9, 14})
_L2_COLORS = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 14})
_L3_COLORS = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14})


def perceive_sk48(frame: Any) -> dict:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    colors = frozenset(int(c) for c in np.unique(arr).tolist() if c >= 0)

    sig = None
    if colors == _L1_COLORS:
        sig = "sk48_l1"
    elif colors == _L2_COLORS:
        sig = "sk48_l2"
    elif colors == _L3_COLORS:
        sig = "sk48_l3"

    return {
        "level_signature": sig,
        "colors": colors,
    }
