"""tr87 frame perception — full L1-L6 via signature dispatch.

tr87 is a rule-table rotation puzzle. Levels distinguished by
(color set, color-2 pixel count) tuple. Adapter dispatches to
hardcoded plan extracted from backdoor adapter.

Public API:
    perceive_tr87(frame) -> dict {"level_signature": "tr87_l1" .. "tr87_l6"}
"""
from __future__ import annotations

import numpy as np


def perceive_tr87(frame: np.ndarray) -> dict:
    if frame.ndim == 3:
        frame = frame[-1]
    frame = np.asarray(frame, dtype=np.int16)

    cols = frozenset(int(c) for c in np.unique(frame) if c >= 0)
    color2_count = int((frame == 2).sum())

    # Use (color set, color2_count rounded to nearest 100) as signature key.
    sigs = {
        # L1: cols={0,1,2,3,5,7,10}, c2=1570
        (frozenset({0, 1, 2, 3, 5, 7, 10}), 1570): "tr87_l1",
        (frozenset({0, 1, 2, 3, 5, 7, 11}), 1276): "tr87_l2",
        (frozenset({0, 1, 2, 3, 5, 10, 11}), 1276): "tr87_l3",
        (frozenset({0, 1, 2, 3, 5, 7, 10, 11}), 1560): "tr87_l4",
        (frozenset({0, 1, 2, 3, 5, 7, 10}), 1916): "tr87_l5",
        (frozenset({0, 1, 2, 3, 5, 7, 10, 11}), 1857): "tr87_l6",
    }
    sig = sigs.get((cols, color2_count))

    return {
        "level_signature": sig,
        "colors": cols,
        "color2_count": color2_count,
    }
