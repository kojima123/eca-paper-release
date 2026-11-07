"""su15 perception — L1-L3."""
from __future__ import annotations
import numpy as np


def perceive_su15(frame: np.ndarray) -> dict:
    frame = np.asarray(frame, dtype=np.int16)
    if frame.ndim == 3: frame = frame[-1]
    cols = frozenset(int(c) for c in np.unique(frame) if c >= 0)
    if cols == frozenset({0, 3, 4, 5, 9, 15}):
        return {"level_signature": "su15_l1"}
    if cols == frozenset({0, 4, 5, 6, 9, 10, 11, 15}):
        # L2 and L3 share color set; disambiguate by count of color 9
        # (aim circles): L2 has 1 (~69 cells), L3 has 2 (~138 cells).
        if int((frame == 9).sum()) > 100:
            return {"level_signature": "su15_l3"}
        return {"level_signature": "su15_l2"}
    return {"level_signature": None}
