"""sc25 frame perception — L1-L6 via signature dispatch."""
from __future__ import annotations
import numpy as np


def perceive_sc25(frame) -> dict:
    frame = np.asarray(frame, dtype=np.int16)
    if frame.ndim == 3:
        frame = frame[-1]
    cols = frozenset(int(c) for c in np.unique(frame) if c >= 0)
    sigs = {
        frozenset({0, 2, 3, 5, 9, 10, 14, 15}): "sc25_l1",
        frozenset({0, 2, 3, 4, 5, 9, 10, 11, 14}): "sc25_l2",
        frozenset({0, 2, 3, 4, 5, 6, 9, 10, 13, 14}): "sc25_l3",
        frozenset({2, 3, 5, 6, 9, 10, 13, 14, 15}): "sc25_l4",
        frozenset({2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15}): "sc25_l5",
        frozenset({2, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15}): "sc25_l6",
    }
    return {"level_signature": sigs.get(cols)}
