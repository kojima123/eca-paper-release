"""dc22 perception — L1 + L2 + L3 signature dispatch."""
from __future__ import annotations
import numpy as np


def perceive_dc22(frame: np.ndarray) -> dict:
    if frame.ndim == 3: frame = frame[-1]
    frame = np.asarray(frame, dtype=np.int16)
    cols = frozenset(int(c) for c in np.unique(frame) if c >= 0)
    sigs = {
        frozenset({0, 2, 3, 4, 5, 8, 9, 11, 13, 14}): "dc22_l1",
        frozenset({0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14}): "dc22_l2",
        frozenset({0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15}): "dc22_l3",
    }
    return {"level_signature": sigs.get(cols)}
