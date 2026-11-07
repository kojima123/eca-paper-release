"""re86 perception — L1-L3 via signature dispatch."""
from __future__ import annotations
import numpy as np


def perceive_re86(frame: np.ndarray) -> dict:
    if frame.ndim == 3: frame = frame[-1]
    frame = np.asarray(frame, dtype=np.int16)
    cols = frozenset(int(c) for c in np.unique(frame) if c >= 0)
    sigs = {
        frozenset({0, 4, 5, 9, 11, 15}): "re86_l1",
        frozenset({0, 4, 5, 9, 12, 13, 15}): "re86_l2",
        frozenset({0, 4, 5, 8, 15}): "re86_l3",
    }
    return {"level_signature": sigs.get(cols)}
