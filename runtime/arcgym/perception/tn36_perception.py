"""tn36 frame perception — L1-L6 via signature dispatch.

L6 (added Phase 36): multi-RUN program puzzle on bzirenxmrg
with 3 chrccc checkpoints. Disambiguated from L3/L4 by colour-11
pixel count (L3=54, L4=110, L6=78); all three share the same
colour set {0,1,2,3,4,5,6,9,11}.
"""
from __future__ import annotations
import numpy as np


def perceive_tn36(frame: np.ndarray) -> dict:
    if frame.ndim == 3:
        frame = frame[-1]
    frame = np.asarray(frame, dtype=np.int16)
    cols = frozenset(int(c) for c in np.unique(frame) if c >= 0)
    c11 = int((frame == 11).sum())
    c15 = int((frame == 15).sum())
    sigs = {
        (frozenset({0, 1, 4, 5, 9, 11}), 30, 0): "tn36_l1",
        (frozenset({0, 1, 2, 3, 4, 5, 9, 11}), 42, 0): "tn36_l2",
        (frozenset({0, 1, 2, 3, 4, 5, 6, 9, 11}), 54, 0): "tn36_l3",
        (frozenset({0, 1, 2, 3, 4, 5, 6, 9, 11}), 110, 0): "tn36_l4",
        (frozenset({0, 1, 2, 3, 4, 5, 6, 9, 11, 15}), 59, 73): "tn36_l5",
        (frozenset({0, 1, 2, 3, 4, 5, 6, 9, 11}), 78, 0): "tn36_l6",
    }
    sig = sigs.get((cols, c11, c15))
    return {"level_signature": sig}
