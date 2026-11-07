"""cd82 perception — L1-L6 frame-only.

cd82 is a paint-the-canvas puzzle: 8 baskets at fixed grid positions
(arranged on a 3x3 minus center) cycle via ACTION1-4; ACTION5 fires
the active basket which paints a region of a 10x10 canvas with the
current paint color (knqmgavuh). ACTION6 click on a colour swatch
sets the paint color; ACTION6 click on the on-screen arrow sprite
(`ctwspzkygu`, levels 3+) triggers an "arrow paint" — a 3x4 / 4x3
region paint matching the cardinal direction. Win = canvas matches
the level's reference sprite (`eoqnvkspoa-pqwme<N>-1`) at every cell
EXCEPT the two diagonals (main + anti).

Each level has a fixed initial reference sprite at frame position
(3, 3), 10x10 pixels. Color set is unique per level — pure colour-set
signature dispatch suffices.

| level | reference colour set      |
| ----- | ------------------------- |
| L1    | {0, 15}                   |
| L2    | {0, 12, 15}               |
| L3    | {8, 12, 14, 15}           |
| L4    | {9, 11, 12, 15}           |
| L5    | {8, 9, 12, 14}            |
| L6    | {0, 8, 11, 14, 15}        |
"""
from __future__ import annotations

import numpy as np


_REF_TOP_LEFT = (3, 3)
_REF_SIZE = 10


_LEVEL_BY_REF_COLORS: dict[frozenset[int], str] = {
    frozenset({0, 15}): "cd82_l1",
    frozenset({0, 12, 15}): "cd82_l2",
    frozenset({8, 12, 14, 15}): "cd82_l3",
    frozenset({9, 11, 12, 15}): "cd82_l4",
    frozenset({8, 9, 12, 14}): "cd82_l5",
    frozenset({0, 8, 11, 14, 15}): "cd82_l6",
}


def perceive_cd82(frame: np.ndarray) -> dict:
    """Return ``{"level_signature": "cd82_l<N>" | None}`` from a 64x64 frame.

    Reads the 10x10 reference sprite at frame[3:13, 3:13] and matches
    its colour-set against the per-level signature table.
    """
    frame = np.asarray(frame, dtype=np.int16)
    if frame.ndim == 3:
        frame = frame[-1]
    r0, c0 = _REF_TOP_LEFT
    ref = frame[r0 : r0 + _REF_SIZE, c0 : c0 + _REF_SIZE]
    cols = frozenset(int(c) for c in np.unique(ref) if c >= 0)
    sig = _LEVEL_BY_REF_COLORS.get(cols)
    return {"level_signature": sig}
