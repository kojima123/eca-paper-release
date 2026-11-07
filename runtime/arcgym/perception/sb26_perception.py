"""sb26 perception — L1-L8 frame-only.

sb26 is a sequence-matching puzzle. Each level has:

* Top row(s) of `quhhhthrri` 6×6 quhh frames at y=1 (and y=8 for L8) — these
  define the goal sequence of colours that the validator must walk.
* One or more `pkpgflvjel`-tagged frame containers (qdmvvkvhaz4 cap=4,
  nyqgqtujsa5 cap=5, pcrvmjfjzg3 cap=3, jvkvqzheok2 cap=2 etc.) on the
  middle of the screen. Each frame has N "slot" positions at
  `(frame.x + 2 + i*6, frame.y + 2)`.
* Bottom row of `lngftsryyw` baskets (and possibly `vgszefyyyp` transitions)
  at y=56 — these are clickable and movable into slots.
* Some pre-placed items in slots (FIXED baskets or FIXED transitions).

Player workflow:
  1. ACTION6 click a bottom basket → selects it (cursor moves there).
  2. ACTION6 click a slot (susublrply at frame slot position) → moves the
     selected basket to that slot. Costs 1 energy.
  3. ACTION5 → submits the configuration. The validator does a DFS-walk
     starting at frame[0], iterating slots. Each slot must match the
     next colour in the top sequence (or be a vgszefyyyp transition that
     pushes the next frame onto the stack). Win when the last top frame
     is matched.

Level signature: top sequence (colour list), unique per level.

| level | top sequence                                  |
| ----- | --------------------------------------------- |
| L1    | (9, 14, 11, 15)                               |
| L2    | (12, 15, 8, 9, 14, 11, 6)                     |
| L3    | (8, 14, 15, 11, 6, 9, 12)                     |
| L4    | (11, 8, 14, 9, 6, 12, 15)                     |
| L5    | (6, 14, 8, 8, 14, 8, 8, 11, 15)               |
| L6    | (9, 11, 11, 12, 15, 15, 14, 6, 6)             |
| L7    | (8, 9, 14, 11, 14, 9, 8)                      |
| L8    | (8, 11, 12, 9, 14, 15, 8, 11, 12, 9, 14, 15)  |

Detection: scan row y=1 (and y=8 for L8) for blocks of 6 consecutive cells
with colour != 5 (frame edge) and != 4 (background) — the inner colour of
each quhh sprite is the visible 6×6 block at that row.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


_BACKGROUND = 4
_FRAME_EDGE = 5
_QUHH_BLOCK_LEN = 6  # quhh sprites are 6×6, inner row at y=1 and y=8
_QUHH_HORIZONTAL_PITCH = 7  # 6 cells + 1 edge


def _scan_row_for_blocks(row: np.ndarray) -> list[tuple[int, int]]:
    """Scan a row for runs of 6 consecutive cells of same colour (the inner
    of a quhh sprite). Returns list of (x_start, colour).
    """
    out: list[tuple[int, int]] = []
    n = row.shape[0]
    i = 0
    while i + _QUHH_BLOCK_LEN <= n:
        c = int(row[i])
        if c != _BACKGROUND and c != _FRAME_EDGE:
            # Check next 5 cells match
            if all(int(row[i + k]) == c for k in range(1, _QUHH_BLOCK_LEN)):
                out.append((i, c))
                i += _QUHH_BLOCK_LEN
                continue
        i += 1
    return out


def perceive_top_sequence(frame: np.ndarray) -> tuple[int, ...]:
    """Extract the goal top sequence from the rendered frame.

    Reads row y=1 (always present) and row y=8 (only L8 has a second row).
    Returns concatenated colours in left-to-right, top-to-bottom order.
    """
    frame = np.asarray(frame, dtype=np.int16)
    if frame.ndim == 3:
        frame = frame[-1]
    seq: list[int] = []
    # row 1
    blocks_r1 = _scan_row_for_blocks(frame[1])
    blocks_r1.sort(key=lambda b: b[0])
    seq.extend(c for (_, c) in blocks_r1)
    # row 8 (L8 only)
    blocks_r8 = _scan_row_for_blocks(frame[8])
    blocks_r8.sort(key=lambda b: b[0])
    seq.extend(c for (_, c) in blocks_r8)
    return tuple(seq)


# Canonical top sequences per level — used as level signature.
_LEVEL_BY_TOP_SEQ: dict[tuple[int, ...], str] = {
    (9, 14, 11, 15): "sb26_l1",
    (12, 15, 8, 9, 14, 11, 6): "sb26_l2",
    (8, 14, 15, 11, 6, 9, 12): "sb26_l3",
    (11, 8, 14, 9, 6, 12, 15): "sb26_l4",
    (6, 14, 8, 8, 14, 8, 8, 11, 15): "sb26_l5",
    (9, 11, 11, 12, 15, 15, 14, 6, 6): "sb26_l6",
    (8, 9, 14, 11, 14, 9, 8): "sb26_l7",
    (8, 11, 12, 9, 14, 15, 8, 11, 12, 9, 14, 15): "sb26_l8",
}


def perceive_sb26(frame) -> dict:
    """Return ``{"level_signature": "sb26_l<N>" | None, "top_seq": tuple}``.

    Identifies the active level by its top-row colour sequence.
    """
    seq = perceive_top_sequence(frame)
    sig = _LEVEL_BY_TOP_SEQ.get(seq)
    return {"level_signature": sig, "top_seq": seq}
