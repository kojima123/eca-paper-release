"""tu93 frame perception (Phase 11 + Phase 35).

Recovers tu93 puzzle state from the rendered 64×64 int8 frame, without
needing access to the game's internal sprite list.

Color encoding extracted from L1+L2 frame inspection:

   5 -> outside-field background (chrome)
   0 -> wall (within field region, blocks movement)
   2 -> walkable corridor cell (field's pixel value 2)
   9 -> walker outer (3×3 sprite)
   4 -> walker direction indicator (small dot inside walker)
  14 -> exit (3×3 sprite)
   8 -> lock outer (3×3 sprite, ``0001haidilggfh`` tag) — L2+
  15 -> lock direction indicator (single dot inside lock; pos encodes rotation)

tu93 sprites are 3-cell-wide × 3-cell-tall on a 6-cell movement grid
(walker auto-advances by 6 cells per ACTION until next 6-multiple
settling point). Walker / exit / lock positions are top-left corners
of their 3×3 block.

Walkability rule (per `tu93.py:step`): each ACTION is allowed if the
pixel at `local + (±3 in dir)` in the field == 2. ``perceive_tu93``
exposes the walkable-cell set in WORLD coordinates so the adapter's
BFS can query it directly.

Lock mechanic (Phase 35 audit): each ``0001haidilggfh`` lock has an
orientation. When the walker arrives at the cell exactly 6 cells in
front of a lock (in the lock's facing direction), the lock activates,
chases the walker along its rotation axis, and explodes the walker on
contact (game over). The lock dot at sprite-local (row, col):
  (0, 1)  -> rotation 0   (faces UP)    trigger = (lock.x, lock.y - 6)
  (1, 2)  -> rotation 90  (faces RIGHT) trigger = (lock.x + 6, lock.y)
  (2, 1)  -> rotation 180 (faces DOWN)  trigger = (lock.x, lock.y + 6)
  (1, 0)  -> rotation 270 (faces LEFT)  trigger = (lock.x - 6, lock.y)

Returns a dict:

  {
    "walker_pos": (x, y) | None,        # 3×3 top-left in world coords
    "exits":      list of (x, y),       # 3×3 top-lefts
    "walkable":   set of (x, y),        # cells where pixel value is 2
    "locks":      list of (x, y, rot),  # 3×3 top-left + rotation in {0,90,180,270}
    "trigger_cells": set of (x, y),     # cells the walker MUST avoid
  }
"""
from __future__ import annotations

import numpy as np

WALKER_SIZE = 3
WALKER_OUTER = 9
WALKER_DOT = 4
EXIT_COLOR = 14
WALKABLE_PIXEL = 2
LOCK_OUTER = 8
LOCK_DOT = 15

# Map dot offset within 3x3 lock sprite -> rotation -> trigger offset
# Dot is at one of the four edge-middles. Rotation is the lock's facing
# direction; the trigger cell sits 6 cells ahead in that direction.
_DOT_OFFSET_TO_ROT_AND_TRIG = {
    (1, 0): (0, (0, -6)),     # dot at top-middle (col=1, row=0): rotation 0, faces UP
    (2, 1): (90, (6, 0)),     # dot at right-middle (col=2, row=1): rotation 90, faces RIGHT
    (1, 2): (180, (0, 6)),    # dot at bottom-middle (col=1, row=2): rotation 180, faces DOWN
    (0, 1): (270, (-6, 0)),   # dot at left-middle (col=0, row=1): rotation 270, faces LEFT
}


def _find_block_top_lefts(frame: np.ndarray, color: int,
                          block_size: int = WALKER_SIZE,
                          stride: int = 1,
                          allow_secondary_color: int | None = None,
                          min_secondary_count: int = 0,
                          min_color_count: int | None = None) -> list[tuple[int, int]]:
    """Find top-left positions of contiguous block_size × block_size
    blocks where:
      - count of pixels matching ``color`` >= ``min_color_count``
        (default: block_size**2 - min_secondary_count, i.e., all
        non-secondary pixels are ``color``)
      - if ``allow_secondary_color`` set, count of secondary >=
        ``min_secondary_count``

    Stricter than the wa30 majority test: requires a near-pure block.
    """
    h, w = frame.shape
    block_total = block_size * block_size
    if min_color_count is None:
        min_color_count = block_total - max(1, min_secondary_count)
    positions: list[tuple[int, int]] = []
    for y in range(0, h - block_size + 1, stride):
        for x in range(0, w - block_size + 1, stride):
            block = frame[y:y + block_size, x:x + block_size]
            count = int(np.sum(block == color))
            if count < min_color_count:
                continue
            if allow_secondary_color is not None:
                sec_count = int(np.sum(block == allow_secondary_color))
                if sec_count < min_secondary_count:
                    continue
            positions.append((x, y))
    return positions


def _dedupe_overlapping(positions: list[tuple[int, int]],
                        block_size: int = WALKER_SIZE) -> list[tuple[int, int]]:
    """Deduplicate positions that overlap (BFS-style: keep first, drop
    any subsequent within block_size of an already-kept position)."""
    out: list[tuple[int, int]] = []
    for p in positions:
        keep = True
        for q in out:
            if abs(p[0] - q[0]) < block_size and abs(p[1] - q[1]) < block_size:
                keep = False
                break
        if keep:
            out.append(p)
    return out


def perceive_tu93(frame: np.ndarray) -> dict:
    """Extract tu93 puzzle state from a rendered frame.

    Returns:
        walker_pos: (x, y) of walker top-left, or None
        exits:      list of (x, y) of each exit top-left
        walkable:   set of (x, y) where frame pixel == 2 (walkable)
    """
    if frame.ndim == 3:
        frame = frame[-1]
    frame = np.asarray(frame, dtype=np.int16)

    # Walker: 3×3 block of color 9 with exactly 1 color-4 dot inside.
    # Require min 8 cells of color 9 + at least 1 of color 4.
    walker_blocks = _find_block_top_lefts(
        frame, WALKER_OUTER, block_size=WALKER_SIZE, stride=1,
        allow_secondary_color=WALKER_DOT, min_secondary_count=1,
        min_color_count=8,
    )
    walker_blocks = _dedupe_overlapping(walker_blocks, block_size=WALKER_SIZE)
    walker_pos = walker_blocks[0] if walker_blocks else None

    # Exits: 3×3 pure color 14 (all 9 cells).
    exit_blocks = _find_block_top_lefts(
        frame, EXIT_COLOR, block_size=WALKER_SIZE, stride=1,
        min_color_count=9,
    )
    exits = _dedupe_overlapping(exit_blocks, block_size=WALKER_SIZE)

    # Locks: 3×3 block of color 8 with exactly 1 color-15 dot inside.
    # Same threshold as walker (8 cells + 1 dot).
    lock_blocks = _find_block_top_lefts(
        frame, LOCK_OUTER, block_size=WALKER_SIZE, stride=1,
        allow_secondary_color=LOCK_DOT, min_secondary_count=1,
        min_color_count=8,
    )
    lock_blocks = _dedupe_overlapping(lock_blocks, block_size=WALKER_SIZE)
    locks: list[tuple[int, int, int]] = []
    trigger_cells: set[tuple[int, int]] = set()
    for (lx, ly) in lock_blocks:
        # Find the dot within the 3×3 block.
        dot_off: tuple[int, int] | None = None
        for dr in range(WALKER_SIZE):
            for dc in range(WALKER_SIZE):
                if int(frame[ly + dr, lx + dc]) == LOCK_DOT:
                    dot_off = (dc, dr)  # (col_in_sprite, row_in_sprite)
                    break
            if dot_off is not None:
                break
        if dot_off is None or dot_off not in _DOT_OFFSET_TO_ROT_AND_TRIG:
            continue
        rot, (tx_off, ty_off) = _DOT_OFFSET_TO_ROT_AND_TRIG[dot_off]
        locks.append((lx, ly, rot))
        trigger_cells.add((lx + tx_off, ly + ty_off))

    # Walkable cells: every pixel in frame with value 2.
    walkable: set[tuple[int, int]] = set()
    h, w = frame.shape
    for y in range(h):
        for x in range(w):
            if int(frame[y, x]) == WALKABLE_PIXEL:
                walkable.add((x, y))

    return {
        "walker_pos": walker_pos,
        "exits": exits,
        "walkable": walkable,
        "locks": locks,
        "trigger_cells": trigger_cells,
    }
