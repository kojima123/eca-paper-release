"""wa30 frame perception (Phase 10 / Phase 11).

Recovers wa30 puzzle state from the rendered 64×64 int8 frame, without
needing access to the game's internal sprite list. Color encoding
extracted from L1 frame inspection:

  14 -> player ("wbmdvjhthc")
   4 -> target sprite ("geezpjgiyd"), surrounding pixels of 4×4 block
   9 -> target dot AND fill zone marker (depending on neighbourhood)
   2 -> fill zone interior
   0 -> wall edge / padding (frame perimeter)
   1 -> background

wa30 sprites are 4 cells wide × 4 cells tall on a 4-cell grid. Player
and target positions are top-left corners of their 4×4 block.

This module exposes ``perceive_wa30(frame)`` which returns a dict:

  {
    "player_pos":      (x, y) | None,
    "target_positions": list of (x, y),
    "fill_zone_cells":  set of (x, y),
    "walls":            set of (x, y) including target positions,
    "qthdiggudy":       set of (x, y) forbidden cells (heuristic),
  }

The adapter can use these in place of `env._env._game` queries.
"""
from __future__ import annotations

import numpy as np

GRID_UNIT = 4
PLAYER_COLOR = 14
TARGET_COLOR = 4
ATTACHED_TARGET_COLOR = 3   # target's outer color when attached to player
FILL_ZONE_INTERIOR = 2
FILL_ZONE_SURROUND = 9


def _find_block_top_lefts(frame: np.ndarray, color: int,
                          block_size: int = GRID_UNIT) -> list[tuple[int, int]]:
    """Return top-left positions of contiguous blocks where the dominant
    color in each `block_size`×`block_size` window is `color`.

    Uses a coarse-grid scan stepping by `block_size`. Each cell aligned
    to (0, 0) modulo `block_size`. Counts pixels of `color` in the block;
    if > 50%, registers the top-left as a block of that color.
    """
    h, w = frame.shape
    positions: list[tuple[int, int]] = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = frame[y:y + block_size, x:x + block_size]
            count = int(np.sum(block == color))
            if count > (block_size * block_size) // 2:
                # World coords use (x, y) not (col, row), but frame is
                # indexed (row, col) = (y, x). Top-left in world = (x, y).
                positions.append((x, y))
    return positions


def _find_fill_zone_cells(frame: np.ndarray) -> set[tuple[int, int]]:
    """A fill-zone cell is a 4×4 block that contains BOTH color 2 AND
    color 9 (the fill-zone interior + surround pattern).

    For wa30 the fill zone region is one contiguous rectangle. Cells
    return as 4-aligned (x, y) in world coords matching the
    `game.wyzquhjerd` set semantics.
    """
    h, w = frame.shape
    cells: set[tuple[int, int]] = set()
    for y in range(0, h - GRID_UNIT + 1, GRID_UNIT):
        for x in range(0, w - GRID_UNIT + 1, GRID_UNIT):
            block = frame[y:y + GRID_UNIT, x:x + GRID_UNIT]
            has2 = int(np.sum(block == FILL_ZONE_INTERIOR)) > 0
            has9 = int(np.sum(block == FILL_ZONE_SURROUND)) > 0
            if has2 and has9:
                cells.add((x, y))
    return cells


def _find_edge_walls(frame: np.ndarray) -> set[tuple[int, int]]:
    """Frame perimeter walls: cells just outside the 64×64 board.

    wa30 stores edge walls in `pkbufziase` at coords (-4, *), (64, *),
    (*, -4), (*, 64) on the 4-cell grid. Reconstruct that set from
    knowledge of the frame size, since the rendered frame doesn't
    actually contain those cells.
    """
    h, w = frame.shape
    walls: set[tuple[int, int]] = set()
    # Edge cells at multiples of GRID_UNIT
    for i in range(0, h, GRID_UNIT):
        walls.add((-GRID_UNIT, i))
        walls.add((w, i))
    for i in range(0, w, GRID_UNIT):
        walls.add((i, -GRID_UNIT))
        walls.add((i, h))
    return walls


def _find_qthdiggudy_walls(frame: np.ndarray) -> set[tuple[int, int]]:
    """Detect ``pmargquscu`` wall blocks (Phase 36 L3 audit).

    The ``pmargquscu`` sprite renders as a 4x4 X-pattern with 4 of 16
    pixels showing the BACKGROUND_COLOR (1). Specifically the holes are
    at sprite-local (0,1), (1,0), (3,2), (2,3). These map to absolute
    pixel offsets within each 4x4 block. Free-space cells are 100%
    color 1. Fill zones contain colors 2+9. Targets contain color 4.
    Player contains color 14. The wall thus has a unique signature:

        - exactly 4 background pixels at the X-corners
        - 12 pixels of color 2 elsewhere

    Empty cells: 16 background pixels.
    Fill-zone interior: 16 pixels of color 2.
    """
    h, w = frame.shape
    walls: set[tuple[int, int]] = set()
    # Wall pattern positions of background pixel within 4x4 block:
    # (0, 1), (1, 0), (2, 3), (3, 2). All other 12 cells == FILL_ZONE_INTERIOR.
    BG = 1  # BACKGROUND_COLOR
    EXPECTED_BG_OFFSETS = {(0, 1), (1, 0), (2, 3), (3, 2)}
    for y in range(0, h - GRID_UNIT + 1, GRID_UNIT):
        for x in range(0, w - GRID_UNIT + 1, GRID_UNIT):
            block = frame[y:y + GRID_UNIT, x:x + GRID_UNIT]
            bg_count = int(np.sum(block == BG))
            two_count = int(np.sum(block == FILL_ZONE_INTERIOR))
            if bg_count != 4 or two_count != 12:
                continue
            # Verify the bg cells are exactly at the X-corners
            bg_offsets = {
                (int(dy), int(dx))
                for dy in range(GRID_UNIT)
                for dx in range(GRID_UNIT)
                if int(block[dy, dx]) == BG
            }
            if bg_offsets == EXPECTED_BG_OFFSETS:
                walls.add((x, y))
    return walls


def perceive_wa30(frame: np.ndarray) -> dict:
    """Extract wa30 puzzle state from a rendered frame.

    Returns:
        player_pos: (x, y) of player block top-left, or None
        target_positions: list of (x, y) for each target block top-left
        fill_zone_cells: set of (x, y) in world coords
        walls: set of (x, y) including frame edges + player + targets
                + qthdiggudy walls
        qthdiggudy: set of (x, y) for ``bnzklblgdk``-tagged wall blocks
                    (e.g. wa30 L3 column-32 wall column)
    """
    if frame.ndim == 3:
        # FrameDataRaw passes layered frames; take last layer.
        frame = frame[-1]
    frame = np.asarray(frame, dtype=np.int16)

    player_blocks = _find_block_top_lefts(frame, PLAYER_COLOR)
    player_pos = player_blocks[0] if player_blocks else None

    # Targets: color 4 outer (free) + color 3 outer (attached to player).
    # Track them separately to identify the attached target.
    free_targets = _find_block_top_lefts(frame, TARGET_COLOR)
    attached_targets = _find_block_top_lefts(frame, ATTACHED_TARGET_COLOR)
    target_positions = sorted(set(free_targets) | set(attached_targets))
    attached_target_pos: tuple[int, int] | None = (
        attached_targets[0] if attached_targets else None
    )

    fill_zone_cells = _find_fill_zone_cells(frame)
    qthdiggudy = _find_qthdiggudy_walls(frame)

    walls = _find_edge_walls(frame)
    if player_pos is not None:
        walls.add(player_pos)
    for tp in target_positions:
        walls.add(tp)
    walls.update(qthdiggudy)

    return {
        "player_pos": player_pos,
        "target_positions": target_positions,
        "attached_target_pos": attached_target_pos,
        "fill_zone_cells": fill_zone_cells,
        "walls": walls,
        "qthdiggudy": qthdiggudy,
    }
