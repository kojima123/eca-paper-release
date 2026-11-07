"""lf52 perception — L1 + L2 + L3 + L4 + L5 + L6 + L7 (signature dispatch).

Detects the lf52 layout by colour-set signature on the rendered frame.

- L1 (English-cross): palette {0, 1, 5, 9, 10, 14}.
- L2 (arrow-chain pivot): palette {0, 1, 5, 9, 10, 11, 12, 14}
  (adds 11, 12 for the {kraubslpehi} arrow tiles); 5 pegs ->
  colour-14 pixel band 50-80.
- L3 (multi-pivot, 14 pegs): same palette as L2 but ~96 colour-14
  pixels (peg pixels). Use band 81-150 to disambiguate from L2 and
  leave room for L4+ (which has additional dgxfozncuiz markers
  rendering as new colours).
- L4 (multi-region with field-scrolling triggers): palette adds 7
  and 15 -> {0, 1, 5, 7, 9, 10, 11, 12, 14, 15}.
  L4 has 6 pegs (~36 colour-14 pixels rendered).
- L5 (3 pivots, 7 markers, 6 pegs, marker-transport mechanic):
  same palette as L4. Distinguishable by colour-14 count: L5
  starts at ~24 pixels (6 pegs at smaller render) — but during
  gameplay can be higher. We dispatch L4 vs L5 via colour-14
  count band 30-50 = L4, 18-29 = L5. Phase 35 unlock relies
  on the marker-on-pivot transport mechanic.
- L6 (red peg + 7 normal pegs): palette adds colour 8
  (red peg) -> {0, 1, 5, 8, 9, 10, 11, 12, 14}. Win condition
  = peg count drops to 2.
- L7 (red peg + 2 normal pegs, 6 pivots, 15 markers): palette
  adds colours 7, 8, 15 -> {0, 1, 5, 7, 8, 9, 10, 11, 12, 14, 15}.
  Distinguished from L4/L5 by presence of colour 8 (red peg).
  Win condition = peg count drops to 2.

For L1 (tier-A runtime BFS), the perception dict additionally
returns:
- ``l1_pegs``: sorted tuple of (gx, gy) cells whose 4x4 block at
  the canonical L1 grid origin is dominated by colour 14.
- ``l1_board_cells``: sorted tuple of (gx, gy) playable holes
  (board cells whose 4x4 block is colour 1 OR colour 14).

Returns a level_signature string keyed in the adapter's _PLANS table,
or None for unrecognised layouts.
"""
from __future__ import annotations

import numpy as np

# L1 grid metadata (also captured in the adapter; duplicated here so
# perception can do all rendering-side reasoning before the adapter
# touches the frame).
_L1_GRID_ORIGIN = (10, 5)  # (ox, oy) in pixels of cell (0, 0)
_L1_TILE = 6
_L1_GRID_W = 9             # scan up to gx=8
_L1_GRID_H = 9             # scan up to gy=8
_L2_GRID_ORIGIN = (6, 8)
_L2_GRID_W = 9
_L2_GRID_H = 8
_TILE = 6


def _classify_l1_cell(frame: np.ndarray, gx: int, gy: int) -> str:
    """Return 'peg', 'hole', or 'wall' for the (gx, gy) cell on the L1
    canonical grid, based on the dominant colour of its 4x4 interior
    block at pixels [(ox+gx*6+1, oy+gy*6+1), (ox+gx*6+5, oy+gy*6+5)].
    """
    ox, oy = _L1_GRID_ORIGIN
    x0 = ox + gx * _L1_TILE + 1
    y0 = oy + gy * _L1_TILE + 1
    if x0 < 0 or y0 < 0 or x0 + 4 > frame.shape[1] or y0 + 4 > frame.shape[0]:
        return "wall"
    block = frame[y0:y0 + 4, x0:x0 + 4]
    n_peg = int((block == 14).sum())
    n_hole = int((block == 1).sum())
    if n_peg >= 8:
        return "peg"
    if n_hole >= 12:
        return "hole"
    return "wall"


def _extract_l1_layout(frame: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    pegs: list[tuple[int, int]] = []
    holes: list[tuple[int, int]] = []
    for gy in range(_L1_GRID_H):
        for gx in range(_L1_GRID_W):
            kind = _classify_l1_cell(frame, gx, gy)
            if kind == "peg":
                pegs.append((gx, gy))
                holes.append((gx, gy))  # peg cell is a board cell too
            elif kind == "hole":
                holes.append((gx, gy))
    return pegs, holes


def _extract_grid_cells_by_color(
    frame: np.ndarray,
    *,
    origin: tuple[int, int],
    width: int,
    height: int,
    color: int | tuple[int, ...],
    threshold: int,
) -> list[tuple[int, int]]:
    colors = (color,) if isinstance(color, int) else tuple(color)
    cells: list[tuple[int, int]] = []
    ox, oy = origin
    for gy in range(height):
        for gx in range(width):
            x0 = ox + gx * _TILE
            y0 = oy + gy * _TILE
            if x0 < 0 or y0 < 0 or x0 + _TILE > frame.shape[1] or y0 + _TILE > frame.shape[0]:
                continue
            block = frame[y0:y0 + _TILE, x0:x0 + _TILE]
            count = int(sum((block == c).sum() for c in colors))
            if count >= threshold:
                cells.append((gx, gy))
    return cells


def perceive_lf52(frame) -> dict:
    # Frame may be: a 64x64 np.ndarray, a 3D stack (T, H, W), or a
    # python list of per-step ndarrays (env returns the latter
    # during multi-frame animations like level transitions).
    if isinstance(frame, list) and len(frame) > 0:
        frame = frame[-1]
    if hasattr(frame, "ndim") and frame.ndim == 3:
        frame = frame[-1]
    frame = np.asarray(frame, dtype=np.int16)
    cols = frozenset(int(c) for c in np.unique(frame) if c >= 0)
    if cols == frozenset({0, 1, 5, 9, 10, 14}):
        pegs, holes = _extract_l1_layout(frame)
        return {
            "level_signature": "lf52_l1",
            "l1_pegs": tuple(sorted(pegs)),
            "l1_board_cells": tuple(sorted(holes)),
        }
    if cols == frozenset({0, 1, 5, 9, 10, 11, 12, 14}):
        # L2 has 5 pegs initially (~60 colour-14 pixels), but drops to
        # ~48/~36/... after each solved jump. L3 has a denser two-chain
        # rail layout (colours 11/12) and 14 pegs initially (~96 pixels).
        # Use rail density first so L3 remains distinguishable after peg
        # count drops; then keep L2 valid below its initial peg band.
        peg_pixels = int((frame == 14).sum())
        rail_pixels = int((frame == 11).sum() + (frame == 12).sum())
        if rail_pixels >= 60 or 81 <= peg_pixels <= 150:
            return {"level_signature": "lf52_l3"}
        if 1 <= peg_pixels <= 80:
            pegs = _extract_grid_cells_by_color(
                frame,
                origin=_L2_GRID_ORIGIN,
                width=_L2_GRID_W,
                height=_L2_GRID_H,
                color=14,
                threshold=8,
            )
            pivots = _extract_grid_cells_by_color(
                frame,
                origin=_L2_GRID_ORIGIN,
                width=_L2_GRID_W,
                height=_L2_GRID_H,
                color=(11, 12),
                threshold=12,
            )
            return {
                "level_signature": "lf52_l2",
                "l2_pegs": tuple(sorted(pegs)),
                "l2_pivots": tuple(sorted(pivots)),
            }
        return {"level_signature": None}
    if cols == frozenset({0, 1, 5, 7, 9, 10, 11, 12, 14, 15}):
        # L4 has 6 fozwvlovdui pegs in initial layout (~36 colour-14
        # pixels rendered). L5 has 6 pegs but renders fewer pixels
        # initially (~24). Phase 35 added L5 dispatch.
        # During L4 execution, scrolling/marker expansion can drop peg
        # pixels into the L5 band while rail/marker density remains much
        # higher than L5. Keep those states on L4 so prefix replanning does
        # not jump to the L5 solver before the level score advances.
        peg_pixels = int((frame == 14).sum())
        rail_pixels = int((frame == 11).sum() + (frame == 12).sum())
        marker_pixels = int((frame == 15).sum())
        if rail_pixels >= 60 or marker_pixels >= 40:
            return {"level_signature": "lf52_l4"}
        if 30 <= peg_pixels <= 50:
            return {"level_signature": "lf52_l4"}
        if 18 <= peg_pixels <= 29:
            return {"level_signature": "lf52_l5"}
        return {"level_signature": None}
    if cols == frozenset({0, 1, 5, 8, 9, 10, 11, 12, 14}):
        # L6: red peg adds colour 8. 8 pegs total (7 normal + 1 red).
        return {"level_signature": "lf52_l6"}
    if cols == frozenset({0, 1, 5, 7, 8, 9, 10, 11, 12, 14, 15}):
        # L7: red peg + multiple pivots/markers.
        return {"level_signature": "lf52_l7"}
    return {"level_signature": None}
