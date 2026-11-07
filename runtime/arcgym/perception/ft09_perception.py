"""ft09 perception — tier-A frame extraction (Phase 33).

ft09 is a click-only colour-cycle puzzle. Each level has:
  - a set of `bsT` template sprites (3x3 patterns with a centre `nRq`
    and surrounding `0`/non-zero markers).
  - a grid of `Hkx` (3x3 colour cells) and/or `NTi` (3x3 cells with a
    "plus" or directional brush mask) clickable cells.
  - a colour palette `gqb` (2 or 3 colours).

ACTION6 click semantics:
  - Click on Hkx at (cx, cy): cycle that single Hkx centre colour by
    +1 mod len(gqb).
  - Click on NTi at (cx, cy): cycle the NTi centre AND its 4-cardinal
    or directional neighbours (defined by the NTi sprite's `==6` mask
    PLUS implicit centre=1) by +1 mod len(gqb).

Win predicate `cgj()`: for each `bsT` template, every neighbour at the
8 surrounding positions (offset (i-1, j-1) * 4 from template centre)
must satisfy:
  - if template `pix[j][i] == 0` -> neighbour cell centre `== nRq`
  - else -> neighbour cell centre `!= nRq`

Frame extraction strategy
-------------------------
The 64×64 frame renders the 32×32 grid at scale 2 (each grid cell =
2×2 pixels). The *clickable* puzzle area is rendered with a colour-4
cell separator between sprites; *decorative* legend sprites use the
ambient colour-5 background. We discriminate by checking the 1-cell
border around each candidate 3×3 block: if the border contains colour
4 and no colour 5, the block is in the puzzle area.

The detector classifies each 3×3 puzzle block as:
  - NTi if any cell == 6 (brush flag)
  - Hkx if all 9 cells are equal (solid colour)
  - bsT otherwise (mixed pattern with centre colour = nRq)

Per-level signature (frozenset of frame colours) is also computed for
backwards-compatibility / sanity-check.
"""
from __future__ import annotations

from typing import Any

import numpy as np


_LEVEL_SIGNATURES = {
    frozenset({0, 2, 4, 5, 8, 9, 12}): "ft09_l1",
    frozenset({0, 2, 4, 9, 12}): "ft09_l2",
    frozenset({0, 2, 4, 8, 12}): "ft09_l3",
    frozenset({0, 2, 4, 8, 9, 12}): "ft09_l4",
    frozenset({0, 2, 3, 4, 6, 12, 14, 15}): "ft09_l5",
    frozenset({0, 2, 3, 4, 6, 11, 12, 14}): "ft09_l6",
}


_BG_COLORS = (4, 5)
_PUZZLE_SEPARATOR = 4
_LEGEND_BG = 5


def _extract_grid(frame: np.ndarray) -> np.ndarray:
    """Sample the 32×32 grid by reading frame[2*gy, 2*gx]."""
    grid = np.zeros((32, 32), dtype=int)
    for gy in range(32):
        for gx in range(32):
            grid[gy, gx] = frame[2 * gy, 2 * gx]
    return grid


def _border_indicates_puzzle_area(grid: np.ndarray, gx: int, gy: int) -> bool:
    """Check 1-cell border around (gx, gy)..(gx+2, gy+2).

    Returns True iff the border touches colour 4 (puzzle separator)
    and not colour 5 (legend background) — distinguishing clickable
    puzzle sprites from decorative legend visuals.
    """
    border_has_4 = False
    border_has_5 = False
    # top + bottom rows
    for dy in (-1, 3):
        for dx in range(-1, 4):
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < 32 and 0 <= ny < 32:
                c = int(grid[ny, nx])
                if c == _PUZZLE_SEPARATOR:
                    border_has_4 = True
                elif c == _LEGEND_BG:
                    border_has_5 = True
    # left + right cols
    for dy in range(-1, 4):
        for dx in (-1, 3):
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < 32 and 0 <= ny < 32:
                c = int(grid[ny, nx])
                if c == _PUZZLE_SEPARATOR:
                    border_has_4 = True
                elif c == _LEGEND_BG:
                    border_has_5 = True
    return border_has_4 and not border_has_5


def _find_puzzle_blocks(grid: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray]]:
    """Find all 3×3 puzzle-area blocks in the grid.

    A block is included iff its 9 cells are non-bg (not 4 or 5) and
    its 1-cell border indicates puzzle area (contains 4, not 5).
    """
    blocks: list[tuple[tuple[int, int], np.ndarray]] = []
    for gy in range(32 - 2):
        for gx in range(32 - 2):
            cells = grid[gy:gy + 3, gx:gx + 3]
            # All 9 cells must be non-bg (not 4 or 5)
            interior_ok = True
            for v in cells.flatten():
                if int(v) in _BG_COLORS:
                    interior_ok = False
                    break
            if not interior_ok:
                continue
            if not _border_indicates_puzzle_area(grid, gx, gy):
                continue
            blocks.append(((gx, gy), cells.copy()))
    return blocks


def _classify_block(cells: np.ndarray) -> str:
    """Return 'Hkx', 'NTi', or 'bsT' for a 3×3 puzzle block."""
    if int(np.any(cells == 6)):
        return "NTi"
    center = int(cells[1][1])
    if bool((cells == center).all()):
        return "Hkx"
    return "bsT"


def _extract_palette_legend(grid: np.ndarray) -> list[int]:
    """Extract gqb palette from the right-edge legend strip.

    From L1 (level_index=1) onwards, ft09 renders a palette legend as
    a vertical stack of 2-grid-cell blocks at the right edge of the
    32×32 grid (typically cols 30-31; L4 places it at cols 27-28).
    Each pair of identical rows represents one palette entry in cycle
    order. The block colours can be ANY palette colour but never the
    background (4 or 5).

    Returns a list of distinct colours in cycle order, deduplicated
    while preserving first-occurrence order. Returns [] if no legend
    is detected.
    """
    # Scan candidate columns for vertical legend strip
    candidates = []
    for col in (30, 28, 27, 26):
        # Check rows 0..7 for consistent non-bg blocks
        col_values = []
        for row in range(0, 12):
            v = int(grid[row, col])
            if v in _BG_COLORS:
                col_values.append(None)
            else:
                col_values.append(v)
        # Need at least 2 distinct colours stacked in pairs of rows
        # (each colour spans 2 rows since legend uses 2×2 blocks)
        non_bg = [v for v in col_values if v is not None]
        if len(non_bg) >= 2:
            candidates.append((col, col_values))

    for col, col_values in candidates:
        # Walk down extracting consecutive 2-row colour blocks
        palette: list[int] = []
        seen: set[int] = set()
        i = 0
        while i + 1 < len(col_values):
            v = col_values[i]
            v2 = col_values[i + 1]
            if v is None or v != v2:
                i += 1
                continue
            if v not in seen:
                palette.append(int(v))
                seen.add(int(v))
            i += 2
        if len(palette) >= 2:
            return palette
    return []


def _infer_palette(
    grid: np.ndarray,
    hkx_blocks: list[tuple[tuple[int, int], np.ndarray]],
    nti_blocks: list[tuple[tuple[int, int], np.ndarray]],
    bst_blocks: list[tuple[tuple[int, int], np.ndarray]],
) -> list[int]:
    """Infer gqb (palette in cycle order) from frame state.

    Strategy:
      1. Try to read the right-edge palette legend (L1-L5 frames have
         a vertical stack of 2-row colour blocks). If at least 2
         colours are found, use that as gqb.
      2. Fallback: collect colours from Hkx/NTi/bsT centres, ordered
         by [Hkx initial colour, then bsT centre colours]. This works
         for L0 which has no legend but is 2-colour.

    Cycle order matters because click count cycles through gqb in
    sequence; the initial colour of every Hkx is gqb[0] (post
    on_set_level color_remap). For 2-colour levels, swapping the
    order is harmless but for 3-colour the order constrains how
    many clicks reach each target colour.
    """
    legend = _extract_palette_legend(grid)
    if legend:
        return legend

    hkx_centers = [int(b[1][1][1]) for b in hkx_blocks]
    nti_centers = [int(b[1][1][1]) for b in nti_blocks]
    bst_centers = [int(b[1][1][1]) for b in bst_blocks]

    ordered: list[int] = []
    # The initial Hkx colour is gqb[0] post on_set_level remap.
    if hkx_centers:
        from collections import Counter
        first = Counter(hkx_centers).most_common(1)[0][0]
        ordered.append(first)
    elif nti_centers:
        from collections import Counter
        first = Counter(nti_centers).most_common(1)[0][0]
        ordered.append(first)
    for c in bst_centers + hkx_centers + nti_centers:
        if c not in ordered:
            ordered.append(c)
    return ordered


def perceive_ft09(frame: Any) -> dict:
    """Extract per-frame ft09 puzzle state from the rendered 64×64 frame.

    Returns dict with:
      - level_signature: 'ft09_l1' .. 'ft09_l6' or None
      - colors: frozenset of frame colours
      - gqb: list[int] palette in cycle order
      - hkx: list of {pos: (gx, gy), color_idx: int} (current cycle index)
      - nti: list of {pos: (gx, gy), color_idx: int, brush_mask: 3x3 list[list[bool]]}
      - bst: list of {pos: (gx, gy), nRq: int, template: 3x3 list[list[int]]}
        where template values are the bsT pixel values; pixel == 0
        means "must==nRq" and != 0 means "must!=nRq" at that offset.
    """
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    arr = np.asarray(arr)

    colors = frozenset(int(c) for c in np.unique(arr).tolist() if c >= 0)
    sig = _LEVEL_SIGNATURES.get(colors)

    grid = _extract_grid(arr)
    blocks = _find_puzzle_blocks(grid)

    hkx_blocks: list[tuple[tuple[int, int], np.ndarray]] = []
    nti_blocks: list[tuple[tuple[int, int], np.ndarray]] = []
    bst_blocks: list[tuple[tuple[int, int], np.ndarray]] = []
    for pos, cells in blocks:
        kind = _classify_block(cells)
        if kind == "Hkx":
            hkx_blocks.append((pos, cells))
        elif kind == "NTi":
            nti_blocks.append((pos, cells))
        else:
            bst_blocks.append((pos, cells))

    gqb = _infer_palette(grid, hkx_blocks, nti_blocks, bst_blocks)

    def _idx(c: int) -> int:
        try:
            return gqb.index(c)
        except ValueError:
            return -1

    hkx_list = []
    for (gx, gy), cells in hkx_blocks:
        hkx_list.append({
            "pos": (int(gx), int(gy)),
            "color": int(cells[1][1]),
            "color_idx": _idx(int(cells[1][1])),
        })

    nti_list = []
    for (gx, gy), cells in nti_blocks:
        # Brush mask: positions where cell==6 (plus implicit centre=1)
        brush = [[False, False, False], [False, False, False], [False, False, False]]
        brush[1][1] = True  # implicit centre
        # Centre colour is the non-6 colour (gqb element)
        center = -1
        for j in range(3):
            for i in range(3):
                v = int(cells[j][i])
                if v == 6:
                    brush[j][i] = True
                else:
                    if center < 0 or (j == 1 and i == 1):
                        center = v
        nti_list.append({
            "pos": (int(gx), int(gy)),
            "color": int(center),
            "color_idx": _idx(int(center)),
            "brush_mask": brush,
        })

    bst_list = []
    for (gx, gy), cells in bst_blocks:
        nRq = int(cells[1][1])
        template = [[int(cells[j][i]) for i in range(3)] for j in range(3)]
        bst_list.append({
            "pos": (int(gx), int(gy)),
            "nRq": nRq,
            "template": template,
        })

    return {
        "level_signature": sig,
        "colors": colors,
        "gqb": gqb,
        "hkx": hkx_list,
        "nti": nti_list,
        "bst": bst_list,
    }
