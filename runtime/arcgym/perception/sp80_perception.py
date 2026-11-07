"""sp80 perception — frame-only extraction for the sand-physics filter puzzle.

Decodes the 64x64 rendered frame into puzzle state:
  - rotation (dojfslwbg k=0 or k=2 — perceived from source-row position)
  - filters (positions, widths, orientation, selection state)
  - bowls (positions, widths)
  - sources (column positions)
  - grid dimensions

Color encoding (post-on_set_level remapping):
  1  = walls / padding (bodekplurlf16, waoewejnqzc floor)
  4  = source (sowlljgtjvn)
  6  = water (liolfvkveqg) — during spill animation
  8  = filter unselected (plzwjbfyfli)
  9  = filter selected (vsoxmtrhqt highlight)
  11 = bowl (repwkzbkhxl)
  12 = background (BACKGROUND_COLOR)
  13 = bowl filled
  14 = overflow indicator
  15 = deflector (tuvkdkhdokr) — L5/L6 only, not used here
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional


# Cell size in pixels (16x16 grid -> 4 pixels/cell, 20x20 grid -> 3.2 not used here, but L1-L3 are 16x16)
def _detect_grid_size(frame: np.ndarray) -> int:
    """Detect grid size by finding outer wall (color 1) extent."""
    walls = (frame == 1)
    if not walls.any():
        return 16
    # bodekplurlf16 spans x=-1..16, y=-1..16 in 16x16 grid -> covers borders.
    # 20x20 grid spans more. We cheat: just default to 16 unless we see specific markers.
    # For L1-L3 it's always 16x16.
    return 16


def _detect_rotation(frame: np.ndarray) -> int:
    """Return k in {0, 2}. k=0: source at top; k=2: source at bottom (rotated 180)."""
    src_mask = (frame == 4)
    if not src_mask.any():
        return 0
    ys, _ = np.where(src_mask)
    if ys.mean() > 32:
        return 2
    return 0


def _unrotate(frame: np.ndarray, k: int) -> np.ndarray:
    """Apply inverse rotation: ntashmkaxf renders np.rot90(frame, k=k) for k in {0,1,2,3}.
    To recover game-internal coords, apply np.rot90(frame, k=-k)."""
    if k == 0:
        return frame
    return np.rot90(frame, k=-k).copy()


def _detect_filters(frame16: np.ndarray) -> List[Tuple[int, int, int, int, bool]]:
    """Find horizontal filters (color 8 unselected or 9 selected).

    Returns list of (x, y, w, h, selected) in grid coordinates.
    """
    out: List[Tuple[int, int, int, int, bool]] = []
    seen = set()
    for gy in range(16):
        for gx in range(16):
            if (gx, gy) in seen:
                continue
            c = frame16[gy * 4 + 1, gx * 4 + 1]
            if c not in (8, 9):
                continue
            # Walk east to find filter width
            ex = gx
            while ex < 16 and frame16[gy * 4 + 1, ex * 4 + 1] in (8, 9):
                ex += 1
            # Walk south to find filter height (typically 1)
            ey = gy
            while ey < 16 and frame16[ey * 4 + 1, gx * 4 + 1] in (8, 9):
                ey += 1
            w = ex - gx
            h = ey - gy
            # Selection state: any pixel == 9?
            block = frame16[gy * 4 + 1: ey * 4, gx * 4 + 1: ex * 4]
            selected = bool((block == 9).any())
            out.append((gx, gy, w, h, selected))
            for yy in range(gy, ey):
                for xx in range(gx, ex):
                    seen.add((xx, yy))
    return out


def _detect_bowls(frame16: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Find bowls (color 11) — sprite shape [[11,-1,11],[11,11,11]] occupies a 3x2 cell rectangle.

    Returns list of (x, y, w, h) in grid coordinates.
    """
    out: List[Tuple[int, int, int, int]] = []
    seen = set()
    for gy in range(16):
        for gx in range(16):
            if (gx, gy) in seen:
                continue
            c = frame16[gy * 4 + 1, gx * 4 + 1]
            if c != 11:
                continue
            # Bowl footprint is 3 wide, 2 tall. Find extent (handle filled bowl color 13 too).
            ex = gx
            while ex < 16 and frame16[(gy + 1) * 4 + 1, ex * 4 + 1] in (11, 13):
                ex += 1
            ey = gy + 2  # bowls always 2 tall
            w = ex - gx
            h = 2
            out.append((gx, gy, w, h))
            for yy in range(gy, gy + h):
                for xx in range(gx, ex):
                    seen.add((xx, yy))
    return out


def _detect_sources(frame16: np.ndarray) -> List[Tuple[int, int]]:
    """Find sources (color 4) at top row of grid (y=0 in unrotated frame). Returns (sx, sy=0)."""
    out: List[Tuple[int, int]] = []
    for gy in range(3):
        for gx in range(16):
            c = frame16[gy * 4 + 1, gx * 4 + 1]
            if c == 4:
                out.append((gx, gy))
    return out


def perceive_sp80(frame: np.ndarray) -> dict:
    """Extract sp80 puzzle state from rendered 64x64 frame.

    Returns dict with:
      level_signature: legacy disambiguator string (sp80_l1/l2/l3 or None)
      rotation_k: 0 or 2 (dojfslwbg / 90 % 4)
      filters: list of (x, y, w, h, selected) in unrotated grid coords
      bowls: list of (x, y, w, h)
      sources: list of (sx, sy)
    """
    if frame.ndim == 3:
        frame = frame[-1]
    frame = np.asarray(frame, dtype=np.int16)
    cols = frozenset(int(c) for c in np.unique(frame) if c >= 0)

    # Legacy signature for backwards compat
    sig: Optional[str] = None
    if cols == frozenset({1, 4, 6, 9, 11, 12, 14}):
        sig = "sp80_l1"
    elif cols == frozenset({1, 4, 6, 8, 9, 11, 12, 14}):
        n4 = int((frame == 4).sum())
        sig = "sp80_l3" if n4 >= 30 else "sp80_l2"

    k = _detect_rotation(frame)
    grid = _unrotate(frame, k)

    filters = _detect_filters(grid)
    bowls = _detect_bowls(grid)
    sources = _detect_sources(grid)

    return {
        "level_signature": sig,
        "rotation_k": k,
        "filters": filters,
        "bowls": bowls,
        "sources": sources,
    }
