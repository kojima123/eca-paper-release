"""m0r0 frame perception bridge — extracts mirror-cursor puzzle state
from rendered 64x64 frames for tier-A runtime BFS solver.

Per-level layouts (origin/cell-size derived from grid_size; background
``psqw`` palette varies per level — see ``M0r0._levels.data["psqw"]``).

Color encoding (per render path in ``M0r0.uuyjjjqcvy.render_interface``):

* ``-1`` wahtyt cells (walls) and the outer letter-box render as the
  per-half ``psqw`` colors. The two halves split horizontally at
  ``x_pixel == 32``.
* ``0`` wahtyt cells render with the BACKGROUND_COLOR=5.
* ``spswjz`` (death-hole) sprites render as color 8 with a 5-color dot
  pattern overlay; the dominant color in a cell is 8.
* Cursors (``pikgci-toljda-*``, ``pikgci-boweok-*``) render as color
  10 blocks before any color_remap (they get remapped to 11/9/1 in
  selection/animation states; we still treat color 10 cells as cursors
  for L1+L2 baseline).

Public API:
    perceive_m0r0(frame) -> dict with keys:
        - level_signature: "m0r0_l1" | "m0r0_l2" | None
        - cursor_a: (gx, gy) or None
        - cursor_b: (gx, gy) or None
        - walkable: frozenset of (x, y) cells (None if signature unknown)
        - holes: frozenset of (x, y) cells (spswjz death traps)
        - grid_size: (W, H) or None
"""
from __future__ import annotations

import numpy as np

# Game-data constants (from environment_files/m0r0/492f87ba/m0r0.py).
_BACKGROUND_COLOR = 5
_CURSOR_COLOR = 10
_HOLE_COLOR = 8

# Per-level palette signatures (psqw + bg + cursor).
# L1: psqw=[11, 12], no spswjz → colors {5, 10, 11, 12}.
# L2: psqw=[ 6, 15], with spswjz → colors {5, 6, 8, 10, 15}.
_L1_SIGNATURE_COLORS = frozenset({5, 10, 11, 12})
_L1_WALL_COLORS = frozenset({11, 12})

_L2_SIGNATURE_COLORS = frozenset({5, 6, 8, 10, 15})
_L2_WALL_COLORS = frozenset({6, 15})


def _grid_geometry(grid_w: int, grid_h: int) -> tuple[int, int, int]:
    """Mirror M0r0 render at lines 649-656: scale = 64 // max(W, H);
    centred letter-box. Returns (cell_size, origin_x, origin_y).
    """
    scale = 64 // max(grid_w, grid_h)
    scaled_w = scale * grid_w
    scaled_h = scale * grid_h
    origin_x = (64 - scaled_w) // 2
    origin_y = (64 - scaled_h) // 2
    return scale, origin_x, origin_y


def _classify_cell(frame: np.ndarray, gx: int, gy: int,
                   origin_x: int, origin_y: int, cell: int) -> int:
    """Return dominant color in cell (gx, gy)'s pixel block.

    Tie-break is *not* simple argmax: ``spswjz`` death-hole cells render
    as a perfect 50/50 checkerboard of color 8 + color 5 (BACKGROUND_COLOR);
    a naive argmax tied between {5, 8} returns 5 (lower index) and the
    hole becomes invisible. Cursors render as a solid 4×4 of color 10
    so they never tie. We resolve ties by preferring the non-background
    color: if the block contains any non-5 color, return its mode;
    otherwise return BACKGROUND_COLOR.
    """
    x0 = origin_x + gx * cell
    y0 = origin_y + gy * cell
    block = frame[y0:y0 + cell, x0:x0 + cell]
    if block.size == 0:
        return -1
    flat = block.flatten()
    non_bg = flat[flat != _BACKGROUND_COLOR]
    if non_bg.size == 0:
        return _BACKGROUND_COLOR
    vals, counts = np.unique(non_bg, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _scan_grid(frame: np.ndarray, grid_w: int, grid_h: int,
               wall_colors: frozenset) -> tuple[
                   frozenset, frozenset, list[tuple[int, int]]]:
    """Scan the grid; return (walkable, holes, cursor_cells)."""
    scale, ox, oy = _grid_geometry(grid_w, grid_h)
    walkable: set[tuple[int, int]] = set()
    holes: set[tuple[int, int]] = set()
    cursor_cells: list[tuple[int, int]] = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            color = _classify_cell(frame, gx, gy, ox, oy, scale)
            if color == _CURSOR_COLOR:
                walkable.add((gx, gy))
                cursor_cells.append((gx, gy))
            elif color == _HOLE_COLOR:
                # spswjz holes: cursors CAN step on (game render still shows
                # color 8), but it triggers a death-flash that resets cursor
                # positions to initial. Treat as walkable-but-deadly.
                walkable.add((gx, gy))
                holes.add((gx, gy))
            elif color == _BACKGROUND_COLOR:
                walkable.add((gx, gy))
            elif color in wall_colors:
                # wahtyt wall (-1 cell rendered as psqw color)
                pass
            # else: unknown -> conservatively wall
    return frozenset(walkable), frozenset(holes), cursor_cells


def perceive_m0r0(frame: np.ndarray) -> dict:
    """Extract m0r0 puzzle state from a rendered frame.

    Recognises L1 (11x11) and L2 (13x13) by frame-color signature.
    Returns ``level_signature=None`` for L3+ (those levels include
    additional sprites — `mosdlc`, `gayktr`, `unobxw` — and need
    `pikgci-boweok-*` cursors plus a placeable-sprite click protocol;
    deferred until a per-level adapter is built).
    """
    if frame.ndim == 3:
        frame = frame[-1]
    frame = np.asarray(frame, dtype=np.int16)
    cols = frozenset(int(c) for c in np.unique(frame) if c >= 0)

    state: dict = {
        "level_signature": None,
        "cursor_a": None,
        "cursor_b": None,
        "walkable": None,
        "holes": frozenset(),
        "grid_size": None,
    }

    if cols == _L1_SIGNATURE_COLORS:
        state["level_signature"] = "m0r0_l1"
        grid = (11, 11)
        wall_colors = _L1_WALL_COLORS
    elif cols == _L2_SIGNATURE_COLORS:
        state["level_signature"] = "m0r0_l2"
        grid = (13, 13)
        wall_colors = _L2_WALL_COLORS
    else:
        return state

    state["grid_size"] = grid
    walkable, holes, cursor_cells = _scan_grid(
        frame, grid[0], grid[1], wall_colors,
    )
    state["walkable"] = walkable
    state["holes"] = holes

    # Cursor A is the leftmost (smallest gx); cursor B the rightmost.
    if len(cursor_cells) >= 2:
        cursor_cells.sort()
        state["cursor_a"] = cursor_cells[0]
        state["cursor_b"] = cursor_cells[-1]
    elif len(cursor_cells) == 1:
        # Single cursor visible: post-collision INTANGIBLE state.
        state["cursor_a"] = cursor_cells[0]
        state["cursor_b"] = cursor_cells[0]

    return state
