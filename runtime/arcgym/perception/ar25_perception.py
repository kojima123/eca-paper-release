"""ar25 frame perception — full L1-L8 multi-cursor multi-axis support.

ar25 uses a 21x21 world grid rendered at 3 frame-pixels per cell
(64x64 frame with 1-pixel padding on right/bottom edges).

Color encoding (frame):
  - 9: background.
  - 10: axis lines (vertical = column, horizontal = row).
  - 5: cursor body pixels (non-(-1) cells).
  - 0: cursor "interior" / hole pattern (rendered when sprite has 0).
  - 11: target cells (1x1).

Public API:
    perceive_ar25(frame) -> dict with keys:
        "cursors": list of dicts {"pos": (x, y), "size": (w, h), "pixels": list of (px, py)}
        "axes": list of dicts {"pos": (x, y), "size": (w, h), "kind": "V" or "H"}
        "targets": list of (x, y) target cells
        "level_signature": "ar25_l1" .. "ar25_l8" if matched

Legacy compat:
    perceive_ar25_l1(frame) -> dict with single-cursor / single-axis fields
"""
from __future__ import annotations

import numpy as np


CELL_PX = 3
AXIS_COLOR = 10
CURSOR_COLOR = 5
TARGET_COLOR = 11
BG_COLOR = 9


def _frame_to_world(fx: int, fy: int) -> tuple[int, int]:
    return fx // CELL_PX, fy // CELL_PX


def _detect_axes(frame: np.ndarray) -> list[dict]:
    """Detect axes (vertical or horizontal lines of color 10).

    Returns list of {"pos": (x, y), "kind": "V"/"H"}.
    """
    axes: list[dict] = []
    # Vertical: column with color 10 spanning >= 30 frame rows.
    col_count = (frame == AXIS_COLOR).sum(axis=0)
    visited_cols: set[int] = set()
    for fx in range(frame.shape[1]):
        if int(col_count[fx]) >= 30 and fx not in visited_cols:
            wx = fx // CELL_PX
            # Mark same world-cell columns as visited.
            for dx in range(CELL_PX):
                visited_cols.add(wx * CELL_PX + dx)
            # Find vertical extent (top and bottom non-zero column rows).
            ys = np.where(frame[:, fx] == AXIS_COLOR)[0]
            wy0 = int(ys.min()) // CELL_PX if len(ys) else 0
            wy1 = int(ys.max()) // CELL_PX if len(ys) else 0
            axes.append({
                "pos": (wx, wy0),
                "size": (1, wy1 - wy0 + 1),
                "kind": "V",
            })
    # Horizontal: row with color 10 spanning >= 30 cols.
    row_count = (frame == AXIS_COLOR).sum(axis=1)
    visited_rows: set[int] = set()
    for fy in range(frame.shape[0]):
        if int(row_count[fy]) >= 30 and fy not in visited_rows:
            wy = fy // CELL_PX
            for dy in range(CELL_PX):
                visited_rows.add(wy * CELL_PX + dy)
            xs = np.where(frame[fy, :] == AXIS_COLOR)[0]
            wx0 = int(xs.min()) // CELL_PX if len(xs) else 0
            wx1 = int(xs.max()) // CELL_PX if len(xs) else 0
            axes.append({
                "pos": (wx0, wy),
                "size": (wx1 - wx0 + 1, 1),
                "kind": "H",
            })
    return axes


def _proximity_cluster(start: tuple[int, int],
                       cells_set: set[tuple[int, int]],
                       visited: set[tuple[int, int]],
                       max_chebyshev: int = 5) -> set[tuple[int, int]]:
    """Single-linkage Chebyshev clustering: any two cells within
    `max_chebyshev` are in the same cluster.
    """
    cluster: set[tuple[int, int]] = set()
    stack = [start]
    while stack:
        c = stack.pop()
        if c in visited or c not in cells_set:
            continue
        visited.add(c)
        cluster.add(c)
        x, y = c
        for nb in cells_set:
            if nb in visited:
                continue
            if abs(nb[0] - x) <= max_chebyshev and abs(nb[1] - y) <= max_chebyshev:
                stack.append(nb)
    return cluster


def _detect_cursors(frame: np.ndarray) -> list[dict]:
    """Detect cursor sprites (color-5 clusters).

    Each cursor occupies a 3x3-frame-cell bbox in world coords. We
    convert color-5 pixels to world cells, cluster by 4-connectivity,
    and extract bbox + pixel pattern for each cluster.
    """
    H, W = frame.shape
    cursor_cells: set[tuple[int, int]] = set()
    for fy in range(min(63, H)):  # skip bottom-row padding
        for fx in range(min(63, W)):
            if frame[fy, fx] == CURSOR_COLOR:
                cursor_cells.add((fx // CELL_PX, fy // CELL_PX))

    visited: set[tuple[int, int]] = set()
    cursors: list[dict] = []
    for cell in sorted(cursor_cells):
        if cell in visited:
            continue
        # Use single-linkage proximity clustering (Chebyshev <= 5)
        # because cursors can have disconnected pixel patterns.
        cluster = _proximity_cluster(cell, cursor_cells, visited, max_chebyshev=5)
        if not cluster:
            continue
        xs = [c[0] for c in cluster]
        ys = [c[1] for c in cluster]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        # Expand cluster to include cells whose 3x3 frame block contains
        # cursor color OR color 0 (interior pattern).
        # Verify each cell in bbox.
        pixels: list[tuple[int, int]] = []
        for cy in range(y0, y1 + 1):
            for cx in range(x0, x1 + 1):
                fx0 = cx * CELL_PX
                fy0 = cy * CELL_PX
                block = frame[fy0:fy0 + CELL_PX, fx0:fx0 + CELL_PX]
                if int((block == CURSOR_COLOR).sum()) > 0 or \
                   int((block == 0).sum()) > 0:
                    pixels.append((cx - x0, cy - y0))
        cursors.append({
            "pos": (x0, y0),
            "size": (x1 - x0 + 1, y1 - y0 + 1),
            "pixels": pixels,
        })
    return cursors


def _detect_targets(frame: np.ndarray) -> list[tuple[int, int]]:
    """Detect target cells (color 11 in 3x3 frame blocks)."""
    H, W = frame.shape
    target_world: set[tuple[int, int]] = set()
    for fy in range(min(63, H)):
        for fx in range(min(63, W)):
            if frame[fy, fx] == TARGET_COLOR:
                target_world.add((fx // CELL_PX, fy // CELL_PX))
    return sorted(target_world)


def perceive_ar25(frame: np.ndarray) -> dict:
    if frame.ndim == 3:
        frame = frame[-1]
    frame = np.asarray(frame, dtype=np.int16)

    out: dict = {
        "cursors": _detect_cursors(frame),
        "axes": _detect_axes(frame),
        "targets": _detect_targets(frame),
    }

    # Level signature: target count is unique across L1-L8.
    sig_map_by_targets = {
        5: "ar25_l1",
        8: "ar25_l2",
        26: "ar25_l3",
        24: "ar25_l4",
        23: "ar25_l5",
        52: "ar25_l6",
        42: "ar25_l7",
        60: "ar25_l8",
    }
    out["level_signature"] = sig_map_by_targets.get(len(out["targets"]))
    # Per-level: which axes are "fixed". Encoded by canonical positions.
    # L1: axis at x=10 is fixed. Other levels: all axes movable.
    fixed_axes_per_level = {
        "ar25_l1": {(10, 0)},  # axis at (10, 0), kind V
    }
    fixed_set = fixed_axes_per_level.get(out["level_signature"], set())
    for ax in out["axes"]:
        ax["fixed"] = ax["pos"] in fixed_set

    return out


def perceive_ar25_l1(frame: np.ndarray) -> dict:
    """Legacy L1 single-cursor format. Map to perceive_ar25 output."""
    state = perceive_ar25(frame)
    out: dict = {
        "cursor_pos": None,
        "cursor_offsets": [],
        "axis_x": None,
        "targets": state["targets"],
    }
    if state["cursors"]:
        c = state["cursors"][0]
        out["cursor_pos"] = c["pos"]
        out["cursor_offsets"] = c["pixels"]
    for ax in state["axes"]:
        if ax["kind"] == "V":
            out["axis_x"] = ax["pos"][0]
            break
    return out
