"""s5i5 perception — frame-only extraction of pieces, sliders,
targets, indicators, and the parent->child chain needed by the
runtime BFS solver.

Game encoding (from environment_files/s5i5/18d95033/s5i5.py):
  - Pieces (`0001qwdmnlybkb`) are coloured rectangles with a
    `fbnsnwoblu = 3` marker pixel at one edge centre encoding
    rotation (top=180, bottom=0, left=90, right=270).
  - Targets (`0064ocqkuqacti`) are 3x3 sprites with one centre
    color-13 pixel — they overlap their parent piece's bbox.
  - Indicators (`0087vvmblxkzdi`) are 3x3 sprites with a 4-pixel
    color-13 diamond pattern (corners empty, edges filled, centre
    empty).
  - Sliders (`0066ghlkyvdbgg`) are 7x13 (horizontal) or 13x7
    (vertical) boxes outlined in colour 2 with a colour-3 divider
    bar splitting the box into a "shrink" and "grow" half. Each
    half contains a 3x3 colour patch of the controlled piece's
    centre colour.
  - Background colour is 5.

`perceive_s5i5(frame)` returns a structured dict with everything
the BFS solver needs to derive the win plan at runtime.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Colour constants — from s5i5.py.
_BG = 5
_MARKER = 3              # `fbnsnwoblu` — rotation marker pixel value.
_TARGET_INDICATOR = 13   # both targets and indicators use colour 13.
_SLIDER_OUTLINE = 2      # colour-2 outer ring.
_SLIDER_FILL = 4         # colour-4 fill cells.
_GRID_UNIT = 3           # `vjqemmsfmx` — slider grow step.

# A piece is a 3-cell-thick coloured rectangle; the frame-only puzzle
# uses 64x64 pixels so we cap search work.
_FRAME_W = 64
_FRAME_H = 64


def _flatten_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    return arr.astype(np.int16, copy=False)


def _is_indicator_at(frame: np.ndarray, x: int, y: int) -> bool:
    """True iff frame[y..y+3, x..x+3] is a 3x3 indicator diamond.

    Indicator pixel pattern (color 13 on -2/-2):
      [.  13 . ]
      [13 .  13]
      [.  13 . ]
    Background (-2 transparent) renders as colour 5 in the frame so
    the four diamond cells are colour 13 and the rest is 5.
    """
    if x + 3 > _FRAME_W or y + 3 > _FRAME_H or x < 0 or y < 0:
        return False
    block = frame[y:y + 3, x:x + 3]
    diamond_mask = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=bool)
    return bool(
        np.all(block[diamond_mask] == _TARGET_INDICATOR)
        and np.all(block[~diamond_mask] == _BG)
    )


def _detect_indicators(frame: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (x, y) top-left corners for all indicator
    diamonds in the frame.
    """
    indicators: List[Tuple[int, int]] = []
    # Indicators sit on bg, so only check positions adjacent to a 13.
    ys, xs = np.where(frame == _TARGET_INDICATOR)
    candidates = set()
    for cy, cx in zip(ys.tolist(), xs.tolist()):
        # The diamond's top is at (cx, cy-1) when cell is the upper-cross
        # piece etc. We just probe the 9 surrounding 3x3 origins.
        for dy in (-2, -1, 0):
            for dx in (-2, -1, 0):
                candidates.add((cx + dx, cy + dy))
    for x, y in candidates:
        if _is_indicator_at(frame, x, y):
            indicators.append((x, y))
    return sorted(indicators)


def _detect_pieces(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Detect coloured pieces with a marker stripe.

    A piece is a 3-cell-thick rectangle of body colour with a colour-3
    marker stripe along one edge (encoding rotation).

    Strategy:
      1. Flood-fill connected components of each non-{0,2,3,4,5}
         colour to find the body region.
      2. Discard regions whose bbox isn't at least 3x3 with one
         dimension exactly 3.
      3. Probe the four edges adjacent to the body for a marker
         stripe (3 colour-3 cells); the matched edge gives rotation.

    Returns list of dicts with keys: x, y, w, h, color, rotation.
    Coordinates include the marker stripe (so bbox matches the
    legacy `_piece_state_bbox` semantics).
    """
    pieces: List[Dict[str, Any]] = []
    visited = np.zeros_like(frame, dtype=bool)

    # Body colours: any colour that is not {bg, marker, slider-fill,
    # slider-outline, target/indicator-13}. We DO NOT include 13
    # because targets/indicators use it; pieces don't.
    body_mask = (
        (frame >= 0)
        & (frame != _BG)
        & (frame != _MARKER)
        & (frame != _SLIDER_OUTLINE)
        & (frame != _SLIDER_FILL)
        & (frame != _TARGET_INDICATOR)
    )

    ys, xs = np.where(body_mask)
    for cy, cx in zip(ys.tolist(), xs.tolist()):
        if visited[cy, cx]:
            continue
        body_color = int(frame[cy, cx])

        # Flood-fill body cells of the same colour only (no markers).
        stack = [(cy, cx)]
        cells: List[Tuple[int, int]] = []
        while stack:
            yy, xx = stack.pop()
            if not (0 <= yy < _FRAME_H and 0 <= xx < _FRAME_W):
                continue
            if visited[yy, xx]:
                continue
            if int(frame[yy, xx]) != body_color:
                continue
            visited[yy, xx] = True
            cells.append((yy, xx))
            stack.extend([(yy + 1, xx), (yy - 1, xx),
                          (yy, xx + 1), (yy, xx - 1)])
        if not cells:
            continue
        cell_ys = [c[0] for c in cells]
        cell_xs = [c[1] for c in cells]
        body_y0, body_y1 = min(cell_ys), max(cell_ys) + 1
        body_x0, body_x1 = min(cell_xs), max(cell_xs) + 1
        bh = body_y1 - body_y0
        bw = body_x1 - body_x0

        # A piece's body has one dim == 3 (the constant cell-unit
        # thickness) and the other dim in {2, 5, 8, 11, ...} (length
        # in cells minus 1 marker row/col). Initial body is 2x3 or 3x2.
        if not ((bh == 3 and bw >= 2) or (bw == 3 and bh >= 2)):
            continue

        # Probe the four edges for a marker stripe of length >= 3.
        # rotation 0   -> stripe BELOW body (bottom-fixed → grows up)
        # rotation 90  -> stripe LEFT  of body (left-fixed → grows right)
        # rotation 180 -> stripe ABOVE body (top-fixed → grows down)
        # rotation 270 -> stripe RIGHT of body (right-fixed → grows left)
        def _stripe_marker(y0: int, y1: int, x0: int, x1: int) -> bool:
            if y0 < 0 or x0 < 0 or y1 > _FRAME_H or x1 > _FRAME_W:
                return False
            return bool(np.all(frame[y0:y1, x0:x1] == _MARKER))

        rotation: Optional[int] = None
        bbox_y0, bbox_y1 = body_y0, body_y1
        bbox_x0, bbox_x1 = body_x0, body_x1
        if _stripe_marker(body_y1, body_y1 + 1, body_x0, body_x1):
            rotation = 0
            bbox_y1 = body_y1 + 1
        elif _stripe_marker(body_y0, body_y1, body_x0 - 1, body_x0):
            rotation = 90
            bbox_x0 = body_x0 - 1
        elif _stripe_marker(body_y0 - 1, body_y0, body_x0, body_x1):
            rotation = 180
            bbox_y0 = body_y0 - 1
        elif _stripe_marker(body_y0, body_y1, body_x1, body_x1 + 1):
            rotation = 270
            bbox_x1 = body_x1 + 1

        if rotation is None:
            continue

        pieces.append({
            "x": bbox_x0, "y": bbox_y0,
            "w": bbox_x1 - bbox_x0, "h": bbox_y1 - bbox_y0,
            "color": body_color, "rotation": rotation,
        })
    return pieces


def _detect_targets(frame: np.ndarray, pieces: List[Dict[str, Any]],
                     indicators: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """Detect 1-pixel collision targets (`0064ocqkuqacti`).

    Each target appears as a single colour-13 pixel that lies INSIDE a
    piece's bbox (otherwise it would be either an indicator diamond
    pixel or part of one). The sprite is 3x3 with origin one cell
    above-left of the visible centre pixel.
    """
    indicator_pixel_set = set()
    for (ix, iy) in indicators:
        # Indicator's color-13 cells: (ix+1, iy), (ix, iy+1), (ix+2, iy+1), (ix+1, iy+2).
        for px, py in [(ix + 1, iy), (ix, iy + 1),
                        (ix + 2, iy + 1), (ix + 1, iy + 2)]:
            indicator_pixel_set.add((px, py))

    targets: List[Dict[str, Any]] = []
    ys, xs = np.where(frame == _TARGET_INDICATOR)
    for cy, cx in zip(ys.tolist(), xs.tolist()):
        if (cx, cy) in indicator_pixel_set:
            continue
        # Must lie strictly inside some piece bbox.
        inside_piece: Optional[Dict[str, Any]] = None
        for p in pieces:
            if p["x"] <= cx < p["x"] + p["w"] and p["y"] <= cy < p["y"] + p["h"]:
                inside_piece = p
                break
        if inside_piece is None:
            continue
        # Sprite origin = (cx - 1, cy - 1).
        targets.append({
            "x": cx - 1, "y": cy - 1, "w": 3, "h": 3,
            "parent_idx": pieces.index(inside_piece),
        })
    return targets


def _detect_sliders(frame: np.ndarray, pieces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect 7x13 / 13x7 slider boxes and link them to their piece.

    A slider is a rectangle of colour-2 outline with colour-4 fill and
    a colour-3 divider bar through the middle. The shrink/grow halves
    each contain a 3x3 patch of the controlled piece's centre colour.

    Returns list of dicts with keys: x, y, w, h, controls_color, click.
      `click` is the (x, y) pixel for the GROW half (per
      `_slider_click` in the legacy adapter).
    """
    sliders: List[Dict[str, Any]] = []
    visited = np.zeros_like(frame, dtype=bool)
    ys, xs = np.where(frame == _SLIDER_OUTLINE)
    if len(ys) == 0:
        return sliders

    # Group connected color-2 cells into bounding boxes.
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        cells = []
        while stack:
            yy, xx = stack.pop()
            if not (0 <= yy < _FRAME_H and 0 <= xx < _FRAME_W):
                continue
            if visited[yy, xx]:
                continue
            if frame[yy, xx] != _SLIDER_OUTLINE:
                continue
            visited[yy, xx] = True
            cells.append((yy, xx))
            stack.extend([(yy + 1, xx), (yy - 1, xx),
                          (yy, xx + 1), (yy, xx - 1)])
        if not cells:
            continue
        cell_ys = [c[0] for c in cells]
        cell_xs = [c[1] for c in cells]
        y0, y1 = min(cell_ys), max(cell_ys) + 1
        x0, x1 = min(cell_xs), max(cell_xs) + 1
        h, w = y1 - y0, x1 - x0

        # Slider dimensions are exactly 7x13 (horizontal) or 13x7 (vertical).
        if not ((h == 7 and w == 13) or (h == 13 and w == 7)):
            continue

        # Controlled-piece colour: the most common non-{2,3,4,5} colour
        # inside the slider's bbox.
        sub = frame[y0:y1, x0:x1]
        counts: Dict[int, int] = {}
        for v in sub.flatten().tolist():
            if v in (_SLIDER_OUTLINE, _MARKER, _SLIDER_FILL, _BG):
                continue
            if v < 0:
                continue
            counts[v] = counts.get(v, 0) + 1
        if not counts:
            continue
        controls_color = max(counts.items(), key=lambda kv: kv[1])[0]

        horizontal = w > h
        if horizontal:
            # Right half grows: click at (x0 + w - 1, y0 + h // 2).
            click = (x0 + w - 1, y0 + h // 2)
            # Left half shrinks: click at (x0, y0 + h // 2).
            shrink_click = (x0, y0 + h // 2)
        else:
            # Bottom half grows: click at (x0 + w // 2, y0 + h - 1).
            click = (x0 + w // 2, y0 + h - 1)
            # Top half shrinks: click at (x0 + w // 2, y0).
            shrink_click = (x0 + w // 2, y0)

        sliders.append({
            "x": x0, "y": y0, "w": w, "h": h,
            "controls_color": controls_color,
            "click": click,
            "shrink_click": shrink_click,
            "horizontal": horizontal,
        })
    return sliders


def _build_chain_links(pieces: List[Dict[str, Any]]) -> Dict[int, Optional[int]]:
    """For each growable piece, find its child piece (or None).

    A piece's child is the piece whose bbox abuts its grow-side edge.
    Encoded as parent_idx -> child_idx (or None).
    """
    rotation_to_grow_dir = {
        0:   (0, -1),  # grows up
        90:  (1, 0),   # grows right
        180: (0, 1),   # grows down
        270: (-1, 0),  # grows left
    }
    parent_to_child: Dict[int, Optional[int]] = {}
    for i, p in enumerate(pieces):
        dx, dy = rotation_to_grow_dir[p["rotation"]]
        # Probe cell just past the grow-edge centre.
        if dx == 1:    # grows right
            probe_x = p["x"] + p["w"]
            probe_y = p["y"] + p["h"] // 2
        elif dx == -1:  # grows left
            probe_x = p["x"] - 1
            probe_y = p["y"] + p["h"] // 2
        elif dy == 1:  # grows down
            probe_x = p["x"] + p["w"] // 2
            probe_y = p["y"] + p["h"]
        else:           # grows up
            probe_x = p["x"] + p["w"] // 2
            probe_y = p["y"] - 1

        child_idx: Optional[int] = None
        for j, q in enumerate(pieces):
            if i == j:
                continue
            if (q["x"] <= probe_x < q["x"] + q["w"]
                    and q["y"] <= probe_y < q["y"] + q["h"]):
                child_idx = j
                break
        parent_to_child[i] = child_idx
    return parent_to_child


def _detect_obstacles(frame: np.ndarray,
                       pieces: List[Dict[str, Any]],
                       sliders: List[Dict[str, Any]]) -> np.ndarray:
    """Build a 64x64 boolean mask of "wall" cells that pieces must
    avoid.

    Walls are non-bg cells that don't belong to any detected piece,
    target, indicator, or slider region. The two large 0001qwdmnlybkb
    "frame" sprites at L2 use colour 15 for their wall pixels, but we
    treat ANY non-empty cell outside known sprite regions as an
    obstacle so the detector also handles unanticipated frame colours.
    """
    h, w = frame.shape
    obstacle = (frame >= 0) & (frame != _BG)
    # Mark known sprite regions as NOT obstacles so legitimate piece
    # cells aren't treated as walls.
    for p in pieces:
        x0, y0 = p["x"], p["y"]
        x1, y1 = x0 + p["w"], y0 + p["h"]
        obstacle[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = False
    for s in sliders:
        x0, y0 = s["x"], s["y"]
        x1, y1 = x0 + s["w"], y0 + s["h"]
        obstacle[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = False
    # Indicator/target colour-13 cells aren't walls.
    obstacle &= (frame != _TARGET_INDICATOR)
    # Marker colour 3 cells aren't walls (they belong to pieces or
    # the bottom decorative bar at row 63).
    obstacle &= (frame != _MARKER)
    return obstacle


def perceive_s5i5(frame: Any) -> Dict[str, Any]:
    """Extract puzzle structure from the rendered frame.

    Returns a dict with keys:
      - level_signature: legacy "s5i5_l1"/"s5i5_l2" tag (kept for any
        downstream consumer that still uses it).
      - pieces: list of {x, y, w, h, color, rotation}.
      - targets: list of {x, y, w, h, parent_idx}.
      - indicators: list of (x, y) — the indicator origins.
      - sliders: list of {x, y, w, h, controls_color, click,
                          horizontal}.
      - parent_to_child: dict mapping piece-idx -> piece-idx or None.
      - obstacle_mask: HxW boolean grid of static walls (frames).
    """
    arr = _flatten_frame(frame)
    cols = frozenset(int(c) for c in np.unique(arr) if c >= 0)

    sigs = {
        frozenset({2, 3, 4, 5, 11, 13, 14}): "s5i5_l1",
        frozenset({2, 3, 4, 5, 10, 11, 12, 13, 14, 15}): "s5i5_l2",
        # L3 has 6 coloured slidable pieces (c=7,8,9,10,12,14) + colour-1
        # follower stripes + colour-15 obstacle/walls.
        frozenset({1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15}): "s5i5_l3",
    }
    sig = sigs.get(cols)

    indicators = _detect_indicators(arr)
    pieces = _detect_pieces(arr)
    targets = _detect_targets(arr, pieces, indicators)
    sliders = _detect_sliders(arr, pieces)
    parent_to_child = _build_chain_links(pieces)
    obstacle = _detect_obstacles(arr, pieces, sliders)

    return {
        "level_signature": sig,
        "pieces": pieces,
        "targets": targets,
        "indicators": indicators,
        "sliders": sliders,
        "parent_to_child": parent_to_child,
        "obstacle_mask": obstacle,
    }
