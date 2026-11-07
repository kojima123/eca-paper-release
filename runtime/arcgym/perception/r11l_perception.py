"""r11l frame perception (Phase 23.1 G2 canary, Phase 24 extension to L1-L4).

Recovers r11l puzzle state from a rendered 64x64 int8 frame, without
accessing `env._env._game`.

Sprite color invariants (per L1-L4 inspection):
  - Each "group" has a body color that all its sprites share:
      orrqlj → 7 (L2), 12 (L2 with remap)... actually depends on level.
      pumlzd → 15 (most levels).
      The body color identifies the group.
  - 5x5 sprites:
      ROEFWU (canvas):   body color + single center pixel = 6.
      ROEFWULEWCUI ACTIVE piece: body color + outer ring = 0.
      ROEFWULEWCUI INACTIVE piece: body color + outer ring = 3.
  - 7x7 sprites:
      FLKDTG (target): body color (sparse) + transparency.
  - Walls:
      `wakneh-*`: color 2, frame perimeter (typically (-X, -Y, large)).
      `defgjl-*`: color 10, internal forbidden region for canvas.

Public API:
    perceive_r11l(frame) -> dict with keys:
        "canvases": list of dicts {"pos": (x, y), "color": c}
        "active_piece": (x, y) | None
        "inactive_pieces": list of dicts {"pos": (x, y), "color": c}
        "targets": list of dicts {"pos": (x, y), "color": c}  # 7x7 top-lefts
        "wakneh_cells": set of (x, y) — color-2 frame cells (1-pixel)
        "defgjl_cells": set of (x, y) — color-10 frame cells (1-pixel)

For convenience:
    perceive_r11l_l1(frame) -> dict (legacy L1 single-group format)
"""
from __future__ import annotations

import numpy as np

PIECE_W = PIECE_H = 5
TARGET_W = TARGET_H = 7
CANVAS_CENTER_COLOR = 6     # roefwu's centre dot
ACTIVE_RING_COLOR = 0       # active piece outer ring
INACTIVE_RING_COLOR = 3     # inactive piece outer ring
WAKNEH_COLOR = 2
DEFGJL_COLOR = 10
BG_COLOR = 5


def _cluster_bbox(pts: list[tuple[int, int]],
                  max_dist: int = 4) -> list[list[tuple[int, int]]]:
    """Group points into clusters where any two within `max_dist` cells
    in both x and y are in the same cluster (single-linkage by Chebyshev).
    """
    if not pts:
        return []
    pts = sorted(set(pts))
    visited = [False] * len(pts)
    clusters: list[list[tuple[int, int]]] = []
    for i in range(len(pts)):
        if visited[i]:
            continue
        # BFS cluster expansion.
        stack = [i]
        cluster = []
        while stack:
            j = stack.pop()
            if visited[j]:
                continue
            visited[j] = True
            cluster.append(pts[j])
            for k in range(len(pts)):
                if visited[k]:
                    continue
                if (abs(pts[k][0] - pts[j][0]) <= max_dist and
                        abs(pts[k][1] - pts[j][1]) <= max_dist):
                    stack.append(k)
        clusters.append(cluster)
    return clusters


def _bbox_top_left(cluster: list[tuple[int, int]],
                   block_w: int, block_h: int) -> tuple[int, int] | None:
    """Given a cluster of pixels, return the top-left of a `block_w x
    block_h` block that contains them all, snapped so all pixels fit.
    Returns None if cluster spans larger than block.
    """
    xs = [p[0] for p in cluster]
    ys = [p[1] for p in cluster]
    if max(xs) - min(xs) >= block_w or max(ys) - min(ys) >= block_h:
        return None
    tl_x = min(xs)
    tl_y = min(ys)
    if max(xs) > tl_x + block_w - 1:
        tl_x = max(xs) - (block_w - 1)
    if max(ys) > tl_y + block_h - 1:
        tl_y = max(ys) - (block_h - 1)
    return (tl_x, tl_y)


_GROUP_COLORS = {7, 8, 9, 11, 12, 14, 15}  # observed body colors in L1-L4


def _body_color_at_center(frame: np.ndarray,
                          pos: tuple[int, int]) -> int | None:
    """Body color = pixel at the center of the block (most reliable
    for pieces, where center is body and ring is 0/3).
    """
    x, y = pos
    cx, cy = x + 2, y + 2
    if not (0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]):
        return None
    val = int(frame[cy, cx])
    return val if val in _GROUP_COLORS else None


def _body_color_canvas(frame: np.ndarray,
                       pos: tuple[int, int]) -> int | None:
    """Canvas body color = pixel at corner (0, 0) of the canvas block,
    since the canvas has body color in outer ring and color 6 only at
    centre.
    """
    x, y = pos
    if not (0 <= x and x + 5 <= frame.shape[1] and 0 <= y and y + 5 <= frame.shape[0]):
        return None
    block = frame[y:y + 5, x:x + 5]
    counts: dict[int, int] = {}
    for v in np.unique(block):
        v = int(v)
        if v in _GROUP_COLORS:
            counts[v] = int((block == v).sum())
    if not counts:
        return None
    return max(counts, key=lambda c: counts[c])


def perceive_r11l(frame: np.ndarray) -> dict:
    if frame.ndim == 3:
        frame = frame[-1]
    frame = np.asarray(frame, dtype=np.int16)

    out: dict = {
        "canvases": [],
        "active_piece": None,
        "active_piece_color": None,
        "inactive_pieces": [],
        "targets": [],
        "wakneh_cells": set(),
        "defgjl_cells": set(),
    }

    # --- Wakneh: cells with color 2 ---
    wak_ys, wak_xs = np.where(frame == WAKNEH_COLOR)
    out["wakneh_cells"] = set(zip(wak_xs.tolist(), wak_ys.tolist()))

    # --- Defgjl: visible color-10 cells. NB: cells under existing
    # sprites (pieces/canvases/targets) are not visible, so this set
    # UNDERESTIMATES the true defgjl shape. Adapter logic should handle
    # that conservatively (e.g., reject paths that visibly fail).
    def_ys, def_xs = np.where(frame == DEFGJL_COLOR)
    out["defgjl_cells"] = set(zip(def_xs.tolist(), def_ys.tolist()))

    # --- Canvases: color-6 pixels are canvas centres OR hawffu-* sprite
    # body. Real canvases have an ISOLATED color-6 pixel (single dot in
    # their 5x5 region). hawffu sprites color_remap'd to 6 have many
    # color-6 pixels clustered together. Filter by counting color-6
    # within a 5x5 window around each candidate.
    six_ys, six_xs = np.where(frame == CANVAS_CENTER_COLOR)
    six_pts = list(zip(six_xs.tolist(), six_ys.tolist()))
    canvases: list[dict] = []
    for cx, cy in six_pts:
        tl = (cx - 2, cy - 2)
        if not (0 <= tl[0] <= 64 - PIECE_W and 0 <= tl[1] <= 64 - PIECE_H):
            continue
        block = frame[tl[1]:tl[1] + PIECE_H, tl[0]:tl[0] + PIECE_W]
        six_count = int((block == CANVAS_CENTER_COLOR).sum())
        if six_count != 1:
            continue
        color = _body_color_canvas(frame, tl)
        if color is not None:
            # Capture all group colors in this canvas (multi-morpheme
            # groups have multiple body colors).
            canvas_colors = {int(v) for v in np.unique(block)
                             if int(v) in _GROUP_COLORS}
            # The canvas's "piece color" (last morpheme color) appears
            # in the BOTTOM row of the canvas sprite. Extract it.
            bottom_row = block[-1]
            piece_color_hint = None
            for v in bottom_row:
                v = int(v)
                if v in _GROUP_COLORS:
                    piece_color_hint = v
                    break
            # Fallback: use second-bottom row if bottom is all -1.
            if piece_color_hint is None:
                for v in block[-2]:
                    v = int(v)
                    if v in _GROUP_COLORS:
                        piece_color_hint = v
                        break
            if piece_color_hint is None:
                piece_color_hint = color
            canvases.append({
                "pos": tl,
                "color": color,
                "colors": canvas_colors,
                "piece_color": piece_color_hint,
            })
    out["canvases"] = canvases

    # --- Pieces: sliding-window 5x5 detection ---
    # A piece is a 5x5 region whose CENTER pixel is a group color AND
    # the surrounding 24 pixels include at least 4 of {0 (active) or 3
    # (inactive)} ring pixels. We scan every (x, y) and keep the
    # canonical top-left positions.
    H, W = frame.shape
    seen_pieces: set[tuple[int, int]] = set()
    for tl_y in range(0, H - PIECE_H + 1):
        for tl_x in range(0, W - PIECE_W + 1):
            center_v = int(frame[tl_y + 2, tl_x + 2])
            if center_v not in _GROUP_COLORS:
                continue
            block = frame[tl_y:tl_y + PIECE_H, tl_x:tl_x + PIECE_W]
            zero_count = int((block == ACTIVE_RING_COLOR).sum())
            three_count = int((block == INACTIVE_RING_COLOR).sum())
            if zero_count + three_count < 4:
                continue
            # Skip if this overlaps any canvas (canvas centre is color 6).
            canvas_overlap = False
            for c in canvases:
                cx, cy = c["pos"]
                if abs(cx - tl_x) < PIECE_W and abs(cy - tl_y) < PIECE_H:
                    canvas_overlap = True
                    break
            # In held-out layouts a selected piece can partially overlap the
            # canvas at reset. Do not drop a strong active-piece signature just
            # because its 5x5 footprint intersects the canvas; the canvas
            # itself has centre colour 6 and is rejected above.
            if canvas_overlap and not (zero_count > three_count and zero_count >= 8):
                continue
            # Skip duplicates (overlapping detections of same piece).
            duplicate = any(abs(tl_x - p[0]) < PIECE_W and abs(tl_y - p[1]) < PIECE_H
                            for p in seen_pieces)
            if duplicate:
                continue
            seen_pieces.add((tl_x, tl_y))
            if zero_count > three_count:
                out["active_piece"] = (tl_x, tl_y)
                out["active_piece_color"] = center_v
            else:
                out["inactive_pieces"].append({"pos": (tl_x, tl_y), "color": center_v})

    # --- Targets: 7x7 regions with body color but no ring (0/3/6) nearby ---
    # Strategy: identify all body colors used by canvases. For each
    # canvas color, find 15-pixel cluster outside known piece/canvas
    # bboxes that fits in 7x7.
    forbidden_blocks: list[tuple[int, int, int, int]] = []
    if out["active_piece"] is not None:
        ap = out["active_piece"]
        forbidden_blocks.append((ap[0], ap[1], PIECE_W, PIECE_H))
    for c in canvases:
        forbidden_blocks.append((c["pos"][0], c["pos"][1], PIECE_W, PIECE_H))
    for p in out["inactive_pieces"]:
        forbidden_blocks.append((p["pos"][0], p["pos"][1], PIECE_W, PIECE_H))

    def in_forbidden(x: int, y: int) -> bool:
        for (fx, fy, fw, fh) in forbidden_blocks:
            if fx <= x < fx + fw and fy <= y < fy + fh:
                return True
        return False

    # Targets: 7x7 regions with body color but no ring (0/3/6) nearby.
    # For each color in {GROUP_COLORS}, find 7x7 cluster outside
    # forbidden bboxes. Each target may use multiple colors (e.g.,
    # flkdtg-blxuubrengnt has cols {8, 9}).
    raw_candidates: list[dict] = []
    for color in _GROUP_COLORS:
        ys, xs = np.where(frame == color)
        free_pts = [(int(x), int(y)) for x, y in zip(xs, ys)
                    if not in_forbidden(int(x), int(y))]
        if not free_pts:
            continue
        clusters = _cluster_bbox(free_pts, max_dist=6)
        for cluster in clusters:
            tl = _bbox_top_left(cluster, TARGET_W, TARGET_H)
            if tl is None:
                continue
            if len(cluster) < 3:
                continue
            tx, ty = tl
            region = frame[ty:ty + TARGET_H, tx:tx + TARGET_W]
            cols = {int(v) for v in np.unique(region)
                    if int(v) in _GROUP_COLORS}
            non_neg_count = int((region != -1).sum())
            raw_candidates.append({
                "pos": tl,
                "color": color,
                "colors": cols,
                "size": len(cluster),
                "density": non_neg_count,
                "points": [(int(x), int(y)) for x, y in cluster],
            })
    # Dedup by position proximity (within 5 cells in both axes). Keep
    # the candidate with the largest cluster size (most likely the
    # actual target sprite, not a spurious sub-region).
    raw_candidates.sort(key=lambda t: -t["size"])
    targets: list[dict] = []
    for c in raw_candidates:
        cx, cy = c["pos"]
        too_close = next(
            (t for t in targets
             if abs(t["pos"][0] - cx) < TARGET_W and abs(t["pos"][1] - cy) < TARGET_H),
            None,
        )
        if too_close is None:
            targets.append(c)
        else:
            # Merge colors into existing target.
            too_close["colors"] = too_close.get("colors", set()) | c["colors"]
            merged_points = list(too_close.get("points") or []) + list(c.get("points") or [])
            if merged_points:
                merged_tl = _bbox_top_left(merged_points, TARGET_W, TARGET_H)
                if merged_tl is not None:
                    too_close["pos"] = merged_tl
                too_close["points"] = merged_points
    # Strip internal "size"/"density" keys.
    for t in targets:
        t.pop("size", None)
        t.pop("density", None)
        t.pop("points", None)
    out["targets"] = targets

    # ===== L5/L6 whkxtx-absorber signature check =====
    # Whkxtx canvases have all-color-0 pattern (no centre dot 6).
    # L5/L6 layouts are fixed at known canonical positions; perception
    # verifies the frame signature so the adapter can select the matching
    # absorber constraint spec.
    L5_WHKXTX_POS = ((32, 32), (40, 50))
    L6_WHKXTX_POS = ((47, 48), (10, 14))

    def _is_whkxtx_canvas(tl: tuple[int, int]) -> bool:
        x, y = tl
        if not (0 <= x <= 60 and 0 <= y <= 60):
            return False
        block = frame[y:y + PIECE_H, x:x + PIECE_W]
        zeros = int((block == 0).sum())
        sixes = int((block == 6).sum())
        return zeros >= 13 and sixes == 0

    if all(_is_whkxtx_canvas(p) for p in L5_WHKXTX_POS):
        out["level_signature"] = "r11l_l5"
    elif all(_is_whkxtx_canvas(p) for p in L6_WHKXTX_POS):
        out["level_signature"] = "r11l_l6"

    return out


def perceive_r11l_l1(frame: np.ndarray) -> dict:
    """Legacy L1 single-group format. Maps to the L2+ perceive_r11l output.

    Returns dict with keys:
        "active_piece", "inactive_pieces" (list of (x, y)),
        "canvas_centre", "canvas_pos", "target_pos".
    """
    state = perceive_r11l(frame)
    out: dict = {
        "active_piece": state["active_piece"],
        "inactive_pieces": [p["pos"] for p in state["inactive_pieces"]],
        "canvas_centre": None,
        "canvas_pos": None,
        "target_pos": None,
    }
    if state["canvases"]:
        c = state["canvases"][0]
        out["canvas_pos"] = c["pos"]
        out["canvas_centre"] = (c["pos"][0] + 2, c["pos"][1] + 2)
    if state["targets"]:
        out["target_pos"] = state["targets"][0]["pos"]
    return out
