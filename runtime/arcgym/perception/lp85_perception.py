"""lp85 perception — frame-only level + goal extraction (tier-A support).

lp85 is a button-driven cycle-rotation puzzle. Each level has buttons
which, when ACTION6-clicked, rotate the sprites lying on a fixed cycle
(or set of cycles, when buttons stack at the same click rect). The
goal is to land each `goal` (colour 11) or `goal-o` (colour 12) token
on a pre-positioned target sprite (`bghvgbtwcb` / `fdgmtkfrxl`).

Frame extracts:
- ``level_signature`` — derived from per-level (count_color_3,
  count_color_4) signature (stable across step-counter empty/full).
- ``level_idx`` — 0..7 corresponding to L1..L8.
- ``goals`` — list of ``(grid_x, grid_y, kind)`` where kind is ``"goal"``
  (color 11 solid 2×2 block) or ``"goal-o"`` (color 12 solid 2×2 block).
  Detection skips ``bghvgbtwcb``/``fdgmtkfrxl`` *target* sprites whose
  pixels are colour 11/12 only at the four corners (4×4 frame), so the
  hollow-corner pattern is filtered out.

The goal positions report the live cycle state of each token, so the
BFS solver in ``arcgym/adapters/lp85_perception_driven.py`` can run
afresh on each frame call — no offline-derived plan dict required.

NOTE: counts of colour-14 / colour-5 fluctuate within a level as the
StepCounter empties (cqizedfwrl=14 -> sayxoyjqpt=5), but counts of 3
and 4 stay stable (grid-fixed letterbox + background). This means the
level signature is invariant during play.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


# Per-level (count_color_3, count_color_4) signatures captured 2026-05-04
# at level-entry from offline run. Both counts derive from fixed-grid
# letterbox + background-tile geometry.
_LEVEL_SIGNATURES: dict[tuple[int, int], str] = {
    (1606, 1978): "lp85_l1",
    (2319, 1445): "lp85_l2",
    (2791, 1041): "lp85_l3",
    (751, 2937): "lp85_l4",
    (576, 2896): "lp85_l5",
    (192, 3426): "lp85_l6",
    (2272, 1624): "lp85_l7",
    (126, 3506): "lp85_l8",
}

# Per-level display scale + letterbox padding (from camera).  Used to
# convert detected display-pixel positions back to grid coords.
# Indexed by level_idx 0..7.
_LEVEL_GEOM: dict[int, dict[str, int]] = {
    0: {"scale": 2, "x_pad": 0, "y_pad": 13},   # L1 kdrsqrvpwb 32x19
    1: {"scale": 1, "x_pad": 11, "y_pad": 11},  # L2 cecdsipmha 41x41
    2: {"scale": 1, "x_pad": 12, "y_pad": 16},  # L3 vvnbesozdj 39x31
    3: {"scale": 1, "x_pad": 3, "y_pad": 3},    # L4 ihpokmcrjm 57x57
    4: {"scale": 2, "x_pad": 5, "y_pad": 0},    # L5 qibynbipmw 27x32
    5: {"scale": 1, "x_pad": 2, "y_pad": 0},    # L6 ksgcbuhgwz 60x64
    6: {"scale": 1, "x_pad": 8, "y_pad": 14},   # L7 yzzznxxvju 48x36
    7: {"scale": 1, "x_pad": 0, "y_pad": 0},    # L8 hmzsasouat 63x63
}

_SIG_TO_IDX: dict[str, int] = {f"lp85_l{i+1}": i for i in range(8)}


def _normalize_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    return arr


def _detect_solid_blocks(arr: np.ndarray, color: int, block: int = 2) -> list[tuple[int, int]]:
    """Detect every ``block``x``block`` solid square of ``color`` in arr.

    Returns top-left (display_x, display_y) of each block. Filters to
    avoid hollow-corner *target* sprites (4×4 frames with color only at
    corners) — for those we'd see isolated 1×1 corners, not 2×2 blocks.
    Using a 2×2 ALL-of-color test eliminates target-corner false hits
    because target-corners are 1×1 (non-2×2). Also dedupes by stride so
    the same 2×2 block isn't reported twice.
    """
    H, W = arr.shape
    mask = (arr == color)
    found: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for y in range(H - block + 1):
        for x in range(W - block + 1):
            if not mask[y, x]:
                continue
            if mask[y:y + block, x:x + block].all():
                # Snap to top-left of the contiguous run to dedupe.
                # Scan up/left boundary.
                tl_x, tl_y = x, y
                while tl_x > 0 and mask[tl_y, tl_x - 1]:
                    tl_x -= 1
                while tl_y > 0 and mask[tl_y - 1, tl_x]:
                    tl_y -= 1
                if (tl_x, tl_y) in seen:
                    continue
                seen.add((tl_x, tl_y))
                found.append((tl_x, tl_y))
    return found


def _disp_to_grid(disp_x: int, disp_y: int, scale: int, x_pad: int, y_pad: int) -> tuple[int, int]:
    return (disp_x - x_pad) // scale, (disp_y - y_pad) // scale


def perceive_lp85(frame: Any) -> dict:
    arr = _normalize_frame(frame)
    flat = arr.flatten()
    count_3 = int(np.sum(flat == 3))
    count_4 = int(np.sum(flat == 4))
    sig = _LEVEL_SIGNATURES.get((count_3, count_4))
    level_idx: Optional[int] = _SIG_TO_IDX.get(sig) if sig else None

    goals: list[tuple[int, int, str]] = []
    if level_idx is not None:
        geom = _LEVEL_GEOM[level_idx]
        scale = geom["scale"]
        x_pad = geom["x_pad"]
        y_pad = geom["y_pad"]
        # goal sprite = solid 2×2 of colour 11; goal-o = solid 2×2 of colour 12.
        # Sprite size 2 in grid coords -> 2*scale display pixels.
        block = 2 * scale
        # We expect every goal to occupy exactly a (2*scale)x(2*scale) block.
        # But target frames have 1*scale-wide corners, NOT 2*scale solids,
        # so they're filtered automatically.
        for color, kind in [(11, "goal"), (12, "goal-o")]:
            for tl_x, tl_y in _detect_solid_blocks(arr, color, block=block):
                gx, gy = _disp_to_grid(tl_x, tl_y, scale, x_pad, y_pad)
                goals.append((gx, gy, kind))

    return {
        "level_signature": sig,
        "level_idx": level_idx,
        "count_3": count_3,
        "count_4": count_4,
        "goals": goals,
    }
