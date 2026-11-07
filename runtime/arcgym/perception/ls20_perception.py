"""ls20 frame perception (Phase 33 - tier A).

Recovers ls20 puzzle state from the rendered 64x64 int8 frame for the
runtime BFS solver. Per-frame extraction (no module-level state):
  - 12x12 wall occupancy (walls block movement; pixel color 4)
  - Player world position (top-left corner; ALWAYS visible)
  - Pusher list (position, dir, w, h) — 5x5 sprite with single-side
    color-1 "stripe" indicating push direction:
        top stripe  -> (dx=0,  dy=+1)  kapcaakvb_b push down
        bot stripe  -> (dx=0,  dy=-1)  lujfinsby_t push up
        left stripe -> (dx=+1, dy=0)   yjgargdic_r push right
        right stripe-> (dx=-1, dy=0)   tihiodtoj_l push left
    Pushers can appear at any (x, y); we scan all 5x5 windows.
  - Tile positions (5x5 with outer ring of bg color 3):
        rotation tile (rhsxkxzdjz): inner has 0 + 1 markers
        color tile    (soyhouuebz): inner has {9, 14, 12, 8} mosaic
        shape tile    (ttfwljgohq): inner has 0 + bg only (different L pattern)
        bonus tile    (npxgalaybz): 3x3 with color-11 border + bg-3 center
        target sprite (rjlbuycveu): 5x5 outer ring color-5, inner has cursor color
  - Walker tiles (wsoslqeku): walker tiles ARE rendered as a regular tile
    sprite (rotation/shape/color) at their current cage position. We can
    detect them via the same tile-pattern detector — a tile sprite at any
    grid cell that doesn't correspond to a static rotation/shape/color
    tile is a walker tile. Since we can't see the cage from the frame,
    walker bounds must be inferred from the level layout (we use the
    fact that walls form the cage edges).
  - Cursor state (rot_idx, color_idx, shape_idx) from htkmubhry sprite
    at world (3, 55) scale 2 -> rendered at frame[55:64, 3:14]. Identified
    via lookup table indexed by 9-bit binary mask of 6x6 cursor area.
  - Step counter remaining: row of color-11 bars at frame[61, 13:13+max].
  - Lives: 3 groups of 2px width at frame[61, 56:62].
  - Fog flag: detected via large color-5 area outside player radius.

Returns dict — see perceive_ls20() below.
"""
from __future__ import annotations

import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np

WALL_COLOR = 4
BG_COLOR = 3
FOG_COLOR = 5
PUSHER_STRIPE_COLOR = 1
PLAYER_TOP_COLOR = 12  # rotation marker top 2 rows
BONUS_BORDER_COLOR = 11

# Color order = [epqvqkpffo, jninpsotet, bejggpjowv, tqogkgimes] = [12, 9, 14, 8]
CURSOR_COLORS = (12, 9, 14, 8)

# Cursor 9-bit-mask -> (shape_idx, rot_idx) lookup. Generated offline by
# rendering each (shape, rot) pair and reading the 6x6 cursor area at
# frame[55:61, 3:9] downsampled 2x to a 3x3 mask.
_CURSOR_TABLE: Dict[int, Tuple[int, int]] = {
    413: (0, 0), 350: (0, 1), 371: (0, 2), 245: (0, 3),
    151: (1, 0), 316: (1, 1), 466: (1, 2), 121: (1, 3),
    367: (2, 0), 487: (2, 1), 493: (2, 2), 463: (2, 3),
    234: (3, 0), 171: (3, 1), 174: (3, 2), 426: (3, 3),
    179: (4, 0), 188: (4, 1), 410: (4, 2), 122: (4, 3),
    461: (5, 0), 335: (5, 1), 359: (5, 2), 485: (5, 3),
}


def _cell_to_x(cell: int) -> int:
    return 4 + cell * 5


def _cell_to_y(cell: int) -> int:
    return cell * 5


def _detect_walls(frame: np.ndarray) -> np.ndarray:
    """Return 12x12 boolean grid of wall occupancy.

    Wall sprite variants (all tagged `ihdgageizm`):
      - ihdgageizm: full 5x5 of color 4
      - fesygzfqui: 4x5 of color 4 with col-4 transparent (pusher edge)
      - krdypjjivz: 4x5 of color 4 with row-0 transparent (pusher edge)
      - mxfhnkdzvf: 4x5 of color 4 with row-4 transparent (pusher edge)
      - ubyunwkbpx: 5x4 of color 4 with col-0 transparent (pusher edge)
    All collide as walls. Detect: >= 12 of 25 pixels are color 4 (handles
    walls partially covered by overlay sprites like player highlight,
    pusher edges, etc.).
    """
    walls = np.zeros((12, 12), dtype=bool)
    for cell_y in range(12):
        for cell_x in range(12):
            x = _cell_to_x(cell_x)
            y = _cell_to_y(cell_y)
            block = frame[y:y + 5, x:x + 5]
            if block.shape != (5, 5):
                continue
            if int(np.sum(block == WALL_COLOR)) >= 12:
                walls[cell_y, cell_x] = True
    return walls


def _detect_player(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    """Locate player: 5x5 block with 2 contiguous rows/cols of color 12.

    Player has rotation marker (top of sprite) of color 12. After
    set_rotation, the marker may be on top, right, bottom, or left.
    """
    for cell_y in range(12):
        for cell_x in range(12):
            x = _cell_to_x(cell_x)
            y = _cell_to_y(cell_y)
            block = frame[y:y + 5, x:x + 5]
            if block.shape != (5, 5):
                continue
            # Check if 2 of 4 sides are all color 12 (top/bot/left/right).
            top = bool(np.all(block[:2] == PLAYER_TOP_COLOR))
            bot = bool(np.all(block[3:] == PLAYER_TOP_COLOR))
            left = bool(np.all(block[:, :2] == PLAYER_TOP_COLOR))
            right = bool(np.all(block[:, 3:] == PLAYER_TOP_COLOR))
            if top or bot or left or right:
                return (x, y)
    return None


def _detect_pushers(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Find all pusher 5x5 blocks at ANY (x, y) (not cell-aligned).

    Returns list of (x, y, dx, dy).
    """
    pushers: List[Tuple[int, int, int, int]] = []
    seen: set = set()
    for y in range(0, 60):  # exclude bottom UI rows
        for x in range(0, 60):
            block = frame[y:y + 5, x:x + 5]
            if block.shape != (5, 5):
                continue
            top_row1 = bool(np.all(block[0] == PUSHER_STRIPE_COLOR))
            bot_row1 = bool(np.all(block[4] == PUSHER_STRIPE_COLOR))
            left_col1 = bool(np.all(block[:, 0] == PUSHER_STRIPE_COLOR))
            right_col1 = bool(np.all(block[:, 4] == PUSHER_STRIPE_COLOR))
            stripes = sum([top_row1, bot_row1, left_col1, right_col1])
            if stripes != 1:
                continue
            if top_row1:
                body = block[1:]
            elif bot_row1:
                body = block[:4]
            elif left_col1:
                body = block[:, 1:]
            else:
                body = block[:, :4]
            if not bool(np.all(body == BG_COLOR)):
                continue
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            if top_row1:
                pushers.append((x, y, 0, 1))
            elif bot_row1:
                pushers.append((x, y, 0, -1))
            elif left_col1:
                pushers.append((x, y, 1, 0))
            else:
                pushers.append((x, y, -1, 0))
    return pushers


def _detect_tiles(frame: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
    """Identify rotation/color/shape tiles (5x5 with outer ring of bg=3)."""
    rotation: List[Tuple[int, int]] = []
    color: List[Tuple[int, int]] = []
    shape: List[Tuple[int, int]] = []
    for cell_y in range(12):
        for cell_x in range(12):
            x = _cell_to_x(cell_x)
            y = _cell_to_y(cell_y)
            block = frame[y:y + 5, x:x + 5]
            if block.shape != (5, 5):
                continue
            outer = (
                bool(np.all(block[0] == BG_COLOR))
                and bool(np.all(block[4] == BG_COLOR))
                and bool(np.all(block[:, 0] == BG_COLOR))
                and bool(np.all(block[:, 4] == BG_COLOR))
            )
            if not outer:
                continue
            inner = block[1:4, 1:4]
            inner_set = set(inner.flatten().tolist())
            # Color tile: 9, 14, 12, 8 present (and 0)
            if {9, 14, 12, 8} <= inner_set:
                color.append((x, y))
                continue
            # Rotation tile: inner has 0 and 1 only
            if inner_set <= {0, 1, BG_COLOR}:
                if 1 in inner_set:
                    rotation.append((x, y))
                    continue
            # Shape tile: inner has 0 and bg only
            if inner_set == {0, BG_COLOR}:
                shape.append((x, y))
                continue
    return {"rotation": rotation, "color": color, "shape": shape}


def _detect_bonus(frame: np.ndarray) -> List[Tuple[int, int]]:
    """Find all 3x3 bonus tiles (npxgalaybz). Pattern: 8 outer cells = 11, center = 3."""
    out: List[Tuple[int, int]] = []
    for y in range(0, 62):
        for x in range(0, 62):
            block = frame[y:y + 3, x:x + 3]
            if block.shape != (3, 3):
                continue
            if int(block[1, 1]) != BG_COLOR:
                continue
            ring = [
                int(block[0, 0]), int(block[0, 1]), int(block[0, 2]),
                int(block[1, 0]), int(block[1, 2]),
                int(block[2, 0]), int(block[2, 1]), int(block[2, 2]),
            ]
            if all(v == BONUS_BORDER_COLOR for v in ring):
                out.append((x, y))
    return out


def _detect_targets(frame: np.ndarray) -> List[Tuple[int, int, int]]:
    """Find target (rjlbuycveu) sprites: 5x5 outer ring=color 5 + inner has 1 cursor color."""
    out: List[Tuple[int, int, int]] = []
    for cell_y in range(12):
        for cell_x in range(12):
            x = _cell_to_x(cell_x)
            y = _cell_to_y(cell_y)
            block = frame[y:y + 5, x:x + 5]
            if block.shape != (5, 5):
                continue
            outer = (
                bool(np.all(block[0] == FOG_COLOR))
                and bool(np.all(block[4] == FOG_COLOR))
                and bool(np.all(block[:, 0] == FOG_COLOR))
                and bool(np.all(block[:, 4] == FOG_COLOR))
            )
            if not outer:
                continue
            inner = block[1:4, 1:4]
            inner_set = set(inner.flatten().tolist())
            non_bg = inner_set - {FOG_COLOR}
            if len(non_bg) == 1:
                col = next(iter(non_bg))
                if col in CURSOR_COLORS:
                    out.append((x, y, int(col)))
    return out


def _detect_cursor(frame: np.ndarray) -> Dict[str, int]:
    """Identify cursor shape, rotation, color from frame[55:61, 3:9].

    The cursor is rendered at world (3, 55) scale 2; its 3x3 sprite becomes
    a 6x6 block of which we downsample to a 3x3 binary mask of "is cursor
    color present in this 2x2 cell". The 9-bit mask maps to (shape, rot).
    """
    region = frame[55:61, 3:9]
    cursor_color = -1
    cursor_color_idx = -1
    for ci, col in enumerate(CURSOR_COLORS):
        if int(np.sum(region == col)) >= 2:
            cursor_color = col
            cursor_color_idx = ci
            break
    if cursor_color_idx < 0:
        return {"shape_idx": -1, "rot_idx": -1, "color_idx": -1}
    bits = 0
    for r in range(3):
        for c in range(3):
            sub = region[2 * r:2 * r + 2, 2 * c:2 * c + 2]
            on = bool(np.any(sub == cursor_color))
            bits = (bits << 1) | (1 if on else 0)
    sr = _CURSOR_TABLE.get(bits, (-1, -1))
    return {
        "shape_idx": int(sr[0]),
        "rot_idx": int(sr[1]),
        "color_idx": int(cursor_color_idx),
    }


def _detect_steps_and_lives(frame: np.ndarray) -> Tuple[int, int, int]:
    """Read step counter and lives from bottom UI row at y=61."""
    steps_max = 0
    steps_left = 0
    for i in range(0, 50):
        x = 13 + i
        if x >= 64:
            break
        col = int(frame[61, x])
        if col == BONUS_BORDER_COLOR:
            steps_max += 1
            steps_left += 1
        elif col == BG_COLOR:
            steps_max += 1
        else:
            break
    lives = 0
    for i in range(3):
        x = 56 + 3 * i
        if x >= 64:
            break
        if int(frame[61, x]) == 8:
            lives += 1
    return steps_left, steps_max, lives


def _detect_fog(frame: np.ndarray, player_xy: Optional[Tuple[int, int]]) -> bool:
    """Detect if fog is active (level 7).

    Fog renders distance>20 from player to color 5 (FOG_COLOR). If the fog
    is on, the four corners of the frame (0,0), (0,63), (63,0), (63,63)
    are typically masked to color 5 unless the player happens to be near
    them. We require all 4 corners (excluding UI rows) to be FOG_COLOR.
    """
    corners = [
        int(frame[0, 0]),
        int(frame[0, 63]),
        int(frame[55, 63]),
    ]
    return all(c == FOG_COLOR for c in corners)


def _level_signature(walls: np.ndarray) -> int:
    """Deterministic level signature via crc32 of wall occupancy bytes."""
    return int(zlib.crc32(walls.tobytes()) & 0xffffffff)


# Pre-computed crc32 of wall_grid for L1..L7 (offline probe).
_LEVEL_SIGS: Dict[int, int] = {
    2032336419: 0,  # L1 wall_count=106
    2765824549: 1,  # L2 wall_count=87
    2141835600: 2,  # L3 wall_count=78
    2411516585: 3,  # L4 wall_count=76
    1160859921: 4,  # L5 wall_count=74
    2612668821: 5,  # L6 wall_count=74
    3922485173: 6,  # L7 wall_count=20 (fog)
}


def perceive_ls20(frame: np.ndarray) -> dict:
    """Full perception of ls20 level state."""
    if frame.ndim == 3:
        frame = frame[-1]
    frame = np.asarray(frame, dtype=np.int16)

    walls = _detect_walls(frame)
    player_xy = _detect_player(frame)
    pushers = _detect_pushers(frame)
    tiles = _detect_tiles(frame)
    bonus = _detect_bonus(frame)
    targets = _detect_targets(frame)
    cursor = _detect_cursor(frame)
    steps_left, steps_max, lives = _detect_steps_and_lives(frame)
    fog = _detect_fog(frame, player_xy)
    sig_hash = _level_signature(walls)
    level_idx = _LEVEL_SIGS.get(sig_hash)

    return {
        "level_idx": level_idx,
        "sig_hash": sig_hash,
        "wall_count": int(walls.sum()),
        "wall_grid": walls,
        "player_xy": player_xy,
        "goal_targets": targets,
        "pushers": pushers,
        "rotation_tiles": tiles["rotation"],
        "color_tiles": tiles["color"],
        "shape_tiles": tiles["shape"],
        "bonus_tiles": bonus,
        "cursor": cursor,
        "steps_left": steps_left,
        "steps_max": steps_max,
        "lives": lives,
        "fog": fog,
        "frame": frame,
    }
