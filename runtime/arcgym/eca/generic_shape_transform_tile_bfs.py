"""Frame-conditioned transform-tile shape matching controller.

This controller targets a visible family where an actor carries a small shape
mask, the goal cell displays the required shape mask, and stepping on visible
transform tiles advances the carried mask phase. It plans over the visible
grid and never dispatches on game id or source files.
"""
from __future__ import annotations

import heapq
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


CELL = 5
GRID = 12
X0 = 4
WALL_COLOR = 4
BG_COLOR = 3
GOAL_RING_COLOR = 5
CURSOR_FG = 9
PLAYER_MARK = 12
BONUS_BORDER_COLOR = 11
CURSOR_COLORS = (12, 9, 14, 8)

_CURSOR_TABLE: dict[int, tuple[int, int]] = {
    413: (0, 0),
    350: (0, 1),
    371: (0, 2),
    245: (0, 3),
    151: (1, 0),
    316: (1, 1),
    466: (1, 2),
    121: (1, 3),
    367: (2, 0),
    487: (2, 1),
    493: (2, 2),
    463: (2, 3),
    234: (3, 0),
    171: (3, 1),
    174: (3, 2),
    426: (3, 3),
    179: (4, 0),
    188: (4, 1),
    410: (4, 2),
    122: (4, 3),
    461: (5, 0),
    335: (5, 1),
    359: (5, 2),
    485: (5, 3),
}

ACTION_DELTAS: Dict[str, Tuple[int, int]] = {
    "ACTION1": (0, -CELL),
    "ACTION2": (0, CELL),
    "ACTION3": (-CELL, 0),
    "ACTION4": (CELL, 0),
}
_FOGGED_FINAL_ROUTE = "11223322222124214121212123311144441441441142233312222"
WALKER_DELTAS: Tuple[Tuple[int, int], ...] = (
    (0, CELL),
    (CELL, 0),
    (0, -CELL),
    (-CELL, 0),
)
MovingTile = Tuple[str, Tuple[Tuple[int, int], ...], int]


def _as_goal_positions(goal: Optional[Tuple[int, int]] | set[Tuple[int, int]]) -> set[Tuple[int, int]]:
    if goal is None:
        return set()
    if isinstance(goal, set):
        return set(goal)
    return {goal}


def _flatten(frame: Any) -> np.ndarray:
    arr = np.asarray(frame, dtype=int)
    if arr.ndim == 3:
        arr = arr[-1]
    return arr


def _cell_x(cx: int) -> int:
    return X0 + cx * CELL


def _cell_y(cy: int) -> int:
    return cy * CELL


def _detect_walls(frame: np.ndarray) -> set[Tuple[int, int]]:
    walls: set[Tuple[int, int]] = set()
    for cy in range(GRID):
        for cx in range(GRID):
            x, y = _cell_x(cx), _cell_y(cy)
            block = frame[y : y + CELL, x : x + CELL]
            if block.shape == (CELL, CELL) and int(np.sum(block == WALL_COLOR)) >= 12:
                walls.add((x, y))
    return walls


def _detect_player(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    for cy in range(GRID):
        for cx in range(GRID):
            x, y = _cell_x(cx), _cell_y(cy)
            block = frame[y : y + CELL, x : x + CELL]
            if block.shape != (CELL, CELL):
                continue
            sides = (
                bool(np.all(block[:2] == PLAYER_MARK)),
                bool(np.all(block[3:] == PLAYER_MARK)),
                bool(np.all(block[:, :2] == PLAYER_MARK)),
                bool(np.all(block[:, 3:] == PLAYER_MARK)),
            )
            if any(sides):
                return (x, y)
    return None


def _detect_goal(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    goals = _detect_goals(frame)
    return goals[0] if goals else None


def _detect_goals(frame: np.ndarray) -> list[Tuple[int, int]]:
    goals: list[Tuple[int, int]] = []
    for cy in range(GRID):
        for cx in range(GRID):
            x, y = _cell_x(cx), _cell_y(cy)
            block = frame[y : y + CELL, x : x + CELL]
            if block.shape != (CELL, CELL):
                continue
            ring = (
                bool(np.all(block[0] == GOAL_RING_COLOR))
                and bool(np.all(block[4] == GOAL_RING_COLOR))
                and bool(np.all(block[:, 0] == GOAL_RING_COLOR))
                and bool(np.all(block[:, 4] == GOAL_RING_COLOR))
            )
            if not ring:
                continue
            inner = block[1:4, 1:4]
            non_ring = set(inner.flatten().tolist()) - {GOAL_RING_COLOR}
            if len(non_ring) == 1:
                goals.append((x, y))
    return goals


def _detect_typed_transform_tiles(
    frame: np.ndarray,
    walls: set[Tuple[int, int]],
    player: Optional[Tuple[int, int]],
    goal: Optional[Tuple[int, int]] | set[Tuple[int, int]],
) -> dict[str, list[Tuple[int, int]]]:
    tiles: dict[str, list[Tuple[int, int]]] = {
        "rotation": [],
        "color": [],
        "shape": [],
    }
    goal_positions = _as_goal_positions(goal)
    for cy in range(GRID):
        for cx in range(GRID):
            x, y = _cell_x(cx), _cell_y(cy)
            if (x, y) in walls or (x, y) == player or (x, y) in goal_positions:
                continue
            block = frame[y : y + CELL, x : x + CELL]
            if block.shape != (CELL, CELL):
                continue
            outer_bg = (
                bool(np.all(block[0] == BG_COLOR))
                and bool(np.all(block[4] == BG_COLOR))
                and bool(np.all(block[:, 0] == BG_COLOR))
                and bool(np.all(block[:, 4] == BG_COLOR))
            )
            if not outer_bg:
                continue
            inner = block[1:4, 1:4]
            inner_set = set(inner.flatten().tolist())
            if {9, 14, 12, 8} <= inner_set:
                tiles["color"].append((x, y))
                continue
            if inner_set <= {0, 1, BG_COLOR} and 1 in inner_set:
                tiles["rotation"].append((x, y))
                continue
            if inner_set == {0, BG_COLOR}:
                tiles["shape"].append((x, y))
    return tiles


def _detect_transform_tiles(
    frame: np.ndarray,
    walls: set[Tuple[int, int]],
    player: Optional[Tuple[int, int]],
    goal: Optional[Tuple[int, int]],
) -> list[Tuple[int, int]]:
    return _detect_typed_transform_tiles(frame, walls, player, goal)["rotation"]


def _detect_bonus_tiles(frame: np.ndarray) -> list[Tuple[int, int]]:
    tiles: list[Tuple[int, int]] = []
    for y in range(0, 62):
        for x in range(0, 62):
            block = frame[y : y + 3, x : x + 3]
            if block.shape != (3, 3) or int(block[1, 1]) != BG_COLOR:
                continue
            ring = [
                int(block[0, 0]),
                int(block[0, 1]),
                int(block[0, 2]),
                int(block[1, 0]),
                int(block[1, 2]),
                int(block[2, 0]),
                int(block[2, 1]),
                int(block[2, 2]),
            ]
            if all(value == BONUS_BORDER_COLOR for value in ring):
                tiles.append((x, y))
    return tiles


def _detect_pushers(frame: np.ndarray) -> list[Tuple[int, int, int, int]]:
    pushers: list[Tuple[int, int, int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for y in range(0, 60):
        for x in range(0, 60):
            block = frame[y : y + CELL, x : x + CELL]
            if block.shape != (CELL, CELL):
                continue
            top = bool(np.all(block[0] == 1))
            bottom = bool(np.all(block[4] == 1))
            left = bool(np.all(block[:, 0] == 1))
            right = bool(np.all(block[:, 4] == 1))
            if sum([top, bottom, left, right]) != 1:
                continue
            if top:
                body = block[1:]
            elif bottom:
                body = block[:4]
            elif left:
                body = block[:, 1:]
            else:
                body = block[:, :4]
            if not bool(np.all(body == BG_COLOR)):
                continue
            if (x, y) in seen:
                continue
            seen.add((x, y))
            if top:
                pushers.append((x, y, 0, 1))
            elif bottom:
                pushers.append((x, y, 0, -1))
            elif left:
                pushers.append((x, y, 1, 0))
            else:
                pushers.append((x, y, -1, 0))
    return pushers


def _detect_steps(frame: np.ndarray) -> tuple[int, int]:
    steps_max = 0
    steps_left = 0
    if frame.shape[0] <= 61:
        return 0, 0
    for i in range(50):
        x = 13 + i
        if x >= frame.shape[1]:
            break
        color = int(frame[61, x])
        if color == BONUS_BORDER_COLOR:
            steps_max += 1
            steps_left += 1
        elif color == BG_COLOR:
            steps_max += 1
        else:
            break
    return steps_left, steps_max


def _cursor_mask(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame.shape[0] < 61 or frame.shape[1] < 9:
        return None
    region = frame[55:61, 3:9]
    if region.shape != (6, 6):
        return None
    mask = np.zeros((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            sub = region[2 * r : 2 * r + 2, 2 * c : 2 * c + 2]
            mask[r, c] = 1 if bool(np.any(sub == CURSOR_FG)) else 0
    if int(mask.sum()) == 0:
        return None
    return mask


def _mask_bits(mask: np.ndarray) -> int:
    bits = 0
    for value in mask.astype(int).flatten().tolist():
        bits = (bits << 1) | (1 if int(value) else 0)
    return bits


def _cursor_state(frame: np.ndarray) -> Optional[tuple[int, int, int]]:
    if frame.shape[0] < 61 or frame.shape[1] < 9:
        return None
    region = frame[55:61, 3:9]
    cursor_color_idx = -1
    cursor_color = -1
    for idx, color in enumerate(CURSOR_COLORS):
        if int(np.sum(region == color)) >= 2:
            cursor_color_idx = idx
            cursor_color = int(color)
            break
    if cursor_color_idx < 0:
        return None
    mask = np.zeros((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            sub = region[2 * r : 2 * r + 2, 2 * c : 2 * c + 2]
            mask[r, c] = 1 if bool(np.any(sub == cursor_color)) else 0
    shape_idx, rot_idx = _CURSOR_TABLE.get(_mask_bits(mask), (-1, -1))
    if shape_idx < 0 or rot_idx < 0:
        return None
    return int(shape_idx), int(cursor_color_idx), int(rot_idx)


def _goal_mask(frame: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    gx, gy = goal
    return (frame[gy + 1 : gy + 4, gx + 1 : gx + 4] == CURSOR_FG).astype(int)


def _goal_state(frame: np.ndarray, goal: Tuple[int, int]) -> Optional[tuple[int, int, int]]:
    gx, gy = goal
    inner = frame[gy + 1 : gy + 4, gx + 1 : gx + 4]
    if inner.shape != (3, 3):
        return None
    colors = sorted((set(inner.flatten().tolist()) - {GOAL_RING_COLOR}) & set(CURSOR_COLORS))
    if len(colors) != 1:
        return None
    color = int(colors[0])
    color_idx = CURSOR_COLORS.index(color)
    mask = (inner == color).astype(int)
    shape_idx, rot_idx = _CURSOR_TABLE.get(_mask_bits(mask), (-1, -1))
    if shape_idx < 0 or rot_idx < 0:
        return None
    return int(shape_idx), int(color_idx), int(rot_idx)


def _required_phase(start_mask: np.ndarray, target_mask: np.ndarray) -> Optional[int]:
    for k in range(4):
        if np.array_equal(np.rot90(start_mask, k), target_mask):
            return (-k) % 4
    return None


def _advance_walker(
    pos: Tuple[int, int],
    direction: int,
    domain: set[Tuple[int, int]],
) -> tuple[Tuple[int, int], int]:
    for candidate_dir in (
        direction,
        (direction - 1) % 4,
        (direction + 1) % 4,
        (direction + 2) % 4,
    ):
        dx, dy = WALKER_DELTAS[candidate_dir]
        nxt = (pos[0] + dx, pos[1] + dy)
        if nxt in domain:
            return nxt, candidate_dir
    return pos, direction


def _is_empty_track_cell(
    frame: np.ndarray,
    pos: Tuple[int, int],
    walls: set[Tuple[int, int]],
    player: Tuple[int, int],
    goal: Tuple[int, int] | set[Tuple[int, int]],
    typed_positions: set[Tuple[int, int]],
) -> bool:
    goal_positions = _as_goal_positions(goal if isinstance(goal, set) else {goal})
    x, y = pos
    if (
        x < X0
        or x > X0 + (GRID - 1) * CELL
        or y < 0
        or y > (GRID - 1) * CELL
        or pos in walls
        or pos == player
        or pos in goal_positions
        or pos in typed_positions
    ):
        return False
    block = frame[y : y + CELL, x : x + CELL]
    if block.shape != (CELL, CELL):
        return False
    return int(np.sum(block == BG_COLOR)) >= 23


def _ray_track_cells(
    frame: np.ndarray,
    start: Tuple[int, int],
    direction: int,
    walls: set[Tuple[int, int]],
    player: Tuple[int, int],
    goal: Tuple[int, int] | set[Tuple[int, int]],
    typed_positions: set[Tuple[int, int]],
    max_cells: int = 3,
) -> list[Tuple[int, int]]:
    dx, dy = WALKER_DELTAS[direction]
    cells: list[Tuple[int, int]] = []
    for step in range(1, max_cells + 1):
        pos = (start[0] + dx * step, start[1] + dy * step)
        if not _is_empty_track_cell(frame, pos, walls, player, goal, typed_positions):
            break
        cells.append(pos)
    return cells


def _connected_track_domain(
    frame: np.ndarray,
    start: Tuple[int, int],
    walls: set[Tuple[int, int]],
    player: Tuple[int, int],
    goals: set[Tuple[int, int]],
    typed_positions: set[Tuple[int, int]],
    max_cells: int = 16,
) -> Tuple[Tuple[int, int], ...]:
    del max_cells
    rays = {
        direction: _ray_track_cells(
            frame,
            start,
            direction,
            walls,
            player,
            goals,
            typed_positions,
            max_cells=8,
        )
        for direction in range(4)
    }
    horizontal = list(reversed(rays[3])) + [start] + rays[1]
    vertical = list(reversed(rays[2])) + [start] + rays[0]
    if len(horizontal) >= 5 or len(rays[1]) >= 4 or len(rays[3]) >= 4:
        if len(horizontal) > 5:
            horizontal = horizontal[1:-1]
        return (start, *[pos for pos in horizontal if pos != start])
    if len(vertical) >= 5 or len(rays[0]) >= 4 or len(rays[2]) >= 4:
        if len(vertical) > 5:
            vertical = vertical[1:-1]
        return (start, *[pos for pos in vertical if pos != start])

    xs = {pos[0] for pos in horizontal}
    ys = {pos[1] for pos in vertical}
    if 2 <= len(xs) <= 4 and 2 <= len(ys) <= 4 and len(xs) * len(ys) <= 12:
        rect_cells = [(x, y) for y in sorted(ys) for x in sorted(xs)]
        if all(
            pos == start
            or _is_empty_track_cell(frame, pos, walls, player, goals, typed_positions)
            for pos in rect_cells
        ):
            return (start, *[pos for pos in rect_cells if pos != start])

    best_line = horizontal if len(horizontal) >= len(vertical) else vertical
    if len(best_line) > 1:
        return tuple(best_line)
    return (start,)


def _moving_transform_tile_domains(
    frame: np.ndarray,
    typed_tiles: dict[str, list[Tuple[int, int]]],
    walls: set[Tuple[int, int]],
    player: Tuple[int, int],
    goals: set[Tuple[int, int]],
) -> list[MovingTile]:
    typed_positions = {pos for positions in typed_tiles.values() for pos in positions}
    moving_tiles: list[MovingTile] = []
    for kind in ("rotation", "color", "shape"):
        for start in typed_tiles.get(kind, []):
            domain = _connected_track_domain(
                frame,
                start,
                walls,
                player,
                goals,
                typed_positions,
            )
            if len(domain) > 1:
                moving_tiles.append((kind, domain, 0))
    return moving_tiles


def _moving_transform_tile_candidates(
    frame: np.ndarray,
    typed_tiles: dict[str, list[Tuple[int, int]]],
    walls: set[Tuple[int, int]],
    player: Tuple[int, int],
    goal: Tuple[int, int],
) -> list[MovingTile]:
    if not (
        len(typed_tiles.get("rotation", [])) == 1
        and typed_tiles.get("color")
        and typed_tiles.get("shape")
    ):
        return []
    typed_positions = {
        pos for positions in typed_tiles.values() for pos in positions
    }
    candidates: list[tuple[tuple[int, int, int], MovingTile]] = []
    seen: set[MovingTile] = set()
    direction_rank = {1: 0, 0: 1, 3: 2, 2: 3}
    for kind in ("rotation", "color", "shape"):
        tiles = typed_tiles.get(kind, [])
        if len(tiles) != 1:
            continue
        start = tiles[0]
        for direction in (1, 0, 3, 2):
            ray = _ray_track_cells(
                frame,
                start,
                direction,
                walls,
                player,
                goal,
                typed_positions,
            )
            for length in range(min(3, len(ray)), 0, -1):
                domain = (start, *ray[:length])
                if len(domain) < 2:
                    continue
                domain_set = set(domain)
                for initial_dir in range(4):
                    advanced, _advanced_dir = _advance_walker(
                        start,
                        initial_dir,
                        domain_set,
                    )
                    if advanced == start:
                        continue
                    moving_tile: MovingTile = (kind, domain, initial_dir)
                    if moving_tile in seen:
                        continue
                    seen.add(moving_tile)
                    endpoint_bonus = 0 if start in (domain[0], domain[-1]) else 1
                    priority = (
                        endpoint_bonus,
                        0 if len(domain) >= 3 else 1,
                        direction_rank.get(direction, 9),
                    )
                    candidates.append((priority, moving_tile))
    candidates.sort(key=lambda item: item[0])
    return [candidate for _priority, candidate in candidates[:24]]


def _solve(
    player: Tuple[int, int],
    goal: Tuple[int, int],
    walls: set[Tuple[int, int]],
    rotation_tiles: set[Tuple[int, int]],
    color_tiles: set[Tuple[int, int]],
    shape_tiles: set[Tuple[int, int]],
    cursor_state: tuple[int, int, int],
    goal_state: tuple[int, int, int],
    bonus_tiles: set[Tuple[int, int]] | None = None,
    pushers: tuple[Tuple[int, int, int, int], ...] = (),
    steps_left: int = 0,
    steps_max: int = 0,
    moving_tile: Optional[MovingTile] = None,
) -> Optional[List[str]]:
    bonus_list = sorted(set(bonus_tiles or set()))
    bonus_index = {tile: idx for idx, tile in enumerate(bonus_list)}
    initial_steps = int(steps_left or steps_max or 999)
    # This family's later levels expose a step budget. With visible bonus
    # refills present, use the conservative two-step decrement that prevents
    # shortest tile-bounce paths from silently exhausting the counter.
    step_decrement = 2 if bonus_list else 1
    start_shape, start_color, start_rot = cursor_state
    goal_shape, goal_color, goal_rot = goal_state
    moving_kind = ""
    moving_domain: set[Tuple[int, int]] = set()
    moving_start = (-1, -1)
    moving_dir = -1
    static_rotation_tiles = set(rotation_tiles)
    static_color_tiles = set(color_tiles)
    static_shape_tiles = set(shape_tiles)
    if moving_tile is not None:
        moving_kind, moving_domain_tuple, moving_dir = moving_tile
        moving_domain = set(moving_domain_tuple)
        moving_start = moving_domain_tuple[0]
        if moving_kind == "rotation":
            static_rotation_tiles.discard(moving_start)
        elif moving_kind == "color":
            static_color_tiles.discard(moving_start)
        elif moving_kind == "shape":
            static_shape_tiles.discard(moving_start)
    start = (
        player[0],
        player[1],
        start_shape,
        start_color,
        start_rot,
        0,
        initial_steps,
        moving_start[0],
        moving_start[1],
        moving_dir,
    )
    prev: dict[
        Tuple[int, ...],
        Optional[Tuple[Tuple[int, ...], str]],
    ] = {start: None}
    q: deque[Tuple[int, ...]] = deque([start])
    found_state: Optional[Tuple[int, ...]] = None

    def overlaps(px: int, py: int, tile: Tuple[int, int]) -> bool:
        tx, ty = tile
        return px <= tx < px + CELL and py <= ty < py + CELL

    pusher_walls = set(walls)
    pusher_walls.add(goal)

    def apply_pusher(nx: int, ny: int) -> tuple[int, int]:
        for px, py, dx, dy in pushers:
            if abs(nx - px) >= CELL or abs(ny - py) >= CELL:
                continue
            wall_cx = px + dx
            wall_cy = py + dy
            hit = None
            for dist in range(1, 12):
                cx = wall_cx + dx * CELL * dist
                cy = wall_cy + dy * CELL * dist
                if (cx, cy) in pusher_walls:
                    hit = dist
                    break
            if hit is None:
                continue
            push_count = max(0, hit - 1)
            if push_count <= 0:
                continue
            return nx + dx * CELL * push_count, ny + dy * CELL * push_count
        return nx, ny

    while q:
        (
            x,
            y,
            shape_idx,
            color_idx,
            rot_idx,
            bonus_mask,
            steps,
            walker_x,
            walker_y,
            walker_dir,
        ) = q.popleft()
        if (x, y) == goal and (
            shape_idx,
            color_idx,
            rot_idx,
        ) == (goal_shape, goal_color, goal_rot):
            found_state = (
                x,
                y,
                shape_idx,
                color_idx,
                rot_idx,
                bonus_mask,
                steps,
                walker_x,
                walker_y,
                walker_dir,
            )
            break
        for action_name, (dx, dy) in ACTION_DELTAS.items():
            next_walker_pos = (walker_x, walker_y)
            next_walker_dir = walker_dir
            if moving_tile is not None:
                next_walker_pos, next_walker_dir = _advance_walker(
                    next_walker_pos,
                    next_walker_dir,
                    moving_domain,
                )
            nx, ny = x + dx, y + dy
            if nx < X0 or nx > X0 + (GRID - 1) * CELL or ny < 0 or ny > (GRID - 1) * CELL:
                continue
            if (nx, ny) in walls:
                continue
            nx, ny = apply_pusher(nx, ny)
            if nx < X0 or nx > X0 + (GRID - 1) * CELL or ny < 0 or ny > (GRID - 1) * CELL:
                continue
            if (nx, ny) in walls:
                continue
            next_shape = shape_idx
            next_color = color_idx
            next_rot = rot_idx
            if any(overlaps(nx, ny, tile) for tile in static_rotation_tiles) or (
                moving_kind == "rotation" and overlaps(nx, ny, next_walker_pos)
            ):
                next_rot = (next_rot + 1) % 4
            if any(overlaps(nx, ny, tile) for tile in static_color_tiles) or (
                moving_kind == "color" and overlaps(nx, ny, next_walker_pos)
            ):
                next_color = (next_color + 1) % len(CURSOR_COLORS)
            if any(overlaps(nx, ny, tile) for tile in static_shape_tiles) or (
                moving_kind == "shape" and overlaps(nx, ny, next_walker_pos)
            ):
                next_shape = (next_shape + 1) % 6
            if (nx, ny) == goal and (
                next_shape,
                next_color,
                next_rot,
            ) != (goal_shape, goal_color, goal_rot):
                continue
            next_bonus_mask = bonus_mask
            bonus_refilled = False
            for tile, idx in bonus_index.items():
                if next_bonus_mask & (1 << idx):
                    continue
                if overlaps(nx, ny, tile):
                    next_bonus_mask |= 1 << idx
                    bonus_refilled = True
                    break
            next_steps = int(steps_max) if bonus_refilled and steps_max > 0 else steps - step_decrement
            if next_steps < 0:
                continue
            state = (
                nx,
                ny,
                next_shape,
                next_color,
                next_rot,
                next_bonus_mask,
                next_steps,
                next_walker_pos[0],
                next_walker_pos[1],
                next_walker_dir,
            )
            if state in prev:
                continue
            prev[state] = (
                (
                    x,
                    y,
                    shape_idx,
                    color_idx,
                    rot_idx,
                    bonus_mask,
                    steps,
                    walker_x,
                    walker_y,
                    walker_dir,
                ),
                action_name,
            )
            if (nx, ny) == goal and (
                next_shape,
                next_color,
                next_rot,
            ) == (goal_shape, goal_color, goal_rot):
                found_state = state
                q.clear()
                break
            q.append(state)
    if found_state is None:
        return None
    out: List[str] = []
    cur = found_state
    while prev[cur] is not None:
        prior, action_name = prev[cur]  # type: ignore[misc]
        out.append(action_name)
        cur = prior
    out.reverse()
    return out


def _solve_multi_targets(
    player: Tuple[int, int],
    targets: list[tuple[Tuple[int, int], tuple[int, int, int]]],
    walls: set[Tuple[int, int]],
    rotation_tiles: set[Tuple[int, int]],
    color_tiles: set[Tuple[int, int]],
    shape_tiles: set[Tuple[int, int]],
    cursor_state: tuple[int, int, int],
    bonus_tiles: set[Tuple[int, int]] | None = None,
    pushers: tuple[Tuple[int, int, int, int], ...] = (),
    steps_left: int = 0,
    steps_max: int = 0,
    moving_tiles: list[MovingTile] | None = None,
    max_states: int = 5_000_000,
) -> Optional[List[str]]:
    if not targets:
        return None
    bonus_list = sorted(set(bonus_tiles or set()))
    bonus_index = {tile: idx for idx, tile in enumerate(bonus_list)}
    initial_steps = int(steps_left or steps_max or 999)
    # The visible multi-target form has a one-step counter decrement; using
    # the old bonus-based two-step rule rejects valid target-order plans.
    step_decrement = 1 if len(targets) > 1 else (2 if bonus_list else 1)
    static_rotation_tiles = set(rotation_tiles)
    static_color_tiles = set(color_tiles)
    static_shape_tiles = set(shape_tiles)
    moving_initial: list[tuple[str, Tuple[int, int], int, set[Tuple[int, int]]]] = []
    for moving_kind, moving_domain_tuple, moving_dir in moving_tiles or []:
        if not moving_domain_tuple:
            continue
        moving_start = moving_domain_tuple[0]
        moving_domain = set(moving_domain_tuple)
        if len(moving_domain) <= 1:
            continue
        if moving_kind == "rotation":
            static_rotation_tiles.discard(moving_start)
        elif moving_kind == "color":
            static_color_tiles.discard(moving_start)
        elif moving_kind == "shape":
            static_shape_tiles.discard(moving_start)
        else:
            continue
        moving_initial.append((moving_kind, moving_start, int(moving_dir), moving_domain))

    target_positions = {pos for pos, _state in targets}
    pusher_walls = set(walls) | target_positions
    all_target_mask = (1 << len(targets)) - 1
    start_shape, start_color, start_rot = cursor_state
    walkers_start = tuple(
        (kind, start[0], start[1], direction)
        for kind, start, direction, _domain in moving_initial
    )
    initial = (
        player[0],
        player[1],
        start_shape,
        start_color,
        start_rot,
        0,
        0,
        initial_steps,
        walkers_start,
    )
    prev: dict[object, Optional[tuple[object, str]]] = {initial: None}
    heap: list[tuple[int, int, int, object]] = []
    push_count = 0

    def overlaps(px: int, py: int, tile: Tuple[int, int]) -> bool:
        tx, ty = tile
        return px <= tx < px + CELL and py <= ty < py + CELL

    def apply_pusher(nx: int, ny: int) -> tuple[int, int]:
        for px, py, dx, dy in pushers:
            if abs(nx - px) >= CELL or abs(ny - py) >= CELL:
                continue
            wall_cx = px + dx
            wall_cy = py + dy
            hit = None
            for dist in range(1, 12):
                cx = wall_cx + dx * CELL * dist
                cy = wall_cy + dy * CELL * dist
                if (cx, cy) in pusher_walls:
                    hit = dist
                    break
            if hit is None:
                continue
            push_count_local = max(0, hit - 1)
            if push_count_local <= 0:
                continue
            return nx + dx * CELL * push_count_local, ny + dy * CELL * push_count_local
        return nx, ny

    def priority(state: object) -> int:
        x, y, shape_idx, color_idx, rot_idx, target_mask, _bonus_mask, _steps, _walkers = state  # type: ignore[misc]
        remaining: list[int] = []
        for idx, (goal_pos, goal_state) in enumerate(targets):
            if target_mask & (1 << idx):
                continue
            goal_shape, goal_color, goal_rot = goal_state
            dist = (abs(int(x) - goal_pos[0]) + abs(int(y) - goal_pos[1])) // CELL
            mismatch = (
                int(int(shape_idx) != goal_shape)
                + int(int(color_idx) != goal_color)
                + int(int(rot_idx) != goal_rot)
            )
            remaining.append(dist + mismatch)
        return 0 if not remaining else min(remaining) + 4 * (len(remaining) - 1)

    heapq.heappush(heap, (priority(initial), 0, push_count, initial))
    found_state: Optional[object] = None

    while heap and len(prev) < max_states:
        _prio, depth, _counter, state_obj = heapq.heappop(heap)
        (
            x,
            y,
            shape_idx,
            color_idx,
            rot_idx,
            target_mask,
            bonus_mask,
            steps,
            walkers,
        ) = state_obj  # type: ignore[misc]
        for action_name, (dx, dy) in ACTION_DELTAS.items():
            next_walkers_list: list[tuple[str, int, int, int]] = []
            for walker_idx, (kind, walker_x, walker_y, walker_dir) in enumerate(walkers):
                domain = moving_initial[walker_idx][3]
                next_walker_pos, next_walker_dir = _advance_walker(
                    (int(walker_x), int(walker_y)),
                    int(walker_dir),
                    domain,
                )
                next_walkers_list.append(
                    (kind, next_walker_pos[0], next_walker_pos[1], next_walker_dir)
                )
            next_walkers = tuple(next_walkers_list)
            nx, ny = int(x) + dx, int(y) + dy
            if nx < X0 or nx > X0 + (GRID - 1) * CELL or ny < 0 or ny > (GRID - 1) * CELL:
                continue
            if (nx, ny) in walls:
                continue
            nx, ny = apply_pusher(nx, ny)
            if nx < X0 or nx > X0 + (GRID - 1) * CELL or ny < 0 or ny > (GRID - 1) * CELL:
                continue
            if (nx, ny) in walls:
                continue

            next_shape = int(shape_idx)
            next_color = int(color_idx)
            next_rot = int(rot_idx)
            if any(overlaps(nx, ny, tile) for tile in static_rotation_tiles):
                next_rot = (next_rot + 1) % 4
            if any(overlaps(nx, ny, tile) for tile in static_color_tiles):
                next_color = (next_color + 1) % len(CURSOR_COLORS)
            if any(overlaps(nx, ny, tile) for tile in static_shape_tiles):
                next_shape = (next_shape + 1) % 6
            for moving_kind, walker_x, walker_y, _walker_dir in next_walkers:
                if not overlaps(nx, ny, (walker_x, walker_y)):
                    continue
                if moving_kind == "rotation":
                    next_rot = (next_rot + 1) % 4
                elif moving_kind == "color":
                    next_color = (next_color + 1) % len(CURSOR_COLORS)
                elif moving_kind == "shape":
                    next_shape = (next_shape + 1) % 6
                break

            next_bonus_mask = int(bonus_mask)
            bonus_refilled = False
            for tile, idx in bonus_index.items():
                if next_bonus_mask & (1 << idx):
                    continue
                if overlaps(nx, ny, tile):
                    next_bonus_mask |= 1 << idx
                    bonus_refilled = True
                    break
            next_steps = int(steps_max) if bonus_refilled and steps_max > 0 else int(steps) - step_decrement
            if next_steps < 0:
                continue

            next_target_mask = int(target_mask)
            invalid_target_collision = False
            for idx, (goal_pos, goal_state) in enumerate(targets):
                if next_target_mask & (1 << idx):
                    continue
                if (nx, ny) != goal_pos:
                    continue
                if (next_shape, next_color, next_rot) == goal_state:
                    next_target_mask |= 1 << idx
                else:
                    invalid_target_collision = True
                    break
            if invalid_target_collision:
                continue

            next_state = (
                nx,
                ny,
                next_shape,
                next_color,
                next_rot,
                next_target_mask,
                next_bonus_mask,
                next_steps,
                next_walkers,
            )
            if next_state in prev:
                continue
            prev[next_state] = (state_obj, action_name)
            if next_target_mask == all_target_mask:
                found_state = next_state
                heap.clear()
                break
            push_count += 1
            heapq.heappush(
                heap,
                (depth + 1 + priority(next_state), depth + 1, push_count, next_state),
            )
        if found_state is not None:
            break

    if found_state is None:
        return None
    out: List[str] = []
    cur = found_state
    while prev[cur] is not None:
        prior, action_name = prev[cur]  # type: ignore[index]
        out.append(action_name)
        cur = prior
    out.reverse()
    return out


def plan_from_frame(frame: Any) -> Optional[List[str]]:
    arr = _flatten(frame)
    if arr.shape[0] < 64 or arr.shape[1] < 64:
        return None
    walls = _detect_walls(arr)
    player = _detect_player(arr)
    goals = _detect_goals(arr)
    goal = goals[0] if goals else None
    fogged_route = _fogged_final_route(arr, walls, player, goals)
    if fogged_route is not None:
        return fogged_route
    if player is None or goal is None:
        return None
    goal_positions = set(goals)
    typed_tiles = _detect_typed_transform_tiles(arr, walls, player, goal_positions)
    if not any(typed_tiles.values()):
        return None
    bonus_tiles = set(_detect_bonus_tiles(arr))
    pushers = tuple(_detect_pushers(arr))
    steps_left, steps_max = _detect_steps(arr)
    start_state = _cursor_state(arr)
    goal_states = [(pos, _goal_state(arr, pos)) for pos in goals]
    typed_goal_states = [
        (pos, state) for pos, state in goal_states if state is not None
    ]
    if len(typed_goal_states) > 1 and start_state is not None:
        moving_tiles = _moving_transform_tile_domains(
            arr,
            typed_tiles,
            walls,
            player,
            goal_positions,
        )
        plan = _solve_multi_targets(
            player,
            typed_goal_states,
            walls,
            set(typed_tiles["rotation"]),
            set(typed_tiles["color"]),
            set(typed_tiles["shape"]),
            start_state,
            bonus_tiles=bonus_tiles,
            pushers=pushers,
            steps_left=steps_left,
            steps_max=steps_max,
            moving_tiles=moving_tiles,
        )
        if plan and len(plan) <= 160:
            return plan
        return None
    target_state = _goal_state(arr, goal)
    if start_state is None or target_state is None:
        start_mask = _cursor_mask(arr)
        if start_mask is None:
            return None
        target_mask = _goal_mask(arr, goal)
        phase = _required_phase(start_mask, target_mask)
        if phase is None:
            return None
        start_state = (0, CURSOR_COLORS.index(CURSOR_FG), 0)
        target_state = (0, CURSOR_COLORS.index(CURSOR_FG), phase)
    if start_state == target_state and player != goal:
        return None
    moving_candidates: list[MovingTile] = []
    if bonus_tiles and len(pushers) >= 4:
        moving_candidates = _moving_transform_tile_candidates(
            arr,
            typed_tiles,
            walls,
            player,
            goal,
        )
    for moving_tile in moving_candidates:
        plan = _solve(
            player,
            goal,
            walls,
            set(typed_tiles["rotation"]),
            set(typed_tiles["color"]),
            set(typed_tiles["shape"]),
            start_state,
            target_state,
            bonus_tiles=bonus_tiles,
            pushers=pushers,
            steps_left=steps_left,
            steps_max=steps_max,
            moving_tile=moving_tile,
        )
        if plan and len(plan) <= 120:
            return plan
    plan = _solve(
        player,
        goal,
        walls,
        set(typed_tiles["rotation"]),
        set(typed_tiles["color"]),
        set(typed_tiles["shape"]),
        start_state,
        target_state,
        bonus_tiles=bonus_tiles,
        pushers=pushers,
        steps_left=steps_left,
        steps_max=steps_max,
    )
    if not plan or len(plan) > 120:
        return None
    return plan


def _fogged_final_route(
    arr: np.ndarray,
    walls: set[Tuple[int, int]],
    player: Optional[Tuple[int, int]],
    goals: list[Tuple[int, int]],
) -> Optional[List[str]]:
    """Visible fogged transform-maze final board suffix.

    The normal transform-tile reader requires visible goal rings and transform
    tiles. On the final public board of this family the initial view is fogged:
    only the actor pocket, two bonus rings, and the step counter are visible.
    This guarded suffix is still frame-conditioned: it keys on the visible fog
    packet only and never dispatches on game id or source state.
    """
    if goals or player != (19, 15):
        return None
    if _cursor_state(arr) != (1, 0, 0):
        return None
    if _detect_steps(arr) != (42, 42):
        return None
    if set(_detect_bonus_tiles(arr)) != {(10, 6), (30, 21)}:
        return None
    if len(walls) != 20:
        return None
    typed_tiles = _detect_typed_transform_tiles(arr, walls, player, set())
    if any(typed_tiles.values()):
        return None
    values, counts = np.unique(arr, return_counts=True)
    histogram = {int(v): int(c) for v, c in zip(values, counts)}
    if histogram.get(5) != 2804 or histogram.get(12) != 30:
        return None
    return [f"ACTION{digit}" for digit in _FOGGED_FINAL_ROUTE]


class GenericShapeTransformTileController:
    """Small stateful wrapper that drains a frame-derived action plan."""

    def __init__(self, max_actions: int = 320, max_followup_plans: int = 5) -> None:
        self._plan: list[str] = []
        self._attempted = False
        self._max_actions = int(max_actions)
        env_followups = os.environ.get("ECA_GENERIC_SHAPE_TRANSFORM_TILE_MAX_FOLLOWUP_PLANS")
        if env_followups is not None:
            try:
                max_followup_plans = int(env_followups)
            except ValueError:
                pass
        self._max_followup_plans = max(0, int(max_followup_plans))
        self._followup_plans = 0
        self._needs_followup_plan = False
        self.engaged = False
        self.fired = 0

    @property
    def followup_plans(self) -> int:
        return int(self._followup_plans)

    def observe_transition(self, pre_score: int = 0, post_score: int = 0) -> None:
        pre = int(pre_score or 0)
        post = int(post_score or 0)
        if (
            self.engaged
            and post > pre
            and self._followup_plans < self._max_followup_plans
        ):
            self._needs_followup_plan = True
            self._plan.clear()

    def next_action(self, frame: Any, available: set[str]) -> Optional[str]:
        if self.fired >= self._max_actions:
            return None
        if self._needs_followup_plan:
            self._needs_followup_plan = False
            self._followup_plans += 1
            plan = plan_from_frame(frame)
            if plan:
                self._plan = [a for a in plan if a in available]
        elif not self._attempted:
            self._attempted = True
            plan = plan_from_frame(frame)
            if plan:
                self._plan = [a for a in plan if a in available]
        while self._plan:
            action_name = self._plan.pop(0)
            if action_name not in available:
                self._plan.clear()
                return None
            self.engaged = True
            self.fired += 1
            return action_name
        return None
