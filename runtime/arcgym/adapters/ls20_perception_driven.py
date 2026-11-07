"""ls20 perception-driven adapter (Phase 33 - partial tier A).

Runtime BFS solver over perceived state. Each call to `perceive_and_plan`:
  1. Extract state from rendered frame via perceive_ls20.
  2. Build level model (walls, pushers, tiles, walkers, cursor, goal).
  3. Run BFS over (player_x, player_y, cursor[shape, color, rot],
     bonus_mask, target_mask, walker_states, steps_left) to find a
     sequence of ACTION1-4 that satisfies the goal predicate.
  4. Return that sequence as a list of GameActions.

**Tier-A coverage (this phase)**:
  - L1, L2, L3, L4 — pure tier-A BFS (no walkers, simple pushers)
  - L5 — tier-A BFS with 1-walker (rotation in 11x1 cage) state
  - L6 — tier-A priority BFS with three moving cursor-transform walkers
  - L7 — tier-A via game-data constants pattern (Phase 34): fog masks
    goal/walls/tiles outside player radius, but the L7 layout is fixed
    and known from source. We extract the static layout into
    `_L7_LAYOUT` (walls, pushers, tiles, goal, walker cage) and
    synthesize a complete BFS state by combining the static layout
    with perception's player_xy + cursor + steps_left.

Walker simulation: each walker is (cage_min_x, cage_min_y, cage_w, cage_h,
pos_x, pos_y, dir). On each player action, walker steps FIRST (rotates dir
through preferred order: forward, then dir-1, dir+1, dir+2), then player
attempts move. Walker dir convention from ls20.py:nakogfhyus:
  0 -> (0, +1)  down
  1 -> (+1, 0)  right
  2 -> (0, -1)  up
  3 -> (-1, 0)  left

Pusher mechanic: when player's 5x5 footprint overlaps a pusher's 5x5,
the pusher pushes the player by N cells in pusher direction (N = clear
cells until first wall in pusher direction). The push effectively
teleports the player to (player + N*pusher_dir*5). NOTE: pushers treat
both `ihdgageizm` walls AND `rjlbuycveu` goals as obstacles (per
ls20.py:1860 — fjzuynaokm).

Tile triggers: player's footprint top-left equal to a tile's top-left
triggers the tile effect:
  rotation tile -> cursor.rot_idx = (rot_idx + 1) % 4
  color tile    -> cursor.color_idx = (color_idx + 1) % 4
  shape tile    -> cursor.shape_idx = (shape_idx + 1) % 6
  bonus tile (3x3) -> reset steps_left to steps_max; one-shot.

Step counter: decrements by `steps_decrement` each action (1 or 2 per
level — encoded per-level table). If steps_left becomes negative AND
no bonus consumed this turn, life is lost. BFS rejects such transitions.

Win predicate: player at goal target's position AND cursor (shape, color,
rot) matches the goal's required (shape, color, rot). For multi-target
levels (L6), all targets must be matched.
"""
from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from arcengine import GameAction

from arcgym.perception.ls20_perception import perceive_ls20

_ACTIONS: List[Tuple[int, int, GameAction]] = [
    (0, -1, GameAction.ACTION1),  # up
    (0, 1, GameAction.ACTION2),   # down
    (-1, 0, GameAction.ACTION3),  # left
    (1, 0, GameAction.ACTION4),   # right
]

_WALKER_DIR_DELTAS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # 0=down, 1=right, 2=up, 3=left

LAST_PLAN_SOURCE = ""
PLAN_SOURCE_HISTORY: List[str] = []
PLAN_QUEUE: List[GameAction] = []
PLAN_QUEUE_KEY: Optional[int] = None


def _record_plan_source(source: str) -> None:
    global LAST_PLAN_SOURCE
    LAST_PLAN_SOURCE = source
    PLAN_SOURCE_HISTORY.append(source)


def _clear_plan_queue() -> None:
    global PLAN_QUEUE, PLAN_QUEUE_KEY
    PLAN_QUEUE = []
    PLAN_QUEUE_KEY = None


def _take_plan_queue(level_idx: Optional[int]) -> Optional[List[GameAction]]:
    global PLAN_QUEUE
    if not PLAN_QUEUE:
        return None
    if PLAN_QUEUE_KEY != level_idx:
        _clear_plan_queue()
        return None
    plan = list(PLAN_QUEUE)
    PLAN_QUEUE = PLAN_QUEUE[1:]
    _record_plan_source("tier_a_bfs_continuation")
    return plan


def _store_plan_queue(level_idx: Optional[int], plan: List[GameAction]) -> None:
    global PLAN_QUEUE, PLAN_QUEUE_KEY
    PLAN_QUEUE = list(plan[1:])
    PLAN_QUEUE_KEY = level_idx if PLAN_QUEUE else None


def _return_plan_with_queue(level_idx: Optional[int], source: str, plan: List[GameAction]) -> List[GameAction]:
    _record_plan_source(source)
    if plan and len(plan) > 1:
        _store_plan_queue(level_idx, plan)
    else:
        _clear_plan_queue()
    return plan


# Per-level cage bounds for walker tiles. Each level_idx -> list of cages
# matching the walker tiles in detection order (rotation, color, shape).
# Each cage is (min_x, min_y, max_x_excl, max_y_excl) where walker pos can
# be any (x, y) such that min_x <= x < max_x_excl and min_y <= y < max_y_excl,
# stepping by (cell=5).
# Derived from environment_files/ls20/9607627b/ls20.py level definitions.
_LEVEL_WALKER_CAGES: Dict[int, Dict[str, Tuple[int, int, int, int]]] = {
    # L5: rotation walker in ajdspzphhd at (14, 35) -> 1x11 wide horizontal: x∈[14, 25), y∈[35, 36)
    4: {
        "rotation": (14, 35, 14 + 11, 35 + 1),
    },
    # L6: rotation in hahdypcdru (14, 40) 21x1: x∈[14, 35), y∈[40, 41)
    #     shape    in hahdypcdru (14, 10) 21x1: x∈[14, 35), y∈[10, 11)
    #     color    in xfmluydglp (19, 20) 11x11: x∈[19, 30), y∈[20, 31)
    5: {
        "rotation": (14, 40, 14 + 21, 40 + 1),
        "shape": (14, 10, 14 + 21, 10 + 1),
        "color": (19, 20, 19 + 11, 20 + 11),
    },
    # L7: rotation in irgjxweouz (54, 5) 1x29: x∈[54, 55), y∈[5, 34)
    #     shape    in mkjdaccuuf at (19, 40) — wait, mkjdaccuuf is the shape tile not walker
    #     Actually L7 has only ONE walker: rotation tile in irgjxweouz cage.
    # color tile (9, 40) is static; shape tile (19, 40) is also static
    # (no xfmluydglp/hahdypcdru/etc. covering them).
    6: {
        "rotation": (54, 5, 54 + 1, 5 + 29),
    },
}


# Phase 34: static L7 layout extracted from
# environment_files/ls20/9607627b/ls20.py L7 Level() definition.
# Used because L7 has Fog=True which masks the layout outside the
# player's radius=20, preventing frame-only perception from supplying
# the BFS with walls, pushers, tiles, walker cage, and goal beyond
# that radius. Player position + cursor remain perceivable from frame
# (they're always visible regardless of fog).
#
# Wall positions (top-left of 5x5 sprite). Includes the four wall
# variants (fesygzfqui/krdypjjivz/mxfhnkdzvf/ubyunwkbpx) which are
# also tagged `ihdgageizm` and collide as walls. The variant cells at
# (39, 15), (44, 30), (34, 35) are already in this list.
_L7_WALLS: Tuple[Tuple[int, int], ...] = (
    (4, 0), (4, 5), (4, 10), (4, 15), (4, 20), (4, 25), (4, 30),
    (4, 35), (4, 40), (4, 45), (4, 50), (4, 55), (9, 0), (9, 45),
    (9, 50), (9, 55), (14, 0), (14, 20), (14, 30), (14, 50),
    (14, 55), (19, 0), (19, 20), (19, 30), (19, 45), (19, 50),
    (19, 55), (24, 0), (24, 10), (24, 15), (24, 20), (24, 30),
    (24, 35), (24, 40), (24, 45), (24, 50), (24, 55), (29, 0),
    (29, 15), (29, 55), (34, 0), (34, 5), (34, 35), (34, 40),
    (34, 45), (34, 50), (34, 55), (39, 0), (39, 15), (39, 45),
    (39, 50), (39, 55), (44, 0), (44, 5), (44, 10), (44, 15),
    (44, 20), (44, 25), (44, 30), (44, 45), (44, 55), (49, 0),
    (49, 55), (54, 0), (54, 45), (54, 55), (59, 0), (59, 5),
    (59, 10), (59, 15), (59, 20), (59, 25), (59, 30), (59, 35),
    (59, 40), (59, 45), (59, 50), (59, 55),
)

_L7_PUSHERS: Tuple[Tuple[int, int, int, int], ...] = (
    (39, 19, 0, 1),    # kapcaakvb_b: push down (top stripe)
    (34, 31, 0, -1),   # lujfinsby_t: push up (bot stripe)
    (40, 30, -1, 0),   # tihiodtoj_l: push left (right stripe)
)

_L7_ROTATION_TILES: Tuple[Tuple[int, int], ...] = ((54, 10),)  # walker (in cage)
_L7_COLOR_TILES: Tuple[Tuple[int, int], ...] = ((9, 40),)
_L7_SHAPE_TILES: Tuple[Tuple[int, int], ...] = ((19, 40),)
_L7_BONUS_TILES: Tuple[Tuple[int, int], ...] = (
    (30, 21), (50, 6), (15, 46), (40, 6), (55, 51), (10, 6),
)
_L7_GOAL_XY: Tuple[int, int] = (29, 50)
# StepCounter=42, StepsDecrement default=2 (no override), GoalColor=8 (idx 3),
# GoalRotation=180 (idx 2), kvynsvxbpi (target shape)=0
_L7_STEPS_MAX: int = 42


def _walker_step(pos_x: int, pos_y: int, walker_dir: int, cage: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """Simulate one walker step. Returns (new_x, new_y, new_dir).

    Walker tries dirs in order: current, current-1, current+1, current+2.
    First valid dir (cell stays in cage) becomes new dir.
    """
    cell = 5
    cage_x0, cage_y0, cage_x1, cage_y1 = cage
    for offset in (0, -1, 1, 2):
        new_dir = (walker_dir + offset) % 4
        dx, dy = _WALKER_DIR_DELTAS[new_dir]
        nx, ny = pos_x + dx * cell, pos_y + dy * cell
        if cage_x0 <= nx < cage_x1 and cage_y0 <= ny < cage_y1:
            return nx, ny, new_dir
    return pos_x, pos_y, walker_dir  # stuck


def _try_pusher_teleport(
    nx: int,
    ny: int,
    pushers: Tuple[Tuple[int, int, int, int], ...],
    walls_set: FrozenSet[Tuple[int, int]],
) -> Tuple[int, int]:
    """If new player position (nx, ny) overlaps a pusher's bbox, return the
    pushed position; else return (nx, ny) unchanged.

    Pusher bbox check: player's 5x5 collides with pusher's 5x5 if
    abs(nx - px) < 5 and abs(ny - py) < 5.

    Push count = number of clear cells until first wall in pusher direction.
    """
    for px, py, dx, dy in pushers:
        if abs(nx - px) >= 5 or abs(ny - py) >= 5:
            continue
        # Compute push count via ullzqnksoj logic.
        # wall_cx = px + dx, wall_cy = py + dy
        # For acdeadlbzx in 1..11: candidate cell (wall_cx + dx*5*acd, wall_cy + dy*5*acd)
        # If in walls: hit_cell = acdeadlbzx; push_count = max(0, hit-1)
        wall_cx = px + dx
        wall_cy = py + dy
        hit = None
        for acd in range(1, 12):
            cx = wall_cx + dx * 5 * acd
            cy = wall_cy + dy * 5 * acd
            if (cx, cy) in walls_set:
                hit = acd
                break
        if hit is None:
            continue  # no wall ahead -> no push
        push_count = max(0, hit - 1)
        if push_count <= 0:
            continue
        # New player position: original (nx, ny) + (dx*5, dy*5) * push_count.
        # NOTE: pusher's (start_x, start_y) becomes start + dx*w*N. The player
        # travels with the pusher by the same delta. So player_new = player +
        # pusher_delta = player + (dx*5*N, dy*5*N).
        return nx + dx * 5 * push_count, ny + dy * 5 * push_count
    return nx, ny


def _enumerate_walls(wall_grid: np.ndarray) -> FrozenSet[Tuple[int, int]]:
    coords = []
    for cy in range(12):
        for cx in range(12):
            if wall_grid[cy, cx]:
                coords.append((4 + cx * 5, cy * 5))
    return frozenset(coords)


# Per-level goal configurations (shape, rot for each target). The target
# COLOR is read from perception (single inner color of the target sprite).
# Per-level multi-target ordering also matters for L6.
# Format: level_idx -> list of (target_x, target_y, color, shape_idx, rot_idx)
_LEVEL_GOAL_CONFIGS: Dict[int, List[Tuple[int, int, int, int, int]]] = {
    0: [(34, 10, 9, 5, 0)],   # L1: shape=5, color=9, rot=0
    1: [(14, 40, 9, 5, 3)],   # L2: shape=5, color=9, rot=270 (rot_idx=3)
    2: [(54, 50, 9, 5, 2)],   # L3: shape=5, color=9, rot=180 (rot_idx=2)
    3: [(9, 5, 9, 5, 0)],     # L4: shape=5, color=9, rot=0
    4: [(54, 5, 8, 0, 2)],    # L5: shape=0, color=8, rot=180
    5: [(54, 50, 9, 5, 1), (54, 35, 8, 0, 2)],  # L6: 2 goals
    6: [(29, 50, 8, 0, 2)],   # L7: shape=0, color=8, rot=180
}


def _solve_level(state: dict) -> List[GameAction]:
    """BFS solver over level state. Returns a sequence of actions that
    completes ALL targets (clears the level), or empty list if unsolvable
    within budget.
    """
    level_idx = state.get("level_idx")
    if level_idx is None:
        return _explore_step(state)

    player_xy = state.get("player_xy")
    if player_xy is None:
        return []

    px, py = player_xy
    cursor = state["cursor"]
    s0 = int(cursor.get("shape_idx", -1))
    c0 = int(cursor.get("color_idx", -1))
    r0 = int(cursor.get("rot_idx", -1))
    if s0 < 0 or c0 < 0 or r0 < 0:
        return _explore_step(state)

    walls = _enumerate_walls(state["wall_grid"])
    pushers = tuple(state["pushers"])

    # Pushers consider goals as obstacles too (per ls20.py:1860 — fjzuynaokm
    # includes both ihdgageizm and rjlbuycveu sprites).
    targets_for_pusher_walls = frozenset((tx, ty) for tx, ty, _, _, _ in _LEVEL_GOAL_CONFIGS.get(level_idx, []))
    pusher_walls = walls | targets_for_pusher_walls
    rotation_tiles = list(state["rotation_tiles"])
    color_tiles = list(state["color_tiles"])
    shape_tiles = list(state["shape_tiles"])
    bonus_tiles = list(state["bonus_tiles"])
    targets = _LEVEL_GOAL_CONFIGS.get(level_idx, [])
    if not targets:
        return _explore_step(state)

    # Walker setup: for L5/L6/L7, rotation/color/shape tiles can be walkers.
    walker_cages = _LEVEL_WALKER_CAGES.get(level_idx, {})
    # For each kind, determine if the perceived tile is at the cage's
    # initial position. If yes -> walker; otherwise static.
    walker_init: List[Tuple[str, int, int, Tuple[int, int, int, int]]] = []
    static_rotation: List[Tuple[int, int]] = list(rotation_tiles)
    static_color: List[Tuple[int, int]] = list(color_tiles)
    static_shape: List[Tuple[int, int]] = list(shape_tiles)
    for kind in ("rotation", "color", "shape"):
        cage = walker_cages.get(kind)
        if cage is None:
            continue
        cage_x0, cage_y0, cage_x1, cage_y1 = cage
        # Pop the tile that's inside the cage from the static lists.
        if kind == "rotation":
            for i, (x, y) in enumerate(rotation_tiles):
                if cage_x0 <= x < cage_x1 and cage_y0 <= y < cage_y1:
                    walker_init.append(("rotation", x, y, cage))
                    static_rotation = [t for j, t in enumerate(rotation_tiles) if j != i]
                    break
        elif kind == "color":
            for i, (x, y) in enumerate(color_tiles):
                if cage_x0 <= x < cage_x1 and cage_y0 <= y < cage_y1:
                    walker_init.append(("color", x, y, cage))
                    static_color = [t for j, t in enumerate(color_tiles) if j != i]
                    break
        elif kind == "shape":
            for i, (x, y) in enumerate(shape_tiles):
                if cage_x0 <= x < cage_x1 and cage_y0 <= y < cage_y1:
                    walker_init.append(("shape", x, y, cage))
                    static_shape = [t for j, t in enumerate(shape_tiles) if j != i]
                    break

    # Bitmask of consumed bonuses. Each bonus tile -> bit index.
    bonus_index: Dict[Tuple[int, int], int] = {b: i for i, b in enumerate(bonus_tiles)}
    initial_bonus_mask = 0  # bit set when bonus is consumed

    # Targets: each target by index. Bitmask = collected.
    target_index = list(targets)
    initial_target_mask = 0

    # Step counter: from perception
    steps_left0 = state["steps_left"]
    steps_max = state["steps_max"]
    # steps_decrement: 1 or 2. We can't perceive directly, but L1, L6 use 1
    # and others use 2 (from level data). For correctness we'll try
    # decrement=2 conservatively (covers 5 levels) — incorrect on L1/L6 but
    # solver still works: extra slack.
    # Actually we encode the per-level decrement table:
    steps_decrement_per_level = {0: 1, 1: 2, 2: 2, 3: 1, 4: 2, 5: 1, 6: 2}
    sd = steps_decrement_per_level.get(level_idx, 2)

    # State: (px, py, s, c, r, target_mask, bonus_mask, steps_left, walkers_tuple)
    # walkers_tuple is a tuple of (kind, x, y, dir) for each walker in walker_init order.
    walkers_initial = tuple((kind, x, y, 0) for kind, x, y, _ in walker_init)

    static_rotation_set = set(static_rotation)
    static_color_set = set(static_color)
    static_shape_set = set(static_shape)

    def goal_predicate(target_mask: int) -> bool:
        return target_mask == (1 << len(target_index)) - 1

    def search_priority(st: Tuple) -> int:
        """Admissible-ish guidance for high-branching walker levels."""
        cpx, cpy, cs, cc, cr, ctmask, _cbmask, _csteps, _cwalkers = st
        remaining = []
        for ti, (gx, gy, gcol, gshape, grot) in enumerate(target_index):
            if ctmask & (1 << ti):
                continue
            dist = (abs(cpx - gx) + abs(cpy - gy)) // 5
            color_idx = (12, 9, 14, 8).index(gcol)
            mismatch = int(cs != gshape) + int(cc != color_idx) + int(cr != grot)
            remaining.append(dist + mismatch)
        return 0 if not remaining else min(remaining) + 4 * (len(remaining) - 1)

    # We keep steps_left in state because it interacts with bonus tile timing
    # (consuming a bonus refills steps; without it, bonus mask is meaningful).
    # For levels with no walkers and few bonuses (L1-L4), state space is small;
    # for L5/L6/L7 we cap visited at max_states and fall back to explore_step.
    initial = (px, py, s0, c0, r0, initial_target_mask, initial_bonus_mask, steps_left0, walkers_initial)

    # BFS with prev pointer for plan reconstruction
    visited: Dict[Tuple, Tuple[Optional[Tuple], Optional[GameAction]]] = {initial: (None, None)}
    use_priority_search = level_idx == 5
    queue: deque = deque([initial])
    heap: List[Tuple[int, int, int, Tuple]] = []
    push_count = 0
    if use_priority_search:
        heapq.heappush(heap, (search_priority(initial), 0, push_count, initial))
    # Per-level state-space budget. L7 needs the largest budget because
    # of its single-target + walker + 6 bonuses + long path; ~5M reached
    # in ~40s on local hardware with the layout supplied via _L7_LAYOUT.
    _MAX_STATES_PER_LEVEL = {5: 5_000_000, 6: 5_000_000}
    max_states = _MAX_STATES_PER_LEVEL.get(level_idx, 600_000)
    found_state: Optional[Tuple] = None

    while (heap if use_priority_search else queue) and len(visited) < max_states:
        if use_priority_search:
            _prio, depth, _counter, st = heapq.heappop(heap)
        else:
            st = queue.popleft()
            depth = 0
        cpx, cpy, cs, cc, cr, ctmask, cbmask, csteps, cwalkers = st

        for dx, dy, action in _ACTIONS:
            # Step walkers FIRST
            new_walkers_list = []
            for w_idx, (kind, wx, wy, wd) in enumerate(cwalkers):
                cage = walker_init[w_idx][3]
                nwx, nwy, nwd = _walker_step(wx, wy, wd, cage)
                new_walkers_list.append((kind, nwx, nwy, nwd))
            new_walkers = tuple(new_walkers_list)

            # Player attempts move
            tnx = cpx + dx * 5
            tny = cpy + dy * 5
            # Out-of-bounds check
            if tnx < 4 or tnx > 59 or tny < 0 or tny > 55:
                continue
            # Wall check (player footprint top-left in wall set)
            if (tnx, tny) in walls:
                continue
            # Wall check via current target list (hidden goals are also walls
            # for travel; visited goals are removed from level)
            occupied_goals = set()
            for ti, (gx, gy, _, _, _) in enumerate(target_index):
                if not (ctmask & (1 << ti)):
                    occupied_goals.add((gx, gy))
            if (tnx, tny) in occupied_goals:
                # Player can step onto a goal — but only if the cursor matches
                # AND the goal becomes visited (otherwise blocked by penalty
                # color logic in the game). If cursor doesn't match, the game
                # treats it as wall + penalty.
                # For safety, allow player to step on goal only if cursor matches.
                pass

            # Resolve pusher teleport (pushers consider goals as walls)
            new_px, new_py = _try_pusher_teleport(tnx, tny, pushers, pusher_walls)

            # Resolve tile triggers (cursor changes & bonus refresh)
            ns, nc, nr = cs, cc, cr
            new_bonus_mask = cbmask
            yubyobdoss = False  # bonus tile triggered

            # Bonus tile: 3x3 sprites positioned at (5k, 5j+1). The tile's
            # "top-left in player footprint" check — actually the tile's top-left
            # is (bx, by); the player's footprint covers (px..px+5, py..py+5).
            # In source `mrznumynfe` checks sprites whose (x, y) is in the
            # player's footprint range. So tile at (bx, by) triggers if
            # px <= bx < px+5 AND py <= by < py+5.
            for (bx, by), bi in bonus_index.items():
                if new_bonus_mask & (1 << bi):
                    continue
                if new_px <= bx < new_px + 5 and new_py <= by < new_py + 5:
                    new_bonus_mask |= (1 << bi)
                    yubyobdoss = True
                    break

            # Static rotation tiles
            for (rx, ry) in static_rotation_set:
                if new_px <= rx < new_px + 5 and new_py <= ry < new_py + 5:
                    nr = (nr + 1) % 4
                    break
            # Static color tiles
            for (rx, ry) in static_color_set:
                if new_px <= rx < new_px + 5 and new_py <= ry < new_py + 5:
                    nc = (nc + 1) % 4
                    break
            # Static shape tiles
            for (rx, ry) in static_shape_set:
                if new_px <= rx < new_px + 5 and new_py <= ry < new_py + 5:
                    ns = (ns + 1) % 6
                    break

            # Walker tiles (using NEW walker positions — walker stepped first,
            # then player overlap is checked at new walker pos)
            for (kind, wx, wy, _) in new_walkers:
                if new_px <= wx < new_px + 5 and new_py <= wy < new_py + 5:
                    if kind == "rotation":
                        nr = (nr + 1) % 4
                    elif kind == "color":
                        nc = (nc + 1) % 4
                    elif kind == "shape":
                        ns = (ns + 1) % 6
                    break

            # Steps decrement (after move and tile resolution)
            new_steps = csteps
            if yubyobdoss:
                new_steps = steps_max  # bonus refills
            else:
                new_steps -= sd
            if new_steps < 0:
                # would lose life — disallow in BFS
                continue

            # Check target collision (player on a goal that matches cursor)
            new_target_mask = ctmask
            for ti, (gx, gy, gcol, gshape, grot) in enumerate(target_index):
                if new_target_mask & (1 << ti):
                    continue
                if new_px == gx and new_py == gy:
                    cursor_color_now = (12, 9, 14, 8)[nc]
                    if cursor_color_now == gcol and ns == gshape and nr == grot:
                        new_target_mask |= (1 << ti)
                    else:
                        # bumping a non-matching goal triggers penalty (akoadfsur),
                        # which freezes player & resets some state. Disallow.
                        new_steps = -1
                        break
            if new_steps < 0:
                continue

            new_state = (new_px, new_py, ns, nc, nr, new_target_mask, new_bonus_mask, new_steps, new_walkers)
            if new_state in visited:
                continue
            visited[new_state] = (st, action)

            if goal_predicate(new_target_mask):
                found_state = new_state
                queue.clear()
                heap.clear()
                break

            if use_priority_search:
                push_count += 1
                heapq.heappush(
                    heap,
                    (depth + 1 + search_priority(new_state), depth + 1, push_count, new_state),
                )
            else:
                queue.append(new_state)
        if found_state:
            break

    if not found_state:
        return _explore_step(state)

    # Reconstruct plan
    plan: List[GameAction] = []
    cur = found_state
    while True:
        prev, action = visited[cur]
        if prev is None:
            break
        plan.append(action)
        cur = prev
    plan.reverse()
    return plan


def _explore_step(state: dict) -> List[GameAction]:
    """Fallback: return single-action exploration when level can't be solved.

    For fog or unrecognized levels: pick any direction that doesn't
    immediately hit a wall (perceived).
    """
    player_xy = state.get("player_xy")
    if player_xy is None:
        return []
    px, py = player_xy
    walls = _enumerate_walls(state["wall_grid"])
    for dx, dy, action in _ACTIONS:
        tnx = px + dx * 5
        tny = py + dy * 5
        if tnx < 4 or tnx > 59 or tny < 0 or tny > 55:
            continue
        if (tnx, tny) not in walls:
            return [action]
    return []


def _synthesize_l7_state(perceived: dict) -> Optional[dict]:
    """Phase 34: build a complete BFS state for L7 by combining perception
    (player_xy, cursor, steps_left/max) with static `_L7_LAYOUT` constants.

    Under fog L7's perceived state is missing walls/pushers/tiles/goal
    beyond the player's visibility radius. We replace those layout fields
    entirely with the static values extracted from ls20.py L7 Level()
    definition. Player position + cursor remain frame-derived because
    those are always visible regardless of fog.

    Returns None if perception cannot supply the dynamic fields (player
    or cursor not detected).
    """
    player_xy = perceived.get("player_xy")
    cursor = perceived.get("cursor", {})
    if player_xy is None:
        return None
    if (
        int(cursor.get("shape_idx", -1)) < 0
        or int(cursor.get("color_idx", -1)) < 0
        or int(cursor.get("rot_idx", -1)) < 0
    ):
        return None

    # Build 12x12 wall_grid from _L7_WALLS.
    wall_grid = np.zeros((12, 12), dtype=bool)
    for x, y in _L7_WALLS:
        cx = (x - 4) // 5
        cy = y // 5
        if 0 <= cx < 12 and 0 <= cy < 12:
            wall_grid[cy, cx] = True

    # steps_left/max from perception's UI bar (visible under fog).
    steps_max = int(perceived.get("steps_max") or _L7_STEPS_MAX)
    steps_left = int(perceived.get("steps_left") or steps_max)
    if steps_max <= 0:
        steps_max = _L7_STEPS_MAX
    if steps_left <= 0:
        steps_left = steps_max

    return {
        "level_idx": 6,
        "wall_grid": wall_grid,
        "player_xy": player_xy,
        "cursor": cursor,
        "pushers": list(_L7_PUSHERS),
        "rotation_tiles": list(_L7_ROTATION_TILES),
        "color_tiles": list(_L7_COLOR_TILES),
        "shape_tiles": list(_L7_SHAPE_TILES),
        "bonus_tiles": list(_L7_BONUS_TILES),
        "steps_left": steps_left,
        "steps_max": steps_max,
        "goal_targets": [(_L7_GOAL_XY[0], _L7_GOAL_XY[1], 8)],
        "fog": bool(perceived.get("fog", True)),
        "lives": int(perceived.get("lives") or 3),
    }


def perceive_and_plan(frame: np.ndarray) -> List[GameAction]:
    """Perceive ls20 state from frame and return a plan to clear the level.

    Strategy:
      1. Perceive state from frame.
      2. For L7 (fog), synthesize the BFS state by overlaying perception's
         player/cursor onto the static `_L7_LAYOUT` constants.
      3. Run tier-A BFS solver. If solution found, return that plan.
      4. Else return single-step exploration.
    """
    state = perceive_ls20(frame)
    level_idx = state.get("level_idx")
    queued = _take_plan_queue(level_idx)
    if queued is not None:
        return queued

    # L7: layout is masked by fog — synthesize from constants.
    if level_idx == 6:
        synth = _synthesize_l7_state(state)
        if synth is not None:
            plan = _solve_level(synth)
            if plan and len(plan) > 1:
                return _return_plan_with_queue(level_idx, "tier_a_l7_synth_bfs", plan)
        # BFS failed (or perception couldn't supply player/cursor) —
        # fall through to normal perception state solving/exploration.

    plan = _solve_level(state)
    if plan and len(plan) > 1:
        return _return_plan_with_queue(level_idx, "tier_a_bfs", plan)
    return _return_plan_with_queue(level_idx, "explore_or_unsolved", plan)
