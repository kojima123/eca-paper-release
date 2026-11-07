"""r11l canvas-centroid adapter, frame-perception-driven.

Phase 23.1 canary shipped L1 only. Phase 24.A.2 ports the full
destination generator + safe-permutation search from
`arcgym.adapters.r11l_canvas` to use perception state (cell sets)
instead of `env._env._game` sprite collisions.

Scope:
  - L1 (1 group, 2 pieces, 1 target).
  - L2-L4 (multi-group centroid puzzles with defgjl shape constraints).
  - L5-L6 (whkxtx colour-absorber): level-static specs plus
    runtime waypoint/centroid solver.

Public API:
    plan_next_actions_from_state(state) -> list of (GameAction, data)
    perceive_and_plan(frame) -> list of (GameAction, data)
"""
from __future__ import annotations

from collections import deque
from itertools import permutations
from typing import Any, List, Tuple, Optional, Set

from arcengine import GameAction

from arcgym.perception.r11l_perception import perceive_r11l


PIECE_W = PIECE_H = 5
CANVAS_W = CANVAS_H = 5
GRID_SIZE = 60

LAST_PLAN_SOURCE = ""
PLAN_SOURCE_HISTORY: List[str] = []
TARGET_CACHE: dict[frozenset[int], dict] = {}
PIECE_CACHE: List[dict] = []
PLAN_QUEUE: List[Tuple[GameAction, dict]] = []
PLAN_QUEUE_KEY: Optional[Tuple[Any, ...]] = None


def _record_plan_source(source: str) -> None:
    global LAST_PLAN_SOURCE
    LAST_PLAN_SOURCE = source
    PLAN_SOURCE_HISTORY.append(source)


def _continuation_key(state: dict) -> Tuple[Any, ...]:
    canvases = state.get("canvases", [])
    canvas_colors = set()
    for canvas in canvases:
        for color in canvas.get("colors", {canvas.get("color")}):
            if color is not None:
                canvas_colors.add(int(color))
    return (
        state.get("level_signature"),
        len(canvases),
        tuple(sorted(canvas_colors)),
    )


def _clear_plan_queue() -> None:
    global PLAN_QUEUE, PLAN_QUEUE_KEY
    PLAN_QUEUE = []
    PLAN_QUEUE_KEY = None


def _take_plan_queue(key: Tuple[Any, ...]) -> Optional[List[Tuple[GameAction, dict]]]:
    global PLAN_QUEUE
    if not PLAN_QUEUE:
        return None
    key_matches = PLAN_QUEUE_KEY == key
    absorber_continuation = (
        isinstance(PLAN_QUEUE_KEY, tuple)
        and PLAN_QUEUE_KEY
        and PLAN_QUEUE_KEY[0] in _ABSORBER_SPECS
        and key[0] in {PLAN_QUEUE_KEY[0], None}
        and key[1:] == PLAN_QUEUE_KEY[1:]
    )
    if not key_matches and not absorber_continuation:
        _clear_plan_queue()
        return None
    plan = list(PLAN_QUEUE)
    PLAN_QUEUE = PLAN_QUEUE[1:]
    _record_plan_source(
        "tier_a_absorber_solver_continuation"
        if absorber_continuation
        else "tier_a_solver_continuation"
    )
    return plan


def _store_plan_queue(key: Tuple[Any, ...], plan: List[Tuple[GameAction, dict]]) -> None:
    global PLAN_QUEUE, PLAN_QUEUE_KEY
    PLAN_QUEUE = list(plan[1:])
    PLAN_QUEUE_KEY = key if PLAN_QUEUE else None


def _target_key(target: dict) -> frozenset[int]:
    return frozenset(int(c) for c in target.get("colors", {target.get("color")}) if c is not None)


def _remember_visible_targets(targets: List[dict]) -> None:
    for target in targets:
        key = _target_key(target)
        if key:
            TARGET_CACHE[key] = dict(target)
            for color in key:
                TARGET_CACHE[frozenset({int(color)})] = dict(target)


def _augment_targets_from_cache(canvases: List[dict], targets: List[dict]) -> List[dict]:
    visible_keys = {_target_key(target) for target in targets}
    augmented = list(targets)
    for canvas in canvases:
        wanted_keys: List[frozenset[int]] = []
        for color in (canvas.get("piece_color"), canvas.get("color")):
            if color is not None:
                wanted_keys.append(frozenset({int(color)}))
        canvas_key = frozenset(int(c) for c in canvas.get("colors", {canvas.get("color")}) if c is not None)
        if canvas_key:
            wanted_keys.append(canvas_key)
        for wanted_key in wanted_keys:
            if not wanted_key or any(wanted_key & key for key in visible_keys):
                continue
            cached = TARGET_CACHE.get(wanted_key)
            if cached is not None:
                augmented.append(dict(cached))
                visible_keys.add(wanted_key)
    return augmented


def _remember_visible_pieces(active: Tuple[int, int], active_color: int, inactive: List[dict]) -> None:
    global PIECE_CACHE
    pieces = [{"pos": active, "color": active_color, "active": True}]
    for piece in inactive:
        color = piece.get("color")
        if color is not None:
            pieces.append({"pos": piece["pos"], "color": int(color), "active": False})
    PIECE_CACHE = [dict(piece) for piece in pieces]


def _clear_piece_cache_on_scene_reset(
    active: Tuple[int, int],
    active_color: int,
    inactive: List[dict],
) -> None:
    if not PIECE_CACHE:
        return
    visible = {(active, int(active_color))}
    visible.update((piece["pos"], int(piece["color"])) for piece in inactive if piece.get("color") is not None)
    cached = {
        (tuple(piece["pos"]), int(piece["color"]))
        for piece in PIECE_CACHE
        if piece.get("pos") is not None and piece.get("color") is not None
    }
    if visible and cached and not (visible & cached):
        PIECE_CACHE.clear()


def _rects_touch(
    a_pos: Tuple[int, int],
    a_size: Tuple[int, int],
    b_pos: Tuple[int, int],
    b_size: Tuple[int, int],
) -> bool:
    ax, ay = a_pos
    aw, ah = a_size
    bx, by = b_pos
    bw, bh = b_size
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _augment_hidden_pieces_from_cache(
    canvases: List[dict],
    active: Tuple[int, int],
    active_color: int,
    inactive: List[dict],
) -> List[dict]:
    visible = {(active, int(active_color))}
    visible.update((piece["pos"], int(piece["color"])) for piece in inactive if piece.get("color") is not None)
    augmented = list(inactive)
    for cached in PIECE_CACHE:
        color = cached.get("color")
        pos = cached.get("pos")
        if color is None or pos is None:
            continue
        piece = (tuple(pos), int(color))
        if piece in visible:
            continue
        if (
            not bool(cached.get("active"))
            and piece[1] == int(active_color)
            and _rects_touch(piece[0], (PIECE_W, PIECE_H), active, (PIECE_W, PIECE_H))
        ):
            augmented.append({"pos": piece[0], "color": piece[1]})
            visible.add(piece)
            continue
        for canvas in canvases:
            piece_color = canvas.get("piece_color", canvas.get("color"))
            if piece[1] == piece_color:
                continue
            if _rects_touch(piece[0], (PIECE_W, PIECE_H), canvas["pos"], (CANVAS_W, CANVAS_H)):
                augmented.append({"pos": piece[0], "color": piece[1]})
                visible.add(piece)
                break
    return augmented


def _piece_center(tl: Tuple[int, int]) -> Tuple[int, int]:
    return (tl[0] + PIECE_W // 2, tl[1] + PIECE_H // 2)


def _click_inside_pieces(click: Tuple[int, int],
                         pieces: List[Tuple[int, int]]) -> bool:
    cx, cy = click
    return any(px <= cx < px + PIECE_W and py <= cy < py + PIECE_H
               for (px, py) in pieces)


def _destination_click_inside_piece(piece: Tuple[int, int], dest: Tuple[int, int]) -> bool:
    click_x, click_y = _piece_center(dest)
    px, py = piece
    return px <= click_x < px + PIECE_W and py <= click_y < py + PIECE_H


def _block_overlaps_cells(pos: Tuple[int, int], w: int, h: int,
                          cells: Set[Tuple[int, int]]) -> bool:
    px, py = pos
    for dy in range(h):
        for dx in range(w):
            if (px + dx, py + dy) in cells:
                return True
    return False


# r11l 5x5 canvas (roefwu-*) has a rounded-square pattern: corners -1.
# 21/25 non-(-1) cells. Pixel-perfect canvas-on-defgjl check uses this
# pattern (verified by inspecting roefwu-grhcew at (25, 13) in L3).
_CANVAS_PATTERN: List[Tuple[int, int]] = [
    (1, 0), (2, 0), (3, 0),
    (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
    (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
    (0, 3), (1, 3), (2, 3), (3, 3), (4, 3),
    (1, 4), (2, 4), (3, 4),
]


def _canvas_safe(canvas_pos: Tuple[int, int],
                 defgjl_cells: Set[Tuple[int, int]]) -> bool:
    """Pixel-perfect canvas-on-defgjl collision using r11l's rounded-
    square pattern (4 corner -1s)."""
    cx, cy = canvas_pos
    if not (0 <= cx <= GRID_SIZE - CANVAS_W and 0 <= cy <= GRID_SIZE - CANVAS_H):
        return False
    for (dx, dy) in _CANVAS_PATTERN:
        if (cx + dx, cy + dy) in defgjl_cells:
            return False
    return True


# r11l 5x5 piece sprites have a DIAMOND/PLUS pattern (verified by
# inspecting roefwulewcui-pumlzd at (43, 33) in L2): non-(-1) cells
# form a diamond. piece-on-wakneh collision must be pixel-perfect, so
# we use this fixed pattern.
_PIECE_PATTERN: List[Tuple[int, int]] = [
    (2, 0),
    (1, 1), (2, 1), (3, 1),
    (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
    (1, 3), (2, 3), (3, 3),
    (2, 4),
]


def _piece_safe(piece_pos: Tuple[int, int],
                wakneh_cells: Set[Tuple[int, int]]) -> bool:
    """Pixel-perfect piece-on-wakneh collision using r11l's
    diamond/plus pattern."""
    px, py = piece_pos
    if not (0 <= px <= GRID_SIZE - PIECE_W and 0 <= py <= GRID_SIZE - PIECE_H):
        return False
    for (dx, dy) in _PIECE_PATTERN:
        if (px + dx, py + dy) in wakneh_cells:
            return False
    return True


def _generate_destination_sets(
    n: int, target_x: int, target_y: int,
) -> List[List[Tuple[int, int]]]:
    """Mirror of `arcgym.adapters.r11l_canvas._generate_destination_sets`.

    Generate candidate piece destination sets for n pieces such that
    centroid → canvas at (target_x, target_y), with pairwise Chebyshev
    separation > 2 (5x5 pieces, click-center separation > 2 in at
    least one axis).
    """
    sets: List[List[Tuple[int, int]]] = []

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x <= GRID_SIZE - PIECE_W and 0 <= y <= GRID_SIZE - PIECE_H

    def pairwise_ok(positions: List[Tuple[int, int]]) -> bool:
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = abs(positions[i][0] - positions[j][0])
                dy = abs(positions[i][1] - positions[j][1])
                if dx <= 2 and dy <= 2:
                    return False
        return True

    if n == 2:
        for dy in (5, 6, 7, 8):
            ps = [(target_x, target_y - dy), (target_x, target_y + dy)]
            if all(in_bounds(*p) for p in ps) and pairwise_ok(ps):
                sets.append(ps)
        for dx in (5, 6, 7, 8):
            ps = [(target_x - dx, target_y), (target_x + dx, target_y)]
            if all(in_bounds(*p) for p in ps) and pairwise_ok(ps):
                sets.append(ps)
        for d in (4, 5, 6):
            ps = [(target_x - d, target_y - d), (target_x + d, target_y + d)]
            if all(in_bounds(*p) for p in ps) and pairwise_ok(ps):
                sets.append(ps)
            ps = [(target_x - d, target_y + d), (target_x + d, target_y - d)]
            if all(in_bounds(*p) for p in ps) and pairwise_ok(ps):
                sets.append(ps)
    elif n == 3:
        for dy in (4, 5, 6):
            ps = [(target_x, target_y - dy), (target_x, target_y),
                  (target_x, target_y + dy)]
            if all(in_bounds(*p) for p in ps) and pairwise_ok(ps):
                sets.append(ps)
        for dx in (4, 5, 6):
            ps = [(target_x - dx, target_y), (target_x, target_y),
                  (target_x + dx, target_y)]
            if all(in_bounds(*p) for p in ps) and pairwise_ok(ps):
                sets.append(ps)
        for d in (4, 5, 6):
            ps = [(target_x, target_y - d),
                  (target_x - d, target_y + d),
                  (target_x + d, target_y)]
            avg_x = sum(p[0] for p in ps) // 3
            avg_y = sum(p[1] for p in ps) // 3
            if (avg_x == target_x and avg_y == target_y and
                    all(in_bounds(*p) for p in ps) and pairwise_ok(ps)):
                sets.append(ps)
    elif n == 4:
        for d in (3, 4, 5, 6, 7):
            ps = [(target_x - d, target_y - d), (target_x + d, target_y - d),
                  (target_x - d, target_y + d), (target_x + d, target_y + d)]
            if all(in_bounds(*p) for p in ps) and pairwise_ok(ps):
                sets.append(ps)
        for dx in (3, 4, 5):
            for dy in (3, 4, 5):
                ps = [(target_x - dx, target_y - dy), (target_x + dx, target_y - dy),
                      (target_x - dx, target_y + dy), (target_x + dx, target_y + dy)]
                if all(in_bounds(*p) for p in ps) and pairwise_ok(ps):
                    sets.append(ps)
        for d in (3, 4, 5):
            ps = [(target_x, target_y - 3 * d), (target_x, target_y - d),
                  (target_x, target_y + d), (target_x, target_y + 3 * d)]
            if all(in_bounds(*p) for p in ps) and pairwise_ok(ps):
                sets.append(ps)
            ps = [(target_x - 3 * d, target_y), (target_x - d, target_y),
                  (target_x + d, target_y), (target_x + 3 * d, target_y)]
            if all(in_bounds(*p) for p in ps) and pairwise_ok(ps):
                sets.append(ps)

    # Final centroid-exact filter.
    valid_sets = []
    for ps in sets:
        sx = sum(p[0] + PIECE_W // 2 for p in ps)
        sy = sum(p[1] + PIECE_H // 2 for p in ps)
        cx = sx // n - CANVAS_W // 2
        cy = sy // n - CANVAS_H // 2
        if cx == target_x and cy == target_y:
            valid_sets.append(ps)
    return valid_sets


def _find_safe_permutation(
    pieces: List[Tuple[int, int]],
    destinations: List[Tuple[int, int]],
    defgjl_cells: Set[Tuple[int, int]],
    wakneh_cells: Set[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    """Find a permutation of destinations such that:
      - all intermediate canvas positions are NOT in defgjl_cells, AND
      - all destinations are NOT in wakneh_cells.

    Returns destinations in the same order as `pieces` (pieces[i] →
    result[i]), or None.
    """
    n = len(pieces)
    init_positions = list(pieces)

    for perm in permutations(range(n)):
        ok = True
        for k in range(1, n + 1):
            sum_x = sum(destinations[perm[i]][0] + PIECE_W // 2 for i in range(k))
            sum_x += sum(init_positions[i][0] + PIECE_W // 2 for i in range(k, n))
            sum_y = sum(destinations[perm[i]][1] + PIECE_H // 2 for i in range(k))
            sum_y += sum(init_positions[i][1] + PIECE_H // 2 for i in range(k, n))
            cx = sum_x // n - CANVAS_W // 2
            cy = sum_y // n - CANVAS_H // 2
            if not _canvas_safe((cx, cy), defgjl_cells):
                ok = False
                break
            dest = destinations[perm[k - 1]]
            if not _piece_safe(dest, wakneh_cells):
                ok = False
                break
        if ok:
            return [destinations[perm[i]] for i in range(n)]
    return None


def _bfs_canvas_path(
    pieces: List[Tuple[int, int]],
    target_x: int, target_y: int,
    wakneh_cells: Set[Tuple[int, int]],
    defgjl_cells: Set[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    """BFS over (move_count, S_x, S_y) state — port of
    `arcgym.adapters.r11l_canvas._solve_canvas_path` using cell sets
    instead of safe_grid arrays. Returns piece destinations in input
    order, or None.
    """
    n = len(pieces)
    if n == 0:
        return None
    init_S_x = sum(p[0] + PIECE_W // 2 for p in pieces)
    init_S_y = sum(p[1] + PIECE_H // 2 for p in pieces)
    target_S_x_min = n * (target_x + CANVAS_W // 2)
    target_S_x_max = target_S_x_min + n - 1
    target_S_y_min = n * (target_y + CANVAS_H // 2)
    target_S_y_max = target_S_y_min + n - 1

    def canvas_safe_S(S_x: int, S_y: int) -> bool:
        cx = S_x // n - CANVAS_W // 2
        cy = S_y // n - CANVAS_H // 2
        return _canvas_safe((cx, cy), defgjl_cells)

    if not canvas_safe_S(init_S_x, init_S_y):
        return None

    visited: dict = {}
    visited[(0, init_S_x, init_S_y)] = None
    queue: deque = deque([(0, init_S_x, init_S_y)])
    goal = None
    while queue:
        k, S_x, S_y = queue.popleft()
        if k == n:
            if (target_S_x_min <= S_x <= target_S_x_max and
                    target_S_y_min <= S_y <= target_S_y_max):
                goal = (k, S_x, S_y)
                break
            continue
        oldp = pieces[k]
        old_center_x = oldp[0] + PIECE_W // 2
        old_center_y = oldp[1] + PIECE_H // 2
        if k == n - 1:
            # Last piece: derive directly from target constraint.
            for tgt_S_x in range(target_S_x_min, target_S_x_max + 1):
                newp_x = tgt_S_x - (S_x - old_center_x) - PIECE_W // 2
                if not (0 <= newp_x <= GRID_SIZE - PIECE_W):
                    continue
                for tgt_S_y in range(target_S_y_min, target_S_y_max + 1):
                    newp_y = tgt_S_y - (S_y - old_center_y) - PIECE_H // 2
                    if not (0 <= newp_y <= GRID_SIZE - PIECE_H):
                        continue
                    if not _piece_safe((newp_x, newp_y), wakneh_cells):
                        continue
                    new_state = (k + 1, tgt_S_x, tgt_S_y)
                    if new_state not in visited:
                        visited[new_state] = ((k, S_x, S_y), (newp_x, newp_y))
                        queue.append(new_state)
        else:
            for newp_x in range(0, GRID_SIZE - PIECE_W + 1):
                new_center_x = newp_x + PIECE_W // 2
                new_S_x = S_x - old_center_x + new_center_x
                cx = new_S_x // n - CANVAS_W // 2
                if not (0 <= cx <= GRID_SIZE - CANVAS_W):
                    continue
                for newp_y in range(0, GRID_SIZE - PIECE_H + 1):
                    if not _piece_safe((newp_x, newp_y), wakneh_cells):
                        continue
                    new_center_y = newp_y + PIECE_H // 2
                    new_S_y = S_y - old_center_y + new_center_y
                    if not canvas_safe_S(new_S_x, new_S_y):
                        continue
                    new_state = (k + 1, new_S_x, new_S_y)
                    if new_state in visited:
                        continue
                    visited[new_state] = ((k, S_x, S_y), (newp_x, newp_y))
                    queue.append(new_state)

    if goal is None:
        return None

    path: List[Tuple[int, int]] = []
    state = goal
    while state is not None:
        entry = visited.get(state)
        if entry is None:
            break
        parent, newp = entry
        path.append(newp)
        state = parent
    path.reverse()
    if len(path) != n:
        return None
    return path


def _solve_group(pieces: List[Tuple[int, int]],
                 target: Tuple[int, int],
                 wakneh_cells: Set[Tuple[int, int]],
                 defgjl_cells: Set[Tuple[int, int]],
                 obstacle_pieces: Optional[List[Tuple[int, int]]] = None,
                 ) -> Optional[List[Tuple[int, int]]]:
    """Try structured destination sets + safe permutation search,
    fall back to free BFS if no structured set works.
    `obstacle_pieces` are other groups' pieces whose bboxes should not
    contain a destination's click point (else click would re-select
    that piece instead of moving the active piece).
    Returns destinations in input order, or None.
    """
    n = len(pieces)
    if n < 2:
        return None
    obstacles = list(obstacle_pieces or [])

    def dest_clear_of_obstacles(dests: List[Tuple[int, int]]) -> bool:
        for d in dests:
            click = (d[0] + PIECE_W // 2, d[1] + PIECE_H // 2)
            for op in obstacles:
                if op in dests:
                    continue
                if op[0] <= click[0] < op[0] + PIECE_W and \
                   op[1] <= click[1] < op[1] + PIECE_H:
                    return False
        return True

    for dest_set in _generate_destination_sets(n, target[0], target[1]):
        result = _find_safe_permutation(pieces, dest_set, defgjl_cells, wakneh_cells)
        if result is not None and dest_clear_of_obstacles(result):
            return result
    # Fall back: free BFS.
    bfs = _bfs_canvas_path(pieces, target[0], target[1], wakneh_cells, defgjl_cells)
    if bfs is not None and dest_clear_of_obstacles(bfs):
        return bfs
    return None


def _build_click_sequence(
    initial_active: Tuple[int, int],
    pieces_to_move: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    all_pieces: List[Tuple[int, int]],
) -> Optional[List[Tuple[GameAction, dict]]]:
    """Generate ACTION6 click sequence: (select)? + move per piece.

    pieces_to_move: list of (current_pos, dest) tuples in EXECUTION
    order. The first piece's `current_pos` should equal `initial_active`
    (skip select for it).

    all_pieces: current positions of ALL pieces in scene (used to check
    click is not inside any piece bbox for moves; or hits the right
    piece for selects).

    Returns plan or None if any click is invalid.
    """
    plan: List[Tuple[GameAction, dict]] = []
    # Live state of all piece positions (mutable).
    pieces_state: List[Tuple[int, int]] = list(all_pieces)
    active = initial_active

    for piece_pos, dest in pieces_to_move:
        # Find piece in state.
        try:
            idx = pieces_state.index(piece_pos)
        except ValueError:
            return None
        # Select if not already active.
        if pieces_state[idx] != active:
            sel_x, sel_y = _piece_center(pieces_state[idx])
            # Click must hit THIS piece (i.e., this piece's bbox is the
            # FIRST piece in iteration order containing the click).
            # Since bbi sort is fixed per level by initial position, we
            # don't re-sort. Verify click hits any piece, but that piece
            # might not be the intended one. We approximate: if no other
            # piece's bbox contains the click, OK.
            hit = None
            for i, pp in enumerate(pieces_state):
                if pp[0] <= sel_x < pp[0] + PIECE_W and pp[1] <= sel_y < pp[1] + PIECE_H:
                    if hit is None:
                        hit = i
                    else:
                        # Multiple pieces overlap click — ambiguous.
                        # The game's iteration uses a fixed bbi order;
                        # we hope the right one is FIRST in iteration.
                        pass
            if hit != idx:
                # Selection click won't pick the right piece.
                return None
            plan.append((GameAction.ACTION6, {"x": sel_x, "y": sel_y}))
            active = pieces_state[idx]
        # Move click: must NOT be inside any piece bbox.
        move_x, move_y = dest[0] + PIECE_W // 2, dest[1] + PIECE_H // 2
        if _click_inside_pieces((move_x, move_y), pieces_state):
            return None
        plan.append((GameAction.ACTION6, {"x": move_x, "y": move_y}))
        # Update state.
        pieces_state[idx] = dest
        active = dest

    return plan


_ABSORBER_SPECS = {
    "r11l_l5": {
        "pieces": [(23, 33), (41, 32), (32, 53), (39, 45), (50, 53)],
        "active_index": 0,
        "groups": [
            {
                "piece_indices": [0, 1],
                "waypoints": [((31, 17), (5, 5)), ((12, 38), (5, 5)), ((10, 25), (7, 7))],
                "avoid": [((54, 41), (5, 5)), ((18, 51), (5, 5))],
            },
            {
                "piece_indices": [2, 3, 4],
                "waypoints": [((54, 41), (5, 5)), ((18, 51), (5, 5)), ((45, 6), (7, 7))],
                "avoid": [((31, 17), (5, 5)), ((12, 38), (5, 5))],
            },
        ],
    },
    "r11l_l6": {
        # bbijaigbknc starts with whkxtx-2 piece B, then A/C, then whkxtx.
        "pieces": [(9, 9), (2, 17), (20, 17), (47, 41), (48, 55)],
        "active_index": 0,
        "groups": [
            {
                "piece_indices": [0, 1, 2],
                "waypoints": [
                    ((9, 31), (5, 5)),
                    ((32, 15), (5, 5)),
                    ((14, 43), (5, 5)),
                    ((48, 9), (7, 7)),
                ],
                "avoid": [
                    ((43, 28), (5, 5)), ((23, 53), (5, 5)), ((30, 42), (5, 5)),
                    ((37, 4), (5, 5)), ((54, 21), (5, 5)), ((36, 55), (5, 5)),
                ],
            },
            {
                "piece_indices": [3, 4],
                "waypoints": [
                    ((43, 28), (5, 5)),
                    ((30, 42), (5, 5)),
                    ((23, 53), (5, 5)),
                    ((8, 52), (7, 7)),
                ],
                "avoid": [
                    ((9, 31), (5, 5)), ((32, 15), (5, 5)), ((14, 43), (5, 5)),
                    ((37, 4), (5, 5)), ((54, 21), (5, 5)), ((36, 55), (5, 5)),
                ],
            },
        ],
    },
}


def _rects_overlap(pos_a: Tuple[int, int], size_a: Tuple[int, int],
                   pos_b: Tuple[int, int], size_b: Tuple[int, int]) -> bool:
    ax, ay = pos_a
    aw, ah = size_a
    bx, by = pos_b
    bw, bh = size_b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _canvas_from_positions(positions: List[Tuple[int, int]]) -> Tuple[int, int]:
    n = len(positions)
    sx = sum(x + PIECE_W // 2 for x, _ in positions)
    sy = sum(y + PIECE_H // 2 for _, y in positions)
    return (sx // n - CANVAS_W // 2, sy // n - CANVAS_H // 2)


def _click_inside_piece_index(click: Tuple[int, int],
                              pieces_state: List[Tuple[int, int]]) -> Optional[int]:
    cx, cy = click
    for i, (px, py) in enumerate(pieces_state):
        if px <= cx < px + PIECE_W and py <= cy < py + PIECE_H:
            return i
    return None


def _absorber_candidate_sets(
    n: int,
    canvas_pos: Tuple[int, int],
) -> List[List[Tuple[int, int]]]:
    cx, cy = canvas_pos
    candidates: List[List[Tuple[int, int]]] = []

    def in_bounds(pos: Tuple[int, int]) -> bool:
        x, y = pos
        # r11l allows clicks near the 64x64 display edge; keep the top-left
        # inside the rendered board rather than the older 60x60 helper grid.
        return 0 <= x <= 59 and 0 <= y <= 59

    def add(points: List[Tuple[int, int]]) -> None:
        if all(in_bounds(p) for p in points) and points not in candidates:
            candidates.append(points)

    if n == 2:
        for d in (2, 3, 4, 5, 6, 7, 8):
            add([(cx - d, cy), (cx + d, cy)])
            add([(cx, cy - d), (cx, cy + d)])
        for d in (3, 4, 5, 6):
            add([(cx - d, cy - d), (cx + d, cy + d)])
            add([(cx - d, cy + d), (cx + d, cy - d)])
    elif n == 3:
        for d in (4, 5, 6, 7):
            add([(cx - d, cy), (cx, cy), (cx + d, cy)])
            add([(cx, cy - d), (cx, cy), (cx, cy + d)])
        for d in (4, 5, 6):
            add([(cx - d, cy), (cx + d, cy), (cx, cy)])
            add([(cx, cy - d), (cx, cy + d), (cx, cy)])
    return candidates


def _collides_any(canvas_pos: Tuple[int, int],
                  bboxes: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> bool:
    return any(_rects_overlap(canvas_pos, (CANVAS_W, CANVAS_H), pos, size)
               for pos, size in bboxes)


def _try_absorber_moves(
    pieces_state: List[Tuple[int, int]],
    active_index: int,
    group_indices: List[int],
    dest_by_index: dict[int, Tuple[int, int]],
    avoid: List[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> Optional[Tuple[List[Tuple[GameAction, dict]], List[Tuple[int, int]], int]]:
    from itertools import permutations

    ordered_indices = [active_index] if active_index in group_indices else []
    ordered_indices.extend(i for i in group_indices if i != active_index)
    best_orders = [ordered_indices]
    for perm in permutations(group_indices):
        p = list(perm)
        if p not in best_orders:
            best_orders.append(p)

    for move_order in best_orders:
        local_pieces = list(pieces_state)
        local_active = active_index
        plan: List[Tuple[GameAction, dict]] = []
        ok = True
        for idx in move_order:
            cur = local_pieces[idx]
            if idx != local_active:
                sel = _piece_center(cur)
                if _click_inside_piece_index(sel, local_pieces) != idx:
                    ok = False
                    break
                plan.append((GameAction.ACTION6, {"x": sel[0], "y": sel[1]}))
                local_active = idx
            dest = dest_by_index[idx]
            move = _piece_center(dest)
            if _click_inside_piece_index(move, local_pieces) is not None:
                ok = False
                break
            local_pieces[idx] = dest
            canvas_pos = _canvas_from_positions([local_pieces[i] for i in group_indices])
            if _collides_any(canvas_pos, avoid):
                ok = False
                break
            plan.append((GameAction.ACTION6, {"x": move[0], "y": move[1]}))
        if ok:
            return plan, local_pieces, local_active
    return None


def _solve_absorber_signature(sig: str) -> List[Tuple[GameAction, dict]]:
    spec = _ABSORBER_SPECS.get(sig)
    if not spec:
        return []
    from itertools import permutations

    pieces_state = list(spec["pieces"])
    active_index = int(spec["active_index"])
    full_plan: List[Tuple[GameAction, dict]] = []

    for group in spec["groups"]:
        group_indices = list(group["piece_indices"])
        avoid = list(group.get("avoid", []))
        for waypoint_pos, waypoint_size in group["waypoints"]:
            canvas_targets: List[Tuple[int, int]] = [waypoint_pos]
            wx, wy = waypoint_pos
            ww, wh = waypoint_size
            for cy in range(max(0, wy - CANVAS_H + 1), min(59, wy + wh - 1) + 1):
                for cx in range(max(0, wx - CANVAS_W + 1), min(59, wx + ww - 1) + 1):
                    cpos = (cx, cy)
                    if cpos != waypoint_pos:
                        canvas_targets.append(cpos)
            canvas_targets.sort(key=lambda p: abs(p[0] - wx) + abs(p[1] - wy))

            solved = None
            for canvas_pos in canvas_targets:
                if _collides_any(canvas_pos, avoid):
                    continue
                if not _rects_overlap(canvas_pos, (CANVAS_W, CANVAS_H), waypoint_pos, waypoint_size):
                    continue
                for dest_set in _absorber_candidate_sets(len(group_indices), canvas_pos):
                    for perm in permutations(range(len(dest_set))):
                        dest_by_index = {
                            idx: dest_set[perm[i]]
                            for i, idx in enumerate(group_indices)
                        }
                        result = _try_absorber_moves(
                            pieces_state, active_index, group_indices, dest_by_index, avoid,
                        )
                        if result is not None:
                            solved = result
                            break
                    if solved is not None:
                        break
                if solved is not None:
                    break
            if solved is None:
                return []
            step_plan, pieces_state, active_index = solved
            final_canvas = _canvas_from_positions([pieces_state[i] for i in group_indices])
            if not _rects_overlap(final_canvas, (CANVAS_W, CANVAS_H), waypoint_pos, waypoint_size):
                return []
            full_plan.extend(step_plan)
    return full_plan


def plan_next_actions_from_state(state: dict) -> List[Tuple[GameAction, dict]]:
    """Plan win clicks for r11l from perceived state.

    Supports L1 (single group) and L2+ multi-group via per-color
    matching (canvas color = target color = piece color), plus L5/L6
    via signature-selected absorber specs and runtime centroid solving.
    """
    sig = state.get("level_signature")
    if sig in _ABSORBER_SPECS:
        plan = _solve_absorber_signature(str(sig))
        _record_plan_source("tier_a_absorber_solver" if plan else "unsolved")
        return plan

    canvases = state.get("canvases", [])
    targets = state.get("targets", [])
    _remember_visible_targets(targets)
    targets = _augment_targets_from_cache(canvases, targets)
    active = state.get("active_piece")
    inactive_list = state.get("inactive_pieces", [])
    wakneh_cells = state.get("wakneh_cells", set())
    defgjl_cells = state.get("defgjl_cells", set())

    if not canvases or not targets or active is None:
        _record_plan_source("unsolved")
        return []
    active_color = state.get("active_piece_color")
    if active_color is None:
        _record_plan_source("unsolved")
        return []
    _clear_piece_cache_on_scene_reset(active, active_color, inactive_list)
    inactive_list = _augment_hidden_pieces_from_cache(canvases, active, active_color, inactive_list)
    _remember_visible_pieces(active, active_color, inactive_list)

    # Build (pos, color) list for all pieces.
    all_pieces: List[Tuple[Tuple[int, int], int]] = [(active, active_color)]
    for p in inactive_list:
        c = p.get("color")
        if c is not None:
            all_pieces.append((p["pos"], c))

    all_pos = [p for (p, _) in all_pieces]

    # For each canvas, find matching target and pieces.
    plan: List[Tuple[GameAction, dict]] = []
    initial_active = active
    pieces_state = list(all_pos)

    # Process group containing the active piece FIRST (saves a select click).
    canvas_order: List[dict] = []
    for canvas in canvases:
        if canvas["color"] == active_color:
            canvas_order.insert(0, canvas)
        else:
            canvas_order.append(canvas)

    for canvas in canvas_order:
        c_color = canvas["color"]
        piece_color = canvas.get("piece_color", c_color)
        canvas_colors = canvas.get("colors", {c_color})
        # Match target by best color overlap with canvas. Both canvas
        # and target may have multiple colors (multi-morpheme groups).
        # Prefer target with maximum |canvas_colors ∩ target_colors|.
        best_target = None
        best_score = (0, 0, 0)
        for t in targets:
            t_cols = t.get("colors", {t["color"]})
            overlap = len(canvas_colors & t_cols)
            score = (
                overlap,
                1 if c_color in t_cols else 0,
                1 if piece_color in t_cols else 0,
            )
            if score > best_score:
                best_score = score
                best_target = t
        if best_target is None:
            _record_plan_source("unsolved")
            return []
        target_colors = best_target.get("colors", {best_target["color"]})
        # Pieces matched to this canvas: prefer the canvas's
        # "piece_color" hint (last-morpheme color extracted from
        # canvas bottom row). For multi-morpheme groups (e.g., L4
        # orrqljliwocqblxuubpumlzd), this disambiguates which pieces
        # belong here when colors are shared across canvases.
        group_pieces = [pos for (pos, c) in all_pieces if c == piece_color]
        if len(group_pieces) < 2:
            # Fallback: use any color in canvas∩target overlap.
            match_colors = canvas_colors & target_colors
            group_pieces = [pos for (pos, c) in all_pieces
                            if c in match_colors]
            if len(group_pieces) < 2:
                _record_plan_source("unsolved")
                return []
        # Other groups' pieces are obstacles for click destinations.
        obstacle_pieces = [p for p in pieces_state if p not in group_pieces]
        match_target = best_target
        # Sort pieces by Euclidean distance from origin (mirrors the
        # game's bbi sort in `on_set_level`). Movement order affects
        # intermediate canvas trajectory and thus solver feasibility.
        group_pieces = sorted(group_pieces,
                              key=lambda p: (p[0] ** 2 + p[1] ** 2) ** 0.5)
        dests = _solve_group(group_pieces, match_target["pos"],
                             wakneh_cells, defgjl_cells,
                             obstacle_pieces=obstacle_pieces)
        if dests is None:
            _record_plan_source("unsolved")
            return []
        # Order pieces such that the active piece (if in this group) is
        # processed first.
        ordered: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        if any(initial_active == p for p in group_pieces):
            for i, p in enumerate(group_pieces):
                if p == initial_active:
                    ordered.append((p, dests[i]))
                    break
            for i, p in enumerate(group_pieces):
                if p != initial_active:
                    ordered.append((p, dests[i]))
        else:
            ordered = list(zip(group_pieces, dests))
        ordered = [
            (piece, dest)
            for piece, dest in ordered
            if piece != dest and not _destination_click_inside_piece(piece, dest)
        ]

        sub_plan = _build_click_sequence(initial_active, ordered, pieces_state)
        if sub_plan is None:
            _record_plan_source("unsolved")
            return []
        plan.extend(sub_plan)
        # Update pieces_state for next group.
        for piece_pos, dest in ordered:
            try:
                idx = pieces_state.index(piece_pos)
                pieces_state[idx] = dest
            except ValueError:
                pass
        # Active piece is now the last moved piece.
        if ordered:
            initial_active = ordered[-1][1]

    _record_plan_source("tier_a_solver" if plan else "unsolved")
    return plan


def perceive_and_plan(frame: Any) -> List[Tuple[GameAction, dict]]:
    state = perceive_r11l(frame)
    key = _continuation_key(state)
    queued = _take_plan_queue(key)
    if queued is not None:
        return queued
    plan = plan_next_actions_from_state(state)
    if plan:
        _store_plan_queue(key, plan)
    else:
        _clear_plan_queue()
    return plan
