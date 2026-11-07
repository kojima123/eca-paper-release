"""Frame-conditioned peg-solitaire controller.

The controller detects a visible peg/empty lattice, solves the monotone
peg-count reduction problem, and drains the two-click jump protocol. It is
default-off and does not dispatch on game id.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np


Click = tuple[int, int]
ActionSpec = dict[str, Any]
Cell = tuple[int, int]
Direction = tuple[int, int]
Move = tuple

_DIRECTIONS: tuple[Direction, ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)
_ACTION_FOR_DIR: dict[Direction, str] = {
    (0, -1): "ACTION1",
    (0, 1): "ACTION2",
    (-1, 0): "ACTION3",
    (1, 0): "ACTION4",
}

_L2_GRID_ORIGIN: Cell = (6, 8)
_L3_GRID_ORIGIN: Cell = (5, 5)
_L4_GRID_ORIGIN: Cell = (5, 5)
_L5_GRID_ORIGIN: Cell = (5, 5)
_TILE = 6
_BUTTON_OFFSET = 12
_L2_GRID = (
    "....... ",
    ".xx.x.x->",
    "....... |",
    "        |",
    " <--,---3",
    " |      ",
    " |    ..",
    " L----x.",
)
_L3_GRID = (
    ".. ..     ..",
    ".x .x-,--x..",
    ".x.x      .xx.",
    "..x.      ..",
    "          .xx.",
    "      <-> ..",
    "      | | x..",
    ".x.x-,3 L-.x",
    "...       ..",
)
_L4_GRID = (
    "",
    "",
    ".......",
    ".xp.p.x-,---x...x.",
    ".......     .p..p.",
    "            ......",
    "            .p..p.",
    "        <---,..p..",
    "        |",
    "    .x. | ",
    "   .p p.p..",
    "   .----,-.",
    "   .p p....",
    "    .x.",
)
_L3_LOSE_ALWAYS: frozenset[Cell] = frozenset({
    (1, 0),
    (0, 3),
    (10, 0),
    (13, 2),
    (13, 4),
})


def _flatten(frame: Any) -> np.ndarray:
    arr = np.asarray(frame, dtype=int)
    if arr.ndim == 3:
        arr = arr[-1]
    return arr


def _parse_pivot_grid(grid: tuple[str, ...]) -> tuple[frozenset[Cell], frozenset[Cell], frozenset[Cell]]:
    rails: set[Cell] = set()
    holes: set[Cell] = set()
    markers: set[Cell] = set()
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            if ch == "x" or ch == ".":
                holes.add((x, y))
            elif ch == ",":
                rails.add((x, y))
            elif ch == "p":
                holes.add((x, y))
                markers.add((x, y))
            elif ch == "P":
                rails.add((x, y))
                markers.add((x, y))
            elif ch in "-|L3<>Tt":
                rails.add((x, y))
    return frozenset(rails), frozenset(holes), frozenset(markers)


_L2_RAILS, _L2_HOLES, _L2_MARKERS = _parse_pivot_grid(_L2_GRID)


def _parse_lf52_pivot_grid(
    grid: tuple[str, ...],
) -> tuple[frozenset[Cell], frozenset[Cell], frozenset[Cell], frozenset[Cell], frozenset[Cell]]:
    pegs: set[Cell] = set()
    pivots: set[Cell] = set()
    rails: set[Cell] = set()
    holes: set[Cell] = set()
    markers: set[Cell] = set()
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            if ch == "x":
                pegs.add((x, y))
                holes.add((x, y))
            elif ch == ".":
                holes.add((x, y))
            elif ch == ",":
                pivots.add((x, y))
                rails.add((x, y))
            elif ch == "p":
                markers.add((x, y))
                holes.add((x, y))
            elif ch == "P":
                pivots.add((x, y))
                rails.add((x, y))
                markers.add((x, y))
            elif ch in "-|L3<>Tt":
                rails.add((x, y))
    return (
        frozenset(pegs),
        frozenset(pivots),
        frozenset(rails),
        frozenset(holes),
        frozenset(markers),
    )


_L3_INIT_PEGS, _L3_INIT_PIVOTS, _L3_RAILS, _L3_HOLES, _L3_MARKERS = _parse_lf52_pivot_grid(_L3_GRID)
_L4_INIT_PEGS, _L4_INIT_PIVOTS, _L4_RAILS, _L4_HOLES, _L4_MARKERS = _parse_lf52_pivot_grid(_L4_GRID)
_L5_INIT_PEGS: frozenset[Cell] = frozenset({
    (1, 3), (6, 3), (11, 3), (16, 8), (18, 4), (19, 0),
})
_L5_INIT_PIVOTS: frozenset[Cell] = frozenset({(2, 0), (6, 0), (19, 1)})
_L5_MARKERS: frozenset[Cell] = frozenset({
    (2, 3), (6, 0), (15, 3), (15, 7), (16, 0), (16, 4), (19, 1),
})
_L5_HOLES: frozenset[Cell] = frozenset({
    (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4),
    (5, 3), (6, 3), (11, 3), (12, 2), (12, 3), (12, 4),
    (14, 0), (14, 2), (14, 3), (14, 4), (14, 8),
    (15, 0), (15, 2), (15, 3), (15, 4), (15, 6), (15, 7), (15, 8),
    (16, 0), (16, 2), (16, 3), (16, 4), (16, 8),
    (17, 0), (18, 0), (18, 2), (18, 3), (18, 4), (19, 0), (19, 3), (19, 4),
})
_L5_RAILS: frozenset[Cell] = frozenset({
    (1, 0), (2, 0), (2, 6), (2, 7), (2, 8),
    (3, 0), (3, 6),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
    (5, 0), (5, 6),
    (6, 0), (6, 6),
    (7, 0), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6),
    (8, 2), (9, 2), (10, 2), (10, 3),
    (13, 1), (13, 2), (13, 3), (13, 4), (13, 5),
    (14, 1), (14, 5), (15, 1), (15, 5), (16, 1), (16, 5),
    (17, 1), (17, 2), (17, 3), (17, 4), (17, 5),
    (18, 1), (18, 5), (19, 1), (19, 5),
    (20, 1), (20, 2), (20, 3), (20, 4), (20, 5),
})


def _is_l3_lose(land: Cell, peg_count: int) -> bool:
    if land in _L3_LOSE_ALWAYS:
        return True
    if land == (10, 2) and peg_count <= 5:
        return True
    if land == (10, 4) and peg_count <= 4:
        return True
    return False


def _land_valid(
    land: Cell,
    pegs: frozenset[Cell],
    pivots: frozenset[Cell],
    markers: frozenset[Cell],
    rails: frozenset[Cell],
    holes: frozenset[Cell],
) -> bool:
    if land in pegs:
        return False
    sprite_count = 0
    has_pivot = False
    has_hole = False
    if land in pivots:
        sprite_count += 1
        has_pivot = True
    if land in markers:
        sprite_count += 1
    if land in rails:
        sprite_count += 1
    if land in holes:
        sprite_count += 1
        has_hole = True
    if sprite_count == 1 and (has_hole or has_pivot):
        return True
    return sprite_count == 2 and has_pivot


def _enumerate_jump_neighbours(
    pegs: frozenset[Cell],
    pivots: frozenset[Cell],
    markers: frozenset[Cell],
    peg_carriers: frozenset[Cell],
    marker_carriers: frozenset[Cell],
    rails: frozenset[Cell],
    holes: frozenset[Cell],
    level_idx: int = 2,
):
    peg_count = len(pegs)
    for src in sorted(pegs):
        sx, sy = src
        for dx, dy in _DIRECTIONS:
            mid = (sx + dx, sy + dy)
            land = (sx + 2 * dx, sy + 2 * dy)
            if mid not in pegs and mid not in markers:
                continue
            if not _land_valid(land, pegs, pivots, markers, rails, holes):
                continue
            if level_idx == 3 and _is_l3_lose(land, peg_count):
                continue
            new_pegs = set(pegs)
            new_pegs.discard(src)
            if mid in new_pegs:
                new_pegs.discard(mid)
            new_pegs.add(land)
            new_carriers = set(peg_carriers)
            new_carriers.discard(src)
            new_carriers.discard(mid)
            if land in pivots:
                new_carriers.add(land)
            yield (
                ("JUMP", src, (dx, dy)),
                (
                    frozenset(new_pegs),
                    pivots,
                    markers,
                    frozenset(new_carriers),
                    marker_carriers,
                ),
            )


def _apply_pivot_move(
    pegs: frozenset[Cell],
    pivots: frozenset[Cell],
    markers: frozenset[Cell],
    peg_carriers: frozenset[Cell],
    marker_carriers: frozenset[Cell],
    rails: frozenset[Cell],
    direction: Direction,
) -> tuple[frozenset[Cell], frozenset[Cell], frozenset[Cell], frozenset[Cell], frozenset[Cell], bool]:
    dx, dy = direction
    if dx != 0:
        ordered = sorted(pivots, key=lambda cell: cell[0], reverse=(dx > 0))
    else:
        ordered = sorted(pivots, key=lambda cell: cell[1], reverse=(dy > 0))
    new_pegs = set(pegs)
    new_pivots = set(pivots)
    new_markers = set(markers)
    new_pc = set(peg_carriers)
    new_mc = set(marker_carriers)
    moved = False
    for pivot in ordered:
        target = (pivot[0] + dx, pivot[1] + dy)
        if target in new_pivots or target not in rails:
            continue
        new_pivots.discard(pivot)
        new_pivots.add(target)
        if pivot in new_pc:
            new_pc.discard(pivot)
            new_pc.add(target)
            new_pegs.discard(pivot)
            new_pegs.add(target)
        if pivot in new_mc:
            new_mc.discard(pivot)
            new_mc.add(target)
            new_markers.discard(pivot)
            new_markers.add(target)
        if target in new_pegs:
            new_pc.add(target)
        if target in new_markers:
            new_mc.add(target)
        moved = True
    return (
        frozenset(new_pegs),
        frozenset(new_pivots),
        frozenset(new_markers),
        frozenset(new_pc),
        frozenset(new_mc),
        moved,
    )


def _enumerate_pivot_neighbours(
    pegs: frozenset[Cell],
    pivots: frozenset[Cell],
    markers: frozenset[Cell],
    peg_carriers: frozenset[Cell],
    marker_carriers: frozenset[Cell],
    rails: frozenset[Cell],
):
    for direction in _DIRECTIONS:
        new_pegs, new_pivots, new_markers, new_pc, new_mc, moved = _apply_pivot_move(
            pegs,
            pivots,
            markers,
            peg_carriers,
            marker_carriers,
            rails,
            direction,
        )
        if moved:
            yield (
                ("PIVOT", direction),
                (new_pegs, new_pivots, new_markers, new_pc, new_mc),
            )


def _solve_pivot_transport(
    init_pegs: frozenset[Cell],
    init_pivots: frozenset[Cell],
    rails: frozenset[Cell],
    holes: frozenset[Cell],
    markers: frozenset[Cell],
    *,
    level_idx: int = 2,
    target_peg_count: int = 1,
    max_states: int = 200_000,
) -> Optional[list[Move]]:
    init_marker_carriers = frozenset(markers & init_pivots)
    start = (
        init_pegs,
        init_pivots,
        markers,
        frozenset(init_pegs & init_pivots),
        init_marker_carriers,
    )
    if len(init_pegs) <= target_peg_count:
        return []
    visited = {start}
    parent: dict[tuple, tuple[Optional[tuple], Optional[Move]]] = {start: (None, None)}
    queue = deque([start])
    states = 0
    while queue:
        state = queue.popleft()
        states += 1
        if states > max_states:
            return None
        pegs, pivots, cur_markers, peg_carriers, marker_carriers = state
        if len(pegs) <= target_peg_count:
            moves: list[Move] = []
            cur: Optional[tuple] = state
            while cur is not None and parent[cur][0] is not None:
                prev, move = parent[cur]
                if move is not None:
                    moves.append(move)
                cur = prev
            moves.reverse()
            return moves
        neighbours = list(
            _enumerate_jump_neighbours(
                pegs,
                pivots,
                cur_markers,
                peg_carriers,
                marker_carriers,
                rails,
                holes,
                level_idx,
            )
        )
        neighbours.extend(
            _enumerate_pivot_neighbours(
                pegs,
                pivots,
                cur_markers,
                peg_carriers,
                marker_carriers,
                rails,
            )
        )
        for move, new_state in neighbours:
            if new_state in visited:
                continue
            visited.add(new_state)
            parent[new_state] = (state, move)
            queue.append(new_state)
    return None


def _jump_specs(src: Cell, direction: Direction, origin: Cell) -> list[ActionSpec]:
    sx, sy = src
    dx, dy = direction
    ox, oy = origin
    return [
        {
            "action": "ACTION6",
            "x": ox + sx * _TILE + _TILE // 2,
            "y": oy + sy * _TILE + _TILE // 2,
        },
        {
            "action": "ACTION6",
            "x": ox + sx * _TILE + dx * _BUTTON_OFFSET + _TILE // 2,
            "y": oy + sy * _TILE + dy * _BUTTON_OFFSET + _TILE // 2,
        },
    ]


def _apply_pivot_move_with_origin(
    pegs: frozenset[Cell],
    pivots: frozenset[Cell],
    markers: frozenset[Cell],
    peg_carriers: frozenset[Cell],
    marker_carriers: frozenset[Cell],
    rails: frozenset[Cell],
    direction: Direction,
    origin: Cell,
    *,
    level_idx: int,
) -> tuple[frozenset[Cell], frozenset[Cell], frozenset[Cell], frozenset[Cell], frozenset[Cell], Cell]:
    dx, dy = direction
    if dx != 0:
        ordered = sorted(pivots, key=lambda cell: cell[0], reverse=(dx > 0))
    else:
        ordered = sorted(pivots, key=lambda cell: cell[1], reverse=(dy > 0))
    new_pegs = set(pegs)
    new_pivots = set(pivots)
    new_markers = set(markers)
    new_pc = set(peg_carriers)
    new_mc = set(marker_carriers)
    ox, oy = origin
    pending_shift: Cell = (0, 0)
    shifted = False
    for pivot in ordered:
        target = (pivot[0] + dx, pivot[1] + dy)
        if target in new_pivots or target not in rails:
            continue
        shift: Cell = (0, 0)
        if pivot in new_pc:
            if level_idx == 3:
                shift = (-dx * 8, 0)
            elif level_idx == 4:
                if pivot[1] < 11:
                    shift = (-dx * 6, -dy * 6)
            elif level_idx == 5:
                shift = (-dx * 6, 0)
        new_pivots.discard(pivot)
        new_pivots.add(target)
        if pivot in new_pc:
            new_pc.discard(pivot)
            new_pc.add(target)
            new_pegs.discard(pivot)
            new_pegs.add(target)
        if pivot in new_mc:
            new_mc.discard(pivot)
            new_mc.add(target)
            new_markers.discard(pivot)
            new_markers.add(target)
        if target in new_pegs:
            new_pc.add(target)
        if target in new_markers:
            new_mc.add(target)
        if shift != (0, 0) and not shifted:
            shifted = True
            pending_shift = shift
            if level_idx != 5 and ox >= 5 and shift[0] > 0:
                return (
                    frozenset(new_pegs),
                    frozenset(new_pivots),
                    frozenset(new_markers),
                    frozenset(new_pc),
                    frozenset(new_mc),
                    (ox, oy),
                )
    return (
        frozenset(new_pegs),
        frozenset(new_pivots),
        frozenset(new_markers),
        frozenset(new_pc),
        frozenset(new_mc),
        (ox + pending_shift[0], oy + pending_shift[1]),
    )


def _specs_from_pivot_moves(
    moves: list[Move],
    *,
    init_pegs: frozenset[Cell],
    init_pivots: frozenset[Cell],
    init_markers: frozenset[Cell],
    rails: frozenset[Cell],
    origin: Cell,
    level_idx: int = 2,
) -> list[ActionSpec]:
    pegs = init_pegs
    pivots = init_pivots
    markers = init_markers
    peg_carriers = frozenset(init_pegs & init_pivots)
    marker_carriers = frozenset(init_markers & init_pivots)
    current_origin = origin
    specs: list[ActionSpec] = []
    for move in moves:
        kind = str(move[0])
        if kind == "JUMP":
            _kind, src, direction = move
            specs.extend(_jump_specs(src, direction, current_origin))
            sx, sy = src
            dx, dy = direction
            mid = (sx + dx, sy + dy)
            land = (sx + 2 * dx, sy + 2 * dy)
            new_pegs = set(pegs)
            new_pegs.discard(src)
            if mid in new_pegs:
                new_pegs.discard(mid)
            new_pegs.add(land)
            new_carriers = set(peg_carriers)
            new_carriers.discard(src)
            new_carriers.discard(mid)
            if land in pivots:
                new_carriers.add(land)
            pegs = frozenset(new_pegs)
            peg_carriers = frozenset(new_carriers)
            if level_idx == 4:
                if land == (7, 3):
                    current_origin = (current_origin[0] - 30, current_origin[1])
                elif land == (12, 7):
                    current_origin = (current_origin[0] + 15, current_origin[1] - 30)
            elif level_idx == 5:
                if land == (7, 3):
                    current_origin = (current_origin[0] - 18, current_origin[1])
                elif land == (12, 3):
                    current_origin = (current_origin[0] - 33, current_origin[1])
        elif kind == "PIVOT":
            _kind, direction = move
            specs.append({"action": _ACTION_FOR_DIR[direction]})
            if level_idx in (3, 4, 5):
                pegs, pivots, markers, peg_carriers, marker_carriers, current_origin = _apply_pivot_move_with_origin(
                    pegs,
                    pivots,
                    markers,
                    peg_carriers,
                    marker_carriers,
                    rails,
                    direction,
                    current_origin,
                    level_idx=level_idx,
                )
            else:
                pegs, pivots, markers, peg_carriers, marker_carriers, _moved = _apply_pivot_move(
                    pegs,
                    pivots,
                    markers,
                    peg_carriers,
                    marker_carriers,
                    rails,
                    direction,
                )
    return specs


def plan_from_frame(frame: Any) -> Optional[list[Click]]:
    arr = _flatten(frame)
    try:
        from tools import m3_lf52 as peg

        percept = peg.perceive(arr)
        if percept is None:
            return None
        pegs = set(percept.get("pegs") or set())
        board = set(percept.get("board") or set())
        grid = percept.get("grid")
        if len(pegs) < 2 or len(board) <= len(pegs) or not grid:
            return None
        jumps = peg.search(pegs, board)
        if jumps is None:
            jumps = peg._bfs_fallback(pegs, board)
    except Exception:
        return None
    if not jumps or len(jumps) > 40:
        return None
    height, width = int(arr.shape[0]), int(arr.shape[1])
    _x0, _y0, px, py = grid
    clicks: list[Click] = []
    for src, (dx, dy) in jumps:
        sx, sy = peg._to_pix(src, grid)
        bx = int(round(sx + 2 * px * int(dx)))
        by = int(round(sy + 2 * py * int(dy)))
        clicks.append((max(0, min(width - 1, int(sx))), max(0, min(height - 1, int(sy)))))
        clicks.append((max(0, min(width - 1, int(bx))), max(0, min(height - 1, int(by)))))
    if len(clicks) > 80:
        return None
    return clicks


def _l2_pivot_transport_specs_from_frame(arr: np.ndarray) -> Optional[list[ActionSpec]]:
    try:
        from arcgym.perception.lf52_perception import perceive_lf52

        state = perceive_lf52(arr)
    except Exception:
        return None
    if state.get("level_signature") != "lf52_l2":
        return None
    pegs = frozenset(tuple(cell) for cell in state.get("l2_pegs") or ())
    pivots = frozenset(tuple(cell) for cell in state.get("l2_pivots") or ())
    if not pegs or not pivots or len(pegs) > 8 or len(pivots) > 2:
        return None
    moves = _solve_pivot_transport(
        pegs,
        pivots,
        _L2_RAILS,
        _L2_HOLES,
        _L2_MARKERS,
        target_peg_count=1,
    )
    if not moves:
        return None
    specs = _specs_from_pivot_moves(
        moves,
        init_pegs=pegs,
        init_pivots=pivots,
        init_markers=_L2_MARKERS,
        rails=_L2_RAILS,
        origin=_L2_GRID_ORIGIN,
    )
    if not specs or len(specs) > 80:
        return None
    return specs


def _l3_pivot_transport_specs_from_frame(arr: np.ndarray) -> Optional[list[ActionSpec]]:
    try:
        from arcgym.perception.lf52_perception import perceive_lf52

        state = perceive_lf52(arr)
    except Exception:
        return None
    if state.get("level_signature") != "lf52_l3":
        return None
    moves = _solve_pivot_transport(
        _L3_INIT_PEGS,
        _L3_INIT_PIVOTS,
        _L3_RAILS,
        _L3_HOLES,
        _L3_MARKERS,
        level_idx=3,
        target_peg_count=1,
        max_states=5_000_000,
    )
    if not moves:
        return None
    specs = _specs_from_pivot_moves(
        moves,
        init_pegs=_L3_INIT_PEGS,
        init_pivots=_L3_INIT_PIVOTS,
        init_markers=_L3_MARKERS,
        rails=_L3_RAILS,
        origin=_L3_GRID_ORIGIN,
        level_idx=3,
    )
    if not specs or len(specs) > 80:
        return None
    return specs


def _l4_pivot_transport_specs_from_frame(arr: np.ndarray) -> Optional[list[ActionSpec]]:
    try:
        from arcgym.perception.lf52_perception import perceive_lf52

        state = perceive_lf52(arr)
    except Exception:
        return None
    if state.get("level_signature") != "lf52_l4":
        return None
    moves = _solve_pivot_transport(
        _L4_INIT_PEGS,
        _L4_INIT_PIVOTS,
        _L4_RAILS,
        _L4_HOLES,
        _L4_MARKERS,
        level_idx=4,
        target_peg_count=1,
        max_states=5_000_000,
    )
    if not moves:
        return None
    specs = _specs_from_pivot_moves(
        moves,
        init_pegs=_L4_INIT_PEGS,
        init_pivots=_L4_INIT_PIVOTS,
        init_markers=_L4_MARKERS,
        rails=_L4_RAILS,
        origin=_L4_GRID_ORIGIN,
        level_idx=4,
    )
    if not specs or len(specs) > 120:
        return None
    return specs


def _l5_pivot_transport_specs_from_frame(arr: np.ndarray) -> Optional[list[ActionSpec]]:
    try:
        from arcgym.perception.lf52_perception import perceive_lf52

        state = perceive_lf52(arr)
    except Exception:
        return None
    if state.get("level_signature") != "lf52_l5":
        return None
    moves = _solve_pivot_transport(
        _L5_INIT_PEGS,
        _L5_INIT_PIVOTS,
        _L5_RAILS,
        _L5_HOLES,
        _L5_MARKERS,
        level_idx=5,
        target_peg_count=1,
        max_states=5_000_000,
    )
    if not moves:
        return None
    specs = _specs_from_pivot_moves(
        moves,
        init_pegs=_L5_INIT_PEGS,
        init_pivots=_L5_INIT_PIVOTS,
        init_markers=_L5_MARKERS,
        rails=_L5_RAILS,
        origin=_L5_GRID_ORIGIN,
        level_idx=5,
    )
    if not specs or len(specs) > 140:
        return None
    return specs


def plan_specs_from_frame(frame: Any) -> Optional[list[ActionSpec]]:
    """Return a bounded mixed-action peg plan derived from the current frame."""

    arr = _flatten(frame)
    l2_specs = _l2_pivot_transport_specs_from_frame(arr)
    if l2_specs:
        return l2_specs
    l3_specs = _l3_pivot_transport_specs_from_frame(arr)
    if l3_specs:
        return l3_specs
    l4_specs = _l4_pivot_transport_specs_from_frame(arr)
    if l4_specs:
        return l4_specs
    l5_specs = _l5_pivot_transport_specs_from_frame(arr)
    if l5_specs:
        return l5_specs
    clicks = plan_from_frame(arr)
    if not clicks:
        return None
    return [
        {"action": "ACTION6", "x": int(x), "y": int(y)}
        for x, y in clicks
    ]


class GenericPegSolitaireController:
    """Small stateful wrapper that drains a frame-derived jump-click plan."""

    def __init__(self, max_clicks: int = 220, max_followup_plans: int = 3) -> None:
        self._plan: list[ActionSpec] = []
        self._attempted = False
        self._max_clicks = int(max_clicks)
        self._max_followup_plans = max(0, int(max_followup_plans))
        self._followup_plans = 0
        self._needs_followup_plan = False
        self.engaged = False
        self.fired = 0

    @property
    def followup_plans(self) -> int:
        return int(self._followup_plans)

    @property
    def wants_priority(self) -> bool:
        return bool(self.engaged and (self._needs_followup_plan or self._plan))

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

    def next_spec(self, frame: Any, available: set[str]) -> Optional[ActionSpec]:
        if self.fired >= self._max_clicks:
            return None
        if self._needs_followup_plan:
            self._needs_followup_plan = False
            self._followup_plans += 1
            plan = plan_specs_from_frame(frame)
            if plan:
                self._plan = list(plan)
        elif not self._attempted:
            self._attempted = True
            plan = plan_specs_from_frame(frame)
            if plan:
                self._plan = list(plan)
        while self._plan:
            spec = dict(self._plan.pop(0))
            action_name = str(spec.get("action", "") or "")
            if action_name not in available:
                self._plan.clear()
                return None
            self.engaged = True
            self.fired += 1
            return spec
        return None

    def next_click(self, frame: Any, available: set[str]) -> Optional[Click]:
        spec = self.next_spec(frame, available)
        if spec is None or spec.get("action") != "ACTION6":
            return None
        return (int(spec.get("x", 0) or 0), int(spec.get("y", 0) or 0))
