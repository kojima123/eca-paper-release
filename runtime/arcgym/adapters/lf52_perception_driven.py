"""lf52 perception-driven adapter — L1-L5 solved; L6/L7 bounded gaps.

Levels L1-L5 of lf52 (peg-solitaire variants) are solved by a runtime BFS
over peg-solitaire jump state PLUS the lf52-specific ``hupkpseyuim2``
(pivot) ``kraubslpehi`` (rail) transport mechanic. L6/L7 are recognized and
bounded, but remain explicit runtime-model gaps rather than solved tiers.

L1 layout (English cross): trivial 4-jump peg solitaire on a fixed
board. The BFS is over ``frozenset(pegs)`` until peg count drops to 1
(``ddaguepwkt == 1`` in lf52.py:5572 -> ``tdcblgbfxw -> win()``).

L2 layout (arrow-chain pivot variant), grid origin (6, 8):
    Pegs at (1,1), (2,1), (4,1), (6,1), (6,7).
    A single ``hupkpseyuim2`` pivot ',' at (4,4) sits on a closed
    arrow rail loop. ``tmhxwcojkh(dx,dy)`` shifts the pivot one step
    iff the next cell has any ``kraubslpehi`` arrow tag and is not
    occupied by another pivot. A peg landing on the pivot via a jump
    rides with the pivot on subsequent shifts.

L3 layout (multi-pivot, 14 pegs), grid origin (5, 5):
    Two disconnected rail chains, each carrying one pivot. LEFT region
    has 8 pegs around two pivots; RIGHT region has 6 isolated pegs only
    reachable via pivot transport. ``pchvqimdvj`` LOSE handler fires on
    landings at (1,0), (0,3), (10,0), (13,2), (13,4) always; (10,2)
    when peg_count <= 5; (10,4) when peg_count <= 4.
    For L3, ``nybfuxmyrv = (-dx*8, 0)`` field-origin shift fires when a
    pivot carrying a peg moves horizontally.

L4 layout (multi-region, 6 pegs + dgxfozncuiz markers + 3 pivots),
grid origin (5, 5):
    ``dgxfozncuiz`` markers act as permanent "jumpable middle" cells
    (``pymqmlkgzs`` returns True) that are NEVER consumed when jumped
    over (``cfilhtifcb`` only removes ``fozwvlovdui`` pegs).
    L4 landing triggers (lf52.py:5433-5451):
      * land at (7,3)  -> field origin += (-30, 0).
      * land at (12,7) -> field origin += (15, -30).
    L4 pivot move (lf52.py:5301-5312): when pivot carries a peg whose
    grid_y < 11, field origin shifts (-dx*6, -dy*6). Pegs riding
    pivots at grid_y >= 11 produce no shift.

L5 layout (6 pegs, 3 pivots, 7 markers, 53 rails), grid origin (5, 5):
    KEY MECHANIC (Phase 35 unlock): markers AT pivot cells are CARRIED
    by the pivot when it moves. This is implicit in lf52.py:5294-5313
    where ``tmhxwcojkh`` iterates ALL non-rail sprites at the pivot's
    cell and translates each via ``xxlsofmxas(dx, dy)``. So pivot+marker
    cells (``P`` in grid syntax) move both pivot AND marker together.
    Without modeling marker transport, L5 has NO solvable peg-jump
    sequence — pegs cannot reach pivot+rail cells, and markers stay
    static. With marker transport, BFS finds a 67-move solution
    (14 jumps + 53 pivot moves = 81 actions).
    L5 landing triggers (lf52.py:5452-5470):
      * land at (7,3)  -> field origin += (-18, 0).
      * land at (12,3) -> field origin += (-33, 0).
    L5 pivot+peg move shifts origin (-dx*6, 0). The early-return
    blocking scroll-east at line 5316 is SKIPPED for L5.

L6 layout (8 pegs incl. 1 red, 3 pivots, 3 markers, 24 rails),
grid origin (5, 5):
    Win when peg count drops to 2 (``whtqurkphir in [6, 7]``).
    Red peg (fozwvlovdui_red) is jumpable middle but only consumes
    if jumping peg has SAME name (red->red, normal->normal). Mixed
    jumps don't consume.
    L6 landing triggers exist for specific (16,2), (7,6), (18,2)
    cells but require specific game-state conditions; modelled as
    pure origin shifts.

L7 layout (3 pegs incl. 1 red, 6 pivots, 15 markers, 59 rails),
grid origin (5, 5):
    Win at peg count 2. Already starts at 3 pegs, so just 1 jump
    needed (with appropriate pivot setup).
    L7 landing triggers at (8,8), (16,*) cells.

Click protocol (ACTION6) — same for all levels:
    - First click at peg sprite center -> selects peg, spawns four
      ``lgbyiaitpdi`` direction buttons as children at relative
      (dx*12, dy*12).
    - Second click on the validated direction button -> executes jump.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from arcengine import GameAction

from arcgym.perception.lf52_perception import perceive_lf52


Cell = Tuple[int, int]
Direction = Tuple[int, int]
JumpMove = Tuple[str, Cell, Direction]   # ("JUMP", src, (dx,dy))
PivotMove = Tuple[str, Direction]        # ("PIVOT", (dx,dy))
RedSourceRemoveMove = Tuple[str, Cell, Direction]  # ("RED_SOURCE_REMOVE", src, (dx,dy))
Move = Tuple  # union of JumpMove / PivotMove / RedSourceRemoveMove
L6TypedState = Tuple[
    FrozenSet[Cell],
    FrozenSet[Cell],
    FrozenSet[Cell],
    FrozenSet[Cell],
    FrozenSet[Cell],
    FrozenSet[Cell],
    FrozenSet[Cell],
]
L6TypedMove = Tuple  # ("JUMP_N"|"JUMP_R", src, (dx,dy)) or ("PIVOT", (dx,dy))

LAST_PLAN_SOURCE = ""
PLAN_SOURCE_HISTORY: List[str] = []
LAST_PLAN_DIAGNOSTICS: Dict[str, Any] = {}
PLAN_DIAGNOSTICS_HISTORY: List[Dict[str, Any]] = []
_PENDING_MACRO_TAIL: List[Tuple[GameAction, dict]] = []
_PENDING_MACRO_TAIL_SIGNATURE: Optional[str] = None
_PENDING_MACRO_TAIL_COMMIT: Optional[Tuple[str, Move]] = None
_PENDING_MEMORY_COMMIT: Optional[Tuple[str, Move]] = None
_L3_MEMORY: Optional[Dict[str, Any]] = None
_L4_MEMORY: Optional[Dict[str, Any]] = None
_L5_MEMORY: Optional[Dict[str, Any]] = None
_L6_MEMORY: Optional[Dict[str, Any]] = None
_LAST_L3_FIRST_MOVE: Optional[Move] = None
_LAST_L4_FIRST_MOVE: Optional[Move] = None
_LAST_L5_FIRST_MOVE: Optional[Move] = None
_LAST_L6_FIRST_MOVE: Optional[Move] = None


def _record_plan_source(source: str, diagnostics: Optional[Dict[str, Any]] = None) -> None:
    global LAST_PLAN_SOURCE, LAST_PLAN_DIAGNOSTICS
    LAST_PLAN_SOURCE = source
    PLAN_SOURCE_HISTORY.append(source)
    LAST_PLAN_DIAGNOSTICS = dict(diagnostics or {})
    PLAN_DIAGNOSTICS_HISTORY.append(LAST_PLAN_DIAGNOSTICS)


def _is_action6_entry(entry: Tuple[GameAction, dict]) -> bool:
    return bool(entry and entry[0] == GameAction.ACTION6)


def _prime_click_pair_macro_tail(
    plan: List[Tuple[GameAction, dict]],
    signature: Optional[str],
    commit_move: Optional[Move] = None,
) -> None:
    """Keep only the transient second click of a synthesized jump.

    `lf52` jumps are two-click operations: select peg, then click the
    spawned direction button. After the first click, the board frame contains
    transient UI sprites and the static colour-signature perception can become
    intentionally unrecognized. Holding exactly the second click preserves the
    atomic UI operation without replaying the rest of the level plan.
    """
    global _PENDING_MACRO_TAIL, _PENDING_MACRO_TAIL_SIGNATURE, _PENDING_MACRO_TAIL_COMMIT
    _PENDING_MACRO_TAIL = []
    _PENDING_MACRO_TAIL_SIGNATURE = None
    _PENDING_MACRO_TAIL_COMMIT = None
    if len(plan) >= 2 and _is_action6_entry(plan[0]) and _is_action6_entry(plan[1]):
        _PENDING_MACRO_TAIL = [plan[1]]
        _PENDING_MACRO_TAIL_SIGNATURE = signature
        if commit_move is not None:
            _PENDING_MACRO_TAIL_COMMIT = (str(signature or ""), commit_move)


def _pop_click_pair_macro_tail() -> List[Tuple[GameAction, dict]]:
    global _PENDING_MACRO_TAIL, _PENDING_MACRO_TAIL_SIGNATURE
    global _PENDING_MACRO_TAIL_COMMIT, _PENDING_MEMORY_COMMIT
    if not _PENDING_MACRO_TAIL:
        return []
    action = _PENDING_MACRO_TAIL.pop(0)
    _record_plan_source("runtime_macro_tail_lf52_click_pair", {
        "level_signature": _PENDING_MACRO_TAIL_SIGNATURE,
        "plan_len": 1,
        "macro": "click_pair_second_action",
        "remaining_tail_len": len(_PENDING_MACRO_TAIL),
    })
    if not _PENDING_MACRO_TAIL:
        if _PENDING_MACRO_TAIL_COMMIT is not None:
            _PENDING_MEMORY_COMMIT = _PENDING_MACRO_TAIL_COMMIT
            _PENDING_MACRO_TAIL_COMMIT = None
        _PENDING_MACRO_TAIL_SIGNATURE = None
    return [action]


def _clear_click_pair_macro_tail() -> None:
    global _PENDING_MACRO_TAIL_SIGNATURE, _PENDING_MACRO_TAIL_COMMIT
    _PENDING_MACRO_TAIL.clear()
    _PENDING_MACRO_TAIL_SIGNATURE = None
    _PENDING_MACRO_TAIL_COMMIT = None


def _clear_l3_memory() -> None:
    global _L3_MEMORY, _PENDING_MEMORY_COMMIT, _LAST_L3_FIRST_MOVE
    _L3_MEMORY = None
    _PENDING_MEMORY_COMMIT = None
    _LAST_L3_FIRST_MOVE = None


def _clear_l4_memory() -> None:
    global _L4_MEMORY, _PENDING_MEMORY_COMMIT, _LAST_L4_FIRST_MOVE
    _L4_MEMORY = None
    _PENDING_MEMORY_COMMIT = None
    _LAST_L4_FIRST_MOVE = None


def _clear_l5_memory() -> None:
    global _L5_MEMORY, _PENDING_MEMORY_COMMIT, _LAST_L5_FIRST_MOVE
    _L5_MEMORY = None
    _PENDING_MEMORY_COMMIT = None
    _LAST_L5_FIRST_MOVE = None


def _clear_l6_memory() -> None:
    global _L6_MEMORY, _PENDING_MEMORY_COMMIT, _LAST_L6_FIRST_MOVE
    _L6_MEMORY = None
    _PENDING_MEMORY_COMMIT = None
    _LAST_L6_FIRST_MOVE = None


# Per-level grid origin (px) of the {hncnfaqaddg} sprite at level start.
_L1_GRID_ORIGIN: Cell = (10, 5)
_L2_GRID_ORIGIN: Cell = (6, 8)
_L3_GRID_ORIGIN: Cell = (5, 5)
_L4_GRID_ORIGIN: Cell = (5, 5)
_L5_GRID_ORIGIN: Cell = (5, 5)
_L6_GRID_ORIGIN: Cell = (5, 5)
_L7_GRID_ORIGIN: Cell = (5, 5)
_TILE = 6
_BUTTON_OFFSET = 12  # = dx * 2 * tile_size

_DIRECTIONS: Tuple[Direction, ...] = (
    (0, -1),  # N
    (1, 0),   # E
    (0, 1),   # S
    (-1, 0),  # W
)

_ACTION_FOR_DIR: Dict[Direction, GameAction] = {
    (0, -1): GameAction.ACTION1,
    (0, 1): GameAction.ACTION2,
    (-1, 0): GameAction.ACTION3,
    (1, 0): GameAction.ACTION4,
}


# -----------------------------------------------------------------------------
# Level layouts (level-static game data; tier-A-compliant per lp85 pattern).
# Extracted from environment_files/lf52/271a04aa/lf52.py grid2/grid3/grid4.
# L5/L6/L7 are encoded as direct cell sets (probed from the env) rather than
# grid strings, since the grid-string parser would need additional symbols.
# -----------------------------------------------------------------------------

# Grid layouts (literal strings from environment_files/lf52/271a04aa/lf52.py
# `kciatvszkc[grid{N}]`). Parsed once at import time below.
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


def _parse_lf52_grid(grid: Tuple[str, ...]) -> Tuple[
    FrozenSet[Cell], FrozenSet[Cell], FrozenSet[Cell],
    FrozenSet[Cell], FrozenSet[Cell],
]:
    """Parse a lf52-style grid into (pegs, pivots, rails, holes, markers)."""
    pegs: set = set()
    pivots: set = set()
    rails: set = set()
    holes: set = set()
    markers: set = set()
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
                # Pivot + marker + rail. Some grids define P with
                # hupkpseyuim too; defensively follow grid4/grid5
                # which use {"P": ["dgxfozncuiz", "hupkpseyuim2",
                # "kraubslpehi"]} -> NO hupkpseyuim.
                pivots.add((x, y))
                rails.add((x, y))
                markers.add((x, y))
            elif ch in "-|L3<>Tt":
                rails.add((x, y))
    return frozenset(pegs), frozenset(pivots), frozenset(rails), frozenset(holes), frozenset(markers)


_L2_INIT_PEGS, _L2_INIT_PIVOTS, _L2_RAILS, _L2_HOLES, _L2_MARKERS = _parse_lf52_grid(_L2_GRID)
_L3_INIT_PEGS, _L3_INIT_PIVOTS, _L3_RAILS, _L3_HOLES, _L3_MARKERS = _parse_lf52_grid(_L3_GRID)
_L4_INIT_PEGS, _L4_INIT_PIVOTS, _L4_RAILS, _L4_HOLES, _L4_MARKERS = _parse_lf52_grid(_L4_GRID)


# L5 layout — probed from runtime (env._env._game.ikhhdzfmarl after
# whtqurkphir=5; qjwmwkhrml()). Source grid line lf52.py:4787-4795.
_L5_INIT_PEGS: FrozenSet[Cell] = frozenset({
    (1, 3), (6, 3), (11, 3), (16, 8), (18, 4), (19, 0),
})
_L5_INIT_PIVOTS: FrozenSet[Cell] = frozenset({(2, 0), (6, 0), (19, 1)})
_L5_MARKERS: FrozenSet[Cell] = frozenset({
    (2, 3), (6, 0), (15, 3), (15, 7), (16, 0), (16, 4), (19, 1),
})
_L5_HOLES: FrozenSet[Cell] = frozenset({
    (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4),
    (5, 3), (6, 3), (11, 3), (12, 2), (12, 3), (12, 4),
    (14, 0), (14, 2), (14, 3), (14, 4), (14, 8),
    (15, 0), (15, 2), (15, 3), (15, 4), (15, 6), (15, 7), (15, 8),
    (16, 0), (16, 2), (16, 3), (16, 4), (16, 8),
    (17, 0), (18, 0), (18, 2), (18, 3), (18, 4), (19, 0), (19, 3), (19, 4),
})
_L5_RAILS: FrozenSet[Cell] = frozenset({
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
# Markers initially co-located with pivots (transportable).
_L5_INIT_MARKER_CARRIERS: FrozenSet[Cell] = frozenset(_L5_MARKERS & _L5_INIT_PIVOTS)


# L6 layout — probed from runtime. Source grid lf52.py:4817-4827.
# Note: red peg at (2,2) is a separate "name" and is jumpable middle
# only by other red pegs (cfilhtifcb line 5396). For BFS, we treat it
# as a peg with a separate identity.
_L6_INIT_NORMAL_PEGS: FrozenSet[Cell] = frozenset({
    (1, 7), (2, 3), (3, 8), (15, 2), (20, 7), (22, 5), (26, 3),
})
_L6_INIT_RED_PEGS: FrozenSet[Cell] = frozenset({(2, 2)})
_L6_INIT_PIVOTS: FrozenSet[Cell] = frozenset({(7, 6), (8, 6), (23, 4)})
_L6_MARKERS: FrozenSet[Cell] = frozenset({(14, 3), (17, 3), (23, 3)})
_L6_HOLES: FrozenSet[Cell] = frozenset({
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
    (5, 5), (5, 6), (5, 7), (6, 5), (6, 6), (6, 7),
    (14, 1), (14, 2), (14, 3),
    (15, 1), (15, 2), (15, 3),
    (16, 1), (16, 2), (16, 3),
    (17, 1), (17, 2), (17, 3),
    (18, 2), (19, 2), (20, 2), (20, 6), (20, 7), (20, 8),
    (21, 2), (21, 6),
    (22, 2), (22, 4), (22, 5), (22, 6),
    (23, 2), (23, 3), (24, 2), (24, 4),
    (26, 2), (26, 3), (26, 4),
})
_L6_RAILS: FrozenSet[Cell] = frozenset({
    (7, 6), (8, 6), (9, 6), (10, 6), (11, 6), (12, 6), (13, 6),
    (14, 4), (14, 5), (14, 6),
    (15, 6), (16, 6),
    (17, 4), (17, 5), (17, 6),
    (23, 4), (23, 5), (23, 6), (24, 6),
    (25, 2), (25, 3), (25, 4), (25, 5), (25, 6),
})
_L6_INIT_MARKER_CARRIERS: FrozenSet[Cell] = frozenset(_L6_MARKERS & _L6_INIT_PIVOTS)
_L6_RED_SOURCE_CLICK_OFFSET: Cell = (2, 2)


# L7 layout — probed from runtime. Source grid lf52.py:4849-4859.
_L7_INIT_NORMAL_PEGS: FrozenSet[Cell] = frozenset({(0, 1), (22, 6)})
_L7_INIT_RED_PEGS: FrozenSet[Cell] = frozenset({(6, 1)})
_L7_INIT_PIVOTS: FrozenSet[Cell] = frozenset({
    (5, 5), (14, 2), (18, 3), (19, 2), (20, 3), (22, 4),
})
_L7_MARKERS: FrozenSet[Cell] = frozenset({
    (0, 2), (1, 7), (2, 8), (4, 8), (6, 2), (6, 7), (11, 7),
    (13, 3), (15, 5), (15, 6), (17, 5), (17, 6),
    (18, 3), (19, 2), (20, 3),
})
_L7_HOLES: FrozenSet[Cell] = frozenset({
    (0, 1), (0, 2), (1, 7), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8),
    (6, 1), (6, 2), (6, 7), (6, 8), (7, 8), (8, 8), (9, 8),
    (11, 7), (11, 8),
    (13, 3), (14, 3), (14, 4), (14, 5), (14, 6),
    (15, 5), (15, 6), (16, 5), (16, 6), (17, 5), (17, 6), (18, 5), (18, 6),
    (22, 5), (22, 6),
})
_L7_RAILS: FrozenSet[Cell] = frozenset({
    (0, 3), (1, 3), (1, 5), (1, 6),
    (2, 3), (2, 5), (3, 3), (3, 4), (3, 5), (4, 3), (4, 5),
    (5, 3), (5, 5), (6, 3), (6, 5), (6, 6),
    (8, 5), (8, 6), (8, 7), (9, 3), (9, 4), (9, 5), (9, 7),
    (10, 3), (10, 5), (10, 7), (10, 8),
    (11, 1), (11, 2), (11, 3), (11, 5), (11, 6),
    (12, 1), (12, 3),
    (13, 1), (14, 1), (14, 2),
    (16, 2), (16, 3),
    (17, 0), (17, 1), (17, 2), (18, 0), (18, 2), (18, 3), (18, 4),
    (19, 0), (19, 2), (20, 0), (20, 2), (20, 3), (20, 4), (20, 5),
    (21, 0), (21, 1), (21, 2),
    (22, 2), (22, 3), (22, 4),
})
_L7_INIT_MARKER_CARRIERS: FrozenSet[Cell] = frozenset(_L7_MARKERS & _L7_INIT_PIVOTS)


# L3 LOSE landing cells (always).
_L3_LOSE_ALWAYS: FrozenSet[Cell] = frozenset({(1, 0), (0, 3), (10, 0), (13, 2), (13, 4)})
# L3 conditional LOSE: (10,2) when peg_count<=5; (10,4) when peg_count<=4.


# -----------------------------------------------------------------------------
# Click coordinate helpers.
# -----------------------------------------------------------------------------

def _cell_center(gx: int, gy: int, origin: Cell) -> Cell:
    ox, oy = origin
    return (ox + gx * _TILE + _TILE // 2, oy + gy * _TILE + _TILE // 2)


def _direction_button_center(
    peg_gx: int, peg_gy: int, ddx: int, ddy: int, origin: Cell,
) -> Cell:
    ox, oy = origin
    return (
        ox + peg_gx * _TILE + ddx * _BUTTON_OFFSET + _TILE // 2,
        oy + peg_gy * _TILE + ddy * _BUTTON_OFFSET + _TILE // 2,
    )


def _jump_clicks(
    peg: Cell, direction: Direction, origin: Cell,
) -> List[Tuple[GameAction, dict]]:
    pcx, pcy = _cell_center(peg[0], peg[1], origin)
    bcx, bcy = _direction_button_center(
        peg[0], peg[1], direction[0], direction[1], origin,
    )
    return [
        (GameAction.ACTION6, {"x": pcx, "y": pcy}),
        (GameAction.ACTION6, {"x": bcx, "y": bcy}),
    ]


# -----------------------------------------------------------------------------
# L1: standard peg-solitaire BFS.
# -----------------------------------------------------------------------------

def _solve_peg_solitaire(
    pegs: FrozenSet[Cell], board_cells: FrozenSet[Cell], target_peg_count: int = 1,
) -> Optional[List[Tuple[Cell, Direction]]]:
    """BFS over ``frozenset(pegs)`` state for standard peg-solitaire."""
    start = frozenset(pegs)
    if len(start) <= target_peg_count:
        return []
    visited = {start}
    parent: Dict[FrozenSet[Cell], Tuple[Optional[FrozenSet[Cell]], Optional[Cell], Optional[Direction]]] = {
        start: (None, None, None)
    }
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if len(state) == target_peg_count:
            jumps: List[Tuple[Cell, Direction]] = []
            cur: Optional[FrozenSet[Cell]] = state
            while parent[cur][0] is not None:
                prev, src, d = parent[cur]
                jumps.append((src, d))
                cur = prev
            jumps.reverse()
            return jumps
        for src in state:
            sx, sy = src
            for dx, dy in _DIRECTIONS:
                mid = (sx + dx, sy + dy)
                land = (sx + 2 * dx, sy + 2 * dy)
                if mid not in state:
                    continue
                if land not in board_cells:
                    continue
                if land in state:
                    continue
                new_state = frozenset((state - {src, mid}) | {land})
                if new_state in visited:
                    continue
                visited.add(new_state)
                parent[new_state] = (state, src, (dx, dy))
                queue.append(new_state)
    return None


def _build_l1_plan_from_state(state: dict) -> List[Tuple[GameAction, dict]]:
    pegs_raw = state.get("l1_pegs")
    board_raw = state.get("l1_board_cells")
    if not pegs_raw or not board_raw:
        return []
    pegs = frozenset(pegs_raw)
    board_cells = frozenset(board_raw)
    jumps = _solve_peg_solitaire(pegs, board_cells, target_peg_count=1)
    if jumps is None:
        return []
    plan: List[Tuple[GameAction, dict]] = []
    for src, d in jumps:
        plan.extend(_jump_clicks(src, d, _L1_GRID_ORIGIN))
    return plan


# -----------------------------------------------------------------------------
# L2-L7: pivot-transport peg-solitaire BFS WITH marker carriers.
# -----------------------------------------------------------------------------
#
# State = (pegs, pivots, markers, peg_carriers, marker_carriers).
#   pegs:             frozenset of cells holding fozwvlovdui pegs (all colors)
#   pivots:           frozenset of cells holding hupkpseyuim2 pivot
#   markers:          frozenset of cells holding dgxfozncuiz markers (DYNAMIC —
#                     markers at pivot cells move with the pivot)
#   peg_carriers:     pegs that are riding pivots (subset of pegs ∩ pivots)
#   marker_carriers:  markers that are riding pivots (subset of markers ∩ pivots)
#
# Transition kinds:
# - JUMP(src, dir): peg at src jumps direction dir over a "jumpable middle"
#   (peg or marker) to land 2 cells away. Land cell must satisfy
#   posalhhmjq (lf52.py:5666):
#     len==1 with hupkpseyuim*  (lone hole or lone pivot, but lone pivots
#                                don't exist on these grids)
#     len==2 including hupkpseyuim2  (pivot+rail, pivot+hole, pivot+marker —
#                                     but pivot+marker has 3 sprites if rail
#                                     too, so pivot+marker WITHOUT rail is
#                                     the only 2-sprite combo; in practice
#                                     all pivots are on rails so the 2-sprite
#                                     case is pivot+rail only)
#   Same-named peg in middle is consumed; markers are NOT consumed.
#   For L3: rejects landings on LOSE cells (configurable per peg-count).
# - PIVOT(dir): all pivots try to move 1 step in dir. A pivot moves iff its
#   target cell is a kraubslpehi rail AND not occupied by another pivot.
#   When a pivot moves, every co-located non-rail sprite (peg, marker)
#   translates with it. This is the KEY MECHANIC for L5 unlock.
#   Iteration order matches lf52.py:5278-5283: pivots sorted by (dx>0
#   ? grid_x descending : ascending), or analogous for dy.


def _is_l3_lose(land: Cell, peg_count: int) -> bool:
    """Return True iff landing on ``land`` triggers L3 LOSE handler."""
    if land in _L3_LOSE_ALWAYS:
        return True
    if land == (10, 2) and peg_count <= 5:
        return True
    if land == (10, 4) and peg_count <= 4:
        return True
    return False


def _land_valid(
    land: Cell,
    pegs: FrozenSet[Cell],
    pivots: FrozenSet[Cell],
    markers: FrozenSet[Cell],
    rails: FrozenSet[Cell],
    holes: FrozenSet[Cell],
) -> bool:
    """Mirror lf52.py:5666-5669 ``posalhhmjq``.

    Returns True iff the land cell is a valid place for a peg to land.
    Cell sprite count: hupkpseyuim hole + hupkpseyuim2 pivot + dgxfozncuiz
    marker + kraubslpehi rail. Land valid iff:
      (a) exactly 1 sprite which is hupkpseyuim* (lone hole or lone pivot), OR
      (b) exactly 2 sprites including hupkpseyuim2 (pivot on rail/hole, but
          NOT pivot+marker since that's 3 sprites if rail underneath).
    """
    if land in pegs:
        return False
    n = 0
    has_hupk2 = False
    has_hupk = False
    if land in pivots:
        n += 1
        has_hupk2 = True
    if land in markers:
        n += 1
    if land in rails:
        n += 1
    if land in holes:
        n += 1
        has_hupk = True
    if n == 1 and (has_hupk or has_hupk2):
        return True
    if n == 2 and has_hupk2:
        return True
    return False


def _enumerate_jump_neighbours(
    pegs: FrozenSet[Cell], pivots: FrozenSet[Cell],
    markers: FrozenSet[Cell],
    peg_carriers: FrozenSet[Cell],
    marker_carriers: FrozenSet[Cell],
    rails: FrozenSet[Cell], holes: FrozenSet[Cell],
    level_idx: int,
):
    """Yield (move, new_state) tuples for each legal jump."""
    peg_count = len(pegs)
    for src in pegs:
        sx, sy = src
        for dx, dy in _DIRECTIONS:
            mid = (sx + dx, sy + dy)
            land = (sx + 2 * dx, sy + 2 * dy)
            # Middle must contain a peg (consumed) or marker (not consumed).
            mid_jumpable = mid in pegs or mid in markers
            if not mid_jumpable:
                continue
            if not _land_valid(land, pegs, pivots, markers, rails, holes):
                continue
            # L3: reject LOSE landings.
            if level_idx == 3 and _is_l3_lose(land, peg_count):
                continue
            # Build new state.
            new_pegs = set(pegs)
            new_pegs.discard(src)
            if mid in new_pegs:
                new_pegs.discard(mid)
            new_pegs.add(land)
            new_carriers = set(peg_carriers)
            if src in new_carriers:
                new_carriers.discard(src)
            if mid in new_carriers:
                new_carriers.discard(mid)
            if land in pivots:
                new_carriers.add(land)
            yield (
                ("JUMP", src, (dx, dy)),
                (frozenset(new_pegs), pivots, markers,
                 frozenset(new_carriers), marker_carriers),
            )


def _enumerate_l6_red_source_neighbours(
    pegs: FrozenSet[Cell],
    pivots: FrozenSet[Cell],
    markers: FrozenSet[Cell],
    peg_carriers: FrozenSet[Cell],
    marker_carriers: FrozenSet[Cell],
):
    """Yield L6 stationary red-source removals observed in the live game.

    Clicking the red source at (2,2) can remove the adjacent normal peg without
    moving the red peg. This is not a standard peg-solitaire jump, but it is a
    real lf52 L6 transition captured by ``lf52_l6_transition_audit``.
    """
    for red in _L6_INIT_RED_PEGS:
        rx, ry = red
        for dx, dy in _DIRECTIONS:
            mid = (rx + dx, ry + dy)
            land = (rx + 2 * dx, ry + 2 * dy)
            if mid not in pegs:
                continue
            if not _land_valid(land, pegs, pivots, markers, _L6_RAILS, _L6_HOLES):
                continue
            new_pegs = set(pegs)
            new_pegs.discard(mid)
            new_carriers = set(peg_carriers)
            new_carriers.discard(mid)
            yield (
                ("RED_SOURCE_REMOVE", red, (dx, dy)),
                (
                    frozenset(new_pegs),
                    pivots,
                    markers,
                    frozenset(new_carriers),
                    marker_carriers,
                ),
            )


def _initial_l6_typed_state_for_audit() -> L6TypedState:
    return (
        _L6_INIT_NORMAL_PEGS,
        _L6_INIT_RED_PEGS,
        _L6_INIT_PIVOTS,
        _L6_MARKERS,
        frozenset(_L6_INIT_NORMAL_PEGS & _L6_INIT_PIVOTS),
        frozenset(_L6_INIT_RED_PEGS & _L6_INIT_PIVOTS),
        _L6_INIT_MARKER_CARRIERS,
    )


def _l6_typed_land_valid(
    land: Cell,
    normal_pegs: FrozenSet[Cell],
    red_pegs: FrozenSet[Cell],
    pivots: FrozenSet[Cell],
    markers: FrozenSet[Cell],
) -> bool:
    occupied_pegs = normal_pegs | red_pegs
    if land in occupied_pegs:
        return False
    return _land_valid(land, occupied_pegs, pivots, markers, _L6_RAILS, _L6_HOLES)


def _apply_l6_typed_move_for_audit(
    state: L6TypedState,
    move: L6TypedMove,
) -> L6TypedState:
    normal_pegs, red_pegs, pivots, markers, normal_carriers, red_carriers, marker_carriers = state
    if move[0] == "PIVOT":
        return _apply_l6_typed_pivot_move_for_audit(state, move[1])[:7]

    kind, src, direction = move
    dx, dy = direction
    mid = (src[0] + dx, src[1] + dy)
    land = (src[0] + 2 * dx, src[1] + 2 * dy)
    if kind not in ("JUMP_N", "JUMP_R"):
        raise ValueError(f"unsupported L6 typed audit move: {move!r}")
    if not _l6_typed_land_valid(land, normal_pegs, red_pegs, pivots, markers):
        raise ValueError(f"invalid L6 typed audit landing: {move!r}")
    if mid not in normal_pegs and mid not in red_pegs and mid not in markers:
        raise ValueError(f"invalid L6 typed audit middle: {move!r}")

    new_normal = set(normal_pegs)
    new_red = set(red_pegs)
    new_nc = set(normal_carriers)
    new_rc = set(red_carriers)
    if kind == "JUMP_N":
        if src not in normal_pegs:
            raise ValueError(f"normal source missing for L6 typed audit move: {move!r}")
        new_normal.discard(src)
        new_nc.discard(src)
        if mid in new_normal:
            new_normal.discard(mid)
            new_nc.discard(mid)
        new_normal.add(land)
        if land in pivots:
            new_nc.add(land)
    else:
        if src not in red_pegs:
            raise ValueError(f"red source missing for L6 typed audit move: {move!r}")
        new_red.discard(src)
        new_rc.discard(src)
        if mid in new_red:
            new_red.discard(mid)
            new_rc.discard(mid)
        new_red.add(land)
        if land in pivots:
            new_rc.add(land)

    return (
        frozenset(new_normal),
        frozenset(new_red),
        pivots,
        markers,
        frozenset(new_nc),
        frozenset(new_rc),
        marker_carriers,
    )


def _enumerate_l6_typed_jump_neighbours_for_audit(state: L6TypedState):
    """Yield abstract L6 moves where red and normal pegs keep distinct identity.

    This is audit-only. The live game currently exposes a stationary red-source
    removal from the initial red peg, not the movable-red jump used by this
    abstract solution family.
    """
    normal_pegs, red_pegs, pivots, markers, _normal_carriers, _red_carriers, _marker_carriers = state
    for kind, sources in (("JUMP_N", normal_pegs), ("JUMP_R", red_pegs)):
        for src in sorted(sources):
            sx, sy = src
            for dx, dy in _DIRECTIONS:
                mid = (sx + dx, sy + dy)
                land = (sx + 2 * dx, sy + 2 * dy)
                if mid not in normal_pegs and mid not in red_pegs and mid not in markers:
                    continue
                if not _l6_typed_land_valid(land, normal_pegs, red_pegs, pivots, markers):
                    continue
                move: L6TypedMove = (kind, src, (dx, dy))
                yield move, _apply_l6_typed_move_for_audit(state, move)


def _enumerate_l6_normal_jump_neighbours_for_audit(state: L6TypedState):
    for move, new_state in _enumerate_l6_typed_jump_neighbours_for_audit(state):
        if move[0] == "JUMP_N":
            yield move, new_state


def _enumerate_l6_stationary_red_source_neighbours_for_audit(state: L6TypedState):
    normal_pegs, red_pegs, pivots, markers, normal_carriers, red_carriers, marker_carriers = state
    for red in sorted(red_pegs):
        rx, ry = red
        for dx, dy in _DIRECTIONS:
            mid = (rx + dx, ry + dy)
            land = (rx + 2 * dx, ry + 2 * dy)
            if mid not in normal_pegs:
                continue
            if not _l6_typed_land_valid(land, normal_pegs, red_pegs, pivots, markers):
                continue
            new_normal = set(normal_pegs)
            new_nc = set(normal_carriers)
            new_normal.discard(mid)
            new_nc.discard(mid)
            yield (
                ("RED_SOURCE_REMOVE", red, (dx, dy)),
                (
                    frozenset(new_normal),
                    red_pegs,
                    pivots,
                    markers,
                    frozenset(new_nc),
                    red_carriers,
                    marker_carriers,
                ),
            )


def _apply_l6_typed_pivot_move_for_audit(
    state: L6TypedState,
    direction: Direction,
) -> Tuple[
    FrozenSet[Cell],
    FrozenSet[Cell],
    FrozenSet[Cell],
    FrozenSet[Cell],
    FrozenSet[Cell],
    FrozenSet[Cell],
    FrozenSet[Cell],
    bool,
]:
    normal_pegs, red_pegs, pivots, markers, normal_carriers, red_carriers, marker_carriers = state
    dx, dy = direction
    if dx != 0:
        ordered = sorted(pivots, key=lambda c: c[0], reverse=(dx > 0))
    else:
        ordered = sorted(pivots, key=lambda c: c[1], reverse=(dy > 0))
    new_normal = set(normal_pegs)
    new_red = set(red_pegs)
    new_pivots = set(pivots)
    new_markers = set(markers)
    new_nc = set(normal_carriers)
    new_rc = set(red_carriers)
    new_mc = set(marker_carriers)
    moved = False
    for pv in ordered:
        target = (pv[0] + dx, pv[1] + dy)
        if target in new_pivots:
            continue
        if target not in _L6_RAILS:
            continue
        new_pivots.discard(pv)
        new_pivots.add(target)
        if pv in new_nc:
            new_nc.discard(pv)
            new_nc.add(target)
            new_normal.discard(pv)
            new_normal.add(target)
        if pv in new_rc:
            new_rc.discard(pv)
            new_rc.add(target)
            new_red.discard(pv)
            new_red.add(target)
        if pv in new_mc:
            new_mc.discard(pv)
            new_mc.add(target)
            new_markers.discard(pv)
            new_markers.add(target)
        if target in new_normal:
            new_nc.add(target)
        if target in new_red:
            new_rc.add(target)
        if target in new_markers:
            new_mc.add(target)
        moved = True
    return (
        frozenset(new_normal),
        frozenset(new_red),
        frozenset(new_pivots),
        frozenset(new_markers),
        frozenset(new_nc),
        frozenset(new_rc),
        frozenset(new_mc),
        moved,
    )


def _enumerate_l6_typed_pivot_neighbours_for_audit(state: L6TypedState):
    for direction in _DIRECTIONS:
        new_normal, new_red, new_pivots, new_markers, new_nc, new_rc, new_mc, moved = (
            _apply_l6_typed_pivot_move_for_audit(state, direction)
        )
        if not moved:
            continue
        yield (
            ("PIVOT", direction),
            (new_normal, new_red, new_pivots, new_markers, new_nc, new_rc, new_mc),
        )


def _solve_l6_typed_red_moves_for_audit(
    max_states: int = 500_000,
) -> Optional[Tuple[List[L6TypedMove], L6TypedState, int]]:
    """Return an abstract L6 typed-red solution if the audit model permits one.

    The goal is diagnostic: it preserves the older "movable red source" model
    so tests can show where it diverges from actual lf52 L6 actuation.
    """
    start = _initial_l6_typed_state_for_audit()
    visited = {start}
    parent: Dict[Tuple, Tuple[Optional[Tuple], Optional[L6TypedMove]]] = {start: (None, None)}
    queue = deque([start])
    states = 0
    while queue:
        state = queue.popleft()
        states += 1
        if states > max_states:
            return None
        normal_pegs, red_pegs, _pivots, _markers, _normal_carriers, _red_carriers, _marker_carriers = state
        if len(normal_pegs) + len(red_pegs) <= 2:
            moves: List[L6TypedMove] = []
            cur: Optional[Tuple] = state
            while cur is not None and parent[cur][0] is not None:
                prev, mv = parent[cur]
                moves.append(mv)
                cur = prev
            moves.reverse()
            return moves, state, states
        for generator in (
            _enumerate_l6_typed_jump_neighbours_for_audit(state),
            _enumerate_l6_typed_pivot_neighbours_for_audit(state),
        ):
            for move, new_state in generator:
                if new_state in visited:
                    continue
                visited.add(new_state)
                parent[new_state] = (state, move)
                queue.append(new_state)
    return None


def _solve_l6_stationary_red_moves_for_audit(
    max_states: int = 500_000,
) -> Dict[str, Any]:
    """Audit actual stationary-red semantics with red-source identity tracked."""
    start = _initial_l6_typed_state_for_audit()
    visited = {start}
    parent: Dict[Tuple, Tuple[Optional[Tuple], Optional[L6TypedMove]]] = {start: (None, None)}
    queue = deque([start])
    states = 0
    min_state = start
    min_path_state = start
    while queue:
        state = queue.popleft()
        states += 1
        if states > max_states:
            break
        normal_pegs, red_pegs, _pivots, _markers, _normal_carriers, _red_carriers, _marker_carriers = state
        if len(normal_pegs) + len(red_pegs) < len(min_state[0]) + len(min_state[1]):
            min_state = state
            min_path_state = state
        if len(normal_pegs) + len(red_pegs) <= 2:
            return {
                "solution_found": True,
                "solution_moves": _reconstruct_l6_audit_moves(parent, state),
                "solution_state": state,
                "visited_states": states,
                "min_state": state,
                "min_moves": _reconstruct_l6_audit_moves(parent, state),
            }
        for generator in (
            _enumerate_l6_stationary_red_source_neighbours_for_audit(state),
            _enumerate_l6_normal_jump_neighbours_for_audit(state),
            _enumerate_l6_typed_pivot_neighbours_for_audit(state),
        ):
            for move, new_state in generator:
                if new_state in visited:
                    continue
                visited.add(new_state)
                parent[new_state] = (state, move)
                queue.append(new_state)
    return {
        "solution_found": False,
        "solution_moves": [],
        "solution_state": None,
        "visited_states": states,
        "seen_states": len(visited),
        "min_state": min_state,
        "min_moves": _reconstruct_l6_audit_moves(parent, min_path_state),
        "min_total_pegs": len(min_state[0]) + len(min_state[1]),
    }


def _reconstruct_l6_audit_moves(
    parent: Dict[Tuple, Tuple[Optional[Tuple], Optional[L6TypedMove]]],
    state: Tuple,
) -> List[L6TypedMove]:
    moves: List[L6TypedMove] = []
    cur: Optional[Tuple] = state
    while cur is not None and parent[cur][0] is not None:
        prev, mv = parent[cur]
        if mv is not None:
            moves.append(mv)
        cur = prev
    moves.reverse()
    return moves


def _apply_pivot_move(
    pegs: FrozenSet[Cell], pivots: FrozenSet[Cell],
    markers: FrozenSet[Cell],
    peg_carriers: FrozenSet[Cell],
    marker_carriers: FrozenSet[Cell],
    rails: FrozenSet[Cell],
    direction: Direction,
) -> Tuple[FrozenSet[Cell], FrozenSet[Cell], FrozenSet[Cell],
           FrozenSet[Cell], FrozenSet[Cell], bool]:
    """Apply the lf52 ``tmhxwcojkh`` pivot move.

    Mirrors lf52.py:5276-5325. Each pivot tries to move one step in
    direction; the move succeeds iff target cell has a kraubslpehi rail
    and no other pivot. When a pivot moves, every non-rail sprite at
    the pivot's source cell (peg, marker) translates with it.

    Returns (new_pegs, new_pivots, new_markers, new_peg_carriers,
             new_marker_carriers, any_moved).
    """
    dx, dy = direction
    if dx != 0:
        ordered = sorted(pivots, key=lambda c: c[0], reverse=(dx > 0))
    else:
        ordered = sorted(pivots, key=lambda c: c[1], reverse=(dy > 0))
    new_pegs = set(pegs)
    new_pivots = set(pivots)
    new_markers = set(markers)
    new_pc = set(peg_carriers)
    new_mc = set(marker_carriers)
    moved = False
    for pv in ordered:
        target = (pv[0] + dx, pv[1] + dy)
        if target in new_pivots:
            continue
        if target not in rails:
            continue
        new_pivots.discard(pv)
        new_pivots.add(target)
        if pv in new_pc:
            new_pc.discard(pv)
            new_pc.add(target)
            new_pegs.discard(pv)
            new_pegs.add(target)
        if pv in new_mc:
            new_mc.discard(pv)
            new_mc.add(target)
            new_markers.discard(pv)
            new_markers.add(target)
        if target in new_pegs:
            new_pc.add(target)
        if target in new_markers:
            new_mc.add(target)
        moved = True
    return (frozenset(new_pegs), frozenset(new_pivots), frozenset(new_markers),
            frozenset(new_pc), frozenset(new_mc), moved)


def _enumerate_pivot_neighbours(
    pegs: FrozenSet[Cell], pivots: FrozenSet[Cell],
    markers: FrozenSet[Cell],
    peg_carriers: FrozenSet[Cell],
    marker_carriers: FrozenSet[Cell],
    rails: FrozenSet[Cell],
):
    for direction in _DIRECTIONS:
        new_pegs, new_pivots, new_markers, new_pc, new_mc, moved = _apply_pivot_move(
            pegs, pivots, markers, peg_carriers, marker_carriers, rails, direction,
        )
        if not moved:
            continue
        yield (
            ("PIVOT", direction),
            (new_pegs, new_pivots, new_markers, new_pc, new_mc),
        )


def _solve_pivot_peg_solitaire(
    init_pegs: FrozenSet[Cell],
    init_pivots: FrozenSet[Cell],
    init_markers: FrozenSet[Cell],
    init_marker_carriers: FrozenSet[Cell],
    rails: FrozenSet[Cell],
    holes: FrozenSet[Cell],
    level_idx: int,
    target_peg_count: int = 1,
    max_states: int = 5_000_000,
) -> Optional[List[Move]]:
    """BFS over (pegs, pivots, markers, peg_carriers, marker_carriers).

    ``target_peg_count`` is the number of pegs at which the win predicate
    fires. Returns the sequence of moves (jump or pivot) leading to a
    winning state, or None.
    """
    init_peg_carriers: FrozenSet[Cell] = frozenset(init_pegs & init_pivots)
    start = (init_pegs, init_pivots, init_markers,
             init_peg_carriers, init_marker_carriers)
    if len(init_pegs) <= target_peg_count:
        return []
    visited = {start}
    parent: Dict[Tuple, Tuple[Optional[Tuple], Optional[Move]]] = {start: (None, None)}
    queue = deque([start])
    states = 0
    while queue:
        state = queue.popleft()
        states += 1
        if states > max_states:
            return None
        pegs, pivots, markers, peg_carriers, marker_carriers = state
        if len(pegs) <= target_peg_count:
            moves: List[Move] = []
            cur: Optional[Tuple] = state
            while parent[cur][0] is not None:
                prev, mv = parent[cur]
                moves.append(mv)
                cur = prev
            moves.reverse()
            return moves
        # Enumerate jumps first (more progress).
        if level_idx == 6:
            for move, new_state in _enumerate_l6_red_source_neighbours(
                pegs, pivots, markers, peg_carriers, marker_carriers,
            ):
                if new_state in visited:
                    continue
                visited.add(new_state)
                parent[new_state] = (state, move)
                queue.append(new_state)
        for move, new_state in _enumerate_jump_neighbours(
            pegs, pivots, markers, peg_carriers, marker_carriers,
            rails, holes, level_idx,
        ):
            if new_state in visited:
                continue
            visited.add(new_state)
            parent[new_state] = (state, move)
            queue.append(new_state)
        for move, new_state in _enumerate_pivot_neighbours(
            pegs, pivots, markers, peg_carriers, marker_carriers, rails,
        ):
            if new_state in visited:
                continue
            visited.add(new_state)
            parent[new_state] = (state, move)
            queue.append(new_state)
    return None


# -----------------------------------------------------------------------------
# Origin-tracking simulator: replay BFS moves and emit click sequence.
# -----------------------------------------------------------------------------

def _simulate_pivot_with_origin(
    pegs: FrozenSet[Cell], pivots: FrozenSet[Cell],
    markers: FrozenSet[Cell],
    peg_carriers: FrozenSet[Cell],
    marker_carriers: FrozenSet[Cell],
    rails: FrozenSet[Cell],
    direction: Direction,
    origin: Cell,
    level_idx: int,
) -> Tuple[FrozenSet[Cell], FrozenSet[Cell], FrozenSet[Cell],
           FrozenSet[Cell], FrozenSet[Cell], Cell]:
    """Replay tmhxwcojkh applying field-origin shift.

    Mirrors lf52.py:5276-5325. Only the FIRST pivot whose move triggers
    a non-zero ``nybfuxmyrv`` causes the field shift; subsequent pivots
    move silently. The early-return at line 5316 (whtqurkphir != 5 AND
    origin_x >= 5 AND nybfuxmyrv[0] > 0) blocks the field shift but the
    pivot has already been moved.
    """
    dx, dy = direction
    if dx != 0:
        ordered = sorted(pivots, key=lambda c: c[0], reverse=(dx > 0))
    else:
        ordered = sorted(pivots, key=lambda c: c[1], reverse=(dy > 0))
    new_pegs = set(pegs)
    new_pivots = set(pivots)
    new_markers = set(markers)
    new_pc = set(peg_carriers)
    new_mc = set(marker_carriers)
    ox, oy = origin
    qifbpmbolw = False
    pending_shift: Cell = (0, 0)
    for pv in ordered:
        target = (pv[0] + dx, pv[1] + dy)
        if target in new_pivots:
            continue
        if target not in rails:
            continue
        # Compute nybfuxmyrv per level (only for peg-carriers).
        shift: Cell = (0, 0)
        if pv in new_pc:
            if level_idx == 3:
                shift = (-dx * 8, 0)
            elif level_idx == 4:
                if pv[1] >= 11:
                    shift = (0, 0)
                else:
                    shift = (-dx * 6, -dy * 6)
            elif level_idx == 5:
                shift = (-dx * 6, 0)
            elif level_idx == 6:
                shift = (-dx * 6, 0)
            # L7+ has different rules; not modelled (L7 rarely needs
            # peg-carrying pivot moves).
        # Move the pivot (xxlsofmxas runs first, BEFORE the early-return).
        new_pivots.discard(pv)
        new_pivots.add(target)
        if pv in new_pc:
            new_pc.discard(pv)
            new_pc.add(target)
            new_pegs.discard(pv)
            new_pegs.add(target)
        if pv in new_mc:
            new_mc.discard(pv)
            new_mc.add(target)
            new_markers.discard(pv)
            new_markers.add(target)
        if target in new_pegs:
            new_pc.add(target)
        if target in new_markers:
            new_mc.add(target)
        if shift != (0, 0) and not qifbpmbolw:
            qifbpmbolw = True
            pending_shift = shift
            # Early-return: skip remaining pivots if scroll-east blocked.
            if level_idx != 5 and ox >= 5 and shift[0] > 0:
                return (
                    frozenset(new_pegs), frozenset(new_pivots),
                    frozenset(new_markers),
                    frozenset(new_pc), frozenset(new_mc), (ox, oy),
                )
    ox += pending_shift[0]
    oy += pending_shift[1]
    return (
        frozenset(new_pegs), frozenset(new_pivots),
        frozenset(new_markers),
        frozenset(new_pc), frozenset(new_mc), (ox, oy),
    )


def _initial_l3_memory() -> Dict[str, Any]:
    return {
        "pegs": _L3_INIT_PEGS,
        "pivots": _L3_INIT_PIVOTS,
        "markers": _L3_MARKERS,
        "marker_carriers": frozenset(_L3_MARKERS & _L3_INIT_PIVOTS),
        "origin": _L3_GRID_ORIGIN,
        "committed_moves": 0,
    }


def _initial_l4_memory() -> Dict[str, Any]:
    return {
        "pegs": _L4_INIT_PEGS,
        "pivots": _L4_INIT_PIVOTS,
        "markers": _L4_MARKERS,
        "marker_carriers": frozenset(_L4_MARKERS & _L4_INIT_PIVOTS),
        "origin": _L4_GRID_ORIGIN,
        "committed_moves": 0,
    }


def _initial_l5_memory() -> Dict[str, Any]:
    return {
        "pegs": _L5_INIT_PEGS,
        "pivots": _L5_INIT_PIVOTS,
        "markers": _L5_MARKERS,
        "marker_carriers": _L5_INIT_MARKER_CARRIERS,
        "origin": _L5_GRID_ORIGIN,
        "committed_moves": 0,
        "move_suffix": [],
    }


def _initial_l6_memory() -> Dict[str, Any]:
    extra_markers = _L6_MARKERS | _L6_INIT_RED_PEGS
    return {
        "pegs": _L6_INIT_NORMAL_PEGS,
        "pivots": _L6_INIT_PIVOTS,
        "markers": extra_markers,
        "marker_carriers": frozenset(extra_markers & _L6_INIT_PIVOTS),
        "origin": _L6_GRID_ORIGIN,
        "committed_moves": 0,
        "move_suffix": [],
    }


def _apply_lf52_move_to_memory(
    memory: Dict[str, Any],
    move: Move,
    *,
    level_idx: int,
    rails: FrozenSet[Cell],
) -> Dict[str, Any]:
    pegs: FrozenSet[Cell] = memory["pegs"]
    pivots: FrozenSet[Cell] = memory["pivots"]
    markers: FrozenSet[Cell] = memory["markers"]
    marker_carriers: FrozenSet[Cell] = memory["marker_carriers"]
    origin: Cell = memory["origin"]
    peg_carriers: FrozenSet[Cell] = frozenset(pegs & pivots)
    kind = move[0]
    if kind == "JUMP":
        _, src, dxy = move
        sx, sy = src
        dx, dy = dxy
        mid = (sx + dx, sy + dy)
        land = (sx + 2 * dx, sy + 2 * dy)
        new_pegs = set(pegs)
        new_pegs.discard(src)
        if mid in new_pegs:
            new_pegs.discard(mid)
        new_pegs.add(land)
        new_peg_carriers = set(peg_carriers)
        if src in new_peg_carriers:
            new_peg_carriers.discard(src)
        if mid in new_peg_carriers:
            new_peg_carriers.discard(mid)
        if land in pivots:
            new_peg_carriers.add(land)
        # Carrier identity is represented by peg/pivot cell overlap.
        pegs = frozenset(new_pegs)
        if level_idx == 4:
            if land == (7, 3):
                origin = (origin[0] - 30, origin[1])
            elif land == (12, 7):
                origin = (origin[0] + 15, origin[1] - 30)
        elif level_idx == 5:
            if land == (7, 3):
                origin = (origin[0] - 18, origin[1])
            elif land == (12, 3):
                origin = (origin[0] - 33, origin[1])
        elif level_idx == 6:
            if land == (7, 6) and origin == (5, 5):
                origin = (origin[0] - 20, origin[1])
            elif land == (18, 2) and origin == (-57, 5):
                origin = (origin[0] - 44, origin[1])
    elif kind == "RED_SOURCE_REMOVE":
        _, src, dxy = move
        sx, sy = src
        dx, dy = dxy
        mid = (sx + dx, sy + dy)
        new_pegs = set(pegs)
        new_pegs.discard(mid)
        pegs = frozenset(new_pegs)
    else:
        _, dxy = move
        (pegs, pivots, markers, _peg_carriers, marker_carriers, origin) = (
            _simulate_pivot_with_origin(
                pegs,
                pivots,
                markers,
                peg_carriers,
                marker_carriers,
                rails,
                dxy,
                origin,
                level_idx,
            )
        )
    move_suffix = list(memory.get("move_suffix") or [])
    if move_suffix and move_suffix[0] == move:
        move_suffix = move_suffix[1:]
    else:
        move_suffix = []
    return {
        "pegs": pegs,
        "pivots": pivots,
        "markers": markers,
        "marker_carriers": marker_carriers,
        "origin": origin,
        "committed_moves": int(memory.get("committed_moves", 0) or 0) + 1,
        "move_suffix": move_suffix,
    }


def _ensure_l3_memory() -> Dict[str, Any]:
    global _L3_MEMORY
    if _L3_MEMORY is None:
        _L3_MEMORY = _initial_l3_memory()
    return _L3_MEMORY


def _ensure_l4_memory() -> Dict[str, Any]:
    global _L4_MEMORY
    if _L4_MEMORY is None:
        _L4_MEMORY = _initial_l4_memory()
    return _L4_MEMORY


def _ensure_l5_memory() -> Dict[str, Any]:
    global _L5_MEMORY
    if _L5_MEMORY is None:
        _L5_MEMORY = _initial_l5_memory()
    return _L5_MEMORY


def _ensure_l6_memory() -> Dict[str, Any]:
    global _L6_MEMORY
    if _L6_MEMORY is None:
        _L6_MEMORY = _initial_l6_memory()
    return _L6_MEMORY


def _solve_l3_moves_from_memory(memory: Dict[str, Any]) -> Optional[List[Move]]:
    return _solve_pivot_peg_solitaire(
        memory["pegs"],
        memory["pivots"],
        memory["markers"],
        memory["marker_carriers"],
        _L3_RAILS,
        _L3_HOLES,
        level_idx=3,
        target_peg_count=1,
    )


def _solve_l4_moves_from_memory(memory: Dict[str, Any]) -> Optional[List[Move]]:
    return _solve_pivot_peg_solitaire(
        memory["pegs"],
        memory["pivots"],
        memory["markers"],
        memory["marker_carriers"],
        _L4_RAILS,
        _L4_HOLES,
        level_idx=4,
        target_peg_count=1,
    )


def _solve_l5_moves_from_memory(memory: Dict[str, Any]) -> Optional[List[Move]]:
    cached = list(memory.get("move_suffix") or [])
    if cached:
        return cached
    moves = _solve_pivot_peg_solitaire(
        memory["pegs"],
        memory["pivots"],
        memory["markers"],
        memory["marker_carriers"],
        _L5_RAILS,
        _L5_HOLES,
        level_idx=5,
        target_peg_count=1,
    )
    if moves:
        memory["move_suffix"] = list(moves)
    return moves


def _solve_l6_moves_from_memory(memory: Dict[str, Any]) -> Optional[List[Move]]:
    cached = list(memory.get("move_suffix") or [])
    if cached:
        return cached
    moves = _solve_pivot_peg_solitaire(
        memory["pegs"],
        memory["pivots"],
        memory["markers"],
        memory["marker_carriers"],
        _L6_RAILS,
        _L6_HOLES,
        level_idx=6,
        target_peg_count=1,
    )
    if moves:
        memory["move_suffix"] = list(moves)
    return moves


def _apply_pending_memory_commit(observed_signature: Optional[str]) -> None:
    global _L3_MEMORY, _L4_MEMORY, _L5_MEMORY, _L6_MEMORY, _PENDING_MEMORY_COMMIT
    if _PENDING_MEMORY_COMMIT is None:
        return
    signature, move = _PENDING_MEMORY_COMMIT
    if signature == "lf52_l3" and observed_signature in (None, "lf52_l3"):
        _L3_MEMORY = _apply_lf52_move_to_memory(
            _ensure_l3_memory(),
            move,
            level_idx=3,
            rails=_L3_RAILS,
        )
    elif signature == "lf52_l4" and observed_signature in (None, "lf52_l4"):
        _L4_MEMORY = _apply_lf52_move_to_memory(
            _ensure_l4_memory(),
            move,
            level_idx=4,
            rails=_L4_RAILS,
        )
    elif signature == "lf52_l5" and observed_signature in (None, "lf52_l5"):
        _L5_MEMORY = _apply_lf52_move_to_memory(
            _ensure_l5_memory(),
            move,
            level_idx=5,
            rails=_L5_RAILS,
        )
    elif signature == "lf52_l6" and observed_signature in (None, "lf52_l6"):
        _L6_MEMORY = _apply_lf52_move_to_memory(
            _ensure_l6_memory(),
            move,
            level_idx=6,
            rails=_L6_RAILS,
        )
    _PENDING_MEMORY_COMMIT = None


def _build_plan_from_moves(
    moves: List[Move],
    init_pegs: FrozenSet[Cell],
    init_pivots: FrozenSet[Cell],
    init_markers: FrozenSet[Cell],
    init_marker_carriers: FrozenSet[Cell],
    rails: FrozenSet[Cell],
    init_origin: Cell,
    level_idx: int,
    trace: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[GameAction, dict]]:
    pegs = init_pegs
    pivots = init_pivots
    markers = init_markers
    peg_carriers: FrozenSet[Cell] = frozenset(init_pegs & init_pivots)
    marker_carriers = init_marker_carriers
    ox, oy = init_origin

    plan: List[Tuple[GameAction, dict]] = []
    for move_index, move in enumerate(moves):
        kind = move[0]
        if kind == "JUMP":
            _, src, dxy = move
            sx, sy = src
            dx, dy = dxy
            origin_before = (ox, oy)
            peg_count_before = len(pegs)
            pcx = ox + sx * _TILE + _TILE // 2
            pcy = oy + sy * _TILE + _TILE // 2
            bcx = ox + sx * _TILE + dx * _BUTTON_OFFSET + _TILE // 2
            bcy = oy + sy * _TILE + dy * _BUTTON_OFFSET + _TILE // 2
            plan.append((GameAction.ACTION6, {"x": pcx, "y": pcy}))
            plan.append((GameAction.ACTION6, {"x": bcx, "y": bcy}))
            mid = (sx + dx, sy + dy)
            land = (sx + 2 * dx, sy + 2 * dy)
            new_pegs = set(pegs)
            new_pegs.discard(src)
            if mid in new_pegs:
                new_pegs.discard(mid)
            new_pegs.add(land)
            new_carriers = set(peg_carriers)
            if src in new_carriers:
                new_carriers.discard(src)
            if mid in new_carriers:
                new_carriers.discard(mid)
            if land in pivots:
                new_carriers.add(land)
            pegs = frozenset(new_pegs)
            peg_carriers = frozenset(new_carriers)
            # L4 LANDING-driven origin shifts (lf52.py:5433-5451).
            if level_idx == 4:
                if land == (7, 3):
                    ox -= 30
                elif land == (12, 7):
                    ox += 15
                    oy -= 30
            # L5 LANDING-driven origin shifts (lf52.py:5452-5470).
            elif level_idx == 5:
                if land == (7, 3):
                    ox -= 18
                elif land == (12, 3):
                    ox -= 33
            elif level_idx == 6:
                if land == (7, 6) and (ox, oy) == (5, 5):
                    ox -= 20
                elif land == (18, 2) and (ox, oy) == (-57, 5):
                    ox -= 44
            if trace is not None:
                trace.append({
                    "move_index": move_index,
                    "kind": "JUMP",
                    "src": [sx, sy],
                    "direction": [dx, dy],
                    "mid": [mid[0], mid[1]],
                    "land": [land[0], land[1]],
                    "origin_before": [origin_before[0], origin_before[1]],
                    "origin_after": [ox, oy],
                    "peg_count_before": peg_count_before,
                    "peg_count_after": len(pegs),
                    "action_count": 2,
                })
        elif kind == "RED_SOURCE_REMOVE":
            _, src, dxy = move
            sx, sy = src
            dx, dy = dxy
            origin_before = (ox, oy)
            peg_count_before = len(pegs)
            click_dx, click_dy = _L6_RED_SOURCE_CLICK_OFFSET
            pcx = ox + sx * _TILE + _TILE // 2 + click_dx
            pcy = oy + sy * _TILE + _TILE // 2 + click_dy
            bcx = pcx + dx * _BUTTON_OFFSET
            bcy = pcy + dy * _BUTTON_OFFSET
            plan.append((GameAction.ACTION6, {"x": pcx, "y": pcy}))
            plan.append((GameAction.ACTION6, {"x": bcx, "y": bcy}))
            mid = (sx + dx, sy + dy)
            new_pegs = set(pegs)
            new_pegs.discard(mid)
            pegs = frozenset(new_pegs)
            if trace is not None:
                trace.append({
                    "move_index": move_index,
                    "kind": "RED_SOURCE_REMOVE",
                    "src": [sx, sy],
                    "direction": [dx, dy],
                    "mid": [mid[0], mid[1]],
                    "origin_before": [origin_before[0], origin_before[1]],
                    "origin_after": [ox, oy],
                    "peg_count_before": peg_count_before,
                    "peg_count_after": len(pegs),
                    "action_count": 2,
                    "click_offset": [click_dx, click_dy],
                })
        else:
            _, dxy = move
            origin_before = (ox, oy)
            peg_count_before = len(pegs)
            plan.append((_ACTION_FOR_DIR[dxy], {}))
            (pegs, pivots, markers, peg_carriers, marker_carriers,
             (ox, oy)) = _simulate_pivot_with_origin(
                pegs, pivots, markers, peg_carriers, marker_carriers,
                rails, dxy, (ox, oy), level_idx,
            )
            if trace is not None:
                trace.append({
                    "move_index": move_index,
                    "kind": "PIVOT",
                    "direction": [dxy[0], dxy[1]],
                    "origin_before": [origin_before[0], origin_before[1]],
                    "origin_after": [ox, oy],
                    "peg_count_before": peg_count_before,
                    "peg_count_after": len(pegs),
                    "action_count": 1,
                    "peg_carrier_count": len(peg_carriers),
                    "marker_carrier_count": len(marker_carriers),
                })
    return plan


# -----------------------------------------------------------------------------
# Per-level entry points.
# -----------------------------------------------------------------------------

def _build_l2_plan_runtime(
    pegs: Optional[FrozenSet[Cell]] = None,
    pivots: Optional[FrozenSet[Cell]] = None,
) -> List[Tuple[GameAction, dict]]:
    current_pegs = pegs if pegs is not None else _L2_INIT_PEGS
    current_pivots = pivots if pivots is not None else _L2_INIT_PIVOTS
    init_marker_carriers = frozenset(_L2_MARKERS & current_pivots)
    moves = _solve_pivot_peg_solitaire(
        current_pegs, current_pivots, _L2_MARKERS,
        init_marker_carriers, _L2_RAILS, _L2_HOLES,
        level_idx=2, target_peg_count=1,
    )
    if moves is None:
        return []
    return _build_plan_from_moves(
        moves, current_pegs, current_pivots, _L2_MARKERS,
        init_marker_carriers, _L2_RAILS, _L2_GRID_ORIGIN, 2,
    )


def _build_l3_plan_runtime(
    trace: Optional[List[Dict[str, Any]]] = None,
    pegs: Optional[FrozenSet[Cell]] = None,
    pivots: Optional[FrozenSet[Cell]] = None,
    markers: Optional[FrozenSet[Cell]] = None,
    marker_carriers: Optional[FrozenSet[Cell]] = None,
    origin: Optional[Cell] = None,
) -> List[Tuple[GameAction, dict]]:
    current_pegs = pegs if pegs is not None else _L3_INIT_PEGS
    current_pivots = pivots if pivots is not None else _L3_INIT_PIVOTS
    current_markers = markers if markers is not None else _L3_MARKERS
    current_marker_carriers = (
        marker_carriers
        if marker_carriers is not None
        else frozenset(current_markers & current_pivots)
    )
    current_origin = origin if origin is not None else _L3_GRID_ORIGIN
    moves = _solve_pivot_peg_solitaire(
        current_pegs, current_pivots, current_markers,
        current_marker_carriers, _L3_RAILS, _L3_HOLES,
        level_idx=3, target_peg_count=1,
    )
    if moves is None:
        return []
    return _build_plan_from_moves(
        moves, current_pegs, current_pivots, current_markers,
        current_marker_carriers, _L3_RAILS, current_origin, 3,
        trace=trace,
    )


def _build_l4_plan_runtime() -> List[Tuple[GameAction, dict]]:
    init_marker_carriers = frozenset(_L4_MARKERS & _L4_INIT_PIVOTS)
    moves = _solve_pivot_peg_solitaire(
        _L4_INIT_PEGS, _L4_INIT_PIVOTS, _L4_MARKERS,
        init_marker_carriers, _L4_RAILS, _L4_HOLES,
        level_idx=4, target_peg_count=1,
    )
    if moves is None:
        return []
    return _build_plan_from_moves(
        moves, _L4_INIT_PEGS, _L4_INIT_PIVOTS, _L4_MARKERS,
        init_marker_carriers, _L4_RAILS, _L4_GRID_ORIGIN, 4,
    )


def _build_l5_plan_runtime() -> List[Tuple[GameAction, dict]]:
    moves = _solve_pivot_peg_solitaire(
        _L5_INIT_PEGS, _L5_INIT_PIVOTS, _L5_MARKERS,
        _L5_INIT_MARKER_CARRIERS, _L5_RAILS, _L5_HOLES,
        level_idx=5, target_peg_count=1,
    )
    if moves is None:
        return []
    return _build_plan_from_moves(
        moves, _L5_INIT_PEGS, _L5_INIT_PIVOTS, _L5_MARKERS,
        _L5_INIT_MARKER_CARRIERS, _L5_RAILS, _L5_GRID_ORIGIN, 5,
    )


def _build_l6_plan_runtime() -> List[Tuple[GameAction, dict]]:
    """L6 has a red peg that's only consumable by another red peg.

    With only 1 red peg, it can never be the consumer (would need to
    jump OVER another peg with same name). It CAN be a middle for
    normal-peg jumps but won't be consumed. So the red peg counts as
    a fozwvlovdui peg for win condition (peg count == 2 needed) but
    can never be REMOVED.

    With 7 normal + 1 red = 8 pegs initially, target == 2 means:
    - red peg always remains (1 peg)
    - need to reduce normal pegs from 7 to 1 = 6 jumps consuming pegs
    """
    # Treat red peg specially: it's a peg for count purposes but
    # cannot be consumed. Encode this by making it a "marker"
    # internally — middle-jumpable but not consumed. Then only normal
    # pegs are consumed; win when total pegs (normal + red) == 2.
    # So normal pegs must drop to 1 (red stays as 1).
    memory = _initial_l6_memory()
    moves = _solve_l6_moves_from_memory(memory)
    if moves is None:
        return []
    return _build_plan_from_moves(
        moves, memory["pegs"], memory["pivots"], memory["markers"],
        memory["marker_carriers"], _L6_RAILS, memory["origin"], 6,
    )


def _build_l7_plan_runtime() -> List[Tuple[GameAction, dict]]:
    """L7 has 2 normal + 1 red = 3 pegs. Win at peg count 2.

    Red peg counts but isn't consumable. So we need to consume 1 normal
    peg → leaves 1 normal + 1 red = 2 total. Win.
    """
    extra_markers = _L7_MARKERS | _L7_INIT_RED_PEGS
    init_pegs = _L7_INIT_NORMAL_PEGS
    init_marker_carriers = frozenset(extra_markers & _L7_INIT_PIVOTS)
    moves = _solve_pivot_peg_solitaire(
        init_pegs, _L7_INIT_PIVOTS, extra_markers,
        init_marker_carriers, _L7_RAILS, _L7_HOLES,
        level_idx=7, target_peg_count=1,  # 1 normal + 1 red = 2 total
        max_states=50_000,
    )
    if moves is None:
        return []
    return _build_plan_from_moves(
        moves, init_pegs, _L7_INIT_PIVOTS, extra_markers,
        init_marker_carriers, _L7_RAILS, _L7_GRID_ORIGIN, 7,
    )


def plan_next_actions_from_state(state: dict) -> List[Tuple[GameAction, dict]]:
    global _LAST_L3_FIRST_MOVE, _LAST_L4_FIRST_MOVE, _LAST_L5_FIRST_MOVE, _LAST_L6_FIRST_MOVE
    sig = state.get("level_signature")
    if sig == "lf52_l1":
        plan = _build_l1_plan_from_state(state)
        _record_plan_source("runtime_bfs_lf52_l1", {
            "level_signature": sig,
            "plan_len": len(plan),
            "peg_count": len(state.get("l1_pegs") or ()),
        })
        return plan
    if sig == "lf52_l2":
        pegs = frozenset(tuple(cell) for cell in state.get("l2_pegs") or _L2_INIT_PEGS)
        pivots = frozenset(tuple(cell) for cell in state.get("l2_pivots") or _L2_INIT_PIVOTS)
        plan = _build_l2_plan_runtime(pegs=pegs, pivots=pivots)
        _record_plan_source("runtime_bfs_lf52_l2", {
            "level_signature": sig,
            "plan_len": len(plan),
            "peg_count": len(pegs),
            "pivot_count": len(pivots),
        })
        return plan
    if sig == "lf52_l3":
        trace: List[Dict[str, Any]] = []
        memory = _ensure_l3_memory()
        moves = _solve_l3_moves_from_memory(memory)
        _LAST_L3_FIRST_MOVE = moves[0] if moves else None
        plan = []
        if moves is not None:
            plan = _build_plan_from_moves(
                moves,
                memory["pegs"],
                memory["pivots"],
                memory["markers"],
                memory["marker_carriers"],
                _L3_RAILS,
                memory["origin"],
                3,
                trace=trace,
            )
        origin_shifts = [
            step for step in trace
            if step.get("origin_before") != step.get("origin_after")
        ]
        _record_plan_source("runtime_bfs_lf52_l3", {
            "level_signature": sig,
            "plan_len": len(plan),
            "move_count": len(trace),
            "memory_backed": bool(state.get("memory_backed")),
            "committed_moves": int(memory.get("committed_moves", 0) or 0),
            "peg_count": len(memory["pegs"]),
            "pivot_count": len(memory["pivots"]),
            "origin_shift_count": len(origin_shifts),
            "final_origin": trace[-1].get("origin_after") if trace else None,
            "origin_trace": trace,
        })
        return plan
    if sig == "lf52_l4":
        trace: List[Dict[str, Any]] = []
        memory = _ensure_l4_memory()
        moves = _solve_l4_moves_from_memory(memory)
        _LAST_L4_FIRST_MOVE = moves[0] if moves else None
        plan = []
        if moves is not None:
            plan = _build_plan_from_moves(
                moves,
                memory["pegs"],
                memory["pivots"],
                memory["markers"],
                memory["marker_carriers"],
                _L4_RAILS,
                memory["origin"],
                4,
                trace=trace,
            )
        origin_shifts = [
            step for step in trace
            if step.get("origin_before") != step.get("origin_after")
        ]
        _record_plan_source("runtime_bfs_lf52_l4", {
            "level_signature": sig,
            "plan_len": len(plan),
            "move_count": len(trace),
            "memory_backed": bool(state.get("memory_backed")),
            "committed_moves": int(memory.get("committed_moves", 0) or 0),
            "peg_count": len(memory["pegs"]),
            "pivot_count": len(memory["pivots"]),
            "origin_shift_count": len(origin_shifts),
            "final_origin": trace[-1].get("origin_after") if trace else None,
        })
        return plan
    if sig == "lf52_l5":
        trace: List[Dict[str, Any]] = []
        memory = _ensure_l5_memory()
        moves = _solve_l5_moves_from_memory(memory)
        _LAST_L5_FIRST_MOVE = moves[0] if moves else None
        plan = []
        if moves is not None:
            plan = _build_plan_from_moves(
                moves,
                memory["pegs"],
                memory["pivots"],
                memory["markers"],
                memory["marker_carriers"],
                _L5_RAILS,
                memory["origin"],
                5,
                trace=trace,
            )
        origin_shifts = [
            step for step in trace
            if step.get("origin_before") != step.get("origin_after")
        ]
        _record_plan_source("runtime_bfs_lf52_l5", {
            "level_signature": sig,
            "plan_len": len(plan),
            "move_count": len(trace),
            "memory_backed": bool(state.get("memory_backed")),
            "committed_moves": int(memory.get("committed_moves", 0) or 0),
            "peg_count": len(memory["pegs"]),
            "pivot_count": len(memory["pivots"]),
            "move_suffix_len": len(memory.get("move_suffix") or ()),
            "origin_shift_count": len(origin_shifts),
            "final_origin": trace[-1].get("origin_after") if trace else None,
        })
        return plan
    if sig == "lf52_l6":
        trace: List[Dict[str, Any]] = []
        memory = _ensure_l6_memory()
        moves = _solve_l6_moves_from_memory(memory)
        _LAST_L6_FIRST_MOVE = moves[0] if moves else None
        plan = []
        if moves is not None:
            plan = _build_plan_from_moves(
                moves,
                memory["pegs"],
                memory["pivots"],
                memory["markers"],
                memory["marker_carriers"],
                _L6_RAILS,
                memory["origin"],
                6,
                trace=trace,
            )
        origin_shifts = [
            step for step in trace
            if step.get("origin_before") != step.get("origin_after")
        ]
        _record_plan_source("runtime_bfs_lf52_l6", {
            "level_signature": sig,
            "plan_len": len(plan),
            "move_count": len(trace),
            "memory_backed": bool(state.get("memory_backed")),
            "committed_moves": int(memory.get("committed_moves", 0) or 0),
            "peg_count": len(memory["pegs"]),
            "pivot_count": len(memory["pivots"]),
            "move_suffix_len": len(memory.get("move_suffix") or ()),
            "origin_shift_count": len(origin_shifts),
            "final_origin": trace[-1].get("origin_after") if trace else None,
        })
        return plan
    if sig == "lf52_l7":
        plan = _build_l7_plan_runtime()
        _record_plan_source("runtime_bfs_lf52_l7", {
            "level_signature": sig,
            "plan_len": len(plan),
        })
        return plan
    _record_plan_source("unrecognized_lf52_state", {
        "level_signature": sig,
        "plan_len": 0,
    })
    return []


def perceive_and_plan(frame: Any) -> List[Tuple[GameAction, dict]]:
    global _PENDING_MEMORY_COMMIT
    state = perceive_lf52(frame)
    sig = state.get("level_signature")
    _apply_pending_memory_commit(sig)
    if _PENDING_MACRO_TAIL and (sig is None or sig == _PENDING_MACRO_TAIL_SIGNATURE):
        pending = _pop_click_pair_macro_tail()
        if pending:
            return pending
    if _PENDING_MACRO_TAIL:
        _clear_click_pair_macro_tail()
    observed_sig = sig
    if _L5_MEMORY is not None and sig in (None, "lf52_l4", "lf52_l5"):
        state = {
            "level_signature": "lf52_l5",
            "memory_backed": sig != "lf52_l5",
            "observed_signature": observed_sig,
        }
        sig = "lf52_l5"
    elif _L4_MEMORY is not None and sig in (None, "lf52_l4"):
        state = {
            "level_signature": "lf52_l4",
            "memory_backed": sig != "lf52_l4",
            "observed_signature": observed_sig,
        }
        sig = "lf52_l4"
    elif _L3_MEMORY is not None and sig in (None, "lf52_l3"):
        state = {
            "level_signature": "lf52_l3",
            "memory_backed": sig != "lf52_l3",
            "observed_signature": observed_sig,
        }
        sig = "lf52_l3"
    elif _L6_MEMORY is not None and sig in (None, "lf52_l6"):
        state = {
            "level_signature": "lf52_l6",
            "memory_backed": sig != "lf52_l6",
            "observed_signature": observed_sig,
        }
        sig = "lf52_l6"
    if sig not in (None, "lf52_l3"):
        _clear_l3_memory()
    if sig not in (None, "lf52_l4"):
        _clear_l4_memory()
    if sig not in (None, "lf52_l5"):
        _clear_l5_memory()
    if sig not in (None, "lf52_l6"):
        _clear_l6_memory()
    if sig is None:
        return plan_next_actions_from_state(state)
    else:
        _clear_click_pair_macro_tail()
    plan = plan_next_actions_from_state(state)
    if sig == "lf52_l3":
        first_move = _LAST_L3_FIRST_MOVE
    elif sig == "lf52_l4":
        first_move = _LAST_L4_FIRST_MOVE
    elif sig == "lf52_l5":
        first_move = _LAST_L5_FIRST_MOVE
    elif sig == "lf52_l6":
        first_move = _LAST_L6_FIRST_MOVE
    else:
        first_move = None
    if first_move is not None and first_move[0] == "PIVOT":
        _PENDING_MEMORY_COMMIT = (sig, first_move)
        _prime_click_pair_macro_tail(plan, sig)
    else:
        _prime_click_pair_macro_tail(plan, sig, commit_move=first_move)
    return plan
