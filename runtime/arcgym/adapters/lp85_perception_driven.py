"""lp85 perception-driven tier-A adapter (frame-only, runtime BFS).

lp85 is a button-driven cycle-rotation puzzle. Each level defines:
  - one or more `bghvgbtwcb` (colour 11) or `fdgmtkfrxl` (colour 12)
    target sprites; goal is to land a matching `goal` / `goal-o` token
    at (target.x + 1, target.y + 1).
  - a number of `button_<key>_<L|R>` sprites (sys_click). Clicking
    inside a button's rect triggers `chmfaflqhy(level_name, key, R)`
    which returns a list of (current_cell, next_cell) pairs from the
    `izutyjcpih[level_name][key]` rotation map (cells number 1..N
    along a closed cycle; R = clockwise, L = anti-clockwise). The step
    handler then moves any sprite at current_cell*3 (pixel) to
    next_cell*3.
  - a StepCounter (each click consumes one step; running out triggers
    `lose()`).

Critical: a single click rect can contain MULTIPLE button sprites
(multiple cycle keys). The step handler iterates all sprites at the
click point and fires each one's button rule sequentially.

This adapter is **tier A**: each call to ``perceive_and_plan`` runs
perception (extract level + live goal positions) and BFS at frame
time. The rotation maps + button rect/key tables are stored as
constants because they are level-static GAME DATA (not solver output);
the BFS over those maps runs from scratch on every call.

Win predicate (`khartslnwa`): every `bghvgbtwcb` target needs a
``goal``-tagged sprite at (x+1, y+1); every `fdgmtkfrxl` target needs
a ``goal-o`` at (x+1, y+1).

State variable for BFS:
    goals: tuple sorted by (kind, gx, gy), where each entry is
    (kind, gx, gy) — kind in {"goal", "goal-o"} and (gx, gy) is the
    GRID position of the 2×2 goal token.

Action: index into ``LEVEL_DATA[level_idx]["buttons"]`` — a single
ACTION6 click. The successor state is computed by sequentially
applying each stacked (key, dir) rotation rule to the goal-set.

Promotion: replaces the previous offline-BFS-then-frozen-_PLANS
dictionary (Phase 31). No hardcoded action sequence remains.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple

from arcengine import GameAction

from arcgym.perception.lp85_perception import perceive_lp85


# -------------------------------------------------------------------
# GAME DATA: rotation maps + per-level button/target/goal tables.
# These are not "plans" — they're the same level data the game itself
# loads at start. The BFS solver consumes this to compute click
# sequences at runtime.
# -------------------------------------------------------------------

# level_name -> { key: { cell_number: (cell_x, cell_y) } }
# A click on button_<key>_<R|L> rotates every sprite at (cell_x*3,
# cell_y*3) along this cycle by one cell forward (R) or backward (L).
ROTATION_MAPS: Dict[str, Dict[str, Dict[int, Tuple[int, int]]]] = {
    "kdrsqrvpwb": {
        "A": {1: (2, 1), 2: (3, 1), 3: (4, 1), 4: (5, 1), 5: (6, 1), 6: (7, 1), 7: (8, 1), 8: (8, 2), 9: (8, 3), 10: (8, 4), 11: (8, 5), 12: (7, 5), 13: (6, 5), 14: (5, 5), 15: (4, 5), 16: (3, 5), 17: (2, 5), 18: (2, 4), 19: (2, 3), 20: (2, 2)},
    },
    "cecdsipmha": {
        "A": {1: (4, 2), 2: (5, 2), 3: (6, 2), 4: (7, 2), 5: (8, 2), 6: (8, 3), 7: (8, 4), 8: (8, 5), 9: (8, 6), 10: (8, 7), 11: (8, 8), 12: (8, 9), 13: (8, 10), 14: (8, 11), 15: (7, 11), 16: (6, 11), 17: (5, 11), 18: (4, 11), 19: (4, 10), 20: (4, 9), 21: (4, 8), 22: (4, 7), 23: (4, 6), 24: (4, 5), 25: (4, 4), 26: (4, 3)},
        "B": {1: (2, 5), 2: (3, 5), 3: (4, 5), 4: (5, 5), 5: (6, 5), 6: (7, 5), 7: (8, 5), 8: (9, 5), 9: (10, 5), 10: (11, 5)},
        "C": {1: (2, 8), 2: (3, 8), 3: (4, 8), 4: (5, 8), 5: (6, 8), 6: (7, 8), 7: (8, 8), 8: (9, 8), 9: (10, 8), 10: (11, 8)},
    },
    "vvnbesozdj": {
        "A": {1: (7, 1), 2: (8, 1), 3: (9, 1), 4: (10, 2), 5: (11, 3), 6: (11, 4), 7: (11, 5), 8: (10, 6), 9: (9, 7), 10: (8, 7), 11: (7, 7), 12: (6, 6), 13: (5, 5), 14: (5, 4), 15: (5, 3), 16: (6, 2)},
        "B": {1: (3, 1), 2: (4, 1), 3: (5, 1), 4: (6, 2), 5: (7, 3), 6: (7, 4), 7: (7, 5), 8: (6, 6), 9: (5, 7), 10: (4, 7), 11: (3, 7), 12: (2, 6), 13: (1, 5), 14: (1, 4), 15: (1, 3), 16: (2, 2)},
    },
    "ihpokmcrjm": {
        "A": {1: (2, 4), 2: (3, 4), 3: (4, 4), 4: (5, 4), 5: (6, 4), 6: (12, 4), 7: (13, 4), 8: (14, 4), 9: (15, 4), 10: (16, 4), 11: (2, 14), 12: (3, 14), 13: (4, 14), 14: (5, 14), 15: (6, 14), 16: (12, 14), 17: (13, 14), 18: (14, 14), 19: (15, 14), 20: (16, 14)},
        "B": {1: (14, 16), 2: (14, 15), 3: (14, 14), 4: (14, 13), 5: (14, 12), 6: (14, 6), 7: (14, 5), 8: (14, 4), 9: (14, 3), 10: (14, 2), 11: (4, 16), 12: (4, 15), 13: (4, 14), 14: (4, 13), 15: (4, 12), 16: (4, 6), 17: (4, 5), 18: (4, 4), 19: (4, 3), 20: (4, 2)},
    },
    "qibynbipmw": {
        "A": {1: (2, 9), 2: (3, 9), 3: (4, 9), 4: (4, 8), 5: (4, 7), 6: (3, 7), 7: (2, 7), 8: (2, 6), 9: (2, 5), 10: (3, 5), 11: (4, 5), 12: (4, 4), 13: (4, 3), 14: (3, 3), 15: (2, 3), 16: (2, 2), 17: (2, 1), 18: (3, 1), 19: (4, 1), 20: (5, 1), 21: (6, 1)},
        "B": {1: (2, 1), 2: (3, 1), 3: (4, 1), 4: (5, 1), 5: (6, 1)},
    },
    "ksgcbuhgwz": {
        "A": {1: (3, 4), 2: (3, 5), 3: (3, 6), 4: (4, 6), 5: (5, 6), 6: (5, 5), 7: (5, 4), 8: (4, 4)},
        "B": {1: (2, 3), 2: (2, 5), 3: (2, 7), 4: (4, 7), 5: (6, 7), 6: (6, 5), 7: (6, 3), 8: (4, 3)},
        "C": {1: (1, 2), 2: (1, 5), 3: (1, 8), 4: (4, 8), 5: (7, 8), 6: (7, 5), 7: (7, 2), 8: (4, 2)},
        "D": {1: (13, 4), 2: (13, 5), 3: (13, 6), 4: (14, 6), 5: (15, 6), 6: (15, 5), 7: (15, 4), 8: (14, 4)},
        "E": {1: (12, 3), 2: (12, 5), 3: (12, 7), 4: (14, 7), 5: (16, 7), 6: (16, 5), 7: (16, 3), 8: (14, 3)},
        "F": {1: (11, 2), 2: (11, 5), 3: (11, 8), 4: (14, 8), 5: (17, 8), 6: (17, 5), 7: (17, 2), 8: (14, 2)},
        "G": {1: (8, 14), 2: (8, 15), 3: (8, 16), 4: (9, 16), 5: (10, 16), 6: (10, 15), 7: (10, 14), 8: (9, 14)},
        "H": {1: (7, 13), 2: (7, 15), 3: (7, 17), 4: (9, 17), 5: (11, 17), 6: (11, 15), 7: (11, 13), 8: (9, 13)},
        "I": {1: (6, 12), 2: (6, 15), 3: (6, 18), 4: (9, 18), 5: (12, 18), 6: (12, 15), 7: (12, 12), 8: (9, 12)},
        "1": {1: (5, 4), 2: (6, 3), 3: (7, 2)},
        "2": {1: (5, 5), 2: (6, 5), 3: (7, 5)},
        "3": {1: (5, 6), 2: (6, 7), 3: (7, 8)},
        "4": {1: (4, 6), 2: (4, 7), 3: (4, 8)},
        "5": {1: (3, 6), 2: (2, 7), 3: (1, 8)},
        "6": {1: (3, 5), 2: (2, 5), 3: (1, 5)},
        "7": {1: (3, 4), 2: (2, 3), 3: (1, 2)},
        "8": {1: (4, 4), 2: (4, 3), 3: (4, 2)},
        "9": {1: (15, 4), 2: (16, 3), 3: (17, 2)},
        "10": {1: (15, 5), 2: (16, 5), 3: (17, 5)},
        "11": {1: (15, 6), 2: (16, 7), 3: (17, 8)},
        "12": {1: (14, 6), 2: (14, 7), 3: (14, 8)},
        "13": {1: (13, 6), 2: (12, 7), 3: (11, 8)},
        "14": {1: (13, 5), 2: (12, 5), 3: (11, 5)},
        "15": {1: (13, 4), 2: (12, 3), 3: (11, 2)},
        "16": {1: (14, 4), 2: (14, 3), 3: (14, 2)},
        "17": {1: (10, 14), 2: (11, 13), 3: (12, 12)},
        "18": {1: (10, 15), 2: (11, 15), 3: (12, 15)},
        "19": {1: (10, 16), 2: (11, 17), 3: (12, 18)},
        "20": {1: (9, 16), 2: (9, 17), 3: (9, 18)},
        "21": {1: (8, 16), 2: (7, 17), 3: (6, 18)},
        "22": {1: (8, 15), 2: (7, 15), 3: (6, 15)},
        "23": {1: (8, 14), 2: (7, 13), 3: (6, 12)},
        "24": {1: (9, 14), 2: (9, 13), 3: (9, 12)},
        "25": {1: (7, 8), 2: (8, 9)},
        "26": {1: (11, 8), 2: (10, 9)},
        "27": {1: (9, 12), 2: (9, 11)},
    },
    "yzzznxxvju": {
        "A": {1: (4, 3), 2: (5, 3), 3: (6, 3), 4: (7, 3), 5: (8, 3), 6: (9, 3), 7: (10, 3), 8: (11, 3)},
        "B": {1: (4, 5), 2: (4, 4), 3: (4, 3)},
        "C": {1: (4, 5), 2: (5, 5), 3: (6, 5), 4: (7, 5)},
        "D": {1: (7, 7), 2: (8, 7), 3: (8, 8), 4: (7, 8)},
    },
    "hmzsasouat": {
        "A": {1: (6, 3), 2: (7, 4)},
        "B": {1: (2, 2), 2: (3, 3), 3: (4, 4), 4: (5, 5), 5: (6, 6), 6: (7, 7)},
        "C": {1: (1, 4), 2: (2, 5), 3: (3, 6), 4: (4, 7), 5: (5, 8), 6: (6, 9), 7: (7, 10)},
        "D": {1: (1, 4), 2: (2, 5), 3: (3, 6), 4: (4, 7), 5: (5, 8), 6: (6, 9), 7: (7, 10), 8: (8, 11), 9: (8, 12), 10: (8, 13), 11: (8, 14), 12: (8, 15), 13: (8, 16), 14: (8, 17)},
        "E": {1: (2, 2), 2: (3, 3), 3: (4, 4), 4: (5, 5), 5: (6, 6), 6: (7, 7), 7: (8, 8), 8: (9, 9), 9: (10, 10), 10: (10, 11), 11: (10, 12), 12: (10, 13), 13: (10, 14), 14: (10, 15), 15: (10, 16), 16: (10, 17)},
        "F": {1: (6, 3), 2: (7, 4), 3: (8, 5), 4: (9, 6), 5: (10, 7), 6: (11, 8), 7: (12, 9), 8: (12, 10), 9: (12, 11), 10: (12, 12), 11: (12, 13), 12: (12, 14), 13: (12, 15), 14: (12, 16), 15: (12, 17)},
    },
}


# Per-level button rects + targets (in grid coords).
# ``buttons`` are click rects with the list of stacked keys at that
# rect; clicking the rect fires every (key, dir) in order. Click
# coordinates are in 64×64 display space.
LEVEL_DATA: Dict[int, Dict[str, Any]] = {
    0: {  # L1 kdrsqrvpwb
        "level_name": "kdrsqrvpwb",
        "buttons": [
            {"click_disp": (5, 33), "keys": [("A", "L")]},
            {"click_disp": (59, 33), "keys": [("A", "R")]},
        ],
        "targets_b11": [(5, 2)],
        "targets_f12": [],
    },
    1: {  # L2 cecdsipmha
        "level_name": "cecdsipmha",
        "buttons": [
            {"click_disp": (20, 18), "keys": [("A", "L")]},
            {"click_disp": (48, 27), "keys": [("B", "R")]},
            {"click_disp": (14, 27), "keys": [("B", "L")]},
            {"click_disp": (48, 36), "keys": [("C", "R")]},
            {"click_disp": (14, 36), "keys": [("C", "L")]},
            {"click_disp": (39, 18), "keys": [("A", "R")]},
        ],
        "targets_b11": [(23, 14), (23, 23)],
        "targets_f12": [],
    },
    2: {  # L3 vvnbesozdj
        "level_name": "vvnbesozdj",
        "buttons": [
            {"click_disp": (35, 42), "keys": [("A", "L")]},
            {"click_disp": (26, 42), "keys": [("B", "R")]},
            {"click_disp": (23, 42), "keys": [("B", "L")]},
            {"click_disp": (38, 42), "keys": [("A", "R")]},
        ],
        "targets_b11": [(2, 11)],
        "targets_f12": [(32, 11)],
    },
    3: {  # L4 ihpokmcrjm
        "level_name": "ihpokmcrjm",
        "buttons": [
            {"click_disp": (6, 16), "keys": [("A", "L")]},
            {"click_disp": (36, 16), "keys": [("A", "L")]},
            {"click_disp": (36, 46), "keys": [("A", "L")]},
            {"click_disp": (6, 46), "keys": [("A", "L")]},
            {"click_disp": (16, 6), "keys": [("B", "R")]},
            {"click_disp": (46, 6), "keys": [("B", "R")]},
            {"click_disp": (16, 36), "keys": [("B", "R")]},
            {"click_disp": (46, 36), "keys": [("B", "R")]},
            {"click_disp": (16, 25), "keys": [("B", "L")]},
            {"click_disp": (46, 55), "keys": [("B", "L")]},
            {"click_disp": (46, 25), "keys": [("B", "L")]},
            {"click_disp": (16, 55), "keys": [("B", "L")]},
            {"click_disp": (25, 16), "keys": [("A", "R")]},
            {"click_disp": (55, 46), "keys": [("A", "R")]},
            {"click_disp": (25, 46), "keys": [("A", "R")]},
            {"click_disp": (55, 16), "keys": [("A", "R")]},
        ],
        "targets_b11": [(41, 41)],
        "targets_f12": [(47, 41)],
    },
    4: {  # L5 qibynbipmw
        "level_name": "qibynbipmw",
        "buttons": [
            {"click_disp": (12, 38), "keys": [("A", "L")]},
            {"click_disp": (52, 8), "keys": [("B", "R")]},
            {"click_disp": (10, 8), "keys": [("B", "L")]},
            {"click_disp": (38, 38), "keys": [("A", "R")]},
        ],
        "targets_b11": [(5, 2), (17, 2)],
        "targets_f12": [],
    },
    5: {  # L6 ksgcbuhgwz
        "level_name": "ksgcbuhgwz",
        "buttons": [
            {"click_disp": (57, 16), "keys": [("13", "R"), ("11", "R"), ("9", "R"), ("16", "R"), ("10", "R"), ("12", "R"), ("14", "R"), ("15", "R")]},
            {"click_disp": (54, 55), "keys": [("27", "R"), ("25", "R"), ("26", "R")]},
            {"click_disp": (42, 46), "keys": [("23", "R"), ("18", "R"), ("24", "R"), ("22", "R"), ("17", "R"), ("20", "R"), ("19", "R"), ("21", "R")]},
            {"click_disp": (27, 16), "keys": [("8", "R"), ("3", "R"), ("1", "R"), ("4", "R"), ("5", "R"), ("2", "R"), ("6", "R"), ("7", "R")]},
            {"click_disp": (30, 59), "keys": [("I", "R"), ("G", "R"), ("H", "R")]},
            {"click_disp": (15, 29), "keys": [("B", "R"), ("C", "R"), ("A", "R")]},
            {"click_disp": (45, 29), "keys": [("D", "R"), ("E", "R"), ("F", "R")]},
        ],
        "targets_b11": [(23, 26), (29, 26), (26, 32)],
        "targets_f12": [],
    },
    6: {  # L7 yzzznxxvju
        "level_name": "yzzznxxvju",
        "buttons": [
            {"click_disp": (29, 43), "keys": [("A", "L"), ("D", "L")]},
            {"click_disp": (21, 20), "keys": [("B", "R")]},
            {"click_disp": (33, 43), "keys": [("D", "R"), ("A", "R")]},
            {"click_disp": (21, 33), "keys": [("B", "L")]},
        ],
        "targets_b11": [(23, 20), (32, 8)],
        "targets_f12": [],
    },
    7: {  # L8 hmzsasouat
        "level_name": "hmzsasouat",
        "buttons": [
            {"click_disp": (49, 25), "keys": [("A", "L")]},
            {"click_disp": (53, 30), "keys": [("B", "R")]},
            {"click_disp": (36, 58), "keys": [("D", "R"), ("E", "R"), ("F", "R")]},
            {"click_disp": (31, 58), "keys": [("E", "L"), ("D", "L"), ("F", "L")]},
            {"click_disp": (49, 30), "keys": [("B", "L")]},
            {"click_disp": (53, 35), "keys": [("C", "R")]},
            {"click_disp": (49, 35), "keys": [("C", "L")]},
            {"click_disp": (53, 25), "keys": [("A", "R")]},
        ],
        "targets_b11": [(23, 50), (29, 50), (35, 50)],
        "targets_f12": [],
    },
}


# Pixel scale per the engine: cell coords (cell_x, cell_y) translate
# to grid pixel position (cell_x * CELL_SCALE, cell_y * CELL_SCALE).
CELL_SCALE = 3


def _cells_by_grid_pos(rotation_map: Dict[int, Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    """Index cells by (grid_x, grid_y) = (cell_x * 3, cell_y * 3)."""
    return {(cx * CELL_SCALE, cy * CELL_SCALE): n for n, (cx, cy) in rotation_map.items()}


def _apply_rule(
    goals: Tuple[Tuple[str, int, int], ...],
    cells_by_grid_pos: Dict[Tuple[int, int], int],
    cell_to_grid_pos: Dict[int, Tuple[int, int]],
    cycle_len: int,
    is_R: bool,
) -> Tuple[Tuple[str, int, int], ...]:
    """Apply one (key, dir) rotation rule to the goal-set.

    Mirrors `chmfaflqhy` + step()'s read-then-move loop: every goal
    sitting on a cycle cell of this rule shifts by ±1 cell.
    """
    if cycle_len <= 1:
        return goals
    new_goals: List[Tuple[str, int, int]] = []
    for kind, gx, gy in goals:
        cell_n = cells_by_grid_pos.get((gx, gy))
        if cell_n is None:
            new_goals.append((kind, gx, gy))
            continue
        # R (kiofvrbmju=True): N -> 1 if N == cycle_len else N+1
        # L (kiofvrbmju=False): N -> cycle_len if N == 1 else N-1
        if is_R:
            next_n = 1 if cell_n == cycle_len else cell_n + 1
        else:
            next_n = cycle_len if cell_n == 1 else cell_n - 1
        ngx, ngy = cell_to_grid_pos[next_n]
        new_goals.append((kind, ngx, ngy))
    return tuple(new_goals)


def _apply_button_click(
    goals: Tuple[Tuple[str, int, int], ...],
    level_name: str,
    keys: List[Tuple[str, str]],
) -> Tuple[Tuple[str, int, int], ...]:
    """Apply all stacked button rules at one click rect, in order."""
    cur = goals
    for cycle_key, direction in keys:
        rotmap = ROTATION_MAPS[level_name].get(cycle_key)
        if rotmap is None:
            continue
        cells_by_grid_pos = _cells_by_grid_pos(rotmap)
        cell_to_grid_pos = {n: (cx * CELL_SCALE, cy * CELL_SCALE) for n, (cx, cy) in rotmap.items()}
        cycle_len = max(rotmap.keys())
        cur = _apply_rule(cur, cells_by_grid_pos, cell_to_grid_pos, cycle_len, direction == "R")
    return cur


def _is_win(
    goals: Tuple[Tuple[str, int, int], ...],
    targets_b11: List[Tuple[int, int]],
    targets_f12: List[Tuple[int, int]],
) -> bool:
    """Mirror `khartslnwa`: every target needs a matching goal/goal-o
    at (target.x + 1, target.y + 1)."""
    goal_set = {(gx, gy) for kind, gx, gy in goals if kind == "goal"}
    goalo_set = {(gx, gy) for kind, gx, gy in goals if kind == "goal-o"}
    for tx, ty in targets_b11:
        if (tx + 1, ty + 1) not in goal_set:
            return False
    for tx, ty in targets_f12:
        if (tx + 1, ty + 1) not in goalo_set:
            return False
    return True


def _normalize_state(goals: Tuple[Tuple[str, int, int], ...]) -> Tuple[Tuple[str, int, int], ...]:
    """Sort for canonical hashing. Goals of the same kind are
    interchangeable for the win predicate; sorting by (kind, gx, gy)
    canonicalises the state."""
    return tuple(sorted(goals))


def _solve_lp85(
    initial_goals: List[Tuple[int, int, str]],
    level_idx: int,
    max_states: int = 200_000,
) -> Tuple[List[int], int]:
    """BFS over goal-set state. Returns (button_index_sequence, states_visited).

    Each button index resolves to a click_disp coordinate via
    ``LEVEL_DATA[level_idx]["buttons"]``. Empty list on failure.
    """
    data = LEVEL_DATA[level_idx]
    level_name = data["level_name"]
    buttons = data["buttons"]
    targets_b11 = data["targets_b11"]
    targets_f12 = data["targets_f12"]

    start = _normalize_state(tuple((kind, gx, gy) for gx, gy, kind in initial_goals))

    if _is_win(start, targets_b11, targets_f12):
        return [], 1

    visited: set = {start}
    queue: deque = deque([(start, [])])
    while queue:
        state, path = queue.popleft()
        if len(visited) > max_states:
            break
        for btn_idx, btn in enumerate(buttons):
            keys = btn["keys"]
            new_goals = _apply_button_click(state, level_name, keys)
            new_state = _normalize_state(new_goals)
            if _is_win(new_state, targets_b11, targets_f12):
                return path + [btn_idx], len(visited)
            if new_state in visited:
                continue
            visited.add(new_state)
            queue.append((new_state, path + [btn_idx]))
    return [], len(visited)


def perceive_and_plan(frame: Any) -> List[Tuple[Any, dict]]:
    """Frame -> [(GameAction.ACTION6, {"x": .., "y": ..}), ...]."""
    state = perceive_lp85(frame)
    level_idx = state.get("level_idx")
    goals = state.get("goals") or []
    if level_idx is None or level_idx not in LEVEL_DATA:
        return []
    if not goals:
        # No goals detected -> can't plan (likely transitional frame).
        return []
    plan_indices, _ = _solve_lp85(goals, level_idx)
    if not plan_indices:
        return []
    buttons = LEVEL_DATA[level_idx]["buttons"]
    out: List[Tuple[Any, dict]] = []
    for idx in plan_indices:
        cx, cy = buttons[idx]["click_disp"]
        out.append((GameAction.ACTION6, {"x": int(cx), "y": int(cy)}))
    return out


def plan_next_actions_from_state(state: dict) -> List[Tuple[Any, dict]]:
    """Compatibility alias used by some test helpers."""
    level_idx = state.get("level_idx")
    goals = state.get("goals") or []
    if level_idx is None or level_idx not in LEVEL_DATA:
        return []
    if not goals:
        return []
    plan_indices, _ = _solve_lp85(goals, level_idx)
    if not plan_indices:
        return []
    buttons = LEVEL_DATA[level_idx]["buttons"]
    out: List[Tuple[Any, dict]] = []
    for idx in plan_indices:
        cx, cy = buttons[idx]["click_disp"]
        out.append((GameAction.ACTION6, {"x": int(cx), "y": int(cy)}))
    return out
