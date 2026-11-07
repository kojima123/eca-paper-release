"""g50t frame perception (L1 signature dispatch).

g50t is a multi-round path-recording puzzle:
  - ACTION1/2/3/4 move the player by a stride of 6 pixels in the
    cardinal directions (up/down/left/right).
  - ACTION5 (`pmlawcgvcp`) submits the current round, undoing the
    player's path step-by-step (`tqzseunaxh` cascade) and converting
    the recorded path into a "ghost" that will replay the same actions
    in subsequent rounds (the ghost moves with the player, mirroring
    the recorded delta until its path is exhausted).
  - Each level has N round-counter sprites (`gpkhwmwioo`/`ekfhaifjds`).
    L1 has 2 rounds. The ghost can hold a switch tile (`mxmeqbrmab`,
    medyellngi) so a downstream `dryrnuvljg` (e.g., a kjrcloicja wall)
    stays open while the player walks past it in round 2.
  - Win: player at (goal.x+1, goal.y+1) — `safkknjslo` predicate.

L1 layout (rendered colours):
  - Background `vsrojdvivb` walkable area painted in colour 5.
  - Walls / blocked regions are -1 in the background sprite, rendered
    as background colour 0.
  - Player `qftsebtxuc` colour set {0, 9}, ~7x7 sprite.
  - Goal `gilbljmfbc` colour 9 outline + 5 fill, 9x9 sprite.
  - medyellngi switch (colour 8), 7x7.
  - kjrcloicja toggleable wall (colours 5/8), 7x7.
  - oanvksd0 door overlay (colour 8), 25x31.
  - Two round counters at top-left at (0,0) and (4,0) painted colour 1.

Signature: the L1 frame's colour set is {0, 1, 5, 8, 9}. L2 has the
same colour set but with 3 round counters (~18 pixels of colour 1).
Disambiguator: count of colour-1 pixels == 9 on L1, == 18 on L2.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# L1 colour signature.
_L1_COLORS = frozenset({0, 1, 5, 8, 9})

# L1 has 2 round-counter sprites at (0,0) and (4,0); each is a 5x5
# bordered square with 3x3 colour-1 fill. Total visible colour-1
# pixels in initial frame = 2 * 3 * 3 = 18 if both visible, but the
# active round-counter is a different "active" sprite so only the
# inactive one shows colour 1 (~9 pixels). Empirically: L1=9, L2=18,
# L3+ also 18+ (more rounds).
_L1_C1_COUNT = 9
_L1_PROGRESS_COLORS = frozenset({0, 1, 2, 5, 8, 9})


def _component_bounds(arr: np.ndarray, color: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    mask = arr == color
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[int, tuple[int, int, int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                px, py = stack.pop()
                xs.append(px)
                ys.append(py)
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            components.append((len(xs), (min(xs), min(ys), max(xs) + 1, max(ys) + 1)))
    components.sort(reverse=True)
    return components


def _player_pos(arr: np.ndarray) -> tuple[int, int] | None:
    for size, bounds in _component_bounds(arr, 9):
        x0, y0, x1, y1 = bounds
        if size < 20 or size > 30:
            continue
        if x0 < 10 or y0 < 7:
            continue
        # The L1 goal outline is also colour 9, but it is not the player.
        if x0 >= 42 and y0 >= 48:
            continue
        if (x1 - x0) <= 6 and (y1 - y0) <= 6:
            return (x0 - 1, y0 - 1)
    return None


def _on_l1_route(pos: tuple[int, int] | None) -> bool:
    if pos is None:
        return False
    x, y = pos
    if y == 7 and 13 <= x <= 37 and (x - 13) % 6 == 0:
        return True
    if x == 13 and 7 <= y <= 49 and (y - 7) % 6 == 0:
        return True
    if y == 49 and 13 <= x <= 43 and (x - 13) % 6 == 0:
        return True
    return False


def perceive_g50t(frame: Any) -> dict:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    colors = frozenset(int(c) for c in np.unique(arr).tolist())
    c1 = int((arr == 1).sum())
    player = _player_pos(arr)

    sig = None
    if colors == _L1_COLORS and c1 == _L1_C1_COUNT:
        sig = "g50t_l1"
    elif colors == _L1_COLORS and _L1_C1_COUNT <= c1 <= 12:
        sig = "g50t_l1"
    elif colors == _L1_PROGRESS_COLORS and c1 <= _L1_C1_COUNT:
        sig = "g50t_l1"
    elif colors in (_L1_COLORS, _L1_PROGRESS_COLORS) and c1 <= 18 and _on_l1_route(player):
        sig = "g50t_l1"

    return {
        "level_signature": sig,
        "colors": colors,
        "c1": c1,
        "player_pos": player,
        "round_submitted": bool(2 in colors or c1 < _L1_C1_COUNT),
    }
