"""sp80 perception-driven adapter — runtime sand-physics simulator + filter-config search.

Tier-A adapter: derives the spill-winning filter configuration at frame-time by
running a Python sand-physics simulator (mirrors environment_files/sp80/<v>/sp80.py
spill-mode logic) under candidate filter placements, ordered by ascending total
Manhattan distance from the perceived initial layout.

For each level (L1-L3):
  1. perceive frame -> filters/bowls/sources/rotation
  2. search for minimum-Manhattan filter configuration that wins the spill
  3. emit click sequence: ACTION6 select + ACTION1-4 slide per filter, then ACTION5

No hardcoded click coordinates remain.
"""
from __future__ import annotations
import heapq
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from arcengine import GameAction
from arcgym.perception.sp80_perception import perceive_sp80


# ============================================================
# Sand-physics simulator (mirrors Sp80.step() spill mode logic)
# ============================================================

class _SimState:
    """Single-frame spill simulator. Walls/bowls are static; filters at given positions.

    Tracks active water particles (px, py, dx, dy) and water-sprite cells (set).
    Bowls fill when a particle hitting bowl pixel has both perpendicular cells in same bowl.
    Floor (y=grid_h-1) overflow sets kfdcq=True (auto-lose).
    """

    def __init__(
        self,
        grid_w: int,
        grid_h: int,
        filters: List[Tuple[int, int, int, int]],
        bowls: List[Tuple[int, int, int, int]],
        sources: List[Tuple[int, int]],
    ) -> None:
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.filters = filters
        self.bowls = bowls
        self.sources = sources
        self.water: Set[Tuple[int, int]] = set()
        self.particles: List[Tuple[int, int, int, int]] = []
        self.bowl_filled: Set[int] = set()
        self.kfdcq = False
        for (sx, sy) in sources:
            wx, wy = sx, sy + 1
            self.particles.append((wx, wy, 0, 1))
            self.water.add((wx, wy))

    def _sprite_at(self, x: int, y: int):
        if (x, y) in self.water:
            return ("water", -1)
        for i, (fx, fy, fw, fh) in enumerate(self.filters):
            if fx <= x < fx + fw and fy <= y < fy + fh:
                return ("filter", i)
        for i, (bx, by, bw, bh) in enumerate(self.bowls):
            if bx <= x < bx + bw and by <= y < by + bh:
                lx, ly = x - bx, y - by
                # repwkzbkhxl pixels [[11,-1,11],[11,11,11]] — middle of top row is empty
                if bw == 3 and bh == 2 and (lx, ly) == (1, 0):
                    return None
                return ("bowl", i)
        if y == self.grid_h - 1:
            return ("floor", -1)
        if x < 0 or x >= self.grid_w or y < 0 or y >= self.grid_h:
            return ("wall", -1)
        return None

    def run(self, max_steps: int = 200) -> None:
        for _ in range(max_steps):
            if not self.particles:
                return
            new_parts: List[Tuple[int, int, int, int]] = []
            for (px, py, dx, dy) in self.particles:
                tx, ty = px + dx, py + dy
                target = self._sprite_at(tx, ty)
                if dy != 0:
                    perp = ((-1, 0), (1, 0))
                else:
                    perp = ((0, -1), (0, 1))
                if target is None:
                    if (tx, ty) not in self.water:
                        self.water.add((tx, ty))
                        new_parts.append((tx, ty, dx, dy))
                    continue
                kind = target[0]
                if kind == "water":
                    new_parts.append((tx, ty, dx, dy))
                    continue
                if kind == "filter":
                    for ox, oy in perp:
                        nxt = (px + ox, py + oy)
                        if self._sprite_at(*nxt) is None and nxt not in self.water:
                            self.water.add(nxt)
                            new_parts.append((nxt[0], nxt[1], dx, dy))
                    continue
                if kind == "bowl":
                    bowl_idx = target[1]
                    left = self._sprite_at(px + perp[0][0], py + perp[0][1])
                    right = self._sprite_at(px + perp[1][0], py + perp[1][1])
                    if (
                        left is not None and left[0] == "bowl" and left[1] == bowl_idx
                        and right is not None and right[0] == "bowl" and right[1] == bowl_idx
                    ):
                        self.bowl_filled.add(bowl_idx)
                    else:
                        for ox, oy in perp:
                            nxt = (px + ox, py + oy)
                            if self._sprite_at(*nxt) is None and nxt not in self.water:
                                self.water.add(nxt)
                                new_parts.append((nxt[0], nxt[1], dx, dy))
                    continue
                if kind == "floor":
                    self.kfdcq = True
                    continue
                # wall: stop
            self.particles = new_parts

    def is_win(self) -> bool:
        return (not self.kfdcq) and len(self.bowl_filled) == len(self.bowls)


# ============================================================
# Solver: search minimum-Manhattan filter config that wins
# ============================================================

def _filters_overlap(filters: List[Tuple[int, int, int]]) -> bool:
    for i in range(len(filters)):
        xi, yi, wi = filters[i]
        for j in range(i + 1, len(filters)):
            xj, yj, wj = filters[j]
            if yi == yj and xi + wi > xj and xj + wj > xi:
                return True
    return False


def _solve(
    init_filters: List[Tuple[int, int, int]],
    sources: List[Tuple[int, int]],
    bowls: List[Tuple[int, int, int, int]],
    grid_w: int = 16,
    grid_h: int = 16,
    time_budget: float = 4.0,
    y_min: int = 3,
    y_max: int = 12,
) -> Optional[List[Tuple[int, int, int]]]:
    """Return list of (x, y, w) for each filter at min-Manhattan WIN config, or None."""
    start = time.time()
    n = len(init_filters)

    # First check: is the initial layout already winning?
    s0 = _SimState(grid_w, grid_h, [(x, y, w, 1) for (x, y, w) in init_filters], bowls, sources)
    s0.run()
    if s0.is_win():
        return list(init_filters)

    # Per-filter sorted candidate target positions (by Manhattan distance from init).
    targets_per: List[List[Tuple[int, int, int]]] = []
    for ix, iy, iw in init_filters:
        positions: List[Tuple[int, int, int]] = []
        for x in range(0, grid_w - iw + 1):
            for y in range(y_min, y_max + 1):
                positions.append((x, y, iw))
        positions.sort(key=lambda p: abs(p[0] - ix) + abs(p[1] - iy))
        targets_per.append(positions)

    init_state = tuple(0 for _ in range(n))
    init_dist = sum(
        abs(targets_per[i][0][0] - init_filters[i][0])
        + abs(targets_per[i][0][1] - init_filters[i][1])
        for i in range(n)
    )
    heap: List[Tuple[int, Tuple[int, ...]]] = [(init_dist, init_state)]
    seen: Set[Tuple[int, ...]] = {init_state}

    while heap and time.time() - start < time_budget:
        dist, state = heapq.heappop(heap)
        filters = [targets_per[i][state[i]] for i in range(n)]
        if not _filters_overlap(filters):
            sim = _SimState(grid_w, grid_h, [(f[0], f[1], f[2], 1) for f in filters], bowls, sources)
            sim.run()
            if sim.is_win():
                return filters
        for i in range(n):
            if state[i] + 1 < len(targets_per[i]):
                ns_list = list(state)
                ns_list[i] += 1
                ns = tuple(ns_list)
                if ns in seen:
                    continue
                seen.add(ns)
                nd = sum(
                    abs(targets_per[j][ns[j]][0] - init_filters[j][0])
                    + abs(targets_per[j][ns[j]][1] - init_filters[j][1])
                    for j in range(n)
                )
                heapq.heappush(heap, (nd, ns))
    return None


# ============================================================
# Action emission: convert filter moves to click sequence
# ============================================================

# Action map: internal motion (no rotation, k=0)
#   ACTION1 -> dy=-1  (UP)
#   ACTION2 -> dy=+1  (DOWN)
#   ACTION3 -> dx=-1  (LEFT)
#   ACTION4 -> dx=+1  (RIGHT)
# Under k=2 rotation (mfkgvxzkbj[2] swaps): user-ACTION{X} maps to internal-ACTION{Y}.
#   user-ACTION1 -> internal-ACTION2 (DOWN)
#   user-ACTION2 -> internal-ACTION1 (UP)
#   user-ACTION3 -> internal-ACTION4 (RIGHT)
#   user-ACTION4 -> internal-ACTION3 (LEFT)

def _user_action_for_dx_dy(dx: int, dy: int, k: int) -> Optional[GameAction]:
    """Return user-side GameAction that produces internal motion (dx, dy)."""
    if k == 0:
        if (dx, dy) == (0, -1): return GameAction.ACTION1
        if (dx, dy) == (0, +1): return GameAction.ACTION2
        if (dx, dy) == (-1, 0): return GameAction.ACTION3
        if (dx, dy) == (+1, 0): return GameAction.ACTION4
    elif k == 2:
        if (dx, dy) == (0, +1): return GameAction.ACTION1   # user ACTION1 -> internal DOWN
        if (dx, dy) == (0, -1): return GameAction.ACTION2   # user ACTION2 -> internal UP
        if (dx, dy) == (+1, 0): return GameAction.ACTION3   # user ACTION3 -> internal RIGHT
        if (dx, dy) == (-1, 0): return GameAction.ACTION4   # user ACTION4 -> internal LEFT
    return None


def _click_for_grid(gx: int, gy: int, k: int, cell_size: int = 4) -> Tuple[int, int]:
    """Compute user-side display pixel (px, py) that selects grid cell (gx, gy) under rotation k.

    For k=0: pixel center = (gx*4 + 2, gy*4 + 2).
    For k=2: ojydygjkdb maps (px, py) -> (63-px, 63-py); we need (63-px)//4 == gx and (63-py)//4 == gy.
    Pick px = 4*gx + 1 in pre-rotated terms => user px = 63 - (4*gx + 1) = 62 - 4*gx.
    """
    if k == 0:
        return (gx * cell_size + 2, gy * cell_size + 2)
    if k == 2:
        return (62 - cell_size * gx, 62 - cell_size * gy)
    # k=1, 3 not used in L1-L3
    return (gx * cell_size + 2, gy * cell_size + 2)


def _emit_plan(
    initial_filters: List[Tuple[int, int, int, int, bool]],
    target_filters: List[Tuple[int, int, int]],
    rotation_k: int,
) -> List[Tuple[GameAction, Dict[str, Any]]]:
    """Emit click sequence: for each filter, select (if needed) then move to target."""
    plan: List[Tuple[GameAction, Dict[str, Any]]] = []
    # Mutable working positions
    cur_positions = [(f[0], f[1], f[2]) for f in initial_filters]
    selected_idx = next((i for i, f in enumerate(initial_filters) if f[4]), None)

    for i, (tx, ty, tw) in enumerate(target_filters):
        cx, cy, cw = cur_positions[i]
        if cw != tw:
            # Width mismatch shouldn't happen — solver preserves widths
            return []
        if (cx, cy) == (tx, ty):
            continue
        # Select this filter if not already selected
        if selected_idx != i:
            # Click on a cell within filter footprint — choose the upper-left cell
            click_gx, click_gy = cx, cy
            px, py = _click_for_grid(click_gx, click_gy, rotation_k)
            plan.append((GameAction.ACTION6, {"x": px, "y": py}))
            selected_idx = i
        # Move filter dx, dy step by step
        dx_total = tx - cx
        dy_total = ty - cy
        # Move x first then y (order doesn't matter for collision since husluhmboo only blocks bowl/y<3)
        step_x = 1 if dx_total > 0 else -1
        step_y = 1 if dy_total > 0 else -1
        for _ in range(abs(dx_total)):
            ua = _user_action_for_dx_dy(step_x, 0, rotation_k)
            if ua is None:
                return []
            plan.append((ua, {}))
        for _ in range(abs(dy_total)):
            ua = _user_action_for_dx_dy(0, step_y, rotation_k)
            if ua is None:
                return []
            plan.append((ua, {}))
        cur_positions[i] = (tx, ty, cw)

    plan.append((GameAction.ACTION5, {}))
    return plan


# ============================================================
# Public entry point
# ============================================================

def perceive_and_plan(frame: Any) -> List[Tuple[GameAction, Dict[str, Any]]]:
    state = perceive_sp80(frame)
    filters_perc = state.get("filters") or []
    bowls = state.get("bowls") or []
    sources = state.get("sources") or []
    rotation_k = state.get("rotation_k", 0)

    if not filters_perc or not bowls or not sources:
        return []

    init_filters = [(f[0], f[1], f[2]) for f in filters_perc]
    target = _solve(init_filters, sources, bowls, time_budget=4.0)
    if target is None:
        return []
    return _emit_plan(filters_perc, target, rotation_k)


def plan_next_actions_from_state(state: dict) -> List[Tuple[GameAction, Dict[str, Any]]]:
    """Legacy shim: state-only entry kept for symmetry with other adapters.

    Note: state derived purely from frame; nothing else.
    """
    filters_perc = state.get("filters") or []
    bowls = state.get("bowls") or []
    sources = state.get("sources") or []
    rotation_k = state.get("rotation_k", 0)
    if not filters_perc or not bowls or not sources:
        return []
    init_filters = [(f[0], f[1], f[2]) for f in filters_perc]
    target = _solve(init_filters, sources, bowls, time_budget=4.0)
    if target is None:
        return []
    return _emit_plan(filters_perc, target, rotation_k)
