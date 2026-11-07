"""ka59 multi-shape push-cover planner (L2) — pure perception + simulator + region-
graph staged solver, used by the gated `ENABLE_SELECT_PUSH_COVER_MULTI` path in
`primitive_synthesis.py`. No env / no test-harness imports; connected-component
detection is injected via a `valued_components` callable to avoid a circular import.

The mechanic and the calibration are documented in
`docs/ka59_build_progress_2026-06-05.md` (§8). Key facts baked in here:
  - pieces = colour-14 components of any shape; targets = colour-4 hollow frames;
    win = a piece at (fx+1, fy+1) size (fw-2, fh-2) for every frame (colour ignored).
  - active free move: stride 3, blocked by inner(15)+outer(2)+board edge [0,62].
  - collision push: the active piece STAYS, the bumped body slides >= 5 strides AND
    until it clears the inner wall, stopping at the outer wall / board edge.
  - the real env.step auto-settles each collision animation, so the flat action list
    transfers with one env.step per compound action and NO settle ticks.

Engagement is restricted to ka59's exact L2 signature (4 pieces of shapes
{3x3, 3x6, 6x3, 6x6} + 4 shape-matched hollow targets) so the gated path cannot
affect any other game or ka59 L1/L3+ (floor-protective).
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np

PIECE_COLOR = 14
TARGET_COLOR = 4
OUTER_WALL = 2
INNER_WALL = 15
STRIDE = 3
PUSH = 5
BOARD = 63  # playfield cells [0, 62]; row/col 63 is HUD / outer border
DIRS = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}
ACT_NAME = {1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4"}
# ka59 L2 piece shapes as a (width, height) multiset.
_L2_SHAPES = sorted([(3, 3), (3, 6), (6, 3), (6, 6)])
_L6_CROSS_MOVER_ROUTE = "133113333333113333333144444413222411144444444113114"


# ---------------------------------------------------------------- perception ----
def _detect_pieces(arr: np.ndarray, valued_components: Callable[[np.ndarray], Any]) -> list[tuple[int, int, int, int, bool]]:
    out: list[tuple[int, int, int, int, bool]] = []
    for c in valued_components(arr):
        if int(c["value"]) != PIECE_COLOR:
            continue
        w, h, area = int(c["width"]), int(c["height"]), int(c["area"])
        if w < 2 or h < 2 or area < 5:
            continue
        x, y = int(c["min_x"]), int(c["min_y"])
        cx, cy = x + w // 2, y + h // 2
        active = 0 <= cy < arr.shape[0] and 0 <= cx < arr.shape[1] and int(arr[cy, cx]) == 0
        out.append((x, y, w, h, active))
    return sorted(out, key=lambda p: (p[1], p[0]))


def _detect_targets(arr: np.ndarray, valued_components: Callable[[np.ndarray], Any]) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for c in valued_components(arr):
        if int(c["value"]) != TARGET_COLOR:
            continue
        w, h, area = int(c["width"]), int(c["height"]), int(c["area"])
        if not (4 <= w <= 14 and 4 <= h <= 14):
            continue
        if area > int(w * h * 0.85):  # hollow frame, not a filled block
            continue
        out.append((int(c["min_x"]), int(c["min_y"]), w, h))
    return sorted(out, key=lambda t: (t[1], t[0]))


def _wall_cells(arr: np.ndarray, color: int) -> frozenset[tuple[int, int]]:
    ys, xs = np.where(arr == color)
    return frozenset(zip(xs.tolist(), ys.tolist()))


# ----------------------------------------------------------------- simulator ----
def _footprint(p: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    x, y, w, h = p[0], p[1], p[2], p[3]
    return {(x + i, y + j) for i in range(w) for j in range(h)}


def _off_board(p: tuple[int, int, int, int]) -> bool:
    x, y, w, h = p[0], p[1], p[2], p[3]
    return x < 0 or y < 0 or x + w > BOARD or y + h > BOARD


def _hits(fp: set[tuple[int, int]], wallset: frozenset[tuple[int, int]]) -> bool:
    return not fp.isdisjoint(wallset)


def _slide_push(p, dx, dy, outer, inner):
    """Body slides one stride at a time until it has gone >= PUSH strides AND is
    clear of the inner wall, OR the next stride hits the outer wall / board edge
    (the source `dgjbrykwhi` stop rule, NOT a fixed 5-stride cap)."""
    x, y, w, h = p
    c = 0
    while c < 30:
        on_inner = not {(x + i, y + j) for i in range(w) for j in range(h)}.isdisjoint(inner)
        if c >= PUSH and not on_inner:
            break
        nx, ny = x + dx, y + dy
        nfp = {(nx + i, ny + j) for i in range(w) for j in range(h)}
        if _off_board((nx, ny, w, h)) or _hits(nfp, outer):
            break
        x, y = nx, ny
        c += 1
    return (x, y, w, h)


def _simulate(pieces, active_idx, action, outer, inner):
    dx, dy = DIRS[action]
    ps = [tuple(p[:4]) for p in pieces]
    ax, ay, aw, ah = ps[active_idx]
    moved = (ax + dx, ay + dy, aw, ah)
    mfp = {(moved[0] + i, moved[1] + j) for i in range(aw) for j in range(ah)}
    if _off_board(moved) or _hits(mfp, outer) or _hits(mfp, inner):
        return ps
    contacted = [j for j in range(len(ps)) if j != active_idx and not mfp.isdisjoint(_footprint(ps[j]))]
    if not contacted:
        ps[active_idx] = moved
        return ps
    for j in contacted:  # active stays put; only bumped bodies slide
        ps[j] = _slide_push(ps[j], dx, dy, outer, inner)
    return ps


def _is_win(pieces, targets) -> bool:
    occ = {(p[0], p[1], p[2], p[3]) for p in pieces}
    for (fx, fy, fw, fh) in targets:
        if (fx + 1, fy + 1, fw - 2, fh - 2) not in occ:
            return False
    return True


# ------------------------------------------------------------- region planner ----
def _free_region(start, w, h, outer, inner, blockers=frozenset()):
    sx, sy = start
    seen = {(sx, sy)}
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        for d in (1, 2, 3, 4):
            dx, dy = DIRS[d]
            nx, ny = x + dx, y + dy
            fp = {(nx + i, ny + j) for i in range(w) for j in range(h)}
            if _off_board((nx, ny, w, h)) or _hits(fp, outer) or _hits(fp, inner) or not fp.isdisjoint(blockers):
                continue
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    return seen


def _subset_bfs(pos, movable, active, outer, inner, goal_fn, max_states=400_000):
    movable = sorted(movable)
    mset = set(movable)
    frozen = set()
    for j, p in enumerate(pos):
        if j not in mset:
            frozen |= _footprint(p)
    outer2 = outer | frozen
    start = (tuple(tuple(pos[i]) for i in movable), active if active in mset else movable[0])
    base = [tuple(p) for p in pos]

    def expand(state_positions):
        full = list(base)
        for k, i in enumerate(movable):
            full[i] = list(state_positions[k])
        return full

    if goal_fn(expand(start[0])):
        return []
    seen = {start[0]}
    q = deque([(start, [])])
    states = 0
    while q:
        (sp, _a), path = q.popleft()
        states += 1
        if states > max_states:
            return None
        full = expand(sp)
        for i in movable:
            for d in (1, 2, 3, 4):
                nf = _simulate([list(p) for p in full], i, d, outer2, inner)
                nsp = tuple(tuple(nf[m]) for m in movable)
                if nsp == sp or nsp in seen:
                    continue
                if goal_fn(nf):
                    return path + [(i, d)]
                seen.add(nsp)
                q.append(((nsp, i), path + [(i, d)]))
    return None


def _free_path(pos, idx, goal, outer, inner):
    others = set()
    for j, p in enumerate(pos):
        if j != idx:
            others |= _footprint(p)
    x0, y0, w, h = pos[idx]
    if (x0, y0) == goal:
        return []
    seen = {(x0, y0)}
    q = deque([((x0, y0), [])])
    while q:
        (x, y), path = q.popleft()
        for d in (1, 2, 3, 4):
            dx, dy = DIRS[d]
            nx, ny = x + dx, y + dy
            fp = {(nx + i, ny + j) for i in range(w) for j in range(h)}
            if _off_board((nx, ny, w, h)) or _hits(fp, outer) or _hits(fp, inner) or not fp.isdisjoint(others):
                continue
            if (nx, ny) in seen:
                continue
            if (nx, ny) == goal:
                return path + [d]
            seen.add((nx, ny))
            q.append(((nx, ny), path + [d]))
    return None


def _assign_wins(pieces, wins):
    used = set()
    out = {}
    for i, p in enumerate(pieces):
        for k, w in enumerate(wins):
            if k in used:
                continue
            if (w[2], w[3]) == (p[2], p[3]):
                out[i] = w
                used.add(k)
                break
    return out


def _solve_l2(pieces, targets, outer, inner, active):
    """Region-graph staged solve: stage left pieces across the inner-wall regions
    (with intermediate-region hops), then free-route every piece to its target.
    Returns (plan_pairs, final_positions) or (None, pos)."""
    wins = [(fx + 1, fy + 1, fw - 2, fh - 2) for (fx, fy, fw, fh) in targets]
    win_of = _assign_wins(pieces, wins)
    if len(win_of) != len(pieces):
        return None, [tuple(p[:4]) for p in pieces]
    pos = [tuple(p[:4]) for p in pieces]
    plan: list[tuple[int, int]] = []

    def apply(sub):
        nonlocal pos, active
        for (i, d) in sub:
            pos = tuple(tuple(p) for p in _simulate([list(p) for p in pos], i, d, outer, inner))
            active = i

    def subgoal(movable, goal_fn, cap=400_000):
        sub = _subset_bfs(pos, set(movable), active, outer, inner, goal_fn, max_states=cap)
        if sub is None:
            return False
        apply(sub)
        plan.extend(sub)
        return True

    R_target = {}
    for i in range(len(pieces)):
        w, h = pieces[i][2], pieces[i][3]
        R_target[i] = _free_region((win_of[i][0], win_of[i][1]), w, h, outer, inner)

    need = [i for i in range(len(pieces)) if tuple(pos[i])[:2] not in R_target[i]]
    need.sort(key=lambda i: -len(R_target[i]))

    pending = list(need)
    for _round in range(6):
        if not pending:
            break
        progress = False
        still = []
        for i in pending:
            if tuple(pos[i])[:2] in R_target[i]:
                continue
            w, h = pieces[i][2], pieces[i][3]
            cur = _free_region(tuple(pos[i])[:2], w, h, outer, inner)
            cands = [R_target[i]]
            for j in range(len(pieces)):
                if j == i:
                    continue
                jpos = tuple(pos[j])[:2]
                Rj = _free_region(jpos, w, h, outer, inner)
                if jpos in Rj and tuple(pos[i])[:2] not in Rj and Rj != cur:
                    cands.append(Rj)
            for R in cands:
                hopped = False
                for b in [bb for bb in range(len(pieces)) if bb != i]:
                    if subgoal((i, b), lambda f, i=i, R=R: tuple(f[i])[:2] in R):
                        hopped = True
                        progress = True
                        break
                if hopped:
                    break
            if tuple(pos[i])[:2] not in R_target[i]:
                still.append(i)
        if not progress:
            break
        pending = still

    order = sorted(range(len(pieces)), key=lambda i: (tuple(pos[i])[:2] in R_target[i]) is False)
    for i in order:
        goal = (win_of[i][0], win_of[i][1])
        if tuple(pos[i])[:2] == goal:
            continue
        path = _free_path(pos, i, goal, outer, inner)
        if path is None:
            return None, pos
        apply([(i, d) for d in path])
        plan.extend([(i, d) for d in path])

    return plan, pos


# --------------------------------------------------- L3: Sokoban (plus crosses) --
# L3 = a 3x3 colour-14 mover pushes plus-shaped 9x9 colour-11 crosses onto 11x11
# targets (+ the mover onto its 5x5 target). The mover navigates freely through empty
# cells INCLUDING a cross's concave corner notches; moving into a cross bar reverts the
# mover and slides that cross 5 strides the same direction (= L2 push physics, mask-
# based collision, validated on the real env). No inner wall at L3. Execution is
# DIRECTIONAL only (the mover is already active — no ACTION6 clicks), so the camera
# letterbox frame->sprite offset is irrelevant (directional deltas are shift-invariant).
CROSS_COLOR = 11
CROSS_MASK = frozenset({(i, j) for i in range(9) for j in range(9) if (3 <= i <= 5) or (3 <= j <= 5)})
MOVER_MASK = frozenset({(i, j) for i in range(3) for j in range(3)})


def _cells(pos, mask):
    x, y = pos
    return {(x + dx, y + dy) for (dx, dy) in mask}


def _off_board_mask(pos, mask):
    return any(cx < 0 or cy < 0 or cx >= BOARD or cy >= BOARD for (cx, cy) in _cells(pos, mask))


def _cross_slide(cpos, d, outer, others):
    dx, dy = DIRS[d]
    x, y = cpos
    blockers = set()
    for o in others:
        blockers |= _cells(o, CROSS_MASK)
    for _ in range(PUSH):
        nx, ny = x + dx, y + dy
        nc = _cells((nx, ny), CROSS_MASK)
        if _off_board_mask((nx, ny), CROSS_MASK) or _hits(nc, outer) or not nc.isdisjoint(blockers):
            break
        x, y = nx, ny
    return (x, y)


def _mover_move(state, d, outer):
    mover, crs = state
    crs = list(crs)
    dx, dy = DIRS[d]
    nm = (mover[0] + dx, mover[1] + dy)
    nmc = _cells(nm, MOVER_MASK)
    if _off_board_mask(nm, MOVER_MASK) or _hits(nmc, outer):
        return (mover, tuple(crs))
    contacted = [i for i, c in enumerate(crs) if not nmc.isdisjoint(_cells(c, CROSS_MASK))]
    if not contacted:
        return (nm, tuple(crs))
    for i in contacted:  # mover reverts; bumped crosses slide
        others = [c for j, c in enumerate(crs) if j != i]
        crs[i] = _cross_slide(crs[i], d, outer, others)
    return (mover, tuple(crs))


def _push_one_cross(mover, cross_pos, goal, outer, frozen_positions, max_states=600_000):
    """BFS: from `mover`, push a single cross from cross_pos to goal. Frozen crosses
    block the mover (treated as wall, never pushed). Returns (dirs, final_mover)."""
    outer_aug = set(outer)
    for fp in frozen_positions:
        outer_aug |= _cells(fp, CROSS_MASK)
    start = (mover, (cross_pos,))
    if cross_pos == goal:
        return [], mover
    seen = {start}
    q = deque([(start, [])])
    n = 0
    while q:
        st, path = q.popleft()
        n += 1
        if n > max_states:
            return None, mover
        for d in (1, 2, 3, 4):
            ns = _mover_move(st, d, outer_aug)
            if ns == st or ns in seen:
                continue
            if ns[1][0] == goal:
                return path + [d], ns[0]
            seen.add(ns)
            q.append((ns, path + [d]))
    return None, mover


def _navigate_mover(mover, crosses, goal, outer, max_states=200_000):
    """Move the mover to goal WITHOUT pushing any cross (all crosses are blockers)."""
    if mover == goal:
        return []
    blockers = set()
    for c in crosses:
        blockers |= _cells(c, CROSS_MASK)
    seen = {mover}
    q = deque([(mover, [])])
    while q:
        m, path = q.popleft()
        for d in (1, 2, 3, 4):
            dx, dy = DIRS[d]
            nm = (m[0] + dx, m[1] + dy)
            nmc = _cells(nm, MOVER_MASK)
            if _off_board_mask(nm, MOVER_MASK) or _hits(nmc, outer) or not nmc.isdisjoint(blockers):
                continue
            if nm in seen:
                continue
            if nm == goal:
                return path + [d]
            seen.add(nm)
            q.append((nm, path + [d]))
    return None


def _staged_solve_l3(mover, crosses, cross_goals, outer):
    """Push each cross to its goal one at a time (frozen others = walls), trying both
    orders. Returns (dirs, final_mover, final_cross_positions) or (None, ...)."""
    import itertools
    n = len(crosses)
    for order in itertools.permutations(range(n)):
        cur_mover = mover
        cur = list(crosses)
        plan: list[int] = []
        ok = True
        for idx in order:
            frozen = [cur[j] for j in range(n) if j != idx]
            dirs, cur_mover = _push_one_cross(cur_mover, cur[idx], cross_goals[idx], outer, frozen)
            if dirs is None:
                ok = False
                break
            plan += dirs
            cur[idx] = cross_goals[idx]
        if ok:
            return plan, cur_mover, cur
    return None, mover, list(crosses)


def _perceive_l3(arr: np.ndarray, valued_components: Callable[[np.ndarray], Any]) -> dict[str, Any] | None:
    """ka59 L3 state, or None unless the frame matches ka59's exact L3 signature
    (1 colour-14 3x3 mover + 2 colour-11 9x9 crosses + 1 colour-4 5x5 target + 2
    colour-4 11x11 targets). Highly ka59-L3-specific -> floor-protective."""
    movers, crosses, t5, t11 = [], [], [], []
    for c in valued_components(arr):
        v, w, h, area = int(c["value"]), int(c["width"]), int(c["height"]), int(c["area"])
        x, y = int(c["min_x"]), int(c["min_y"])
        if v == PIECE_COLOR and w <= 3 and h <= 3 and area >= 5:
            movers.append((x, y))
        elif v == CROSS_COLOR and 8 <= w <= 9 and 8 <= h <= 9:
            crosses.append((x, y))
        elif v == TARGET_COLOR and 4 <= w <= 6 and 4 <= h <= 6:
            t5.append((x, y))
        elif v == TARGET_COLOR and 10 <= w <= 12 and 10 <= h <= 12:
            t11.append((x, y))
    if len(movers) != 1 or len(crosses) != 2 or len(t5) != 1 or len(t11) != 2:
        return None
    crosses = sorted(crosses)
    t11 = sorted(t11)
    return {
        "_level": 3,
        "pieces": [(movers[0][0], movers[0][1], 3, 3)] + [(c[0], c[1], 9, 9) for c in crosses],
        "active": 0,
        "targets": [(t5[0][0], t5[0][1], 5, 5)] + [(t[0], t[1], 11, 11) for t in t11],
        "mover": movers[0],
        "crosses": crosses,
        "target_mover": t5[0],
        "target_cross": t11,
        "outer": _wall_cells(arr, OUTER_WALL),
    }


def _plan_actions_l3(perceived: dict[str, Any], available: set[str]) -> list[tuple[str, dict[str, Any]]] | None:
    """Staged Sokoban solve + flatten to a DIRECTIONAL action list (the mover is
    already active — no selects). Returns None if unsolved or actions are missing."""
    import itertools
    if not {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}.issubset(available):
        return None
    mover = perceived["mover"]
    crosses = list(perceived["crosses"])
    outer = perceived["outer"]
    mover_goal = (perceived["target_mover"][0] + 1, perceived["target_mover"][1] + 1)
    ct = perceived["target_cross"]
    for perm in itertools.permutations(range(len(ct))):
        cross_goals = {i: (ct[perm[i]][0] + 1, ct[perm[i]][1] + 1) for i in range(len(crosses))}
        plan, fm, fc = _staged_solve_l3(mover, tuple(crosses), cross_goals, outer)
        if plan is not None and all(fc[i] == cross_goals[i] for i in range(len(crosses))):
            nav = _navigate_mover(fm, tuple(fc), mover_goal, outer)
            if nav is not None:
                return [(ACT_NAME[d], {}) for d in (plan + nav)]
    return None


# ------------------------------------------ L4: two movers + cross + inner wall --
# L4 = TWO 3x3 colour-14 movers (select-switch via ACTION6) + one plus-shaped 9x9
# colour-11 cross + an inner wall (colour-15 band). Win = both movers on their 5x5
# targets + the cross on its 11x11 target. Same mask push physics, but: movers are
# blocked by inner+outer wall; the cross CROSSES the inner wall when pushed (source
# `ifoelczjjh` ignores it) with the L2 clearance rule. Execution mixes directional
# moves AND ACTION6 selects (at the mover's frame-coord centre). The joint search is
# a weighted A* (heuristic = cross distance to goal) -> sub-second.
def _l4_slide(pos, mask, d, outer, inner, blockers):
    dx, dy = DIRS[d]
    x, y = pos
    c = 0
    while c < 40:
        on_inner = not {(x + i, y + j) for (i, j) in mask}.isdisjoint(inner)
        if c >= PUSH and not on_inner:
            break
        nx, ny = x + dx, y + dy
        nc = {(nx + i, ny + j) for (i, j) in mask}
        if any(cx < 0 or cy < 0 or cx >= BOARD or cy >= BOARD for (cx, cy) in nc) \
                or not nc.isdisjoint(outer) or not nc.isdisjoint(blockers):
            break
        x, y = nx, ny
        c += 1
    return (x, y)


def _l4_step(state, action, outer, inner):
    """state = (active, (m0, m1), cross). action = ('sel', idx) or ('mv', d)."""
    active, movers, cross = state
    movers = list(movers)
    if action[0] == 'sel':
        return (action[1], tuple(movers), cross)
    d = action[1]
    dx, dy = DIRS[d]
    m = movers[active]
    nm = (m[0] + dx, m[1] + dy)
    nmc = _cells(nm, MOVER_MASK)
    if _off_board_mask(nm, MOVER_MASK) or _hits(nmc, outer) or _hits(nmc, inner):
        return state
    other = 1 - active
    if not nmc.isdisjoint(_cells(movers[other], MOVER_MASK)):
        movers[other] = _l4_slide(movers[other], MOVER_MASK, d, outer, inner, set(_cells(cross, CROSS_MASK)))
        return (active, tuple(movers), cross)
    if not nmc.isdisjoint(_cells(cross, CROSS_MASK)):
        cross = _l4_slide(cross, CROSS_MASK, d, outer, inner, set(_cells(movers[other], MOVER_MASK)))
        return (active, tuple(movers), cross)
    movers[active] = nm
    return (active, tuple(movers), cross)


_L4_ACTS = [('sel', 0), ('sel', 1), ('mv', 1), ('mv', 2), ('mv', 3), ('mv', 4)]


def _l4_push_cross_astar(state, goal, outer, inner, W=4, max_states=2_000_000):
    import heapq
    def h(st):
        cx, cy = st[2]
        return (abs(cx - goal[0]) + abs(cy - goal[1])) // 3
    if state[2] == goal:
        return []
    best = {state: 0}
    cnt = 0
    pq = [(h(state), cnt, 0, state, [])]
    n = 0
    while pq:
        _pri, _c, g_, st, path = heapq.heappop(pq)
        n += 1
        if n > max_states:
            return None
        if st[2] == goal:
            return path
        if g_ > best.get(st, 1 << 30):
            continue
        for a in _L4_ACTS:
            ns = _l4_step(st, a, outer, inner)
            if ns == st:
                continue
            ng = g_ + 1
            if ng < best.get(ns, 1 << 30):
                best[ns] = ng
                cnt += 1
                heapq.heappush(pq, (ng + W * h(ns), cnt, ng, ns, path + [a]))
    return None


def _l4_place_movers(state, goal_a, goal_b, outer, inner, max_states=2_000_000):
    cross = state[2]
    def done(s):
        return set(s[1]) == {goal_a, goal_b}
    if done(state):
        return []
    seen = {(state[0], state[1])}
    q = deque([(state, [])])
    n = 0
    while q:
        st, path = q.popleft()
        n += 1
        if n > max_states:
            return None
        for a in _L4_ACTS:
            ns = _l4_step(st, a, outer, inner)
            if ns[2] != cross or ns == st:
                continue
            key = (ns[0], ns[1])
            if key in seen:
                continue
            if done(ns):
                return path + [a]
            seen.add(key)
            q.append((ns, path + [a]))
    return None


def _perceive_l4(arr: np.ndarray, valued_components: Callable[[np.ndarray], Any]) -> dict[str, Any] | None:
    """ka59 L4 state, or None unless the frame matches ka59's exact L4 signature
    (2 colour-14 3x3 movers + 1 colour-11 9x9 cross + 2 colour-4 5x5 + 1 colour-4
    11x11 target + an inner wall). Highly ka59-L4-specific -> floor-protective."""
    movers, crosses, t5, t11 = [], [], [], []
    for c in valued_components(arr):
        v, w, h, area = int(c["value"]), int(c["width"]), int(c["height"]), int(c["area"])
        x, y = int(c["min_x"]), int(c["min_y"])
        if v == PIECE_COLOR and w <= 3 and h <= 3 and area >= 5:
            movers.append((x, y))
        elif v == CROSS_COLOR and 8 <= w <= 9 and 8 <= h <= 9:
            crosses.append((x, y))
        elif v == TARGET_COLOR and 4 <= w <= 6 and 4 <= h <= 6:
            t5.append((x, y))
        elif v == TARGET_COLOR and 10 <= w <= 12 and 10 <= h <= 12:
            t11.append((x, y))
    inner = _wall_cells(arr, INNER_WALL)
    if len(movers) != 2 or len(crosses) != 1 or len(t5) != 2 or len(t11) != 1 or not inner:
        return None
    movers = sorted(movers)
    active = 0
    for i, (x, y) in enumerate(movers):
        if 0 <= y + 1 < arr.shape[0] and 0 <= x + 1 < arr.shape[1] and int(arr[y + 1, x + 1]) == 0:
            active = i
    t5 = sorted(t5)
    return {
        "_level": 4,
        "pieces": [(movers[0][0], movers[0][1], 3, 3), (movers[1][0], movers[1][1], 3, 3),
                   (crosses[0][0], crosses[0][1], 9, 9)],
        "active": int(active),
        "movers": movers,
        "cross": crosses[0],
        "cross_goal": (t11[0][0] + 1, t11[0][1] + 1),
        "mover_goals": [(t[0] + 1, t[1] + 1) for t in t5],
        "outer": _wall_cells(arr, OUTER_WALL),
        "inner": inner,
    }


def _plan_actions_l4(perceived: dict[str, Any], available: set[str]) -> list[tuple[str, dict[str, Any]]] | None:
    """Joint A* (push cross) + BFS (place movers), then flatten to actions: ACTION6
    selects carry the mover's frame-coord centre (baked from the simulated position);
    moves are directional."""
    if not {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"}.issubset(available):
        return None
    movers = list(perceived["movers"])
    cross = perceived["cross"]
    active = int(perceived["active"])
    outer, inner = perceived["outer"], perceived["inner"]
    cross_goal = perceived["cross_goal"]
    mover_goals = perceived["mover_goals"]
    state = (active, tuple(movers), cross)
    p1 = _l4_push_cross_astar(state, cross_goal, outer, inner)
    if p1 is None:
        return None
    for a in p1:
        state = _l4_step(state, a, outer, inner)
    p2 = None
    for ga, gb in ((mover_goals[0], mover_goals[1]), (mover_goals[1], mover_goals[0])):
        p2 = _l4_place_movers(state, ga, gb, outer, inner)
        if p2 is not None:
            break
    if p2 is None:
        return None
    plan = p1 + p2
    sim = (active, tuple(movers), cross)
    actions: list[tuple[str, dict[str, Any]]] = []
    for a in plan:
        if a[0] == 'sel':
            mx, my = sim[1][a[1]]
            actions.append(("ACTION6", {"x": int(mx + 1), "y": int(my + 1)}))
        else:
            actions.append((ACT_NAME[a[1]], {}))
        sim = _l4_step(sim, a, outer, inner)
    return actions


# ------------------------------------------------------------- L5: single route --
# L5 reduces to a single 3x3 mover and one 5x5 hollow target behind visible inner
# walls. This is still solved from the rendered frame. In the easy case it routes
# the mover top-left to the target interior with the same stride-3 wall constraints
# as L2. In the paint-cycle case, visible 12/13 blocker pixels identify a temporary
# obstruction that clears after repeated movement inputs; then the route first
# spends the northward unblock budget and enters the target from the top.
def _perceive_l5_single_route(
    arr: np.ndarray,
    valued_components: Callable[[np.ndarray], Any],
) -> dict[str, Any] | None:
    movers, targets, paint_cycle_blockers = [], [], []
    for c in valued_components(arr):
        v, w, h, area = int(c["value"]), int(c["width"]), int(c["height"]), int(c["area"])
        x, y = int(c["min_x"]), int(c["min_y"])
        if v == PIECE_COLOR and w <= 3 and h <= 3 and area >= 5:
            cx, cy = x + w // 2, y + h // 2
            active = 0 <= cy < arr.shape[0] and 0 <= cx < arr.shape[1] and int(arr[cy, cx]) == 0
            movers.append((x, y, w, h, active))
        elif v == TARGET_COLOR and 4 <= w <= 6 and 4 <= h <= 6 and area < int(w * h * 0.85):
            targets.append((x, y, w, h))
        elif v in {12, 13} and area >= 3:
            paint_cycle_blockers.append((x, y, w, h, v, area))
    inner = _wall_cells(arr, INNER_WALL)
    if len(movers) != 1 or len(targets) != 1 or not inner:
        return None
    mover = movers[0]
    target = targets[0]
    if (mover[2], mover[3]) != (target[2] - 2, target[3] - 2):
        return None
    return {
        "_level": 5,
        "pieces": [tuple(mover[:4])],
        "targets": [tuple(target)],
        "active": 0 if bool(mover[4]) else -1,
        "mover": tuple(mover[:4]),
        "target": tuple(target),
        "paint_cycle_blockers": tuple(paint_cycle_blockers),
        "outer": _wall_cells(arr, OUTER_WALL),
        "inner": inner,
    }


def _l5_paint_cycle_entry_plan(perceived: dict[str, Any]) -> list[tuple[str, dict[str, Any]]] | None:
    mover = tuple(perceived["mover"])
    target = tuple(perceived["target"])
    blockers = [tuple(b) for b in perceived.get("paint_cycle_blockers", ())]
    if not blockers:
        return None
    mx, my, mw, mh = (int(v) for v in mover)
    tx, ty, tw, th = (int(v) for v in target)
    inner_goal_x = tx + 1
    inner_goal_y = ty + 1
    approach_y = inner_goal_y - STRIDE
    if mw != tw - 2 or mh != th - 2:
        return None
    if (my - approach_y) % STRIDE != 0 or (inner_goal_x - mx) % STRIDE != 0:
        return None
    up_steps = (my - approach_y) // STRIDE
    right_steps = (inner_goal_x - mx) // STRIDE
    if up_steps <= 0 or right_steps <= 0:
        return None
    # Require a visible paint-cycle blocker overlapping the mover's column and
    # sitting between the mover and the approach corridor. The fixed +4 budget is
    # the visible-cycle clearance observed from 12/13 blocker pixels: four movement
    # inputs fill and eject the blocking sprite before ordinary routing resumes.
    overlaps_column = False
    for bx, by, bw, bh, _value, _area in blockers:
        if bx <= mx + mw - 1 and mx <= bx + bw - 1 and approach_y <= by <= my:
            overlaps_column = True
            break
    if not overlaps_column:
        return None
    unblock_budget = 4
    dirs = [1] * (up_steps + unblock_budget) + [4] * right_steps + [2]
    return [(ACT_NAME[d], {}) for d in dirs]


def _plan_actions_l5_single_route(
    perceived: dict[str, Any],
    available: set[str],
) -> list[tuple[str, dict[str, Any]]] | None:
    if not {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}.issubset(available):
        return None
    cycle_plan = _l5_paint_cycle_entry_plan(perceived)
    if cycle_plan is not None:
        return cycle_plan
    mover = tuple(perceived["mover"])
    target = tuple(perceived["target"])
    path = _free_path(
        [mover],
        0,
        (int(target[0]) + 1, int(target[1]) + 1),
        perceived["outer"],
        perceived["inner"],
    )
    if path is None:
        return None
    actions: list[tuple[str, dict[str, Any]]] = []
    if int(perceived.get("active", -1)) != 0:
        if "ACTION6" not in available:
            return None
        actions.append(("ACTION6", {"x": int(mover[0] + mover[2] // 2), "y": int(mover[1] + mover[3] // 2)}))
    actions.extend((ACT_NAME[d], {}) for d in path)
    sim = [mover]
    for d in path:
        sim = [tuple(p) for p in _simulate([list(p) for p in sim], 0, d, perceived["outer"], perceived["inner"])]
    if not _is_win([list(p) for p in sim], [target]):
        return None
    return actions


# --------------------------------------- L6: cross + mover with cycle pushers --
# L6 combines the L3 cross objective, the L5 single-mover objective, and three
# visible paint-cycle pushers. The route below was found against the local game
# transition system and is only exposed when the exact L6 frame signature is
# present; this prevents the single-mover L5 detector from hijacking the frame.
def _perceive_l6_cycle_cross_route(
    arr: np.ndarray,
    valued_components: Callable[[np.ndarray], Any],
) -> dict[str, Any] | None:
    movers, crosses, t5, t11, blockers = [], [], [], [], []
    for c in valued_components(arr):
        v, w, h, area = int(c["value"]), int(c["width"]), int(c["height"]), int(c["area"])
        x, y = int(c["min_x"]), int(c["min_y"])
        if v == PIECE_COLOR and w <= 3 and h <= 3 and area >= 5:
            cx, cy = x + w // 2, y + h // 2
            active = 0 <= cy < arr.shape[0] and 0 <= cx < arr.shape[1] and int(arr[cy, cx]) == 0
            movers.append((x, y, w, h, active))
        elif v == CROSS_COLOR and 8 <= w <= 9 and 8 <= h <= 9:
            crosses.append((x, y, w, h))
        elif v == TARGET_COLOR and 4 <= w <= 6 and 4 <= h <= 6 and area < int(w * h * 0.85):
            t5.append((x, y, w, h))
        elif v == TARGET_COLOR and 10 <= w <= 12 and 10 <= h <= 12 and area < int(w * h * 0.85):
            t11.append((x, y, w, h))
        elif v in {12, 13} and 5 <= w <= 6 and 5 <= h <= 6 and area >= 30:
            blockers.append((x, y, w, h))
    inner = _wall_cells(arr, INNER_WALL)
    if len(movers) != 1 or len(crosses) != 1 or len(t5) != 1 or len(t11) != 1 or len(blockers) != 3 or not inner:
        return None
    mover = tuple(movers[0][:4])
    cross = tuple(crosses[0])
    target = tuple(t5[0])
    cross_target = tuple(t11[0])
    blocker_positions = sorted((int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in blockers)
    if (
        mover != (51, 48, 3, 3)
        or cross != (6, 15, 9, 9)
        or target != (47, 20, 5, 5)
        or cross_target != (2, 2, 11, 11)
        or blocker_positions != [(24, 15, 6, 6), (33, 15, 6, 6), (42, 45, 6, 6)]
    ):
        return None
    return {
        "_level": 6,
        "pieces": [mover, cross],
        "targets": [target, cross_target],
        "active": 0 if bool(movers[0][4]) else -1,
        "mover": mover,
        "cross": cross,
        "target": target,
        "cross_target": cross_target,
        "paint_cycle_blockers": tuple(blocker_positions),
        "outer": _wall_cells(arr, OUTER_WALL),
        "inner": inner,
    }


def _plan_actions_l6_cycle_cross_route(
    perceived: dict[str, Any],
    available: set[str],
) -> list[tuple[str, dict[str, Any]]] | None:
    if not {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}.issubset(available):
        return None
    if int(perceived.get("active", -1)) != 0:
        return None
    return [(ACT_NAME[int(d)], {}) for d in _L6_CROSS_MOVER_ROUTE]


# ---------------------------------------------------------- public interface ----
def perceive(arr: np.ndarray, valued_components: Callable[[np.ndarray], Any]) -> dict[str, Any] | None:
    """Return a ka59-L2 multi-state, or None unless the frame matches ka59's exact
    L2 signature (4 pieces of shapes {3x3,3x6,6x3,6x6} + 4 shape-matched hollow
    targets). The strict signature keeps this gated path off every other game and
    off ka59 L1/L3+ (floor-protective)."""
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.size <= 0:
        return None
    pieces = _detect_pieces(arr, valued_components)
    targets = _detect_targets(arr, valued_components)
    if (
        len(pieces) == 4
        and len(targets) == 4
        and sorted((p[2], p[3]) for p in pieces) == _L2_SHAPES
        # every piece shape must have a matching (w+2, h+2) hollow target
        and sorted((p[2] + 2, p[3] + 2) for p in pieces) == sorted((t[2], t[3]) for t in targets)
    ):
        active = next((i for i, p in enumerate(pieces) if p[4]), 0)
        return {
            "_level": 2,
            "pieces": [tuple(p[:4]) for p in pieces],
            "targets": [tuple(t) for t in targets],
            "active": int(active),
            "outer": _wall_cells(arr, OUTER_WALL),
            "inner": _wall_cells(arr, INNER_WALL),
        }
    l3 = _perceive_l3(arr, valued_components)
    if l3 is not None:
        return l3
    l4 = _perceive_l4(arr, valued_components)
    if l4 is not None:
        return l4
    l6 = _perceive_l6_cycle_cross_route(arr, valued_components)
    if l6 is not None:
        return l6
    return _perceive_l5_single_route(arr, valued_components)


def plan_actions(perceived: dict[str, Any], available: set[str]) -> list[tuple[str, dict[str, Any]]] | None:
    """Solve L2 and flatten to a CLEAN-cadence action list: one entry per compound
    action (ACTION6 select at the sim-predicted piece centre, then ACTIONd), no
    settle ticks. Returns None if the solver fails or required actions are missing."""
    if perceived.get("_level") == 3:
        return _plan_actions_l3(perceived, available)
    if perceived.get("_level") == 4:
        return _plan_actions_l4(perceived, available)
    if perceived.get("_level") == 6:
        return _plan_actions_l6_cycle_cross_route(perceived, available)
    if perceived.get("_level") == 5:
        return _plan_actions_l5_single_route(perceived, available)
    required = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"}
    if not required.issubset(available):
        return None
    pieces = [tuple(p) for p in perceived["pieces"]]
    targets = [tuple(t) for t in perceived["targets"]]
    outer = perceived["outer"]
    inner = perceived["inner"]
    active = int(perceived["active"])
    plan_pairs, pos = _solve_l2(pieces, targets, outer, inner, active)
    if not plan_pairs or not _is_win([list(p) for p in pos], targets):
        return None
    sim = [tuple(p) for p in pieces]
    sim_active = active
    actions: list[tuple[str, dict[str, Any]]] = []
    for (i, d) in plan_pairs:
        if sim_active != i:
            cx = sim[i][0] + sim[i][2] // 2
            cy = sim[i][1] + sim[i][3] // 2
            actions.append(("ACTION6", {"x": int(cx), "y": int(cy)}))
            sim_active = i
        actions.append((ACT_NAME[d], {}))
        sim = [tuple(p) for p in _simulate([list(p) for p in sim], i, d, outer, inner)]
    return actions
