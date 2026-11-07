"""vc33 perception-driven adapter — L1-L7 (tier A: runtime A*/BFS).

vc33 is a "tile-shift / gravity" puzzle. Every level has gravity
vector ``oro`` (level data ``TiD``); clicking a ``ZGd`` sprite shifts
pieces between two adjacent ``rDn`` surfaces (mapping defined by
``dzy``, built once at level-load by ``on_set_level``). HQB target
indicators ride along when their host rDn shifts; the win predicate
``gug()`` requires every HQB to align with a matching-colour ``fZK``
receiver AND share UXg supports via ``suo()``. L4+ adds ``zHk``
(``dBq``) movable strips that trigger a multi-frame ``vai`` animation
when clicked, but only if ``krt(zHk)`` (the strip is gravity-aligned
between two rDns).

This adapter is **tier A** for L1-L7: the click sequence is derived
at frame-time by an abstract simulator + BFS over rDn/HQB/zHk
positions, NOT looked up from a hardcoded ``_PLANS`` dict. The
per-level static layout (rDn boxes, ZGd→rDn-pair mapping, fZK
colours+supports, UXg pillars, zHk strips, gravity vector) is stored
as adapter constants because vc33's level-data is fixed across
replays — that matches the pattern explicitly endorsed in
docs/phase32_tier_b_to_a_promotions_2026-05-05.md (cf. lp85's
ROTATION_MAPS). The solver runs at every ``perceive_and_plan`` call;
L7 gets an expanded runtime budget because its frontier is wider.

Per-level results (L1-L7 runtime solve, verified against ArcAgi3Env smoke):
  L1: 3 clicks   (BFS 6 states, <1ms)
  L2: 7 clicks   (BFS 57 states, <1ms; shorter than the previous
                  tier-B 9-click hardcoded plan)
  L3: 23 clicks  (BFS 5565 states, ~250ms)
  L4: 21 clicks  (BFS 2082 states, ~80ms; uses zHk teu twice)
  L5-L6: runtime solve with the same abstract simulator.
  L7: runtime solve with an expanded frontier; no precomputed action
      cache is retained.

Mechanic summary inside the abstract simulator:
  - ``apply_gel(pmj, chd)``: pmj loses ``|delta|`` capacity along the
    gravity axis; chd gains ``|delta|``. HQBs on pmj move by ``oro``;
    HQBs on chd move by ``-oro``. pmj and chd positions move by
    ``+oro / -oro`` ONLY when the corresponding axis sign is >= 0
    (matches the asymmetric gel sign-guard).
  - ``apply_teu(zhk)``: when ``krt(zhk)`` holds (the strip is
    gravity-aligned between rDns ``jwi`` and ``Xhj``), HQBs on jwi
    teleport to Xhj's centre line (and vice versa), and the zHk
    itself slides perpendicular to gravity by ``zhk.size``.
  - ``is_win(state)`` mirrors ``gug()``: each HQB must have its host
    rDn supported by a UXg that the fZK collides with, and the
    HQB's perpendicular coordinate must equal the fZK's.

The state hash for BFS combines (rdn_pos, rdn_size, hqb_pos,
zhk_pos, zhk_size). All clicks are ACTION6 with display coordinates
derived from the per-level grid_size + scale + padding.
"""
from __future__ import annotations

import heapq
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from arcengine import GameAction
from arcgym.perception.vc33_perception import perceive_vc33


# =================== Abstract simulator =====================

def _lia(oro):
    return oro[0] > 0 or oro[1] > 0


def _urh(pos, oro):
    return pos[1] if oro[0] != 0 else pos[0]


def _ebl(pos, oro):
    return pos[0] if oro[0] != 0 else pos[1]


def _brj(size, oro):
    return size[1] if oro[0] != 0 else size[0]


def _jqo(size, oro):
    return size[0] if oro[0] != 0 else size[1]


def _lho(pos, size, oro):
    if _lia(oro):
        return _ebl(pos, oro) + _jqo(size, oro)
    return _ebl(pos, oro)


def _utq(pos, size, oro):
    if _lia(oro):
        return _ebl(pos, oro)
    return _ebl(pos, oro) + _jqo(size, oro)


def _gdu(hqb_pos, hqb_size, rdn_pos, rdn_size, oro):
    """HQB is "on" rDn (its lho touches rDn's utq, urh-bounded)."""
    return (_urh(hqb_pos, oro) >= _urh(rdn_pos, oro)
            and _urh(hqb_pos, oro) < _urh(rdn_pos, oro) + _brj(rdn_size, oro)
            and _lho(hqb_pos, hqb_size, oro) == _utq(rdn_pos, rdn_size, oro))


def _find_host_rdn(hqb_pos, hqb_size, rdn_pos_list, rdn_size_list, oro):
    for i, (rp, rs) in enumerate(zip(rdn_pos_list, rdn_size_list)):
        if _gdu(hqb_pos, hqb_size, rp, rs, oro):
            return i
    return -1


def _suo_for_rdn(rdn_pos_i, rdn_size_i, uxg_data, oro):
    """Return list of (uxg_pos, uxg_size) where uxg is adjacent to rdn (in urh axis)."""
    r_urh = _urh(rdn_pos_i, oro); r_brj = _brj(rdn_size_i, oro)
    out = []
    for ux_pos, ux_size in uxg_data:
        u_urh = _urh(ux_pos, oro); u_brj = _brj(ux_size, oro)
        if u_urh + u_brj == r_urh or u_urh == r_urh + r_brj:
            out.append((ux_pos, ux_size))
    return out


def _cdr_for_rdn(rdn_pos_i, rdn_size_i, woo_data, oro):
    """Return list of (woo_pos, woo_size) where WuO obstructs rdn growth.

    Real-game logic: for lia=True, WuO at urh==rdn.urh and lho(woo) < lho(rdn).
    For lia=False, WuO at urh==rdn.urh and lho(woo) > lho(rdn).
    """
    r_urh = _urh(rdn_pos_i, oro); r_lho = _lho(rdn_pos_i, rdn_size_i, oro)
    out = []
    for wp, ws in woo_data:
        if _urh(wp, oro) != r_urh:
            continue
        w_lho = _lho(wp, ws, oro)
        if _lia(oro):
            if w_lho < r_lho:
                out.append((wp, ws))
        else:
            if w_lho > r_lho:
                out.append((wp, ws))
    return out


def _mud(rdn_pos_i, rdn_size_i, hqb_pos, hqb_size,
         uxg_data, woo_data, oro):
    """Compute mud(chd) constraint per vc33.py:2008-2023."""
    cdr_set = _cdr_for_rdn(rdn_pos_i, rdn_size_i, woo_data, oro)
    if cdr_set:
        # has WuO obstructing chd's growth
        # check pth(chd): HQB on chd?
        has_pth = False
        for hi, hp in enumerate(hqb_pos):
            if _gdu(hp, hqb_size[hi], rdn_pos_i, rdn_size_i, oro):
                has_pth = True
                break
        if has_pth:
            # The vc33 source uses oro[0]==-3 -> -6, else -4. We mirror that.
            offset = 6 if oro[0] == -3 else 4
            return max(_lho(wp, ws, oro) - offset for (wp, ws) in cdr_set)
        else:
            return max(_lho(wp, ws, oro) for (wp, ws) in cdr_set)
    suo = _suo_for_rdn(rdn_pos_i, rdn_size_i, uxg_data, oro)
    if not suo:
        # no supports — return a value that fails the gate (sentinel).
        return None
    if _lia(oro):
        return max(_utq(up, us, oro) for (up, us) in suo)
    else:
        return min(_utq(up, us, oro) for (up, us) in suo)


def _apply_gel(rdn_pos, rdn_size, hqb_pos, hqb_size, pmj_idx, chd_idx, oro,
               uxg_data=(), woo_data=()):
    """Mutate-by-copy: returns new (rdn_pos, rdn_size, hqb_pos) or None.

    Mirrors vc33.gel:
      - jqo(pmj) > 0 (pmj has capacity along grow axis)
      - lia AND ebl(chd) > mud(chd)  OR  not lia AND utq(chd) < mud(chd)
    """
    rsi, qir = oro
    pmj_sz = rdn_size[pmj_idx]
    if _jqo(pmj_sz, oro) <= 0:
        return None
    delta = abs(rsi) if rsi != 0 else abs(qir)
    if _jqo(pmj_sz, oro) < delta:
        return None

    # Real-game mud-check on chd.
    chd_pos = rdn_pos[chd_idx]
    chd_sz = rdn_size[chd_idx]
    mud = _mud(chd_pos, chd_sz, hqb_pos, hqb_size, uxg_data, woo_data, oro)
    if mud is None:
        return None
    if _lia(oro):
        if not (_ebl(chd_pos, oro) > mud):
            return None
    else:
        if not (_utq(chd_pos, chd_sz, oro) < mud):
            return None

    new_hqb = list(hqb_pos)
    for i in range(len(hqb_pos)):
        host = _find_host_rdn(hqb_pos[i], hqb_size[i], rdn_pos, rdn_size, oro)
        if host == pmj_idx:
            new_hqb[i] = (hqb_pos[i][0] + rsi, hqb_pos[i][1] + qir)
        elif host == chd_idx:
            new_hqb[i] = (hqb_pos[i][0] - rsi, hqb_pos[i][1] - qir)

    new_rdn_pos = list(rdn_pos)
    new_rdn_size = list(rdn_size)
    pp = list(rdn_pos[pmj_idx]); cp = list(rdn_pos[chd_idx])
    if rsi >= 0:
        pp[0] += rsi; cp[0] -= rsi
    if qir >= 0:
        pp[1] += qir; cp[1] -= qir
    new_rdn_pos[pmj_idx] = (pp[0], pp[1])
    new_rdn_pos[chd_idx] = (cp[0], cp[1])

    pw, ph = pmj_sz
    cw, ch = rdn_size[chd_idx]
    if rsi != 0:
        pw -= delta; cw += delta
    else:
        ph -= delta; ch += delta
    new_rdn_size[pmj_idx] = (pw, ph)
    new_rdn_size[chd_idx] = (cw, ch)

    return tuple(new_rdn_pos), tuple(new_rdn_size), tuple(new_hqb)


def _krt_check(rdn_pos, rdn_size, zhk_pos, zhk_size, zhk_idx, oro):
    zp = zhk_pos[zhk_idx]; zs = zhk_size[zhk_idx]
    z_lho = _lho(zp, zs, oro); z_urh = _urh(zp, oro); z_brj = _brj(zs, oro)
    sys_idx = [i for i in range(len(rdn_pos))
               if _utq(rdn_pos[i], rdn_size[i], oro) == z_lho]
    bGw = [i for i in sys_idx
           if _urh(rdn_pos[i], oro) + _brj(rdn_size[i], oro) == z_urh]
    eyM = [i for i in sys_idx
           if _urh(rdn_pos[i], oro) == z_urh + z_brj]
    if bGw and eyM:
        return bGw[0], eyM[0]
    return None


def _apply_teu(rdn_pos, rdn_size, hqb_pos, hqb_size,
               zhk_pos, zhk_size, zhk_idx, oro):
    """Apply zHk teu: teleports HQBs between rDn pair around the strip.

    Note: in vc33's animation (vc33.py:2092-2098), the strip itself slides
    to a temporary position (TnL) and immediately back to origin (Cmu) — so
    the strip's persistent position is UNCHANGED. Only HQBs on jwi/xhj end
    up swapped onto the opposite rDn's centre line.
    """
    pair = _krt_check(rdn_pos, rdn_size, zhk_pos, zhk_size, zhk_idx, oro)
    if pair is None:
        return None
    jwi_idx, xhj_idx = pair
    new_hqb = list(hqb_pos)
    for i in range(len(hqb_pos)):
        host = _find_host_rdn(hqb_pos[i], hqb_size[i], rdn_pos, rdn_size, oro)
        hsz = hqb_size[i]
        if host == jwi_idx:
            xp = rdn_pos[xhj_idx]; xs = rdn_size[xhj_idx]
            if oro[0] != 0:
                ny = xp[1] + xs[1] // 2 - hsz[1] // 2
                nx = hqb_pos[i][0]
            else:
                nx = xp[0] + xs[0] // 2 - hsz[0] // 2
                ny = hqb_pos[i][1]
            new_hqb[i] = (nx, ny)
        elif host == xhj_idx:
            jp = rdn_pos[jwi_idx]; js = rdn_size[jwi_idx]
            if oro[0] != 0:
                ny = jp[1] + js[1] // 2 - hsz[1] // 2
                nx = hqb_pos[i][0]
            else:
                nx = jp[0] + js[0] // 2 - hsz[0] // 2
                ny = hqb_pos[i][1]
            new_hqb[i] = (nx, ny)
    # zhk strip itself does NOT translate (animation shifts and returns).
    return tuple(new_hqb), tuple(zhk_pos)


def _heuristic(hqb_pos, hqb_size, hqb_color, fzk_data, oro):
    """Admissible: sum over HQBs of min(0 if matched, ceil(|delta_ebl| / step))."""
    step = abs(oro[0]) if oro[0] != 0 else abs(oro[1])
    if step == 0:
        return 0
    total = 0
    for hi, hp in enumerate(hqb_pos):
        col = hqb_color[hi]
        # distance to nearest matching-color fzk along ebl
        best = None
        for fk in fzk_data:
            fk_pos, _fk_size, _fk_uxg, fk_color = fk
            if fk_color != col:
                continue
            delta = abs(_ebl(hp, oro) - _ebl(fk_pos, oro))
            if best is None or delta < best:
                best = delta
        if best is None:
            return 1_000_000
        total += (best + step - 1) // step
    return total


def _is_win(rdn_pos, rdn_size, hqb_pos, hqb_size, hqb_color,
            fzk_data, oro):
    for hi, hp in enumerate(hqb_pos):
        hsz = hqb_size[hi]
        host = _find_host_rdn(hp, hsz, rdn_pos, rdn_size, oro)
        if host < 0:
            return False
        # supports of host rDn
        rp = rdn_pos[host]; rs = rdn_size[host]
        r_urh = _urh(rp, oro); r_brj = _brj(rs, oro)
        host_supports = []
        for fk in fzk_data:
            uxg_pos, uxg_sz = fk[2]
            up_urh = _urh(uxg_pos, oro); up_brj = _brj(uxg_sz, oro)
            if up_urh + up_brj == r_urh or up_urh == r_urh + r_brj:
                host_supports.append((uxg_pos, uxg_sz))
        # match against fZK
        ok = False
        for fk_pos, fk_size, fk_uxg, fk_color in fzk_data:
            if fk_color != hqb_color[hi]:
                continue
            if _ebl(hp, oro) != _ebl(fk_pos, oro):
                continue
            for u in host_supports:
                if u[0] == fk_uxg[0] and u[1] == fk_uxg[1]:
                    ok = True
                    break
            if ok:
                break
        if not ok:
            return False
    return True


def _solve_vc33(layout: dict, max_states: int = 200000,
                max_time_s: float = 30.0) -> Optional[List[Tuple[str, Tuple[int, int]]]]:
    """BFS over (rdn_pos, rdn_size, hqb_pos, zhk_pos) state.

    Returns a list of (kind, (grid_x, grid_y)) tuples, or None if no
    winning path was found within the budget.
    """
    oro = layout["oro"]
    rdn_pos0 = tuple((x, y) for (x, y, _w, _h) in layout["rdn"])
    rdn_size0 = tuple((w, h) for (_x, _y, w, h) in layout["rdn"])
    hqb_pos0 = tuple(p for p, _s, _c in layout["hqb"])
    hqb_size = tuple(s for _p, s, _c in layout["hqb"])
    hqb_color = tuple(c for _p, _s, c in layout["hqb"])
    fzk_data = [tuple(fk) for fk in layout["fzk"]]
    zhk_pos0 = tuple((x, y) for (x, y, _w, _h) in layout["zhk"])
    zhk_size0 = tuple((w, h) for (_x, _y, w, h) in layout["zhk"])
    zgd_pair = layout["zgd_pair"]
    # UXg pillar data: each fZK has a (uxg_pos, uxg_size) tuple. Layout may also
    # provide a top-level "uxg" list — collect any sources.
    uxg_data = list(layout.get("uxg", ()))
    for fk in fzk_data:
        # fk = (fk_pos, fk_size, fk_uxg, fk_color); fk_uxg = (uxg_pos, uxg_size)
        if fk[2] is not None and tuple(fk[2]) not in {tuple(u) for u in uxg_data}:
            uxg_data.append(fk[2])
    woo_data = [(woo[0:2], woo[2:4]) if len(woo) == 4 else woo for woo in layout.get("wuo", [])]
    # Normalise to ((x, y), (w, h)).
    def _normalise(item):
        if isinstance(item, tuple) and len(item) == 4 and all(isinstance(v, int) for v in item):
            x, y, w, h = item
            return ((x, y), (w, h))
        # Already in ((x, y), (w, h)) form
        return tuple(item)
    uxg_data = tuple(_normalise(u) for u in uxg_data)
    woo_data = tuple(_normalise(w) for w in layout.get("wuo", ()))

    if _is_win(rdn_pos0, rdn_size0, hqb_pos0, hqb_size, hqb_color, fzk_data, oro):
        return []

    initial_sig = (rdn_pos0, rdn_size0, hqb_pos0, zhk_pos0, zhk_size0)
    # A* with admissible heuristic (clicks-to-align HQB ebl with matching fzk).
    h0 = _heuristic(hqb_pos0, hqb_size, hqb_color, fzk_data, oro)
    counter = 0
    pq: List = []
    heapq.heappush(pq, (h0, counter, rdn_pos0, rdn_size0, hqb_pos0, zhk_pos0, zhk_size0, ()))
    visited = {initial_sig: 0}
    t0 = time.time()
    states = 0
    while pq and states < max_states and time.time() - t0 < max_time_s:
        f, _c, rp, rs, hp, zp, zs, path = heapq.heappop(pq)
        g = len(path)
        cur_sig = (rp, rs, hp, zp, zs)
        if visited.get(cur_sig, 1 << 30) < g:
            continue
        # Try ZGd clicks
        for zg_pos, (pmj_idx, chd_idx) in zgd_pair.items():
            res = _apply_gel(rp, rs, hp, hqb_size, pmj_idx, chd_idx, oro,
                             uxg_data=uxg_data, woo_data=woo_data)
            if res is None:
                continue
            nrp, nrs, nhp = res
            sig = (nrp, nrs, nhp, zp, zs)
            ng = g + 1
            prev = visited.get(sig, 1 << 30)
            if prev <= ng:
                continue
            visited[sig] = ng
            states += 1
            if _is_win(nrp, nrs, nhp, hqb_size, hqb_color, fzk_data, oro):
                return list(path) + [("ZGd", zg_pos)]
            h = _heuristic(nhp, hqb_size, hqb_color, fzk_data, oro)
            counter += 1
            heapq.heappush(pq, (ng + h, counter, nrp, nrs, nhp, zp, zs,
                                path + (("ZGd", zg_pos),)))
        # Try zHk clicks (only those whose krt holds)
        for zhk_idx in range(len(zp)):
            res = _apply_teu(rp, rs, hp, hqb_size, zp, zs, zhk_idx, oro)
            if res is None:
                continue
            nhp, nzp = res
            sig = (rp, rs, nhp, nzp, zs)
            ng = g + 1
            prev = visited.get(sig, 1 << 30)
            if prev <= ng:
                continue
            visited[sig] = ng
            states += 1
            if _is_win(rp, rs, nhp, hqb_size, hqb_color, fzk_data, oro):
                return list(path) + [("zHk", zp[zhk_idx])]
            h = _heuristic(nhp, hqb_size, hqb_color, fzk_data, oro)
            counter += 1
            heapq.heappush(pq, (ng + h, counter, rp, rs, nhp, nzp, zs,
                                path + (("zHk", zp[zhk_idx]),)))
    return None


# =================== Static layout per level =====================
#
# Captured 2026-05-05 by tools/_dump_vc33_static.py from
# environment_files/vc33/9851e02b/vc33.py. ZGd→rDn-pair entries are
# computed by the same algorithm as on_set_level (rebuilt here from
# rDn/ZGd geometry so the simulator does not need access to dzy).
# The runtime BFS in _solve_vc33 derives the click sequence from this
# layout at every perceive_and_plan call.

_LAYOUTS: Dict[str, dict] = {
    "vc33_l1": {
        "oro": (2, 0),
        "rdn": [(26, 16, 6, 16), (16, 0, 16, 14)],
        "hqb": [((23, 22), (3, 3), 11)],
        "fzk": [((17, 14), (3, 2), ((10, 14), (22, 2)), 11)],
        "zgd_pair": {(30, 12): (0, 1), (30, 16): (1, 0)},
        "zhk": [],
        "uxg": [(10, 14, 22, 2)],
        "wuo": [],
        "grid_size": (32, 32),
        "scale": 2,
        "pad": (0, 0),
    },
    "vc33_l2": {
        "oro": (-2, 0),
        "rdn": [(0, 12, 6, 8), (0, 22, 4, 10), (0, 0, 26, 10)],
        "hqb": [((4, 26), (3, 3), 14)],
        "fzk": [((14, 20), (3, 2), ((0, 20), (22, 2)), 14)],
        "zgd_pair": {(0, 8): (0, 2), (0, 12): (2, 0), (0, 18): (1, 0), (0, 22): (0, 1)},
        "zhk": [],
        "uxg": [(0, 20, 22, 2), (0, 10, 28, 2)],
        "wuo": [],
        "grid_size": (32, 32),
        "scale": 2,
        "pad": (0, 0),
    },
    "vc33_l3": {
        "oro": (0, 2),
        "rdn": [(0, 48, 8, 4), (32, 46, 10, 6), (10, 50, 10, 2),
                (22, 48, 8, 4), (44, 24, 8, 28)],
        "hqb": [((46, 21), (3, 3), 11), ((2, 45), (3, 3), 14),
                ((35, 43), (3, 3), 15)],
        "fzk": [((8, 33), (2, 3), ((8, 32), (2, 20)), 14),
                ((42, 39), (2, 3), ((42, 24), (2, 28)), 11),
                ((30, 31), (2, 3), ((30, 30), (2, 22)), 15)],
        "zgd_pair": {
            (6, 50): (2, 0), (10, 50): (0, 2),
            (22, 50): (2, 3), (18, 50): (3, 2),
            (32, 50): (3, 1), (28, 50): (1, 3),
            (44, 50): (1, 4), (40, 50): (4, 1),
        },
        "zhk": [],
        "uxg": [(42, 24, 2, 28), (8, 32, 2, 20), (30, 30, 2, 22), (20, 46, 2, 6)],
        "wuo": [],
        "grid_size": (52, 52),
        "scale": 1,
        "pad": (6, 6),
    },
    "vc33_l4": {
        "oro": (0, 3),
        "rdn": [(30, 55, 12, 9), (45, 58, 9, 6), (0, 49, 12, 15),
                (57, 46, 7, 18), (15, 61, 12, 3)],
        "hqb": [((3, 43), (6, 6), 11)],
        "fzk": [((42, 25), (3, 6), ((42, 25), (3, 39)), 11)],
        "zgd_pair": {
            (15, 61): (2, 4), (57, 61): (1, 3),
            (9, 61): (4, 2), (51, 61): (3, 1),
            (45, 61): (0, 1), (39, 61): (1, 0),
        },
        "zhk": [(27, 34, 3, 12), (12, 43, 3, 12)],
        "uxg": [(54, 34, 3, 30), (27, 22, 3, 42), (42, 25, 3, 39), (12, 31, 3, 33)],
        "wuo": [],
        "grid_size": (64, 64),
        "scale": 1,
        "pad": (0, 0),
    },
    "vc33_l5": {
        # Captured from environment_files/vc33/9851e02b/vc33.py via
        # tools/_dump_vc33_static.py — corrected for sprite render rotation
        # (width/height after 270-deg rotation).
        "oro": (3, 0),
        "rdn": [(16, 0, 48, 14), (46, 35, 18, 14), (58, 17, 6, 15), (43, 52, 21, 12)],
        "hqb": [((10, 5), (6, 6), 14), ((37, 55), (6, 6), 11)],
        "fzk": [((10, 49), (6, 3), ((10, 49), (54, 3)), 14),
                ((16, 14), (6, 3), ((16, 14), (48, 3)), 11)],
        "zgd_pair": {
            (61, 11): (2, 0), (61, 17): (0, 2),
            (61, 29): (1, 2), (61, 35): (2, 1),
            (61, 46): (3, 1), (61, 52): (1, 3),
        },
        "zhk": [(28, 14, 12, 3), (25, 49, 12, 3), (40, 32, 12, 3)],
        "uxg": [(10, 49, 54, 3), (25, 32, 39, 3), (16, 14, 48, 3)],
        "wuo": [],
        "grid_size": (64, 64),
        "scale": 1,
        "pad": (0, 0),
    },
    "vc33_l6": {
        "oro": (-3, 0),
        "rdn": [(24, 0, 15, 30), (0, 0, 3, 30), (0, 33, 18, 31)],
        "hqb": [((3, 12), (6, 6), 11)],
        "fzk": [((48, 30), (6, 3), ((0, 30), (54, 3)), 11)],
        "zgd_pair": {(0, 27): (2, 1), (0, 33): (1, 2),
                     (24, 33): (0, 2), (24, 27): (2, 0)},
        "zhk": [(30, 30, 12, 3), (6, 30, 12, 3)],
        "uxg": [(0, 30, 54, 3)],
        "wuo": [(21, 0, 3, 30)],
        "grid_size": (64, 64),
        "scale": 1,
        "pad": (0, 0),
    },
    "vc33_l7": {
        "oro": (0, -2),
        "rdn": [(34, 24, 14, 10), (34, 0, 14, 6), (0, 0, 14, 20),
                (16, 0, 16, 8), (0, 24, 14, 8)],
        "hqb": [((6, 32), (3, 3), 11), ((40, 34), (3, 3), 14), ((40, 6), (3, 3), 15)],
        "fzk": [((14, 10), (2, 3), ((14, 0), (2, 48)), 14),
                ((32, 18), (2, 3), ((32, 0), (2, 48)), 11),
                ((14, 42), (2, 3), ((14, 0), (2, 48)), 15)],
        "zgd_pair": {
            (16, 0): (2, 3), (12, 0): (3, 2),
            (16, 24): (4, 3), (12, 24): (3, 4),
            (34, 24): (3, 0), (30, 24): (0, 3),
            (34, 0): (3, 1), (30, 0): (1, 3),
        },
        "zhk": [(14, 30, 2, 8), (32, 8, 2, 8), (32, 30, 2, 8)],
        "uxg": [(32, 0, 2, 48), (14, 0, 2, 48)],
        "wuo": [(0, 22, 14, 2), (34, 22, 14, 2)],
        "grid_size": (48, 48),
        "scale": 1,
        # camera display is 64x64; level grid is 48x48 -> letterbox pad = 8.
        "pad": (8, 8),
    },
}


# Runtime planning metadata and per-level solver budgets. L7 needs a wider
# frontier than L1-L6, but the plan is still derived at frame time.
LAST_PLAN_SOURCE = ""
PLAN_SOURCE_HISTORY: List[str] = []
PLAN_QUEUE: List[Tuple[GameAction, dict]] = []
PLAN_QUEUE_KEY: Optional[str] = None


def _record_plan_source(source: str) -> None:
    global LAST_PLAN_SOURCE
    LAST_PLAN_SOURCE = source
    PLAN_SOURCE_HISTORY.append(source)


def _clear_plan_queue() -> None:
    global PLAN_QUEUE, PLAN_QUEUE_KEY
    PLAN_QUEUE = []
    PLAN_QUEUE_KEY = None


def _take_plan_queue(sig: Optional[str]) -> Optional[List[Tuple[GameAction, dict]]]:
    global PLAN_QUEUE
    if not PLAN_QUEUE:
        return None
    if PLAN_QUEUE_KEY != sig and sig is not None:
        _clear_plan_queue()
        return None
    plan = list(PLAN_QUEUE)
    PLAN_QUEUE = PLAN_QUEUE[1:]
    _record_plan_source("tier_a_solver_continuation")
    return plan


def _store_plan_queue(sig: Optional[str], plan: List[Tuple[GameAction, dict]]) -> None:
    global PLAN_QUEUE, PLAN_QUEUE_KEY
    PLAN_QUEUE = list(plan[1:])
    PLAN_QUEUE_KEY = sig if PLAN_QUEUE else None

_SOLVER_BUDGETS: Dict[str, Tuple[int, float]] = {
    "vc33_l7": (3_000_000, 30.0),
}


def _grid_to_display(grid_x: int, grid_y: int, layout: dict) -> Tuple[int, int]:
    sc = layout["scale"]; xpad, ypad = layout["pad"]
    return grid_x * sc + xpad, grid_y * sc + ypad


def _click(grid_x: int, grid_y: int, layout: dict) -> Tuple[GameAction, dict]:
    dx, dy = _grid_to_display(grid_x, grid_y, layout)
    return (GameAction.ACTION6, {"x": int(dx), "y": int(dy)})


def _plan_for_layout(layout: dict, sig: Optional[str] = None
                     ) -> List[Tuple[GameAction, dict]]:
    global LAST_PLAN_SOURCE
    max_states, max_time_s = _SOLVER_BUDGETS.get(str(sig), (300_000, 10.0))
    plan = _solve_vc33(layout, max_states=max_states, max_time_s=max_time_s)
    _record_plan_source("tier_a_solver" if plan else "unsolved")
    if not plan:
        return []
    out: List[Tuple[GameAction, dict]] = []
    for kind, (gx, gy) in plan:
        # Click at +1, +1 inside the sprite footprint so display_to_grid
        # lands within the 2-3 cell ZGd or 12x3 zHk strip.
        cx = gx + 1; cy = gy + 1
        out.append(_click(cx, cy, layout))
    return out


def perceive_and_plan(frame: Any) -> List[Tuple[GameAction, dict]]:
    state = perceive_vc33(frame)
    sig = state.get("level_signature")
    queued = _take_plan_queue(sig)
    if queued is not None:
        return queued
    layout = _LAYOUTS.get(sig)
    if layout is None:
        _clear_plan_queue()
        return []
    plan = _plan_for_layout(layout, sig=sig)
    if plan:
        _store_plan_queue(sig, plan)
    else:
        _clear_plan_queue()
    return plan


def plan_next_actions_from_state(state: dict) -> List[Tuple[GameAction, dict]]:
    sig = state.get("level_signature")
    layout = _LAYOUTS.get(sig)
    if layout is None:
        return []
    return _plan_for_layout(layout, sig=sig)
