"""Generic grow-drag solver (autonomous, adapter-free).

A frame-only, probe-based induction controller for the "grow-a-piece-to-
drag-a-carried-sprite-onto-a-goal-marker" mechanic family. It learns the
mechanic at RUNTIME by probing — clicking a candidate control box once and
diffing the frame — with NO game-specific colour names, coordinates, or
ids, and NO reading of any <game>.py source. The growth delta and
direction, the box->piece binding, and which small sprites are carried vs
goals are all inferred from observed click effects.

Designed as a per-call step-machine so it can drive
`PrimitiveSynthesisLoop.next_decision` (one ACTION6 click per call). It
self-gates: if probing finds no growable piece, `next_click` returns None
forever and the host loop falls back to its normal logic.

Default-off in the host loop (env ENABLE_GENERIC_GROW_DRAG); ships without
effect when the flag is unset.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _flatten(frame: Any) -> np.ndarray:
    a = np.asarray(frame)
    if a.ndim == 3:
        a = a[-1]
    return a.astype(np.int16, copy=False)


def _components(arr: np.ndarray) -> Tuple[int, List[Dict[str, Any]]]:
    """4-connected same-colour components, excluding the background
    (most frequent colour). Pure structure — no colour priors."""
    H, W = arr.shape
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])
    seen = np.zeros_like(arr, dtype=bool)
    out: List[Dict[str, Any]] = []
    for y in range(H):
        row = arr[y]
        for x in range(W):
            if seen[y, x] or int(row[x]) == bg:
                continue
            col = int(row[x])
            st = [(y, x)]
            cells: List[Tuple[int, int]] = []
            while st:
                yy, xx = st.pop()
                if yy < 0 or yy >= H or xx < 0 or xx >= W or seen[yy, xx]:
                    continue
                if int(arr[yy, xx]) != col:
                    continue
                seen[yy, xx] = True
                cells.append((yy, xx))
                st += [(yy + 1, xx), (yy - 1, xx), (yy, xx + 1), (yy, xx - 1)]
            xs = [c[1] for c in cells]
            ys = [c[0] for c in cells]
            out.append({
                "color": col, "x": min(xs), "y": min(ys),
                "w": max(xs) - min(xs) + 1, "h": max(ys) - min(ys) + 1,
                "n": len(cells), "cx": (min(xs) + max(xs)) // 2,
                "cy": (min(ys) + max(ys)) // 2,
            })
    return bg, out


def _inside(c: Dict[str, Any], box: Dict[str, Any]) -> bool:
    return (box["x"] <= c["x"] and c["x"] + c["w"] <= box["x"] + box["w"]
            and box["y"] <= c["y"] and c["y"] + c["h"] <= box["y"] + box["h"])


def _detect_sliders(comps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # compact control boxes: both dims small-ish, reasonably filled.
    return [c for c in comps
            if 6 <= c["w"] <= 16 and 6 <= c["h"] <= 16 and c["n"] >= 25]


def _small_sprites(comps: List[Dict[str, Any]],
                   sliders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in comps
            if c["n"] <= 12 and not any(_inside(c, b) for b in sliders)]


def _elongated_pieces(comps: List[Dict[str, Any]],
                      sliders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for c in comps:
        if c in sliders or c["n"] < 5 or c["n"] > 200:
            continue
        if any(_inside(c, b) for b in sliders):
            continue
        if min(c["w"], c["h"]) <= 4 and max(c["w"], c["h"]) >= 3:
            out.append(c)
    return out


def _structural_gate(comps: List[Dict[str, Any]],
                     sliders: List[Dict[str, Any]]) -> bool:
    """Non-destructive pre-probe gate: only engage on a frame that shows
    the grow-drag family signature — a small number of control boxes, at
    least one small 'carried' sprite sitting ON an elongated piece, and at
    least one standalone 'goal' sprite, with the standalone count not
    dwarfing the carried count. Excludes toggle/canvas/push games whose
    boxes would otherwise be probed destructively. Structural counts only;
    no colours / coords / ids."""
    if not (1 <= len(sliders) <= 3):
        return False
    pieces = _elongated_pieces(comps, sliders)
    if not pieces:
        return False
    small = _small_sprites(comps, sliders)

    def _on_piece(c):
        return any(p["x"] - 1 <= c["cx"] <= p["x"] + p["w"]
                   and p["y"] - 1 <= c["cy"] <= p["y"] + p["h"] for p in pieces)
    carried = [c for c in small if _on_piece(c)]
    standalone = [c for c in small if not _on_piece(c)]
    if len(carried) < 1 or len(standalone) < 1:
        return False
    if len(standalone) > 3 * len(carried):
        return False
    return True


def _match_grown(pc0: List[Dict[str, Any]], pc1: List[Dict[str, Any]]
                 ) -> Optional[Tuple[int, Tuple[int, int], int]]:
    """Return (piece_color, (dx,dy) grow-dir, delta) for a component that
    expanded between pc0 and pc1, else None."""
    for c1 in pc1:
        cands = [c0 for c0 in pc0 if c0["color"] == c1["color"]]
        if not cands:
            continue
        c0 = min(cands, key=lambda z: abs(z["x"] - c1["x"]) + abs(z["y"] - c1["y"]))
        da = (c1["w"] * c1["h"]) - (c0["w"] * c0["h"])
        if da <= 0:
            continue
        if c1["w"] > c0["w"]:
            ddir = (1, 0) if c1["x"] == c0["x"] else (-1, 0)
            return (c1["color"], ddir, c1["w"] - c0["w"])
        if c1["h"] > c0["h"]:
            ddir = (0, 1) if c1["y"] == c0["y"] else (0, -1)
            return (c1["color"], ddir, c1["h"] - c0["h"])
    return None


class GenericGrowDragController:
    """Per-call step-machine. Feed the current frame each call; get the
    next (x, y) ACTION6 click or None (yield to host)."""

    def __init__(self, max_grow_per_piece: int = 16) -> None:
        self._phase = "init"          # init -> probe -> plan -> execute -> done
        self._max_grow = max_grow_per_piece
        self._frame0: Optional[np.ndarray] = None
        self._sliders: List[Dict[str, Any]] = []
        self._probe_jobs: List[Tuple[Dict[str, Any], Tuple[int, int]]] = []
        self._pending: Optional[Tuple[Dict[str, Any], Tuple[int, int], np.ndarray]] = None
        self._learned: List[Dict[str, Any]] = []
        self._plan: List[Tuple[int, int]] = []
        self.engaged = False           # set True once a growable piece is confirmed
        self.fired = 0

    # -- helpers -------------------------------------------------------
    def _probe_edges(self, s: Dict[str, Any]) -> List[Tuple[int, int]]:
        if s["w"] >= s["h"]:
            return [(s["x"] + s["w"] - 1, s["y"] + s["h"] // 2),
                    (s["x"], s["y"] + s["h"] // 2)]
        return [(s["x"] + s["w"] // 2, s["y"] + s["h"] - 1),
                (s["x"] + s["w"] // 2, s["y"])]

    def _build_plan(self, arrN: np.ndarray) -> None:
        bg, compsN = _components(arrN)
        learn_by_color = {r["piece_color"]: r for r in self._learned}
        small0 = _small_sprites(_components(self._frame0)[1], self._sliders)
        smallN = _small_sprites(compsN, self._sliders)
        # carried = small sprite whose centre moved; goal = static sprite.
        moved: List[Dict[str, Any]] = []
        static: List[Dict[str, Any]] = []
        usedb = set()
        for ca in small0:
            best = None
            bd = 99
            for j, cb in enumerate(smallN):
                if j in usedb or cb["color"] != ca["color"]:
                    continue
                d = abs(cb["cx"] - ca["cx"]) + abs(cb["cy"] - ca["cy"])
                if d < bd:
                    bd = d
                    best = (j, cb)
            if best is None:
                continue
            usedb.add(best[0])
            (moved if bd > 0 else static).append(best[1] if bd > 0 else ca)
        if not moved:
            return
        piece_now: Dict[int, Dict[str, Any]] = {}
        for r in self._learned:
            cands = [c for c in compsN if c["color"] == r["piece_color"]
                     and not any(_inside(c, b) for b in self._sliders) and c["n"] >= 5]
            if cands:
                piece_now[r["piece_color"]] = max(cands, key=lambda z: z["n"])
        goals = [(c["cx"], c["cy"]) for c in static]
        if not goals:
            return
        carried = []
        for cN in moved:
            parent = None
            for col, pc in piece_now.items():
                if (pc["x"] - 1 <= cN["cx"] <= pc["x"] + pc["w"]
                        and pc["y"] - 1 <= cN["cy"] <= pc["y"] + pc["h"]):
                    parent = col
                    break
            if parent is None and learn_by_color:
                parent = next(iter(learn_by_color))
            if parent is not None:
                carried.append({"cur": (cN["cx"], cN["cy"]), "parent": parent})
        need: Dict[int, int] = {}
        usedg = set()
        for cs in carried:
            r = learn_by_color[cs["parent"]]
            dx, dy = r["dir"]
            delta = r["delta"]
            cx, cy = cs["cur"]
            best_k = None
            best_g = None
            for gi, (gx, gy) in enumerate(goals):
                if gi in usedg:
                    continue
                if dx != 0:
                    if abs(gy - cy) > 1 or delta == 0 or (gx - cx) % (dx * delta) != 0:
                        continue
                    k = (gx - cx) // (dx * delta)
                else:
                    if abs(gx - cx) > 1 or delta == 0 or (gy - cy) % (dy * delta) != 0:
                        continue
                    k = (gy - cy) // (dy * delta)
                if k < 0 or k > self._max_grow:
                    continue
                if best_k is None or k < best_k:
                    best_k = k
                    best_g = gi
            if best_g is not None:
                usedg.add(best_g)
                need[cs["parent"]] = max(need.get(cs["parent"], 0), best_k)
        plan: List[Tuple[int, int]] = []
        for col, k in need.items():
            gx, gy = learn_by_color[col]["grow_click"]
            plan += [(gx, gy)] * k
        self._plan = plan

    # -- main step -----------------------------------------------------
    def next_click(self, frame: Any) -> Optional[Tuple[int, int]]:
        arr = _flatten(frame)

        if self._phase == "init":
            self._frame0 = arr.copy()
            bg, comps = _components(arr)
            self._sliders = _detect_sliders(comps)
            # Non-destructive structural gate: only probe frames that show
            # the grow-drag family signature. Protects non-grow-drag games
            # (toggle/push/canvas) from destructive probe clicks.
            if not self._sliders or not _structural_gate(comps, self._sliders):
                self._phase = "done"
                return None
            for s in self._sliders:
                for e in self._probe_edges(s):
                    self._probe_jobs.append((s, e))
            self._phase = "probe"
            # fall through to probe

        if self._phase == "probe":
            # First, resolve the previous probe's effect (if any).
            if self._pending is not None:
                s, edge, pre = self._pending
                self._pending = None
                grown = _match_grown(_components(pre)[1], _components(arr)[1])
                if grown:
                    self.engaged = True
                    self._learned.append({
                        "slider": s, "grow_click": edge,
                        "piece_color": grown[0], "dir": grown[1],
                        "delta": grown[2],
                    })
                    # stop probing remaining edges of this same slider
                    self._probe_jobs = [(ss, ee) for (ss, ee) in self._probe_jobs
                                        if ss is not s]
            # Issue the next probe job.
            while self._probe_jobs:
                s, edge = self._probe_jobs.pop(0)
                # skip sliders we already learned
                if any(r["slider"] is s for r in self._learned):
                    continue
                self._pending = (s, edge, arr.copy())
                self.fired += 1
                return (int(edge[0]), int(edge[1]))
            # probing exhausted -> plan
            if not self._learned:
                self._phase = "done"
                return None
            self._build_plan(arr)
            self._phase = "execute"

        if self._phase == "execute":
            if self._plan:
                x, y = self._plan.pop(0)
                self.fired += 1
                return (int(x), int(y))
            self._phase = "done"
            return None

        return None
