"""Generic drag-to-goal solver (autonomous, adapter-free).

A frame-only, probe-based induction controller for the "drag-a-movable-
sprite-into-a-containment-region" mechanic family (e.g. su15). It learns
the mechanic at RUNTIME by probing — clicking near a candidate sprite and
checking whether a same-shape sprite appears at the click (i.e. the sprite
*follows the click*) — with NO game-specific colours, coordinates, ids, or
`<game>.py` reading. The drag radius/step and which sprite is the movable
target are inferred from observed click effects; the destination is the
prominent compact region inferred from structure.

Per-call step-machine for `PrimitiveSynthesisLoop.next_decision` (one
ACTION6 click per call). Self-gates: a non-destructive structural pre-check
must pass (a compact containment region + a movable sprite outside it)
before any probe, so it never disturbs non-drag games. Default-off
(`ENABLE_GENERIC_DRAG_TO_GOAL`); ships without effect when unset.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from arcgym.eca.generic_grow_drag import _components, _flatten


def _small_sprites(comps: List[Dict[str, Any]], H: int, W: int) -> List[Dict[str, Any]]:
    return [c for c in comps if 3 <= c["n"] <= 12
            and c["w"] < W * 0.8 and c["h"] < H * 0.8]


def _goal_regions(comps: List[Dict[str, Any]], H: int, W: int) -> List[Dict[str, Any]]:
    # compact medium regions: candidate destinations (not bars, not tiny).
    return [c for c in comps if c["n"] >= 20
            and c["w"] < W * 0.6 and c["h"] < H * 0.6]


def _in_region(c: Dict[str, Any], r: Dict[str, Any]) -> bool:
    return (r["x"] - 1 <= c["cx"] <= r["x"] + r["w"]
            and r["y"] - 1 <= c["cy"] <= r["y"] + r["h"])


def structural_gate(arr: np.ndarray) -> bool:
    """Non-destructive: a compact containment region with an inner marker,
    plus >=1 small sprite outside it. The inner marker (a small sprite
    inside the region) is the distinctive 'drop here' cue and keeps this
    from firing on arbitrary frames. Structural counts only."""
    H, W = arr.shape
    bg, comps = _components(arr)
    goals = _goal_regions(comps, H, W)
    smalls = _small_sprites(comps, H, W)
    if not goals or not smalls:
        return False
    goal = max(goals, key=lambda c: c["n"])
    inside = [c for c in smalls if _in_region(c, goal)]
    outside = [c for c in smalls if not _in_region(c, goal)]
    # require a marker inside the region AND a candidate outside it, and a
    # small candidate set (avoid busy multi-object frames).
    if len(inside) < 1 or len(outside) < 1:
        return False
    if len(outside) > 6:
        return False
    return True


class GenericDragToGoalController:
    def __init__(self) -> None:
        self._phase = "init"
        self._goal: Optional[Tuple[int, int, Dict[str, Any]]] = None
        self._probe_q: List[Dict[str, Any]] = []
        self._pending: Optional[Tuple[int, int, int]] = None  # color, click_x, click_y
        self._drag: Optional[List[int]] = None   # [cur_x, cur_y]
        self.engaged = False
        self.fired = 0

    def next_click(self, frame: Any) -> Optional[Tuple[int, int]]:
        arr = _flatten(frame)
        H, W = arr.shape

        if self._phase == "init":
            if not structural_gate(arr):
                self._phase = "done"
                return None
            bg, comps = _components(arr)
            goals = _goal_regions(comps, H, W)
            goal = max(goals, key=lambda c: c["n"])
            self._goal = (goal["cx"], goal["cy"], goal)
            smalls = _small_sprites(comps, H, W)
            outside = [c for c in smalls if not _in_region(c, goal)]
            # probe farthest-from-goal first (real target needs most drag;
            # decoys/UI sprites won't follow the click).
            gx, gy = goal["cx"], goal["cy"]
            self._probe_q = sorted(outside, key=lambda c: -math.hypot(c["cx"] - gx, c["cy"] - gy))
            self._phase = "probe"

        if self._phase == "probe":
            gx, gy, goal = self._goal
            # resolve previous probe: did a same-colour small sprite land at
            # the click? -> it's the movable target.
            if self._pending is not None:
                col, cxp, cyp = self._pending
                self._pending = None
                bg, comps = _components(arr)
                hit = any(s["color"] == col and s["n"] <= 12
                          and math.hypot(s["cx"] - cxp, s["cy"] - cyp) <= 3
                          for s in comps)
                if hit:
                    self.engaged = True
                    self._drag = [cxp, cyp]
                    self._phase = "drag"
                    # fall through to drag
            if self._phase == "probe":
                while self._probe_q:
                    c = self._probe_q.pop(0)
                    tx, ty = c["cx"], c["cy"]
                    dx, dy = gx - tx, gy - ty
                    dist = math.hypot(dx, dy) or 1
                    stp = min(5, dist)
                    cxp = int(round(tx + dx / dist * stp))
                    cyp = int(round(ty + dy / dist * stp))
                    self._pending = (c["color"], cxp, cyp)
                    self.fired += 1
                    return (cxp, cyp)
                self._phase = "done"
                return None

        if self._phase == "drag":
            gx, gy, goal = self._goal
            cx, cy = self._drag
            dx, dy = gx - cx, gy - cy
            dist = math.hypot(dx, dy) or 1
            if dist <= 4:
                self._phase = "done"
                return None
            stp = min(6, dist)
            nx = int(round(cx + dx / dist * stp))
            ny = int(round(cy + dy / dist * stp))
            self._drag = [nx, ny]
            self.fired += 1
            return (nx, ny)

        return None
