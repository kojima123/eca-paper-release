"""Generic affordance prober — the human "what are these objects and what can I
do with them?" first move, automated.

Given a fresh ARC-AGI-3 game, this probes (observation-only, NO source, NO
hardcoded colours/coords/ids) and returns a structured AffordanceMap that labels
the game's components by FUNCTIONAL ROLE:

  - agent       : the component that moves under directional actions (+ its
                  per-action displacement = the body's motor model)
  - editor_cells: cells that TOGGLE in place under a click (a small same-site
                  recolour) — a configuration surface (e.g. sc25's 3x3 grid)
  - destructible: a component that DISAPPEARS under a click (e.g. bp35's blocks)
  - buttons     : a click that causes a LARGE REMOTE change (e.g. vc33's pump)
  - inert       : a click that only ticks a counter / does nothing

This is the affordance-prior foundation for hypothesis-driven solving — it
recovers, automatically, the roles I had to find by hand for vc33/cn04/sc25/bp35.
It also handles the "first action absorbed" intro-frame by priming.

No deepcopy needed: each probe is a fresh OFFLINE env (reset + a few steps).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arc_agi  # noqa: E402
from arcgym.environments import ArcAgi3Env  # noqa: E402
from arcgym.eca.agent_identifier import _frame_components  # noqa: E402
from arcengine import GameAction  # noqa: E402


@dataclass
class AffordanceMap:
    available_actions: list = field(default_factory=list)
    prime_steps: int = 0
    agent: Optional[tuple] = None          # (color, x, y) of the movable agent
    move_model: dict = field(default_factory=dict)   # action_name -> (dx, dy)
    editor_cells: list = field(default_factory=list) # [(x, y)] toggle-in-place
    destructibles: list = field(default_factory=list)# [(color, x, y)] removed on click
    buttons: list = field(default_factory=list)      # [(x, y, changed)] large remote change
    inert_clicks: int = 0
    bg: int = 0

    def summary(self) -> dict:
        return {
            "available_actions": list(self.available_actions),
            "prime_steps": self.prime_steps,
            "agent": self.agent,
            "move_model": {k: list(v) for k, v in self.move_model.items()},
            "n_editor_cells": len(self.editor_cells),
            "n_destructibles": len(self.destructibles),
            "n_buttons": len(self.buttons),
            "inert_clicks": self.inert_clicks,
        }


def _frame(o):
    a = np.asarray(o.get("frame"))
    return (a[-1] if a.ndim == 3 else a).astype(np.int16)


def _bg(arr):
    v, c = np.unique(arr, return_counts=True)
    return int(v[int(np.argmax(c))])


def _comps(arr, bg):
    out = []
    for c in _frame_components(arr):
        if c["color"] == bg:
            continue
        out.append((int(c["color"]), round(c["cx"]), round(c["cy"]), int(c["n"])))
    return out


def _new(game, max_actions=300):
    return ArcAgi3Env(game_id=game, max_actions=max_actions,
                      operation_mode=arc_agi.OperationMode.OFFLINE)


def _changed(a, b):
    return int(np.sum(a != b))


def _move_actions(available):
    # directional actions are the non-click available actions (ACTION6 = click)
    names = []
    for a in available:
        if int(a) == 6:
            continue
        names.append("ACTION%d" % int(a))
    return names


def _detect_prime(game, available):
    """Return how many leading actions are absorbed (produce no frame change)."""
    env = _new(game)
    p = _frame(env.reset())
    a = _move_actions(available) or ["ACTION6"]
    spec = a[0]
    payload = {"action": GameAction.from_name(spec).value}
    if spec == "ACTION6":
        payload.update(x=0, y=0)
    o2, _, _ = env.step(payload)
    env.close()
    return 1 if _changed(p, _frame(o2)) == 0 else 0


def _agent_centroid(arr, bg, color):
    cs = [c for c in _comps(arr, bg) if c[0] == color]
    return cs


def probe_affordances(game: str, max_click_sites: int = 22) -> AffordanceMap:
    base = _new(game)
    o0 = base.reset()
    available = list(o0.get("available_actions") or [1, 2, 3, 4, 5, 6])
    f0 = _frame(o0)
    bg = _bg(f0)
    base.close()
    amap = AffordanceMap(available_actions=available, bg=bg)
    amap.prime_steps = _detect_prime(game, available)

    def primed_env():
        env = _new(game)
        env.reset()
        for _ in range(amap.prime_steps):
            ma = _move_actions(available)
            spec = ("ACTION6", 0, 0) if not ma else (ma[0],)
            pl = {"action": GameAction.from_name(spec[0]).value}
            if len(spec) == 3:
                pl.update(x=spec[1], y=spec[2])
            env.step(pl)
        return env

    # post-prime baseline frame + components
    env = primed_env()
    o = env.step({"action": GameAction.ACTION6.value, "x": 0, "y": 0})[0]  # inert read
    fb = _frame(o)
    env.close()
    comps0 = _comps(fb, bg)

    # 1) AGENT detection: the component whose displacement VARIES with the action
    # (a UI element / camera-scroll moves the SAME way for every action; the
    # agent responds DIFFERENTLY to different directions). Per colour, collect
    # the displacement under each action, then prefer the colour with the most
    # DISTINCT displacements (and the smallest footprint), keeping only small,
    # plausibly-agent-sized components (not the whole scrolled field).
    move_names = _move_actions(available)
    disp_by_color = {}  # color -> {action: (dx,dy)}
    size_by_color = {co: n for (co, x, y, n) in comps0}
    for a in move_names:
        env = primed_env()
        before = _frame(env.step({"action": GameAction.ACTION6.value, "x": 0, "y": 0})[0])
        oo, _, _ = env.step({"action": GameAction.from_name(a).value})
        after = _frame(oo)
        env.close()
        b_by = {co: (x, y) for (co, x, y, n) in _comps(before, bg)}
        for (co, x, y, n) in _comps(after, bg):
            if co in b_by and (x, y) != b_by[co]:
                dx, dy = x - b_by[co][0], y - b_by[co][1]
                if abs(dx) + abs(dy) > 0:
                    disp_by_color.setdefault(co, {})[a] = (dx, dy)
    if disp_by_color:
        def agent_score(co):
            disps = disp_by_color[co]
            distinct = len(set(disps.values()))
            # small footprint preferred; reject field-sized blobs
            sz = size_by_color.get(co, 999)
            return (distinct, -sz)
        # prefer a colour with >=2 distinct displacements and a modest size
        cands = [co for co in disp_by_color
                 if size_by_color.get(co, 0) <= max(40, (f0.size // 20))]
        pool = cands or list(disp_by_color)
        agent_color = max(pool, key=agent_score)
        amap.move_model = dict(disp_by_color[agent_color])
        cs = _agent_centroid(fb, bg, agent_color)
        if cs:
            c = min(cs, key=lambda t: t[3])
            amap.agent = (agent_color, c[1], c[2])

    # 2) CLICK classification on each component centroid
    sites = sorted(comps0, key=lambda t: t[3])[:max_click_sites]
    changes = []
    raw = []
    for (co, x, y, n) in sites:
        env = primed_env()
        before = _frame(env.step({"action": GameAction.ACTION6.value, "x": 0, "y": 0})[0])
        oo, _, _ = env.step({"action": GameAction.ACTION6.value, "x": int(x), "y": int(y)})
        after = _frame(oo)
        env.close()
        ch = _changed(before, after)
        # remote change fraction: cells far from the click
        ys, xs = np.where(before != after)
        remote = 0
        for cx, cy in zip(xs.tolist(), ys.tolist()):
            if abs(cx - x) + abs(cy - y) > 8:
                remote += 1
        # what is the clicked cell AFTER? bg => destroyed/emptied; other non-bg => toggled
        def cellcolor(arr):
            xa, ya = int(np.clip(x, 0, arr.shape[1] - 1)), int(np.clip(y, 0, arr.shape[0] - 1))
            return int(arr[ya, xa])
        c_before, c_after = cellcolor(before), cellcolor(after)
        recolored = (c_after != bg and c_after != c_before)
        # net non-bg pixel change: a DESTROY removes pixels (drop); a TOGGLE
        # recolours non-bg -> non-bg (≈preserved).
        nonbg_drop = int((before != bg).sum()) - int((after != bg).sum())
        raw.append((co, x, y, n, ch, remote, recolored, nonbg_drop))
        changes.append(ch)

    med = sorted(changes)[len(changes) // 2] if changes else 1
    big_thr = max(40, 8 * (med or 1))
    for (co, x, y, n, ch, remote, recolored, nonbg_drop) in raw:
        if ch <= 2:
            amap.inert_clicks += 1
        elif remote >= big_thr:
            amap.buttons.append((x, y, ch))
        elif nonbg_drop >= max(8, int(n * 0.4)):
            amap.destructibles.append((co, x, y))  # a chunk of pixels removed
        elif recolored:
            amap.editor_cells.append((x, y))        # toggled in place (non-bg -> non-bg)
        else:
            amap.inert_clicks += 1
    amap.buttons.sort(key=lambda t: -t[2])
    return amap


if __name__ == "__main__":
    g = sys.argv[1] if len(sys.argv) > 1 else "sc25"
    import json
    print(json.dumps(probe_affordances(g).summary(), indent=2))
