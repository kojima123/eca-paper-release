"""Agent identification by action-contingency (autonomous, adapter-free).

The frame problem for a generic forward model: on a frame, several things
move per action — the AGENT (the thing my action controls), step counters,
scrolling backgrounds, walkers. A "dominant moving colour" heuristic
mis-attributes (it picks the counter / scroll / a walker). This module
isolates the agent by ACTION-CONTINGENCY:

    the AGENT is the colour whose per-action displacement DIRECTION tracks
    the action choice (ACTION1 vs ACTION2 vs ... give DIFFERENT
    displacements), while counters/scroll/walkers displace the SAME
    regardless of which action I pressed (or do not move at all).

So: feed per-(action, colour) displacement samples; the agent = the colour
with the largest spread of mean-displacement ACROSS actions (and which
actually moves). Its learned per-action displacement is the forward model.

No game-specific colours/coords/ids; pure structure over observed
action-effect history.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def color_centroids(arr: np.ndarray) -> dict[int, tuple[float, float, int]]:
    """Per-colour centroid (x, y) and cell count for one frame."""
    out: dict[int, tuple[float, float, int]] = {}
    a = np.asarray(arr)
    for c in np.unique(a):
        ys, xs = np.where(a == c)
        if len(xs):
            out[int(c)] = (float(xs.mean()), float(ys.mean()), int(len(xs)))
    return out


def frame_displacements(prev: np.ndarray, cur: np.ndarray,
                        min_count: int = 1, max_frac: float = 0.5
                        ) -> dict[int, tuple[float, float]]:
    """Per-colour centroid displacement prev->cur, for colours present in
    both with a stable cell count (so we track a moving sprite, not a
    growing/recoloring region). Skips the background sheet (> max_frac of
    the frame)."""
    a = np.asarray(prev); b = np.asarray(cur)
    if a.shape != b.shape:
        return {}
    total = a.size
    pc = color_centroids(a); cc = color_centroids(b)
    out: dict[int, tuple[float, float]] = {}
    for c, (px, py, pn) in pc.items():
        if c not in cc:
            continue
        cx, cy, cn = cc[c]
        if pn < min_count or pn > total * max_frac:
            continue
        # require a roughly conserved sprite (cell count stable) so the
        # centroid delta is a translation, not a size change.
        if abs(cn - pn) > max(2, 0.5 * pn):
            continue
        dx = round(cx - px, 2); dy = round(cy - py, 2)
        if abs(dx) > 0.1 or abs(dy) > 0.1:
            out[c] = (dx, dy)
    return out


@dataclass
class AgentIdentifier:
    """Accumulates per-(action, colour) displacement samples and reports the
    agent colour (max action-contingency) + its one-step forward model."""
    # colour -> action -> list[(dx,dy)]
    samples: dict[int, dict[str, list]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))

    def observe(self, action: str, disp_by_color: dict[int, tuple[float, float]]) -> None:
        if not action:
            return
        for color, (dx, dy) in disp_by_color.items():
            self.samples[color][action].append((float(dx), float(dy)))

    def observe_frames(self, action: str, prev: np.ndarray, cur: np.ndarray) -> None:
        self.observe(action, frame_displacements(prev, cur))

    def _mean_disp(self, color: int) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for action, lst in self.samples.get(color, {}).items():
            if lst:
                out[action] = (round(sum(s[0] for s in lst) / len(lst), 2),
                               round(sum(s[1] for s in lst) / len(lst), 2))
        return out

    def contingency(self, color: int) -> float:
        """Spread of mean per-action displacement across actions. High =
        the colour moves DIFFERENTLY depending on the action = controllable
        agent. ~0 = same displacement on every action (counter/scroll) or
        no movement."""
        md = self._mean_disp(color)
        if len(md) < 2:
            # only ever moved under one action: contingency = its magnitude
            # (a single directional affordance still beats a static colour)
            if len(md) == 1:
                (dx, dy), = md.values()
                return float((dx * dx + dy * dy) ** 0.5)
            return 0.0
        xs = [v[0] for v in md.values()]
        ys = [v[1] for v in md.values()]
        # spread = sum of per-axis (max-min); a counter has the same vector
        # for every action -> spread 0.
        return round((max(xs) - min(xs)) + (max(ys) - min(ys)), 3)

    def agent_color(self, min_contingency: float = 1.0) -> Optional[int]:
        best: Optional[int] = None
        best_c = 0.0
        for color in self.samples:
            c = self.contingency(color)
            if c > best_c:
                best_c = c; best = color
        if best is None or best_c < min_contingency:
            return None
        return best

    def forward_model(self, color: Optional[int] = None) -> dict[str, tuple[float, float]]:
        """Learned per-action mean displacement for the agent (the forward
        model used to plan which action reaches a target)."""
        if color is None:
            color = self.agent_color()
        if color is None:
            return {}
        return self._mean_disp(color)

    def summary(self) -> dict:
        ranked = sorted(((self.contingency(c), c) for c in self.samples), reverse=True)
        agent = self.agent_color()
        return {
            "agent_color": agent,
            "agent_forward_model": self.forward_model(agent) if agent is not None else {},
            "contingency_ranked": [(c, score) for score, c in ranked[:6]],
        }


# --------------------------------------------------------------------------- #
# Component-level agent identification (handles an agent that SHARES a colour
# with a large region or other same-colour sprites: track per connected
# COMPONENT instance, not per colour centroid). 2026-06-06.
# --------------------------------------------------------------------------- #
def _frame_components(arr: np.ndarray) -> list[dict]:
    """Connected components (per-colour 4-conn) -> [{color, cx, cy, n}],
    background sheet dropped. Uses arcgym.utils.grid_utils."""
    from arcgym.utils.grid_utils import find_connected_components
    a = np.asarray(arr)
    grid = a.tolist()
    if not grid or not grid[0]:
        return []
    cm = find_connected_components(grid)
    g: dict[int, dict] = {}
    for (r, c), cid in cm.items():
        e = g.get(cid)
        if e is None:
            e = {"val": int(grid[r][c]), "n": 0, "minr": r, "maxr": r, "minc": c, "maxc": c}
            g[cid] = e
        e["n"] += 1
        if r < e["minr"]: e["minr"] = r
        if r > e["maxr"]: e["maxr"] = r
        if c < e["minc"]: e["minc"] = c
        if c > e["maxc"]: e["maxc"] = c
    total = a.size
    out = []
    for e in g.values():
        if e["n"] >= total * 0.5:  # background / huge sheet
            continue
        out.append({"color": e["val"],
                    "cx": (e["minc"] + e["maxc"]) / 2.0,
                    "cy": (e["minr"] + e["maxr"]) / 2.0,
                    "n": e["n"]})
    return out


class ComponentAgentIdentifier:
    """Like AgentIdentifier but tracks per-COMPONENT instances across frames
    (multi-object chaining) so an agent that shares a colour with a region or
    other same-colour sprites is isolated. The agent = the tracked component
    whose per-action displacement DIRECTION varies with the action."""

    def __init__(self, match_radius: float = 9.0) -> None:
        self.match_radius = float(match_radius)
        # tracks: list of {color, x, y, n, samples: {action: [(dx,dy)]}}
        self._tracks: list[dict] = []

    def _match(self, comp: dict) -> Optional[dict]:
        best = None; bd = self.match_radius
        for t in self._tracks:
            if t["color"] != comp["color"]:
                continue
            if abs(t["n"] - comp["n"]) > max(3, 0.6 * t["n"]):
                continue
            d = ((t["x"] - comp["cx"]) ** 2 + (t["y"] - comp["cy"]) ** 2) ** 0.5
            if d < bd:
                bd = d; best = t
        return best

    def observe_frames(self, action: str, prev: np.ndarray, cur: np.ndarray) -> None:
        """Drop-in alias matching AgentIdentifier.observe_frames."""
        self.observe(action, prev, cur)

    def observe(self, action: str, prev: np.ndarray, cur: np.ndarray) -> None:
        comps = _frame_components(cur)
        used = set()
        for comp in comps:
            t = self._match(comp)
            if t is not None and id(t) not in used:
                used.add(id(t))
                dx = round(comp["cx"] - t["x"], 2); dy = round(comp["cy"] - t["y"], 2)
                if action and (abs(dx) > 0.1 or abs(dy) > 0.1):
                    t["samples"].setdefault(action, []).append((dx, dy))
                t["x"] = comp["cx"]; t["y"] = comp["cy"]; t["n"] = comp["n"]
            else:
                self._tracks.append({"color": comp["color"], "x": comp["cx"],
                                     "y": comp["cy"], "n": comp["n"], "samples": {}})

    def _mean_disp(self, t: dict) -> dict:
        out = {}
        for a, lst in t["samples"].items():
            if lst:
                out[a] = (round(sum(s[0] for s in lst) / len(lst), 2),
                          round(sum(s[1] for s in lst) / len(lst), 2))
        return out

    def _contingency(self, t: dict) -> float:
        md = self._mean_disp(t)
        if len(md) < 2:
            if len(md) == 1:
                (dx, dy), = md.values()
                return float((dx * dx + dy * dy) ** 0.5)
            return 0.0
        xs = [v[0] for v in md.values()]; ys = [v[1] for v in md.values()]
        return round((max(xs) - min(xs)) + (max(ys) - min(ys)), 3)

    def agent_track(self, min_contingency: float = 1.0) -> Optional[dict]:
        best = None; bc = 0.0
        for t in self._tracks:
            c = self._contingency(t)
            if c > bc:
                bc = c; best = t
        if best is None or bc < min_contingency:
            return None
        return best

    def agent_color(self, min_contingency: float = 1.0) -> Optional[int]:
        t = self.agent_track(min_contingency)
        return int(t["color"]) if t else None

    def forward_model(self) -> dict:
        t = self.agent_track()
        return self._mean_disp(t) if t else {}

    def summary(self) -> dict:
        t = self.agent_track()
        return {"agent_color": int(t["color"]) if t else None,
                "agent_xy": (round(t["x"], 1), round(t["y"], 1)) if t else None,
                "agent_size": t["n"] if t else None,
                "agent_forward_model": self._mean_disp(t) if t else {},
                "n_tracks": len(self._tracks)}
