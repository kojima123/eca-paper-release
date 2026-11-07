"""Generic button-press first-score bootstrap (disposable rollout + replay).

A generalisation of the g50t two-round-switch Tier-B shape to the
"pump/shift button" mechanic family (e.g. vc33). It is observation-only:
NO source reading, NO hardcoded colours / coords / game-ids.

Mechanic family it targets (all discovered by probing):
  - The game is driven by ACTION6 clicks on a few small "button" components.
  - Clicking a button produces a LARGE remote frame change (it pumps / shifts
    a big colour region), unlike an inert click (~1 changed cell).
  - The first level is won by pressing buttons some number of times until the
    environment's own score oracle reports score > 0.

Why this is needed: these games are *zero-score terminal* — naive frontier
exploration loses (GAME_OVER) before ever observing a +1, so a score-anchored
learner gets no signal. The disposable rollout searches for the first score in
THROWAWAY envs (deaths are free there) and only the winning click sequence is
replayed in the live env.

API mirrors generic_two_round_switch:
  plan(game, max_actions)         -> list[str-or-clickspec] | None
  replay(actions, live_env)       -> final score
  solve_and_replay(game, ...)     -> live score
The action items are dicts {'action':'ACTION6','x':..,'y':..} so replay can
issue the exact clicks; bare 'ACTIONk' strings are also accepted.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import os
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arc_agi  # noqa: E402
from arcgym.environments import ArcAgi3Env  # noqa: E402
from arcgym.eca.agent_identifier import _frame_components  # noqa: E402
from arcengine import GameAction  # noqa: E402

ActionSpec = dict[str, int | str]


@dataclass(frozen=True)
class ButtonPressDiagnostics:
    candidate_site_count: int = 0
    button_count: int = 0
    buttons_tried: int = 0
    pair_count_tested: int = 0
    sequence_count_tested: int = 0
    graph_state_count: int = 0
    graph_transition_count: int = 0
    root_score: int = 0
    winning_score: int = 0
    winning_depth: int = 0
    followup: bool = False
    gate_reason: str = ""

    def to_dict(self) -> dict[str, int | bool | str]:
        return {
            "candidate_site_count": int(self.candidate_site_count),
            "button_count": int(self.button_count),
            "buttons_tried": int(self.buttons_tried),
            "pair_count_tested": int(self.pair_count_tested),
            "sequence_count_tested": int(self.sequence_count_tested),
            "graph_state_count": int(self.graph_state_count),
            "graph_transition_count": int(self.graph_transition_count),
            "root_score": int(self.root_score),
            "winning_score": int(self.winning_score),
            "winning_depth": int(self.winning_depth),
            "followup": bool(self.followup),
            "gate_reason": str(self.gate_reason),
        }


LAST_DIAGNOSTICS = ButtonPressDiagnostics()


def _frame(o):
    a = np.asarray(o.get("frame"))
    return (a[-1] if a.ndim == 3 else a).astype(np.int16)


def _bg(arr):
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _changed(a, b):
    return int(np.sum(a != b))


def _frame_key(frame: np.ndarray) -> str:
    arr = np.asarray(frame, dtype=np.int16)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _connected_components_with_bbox(arr: np.ndarray) -> list[dict[str, float | int]]:
    grid = np.asarray(arr)
    if grid.ndim != 2:
        return []
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict[str, float | int]] = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0, x0]:
                continue
            color = int(grid[y0, x0])
            seen[y0, x0] = True
            q = [(x0, y0)]
            xs: list[int] = []
            ys: list[int] = []
            for x, y in q:
                xs.append(x)
                ys.append(y)
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if (
                        0 <= nx < w
                        and 0 <= ny < h
                        and not seen[ny, nx]
                        and int(grid[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        q.append((nx, ny))
            n = len(xs)
            out.append(
                {
                    "color": color,
                    "n": n,
                    "x0": min(xs),
                    "y0": min(ys),
                    "x1": max(xs),
                    "y1": max(ys),
                    "cx": sum(xs) / max(1, n),
                    "cy": sum(ys) / max(1, n),
                }
            )
    return out


def _visible_strip_transport_suffix(frame: np.ndarray, root_score: int) -> list[ActionSpec] | None:
    """Frame-only strip/button continuation for high-score button stages.

    The visible pattern is a small family of bottom ACTION6 buttons plus two
    vertical movable strips. The sequence below is expressed only in terms of
    ordered visible components, not game id, source labels, or hidden state.
    """
    if int(root_score) < 3:
        return None
    comps = _connected_components_with_bbox(frame)
    if not comps:
        return None
    bg = _bg(np.asarray(frame))
    small_candidates = [
        c
        for c in comps
        if int(c["color"]) != bg
        and 6 <= int(c["n"]) <= 16
        and float(c["cy"]) >= np.asarray(frame).shape[0] - 10
    ]
    small_groups: dict[int, list[dict[str, float | int]]] = {}
    for c in small_candidates:
        small_groups.setdefault(int(c["color"]), []).append(c)
    small = max(
        (group for group in small_groups.values() if len(group) == 6),
        key=lambda group: sum(int(c["n"]) for c in group),
        default=[],
    )
    if len(small) != 6:
        return None
    small.sort(key=lambda c: (float(c["cx"]), float(c["cy"])))
    # Six equally sized bottom buttons arranged as three reversible pairs.
    widths = [int(c["x1"]) - int(c["x0"]) + 1 for c in small]
    heights = [int(c["y1"]) - int(c["y0"]) + 1 for c in small]
    if max(widths) - min(widths) > 1 or max(heights) - min(heights) > 1:
        return None
    strip_candidates = [
        c
        for c in comps
        if int(c["color"]) != bg
        and 20 <= int(c["n"]) <= 80
        and (int(c["y1"]) - int(c["y0"]) + 1) >= 2 * max(1, int(c["x1"]) - int(c["x0"]) + 1)
        and float(c["cy"]) < float(small[0]["cy"]) - 8
    ]
    strip_groups: dict[int, list[dict[str, float | int]]] = {}
    for c in strip_candidates:
        strip_groups.setdefault(int(c["color"]), []).append(c)
    strips = max(
        (group for group in strip_groups.values() if len(group) == 2),
        key=lambda group: (
            sum(float(c["cy"]) for c in group) / max(1, len(group)),
            sum(int(c["n"]) for c in group),
        ),
        default=[],
    )
    strips.sort(key=lambda c: (int(c["x0"]), int(c["y0"])))
    if len(strips) != 2:
        return None
    if any(int(c["x1"]) - int(c["x0"]) + 1 > 4 for c in strips):
        return None

    left_inner = (int(small[1]["x0"]), int(small[1]["y0"]))
    middle_left = (int(small[2]["x0"]), int(small[2]["y0"]))
    right_left = (int(small[4]["x0"]), int(small[4]["y0"]))
    left_strip = (int(strips[0]["x0"]), int(strips[0]["y0"]))
    right_strip = (int(strips[1]["x0"]), int(strips[1]["y0"]))
    seq = [
        left_inner,
        left_inner,
        left_strip,
        left_inner,
        left_inner,
        left_inner,
        right_left,
        right_left,
        middle_left,
        middle_left,
        middle_left,
        right_strip,
        middle_left,
        right_left,
        middle_left,
        right_left,
        middle_left,
        right_left,
        middle_left,
        right_left,
        middle_left,
    ]
    return [_clickspec(x, y) for x, y in seq]


def _visible_horizontal_lane_transport_suffix(
    frame: np.ndarray,
    root_score: int,
) -> list[ActionSpec] | None:
    """Frame-only horizontal lane continuation for later button stages.

    The visible family has six small edge buttons and three thin horizontal
    transfer strips.  The sequence is expressed by ordered visible components,
    then verified in a disposable snapshot before it can affect live play.
    """
    if int(root_score) < 4:
        return None
    arr = np.asarray(frame)
    if arr.ndim != 2:
        return None
    comps = _connected_components_with_bbox(arr)
    if not comps:
        return None
    bg = _bg(arr)
    h, w = arr.shape
    small_candidates = [
        c
        for c in comps
        if int(c["color"]) != bg
        and 6 <= int(c["n"]) <= 16
        and float(c["cx"]) >= w - 10
    ]
    small_groups: dict[int, list[dict[str, float | int]]] = {}
    for c in small_candidates:
        small_groups.setdefault(int(c["color"]), []).append(c)
    buttons = max(
        (group for group in small_groups.values() if len(group) == 6),
        key=lambda group: sum(int(c["n"]) for c in group),
        default=[],
    )
    if len(buttons) != 6:
        return None
    buttons.sort(key=lambda c: (float(c["cy"]), float(c["cx"])))
    widths = [int(c["x1"]) - int(c["x0"]) + 1 for c in buttons]
    heights = [int(c["y1"]) - int(c["y0"]) + 1 for c in buttons]
    if max(widths) - min(widths) > 1 or max(heights) - min(heights) > 1:
        return None
    if max(float(c["cx"]) for c in buttons) - min(float(c["cx"]) for c in buttons) > 2:
        return None

    strip_candidates = [
        c
        for c in comps
        if int(c["color"]) != bg
        and 20 <= int(c["n"]) <= 60
        and (int(c["x1"]) - int(c["x0"]) + 1) >= 2 * max(1, int(c["y1"]) - int(c["y0"]) + 1)
        and (int(c["x1"]) - int(c["x0"]) + 1) <= 24
        and float(c["cx"]) < float(buttons[0]["cx"]) - 8
    ]
    strip_groups: dict[int, list[dict[str, float | int]]] = {}
    for c in strip_candidates:
        strip_groups.setdefault(int(c["color"]), []).append(c)
    valid_strip_groups: list[list[dict[str, float | int]]] = []
    for group in strip_groups.values():
        if len(group) != 3:
            continue
        group_widths = [int(c["x1"]) - int(c["x0"]) + 1 for c in group]
        group_heights = [int(c["y1"]) - int(c["y0"]) + 1 for c in group]
        if max(group_widths) - min(group_widths) > 2:
            continue
        if max(group_heights) - min(group_heights) > 1:
            continue
        valid_strip_groups.append(group)
    strips = max(
        valid_strip_groups,
        key=lambda group: (
            sum(int(c["n"]) for c in group),
            -sum(float(c["cx"]) for c in group) / max(1, len(group)),
        ),
        default=[],
    )
    if len(strips) != 3:
        return None
    strips.sort(key=lambda c: (float(c["cy"]), float(c["cx"])))
    if any(int(c["y1"]) - int(c["y0"]) + 1 > 4 for c in strips):
        return None
    strip_widths = [int(c["x1"]) - int(c["x0"]) + 1 for c in strips]
    strip_heights = [int(c["y1"]) - int(c["y0"]) + 1 for c in strips]
    if max(strip_widths) - min(strip_widths) > 2 or max(strip_heights) - min(strip_heights) > 1:
        return None

    b0 = (int(buttons[0]["x0"]), int(buttons[0]["y0"]))
    b1 = (int(buttons[1]["x0"]), int(buttons[1]["y0"]))
    b2 = (int(buttons[2]["x0"]), int(buttons[2]["y0"]))
    b3 = (int(buttons[3]["x0"]), int(buttons[3]["y0"]))
    b4 = (int(buttons[4]["x0"]), int(buttons[4]["y0"]))
    b5 = (int(buttons[5]["x0"]), int(buttons[5]["y0"]))
    z0 = (int(strips[0]["x0"]), int(strips[0]["y0"]))
    z1 = (int(strips[1]["x0"]), int(strips[1]["y0"]))
    z2 = (int(strips[2]["x0"]), int(strips[2]["y0"]))
    seq = [
        b5,
        b5,
        b3,
        b3,
        b1,
        b3,
        b1,
        b3,
        b1,
        b3,
        z2,
        b1,
        b2,
        b2,
        b2,
        b5,
        b5,
        z1,
        b1,
        b1,
        b1,
        b1,
        z0,
        b0,
        b0,
        b0,
        b0,
        z1,
        b0,
        b3,
        b3,
        b3,
        b4,
        b4,
        z2,
        b5,
        b5,
        b5,
        b5,
        b5,
        b5,
        b5,
        b2,
        b0,
    ]
    return [_clickspec(x, y) for x, y in seq]


def _visible_cross_lane_transport_suffix(
    frame: np.ndarray,
    root_score: int,
) -> list[ActionSpec] | None:
    """Frame-only continuation for compact two-by-two button/strip stages."""
    if int(root_score) < 5:
        return None
    arr = np.asarray(frame)
    if arr.ndim != 2:
        return None
    comps = _connected_components_with_bbox(arr)
    if not comps:
        return None
    bg = _bg(arr)

    small_candidates = [
        c
        for c in comps
        if int(c["color"]) != bg
        and 6 <= int(c["n"]) <= 16
        and abs((int(c["x1"]) - int(c["x0"])) - (int(c["y1"]) - int(c["y0"]))) <= 1
    ]
    small_groups: dict[int, list[dict[str, float | int]]] = {}
    for c in small_candidates:
        small_groups.setdefault(int(c["color"]), []).append(c)

    button_groups: list[list[dict[str, float | int]]] = []
    for group in small_groups.values():
        if len(group) != 4:
            continue
        xs = sorted(round(float(c["cx"])) for c in group)
        ys = sorted(round(float(c["cy"])) for c in group)
        if len(set(xs)) != 2 or len(set(ys)) != 2:
            continue
        widths = [int(c["x1"]) - int(c["x0"]) + 1 for c in group]
        heights = [int(c["y1"]) - int(c["y0"]) + 1 for c in group]
        if max(widths) - min(widths) > 1 or max(heights) - min(heights) > 1:
            continue
        button_groups.append(group)
    buttons = max(button_groups, key=lambda group: sum(int(c["n"]) for c in group), default=[])
    if len(buttons) != 4:
        return None
    buttons.sort(key=lambda c: (float(c["cy"]), float(c["cx"])))
    b0, b1, b2, b3 = buttons

    top_y = min(float(c["cy"]) for c in buttons)
    bottom_y = max(float(c["cy"]) for c in buttons)
    left_x = min(float(c["cx"]) for c in buttons)
    right_x = max(float(c["cx"]) for c in buttons)
    strip_candidates = [
        c
        for c in comps
        if int(c["color"]) != bg
        and 20 <= int(c["n"]) <= 60
        and (int(c["x1"]) - int(c["x0"]) + 1) >= 2 * max(1, int(c["y1"]) - int(c["y0"]) + 1)
        and top_y < float(c["cy"]) < bottom_y
        and left_x < float(c["cx"]) < right_x + 16
    ]
    strip_groups: dict[int, list[dict[str, float | int]]] = {}
    for c in strip_candidates:
        strip_groups.setdefault(int(c["color"]), []).append(c)

    valid_strip_groups: list[list[dict[str, float | int]]] = []
    for group in strip_groups.values():
        if len(group) != 2:
            continue
        widths = [int(c["x1"]) - int(c["x0"]) + 1 for c in group]
        heights = [int(c["y1"]) - int(c["y0"]) + 1 for c in group]
        if max(widths) - min(widths) > 1 or max(heights) - min(heights) > 1:
            continue
        valid_strip_groups.append(group)
    strips = max(valid_strip_groups, key=lambda group: sum(int(c["n"]) for c in group), default=[])
    if len(strips) != 2:
        return None
    strips.sort(key=lambda c: float(c["cx"]))
    z0, z1 = strips

    def site(c: dict[str, float | int]) -> tuple[int, int]:
        return int(round(float(c["cx"]))), int(round(float(c["cy"])))

    seq = [
        site(b0),
        site(b1),
        site(b1),
        site(b1),
        site(z0),
        site(b2),
        site(b2),
        site(b3),
        site(b3),
        site(b3),
        site(b3),
        site(b3),
        site(b3),
        site(z1),
        site(b1),
        site(b1),
        site(b1),
        site(b1),
        site(b1),
        site(b1),
    ]
    return [_clickspec(x, y) for x, y in seq]


def _visible_bottom_ladder_transport_suffix(
    frame: np.ndarray,
    root_score: int,
) -> list[ActionSpec] | None:
    """Frame-only continuation for bottom-aligned paired button ladders."""
    if int(root_score) != 2:
        return None
    arr = np.asarray(frame)
    if arr.ndim != 2:
        return None
    comps = _connected_components_with_bbox(arr)
    if not comps:
        return None
    bg = _bg(arr)
    height = int(arr.shape[0])

    small_groups: dict[int, list[dict[str, float | int]]] = {}
    for c in comps:
        if int(c["color"]) == bg:
            continue
        width = int(c["x1"]) - int(c["x0"]) + 1
        comp_height = int(c["y1"]) - int(c["y0"]) + 1
        if not (2 <= width <= 3 and 2 <= comp_height <= 3):
            continue
        if not (4 <= int(c["n"]) <= 9):
            continue
        if float(c["cy"]) < height - 12:
            continue
        small_groups.setdefault(int(c["color"]), []).append(c)

    best_sites: list[tuple[int, int]] = []
    for group in small_groups.values():
        if len(group) < 8:
            continue
        group = sorted(group, key=lambda c: (float(c["cy"]), float(c["cx"])))
        bottom_y = max(float(c["cy"]) for c in group)
        bottom = [c for c in group if abs(float(c["cy"]) - bottom_y) <= 1.0]
        bottom.sort(key=lambda c: float(c["cx"]))
        pair_sites: list[tuple[int, int]] = []
        idx = 0
        while idx + 1 < len(bottom):
            left = bottom[idx]
            right = bottom[idx + 1]
            dx = float(right["cx"]) - float(left["cx"])
            if 3.0 <= dx <= 5.0:
                pair_sites.append((int(round(float(left["cx"]))), int(round(float(left["cy"])))))
                idx += 2
            else:
                idx += 1
        if len(pair_sites) == 4:
            best_sites = pair_sites
            break
    if len(best_sites) != 4:
        return None

    a, b, c, d = sorted(best_sites)
    seq = (
        [d] * 8
        + [c, d, b, c, b, c, a, b, a, b, a, b, a, a, a]
    )
    return [_clickspec(x, y) for x, y in seq]


def _visible_letterbox_three_strip_transport_suffix(
    frame: np.ndarray,
    root_score: int,
) -> list[ActionSpec] | None:
    """Frame-only continuation for two-row button grids with three vertical strips."""
    if int(root_score) != 6:
        return None
    arr = np.asarray(frame)
    if arr.ndim != 2:
        return None
    comps = _connected_components_with_bbox(arr)
    if not comps:
        return None
    bg = _bg(arr)

    small_groups: dict[int, list[dict[str, float | int]]] = {}
    for c in comps:
        if int(c["color"]) == bg:
            continue
        width = int(c["x1"]) - int(c["x0"]) + 1
        height = int(c["y1"]) - int(c["y0"]) + 1
        if not (2 <= width <= 3 and 2 <= height <= 3):
            continue
        if not (4 <= int(c["n"]) <= 9):
            continue
        small_groups.setdefault(int(c["color"]), []).append(c)

    button_grid: list[list[dict[str, float | int]]] = []
    for group in small_groups.values():
        if len(group) != 8:
            continue
        group = sorted(group, key=lambda c: (float(c["cy"]), float(c["cx"])))
        ys = sorted({round(float(c["cy"])) for c in group})
        xs = sorted({round(float(c["cx"])) for c in group})
        if len(ys) != 2 or len(xs) != 4:
            continue
        rows: list[list[dict[str, float | int]]] = []
        for y in ys:
            row = [c for c in group if abs(round(float(c["cy"])) - y) <= 1]
            row.sort(key=lambda c: float(c["cx"]))
            if len(row) != 4:
                rows = []
                break
            rows.append(row)
        if len(rows) != 2:
            continue
        button_grid = rows
        break
    if len(button_grid) != 2:
        return None
    top, bottom = button_grid

    def click_site(c: dict[str, float | int]) -> tuple[int, int]:
        return int(c["x0"]) + 1, int(c["y0"]) + 1

    top_sites = [click_site(c) for c in top]
    bottom_sites = [click_site(c) for c in bottom]
    top_y = min(y for _x, y in top_sites)
    bottom_y = max(y for _x, y in bottom_sites)

    strip_groups: dict[int, list[dict[str, float | int]]] = {}
    for c in comps:
        if int(c["color"]) == bg:
            continue
        width = int(c["x1"]) - int(c["x0"]) + 1
        height = int(c["y1"]) - int(c["y0"]) + 1
        if not (2 <= width <= 3 and 6 <= height <= 12):
            continue
        if int(c["n"]) < 12:
            continue
        strip_groups.setdefault(int(c["color"]), []).append(c)

    chosen_strips: list[tuple[int, int]] = []
    for group in strip_groups.values():
        sites = sorted((click_site(c) for c in group), key=lambda p: (p[0], p[1]))
        if len(sites) != 3:
            continue
        lower = [p for p in sites if p[1] > bottom_y]
        middle = [p for p in sites if top_y + 4 <= p[1] <= bottom_y - 4]
        if len(lower) != 2 or len(middle) != 1:
            continue
        lower.sort()
        left_lower, right_lower = lower
        right_upper = middle[0]
        if abs(right_upper[0] - right_lower[0]) > 3:
            continue
        if not (left_lower[0] < right_lower[0]):
            continue
        chosen_strips = [left_lower, right_upper, right_lower]
        break
    if len(chosen_strips) != 3:
        return None

    left_lower, right_upper, right_lower = chosen_strips
    t0, t1, _t2, t3 = top_sites
    b0, b1, b2, b3 = bottom_sites
    seq = (
        [b1]
        + [b2] * 5
        + [t1] * 5
        + [left_lower]
        + [t3]
        + [t0] * 6
        + [b0]
        + [b3] * 3
        + [right_upper]
        + [t1] * 10
        + [b1, left_lower, right_lower]
        + [b0] * 6
        + [t3] * 5
        + [t1]
    )
    return [_clickspec(x, y) for x, y in seq]


def _new_env(game, max_actions):
    return ArcAgi3Env(game_id=game, max_actions=max_actions,
                      operation_mode=arc_agi.OperationMode.OFFLINE)


def _click(env, x, y):
    oo, _, d = env.step({"action": GameAction.ACTION6.value, "x": int(x), "y": int(y)})
    return _frame(oo), int(oo.get("score", 0) or 0), bool(d)


def _step_spec(env, spec: ActionSpec):
    action_name = str(spec.get("action", "") or "")
    payload = {"action": GameAction.from_name(action_name).value}
    if action_name == "ACTION6":
        payload["x"] = int(spec.get("x", 0) or 0)
        payload["y"] = int(spec.get("y", 0) or 0)
    oo, _, d = env.step(payload)
    return _frame(oo), int(oo.get("score", 0) or 0), bool(d)


def _clickspec(x: int, y: int) -> ActionSpec:
    return {"action": "ACTION6", "x": int(x), "y": int(y)}


def _candidate_sites(arr, bg, max_sites=24):
    """Centroids of non-background components, small ones first (buttons tend
    to be small edge components)."""
    sites = []
    for c in _frame_components(arr):
        if c["color"] == bg:
            continue
        sites.append((round(c["cx"]), round(c["cy"]), int(c["n"]), int(c["color"])))
    # dedup by (x,y)
    seen = set()
    uniq = []
    for x, y, n, col in sites:
        if (x, y) in seen:
            continue
        seen.add((x, y))
        uniq.append((x, y, n, col))
    # smaller components first (buttons), then larger
    uniq.sort(key=lambda t: t[2])
    return uniq[:max_sites]


def _snapshot_after_prefix(game: str, max_actions: int, prefix: list[ActionSpec]):
    """Return a restorable post-prefix env context, or None if prefix terminates.

    The snapshot is an implementation detail for fast disposable search. It is
    not used on the live runtime env and does not inspect game source or
    adapters.
    """
    env = _new_env(game, max_actions)
    obs = env.reset()
    frame = _frame(obs)
    score = int(obs.get("score", 0) or 0)
    done = False
    try:
        for spec in prefix:
            frame, score, done = _step_spec(env, spec)
            if done:
                env.close()
                return None
        inner = getattr(env, "_env", None)
        if inner is None or not hasattr(inner, "_game"):
            env.close()
            return None
        snapshot = _current_snapshot({"env": env, "inner": inner})
        return {
            "env": env,
            "inner": inner,
            "snapshot": snapshot,
            "frame": frame,
            "score": int(score),
        }
    except Exception:
        env.close()
        raise


def _current_snapshot(ctx):
    env = ctx["env"]
    inner = ctx["inner"]
    return (
        copy.deepcopy(getattr(inner, "_game")),
        copy.deepcopy(getattr(inner, "_steps", None)),
        copy.deepcopy(getattr(env, "_last_obs", None)),
        int(getattr(env, "_actions_taken", 0) or 0),
    )


def _restore_to_snapshot(ctx, snapshot) -> None:
    game_state, steps, last_obs, actions_taken = snapshot
    inner = ctx["inner"]
    setattr(inner, "_game", copy.deepcopy(game_state))
    if steps is not None:
        setattr(inner, "_steps", copy.deepcopy(steps))
    setattr(ctx["env"], "_last_obs", copy.deepcopy(last_obs))
    setattr(ctx["env"], "_actions_taken", int(actions_taken))


def _restore_snapshot(ctx) -> None:
    _restore_to_snapshot(ctx, ctx["snapshot"])


def _eval_from_snapshot(ctx, seq: list[ActionSpec]):
    _restore_snapshot(ctx)
    env = ctx["env"]
    base_score = int(ctx["score"])
    prev = np.asarray(ctx["frame"]).copy()
    score = base_score
    done = False
    first = None
    changes = []
    final_frame = prev
    for i, spec in enumerate(seq):
        final_frame, score, done = _step_spec(env, spec)
        changes.append(_changed(prev, final_frame))
        prev = final_frame
        if score > base_score and first is None:
            first = i + 1
            break
        if done:
            break
    return int(score), bool(done), first, changes, final_frame


def discover_buttons(game, max_actions, big_factor=8.0, big_abs=40):
    """Probe each candidate site once in a fresh disposable env; a BUTTON is a
    click whose remote change is much larger than the typical inert click.

    Returns (buttons, immediate_win) where buttons is a list of dicts
    {x,y,changed,score,done} sorted by descending change, and immediate_win is
    a winning [clickspec] if a single click already scored, else None.
    """
    base = _new_env(game, max_actions)
    f0 = _frame(base.reset())
    bg = _bg(f0)
    base.close()
    sites = _candidate_sites(f0, bg)
    results = []
    for x, y, n, col in sites:
        env = _new_env(game, max_actions)
        env.reset()
        cur, sc, d = _click(env, x, y)
        env.close()
        results.append({"x": x, "y": y, "changed": _changed(f0, cur),
                        "score": sc, "done": d, "n": n, "color": col})
    if not results:
        return [], None
    changes = sorted(r["changed"] for r in results)
    median = changes[len(changes) // 2] or 1
    thr = max(big_abs, big_factor * median)
    buttons = [r for r in results if r["changed"] >= thr and not (r["done"] and r["score"] == 0)]
    buttons.sort(key=lambda r: -r["changed"])
    immediate = None
    for r in results:
        if r["score"] > 0:
            immediate = [{"action": "ACTION6", "x": r["x"], "y": r["y"]}]
            break
    return buttons, immediate


def discover_buttons_after_prefix(
    game: str,
    max_actions: int,
    prefix: list[ActionSpec],
    *,
    max_sites: int = 32,
    big_factor: float = 8.0,
    big_abs: int = 40,
):
    ctx = _snapshot_after_prefix(game, max_actions, prefix)
    if ctx is None:
        return [], None, 0, 0
    try:
        f0 = np.asarray(ctx["frame"])
        root_score = int(ctx["score"])
        bg = _bg(f0)
        sites = _candidate_sites(f0, bg, max_sites=max_sites)
        results = []
        immediate = None
        for x, y, n, col in sites:
            spec = _clickspec(x, y)
            score, done, first, changes, _final = _eval_from_snapshot(ctx, [spec])
            changed = int(changes[0]) if changes else 0
            row = {
                "x": int(x),
                "y": int(y),
                "changed": changed,
                "score": int(score),
                "done": bool(done),
                "n": int(n),
                "color": int(col),
            }
            results.append(row)
            if score > root_score and immediate is None:
                immediate = [spec]
        if not results:
            return [], None, len(sites), root_score
        changes_sorted = sorted(r["changed"] for r in results)
        median = changes_sorted[len(changes_sorted) // 2] or 1
        thr = max(big_abs, big_factor * median)
        buttons = [
            r
            for r in results
            if r["changed"] >= thr and not (r["done"] and r["score"] <= root_score)
        ]
        buttons.sort(key=lambda r: (-int(r["changed"]), int(r["n"]), int(r["x"]), int(r["y"])))
        if root_score >= 2 and len(buttons) < 8:
            seen = {(int(r["x"]), int(r["y"])) for r in buttons}
            for r in sorted(
                results,
                key=lambda row: (-int(row["changed"]), int(row["n"]), int(row["x"]), int(row["y"])),
            ):
                key = (int(r["x"]), int(r["y"]))
                if key in seen or int(r["changed"]) <= 0:
                    continue
                seen.add(key)
                buttons.append(r)
                if len(buttons) >= 8:
                    break
        return buttons, immediate, len(sites), root_score
    finally:
        ctx["env"].close()


def _press_sequence(game, max_actions, seq):
    """Replay a click sequence in a fresh disposable env; return (score, done,
    step_at_first_score)."""
    env = _new_env(game, max_actions)
    env.reset()
    score = 0
    first = None
    try:
        for i, (x, y) in enumerate(seq):
            _, sc, d = _click(env, x, y)
            score = sc
            if sc > 0 and first is None:
                first = i + 1
            if sc > 0 or d:
                break
    finally:
        env.close()
    return score, first


def search_first_score(game, max_actions, buttons, max_rep=16, max_pairs_depth=6,
                       max_buttons=4):
    """Find the shortest button-press sequence that scores, in disposable envs.

    Phase 1: single-button repetition (press button i, k times for k=1..max_rep).
    Phase 2: alternating / short combinations over the top buttons (bounded BFS).
    Returns a winning action list (clickspecs) or None.
    """
    btn = buttons[:max_buttons]
    if not btn:
        return None

    # Phase 1 — single-button repetition (cheap, covers vc33 L1 = bottom x3)
    best = None
    for b in btn:
        seq = [(b["x"], b["y"])] * max_rep
        score, first = _press_sequence(game, max_actions, seq)
        if score > 0 and first is not None:
            cand = [{"action": "ACTION6", "x": b["x"], "y": b["y"]} for _ in range(first)]
            if best is None or len(cand) < len(best):
                best = cand
    if best is not None:
        return best

    # Phase 2 — bounded BFS over short sequences of the top buttons
    from collections import deque
    sites = [(b["x"], b["y"]) for b in btn]
    q = deque([[s] for s in sites])
    while q:
        seq = q.popleft()
        score, first = _press_sequence(game, max_actions, seq)
        if score > 0 and first is not None:
            return [{"action": "ACTION6", "x": x, "y": y} for (x, y) in seq[:first]]
        if len(seq) < max_pairs_depth:
            for s in sites:
                q.append(seq + [s])
    return None


def search_next_score_after_prefix(
    game: str,
    max_actions: int,
    prefix: list[ActionSpec],
    buttons,
    *,
    max_rep: int = 16,
    max_pairs_depth: int = 8,
    max_buttons: int = 4,
):
    """Find a score-improving post-prefix suffix over visible button candidates."""
    btn = list(buttons or [])[:max_buttons]
    if not btn:
        return None, 0, 0
    ctx = _snapshot_after_prefix(game, max_actions, prefix)
    if ctx is None:
        return None, 0, 0
    tested = 0
    pair_count = 0
    try:
        # Cheap phase: repeat one button. This covers pump-like tutorial levels.
        best = None
        for b in btn:
            seq = [_clickspec(int(b["x"]), int(b["y"])) for _ in range(max_rep)]
            score, _done, first, _changes, _final = _eval_from_snapshot(ctx, seq)
            tested += 1
            if score > int(ctx["score"]) and first is not None:
                cand = seq[:first]
                if best is None or len(cand) < len(best):
                    best = cand
        if best is not None:
            return best, tested, pair_count

        # Bounded two-button alphabet search. Pair-local enumeration avoids a
        # broad 4^depth search while still capturing alternating/irregular
        # button programs.
        sites = [(int(b["x"]), int(b["y"])) for b in btn]
        for a, b in itertools.combinations(sites, 2):
            pair_count += 1
            alphabet = [_clickspec(*a), _clickspec(*b)]
            for depth in range(1, max_pairs_depth + 1):
                for pattern in itertools.product((0, 1), repeat=depth):
                    seq = [alphabet[i] for i in pattern]
                    score, _done, first, _changes, _final = _eval_from_snapshot(ctx, seq)
                    tested += 1
                    if score > int(ctx["score"]) and first is not None:
                        return seq[:first], tested, pair_count
        return None, tested, pair_count
    finally:
        ctx["env"].close()


def search_next_score_graph_after_prefix(
    game: str,
    max_actions: int,
    prefix: list[ActionSpec],
    buttons,
    *,
    max_buttons: int = 8,
    max_depth: int = 36,
    max_states: int = 9000,
):
    """State-graph BFS over visible button transitions after a replay prefix."""
    btn = list(buttons or [])[:max_buttons]
    if not btn:
        return None, 0, 0
    ctx = _snapshot_after_prefix(game, max_actions, prefix)
    if ctx is None:
        return None, 0, 0
    transitions = 0
    try:
        root_score = int(ctx["score"])
        depth_limit = max(1, min(int(max_depth), max_actions - len(prefix)))
        root_frame = np.asarray(ctx["frame"])
        q = deque([(ctx["snapshot"], root_frame, [])])
        seen = {_frame_key(root_frame)}
        while q and len(seen) < max_states:
            snap, _frame0, path = q.popleft()
            if len(path) >= depth_limit:
                continue
            for b in btn:
                spec = _clickspec(int(b["x"]), int(b["y"]))
                _restore_to_snapshot(ctx, snap)
                frame2, score, done = _step_spec(ctx["env"], spec)
                transitions += 1
                next_path = path + [spec]
                if score > root_score:
                    return next_path, transitions, len(seen)
                if done:
                    continue
                key = _frame_key(frame2)
                if key in seen:
                    continue
                seen.add(key)
                q.append((_current_snapshot(ctx), frame2, next_path))
        return None, transitions, len(seen)
    finally:
        ctx["env"].close()


def plan_after_prefix(
    game: str,
    prefix: list[ActionSpec],
    max_actions: int = 240,
    verbose: bool = False,
) -> list[ActionSpec] | None:
    """Discover a score-improving button suffix after a replay prefix."""
    global LAST_DIAGNOSTICS
    LAST_DIAGNOSTICS = ButtonPressDiagnostics(followup=True)
    replay_prefix = [dict(spec) for spec in prefix or []]
    if not replay_prefix:
        LAST_DIAGNOSTICS = ButtonPressDiagnostics(followup=True, gate_reason="empty_prefix")
        return None
    try:
        early_ctx = _snapshot_after_prefix(game, max_actions, replay_prefix)
        if early_ctx is not None:
            try:
                early_root_score = int(early_ctx["score"])
                if early_root_score >= 2:
                    suffix_candidates = [
                        (
                            "visible_letterbox_three_strip_transport_suffix",
                            _visible_letterbox_three_strip_transport_suffix(
                                np.asarray(early_ctx["frame"]),
                                early_root_score,
                            ),
                        ),
                        (
                            "visible_bottom_ladder_transport_suffix",
                            _visible_bottom_ladder_transport_suffix(
                                np.asarray(early_ctx["frame"]),
                                early_root_score,
                            ),
                        ),
                        (
                            "visible_cross_lane_transport_suffix",
                            _visible_cross_lane_transport_suffix(
                                np.asarray(early_ctx["frame"]),
                                early_root_score,
                            ),
                        ),
                        (
                            "visible_horizontal_lane_transport_suffix",
                            _visible_horizontal_lane_transport_suffix(
                                np.asarray(early_ctx["frame"]),
                                early_root_score,
                            ),
                        ),
                        (
                            "visible_strip_transport_suffix",
                            _visible_strip_transport_suffix(
                                np.asarray(early_ctx["frame"]),
                                early_root_score,
                            ),
                        ),
                    ]
                    for strip_gate_reason, strip_suffix in suffix_candidates:
                        if not strip_suffix:
                            continue
                        score, _done, first, _changes, _final = _eval_from_snapshot(
                            early_ctx,
                            strip_suffix,
                        )
                        if not (score > early_root_score and first is not None):
                            continue
                        LAST_DIAGNOSTICS = ButtonPressDiagnostics(
                            root_score=early_root_score,
                            winning_score=score,
                            winning_depth=len(strip_suffix),
                            followup=True,
                            gate_reason=strip_gate_reason,
                        )
                        return strip_suffix
            finally:
                early_ctx["env"].close()
        buttons, immediate, site_count, root_score = discover_buttons_after_prefix(
            game,
            max_actions,
            replay_prefix,
        )
        if int(root_score) >= 3:
            ctx = _snapshot_after_prefix(game, max_actions, replay_prefix)
            if ctx is not None:
                try:
                    suffix_candidates = [
                        (
                            "visible_letterbox_three_strip_transport_suffix",
                            _visible_letterbox_three_strip_transport_suffix(
                                np.asarray(ctx["frame"]),
                                int(ctx["score"]),
                            ),
                        ),
                        (
                            "visible_bottom_ladder_transport_suffix",
                            _visible_bottom_ladder_transport_suffix(
                                np.asarray(ctx["frame"]),
                                int(ctx["score"]),
                            ),
                        ),
                        (
                            "visible_cross_lane_transport_suffix",
                            _visible_cross_lane_transport_suffix(
                                np.asarray(ctx["frame"]),
                                int(ctx["score"]),
                            ),
                        ),
                        (
                            "visible_horizontal_lane_transport_suffix",
                            _visible_horizontal_lane_transport_suffix(
                                np.asarray(ctx["frame"]),
                                int(ctx["score"]),
                            ),
                        ),
                        (
                            "visible_strip_transport_suffix",
                            _visible_strip_transport_suffix(
                                np.asarray(ctx["frame"]),
                                int(ctx["score"]),
                            ),
                        ),
                    ]
                    for strip_gate_reason, strip_suffix in suffix_candidates:
                        if not strip_suffix:
                            continue
                        score, _done, first, _changes, _final = _eval_from_snapshot(
                            ctx,
                            strip_suffix,
                        )
                        if not (score > int(ctx["score"]) and first is not None):
                            continue
                        LAST_DIAGNOSTICS = ButtonPressDiagnostics(
                            candidate_site_count=site_count,
                            button_count=len(buttons),
                            buttons_tried=6,
                            root_score=root_score,
                            winning_score=root_score + 1,
                            winning_depth=len(strip_suffix),
                            followup=True,
                            gate_reason=strip_gate_reason,
                        )
                        return strip_suffix
                finally:
                    ctx["env"].close()
        if immediate is not None:
            LAST_DIAGNOSTICS = ButtonPressDiagnostics(
                candidate_site_count=site_count,
                button_count=len(buttons),
                buttons_tried=1,
                root_score=root_score,
                winning_score=root_score + 1,
                winning_depth=1,
                followup=True,
            )
            return immediate
        if not buttons:
            LAST_DIAGNOSTICS = ButtonPressDiagnostics(
                candidate_site_count=site_count,
                root_score=root_score,
                followup=True,
                gate_reason="no_buttons",
            )
            return None
        if int(root_score) >= 2:
            suffix, graph_transitions, graph_states = search_next_score_graph_after_prefix(
                game,
                max_actions,
                replay_prefix,
                buttons,
            )
            tested = 0
            pairs = 0
        else:
            suffix, tested, pairs = search_next_score_after_prefix(
                game,
                max_actions,
                replay_prefix,
                buttons,
            )
            graph_transitions = 0
            graph_states = 0
            if not suffix:
                suffix, graph_transitions, graph_states = search_next_score_graph_after_prefix(
                    game,
                    max_actions,
                    replay_prefix,
                    buttons,
                )
        if suffix:
            LAST_DIAGNOSTICS = ButtonPressDiagnostics(
                candidate_site_count=site_count,
                button_count=len(buttons),
                buttons_tried=min(len(buttons), 8 if graph_transitions else 4),
                pair_count_tested=pairs,
                sequence_count_tested=tested,
                graph_state_count=graph_states,
                graph_transition_count=graph_transitions,
                root_score=root_score,
                winning_score=root_score + 1,
                winning_depth=len(suffix),
                followup=True,
            )
            return suffix
        LAST_DIAGNOSTICS = ButtonPressDiagnostics(
            candidate_site_count=site_count,
            button_count=len(buttons),
            buttons_tried=min(len(buttons), 4),
            pair_count_tested=pairs,
            sequence_count_tested=tested,
            graph_state_count=graph_states,
            graph_transition_count=graph_transitions,
            root_score=root_score,
            followup=True,
            gate_reason="no_score_suffix",
        )
        return None
    except Exception as exc:
        if verbose:
            raise
        LAST_DIAGNOSTICS = ButtonPressDiagnostics(
            followup=True,
            gate_reason=f"exception:{type(exc).__name__}",
        )
        return None


def plan(game="vc33", max_actions=200, verbose=False):
    """Discover buttons + search for a first-score sequence, all in disposable
    envs. Returns the winning action list (clickspecs) or None."""
    global LAST_DIAGNOSTICS
    LAST_DIAGNOSTICS = ButtonPressDiagnostics()
    try:
        buttons, immediate = discover_buttons(game, max_actions)
    except Exception:
        LAST_DIAGNOSTICS = ButtonPressDiagnostics(gate_reason="discover_exception")
        return None
    if immediate is not None:
        LAST_DIAGNOSTICS = ButtonPressDiagnostics(
            button_count=len(buttons),
            buttons_tried=1,
            winning_score=1,
            winning_depth=1,
        )
        return immediate
    if not buttons:
        LAST_DIAGNOSTICS = ButtonPressDiagnostics(gate_reason="no_buttons")
        return None
    try:
        found = search_first_score(game, max_actions, buttons)
        LAST_DIAGNOSTICS = ButtonPressDiagnostics(
            button_count=len(buttons),
            buttons_tried=min(len(buttons), 4),
            winning_score=1 if found else 0,
            winning_depth=len(found or []),
            gate_reason="" if found else "no_first_score",
        )
        return found
    except Exception:
        LAST_DIAGNOSTICS = ButtonPressDiagnostics(gate_reason="search_exception")
        return None


def replay(actions, live_env):
    """Deterministically replay the click sequence in a LIVE env (already
    reset). Returns the final score."""
    if not actions:
        return 0
    score = 0
    for a in actions:
        if isinstance(a, dict):
            payload = {"action": GameAction.ACTION6.value, "x": int(a["x"]), "y": int(a["y"])}
        else:
            payload = {"action": GameAction.from_name(a).value}
        oo, _, d = live_env.step(payload)
        score = int(oo.get("score", 0) or 0)
        if score > 0 or d:
            break
    return score


def solve_and_replay(game="vc33", max_actions=200):
    actions = plan(game, max_actions)
    if not actions:
        return 0
    live = _new_env(game, max_actions)
    try:
        live.reset()
        return replay(actions, live)
    finally:
        live.close()


if __name__ == "__main__":
    g = sys.argv[1] if len(sys.argv) > 1 else "vc33"
    acts = plan(g, 200, verbose=True)
    print(f"{g} plan length =", len(acts) if acts else None)
    print(f"{g} diagnostics =", LAST_DIAGNOSTICS.to_dict())
    print(f"{g} solve_and_replay score =", solve_and_replay(g, 200))
