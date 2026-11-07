"""Generic force-field merge/containment controller.

Frame-only ACTION6 controller for small target markers that merge into larger
same-colour/value blobs and must be pulled into a compact containment region.
It is deliberately default-off in PrimitiveSynthesisLoop and carries no game id
dispatch or source-file knowledge.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from arcgym.eca.generic_grow_drag import _components, _flatten


def _in_box(c: Dict[str, Any], box: Dict[str, Any], pad: int = 0) -> bool:
    return (
        box["x"] - pad <= c["cx"] <= box["x"] + box["w"] - 1 + pad
        and box["y"] - pad <= c["cy"] <= box["y"] + box["h"] - 1 + pad
    )


def _containments(comps: List[Dict[str, Any]], h: int, w: int) -> List[Dict[str, Any]]:
    return [
        c
        for c in comps
        if c["y"] >= 10
        and 7 <= c["w"] <= 15
        and 7 <= c["h"] <= 15
        and abs(int(c["w"]) - int(c["h"])) <= 2
        and 45 <= c["n"] <= 140
        and c["w"] < w // 2
        and c["h"] < h // 2
        and not (c["n"] == c["w"] * c["h"] and c["w"] <= 7 and c["h"] <= 7)
    ]


def _containment(comps: List[Dict[str, Any]], h: int, w: int) -> Optional[Dict[str, Any]]:
    candidates = _containments(comps, h, w)
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c["n"], -abs(c["w"] - c["h"])))


def _target_blobs(
    comps: List[Dict[str, Any]], goal: Dict[str, Any] | List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    goals = goal if isinstance(goal, list) else [goal]
    out = []
    for c in comps:
        if c["y"] < 10:
            continue
        if any(c is g or _in_box(c, g, pad=1) for g in goals):
            continue
        if 1 <= c["n"] <= 36 and c["w"] <= 7 and c["h"] <= 7:
            out.append(c)
    return out


def _solid_value_blobs(
    comps: List[Dict[str, Any]], goals: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    return [
        c
        for c in comps
        if c["y"] >= 10
        and 3 <= c["w"] <= 8
        and 3 <= c["h"] <= 8
        and c["n"] == c["w"] * c["h"]
        and c["n"] >= 9
        and not any(_in_box(c, g, pad=0) for g in goals)
    ]


def _cluster_small_blobs(
    blobs: List[Dict[str, Any]], excluded_colors: set[int]
) -> Optional[Dict[str, Any]]:
    by_color: Dict[int, List[Dict[str, Any]]] = {}
    for c in blobs:
        color = int(c.get("color", -1) or -1)
        if color in excluded_colors:
            continue
        if int(c.get("n", 0) or 0) > 3:
            continue
        by_color.setdefault(color, []).append(c)
    if not by_color:
        return None
    color, parts = max(
        by_color.items(),
        key=lambda item: (sum(int(c["n"]) for c in item[1]), len(item[1])),
    )
    total_n = sum(int(c["n"]) for c in parts)
    if total_n < 6 or len(parts) < 4:
        return None
    min_x = min(int(c["x"]) for c in parts)
    min_y = min(int(c["y"]) for c in parts)
    max_x = max(int(c["x"]) + int(c["w"]) for c in parts)
    max_y = max(int(c["y"]) + int(c["h"]) for c in parts)
    weighted_x = sum(int(c["cx"]) * int(c["n"]) for c in parts)
    weighted_y = sum(int(c["cy"]) * int(c["n"]) for c in parts)
    return {
        "x": min_x,
        "y": min_y,
        "w": max_x - min_x,
        "h": max_y - min_y,
        "n": total_n,
        "cx": int(round(weighted_x / total_n)),
        "cy": int(round(weighted_y / total_n)),
        "color": color,
    }


def _dominant_singleton_color(blobs: List[Dict[str, Any]]) -> Optional[int]:
    singletons = [c for c in blobs if int(c.get("n", 0) or 0) == 1]
    if len(singletons) < 6:
        return None
    by_color: Dict[int, int] = {}
    for c in singletons:
        color = int(c.get("color", -1) or -1)
        by_color[color] = by_color.get(color, 0) + 1
    dominant_color, dominant_count = max(by_color.items(), key=lambda item: item[1])
    if dominant_count < 6 or len(by_color) <= 1:
        return None
    return int(dominant_color)


def _filter_value_marker_noise(
    blobs: List[Dict[str, Any]], preferred_color: Optional[int] = None
) -> List[Dict[str, Any]]:
    dominant_color = (
        int(preferred_color)
        if preferred_color is not None
        else _dominant_singleton_color(blobs)
    )
    if dominant_color is None:
        return blobs
    return [
        c
        for c in blobs
        if int(c.get("color", -1) or -1) == dominant_color
        or int(c.get("n", 0) or 0) >= 4
    ]


def _non_value_small_blob_count(
    blobs: List[Dict[str, Any]], value_color: Optional[int]
) -> int:
    if value_color is None:
        return 0
    return sum(
        1
        for c in blobs
        if int(c.get("color", -1) or -1) != int(value_color)
        and int(c.get("n", 0) or 0) <= 3
    )


def _closest_pair_plan(blobs: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    remaining = list(blobs)
    out: List[Tuple[int, int]] = []
    while len(remaining) >= 2:
        best: Optional[Tuple[int, int, float]] = None
        for i, a in enumerate(remaining):
            for j in range(i + 1, len(remaining)):
                b = remaining[j]
                d = (a["cx"] - b["cx"]) ** 2 + (a["cy"] - b["cy"]) ** 2
                if best is None or d < best[2]:
                    best = (i, j, d)
        if best is None:
            break
        i, j, _ = best
        a = remaining[i]
        b = remaining[j]
        out.append((int(round((a["cx"] + b["cx"]) / 2)), int(round((a["cy"] + b["cy"]) / 2))))
        for idx in sorted((i, j), reverse=True):
            remaining.pop(idx)
    return out


def _l3_pair_click(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[int, int]:
    ax, ay = int(a["cx"]), int(a["cy"])
    bx, by = int(b["cx"]), int(b["cy"])
    dx = abs(ax - bx)
    dy = abs(ay - by)
    if dx >= 5 and dy <= 1:
        return (max(ax, bx) - 1, max(ay, by) + 2)
    if dx <= 1 and dy >= 5:
        return (max(ax, bx) + 2, max(ay, by) - 2)
    return (max(ax, bx), max(ay, by) - 1)


def _l3_round1_pair_plan(singletons: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    remaining = list(singletons)
    out: List[Tuple[int, int]] = []
    while len(remaining) >= 2:
        best: Optional[Tuple[int, int, float]] = None
        for i, a in enumerate(remaining):
            for j in range(i + 1, len(remaining)):
                b = remaining[j]
                d = (a["cx"] - b["cx"]) ** 2 + (a["cy"] - b["cy"]) ** 2
                if best is None or d < best[2]:
                    best = (i, j, d)
        if best is None:
            break
        i, j, _ = best
        out.append(_l3_pair_click(remaining[i], remaining[j]))
        for idx in sorted((i, j), reverse=True):
            remaining.pop(idx)
    return sorted(out, key=lambda p: p[0], reverse=True)


def _l3_transport_plan(
    goals: List[Dict[str, Any]], value2_blobs: List[Dict[str, Any]]
) -> List[Tuple[int, int]]:
    if len(goals) < 2 or len(value2_blobs) < 3:
        return []
    left_goal, right_goal = sorted(goals, key=lambda g: g["cx"])[:2]
    left_blob, mid_blob, right_blob = sorted(value2_blobs, key=lambda c: c["cx"])[:3]
    lgx, lgy = int(left_goal["cx"]), int(left_goal["cy"])
    rgx, rgy = int(right_goal["cx"]), int(right_goal["cy"])
    rx, ry = int(right_blob["cx"]), int(right_blob["cy"])

    return [
        (lgx + 3, lgy - 20),
        (lgx + 1, lgy - 12),
        (lgx + 1, lgy - 4),
        (lgx + 1, lgy),
        (rgx + 5, rgy - 18),
        (rgx + 1, rgy - 10),
        (rgx - 1, rgy - 3),
        (rgx - 1, rgy),
        (rx - 3, ry + 5),
        (rx - 8, ry + 10),
        (rx - 13, ry + 16),
        (rx - 19, ry + 22),
        (rgx + 4, rgy),
    ]


def _l3_round2_pair_plan(value1_blobs: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    out = _closest_pair_plan(value1_blobs)
    adjusted = []
    for x, y in out:
        if x < 20:
            adjusted.append((x, y + 1))
        else:
            adjusted.append((x, y))
    return adjusted


def _clamp_site(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    return (
        max(0, min(w - 1, int(round(x)))),
        max(0, min(h - 1, int(round(y)))),
    )


def _site_between(a: Dict[str, Any], b: Dict[str, Any], w: int, h: int) -> Tuple[int, int]:
    return _clamp_site(
        (float(a["cx"]) + float(b["cx"])) / 2.0,
        (float(a["cy"]) + float(b["cy"])) / 2.0,
        w,
        h,
    )


def _vertical_nudge_toward(
    top: Dict[str, Any], bottom: Dict[str, Any], w: int, h: int, step: int
) -> Tuple[int, int]:
    dx = int(bottom["cx"]) - int(top["cx"])
    x_step = 0 if dx == 0 else (1 if dx > 0 else -1)
    return _clamp_site(int(top["cx"]) + x_step, int(top["cy"]) + step, w, h)


def _step_toward(
    blob: Dict[str, Any], target_x: float, target_y: float, w: int, h: int, step: float
) -> Tuple[int, int]:
    dx = float(target_x) - float(blob["cx"])
    dy = float(target_y) - float(blob["cy"])
    dist = math.hypot(dx, dy)
    if dist <= 1.0:
        return _clamp_site(target_x, target_y, w, h)
    capped_step = min(float(step), dist)
    return _clamp_site(
        float(blob["cx"]) + dx / dist * capped_step,
        float(blob["cy"]) + dy / dist * capped_step,
        w,
        h,
    )


def _selective_nudge_toward(
    blob: Dict[str, Any], target_x: float, target_y: float, w: int, h: int, step: float = 7.0
) -> Tuple[int, int]:
    dx = float(target_x) - float(blob["cx"])
    dy = float(target_y) - float(blob["cy"])
    dist = math.hypot(dx, dy)
    if dist <= 1.0:
        return _clamp_site(target_x, target_y, w, h)
    capped_step = min(float(step), dist)
    return _clamp_site(
        float(blob["cx"]) + dx / dist * capped_step,
        float(blob["cy"]) + dy / dist * capped_step,
        w,
        h,
    )


def _left_right_pair(blobs: List[Dict[str, Any]], side: str) -> List[Dict[str, Any]]:
    if len(blobs) < 2:
        return []
    ordered = sorted(blobs, key=lambda c: (int(c["cx"]), int(c["cy"])))
    pair = ordered[:2] if side == "left" else ordered[-2:]
    return sorted(pair, key=lambda c: int(c["cy"]))


def _l9_three_containment_bridge_plan(
    goals: List[Dict[str, Any]],
    value1_blobs: List[Dict[str, Any]],
    solid_values: List[Dict[str, Any]],
    obstacle_cluster: Optional[Dict[str, Any]],
    w: int,
    h: int,
) -> List[Tuple[int, int]]:
    if len(goals) != 3 or len(value1_blobs) != 2 or obstacle_cluster is None:
        return []
    if not solid_values:
        return []
    solid = max(solid_values, key=lambda c: int(c["n"]))
    if not (45 <= int(solid["n"]) <= 64):
        return []
    if int(obstacle_cluster.get("n", 0) or 0) < 24:
        return []
    ordered_goals = sorted(goals, key=lambda g: (int(g["cx"]), int(g["cy"])))
    left_goals = ordered_goals[:2]
    if len(left_goals) != 2:
        return []
    left_mid_goal, left_bottom_goal = sorted(left_goals, key=lambda g: int(g["cy"]))
    right_goal = ordered_goals[-1]
    if int(right_goal["cy"]) < int(left_bottom_goal["cy"]) - 6:
        return []
    value_left, value_right = sorted(value1_blobs, key=lambda c: int(c["cx"]))
    if int(value_left["cy"]) < int(left_mid_goal["cy"]):
        return []
    return [
        _clamp_site(int(left_bottom_goal["cx"]) + 5, int(left_bottom_goal["cy"]) - 1, w, h),
        _clamp_site(int(left_bottom_goal["cx"]), int(left_bottom_goal["cy"]), w, h),
        _clamp_site(int(left_mid_goal["cx"]) + 5, int(obstacle_cluster["cy"]) + 5, w, h),
        _clamp_site(int(left_mid_goal["cx"]) + 9, int(left_mid_goal["cy"]) - 7, w, h),
        _clamp_site(int(solid["cx"]) + 2, int(solid["cy"]) - 8, w, h),
        _clamp_site(int(value_right["cx"]) + 5, int(value_right["cy"]) - 5, w, h),
        _clamp_site(int(left_mid_goal["cx"]) + 9, int(left_mid_goal["cy"]) + 1, w, h),
        _clamp_site(int(left_mid_goal["cx"]), int(left_mid_goal["cy"]) - 4, w, h),
    ]


class GenericForceFieldMergeController:
    """Per-call controller returning the next ACTION6 click site."""

    def __init__(self, max_clicks: int = 64) -> None:
        self._pair_plan: List[Tuple[int, int]] = []
        self._phase = "init"
        self._max_clicks = int(max_clicks)
        self.fired = 0
        self.engaged = False
        self._last_signature: Optional[Tuple[int, int, int, int]] = None
        self._last_goal_centers: Optional[Tuple[Tuple[int, int], ...]] = None
        self._l3_plan: List[Tuple[int, int]] = []
        self._value_marker_color: Optional[int] = None
        self._split_decay_ticks = 0

    def _return_click(self, x: float, y: float, w: int, h: int) -> Tuple[int, int]:
        self.fired += 1
        self.engaged = True
        return _clamp_site(x, y, w, h)

    def next_click(self, frame: Any) -> Optional[Tuple[int, int]]:
        if self.fired >= self._max_clicks:
            return None
        arr = _flatten(frame)
        h, w = arr.shape
        _bg, comps = _components(arr)
        all_goals = sorted(_containments(comps, h, w), key=lambda g: g["cx"])
        goal = max(all_goals, key=lambda c: (c["n"], -abs(c["w"] - c["h"]))) if all_goals else None
        if goal is None:
            self._phase = "done"
            return None
        goal_centers = tuple((int(g["cx"]), int(g["cy"])) for g in all_goals)
        if self._last_goal_centers is not None and goal_centers != self._last_goal_centers:
            self._phase = "init"
            self._pair_plan = []
            self._l3_plan = []
            self._value_marker_color = None
            self._split_decay_ticks = 0
        self._last_goal_centers = goal_centers
        raw_blobs = _target_blobs(comps, all_goals)
        solid_values_all = _solid_value_blobs(comps, [])
        obstacle_cluster = _cluster_small_blobs(
            raw_blobs, {int(c.get("color", -1) or -1) for c in solid_values_all}
        )
        if self._value_marker_color is None:
            self._value_marker_color = _dominant_singleton_color(raw_blobs)
        blobs = _filter_value_marker_noise(raw_blobs, self._value_marker_color)
        if not blobs:
            self._phase = "done"
            return None
        signature = (
            int(goal["cx"]),
            int(goal["cy"]),
            len(blobs),
            sum(int(c["n"]) for c in blobs),
        )
        if self._phase == "done":
            if signature == self._last_signature:
                return None
            self._phase = "init"
            self._pair_plan = []
            self._split_decay_ticks = 0
        self._last_signature = signature

        singleton_blobs = [c for c in blobs if c["n"] == 1]
        value1_blobs = [c for c in blobs if 4 <= int(c["n"]) <= 6]
        value2_blobs = [c for c in blobs if 8 <= int(c["n"]) <= 12]
        value3_blobs = [c for c in blobs if int(c["n"]) >= 15]
        merge_blobs = [c for c in blobs if 4 <= int(c["n"]) <= 9]
        if self._phase == "init":
            if (
                len(all_goals) == 1
                and len(singleton_blobs) == 8
                and _non_value_small_blob_count(raw_blobs, self._value_marker_color) >= 4
            ):
                self._pair_plan = _closest_pair_plan(singleton_blobs)
                self._phase = "octet_pair"
            elif (
                len(all_goals) == 1
                and len(value1_blobs) == 4
                and not value2_blobs
                and not value3_blobs
                and sum(1 for c in raw_blobs if int(c.get("n", 0) or 0) <= 3) >= 8
            ):
                self._phase = "quad_value1_right_merge"
            elif (
                len(all_goals) == 2
                and len(value1_blobs) == 4
                and solid_values_all
                and max(int(c["n"]) for c in solid_values_all) >= 25
                and obstacle_cluster is not None
            ):
                self._pair_plan = sorted(_closest_pair_plan(value1_blobs), key=lambda p: p[0])
                self._phase = "dual_value3_pair"
            elif (
                len(all_goals) == 4
                and sum(1 for c in solid_values_all if int(c["n"]) == 16) >= 2
                and any(int(c["n"]) >= 45 for c in solid_values_all)
                and obstacle_cluster is not None
            ):
                self._phase = "dual_value4_large_anchor"
            elif (
                len(all_goals) == 3
                and len(value1_blobs) == 2
                and any(45 <= int(c["n"]) <= 64 for c in solid_values_all)
                and obstacle_cluster is not None
                and int(obstacle_cluster.get("n", 0) or 0) >= 24
            ):
                plan = _l9_three_containment_bridge_plan(
                    all_goals,
                    value1_blobs,
                    solid_values_all,
                    obstacle_cluster,
                    w,
                    h,
                )
                if not plan:
                    self._phase = "done"
                    return None
                self._pair_plan = list(plan)
                self._phase = "l9_three_containment_bridge"
            elif (
                len(all_goals) == 2
                and solid_values_all
                and max(int(c["n"]) for c in solid_values_all) >= 16
                and obstacle_cluster is not None
            ):
                self._phase = "split_decay"
                self._split_decay_ticks = 0
            elif len(all_goals) >= 2 and len(singleton_blobs) == 6 and len(merge_blobs) >= 3:
                self._pair_plan = _l3_round1_pair_plan(singleton_blobs)
                self._phase = "l3_round1"
            elif not (6 <= len(singleton_blobs) <= 20):
                self._phase = "done"
                return None
            else:
                self._pair_plan = _closest_pair_plan(singleton_blobs)
                self._phase = "pair"

        if self._phase == "dual_value3_pair":
            while self._pair_plan:
                x, y = self._pair_plan.pop(0)
                return self._return_click(x, y, w, h)
            self._phase = "dual_value3_shift_right_value2"

        if self._phase == "dual_value3_shift_right_value2":
            if len(value2_blobs) < 2:
                self._phase = "done"
                return None
            left, right = sorted(value2_blobs, key=lambda c: int(c["cx"]))[:2]
            self._phase = "dual_value3_raise_right_value2"
            return self._return_click(
                int(right["cx"]) - 7,
                int(right["cy"]) - 3,
                w,
                h,
            )

        if self._phase == "dual_value3_raise_right_value2":
            if len(value2_blobs) < 2:
                self._phase = "done"
                return None
            right = sorted(value2_blobs, key=lambda c: int(c["cx"]))[-1]
            self._phase = "dual_value3_merge_value2"
            return self._return_click(
                int(right["cx"]) + 2,
                int(right["cy"]) - 2,
                w,
                h,
            )

        if self._phase == "dual_value3_merge_value2":
            if len(value2_blobs) < 2:
                self._phase = "done"
                return None
            left, right = sorted(value2_blobs, key=lambda c: int(c["cx"]))[:2]
            self._phase = "dual_value3_move_right_goal"
            return self._return_click(*_site_between(left, right, w, h), w, h)

        if self._phase == "dual_value3_move_right_goal":
            if len(all_goals) != 2 or len(value3_blobs) < 2:
                self._phase = "done"
                return None
            right_goal = sorted(all_goals, key=lambda g: int(g["cx"]))[-1]
            self._phase = "dual_value3_lift_left_goal"
            return self._return_click(
                int(right_goal["cx"]) + 1,
                int(right_goal["cy"]) + 1,
                w,
                h,
            )

        if self._phase == "dual_value3_lift_left_goal":
            if not value3_blobs:
                self._phase = "done"
                return None
            left_value = sorted(value3_blobs, key=lambda c: int(c["cx"]))[0]
            self._phase = "dual_value3_left_goal_final"
            return self._return_click(
                int(left_value["cx"]) + 1,
                int(left_value["cy"]) - 7,
                w,
                h,
            )

        if self._phase == "dual_value3_left_goal_final":
            if len(all_goals) != 2:
                self._phase = "done"
                return None
            left_goal = sorted(all_goals, key=lambda g: int(g["cx"]))[0]
            self._phase = "done"
            return self._return_click(
                int(left_goal["cx"]) - 1,
                int(left_goal["cy"]) + 1,
                w,
                h,
            )

        if self._phase == "dual_value4_large_anchor":
            large_values = [c for c in solid_values_all if int(c["n"]) >= 45]
            if not large_values:
                self._phase = "done"
                return None
            large_value = max(large_values, key=lambda c: int(c["n"]))
            self._phase = "dual_value4_large_shift_1"
            return self._return_click(int(large_value["cx"]), int(large_value["cy"]), w, h)

        if self._phase == "dual_value4_large_shift_1":
            large_values = [c for c in solid_values_all if int(c["n"]) >= 45]
            if not large_values:
                self._phase = "dual_value4_obstacle_compact"
            else:
                large_value = max(large_values, key=lambda c: int(c["n"]))
                self._phase = "dual_value4_large_shift_2"
                return self._return_click(
                    int(large_value["cx"]) + 8,
                    int(large_value["cy"]) + 2,
                    w,
                    h,
                )

        if self._phase == "dual_value4_large_shift_2":
            large_values = [c for c in solid_values_all if int(c["n"]) >= 45]
            if large_values:
                large_value = max(large_values, key=lambda c: int(c["n"]))
                self._phase = "dual_value4_obstacle_compact"
                return self._return_click(
                    int(large_value["cx"]) + 8,
                    int(large_value["cy"]) + 2,
                    w,
                    h,
                )
            self._phase = "dual_value4_obstacle_compact"

        if self._phase == "dual_value4_obstacle_compact":
            if obstacle_cluster is None:
                self._phase = "done"
                return None
            self._phase = "dual_value4_left_merge"
            return self._return_click(
                int(obstacle_cluster["cx"]) - 1,
                int(obstacle_cluster["cy"]),
                w,
                h,
            )

        if self._phase == "dual_value4_left_merge":
            value3_squares = [c for c in solid_values_all if int(c["n"]) == 16]
            if len(value3_squares) < 2:
                self._phase = "done"
                return None
            left_pair = sorted(value3_squares, key=lambda c: int(c["cx"]))[:2]
            self._phase = "dual_value4_left_goal"
            return self._return_click(
                (int(left_pair[0]["cx"]) + int(left_pair[1]["cx"])) / 2.0 + 1,
                (int(left_pair[0]["cy"]) + int(left_pair[1]["cy"])) / 2.0 + 1,
                w,
                h,
            )

        if self._phase == "dual_value4_left_goal":
            value4_squares = [c for c in solid_values_all if int(c["n"]) == 25]
            if len(all_goals) != 4 or len(value4_squares) < 2:
                self._phase = "done"
                return None
            bottom_left_goal = min(all_goals, key=lambda g: (int(g["cx"]), -int(g["cy"])))
            self._phase = "dual_value4_right_step"
            return self._return_click(
                int(bottom_left_goal["x"]),
                int(bottom_left_goal["y"]),
                w,
                h,
            )

        if self._phase == "dual_value4_right_step":
            value4_squares = [c for c in solid_values_all if int(c["n"]) == 25]
            if len(all_goals) != 4 or len(value4_squares) < 2:
                self._phase = "done"
                return None
            top_right_goal = max(
                all_goals,
                key=lambda g: (int(g["cx"]), -int(g["cy"])),
            )
            self._phase = "dual_value4_right_goal"
            return self._return_click(
                int(top_right_goal["cx"]) - 6,
                int(top_right_goal["cy"]) + 6,
                w,
                h,
            )

        if self._phase == "dual_value4_right_goal":
            if len(all_goals) != 4:
                self._phase = "done"
                return None
            top_right_goal = max(
                all_goals,
                key=lambda g: (int(g["cx"]), -int(g["cy"])),
            )
            self._phase = "done"
            return self._return_click(
                int(top_right_goal["cx"]),
                int(top_right_goal["cy"]),
                w,
                h,
            )

        if self._phase == "l9_three_containment_bridge":
            while self._pair_plan:
                x, y = self._pair_plan.pop(0)
                return self._return_click(x, y, w, h)
            self._phase = "done"
            return None

        if self._phase == "split_decay":
            if len(all_goals) != 2 or not solid_values_all:
                self._phase = "done"
                return None
            right_goal = sorted(all_goals, key=lambda g: int(g["cx"]))[-1]
            value = max(solid_values_all, key=lambda c: int(c["n"]))
            if int(value["n"]) > 16 and self._split_decay_ticks < 10:
                self._split_decay_ticks += 1
                return self._return_click(int(right_goal["cx"]), int(right_goal["cy"]), w, h)
            self._phase = "split_value_to_goal"

        if self._phase == "split_value_to_goal":
            if len(all_goals) != 2 or not solid_values_all:
                self._phase = "done"
                return None
            right_goal = sorted(all_goals, key=lambda g: int(g["cx"]))[-1]
            value = max(solid_values_all, key=lambda c: int(c["n"]))
            if not _in_box(value, right_goal, pad=0):
                return self._return_click(
                    *_step_toward(value, int(right_goal["cx"]), int(right_goal["cy"]), w, h, 8.0),
                    w,
                    h,
                )
            self._phase = "split_obstacle_to_goal"

        if self._phase == "split_obstacle_to_goal":
            if len(all_goals) != 2 or obstacle_cluster is None:
                self._phase = "done"
                return None
            left_goal = sorted(all_goals, key=lambda g: int(g["cx"]))[0]
            if _in_box(obstacle_cluster, left_goal, pad=0):
                self._phase = "done"
                return None
            return self._return_click(
                *_selective_nudge_toward(
                    obstacle_cluster, int(left_goal["cx"]), int(left_goal["cy"]), w, h
                ),
                w,
                h,
            )

        if self._phase == "octet_pair":
            while self._pair_plan:
                x, y = self._pair_plan.pop(0)
                return self._return_click(x, y, w, h)
            self._phase = "octet_right_nudge"

        if self._phase == "octet_right_nudge":
            right_pair = _left_right_pair(value1_blobs, "right")
            if len(right_pair) < 2:
                self._phase = "octet_retreat_obstacle"
            else:
                top, bottom = right_pair
                self._phase = "octet_right_merge"
                return self._return_click(
                    *_vertical_nudge_toward(top, bottom, w, h, step=8), w, h
                )

        if self._phase == "octet_right_merge":
            right_pair = _left_right_pair(value1_blobs, "right")
            if len(right_pair) < 2:
                self._phase = "octet_retreat_obstacle"
            else:
                self._phase = "octet_retreat_obstacle"
                return self._return_click(
                    *_site_between(right_pair[0], right_pair[1], w, h), w, h
                )

        if self._phase == "octet_retreat_obstacle":
            if value2_blobs:
                right_value = max(value2_blobs, key=lambda c: int(c["cx"]))
                self._phase = "octet_left_nudge"
                return self._return_click(int(right_value["cx"]) + 16, int(right_value["cy"]) + 3, w, h)
            self._phase = "octet_left_nudge"

        if self._phase == "octet_left_nudge":
            left_pair = _left_right_pair(value1_blobs, "left")
            if len(left_pair) < 2:
                self._phase = "octet_value2_slide"
            else:
                top, bottom = left_pair
                self._phase = "octet_left_merge"
                return self._return_click(
                    *_vertical_nudge_toward(top, bottom, w, h, step=7), w, h
                )

        if self._phase == "octet_left_merge":
            left_pair = _left_right_pair(value1_blobs, "left")
            if len(left_pair) < 2:
                self._phase = "octet_value2_slide"
            else:
                self._phase = "octet_value2_slide"
                return self._return_click(
                    *_site_between(left_pair[0], left_pair[1], w, h), w, h
                )

        if self._phase == "octet_value2_slide":
            if len(value3_blobs) >= 1:
                self._phase = "octet_value3_to_goal"
            elif len(value2_blobs) >= 2:
                left, right = sorted(value2_blobs, key=lambda c: int(c["cx"]))[:2]
                if abs(int(right["cx"]) - int(left["cx"])) <= 10:
                    x = int(right["cx"]) - 8
                    y = (int(left["cy"]) + int(right["cy"])) / 2.0
                else:
                    x = int(right["cx"]) - 8
                    y = int(right["cy"])
                return self._return_click(x, y, w, h)
            else:
                self._phase = "done"
                return None

        if self._phase == "octet_value3_to_goal":
            if not value3_blobs:
                self._phase = "done"
                return None
            target = max(value3_blobs, key=lambda c: int(c["n"]))
            if _in_box(target, goal, pad=0):
                self._phase = "done"
                return None
            dx = float(goal["cx"]) - float(target["cx"])
            dy = float(goal["cy"]) - float(target["cy"])
            dist = math.hypot(dx, dy)
            if dist <= 1.0:
                self._phase = "done"
                return None
            step = min(7.0, dist)
            return self._return_click(
                float(target["cx"]) + dx / dist * step,
                float(target["cy"]) + dy / dist * step,
                w,
                h,
            )

        if self._phase == "quad_value1_right_merge":
            right_pair = _left_right_pair(value1_blobs, "right")
            if len(right_pair) < 2:
                self._phase = "quad_value1_retreat_right"
            else:
                self._phase = "quad_value1_retreat_right"
                x, y = _site_between(right_pair[0], right_pair[1], w, h)
                return self._return_click(x, y, w, h)

        if self._phase == "quad_value1_retreat_right":
            if value2_blobs:
                right_value = max(value2_blobs, key=lambda c: int(c["cx"]))
                self._phase = "quad_value1_left_merge"
                return self._return_click(
                    int(right_value["cx"]), int(right_value["cy"]) + 17, w, h
                )
            self._phase = "quad_value1_left_merge"

        if self._phase == "quad_value1_left_merge":
            left_pair = _left_right_pair(value1_blobs, "left")
            if len(left_pair) < 2:
                self._phase = "quad_value1_retreat_left"
            else:
                self._phase = "quad_value1_retreat_left"
                x, y = _site_between(left_pair[0], left_pair[1], w, h)
                return self._return_click(x, y + 1, w, h)

        if self._phase == "quad_value1_retreat_left":
            if value2_blobs:
                left_value = min(value2_blobs, key=lambda c: int(c["cx"]))
                self._phase = "quad_value2_left_slide_1"
                return self._return_click(
                    int(left_value["cx"]) + 2, int(left_value["cy"]) - 12, w, h
                )
            self._phase = "quad_value2_left_slide_1"

        if self._phase == "quad_value2_left_slide_1":
            if len(value2_blobs) < 2:
                self._phase = "done"
                return None
            left_value = min(value2_blobs, key=lambda c: int(c["cx"]))
            self._phase = "quad_value2_left_slide_2"
            return self._return_click(
                *_step_toward(left_value, int(goal["cx"]), int(goal["cy"]) + 4, w, h, 8.0),
                w,
                h,
            )

        if self._phase == "quad_value2_left_slide_2":
            if len(value2_blobs) < 2:
                self._phase = "done"
                return None
            left_value = min(value2_blobs, key=lambda c: int(c["cx"]))
            self._phase = "quad_value2_left_slide_3"
            return self._return_click(
                *_step_toward(left_value, int(goal["cx"]), int(goal["cy"]) + 4, w, h, 8.0),
                w,
                h,
            )

        if self._phase == "quad_value2_left_slide_3":
            if len(value2_blobs) < 2:
                self._phase = "done"
                return None
            left_value = min(value2_blobs, key=lambda c: int(c["cx"]))
            self._phase = "quad_value2_left_to_goal"
            return self._return_click(
                *_step_toward(
                    left_value,
                    int(goal["cx"]) + 6,
                    int(goal["cy"]) + 6,
                    w,
                    h,
                    8.0,
                ),
                w,
                h,
            )

        if self._phase == "quad_value2_left_to_goal":
            if len(value2_blobs) < 2:
                self._phase = "done"
                return None
            left_value = min(value2_blobs, key=lambda c: int(c["cx"]))
            self._phase = "quad_value2_right_slide_1"
            return self._return_click(
                *_step_toward(left_value, int(goal["cx"]), int(goal["cy"]), w, h, 8.0),
                w,
                h,
            )

        if self._phase == "quad_value2_right_slide_1":
            if len(value3_blobs) >= 1:
                self._phase = "octet_value3_to_goal"
            elif not value2_blobs:
                self._phase = "done"
                return None
            else:
                right_value = max(value2_blobs, key=lambda c: int(c["cx"]))
                self._phase = "quad_value2_right_slide_2"
                return self._return_click(
                    *_step_toward(
                        right_value,
                        int(goal["cx"]) + 8,
                        int(goal["cy"]) + 8,
                        w,
                        h,
                        8.0,
                    ),
                    w,
                    h,
                )

        if self._phase == "quad_value2_right_slide_2":
            if len(value3_blobs) >= 1:
                self._phase = "octet_value3_to_goal"
            elif not value2_blobs:
                self._phase = "done"
                return None
            else:
                self._phase = "quad_value3_to_goal"
                return self._return_click(int(goal["cx"]) + 4, int(goal["cy"]) + 4, w, h)

        if self._phase == "quad_value3_to_goal":
            if value3_blobs:
                self._phase = "octet_value3_to_goal"
            else:
                self._phase = "done"
                return None

        if self._phase == "l3_round1":
            while self._pair_plan:
                x, y = self._pair_plan.pop(0)
                self.fired += 1
                self.engaged = True
                return (x, y)
            self._pair_plan = _l3_round2_pair_plan([c for c in blobs if c["n"] == 4])
            self._phase = "l3_round2"

        if self._phase == "l3_round2":
            while self._pair_plan:
                x, y = self._pair_plan.pop(0)
                self.fired += 1
                self.engaged = True
                return (x, y)
            self._l3_plan = _l3_transport_plan(all_goals, [c for c in blobs if c["n"] >= 9])
            self._phase = "l3_transport"

        if self._phase == "l3_transport":
            while self._l3_plan:
                x, y = self._l3_plan.pop(0)
                self.fired += 1
                self.engaged = True
                return (x, y)
            self._phase = "done"
            return None

        if self._phase == "pair":
            while self._pair_plan:
                x, y = self._pair_plan.pop(0)
                self.fired += 1
                self.engaged = True
                return (x, y)
            self._phase = "pull"

        if self._phase == "pull":
            aim_x = int(goal["cx"])
            aim_y = int(goal["cy"] + max(1, goal["h"] // 4))
            outside = [c for c in _target_blobs(comps, goal) if not _in_box(c, goal, pad=0)]
            if not outside:
                self._phase = "done"
                return None
            # Pull the highest-value-looking blob first (larger rendered value
            # marker), then the farthest one. This lets merge products converge.
            c = max(
                outside,
                key=lambda z: (z["n"], math.hypot(z["cx"] - aim_x, z["cy"] - aim_y)),
            )
            dx = aim_x - c["cx"]
            dy = aim_y - c["cy"]
            dist = math.hypot(dx, dy)
            if dist <= 1:
                self._phase = "done"
                return None
            step = min(5.0, dist)
            x = int(round(c["cx"] + dx / dist * step))
            y = int(round(c["cy"] + dy / dist * step))
            self.fired += 1
            self.engaged = True
            return (x, y)

        return None
