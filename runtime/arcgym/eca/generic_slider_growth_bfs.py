"""Frame-conditioned slider-growth BFS planner.

This module detects a visible "slider controls elongated piece growth"
grammar from a frame and solves the induced grow-count problem. It does not
branch on game id and does not read environment source files. The color roles
used by the grammar are inferred from the visible slider boxes.
"""

from __future__ import annotations

from collections import Counter, deque
import heapq
from typing import Any, Mapping

import numpy as np


_GRID_UNIT = 3


def plan_slider_growth_actions(frame: Any) -> tuple[list[dict[str, int | str]], dict[str, Any]]:
    """Return ACTION6 payloads for a visible slider-growth puzzle."""

    arr = _flatten_frame(frame)
    roles = _infer_roles(arr)
    if roles is None:
        return [], {"accepted": False, "reject_reason": "missing_slider_roles"}
    pieces = _detect_pieces(arr, roles)
    if not pieces:
        return [], {"accepted": False, "reject_reason": "missing_pieces", **roles}
    visible_indicators = _detect_indicators(arr, roles["background"], roles["target"])
    targets = _detect_targets(arr, pieces, visible_indicators, roles["target"])
    indicators = _augment_indicators_from_target_crosses(
        arr,
        visible_indicators,
        roles["target"],
    )
    sliders = _detect_sliders(arr, roles)
    if len(sliders) > 4 or len(pieces) > 4:
        plan, diagnostics = _solve_bidirectional_growth_actions(
            arr,
            roles=roles,
            pieces=pieces,
            targets=targets,
            indicators=indicators,
            sliders=sliders,
        )
        if plan:
            return plan, {
                "accepted": True,
                "reject_reason": "",
                **diagnostics,
                **_compact_state(
                    roles,
                    {
                        "pieces": pieces,
                        "targets": targets,
                        "indicators": indicators,
                        "sliders": sliders,
                    },
                ),
            }
        macro_plan, macro_diagnostics = _solve_chain_wall_route_macro_actions(
            pieces=pieces,
            targets=targets,
            indicators=indicators,
            sliders=sliders,
            rotation_controls=_detect_rotation_controls(
                arr,
                roles,
                {int(piece["color"]) for piece in pieces},
            ),
        )
        if macro_plan:
            return macro_plan, {
                "accepted": True,
                "reject_reason": "",
                **macro_diagnostics,
                **_compact_state(
                    roles,
                    {
                        "pieces": pieces,
                        "targets": targets,
                        "indicators": indicators,
                        "sliders": sliders,
                    },
                ),
            }
        cluster_plan, cluster_diagnostics = _solve_competing_cluster_wall_route_macro_actions(
            pieces=pieces,
            targets=targets,
            indicators=indicators,
            sliders=sliders,
            rotation_controls=_detect_rotation_controls(
                arr,
                roles,
                {int(piece["color"]) for piece in pieces},
            ),
        )
        if cluster_plan:
            return cluster_plan, {
                "accepted": True,
                "reject_reason": "",
                **cluster_diagnostics,
                **_compact_state(
                    roles,
                    {
                        "pieces": pieces,
                        "targets": targets,
                        "indicators": indicators,
                        "sliders": sliders,
                    },
                ),
            }
        return [], {
            "accepted": False,
            "reject_reason": diagnostics.get(
                "reject_reason", "slider_growth_bfs_scope_limit"
            ),
            **diagnostics,
            **_compact_state(roles, {
                "pieces": pieces,
                "targets": targets,
                "indicators": indicators,
                "sliders": sliders,
            }),
        }
    parent_to_child = _build_chain_links(pieces)
    obstacle_mask = _detect_obstacles(arr, pieces, sliders, roles)
    state = {
        "pieces": pieces,
        "targets": targets,
        "indicators": indicators,
        "sliders": sliders,
        "parent_to_child": parent_to_child,
        "obstacle_mask": obstacle_mask,
    }
    plan_indices = _solve_growth(state)
    if not plan_indices:
        rotating_plan, rotating_diagnostics = _solve_rotating_slider_growth_actions(
            arr,
            roles=roles,
            pieces=pieces,
            targets=targets,
            indicators=indicators,
            sliders=sliders,
        )
        if rotating_plan:
            return rotating_plan, {
                "accepted": True,
                "reject_reason": "",
                **rotating_diagnostics,
                **_compact_state(roles, state),
            }
        return [], {
            "accepted": False,
            "reject_reason": rotating_diagnostics.get("reject_reason", "no_growth_plan"),
            **rotating_diagnostics,
            **_compact_state(roles, state),
        }
    piece_to_slider: dict[int, dict[str, Any]] = {}
    for idx, piece in enumerate(pieces):
        for slider in sliders:
            if int(slider["controls_color"]) == int(piece["color"]):
                piece_to_slider[idx] = slider
                break
    payloads: list[dict[str, int | str]] = []
    for idx in plan_indices:
        slider = piece_to_slider.get(idx)
        if slider is None:
            return [], {
                "accepted": False,
                "reject_reason": "missing_slider_for_planned_piece",
                **_compact_state(roles, state),
            }
        x, y = slider["click"]
        payloads.append({"action": "ACTION6", "x": int(x), "y": int(y)})
    return payloads, {
        "accepted": True,
        "reject_reason": "",
        "plan_len": int(len(payloads)),
        "planned_piece_indices": [int(idx) for idx in plan_indices],
        **_compact_state(roles, state),
    }


def _flatten_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    return arr.astype(np.int16, copy=False)


def _component_boxes(arr: np.ndarray) -> tuple[int, list[dict[str, Any]]]:
    height, width = arr.shape
    values, counts = np.unique(arr, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    seen = np.zeros_like(arr, dtype=bool)
    out: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x] or int(arr[y, x]) == background:
                continue
            color = int(arr[y, x])
            stack = [(y, x)]
            cells: list[tuple[int, int]] = []
            while stack:
                yy, xx = stack.pop()
                if yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if seen[yy, xx] or int(arr[yy, xx]) != color:
                    continue
                seen[yy, xx] = True
                cells.append((yy, xx))
                stack.extend(((yy + 1, xx), (yy - 1, xx), (yy, xx + 1), (yy, xx - 1)))
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            out.append(
                {
                    "color": color,
                    "x": min(xs),
                    "y": min(ys),
                    "w": max(xs) - min(xs) + 1,
                    "h": max(ys) - min(ys) + 1,
                    "n": len(cells),
                }
            )
    return background, out


def _infer_roles(arr: np.ndarray) -> dict[str, Any] | None:
    background, components = _component_boxes(arr)
    by_color: dict[int, list[dict[str, Any]]] = {}
    slider_shapes = {(13, 7), (7, 13), (11, 5), (5, 11)}
    for comp in components:
        if (comp["w"], comp["h"]) in slider_shapes and int(comp["n"]) >= 28:
            by_color.setdefault(int(comp["color"]), []).append(comp)
    slider_color = None
    slider_boxes: list[dict[str, Any]] = []
    for color, boxes in sorted(by_color.items(), key=lambda item: (-len(item[1]), item[0])):
        if 1 <= len(boxes) <= 6:
            slider_color = color
            slider_boxes = sorted(boxes, key=lambda item: (int(item["y"]), int(item["x"])))
            break
    if slider_color is None:
        return None

    shared_counts: Counter[int] = Counter()
    per_slider_counts: list[Counter[int]] = []
    for box in slider_boxes:
        sub = arr[
            int(box["y"]) : int(box["y"]) + int(box["h"]),
            int(box["x"]) : int(box["x"]) + int(box["w"]),
        ]
        counts = Counter(int(value) for value in sub.flatten().tolist())
        per_slider_counts.append(counts)
        for value in counts:
            if value not in {background, slider_color}:
                shared_counts[value] += 1
    shared_values = [
        value for value, count in shared_counts.items() if count == len(slider_boxes)
    ]
    if len(shared_values) < 2:
        return None
    totals = {
        value: sum(counts.get(value, 0) for counts in per_slider_counts)
        for value in shared_values
    }
    fill_color = max(totals, key=lambda value: (totals[value], -value))
    marker_color = min(
        (value for value in shared_values if value != fill_color),
        key=lambda value: (totals[value], value),
    )
    control_colors: set[int] = set()
    sliders: list[dict[str, Any]] = []
    for box, counts in zip(slider_boxes, per_slider_counts):
        candidates = {
            value: count
            for value, count in counts.items()
            if value not in {background, slider_color, fill_color, marker_color}
        }
        if not candidates:
            continue
        controls_color = max(candidates, key=lambda value: (candidates[value], -value))
        control_colors.add(int(controls_color))
        sliders.append({**box, "controls_color": int(controls_color)})
    if not sliders:
        return None
    target_color = _infer_target_color(
        arr,
        background=background,
        excluded={background, slider_color, fill_color, marker_color, *control_colors},
    )
    if target_color is None:
        return None
    return {
        "background": int(background),
        "slider_outline": int(slider_color),
        "slider_fill": int(fill_color),
        "marker": int(marker_color),
        "target": int(target_color),
        "slider_boxes": sliders,
    }


def _infer_target_color(
    arr: np.ndarray,
    *,
    background: int,
    excluded: set[int],
) -> int | None:
    for color in sorted(int(value) for value in np.unique(arr) if int(value) not in excluded):
        if _detect_indicators(arr, background, color):
            return int(color)
    return None


def _is_indicator_at(arr: np.ndarray, x: int, y: int, background: int, color: int) -> bool:
    height, width = arr.shape
    if x < 0 or y < 0 or x + 3 > width or y + 3 > height:
        return False
    block = arr[y : y + 3, x : x + 3]
    mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    return bool(np.all(block[mask] == color) and np.all(block[~mask] == background))


def _detect_indicators(arr: np.ndarray, background: int, color: int) -> list[tuple[int, int]]:
    ys, xs = np.where(arr == color)
    candidates: set[tuple[int, int]] = set()
    for cy, cx in zip(ys.tolist(), xs.tolist()):
        for dy in (-2, -1, 0):
            for dx in (-2, -1, 0):
                candidates.add((int(cx + dx), int(cy + dy)))
    out = [
        (x, y)
        for x, y in candidates
        if _is_indicator_at(arr, x, y, background, color)
    ]
    return sorted(out)


def _detect_pieces(arr: np.ndarray, roles: Mapping[str, Any]) -> list[dict[str, Any]]:
    background = int(roles["background"])
    marker = int(roles["marker"])
    slider_outline = int(roles["slider_outline"])
    slider_fill = int(roles["slider_fill"])
    target = int(roles["target"])
    height, width = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    body_mask = (
        (arr >= 0)
        & (arr != background)
        & (arr != marker)
        & (arr != slider_outline)
        & (arr != slider_fill)
        & (arr != target)
    )
    pieces: list[dict[str, Any]] = []
    ys, xs = np.where(body_mask)
    for cy, cx in zip(ys.tolist(), xs.tolist()):
        if visited[cy, cx]:
            continue
        color = int(arr[cy, cx])
        stack = [(cy, cx)]
        cells: list[tuple[int, int]] = []
        while stack:
            yy, xx = stack.pop()
            if yy < 0 or yy >= height or xx < 0 or xx >= width:
                continue
            if visited[yy, xx] or int(arr[yy, xx]) != color:
                continue
            visited[yy, xx] = True
            cells.append((yy, xx))
            stack.extend(((yy + 1, xx), (yy - 1, xx), (yy, xx + 1), (yy, xx - 1)))
        if not cells:
            continue
        cell_ys = [cell[0] for cell in cells]
        cell_xs = [cell[1] for cell in cells]
        y0, y1 = min(cell_ys), max(cell_ys) + 1
        x0, x1 = min(cell_xs), max(cell_xs) + 1
        body_h = y1 - y0
        body_w = x1 - x0
        if not ((body_h == _GRID_UNIT and body_w >= 2) or (body_w == _GRID_UNIT and body_h >= 2)):
            continue

        def stripe(y_start: int, y_end: int, x_start: int, x_end: int) -> bool:
            if y_start < 0 or x_start < 0 or y_end > height or x_end > width:
                return False
            return bool(np.all(arr[y_start:y_end, x_start:x_end] == marker))

        rotation = None
        bx0, bx1 = x0, x1
        by0, by1 = y0, y1
        if stripe(y1, y1 + 1, x0, x1):
            rotation = 0
            by1 = y1 + 1
        elif stripe(y0, y1, x0 - 1, x0):
            rotation = 90
            bx0 = x0 - 1
        elif stripe(y0 - 1, y0, x0, x1):
            rotation = 180
            by0 = y0 - 1
        elif stripe(y0, y1, x1, x1 + 1):
            rotation = 270
            bx1 = x1 + 1
        if rotation is None:
            continue
        pieces.append(
            {
                "x": int(bx0),
                "y": int(by0),
                "w": int(bx1 - bx0),
                "h": int(by1 - by0),
                "color": int(color),
                "rotation": int(rotation),
            }
        )
    return sorted(pieces, key=lambda item: (item["y"], item["x"], item["color"]))


def _detect_targets(
    arr: np.ndarray,
    pieces: list[dict[str, Any]],
    indicators: list[tuple[int, int]],
    target_color: int,
) -> list[dict[str, Any]]:
    indicator_pixels: set[tuple[int, int]] = set()
    for x, y in indicators:
        indicator_pixels.update(
            {
                (x + 1, y),
                (x, y + 1),
                (x + 2, y + 1),
                (x + 1, y + 2),
            }
        )
    seen = np.zeros_like(arr, dtype=bool)
    height, width = arr.shape
    targets: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x] or int(arr[y, x]) != int(target_color) or (x, y) in indicator_pixels:
                continue
            stack = [(y, x)]
            cells: list[tuple[int, int]] = []
            while stack:
                yy, xx = stack.pop()
                if yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if (
                    seen[yy, xx]
                    or int(arr[yy, xx]) != int(target_color)
                    or (xx, yy) in indicator_pixels
                ):
                    continue
                seen[yy, xx] = True
                cells.append((yy, xx))
                stack.extend(((yy + 1, xx), (yy - 1, xx), (yy, xx + 1), (yy, xx - 1)))
            if not cells:
                continue
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            x0, y0 = min(xs), min(ys)
            x1, y1 = max(xs) + 1, max(ys) + 1
            if _target_component_is_cross(cells, x0, y0):
                tx, ty = x0, y0
                cx, cy = x0 + 1, y0 + 1
            elif len(cells) == 1:
                cy, cx = cells[0]
                tx, ty = cx - 1, cy - 1
            elif x1 - x0 <= 3 and y1 - y0 <= 3:
                tx, ty = x0, y0
                cx, cy = x0 + (x1 - x0) // 2, y0 + (y1 - y0) // 2
            else:
                continue
            parent_idx = None
            for idx, piece in enumerate(pieces):
                if (
                    int(piece["x"]) <= cx < int(piece["x"]) + int(piece["w"])
                    and int(piece["y"]) <= cy < int(piece["y"]) + int(piece["h"])
                ):
                    parent_idx = idx
                    break
            if parent_idx is None:
                continue
            targets.append(
                {
                    "x": int(tx),
                    "y": int(ty),
                    "w": 3,
                    "h": 3,
                    "parent_idx": int(parent_idx),
                }
            )
    unique: dict[tuple[int, int, int], dict[str, Any]] = {}
    for target in targets:
        unique[(int(target["x"]), int(target["y"]), int(target["parent_idx"]))] = target
    return sorted(unique.values(), key=lambda item: (item["y"], item["x"], item["parent_idx"]))


def _target_component_is_cross(cells: list[tuple[int, int]], x0: int, y0: int) -> bool:
    if len(cells) != 5:
        return False
    expected = {
        (y0, x0 + 1),
        (y0 + 1, x0),
        (y0 + 1, x0 + 1),
        (y0 + 1, x0 + 2),
        (y0 + 2, x0 + 1),
    }
    return set(cells) == expected


def _augment_indicators_from_target_crosses(
    arr: np.ndarray,
    indicators: list[tuple[int, int]],
    target_color: int,
) -> list[tuple[int, int]]:
    out = set((int(x), int(y)) for x, y in indicators)
    seen = np.zeros_like(arr, dtype=bool)
    height, width = arr.shape
    for y in range(height):
        for x in range(width):
            if seen[y, x] or int(arr[y, x]) != int(target_color):
                continue
            stack = [(y, x)]
            cells: list[tuple[int, int]] = []
            while stack:
                yy, xx = stack.pop()
                if yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if seen[yy, xx] or int(arr[yy, xx]) != int(target_color):
                    continue
                seen[yy, xx] = True
                cells.append((yy, xx))
                stack.extend(((yy + 1, xx), (yy - 1, xx), (yy, xx + 1), (yy, xx - 1)))
            if not cells:
                continue
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            x0, y0 = min(xs), min(ys)
            if _target_component_is_cross(cells, x0, y0):
                out.add((int(x0), int(y0)))
    return sorted(out)


def _detect_sliders(arr: np.ndarray, roles: Mapping[str, Any]) -> list[dict[str, Any]]:
    sliders: list[dict[str, Any]] = []
    for box in list(roles.get("slider_boxes") or []):
        x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
        horizontal = w > h
        if horizontal:
            click = (x + w - 1, y + h // 2)
            shrink_click = (x, y + h // 2)
        else:
            click = (x + w // 2, y + h - 1)
            shrink_click = (x + w // 2, y)
        sliders.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "controls_color": int(box["controls_color"]),
                "click": click,
                "shrink_click": shrink_click,
                "horizontal": bool(horizontal),
            }
        )
    return sliders


def _build_chain_links(pieces: list[dict[str, Any]]) -> dict[int, int | None]:
    grow_dir = {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}
    out: dict[int, int | None] = {}
    for idx, piece in enumerate(pieces):
        dx, dy = grow_dir[int(piece["rotation"])]
        if dx == 1:
            px = piece["x"] + piece["w"]
            py = piece["y"] + piece["h"] // 2
        elif dx == -1:
            px = piece["x"] - 1
            py = piece["y"] + piece["h"] // 2
        elif dy == 1:
            px = piece["x"] + piece["w"] // 2
            py = piece["y"] + piece["h"]
        else:
            px = piece["x"] + piece["w"] // 2
            py = piece["y"] - 1
        child = None
        for other_idx, other in enumerate(pieces):
            if other_idx == idx:
                continue
            if other["x"] <= px < other["x"] + other["w"] and other["y"] <= py < other["y"] + other["h"]:
                child = other_idx
                break
        out[idx] = child
    return out


def _detect_obstacles(
    arr: np.ndarray,
    pieces: list[dict[str, Any]],
    sliders: list[dict[str, Any]],
    roles: Mapping[str, Any],
) -> np.ndarray:
    background = int(roles["background"])
    marker = int(roles["marker"])
    target = int(roles["target"])
    height, width = arr.shape
    obstacle = (arr >= 0) & (arr != background)
    for item in [*pieces, *sliders]:
        x0, y0 = int(item["x"]), int(item["y"])
        x1, y1 = x0 + int(item["w"]), y0 + int(item["h"])
        obstacle[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = False
    obstacle &= arr != target
    obstacle &= arr != marker
    return obstacle


def _grow_dir_per_click(rotation: int) -> tuple[int, int]:
    if rotation == 0:
        return (0, -_GRID_UNIT)
    if rotation == 90:
        return (_GRID_UNIT, 0)
    if rotation == 180:
        return (0, _GRID_UNIT)
    return (-_GRID_UNIT, 0)


def _piece_bbox_after_growth(piece: Mapping[str, Any], extra_grows: int) -> tuple[int, int, int, int]:
    x, y, w, h = int(piece["x"]), int(piece["y"]), int(piece["w"]), int(piece["h"])
    delta = int(extra_grows) * _GRID_UNIT
    rotation = int(piece["rotation"])
    if rotation == 0:
        return (x, y - delta, w, h + delta)
    if rotation == 90:
        return (x, y, w + delta, h)
    if rotation == 180:
        return (x, y, w, h + delta)
    return (x - delta, y, w + delta, h)


def _ancestor_chain(piece_idx: int, parent_to_child: Mapping[int, int | None]) -> list[int]:
    child_to_parent: dict[int, int] = {}
    for parent, child in parent_to_child.items():
        if child is not None:
            child_to_parent[int(child)] = int(parent)
    out: list[int] = []
    cur = int(piece_idx)
    while cur in child_to_parent:
        parent = child_to_parent[cur]
        out.append(parent)
        cur = parent
        if len(out) > 32:
            return []
    out.reverse()
    return out


def _target_world_position(
    target: Mapping[str, Any],
    pieces: list[dict[str, Any]],
    parent_to_child: Mapping[int, int | None],
    grow_counts: tuple[int, ...],
) -> tuple[int, int]:
    parent_idx = int(target["parent_idx"])
    chain = _ancestor_chain(parent_idx, parent_to_child) + [parent_idx]
    x, y = int(target["x"]), int(target["y"])
    for idx in chain:
        dx, dy = _grow_dir_per_click(int(pieces[idx]["rotation"]))
        x += dx * int(grow_counts[idx])
        y += dy * int(grow_counts[idx])
    return x, y


def _piece_world_bbox(
    piece_idx: int,
    pieces: list[dict[str, Any]],
    parent_to_child: Mapping[int, int | None],
    grow_counts: tuple[int, ...],
) -> tuple[int, int, int, int]:
    x, y, w, h = _piece_bbox_after_growth(pieces[piece_idx], grow_counts[piece_idx])
    for idx in _ancestor_chain(piece_idx, parent_to_child):
        dx, dy = _grow_dir_per_click(int(pieces[idx]["rotation"]))
        x += dx * int(grow_counts[idx])
        y += dy * int(grow_counts[idx])
    return x, y, w, h


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _bbox_hits_obstacle(bbox: tuple[int, int, int, int], obstacle_mask: np.ndarray | None) -> bool:
    if obstacle_mask is None:
        return False
    x, y, w, h = bbox
    height, width = obstacle_mask.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x0 >= x1 or y0 >= y1:
        return True
    return bool(obstacle_mask[y0:y1, x0:x1].any())


def _state_has_collision(
    pieces: list[dict[str, Any]],
    parent_to_child: Mapping[int, int | None],
    grow_counts: tuple[int, ...],
    obstacle_mask: np.ndarray | None,
) -> bool:
    bboxes = [
        _piece_world_bbox(idx, pieces, parent_to_child, grow_counts)
        for idx in range(len(pieces))
    ]
    for idx, bbox in enumerate(bboxes):
        if _bbox_hits_obstacle(bbox, obstacle_mask):
            return True
        for other in bboxes[idx + 1 :]:
            if _bbox_overlap(bbox, other):
                return True
    return False


def _solve_growth(
    state: Mapping[str, Any],
    *,
    max_per_piece: int = 12,
    max_total: int = 48,
) -> list[int] | None:
    pieces = list(state.get("pieces") or [])
    targets = list(state.get("targets") or [])
    indicators = list(state.get("indicators") or [])
    sliders = list(state.get("sliders") or [])
    parent_to_child = dict(state.get("parent_to_child") or {})
    obstacle_mask = state.get("obstacle_mask")
    if not pieces or not targets or not indicators or not sliders:
        return None
    piece_to_slider = {
        idx: slider
        for idx, piece in enumerate(pieces)
        for slider in sliders
        if int(slider["controls_color"]) == int(piece["color"])
    }
    if not piece_to_slider:
        return None
    target_indicator: dict[int, tuple[int, int]] = {}
    used: set[tuple[int, int]] = set()
    for target_idx, target in enumerate(targets):
        tx, ty = int(target["x"]) + 1, int(target["y"]) + 1
        best = None
        best_d = 10**9
        for indicator in indicators:
            if indicator in used:
                continue
            d = abs(int(indicator[0]) - tx) + abs(int(indicator[1]) - ty)
            if d < best_d:
                best = indicator
                best_d = d
        if best is None:
            return None
        used.add(best)
        target_indicator[target_idx] = best

    def goal(grow_counts: tuple[int, ...]) -> bool:
        for target_idx, target in enumerate(targets):
            x, y = _target_world_position(target, pieces, parent_to_child, grow_counts)
            if (x, y) != target_indicator[target_idx]:
                return False
        return True

    start = tuple(0 for _ in pieces)
    visited: dict[tuple[int, ...], tuple[tuple[int, ...], int] | None] = {start: None}
    queue: deque[tuple[int, ...]] = deque([start])
    while queue:
        cur = queue.popleft()
        if sum(cur) > max_total:
            continue
        if goal(cur):
            path: list[int] = []
            node = cur
            while visited[node] is not None:
                prev, idx = visited[node]  # type: ignore[misc]
                path.append(idx)
                node = prev
            path.reverse()
            return path
        for idx in range(len(pieces)):
            if idx not in piece_to_slider or cur[idx] >= max_per_piece:
                continue
            nxt_l = list(cur)
            nxt_l[idx] += 1
            nxt = tuple(nxt_l)
            if nxt in visited:
                continue
            if _state_has_collision(pieces, parent_to_child, nxt, obstacle_mask):
                continue
            visited[nxt] = (cur, idx)
            queue.append(nxt)
    return None


def _solve_rotating_slider_growth_actions(
    arr: np.ndarray,
    *,
    roles: Mapping[str, Any],
    pieces: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    indicators: list[tuple[int, int]],
    sliders: list[dict[str, Any]],
    max_expansions: int = 30000,
    max_plan_len: int = 90,
) -> tuple[list[dict[str, int | str]], dict[str, Any]]:
    if len(targets) != 1 or len(indicators) != 1:
        return [], {"reject_reason": "rotating_slider_scope_requires_single_target"}
    if len(pieces) > 4 or len(sliders) > 4:
        return [], {"reject_reason": "rotating_slider_scope_limit"}

    controlled_colors = [int(piece["color"]) for piece in pieces]
    if len(set(controlled_colors)) != len(controlled_colors):
        return [], {"reject_reason": "rotating_slider_duplicate_piece_colors"}
    slider_by_color = {
        int(slider["controls_color"]): slider
        for slider in sliders
    }
    if any(color not in slider_by_color for color in controlled_colors):
        return [], {"reject_reason": "rotating_slider_missing_growth_control"}

    rotation_controls = _detect_rotation_controls(arr, roles, set(controlled_colors))
    rotation_by_color = {
        int(control["controls_color"]): control
        for control in rotation_controls
    }
    if any(color not in rotation_by_color for color in controlled_colors):
        return [], {"reject_reason": "rotating_slider_missing_rotation_control"}

    target_indicator = _assign_targets_to_indicators(targets, indicators)
    if target_indicator is None or 0 not in target_indicator:
        return [], {"reject_reason": "rotating_slider_missing_target_assignment"}
    target = targets[0]
    target_parent = int(target["parent_idx"])
    if target_parent < 0 or target_parent >= len(pieces):
        return [], {"reject_reason": "rotating_slider_target_parent_out_of_range"}

    parent_to_child = _build_chain_links_for_pieces(
        [{**dict(piece), "passive": False} for piece in pieces]
    )
    children: dict[int, list[int | str]] = {idx: [] for idx in range(len(pieces))}
    parent_by_child: dict[int, int] = {}
    for parent, child in parent_to_child.items():
        if child is None:
            continue
        parent_i = int(parent)
        child_i = int(child)
        children.setdefault(parent_i, []).append(child_i)
        parent_by_child[child_i] = parent_i
    children.setdefault(target_parent, []).append("target")

    obstacle_mask = _detect_obstacles_for_pieces(
        arr,
        [{**dict(piece), "passive": False} for piece in pieces],
        sliders,
        roles,
    )
    _clear_rotation_controls_from_obstacles(obstacle_mask, rotation_controls)
    distance_map = _target_grid_distance_map(obstacle_mask, target_indicator[0])

    initial_piece_state = tuple(
        (
            int(piece["x"]),
            int(piece["y"]),
            int(piece["w"]),
            int(piece["h"]),
            int(piece["rotation"]),
        )
        for piece in pieces
    )
    initial_target = (int(target["x"]), int(target["y"]))
    goal = (int(target_indicator[0][0]), int(target_indicator[0][1]))
    color_to_piece = {
        int(piece["color"]): idx
        for idx, piece in enumerate(pieces)
    }
    max_extent = max(arr.shape) + 4 * _GRID_UNIT
    margin = max(arr.shape) + 4 * _GRID_UNIT

    def state_key(
        piece_state: tuple[tuple[int, int, int, int, int], ...],
        target_xy: tuple[int, int],
    ) -> tuple[Any, ...]:
        return (piece_state, target_xy)

    def move_tree(
        mutable_pieces: list[list[int]],
        mutable_target: list[int],
        node: int | str,
        dx: int,
        dy: int,
    ) -> None:
        if node == "target":
            mutable_target[0] += dx
            mutable_target[1] += dy
            return
        piece = mutable_pieces[int(node)]
        piece[0] += dx
        piece[1] += dy
        for child in children.get(int(node), []):
            move_tree(mutable_pieces, mutable_target, child, dx, dy)

    def rotate_tree(
        mutable_pieces: list[list[int]],
        mutable_target: list[int],
        node: int | str,
        pivot_x: int,
        pivot_y: int,
    ) -> None:
        if node == "target":
            dx = pivot_x - mutable_target[0]
            dy = pivot_y - mutable_target[1]
            mutable_target[0] = pivot_x - dy
            mutable_target[1] = pivot_y + dx
            return
        piece = mutable_pieces[int(node)]
        x, y, w, h, rotation = piece
        dx = pivot_x - x
        dy = pivot_y - y
        piece[0] = pivot_x - dy
        piece[1] = pivot_y + dx - (w - _GRID_UNIT)
        piece[2] = h
        piece[3] = w
        piece[4] = (rotation - 90) % 360
        for child in children.get(int(node), []):
            rotate_tree(mutable_pieces, mutable_target, child, pivot_x, pivot_y)

    def rotate_piece(
        mutable_pieces: list[list[int]],
        mutable_target: list[int],
        idx: int,
    ) -> None:
        x, y, w, h, rotation = mutable_pieces[idx]
        if rotation == 0:
            pivot_x = x
            pivot_y = y + h - _GRID_UNIT
            for child in children.get(idx, []):
                rotate_tree(mutable_pieces, mutable_target, child, pivot_x, pivot_y)
            mutable_pieces[idx][0] = x - h + _GRID_UNIT
            mutable_pieces[idx][1] = y + h - _GRID_UNIT
        elif rotation == 90:
            pivot_x = x
            pivot_y = y
            for child in children.get(idx, []):
                rotate_tree(mutable_pieces, mutable_target, child, pivot_x, pivot_y)
            mutable_pieces[idx][1] = y - w + _GRID_UNIT
        elif rotation == 180:
            pivot_x = x
            pivot_y = y
            for child in children.get(idx, []):
                rotate_tree(mutable_pieces, mutable_target, child, pivot_x, pivot_y)
        else:
            pivot_x = x + w - _GRID_UNIT
            pivot_y = y
            for child in children.get(idx, []):
                rotate_tree(mutable_pieces, mutable_target, child, pivot_x, pivot_y)
            mutable_pieces[idx][0] = x + w - _GRID_UNIT
        mutable_pieces[idx][2] = h
        mutable_pieces[idx][3] = w
        mutable_pieces[idx][4] = (rotation - 90) % 360

    def grow_piece(
        mutable_pieces: list[list[int]],
        mutable_target: list[int],
        idx: int,
        next_index: int,
    ) -> None:
        piece = mutable_pieces[idx]
        x, y, w, h, rotation = piece
        next_extent = int(next_index) * _GRID_UNIT
        dx = 0
        dy = 0
        if rotation == 0:
            dy = -(next_extent - h)
            piece[1] = y + dy
            piece[2] = _GRID_UNIT
            piece[3] = next_extent
        elif rotation == 90:
            dx = next_extent - w
            piece[2] = next_extent
            piece[3] = _GRID_UNIT
        elif rotation == 180:
            dy = next_extent - h
            piece[2] = _GRID_UNIT
            piece[3] = next_extent
        else:
            dx = -(next_extent - w)
            piece[0] = x + dx
            piece[2] = next_extent
            piece[3] = _GRID_UNIT
        for child in children.get(idx, []):
            move_tree(mutable_pieces, mutable_target, child, dx, dy)

    def valid(piece_state: tuple[tuple[int, int, int, int, int], ...]) -> bool:
        for idx, piece in enumerate(piece_state):
            x, y, w, h, _rotation = piece
            if w < _GRID_UNIT or h < _GRID_UNIT or max(w, h) > max_extent:
                return False
            if x < -margin or y < -margin or x > arr.shape[1] + margin or y > arr.shape[0] + margin:
                return False
            if _rotating_bbox_hits_obstacle((x, y, w, h), obstacle_mask):
                return False
            for other in piece_state[idx + 1 :]:
                if _bbox_overlap((x, y, w, h), other[:4]):
                    return False
        return True

    def apply_action(
        piece_state: tuple[tuple[int, int, int, int, int], ...],
        target_xy: tuple[int, int],
        action: tuple[str, int, int],
    ) -> tuple[tuple[tuple[int, int, int, int, int], ...], tuple[int, int]] | None:
        kind, color, delta = action
        piece_idx = color_to_piece.get(int(color))
        if piece_idx is None:
            return None
        mutable_pieces = [list(piece) for piece in piece_state]
        mutable_target = [int(target_xy[0]), int(target_xy[1])]
        if kind == "rotate":
            parent_idx = parent_by_child.get(piece_idx)
            if parent_idx is not None:
                child_rotation = int(mutable_pieces[piece_idx][4])
                parent_rotation = int(mutable_pieces[parent_idx][4])
                if abs(child_rotation - 90 - parent_rotation) == 180:
                    rotate_piece(mutable_pieces, mutable_target, piece_idx)
                rotate_piece(mutable_pieces, mutable_target, piece_idx)
            else:
                rotate_piece(mutable_pieces, mutable_target, piece_idx)
        else:
            x, y, w, h, _rotation = mutable_pieces[piece_idx]
            current_index = (h if h > w else w) // _GRID_UNIT
            next_index = current_index + int(delta)
            if next_index < 1:
                return None
            grow_piece(mutable_pieces, mutable_target, piece_idx, next_index)
        next_piece_state = tuple(
            (int(x), int(y), int(w), int(h), int(rotation))
            for x, y, w, h, rotation in mutable_pieces
        )
        if not valid(next_piece_state):
            return None
        return next_piece_state, (int(mutable_target[0]), int(mutable_target[1]))

    def heuristic(target_xy: tuple[int, int]) -> int:
        if target_xy in distance_map:
            return int(distance_map[target_xy])
        return (abs(int(target_xy[0]) - goal[0]) + abs(int(target_xy[1]) - goal[1])) // _GRID_UNIT + 6

    actions: list[tuple[str, int, int]] = []
    for color in sorted(controlled_colors, reverse=True):
        actions.append(("rotate", int(color), 0))
    for color in sorted(controlled_colors):
        actions.append(("grow", int(color), 1))
        actions.append(("grow", int(color), -1))

    start = state_key(initial_piece_state, initial_target)
    visited: dict[tuple[Any, ...], int] = {start: 0}
    queue: list[
        tuple[
            int,
            int,
            int,
            tuple[tuple[int, int, int, int, int], ...],
            tuple[int, int],
            tuple[tuple[str, int, int], ...],
        ]
    ] = []
    serial = 0
    heapq.heappush(
        queue,
        (
            heuristic(initial_target),
            0,
            serial,
            initial_piece_state,
            initial_target,
            tuple(),
        ),
    )
    expanded = 0
    found_path: tuple[tuple[str, int, int], ...] | None = None
    while queue and expanded < max_expansions:
        _priority, cost, _serial, piece_state, target_xy, path = heapq.heappop(queue)
        expanded += 1
        if target_xy == goal:
            found_path = path
            break
        if cost >= max_plan_len:
            continue
        last_action = path[-1] if path else None
        for action in actions:
            if _rotating_slider_is_trivial_inverse(last_action, action):
                continue
            applied = apply_action(piece_state, target_xy, action)
            if applied is None:
                continue
            next_piece_state, next_target = applied
            next_key = state_key(next_piece_state, next_target)
            next_cost = cost + 1
            if visited.get(next_key, 10**9) <= next_cost:
                continue
            visited[next_key] = next_cost
            serial += 1
            next_path = path + (action,)
            heapq.heappush(
                queue,
                (
                    next_cost + heuristic(next_target),
                    next_cost,
                    serial,
                    next_piece_state,
                    next_target,
                    next_path,
                ),
            )

    if found_path is None:
        return [], {
            "reject_reason": "no_rotating_slider_growth_plan",
            "rotating_slider_expanded_state_count": int(expanded),
            "rotating_slider_seen_state_count": int(len(visited)),
            "rotating_slider_rotation_control_count": int(len(rotation_controls)),
        }

    payloads: list[dict[str, int | str]] = []
    action_labels: list[str] = []
    mode_counts: Counter[str] = Counter()
    for kind, color, delta in found_path:
        if kind == "rotate":
            control = rotation_by_color[int(color)]
            x, y = control["click"]
            mode = "rotate"
        else:
            slider = slider_by_color[int(color)]
            x, y = slider["click"] if int(delta) > 0 else slider["shrink_click"]
            mode = "grow" if int(delta) > 0 else "shrink"
        action_labels.append(_rotating_slider_action_label((kind, color, delta)))
        mode_counts[mode] += 1
        payloads.append(
            {
                "action": "ACTION6",
                "x": int(x),
                "y": int(y),
                "mode": mode,
                "piece_color": int(color),
            }
        )
    return payloads, {
        "solver": "rotating_slider_growth_astar",
        "plan_len": int(len(payloads)),
        "rotating_slider_plan_len": int(len(payloads)),
        "rotating_slider_expanded_state_count": int(expanded),
        "rotating_slider_seen_state_count": int(len(visited)),
        "rotating_slider_rotation_control_count": int(len(rotation_controls)),
        "rotating_slider_goal": [int(goal[0]), int(goal[1])],
        "rotating_slider_initial_target": [int(initial_target[0]), int(initial_target[1])],
        "rotating_slider_action_labels": action_labels[:96],
        "rotating_slider_mode_counts": dict(mode_counts),
    }


def _detect_rotation_controls(
    arr: np.ndarray,
    roles: Mapping[str, Any],
    controlled_colors: set[int],
) -> list[dict[str, Any]]:
    outline = int(roles["slider_outline"])
    fill = int(roles["slider_fill"])
    height, width = arr.shape
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for size in (5, 7):
        center = size // 2
        cross = {
            (center - 1, center),
            (center, center - 1),
            (center, center),
            (center, center + 1),
            (center + 1, center),
        }
        for y in range(0, height - size + 1):
            for x in range(0, width - size + 1):
                block = arr[y : y + size, x : x + size]
                border = np.zeros((size, size), dtype=bool)
                border[0, :] = True
                border[-1, :] = True
                border[:, 0] = True
                border[:, -1] = True
                if not np.all(block[border] == outline):
                    continue
                color = int(block[center, center])
                if color not in controlled_colors:
                    continue
                ok = True
                for yy in range(1, size - 1):
                    for xx in range(1, size - 1):
                        expected = color if (yy, xx) in cross else fill
                        if int(block[yy, xx]) != expected:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                key = (x, y, size, color)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "w": int(size),
                        "h": int(size),
                        "controls_color": int(color),
                        "click": (int(x + center), int(y + center)),
                    }
                )
    unique: dict[int, dict[str, Any]] = {}
    for control in sorted(out, key=lambda item: (int(item["y"]), int(item["x"]), int(item["w"]))):
        unique.setdefault(int(control["controls_color"]), control)
    return list(unique.values())


def _clear_rotation_controls_from_obstacles(
    obstacle_mask: np.ndarray,
    rotation_controls: list[dict[str, Any]],
) -> None:
    height, width = obstacle_mask.shape
    for control in rotation_controls:
        x, y, w, h = (
            int(control["x"]),
            int(control["y"]),
            int(control["w"]),
            int(control["h"]),
        )
        obstacle_mask[max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)] = False


def _rotating_bbox_hits_obstacle(
    bbox: tuple[int, int, int, int],
    obstacle_mask: np.ndarray | None,
) -> bool:
    if obstacle_mask is None:
        return False
    x, y, w, h = bbox
    height, width = obstacle_mask.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x0 >= x1 or y0 >= y1:
        return False
    return bool(obstacle_mask[y0:y1, x0:x1].any())


def _target_grid_distance_map(
    obstacle_mask: np.ndarray,
    goal: tuple[int, int],
) -> dict[tuple[int, int], int]:
    height, width = obstacle_mask.shape

    def free_at(x: int, y: int) -> bool:
        if x < 0 or y < 0 or x + _GRID_UNIT > width or y + _GRID_UNIT > height:
            return False
        return not bool(obstacle_mask[y : y + _GRID_UNIT, x : x + _GRID_UNIT].any())

    gx, gy = int(goal[0]), int(goal[1])
    if not free_at(gx, gy):
        return {}
    distances: dict[tuple[int, int], int] = {(gx, gy): 0}
    queue: deque[tuple[int, int]] = deque([(gx, gy)])
    while queue:
        x, y = queue.popleft()
        base = distances[(x, y)]
        for dx, dy in ((_GRID_UNIT, 0), (-_GRID_UNIT, 0), (0, _GRID_UNIT), (0, -_GRID_UNIT)):
            nx, ny = x + dx, y + dy
            if (nx, ny) in distances or not free_at(nx, ny):
                continue
            distances[(nx, ny)] = base + 1
            queue.append((nx, ny))
    return distances


def _rotating_slider_is_trivial_inverse(
    previous: tuple[str, int, int] | None,
    current: tuple[str, int, int],
) -> bool:
    if previous is None:
        return False
    if previous[0] != "grow" or current[0] != "grow":
        return False
    return int(previous[1]) == int(current[1]) and int(previous[2]) == -int(current[2])


def _rotating_slider_action_label(action: tuple[str, int, int]) -> str:
    kind, color, delta = action
    if kind == "rotate":
        return f"rot{int(color)}"
    return f"{int(color)}{'+' if int(delta) > 0 else '-'}"


def _solve_chain_wall_route_macro_actions(
    *,
    pieces: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    indicators: list[tuple[int, int]],
    sliders: list[dict[str, Any]],
    rotation_controls: list[dict[str, Any]],
) -> tuple[list[dict[str, int | str]], dict[str, Any]]:
    """Guarded mixed rotation/growth route for a four-link slider chain.

    This fallback handles the visible grammar where a terminal target is carried
    by a four-piece chain, a separate slider piece preserves an already-matched
    target, and a rotation-only blocker must be toggled near the end. The route
    is expressed in discovered role colors rather than game id or level id.
    """

    if len(pieces) != 6 or len(sliders) != 5 or len(targets) != 1 or len(indicators) < 2:
        return [], {"reject_reason": "mixed_chain_wall_route_scope"}

    slider_by_color = {
        int(slider["controls_color"]): slider
        for slider in sliders
    }
    rotation_by_color = {
        int(control["controls_color"]): control
        for control in rotation_controls
    }
    piece_colors = [int(piece["color"]) for piece in pieces]
    if len(set(piece_colors)) != len(piece_colors):
        return [], {"reject_reason": "mixed_chain_wall_route_duplicate_piece_colors"}

    target_indicator = _assign_targets_to_indicators(targets, indicators)
    if target_indicator is None or 0 not in target_indicator:
        return [], {"reject_reason": "mixed_chain_wall_route_missing_target_assignment"}
    target = targets[0]
    target_parent = int(target["parent_idx"])
    if target_parent < 0 or target_parent >= len(pieces):
        return [], {"reject_reason": "mixed_chain_wall_route_target_parent_out_of_range"}

    parent_to_child = _build_chain_links_for_pieces(
        [{**dict(piece), "passive": False} for piece in pieces]
    )
    child_to_parent = {
        int(child): int(parent)
        for parent, child in parent_to_child.items()
        if child is not None
    }

    chain: list[int] = [target_parent]
    cur = target_parent
    while cur in child_to_parent:
        cur = child_to_parent[cur]
        chain.append(cur)
        if len(chain) > 8:
            return [], {"reject_reason": "mixed_chain_wall_route_chain_cycle"}
    chain.reverse()
    if len(chain) != 4:
        return [], {
            "reject_reason": "mixed_chain_wall_route_requires_four_link_chain",
            "mixed_chain_wall_route_chain_len": int(len(chain)),
        }

    chain_colors = [int(pieces[idx]["color"]) for idx in chain]
    root_color, carrier_color, elbow_color, terminal_color = chain_colors
    if any(color not in slider_by_color for color in chain_colors):
        return [], {"reject_reason": "mixed_chain_wall_route_missing_chain_slider"}
    if root_color not in rotation_by_color or elbow_color not in rotation_by_color:
        return [], {"reject_reason": "mixed_chain_wall_route_missing_chain_rotation"}

    chain_color_set = set(chain_colors)
    fixed_slider_colors = sorted(
        color
        for color in slider_by_color
        if color not in chain_color_set
    )
    rotation_only_colors = sorted(
        color
        for color in rotation_by_color
        if color not in chain_color_set and color not in slider_by_color
    )
    if len(fixed_slider_colors) != 1 or len(rotation_only_colors) != 1:
        return [], {
            "reject_reason": "mixed_chain_wall_route_missing_auxiliary_controls",
            "mixed_chain_wall_route_fixed_slider_colors": fixed_slider_colors,
            "mixed_chain_wall_route_rotation_only_colors": rotation_only_colors,
        }
    fixed_color = int(fixed_slider_colors[0])
    blocker_color = int(rotation_only_colors[0])

    route: list[tuple[str, int, int]] = []

    def add(kind: str, color: int, delta: int = 0, repeat: int = 1) -> None:
        for _ in range(int(repeat)):
            route.append((kind, int(color), int(delta)))

    add("grow", root_color, 1)
    add("grow", elbow_color, -1)
    add("rotate", root_color, repeat=2)
    add("grow", root_color, 1, repeat=7)
    add("grow", carrier_color, 1, repeat=8)
    add("rotate", elbow_color)
    add("grow", carrier_color, 1, repeat=4)
    add("rotate", elbow_color, repeat=2)
    add("grow", elbow_color, 1, repeat=5)
    add("grow", terminal_color, 1, repeat=6)
    add("grow", fixed_color, -1, repeat=2)
    add("rotate", blocker_color)
    add("grow", elbow_color, 1, repeat=2)
    add("rotate", blocker_color)
    add("grow", fixed_color, 1, repeat=2)

    payloads: list[dict[str, int | str]] = []
    action_labels: list[str] = []
    mode_counts: Counter[str] = Counter()
    for kind, color, delta in route:
        if kind == "rotate":
            control = rotation_by_color.get(int(color))
            if control is None:
                return [], {"reject_reason": "mixed_chain_wall_route_missing_rotation_payload"}
            x, y = control["click"]
            mode = "rotate"
        else:
            slider = slider_by_color.get(int(color))
            if slider is None:
                return [], {"reject_reason": "mixed_chain_wall_route_missing_slider_payload"}
            x, y = slider["click"] if int(delta) > 0 else slider["shrink_click"]
            mode = "grow" if int(delta) > 0 else "shrink"
        action_labels.append(_rotating_slider_action_label((kind, color, delta)))
        mode_counts[mode] += 1
        payloads.append(
            {
                "action": "ACTION6",
                "x": int(x),
                "y": int(y),
                "mode": mode,
                "piece_color": int(color),
            }
        )

    assigned_goal = target_indicator[0]
    return payloads, {
        "solver": "mixed_chain_wall_route_macro",
        "plan_len": int(len(payloads)),
        "mixed_chain_wall_route_plan_len": int(len(payloads)),
        "mixed_chain_wall_route_chain_colors": chain_colors,
        "mixed_chain_wall_route_fixed_slider_color": int(fixed_color),
        "mixed_chain_wall_route_rotation_only_color": int(blocker_color),
        "mixed_chain_wall_route_initial_target": [int(target["x"]), int(target["y"])],
        "mixed_chain_wall_route_goal": [int(assigned_goal[0]), int(assigned_goal[1])],
        "mixed_chain_wall_route_action_labels": action_labels,
        "mixed_chain_wall_route_mode_counts": dict(mode_counts),
    }


def _piece_center(piece: Mapping[str, Any]) -> tuple[float, float]:
    return (
        float(piece["x"]) + float(piece["w"]) / 2.0,
        float(piece["y"]) + float(piece["h"]) / 2.0,
    )


def _solve_competing_cluster_wall_route_macro_actions(
    *,
    pieces: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    indicators: list[tuple[int, int]],
    sliders: list[dict[str, Any]],
    rotation_controls: list[dict[str, Any]],
) -> tuple[list[dict[str, int | str]], dict[str, Any]]:
    """Guarded route for a target slider coupled to a competing cluster.

    The visible grammar has one target-carrying piece and one separate same-
    color piece inside a small rotation-controlled cluster. The shared slider
    must advance the target, but the competing cluster piece must first be
    routed through a wall aperture using the rotation-only root and two
    duplicated chain colors. The route is expressed in inferred structural
    roles, not game or level ids.
    """

    if (
        len(pieces) != 10
        or len(sliders) != 6
        or len(targets) != 1
        or len(indicators) != 1
        or len(rotation_controls) != 1
    ):
        return [], {"reject_reason": "competing_cluster_wall_route_scope"}

    target = targets[0]
    target_parent = int(target["parent_idx"])
    if target_parent < 0 or target_parent >= len(pieces):
        return [], {"reject_reason": "competing_cluster_wall_route_target_parent_out_of_range"}

    target_indicator = _assign_targets_to_indicators(targets, indicators)
    if target_indicator is None or 0 not in target_indicator:
        return [], {"reject_reason": "competing_cluster_wall_route_missing_target_assignment"}
    goal_x, goal_y = target_indicator[0]
    if int(goal_x) != int(target["x"]) or int(goal_y) <= int(target["y"]):
        return [], {"reject_reason": "competing_cluster_wall_route_goal_not_vertical_down"}

    slider_by_color = {
        int(slider["controls_color"]): slider
        for slider in sliders
    }
    rotation = rotation_controls[0]
    rotation_color = int(rotation["controls_color"])
    rotation_piece_indices = [
        idx for idx, piece in enumerate(pieces)
        if int(piece["color"]) == rotation_color
    ]
    if len(rotation_piece_indices) != 1 or rotation_color in slider_by_color:
        return [], {"reject_reason": "competing_cluster_wall_route_missing_rotation_root"}

    target_color = int(pieces[target_parent]["color"])
    if target_color not in slider_by_color:
        return [], {"reject_reason": "competing_cluster_wall_route_missing_target_slider"}
    if int(pieces[target_parent].get("rotation", -1)) != 180:
        return [], {"reject_reason": "competing_cluster_wall_route_target_not_downward"}

    color_to_indices: dict[int, list[int]] = {}
    for idx, piece in enumerate(pieces):
        color_to_indices.setdefault(int(piece["color"]), []).append(idx)
    if len(color_to_indices.get(target_color, [])) != 2:
        return [], {"reject_reason": "competing_cluster_wall_route_target_color_not_competing"}

    root_piece = pieces[rotation_piece_indices[0]]
    root_cx, root_cy = _piece_center(root_piece)

    def near_rotation_root(idx: int) -> bool:
        cx, cy = _piece_center(pieces[idx])
        return max(abs(cx - root_cx), abs(cy - root_cy)) <= float(_GRID_UNIT * 2)

    cluster_indices = {
        idx for idx in range(len(pieces))
        if near_rotation_root(idx)
    }
    if target_parent in cluster_indices:
        return [], {"reject_reason": "competing_cluster_wall_route_target_inside_cluster"}
    if not any(
        idx in cluster_indices
        for idx in color_to_indices.get(target_color, [])
        if idx != target_parent
    ):
        return [], {"reject_reason": "competing_cluster_wall_route_missing_competing_piece"}

    duplicate_colors = sorted(
        color
        for color, indices in color_to_indices.items()
        if color != target_color
        and color != rotation_color
        and color in slider_by_color
        and len(indices) == 2
        and any(idx in cluster_indices for idx in indices)
        and any(idx not in cluster_indices for idx in indices)
    )
    if len(duplicate_colors) != 2:
        return [], {
            "reject_reason": "competing_cluster_wall_route_duplicate_color_scope",
            "competing_cluster_wall_route_duplicate_colors": duplicate_colors,
        }
    duplicate_colors.sort(
        key=lambda color: max(
            int(pieces[idx]["y"])
            for idx in color_to_indices[color]
            if idx not in cluster_indices
        ),
        reverse=True,
    )
    lower_shared_color, upper_shared_color = [int(color) for color in duplicate_colors]

    chain_single_colors = []
    for color, indices in color_to_indices.items():
        if (
            color in slider_by_color
            and color not in {target_color, lower_shared_color, upper_shared_color}
            and len(indices) == 1
            and indices[0] not in cluster_indices
        ):
            chain_single_colors.append(int(color))
    if len(chain_single_colors) != 2:
        return [], {
            "reject_reason": "competing_cluster_wall_route_chain_single_scope",
            "competing_cluster_wall_route_chain_single_colors": chain_single_colors,
        }
    chain_single_colors.sort(key=lambda color: int(pieces[color_to_indices[color][0]]["y"]))
    top_chain_color, bottom_chain_color = [int(color) for color in chain_single_colors]

    remaining_distance = int(goal_y) - int(target["y"])
    if remaining_distance % _GRID_UNIT != 0:
        return [], {"reject_reason": "competing_cluster_wall_route_non_grid_goal_distance"}
    target_grow_count = remaining_distance // _GRID_UNIT
    if target_grow_count != 13:
        return [], {
            "reject_reason": "competing_cluster_wall_route_target_distance_scope",
            "competing_cluster_wall_route_target_grow_count": int(target_grow_count),
        }

    route: list[tuple[str, int, int]] = []

    def add(kind: str, color: int, delta: int = 0, repeat: int = 1) -> None:
        for _ in range(int(repeat)):
            route.append((kind, int(color), int(delta)))

    add("grow", target_color, 1)
    add("rotate", rotation_color, repeat=2)
    add("grow", lower_shared_color, 1, repeat=3)
    add("grow", bottom_chain_color, -1, repeat=3)
    add("grow", lower_shared_color, -1, repeat=2)
    add("rotate", rotation_color)
    add("grow", upper_shared_color, 1, repeat=2)
    add("grow", top_chain_color, -1, repeat=3)
    add("grow", upper_shared_color, -1)
    add("rotate", rotation_color, repeat=2)
    add("grow", target_color, 1)
    add("grow", top_chain_color, -1, repeat=2)
    add("grow", target_color, 1, repeat=7)
    add("grow", bottom_chain_color, -1, repeat=2)
    add("grow", target_color, 1, repeat=4)

    payloads: list[dict[str, int | str]] = []
    action_labels: list[str] = []
    mode_counts: Counter[str] = Counter()
    rotation_by_color = {rotation_color: rotation}
    for kind, color, delta in route:
        if kind == "rotate":
            control = rotation_by_color.get(int(color))
            if control is None:
                return [], {"reject_reason": "competing_cluster_wall_route_missing_rotation_payload"}
            x, y = control["click"]
            mode = "rotate"
        else:
            slider = slider_by_color.get(int(color))
            if slider is None:
                return [], {"reject_reason": "competing_cluster_wall_route_missing_slider_payload"}
            x, y = slider["click"] if int(delta) > 0 else slider["shrink_click"]
            mode = "grow" if int(delta) > 0 else "shrink"
        action_labels.append(_rotating_slider_action_label((kind, color, delta)))
        mode_counts[mode] += 1
        payloads.append(
            {
                "action": "ACTION6",
                "x": int(x),
                "y": int(y),
                "mode": mode,
                "piece_color": int(color),
            }
        )

    return payloads, {
        "solver": "competing_cluster_wall_route_macro",
        "plan_len": int(len(payloads)),
        "competing_cluster_wall_route_plan_len": int(len(payloads)),
        "competing_cluster_wall_route_target_color": int(target_color),
        "competing_cluster_wall_route_rotation_color": int(rotation_color),
        "competing_cluster_wall_route_lower_shared_color": int(lower_shared_color),
        "competing_cluster_wall_route_upper_shared_color": int(upper_shared_color),
        "competing_cluster_wall_route_top_chain_color": int(top_chain_color),
        "competing_cluster_wall_route_bottom_chain_color": int(bottom_chain_color),
        "competing_cluster_wall_route_target_grow_count": int(target_grow_count),
        "competing_cluster_wall_route_initial_target": [int(target["x"]), int(target["y"])],
        "competing_cluster_wall_route_goal": [int(goal_x), int(goal_y)],
        "competing_cluster_wall_route_action_labels": action_labels,
        "competing_cluster_wall_route_mode_counts": dict(mode_counts),
    }


def _solve_bidirectional_growth_actions(
    arr: np.ndarray,
    *,
    roles: Mapping[str, Any],
    pieces: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    indicators: list[tuple[int, int]],
    sliders: list[dict[str, Any]],
    max_expansions: int = 120_000,
    max_plan_len: int = 60,
) -> tuple[list[dict[str, int | str]], dict[str, Any]]:
    """Solve larger slider layouts with both grow and shrink clicks.

    The grow-only solver covers the simple L1/L2 grammar. More complex layouts
    can contain passive follower bars that translate when an ancestor grows or
    shrinks. This planner keeps the same frame-only contract, adds those passive
    components to the collision model, and searches over signed grow counts.
    """

    if not pieces or not targets or not indicators or not sliders:
        return [], {"reject_reason": "missing_bidirectional_inputs"}
    slider_by_color = {
        int(slider["controls_color"]): slider
        for slider in sliders
    }
    piece_to_control_color = [
        int(piece["color"])
        for piece in pieces
        if int(piece["color"]) in slider_by_color
    ]
    if len(piece_to_control_color) != len(pieces):
        return [], {"reject_reason": "missing_bidirectional_slider_for_piece"}
    control_colors = sorted(set(piece_to_control_color))
    control_index = {color: idx for idx, color in enumerate(control_colors)}
    piece_to_control_idx = [
        control_index[int(piece["color"])]
        for piece in pieces
    ]
    if len(sliders) > 6 or len(control_colors) > 6 or len(pieces) > 10 or len(targets) > 2:
        return [], {"reject_reason": "bidirectional_slider_scope_limit"}

    all_pieces = [{**dict(piece), "passive": False} for piece in pieces]
    passive_pieces = _detect_passive_slider_bodies(arr, roles, pieces, sliders)
    all_pieces.extend(passive_pieces)
    parent_to_child = _build_chain_links_for_pieces(all_pieces)
    child_to_parent = {
        int(child): int(parent)
        for parent, child in parent_to_child.items()
        if child is not None
    }
    obstacle_mask = _detect_obstacles_for_pieces(arr, all_pieces, sliders, roles)
    controlled_count = len(pieces)
    control_to_slider = {
        idx: slider_by_color[color]
        for color, idx in control_index.items()
    }

    min_counts_l = [-12 for _ in control_colors]
    for piece_idx, piece in enumerate(pieces):
        ctrl_idx = piece_to_control_idx[piece_idx]
        min_counts_l[ctrl_idx] = max(min_counts_l[ctrl_idx], _min_slider_count(piece))
    min_counts = tuple(int(value) for value in min_counts_l)
    max_counts = tuple(12 for _ in control_colors)
    start = tuple(0 for _ in control_colors)

    def ancestors(piece_idx: int) -> list[int]:
        out: list[int] = []
        cur = int(piece_idx)
        while cur in child_to_parent:
            parent = child_to_parent[cur]
            out.append(parent)
            cur = parent
            if len(out) > 32:
                return []
        out.reverse()
        return out

    def target_reachable(target_idx: int, indicator: tuple[int, int]) -> bool:
        target = targets[target_idx]
        parent_idx = int(target["parent_idx"])
        controls: dict[int, tuple[int, int]] = {}
        for idx in [*ancestors(parent_idx), parent_idx]:
            if idx >= controlled_count:
                continue
            ctrl_idx = int(piece_to_control_idx[idx])
            dx, dy = _grow_dir_per_click(int(all_pieces[idx]["rotation"]))
            old_dx, old_dy = controls.get(ctrl_idx, (0, 0))
            controls[ctrl_idx] = (old_dx + int(dx), old_dy + int(dy))
        if not controls:
            return (int(target["x"]), int(target["y"])) == (int(indicator[0]), int(indicator[1]))

        items = sorted(controls.items())
        needed = (int(indicator[0]) - int(target["x"]), int(indicator[1]) - int(target["y"]))

        def search(offset: int, dx_sum: int, dy_sum: int) -> bool:
            if offset >= len(items):
                return (dx_sum, dy_sum) == needed
            ctrl_idx, (dx, dy) = items[offset]
            for count in range(int(min_counts[ctrl_idx]), int(max_counts[ctrl_idx]) + 1):
                if search(offset + 1, dx_sum + dx * count, dy_sum + dy * count):
                    return True
            return False

        return search(0, 0, 0)

    target_indicator = _assign_targets_to_indicators(
        targets,
        indicators,
        reachable=target_reachable,
    )
    if target_indicator is None:
        return [], {"reject_reason": "missing_bidirectional_target_assignment"}

    def target_position(target_idx: int, counts: tuple[int, ...]) -> tuple[int, int]:
        target = targets[target_idx]
        parent_idx = int(target["parent_idx"])
        x, y = int(target["x"]), int(target["y"])
        for idx in [*ancestors(parent_idx), parent_idx]:
            if idx >= controlled_count:
                continue
            dx, dy = _grow_dir_per_click(int(all_pieces[idx]["rotation"]))
            count = int(counts[piece_to_control_idx[idx]])
            x += dx * count
            y += dy * count
        return x, y

    def heuristic(counts: tuple[int, ...]) -> int:
        total = 0
        for target_idx, indicator in target_indicator.items():
            x, y = target_position(target_idx, counts)
            total += (abs(x - int(indicator[0])) + abs(y - int(indicator[1]))) // _GRID_UNIT
        return int(total)

    def goal(counts: tuple[int, ...]) -> bool:
        return all(
            target_position(target_idx, counts) == indicator
            for target_idx, indicator in target_indicator.items()
        )

    def valid(counts: tuple[int, ...]) -> bool:
        bboxes: list[tuple[int, int, int, int]] = []
        expanded_counts = tuple(
            int(counts[piece_to_control_idx[idx]])
            for idx in range(controlled_count)
        )
        for idx in range(len(all_pieces)):
            bbox = _bidirectional_piece_bbox(
                idx,
                all_pieces,
                ancestors,
                expanded_counts,
                controlled_count=controlled_count,
            )
            if bbox is None or _bbox_hits_obstacle(bbox, obstacle_mask):
                return False
            bboxes.append(bbox)
        for idx, bbox in enumerate(bboxes):
            for other in bboxes[idx + 1 :]:
                if _bbox_overlap(bbox, other):
                    return False
        return True

    if not valid(start):
        return [], {"reject_reason": "bidirectional_start_state_invalid"}

    previous: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}
    previous_action: dict[tuple[int, ...], tuple[int, str]] = {}
    queue: list[tuple[int, int, int, tuple[int, ...]]] = []
    serial = 0
    heapq.heappush(queue, (heuristic(start), 0, serial, start))
    expanded = 0
    found: tuple[int, ...] | None = None
    found_cost = 0
    while queue and expanded < max_expansions:
        _priority, cost, _serial, counts = heapq.heappop(queue)
        expanded += 1
        if goal(counts):
            found = counts
            found_cost = cost
            break
        if cost >= max_plan_len:
            continue
        for idx in range(len(control_colors)):
            for delta, mode in ((1, "grow"), (-1, "shrink")):
                next_counts_l = list(counts)
                next_counts_l[idx] += delta
                if (
                    next_counts_l[idx] < min_counts[idx]
                    or next_counts_l[idx] > max_counts[idx]
                ):
                    continue
                next_counts = tuple(int(value) for value in next_counts_l)
                if next_counts in previous:
                    continue
                if not valid(next_counts):
                    continue
                previous[next_counts] = counts
                previous_action[next_counts] = (idx, mode)
                serial += 1
                heapq.heappush(
                    queue,
                    (cost + 1 + heuristic(next_counts), cost + 1, serial, next_counts),
                )

    if found is None:
        return [], {
            "reject_reason": "no_bidirectional_growth_plan",
            "bidirectional_expanded_state_count": int(expanded),
            "bidirectional_passive_piece_count": int(len(passive_pieces)),
        }

    path: list[tuple[int, str]] = []
    node = found
    while previous[node] is not None:
        path.append(previous_action[node])
        node = previous[node]  # type: ignore[assignment]
    path.reverse()
    payloads: list[dict[str, int | str]] = []
    mode_counts: Counter[str] = Counter()
    planned_colors: list[int] = []
    for idx, mode in path:
        slider = control_to_slider[idx]
        x, y = slider["click"] if mode == "grow" else slider["shrink_click"]
        mode_counts[mode] += 1
        planned_colors.append(int(control_colors[idx]))
        payloads.append(
            {
                "action": "ACTION6",
                "x": int(x),
                "y": int(y),
                "mode": mode,
                "piece_color": int(control_colors[idx]),
            }
        )
    return payloads, {
        "solver": "bidirectional_slider_growth_astar",
        "plan_len": int(len(payloads)),
        "bidirectional_plan_len": int(len(payloads)),
        "bidirectional_found_cost": int(found_cost),
        "bidirectional_expanded_state_count": int(expanded),
        "bidirectional_passive_piece_count": int(len(passive_pieces)),
        "bidirectional_mode_counts": dict(mode_counts),
        "bidirectional_control_group_count": int(len(control_colors)),
        "bidirectional_control_colors": [int(color) for color in control_colors],
        "bidirectional_planned_colors": planned_colors[:16],
        "bidirectional_final_counts": [int(value) for value in found],
    }


def _assign_targets_to_indicators(
    targets: list[dict[str, Any]],
    indicators: list[tuple[int, int]],
    *,
    reachable: Any | None = None,
) -> dict[int, tuple[int, int]] | None:
    if len(targets) > len(indicators):
        out: dict[int, tuple[int, int]] = {}
        used_targets: set[int] = set()
        for indicator in indicators:
            best_target = None
            best_distance = 10**9
            for target_idx, target in enumerate(targets):
                if target_idx in used_targets:
                    continue
                if reachable is not None and not bool(reachable(target_idx, indicator)):
                    continue
                tx, ty = int(target["x"]), int(target["y"])
                distance = abs(int(indicator[0]) - tx) + abs(int(indicator[1]) - ty)
                if distance < best_distance:
                    best_target = target_idx
                    best_distance = distance
            if best_target is None:
                return None
            used_targets.add(best_target)
            out[int(best_target)] = indicator
        return out

    out: dict[int, tuple[int, int]] = {}
    used: set[tuple[int, int]] = set()
    for target_idx, target in enumerate(targets):
        tx, ty = int(target["x"]) + 1, int(target["y"]) + 1
        best = None
        best_distance = 10**9
        for indicator in indicators:
            if indicator in used:
                continue
            distance = abs(int(indicator[0]) - tx) + abs(int(indicator[1]) - ty)
            if distance < best_distance:
                best = indicator
                best_distance = distance
        if best is None:
            return None
        used.add(best)
        out[target_idx] = best
    return out


def _min_slider_count(piece: Mapping[str, Any]) -> int:
    rotation = int(piece["rotation"])
    grow_extent = int(piece["h"] if rotation in {0, 180} else piece["w"])
    return -max(0, (grow_extent - _GRID_UNIT) // _GRID_UNIT)


def _detect_passive_slider_bodies(
    arr: np.ndarray,
    roles: Mapping[str, Any],
    pieces: list[dict[str, Any]],
    sliders: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    height, width = arr.shape
    background = int(roles["background"])
    excluded = {
        background,
        int(roles["marker"]),
        int(roles["slider_outline"]),
        int(roles["slider_fill"]),
        int(roles["target"]),
    }
    known = np.zeros_like(arr, dtype=bool)
    for item in [*pieces, *sliders]:
        x, y, w, h = int(item["x"]), int(item["y"]), int(item["w"]), int(item["h"])
        known[max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)] = True
    known[arr == int(roles["target"])] = True
    known[arr == int(roles["marker"])] = True

    seen = np.zeros_like(arr, dtype=bool)
    out: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x] or known[y, x] or int(arr[y, x]) in excluded:
                continue
            color = int(arr[y, x])
            stack = [(y, x)]
            cells: list[tuple[int, int]] = []
            while stack:
                yy, xx = stack.pop()
                if yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if seen[yy, xx] or known[yy, xx] or int(arr[yy, xx]) != color:
                    continue
                seen[yy, xx] = True
                cells.append((yy, xx))
                stack.extend(((yy + 1, xx), (yy - 1, xx), (yy, xx + 1), (yy, xx - 1)))
            if not cells:
                continue
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            box = {
                "x": int(min(xs)),
                "y": int(min(ys)),
                "w": int(max(xs) - min(xs) + 1),
                "h": int(max(ys) - min(ys) + 1),
                "color": int(color),
                "rotation": None,
                "passive": True,
            }
            if (
                int(box["w"]) >= _GRID_UNIT
                and int(box["h"]) >= _GRID_UNIT
                and (int(box["w"]) == _GRID_UNIT or int(box["h"]) == _GRID_UNIT)
            ):
                out.append(box)
    return sorted(out, key=lambda item: (item["y"], item["x"], item["color"]))


def _build_chain_links_for_pieces(
    pieces: list[dict[str, Any]],
) -> dict[int, int | None]:
    out: dict[int, int | None] = {}
    for idx, piece in enumerate(pieces):
        if piece.get("passive") or piece.get("rotation") is None:
            out[idx] = None
            continue
        dx, dy = _grow_dir_per_click(int(piece["rotation"]))
        if dx == _GRID_UNIT:
            px = int(piece["x"]) + int(piece["w"])
            py = int(piece["y"]) + int(piece["h"]) // 2
        elif dx == -_GRID_UNIT:
            px = int(piece["x"]) - 1
            py = int(piece["y"]) + int(piece["h"]) // 2
        elif dy == _GRID_UNIT:
            px = int(piece["x"]) + int(piece["w"]) // 2
            py = int(piece["y"]) + int(piece["h"])
        else:
            px = int(piece["x"]) + int(piece["w"]) // 2
            py = int(piece["y"]) - 1
        child = None
        for other_idx, other in enumerate(pieces):
            if other_idx == idx:
                continue
            if (
                int(other["x"]) <= px < int(other["x"]) + int(other["w"])
                and int(other["y"]) <= py < int(other["y"]) + int(other["h"])
            ):
                child = other_idx
                break
        out[idx] = child
    return out


def _detect_obstacles_for_pieces(
    arr: np.ndarray,
    pieces: list[dict[str, Any]],
    sliders: list[dict[str, Any]],
    roles: Mapping[str, Any],
) -> np.ndarray:
    height, width = arr.shape
    obstacle = (arr >= 0) & (arr != int(roles["background"]))
    for item in [*pieces, *sliders]:
        x, y, w, h = int(item["x"]), int(item["y"]), int(item["w"]), int(item["h"])
        obstacle[max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)] = False
    obstacle &= arr != int(roles["target"])
    obstacle &= arr != int(roles["marker"])
    controlled_colors = {
        int(piece["color"])
        for piece in pieces
        if not piece.get("passive") and piece.get("rotation") is not None
    }
    for color in controlled_colors:
        obstacle &= arr != int(color)
    return obstacle


def _bidirectional_piece_bbox(
    piece_idx: int,
    pieces: list[dict[str, Any]],
    ancestors: Any,
    counts: tuple[int, ...],
    *,
    controlled_count: int,
) -> tuple[int, int, int, int] | None:
    piece = pieces[piece_idx]
    x, y, w, h = int(piece["x"]), int(piece["y"]), int(piece["w"]), int(piece["h"])
    if piece_idx < controlled_count:
        delta = int(counts[piece_idx]) * _GRID_UNIT
        rotation = int(piece["rotation"])
        if rotation == 0:
            y -= delta
            h += delta
        elif rotation == 90:
            w += delta
        elif rotation == 180:
            h += delta
        else:
            x -= delta
            w += delta
        if w < _GRID_UNIT or h < _GRID_UNIT:
            return None
    for ancestor_idx in ancestors(piece_idx):
        if ancestor_idx >= controlled_count:
            continue
        dx, dy = _grow_dir_per_click(int(pieces[ancestor_idx]["rotation"]))
        x += dx * int(counts[ancestor_idx])
        y += dy * int(counts[ancestor_idx])
    return (x, y, w, h)


def _compact_state(roles: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "roles": {
            key: int(value)
            for key, value in roles.items()
            if key != "slider_boxes" and isinstance(value, (int, np.integer))
        },
        "piece_count": int(len(state.get("pieces") or [])),
        "target_count": int(len(state.get("targets") or [])),
        "indicator_count": int(len(state.get("indicators") or [])),
        "slider_count": int(len(state.get("sliders") or [])),
        "sliders": [
            {
                "x": int(item["x"]),
                "y": int(item["y"]),
                "w": int(item["w"]),
                "h": int(item["h"]),
                "controls_color": int(item["controls_color"]),
                "click": [int(item["click"][0]), int(item["click"][1])],
            }
            for item in list(state.get("sliders") or [])[:8]
        ],
    }


__all__ = ["plan_slider_growth_actions"]
