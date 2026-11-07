"""Visible region-paint programs with exact one-step certificates.

The mechanic is rebound from the current frame on every call.  Framed
swatches identify the paint alphabet, two isolated palette squares identify
the reference and canvas, and the large palette component adjacent to the
canvas identifies the active basket sector and paint value.  Small cardinal
arrow controls are accepted only when their visible footprint can be parsed.

No episode-specific identity, source data, or remembered coordinate enters
the runtime path.  Plans expose one action, then the next frame is parsed
again.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .base import stable_digest


FAMILY_ID = "region_paint_program"

_ENGINE_TO_MODEL = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "ACTION6": "MOUSE",
}
_MODEL_TO_ENGINE = {value: key for key, value in _ENGINE_TO_MODEL.items()}
_MOVES = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
_SECTOR_POSITIONS = {
    0: (1, 0),
    1: (2, 0),
    2: (2, 1),
    3: (2, 2),
    4: (1, 2),
    5: (0, 2),
    6: (0, 1),
    7: (0, 0),
}
_POSITION_TO_SECTOR = {position: index for index, position in _SECTOR_POSITIONS.items()}
_SECTOR_NAMES = (
    "north_half",
    "north_east_triangle",
    "east_half",
    "south_east_triangle",
    "south_half",
    "south_west_triangle",
    "west_half",
    "north_west_triangle",
)
_CARDINAL_SECTORS = {0: "north", 2: "east", 4: "south", 6: "west"}
_DIRECTION_TO_SECTOR = {value: key for key, value in _CARDINAL_SECTORS.items()}
_ALL_ACTIONS = frozenset(_MODEL_TO_ENGINE)


@dataclass(frozen=True)
class _Component:
    value: int | None
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]

    @property
    def area(self) -> int:
        return len(self.cells)


@dataclass(frozen=True)
class _Swatch:
    color: int
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    frame_color: int


@dataclass
class _State:
    frame: np.ndarray
    background: int
    size: int
    reference_bbox: tuple[int, int, int, int]
    canvas_bbox: tuple[int, int, int, int]
    reference: np.ndarray
    canvas: np.ndarray
    swatches: dict[int, _Swatch]
    active_sector: int
    active_color: int
    selector_bbox: tuple[int, int, int, int]
    goal_mask: np.ndarray
    ignored_mask: np.ndarray
    basket_masks: tuple[np.ndarray, ...]
    visible_arrows: dict[str, dict[str, Any]]
    inferred_arrow_blocks: tuple[dict[str, Any], ...]


def _frame2d(frame: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(frame)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2 or arr.size == 0:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return np.asarray(arr, dtype=np.int64)


def _mode(arr: np.ndarray) -> int:
    values, counts = np.unique(arr, return_counts=True)
    return int(values[int(counts.argmax())])


def _action_name(action: Any) -> str:
    if isinstance(action, Mapping):
        action = action.get("action")
    if hasattr(action, "name"):
        action = action.name
    name = str(action or "").split(".")[-1].strip().upper()
    return _ENGINE_TO_MODEL.get(name, name)


def _action_point(action: Any) -> tuple[int, int] | None:
    if not isinstance(action, Mapping):
        return None
    try:
        if "row" in action and "col" in action:
            return int(action["col"]), int(action["row"])
        if "x" in action and "y" in action:
            return int(action["x"]), int(action["y"])
    except (TypeError, ValueError):
        return None
    return None


def _allowed_actions(valid_actions: Sequence[Any] | set[Any] | None) -> set[str]:
    if not valid_actions:
        return set(_ALL_ACTIONS)
    return {
        name for item in valid_actions if (name := _action_name(item)) in _ALL_ACTIONS
    }


def _bbox(cells: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [cell[0] for cell in cells]
    ys = [cell[1] for cell in cells]
    return min(xs), min(ys), max(xs), max(ys)


def _components(
    mask: np.ndarray,
    *,
    values: np.ndarray | None = None,
    same_value: bool = False,
) -> list[_Component]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[_Component] = []
    for y in range(height):
        for x in range(width):
            if not bool(mask[y, x]) or bool(seen[y, x]):
                continue
            value = int(values[y, x]) if values is not None else None
            stack = [(x, y)]
            seen[y, x] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if bool(seen[ny, nx]) or not bool(mask[ny, nx]):
                        continue
                    if (
                        same_value
                        and values is not None
                        and int(values[ny, nx]) != value
                    ):
                        continue
                    seen[ny, nx] = True
                    stack.append((nx, ny))
            box = _bbox(cells)
            out.append(
                _Component(
                    value=value,
                    cells=tuple(cells),
                    bbox=box,
                    centroid=(
                        sum(cell[0] for cell in cells) / len(cells),
                        sum(cell[1] for cell in cells) / len(cells),
                    ),
                )
            )
    return out


def _border_values(patch: np.ndarray) -> np.ndarray:
    return np.concatenate((patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]))


def _detect_swatches(frame: np.ndarray) -> dict[int, _Swatch]:
    raw: list[tuple[int, int, int, int, int, int]] = []
    for component in _components(
        np.ones(frame.shape, dtype=bool), values=frame, same_value=True
    ):
        x0, y0, x1, y1 = component.bbox
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        if width != height or not (3 <= width <= 9):
            continue
        patch = frame[y0 : y1 + 1, x0 : x1 + 1]
        border = _border_values(patch)
        if component.area != len(border) or not bool(np.all(border == component.value)):
            continue
        interior = patch[1:-1, 1:-1]
        if interior.size == 0 or not bool(np.all(interior == interior.flat[0])):
            continue
        border_color = int(component.value) if component.value is not None else 0
        color = int(interior.flat[0])
        if color == border_color:
            continue
        raw.append((width, border_color, color, x0, y0, x1))
    if not raw:
        return {}

    groups: dict[tuple[int, int], list[tuple[int, int, int, int, int, int]]] = {}
    for item in raw:
        groups.setdefault((item[0], item[1]), []).append(item)

    choices: list[tuple[int, int, int, list[tuple[int, int, int, int, int, int]]]] = []
    for (size, _border), items in groups.items():
        by_row: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
        by_col: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
        for item in items:
            center_x = item[3] + size // 2
            center_y = item[4] + size // 2
            by_row.setdefault(center_y, []).append(item)
            by_col.setdefault(center_x, []).append(item)
        for aligned in (*by_row.values(), *by_col.values()):
            unique: dict[int, tuple[int, int, int, int, int, int]] = {}
            for item in aligned:
                unique.setdefault(item[2], item)
            selected = list(unique.values())
            if len(selected) < 2:
                continue
            centers = sorted(
                (item[3] + size // 2, item[4] + size // 2) for item in selected
            )
            extent = centers[-1][0] - centers[0][0] + centers[-1][1] - centers[0][1]
            choices.append((len(selected), size, -extent, selected))
    if not choices:
        return {}
    _count, size, _extent, selected = max(choices, key=lambda item: item[:3])
    out: dict[int, _Swatch] = {}
    for item in selected:
        _, border_color, color, x, y, _right = item
        out[color] = _Swatch(
            color=color,
            center=(x + size // 2, y + size // 2),
            bbox=(x, y, x + size - 1, y + size - 1),
            frame_color=border_color,
        )
    return out


def _integral(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _rect_sum(integral: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int:
    return int(
        integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    )


def _isolated_palette_squares(
    frame: np.ndarray, palette: set[int]
) -> tuple[int, list[tuple[int, int, int, int]]] | None:
    palette_mask = np.isin(frame, list(palette))
    summed = _integral(palette_mask)
    height, width = frame.shape
    upper = min(24, (min(height, width) - 2) // 2)
    for size in range(upper, 3, -1):
        boxes: list[tuple[int, int, int, int]] = []
        for y in range(1, height - size):
            for x in range(1, width - size):
                if _rect_sum(summed, x, y, x + size, y + size) != size * size:
                    continue
                ring_sum = (
                    _rect_sum(summed, x - 1, y - 1, x + size + 1, y + size + 1)
                    - size * size
                )
                if ring_sum != 0:
                    continue
                boxes.append((x, y, x + size - 1, y + size - 1))
        if len(boxes) >= 2:
            return size, boxes
    return None


def _box_intersects(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> bool:
    return not (
        left[2] < right[0]
        or right[2] < left[0]
        or left[3] < right[1]
        or right[3] < left[1]
    )


def _component_hits_box(component: _Component, box: tuple[int, int, int, int]) -> bool:
    return any(
        box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in component.cells
    )


def _selector_candidate(
    frame: np.ndarray,
    palette: set[int],
    field_bbox: tuple[int, int, int, int],
    size: int,
) -> _Component | None:
    field_center = (
        (field_bbox[0] + field_bbox[2]) / 2.0,
        (field_bbox[1] + field_bbox[3]) / 2.0,
    )
    palette_mask = np.isin(frame, list(palette))
    candidates: list[_Component] = []
    for component in _components(palette_mask, values=frame, same_value=True):
        if component.area < max(8, size * size // 4):
            continue
        if _component_hits_box(component, field_bbox):
            continue
        dx = component.centroid[0] - field_center[0]
        dy = component.centroid[1] - field_center[1]
        distance = math.hypot(dx, dy)
        if size * 0.55 <= distance <= size * 2.0:
            candidates.append(component)
    if not candidates:
        return None
    return max(candidates, key=lambda component: component.area)


def _sector_from_component(
    component: _Component, canvas_bbox: tuple[int, int, int, int]
) -> int:
    center_x = (canvas_bbox[0] + canvas_bbox[2]) / 2.0
    center_y = (canvas_bbox[1] + canvas_bbox[3]) / 2.0
    dx = component.centroid[0] - center_x
    dy = component.centroid[1] - center_y
    angle = math.atan2(dx, -dy)
    return int(round(angle / (math.pi / 4.0))) % 8


def _basket_masks(size: int) -> tuple[np.ndarray, ...]:
    yy, xx = np.indices((size, size))
    half = size // 2
    return (
        yy < half,
        xx >= yy,
        xx >= half,
        xx >= size - 1 - yy,
        yy >= half,
        xx <= yy,
        xx < half,
        xx <= size - 1 - yy,
    )


def _ignored_mask(masks: tuple[np.ndarray, ...]) -> np.ndarray:
    # Complementary triangular tools overlap on these boundary cells.  Their
    # order cannot determine both sides, so the visible operator grammar marks
    # those overlap boundaries as ignored rather than assuming a fixed grid.
    return (masks[1] & masks[5]) | (masks[3] & masks[7])


def _arrow_target_blocks(
    reference: np.ndarray,
    goal_mask: np.ndarray,
    palette: set[int],
) -> tuple[dict[str, Any], ...]:
    size = int(reference.shape[0])
    out: list[dict[str, Any]] = []
    for color in sorted(palette):
        color_mask = (reference == color) & goal_mask
        for component in _components(color_mask):
            x0, y0, x1, y1 = component.bbox
            width = x1 - x0 + 1
            height = y1 - y0 + 1
            if component.area != width * height or component.area < 4:
                continue
            touches = [y0 == 0, x1 == size - 1, y1 == size - 1, x0 == 0]
            if sum(touches) != 1:
                continue
            if touches[0] or touches[2]:
                if height >= size / 2 or width >= size:
                    continue
                if abs((x0 + x1) - (size - 1)) > 1:
                    continue
                direction = "north" if touches[0] else "south"
            else:
                if width >= size / 2 or height >= size:
                    continue
                if abs((y0 + y1) - (size - 1)) > 1:
                    continue
                direction = "east" if touches[1] else "west"
            mask = np.zeros((size, size), dtype=bool)
            mask[y0 : y1 + 1, x0 : x1 + 1] = True
            out.append(
                {
                    "direction": direction,
                    "sector": _DIRECTION_TO_SECTOR[direction],
                    "color": int(color),
                    "relative_bbox": [x0, y0, x1, y1],
                    "mask": mask,
                }
            )
    out.sort(key=lambda item: (int(item["sector"]), int(item["color"])))
    return tuple(out)


def _foreground_components_near(
    frame: np.ndarray,
    background: int,
    canvas_bbox: tuple[int, int, int, int],
    size: int,
) -> list[_Component]:
    center_x = (canvas_bbox[0] + canvas_bbox[2]) / 2.0
    center_y = (canvas_bbox[1] + canvas_bbox[3]) / 2.0
    out: list[_Component] = []
    for component in _components(frame != background, values=frame):
        if _component_hits_box(component, canvas_bbox):
            continue
        distance = math.hypot(
            component.centroid[0] - center_x,
            component.centroid[1] - center_y,
        )
        if distance <= size * 2.5:
            out.append(component)
    return out


def _repeated_near_side_run(
    frame: np.ndarray,
    component: _Component,
    *,
    active_color: int,
    direction: str,
) -> tuple[int, tuple[int, int]] | None:
    x0, y0, x1, y1 = component.bbox
    if direction in {"north", "south"}:
        coordinates = (
            range(y1, y0 - 1, -1) if direction == "north" else range(y0, y1 + 1)
        )
        lines = [
            tuple(int(x) for x in range(x0, x1 + 1) if int(frame[y, x]) == active_color)
            for y in coordinates
        ]
    else:
        coordinates = (
            range(x1, x0 - 1, -1) if direction == "west" else range(x0, x1 + 1)
        )
        lines = [
            tuple(int(y) for y in range(y0, y1 + 1) if int(frame[y, x]) == active_color)
            for x in coordinates
        ]
    first = next((index for index, line in enumerate(lines) if len(line) >= 2), None)
    if first is None:
        return None
    run = lines[first]
    depth = 1
    for line in lines[first + 1 :]:
        if line != run:
            break
        depth += 1
    if depth < 2 or tuple(range(run[0], run[-1] + 1)) != run:
        return None
    return depth, (run[0], run[-1])


def _detect_visible_arrows(state: _State) -> dict[str, dict[str, Any]]:
    direction = _CARDINAL_SECTORS.get(state.active_sector)
    if direction is None:
        return {}
    center_x = (state.canvas_bbox[0] + state.canvas_bbox[2]) / 2.0
    center_y = (state.canvas_bbox[1] + state.canvas_bbox[3]) / 2.0
    selector_distance = math.hypot(
        (state.selector_bbox[0] + state.selector_bbox[2]) / 2.0 - center_x,
        (state.selector_bbox[1] + state.selector_bbox[3]) / 2.0 - center_y,
    )
    candidates: list[tuple[float, _Component]] = []
    for component in _foreground_components_near(
        state.frame, state.background, state.canvas_bbox, state.size
    ):
        width = component.bbox[2] - component.bbox[0] + 1
        height = component.bbox[3] - component.bbox[1] + 1
        if not (3 <= width <= state.size and 3 <= height <= state.size):
            continue
        if component.area < state.size:
            continue
        if not any(
            int(state.frame[y, x]) == state.active_color for x, y in component.cells
        ):
            continue
        dx = component.centroid[0] - center_x
        dy = component.centroid[1] - center_y
        distance = math.hypot(dx, dy)
        if distance <= selector_distance + 1:
            continue
        sector = int(round(math.atan2(dx, -dy) / (math.pi / 4.0))) % 8
        if sector != state.active_sector:
            continue
        candidates.append((distance, component))
    if not candidates:
        return {}
    _distance, component = min(candidates, key=lambda item: item[0])
    run = _repeated_near_side_run(
        state.frame,
        component,
        active_color=state.active_color,
        direction=direction,
    )
    if run is None:
        return {}
    depth, (span0, span1) = run
    cx0, cy0, cx1, cy1 = state.canvas_bbox
    mask = np.zeros((state.size, state.size), dtype=bool)
    if direction == "north":
        rx0, rx1 = span0 - cx0, span1 - cx0
        if not (0 <= rx0 <= rx1 < state.size):
            return {}
        mask[:depth, rx0 : rx1 + 1] = True
    elif direction == "south":
        rx0, rx1 = span0 - cx0, span1 - cx0
        if not (0 <= rx0 <= rx1 < state.size):
            return {}
        mask[state.size - depth :, rx0 : rx1 + 1] = True
    elif direction == "west":
        ry0, ry1 = span0 - cy0, span1 - cy0
        if not (0 <= ry0 <= ry1 < state.size):
            return {}
        mask[ry0 : ry1 + 1, :depth] = True
    else:
        ry0, ry1 = span0 - cy0, span1 - cy0
        if not (0 <= ry0 <= ry1 < state.size):
            return {}
        mask[ry0 : ry1 + 1, state.size - depth :] = True
    ys, xs = np.where(mask)
    if not len(xs):
        return {}
    if direction in {"north", "south"}:
        if abs((int(xs.min()) + int(xs.max())) - (state.size - 1)) > 1:
            return {}
    elif abs((int(ys.min()) + int(ys.max())) - (state.size - 1)) > 1:
        return {}
    x0, y0, x1, y1 = component.bbox
    return {
        direction: {
            "direction": direction,
            "sector": state.active_sector,
            "click": [x0 + (x1 - x0 + 1) // 2, y0 + (y1 - y0 + 1) // 2],
            "bbox": [x0, y0, x1, y1],
            "relative_bbox": [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            ],
            "mask": mask,
        }
    }


def _perceive(frame: Any) -> tuple[_State | None, str]:
    arr = _frame2d(frame)
    if arr is None or min(arr.shape) < 12:
        return None, "invalid_or_small_frame"
    background = _mode(arr)
    swatches = _detect_swatches(arr)
    if len(swatches) < 2:
        return None, "swatch_grammar_not_found"
    palette = set(swatches)
    fields = _isolated_palette_squares(arr, palette)
    if fields is None:
        return None, "paired_palette_fields_not_found"
    size, boxes = fields
    selector_rows: list[tuple[int, _Component, tuple[int, int, int, int]]] = []
    for box in boxes:
        component = _selector_candidate(arr, palette, box, size)
        if component is not None:
            selector_rows.append((component.area, component, box))
    if not selector_rows:
        return None, "active_region_selector_not_found"
    selector_rows.sort(key=lambda item: item[0], reverse=True)
    best_area, selector, canvas_bbox = selector_rows[0]
    if best_area < max(8, size * size // 3):
        return None, "active_region_selector_too_weak"
    if len(selector_rows) > 1 and selector_rows[1][0] * 4 >= best_area * 3:
        return None, "ambiguous_canvas_binding"

    reference_candidates = [box for box in boxes if box != canvas_bbox]
    if not reference_candidates:
        return None, "reference_field_not_found"
    reference_candidates.sort(
        key=lambda box: (
            len(np.unique(arr[box[1] : box[3] + 1, box[0] : box[2] + 1])),
            -abs(box[0] - canvas_bbox[0]) - abs(box[1] - canvas_bbox[1]),
        ),
        reverse=True,
    )
    reference_bbox = reference_candidates[0]
    reference = arr[
        reference_bbox[1] : reference_bbox[3] + 1,
        reference_bbox[0] : reference_bbox[2] + 1,
    ].copy()
    canvas = arr[
        canvas_bbox[1] : canvas_bbox[3] + 1,
        canvas_bbox[0] : canvas_bbox[2] + 1,
    ].copy()
    if reference.shape != (size, size) or canvas.shape != (size, size):
        return None, "field_shape_mismatch"
    if not set(int(value) for value in np.unique(reference)).issubset(palette):
        return None, "reference_uses_unbound_paint_value"
    if not set(int(value) for value in np.unique(canvas)).issubset(palette):
        return None, "canvas_uses_unbound_paint_value"

    active_sector = _sector_from_component(selector, canvas_bbox)
    if selector.value is None or int(selector.value) not in palette:
        return None, "active_paint_value_not_bound"
    masks = _basket_masks(size)
    ignored = _ignored_mask(masks)
    goal = ~ignored
    state = _State(
        frame=arr,
        background=background,
        size=size,
        reference_bbox=reference_bbox,
        canvas_bbox=canvas_bbox,
        reference=reference,
        canvas=canvas,
        swatches=swatches,
        active_sector=active_sector,
        active_color=int(selector.value),
        selector_bbox=selector.bbox,
        goal_mask=goal,
        ignored_mask=ignored,
        basket_masks=masks,
        visible_arrows={},
        inferred_arrow_blocks=_arrow_target_blocks(reference, goal, palette),
    )
    state.visible_arrows = _detect_visible_arrows(state)
    return state, "bound"


def _relative_cells(mask: np.ndarray) -> list[list[int]]:
    ys, xs = np.where(mask)
    return [[int(y), int(x)] for y, x in zip(ys.tolist(), xs.tolist())]


def _absolute_cells(
    mask: np.ndarray, canvas_bbox: tuple[int, int, int, int]
) -> list[list[int]]:
    ys, xs = np.where(mask)
    return [
        [int(y + canvas_bbox[1]), int(x + canvas_bbox[0])]
        for y, x in zip(ys.tolist(), xs.tolist())
    ]


def _swatch_rows(state: _State) -> list[dict[str, Any]]:
    return [
        {
            "role": "swatch",
            "value": int(swatch.color),
            "center": [int(swatch.center[0]), int(swatch.center[1])],
            "bbox": [int(value) for value in swatch.bbox],
        }
        for swatch in sorted(state.swatches.values(), key=lambda item: item.center)
    ]


def _static_signature(state: _State, *, include_canvas: bool) -> dict[str, Any]:
    signature: dict[str, Any] = {
        "field_size": state.size,
        "reference_bbox": list(state.reference_bbox),
        "canvas_bbox": list(state.canvas_bbox),
        "reference_cells": state.reference.tolist(),
        "ignored_cells": _relative_cells(state.ignored_mask),
        "swatches": _swatch_rows(state),
    }
    if include_canvas:
        signature["canvas_cells"] = state.canvas.tolist()
    return signature


def _binding(state: _State) -> dict[str, Any]:
    mismatch = (state.canvas != state.reference) & state.goal_mask
    arrows = [
        {
            "role": "arrow_operator",
            "direction": direction,
            "sector": int(item["sector"]),
            "click": list(item["click"]),
            "bbox": list(item["bbox"]),
            "footprint_bbox": list(item["relative_bbox"]),
        }
        for direction, item in sorted(state.visible_arrows.items())
    ]
    roles: list[dict[str, Any]] = [
        {
            "role": "reference",
            "bbox": list(state.reference_bbox),
            "size": [state.size, state.size],
            "values": sorted(int(value) for value in np.unique(state.reference)),
        },
        {
            "role": "canvas",
            "bbox": list(state.canvas_bbox),
            "size": [state.size, state.size],
            "mismatch_count": int(np.count_nonzero(mismatch)),
        },
        {
            "role": "paint_value",
            "value": state.active_color,
        },
        {
            "role": "region_selector",
            "sector": state.active_sector,
            "name": _SECTOR_NAMES[state.active_sector],
            "bbox": list(state.selector_bbox),
        },
        *_swatch_rows(state),
        *arrows,
        {
            "role": "paint_footprint_model",
            "basket_footprint_count": len(state.basket_masks),
            "arrow_footprint_count": len(state.inferred_arrow_blocks),
            "ignored_cell_count": int(np.count_nonzero(state.ignored_mask)),
        },
    ]
    return {
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "status": "bound",
        "supported": True,
        "roles": roles,
        "role_bindings": roles,
        "field_size": [state.size, state.size],
        "reference_bbox": list(state.reference_bbox),
        "canvas_bbox": list(state.canvas_bbox),
        "reference_cells": state.reference.tolist(),
        "canvas_cells": state.canvas.tolist(),
        "active_region": {
            "sector": state.active_sector,
            "name": _SECTOR_NAMES[state.active_sector],
        },
        "paint_value": state.active_color,
        "swatches": _swatch_rows(state),
        "visible_arrows": arrows,
        "ignored_cells": {
            "basis": "complementary_triangle_boundary_overlap",
            "relative": _relative_cells(state.ignored_mask),
            "absolute": _absolute_cells(state.ignored_mask, state.canvas_bbox),
        },
        "mismatch_cells": _relative_cells(mismatch),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "binding_signature": stable_digest(
            _static_signature(state, include_canvas=True)
        ),
    }


def bind_current(frame: Any, *, context: Any = None, **_kwargs: Any) -> dict[str, Any]:
    del context
    state, reason = _perceive(frame)
    if state is None:
        return {
            "family_id": FAMILY_ID,
            "mechanic_id": FAMILY_ID,
            "status": "unbound",
            "supported": False,
            "reason": reason,
            "roles": [],
            "role_bindings": [],
        }
    return _binding(state)


def compile_schema(
    frame: Any,
    level: int = 0,
    *,
    context: Any = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    binding = bind_current(frame, context=context)
    return {
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "level": int(level),
        "supported": bool(binding.get("supported")),
        "roles": list(binding.get("roles") or []),
        "required_bindings": [
            "canvas",
            "reference",
            "paint_value",
            "region_selector",
            "paint_footprint_model",
        ],
        "state_variables": [
            "canvas_cells",
            "active_region",
            "paint_value",
            "ignored_mask",
        ],
        "operator": {
            "name": FAMILY_ID,
            "transition": (
                "select one visible paint value or region, or paint one exact "
                "bound footprint"
            ),
        },
        "goal": "canvas equals the visible reference outside inferred overlap boundaries",
        "invariants": [
            "reference, swatches, and ignored boundaries remain static within a level",
            "a paint changes exactly the selected footprint to the active value",
            "each proposal contains one current-frame action",
        ],
        "falsifiers": [
            "a selection binds to a different sector or value",
            "a painted canvas differs from the exact predicted projection",
            "a visible role changes outside a certified level closure",
        ],
        "current_binding": binding,
    }


def _operation_rows(state: _State) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    palette = sorted(state.swatches)
    for sector, mask in enumerate(state.basket_masks):
        target_values = sorted(
            int(value)
            for value in np.unique(state.reference[mask & state.goal_mask])
            if int(value) in state.swatches
        )
        for color in target_values:
            rows.append(
                {
                    "kind": "basket",
                    "sector": sector,
                    "name": _SECTOR_NAMES[sector],
                    "color": color,
                    "mask": mask,
                }
            )
    for item in state.inferred_arrow_blocks:
        if int(item["color"]) not in palette:
            continue
        rows.append(
            {
                "kind": "arrow",
                "sector": int(item["sector"]),
                "name": f"{item['direction']}_arrow_block",
                "direction": str(item["direction"]),
                "color": int(item["color"]),
                "mask": np.asarray(item["mask"], dtype=bool),
            }
        )
    return tuple(rows)


def _operation_key(operation: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(operation["kind"]),
        int(operation["sector"]),
        int(operation["color"]),
        str(operation["name"]),
    )


@lru_cache(maxsize=256)
def _solve_cached(
    size: int,
    canvas_values: tuple[int, ...],
    reference_values: tuple[int, ...],
    arrow_specs: tuple[tuple[str, int, int, int, int, int], ...],
) -> tuple[tuple[str, int, int, str], ...] | None:
    canvas = np.asarray(canvas_values, dtype=np.int64).reshape((size, size))
    reference = np.asarray(reference_values, dtype=np.int64).reshape((size, size))
    masks = _basket_masks(size)
    ignored = _ignored_mask(masks)
    goal = ~ignored
    colors = sorted(int(value) for value in np.unique(reference))
    operations: list[dict[str, Any]] = []
    for sector, mask in enumerate(masks):
        for color in sorted(int(value) for value in np.unique(reference[mask & goal])):
            operations.append(
                {
                    "kind": "basket",
                    "sector": sector,
                    "color": color,
                    "name": _SECTOR_NAMES[sector],
                    "mask": mask,
                }
            )
    for direction, color, x0, y0, x1, y1 in arrow_specs:
        mask = np.zeros((size, size), dtype=bool)
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
        operations.append(
            {
                "kind": "arrow",
                "sector": _DIRECTION_TO_SECTOR[direction],
                "color": color,
                "name": f"{direction}_arrow_block",
                "mask": mask,
            }
        )
    initial_mismatch = int(np.count_nonzero((canvas != reference) & goal))
    if initial_mismatch == 0:
        return ()

    states: list[tuple[int, np.ndarray, tuple[tuple[str, int, int, str], ...]]] = [
        (initial_mismatch, canvas, ())
    ]
    seen: dict[bytes, int] = {bytes(canvas[goal].tobytes()): 0}
    max_depth = min(8, max(4, len(colors) + 2))
    beam_width = 2200
    for depth in range(1, max_depth + 1):
        next_states: dict[
            bytes, tuple[int, np.ndarray, tuple[tuple[str, int, int, str], ...]]
        ] = {}
        solutions: list[tuple[tuple[str, int, int, str], ...]] = []
        for _prior_mismatch, current, path in states:
            for operation in operations:
                mask = np.asarray(operation["mask"], dtype=bool)
                color = int(operation["color"])
                if bool(np.all(current[mask] == color)):
                    continue
                updated = current.copy()
                updated[mask] = color
                mismatch = int(np.count_nonzero((updated != reference) & goal))
                next_path = path + (_operation_key(operation),)
                if mismatch == 0:
                    solutions.append(next_path)
                    continue
                key = bytes(updated[goal].tobytes())
                if seen.get(key, depth + 1) <= depth:
                    continue
                seen[key] = depth
                existing = next_states.get(key)
                if existing is None or mismatch < existing[0]:
                    next_states[key] = (mismatch, updated, next_path)
        if solutions:
            return min(solutions, key=lambda path: (len(path), path))
        states = sorted(next_states.values(), key=lambda item: (item[0], item[2]))[
            :beam_width
        ]
        if not states:
            break
    return None


def _solve(state: _State) -> list[dict[str, Any]] | None:
    arrow_specs = tuple(
        (
            str(item["direction"]),
            int(item["color"]),
            *(int(value) for value in item["relative_bbox"]),
        )
        for item in state.inferred_arrow_blocks
    )
    keys = _solve_cached(
        state.size,
        tuple(int(value) for value in state.canvas.ravel()),
        tuple(int(value) for value in state.reference.ravel()),
        arrow_specs,
    )
    if keys is None:
        return None
    by_key = {_operation_key(item): item for item in _operation_rows(state)}
    try:
        return [by_key[key] for key in keys]
    except KeyError:
        return None


def _navigation_step(
    source_sector: int,
    target_sector: int,
    allowed: set[str],
) -> str | None:
    source = _SECTOR_POSITIONS[source_sector]
    target = _SECTOR_POSITIONS[target_sector]
    queue: deque[tuple[tuple[int, int], list[str]]] = deque([(source, [])])
    seen = {source}
    while queue:
        position, path = queue.popleft()
        if position == target:
            return path[0] if path else None
        for action in ("UP", "DOWN", "LEFT", "RIGHT"):
            if action not in allowed:
                continue
            dx, dy = _MOVES[action]
            next_position = (position[0] + dx, position[1] + dy)
            if next_position not in _POSITION_TO_SECTOR or next_position in seen:
                continue
            seen.add(next_position)
            queue.append((next_position, path + [action]))
    return None


def _decision(
    state: _State, allowed: set[str]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]] | None:
    program = _solve(state)
    if not program:
        return None
    operation = program[0]
    target_color = int(operation["color"])
    target_sector = int(operation["sector"])
    if state.active_color != target_color:
        swatch = state.swatches.get(target_color)
        if swatch is None or "MOUSE" not in allowed:
            return None
        action = {
            "action": "MOUSE",
            "row": int(swatch.center[1]),
            "col": int(swatch.center[0]),
        }
        return action, operation, program
    if state.active_sector != target_sector:
        move = _navigation_step(state.active_sector, target_sector, allowed)
        if move is None:
            return None
        return {"action": move}, operation, program
    if str(operation["kind"]) == "basket":
        if "SPACE" not in allowed:
            return None
        return {"action": "SPACE"}, operation, program
    direction = str(operation.get("direction", ""))
    arrow = state.visible_arrows.get(direction)
    if arrow is None or "MOUSE" not in allowed:
        return None
    if not np.array_equal(
        np.asarray(arrow["mask"], dtype=bool),
        np.asarray(operation["mask"], dtype=bool),
    ):
        return None
    return (
        {
            "action": "MOUSE",
            "row": int(arrow["click"][1]),
            "col": int(arrow["click"][0]),
        },
        operation,
        program,
    )


def _action_equals(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if _action_name(left) != _action_name(right):
        return False
    if _action_name(left) != "MOUSE":
        return True
    return _action_point(left) == _action_point(right)


def _expected_after_selection(
    state: _State,
    *,
    action_name: str,
    action: Mapping[str, Any],
    operation: Mapping[str, Any],
) -> dict[str, Any] | None:
    target_sector = int(operation["sector"])
    target_color = int(operation["color"])
    if state.active_color != target_color:
        swatch = state.swatches.get(target_color)
        if action_name != "MOUSE" or swatch is None:
            return None
        if _action_point(action) != swatch.center:
            return None
        after_sector = state.active_sector
        after_color = target_color
        delta_kind = "paint_value"
    elif state.active_sector != target_sector:
        if action_name not in _MOVES:
            return None
        expected_move = _navigation_step(
            state.active_sector,
            target_sector,
            {"UP", "DOWN", "LEFT", "RIGHT"},
        )
        if action_name != expected_move:
            return None
        source = _SECTOR_POSITIONS[state.active_sector]
        dx, dy = _MOVES[action_name]
        next_position = (source[0] + dx, source[1] + dy)
        after_sector = _POSITION_TO_SECTOR.get(next_position, -1)
        if after_sector < 0:
            return None
        after_color = state.active_color
        delta_kind = "active_region"
    else:
        return None
    return {
        "family_id": FAMILY_ID,
        "accepted": True,
        "transition_kind": "selection_delta",
        "selection_kind": delta_kind,
        "action": dict(action),
        "expected_selection_delta": {
            "before_region": state.active_sector,
            "after_region": after_sector,
            "before_value": state.active_color,
            "after_value": after_color,
        },
        "static_signature": _static_signature(state, include_canvas=True),
        "static_digest": stable_digest(_static_signature(state, include_canvas=True)),
    }


def _expected_after_paint(
    state: _State,
    *,
    action_name: str,
    action: Mapping[str, Any],
    operation: Mapping[str, Any],
) -> dict[str, Any] | None:
    if state.active_color != int(operation["color"]):
        return None
    if state.active_sector != int(operation["sector"]):
        return None
    kind = str(operation["kind"])
    if kind == "basket":
        if action_name != "SPACE":
            return None
    elif kind == "arrow":
        direction = str(operation.get("direction", ""))
        arrow = state.visible_arrows.get(direction)
        if action_name != "MOUSE" or arrow is None:
            return None
        if _action_point(action) != tuple(arrow["click"]):
            return None
        if not np.array_equal(arrow["mask"], operation["mask"]):
            return None
    else:
        return None
    footprint = np.asarray(operation["mask"], dtype=bool)
    canvas_after = state.canvas.copy()
    canvas_after[footprint] = state.active_color
    changed = footprint & (state.canvas != state.active_color)
    changed_cells: list[dict[str, int]] = []
    for row, col in _relative_cells(changed):
        changed_cells.append(
            {
                "row": row + state.canvas_bbox[1],
                "col": col + state.canvas_bbox[0],
                "before": int(state.canvas[row, col]),
                "after": state.active_color,
            }
        )
    mismatch_before = int(
        np.count_nonzero((state.canvas != state.reference) & state.goal_mask)
    )
    mismatch_after = int(
        np.count_nonzero((canvas_after != state.reference) & state.goal_mask)
    )
    if not changed_cells or mismatch_after >= mismatch_before:
        return None
    return {
        "family_id": FAMILY_ID,
        "accepted": True,
        "transition_kind": "canvas_paint",
        "paint_operator": kind,
        "action": dict(action),
        "paint_value": state.active_color,
        "active_region": state.active_sector,
        "footprint_cells": _absolute_cells(footprint, state.canvas_bbox),
        "changed_cells": changed_cells,
        "changed_cell_count": len(changed_cells),
        "canvas_before": state.canvas.tolist(),
        "canvas_after": canvas_after.tolist(),
        "mismatch_before": mismatch_before,
        "mismatch_after": mismatch_after,
        "closure_expected": mismatch_after == 0,
        "static_signature": _static_signature(state, include_canvas=False),
        "static_digest": stable_digest(_static_signature(state, include_canvas=False)),
    }


def predict_transition(
    frame: Any,
    action: Any,
    *,
    context: Any = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    del context
    state, reason = _perceive(frame)
    if state is None or not isinstance(action, Mapping):
        return {
            "family_id": FAMILY_ID,
            "accepted": False,
            "reason": reason if state is None else "invalid_action",
        }
    program = _solve(state)
    if not program:
        return {
            "family_id": FAMILY_ID,
            "accepted": False,
            "reason": "no_exact_residual_program",
        }
    operation = program[0]
    action_name = _action_name(action)
    selection = _expected_after_selection(
        state,
        action_name=action_name,
        action=action,
        operation=operation,
    )
    if selection is not None:
        return selection
    paint = _expected_after_paint(
        state,
        action_name=action_name,
        action=action,
        operation=operation,
    )
    if paint is not None:
        return paint
    return {
        "family_id": FAMILY_ID,
        "accepted": False,
        "reason": "action_is_not_the_current_exact_program_prefix",
    }


def propose_plan(
    frame: Any,
    *,
    valid_actions: Sequence[Any] | set[Any] | None = None,
    context: Any = None,
    **_kwargs: Any,
) -> dict[str, Any] | None:
    del context
    state, _reason = _perceive(frame)
    if state is None:
        return None
    decision = _decision(state, _allowed_actions(valid_actions))
    if decision is None:
        return None
    action, operation, program = decision
    prediction = predict_transition(frame, action)
    if not bool(prediction.get("accepted")):
        return None
    mismatch = int(
        np.count_nonzero((state.canvas != state.reference) & state.goal_mask)
    )
    return {
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "status": "proposed",
        "certified": True,
        "certificate_reason": (
            "visible roles and an exact one-step selection or paint projection"
        ),
        "actions": [action],
        "action_count": 1,
        "suffix_exposed": False,
        "transition_rule": (
            "rebind the current visible region program after one certified action"
        ),
        "first_prediction": prediction,
        "progress_measure": {
            "metric": "non_ignored_canvas_mismatch",
            "before": mismatch,
            "expected_after": int(prediction.get("mismatch_after", mismatch)),
            "strict": False,
            "after_current_operation": int(prediction.get("mismatch_after", mismatch)),
        },
        "grounded_observation": {
            "invariants": [
                "reference and swatch bindings remain static within the level",
                "only the certified selection or canvas projection is accepted",
            ],
            "falsifiers": [
                "the next binding has a different selection delta",
                "the next canvas differs from the exact predicted projection",
            ],
        },
        "detail": {
            "operation_kind": str(operation["kind"]),
            "operation_name": str(operation["name"]),
            "target_sector": int(operation["sector"]),
            "residual_operation_count": len(program),
            "visible_arrow_count": len(state.visible_arrows),
            "private_plan_length": 1,
        },
    }


def _current_matches_static(
    state: _State, expected: Mapping[str, Any], *, include_canvas: bool
) -> bool:
    signature = _static_signature(state, include_canvas=include_canvas)
    return signature == expected.get("static_signature") and stable_digest(
        signature
    ) == str(expected.get("static_digest", ""))


def verify_transition(expected: Any, current_frame: Any) -> bool:
    if not isinstance(expected, Mapping) or not bool(expected.get("accepted")):
        return False
    if str(expected.get("family_id", "")) != FAMILY_ID:
        return False
    state, _reason = _perceive(current_frame)
    kind = str(expected.get("transition_kind", ""))
    if kind == "selection_delta":
        if state is None or not _current_matches_static(
            state, expected, include_canvas=True
        ):
            return False
        delta = expected.get("expected_selection_delta")
        if not isinstance(delta, Mapping):
            return False
        return state.active_sector == int(
            delta.get("after_region", -1)
        ) and state.active_color == int(delta.get("after_value", -1))
    if kind != "canvas_paint":
        return False

    expected_canvas = np.asarray(expected.get("canvas_after"), dtype=np.int64)
    if expected_canvas.ndim != 2:
        return False
    if state is not None and _current_matches_static(
        state, expected, include_canvas=False
    ):
        if not np.array_equal(state.canvas, expected_canvas):
            return False
        if state.active_sector != int(expected.get("active_region", -1)):
            return False
        if state.active_color != int(expected.get("paint_value", -1)):
            return False
        return True

    if not bool(expected.get("closure_expected")) or state is None:
        return False
    prior_static = expected.get("static_signature")
    if not isinstance(prior_static, Mapping):
        return False
    # A scoring paint swaps in a fresh, independently valid visible program.
    # Accept that discontinuity only after the prior exact projection closed
    # every non-ignored mismatch and the new reference is demonstrably fresh.
    return (
        int(expected.get("mismatch_after", -1)) == 0
        and state.size == expected_canvas.shape[0] == expected_canvas.shape[1]
        and state.reference.tolist() != prior_static.get("reference_cells")
        and int(np.count_nonzero((state.canvas != state.reference) & state.goal_mask))
        > 0
    )


__all__ = [
    "FAMILY_ID",
    "bind_current",
    "compile_schema",
    "predict_transition",
    "propose_plan",
    "verify_transition",
]
