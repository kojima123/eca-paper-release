"""Frame-grounded filter-flow routing with exact one-step verification.

The planner reads only the visible frame.  It binds emitters, movable bars,
and receptacles by shape, normalizes display rotation, and searches the small
action space induced by selecting and translating bars.  Candidate layouts
are evaluated both as a stream graph and by the exact local particle model;
no game identifier, environment source, or memorized action sequence enters
the runtime path.
"""
from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from arcgym.eca import primitive_synthesis as _sand


FAMILY_ID = "filter_flow_routing"
_EXACT_GRID_FAST_PATH_MIN_QUALITY = 0.98

_ENGINE_TO_MODEL = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "ACTION6": "MOUSE",
}
_MODEL_TO_ENGINE = {value: key for key, value in _ENGINE_TO_MODEL.items()}
_MOVE_TO_MODEL = {
    (0, -1): "UP",
    (0, 1): "DOWN",
    (-1, 0): "LEFT",
    (1, 0): "RIGHT",
}


def _frame2d(frame: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(frame)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


def _sample_grid(frame: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """Return a modal cell grid and its mean within-cell purity."""

    height, width = frame.shape
    grid = np.zeros((size, size), dtype=np.int64)
    purities: list[float] = []
    for gy in range(size):
        y0 = int(gy * height / size)
        y1 = max(y0 + 1, int((gy + 1) * height / size))
        for gx in range(size):
            x0 = int(gx * width / size)
            x1 = max(x0 + 1, int((gx + 1) * width / size))
            block = frame[y0:y1, x0:x1]
            values, counts = np.unique(block, return_counts=True)
            index = int(counts.argmax())
            grid[gy, gx] = int(values[index])
            purities.append(float(counts[index]) / float(block.size))
    return grid, float(sum(purities) / max(1, len(purities)))


def _same_value_components(grid: np.ndarray, background: int) -> list[dict[str, Any]]:
    height, width = grid.shape
    seen: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            value = int(grid[y, x])
            if value == background or (x, y) in seen:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and (nx, ny) not in seen
                        and int(grid[ny, nx]) == value
                    ):
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            components.append(
                {
                    "value": value,
                    "cells": cells,
                    "x": min(xs),
                    "y": min(ys),
                    "w": max(xs) - min(xs) + 1,
                    "h": max(ys) - min(ys) + 1,
                }
            )
    return components


def _detect_roles(grid: np.ndarray) -> dict[str, Any] | None:
    """Bind the gravity-oriented visible grammar without palette literals."""

    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(counts.argmax())])
    components = _same_value_components(grid, background)
    height, width = grid.shape

    bowls: list[dict[str, int]] = []
    bowl_components: set[int] = set()
    for index, item in enumerate(components):
        if int(item["w"]) != 3 or int(item["h"]) != 2 or len(item["cells"]) != 5:
            continue
        x0, y0 = int(item["x"]), int(item["y"])
        relative = {(x - x0, y - y0) for x, y in item["cells"]}
        if relative != {(0, 0), (2, 0), (0, 1), (1, 1), (2, 1)}:
            continue
        bowls.append({"x": x0, "y": y0, "w": 3, "h": 2})
        bowl_components.add(index)
    bowls.sort(key=lambda item: (item["y"], item["x"]))
    if len(bowls) < 2:
        return None

    first_bowl_y = min(item["y"] for item in bowls)
    filters: list[dict[str, Any]] = []
    filter_values: list[int] = []
    for index, item in enumerate(components):
        if index in bowl_components:
            continue
        x0, y0 = int(item["x"]), int(item["y"])
        bar_width, bar_height = int(item["w"]), int(item["h"])
        if (
            bar_height == 1
            and 2 <= bar_width <= max(2, width // 2)
            and 3 <= y0 < first_bowl_y - 1
            and len(item["cells"]) == bar_width
        ):
            filters.append(
                {
                    "x": x0,
                    "y": y0,
                    "w": bar_width,
                    "h": 1,
                    "selected": False,
                }
            )
            filter_values.append(int(item["value"]))
    if not filters or len(filters) > 6:
        return None

    order = sorted(range(len(filters)), key=lambda i: (filters[i]["y"], filters[i]["x"], filters[i]["w"]))
    filters = [filters[i] for i in order]
    filter_values = [filter_values[i] for i in order]
    if len(filters) == 1:
        filters[0]["selected"] = True
    else:
        colour_counts = Counter(filter_values)
        repeated = {value for value, count in colour_counts.items() if count > 1}
        selected_colours = {
            value for value, count in colour_counts.items() if count == 1
        }
        if repeated and len(selected_colours) == 1:
            selected_value = next(iter(selected_colours))
            for item, value in zip(filters, filter_values):
                item["selected"] = value == selected_value

    sources: list[dict[str, Any]] = []
    if height >= 2:
        for x in range(width):
            if int(grid[1, x]) == background or int(grid[0, x]) == background:
                continue
            left_clear = x == 0 or int(grid[1, x - 1]) == background
            right_clear = x == width - 1 or int(grid[1, x + 1]) == background
            below_clear = height <= 2 or int(grid[2, x]) == background
            if left_clear and right_clear and below_clear:
                sources.append({"x": int(x), "y": 0, "source_kind": "fixed"})
    if not sources or len(sources) > 4:
        return None

    return {
        "background": background,
        "filters": filters,
        "bowls": bowls,
        "sources": sorted(sources, key=lambda item: (item["y"], item["x"])),
        "source_links": [],
    }


def _candidate_grid_sizes(frame: np.ndarray) -> range | tuple[int, ...]:
    shortest = min(frame.shape)
    if shortest <= 32:
        return (int(shortest),)
    lower = max(8, int(shortest // 7))
    upper = min(24, int(shortest // 2))
    return range(lower, upper + 1)


def _perceive_state(frame: Any) -> dict[str, Any] | None:
    arr = _frame2d(frame)
    if arr is None or min(arr.shape) < 8:
        return None

    candidates: list[tuple[float, int, int, dict[str, Any]]] = []

    def scan(*, exact_scale: bool) -> None:
        for rotation_k in range(4):
            unrotated = np.rot90(arr, k=-rotation_k).copy() if rotation_k else arr
            height, width = unrotated.shape
            grid_sizes = tuple(_candidate_grid_sizes(unrotated))
            for grid_size in grid_sizes:
                divides_frame = height % grid_size == 0 and width % grid_size == 0
                if divides_frame != exact_scale:
                    continue
                grid, purity = _sample_grid(unrotated, int(grid_size))
                detected = _detect_roles(grid)
                if detected is None:
                    continue
                # Correct logical sampling makes bars and U-shaped bowls crisp.
                # Purity chooses that scale over interpolated phantom components.
                role_density = (
                    len(detected["filters"])
                    + len(detected["bowls"])
                    + len(detected["sources"])
                ) / float(grid_size * grid_size)
                quality = float(purity) + min(0.01, role_density)
                candidates.append(
                    (quality, -int(grid_size), -int(rotation_k), detected)
                )

    # ARC frames commonly render logical cells at an exact integer scale. Scan
    # those structurally privileged sizes first; a crisp role parse makes the
    # much larger non-divisor sweep redundant. Rotated layouts still test all
    # four orientations, and ambiguous/non-integer renderings retain the full
    # fallback below.
    scan(exact_scale=True)
    if not candidates or max(row[0] for row in candidates) < _EXACT_GRID_FAST_PATH_MIN_QUALITY:
        scan(exact_scale=False)

    if not candidates:
        return None

    quality, neg_size, neg_rotation, detected = max(candidates, key=lambda row: row[:3])
    grid_size = -neg_size
    rotation_k = -neg_rotation
    return {
        "mechanic_state_type": "filter_flow_routing",
        "grid_size": (grid_size, grid_size),
        "frame_shape": (int(arr.shape[0]), int(arr.shape[1])),
        "rotation_k": rotation_k,
        "perception_quality": float(quality),
        "filters": [dict(item) for item in detected["filters"]],
        "bowls": [dict(item) for item in detected["bowls"]],
        "sources": [dict(item) for item in detected["sources"]],
        "source_links": [],
    }


def _component(item: Mapping[str, Any], *, selected: bool = False) -> dict[str, Any]:
    return {
        "x": int(item.get("x", 0) or 0),
        "y": int(item.get("y", 0) or 0),
        "w": int(item.get("w", 1) or 1),
        "h": int(item.get("h", 1) or 1),
        **({"selected": bool(item.get("selected", False))} if selected else {}),
    }


def _roles(state: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    sources = [
        {
            "x": int(item.get("x", 0) or 0),
            "y": int(item.get("y", 0) or 0),
            "source_kind": str(item.get("source_kind", "fixed") or "fixed"),
        }
        for item in state.get("sources", ())
        if isinstance(item, Mapping)
    ]
    return {
        "sources": sorted(sources, key=lambda item: (item["y"], item["x"])),
        "filters": [
            _component(item, selected=True)
            for item in state.get("filters", ())
            if isinstance(item, Mapping)
        ],
        "bowls": sorted(
            (
                _component(item)
                for item in state.get("bowls", ())
                if isinstance(item, Mapping)
            ),
            key=lambda item: (item["y"], item["x"], item["w"], item["h"]),
        ),
    }


def _geometry(
    state: Mapping[str, Any],
    positions: Sequence[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]], list[tuple[int, int]]]:
    filters = [item for item in state.get("filters", ()) if isinstance(item, Mapping)]
    filter_geometry = []
    for index, item in enumerate(filters):
        x, y = (
            positions[index]
            if positions is not None
            else (int(item.get("x", 0) or 0), int(item.get("y", 0) or 0))
        )
        filter_geometry.append(
            (int(x), int(y), int(item.get("w", 1) or 1), int(item.get("h", 1) or 1))
        )
    bowls = [
        (int(item["x"]), int(item["y"]), int(item["w"]), int(item["h"]))
        for item in state.get("bowls", ())
        if isinstance(item, Mapping)
    ]
    sources = [
        (int(item["x"]), int(item["y"]))
        for item in state.get("sources", ())
        if isinstance(item, Mapping)
    ]
    return filter_geometry, bowls, sources


def _simulate_flow(
    state: Mapping[str, Any],
    positions: Sequence[tuple[int, int]] | None = None,
) -> Any:
    filters, bowls, sources = _geometry(state, positions)
    grid_size = list(state.get("grid_size") or (16, 16))
    sim = _sand._SandFilterFlowSim(
        filters=filters,
        bowls=bowls,
        sources=sources,
        grid_w=int(grid_size[0]),
        grid_h=int(grid_size[1]),
    )
    sim.run()
    return sim


def _next_stream_target(
    lane_x: int,
    after_y: int,
    filters: Sequence[Mapping[str, Any]],
    bowls: Sequence[Mapping[str, Any]],
) -> tuple[str, int] | None:
    candidates: list[tuple[int, int, str, int]] = []
    for index, item in enumerate(filters):
        x, y, width = int(item["x"]), int(item["y"]), int(item["w"])
        if y > after_y and x <= lane_x < x + width:
            candidates.append((y, 0, "filter", index))
    for index, item in enumerate(bowls):
        x, y, width = int(item["x"]), int(item["y"]), int(item["w"])
        if y <= after_y or not (x <= lane_x < x + width):
            continue
        opening_x = x + width // 2
        kind = "bowl" if lane_x == opening_x else "overflow"
        candidates.append((y, 1, kind, index))
    if not candidates:
        return None
    _y, _priority, kind, index = min(candidates)
    return kind, index


def _flow_graph(
    state: Mapping[str, Any], roles: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    sources = list(roles.get("sources", ()))
    filters = list(roles.get("filters", ()))
    bowls = list(roles.get("bowls", ()))
    sim = _simulate_flow(state)
    filled = {int(index) for index in sim.filled_bowls}

    nodes: list[dict[str, Any]] = []
    for index, item in enumerate(sources):
        nodes.append(
            {"id": f"source:{index}", "role": "source", "x": int(item["x"]), "y": int(item["y"])}
        )
    for index, item in enumerate(filters):
        nodes.append(
            {
                "id": f"filter:{index}",
                "role": "filter",
                "x": int(item["x"]),
                "y": int(item["y"]),
                "w": int(item["w"]),
                "h": int(item["h"]),
                "selected": bool(item.get("selected", False)),
            }
        )
    for index, item in enumerate(bowls):
        nodes.append(
            {
                "id": f"bowl:{index}",
                "role": "bowl",
                "x": int(item["x"]),
                "y": int(item["y"]),
                "w": int(item["w"]),
                "h": int(item["h"]),
                "demand": 1,
                "filled": index in filled,
                "residual": 0 if index in filled else 1,
            }
        )
    nodes.append({"id": "overflow", "role": "sink"})

    edges: list[dict[str, Any]] = []
    reached_filters: set[int] = set()
    pending: list[tuple[str, int, int, str]] = [
        (f"source:{index}", int(item["x"]), int(item["y"]), "down")
        for index, item in enumerate(sources)
    ]
    while pending:
        source_id, lane_x, after_y, port = pending.pop(0)
        downstream = _next_stream_target(lane_x, after_y, filters, bowls)
        if downstream is None or downstream[0] == "overflow":
            target = "overflow"
        else:
            target = f"{downstream[0]}:{downstream[1]}"
        edges.append(
            {"from": source_id, "to": target, "port": port, "lane_x": int(lane_x)}
        )
        if downstream is None or downstream[0] != "filter":
            continue
        filter_index = int(downstream[1])
        if filter_index in reached_filters:
            continue
        reached_filters.add(filter_index)
        item = filters[filter_index]
        pending.extend(
            [
                (
                    f"filter:{filter_index}",
                    int(item["x"]) - 1,
                    int(item["y"]),
                    "split_left",
                ),
                (
                    f"filter:{filter_index}",
                    int(item["x"]) + int(item["w"]),
                    int(item["y"]),
                    "split_right",
                ),
            ]
        )
    edges.sort(key=lambda item: (item["from"], item["port"], item["to"], item["lane_x"]))
    residual_ids = [f"bowl:{index}" for index in range(len(bowls)) if index not in filled]
    return {
        "nodes": nodes,
        "edges": edges,
        "reached_filter_ids": [f"filter:{index}" for index in sorted(reached_filters)],
        "filled_bowl_ids": [f"bowl:{index}" for index in sorted(filled)],
        "residual_bowl_ids": residual_ids,
        "residual_demand": len(residual_ids),
        "overflow": bool(sim.overflow),
    }


def _valid_positions(
    state: Mapping[str, Any], positions: Sequence[tuple[int, int]]
) -> bool:
    filters = [item for item in state.get("filters", ()) if isinstance(item, Mapping)]
    bowls = [item for item in state.get("bowls", ()) if isinstance(item, Mapping)]
    grid_w, grid_h = [int(value) for value in list(state.get("grid_size") or (16, 16))[:2]]
    rows: list[tuple[int, int, int]] = []
    for index, (x, y) in enumerate(positions):
        width = int(filters[index].get("w", 1) or 1)
        height = int(filters[index].get("h", 1) or 1)
        if x < 0 or y < 3 or x + width > grid_w or y + height >= grid_h:
            return False
        for bowl in bowls:
            bx, by = int(bowl["x"]), int(bowl["y"])
            bw, bh = int(bowl["w"]), int(bowl["h"])
            if (
                x < bx + bw + 1
                and x + width > bx - 1
                and y < by + bh + 1
                and y + height > by - 1
            ):
                return False
        rows.append((int(x), int(y), width))
    for index, (x, y, width) in enumerate(rows):
        for other_x, other_y, other_width in rows[:index]:
            if y == other_y and x < other_x + other_width and other_x < x + width:
                return False
    return True


def _search_abstract_plan(
    state: Mapping[str, Any],
) -> tuple[list[dict[str, int | str]], tuple[tuple[int, int], ...]] | None:
    filters = [item for item in state.get("filters", ()) if isinstance(item, Mapping)]
    if not filters:
        return None
    start_positions = tuple((int(item["x"]), int(item["y"])) for item in filters)
    selected_rows = [index for index, item in enumerate(filters) if bool(item.get("selected", False))]
    selected = selected_rows[0] if len(selected_rows) == 1 else None
    start = (start_positions, selected)
    frontier: list[tuple[tuple[tuple[int, int], ...], int | None, tuple[dict[str, int | str], ...]]] = [
        (start_positions, selected, ())
    ]
    seen = {start}
    outcome_cache: dict[tuple[tuple[int, int], ...], tuple[int, bool, int, bool]] = {}

    def outcome(positions: tuple[tuple[int, int], ...]) -> tuple[int, bool, int, bool]:
        cached = outcome_cache.get(positions)
        if cached is not None:
            return cached
        sim = _simulate_flow(state, positions)
        residual = max(0, len(state.get("bowls", ())) - len(sim.filled_bowls))
        row = (residual, bool(sim.overflow), len(sim.filled_bowls), bool(sim.is_win()))
        outcome_cache[positions] = row
        return row

    max_depth = min(14, max(6, len(filters) * 3 + 2))
    max_states = 60000
    visited = 0
    for depth in range(max_depth + 1):
        frontier.sort(
            key=lambda row: (
                outcome(row[0])[0],
                outcome(row[0])[1],
                -outcome(row[0])[2],
                row[0],
                -1 if row[1] is None else row[1],
            )
        )
        for positions, current_selected, path in frontier:
            visited += 1
            if outcome(positions)[3]:
                return [dict(item) for item in path], positions
            if visited >= max_states:
                return None
        if depth >= max_depth:
            break

        next_frontier: list[
            tuple[tuple[tuple[int, int], ...], int | None, tuple[dict[str, int | str], ...]]
        ] = []
        for positions, current_selected, path in frontier:
            if current_selected is not None:
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    moved = list(positions)
                    x, y = moved[current_selected]
                    moved[current_selected] = (x + dx, y + dy)
                    moved_tuple = tuple(moved)
                    key = (moved_tuple, current_selected)
                    if key in seen or not _valid_positions(state, moved_tuple):
                        continue
                    seen.add(key)
                    next_frontier.append(
                        (
                            moved_tuple,
                            current_selected,
                            path + ({"kind": "move", "dx": dx, "dy": dy},),
                        )
                    )
            for index in range(len(filters)):
                if index == current_selected:
                    continue
                key = (positions, index)
                if key in seen:
                    continue
                seen.add(key)
                next_frontier.append(
                    (positions, index, path + ({"kind": "select", "index": index},))
                )
        frontier = next_frontier
        if not frontier:
            break
    return None


def _display_vector(dx: int, dy: int, rotation_k: int) -> tuple[int, int]:
    if rotation_k % 4 == 1:
        return dy, -dx
    if rotation_k % 4 == 2:
        return -dx, -dy
    if rotation_k % 4 == 3:
        return -dy, dx
    return dx, dy


def _model_action_for_delta(dx: int, dy: int, rotation_k: int) -> str | None:
    return _MOVE_TO_MODEL.get(_display_vector(int(dx), int(dy), int(rotation_k)))


def _internal_delta(model_action: str, rotation_k: int) -> tuple[int, int] | None:
    for dx, dy in _MOVE_TO_MODEL:
        if _model_action_for_delta(dx, dy, rotation_k) == model_action:
            return dx, dy
    return None


def _grid_click(
    gx: float,
    gy: float,
    *,
    state: Mapping[str, Any],
) -> tuple[int, int]:
    grid_w, grid_h = [float(value) for value in list(state.get("grid_size") or (16, 16))[:2]]
    frame_h, frame_w = [int(value) for value in list(state.get("frame_shape") or (64, 64))[:2]]
    ux = (float(gx) + 0.5) * frame_w / grid_w
    uy = (float(gy) + 0.5) * frame_h / grid_h
    rotation_k = int(state.get("rotation_k", 0) or 0) % 4
    if rotation_k == 1:
        px, py = uy, frame_w - 1 - ux
    elif rotation_k == 2:
        px, py = frame_w - 1 - ux, frame_h - 1 - uy
    elif rotation_k == 3:
        px, py = frame_h - 1 - uy, ux
    else:
        px, py = ux, uy
    return (
        max(0, min(frame_w - 1, int(round(px)))),
        max(0, min(frame_h - 1, int(round(py)))),
    )


def _emit_model_actions(
    state: Mapping[str, Any], abstract: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]] | None:
    filters = [item for item in state.get("filters", ()) if isinstance(item, Mapping)]
    positions = [(int(item["x"]), int(item["y"])) for item in filters]
    selected_rows = [
        index for index, item in enumerate(filters) if bool(item.get("selected", False))
    ]
    selected = selected_rows[0] if len(selected_rows) == 1 else None
    actions: list[dict[str, Any]] = []
    rotation_k = int(state.get("rotation_k", 0) or 0)
    for step in abstract:
        if step.get("kind") == "select":
            index = int(step["index"])
            x, y = positions[index]
            width = int(filters[index].get("w", 1) or 1)
            col, row = _grid_click(x + (width - 1) / 2.0, y, state=state)
            actions.append({"action": "MOUSE", "row": row, "col": col})
            selected = index
            continue
        if step.get("kind") != "move":
            return None
        dx, dy = int(step["dx"]), int(step["dy"])
        name = _model_action_for_delta(dx, dy, rotation_k)
        if name is None:
            return None
        actions.append({"action": name})
        if selected is None:
            return None
        x, y = positions[selected]
        positions[selected] = (x + dx, y + dy)
    actions.append({"action": "SPACE"})
    return actions


def _attach_plan(state: dict[str, Any]) -> dict[str, Any]:
    solved = _search_abstract_plan(state)
    if solved is None:
        state["abstract_plan"] = []
        state["operation_plan"] = []
        state["operation_plan_len"] = 0
        state["target_filters"] = []
        return state
    abstract, target_positions = solved
    actions = _emit_model_actions(state, abstract)
    if not actions:
        state["abstract_plan"] = []
        state["operation_plan"] = []
        state["operation_plan_len"] = 0
        state["target_filters"] = []
        return state
    state["abstract_plan"] = abstract
    state["operation_plan"] = actions
    state["operation_plan_len"] = len(actions)
    state["target_filters"] = [
        {
            "x": int(position[0]),
            "y": int(position[1]),
            "w": int(state["filters"][index]["w"]),
            "h": int(state["filters"][index]["h"]),
        }
        for index, position in enumerate(target_positions)
    ]
    return state


@lru_cache(maxsize=32)
def _cached_state(
    shape: tuple[int, int], dtype: str, payload: bytes
) -> dict[str, Any] | None:
    arr = np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(shape)
    state = _perceive_state(arr)
    return None if state is None else _attach_plan(state)


def _state(frame: Any) -> dict[str, Any] | None:
    arr = _frame2d(frame)
    if arr is None:
        return None
    contiguous = np.ascontiguousarray(arr)
    cached = _cached_state(
        (int(contiguous.shape[0]), int(contiguous.shape[1])),
        contiguous.dtype.str,
        contiguous.tobytes(),
    )
    return None if cached is None else copy.deepcopy(cached)


def _flow_potential(state: Mapping[str, Any], roles: Mapping[str, Any]) -> dict[str, Any]:
    graph = _flow_graph(state, roles)
    residual = int(graph["residual_demand"])
    remaining = int(state.get("operation_plan_len", 0) or 0)
    overflow = bool(graph["overflow"])
    # A useful cascade may temporarily trade one directly filled bowl for two
    # downstream branches.  The exact shortest suffix is therefore the
    # monotone term; residual demand and overflow remain explicit tie-breakers.
    value = remaining * 100000 + residual * 100 + int(overflow)
    return {
        "unfilled_bowls": residual,
        "filled_bowls": len(graph["filled_bowl_ids"]),
        "filled_bowl_ids": list(graph["filled_bowl_ids"]),
        "residual_bowl_ids": list(graph["residual_bowl_ids"]),
        "residual_demand": residual,
        "overflow": overflow,
        "remaining_operations": remaining,
        "value": int(value),
    }


def _binding_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    roles = _roles(state)
    graph = _flow_graph(state, roles)
    return {
        "mechanic_id": FAMILY_ID,
        "accepted": True,
        "status": "bound",
        "grid_size": [int(value) for value in list(state.get("grid_size") or ())[:2]],
        "rotation_k": int(state.get("rotation_k", 0) or 0),
        "roles": roles,
        "flow_graph": graph,
        "potential": _flow_potential(state, roles),
    }


def bind_current(frame: Any) -> dict[str, Any] | None:
    """Bind visible source, filter, bowl, graph, and residual-demand roles."""

    state = _state(frame)
    return None if state is None else _binding_from_state(state)


def compile_schema(frame: Any, *, level: int) -> dict[str, Any]:
    """Compile a JSON-safe transferable mechanic schema."""

    binding = bind_current(frame)
    roles = {} if binding is None else binding["roles"]
    return {
        "schema_id": "tufa.filter_flow_routing.v2",
        "mechanic_id": FAMILY_ID,
        "source_family": "frame_only_visible_stream_graph",
        "level": int(level),
        "roles": roles,
        "required_bindings": ["sources", "filters", "bowls"],
        "preconditions": [
            "at least one visible compact source and movable thin filter",
            "at least two visible downstream bowl demand sites",
            "a filter is selected before each directional translation",
        ],
        "operator": {
            "name": "select_translate_then_release_flow",
            "actions": ["MOUSE", "UP", "DOWN", "LEFT", "RIGHT", "SPACE"],
            "effect": "selection changes control; translation rewires cascade lanes; SPACE releases flow",
        },
        "invariants": [
            "source and bowl geometry remain fixed during filter placement",
            "filter dimensions and count remain fixed during translation",
            "each reached horizontal filter exposes two downstream split ports",
            "display rotation transforms controls but not normalized stream topology",
        ],
        "goal_relation": "route released flow to consume every bowl demand without overflow",
        "binding_recipe": {
            "source": "isolated compact emitter on the upstream boundary",
            "filter": "thin movable component upstream of the receptacle band",
            "bowl": "visible U-shaped downstream receptacle",
            "normalization": "infer logical cell scale, orient gravity downward, and trace split ports recursively",
        },
        "falsifiers": [
            "a required visible role disappears during placement",
            "the exact selected filter geometry differs after one action",
            "the rebound cascade graph differs from the predicted graph",
            "bowl residual demand or shortest remaining action count fails to decrease",
            "overflow becomes true when it was predicted false",
        ],
    }


def _normal_action(action: Any) -> dict[str, Any] | None:
    if isinstance(action, str):
        raw: Mapping[str, Any] = {"action": action}
    elif isinstance(action, Mapping):
        raw = action
    else:
        name = str(getattr(action, "name", "") or "")
        raw = {"action": name}
    name = str(raw.get("action", raw.get("action_name", "")) or "").upper()
    name = _ENGINE_TO_MODEL.get(name, name)
    if name not in _MODEL_TO_ENGINE:
        return None
    out: dict[str, Any] = {"action": name}
    if name == "MOUSE":
        try:
            out["row"] = int(raw.get("row", raw.get("y")))
            out["col"] = int(raw.get("col", raw.get("x")))
        except (TypeError, ValueError):
            return None
    return out


def _allowed_actions(valid_actions: Sequence[Any] | None) -> set[str] | None:
    if valid_actions is None:
        return None
    allowed: set[str] = set()
    for item in valid_actions:
        if isinstance(item, Mapping) and str(item.get("action", "")).upper() in {"MOUSE", "ACTION6"}:
            allowed.add("MOUSE")
            continue
        normalized = _normal_action(item)
        if normalized is not None:
            allowed.add(str(normalized["action"]))
            continue
        raw = str(item).upper()
        raw = raw.rsplit(".", 1)[-1]
        model = _ENGINE_TO_MODEL.get(raw, raw)
        if model in _MODEL_TO_ENGINE:
            allowed.add(model)
    return allowed


def _model_actions(
    state: Mapping[str, Any], valid_actions: Sequence[Any] | None
) -> list[dict[str, Any]] | None:
    allowed = _allowed_actions(valid_actions)
    actions = [
        dict(item)
        for item in state.get("operation_plan", ())
        if isinstance(item, Mapping)
    ]
    if not actions:
        return None
    if allowed is not None and any(str(item.get("action")) not in allowed for item in actions):
        return None
    return actions


def _display_to_grid(state: Mapping[str, Any], col: int, row: int) -> tuple[int, int]:
    frame_h, frame_w = [int(value) for value in list(state.get("frame_shape") or (64, 64))[:2]]
    rotation_k = int(state.get("rotation_k", 0) or 0) % 4
    if rotation_k == 1:
        ux, uy = frame_w - 1 - row, col
    elif rotation_k == 2:
        ux, uy = frame_w - 1 - col, frame_h - 1 - row
    elif rotation_k == 3:
        ux, uy = row, frame_h - 1 - col
    else:
        ux, uy = col, row
    grid_w, grid_h = [int(value) for value in list(state.get("grid_size") or (16, 16))[:2]]
    gx = max(0, min(grid_w - 1, int(ux * grid_w / frame_w)))
    gy = max(0, min(grid_h - 1, int(uy * grid_h / frame_h)))
    return gx, gy


def _predicted_state(state: Mapping[str, Any], action: Mapping[str, Any]) -> dict[str, Any] | None:
    predicted = copy.deepcopy(dict(state))
    filters = [dict(item) for item in predicted.get("filters", ()) if isinstance(item, Mapping)]
    name = str(action["action"])
    if name in {"UP", "DOWN", "LEFT", "RIGHT"}:
        selected = [index for index, item in enumerate(filters) if bool(item.get("selected", False))]
        delta = _internal_delta(name, int(predicted.get("rotation_k", 0) or 0))
        if len(selected) != 1 or delta is None:
            return None
        index = selected[0]
        filters[index]["x"] = int(filters[index]["x"]) + delta[0]
        filters[index]["y"] = int(filters[index]["y"]) + delta[1]
        positions = tuple((int(item["x"]), int(item["y"])) for item in filters)
        predicted["filters"] = filters
        if not _valid_positions(predicted, positions):
            return None
    elif name == "MOUSE":
        gx, gy = _display_to_grid(predicted, int(action["col"]), int(action["row"]))
        hit_index = None
        for index, item in enumerate(filters):
            inside = (
                int(item["x"]) <= gx < int(item["x"]) + int(item["w"])
                and int(item["y"]) <= gy < int(item["y"]) + int(item["h"])
            )
            item["selected"] = bool(inside)
            if inside:
                hit_index = index
        if hit_index is None:
            return None
        predicted["filters"] = filters
    else:
        return None
    for key in ("abstract_plan", "operation_plan", "target_filters"):
        predicted.pop(key, None)
    predicted.pop("operation_plan_len", None)
    return _attach_plan(predicted)


def _transition_from_state(
    state: Mapping[str, Any], action: Mapping[str, Any]
) -> dict[str, Any] | None:
    before = _binding_from_state(state)
    if action["action"] == "SPACE":
        potential = before["potential"]
        if int(potential["residual_demand"]) != 0 or bool(potential["overflow"]):
            return None
        return {
            "accepted": True,
            "mechanic_id": FAMILY_ID,
            "transition_kind": "release_flow",
            "action": dict(action),
            "before_graph": before["flow_graph"],
            "before_roles": before["roles"],
            "before_potential": potential,
            "requires_level_or_score_transition": True,
        }
    predicted_state = _predicted_state(state, action)
    if predicted_state is None:
        return None
    after = _binding_from_state(predicted_state)
    if int(after["potential"]["value"]) >= int(before["potential"]["value"]):
        return None
    return {
        "accepted": True,
        "mechanic_id": FAMILY_ID,
        "transition_kind": "placement",
        "action": dict(action),
        "expected_graph": after["flow_graph"],
        "expected_filters": after["roles"]["filters"],
        "expected_role_counts": {
            role: len(items) for role, items in after["roles"].items()
        },
        "potential_before": before["potential"],
        "potential_after": after["potential"],
    }


def _target_has_filter_cascade(state: Mapping[str, Any]) -> bool:
    targets = [item for item in state.get("target_filters", ()) if isinstance(item, Mapping)]
    filters = [item for item in state.get("filters", ()) if isinstance(item, Mapping)]
    if len(targets) != len(filters):
        return False
    solved = copy.deepcopy(dict(state))
    solved["filters"] = [
        {
            **dict(filters[index]),
            "x": int(target["x"]),
            "y": int(target["y"]),
        }
        for index, target in enumerate(targets)
    ]
    roles = _roles(solved)
    graph = _flow_graph(solved, roles)
    return any(
        str(edge.get("from", "")).startswith("filter:")
        and str(edge.get("to", "")).startswith("filter:")
        for edge in graph["edges"]
    )


def propose_plan(
    frame: Any, valid_actions: Sequence[Any] | None = None
) -> dict[str, Any] | None:
    """Return the visible-graph plan and certify its exact first transition."""

    state = _state(frame)
    if state is None:
        return None
    actions = _model_actions(state, valid_actions)
    if not actions:
        return None
    binding = _binding_from_state(state)
    first = _transition_from_state(state, actions[0])
    if first is None:
        return None
    certified = _target_has_filter_cascade(state)
    return {
        "mechanic_id": FAMILY_ID,
        "actions": actions,
        "certified": certified,
        "certificate_reason": (
            "visible multi-stage cascade graph with exact rebound first transition"
            if certified
            else "single-stage graph remains on the established exact lease"
        ),
        "transition_rule": "select_translate_then_release_flow",
        "requires_transition_verification": True,
        "progress_measure": {
            "metric": "remaining_filter_flow_actions",
            "before": len(actions),
            "expected_after": max(0, len(actions) - 1),
            "strict": True,
        },
        "first_prediction": first,
        "detail": {
            "planner": "visible_stream_graph_shortest_action_search",
            "private_plan_length": len(actions),
            "grid_size": list(state["grid_size"]),
            "rotation_k": int(state["rotation_k"]),
        },
        "private_verification": {
            "binding": binding,
            "first_transition": first,
        },
    }


def predict_transition(frame: Any, action: Any) -> dict[str, Any] | None:
    """Predict one exact graph, filter, and residual-demand transition."""

    state = _state(frame)
    normalized = _normal_action(action)
    if state is None or normalized is None:
        return None
    actions = _model_actions(state, None)
    if not actions or normalized != actions[0]:
        return None
    return _transition_from_state(state, normalized)


def verify_transition(expected: Any, current_frame: Any) -> bool:
    """Rebind the next frame and fail closed on any one-step mismatch."""

    if not isinstance(expected, Mapping) or expected.get("mechanic_id") != FAMILY_ID:
        return False
    if expected.get("transition_kind") == "release_flow":
        state = _perceive_state(current_frame)
        if state is None:
            return False
        roles = _roles(state)
        graph = _flow_graph(state, roles)
        return graph != expected.get("before_graph") and roles != expected.get("before_roles")
    if expected.get("transition_kind") != "placement":
        return False
    current = bind_current(current_frame)
    if current is None:
        return False
    counts = {role: len(items) for role, items in current["roles"].items()}
    if counts != expected.get("expected_role_counts"):
        return False
    if current["roles"]["filters"] != expected.get("expected_filters"):
        return False
    if current["flow_graph"] != expected.get("expected_graph"):
        return False
    actual_potential = current["potential"]
    if actual_potential != expected.get("potential_after"):
        return False
    try:
        return int(actual_potential["value"]) < int(expected["potential_before"]["value"])
    except (KeyError, TypeError, ValueError):
        return False
