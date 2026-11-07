"""Visible recorded-path replay with exact one-step temporal binding.

The strict-generic round-memory reader is used only to identify a structural
route on a fresh level frame.  This module then rebinds that route against
observed history, assigns ordinal identities to promoted paths, simulates one
actor/ghost tick at a time, and leases only the rebound prefix action.

The certified scope is deliberately limited to vector replay.  Frames whose
generic descriptor implies a portal or another non-vector transition remain
observable through the reader but receive no execution lease.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from typing import Any

import numpy as np

from arcgym.eca.round_memory_sensor import build_round_memory_protected_proposal

from .base import MechanicContext, as_frame, stable_data, stable_digest


FAMILY_ID = "recorded_path_replay"

_DIRECTIONS = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}
_SUBMIT = "ACTION5"
_ACTIONS = frozenset((*_DIRECTIONS, _SUBMIT))
_ALIASES = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
    "SPACE": _SUBMIT,
    "SUBMIT": _SUBMIT,
}
_CERTIFIED_GOAL_LINKS = frozenset(
    {
        "round_memory_switch_hold_route",
        "round_memory_dual_switch_post_score_route",
        "round_memory_multi_switch_post_score_route",
    }
)
_MAX_ORACLE_STEPS = 96
_ANCHOR_CACHE: dict[str, dict[str, Any] | None] = {}


def _json(value: Any) -> Any:
    return stable_data(value)


def _safe_metadata(value: Any) -> Any:
    normalized = stable_data(value)
    try:
        json.dumps(normalized, ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)
    return normalized


def _latest_frame(value: Any) -> np.ndarray | None:
    frame = as_frame(value)
    if frame is None or not np.issubdtype(frame.dtype, np.number):
        return None
    return frame


def _frame_digest(frame: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(int(item) for item in frame.shape)).encode("ascii"))
    digest.update(str(frame.dtype).encode("ascii"))
    digest.update(np.ascontiguousarray(frame).tobytes())
    return digest.hexdigest()


def _action_name(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("action", value.get("action_name", ""))
    if hasattr(value, "name"):
        value = value.name
    name = str(value or "").split(".")[-1].strip().upper()
    return _ALIASES.get(name, name) if name else ""


def _allowed_actions(value: Iterable[Any] | None) -> set[str]:
    if value is None:
        return set(_ACTIONS)
    rows = list(value)
    if not rows:
        return set(_ACTIONS)
    return {name for row in rows if (name := _action_name(row)) in _ACTIONS}


def _position(value: Sequence[int] | None) -> list[int] | None:
    if value is None or len(value) < 2:
        return None
    return [int(value[0]), int(value[1])]


def _positions(values: Iterable[Sequence[int]]) -> list[list[int]]:
    return [
        [int(value[0]), int(value[1])]
        for value in values
    ]


def _connected_components(frame: np.ndarray) -> list[dict[str, Any]]:
    height, width = frame.shape
    seen = np.zeros(frame.shape, dtype=bool)
    components: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x]:
                continue
            value = int(frame[y, x])
            seen[y, x] = True
            queue = deque([(x, y)])
            cells: list[tuple[int, int]] = []
            while queue:
                current_x, current_y = queue.popleft()
                cells.append((current_x, current_y))
                for delta_x, delta_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_x = current_x + delta_x
                    next_y = current_y + delta_y
                    if (
                        0 <= next_x < width
                        and 0 <= next_y < height
                        and not seen[next_y, next_x]
                        and int(frame[next_y, next_x]) == value
                    ):
                        seen[next_y, next_x] = True
                        queue.append((next_x, next_y))
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            components.append(
                {
                    "value": value,
                    "cells": tuple(cells),
                    "area": len(cells),
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                    "centroid": (
                        sum(xs) / max(1, len(xs)),
                        sum(ys) / max(1, len(ys)),
                    ),
                }
            )
    return components


def _role_component(
    frame: np.ndarray,
    *,
    value: int,
    anchor: tuple[int, int],
    stride: int,
) -> dict[str, Any] | None:
    target = (anchor[0] + stride / 2.0, anchor[1] + stride / 2.0)
    candidates: list[tuple[tuple[float, int], dict[str, Any]]] = []
    for component in _connected_components(frame):
        if int(component["value"]) != int(value):
            continue
        x0, y0, x1, y1 = component["bbox"]
        component_width = x1 - x0 + 1
        component_height = y1 - y0 + 1
        if (
            int(component["area"]) < max(3, stride)
            or component_width > stride * 2
            or component_height > stride * 2
        ):
            continue
        centroid_x, centroid_y = component["centroid"]
        distance = (centroid_x - target[0]) ** 2 + (centroid_y - target[1]) ** 2
        contains = int(
            x0 - stride <= anchor[0] <= x1
            and y0 - stride <= anchor[1] <= y1
        )
        candidates.append(((-float(contains), float(distance)), component))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _slot_block(
    frame: np.ndarray,
    position: tuple[int, int],
) -> dict[str, Any] | None:
    x, y = position
    if x < 0 or y < 0 or x + 3 > frame.shape[1] or y + 3 > frame.shape[0]:
        return None
    block = frame[y : y + 3, x : x + 3]
    border = np.concatenate(
        (block[0, :], block[-1, :], block[1:-1, 0], block[1:-1, -1])
    )
    values = np.unique(border)
    if values.size != 1:
        return None
    border_value = int(values[0])
    center_value = int(block[1, 1])
    return {
        "position": position,
        "border_value": border_value,
        "center_value": center_value,
        "ring": center_value != border_value,
    }


def _find_marker_slots(
    frame: np.ndarray,
    *,
    actor_value: int,
    slot_count: int,
    stride: int,
    role_positions: Sequence[tuple[int, int]],
) -> dict[str, Any] | None:
    positive_y = [position[1] for position in role_positions if position[1] > 0]
    search_height = min(
        frame.shape[0] - 2,
        min(positive_y) if positive_y else max(4, frame.shape[0] // 4),
    )
    search_width = min(frame.shape[1] - 2, max(12, frame.shape[1] // 3))
    candidates: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for y in range(max(0, search_height)):
        for x in range(max(0, search_width)):
            first = _slot_block(frame, (x, y))
            if (
                first is None
                or not first["ring"]
                or int(first["border_value"]) != int(actor_value)
            ):
                continue
            for spacing in range(3, max(4, stride + 3)):
                positions = [(x + index * spacing, y) for index in range(slot_count)]
                blocks = [_slot_block(frame, position) for position in positions]
                if any(block is None for block in blocks):
                    continue
                typed = [block for block in blocks if block is not None]
                if any(block["ring"] for block in typed[1:]):
                    continue
                empty_values = {int(block["border_value"]) for block in typed[1:]}
                if len(empty_values) != 1:
                    continue
                empty_value = next(iter(empty_values))
                if empty_value == int(actor_value):
                    continue
                candidates.append(
                    (
                        (y, x, spacing),
                        {
                            "positions": positions,
                            "empty_value": empty_value,
                            "center_value": int(first["center_value"]),
                            "spacing": spacing,
                        },
                    )
                )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _line_runs(mask: np.ndarray, minimum: int) -> list[dict[str, Any]]:
    height, width = mask.shape
    rows: list[dict[str, Any]] = []
    for y in range(height):
        x = 0
        while x < width:
            if not mask[y, x]:
                x += 1
                continue
            end = x
            while end + 1 < width and mask[y, end + 1]:
                end += 1
            if end - x + 1 >= minimum:
                rows.append(
                    {
                        "orientation": "horizontal",
                        "from": [x, y],
                        "to": [end, y],
                        "length": end - x + 1,
                    }
                )
            x = end + 1
    for x in range(width):
        y = 0
        while y < height:
            if not mask[y, x]:
                y += 1
                continue
            end = y
            while end + 1 < height and mask[end + 1, x]:
                end += 1
            if end - y + 1 >= minimum:
                rows.append(
                    {
                        "orientation": "vertical",
                        "from": [x, y],
                        "to": [x, end],
                        "length": end - y + 1,
                    }
                )
            y = end + 1
    rows.sort(
        key=lambda row: (
            -int(row["length"]),
            str(row["orientation"]),
            row["from"],
        )
    )
    return rows[:16]


def _mechanism_networks(
    frame: np.ndarray,
    *,
    stride: int,
    excluded_values: set[int],
) -> list[dict[str, Any]]:
    """Find sparse elongated networks without assigning palette semantics."""

    networks: list[dict[str, Any]] = []
    maximum_area = max(stride * 4, int(frame.size) // 8)
    for component in _connected_components(frame):
        value = int(component["value"])
        if value in excluded_values:
            continue
        x0, y0, x1, y1 = component["bbox"]
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        area = int(component["area"])
        if (
            area < stride
            or area > maximum_area
            or max(width, height) < stride * 2
            or float(area) / float(max(1, width * height)) > 0.8
        ):
            continue
        mask = np.zeros(frame.shape, dtype=bool)
        for x, y in component["cells"]:
            mask[int(y), int(x)] = True
        segments = _line_runs(mask, max(2, stride))
        if not segments:
            continue
        networks.append(
            {
                "value": value,
                "cells": tuple(
                    sorted((int(x), int(y)) for x, y in component["cells"])
                ),
                "segments": tuple(segments),
            }
        )
    networks.sort(
        key=lambda row: (
            -len(row["cells"]),
            stable_digest(row["cells"]),
        )
    )
    return networks


def _switches(round_memory: Mapping[str, Any]) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for key in (
        "primary_switch_anchor",
        "secondary_switch_anchor",
        "tertiary_switch_anchor",
        "switch_anchor",
    ):
        value = round_memory.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            continue
        if len(value) < 2:
            continue
        position = (int(value[0]), int(value[1]))
        if position not in rows:
            rows.append(position)
    return rows


def _oracle_model(frame: np.ndarray) -> dict[str, Any] | None:
    cache_key = _frame_digest(frame)
    if cache_key in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[cache_key]

    packet: Mapping[str, Any] | None = None
    for post_score in (True, False):
        candidate = build_round_memory_protected_proposal(
            frame,
            max_plan_steps=_MAX_ORACLE_STEPS,
            role_discovery=True,
            post_score_replan=post_score,
        )
        if bool(candidate.get("accepted")):
            packet = candidate
            break
    if packet is None:
        return _remember_model(cache_key, None)

    goal_link = str(packet.get("goal_link_type", "") or "")
    if goal_link not in _CERTIFIED_GOAL_LINKS:
        return _remember_model(cache_key, None)
    raw_plan = list(packet.get("plan") or [])
    plan = tuple(_action_name(step) for step in raw_plan)
    if not plan or any(action not in _ACTIONS for action in plan):
        return _remember_model(cache_key, None)

    round_memory = packet.get("round_memory")
    if not isinstance(round_memory, Mapping):
        return _remember_model(cache_key, None)
    role_discovery = round_memory.get("role_discovery")
    if not isinstance(role_discovery, Mapping):
        return _remember_model(cache_key, None)
    try:
        start = tuple(int(value) for value in round_memory["player_anchor"][:2])
        goal = tuple(int(value) for value in round_memory["goal_anchor"][:2])
        stride = int(round_memory["stride"])
        actor_value = int(role_discovery["actor_goal_color"])
        door_value = int(role_discovery["door_switch_color"])
        background_value = int(role_discovery["background_color"])
    except (KeyError, TypeError, ValueError):
        return _remember_model(cache_key, None)
    if len(start) != 2 or len(goal) != 2 or stride <= 0:
        return _remember_model(cache_key, None)

    switch_positions = _switches(round_memory)
    if not switch_positions:
        return _remember_model(cache_key, None)
    submit_count = sum(action == _SUBMIT for action in plan)
    if submit_count <= 0:
        return _remember_model(cache_key, None)
    slot_count = submit_count + 1
    marker = _find_marker_slots(
        frame,
        actor_value=actor_value,
        slot_count=slot_count,
        stride=stride,
        role_positions=[start, goal, *switch_positions],
    )
    actor_component = _role_component(
        frame,
        value=actor_value,
        anchor=start,
        stride=stride,
    )
    goal_component = _role_component(
        frame,
        value=actor_value,
        anchor=goal,
        stride=stride,
    )
    if marker is None or actor_component is None or goal_component is None:
        return _remember_model(cache_key, None)

    body_offsets = tuple(
        sorted(
            (int(x) - start[0], int(y) - start[1])
            for x, y in actor_component["cells"]
        )
    )
    goal_offsets = tuple(
        sorted(
            (int(x) - goal[0], int(y) - goal[1])
            for x, y in goal_component["cells"]
        )
    )
    if not body_offsets or body_offsets == goal_offsets:
        return _remember_model(cache_key, None)

    door_cells = tuple(
        (int(x), int(y))
        for y, x in np.argwhere(frame == int(door_value)).tolist()
    )
    if not door_cells:
        return _remember_model(cache_key, None)
    networks = _mechanism_networks(
        frame,
        stride=stride,
        excluded_values={
            actor_value,
            background_value,
            int(marker["empty_value"]),
            int(marker["center_value"]),
        },
    )
    if not networks:
        return _remember_model(cache_key, None)
    mechanism_cells = tuple(
        sorted(
            {
                tuple(cell)
                for network in networks
                for cell in network["cells"]
            }
        )
    )
    gate_segments = tuple(
        {
            "network_id": f"network_{network_index}",
            **segment,
        }
        for network_index, network in enumerate(networks)
        for segment in network["segments"]
    )

    model = {
        "anchor_frame": frame.copy(),
        "anchor_digest": cache_key,
        "frame_shape": tuple(int(value) for value in frame.shape),
        "goal_link_type": goal_link,
        "plan": plan,
        "start": start,
        "goal": goal,
        "stride": stride,
        "switches": tuple(switch_positions),
        "round_capacity": slot_count,
        "actor_value": actor_value,
        "door_value": door_value,
        "background_value": background_value,
        "empty_marker_value": int(marker["empty_value"]),
        "marker_center_value": int(marker["center_value"]),
        "marker_slots": tuple(tuple(position) for position in marker["positions"]),
        "body_offsets": body_offsets,
        "goal_offsets": goal_offsets,
        "door_cells": door_cells,
        "mechanism_cells": mechanism_cells,
        "mechanism_values": tuple(int(network["value"]) for network in networks),
        "gate_segments": gate_segments,
    }
    return _remember_model(cache_key, model)


def _remember_model(
    key: str,
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if len(_ANCHOR_CACHE) >= 16:
        _ANCHOR_CACHE.clear()
    _ANCHOR_CACHE[key] = value
    return value


def _initial_state(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prefix_index": 0,
        "actor_position": tuple(model["start"]),
        "recorded_paths": [],
        "current_path": [],
        "ghost_positions": [],
        "round_index": 0,
        "replay_tick": 0,
    }


def _advance_position(
    position: tuple[int, int],
    action: str,
    stride: int,
) -> tuple[int, int]:
    delta_x, delta_y = _DIRECTIONS[action]
    return (
        int(position[0]) + delta_x * stride,
        int(position[1]) + delta_y * stride,
    )


def _simulate_one(
    model: Mapping[str, Any],
    state: Mapping[str, Any],
    action: str,
) -> dict[str, Any] | None:
    if action not in _ACTIONS:
        return None
    next_state = {
        "prefix_index": int(state["prefix_index"]) + 1,
        "actor_position": tuple(state["actor_position"]),
        "recorded_paths": [list(path) for path in state["recorded_paths"]],
        "current_path": list(state["current_path"]),
        "ghost_positions": [tuple(position) for position in state["ghost_positions"]],
        "round_index": int(state["round_index"]),
        "replay_tick": int(state["replay_tick"]),
    }
    if action == _SUBMIT:
        if (
            not next_state["current_path"]
            or len(next_state["recorded_paths"])
            >= int(model["round_capacity"]) - 1
        ):
            return None
        next_state["recorded_paths"].append(list(next_state["current_path"]))
        next_state["current_path"] = []
        next_state["round_index"] += 1
        next_state["replay_tick"] = 0
        next_state["actor_position"] = tuple(model["start"])
        next_state["ghost_positions"] = [
            tuple(model["start"])
            for _path in next_state["recorded_paths"]
        ]
        return next_state

    stride = int(model["stride"])
    tick = int(next_state["replay_tick"])
    next_state["actor_position"] = _advance_position(
        tuple(next_state["actor_position"]),
        action,
        stride,
    )
    next_state["current_path"].append(action)
    advanced_ghosts: list[tuple[int, int]] = []
    for index, position in enumerate(next_state["ghost_positions"]):
        path = next_state["recorded_paths"][index]
        if tick < len(path):
            position = _advance_position(tuple(position), path[tick], stride)
        advanced_ghosts.append(tuple(position))
    next_state["ghost_positions"] = advanced_ghosts
    next_state["replay_tick"] += 1
    return next_state


def _simulate_prefix(
    model: Mapping[str, Any],
    actions: Sequence[str],
) -> dict[str, Any] | None:
    state = _initial_state(model)
    for action in actions:
        next_state = _simulate_one(model, state, action)
        if next_state is None:
            return None
        state = next_state
    return state


def _context_records(
    current: np.ndarray,
    context: MechanicContext | Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if isinstance(context, MechanicContext):
        rows = context.history
        level = int(context.level)
        previous_action = _action_name(context.previous_action)
    elif isinstance(context, Mapping):
        raw_rows = context.get("history", ())
        rows = tuple(
            row for row in raw_rows if isinstance(row, Mapping)
        ) if isinstance(raw_rows, Sequence) else ()
        level = int(context.get("level", 0) or 0)
        previous_action = _action_name(context.get("previous_action"))
    else:
        rows = ()
        level = 0
        previous_action = ""

    records: list[dict[str, Any]] = []
    for row in rows[-96:]:
        row_level = int(row.get("level", level) or level)
        if level and row_level and row_level != level:
            continue
        row_frame = _latest_frame(row.get("frame"))
        if row_frame is None or row_frame.shape != current.shape:
            continue
        records.append(
            {
                "frame": row_frame,
                "action": _action_name(row.get("action")),
            }
        )
    current_index = next(
        (
            index
            for index in range(len(records) - 1, -1, -1)
            if np.array_equal(records[index]["frame"], current)
        ),
        -1,
    )
    if current_index >= 0:
        return records[: current_index + 1]
    records.append({"frame": current, "action": previous_action})
    return records


def _slot_observation(
    frame: np.ndarray,
    model: Mapping[str, Any],
    round_index: int,
) -> dict[str, Any] | None:
    slots = [
        _slot_block(frame, tuple(position))
        for position in model["marker_slots"]
    ]
    if any(slot is None for slot in slots):
        return None
    typed = [slot for slot in slots if slot is not None]
    ghost_values: list[int] = []
    for index, slot in enumerate(typed):
        border = int(slot["border_value"])
        center = int(slot["center_value"])
        if index < round_index:
            if (
                not slot["ring"]
                or center != int(model["marker_center_value"])
                or border in {
                    int(model["actor_value"]),
                    int(model["empty_marker_value"]),
                }
            ):
                return None
            ghost_values.append(border)
        elif index == round_index:
            if (
                not slot["ring"]
                or border != int(model["actor_value"])
                or center != int(model["marker_center_value"])
            ):
                return None
        elif (
            slot["ring"]
            or border != int(model["empty_marker_value"])
            or center != int(model["empty_marker_value"])
        ):
            return None
    return {
        "round_index": int(round_index),
        "ghost_values": ghost_values,
        "slot_roles": [
            "ghost" if index < round_index else "actor" if index == round_index else "empty"
            for index in range(len(typed))
        ],
    }


def _body_cells(
    position: tuple[int, int],
    offsets: Sequence[Sequence[int]],
) -> set[tuple[int, int]]:
    return {
        (int(position[0]) + int(offset[0]), int(position[1]) + int(offset[1]))
        for offset in offsets
    }


def _body_has_value(
    frame: np.ndarray,
    position: tuple[int, int],
    offsets: Sequence[Sequence[int]],
    value: int,
) -> bool:
    cells = _body_cells(position, offsets)
    if not cells:
        return False
    height, width = frame.shape
    return all(
        0 <= x < width
        and 0 <= y < height
        and int(frame[y, x]) == int(value)
        for x, y in cells
    )


def _entity_positions(state: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    positions = {"actor": tuple(state["actor_position"])}
    positions.update(
        {
            f"ghost_{index}": tuple(position)
            for index, position in enumerate(state["ghost_positions"])
        }
    )
    return positions


def _switch_occupancy(
    model: Mapping[str, Any],
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entities = _entity_positions(state)
    return [
        {
            "switch_id": f"switch_{index}",
            "position": _position(switch),
            "occupants": sorted(
                entity_id
                for entity_id, position in entities.items()
                if tuple(position) == tuple(switch)
            ),
        }
        for index, switch in enumerate(model["switches"])
    ]


def _state_projection(
    model: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    tick = int(state["replay_tick"])
    path_rows = [
        {
            "ghost_id": f"ghost_{index}",
            "action_count": len(path),
            "actions": list(path),
            "path_digest": stable_digest(list(path)),
        }
        for index, path in enumerate(state["recorded_paths"])
    ]
    ghosts = [
        {
            "ghost_id": f"ghost_{index}",
            "position": _position(position),
            "replay_cursor": min(tick, len(state["recorded_paths"][index])),
            "path_length": len(state["recorded_paths"][index]),
        }
        for index, position in enumerate(state["ghost_positions"])
    ]
    switches = _switch_occupancy(model, state)
    engaged = [row["switch_id"] for row in switches if row["occupants"]]
    return {
        "actor": {
            "actor_id": "actor",
            "position": _position(state["actor_position"]),
        },
        "ghosts": ghosts,
        "round": {
            "index": int(state["round_index"]),
            "capacity": int(model["round_capacity"]),
            "replay_tick": tick,
            "current_path": list(state["current_path"]),
            "current_path_length": len(state["current_path"]),
            "recorded_paths": path_rows,
        },
        "switches": switches,
        "gates": {
            "mode": "engaged" if engaged else "resting",
            "engaged_switch_ids": engaged,
        },
        "goal": {
            "position": _position(model["goal"]),
            "actor_at_goal": tuple(state["actor_position"]) == tuple(model["goal"]),
        },
    }


def _expanded_door_cells(
    shape: tuple[int, int],
    door_cells: Sequence[Sequence[int]],
    radius: int,
) -> set[tuple[int, int]]:
    height, width = shape
    rows: set[tuple[int, int]] = set()
    for cell in door_cells:
        x, y = int(cell[0]), int(cell[1])
        for next_y in range(max(0, y - radius), min(height, y + radius + 1)):
            for next_x in range(max(0, x - radius), min(width, x + radius + 1)):
                rows.add((next_x, next_y))
    return rows


def _dynamic_cells(
    model: Mapping[str, Any],
    positions: Iterable[tuple[int, int]],
) -> set[tuple[int, int]]:
    height, width = tuple(int(value) for value in model["frame_shape"])
    stride = int(model["stride"])
    marker_bottom = max(int(position[1]) + 3 for position in model["marker_slots"])
    top_cutoff = min(height, marker_bottom + max(2, stride // 2))
    bottom_cutoff = max(0, height - max(1, stride // 2))
    dynamic = {(x, y) for y in range(top_cutoff) for x in range(width)}
    dynamic.update(
        (x, y)
        for y in range(bottom_cutoff, height)
        for x in range(width)
    )
    dynamic.update(
        _expanded_door_cells(
            (height, width),
            model.get("mechanism_cells", model["door_cells"]),
            radius=max(1, stride * 2),
        )
    )
    for position in positions:
        dynamic.update(_body_cells(position, model["body_offsets"]))
    return dynamic


def _masked_digest(
    frame: np.ndarray,
    dynamic: set[tuple[int, int]],
) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in frame.shape)).encode("ascii"))
    digest.update(str(frame.dtype).encode("ascii"))
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            if (x, y) in dynamic:
                continue
            digest.update(f"{x},{y}:{int(frame[y, x])};".encode("ascii"))
    return digest.hexdigest()


def _gate_projection(
    frame: np.ndarray,
    model: Mapping[str, Any],
    ignored_positions: Iterable[tuple[int, int]],
) -> dict[str, Any]:
    zone = _expanded_door_cells(
        tuple(int(value) for value in model["frame_shape"]),
        model["door_cells"],
        radius=max(1, int(model["stride"]) * 2),
    )
    ignored: set[tuple[int, int]] = set()
    for position in ignored_positions:
        ignored.update(_body_cells(position, model["body_offsets"]))
    cells = sorted(
        (x, y)
        for x, y in zone - ignored
        if int(frame[y, x]) == int(model["door_value"])
    )
    return {
        "cell_count": len(cells),
        "digest": stable_digest(cells),
    }


def _matches_state(
    frame: np.ndarray,
    model: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    slot_state = _slot_observation(frame, model, int(state["round_index"]))
    reasons: list[str] = []
    if slot_state is None:
        reasons.append("round_slots_do_not_match_temporal_state")
        ghost_values: list[int] = []
    else:
        ghost_values = list(slot_state["ghost_values"])

    actor_position = tuple(state["actor_position"])
    if not _body_has_value(
        frame,
        actor_position,
        model["body_offsets"],
        int(model["actor_value"]),
    ):
        reasons.append("actor_not_at_predicted_position")

    positions = _entity_positions(state)
    grouped: dict[tuple[int, int], list[str]] = {}
    for entity_id, position in positions.items():
        grouped.setdefault(tuple(position), []).append(entity_id)
    for position, entity_ids in grouped.items():
        if "actor" in entity_ids:
            continue
        indices = [int(entity_id.split("_")[-1]) for entity_id in entity_ids]
        palettes = {
            ghost_values[index]
            for index in indices
            if index < len(ghost_values)
        }
        if not palettes or not any(
            _body_has_value(frame, position, model["body_offsets"], palette)
            for palette in palettes
        ):
            reasons.append("ghost_not_at_predicted_position")
            break

    all_positions = [tuple(model["start"]), *positions.values()]
    dynamic = _dynamic_cells(model, all_positions)
    anchor_digest = _masked_digest(model["anchor_frame"], dynamic)
    current_digest = _masked_digest(frame, dynamic)
    if anchor_digest != current_digest:
        reasons.append("static_topology_changed")
    return not reasons, {
        "reasons": reasons,
        "slot_state": slot_state,
        "static_topology_digest": current_digest,
    }


def _resolve(
    frame: Any,
    context: MechanicContext | Mapping[str, Any] | None,
) -> dict[str, Any]:
    current = _latest_frame(frame)
    if current is None:
        return {"accepted": False, "reason": "invalid_numeric_frame"}
    records = _context_records(current, context)
    if not records:
        return {"accepted": False, "reason": "missing_current_level_history"}
    anchor = records[0]["frame"]
    model = _oracle_model(anchor)
    if model is None:
        return {
            "accepted": False,
            "reason": "structural_round_memory_oracle_not_certified",
        }
    executed = [str(row["action"]) for row in records[1:]]
    if any(action not in _ACTIONS for action in executed):
        return {"accepted": False, "reason": "history_contains_unbound_action"}
    plan = list(model["plan"])
    if len(executed) >= len(plan) or executed != plan[: len(executed)]:
        return {"accepted": False, "reason": "history_is_not_oracle_prefix"}
    state = _simulate_prefix(model, executed)
    if state is None:
        return {"accepted": False, "reason": "history_simulation_failed"}
    matches, evidence = _matches_state(current, model, state)
    if not matches:
        return {
            "accepted": False,
            "reason": "current_frame_rejects_temporal_state",
            "evidence": evidence,
        }
    return {
        "accepted": True,
        "frame": current,
        "model": model,
        "state": state,
        "executed": executed,
        "remaining_plan": plan[len(executed) :],
        "evidence": evidence,
    }


def _unbound_binding(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "tufa.recorded_replay.binding.v1",
        "mechanic_id": FAMILY_ID,
        "family_id": FAMILY_ID,
        "status": "unbound",
        "supported": False,
        "reason": str(reason),
        "role_bindings": [],
    }


def bind_current(
    frame: Any,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind actor, ordinal ghosts, path logs, switches, gates, and goal."""

    resolved = _resolve(frame, context)
    if not resolved.get("accepted"):
        return _unbound_binding(str(resolved.get("reason") or "unbound"))
    model = resolved["model"]
    state = resolved["state"]
    projection = _state_projection(model, state)
    slots = resolved["evidence"]["slot_state"]
    gate_segments = [
        {
            "network_id": f"network_{network_index}",
            **segment,
        }
        for network_index, value in enumerate(model["mechanism_values"])
        for segment in _line_runs(
            resolved["frame"] == int(value),
            max(2, int(model["stride"])),
        )
    ]

    actor_position = tuple(state["actor_position"])
    ghost_positions = [tuple(position) for position in state["ghost_positions"]]
    roles: list[dict[str, Any]] = [
        {
            "role": "actor",
            "identity": "actor",
            "position": _position(actor_position),
        },
        {
            "role": "goal",
            "position": _position(model["goal"]),
        },
    ]
    roles.extend(
        {
            "role": "ghost",
            "identity": f"ghost_{index}",
            "position": _position(position),
            "occluded": position == actor_position
            or ghost_positions.count(position) > 1,
        }
        for index, position in enumerate(ghost_positions)
    )
    roles.extend(
        {
            "role": "switch",
            "identity": row["switch_id"],
            "position": row["position"],
            "occupants": row["occupants"],
        }
        for row in projection["switches"]
    )
    roles.extend(
        {
            "role": "gate_segment",
            "identity": f"gate_segment_{index}",
            **segment,
        }
        for index, segment in enumerate(gate_segments)
    )
    return _json(
        {
            "schema_version": "tufa.recorded_replay.binding.v1",
            "mechanic_id": FAMILY_ID,
            "family_id": FAMILY_ID,
            "status": "bound",
            "supported": True,
            "reason": "history_rebound_vector_replay_state",
            "frame_shape": list(model["frame_shape"]),
            "stride": int(model["stride"]),
            "round_state": projection["round"],
            "actor": projection["actor"],
            "ghosts": projection["ghosts"],
            "switches": projection["switches"],
            "gates": {
                **projection["gates"],
                "segments": gate_segments,
            },
            "goal": projection["goal"],
            "round_slots": {
                "positions": _positions(model["marker_slots"]),
                "roles": list(slots["slot_roles"]),
                "ordinal_identity_source": "slot_order_plus_recorded_path",
            },
            "body_mask": {
                "pixel_count": len(model["body_offsets"]),
                "relative_digest": stable_digest(model["body_offsets"]),
            },
            "static_topology_digest": resolved["evidence"][
                "static_topology_digest"
            ],
            "remaining_private_plan_length": len(resolved["remaining_plan"]),
            "suffix_exposed": False,
            "role_bindings": roles,
        }
    )


def compile_schema(
    frame: Any,
    level: Any,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile the transferable recorded-replay transition schema."""

    binding = bind_current(frame, context=context)
    return _json(
        {
            "schema_id": "tufa.recorded_path_replay.v1",
            "schema_version": "tufa.recorded_replay.schema.v1",
            "mechanic_id": FAMILY_ID,
            "family_id": FAMILY_ID,
            "level": _safe_metadata(level),
            "supported": bool(binding.get("supported")),
            "roles": [
                "actor",
                "round_slot",
                "recorded_path",
                "ordinal_ghost",
                "switch",
                "gate",
                "goal",
            ],
            "required_bindings": [
                "actor_identity",
                "round_slot_order",
                "recorded_path_log",
                "switch_gate_network",
                "goal",
            ],
            "state_variables": [
                "actor_position",
                "round_index",
                "current_path",
                "recorded_paths",
                "ghost_positions",
                "replay_tick",
                "switch_occupancy",
                "gate_mode",
            ],
            "operator": {
                "name": "record_or_replay_one_tick",
                "actions": sorted(_ACTIONS),
                "transition": (
                    "movement appends to the actor path while every ordinal ghost "
                    "executes its recorded action at the same replay tick; submit "
                    "promotes the current path and rewinds every identity"
                ),
            },
            "goal": {
                "name": "switch_supported_goal_entry",
                "condition": "actor reaches the visible goal under replayed switch occupancy",
            },
            "certification_scope": {
                "transition_class": "vector_replay_only",
                "unknown_portal_or_nonvector_transition": "fail_closed",
                "lease": "one_action",
                "future_suffix": "private",
            },
            "invariants": [
                "ordinal ghost identity is path-slot identity, not palette identity",
                "recorded paths are immutable after promotion",
                "all identities rewind to the same visible start on submit",
                "static topology outside sprites, counters, and gate motion is unchanged",
            ],
            "falsifiers": [
                "history is not the rebound oracle-plan prefix",
                "actor or any visible ghost misses its exact predicted endpoint",
                "round slots do not encode the predicted promoted-path count",
                "static topology changes outside the bounded dynamic mask",
                "gate projection disagrees with the predicted occupancy transition",
            ],
            "current_binding": binding,
        }
    )


def _blocked_plan(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "tufa.recorded_replay.plan.v1",
        "family_id": FAMILY_ID,
        "status": "blocked",
        "actions": [],
        "action_count": 0,
        "certified": False,
        "certificate_reason": str(reason),
        "suffix_exposed": False,
        "requires_transition_verification": True,
    }


def _prediction_rejection(
    frame: Any,
    action: Any,
    reason: str,
) -> dict[str, Any]:
    latest = _latest_frame(frame)
    return {
        "schema_version": "tufa.recorded_replay.transition.v1",
        "family_id": FAMILY_ID,
        "accepted": False,
        "lease_safe": False,
        "reason": str(reason),
        "action": {"action": _action_name(action)},
        "before_frame_digest": _frame_digest(latest) if latest is not None else "",
    }


def _verification_model(
    model: Mapping[str, Any],
    state: Mapping[str, Any],
    next_state: Mapping[str, Any],
    slot_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "frame_shape": list(model["frame_shape"]),
        "stride": int(model["stride"]),
        "actor_value": int(model["actor_value"]),
        "door_value": int(model["door_value"]),
        "empty_marker_value": int(model["empty_marker_value"]),
        "marker_center_value": int(model["marker_center_value"]),
        "marker_slots": _positions(model["marker_slots"]),
        "body_offsets": _positions(model["body_offsets"]),
        "door_cells": _positions(model["door_cells"]),
        "mechanism_cells": _positions(model["mechanism_cells"]),
        "known_ghost_values": list(slot_state.get("ghost_values") or []),
        "before_positions": {
            key: _position(value)
            for key, value in _entity_positions(state).items()
        },
        "after_positions": {
            key: _position(value)
            for key, value in _entity_positions(next_state).items()
        },
    }


def predict_transition(
    frame: Any,
    action: Any,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict the exact rebound actor/ghost tick or submit transition."""

    resolved = _resolve(frame, context)
    if not resolved.get("accepted"):
        return _prediction_rejection(
            frame,
            action,
            str(resolved.get("reason") or "temporal_binding_unavailable"),
        )
    action_name = _action_name(action)
    remaining = list(resolved["remaining_plan"])
    if action_name not in _ACTIONS:
        return _prediction_rejection(frame, action, "unbound_action")
    if not remaining or action_name != remaining[0]:
        return _prediction_rejection(
            frame,
            action,
            "action_is_not_rebound_plan_prefix",
        )

    model = resolved["model"]
    state = resolved["state"]
    next_state = _simulate_one(model, state, action_name)
    if next_state is None:
        return _prediction_rejection(frame, action, "transition_simulation_rejected")
    before_projection = _state_projection(model, state)
    after_projection = _state_projection(model, next_state)
    slot_state = resolved["evidence"]["slot_state"]
    verification_model = _verification_model(
        model,
        state,
        next_state,
        slot_state,
    )
    ignored_positions = {
        *(_entity_positions(state).values()),
        *(_entity_positions(next_state).values()),
    }
    dynamic = _dynamic_cells(model, ignored_positions)
    static_digest = _masked_digest(resolved["frame"], dynamic)
    gate_before = _gate_projection(
        resolved["frame"],
        model,
        ignored_positions,
    )
    occupancy_changed = (
        before_projection["gates"]["engaged_switch_ids"]
        != after_projection["gates"]["engaged_switch_ids"]
    )
    terminal = bool(after_projection["goal"]["actor_at_goal"])
    promoted = None
    if action_name == _SUBMIT:
        path = next_state["recorded_paths"][-1]
        promoted = {
            "ghost_id": f"ghost_{len(next_state['recorded_paths']) - 1}",
            "action_count": len(path),
            "path_digest": stable_digest(path),
        }
    expected_digest = stable_digest(after_projection)
    return _json(
        {
            "schema_version": "tufa.recorded_replay.transition.v1",
            "family_id": FAMILY_ID,
            "accepted": True,
            "lease_safe": True,
            "reason": "exact_rebound_temporal_successor",
            "action": {"action": action_name},
            "transition_kind": (
                "round_submit" if action_name == _SUBMIT else "simultaneous_replay_step"
            ),
            "before_frame_digest": _frame_digest(resolved["frame"]),
            "before": before_projection,
            "expected_after": after_projection,
            "expected_successor_digest": expected_digest,
            "promoted_path": promoted,
            "terminal_goal_entry": terminal,
            "remaining_plan_before": len(remaining),
            "expected_remaining_plan_after": max(0, len(remaining) - 1),
            "static_topology_digest": static_digest,
            "gate_projection_before": gate_before,
            "gate_occupancy_changed": occupancy_changed,
            "verification_model": verification_model,
            "verification_rule": (
                "score closure owns terminal level replacement"
                if terminal
                else "exact ordinal entity endpoints, round slots, gate mode, and static topology"
            ),
        }
    )


def propose_plan(
    frame: Any,
    valid_actions: Iterable[Any] | None = None,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return only one certified action from the private rebound route."""

    resolved = _resolve(frame, context)
    if not resolved.get("accepted"):
        return _blocked_plan(str(resolved.get("reason") or "temporal_binding_unavailable"))
    remaining = list(resolved["remaining_plan"])
    if not remaining:
        return _blocked_plan("private_route_exhausted")
    action = remaining[0]
    if action not in _allowed_actions(valid_actions):
        return _blocked_plan("rebound_prefix_action_not_available")
    prediction = predict_transition(
        frame,
        {"action": action},
        context=context,
    )
    if not prediction.get("accepted"):
        return _blocked_plan(str(prediction.get("reason") or "prediction_rejected"))
    before_round = prediction["before"]["round"]
    after_round = prediction["expected_after"]["round"]
    return _json(
        {
            "schema_version": "tufa.recorded_replay.plan.v1",
            "family_id": FAMILY_ID,
            "status": "proposed",
            "actions": [{"action": action}],
            "action_count": 1,
            "certified": True,
            "certificate_reason": "history-rebound exact temporal prefix",
            "transition_rule": str(prediction["transition_kind"]),
            "suffix_exposed": False,
            "requires_transition_verification": True,
            "first_prediction": prediction,
            "progress_measure": {
                "metric": "remaining_private_recorded_replay_actions",
                "before": len(remaining),
                "expected_after": max(0, len(remaining) - 1),
                "strict": True,
            },
            "grounded_observation": {
                "entities": bind_current(frame, context=context).get(
                    "role_bindings", []
                ),
                "invariants": [
                    "promoted path slots retain ordinal identity",
                    "every identity has one exact predicted endpoint",
                    "static topology is unchanged outside bounded dynamics",
                ],
                "falsifiers": [
                    "actor or ghost endpoint mismatch",
                    "submit does not rewind and promote exactly one path",
                    "round slot, gate mode, or static topology mismatch",
                ],
            },
            "detail": {
                "private_plan_length": len(remaining),
                "round_index_before": int(before_round["index"]),
                "round_index_after": int(after_round["index"]),
                "ghost_count_before": len(prediction["before"]["ghosts"]),
                "ghost_count_after": len(prediction["expected_after"]["ghosts"]),
                "oracle_descriptor": str(resolved["model"]["goal_link_type"]),
            },
        }
    )


def _payload_model(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        shape = tuple(int(value) for value in payload["frame_shape"][:2])
        model = {
            "frame_shape": shape,
            "stride": int(payload["stride"]),
            "actor_value": int(payload["actor_value"]),
            "door_value": int(payload["door_value"]),
            "empty_marker_value": int(payload["empty_marker_value"]),
            "marker_center_value": int(payload["marker_center_value"]),
            "marker_slots": tuple(
                tuple(int(value) for value in position[:2])
                for position in payload["marker_slots"]
            ),
            "body_offsets": tuple(
                tuple(int(value) for value in position[:2])
                for position in payload["body_offsets"]
            ),
            "door_cells": tuple(
                tuple(int(value) for value in position[:2])
                for position in payload["door_cells"]
            ),
            "mechanism_cells": tuple(
                tuple(int(value) for value in position[:2])
                for position in payload["mechanism_cells"]
            ),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if (
        len(shape) != 2
        or model["stride"] <= 0
        or not model["marker_slots"]
        or not model["body_offsets"]
        or not model["door_cells"]
        or not model["mechanism_cells"]
    ):
        return None
    return model


def _projection_positions(projection: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    rows: dict[str, tuple[int, int]] = {}
    actor = projection.get("actor")
    if isinstance(actor, Mapping):
        position = actor.get("position")
        if isinstance(position, Sequence) and len(position) >= 2:
            rows["actor"] = (int(position[0]), int(position[1]))
    ghosts = projection.get("ghosts")
    if isinstance(ghosts, Sequence):
        for row in ghosts:
            if not isinstance(row, Mapping):
                continue
            position = row.get("position")
            ghost_id = str(row.get("ghost_id", "") or "")
            if ghost_id and isinstance(position, Sequence) and len(position) >= 2:
                rows[ghost_id] = (int(position[0]), int(position[1]))
    return rows


def verify_transition(expected: dict[str, Any], current_frame: Any) -> bool:
    """Fail closed unless the exact non-terminal temporal successor is visible."""

    if not isinstance(expected, Mapping):
        return False
    if (
        str(expected.get("family_id", "")) != FAMILY_ID
        or not expected.get("accepted")
        or not expected.get("lease_safe")
        or expected.get("terminal_goal_entry")
    ):
        return False
    frame = _latest_frame(current_frame)
    payload = expected.get("verification_model")
    after = expected.get("expected_after")
    before = expected.get("before")
    if (
        frame is None
        or not isinstance(payload, Mapping)
        or not isinstance(after, Mapping)
        or not isinstance(before, Mapping)
    ):
        return False
    model = _payload_model(payload)
    if model is None or tuple(frame.shape) != tuple(model["frame_shape"]):
        return False
    if _frame_digest(frame) == str(expected.get("before_frame_digest", "")):
        return False
    if stable_digest(after) != str(expected.get("expected_successor_digest", "")):
        return False

    round_state = after.get("round")
    if not isinstance(round_state, Mapping):
        return False
    round_index = int(round_state.get("index", -1))
    slots = _slot_observation(frame, model, round_index)
    if slots is None:
        return False
    known_ghost_values = [int(value) for value in payload.get("known_ghost_values", [])]
    observed_ghost_values = [int(value) for value in slots["ghost_values"]]
    if observed_ghost_values[: len(known_ghost_values)] != known_ghost_values:
        return False

    positions_after = _projection_positions(after)
    positions_before = _projection_positions(before)
    actor_position = positions_after.get("actor")
    if actor_position is None or not _body_has_value(
        frame,
        actor_position,
        model["body_offsets"],
        int(model["actor_value"]),
    ):
        return False

    grouped: dict[tuple[int, int], list[str]] = {}
    for entity_id, position in positions_after.items():
        grouped.setdefault(position, []).append(entity_id)
    for position, entity_ids in grouped.items():
        if "actor" in entity_ids:
            continue
        palettes = {
            observed_ghost_values[int(entity_id.split("_")[-1])]
            for entity_id in entity_ids
            if entity_id.startswith("ghost_")
            and int(entity_id.split("_")[-1]) < len(observed_ghost_values)
        }
        if not palettes or not any(
            _body_has_value(frame, position, model["body_offsets"], value)
            for value in palettes
        ):
            return False

    for entity_id, old_position in positions_before.items():
        new_occupants = [
            next_id
            for next_id, next_position in positions_after.items()
            if next_position == old_position
        ]
        if new_occupants:
            continue
        if entity_id == "actor":
            old_value = int(model["actor_value"])
        elif entity_id.startswith("ghost_"):
            index = int(entity_id.split("_")[-1])
            if index >= len(known_ghost_values):
                continue
            old_value = known_ghost_values[index]
        else:
            continue
        if _body_has_value(frame, old_position, model["body_offsets"], old_value):
            return False

    dynamic = _dynamic_cells(model, [*positions_before.values(), *positions_after.values()])
    if _masked_digest(frame, dynamic) != str(
        expected.get("static_topology_digest", "")
    ):
        return False

    gate_before = expected.get("gate_projection_before")
    if not isinstance(gate_before, Mapping):
        return False
    gate_after = _gate_projection(
        frame,
        model,
        [*positions_before.values(), *positions_after.values()],
    )
    changed = bool(expected.get("gate_occupancy_changed"))
    if changed == (gate_after.get("digest") == gate_before.get("digest")):
        return False
    return True


__all__ = [
    "FAMILY_ID",
    "bind_current",
    "compile_schema",
    "predict_transition",
    "propose_plan",
    "verify_transition",
]
