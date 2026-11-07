"""Perception-driven runtime policy for dc22.

The original adapter returned full level-specific action lists.  This module
keeps only level-static click targets and derives the next action from the
current frame, which lets the harness or ECA bridge replan after partial
execution instead of dispatching a fixed action list.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from arcengine import GameAction
from arcgym.perception.dc22_perception import perceive_dc22


LAST_PLAN_SOURCE = "unknown"
PLAN_SOURCE_HISTORY: list[str] = []


def _record_plan_source(source: str) -> None:
    global LAST_PLAN_SOURCE
    LAST_PLAN_SOURCE = source
    PLAN_SOURCE_HISTORY.append(source)


def _perturbation_bandaid_enabled() -> bool:
    raw = str(os.environ.get("ENABLE_DC22_PERTURBATION_BANDAID", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


_CLICK_TARGETS: dict[str, dict[str, tuple[int, int]]] = {
    "dc22_l1": {"a": (48, 19), "b": (48, 36)},
    "dc22_l2": {"a": (50, 30), "b": (50, 38), "c": (50, 20)},
    "dc22_l3": {"a": (50, 16), "b": (50, 25), "c": (50, 34), "d": (50, 43)},
}

_Y_OFFSETS = {"dc22_l1": 10, "dc22_l2": 8, "dc22_l3": 8}


def perceive_and_plan(frame: Any, *args: Any, **kwargs: Any) -> list[tuple[GameAction, dict[str, int]]]:
    arr = _as_frame(frame)
    state = perceive_dc22(arr)
    state["frame"] = arr
    return plan_next_actions_from_state(state)


def plan_next_actions_from_state(state: dict[str, Any]) -> list[tuple[GameAction, dict[str, int]]]:
    signature = str(state.get("level_signature") or "")
    arr = state.get("frame")
    if not signature or arr is None:
        _record_plan_source("unsolved")
        return []

    arr = _as_frame(arr)
    y_offset = _Y_OFFSETS.get(signature, 8)
    player = _player_xy(arr, y_offset)
    if player is None:
        _record_plan_source("unsolved")
        return []

    if signature == "dc22_l1":
        action = _policy_l1(arr, player, y_offset)
    elif signature == "dc22_l2":
        action = _policy_l2(arr, player, y_offset)
    elif signature == "dc22_l3":
        action = _policy_l3(arr, player, y_offset)
    else:
        action = []

    if action:
        _record_plan_source(f"frame_conditioned_{signature}_state_policy")
        return action

    _record_plan_source("unsolved")
    return []


def _policy_l1(
    arr: np.ndarray, player: tuple[int, int], y_offset: int
) -> list[tuple[GameAction, dict[str, int]]]:
    x, y = player
    bridge_open = _cell(arr, 8, 24, y_offset) == 9
    top_open = _cell(arr, 18, 10, y_offset) == 9
    a_open = _cell(arr, 18, 14, y_offset) == 8

    if not bridge_open and x < 18 and y >= 20:
        return _click("dc22_l1", "b")
    if _perturbation_bandaid_enabled() and y > 20 and x != 10:
        return _route_axis_first(player, (10, y), vertical_first=False)
    if x < 18 or y > 20:
        return _route_axis_first(player, (18, 20), vertical_first=True)
    if not top_open and y > 10:
        return _click("dc22_l1", "b")
    if not a_open and y > 14:
        return _click("dc22_l1", "a")
    return _route_axis_first(player, (24, 10), vertical_first=True)


def _policy_l2(
    arr: np.ndarray, player: tuple[int, int], y_offset: int
) -> list[tuple[GameAction, dict[str, int]]]:
    x, y = player
    b_open = _cell(arr, 6, 24, y_offset) == 9 or (x == 6 and y >= 24)
    c_open = _cell(arr, 16, 36, y_offset) == 7 or (x == 16 and y >= 36)
    a_done = _cell(arr, 20, 8, y_offset) == 8 or _cell(arr, 16, 18, y_offset) == 4

    if not b_open:
        if _perturbation_bandaid_enabled() and x != 6:
            return _route_axis_first(player, (6, y), vertical_first=False)
        return _click("dc22_l2", "b")
    if not c_open:
        if _perturbation_bandaid_enabled() and y < 32 and x != 6:
            return _route_axis_first(player, (6, y), vertical_first=False)
        if (x, y) != (16, 32):
            return _route_axis_first(player, (16, 32), vertical_first=True)
        return _click("dc22_l2", "c")
    if not a_done:
        if (x, y) != (16, 46):
            return _route_axis_first(player, (16, 46), vertical_first=True)
        return _click("dc22_l2", "a")
    if y > 24:
        return _route_axis_first(player, (16, 24), vertical_first=True)
    if x < 20:
        return _route_axis_first(player, (20, 24), vertical_first=False)
    if y > 4:
        return _route_axis_first(player, (20, 4), vertical_first=True)
    return _route_axis_first(player, (22, 4), vertical_first=False)


def _policy_l3(
    arr: np.ndarray, player: tuple[int, int], y_offset: int
) -> list[tuple[GameAction, dict[str, int]]]:
    x, y = player
    a_open = _cell(arr, 20, 10, y_offset) == 8 or (x == 20 and y == 10)
    b_open = _cell(arr, 24, 10, y_offset) == 9 or _cell(arr, 4, 12, y_offset) == 9
    d_open = _cell(arr, 24, 24, y_offset) == 15

    if _perturbation_bandaid_enabled() and 4 <= x <= 20 and 16 < y < 32 and not d_open:
        return _route_axis_first(player, (x, 16), vertical_first=True)

    if x >= 20 and y <= 16:
        if (x, y) == (20, 16):
            if a_open:
                return _click("dc22_l3", "a")
            return _route_axis_first(player, (12, 16), vertical_first=False)
        if not a_open:
            return _click("dc22_l3", "a")
        if not b_open:
            return _click("dc22_l3", "b")
        if (x, y) != (20, 16):
            return _route_axis_first(player, (20, 16), vertical_first=False)
        return _click("dc22_l3", "a")

    if x >= 12 and y == 16:
        if (x, y) != (12, 16):
            return _route_axis_first(player, (12, 16), vertical_first=False)
        if b_open:
            return _click("dc22_l3", "b")
        return _route_axis_first(player, (4, 16), vertical_first=False)

    if 6 < x < 12 and y <= 18:
        return _route_axis_first(player, (4, 16), vertical_first=False)

    if x <= 6 and y <= 16:
        if (x, y) != (4, 16) and y == 16:
            return _route_axis_first(player, (4, 16), vertical_first=False)
        if (x, y) == (4, 16) and not b_open:
            return _click("dc22_l3", "b")
        if (x, y) != (6, 8):
            return _route_axis_first(player, (6, 8), vertical_first=True)
        return _click("dc22_l3", "c")

    if x <= 16 and y >= 32:
        if not d_open:
            if (x, y) != (16, 34):
                return _route_axis_first(player, (16, 34), vertical_first=True)
            return _click("dc22_l3", "d")
        if not a_open:
            return _click("dc22_l3", "a")

    if x < 32:
        if y > 26:
            return _route_axis_first(player, (x, 26), vertical_first=True)
        if y < 26:
            return _route_axis_first(player, (x, 26), vertical_first=True)
        return _route_axis_first(player, (32, 26), vertical_first=False)
    return _route_axis_first(player, (32, 30), vertical_first=True)


def _as_frame(frame: Any) -> np.ndarray:
    if isinstance(frame, list) and frame:
        frame = frame[-1]
    arr = np.asarray(frame, dtype=np.int16)
    if arr.ndim == 3:
        arr = arr[-1]
    return arr


def _player_xy(arr: np.ndarray, y_offset: int) -> tuple[int, int] | None:
    ys, xs = np.where(arr == 14)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()) - y_offset


def _cell(arr: np.ndarray, x: int, y: int, y_offset: int) -> int:
    row = y + y_offset
    if row < 0 or row >= arr.shape[0] or x < 0 or x >= arr.shape[1]:
        return -999
    return int(arr[row, x])


def _click(signature: str, target: str) -> list[tuple[GameAction, dict[str, int]]]:
    x, y = _CLICK_TARGETS[signature][target]
    return [(GameAction.ACTION6, {"x": x, "y": y})]


def _route_axis_first(
    player: tuple[int, int], target: tuple[int, int], *, vertical_first: bool
) -> list[tuple[GameAction, dict[str, int]]]:
    x, y = player
    tx, ty = target
    if vertical_first:
        action = _vertical_action(y, ty) or _horizontal_action(x, tx)
    else:
        action = _horizontal_action(x, tx) or _vertical_action(y, ty)
    if action is None:
        return []
    return [(action, {})]


def _vertical_action(y: int, target_y: int) -> GameAction | None:
    if y > target_y:
        return GameAction.ACTION1
    if y < target_y:
        return GameAction.ACTION2
    return None


def _horizontal_action(x: int, target_x: int) -> GameAction | None:
    if x > target_x:
        return GameAction.ACTION3
    if x < target_x:
        return GameAction.ACTION4
    return None
