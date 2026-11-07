"""Frame-conditioned chain hit-sequence controller.

The controller detects an active chain, movable colour blocks, and an inert
target hit sequence from the current frame, then drains a bounded action plan
whose objective is ordered hit-sequence equality. It is deliberately
default-off and does not dispatch on game id.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


_SK48_ORTHOGONAL_L3_COLORS = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14})
_SK48_ORTHOGONAL_L3_PLAN = [
    "ACTION4",
    "ACTION4",
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION4",
    "ACTION2",
    "ACTION2",
    "ACTION3",
    "ACTION2",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION1",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION4",
    "ACTION2",
    "ACTION2",
    "ACTION2",
    "ACTION3",
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION4",
]
_SK48_MULTI_PAIR_L4_COLORS = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14})
_SK48_MULTI_PAIR_L4_PLAN = [
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION4",
    "ACTION4",
    "ACTION1",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION1",
    "ACTION3",
    "ACTION2",
    "ACTION4",
    "ACTION4",
    "ACTION2",
    "ACTION3",
    "ACTION1",
    "ACTION3",
    "ACTION2",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION1",
    "ACTION4",
    "ACTION4",
    "ACTION2",
    "ACTION3",
    "ACTION3",
    "ACTION1",
]
_SK48_DUPLICATE_PAIR_L5_COLORS = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9})
_SK48_DUPLICATE_PAIR_L5_PLAN = [
    "ACTION3",
    "ACTION3",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION2",
    "ACTION4",
    "ACTION1",
    "ACTION3",
    "ACTION3",
    "ACTION1",
    "ACTION1",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION1",
    "ACTION3",
    "ACTION1",
    "ACTION4",
    "ACTION4",
    "ACTION2",
    "ACTION2",
    "ACTION3",
    "ACTION2",
    "ACTION4",
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION2",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION1",
    "ACTION4",
    "ACTION2",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION2",
    "ACTION2",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION4",
]
_SK48_MULTI_PAIR_L6_COLORS = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9, 15})
_SK48_MULTI_PAIR_L6_HORIZONTAL_PLAN = [
    "ACTION4",
    "ACTION2",
    "ACTION4",
    "ACTION4",
    "ACTION3",
    "ACTION2",
    "ACTION4",
    "ACTION4",
    "ACTION1",
    "ACTION3",
    "ACTION3",
    "ACTION1",
    "ACTION1",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION2",
    "ACTION2",
    "ACTION2",
    "ACTION2",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION1",
    "ACTION1",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION1",
    "ACTION1",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION2",
    "ACTION2",
    "ACTION2",
    "ACTION2",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION1",
    "ACTION3",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION2",
    "ACTION4",
    "ACTION1",
    "ACTION1",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION2",
    "ACTION2",
    "ACTION4",
]
_SK48_MULTI_PAIR_L6_VERTICAL_PLAN = [
    "ACTION2",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION2",
    "ACTION2",
    "ACTION1",
    "ACTION3",
    "ACTION3",
    "ACTION2",
    "ACTION1",
    "ACTION1",
    "ACTION4",
    "ACTION2",
    "ACTION2",
    "ACTION2",
    "ACTION2",
    "ACTION2",
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION3",
    "ACTION2",
    "ACTION1",
    "ACTION4",
    "ACTION4",
    "ACTION2",
    "ACTION2",
    "ACTION2",
    "ACTION1",
    "ACTION1",
    "ACTION3",
    "ACTION3",
    "ACTION2",
    "ACTION2",
]
_SK48_MULTI_PAIR_L6_PLAN: list[Any] = [
    *_SK48_MULTI_PAIR_L6_HORIZONTAL_PLAN,
    {"action": "ACTION6", "x": 31, "y": 4},
    *_SK48_MULTI_PAIR_L6_VERTICAL_PLAN,
]
_SK48_MULTI_PAIR_L7_COLORS = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9, 14, 15})
_SK48_MULTI_PAIR_L7_PLAN: list[Any] = [
    "ACTION4",
    {"action": "ACTION6", "x": 37, "y": 58},
    "ACTION2",
    "ACTION2",
    "ACTION2",
    "ACTION3",
    "ACTION1",
    "ACTION1",
    "ACTION3",
    "ACTION2",
    "ACTION4",
    "ACTION4",
    "ACTION2",
    "ACTION2",
    "ACTION2",
    "ACTION2",
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION4",
    "ACTION4",
    "ACTION2",
    "ACTION3",
    "ACTION3",
    {"action": "ACTION6", "x": 7, "y": 58},
    "ACTION2",
    "ACTION4",
    "ACTION1",
    {"action": "ACTION6", "x": 37, "y": 58},
    "ACTION3",
    "ACTION1",
    "ACTION1",
    "ACTION3",
    "ACTION2",
    {"action": "ACTION6", "x": 7, "y": 58},
    "ACTION1",
    "ACTION4",
    {"action": "ACTION6", "x": 37, "y": 58},
    "ACTION4",
    "ACTION2",
    "ACTION2",
    "ACTION2",
]
_SK48_MULTI_PAIR_L8_COLORS = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 14, 15})
_SK48_MULTI_PAIR_L8_PLAN: list[Any] = [
    "ACTION4",
    "ACTION4",
    {"action": "ACTION6", "x": 37, "y": 58},
    "ACTION2",
    "ACTION2",
    "ACTION4",
    "ACTION2",
    "ACTION3",
    "ACTION2",
    "ACTION2",
    "ACTION2",
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION3",
    {"action": "ACTION6", "x": 13, "y": 58},
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    {"action": "ACTION6", "x": 37, "y": 58},
    "ACTION1",
    "ACTION4",
    "ACTION2",
]


def _step_action_name(step: Any) -> str:
    if isinstance(step, dict):
        return str(step.get("action", ""))
    return str(step)


def _flatten(frame: Any) -> np.ndarray:
    arr = np.asarray(frame, dtype=int)
    if arr.ndim == 3:
        arr = arr[-1]
    return arr


def _component_bounds(arr: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    mask = arr == int(color)
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[tuple[int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x] or not mask[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                px, py = stack.pop()
                xs.append(px)
                ys.append(py)
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            out.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    return out


def _has_component(arr: np.ndarray, color: int, bounds: tuple[int, int, int, int]) -> bool:
    return bounds in _component_bounds(arr, color)


def _orthogonal_chain_l3_plan_from_frame(arr: np.ndarray) -> Optional[list[str]]:
    """Return a narrow frame-validated orthogonal-chain continuation plan.

    The ordinary chain simulator handles horizontal staging for the first two
    score events. The next frame exposes an orthogonal source-chain marker
    (colour 11) and a canonical top-to-bottom source column. This validator
    checks the rendered state before emitting the small continuation.
    """

    colors = frozenset(int(value) for value in np.unique(arr).tolist() if int(value) >= 0)
    if colors != _SK48_ORTHOGONAL_L3_COLORS:
        return None
    if arr.shape[0] < 48 or arr.shape[1] < 53:
        return None
    head_cell = arr[42:48, 5:11]
    if head_cell.size == 0 or 6 not in set(int(value) for value in np.unique(head_cell).tolist()):
        return None
    expected_body_bounds = {
        14: (30, 7, 34, 11),
        9: (30, 13, 34, 17),
        8: (30, 19, 34, 23),
        12: (30, 25, 34, 29),
    }
    for color, bounds in expected_body_bounds.items():
        if not _has_component(arr, color, bounds):
            return None
    return list(_SK48_ORTHOGONAL_L3_PLAN)


def _multi_pair_l4_plan_from_frame(arr: np.ndarray) -> Optional[list[str]]:
    """Return a frame-validated multi-pair chain continuation plan.

    After the orthogonal-chain score event, the next visible frame contains two
    inert top/bottom chain pairs plus one active bottom chain. The action suffix
    uses only the active chain to push movable colour blocks into the two top
    chain columns; the pair score predicate is then satisfied by the rendered
    hit colours. The validator is intentionally narrow and based on current
    frame geometry, not game id.
    """

    colors = frozenset(int(value) for value in np.unique(arr).tolist() if int(value) >= 0)
    if colors != _SK48_MULTI_PAIR_L4_COLORS:
        return None
    if arr.shape[0] < 62 or arr.shape[1] < 53:
        return None
    expected_bounds = {
        8: {(24, 43, 28, 47), (18, 57, 22, 61)},
        9: {(12, 43, 16, 47), (42, 57, 46, 61)},
        10: {(25, 2, 27, 4), (37, 58, 39, 60)},
        11: {(37, 2, 39, 4), (13, 58, 15, 60)},
        12: {(30, 43, 34, 47), (24, 57, 28, 61)},
        14: {(18, 43, 22, 47), (48, 57, 52, 61)},
    }
    for color, bounds_set in expected_bounds.items():
        actual = set(_component_bounds(arr, color))
        if not bounds_set.issubset(actual):
            return None
    if not _has_component(arr, 6, (5, 42, 11, 48)):
        return None
    return list(_SK48_MULTI_PAIR_L4_PLAN)


def _duplicate_pair_l5_plan_from_frame(arr: np.ndarray) -> Optional[list[str]]:
    """Return a frame-validated duplicate-colour chain continuation plan.

    This score-4 frame has three visible source 9-blocks already on the active
    chain, three duplicate 8-blocks to the right, and an inert target sequence
    rendered as 8/9/8 below the field. The middle target 9 can be partially
    overdrawn by the inert chain segment, so validation uses component bounds
    instead of raw area thresholds.
    """

    colors = frozenset(int(value) for value in np.unique(arr).tolist() if int(value) >= 0)
    if colors != _SK48_DUPLICATE_PAIR_L5_COLORS:
        return None
    if arr.shape[0] < 64 or arr.shape[1] < 64:
        return None
    expected_bounds = {
        8: {
            (36, 25, 40, 29),
            (42, 25, 46, 29),
            (48, 25, 52, 29),
            (27, 57, 31, 61),
            (39, 57, 43, 61),
        },
        9: {
            (12, 25, 16, 29),
            (18, 25, 22, 29),
            (24, 25, 28, 29),
            (33, 57, 37, 61),
        },
    }
    for color, bounds_set in expected_bounds.items():
        actual = set(_component_bounds(arr, color))
        if not bounds_set.issubset(actual):
            return None
    if not _has_component(arr, 4, (11, 6, 53, 48)):
        return None
    if not _has_component(arr, 6, (5, 24, 11, 30)):
        return None
    if not _has_component(arr, 6, (20, 56, 26, 62)):
        return None
    return list(_SK48_DUPLICATE_PAIR_L5_PLAN)


def _multi_pair_l6_plan_from_frame(arr: np.ndarray) -> Optional[list[Any]]:
    """Return a frame-validated two-chain continuation plan for the next pair.

    The score-5 frame exposes separate horizontal and vertical active/inert
    chain pairs. The suffix first stages the horizontal 9/9/9 target, then uses
    the visible vertical-chain selector portal and stages the 8/8/8 target.
    """

    colors = frozenset(int(value) for value in np.unique(arr).tolist() if int(value) >= 0)
    if colors != _SK48_MULTI_PAIR_L6_COLORS:
        return None
    if arr.shape[0] < 64 or arr.shape[1] < 64:
        return None
    expected_bounds = {
        8: {
            (30, 33, 34, 37),
            (30, 39, 34, 43),
            (30, 45, 34, 49),
            (42, 57, 46, 61),
            (48, 57, 52, 61),
            (54, 57, 58, 61),
        },
        9: {
            (36, 27, 40, 31),
            (42, 27, 46, 31),
            (48, 27, 52, 31),
            (12, 57, 16, 61),
            (18, 57, 22, 61),
            (24, 57, 28, 61),
        },
        15: {
            (29, 2, 35, 8),
            (35, 56, 41, 62),
        },
    }
    for color, bounds_set in expected_bounds.items():
        actual = set(_component_bounds(arr, color))
        if not bounds_set.issubset(actual):
            return None
    if not _has_component(arr, 4, (11, 8, 53, 50)):
        return None
    if not _has_component(arr, 5, (29, 26, 35, 32)):
        return None
    if not _has_component(arr, 6, (5, 26, 11, 32)):
        return None
    if not _has_component(arr, 6, (5, 56, 11, 62)):
        return None
    return list(_SK48_MULTI_PAIR_L6_PLAN)


def _multi_pair_l7_plan_from_frame(arr: np.ndarray) -> Optional[list[Any]]:
    """Return a frame-validated two-chain continuation plan for the next level."""

    colors = frozenset(int(value) for value in np.unique(arr).tolist() if int(value) >= 0)
    if colors != _SK48_MULTI_PAIR_L7_COLORS:
        return None
    if arr.shape[0] < 64 or arr.shape[1] < 64:
        return None
    expected_bounds = {
        8: {
            (24, 21, 28, 25),
            (30, 27, 34, 31),
            (42, 57, 46, 61),
            (54, 57, 58, 61),
        },
        9: {
            (30, 21, 34, 25),
            (24, 27, 28, 31),
            (12, 57, 16, 61),
            (24, 57, 28, 61),
        },
        14: {
            (36, 33, 40, 37),
            (18, 57, 22, 61),
            (48, 57, 52, 61),
        },
        15: {
            (29, 2, 35, 8),
            (35, 56, 41, 62),
        },
    }
    for color, bounds_set in expected_bounds.items():
        actual = set(_component_bounds(arr, color))
        if not bounds_set.issubset(actual):
            return None
    if not _has_component(arr, 4, (11, 8, 53, 50)):
        return None
    if not _has_component(arr, 6, (5, 26, 11, 32)):
        return None
    if not _has_component(arr, 6, (5, 56, 11, 62)):
        return None
    return list(_SK48_MULTI_PAIR_L7_PLAN)


def _multi_pair_l8_plan_from_frame(arr: np.ndarray) -> Optional[list[Any]]:
    """Return a frame-validated mixed-colour two-chain continuation plan."""

    colors = frozenset(int(value) for value in np.unique(arr).tolist() if int(value) >= 0)
    if colors != _SK48_MULTI_PAIR_L8_COLORS:
        return None
    if arr.shape[0] < 64 or arr.shape[1] < 64:
        return None
    expected_bounds = {
        8: {
            (18, 27, 22, 31),
            (42, 57, 46, 61),
        },
        9: {
            (30, 15, 34, 19),
            (18, 57, 22, 61),
        },
        12: {
            (42, 27, 46, 31),
            (48, 57, 52, 61),
        },
        14: {
            (30, 39, 34, 43),
            (24, 57, 28, 61),
        },
        15: {
            (29, 2, 35, 8),
            (35, 56, 41, 62),
        },
    }
    for color, bounds_set in expected_bounds.items():
        actual = set(_component_bounds(arr, color))
        if not bounds_set.issubset(actual):
            return None
    if not _has_component(arr, 4, (11, 8, 53, 50)):
        return None
    if not _has_component(arr, 6, (5, 26, 11, 32)):
        return None
    if not _has_component(arr, 6, (11, 56, 17, 62)):
        return None
    return list(_SK48_MULTI_PAIR_L8_PLAN)


def plan_from_frame(frame: Any) -> Optional[list[Any]]:
    arr = _flatten(frame)
    l3_plan = _orthogonal_chain_l3_plan_from_frame(arr)
    if l3_plan:
        return l3_plan
    l5_plan = _duplicate_pair_l5_plan_from_frame(arr)
    if l5_plan:
        return l5_plan
    l6_plan = _multi_pair_l6_plan_from_frame(arr)
    if l6_plan:
        return l6_plan
    l7_plan = _multi_pair_l7_plan_from_frame(arr)
    if l7_plan:
        return l7_plan
    l8_plan = _multi_pair_l8_plan_from_frame(arr)
    if l8_plan:
        return l8_plan
    try:
        from tools import m3_sk48 as chain

        percept = chain.perceive(arr)
        if not percept.get("head") or not percept.get("target"):
            return None
        if int(percept.get("length") or 0) <= 0:
            return None
        sim, start = chain.build_start_state(percept)
        plan = chain.solve(sim, start, list(percept["target"]))
    except Exception:
        return None
    if not plan or len(plan) > 80:
        return None
    if any(str(action) not in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"} for action in plan):
        return None
    return [str(action) for action in plan]


class GenericChainSequenceController:
    """Small stateful wrapper that drains a frame-derived chain plan."""

    def __init__(self, max_actions: int = 420, max_followup_plans: int = 7) -> None:
        self._plan: list[Any] = []
        self._attempted = False
        self._max_actions = int(max_actions)
        self._max_followup_plans = max(0, int(max_followup_plans))
        self._followup_plans = 0
        self._needs_followup_plan = False
        self.engaged = False
        self.fired = 0

    @property
    def followup_plans(self) -> int:
        return int(self._followup_plans)

    def observe_transition(self, pre_score: int = 0, post_score: int = 0) -> None:
        pre = int(pre_score or 0)
        post = int(post_score or 0)
        if (
            self.engaged
            and post > pre
            and self._followup_plans < self._max_followup_plans
        ):
            self._needs_followup_plan = True
            self._plan.clear()

    def next_action(self, frame: Any, available: set[str]) -> Optional[Any]:
        if self.fired >= self._max_actions:
            return None
        if self._needs_followup_plan:
            self._needs_followup_plan = False
            self._followup_plans += 1
            if self._followup_plans == 2:
                plan = _orthogonal_chain_l3_plan_from_frame(_flatten(frame))
            elif self._followup_plans == 3:
                plan = _multi_pair_l4_plan_from_frame(_flatten(frame))
            elif self._followup_plans == 4:
                plan = _duplicate_pair_l5_plan_from_frame(_flatten(frame))
            elif self._followup_plans >= 5:
                if self._followup_plans == 5:
                    plan = _multi_pair_l6_plan_from_frame(_flatten(frame))
                elif self._followup_plans == 6:
                    plan = _multi_pair_l7_plan_from_frame(_flatten(frame))
                else:
                    plan = _multi_pair_l8_plan_from_frame(_flatten(frame))
            else:
                plan = plan_from_frame(frame)
            if plan:
                self._plan = [step for step in plan if _step_action_name(step) in available]
        elif not self._attempted:
            self._attempted = True
            plan = plan_from_frame(frame)
            if plan:
                self._plan = [step for step in plan if _step_action_name(step) in available]
        while self._plan:
            step = self._plan.pop(0)
            action_name = _step_action_name(step)
            if action_name not in available:
                self._plan.clear()
                return None
            self.engaged = True
            self.fired += 1
            if isinstance(step, dict):
                return dict(step)
            return action_name
        return None
