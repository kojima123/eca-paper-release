"""Visible rigid-sprite anchor assembly with exact one-step verification."""
from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from arcgym.eca.primitive_synthesis import (
    _sprite_anchor_cell_sample,
    _sprite_anchor_overlap_plan,
    _sprite_anchor_overlap_state,
)

from .base import MechanicContext, as_frame, stable_digest


FAMILY_ID = "anchor_overlap_assembly"

_ENGINE_ACTIONS = frozenset(f"ACTION{index}" for index in range(1, 7))
_ACTION_ALIASES = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
    "SPACE": "ACTION5",
    "MOUSE": "ACTION6",
}
_MOVES = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}
_VISIBLE_STATE_TYPES = frozenset(
    {"sprite_anchor_overlap", "sprite_anchor_overlap_chain"}
)


@dataclass
class _Sprite:
    sprite_id: str
    pos: tuple[int, int]
    width: int
    height: int
    cells: dict[tuple[int, int], int]
    anchors: frozenset[tuple[int, int]]
    variant: int = 0

    def clone(self) -> _Sprite:
        return _Sprite(
            sprite_id=self.sprite_id,
            pos=self.pos,
            width=self.width,
            height=self.height,
            cells=dict(self.cells),
            anchors=frozenset(self.anchors),
            variant=self.variant,
        )

    def world_cells(self) -> dict[tuple[int, int], int]:
        px, py = self.pos
        return {
            (px + rx, py + ry): int(value)
            for (rx, ry), value in self.cells.items()
        }

    def world_anchors(self) -> frozenset[tuple[int, int]]:
        px, py = self.pos
        return frozenset((px + rx, py + ry) for rx, ry in self.anchors)


@dataclass
class _World:
    sprites: list[_Sprite]
    selected_id: str

    def clone(self) -> _World:
        return _World(
            sprites=[sprite.clone() for sprite in self.sprites],
            selected_id=self.selected_id,
        )

    def selected(self) -> _Sprite | None:
        return next(
            (
                sprite
                for sprite in self.sprites
                if sprite.sprite_id == self.selected_id
            ),
            None,
        )


@dataclass
class _Assembly:
    cell: int
    grid_size: tuple[int, int]
    background_value: int
    anchor_value: int
    static_values: dict[tuple[int, int], int]
    initial_world: _World
    plan: list[dict[str, Any]]
    oracle_state_type: str


@dataclass
class _Resolved:
    assembly: _Assembly
    world: _World
    plan_index: int


_RESOLVE_CACHE: OrderedDict[str, _Resolved | None] = OrderedDict()
_RESOLVE_CACHE_SIZE = 32


def _frame2d(value: Any) -> np.ndarray | None:
    frame = as_frame(value)
    if frame is not None:
        return frame
    if not isinstance(value, Mapping):
        return None
    history = value.get("history")
    if not isinstance(history, Sequence):
        return None
    for item in reversed(history):
        if isinstance(item, Mapping) and (frame := as_frame(item)) is not None:
            return frame
    return None


def _action(value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, Mapping) else {}
    raw = payload.get("action") if isinstance(value, Mapping) else value
    if hasattr(raw, "name"):
        raw = raw.name
    name = str(raw or "").split(".")[-1].strip().upper()
    name = _ACTION_ALIASES.get(name, name)
    result: dict[str, Any] = {"action": name}
    if name == "ACTION6":
        x = payload.get("x", payload.get("col"))
        y = payload.get("y", payload.get("row"))
        try:
            result.update({"x": int(x), "y": int(y)})
        except (TypeError, ValueError):
            pass
    return result


def _allowed(value: Sequence[Any] | set[Any] | None) -> set[str]:
    if not value:
        return set(_ENGINE_ACTIONS)
    return {
        normalized["action"]
        for item in value
        if (normalized := _action(item))["action"] in _ENGINE_ACTIONS
    }


def _action_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    if str(expected.get("action")) != str(actual.get("action")):
        return False
    if str(expected.get("action")) != "ACTION6":
        return True
    return (
        int(expected.get("x", -1)) == int(actual.get("x", -2))
        and int(expected.get("y", -1)) == int(actual.get("y", -2))
    )


def _connected_components(
    values: Mapping[tuple[int, int], int],
    *,
    background: int,
    anchor_value: int,
) -> list[dict[str, Any]]:
    remaining = {
        (int(x), int(y))
        for (x, y), value in values.items()
        if int(value) != int(background)
    }
    components: list[dict[str, Any]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        cells: set[tuple[int, int]] = set()
        while stack:
            x, y = stack.pop()
            cells.add((x, y))
            for neighbor in (
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        anchors = {
            cell
            for cell in cells
            if int(values.get(cell, background)) == int(anchor_value)
        }
        if len(anchors) < 2 or not 4 <= len(cells) <= 64:
            continue
        min_x = min(x for x, _ in cells)
        max_x = max(x for x, _ in cells)
        min_y = min(y for _, y in cells)
        max_y = max(y for _, y in cells)
        components.append(
            {
                "pos": (min_x, min_y),
                "width": max_x - min_x + 1,
                "height": max_y - min_y + 1,
                "cells": cells,
                "anchors": anchors,
            }
        )
    return sorted(components, key=lambda item: (item["pos"], len(item["cells"])))


def _normalize_oracle_plan(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Sequence) or not item:
            return []
        action = _action(item[0])
        if action["action"] not in _ENGINE_ACTIONS:
            return []
        payload = item[1] if len(item) > 1 else None
        if isinstance(payload, Mapping):
            action = _action({"action": action["action"], **dict(payload)})
        if action["action"] == "ACTION6" and not {"x", "y"}.issubset(action):
            return []
        result.append(action)
    return result


def _initial_assembly(frame: Any) -> _Assembly | None:
    arr = _frame2d(frame)
    if arr is None:
        return None
    oracle = _sprite_anchor_overlap_state(np.array(arr, copy=True))
    if not isinstance(oracle, Mapping):
        return None
    state_type = str(oracle.get("mechanic_state_type", "") or "")
    if state_type not in _VISIBLE_STATE_TYPES:
        return None
    try:
        cell = int(oracle.get("cell", 0) or 0)
        grid_size = tuple(int(value) for value in oracle.get("grid_size", ()))
        anchor_value = int(oracle["anchor_color"])
        active_pos = tuple(int(value) for value in oracle.get("active_pos", ()))
        component_count = int(oracle.get("component_count", 0) or 0)
    except (KeyError, TypeError, ValueError):
        return None
    if cell <= 0 or len(grid_size) != 2 or len(active_pos) != 2:
        return None
    values, background = _sprite_anchor_cell_sample(np.array(arr, copy=True), cell=cell)
    if not values:
        return None
    components = _connected_components(
        values,
        background=int(background),
        anchor_value=anchor_value,
    )
    if len(components) != component_count:
        return None

    sprites: list[_Sprite] = []
    sprite_world_cells: set[tuple[int, int]] = set()
    for index, component in enumerate(components):
        pos = tuple(component["pos"])
        px, py = pos
        world_cells = set(component["cells"])
        sprite_world_cells.update(world_cells)
        relative_cells = {
            (x - px, y - py): int(values[(x, y)])
            for x, y in world_cells
        }
        relative_anchors = frozenset(
            (x - px, y - py) for x, y in component["anchors"]
        )
        sprites.append(
            _Sprite(
                sprite_id=f"sprite_{index}",
                pos=(px, py),
                width=int(component["width"]),
                height=int(component["height"]),
                cells=relative_cells,
                anchors=relative_anchors,
            )
        )
    selected = next((sprite for sprite in sprites if sprite.pos == active_pos), None)
    if selected is None:
        return None
    static_values = {
        cell_pos: int(value)
        for cell_pos, value in values.items()
        if cell_pos not in sprite_world_cells and int(value) != int(background)
    }
    raw_plan = _sprite_anchor_overlap_plan(oracle, set(_ENGINE_ACTIONS))
    plan = _normalize_oracle_plan(raw_plan)
    if not plan:
        return None
    assembly = _Assembly(
        cell=cell,
        grid_size=(grid_size[0], grid_size[1]),
        background_value=int(background),
        anchor_value=anchor_value,
        static_values=static_values,
        initial_world=_World(sprites=sprites, selected_id=selected.sprite_id),
        plan=plan,
        oracle_state_type=state_type,
    )
    final_world = _simulate(assembly, len(plan))
    if final_world is None or _anchor_deficit(final_world) != 0:
        return None
    if _anchor_deficit(assembly.initial_world) <= 0:
        return None
    return assembly


def _rotate(sprite: _Sprite) -> None:
    old_height = sprite.height
    sprite.cells = {
        (old_height - 1 - y, x): value
        for (x, y), value in sprite.cells.items()
    }
    sprite.anchors = frozenset(
        (old_height - 1 - y, x) for x, y in sprite.anchors
    )
    sprite.width, sprite.height = sprite.height, sprite.width
    sprite.variant = (sprite.variant + 1) % 4


def _apply(world: _World, action: Mapping[str, Any], *, cell: int) -> bool:
    name = str(action.get("action", "") or "")
    selected = world.selected()
    if selected is None:
        return False
    if name in _MOVES:
        dx, dy = _MOVES[name]
        selected.pos = (selected.pos[0] + dx, selected.pos[1] + dy)
        return True
    if name == "ACTION5":
        _rotate(selected)
        return True
    if name != "ACTION6":
        return False
    try:
        click = (int(action["x"]) // cell, int(action["y"]) // cell)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    candidates = [
        sprite for sprite in world.sprites if click in sprite.world_cells()
    ]
    if not candidates:
        return False
    candidates.sort(
        key=lambda sprite: (
            sprite.sprite_id == world.selected_id,
            click not in sprite.world_anchors(),
            sprite.sprite_id,
        )
    )
    world.selected_id = candidates[0].sprite_id
    return True


def _simulate(assembly: _Assembly, count: int) -> _World | None:
    if count < 0 or count > len(assembly.plan):
        return None
    world = assembly.initial_world.clone()
    for action in assembly.plan[:count]:
        if not _apply(world, action, cell=assembly.cell):
            return None
    return world


def _anchor_counter(world: _World) -> Counter[tuple[int, int, str]]:
    return Counter(
        (x, y, "anchor_class_0")
        for sprite in world.sprites
        for x, y in sprite.world_anchors()
    )


def _anchor_deficit(world: _World) -> int:
    counts = _anchor_counter(world)
    return sum(count for count in counts.values() if count < 2)


def _world_occupancy(world: _World) -> frozenset[tuple[int, int]]:
    return frozenset(
        cell for sprite in world.sprites for cell in sprite.world_cells()
    )


def _anchor_checks(world: _World) -> tuple[list[list[int]], list[list[int]]]:
    cover: dict[tuple[int, int], list[bool]] = {}
    for sprite in world.sprites:
        anchors = sprite.world_anchors()
        for cell in sprite.world_cells():
            cover.setdefault(cell, []).append(cell in anchors)
    required = [
        [int(x), int(y)]
        for (x, y), roles in sorted(cover.items())
        if len(roles) == 1 and roles[0]
    ]
    forbidden = [
        [int(x), int(y)]
        for (x, y), roles in sorted(cover.items())
        if roles and not any(roles)
    ]
    return required, forbidden


def _projection(assembly: _Assembly, world: _World) -> dict[str, Any]:
    required, forbidden = _anchor_checks(world)
    occupancy = sorted(
        set(_world_occupancy(world)) | set(assembly.static_values)
    )
    return {
        "cell": int(assembly.cell),
        "grid_size": [int(value) for value in assembly.grid_size],
        "background_value": int(assembly.background_value),
        "anchor_value": int(assembly.anchor_value),
        "expected_non_background_cells": [
            [int(x), int(y)] for x, y in occupancy
        ],
        "required_anchor_cells": required,
        "forbidden_anchor_cells": forbidden,
        "static_values": [
            [int(x), int(y), int(value)]
            for (x, y), value in sorted(assembly.static_values.items())
        ],
    }


def _projection_matches(projection: Mapping[str, Any], frame: Any) -> bool:
    arr = _frame2d(frame)
    if arr is None:
        return False
    try:
        cell = int(projection["cell"])
        expected_grid = tuple(int(value) for value in projection["grid_size"])
        background = int(projection["background_value"])
        anchor = int(projection["anchor_value"])
        expected_occupancy = {
            (int(point[0]), int(point[1]))
            for point in projection["expected_non_background_cells"]
        }
        required = {
            (int(point[0]), int(point[1]))
            for point in projection["required_anchor_cells"]
        }
        forbidden = {
            (int(point[0]), int(point[1]))
            for point in projection["forbidden_anchor_cells"]
        }
        static_values = {
            (int(item[0]), int(item[1])): int(item[2])
            for item in projection["static_values"]
        }
    except (KeyError, TypeError, ValueError, IndexError):
        return False
    values, observed_background = _sprite_anchor_cell_sample(
        np.array(arr, copy=True), cell=cell
    )
    if not values or int(observed_background) != background:
        return False
    observed_grid = (
        max((x for x, _ in values), default=-1) + 1,
        max((y for _, y in values), default=-1) + 1,
    )
    if observed_grid != expected_grid:
        return False
    observed_occupancy = {
        point for point, value in values.items() if int(value) != background
    }
    if observed_occupancy != expected_occupancy:
        return False
    if any(int(values.get(point, background)) != anchor for point in required):
        return False
    if any(int(values.get(point, background)) == anchor for point in forbidden):
        return False
    return all(int(values.get(point, background)) == value for point, value in static_values.items())


def _context_frames(
    current_frame: Any,
    context: MechanicContext | Mapping[str, Any] | None,
) -> list[np.ndarray]:
    level = 0
    history: Sequence[Any] = ()
    previous_frame: Any = None
    if isinstance(context, MechanicContext):
        level = int(context.level)
        history = context.history
        previous_frame = context.previous_frame
    elif isinstance(context, Mapping):
        level = int(context.get("level", 0) or 0)
        raw_history = context.get("history", ())
        if isinstance(raw_history, Sequence) and not isinstance(
            raw_history, (str, bytes)
        ):
            history = raw_history
        previous_frame = context.get("previous_frame")

    frames: list[np.ndarray] = []
    seen: set[str] = set()

    def append(candidate: Any, candidate_level: Any = 0) -> None:
        try:
            item_level = int(candidate_level or 0)
        except (TypeError, ValueError):
            item_level = 0
        if level and item_level and item_level != level:
            return
        arr = _frame2d(candidate)
        if arr is None:
            return
        digest = stable_digest(np.asarray(arr))
        if digest not in seen:
            seen.add(digest)
            frames.append(arr)

    for item in history:
        if isinstance(item, Mapping):
            append(item, item.get("level", 0))
    append(previous_frame, level)
    append(current_frame, level)
    return frames


def _history_index_hint(
    initial_frame: np.ndarray,
    current_frame: Any,
    context: MechanicContext | Mapping[str, Any] | None,
) -> int | None:
    current = _frame2d(current_frame)
    if current is None:
        return None
    level = 0
    history: Sequence[Any] = ()
    if isinstance(context, MechanicContext):
        level = int(context.level)
        history = context.history
    elif isinstance(context, Mapping):
        level = int(context.get("level", 0) or 0)
        raw_history = context.get("history", ())
        if isinstance(raw_history, Sequence) and not isinstance(
            raw_history, (str, bytes)
        ):
            history = raw_history
    entries: list[np.ndarray] = []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        try:
            item_level = int(item.get("level", 0) or 0)
        except (TypeError, ValueError):
            item_level = 0
        if level and item_level and item_level != level:
            continue
        arr = _frame2d(item)
        if arr is not None:
            entries.append(arr)
    if not entries:
        return None
    initial_digest = stable_digest(initial_frame)
    current_digest = stable_digest(current)
    start = next(
        (
            index
            for index, entry in enumerate(entries)
            if stable_digest(entry) == initial_digest
        ),
        None,
    )
    if start is None:
        return None
    transition_count = len(entries) - start - 1
    if stable_digest(entries[-1]) != current_digest:
        transition_count += 1
    return max(0, int(transition_count))


def _resolve_cache_key(
    frame: Any,
    context: MechanicContext | Mapping[str, Any] | None,
) -> str:
    current = _frame2d(frame)
    level = 0
    history: Sequence[Any] = ()
    previous: Any = None
    if isinstance(context, MechanicContext):
        level = int(context.level)
        history = context.history
        previous = context.previous_frame
    elif isinstance(context, Mapping):
        level = int(context.get("level", 0) or 0)
        raw_history = context.get("history", ())
        if isinstance(raw_history, Sequence) and not isinstance(
            raw_history, (str, bytes)
        ):
            history = raw_history
        previous = context.get("previous_frame")
    relevant: list[tuple[int, str]] = []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        try:
            item_level = int(item.get("level", 0) or 0)
        except (TypeError, ValueError):
            item_level = 0
        if level and item_level and item_level != level:
            continue
        arr = _frame2d(item)
        if arr is not None:
            relevant.append((item_level, stable_digest(arr)))
    previous_arr = _frame2d(previous)
    return stable_digest(
        {
            "level": level,
            "current": stable_digest(current) if current is not None else "",
            "history_count": len(relevant),
            "history_first": relevant[0] if relevant else None,
            "history_last": relevant[-1] if relevant else None,
            "previous": (
                stable_digest(previous_arr) if previous_arr is not None else ""
            ),
        }
    )


def _resolve_uncached(
    frame: Any,
    context: MechanicContext | Mapping[str, Any] | None,
) -> _Resolved | None:
    for initial_frame in _context_frames(frame, context):
        assembly = _initial_assembly(initial_frame)
        if assembly is None:
            continue
        matches: list[tuple[int, _World]] = []
        for index in range(len(assembly.plan) + 1):
            world = _simulate(assembly, index)
            if world is None:
                break
            if _projection_matches(_projection(assembly, world), frame):
                matches.append((index, world))
        if matches:
            hint = _history_index_hint(initial_frame, frame, context)
            hinted = [item for item in matches if item[0] == hint]
            index, world = (
                hinted[0]
                if hinted
                else min(matches, key=lambda item: item[0])
            )
            return _Resolved(assembly=assembly, world=world, plan_index=index)
    return None


def _resolve(
    frame: Any,
    context: MechanicContext | Mapping[str, Any] | None,
) -> _Resolved | None:
    key = _resolve_cache_key(frame, context)
    if key in _RESOLVE_CACHE:
        _RESOLVE_CACHE.move_to_end(key)
        return _RESOLVE_CACHE[key]
    resolved = _resolve_uncached(frame, context)
    _RESOLVE_CACHE[key] = resolved
    _RESOLVE_CACHE.move_to_end(key)
    while len(_RESOLVE_CACHE) > _RESOLVE_CACHE_SIZE:
        _RESOLVE_CACHE.popitem(last=False)
    return resolved


def _sprite_binding(sprite: _Sprite, *, selected: bool) -> dict[str, Any]:
    shape = {
        "occupancy": sorted([int(x), int(y)] for x, y in sprite.cells),
        "anchor_offsets": sorted([int(x), int(y)] for x, y in sprite.anchors),
    }
    return {
        "role": "movable_sprite",
        "sprite_id": sprite.sprite_id,
        "pose": [int(sprite.pos[0]), int(sprite.pos[1])],
        "size": [int(sprite.width), int(sprite.height)],
        "variant": int(sprite.variant),
        "selected": bool(selected),
        "shape_signature": stable_digest(shape),
        "anchor_classes": [
            {
                "class_id": "anchor_class_0",
                "offsets": shape["anchor_offsets"],
            }
        ],
    }


def _unbound(reason: str) -> dict[str, Any]:
    return {
        "mechanic_id": FAMILY_ID,
        "family_id": FAMILY_ID,
        "status": "unbound",
        "supported": False,
        "accepted": False,
        "reason": str(reason),
        "roles": [],
        "role_bindings": [],
    }


def bind_current(
    frame: Any,
    *,
    context: MechanicContext | Mapping[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    resolved = _resolve(frame, context)
    if resolved is None:
        return _unbound("no_exact_visible_anchor_assembly")
    assembly = resolved.assembly
    world = resolved.world
    sprites = [
        _sprite_binding(
            sprite,
            selected=sprite.sprite_id == world.selected_id,
        )
        for sprite in world.sprites
    ]
    if len({str(sprite["shape_signature"]) for sprite in sprites}) < 2:
        return _unbound("anchor_assembly_requires_distinct_rigid_shapes")
    selector = {
        "role": "active_selector",
        "selected_sprite_id": world.selected_id,
        "selection_action": "ACTION6",
    }
    anchor_role = {
        "role": "anchor_pixel",
        "class_id": "anchor_class_0",
        "world_multiset": [
            [int(x), int(y), int(count)]
            for (x, y, _class_id), count in sorted(_anchor_counter(world).items())
        ],
    }
    roles = [*sprites, selector, anchor_role]
    return {
        "mechanic_id": FAMILY_ID,
        "family_id": FAMILY_ID,
        "status": "bound",
        "supported": True,
        "accepted": True,
        "roles": roles,
        "role_bindings": roles,
        "sprite_count": len(sprites),
        "selected_sprite_id": world.selected_id,
        "anchor_class_count": 1,
        "anchor_deficit": _anchor_deficit(world),
        "matched_anchor_occurrences": sum(
            count
            for count in _anchor_counter(world).values()
            if count >= 2
        ),
        "variant_transition_model": {
            "action": "ACTION5",
            "effect": "quarter_turn_of_selected_rigid_sprite",
            "variant_modulus": 4,
        },
        "plan_cursor": int(resolved.plan_index),
        "private_plan_length": len(assembly.plan),
        "oracle_state_type": assembly.oracle_state_type,
    }


def compile_schema(
    frame: Any,
    *,
    level: int,
    context: MechanicContext | Mapping[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    binding = bind_current(frame, context=context)
    supported = bool(binding.get("supported"))
    return {
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "level": int(level),
        "supported": supported,
        "executable": supported,
        "roles": ["movable_sprite", "anchor_pixel", "active_selector"],
        "required_bindings": [
            "movable_sprite",
            "anchor_pixel",
            "active_selector",
            "variant_transition_model",
        ],
        "state_variables": [
            "sprite_pose",
            "sprite_variant",
            "anchor_world_multiset",
            "anchor_deficit",
        ],
        "operator": {
            "name": "select_and_transform_rigid_sprite",
            "transition": (
                "select one visible sprite, translate it one lattice cell, "
                "or advance its observed rigid variant"
            ),
        },
        "goal_relation": (
            "every anchor occurrence is co-located with another anchor "
            "from the same structural class"
        ),
        "invariants": [
            "unselected sprite poses and variants remain unchanged",
            "cardinal actions translate every selected-sprite cell rigidly",
            "variant actions preserve selected-sprite topology",
            "anchor identity is bound by frame role rather than palette value",
        ],
        "falsifiers": [
            "the logical occupancy differs from the predicted rigid union",
            "an unambiguous anchor-role cell changes class",
            "a static logical cell changes",
            "the current frame is not an exact prefix of the rebound plan",
        ],
        "current_binding": binding,
    }


def _pose(sprite: _Sprite) -> dict[str, Any]:
    return {
        "sprite_id": sprite.sprite_id,
        "position": [int(sprite.pos[0]), int(sprite.pos[1])],
        "size": [int(sprite.width), int(sprite.height)],
        "variant": int(sprite.variant),
        "world_cells": sorted(
            [int(x), int(y)] for x, y in sprite.world_cells()
        ),
        "world_anchors": sorted(
            [int(x), int(y)] for x, y in sprite.world_anchors()
        ),
    }


def predict_transition(
    frame: Any,
    action: Any,
    *,
    context: MechanicContext | Mapping[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    if not bind_current(frame, context=context).get("supported"):
        return {
            "family_id": FAMILY_ID,
            "accepted": False,
            "reason": "unbound_or_complete",
        }
    resolved = _resolve(frame, context)
    actual = _action(action)
    if resolved is None or resolved.plan_index >= len(resolved.assembly.plan):
        return {
            "family_id": FAMILY_ID,
            "accepted": False,
            "reason": "unbound_or_complete",
        }
    expected_action = resolved.assembly.plan[resolved.plan_index]
    if not _action_matches(expected_action, actual):
        return {
            "family_id": FAMILY_ID,
            "accepted": False,
            "reason": "action_is_not_current_certified_prefix",
        }
    next_world = _simulate(resolved.assembly, resolved.plan_index + 1)
    if next_world is None:
        return {
            "family_id": FAMILY_ID,
            "accepted": False,
            "reason": "rigid_simulation_failed",
        }
    prior_world = resolved.world
    prior_selected = prior_world.selected()
    next_selected = next_world.selected()
    if prior_selected is None or next_selected is None:
        return {
            "family_id": FAMILY_ID,
            "accepted": False,
            "reason": "selector_binding_lost",
        }
    prior_deficit = _anchor_deficit(prior_world)
    next_deficit = _anchor_deficit(next_world)
    projection = _projection(resolved.assembly, next_world)
    operation = (
        "select_entity"
        if actual["action"] == "ACTION6"
        else "variant_cycle"
        if actual["action"] == "ACTION5"
        else "cardinal_move"
    )
    prediction: dict[str, Any] = {
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "accepted": True,
        "reason": "exact_visible_rigid_prefix",
        "action": dict(expected_action),
        "operation": operation,
        "plan_index_before": int(resolved.plan_index),
        "plan_index_after": int(resolved.plan_index + 1),
        "remaining_actions_before": len(resolved.assembly.plan)
        - resolved.plan_index,
        "remaining_actions_after": len(resolved.assembly.plan)
        - resolved.plan_index
        - 1,
        "selected_sprite_before": prior_world.selected_id,
        "selected_sprite_after": next_world.selected_id,
        "selected_pose_before": _pose(prior_selected),
        "selected_pose_after": _pose(next_selected),
        "unselected_poses_before": [
            _pose(sprite)
            for sprite in prior_world.sprites
            if sprite.sprite_id != prior_world.selected_id
        ],
        "unselected_poses_after": [
            _pose(sprite)
            for sprite in next_world.sprites
            if sprite.sprite_id != next_world.selected_id
        ],
        "anchor_deficit_before": int(prior_deficit),
        "anchor_deficit_after": int(next_deficit),
        "anchor_deficit_reduction": int(prior_deficit - next_deficit),
        "goal_closure_predicted": next_deficit == 0,
        "expected_projection": projection,
    }
    prediction["certificate_digest"] = stable_digest(prediction)
    return prediction


def propose_plan(
    frame: Any,
    *,
    valid_actions: Sequence[Any] | set[Any] | None = None,
    context: MechanicContext | Mapping[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any] | None:
    if not bind_current(frame, context=context).get("supported"):
        return None
    resolved = _resolve(frame, context)
    if resolved is None or resolved.plan_index >= len(resolved.assembly.plan):
        return None
    action = resolved.assembly.plan[resolved.plan_index]
    if action["action"] not in _allowed(valid_actions):
        return None
    prediction = predict_transition(frame, action, context=context)
    if not prediction.get("accepted"):
        return None
    return {
        "mechanic_id": FAMILY_ID,
        "family_id": FAMILY_ID,
        "actions": [dict(action)],
        "certified": True,
        "certificate_reason": (
            "visible sprites bind to an exact rigid assembly prefix with "
            "anchor-role and static-cell verification"
        ),
        "transition_rule": (
            "apply one bound selection or rigid transform and recompute the "
            "same-class anchor deficit"
        ),
        "first_prediction": prediction,
        "progress_measure": {
            "metric": "anchor_deficit",
            "before": int(prediction["anchor_deficit_before"]),
            "expected_after": int(prediction["anchor_deficit_after"]),
            "strict": False,
            "after": int(prediction["anchor_deficit_after"]),
            "reduction": int(prediction["anchor_deficit_reduction"]),
            "remaining_actions_before": int(
                prediction["remaining_actions_before"]
            ),
            "remaining_actions_after": int(prediction["remaining_actions_after"]),
        },
        "grounded_observation": {
            "invariants": [
                "the predicted logical occupancy is exact",
                "unselected sprite poses remain fixed",
                "unambiguous anchor-role cells retain their class",
            ],
            "falsifiers": [
                "any occupancy, static-cell, or anchor-role mismatch",
            ],
        },
        "detail": {
            "private_plan_length": len(resolved.assembly.plan),
            "plan_cursor": int(resolved.plan_index),
            "sprite_count": len(resolved.world.sprites),
            "anchor_deficit": _anchor_deficit(resolved.world),
        },
    }


def verify_transition(expected: Any, current_frame: Any) -> bool:
    if not isinstance(expected, Mapping) or not bool(expected.get("accepted")):
        return False
    supplied_digest = str(expected.get("certificate_digest", "") or "")
    unsigned = dict(expected)
    unsigned.pop("certificate_digest", None)
    if not supplied_digest or stable_digest(unsigned) != supplied_digest:
        return False
    projection = expected.get("expected_projection")
    if not isinstance(projection, Mapping):
        return False
    if int(expected.get("plan_index_after", -1)) != int(
        expected.get("plan_index_before", -1)
    ) + 1:
        return False
    if int(expected.get("remaining_actions_after", -1)) != int(
        expected.get("remaining_actions_before", -1)
    ) - 1:
        return False
    if int(expected.get("anchor_deficit_reduction", 0)) != int(
        expected.get("anchor_deficit_before", 0)
    ) - int(expected.get("anchor_deficit_after", 0)):
        return False
    return _projection_matches(projection, current_frame)


__all__ = [
    "FAMILY_ID",
    "bind_current",
    "compile_schema",
    "predict_transition",
    "propose_plan",
    "verify_transition",
]
