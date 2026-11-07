"""Visible hierarchical slot-sequence execution with exact one-step leases.

The strict-generic slot reader is used as a mechanics oracle, not as an
executor.  A live board is rebound on every call.  Partially consumed pool
sites are projected back to an unfilled board only long enough to recover the
coordinate-free traversal grammar; the projection is accepted only when the
live occupancies form one unique, exact prefix of that grammar.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np

from arcgym.eca.primitive_synthesis import (
    _slot_pool_sequence_state_with_diagnostics,
    _valued_components,
)


FAMILY_ID = "hierarchical_slot_sequence"

_CLICK = "ACTION6"
_SUBMIT = "ACTION5"
_ALIASES = {
    "MOUSE": _CLICK,
    "SPACE": _SUBMIT,
}
_MAX_RECONSTRUCTION_CANDIDATES = 4096
_RECENT_GRAMMAR_KEYS: dict[str, tuple[Any, ...]] = {}
_PARSE_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_PARSE_CACHE_SIZE = 8


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


def _action(action: Any) -> dict[str, Any] | None:
    if isinstance(action, str):
        raw: Mapping[str, Any] = {"action": action}
    elif isinstance(action, Mapping):
        raw = action
    else:
        raw = {"action": getattr(action, "name", "")}
    name = str(raw.get("action", raw.get("action_name", "")) or "")
    name = _ALIASES.get(name.rsplit(".", 1)[-1].upper(), name.rsplit(".", 1)[-1].upper())
    if name not in {_CLICK, _SUBMIT}:
        return None
    out: dict[str, Any] = {"action": name}
    if name == _CLICK:
        try:
            out["x"] = int(raw.get("x", raw.get("col")))
            out["y"] = int(raw.get("y", raw.get("row")))
        except (TypeError, ValueError):
            return None
    return out


def _allowed(valid_actions: Sequence[Any] | set[Any] | None) -> set[str]:
    if valid_actions is None:
        return {_CLICK, _SUBMIT}
    names: set[str] = set()
    for item in valid_actions:
        normalized = _action(item)
        if normalized is not None:
            names.add(str(normalized["action"]))
            continue
        name = str(item or "").rsplit(".", 1)[-1].upper()
        name = _ALIASES.get(name, name)
        if name in {_CLICK, _SUBMIT}:
            names.add(name)
    return names


def _dominant_value(arr: np.ndarray) -> int:
    values, counts = np.unique(arr, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _palette_normalized(arr: np.ndarray) -> np.ndarray:
    """Replace palette values by stable first-appearance class identifiers."""

    flat = np.asarray(arr).reshape(-1)
    values, first_indices = np.unique(flat, return_index=True)
    ordered = values[np.argsort(first_indices)]
    out = np.empty(arr.shape, dtype=np.int32)
    for index, value in enumerate(ordered):
        out[arr == value] = -(index + 1)
    return out


def _separator_component(arr: np.ndarray) -> dict[str, int] | None:
    components = _valued_components(arr)
    candidates = [
        comp
        for comp in components
        if int(comp["height"]) <= 2
        and int(comp["width"]) >= max(12, int(arr.shape[1] * 0.3))
        and int(comp["area"]) >= int(comp["width"])
        and int(comp["width"]) >= 8 * max(1, int(comp["height"]))
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda comp: (
            int(comp["width"]),
            int(comp["cy"]),
            int(comp["area"]),
        ),
    )


def _local_view(frame: Any) -> dict[str, Any] | None:
    arr = _frame2d(frame)
    if arr is None:
        return None
    separator = _separator_component(arr)
    if separator is None:
        return None
    background = _dominant_value(arr)
    components = _valued_components(arr)
    upper = [
        comp
        for comp in components
        if int(comp["value"]) != background
        and int(comp["min_y"]) < int(separator["min_y"])
    ]
    if not upper:
        return None
    offset_y = min(int(comp["min_y"]) for comp in upper)
    local = np.asarray(arr[offset_y:, :])
    normalized = _palette_normalized(local)
    local_separator = _separator_component(normalized)
    if local_separator is None:
        return None
    return {
        "source": arr,
        "normalized": normalized,
        "offset_x": 0,
        "offset_y": int(offset_y),
        "separator": local_separator,
        "background": _dominant_value(normalized),
        "components": _valued_components(normalized),
    }


def _solid_square(component: Mapping[str, Any]) -> bool:
    width = int(component["width"])
    height = int(component["height"])
    area = int(component["area"])
    perimeter = 2 * width + 2 * height - 4
    return (
        width >= 3
        and height >= 3
        and abs(width - height) <= 1
        and perimeter <= area <= width * height
    )


def _mode_shape(components: Sequence[Mapping[str, Any]]) -> tuple[int, int] | None:
    if not components:
        return None
    counts = Counter(
        (int(component["width"]), int(component["height"]))
        for component in components
    )
    highest = max(counts.values())
    return min(
        (shape for shape, count in counts.items() if count == highest),
        key=lambda shape: (shape[0] * shape[1], shape),
    )


def _live_geometry(view: Mapping[str, Any]) -> dict[str, Any] | None:
    separator = view["separator"]
    separator_y = int(separator["cy"])
    separator_value = int(separator["value"])
    background = int(view["background"])
    components = list(view["components"])

    pool_components = [
        comp
        for comp in components
        if int(comp["min_y"]) > separator_y
        and int(comp["value"]) not in {background, separator_value}
        and _solid_square(comp)
    ]
    upper_solids = [
        comp
        for comp in components
        if int(comp["max_y"]) < separator_y
        and int(comp["value"]) not in {background, separator_value}
        and _solid_square(comp)
    ]
    shape = _mode_shape(pool_components)
    if shape is None:
        shape = _mode_shape(upper_solids)
    if shape is None:
        return None
    basket_width, basket_height = shape
    pools = [
        comp
        for comp in pool_components
        if (int(comp["width"]), int(comp["height"])) == shape
    ]

    top_candidates = [
        comp
        for comp in upper_solids
        if (int(comp["width"]), int(comp["height"])) == shape
    ]
    if not top_candidates:
        return None
    top_y = min(int(comp["cy"]) for comp in top_candidates)
    top_cut = top_y + max(basket_height + 1, int(round(basket_height * 1.5)))
    board_items = [
        comp
        for comp in top_candidates
        if int(comp["cy"]) > top_cut
    ]

    marker_components = [
        comp
        for comp in components
        if int(comp["value"]) == separator_value
        and int(comp["min_y"]) > separator_y
        and 1 < int(comp["width"]) < basket_width
        and 1 < int(comp["height"]) < basket_height
        and int(comp["area"]) == int(comp["width"]) * int(comp["height"])
    ]
    marker_shape = _mode_shape(marker_components)
    if marker_shape is None and marker_components:
        return None
    used_markers = [
        comp
        for comp in marker_components
        if (int(comp["width"]), int(comp["height"])) == marker_shape
    ]

    selected: dict[str, int] | None = None
    normalized = np.asarray(view["normalized"])
    selected_rows: list[dict[str, int]] = []
    for pool in pools:
        x0 = int(pool["min_x"]) - 1
        y0 = int(pool["min_y"]) - 1
        x1 = int(pool["max_x"]) + 1
        y1 = int(pool["max_y"]) + 1
        if x0 < 0 or y0 < 0 or x1 >= normalized.shape[1] or y1 >= normalized.shape[0]:
            continue
        patch = normalized[y0 : y1 + 1, x0 : x1 + 1]
        ring = np.concatenate(
            (patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1])
        )
        ring_values = np.unique(ring)
        if len(ring_values) == 1 and int(ring_values[0]) not in {
            background,
            int(pool["value"]),
        }:
            selected_rows.append(
                {
                    "color": int(pool["value"]),
                    "x": int(pool["cx"]),
                    "y": int(pool["cy"]),
                    "basket_kind": (
                        "solid"
                        if int(pool["area"])
                        == int(pool["width"]) * int(pool["height"])
                        else "transition"
                    ),
                }
            )
    if len(selected_rows) > 1:
        return None
    if selected_rows:
        selected = selected_rows[0]

    return {
        "basket_shape": [int(basket_width), int(basket_height)],
        "marker_shape": (
            [] if marker_shape is None else [int(marker_shape[0]), int(marker_shape[1])]
        ),
        "pools": [
            {
                "color": int(comp["value"]),
                "x": int(comp["cx"]),
                "y": int(comp["cy"]),
                "basket_kind": (
                    "solid"
                    if int(comp["area"]) == int(comp["width"]) * int(comp["height"])
                    else "transition"
                ),
                "bbox": [
                    int(comp["min_x"]),
                    int(comp["min_y"]),
                    int(comp["max_x"]),
                    int(comp["max_y"]),
                ],
            }
            for comp in sorted(
                pools,
                key=lambda item: (int(item["cx"]), int(item["cy"])),
            )
        ],
        "board_items": [dict(comp) for comp in board_items],
        "used_markers": [
            dict(comp)
            for comp in sorted(
                used_markers,
                key=lambda item: (int(item["cx"]), int(item["cy"])),
            )
        ],
        "selected": selected,
        "top_cut": int(top_cut),
    }


def _site(item: Mapping[str, Any]) -> tuple[int, int]:
    raw = item.get("slot_site")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 2:
        return int(raw[0]), int(raw[1])
    return int(item.get("x", 0)), int(item.get("y", 0))


def _transition_color(item: Mapping[str, Any]) -> int:
    return int(item.get("transition_color", item.get("color", 0)))


def _movable_bindings(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(binding)
        for binding in state.get("bindings", ())
        if isinstance(binding, Mapping) and str(binding.get("role", "")) != "fixed"
    ]


def _current_pool_signatures(geometry: Mapping[str, Any]) -> Counter[tuple[int, int, int]]:
    return Counter(
        (int(item["color"]), int(item["x"]), int(item["y"]))
        for item in geometry["pools"]
    )


def _candidate_matches_live_prefix(
    state: Mapping[str, Any],
    geometry: Mapping[str, Any],
    placed_items: Sequence[Mapping[str, Any]],
) -> bool:
    used_count = len(geometry["used_markers"])
    movable = _movable_bindings(state)
    if len(movable) != int(state.get("pool_count", -1)):
        return False
    if used_count > len(movable) or len(placed_items) != used_count:
        return False
    placed_by_site = {
        (int(item["cx"]), int(item["cy"])): dict(item)
        for item in placed_items
    }
    if len(placed_by_site) != len(placed_items):
        return False
    for index, binding in enumerate(movable):
        actual = placed_by_site.get(_site(binding))
        if index < used_count:
            if actual is None or int(actual["value"]) != int(
                binding.get("color", -10**9)
            ):
                return False
            if (
                str(binding.get("role", "")) == "fillable"
                and int(actual["area"])
                != int(actual["width"]) * int(actual["height"])
            ):
                return False
            if (
                str(binding.get("role", "")) == "transition_entry"
                and int(actual["area"])
                == int(actual["width"]) * int(actual["height"])
            ):
                return False
        elif actual is not None:
            return False

    full_pool = Counter(
        (
            int(item["color"]),
            int(item["x"]),
            int(item["y"]),
        )
        for item in state.get("pool_items", ())
        if isinstance(item, Mapping)
    )
    current_pool = _current_pool_signatures(geometry)
    if any(current_pool[key] > full_pool[key] for key in current_pool):
        return False
    full_colors = Counter(key[0] for key in full_pool.elements())
    current_colors = Counter(key[0] for key in current_pool.elements())
    consumed_colors = Counter(int(binding["color"]) for binding in movable[:used_count])
    return full_colors == current_colors + consumed_colors


def _restore_projection(
    view: Mapping[str, Any],
    geometry: Mapping[str, Any],
    placed_items: Sequence[Mapping[str, Any]],
) -> np.ndarray | None:
    marker_shape = list(geometry.get("marker_shape") or [])
    basket_shape = list(geometry.get("basket_shape") or [])
    if len(marker_shape) != 2 or len(basket_shape) != 2:
        return None
    marker_width, marker_height = marker_shape
    basket_width, basket_height = basket_shape
    if basket_width <= marker_width or basket_height <= marker_height:
        return None

    projected = _oracle_input(view, geometry)
    background = int(view["background"])
    separator_value = int(view["separator"]["value"])
    for item in placed_items:
        x0 = int(item["min_x"])
        y0 = int(item["min_y"])
        x1 = int(item["max_x"]) + 1
        y1 = int(item["max_y"]) + 1
        projected[y0:y1, x0:x1] = background
        inset_x = max(0, (int(item["width"]) - marker_width) // 2)
        inset_y = max(0, (int(item["height"]) - marker_height) // 2)
        projected[
            y0 + inset_y : y0 + inset_y + marker_height,
            x0 + inset_x : x0 + inset_x + marker_width,
        ] = separator_value

    ordered_items = sorted(
        placed_items,
        key=lambda item: (int(item["cy"]), int(item["cx"]), int(item["value"])),
    )
    for marker, item in zip(geometry["used_markers"], ordered_items):
        expand_x = max(0, (basket_width - int(marker["width"])) // 2)
        expand_y = max(0, (basket_height - int(marker["height"])) // 2)
        x0 = int(marker["min_x"]) - expand_x
        y0 = int(marker["min_y"]) - expand_y
        x1 = x0 + basket_width
        y1 = y0 + basket_height
        if x0 < 0 or y0 < 0 or x1 > projected.shape[1] or y1 > projected.shape[0]:
            return None
        projected[y0:y1, x0:x1] = int(item["value"])
    return projected


def _oracle_input(
    view: Mapping[str, Any], geometry: Mapping[str, Any]
) -> np.ndarray:
    """Remove the visible selection halo before invoking the slot oracle."""

    projected = np.asarray(view["normalized"]).copy()
    selected = geometry.get("selected")
    if not isinstance(selected, Mapping):
        return projected
    background = int(view["background"])
    for pool in geometry.get("pools", ()):
        if not isinstance(pool, Mapping):
            continue
        if (int(pool["x"]), int(pool["y"])) != (
            int(selected["x"]),
            int(selected["y"]),
        ):
            continue
        x0, y0, x1, y1 = [int(value) for value in pool["bbox"]]
        x0 -= 1
        y0 -= 1
        x1 += 1
        y1 += 1
        projected[y0, x0 : x1 + 1] = background
        projected[y1, x0 : x1 + 1] = background
        projected[y0 : y1 + 1, x0] = background
        projected[y0 : y1 + 1, x1] = background
        break
    return projected


def _state_semantic_key(state: Mapping[str, Any]) -> tuple[Any, ...]:
    units = tuple(
        (
            str(item.get("role", "")),
            _site(item),
            int(item.get("color", 0)),
            _transition_color(item),
            tuple(int(value) for value in item.get("reentry_colors", ())),
        )
        for item in state.get("bindings", ())
        if isinstance(item, Mapping)
    )
    return (
        str(state.get("slot_traversal_kind", "")),
        tuple(int(value) for value in state.get("sequence_colors", ())),
        units,
        tuple(
            sorted(
                (int(item["x"]), int(item["y"]))
                for item in state.get("pool_items", ())
                if isinstance(item, Mapping)
            )
        ),
    )


def _grammar_scope_key(
    view: Mapping[str, Any], geometry: Mapping[str, Any]
) -> str:
    """Scope ambiguity memory to the visible, static hierarchy header."""

    normalized = np.asarray(view["normalized"])
    top_cut = max(0, min(int(geometry["top_cut"]), normalized.shape[0] - 1))
    digest = hashlib.sha256()
    digest.update(str(normalized.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(normalized[: top_cut + 1]).tobytes())
    digest.update(str(tuple(geometry["basket_shape"])).encode("ascii"))
    return digest.hexdigest()


def _recover_oracle_state(
    view: Mapping[str, Any], geometry: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    normalized = _oracle_input(view, geometry)
    used_count = len(geometry["used_markers"])
    grammar_scope = _grammar_scope_key(view, geometry)
    if used_count == 0:
        state, _diagnostics = _slot_pool_sequence_state_with_diagnostics(normalized)
        if isinstance(state, Mapping) and _candidate_matches_live_prefix(state, geometry, []):
            key = _state_semantic_key(state)
            _RECENT_GRAMMAR_KEYS[grammar_scope] = key
            return dict(state), []
        return None

    candidates = list(geometry["board_items"])
    if used_count > len(candidates):
        return None
    combination_count = math.comb(len(candidates), used_count)
    if combination_count > _MAX_RECONSTRUCTION_CANDIDATES:
        return None

    accepted: dict[
        tuple[Any, ...],
        tuple[tuple[int, int, int, int], dict[str, Any], list[dict[str, Any]]],
    ] = {}
    for selected_indices in combinations(range(len(candidates)), used_count):
        placed = [dict(candidates[index]) for index in selected_indices]
        projected = _restore_projection(view, geometry, placed)
        if projected is None:
            continue
        state, _diagnostics = _slot_pool_sequence_state_with_diagnostics(projected)
        if not isinstance(state, Mapping):
            continue
        if not _candidate_matches_live_prefix(state, geometry, placed):
            continue
        key = _state_semantic_key(state)
        bindings = [
            item for item in state.get("bindings", ()) if isinstance(item, Mapping)
        ]
        rank = (
            len(bindings),
            int(str(state.get("slot_traversal_kind", "")) != "frame_graph_transition_tree"),
            sum(1 for item in bindings if str(item.get("role", "")) == "transition_entry"),
            sum(1 for item in bindings if str(item.get("role", "")) == "fixed"),
        )
        accepted.setdefault(key, (rank, dict(state), placed))
    if not accepted:
        return None
    best_rank = max(row[0] for row in accepted.values())
    best = [
        (key, row)
        for key, row in accepted.items()
        if row[0] == best_rank
    ]
    if len(best) != 1:
        remembered_key = _RECENT_GRAMMAR_KEYS.get(grammar_scope)
        remembered = [item for item in best if item[0] == remembered_key]
        if len(remembered) != 1:
            return None
        best = remembered
    key, (_rank, state, placed) = best[0]
    _RECENT_GRAMMAR_KEYS[grammar_scope] = key
    return state, placed


def _tokenizer(state: Mapping[str, Any]) -> dict[int, str]:
    colors: list[int] = [int(value) for value in state.get("sequence_colors", ())]
    for item in state.get("bindings", ()):
        if not isinstance(item, Mapping):
            continue
        colors.append(int(item.get("color", 0)))
        if item.get("transition_color") is not None:
            colors.append(int(item["transition_color"]))
        colors.extend(int(value) for value in item.get("reentry_colors", ()))
    tokens: dict[int, str] = {}
    for color in colors:
        if color not in tokens:
            tokens[color] = f"symbol_{len(tokens)}"
    return tokens


def _translate_site(site: tuple[int, int], view: Mapping[str, Any]) -> list[int]:
    return [
        int(site[0]) + int(view["offset_x"]),
        int(site[1]) + int(view["offset_y"]),
    ]


def _frame_graph(
    state: Mapping[str, Any],
    view: Mapping[str, Any],
    tokens: Mapping[int, str],
    basket_shape: Sequence[int],
) -> tuple[dict[str, Any], list[str]]:
    units = [dict(item) for item in state.get("bindings", ()) if isinstance(item, Mapping)]
    sites = [_site(item) for item in units]
    width, height = int(basket_shape[0]), int(basket_shape[1])
    background = int(view["background"])
    separator_value = int(view["separator"]["value"])
    containers: list[dict[str, Any]] = []
    for component in view["components"]:
        if int(component["value"]) in {background, separator_value}:
            continue
        if int(component["width"]) < width * 2 or int(component["height"]) < height * 2:
            continue
        contained = [
            index
            for index, (x, y) in enumerate(sites)
            if int(component["min_x"]) <= x <= int(component["max_x"])
            and int(component["min_y"]) <= y <= int(component["max_y"])
        ]
        if not contained:
            continue
        containers.append({"component": component, "unit_indices": contained})

    keys: list[tuple[Any, ...]] = []
    key_components: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for index, (x, y) in enumerate(sites):
        candidates = [
            row
            for row in containers
            if index in row["unit_indices"]
        ]
        if candidates:
            selected = min(
                candidates,
                key=lambda row: (
                    int(row["component"]["width"]) * int(row["component"]["height"]),
                    int(row["component"]["min_y"]),
                    int(row["component"]["min_x"]),
                ),
            )["component"]
            key = (
                "box",
                int(selected["min_x"]),
                int(selected["min_y"]),
                int(selected["max_x"]),
                int(selected["max_y"]),
            )
            key_components[key] = selected
        else:
            key = ("row", int(y))
        keys.append(key)

    key_to_id: dict[tuple[Any, ...], str] = {}
    frame_ids: list[str] = []
    for key in keys:
        if key not in key_to_id:
            key_to_id[key] = f"frame_{len(key_to_id)}"
        frame_ids.append(key_to_id[key])

    nodes: list[dict[str, Any]] = []
    for key, frame_id in key_to_id.items():
        indices = [index for index, candidate in enumerate(keys) if candidate == key]
        component = key_components.get(key)
        node: dict[str, Any] = {
            "frame_id": frame_id,
            "unit_indices": indices,
        }
        if component is not None:
            node["bounds"] = [
                int(component["min_x"]) + int(view["offset_x"]),
                int(component["min_y"]) + int(view["offset_y"]),
                int(component["max_x"]) + int(view["offset_x"]),
                int(component["max_y"]) + int(view["offset_y"]),
            ]
            color = int(component["value"])
            if color in tokens:
                node["frame_symbol"] = tokens[color]
        else:
            node["row"] = int(key[1]) + int(view["offset_y"])
        nodes.append(node)

    root = frame_ids[0] if frame_ids else ""
    frame_symbol_to_id = {
        str(node["frame_symbol"]): str(node["frame_id"])
        for node in nodes
        if node.get("frame_symbol")
    }
    repeated_targets: dict[str, str] = {}
    edges: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        if str(unit.get("role", "")) != "transition_entry":
            continue
        source = frame_ids[index]
        transition_token = tokens.get(_transition_color(unit), "")
        target = frame_symbol_to_id.get(transition_token, "")
        if not target and transition_token in repeated_targets:
            target = repeated_targets[transition_token]
        if not target and index + 1 < len(frame_ids) and frame_ids[index + 1] != source:
            target = frame_ids[index + 1]
        reentry = [tokens[int(value)] for value in unit.get("reentry_colors", ())]
        if not target and reentry and source != root:
            target = root
        if not target:
            target = source
        if transition_token and target != source:
            repeated_targets.setdefault(transition_token, target)
        edges.append(
            {
                "source": source,
                "target": target,
                "unit_index": int(index),
                "transition_symbol": transition_token,
                "reentry_symbols": reentry,
            }
        )
    return {
        "root": root,
        "nodes": nodes,
        "edges": edges,
    }, frame_ids


def _path_to_frame(graph: Mapping[str, Any], target: str) -> list[str]:
    root = str(graph.get("root", ""))
    if not root or not target:
        return []
    queue: deque[tuple[str, list[str]]] = deque([(root, [root])])
    seen = {root}
    adjacency: defaultdict[str, list[str]] = defaultdict(list)
    for edge in graph.get("edges", ()):
        if not isinstance(edge, Mapping):
            continue
        source = str(edge.get("source", ""))
        destination = str(edge.get("target", ""))
        if source and destination and source != destination:
            adjacency[source].append(destination)
    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        for neighbor in adjacency[node]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    return [target]


def _static_render_digest(
    view: Mapping[str, Any],
    state: Mapping[str, Any],
    basket_shape: Sequence[int],
) -> str:
    masked = np.asarray(view["normalized"]).copy()
    sentinel = 0
    separator_y = int(view["separator"]["cy"])
    masked[max(0, separator_y - 1) :, :] = sentinel
    width, height = int(basket_shape[0]), int(basket_shape[1])
    sites = [_site(binding) for binding in state.get("bindings", ()) if isinstance(binding, Mapping)]
    background = int(view["background"])
    for component in view["components"]:
        if int(component["value"]) == background:
            continue
        if int(component["width"]) < width * 2 or int(component["height"]) < height * 2:
            continue
        if not any(
            int(component["min_x"]) <= x <= int(component["max_x"])
            and int(component["min_y"]) <= y <= int(component["max_y"])
            for x, y in sites
        ):
            continue
        masked[
            int(component["min_y"]) : int(component["max_y"]) + 1,
            int(component["min_x"]) : int(component["max_x"]) + 1,
        ] = sentinel
    for binding in _movable_bindings(state):
        x, y = _site(binding)
        x0 = max(0, x - width)
        x1 = min(masked.shape[1], x + width + 1)
        y0 = max(0, y - height)
        y1 = min(masked.shape[0], y + height + 1)
        masked[y0:y1, x0:x1] = sentinel
    payload = np.ascontiguousarray(masked).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _full_frame_digest(view: Mapping[str, Any]) -> str:
    normalized = np.asarray(view["normalized"])
    return hashlib.sha256(np.ascontiguousarray(normalized).tobytes()).hexdigest()


def _public_unit(
    binding: Mapping[str, Any],
    *,
    index: int,
    frame_id: str,
    view: Mapping[str, Any],
    tokens: Mapping[int, str],
) -> dict[str, Any]:
    role = str(binding.get("role", "fillable") or "fillable")
    unit = {
        "unit_index": int(index),
        "role": role,
        "frame_id": frame_id,
        "site": _translate_site(_site(binding), view),
        "symbol": tokens[int(binding.get("color", 0))],
    }
    if role == "transition_entry":
        unit["transition_symbol"] = tokens[_transition_color(binding)]
        unit["reentry_symbols"] = [
            tokens[int(value)] for value in binding.get("reentry_colors", ())
        ]
    return unit


def _reference_roles(
    state: Mapping[str, Any],
    view: Mapping[str, Any],
    geometry: Mapping[str, Any],
    tokens: Mapping[int, str],
) -> list[dict[str, Any]] | None:
    sequence = [int(value) for value in state.get("sequence_colors", ())]
    basket_width, basket_height = [int(value) for value in geometry["basket_shape"]]
    candidates = [
        component
        for component in view["components"]
        if int(component["value"]) in tokens
        and int(component["cy"]) <= int(geometry["top_cut"])
        and basket_width <= int(component["width"]) <= basket_width * 2
        and basket_height <= int(component["height"]) <= basket_height * 2
    ]
    candidates.sort(key=lambda item: (int(item["cy"]), int(item["cx"])))
    if [int(item["value"]) for item in candidates] != sequence:
        return None
    return [
        {
            "role": "reference_symbol",
            "index": int(index),
            "symbol": tokens[int(component["value"])],
            "site": _translate_site(
                (int(component["cx"]), int(component["cy"])),
                view,
            ),
            "bounds": [
                int(component["min_x"]) + int(view["offset_x"]),
                int(component["min_y"]) + int(view["offset_y"]),
                int(component["max_x"]) + int(view["offset_x"]),
                int(component["max_y"]) + int(view["offset_y"]),
            ],
        }
        for index, component in enumerate(candidates)
    ]


def _dynamic_projection(parsed: Mapping[str, Any], completed: int) -> dict[str, Any]:
    units = list(parsed["units"])
    movable_seen = 0
    emitted: list[str] = []
    occupied: list[dict[str, Any]] = []
    next_unit_index: int | None = None
    for index, unit in enumerate(units):
        role = str(unit["role"])
        if role == "fixed":
            if movable_seen <= completed:
                emitted.append(str(unit["symbol"]))
            continue
        if movable_seen >= completed:
            next_unit_index = index
            break
        occupied.append(
            {
                "unit_index": int(index),
                "role": role,
                "site": list(unit["site"]),
                "symbol": (
                    str(unit.get("transition_symbol", unit["symbol"]))
                    if role == "transition_entry"
                    else str(unit["symbol"])
                ),
            }
        )
        if role == "fillable":
            emitted.append(str(unit["symbol"]))
        else:
            emitted.extend(str(value) for value in unit.get("reentry_symbols", ()))
        movable_seen += 1
    else:
        next_unit_index = None

    reference = list(parsed["reference_sequence"])
    if reference[: len(emitted)] != emitted:
        return {"valid": False}
    remaining = len(parsed["movable_units"]) - completed
    next_frame = (
        str(units[next_unit_index]["frame_id"])
        if next_unit_index is not None
        else str(parsed["frame_graph"].get("root", ""))
    )
    return {
        "valid": True,
        "completed_placements": int(completed),
        "remaining_placements": int(remaining),
        "occupied_slots": occupied,
        "emitted_sequence": emitted,
        "sequence_residual": reference[len(emitted) :],
        "next_unit_index": next_unit_index,
        "frame_stack": _path_to_frame(parsed["frame_graph"], next_frame),
    }


def _parse_uncached(frame: Any) -> dict[str, Any] | None:
    view = _local_view(frame)
    if view is None:
        return None
    geometry = _live_geometry(view)
    if geometry is None:
        return None
    recovered = _recover_oracle_state(view, geometry)
    if recovered is None:
        return None
    state, placed_items = recovered
    tokens = _tokenizer(state)
    sequence = [tokens[int(value)] for value in state.get("sequence_colors", ())]
    reference_roles = _reference_roles(state, view, geometry, tokens)
    if reference_roles is None:
        return None
    frame_graph, frame_ids = _frame_graph(
        state,
        view,
        tokens,
        geometry["basket_shape"],
    )
    bindings = [dict(item) for item in state.get("bindings", ()) if isinstance(item, Mapping)]
    if len(frame_ids) != len(bindings):
        return None
    units = [
        _public_unit(
            binding,
            index=index,
            frame_id=frame_ids[index],
            view=view,
            tokens=tokens,
        )
        for index, binding in enumerate(bindings)
    ]
    movable_units = [unit for unit in units if unit["role"] != "fixed"]
    completed = len(geometry["used_markers"])

    public_pools = [
        {
            "role": "basket",
            "symbol": tokens.get(int(item["color"]), ""),
            "basket_kind": str(item["basket_kind"]),
            "site": _translate_site((int(item["x"]), int(item["y"])), view),
        }
        for item in geometry["pools"]
    ]
    if any(not item["symbol"] for item in public_pools):
        return None
    selected_raw = geometry.get("selected")
    selected = None
    if isinstance(selected_raw, Mapping):
        symbol = tokens.get(int(selected_raw["color"]), "")
        if not symbol:
            return None
        selected = {
            "role": "selection",
            "symbol": symbol,
            "basket_kind": str(selected_raw["basket_kind"]),
            "site": _translate_site(
                (int(selected_raw["x"]), int(selected_raw["y"])),
                view,
            ),
        }

    pool_sites = sorted(
        _translate_site((int(item["x"]), int(item["y"])), view)
        for item in state.get("pool_items", ())
        if isinstance(item, Mapping)
    )
    pool_symbols = sorted(
        tokens[int(item["color"])]
        for item in state.get("pool_items", ())
        if isinstance(item, Mapping)
    )
    static_signature = {
        "traversal_kind": str(state.get("slot_traversal_kind", "")),
        "reference_sequence": sequence,
        "reference_roles": reference_roles,
        "frame_graph": frame_graph,
        "units": units,
        "pool_sites": pool_sites,
        "pool_symbol_multiset": pool_symbols,
        "static_render_digest": _static_render_digest(
            view,
            state,
            geometry["basket_shape"],
        ),
    }
    parsed: dict[str, Any] = {
        "view": view,
        "state": state,
        "tokens": tokens,
        "reference_sequence": sequence,
        "reference_roles": reference_roles,
        "frame_graph": frame_graph,
        "units": units,
        "movable_units": movable_units,
        "available_baskets": public_pools,
        "selected_basket": selected,
        "completed_placements": completed,
        "static_signature": static_signature,
        "placed_item_count": len(placed_items),
    }
    dynamic = _dynamic_projection(parsed, completed)
    if not dynamic.get("valid"):
        return None
    parsed["dynamic"] = dynamic
    return parsed


def _parse(frame: Any) -> dict[str, Any] | None:
    arr = _frame2d(frame)
    if arr is None:
        return None
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(np.ascontiguousarray(arr).tobytes())
    key = digest.hexdigest()
    cached = _PARSE_CACHE.get(key)
    if cached is not None:
        _PARSE_CACHE.move_to_end(key)
        return cached
    parsed = _parse_uncached(arr)
    if parsed is None:
        return None
    _PARSE_CACHE[key] = parsed
    _PARSE_CACHE.move_to_end(key)
    while len(_PARSE_CACHE) > _PARSE_CACHE_SIZE:
        _PARSE_CACHE.popitem(last=False)
    return parsed


def _remaining_action_count(parsed: Mapping[str, Any]) -> int:
    dynamic = parsed["dynamic"]
    remaining = int(dynamic["remaining_placements"])
    if remaining == 0:
        return 1
    next_index = int(dynamic["next_unit_index"])
    next_unit = parsed["units"][next_index]
    expected_symbol = (
        str(next_unit.get("transition_symbol", ""))
        if next_unit["role"] == "transition_entry"
        else str(next_unit["symbol"])
    )
    expected_kind = (
        "transition" if next_unit["role"] == "transition_entry" else "solid"
    )
    selected = parsed.get("selected_basket")
    selection_ready = bool(
        isinstance(selected, Mapping)
        and str(selected.get("symbol", "")) == expected_symbol
        and str(selected.get("basket_kind", "")) == expected_kind
    )
    return remaining * 2 + 1 - int(selection_ready)


def _next_action(parsed: Mapping[str, Any]) -> tuple[dict[str, Any], str] | None:
    dynamic = parsed["dynamic"]
    if int(dynamic["remaining_placements"]) == 0:
        if list(dynamic["emitted_sequence"]) != list(parsed["reference_sequence"]):
            return None
        return {"action": _SUBMIT}, "submit_configuration"

    next_index = dynamic.get("next_unit_index")
    if not isinstance(next_index, int):
        return None
    unit = parsed["units"][next_index]
    desired_symbol = (
        str(unit.get("transition_symbol", ""))
        if unit["role"] == "transition_entry"
        else str(unit["symbol"])
    )
    selected = parsed.get("selected_basket")
    desired_kind = "transition" if unit["role"] == "transition_entry" else "solid"
    if (
        isinstance(selected, Mapping)
        and str(selected.get("symbol", "")) == desired_symbol
        and str(selected.get("basket_kind", "")) == desired_kind
    ):
        x, y = [int(value) for value in unit["site"]]
        return {"action": _CLICK, "x": x, "y": y}, "place_one_basket"

    candidates = [
        item
        for item in parsed["available_baskets"]
        if str(item.get("symbol", "")) == desired_symbol
        and str(item.get("basket_kind", ""))
        == desired_kind
    ]
    if not candidates:
        return None
    chosen = min(candidates, key=lambda item: (int(item["site"][0]), int(item["site"][1])))
    x, y = [int(value) for value in chosen["site"]]
    return {"action": _CLICK, "x": x, "y": y}, "select_basket"


def _prediction(parsed: Mapping[str, Any], action: Mapping[str, Any], kind: str) -> dict[str, Any]:
    before_dynamic = dict(parsed["dynamic"])
    before_count = _remaining_action_count(parsed)
    base: dict[str, Any] = {
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "accepted": True,
        "transition_kind": kind,
        "action": dict(action),
        "static_signature": parsed["static_signature"],
        "before_dynamic": before_dynamic,
        "remaining_actions_before": int(before_count),
        "remaining_actions_after": max(0, int(before_count) - 1),
    }
    if kind == "select_basket":
        selected = next(
            item
            for item in parsed["available_baskets"]
            if list(item["site"]) == [int(action["x"]), int(action["y"])]
        )
        base["expected_selected_basket"] = {
            **dict(selected),
            "role": "selection",
        }
        base["expected_dynamic"] = before_dynamic
        base["expected_available_baskets"] = parsed["available_baskets"]
    elif kind == "place_one_basket":
        after_dynamic = _dynamic_projection(
            parsed,
            int(parsed["completed_placements"]) + 1,
        )
        selected = parsed["selected_basket"]
        base["placed_basket"] = selected
        base["expected_selected_basket"] = None
        base["expected_dynamic"] = after_dynamic
        base["expected_available_baskets"] = [
            item
            for item in parsed["available_baskets"]
            if list(item["site"]) != list(selected["site"])
        ]
    else:
        view = parsed["view"]
        base.update(
            {
                "requires_level_or_score_transition": True,
                "expected_score_delta": 1,
                "before_frame_digest": _full_frame_digest(view),
                "expected_outcome": "fresh_hierarchy_or_terminal_acknowledgement",
            }
        )
    return base


def bind_current(frame: Any, **_kwargs: Any) -> dict[str, Any]:
    """Bind reference, frame graph, live baskets, selection, and residual."""

    parsed = _parse(frame)
    if parsed is None:
        return {
            "family_id": FAMILY_ID,
            "mechanic_id": FAMILY_ID,
            "status": "unbound",
            "supported": False,
            "roles": [],
            "role_bindings": [],
            "reason": "visible_hierarchy_not_uniquely_reconstructible",
        }
    roles: list[dict[str, Any]] = [
        {
            "role": "reference_sequence",
            "symbols": list(parsed["reference_sequence"]),
            "sites": [list(item["site"]) for item in parsed["reference_roles"]],
        }
    ]
    roles.extend(dict(item) for item in parsed["reference_roles"])
    roles.extend(
        {"role": "frame", **dict(node)}
        for node in parsed["frame_graph"]["nodes"]
    )
    roles.extend(
        {
            "role": "slot",
            **dict(unit),
            "occupancy": next(
                (
                    item["symbol"]
                    for item in parsed["dynamic"]["occupied_slots"]
                    if int(item["unit_index"]) == int(unit["unit_index"])
                ),
                None,
            ),
        }
        for unit in parsed["units"]
    )
    roles.extend(dict(item) for item in parsed["available_baskets"])
    if parsed["selected_basket"] is not None:
        roles.append(dict(parsed["selected_basket"]))
    return {
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "status": "bound",
        "supported": True,
        "roles": roles,
        "role_bindings": roles,
        "reference_sequence": list(parsed["reference_sequence"]),
        "reference_roles": list(parsed["reference_roles"]),
        "frame_graph": parsed["frame_graph"],
        "traversal_kind": parsed["static_signature"]["traversal_kind"],
        "selected_basket": parsed["selected_basket"],
        "slot_occupancy": list(parsed["dynamic"]["occupied_slots"]),
        "frame_stack": list(parsed["dynamic"]["frame_stack"]),
        "emitted_sequence": list(parsed["dynamic"]["emitted_sequence"]),
        "residual": {
            "sequence": list(parsed["dynamic"]["sequence_residual"]),
            "placements": int(parsed["dynamic"]["remaining_placements"]),
            "next_unit_index": parsed["dynamic"]["next_unit_index"],
        },
        "static_signature": parsed["static_signature"],
    }


def compile_schema(frame: Any, *, level: int, **_kwargs: Any) -> dict[str, Any]:
    """Compile the transferable, frame-only hierarchy schema."""

    binding = bind_current(frame)
    supported = bool(binding.get("supported"))
    return {
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "schema_id": "tufa.hierarchical_slot_sequence.v1",
        "level": int(level),
        "supported": supported,
        "roles": list(binding.get("roles") or []),
        "required_bindings": [
            "reference_sequence",
            "frame",
            "slot",
            "basket",
            "frame_traversal_model",
        ],
        "state_variables": [
            "slot_occupancy",
            "selected_basket",
            "frame_stack",
            "emitted_sequence",
            "sequence_residual",
        ],
        "operator": {
            "name": "select_place_or_submit_current_hierarchy_step",
            "actions": [_CLICK, _SUBMIT],
            "transition": (
                "select one visible basket, occupy exactly one traversal slot, "
                "or submit an exactly completed hierarchy"
            ),
        },
        "goal": "depth-first frame traversal emits the visible reference sequence",
        "invariants": [
            "reference sequence and visible frame graph remain fixed during placement",
            "fixed slot contents remain fixed",
            "each placement consumes one pool basket and occupies one traversal unit",
            "frame transitions preserve stack order and re-entry residuals",
        ],
        "falsifiers": [
            "the hierarchy has more than one valid reverse projection",
            "an occupied slot is not an exact traversal prefix",
            "selection changes slot occupancy",
            "placement changes anything except one selected basket and one slot",
            "submit is requested before emitted sequence equals the reference",
        ],
        "current_binding": binding,
    }


def propose_plan(
    frame: Any,
    *,
    valid_actions: Sequence[Any] | set[Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any] | None:
    """Propose only the next currently grounded hierarchy action."""

    parsed = _parse(frame)
    if parsed is None:
        return None
    next_step = _next_action(parsed)
    if next_step is None:
        return None
    action, kind = next_step
    if str(action["action"]) not in _allowed(valid_actions):
        return None
    prediction = _prediction(parsed, action, kind)
    before = _remaining_action_count(parsed)
    return {
        "family_id": FAMILY_ID,
        "mechanic_id": FAMILY_ID,
        "actions": [action],
        "certified": True,
        "certificate_reason": "unique visible hierarchy prefix with exact one-step rebound",
        "transition_rule": "select_place_or_submit_current_hierarchy_step",
        "requires_transition_verification": True,
        "progress_measure": {
            "metric": "hierarchical_action_residual",
            "before": int(before),
            "expected_after": max(0, int(before) - 1),
            "strict": True,
        },
        "first_prediction": prediction,
        "grounded_observation": {
            "invariants": [
                "reference and frame graph stay fixed before submit",
                "live occupancy remains an exact traversal prefix",
            ],
            "falsifiers": [
                "selection or occupancy differs from the predicted one-step delta",
                "fresh parsing yields an ambiguous or non-prefix hierarchy",
            ],
        },
        "detail": {
            "traversal_kind": parsed["static_signature"]["traversal_kind"],
            "next_transition_kind": kind,
            "private_plan_length": int(before),
            "remaining_placements": int(parsed["dynamic"]["remaining_placements"]),
            "frame_stack": list(parsed["dynamic"]["frame_stack"]),
        },
    }


def predict_transition(frame: Any, action: Any, **_kwargs: Any) -> dict[str, Any]:
    """Predict one exact selection, placement, or guarded submit transition."""

    parsed = _parse(frame)
    normalized = _action(action)
    if parsed is None:
        return {
            "family_id": FAMILY_ID,
            "accepted": False,
            "reason": "visible_hierarchy_not_uniquely_reconstructible",
        }
    next_step = _next_action(parsed)
    if normalized is None or next_step is None or normalized != next_step[0]:
        return {
            "family_id": FAMILY_ID,
            "accepted": False,
            "reason": "action_is_not_the_current_hierarchy_prefix",
        }
    return _prediction(parsed, normalized, next_step[1])


def verify_transition(expected: Any, current_frame: Any) -> bool:
    """Rebind the next frame and reject every non-exact semantic delta."""

    if (
        not isinstance(expected, Mapping)
        or expected.get("family_id") != FAMILY_ID
        or not bool(expected.get("accepted"))
    ):
        return False
    kind = str(expected.get("transition_kind", ""))
    if kind == "submit_configuration":
        view = _local_view(current_frame)
        if view is None:
            return False
        if _full_frame_digest(view) == str(expected.get("before_frame_digest", "")):
            return False
        parsed = _parse(current_frame)
        if parsed is None:
            return True
        return bool(
            int(parsed["completed_placements"]) == 0
            and parsed.get("selected_basket") is None
            and parsed["static_signature"] != expected.get("static_signature")
        )

    parsed = _parse(current_frame)
    if parsed is None:
        return False
    if parsed["static_signature"] != expected.get("static_signature"):
        return False
    if parsed["dynamic"] != expected.get("expected_dynamic"):
        return False
    if parsed.get("selected_basket") != expected.get("expected_selected_basket"):
        return False
    if parsed["available_baskets"] != expected.get("expected_available_baskets"):
        return False
    try:
        return _remaining_action_count(parsed) == int(expected["remaining_actions_after"])
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "FAMILY_ID",
    "bind_current",
    "compile_schema",
    "predict_transition",
    "propose_plan",
    "verify_transition",
]
