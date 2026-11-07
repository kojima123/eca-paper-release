"""Visible, palette-invariant rule-table rewrite mechanic.

The mechanic reads an ordered premise, consequence, and visible rewrite
entries from a frame.  It deliberately exposes only one selector move or one
modular symbol rotation at a time.  Transition verification reparses the next
frame and fails closed unless the bound operation is the only semantic change.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from arcgym.eca.primitive_synthesis import (
    RULE_TABLE_CYCLE_MODULUS,
    _rule_table_composed_desired_bottom_signatures,
    _rule_table_desired_bottom_signatures,
    _rule_table_entry_slots_from_visible_chains,
    _rule_table_glyph_candidates,
    _rule_table_glyph_rows,
    _rule_table_lower_rows,
    _rule_table_rotation_slots_from_deltas,
    _rule_table_rule_chains_from_visible,
    _rule_table_selected_entry_index,
    _rule_table_selected_slot_index,
    _rule_table_short_rotation_steps,
    _rule_table_solve_alter_rules_by_signatures,
    _rule_table_solve_tree_alter_rules_by_signatures,
    _rule_table_visible_rule_chains,
    _single_foreground_color,
    _single_row_category,
)
from arcgym.eca.tufa_mechanics.base import MechanicContext, stable_digest


FAMILY_ID = "rule_table_rewrite"
_ROTATION_DELTAS = {"ACTION1": -1, "ACTION2": 1}
_SELECTOR_DELTAS = {"ACTION3": -1, "ACTION4": 1}
_MODEL_TO_ENGINE_ACTION = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
}
_PROBE_WEIGHT = 100
_PROBE_TAIL = _PROBE_WEIGHT - 1


def _action_name(action: Any) -> str:
    if isinstance(action, Mapping):
        value = action.get("action", action.get("action_name", ""))
    elif isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
        value = action[0] if action else ""
    else:
        value = action
    name = getattr(value, "name", value)
    text = str(name or "").upper()
    normalized = text.rsplit(".", 1)[-1].split("(", 1)[0].strip()
    return _MODEL_TO_ENGINE_ACTION.get(normalized, normalized)


def _available_actions(valid_actions: Iterable[Any] | None) -> set[str]:
    if valid_actions is None:
        return set(_ROTATION_DELTAS) | set(_SELECTOR_DELTAS)
    return {name for item in valid_actions if (name := _action_name(item))}


def _history_records(value: Any) -> list[tuple[np.ndarray, str]]:
    records: list[tuple[np.ndarray, str]] = []
    if isinstance(value, Mapping):
        history = value.get("history")
        if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
            for item in history:
                if isinstance(item, Mapping) and item.get("frame") is not None:
                    arr = _as_2d(item["frame"])
                    if arr is not None:
                        records.append((arr, _action_name(item.get("action", ""))))
        frames = value.get("frames")
        if not records and frames is not None:
            array = np.asarray(frames)
            if array.ndim == 3:
                records.extend((np.asarray(part), "") for part in array)
        current = _as_2d(value.get("frame")) if value.get("frame") is not None else None
        if current is not None and (
            not records or not np.array_equal(records[-1][0], current)
        ):
            records.append((current, _action_name(value.get("action", ""))))
        return records
    array = np.asarray(value)
    if array.ndim == 3:
        return [(np.asarray(part), "") for part in array]
    current = _as_2d(array)
    return [(current, "")] if current is not None else []


def _as_2d(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2 or arr.shape[0] < 48 or arr.shape[1] < 48:
        return None
    return arr


def _role_map(
    visible_chains: list[dict[str, Any]],
    premise_bg: int,
    consequence_bg: int,
) -> dict[int, str]:
    backgrounds: list[int] = []
    for chain in visible_chains:
        for key in ("left_bg", "right_bg"):
            background = int(chain[key])
            if (
                background not in {int(premise_bg), int(consequence_bg)}
                and background not in backgrounds
            ):
                backgrounds.append(background)
    roles = {int(premise_bg): "premise", int(consequence_bg): "consequence"}
    roles.update({bg: f"intermediate_{index}" for index, bg in enumerate(backgrounds)})
    return roles


def _semantic_visible_chains(
    visible_chains: Iterable[Mapping[str, Any]],
    roles: Mapping[int, str],
) -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    for raw in visible_chains:
        chain = dict(raw)
        left_role = roles.get(int(chain.get("left_bg", -1)), "unknown")
        right_role = roles.get(int(chain.get("right_bg", -1)), "unknown")
        # Adjacent runs between displayed rules point backwards through the
        # role graph. They are visual boundaries, not rewrite arrows.
        if (
            left_role in {"unknown", "consequence"}
            or right_role in {"unknown", "premise"}
            or left_role == right_role
        ):
            continue
        left = tuple(str(symbol) for symbol in chain.get("left") or ())
        right = tuple(str(symbol) for symbol in chain.get("right") or ())
        if not left or not right:
            continue
        chain["left"] = left
        chain["right"] = right
        chain["left_role"] = left_role
        chain["right_role"] = right_role
        chains.append(chain)
    return chains


def _rule_entry_mode(
    entries: Sequence[Mapping[str, Any]],
    premise_count: int,
    consequence_count: int,
) -> str:
    if entries and all(
        entry.get("left_role") == "premise" and entry.get("right_role") == "consequence"
        for entry in entries
    ):
        return "direct_alter_rules"

    intermediate_roles = _ordered_cycle(
        role
        for entry in entries
        for role in (entry.get("left_role"), entry.get("right_role"))
        if str(role).startswith("intermediate_")
    )
    for intermediate in intermediate_roles:
        outer = [
            entry
            for entry in entries
            if entry.get("left_role") == "premise"
            and entry.get("right_role") == intermediate
            and len(entry.get("left_symbols") or ()) == 1
            and len(entry.get("right_symbols") or ()) == 2
        ]
        leaves = [
            entry
            for entry in entries
            if entry.get("left_role") == intermediate
            and entry.get("right_role") == "consequence"
            and len(entry.get("left_symbols") or ()) == 1
            and len(entry.get("right_symbols") or ()) == 1
        ]
        if (
            len(outer) == int(premise_count)
            and len(leaves) == int(premise_count)
            and len(entries) == len(outer) + len(leaves)
            and int(consequence_count) == 2 * int(premise_count)
        ):
            return "tree_alter_rules"
    return ""


def _symbols(items: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(item["signature"]) for item in items]


def _centers(items: Iterable[Mapping[str, Any]]) -> list[list[int]]:
    return [[int(item["x"]) + 3, int(item["y"]) + 3] for item in items]


def _ordered_cycle(symbols: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    for symbol in symbols:
        text = str(symbol)
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def _complete_cycles(pairs: Iterable[Sequence[str]]) -> list[list[str]]:
    rows = [row for row in pairs if isinstance(row, Sequence) and len(row) == 2]
    forward: dict[str, str] = {}
    for row in rows:
        source, target = str(row[0]), str(row[1])
        if not source or not target or source == target:
            continue
        previous = forward.get(source)
        if previous is not None and previous != target:
            return []
        forward[source] = target
    cycles: list[list[str]] = []
    consumed: set[str] = set()
    for start in sorted(forward):
        if start in consumed:
            continue
        current = start
        component: list[str] = []
        for _ in range(RULE_TABLE_CYCLE_MODULUS):
            if current in component or current not in forward:
                component = []
                break
            component.append(current)
            current = forward[current]
        if (
            len(component) == RULE_TABLE_CYCLE_MODULUS
            and current == start
            and len(set(component)) == RULE_TABLE_CYCLE_MODULUS
        ):
            canonical_start = min(component)
            offset = component.index(canonical_start)
            cycle = component[offset:] + component[:offset]
            cycles.append(cycle)
            consumed.update(component)
    return sorted(cycles, key=lambda cycle: cycle[0])


def _global_cycles_from_roles(role_cycles: Mapping[str, Any]) -> list[list[str]]:
    pairs: list[Sequence[str]] = []
    for rows in role_cycles.values():
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            pairs.extend(row for row in rows if isinstance(row, Sequence))
    return _complete_cycles(pairs)


def _parse(value: Any) -> dict[str, Any] | None:
    records = _history_records(value)
    if not records:
        return None
    arr = records[-1][0]
    glyphs = _rule_table_glyph_candidates(arr)
    if len(glyphs) < 10:
        return None
    rows = _rule_table_glyph_rows(glyphs)
    premise_row, consequence_row = _rule_table_lower_rows(rows)
    if premise_row is None or consequence_row is None:
        return None
    premise_items = list(premise_row["items"])
    consequence_items = list(consequence_row["items"])
    premise_bg = _single_row_category(premise_items)
    consequence_bg = _single_row_category(consequence_items)
    foreground = _single_foreground_color(premise_items + consequence_items)
    if (
        premise_bg is None
        or consequence_bg is None
        or foreground is None
        or premise_bg == consequence_bg
        or not 3 <= len(premise_items) <= 8
        or not 3 <= len(consequence_items) <= 10
    ):
        return None

    visible = _rule_table_visible_rule_chains(rows, lower_top_y=int(premise_row["y"]))
    roles = _role_map(visible, premise_bg, consequence_bg)
    semantic_chains = _semantic_visible_chains(visible, roles)
    entries: list[dict[str, Any]] = []
    for index, chain in enumerate(semantic_chains):
        left = [str(symbol) for symbol in chain.get("left") or ()]
        right = [str(symbol) for symbol in chain.get("right") or ()]
        entries.append(
            {
                "index": int(index),
                "left_role": str(chain["left_role"]),
                "right_role": str(chain["right_role"]),
                "left_symbols": left,
                "right_symbols": right,
                "left_centers": _centers(chain.get("left_items") or ()),
                "right_centers": _centers(chain.get("right_items") or ()),
            }
        )

    direct = _rule_table_rule_chains_from_visible(
        semantic_chains,
        top_bg=int(premise_bg),
        bottom_bg=int(consequence_bg),
    )
    premise = _symbols(premise_items)
    consequence = _symbols(consequence_items)
    desired = _rule_table_desired_bottom_signatures(premise, direct)
    depth = 1
    if desired is None or len(desired) != len(consequence):
        composed = _rule_table_composed_desired_bottom_signatures(
            premise,
            semantic_chains,
            top_bg=int(premise_bg),
            bottom_bg=int(consequence_bg),
            expected_len=len(consequence),
        )
        if composed is not None:
            desired, _composed_rules = composed
            depth = 2
    derived_symbols = list(desired or [])
    static_consistent = bool(
        derived_symbols and len(derived_symbols) == len(consequence) and entries
    )

    entry_slots = _rule_table_entry_slots_from_visible_chains(semantic_chains)
    selected_entry = _rule_table_selected_entry_index(
        arr, entry_slots, fg_color=int(foreground)
    )
    selected_slot = _rule_table_selected_slot_index(
        arr, consequence_items, fg_color=int(foreground)
    )
    selector: dict[str, Any] | None = None
    if selected_entry is not None:
        selected = entry_slots[int(selected_entry)]
        selected_items = list(selected.get("items") or ())
        selector = {
            "domain": "rule_entry",
            "index": int(selected_entry),
            "count": len(entry_slots),
            "entry_index": int(selected.get("chain_index", -1)),
            "entry_side": str(selected.get("side", "")),
            "role": str(roles.get(int(selected.get("bg", -1)), "unknown")),
            "symbols": [str(symbol) for symbol in selected.get("signatures") or ()],
            "centers": _centers(selected_items),
        }
    elif selected_slot is not None:
        selector = {
            "domain": "consequence_slot",
            "index": int(selected_slot),
            "count": len(consequence_items),
            "role": "consequence",
            "symbols": [consequence[int(selected_slot)]],
            "centers": [_centers([consequence_items[int(selected_slot)]])[0]],
        }
    if selector is None or int(selector["count"]) <= 0:
        return None

    entry_mode = (
        _rule_entry_mode(entries, len(premise), len(consequence))
        if selector["domain"] == "rule_entry"
        else ""
    )
    mutable_rules = bool(entry_mode)
    consistent = bool(static_consistent or mutable_rules)
    if not consistent:
        return None
    if mutable_rules:
        desired_symbols = list(consequence)
        mismatch_indices = (
            [
                index
                for index, pair in enumerate(zip(consequence, derived_symbols))
                if pair[0] != pair[1]
            ]
            if len(derived_symbols) == len(consequence)
            else list(range(len(consequence)))
        )
        depth = 3 if entry_mode == "tree_alter_rules" else 1
        composition_kind = entry_mode
    else:
        desired_symbols = list(derived_symbols)
        mismatch_indices = [
            index
            for index, pair in enumerate(zip(consequence, desired_symbols))
            if pair[0] != pair[1]
        ]
        composition_kind = "double_composition" if depth > 1 else "direct"

    cycles = _learn_cycles(records)
    role_cycles: dict[str, list[list[str]]] = {}
    for bg, pairs in cycles.items():
        role = roles.get(int(bg))
        if role:
            role_cycles[role] = [[source, target] for source, target in pairs]
    symbol_cycles = _global_cycles_from_roles(role_cycles)
    observed_cycle = _ordered_cycle(
        consequence
        + premise
        + [
            symbol
            for entry in entries
            for key in ("left_symbols", "right_symbols")
            for symbol in entry[key]
        ]
    )
    return {
        "family_id": FAMILY_ID,
        "parse_status": "bound" if consistent else "insufficient_rewrite",
        "premise": {
            "symbols": premise,
            "centers": _centers(premise_items),
        },
        "consequence": {
            "symbols": consequence,
            "centers": _centers(consequence_items),
        },
        "rule_entries": entries,
        "modular_symbol_states": {
            "modulus": int(RULE_TABLE_CYCLE_MODULUS),
            "observed_order": observed_cycle,
            "learned_successors": role_cycles,
            "symbol_cycles": symbol_cycles,
        },
        "selector": selector,
        "composition": {
            "visible": bool(depth > 1),
            "depth": int(depth),
            "kind": composition_kind,
            "mutable_rules": mutable_rules,
            "intermediate_role_count": len(
                {role for role in roles.values() if role.startswith("intermediate_")}
            ),
        },
        "rewrite": {
            "consistent": consistent,
            "desired_symbols": desired_symbols,
            "derived_symbols": derived_symbols,
            "mismatch_indices": mismatch_indices,
            "satisfied_count": len(consequence) - len(mismatch_indices)
            if consistent
            else 0,
        },
    }


def _learn_cycles(
    records: list[tuple[np.ndarray, str]],
) -> dict[int, list[tuple[str, str]]]:
    learned: dict[int, list[tuple[str, str]]] = {}
    for (before, _), (after, action) in zip(records, records[1:]):
        if action not in _ROTATION_DELTAS:
            continue
        before_obs = _selected_observation(before)
        after_obs = _selected_observation(after)
        if before_obs is None or after_obs is None:
            continue
        if (
            before_obs[:3] != after_obs[:3]
            or before_obs[4] != after_obs[4]
            or len(before_obs[3]) != len(after_obs[3])
        ):
            continue
        bg = int(before_obs[2])
        pairs = learned.setdefault(bg, [])
        for source, target in zip(before_obs[3], after_obs[3]):
            forward = (source, target) if action == "ACTION2" else (target, source)
            if forward[0] != forward[1] and forward not in pairs:
                pairs.append(forward)
    return learned


def _selected_observation(
    arr: np.ndarray,
) -> tuple[str, int, int, list[str], str] | None:
    normalized = np.ascontiguousarray(np.asarray(arr), dtype=np.int16)
    return _selected_observation_cached(
        tuple(int(value) for value in normalized.shape),
        normalized.tobytes(),
    )


@lru_cache(maxsize=192)
def _selected_observation_cached(
    shape: tuple[int, int],
    payload: bytes,
) -> tuple[str, int, int, list[str], str] | None:
    arr = np.frombuffer(payload, dtype=np.int16).reshape(shape)
    glyphs = _rule_table_glyph_candidates(arr)
    rows = _rule_table_glyph_rows(glyphs)
    top, bottom = _rule_table_lower_rows(rows)
    if top is None or bottom is None:
        return None
    top_items = list(top["items"])
    bottom_items = list(bottom["items"])
    top_bg = _single_row_category(top_items)
    bottom_bg = _single_row_category(bottom_items)
    foreground = _single_foreground_color(top_items + bottom_items)
    if top_bg is None or bottom_bg is None or foreground is None:
        return None
    visible = _rule_table_visible_rule_chains(rows, lower_top_y=int(top["y"]))
    roles = _role_map(visible, int(top_bg), int(bottom_bg))
    chains = _semantic_visible_chains(visible, roles)
    entries = _rule_table_entry_slots_from_visible_chains(chains)
    selected_entry = _rule_table_selected_entry_index(
        arr, entries, fg_color=int(foreground)
    )
    if selected_entry is not None:
        entry = entries[int(selected_entry)]
        chain_index = int(entry["chain_index"])
        side = str(entry["side"])
        rule_fingerprint = []
        for index, chain in enumerate(chains):
            left = [str(symbol) for symbol in chain.get("left") or ()]
            right = [str(symbol) for symbol in chain.get("right") or ()]
            if index == chain_index:
                if side == "left":
                    left = ["*"] * len(left)
                else:
                    right = ["*"] * len(right)
            rule_fingerprint.append(
                [str(chain["left_role"]), left, str(chain["right_role"]), right]
            )
        fingerprint = stable_digest(
            {
                "premise": _symbols(top_items),
                "consequence": _symbols(bottom_items),
                "rules": rule_fingerprint,
                "slot_count": len(entries),
            }
        )
        return (
            "rule_entry",
            int(selected_entry),
            int(entry["bg"]),
            [str(symbol) for symbol in entry.get("signatures") or ()],
            fingerprint,
        )
    selected_slot = _rule_table_selected_slot_index(
        arr, bottom_items, fg_color=int(foreground)
    )
    if selected_slot is None:
        return None
    item = bottom_items[int(selected_slot)]
    consequence = _symbols(bottom_items)
    consequence[int(selected_slot)] = "*"
    fingerprint = stable_digest(
        {
            "premise": _symbols(top_items),
            "consequence": consequence,
            "rules": [
                [
                    str(chain["left_role"]),
                    [str(symbol) for symbol in chain.get("left") or ()],
                    str(chain["right_role"]),
                    [str(symbol) for symbol in chain.get("right") or ()],
                ]
                for chain in chains
            ],
            "slot_count": len(bottom_items),
        }
    )
    return (
        "consequence_slot",
        int(selected_slot),
        int(item["bg"]),
        [str(item["signature"])],
        fingerprint,
    )


def bind_current(
    frame: Any,
    context: MechanicContext | None = None,
) -> dict[str, Any] | None:
    """Bind visible rule-table roles and the current selector."""
    binding = _parse(context.module_input() if context is not None else frame)
    if binding is None:
        return None
    inherited = _inherited_schema(context)
    current_cycles = [
        [str(value) for value in cycle]
        for cycle in binding["modular_symbol_states"].get("symbol_cycles", [])
        if isinstance(cycle, Sequence) and len(cycle) == RULE_TABLE_CYCLE_MODULUS
    ]
    inherited_cycles = [
        [str(value) for value in cycle]
        for cycle in inherited.get("symbol_cycles", [])
        if isinstance(cycle, Sequence) and len(cycle) == RULE_TABLE_CYCLE_MODULUS
    ]
    merged = {tuple(cycle): cycle for cycle in inherited_cycles + current_cycles}
    binding["modular_symbol_states"]["symbol_cycles"] = [
        merged[key] for key in sorted(merged)
    ]
    binding["inherited_schema"] = inherited
    return binding


def _inherited_schema(context: MechanicContext | None) -> dict[str, Any]:
    empty = {
        "available": False,
        "verified": False,
        "learned_level": 0,
        "bound_level": (
            int(context.level) if isinstance(context, MechanicContext) else 0
        ),
        "symbol_cycles": [],
    }
    if not isinstance(context, MechanicContext) or not isinstance(
        context.memory, Mapping
    ):
        return empty
    schemas = context.memory.get("schemas")
    if not isinstance(schemas, list):
        return empty
    candidates = [
        row
        for row in schemas
        if isinstance(row, Mapping)
        and str(row.get("source_family", "")) == "rule_rewrite"
        and int(row.get("learned_level", 0) or 0) <= int(context.level)
    ]
    if not candidates:
        return empty
    selected = max(
        candidates,
        key=lambda row: (
            str(row.get("status", "")) == "verified",
            float(row.get("confidence", 0.0) or 0.0),
        ),
    )
    operator = selected.get("operator")
    operator = operator if isinstance(operator, Mapping) else {}
    cycles = [
        [str(value) for value in cycle]
        for cycle in operator.get("symbol_cycles", [])
        if isinstance(cycle, Sequence)
        and not isinstance(cycle, (str, bytes))
        and len(cycle) == RULE_TABLE_CYCLE_MODULUS
    ]
    return {
        "available": True,
        "verified": str(selected.get("status", "")) == "verified",
        "status": str(selected.get("status", "observed")),
        "learned_level": int(selected.get("learned_level", 0) or 0),
        "bound_level": int(context.level),
        "symbol_cycles": cycles,
    }


def _level_ordinal(level: Any) -> int | None:
    if isinstance(level, (int, np.integer)):
        return int(level)
    if isinstance(level, Mapping):
        for key in ("ordinal", "level", "level_number", "index"):
            value = level.get(key)
            if isinstance(value, (int, np.integer)):
                return int(value)
    return None


def compile_schema(
    frame: Any,
    level: Any,
    context: MechanicContext | None = None,
) -> dict[str, Any]:
    """Compile a JSON-safe, structural schema from a frame or frame history."""
    binding = bind_current(frame, context=context)
    cycles = (
        list(binding["modular_symbol_states"].get("symbol_cycles") or [])
        if binding
        else []
    )
    return {
        "schema_version": "tufa.rule_table_rewrite.v2",
        "family_id": FAMILY_ID,
        "level_ordinal": _level_ordinal(level),
        "roles": ["premise", "consequence", "rule_entry", "selector", "intermediate"],
        "required_bindings": ["premise", "consequence", "rule_entry", "selector"],
        "operators": [
            {
                "kind": "rotate_symbol",
                "actions": ["ACTION1", "ACTION2"],
                "delta": [-1, 1],
            },
            {
                "kind": "move_selector",
                "actions": ["ACTION3", "ACTION4"],
                "delta": [-1, 1],
            },
            {"kind": "compose_rewrite", "max_visible_depth": 3},
        ],
        "invariants": [
            "premise_order_is_stable",
            "only_the_bound_selector_target_may_rotate",
            "editable_rule_sides_follow_visible_reading_order",
            "rewrite_is_reparsed_after_every_action",
        ],
        "operator": {
            "kind": "rule_table_rewrite",
            "transition": "move the selector or rotate one selected symbol",
            "symbol_cycles": cycles,
            "cycle_count": len(cycles),
        },
        "goal_relation": "visible consequences satisfy the composed premise rewrite",
        "falsifiers": [
            "a non-selected symbol changes",
            "the reparsed rewrite loses premise-consequence consistency",
        ],
        "binding": binding,
        "inherited_schema": _inherited_schema(context),
        "executable": bool(binding and binding["rewrite"]["consistent"]),
    }


def _successor(
    binding: Mapping[str, Any], role: str, symbol: str, delta: int
) -> str | None:
    learned = (
        binding["modular_symbol_states"].get("learned_successors", {}).get(role, [])
    )
    forward = {str(source): str(target) for source, target in learned}
    if delta < 0:
        forward = {target: source for source, target in forward.items()}
    if symbol in forward:
        return forward[symbol]
    cycle = _symbol_cycle(binding, symbol)
    if cycle is None:
        return None
    index = cycle.index(symbol)
    return cycle[(index + delta) % len(cycle)]


def _symbol_cycle(
    binding: Mapping[str, Any],
    symbol: str,
) -> list[str] | None:
    for raw in binding["modular_symbol_states"].get("symbol_cycles", []):
        cycle = [str(value) for value in raw]
        if len(cycle) == RULE_TABLE_CYCLE_MODULUS and symbol in cycle:
            return cycle
    return None


def _learned_edge_count(binding: Mapping[str, Any]) -> int:
    sources: dict[str, str] = {}
    for cycle in binding["modular_symbol_states"].get("symbol_cycles", []):
        values = [str(value) for value in cycle]
        if len(values) != RULE_TABLE_CYCLE_MODULUS:
            continue
        for index, source in enumerate(values):
            sources[source] = values[(index + 1) % len(values)]
    role_cycles = binding["modular_symbol_states"].get("learned_successors", {})
    for rows in role_cycles.values():
        for row in rows or []:
            if not isinstance(row, Sequence) or len(row) != 2:
                continue
            source, target = str(row[0]), str(row[1])
            if source and target and source != target:
                sources.setdefault(source, target)
    return len(sources)


def _selected_cycle_edge_count(binding: Mapping[str, Any]) -> int:
    symbols = [str(value) for value in binding["selector"].get("symbols") or []]
    if not symbols:
        return 0
    selected = symbols[0]
    cycle = _symbol_cycle(binding, selected)
    if cycle is not None:
        return RULE_TABLE_CYCLE_MODULUS
    edges: list[tuple[str, str]] = []
    for rows in binding["modular_symbol_states"].get("learned_successors", {}).values():
        for row in rows or []:
            if isinstance(row, Sequence) and len(row) == 2:
                edges.append((str(row[0]), str(row[1])))
    component = {selected}
    changed = True
    while changed:
        changed = False
        for source, target in edges:
            if source in component or target in component:
                before = len(component)
                component.update((source, target))
                changed = changed or len(component) != before
    return min(
        RULE_TABLE_CYCLE_MODULUS,
        len(
            {
                source
                for source, target in edges
                if source in component and target in component
            }
        ),
    )


def _solver_role_inputs(
    binding: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[int, dict[str, str]]]:
    entries = list(binding.get("rule_entries") or [])
    role_order = _ordered_cycle(
        ["premise"]
        + [
            str(role)
            for entry in entries
            for role in (entry.get("left_role"), entry.get("right_role"))
        ]
        + ["consequence"]
    )
    role_ids = {role: index for index, role in enumerate(role_order)}
    learned = binding["modular_symbol_states"].get("learned_successors", {})
    cycles = [
        [str(value) for value in raw]
        for raw in binding["modular_symbol_states"].get("symbol_cycles", [])
        if isinstance(raw, Sequence)
        and not isinstance(raw, (str, bytes))
        and len(raw) == RULE_TABLE_CYCLE_MODULUS
    ]
    successors: dict[int, dict[str, str]] = {}
    for role, role_id in role_ids.items():
        mapping: dict[str, str] = {}
        for row in learned.get(role, []) or []:
            if isinstance(row, Sequence) and len(row) == 2:
                source, target = str(row[0]), str(row[1])
                if source and target and source != target:
                    mapping.setdefault(source, target)
        for cycle in cycles:
            for index, source in enumerate(cycle):
                mapping.setdefault(source, cycle[(index + 1) % len(cycle)])
        successors[int(role_id)] = mapping

    chains = [
        {
            "left_bg": role_ids[str(entry["left_role"])],
            "right_bg": role_ids[str(entry["right_role"])],
            "left": tuple(str(symbol) for symbol in entry.get("left_symbols") or ()),
            "right": tuple(str(symbol) for symbol in entry.get("right_symbols") or ()),
        }
        for entry in entries
        if str(entry.get("left_role")) in role_ids
        and str(entry.get("right_role")) in role_ids
    ]
    return chains, role_ids, successors


def _rule_entry_rotation_deltas(
    binding: Mapping[str, Any],
) -> tuple[list[int], list[int]] | None:
    chains, role_ids, successors = _solver_role_inputs(binding)
    if len(chains) != len(binding.get("rule_entries") or []):
        return None
    premise = [str(value) for value in binding["premise"]["symbols"]]
    consequence = [str(value) for value in binding["consequence"]["symbols"]]
    kind = str(binding.get("composition", {}).get("kind", ""))
    if kind == "direct_alter_rules":
        return _rule_table_solve_alter_rules_by_signatures(
            premise,
            consequence,
            chains,
            successors,
        )
    if kind != "tree_alter_rules":
        return None
    intermediate = _ordered_cycle(
        role
        for entry in binding.get("rule_entries") or []
        for role in (entry.get("left_role"), entry.get("right_role"))
        if str(role).startswith("intermediate_")
    )
    if len(intermediate) != 1:
        return None
    return _rule_table_solve_tree_alter_rules_by_signatures(
        premise,
        consequence,
        chains,
        top_bg=role_ids["premise"],
        middle_bg=role_ids[intermediate[0]],
        bottom_bg=role_ids["consequence"],
        successors=successors,
    )


def _rule_entry_abstract_plan(binding: Mapping[str, Any]) -> list[dict[str, str]]:
    selector = binding["selector"]
    if selector["domain"] != "rule_entry":
        return []
    solution = _rule_entry_rotation_deltas(binding)
    if solution is None:
        return []
    left_deltas, right_deltas = solution
    rotation_slots = _rule_table_rotation_slots_from_deltas(left_deltas, right_deltas)
    if not rotation_slots:
        return []
    count = int(selector["count"])
    if count != 2 * len(binding.get("rule_entries") or []):
        return []
    selected = int(selector["index"])
    plan: list[dict[str, str]] = []
    for target, delta in rotation_slots:
        target = int(target)
        for _ in range(count):
            if selected == target:
                break
            forward = (target - selected) % count
            backward = (selected - target) % count
            action = "ACTION4" if forward <= backward else "ACTION3"
            selected = (selected + _SELECTOR_DELTAS[action]) % count
            plan.append({"action": action})
        else:
            return []
        steps, direction = _rule_table_short_rotation_steps(int(delta))
        action = "ACTION2" if direction > 0 else "ACTION1"
        plan.extend({"action": action} for _ in range(int(steps)))
    return plan


def _consequence_abstract_plan(binding: Mapping[str, Any]) -> list[dict[str, str]]:
    if not binding["rewrite"]["consistent"]:
        return []
    selector = binding["selector"]
    if selector["domain"] != "consequence_slot":
        return []
    current = [str(value) for value in binding["consequence"]["symbols"]]
    desired = [str(value) for value in binding["rewrite"]["desired_symbols"]]
    if len(current) != len(desired) or not current:
        return []
    selected = int(selector["index"])
    plan: list[dict[str, str]] = []
    for _ in range(256):
        mismatches = [
            index
            for index, pair in enumerate(zip(current, desired))
            if pair[0] != pair[1]
        ]
        if not mismatches:
            return plan
        target = int(mismatches[0])
        count = len(current)
        if selected != target:
            forward = (target - selected) % count
            backward = (selected - target) % count
            action = "ACTION4" if forward <= backward else "ACTION3"
            selected = (selected + _SELECTOR_DELTAS[action]) % count
            plan.append({"action": action})
            continue
        cycle = _symbol_cycle(binding, current[selected])
        if cycle is None or desired[selected] not in cycle:
            return []
        current_index = cycle.index(current[selected])
        target_index = cycle.index(desired[selected])
        forward = (target_index - current_index) % len(cycle)
        backward = (current_index - target_index) % len(cycle)
        action = "ACTION2" if forward <= backward else "ACTION1"
        current[selected] = cycle[
            (current_index + _ROTATION_DELTAS[action]) % len(cycle)
        ]
        plan.append({"action": action})
    return []


def _proposal_actions(binding: Mapping[str, Any]) -> tuple[list[dict[str, str]], str]:
    if not binding["rewrite"]["consistent"]:
        return [], "inconsistent_visible_rewrite"
    selected_symbols = [
        str(value) for value in binding["selector"].get("symbols") or []
    ]
    if not selected_symbols:
        return [], "selected_symbol_not_bound"
    if _symbol_cycle(binding, selected_symbols[0]) is None:
        return [{"action": "ACTION2"}], "observe_symbol_successor"
    if binding["selector"]["domain"] == "consequence_slot":
        plan = _consequence_abstract_plan(binding)
        if plan:
            return plan, "solve_visible_consequence"
    elif binding["selector"]["domain"] == "rule_entry":
        plan = _rule_entry_abstract_plan(binding)
        if plan:
            return plan, "solve_visible_rule_entries"
    return [], "unsupported_rule_entry_rewrite"


def _work_measure(
    binding: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    phase: str,
) -> tuple[int, int]:
    if phase == "observe_symbol_successor":
        unknown = RULE_TABLE_CYCLE_MODULUS - _selected_cycle_edge_count(binding)
        before = unknown * _PROBE_WEIGHT + _PROBE_TAIL
        return before, max(0, before - _PROBE_WEIGHT)
    before = len(actions)
    return before, max(0, before - 1)


def _scope_reason(
    binding: Mapping[str, Any],
    context: MechanicContext | None,
) -> str:
    level = int(context.level) if isinstance(context, MechanicContext) else 0
    if level > 1 and not _inherited_schema(context).get("verified"):
        return "higher_level_requires_verified_l1_rule_schema"
    domain = str(binding["selector"]["domain"])
    kind = str(binding.get("composition", {}).get("kind", ""))
    if domain == "rule_entry" and kind not in {
        "direct_alter_rules",
        "tree_alter_rules",
    }:
        return "rule_entry_structure_not_certified"
    if domain not in {"consequence_slot", "rule_entry"}:
        return "selector_domain_not_certified"
    return ""


def propose_plan(
    frame: Any,
    valid_actions: Iterable[Any] | None = None,
    context: MechanicContext | None = None,
) -> dict[str, Any]:
    """Return one certified observation or rewrite action."""
    binding = bind_current(frame, context=context)
    if binding is None:
        return _blocked_plan("rule_table_roles_not_bound", context)
    actions, phase = _proposal_actions(binding)
    if not actions:
        return _blocked_plan(phase, context)
    action = str(actions[0]["action"])
    if action not in _available_actions(valid_actions):
        return _blocked_plan("required_action_not_available", context)
    expected = _predict_from_binding(binding, action, actions, phase)
    if expected is None:
        return _blocked_plan("transition_not_predictable", context)
    reason = _scope_reason(binding, context)
    before, expected_after = _work_measure(binding, actions, phase)
    inherited = _inherited_schema(context)
    return {
        "schema_version": "tufa.rule_table_rewrite.plan.v2",
        "family_id": FAMILY_ID,
        "status": "proposed",
        "actions": [{"action": action}],
        "action_count": 1,
        "certified": not reason,
        "certificate_reason": (
            "selected-glyph observation with exact semantic change scope"
            if not reason and phase == "observe_symbol_successor"
            else "current rewrite binding with exact one-step verifier"
            if not reason
            else reason
        ),
        "transition_rule": phase,
        "progress_measure": {
            "metric": "rule_rewrite_work",
            "before": before,
            "expected_after": expected_after,
            "strict": True,
        },
        "suffix_exposed": False,
        "requires_transition_verification": True,
        "first_prediction": expected,
        "detail": {
            "phase": phase,
            "private_plan_length": len(actions),
            "learned_cycle_edges": _learned_edge_count(binding),
            "selected_cycle_edges": _selected_cycle_edge_count(binding),
            "symbol_cycle_count": len(
                binding["modular_symbol_states"].get("symbol_cycles", [])
            ),
            "inherited_schema": inherited,
        },
    }


def _blocked_plan(
    reason: str,
    context: MechanicContext | None,
) -> dict[str, Any]:
    return {
        "schema_version": "tufa.rule_table_rewrite.plan.v2",
        "family_id": FAMILY_ID,
        "status": "blocked",
        "actions": [],
        "action_count": 0,
        "certified": False,
        "certificate_reason": str(reason),
        "suffix_exposed": False,
        "detail": {"inherited_schema": _inherited_schema(context)},
    }


def predict_transition(
    frame: Any,
    action: Any,
    context: MechanicContext | None = None,
) -> dict[str, Any] | None:
    """Predict one exact bound selector/rotation delta, otherwise fail closed."""
    binding = bind_current(frame, context=context)
    name = _action_name(action)
    if binding is None or not binding["rewrite"]["consistent"]:
        return None
    actions, phase = _proposal_actions(binding)
    if not actions or name != str(actions[0].get("action", "")):
        return None
    return _predict_from_binding(binding, name, actions, phase)


def _predict_from_binding(
    binding: Mapping[str, Any],
    name: str,
    actions: Sequence[Mapping[str, Any]],
    phase: str,
) -> dict[str, Any] | None:
    selector = binding["selector"]
    expected: dict[str, Any] = {
        "family_id": FAMILY_ID,
        "accepted": True,
        "action": name,
        "operation": "",
        "delta": 0,
        "selector_before": {
            "domain": str(selector["domain"]),
            "index": int(selector["index"]),
            "count": int(selector["count"]),
            "entry_index": int(selector.get("entry_index", -1)),
            "entry_side": str(selector.get("entry_side", "")),
            "role": str(selector.get("role", "")),
        },
        "premise_before": list(binding["premise"]["symbols"]),
        "consequence_before": list(binding["consequence"]["symbols"]),
        "rules_before": list(binding["rule_entries"]),
        "desired_before": list(binding["rewrite"]["desired_symbols"]),
        "composition_before": dict(binding["composition"]),
        "symbol_cycles": list(
            binding["modular_symbol_states"].get("symbol_cycles") or []
        ),
    }
    if phase == "observe_symbol_successor":
        if name != "ACTION2":
            return None
        symbols = [str(symbol) for symbol in selector.get("symbols") or ()]
        if not symbols:
            return None
        before, expected_after = _work_measure(binding, actions, phase)
        expected.update(
            {
                "operation": "observe_successor",
                "delta": 1,
                "symbols_before": symbols,
                "abstract_progress": {
                    "metric": "rule_rewrite_work",
                    "before": before,
                    "expected_after": expected_after,
                    "phase": phase,
                },
            }
        )
        return expected

    expected["abstract_progress"] = {
        "metric": "rule_rewrite_work",
        "before": len(actions),
        "expected_after": max(0, len(actions) - 1),
        "expected_successor_digest": stable_digest(list(actions[1:])),
        "phase": phase,
    }
    if name in _SELECTOR_DELTAS:
        delta = _SELECTOR_DELTAS[name]
        expected["operation"] = "move_selector"
        expected["delta"] = delta
        expected["selector_after_index"] = (int(selector["index"]) + delta) % int(
            selector["count"]
        )
        return expected
    if name not in _ROTATION_DELTAS:
        return None
    delta = _ROTATION_DELTAS[name]
    role = str(selector.get("role", ""))
    symbols = [str(symbol) for symbol in selector.get("symbols") or ()]
    if not symbols:
        return None
    expected_symbols = [_successor(binding, role, symbol, delta) for symbol in symbols]
    if any(symbol is None for symbol in expected_symbols):
        return None
    expected["operation"] = "rotate_symbol"
    expected["delta"] = delta
    expected["symbols_before"] = symbols
    expected["symbols_after"] = [str(symbol) for symbol in expected_symbols]
    return expected


def _entry_symbols(
    binding: Mapping[str, Any], selector: Mapping[str, Any]
) -> list[str] | None:
    index = int(selector.get("entry_index", -1))
    side = str(selector.get("entry_side", ""))
    entries = list(binding.get("rule_entries") or [])
    if index < 0 or index >= len(entries) or side not in {"left", "right"}:
        return None
    return [str(symbol) for symbol in entries[index].get(f"{side}_symbols") or ()]


def verify_transition(expected: Any, current_frame: Any) -> bool:
    """Strictly verify one bound operation and the reparsed rewrite relation."""
    if not isinstance(expected, Mapping) or expected.get("family_id") != FAMILY_ID:
        return False
    operation = str(expected.get("operation", ""))
    if operation not in {"move_selector", "rotate_symbol", "observe_successor"}:
        return False
    delta = expected.get("delta")
    if delta not in {-1, 1}:
        return False
    action = _action_name(expected.get("action", ""))
    action_delta = (
        _SELECTOR_DELTAS.get(action)
        if operation == "move_selector"
        else _ROTATION_DELTAS.get(action)
    )
    if action_delta != delta:
        return False
    current = bind_current(current_frame)
    if current is None or not current["rewrite"]["consistent"]:
        return False
    cycles = [
        [str(value) for value in cycle]
        for cycle in expected.get("symbol_cycles") or []
        if isinstance(cycle, Sequence) and len(cycle) == RULE_TABLE_CYCLE_MODULUS
    ]
    current["modular_symbol_states"]["symbol_cycles"] = cycles
    before_selector = expected.get("selector_before")
    if not isinstance(before_selector, Mapping):
        return False
    selector = current["selector"]
    if (
        str(selector["domain"]) != str(before_selector.get("domain", ""))
        or int(selector["count"]) != int(before_selector.get("count", -1))
        or list(current["premise"]["symbols"])
        != list(expected.get("premise_before") or [])
        or dict(current["composition"])
        != dict(expected.get("composition_before") or {})
    ):
        return False

    if operation == "observe_successor":
        if int(selector["index"]) != int(before_selector.get("index", -1)):
            return False
        if selector["domain"] == "consequence_slot":
            before = [
                str(symbol) for symbol in expected.get("consequence_before") or ()
            ]
            current_symbols = [
                str(symbol) for symbol in current["consequence"]["symbols"]
            ]
            index = int(selector["index"])
            if index < 0 or index >= len(before) or len(current_symbols) != len(before):
                return False
            changed = [
                idx
                for idx, pair in enumerate(zip(before, current_symbols))
                if pair[0] != pair[1]
            ]
            return bool(
                changed == [index]
                and current_symbols[index]
                not in set(expected.get("symbols_before") or [])
                and list(current["rule_entries"])
                == list(expected.get("rules_before") or [])
                and list(current["rewrite"]["desired_symbols"])
                == list(expected.get("desired_before") or [])
            )
        if selector["domain"] != "rule_entry":
            return False
        if (
            int(selector.get("entry_index", -1))
            != int(before_selector.get("entry_index", -2))
            or str(selector.get("entry_side", ""))
            != str(before_selector.get("entry_side", ""))
            or str(selector.get("role", "")) != str(before_selector.get("role", ""))
        ):
            return False
        before_symbols = [
            str(symbol) for symbol in expected.get("symbols_before") or ()
        ]
        current_symbols = _entry_symbols(current, before_selector)
        if (
            current_symbols is None
            or len(current_symbols) != len(before_symbols)
            or any(
                source == target
                for source, target in zip(before_symbols, current_symbols)
            )
        ):
            return False
        before_rules = list(expected.get("rules_before") or [])
        entry_index = int(before_selector.get("entry_index", -1))
        side = str(before_selector.get("entry_side", ""))
        if (
            entry_index < 0
            or entry_index >= len(before_rules)
            or side not in {"left", "right"}
        ):
            return False
        patched = [dict(entry) for entry in before_rules]
        patched[entry_index][f"{side}_symbols"] = current_symbols
        return bool(
            list(current["rule_entries"]) == patched
            and list(current["consequence"]["symbols"])
            == list(expected.get("consequence_before") or [])
        )

    if operation == "move_selector":
        semantic_match = bool(
            int(selector["index"]) == int(expected.get("selector_after_index", -1))
            and list(current["consequence"]["symbols"])
            == list(expected.get("consequence_before") or [])
            and list(current["rule_entries"])
            == list(expected.get("rules_before") or [])
            and list(current["rewrite"]["desired_symbols"])
            == list(expected.get("desired_before") or [])
        )
        return semantic_match and _verify_abstract_rewrite_successor(expected, current)

    if int(selector["index"]) != int(before_selector.get("index", -1)):
        return False
    expected_after = [str(symbol) for symbol in expected.get("symbols_after") or ()]
    if not expected_after:
        return False
    if selector["domain"] == "consequence_slot":
        before = [str(symbol) for symbol in expected.get("consequence_before") or ()]
        index = int(selector["index"])
        if index < 0 or index >= len(before):
            return False
        wanted = list(before)
        if len(expected_after) != 1:
            return False
        wanted[index] = expected_after[0]
        semantic_match = bool(
            list(current["consequence"]["symbols"]) == wanted
            and list(current["rule_entries"])
            == list(expected.get("rules_before") or [])
            and list(current["rewrite"]["desired_symbols"])
            == list(expected.get("desired_before") or [])
        )
        return semantic_match and _verify_abstract_rewrite_successor(expected, current)

    if (
        int(selector.get("entry_index", -1))
        != int(before_selector.get("entry_index", -2))
        or str(selector.get("entry_side", ""))
        != str(before_selector.get("entry_side", ""))
        or str(selector.get("role", "")) != str(before_selector.get("role", ""))
    ):
        return False
    current_symbols = _entry_symbols(current, before_selector)
    if current_symbols != expected_after:
        return False
    before_rules = list(expected.get("rules_before") or [])
    current_rules = list(current["rule_entries"])
    entry_index = int(before_selector.get("entry_index", -1))
    side = str(before_selector.get("entry_side", ""))
    if (
        entry_index < 0
        or entry_index >= len(before_rules)
        or len(current_rules) != len(before_rules)
    ):
        return False
    patched = [dict(entry) for entry in before_rules]
    patched[entry_index][f"{side}_symbols"] = expected_after
    semantic_match = bool(
        current_rules == patched
        and list(current["consequence"]["symbols"])
        == list(expected.get("consequence_before") or [])
    )
    return semantic_match and _verify_abstract_rewrite_successor(expected, current)


def _verify_abstract_rewrite_successor(
    expected: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    progress = expected.get("abstract_progress")
    if not isinstance(progress, Mapping):
        return False
    try:
        expected_after = int(progress["expected_after"])
    except (KeyError, TypeError, ValueError):
        return False
    if expected_after <= 0:
        return False
    actions, phase = _proposal_actions(current)
    expected_phase = str(progress.get("phase", ""))
    if phase != expected_phase or len(actions) != expected_after:
        return False
    if expected_phase == "solve_visible_rule_entries":
        # Tree rewrites can have several equal-cost assignments. The exact
        # selected-side delta above is still verified, while reparsing must
        # prove that globally minimal remaining work fell by one.
        return True
    expected_digest = str(progress.get("expected_successor_digest", "") or "")
    return bool(expected_digest and stable_digest(actions) == expected_digest)
