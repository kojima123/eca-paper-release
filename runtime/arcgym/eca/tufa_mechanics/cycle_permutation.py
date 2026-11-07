"""Visible transition learner for reversible button permutations.

This mechanic intentionally knows nothing about a game catalogue.  It uses the
generic strict-synthesis button-cycle readers only to bind visible tokens,
hollow targets, and click controls.  Button effects are learned from same-level
frame transitions: a control is executable only when the observations identify
one unique slot permutation and another visible control is its exact inverse.

Planning tracks only target-kind token positions, but transition prediction and
verification cover the complete token assignment.  A learned permutation may
contain several disjoint cycles, which is how stacked button memberships are
represented without a level-specific topology or stored action suffix.
"""

from __future__ import annotations

import re
from math import gcd
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from arcgym.eca.primitive_synthesis import (
    _button_cycle_click_button_components,
    _button_cycle_components,
    _button_cycle_hollow_targets,
    _button_cycle_movable_square_token_positions,
)
from arcgym.eca.tufa_mechanics.base import (
    MechanicContext,
    as_frame,
    json_safe_mapping,
    stable_digest,
)


FAMILY_ID = "cycle_permutation_alignment"
_SCHEMA_VERSION = "tufa.cycle_permutation.schema.v1"
_BINDING_VERSION = "tufa.cycle_permutation.binding.v1"
_PLAN_VERSION = "tufa.cycle_permutation.plan.v1"
_PREDICTION_VERSION = "tufa.cycle_permutation.transition.v1"
_VERIFICATION_VERSION = "tufa.cycle_permutation.verification.v1"

_MIN_PERMUTATION_SUPPORT = 2
_MAX_HISTORY_RECORDS = 96
_MAX_SEARCH_STATES = 100_000
_MAX_SEARCH_DEPTH = 64

Position = tuple[int, int]


@dataclass(frozen=True)
class _ParsedFrame:
    frame: np.ndarray
    token_size: int
    tokens: dict[Position, int]
    targets: tuple[tuple[int, Position], ...]
    buttons: dict[Position, dict[str, int]]
    static_digest: str


@dataclass(frozen=True)
class _Record:
    frame: np.ndarray
    action: Any
    level: int | None
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _TransitionEvidence:
    site: Position
    before: np.ndarray
    after: np.ndarray
    before_tokens: dict[Position, int]
    after_tokens: dict[Position, int]
    token_size: int
    evidence_digest: str


@dataclass
class _PermutationModel:
    site: Position
    source_to_dest: dict[Position, Position]
    cycles: tuple[tuple[Position, ...], ...]
    support_count: int
    volatile_edges: dict[str, int]
    evidence_digests: tuple[str, ...]
    inverse_sites: tuple[Position, ...] = field(default_factory=tuple)
    stacked_slots: tuple[Position, ...] = field(default_factory=tuple)

    @property
    def certified(self) -> bool:
        return bool(self.inverse_sites)


def compile_schema(
    frame: Any,
    level: Any = 0,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a coordinate-free schema plus current evidence summary."""

    binding = bind_current(frame, context=context)
    models = list(binding.get("permutation_models", []))
    executable = bool(binding.get("executable"))
    return _json(
        {
            "schema_version": _SCHEMA_VERSION,
            "family_id": FAMILY_ID,
            "supported": bool(binding.get("supported")),
            "executable": executable,
            "level_ordinal": _level_ordinal(level),
            "roles": ["token", "target", "button", "slot", "cycle_membership"],
            "required_bindings": ["token", "target", "button", "slot"],
            "state_variables": [
                "token_kind_by_slot",
                "target_kind_by_slot",
                "button_permutation",
                "stacked_cycle_memberships",
                "alignment_deficit",
            ],
            "operators": [
                {
                    "kind": "apply_visible_button_permutation",
                    "action_family": "click",
                    "effect": "simultaneously_apply_all_inferred_membership_cycles",
                }
            ],
            "operator": {
                "kind": FAMILY_ID,
                "transition": "a visible button applies one inferred reversible slot permutation",
                "reversible_model_count": sum(
                    bool(row.get("inverse_button_count")) for row in models
                ),
                "stacked_model_count": sum(
                    bool(row.get("stacked_membership")) for row in models
                ),
            },
            "goal_relation": "every visible target slot contains its required token kind",
            "invariants": [
                "the visible slot set is conserved",
                "the complete token-kind multiset is conserved",
                "each admitted button effect is a bijection",
                "an admitted button has an observed exact inverse",
            ],
            "falsifiers": [
                "a click creates or destroys a token",
                "a button effect is not a unique bijection",
                "the inferred inverse does not compose to identity",
                "the visible target or button structure changes within a level",
                "the rebound frame disagrees with any predicted token slot",
            ],
            "verification": {
                "token_projection": "exact_full_assignment",
                "conservation": "exact_multiset_equality",
                "static_structure": "exact_current_level_rebinding",
                "fail_closed": True,
            },
            "evidence": {
                "transition_count": int(binding.get("transition_evidence_count", 0)),
                "permutation_model_count": len(models),
                "reversible_button_count": int(
                    binding.get("reversible_button_count", 0)
                ),
                "stacked_membership_button_count": int(
                    binding.get("stacked_membership_button_count", 0)
                ),
            },
            "binding_digest": str(binding.get("binding_digest", "")),
        }
    )


def bind_current(
    frame: Any,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind visible roles and induce uniquely identified button permutations."""

    parsed = _parse_frame(frame)
    if parsed is None:
        return _unbound("visible_cycle_roles_not_bound")

    evidence = _transition_evidence(parsed, context)
    all_models = _induce_models(parsed, evidence)
    certified = {
        site: model for site, model in all_models.items() if model.certified
    }
    solution, search = _shortest_alignment_plan(parsed, certified)
    deficit = _alignment_deficit(parsed.tokens, parsed.targets)
    kind_roles = _kind_roles(parsed)

    target_rows = [
        {
            "role": "target",
            "kind_role": kind_roles[kind],
            "position": [int(position[0]), int(position[1])],
            "satisfied": parsed.tokens.get(position) == kind,
        }
        for kind, position in parsed.targets
    ]
    token_rows = [
        {
            "role": "token",
            "kind_role": kind_roles[kind],
            "position": [int(position[0]), int(position[1])],
        }
        for position, kind in sorted(parsed.tokens.items(), key=_position_item_key)
    ]
    button_rows = [
        _button_binding_row(parsed, all_models.get(site), site)
        for site in sorted(parsed.buttons, key=_position_key)
    ]
    mismatch_cycles = _mismatch_cycles(parsed, certified, kind_roles)
    core = {
        "frame_shape": [int(value) for value in parsed.frame.shape],
        "token_size": int(parsed.token_size),
        "slot_count": len(parsed.tokens),
        "token_count": len(parsed.tokens),
        "target_count": len(parsed.targets),
        "button_count": len(parsed.buttons),
        "target_kind_count": len({kind for kind, _position in parsed.targets}),
        "alignment_deficit": int(deficit),
        "tokens": token_rows,
        "targets": target_rows,
        "buttons": button_rows,
        "permutation_models": [
            _model_summary(model) for _site, model in sorted(all_models.items())
        ],
        "mismatch_cycles": mismatch_cycles,
        "transition_evidence_count": len(evidence),
        "reversible_button_count": len(certified),
        "stacked_membership_button_count": sum(
            bool(model.stacked_slots or len(model.cycles) > 1)
            for model in certified.values()
        ),
        "search": search,
        "private_plan_length": len(solution or []),
        "static_digest": parsed.static_digest,
        "state_digest": stable_digest(_token_rows(parsed.tokens)),
    }
    executable = bool(deficit > 0 and certified and solution)
    return _json(
        {
            "schema_version": _BINDING_VERSION,
            "family_id": FAMILY_ID,
            "supported": True,
            "executable": executable,
            "status": "bound",
            "reason": (
                "reversible_permutations_bound"
                if executable
                else "goal_already_satisfied"
                if deficit == 0
                else str(search.get("blocked_reason", "permutation_evidence_incomplete"))
            ),
            "binding_digest": stable_digest(core),
            "role_bindings": [*target_rows, *button_rows[:8]],
            **core,
        }
    )


def propose_plan(
    frame: Any,
    valid_actions: Iterable[Any] | None = None,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Propose exactly one current-frame button click, or fail closed."""

    parsed = _parse_frame(frame)
    if parsed is None:
        return _blocked_plan("visible_cycle_roles_not_bound")
    if not _mouse_allowed(valid_actions):
        return _blocked_plan("mouse_action_not_available")

    evidence = _transition_evidence(parsed, context)
    models = {
        site: model
        for site, model in _induce_models(parsed, evidence).items()
        if model.certified
    }
    plan, search = _shortest_alignment_plan(parsed, models)
    if not plan:
        return _blocked_plan(str(search.get("blocked_reason", "no_certified_plan")))
    site = plan[0]
    action = {"action": "MOUSE", "row": int(site[1]), "col": int(site[0])}
    prediction = _prediction(parsed, action, models.get(site))
    if not prediction.get("accepted"):
        return _blocked_plan(str(prediction.get("reason", "prediction_not_exact")))

    return _json(
        {
            "schema_version": _PLAN_VERSION,
            "family_id": FAMILY_ID,
            "status": "proposed",
            "actions": [action],
            "action_count": 1,
            "certified": True,
            "certificate_reason": (
                "unique observed permutation, exact visible inverse, and full-token prediction"
            ),
            "transition_rule": "apply_reversible_visible_button_permutation",
            "progress_measure": {
                "metric": "remaining_alignment_clicks",
                "before": len(plan),
                "expected_after": len(plan) - 1,
                "strict": True,
            },
            "suffix_exposed": False,
            "requires_transition_verification": True,
            "first_prediction": prediction,
            "grounded_observation": {
                "entities": [
                    {
                        "role": "button",
                        "position": [int(site[0]), int(site[1])],
                    }
                ],
                "alignment_deficit": int(
                    _alignment_deficit(parsed.tokens, parsed.targets)
                ),
            },
            "detail": {
                "private_plan_length": len(plan),
                "search_expanded_count": int(search.get("expanded_count", 0)),
                "nontrivial_cycle_count": len(models[site].cycles),
                "stacked_membership": bool(
                    models[site].stacked_slots or len(models[site].cycles) > 1
                ),
            },
        }
    )


def predict_transition(
    frame: Any,
    action: Any,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict the exact complete token permutation for one bound click."""

    parsed = _parse_frame(frame)
    raw_site = _action_site(action)
    if parsed is None:
        return _unsupported_prediction("visible_cycle_roles_not_bound", raw_site)
    site = _bound_button_site(parsed, raw_site)
    if site is None:
        return _unsupported_prediction("action_is_not_a_bound_button", raw_site)
    evidence = _transition_evidence(parsed, context)
    model = _induce_models(parsed, evidence).get(site)
    if model is None:
        return _unsupported_prediction("button_permutation_not_identified", site)
    if not model.certified:
        return _unsupported_prediction("button_inverse_not_identified", site)
    return _prediction(parsed, _normalized_mouse(site), model)


def verify_transition(expected: Any, current_frame: Any) -> dict[str, Any]:
    """Verify the complete predicted assignment and reject unmodelled pixels."""

    if not isinstance(expected, Mapping) or expected.get("family_id") != FAMILY_ID:
        return _verification(False, "invalid_prediction_family")
    if not bool(expected.get("accepted")):
        return _verification(False, "unsupported_prediction")

    before = as_frame(expected.get("before_frame"))
    current = as_frame(current_frame)
    if before is None or current is None or before.shape != current.shape:
        return _verification(False, "invalid_transition_frames")
    if stable_digest(before) != str(expected.get("before_frame_digest", "")):
        return _verification(False, "before_frame_digest_mismatch")

    parsed = _parse_frame(current)
    if parsed is None:
        return _verification(False, "post_frame_roles_not_bound")
    if parsed.static_digest != str(expected.get("static_digest", "")):
        return _verification(False, "static_role_structure_changed")

    expected_tokens = _assignment_from_rows(expected.get("expected_tokens"))
    before_tokens = _assignment_from_rows(expected.get("before_tokens"))
    if expected_tokens is None or before_tokens is None:
        return _verification(False, "invalid_token_projection")
    if parsed.tokens != expected_tokens:
        return _verification(False, "token_assignment_mismatch")

    expected_counts = _counter_payload(Counter(expected_tokens.values()))
    if expected_counts != dict(expected.get("token_multiset_after") or {}):
        return _verification(False, "prediction_conservation_payload_invalid")
    if Counter(parsed.tokens.values()) != Counter(before_tokens.values()):
        return _verification(False, "token_conservation_failed")

    token_size = _safe_int(expected.get("token_size"), 0)
    if token_size <= 0 or token_size != parsed.token_size:
        return _verification(False, "token_extent_changed")
    rendered = _render_assignment(before, expected_tokens, token_size)
    volatile = expected.get("volatile_edges")
    volatile = volatile if isinstance(volatile, Mapping) else {}
    allowed = _edge_mask(current.shape, volatile)
    unexpected = (rendered != current) & ~allowed
    if np.any(unexpected):
        return _verification(
            False,
            "unmodelled_frame_delta",
            unexpected_pixel_count=int(np.count_nonzero(unexpected)),
        )

    actual_deficit = _alignment_deficit(parsed.tokens, parsed.targets)
    expected_deficit = _safe_int(expected.get("deficit_after"), -1)
    if actual_deficit != expected_deficit:
        return _verification(False, "alignment_deficit_mismatch")
    return _verification(
        True,
        "exact_reversible_token_permutation",
        changed_slot_count=int(expected.get("changed_slot_count", 0) or 0),
        alignment_deficit=int(actual_deficit),
        token_count=len(parsed.tokens),
    )


def _parse_frame(value: Any) -> _ParsedFrame | None:
    arr = as_frame(value)
    if arr is None or min(arr.shape) < 12:
        return None
    arr = np.asarray(arr)
    components = _button_cycle_components(arr)
    raw_targets = _button_cycle_hollow_targets(components)
    if not raw_targets:
        return None

    raw_tokens = _button_cycle_movable_square_token_positions(
        components,
        targets=raw_targets,
    )
    components_by_origin = {
        (int(row["min_x"]), int(row["min_y"])): row for row in components
    }
    size_counts: Counter[int] = Counter()
    token_candidates: list[tuple[Position, int, int]] = []
    for kind, positions in raw_tokens.items():
        for raw_position in positions:
            position = (int(raw_position[0]), int(raw_position[1]))
            component = components_by_origin.get(position)
            if component is None:
                continue
            width = int(component["width"])
            height = int(component["height"])
            area = int(component["area"])
            if width != height or area != width * height:
                continue
            size_counts[width] += 1
            token_candidates.append((position, int(kind), width))
    if not size_counts:
        return None
    token_size = max(size_counts, key=lambda size: (size_counts[size], size))
    tokens = {
        position: kind
        for position, kind, size in token_candidates
        if size == token_size
    }
    if len(tokens) < 3:
        return None

    targets_by_key: dict[tuple[int, Position], tuple[int, Position]] = {}
    for raw in raw_targets:
        try:
            kind = int(raw["value"])
            position = (int(raw["target_x"]), int(raw["target_y"]))
        except (KeyError, TypeError, ValueError):
            continue
        if position not in tokens or kind not in set(tokens.values()):
            continue
        targets_by_key[(kind, position)] = (kind, position)
    targets = tuple(
        sorted(targets_by_key.values(), key=lambda item: (*_position_key(item[1]), item[0]))
    )
    if not targets:
        return None
    token_counts = Counter(tokens.values())
    target_counts = Counter(kind for kind, _position in targets)
    if any(target_counts[kind] > token_counts[kind] for kind in target_counts):
        return None

    raw_buttons = _button_cycle_click_button_components(
        components,
        targets=raw_targets,
    )
    buttons: dict[Position, dict[str, int]] = {}
    for row in raw_buttons:
        site = (int(row["cx"]), int(row["cy"]))
        if site in tokens:
            continue
        buttons[site] = {
            "min_x": int(row["min_x"]),
            "min_y": int(row["min_y"]),
            "max_x": int(row["max_x"]),
            "max_y": int(row["max_y"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "area": int(row["area"]),
        }
    if len(buttons) < 2:
        return None

    static_payload = {
        "frame_shape": [int(value) for value in arr.shape],
        "token_size": int(token_size),
        "slots": [[int(x), int(y)] for x, y in sorted(tokens, key=_position_key)],
        "targets": [
            [int(kind), int(position[0]), int(position[1])]
            for kind, position in targets
        ],
        "buttons": [
            {
                "site": [int(site[0]), int(site[1])],
                "bbox": [
                    int(row["min_x"]),
                    int(row["min_y"]),
                    int(row["max_x"]),
                    int(row["max_y"]),
                ],
            }
            for site, row in sorted(buttons.items())
        ],
    }
    return _ParsedFrame(
        frame=np.asarray(arr).copy(),
        token_size=int(token_size),
        tokens=tokens,
        targets=targets,
        buttons=buttons,
        static_digest=stable_digest(static_payload),
    )


def _transition_evidence(
    current: _ParsedFrame,
    context: MechanicContext | Mapping[str, Any] | None,
) -> list[_TransitionEvidence]:
    records = _history_records(current.frame, context)
    evidence: list[_TransitionEvidence] = []
    seen: set[str] = set()
    parsed_cache: dict[tuple[tuple[int, ...], str, bytes], _ParsedFrame | None] = {}

    def cached_parse(frame: np.ndarray) -> _ParsedFrame | None:
        array = np.asarray(frame)
        key = (tuple(int(value) for value in array.shape), array.dtype.str, array.tobytes())
        if key not in parsed_cache:
            parsed_cache[key] = _parse_frame(array)
        return parsed_cache[key]

    for before_record, after_record in zip(records, records[1:]):
        raw_site = _action_site(after_record.action, after_record.raw)
        if raw_site is None:
            continue
        before = cached_parse(before_record.frame)
        after = cached_parse(after_record.frame)
        if before is None or after is None:
            continue
        site = _bound_button_site(before, raw_site)
        if (
            before.static_digest != current.static_digest
            or after.static_digest != current.static_digest
            or site is None
            or set(before.tokens) != set(after.tokens)
            or Counter(before.tokens.values()) != Counter(after.tokens.values())
            or before.tokens == after.tokens
        ):
            continue
        digest = stable_digest(
            {
                "site": site,
                "before": _token_rows(before.tokens),
                "after": _token_rows(after.tokens),
            }
        )
        if digest in seen:
            continue
        seen.add(digest)
        evidence.append(
            _TransitionEvidence(
                site=site,
                before=before.frame,
                after=after.frame,
                before_tokens=before.tokens,
                after_tokens=after.tokens,
                token_size=before.token_size,
                evidence_digest=digest,
            )
        )
    return evidence


def _history_records(
    current_frame: np.ndarray,
    context: MechanicContext | Mapping[str, Any] | None,
) -> list[_Record]:
    if isinstance(context, MechanicContext):
        history: Any = context.history
        previous_frame = context.previous_frame
        previous_action: Any = context.previous_action
        current_level = int(context.level)
    elif isinstance(context, Mapping):
        history = context.get("history", ())
        previous_frame = context.get("previous_frame")
        previous_action = context.get("previous_action", "")
        current_level = _safe_int(context.get("level"), 0)
    else:
        history = ()
        previous_frame = None
        previous_action = ""
        current_level = 0

    records: list[_Record] = []
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        for row in history[-_MAX_HISTORY_RECORDS:]:
            if not isinstance(row, Mapping):
                continue
            arr = as_frame(row.get("frame"))
            if arr is None:
                continue
            raw_level = row.get("level")
            level = _safe_int(raw_level, current_level) if raw_level is not None else None
            records.append(
                _Record(
                    frame=np.asarray(arr),
                    action=row.get("action", row.get("previous_action", "")),
                    level=level,
                    raw=row,
                )
            )

    current = np.asarray(current_frame)
    if records and np.array_equal(records[-1].frame, current):
        return records
    previous = as_frame(previous_frame)
    if previous is not None and (
        not records or not np.array_equal(records[-1].frame, previous)
    ):
        records.append(
            _Record(
                frame=np.asarray(previous),
                action="",
                level=current_level,
            )
        )
    records.append(
        _Record(
            frame=current,
            action=previous_action,
            level=current_level,
        )
    )
    return records[-_MAX_HISTORY_RECORDS:]


def _induce_models(
    parsed: _ParsedFrame,
    evidence: Sequence[_TransitionEvidence],
) -> dict[Position, _PermutationModel]:
    by_site: dict[Position, list[_TransitionEvidence]] = defaultdict(list)
    for row in evidence:
        if row.site in parsed.buttons:
            by_site[row.site].append(row)

    models: dict[Position, _PermutationModel] = {}
    slots = tuple(sorted(parsed.tokens, key=_position_key))
    for site, rows in by_site.items():
        if not rows:
            continue
        source_groups: dict[tuple[int, ...], list[Position]] = defaultdict(list)
        dest_groups: dict[tuple[int, ...], list[Position]] = defaultdict(list)
        for source in slots:
            source_groups[tuple(row.before_tokens[source] for row in rows)].append(
                source
            )
        for dest in slots:
            dest_groups[tuple(row.after_tokens[dest] for row in rows)].append(dest)
        source_to_dest: dict[Position, Position] | None = None
        if len(rows) >= _MIN_PERMUTATION_SUPPORT and set(source_groups) == set(
            dest_groups
        ) and all(
            len(source_groups[signature]) == 1
            and len(dest_groups[signature]) == 1
            for signature in source_groups
        ):
            source_to_dest = {
                sources[0]: dest_groups[signature][0]
                for signature, sources in source_groups.items()
            }
        if source_to_dest is None:
            source_to_dest = _unique_visible_cycle_permutation(slots, rows)
        if source_to_dest is None:
            continue
        if set(source_to_dest) != set(slots) or set(source_to_dest.values()) != set(
            slots
        ):
            continue
        if not all(
            _apply_assignment(row.before_tokens, source_to_dest) == row.after_tokens
            for row in rows
        ):
            continue
        cycles = _permutation_cycles(source_to_dest)
        if not cycles:
            continue
        volatile = _volatile_edges(rows, source_to_dest)
        if volatile is None:
            continue
        models[site] = _PermutationModel(
            site=site,
            source_to_dest=source_to_dest,
            cycles=cycles,
            support_count=len(rows),
            volatile_edges=volatile,
            evidence_digests=tuple(row.evidence_digest for row in rows),
        )

    for site, model in models.items():
        inverses = [
            other_site
            for other_site, other in models.items()
            if other_site != site
            and all(
                other.source_to_dest.get(dest) == source
                for source, dest in model.source_to_dest.items()
            )
        ]
        model.inverse_sites = tuple(sorted(inverses, key=_position_key))
    cycle_supports = {
        frozenset(cycle)
        for model in models.values()
        if model.certified
        for cycle in model.cycles
    }
    memberships: Counter[Position] = Counter(
        position for support in cycle_supports for position in support
    )
    for model in models.values():
        model.stacked_slots = tuple(
            sorted(
                {
                    position
                    for cycle in model.cycles
                    for position in cycle
                    if memberships[position] > 1
                },
                key=_position_key,
            )
        )
    return models


def _unique_visible_cycle_permutation(
    slots: Sequence[Position],
    rows: Sequence[_TransitionEvidence],
) -> dict[Position, Position] | None:
    """Fit one visible line/perimeter rotation, rejecting any ambiguity."""

    matches: dict[str, dict[Position, Position]] = {}
    for cycle in _visible_cycle_candidates(slots):
        for delta in (-1, 1):
            source_to_dest = {position: position for position in slots}
            for index, source in enumerate(cycle):
                source_to_dest[source] = cycle[(index + delta) % len(cycle)]
            if not all(
                _apply_assignment(row.before_tokens, source_to_dest)
                == row.after_tokens
                for row in rows
            ):
                continue
            digest = stable_digest(sorted(source_to_dest.items()))
            matches[digest] = source_to_dest
    if len(matches) != 1:
        return None
    return next(iter(matches.values()))


def _visible_cycle_candidates(
    raw_slots: Sequence[Position],
) -> tuple[tuple[Position, ...], ...]:
    """Enumerate palette-free cycles directly supported by the slot lattice."""

    slots = set(raw_slots)
    if len(slots) < 3:
        return ()
    deltas: list[int] = []
    for axis in (0, 1):
        values = sorted({position[axis] for position in slots})
        deltas.extend(
            right - left
            for left, right in zip(values, values[1:])
            if right > left
        )
    if not deltas:
        return ()
    stride = deltas[0]
    for value in deltas[1:]:
        stride = gcd(stride, value)
    if stride <= 0:
        return ()

    candidates: dict[tuple[Position, ...], tuple[Position, ...]] = {}
    by_y: dict[int, list[int]] = defaultdict(list)
    by_x: dict[int, list[int]] = defaultdict(list)
    for x, y in slots:
        by_y[y].append(x)
        by_x[x].append(y)

    def add_runs(groups: Mapping[int, Sequence[int]], *, horizontal: bool) -> None:
        for fixed, raw_values in groups.items():
            values = sorted(set(int(value) for value in raw_values))
            run: list[int] = []
            for value in [*values, values[-1] + stride + 1]:
                if not run or value - run[-1] == stride:
                    run.append(value)
                    continue
                if len(run) >= 3:
                    cycle = tuple(
                        (item, fixed) if horizontal else (fixed, item)
                        for item in run
                    )
                    candidates[cycle] = cycle
                run = [value]

    add_runs(by_y, horizontal=True)
    add_runs(by_x, horizontal=False)

    xs = sorted({x for x, _y in slots})
    ys = sorted({y for _x, y in slots})
    for left_index, x0 in enumerate(xs):
        for x1 in xs[left_index + 1 :]:
            if (x1 - x0) % stride:
                continue
            for top_index, y0 in enumerate(ys):
                for y1 in ys[top_index + 1 :]:
                    if (y1 - y0) % stride:
                        continue
                    cycle = (
                        *((x, y0) for x in range(x0, x1 + stride, stride)),
                        *((x1, y) for y in range(y0 + stride, y1 + stride, stride)),
                        *((x, y1) for x in range(x1 - stride, x0 - stride, -stride)),
                        *((x0, y) for y in range(y1 - stride, y0, -stride)),
                    )
                    if len(cycle) >= 8 and set(cycle).issubset(slots):
                        candidates[cycle] = cycle
    return tuple(candidates)


def _volatile_edges(
    rows: Sequence[_TransitionEvidence],
    source_to_dest: Mapping[Position, Position],
) -> dict[str, int] | None:
    bands: dict[str, int] = {}
    for row in rows:
        expected_tokens = _apply_assignment(row.before_tokens, source_to_dest)
        rendered = _render_assignment(row.before, expected_tokens, row.token_size)
        residual_y, residual_x = np.where(rendered != row.after)
        height, width = row.after.shape
        maximum_band = max(1, min(height, width) // 10)
        for y_value, x_value in zip(residual_y.tolist(), residual_x.tolist()):
            distances = {
                "top": int(y_value),
                "bottom": int(height - 1 - y_value),
                "left": int(x_value),
                "right": int(width - 1 - x_value),
            }
            edge, distance = min(distances.items(), key=lambda item: (item[1], item[0]))
            thickness = int(distance) + 1
            if thickness > maximum_band:
                return None
            bands[edge] = max(int(bands.get(edge, 0)), thickness)
    return dict(sorted(bands.items()))


def _permutation_cycles(
    source_to_dest: Mapping[Position, Position],
) -> tuple[tuple[Position, ...], ...]:
    seen: set[Position] = set()
    cycles: list[tuple[Position, ...]] = []
    for start in sorted(source_to_dest, key=_position_key):
        if start in seen:
            continue
        cycle: list[Position] = []
        current = start
        while current not in seen:
            seen.add(current)
            cycle.append(current)
            next_position = source_to_dest.get(current)
            if next_position is None:
                return ()
            current = next_position
        if current != start:
            return ()
        if len(cycle) > 1:
            minimum_index = min(
                range(len(cycle)), key=lambda index: _position_key(cycle[index])
            )
            normalized = tuple(cycle[minimum_index:] + cycle[:minimum_index])
            cycles.append(normalized)
    return tuple(sorted(cycles, key=lambda cycle: (_position_key(cycle[0]), len(cycle))))


def _shortest_alignment_plan(
    parsed: _ParsedFrame,
    models: Mapping[Position, _PermutationModel],
) -> tuple[list[Position] | None, dict[str, Any]]:
    target_kinds = tuple(sorted({kind for kind, _position in parsed.targets}))
    start = tuple(
        tuple(
            sorted(
                (position for position, value in parsed.tokens.items() if value == kind),
                key=_position_key,
            )
        )
        for kind in target_kinds
    )
    target_by_kind = {
        kind: {
            position for target_kind, position in parsed.targets if target_kind == kind
        }
        for kind in target_kinds
    }
    diagnostic: dict[str, Any] = {
        "found": False,
        "expanded_count": 0,
        "model_count": len(models),
        "target_kind_count": len(target_kinds),
        "max_states": _MAX_SEARCH_STATES,
        "max_depth": _MAX_SEARCH_DEPTH,
    }
    if _target_state_satisfied(start, target_kinds, target_by_kind):
        diagnostic.update({"found": True, "plan_length": 0, "blocked_reason": "goal_already_satisfied"})
        return [], diagnostic
    if not models:
        diagnostic["blocked_reason"] = "reversible_permutation_evidence_incomplete"
        return None, diagnostic

    queue: deque[tuple[tuple[tuple[Position, ...], ...], list[Position]]] = deque(
        [(start, [])]
    )
    seen = {start}
    ordered_models = [models[site] for site in sorted(models, key=_position_key)]
    while queue:
        state, path = queue.popleft()
        diagnostic["expanded_count"] = int(diagnostic["expanded_count"]) + 1
        if int(diagnostic["expanded_count"]) > _MAX_SEARCH_STATES:
            diagnostic["blocked_reason"] = "state_expansion_limit"
            return None, diagnostic
        if len(path) >= _MAX_SEARCH_DEPTH:
            continue
        for model in ordered_models:
            next_state = tuple(
                tuple(
                    sorted(
                        (model.source_to_dest[position] for position in positions),
                        key=_position_key,
                    )
                )
                for positions in state
            )
            if next_state in seen:
                continue
            next_path = [*path, model.site]
            if _target_state_satisfied(next_state, target_kinds, target_by_kind):
                diagnostic.update(
                    {
                        "found": True,
                        "plan_length": len(next_path),
                        "blocked_reason": "",
                    }
                )
                return next_path, diagnostic
            seen.add(next_state)
            queue.append((next_state, next_path))
    diagnostic["blocked_reason"] = "alignment_state_not_reachable"
    return None, diagnostic


def _target_state_satisfied(
    state: Sequence[Sequence[Position]],
    target_kinds: Sequence[int],
    target_by_kind: Mapping[int, set[Position]],
) -> bool:
    return all(
        target_by_kind[kind].issubset(set(state[index]))
        for index, kind in enumerate(target_kinds)
    )


def _prediction(
    parsed: _ParsedFrame,
    action: Mapping[str, Any],
    model: _PermutationModel | None,
) -> dict[str, Any]:
    raw_site = _action_site(action)
    site = _bound_button_site(parsed, raw_site)
    if model is None or site is None or model.site != site or not model.certified:
        return _unsupported_prediction("button_permutation_not_certified", raw_site)
    expected_tokens = _apply_assignment(parsed.tokens, model.source_to_dest)
    if set(expected_tokens) != set(parsed.tokens):
        return _unsupported_prediction("predicted_slot_set_changed", site)
    before_counts = Counter(parsed.tokens.values())
    after_counts = Counter(expected_tokens.values())
    if before_counts != after_counts:
        return _unsupported_prediction("predicted_token_conservation_failed", site)

    deficit_before = _alignment_deficit(parsed.tokens, parsed.targets)
    deficit_after = _alignment_deficit(expected_tokens, parsed.targets)
    changed_slots = sum(
        parsed.tokens[position] != expected_tokens[position] for position in parsed.tokens
    )
    return _json(
        {
            "schema_version": _PREDICTION_VERSION,
            "family_id": FAMILY_ID,
            "accepted": True,
            "supported": True,
            "reason": "exact_reversible_button_permutation",
            "operation": "apply_button_permutation",
            "action": _normalized_mouse(site),
            "button_site": [int(site[0]), int(site[1])],
            "inverse_sites": [
                [int(position[0]), int(position[1])]
                for position in model.inverse_sites
            ],
            "before_frame": parsed.frame.tolist(),
            "before_frame_digest": stable_digest(parsed.frame),
            "static_digest": parsed.static_digest,
            "token_size": int(parsed.token_size),
            "before_tokens": _token_rows(parsed.tokens),
            "expected_tokens": _token_rows(expected_tokens),
            "token_multiset_before": _counter_payload(before_counts),
            "token_multiset_after": _counter_payload(after_counts),
            "permutation": [
                {
                    "source": [int(source[0]), int(source[1])],
                    "dest": [int(dest[0]), int(dest[1])],
                }
                for source, dest in sorted(model.source_to_dest.items())
            ],
            "cycles": [
                [[int(x), int(y)] for x, y in cycle] for cycle in model.cycles
            ],
            "nontrivial_cycle_count": len(model.cycles),
            "stacked_membership": bool(model.stacked_slots or len(model.cycles) > 1),
            "stacked_slots": [
                [int(position[0]), int(position[1])]
                for position in model.stacked_slots
            ],
            "support_count": int(model.support_count),
            "model_digest": stable_digest(
                {
                    "permutation": sorted(model.source_to_dest.items()),
                    "inverse_sites": model.inverse_sites,
                    "evidence": model.evidence_digests,
                }
            ),
            "volatile_edges": dict(model.volatile_edges),
            "changed_slot_count": int(changed_slots),
            "deficit_before": int(deficit_before),
            "deficit_after": int(deficit_after),
            "deficit_delta": int(deficit_before - deficit_after),
        }
    )


def _apply_assignment(
    tokens: Mapping[Position, int],
    source_to_dest: Mapping[Position, Position],
) -> dict[Position, int]:
    return {
        source_to_dest[source]: int(kind) for source, kind in tokens.items()
    }


def _render_assignment(
    frame: np.ndarray,
    tokens: Mapping[Position, int],
    token_size: int,
) -> np.ndarray:
    rendered = np.asarray(frame).copy()
    height, width = rendered.shape
    for (x, y), kind in tokens.items():
        if x < 0 or y < 0 or x + token_size > width or y + token_size > height:
            continue
        rendered[y : y + token_size, x : x + token_size] = int(kind)
    return rendered


def _edge_mask(shape: Sequence[int], volatile: Mapping[str, Any]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=bool)
    for edge in ("top", "bottom", "left", "right"):
        thickness = max(0, _safe_int(volatile.get(edge), 0))
        if thickness <= 0:
            continue
        if edge == "top":
            mask[: min(height, thickness), :] = True
        elif edge == "bottom":
            mask[max(0, height - thickness) :, :] = True
        elif edge == "left":
            mask[:, : min(width, thickness)] = True
        else:
            mask[:, max(0, width - thickness) :] = True
    return mask


def _alignment_deficit(
    tokens: Mapping[Position, int],
    targets: Sequence[tuple[int, Position]],
) -> int:
    return sum(tokens.get(position) != kind for kind, position in targets)


def _kind_roles(parsed: _ParsedFrame) -> dict[int, str]:
    target_order: list[int] = []
    for kind, _position in parsed.targets:
        if kind not in target_order:
            target_order.append(kind)
    other = sorted(set(parsed.tokens.values()) - set(target_order))
    return {
        kind: f"kind_{index}"
        for index, kind in enumerate([*target_order, *other])
    }


def _mismatch_cycles(
    parsed: _ParsedFrame,
    models: Mapping[Position, _PermutationModel],
    kind_roles: Mapping[int, str],
) -> list[dict[str, Any]]:
    mismatches = [
        (kind, position)
        for kind, position in parsed.targets
        if parsed.tokens.get(position) != kind
    ]
    rows: list[dict[str, Any]] = []
    for site, model in sorted(models.items()):
        for membership_index, cycle in enumerate(model.cycles):
            target_rows = [
                {
                    "kind_role": kind_roles[kind],
                    "position": [int(position[0]), int(position[1])],
                }
                for kind, position in mismatches
                if position in cycle
            ]
            target_token_count = sum(
                parsed.tokens[position] in {kind for kind, _target in mismatches}
                for position in cycle
            )
            if not target_rows and target_token_count == 0:
                continue
            rows.append(
                {
                    "button_site": [int(site[0]), int(site[1])],
                    "membership_index": int(membership_index),
                    "cycle_length": len(cycle),
                    "mismatched_targets": target_rows,
                    "target_kind_token_count": int(target_token_count),
                }
            )
    return rows


def _button_binding_row(
    parsed: _ParsedFrame,
    model: _PermutationModel | None,
    site: Position,
) -> dict[str, Any]:
    button = parsed.buttons[site]
    return {
        "role": "button",
        "site": [int(site[0]), int(site[1])],
        "bbox": [
            int(button["min_x"]),
            int(button["min_y"]),
            int(button["max_x"]),
            int(button["max_y"]),
        ],
        "permutation_identified": model is not None,
        "reversible": bool(model and model.certified),
        "support_count": int(model.support_count if model else 0),
        "nontrivial_cycle_count": len(model.cycles) if model else 0,
        "stacked_membership": bool(
            model and (model.stacked_slots or len(model.cycles) > 1)
        ),
        "stacked_slot_count": len(model.stacked_slots) if model else 0,
    }


def _model_summary(model: _PermutationModel) -> dict[str, Any]:
    return {
        "button_site": [int(model.site[0]), int(model.site[1])],
        "support_count": int(model.support_count),
        "slot_count": len(model.source_to_dest),
        "nontrivial_cycle_count": len(model.cycles),
        "cycle_lengths": [len(cycle) for cycle in model.cycles],
        "stacked_membership": bool(model.stacked_slots or len(model.cycles) > 1),
        "stacked_slot_count": len(model.stacked_slots),
        "inverse_button_count": len(model.inverse_sites),
        "inverse_sites": [
            [int(position[0]), int(position[1])] for position in model.inverse_sites
        ],
        "volatile_edges": dict(model.volatile_edges),
    }


def _action_site(action: Any, row: Mapping[str, Any] | None = None) -> Position | None:
    candidates: list[Any] = [action]
    if isinstance(row, Mapping):
        candidates.extend(
            [
                row,
                row.get("action_payload"),
                row.get("payload"),
                row.get("data"),
            ]
        )
    action_data = getattr(action, "data", None)
    if action_data is not None:
        candidates.append(action_data)

    x_value: Any = None
    y_value: Any = None
    action_texts: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            action_texts.append(
                str(
                    candidate.get(
                        "action",
                        candidate.get("action_name", candidate.get("id", "")),
                    )
                )
            )
            data = candidate.get("data")
            if isinstance(data, Mapping):
                candidates.append(data)
            if x_value is None:
                x_value = candidate.get("x", candidate.get("col"))
            if y_value is None:
                y_value = candidate.get("y", candidate.get("row"))
        elif candidate is not None:
            action_texts.append(str(candidate))
    text = " ".join(action_texts)
    if x_value is None:
        match = re.search(r"[\"']?x[\"']?\s*[:=]\s*(-?\d+)", text)
        if match:
            x_value = match.group(1)
    if y_value is None:
        match = re.search(r"[\"']?y[\"']?\s*[:=]\s*(-?\d+)", text)
        if match:
            y_value = match.group(1)
    action_name = text.upper()
    click_action = bool(
        "ACTION6" in action_name
        or "MOUSE" in action_name
        or re.search(r"\bID\s*=\s*6\b", action_name)
        or (not action_name.strip() and x_value is not None and y_value is not None)
    )
    if not click_action:
        return None
    try:
        return int(x_value), int(y_value)
    except (TypeError, ValueError):
        return None


def _bound_button_site(
    parsed: _ParsedFrame,
    raw_site: Position | None,
) -> Position | None:
    if raw_site is None:
        return None
    if raw_site in parsed.buttons:
        return raw_site
    x, y = raw_site
    candidates = [
        site
        for site, button in parsed.buttons.items()
        if int(button["min_x"]) <= x <= int(button["max_x"])
        and int(button["min_y"]) <= y <= int(button["max_y"])
    ]
    if len(candidates) != 1:
        return None
    return candidates[0]


def _mouse_allowed(valid_actions: Iterable[Any] | None) -> bool:
    if valid_actions is None:
        return True
    for action in valid_actions:
        text = str(getattr(action, "name", action)).upper()
        if "MOUSE" in text or "ACTION6" in text or text.strip() in {"6", "CLICK"}:
            return True
    return False


def _normalized_mouse(site: Position) -> dict[str, Any]:
    return {"action": "MOUSE", "row": int(site[1]), "col": int(site[0])}


def _assignment_from_rows(value: Any) -> dict[Position, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    assignment: dict[Position, int] = {}
    for row in value:
        if not isinstance(row, Mapping):
            return None
        position = row.get("position")
        if not isinstance(position, Sequence) or len(position) < 2:
            return None
        try:
            site = (int(position[0]), int(position[1]))
            kind = int(row["kind"])
        except (KeyError, TypeError, ValueError):
            return None
        if site in assignment:
            return None
        assignment[site] = kind
    return assignment


def _token_rows(tokens: Mapping[Position, int]) -> list[dict[str, Any]]:
    return [
        {
            "position": [int(position[0]), int(position[1])],
            "kind": int(kind),
        }
        for position, kind in sorted(tokens.items(), key=_position_item_key)
    ]


def _counter_payload(counter: Counter[int]) -> dict[str, int]:
    return {str(kind): int(count) for kind, count in sorted(counter.items())}


def _position_key(position: Position) -> tuple[int, int]:
    return int(position[1]), int(position[0])


def _position_item_key(item: tuple[Position, Any]) -> tuple[int, int]:
    return _position_key(item[0])


def _level_ordinal(level: Any) -> int | None:
    if isinstance(level, (int, np.integer)):
        return int(level)
    if isinstance(level, Mapping):
        for key in ("ordinal", "level", "level_number", "index"):
            if isinstance(level.get(key), (int, np.integer)):
                return int(level[key])
    return None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _json(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json_safe_mapping(value)
    if normalized is None:
        raise ValueError("cycle-permutation payload is not JSON-safe")
    return normalized


def _unbound(reason: str) -> dict[str, Any]:
    return _json(
        {
            "schema_version": _BINDING_VERSION,
            "family_id": FAMILY_ID,
            "supported": False,
            "executable": False,
            "status": "unbound",
            "reason": str(reason),
            "role_bindings": [],
            "tokens": [],
            "targets": [],
            "buttons": [],
            "permutation_models": [],
            "mismatch_cycles": [],
            "transition_evidence_count": 0,
            "reversible_button_count": 0,
            "stacked_membership_button_count": 0,
        }
    )


def _blocked_plan(reason: str) -> dict[str, Any]:
    return _json(
        {
            "schema_version": _PLAN_VERSION,
            "family_id": FAMILY_ID,
            "status": "blocked",
            "actions": [],
            "action_count": 0,
            "certified": False,
            "certificate_reason": str(reason),
            "suffix_exposed": False,
            "detail": {"private_plan_length": 0},
        }
    )


def _unsupported_prediction(reason: str, site: Position | None) -> dict[str, Any]:
    return _json(
        {
            "schema_version": _PREDICTION_VERSION,
            "family_id": FAMILY_ID,
            "accepted": False,
            "supported": False,
            "reason": str(reason),
            "button_site": (
                [int(site[0]), int(site[1])] if site is not None else None
            ),
        }
    )


def _verification(accepted: bool, reason: str, **detail: Any) -> dict[str, Any]:
    return _json(
        {
            "schema_version": _VERIFICATION_VERSION,
            "family_id": FAMILY_ID,
            "accepted": bool(accepted),
            "reason": str(reason),
            **detail,
        }
    )


__all__ = [
    "FAMILY_ID",
    "bind_current",
    "compile_schema",
    "predict_transition",
    "propose_plan",
    "verify_transition",
]
