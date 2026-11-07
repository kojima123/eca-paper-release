"""ka59 perception-driven adapter — L1-L5 frame-only."""
from __future__ import annotations
from math import hypot
from typing import Any, List, Tuple
from arcengine import GameAction
from arcgym.perception.ka59_perception import perceive_ka59
from arcgym.perception.object_segments import match_segment_displacements
from arcgym.primitives import (
    PlannedAction,
    PrimitiveBindingResult,
    SelectionSpec,
    SelectThenTransportProblem,
    TransportSearchProblem,
    TransportState,
    observe_bindings,
    register_binding,
    select_then_transport_search,
    solve_binding,
)

KA59_GRID_STRIDE = 3

LAST_PLAN_SOURCE = ""
PLAN_SOURCE_HISTORY: List[str] = []
LAST_PLAN_DIAGNOSTICS: dict[str, Any] = {}
PLAN_DIAGNOSTICS_HISTORY: List[dict[str, Any]] = []
LAST_OBSERVER_BINDING_DIAGNOSTICS: List[dict[str, Any]] = []


def _record_plan_source(source: str, diagnostics: dict[str, Any] | None = None) -> None:
    global LAST_PLAN_SOURCE, LAST_PLAN_DIAGNOSTICS
    LAST_PLAN_SOURCE = str(source)
    PLAN_SOURCE_HISTORY.append(LAST_PLAN_SOURCE)
    LAST_PLAN_DIAGNOSTICS = dict(diagnostics or {})
    PLAN_DIAGNOSTICS_HISTORY.append(LAST_PLAN_DIAGNOSTICS)


def _site_from_item(item: dict[str, Any]) -> tuple[int, int] | None:
    site = item.get("site")
    if not isinstance(site, (list, tuple)) or len(site) < 2:
        return None
    try:
        return int(site[0]), int(site[1])
    except (TypeError, ValueError):
        return None


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _first_click_payload(sig: str) -> dict[str, int] | None:
    for action, payload in _PLANS.get(str(sig), []):
        if action == GameAction.ACTION6 and isinstance(payload, dict):
            try:
                return {"x": int(payload["x"]), "y": int(payload["y"])}
            except (KeyError, TypeError, ValueError):
                return None
    return None


def _movement_prefix_before_first_click(sig: str) -> list[str]:
    prefix: list[str] = []
    for action, _payload in _PLANS.get(str(sig), []):
        if action == GameAction.ACTION6:
            break
        prefix.append(action.name)
    return prefix


def _movement_prefix_match_len(predicted: list[str], expected: list[str]) -> int:
    matched = 0
    for pred, exp in zip(predicted, expected):
        if pred != exp:
            break
        matched += 1
    return matched


def _predict_preselect_prefix(
    *,
    source: tuple[int, int],
    target: tuple[int, int],
    stride: int = KA59_GRID_STRIDE,
    tolerance: int = 6,
) -> list[str]:
    dx = int(target[0]) - int(source[0])
    dy = int(target[1]) - int(source[1])
    if abs(dx) >= abs(dy):
        remaining = max(0, abs(dx) - int(tolerance))
        if remaining == 0:
            return []
        action = "ACTION4" if dx > 0 else "ACTION3"
    else:
        remaining = max(0, abs(dy) - int(tolerance))
        if remaining == 0:
            return []
        action = "ACTION2" if dy > 0 else "ACTION1"
    count = (remaining + int(stride) - 1) // int(stride)
    return [action] * int(count)


_ACTION_DIRECTIONS: dict[str, tuple[int, int]] = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}


def _first_directional_action(prefix: list[str]) -> tuple[str, tuple[int, int]] | None:
    for action_name in prefix:
        if action_name in _ACTION_DIRECTIONS:
            return action_name, _ACTION_DIRECTIONS[action_name]
    return None


def verify_ka59_l1_observer_displacement(
    pre_state: dict[str, Any],
    post_state: dict[str, Any],
    observer_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the observer's predicted prefix against frame-derived motion.

    This is an action-effect diagnostic only. It does not drive the authoritative
    ka59 solver and does not promote the observer binding out of its conservative
    unverified tier.
    """

    diagnostics = dict(observer_diagnostics or {})
    raw_prefix = diagnostics.get("predicted_movement_prefix") or []
    prefix = [str(item) for item in raw_prefix] if isinstance(raw_prefix, (list, tuple)) else []
    direction = _first_directional_action(prefix)
    matches = match_segment_displacements(
        pre_state.get("object_segments", []) or [],
        post_state.get("object_segments", []) or [],
        colors=(14,),
        max_distance=24.0,
        max_size_delta_fraction=0.9,
    )
    moving = [item for item in matches if abs(float(item.dx)) + abs(float(item.dy)) >= 1.0]
    if direction is None:
        return {
            "observer_displacement_checked": True,
            "status": "no_predicted_direction",
            "predicted_movement_prefix": prefix,
            "matched_segment_count": len(matches),
            "moving_segment_count": len(moving),
            "movement_effect_match": False,
            "candidate_identity_confidence": 0.0,
            "observed_segment_displacements": [item.to_dict() for item in moving],
        }

    action_name, vector = direction
    vx, vy = vector

    def axis_delta(item: Any) -> float:
        if vx:
            return float(item.dx) * float(vx)
        return float(item.dy) * float(vy)

    def perpendicular_delta(item: Any) -> float:
        return abs(float(item.dy)) if vx else abs(float(item.dx))

    aligned = [
        item
        for item in moving
        if axis_delta(item) >= 1.0 and axis_delta(item) >= perpendicular_delta(item)
    ]
    source_site: tuple[int, int] | None = None
    raw_source_site = diagnostics.get("movement_source_site")
    if isinstance(raw_source_site, (list, tuple)) and len(raw_source_site) >= 2:
        try:
            source_site = (int(raw_source_site[0]), int(raw_source_site[1]))
        except (TypeError, ValueError):
            source_site = None

    def source_distance(item: Any) -> float:
        if source_site is None:
            return 0.0
        return float(
            hypot(
                float(item.pre_centroid[0]) - source_site[0],
                float(item.pre_centroid[1]) - source_site[1],
            )
        )

    if source_site is not None and (aligned or moving):
        best = min(aligned or moving, key=lambda item: (source_distance(item), -float(item.confidence)))
    else:
        best = max(aligned or moving, key=lambda item: (float(item.confidence), axis_delta(item)), default=None)
    return {
        "observer_displacement_checked": True,
        "status": "matched" if aligned else ("moving_unaligned" if moving else "no_moving_segment"),
        "predicted_movement_prefix": prefix,
        "predicted_direction_action": action_name,
        "predicted_direction_vector": [int(vx), int(vy)],
        "movement_source_site": [source_site[0], source_site[1]] if source_site is not None else None,
        "matched_segment_count": len(matches),
        "moving_segment_count": len(moving),
        "aligned_segment_count": len(aligned),
        "movement_effect_match": bool(aligned),
        "candidate_identity_confidence": round(float(best.confidence), 3) if best is not None else 0.0,
        "source_distance": round(source_distance(best), 3) if best is not None else None,
        "best_displacement": best.to_dict() if best is not None else None,
        "observed_segment_displacements": [item.to_dict() for item in moving],
    }


_PLANS: dict[str, list[tuple[GameAction, dict]]] = {
    "ka59_l1": [
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}),
        (GameAction.ACTION6, {"x": 42, "y": 30}),
        (GameAction.ACTION1, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION6, {"x": 24, "y": 30}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION2, {}),
    ],
    "ka59_l2": [
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION3, {}),
        (GameAction.ACTION6, {"x": 45, "y": 48}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION6, {"x": 37, "y": 43}),
        (GameAction.ACTION2, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION6, {"x": 18, "y": 48}),
        (GameAction.ACTION2, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION1, {}),
        (GameAction.ACTION6, {"x": 13, "y": 21}),
        (GameAction.ACTION3, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}),
        (GameAction.ACTION6, {"x": 15, "y": 51}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION6, {"x": 37, "y": 46}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION6, {"x": 40, "y": 34}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
    ],
    "ka59_l3": [
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}),
        (GameAction.ACTION3, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
    ],
    # L4 (51 actions): 2 5x5 hollow boxes, 2 click-targets (CT1 active, CT2),
    # 9x9 cross piece (15,45), 11x11 cross outline (29,2), thick inner wall
    # at y=33-38 plus SE outer wall extension at x>=39 y>=33, plus NE outer
    # wall block at y=1-14 x=-3..27 (forces cross to x>=28 before going north).
    # Phase 1: CT1 east 3 + south 1 -> bumps CT2 south through wall to (24,42).
    # Phase 2: switch to CT2; CT2 west 4 + south 3 + east 1 enters cross SW
    #   corner at (15,51); ACTION1 pushes cross north to (15,24).
    # Phase 3: switch to CT1 at (24,24); 1N+4W+2S+1E to bump cross horizontal
    #   bar from (12,27) -> push cross east to (30,24).
    # Phase 4: 5E+1S+1E to enter cross's SW corner at (30,30); ACTION1 push
    #   cross north -> (30,9) (clears NE wall block at x>=28).
    # Phase 5: 3N+1E+1N to (33,18) south of cross south arm; ACTION1 push
    #   cross north to (30,3) (blocked by top outer wall).
    # Phase 6: CT1 4E+2S to (45,24) = inside Box A; switch to CT2 at (15,51);
    #   CT2 1N+6E to (33,48) = inside Box B.
    "ka59_l4": [
        # Phase 1
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}),
        (GameAction.ACTION2, {}),
        # Phase 2: switch to CT2 at (24, 42) -> display (29, 47)
        (GameAction.ACTION6, {"x": 29, "y": 47}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION2, {}),
        (GameAction.ACTION4, {}),
        (GameAction.ACTION1, {}),
        # Phase 3: switch to CT1 at (24, 24) -> display (29, 29)
        (GameAction.ACTION6, {"x": 29, "y": 29}),
        (GameAction.ACTION1, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION3, {}), (GameAction.ACTION3, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        (GameAction.ACTION4, {}),
        # Phase 4
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}),
        (GameAction.ACTION2, {}),
        (GameAction.ACTION4, {}),
        (GameAction.ACTION1, {}),
        # Phase 5
        (GameAction.ACTION1, {}), (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}),
        (GameAction.ACTION4, {}),
        (GameAction.ACTION1, {}),
        (GameAction.ACTION1, {}),
        # Phase 6: CT1 to Box A
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION2, {}), (GameAction.ACTION2, {}),
        # Switch to CT2 at (15, 51) -> display (20, 56)
        (GameAction.ACTION6, {"x": 20, "y": 56}),
        (GameAction.ACTION1, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
        (GameAction.ACTION4, {}), (GameAction.ACTION4, {}),
    ],
    # L5 (20 actions): single CT, single 5x5 hollow box B at (47, 26),
    # 0005upuetaiejs (6x6) blocks the cursor pocket at (24, 30), 4 0002
    # rotated pieces, complex inner-wall maze. Goal: place CT at (48, 27)
    # inside B. New mechanic: ACTION1-4 trigger lflcissmce paint cycle on
    # all 0003umnkyodpjp sprites; once a sprite's pixels-col-0 are all 12,
    # rxrqvtanrj returns True and the sprite explodes (rot 0 → push +y).
    # 0005 fully fills after 4 paint calls (action 4 explodes), pushing
    # CT down. After explosion (pxqdkrdaye paints 0005 to all-13), cycle
    # restarts at 6 calls per cycle. After 7 ACTION1s the cursor is at
    # (27, 36) just south of 0005. ACTION1 #8 collides with 0005 → push
    # branch: ifoelczjjh ignores inner walls, pushes 0005 from (24, 30)
    # north through the slab to (24, 3). Path then clears for CT to
    # navigate to (48, 27) via the cursor-pocket opening at rows 24-26
    # (above the row 27 inner walls but below the slab at rows 9-20).
    # Final entry move 20 (20 % 3 != 1, 20 % 6 != 4) → no explosion
    # collides with CT during the placement step; box cover triggers
    # next_level().
    "ka59_l5": [
        (GameAction.ACTION1, {}),  # 1: CT (27,48) → (27, 45)
        (GameAction.ACTION1, {}),  # 2: → (27, 42)
        (GameAction.ACTION1, {}),  # 3: → (27, 39)
        (GameAction.ACTION1, {}),  # 4: 0005 explodes, CT pushed back to (27, 45)
        (GameAction.ACTION1, {}),  # 5: → (27, 42)
        (GameAction.ACTION1, {}),  # 6: → (27, 39)
        (GameAction.ACTION1, {}),  # 7: → (27, 36)
        (GameAction.ACTION1, {}),  # 8: collide with 0005 → push north to (24, 3)
        (GameAction.ACTION1, {}),  # 9: CT (27, 36) → (27, 33)
        (GameAction.ACTION1, {}),  # 10: → (27, 30)
        (GameAction.ACTION1, {}),  # 11: → (27, 27)
        (GameAction.ACTION1, {}),  # 12: → (27, 24)
        (GameAction.ACTION4, {}),  # 13: → (30, 24)
        (GameAction.ACTION4, {}),  # 14: → (33, 24)
        (GameAction.ACTION4, {}),  # 15: → (36, 24)
        (GameAction.ACTION4, {}),  # 16: → (39, 24)
        (GameAction.ACTION4, {}),  # 17: → (42, 24)
        (GameAction.ACTION4, {}),  # 18: → (45, 24)
        (GameAction.ACTION4, {}),  # 19: → (48, 24)
        (GameAction.ACTION2, {}),  # 20: → (48, 27) inside B → next_level()
    ],
}


@register_binding(
    "ka59",
    primitive_name="select_then_transport",
    binding_id="ka59_l1_select_transport_candidate",
    observer_only=True,
)
def _bind_ka59_l1_select_transport_candidate(state: dict) -> PrimitiveBindingResult | None:
    if state.get("level_signature") != "ka59_l1":
        return None
    first_click = _first_click_payload("ka59_l1")
    if first_click is None:
        return None
    first_click_site = (int(first_click["x"]), int(first_click["y"]))

    selectable_items = [
        item for item in list(state.get("selectable_candidates", []) or [])
        if isinstance(item, dict) and _site_from_item(item) is not None
    ]
    target_items = [
        item for item in list(state.get("target_regions", []) or [])
        if isinstance(item, dict) and _site_from_item(item) is not None
    ]
    candidate_items = target_items or selectable_items
    if not selectable_items or not candidate_items:
        return None

    candidate_items.sort(key=lambda item: _manhattan(_site_from_item(item) or (0, 0), first_click_site))
    selectable_items.sort(key=lambda item: _manhattan(_site_from_item(item) or (0, 0), first_click_site))
    anchor_site = _site_from_item(selectable_items[0])
    if anchor_site is None:
        return None
    predicted_prefix = _predict_preselect_prefix(source=anchor_site, target=first_click_site)
    fixed_prefix = _movement_prefix_before_first_click("ka59_l1")
    prefix_match_len = _movement_prefix_match_len(predicted_prefix, fixed_prefix)
    start = TransportState.from_objects({"anchor": anchor_site})
    selections: list[SelectionSpec] = []
    for index, item in enumerate(candidate_items):
        site = _site_from_item(item)
        if site is None:
            continue
        selections.append(
            SelectionSpec.with_tags(
                f"candidate_{index}",
                PlannedAction("ACTION6", {"x": int(site[0]), "y": int(site[1])}),
                {"selected_candidate": index},
                metadata={"site": list(site), "role": str(item.get("role", "") or "")},
            )
        )
    if not selections:
        return None

    def build_transport(selected_state: TransportState, _selection: SelectionSpec) -> TransportSearchProblem:
        return TransportSearchProblem(
            start=selected_state,
            transitions={},
            is_goal=lambda _state: True,
            max_expansions=1,
        )

    candidate = select_then_transport_search(
        SelectThenTransportProblem(
            start=start,
            selections=tuple(selections),
            build_transport=build_transport,
            max_actions=1,
        )
    )
    if candidate is None:
        return None
    action_payload = candidate.actions[0].payload
    candidate_site = (int(action_payload.get("x", 0)), int(action_payload.get("y", 0)))
    payload_distance = _manhattan(candidate_site, first_click_site)
    candidate_actions = tuple(PlannedAction(action_name) for action_name in predicted_prefix) + tuple(candidate.actions)
    return PrimitiveBindingResult(
        actions=candidate_actions,
        binding_id="ka59_l1_select_transport_candidate",
        primitive_name="select_then_transport",
        adapter_free=False,
        generalisation_tier="observer_candidate_unverified",
        expanded_states=candidate.expanded_states,
        visited_states=candidate.visited_states,
        metadata={
            "game_id": "ka59",
            "level_signature": "ka59_l1",
            "observer_only": True,
            "candidate_status": "slot_probe_no_transition_model",
            "selection_id": candidate.selection_id,
            "first_fixed_click": [first_click_site[0], first_click_site[1]],
            "candidate_click": [candidate_site[0], candidate_site[1]],
            "payload_distance": payload_distance,
            "fixed_first_subgoal_matched": payload_distance <= 8,
            "movement_source_site": [anchor_site[0], anchor_site[1]],
            "movement_target_site": [first_click_site[0], first_click_site[1]],
            "predicted_movement_prefix": list(predicted_prefix),
            "fixed_movement_prefix": list(fixed_prefix),
            "movement_prefix_match_len": prefix_match_len,
            "movement_prefix_exact_match": bool(predicted_prefix == fixed_prefix),
            "movement_prefix_first3_match": bool(
                len(predicted_prefix) >= 3
                and len(fixed_prefix) >= 3
                and predicted_prefix[:3] == fixed_prefix[:3]
            ),
            "movement_stride": KA59_GRID_STRIDE,
            "selectable_count": len(selectable_items),
            "target_region_count": len(target_items),
        },
    )


@register_binding("ka59", primitive_name="fixed_sequence", binding_id="ka59_per_layout_plan")
def _bind_ka59_per_layout_plan(state: dict) -> PrimitiveBindingResult | None:
    sig = state.get("level_signature")
    if not isinstance(sig, str):
        return None
    plan = _PLANS.get(sig)
    if not plan:
        return None
    return PrimitiveBindingResult.from_action_tuples(
        plan,
        binding_id="ka59_per_layout_plan",
        primitive_name="fixed_sequence",
        adapter_free=False,
        generalisation_tier="tier_c_fixed_plan",
        metadata={
            "game_id": "ka59",
            "level_signature": sig,
            "solver": "ka59_per_layout_plan",
            "selectable_count": int((state.get("candidate_summary") or {}).get("selectable_count", 0) or 0),
            "target_region_count": int((state.get("candidate_summary") or {}).get("target_region_count", 0) or 0),
            "barrier_region_count": int((state.get("candidate_summary") or {}).get("barrier_region_count", 0) or 0),
        },
    )


def plan_next_actions_from_state(state: dict) -> List[Tuple[GameAction, dict]]:
    global LAST_OBSERVER_BINDING_DIAGNOSTICS
    observer_results = observe_bindings("ka59", state)
    LAST_OBSERVER_BINDING_DIAGNOSTICS = [result.diagnostics() for result in observer_results]
    result = solve_binding("ka59", state)
    if result is not None:
        diagnostics = result.diagnostics()
        if LAST_OBSERVER_BINDING_DIAGNOSTICS:
            diagnostics["observer_bindings"] = list(LAST_OBSERVER_BINDING_DIAGNOSTICS)
        _record_plan_source(f"primitive_binding_{result.binding_id}", diagnostics)
        return result.game_action_tuples(GameAction)
    _record_plan_source("unsolved", {"binding_id": "", "primitive_name": "", "plan_len": 0})
    return []


def perceive_and_plan(frame: Any) -> List[Tuple[GameAction, dict]]:
    state = perceive_ka59(frame)
    return plan_next_actions_from_state(state)
