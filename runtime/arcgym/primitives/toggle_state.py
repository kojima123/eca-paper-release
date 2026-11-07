"""Generic BFS for click/toggle reachability mechanics.

The primitive models games where movement is combined with discrete state
variables changed by click-like actions.  It does not know about game ids,
levels, sprite names, or hardcoded action sequences.  A game-specific adapter is
responsible only for binding frame-extracted state into this problem shape.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any


Coord = tuple[int, int]
ToggleMap = Mapping[str, Hashable]
ClickEffect = Mapping[str, Hashable | Callable[[Hashable | None], Hashable]]
ClickEffectFn = Callable[["ToggleSearchState"], "ToggleSearchState | ToggleMap | None"]


DEFAULT_WALK_ACTIONS: dict[str, Coord] = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}


@dataclass(frozen=True)
class PlannedAction:
    action: str
    payload: dict[str, int] = field(default_factory=dict)

    def to_tuple(self) -> tuple[str, dict[str, int]]:
        return self.action, dict(self.payload)


@dataclass(frozen=True)
class ToggleSearchState:
    player: Coord
    toggles: tuple[tuple[str, Hashable], ...] = ()

    @classmethod
    def from_map(cls, player: Coord, toggles: ToggleMap | None = None) -> "ToggleSearchState":
        return cls(
            (int(player[0]), int(player[1])),
            tuple(sorted((str(key), value) for key, value in dict(toggles or {}).items())),
        )

    def toggle_map(self) -> dict[str, Hashable]:
        return dict(self.toggles)

    def with_toggles(self, updates: ToggleMap) -> "ToggleSearchState":
        values = self.toggle_map()
        values.update({str(key): value for key, value in dict(updates).items()})
        return ToggleSearchState.from_map(self.player, values)

    def with_player(self, player: Coord) -> "ToggleSearchState":
        return ToggleSearchState.from_map(player, self.toggle_map())


@dataclass(frozen=True)
class ToggleSearchResult:
    actions: tuple[PlannedAction, ...]
    final_state: ToggleSearchState
    expanded_states: int
    visited_states: int

    def action_tuples(self) -> list[tuple[str, dict[str, int]]]:
        return [action.to_tuple() for action in self.actions]


@dataclass(frozen=True)
class ToggleSearchProblem:
    start: ToggleSearchState
    click_effects: Mapping[Coord, ClickEffect | ClickEffectFn]
    is_goal: Callable[[ToggleSearchState], bool]
    is_passable: Callable[[Coord, ToggleSearchState], bool]
    can_click: Callable[[ToggleSearchState, Coord], bool] = lambda _state, _site: True
    walk_actions: Mapping[str, Coord] = field(default_factory=lambda: dict(DEFAULT_WALK_ACTIONS))
    click_action: str = "ACTION6"
    bounds: tuple[int, int] | None = None
    max_expansions: int = 100_000


def toggle_state_bfs(problem: ToggleSearchProblem) -> ToggleSearchResult | None:
    """Solve a toggle-state reachability problem with BFS.

    The search state is `(player_pos, toggle_vector)`.  Movement transitions are
    supplied by `walk_actions`; click transitions are supplied by `click_effects`.
    Click effects may be simple assignments or callback transitions, allowing a
    caller to model teleports or cascading toggles without changing this kernel.
    """

    start = problem.start
    if problem.is_goal(start):
        return ToggleSearchResult((), start, expanded_states=0, visited_states=1)

    queue: deque[tuple[ToggleSearchState, tuple[PlannedAction, ...]]] = deque([(start, ())])
    visited: set[ToggleSearchState] = {start}
    expanded = 0

    while queue and expanded < int(problem.max_expansions):
        state, path = queue.popleft()
        expanded += 1

        for action_name, delta in problem.walk_actions.items():
            next_player = (int(state.player[0] + delta[0]), int(state.player[1] + delta[1]))
            if not _in_bounds(next_player, problem.bounds):
                continue
            next_state = state.with_player(next_player)
            if not problem.is_passable(next_player, next_state):
                continue
            result = _maybe_enqueue(
                next_state,
                path + (PlannedAction(str(action_name)),),
                problem,
                queue,
                visited,
                expanded,
            )
            if result is not None:
                return result

        for site, effect in sorted(problem.click_effects.items()):
            click_site = (int(site[0]), int(site[1]))
            if not problem.can_click(state, click_site):
                continue
            next_state = _apply_click_effect(state, effect)
            if next_state is None or next_state == state:
                continue
            result = _maybe_enqueue(
                next_state,
                path + (PlannedAction(str(problem.click_action), {"x": click_site[0], "y": click_site[1]}),),
                problem,
                queue,
                visited,
                expanded,
            )
            if result is not None:
                return result

    return None


def _maybe_enqueue(
    state: ToggleSearchState,
    path: tuple[PlannedAction, ...],
    problem: ToggleSearchProblem,
    queue: deque[tuple[ToggleSearchState, tuple[PlannedAction, ...]]],
    visited: set[ToggleSearchState],
    expanded_states: int,
) -> ToggleSearchResult | None:
    if state in visited:
        return None
    visited.add(state)
    if problem.is_goal(state):
        return ToggleSearchResult(
            path,
            state,
            expanded_states=expanded_states,
            visited_states=len(visited),
        )
    queue.append((state, path))
    return None


def _apply_click_effect(
    state: ToggleSearchState,
    effect: ClickEffect | ClickEffectFn,
) -> ToggleSearchState | None:
    if callable(effect):
        result = effect(state)
        if result is None:
            return None
        if isinstance(result, ToggleSearchState):
            return result
        return state.with_toggles(result)

    current = state.toggle_map()
    updates: dict[str, Hashable] = {}
    for key, value_or_fn in dict(effect).items():
        old_value = current.get(str(key))
        if callable(value_or_fn):
            updates[str(key)] = value_or_fn(old_value)
        else:
            updates[str(key)] = value_or_fn
    return state.with_toggles(updates)


def _in_bounds(coord: Coord, bounds: tuple[int, int] | None) -> bool:
    if bounds is None:
        return True
    width, height = int(bounds[0]), int(bounds[1])
    return 0 <= int(coord[0]) < width and 0 <= int(coord[1]) < height


__all__ = [
    "DEFAULT_WALK_ACTIONS",
    "PlannedAction",
    "ToggleSearchProblem",
    "ToggleSearchResult",
    "ToggleSearchState",
    "toggle_state_bfs",
]
