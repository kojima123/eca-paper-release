"""Generic BFS for constrained transport mechanics.

The primitive models games where one or more named objects move through a
finite state graph.  It is intentionally game-id free: callers bind perceived
objects, transition callbacks, and the local win predicate.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field

from .toggle_state import PlannedAction


Coord = tuple[int, int]
TransitionFn = Callable[["TransportState"], "TransportState | None"]
BlockedFn = Callable[[Coord, "TransportState"], bool]


DEFAULT_WALK_DELTAS: dict[str, Coord] = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}


@dataclass(frozen=True)
class TransportState:
    objects: tuple[tuple[str, Coord], ...]
    tags: tuple[tuple[str, Hashable], ...] = ()

    @classmethod
    def from_objects(
        cls,
        objects: Mapping[str, Coord],
        tags: Mapping[str, Hashable] | None = None,
    ) -> "TransportState":
        return cls(
            tuple(sorted((str(key), (int(value[0]), int(value[1]))) for key, value in objects.items())),
            tuple(sorted((str(key), value) for key, value in dict(tags or {}).items())),
        )

    def object_map(self) -> dict[str, Coord]:
        return dict(self.objects)

    def tag_map(self) -> dict[str, Hashable]:
        return dict(self.tags)

    def with_objects(self, updates: Mapping[str, Coord]) -> "TransportState":
        objects = self.object_map()
        objects.update({str(key): (int(value[0]), int(value[1])) for key, value in updates.items()})
        return TransportState.from_objects(objects, self.tag_map())

    def with_tags(self, updates: Mapping[str, Hashable]) -> "TransportState":
        tags = self.tag_map()
        tags.update({str(key): value for key, value in updates.items()})
        return TransportState.from_objects(self.object_map(), tags)


@dataclass(frozen=True)
class TransportSearchResult:
    actions: tuple[PlannedAction, ...]
    final_state: TransportState
    expanded_states: int
    visited_states: int

    def action_tuples(self) -> list[tuple[str, dict[str, int]]]:
        return [action.to_tuple() for action in self.actions]


@dataclass(frozen=True)
class TransportSearchProblem:
    start: TransportState
    transitions: Mapping[str, TransitionFn]
    is_goal: Callable[[TransportState], bool]
    max_expansions: int = 100_000


def transport_bfs(problem: TransportSearchProblem) -> TransportSearchResult | None:
    """Solve a finite transport problem with BFS."""

    start = problem.start
    if problem.is_goal(start):
        return TransportSearchResult((), start, expanded_states=0, visited_states=1)

    queue: deque[tuple[TransportState, tuple[PlannedAction, ...]]] = deque([(start, ())])
    visited: set[TransportState] = {start}
    expanded = 0

    transition_items = [(str(action), fn) for action, fn in problem.transitions.items()]
    while queue and expanded < int(problem.max_expansions):
        state, path = queue.popleft()
        expanded += 1
        for action_name, transition in transition_items:
            next_state = transition(state)
            if next_state is None or next_state == state or next_state in visited:
                continue
            next_path = path + (PlannedAction(action_name),)
            visited.add(next_state)
            if problem.is_goal(next_state):
                return TransportSearchResult(
                    next_path,
                    next_state,
                    expanded_states=expanded,
                    visited_states=len(visited),
                )
            queue.append((next_state, next_path))

    return None


def grid_push_transition(
    *,
    actor_id: str,
    movable_ids: set[str],
    delta: Coord,
    bounds: tuple[int, int] | None = None,
    blocked: BlockedFn | None = None,
    object_blocks: bool = True,
) -> TransitionFn:
    """Build a grid move/push transition for Sokoban-like mechanics."""

    dx, dy = int(delta[0]), int(delta[1])
    actor_key = str(actor_id)
    movable_keys = {str(item) for item in movable_ids}

    def transition(state: TransportState) -> TransportState | None:
        objects = state.object_map()
        if actor_key not in objects:
            return None
        actor = objects[actor_key]
        target = (actor[0] + dx, actor[1] + dy)
        if not _coord_allowed(target, state, bounds, blocked):
            return None

        target_object = _object_at(objects, target, exclude={actor_key})
        if target_object in movable_keys:
            pushed = (target[0] + dx, target[1] + dy)
            if not _coord_allowed(pushed, state, bounds, blocked):
                return None
            blocking_object = _object_at(objects, pushed, exclude={actor_key, target_object})
            if object_blocks and blocking_object is not None:
                return None
            return state.with_objects({actor_key: target, target_object: pushed})

        if object_blocks and target_object is not None:
            return None
        return state.with_objects({actor_key: target})

    return transition


def four_way_grid_push_transitions(
    *,
    actor_id: str,
    movable_ids: set[str],
    bounds: tuple[int, int] | None = None,
    blocked: BlockedFn | None = None,
    object_blocks: bool = True,
) -> dict[str, TransitionFn]:
    return {
        action: grid_push_transition(
            actor_id=actor_id,
            movable_ids=movable_ids,
            delta=delta,
            bounds=bounds,
            blocked=blocked,
            object_blocks=object_blocks,
        )
        for action, delta in DEFAULT_WALK_DELTAS.items()
    }


def _coord_allowed(
    coord: Coord,
    state: TransportState,
    bounds: tuple[int, int] | None,
    blocked: BlockedFn | None,
) -> bool:
    if bounds is not None:
        width, height = int(bounds[0]), int(bounds[1])
        if not (0 <= coord[0] < width and 0 <= coord[1] < height):
            return False
    if blocked is not None and blocked(coord, state):
        return False
    return True


def _object_at(
    objects: Mapping[str, Coord],
    coord: Coord,
    *,
    exclude: set[str],
) -> str | None:
    for object_id, object_coord in objects.items():
        if object_id in exclude:
            continue
        if object_coord == coord:
            return str(object_id)
    return None


__all__ = [
    "DEFAULT_WALK_DELTAS",
    "TransportSearchProblem",
    "TransportSearchResult",
    "TransportState",
    "four_way_grid_push_transitions",
    "grid_push_transition",
    "transport_bfs",
]
