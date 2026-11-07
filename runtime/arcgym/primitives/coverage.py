"""Generic search for coverage / painting / placement mechanics."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .toggle_state import PlannedAction


Coord = tuple[int, int]
CellMap = Mapping[Coord, Hashable]
TransitionFn = Callable[["CoverageState"], "CoverageState | None"]
PredicateFn = Callable[["CoverageState"], bool]


@dataclass(frozen=True)
class CoverageState:
    cells: tuple[tuple[Coord, Hashable], ...] = ()
    tags: tuple[tuple[str, Hashable], ...] = ()

    @classmethod
    def from_cells(
        cls,
        cells: CellMap | None = None,
        tags: Mapping[str, Hashable] | None = None,
    ) -> "CoverageState":
        return cls(
            tuple(sorted(((int(coord[0]), int(coord[1])), value) for coord, value in dict(cells or {}).items())),
            tuple(sorted((str(key), value) for key, value in dict(tags or {}).items())),
        )

    def cell_map(self) -> dict[Coord, Hashable]:
        return dict(self.cells)

    def tag_map(self) -> dict[str, Hashable]:
        return dict(self.tags)

    def with_cells(self, updates: CellMap) -> "CoverageState":
        cells = self.cell_map()
        cells.update({(int(coord[0]), int(coord[1])): value for coord, value in dict(updates).items()})
        return CoverageState.from_cells(cells, self.tag_map())

    def with_tags(self, updates: Mapping[str, Hashable]) -> "CoverageState":
        tags = self.tag_map()
        tags.update({str(key): value for key, value in dict(updates).items()})
        return CoverageState.from_cells(self.cell_map(), tags)


@dataclass(frozen=True)
class CoverageOperation:
    action: str
    writes: tuple[tuple[Coord, Hashable], ...] = ()
    payload: dict[str, int] = field(default_factory=dict)
    transition: TransitionFn | None = None
    precondition: PredicateFn | None = None

    @classmethod
    def paint(
        cls,
        action: str,
        writes: CellMap,
        *,
        payload: Mapping[str, int] | None = None,
        precondition: PredicateFn | None = None,
    ) -> "CoverageOperation":
        return cls(
            action=str(action),
            writes=tuple(
                sorted(((int(coord[0]), int(coord[1])), value) for coord, value in dict(writes).items())
            ),
            payload={str(key): int(value) for key, value in dict(payload or {}).items()},
            precondition=precondition,
        )

    def apply(self, state: CoverageState) -> CoverageState | None:
        if self.precondition is not None and not self.precondition(state):
            return None
        if self.transition is not None:
            return self.transition(state)
        if not self.writes:
            return None
        return state.with_cells(dict(self.writes))

    def planned_action(self) -> PlannedAction:
        return PlannedAction(self.action, dict(self.payload))


@dataclass(frozen=True)
class CoverageSearchResult:
    actions: tuple[PlannedAction, ...]
    final_state: CoverageState
    expanded_states: int
    visited_states: int

    def action_tuples(self) -> list[tuple[str, dict[str, int]]]:
        return [action.to_tuple() for action in self.actions]


@dataclass(frozen=True)
class CoverageSearchProblem:
    start: CoverageState
    operations: tuple[CoverageOperation, ...]
    is_goal: PredicateFn
    max_expansions: int = 100_000


def coverage_search(problem: CoverageSearchProblem) -> CoverageSearchResult | None:
    """Search for an operation sequence that reaches the coverage goal."""

    start = problem.start
    if problem.is_goal(start):
        return CoverageSearchResult((), start, expanded_states=0, visited_states=1)

    queue: deque[tuple[CoverageState, tuple[PlannedAction, ...]]] = deque([(start, ())])
    visited: set[CoverageState] = {start}
    expanded = 0

    while queue and expanded < int(problem.max_expansions):
        state, path = queue.popleft()
        expanded += 1
        for operation in problem.operations:
            next_state = operation.apply(state)
            if next_state is None or next_state == state or next_state in visited:
                continue
            next_path = path + (operation.planned_action(),)
            visited.add(next_state)
            if problem.is_goal(next_state):
                return CoverageSearchResult(
                    next_path,
                    next_state,
                    expanded_states=expanded,
                    visited_states=len(visited),
                )
            queue.append((next_state, next_path))

    return None


def target_cells_goal(targets: CellMap) -> PredicateFn:
    expected = {(int(coord[0]), int(coord[1])): value for coord, value in dict(targets).items()}

    def predicate(state: CoverageState) -> bool:
        cells = state.cell_map()
        return all(cells.get(coord) == value for coord, value in expected.items())

    return predicate


__all__ = [
    "CoverageOperation",
    "CoverageSearchProblem",
    "CoverageSearchResult",
    "CoverageState",
    "coverage_search",
    "target_cells_goal",
]
