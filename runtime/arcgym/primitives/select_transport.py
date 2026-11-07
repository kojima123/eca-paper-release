"""Generic select-then-transport primitive.

This kernel models puzzles where the controller must first bind a selected
object/cursor/piece, then run a transport search under that selection.  It is
game-id free: adapters provide selection candidates and a transport problem
factory for the selected state.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .toggle_state import PlannedAction
from .transport import TransportSearchProblem, TransportSearchResult, TransportState, transport_bfs


SelectionBindFn = Callable[[TransportState], TransportState | None]
TransportProblemFactory = Callable[[TransportState, "SelectionSpec"], TransportSearchProblem | None]


@dataclass(frozen=True)
class SelectionSpec:
    selection_id: str
    action: PlannedAction
    bind: SelectionBindFn
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def with_tags(
        cls,
        selection_id: str,
        action: PlannedAction,
        tags: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> "SelectionSpec":
        return cls(
            selection_id=str(selection_id),
            action=action,
            bind=lambda state: state.with_tags(dict(tags)),
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class SelectThenTransportProblem:
    start: TransportState
    selections: tuple[SelectionSpec, ...]
    build_transport: TransportProblemFactory
    max_actions: int | None = None


@dataclass(frozen=True)
class SelectThenTransportResult:
    actions: tuple[PlannedAction, ...]
    final_state: TransportState
    selection_id: str
    transport_result: TransportSearchResult
    expanded_states: int
    visited_states: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def action_tuples(self) -> list[tuple[str, dict[str, int]]]:
        return [action.to_tuple() for action in self.actions]


def select_then_transport_search(
    problem: SelectThenTransportProblem,
) -> SelectThenTransportResult | None:
    """Try selection candidates and return the shortest selected transport plan."""

    best: SelectThenTransportResult | None = None
    for selection in problem.selections:
        selected_state = selection.bind(problem.start)
        if selected_state is None:
            continue
        transport_problem = problem.build_transport(selected_state, selection)
        if transport_problem is None:
            continue
        transport_result = transport_bfs(transport_problem)
        if transport_result is None:
            continue
        actions = (selection.action,) + tuple(transport_result.actions)
        if problem.max_actions is not None and len(actions) > int(problem.max_actions):
            continue
        candidate = SelectThenTransportResult(
            actions=actions,
            final_state=transport_result.final_state,
            selection_id=str(selection.selection_id),
            transport_result=transport_result,
            expanded_states=transport_result.expanded_states,
            visited_states=transport_result.visited_states,
            metadata=dict(selection.metadata),
        )
        if best is None or len(candidate.actions) < len(best.actions):
            best = candidate
    return best


__all__ = [
    "SelectionSpec",
    "SelectThenTransportProblem",
    "SelectThenTransportResult",
    "select_then_transport_search",
]
