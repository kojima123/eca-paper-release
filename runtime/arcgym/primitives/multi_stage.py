"""Generic sequential planner for staged puzzle mechanics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .toggle_state import PlannedAction


State = Mapping[str, Any]
StageSolveFn = Callable[[State], "StageResult | None"]
FinalPredicate = Callable[[State], bool]


@dataclass(frozen=True)
class StageResult:
    actions: tuple[PlannedAction, ...]
    next_state: State
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_actions(
        cls,
        actions: list[str | PlannedAction],
        next_state: State,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StageResult":
        planned: list[PlannedAction] = []
        for action in actions:
            if isinstance(action, PlannedAction):
                planned.append(action)
            else:
                planned.append(PlannedAction(str(action)))
        return cls(tuple(planned), dict(next_state), dict(metadata or {}))


@dataclass(frozen=True)
class StageSpec:
    name: str
    solve: StageSolveFn
    is_complete: FinalPredicate | None = None


@dataclass(frozen=True)
class MultiStagePlanResult:
    actions: tuple[PlannedAction, ...]
    final_state: State
    completed_stages: tuple[str, ...]
    stage_metadata: tuple[tuple[str, dict[str, Any]], ...]

    def action_tuples(self) -> list[tuple[str, dict[str, int]]]:
        return [action.to_tuple() for action in self.actions]


def multi_stage_planner(
    initial_state: State,
    stages: list[StageSpec] | tuple[StageSpec, ...],
    *,
    final_win: FinalPredicate | None = None,
) -> MultiStagePlanResult | None:
    """Run stage solvers in order and concatenate their action plans."""

    state: State = dict(initial_state)
    actions: list[PlannedAction] = []
    completed: list[str] = []
    metadata: list[tuple[str, dict[str, Any]]] = []

    if final_win is not None and final_win(state):
        return MultiStagePlanResult((), dict(state), (), ())

    for stage in stages:
        if stage.is_complete is not None and stage.is_complete(state):
            completed.append(str(stage.name))
            metadata.append((str(stage.name), {"skipped": True}))
            continue
        result = stage.solve(state)
        if result is None:
            return None
        actions.extend(result.actions)
        state = dict(result.next_state)
        completed.append(str(stage.name))
        metadata.append((str(stage.name), dict(result.metadata)))

    if final_win is not None and not final_win(state):
        return None
    return MultiStagePlanResult(tuple(actions), dict(state), tuple(completed), tuple(metadata))


__all__ = [
    "MultiStagePlanResult",
    "StageResult",
    "StageSpec",
    "multi_stage_planner",
]
