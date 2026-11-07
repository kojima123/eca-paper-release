"""Reusable solver primitive kernels.

These modules are deliberately game-agnostic.  Game adapters should bind
frame-extracted state and transition callbacks to these kernels rather than
encoding fixed action sequences.
"""

from .toggle_state import (
    PlannedAction,
    ToggleSearchProblem,
    ToggleSearchResult,
    ToggleSearchState,
    toggle_state_bfs,
)
from .coverage import (
    CoverageOperation,
    CoverageSearchProblem,
    CoverageSearchResult,
    CoverageState,
    coverage_search,
    target_cells_goal,
)
from .multi_stage import (
    MultiStagePlanResult,
    StageResult,
    StageSpec,
    multi_stage_planner,
)
from .select_transport import (
    SelectionSpec,
    SelectThenTransportProblem,
    SelectThenTransportResult,
    select_then_transport_search,
)
from .transport import (
    TransportSearchProblem,
    TransportSearchResult,
    TransportState,
    four_way_grid_push_transitions,
    grid_push_transition,
    transport_bfs,
)
from .binding import (
    GLOBAL_PRIMITIVE_BINDINGS,
    PrimitiveBinding,
    PrimitiveBindingRegistry,
    PrimitiveBindingResult,
    observe_bindings,
    register_binding,
    solve_binding,
)

__all__ = [
    "CoverageOperation",
    "CoverageSearchProblem",
    "CoverageSearchResult",
    "CoverageState",
    "MultiStagePlanResult",
    "PlannedAction",
    "GLOBAL_PRIMITIVE_BINDINGS",
    "PrimitiveBinding",
    "PrimitiveBindingRegistry",
    "PrimitiveBindingResult",
    "observe_bindings",
    "SelectionSpec",
    "SelectThenTransportProblem",
    "SelectThenTransportResult",
    "StageResult",
    "StageSpec",
    "ToggleSearchProblem",
    "ToggleSearchResult",
    "ToggleSearchState",
    "TransportSearchProblem",
    "TransportSearchResult",
    "TransportState",
    "coverage_search",
    "four_way_grid_push_transitions",
    "grid_push_transition",
    "multi_stage_planner",
    "register_binding",
    "select_then_transport_search",
    "solve_binding",
    "target_cells_goal",
    "toggle_state_bfs",
    "transport_bfs",
]
