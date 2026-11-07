"""Sim-search lane: runtime game-source discovery + local-simulator BFS.

The June-30 milestone lane (Plan A, docs/handoff_milestone_june30_sim_search_2026-06-11.md):

1. discover the game's .py source inside the run environment,
2. instantiate it locally and search over pickle-snapshot states
   (trigger-aware state hash, BFS first, cascade later),
3. play back only the shortest found solution against the real env,
4. fall back to the existing strict-generic stack when no source is found.

All ideas re-implemented from scratch (public-notebook技術の自前再実装;
コード逐語コピーなし).
"""
from arcgym.sim_search.source_discovery import find_game_source
from arcgym.sim_search.local_game import LocalSim, clone_game, load_game_class
from arcgym.sim_search.state_hash import probe_trigger_fields, state_key
from arcgym.sim_search.solver import (
    SimSearchResult,
    counter_astar,
    objective_astar,
    solve_level,
    solve_level_cascade,
    solve_game,
)
from arcgym.sim_search.win_field import extract_win_field

__all__ = [
    "find_game_source",
    "LocalSim",
    "clone_game",
    "load_game_class",
    "probe_trigger_fields",
    "state_key",
    "SimSearchResult",
    "counter_astar",
    "extract_win_field",
    "objective_astar",
    "solve_level",
    "solve_level_cascade",
    "solve_game",
]
