"""Cluster-A puzzle adapters (Phase 9.C.1+).

These bypass ECA's reactive control with explicit per-puzzle solvers.
Each game-specific adapter exports a ``plan_next_actions(game)`` callable
that returns a list of GameAction values to execute next.
"""
