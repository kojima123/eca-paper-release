"""Trigger-aware state hashing for the sim-search BFS.

Visually identical frames can differ in hidden game state (toggles,
counters). We probe the game object's scalar attributes under the
effective actions and mix the ones that change ("trigger fields") into
the visited-set key — EXCEPT fields that advance again when the same
action is repeated (those are clocks/timers; including them would make
every node unique and blow up the visited set).
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from arcgym.sim_search.local_game import LocalSim, clone_game

# Bookkeeping fields that always change but carry no game semantics.
_EXCLUDED_FIELDS = {"_action_count", "_full_reset", "_action_complete"}
# Underscore-prefixed fields are internal except these two, which encode
# real progress.
_ALLOWED_PRIVATE = {"_current_level_index", "_score"}


def _scalar_fields(game: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(game).items():
        if key in _EXCLUDED_FIELDS:
            continue
        if key.startswith("_") and key not in _ALLOWED_PRIVATE:
            continue
        if isinstance(value, (bool, int, float)):
            out[key] = value
    return out


def probe_trigger_fields(
    game: Any,
    action_specs: Iterable[tuple],
    max_probe: int = 10,
) -> tuple[str, ...]:
    """Hidden scalar fields that change under actions, minus clock fields.

    ``game`` is not mutated (all probes run on clones).
    """
    specs = list(action_specs)[:max_probe]
    if not specs:
        return ()
    base = _scalar_fields(game)
    changed: list[str] = []
    first_effective: tuple | None = None
    for spec in specs:
        probe = clone_game(game)
        try:
            LocalSim.step(probe, spec)
        except Exception:
            continue
        after = _scalar_fields(probe)
        for key, value in after.items():
            if key in base and base[key] != value and key not in changed:
                changed.append(key)
                if first_effective is None:
                    first_effective = spec
        if first_effective is None and after != base:
            first_effective = spec
    if not changed or first_effective is None:
        return tuple(changed)

    # Clock exclusion: apply the SAME action twice; a field that advances
    # again on the repeat is a timer, not a trigger.
    try:
        once = clone_game(game)
        LocalSim.step(once, first_effective)
        twice = clone_game(once)
        LocalSim.step(twice, first_effective)
        clocks = {
            key
            for key in changed
            if getattr(once, key, None) != getattr(twice, key, None)
        }
    except Exception:
        clocks = set()
    return tuple(key for key in changed if key not in clocks)


def state_key(
    game: Any,
    frame: np.ndarray,
    trigger_fields: tuple[str, ...],
) -> Any:
    """Visited-set key: fast frame hash + trigger field values."""
    base = hash(frame.tobytes())
    if not trigger_fields:
        return base
    return (base, tuple(getattr(game, field, None) for field in trigger_fields))


__all__ = ["probe_trigger_fields", "state_key"]
