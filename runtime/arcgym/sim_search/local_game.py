"""Local instantiation of a discovered ARC-AGI-3 game source.

Loads the game module from a file path, instantiates the ``ARCBaseGame``
subclass, and exposes a minimal step/snapshot interface for search:

- ``perform_action(ActionInput(...), raw=True)`` drives the game locally
  (no API round-trip),
- snapshots are pickle round-trips (measured 2-3x faster than
  ``copy.deepcopy`` on these game objects; we wrap it in ``clone_game``
  instead of monkey-patching ``copy.deepcopy`` globally).
"""
from __future__ import annotations

import copy
import importlib.util
import os
import pickle
import sys
from typing import Any

import numpy as np

try:
    from arcengine import ActionInput, GameAction
except ImportError:  # pragma: no cover - toolkit absent on dev branch
    ActionInput = Any  # type: ignore[assignment,misc]
    GameAction = Any  # type: ignore[assignment,misc]

_MOVE_ACTIONS: dict[str, Any] = {}


def _game_action(name: str) -> Any:
    """Fresh GameAction reference by name (enum members are shared/mutable)."""
    cached = _MOVE_ACTIONS.get(name)
    if cached is not None:
        return cached
    action = GameAction.from_name(name)
    _MOVE_ACTIONS[name] = action
    return action


def load_game_class(source_path: str, class_name: str) -> type:
    module_name = f"_sim_search_game_{class_name.lower()}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load game module from {source_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the game objects stay picklable (snapshotting
    # pickles instances, which requires the defining module to be importable).
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


_PICKLE_CLONE_OK: bool | None = None


def clone_game(game: Any) -> Any:
    """Snapshot of the game object.

    Pickle round-trip when the game supports it (faster), otherwise
    ``copy.deepcopy`` (always works — the proven p8 snapshot path).
    The choice is probed once per process.
    """
    global _PICKLE_CLONE_OK
    if _PICKLE_CLONE_OK is None:
        try:
            pickle.loads(pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL))
            _PICKLE_CLONE_OK = True
        except Exception:
            _PICKLE_CLONE_OK = False
    if _PICKLE_CLONE_OK:
        try:
            return pickle.loads(pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            _PICKLE_CLONE_OK = False
    return copy.deepcopy(game)


def frame_of(result: Any) -> np.ndarray:
    """Final 2D grid of a perform_action result (frame is a list of layers)."""
    frame = getattr(result, "frame", None)
    if frame is None:
        raise RuntimeError("perform_action result has no frame")
    if isinstance(frame, list) and frame:
        frame = frame[-1]
    return np.asarray(frame)


class LocalSim:
    """Thin wrapper around a locally-instantiated game for search."""

    def __init__(self, game_cls: type) -> None:
        self._game_cls = game_cls
        self.game: Any = None
        self.base_levels: int = 0

    def reset_level(self, level_idx: int) -> np.ndarray:
        """Fresh game at ``level_idx``; returns the initial frame.

        ARCBaseGame.handle_reset full-resets to level 0 whenever
        ``_action_count == 0``, so a plain ``set_level(k); RESET`` lands
        back on level 0. The level-scoped reset path is gated by the
        engine's ``ONLY_RESET_LEVELS`` env var; we enable it just for the
        second RESET so the game stays on ``level_idx`` with a clean,
        rendered initial frame.
        """
        game = self._game_cls()
        # Full reset: clean clones of every level, score 0, level 0.
        result = game.perform_action(ActionInput(id=_game_action("RESET")), raw=True)
        if level_idx > 0 and hasattr(game, "set_level"):
            game.set_level(level_idx)
            saved = os.environ.get("ONLY_RESET_LEVELS")
            os.environ["ONLY_RESET_LEVELS"] = "true"
            try:
                result = game.perform_action(
                    ActionInput(id=_game_action("RESET")), raw=True
                )
            finally:
                if saved is None:
                    os.environ.pop("ONLY_RESET_LEVELS", None)
                else:
                    os.environ["ONLY_RESET_LEVELS"] = saved
        self.game = game
        self.base_levels = int(getattr(result, "levels_completed", 0) or 0)
        return frame_of(result)

    @staticmethod
    def step(game: Any, spec: tuple) -> tuple[np.ndarray, int, int]:
        """Apply one action spec to ``game`` IN PLACE.

        ``spec`` is ``("ACTION1",)`` ... or ``("ACTION6", x, y)``.
        Returns ``(frame, levels_completed, current_level_index)``.
        """
        data: dict[str, int] = {}
        if len(spec) == 3:
            data = {"x": int(spec[1]), "y": int(spec[2])}
        result = game.perform_action(
            ActionInput(id=_game_action(spec[0]), data=data), raw=True
        )
        levels = int(getattr(result, "levels_completed", 0) or 0)
        level_idx = int(getattr(game, "_current_level_index", 0) or 0)
        return frame_of(result), levels, level_idx

    def available_action_names(self) -> list[str]:
        raw = getattr(self.game, "_available_actions", None) or []
        names: list[str] = []
        for entry in raw:
            name = str(getattr(entry, "name", entry) or "")
            if name.startswith("GameAction."):
                name = name.split(".", 1)[1]
            if name.isdigit():
                name = f"ACTION{int(name)}"
            names.append(name)
        return names


__all__ = ["LocalSim", "clone_game", "frame_of", "load_game_class"]
