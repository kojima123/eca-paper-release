"""Minimal environment interface used by the ECA runtime."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEnv(ABC):
    """An environment can be reset, stepped with an action, and closed."""

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Start an episode and return the first observation."""

    @abstractmethod
    def step(self, action: Any) -> Any:
        """Apply one action and return what the environment reports."""

    def close(self) -> None:
        """Release resources; nothing to do by default."""
        return None
