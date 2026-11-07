"""Registry for binding perceived game state to primitive search kernels."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .toggle_state import PlannedAction


BindingFn = Callable[[Mapping[str, Any]], "PrimitiveBindingResult | None"]


@dataclass(frozen=True)
class PrimitiveBindingResult:
    """Result emitted by a primitive binding.

    The result deliberately carries provenance fields beside the planned
    actions.  Generalisation reports need to distinguish adapter-free primitive
    bindings from fallback or fixed-plan paths.
    """

    actions: tuple[PlannedAction, ...]
    binding_id: str
    primitive_name: str
    adapter_free: bool = False
    generalisation_tier: str = "frame_conditioned_game_primitive"
    expanded_states: int | None = None
    visited_states: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_action_tuples(
        cls,
        actions: list[tuple[Any, Mapping[str, Any]]] | tuple[tuple[Any, Mapping[str, Any]], ...],
        *,
        binding_id: str,
        primitive_name: str,
        adapter_free: bool = False,
        generalisation_tier: str = "frame_conditioned_game_primitive",
        expanded_states: int | None = None,
        visited_states: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PrimitiveBindingResult":
        planned: list[PlannedAction] = []
        for action, payload in actions:
            action_name = getattr(action, "name", str(action))
            planned.append(
                PlannedAction(
                    str(action_name),
                    {str(key): int(value) for key, value in dict(payload or {}).items()},
                )
            )
        return cls(
            actions=tuple(planned),
            binding_id=str(binding_id),
            primitive_name=str(primitive_name),
            adapter_free=bool(adapter_free),
            generalisation_tier=str(generalisation_tier),
            expanded_states=expanded_states,
            visited_states=visited_states,
            metadata=dict(metadata or {}),
        )

    def action_tuples(self) -> list[tuple[str, dict[str, int]]]:
        return [action.to_tuple() for action in self.actions]

    def game_action_tuples(self, action_enum: Any) -> list[tuple[Any, dict[str, int]]]:
        return [(action_enum[action.action], dict(action.payload)) for action in self.actions]

    def diagnostics(self) -> dict[str, Any]:
        out = {
            "binding_id": self.binding_id,
            "primitive_name": self.primitive_name,
            "adapter_free": self.adapter_free,
            "generalisation_tier": self.generalisation_tier,
            "plan_len": len(self.actions),
        }
        if self.expanded_states is not None:
            out["expanded_states"] = int(self.expanded_states)
        if self.visited_states is not None:
            out["visited_states"] = int(self.visited_states)
        out.update(dict(self.metadata))
        return out


@dataclass(frozen=True)
class PrimitiveBinding:
    game_id: str
    binding_id: str
    primitive_name: str
    solve: BindingFn
    observer_only: bool = False


class PrimitiveBindingRegistry:
    def __init__(self) -> None:
        self._bindings: dict[str, list[PrimitiveBinding]] = {}

    def register(
        self,
        game_id: str,
        *,
        primitive_name: str,
        binding_id: str | None = None,
        observer_only: bool = False,
    ) -> Callable[[BindingFn], BindingFn]:
        resolved_game = str(game_id)
        resolved_binding = str(binding_id or f"{resolved_game}:{primitive_name}")

        def decorator(fn: BindingFn) -> BindingFn:
            existing = [
                binding
                for binding in self._bindings.get(resolved_game, [])
                if binding.binding_id != resolved_binding
            ]
            existing.append(
                PrimitiveBinding(
                    game_id=resolved_game,
                    binding_id=resolved_binding,
                    primitive_name=str(primitive_name),
                    solve=fn,
                    observer_only=bool(observer_only),
                )
            )
            self._bindings[resolved_game] = existing
            return fn

        return decorator

    def bindings_for(
        self,
        game_id: str,
        *,
        include_observer: bool = False,
    ) -> tuple[PrimitiveBinding, ...]:
        bindings = tuple(self._bindings.get(str(game_id), ()))
        if include_observer:
            return bindings
        return tuple(binding for binding in bindings if not binding.observer_only)

    def solve(self, game_id: str, state: Mapping[str, Any]) -> PrimitiveBindingResult | None:
        for binding in self.bindings_for(game_id):
            result = binding.solve(state)
            if result is not None:
                return result
        return None

    def observe(self, game_id: str, state: Mapping[str, Any]) -> tuple[PrimitiveBindingResult, ...]:
        results: list[PrimitiveBindingResult] = []
        for binding in self.bindings_for(game_id, include_observer=True):
            if not binding.observer_only:
                continue
            result = binding.solve(state)
            if result is not None:
                results.append(result)
        return tuple(results)


GLOBAL_PRIMITIVE_BINDINGS = PrimitiveBindingRegistry()


def register_binding(
    game_id: str,
    *,
    primitive_name: str,
    binding_id: str | None = None,
    observer_only: bool = False,
) -> Callable[[BindingFn], BindingFn]:
    return GLOBAL_PRIMITIVE_BINDINGS.register(
        game_id,
        primitive_name=primitive_name,
        binding_id=binding_id,
        observer_only=observer_only,
    )


def solve_binding(game_id: str, state: Mapping[str, Any]) -> PrimitiveBindingResult | None:
    return GLOBAL_PRIMITIVE_BINDINGS.solve(game_id, state)


def observe_bindings(game_id: str, state: Mapping[str, Any]) -> tuple[PrimitiveBindingResult, ...]:
    return GLOBAL_PRIMITIVE_BINDINGS.observe(game_id, state)


__all__ = [
    "GLOBAL_PRIMITIVE_BINDINGS",
    "PrimitiveBinding",
    "PrimitiveBindingRegistry",
    "PrimitiveBindingResult",
    "observe_bindings",
    "register_binding",
    "solve_binding",
]
