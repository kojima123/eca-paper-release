"""EffectModelBus: solver-agnostic readout of learned effect models (W2).

Phase L proved the self-repair loop on ONE effect family (hazards) wired
into ONE solver (mirror). Phase M added a second learner (transport)
observed from the generic actor-graph layer. This bus is the readout
side: any planner can query the repaired world model without knowing
which solver supplied the observations.

Contract (architecture remediation design, pillar W2): exactly two
connection shapes are exposed —

  1. hazard query   : cells whose visual-signature class resets/kills
  2. transport query: the displacement a signature class applies

Readers go through this bus instead of poking learner internals, so the
observation side (facts) and the prediction side (inference) stay
separate, and adding a third effect family is a bus method, not another
solver-specific wiring.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

Cell = tuple[int, int]
Signature = tuple
Vector = tuple[int, int]


class EffectModelBus:
    """Read-only facade over the learned effect models."""

    def __init__(
        self,
        tile_learner: Any,
        transport_learner: Any,
        action_effect_learner: Any = None,
    ) -> None:
        self._tile = tile_learner
        self._transport = transport_learner
        self._action_effect = action_effect_learner

    # -- hazard family (TileEffectLearner, Phase L) ---------------------------

    def hazard_cells(
        self,
        signatures: Mapping[Cell, Signature],
        *,
        exclude: Iterable[Cell] = (),
    ) -> set[Cell]:
        if self._tile is None:
            return set()
        try:
            return set(self._tile.hazard_cells(signatures, exclude=tuple(exclude)))
        except Exception:
            return set()

    # -- transport family (TransportEffectLearner, Phase M) -------------------

    def transport_vector(self, signature: Signature | None) -> Vector | None:
        if self._transport is None or signature is None:
            return None
        try:
            return self._transport.transport_vector(signature)
        except Exception:
            return None

    def transport_destination(
        self,
        cell: Cell,
        signatures: Mapping[Cell, Signature],
    ) -> Cell | None:
        """Where stepping onto `cell` actually lands, per the repaired
        model — or None when the cell's class is not a known transport."""
        vec = self.transport_vector(signatures.get(cell))
        if vec is None:
            return None
        return (int(cell[0] + vec[0]), int(cell[1] + vec[1]))

    # -- click-effect family (ActionEffectLearner, E-1 S1) --------------------

    def click_effect_sites(self, frame: Any) -> dict[Cell, str]:
        """Admitted click-operator sites on the current frame:
        {(x, y) centroid: effect_class}. The third effect family the bus
        docstring reserved — readers stay learner-agnostic."""
        if self._action_effect is None:
            return {}
        try:
            return dict(self._action_effect.click_effect_sites(frame))
        except Exception:
            return {}

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, learner in (
            ("tile", self._tile),
            ("transport", self._transport),
            ("action_effect", self._action_effect),
        ):
            try:
                out[name] = dict(learner.summary()) if learner is not None else {}
            except Exception:
                out[name] = {}
        return out


__all__ = ["EffectModelBus"]
