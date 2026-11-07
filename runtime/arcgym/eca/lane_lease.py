"""LaneLease: the episode-local no-score lease arbiter (pillar A1).

Design (docs/architecture_remediation_design_2026-06-10.md):

The lane-selection stack orders lanes by CODE POSITION, so a lane placed
high either monopolises the episode or, placed low, starves. Seven
recorded incidents share this shape (BODY_SCHEMA monopoly: s5i5 ON=0 /
OFF=3; click_toggle_search 118/120 steps on tn36; the cn04 mirror-cert
mis-latch; ...). Detection-side fixes have a hard ceiling — the cn04
pair passed both the geometric certificate and 40 dynamic prediction
confirmations — so arbitration must be a SEPARATE layer.

LaneLease implements the minimal arbitration rule:

  * every decision a primitive wins consumes lease budget;
  * a primitive that has produced a positive score delta this episode
    becomes a SCORE OWNER and never exhausts (established solvers keep
    their slot);
  * a primitive that has NOT scored exhausts after `grant_steps`
    consumed decisions and is skipped by participating gates, letting
    lower-priority lanes run.

Everything here uses only in-episode observations (no canonical
baseline, no game ids), so the same arbitration works on unseen
holdout games — the property that made the phase898b ownership yield
fix cn04 while keeping gtm0.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


DEFAULT_GRANT_STEPS = 40


class LaneLease:
    """Per-primitive no-score lease with score-owner permanence."""

    def __init__(self, grant_steps: int = DEFAULT_GRANT_STEPS) -> None:
        self._grant_steps = max(1, int(grant_steps))
        self._consumed: Counter[str] = Counter()
        self._score_owners: set[str] = set()

    # -- bookkeeping ---------------------------------------------------------

    def consume(self, primitive_id: str | None) -> None:
        pid = str(primitive_id or "").strip()
        if pid:
            self._consumed[pid] += 1

    def on_score(self, primitive_id: str | None) -> None:
        pid = str(primitive_id or "").strip()
        if pid:
            self._score_owners.add(pid)

    # -- arbitration queries -------------------------------------------------

    def exhausted(self, primitive_id: str | None) -> bool:
        """True when the primitive has used its grant WITHOUT ever scoring.

        Score owners never exhaust: a primitive that produced score this
        episode has earned its slot (the cn04 lesson — sprite_anchor owns
        cn04's score, so a later perceptually-convincing impostor must
        yield to it, not the other way around).
        """
        pid = str(primitive_id or "").strip()
        if not pid or pid in self._score_owners:
            return False
        return self._consumed[pid] >= self._grant_steps

    @property
    def score_owners(self) -> set[str]:
        return self._score_owners

    def owned_by_other(self, primitive_id: str | None) -> bool:
        """True when a DIFFERENT primitive owns score this episode.

        This is the phase898b ownership-yield predicate, generalised.
        """
        pid = str(primitive_id or "").strip()
        owners = self._score_owners
        if not owners:
            return False
        return pid not in owners

    def summary(self) -> dict[str, Any]:
        return {
            "grant_steps": self._grant_steps,
            "consumed": dict(self._consumed),
            "score_owners": sorted(self._score_owners),
            "exhausted": sorted(
                pid
                for pid in self._consumed
                if self.exhausted(pid)
            ),
        }


__all__ = ["DEFAULT_GRANT_STEPS", "LaneLease"]
