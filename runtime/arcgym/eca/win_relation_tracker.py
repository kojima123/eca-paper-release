"""WinRelationTracker: runtime win-relation hypotheses for exploration.

Stage 2 (phase902) closed the loop AFTER the first score event; the
null-controls then showed that is structurally too late — in the
generic-only pseudo-holdout 4/5 games never produce the first score the
bias waits for. Stage 3 (D-1, this version) closes the gap with
PRE-SCORE hypotheses:

  * the first PROBE_STEPS frames feed the role-signature prior layer
    (arcgym/eca/win_relation_prior.py, D1-a gate passed at 64% /
    pseudo-holdout 4/5);
  * until a score event confirms a relation, the top prior hypothesis
    drives the exploration bias;
  * a HYPOTHESIS LEASE (the arbitration-layer principle applied to
    beliefs) rotates to the next candidate when fulfilment stops
    improving for HYPOTHESIS_LEASE_STEPS — no lock-in on a wrong guess
    (the cn04 lesson at the belief level);
  * a confirmed attribution (score event, 92% accuracy) always
    overrides the prior.

Everything is in-episode and frame-only — holdout-safe by construction.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .win_relation import TREND_WINDOW, RelationHypothesis, attribute_score_event
from .win_relation_prior import (
    PriorHypothesis,
    collect_class_responses,
    filter_with_probe_evidence,
    relation_priors,
)


PROBE_STEPS = 30
HYPOTHESIS_LEASE_STEPS = 40

_DIRS: dict[str, tuple[int, int]] = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}


def _centroid(arr: np.ndarray, value: int) -> tuple[float, float] | None:
    ys, xs = np.where(arr == value)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


class WinRelationTracker:
    """Pre-score priors + post-score confirmation + hypothesis lease."""

    def __init__(self) -> None:
        self._frames: deque[np.ndarray] = deque(maxlen=TREND_WINDOW)
        self._probe_frames: list[np.ndarray] = []
        self._prior_computed = False
        self._priors: list[PriorHypothesis] = []
        self._prior_index = 0
        self._lease_steps = 0
        self._best_fulfilment: float | None = None
        self._confirmed: RelationHypothesis | None = None
        self.attribution_count = 0
        # Lease consumption is gated on the bias actually being CONSULTED
        # since the last observation — measured failure mode: in tu93 the
        # exploration never reached the injection point, so the lease
        # free-ran and exhausted all candidates before the bias could act.
        self._bonus_queried = False

    # -- observation (wired from observe_transition) -------------------------

    def observe(
        self,
        post_frame: Any,
        score_delta: int,
        *,
        actor_class: int | None = None,
    ) -> None:
        if int(score_delta) > 0:
            if len(self._frames) >= 3:
                hypotheses = attribute_score_event(list(self._frames))
                if hypotheses:
                    self._confirmed = hypotheses[0]
                    self.attribution_count += 1
            self._frames.clear()
            return
        try:
            arr = np.asarray(post_frame)
        except Exception:
            return
        if arr.ndim != 2:
            return
        self._frames.append(arr)
        if not self._prior_computed:
            self._probe_frames.append(arr)
            if len(self._probe_frames) > PROBE_STEPS:
                self._compute_priors(actor_class)
        elif self._confirmed is None:
            self._update_hypothesis_lease(arr)

    def _compute_priors(self, actor_class: int | None) -> None:
        self._prior_computed = True
        try:
            responses = collect_class_responses(self._probe_frames)
            static = relation_priors(
                self._probe_frames[0], actor_class=actor_class,
            )
            self._priors = filter_with_probe_evidence(
                static,
                frame=self._probe_frames[-1],
                actor_class=actor_class,
                responsive_classes=set(responses),
                class_responses=responses,
            )
        except Exception:
            self._priors = []
        finally:
            self._probe_frames.clear()

    # -- hypothesis lease ------------------------------------------------------

    def _active_prior(self) -> PriorHypothesis | None:
        if 0 <= self._prior_index < len(self._priors):
            return self._priors[self._prior_index]
        return None

    def _fulfilment(self, arr: np.ndarray, hyp: PriorHypothesis) -> float | None:
        """Cheap fulfilment proxy used only for lease progress."""
        if hyp.relation in ("reach", "coverage") and hyp.object is not None:
            subject = _centroid(arr, int(hyp.subject))
            target = _centroid(arr, int(hyp.object))
            if subject is None or target is None:
                return None
            h, w = arr.shape
            diag = float(np.hypot(w, h))
            dist = float(np.hypot(subject[0] - target[0], subject[1] - target[1]))
            return max(0.0, 1.0 - dist / diag)
        if hyp.relation == "count_drop":
            pixels = int((arr == int(hyp.subject)).sum())
            return -float(pixels)  # fewer pixels == progress
        return None

    def _update_hypothesis_lease(self, arr: np.ndarray) -> None:
        hyp = self._active_prior()
        if hyp is None:
            return
        if not self._bonus_queried:
            # The bias was never consulted this step: the hypothesis got
            # no chance to act, so its lease must not burn.
            return
        self._bonus_queried = False
        value = self._fulfilment(arr, hyp)
        if value is not None and (
            self._best_fulfilment is None or value > self._best_fulfilment + 1e-9
        ):
            self._best_fulfilment = value
            self._lease_steps = 0
            return
        self._lease_steps += 1
        if self._lease_steps >= HYPOTHESIS_LEASE_STEPS:
            # Rotate to the next candidate; after the last one, fall back
            # to no-hypothesis (undistorted exploration) for the episode.
            self._prior_index += 1
            self._lease_steps = 0
            self._best_fulfilment = None

    # -- active belief ---------------------------------------------------------

    def active_hypothesis(self) -> Any | None:
        """Confirmed attribution wins; otherwise the leased prior."""
        if self._confirmed is not None:
            return self._confirmed
        return self._active_prior()

    @property
    def confirmed(self) -> RelationHypothesis | None:
        return self._confirmed

    # -- exploration bias ------------------------------------------------------

    def active_object_centroid(self, frame: Any) -> tuple[int, int] | None:
        """Integer (x, y) centroid of the ACTIVE hypothesis's object on the
        current frame, or None. Used to hand a concrete goal cell to the
        frame-only navigator (learned-walkability BFS), so the directed
        walk can route AROUND obstacles instead of greedily descending the
        centroid gradient into walls."""
        hypothesis = self.active_hypothesis()
        if hypothesis is None or getattr(hypothesis, "object", None) is None:
            return None
        arr = np.asarray(frame)
        if arr.ndim != 2:
            return None
        c = _centroid(arr, int(hypothesis.object))
        if c is None:
            return None
        return int(round(c[0])), int(round(c[1]))

    def goal_ward_actions(self) -> list[str]:
        """Directional actions ordered by direction bonus (descending),
        positive-bonus only, computed on the tracker's last buffered
        frame. Lets call sites without a frame in scope (e.g. the
        fixed-position actor heuristic) consult the active hypothesis.
        Empty when no hypothesis / no object / no frame yet."""
        if not self._frames:
            return []
        frame = self._frames[-1]
        ranked: list[tuple[float, str]] = []
        for action in _DIRS:
            bonus = self.direction_bonus(frame, action)
            if bonus > 0.0:
                ranked.append((bonus, action))
        ranked.sort(reverse=True)
        return [action for _bonus, action in ranked]

    def direction_bonus(self, frame: Any, action_name: str) -> float:
        """Bonus in [0, 1] for a directional action moving the ACTIVE
        hypothesis's subject toward its object on the current frame."""
        hypothesis = self.active_hypothesis()
        self._bonus_queried = True
        if hypothesis is None or getattr(hypothesis, "object", None) is None:
            return 0.0
        direction = _DIRS.get(str(action_name or "").strip().upper())
        if direction is None:
            return 0.0
        arr = np.asarray(frame)
        if arr.ndim != 2:
            return 0.0
        subject = _centroid(arr, int(hypothesis.subject))
        target = _centroid(arr, int(hypothesis.object))
        if subject is None or target is None:
            return 0.0
        vx, vy = target[0] - subject[0], target[1] - subject[1]
        norm = float(np.hypot(vx, vy))
        if norm < 1e-6:
            return 0.0
        dx, dy = direction
        return max(0.0, (dx * vx + dy * vy) / norm)

    def summary(self) -> dict[str, Any]:
        active = self.active_hypothesis()
        return {
            "attribution_count": self.attribution_count,
            "prior_count": len(self._priors),
            "prior_index": self._prior_index,
            "confirmed": (
                {
                    "relation": self._confirmed.relation,
                    "subject": self._confirmed.subject,
                    "object": self._confirmed.object,
                }
                if self._confirmed is not None
                else None
            ),
            "active": (
                {
                    "relation": active.relation,
                    "subject": active.subject,
                    "object": getattr(active, "object", None),
                }
                if active is not None
                else None
            ),
        }


__all__ = ["HYPOTHESIS_LEASE_STEPS", "PROBE_STEPS", "WinRelationTracker"]
