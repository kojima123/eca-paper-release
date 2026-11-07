"""Win-relation hypothesis layer, prototype (pillar W1).

The deepest unknown-game gap (capability analysis D-1): the agent can
CAUSE change but cannot tell which changes move TOWARD a win, because
no win relation is named and no roles are perceived. P12 demonstrated
the converse — once the relation is named (C1 vocabulary) and the roles
are perceived (C2), the win quantity is always a dense gradient.

This module lifts that finding from per-game code into a meta layer:

  1. a small RELATION VOCABULARY, inducted from the 23 M-3 win
     predicates: reach / coverage / count_drop (+ alignment-by-coverage);
  2. palette-free MEASUREMENT of each relation instance's fulfilment on
     a single frame (colour classes are roles, never literal ids);
  3. ATTRIBUTION: given the frames leading up to a score event, find
     the relation instances whose fulfilment rose monotonically and
     peaked at the score — those are the win-relation hypotheses.

Prototype scope: attribution runs OFFLINE (tools/win_relation_probe.py)
and is gated on >= 70% agreement with the known M-3 win relations
before any runtime bias injection. No game ids, no colour literals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


MAX_CLASSES = 8          # interest colour classes considered per frame
TREND_WINDOW = 12        # frames before the score event used for trends


# -- per-frame colour-class structure ---------------------------------------


def _class_stats(frame: np.ndarray) -> dict[int, dict[str, Any]]:
    """Pixel stats per colour class, background (modal colour) excluded."""
    arr = np.asarray(frame)
    if arr.ndim != 2 or arr.size == 0:
        return {}
    values, counts = np.unique(arr, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    stats: dict[int, dict[str, Any]] = {}
    order = sorted(
        (int(v) for v in values if int(v) != background),
        key=lambda v: int(counts[list(values).index(v)]),
    )
    for value in order[:MAX_CLASSES]:
        ys, xs = np.where(arr == value)
        stats[value] = {
            "pixels": int(len(xs)),
            "cx": float(np.mean(xs)),
            "cy": float(np.mean(ys)),
            "min_x": int(np.min(xs)),
            "max_x": int(np.max(xs)),
            "min_y": int(np.min(ys)),
            "max_y": int(np.max(ys)),
            "coords": set(zip(xs.tolist(), ys.tolist())),
        }
    return stats


# -- relation measurements (fulfilment in [0, 1], higher = closer to win) ----


def measure_reach(a: dict[str, Any], b: dict[str, Any], diag: float) -> float:
    """reach(A, B): A's centroid approaching B's centroid."""
    dist = float(np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]))
    if diag <= 0:
        return 0.0
    return max(0.0, 1.0 - dist / diag)


def measure_coverage(a: dict[str, Any], b: dict[str, Any]) -> float:
    """coverage(A, B): fraction of B's pixels adjacent to (Chebyshev
    distance <= 1) an ACTUAL pixel of A.

    Measured on real pixels, not the bounding box: a bbox proxy rises
    during mere approach and shadows the reach relation (measured
    2026-06-10: dc22/su15 mis-attributions all came from bbox coverage
    outranking reach on the same role pair)."""
    if not b["coords"]:
        return 0.0
    a_coords = a["coords"]
    covered = 0
    for (x, y) in b["coords"]:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (x + dx, y + dy) in a_coords:
                    covered += 1
                    break
            else:
                continue
            break
    return covered / len(b["coords"])


def measure_count_drop(pixels_now: int, pixels_first: int) -> float:
    """count_drop(X): X disappearing (peg removal / target consumption)."""
    if pixels_first <= 0:
        return 0.0
    return max(0.0, 1.0 - pixels_now / pixels_first)


# -- attribution over a frame window -----------------------------------------


@dataclass
class RelationHypothesis:
    relation: str            # "reach" | "coverage" | "count_drop"
    subject: int             # colour class id (role: the moving/consumed thing)
    object: int | None       # colour class id (role: the target), None for count_drop
    final: float             # fulfilment at the scoring frame
    rise: float              # fulfilment gain over the window
    monotonicity: float      # fraction of non-decreasing steps

    @property
    def strength(self) -> float:
        return self.final * max(0.0, self.rise) * self.monotonicity


def _series_features(series: Sequence[float]) -> tuple[float, float, float]:
    if not series:
        return 0.0, 0.0, 0.0
    final = float(series[-1])
    rise = float(series[-1] - series[0])
    if len(series) < 2:
        return final, rise, 1.0
    non_decreasing = sum(
        1 for i in range(1, len(series)) if series[i] >= series[i - 1] - 1e-9
    )
    return final, rise, non_decreasing / (len(series) - 1)


def attribute_score_event(
    frames: Sequence[np.ndarray],
) -> list[RelationHypothesis]:
    """Given the frames LEADING UP TO a score event (oldest first, last =
    the frame on which the scoring action was taken), rank the relation
    instances whose fulfilment rose toward the score.

    Returns hypotheses sorted by strength (best first). Empty when the
    window is too short or no class persists across it.
    """
    window = list(frames)[-TREND_WINDOW:]
    if len(window) < 3:
        return []
    per_frame = [_class_stats(f) for f in window]
    # Only classes present in every frame of the window are role
    # candidates (level transitions / flicker drop out naturally).
    common = set(per_frame[0])
    for stats in per_frame[1:]:
        common &= set(stats)
    if not common:
        return []
    h, w = np.asarray(window[0]).shape[:2]
    diag = float(np.hypot(w, h))

    hypotheses: list[RelationHypothesis] = []
    common_list = sorted(common)
    for subject in common_list:
        # count_drop(subject)
        series = [
            measure_count_drop(stats[subject]["pixels"], per_frame[0][subject]["pixels"])
            for stats in per_frame
        ]
        final, rise, mono = _series_features(series)
        if rise > 0:
            hypotheses.append(RelationHypothesis(
                "count_drop", subject, None, final, rise, mono,
            ))
        for obj in common_list:
            if obj == subject:
                continue
            reach_series = [
                measure_reach(stats[subject], stats[obj], diag)
                for stats in per_frame
            ]
            final, rise, mono = _series_features(reach_series)
            if rise > 0:
                hypotheses.append(RelationHypothesis(
                    "reach", subject, obj, final, rise, mono,
                ))
            cov_series = [
                measure_coverage(stats[subject], stats[obj])
                for stats in per_frame
            ]
            final, rise, mono = _series_features(cov_series)
            if rise > 0:
                hypotheses.append(RelationHypothesis(
                    "coverage", subject, obj, final, rise, mono,
                ))
    hypotheses.sort(key=lambda hyp: hyp.strength, reverse=True)
    return hypotheses


__all__ = [
    "MAX_CLASSES",
    "TREND_WINDOW",
    "RelationHypothesis",
    "attribute_score_event",
    "measure_count_drop",
    "measure_coverage",
    "measure_reach",
]
