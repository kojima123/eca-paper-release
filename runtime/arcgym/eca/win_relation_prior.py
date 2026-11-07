"""Pre-score win-relation hypothesis layer (D-1 stage 1+2).

Design: docs/d1_prescore_win_hypothesis_design_2026-06-10.md.

A win relation leaves a signature in the board's ROLE COMPOSITION, so a
relation hypothesis can be formed BEFORE the first score event — the
structural gap behind every bias null-result so far (the confirmed-
relation bias could never act because 4/5 generic-only games never
produce the first score it waits for).

Stage 1 (this module, static): per-colour-class component statistics ->
rule-based relation priors (rules R1-R5, inducted from the 23 M-3 win
predicates, shared across ALL games — no per-game tuning, leave-one-out
validated).

Stage 2 (this module, dynamic filter): given probe-period observations
(which classes responded to the agent's actions, and the actor class if
identified), drop hypotheses whose roles the agent cannot influence
(F1/F2) and whose fulfilment is not measurable (F3).

Vocabulary-honesty: games whose win is outside the prototype vocabulary
(sequence/alignment families, e.g. lp85/sk48/tr87) should get NO
hypothesis — an empty list is a first-class, CORRECT output. A wrong
hypothesis distorts exploration (the v4=0.13 lesson); no hypothesis
leaves it unchanged.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


MAX_CLASSES = 8
MAX_HYPOTHESES = 3

# R1: repeated-small-component signature (pegs / markers).
R1_MIN_COMPONENTS = 4
R1_MAX_SIZE_CV = 0.45
R1_MAX_AREA_SHARE = 0.25
# R2: matched-count pair signature (pieces onto targets).
R2_MIN_COMPONENTS = 2
R2_MAX_COUNT_DIFF = 1
# R3: unique-small-distinct signature (a goal).
R3_MAX_AREA_SHARE = 0.06
# A goal is a COMPACT blob, not a full-width UI bar / rail. The bbox
# aspect ratio (longer side / shorter side) caps how elongated a goal
# candidate may be. Calibrated from measurement: the step-counter bars
# are full board width (64x1 -> aspect 64); genuine but bar-ish goal
# candidates seen in the gate sit at aspect <= 21 (ls20) / <= 5 (wa30).
# A threshold of 24 drops ONLY the full-width counters (re86/wa30/tu93
# false reach->stepbar) without suppressing the legitimate reach-class
# hypotheses that keep the relation-class gate above 60%.
R3_MAX_GOAL_ASPECT = 24.0


@dataclass
class ClassStats:
    value: int
    n_components: int
    mean_size: float
    size_cv: float            # coefficient of variation of component sizes
    area_share: float
    cx: float
    cy: float
    bbox_shapes: tuple[tuple[int, int], ...]  # (w, h) per component


@dataclass
class PriorHypothesis:
    relation: str             # "reach" | "coverage" | "count_drop"
    subject: int
    object: int | None
    prior: float
    evidence: str             # which rule fired (trace / audit)


def _components_of(mask: np.ndarray) -> list[dict[str, int]]:
    """4-connected components of a boolean mask (BFS, numpy-light)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[dict[str, int]] = []
    ys, xs = np.where(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        queue = deque([(x0, y0)])
        seen[y0, x0] = True
        size = 0
        min_x = max_x = x0
        min_y = max_y = y0
        while queue:
            x, y = queue.popleft()
            size += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((nx, ny))
        out.append({
            "size": size,
            "w": max_x - min_x + 1,
            "h": max_y - min_y + 1,
        })
    return out


def class_stats(frame: Any) -> dict[int, ClassStats]:
    """Per-colour-class component statistics, background (modal) excluded."""
    arr = np.asarray(frame)
    if arr.ndim != 2 or arr.size == 0:
        return {}
    values, counts = np.unique(arr, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    total = float(arr.size)
    # interest classes: smallest pixel-share first (structure over sheets)
    order = sorted(
        (int(v) for v in values if int(v) != background),
        key=lambda v: int(counts[values.tolist().index(v)]),
    )[:MAX_CLASSES]
    stats: dict[int, ClassStats] = {}
    for value in order:
        mask = arr == value
        comps = _components_of(mask)
        if not comps:
            continue
        sizes = np.array([c["size"] for c in comps], dtype=float)
        ys, xs = np.where(mask)
        stats[value] = ClassStats(
            value=value,
            n_components=len(comps),
            mean_size=float(sizes.mean()),
            size_cv=float(sizes.std() / sizes.mean()) if sizes.mean() > 0 else 0.0,
            area_share=float(sizes.sum()) / total,
            cx=float(np.mean(xs)),
            cy=float(np.mean(ys)),
            bbox_shapes=tuple((c["w"], c["h"]) for c in comps),
        )
    return stats


# -- stage 1: static relation priors (rules R1-R5) ---------------------------


def relation_priors(
    frame: Any,
    *,
    actor_class: int | None = None,
) -> list[PriorHypothesis]:
    """Rule-based relation priors from the static role signature.

    Rules are SHARED across all games (no game ids, no colour literals);
    they encode role archetypes inducted from the 23 M-3 win predicates.
    Returns up to MAX_HYPOTHESES ranked by prior; an EMPTY list means
    "no vocabulary-consistent hypothesis" and is a correct output for
    out-of-vocabulary mechanics.
    """
    stats = class_stats(frame)
    if not stats:
        return []
    hypotheses: list[PriorHypothesis] = []

    # R1: many same-shaped small components -> consumable / coverable set.
    for value, s in stats.items():
        if (
            s.n_components >= R1_MIN_COMPONENTS
            and s.size_cv <= R1_MAX_SIZE_CV
            and s.area_share <= R1_MAX_AREA_SHARE
        ):
            hypotheses.append(PriorHypothesis(
                "count_drop", value, None, 0.5, "R1 repeated-small-set",
            ))

    # R2: matched-count class pair -> coverage (both directions kept;
    # the probe-period filter or the post-score attribution decides).
    values = sorted(stats)
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            sa, sb = stats[a], stats[b]
            if (
                min(sa.n_components, sb.n_components) >= R2_MIN_COMPONENTS
                and abs(sa.n_components - sb.n_components) <= R2_MAX_COUNT_DIFF
            ):
                separation = float(np.hypot(sa.cx - sb.cx, sa.cy - sb.cy))
                if separation < 2.0:
                    continue
                hypotheses.append(PriorHypothesis(
                    "coverage", a, b, 0.6, "R2 matched-count-pair",
                ))
                hypotheses.append(PriorHypothesis(
                    "coverage", b, a, 0.6, "R2 matched-count-pair",
                ))

    # R3: unique, small, distinct component -> a goal to reach.
    for value, s in stats.items():
        # The actor's own body is never the goal: skip a unique-small
        # class that the probe identified as the agent's body (else the
        # else-branch below would emit reach(other -> actor), walking a
        # decoy toward the actor — the measured tu93/object-identity bug).
        if actor_class is not None and value == actor_class:
            continue
        if s.n_components == 1 and s.area_share <= R3_MAX_AREA_SHARE:
            # Compactness gate: UI bars / rails (highly elongated bbox)
            # are not goals even when unique and small in area.
            gw, gh = s.bbox_shapes[0]
            if max(gw, gh) / max(1, min(gw, gh)) > R3_MAX_GOAL_ASPECT:
                continue
            subject = actor_class if actor_class not in (None, value) else None
            if subject is not None:
                hypotheses.append(PriorHypothesis(
                    "reach", int(subject), value, 0.65, "R3 unique-goal+actor",
                ))
            else:
                # subject unknown until the probe period identifies the
                # actor; keep a weaker placeholder against every other
                # mobile-capable class? No — placeholder subjects invite
                # false roles. Emit only when ONE other class exists.
                others = [v for v in stats if v != value]
                if len(others) == 1:
                    hypotheses.append(PriorHypothesis(
                        "reach", others[0], value, 0.5, "R3 unique-goal",
                    ))

    # R5: merge signature — exactly two same-shaped, spatially separated
    # components of one class (two cursors that must collide / two blobs
    # that must unify). The win is phase-equivalent to count_drop
    # (2 -> 1 components). Always evaluated; the shape identity + pair
    # separation keep it from firing on arbitrary multi-blob classes.
    for value, s in stats.items():
        if s.n_components == 2 and s.size_cv <= 0.25:
            shapes = set(s.bbox_shapes)
            if len(shapes) == 1:
                hypotheses.append(PriorHypothesis(
                    "count_drop", value, None, 0.58, "R5 merge-pair",
                ))

    hypotheses.sort(key=lambda h: h.prior, reverse=True)
    return hypotheses[:MAX_HYPOTHESES]


# -- stage 2: probe-period filtering (F1-F3) ----------------------------------


@dataclass
class ClassResponse:
    """Per-class response taxonomy collected over the probe period.

    The ECA observation layer's insight applied to roles: HOW a class
    responds to the agent separates win-consistent roles from decoys.
      moved          : pixel count roughly conserved while position
                       changed (a steerable piece -> coverage subject)
      count_dropped  : pixel count dropped at least once (consumable ->
                       count_drop evidence)
      change_rate    : fraction of probe steps where the class changed
                       at all; ~1.0 == self-moving (bounce sprites /
                       animations) -- excluded from every role
    """
    moved: bool = False
    count_dropped: bool = False
    change_rate: float = 0.0


SELF_MOVING_RATE = 0.8
MIN_CONFIDENT_PRIOR = 0.55
# F4: an object whose centroid stays within this Manhattan distance of the
# actor's centroid AND moves with the probe is treated as part of the
# actor's own sprite (a co-mover), not a goal. Tuned to the tu93 walker
# (dir-dot colour 4 + body colour 9 in the same 3x3 sprite ~1-2 cells
# apart); independently-moving goals in other games sit much farther away.
ACTOR_COMOVER_DIST = 2


def _centroid(arr: np.ndarray, value: int) -> tuple[float, float] | None:
    ys, xs = np.where(arr == value)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def filter_with_probe_evidence(
    hypotheses: Iterable[PriorHypothesis],
    *,
    frame: Any,
    actor_class: int | None,
    responsive_classes: Iterable[int],
    class_responses: dict[int, ClassResponse] | None = None,
) -> list[PriorHypothesis]:
    """F1-F3: refine static priors with probe-period observations.

    F1 actor knowledge: with a known actor, reach hypotheses get their
       subject pinned to it (and gain confidence); without any actor
       after probing, reach hypotheses are demoted.
    F2 influence filter: keep a hypothesis only if at least one of its
       roles was OBSERVED to change in response to the agent's actions
       (a win relation must be steerable). With the response taxonomy
       (ECA-grade observation) the requirement sharpens per relation:
         * self-moving classes (change_rate >= SELF_MOVING_RATE) are
           excluded from every role -- bounce sprites pass naive
           "responsive" checks (the cn04/m0r0 false count_drops);
         * a coverage SUBJECT must have a MOVED response -- click/cycle
           boards recolour but nothing translates, separating UI rows
           from genuine pieces-onto-targets (the cd82/ft09/sb26/sk48/
           lp85 false coverages);
         * a count_drop subject with an observed drop is boosted.
    F3 measurability: both roles must exist on the current frame.
    """
    arr = np.asarray(frame)
    present = set(int(v) for v in np.unique(arr))
    responsive = set(int(v) for v in responsive_classes)
    responses = class_responses or {}

    def _self_moving(value: int) -> bool:
        # Bounce/animation decoys change every step WITHOUT being the
        # agent's body and WITHOUT being consumed. Two protections (both
        # learned from measurement): the identified ACTOR changes every
        # step precisely BECAUSE the agent steers it, and a consumable
        # being rapidly consumed also changes every step.
        if actor_class is not None and value == actor_class:
            return False
        resp = responses.get(value)
        return bool(
            resp
            and resp.change_rate >= SELF_MOVING_RATE
            and not resp.count_dropped
        )

    out: list[PriorHypothesis] = []
    for hyp in hypotheses:
        relation, subject, obj = hyp.relation, hyp.subject, hyp.object
        prior, evidence = hyp.prior, hyp.evidence
        # F1
        if relation == "reach":
            if actor_class is not None:
                if subject != actor_class:
                    subject = int(actor_class)
                    evidence += "+F1 actor-pinned"
                prior = min(1.0, prior + 0.15)
            else:
                prior -= 0.25
                evidence += "+F1 no-actor-demote"
        # F3
        if subject not in present or (obj is not None and obj not in present):
            continue
        # F4 actor-comover exclusion: a reach/coverage OBJECT that is part
        # of the ACTOR's own sprite is not a goal. Measured tu93 failure:
        # the walker BODY (colour 9) co-moves with the actor (colour 4
        # dir-dot) because they are the same sprite, so the directed walk
        # chased the actor itself instead of the static exit (colour 14).
        # Precise condition (not just "moved", which would drop genuine
        # independently-moving goals in ar25/bp35): the object MOVED during
        # the probe AND its current centroid sits within ACTOR_COMOVER_DIST
        # of the actor's centroid -> it travels WITH the actor = part of
        # the actor body.
        if (
            obj is not None
            and relation in ("reach", "coverage")
            and actor_class is not None
        ):
            obj_resp = responses.get(obj)
            if obj_resp is not None and obj_resp.moved:
                ac = _centroid(arr, int(actor_class))
                oc = _centroid(arr, int(obj))
                if (
                    ac is not None
                    and oc is not None
                    and abs(ac[0] - oc[0]) + abs(ac[1] - oc[1])
                    <= ACTOR_COMOVER_DIST
                ):
                    continue
        # F2 -- self-moving exclusion applies to every role.
        roles = {subject} | ({obj} if obj is not None else set())
        if any(_self_moving(role) for role in roles):
            continue
        if responsive and not (roles & responsive):
            continue
        # F2 sharpening with the response taxonomy. Evidence BOOSTS, it
        # does not gate: a 30-action probe often never reaches the stage
        # where a piece is actually pushed, so requiring a moved subject
        # killed true coverages in measurement (ar25/r11l). Boosting
        # observed evidence + the weak-hypothesis threshold below give
        # the same selectivity without the false negatives.
        if responses:
            if relation == "coverage":
                subject_resp = responses.get(subject)
                if subject_resp is not None and subject_resp.moved:
                    prior = min(1.0, prior + 0.1)
                    evidence += "+F2 moved-subject"
            elif relation == "count_drop":
                subject_resp = responses.get(subject)
                if subject_resp is not None and subject_resp.count_dropped:
                    prior = min(1.0, prior + 0.2)
                    evidence += "+F2 drop-observed"
        if prior <= 0.0:
            continue
        out.append(PriorHypothesis(relation, subject, obj, prior, evidence))
    out.sort(key=lambda h: h.prior, reverse=True)
    # Weak-hypothesis threshold (vocabulary honesty): when even the best
    # surviving hypothesis is weak, exploring undistorted beats steering
    # toward a guess (the v4=0.13 lesson). Out-of-vocabulary boards tend
    # to produce only weak accidental signatures.
    if out and out[0].prior < MIN_CONFIDENT_PRIOR:
        return []
    return out[:MAX_HYPOTHESES]


def collect_class_responses(
    frames: list,
) -> dict[int, ClassResponse]:
    """Build the per-class response taxonomy from consecutive probe frames."""
    responses: dict[int, ClassResponse] = {}
    if len(frames) < 2:
        return responses
    steps = 0
    change_counts: dict[int, int] = {}
    first = np.asarray(frames[0])
    last = np.asarray(frames[-1])
    for prev, cur in zip(frames, frames[1:]):
        prev = np.asarray(prev)
        cur = np.asarray(cur)
        if prev.shape != cur.shape:
            continue
        steps += 1
        diff = prev != cur
        if not bool(diff.any()):
            continue
        changed_values = set(
            int(v) for v in np.unique(np.concatenate([prev[diff], cur[diff]]))
        )
        for value in changed_values:
            change_counts[value] = change_counts.get(value, 0) + 1
            resp = responses.setdefault(value, ClassResponse())
            prev_mask = prev == value
            cur_mask = cur == value
            prev_n = int(prev_mask.sum())
            cur_n = int(cur_mask.sum())
            if prev_n > 0 and cur_n > 0 and abs(cur_n - prev_n) <= max(2, prev_n // 5):
                pys, pxs = np.where(prev_mask)
                cys, cxs = np.where(cur_mask)
                moved = (
                    abs(float(np.mean(pxs)) - float(np.mean(cxs))) >= 1.0
                    or abs(float(np.mean(pys)) - float(np.mean(cys))) >= 1.0
                )
                if moved:
                    resp.moved = True
    for value, resp in responses.items():
        resp.change_rate = change_counts.get(value, 0) / max(1, steps)
        if first.shape == last.shape:
            # NET consumption only: transient dips (cursor overwrite,
            # animation frames) are not consumption evidence — they were
            # the measured false count_drop boosts (ar25/ft09/r11l/tn36).
            first_n = int((first == value).sum())
            last_n = int((last == value).sum())
            resp.count_dropped = bool(first_n > 0 and last_n < first_n * 0.9)
    return responses


__all__ = [
    "MAX_HYPOTHESES",
    "SELF_MOVING_RATE",
    "ClassResponse",
    "collect_class_responses",
    "ClassStats",
    "PriorHypothesis",
    "class_stats",
    "filter_with_probe_evidence",
    "relation_priors",
]
