"""RelationalObjectiveInducer: induce per-movable-component target relations
from interaction, so the objective is a movable<->target relation rather than an
all-color aggregate.

The A0-B3b arc isolated the single hard core of requirement 3: the precondition /
objective is a PER-FAMILY relation between a movable object and its target
(proximity / coverage / reflection / slot-recolor) that no all-color generic
primitive (A4/A7/A8), local LLM (A9), or geometric+fill basis (B2/B3/B3b) could
isolate. This module produces the missing piece: given interaction (which
component moved, via moving_sprite_tracker), it emits CANDIDATE relations for the
movable component against the static structure, each with a per-frame evaluator.
A downstream score_closure_learner then grounds the candidate whose pre-score
value actually improves before score events.

It owns no game knowledge and selects no relation by itself — it enumerates
candidates and evaluates them; grounding is done by score-closure. Pure (frame in,
candidates/scalars out); diagnostic.
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None

from .moving_sprite_tracker import detect_sprite_moves

SCHEMA_VERSION = "eca_relational_objective_inducer.v1"
SPRITE_MAX_AREA = 120

RELATION_TYPES = ("approach_same_color", "approach_marker", "cover_markers", "reflect_axis")
_SPRITE_CENTROID_CACHE: dict[tuple[tuple[int, ...], int, bytes], list[tuple[float, float]]] = {}
_SPRITE_CENTROID_CACHE_LIMIT = 4096


@dataclass(frozen=True)
class TargetRelation:
    """A candidate objective: a movable color + a relation type + params."""
    movable_color: int
    relation_type: str
    params: tuple  # relation-specific (e.g. marker color, axis x)

    def key(self) -> str:
        return f"{self.relation_type}:c{self.movable_color}:{self.params}"


def _bg(frame: np.ndarray) -> int:
    vals, counts = np.unique(frame, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _sprite_centroids(frame: np.ndarray, color: int) -> list[tuple[float, float]]:
    arr = np.asarray(frame)
    key = (tuple(int(v) for v in arr.shape), int(color), arr.tobytes())
    cached = _SPRITE_CENTROID_CACHE.get(key)
    if cached is not None:
        return list(cached)
    out = []
    if _ndi is not None:
        lab, n = _ndi.label(arr == color)
        masks = [lab == i for i in range(1, n + 1)]
        for mask in masks:
            area = int(mask.sum())
            if 1 <= area <= SPRITE_MAX_AREA:
                ys, xs = np.where(mask)
                out.append((float(xs.mean()), float(ys.mean())))
    else:
        out = [
            (cx, cy)
            for area, cx, cy in _connected_component_centroids(arr == color)
            if 1 <= area <= SPRITE_MAX_AREA
        ]
    if len(_SPRITE_CENTROID_CACHE) >= _SPRITE_CENTROID_CACHE_LIMIT:
        _SPRITE_CENTROID_CACHE.pop(next(iter(_SPRITE_CENTROID_CACHE)))
    _SPRITE_CENTROID_CACHE[key] = list(out)
    return out


def _connected_component_centroids(mask: np.ndarray) -> list[tuple[int, float, float]]:
    """Return (area, cx, cy) for 4-neighbour components without scipy."""
    m = np.asarray(mask, dtype=bool)
    height, width = m.shape
    seen = np.zeros((height, width), dtype=bool)
    out: list[tuple[int, float, float]] = []
    for y0 in range(height):
        for x0 in range(width):
            if not m[y0, x0] or seen[y0, x0]:
                continue
            area = 0
            sum_x = 0
            sum_y = 0
            q: deque[tuple[int, int]] = deque([(y0, x0)])
            seen[y0, x0] = True
            while q:
                y, x = q.popleft()
                area += 1
                sum_x += x
                sum_y += y
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy = y + dy
                    xx = x + dx
                    if 0 <= yy < height and 0 <= xx < width and m[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        q.append((yy, xx))
            out.append((area, float(sum_x) / float(area), float(sum_y) / float(area)))
    return out


def _connected_component_masks(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-neighbour connected masks without scipy."""
    m = np.asarray(mask, dtype=bool)
    height, width = m.shape
    seen = np.zeros((height, width), dtype=bool)
    out: list[np.ndarray] = []
    for y0 in range(height):
        for x0 in range(width):
            if not m[y0, x0] or seen[y0, x0]:
                continue
            comp = np.zeros((height, width), dtype=bool)
            q: deque[tuple[int, int]] = deque([(y0, x0)])
            seen[y0, x0] = True
            while q:
                y, x = q.popleft()
                comp[y, x] = True
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy = y + dy
                    xx = x + dx
                    if 0 <= yy < height and 0 <= xx < width and m[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        q.append((yy, xx))
            out.append(comp)
    return out


def _binary_dilation4(mask: np.ndarray) -> np.ndarray:
    """One-step 4-neighbour dilation without scipy."""
    m = np.asarray(mask, dtype=bool)
    out = m.copy()
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def movable_colors(prev_frame: np.ndarray, cur_frame: np.ndarray) -> list[int]:
    """Colors of components that moved (interaction-grounded)."""
    moved = sorted({int(m.identity[0]) for m in detect_sprite_moves(prev_frame, cur_frame)})
    return moved


def induce_candidates(frame: np.ndarray, movable_colors_: Iterable[int]) -> list[TargetRelation]:
    """For each movable color, emit candidate target relations vs the static
    structure of `frame`. No selection — enumerate; grounding is downstream."""
    f = np.asarray(frame)
    bg = _bg(f)
    vals, counts = np.unique(f, return_counts=True)
    nonbg = [(int(v), int(c)) for v, c in zip(vals, counts) if int(v) != bg and c > 0]
    # Top-K rarest distinct non-bg colours are candidate TARGETS. A single global
    # rare colour mis-targets multi-object games (push/sokoban: the box must reach
    # a specific target, not whichever colour is globally rarest). Emitting the few
    # rarest as candidate targets lets object<->target pairs (box->target) form;
    # score_closure / the planner pick the pair that drives the score.
    rare_colors = [v for v, _c in sorted(nonbg, key=lambda x: x[1])[:3]]
    h, w = f.shape[:2]
    cands: list[TargetRelation] = []
    for mc in movable_colors_:
        mc = int(mc)
        # approach_same_color: only if the movable color has >=2 components (a partner)
        if len(_sprite_centroids(f, mc)) >= 2:
            cands.append(TargetRelation(mc, "approach_same_color", ()))
        # cover_markers / approach_marker: object mc -> each candidate target colour.
        # cover_markers is a SPARSE overlap count (0 until adjacent); approach_marker
        # is the DENSE -distance gradient that navigation/goal-reach AND push families
        # need for planning (A0's "actor-goal distance"). Both emitted per target.
        for tgt in rare_colors:
            if tgt != mc:
                cands.append(TargetRelation(mc, "approach_marker", (tgt,)))
                cands.append(TargetRelation(mc, "cover_markers", (tgt,)))
                # aligned_coverage (P18) is GATED default-off so the shipped candidate set
                # (and the strict-generic floor) is bit-identical unless explicitly enabled.
                if os.environ.get("ENABLE_ALIGNED_COVERAGE") == "1":
                    cands.append(TargetRelation(mc, "aligned_coverage", (tgt,)))
        # reflect_axis: BEST-AXIS SEARCH instead of a board-centre default. The
        # true symmetry axis of a mirror pair is the midpoint between its two
        # component centroids (geometry-derived, palette/board agnostic), not the
        # board centre w//2 (B6 grounded the coarse default only because the
        # collision happened to be near centre — fragile). Emit the geometric
        # axis candidates + the overall movable centroid + the coarse default;
        # score_closure_learner grounds whichever axis correlates with score.
        cents_mc = _sprite_centroids(f, mc)
        axes: set[int] = {w // 2}
        for a in range(len(cents_mc)):
            for b in range(a + 1, len(cents_mc)):
                axes.add(int(round((cents_mc[a][0] + cents_mc[b][0]) / 2.0)))
        xs_mc = np.where(f == mc)[1]
        if xs_mc.size:
            axes.add(int(round(float(xs_mc.mean()))))
        for ax in sorted(a for a in axes if 0 < a < w):
            cands.append(TargetRelation(mc, "reflect_axis", (ax,)))
    # aligned_coverage_global (P19): one frame-global candidate (component-level multi-colour
    # same-colour coverage). Gated default-off -> floor bit-identical unless enabled.
    if os.environ.get("ENABLE_ALIGNED_COVERAGE") == "1":
        cands.append(TargetRelation(0, "aligned_coverage_global", ()))
    return cands


def evaluate_relation(frame: np.ndarray, relation: TargetRelation) -> float:
    """Per-frame scalar for a candidate relation. Higher = closer to satisfying.
    Restricted to the movable color (not an all-color aggregate)."""
    f = np.asarray(frame)
    mc = relation.movable_color
    if relation.relation_type == "approach_same_color":
        cents = _sprite_centroids(f, mc)
        if len(cents) == 1:
            # MERGE = the two same-color components collided into ONE. For a
            # "make them meet" objective (m0r0's mirror cursor pair, any
            # collide-to-win family) the merge IS the goal, so it is MAXIMAL
            # proximity, not far. The old -300 fallback (P0 diagnostic
            # tools/p0_objective_diag_m0r0.py) poisoned the metric at exactly the
            # score moment, making greedy/best-first AVOID the scoring move (the
            # B7 active-scoring failure). Returning 0.0 (distance 0 = collided)
            # makes proximity perfectly monotone toward the score (monotone=1.00).
            return 0.0
        if len(cents) == 0:
            # The color vanished entirely (not a merge) -> genuinely far/absent.
            return -300.0
        return -min(
            abs(cents[a][0] - cents[b][0]) + abs(cents[a][1] - cents[b][1])
            for a in range(len(cents)) for b in range(len(cents)) if a != b
        )
    if relation.relation_type == "approach_marker":
        # DENSE -distance from the movable to the nearest marker (the goal). Unlike
        # cover_markers (sparse, 0 until adjacent), this gives a gradient at every
        # frame so a planner can navigate toward a goal (walk-to-goal families).
        marker = int(relation.params[0])
        cm = _sprite_centroids(f, mc)
        ck = _sprite_centroids(f, marker)
        if not cm or not ck:
            return -300.0
        return -min(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in cm for b in ck)
    if relation.relation_type == "cover_markers":
        marker = int(relation.params[0])
        marker_mask = (f == marker)
        if marker_mask.sum() == 0:
            return 0.0
        mov_mask = (f == mc)
        if _ndi is not None:
            mov_dil = _ndi.binary_dilation(mov_mask, iterations=1)
        else:
            mov_dil = _binary_dilation4(mov_mask)
        return float((marker_mask & mov_dil).sum())
    if relation.relation_type == "reflect_axis":
        a = int(relation.params[0])
        h, w = f.shape[:2]
        mov = (f == mc)
        cnt = 0
        for x in range(max(0, 2 * a - w + 1), min(w, 2 * a)):
            mx = 2 * a - x
            if 0 <= mx < w:
                cnt += int((mov[:, x] & (f[:, mx] == mc)).sum())
        return float(cnt)
    if relation.relation_type == "aligned_coverage":
        # ALIGNMENT coverage (P18): a marker cell is "covered" iff a movable-colour cell
        # shares its ROW or COLUMN (line-of-sight / cross-paint coverage, e.g. re86 where a
        # +-shaped piece paints a full row+column). Distinct from cover_markers (proximity
        # overlap) and from approach_marker (centroid distance). Higher = more covered.
        marker = int(relation.params[0])
        tgt_ys, tgt_xs = np.where(f == marker)
        if tgt_xs.size == 0:
            return 0.0
        mov_ys, mov_xs = np.where(f == mc)
        if mov_xs.size == 0:
            return -float(tgt_xs.size)
        mov_rows = set(int(y) for y in mov_ys)
        mov_cols = set(int(x) for x in mov_xs)
        covered = sum(1 for x, y in zip(tgt_xs.tolist(), tgt_ys.tolist())
                      if int(y) in mov_rows or int(x) in mov_cols)
        return float(covered)
    if relation.relation_type == "aligned_coverage_global":
        # COMPONENT-LEVEL, MULTI-COLOUR same-colour coverage (P19, the C2 step): for each
        # non-bg colour with >=2 components, treat the LARGEST component as the coverer
        # (e.g. a +-cross) and the rest as markers; a marker is covered iff it shares a row
        # or column with a coverer cell. Objective = total markers covered across ALL
        # colours. Captures re86's win (each colour's cross paints its same-colour markers'
        # rows/cols) WITHOUT a per-colour param, using a frame-only size heuristic to
        # distinguish coverer (big) from markers (small) within a colour (the same-colour
        # degeneracy C2). Higher = more covered.
        bg = _bg(f)
        total = 0
        for c in np.unique(f):
            c = int(c)
            if c == bg:
                continue
            if _ndi is not None:
                lab, n = _ndi.label(f == c)
                comps = [(int((lab == i).sum()), lab == i) for i in range(1, n + 1)]
            else:
                comps = [(int(mask.sum()), mask) for mask in _connected_component_masks(f == c)]
            if len(comps) < 2:
                continue
            sizes = sorted(comps, key=lambda item: item[0], reverse=True)
            cover_mask = sizes[0][1]
            cy, cx = np.where(cover_mask)
            crows = set(int(y) for y in cy)
            ccols = set(int(x) for x in cx)
            for _s, marker_mask in sizes[1:]:
                mys, mxs = np.where(marker_mask)
                if any(int(y) in crows or int(x) in ccols
                       for x, y in zip(mxs.tolist(), mys.tolist())):
                    total += 1
        return float(total)
    return 0.0


def relation_series(frames: list[np.ndarray], relations: list[TargetRelation]) -> list[dict]:
    """Evaluate all candidate relations over a frame sequence -> per-frame dict
    {relation.key: value}, ready to feed score_closure_learner."""
    return [{r.key(): evaluate_relation(f, r) for r in relations} for f in frames]
