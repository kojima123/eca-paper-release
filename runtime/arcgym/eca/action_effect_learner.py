"""ActionEffectLearner: observed action-semantics operators (E-1, S1).

Design: docs/e1_execution_vocabulary_design_2026-06-11.md (review branch).

D-1 closed the prior layer (which colour class is the win goal) and the
directed-exploration layer (walking toward it); its verified residual is
the EXECUTION vocabulary — what to DO at the goal when movement alone
does not convert approach into score (dc22 click-open, wa30 attach,
tu93 stride lattice). This module is the S1 slice: the E1 CLICK-EFFECT
family, learned with the Phase-L protected-validation pattern
(TileEffectLearner is the reference implementation of the loop:
observe -> hypothesise -> validate -> admit -> demote on falsification).

Two-tier fitness contract (E-1 design §4 — DO NOT MIX):
  * operator life/death here = effect REPRODUCTION only (dense,
    score-free; avoids the K-series flat-reward trap);
  * operator selection elsewhere = win-relation fulfilment (D-1 prior),
    never consulted by this module.

Signature discipline (gate E-d): signatures are colour-CLASS-relative.
The colour value inside a signature is a per-episode observation (the
learner is reset per episode and relearns), never a literal baked into
code. Structural features (bbox bucket, uniqueness) carry the
cross-palette weight.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Protected-validation thresholds (Phase-L pattern).
ADMIT_REPRODUCTIONS = 2          # reproductions required to admit
DEMOTE_PENALTY_REPRODUCTIONS = 2 # extra reproductions after a falsification
SELF_CHANGE_RATE_MAX = 0.5       # class changing this often WITHOUT clicks
                                 # is animation/self-moving -> never admit
PASSIVE_WINDOW = 32              # passive steps tracked for self-change rate

# Click-effect classes (E1 family).
EFFECT_VANISH = "vanish"         # clicked component disappeared (open/destroy)
EFFECT_MORPH = "morph"           # clicked component became another class (toggle)
EFFECT_REMOTE = "remote"         # clicked component unchanged, other class changed
EFFECT_NONE = "none"             # nothing changed (falsification evidence)

_BBOX_BUCKETS = (2, 4, 8, 16, 64)


def _bbox_bucket(size: int) -> int:
    for bucket in _BBOX_BUCKETS:
        if size <= bucket:
            return bucket
    return _BBOX_BUCKETS[-1]


@dataclass(frozen=True)
class SiteSignature:
    """Colour-class-relative signature of a clicked site.

    `class_value` is a per-episode observation (reset each episode), the
    rest is structural — the cross-palette content (gate E-d).
    """
    class_value: int
    bbox_bucket: tuple[int, int]
    unique_on_frame: bool


@dataclass
class ClickOperator:
    signature: SiteSignature
    effect_class: str
    reproductions: int = 0
    falsifications: int = 0
    required: int = ADMIT_REPRODUCTIONS
    sites_seen: set = field(default_factory=set)

    @property
    def admitted(self) -> bool:
        return self.reproductions >= self.required and self.falsifications == 0

    def reproduce(self, site: tuple[int, int]) -> None:
        self.reproductions += 1
        self.sites_seen.add(site)

    def falsify(self) -> None:
        # One falsification demotes; re-admission needs extra evidence
        # (required is raised past the current reproduction count, and
        # the strike clears only once that new bar is reached — re_arm).
        self.falsifications += 1
        self.required = self.reproductions + DEMOTE_PENALTY_REPRODUCTIONS

    def re_arm(self) -> None:
        """Clear the falsification strike once enough NEW reproductions
        arrived (required was raised by falsify)."""
        if self.falsifications and self.reproductions >= self.required:
            self.falsifications = 0


def _component_at(arr: np.ndarray, x: int, y: int) -> tuple[int, set] | None:
    """(class_value, cell set) of the 4-connected component at (x, y)."""
    h, w = arr.shape
    if not (0 <= x < w and 0 <= y < h):
        return None
    value = int(arr[y, x])
    seen = {(x, y)}
    queue = deque([(x, y)])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen \
                    and int(arr[ny, nx]) == value:
                seen.add((nx, ny))
                queue.append((nx, ny))
        if len(seen) > 4096:     # degenerate sheet; bail (background click)
            break
    return value, seen


class ActionEffectLearner:
    """E1 click-effect operator induction (solver-agnostic, frame-only)."""

    def __init__(self) -> None:
        self._operators: dict[tuple[SiteSignature, str], ClickOperator] = {}
        # passive per-class change counters (self-moving exclusion)
        self._passive_steps = 0
        self._passive_changes: Counter[int] = Counter()

    # -- observation ---------------------------------------------------------

    def observe(
        self,
        action_name: str,
        action_payload: dict[str, Any] | None,
        pre_frame: Any,
        post_frame: Any,
    ) -> None:
        pre = np.asarray(pre_frame)
        post = np.asarray(post_frame)
        if pre.ndim != 2 or post.shape != pre.shape:
            return
        name = str(action_name or "").strip().upper()
        if name != "ACTION6":
            self._observe_passive(pre, post)
            return
        payload = action_payload or {}
        try:
            x, y = int(payload.get("x")), int(payload.get("y"))
        except (TypeError, ValueError):
            return
        self._observe_click(x, y, pre, post)

    def _observe_passive(self, pre: np.ndarray, post: np.ndarray) -> None:
        """Track per-class change rate on NON-click steps: a class that
        changes on its own is animation, never click-effect evidence
        (the cn04/m0r0 false-count_drop lesson applied to operators)."""
        self._passive_steps = min(self._passive_steps + 1, PASSIVE_WINDOW)
        diff = pre != post
        if not bool(diff.any()):
            return
        for value in np.unique(pre[diff]):
            self._passive_changes[int(value)] += 1

    def _self_changing(self, class_value: int) -> bool:
        if self._passive_steps < 4:
            return False
        rate = self._passive_changes.get(class_value, 0) / self._passive_steps
        return rate >= SELF_CHANGE_RATE_MAX

    def _observe_click(
        self, x: int, y: int, pre: np.ndarray, post: np.ndarray
    ) -> None:
        comp = _component_at(pre, x, y)
        if comp is None:
            return
        class_value, cells = comp
        # Background sheets are not sites (degenerate component).
        if len(cells) > 4096:
            return
        n_class_components = self._count_components(pre, class_value)
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        signature = SiteSignature(
            class_value=class_value,
            bbox_bucket=(
                _bbox_bucket(max(xs) - min(xs) + 1),
                _bbox_bucket(max(ys) - min(ys) + 1),
            ),
            unique_on_frame=(n_class_components == 1),
        )
        effect, changed_classes = self._classify_effect(class_value, cells, pre, post)
        if effect == EFFECT_REMOTE and changed_classes and all(
            self._self_changing(c) for c in changed_classes
        ):
            # Every remotely-changed class is animation: the click did
            # nothing attributable — treat as a falsifying no-op.
            effect = EFFECT_NONE
        if effect == EFFECT_NONE:
            # Falsifies every operator on this signature (the click had
            # its chance and produced nothing).
            for (sig, _eff), op in self._operators.items():
                if sig == signature:
                    op.falsify()
            return
        if self._self_changing(class_value):
            return  # animation class: never operator evidence
        key = (signature, effect)
        op = self._operators.get(key)
        if op is None:
            op = ClickOperator(signature=signature, effect_class=effect)
            self._operators[key] = op
        op.reproduce((x, y))
        op.re_arm()

    @staticmethod
    def _count_components(arr: np.ndarray, value: int) -> int:
        mask = arr == value
        seen = np.zeros_like(mask, dtype=bool)
        h, w = mask.shape
        count = 0
        ys, xs = np.where(mask)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if seen[y0, x0]:
                continue
            count += 1
            queue = deque([(x0, y0)])
            seen[y0, x0] = True
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] \
                            and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((nx, ny))
        return count

    @staticmethod
    def _classify_effect(
        class_value: int, cells: set, pre: np.ndarray, post: np.ndarray
    ) -> tuple[str, set]:
        """(effect, remotely-changed classes) for the CLICKED component —
        animation elsewhere must not be credited to the click; the caller
        cross-checks remote classes against the self-changing tracker."""
        post_values = Counter(int(post[y, x]) for x, y in cells)
        unchanged = post_values.get(class_value, 0)
        if unchanged == len(cells):
            # Clicked component intact; any other class changed?
            diff = pre != post
            if bool(diff.any()):
                changed = {int(v) for v in np.unique(pre[diff])}
                return EFFECT_REMOTE, changed
            return EFFECT_NONE, set()
        if unchanged == 0:
            replacement = post_values.most_common(1)[0]
            # All cells became one other class -> morph; mixed -> vanish.
            if replacement[1] == len(cells):
                return EFFECT_MORPH, set()
            return EFFECT_VANISH, set()
        # Partially changed: treat as morph evidence (toggle-with-anim).
        return EFFECT_MORPH, set()

    # -- queries (EffectModelBus face) ----------------------------------------

    def admitted_operators(self) -> list[ClickOperator]:
        # The self-changing exclusion is re-checked at QUERY time, not
        # only at observation time: clicks issued before the passive
        # tracker warms up (< 4 passive steps) can admit an animation
        # class (e.g. a per-frame checkerboard flicker); later passive
        # evidence must retroactively disqualify those operators.
        return [
            op
            for op in self._operators.values()
            if op.admitted and not self._self_changing(op.signature.class_value)
        ]

    def click_effect_sites(self, frame: Any) -> dict[tuple[int, int], str]:
        """Admitted-operator sites present on the current frame: for each
        admitted signature, the centroid of each matching component."""
        arr = np.asarray(frame)
        if arr.ndim != 2:
            return {}
        out: dict[tuple[int, int], str] = {}
        admitted = self.admitted_operators()
        if not admitted:
            return {}
        wanted = {op.signature.class_value: op for op in admitted}
        for value, op in wanted.items():
            mask = arr == value
            if not bool(mask.any()):
                continue
            seen = np.zeros_like(mask, dtype=bool)
            h, w = mask.shape
            ys, xs = np.where(mask)
            for y0, x0 in zip(ys.tolist(), xs.tolist()):
                if seen[y0, x0]:
                    continue
                queue = deque([(x0, y0)])
                seen[y0, x0] = True
                comp = [(x0, y0)]
                while queue:
                    cx, cy = queue.popleft()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] \
                                and not seen[ny, nx]:
                            seen[ny, nx] = True
                            comp.append((nx, ny))
                            queue.append((nx, ny))
                bw = max(c[0] for c in comp) - min(c[0] for c in comp) + 1
                bh = max(c[1] for c in comp) - min(c[1] for c in comp) + 1
                if (_bbox_bucket(bw), _bbox_bucket(bh)) != op.signature.bbox_bucket:
                    continue
                cx = int(round(sum(c[0] for c in comp) / len(comp)))
                cy = int(round(sum(c[1] for c in comp) / len(comp)))
                out[(cx, cy)] = op.effect_class
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "operator_count": len(self._operators),
            "admitted_count": len(self.admitted_operators()),
            "admitted": [
                {
                    "class": op.signature.class_value,
                    "bbox": list(op.signature.bbox_bucket),
                    "unique": op.signature.unique_on_frame,
                    "effect": op.effect_class,
                    "repro": op.reproductions,
                }
                for op in self.admitted_operators()
            ],
        }

    def reset(self) -> None:
        self._operators.clear()
        self._passive_steps = 0
        self._passive_changes.clear()


__all__ = [
    "ADMIT_REPRODUCTIONS",
    "ActionEffectLearner",
    "ClickOperator",
    "EFFECT_MORPH",
    "EFFECT_NONE",
    "EFFECT_REMOTE",
    "EFFECT_VANISH",
    "SiteSignature",
]
