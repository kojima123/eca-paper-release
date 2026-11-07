"""Template-goal induction: pure board analysis for click puzzles.

The induction core behind the ft09 FULL WIN (0->6, 2026-08-02),
extracted from the bootstrap probe so the interactive drivers (the
offline probe, the recommendation-only executive) share one
implementation. Everything here is PURE over (board, confirmed
targets, value cycle): no environment access, no clicking, no
randomness.

The measured goal-encoding family:

  arrangement   confirmed click targets form a lattice; non-target
                lattice cells with NON-UNIFORM windows are holes
  holes         goal SPRITES (a mini-image whose 3x3 sub-pattern
                depicts the required neighborhood centered on
                itself), DITHER cells (two-value full-checker =
                plus-brush click targets rendered dithered; the
                m0r0 checkerboard-render lesson), or legend/edge
                bleed (window contains the board background)
  mapping       a sprite's center sub-value is the block value one
                anchor ring symbol maps to; both anchor conventions
                fan out; ring symbols map injectively into the
                remaining cycle values; OVERLAP consistency between
                neighboring sprites prunes the assignment space
  goals         each consistent assignment yields one per-block
                required pattern (a "goal variant")

Drivers own all clicking, budgets, and honesty bookkeeping.
"""
from __future__ import annotations

from collections import Counter, deque
from itertools import permutations
from typing import Any

import numpy as np


def board_components(a: np.ndarray):
    """4-connected same-value components + label map."""
    H, W = a.shape
    seen = np.zeros((H, W), dtype=bool)
    label = np.full((H, W), -1, dtype=int)
    comps = []
    for sy in range(H):
        for sx in range(W):
            if seen[sy, sx]:
                continue
            v = int(a[sy, sx])
            queue = deque([(sx, sy)])
            seen[sy, sx] = True
            cells = []
            while queue:
                x, y = queue.popleft()
                cells.append((x, y))
                label[y, x] = len(comps)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < W
                        and 0 <= ny < H
                        and not seen[ny, nx]
                        and int(a[ny, nx]) == v
                    ):
                        seen[ny, nx] = True
                        queue.append((nx, ny))
            comps.append((v, cells))
    return comps, label


def component_representatives(
    a: np.ndarray, lo: int = 9, hi: int = 400
) -> list[tuple[int, int]]:
    """Centroids of same-value components sized [lo, hi] whose
    centroid pixel lies inside the component (sweep click set).

    Ordered by RISK: background-valued pockets go last. Measured on
    tn36 -- 36 of its 75 representatives are background pockets, and
    clicking one of them ENDS the episode (the corner counter drops
    and the game is over), killing the sweep at 60/75 before the
    algebra could be used. Interactive elements are drawn, so a
    background-coloured cell is rarely a control and often a trap;
    keeping them in the set (just later) preserves access without
    risking the informative prefix.
    """
    comps, _label = board_components(a)
    background = int(np.bincount(np.asarray(a).ravel()).argmax())
    reps: list[tuple[int, int]] = []
    deferred: list[tuple[int, int]] = []
    for value, cells in comps:
        if not (lo <= len(cells) <= hi):
            continue
        cx = int(round(sum(c[0] for c in cells) / len(cells)))
        cy = int(round(sum(c[1] for c in cells) / len(cells)))
        if (cx, cy) not in cells:
            continue
        if int(value) == background:
            deferred.append((cx, cy))
        else:
            reps.append((cx, cy))
    return reps + deferred


class TemplateGoalInducer:
    """Arrangement + goal induction over one board snapshot."""

    def __init__(
        self,
        board: Any,
        targets: list[tuple[int, int]],
        cycle: list[int],
    ) -> None:
        self.board = np.asarray(board)
        self.targets = sorted(
            (int(t[0]), int(t[1])) for t in targets
        )
        self.cycle = list(cycle)
        self.ok = (
            self.board.ndim == 2
            and self.board.size > 0
            and len(self.targets) >= 2
            and len(self.cycle) >= 2
        )
        if not self.ok:
            return
        self.ox = min(t[0] for t in self.targets)
        self.oy = min(t[1] for t in self.targets)
        self.offsets = sorted(
            (t[0] - self.ox, t[1] - self.oy)
            for t in self.targets
        )
        self.span_x = max(o[0] for o in self.offsets)
        self.span_y = max(o[1] for o in self.offsets)
        pos_x = sorted(
            {o[0] for o in self.offsets if o[0] > 0}
        )
        pos_y = sorted(
            {o[1] for o in self.offsets if o[1] > 0}
        )
        self.step_x = pos_x[0] if pos_x else 1
        self.step_y = pos_y[0] if pos_y else 1
        self.cols = 1 + self.span_x // self.step_x
        self.rows = 1 + self.span_y // self.step_y
        self.center = (self.span_x // 2, self.span_y // 2)
        self.comps, self.label = board_components(self.board)
        # plus-brush (dither) targets have fragment components and
        # non-cycle centroid pixels; geometry references must come
        # from SOLID block targets only
        self.solid_offsets = [
            o
            for o in self.offsets
            if int(
                self.board[self.oy + o[1], self.ox + o[0]]
            )
            in self.cycle
        ] or self.offsets
        self.lattice = [
            (c * self.step_x, r * self.step_y)
            for r in range(self.rows)
            for c in range(self.cols)
        ]
        self.target_offsets = set(self.offsets)
        self.background = int(
            np.bincount(self.board.ravel()).argmax()
        )
        self.sprite_offsets: list[tuple[int, int]] = []
        self.dither_holes: list[tuple[int, int]] = []
        self._classify_holes()

    # ---- geometry ---------------------------------------------------
    def block_box(self, x: int, y: int):
        _v, cells = self.comps[self.label[y, x]]
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        return (
            min(xs),
            min(ys),
            max(xs) - min(xs) + 1,
            max(ys) - min(ys) + 1,
        )

    def _anchor_of(self, x: int, y: int):
        _v, cells = self.comps[self.label[y, x]]
        cx = int(round(sum(c[0] for c in cells) / len(cells)))
        cy = int(round(sum(c[1] for c in cells) / len(cells)))
        return (cx, cy) if (cx, cy) in cells else None

    def _nearest_solid(self, off):
        return min(
            self.solid_offsets,
            key=lambda o: abs(o[0] - off[0])
            + abs(o[1] - off[1]),
        )

    def _window_origin(self, off):
        ref = self._nearest_solid(off)
        bx0, by0, bw, bh = self.block_box(
            self.ox + ref[0], self.oy + ref[1]
        )
        return (
            bx0 + (off[0] - ref[0]),
            by0 + (off[1] - ref[1]),
            bw,
            bh,
        )

    def window_values(self, off):
        try:
            x0, y0, bw, bh = self._window_origin(off)
        except Exception:  # noqa: BLE001
            return None
        patch = self.board[y0 : y0 + bh, x0 : x0 + bw]
        if patch.size == 0:
            return None
        return set(int(v) for v in patch.ravel())

    def sprite_neighborhood(self, off):
        """3x3 sub-pattern of the sprite cell at lattice off."""
        try:
            x0, y0, bw, bh = self._window_origin(off)
        except Exception:  # noqa: BLE001
            return None
        sub_w = max(1, bw // 3)
        sub_h = max(1, bh // 3)
        out = {}
        for sdy in (-1, 0, 1):
            for sdx in (-1, 0, 1):
                px0 = x0 + (sdx + 1) * sub_w
                py0 = y0 + (sdy + 1) * sub_h
                patch = self.board[
                    py0 : py0 + sub_h, px0 : px0 + sub_w
                ]
                if patch.size == 0:
                    return None
                vals, cnt = np.unique(
                    patch, return_counts=True
                )
                out[(sdx, sdy)] = int(vals[np.argmax(cnt)])
        return out

    @staticmethod
    def _is_dither(sub) -> bool:
        """Full 3x3 alternation of two values = a dithered special
        TILE, not a goal sprite (goal sprites break the alternation
        at their center: it shows a cycle value)."""
        values_at: dict[int, set] = {}
        for (sdx, sdy), sv in sub.items():
            values_at.setdefault(
                (sdx + sdy) % 2, set()
            ).add(sv)
        return (
            len(values_at.get(0, set())) == 1
            and len(values_at.get(1, set())) == 1
            and values_at[0] != values_at[1]
        )

    def _classify_holes(self) -> None:
        for off in self.lattice:
            if off in self.target_offsets:
                continue
            values_here = self.window_values(off)
            if not values_here or len(values_here) < 2:
                continue
            # legend / edge bleed: real sprites sit inside the
            # arrangement and never show the board background
            if self.background in values_here:
                continue
            sub = self.sprite_neighborhood(off)
            if sub and self._is_dither(sub):
                self.dither_holes.append(off)
                continue
            self.sprite_offsets.append(off)

    def frame_rings(self) -> list[tuple[int, int, int, int]]:
        """Bounding boxes of hollow FRAME components.

        Semantic units (goal displays, playfields) sit inside drawn
        rings; a ring is a large component whose bbox area exceeds
        its own cell count. Boxes that contain a clickable target
        are the playfield, not a display, and are dropped.
        """
        out: list[tuple[int, int, int, int]] = []
        for _value, cells in self.comps:
            if len(cells) < 20:
                continue
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            bx0, bx1 = min(xs), max(xs)
            by0, by1 = min(ys), max(ys)
            area = (bx1 - bx0 + 1) * (by1 - by0 + 1)
            if area < 25 or area / len(cells) < 1.2:
                continue
            if any(
                bx0 < t[0] < bx1 and by0 < t[1] < by1
                for t in self.targets
            ):
                continue
            out.append((bx0, by0, bx1, by1))
        return out

    def dither_positions(self) -> list[tuple[int, int]]:
        """Absolute click positions of the dithered plus-brushes."""
        return [
            (self.ox + off[0], self.oy + off[1])
            for off in self.dither_holes
        ]

    # ---- value reads ------------------------------------------------
    def refreshed(self, raw: Any) -> np.ndarray:
        """Board copy with dither centroids patched to their
        window's cycle singleton (solve/verify value reads)."""
        out = np.asarray(raw).copy()
        for dx, dy in self.offsets:
            x, y = self.ox + dx, self.oy + dy
            if int(out[y, x]) in self.cycle:
                continue
            try:
                x0, y0, bw, bh = self._window_origin((dx, dy))
            except Exception:  # noqa: BLE001
                continue
            patch = out[y0 : y0 + bh, x0 : x0 + bw]
            inside = set(
                int(v) for v in patch.ravel()
            ) & set(self.cycle)
            if len(inside) == 1:
                out[y, x] = inside.pop()
        return out

    # ---- example grids ---------------------------------------------
    def block_pattern(self, gx: int, gy: int):
        H, W = self.board.shape
        out = {}
        for dx, dy in self.offsets:
            x, y = gx + dx, gy + dy
            if not (0 <= x < W and 0 <= y < H):
                return None
            v = int(self.board[y, x])
            if v not in self.cycle:
                return None
            out[(dx, dy)] = v
        return out

    def _sprite_pattern(self, gx: int, gy: int):
        bx0, by0, bw, bh = self.block_box(gx, gy)
        x0 = bx0 + self.center[0]
        y0 = by0 + self.center[1]
        sub_w = max(1, bw // self.cols)
        sub_h = max(1, bh // self.rows)
        out = {}
        for r in range(self.rows):
            for c in range(self.cols):
                patch = self.board[
                    y0 + r * sub_h : y0 + (r + 1) * sub_h,
                    x0 + c * sub_w : x0 + (c + 1) * sub_w,
                ]
                if patch.size == 0:
                    return None
                vals, cnt = np.unique(
                    patch, return_counts=True
                )
                out[
                    (c * self.step_x, r * self.step_y)
                ] = int(vals[np.argmax(cnt)])
        return out

    def example_votes(self) -> Counter:
        """Congruent example grids elsewhere on the board vote a
        global sprite->block mapping (the L1 mechanism)."""
        H, W = self.board.shape
        snapped = set()
        for gy in range(H - self.span_y):
            for gx in range(W - self.span_x):
                if self.block_pattern(gx, gy) is None:
                    continue
                a = self._anchor_of(gx, gy)
                if a is not None:
                    snapped.add(a)
        votes: Counter = Counter()
        for gx, gy in sorted(snapped):
            if (gx, gy) == (self.ox, self.oy):
                continue
            blocks = self.block_pattern(gx, gy)
            sprite = (
                self._sprite_pattern(gx, gy)
                if blocks
                else None
            )
            if blocks is None or sprite is None:
                continue
            m: dict[int, int] = {}
            ok = True
            for off in self.offsets:
                sv, bv = sprite[off], blocks[off]
                if sv in m and m[sv] != bv:
                    ok = False
                    break
                m[sv] = bv
            if (
                ok
                and len(m) == len(self.cycle)
                and len(set(m.values())) == len(self.cycle)
            ):
                votes[tuple(sorted(m.items()))] += 1
        # rank correspondence, both orders, as 0-vote fallbacks
        pool = set()
        for off in self.sprite_offsets:
            sub = self.sprite_neighborhood(off)
            if sub:
                for key, sv in sub.items():
                    if key != (0, 0):
                        pool.add(sv)
        ranked = sorted(pool)
        if len(ranked) == len(self.cycle):
            votes.setdefault(
                tuple(
                    sorted(
                        zip(ranked, sorted(self.cycle))
                    )
                ),
                0,
            )
            votes.setdefault(
                tuple(
                    sorted(
                        zip(
                            ranked,
                            sorted(
                                self.cycle, reverse=True
                            ),
                        )
                    )
                ),
                0,
            )
        return votes

    # ---- goal variants ---------------------------------------------
    def goal_variants(
        self, fallback: dict[int, int]
    ) -> list[dict[tuple[int, int], int]]:
        """Per-block goal variants from the per-sprite assignment
        search with overlap-consistency pruning."""
        sprites = []
        for off in self.sprite_offsets:
            sub = self.sprite_neighborhood(off)
            if not sub:
                continue
            # only symbols that hit a clickable position need
            # images (ft09 L5's off-shape marker stays unbound)
            ring = sorted(
                {
                    sv
                    for (sdx, sdy), sv in sub.items()
                    if (sdx, sdy) != (0, 0)
                    and (
                        off[0] + sdx * self.step_x,
                        off[1] + sdy * self.step_y,
                    )
                    in self.target_offsets
                }
            )
            sprites.append(
                (off, sub, ring, sub.get((0, 0)))
            )
        variants: list[dict] = []

        def impose(required, off, sub, local):
            for (sdx, sdy), sv in sub.items():
                if (sdx, sdy) == (0, 0):
                    continue
                pos = (
                    off[0] + sdx * self.step_x,
                    off[1] + sdy * self.step_y,
                )
                if pos not in self.target_offsets:
                    continue
                want = local.get(sv)
                if want is None:
                    return False
                prior = required.get(pos)
                if prior is not None and prior != want:
                    return False
                required[pos] = want
            return True

        def assign(idx, required):
            if len(variants) >= 8:
                return
            if idx == len(sprites):
                if required:
                    variants.append(dict(required))
                return
            off, sub, ring, center_value = sprites[idx]
            if center_value in self.cycle and ring:
                rest = [
                    v
                    for v in self.cycle
                    if v != center_value
                ]
                anchors = [ring[0]]
                if ring[-1] != ring[0]:
                    # the center-key convention can flip per
                    # level; overlap consistency prunes
                    anchors.append(ring[-1])
                for anchor in anchors:
                    others = [
                        s for s in ring if s != anchor
                    ]
                    for perm in permutations(
                        rest, len(others)
                    ):
                        local = {anchor: center_value}
                        local.update(
                            dict(zip(others, perm))
                        )
                        trial = dict(required)
                        if impose(trial, off, sub, local):
                            assign(idx + 1, trial)
            else:
                trial = dict(required)
                if impose(trial, off, sub, fallback):
                    assign(idx + 1, trial)

        assign(0, {})
        return variants

    def absolute(
        self, required: dict[tuple[int, int], int]
    ) -> dict[tuple[int, int], int]:
        """Offset-keyed required pattern -> absolute positions."""
        return {
            (self.ox + dx, self.oy + dy): v
            for (dx, dy), v in required.items()
        }
