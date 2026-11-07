"""Click-effect algebra: exact solving for board-EDITING games.

The exploration organs walk games well but the click-driven family
edits the BOARD (toggle/cycle mechanics), where search over 2^n
patterns is hopeless and the winning tool is algebra: learn the
effect structure from observed clicks, detect linearity over a value
cycle, and either solve exactly by elimination or Gray-walk the
whole pattern space when it is small.

The ft09 dissection taught the load-bearing lesson: a click's pixel
diff mixes the GAME state change (9<->8 toggling, reversible) with
INDICATOR drift (a click-budget bar depleting 12->11 on different
cells every time, monotone). One raw diff cannot tell them apart --
but repetition can:

  confirmed stencil   the INTERSECTION of changed-cell sets across
                      >=2 observations of the SAME click target;
                      indicator cells change once each and drop out,
                      the persistent game cells remain
  cycle               the value-successor map restricted to
                      confirmed cells; certification requires it to
                      close into a single cycle with no conflicts
  solve               per-cell needed advance d(c) over Z_k by
                      Gaussian elimination (exact for prime k)
  gray_programs       for small certified systems the goal needs no
                      hypothesis at all: enumerate every click
                      pattern in Gray order so consecutive patterns
                      differ by ONE extra click

Data-only: the solver returns click programs as data; nothing here
executes anything.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class ClickEffectLearner:
    """Learn persistent click stencils and the value cycle."""

    def __init__(self) -> None:
        # click target -> list of {cell: (v0, v1)} observations
        self.history: dict[
            tuple[int, int],
            list[dict[tuple[int, int], tuple[int, int]]],
        ] = {}
        # single-observation targets accepted via class evidence
        self.accepted: dict[
            tuple[int, int], set[tuple[int, int]]
        ] = {}
        self.conflicts = 0
        self.observations = 0

    # ---- learning ---------------------------------------------------
    def observe(
        self, before: Any, after: Any, x: int, y: int
    ) -> dict[str, Any]:
        b = np.asarray(before)
        a = np.asarray(after)
        if b.shape != a.shape or b.ndim != 2:
            return {"learned": None, "reason": "bad_frames"}
        self.observations += 1
        changed = np.argwhere(b != a)
        if not len(changed):
            return {"learned": "no_effect"}
        transitions: dict[
            tuple[int, int], tuple[int, int]
        ] = {}
        for cy, cx in changed:
            transitions[(int(cx), int(cy))] = (
                int(b[cy, cx]),
                int(a[cy, cx]),
            )
        key = (int(x), int(y))
        self.history.setdefault(key, []).append(transitions)
        stencil = self._confirmed_stencil(key)
        if stencil is not None and not stencil:
            # repeated clicks share NO persistent cell: the target
            # has no stable effect under this model
            self.conflicts += 1
            return {"learned": None, "reason": "stencil_conflict"}
        return {
            "learned": "stencil",
            "cells": len(transitions),
            "confirmed": stencil is not None,
        }

    # ---- model ------------------------------------------------------
    def _confirmed_stencil(
        self, key: tuple[int, int]
    ) -> set[tuple[int, int]] | None:
        """Cells changing in EVERY observation of key; None if <2."""
        observations = self.history.get(key) or []
        if len(observations) < 2:
            return None
        cells = set(observations[0])
        for entry in observations[1:]:
            cells &= set(entry)
        return cells

    def confirmed_targets(
        self,
    ) -> dict[tuple[int, int], set[tuple[int, int]]]:
        out = {}
        for key in self.history:
            stencil = self._confirmed_stencil(key)
            if stencil:
                out[key] = stencil
            elif stencil is None and key in self.accepted:
                out[key] = set(self.accepted[key])
        return out

    def accept_with_filter(
        self,
        key: tuple[int, int],
        exclude_cells: set[tuple[int, int]],
    ) -> set[tuple[int, int]] | None:
        """Promote a single-observation target via class evidence.

        Repetition is the gold standard for separating game state
        from indicator drift, but it costs a click per target. Once
        a couple of CONFIRMED targets have identified the indicator
        region, the remaining targets can be accepted from ONE
        observation by subtracting that region — weaker evidence,
        explicitly caller-owned. Returns the accepted stencil.
        """
        entries = self.history.get(key) or []
        if len(entries) != 1:
            return None
        stencil = {
            cell
            for cell in entries[0]
            if cell not in exclude_cells
        }
        if not stencil:
            return None
        self.accepted[key] = stencil
        return stencil

    def accept_by_congruence(
        self,
        key: tuple[int, int],
        reference: tuple[int, int],
    ) -> set[tuple[int, int]] | None:
        """Give an UNCLICKED target the reference's stencil, shifted.

        Boards in this family are lattices of congruent elements, so
        a stencil learned at one element predicts its siblings up to
        translation. Clicking every element is not merely wasteful:
        tn36's click BUDGET (~61, one indicator cell per click) is
        smaller than its element count (75), so enumeration cannot
        finish. Assuming congruence and letting the solve's
        achieved-pattern falsifier catch a wrong assumption is the
        only affordable route.
        """
        base = self.confirmed_targets().get(reference)
        if not base:
            return None
        dx = int(key[0]) - int(reference[0])
        dy = int(key[1]) - int(reference[1])
        stencil = {
            (cell[0] + dx, cell[1] + dy) for cell in base
        }
        self.accepted[key] = stencil
        # a congruent sibling inherits the reference's value
        # transitions too, so the per-cell chains stay populated
        entries = self.history.get(reference) or []
        if entries:
            shifted = []
            for entry in entries:
                shifted.append(
                    {
                        (c[0] + dx, c[1] + dy): v
                        for c, v in entry.items()
                        if (c[0] + dx, c[1] + dy) in stencil
                    }
                )
            self.history.setdefault(key, []).extend(shifted)
        return stencil

    def indicator_cells(self) -> set[tuple[int, int]]:
        """Cells seen changing that persistence rejected."""
        out: set[tuple[int, int]] = set()
        for key in self.history:
            stencil = self._confirmed_stencil(key)
            if stencil is None:
                continue
            for entry in self.history[key]:
                out.update(
                    cell
                    for cell in entry
                    if cell not in stencil
                )
        return out

    def indicator_exclusion_region(
        self, board_width: int, board_height: int
    ) -> set[tuple[int, int]]:
        """Exclusion region for class acceptance.

        Indicators DRIFT (a budget bar advances into cells never
        seen changing), so the raw rejected cells under-cover. A
        bar is elongated: extend along its long axis across the
        whole board — the full row band for a horizontal bar, the
        full column band for a vertical one, the bbox otherwise.
        """
        cells = self.indicator_cells()
        if not cells:
            return set()
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        if width >= height:
            return {
                (x, y)
                for x in range(board_width)
                for y in range(y0, y1 + 1)
            }
        return {
            (x, y)
            for x in range(x0, x1 + 1)
            for y in range(board_height)
        }

    def unconfirmed_targets(self) -> list[tuple[int, int]]:
        return [
            key
            for key, entries in self.history.items()
            if len(entries) == 1
        ]

    def _successor(self) -> dict[int, int] | None:
        """Value successors over confirmed cells; None on conflict."""
        successor: dict[int, int] = {}
        for key, stencil in self.confirmed_targets().items():
            for entry in self.history[key]:
                for cell in stencil:
                    if cell not in entry:
                        continue
                    v0, v1 = entry[cell]
                    prior = successor.get(v0)
                    if prior is None:
                        successor[v0] = v1
                    elif prior != v1:
                        return None
        return successor

    # ---- per-cell cycles (family A: glyph-rendered alphabets) ------
    def _cell_successors(
        self,
    ) -> dict[tuple[int, int], dict[int, int]] | None:
        """Per-CELL value successors; None on a per-cell conflict.

        Glyph-rendered games (ft09's tn36/sc25 neighbours) cycle a
        cell through an op alphabet whose RENDERING differs pixel by
        pixel: op1->op2 sends one pixel 3->0 and another 5->3, so
        the GLOBAL successor map conflicts by construction while
        every individual cell's map is perfectly consistent.
        """
        out: dict[tuple[int, int], dict[int, int]] = {}
        for key, stencil in self.confirmed_targets().items():
            for entry in self.history[key]:
                for cell in stencil:
                    if cell not in entry:
                        continue
                    v0, v1 = entry[cell]
                    per = out.setdefault(cell, {})
                    prior = per.get(v0)
                    if prior is None:
                        per[v0] = v1
                    elif prior != v1:
                        return None
        return out

    @staticmethod
    def _walk_cycle(
        successor: dict[int, int]
    ) -> list[int] | None:
        if not successor:
            return None
        start = min(successor)
        seen = [start]
        current = start
        for _ in range(len(successor) + 1):
            nxt = successor.get(current)
            if nxt is None:
                return None
            if nxt == start:
                return seen
            if nxt in seen:
                return None
            seen.append(nxt)
            current = nxt
        return None

    def cell_cycles(
        self,
    ) -> dict[tuple[int, int], list[int]] | None:
        """Closed per-cell cycles for every confirmed cell, or None.

        The strict form (every cell closed). Kept for callers that
        need full rings; certification and solving use the cheaper
        chain form below (closing EVERY cell costs targets x k
        clicks, closing ONE target's cells costs k)."""
        successors = self._cell_successors()
        if successors is None:
            return None
        cycles: dict[tuple[int, int], list[int]] = {}
        needed = set()
        for stencil in self.confirmed_targets().values():
            needed.update(stencil)
        for cell in needed:
            per = successors.get(cell)
            cycle = self._walk_cycle(per or {})
            if cycle is None:
                return None
            cycles[cell] = cycle
        return cycles or None

    def cell_chains(
        self,
    ) -> dict[tuple[int, int], list[int]] | None:
        """Per-cell value CHAINS (closed rings or open paths).

        None only on a genuine per-cell conflict. An open chain is
        the consistent linear order of the values a cell was seen
        stepping through; with the alphabet size k known (from any
        CLOSED cell), chain positions mod k are enough to solve."""
        successors = self._cell_successors()
        if successors is None:
            return None
        chains: dict[tuple[int, int], list[int]] = {}
        needed = set()
        for stencil in self.confirmed_targets().values():
            needed.update(stencil)
        for cell in needed:
            per = successors.get(cell) or {}
            if not per:
                continue
            closed = self._walk_cycle(per)
            if closed is not None:
                chains[cell] = closed
                continue
            starts = [
                v
                for v in per
                if v not in set(per.values())
            ]
            if len(starts) != 1:
                return None
            chain = [starts[0]]
            current = starts[0]
            for _ in range(len(per)):
                nxt = per.get(current)
                if nxt is None:
                    break
                if nxt in chain:
                    return None
                chain.append(nxt)
                current = nxt
            chains[cell] = chain
        return chains or None

    def _closed_k(self) -> int | None:
        """Alphabet size from the cells whose rings CLOSED."""
        successors = self._cell_successors()
        if successors is None:
            return None
        lengths = set()
        for per in successors.values():
            closed = self._walk_cycle(per)
            if closed is not None:
                lengths.add(len(closed))
        if len(lengths) != 1:
            return None
        return lengths.pop()

    def value_cycle_over_cells(
        self, cells
    ) -> list[int] | None:
        """A closed value cycle restricted to the given cells.

        Family-A games break the global map on glyph interiors while
        the block ANCHOR pixels may still share one clean cycle --
        the block-level vocabulary the goal induction needs."""
        wanted = {(int(c[0]), int(c[1])) for c in cells}
        successor: dict[int, int] = {}
        for key, stencil in self.confirmed_targets().items():
            for entry in self.history[key]:
                for cell in stencil:
                    if cell not in entry or cell not in wanted:
                        continue
                    v0, v1 = entry[cell]
                    prior = successor.get(v0)
                    if prior is None:
                        successor[v0] = v1
                    elif prior != v1:
                        return None
        return self._walk_cycle(successor)

    def common_cycle_length(self) -> int | None:
        """The shared per-cell alphabet size k, or None.

        Requires: no per-cell conflicts, at least one cell's ring
        CLOSED (establishing k), and every open chain short enough
        to fit the alphabet (len <= k)."""
        k = self._closed_k()
        if k is None:
            return None
        chains = self.cell_chains()
        if chains is None:
            return None
        for chain in chains.values():
            if len(chain) > k:
                return None
        return k

    # ---- detection --------------------------------------------------
    def cycle(self) -> list[int] | None:
        """The value cycle, if confirmed successors close into one."""
        successor = self._successor()
        if not successor:
            return None
        start = min(successor)
        seen = [start]
        current = start
        for _ in range(len(successor) + 1):
            nxt = successor.get(current)
            if nxt is None:
                return None
            if nxt == start:
                return seen
            if nxt in seen:
                return None
            seen.append(nxt)
            current = nxt
        return None

    def linear_certified(self) -> bool:
        if len(self.confirmed_targets()) < 2:
            return False
        if (
            self._successor() is not None
            and self.cycle() is not None
        ):
            return True
        # family-A path: per-cell cycles with a shared length
        return self.common_cycle_length() is not None

    # ---- goal hypotheses -------------------------------------------
    def editable_cells(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for stencil in self.confirmed_targets().values():
            cells.update(stencil)
        return cells

    def target_hypotheses(
        self, board: Any
    ) -> list[dict[str, Any]]:
        """Uniform-value targets over the editable region."""
        cycle = self.cycle()
        if not cycle:
            return []
        cells = sorted(self.editable_cells())
        if not cells:
            return []
        return [
            {"kind": "uniform", "value": value, "cells": cells}
            for value in cycle
        ]

    # ---- solving ----------------------------------------------------
    def _steps_needed(
        self, current: int, target: int
    ) -> int | None:
        cycle = self.cycle()
        if not cycle:
            return None
        if current not in cycle or target not in cycle:
            return None
        k = len(cycle)
        return (
            cycle.index(target) - cycle.index(current)
        ) % k

    def solve(
        self, board: Any, hypothesis: dict[str, Any]
    ) -> list[dict[str, int]] | None:
        """Click program reaching the hypothesis, or None.

        Gaussian elimination over Z_k; exact when k is prime (the
        k=2 lights-out case included). Composite k is refused
        honestly rather than solved wrongly. The hypothesis gives
        either one "value" for every cell (uniform) or a per-cell
        "values" dict (pattern — the template-induced goal family).
        """
        cycle = self.cycle()
        cell_cycles = None
        if cycle:
            k = len(cycle)
        else:
            k = self.common_cycle_length()
            cell_cycles = self.cell_chains()
            if k is None or cell_cycles is None:
                return None
        if k > 1 and k not in (2, 3, 5, 7, 11, 13):
            return None
        b = np.asarray(board)
        stencils = self.confirmed_targets()
        targets = sorted(stencils)
        cells = sorted(hypothesis.get("cells") or [])
        if not targets or not cells:
            return None
        values = hypothesis.get("values") or {}
        rows = len(cells)
        cols = len(targets)
        matrix = np.zeros((rows, cols + 1), dtype=int)
        for i, cell in enumerate(cells):
            for j, target in enumerate(targets):
                matrix[i, j] = (
                    1 if cell in stencils[target] else 0
                )
            x, y = cell
            if not (
                0 <= y < b.shape[0] and 0 <= x < b.shape[1]
            ):
                return None
            wanted = values.get(
                cell, hypothesis.get("value")
            )
            if wanted is None:
                return None
            if cycle:
                need = self._steps_needed(
                    int(b[y, x]), int(wanted)
                )
            else:
                per = cell_cycles.get(cell)
                if per is None:
                    return None
                current = int(b[y, x])
                want_v = int(wanted)
                if current not in per or want_v not in per:
                    return None
                need = (
                    per.index(want_v) - per.index(current)
                ) % k
            if need is None:
                return None
            matrix[i, cols] = need
        solution = _solve_mod(matrix, k)
        if solution is None:
            return None
        program = []
        for j, target in enumerate(targets):
            for _ in range(int(solution[j]) % k):
                program.append(
                    {"x": target[0], "y": target[1]}
                )
        return program

    def gray_programs(
        self, max_targets: int = 8
    ) -> list[dict[str, int]] | None:
        """One click per step visiting EVERY k=2 pattern once.

        For a certified toggle system with n confirmed targets the
        goal needs no hypothesis: executing the returned clicks in
        order walks all 2^n reachable patterns (standard reflected
        Gray code), so ANY win state among them is hit. Caller
        checks score after each click and stops.
        """
        cycle = self.cycle()
        k = (
            len(cycle)
            if cycle
            else self.common_cycle_length()
        )
        if k != 2:
            return None
        targets = sorted(self.confirmed_targets())
        n = len(targets)
        if not 1 <= n <= max_targets:
            return None
        program = []
        for step in range(1, 2**n):
            bit = (step & -step).bit_length() - 1
            program.append(
                {"x": targets[bit][0], "y": targets[bit][1]}
            )
        return program

    def report(self) -> dict[str, Any]:
        return {
            "observations": self.observations,
            "targets": len(self.history),
            "confirmed": len(self.confirmed_targets()),
            "cycle": self.cycle(),
            "cell_k": self.common_cycle_length(),
            "conflicts": self.conflicts,
            "linear_certified": self.linear_certified(),
        }


def _solve_mod(
    matrix: np.ndarray, k: int
) -> np.ndarray | None:
    """Solve A x = d (mod k) by elimination; k must be prime."""
    m = matrix.copy() % k
    rows, cols_plus = m.shape
    cols = cols_plus - 1
    pivot_row = 0
    pivots = []
    for col in range(cols):
        pivot = None
        for r in range(pivot_row, rows):
            if m[r, col] % k:
                pivot = r
                break
        if pivot is None:
            continue
        m[[pivot_row, pivot]] = m[[pivot, pivot_row]]
        inv = pow(int(m[pivot_row, col]), -1, k)
        m[pivot_row] = (m[pivot_row] * inv) % k
        for r in range(rows):
            if r != pivot_row and m[r, col] % k:
                m[r] = (
                    m[r] - m[r, col] * m[pivot_row]
                ) % k
        pivots.append(col)
        pivot_row += 1
        if pivot_row == rows:
            break
    # consistency: zero rows must have zero rhs
    for r in range(pivot_row, rows):
        if m[r, cols] % k:
            return None
    solution = np.zeros(cols, dtype=int)
    for idx, col in enumerate(pivots):
        solution[col] = m[idx, cols] % k
    return solution
