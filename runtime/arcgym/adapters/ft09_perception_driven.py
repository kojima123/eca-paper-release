"""ft09 perception-driven tier-A adapter (Phase 33).

Runtime-derived plans for L1-L6 of the ft09 colour-cycle puzzle. Each
call to ``perceive_and_plan(frame)`` re-extracts the puzzle layout
from the rendered frame, builds a constraint system from the bsT
templates, and solves it on the fly:

  - 2-colour levels (L1, L2, L3, L5, L6): GF(2) Gauss elimination over
    the click-effect matrix. Variables = click counts (mod 2) per
    clickable Hkx / NTi sprite. Right-hand side = required cycle
    parity per cell, derived from current colour vs constraint.
  - 3-colour levels (L4): no NTi sprites in the canonical levels;
    each Hkx is independent, so per-cell minimum cycle count is used.

Click semantics (recap):
  - Click on Hkx at (gx, gy) -> cycles that single Hkx centre by
    +1 mod len(gqb).
  - Click on NTi at (gx, gy) -> cycles the NTi centre AND its
    brush-masked neighbours (defined by NTi.pixels[j][i] == 6, plus
    implicit centre).

All clicks are ACTION6 with display coords ``(gx*2, gy*2)``, where
(gx, gy) is the grid-cell position the planner targets. Camera scale=2.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from arcengine import GameAction

from arcgym.perception.ft09_perception import perceive_ft09


def _click(gx: int, gy: int) -> Tuple[GameAction, dict]:
    """ACTION6 click on grid position (gx, gy). Camera scale=2."""
    return (GameAction.ACTION6, {"x": int(gx) * 2, "y": int(gy) * 2})


def _cell_position_for_offset(bst_pos: Tuple[int, int], i: int, j: int) -> Tuple[int, int]:
    """Compute the grid-position of the cell at template offset (i, j).

    bsT is a 3×3 sprite at top-left ``bst_pos = (gx, gy)`` with centre
    at grid (gx+1, gy+1). The neighbour at template offset (i, j)
    references the cell whose top-left is at::

        (bst_pos[0] + (i-1)*4, bst_pos[1] + (j-1)*4)

    matching the ``cAw = (self.blr.x + (i-1)*4, self.blr.y + (j-1)*4)``
    semantics from ft09.py::cgj.
    """
    return (bst_pos[0] + (i - 1) * 4, bst_pos[1] + (j - 1) * 4)


def _build_constraints(state: dict) -> dict:
    """Build per-cell constraint info from the perceived state.

    Returns a dict mapping cell-grid-position (gx, gy) -> {
        'kind': 'Hkx' or 'NTi',
        'sprite_idx': index into hkx[] or nti[] list,
        'current_idx': cycle index of current colour (0..len(gqb)-1),
        'allowed_idx': set[int] of allowed final cycle indices,
        'brush_mask': only for NTi (3x3 booleans),
    }.

    The 'allowed_idx' is the intersection of constraint sets imposed
    by all bsT templates touching this cell.
    """
    gqb = state["gqb"]
    n_colors = len(gqb)
    cell_index_by_pos: dict[Tuple[int, int], dict] = {}

    # Index Hkx and NTi cells by their (gx, gy) position
    for idx, h in enumerate(state["hkx"]):
        cell_index_by_pos[h["pos"]] = {
            "kind": "Hkx",
            "sprite_idx": idx,
            "current_idx": h["color_idx"],
            "allowed_idx": set(range(n_colors)),
        }
    for idx, n in enumerate(state["nti"]):
        cell_index_by_pos[n["pos"]] = {
            "kind": "NTi",
            "sprite_idx": idx,
            "current_idx": n["color_idx"],
            "allowed_idx": set(range(n_colors)),
            "brush_mask": n["brush_mask"],
        }

    # Apply each bsT's 8 neighbour constraints
    for bst in state["bst"]:
        pos = bst["pos"]
        nRq = bst["nRq"]
        template = bst["template"]
        if nRq not in gqb:
            # bsT centre colour outside palette (e.g., -1 sentinel) ->
            # no usable constraint. Skip.
            continue
        nRq_idx = gqb.index(nRq)
        for j in range(3):
            for i in range(3):
                if i == 1 and j == 1:
                    continue
                cell_pos = _cell_position_for_offset(pos, i, j)
                if cell_pos not in cell_index_by_pos:
                    continue  # no clickable cell at this offset
                cell = cell_index_by_pos[cell_pos]
                if int(template[j][i]) == 0:
                    # must == nRq -> allowed_idx restricted to {nRq_idx}
                    cell["allowed_idx"] &= {nRq_idx}
                else:
                    # must != nRq -> allowed_idx removes {nRq_idx}
                    cell["allowed_idx"] -= {nRq_idx}

    return cell_index_by_pos


def _solve_2color(
    state: dict, cell_constraints: dict
) -> List[Tuple[GameAction, dict]] | None:
    """GF(2) solver: 2-colour levels with Hkx + NTi clicks.

    Variables = click counts mod 2 per clickable sprite (Hkx then
    NTi). For each cell, the parity of total clicks affecting it
    (its own click + sum of NTi-brush clicks landing on it) must
    equal the required parity (current_idx XOR target_idx).

    Returns None if the system is inconsistent (puzzle unsolvable).
    """
    gqb = state["gqb"]
    assert len(gqb) == 2

    hkx = state["hkx"]
    nti = state["nti"]
    n_hkx = len(hkx)
    n_nti = len(nti)
    n_vars = n_hkx + n_nti

    # Build per-cell row in the GF(2) system.
    # For each clickable cell with at least one constraint, we add a row.
    # cell.allowed_idx is the set of allowed cycle indices; for 2-colour
    # we expect |allowed_idx| ∈ {0, 1, 2}. If 0 -> infeasible. If 2 ->
    # no constraint (skip). If 1 -> required idx.

    rows: list[tuple[list[int], int]] = []  # (coefficient_row, rhs)

    # Build a map: cell_pos -> list of (var_idx) whose brush click
    # cycles this cell.
    #   - Hkx brush is irw = [[0,0,0],[0,1,0],[0,0,0]] (centre only),
    #     so a Hkx click cycles only its own centre cell.
    #   - NTi brush is the perceived `brush_mask` (centre always True
    #     plus any pixel == 6 in the sprite); each True position at
    #     (i, j) cycles the cell at offset ((i-1)*4, (j-1)*4) from
    #     the NTi sprite.
    brushes_hitting_cell: dict[Tuple[int, int], list[int]] = {}
    for hkx_idx, h in enumerate(hkx):
        var_idx = hkx_idx
        # Hkx click only cycles its own cell.
        brushes_hitting_cell.setdefault(h["pos"], []).append(var_idx)
    for nti_idx, n in enumerate(nti):
        var_idx = n_hkx + nti_idx
        bx, by = n["pos"]
        brush = n["brush_mask"]
        for j in range(3):
            for i in range(3):
                if not brush[j][i]:
                    continue
                target = (bx + (i - 1) * 4, by + (j - 1) * 4)
                brushes_hitting_cell.setdefault(target, []).append(var_idx)

    for cell_pos, cell in cell_constraints.items():
        allowed = cell["allowed_idx"]
        if len(allowed) == 0:
            return None  # infeasible
        if len(allowed) >= 2:
            continue  # no constraint
        target_idx = next(iter(allowed))
        rhs = (target_idx - cell["current_idx"]) % 2
        # Coefficient row: parity of total brush hits on this cell.
        row = [0] * n_vars
        for var_idx in brushes_hitting_cell.get(cell_pos, []):
            row[var_idx] = (row[var_idx] + 1) % 2
        rows.append((row, int(rhs)))

    # Gauss elimination over GF(2)
    n_rows = len(rows)
    if n_rows == 0:
        # No constraints -> trivially solvable (no clicks needed)
        return []

    # Build augmented matrix
    mat = [list(r[0]) + [r[1]] for r in rows]
    # mat[i] has length n_vars + 1
    rank = 0
    for col in range(n_vars):
        # Find pivot row at or below `rank` with mat[r][col] == 1
        pivot = -1
        for r in range(rank, n_rows):
            if mat[r][col] == 1:
                pivot = r
                break
        if pivot < 0:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        # Eliminate other rows
        for r in range(n_rows):
            if r != rank and mat[r][col] == 1:
                for c in range(col, n_vars + 1):
                    mat[r][c] ^= mat[rank][c]
        rank += 1

    # Check consistency: any row with all-zero LHS but rhs=1 -> infeasible
    for r in range(n_rows):
        if all(mat[r][c] == 0 for c in range(n_vars)) and mat[r][n_vars] == 1:
            return None

    # Solve: free variables set to 0, pivot variables solved.
    x = [0] * n_vars
    # Find pivot column for each row
    row_pivot = [-1] * n_rows
    for r in range(n_rows):
        for c in range(n_vars):
            if mat[r][c] == 1:
                row_pivot[r] = c
                break
    for r in range(n_rows):
        if row_pivot[r] < 0:
            continue
        # x[pivot] = rhs (since other pivot columns are 0 here, but
        # free columns may have 1s. With free=0, x[pivot] = rhs.)
        x[row_pivot[r]] = mat[r][n_vars]

    # Generate plan: for each variable with x[var]==1, click once.
    plan: list[tuple] = []
    for idx in range(n_hkx):
        if x[idx] == 1:
            gx, gy = hkx[idx]["pos"]
            plan.append(_click(gx, gy))
    for idx in range(n_nti):
        if x[n_hkx + idx] == 1:
            gx, gy = nti[idx]["pos"]
            plan.append(_click(gx, gy))
    return plan


def _solve_ncolor_per_cell(
    state: dict, cell_constraints: dict
) -> List[Tuple[GameAction, dict]] | None:
    """Per-cell minimum click count solver for n-colour levels with
    no NTi sprites (each Hkx is independent).

    For each cell with constraint, find the smallest k >= 0 such that
    (current_idx + k) mod n_colors is in allowed_idx. Click that cell
    k times.

    Returns None if any cell has no allowed colour (infeasible).
    """
    gqb = state["gqb"]
    n_colors = len(gqb)
    plan: list[tuple] = []
    for cell_pos, cell in cell_constraints.items():
        allowed = cell["allowed_idx"]
        if len(allowed) == 0:
            return None
        if len(allowed) >= n_colors:
            continue  # all colours allowed -> no constraint
        cur = cell["current_idx"]
        # Find smallest k in [0, n_colors-1] with (cur+k) % n_colors in allowed
        best_k = None
        for k in range(n_colors):
            if (cur + k) % n_colors in allowed:
                best_k = k
                break
        if best_k is None:
            return None
        for _ in range(best_k):
            gx, gy = cell_pos
            plan.append(_click(gx, gy))
    return plan


def _solve_color_cycle(state: dict) -> List[Tuple[GameAction, dict]] | None:
    """Top-level dispatcher: choose 2-colour or n-colour solver."""
    gqb = state["gqb"]
    cell_constraints = _build_constraints(state)
    if len(gqb) == 2:
        return _solve_2color(state, cell_constraints)
    # 3-colour or more. ft09 canonical levels with 3 colours have no
    # NTi (L4); fall back to per-cell solver.
    if state["nti"]:
        # Mixed n-colour with NTi is not currently supported. Emit
        # an empty plan so the runner ends cleanly rather than emit
        # incorrect clicks.
        return None
    return _solve_ncolor_per_cell(state, cell_constraints)


def perceive_and_plan(frame: Any) -> List[Tuple[GameAction, dict]]:
    """Extract puzzle state from the rendered frame, solve, return click plan.

    The plan is freshly computed at every call: the perception layer
    extracts gqb / Hkx / NTi / bsT from the frame, the solver runs
    GF(2) Gauss elimination (2-colour) or per-cell minimum-cycle
    enumeration (n-colour). No state is hardcoded.

    Returns an empty list when:
      - the frame doesn't look like an ft09 puzzle (no palette or
        no clickable cells); or
      - the constraint system is infeasible.
    Both terminate the run cleanly.
    """
    state = perceive_ft09(frame)
    if not state["gqb"]:
        return []
    if not (state["hkx"] or state["nti"]):
        return []
    if not state["bst"]:
        return []
    plan = _solve_color_cycle(state)
    if plan is None:
        return []
    return plan
