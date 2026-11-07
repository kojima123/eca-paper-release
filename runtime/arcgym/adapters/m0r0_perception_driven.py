"""m0r0 perception-driven tier-A adapter (frame-only, runtime BFS).

Mechanic (mirror_x cursor pair, m0r0.py:837):
- ACTION1/2 move BOTH cursors with the same dy ("tilt").
- ACTION3/4 move cursors with MIRRORED dx (cursor A: +dx, cursor B: -dx).
- Win when |a.x - b.x| == 1 AND a.y == b.y AND the next mirrored move
  SWAPS them (a_new == b_old or b_new == a_old): they collide at the
  midpoint and become INTANGIBLE.

L2 adds ``spswjz`` death-hole cells (m0r0.py:822-826): if a cursor lands
on one after a move, the game enters a flash animation that resets all
``sys_click`` sprite positions to their level-init values. BFS treats
landing on a hole as an invalid transition (skipped successor).

Solver: BFS over joint cursor state ((a_pos, b_pos)) using the
walkable cell set and hole set extracted from the rendered frame by
``perceive_m0r0``. Each call recomputes the plan from current state,
so the adapter is fully runtime — no offline-derived hardcoded plan.
"""
from __future__ import annotations

from collections import deque
from typing import Any, List, Tuple

import numpy as np

from arcengine import GameAction
from arcgym.perception.m0r0_perception import perceive_m0r0

# (action, dx_a, dy)  — ACTION1/2 share dy on both; ACTION3/4 use ±dx.
# For ACTION1/2 (vertical tilt), dx_a = 0 (b also gets dx=0 via mirror).
# For ACTION3/4 (mirror_x), dx_a = ±1 and dx_b = -dx_a.
_DIRS: list[tuple[Any, int, int]] = [
    (GameAction.ACTION1, 0, -1),
    (GameAction.ACTION2, 0, 1),
    (GameAction.ACTION3, -1, 0),
    (GameAction.ACTION4, 1, 0),
]


def _bfs_joint_cursors(
    start_a: Tuple[int, int],
    start_b: Tuple[int, int],
    walkable: frozenset,
    holes: frozenset,
) -> Tuple[List[Any], int]:
    """BFS over joint (a, b) state. Returns (action_list, states_visited).

    Wall-blocked moves leave the cursor in place (matches `jpwxcqabja`
    semantics). Death-hole moves are skipped (treating them as invalid
    transitions; landing on a hole resets all cursors to start, which
    BFS would handle by re-visiting initial state but we prune for
    efficiency). Win check fires BEFORE the candidate step is committed:
    if current state already satisfies |dx|=1 AND same y AND the
    candidate move would swap them, the action is the winning move.
    """
    initial = (start_a, start_b)
    visited = {initial}
    queue: deque = deque([(initial, [])])
    while queue:
        (a, b), path = queue.popleft()
        for action, dx, dy in _DIRS:
            a_new = (a[0] + dx, a[1] + dy)
            b_new = (b[0] - dx, b[1] + dy)  # mirror_x: b's dx is negated
            if a_new not in walkable:
                a_new = a
            if b_new not in walkable:
                b_new = b
            # Win predicate (m0r0.py:837): pre-move |dx|=1, same y, and
            # the mirrored move SWAPS the pair.
            if abs(a[0] - b[0]) == 1 and a[1] == b[1]:
                if a_new == b or b_new == a:
                    return path + [action], len(visited)
            if a_new == a and b_new == b:
                continue  # both blocked; skip
            # Skip moves that drop a cursor on a death hole — they cause
            # a positional reset (m0r0.py:822-826), wasted progress.
            if a_new in holes or b_new in holes:
                continue
            new_state = (a_new, b_new)
            if new_state in visited:
                continue
            visited.add(new_state)
            queue.append((new_state, path + [action]))
    return [], len(visited)


def perceive_and_plan(frame: Any) -> List[Tuple[Any, dict]]:
    """Frame -> (action, data) tuple list.

    Runs perception then BFS at every call. Returns an empty list on
    levels other than L1/L2 (signature mismatch), causing the test
    runner to terminate the run cleanly.
    """
    state = perceive_m0r0(np.asarray(frame))
    if state["level_signature"] not in ("m0r0_l1", "m0r0_l2"):
        return []
    a = state["cursor_a"]
    b = state["cursor_b"]
    walkable = state["walkable"]
    holes = state.get("holes") or frozenset()
    if a is None or b is None or walkable is None:
        return []
    plan, _ = _bfs_joint_cursors(a, b, walkable, holes)
    return [(action, {}) for action in plan]
