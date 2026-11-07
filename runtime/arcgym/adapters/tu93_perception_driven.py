"""tu93 maze adapter, frame-perception-driven (Phase 11 + Phase 35).

Mirror of ``arcgym.adapters.tu93_maze`` but operating on perceived
state from `arcgym.perception.tu93_perception.perceive_tu93(frame)`
instead of `env._env._game.current_level.get_sprites_by_tag(...)`
queries. Enables remote ARC-AGI-3 API deployment.

Algorithm: stride=6 BFS over walkable cells (walkability check derived
from perception's color-2 set), with **trigger-cell avoidance** to
handle the L2+ lock mechanic.

Walkability semantics (extracted from `tu93.py:step` + L1 frame
inspection):
  - Each ACTION1-4 advances walker by 6 cells (auto-advance to next
    settling point).
  - Move allowed if pixel at `local + (±3 in dir)` == 2.
  - Perception's `walkable` set is exactly the world cells where
    frame == 2. Equivalent to the game's field.pixels==2 in world
    coordinates.

Lock mechanic (Phase 35 audit):
  - Each ``0001haidilggfh`` lock has a rotation in {0, 90, 180, 270}.
  - When the walker arrives at the cell **6 cells in front of the lock
    in its rotation direction** (the "trigger cell"), the lock fires
    in `tu93.py:ixnhjkzwic`, marks itself active (pixel[0,1]=11), and
    chases the walker along its facing axis. On contact the walker is
    morphed 3x3->5x5->7x7->removed via `uneirnujpq`, triggering the
    `lose()` path.
  - Avoidance is purely positional: the walker must never settle in a
    trigger cell. Perception extracts every trigger cell into
    `state["trigger_cells"]`; BFS forbids those cells.
  - The CLAUDE.md catalog claim "tu93 L2 = walker tag absent at L2"
    was wrong: walker tag (`0017unajnymcki`) IS present at L2; the L1
    adapter just stepped through the trigger cell (21, 18) and lost.

Public API:
  plan_next_actions_from_state(perceived_state) -> list of GameAction
  perceive_and_plan(frame) -> list of GameAction
"""
from __future__ import annotations

from collections import deque
from typing import List

from arcengine import GameAction

from arcgym.perception.tu93_perception import perceive_tu93

EDGE_OFFSET = 3
STRIDE = 6

DIRS = [
    (GameAction.ACTION1, 0, -STRIDE, 0, -EDGE_OFFSET),  # UP
    (GameAction.ACTION2, 0, STRIDE, 0, EDGE_OFFSET),    # DOWN
    (GameAction.ACTION3, -STRIDE, 0, -EDGE_OFFSET, 0),  # LEFT
    (GameAction.ACTION4, STRIDE, 0, EDGE_OFFSET, 0),    # RIGHT
]


def _can_move(walkable: set[tuple[int, int]],
              walker_pos: tuple[int, int],
              leading_dx: int, leading_dy: int) -> bool:
    """Move is allowed if the leading-edge cell (walker_pos + leading)
    is in the walkable set.
    """
    check = (walker_pos[0] + leading_dx, walker_pos[1] + leading_dy)
    return check in walkable


def _bfs_walker(start: tuple[int, int], goal: tuple[int, int],
                walkable: set[tuple[int, int]],
                forbidden: set[tuple[int, int]] | None = None) -> list | None:
    """BFS from start to goal over the 6-stride walker grid.

    Args:
        start: walker top-left in world coords.
        goal: target cell (an exit's top-left).
        walkable: set of world cells where field == 2.
        forbidden: set of world cells the walker must NEVER settle on
            (lock trigger cells; entering them activates a lock chase
            that ends the game).
    """
    if start == goal:
        return []
    forbidden = forbidden or set()
    if start in forbidden:
        # Walker already on a trigger; the lock would have fired by now,
        # so we can't plan from here. Caller will get None and fall back.
        return None
    visited = {start}
    queue: deque = deque([(start, [])])
    while queue:
        pos, path = queue.popleft()
        for action_id, dx, dy, leading_dx, leading_dy in DIRS:
            if not _can_move(walkable, pos, leading_dx, leading_dy):
                continue
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in visited:
                continue
            # Locks fire whenever the walker SETTLES at a trigger cell,
            # regardless of whether it's the goal — phase 1 fires the
            # lock before phase 2 checks the exit. So forbid trigger
            # cells unconditionally.
            if nxt in forbidden:
                continue
            new_path = path + [action_id]
            if nxt == goal:
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))
    return None


def plan_next_actions_from_state(perceived: dict) -> List[GameAction]:
    """Plan walker path to nearest exit using perceived state.

    Args:
        perceived: output of ``perceive_tu93(frame)`` with keys
            walker_pos, exits, walkable, trigger_cells (optional).

    Returns: list of GameAction, or [] if win or no plan.
    """
    walker_pos = perceived["walker_pos"]
    exits = perceived["exits"]
    walkable = perceived["walkable"]
    forbidden = perceived.get("trigger_cells") or set()
    if walker_pos is None or not exits:
        return []
    # Win: walker on any exit
    if walker_pos in {tuple(e) for e in exits}:
        return []
    # Plan to nearest reachable exit, avoiding lock trigger cells.
    best: list | None = None
    for ex in exits:
        ep = (ex[0], ex[1])
        if ep == walker_pos:
            continue
        path = _bfs_walker(walker_pos, ep, walkable, forbidden=forbidden)
        if path is not None and (best is None or len(path) < len(best)):
            best = path
    if best is not None:
        return best
    # No avoidance path. Returning [] tells the runner to stop rather
    # than blunder into a trigger and lose the game. Levels that have
    # only trigger-blocked paths (e.g., L3, L4) will halt cleanly here.
    return []


def perceive_and_plan(frame) -> List[GameAction]:
    state = perceive_tu93(frame)
    return plan_next_actions_from_state(state)
