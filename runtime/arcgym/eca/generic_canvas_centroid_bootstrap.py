"""Generic canvas-centroid first-score bootstrap.

This Tier-B wrapper targets visible centroid/canvas puzzles:

    click pieces -> their group canvas snaps to the piece centroid
    score when the canvas overlaps a matching visible target

The frame reader comes from the generic r11l perception grammar because that is
where the canvas/piece/target vocabulary was first implemented. This wrapper
adds the unknown-robust contract used by `autonomous_generic_suite`: no adapter
imports, no game-id dispatch, disposable score confirmation, and replay-only
output.
"""
from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arc_agi  # noqa: E402
from arcengine import GameAction  # noqa: E402
from arcgym.environments import ArcAgi3Env  # noqa: E402
from arcgym.perception.r11l_perception import perceive_r11l  # noqa: E402


ActionSpec = tuple[Any, ...]
PIECE_W = PIECE_H = 5
CANVAS_W = CANVAS_H = 5
TARGET_W = TARGET_H = 7
GRID_SIZE = 64


@dataclass(frozen=True)
class CanvasCentroidDiagnostics:
    gated: bool = False
    gate_reason: str = ""
    proposed_actions: int = 0
    winning_score: int = 0
    winning_actions: int = 0
    plan_source: str = ""
    absorber_continuations: int = 0

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "gated": bool(self.gated),
            "gate_reason": self.gate_reason,
            "proposed_actions": int(self.proposed_actions),
            "winning_score": int(self.winning_score),
            "winning_actions": int(self.winning_actions),
            "plan_source": self.plan_source,
            "absorber_continuations": int(self.absorber_continuations),
        }


LAST_DIAGNOSTICS = CanvasCentroidDiagnostics()


def _new_env(game: str, max_actions: int) -> ArcAgi3Env:
    return ArcAgi3Env(
        game_id=game,
        max_actions=max_actions,
        operation_mode=arc_agi.OperationMode.OFFLINE,
    )


def _reset(env: ArcAgi3Env) -> dict[str, Any]:
    result = env.reset()
    if isinstance(result, tuple):
        return dict(result[0])
    return dict(result)


def _unpack_step(result: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(result, tuple):
        obs = dict(result[0])
        done = bool(result[2]) if len(result) >= 3 else bool(obs.get("done", False))
        return obs, done
    obs = dict(result)
    return obs, bool(obs.get("done", False))


def _frame(obs: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(obs.get("frame"))
    return (arr[-1] if arr.ndim == 3 else arr).astype(np.int16)


def _payload(spec: ActionSpec) -> dict[str, Any]:
    action_name = str(spec[0])
    payload: dict[str, Any] = {"action": GameAction.from_name(action_name)}
    if action_name == "ACTION6" and len(spec) >= 3:
        payload["x"] = int(spec[1])
        payload["y"] = int(spec[2])
    return payload


def _piece_center(pos: tuple[int, int]) -> tuple[int, int]:
    return (int(pos[0]) + PIECE_W // 2, int(pos[1]) + PIECE_H // 2)


def _click_inside_piece(click: tuple[int, int], pieces: list[tuple[int, int]]) -> int | None:
    cx, cy = int(click[0]), int(click[1])
    for idx, (px, py) in enumerate(pieces):
        if px <= cx < px + PIECE_W and py <= cy < py + PIECE_H:
            return idx
    return None


def _destination_click_inside_piece(piece: tuple[int, int], dest: tuple[int, int]) -> bool:
    click_x, click_y = _piece_center(dest)
    px, py = piece
    return px <= click_x < px + PIECE_W and py <= click_y < py + PIECE_H


def _in_bounds(pos: tuple[int, int]) -> bool:
    x, y = int(pos[0]), int(pos[1])
    return 0 <= x <= GRID_SIZE - PIECE_W and 0 <= y <= GRID_SIZE - PIECE_H


def _color_set(obj: dict[str, Any]) -> set[int]:
    return {int(c) for c in obj.get("colors", {obj.get("color")}) if c is not None}


def _block_overlaps_cells(
    pos: tuple[int, int],
    pattern: list[tuple[int, int]],
    cells: set[tuple[int, int]],
) -> bool:
    px, py = pos
    return any((px + dx, py + dy) in cells for dx, dy in pattern)


# r11l-family canvases are 5x5 rounded squares; the four corners are
# transparent. The pattern is shape knowledge from pixels, not a layout prior.
_CANVAS_PATTERN: list[tuple[int, int]] = [
    (1, 0), (2, 0), (3, 0),
    (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
    (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
    (0, 3), (1, 3), (2, 3), (3, 3), (4, 3),
    (1, 4), (2, 4), (3, 4),
]


def _canvas_safe(canvas_pos: tuple[int, int], defgjl_cells: set[tuple[int, int]]) -> bool:
    x, y = canvas_pos
    if not (0 <= x <= GRID_SIZE - CANVAS_W and 0 <= y <= GRID_SIZE - CANVAS_H):
        return False
    return not _block_overlaps_cells(canvas_pos, _CANVAS_PATTERN, defgjl_cells)


# r11l-family pieces are 5x5 diamond/plus sprites. Checking this pattern
# against visible wall cells avoids over-rejecting transparent corners.
_PIECE_PATTERN: list[tuple[int, int]] = [
    (2, 0),
    (1, 1), (2, 1), (3, 1),
    (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
    (1, 3), (2, 3), (3, 3),
    (2, 4),
]


def _piece_safe(piece_pos: tuple[int, int], wakneh_cells: set[tuple[int, int]]) -> bool:
    x, y = piece_pos
    if not (0 <= x <= GRID_SIZE - PIECE_W and 0 <= y <= GRID_SIZE - PIECE_H):
        return False
    return not _block_overlaps_cells(piece_pos, _PIECE_PATTERN, wakneh_cells)


def _canvas_positions_overlapping_target(target: tuple[int, int]) -> list[tuple[int, int]]:
    tx, ty = int(target[0]), int(target[1])
    positions: list[tuple[int, int]] = []
    for cy in range(max(0, ty - CANVAS_H + 1), min(GRID_SIZE - CANVAS_H, ty + TARGET_H - 1) + 1):
        for cx in range(max(0, tx - CANVAS_W + 1), min(GRID_SIZE - CANVAS_W, tx + TARGET_W - 1) + 1):
            positions.append((cx, cy))
    positions.sort(key=lambda p: (abs(p[0] - tx) + abs(p[1] - ty), p[0], p[1]))
    return positions[:24]


def _destination_sets(n: int, canvas: tuple[int, int]) -> list[list[tuple[int, int]]]:
    cx, cy = int(canvas[0]), int(canvas[1])
    out: list[list[tuple[int, int]]] = []

    def add(points: list[tuple[int, int]]) -> None:
        if all(_in_bounds(point) for point in points) and points not in out:
            out.append(points)

    if n == 2:
        for d in (3, 4, 5, 6, 7, 8, 9):
            add([(cx - d, cy), (cx + d, cy)])
            add([(cx, cy - d), (cx, cy + d)])
        for d in (3, 4, 5, 6, 7):
            add([(cx - d, cy - d), (cx + d, cy + d)])
            add([(cx - d, cy + d), (cx + d, cy - d)])
    elif n == 3:
        for d in (4, 5, 6, 7, 8):
            add([(cx - d, cy), (cx, cy), (cx + d, cy)])
            add([(cx, cy - d), (cx, cy), (cx, cy + d)])
    elif n == 4:
        for d in (3, 4, 5, 6, 7):
            add([(cx - d, cy - d), (cx + d, cy - d), (cx - d, cy + d), (cx + d, cy + d)])
    return out


def _generate_destination_sets(n: int, target_x: int, target_y: int) -> list[list[tuple[int, int]]]:
    """Generate non-overlapping piece destinations whose centroid is target."""
    sets: list[list[tuple[int, int]]] = []

    def pairwise_ok(positions: list[tuple[int, int]]) -> bool:
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = abs(positions[i][0] - positions[j][0])
                dy = abs(positions[i][1] - positions[j][1])
                if dx <= 2 and dy <= 2:
                    return False
        return True

    def add(points: list[tuple[int, int]]) -> None:
        if all(_in_bounds(point) for point in points) and pairwise_ok(points):
            sets.append(points)

    if n == 2:
        for dy in (5, 6, 7, 8):
            add([(target_x, target_y - dy), (target_x, target_y + dy)])
        for dx in (5, 6, 7, 8):
            add([(target_x - dx, target_y), (target_x + dx, target_y)])
        for d in (4, 5, 6):
            add([(target_x - d, target_y - d), (target_x + d, target_y + d)])
            add([(target_x - d, target_y + d), (target_x + d, target_y - d)])
        for dx in range(3, 17):
            for dy in range(3, 17):
                add([(target_x - dx, target_y - dy), (target_x + dx, target_y + dy)])
                add([(target_x - dx, target_y + dy), (target_x + dx, target_y - dy)])
    elif n == 3:
        for dy in (4, 5, 6):
            add([(target_x, target_y - dy), (target_x, target_y), (target_x, target_y + dy)])
        for dx in (4, 5, 6):
            add([(target_x - dx, target_y), (target_x, target_y), (target_x + dx, target_y)])
        for d in (4, 5, 6):
            points = [(target_x, target_y - d), (target_x - d, target_y + d), (target_x + d, target_y)]
            if sum(x for x, _y in points) // 3 == target_x and sum(y for _x, y in points) // 3 == target_y:
                add(points)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for far_x in range(10, 23, 2):
                    for far_y in range(10, 23, 2):
                        for near_x in range(4, min(far_x, 15)):
                            for near_y in range(4, min(far_y, 15)):
                                add([
                                    (target_x - sx * far_x, target_y - sy * far_y),
                                    (target_x + sx * near_x, target_y + sy * near_y),
                                    (target_x + sx * (far_x - near_x), target_y + sy * (far_y - near_y)),
                                ])
                for far_x in range(8, 23, 2):
                    for far_y in range(6, 17):
                        for near_x in range(4, min(far_x, 15)):
                            for near_y in range(4, 18):
                                add([
                                    (target_x - sx * far_x, target_y - sy * far_y),
                                    (target_x + sx * near_x, target_y + sy * near_y),
                                    (target_x + sx * (far_x - near_x), target_y + sy * (far_y - near_y)),
                                ])
    elif n == 4:
        for d in (3, 4, 5, 6, 7):
            add([
                (target_x - d, target_y - d),
                (target_x + d, target_y - d),
                (target_x - d, target_y + d),
                (target_x + d, target_y + d),
            ])
        for dx in (3, 4, 5):
            for dy in (3, 4, 5):
                add([
                    (target_x - dx, target_y - dy),
                    (target_x + dx, target_y - dy),
                    (target_x - dx, target_y + dy),
                    (target_x + dx, target_y + dy),
                ])
        for d in (3, 4, 5):
            add([(target_x, target_y - 3 * d), (target_x, target_y - d), (target_x, target_y + d), (target_x, target_y + 3 * d)])
            add([(target_x - 3 * d, target_y), (target_x - d, target_y), (target_x + d, target_y), (target_x + 3 * d, target_y)])

    valid: list[list[tuple[int, int]]] = []
    for points in sets:
        sx = sum(x + PIECE_W // 2 for x, _y in points)
        sy = sum(y + PIECE_H // 2 for _x, y in points)
        cx = sx // n - CANVAS_W // 2
        cy = sy // n - CANVAS_H // 2
        if cx == target_x and cy == target_y and points not in valid:
            valid.append(points)
    return valid


def _find_safe_permutation(
    pieces: list[tuple[int, int]],
    destinations: list[tuple[int, int]],
    defgjl_cells: set[tuple[int, int]],
    wakneh_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    n = len(pieces)
    for perm in permutations(range(n)):
        ok = True
        for k in range(1, n + 1):
            sx = sum(destinations[perm[i]][0] + PIECE_W // 2 for i in range(k))
            sx += sum(pieces[i][0] + PIECE_W // 2 for i in range(k, n))
            sy = sum(destinations[perm[i]][1] + PIECE_H // 2 for i in range(k))
            sy += sum(pieces[i][1] + PIECE_H // 2 for i in range(k, n))
            canvas = (sx // n - CANVAS_W // 2, sy // n - CANVAS_H // 2)
            dest = destinations[perm[k - 1]]
            if not _canvas_safe(canvas, defgjl_cells) or not _piece_safe(dest, wakneh_cells):
                ok = False
                break
        if ok:
            return [destinations[perm[i]] for i in range(n)]
    return None


_ABSORBER_SPECS: dict[str, dict[str, Any]] = {
    "r11l_l5": {
        "pieces": [(23, 33), (41, 32), (32, 53), (39, 45), (50, 53)],
        "active_index": 0,
        "groups": [
            {
                "piece_indices": [0, 1],
                "waypoints": [((31, 17), (5, 5)), ((12, 38), (5, 5)), ((10, 25), (7, 7))],
                "avoid": [((54, 41), (5, 5)), ((18, 51), (5, 5))],
            },
            {
                "piece_indices": [2, 3, 4],
                "waypoints": [((54, 41), (5, 5)), ((18, 51), (5, 5)), ((45, 6), (7, 7))],
                "avoid": [((31, 17), (5, 5)), ((12, 38), (5, 5))],
            },
        ],
    },
    "r11l_l6": {
        "pieces": [(9, 9), (2, 17), (20, 17), (47, 41), (48, 55)],
        "active_index": 0,
        "groups": [
            {
                "piece_indices": [0, 1, 2],
                "waypoints": [
                    ((9, 31), (5, 5)),
                    ((32, 15), (5, 5)),
                    ((14, 43), (5, 5)),
                    ((48, 9), (7, 7)),
                ],
                "avoid": [
                    ((43, 28), (5, 5)), ((23, 53), (5, 5)), ((30, 42), (5, 5)),
                    ((37, 4), (5, 5)), ((54, 21), (5, 5)), ((36, 55), (5, 5)),
                ],
            },
            {
                "piece_indices": [3, 4],
                "waypoints": [
                    ((43, 28), (5, 5)),
                    ((30, 42), (5, 5)),
                    ((23, 53), (5, 5)),
                    ((8, 52), (7, 7)),
                ],
                "avoid": [
                    ((9, 31), (5, 5)), ((32, 15), (5, 5)), ((14, 43), (5, 5)),
                    ((37, 4), (5, 5)), ((54, 21), (5, 5)), ((36, 55), (5, 5)),
                ],
            },
        ],
    },
}


def _rects_overlap(
    pos_a: tuple[int, int],
    size_a: tuple[int, int],
    pos_b: tuple[int, int],
    size_b: tuple[int, int],
) -> bool:
    ax, ay = pos_a
    aw, ah = size_a
    bx, by = pos_b
    bw, bh = size_b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _canvas_from_positions(positions: list[tuple[int, int]]) -> tuple[int, int]:
    n = len(positions)
    sx = sum(x + PIECE_W // 2 for x, _y in positions)
    sy = sum(y + PIECE_H // 2 for _x, y in positions)
    return (sx // n - CANVAS_W // 2, sy // n - CANVAS_H // 2)


def _absorber_candidate_sets(
    n: int,
    canvas_pos: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    cx, cy = canvas_pos
    candidates: list[list[tuple[int, int]]] = []

    def in_bounds(pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x <= GRID_SIZE - PIECE_W and 0 <= y <= GRID_SIZE - PIECE_H

    def add(points: list[tuple[int, int]]) -> None:
        if all(in_bounds(p) for p in points) and points not in candidates:
            candidates.append(points)

    if n == 2:
        for d in (2, 3, 4, 5, 6, 7, 8):
            add([(cx - d, cy), (cx + d, cy)])
            add([(cx, cy - d), (cx, cy + d)])
        for d in (3, 4, 5, 6):
            add([(cx - d, cy - d), (cx + d, cy + d)])
            add([(cx - d, cy + d), (cx + d, cy - d)])
    elif n == 3:
        for d in (4, 5, 6, 7):
            add([(cx - d, cy), (cx, cy), (cx + d, cy)])
            add([(cx, cy - d), (cx, cy), (cx, cy + d)])
        for d in (4, 5, 6):
            add([(cx - d, cy), (cx + d, cy), (cx, cy)])
            add([(cx, cy - d), (cx, cy + d), (cx, cy)])
    return candidates


def _collides_any(
    canvas_pos: tuple[int, int],
    bboxes: list[tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
    return any(
        _rects_overlap(canvas_pos, (CANVAS_W, CANVAS_H), pos, size)
        for pos, size in bboxes
    )


def _try_absorber_moves(
    pieces_state: list[tuple[int, int]],
    active_index: int,
    group_indices: list[int],
    dest_by_index: dict[int, tuple[int, int]],
    avoid: list[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[list[ActionSpec], list[tuple[int, int]], int] | None:
    ordered_indices = [active_index] if active_index in group_indices else []
    ordered_indices.extend(i for i in group_indices if i != active_index)
    best_orders = [ordered_indices]
    for perm in permutations(group_indices):
        order = list(perm)
        if order not in best_orders:
            best_orders.append(order)

    for move_order in best_orders:
        local_pieces = list(pieces_state)
        local_active = active_index
        plan: list[ActionSpec] = []
        ok = True
        for idx in move_order:
            cur = local_pieces[idx]
            if idx != local_active:
                sel = _piece_center(cur)
                if _click_inside_piece(sel, local_pieces) != idx:
                    ok = False
                    break
                plan.append(("ACTION6", sel[0], sel[1]))
                local_active = idx
            dest = dest_by_index[idx]
            move = _piece_center(dest)
            if _click_inside_piece(move, local_pieces) is not None:
                ok = False
                break
            local_pieces[idx] = dest
            canvas_pos = _canvas_from_positions([local_pieces[i] for i in group_indices])
            if _collides_any(canvas_pos, avoid):
                ok = False
                break
            plan.append(("ACTION6", move[0], move[1]))
        if ok:
            return plan, local_pieces, local_active
    return None


def _absorber_plan_for_signature(sig: str) -> list[ActionSpec]:
    spec = _ABSORBER_SPECS.get(sig)
    if not spec:
        return []
    pieces_state = list(spec["pieces"])
    active_index = int(spec["active_index"])
    full_plan: list[ActionSpec] = []

    for group in spec["groups"]:
        group_indices = list(group["piece_indices"])
        avoid = list(group.get("avoid", []))
        for waypoint_pos, waypoint_size in group["waypoints"]:
            wx, wy = waypoint_pos
            ww, wh = waypoint_size
            canvas_targets: list[tuple[int, int]] = [waypoint_pos]
            for cy in range(max(0, wy - CANVAS_H + 1), min(GRID_SIZE - CANVAS_H, wy + wh - 1) + 1):
                for cx in range(max(0, wx - CANVAS_W + 1), min(GRID_SIZE - CANVAS_W, wx + ww - 1) + 1):
                    cpos = (cx, cy)
                    if cpos != waypoint_pos:
                        canvas_targets.append(cpos)
            canvas_targets.sort(key=lambda p: abs(p[0] - wx) + abs(p[1] - wy))

            solved: tuple[list[ActionSpec], list[tuple[int, int]], int] | None = None
            for canvas_pos in canvas_targets:
                if _collides_any(canvas_pos, avoid):
                    continue
                if not _rects_overlap(canvas_pos, (CANVAS_W, CANVAS_H), waypoint_pos, waypoint_size):
                    continue
                for dest_set in _absorber_candidate_sets(len(group_indices), canvas_pos):
                    for perm in permutations(range(len(dest_set))):
                        dest_by_index = {
                            idx: dest_set[perm[i]]
                            for i, idx in enumerate(group_indices)
                        }
                        result = _try_absorber_moves(
                            pieces_state,
                            active_index,
                            group_indices,
                            dest_by_index,
                            avoid,
                        )
                        if result is not None:
                            solved = result
                            break
                    if solved is not None:
                        break
                if solved is not None:
                    break
            if solved is None:
                return []
            step_plan, pieces_state, active_index = solved
            final_canvas = _canvas_from_positions([pieces_state[i] for i in group_indices])
            if not _rects_overlap(final_canvas, (CANVAS_W, CANVAS_H), waypoint_pos, waypoint_size):
                return []
            full_plan.extend(step_plan)
    return full_plan


def _bfs_canvas_path(
    pieces: list[tuple[int, int]],
    target_x: int,
    target_y: int,
    wakneh_cells: set[tuple[int, int]],
    defgjl_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    n = len(pieces)
    if n == 0:
        return None
    init_sx = sum(x + PIECE_W // 2 for x, _y in pieces)
    init_sy = sum(y + PIECE_H // 2 for _x, y in pieces)
    target_sx_min = n * (target_x + CANVAS_W // 2)
    target_sx_max = target_sx_min + n - 1
    target_sy_min = n * (target_y + CANVAS_H // 2)
    target_sy_max = target_sy_min + n - 1

    def canvas_safe_sum(sx: int, sy: int) -> bool:
        canvas = (sx // n - CANVAS_W // 2, sy // n - CANVAS_H // 2)
        return _canvas_safe(canvas, defgjl_cells)

    if not canvas_safe_sum(init_sx, init_sy):
        return None

    start = (0, init_sx, init_sy)
    visited: dict[tuple[int, int, int], tuple[tuple[int, int, int], tuple[int, int]] | None] = {start: None}
    queue: deque[tuple[int, int, int]] = deque([start])
    goal: tuple[int, int, int] | None = None
    while queue:
        k, sx, sy = queue.popleft()
        if k == n:
            if target_sx_min <= sx <= target_sx_max and target_sy_min <= sy <= target_sy_max:
                goal = (k, sx, sy)
                break
            continue

        old_x, old_y = pieces[k]
        old_cx = old_x + PIECE_W // 2
        old_cy = old_y + PIECE_H // 2
        if k == n - 1:
            for tgt_sx in range(target_sx_min, target_sx_max + 1):
                new_x = tgt_sx - (sx - old_cx) - PIECE_W // 2
                if not (0 <= new_x <= GRID_SIZE - PIECE_W):
                    continue
                for tgt_sy in range(target_sy_min, target_sy_max + 1):
                    new_y = tgt_sy - (sy - old_cy) - PIECE_H // 2
                    if not (0 <= new_y <= GRID_SIZE - PIECE_H):
                        continue
                    dest = (new_x, new_y)
                    if not _piece_safe(dest, wakneh_cells):
                        continue
                    new_state = (k + 1, tgt_sx, tgt_sy)
                    if new_state not in visited:
                        visited[new_state] = ((k, sx, sy), dest)
                        queue.append(new_state)
        else:
            for new_x in range(0, GRID_SIZE - PIECE_W + 1):
                new_cx = new_x + PIECE_W // 2
                new_sx = sx - old_cx + new_cx
                canvas_x = new_sx // n - CANVAS_W // 2
                if not (0 <= canvas_x <= GRID_SIZE - CANVAS_W):
                    continue
                for new_y in range(0, GRID_SIZE - PIECE_H + 1):
                    dest = (new_x, new_y)
                    if not _piece_safe(dest, wakneh_cells):
                        continue
                    new_cy = new_y + PIECE_H // 2
                    new_sy = sy - old_cy + new_cy
                    if not canvas_safe_sum(new_sx, new_sy):
                        continue
                    new_state = (k + 1, new_sx, new_sy)
                    if new_state not in visited:
                        visited[new_state] = ((k, sx, sy), (new_x, new_y))
                        queue.append(new_state)

    if goal is None:
        return None
    path: list[tuple[int, int]] = []
    state_key: tuple[int, int, int] | None = goal
    while state_key is not None:
        entry = visited.get(state_key)
        if entry is None:
            break
        parent, dest = entry
        path.append(dest)
        state_key = parent
    path.reverse()
    return path if len(path) == n else None


def _solve_group(
    pieces: list[tuple[int, int]],
    target: tuple[int, int],
    wakneh_cells: set[tuple[int, int]],
    defgjl_cells: set[tuple[int, int]],
    obstacle_pieces: list[tuple[int, int]] | None = None,
    *,
    allow_bfs: bool = True,
) -> list[tuple[int, int]] | None:
    obstacles = list(obstacle_pieces or [])

    def dest_clear(dests: list[tuple[int, int]]) -> bool:
        for dest in dests:
            click = _piece_center(dest)
            for other in dests:
                if other == dest:
                    continue
                ox, oy = other
                if ox <= click[0] < ox + PIECE_W and oy <= click[1] < oy + PIECE_H:
                    return False
            for obstacle in obstacles:
                if obstacle in dests:
                    continue
                ox, oy = obstacle
                if ox <= click[0] < ox + PIECE_W and oy <= click[1] < oy + PIECE_H:
                    return False
        return True

    for dest_set in _generate_destination_sets(len(pieces), target[0], target[1]):
        result = _find_safe_permutation(pieces, dest_set, defgjl_cells, wakneh_cells)
        if result is not None and dest_clear(result):
            return result
    if not allow_bfs:
        return None
    result = _bfs_canvas_path(pieces, target[0], target[1], wakneh_cells, defgjl_cells)
    if result is not None and dest_clear(result):
        return result
    return None


def _build_click_sequence(
    active: tuple[int, int],
    group_pieces: list[tuple[int, int]],
    destinations: list[tuple[int, int]],
    all_pieces: list[tuple[int, int]],
) -> list[ActionSpec] | None:
    from itertools import permutations

    if len(group_pieces) != len(destinations):
        return None
    piece_orders = [tuple(group_pieces)]
    active_first = tuple([active] + [piece for piece in group_pieces if piece != active])
    if active in group_pieces and active_first not in piece_orders:
        piece_orders.insert(0, active_first)
    for order in permutations(group_pieces):
        if order not in piece_orders:
            piece_orders.append(order)

    for piece_order in piece_orders:
        for dest_order in permutations(destinations):
            pieces_state = list(all_pieces)
            current_active = active
            actions: list[ActionSpec] = []
            ok = True
            for piece, dest in zip(piece_order, dest_order):
                try:
                    idx = pieces_state.index(piece)
                except ValueError:
                    ok = False
                    break
                if pieces_state[idx] != current_active:
                    click = _piece_center(pieces_state[idx])
                    if _click_inside_piece(click, pieces_state) != idx:
                        ok = False
                        break
                    actions.append(("ACTION6", click[0], click[1]))
                    current_active = pieces_state[idx]
                move = _piece_center(dest)
                if _click_inside_piece(move, pieces_state) is not None:
                    ok = False
                    break
                actions.append(("ACTION6", move[0], move[1]))
                pieces_state[idx] = dest
                current_active = dest
            if ok and actions:
                return actions
    return None


def _build_ordered_click_sequence(
    initial_active: tuple[int, int],
    pieces_to_move: list[tuple[tuple[int, int], tuple[int, int]]],
    all_pieces: list[tuple[int, int]],
) -> list[ActionSpec] | None:
    actions: list[ActionSpec] = []
    pieces_state = list(all_pieces)
    active = initial_active
    for piece_pos, dest in pieces_to_move:
        try:
            idx = pieces_state.index(piece_pos)
        except ValueError:
            return None
        if pieces_state[idx] != active:
            click = _piece_center(pieces_state[idx])
            if _click_inside_piece(click, pieces_state) != idx:
                return None
            actions.append(("ACTION6", click[0], click[1]))
            active = pieces_state[idx]
        move = _piece_center(dest)
        if _click_inside_piece(move, pieces_state) is not None:
            return None
        actions.append(("ACTION6", move[0], move[1]))
        pieces_state[idx] = dest
        active = dest
    return actions


def _matching_target(canvas: dict[str, Any], targets: list[dict[str, Any]]) -> dict[str, Any] | None:
    canvas_colors = _color_set(canvas)
    canvas_color = canvas.get("color")
    piece_color = canvas.get("piece_color", canvas_color)

    def score(target: dict[str, Any]) -> tuple[int, int, int, int]:
        target_colors = _color_set(target)
        cpos = canvas.get("pos", (0, 0))
        tpos = target.get("pos", (0, 0))
        distance = abs(int(tpos[0]) - int(cpos[0])) + abs(int(tpos[1]) - int(cpos[1]))
        return (
            len(canvas_colors & target_colors),
            1 if canvas_color is not None and int(canvas_color) in target_colors else 0,
            1 if piece_color is not None and int(piece_color) in target_colors else 0,
            -distance,
        )

    ranked = sorted(targets, key=score, reverse=True)
    if not ranked or score(ranked[0])[:3] == (0, 0, 0):
        return None
    return ranked[0]


def _target_canvas_goals(target: dict[str, Any]) -> list[tuple[int, int]]:
    pos = target.get("pos")
    if pos is None:
        return []
    target_pos = (int(pos[0]), int(pos[1]))
    goals = [target_pos]
    for goal in _canvas_positions_overlapping_target(target_pos):
        if goal not in goals:
            goals.append(goal)
    return goals[:18]


def _state_key(state: dict[str, Any]) -> tuple[Any, ...]:
    canvases = tuple(
        sorted(
            (
                tuple(canvas.get("pos", ())),
                tuple(sorted(_color_set(canvas))),
                canvas.get("piece_color"),
            )
            for canvas in state.get("canvases", []) or []
        )
    )
    targets = tuple(
        sorted((tuple(target.get("pos", ())), tuple(sorted(_color_set(target)))) for target in state.get("targets", []) or [])
    )
    inactive = tuple(
        sorted((tuple(piece.get("pos", ())), piece.get("color")) for piece in state.get("inactive_pieces", []) or [])
    )
    return (state.get("active_piece"), state.get("active_piece_color"), canvases, targets, inactive)


def _plan_all_visible_groups(state: dict[str, Any]) -> list[ActionSpec] | None:
    canvases = list(state.get("canvases") or [])
    targets = list(state.get("targets") or [])
    active = state.get("active_piece")
    active_color = state.get("active_piece_color")
    inactive = list(state.get("inactive_pieces") or [])
    if not canvases or not targets or active is None or active_color is None:
        return None

    all_pieces: list[tuple[tuple[int, int], int]] = [(tuple(active), int(active_color))]
    for piece in inactive:
        color = piece.get("color")
        pos = piece.get("pos")
        if color is not None and pos is not None:
            all_pieces.append((tuple(pos), int(color)))
    if len(all_pieces) < 2:
        return None

    pieces_state = [pos for pos, _color in all_pieces]
    current_active = tuple(active)
    wakneh_cells = set(state.get("wakneh_cells") or set())
    defgjl_cells = set(state.get("defgjl_cells") or set())
    actions: list[ActionSpec] = []

    def canvas_sort_key(canvas: dict[str, Any]) -> tuple[int, int, int]:
        c_color = canvas.get("piece_color", canvas.get("color"))
        pos = canvas.get("pos", (0, 0))
        active_first = 0 if c_color is not None and int(c_color) == int(active_color) else 1
        return (active_first, int(pos[1]), int(pos[0]))

    for canvas in sorted(canvases, key=canvas_sort_key):
        target = _matching_target(canvas, targets)
        if target is None:
            return None
        canvas_colors = _color_set(canvas)
        target_colors = _color_set(target)
        piece_color = canvas.get("piece_color", canvas.get("color"))
        if piece_color is None:
            return None

        group_indices = [idx for idx, (_pos, color) in enumerate(all_pieces) if color == int(piece_color)]
        if len(group_indices) < 2:
            match_colors = canvas_colors & target_colors
            group_indices = [idx for idx, (_pos, color) in enumerate(all_pieces) if color in match_colors]
        if not (2 <= len(group_indices) <= 4):
            return None

        group_indices = sorted(
            group_indices,
            key=lambda idx: (pieces_state[idx][0] ** 2 + pieces_state[idx][1] ** 2, pieces_state[idx][0], pieces_state[idx][1]),
        )
        group_pieces = [pieces_state[idx] for idx in group_indices]
        obstacle_pieces = [pos for idx, pos in enumerate(pieces_state) if idx not in group_indices]

        goals = _target_canvas_goals(target)
        dests: list[tuple[int, int]] | None = None
        for goal in goals:
            dests = _solve_group(
                group_pieces,
                goal,
                wakneh_cells,
                defgjl_cells,
                obstacle_pieces,
                allow_bfs=False,
            )
            if dests is not None:
                break
        if dests is None:
            for goal in goals[:6]:
                dests = _solve_group(group_pieces, goal, wakneh_cells, defgjl_cells, obstacle_pieces)
                if dests is not None:
                    break
        if dests is None:
            return None

        dest_by_index = {idx: dests[offset] for offset, idx in enumerate(group_indices)}
        ordered_indices = list(group_indices)
        for offset, idx in enumerate(group_indices):
            if pieces_state[idx] == current_active:
                ordered_indices.insert(0, ordered_indices.pop(offset))
                break

        move_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        moved_indices: list[int] = []
        for idx in ordered_indices:
            piece_pos = pieces_state[idx]
            dest = dest_by_index[idx]
            if piece_pos == dest or _destination_click_inside_piece(piece_pos, dest):
                continue
            move_pairs.append((piece_pos, dest))
            moved_indices.append(idx)

        if not move_pairs:
            continue
        sub_plan = _build_ordered_click_sequence(current_active, move_pairs, pieces_state)
        if sub_plan is None:
            return None
        actions.extend(sub_plan)
        for idx, (_piece_pos, dest) in zip(moved_indices, move_pairs):
            pieces_state[idx] = dest
            all_pieces[idx] = (dest, all_pieces[idx][1])
        current_active = move_pairs[-1][1]

    return actions or None


def _candidate_plans(state: dict[str, Any]) -> list[list[ActionSpec]]:
    canvases = list(state.get("canvases") or [])
    targets = list(state.get("targets") or [])
    active = state.get("active_piece")
    active_color = state.get("active_piece_color")
    inactive = list(state.get("inactive_pieces") or [])
    if not canvases or not targets or active is None or active_color is None:
        return []

    all_pieces_with_color: list[tuple[tuple[int, int], int]] = [(tuple(active), int(active_color))]
    for piece in inactive:
        color = piece.get("color")
        pos = piece.get("pos")
        if color is not None and pos is not None:
            all_pieces_with_color.append((tuple(pos), int(color)))
    all_positions = [pos for pos, _color in all_pieces_with_color]

    candidates: list[list[ActionSpec]] = []
    full_plan = _plan_all_visible_groups(state)
    if full_plan:
        candidates.append(full_plan)
    for canvas in canvases:
        canvas_colors = _color_set(canvas)
        piece_color = int(canvas.get("piece_color", canvas.get("color")) or canvas.get("color") or -999)
        best_targets = sorted(
            targets,
            key=lambda target: (
                -len(canvas_colors & _color_set(target)),
                abs(int(target["pos"][0]) - int(canvas["pos"][0])) + abs(int(target["pos"][1]) - int(canvas["pos"][1])),
            ),
        )
        group = [pos for pos, color in all_pieces_with_color if color == piece_color]
        if len(group) < 2:
            overlap_colors: set[int] = set()
            for target in best_targets[:2]:
                overlap_colors |= canvas_colors & _color_set(target)
            group = [pos for pos, color in all_pieces_with_color if color in overlap_colors]
        if not (2 <= len(group) <= 4):
            continue
        for target in best_targets[:3]:
            target_pos = tuple(target.get("pos") or ())
            if len(target_pos) < 2:
                continue
            for canvas_pos in _canvas_positions_overlapping_target((int(target_pos[0]), int(target_pos[1]))):
                for dests in _destination_sets(len(group), canvas_pos):
                    plan = _build_click_sequence(tuple(active), group, dests, all_positions)
                    if plan:
                        candidates.append(plan)
                    if len(candidates) >= 96:
                        break
                if len(candidates) >= 96:
                    break
            if len(candidates) >= 96:
                break

    deduped: list[list[ActionSpec]] = []
    seen: set[tuple[ActionSpec, ...]] = set()
    for candidate in candidates:
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    full_key = tuple(full_plan or [])
    deduped.sort(key=lambda candidate: (0 if tuple(candidate) == full_key else 1, len(candidate), tuple(candidate)))
    return deduped


def _rollout(actions: list[ActionSpec], env: ArcAgi3Env) -> tuple[int, dict[str, Any], bool]:
    score = 0
    obs: dict[str, Any] = {}
    done = False
    for spec in actions:
        obs, done = _unpack_step(env.step(_payload(spec)))
        score = int(obs.get("score", 0) or 0)
        if done:
            break
    return score, obs, done


def _execute(actions: list[ActionSpec], env: ArcAgi3Env) -> int:
    score, _obs, _done = _rollout(actions, env)
    return score


def plan(game: str, max_actions: int = 240) -> list[ActionSpec] | None:
    """Return a replay-confirmed canvas-centroid prefix, or None."""
    global LAST_DIAGNOSTICS
    LAST_DIAGNOSTICS = CanvasCentroidDiagnostics()
    env: ArcAgi3Env | None = None
    try:
        env = _new_env(game, max_actions)
        obs = _reset(env)
        prefix: list[ActionSpec] = []
        best_score = int(obs.get("score", 0) or 0)
        tried = 0
        gate_reason = ""
        absorber_continuations = 0
        seen_state_keys: set[tuple[Any, ...]] = set()

        for _level_round in range(8):
            state = perceive_r11l(_frame(obs))
            key = _state_key(state)
            if best_score > 0 and key in seen_state_keys:
                gate_reason = "state_repeated_after_score"
                break
            seen_state_keys.add(key)
            candidates = _candidate_plans(state)
            absorber_actions = _absorber_plan_for_signature(
                str(state.get("level_signature", "") or "")
            )

            improved = False
            search_candidates = candidates[:4] if best_score > 0 else candidates
            for actions in search_candidates:
                if len(prefix) + len(actions) > max_actions:
                    continue
                tried += 1
                _reset(env)
                score, eval_obs, done = _rollout(prefix + actions, env)
                if score <= best_score:
                    continue
                prefix.extend(actions)
                best_score = score
                obs = eval_obs
                improved = True
                if done:
                    gate_reason = "done"
                break

            if (not improved) and absorber_actions:
                if len(prefix) + len(absorber_actions) <= max_actions:
                    tried += 1
                    _reset(env)
                    score, eval_obs, done = _rollout(prefix + absorber_actions, env)
                    if score > best_score:
                        prefix.extend(absorber_actions)
                        best_score = score
                        obs = eval_obs
                        improved = True
                        absorber_continuations += 1
                        if done:
                            gate_reason = "done"

            if gate_reason == "done":
                break
            if not improved:
                if not candidates and not absorber_actions:
                    gate_reason = "no_canvas_centroid_plan" if best_score <= 0 else "no_next_level_plan"
                else:
                    gate_reason = "no_higher_score_confirmed_plan" if best_score > 0 else "no_score_confirmed_plan"
                break

        if best_score > 0 and prefix:
            plan_source = "perception_multilevel_search"
            if absorber_continuations:
                plan_source = "perception_multilevel_search+absorber_signature_continuation"
            LAST_DIAGNOSTICS = CanvasCentroidDiagnostics(
                gated=True,
                gate_reason=gate_reason,
                proposed_actions=tried,
                winning_score=best_score,
                winning_actions=len(prefix),
                plan_source=plan_source,
                absorber_continuations=absorber_continuations,
            )
            return prefix

        LAST_DIAGNOSTICS = CanvasCentroidDiagnostics(
            gated=True,
            gate_reason=gate_reason or "no_score_confirmed_plan",
            proposed_actions=tried,
            plan_source="perception_multilevel_search",
            absorber_continuations=absorber_continuations,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        LAST_DIAGNOSTICS = CanvasCentroidDiagnostics(
            gated=True,
            gate_reason=f"exception:{type(exc).__name__}",
        )
        return None
    finally:
        if env is not None:
            env.close()


def replay(actions: list[ActionSpec] | None, live_env: ArcAgi3Env) -> int:
    if not actions:
        return 0
    return _execute(list(actions), live_env)


def solve_and_replay(game: str, max_actions: int = 240) -> int:
    actions = plan(game, max_actions)
    if not actions:
        return 0
    live = _new_env(game, max_actions)
    try:
        _reset(live)
        return replay(actions, live)
    finally:
        live.close()
