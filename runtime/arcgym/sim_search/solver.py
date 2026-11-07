"""Sim-search BFS over a locally-instantiated game.

S1 core: effective-action scan (with ACTION6 click compression) + plain
BFS with the trigger-aware state hash. The cascade extensions
(counter-A*, cross-level transfer, warm-up unlock) land in S3.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from arcgym.sim_search.local_game import LocalSim, clone_game, load_game_class
from arcgym.sim_search.source_discovery import find_game_source
from arcgym.sim_search.state_hash import probe_trigger_fields, state_key

try:
    from scipy import ndimage as _ndi
except ImportError:  # pragma: no cover - scipy absent in minimal envs
    _ndi = None

MOVE_NAMES: tuple[str, ...] = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7")


@dataclass
class SimSearchResult:
    found: bool
    plan: list[tuple] = field(default_factory=list)
    nodes_expanded: int = 0
    elapsed: float = 0.0
    reason: str = ""
    # Per-rung trail filled by solve_level_cascade:
    # [(rung_name, reason, nodes, elapsed_s), ...]
    rungs: list[tuple] = field(default_factory=list)


def _background_color(frame: np.ndarray) -> int:
    values, counts = np.unique(frame, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _click_candidates(
    frame: np.ndarray,
    max_sites: int,
    stride: int = 6,
    dense_non_bg: bool = False,
) -> list[tuple[int, int]]:
    """Candidate ACTION6 coordinates.

    Default: object centroids + bbox corners, then a stride grid
    backstop (compression happens later via the frame-change + novel-
    result filter in scan_actions). ``dense_non_bg`` switches to a
    stride-2 sweep over non-background cells — the high-recall mode for
    games whose decisive click is not on a centroid/corner/grid point.
    """
    height, width = frame.shape[:2]
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def push(x: int, y: int) -> None:
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        if (x, y) not in seen:
            seen.add((x, y))
            coords.append((x, y))

    bg = _background_color(frame)
    if dense_non_bg:
        # Every non-background cell. The scan deadline in scan_actions
        # bounds the probe cost; recall matters more than speed here.
        for y in range(0, height):
            for x in range(0, width):
                if int(frame[y, x]) != bg:
                    push(x, y)
        return coords[: max_sites if max_sites > 0 else len(coords)]
    if _ndi is not None:
        comps: list[tuple[int, int, int, int, int, int, int]] = []
        for value in np.unique(frame):
            if int(value) == bg:
                continue
            labeled, count = _ndi.label(frame == value)
            for idx in range(1, count + 1):
                mask = labeled == idx
                area = int(mask.sum())
                if not (2 <= area <= 3000):
                    continue
                ys, xs = np.where(mask)
                comps.append(
                    (area, int(round(xs.mean())), int(round(ys.mean())),
                     int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
                )
        comps.sort(reverse=True)
        for area, cx, cy, x0, x1, y0, y1 in comps:
            push(cx, cy)
            push(x0, y0)
            push(x1, y0)
            push(x0, y1)
            push(x1, y1)

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            push(x, y)
    return coords[: max_sites if max_sites > 0 else len(coords)]


def scan_actions(
    sim: LocalSim,
    frame0: np.ndarray,
    scan_timeout: float = 6.0,
    max_sites: int = 400,
    stride: int = 6,
    dedup: bool = True,
    dense_non_bg: bool = False,
) -> list[tuple]:
    """Effective actions from the root state.

    Moves: keep the ones that change the frame. Clicks: probe candidate
    coordinates on clones and keep only frame-changing clicks whose
    RESULT frame is novel (dedup compresses equivalent clicks to one).
    """
    deadline = time.monotonic() + scan_timeout
    available = set(sim.available_action_names())
    effective: list[tuple] = []
    result_hashes: set[int] = {hash(frame0.tobytes())}

    for name in MOVE_NAMES:
        if name not in available:
            continue
        probe = clone_game(sim.game)
        try:
            frame, _levels, _idx = LocalSim.step(probe, (name,))
        except Exception:
            continue
        if int((frame != frame0).sum()) > 0:
            effective.append((name,))
            result_hashes.add(hash(frame.tobytes()))

    if "ACTION6" in available:
        for x, y in _click_candidates(frame0, max_sites, stride=stride,
                                      dense_non_bg=dense_non_bg):
            if time.monotonic() > deadline:
                break
            probe = clone_game(sim.game)
            try:
                frame, _levels, _idx = LocalSim.step(probe, ("ACTION6", x, y))
            except Exception:
                continue
            if int((frame != frame0).sum()) == 0:
                continue
            key = hash(frame.tobytes())
            if dedup and key in result_hashes:
                continue
            result_hashes.add(key)
            effective.append(("ACTION6", x, y))
    return effective


def solve_level(
    sim: LocalSim,
    level_idx: int,
    frame0: np.ndarray,
    budget_s: float = 60.0,
    max_states: int = 200_000,
    max_depth: int = 40,
    scan_timeout: float = 6.0,
    max_sites: int = 400,
    stride: int = 6,
    dedup: bool = True,
    dense_non_bg: bool = False,
) -> SimSearchResult:
    """BFS for one level on the local sim. ``sim`` must be at the level's
    initial state (``frame0`` = the frame ``reset_level(level_idx)`` returned)."""
    start = time.monotonic()

    actions = scan_actions(sim, frame0, scan_timeout=scan_timeout, max_sites=max_sites,
                           stride=stride, dedup=dedup, dense_non_bg=dense_non_bg)
    if not actions:
        return SimSearchResult(found=False, reason="no_effective_actions",
                               elapsed=time.monotonic() - start)

    trigger_fields = probe_trigger_fields(sim.game, actions)
    visited = {state_key(sim.game, frame0, trigger_fields)}
    queue: deque[tuple[object, list[tuple], int]] = deque()
    queue.append((sim.game, [], 0))
    nodes = 0
    deadline = start + budget_s

    while queue:
        if time.monotonic() > deadline:
            return SimSearchResult(found=False, nodes_expanded=nodes, reason="timeout",
                                   elapsed=time.monotonic() - start)
        if len(visited) > max_states:
            return SimSearchResult(found=False, nodes_expanded=nodes, reason="max_states",
                                   elapsed=time.monotonic() - start)
        game, hist, depth = queue.popleft()
        if depth >= max_depth:
            continue
        nodes += 1
        for spec in actions:
            child = clone_game(game)
            try:
                frame, levels, current_idx = LocalSim.step(child, spec)
            except Exception:
                continue
            if levels > sim.base_levels or current_idx > level_idx:
                return SimSearchResult(found=True, plan=hist + [spec], nodes_expanded=nodes,
                                       elapsed=time.monotonic() - start, reason="solved")
            key = state_key(child, frame, trigger_fields)
            if key in visited:
                continue
            visited.add(key)
            queue.append((child, hist + [spec], depth + 1))

    return SimSearchResult(found=False, nodes_expanded=nodes, reason="exhausted",
                           elapsed=time.monotonic() - start)


def counter_astar(
    sim: LocalSim,
    level_idx: int,
    frame0: np.ndarray,
    win_field: str,
    direction: int,
    budget_s: float = 300.0,
    max_states: int = 400_000,
    max_depth: int = 40,
    scan_timeout: float = 6.0,
) -> SimSearchResult:
    """Best-first search prioritised by the win-field counter.

    Priority is lexicographic ``(-counter*direction, depth, fifo)`` — the
    dense signal plain BFS lacks on wide click games whose win condition
    is "self.X reaches a threshold" in the source.
    """
    import heapq

    start = time.monotonic()
    actions = scan_actions(sim, frame0, scan_timeout=scan_timeout)
    if not actions:
        return SimSearchResult(found=False, reason="no_effective_actions",
                               elapsed=time.monotonic() - start)
    base_value = getattr(sim.game, win_field, 0) or 0
    if not isinstance(base_value, (bool, int, float)):
        # Extracted name is not a scalar counter (e.g. a method or a
        # container); the dense-priority premise does not hold.
        return SimSearchResult(found=False, reason="win_field_not_scalar",
                               elapsed=time.monotonic() - start)
    trigger_fields = probe_trigger_fields(sim.game, actions)
    if win_field not in trigger_fields:
        trigger_fields = trigger_fields + (win_field,)
    visited = {state_key(sim.game, frame0, trigger_fields)}
    seq = 0
    heap: list[tuple[float, int, int, object, list[tuple]]] = [
        (-(float(base_value) * direction), 0, seq, sim.game, [])
    ]
    nodes = 0
    deadline = start + budget_s
    while heap:
        if time.monotonic() > deadline:
            return SimSearchResult(found=False, nodes_expanded=nodes, reason="timeout",
                                   elapsed=time.monotonic() - start)
        if len(visited) > max_states:
            return SimSearchResult(found=False, nodes_expanded=nodes, reason="max_states",
                                   elapsed=time.monotonic() - start)
        _prio, depth, _seq, game, hist = heapq.heappop(heap)
        if depth >= max_depth:
            continue
        nodes += 1
        for spec in actions:
            child = clone_game(game)
            try:
                frame, levels, current_idx = LocalSim.step(child, spec)
            except Exception:
                continue
            if levels > sim.base_levels or current_idx > level_idx:
                return SimSearchResult(found=True, plan=hist + [spec], nodes_expanded=nodes,
                                       elapsed=time.monotonic() - start,
                                       reason="counter_astar")
            key = state_key(child, frame, trigger_fields)
            if key in visited:
                continue
            visited.add(key)
            seq += 1
            value = getattr(child, win_field, 0) or 0
            heapq.heappush(
                heap,
                (-(float(value) * direction), depth + 1, seq, child, hist + [spec]),
            )
    return SimSearchResult(found=False, nodes_expanded=nodes, reason="exhausted",
                           elapsed=time.monotonic() - start)


def objective_astar(
    sim: LocalSim,
    level_idx: int,
    frame0: np.ndarray,
    budget_s: float = 300.0,
    max_nodes_per_candidate: int = 2500,
    max_depth: int = 40,
    scan_timeout: float = 6.0,
    top_candidates: int = 3,
) -> SimSearchResult:
    """Best-first search guided by induced relational objectives.

    Reuses the shipped generic inducer (P13-P17 line: approach_marker /
    approach_same_color / cover_markers proved load-bearing on
    cn04/ka59/dc22 at ~1.2-2k nodes where blind BFS drowns). Candidates
    are ranked by one-step gradient (p8 ranker) and tried in order under
    a node cap per candidate.
    """
    import heapq

    try:
        from arcgym.eca.relational_objective_inducer import (
            evaluate_relation,
            induce_candidates,
        )
    except ImportError:
        return SimSearchResult(found=False, reason="inducer_unavailable")

    start = time.monotonic()
    deadline = start + budget_s
    actions = scan_actions(sim, frame0, scan_timeout=scan_timeout)
    if not actions:
        return SimSearchResult(found=False, reason="no_effective_actions",
                               elapsed=time.monotonic() - start)

    bg = _background_color(frame0)
    obj_colors: set[int] = set()
    root_succ_frames: list[np.ndarray] = []
    for spec in actions:
        probe = clone_game(sim.game)
        try:
            frame, _levels, _idx = LocalSim.step(probe, spec)
        except Exception:
            continue
        root_succ_frames.append(frame)
        changed = frame != frame0
        if bool(changed.any()):
            for fr in (frame0, frame):
                obj_colors |= {int(v) for v in np.unique(fr[changed]) if int(v) != bg}
    if not obj_colors:
        return SimSearchResult(found=False, reason="no_object_colors",
                               elapsed=time.monotonic() - start)

    candidates = induce_candidates(frame0, sorted(obj_colors))
    if not candidates:
        return SimSearchResult(found=False, reason="no_candidates",
                               elapsed=time.monotonic() - start)

    def actionability(rel) -> float:
        base = float(evaluate_relation(frame0, rel))
        best = max((float(evaluate_relation(fr, rel)) for fr in root_succ_frames),
                   default=base)
        return best - base

    ranked = sorted(candidates, key=lambda r: -actionability(r))[:top_candidates]
    total_nodes = 0
    for rel in ranked:
        visited = {hash(frame0.tobytes())}
        seq = 0
        heap = [(-float(evaluate_relation(frame0, rel)), 0, seq, sim.game, [])]
        nodes = 0
        while heap and nodes < max_nodes_per_candidate:
            if time.monotonic() > deadline:
                return SimSearchResult(found=False, nodes_expanded=total_nodes + nodes,
                                       reason="timeout",
                                       elapsed=time.monotonic() - start)
            _prio, depth, _seq, game, hist = heapq.heappop(heap)
            if depth >= max_depth:
                continue
            nodes += 1
            for spec in actions:
                child = clone_game(game)
                try:
                    frame, levels, current_idx = LocalSim.step(child, spec)
                except Exception:
                    continue
                if levels > sim.base_levels or current_idx > level_idx:
                    return SimSearchResult(
                        found=True, plan=hist + [spec],
                        nodes_expanded=total_nodes + nodes,
                        elapsed=time.monotonic() - start,
                        reason=f"objective:{rel.key()}",
                    )
                key = hash(frame.tobytes())
                if key in visited:
                    continue
                visited.add(key)
                seq += 1
                heapq.heappush(
                    heap,
                    (-float(evaluate_relation(frame, rel)), depth + 1, seq,
                     child, hist + [spec]),
                )
        total_nodes += nodes
    return SimSearchResult(found=False, nodes_expanded=total_nodes,
                           reason="objective_exhausted",
                           elapsed=time.monotonic() - start)


def _warm_sim(sim: LocalSim, move: str) -> tuple[LocalSim, np.ndarray] | None:
    """Clone-level sim advanced by one warm-up move (frame must change)."""
    warm_game = clone_game(sim.game)
    try:
        frame, _levels, _idx = LocalSim.step(warm_game, (move,))
    except Exception:
        return None
    warm = LocalSim.__new__(LocalSim)
    warm._game_cls = sim._game_cls
    warm.game = warm_game
    warm.base_levels = sim.base_levels
    return warm, frame


def _try_transfer_replay(
    sim: LocalSim,
    level_idx: int,
    prev_plan: list[tuple],
    multipliers: tuple[int, ...] = (1, 2, 3),
) -> list[tuple] | None:
    """Cross-level transfer, cheapest form: replay the previous level's
    plan (optionally with each action repeated m times) on a clone and
    keep the prefix if it clears the level."""
    if not prev_plan:
        return None
    for mult in multipliers:
        candidate: list[tuple] = []
        for spec in prev_plan:
            candidate.extend([spec] * mult)
        clone = clone_game(sim.game)
        executed: list[tuple] = []
        for spec in candidate:
            try:
                _frame, levels, current_idx = LocalSim.step(clone, spec)
            except Exception:
                break
            executed.append(spec)
            if levels > sim.base_levels or current_idx > level_idx:
                return executed
    return None


def solve_level_cascade(
    sim: LocalSim,
    level_idx: int,
    frame0: np.ndarray,
    prev_plan: list[tuple] | None = None,
    budget_ladder: tuple[float, ...] = (60.0, 300.0),
    max_states: int = 400_000,
    max_depth: int = 40,
    source_path: str | None = None,
    total_budget_s: float | None = None,
) -> SimSearchResult:
    """S2.5/S3a cascade entry.

    Rungs, cheapest first (each maps to a measured failure mode):
      1. transfer        — replay/multiply the previous level's plan
      2. BFS ladder      — budget-only failures (ar25/dc22/ka59/r11l class)
      3. warm-up unlock  — no-effective-actions (sc25 class)
      4. counter-A*      — wide click spaces with a source win-field
                           (cn04/re86 class)
      5. dense rescan    — decisive click off the candidate lattice or
                           killed by dedup (su15 class): stride-2 non-bg,
                           dedup off
    The WHOLE cascade is bounded by ``total_budget_s`` (default 2x the
    deepest ladder rung): without it a per-level miss costs the sum of
    every rung's budget, which starves the fallback stack (measured:
    sb26 truncated at 3 levels by the suite budget cutoff because the
    lane burned 600s+ of its wall-time re-missing at every boundary).
    """
    start = time.monotonic()
    if total_budget_s is None:
        total_budget_s = 2.0 * max(budget_ladder)
    cascade_deadline = start + total_budget_s

    def _remaining() -> float:
        return cascade_deadline - time.monotonic()

    trail: list[tuple] = []

    def _finish(result: SimSearchResult, rung: str) -> SimSearchResult:
        trail.append((rung, result.reason, result.nodes_expanded,
                      round(result.elapsed, 1)))
        result.rungs = list(trail)
        return result

    transferred = _try_transfer_replay(sim, level_idx, prev_plan or [])
    if transferred is not None:
        return _finish(
            SimSearchResult(found=True, plan=transferred, nodes_expanded=0,
                            elapsed=time.monotonic() - start,
                            reason="transfer_replay"),
            "transfer",
        )

    last = SimSearchResult(found=False, reason="no_budget")
    for budget in budget_ladder:
        if _remaining() < 5.0:
            break
        last = solve_level(
            sim, level_idx, frame0,
            budget_s=min(budget, _remaining()),
            max_states=max_states, max_depth=max_depth,
        )
        trail.append((f"bfs_{int(budget)}s", last.reason, last.nodes_expanded,
                      round(last.elapsed, 1)))
        if last.found or last.reason in ("no_effective_actions", "exhausted",
                                         "max_states"):
            break
    if last.found:
        last.rungs = list(trail)
        return last

    # Warm-up unlock: nothing reacts from the initial state; a single
    # directional move often opens the mechanism. The warm-up move is
    # prepended to any solution found from the warmed state.
    if last.reason in ("no_effective_actions", "exhausted"):
        for move in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if _remaining() < 5.0:
                break
            warmed = _warm_sim(sim, move)
            if warmed is None:
                continue
            warm, warm_frame = warmed
            # no_effective_actions: re-scan even when the warm move left
            # the frame unchanged — some games swallow the first input on
            # a hidden latch (sc25's qytejzcythm) with no visible effect.
            # exhausted: a frame-identical warm state would just repeat
            # the same exhausted search, so skip those.
            if int((warm_frame != frame0).sum()) == 0 and last.reason == "exhausted":
                continue
            warm_res = solve_level(
                warm, level_idx, warm_frame,
                budget_s=min(budget_ladder[0], max(5.0, _remaining())),
                max_states=max_states,
                max_depth=max_depth,
            )
            trail.append((f"warmup_{move}", warm_res.reason,
                          warm_res.nodes_expanded, round(warm_res.elapsed, 1)))
            if warm_res.found:
                warm_res.plan = [(move,)] + warm_res.plan
                warm_res.reason = "warmup_unlock"
                warm_res.rungs = list(trail)
                return warm_res

    # Counter-A*: needs the source text for win-field extraction.
    if source_path and _remaining() >= 5.0:
        from arcgym.sim_search.win_field import extract_win_field

        field, direction = extract_win_field(source_path)
        if field and direction:
            try:
                astar_res = counter_astar(
                    sim, level_idx, frame0, field, direction,
                    budget_s=min(budget_ladder[-1], _remaining()),
                    max_states=max_states,
                    max_depth=max_depth,
                )
            except Exception as exc:
                astar_res = SimSearchResult(
                    found=False, reason=f"counter_astar_error:{type(exc).__name__}")
            trail.append((f"counter_astar:{field}", astar_res.reason,
                          astar_res.nodes_expanded, round(astar_res.elapsed, 1)))
            if astar_res.found:
                astar_res.rungs = list(trail)
                return astar_res
            last = astar_res if astar_res.nodes_expanded > last.nodes_expanded else last

    # Objective rung: induced relational objectives over the local sim
    # (proven load-bearing on cn04/ka59/dc22 in the P13-P17 line).
    if _remaining() < 5.0:
        last.rungs = list(trail)
        return last
    try:
        obj_res = objective_astar(
            sim, level_idx, frame0,
            budget_s=min(budget_ladder[-1], _remaining()), max_depth=max_depth,
        )
    except Exception as exc:
        obj_res = SimSearchResult(found=False,
                                  reason=f"objective_error:{type(exc).__name__}")
    trail.append(("objective", obj_res.reason, obj_res.nodes_expanded,
                  round(obj_res.elapsed, 1)))
    if obj_res.found:
        obj_res.rungs = list(trail)
        return obj_res

    # Dense rescan: high-recall click lattice, dedup off.
    if _remaining() < 5.0:
        last.rungs = list(trail)
        return last
    dense_res = solve_level(
        sim, level_idx, frame0,
        budget_s=min(budget_ladder[0], _remaining()),
        max_states=max_states, max_depth=max_depth,
        dedup=False, dense_non_bg=True, max_sites=1200, scan_timeout=20.0,
    )
    trail.append(("dense_rescan", dense_res.reason, dense_res.nodes_expanded,
                  round(dense_res.elapsed, 1)))
    if dense_res.found:
        dense_res.reason = "dense_rescan"
        dense_res.rungs = list(trail)
        return dense_res
    last.rungs = list(trail)
    return last


def solve_game(
    game_id: str,
    source_path: str | None = None,
    class_name: str | None = None,
    max_levels: int = 12,
    per_level_budget_s: float = 60.0,
    max_states: int = 200_000,
    max_depth: int = 40,
    arc_env: object | None = None,
) -> dict:
    """Solve consecutive levels of ``game_id`` on the local sim.

    Returns ``{"levels": {idx: SimSearchResult}, "plan": [...specs...],
    "source": path}``. ``plan`` is the concatenation of solved-level plans
    starting at level 0, suitable for real-env playback; it stops at the
    first unsolved level.
    """
    if source_path is None or class_name is None:
        source_path, class_name = find_game_source(game_id, arc_env=arc_env)
    if source_path is None or class_name is None:
        return {"levels": {}, "plan": [], "source": None, "reason": "source_not_found"}

    game_cls = load_game_class(source_path, class_name)
    sim = LocalSim(game_cls)
    results: dict[int, SimSearchResult] = {}
    plan: list[tuple] = []
    prev_plan: list[tuple] | None = None
    for level_idx in range(max_levels):
        try:
            frame0 = sim.reset_level(level_idx)
        except Exception:
            break
        result = solve_level_cascade(
            sim,
            level_idx,
            frame0,
            prev_plan=prev_plan,
            budget_ladder=(per_level_budget_s,),
            max_states=max_states,
            max_depth=max_depth,
            source_path=source_path,
        )
        results[level_idx] = result
        if not result.found:
            break
        prev_plan = list(result.plan)
        plan.extend(result.plan)
    return {"levels": results, "plan": plan, "source": source_path,
            "reason": "ok" if plan else "level0_unsolved"}


__all__ = ["SimSearchResult", "scan_actions", "solve_level", "solve_game"]
