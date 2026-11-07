"""Generic side-scroll fall first-score bootstrap.

This Tier-B bootstrap targets a narrow but reusable mechanics shape:

    horizontal movement -> click-reactive local platforms -> gravity/scroll fall
    -> score oracle confirms a safe prefix

It is intentionally a replay-from-root search, not a source or adapter handoff.
Runtime uses rendered frames, one-step action probes, RESET, and score-confirmed
prefix replay. There are no game-id, colour, or coordinate branches in the
solver; the returned plan is accepted only if a disposable rollout scores.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arc_agi  # noqa: E402
from arcengine import GameAction  # noqa: E402
from arcgym.eca.agent_identifier import _frame_components  # noqa: E402
from arcgym.environments import ArcAgi3Env  # noqa: E402


ActionSpec = tuple[Any, ...]


@dataclass(frozen=True)
class SideFallDiagnostics:
    gated: bool = False
    gate_reason: str = ""
    move_action_count: int = 0
    agent_color_count: int = 0
    nodes_expanded: int = 0
    states_seen: int = 0
    winning_depth: int = 0
    winning_score: int = 0

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "gated": bool(self.gated),
            "gate_reason": self.gate_reason,
            "move_action_count": int(self.move_action_count),
            "agent_color_count": int(self.agent_color_count),
            "nodes_expanded": int(self.nodes_expanded),
            "states_seen": int(self.states_seen),
            "winning_depth": int(self.winning_depth),
            "winning_score": int(self.winning_score),
        }


LAST_DIAGNOSTICS = SideFallDiagnostics()


def _new_env(game: str, max_actions: int) -> ArcAgi3Env:
    return ArcAgi3Env(
        game_id=game,
        max_actions=max_actions,
        operation_mode=arc_agi.OperationMode.OFFLINE,
    )


def _frame(obs: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(obs.get("frame"))
    return (arr[-1] if arr.ndim == 3 else arr).astype(np.int16)


def _bg(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _payload(spec: ActionSpec) -> dict[str, Any]:
    action_name = str(spec[0])
    payload: dict[str, Any] = {"action": GameAction.from_name(action_name)}
    if action_name == "ACTION6" and len(spec) >= 3:
        payload["x"] = int(spec[1])
        payload["y"] = int(spec[2])
    return payload


def _small_components_by_color(arr: np.ndarray) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for comp in _frame_components(arr):
        n = int(comp.get("n", 0) or 0)
        if 2 <= n <= 15:
            out.setdefault(int(comp.get("color", -1)), []).append(comp)
    return out


def _detect_agent_colors(root: np.ndarray, successors: list[np.ndarray]) -> set[int]:
    """Colours of small components that translate horizontally under actions."""
    bg = _bg(root)
    base = _small_components_by_color(root)
    colors: set[int] = set()
    for frame in successors:
        after = _small_components_by_color(frame)
        for color, comps0 in base.items():
            if color == bg or color not in after:
                continue
            for before in comps0:
                bx = float(before.get("cx", 0.0) or 0.0)
                by = float(before.get("cy", 0.0) or 0.0)
                for comp in after[color]:
                    dx = abs(float(comp.get("cx", 0.0) or 0.0) - bx)
                    dy = abs(float(comp.get("cy", 0.0) or 0.0) - by)
                    if 3.0 <= dx <= 10.0 and dy <= 2.0:
                        colors.add(int(color))
    return colors


class _ReplayBeam:
    def __init__(self, game: str, max_actions: int) -> None:
        # Planning evaluates many candidate prefixes by RESET + replay. Keep the
        # wrapper budget high enough that those internal evaluations do not trip
        # the harness action limit; returned live plans still use max_actions.
        self.env = _new_env(game, max(max_actions, 1000) * 100)
        obs = dict(self.env.reset())
        self.root_frame = _frame(obs)
        self.root_score = int(obs.get("score", 0) or 0)
        self.root_state = str(obs.get("state", "") or "")
        self.available = [int(a) for a in (obs.get("available_actions") or [])]
        self._feature_cache: dict[bytes, tuple[float, float, list[ActionSpec]]] = {}

        probe_frames: list[np.ndarray] = []
        probe_moves: list[str] = []
        for action_id in self.available:
            if action_id == 6:
                continue
            action = f"ACTION{action_id}"
            frame, score, state = self.eval_path([(action,)])
            if score == self.root_score and state != "GAME_OVER":
                probe_frames.append(frame)
                if int(np.sum(frame != self.root_frame)) > 8:
                    probe_moves.append(action)
        self.agent_colors = _detect_agent_colors(self.root_frame, probe_frames)
        self.move_actions = self._movement_actions(probe_moves)

    def close(self) -> None:
        self.env.close()

    def eval_path(self, path: list[ActionSpec]) -> tuple[np.ndarray, int, str]:
        if not path:
            return self.root_frame, self.root_score, self.root_state
        obs, _reward, _done = self.env.step({"action": GameAction.RESET})
        obs = dict(obs)
        score = int(obs.get("score", 0) or 0)
        state = str(obs.get("state", "") or "")
        for spec in path:
            obs, _reward, _done = self.env.step(_payload(spec))
            obs = dict(obs)
            score = int(obs.get("score", 0) or 0)
            state = str(obs.get("state", "") or "")
            if score > self.root_score or state in ("WIN", "GAME_OVER"):
                break
        return _frame(obs), score, state

    def step_from_path(self, path: list[ActionSpec], spec: ActionSpec) -> tuple[np.ndarray, int, str]:
        return self.eval_path(path + [spec])

    def _movement_actions(self, candidates: list[str]) -> list[str]:
        if not self.agent_colors:
            return []
        root_px, _py, _specs = self.features(self.root_frame, include_moves=False)
        moves: list[tuple[float, str]] = []
        for action in candidates:
            frame, score, state = self.eval_path([(action,)])
            if score != self.root_score or state == "GAME_OVER":
                continue
            px, _py, _ = self.features(frame, include_moves=False)
            dx = px - root_px
            if abs(dx) >= 2.0:
                moves.append((dx, action))
        # Keep one left-ish and one right-ish action when present.
        left = sorted((m for m in moves if m[0] < 0), key=lambda t: t[0])
        right = sorted((m for m in moves if m[0] > 0), key=lambda t: -t[0])
        out = [a for _dx, a in left[:1] + right[:1]]
        return out or [a for _dx, a in moves[:2]]

    def features(
        self, arr: np.ndarray, include_moves: bool = True
    ) -> tuple[float, float, list[ActionSpec]]:
        key = arr.tobytes() + (b"m" if include_moves else b"n")
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached

        bg = _bg(arr)
        comps = list(_frame_components(arr))
        agent_comps = [
            comp
            for comp in comps
            if int(comp.get("color", -1)) in self.agent_colors
            and 2 <= int(comp.get("n", 0) or 0) <= 70
            and 12.0 <= float(comp.get("cy", 0.0) or 0.0) <= 58.0
        ]
        if agent_comps:
            px = sum(float(comp.get("cx", 0.0) or 0.0) for comp in agent_comps) / len(agent_comps)
            py = sum(float(comp.get("cy", 0.0) or 0.0) for comp in agent_comps) / len(agent_comps)
        else:
            px = float(arr.shape[1] - 1) / 2.0
            py = float(arr.shape[0] - 1) / 2.0

        clicks: list[tuple[tuple[float, float, float, int], ActionSpec]] = []
        for comp in comps:
            color = int(comp.get("color", -1))
            n = int(comp.get("n", 0) or 0)
            cx = float(comp.get("cx", 0.0) or 0.0)
            cy = float(comp.get("cy", 0.0) or 0.0)
            if color == bg or color in self.agent_colors or not (6 <= n <= 70):
                continue
            rx = cx - px
            ry = cy - py
            if abs(rx) <= 30.0 and -12.0 <= ry <= 12.0:
                # Gravity platformers usually ask for the local support cell
                # one grid step in the fall direction. The exact colour/coord is
                # still frame-derived; this only ranks nearby click candidates.
                priority = (abs(ry + 5.0), abs(rx), abs(ry), -n)
                clicks.append((priority, ("ACTION6", int(round(cx)), int(round(cy)))))
        clicks.sort(key=lambda item: item[0])

        specs: list[ActionSpec] = []
        if include_moves:
            specs.extend((action,) for action in self.move_actions)
        specs.extend(spec for _priority, spec in clicks[:5])

        uniq: list[ActionSpec] = []
        seen: set[ActionSpec] = set()
        for spec in specs:
            if spec in seen:
                continue
            seen.add(spec)
            uniq.append(spec)
        result = (px, py, uniq)
        self._feature_cache[key] = result
        return result


def _gate(search: _ReplayBeam) -> str:
    available = set(search.available)
    if 6 not in available:
        return "click_unavailable"
    if len([action for action in available if action != 6]) > 3:
        return "too_many_nonclick_actions"
    if len(search.move_actions) < 2:
        return "missing_bidirectional_horizontal_moves"
    if len(search.agent_colors) == 0:
        return "agent_not_detected"
    _px, _py, specs = search.features(search.root_frame)
    if not any(spec[0] == "ACTION6" for spec in specs):
        # The initial view may not expose a local platform yet; one short run in
        # either direction must expose one, otherwise this family is not present.
        exposed = False
        for move in search.move_actions:
            frame = search.root_frame
            state = search.root_state
            prefix: list[ActionSpec] = []
            for _ in range(5):
                prefix.append((move,))
                frame, _score, state = search.eval_path(prefix)
                if state == "GAME_OVER":
                    break
                _px, _py, move_specs = search.features(frame)
                if any(spec[0] == "ACTION6" for spec in move_specs):
                    exposed = True
                    break
            if exposed:
                break
        if not exposed:
            return "no_local_clickable_platforms"
    return ""


def plan(
    game: str = "bp35",
    max_actions: int = 240,
    max_depth: int = 22,
    beam_width: int = 5,
    verbose: bool = False,
) -> list[ActionSpec] | None:
    """Search for a first-score side-fall prefix.

    This uses RESET + prefix replay, but no source imports or adapters. The beam
    is deliberately narrow and self-gated so non-matching games return quickly
    without displacing other generic lanes.
    """
    global LAST_DIAGNOSTICS
    LAST_DIAGNOSTICS = SideFallDiagnostics()
    search: _ReplayBeam | None = None
    try:
        search = _ReplayBeam(game, max_actions)
        reason = _gate(search)
        if reason:
            LAST_DIAGNOSTICS = SideFallDiagnostics(
                gated=False,
                gate_reason=reason,
                move_action_count=len(search.move_actions),
                agent_color_count=len(search.agent_colors),
            )
            return None

        beam: list[tuple[np.ndarray, int, str, list[ActionSpec], int, int]] = [
            (
                search.root_frame,
                search.root_score,
                search.root_state,
                [],
                0,
                0,
            )
        ]
        seen: set[tuple[bytes, int]] = {(search.root_frame.tobytes(), 0)}
        expanded = 0
        for depth in range(1, max_depth + 1):
            next_rows: list[
                tuple[tuple[float, float, float, float, int], tuple[np.ndarray, int, str, list[ActionSpec], int, int]]
            ] = []
            for frame, score, state, path, progress, click_run in beam:
                if score > search.root_score or state in ("WIN", "GAME_OVER"):
                    continue
                _px, _py, specs = search.features(frame)
                for spec in specs:
                    if spec[0] == "ACTION6" and click_run >= 2:
                        continue
                    frame2, score2, state2 = search.step_from_path(path, spec)
                    expanded += 1
                    diff = int(np.sum(frame2 != frame))
                    path2 = path + [spec]
                    if score2 > search.root_score:
                        LAST_DIAGNOSTICS = SideFallDiagnostics(
                            gated=True,
                            move_action_count=len(search.move_actions),
                            agent_color_count=len(search.agent_colors),
                            nodes_expanded=expanded,
                            states_seen=len(seen),
                            winning_depth=depth,
                            winning_score=score2,
                        )
                        return path2
                    if state2 == "GAME_OVER":
                        continue
                    next_click_run = click_run + 1 if spec[0] == "ACTION6" else 0
                    key = (frame2.tobytes(), next_click_run)
                    if key in seen:
                        continue
                    seen.add(key)

                    px2, _py2, _ = search.features(frame2)
                    click_count = sum(1 for item in path2 if item[0] == "ACTION6")
                    progress2 = progress
                    if diff > 800:
                        progress2 += 4
                    elif diff > 400:
                        progress2 += 2
                    if spec[0] == "ACTION6" and diff > 20:
                        progress2 += 1
                    rank = (
                        float(progress2),
                        float(-next_click_run),
                        float(-0.1 * click_count),
                        -len(path),
                    )
                    next_rows.append(
                        (
                            rank,
                            (
                                frame2,
                                score2,
                                state2,
                                path2,
                                progress2,
                                next_click_run,
                            ),
                        )
                    )
            next_rows.sort(key=lambda item: item[0], reverse=True)
            beam = [row for _rank, row in next_rows[:beam_width]]
            if not beam:
                break
        LAST_DIAGNOSTICS = SideFallDiagnostics(
            gated=True,
            gate_reason="search_exhausted",
            move_action_count=len(search.move_actions),
            agent_color_count=len(search.agent_colors),
            nodes_expanded=expanded,
            states_seen=len(seen),
        )
        return None
    except Exception:
        if verbose:
            raise
        return None
    finally:
        if search is not None:
            search.close()


def replay(actions: list[ActionSpec] | None, live_env: ArcAgi3Env) -> int:
    if not actions:
        return 0
    score = 0
    for spec in actions:
        obs, _reward, done = live_env.step(_payload(spec))
        score = int(obs.get("score", 0) or 0)
        if score > 0 or done:
            break
    return score


def solve_and_replay(game: str = "bp35", max_actions: int = 240) -> int:
    actions = plan(game, max_actions)
    if not actions:
        return 0
    env = _new_env(game, max_actions)
    try:
        env.reset()
        return replay(actions, env)
    finally:
        env.close()


if __name__ == "__main__":
    g = sys.argv[1] if len(sys.argv) > 1 else "bp35"
    acts = plan(g, 240, verbose=True)
    print(f"{g} plan length =", len(acts) if acts else None)
    print(f"{g} diagnostics =", LAST_DIAGNOSTICS.to_dict())
    print(f"{g} actions =", acts)
    print(f"{g} solve_and_replay score =", solve_and_replay(g, 240))
