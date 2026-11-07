"""ActorIdentityAnchor: pin the actor in a multi-mover scene (Phase M-2c).

M-2b found that on a live multi-mover game (ls20: walkers + actor +
animations) the MovingSpriteTracker's protected matching correctly refuses
to guess which moving sprite is the actor — so 0 clean moves are reported.
To feed the self-repair learners a clean actor transition stream, the actor
must be identified ONCE and then tracked by its identity through the noise.

This module accumulates evidence across steps and commits to an actor
identity using two game-agnostic signals:

  - the actor moves FREQUENTLY (the agent is trying to move it every step),
  - the actor is a SINGLE instance of its identity (colour + shape + area),
    whereas walkers usually come in interchangeable multiples.

Once an identity has moved on enough steps AND stayed single-instance, it is
committed; thereafter `actor_identity` returns it and the caller passes it to
`primary_actor_move(actor_identity=...)`. Committing is conservative: it
waits for a margin over the runner-up so a transient walker burst does not
capture the anchor. ZERO game-specific code; identities are the same
palette/scale-free tuples the tracker uses.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .moving_sprite_tracker import detect_sprite_moves

ACTOR_IDENTITY_ANCHOR_SENSOR_SCHEMA_VERSION = (
    "eca_actor_identity_anchor_sensor.v1"
)
ACTOR_IDENTITY_ANCHOR_POSTERIOR_SCHEMA_VERSION = (
    "eca_actor_identity_anchor_posterior.v1"
)
ACTOR_IDENTITY_ANCHOR_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION = (
    "eca_actor_identity_anchor_guarded_planner_connection.v1"
)


class ActorIdentityAnchor:
    """Commits to the actor's sprite identity from action-driven movement."""

    def __init__(
        self,
        *,
        min_moves: int = 6,
        commit_margin: int = 3,
        max_displacement: int | None = 40,
    ) -> None:
        # An identity must move on at least this many steps before it can be
        # committed as the actor.
        self._min_moves = max(1, int(min_moves))
        # And it must lead the runner-up identity by this many move-steps, so
        # a brief walker burst cannot capture the anchor.
        self._commit_margin = max(0, int(commit_margin))
        self._max_displacement = max_displacement
        # Per-identity count of steps on which it was the UNIQUE mover (the
        # strongest actor evidence) and of steps it moved at all.
        self._solo_moves: Counter[tuple] = Counter()
        self._any_moves: Counter[tuple] = Counter()
        self._committed: tuple | None = None

    def observe(self, prev_frame: np.ndarray, cur_frame: np.ndarray) -> None:
        """Accumulate movement evidence from one frame transition."""
        if self._committed is not None:
            return
        moves = detect_sprite_moves(
            prev_frame, cur_frame, cell=1, max_displacement=self._max_displacement
        )
        if not moves:
            return
        for m in moves:
            self._any_moves[m.identity] += 1
        if len(moves) == 1:
            # The unique mover this step is the strongest actor signal.
            self._solo_moves[moves[0].identity] += 1
        self._maybe_commit()

    def _maybe_commit(self) -> None:
        # Prefer solo-move evidence; fall back to any-move frequency.
        ranking = self._solo_moves if self._solo_moves else self._any_moves
        if not ranking:
            return
        top = ranking.most_common(2)
        best_id, best_n = top[0]
        if best_n < self._min_moves:
            return
        runner_n = top[1][1] if len(top) > 1 else 0
        if best_n - runner_n >= self._commit_margin:
            self._committed = best_id

    @property
    def actor_identity(self) -> tuple | None:
        return self._committed

    def is_committed(self) -> bool:
        return self._committed is not None

    def summary(self) -> dict[str, Any]:
        return {
            "committed": self._committed,
            "solo_leaders": self._solo_moves.most_common(3),
            "any_leaders": self._any_moves.most_common(3),
        }


def summarize_actor_identity_anchor_packets(
    packets: Any,
    *,
    min_moves: int = 6,
    commit_margin: int = 3,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize actor identity evidence from moving-sprite runtime packets.

    This is diagnostic-only glue between Phase671 runtime packets and the
    actor-anchor stage. It does not select actions or choose a planner.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, dict)]
    by_game: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    total_packets = 0
    total_move_packets = 0
    total_solo_packets = 0
    total_ambiguous_packets = 0
    total_clean_actor_packets = 0

    for packet in packet_rows:
        total_packets += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(
            game_id,
            {
                "packet_count": 0,
                "packet_with_moves_count": 0,
                "solo_move_packet_count": 0,
                "multi_move_packet_count": 0,
                "ambiguous_identity_packet_count": 0,
                "score_closure_packet_count": 0,
                "any_moves": Counter(),
                "solo_moves": Counter(),
                "score_closure_moves": Counter(),
            },
        )
        state["packet_count"] += 1
        move_rows = [
            move for move in list(packet.get("moves") or []) if isinstance(move, dict)
        ]
        if move_rows:
            state["packet_with_moves_count"] += 1
            total_move_packets += 1
        if len(move_rows) == 1:
            state["solo_move_packet_count"] += 1
            total_solo_packets += 1
        elif len(move_rows) > 1:
            state["multi_move_packet_count"] += 1
        if int(packet.get("ambiguous_identity_count", 0) or 0) > 0:
            state["ambiguous_identity_packet_count"] += 1
            total_ambiguous_packets += 1
        score_closure = bool(packet.get("score_closure_candidate"))
        if score_closure:
            state["score_closure_packet_count"] += 1
        for move in move_rows:
            key = _identity_key_from_move(move)
            if not key:
                continue
            state["any_moves"][key] += 1
            if len(move_rows) == 1:
                state["solo_moves"][key] += 1
            if score_closure:
                state["score_closure_moves"][key] += 1

    rows: list[dict[str, Any]] = []
    committed_game_count = 0
    committed_identity_count = 0
    for game_id, state in sorted(by_game.items()):
        ranking = (
            state["solo_moves"]
            if len(state["solo_moves"]) > 0
            else state["any_moves"]
        )
        top = ranking.most_common(2)
        committed_key: tuple[Any, ...] | None = None
        committed_reason = "insufficient_identity_evidence"
        if top:
            best_key, best_n = top[0]
            runner_n = top[1][1] if len(top) > 1 else 0
            if best_n >= max(1, int(min_moves)) and (
                best_n - runner_n >= max(0, int(commit_margin))
            ):
                committed_key = best_key
                committed_reason = "committed_by_frequency_margin"
            else:
                committed_reason = "below_min_moves_or_margin"
        clean_actor_packets = 0
        if committed_key is not None:
            committed_game_count += 1
            committed_identity_count += 1
            for packet in packet_rows:
                if str(packet.get("game_id", "") or "unknown") != game_id:
                    continue
                for move in list(packet.get("moves") or []):
                    if isinstance(move, dict) and _identity_key_from_move(move) == committed_key:
                        clean_actor_packets += 1
                        break
        total_clean_actor_packets += clean_actor_packets
        row = {
            "game_id": game_id,
            "packet_count": int(state["packet_count"]),
            "packet_with_moves_count": int(state["packet_with_moves_count"]),
            "solo_move_packet_count": int(state["solo_move_packet_count"]),
            "multi_move_packet_count": int(state["multi_move_packet_count"]),
            "ambiguous_identity_packet_count": int(
                state["ambiguous_identity_packet_count"]
            ),
            "score_closure_packet_count": int(state["score_closure_packet_count"]),
            "committed": committed_key is not None,
            "commit_reason": committed_reason,
            "committed_identity": (
                _identity_key_to_packet(committed_key)
                if committed_key is not None
                else {}
            ),
            "clean_actor_trajectory_packet_count": int(clean_actor_packets),
            "solo_leaders": _leader_rows(state["solo_moves"]),
            "any_leaders": _leader_rows(state["any_moves"]),
            "score_closure_leaders": _leader_rows(state["score_closure_moves"]),
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if committed_game_count:
        promotion_state = "actor_identity_anchor_connection_observed"
    elif total_move_packets:
        promotion_state = "actor_identity_anchor_connection_no_commit"
    elif total_packets:
        promotion_state = "actor_identity_anchor_connection_no_moves"
    else:
        promotion_state = "actor_identity_anchor_connection_no_packets"

    return {
        "schema_version": ACTOR_IDENTITY_ANCHOR_SENSOR_SCHEMA_VERSION,
        "diagnostic_only": True,
        "packet_count": int(total_packets),
        "packet_with_moves_count": int(total_move_packets),
        "solo_move_packet_count": int(total_solo_packets),
        "ambiguous_identity_packet_count": int(total_ambiguous_packets),
        "committed_game_count": int(committed_game_count),
        "committed_identity_count": int(committed_identity_count),
        "clean_actor_trajectory_packet_count": int(total_clean_actor_packets),
        "promotion_state": promotion_state,
        "actor_identity_anchor_connection_observed": committed_game_count > 0,
        "min_moves": max(1, int(min_moves)),
        "commit_margin": max(0, int(commit_margin)),
        "by_game": rows,
        "samples": samples,
        "planner_connection_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
    }


def summarize_actor_identity_anchor_posterior(
    packets: Any,
    *,
    min_moves: int = 6,
    commit_margin: int = 3,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Build a diagnostic posterior over actor identity anchors.

    The posterior consumes only observed moving-sprite runtime packets. It does
    not choose actions, rank candidates, or connect to a planner.
    """

    anchor = summarize_actor_identity_anchor_packets(
        packets,
        min_moves=min_moves,
        commit_margin=commit_margin,
        sample_limit=sample_limit,
    )
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    confirmed_count = 0
    score_supported_count = 0
    score_conflict_count = 0
    ambiguous_count = 0
    insufficient_count = 0
    no_motion_count = 0
    clean_packet_count = 0

    for anchor_row in list(anchor.get("by_game") or []):
        if not isinstance(anchor_row, dict):
            continue
        leaders = list(anchor_row.get("solo_leaders") or [])
        evidence_source = "solo_moves"
        if not leaders:
            leaders = list(anchor_row.get("any_leaders") or [])
            evidence_source = "any_moves"
        best_count = int(leaders[0].get("count", 0) or 0) if leaders else 0
        runner_count = int(leaders[1].get("count", 0) or 0) if len(leaders) > 1 else 0
        evidence_margin = best_count - runner_count
        committed_identity = (
            dict(anchor_row.get("committed_identity") or {})
            if bool(anchor_row.get("committed"))
            else {}
        )
        score_leaders = [
            row for row in list(anchor_row.get("score_closure_leaders") or [])
            if isinstance(row, dict)
        ]
        score_identity = (
            dict(score_leaders[0].get("identity") or {})
            if score_leaders
            else {}
        )
        score_support_count = (
            int(score_leaders[0].get("count", 0) or 0)
            if score_leaders
            else 0
        )
        score_closure_supported = (
            bool(committed_identity)
            and bool(score_identity)
            and _identity_packet_key(score_identity)
            == _identity_packet_key(committed_identity)
        )
        score_closure_conflict = (
            bool(committed_identity)
            and bool(score_identity)
            and _identity_packet_key(score_identity)
            != _identity_packet_key(committed_identity)
        )

        packet_with_moves = int(anchor_row.get("packet_with_moves_count", 0) or 0)
        if score_closure_conflict:
            posterior_state = "ambiguous_score_closure_conflict"
            score_conflict_count += 1
            ambiguous_count += 1
        elif bool(committed_identity) and score_closure_supported:
            posterior_state = "confirmed_score_closure_supported"
            confirmed_count += 1
            score_supported_count += 1
        elif bool(committed_identity):
            posterior_state = "confirmed_motion_only"
            confirmed_count += 1
        elif packet_with_moves <= 0:
            posterior_state = "no_motion_evidence"
            no_motion_count += 1
        elif best_count < max(1, int(min_moves)):
            posterior_state = "insufficient_evidence"
            insufficient_count += 1
        else:
            posterior_state = "ambiguous_identity_margin"
            ambiguous_count += 1

        clean_packets = int(anchor_row.get("clean_actor_trajectory_packet_count", 0) or 0)
        clean_packet_count += clean_packets
        row = {
            "game_id": str(anchor_row.get("game_id", "") or "unknown"),
            "posterior_state": posterior_state,
            "committed_identity": committed_identity,
            "score_closure_identity": score_identity,
            "score_closure_supported": score_closure_supported,
            "score_closure_conflict": score_closure_conflict,
            "evidence_source": evidence_source,
            "evidence_count": best_count,
            "runner_up_count": runner_count,
            "evidence_margin": evidence_margin,
            "packet_count": int(anchor_row.get("packet_count", 0) or 0),
            "packet_with_moves_count": packet_with_moves,
            "solo_move_packet_count": int(anchor_row.get("solo_move_packet_count", 0) or 0),
            "multi_move_packet_count": int(anchor_row.get("multi_move_packet_count", 0) or 0),
            "ambiguous_identity_packet_count": int(
                anchor_row.get("ambiguous_identity_packet_count", 0) or 0
            ),
            "score_closure_packet_count": int(
                anchor_row.get("score_closure_packet_count", 0) or 0
            ),
            "score_support_count": score_support_count,
            "clean_actor_trajectory_packet_count": clean_packets,
            "commit_reason": str(anchor_row.get("commit_reason", "") or ""),
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if score_supported_count:
        promotion_state = "actor_identity_anchor_posterior_score_supported"
    elif confirmed_count:
        promotion_state = "actor_identity_anchor_posterior_motion_confirmed"
    elif ambiguous_count:
        promotion_state = "actor_identity_anchor_posterior_ambiguous"
    elif insufficient_count:
        promotion_state = "actor_identity_anchor_posterior_insufficient_evidence"
    elif no_motion_count:
        promotion_state = "actor_identity_anchor_posterior_no_motion"
    else:
        promotion_state = "actor_identity_anchor_posterior_no_packets"

    return {
        "schema_version": ACTOR_IDENTITY_ANCHOR_POSTERIOR_SCHEMA_VERSION,
        "diagnostic_only": True,
        "shadow_only": True,
        "packet_count": int(anchor.get("packet_count", 0) or 0),
        "packet_with_moves_count": int(anchor.get("packet_with_moves_count", 0) or 0),
        "posterior_entry_count": len(rows),
        "confirmed_identity_game_count": int(confirmed_count),
        "score_supported_identity_game_count": int(score_supported_count),
        "score_conflict_game_count": int(score_conflict_count),
        "ambiguous_identity_game_count": int(ambiguous_count),
        "insufficient_evidence_game_count": int(insufficient_count),
        "no_motion_evidence_game_count": int(no_motion_count),
        "clean_actor_trajectory_packet_count": int(clean_packet_count),
        "connection_promotion_state": str(anchor.get("promotion_state", "") or ""),
        "promotion_state": promotion_state,
        "min_moves": max(1, int(min_moves)),
        "commit_margin": max(0, int(commit_margin)),
        "by_game": rows,
        "samples": samples,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
    }


def summarize_actor_identity_anchor_guarded_planner_connection(
    packets: Any,
    *,
    min_moves: int = 6,
    commit_margin: int = 3,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Shadow admission guard for actor-identity planner connection.

    A motion-only actor identity is not sufficient. The guard admits only when
    the same identity is both motion-confirmed and score-closure-supported.
    """

    posterior = summarize_actor_identity_anchor_posterior(
        packets,
        min_moves=min_moves,
        commit_margin=commit_margin,
        sample_limit=sample_limit,
    )
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    admitted_count = 0
    blocked_count = 0
    missing_score_support_count = 0
    score_conflict_count = 0
    identity_ambiguity_count = 0
    insufficient_count = 0
    no_motion_count = 0

    for posterior_row in list(posterior.get("by_game") or []):
        if not isinstance(posterior_row, dict):
            continue
        state = str(posterior_row.get("posterior_state", "") or "")
        blocked_reasons: list[str] = []
        admitted = state == "confirmed_score_closure_supported"
        if admitted:
            connection_state = "actor_identity_planner_connection_shadow_admitted"
            admitted_count += 1
        else:
            blocked_count += 1
            if state == "confirmed_motion_only":
                blocked_reasons.append("missing_score_closure_support")
                missing_score_support_count += 1
            elif state == "ambiguous_score_closure_conflict":
                blocked_reasons.append("score_closure_conflict")
                score_conflict_count += 1
            elif state == "ambiguous_identity_margin":
                blocked_reasons.append("identity_margin_ambiguous")
                identity_ambiguity_count += 1
            elif state == "insufficient_evidence":
                blocked_reasons.append("insufficient_identity_evidence")
                insufficient_count += 1
            elif state == "no_motion_evidence":
                blocked_reasons.append("no_motion_evidence")
                no_motion_count += 1
            else:
                blocked_reasons.append("not_score_supported")
            connection_state = (
                "actor_identity_planner_connection_blocked_"
                + blocked_reasons[0]
            )
        row = {
            "game_id": str(posterior_row.get("game_id", "") or "unknown"),
            "connection_state": connection_state,
            "admitted_shadow": admitted,
            "blocked_reasons": blocked_reasons,
            "posterior_state": state,
            "committed_identity": dict(
                posterior_row.get("committed_identity") or {}
            ),
            "score_closure_identity": dict(
                posterior_row.get("score_closure_identity") or {}
            ),
            "score_closure_supported": bool(
                posterior_row.get("score_closure_supported", False)
            ),
            "score_closure_conflict": bool(
                posterior_row.get("score_closure_conflict", False)
            ),
            "evidence_count": int(posterior_row.get("evidence_count", 0) or 0),
            "runner_up_count": int(posterior_row.get("runner_up_count", 0) or 0),
            "evidence_margin": int(posterior_row.get("evidence_margin", 0) or 0),
            "clean_actor_trajectory_packet_count": int(
                posterior_row.get("clean_actor_trajectory_packet_count", 0) or 0
            ),
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if admitted_count:
        promotion_state = "actor_identity_guarded_planner_connection_shadow_admitted"
    elif score_conflict_count:
        promotion_state = "actor_identity_guarded_planner_connection_blocked_score_conflict"
    elif missing_score_support_count:
        promotion_state = "actor_identity_guarded_planner_connection_blocked_missing_score_support"
    elif identity_ambiguity_count:
        promotion_state = "actor_identity_guarded_planner_connection_blocked_ambiguous_identity"
    elif insufficient_count:
        promotion_state = "actor_identity_guarded_planner_connection_blocked_insufficient_evidence"
    elif no_motion_count:
        promotion_state = "actor_identity_guarded_planner_connection_blocked_no_motion"
    else:
        promotion_state = "actor_identity_guarded_planner_connection_no_packets"

    return {
        "schema_version": (
            ACTOR_IDENTITY_ANCHOR_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION
        ),
        "diagnostic_only": True,
        "shadow_only": True,
        "packet_count": int(posterior.get("packet_count", 0) or 0),
        "posterior_entry_count": int(posterior.get("posterior_entry_count", 0) or 0),
        "planner_connection_admitted_game_count": int(admitted_count),
        "planner_connection_blocked_game_count": int(blocked_count),
        "blocked_missing_score_support_game_count": int(missing_score_support_count),
        "blocked_score_conflict_game_count": int(score_conflict_count),
        "blocked_identity_ambiguity_game_count": int(identity_ambiguity_count),
        "blocked_insufficient_evidence_game_count": int(insufficient_count),
        "blocked_no_motion_evidence_game_count": int(no_motion_count),
        "posterior_promotion_state": str(
            posterior.get("promotion_state", "") or ""
        ),
        "promotion_state": promotion_state,
        "by_game": rows,
        "samples": samples,
        "admission_guard_enabled": True,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
    }


def _identity_key_from_move(move: dict[str, Any]) -> tuple[Any, ...]:
    identity = move.get("identity")
    if not isinstance(identity, dict):
        return ()
    shape = tuple(int(v) for v in list(identity.get("shape_signature") or []))
    return (
        int(identity.get("dominant_color", 0) or 0),
        shape,
        int(identity.get("area", 0) or 0),
    )


def _identity_key_to_packet(key: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "dominant_color": int(key[0]),
        "shape_signature": [int(v) for v in tuple(key[1])],
        "area": int(key[2]),
    }


def _identity_packet_key(identity: dict[str, Any]) -> tuple[Any, ...]:
    shape = tuple(int(v) for v in list(identity.get("shape_signature") or []))
    return (
        int(identity.get("dominant_color", 0) or 0),
        shape,
        int(identity.get("area", 0) or 0),
    )


def _leader_rows(counter: Counter[tuple[Any, ...]], limit: int = 3) -> list[dict[str, Any]]:
    return [
        {"identity": _identity_key_to_packet(key), "count": int(count)}
        for key, count in counter.most_common(limit)
    ]


__all__ = [
    "ACTOR_IDENTITY_ANCHOR_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION",
    "ACTOR_IDENTITY_ANCHOR_SENSOR_SCHEMA_VERSION",
    "ACTOR_IDENTITY_ANCHOR_POSTERIOR_SCHEMA_VERSION",
    "ActorIdentityAnchor",
    "summarize_actor_identity_anchor_packets",
    "summarize_actor_identity_anchor_posterior",
    "summarize_actor_identity_anchor_guarded_planner_connection",
]
