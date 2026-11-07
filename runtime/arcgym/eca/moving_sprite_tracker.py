"""MovingSpriteTracker: the perception bridge for multi-cell-sprite games.

Phase M-2b (2026-05-31). The self-repair learners (TileEffectLearner,
TransportEffectLearner) consume LOGICAL actor positions: prev/observed cells
plus a predicted target. That was trivial for m0r0 (the cursor is a single
logical cell), but most action games render the actor as a MULTI-CELL sprite
(M-2a found ls20's actor spans ~26 cells), so raw per-cell diffing cannot
tell a sprite's normal move from a transport jump.

This module is the generic, game-agnostic bridge: given two consecutive
frames it identifies sprites that MOVED by matching connected components of
the SAME palette-independent identity (colour + shape signature + area)
across the frames, and reports each move as a displacement vector between
centroids. It owns no game knowledge and no forward model.

Protected matching (mirrors the learners' protected validation): a move is
reported ONLY when the identity match is UNAMBIGUOUS — exactly one component
of that identity in each frame. Ambiguous identities (e.g. several identical
walkers) are skipped rather than guessed, so a downstream learner never gets
a fabricated transition.

The same `extract_structural_components` the L-6 shadow audit uses provides
the palette/scale-independent component identities, so this bridge composes
with the certificate + shadow stack rather than introducing a new perception.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np

from .palette_invariant_perception import (
    StructuralComponent,
    extract_structural_components,
)

Vector = tuple[int, int]

MOVING_SPRITE_RUNTIME_TRACE_PACKET_SCHEMA_VERSION = (
    "eca_moving_sprite_runtime_trace_packet.v1"
)
MOVING_SPRITE_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION = (
    "eca_moving_sprite_runtime_trace_sensor.v1"
)
MOVING_SPRITE_MOTION_POSTERIOR_SCHEMA_VERSION = (
    "eca_moving_sprite_motion_posterior.v1"
)
MOVING_SPRITE_GUARDED_PLANNER_READINESS_SCHEMA_VERSION = (
    "eca_moving_sprite_guarded_planner_readiness.v1"
)
MOVING_SPRITE_ACTOR_AMBIGUITY_SPLITTER_SCHEMA_VERSION = (
    "eca_moving_sprite_actor_ambiguity_splitter.v1"
)
MOVING_SPRITE_GUARDED_ACTOR_ANCHOR_POSTERIOR_SCHEMA_VERSION = (
    "eca_moving_sprite_guarded_actor_anchor_posterior.v1"
)
MOVING_SPRITE_SOFT_ANCHOR_MOTION_POSTERIOR_SCHEMA_VERSION = (
    "eca_moving_sprite_soft_anchor_motion_posterior.v1"
)
MOVING_SPRITE_COUPLED_MOTION_SYMMETRY_SENSOR_SCHEMA_VERSION = (
    "eca_moving_sprite_coupled_motion_symmetry_sensor.v1"
)
MOVING_SPRITE_COUPLED_PAIR_GOAL_EFFECT_BRIDGE_SCHEMA_VERSION = (
    "eca_moving_sprite_coupled_pair_goal_effect_bridge.v1"
)
MOVING_SPRITE_COUPLED_PAIR_GOAL_SCORE_ATTRIBUTION_GUARD_SCHEMA_VERSION = (
    "eca_moving_sprite_coupled_pair_goal_score_attribution_guard.v1"
)
MOVING_SPRITE_TARGET_GOAL_COUNTERFACTUAL_ATTRIBUTION_SCHEMA_VERSION = (
    "eca_moving_sprite_target_goal_counterfactual_attribution.v1"
)
MOVING_SPRITE_TARGET_BOUND_EFFECT_COMPONENT_SENSOR_SCHEMA_VERSION = (
    "eca_moving_sprite_target_bound_effect_component_sensor.v1"
)
MOVING_SPRITE_EFFECT_COMPONENT_SPATIAL_STABILITY_SENSOR_SCHEMA_VERSION = (
    "eca_moving_sprite_effect_component_spatial_stability_sensor.v1"
)
MOVING_SPRITE_SPARSE_CHAIN_RAIL_STATE_SENSOR_SCHEMA_VERSION = (
    "eca_moving_sprite_sparse_chain_rail_state_sensor.v1"
)
MOVING_SPRITE_TARGET_GOAL_BINDING_STATE_SENSOR_SCHEMA_VERSION = (
    "eca_moving_sprite_target_goal_binding_state_sensor.v1"
)
MOVING_SPRITE_GOAL_ANCHOR_PERCEPTION_SENSOR_SCHEMA_VERSION = (
    "eca_moving_sprite_goal_anchor_perception_sensor.v1"
)
MOVING_SPRITE_GUARDED_PLANNER_CONNECTION_SENSOR_SCHEMA_VERSION = (
    "eca_moving_sprite_guarded_planner_connection_sensor.v1"
)


@dataclass(frozen=True)
class SpriteMove:
    """One sprite that moved between two frames (logical-cell units)."""

    identity: tuple  # (colour, shape_signature, area) — palette/scale free
    prev_cell: tuple[int, int]   # rounded prev centroid
    cur_cell: tuple[int, int]    # rounded current centroid
    displacement: Vector         # cur_cell - prev_cell
    area: int


def _identity(c: StructuralComponent) -> tuple:
    return (int(c.dominant_color), c.shape_signature, int(c.area))


def _by_identity(
    comps: list[StructuralComponent],
) -> dict[tuple, list[StructuralComponent]]:
    out: dict[tuple, list[StructuralComponent]] = {}
    for c in comps:
        out.setdefault(_identity(c), []).append(c)
    return out


def detect_sprite_moves(
    prev_frame: np.ndarray,
    cur_frame: np.ndarray,
    *,
    cell: int = 1,
    max_displacement: int | None = None,
) -> list[SpriteMove]:
    """Return the sprites that moved between two frames.

    A sprite is reported only when its identity (colour + shape + area) is
    UNAMBIGUOUS — present exactly once in each frame — and its centroid
    shifted. Ambiguous identities (duplicate walkers) and static sprites are
    skipped. `max_displacement` (if set) drops matches that jumped further
    than that many cells (likely a mis-match rather than a move/transport).
    """
    prev = _by_identity(extract_structural_components(prev_frame, cell=cell))
    cur = _by_identity(extract_structural_components(cur_frame, cell=cell))
    moves: list[SpriteMove] = []
    for identity, cur_list in cur.items():
        prev_list = prev.get(identity)
        if prev_list is None:
            continue
        # Protected: only unambiguous one-to-one identities.
        if len(cur_list) != 1 or len(prev_list) != 1:
            continue
        pc = prev_list[0]
        cc = cur_list[0]
        pcell = (round(pc.centroid[0]), round(pc.centroid[1]))
        ccell = (round(cc.centroid[0]), round(cc.centroid[1]))
        disp = (ccell[0] - pcell[0], ccell[1] - pcell[1])
        if disp == (0, 0):
            continue
        if max_displacement is not None and (
            abs(disp[0]) + abs(disp[1]) > max_displacement
        ):
            continue
        moves.append(
            SpriteMove(
                identity=identity,
                prev_cell=pcell,
                cur_cell=ccell,
                displacement=disp,
                area=int(cc.area),
            )
        )
    return moves


def primary_actor_move(
    prev_frame: np.ndarray,
    cur_frame: np.ndarray,
    *,
    cell: int = 1,
    actor_identity: tuple | None = None,
    max_displacement: int | None = None,
) -> SpriteMove | None:
    """Pick the single actor move from the detected sprite moves.

    If `actor_identity` is known (a sprite identity tracked across the
    episode), return that sprite's move. Otherwise, when exactly one sprite
    moved, return it (unambiguous actor); when several moved, return None
    rather than guess (the caller may have walkers as well as an actor).
    """
    moves = detect_sprite_moves(
        prev_frame, cur_frame, cell=cell, max_displacement=max_displacement
    )
    if not moves:
        return None
    if actor_identity is not None:
        for m in moves:
            if m.identity == actor_identity:
                return m
        return None
    if len(moves) == 1:
        return moves[0]
    return None


def summary(moves: list[SpriteMove]) -> dict[str, Any]:
    return {
        "move_count": len(moves),
        "displacements": [m.displacement for m in moves],
    }


def build_moving_sprite_runtime_trace_packet(
    before_frame: Any,
    after_frame: Any,
    *,
    game_id: str = "",
    state_hash: str = "",
    step: int = 0,
    action_name: str = "",
    action_payload: dict[str, Any] | None = None,
    score_delta: int = 0,
    cell: int = 1,
    max_displacement: int | None = 40,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Build a diagnostic packet from one public before/after transition.

    The packet is observational only. It reports unambiguous sprite identity
    matches and does not select actions, rank candidates, or infer a solver.
    """

    before = _as_frame(before_frame)
    after = _as_frame(after_frame)
    moves = detect_sprite_moves(
        before,
        after,
        cell=max(1, int(cell)),
        max_displacement=max_displacement,
    )
    prev_components = extract_structural_components(before, cell=max(1, int(cell)))
    cur_components = extract_structural_components(after, cell=max(1, int(cell)))
    prev_by_identity = _by_identity(prev_components)
    cur_by_identity = _by_identity(cur_components)
    matched_identity_count = 0
    ambiguous_identity_count = 0
    for identity, cur_list in cur_by_identity.items():
        prev_list = prev_by_identity.get(identity)
        if prev_list is None:
            continue
        matched_identity_count += 1
        if len(prev_list) != 1 or len(cur_list) != 1:
            ambiguous_identity_count += 1

    move_rows = [_move_to_packet_row(move) for move in moves[: max(0, sample_limit)]]
    displacement_histogram: Counter[str] = Counter()
    for move in moves:
        displacement_histogram[f"{move.displacement[0]},{move.displacement[1]}"] += 1

    move_count = len(moves)
    single_actor_motion_candidate = move_count == 1
    multi_sprite_motion_candidate = move_count > 1
    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_RUNTIME_TRACE_PACKET_SCHEMA_VERSION,
            "sensor_family": "moving_sprite_tracker",
            "game_id": str(game_id or "").strip().lower(),
            "state_hash": str(state_hash or _frame_hash(before)),
            "step": int(step),
            "action_name": str(action_name or ""),
            "action_payload": dict(action_payload or {}),
            "score_delta": int(score_delta),
            "cell": max(1, int(cell)),
            "max_displacement": (
                None if max_displacement is None else int(max_displacement)
            ),
            "before_component_count": int(len(prev_components)),
            "after_component_count": int(len(cur_components)),
            "matched_identity_count": int(matched_identity_count),
            "ambiguous_identity_count": int(ambiguous_identity_count),
            "move_count": int(move_count),
            "moves": move_rows,
            "displacement_histogram": dict(sorted(displacement_histogram.items())),
            "moving_sprite_candidate": move_count > 0,
            "single_actor_motion_candidate": single_actor_motion_candidate,
            "multi_sprite_motion_candidate": multi_sprite_motion_candidate,
            "ambiguous_actor_identity_candidate": bool(
                ambiguous_identity_count and move_count == 0
            ),
            "score_closure_candidate": bool(int(score_delta) > 0 and move_count > 0),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def validate_moving_sprite_runtime_trace_packet(
    packet: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if (
        str(packet.get("schema_version", "") or "")
        != MOVING_SPRITE_RUNTIME_TRACE_PACKET_SCHEMA_VERSION
    ):
        reasons.append("schema_version_mismatch")
    if str(packet.get("sensor_family", "") or "") != "moving_sprite_tracker":
        reasons.append("sensor_family_mismatch")
    action = str(packet.get("action_name", "") or "")
    if action and not action.startswith("ACTION"):
        reasons.append("unknown_action_name")
    return {
        "schema_version": "eca_moving_sprite_runtime_trace_validation.v1",
        "accepted": not reasons,
        "rejected_reasons": reasons,
        "family": "moving_sprite_tracker",
        "mechanic_state_type": _mechanic_state_type(packet),
        "move_count": _safe_int(packet.get("move_count")),
    }


def summarize_moving_sprite_runtime_trace_packets(
    packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, dict)]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_action: Counter[str] = Counter()
    displacement_histogram: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    move_count_sum = 0
    score_delta_sum = 0

    for packet in packet_rows:
        validation = validate_moving_sprite_runtime_trace_packet(packet)
        accepted = bool(validation.get("accepted"))
        counts["accepted_packet_count" if accepted else "rejected_packet_count"] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        action = str(packet.get("action_name", "") or "")
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["packet_count"] += 1
        by_action[action] += 1
        move_count = _safe_int(packet.get("move_count"))
        score_delta = _safe_int(packet.get("score_delta"))
        move_count_sum += move_count
        score_delta_sum += score_delta
        if move_count > 0:
            counts["packet_with_moves_count"] += 1
            by_game[game_id]["packet_with_moves_count"] += 1
        if bool(packet.get("single_actor_motion_candidate")):
            counts["single_actor_motion_candidate_count"] += 1
        if bool(packet.get("multi_sprite_motion_candidate")):
            counts["multi_sprite_motion_candidate_count"] += 1
        if bool(packet.get("ambiguous_actor_identity_candidate")):
            counts["ambiguous_actor_identity_candidate_count"] += 1
        if bool(packet.get("score_closure_candidate")):
            counts["score_closure_candidate_count"] += 1
        for key, value in dict(packet.get("displacement_histogram") or {}).items():
            displacement_histogram[str(key)] += _safe_int(value)
        if len(samples) < sample_limit:
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "action_name": action,
                    "mechanic_state_type": str(
                        validation.get("mechanic_state_type", "") or ""
                    ),
                    "move_count": move_count,
                    "matched_identity_count": _safe_int(
                        packet.get("matched_identity_count")
                    ),
                    "ambiguous_identity_count": _safe_int(
                        packet.get("ambiguous_identity_count")
                    ),
                    "score_delta": score_delta,
                    "moves": list(packet.get("moves") or [])[:3],
                }
            )

    packet_count = len(packet_rows)
    packet_with_moves_count = int(counts.get("packet_with_moves_count", 0))
    if packet_with_moves_count > 0:
        promotion_state = "moving_sprite_runtime_trace_connection_observed"
    elif packet_count > 0:
        promotion_state = "moving_sprite_runtime_trace_connection_no_moves"
    else:
        promotion_state = "moving_sprite_runtime_trace_connection_no_packets"
    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "packet_count": int(packet_count),
            "accepted_packet_count": int(counts.get("accepted_packet_count", 0)),
            "rejected_packet_count": int(counts.get("rejected_packet_count", 0)),
            "packet_with_moves_count": packet_with_moves_count,
            "total_sprite_move_count": int(move_count_sum),
            "score_delta_sum": int(score_delta_sum),
            "single_actor_motion_candidate_count": int(
                counts.get("single_actor_motion_candidate_count", 0)
            ),
            "multi_sprite_motion_candidate_count": int(
                counts.get("multi_sprite_motion_candidate_count", 0)
            ),
            "ambiguous_actor_identity_candidate_count": int(
                counts.get("ambiguous_actor_identity_candidate_count", 0)
            ),
            "score_closure_candidate_count": int(
                counts.get("score_closure_candidate_count", 0)
            ),
            "promotion_state": promotion_state,
            "runtime_trace_connection_observed": packet_with_moves_count > 0,
            "by_game": {
                game: dict(sorted(counter.items()))
                for game, counter in sorted(by_game.items())
            },
            "by_action": dict(sorted(by_action.items())),
            "displacement_histogram": dict(sorted(displacement_histogram.items())),
            "samples": samples,
            "planner_connection_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_motion_posterior(
    packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    min_actor_moves: int = 6,
    actor_commit_margin: int = 3,
    min_relation_observations: int = 3,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize action->motion relations for committed actor identities.

    This is a posterior-only diagnostic. It infers a stable actor identity from
    the same runtime packets as the actor anchor, then aggregates that actor's
    observed displacement by action name. It does not rank candidates or emit
    actions.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, dict)]
    actor_by_game = _infer_actor_identities_from_packets(
        packet_rows,
        min_moves=max(1, int(min_actor_moves)),
        commit_margin=max(0, int(actor_commit_margin)),
    )
    by_game_state: dict[str, dict[str, Any]] = {}
    relation_counts: Counter[tuple[str, str, str]] = Counter()
    relation_score_delta: Counter[tuple[str, str, str]] = Counter()
    relation_score_closure: Counter[tuple[str, str, str]] = Counter()
    packet_count = 0
    actor_motion_packet_count = 0
    no_actor_motion_packet_count = 0
    score_closure_without_actor_motion_count = 0

    for packet in packet_rows:
        packet_count += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        action = str(packet.get("action_name", "") or "")
        committed_identity = actor_by_game.get(game_id)
        state = by_game_state.setdefault(
            game_id,
            {
                "packet_count": 0,
                "committed_identity": committed_identity,
                "actor_motion_packet_count": 0,
                "no_actor_motion_packet_count": 0,
                "score_closure_without_actor_motion_count": 0,
                "relations": Counter(),
                "relation_score_delta": Counter(),
                "relation_score_closure": Counter(),
            },
        )
        state["packet_count"] += 1
        if not committed_identity:
            continue
        actor_moves = [
            move
            for move in list(packet.get("moves") or [])
            if isinstance(move, dict)
            and _identity_key_from_packet(move.get("identity")) == committed_identity
        ]
        if not actor_moves:
            state["no_actor_motion_packet_count"] += 1
            no_actor_motion_packet_count += 1
            if bool(packet.get("score_closure_candidate")):
                state["score_closure_without_actor_motion_count"] += 1
                score_closure_without_actor_motion_count += 1
            continue
        actor_motion_packet_count += 1
        state["actor_motion_packet_count"] += 1
        for move in actor_moves:
            disp = _displacement_key(move.get("displacement"))
            key = (game_id, action, disp)
            relation_counts[key] += 1
            relation_score_delta[key] += _safe_int(packet.get("score_delta"))
            state["relations"][(action, disp)] += 1
            state["relation_score_delta"][(action, disp)] += _safe_int(
                packet.get("score_delta")
            )
            if bool(packet.get("score_closure_candidate")):
                relation_score_closure[key] += 1
                state["relation_score_closure"][(action, disp)] += 1

    rows: list[dict[str, Any]] = []
    confirmed_relation_count = 0
    score_linked_relation_count = 0
    for game_id, state in sorted(by_game_state.items()):
        relation_rows: list[dict[str, Any]] = []
        for (action, disp), count in state["relations"].most_common(sample_limit):
            score_closure_count = int(
                state["relation_score_closure"].get((action, disp), 0)
            )
            if score_closure_count:
                posterior_state = "score_linked_observed"
                score_linked_relation_count += 1
            elif count >= max(1, int(min_relation_observations)):
                posterior_state = "confirmed_motion_relation"
                confirmed_relation_count += 1
            else:
                posterior_state = "insufficient_observations"
            relation_rows.append(
                {
                    "action_name": action,
                    "displacement": _displacement_list(disp),
                    "observation_count": int(count),
                    "score_delta_sum": int(
                        state["relation_score_delta"].get((action, disp), 0)
                    ),
                    "score_closure_count": score_closure_count,
                    "posterior_state": posterior_state,
                }
            )
        rows.append(
            {
                "game_id": game_id,
                "packet_count": int(state["packet_count"]),
                "committed": bool(state["committed_identity"]),
                "committed_identity": (
                    _identity_to_packet(state["committed_identity"])
                    if state["committed_identity"]
                    else {}
                ),
                "actor_motion_packet_count": int(
                    state["actor_motion_packet_count"]
                ),
                "no_actor_motion_packet_count": int(
                    state["no_actor_motion_packet_count"]
                ),
                "score_closure_without_actor_motion_count": int(
                    state["score_closure_without_actor_motion_count"]
                ),
                "relation_count": len(state["relations"]),
                "relations": relation_rows,
            }
        )

    action_motion_relation_count = len(relation_counts)
    if action_motion_relation_count:
        promotion_state = "moving_sprite_motion_posterior_observed"
    elif actor_by_game:
        promotion_state = "moving_sprite_motion_posterior_no_relations"
    elif packet_count:
        promotion_state = "moving_sprite_motion_posterior_no_actor_commit"
    else:
        promotion_state = "moving_sprite_motion_posterior_no_packets"

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_MOTION_POSTERIOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "packet_count": int(packet_count),
            "committed_game_count": len(actor_by_game),
            "actor_motion_packet_count": int(actor_motion_packet_count),
            "no_actor_motion_packet_count": int(no_actor_motion_packet_count),
            "score_closure_without_actor_motion_count": int(
                score_closure_without_actor_motion_count
            ),
            "action_motion_relation_count": int(action_motion_relation_count),
            "confirmed_motion_relation_count": int(confirmed_relation_count),
            "score_linked_motion_relation_count": int(score_linked_relation_count),
            "promotion_state": promotion_state,
            "min_actor_moves": max(1, int(min_actor_moves)),
            "actor_commit_margin": max(0, int(actor_commit_margin)),
            "min_relation_observations": max(1, int(min_relation_observations)),
            "by_game": rows,
            "planner_connection_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_guarded_planner_readiness(
    packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    min_actor_moves: int = 6,
    actor_commit_margin: int = 3,
    min_relation_observations: int = 3,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Shadow-readiness gate for using actor motion posterior in planning.

    The readiness sensor is deliberately conservative. It can admit a
    confirmed action->displacement relation as a motion-model fact, but it
    blocks score-bearing credit whenever score closure occurred without the
    committed actor moving. It does not execute, rank, or generate actions.
    """

    posterior = summarize_moving_sprite_motion_posterior(
        packets,
        min_actor_moves=min_actor_moves,
        actor_commit_margin=actor_commit_margin,
        min_relation_observations=min_relation_observations,
        sample_limit=sample_limit,
    )
    rows: list[dict[str, Any]] = []
    motion_model_ready_game_count = 0
    score_bearing_ready_game_count = 0
    shadow_admissible_relation_count = 0
    score_credit_blocked_count = 0
    relation_shadow_rows: list[dict[str, Any]] = []

    for game_row in list(posterior.get("by_game") or []):
        if not isinstance(game_row, dict):
            continue
        game_id = str(game_row.get("game_id", "") or "unknown")
        committed = bool(game_row.get("committed"))
        score_without_actor = _safe_int(
            game_row.get("score_closure_without_actor_motion_count")
        )
        relations = [
            relation
            for relation in list(game_row.get("relations") or [])
            if isinstance(relation, dict)
        ]
        admissible_relations: list[dict[str, Any]] = []
        blocked_relations: list[dict[str, Any]] = []
        score_linked_relations: list[dict[str, Any]] = []
        for relation in relations:
            posterior_state = str(relation.get("posterior_state", "") or "")
            relation_row = {
                "action_name": str(relation.get("action_name", "") or ""),
                "displacement": list(relation.get("displacement") or [0, 0])[:2],
                "observation_count": _safe_int(relation.get("observation_count")),
                "score_delta_sum": _safe_int(relation.get("score_delta_sum")),
                "score_closure_count": _safe_int(relation.get("score_closure_count")),
                "posterior_state": posterior_state,
            }
            if posterior_state == "score_linked_observed":
                score_linked_relations.append(relation_row)
            if posterior_state not in {
                "confirmed_motion_relation",
                "score_linked_observed",
            }:
                blocked = dict(relation_row)
                blocked["blocked_reason"] = "insufficient_relation_evidence"
                blocked_relations.append(blocked)
                continue
            if score_without_actor > 0 and posterior_state == "score_linked_observed":
                blocked = dict(relation_row)
                blocked["blocked_reason"] = (
                    "score_closure_without_actor_motion_observed"
                )
                blocked_relations.append(blocked)
                score_credit_blocked_count += 1
                continue
            admissible = dict(relation_row)
            admissible["readiness_state"] = (
                "score_bearing_shadow_ready"
                if posterior_state == "score_linked_observed"
                else "motion_model_shadow_ready"
            )
            admissible_relations.append(admissible)
            relation_shadow_rows.append(
                {"game_id": game_id, **admissible}
            )

        if admissible_relations:
            motion_model_ready_game_count += 1
        if score_linked_relations and not score_without_actor:
            score_bearing_ready_game_count += 1
        shadow_admissible_relation_count += len(admissible_relations)
        if score_without_actor > 0 and not score_linked_relations:
            score_credit_blocked_count += score_without_actor
        rows.append(
            {
                "game_id": game_id,
                "committed": committed,
                "committed_identity": dict(game_row.get("committed_identity") or {}),
                "actor_motion_packet_count": _safe_int(
                    game_row.get("actor_motion_packet_count")
                ),
                "score_closure_without_actor_motion_count": score_without_actor,
                "relation_count": _safe_int(game_row.get("relation_count")),
                "shadow_admissible_relation_count": len(admissible_relations),
                "blocked_relation_count": len(blocked_relations),
                "score_linked_relation_count": len(score_linked_relations),
                "motion_model_shadow_ready": bool(admissible_relations),
                "score_bearing_shadow_ready": bool(
                    score_linked_relations and not score_without_actor
                ),
                "admissible_relations": admissible_relations[:sample_limit],
                "blocked_relations": blocked_relations[:sample_limit],
            }
        )

    if shadow_admissible_relation_count:
        promotion_state = "moving_sprite_guarded_planner_readiness_observed"
    elif _safe_int(posterior.get("action_motion_relation_count")):
        promotion_state = "moving_sprite_guarded_planner_readiness_blocked"
    elif _safe_int(posterior.get("committed_game_count")):
        promotion_state = "moving_sprite_guarded_planner_readiness_no_relations"
    elif _safe_int(posterior.get("packet_count")):
        promotion_state = "moving_sprite_guarded_planner_readiness_no_actor_commit"
    else:
        promotion_state = "moving_sprite_guarded_planner_readiness_no_packets"

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_GUARDED_PLANNER_READINESS_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": _safe_int(posterior.get("packet_count")),
            "committed_game_count": _safe_int(posterior.get("committed_game_count")),
            "motion_model_ready_game_count": int(motion_model_ready_game_count),
            "score_bearing_ready_game_count": int(score_bearing_ready_game_count),
            "shadow_admissible_relation_count": int(
                shadow_admissible_relation_count
            ),
            "score_credit_blocked_count": int(score_credit_blocked_count),
            "posterior_promotion_state": str(
                posterior.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "by_game": rows,
            "relation_shadow_samples": relation_shadow_rows[:sample_limit],
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_actor_ambiguity_splitter(
    packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    min_soft_evidence: int = 8,
    min_relation_observations: int = 4,
    relation_margin: int = 2,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Diagnose actor ambiguity when strict actor anchoring cannot commit.

    This splitter is intentionally soft. It looks for identities with repeated
    action/displacement consistency in observed packets and reports candidate
    separation quality, but it does not commit the actor or authorize planner
    use.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, dict)]
    strict_commits = _infer_actor_identities_from_packets(
        packet_rows,
        min_moves=6,
        commit_margin=3,
    )
    by_game: dict[str, dict[str, Any]] = {}
    for packet in packet_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(
            game_id,
            {
                "packet_count": 0,
                "packet_with_moves_count": 0,
                "solo_moves": Counter(),
                "any_moves": Counter(),
                "identity_relations": defaultdict(Counter),
                "identity_score_closure": Counter(),
                "ambiguous_packet_count": 0,
                "multi_move_packet_count": 0,
            },
        )
        state["packet_count"] += 1
        move_rows = [
            move for move in list(packet.get("moves") or []) if isinstance(move, dict)
        ]
        if move_rows:
            state["packet_with_moves_count"] += 1
        if len(move_rows) > 1:
            state["multi_move_packet_count"] += 1
        if _safe_int(packet.get("ambiguous_identity_count")) > 0:
            state["ambiguous_packet_count"] += 1
        keys: list[tuple[Any, ...]] = []
        for move in move_rows:
            key = _identity_key_from_packet(move.get("identity"))
            if not key:
                continue
            keys.append(key)
            state["any_moves"][key] += 1
            relation_key = (
                str(packet.get("action_name", "") or ""),
                _displacement_key(move.get("displacement")),
            )
            state["identity_relations"][key][relation_key] += 1
            if bool(packet.get("score_closure_candidate")):
                state["identity_score_closure"][key] += 1
        if len(keys) == 1:
            state["solo_moves"][keys[0]] += 1

    rows: list[dict[str, Any]] = []
    soft_candidate_game_count = 0
    unresolved_ambiguous_game_count = 0
    strict_committed_game_count = 0
    for game_id, state in sorted(by_game.items()):
        candidate_rows: list[dict[str, Any]] = []
        for identity, any_count in state["any_moves"].most_common(sample_limit):
            relation_counter = state["identity_relations"].get(identity, Counter())
            best_relation = relation_counter.most_common(1)
            best_action = ""
            best_disp = "0,0"
            best_relation_count = 0
            if best_relation:
                (best_action, best_disp), best_relation_count = best_relation[0]
            candidate_rows.append(
                {
                    "identity": _identity_to_packet(identity),
                    "any_move_count": int(any_count),
                    "solo_move_count": int(state["solo_moves"].get(identity, 0)),
                    "relation_count": len(relation_counter),
                    "best_relation": {
                        "action_name": str(best_action),
                        "displacement": _displacement_list(str(best_disp)),
                        "observation_count": int(best_relation_count),
                    },
                    "score_closure_count": int(
                        state["identity_score_closure"].get(identity, 0)
                    ),
                }
            )
        any_leader = candidate_rows[0] if candidate_rows else {}
        relation_ranked_candidates = sorted(
            candidate_rows,
            key=lambda row: (
                -_safe_int(
                    dict(row.get("best_relation") or {}).get("observation_count")
                ),
                -_safe_int(row.get("any_move_count")),
                _identity_sort_key(dict(row.get("identity") or {})),
            ),
        )
        top = relation_ranked_candidates[0] if relation_ranked_candidates else {}
        runner = (
            relation_ranked_candidates[1]
            if len(relation_ranked_candidates) > 1
            else {}
        )
        top_evidence = _safe_int(top.get("any_move_count"))
        runner_evidence = _safe_int(runner.get("any_move_count"))
        top_relation = top.get("best_relation") if isinstance(top, dict) else {}
        top_relation_count = _safe_int(
            top_relation.get("observation_count") if isinstance(top_relation, dict) else 0
        )
        runner_relation = runner.get("best_relation") if isinstance(runner, dict) else {}
        runner_relation_count = _safe_int(
            runner_relation.get("observation_count")
            if isinstance(runner_relation, dict)
            else 0
        )
        strict_committed = game_id in strict_commits
        if strict_committed:
            strict_committed_game_count += 1
            split_state = "strict_actor_already_committed"
        elif (
            top_evidence >= max(1, int(min_soft_evidence))
            and top_relation_count >= max(1, int(min_relation_observations))
            and top_relation_count - runner_relation_count >= max(0, int(relation_margin))
        ):
            split_state = "soft_actor_candidate_observed"
            soft_candidate_game_count += 1
        elif state["packet_with_moves_count"] > 0:
            split_state = "unresolved_actor_ambiguity"
            unresolved_ambiguous_game_count += 1
        else:
            split_state = "no_actor_motion_evidence"
        rows.append(
            {
                "game_id": game_id,
                "packet_count": int(state["packet_count"]),
                "packet_with_moves_count": int(state["packet_with_moves_count"]),
                "multi_move_packet_count": int(state["multi_move_packet_count"]),
                "ambiguous_packet_count": int(state["ambiguous_packet_count"]),
                "strict_committed": strict_committed,
                "split_state": split_state,
                "top_candidate": top,
                "runner_candidate": runner,
                "any_leader_candidate": any_leader,
                "candidate_count": len(candidate_rows),
                "candidates": relation_ranked_candidates,
            }
        )

    if soft_candidate_game_count:
        promotion_state = "moving_sprite_actor_ambiguity_soft_candidates_observed"
    elif unresolved_ambiguous_game_count:
        promotion_state = "moving_sprite_actor_ambiguity_unresolved"
    elif strict_committed_game_count:
        promotion_state = "moving_sprite_actor_ambiguity_all_committed_or_static"
    elif packet_rows:
        promotion_state = "moving_sprite_actor_ambiguity_no_motion_evidence"
    else:
        promotion_state = "moving_sprite_actor_ambiguity_no_packets"

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_ACTOR_AMBIGUITY_SPLITTER_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": len(packet_rows),
            "strict_committed_game_count": int(strict_committed_game_count),
            "soft_candidate_game_count": int(soft_candidate_game_count),
            "unresolved_ambiguous_game_count": int(unresolved_ambiguous_game_count),
            "promotion_state": promotion_state,
            "min_soft_evidence": max(1, int(min_soft_evidence)),
            "min_relation_observations": max(1, int(min_relation_observations)),
            "relation_margin": max(0, int(relation_margin)),
            "by_game": rows,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_guarded_actor_anchor_posterior(
    packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    min_soft_evidence: int = 8,
    min_relation_observations: int = 4,
    relation_margin: int = 2,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Turn soft actor candidates into guarded shadow anchor posterior rows.

    This is not a strict actor commit. A soft candidate becomes admitted only
    when its best action/displacement relation clears the splitter threshold
    and has no score-credit dependency. The result can feed later shadow
    planning analysis, but cannot execute or rank actions.
    """

    splitter = summarize_moving_sprite_actor_ambiguity_splitter(
        packets,
        min_soft_evidence=min_soft_evidence,
        min_relation_observations=min_relation_observations,
        relation_margin=relation_margin,
        sample_limit=sample_limit,
    )
    rows: list[dict[str, Any]] = []
    strict_committed_game_count = 0
    soft_candidate_game_count = 0
    shadow_admitted_soft_anchor_count = 0
    blocked_soft_anchor_count = 0
    for game_row in list(splitter.get("by_game") or []):
        if not isinstance(game_row, dict):
            continue
        game_id = str(game_row.get("game_id", "") or "unknown")
        split_state = str(game_row.get("split_state", "") or "")
        top_candidate = dict(game_row.get("top_candidate") or {})
        runner_candidate = dict(game_row.get("runner_candidate") or {})
        top_relation = dict(top_candidate.get("best_relation") or {})
        runner_relation = dict(runner_candidate.get("best_relation") or {})
        top_relation_count = _safe_int(top_relation.get("observation_count"))
        runner_relation_count = _safe_int(runner_relation.get("observation_count"))
        relation_margin_observed = top_relation_count - runner_relation_count
        top_any_count = _safe_int(top_candidate.get("any_move_count"))
        runner_any_count = _safe_int(runner_candidate.get("any_move_count"))
        score_closure_count = _safe_int(top_candidate.get("score_closure_count"))
        posterior_state = "not_applicable"
        blocked_reason = ""
        admitted = False
        if split_state == "strict_actor_already_committed":
            strict_committed_game_count += 1
            posterior_state = "strict_anchor_already_available"
        elif split_state == "soft_actor_candidate_observed":
            soft_candidate_game_count += 1
            if top_any_count < max(1, int(min_soft_evidence)):
                posterior_state = "soft_anchor_blocked"
                blocked_reason = "below_min_soft_evidence"
            elif top_relation_count < max(1, int(min_relation_observations)):
                posterior_state = "soft_anchor_blocked"
                blocked_reason = "below_min_relation_observations"
            elif relation_margin_observed < max(0, int(relation_margin)):
                posterior_state = "soft_anchor_blocked"
                blocked_reason = "relation_margin_too_small"
            elif score_closure_count > 0:
                posterior_state = "soft_anchor_blocked"
                blocked_reason = "candidate_has_score_closure_credit"
            else:
                admitted = True
                posterior_state = "soft_anchor_shadow_admitted"
                shadow_admitted_soft_anchor_count += 1
        elif split_state == "unresolved_actor_ambiguity":
            posterior_state = "unresolved_actor_ambiguity"
        elif split_state == "no_actor_motion_evidence":
            posterior_state = "no_actor_motion_evidence"
        if split_state == "soft_actor_candidate_observed" and not admitted:
            blocked_soft_anchor_count += 1
        rows.append(
            {
                "game_id": game_id,
                "split_state": split_state,
                "posterior_state": posterior_state,
                "blocked_reason": blocked_reason,
                "shadow_admitted": admitted,
                "soft_anchor_identity": dict(top_candidate.get("identity") or {}),
                "supporting_relation": {
                    "action_name": str(top_relation.get("action_name", "") or ""),
                    "displacement": list(top_relation.get("displacement") or [0, 0])[:2],
                    "observation_count": top_relation_count,
                },
                "runner_relation": {
                    "action_name": str(runner_relation.get("action_name", "") or ""),
                    "displacement": list(runner_relation.get("displacement") or [0, 0])[:2],
                    "observation_count": runner_relation_count,
                },
                "relation_margin_observed": int(relation_margin_observed),
                "any_move_count": int(top_any_count),
                "any_move_margin": int(top_any_count - runner_any_count),
                "score_closure_count": int(score_closure_count),
                "score_credit_safe": score_closure_count == 0,
                "actor_commit_enabled": False,
                "planner_connection_enabled": False,
                "ranking_enabled": False,
            }
        )

    if shadow_admitted_soft_anchor_count:
        promotion_state = "moving_sprite_guarded_actor_anchor_posterior_observed"
    elif soft_candidate_game_count:
        promotion_state = "moving_sprite_guarded_actor_anchor_posterior_blocked"
    elif _safe_int(splitter.get("packet_count")):
        promotion_state = "moving_sprite_guarded_actor_anchor_posterior_no_soft_candidates"
    else:
        promotion_state = "moving_sprite_guarded_actor_anchor_posterior_no_packets"

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_GUARDED_ACTOR_ANCHOR_POSTERIOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": _safe_int(splitter.get("packet_count")),
            "strict_committed_game_count": int(strict_committed_game_count),
            "soft_candidate_game_count": int(soft_candidate_game_count),
            "shadow_admitted_soft_anchor_count": int(
                shadow_admitted_soft_anchor_count
            ),
            "blocked_soft_anchor_count": int(blocked_soft_anchor_count),
            "splitter_promotion_state": str(
                splitter.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "min_soft_evidence": max(1, int(min_soft_evidence)),
            "min_relation_observations": max(1, int(min_relation_observations)),
            "relation_margin": max(0, int(relation_margin)),
            "by_game": rows,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_soft_anchor_motion_posterior(
    packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    min_soft_evidence: int = 8,
    min_relation_observations: int = 4,
    relation_margin: int = 2,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Build a shadow motion posterior for guarded soft actor anchors.

    This consumes the guarded actor-anchor posterior, not raw ambiguity output.
    An admitted soft anchor may explain repeated action->displacement facts, but
    it still does not become a strict actor commit and cannot drive runtime
    ranking, planning, or action generation.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, dict)]
    anchor_posterior = summarize_moving_sprite_guarded_actor_anchor_posterior(
        packet_rows,
        min_soft_evidence=min_soft_evidence,
        min_relation_observations=min_relation_observations,
        relation_margin=relation_margin,
        sample_limit=sample_limit,
    )
    soft_anchor_by_game: dict[str, tuple[Any, ...]] = {}
    blocked_soft_anchor_count = 0
    for game_row in list(anchor_posterior.get("by_game") or []):
        if not isinstance(game_row, dict):
            continue
        game_id = str(game_row.get("game_id", "") or "unknown")
        if bool(game_row.get("shadow_admitted")) and (
            str(game_row.get("posterior_state", "") or "")
            == "soft_anchor_shadow_admitted"
        ):
            identity = _identity_key_from_packet(game_row.get("soft_anchor_identity"))
            if identity:
                soft_anchor_by_game[game_id] = identity
        elif str(game_row.get("posterior_state", "") or "") == "soft_anchor_blocked":
            blocked_soft_anchor_count += 1

    by_game_state: dict[str, dict[str, Any]] = {}
    relation_counts: Counter[tuple[str, str, str]] = Counter()
    packet_count = 0
    soft_anchor_motion_packet_count = 0
    no_soft_anchor_motion_packet_count = 0
    score_closure_without_soft_anchor_motion_count = 0

    for packet in packet_rows:
        packet_count += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        action = str(packet.get("action_name", "") or "")
        soft_anchor_identity = soft_anchor_by_game.get(game_id)
        state = by_game_state.setdefault(
            game_id,
            {
                "packet_count": 0,
                "soft_anchor_identity": soft_anchor_identity,
                "soft_anchor_motion_packet_count": 0,
                "no_soft_anchor_motion_packet_count": 0,
                "score_closure_without_soft_anchor_motion_count": 0,
                "relations": Counter(),
                "relation_score_delta": Counter(),
                "relation_score_closure": Counter(),
            },
        )
        state["packet_count"] += 1
        if not soft_anchor_identity:
            continue
        soft_anchor_moves = [
            move
            for move in list(packet.get("moves") or [])
            if isinstance(move, dict)
            and _identity_key_from_packet(move.get("identity"))
            == soft_anchor_identity
        ]
        if not soft_anchor_moves:
            state["no_soft_anchor_motion_packet_count"] += 1
            no_soft_anchor_motion_packet_count += 1
            if bool(packet.get("score_closure_candidate")):
                state["score_closure_without_soft_anchor_motion_count"] += 1
                score_closure_without_soft_anchor_motion_count += 1
            continue
        soft_anchor_motion_packet_count += 1
        state["soft_anchor_motion_packet_count"] += 1
        for move in soft_anchor_moves:
            disp = _displacement_key(move.get("displacement"))
            relation_counts[(game_id, action, disp)] += 1
            state["relations"][(action, disp)] += 1
            state["relation_score_delta"][(action, disp)] += _safe_int(
                packet.get("score_delta")
            )
            if bool(packet.get("score_closure_candidate")):
                state["relation_score_closure"][(action, disp)] += 1

    rows: list[dict[str, Any]] = []
    confirmed_relation_count = 0
    score_linked_relation_count = 0
    score_credit_blocked_count = int(blocked_soft_anchor_count)
    motion_model_ready_game_count = 0
    relation_shadow_rows: list[dict[str, Any]] = []
    for game_id, state in sorted(by_game_state.items()):
        relation_rows: list[dict[str, Any]] = []
        game_score_without_soft_anchor = _safe_int(
            state.get("score_closure_without_soft_anchor_motion_count")
        )
        game_motion_ready = False
        for (action, disp), count in state["relations"].most_common(sample_limit):
            score_closure_count = int(
                state["relation_score_closure"].get((action, disp), 0)
            )
            if score_closure_count:
                posterior_state = "soft_anchor_score_credit_blocked"
                score_linked_relation_count += 1
                score_credit_blocked_count += score_closure_count
            elif count >= max(1, int(min_relation_observations)):
                posterior_state = "confirmed_soft_anchor_motion_relation"
                confirmed_relation_count += 1
                game_motion_ready = True
            else:
                posterior_state = "insufficient_observations"
            relation_row = {
                "action_name": str(action),
                "displacement": _displacement_list(str(disp)),
                "observation_count": int(count),
                "score_delta_sum": int(
                    state["relation_score_delta"].get((action, disp), 0)
                ),
                "score_closure_count": score_closure_count,
                "posterior_state": posterior_state,
            }
            relation_rows.append(relation_row)
            if posterior_state == "confirmed_soft_anchor_motion_relation":
                relation_shadow_rows.append({"game_id": game_id, **relation_row})
        if game_score_without_soft_anchor:
            score_credit_blocked_count += game_score_without_soft_anchor
        if game_motion_ready:
            motion_model_ready_game_count += 1
        rows.append(
            {
                "game_id": game_id,
                "soft_anchor_admitted": bool(state["soft_anchor_identity"]),
                "soft_anchor_identity": (
                    _identity_to_packet(state["soft_anchor_identity"])
                    if state["soft_anchor_identity"]
                    else {}
                ),
                "actor_commit_enabled": False,
                "packet_count": int(state["packet_count"]),
                "soft_anchor_motion_packet_count": int(
                    state["soft_anchor_motion_packet_count"]
                ),
                "no_soft_anchor_motion_packet_count": int(
                    state["no_soft_anchor_motion_packet_count"]
                ),
                "score_closure_without_soft_anchor_motion_count": int(
                    game_score_without_soft_anchor
                ),
                "relation_count": len(state["relations"]),
                "motion_model_shadow_ready": bool(game_motion_ready),
                "score_bearing_shadow_ready": False,
                "relations": relation_rows,
            }
        )

    action_motion_relation_count = len(relation_counts)
    if confirmed_relation_count:
        promotion_state = "moving_sprite_soft_anchor_motion_posterior_observed"
    elif action_motion_relation_count:
        promotion_state = (
            "moving_sprite_soft_anchor_motion_posterior_insufficient_evidence"
        )
    elif soft_anchor_by_game:
        promotion_state = "moving_sprite_soft_anchor_motion_posterior_no_relations"
    elif packet_count:
        promotion_state = "moving_sprite_soft_anchor_motion_posterior_no_soft_anchor"
    else:
        promotion_state = "moving_sprite_soft_anchor_motion_posterior_no_packets"

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_SOFT_ANCHOR_MOTION_POSTERIOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(packet_count),
            "soft_anchor_game_count": len(soft_anchor_by_game),
            "blocked_soft_anchor_count": int(blocked_soft_anchor_count),
            "soft_anchor_motion_packet_count": int(
                soft_anchor_motion_packet_count
            ),
            "no_soft_anchor_motion_packet_count": int(
                no_soft_anchor_motion_packet_count
            ),
            "score_closure_without_soft_anchor_motion_count": int(
                score_closure_without_soft_anchor_motion_count
            ),
            "action_motion_relation_count": int(action_motion_relation_count),
            "confirmed_motion_relation_count": int(confirmed_relation_count),
            "score_linked_motion_relation_count": int(score_linked_relation_count),
            "motion_model_ready_game_count": int(motion_model_ready_game_count),
            "score_bearing_ready_game_count": 0,
            "score_credit_blocked_count": int(score_credit_blocked_count),
            "anchor_posterior_promotion_state": str(
                anchor_posterior.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "min_soft_evidence": max(1, int(min_soft_evidence)),
            "min_relation_observations": max(1, int(min_relation_observations)),
            "relation_margin": max(0, int(relation_margin)),
            "by_game": rows,
            "relation_shadow_samples": relation_shadow_rows[:sample_limit],
            "actor_commit_enabled": False,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_coupled_motion_symmetry_sensor(
    packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    min_pair_observations: int = 4,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Detect stable two-sprite coupled motion instead of forcing one actor.

    Some games expose two equally consistent moving identities. In that case a
    single-actor posterior is the wrong abstraction: the safer sensor output is
    a paired action->displacement relation and an explicit recommendation not
    to commit either identity as the sole actor. This remains diagnostic only.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, dict)]
    ambiguity = summarize_moving_sprite_actor_ambiguity_splitter(packet_rows)
    split_by_game = {
        str(row.get("game_id", "") or "unknown"): str(row.get("split_state", "") or "")
        for row in list(ambiguity.get("by_game") or [])
        if isinstance(row, dict)
    }
    by_game: dict[str, dict[str, Any]] = {}
    for packet in packet_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(
            game_id,
            {
                "packet_count": 0,
                "paired_motion_packet_count": 0,
                "relations": Counter(),
                "relation_score_delta": Counter(),
                "relation_score_closure": Counter(),
            },
        )
        state["packet_count"] += 1
        relation_key = _coupled_relation_key_from_moving_packet(packet)
        if relation_key is None:
            continue
        state["paired_motion_packet_count"] += 1
        state["relations"][relation_key] += 1
        state["relation_score_delta"][relation_key] += _safe_int(
            packet.get("score_delta")
        )
        if bool(packet.get("score_closure_candidate")):
            state["relation_score_closure"][relation_key] += 1

    rows: list[dict[str, Any]] = []
    coupled_pair_game_count = 0
    paired_motion_packet_count = 0
    confirmed_relation_count = 0
    score_linked_relation_count = 0
    single_actor_commit_blocked_count = 0
    relation_samples: list[dict[str, Any]] = []
    for game_id, state in sorted(by_game.items()):
        relation_rows: list[dict[str, Any]] = []
        game_confirmed_relation_count = 0
        for (action, keyed_moves), count in state["relations"].most_common(sample_limit):
            score_closure_count = int(
                state["relation_score_closure"].get((action, keyed_moves), 0)
            )
            if score_closure_count:
                posterior_state = "score_linked_coupled_motion_blocked"
                score_linked_relation_count += 1
            elif count >= max(1, int(min_pair_observations)):
                posterior_state = "confirmed_coupled_motion_relation"
                confirmed_relation_count += 1
                game_confirmed_relation_count += 1
            else:
                posterior_state = "insufficient_observations"
            relation_row = {
                "action_name": str(action),
                "motion_symmetry_type": _motion_symmetry_type(
                    [disp for _, disp in keyed_moves]
                ),
                "observation_count": int(count),
                "score_delta_sum": int(
                    state["relation_score_delta"].get((action, keyed_moves), 0)
                ),
                "score_closure_count": score_closure_count,
                "posterior_state": posterior_state,
                "displacement_by_identity": [
                    {
                        "identity": _identity_to_packet(identity),
                        "displacement": _displacement_list(str(displacement)),
                    }
                    for identity, displacement in keyed_moves
                ],
            }
            relation_rows.append(relation_row)
            if posterior_state == "confirmed_coupled_motion_relation":
                relation_samples.append({"game_id": game_id, **relation_row})
        game_paired_packets = int(state["paired_motion_packet_count"])
        paired_motion_packet_count += game_paired_packets
        actor_split_state = split_by_game.get(game_id, "")
        reframed = bool(
            game_confirmed_relation_count
            and actor_split_state == "unresolved_actor_ambiguity"
        )
        if game_confirmed_relation_count:
            coupled_pair_game_count += 1
        if reframed:
            single_actor_commit_blocked_count += 1
        rows.append(
            {
                "game_id": game_id,
                "packet_count": int(state["packet_count"]),
                "paired_motion_packet_count": game_paired_packets,
                "actor_ambiguity_split_state": actor_split_state,
                "single_actor_commit_recommendation": (
                    "do_not_commit_single_actor_use_coupled_pair_model"
                    if reframed
                    else "not_applicable"
                ),
                "coupled_pair_shadow_ready": bool(game_confirmed_relation_count),
                "score_bearing_shadow_ready": False,
                "coupled_motion_relation_count": len(state["relations"]),
                "confirmed_coupled_motion_relation_count": int(
                    game_confirmed_relation_count
                ),
                "relations": relation_rows,
                "actor_commit_enabled": False,
            }
        )

    if confirmed_relation_count:
        promotion_state = "moving_sprite_coupled_motion_symmetry_observed"
    elif paired_motion_packet_count:
        promotion_state = "moving_sprite_coupled_motion_symmetry_insufficient_evidence"
    elif packet_rows:
        promotion_state = "moving_sprite_coupled_motion_symmetry_no_pairs"
    else:
        promotion_state = "moving_sprite_coupled_motion_symmetry_no_packets"

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_COUPLED_MOTION_SYMMETRY_SENSOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": len(packet_rows),
            "paired_motion_packet_count": int(paired_motion_packet_count),
            "coupled_pair_game_count": int(coupled_pair_game_count),
            "coupled_motion_relation_count": int(
                sum(len(state["relations"]) for state in by_game.values())
            ),
            "confirmed_coupled_motion_relation_count": int(
                confirmed_relation_count
            ),
            "score_linked_coupled_motion_relation_count": int(
                score_linked_relation_count
            ),
            "single_actor_commit_blocked_count": int(
                single_actor_commit_blocked_count
            ),
            "score_bearing_ready_game_count": 0,
            "ambiguity_promotion_state": str(
                ambiguity.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "min_pair_observations": max(1, int(min_pair_observations)),
            "by_game": rows,
            "relation_shadow_samples": relation_samples[:sample_limit],
            "actor_commit_enabled": False,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_coupled_pair_goal_effect_bridge(
    moving_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    spell_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    min_pair_observations: int = 4,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Join coupled-pair motion evidence to visible effect/goal packets.

    The bridge is deliberately conservative. It can say that a confirmed pair
    motion relation coincides with visible non-grid effect deltas, while
    keeping score and planner credit blocked until a later guarded stage proves
    objective relevance.
    """

    moving_rows = [
        dict(packet) for packet in moving_packets if isinstance(packet, dict)
    ]
    spell_rows = [
        dict(packet) for packet in spell_packets if isinstance(packet, dict)
    ]
    coupled = summarize_moving_sprite_coupled_motion_symmetry_sensor(
        moving_rows,
        min_pair_observations=min_pair_observations,
        sample_limit=sample_limit,
    )
    confirmed_relation_keys: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for game_row in list(coupled.get("by_game") or []):
        if not isinstance(game_row, dict):
            continue
        game_id = str(game_row.get("game_id", "") or "unknown")
        for relation in list(game_row.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            if str(relation.get("posterior_state", "") or "") != (
                "confirmed_coupled_motion_relation"
            ):
                continue
            key = _coupled_relation_key_from_relation_row(relation)
            if key:
                confirmed_relation_keys[game_id].add(key)

    spell_by_step: dict[tuple[str, int], dict[str, Any]] = {}
    spell_effect_step_keys: set[tuple[str, int]] = set()
    for packet in spell_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        step_key = (game_id, _safe_int(packet.get("step")))
        spell_by_step[step_key] = packet
        if bool(packet.get("spell_effect_candidate")):
            spell_effect_step_keys.add(step_key)

    by_game: dict[str, dict[str, Any]] = {}
    pair_step_keys: set[tuple[str, int]] = set()
    for packet in moving_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(
            game_id,
            {
                "moving_packet_count": 0,
                "spell_packet_count": 0,
                "joined_pair_motion_packet_count": 0,
                "pair_motion_effect_delta_packet_count": 0,
                "pair_motion_spell_state_delta_packet_count": 0,
                "pair_motion_score_delta_sum": 0,
                "pair_motion_effect_region_changed_cell_count_sum": 0,
                "relations": {},
            },
        )
        state["moving_packet_count"] += 1
        relation_key = _coupled_relation_key_from_moving_packet(packet)
        if relation_key is None:
            continue
        if relation_key not in confirmed_relation_keys.get(game_id, set()):
            continue
        step_key = (game_id, _safe_int(packet.get("step")))
        pair_step_keys.add(step_key)
        spell = spell_by_step.get(step_key, {})
        effect_changed = _safe_int(spell.get("effect_region_changed_cell_count"))
        spell_state_delta = bool(spell.get("spell_state_delta_candidate"))
        score_delta = _safe_int(packet.get("score_delta"))
        state["joined_pair_motion_packet_count"] += 1
        state["pair_motion_score_delta_sum"] += score_delta
        state["pair_motion_effect_region_changed_cell_count_sum"] += effect_changed
        if effect_changed > 0:
            state["pair_motion_effect_delta_packet_count"] += 1
        if spell_state_delta:
            state["pair_motion_spell_state_delta_packet_count"] += 1
        relation_state = state["relations"].setdefault(
            relation_key,
            {
                "observation_count": 0,
                "effect_delta_packet_count": 0,
                "spell_state_delta_packet_count": 0,
                "effect_region_changed_cell_count_sum": 0,
                "score_delta_sum": 0,
            },
        )
        relation_state["observation_count"] += 1
        relation_state["effect_region_changed_cell_count_sum"] += effect_changed
        relation_state["score_delta_sum"] += score_delta
        if effect_changed > 0:
            relation_state["effect_delta_packet_count"] += 1
        if spell_state_delta:
            relation_state["spell_state_delta_packet_count"] += 1

    for packet in spell_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(
            game_id,
            {
                "moving_packet_count": 0,
                "spell_packet_count": 0,
                "joined_pair_motion_packet_count": 0,
                "pair_motion_effect_delta_packet_count": 0,
                "pair_motion_spell_state_delta_packet_count": 0,
                "pair_motion_score_delta_sum": 0,
                "pair_motion_effect_region_changed_cell_count_sum": 0,
                "relations": {},
            },
        )
        state["spell_packet_count"] += 1

    rows: list[dict[str, Any]] = []
    relation_samples: list[dict[str, Any]] = []
    effect_bridge_ready_game_count = 0
    pair_motion_effect_delta_packet_count = 0
    joined_pair_motion_packet_count = 0
    score_credit_blocked_count = 0
    for game_id, state in sorted(by_game.items()):
        relation_rows: list[dict[str, Any]] = []
        for relation_key, relation_state in sorted(
            state["relations"].items(),
            key=lambda item: (
                -_safe_int(item[1].get("effect_delta_packet_count")),
                str(item[0]),
            ),
        ):
            relation_row = _coupled_relation_key_to_bridge_row(
                relation_key,
                relation_state,
            )
            relation_rows.append(relation_row)
            if relation_row["effect_delta_packet_count"] > 0:
                relation_samples.append({"game_id": game_id, **relation_row})
        joined_count = _safe_int(state.get("joined_pair_motion_packet_count"))
        effect_count = _safe_int(state.get("pair_motion_effect_delta_packet_count"))
        score_delta_sum = _safe_int(state.get("pair_motion_score_delta_sum"))
        joined_pair_motion_packet_count += joined_count
        pair_motion_effect_delta_packet_count += effect_count
        if effect_count:
            effect_bridge_ready_game_count += 1
        if score_delta_sum:
            score_credit_blocked_count += 1
        spell_effect_without_pair = len(
            [
                key
                for key in spell_effect_step_keys
                if key[0] == game_id and key not in pair_step_keys
            ]
        )
        rows.append(
            {
                "game_id": game_id,
                "moving_packet_count": int(state["moving_packet_count"]),
                "spell_packet_count": int(state["spell_packet_count"]),
                "confirmed_coupled_motion_relation_count": len(
                    confirmed_relation_keys.get(game_id, set())
                ),
                "joined_pair_motion_packet_count": int(joined_count),
                "pair_motion_effect_delta_packet_count": int(effect_count),
                "pair_motion_spell_state_delta_packet_count": int(
                    state["pair_motion_spell_state_delta_packet_count"]
                ),
                "pair_motion_effect_region_changed_cell_count_sum": int(
                    state["pair_motion_effect_region_changed_cell_count_sum"]
                ),
                "spell_effect_without_pair_motion_count": int(
                    spell_effect_without_pair
                ),
                "pair_motion_score_delta_sum": int(score_delta_sum),
                "effect_bridge_shadow_ready": bool(effect_count),
                "score_bearing_shadow_ready": False,
                "bridge_state": (
                    "pair_motion_visible_effect_bridge_observed"
                    if effect_count
                    else "no_pair_motion_visible_effect_bridge"
                ),
                "score_credit_policy": (
                    "blocked_until_goal_or_score_controls"
                    if effect_count
                    else "not_applicable"
                ),
                "relations": relation_rows[:sample_limit],
                "actor_commit_enabled": False,
            }
        )

    spell_effect_without_pair_motion_count = len(spell_effect_step_keys - pair_step_keys)
    if effect_bridge_ready_game_count:
        promotion_state = "moving_sprite_coupled_pair_goal_effect_bridge_observed"
    elif joined_pair_motion_packet_count:
        promotion_state = (
            "moving_sprite_coupled_pair_goal_effect_bridge_no_effect_delta"
        )
    elif moving_rows or spell_rows:
        promotion_state = "moving_sprite_coupled_pair_goal_effect_bridge_no_join"
    else:
        promotion_state = "moving_sprite_coupled_pair_goal_effect_bridge_no_packets"

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_COUPLED_PAIR_GOAL_EFFECT_BRIDGE_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "moving_packet_count": len(moving_rows),
            "spell_packet_count": len(spell_rows),
            "coupled_pair_game_count": _safe_int(
                coupled.get("coupled_pair_game_count")
            ),
            "confirmed_coupled_motion_relation_count": _safe_int(
                coupled.get("confirmed_coupled_motion_relation_count")
            ),
            "joined_pair_motion_packet_count": int(joined_pair_motion_packet_count),
            "pair_motion_effect_delta_packet_count": int(
                pair_motion_effect_delta_packet_count
            ),
            "effect_bridge_ready_game_count": int(effect_bridge_ready_game_count),
            "spell_effect_without_pair_motion_count": int(
                spell_effect_without_pair_motion_count
            ),
            "score_credit_blocked_count": int(score_credit_blocked_count),
            "score_bearing_ready_game_count": 0,
            "coupled_motion_promotion_state": str(
                coupled.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "min_pair_observations": max(1, int(min_pair_observations)),
            "by_game": rows,
            "relation_shadow_samples": relation_samples[:sample_limit],
            "actor_commit_enabled": False,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_coupled_pair_goal_score_attribution_guard(
    moving_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    spell_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    min_pair_observations: int = 4,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Guard score/goal attribution for coupled-pair visible effects.

    A visible effect bridge is not enough to rank or execute. This guard keeps
    score-bearing credit blocked unless paired motion has score support and no
    independent spell-effect confound is present.
    """

    moving_rows = [
        dict(packet) for packet in moving_packets if isinstance(packet, dict)
    ]
    spell_rows = [
        dict(packet) for packet in spell_packets if isinstance(packet, dict)
    ]
    bridge = summarize_moving_sprite_coupled_pair_goal_effect_bridge(
        moving_rows,
        spell_rows,
        min_pair_observations=min_pair_observations,
        sample_limit=sample_limit,
    )
    rows: list[dict[str, Any]] = []
    relation_samples: list[dict[str, Any]] = []
    effect_bridge_ready_game_count = 0
    score_bearing_ready_game_count = 0
    attribution_blocked_game_count = 0
    effect_bridge_candidate_relation_count = 0
    score_supported_relation_count = 0
    score_credit_blocked_relation_count = 0
    independent_spell_effect_confound_count = 0
    for game_row in list(bridge.get("by_game") or []):
        if not isinstance(game_row, dict):
            continue
        game_id = str(game_row.get("game_id", "") or "unknown")
        effect_ready = bool(game_row.get("effect_bridge_shadow_ready"))
        if effect_ready:
            effect_bridge_ready_game_count += 1
        pair_score_delta = _safe_int(game_row.get("pair_motion_score_delta_sum"))
        spell_confound = _safe_int(
            game_row.get("spell_effect_without_pair_motion_count")
        )
        independent_spell_effect_confound_count += spell_confound
        game_reasons: list[str] = []
        if effect_ready and pair_score_delta <= 0:
            game_reasons.append("pair_motion_effect_without_score_delta")
        if spell_confound > 0:
            game_reasons.append("independent_spell_effect_confound")
        score_ready = bool(effect_ready and pair_score_delta > 0 and not spell_confound)
        if score_ready:
            score_bearing_ready_game_count += 1
            guard_state = "score_attribution_shadow_ready"
        elif effect_ready:
            attribution_blocked_game_count += 1
            guard_state = "score_attribution_blocked"
        else:
            guard_state = "no_effect_bridge"

        relation_rows: list[dict[str, Any]] = []
        for relation in list(game_row.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            effect_count = _safe_int(relation.get("effect_delta_packet_count"))
            if effect_count <= 0:
                continue
            effect_bridge_candidate_relation_count += 1
            relation_score_delta = _safe_int(relation.get("score_delta_sum"))
            relation_reasons: list[str] = []
            if relation_score_delta <= 0:
                relation_reasons.append("effect_relation_without_score_delta")
            if spell_confound > 0:
                relation_reasons.append("independent_spell_effect_confound")
            relation_ready = bool(relation_score_delta > 0 and not spell_confound)
            if relation_ready:
                score_supported_relation_count += 1
                relation_guard_state = "score_attribution_shadow_ready"
            else:
                score_credit_blocked_relation_count += 1
                relation_guard_state = "score_attribution_blocked"
            relation_row = {
                "action_name": str(relation.get("action_name", "") or ""),
                "motion_symmetry_type": str(
                    relation.get("motion_symmetry_type", "") or ""
                ),
                "observation_count": _safe_int(
                    relation.get("observation_count")
                ),
                "effect_delta_packet_count": effect_count,
                "effect_region_changed_cell_count_sum": _safe_int(
                    relation.get("effect_region_changed_cell_count_sum")
                ),
                "score_delta_sum": relation_score_delta,
                "guard_state": relation_guard_state,
                "blocked_reasons": relation_reasons,
                "displacement_by_identity": list(
                    relation.get("displacement_by_identity") or []
                ),
            }
            relation_rows.append(relation_row)
            if relation_guard_state == "score_attribution_blocked":
                relation_samples.append({"game_id": game_id, **relation_row})
        rows.append(
            {
                "game_id": game_id,
                "effect_bridge_shadow_ready": effect_ready,
                "score_bearing_shadow_ready": score_ready,
                "guard_state": guard_state,
                "blocked_reasons": game_reasons,
                "joined_pair_motion_packet_count": _safe_int(
                    game_row.get("joined_pair_motion_packet_count")
                ),
                "pair_motion_effect_delta_packet_count": _safe_int(
                    game_row.get("pair_motion_effect_delta_packet_count")
                ),
                "pair_motion_score_delta_sum": pair_score_delta,
                "spell_effect_without_pair_motion_count": spell_confound,
                "effect_bridge_candidate_relation_count": len(relation_rows),
                "score_supported_relation_count": len(
                    [
                        relation
                        for relation in relation_rows
                        if relation["guard_state"] == "score_attribution_shadow_ready"
                    ]
                ),
                "score_credit_blocked_relation_count": len(
                    [
                        relation
                        for relation in relation_rows
                        if relation["guard_state"] == "score_attribution_blocked"
                    ]
                ),
                "relations": relation_rows[:sample_limit],
                "actor_commit_enabled": False,
            }
        )

    if score_bearing_ready_game_count:
        promotion_state = (
            "moving_sprite_coupled_pair_goal_score_attribution_shadow_ready"
        )
    elif attribution_blocked_game_count:
        promotion_state = (
            "moving_sprite_coupled_pair_goal_score_attribution_blocked"
        )
    elif _safe_int(bridge.get("effect_bridge_ready_game_count")):
        promotion_state = (
            "moving_sprite_coupled_pair_goal_score_attribution_no_score_support"
        )
    elif moving_rows or spell_rows:
        promotion_state = "moving_sprite_coupled_pair_goal_score_attribution_no_bridge"
    else:
        promotion_state = "moving_sprite_coupled_pair_goal_score_attribution_no_packets"

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_COUPLED_PAIR_GOAL_SCORE_ATTRIBUTION_GUARD_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "moving_packet_count": len(moving_rows),
            "spell_packet_count": len(spell_rows),
            "effect_bridge_ready_game_count": int(effect_bridge_ready_game_count),
            "score_bearing_ready_game_count": int(score_bearing_ready_game_count),
            "attribution_blocked_game_count": int(attribution_blocked_game_count),
            "effect_bridge_candidate_relation_count": int(
                effect_bridge_candidate_relation_count
            ),
            "score_supported_relation_count": int(score_supported_relation_count),
            "score_credit_blocked_relation_count": int(
                score_credit_blocked_relation_count
            ),
            "independent_spell_effect_confound_count": int(
                independent_spell_effect_confound_count
            ),
            "bridge_promotion_state": str(bridge.get("promotion_state", "") or ""),
            "promotion_state": promotion_state,
            "by_game": rows,
            "blocked_relation_samples": relation_samples[:sample_limit],
            "actor_commit_enabled": False,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_target_goal_counterfactual_attribution(
    moving_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    spell_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    min_pair_observations: int = 4,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Attribute coupled-pair visible effects to target/goal counterfactuals.

    This is the last shadow gate before planner/ranking use. A relation must be
    target-bound, show positive goal movement, and carry a counterfactual
    control marker before it can become score-bearing-ready. Current packets do
    not synthesize those facts; absence is recorded explicitly.
    """

    moving_rows = [
        dict(packet) for packet in moving_packets if isinstance(packet, dict)
    ]
    spell_rows = [
        dict(packet) for packet in spell_packets if isinstance(packet, dict)
    ]
    guard = summarize_moving_sprite_coupled_pair_goal_score_attribution_guard(
        moving_rows,
        spell_rows,
        min_pair_observations=min_pair_observations,
        sample_limit=sample_limit,
    )
    spell_by_game_action: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for packet in spell_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        action_name = str(packet.get("action_name", "") or "")
        if not action_name:
            continue
        if _safe_int(packet.get("effect_region_changed_cell_count")) <= 0:
            continue
        spell_by_game_action[(game_id, action_name)].append(packet)

    rows: list[dict[str, Any]] = []
    relation_samples: list[dict[str, Any]] = []
    candidate_relation_count = 0
    target_bound_relation_count = 0
    positive_goal_delta_relation_count = 0
    counterfactual_supported_relation_count = 0
    attribution_ready_relation_count = 0
    attribution_blocked_relation_count = 0
    missing_target_binding_relation_count = 0
    attribution_ready_game_count = 0
    attribution_blocked_game_count = 0

    for game_row in list(guard.get("by_game") or []):
        if not isinstance(game_row, dict):
            continue
        game_id = str(game_row.get("game_id", "") or "unknown")
        game_relations: list[dict[str, Any]] = []
        for relation in list(game_row.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            candidate_relation_count += 1
            action_name = str(relation.get("action_name", "") or "")
            packets = spell_by_game_action.get((game_id, action_name), [])
            evidence = [
                _target_goal_evidence_from_spell_packet(packet)
                for packet in packets
            ]
            target_bound_packet_count = sum(
                1 for item in evidence if item["target_bound"]
            )
            positive_goal_delta_packet_count = sum(
                1 for item in evidence if item["positive_goal_delta"]
            )
            counterfactual_control_packet_count = sum(
                1 for item in evidence if item["counterfactual_control"]
            )
            target_keys = sorted(
                {
                    str(item["target_key"])
                    for item in evidence
                    if str(item.get("target_key") or "")
                }
            )
            blocked_reasons: list[str] = []
            if target_bound_packet_count <= 0:
                blocked_reasons.append("missing_target_binding_evidence")
            if positive_goal_delta_packet_count <= 0:
                blocked_reasons.append("missing_positive_goal_delta")
            if counterfactual_control_packet_count <= 0:
                blocked_reasons.append("missing_counterfactual_control")
            if _safe_int(relation.get("score_delta_sum")) <= 0:
                blocked_reasons.append("effect_relation_without_score_delta")
            if "independent_spell_effect_confound" in list(
                relation.get("blocked_reasons") or []
            ):
                blocked_reasons.append("independent_spell_effect_confound")

            target_bound = target_bound_packet_count > 0
            positive_goal = positive_goal_delta_packet_count > 0
            counterfactual_supported = counterfactual_control_packet_count > 0
            if target_bound:
                target_bound_relation_count += 1
            else:
                missing_target_binding_relation_count += 1
            if positive_goal:
                positive_goal_delta_relation_count += 1
            if counterfactual_supported:
                counterfactual_supported_relation_count += 1

            ready = bool(target_bound and positive_goal and counterfactual_supported and not blocked_reasons)
            if ready:
                attribution_ready_relation_count += 1
                attribution_state = "target_goal_counterfactual_ready"
            else:
                attribution_blocked_relation_count += 1
                attribution_state = "target_goal_counterfactual_blocked"

            relation_row = {
                "action_name": action_name,
                "motion_symmetry_type": str(
                    relation.get("motion_symmetry_type", "") or ""
                ),
                "effect_delta_packet_count": _safe_int(
                    relation.get("effect_delta_packet_count")
                ),
                "effect_region_changed_cell_count_sum": _safe_int(
                    relation.get("effect_region_changed_cell_count_sum")
                ),
                "score_delta_sum": _safe_int(relation.get("score_delta_sum")),
                "target_bound_packet_count": int(target_bound_packet_count),
                "positive_goal_delta_packet_count": int(
                    positive_goal_delta_packet_count
                ),
                "counterfactual_control_packet_count": int(
                    counterfactual_control_packet_count
                ),
                "spell_effect_packet_count": int(len(packets)),
                "target_keys": target_keys[:sample_limit],
                "attribution_state": attribution_state,
                "blocked_reasons": blocked_reasons,
            }
            game_relations.append(relation_row)
            if not ready:
                relation_samples.append({"game_id": game_id, **relation_row})

        ready_relations = [
            row
            for row in game_relations
            if row["attribution_state"] == "target_goal_counterfactual_ready"
        ]
        blocked_relations = [
            row
            for row in game_relations
            if row["attribution_state"] == "target_goal_counterfactual_blocked"
        ]
        if ready_relations:
            attribution_ready_game_count += 1
            game_state = "target_goal_counterfactual_ready"
        elif blocked_relations:
            attribution_blocked_game_count += 1
            game_state = "target_goal_counterfactual_blocked"
        else:
            game_state = "no_target_goal_candidate_relations"
        rows.append(
            {
                "game_id": game_id,
                "effect_bridge_shadow_ready": bool(
                    game_row.get("effect_bridge_shadow_ready")
                ),
                "score_bearing_shadow_ready": bool(ready_relations),
                "target_goal_counterfactual_state": game_state,
                "candidate_relation_count": len(game_relations),
                "ready_relation_count": len(ready_relations),
                "blocked_relation_count": len(blocked_relations),
                "relations": game_relations[:sample_limit],
                "actor_commit_enabled": False,
            }
        )

    if attribution_ready_game_count:
        promotion_state = "moving_sprite_target_goal_counterfactual_attribution_ready"
    elif attribution_blocked_game_count:
        promotion_state = "moving_sprite_target_goal_counterfactual_attribution_blocked"
    elif _safe_int(guard.get("effect_bridge_ready_game_count")):
        promotion_state = "moving_sprite_target_goal_counterfactual_attribution_no_target_evidence"
    elif moving_rows or spell_rows:
        promotion_state = "moving_sprite_target_goal_counterfactual_attribution_no_candidates"
    else:
        promotion_state = "moving_sprite_target_goal_counterfactual_attribution_no_packets"

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_TARGET_GOAL_COUNTERFACTUAL_ATTRIBUTION_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "moving_packet_count": len(moving_rows),
            "spell_packet_count": len(spell_rows),
            "candidate_relation_count": int(candidate_relation_count),
            "target_bound_relation_count": int(target_bound_relation_count),
            "positive_goal_delta_relation_count": int(
                positive_goal_delta_relation_count
            ),
            "counterfactual_supported_relation_count": int(
                counterfactual_supported_relation_count
            ),
            "attribution_ready_relation_count": int(
                attribution_ready_relation_count
            ),
            "attribution_blocked_relation_count": int(
                attribution_blocked_relation_count
            ),
            "missing_target_binding_relation_count": int(
                missing_target_binding_relation_count
            ),
            "attribution_ready_game_count": int(attribution_ready_game_count),
            "attribution_blocked_game_count": int(attribution_blocked_game_count),
            "score_bearing_ready_game_count": int(attribution_ready_game_count),
            "guard_promotion_state": str(guard.get("promotion_state", "") or ""),
            "promotion_state": promotion_state,
            "by_game": rows,
            "blocked_relation_samples": relation_samples[:sample_limit],
            "actor_commit_enabled": False,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_target_bound_effect_component_sensor(
    moving_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    spell_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    min_pair_observations: int = 4,
    min_component_observations: int = 2,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Join coupled-pair relations to stable effect components.

    The output is a target/effect proxy source, not goal proof. It says that a
    coupled pair motion reliably changes a bounded effect component; later
    stages must still prove positive goal delta and counterfactual controls.
    """

    moving_rows = [
        dict(packet) for packet in moving_packets if isinstance(packet, dict)
    ]
    spell_rows = [
        dict(packet) for packet in spell_packets if isinstance(packet, dict)
    ]
    bridge = summarize_moving_sprite_coupled_pair_goal_effect_bridge(
        moving_rows,
        spell_rows,
        min_pair_observations=min_pair_observations,
        sample_limit=sample_limit,
    )

    relation_keys_by_game: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for game_row in list(bridge.get("by_game") or []):
        if not isinstance(game_row, dict):
            continue
        game_id = str(game_row.get("game_id", "") or "unknown")
        for relation in list(game_row.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            if _safe_int(relation.get("effect_delta_packet_count")) <= 0:
                continue
            key = _coupled_relation_key_from_relation_row(relation)
            if key:
                relation_keys_by_game[game_id].add(key)

    spell_by_step: dict[tuple[str, int], dict[str, Any]] = {}
    for packet in spell_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        spell_by_step[(game_id, _safe_int(packet.get("step")))] = packet

    by_game: dict[str, dict[str, Any]] = {}
    for packet in moving_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(
            game_id,
            {
                "moving_packet_count": 0,
                "spell_packet_count": 0,
                "relations": {},
            },
        )
        state["moving_packet_count"] += 1
        relation_key = _coupled_relation_key_from_moving_packet(packet)
        if relation_key is None:
            continue
        if relation_key not in relation_keys_by_game.get(game_id, set()):
            continue
        spell = spell_by_step.get((game_id, _safe_int(packet.get("step"))), {})
        components = [
            component
            for component in list(spell.get("effect_change_components") or [])
            if isinstance(component, dict)
        ]
        relation_state = state["relations"].setdefault(
            relation_key,
            {
                "joined_pair_motion_packet_count": 0,
                "effect_delta_packet_count": 0,
                "effect_component_packet_count": 0,
                "effect_component_count": 0,
                "effect_region_changed_cell_count_sum": 0,
                "component_signatures": Counter(),
                "component_examples": {},
            },
        )
        relation_state["joined_pair_motion_packet_count"] += 1
        relation_state["effect_region_changed_cell_count_sum"] += _safe_int(
            spell.get("effect_region_changed_cell_count")
        )
        if _safe_int(spell.get("effect_region_changed_cell_count")) > 0:
            relation_state["effect_delta_packet_count"] += 1
        if not components:
            continue
        relation_state["effect_component_packet_count"] += 1
        relation_state["effect_component_count"] += len(components)
        for component in components:
            signature = _effect_component_signature(component)
            relation_state["component_signatures"][signature] += 1
            relation_state["component_examples"].setdefault(
                signature,
                _effect_component_example(component),
            )

    for packet in spell_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(
            game_id,
            {
                "moving_packet_count": 0,
                "spell_packet_count": 0,
                "relations": {},
            },
        )
        state["spell_packet_count"] += 1

    rows: list[dict[str, Any]] = []
    relation_samples: list[dict[str, Any]] = []
    candidate_relation_count = 0
    effect_component_relation_count = 0
    stable_effect_component_relation_count = 0
    stable_effect_component_proxy_count = 0
    target_proxy_ready_game_count = 0
    for game_id, state in sorted(by_game.items()):
        relation_rows: list[dict[str, Any]] = []
        for relation_key, relation_state in sorted(
            state["relations"].items(),
            key=lambda item: (
                -_safe_int(item[1].get("effect_component_packet_count")),
                str(item[0]),
            ),
        ):
            candidate_relation_count += 1
            component_packet_count = _safe_int(
                relation_state.get("effect_component_packet_count")
            )
            if component_packet_count > 0:
                effect_component_relation_count += 1
            stable_components: list[dict[str, Any]] = []
            for signature, count in relation_state["component_signatures"].most_common(
                sample_limit
            ):
                if count < max(1, int(min_component_observations)):
                    continue
                example = dict(relation_state["component_examples"].get(signature) or {})
                example["observation_count"] = int(count)
                example["target_proxy_key"] = _stable_proxy_key(signature)
                stable_components.append(example)
            if stable_components:
                stable_effect_component_relation_count += 1
                stable_effect_component_proxy_count += len(stable_components)
                relation_state_name = "stable_effect_component_target_proxy_observed"
            elif component_packet_count > 0:
                relation_state_name = "effect_component_target_proxy_observed"
            else:
                relation_state_name = "missing_effect_component_target_proxy"
            relation_row = {
                **_coupled_relation_key_to_bridge_row(relation_key, relation_state),
                "joined_pair_motion_packet_count": _safe_int(
                    relation_state.get("joined_pair_motion_packet_count")
                ),
                "effect_component_packet_count": component_packet_count,
                "effect_component_count": _safe_int(
                    relation_state.get("effect_component_count")
                ),
                "stable_effect_component_proxy_count": len(stable_components),
                "stable_effect_components": stable_components[:sample_limit],
                "target_proxy_state": relation_state_name,
                "score_bearing_shadow_ready": False,
            }
            relation_rows.append(relation_row)
            if component_packet_count > 0:
                relation_samples.append({"game_id": game_id, **relation_row})
        ready_relation_count = len(
            [
                relation
                for relation in relation_rows
                if relation["stable_effect_component_proxy_count"] > 0
            ]
        )
        if ready_relation_count:
            target_proxy_ready_game_count += 1
            game_state = "stable_effect_component_target_proxy_observed"
        elif relation_rows:
            game_state = "missing_stable_effect_component_target_proxy"
        else:
            game_state = "no_pair_motion_effect_relations"
        rows.append(
            {
                "game_id": game_id,
                "moving_packet_count": int(state["moving_packet_count"]),
                "spell_packet_count": int(state["spell_packet_count"]),
                "candidate_relation_count": len(relation_rows),
                "effect_component_relation_count": len(
                    [
                        relation
                        for relation in relation_rows
                        if relation["effect_component_packet_count"] > 0
                    ]
                ),
                "stable_effect_component_relation_count": ready_relation_count,
                "target_proxy_state": game_state,
                "score_bearing_shadow_ready": False,
                "relations": relation_rows[:sample_limit],
                "actor_commit_enabled": False,
            }
        )

    if stable_effect_component_relation_count:
        promotion_state = (
            "moving_sprite_target_bound_effect_component_sensor_observed"
        )
    elif effect_component_relation_count:
        promotion_state = (
            "moving_sprite_target_bound_effect_component_sensor_unstable_proxy"
        )
    elif candidate_relation_count:
        promotion_state = (
            "moving_sprite_target_bound_effect_component_sensor_no_components"
        )
    elif moving_rows or spell_rows:
        promotion_state = (
            "moving_sprite_target_bound_effect_component_sensor_no_candidates"
        )
    else:
        promotion_state = "moving_sprite_target_bound_effect_component_sensor_no_packets"

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_TARGET_BOUND_EFFECT_COMPONENT_SENSOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "moving_packet_count": len(moving_rows),
            "spell_packet_count": len(spell_rows),
            "candidate_relation_count": int(candidate_relation_count),
            "effect_component_relation_count": int(effect_component_relation_count),
            "stable_effect_component_relation_count": int(
                stable_effect_component_relation_count
            ),
            "stable_effect_component_proxy_count": int(
                stable_effect_component_proxy_count
            ),
            "target_proxy_ready_game_count": int(target_proxy_ready_game_count),
            "score_bearing_ready_game_count": 0,
            "bridge_promotion_state": str(bridge.get("promotion_state", "") or ""),
            "promotion_state": promotion_state,
            "min_pair_observations": max(1, int(min_pair_observations)),
            "min_component_observations": max(1, int(min_component_observations)),
            "by_game": rows,
            "relation_shadow_samples": relation_samples[:sample_limit],
            "actor_commit_enabled": False,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_effect_component_spatial_stability_sensor(
    moving_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    spell_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    min_pair_observations: int = 4,
    min_component_observations: int = 2,
    max_stable_centroid_span: float = 2.0,
    max_stable_distinct_bbox: int = 1,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Reject shape-stable effect proxies that are spatially unstable.

    Phase682 intentionally grouped effect components by colour/shape so it
    could find target/effect proxy candidates. That is too weak for planner
    credit: progress tickers and moving local effects can share a stable
    colour/shape signature while their bbox sweeps across the frame. This
    shadow sensor records that spatial confound explicitly.
    """

    moving_rows = [
        dict(packet) for packet in moving_packets if isinstance(packet, dict)
    ]
    spell_rows = [
        dict(packet) for packet in spell_packets if isinstance(packet, dict)
    ]
    bridge = summarize_moving_sprite_coupled_pair_goal_effect_bridge(
        moving_rows,
        spell_rows,
        min_pair_observations=min_pair_observations,
        sample_limit=sample_limit,
    )

    relation_keys_by_game: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for game_row in list(bridge.get("by_game") or []):
        if not isinstance(game_row, dict):
            continue
        game_id = str(game_row.get("game_id", "") or "unknown")
        for relation in list(game_row.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            if _safe_int(relation.get("effect_delta_packet_count")) <= 0:
                continue
            key = _coupled_relation_key_from_relation_row(relation)
            if key:
                relation_keys_by_game[game_id].add(key)

    spell_by_step: dict[tuple[str, int], dict[str, Any]] = {}
    for packet in spell_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        spell_by_step[(game_id, _safe_int(packet.get("step")))] = packet

    by_game: dict[str, dict[str, Any]] = {}
    for packet in moving_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(
            game_id,
            {"moving_packet_count": 0, "spell_packet_count": 0, "relations": {}},
        )
        state["moving_packet_count"] += 1
        relation_key = _coupled_relation_key_from_moving_packet(packet)
        if relation_key is None:
            continue
        if relation_key not in relation_keys_by_game.get(game_id, set()):
            continue
        spell = spell_by_step.get((game_id, _safe_int(packet.get("step"))), {})
        components = [
            component
            for component in list(spell.get("effect_change_components") or [])
            if isinstance(component, dict)
        ]
        relation_state = state["relations"].setdefault(
            relation_key,
            {
                "joined_pair_motion_packet_count": 0,
                "effect_delta_packet_count": 0,
                "effect_component_packet_count": 0,
                "effect_component_count": 0,
                "effect_region_changed_cell_count_sum": 0,
                "component_states": {},
            },
        )
        relation_state["joined_pair_motion_packet_count"] += 1
        relation_state["effect_region_changed_cell_count_sum"] += _safe_int(
            spell.get("effect_region_changed_cell_count")
        )
        if _safe_int(spell.get("effect_region_changed_cell_count")) > 0:
            relation_state["effect_delta_packet_count"] += 1
        if not components:
            continue
        relation_state["effect_component_packet_count"] += 1
        relation_state["effect_component_count"] += len(components)
        for component in components:
            signature = _effect_component_signature(component)
            component_state = relation_state["component_states"].setdefault(
                signature,
                {
                    "observation_count": 0,
                    "bbox_counts": Counter(),
                    "centroid_observations": [],
                    "example": _effect_component_example(component),
                },
            )
            bbox = _effect_component_bbox(component)
            centroid = _effect_component_centroid(component)
            component_state["observation_count"] += 1
            component_state["bbox_counts"][bbox] += 1
            component_state["centroid_observations"].append(
                (_safe_int(packet.get("step")), centroid)
            )

    for packet in spell_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(
            game_id,
            {"moving_packet_count": 0, "spell_packet_count": 0, "relations": {}},
        )
        state["spell_packet_count"] += 1

    rows: list[dict[str, Any]] = []
    component_samples: list[dict[str, Any]] = []
    candidate_relation_count = 0
    component_proxy_count = 0
    spatially_stable_proxy_count = 0
    spatially_unstable_proxy_count = 0
    monotonic_sweep_confound_count = 0
    spatially_stable_relation_count = 0
    spatial_confound_relation_count = 0
    spatially_stable_game_count = 0
    spatial_confound_game_count = 0

    for game_id, state in sorted(by_game.items()):
        relation_rows: list[dict[str, Any]] = []
        for relation_key, relation_state in sorted(
            state["relations"].items(),
            key=lambda item: (
                -_safe_int(item[1].get("effect_component_packet_count")),
                str(item[0]),
            ),
        ):
            candidate_relation_count += 1
            proxy_rows: list[dict[str, Any]] = []
            relation_stable_count = 0
            relation_unstable_count = 0
            relation_monotonic_count = 0
            component_states = relation_state.get("component_states") or {}
            for signature, component_state in sorted(
                component_states.items(),
                key=lambda item: (
                    -_safe_int(item[1].get("observation_count")),
                    str(item[0]),
                ),
            ):
                observation_count = _safe_int(
                    component_state.get("observation_count")
                )
                if observation_count < max(1, int(min_component_observations)):
                    continue
                component_proxy_count += 1
                bbox_counts = component_state.get("bbox_counts") or Counter()
                bboxes = list(bbox_counts.keys())
                observations = sorted(
                    list(component_state.get("centroid_observations") or []),
                    key=lambda item: int(item[0]),
                )
                xs = [float(point[0]) for _, point in observations]
                ys = [float(point[1]) for _, point in observations]
                span_x = (max(xs) - min(xs)) if xs else 0.0
                span_y = (max(ys) - min(ys)) if ys else 0.0
                max_span = max(span_x, span_y)
                distinct_bbox_count = len(bboxes)
                monotonic_x = _is_monotonic(xs)
                monotonic_y = _is_monotonic(ys)
                dominant_axis = "x" if span_x > span_y else "y" if span_y > span_x else "none"
                spatially_stable = bool(
                    distinct_bbox_count <= max(1, int(max_stable_distinct_bbox))
                    and max_span <= float(max_stable_centroid_span)
                )
                monotonic_sweep = bool(
                    not spatially_stable
                    and (
                        (monotonic_x and span_x > float(max_stable_centroid_span))
                        or (monotonic_y and span_y > float(max_stable_centroid_span))
                    )
                )
                if spatially_stable:
                    spatially_stable_proxy_count += 1
                    relation_stable_count += 1
                    proxy_state = "spatially_stable_target_proxy_candidate"
                else:
                    spatially_unstable_proxy_count += 1
                    relation_unstable_count += 1
                    if monotonic_sweep:
                        monotonic_sweep_confound_count += 1
                        relation_monotonic_count += 1
                        proxy_state = "monotonic_spatial_sweep_confound"
                    else:
                        proxy_state = "spatially_unstable_effect_component_confound"
                example = dict(component_state.get("example") or {})
                proxy_row = {
                    "target_proxy_key": _stable_proxy_key(signature),
                    "component_signature": str(signature),
                    "observation_count": int(observation_count),
                    "distinct_bbox_count": int(distinct_bbox_count),
                    "centroid_span": {
                        "x": round(float(span_x), 3),
                        "y": round(float(span_y), 3),
                    },
                    "dominant_spatial_axis": dominant_axis,
                    "monotonic_x": bool(monotonic_x),
                    "monotonic_y": bool(monotonic_y),
                    "proxy_state": proxy_state,
                    "sample_bboxes": [
                        [int(value) for value in bbox]
                        for bbox in bboxes[:sample_limit]
                    ],
                    "example": example,
                }
                proxy_rows.append(proxy_row)
                component_samples.append({"game_id": game_id, **proxy_row})
            if relation_stable_count:
                spatially_stable_relation_count += 1
                relation_state_name = "spatially_stable_proxy_observed"
            elif relation_unstable_count:
                spatial_confound_relation_count += 1
                relation_state_name = "effect_component_spatial_confound_observed"
            else:
                relation_state_name = "no_component_proxy_observations"
            relation_row = {
                **_coupled_relation_key_to_bridge_row(relation_key, relation_state),
                "joined_pair_motion_packet_count": _safe_int(
                    relation_state.get("joined_pair_motion_packet_count")
                ),
                "effect_component_packet_count": _safe_int(
                    relation_state.get("effect_component_packet_count")
                ),
                "effect_component_count": _safe_int(
                    relation_state.get("effect_component_count")
                ),
                "component_proxy_count": len(proxy_rows),
                "spatially_stable_proxy_count": int(relation_stable_count),
                "spatially_unstable_proxy_count": int(relation_unstable_count),
                "monotonic_sweep_confound_count": int(relation_monotonic_count),
                "spatial_stability_state": relation_state_name,
                "component_proxies": proxy_rows[:sample_limit],
                "score_bearing_shadow_ready": False,
            }
            relation_rows.append(relation_row)

        game_stable_count = sum(
            _safe_int(row.get("spatially_stable_proxy_count"))
            for row in relation_rows
        )
        game_confound_count = sum(
            _safe_int(row.get("spatially_unstable_proxy_count"))
            for row in relation_rows
        )
        if game_stable_count:
            spatially_stable_game_count += 1
            game_state = "spatially_stable_proxy_observed"
        elif game_confound_count:
            spatial_confound_game_count += 1
            game_state = "effect_component_spatial_confound_observed"
        elif relation_rows:
            game_state = "no_component_proxy_observations"
        else:
            game_state = "no_pair_motion_effect_relations"
        rows.append(
            {
                "game_id": game_id,
                "moving_packet_count": int(state["moving_packet_count"]),
                "spell_packet_count": int(state["spell_packet_count"]),
                "candidate_relation_count": len(relation_rows),
                "spatially_stable_relation_count": sum(
                    1
                    for row in relation_rows
                    if _safe_int(row.get("spatially_stable_proxy_count")) > 0
                ),
                "spatial_confound_relation_count": sum(
                    1
                    for row in relation_rows
                    if _safe_int(row.get("spatially_unstable_proxy_count")) > 0
                ),
                "spatial_stability_state": game_state,
                "score_bearing_shadow_ready": False,
                "relations": relation_rows[:sample_limit],
                "actor_commit_enabled": False,
            }
        )

    if spatially_stable_proxy_count:
        promotion_state = (
            "moving_sprite_effect_component_spatial_stability_sensor_observed"
        )
    elif spatially_unstable_proxy_count:
        promotion_state = (
            "moving_sprite_effect_component_spatial_stability_sensor_blocks_unstable_proxy"
        )
    elif candidate_relation_count:
        promotion_state = (
            "moving_sprite_effect_component_spatial_stability_sensor_no_proxy_components"
        )
    elif moving_rows or spell_rows:
        promotion_state = (
            "moving_sprite_effect_component_spatial_stability_sensor_no_candidates"
        )
    else:
        promotion_state = (
            "moving_sprite_effect_component_spatial_stability_sensor_no_packets"
        )

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_EFFECT_COMPONENT_SPATIAL_STABILITY_SENSOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "moving_packet_count": len(moving_rows),
            "spell_packet_count": len(spell_rows),
            "candidate_relation_count": int(candidate_relation_count),
            "component_proxy_count": int(component_proxy_count),
            "spatially_stable_proxy_count": int(spatially_stable_proxy_count),
            "spatially_unstable_proxy_count": int(spatially_unstable_proxy_count),
            "monotonic_sweep_confound_count": int(monotonic_sweep_confound_count),
            "spatially_stable_relation_count": int(spatially_stable_relation_count),
            "spatial_confound_relation_count": int(spatial_confound_relation_count),
            "spatially_stable_game_count": int(spatially_stable_game_count),
            "spatial_confound_game_count": int(spatial_confound_game_count),
            "target_proxy_ready_game_count": int(spatially_stable_game_count),
            "score_bearing_ready_game_count": 0,
            "bridge_promotion_state": str(bridge.get("promotion_state", "") or ""),
            "promotion_state": promotion_state,
            "min_pair_observations": max(1, int(min_pair_observations)),
            "min_component_observations": max(1, int(min_component_observations)),
            "max_stable_centroid_span": float(max_stable_centroid_span),
            "max_stable_distinct_bbox": max(1, int(max_stable_distinct_bbox)),
            "by_game": rows,
            "component_proxy_samples": component_samples[:sample_limit],
            "actor_commit_enabled": False,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_sparse_chain_rail_state_sensor(
    moving_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    contact_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    min_shared_motion_packets: int = 3,
    min_dominant_shared_delta_observations: int = 3,
    min_large_shared_group_components: int = 4,
    min_large_motion_displacement: int = 6,
    min_high_component_motion_count: int = 8,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize sparse chain/rail motion from observed trace packets.

    This is a family-specific state sensor for games where the generic
    sprite matcher either sees one rare pivot jump or cannot safely bind
    identities because many chain pieces move together. It consumes only
    already-recorded moving-sprite and multi-body-contact packets.
    """

    moving_rows = [
        dict(packet) for packet in moving_packets if isinstance(packet, dict)
    ]
    contact_rows = [
        dict(packet) for packet in contact_packets if isinstance(packet, dict)
    ]
    by_game: dict[str, dict[str, Any]] = {}

    for packet in moving_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(game_id, _empty_sparse_chain_rail_state())
        state["moving_packet_count"] += 1
        move_count = _safe_int(packet.get("move_count"))
        state["sprite_move_count"] += move_count
        if move_count > 0:
            state["sprite_move_packet_count"] += 1
        for move in list(packet.get("moves") or []):
            if not isinstance(move, Mapping):
                continue
            disp = _vector_from_any(move.get("displacement"))
            if disp is None:
                continue
            displacement_key = _vector_key(disp)
            state["sprite_displacement_histogram"][displacement_key] += 1
            distance = abs(disp[0]) + abs(disp[1])
            if distance >= max(1, int(min_large_motion_displacement)):
                state["large_jump_move_count"] += 1
                if len(state["large_jump_samples"]) < sample_limit:
                    state["large_jump_samples"].append(
                        {
                            "step": _safe_int(packet.get("step")),
                            "action_name": str(packet.get("action_name", "") or ""),
                            "displacement": [int(disp[0]), int(disp[1])],
                            "distance": int(distance),
                            "identity": dict(move.get("identity") or {})
                            if isinstance(move.get("identity"), Mapping)
                            else {},
                            "prev_cell": list(move.get("prev_cell") or [])[:2],
                            "cur_cell": list(move.get("cur_cell") or [])[:2],
                        }
                    )

    for packet in contact_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(game_id, _empty_sparse_chain_rail_state())
        state["contact_packet_count"] += 1
        delta = packet.get("contact_topology_delta")
        delta = delta if isinstance(delta, Mapping) else {}
        moved_count = _safe_int(delta.get("moved_component_count"))
        state["moved_component_count_sum"] += moved_count
        if moved_count >= max(1, int(min_high_component_motion_count)):
            state["high_component_motion_packet_count"] += 1
        shared_groups = [
            group
            for group in list(delta.get("shared_delta_groups") or [])
            if isinstance(group, Mapping)
        ]
        if shared_groups:
            state["shared_motion_packet_count"] += 1
        if bool(packet.get("contact_event_candidate")):
            state["contact_event_packet_count"] += 1
        for group in shared_groups:
            disp = _vector_from_any(group.get("delta"))
            if disp is None:
                continue
            key = _vector_key(disp)
            component_count = _safe_int(group.get("component_count"))
            state["shared_delta_observation_histogram"][key] += 1
            state["shared_delta_component_histogram"][key] += component_count
            state["max_shared_group_component_count"] = max(
                state["max_shared_group_component_count"],
                component_count,
            )
            if len(state["shared_motion_samples"]) < sample_limit:
                state["shared_motion_samples"].append(
                    {
                        "step": _safe_int(packet.get("step")),
                        "action_name": str(packet.get("action_name", "") or ""),
                        "delta": [float(disp[0]), float(disp[1])],
                        "component_count": int(component_count),
                        "colors": list(group.get("colors") or [])[:8],
                    }
                )

    rows: list[dict[str, Any]] = []
    candidate_game_count = 0
    sparse_pivot_game_count = 0
    ambiguous_chain_game_count = 0
    contact_only_chain_game_count = 0
    large_jump_move_count = 0
    shared_motion_packet_count = 0
    high_component_motion_packet_count = 0
    max_shared_group_component_count = 0

    for game_id, state in sorted(by_game.items()):
        dominant_key, dominant_observations = _dominant_counter_item(
            state["shared_delta_observation_histogram"]
        )
        _, dominant_component_sum = _dominant_counter_item(
            state["shared_delta_component_histogram"]
        )
        large_jump_move_count += int(state["large_jump_move_count"])
        shared_motion_packet_count += int(state["shared_motion_packet_count"])
        high_component_motion_packet_count += int(
            state["high_component_motion_packet_count"]
        )
        max_shared_group_component_count = max(
            max_shared_group_component_count,
            int(state["max_shared_group_component_count"]),
        )
        large_shared_group = (
            int(state["max_shared_group_component_count"])
            >= max(1, int(min_large_shared_group_components))
        )
        repeated_shared_motion = (
            int(state["shared_motion_packet_count"])
            >= max(1, int(min_shared_motion_packets))
            and int(dominant_observations)
            >= max(1, int(min_dominant_shared_delta_observations))
            and large_shared_group
        )
        high_component_chain = (
            int(state["high_component_motion_packet_count"])
            >= max(1, int(min_shared_motion_packets))
            and large_shared_group
        )
        sparse_pivot = (
            int(state["large_jump_move_count"]) > 0
            and int(state["shared_motion_packet_count"]) > 0
            and large_shared_group
        )
        contact_only_chain = (
            repeated_shared_motion
            and int(state["sprite_move_packet_count"]) == 0
            and int(state["contact_packet_count"]) > 0
        )
        ambiguous_chain = bool(high_component_chain or contact_only_chain)
        candidate = bool(sparse_pivot or ambiguous_chain)
        if candidate:
            candidate_game_count += 1
        if sparse_pivot:
            sparse_pivot_game_count += 1
        if ambiguous_chain:
            ambiguous_chain_game_count += 1
        if contact_only_chain:
            contact_only_chain_game_count += 1
        if sparse_pivot and ambiguous_chain:
            state_name = "sparse_pivot_and_ambiguous_chain_state_observed"
        elif sparse_pivot:
            state_name = "sparse_pivot_rail_state_observed"
        elif ambiguous_chain:
            state_name = "ambiguous_chain_rail_state_observed"
        elif int(state["shared_motion_packet_count"]) > 0:
            state_name = "shared_motion_below_chain_threshold"
        elif int(state["sprite_move_count"]) > 0:
            state_name = "sprite_motion_without_chain_rail_state"
        else:
            state_name = "no_sparse_chain_rail_signal"
        rows.append(
            {
                "game_id": game_id,
                "moving_packet_count": int(state["moving_packet_count"]),
                "contact_packet_count": int(state["contact_packet_count"]),
                "sprite_move_packet_count": int(state["sprite_move_packet_count"]),
                "sprite_move_count": int(state["sprite_move_count"]),
                "large_jump_move_count": int(state["large_jump_move_count"]),
                "shared_motion_packet_count": int(state["shared_motion_packet_count"]),
                "high_component_motion_packet_count": int(
                    state["high_component_motion_packet_count"]
                ),
                "contact_event_packet_count": int(state["contact_event_packet_count"]),
                "max_shared_group_component_count": int(
                    state["max_shared_group_component_count"]
                ),
                "dominant_shared_delta": dominant_key,
                "dominant_shared_delta_observations": int(dominant_observations),
                "dominant_shared_delta_component_sum": int(dominant_component_sum),
                "sparse_pivot_rail_candidate": bool(sparse_pivot),
                "ambiguous_chain_rail_candidate": bool(ambiguous_chain),
                "contact_only_chain_rail_candidate": bool(contact_only_chain),
                "family_state_candidate": bool(candidate),
                "state": state_name,
                "sprite_displacement_histogram": dict(
                    sorted(state["sprite_displacement_histogram"].items())
                ),
                "shared_delta_observation_histogram": dict(
                    sorted(state["shared_delta_observation_histogram"].items())
                ),
                "shared_delta_component_histogram": dict(
                    sorted(state["shared_delta_component_histogram"].items())
                ),
                "large_jump_samples": state["large_jump_samples"][:sample_limit],
                "shared_motion_samples": state["shared_motion_samples"][:sample_limit],
                "score_bearing_shadow_ready": False,
            }
        )

    if candidate_game_count:
        promotion_state = "moving_sprite_sparse_chain_rail_state_observed"
    elif moving_rows or contact_rows:
        promotion_state = "moving_sprite_sparse_chain_rail_state_no_candidates"
    else:
        promotion_state = "moving_sprite_sparse_chain_rail_state_no_packets"

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_SPARSE_CHAIN_RAIL_STATE_SENSOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "moving_packet_count": int(len(moving_rows)),
            "contact_packet_count": int(len(contact_rows)),
            "candidate_game_count": int(candidate_game_count),
            "sparse_pivot_game_count": int(sparse_pivot_game_count),
            "ambiguous_chain_game_count": int(ambiguous_chain_game_count),
            "contact_only_chain_game_count": int(contact_only_chain_game_count),
            "large_jump_move_count": int(large_jump_move_count),
            "shared_motion_packet_count": int(shared_motion_packet_count),
            "high_component_motion_packet_count": int(
                high_component_motion_packet_count
            ),
            "max_shared_group_component_count": int(
                max_shared_group_component_count
            ),
            "promotion_state": promotion_state,
            "min_shared_motion_packets": max(1, int(min_shared_motion_packets)),
            "min_dominant_shared_delta_observations": max(
                1, int(min_dominant_shared_delta_observations)
            ),
            "min_large_shared_group_components": max(
                1, int(min_large_shared_group_components)
            ),
            "min_large_motion_displacement": max(
                1, int(min_large_motion_displacement)
            ),
            "min_high_component_motion_count": max(
                1, int(min_high_component_motion_count)
            ),
            "by_game": rows[:sample_limit],
            "next_planner_contract": {
                "required_before_ranking": "target/goal binding plus score/counterfactual control",
                "allowed_connection": "diagnostic_only_until_guarded_planner_connection",
                "action_generation_allowed": False,
            },
            "actor_commit_enabled": False,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_target_goal_binding_state_sensor(
    moving_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    spell_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    platform_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize whether visible effects can be bound to a goal anchor.

    The sensor is deliberately conservative. It does not treat visible spell
    effects, destructible deltas, or coupled sprite motion as goal evidence
    unless an explicit goal relation/target-bound packet exists. When effects
    are present but no goal anchor is extractable, the output records that
    missing sensor dependency instead of granting planner credit.
    """

    moving_rows = [
        dict(packet) for packet in moving_packets if isinstance(packet, dict)
    ]
    spell_rows = [
        dict(packet) for packet in spell_packets if isinstance(packet, dict)
    ]
    platform_rows = [
        dict(packet) for packet in platform_packets if isinstance(packet, dict)
    ]
    spatial = summarize_moving_sprite_effect_component_spatial_stability_sensor(
        moving_rows,
        spell_rows,
    )

    by_game: dict[str, dict[str, Any]] = {}
    for packet in moving_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(game_id, _empty_target_goal_binding_state())
        state["moving_packet_count"] += 1
        if _safe_int(packet.get("move_count")) > 0:
            state["moving_effect_packet_count"] += 1

    for packet in spell_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(game_id, _empty_target_goal_binding_state())
        state["spell_packet_count"] += 1
        evidence = _target_goal_evidence_from_spell_packet(packet)
        if bool(packet.get("spell_effect_candidate")):
            state["spell_effect_packet_count"] += 1
        if bool(packet.get("spell_cast_candidate")):
            state["spell_cast_packet_count"] += 1
        if _safe_int(packet.get("effect_region_changed_cell_count")) > 0:
            state["visible_effect_packet_count"] += 1
        if evidence["target_bound"]:
            state["target_bound_packet_count"] += 1
        if evidence["positive_goal_delta"]:
            state["positive_goal_delta_packet_count"] += 1
        if evidence["counterfactual_control"]:
            state["counterfactual_control_packet_count"] += 1
        target_key = str(evidence.get("target_key", "") or "")
        if target_key:
            state["target_keys"][target_key] += 1
        if (
            _safe_int(packet.get("effect_region_changed_cell_count")) > 0
            and len(state["effect_samples"]) < sample_limit
        ):
            state["effect_samples"].append(
                {
                    "step": _safe_int(packet.get("step")),
                    "action_name": str(packet.get("action_name", "") or ""),
                    "selected_site": packet.get("selected_site"),
                    "candidate_types": list(packet.get("candidate_types") or []),
                    "effect_region_changed_cell_count": _safe_int(
                        packet.get("effect_region_changed_cell_count")
                    ),
                    "target_bound": bool(evidence["target_bound"]),
                    "positive_goal_delta": bool(evidence["positive_goal_delta"]),
                    "counterfactual_control": bool(
                        evidence["counterfactual_control"]
                    ),
                }
            )

    for packet in platform_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(game_id, _empty_target_goal_binding_state())
        state["platform_packet_count"] += 1
        goal = packet.get("goal_relation")
        goal = goal if isinstance(goal, Mapping) else {}
        distance_delta = goal.get("distance_delta")
        has_goal_relation = distance_delta is not None
        if has_goal_relation:
            state["goal_relation_packet_count"] += 1
        if bool(packet.get("goal_route_progress_candidate")):
            state["goal_route_progress_packet_count"] += 1
        if bool(packet.get("score_closure_candidate")):
            state["score_closure_packet_count"] += 1
        if bool(packet.get("destructible_removed_candidate")):
            state["destructible_removed_packet_count"] += 1
        if (
            has_goal_relation or bool(packet.get("goal_route_progress_candidate"))
        ) and len(state["goal_relation_samples"]) < sample_limit:
            state["goal_relation_samples"].append(
                {
                    "step": _safe_int(packet.get("step")),
                    "action_name": str(packet.get("action_name", "") or ""),
                    "selected_site": packet.get("selected_site"),
                    "candidate_types": list(packet.get("candidate_types") or []),
                    "goal_relation": dict(goal),
                    "goal_route_progress_candidate": bool(
                        packet.get("goal_route_progress_candidate")
                    ),
                }
            )

    spatial_by_game: dict[str, Mapping[str, Any]] = {}
    for row in list(spatial.get("by_game") or []):
        if isinstance(row, Mapping):
            spatial_by_game[str(row.get("game_id", "") or "unknown")] = row

    rows: list[dict[str, Any]] = []
    target_goal_binding_game_count = 0
    missing_goal_anchor_game_count = 0
    visible_effect_without_goal_anchor_game_count = 0
    spatial_confound_game_count = 0
    target_proxy_ready_game_count = 0
    score_bearing_ready_game_count = 0

    for game_id, state in sorted(by_game.items()):
        spatial_row = spatial_by_game.get(game_id, {})
        spatial_target_proxy_count = _safe_int(
            spatial_row.get("spatially_stable_relation_count")
        )
        spatial_confound_count = _safe_int(
            spatial_row.get("spatial_confound_relation_count")
        )
        target_bound_count = int(state["target_bound_packet_count"])
        goal_relation_count = int(state["goal_relation_packet_count"])
        positive_goal_count = int(state["positive_goal_delta_packet_count"]) + int(
            state["goal_route_progress_packet_count"]
        )
        counterfactual_count = int(state["counterfactual_control_packet_count"])
        visible_effect_count = int(state["visible_effect_packet_count"]) + int(
            state["moving_effect_packet_count"]
        )
        target_proxy_ready = spatial_target_proxy_count > 0 or target_bound_count > 0
        explicit_goal_anchor = goal_relation_count > 0 or target_bound_count > 0
        target_goal_binding = bool(
            explicit_goal_anchor
            and positive_goal_count > 0
            and spatial_confound_count == 0
        )
        score_bearing_ready = bool(target_goal_binding and counterfactual_count > 0)
        missing_goal_anchor = bool(visible_effect_count > 0 and not explicit_goal_anchor)
        if target_goal_binding:
            target_goal_binding_game_count += 1
        if target_proxy_ready:
            target_proxy_ready_game_count += 1
        if score_bearing_ready:
            score_bearing_ready_game_count += 1
        if missing_goal_anchor:
            missing_goal_anchor_game_count += 1
        if visible_effect_count > 0 and not explicit_goal_anchor:
            visible_effect_without_goal_anchor_game_count += 1
        if spatial_confound_count > 0:
            spatial_confound_game_count += 1
        blocked_reasons: list[str] = []
        if not explicit_goal_anchor and visible_effect_count > 0:
            blocked_reasons.append("missing_goal_anchor_perception")
        if spatial_confound_count > 0:
            blocked_reasons.append("spatial_effect_component_confound")
        if explicit_goal_anchor and positive_goal_count <= 0:
            blocked_reasons.append("missing_positive_goal_delta")
        if target_goal_binding and counterfactual_count <= 0:
            blocked_reasons.append("missing_counterfactual_control")
        if score_bearing_ready:
            binding_state = "target_goal_binding_score_ready"
        elif target_goal_binding:
            binding_state = "target_goal_binding_observed_without_counterfactual"
        elif missing_goal_anchor:
            binding_state = "target_goal_binding_blocked_missing_goal_anchor"
        elif spatial_confound_count > 0:
            binding_state = "target_goal_binding_blocked_spatial_confound"
        elif visible_effect_count > 0:
            binding_state = "target_goal_binding_blocked"
        else:
            binding_state = "no_visible_effect_for_target_goal_binding"
        rows.append(
            {
                "game_id": game_id,
                "moving_packet_count": int(state["moving_packet_count"]),
                "spell_packet_count": int(state["spell_packet_count"]),
                "platform_packet_count": int(state["platform_packet_count"]),
                "visible_effect_packet_count": int(visible_effect_count),
                "spell_effect_packet_count": int(state["spell_effect_packet_count"]),
                "spell_cast_packet_count": int(state["spell_cast_packet_count"]),
                "target_bound_packet_count": int(target_bound_count),
                "goal_relation_packet_count": int(goal_relation_count),
                "goal_route_progress_packet_count": int(
                    state["goal_route_progress_packet_count"]
                ),
                "positive_goal_delta_packet_count": int(positive_goal_count),
                "counterfactual_control_packet_count": int(counterfactual_count),
                "destructible_removed_packet_count": int(
                    state["destructible_removed_packet_count"]
                ),
                "spatial_target_proxy_relation_count": int(
                    spatial_target_proxy_count
                ),
                "spatial_confound_relation_count": int(spatial_confound_count),
                "target_goal_binding_observed": bool(target_goal_binding),
                "target_proxy_ready": bool(target_proxy_ready),
                "score_bearing_ready": bool(score_bearing_ready),
                "missing_goal_anchor": bool(missing_goal_anchor),
                "binding_state": binding_state,
                "blocked_reasons": blocked_reasons,
                "target_keys": [
                    key
                    for key, _ in state["target_keys"].most_common(sample_limit)
                ],
                "effect_samples": state["effect_samples"][:sample_limit],
                "goal_relation_samples": state["goal_relation_samples"][:sample_limit],
            }
        )

    if score_bearing_ready_game_count:
        promotion_state = "moving_sprite_target_goal_binding_score_ready"
    elif target_goal_binding_game_count:
        promotion_state = "moving_sprite_target_goal_binding_observed"
    elif missing_goal_anchor_game_count:
        promotion_state = "moving_sprite_target_goal_binding_missing_goal_anchor"
    elif spatial_confound_game_count:
        promotion_state = "moving_sprite_target_goal_binding_spatial_confound"
    elif moving_rows or spell_rows or platform_rows:
        promotion_state = "moving_sprite_target_goal_binding_no_goal_evidence"
    else:
        promotion_state = "moving_sprite_target_goal_binding_no_packets"

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_TARGET_GOAL_BINDING_STATE_SENSOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "moving_packet_count": int(len(moving_rows)),
            "spell_packet_count": int(len(spell_rows)),
            "platform_packet_count": int(len(platform_rows)),
            "target_goal_binding_game_count": int(target_goal_binding_game_count),
            "target_proxy_ready_game_count": int(target_proxy_ready_game_count),
            "score_bearing_ready_game_count": int(score_bearing_ready_game_count),
            "missing_goal_anchor_game_count": int(missing_goal_anchor_game_count),
            "visible_effect_without_goal_anchor_game_count": int(
                visible_effect_without_goal_anchor_game_count
            ),
            "spatial_confound_game_count": int(spatial_confound_game_count),
            "spatial_sensor_promotion_state": str(
                spatial.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "by_game": rows[:sample_limit],
            "next_planner_contract": {
                "required_before_ranking": "goal anchor perception plus positive goal delta and counterfactual controls",
                "allowed_connection": "diagnostic_only_until_goal_anchor_and_counterfactual_evidence",
                "action_generation_allowed": False,
            },
            "actor_commit_enabled": False,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_goal_anchor_perception_sensor(
    moving_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    spell_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    platform_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize whether a moving-sprite trace exposes a goal anchor.

    This is an observation-only dependency sensor for the target-goal binding
    lane. It deliberately does not turn a visible effect component into a goal:
    an anchor must come from explicit target fields, a platform goal relation,
    a visible goal component count, or a score-closure packet.
    """

    moving_rows = [
        dict(packet) for packet in moving_packets if isinstance(packet, dict)
    ]
    spell_rows = [
        dict(packet) for packet in spell_packets if isinstance(packet, dict)
    ]
    platform_rows = [
        dict(packet) for packet in platform_packets if isinstance(packet, dict)
    ]

    by_game: dict[str, dict[str, Any]] = {}
    for packet in moving_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(game_id, _empty_goal_anchor_perception_state())
        state["moving_packet_count"] += 1
        if _safe_int(packet.get("score_delta")) > 0 or bool(
            packet.get("score_closure_candidate")
        ):
            state["score_closure_anchor_packet_count"] += 1
            state["anchor_sources"]["moving_score_closure"] += 1
            if len(state["anchor_samples"]) < sample_limit:
                state["anchor_samples"].append(
                    _goal_anchor_sample(
                        packet,
                        anchor_source="moving_score_closure",
                        anchor_kind="score_closure",
                    )
                )

    for packet in spell_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(game_id, _empty_goal_anchor_perception_state())
        state["spell_packet_count"] += 1
        effect_count = _safe_int(packet.get("effect_region_changed_cell_count"))
        evidence = _target_goal_evidence_from_spell_packet(packet)
        if effect_count > 0 or bool(packet.get("spell_effect_candidate")):
            state["visible_effect_packet_count"] += 1
        if evidence["target_bound"]:
            state["explicit_target_anchor_packet_count"] += 1
            state["anchor_sources"]["spell_target_field"] += 1
            target_key = str(evidence.get("target_key", "") or "")
            if target_key:
                state["target_keys"][target_key] += 1
            if len(state["anchor_samples"]) < sample_limit:
                state["anchor_samples"].append(
                    _goal_anchor_sample(
                        packet,
                        anchor_source="spell_target_field",
                        anchor_kind="explicit_target",
                    )
                )
        if evidence["positive_goal_delta"]:
            state["positive_goal_delta_packet_count"] += 1
        if bool(packet.get("score_closure_candidate")) or _safe_int(
            packet.get("score_delta")
        ) > 0:
            state["score_closure_anchor_packet_count"] += 1
            state["anchor_sources"]["spell_score_closure"] += 1
            if len(state["anchor_samples"]) < sample_limit:
                state["anchor_samples"].append(
                    _goal_anchor_sample(
                        packet,
                        anchor_source="spell_score_closure",
                        anchor_kind="score_closure",
                    )
                )
        if (
            (effect_count > 0 or bool(packet.get("spell_effect_candidate")))
            and not evidence["target_bound"]
            and len(state["missing_anchor_samples"]) < sample_limit
        ):
            state["missing_anchor_samples"].append(
                _goal_anchor_sample(
                    packet,
                    anchor_source="visible_spell_effect_without_target",
                    anchor_kind="missing",
                )
            )

    for packet in platform_rows:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(game_id, _empty_goal_anchor_perception_state())
        state["platform_packet_count"] += 1
        goal = packet.get("goal_relation")
        goal = goal if isinstance(goal, Mapping) else {}
        before_counts = packet.get("before_counts")
        before_counts = before_counts if isinstance(before_counts, Mapping) else {}
        after_counts = packet.get("after_counts")
        after_counts = after_counts if isinstance(after_counts, Mapping) else {}
        goal_component_count = max(
            _safe_int(before_counts.get("goal_component_count")),
            _safe_int(after_counts.get("goal_component_count")),
        )
        finite_goal_relation = goal.get("distance_delta") is not None
        if finite_goal_relation:
            state["platform_goal_relation_anchor_packet_count"] += 1
            state["anchor_sources"]["platform_goal_relation"] += 1
            if len(state["anchor_samples"]) < sample_limit:
                state["anchor_samples"].append(
                    _goal_anchor_sample(
                        packet,
                        anchor_source="platform_goal_relation",
                        anchor_kind="explicit_goal_relation",
                    )
                )
        if goal_component_count > 0:
            state["platform_goal_component_anchor_packet_count"] += 1
            state["anchor_sources"]["platform_goal_component_color_11"] += 1
            if len(state["anchor_samples"]) < sample_limit:
                state["anchor_samples"].append(
                    _goal_anchor_sample(
                        packet,
                        anchor_source="platform_goal_component_color_11",
                        anchor_kind="weak_goal_component",
                    )
                )
        else:
            state["hardcoded_goal_color_absent_packet_count"] += 1
        if bool(packet.get("goal_route_progress_candidate")):
            state["goal_route_progress_anchor_packet_count"] += 1
            state["anchor_sources"]["platform_goal_route_progress"] += 1
        if bool(packet.get("score_closure_candidate")) or _safe_int(
            packet.get("score_delta")
        ) > 0:
            state["score_closure_anchor_packet_count"] += 1
            state["anchor_sources"]["platform_score_closure"] += 1
            if len(state["anchor_samples"]) < sample_limit:
                state["anchor_samples"].append(
                    _goal_anchor_sample(
                        packet,
                        anchor_source="platform_score_closure",
                        anchor_kind="score_closure",
                    )
                )
        if bool(packet.get("destructible_removed_candidate")):
            state["destructible_effect_packet_count"] += 1
        if (
            bool(packet.get("destructible_removed_candidate"))
            and not finite_goal_relation
            and goal_component_count <= 0
            and len(state["missing_anchor_samples"]) < sample_limit
        ):
            state["missing_anchor_samples"].append(
                _goal_anchor_sample(
                    packet,
                    anchor_source="destructible_effect_without_goal",
                    anchor_kind="missing",
                )
            )

    rows: list[dict[str, Any]] = []
    goal_anchor_observed_game_count = 0
    explicit_goal_anchor_game_count = 0
    weak_goal_component_anchor_game_count = 0
    score_closure_anchor_game_count = 0
    visible_effect_without_goal_anchor_game_count = 0
    hardcoded_goal_color_absent_game_count = 0

    for game_id, state in sorted(by_game.items()):
        explicit_count = int(state["explicit_target_anchor_packet_count"]) + int(
            state["platform_goal_relation_anchor_packet_count"]
        )
        weak_count = int(state["platform_goal_component_anchor_packet_count"])
        score_closure_count = int(state["score_closure_anchor_packet_count"])
        positive_goal_count = int(state["positive_goal_delta_packet_count"]) + int(
            state["goal_route_progress_anchor_packet_count"]
        )
        visible_effect_count = int(state["visible_effect_packet_count"]) + int(
            state["destructible_effect_packet_count"]
        )
        anchor_count = explicit_count + weak_count + score_closure_count
        missing_anchor = bool(visible_effect_count > 0 and anchor_count == 0)
        hardcoded_goal_color_absent = bool(
            int(state["platform_packet_count"]) > 0
            and int(state["hardcoded_goal_color_absent_packet_count"])
            >= int(state["platform_packet_count"])
            and int(state["platform_goal_component_anchor_packet_count"]) == 0
        )
        if anchor_count > 0:
            goal_anchor_observed_game_count += 1
        if explicit_count > 0:
            explicit_goal_anchor_game_count += 1
        if weak_count > 0:
            weak_goal_component_anchor_game_count += 1
        if score_closure_count > 0:
            score_closure_anchor_game_count += 1
        if missing_anchor:
            visible_effect_without_goal_anchor_game_count += 1
        if hardcoded_goal_color_absent:
            hardcoded_goal_color_absent_game_count += 1

        if score_closure_count > 0:
            anchor_state = "goal_anchor_score_closure_observed"
        elif explicit_count > 0:
            anchor_state = "goal_anchor_explicit_observed"
        elif weak_count > 0:
            anchor_state = "goal_anchor_weak_component_observed"
        elif missing_anchor:
            anchor_state = "goal_anchor_missing_for_visible_effect"
        else:
            anchor_state = "no_goal_anchor_signal"

        rows.append(
            {
                "game_id": game_id,
                "moving_packet_count": int(state["moving_packet_count"]),
                "spell_packet_count": int(state["spell_packet_count"]),
                "platform_packet_count": int(state["platform_packet_count"]),
                "visible_effect_packet_count": int(visible_effect_count),
                "explicit_target_anchor_packet_count": int(
                    state["explicit_target_anchor_packet_count"]
                ),
                "platform_goal_relation_anchor_packet_count": int(
                    state["platform_goal_relation_anchor_packet_count"]
                ),
                "platform_goal_component_anchor_packet_count": int(
                    state["platform_goal_component_anchor_packet_count"]
                ),
                "score_closure_anchor_packet_count": int(score_closure_count),
                "positive_goal_delta_packet_count": int(positive_goal_count),
                "destructible_effect_packet_count": int(
                    state["destructible_effect_packet_count"]
                ),
                "hardcoded_goal_color_absent_packet_count": int(
                    state["hardcoded_goal_color_absent_packet_count"]
                ),
                "goal_anchor_observed": bool(anchor_count > 0),
                "explicit_goal_anchor_observed": bool(explicit_count > 0),
                "weak_goal_component_anchor_observed": bool(weak_count > 0),
                "score_closure_anchor_observed": bool(score_closure_count > 0),
                "visible_effect_without_goal_anchor": bool(missing_anchor),
                "hardcoded_goal_color_absent": bool(hardcoded_goal_color_absent),
                "anchor_state": anchor_state,
                "anchor_sources": dict(
                    sorted(state["anchor_sources"].items())
                ),
                "target_keys": [
                    key
                    for key, _ in state["target_keys"].most_common(sample_limit)
                ],
                "anchor_samples": state["anchor_samples"][:sample_limit],
                "missing_anchor_samples": state["missing_anchor_samples"][:sample_limit],
            }
        )

    if score_closure_anchor_game_count:
        promotion_state = "moving_sprite_goal_anchor_perception_score_closure_observed"
    elif explicit_goal_anchor_game_count:
        promotion_state = "moving_sprite_goal_anchor_perception_observed"
    elif weak_goal_component_anchor_game_count:
        promotion_state = "moving_sprite_goal_anchor_perception_weak_component_observed"
    elif visible_effect_without_goal_anchor_game_count:
        promotion_state = "moving_sprite_goal_anchor_perception_missing_goal_anchor"
    elif moving_rows or spell_rows or platform_rows:
        promotion_state = "moving_sprite_goal_anchor_perception_no_anchor"
    else:
        promotion_state = "moving_sprite_goal_anchor_perception_no_packets"

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_GOAL_ANCHOR_PERCEPTION_SENSOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "moving_packet_count": int(len(moving_rows)),
            "spell_packet_count": int(len(spell_rows)),
            "platform_packet_count": int(len(platform_rows)),
            "goal_anchor_observed_game_count": int(
                goal_anchor_observed_game_count
            ),
            "explicit_goal_anchor_game_count": int(explicit_goal_anchor_game_count),
            "weak_goal_component_anchor_game_count": int(
                weak_goal_component_anchor_game_count
            ),
            "score_closure_anchor_game_count": int(score_closure_anchor_game_count),
            "visible_effect_without_goal_anchor_game_count": int(
                visible_effect_without_goal_anchor_game_count
            ),
            "hardcoded_goal_color_absent_game_count": int(
                hardcoded_goal_color_absent_game_count
            ),
            "promotion_state": promotion_state,
            "by_game": rows[:sample_limit],
            "next_planner_contract": {
                "required_before_ranking": "explicit goal anchor plus target-goal positive delta and counterfactual controls",
                "allowed_connection": "diagnostic_only_until_target_goal_binding_and_counterfactual_evidence",
                "action_generation_allowed": False,
            },
            "actor_commit_enabled": False,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_moving_sprite_guarded_planner_connection_sensor(
    moving_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    spell_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    platform_packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Admission guard before moving-sprite evidence can affect a planner.

    The guard consumes the preceding diagnostic summaries and emits an
    admission decision. It does not reorder candidates or create actions. A
    game is admitted only when target/goal binding is score-ready; missing goal
    anchors, spatial confounds, and missing counterfactual controls block the
    lane explicitly.
    """

    moving_rows = [
        dict(packet) for packet in moving_packets if isinstance(packet, dict)
    ]
    spell_rows = [
        dict(packet) for packet in spell_packets if isinstance(packet, dict)
    ]
    platform_rows = [
        dict(packet) for packet in platform_packets if isinstance(packet, dict)
    ]
    binding = summarize_moving_sprite_target_goal_binding_state_sensor(
        moving_rows,
        spell_rows,
        platform_rows,
        sample_limit=sample_limit,
    )
    anchor = summarize_moving_sprite_goal_anchor_perception_sensor(
        moving_rows,
        spell_rows,
        platform_rows,
        sample_limit=sample_limit,
    )

    binding_by_game: dict[str, Mapping[str, Any]] = {}
    for row in list(binding.get("by_game") or []):
        if isinstance(row, Mapping):
            binding_by_game[str(row.get("game_id", "") or "unknown")] = row
    anchor_by_game: dict[str, Mapping[str, Any]] = {}
    for row in list(anchor.get("by_game") or []):
        if isinstance(row, Mapping):
            anchor_by_game[str(row.get("game_id", "") or "unknown")] = row

    game_ids = sorted(set(binding_by_game) | set(anchor_by_game))
    rows: list[dict[str, Any]] = []
    admitted_game_count = 0
    blocked_game_count = 0
    blocked_missing_goal_anchor_game_count = 0
    blocked_spatial_confound_game_count = 0
    blocked_missing_counterfactual_game_count = 0
    blocked_missing_positive_goal_delta_game_count = 0
    blocked_no_visible_effect_game_count = 0

    for game_id in game_ids:
        binding_row = binding_by_game.get(game_id, {})
        anchor_row = anchor_by_game.get(game_id, {})
        score_ready = bool(binding_row.get("score_bearing_ready"))
        target_goal_binding = bool(binding_row.get("target_goal_binding_observed"))
        missing_goal_anchor = bool(
            binding_row.get("missing_goal_anchor")
            or anchor_row.get("visible_effect_without_goal_anchor")
        )
        spatial_confound = _safe_int(
            binding_row.get("spatial_confound_relation_count")
        ) > 0
        visible_effect = (
            _safe_int(binding_row.get("visible_effect_packet_count")) > 0
            or _safe_int(anchor_row.get("visible_effect_packet_count")) > 0
        )
        positive_goal_delta = (
            _safe_int(binding_row.get("positive_goal_delta_packet_count")) > 0
            or _safe_int(anchor_row.get("positive_goal_delta_packet_count")) > 0
        )
        counterfactual = (
            _safe_int(binding_row.get("counterfactual_control_packet_count")) > 0
        )
        goal_anchor = bool(anchor_row.get("goal_anchor_observed"))
        explicit_anchor = bool(anchor_row.get("explicit_goal_anchor_observed"))
        hardcoded_absent = bool(anchor_row.get("hardcoded_goal_color_absent"))

        blocked_reasons: list[str] = []
        if not visible_effect:
            blocked_reasons.append("no_visible_effect")
        if missing_goal_anchor or not goal_anchor:
            blocked_reasons.append("missing_goal_anchor")
        if hardcoded_absent and not goal_anchor:
            blocked_reasons.append("hardcoded_goal_color_absent")
        if spatial_confound:
            blocked_reasons.append("spatial_effect_component_confound")
        if goal_anchor and not explicit_anchor:
            blocked_reasons.append("weak_goal_anchor_only")
        if not positive_goal_delta:
            blocked_reasons.append("missing_positive_goal_delta")
        if target_goal_binding and not counterfactual:
            blocked_reasons.append("missing_counterfactual_control")
        if not target_goal_binding:
            blocked_reasons.append("target_goal_binding_not_observed")

        admitted = bool(score_ready and not blocked_reasons)
        if admitted:
            admitted_game_count += 1
            connection_state = "planner_connection_admitted_shadow"
        else:
            blocked_game_count += 1
            if missing_goal_anchor or not goal_anchor:
                blocked_missing_goal_anchor_game_count += 1
            if spatial_confound:
                blocked_spatial_confound_game_count += 1
            if target_goal_binding and not counterfactual:
                blocked_missing_counterfactual_game_count += 1
            if not positive_goal_delta:
                blocked_missing_positive_goal_delta_game_count += 1
            if not visible_effect:
                blocked_no_visible_effect_game_count += 1
            if missing_goal_anchor or not goal_anchor:
                connection_state = "planner_connection_blocked_missing_goal_anchor"
            elif spatial_confound:
                connection_state = "planner_connection_blocked_spatial_confound"
            elif target_goal_binding and not counterfactual:
                connection_state = "planner_connection_blocked_missing_counterfactual"
            elif not positive_goal_delta:
                connection_state = "planner_connection_blocked_missing_positive_goal_delta"
            elif not visible_effect:
                connection_state = "planner_connection_blocked_no_visible_effect"
            else:
                connection_state = "planner_connection_blocked"

        rows.append(
            {
                "game_id": game_id,
                "connection_state": connection_state,
                "admitted_shadow": bool(admitted),
                "blocked_reasons": blocked_reasons,
                "goal_anchor_observed": bool(goal_anchor),
                "explicit_goal_anchor_observed": bool(explicit_anchor),
                "hardcoded_goal_color_absent": bool(hardcoded_absent),
                "target_goal_binding_observed": bool(target_goal_binding),
                "score_bearing_ready": bool(score_ready),
                "positive_goal_delta_packet_count": _safe_int(
                    binding_row.get("positive_goal_delta_packet_count")
                ),
                "counterfactual_control_packet_count": _safe_int(
                    binding_row.get("counterfactual_control_packet_count")
                ),
                "visible_effect_packet_count": max(
                    _safe_int(binding_row.get("visible_effect_packet_count")),
                    _safe_int(anchor_row.get("visible_effect_packet_count")),
                ),
                "spatial_confound_relation_count": _safe_int(
                    binding_row.get("spatial_confound_relation_count")
                ),
                "anchor_state": str(anchor_row.get("anchor_state", "") or ""),
                "binding_state": str(binding_row.get("binding_state", "") or ""),
            }
        )

    if admitted_game_count:
        promotion_state = "moving_sprite_guarded_planner_connection_admitted_shadow"
    elif blocked_missing_goal_anchor_game_count:
        promotion_state = "moving_sprite_guarded_planner_connection_blocked_missing_goal_anchor"
    elif blocked_spatial_confound_game_count:
        promotion_state = "moving_sprite_guarded_planner_connection_blocked_spatial_confound"
    elif blocked_missing_counterfactual_game_count:
        promotion_state = "moving_sprite_guarded_planner_connection_blocked_missing_counterfactual"
    elif blocked_missing_positive_goal_delta_game_count:
        promotion_state = "moving_sprite_guarded_planner_connection_blocked_missing_positive_goal_delta"
    elif game_ids:
        promotion_state = "moving_sprite_guarded_planner_connection_blocked"
    elif moving_rows or spell_rows or platform_rows:
        promotion_state = "moving_sprite_guarded_planner_connection_no_admission_candidates"
    else:
        promotion_state = "moving_sprite_guarded_planner_connection_no_packets"

    return _json_safe(
        {
            "schema_version": MOVING_SPRITE_GUARDED_PLANNER_CONNECTION_SENSOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "moving_packet_count": int(len(moving_rows)),
            "spell_packet_count": int(len(spell_rows)),
            "platform_packet_count": int(len(platform_rows)),
            "planner_connection_admitted_game_count": int(admitted_game_count),
            "planner_connection_blocked_game_count": int(blocked_game_count),
            "blocked_missing_goal_anchor_game_count": int(
                blocked_missing_goal_anchor_game_count
            ),
            "blocked_spatial_confound_game_count": int(
                blocked_spatial_confound_game_count
            ),
            "blocked_missing_counterfactual_game_count": int(
                blocked_missing_counterfactual_game_count
            ),
            "blocked_missing_positive_goal_delta_game_count": int(
                blocked_missing_positive_goal_delta_game_count
            ),
            "blocked_no_visible_effect_game_count": int(
                blocked_no_visible_effect_game_count
            ),
            "target_goal_binding_promotion_state": str(
                binding.get("promotion_state", "") or ""
            ),
            "goal_anchor_promotion_state": str(
                anchor.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "by_game": rows[:sample_limit],
            "next_planner_contract": {
                "required_before_runtime_connection": "score-ready target/goal binding with explicit goal anchor, positive goal delta, counterfactual control, and no spatial confound",
                "allowed_connection": "blocked_until_admission_is_positive_and_regression_checked",
                "action_generation_allowed": False,
            },
            "admission_guard_enabled": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def _move_to_packet_row(move: SpriteMove) -> dict[str, Any]:
    return {
        "identity": _identity_to_packet(move.identity),
        "prev_cell": [int(move.prev_cell[0]), int(move.prev_cell[1])],
        "cur_cell": [int(move.cur_cell[0]), int(move.cur_cell[1])],
        "displacement": [int(move.displacement[0]), int(move.displacement[1])],
        "area": int(move.area),
    }


def _target_goal_evidence_from_spell_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    target_key = (
        packet.get("target_key")
        or packet.get("goal_target_key")
        or packet.get("target_id")
        or ""
    )
    target_bbox = packet.get("target_bbox") or packet.get("goal_target_bbox")
    overlap_delta = _safe_int(
        packet.get("target_overlap_delta")
        if "target_overlap_delta" in packet
        else packet.get("goal_overlap_delta")
    )
    distance_delta = _safe_int(
        packet.get("goal_distance_delta")
        if "goal_distance_delta" in packet
        else packet.get("target_distance_delta")
    )
    score_delta = _safe_int(packet.get("score_delta"))
    target_bound = bool(str(target_key or "") or target_bbox)
    positive_goal_delta = bool(
        score_delta > 0
        or overlap_delta > 0
        or distance_delta < 0
        or bool(packet.get("goal_progress_delta_positive", False))
        or bool(packet.get("target_goal_delta_positive", False))
    )
    counterfactual_control = bool(
        packet.get("counterfactual_control_passed", False)
        or packet.get("target_counterfactual_control_passed", False)
        or packet.get("goal_counterfactual_control_passed", False)
    )
    return {
        "target_key": str(target_key or ""),
        "target_bound": target_bound,
        "positive_goal_delta": positive_goal_delta,
        "counterfactual_control": counterfactual_control,
    }


def _effect_component_signature(component: Mapping[str, Any]) -> str:
    bbox = component.get("bbox") or []
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x0, y0, x1, y1 = [_safe_int(value) for value in bbox[:4]]
        size = [max(0, x1 - x0 + 1), max(0, y1 - y0 + 1)]
    else:
        size = [0, 0]
    pair_counts = component.get("color_pair_counts") or {}
    if isinstance(pair_counts, Mapping):
        pairs = [
            f"{str(key)}:{_safe_int(value)}"
            for key, value in sorted(pair_counts.items(), key=lambda item: str(item[0]))
        ]
    else:
        pairs = []
    area = _safe_int(component.get("area"))
    return "|".join([f"area={area}", f"size={size[0]}x{size[1]}", *pairs])


def _effect_component_example(component: Mapping[str, Any]) -> dict[str, Any]:
    bbox = component.get("bbox") or []
    centroid = component.get("centroid") or []
    return {
        "area": _safe_int(component.get("area")),
        "bbox": [_safe_int(value) for value in list(bbox)[:4]]
        if isinstance(bbox, (list, tuple))
        else [],
        "centroid": [
            round(float(value), 3)
            for value in list(centroid)[:2]
            if isinstance(value, (int, float))
        ]
        if isinstance(centroid, (list, tuple))
        else [],
        "color_pair_counts": dict(component.get("color_pair_counts") or {})
        if isinstance(component.get("color_pair_counts"), Mapping)
        else {},
    }


def _effect_component_bbox(component: Mapping[str, Any]) -> tuple[int, int, int, int]:
    bbox = component.get("bbox") or []
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return tuple(_safe_int(value) for value in bbox[:4])  # type: ignore[return-value]
    return (0, 0, 0, 0)


def _effect_component_centroid(component: Mapping[str, Any]) -> tuple[float, float]:
    centroid = component.get("centroid") or []
    if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
        x, y = centroid[:2]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return (float(x), float(y))
    x0, y0, x1, y1 = _effect_component_bbox(component)
    return ((float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0)


def _is_monotonic(values: list[float]) -> bool:
    if len(values) <= 2:
        return True
    nondecreasing = all(b >= a for a, b in zip(values, values[1:]))
    nonincreasing = all(b <= a for a, b in zip(values, values[1:]))
    return bool(nondecreasing or nonincreasing)


def _empty_sparse_chain_rail_state() -> dict[str, Any]:
    return {
        "moving_packet_count": 0,
        "contact_packet_count": 0,
        "sprite_move_packet_count": 0,
        "sprite_move_count": 0,
        "large_jump_move_count": 0,
        "shared_motion_packet_count": 0,
        "high_component_motion_packet_count": 0,
        "contact_event_packet_count": 0,
        "moved_component_count_sum": 0,
        "max_shared_group_component_count": 0,
        "sprite_displacement_histogram": Counter(),
        "shared_delta_observation_histogram": Counter(),
        "shared_delta_component_histogram": Counter(),
        "large_jump_samples": [],
        "shared_motion_samples": [],
    }


def _empty_target_goal_binding_state() -> dict[str, Any]:
    return {
        "moving_packet_count": 0,
        "spell_packet_count": 0,
        "platform_packet_count": 0,
        "moving_effect_packet_count": 0,
        "spell_effect_packet_count": 0,
        "spell_cast_packet_count": 0,
        "visible_effect_packet_count": 0,
        "target_bound_packet_count": 0,
        "positive_goal_delta_packet_count": 0,
        "counterfactual_control_packet_count": 0,
        "goal_relation_packet_count": 0,
        "goal_route_progress_packet_count": 0,
        "score_closure_packet_count": 0,
        "destructible_removed_packet_count": 0,
        "target_keys": Counter(),
        "effect_samples": [],
        "goal_relation_samples": [],
    }


def _empty_goal_anchor_perception_state() -> dict[str, Any]:
    return {
        "moving_packet_count": 0,
        "spell_packet_count": 0,
        "platform_packet_count": 0,
        "visible_effect_packet_count": 0,
        "destructible_effect_packet_count": 0,
        "explicit_target_anchor_packet_count": 0,
        "platform_goal_relation_anchor_packet_count": 0,
        "platform_goal_component_anchor_packet_count": 0,
        "goal_route_progress_anchor_packet_count": 0,
        "positive_goal_delta_packet_count": 0,
        "score_closure_anchor_packet_count": 0,
        "hardcoded_goal_color_absent_packet_count": 0,
        "anchor_sources": Counter(),
        "target_keys": Counter(),
        "anchor_samples": [],
        "missing_anchor_samples": [],
    }


def _goal_anchor_sample(
    packet: Mapping[str, Any],
    *,
    anchor_source: str,
    anchor_kind: str,
) -> dict[str, Any]:
    goal = packet.get("goal_relation")
    goal = goal if isinstance(goal, Mapping) else {}
    before_counts = packet.get("before_counts")
    before_counts = before_counts if isinstance(before_counts, Mapping) else {}
    after_counts = packet.get("after_counts")
    after_counts = after_counts if isinstance(after_counts, Mapping) else {}
    return {
        "step": _safe_int(packet.get("step")),
        "action_name": str(packet.get("action_name", "") or ""),
        "selected_site": packet.get("selected_site"),
        "candidate_types": list(packet.get("candidate_types") or []),
        "anchor_source": str(anchor_source),
        "anchor_kind": str(anchor_kind),
        "target_key": str(
            packet.get("target_key")
            or packet.get("goal_target_key")
            or packet.get("target_id")
            or ""
        ),
        "target_bbox": packet.get("target_bbox") or packet.get("goal_target_bbox"),
        "effect_region_changed_cell_count": _safe_int(
            packet.get("effect_region_changed_cell_count")
        ),
        "destructible_removed_candidate": bool(
            packet.get("destructible_removed_candidate")
        ),
        "goal_relation": {
            key: goal.get(key)
            for key in (
                "distance_before",
                "distance_after",
                "distance_delta",
                "before_goal_centroid",
                "after_goal_centroid",
            )
            if key in goal
        },
        "goal_component_count_before": _safe_int(
            before_counts.get("goal_component_count")
        ),
        "goal_component_count_after": _safe_int(
            after_counts.get("goal_component_count")
        ),
        "score_delta": _safe_int(packet.get("score_delta")),
    }


def _vector_from_any(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = value[:2]
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    return (float(x), float(y))


def _vector_key(vector: tuple[float, float]) -> str:
    x, y = vector
    if float(x).is_integer() and float(y).is_integer():
        return f"{int(x)},{int(y)}"
    return f"{round(float(x), 3)},{round(float(y), 3)}"


def _dominant_counter_item(counter: Counter[str]) -> tuple[str, int]:
    if not counter:
        return ("", 0)
    key, value = max(counter.items(), key=lambda item: (int(item[1]), str(item[0])))
    return (str(key), int(value))


def _stable_proxy_key(signature: str) -> str:
    digest = hashlib.sha1(str(signature).encode("utf-8")).hexdigest()[:12]
    return f"effect_component_proxy:{digest}"


def _identity_to_packet(identity: tuple) -> dict[str, Any]:
    color = identity[0] if len(identity) >= 1 else 0
    shape = identity[1] if len(identity) >= 2 else ()
    area = identity[2] if len(identity) >= 3 else 0
    return {
        "dominant_color": int(color),
        "shape_signature": [int(v) for v in tuple(shape or ())],
        "area": int(area),
    }


def _identity_key_from_packet(identity: Any) -> tuple[Any, ...]:
    if not isinstance(identity, dict):
        return ()
    return (
        int(identity.get("dominant_color", 0) or 0),
        tuple(int(v) for v in list(identity.get("shape_signature") or [])),
        int(identity.get("area", 0) or 0),
    )


def _identity_sort_key(identity: dict[str, Any]) -> tuple[int, tuple[int, ...], int]:
    return (
        _safe_int(identity.get("dominant_color")),
        tuple(int(v) for v in list(identity.get("shape_signature") or [])),
        _safe_int(identity.get("area")),
    )


def _identity_tuple_sort_key(identity: tuple[Any, ...]) -> tuple[int, tuple[int, ...], int]:
    return (
        int(identity[0]) if len(identity) >= 1 else 0,
        tuple(int(v) for v in (identity[1] if len(identity) >= 2 else ())),
        int(identity[2]) if len(identity) >= 3 else 0,
    )


def _identity_shape_area_key(identity: tuple[Any, ...]) -> tuple[tuple[int, ...], int]:
    shape = tuple(int(v) for v in (identity[1] if len(identity) >= 2 else ()))
    area = int(identity[2]) if len(identity) >= 3 else 0
    return (shape, area)


def _coupled_relation_key_from_moving_packet(
    packet: dict[str, Any],
) -> tuple[Any, ...] | None:
    moves = [move for move in list(packet.get("moves") or []) if isinstance(move, dict)]
    if len(moves) != 2:
        return None
    keyed_moves: list[tuple[tuple[Any, ...], str]] = []
    for move in moves:
        identity = _identity_key_from_packet(move.get("identity"))
        if not identity:
            return None
        keyed_moves.append((identity, _displacement_key(move.get("displacement"))))
    if keyed_moves[0][0] == keyed_moves[1][0]:
        return None
    if _identity_shape_area_key(keyed_moves[0][0]) != _identity_shape_area_key(
        keyed_moves[1][0]
    ):
        return None
    return (
        str(packet.get("action_name", "") or ""),
        tuple(sorted(keyed_moves, key=lambda item: _identity_tuple_sort_key(item[0]))),
    )


def _coupled_relation_key_from_relation_row(
    relation: dict[str, Any],
) -> tuple[Any, ...] | None:
    keyed_moves: list[tuple[tuple[Any, ...], str]] = []
    for item in list(relation.get("displacement_by_identity") or []):
        if not isinstance(item, dict):
            continue
        identity = _identity_key_from_packet(item.get("identity"))
        if not identity:
            continue
        keyed_moves.append((identity, _displacement_key(item.get("displacement"))))
    if len(keyed_moves) != 2:
        return None
    return (
        str(relation.get("action_name", "") or ""),
        tuple(sorted(keyed_moves, key=lambda item: _identity_tuple_sort_key(item[0]))),
    )


def _coupled_relation_key_to_bridge_row(
    relation_key: tuple[Any, ...],
    relation_state: dict[str, Any],
) -> dict[str, Any]:
    action = str(relation_key[0]) if len(relation_key) >= 1 else ""
    keyed_moves = relation_key[1] if len(relation_key) >= 2 else ()
    return {
        "action_name": action,
        "motion_symmetry_type": _motion_symmetry_type(
            [str(displacement) for _, displacement in keyed_moves]
        ),
        "observation_count": _safe_int(relation_state.get("observation_count")),
        "effect_delta_packet_count": _safe_int(
            relation_state.get("effect_delta_packet_count")
        ),
        "spell_state_delta_packet_count": _safe_int(
            relation_state.get("spell_state_delta_packet_count")
        ),
        "effect_region_changed_cell_count_sum": _safe_int(
            relation_state.get("effect_region_changed_cell_count_sum")
        ),
        "score_delta_sum": _safe_int(relation_state.get("score_delta_sum")),
        "bridge_state": (
            "pair_motion_visible_effect_bridge_observed"
            if _safe_int(relation_state.get("effect_delta_packet_count")) > 0
            else "pair_motion_without_visible_effect_delta"
        ),
        "displacement_by_identity": [
            {
                "identity": _identity_to_packet(identity),
                "displacement": _displacement_list(str(displacement)),
            }
            for identity, displacement in keyed_moves
        ],
    }


def _infer_actor_identities_from_packets(
    packets: list[dict[str, Any]],
    *,
    min_moves: int,
    commit_margin: int,
) -> dict[str, tuple[Any, ...]]:
    by_game: dict[str, dict[str, Counter[tuple[Any, ...]]]] = {}
    for packet in packets:
        game_id = str(packet.get("game_id", "") or "unknown")
        state = by_game.setdefault(
            game_id,
            {"solo": Counter(), "any": Counter()},
        )
        moves = [
            move for move in list(packet.get("moves") or []) if isinstance(move, dict)
        ]
        keys = [
            key
            for key in (_identity_key_from_packet(move.get("identity")) for move in moves)
            if key
        ]
        for key in keys:
            state["any"][key] += 1
        if len(keys) == 1:
            state["solo"][keys[0]] += 1
    committed: dict[str, tuple[Any, ...]] = {}
    for game_id, state in by_game.items():
        ranking = state["solo"] if state["solo"] else state["any"]
        top = ranking.most_common(2)
        if not top:
            continue
        best_key, best_count = top[0]
        runner_count = top[1][1] if len(top) > 1 else 0
        if best_count >= min_moves and best_count - runner_count >= commit_margin:
            committed[game_id] = best_key
    return committed


def _displacement_key(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return f"{_safe_int(value[0])},{_safe_int(value[1])}"
    return "0,0"


def _displacement_list(value: str) -> list[int]:
    try:
        x_str, y_str = value.split(",", 1)
        return [int(x_str), int(y_str)]
    except ValueError:
        return [0, 0]


def _motion_symmetry_type(displacements: list[str]) -> str:
    if len(displacements) != 2:
        return "paired_motion"
    a = _displacement_list(str(displacements[0]))
    b = _displacement_list(str(displacements[1]))
    if a[0] == -b[0] and a[1] == -b[1]:
        return "opposed_vector"
    if a[0] == 0 and b[0] == 0 and a[1] == -b[1]:
        return "opposed_axis"
    if a[1] == 0 and b[1] == 0 and a[0] == -b[0]:
        return "opposed_axis"
    if a[0] == 0 and b[0] == 0:
        return "same_axis_vertical"
    if a[1] == 0 and b[1] == 0:
        return "same_axis_horizontal"
    return "paired_motion"


def _as_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError("frame must reduce to a 2-D grid")
    return arr


def _frame_hash(frame: np.ndarray) -> str:
    arr = np.asarray(frame)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def _mechanic_state_type(packet: dict[str, Any]) -> str:
    if bool(packet.get("score_closure_candidate")):
        return "sprite_motion_score_closure"
    if bool(packet.get("multi_sprite_motion_candidate")):
        return "multi_sprite_motion"
    if bool(packet.get("single_actor_motion_candidate")):
        return "single_sprite_motion"
    if bool(packet.get("ambiguous_actor_identity_candidate")):
        return "ambiguous_sprite_identity"
    return "no_sprite_motion"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "MOVING_SPRITE_ACTOR_AMBIGUITY_SPLITTER_SCHEMA_VERSION",
    "MOVING_SPRITE_COUPLED_PAIR_GOAL_EFFECT_BRIDGE_SCHEMA_VERSION",
    "MOVING_SPRITE_COUPLED_PAIR_GOAL_SCORE_ATTRIBUTION_GUARD_SCHEMA_VERSION",
    "MOVING_SPRITE_COUPLED_MOTION_SYMMETRY_SENSOR_SCHEMA_VERSION",
    "MOVING_SPRITE_EFFECT_COMPONENT_SPATIAL_STABILITY_SENSOR_SCHEMA_VERSION",
    "MOVING_SPRITE_GOAL_ANCHOR_PERCEPTION_SENSOR_SCHEMA_VERSION",
    "MOVING_SPRITE_GUARDED_PLANNER_CONNECTION_SENSOR_SCHEMA_VERSION",
    "MOVING_SPRITE_GUARDED_PLANNER_READINESS_SCHEMA_VERSION",
    "MOVING_SPRITE_GUARDED_ACTOR_ANCHOR_POSTERIOR_SCHEMA_VERSION",
    "MOVING_SPRITE_MOTION_POSTERIOR_SCHEMA_VERSION",
    "MOVING_SPRITE_RUNTIME_TRACE_PACKET_SCHEMA_VERSION",
    "MOVING_SPRITE_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION",
    "MOVING_SPRITE_SPARSE_CHAIN_RAIL_STATE_SENSOR_SCHEMA_VERSION",
    "MOVING_SPRITE_SOFT_ANCHOR_MOTION_POSTERIOR_SCHEMA_VERSION",
    "MOVING_SPRITE_TARGET_BOUND_EFFECT_COMPONENT_SENSOR_SCHEMA_VERSION",
    "MOVING_SPRITE_TARGET_GOAL_COUNTERFACTUAL_ATTRIBUTION_SCHEMA_VERSION",
    "MOVING_SPRITE_TARGET_GOAL_BINDING_STATE_SENSOR_SCHEMA_VERSION",
    "SpriteMove",
    "build_moving_sprite_runtime_trace_packet",
    "detect_sprite_moves",
    "primary_actor_move",
    "summarize_moving_sprite_actor_ambiguity_splitter",
    "summarize_moving_sprite_coupled_pair_goal_effect_bridge",
    "summarize_moving_sprite_coupled_pair_goal_score_attribution_guard",
    "summarize_moving_sprite_coupled_motion_symmetry_sensor",
    "summarize_moving_sprite_effect_component_spatial_stability_sensor",
    "summarize_moving_sprite_goal_anchor_perception_sensor",
    "summarize_moving_sprite_guarded_planner_connection_sensor",
    "summarize_moving_sprite_guarded_actor_anchor_posterior",
    "summarize_moving_sprite_guarded_planner_readiness",
    "summarize_moving_sprite_motion_posterior",
    "summarize_moving_sprite_sparse_chain_rail_state_sensor",
    "summarize_moving_sprite_soft_anchor_motion_posterior",
    "summarize_moving_sprite_runtime_trace_packets",
    "summarize_moving_sprite_target_bound_effect_component_sensor",
    "summarize_moving_sprite_target_goal_counterfactual_attribution",
    "summarize_moving_sprite_target_goal_binding_state_sensor",
    "summary",
    "validate_moving_sprite_runtime_trace_packet",
]
