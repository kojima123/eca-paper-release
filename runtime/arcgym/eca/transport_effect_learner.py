"""TransportEffectLearner: the second solver-agnostic self-repair family.

Phase M (2026-05-31). Phase L proved the self-repairing world-model loop on
ONE effect family — hazards (reset/teleport tiles) — via `TileEffectLearner`,
lifting m0r0 source-free (40 -> 41). This module is the SECOND family: it
learns a DISPLACEMENT effect (conveyor / pusher / directed-transport tiles)
from the same loop shape, demonstrating the loop generalises beyond hazards.

    prediction error (actor moved by an UNEXPECTED but consistent vector)
      -> structural hypothesis (the stepped-into tile's VISUAL SIGNATURE
         class transports an actor by that delta)
      -> protected validation
           * a signature ever seen as plain-move floor (actor moved by the
             predicted vector over it) can NEVER be a transport tile
           * a transport must be DETERMINISTIC: the same signature must
             yield the SAME delta every time; an inconsistent delta demotes
             the class back to unknown
      -> model repair (signature -> displacement vector)
      -> a generic planner consumes the repaired model (stepping onto a
         transport cell moves the actor by the learned vector)
      -> score calibration (the delta at learn-time is recorded for
         promote/demote, mirroring TileEffectLearner)

Contrast with the hazard learner: a hazard is a BOOLEAN tile class (avoid);
a transport is a VECTOR-VALUED tile class (route through, predict the jump).
The loop scaffolding is identical; only the learned effect type differs.

There is ZERO game-specific code: no coordinates, game ids, or rendering
rules. A "signature" is the sorted (value, count) tuple of a grid cell's
pixels — palette/scale agnostic, the same key the hazard learner uses.

Usage (per solver, per step):
    learner.observe_plain_move(plain_cells, signatures)   # validation evid.
    learner.observe_transition(prev_actors, observed_actors,
                               predicted_targets, prev_signatures)
    vec = learner.transport_vector(signatures.get(cell))  # None if unknown
    cells = learner.transport_cells(signatures)           # {cell: vector}
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

Cell = tuple[int, int]
Signature = tuple
Vector = tuple[int, int]


TRANSPORT_EFFECT_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION = (
    "eca_transport_effect_runtime_trace_sensor.v1"
)
TRANSPORT_EFFECT_POSTERIOR_SCHEMA_VERSION = "eca_transport_effect_posterior.v1"
TRANSPORT_EFFECT_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION = (
    "eca_transport_effect_guarded_planner_connection.v1"
)


class TransportEffectLearner:
    """Learns deterministic displacement (transport) tile classes."""

    def __init__(self, *, predicted_step_max: int = 1) -> None:
        # The largest delta a normal move produces. A displacement strictly
        # larger than this AND different from the predicted move is the
        # candidate transport signal.
        self._predicted_step_max = max(1, int(predicted_step_max))
        # Confirmed transport classes: signature -> displacement vector.
        self._transport_vector: dict[Signature, Vector] = {}
        # Signatures seen as plain-move floor (actor moved by the predicted
        # vector over them) -> can never be a transport tile. Protected.
        self._plain_signatures: set[Signature] = set()
        # Signatures demoted for inconsistent deltas -> never re-promote
        # (a non-deterministic tile is not a transport we can plan through).
        self._inconsistent_signatures: set[Signature] = set()
        self._suspect: Counter[Signature] = Counter()
        # Displacement magnitude recorded at confirm time, for calibration.
        self._sig_delta_at_learn: dict[Signature, Vector] = {}
        # Episode-local action displacement votes used by the generic
        # stride-lattice route planner. This is not a transport effect: it
        # records the modal actor displacement produced by each directional
        # action when ordinary movement succeeds.
        self._action_displacements: dict[str, Counter[Vector]] = defaultdict(Counter)

    # -- validation evidence -------------------------------------------------

    def observe_plain_move(
        self,
        plain_cells: Iterable[Cell],
        signatures: dict[Cell, Signature],
    ) -> None:
        """Record cells the actor moved over by the PREDICTED vector (no
        transport) -> their signature class is plain floor, not a transport.
        Retract any earlier false positive."""
        for cell in plain_cells:
            sig = signatures.get(cell)
            if sig is not None:
                self._plain_signatures.add(sig)
                self._transport_vector.pop(sig, None)

    # -- prediction-error -> hypothesis -------------------------------------

    def observe_transition(
        self,
        prev_actors: list[Cell],
        observed_actors: list[Cell],
        predicted_targets: list[Cell],
        prev_signatures: dict[Cell, Signature],
        *,
        progress_score: int = 0,
    ) -> bool:
        """Compare predicted vs observed actor positions. When an actor lands
        a consistent unexpected displacement after stepping toward a tile,
        hypothesise that tile's signature transports by the OBSERVED delta
        (observed - predicted_target), subject to protected validation.

        Returns True if a transport displacement was detected this step.
        """
        if not prev_actors or not observed_actors:
            return False
        n = min(len(prev_actors), len(observed_actors), len(predicted_targets))
        detected = False
        for i in range(n):
            pa = prev_actors[i]
            oa = observed_actors[i]
            target = predicted_targets[i]
            if target == pa:
                continue  # no move was attempted into a new tile
            # The transport delta is how far the actor ended up FROM where a
            # plain move would have put it (the predicted target).
            delta = (oa[0] - target[0], oa[1] - target[1])
            if delta == (0, 0):
                # Plain move succeeded -> the target tile is plain floor.
                sig = prev_signatures.get(target)
                if sig is not None:
                    self._plain_signatures.add(sig)
                    self._transport_vector.pop(sig, None)
                continue
            # A real displacement happened. Blame the stepped-into tile.
            sig = prev_signatures.get(target)
            if sig is None or sig in self._plain_signatures:
                continue
            if sig in self._inconsistent_signatures:
                continue
            self._suspect[sig] += 1
            prior = self._transport_vector.get(sig)
            if prior is None:
                self._transport_vector[sig] = delta
                self._sig_delta_at_learn[sig] = delta
                detected = True
            elif prior == delta:
                detected = True  # consistent re-confirmation
            else:
                # Inconsistent delta -> not a deterministic transport.
                self._transport_vector.pop(sig, None)
                self._inconsistent_signatures.add(sig)
        return detected

    def observe_move_displacement(self, action_name: str, delta: Vector) -> None:
        """Record an observed actor movement vector for a directional action.

        The caller is responsible for feeding only non-reset moves. Keeping
        this in the learner lets the planner reuse the same effect-model
        diagnostics and protected modal-vote pattern without adding a new
        state object in the loop.
        """

        action = str(action_name or "").strip()
        if not action:
            return
        dx, dy = int(delta[0]), int(delta[1])
        if dx == 0 and dy == 0:
            return
        self._action_displacements[action][(dx, dy)] += 1

    def stride_lattice(
        self,
        *,
        min_observations: int = 5,
        min_dominance: float = 0.8,
        max_manhattan: int = 10,
    ) -> dict[str, Vector]:
        """Return stable per-action modal stride vectors.

        Admission is deliberately conservative: the modal vector must have
        enough support, dominate the action's displacement observations, and
        stay within the normal non-reset move range.
        """

        admitted: dict[str, Vector] = {}
        min_obs = max(1, int(min_observations))
        threshold = max(0.0, min(1.0, float(min_dominance)))
        max_dist = max(1, int(max_manhattan))
        for action, votes in self._action_displacements.items():
            total = sum(int(count) for count in votes.values())
            if total <= 0:
                continue
            vector, count = votes.most_common(1)[0]
            distance = abs(int(vector[0])) + abs(int(vector[1]))
            if int(count) < min_obs:
                continue
            if distance < 2 or distance > max_dist:
                continue
            if float(count) / float(total) < threshold:
                continue
            admitted[str(action)] = (int(vector[0]), int(vector[1]))
        return admitted

    # -- repaired model query ------------------------------------------------

    def transport_vector(self, signature: Signature | None) -> Vector | None:
        """Return the learned displacement vector for a signature class, or
        None if it is unknown / plain / demoted."""
        if signature is None or signature in self._plain_signatures:
            return None
        return self._transport_vector.get(signature)

    def transport_cells(
        self,
        signatures: dict[Cell, Signature],
        *,
        exclude: Iterable[Cell] = (),
    ) -> dict[Cell, Vector]:
        """Return every cell whose signature is a confirmed transport class
        mapped to its displacement vector, minus `exclude`."""
        out: dict[Cell, Vector] = {}
        for cell, sig in signatures.items():
            if sig in self._plain_signatures:
                continue
            vec = self._transport_vector.get(sig)
            if vec is not None:
                out[cell] = vec
        for cell in exclude:
            out.pop(cell, None)
        return out

    # -- introspection -------------------------------------------------------

    def transport_signatures(self) -> dict[Signature, Vector]:
        return dict(self._transport_vector)

    def summary(self) -> dict[str, Any]:
        return {
            "transport_signatures": sorted(self._transport_vector.items()),
            "plain_signature_count": len(self._plain_signatures),
            "inconsistent_count": len(self._inconsistent_signatures),
            "suspect_count": len(self._suspect),
            "stride_lattice": {
                action: [int(vector[0]), int(vector[1])]
                for action, vector in sorted(self.stride_lattice().items())
            },
            "stride_observation_counts": {
                action: int(sum(votes.values()))
                for action, votes in sorted(self._action_displacements.items())
            },
        }


def summarize_transport_effect_runtime_trace_packets(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_consistent_observations: int = 2,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize transport-effect evidence from moving-sprite trace packets.

    This is a runtime-trace bridge only. It consumes already-emitted ECA
    moving-sprite diagnostics and records displacement candidates that exceed
    a normal one-step move. It does not infer actions, rank candidates, or
    update the planner.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    normal_step = max(1, int(normal_step_max))
    consistent_min = max(1, int(min_consistent_observations))
    by_game: dict[str, Counter[str]] = {}
    by_identity_vector: dict[tuple[str, str, Vector], Counter[str]] = defaultdict(Counter)
    vectors_by_identity: dict[tuple[str, str], Counter[Vector]] = defaultdict(Counter)
    displacement_histogram: Counter[str] = Counter()
    candidate_samples: list[dict[str, Any]] = []
    large_packet_keys: set[tuple[str, int, str]] = set()
    packet_with_moves_count = 0

    for packet in packet_rows:
        game_id = str(packet.get("game_id", "") or "unknown").strip().lower()
        step = _safe_int(packet.get("step"))
        state_hash = str(packet.get("state_hash", "") or "")
        packet_key = (game_id, step, state_hash)
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["packet_count"] += 1
        moves = packet.get("moves") if isinstance(packet.get("moves"), list) else []
        if moves:
            packet_with_moves_count += 1
            game_counter["packet_with_moves_count"] += 1
        score_delta = _safe_int(packet.get("score_delta"))
        score_linked = bool(packet.get("score_closure_candidate")) or score_delta > 0
        action_name = str(packet.get("action_name", "") or "")

        for move in moves:
            if not isinstance(move, Mapping):
                continue
            displacement = _vector_from_value(move.get("displacement"))
            if displacement is None:
                displacement = _displacement_from_cells(
                    move.get("prev_cell"),
                    move.get("cur_cell"),
                )
            if displacement is None:
                continue
            dx, dy = displacement
            displacement_histogram[f"{dx},{dy}"] += 1
            if abs(dx) + abs(dy) <= normal_step:
                continue
            identity_key = _identity_key_from_move(move)
            relation_key = (game_id, identity_key, displacement)
            by_identity_vector[relation_key]["observation_count"] += 1
            vectors_by_identity[(game_id, identity_key)][displacement] += 1
            game_counter["unexpected_displacement_candidate_count"] += 1
            if score_linked:
                by_identity_vector[relation_key]["score_linked_count"] += 1
                game_counter["score_linked_candidate_count"] += 1
            large_packet_keys.add(packet_key)
            if len(candidate_samples) < sample_limit:
                candidate_samples.append(
                    {
                        "game_id": game_id,
                        "step": step,
                        "action_name": action_name,
                        "identity_key": identity_key,
                        "displacement": [dx, dy],
                        "score_delta": score_delta,
                        "score_linked": score_linked,
                    }
                )

    relation_rows: list[dict[str, Any]] = []
    consistent_candidate_count = 0
    score_linked_candidate_count = 0
    inconsistent_identity_count = 0
    consistent_games: set[str] = set()
    score_linked_games: set[str] = set()
    candidate_games: set[str] = set()
    inconsistent_games: set[str] = set()
    game_displacements: dict[str, Counter[str]] = defaultdict(Counter)

    for (game_id, identity_key, displacement), counter in sorted(
        by_identity_vector.items()
    ):
        observation_count = int(counter.get("observation_count", 0))
        score_linked_count = int(counter.get("score_linked_count", 0))
        vector_options = vectors_by_identity[(game_id, identity_key)]
        ambiguous_vectors = len(vector_options) > 1
        consistent = observation_count >= consistent_min and not ambiguous_vectors
        score_linked = score_linked_count > 0
        if consistent:
            consistent_candidate_count += 1
            consistent_games.add(game_id)
            by_game.setdefault(game_id, Counter())["consistent_candidate_count"] += 1
        if score_linked:
            score_linked_candidate_count += 1
            score_linked_games.add(game_id)
        if ambiguous_vectors:
            inconsistent_games.add(game_id)
        candidate_games.add(game_id)
        game_displacements[game_id][f"{displacement[0]},{displacement[1]}"] += (
            observation_count
        )
        relation_rows.append(
            {
                "game_id": game_id,
                "identity_key": identity_key,
                "displacement": [displacement[0], displacement[1]],
                "observation_count": observation_count,
                "score_linked_count": score_linked_count,
                "consistent": consistent,
                "ambiguous_vectors": ambiguous_vectors,
            }
        )

    for (game_id, identity_key), vector_counter in vectors_by_identity.items():
        if len(vector_counter) > 1:
            inconsistent_identity_count += 1
            by_game.setdefault(game_id, Counter())["inconsistent_candidate_count"] += 1

    for game_id in candidate_games:
        by_game.setdefault(game_id, Counter())["transport_relation_candidate_count"] = (
            sum(
                1
                for row in relation_rows
                if row.get("game_id") == game_id
            )
        )

    for game_id in score_linked_games:
        by_game.setdefault(game_id, Counter())["score_linked_relation_count"] = sum(
            1
            for row in relation_rows
            if row.get("game_id") == game_id and row.get("score_linked_count", 0) > 0
        )

    candidate_count = len(relation_rows)
    if score_linked_candidate_count > 0:
        promotion_state = (
            "transport_effect_runtime_trace_connection_score_linked_observed"
        )
    elif candidate_count > 0:
        promotion_state = "transport_effect_runtime_trace_connection_observed"
    elif packet_rows:
        promotion_state = "transport_effect_runtime_trace_connection_no_candidates"
    else:
        promotion_state = "transport_effect_runtime_trace_connection_no_packets"

    return _json_safe(
        {
            "schema_version": TRANSPORT_EFFECT_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": len(packet_rows),
            "packet_with_moves_count": int(packet_with_moves_count),
            "large_displacement_packet_count": len(large_packet_keys),
            "unexpected_displacement_candidate_count": sum(
                counter.get("unexpected_displacement_candidate_count", 0)
                for counter in by_game.values()
            ),
            "transport_relation_candidate_count": candidate_count,
            "consistent_transport_candidate_count": int(consistent_candidate_count),
            "score_linked_transport_candidate_count": int(score_linked_candidate_count),
            "inconsistent_transport_candidate_count": int(inconsistent_identity_count),
            "transport_candidate_game_count": len(candidate_games),
            "consistent_transport_candidate_game_count": len(consistent_games),
            "score_linked_transport_candidate_game_count": len(score_linked_games),
            "inconsistent_transport_candidate_game_count": len(inconsistent_games),
            "normal_step_max": normal_step,
            "min_consistent_observations": consistent_min,
            "promotion_state": promotion_state,
            "runtime_trace_connection_observed": candidate_count > 0,
            "by_game": {
                game_id: {
                    **dict(sorted(counter.items())),
                    "dominant_displacements": dict(
                        game_displacements.get(game_id, Counter()).most_common(6)
                    ),
                }
                for game_id, counter in sorted(by_game.items())
            },
            "displacement_histogram": dict(sorted(displacement_histogram.items())),
            "relations": relation_rows[:sample_limit],
            "samples": candidate_samples,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_transport_effect_posterior(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_confirmed_observations: int = 2,
    dominance_margin: int = 2,
    min_score_linked_observations: int = 1,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Build a conservative posterior over transport identity/vector evidence.

    The posterior consumes the same moving-sprite runtime trace packets as the
    Phase692 bridge. A relation is confirmed only when a dominant displacement
    vector has enough observations and is not tied with competing vectors for
    the same identity. Score-linked confirmations are separated from
    motion-only confirmations so a later guard can decide whether the relation
    is safe to expose to planning.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    normal_step = max(1, int(normal_step_max))
    confirmed_min = max(1, int(min_confirmed_observations))
    margin = max(0, int(dominance_margin))
    score_min = max(1, int(min_score_linked_observations))
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    by_game: dict[str, Counter[str]] = {}
    packet_with_moves_count = 0
    large_packet_keys: set[tuple[str, int, str]] = set()

    for packet in packet_rows:
        game_id = str(packet.get("game_id", "") or "unknown").strip().lower()
        step = _safe_int(packet.get("step"))
        state_hash = str(packet.get("state_hash", "") or "")
        packet_key = (game_id, step, state_hash)
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["packet_count"] += 1
        moves = packet.get("moves") if isinstance(packet.get("moves"), list) else []
        if moves:
            packet_with_moves_count += 1
            game_counter["packet_with_moves_count"] += 1
        score_delta = _safe_int(packet.get("score_delta"))
        score_linked = bool(packet.get("score_closure_candidate")) or score_delta > 0
        action_name = str(packet.get("action_name", "") or "")

        for move in moves:
            if not isinstance(move, Mapping):
                continue
            displacement = _vector_from_value(move.get("displacement"))
            if displacement is None:
                displacement = _displacement_from_cells(
                    move.get("prev_cell"),
                    move.get("cur_cell"),
                )
            if displacement is None:
                continue
            if abs(displacement[0]) + abs(displacement[1]) <= normal_step:
                continue
            identity_key = _identity_key_from_move(move)
            row_key = (game_id, identity_key)
            state = by_identity.setdefault(
                row_key,
                {
                    "vector_counts": Counter(),
                    "score_linked_counts": Counter(),
                    "score_delta_sum": 0,
                    "actions": Counter(),
                    "steps": set(),
                },
            )
            state["vector_counts"][displacement] += 1
            state["actions"][action_name] += 1
            state["steps"].add(step)
            if score_linked:
                state["score_linked_counts"][displacement] += 1
                state["score_delta_sum"] += score_delta
            large_packet_keys.add(packet_key)
            game_counter["large_displacement_observation_count"] += 1

    rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    confirmed_games: set[str] = set()
    score_linked_games: set[str] = set()
    ambiguous_games: set[str] = set()
    insufficient_games: set[str] = set()

    for (game_id, identity_key), state in sorted(by_identity.items()):
        vector_counts: Counter[Vector] = state["vector_counts"]
        score_counts: Counter[Vector] = state["score_linked_counts"]
        ranked = vector_counts.most_common()
        dominant_vector = ranked[0][0]
        dominant_count = int(ranked[0][1])
        runner_count = int(ranked[1][1]) if len(ranked) > 1 else 0
        total_count = int(sum(vector_counts.values()))
        dominant_score_count = int(score_counts.get(dominant_vector, 0))
        total_score_count = int(sum(score_counts.values()))
        competing_vector_confirmed = runner_count >= confirmed_min
        dominance_ok = dominant_count >= confirmed_min and (
            runner_count == 0 or dominant_count - runner_count >= margin
        ) and not competing_vector_confirmed
        if dominant_count < confirmed_min:
            posterior_state = "insufficient_evidence"
            insufficient_games.add(game_id)
        elif not dominance_ok:
            posterior_state = "ambiguous_inconsistent_vectors"
            ambiguous_games.add(game_id)
        elif dominant_score_count >= score_min:
            posterior_state = "confirmed_score_linked"
            confirmed_games.add(game_id)
            score_linked_games.add(game_id)
        else:
            posterior_state = "confirmed_motion_only"
            confirmed_games.add(game_id)

        state_counts[posterior_state] += 1
        by_game.setdefault(game_id, Counter())["posterior_entry_count"] += 1
        by_game[game_id][f"{posterior_state}_count"] += 1
        rows.append(
            {
                "game_id": game_id,
                "identity_key": identity_key,
                "posterior_state": posterior_state,
                "dominant_vector": [dominant_vector[0], dominant_vector[1]],
                "dominant_observation_count": dominant_count,
                "runner_up_observation_count": runner_count,
                "total_observation_count": total_count,
                "dominant_score_linked_count": dominant_score_count,
                "total_score_linked_count": total_score_count,
                "score_delta_sum": int(state["score_delta_sum"]),
                "vector_histogram": {
                    f"{vec[0]},{vec[1]}": int(count)
                    for vec, count in sorted(vector_counts.items())
                },
                "action_histogram": dict(sorted(state["actions"].items())),
                "step_count": len(state["steps"]),
                "admission_ready_for_guard": posterior_state
                == "confirmed_score_linked",
            }
        )

    posterior_entry_count = len(rows)
    confirmed_score_linked_count = int(state_counts.get("confirmed_score_linked", 0))
    confirmed_motion_only_count = int(state_counts.get("confirmed_motion_only", 0))
    ambiguous_count = int(state_counts.get("ambiguous_inconsistent_vectors", 0))
    insufficient_count = int(state_counts.get("insufficient_evidence", 0))
    if confirmed_score_linked_count > 0:
        promotion_state = "transport_effect_posterior_score_linked_confirmed"
    elif confirmed_motion_only_count > 0:
        promotion_state = "transport_effect_posterior_motion_confirmed"
    elif posterior_entry_count > 0:
        promotion_state = "transport_effect_posterior_candidates_unconfirmed"
    elif packet_rows:
        promotion_state = "transport_effect_posterior_no_candidates"
    else:
        promotion_state = "transport_effect_posterior_no_packets"

    return _json_safe(
        {
            "schema_version": TRANSPORT_EFFECT_POSTERIOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": len(packet_rows),
            "packet_with_moves_count": int(packet_with_moves_count),
            "large_displacement_packet_count": len(large_packet_keys),
            "posterior_entry_count": posterior_entry_count,
            "confirmed_transport_count": confirmed_score_linked_count
            + confirmed_motion_only_count,
            "confirmed_score_linked_count": confirmed_score_linked_count,
            "confirmed_motion_only_count": confirmed_motion_only_count,
            "ambiguous_inconsistent_vector_count": ambiguous_count,
            "insufficient_evidence_count": insufficient_count,
            "confirmed_transport_game_count": len(confirmed_games),
            "score_linked_transport_game_count": len(score_linked_games),
            "ambiguous_transport_game_count": len(ambiguous_games),
            "insufficient_transport_game_count": len(insufficient_games),
            "normal_step_max": normal_step,
            "min_confirmed_observations": confirmed_min,
            "dominance_margin": margin,
            "min_score_linked_observations": score_min,
            "promotion_state": promotion_state,
            "posterior_observed": posterior_entry_count > 0,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "rows": rows[:sample_limit],
            "samples": rows[:sample_limit],
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_transport_effect_guarded_planner_connection(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_confirmed_observations: int = 2,
    dominance_margin: int = 2,
    min_score_linked_observations: int = 1,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Shadow admission guard for transport-effect planner connection.

    The guard admits only score-linked, posterior-confirmed transport
    relations. Motion-only confirmations, ambiguous competing vectors, and
    insufficient evidence are kept blocked. This records admission state only;
    it does not generate actions, rank candidates, or alter runtime planning.
    """

    posterior = summarize_transport_effect_posterior(
        packets,
        normal_step_max=normal_step_max,
        min_confirmed_observations=min_confirmed_observations,
        dominance_margin=dominance_margin,
        min_score_linked_observations=min_score_linked_observations,
        sample_limit=100000,
    )
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    admitted_count = 0
    blocked_count = 0
    blocked_motion_only_count = 0
    blocked_ambiguous_count = 0
    blocked_insufficient_count = 0
    blocked_other_count = 0
    admitted_games: set[str] = set()
    blocked_games: set[str] = set()
    by_game: dict[str, Counter[str]] = {}

    for posterior_row in list(posterior.get("rows") or []):
        if not isinstance(posterior_row, Mapping):
            continue
        game_id = str(posterior_row.get("game_id", "") or "unknown")
        state = str(posterior_row.get("posterior_state", "") or "")
        admitted = state == "confirmed_score_linked"
        blocked_reasons: list[str] = []
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["posterior_entry_count"] += 1
        if admitted:
            connection_state = "transport_effect_planner_connection_shadow_admitted"
            admitted_count += 1
            admitted_games.add(game_id)
            game_counter["admitted_shadow_count"] += 1
        else:
            blocked_count += 1
            blocked_games.add(game_id)
            if state == "confirmed_motion_only":
                blocked_reasons.append("missing_score_linked_confirmation")
                blocked_motion_only_count += 1
                game_counter["blocked_motion_only_count"] += 1
            elif state == "ambiguous_inconsistent_vectors":
                blocked_reasons.append("competing_transport_vectors")
                blocked_ambiguous_count += 1
                game_counter["blocked_ambiguous_vector_count"] += 1
            elif state == "insufficient_evidence":
                blocked_reasons.append("insufficient_transport_evidence")
                blocked_insufficient_count += 1
                game_counter["blocked_insufficient_evidence_count"] += 1
            else:
                blocked_reasons.append("not_score_linked_confirmed")
                blocked_other_count += 1
                game_counter["blocked_other_count"] += 1
            connection_state = (
                "transport_effect_planner_connection_blocked_"
                + blocked_reasons[0]
            )
        row = {
            "game_id": game_id,
            "identity_key": str(posterior_row.get("identity_key", "") or ""),
            "connection_state": connection_state,
            "admitted_shadow": admitted,
            "blocked_reasons": blocked_reasons,
            "posterior_state": state,
            "dominant_vector": list(posterior_row.get("dominant_vector") or []),
            "dominant_observation_count": _safe_int(
                posterior_row.get("dominant_observation_count")
            ),
            "runner_up_observation_count": _safe_int(
                posterior_row.get("runner_up_observation_count")
            ),
            "dominant_score_linked_count": _safe_int(
                posterior_row.get("dominant_score_linked_count")
            ),
            "total_score_linked_count": _safe_int(
                posterior_row.get("total_score_linked_count")
            ),
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if admitted_count > 0:
        promotion_state = "transport_effect_guarded_planner_connection_shadow_admitted"
    elif blocked_motion_only_count > 0:
        promotion_state = (
            "transport_effect_guarded_planner_connection_blocked_motion_only"
        )
    elif blocked_ambiguous_count > 0:
        promotion_state = (
            "transport_effect_guarded_planner_connection_blocked_competing_vectors"
        )
    elif blocked_insufficient_count > 0:
        promotion_state = (
            "transport_effect_guarded_planner_connection_blocked_insufficient_evidence"
        )
    elif blocked_count > 0:
        promotion_state = "transport_effect_guarded_planner_connection_blocked"
    elif int(posterior.get("packet_count", 0) or 0) > 0:
        promotion_state = "transport_effect_guarded_planner_connection_no_candidates"
    else:
        promotion_state = "transport_effect_guarded_planner_connection_no_packets"

    return _json_safe(
        {
            "schema_version": TRANSPORT_EFFECT_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(posterior.get("packet_count", 0) or 0),
            "posterior_entry_count": int(
                posterior.get("posterior_entry_count", 0) or 0
            ),
            "planner_connection_admitted_count": int(admitted_count),
            "planner_connection_blocked_count": int(blocked_count),
            "planner_connection_admitted_game_count": len(admitted_games),
            "planner_connection_blocked_game_count": len(blocked_games),
            "blocked_motion_only_count": int(blocked_motion_only_count),
            "blocked_ambiguous_vector_count": int(blocked_ambiguous_count),
            "blocked_insufficient_evidence_count": int(blocked_insufficient_count),
            "blocked_other_count": int(blocked_other_count),
            "posterior_promotion_state": str(
                posterior.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "rows": rows[:sample_limit],
            "samples": samples,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def _identity_key_from_move(move: Mapping[str, Any]) -> str:
    identity = move.get("identity")
    if not isinstance(identity, Mapping):
        return "unknown"
    dominant_color = identity.get("dominant_color", identity.get("color", "unknown"))
    area = identity.get("area", move.get("area", "unknown"))
    shape = identity.get("shape_signature")
    if isinstance(shape, (list, tuple)):
        shape_key = ",".join(str(_safe_int(item)) for item in shape[:8])
    else:
        shape_key = str(shape or "")
    return f"color={dominant_color}|area={area}|shape={shape_key}"


def _vector_from_value(value: Any) -> Vector | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return (_safe_int(value[0]), _safe_int(value[1]))


def _displacement_from_cells(prev_cell: Any, cur_cell: Any) -> Vector | None:
    prev = _vector_from_value(prev_cell)
    cur = _vector_from_value(cur_cell)
    if prev is None or cur is None:
        return None
    return (cur[0] - prev[0], cur[1] - prev[1])


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


__all__ = [
    "TRANSPORT_EFFECT_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION",
    "TRANSPORT_EFFECT_POSTERIOR_SCHEMA_VERSION",
    "TRANSPORT_EFFECT_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION",
    "TransportEffectLearner",
    "summarize_transport_effect_guarded_planner_connection",
    "summarize_transport_effect_posterior",
    "summarize_transport_effect_runtime_trace_packets",
]
