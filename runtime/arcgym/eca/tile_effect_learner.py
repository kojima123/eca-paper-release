"""TileEffectLearner: solver-agnostic self-repairing transition model.

Phase L (2026-05-30). The first proven path to private-holdout transfer
is a loop that repairs the agent's OWN world model from reality:

    prediction error
      -> structural hypothesis (tile class by visual signature)
      -> protected validation (walkable evidence rejects false positives)
      -> model repair (hazard tile set)
      -> generic planner consumes the repaired model
      -> score calibration

This module is the SOLVER-AGNOSTIC core of that loop. It owns NO forward
model and NO planner — each solver supplies its own prediction (where it
expected actors to go) and its own perception (walkable cells + per-cell
visual signatures). The learner only does the generic part:

  - detect a reset/hazard from a prediction error (an actor jumped
    discontinuously instead of moving/staying as predicted),
  - hypothesise that the stepped-into tile's VISUAL SIGNATURE class is a
    hazard, generalising from one observation to every same-class cell,
  - REJECT that hypothesis if the signature is ever observed in the
    walkable set (a cell an actor can stand on cannot be a reset tile) —
    this kills the false positives a single noisy reset produces,
  - expose the confirmed hazard cells for the solver to avoid.

There is ZERO game-specific code here: no coordinates, no game id, no
rendering rule. A "signature" is just the sorted (value, count) tuple of
a grid cell's pixels. This is why it can transfer to unseen games whose
hazards render differently.

Usage (per solver, per step):
    learner.observe_safe(walkable, signatures)          # validation evid.
    learner.observe_transition(prev_actors, observed_actors,
                               predicted_targets, prev_signatures)
    hazard_cells = learner.hazard_cells(signatures, exclude=current_actors)
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

Cell = tuple[int, int]
Signature = tuple


TILE_EFFECT_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION = (
    "eca_tile_effect_runtime_trace_sensor.v1"
)
TILE_EFFECT_RUNTIME_TRACE_PACKET_SCHEMA_VERSION = (
    "eca_tile_effect_runtime_trace_packet.v1"
)
TILE_EFFECT_POSTERIOR_SCHEMA_VERSION = "eca_tile_effect_posterior.v1"
TILE_EFFECT_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION = (
    "eca_tile_effect_guarded_planner_connection.v1"
)
TILE_EFFECT_GUARD_PRECISION_REVIEW_SCHEMA_VERSION = (
    "eca_tile_effect_guard_precision_review.v1"
)
TILE_EFFECT_GUARD_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION = (
    "eca_tile_effect_guard_counterfactual_control.v1"
)
TILE_EFFECT_FAMILY_SCOPE_NEGATIVE_CONTROL_SCHEMA_VERSION = (
    "eca_tile_effect_family_scope_negative_control.v1"
)
TILE_EFFECT_HAZARD_SOURCE_REPAIR_BRIDGE_SCHEMA_VERSION = (
    "eca_tile_effect_hazard_source_repair_bridge.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROPOSAL_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_proposal.v1"
)
TILE_EFFECT_MODEL_REPAIR_WRITE_GUARD_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_write_guard.v1"
)
TILE_EFFECT_MODEL_REPAIR_WRITE_AUDIT_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_write_audit.v1"
)
TILE_EFFECT_MODEL_REPAIR_RUNTIME_PROBE_PROPOSAL_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_runtime_probe_proposal.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_EXECUTION_GATE_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_protected_probe_execution_gate.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_BRIDGE_GUARD_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_protected_probe_runtime_bridge_guard.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_BRIDGE_OUTCOME_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_protected_probe_runtime_bridge_outcome.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_EXECUTOR_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_protected_probe_runtime_executor.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_OUTCOME_POSTERIOR_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_protected_probe_outcome_posterior.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RANK_GUARD_SHADOW_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_protected_probe_rank_guard_shadow.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RANK_GUARD_RUNTIME_GUARD_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_protected_probe_rank_guard_runtime_guard.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_FOCUSED_RUNTIME_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_ACTION_SELECTION_GUARD_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_protected_probe_live_rank_guard_action_selection_guard.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_POSITIVE_ACTION_SELECTION_ADMISSION_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_admission.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_POSITIVE_ACTION_SELECTION_SCORE_ATTRIBUTION_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_SCORE_ATTRIBUTED_POSITIVE_RANK_PROMOTION_GUARD_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_positive_rank_promotion_guard.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_SCORE_ATTRIBUTED_CANDIDATE_DISCOVERY_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery.v1"
)
TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_SCORE_ATTRIBUTED_CANDIDATE_FOCUSED_RUNTIME_REPLAY_SCHEMA_VERSION = (
    "eca_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_focused_runtime_replay.v1"
)


class TileEffectLearner:
    """Learns hazard (reset) tile classes from prediction errors."""

    def __init__(self, *, jump_threshold: int = 1) -> None:
        # A move shifts an actor by at most 1 cell; a larger jump between
        # consecutive observations is a reset/teleport (prediction error).
        self._jump_threshold = max(1, int(jump_threshold))
        # Confirmed hazard signature classes.
        self._hazard_signatures: set[Signature] = set()
        # Signatures observed in the walkable set (an actor CAN stand on
        # them) -> can never be a hazard. Protected validation.
        self._safe_signatures: set[Signature] = set()
        # Per-signature first-seen count, for diagnostics.
        self._hazard_suspect: Counter[Signature] = Counter()
        # Score (or any monotone progress scalar) at the moment a hazard
        # class was first confirmed, for promote/demote calibration.
        self._sig_score_at_learn: dict[Signature, int] = {}

    # -- validation evidence -------------------------------------------------

    def observe_safe(
        self,
        walkable: Iterable[Cell],
        signatures: dict[Cell, Signature],
    ) -> None:
        """Record that every walkable cell's signature is a class an actor
        can stand on -> not a hazard. Retract any earlier false positive."""
        for cell in walkable:
            sig = signatures.get(cell)
            if sig is not None:
                self._safe_signatures.add(sig)
                self._hazard_signatures.discard(sig)

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
        """Compare predicted vs observed actor positions. On a reset
        (discontinuous jump), hypothesise the stepped-into target tiles'
        signature classes are hazards, subject to protected validation.

        Returns True if a reset was detected this step.
        """
        if not prev_actors or not observed_actors:
            return False
        n = min(len(prev_actors), len(observed_actors))
        reset = False
        for i in range(n):
            pa = prev_actors[i]
            oa = observed_actors[i]
            jump = abs(oa[0] - pa[0]) + abs(oa[1] - pa[1])
            if jump > self._jump_threshold:
                reset = True
                break
        if not reset:
            return False
        # Blame the tiles the actors tried to enter. Validation will
        # reject the ones that are actually walkable (floor / actor).
        for i, target in enumerate(predicted_targets):
            prev = prev_actors[i] if i < len(prev_actors) else None
            if target == prev:
                continue
            sig = prev_signatures.get(target)
            if sig is None or sig in self._safe_signatures:
                continue
            self._hazard_suspect[sig] += 1
            if sig not in self._hazard_signatures:
                self._hazard_signatures.add(sig)
                self._sig_score_at_learn[sig] = int(progress_score)
        return True

    # -- repaired model query ------------------------------------------------

    def hazard_cells(
        self,
        signatures: dict[Cell, Signature],
        *,
        exclude: Iterable[Cell] = (),
    ) -> set[Cell]:
        """Return every cell whose signature is a confirmed hazard class,
        minus `exclude` (typically the actors' own current cells)."""
        confirmed = self._hazard_signatures - self._safe_signatures
        if not confirmed:
            return set()
        out = {
            cell for cell, sig in signatures.items() if sig in confirmed
        }
        for cell in exclude:
            out.discard(cell)
        return out

    # -- introspection -------------------------------------------------------

    def hazard_signatures(self) -> set[Signature]:
        return set(self._hazard_signatures - self._safe_signatures)

    def summary(self) -> dict[str, Any]:
        return {
            "hazard_signatures": sorted(
                self._hazard_signatures - self._safe_signatures
            ),
            "safe_signature_count": len(self._safe_signatures),
            "suspect_count": len(self._hazard_suspect),
        }


def summarize_tile_effect_runtime_trace_packets(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize reset/hazard evidence from moving-sprite trace packets.

    This is a runtime-trace bridge only. It consumes already-emitted ECA
    moving-sprite diagnostics and records discontinuous jumps that look like
    tile-effect resets. It does not infer tile signatures, rank candidates,
    generate actions, or update a planner.

    A reset-like event is a sprite move whose Manhattan displacement exceeds
    the normal one-step move. The first bridge groups those events by the
    observed return anchor, because a hazard reset often sends the actor back
    to a stable start cell. Signature-level posterior remains a later phase.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    normal_step = max(1, int(normal_step_max))
    anchor_min = max(1, int(min_anchor_observations))
    by_game: dict[str, Counter[str]] = {}
    by_anchor: dict[tuple[str, str, Cell], Counter[str]] = defaultdict(Counter)
    displacement_histogram: Counter[str] = Counter()
    reset_packet_keys: set[tuple[str, int, str]] = set()
    candidate_samples: list[dict[str, Any]] = []
    packet_with_moves_count = 0

    for packet in packet_rows:
        game_id = str(packet.get("game_id", "") or "unknown").strip().lower()
        step = _safe_int(packet.get("step"))
        state_hash = str(packet.get("state_hash", "") or "")
        packet_key = (game_id, step, state_hash)
        action_name = str(packet.get("action_name", "") or "")
        score_delta = _safe_int(packet.get("score_delta"))
        score_linked = bool(packet.get("score_closure_candidate")) or score_delta > 0
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["packet_count"] += 1
        moves = packet.get("moves") if isinstance(packet.get("moves"), list) else []
        if moves:
            packet_with_moves_count += 1
            game_counter["packet_with_moves_count"] += 1

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
            anchor = _vector_from_value(move.get("cur_cell"))
            if anchor is None:
                continue
            identity_key = _identity_key_from_move(move)
            key = (game_id, identity_key, anchor)
            by_anchor[key]["observation_count"] += 1
            by_anchor[key]["score_delta_sum"] += score_delta
            by_anchor[key]["score_linked_count"] += 1 if score_linked else 0
            by_anchor[key][f"action:{action_name}"] += 1
            game_counter["reset_jump_observation_count"] += 1
            if score_linked:
                game_counter["score_linked_reset_observation_count"] += 1
            reset_packet_keys.add(packet_key)
            if len(candidate_samples) < sample_limit:
                candidate_samples.append(
                    {
                        "game_id": game_id,
                        "step": step,
                        "action_name": action_name,
                        "identity_key": identity_key,
                        "return_anchor": [anchor[0], anchor[1]],
                        "displacement": [dx, dy],
                        "score_delta": score_delta,
                        "score_linked": score_linked,
                    }
                )

    anchor_rows: list[dict[str, Any]] = []
    reset_candidate_games: set[str] = set()
    consistent_games: set[str] = set()
    score_linked_games: set[str] = set()
    consistent_reset_anchor_count = 0
    score_linked_reset_anchor_count = 0
    for (game_id, identity_key, anchor), counter in sorted(by_anchor.items()):
        observation_count = int(counter.get("observation_count", 0))
        score_linked_count = int(counter.get("score_linked_count", 0))
        consistent_anchor = observation_count >= anchor_min
        if consistent_anchor:
            consistent_reset_anchor_count += 1
            consistent_games.add(game_id)
            by_game.setdefault(game_id, Counter())[
                "consistent_reset_anchor_count"
            ] += 1
        if score_linked_count > 0:
            score_linked_reset_anchor_count += 1
            score_linked_games.add(game_id)
            by_game.setdefault(game_id, Counter())[
                "score_linked_reset_anchor_count"
            ] += 1
        reset_candidate_games.add(game_id)
        action_histogram = {
            key.removeprefix("action:"): int(value)
            for key, value in sorted(counter.items())
            if str(key).startswith("action:")
        }
        anchor_rows.append(
            {
                "game_id": game_id,
                "identity_key": identity_key,
                "return_anchor": [anchor[0], anchor[1]],
                "observation_count": observation_count,
                "score_linked_count": score_linked_count,
                "score_delta_sum": int(counter.get("score_delta_sum", 0)),
                "consistent_reset_anchor": consistent_anchor,
                "action_histogram": action_histogram,
            }
        )

    candidate_count = len(anchor_rows)
    if score_linked_reset_anchor_count > 0:
        promotion_state = "tile_effect_runtime_trace_score_linked_reset_observed"
    elif consistent_reset_anchor_count > 0:
        promotion_state = "tile_effect_runtime_trace_consistent_reset_observed"
    elif candidate_count > 0:
        promotion_state = "tile_effect_runtime_trace_reset_candidates_observed"
    elif packet_rows:
        promotion_state = "tile_effect_runtime_trace_no_candidates"
    else:
        promotion_state = "tile_effect_runtime_trace_no_packets"

    return _json_safe(
        {
            "schema_version": TILE_EFFECT_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": len(packet_rows),
            "packet_with_moves_count": int(packet_with_moves_count),
            "reset_jump_packet_count": len(reset_packet_keys),
            "reset_jump_observation_count": sum(
                counter.get("reset_jump_observation_count", 0)
                for counter in by_game.values()
            ),
            "reset_anchor_candidate_count": candidate_count,
            "consistent_reset_anchor_count": int(consistent_reset_anchor_count),
            "score_linked_reset_anchor_count": int(score_linked_reset_anchor_count),
            "reset_candidate_game_count": len(reset_candidate_games),
            "consistent_reset_candidate_game_count": len(consistent_games),
            "score_linked_reset_candidate_game_count": len(score_linked_games),
            "normal_step_max": normal_step,
            "min_anchor_observations": anchor_min,
            "promotion_state": promotion_state,
            "runtime_trace_connection_observed": candidate_count > 0,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "displacement_histogram": dict(sorted(displacement_histogram.items())),
            "anchors": anchor_rows[:sample_limit],
            "samples": candidate_samples,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def build_tile_effect_runtime_trace_packet_from_action_effect_record(
    record: Any,
    *,
    game_id: str = "",
    step: int = 0,
    trace_context: Mapping[str, Any] | None = None,
    action_payload: Mapping[str, Any] | None = None,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Build a tile-effect diagnostic packet from an action-effect record.

    This uses the already-computed `segment_displacements` from the generic
    action-effect tracker. It does not inspect private environment state and
    does not update `TileEffectLearner`; it only exposes runtime evidence for
    later summarization/posterior phases.
    """

    moves: list[dict[str, Any]] = []
    context_hazard_source = _hazard_source_cell_from_trace_context(trace_context)
    payload_hazard_source = _hazard_source_cell_from_action_payload(action_payload)
    for raw in list(getattr(record, "segment_displacements", []) or [])[
        : max(0, int(sample_limit))
    ]:
        if not isinstance(raw, Mapping):
            continue
        prev = _centroid_cell(raw.get("pre_centroid"))
        cur = _centroid_cell(raw.get("post_centroid"))
        if prev is None or cur is None:
            prev = _bbox_centroid_cell(raw.get("pre_bbox"))
            cur = _bbox_centroid_cell(raw.get("post_bbox"))
        if prev is None or cur is None:
            continue
        dx = _safe_int(round(float(raw.get("dx", cur[0] - prev[0]) or 0)))
        dy = _safe_int(round(float(raw.get("dy", cur[1] - prev[1]) or 0)))
        if dx == 0 and dy == 0:
            dx = cur[0] - prev[0]
            dy = cur[1] - prev[1]
        color = _safe_int(raw.get("color"))
        pre_size = _safe_int(raw.get("pre_size"))
        post_size = _safe_int(raw.get("post_size"))
        area = post_size or pre_size
        moves.append(
            {
                "identity": {
                    "dominant_color": color,
                    "shape_signature": [
                        pre_size,
                        post_size,
                        _safe_int(raw.get("size_delta")),
                    ],
                    "area": area,
                },
                "prev_cell": [prev[0], prev[1]],
                "cur_cell": [cur[0], cur[1]],
                "displacement": [dx, dy],
                "area": area,
                "source": "action_effect_segment_displacement",
                "confidence": float(raw.get("confidence", 0.0) or 0.0),
            }
        )
        hazard_source_source = ""
        hazard_source = _hazard_source_cell_from_move(raw)
        if hazard_source is not None:
            hazard_source_source = "segment_displacement"
        if hazard_source is None and context_hazard_source is not None:
            hazard_source = context_hazard_source
            hazard_source_source = "trace_context"
        if hazard_source is None and payload_hazard_source is not None:
            hazard_source = payload_hazard_source
            hazard_source_source = "action_payload"
        if hazard_source is not None:
            moves[-1]["hazard_source_cell"] = [
                hazard_source[0],
                hazard_source[1],
            ]
            moves[-1]["hazard_source_cell_source"] = hazard_source_source

    score_delta = _safe_int(getattr(record, "score_delta", 0))
    hazard_source_move_count = sum(
        1 for move in moves if _vector_from_value(move.get("hazard_source_cell")) is not None
    )
    return _json_safe(
        {
            "schema_version": TILE_EFFECT_RUNTIME_TRACE_PACKET_SCHEMA_VERSION,
            "sensor_family": "tile_effect_self_repair",
            "source": "action_effect_segment_displacement",
            "game_id": str(game_id or "").strip().lower(),
            "state_hash": str(getattr(record, "pre_state_hash", "") or ""),
            "post_state_hash": str(getattr(record, "post_state_hash", "") or ""),
            "step": int(step),
            "action_name": str(getattr(record, "action_name", "") or ""),
            "score_delta": score_delta,
            "move_count": len(moves),
            "hazard_source_move_count": int(hazard_source_move_count),
            "moves": moves,
            "score_closure_candidate": bool(score_delta > 0 and moves),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_tile_effect_posterior(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Build a conservative posterior over tile-effect reset anchors.

    The posterior separates score-linked reset candidates from broad
    scoreless segment motion. It is still shadow-only: no planner state is
    updated, and no candidate is ranked or executed from this summary.
    """

    runtime = summarize_tile_effect_runtime_trace_packets(
        packets,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        sample_limit=100000,
    )
    anchor_rows = [
        dict(row) for row in list(runtime.get("anchors") or []) if isinstance(row, Mapping)
    ]
    score_min = max(1, int(min_score_linked_observations))
    anchor_min = max(1, int(min_anchor_observations))
    rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    confirmed_games: set[str] = set()
    ambiguous_games: set[str] = set()
    insufficient_games: set[str] = set()

    for row in anchor_rows:
        game_id = str(row.get("game_id", "") or "unknown")
        observation_count = _safe_int(row.get("observation_count"))
        score_linked_count = _safe_int(row.get("score_linked_count"))
        score_delta_sum = _safe_int(row.get("score_delta_sum"))
        if score_linked_count >= score_min and score_delta_sum > 0:
            posterior_state = "confirmed_score_linked"
            confirmed_games.add(game_id)
        elif observation_count >= anchor_min:
            posterior_state = "ambiguous_scoreless_motion"
            ambiguous_games.add(game_id)
        else:
            posterior_state = "insufficient_evidence"
            insufficient_games.add(game_id)
        state_counts[posterior_state] += 1
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["posterior_entry_count"] += 1
        game_counter[f"{posterior_state}_count"] += 1
        rows.append(
            {
                "game_id": game_id,
                "posterior_key": _posterior_key(row),
                "identity_key": str(row.get("identity_key", "") or ""),
                "return_anchor": list(row.get("return_anchor") or []),
                "posterior_state": posterior_state,
                "observation_count": observation_count,
                "score_linked_count": score_linked_count,
                "score_delta_sum": score_delta_sum,
                "action_histogram": dict(row.get("action_histogram") or {}),
                "admission_ready_for_guard": posterior_state
                == "confirmed_score_linked",
            }
        )

    confirmed_count = int(state_counts.get("confirmed_score_linked", 0))
    ambiguous_count = int(state_counts.get("ambiguous_scoreless_motion", 0))
    insufficient_count = int(state_counts.get("insufficient_evidence", 0))
    if confirmed_count > 0:
        promotion_state = "tile_effect_posterior_score_linked_confirmed"
    elif ambiguous_count > 0:
        promotion_state = "tile_effect_posterior_scoreless_motion_ambiguous"
    elif rows:
        promotion_state = "tile_effect_posterior_candidates_unconfirmed"
    elif int(runtime.get("packet_count", 0) or 0) > 0:
        promotion_state = "tile_effect_posterior_no_candidates"
    else:
        promotion_state = "tile_effect_posterior_no_packets"

    return _json_safe(
        {
            "schema_version": TILE_EFFECT_POSTERIOR_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(runtime.get("packet_count", 0) or 0),
            "reset_anchor_candidate_count": int(
                runtime.get("reset_anchor_candidate_count", 0) or 0
            ),
            "posterior_entry_count": len(rows),
            "confirmed_score_linked_count": confirmed_count,
            "ambiguous_scoreless_motion_count": ambiguous_count,
            "insufficient_evidence_count": insufficient_count,
            "confirmed_score_linked_game_count": len(confirmed_games),
            "ambiguous_scoreless_motion_game_count": len(ambiguous_games),
            "insufficient_evidence_game_count": len(insufficient_games),
            "normal_step_max": max(1, int(normal_step_max)),
            "min_anchor_observations": anchor_min,
            "min_score_linked_observations": score_min,
            "promotion_state": promotion_state,
            "posterior_observed": bool(rows),
            "runtime_trace_promotion_state": str(
                runtime.get("promotion_state", "") or ""
            ),
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


def summarize_tile_effect_guarded_planner_connection(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Shadow admission guard for tile-effect planner connection.

    Only score-linked, posterior-confirmed reset anchors are admitted for a
    future planner connection. Scoreless motion and insufficient evidence stay
    blocked. This is still diagnostic-only: it does not rank, execute, or
    generate any action.
    """

    posterior = summarize_tile_effect_posterior(
        packets,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        sample_limit=100000,
    )
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    admitted_count = 0
    blocked_count = 0
    blocked_scoreless_motion_count = 0
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
            connection_state = "tile_effect_planner_connection_shadow_admitted"
            admitted_count += 1
            admitted_games.add(game_id)
            game_counter["admitted_shadow_count"] += 1
        else:
            blocked_count += 1
            blocked_games.add(game_id)
            if state == "ambiguous_scoreless_motion":
                blocked_reasons.append("scoreless_reset_motion")
                blocked_scoreless_motion_count += 1
                game_counter["blocked_scoreless_motion_count"] += 1
            elif state == "insufficient_evidence":
                blocked_reasons.append("insufficient_tile_effect_evidence")
                blocked_insufficient_count += 1
                game_counter["blocked_insufficient_evidence_count"] += 1
            else:
                blocked_reasons.append("not_score_linked_confirmed")
                blocked_other_count += 1
                game_counter["blocked_other_count"] += 1
            connection_state = (
                "tile_effect_planner_connection_blocked_"
                + blocked_reasons[0]
            )
        row = {
            "game_id": game_id,
            "posterior_key": str(posterior_row.get("posterior_key", "") or ""),
            "identity_key": str(posterior_row.get("identity_key", "") or ""),
            "return_anchor": list(posterior_row.get("return_anchor") or []),
            "connection_state": connection_state,
            "admitted_shadow": admitted,
            "blocked_reasons": blocked_reasons,
            "posterior_state": state,
            "observation_count": _safe_int(posterior_row.get("observation_count")),
            "score_linked_count": _safe_int(posterior_row.get("score_linked_count")),
            "score_delta_sum": _safe_int(posterior_row.get("score_delta_sum")),
            "action_histogram": dict(posterior_row.get("action_histogram") or {}),
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if admitted_count > 0:
        promotion_state = "tile_effect_guarded_planner_connection_shadow_admitted"
    elif blocked_scoreless_motion_count > 0:
        promotion_state = (
            "tile_effect_guarded_planner_connection_blocked_scoreless_motion"
        )
    elif blocked_insufficient_count > 0:
        promotion_state = (
            "tile_effect_guarded_planner_connection_blocked_insufficient_evidence"
        )
    elif blocked_count > 0:
        promotion_state = "tile_effect_guarded_planner_connection_blocked"
    elif int(posterior.get("packet_count", 0) or 0) > 0:
        promotion_state = "tile_effect_guarded_planner_connection_no_candidates"
    else:
        promotion_state = "tile_effect_guarded_planner_connection_no_packets"

    return _json_safe(
        {
            "schema_version": TILE_EFFECT_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION,
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
            "blocked_scoreless_motion_count": int(blocked_scoreless_motion_count),
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


def summarize_tile_effect_guard_precision_review(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Review tile-effect guard admissions before any real promotion.

    This is a stricter shadow-only review over guard-admitted rows. It keeps
    the criteria generic: repeated score-linked support, optional purity
    against scoreless observations for the same posterior key, and stable
    action evidence. It intentionally does not inspect game ids, fixed states,
    or known solution coordinates.
    """

    guard = summarize_tile_effect_guarded_planner_connection(
        packets,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        sample_limit=100000,
    )
    repeat_min = max(1, int(min_repeat_score_linked_observations))
    dominance_min = max(0.0, min(1.0, float(min_action_dominance)))
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    ready_count = 0
    blocked_count = 0
    blocked_singleton_count = 0
    blocked_mixed_count = 0
    blocked_unstable_action_count = 0
    blocked_other_count = 0
    ready_games: set[str] = set()
    blocked_games: set[str] = set()
    by_game: dict[str, Counter[str]] = {}

    for guard_row in list(guard.get("rows") or []):
        if not isinstance(guard_row, Mapping) or not bool(
            guard_row.get("admitted_shadow")
        ):
            continue
        game_id = str(guard_row.get("game_id", "") or "unknown")
        observation_count = _safe_int(guard_row.get("observation_count"))
        score_linked_count = _safe_int(guard_row.get("score_linked_count"))
        action_histogram = dict(guard_row.get("action_histogram") or {})
        top_action_count = 0
        if action_histogram:
            top_action_count = max(_safe_int(value) for value in action_histogram.values())
        action_dominance = (
            float(top_action_count) / float(observation_count)
            if observation_count > 0
            else 0.0
        )
        blocked_reasons: list[str] = []
        if score_linked_count < repeat_min:
            blocked_reasons.append("score_linked_support_not_repeated")
            blocked_singleton_count += 1
        if require_score_linked_purity and score_linked_count < observation_count:
            blocked_reasons.append("mixed_scoreless_and_score_linked_evidence")
            blocked_mixed_count += 1
        if action_dominance < dominance_min:
            blocked_reasons.append("unstable_action_evidence")
            blocked_unstable_action_count += 1
        if not blocked_reasons and score_linked_count <= 0:
            blocked_reasons.append("missing_score_linked_evidence")
            blocked_other_count += 1

        ready = not blocked_reasons
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["reviewed_admission_count"] += 1
        if ready:
            ready_count += 1
            ready_games.add(game_id)
            game_counter["precision_ready_count"] += 1
            review_state = "tile_effect_precision_review_shadow_ready"
        else:
            blocked_count += 1
            blocked_games.add(game_id)
            game_counter["precision_blocked_count"] += 1
            for reason in blocked_reasons:
                game_counter[f"blocked_{reason}_count"] += 1
            review_state = "tile_effect_precision_review_blocked"

        row = {
            "game_id": game_id,
            "posterior_key": str(guard_row.get("posterior_key", "") or ""),
            "identity_key": str(guard_row.get("identity_key", "") or ""),
            "return_anchor": list(guard_row.get("return_anchor") or []),
            "review_state": review_state,
            "precision_ready_shadow": ready,
            "blocked_reasons": blocked_reasons,
            "observation_count": observation_count,
            "score_linked_count": score_linked_count,
            "score_delta_sum": _safe_int(guard_row.get("score_delta_sum")),
            "action_histogram": action_histogram,
            "action_dominance": round(action_dominance, 6),
            "min_repeat_score_linked_observations": repeat_min,
            "require_score_linked_purity": bool(require_score_linked_purity),
            "min_action_dominance": round(dominance_min, 6),
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if ready_count > 0 and blocked_count > 0:
        promotion_state = "tile_effect_guard_precision_review_ready_with_blocks"
    elif ready_count > 0:
        promotion_state = "tile_effect_guard_precision_review_ready"
    elif blocked_count > 0:
        promotion_state = "tile_effect_guard_precision_review_all_blocked"
    elif int(guard.get("planner_connection_admitted_count", 0) or 0) > 0:
        promotion_state = "tile_effect_guard_precision_review_no_review_rows"
    elif int(guard.get("packet_count", 0) or 0) > 0:
        promotion_state = "tile_effect_guard_precision_review_no_admissions"
    else:
        promotion_state = "tile_effect_guard_precision_review_no_packets"

    return _json_safe(
        {
            "schema_version": TILE_EFFECT_GUARD_PRECISION_REVIEW_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(guard.get("packet_count", 0) or 0),
            "posterior_entry_count": int(guard.get("posterior_entry_count", 0) or 0),
            "guard_admitted_count": int(
                guard.get("planner_connection_admitted_count", 0) or 0
            ),
            "guard_blocked_count": int(
                guard.get("planner_connection_blocked_count", 0) or 0
            ),
            "precision_reviewed_admission_count": len(rows),
            "precision_ready_count": int(ready_count),
            "precision_blocked_count": int(blocked_count),
            "precision_ready_game_count": len(ready_games),
            "precision_blocked_game_count": len(blocked_games),
            "blocked_singleton_score_link_count": int(blocked_singleton_count),
            "blocked_mixed_scoreless_score_linked_count": int(blocked_mixed_count),
            "blocked_unstable_action_count": int(blocked_unstable_action_count),
            "blocked_other_count": int(blocked_other_count),
            "min_repeat_score_linked_observations": repeat_min,
            "require_score_linked_purity": bool(require_score_linked_purity),
            "min_action_dominance": round(dominance_min, 6),
            "guard_promotion_state": str(guard.get("promotion_state", "") or ""),
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


def summarize_tile_effect_guard_counterfactual_control(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    min_return_anchor_prior_non_reset_observations: int = 1,
    allow_return_anchor_prior_occupancy_trace: bool = False,
    require_hazard_source_evidence: bool = True,
    require_hazard_source_distinct_from_return_anchor: bool = True,
    min_scoreless_counterfactual_same_identity: int = 0,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Counterfactual control for tile-effect planner promotion.

    Precision-ready rows are still not enough for a real tile-effect planner:
    a planner must know which source/hazard tile to avoid, not only the
    return anchor a moving component jumped to. This shadow-only control
    therefore requires hazard-source evidence and an independently observed
    non-reset prior for the return anchor before any row can be considered
    promotion-ready.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    normal_step = max(1, int(normal_step_max))
    precision = summarize_tile_effect_guard_precision_review(
        packet_rows,
        normal_step_max=normal_step,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        sample_limit=100000,
    )
    source_required = bool(require_hazard_source_evidence)
    source_distinct_required = bool(require_hazard_source_distinct_from_return_anchor)
    prior_min = max(0, int(min_return_anchor_prior_non_reset_observations))
    scoreless_min = max(0, int(min_scoreless_counterfactual_same_identity))
    stats_by_key, scoreless_by_identity = _tile_effect_counterfactual_stats(
        packet_rows,
        normal_step_max=normal_step,
    )

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    ready_count = 0
    blocked_count = 0
    blocked_missing_source_count = 0
    blocked_source_not_distinct_count = 0
    blocked_missing_prior_count = 0
    blocked_missing_scoreless_counterfactual_count = 0
    blocked_other_count = 0
    ready_games: set[str] = set()
    blocked_games: set[str] = set()
    by_game: dict[str, Counter[str]] = {}

    for precision_row in list(precision.get("rows") or []):
        if not isinstance(precision_row, Mapping) or not bool(
            precision_row.get("precision_ready_shadow")
        ):
            continue
        game_id = str(precision_row.get("game_id", "") or "unknown")
        identity_key = str(precision_row.get("identity_key", "") or "")
        posterior_key = str(precision_row.get("posterior_key", "") or "")
        counter = stats_by_key.get(posterior_key, Counter())
        identity_scoreless_count = int(
            scoreless_by_identity.get((game_id, identity_key), 0)
        )
        hazard_source_observed_count = _safe_int(
            counter.get("hazard_source_observed_count")
        )
        hazard_source_distinct_count = _safe_int(
            counter.get("hazard_source_distinct_from_return_anchor_count")
        )
        prior_seen_count = _safe_int(
            counter.get("return_anchor_prior_non_reset_seen_count")
        )
        prior_occupied_count = _safe_int(
            counter.get("return_anchor_prior_occupancy_seen_count")
        )
        effective_prior_count = prior_seen_count
        if bool(allow_return_anchor_prior_occupancy_trace):
            effective_prior_count = max(effective_prior_count, prior_occupied_count)

        blocked_reasons: list[str] = []
        if source_required and hazard_source_observed_count <= 0:
            blocked_reasons.append("missing_hazard_source_evidence")
            blocked_missing_source_count += 1
        if (
            source_distinct_required
            and hazard_source_observed_count > 0
            and hazard_source_distinct_count <= 0
        ):
            blocked_reasons.append("hazard_source_not_distinct_from_return_anchor")
            blocked_source_not_distinct_count += 1
        if effective_prior_count < prior_min:
            blocked_reasons.append("return_anchor_not_seen_as_non_reset_prior")
            blocked_missing_prior_count += 1
        if identity_scoreless_count < scoreless_min:
            blocked_reasons.append("missing_scoreless_counterfactual_same_identity")
            blocked_missing_scoreless_counterfactual_count += 1
        if not blocked_reasons and source_required and hazard_source_observed_count <= 0:
            blocked_reasons.append("missing_counterfactual_control_evidence")
            blocked_other_count += 1

        ready = not blocked_reasons
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["counterfactual_reviewed_count"] += 1
        if ready:
            ready_count += 1
            ready_games.add(game_id)
            game_counter["counterfactual_ready_count"] += 1
            control_state = "tile_effect_counterfactual_control_shadow_ready"
        else:
            blocked_count += 1
            blocked_games.add(game_id)
            game_counter["counterfactual_blocked_count"] += 1
            for reason in blocked_reasons:
                game_counter[f"blocked_{reason}_count"] += 1
            control_state = "tile_effect_counterfactual_control_blocked"

        row = {
            "game_id": game_id,
            "posterior_key": posterior_key,
            "identity_key": identity_key,
            "return_anchor": list(precision_row.get("return_anchor") or []),
            "control_state": control_state,
            "counterfactual_ready_shadow": ready,
            "blocked_reasons": blocked_reasons,
            "precision_review_state": str(
                precision_row.get("review_state", "") or ""
            ),
            "observation_count": _safe_int(precision_row.get("observation_count")),
            "score_linked_count": _safe_int(precision_row.get("score_linked_count")),
            "score_delta_sum": _safe_int(precision_row.get("score_delta_sum")),
            "hazard_source_observed_count": hazard_source_observed_count,
            "hazard_source_distinct_from_return_anchor_count": (
                hazard_source_distinct_count
            ),
            "return_anchor_prior_non_reset_seen_count": prior_seen_count,
            "return_anchor_prior_occupancy_seen_count": prior_occupied_count,
            "return_anchor_effective_prior_seen_count": effective_prior_count,
            "scoreless_counterfactual_same_identity_count": identity_scoreless_count,
            "score_linked_reset_packet_count": _safe_int(
                counter.get("score_linked_reset_packet_count")
            ),
            "score_linked_reset_packet_move_count_max": _safe_int(
                counter.get("score_linked_reset_packet_move_count_max")
            ),
            "score_linked_reset_packet_move_count_sum": _safe_int(
                counter.get("score_linked_reset_packet_move_count_sum")
            ),
            "hazard_source_histogram": dict(
                sorted(
                    {
                        str(key).removeprefix("source:"): int(value)
                        for key, value in counter.items()
                        if str(key).startswith("source:")
                    }.items()
                )
            ),
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if ready_count > 0 and blocked_count > 0:
        promotion_state = "tile_effect_guard_counterfactual_control_ready_with_blocks"
    elif ready_count > 0:
        promotion_state = "tile_effect_guard_counterfactual_control_ready"
    elif blocked_count > 0:
        promotion_state = "tile_effect_guard_counterfactual_control_all_blocked"
    elif int(precision.get("precision_ready_count", 0) or 0) > 0:
        promotion_state = "tile_effect_guard_counterfactual_control_no_review_rows"
    elif int(precision.get("packet_count", 0) or 0) > 0:
        promotion_state = "tile_effect_guard_counterfactual_control_no_precision_ready"
    else:
        promotion_state = "tile_effect_guard_counterfactual_control_no_packets"

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_GUARD_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(precision.get("packet_count", 0) or 0),
            "posterior_entry_count": int(precision.get("posterior_entry_count", 0) or 0),
            "guard_admitted_count": int(precision.get("guard_admitted_count", 0) or 0),
            "precision_ready_input_count": int(
                precision.get("precision_ready_count", 0) or 0
            ),
            "counterfactual_reviewed_count": len(rows),
            "counterfactual_ready_count": int(ready_count),
            "counterfactual_blocked_count": int(blocked_count),
            "counterfactual_ready_game_count": len(ready_games),
            "counterfactual_blocked_game_count": len(blocked_games),
            "blocked_missing_hazard_source_count": int(blocked_missing_source_count),
            "blocked_hazard_source_not_distinct_from_return_anchor_count": int(
                blocked_source_not_distinct_count
            ),
            "blocked_missing_return_anchor_prior_count": int(
                blocked_missing_prior_count
            ),
            "blocked_missing_scoreless_counterfactual_count": int(
                blocked_missing_scoreless_counterfactual_count
            ),
            "blocked_other_count": int(blocked_other_count),
            "min_return_anchor_prior_non_reset_observations": prior_min,
            "allow_return_anchor_prior_occupancy_trace": bool(
                allow_return_anchor_prior_occupancy_trace
            ),
            "require_hazard_source_evidence": source_required,
            "require_hazard_source_distinct_from_return_anchor": (
                source_distinct_required
            ),
            "min_scoreless_counterfactual_same_identity": scoreless_min,
            "precision_promotion_state": str(
                precision.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "hazard_source_trace_required_before_real_promotion": True,
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


def summarize_tile_effect_family_scope_negative_control(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    min_return_anchor_prior_non_reset_observations: int = 1,
    allow_return_anchor_prior_occupancy_trace: bool = False,
    require_hazard_source_evidence: bool = True,
    require_hazard_source_distinct_from_return_anchor: bool = True,
    min_scoreless_reset_same_identity_observations: int = 0,
    max_score_linked_reset_packet_move_count: int = 4,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Family-scope negative control for tile-effect planner promotion.

    Counterfactual-ready rows can still be score-synchronous terminal effects
    from other mechanism families. A localized tile-effect/hazard event should
    be supported by small reset packets. Broad packets where many components
    move at once are treated as terminal/global animation evidence and blocked.
    Scoreless same-identity reset controls remain available as an optional
    stricter check, but are not required by default because canonical m0r0
    focused traces only expose the reset at score closure.

    This remains shadow-only. It does not use game ids, coordinates, known
    states, ranking, action generation, or runtime execution.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    scoreless_min = max(0, int(min_scoreless_reset_same_identity_observations))
    packet_move_max = max(1, int(max_score_linked_reset_packet_move_count))
    counterfactual = summarize_tile_effect_guard_counterfactual_control(
        packet_rows,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_counterfactual_same_identity=0,
        sample_limit=100000,
    )

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    ready_count = 0
    blocked_count = 0
    blocked_missing_scoreless_control_count = 0
    blocked_broad_packet_count = 0
    blocked_other_count = 0
    ready_games: set[str] = set()
    blocked_games: set[str] = set()
    by_game: dict[str, Counter[str]] = {}

    for control_row in list(counterfactual.get("rows") or []):
        if not isinstance(control_row, Mapping) or not bool(
            control_row.get("counterfactual_ready_shadow")
        ):
            continue
        game_id = str(control_row.get("game_id", "") or "unknown")
        same_identity_scoreless = _safe_int(
            control_row.get("scoreless_counterfactual_same_identity_count")
        )
        score_linked_packet_move_count_max = _safe_int(
            control_row.get("score_linked_reset_packet_move_count_max")
        )
        blocked_reasons: list[str] = []
        if same_identity_scoreless < scoreless_min:
            blocked_reasons.append(
                "missing_scoreless_reset_same_identity_control"
            )
            blocked_missing_scoreless_control_count += 1
        if score_linked_packet_move_count_max > packet_move_max:
            blocked_reasons.append("score_linked_reset_packet_too_broad")
            blocked_broad_packet_count += 1
        if not blocked_reasons and scoreless_min < 0:
            blocked_reasons.append("invalid_family_scope_control_threshold")
            blocked_other_count += 1

        ready = not blocked_reasons
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["family_scope_reviewed_count"] += 1
        if ready:
            ready_count += 1
            ready_games.add(game_id)
            game_counter["family_scope_ready_count"] += 1
            control_state = "tile_effect_family_scope_negative_control_shadow_ready"
        else:
            blocked_count += 1
            blocked_games.add(game_id)
            game_counter["family_scope_blocked_count"] += 1
            for reason in blocked_reasons:
                game_counter[f"blocked_{reason}_count"] += 1
            control_state = "tile_effect_family_scope_negative_control_blocked"

        row = {
            "game_id": game_id,
            "posterior_key": str(control_row.get("posterior_key", "") or ""),
            "identity_key": str(control_row.get("identity_key", "") or ""),
            "return_anchor": list(control_row.get("return_anchor") or []),
            "control_state": control_state,
            "family_scope_ready_shadow": ready,
            "blocked_reasons": blocked_reasons,
            "counterfactual_control_state": str(
                control_row.get("control_state", "") or ""
            ),
            "observation_count": _safe_int(control_row.get("observation_count")),
            "score_linked_count": _safe_int(control_row.get("score_linked_count")),
            "score_delta_sum": _safe_int(control_row.get("score_delta_sum")),
            "hazard_source_observed_count": _safe_int(
                control_row.get("hazard_source_observed_count")
            ),
            "hazard_source_distinct_from_return_anchor_count": _safe_int(
                control_row.get("hazard_source_distinct_from_return_anchor_count")
            ),
            "return_anchor_effective_prior_seen_count": _safe_int(
                control_row.get("return_anchor_effective_prior_seen_count")
            ),
            "scoreless_reset_same_identity_control_count": same_identity_scoreless,
            "min_scoreless_reset_same_identity_observations": scoreless_min,
            "score_linked_reset_packet_move_count_max": (
                score_linked_packet_move_count_max
            ),
            "max_score_linked_reset_packet_move_count": packet_move_max,
            "hazard_source_histogram": dict(
                control_row.get("hazard_source_histogram") or {}
            ),
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if ready_count > 0 and blocked_count > 0:
        promotion_state = "tile_effect_family_scope_negative_control_ready_with_blocks"
    elif ready_count > 0:
        promotion_state = "tile_effect_family_scope_negative_control_ready"
    elif blocked_count > 0:
        promotion_state = "tile_effect_family_scope_negative_control_all_blocked"
    elif int(counterfactual.get("counterfactual_ready_count", 0) or 0) > 0:
        promotion_state = "tile_effect_family_scope_negative_control_no_review_rows"
    elif int(counterfactual.get("packet_count", 0) or 0) > 0:
        promotion_state = (
            "tile_effect_family_scope_negative_control_no_counterfactual_ready"
        )
    else:
        promotion_state = "tile_effect_family_scope_negative_control_no_packets"

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_FAMILY_SCOPE_NEGATIVE_CONTROL_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(counterfactual.get("packet_count", 0) or 0),
            "posterior_entry_count": int(
                counterfactual.get("posterior_entry_count", 0) or 0
            ),
            "counterfactual_ready_input_count": int(
                counterfactual.get("counterfactual_ready_count", 0) or 0
            ),
            "family_scope_reviewed_count": len(rows),
            "family_scope_ready_count": int(ready_count),
            "family_scope_blocked_count": int(blocked_count),
            "family_scope_ready_game_count": len(ready_games),
            "family_scope_blocked_game_count": len(blocked_games),
            "blocked_missing_scoreless_reset_same_identity_control_count": int(
                blocked_missing_scoreless_control_count
            ),
            "blocked_score_linked_reset_packet_too_broad_count": int(
                blocked_broad_packet_count
            ),
            "blocked_other_count": int(blocked_other_count),
            "min_scoreless_reset_same_identity_observations": scoreless_min,
            "max_score_linked_reset_packet_move_count": packet_move_max,
            "allow_return_anchor_prior_occupancy_trace": bool(
                allow_return_anchor_prior_occupancy_trace
            ),
            "counterfactual_promotion_state": str(
                counterfactual.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "family_scope_negative_control_required_before_real_promotion": True,
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


def summarize_tile_effect_hazard_source_repair_bridge(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    min_return_anchor_prior_non_reset_observations: int = 1,
    allow_return_anchor_prior_occupancy_trace: bool = False,
    require_hazard_source_evidence: bool = True,
    require_hazard_source_distinct_from_return_anchor: bool = True,
    min_scoreless_reset_same_identity_observations: int = 0,
    max_score_linked_reset_packet_move_count: int = 4,
    min_hazard_source_support: int = 1,
    min_hazard_source_dominance: float = 1.0,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Shadow bridge from family-scoped tile-effect evidence to model repair.

    This does not mutate a planner. It only asks whether a family-ready row has
    a stable source cell that could become a future hazard/avoidance repair.
    The source must come from ECA trace evidence and must be unique enough to
    pass the dominance threshold.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    family = summarize_tile_effect_family_scope_negative_control(
        packet_rows,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_reset_same_identity_observations=(
            min_scoreless_reset_same_identity_observations
        ),
        max_score_linked_reset_packet_move_count=(
            max_score_linked_reset_packet_move_count
        ),
        sample_limit=100000,
    )
    support_min = max(1, int(min_hazard_source_support))
    dominance_min = max(0.0, min(1.0, float(min_hazard_source_dominance)))

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    ready_count = 0
    blocked_count = 0
    blocked_missing_source_count = 0
    blocked_unstable_source_count = 0
    ready_games: set[str] = set()
    blocked_games: set[str] = set()
    by_game: dict[str, Counter[str]] = {}

    for family_row in list(family.get("rows") or []):
        if not isinstance(family_row, Mapping) or not bool(
            family_row.get("family_scope_ready_shadow")
        ):
            continue
        game_id = str(family_row.get("game_id", "") or "unknown")
        source_histogram = {
            str(key): _safe_int(value)
            for key, value in dict(
                family_row.get("hazard_source_histogram") or {}
            ).items()
            if _safe_int(value) > 0
        }
        total_source_count = sum(source_histogram.values())
        top_source = ""
        top_source_count = 0
        if source_histogram:
            top_source, top_source_count = max(
                sorted(source_histogram.items()),
                key=lambda item: (int(item[1]), str(item[0])),
            )
        source_dominance = (
            float(top_source_count) / float(total_source_count)
            if total_source_count > 0
            else 0.0
        )
        hazard_source_cell = _cell_from_source_key(top_source)

        blocked_reasons: list[str] = []
        if hazard_source_cell is None or top_source_count <= 0:
            blocked_reasons.append("missing_stable_hazard_source")
            blocked_missing_source_count += 1
        if top_source_count < support_min or source_dominance < dominance_min:
            blocked_reasons.append("unstable_hazard_source_evidence")
            blocked_unstable_source_count += 1

        ready = not blocked_reasons
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["repair_bridge_reviewed_count"] += 1
        if ready:
            ready_count += 1
            ready_games.add(game_id)
            game_counter["repair_bridge_ready_count"] += 1
            bridge_state = "tile_effect_hazard_source_repair_bridge_shadow_ready"
        else:
            blocked_count += 1
            blocked_games.add(game_id)
            game_counter["repair_bridge_blocked_count"] += 1
            for reason in blocked_reasons:
                game_counter[f"blocked_{reason}_count"] += 1
            bridge_state = "tile_effect_hazard_source_repair_bridge_blocked"

        row = {
            "game_id": game_id,
            "posterior_key": str(family_row.get("posterior_key", "") or ""),
            "identity_key": str(family_row.get("identity_key", "") or ""),
            "return_anchor": list(family_row.get("return_anchor") or []),
            "bridge_state": bridge_state,
            "repair_bridge_ready_shadow": ready,
            "blocked_reasons": blocked_reasons,
            "family_scope_control_state": str(
                family_row.get("control_state", "") or ""
            ),
            "hazard_source_cell": (
                [hazard_source_cell[0], hazard_source_cell[1]]
                if hazard_source_cell is not None
                else []
            ),
            "hazard_source_histogram": dict(sorted(source_histogram.items())),
            "hazard_source_top_count": int(top_source_count),
            "hazard_source_total_count": int(total_source_count),
            "hazard_source_dominance": round(source_dominance, 6),
            "min_hazard_source_support": support_min,
            "min_hazard_source_dominance": round(dominance_min, 6),
            "scoreless_reset_same_identity_control_count": _safe_int(
                family_row.get("scoreless_reset_same_identity_control_count")
            ),
            "score_linked_reset_packet_move_count_max": _safe_int(
                family_row.get("score_linked_reset_packet_move_count_max")
            ),
            "max_score_linked_reset_packet_move_count": _safe_int(
                family_row.get("max_score_linked_reset_packet_move_count")
            ),
            "shadow_repair_candidate_kind": (
                "localized_hazard_source_avoidance_cell" if ready else ""
            ),
            "model_repair_write_enabled": False,
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if ready_count > 0 and blocked_count > 0:
        promotion_state = "tile_effect_hazard_source_repair_bridge_ready_with_blocks"
    elif ready_count > 0:
        promotion_state = "tile_effect_hazard_source_repair_bridge_ready"
    elif blocked_count > 0:
        promotion_state = "tile_effect_hazard_source_repair_bridge_all_blocked"
    elif int(family.get("family_scope_ready_count", 0) or 0) > 0:
        promotion_state = "tile_effect_hazard_source_repair_bridge_no_review_rows"
    elif int(family.get("packet_count", 0) or 0) > 0:
        promotion_state = "tile_effect_hazard_source_repair_bridge_no_family_ready"
    else:
        promotion_state = "tile_effect_hazard_source_repair_bridge_no_packets"

    return _json_safe(
        {
            "schema_version": TILE_EFFECT_HAZARD_SOURCE_REPAIR_BRIDGE_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(family.get("packet_count", 0) or 0),
            "family_scope_ready_input_count": int(
                family.get("family_scope_ready_count", 0) or 0
            ),
            "repair_bridge_reviewed_count": len(rows),
            "repair_bridge_ready_count": int(ready_count),
            "repair_bridge_blocked_count": int(blocked_count),
            "repair_bridge_ready_game_count": len(ready_games),
            "repair_bridge_blocked_game_count": len(blocked_games),
            "blocked_missing_stable_hazard_source_count": int(
                blocked_missing_source_count
            ),
            "blocked_unstable_hazard_source_evidence_count": int(
                blocked_unstable_source_count
            ),
            "min_hazard_source_support": support_min,
            "min_hazard_source_dominance": round(dominance_min, 6),
            "family_scope_promotion_state": str(
                family.get("promotion_state", "") or ""
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
            "model_repair_write_enabled": False,
        }
    )


def summarize_tile_effect_model_repair_proposal(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    min_return_anchor_prior_non_reset_observations: int = 1,
    allow_return_anchor_prior_occupancy_trace: bool = False,
    require_hazard_source_evidence: bool = True,
    require_hazard_source_distinct_from_return_anchor: bool = True,
    min_scoreless_reset_same_identity_observations: int = 0,
    max_score_linked_reset_packet_move_count: int = 4,
    min_hazard_source_support: int = 1,
    min_hazard_source_dominance: float = 1.0,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Shadow proposal review for a future tile-effect model repair.

    The proposal is deliberately not a model write. It converts only
    bridge-ready rows into a stable repair-candidate record with the minimal
    local relation: avoid this hazard source before relying on this return
    anchor for this identity key.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    bridge = summarize_tile_effect_hazard_source_repair_bridge(
        packet_rows,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_reset_same_identity_observations=(
            min_scoreless_reset_same_identity_observations
        ),
        max_score_linked_reset_packet_move_count=(
            max_score_linked_reset_packet_move_count
        ),
        min_hazard_source_support=min_hazard_source_support,
        min_hazard_source_dominance=min_hazard_source_dominance,
        sample_limit=100000,
    )

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    ready_count = 0
    blocked_count = 0
    blocked_missing_anchor_count = 0
    blocked_missing_source_count = 0
    blocked_source_equals_anchor_count = 0
    ready_games: set[str] = set()
    blocked_games: set[str] = set()
    by_game: dict[str, Counter[str]] = {}

    for bridge_row in list(bridge.get("rows") or []):
        if not isinstance(bridge_row, Mapping) or not bool(
            bridge_row.get("repair_bridge_ready_shadow")
        ):
            continue
        game_id = str(bridge_row.get("game_id", "") or "unknown")
        return_anchor = _vector_from_value(bridge_row.get("return_anchor"))
        hazard_source = _vector_from_value(bridge_row.get("hazard_source_cell"))
        blocked_reasons: list[str] = []
        if return_anchor is None:
            blocked_reasons.append("missing_return_anchor")
            blocked_missing_anchor_count += 1
        if hazard_source is None:
            blocked_reasons.append("missing_hazard_source_cell")
            blocked_missing_source_count += 1
        if (
            return_anchor is not None
            and hazard_source is not None
            and return_anchor == hazard_source
        ):
            blocked_reasons.append("hazard_source_equals_return_anchor")
            blocked_source_equals_anchor_count += 1

        ready = not blocked_reasons
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["model_repair_proposal_reviewed_count"] += 1
        if ready:
            ready_count += 1
            ready_games.add(game_id)
            game_counter["model_repair_proposal_ready_count"] += 1
            proposal_state = "tile_effect_model_repair_proposal_shadow_ready"
        else:
            blocked_count += 1
            blocked_games.add(game_id)
            game_counter["model_repair_proposal_blocked_count"] += 1
            for reason in blocked_reasons:
                game_counter[f"blocked_{reason}_count"] += 1
            proposal_state = "tile_effect_model_repair_proposal_blocked"

        row = {
            "game_id": game_id,
            "posterior_key": str(bridge_row.get("posterior_key", "") or ""),
            "identity_key": str(bridge_row.get("identity_key", "") or ""),
            "return_anchor": (
                [return_anchor[0], return_anchor[1]]
                if return_anchor is not None
                else []
            ),
            "hazard_source_cell": (
                [hazard_source[0], hazard_source[1]]
                if hazard_source is not None
                else []
            ),
            "proposal_state": proposal_state,
            "repair_proposal_ready_shadow": ready,
            "blocked_reasons": blocked_reasons,
            "bridge_state": str(bridge_row.get("bridge_state", "") or ""),
            "hazard_source_dominance": _safe_float(
                bridge_row.get("hazard_source_dominance")
            ),
            "hazard_source_top_count": _safe_int(
                bridge_row.get("hazard_source_top_count")
            ),
            "hazard_source_total_count": _safe_int(
                bridge_row.get("hazard_source_total_count")
            ),
            "scoreless_reset_same_identity_control_count": _safe_int(
                bridge_row.get("scoreless_reset_same_identity_control_count")
            ),
            "score_linked_reset_packet_move_count_max": _safe_int(
                bridge_row.get("score_linked_reset_packet_move_count_max")
            ),
            "max_score_linked_reset_packet_move_count": _safe_int(
                bridge_row.get("max_score_linked_reset_packet_move_count")
            ),
            "shadow_repair_candidate_kind": str(
                bridge_row.get("shadow_repair_candidate_kind", "") or ""
            ),
            "model_repair_proposal_kind": (
                "avoid_hazard_source_before_return_anchor" if ready else ""
            ),
            "proposal_scope": (
                "localized_return_anchor_identity" if ready else ""
            ),
            "model_repair_write_enabled": False,
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if ready_count > 0 and blocked_count > 0:
        promotion_state = "tile_effect_model_repair_proposal_ready_with_blocks"
    elif ready_count > 0:
        promotion_state = "tile_effect_model_repair_proposal_ready"
    elif blocked_count > 0:
        promotion_state = "tile_effect_model_repair_proposal_all_blocked"
    elif int(bridge.get("repair_bridge_ready_count", 0) or 0) > 0:
        promotion_state = "tile_effect_model_repair_proposal_no_review_rows"
    elif int(bridge.get("packet_count", 0) or 0) > 0:
        promotion_state = "tile_effect_model_repair_proposal_no_bridge_ready"
    else:
        promotion_state = "tile_effect_model_repair_proposal_no_packets"

    return _json_safe(
        {
            "schema_version": TILE_EFFECT_MODEL_REPAIR_PROPOSAL_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(bridge.get("packet_count", 0) or 0),
            "repair_bridge_ready_input_count": int(
                bridge.get("repair_bridge_ready_count", 0) or 0
            ),
            "model_repair_proposal_reviewed_count": len(rows),
            "model_repair_proposal_ready_count": int(ready_count),
            "model_repair_proposal_blocked_count": int(blocked_count),
            "model_repair_proposal_ready_game_count": len(ready_games),
            "model_repair_proposal_blocked_game_count": len(blocked_games),
            "blocked_missing_return_anchor_count": int(
                blocked_missing_anchor_count
            ),
            "blocked_missing_hazard_source_cell_count": int(
                blocked_missing_source_count
            ),
            "blocked_hazard_source_equals_return_anchor_count": int(
                blocked_source_equals_anchor_count
            ),
            "bridge_promotion_state": str(
                bridge.get("promotion_state", "") or ""
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
            "model_repair_write_enabled": False,
        }
    )


def summarize_tile_effect_model_repair_write_guard(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    min_return_anchor_prior_non_reset_observations: int = 1,
    allow_return_anchor_prior_occupancy_trace: bool = False,
    require_hazard_source_evidence: bool = True,
    require_hazard_source_distinct_from_return_anchor: bool = True,
    min_scoreless_reset_same_identity_observations: int = 0,
    max_score_linked_reset_packet_move_count: int = 4,
    min_hazard_source_support: int = 1,
    min_hazard_source_dominance: float = 1.0,
    min_write_guard_hazard_source_support: int = 2,
    min_write_guard_hazard_source_dominance: float = 1.0,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Shadow guard for a future tile-effect model repair write.

    This is the final review before any future write path. It never mutates
    the model in this phase; it only records whether a proposal would satisfy
    stable localized evidence requirements.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    proposal = summarize_tile_effect_model_repair_proposal(
        packet_rows,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_reset_same_identity_observations=(
            min_scoreless_reset_same_identity_observations
        ),
        max_score_linked_reset_packet_move_count=(
            max_score_linked_reset_packet_move_count
        ),
        min_hazard_source_support=min_hazard_source_support,
        min_hazard_source_dominance=min_hazard_source_dominance,
        sample_limit=100000,
    )
    support_min = max(1, int(min_write_guard_hazard_source_support))
    dominance_min = max(
        0.0, min(1.0, float(min_write_guard_hazard_source_dominance))
    )

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    ready_count = 0
    blocked_count = 0
    blocked_unexpected_kind_count = 0
    blocked_unexpected_scope_count = 0
    blocked_weak_source_count = 0
    ready_games: set[str] = set()
    blocked_games: set[str] = set()
    by_game: dict[str, Counter[str]] = {}

    for proposal_row in list(proposal.get("rows") or []):
        if not isinstance(proposal_row, Mapping) or not bool(
            proposal_row.get("repair_proposal_ready_shadow")
        ):
            continue
        game_id = str(proposal_row.get("game_id", "") or "unknown")
        proposal_kind = str(proposal_row.get("model_repair_proposal_kind", "") or "")
        proposal_scope = str(proposal_row.get("proposal_scope", "") or "")
        top_count = _safe_int(proposal_row.get("hazard_source_top_count"))
        dominance = _safe_float(proposal_row.get("hazard_source_dominance"))
        blocked_reasons: list[str] = []
        if proposal_kind != "avoid_hazard_source_before_return_anchor":
            blocked_reasons.append("unexpected_model_repair_proposal_kind")
            blocked_unexpected_kind_count += 1
        if proposal_scope != "localized_return_anchor_identity":
            blocked_reasons.append("unexpected_model_repair_proposal_scope")
            blocked_unexpected_scope_count += 1
        if top_count < support_min or dominance < dominance_min:
            blocked_reasons.append("weak_hazard_source_evidence_for_write")
            blocked_weak_source_count += 1

        ready = not blocked_reasons
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["model_repair_write_guard_reviewed_count"] += 1
        if ready:
            ready_count += 1
            ready_games.add(game_id)
            game_counter["model_repair_write_guard_ready_count"] += 1
            guard_state = "tile_effect_model_repair_write_guard_shadow_ready"
        else:
            blocked_count += 1
            blocked_games.add(game_id)
            game_counter["model_repair_write_guard_blocked_count"] += 1
            for reason in blocked_reasons:
                game_counter[f"blocked_{reason}_count"] += 1
            guard_state = "tile_effect_model_repair_write_guard_blocked"

        row = {
            "game_id": game_id,
            "posterior_key": str(proposal_row.get("posterior_key", "") or ""),
            "identity_key": str(proposal_row.get("identity_key", "") or ""),
            "return_anchor": list(proposal_row.get("return_anchor") or []),
            "hazard_source_cell": list(
                proposal_row.get("hazard_source_cell") or []
            ),
            "write_guard_state": guard_state,
            "model_repair_write_permitted_shadow": ready,
            "blocked_reasons": blocked_reasons,
            "model_repair_proposal_kind": proposal_kind,
            "proposal_scope": proposal_scope,
            "hazard_source_dominance": round(dominance, 6),
            "hazard_source_top_count": top_count,
            "hazard_source_total_count": _safe_int(
                proposal_row.get("hazard_source_total_count")
            ),
            "scoreless_reset_same_identity_control_count": _safe_int(
                proposal_row.get("scoreless_reset_same_identity_control_count")
            ),
            "score_linked_reset_packet_move_count_max": _safe_int(
                proposal_row.get("score_linked_reset_packet_move_count_max")
            ),
            "max_score_linked_reset_packet_move_count": _safe_int(
                proposal_row.get("max_score_linked_reset_packet_move_count")
            ),
            "min_write_guard_hazard_source_support": support_min,
            "min_write_guard_hazard_source_dominance": round(dominance_min, 6),
            "model_repair_write_enabled": False,
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if ready_count > 0 and blocked_count > 0:
        promotion_state = "tile_effect_model_repair_write_guard_ready_with_blocks"
    elif ready_count > 0:
        promotion_state = "tile_effect_model_repair_write_guard_ready"
    elif blocked_count > 0:
        promotion_state = "tile_effect_model_repair_write_guard_all_blocked"
    elif int(proposal.get("model_repair_proposal_ready_count", 0) or 0) > 0:
        promotion_state = "tile_effect_model_repair_write_guard_no_review_rows"
    elif int(proposal.get("packet_count", 0) or 0) > 0:
        promotion_state = "tile_effect_model_repair_write_guard_no_proposals_ready"
    else:
        promotion_state = "tile_effect_model_repair_write_guard_no_packets"

    return _json_safe(
        {
            "schema_version": TILE_EFFECT_MODEL_REPAIR_WRITE_GUARD_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(proposal.get("packet_count", 0) or 0),
            "model_repair_proposal_ready_input_count": int(
                proposal.get("model_repair_proposal_ready_count", 0) or 0
            ),
            "model_repair_write_guard_reviewed_count": len(rows),
            "model_repair_write_guard_ready_count": int(ready_count),
            "model_repair_write_guard_blocked_count": int(blocked_count),
            "model_repair_write_guard_ready_game_count": len(ready_games),
            "model_repair_write_guard_blocked_game_count": len(blocked_games),
            "blocked_unexpected_model_repair_proposal_kind_count": int(
                blocked_unexpected_kind_count
            ),
            "blocked_unexpected_model_repair_proposal_scope_count": int(
                blocked_unexpected_scope_count
            ),
            "blocked_weak_hazard_source_evidence_for_write_count": int(
                blocked_weak_source_count
            ),
            "min_write_guard_hazard_source_support": support_min,
            "min_write_guard_hazard_source_dominance": round(dominance_min, 6),
            "proposal_promotion_state": str(
                proposal.get("promotion_state", "") or ""
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
            "model_repair_write_enabled": False,
        }
    )


def summarize_tile_effect_model_repair_write_audit(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    min_return_anchor_prior_non_reset_observations: int = 1,
    allow_return_anchor_prior_occupancy_trace: bool = False,
    require_hazard_source_evidence: bool = True,
    require_hazard_source_distinct_from_return_anchor: bool = True,
    min_scoreless_reset_same_identity_observations: int = 0,
    max_score_linked_reset_packet_move_count: int = 4,
    min_hazard_source_support: int = 1,
    min_hazard_source_dominance: float = 1.0,
    min_write_guard_hazard_source_support: int = 2,
    min_write_guard_hazard_source_dominance: float = 1.0,
    min_write_audit_scoreless_control_observations: int = 0,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Dry-run audit for a future tile-effect model repair write.

    This summary produces rollback metadata and a non-mutating patch payload.
    It is intentionally one step short of any write path: the patch has
    `write_enabled=false`, and the summary keeps all planner/runtime switches
    disabled.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    guard = summarize_tile_effect_model_repair_write_guard(
        packet_rows,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_reset_same_identity_observations=(
            min_scoreless_reset_same_identity_observations
        ),
        max_score_linked_reset_packet_move_count=(
            max_score_linked_reset_packet_move_count
        ),
        min_hazard_source_support=min_hazard_source_support,
        min_hazard_source_dominance=min_hazard_source_dominance,
        min_write_guard_hazard_source_support=min_write_guard_hazard_source_support,
        min_write_guard_hazard_source_dominance=(
            min_write_guard_hazard_source_dominance
        ),
        sample_limit=100000,
    )
    scoreless_min = max(0, int(min_write_audit_scoreless_control_observations))

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    ready_count = 0
    blocked_count = 0
    blocked_missing_rollback_count = 0
    blocked_missing_scoreless_control_count = 0
    ready_games: set[str] = set()
    blocked_games: set[str] = set()
    by_game: dict[str, Counter[str]] = {}

    for guard_row in list(guard.get("rows") or []):
        if not isinstance(guard_row, Mapping) or not bool(
            guard_row.get("model_repair_write_permitted_shadow")
        ):
            continue
        game_id = str(guard_row.get("game_id", "") or "unknown")
        identity_key = str(guard_row.get("identity_key", "") or "")
        return_anchor = _vector_from_value(guard_row.get("return_anchor"))
        hazard_source = _vector_from_value(guard_row.get("hazard_source_cell"))
        scoreless_control = _safe_int(
            guard_row.get("scoreless_reset_same_identity_control_count")
        )
        rollback_key = ""
        if identity_key and return_anchor is not None and hazard_source is not None:
            rollback_key = (
                f"identity={identity_key}|return={return_anchor[0]},{return_anchor[1]}"
                f"|avoid={hazard_source[0]},{hazard_source[1]}"
            )

        blocked_reasons: list[str] = []
        if not rollback_key:
            blocked_reasons.append("missing_rollback_key")
            blocked_missing_rollback_count += 1
        if scoreless_control < scoreless_min:
            blocked_reasons.append("missing_scoreless_control_for_write_audit")
            blocked_missing_scoreless_control_count += 1

        ready = not blocked_reasons
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["model_repair_write_audit_reviewed_count"] += 1
        if ready:
            ready_count += 1
            ready_games.add(game_id)
            game_counter["model_repair_write_audit_ready_count"] += 1
            audit_state = "tile_effect_model_repair_write_audit_shadow_ready"
        else:
            blocked_count += 1
            blocked_games.add(game_id)
            game_counter["model_repair_write_audit_blocked_count"] += 1
            for reason in blocked_reasons:
                game_counter[f"blocked_{reason}_count"] += 1
            audit_state = "tile_effect_model_repair_write_audit_blocked"

        dry_run_patch = {
            "schema_version": "eca_tile_effect_model_repair_patch_dry_run.v1",
            "patch_kind": "avoid_hazard_source_before_return_anchor",
            "identity_key": identity_key,
            "return_anchor": (
                [return_anchor[0], return_anchor[1]]
                if return_anchor is not None
                else []
            ),
            "avoid_hazard_source_cell": (
                [hazard_source[0], hazard_source[1]]
                if hazard_source is not None
                else []
            ),
            "rollback_key": rollback_key,
            "write_enabled": False,
        }
        row = {
            "game_id": game_id,
            "posterior_key": str(guard_row.get("posterior_key", "") or ""),
            "identity_key": identity_key,
            "audit_state": audit_state,
            "model_repair_write_audit_ready_shadow": ready,
            "blocked_reasons": blocked_reasons,
            "rollback_key": rollback_key,
            "rollback_plan_available_shadow": bool(rollback_key),
            "scoreless_reset_same_identity_control_count": scoreless_control,
            "min_write_audit_scoreless_control_observations": scoreless_min,
            "score_linked_reset_packet_move_count_max": _safe_int(
                guard_row.get("score_linked_reset_packet_move_count_max")
            ),
            "max_score_linked_reset_packet_move_count": _safe_int(
                guard_row.get("max_score_linked_reset_packet_move_count")
            ),
            "dry_run_patch": dry_run_patch,
            "model_repair_write_enabled": False,
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if ready_count > 0 and blocked_count > 0:
        promotion_state = "tile_effect_model_repair_write_audit_ready_with_blocks"
    elif ready_count > 0:
        promotion_state = "tile_effect_model_repair_write_audit_ready"
    elif blocked_count > 0:
        promotion_state = "tile_effect_model_repair_write_audit_all_blocked"
    elif int(guard.get("model_repair_write_guard_ready_count", 0) or 0) > 0:
        promotion_state = "tile_effect_model_repair_write_audit_no_review_rows"
    elif int(guard.get("packet_count", 0) or 0) > 0:
        promotion_state = "tile_effect_model_repair_write_audit_no_write_guard_ready"
    else:
        promotion_state = "tile_effect_model_repair_write_audit_no_packets"

    return _json_safe(
        {
            "schema_version": TILE_EFFECT_MODEL_REPAIR_WRITE_AUDIT_SCHEMA_VERSION,
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(guard.get("packet_count", 0) or 0),
            "model_repair_write_guard_ready_input_count": int(
                guard.get("model_repair_write_guard_ready_count", 0) or 0
            ),
            "model_repair_write_audit_reviewed_count": len(rows),
            "model_repair_write_audit_ready_count": int(ready_count),
            "model_repair_write_audit_blocked_count": int(blocked_count),
            "model_repair_write_audit_ready_game_count": len(ready_games),
            "model_repair_write_audit_blocked_game_count": len(blocked_games),
            "blocked_missing_rollback_key_count": int(
                blocked_missing_rollback_count
            ),
            "blocked_missing_scoreless_control_for_write_audit_count": int(
                blocked_missing_scoreless_control_count
            ),
            "min_write_audit_scoreless_control_observations": scoreless_min,
            "write_guard_promotion_state": str(
                guard.get("promotion_state", "") or ""
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
            "model_repair_write_enabled": False,
        }
    )


def summarize_tile_effect_model_repair_runtime_probe_proposal(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    min_return_anchor_prior_non_reset_observations: int = 1,
    allow_return_anchor_prior_occupancy_trace: bool = False,
    require_hazard_source_evidence: bool = True,
    require_hazard_source_distinct_from_return_anchor: bool = True,
    min_scoreless_reset_same_identity_observations: int = 0,
    max_score_linked_reset_packet_move_count: int = 4,
    min_hazard_source_support: int = 1,
    min_hazard_source_dominance: float = 1.0,
    min_write_guard_hazard_source_support: int = 2,
    min_write_guard_hazard_source_dominance: float = 1.0,
    min_write_audit_scoreless_control_observations: int = 0,
    min_runtime_probe_scoreless_control_observations: int = 0,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Shadow contract for a future protected runtime probe."""

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    audit = summarize_tile_effect_model_repair_write_audit(
        packet_rows,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_reset_same_identity_observations=(
            min_scoreless_reset_same_identity_observations
        ),
        max_score_linked_reset_packet_move_count=(
            max_score_linked_reset_packet_move_count
        ),
        min_hazard_source_support=min_hazard_source_support,
        min_hazard_source_dominance=min_hazard_source_dominance,
        min_write_guard_hazard_source_support=min_write_guard_hazard_source_support,
        min_write_guard_hazard_source_dominance=(
            min_write_guard_hazard_source_dominance
        ),
        min_write_audit_scoreless_control_observations=(
            min_write_audit_scoreless_control_observations
        ),
        sample_limit=100000,
    )
    scoreless_min = max(0, int(min_runtime_probe_scoreless_control_observations))

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    ready_count = 0
    blocked_count = 0
    blocked_missing_patch_count = 0
    blocked_patch_write_enabled_count = 0
    blocked_missing_coordinates_count = 0
    blocked_missing_rollback_count = 0
    blocked_missing_scoreless_control_count = 0
    ready_games: set[str] = set()
    blocked_games: set[str] = set()
    by_game: dict[str, Counter[str]] = {}

    for audit_row in list(audit.get("rows") or []):
        if not isinstance(audit_row, Mapping) or not bool(
            audit_row.get("model_repair_write_audit_ready_shadow")
        ):
            continue
        game_id = str(audit_row.get("game_id", "") or "unknown")
        identity_key = str(audit_row.get("identity_key", "") or "")
        patch_value = audit_row.get("dry_run_patch")
        patch = dict(patch_value) if isinstance(patch_value, Mapping) else {}
        return_anchor = _vector_from_value(patch.get("return_anchor"))
        hazard_source = _vector_from_value(patch.get("avoid_hazard_source_cell"))
        rollback_key = str(
            patch.get("rollback_key") or audit_row.get("rollback_key") or ""
        )
        scoreless_control = _safe_int(
            audit_row.get("scoreless_reset_same_identity_control_count")
        )

        blocked_reasons: list[str] = []
        if not patch:
            blocked_reasons.append("missing_dry_run_patch")
            blocked_missing_patch_count += 1
        if bool(patch.get("write_enabled")):
            blocked_reasons.append("dry_run_patch_write_enabled_not_allowed")
            blocked_patch_write_enabled_count += 1
        if return_anchor is None or hazard_source is None:
            blocked_reasons.append("missing_probe_contract_coordinates")
            blocked_missing_coordinates_count += 1
        if not rollback_key:
            blocked_reasons.append("missing_rollback_key")
            blocked_missing_rollback_count += 1
        if scoreless_control < scoreless_min:
            blocked_reasons.append("missing_scoreless_control_for_runtime_probe")
            blocked_missing_scoreless_control_count += 1

        ready = not blocked_reasons
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["runtime_probe_proposal_reviewed_count"] += 1
        if ready:
            ready_count += 1
            ready_games.add(game_id)
            game_counter["runtime_probe_proposal_ready_count"] += 1
            proposal_state = "tile_effect_runtime_probe_proposal_shadow_ready"
        else:
            blocked_count += 1
            blocked_games.add(game_id)
            game_counter["runtime_probe_proposal_blocked_count"] += 1
            for reason in blocked_reasons:
                game_counter[f"blocked_{reason}_count"] += 1
            proposal_state = "tile_effect_runtime_probe_proposal_blocked"

        protected_probe_contract = {
            "schema_version": "eca_tile_effect_model_repair_runtime_probe_contract.v1",
            "probe_kind": "avoid_hazard_source_before_return_anchor",
            "identity_key": identity_key,
            "return_anchor": (
                [return_anchor[0], return_anchor[1]]
                if return_anchor is not None
                else []
            ),
            "avoid_hazard_source_cell": (
                [hazard_source[0], hazard_source[1]]
                if hazard_source is not None
                else []
            ),
            "rollback_key": rollback_key,
            "execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
        }
        row = {
            "game_id": game_id,
            "posterior_key": str(audit_row.get("posterior_key", "") or ""),
            "identity_key": identity_key,
            "proposal_state": proposal_state,
            "runtime_probe_proposal_ready_shadow": ready,
            "blocked_reasons": blocked_reasons,
            "rollback_key": rollback_key,
            "scoreless_reset_same_identity_control_count": scoreless_control,
            "min_runtime_probe_scoreless_control_observations": scoreless_min,
            "protected_probe_contract": protected_probe_contract,
            "runtime_probe_execution_enabled": False,
            "model_repair_write_enabled": False,
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if ready_count > 0 and blocked_count > 0:
        promotion_state = "tile_effect_model_repair_runtime_probe_proposal_ready_with_blocks"
    elif ready_count > 0:
        promotion_state = "tile_effect_model_repair_runtime_probe_proposal_ready"
    elif blocked_count > 0:
        promotion_state = "tile_effect_model_repair_runtime_probe_proposal_all_blocked"
    elif int(audit.get("model_repair_write_audit_ready_count", 0) or 0) > 0:
        promotion_state = "tile_effect_model_repair_runtime_probe_proposal_no_review_rows"
    elif int(audit.get("packet_count", 0) or 0) > 0:
        promotion_state = "tile_effect_model_repair_runtime_probe_proposal_no_audit_ready"
    else:
        promotion_state = "tile_effect_model_repair_runtime_probe_proposal_no_packets"

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_RUNTIME_PROBE_PROPOSAL_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(audit.get("packet_count", 0) or 0),
            "model_repair_write_audit_ready_input_count": int(
                audit.get("model_repair_write_audit_ready_count", 0) or 0
            ),
            "runtime_probe_proposal_reviewed_count": len(rows),
            "runtime_probe_proposal_ready_count": int(ready_count),
            "runtime_probe_proposal_blocked_count": int(blocked_count),
            "runtime_probe_proposal_ready_game_count": len(ready_games),
            "runtime_probe_proposal_blocked_game_count": len(blocked_games),
            "blocked_missing_dry_run_patch_count": int(blocked_missing_patch_count),
            "blocked_dry_run_patch_write_enabled_not_allowed_count": int(
                blocked_patch_write_enabled_count
            ),
            "blocked_missing_probe_contract_coordinates_count": int(
                blocked_missing_coordinates_count
            ),
            "blocked_missing_rollback_key_count": int(
                blocked_missing_rollback_count
            ),
            "blocked_missing_scoreless_control_for_runtime_probe_count": int(
                blocked_missing_scoreless_control_count
            ),
            "min_runtime_probe_scoreless_control_observations": scoreless_min,
            "write_audit_promotion_state": str(
                audit.get("promotion_state", "") or ""
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
            "model_repair_write_enabled": False,
        }
    )


def summarize_tile_effect_model_repair_protected_probe_execution_gate(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    min_return_anchor_prior_non_reset_observations: int = 1,
    allow_return_anchor_prior_occupancy_trace: bool = False,
    require_hazard_source_evidence: bool = True,
    require_hazard_source_distinct_from_return_anchor: bool = True,
    min_scoreless_reset_same_identity_observations: int = 0,
    max_score_linked_reset_packet_move_count: int = 4,
    min_hazard_source_support: int = 1,
    min_hazard_source_dominance: float = 1.0,
    min_write_guard_hazard_source_support: int = 2,
    min_write_guard_hazard_source_dominance: float = 1.0,
    min_write_audit_scoreless_control_observations: int = 0,
    min_runtime_probe_scoreless_control_observations: int = 0,
    min_execution_gate_scoreless_control_observations: int = 0,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Shadow admission gate for future protected probe execution.

    The output says a probe *would be admissible* in a later execution phase.
    This function intentionally keeps execution, ranking, action generation,
    and model writes disabled.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    proposal = summarize_tile_effect_model_repair_runtime_probe_proposal(
        packet_rows,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_reset_same_identity_observations=(
            min_scoreless_reset_same_identity_observations
        ),
        max_score_linked_reset_packet_move_count=(
            max_score_linked_reset_packet_move_count
        ),
        min_hazard_source_support=min_hazard_source_support,
        min_hazard_source_dominance=min_hazard_source_dominance,
        min_write_guard_hazard_source_support=min_write_guard_hazard_source_support,
        min_write_guard_hazard_source_dominance=(
            min_write_guard_hazard_source_dominance
        ),
        min_write_audit_scoreless_control_observations=(
            min_write_audit_scoreless_control_observations
        ),
        min_runtime_probe_scoreless_control_observations=(
            min_runtime_probe_scoreless_control_observations
        ),
        sample_limit=100000,
    )
    scoreless_min = max(0, int(min_execution_gate_scoreless_control_observations))

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    ready_count = 0
    blocked_count = 0
    blocked_missing_contract_count = 0
    blocked_contract_execution_enabled_count = 0
    blocked_missing_coordinates_count = 0
    blocked_missing_rollback_count = 0
    blocked_missing_scoreless_control_count = 0
    ready_games: set[str] = set()
    blocked_games: set[str] = set()
    by_game: dict[str, Counter[str]] = {}

    for proposal_row in list(proposal.get("rows") or []):
        if not isinstance(proposal_row, Mapping) or not bool(
            proposal_row.get("runtime_probe_proposal_ready_shadow")
        ):
            continue
        game_id = str(proposal_row.get("game_id", "") or "unknown")
        identity_key = str(proposal_row.get("identity_key", "") or "")
        contract_value = proposal_row.get("protected_probe_contract")
        contract = (
            dict(contract_value) if isinstance(contract_value, Mapping) else {}
        )
        return_anchor = _vector_from_value(contract.get("return_anchor"))
        hazard_source = _vector_from_value(contract.get("avoid_hazard_source_cell"))
        rollback_key = str(
            contract.get("rollback_key") or proposal_row.get("rollback_key") or ""
        )
        scoreless_control = _safe_int(
            proposal_row.get("scoreless_reset_same_identity_control_count")
        )

        blocked_reasons: list[str] = []
        if not contract:
            blocked_reasons.append("missing_protected_probe_contract")
            blocked_missing_contract_count += 1
        if bool(contract.get("execution_enabled")):
            blocked_reasons.append("preenabled_contract_execution_not_allowed")
            blocked_contract_execution_enabled_count += 1
        if return_anchor is None or hazard_source is None:
            blocked_reasons.append("missing_execution_gate_coordinates")
            blocked_missing_coordinates_count += 1
        if not rollback_key:
            blocked_reasons.append("missing_rollback_key")
            blocked_missing_rollback_count += 1
        if scoreless_control < scoreless_min:
            blocked_reasons.append("missing_scoreless_control_for_execution_gate")
            blocked_missing_scoreless_control_count += 1

        ready = not blocked_reasons
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["protected_probe_execution_gate_reviewed_count"] += 1
        if ready:
            ready_count += 1
            ready_games.add(game_id)
            game_counter["protected_probe_execution_gate_ready_count"] += 1
            gate_state = "tile_effect_protected_probe_execution_gate_shadow_ready"
        else:
            blocked_count += 1
            blocked_games.add(game_id)
            game_counter["protected_probe_execution_gate_blocked_count"] += 1
            for reason in blocked_reasons:
                game_counter[f"blocked_{reason}_count"] += 1
            gate_state = "tile_effect_protected_probe_execution_gate_blocked"

        execution_admission_contract = {
            "schema_version": "eca_tile_effect_model_repair_execution_admission_contract.v1",
            "admission_kind": "protected_probe",
            "probe_kind": "avoid_hazard_source_before_return_anchor",
            "identity_key": identity_key,
            "return_anchor": (
                [return_anchor[0], return_anchor[1]]
                if return_anchor is not None
                else []
            ),
            "avoid_hazard_source_cell": (
                [hazard_source[0], hazard_source[1]]
                if hazard_source is not None
                else []
            ),
            "rollback_key": rollback_key,
            "would_execute_under_next_phase": ready,
            "execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
        }
        row = {
            "game_id": game_id,
            "posterior_key": str(proposal_row.get("posterior_key", "") or ""),
            "identity_key": identity_key,
            "gate_state": gate_state,
            "protected_probe_execution_admissible_shadow": ready,
            "blocked_reasons": blocked_reasons,
            "rollback_key": rollback_key,
            "scoreless_reset_same_identity_control_count": scoreless_control,
            "min_execution_gate_scoreless_control_observations": scoreless_min,
            "execution_admission_contract": execution_admission_contract,
            "runtime_probe_execution_enabled": False,
            "model_repair_write_enabled": False,
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if ready_count > 0 and blocked_count > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_execution_gate_ready_with_blocks"
    elif ready_count > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_execution_gate_ready"
    elif blocked_count > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_execution_gate_all_blocked"
    elif int(proposal.get("runtime_probe_proposal_ready_count", 0) or 0) > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_execution_gate_no_review_rows"
    elif int(proposal.get("packet_count", 0) or 0) > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_execution_gate_no_runtime_probe_ready"
    else:
        promotion_state = "tile_effect_model_repair_protected_probe_execution_gate_no_packets"

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_EXECUTION_GATE_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(proposal.get("packet_count", 0) or 0),
            "runtime_probe_proposal_ready_input_count": int(
                proposal.get("runtime_probe_proposal_ready_count", 0) or 0
            ),
            "protected_probe_execution_gate_reviewed_count": len(rows),
            "protected_probe_execution_gate_ready_count": int(ready_count),
            "protected_probe_execution_gate_blocked_count": int(blocked_count),
            "protected_probe_execution_gate_ready_game_count": len(ready_games),
            "protected_probe_execution_gate_blocked_game_count": len(blocked_games),
            "blocked_missing_protected_probe_contract_count": int(
                blocked_missing_contract_count
            ),
            "blocked_preenabled_contract_execution_not_allowed_count": int(
                blocked_contract_execution_enabled_count
            ),
            "blocked_missing_execution_gate_coordinates_count": int(
                blocked_missing_coordinates_count
            ),
            "blocked_missing_rollback_key_count": int(
                blocked_missing_rollback_count
            ),
            "blocked_missing_scoreless_control_for_execution_gate_count": int(
                blocked_missing_scoreless_control_count
            ),
            "min_execution_gate_scoreless_control_observations": scoreless_min,
            "runtime_probe_promotion_state": str(
                proposal.get("promotion_state", "") or ""
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
            "model_repair_write_enabled": False,
        }
    )


def summarize_tile_effect_model_repair_protected_probe_runtime_bridge_guard(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    min_return_anchor_prior_non_reset_observations: int = 1,
    allow_return_anchor_prior_occupancy_trace: bool = False,
    require_hazard_source_evidence: bool = True,
    require_hazard_source_distinct_from_return_anchor: bool = True,
    min_scoreless_reset_same_identity_observations: int = 0,
    max_score_linked_reset_packet_move_count: int = 4,
    min_hazard_source_support: int = 1,
    min_hazard_source_dominance: float = 1.0,
    min_write_guard_hazard_source_support: int = 2,
    min_write_guard_hazard_source_dominance: float = 1.0,
    min_write_audit_scoreless_control_observations: int = 0,
    min_runtime_probe_scoreless_control_observations: int = 0,
    min_execution_gate_scoreless_control_observations: int = 0,
    min_runtime_bridge_scoreless_control_observations: int = 0,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Guard handoff from admitted contracts to a future runtime executor.

    This does not execute the protected probe. It creates a stable event schema
    that a later executor can consume, and keeps action generation, ranking,
    and model writes disabled.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    gate = summarize_tile_effect_model_repair_protected_probe_execution_gate(
        packet_rows,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_reset_same_identity_observations=(
            min_scoreless_reset_same_identity_observations
        ),
        max_score_linked_reset_packet_move_count=(
            max_score_linked_reset_packet_move_count
        ),
        min_hazard_source_support=min_hazard_source_support,
        min_hazard_source_dominance=min_hazard_source_dominance,
        min_write_guard_hazard_source_support=min_write_guard_hazard_source_support,
        min_write_guard_hazard_source_dominance=(
            min_write_guard_hazard_source_dominance
        ),
        min_write_audit_scoreless_control_observations=(
            min_write_audit_scoreless_control_observations
        ),
        min_runtime_probe_scoreless_control_observations=(
            min_runtime_probe_scoreless_control_observations
        ),
        min_execution_gate_scoreless_control_observations=(
            min_execution_gate_scoreless_control_observations
        ),
        sample_limit=100000,
    )
    scoreless_min = max(0, int(min_runtime_bridge_scoreless_control_observations))

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    admitted_count = 0
    blocked_count = 0
    blocked_missing_contract_count = 0
    blocked_not_admissible_count = 0
    blocked_missing_coordinates_count = 0
    blocked_missing_rollback_count = 0
    blocked_missing_scoreless_control_count = 0
    admitted_games: set[str] = set()
    blocked_games: set[str] = set()
    by_game: dict[str, Counter[str]] = {}

    for gate_row in list(gate.get("rows") or []):
        if not isinstance(gate_row, Mapping):
            continue
        game_id = str(gate_row.get("game_id", "") or "unknown")
        identity_key = str(gate_row.get("identity_key", "") or "")
        contract_value = gate_row.get("execution_admission_contract")
        contract = (
            dict(contract_value) if isinstance(contract_value, Mapping) else {}
        )
        return_anchor = _vector_from_value(contract.get("return_anchor"))
        hazard_source = _vector_from_value(contract.get("avoid_hazard_source_cell"))
        rollback_key = str(
            contract.get("rollback_key") or gate_row.get("rollback_key") or ""
        )
        scoreless_control = _safe_int(
            gate_row.get("scoreless_reset_same_identity_control_count")
        )

        blocked_reasons: list[str] = []
        if not bool(gate_row.get("protected_probe_execution_admissible_shadow")):
            blocked_reasons.append("execution_gate_not_admissible")
            blocked_not_admissible_count += 1
        if not contract:
            blocked_reasons.append("missing_execution_admission_contract")
            blocked_missing_contract_count += 1
        if return_anchor is None or hazard_source is None:
            blocked_reasons.append("missing_runtime_bridge_coordinates")
            blocked_missing_coordinates_count += 1
        if not rollback_key:
            blocked_reasons.append("missing_rollback_key")
            blocked_missing_rollback_count += 1
        if scoreless_control < scoreless_min:
            blocked_reasons.append("missing_scoreless_control_for_runtime_bridge")
            blocked_missing_scoreless_control_count += 1

        admitted = not blocked_reasons
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["protected_probe_runtime_bridge_guard_reviewed_count"] += 1
        if admitted:
            admitted_count += 1
            admitted_games.add(game_id)
            game_counter["protected_probe_runtime_bridge_guard_admitted_count"] += 1
            bridge_state = "tile_effect_protected_probe_runtime_bridge_guard_admitted"
        else:
            blocked_count += 1
            blocked_games.add(game_id)
            game_counter["protected_probe_runtime_bridge_guard_blocked_count"] += 1
            for reason in blocked_reasons:
                game_counter[f"blocked_{reason}_count"] += 1
            bridge_state = "tile_effect_protected_probe_runtime_bridge_guard_blocked"

        runtime_bridge_event = {
            "schema_version": "eca_tile_effect_model_repair_runtime_bridge_guard_event.v1",
            "event_kind": "protected_probe_runtime_bridge_guard",
            "probe_kind": "avoid_hazard_source_before_return_anchor",
            "game_id": game_id,
            "identity_key": identity_key,
            "return_anchor": (
                [return_anchor[0], return_anchor[1]]
                if return_anchor is not None
                else []
            ),
            "avoid_hazard_source_cell": (
                [hazard_source[0], hazard_source[1]]
                if hazard_source is not None
                else []
            ),
            "rollback_key": rollback_key,
            "guard_admitted": admitted,
            "guard_state": bridge_state,
            "blocked_reasons": blocked_reasons,
            "executor_handoff_ready": admitted,
            "requires_runtime_executor": True,
            "runtime_probe_execution_enabled": False,
            "protected_probe_executed": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
            "allowed_connection": "admitted_contract_to_runtime_bridge_guard_only",
        }
        row = {
            "game_id": game_id,
            "posterior_key": str(gate_row.get("posterior_key", "") or ""),
            "identity_key": identity_key,
            "rollback_key": rollback_key,
            "runtime_bridge_guard_state": bridge_state,
            "runtime_bridge_guard_admitted": admitted,
            "blocked_reasons": blocked_reasons,
            "scoreless_reset_same_identity_control_count": scoreless_control,
            "min_runtime_bridge_scoreless_control_observations": scoreless_min,
            "execution_admission_contract": contract,
            "runtime_bridge_event": runtime_bridge_event,
            "runtime_probe_execution_enabled": False,
            "model_repair_write_enabled": False,
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if admitted_count > 0 and blocked_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_guard_ready_with_blocks"
        )
    elif admitted_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_guard_ready"
        )
    elif blocked_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_guard_all_blocked"
        )
    elif int(gate.get("protected_probe_execution_gate_ready_count", 0) or 0) > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_guard_no_review_rows"
        )
    elif int(gate.get("packet_count", 0) or 0) > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_guard_no_execution_gate_ready"
        )
    else:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_guard_no_packets"
        )

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_BRIDGE_GUARD_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(gate.get("packet_count", 0) or 0),
            "protected_probe_execution_gate_ready_input_count": int(
                gate.get("protected_probe_execution_gate_ready_count", 0) or 0
            ),
            "protected_probe_runtime_bridge_guard_reviewed_count": len(rows),
            "protected_probe_runtime_bridge_guard_admitted_count": int(
                admitted_count
            ),
            "protected_probe_runtime_bridge_guard_blocked_count": int(blocked_count),
            "protected_probe_runtime_bridge_guard_admitted_game_count": len(
                admitted_games
            ),
            "protected_probe_runtime_bridge_guard_blocked_game_count": len(
                blocked_games
            ),
            "blocked_execution_gate_not_admissible_count": int(
                blocked_not_admissible_count
            ),
            "blocked_missing_execution_admission_contract_count": int(
                blocked_missing_contract_count
            ),
            "blocked_missing_runtime_bridge_coordinates_count": int(
                blocked_missing_coordinates_count
            ),
            "blocked_missing_rollback_key_count": int(
                blocked_missing_rollback_count
            ),
            "blocked_missing_scoreless_control_for_runtime_bridge_count": int(
                blocked_missing_scoreless_control_count
            ),
            "min_runtime_bridge_scoreless_control_observations": scoreless_min,
            "execution_gate_promotion_state": str(
                gate.get("promotion_state", "") or ""
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
            "model_repair_write_enabled": False,
            "executor_handoff_ready": bool(admitted_count > 0),
        }
    )


def summarize_tile_effect_model_repair_protected_probe_runtime_bridge_outcome(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    runtime_bridge_events: Iterable[Mapping[str, Any]] | None = None,
    probe_outcomes: Iterable[Mapping[str, Any]] | None = None,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    min_return_anchor_prior_non_reset_observations: int = 1,
    allow_return_anchor_prior_occupancy_trace: bool = False,
    require_hazard_source_evidence: bool = True,
    require_hazard_source_distinct_from_return_anchor: bool = True,
    min_scoreless_reset_same_identity_observations: int = 0,
    max_score_linked_reset_packet_move_count: int = 4,
    min_hazard_source_support: int = 1,
    min_hazard_source_dominance: float = 1.0,
    min_write_guard_hazard_source_support: int = 2,
    min_write_guard_hazard_source_dominance: float = 1.0,
    min_write_audit_scoreless_control_observations: int = 0,
    min_runtime_probe_scoreless_control_observations: int = 0,
    min_execution_gate_scoreless_control_observations: int = 0,
    min_runtime_bridge_scoreless_control_observations: int = 0,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Join bridge-guard events to runtime probe outcomes.

    This is the runtime-facing contract after the Phase727 bridge guard. It
    does not create actions or write model repairs; it verifies that an admitted
    bridge event can carry a real protected-probe outcome into the existing
    outcome posterior.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    guard = summarize_tile_effect_model_repair_protected_probe_runtime_bridge_guard(
        packet_rows,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_reset_same_identity_observations=(
            min_scoreless_reset_same_identity_observations
        ),
        max_score_linked_reset_packet_move_count=(
            max_score_linked_reset_packet_move_count
        ),
        min_hazard_source_support=min_hazard_source_support,
        min_hazard_source_dominance=min_hazard_source_dominance,
        min_write_guard_hazard_source_support=min_write_guard_hazard_source_support,
        min_write_guard_hazard_source_dominance=(
            min_write_guard_hazard_source_dominance
        ),
        min_write_audit_scoreless_control_observations=(
            min_write_audit_scoreless_control_observations
        ),
        min_runtime_probe_scoreless_control_observations=(
            min_runtime_probe_scoreless_control_observations
        ),
        min_execution_gate_scoreless_control_observations=(
            min_execution_gate_scoreless_control_observations
        ),
        min_runtime_bridge_scoreless_control_observations=(
            min_runtime_bridge_scoreless_control_observations
        ),
        sample_limit=100000,
    )
    generated_events = [
        dict(row.get("runtime_bridge_event") or {})
        for row in list(guard.get("rows") or [])
        if isinstance(row, Mapping)
        and isinstance(row.get("runtime_bridge_event"), Mapping)
    ]
    supplied_events = [
        dict(event)
        for event in list(runtime_bridge_events or [])
        if isinstance(event, Mapping)
    ]
    event_rows = supplied_events if supplied_events else generated_events
    outcome_rows = [
        dict(row) for row in list(probe_outcomes or []) if isinstance(row, Mapping)
    ]
    outcome_by_rollback = {
        str(row.get("rollback_key", "") or ""): row
        for row in outcome_rows
        if str(row.get("rollback_key", "") or "")
    }
    admitted_rollbacks = {
        str(row.get("rollback_key", "") or "")
        for row in list(guard.get("rows") or [])
        if isinstance(row, Mapping)
        and bool(row.get("runtime_bridge_guard_admitted"))
        and str(row.get("rollback_key", "") or "")
    }
    posterior = summarize_tile_effect_model_repair_protected_probe_outcome_posterior(
        packet_rows,
        probe_outcomes=outcome_rows,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_reset_same_identity_observations=(
            min_scoreless_reset_same_identity_observations
        ),
        max_score_linked_reset_packet_move_count=(
            max_score_linked_reset_packet_move_count
        ),
        min_hazard_source_support=min_hazard_source_support,
        min_hazard_source_dominance=min_hazard_source_dominance,
        min_write_guard_hazard_source_support=min_write_guard_hazard_source_support,
        min_write_guard_hazard_source_dominance=(
            min_write_guard_hazard_source_dominance
        ),
        min_write_audit_scoreless_control_observations=(
            min_write_audit_scoreless_control_observations
        ),
        min_runtime_probe_scoreless_control_observations=(
            min_runtime_probe_scoreless_control_observations
        ),
        min_execution_gate_scoreless_control_observations=(
            min_execution_gate_scoreless_control_observations
        ),
        sample_limit=100000,
    )
    posterior_by_rollback = {
        str(row.get("rollback_key", "") or ""): row
        for row in list(posterior.get("rows") or [])
        if isinstance(row, Mapping) and str(row.get("rollback_key", "") or "")
    }

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    pending_count = 0
    attempted_count = 0
    confirmed_count = 0
    falsified_count = 0
    inconclusive_count = 0
    blocked_count = 0
    by_game: dict[str, Counter[str]] = {}

    for index, event in enumerate(event_rows):
        game_id = str(event.get("game_id", "") or "unknown")
        rollback_key = str(event.get("rollback_key", "") or "")
        identity_key = str(event.get("identity_key", "") or "")
        blocked_reasons = [
            str(reason)
            for reason in list(event.get("blocked_reasons") or [])
            if str(reason)
        ]
        if not bool(event.get("guard_admitted")):
            blocked_reasons.append("runtime_bridge_guard_not_admitted")
        if not bool(event.get("executor_handoff_ready")):
            blocked_reasons.append("executor_handoff_not_ready")
        if not rollback_key:
            blocked_reasons.append("missing_rollback_key")
        elif rollback_key not in admitted_rollbacks:
            blocked_reasons.append("runtime_bridge_event_not_backed_by_guard")

        outcome = outcome_by_rollback.get(rollback_key)
        posterior_row = posterior_by_rollback.get(rollback_key)
        posterior_state = str(
            (posterior_row or {}).get(
                "protected_probe_outcome_posterior_state", ""
            )
            or ""
        )
        attempted = bool(
            posterior_row
            and posterior_row.get("protected_probe_execution_attempted")
        )
        if blocked_reasons:
            bridge_state = "runtime_bridge_outcome_blocked"
            blocked_count += 1
        elif not outcome:
            bridge_state = "runtime_bridge_outcome_pending_runtime_execution"
            posterior_state = posterior_state or "pending_runtime_execution"
            pending_count += 1
        elif posterior_state == "confirmed":
            bridge_state = "runtime_bridge_outcome_confirmed"
            attempted_count += 1
            confirmed_count += 1
        elif posterior_state == "falsified":
            bridge_state = "runtime_bridge_outcome_falsified"
            attempted_count += 1
            falsified_count += 1
        else:
            bridge_state = "runtime_bridge_outcome_inconclusive"
            if attempted:
                attempted_count += 1
            inconclusive_count += 1

        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["runtime_bridge_outcome_reviewed_count"] += 1
        game_counter[f"{bridge_state}_count"] += 1
        if attempted:
            game_counter["runtime_bridge_outcome_attempted_count"] += 1

        row = {
            "game_id": game_id,
            "identity_key": identity_key,
            "rollback_key": rollback_key,
            "event_index": int(index),
            "runtime_bridge_outcome_state": bridge_state,
            "blocked_reasons": blocked_reasons,
            "protected_probe_execution_attempted": bool(attempted),
            "protected_probe_outcome_posterior_state": posterior_state,
            "observed_score_delta": _safe_float(
                (posterior_row or {}).get("observed_score_delta")
            ),
            "guard_admitted": bool(event.get("guard_admitted")),
            "executor_handoff_ready": bool(event.get("executor_handoff_ready")),
            "runtime_probe_execution_enabled": bool(attempted),
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if confirmed_count > 0 and falsified_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_outcome_mixed"
        )
    elif confirmed_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_outcome_confirmed"
        )
    elif falsified_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_outcome_falsified"
        )
    elif inconclusive_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_outcome_inconclusive"
        )
    elif pending_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_outcome_pending_runtime_execution"
        )
    elif blocked_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_outcome_all_blocked"
        )
    elif int(guard.get("protected_probe_runtime_bridge_guard_admitted_count", 0) or 0) > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_outcome_no_review_rows"
        )
    elif int(guard.get("packet_count", 0) or 0) > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_outcome_no_runtime_bridge_guard_ready"
        )
    else:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_bridge_outcome_no_packets"
        )

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_BRIDGE_OUTCOME_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "packet_count": int(guard.get("packet_count", 0) or 0),
            "runtime_bridge_guard_admitted_input_count": int(
                guard.get("protected_probe_runtime_bridge_guard_admitted_count", 0)
                or 0
            ),
            "runtime_bridge_event_source": (
                "supplied_runtime_bridge_events"
                if supplied_events
                else "derived_runtime_bridge_guard_events"
            ),
            "runtime_bridge_event_reviewed_count": len(rows),
            "runtime_bridge_outcome_pending_count": int(pending_count),
            "runtime_bridge_outcome_attempted_count": int(attempted_count),
            "runtime_bridge_outcome_confirmed_count": int(confirmed_count),
            "runtime_bridge_outcome_falsified_count": int(falsified_count),
            "runtime_bridge_outcome_inconclusive_count": int(inconclusive_count),
            "runtime_bridge_outcome_blocked_count": int(blocked_count),
            "probe_outcome_row_count": int(len(outcome_rows)),
            "outcome_posterior_promotion_state": str(
                posterior.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "rows": rows[:sample_limit],
            "samples": samples,
            "runtime_execution_observed": bool(attempted_count > 0),
            "runtime_execution_enabled": bool(attempted_count > 0),
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
            "planner_connection_enabled": False,
        }
    )


def summarize_tile_effect_model_repair_protected_probe_runtime_executor(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    runtime_bridge_events: Iterable[Mapping[str, Any]] | None = None,
    probe_outcomes: Iterable[Mapping[str, Any]] | None = None,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    min_return_anchor_prior_non_reset_observations: int = 1,
    allow_return_anchor_prior_occupancy_trace: bool = False,
    require_hazard_source_evidence: bool = True,
    require_hazard_source_distinct_from_return_anchor: bool = True,
    min_scoreless_reset_same_identity_observations: int = 0,
    max_score_linked_reset_packet_move_count: int = 4,
    min_hazard_source_support: int = 1,
    min_hazard_source_dominance: float = 1.0,
    min_write_guard_hazard_source_support: int = 2,
    min_write_guard_hazard_source_dominance: float = 1.0,
    min_write_audit_scoreless_control_observations: int = 0,
    min_runtime_probe_scoreless_control_observations: int = 0,
    min_execution_gate_scoreless_control_observations: int = 0,
    min_runtime_bridge_scoreless_control_observations: int = 0,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Materialize admitted bridge events as protected-probe executor requests.

    The executor remains bounded by the previously admitted contract: it can
    request the protected probe identified by the rollback key and verifier
    coordinates, but it does not rank candidates, generate planner actions, or
    write repaired models. If a runtime outcome is supplied, the request is
    joined through the existing outcome posterior.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    guard = summarize_tile_effect_model_repair_protected_probe_runtime_bridge_guard(
        packet_rows,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_reset_same_identity_observations=(
            min_scoreless_reset_same_identity_observations
        ),
        max_score_linked_reset_packet_move_count=(
            max_score_linked_reset_packet_move_count
        ),
        min_hazard_source_support=min_hazard_source_support,
        min_hazard_source_dominance=min_hazard_source_dominance,
        min_write_guard_hazard_source_support=min_write_guard_hazard_source_support,
        min_write_guard_hazard_source_dominance=(
            min_write_guard_hazard_source_dominance
        ),
        min_write_audit_scoreless_control_observations=(
            min_write_audit_scoreless_control_observations
        ),
        min_runtime_probe_scoreless_control_observations=(
            min_runtime_probe_scoreless_control_observations
        ),
        min_execution_gate_scoreless_control_observations=(
            min_execution_gate_scoreless_control_observations
        ),
        min_runtime_bridge_scoreless_control_observations=(
            min_runtime_bridge_scoreless_control_observations
        ),
        sample_limit=100000,
    )
    generated_events = [
        dict(row.get("runtime_bridge_event") or {})
        for row in list(guard.get("rows") or [])
        if isinstance(row, Mapping)
        and isinstance(row.get("runtime_bridge_event"), Mapping)
    ]
    supplied_events = [
        dict(event)
        for event in list(runtime_bridge_events or [])
        if isinstance(event, Mapping)
    ]
    event_rows = supplied_events if supplied_events else generated_events
    outcome_rows = [
        dict(row) for row in list(probe_outcomes or []) if isinstance(row, Mapping)
    ]
    outcome_by_rollback = {
        str(row.get("rollback_key", "") or ""): row
        for row in outcome_rows
        if str(row.get("rollback_key", "") or "")
    }
    admitted_rollbacks = {
        str(row.get("rollback_key", "") or "")
        for row in list(guard.get("rows") or [])
        if isinstance(row, Mapping)
        and bool(row.get("runtime_bridge_guard_admitted"))
        and str(row.get("rollback_key", "") or "")
    }
    bridge_outcome = summarize_tile_effect_model_repair_protected_probe_runtime_bridge_outcome(
        packet_rows,
        runtime_bridge_events=event_rows,
        probe_outcomes=outcome_rows,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_reset_same_identity_observations=(
            min_scoreless_reset_same_identity_observations
        ),
        max_score_linked_reset_packet_move_count=(
            max_score_linked_reset_packet_move_count
        ),
        min_hazard_source_support=min_hazard_source_support,
        min_hazard_source_dominance=min_hazard_source_dominance,
        min_write_guard_hazard_source_support=min_write_guard_hazard_source_support,
        min_write_guard_hazard_source_dominance=(
            min_write_guard_hazard_source_dominance
        ),
        min_write_audit_scoreless_control_observations=(
            min_write_audit_scoreless_control_observations
        ),
        min_runtime_probe_scoreless_control_observations=(
            min_runtime_probe_scoreless_control_observations
        ),
        min_execution_gate_scoreless_control_observations=(
            min_execution_gate_scoreless_control_observations
        ),
        min_runtime_bridge_scoreless_control_observations=(
            min_runtime_bridge_scoreless_control_observations
        ),
        sample_limit=100000,
    )
    bridge_outcome_by_rollback = {
        str(row.get("rollback_key", "") or ""): row
        for row in list(bridge_outcome.get("rows") or [])
        if isinstance(row, Mapping) and str(row.get("rollback_key", "") or "")
    }

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    emitted_count = 0
    admitted_count = 0
    pending_count = 0
    attempted_count = 0
    confirmed_count = 0
    falsified_count = 0
    inconclusive_count = 0
    blocked_count = 0
    by_game: dict[str, Counter[str]] = {}

    for index, event in enumerate(event_rows):
        game_id = str(event.get("game_id", "") or "unknown")
        rollback_key = str(event.get("rollback_key", "") or "")
        identity_key = str(event.get("identity_key", "") or "")
        return_anchor = _vector_from_value(event.get("return_anchor"))
        hazard_source = _vector_from_value(event.get("avoid_hazard_source_cell"))
        blocked_reasons = [
            str(reason)
            for reason in list(event.get("blocked_reasons") or [])
            if str(reason)
        ]
        if not bool(event.get("guard_admitted")):
            blocked_reasons.append("runtime_bridge_guard_not_admitted")
        if not bool(event.get("executor_handoff_ready")):
            blocked_reasons.append("executor_handoff_not_ready")
        if not rollback_key:
            blocked_reasons.append("missing_rollback_key")
        elif rollback_key not in admitted_rollbacks:
            blocked_reasons.append("runtime_bridge_event_not_backed_by_guard")
        if return_anchor is None or hazard_source is None:
            blocked_reasons.append("missing_executor_coordinates")

        outcome = outcome_by_rollback.get(rollback_key)
        bridge_row = bridge_outcome_by_rollback.get(rollback_key, {})
        posterior_state = str(
            bridge_row.get("protected_probe_outcome_posterior_state", "") or ""
        )
        attempted = bool(
            bridge_row and bridge_row.get("protected_probe_execution_attempted")
        )
        executor_request: dict[str, Any] | None = None
        if blocked_reasons:
            executor_state = "runtime_executor_request_blocked"
            blocked_count += 1
        else:
            admitted_count += 1
            emitted_count += 1
            executor_request = {
                "schema_version": (
                    "eca_tile_effect_model_repair_protected_probe_executor_request.v1"
                ),
                "event_kind": "protected_probe_executor_request",
                "request_kind": "avoid_hazard_source_before_return_anchor",
                "probe_kind": "avoid_hazard_source_before_return_anchor",
                "game_id": game_id,
                "identity_key": identity_key,
                "rollback_key": rollback_key,
                "return_anchor": [return_anchor[0], return_anchor[1]],
                "avoid_hazard_source_cell": [hazard_source[0], hazard_source[1]],
                "requires_runtime_executor": False,
                "protected_probe_execution_enabled": True,
                "runtime_probe_execution_enabled": True,
                "verifier_requires_hazard_avoided": True,
                "verifier_requires_return_anchor_reached": True,
                "verifier_uses_score_delta": True,
                "planner_connection_enabled": False,
                "ranking_enabled": False,
                "action_generation_allowed": False,
                "model_repair_write_enabled": False,
                "allowed_connection": "admitted_contract_to_protected_probe_executor_request",
            }
            if not outcome:
                executor_state = "runtime_executor_request_pending_outcome"
                posterior_state = posterior_state or "pending_runtime_execution"
                pending_count += 1
            elif posterior_state == "confirmed":
                executor_state = "runtime_executor_outcome_confirmed"
                attempted_count += 1
                confirmed_count += 1
            elif posterior_state == "falsified":
                executor_state = "runtime_executor_outcome_falsified"
                attempted_count += 1
                falsified_count += 1
            else:
                executor_state = "runtime_executor_outcome_inconclusive"
                if attempted:
                    attempted_count += 1
                inconclusive_count += 1

        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["runtime_executor_reviewed_count"] += 1
        game_counter[f"{executor_state}_count"] += 1
        if executor_request is not None:
            game_counter["runtime_executor_request_emitted_count"] += 1
        if attempted:
            game_counter["runtime_executor_outcome_attempted_count"] += 1

        row = {
            "game_id": game_id,
            "identity_key": identity_key,
            "rollback_key": rollback_key,
            "event_index": int(index),
            "runtime_executor_state": executor_state,
            "blocked_reasons": blocked_reasons,
            "runtime_bridge_guard_admitted": bool(event.get("guard_admitted")),
            "executor_handoff_ready": bool(event.get("executor_handoff_ready")),
            "protected_probe_executor_request": executor_request or {},
            "protected_probe_execution_attempted": bool(attempted),
            "protected_probe_outcome_posterior_state": posterior_state,
            "observed_score_delta": _safe_float(
                bridge_row.get("observed_score_delta")
            ),
            "runtime_probe_execution_enabled": executor_request is not None,
            "protected_probe_execution_enabled": executor_request is not None,
            "planner_connection_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if confirmed_count > 0 and falsified_count > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_runtime_executor_mixed"
    elif confirmed_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_executor_confirmed"
        )
    elif falsified_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_executor_falsified"
        )
    elif inconclusive_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_executor_inconclusive"
        )
    elif pending_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_executor_pending_outcome"
        )
    elif blocked_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_executor_all_blocked"
        )
    elif int(guard.get("protected_probe_runtime_bridge_guard_admitted_count", 0) or 0) > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_executor_no_review_rows"
        )
    elif int(guard.get("packet_count", 0) or 0) > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_executor_no_runtime_bridge_guard_ready"
        )
    else:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_runtime_executor_no_packets"
        )

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_EXECUTOR_SCHEMA_VERSION
            ),
            "diagnostic_only": False,
            "packet_count": int(guard.get("packet_count", 0) or 0),
            "runtime_bridge_guard_admitted_input_count": int(
                guard.get("protected_probe_runtime_bridge_guard_admitted_count", 0)
                or 0
            ),
            "runtime_executor_event_source": (
                "supplied_runtime_bridge_events"
                if supplied_events
                else "derived_runtime_bridge_guard_events"
            ),
            "runtime_executor_reviewed_count": len(rows),
            "runtime_executor_admitted_count": int(admitted_count),
            "runtime_executor_request_emitted_count": int(emitted_count),
            "runtime_executor_pending_outcome_count": int(pending_count),
            "runtime_executor_attempted_count": int(attempted_count),
            "runtime_executor_confirmed_count": int(confirmed_count),
            "runtime_executor_falsified_count": int(falsified_count),
            "runtime_executor_inconclusive_count": int(inconclusive_count),
            "runtime_executor_blocked_count": int(blocked_count),
            "probe_outcome_row_count": int(len(outcome_rows)),
            "runtime_bridge_outcome_promotion_state": str(
                bridge_outcome.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "rows": rows[:sample_limit],
            "samples": samples,
            "runtime_executor_enabled": bool(emitted_count > 0),
            "runtime_execution_enabled": bool(emitted_count > 0),
            "runtime_execution_observed": bool(attempted_count > 0),
            "protected_probe_execution_observed": bool(attempted_count > 0),
            "planner_connection_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
        }
    )


def summarize_tile_effect_model_repair_protected_probe_outcome_posterior(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    probe_outcomes: Iterable[Mapping[str, Any]] | None = None,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    min_return_anchor_prior_non_reset_observations: int = 1,
    allow_return_anchor_prior_occupancy_trace: bool = False,
    require_hazard_source_evidence: bool = True,
    require_hazard_source_distinct_from_return_anchor: bool = True,
    min_scoreless_reset_same_identity_observations: int = 0,
    max_score_linked_reset_packet_move_count: int = 4,
    min_hazard_source_support: int = 1,
    min_hazard_source_dominance: float = 1.0,
    min_write_guard_hazard_source_support: int = 2,
    min_write_guard_hazard_source_dominance: float = 1.0,
    min_write_audit_scoreless_control_observations: int = 0,
    min_runtime_probe_scoreless_control_observations: int = 0,
    min_execution_gate_scoreless_control_observations: int = 0,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Outcome posterior contract for protected tile-effect repair probes.

    This is the next handoff after execution admission. It records how a
    protected probe outcome would be folded into posterior state, while keeping
    action generation, ranking, and model writes disabled. Real execution rows
    may be supplied by a later runtime bridge via ``probe_outcomes``.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    gate = summarize_tile_effect_model_repair_protected_probe_execution_gate(
        packet_rows,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_reset_same_identity_observations=(
            min_scoreless_reset_same_identity_observations
        ),
        max_score_linked_reset_packet_move_count=(
            max_score_linked_reset_packet_move_count
        ),
        min_hazard_source_support=min_hazard_source_support,
        min_hazard_source_dominance=min_hazard_source_dominance,
        min_write_guard_hazard_source_support=min_write_guard_hazard_source_support,
        min_write_guard_hazard_source_dominance=(
            min_write_guard_hazard_source_dominance
        ),
        min_write_audit_scoreless_control_observations=(
            min_write_audit_scoreless_control_observations
        ),
        min_runtime_probe_scoreless_control_observations=(
            min_runtime_probe_scoreless_control_observations
        ),
        min_execution_gate_scoreless_control_observations=(
            min_execution_gate_scoreless_control_observations
        ),
        sample_limit=100000,
    )
    outcome_rows = [
        dict(row) for row in (probe_outcomes or []) if isinstance(row, Mapping)
    ]
    outcome_by_rollback = {
        str(row.get("rollback_key", "") or ""): row
        for row in outcome_rows
        if str(row.get("rollback_key", "") or "")
    }

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    reviewed_count = 0
    pending_count = 0
    attempted_count = 0
    confirmed_count = 0
    falsified_count = 0
    inconclusive_count = 0
    ready_games: set[str] = set()
    confirmed_games: set[str] = set()
    falsified_games: set[str] = set()
    by_game: dict[str, Counter[str]] = {}

    for gate_row in list(gate.get("rows") or []):
        if not isinstance(gate_row, Mapping) or not bool(
            gate_row.get("protected_probe_execution_admissible_shadow")
        ):
            continue
        reviewed_count += 1
        game_id = str(gate_row.get("game_id", "") or "unknown")
        ready_games.add(game_id)
        identity_key = str(gate_row.get("identity_key", "") or "")
        contract_value = gate_row.get("execution_admission_contract")
        contract = (
            dict(contract_value) if isinstance(contract_value, Mapping) else {}
        )
        rollback_key = str(
            contract.get("rollback_key") or gate_row.get("rollback_key") or ""
        )
        outcome = outcome_by_rollback.get(rollback_key)
        attempted = bool(
            outcome
            and (
                outcome.get("protected_probe_executed")
                or outcome.get("execution_attempted")
                or outcome.get("runtime_probe_execution_enabled")
            )
        )
        score_delta = _safe_float(outcome.get("score_delta")) if outcome else 0.0
        reset_observed = bool(outcome.get("reset_observed")) if outcome else False
        hazard_avoided = bool(outcome.get("hazard_avoided")) if outcome else False
        return_anchor_reached = (
            bool(outcome.get("return_anchor_reached")) if outcome else False
        )
        no_effect = bool(outcome.get("no_effect")) if outcome else False

        if not outcome:
            posterior_state = "pending_runtime_execution"
            pending_count += 1
        elif not attempted:
            posterior_state = "inconclusive_outcome_without_execution"
            inconclusive_count += 1
        elif score_delta > 0 or (
            hazard_avoided and return_anchor_reached and not reset_observed
        ):
            posterior_state = "confirmed"
            attempted_count += 1
            confirmed_count += 1
            confirmed_games.add(game_id)
        elif reset_observed or no_effect or not hazard_avoided:
            posterior_state = "falsified"
            attempted_count += 1
            falsified_count += 1
            falsified_games.add(game_id)
        else:
            posterior_state = "inconclusive"
            attempted_count += 1
            inconclusive_count += 1

        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["protected_probe_outcome_reviewed_count"] += 1
        game_counter[f"protected_probe_outcome_{posterior_state}_count"] += 1
        if attempted:
            game_counter["protected_probe_execution_attempted_count"] += 1

        outcome_posterior = {
            "schema_version": "eca_tile_effect_model_repair_probe_outcome_posterior_row.v1",
            "posterior_key": str(gate_row.get("posterior_key", "") or ""),
            "identity_key": identity_key,
            "rollback_key": rollback_key,
            "posterior_state": posterior_state,
            "confirmed_by_runtime_probe": posterior_state == "confirmed",
            "falsified_by_runtime_probe": posterior_state == "falsified",
            "model_repair_write_enabled": False,
        }
        row = {
            "game_id": game_id,
            "posterior_key": str(gate_row.get("posterior_key", "") or ""),
            "identity_key": identity_key,
            "rollback_key": rollback_key,
            "protected_probe_outcome_posterior_state": posterior_state,
            "protected_probe_execution_attempted": attempted,
            "observed_score_delta": score_delta,
            "hazard_avoided": hazard_avoided,
            "return_anchor_reached": return_anchor_reached,
            "reset_observed": reset_observed,
            "no_effect": no_effect,
            "execution_admission_contract": contract,
            "outcome_posterior": outcome_posterior,
            "runtime_probe_execution_enabled": attempted,
            "model_repair_write_enabled": False,
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if confirmed_count > 0 and falsified_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_outcome_posterior_mixed"
        )
    elif confirmed_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_outcome_posterior_confirmed"
        )
    elif falsified_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_outcome_posterior_falsified"
        )
    elif inconclusive_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_outcome_posterior_inconclusive"
        )
    elif pending_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_outcome_posterior_pending_runtime_execution"
        )
    elif int(gate.get("protected_probe_execution_gate_ready_count", 0) or 0) > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_outcome_posterior_no_review_rows"
        )
    elif int(gate.get("packet_count", 0) or 0) > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_outcome_posterior_no_execution_gate_ready"
        )
    else:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_outcome_posterior_no_packets"
        )

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_OUTCOME_POSTERIOR_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(gate.get("packet_count", 0) or 0),
            "protected_probe_execution_gate_ready_input_count": int(
                gate.get("protected_probe_execution_gate_ready_count", 0) or 0
            ),
            "protected_probe_outcome_reviewed_count": int(reviewed_count),
            "protected_probe_execution_pending_count": int(pending_count),
            "protected_probe_execution_attempted_count": int(attempted_count),
            "protected_probe_outcome_confirmed_count": int(confirmed_count),
            "protected_probe_outcome_falsified_count": int(falsified_count),
            "protected_probe_outcome_inconclusive_count": int(inconclusive_count),
            "protected_probe_outcome_ready_game_count": len(ready_games),
            "protected_probe_outcome_confirmed_game_count": len(confirmed_games),
            "protected_probe_outcome_falsified_game_count": len(falsified_games),
            "execution_gate_promotion_state": str(
                gate.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "rows": rows[:sample_limit],
            "samples": samples,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": bool(attempted_count > 0),
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
        }
    )


def summarize_tile_effect_model_repair_protected_probe_rank_guard_shadow(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    probe_outcomes: Iterable[Mapping[str, Any]] | None = None,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    min_return_anchor_prior_non_reset_observations: int = 1,
    allow_return_anchor_prior_occupancy_trace: bool = False,
    require_hazard_source_evidence: bool = True,
    require_hazard_source_distinct_from_return_anchor: bool = True,
    min_scoreless_reset_same_identity_observations: int = 0,
    max_score_linked_reset_packet_move_count: int = 4,
    min_hazard_source_support: int = 1,
    min_hazard_source_dominance: float = 1.0,
    min_write_guard_hazard_source_support: int = 2,
    min_write_guard_hazard_source_dominance: float = 1.0,
    min_write_audit_scoreless_control_observations: int = 0,
    min_runtime_probe_scoreless_control_observations: int = 0,
    min_execution_gate_scoreless_control_observations: int = 0,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Translate protected-probe posterior rows into shadow rank-guard signals.

    This is a design-time gate between posterior and live ranking. Confirmed
    outcomes become shadow boost annotations; falsified outcomes become shadow
    suppress annotations. No candidate order is changed here.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    posterior = summarize_tile_effect_model_repair_protected_probe_outcome_posterior(
        packet_rows,
        probe_outcomes=probe_outcomes,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_reset_same_identity_observations=(
            min_scoreless_reset_same_identity_observations
        ),
        max_score_linked_reset_packet_move_count=(
            max_score_linked_reset_packet_move_count
        ),
        min_hazard_source_support=min_hazard_source_support,
        min_hazard_source_dominance=min_hazard_source_dominance,
        min_write_guard_hazard_source_support=min_write_guard_hazard_source_support,
        min_write_guard_hazard_source_dominance=(
            min_write_guard_hazard_source_dominance
        ),
        min_write_audit_scoreless_control_observations=(
            min_write_audit_scoreless_control_observations
        ),
        min_runtime_probe_scoreless_control_observations=(
            min_runtime_probe_scoreless_control_observations
        ),
        min_execution_gate_scoreless_control_observations=(
            min_execution_gate_scoreless_control_observations
        ),
        sample_limit=100000,
    )

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    reviewed_count = 0
    boost_count = 0
    suppress_count = 0
    no_op_count = 0
    by_game: dict[str, Counter[str]] = {}

    for index, posterior_row in enumerate(list(posterior.get("rows") or [])):
        if not isinstance(posterior_row, Mapping):
            continue
        reviewed_count += 1
        game_id = str(posterior_row.get("game_id", "") or "unknown")
        state = str(
            posterior_row.get("protected_probe_outcome_posterior_state", "") or ""
        )
        if state == "confirmed":
            guard_state = "rank_guard_shadow_boost_confirmed_probe"
            guard_effect = "shadow_boost_existing_matching_candidates"
            boost_count += 1
        elif state == "falsified":
            guard_state = "rank_guard_shadow_suppress_falsified_probe"
            guard_effect = "shadow_suppress_matching_candidates"
            suppress_count += 1
        else:
            guard_state = "rank_guard_shadow_no_op_unconfirmed_probe"
            guard_effect = "shadow_no_rank_change"
            no_op_count += 1

        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["rank_guard_shadow_reviewed_count"] += 1
        game_counter[f"{guard_state}_count"] += 1
        if state == "confirmed":
            game_counter["rank_guard_shadow_boost_count"] += 1
        elif state == "falsified":
            game_counter["rank_guard_shadow_suppress_count"] += 1
        else:
            game_counter["rank_guard_shadow_no_op_count"] += 1

        decision = {
            "schema_version": "eca_tile_effect_model_repair_rank_guard_shadow_decision.v1",
            "decision_kind": "protected_probe_rank_guard_shadow",
            "posterior_state": state,
            "guard_effect": guard_effect,
            "shadow_only": True,
            "would_boost_existing_candidate": state == "confirmed",
            "would_suppress_matching_candidate": state == "falsified",
            "rank_guard_runtime_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
            "allowed_connection": "protected_probe_outcome_posterior_to_shadow_rank_guard",
            "disallowed_effects": [
                "candidate_order_mutation",
                "new_action_generation",
                "model_repair_write",
            ],
        }
        row = {
            "game_id": game_id,
            "posterior_key": str(posterior_row.get("posterior_key", "") or ""),
            "identity_key": str(posterior_row.get("identity_key", "") or ""),
            "rollback_key": str(posterior_row.get("rollback_key", "") or ""),
            "row_index": int(index),
            "protected_probe_outcome_posterior_state": state,
            "rank_guard_shadow_state": guard_state,
            "rank_guard_shadow_decision": decision,
            "observed_score_delta": _safe_float(
                posterior_row.get("observed_score_delta")
            ),
            "protected_probe_execution_attempted": bool(
                posterior_row.get("protected_probe_execution_attempted")
            ),
            "shadow_rank_signal_ready": state in {"confirmed", "falsified"},
            "shadow_boost_ready": state == "confirmed",
            "shadow_suppress_ready": state == "falsified",
            "rank_guard_runtime_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
            "planner_connection_enabled": False,
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if boost_count > 0 and suppress_count > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_rank_guard_shadow_mixed"
    elif boost_count > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_rank_guard_shadow_boost_ready"
    elif suppress_count > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_rank_guard_shadow_suppress_ready"
    elif no_op_count > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_rank_guard_shadow_no_confirmed_or_falsified"
    elif int(posterior.get("protected_probe_outcome_reviewed_count", 0) or 0) > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_rank_guard_shadow_no_review_rows"
    elif int(posterior.get("packet_count", 0) or 0) > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_rank_guard_shadow_no_outcome_posterior_ready"
    else:
        promotion_state = "tile_effect_model_repair_protected_probe_rank_guard_shadow_no_packets"

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RANK_GUARD_SHADOW_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(posterior.get("packet_count", 0) or 0),
            "protected_probe_outcome_reviewed_input_count": int(
                posterior.get("protected_probe_outcome_reviewed_count", 0) or 0
            ),
            "protected_probe_outcome_confirmed_input_count": int(
                posterior.get("protected_probe_outcome_confirmed_count", 0) or 0
            ),
            "protected_probe_outcome_falsified_input_count": int(
                posterior.get("protected_probe_outcome_falsified_count", 0) or 0
            ),
            "rank_guard_shadow_reviewed_count": int(reviewed_count),
            "rank_guard_shadow_ready_count": int(boost_count + suppress_count),
            "rank_guard_shadow_boost_count": int(boost_count),
            "rank_guard_shadow_suppress_count": int(suppress_count),
            "rank_guard_shadow_no_op_count": int(no_op_count),
            "outcome_posterior_promotion_state": str(
                posterior.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "rows": rows[:sample_limit],
            "samples": samples,
            "shadow_rank_signal_enabled": bool(boost_count + suppress_count > 0),
            "rank_guard_runtime_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
            "planner_connection_enabled": False,
        }
    )


def summarize_tile_effect_model_repair_protected_probe_rank_guard_runtime_guard(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    probe_outcomes: Iterable[Mapping[str, Any]] | None = None,
    normal_step_max: int = 1,
    min_anchor_observations: int = 2,
    min_score_linked_observations: int = 1,
    min_repeat_score_linked_observations: int = 2,
    require_score_linked_purity: bool = True,
    min_action_dominance: float = 1.0,
    min_return_anchor_prior_non_reset_observations: int = 1,
    allow_return_anchor_prior_occupancy_trace: bool = False,
    require_hazard_source_evidence: bool = True,
    require_hazard_source_distinct_from_return_anchor: bool = True,
    min_scoreless_reset_same_identity_observations: int = 0,
    max_score_linked_reset_packet_move_count: int = 4,
    min_hazard_source_support: int = 1,
    min_hazard_source_dominance: float = 1.0,
    min_write_guard_hazard_source_support: int = 2,
    min_write_guard_hazard_source_dominance: float = 1.0,
    min_write_audit_scoreless_control_observations: int = 0,
    min_runtime_probe_scoreless_control_observations: int = 0,
    min_execution_gate_scoreless_control_observations: int = 0,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Materialize shadow rank-guard signals as runtime guard events.

    This is still not live ranking. It checks that a confirmed/falsified
    protected-probe posterior can cross the next safety boundary without
    enabling candidate reordering, action generation, or model writes.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    shadow = summarize_tile_effect_model_repair_protected_probe_rank_guard_shadow(
        packet_rows,
        probe_outcomes=probe_outcomes,
        normal_step_max=normal_step_max,
        min_anchor_observations=min_anchor_observations,
        min_score_linked_observations=min_score_linked_observations,
        min_repeat_score_linked_observations=min_repeat_score_linked_observations,
        require_score_linked_purity=require_score_linked_purity,
        min_action_dominance=min_action_dominance,
        min_return_anchor_prior_non_reset_observations=(
            min_return_anchor_prior_non_reset_observations
        ),
        allow_return_anchor_prior_occupancy_trace=(
            allow_return_anchor_prior_occupancy_trace
        ),
        require_hazard_source_evidence=require_hazard_source_evidence,
        require_hazard_source_distinct_from_return_anchor=(
            require_hazard_source_distinct_from_return_anchor
        ),
        min_scoreless_reset_same_identity_observations=(
            min_scoreless_reset_same_identity_observations
        ),
        max_score_linked_reset_packet_move_count=(
            max_score_linked_reset_packet_move_count
        ),
        min_hazard_source_support=min_hazard_source_support,
        min_hazard_source_dominance=min_hazard_source_dominance,
        min_write_guard_hazard_source_support=min_write_guard_hazard_source_support,
        min_write_guard_hazard_source_dominance=(
            min_write_guard_hazard_source_dominance
        ),
        min_write_audit_scoreless_control_observations=(
            min_write_audit_scoreless_control_observations
        ),
        min_runtime_probe_scoreless_control_observations=(
            min_runtime_probe_scoreless_control_observations
        ),
        min_execution_gate_scoreless_control_observations=(
            min_execution_gate_scoreless_control_observations
        ),
        sample_limit=100000,
    )

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    by_game: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    reviewed_count = 0
    admitted_count = 0
    blocked_count = 0
    boost_admitted_count = 0
    suppress_admitted_count = 0

    for index, shadow_row in enumerate(list(shadow.get("rows") or [])):
        if not isinstance(shadow_row, Mapping):
            continue
        if not bool(shadow_row.get("shadow_rank_signal_ready", False)):
            continue

        reviewed_count += 1
        game_id = str(shadow_row.get("game_id", "") or "unknown")
        by_game.setdefault(game_id, Counter())
        by_game[game_id]["rank_guard_runtime_guard_reviewed_count"] += 1

        state = str(
            shadow_row.get("protected_probe_outcome_posterior_state", "") or ""
        )
        decision = (
            shadow_row.get("rank_guard_shadow_decision")
            if isinstance(shadow_row.get("rank_guard_shadow_decision"), Mapping)
            else {}
        )
        rollback_key = str(shadow_row.get("rollback_key", "") or "")
        reasons: list[str] = []
        if state not in {"confirmed", "falsified"}:
            reasons.append("posterior_state_not_confirmed_or_falsified")
        if not rollback_key:
            reasons.append("missing_rollback_key")
        if not bool(shadow_row.get("protected_probe_execution_attempted", False)):
            reasons.append("protected_probe_not_attempted")
        if bool(shadow_row.get("rank_guard_runtime_enabled", False)):
            reasons.append("shadow_row_already_enabled_runtime_guard")
        if bool(decision.get("rank_guard_runtime_enabled", False)):
            reasons.append("shadow_decision_already_enabled_runtime_guard")
        if bool(shadow_row.get("ranking_enabled", False)) or bool(
            decision.get("ranking_enabled", False)
        ):
            reasons.append("ranking_already_enabled")
        if bool(shadow_row.get("action_generation_allowed", False)) or bool(
            decision.get("action_generation_allowed", False)
        ):
            reasons.append("action_generation_already_allowed")
        if bool(shadow_row.get("model_repair_write_enabled", False)) or bool(
            decision.get("model_repair_write_enabled", False)
        ):
            reasons.append("model_repair_write_already_enabled")
        if bool(shadow_row.get("planner_connection_enabled", False)):
            reasons.append("planner_connection_already_enabled")

        guard_effect = (
            "boost_existing_matching_candidates"
            if state == "confirmed"
            else "suppress_matching_candidates"
        )
        event = {
            "schema_version": (
                "eca_tile_effect_model_repair_rank_guard_runtime_guard_event.v1"
            ),
            "event_kind": "protected_probe_rank_guard_runtime_guard",
            "game_id": game_id,
            "posterior_key": str(shadow_row.get("posterior_key", "") or ""),
            "identity_key": str(shadow_row.get("identity_key", "") or ""),
            "rollback_key": rollback_key,
            "shadow_row_index": int(index),
            "protected_probe_outcome_posterior_state": state,
            "rank_guard_shadow_state": str(
                shadow_row.get("rank_guard_shadow_state", "") or ""
            ),
            "rank_guard_effect": guard_effect,
            "allowed_connection": "rank_guard_shadow_to_runtime_guard_event",
            "shadow_rank_guard_matched": True,
            "runtime_rank_guard_ready": not reasons,
            "rank_guard_runtime_guard_admitted": not reasons,
            "candidate_order_mutation_allowed": False,
            "live_ranking_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
            "planner_connection_enabled": False,
        }

        if reasons:
            blocked_count += 1
            reason = reasons[0]
            reason_counts[reason] += 1
            by_game[game_id]["rank_guard_runtime_guard_blocked_count"] += 1
            by_game[game_id][f"blocked_reason:{reason}"] += 1
            event["blocked_reason"] = reason
            event["blocked_reasons"] = reasons
        else:
            admitted_count += 1
            reason_counts[
                "tile_effect_model_repair_rank_guard_runtime_guard_admitted"
            ] += 1
            by_game[game_id]["rank_guard_runtime_guard_admitted_count"] += 1
            if state == "confirmed":
                boost_admitted_count += 1
                by_game[game_id]["rank_guard_runtime_guard_boost_admitted_count"] += 1
            else:
                suppress_admitted_count += 1
                by_game[game_id][
                    "rank_guard_runtime_guard_suppress_admitted_count"
                ] += 1

        rows.append(event)
        if len(samples) < sample_limit:
            samples.append(event)

    if boost_admitted_count > 0 and suppress_admitted_count > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_rank_guard_runtime_guard_mixed"
    elif boost_admitted_count > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_rank_guard_runtime_guard_boost_ready"
    elif suppress_admitted_count > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_rank_guard_runtime_guard_suppress_ready"
    elif blocked_count > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_rank_guard_runtime_guard_all_blocked"
    elif int(shadow.get("rank_guard_shadow_ready_count", 0) or 0) > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_rank_guard_runtime_guard_no_review_rows"
    elif int(shadow.get("packet_count", 0) or 0) > 0:
        promotion_state = "tile_effect_model_repair_protected_probe_rank_guard_runtime_guard_no_shadow_ready"
    else:
        promotion_state = "tile_effect_model_repair_protected_probe_rank_guard_runtime_guard_no_packets"

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RANK_GUARD_RUNTIME_GUARD_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "shadow_only": True,
            "packet_count": int(shadow.get("packet_count", 0) or 0),
            "rank_guard_shadow_ready_input_count": int(
                shadow.get("rank_guard_shadow_ready_count", 0) or 0
            ),
            "rank_guard_shadow_boost_input_count": int(
                shadow.get("rank_guard_shadow_boost_count", 0) or 0
            ),
            "rank_guard_shadow_suppress_input_count": int(
                shadow.get("rank_guard_shadow_suppress_count", 0) or 0
            ),
            "rank_guard_runtime_guard_reviewed_count": int(reviewed_count),
            "rank_guard_runtime_guard_admitted_count": int(admitted_count),
            "rank_guard_runtime_guard_blocked_count": int(blocked_count),
            "rank_guard_runtime_guard_boost_admitted_count": int(
                boost_admitted_count
            ),
            "rank_guard_runtime_guard_suppress_admitted_count": int(
                suppress_admitted_count
            ),
            "rank_guard_shadow_promotion_state": str(
                shadow.get("promotion_state", "") or ""
            ),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "rows": rows[:sample_limit],
            "samples": samples,
            "rank_guard_runtime_guard_event_materialized": bool(admitted_count > 0),
            "rank_guard_runtime_guard_enabled": bool(admitted_count > 0),
            "rank_guard_runtime_enabled": False,
            "runtime_execution_enabled": False,
            "runtime_execution_observed": False,
            "live_ranking_enabled": False,
            "candidate_order_mutation_allowed": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
            "planner_connection_enabled": False,
            "next_runtime_contract": {
                "required_before_execution": (
                    "runtime guard events must pass a separate default-off "
                    "live rank-guard phase before any candidate reordering"
                ),
                "allowed_connection": "rank_guard_runtime_guard_event_only",
                "action_generation_allowed": False,
            },
        }
    )


def summarize_tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime(
    events: Iterable[Mapping[str, Any]] | Any,
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize default-off live rank-guard events.

    The live stage may reorder or select from existing candidates only. It
    must not synthesize new actions, write a repaired model, or consume a
    post-action event from the same step as pre-action ranking evidence.
    """

    event_rows = [dict(event) for event in events if isinstance(event, Mapping)]
    reviewed_count = 0
    candidate_reviewed_count = 0
    candidate_rank_signal_count = 0
    candidate_order_mutated_count = 0
    action_selection_changed_count = 0
    boost_candidate_count = 0
    suppress_candidate_count = 0
    blocked_count = 0
    ready_count = 0
    stale_blocked_count = 0
    unsafe_blocked_count = 0
    by_game: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    for event in event_rows:
        if str(event.get("event_kind", "") or "") != (
            "protected_probe_live_rank_guard_focused_runtime"
        ):
            continue
        reviewed_count += 1
        game_id = str(event.get("game_id", "") or "unknown")
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["live_rank_guard_reviewed_count"] += 1

        candidate_count = _safe_int(event.get("candidate_reviewed_count"))
        rank_signal_count = _safe_int(event.get("candidate_rank_signal_count"))
        boost_count = _safe_int(event.get("boost_candidate_count"))
        suppress_count = _safe_int(event.get("suppress_candidate_count"))
        order_mutated = bool(event.get("candidate_order_mutated"))
        action_changed = bool(event.get("action_selection_changed"))
        admitted = bool(event.get("live_rank_guard_admitted"))
        stale = bool(event.get("same_step_post_action_evidence_blocked"))
        unsafe = (
            bool(event.get("action_generation_allowed"))
            or bool(event.get("model_repair_write_enabled"))
            or bool(event.get("generated_new_candidate"))
        )
        reasons = [
            str(reason)
            for reason in list(event.get("blocked_reasons") or [])
            if str(reason)
        ]
        reason = str(event.get("blocked_reason", "") or "")
        if reason and reason not in reasons:
            reasons.insert(0, reason)

        candidate_reviewed_count += candidate_count
        candidate_rank_signal_count += rank_signal_count
        boost_candidate_count += boost_count
        suppress_candidate_count += suppress_count
        if order_mutated:
            candidate_order_mutated_count += 1
            game_counter["candidate_order_mutated_count"] += 1
        if action_changed:
            action_selection_changed_count += 1
            game_counter["action_selection_changed_count"] += 1
        if stale:
            stale_blocked_count += 1
            game_counter["same_step_post_action_evidence_blocked_count"] += 1
        if unsafe:
            unsafe_blocked_count += 1
            game_counter["unsafe_event_blocked_count"] += 1

        if admitted and not unsafe:
            ready_count += 1
            game_counter["live_rank_guard_admitted_count"] += 1
        else:
            blocked_count += 1
            game_counter["live_rank_guard_blocked_count"] += 1
            if not reasons:
                reasons = ["live_rank_guard_not_admitted"]
            for item in reasons:
                reason_counts[item] += 1
                game_counter[f"blocked_reason:{item}"] += 1

        row = {
            "game_id": game_id,
            "step": _safe_int(event.get("step")),
            "state_hash": str(event.get("state_hash", "") or ""),
            "live_rank_guard_admitted": bool(admitted and not unsafe),
            "candidate_reviewed_count": candidate_count,
            "candidate_rank_signal_count": rank_signal_count,
            "boost_candidate_count": boost_count,
            "suppress_candidate_count": suppress_count,
            "candidate_order_mutated": order_mutated,
            "action_selection_changed": action_changed,
            "same_step_post_action_evidence_blocked": stale,
            "blocked_reasons": reasons,
            "source_runtime_guard_count": _safe_int(
                event.get("source_runtime_guard_count")
            ),
            "candidate_order_mutation_allowed": bool(
                event.get("candidate_order_mutation_allowed")
            ),
            "action_generation_allowed": bool(event.get("action_generation_allowed")),
            "model_repair_write_enabled": bool(
                event.get("model_repair_write_enabled")
            ),
            "generated_new_candidate": bool(event.get("generated_new_candidate")),
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if action_selection_changed_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime_action_changed"
        )
    elif candidate_order_mutated_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime_ranked"
        )
    elif ready_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime_ready_no_rank_change"
        )
    elif blocked_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime_all_blocked"
        )
    elif event_rows:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime_no_review_rows"
        )
    else:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime_no_events"
        )

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_FOCUSED_RUNTIME_SCHEMA_VERSION
            ),
            "diagnostic_only": False,
            "default_off": True,
            "live_rank_guard_reviewed_count": int(reviewed_count),
            "live_rank_guard_admitted_count": int(ready_count),
            "live_rank_guard_blocked_count": int(blocked_count),
            "candidate_reviewed_count": int(candidate_reviewed_count),
            "candidate_rank_signal_count": int(candidate_rank_signal_count),
            "candidate_order_mutated_count": int(candidate_order_mutated_count),
            "action_selection_changed_count": int(action_selection_changed_count),
            "boost_candidate_count": int(boost_candidate_count),
            "suppress_candidate_count": int(suppress_candidate_count),
            "same_step_post_action_evidence_blocked_count": int(stale_blocked_count),
            "unsafe_event_blocked_count": int(unsafe_blocked_count),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "rows": rows[:sample_limit],
            "samples": samples,
            "live_ranking_enabled": bool(ready_count > 0),
            "candidate_order_mutation_allowed": bool(
                candidate_order_mutated_count > 0
            ),
            "action_selection_changed": bool(action_selection_changed_count > 0),
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
            "generated_new_candidate": False,
            "planner_connection_enabled": bool(ready_count > 0),
        }
    )


def summarize_tile_effect_model_repair_protected_probe_live_rank_guard_action_selection_guard(
    events: Iterable[Mapping[str, Any]] | Any,
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize the guard that decides whether live ranking may change action.

    Phase733 proved that broad executed-action mutation can regress all25. This
    summary therefore treats every requested action change as unsafe unless the
    runner's explicit action-selection guard admitted it.
    """

    event_rows = [dict(event) for event in events if isinstance(event, Mapping)]
    reviewed_count = 0
    requested_count = 0
    admitted_count = 0
    blocked_count = 0
    changed_count = 0
    blocked_click_to_non_click_count = 0
    blocked_commit_operator_count = 0
    blocked_disabled_count = 0
    reason_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    for event in event_rows:
        if str(event.get("event_kind", "") or "") != (
            "protected_probe_live_rank_guard_focused_runtime"
        ):
            continue
        if not bool(event.get("action_selection_guard_checked")):
            continue
        reviewed_count += 1
        game_id = str(event.get("game_id", "") or "unknown")
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["action_selection_guard_reviewed_count"] += 1

        requested = bool(event.get("action_selection_change_requested"))
        admitted = bool(event.get("action_selection_guard_admitted"))
        changed = bool(event.get("action_selection_changed"))
        reasons = [
            str(reason)
            for reason in list(event.get("action_selection_guard_blocked_reasons") or [])
            if str(reason)
        ]
        reason = str(event.get("action_selection_guard_blocked_reason", "") or "")
        if reason and reason not in reasons:
            reasons.insert(0, reason)

        if requested:
            requested_count += 1
            game_counter["action_selection_change_requested_count"] += 1
        if admitted:
            admitted_count += 1
            game_counter["action_selection_guard_admitted_count"] += 1
        else:
            blocked_count += 1
            game_counter["action_selection_guard_blocked_count"] += 1
            if not reasons:
                reasons = ["action_selection_guard_not_admitted"]
            for item in reasons:
                reason_counts[item] += 1
                game_counter[f"blocked_reason:{item}"] += 1
            if "click_to_non_click_preemption" in reasons:
                blocked_click_to_non_click_count += 1
            if "commit_operator_preemption" in reasons:
                blocked_commit_operator_count += 1
            if "action_selection_disabled" in reasons:
                blocked_disabled_count += 1
        if changed:
            changed_count += 1
            game_counter["action_selection_changed_count"] += 1

        row = {
            "game_id": game_id,
            "step": _safe_int(event.get("step")),
            "state_hash": str(event.get("state_hash", "") or ""),
            "action_selection_change_requested": requested,
            "action_selection_guard_admitted": admitted,
            "action_selection_changed": changed,
            "selected_action_before": str(event.get("selected_action_before", "") or ""),
            "selected_action_after": str(event.get("selected_action_after", "") or ""),
            "action_selection_guard_blocked_reasons": reasons,
            "current_primitive": str(event.get("current_primitive", "") or ""),
            "current_primary_operator": str(
                event.get("current_primary_operator", "") or ""
            ),
            "action_generation_allowed": bool(event.get("action_generation_allowed")),
            "model_repair_write_enabled": bool(
                event.get("model_repair_write_enabled")
            ),
            "generated_new_candidate": bool(event.get("generated_new_candidate")),
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if changed_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_action_selection_guard_action_changed"
        )
    elif admitted_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_action_selection_guard_admitted_no_change"
        )
    elif blocked_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_action_selection_guard_blocked_unsafe"
        )
    elif event_rows:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_action_selection_guard_no_review_rows"
        )
    else:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_action_selection_guard_no_events"
        )

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_ACTION_SELECTION_GUARD_SCHEMA_VERSION
            ),
            "diagnostic_only": False,
            "default_off": True,
            "action_selection_guard_reviewed_count": int(reviewed_count),
            "action_selection_change_requested_count": int(requested_count),
            "action_selection_guard_admitted_count": int(admitted_count),
            "action_selection_guard_blocked_count": int(blocked_count),
            "action_selection_changed_count": int(changed_count),
            "blocked_click_to_non_click_count": int(blocked_click_to_non_click_count),
            "blocked_commit_operator_count": int(blocked_commit_operator_count),
            "blocked_disabled_count": int(blocked_disabled_count),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "rows": rows[:sample_limit],
            "samples": samples,
            "action_selection_guard_enabled": bool(reviewed_count > 0),
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
            "generated_new_candidate": False,
        }
    )


def summarize_tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_admission(
    events: Iterable[Mapping[str, Any]] | Any,
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize admitted live-rank action selections.

    This is the positive half of the Phase734 guard: an admission is counted
    only when the runner changed to an already-ranked existing candidate,
    stayed within the same action family, and kept action generation/model
    repair writes disabled.
    """

    event_rows = [dict(event) for event in events if isinstance(event, Mapping)]
    reviewed_count = 0
    requested_count = 0
    guard_admitted_count = 0
    changed_count = 0
    positive_count = 0
    same_action_family_count = 0
    site_changed_count = 0
    same_site_count = 0
    unsafe_admission_count = 0
    generated_candidate_admission_count = 0
    blocked_reason_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    for event in event_rows:
        if str(event.get("event_kind", "") or "") != (
            "protected_probe_live_rank_guard_focused_runtime"
        ):
            continue
        if not bool(event.get("action_selection_guard_checked")):
            continue
        reviewed_count += 1
        game_id = str(event.get("game_id", "") or "unknown")
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["positive_action_selection_reviewed_count"] += 1

        requested = bool(event.get("action_selection_change_requested"))
        guard_admitted = bool(event.get("action_selection_guard_admitted"))
        changed = bool(event.get("action_selection_changed"))
        selected_existing = bool(event.get("selected_existing_candidate_only"))
        generated = bool(event.get("generated_new_candidate"))
        unsafe = (
            bool(event.get("action_generation_allowed"))
            or bool(event.get("model_repair_write_enabled"))
            or generated
        )
        before_action = str(event.get("selected_action_before", "") or "")
        after_action = str(event.get("selected_action_after", "") or "")
        same_action_family = bool(before_action and before_action == after_action)
        before_site = _xy_list(event.get("selected_site_before"))
        after_site = _xy_list(
            event.get("selected_site_after")
            if event.get("selected_site_after") is not None
            else event.get("selected_candidate_site")
        )
        site_changed = bool(before_site and after_site and before_site != after_site)
        same_site = bool(before_site and after_site and before_site == after_site)

        if requested:
            requested_count += 1
            game_counter["action_selection_change_requested_count"] += 1
        if guard_admitted:
            guard_admitted_count += 1
            game_counter["action_selection_guard_admitted_count"] += 1
        if changed:
            changed_count += 1
            game_counter["action_selection_changed_count"] += 1
        if guard_admitted and unsafe:
            unsafe_admission_count += 1
            game_counter["unsafe_admission_count"] += 1
        if guard_admitted and generated:
            generated_candidate_admission_count += 1
            game_counter["generated_candidate_admission_count"] += 1

        reasons = [
            str(reason)
            for reason in list(event.get("action_selection_guard_blocked_reasons") or [])
            if str(reason)
        ]
        if not guard_admitted:
            for reason in reasons or ["action_selection_guard_not_admitted"]:
                blocked_reason_counts[reason] += 1
                game_counter[f"blocked_reason:{reason}"] += 1

        positive = bool(
            requested
            and guard_admitted
            and changed
            and selected_existing
            and same_action_family
            and not unsafe
        )
        if positive:
            positive_count += 1
            game_counter["positive_action_selection_admission_count"] += 1
            if same_action_family:
                same_action_family_count += 1
                game_counter["same_action_family_count"] += 1
            if site_changed:
                site_changed_count += 1
                game_counter["site_changed_count"] += 1
            if same_site:
                same_site_count += 1
                game_counter["same_site_count"] += 1

        row = {
            "game_id": game_id,
            "step": _safe_int(event.get("step")),
            "state_hash": str(event.get("state_hash", "") or ""),
            "action_selection_change_requested": requested,
            "action_selection_guard_admitted": guard_admitted,
            "action_selection_changed": changed,
            "positive_action_selection_admitted": positive,
            "selected_existing_candidate_only": selected_existing,
            "selected_action_before": before_action,
            "selected_action_after": after_action,
            "same_action_family": same_action_family,
            "selected_site_before": before_site,
            "selected_site_after": after_site,
            "site_changed": site_changed,
            "same_site": same_site,
            "selected_candidate_rank_effect": str(
                event.get("selected_candidate_rank_effect", "") or ""
            ),
            "selected_candidate_original_index": _safe_int(
                event.get("selected_candidate_original_index")
            ),
            "current_primitive": str(event.get("current_primitive", "") or ""),
            "current_primary_operator": str(
                event.get("current_primary_operator", "") or ""
            ),
            "action_generation_allowed": bool(event.get("action_generation_allowed")),
            "model_repair_write_enabled": bool(
                event.get("model_repair_write_enabled")
            ),
            "generated_new_candidate": generated,
            "blocked_reasons": reasons,
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if positive_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_admission_admitted"
        )
    elif guard_admitted_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_admission_guard_admitted_without_positive_change"
        )
    elif reviewed_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_admission_no_positive_admission"
        )
    elif event_rows:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_admission_no_review_rows"
        )
    else:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_admission_no_events"
        )

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_POSITIVE_ACTION_SELECTION_ADMISSION_SCHEMA_VERSION
            ),
            "diagnostic_only": False,
            "default_off": True,
            "positive_action_selection_reviewed_count": int(reviewed_count),
            "action_selection_change_requested_count": int(requested_count),
            "action_selection_guard_admitted_count": int(guard_admitted_count),
            "action_selection_changed_count": int(changed_count),
            "positive_action_selection_admission_count": int(positive_count),
            "same_action_family_admission_count": int(same_action_family_count),
            "site_changed_admission_count": int(site_changed_count),
            "same_site_admission_count": int(same_site_count),
            "unsafe_admission_count": int(unsafe_admission_count),
            "generated_candidate_admission_count": int(
                generated_candidate_admission_count
            ),
            "promotion_state": promotion_state,
            "blocked_reason_counts": {
                key: int(value)
                for key, value in sorted(blocked_reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "rows": rows[:sample_limit],
            "samples": samples,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
            "generated_new_candidate": False,
            "planner_connection_enabled": bool(positive_count > 0),
            "action_selection_positive_admission_enabled": bool(positive_count > 0),
        }
    )


def summarize_tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution(
    events: Iterable[Mapping[str, Any]] | Any,
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize score attribution for admitted live-rank action selections."""

    event_rows = [dict(event) for event in events if isinstance(event, Mapping)]
    reviewed_count = 0
    score_attributed_count = 0
    immediate_score_count = 0
    delayed_score_count = 0
    delayed_control_margin_count = 0
    delayed_control_matched_count = 0
    no_score_association_count = 0
    immediate_score_delta_sum = 0
    delayed_score_delta_sum = 0
    negative_control_max_score_delta_sum = 0
    join_margin_sum = 0
    by_game: dict[str, Counter[str]] = {}
    state_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    for event in event_rows:
        positive = bool(event.get("positive_action_selection_admitted"))
        if not positive:
            continue
        reviewed_count += 1
        game_id = str(event.get("game_id", "") or "unknown")
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["score_attribution_reviewed_count"] += 1

        immediate_delta = _safe_int(event.get("immediate_score_delta"))
        delayed_delta = _safe_int(event.get("delayed_score_delta"))
        control_delta = _safe_int(event.get("negative_control_max_score_delta"))
        join_margin = _safe_int(event.get("join_margin"))
        delay_steps = _safe_int(event.get("delay_steps"))

        immediate_score_delta_sum += immediate_delta
        delayed_score_delta_sum += delayed_delta
        negative_control_max_score_delta_sum += control_delta
        join_margin_sum += join_margin
        game_counter["immediate_score_delta_sum"] += immediate_delta
        game_counter["delayed_score_delta_sum"] += delayed_delta
        game_counter["negative_control_max_score_delta_sum"] += control_delta
        game_counter["join_margin_sum"] += join_margin

        if immediate_delta > 0:
            attribution_state = "immediate_score_attributed"
            score_attributed = True
            immediate_score_count += 1
            game_counter["immediate_score_attributed_count"] += 1
        elif delayed_delta > 0 and join_margin > 0:
            attribution_state = "delayed_score_attributed_with_control_margin"
            score_attributed = True
            delayed_score_count += 1
            delayed_control_margin_count += 1
            game_counter["delayed_score_attributed_count"] += 1
            game_counter["delayed_control_margin_count"] += 1
        elif delayed_delta > 0:
            attribution_state = "delayed_score_associated_control_matched"
            score_attributed = False
            delayed_score_count += 1
            delayed_control_matched_count += 1
            game_counter["delayed_score_associated_count"] += 1
            game_counter["delayed_control_matched_count"] += 1
        else:
            attribution_state = "no_score_association"
            score_attributed = False
            no_score_association_count += 1
            game_counter["no_score_association_count"] += 1

        if score_attributed:
            score_attributed_count += 1
            game_counter["score_attributed_count"] += 1
        state_counts[attribution_state] += 1
        game_counter[f"state:{attribution_state}"] += 1

        before_action = str(event.get("selected_action_before", "") or "")
        after_action = str(event.get("selected_action_after", "") or "")
        same_action_family = bool(
            event.get("same_action_family")
            or (before_action and before_action == after_action)
        )
        row = {
            "game_id": game_id,
            "variant": str(event.get("variant", "") or ""),
            "step": _safe_int(event.get("step")),
            "state_hash": str(event.get("state_hash", "") or ""),
            "positive_action_selection_admitted": True,
            "score_attributed": score_attributed,
            "attribution_state": attribution_state,
            "selected_action_before": before_action,
            "selected_action_after": after_action,
            "same_action_family": same_action_family,
            "selected_site_before": _xy_list(event.get("selected_site_before")),
            "selected_site_after": _xy_list(event.get("selected_site_after")),
            "current_primitive": str(event.get("current_primitive", "") or ""),
            "current_primary_operator": str(
                event.get("current_primary_operator", "") or ""
            ),
            "score_before": _safe_int(event.get("score_before")),
            "immediate_score_delta": immediate_delta,
            "delayed_score_delta": delayed_delta,
            "delay_steps": delay_steps,
            "shifted_delayed_score_delta": _safe_int(
                event.get("shifted_delayed_score_delta")
            ),
            "pre_action_score_delta": _safe_int(event.get("pre_action_score_delta")),
            "random_step_score_delta": _safe_int(event.get("random_step_score_delta")),
            "negative_control_max_score_delta": control_delta,
            "join_margin": join_margin,
            "action_generation_allowed": bool(event.get("action_generation_allowed")),
            "model_repair_write_enabled": bool(
                event.get("model_repair_write_enabled")
            ),
            "generated_new_candidate": bool(event.get("generated_new_candidate")),
        }
        rows.append(row)
        if len(samples) < sample_limit:
            samples.append(row)

    if score_attributed_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution_confirmed"
        )
    elif delayed_score_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution_ambiguous_delayed"
        )
    elif reviewed_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution_no_score"
        )
    elif event_rows:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution_no_positive_rows"
        )
    else:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution_no_events"
        )

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_POSITIVE_ACTION_SELECTION_SCORE_ATTRIBUTION_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "default_off": True,
            "score_attribution_reviewed_count": int(reviewed_count),
            "positive_action_selection_admission_count": int(reviewed_count),
            "score_attributed_admission_count": int(score_attributed_count),
            "immediate_score_attributed_count": int(immediate_score_count),
            "delayed_score_associated_count": int(delayed_score_count),
            "delayed_control_margin_count": int(delayed_control_margin_count),
            "delayed_control_matched_count": int(delayed_control_matched_count),
            "no_score_association_count": int(no_score_association_count),
            "immediate_score_delta_sum": int(immediate_score_delta_sum),
            "delayed_score_delta_sum": int(delayed_score_delta_sum),
            "negative_control_max_score_delta_sum": int(
                negative_control_max_score_delta_sum
            ),
            "join_margin_sum": int(join_margin_sum),
            "promotion_state": promotion_state,
            "state_counts": {
                key: int(value) for key, value in sorted(state_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "rows": rows[:sample_limit],
            "samples": samples,
            "planner_connection_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
            "generated_new_candidate": False,
            "next_required_connection": (
                "score-attributed admission dominance before positive rank promotion"
            ),
        }
    )


def summarize_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_positive_rank_promotion_guard(
    attribution: Mapping[str, Any] | Iterable[Mapping[str, Any]] | Any,
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Guard positive rank promotion using score-attributed admissions only."""

    if isinstance(attribution, Mapping):
        attr_summary = dict(attribution)
        rows_source = list(attr_summary.get("rows") or attr_summary.get("samples") or [])
    else:
        rows_source = [dict(row) for row in attribution if isinstance(row, Mapping)]
        attr_summary = (
            summarize_tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution(
                rows_source,
                sample_limit=sample_limit,
            )
        )

    reviewed_count = _safe_int(attr_summary.get("score_attribution_reviewed_count"))
    positive_admission_count = _safe_int(
        attr_summary.get("positive_action_selection_admission_count")
    )
    score_attributed_count = _safe_int(
        attr_summary.get("score_attributed_admission_count")
    )
    immediate_count = _safe_int(attr_summary.get("immediate_score_attributed_count"))
    delayed_margin_count = _safe_int(attr_summary.get("delayed_control_margin_count"))
    delayed_matched_count = _safe_int(attr_summary.get("delayed_control_matched_count"))
    no_score_count = _safe_int(attr_summary.get("no_score_association_count"))
    blocked_no_score_attribution_count = max(
        0, positive_admission_count - score_attributed_count
    )
    ready = bool(score_attributed_count > 0)
    promotion_guard_admitted_count = score_attributed_count if ready else 0
    promotion_guard_blocked_count = blocked_no_score_attribution_count

    by_game: dict[str, Counter[str]] = {}
    for game_id, stats in dict(attr_summary.get("by_game") or {}).items():
        counter = by_game.setdefault(str(game_id), Counter())
        if isinstance(stats, Mapping):
            game_score_attributed = _safe_int(stats.get("score_attributed_count"))
            game_reviewed = _safe_int(stats.get("score_attribution_reviewed_count"))
            game_delayed_matched = _safe_int(stats.get("delayed_control_matched_count"))
            game_no_score = _safe_int(stats.get("no_score_association_count"))
            counter["score_attribution_reviewed_count"] += game_reviewed
            counter["score_attributed_admission_count"] += game_score_attributed
            counter["blocked_no_score_attribution_count"] += max(
                0, game_reviewed - game_score_attributed
            )
            counter["delayed_control_matched_count"] += game_delayed_matched
            counter["no_score_association_count"] += game_no_score
            if game_score_attributed > 0:
                counter["promotion_guard_admitted_count"] += game_score_attributed
            elif game_reviewed > 0:
                counter["promotion_guard_blocked_count"] += game_reviewed

    rows: list[dict[str, Any]] = []
    for row in rows_source:
        if not isinstance(row, Mapping):
            continue
        score_attributed = bool(row.get("score_attributed"))
        guard_state = (
            "promotion_guard_admitted_score_attributed"
            if score_attributed
            else "promotion_guard_blocked_no_score_attribution"
        )
        rows.append(
            {
                "game_id": str(row.get("game_id", "") or "unknown"),
                "variant": str(row.get("variant", "") or ""),
                "step": _safe_int(row.get("step")),
                "state_hash": str(row.get("state_hash", "") or ""),
                "guard_state": guard_state,
                "score_attributed": score_attributed,
                "attribution_state": str(row.get("attribution_state", "") or ""),
                "selected_action_before": str(
                    row.get("selected_action_before", "") or ""
                ),
                "selected_action_after": str(row.get("selected_action_after", "") or ""),
                "selected_site_before": _xy_list(row.get("selected_site_before")),
                "selected_site_after": _xy_list(row.get("selected_site_after")),
                "immediate_score_delta": _safe_int(row.get("immediate_score_delta")),
                "delayed_score_delta": _safe_int(row.get("delayed_score_delta")),
                "negative_control_max_score_delta": _safe_int(
                    row.get("negative_control_max_score_delta")
                ),
                "join_margin": _safe_int(row.get("join_margin")),
            }
        )

    if promotion_guard_admitted_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_positive_rank_promotion_guard_ready"
        )
    elif reviewed_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_positive_rank_promotion_guard_blocked_no_score_attribution"
        )
    elif positive_admission_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_positive_rank_promotion_guard_no_review_rows"
        )
    else:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_positive_rank_promotion_guard_no_positive_admissions"
        )

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_SCORE_ATTRIBUTED_POSITIVE_RANK_PROMOTION_GUARD_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "default_off": True,
            "score_attribution_reviewed_count": int(reviewed_count),
            "positive_action_selection_admission_count": int(
                positive_admission_count
            ),
            "score_attributed_admission_count": int(score_attributed_count),
            "immediate_score_attributed_count": int(immediate_count),
            "delayed_control_margin_count": int(delayed_margin_count),
            "delayed_control_matched_count": int(delayed_matched_count),
            "no_score_association_count": int(no_score_count),
            "promotion_guard_admitted_count": int(promotion_guard_admitted_count),
            "promotion_guard_blocked_count": int(promotion_guard_blocked_count),
            "blocked_no_score_attribution_count": int(
                blocked_no_score_attribution_count
            ),
            "promotion_state": promotion_state,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "rows": rows[:sample_limit],
            "samples": rows[:sample_limit],
            "ready_for_positive_rank_promotion": bool(ready),
            "planner_connection_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
            "generated_new_candidate": False,
            "next_required_connection": (
                "score-attributed promotion may be considered only after focused and all25 non-regression"
            ),
        }
    )


def summarize_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery(
    attribution: Mapping[str, Any] | Iterable[Mapping[str, Any]] | Any,
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Discover score-attributed live-rank candidates without promotion.

    This is the candidate-discovery stage after the promotion guard. It records
    only candidates whose admitted action-selection evidence is already
    score-attributed by the Phase736 attribution rule.
    """

    if isinstance(attribution, Mapping):
        attr_summary = dict(attribution)
        rows_source = list(attr_summary.get("rows") or attr_summary.get("samples") or [])
    else:
        event_rows = [dict(row) for row in attribution if isinstance(row, Mapping)]
        attr_summary = (
            summarize_tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution(
                event_rows,
                sample_limit=max(sample_limit, 32),
            )
        )
        rows_source = list(attr_summary.get("rows") or attr_summary.get("samples") or [])

    reviewed_count = _safe_int(attr_summary.get("score_attribution_reviewed_count"))
    positive_admission_count = _safe_int(
        attr_summary.get("positive_action_selection_admission_count")
    )
    score_attributed_count = _safe_int(
        attr_summary.get("score_attributed_admission_count")
    )
    immediate_count = _safe_int(attr_summary.get("immediate_score_attributed_count"))
    delayed_margin_count = _safe_int(attr_summary.get("delayed_control_margin_count"))
    delayed_matched_count = _safe_int(attr_summary.get("delayed_control_matched_count"))
    no_score_count = _safe_int(attr_summary.get("no_score_association_count"))
    rejected_control_matched_count = 0
    rejected_no_score_count = 0
    rejected_other_count = 0
    score_bearing_delta_sum = 0
    join_margin_sum = 0
    reason_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    discovered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for row in rows_source:
        if not isinstance(row, Mapping):
            continue
        game_id = str(row.get("game_id", "") or "unknown")
        counter = by_game.setdefault(game_id, Counter())
        counter["candidate_reviewed_count"] += 1
        score_attributed = bool(row.get("score_attributed"))
        attribution_state = str(row.get("attribution_state", "") or "")
        immediate_delta = _safe_int(row.get("immediate_score_delta"))
        delayed_delta = _safe_int(row.get("delayed_score_delta"))
        control_delta = _safe_int(row.get("negative_control_max_score_delta"))
        join_margin = _safe_int(row.get("join_margin"))
        evidence_kind = ""
        if score_attributed and immediate_delta > 0:
            evidence_kind = "immediate_score_attributed_candidate"
        elif score_attributed and delayed_delta > 0 and join_margin > 0:
            evidence_kind = "delayed_control_margin_score_attributed_candidate"

        if evidence_kind:
            reason_counts["score_attributed_candidate_discovered"] += 1
            counter["score_attributed_candidate_count"] += 1
            if evidence_kind == "immediate_score_attributed_candidate":
                counter["immediate_score_candidate_count"] += 1
            else:
                counter["delayed_control_margin_candidate_count"] += 1
            score_bearing_delta_sum += max(immediate_delta, delayed_delta)
            join_margin_sum += join_margin
            if len(discovered) < sample_limit:
                discovered.append(
                    {
                        "candidate_key": (
                            f"{game_id}:{row.get('state_hash', '')}:"
                            f"{row.get('step', '')}:"
                            f"{row.get('selected_action_after', '')}:"
                            f"{_xy_list(row.get('selected_site_after'))}"
                        ),
                        "game_id": game_id,
                        "variant": str(row.get("variant", "") or ""),
                        "step": _safe_int(row.get("step")),
                        "state_hash": str(row.get("state_hash", "") or ""),
                        "discovery_state": "score_attributed_candidate_discovered",
                        "evidence_kind": evidence_kind,
                        "attribution_state": attribution_state,
                        "selected_action_before": str(
                            row.get("selected_action_before", "") or ""
                        ),
                        "selected_action_after": str(
                            row.get("selected_action_after", "") or ""
                        ),
                        "selected_site_before": _xy_list(
                            row.get("selected_site_before")
                        ),
                        "selected_site_after": _xy_list(row.get("selected_site_after")),
                        "score_bearing_delta": int(max(immediate_delta, delayed_delta)),
                        "immediate_score_delta": int(immediate_delta),
                        "delayed_score_delta": int(delayed_delta),
                        "negative_control_max_score_delta": int(control_delta),
                        "join_margin": int(join_margin),
                        "state_hash_guard_ready": bool(row.get("state_hash")),
                        "existing_candidate_only": True,
                        "same_action_family_required": True,
                        "runtime_execution_enabled": False,
                        "planner_connection_enabled": False,
                        "ranking_enabled": False,
                        "action_generation_allowed": False,
                    }
                )
            continue

        if attribution_state == "delayed_score_associated_control_matched":
            reason = "delayed_score_control_matched"
            rejected_control_matched_count += 1
        elif attribution_state == "no_score_association":
            reason = "no_score_association"
            rejected_no_score_count += 1
        else:
            reason = "missing_score_attribution"
            rejected_other_count += 1
        reason_counts[reason] += 1
        counter[f"rejected_reason:{reason}"] += 1
        if len(rejected) < sample_limit:
            rejected.append(
                {
                    "game_id": game_id,
                    "variant": str(row.get("variant", "") or ""),
                    "step": _safe_int(row.get("step")),
                    "state_hash": str(row.get("state_hash", "") or ""),
                    "blocked_reason": reason,
                    "attribution_state": attribution_state,
                    "selected_action_before": str(
                        row.get("selected_action_before", "") or ""
                    ),
                    "selected_action_after": str(
                        row.get("selected_action_after", "") or ""
                    ),
                    "selected_site_before": _xy_list(row.get("selected_site_before")),
                    "selected_site_after": _xy_list(row.get("selected_site_after")),
                    "immediate_score_delta": int(immediate_delta),
                    "delayed_score_delta": int(delayed_delta),
                    "negative_control_max_score_delta": int(control_delta),
                    "join_margin": int(join_margin),
                }
            )

    discovered_count = int(reason_counts.get("score_attributed_candidate_discovered", 0))
    rejected_count = (
        rejected_control_matched_count + rejected_no_score_count + rejected_other_count
    )
    if discovered_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery_supported"
        )
    elif rejected_control_matched_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery_control_matched"
        )
    elif reviewed_count > 0 or rejected_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery_not_found"
        )
    else:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery_no_evidence"
        )

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_SCORE_ATTRIBUTED_CANDIDATE_DISCOVERY_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "default_off": True,
            "score_attribution_reviewed_count": int(reviewed_count),
            "positive_action_selection_admission_count": int(
                positive_admission_count
            ),
            "score_attributed_admission_count": int(score_attributed_count),
            "candidate_reviewed_count": int(len(rows_source)),
            "discovered_score_attributed_candidate_count": int(discovered_count),
            "immediate_score_candidate_count": int(immediate_count),
            "delayed_control_margin_candidate_count": int(delayed_margin_count),
            "delayed_control_matched_rejected_candidate_count": int(
                rejected_control_matched_count
            ),
            "no_score_rejected_candidate_count": int(rejected_no_score_count),
            "other_rejected_candidate_count": int(rejected_other_count),
            "rejected_candidate_count": int(rejected_count),
            "delayed_control_matched_count": int(delayed_matched_count),
            "no_score_association_count": int(no_score_count),
            "score_bearing_delta_sum": int(score_bearing_delta_sum),
            "join_margin_sum": int(join_margin_sum),
            "promotion_state": promotion_state,
            "reason_counts": {
                key: int(value) for key, value in sorted(reason_counts.items())
            },
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "discovered": discovered,
            "rejected": rejected,
            "ready_for_positive_rank_promotion": bool(discovered_count > 0),
            "planner_connection_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "runtime_execution_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
            "generated_new_candidate": False,
            "next_required_connection": (
                "score-attributed candidates require focused runtime replay "
                "and all25 non-regression before any ranking promotion"
            ),
        }
    )


def summarize_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_focused_runtime_replay(
    discovery: Mapping[str, Any] | Iterable[Mapping[str, Any]] | Any,
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Review score-attributed candidates for a future focused runtime replay.

    This stage is still diagnostic-only. It does not replay candidates or
    change ranking; it only records whether candidate-discovery produced
    score-attributed candidates that would be eligible for a later replay gate.
    """

    if isinstance(discovery, Mapping):
        discovery_summary = dict(discovery)
    else:
        discovery_summary = (
            summarize_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery(
                discovery,
                sample_limit=max(sample_limit, 32),
            )
        )

    reviewed_count = _safe_int(discovery_summary.get("candidate_reviewed_count"))
    discovered_count = _safe_int(
        discovery_summary.get("discovered_score_attributed_candidate_count")
    )
    rejected_count = _safe_int(discovery_summary.get("rejected_candidate_count"))
    control_matched_count = _safe_int(
        discovery_summary.get("delayed_control_matched_rejected_candidate_count")
    )
    no_score_count = _safe_int(discovery_summary.get("no_score_rejected_candidate_count"))
    score_bearing_delta_sum = _safe_int(
        discovery_summary.get("score_bearing_delta_sum")
    )
    join_margin_sum = _safe_int(discovery_summary.get("join_margin_sum"))
    discovered_rows = [
        dict(row)
        for row in discovery_summary.get("discovered", [])
        if isinstance(row, Mapping)
    ]
    rejected_rows = [
        dict(row)
        for row in discovery_summary.get("rejected", [])
        if isinstance(row, Mapping)
    ]

    by_game: dict[str, Counter[str]] = {}
    replay_candidates: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for row in discovered_rows:
        game_id = str(row.get("game_id", "") or "unknown")
        counter = by_game.setdefault(game_id, Counter())
        counter["score_attributed_candidate_replay_reviewed_count"] += 1
        counter["score_attributed_candidate_replay_pending_count"] += 1
        if len(replay_candidates) < sample_limit:
            replay_candidates.append(
                {
                    "candidate_key": str(row.get("candidate_key", "") or ""),
                    "game_id": game_id,
                    "variant": str(row.get("variant", "") or ""),
                    "step": _safe_int(row.get("step")),
                    "state_hash": str(row.get("state_hash", "") or ""),
                    "selected_action_after": str(
                        row.get("selected_action_after", "") or ""
                    ),
                    "selected_site_after": _xy_list(row.get("selected_site_after")),
                    "score_bearing_delta": _safe_int(row.get("score_bearing_delta")),
                    "join_margin": _safe_int(row.get("join_margin")),
                    "focused_runtime_replay_state": "pending_future_runtime_replay",
                    "state_hash_guard_ready": bool(row.get("state_hash")),
                    "existing_candidate_only": True,
                    "focused_runtime_replay_enabled": False,
                    "runtime_execution_enabled": False,
                    "action_generation_allowed": False,
                }
            )

    for row in rejected_rows:
        game_id = str(row.get("game_id", "") or "unknown")
        counter = by_game.setdefault(game_id, Counter())
        reason = str(row.get("blocked_reason", "") or "not_score_attributed")
        counter[f"blocked_reason:{reason}"] += 1
        if len(blocked) < sample_limit:
            blocked.append(
                {
                    "game_id": game_id,
                    "variant": str(row.get("variant", "") or ""),
                    "step": _safe_int(row.get("step")),
                    "state_hash": str(row.get("state_hash", "") or ""),
                    "blocked_reason": reason,
                    "selected_action_after": str(
                        row.get("selected_action_after", "") or ""
                    ),
                    "selected_site_after": _xy_list(row.get("selected_site_after")),
                    "immediate_score_delta": _safe_int(
                        row.get("immediate_score_delta")
                    ),
                    "delayed_score_delta": _safe_int(row.get("delayed_score_delta")),
                    "negative_control_max_score_delta": _safe_int(
                        row.get("negative_control_max_score_delta")
                    ),
                    "join_margin": _safe_int(row.get("join_margin")),
                    "focused_runtime_replay_state": "blocked_before_replay",
                }
            )

    if discovered_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_focused_runtime_replay_pending"
        )
    elif reviewed_count > 0 or rejected_count > 0:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_focused_runtime_replay_no_score_attributed_candidates"
        )
    else:
        promotion_state = (
            "tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_focused_runtime_replay_no_evidence"
        )

    return _json_safe(
        {
            "schema_version": (
                TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_SCORE_ATTRIBUTED_CANDIDATE_FOCUSED_RUNTIME_REPLAY_SCHEMA_VERSION
            ),
            "diagnostic_only": True,
            "default_off": True,
            "candidate_reviewed_count": int(reviewed_count),
            "discovered_score_attributed_candidate_count": int(discovered_count),
            "rejected_candidate_count": int(rejected_count),
            "delayed_control_matched_rejected_candidate_count": int(
                control_matched_count
            ),
            "no_score_rejected_candidate_count": int(no_score_count),
            "focused_runtime_replay_reviewed_candidate_count": int(discovered_count),
            "focused_runtime_replay_pending_candidate_count": int(discovered_count),
            "focused_runtime_replay_executed_count": 0,
            "focused_runtime_replay_score_delta_sum": 0,
            "focused_runtime_replay_positive_margin_count": 0,
            "blocked_before_replay_count": int(rejected_count),
            "blocked_no_score_attributed_candidate_count": int(
                max(0, reviewed_count - discovered_count)
            ),
            "score_bearing_delta_sum": int(score_bearing_delta_sum),
            "join_margin_sum": int(join_margin_sum),
            "promotion_state": promotion_state,
            "by_game": {
                game_id: dict(sorted(counter.items()))
                for game_id, counter in sorted(by_game.items())
            },
            "replay_candidates": replay_candidates,
            "blocked": blocked,
            "ready_for_runtime_replay_review": bool(discovered_count > 0),
            "focused_runtime_replay_enabled": False,
            "planner_connection_enabled": False,
            "ranking_enabled": False,
            "positive_boost_enabled": False,
            "runtime_execution_enabled": False,
            "action_generation_allowed": False,
            "model_repair_write_enabled": False,
            "generated_new_candidate": False,
            "next_required_connection": (
                "positive focused runtime replay requires a nonzero "
                "score-attributed candidate set plus a separate runtime "
                "execution/admission gate"
            ),
        }
    )


def _tile_effect_counterfactual_stats(
    packets: Iterable[Mapping[str, Any]],
    *,
    normal_step_max: int,
) -> tuple[dict[str, Counter[str]], Counter[tuple[str, str]]]:
    stats_by_key: dict[str, Counter[str]] = defaultdict(Counter)
    non_reset_seen_cells: dict[tuple[str, str], set[Cell]] = defaultdict(set)
    prior_occupied_cells: dict[tuple[str, str], set[Cell]] = defaultdict(set)
    scoreless_reset_by_identity: Counter[tuple[str, str]] = Counter()
    normal_step = max(1, int(normal_step_max))

    for packet in packets:
        if not isinstance(packet, Mapping):
            continue
        game_id = str(packet.get("game_id", "") or "unknown").strip().lower()
        score_delta = _safe_int(packet.get("score_delta"))
        score_linked = bool(packet.get("score_closure_candidate")) or score_delta > 0
        moves = packet.get("moves") if isinstance(packet.get("moves"), list) else []
        for move in moves:
            if not isinstance(move, Mapping):
                continue
            prev = _vector_from_value(move.get("prev_cell"))
            cur = _vector_from_value(move.get("cur_cell"))
            if prev is None or cur is None:
                continue
            displacement = _vector_from_value(move.get("displacement"))
            if displacement is None:
                displacement = (cur[0] - prev[0], cur[1] - prev[1])
            identity_key = _identity_key_from_move(move)
            identity_scope = (game_id, identity_key)
            if abs(displacement[0]) + abs(displacement[1]) <= normal_step:
                non_reset_seen_cells[identity_scope].add(prev)
                non_reset_seen_cells[identity_scope].add(cur)
                prior_occupied_cells[identity_scope].add(prev)
                prior_occupied_cells[identity_scope].add(cur)
                continue

            posterior_key = _posterior_key(
                {
                    "game_id": game_id,
                    "identity_key": identity_key,
                    "return_anchor": [cur[0], cur[1]],
                }
            )
            counter = stats_by_key[posterior_key]
            counter["reset_observation_count"] += 1
            if score_linked:
                counter["score_linked_reset_count"] += 1
                move_count = len(moves)
                counter["score_linked_reset_packet_count"] += 1
                counter["score_linked_reset_packet_move_count_sum"] += move_count
                counter["score_linked_reset_packet_move_count_max"] = max(
                    _safe_int(
                        counter.get("score_linked_reset_packet_move_count_max")
                    ),
                    move_count,
                )
            else:
                counter["scoreless_reset_count"] += 1
                scoreless_reset_by_identity[identity_scope] += 1
            if cur in non_reset_seen_cells.get(identity_scope, set()):
                counter["return_anchor_prior_non_reset_seen_count"] += 1
            if cur in prior_occupied_cells.get(identity_scope, set()):
                counter["return_anchor_prior_occupancy_seen_count"] += 1
            prior_occupied_cells[identity_scope].add(prev)
            prior_occupied_cells[identity_scope].add(cur)
            source = _hazard_source_cell_from_move(move)
            if source is None:
                counter["missing_hazard_source_observation_count"] += 1
                continue
            counter["hazard_source_observed_count"] += 1
            if source != cur:
                counter["hazard_source_distinct_from_return_anchor_count"] += 1
            counter[f"source:{source[0]},{source[1]}"] += 1

    return stats_by_key, scoreless_reset_by_identity


def _hazard_source_cell_from_move(move: Mapping[str, Any]) -> Cell | None:
    for key in (
        "hazard_source_cell",
        "attempted_cell",
        "predicted_target_cell",
        "predicted_target",
        "candidate_tile_cell",
        "source_cell",
    ):
        cell = _vector_from_value(move.get(key))
        if cell is not None:
            return cell
    return None


def _hazard_source_cell_from_trace_context(
    trace_context: Mapping[str, Any] | None,
) -> Cell | None:
    if not isinstance(trace_context, Mapping):
        return None
    for path in (
        ("target_verification", "site"),
        ("target_binding", "site"),
        ("rod_target_binding", "selected_site"),
        ("rod_target_binding", "current_site"),
        ("rod_target_binding", "last", "selected_site"),
        ("rod_target_binding", "last", "current_site"),
        ("control_target", "primary_target", "site"),
        ("control_target", "site"),
    ):
        cell = _cell_from_path(trace_context, path)
        if cell is not None:
            return cell
    target_key = _value_from_path(trace_context, ("target_verification", "target_key"))
    cell = _cell_from_target_key(target_key)
    if cell is not None:
        return cell
    target_key = _value_from_path(trace_context, ("control_target", "primary_key"))
    return _cell_from_target_key(target_key)


def _hazard_source_cell_from_action_payload(
    action_payload: Mapping[str, Any] | None,
) -> Cell | None:
    if not isinstance(action_payload, Mapping):
        return None
    for key in (
        "attempted_site",
        "click_site",
        "site",
        "target_site",
        "source_site",
        "position_memory_anchor",
    ):
        cell = _vector_from_value(action_payload.get(key))
        if cell is not None:
            return cell
    x = action_payload.get("x")
    y = action_payload.get("y")
    if x is None or y is None:
        return None
    try:
        return (int(x), int(y))
    except (TypeError, ValueError):
        return None


def _cell_from_path(root: Mapping[str, Any], path: tuple[str, ...]) -> Cell | None:
    value = _value_from_path(root, path)
    return _vector_from_value(value)


def _value_from_path(root: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = root
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _cell_from_target_key(value: Any) -> Cell | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    for offset in range(len(value) - 1):
        try:
            return (int(value[offset]), int(value[offset + 1]))
        except (TypeError, ValueError):
            continue
    return None


def _cell_from_source_key(value: Any) -> Cell | None:
    if not isinstance(value, str) or "," not in value:
        return None
    left, right = value.split(",", 1)
    try:
        return (int(left), int(right))
    except (TypeError, ValueError):
        return None


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


def _posterior_key(row: Mapping[str, Any]) -> str:
    anchor = _vector_from_value(row.get("return_anchor")) or (0, 0)
    return (
        f"{row.get('game_id', 'unknown')}:"
        f"{row.get('identity_key', 'unknown')}:"
        f"{anchor[0]},{anchor[1]}"
    )


def _vector_from_value(value: Any) -> Cell | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return (_safe_int(value[0]), _safe_int(value[1]))


def _displacement_from_cells(prev_cell: Any, cur_cell: Any) -> Cell | None:
    prev = _vector_from_value(prev_cell)
    cur = _vector_from_value(cur_cell)
    if prev is None or cur is None:
        return None
    return (cur[0] - prev[0], cur[1] - prev[1])


def _centroid_cell(value: Any) -> Cell | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return (int(round(float(value[0]))), int(round(float(value[1]))))
    except (TypeError, ValueError):
        return None


def _bbox_centroid_cell(value: Any) -> Cell | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        y0 = float(value[0])
        x0 = float(value[1])
        y1 = float(value[2])
        x1 = float(value[3])
    except (TypeError, ValueError):
        return None
    return (int(round((y0 + y1) / 2.0)), int(round((x0 + x1) / 2.0)))


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _xy_list(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            return [_safe_int(value.get("x")), _safe_int(value.get("y"))]
        for key in ("site", "selected_site", "selected_candidate_site"):
            nested = _xy_list(value.get(key))
            if nested:
                return nested
        return []
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return []
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError):
        return []


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
    "TILE_EFFECT_FAMILY_SCOPE_NEGATIVE_CONTROL_SCHEMA_VERSION",
    "TILE_EFFECT_GUARD_COUNTERFACTUAL_CONTROL_SCHEMA_VERSION",
    "TILE_EFFECT_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION",
    "TILE_EFFECT_GUARD_PRECISION_REVIEW_SCHEMA_VERSION",
    "TILE_EFFECT_HAZARD_SOURCE_REPAIR_BRIDGE_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_ACTION_SELECTION_GUARD_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_POSITIVE_ACTION_SELECTION_ADMISSION_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_POSITIVE_ACTION_SELECTION_SCORE_ATTRIBUTION_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_SCORE_ATTRIBUTED_CANDIDATE_FOCUSED_RUNTIME_REPLAY_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_SCORE_ATTRIBUTED_CANDIDATE_DISCOVERY_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_SCORE_ATTRIBUTED_POSITIVE_RANK_PROMOTION_GUARD_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROPOSAL_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_LIVE_RANK_GUARD_FOCUSED_RUNTIME_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_EXECUTION_GATE_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_OUTCOME_POSTERIOR_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RANK_GUARD_RUNTIME_GUARD_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RANK_GUARD_SHADOW_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_BRIDGE_OUTCOME_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_BRIDGE_GUARD_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_PROTECTED_PROBE_RUNTIME_EXECUTOR_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_RUNTIME_PROBE_PROPOSAL_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_WRITE_AUDIT_SCHEMA_VERSION",
    "TILE_EFFECT_MODEL_REPAIR_WRITE_GUARD_SCHEMA_VERSION",
    "TILE_EFFECT_POSTERIOR_SCHEMA_VERSION",
    "TILE_EFFECT_RUNTIME_TRACE_PACKET_SCHEMA_VERSION",
    "TILE_EFFECT_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION",
    "TileEffectLearner",
    "build_tile_effect_runtime_trace_packet_from_action_effect_record",
    "summarize_tile_effect_family_scope_negative_control",
    "summarize_tile_effect_hazard_source_repair_bridge",
    "summarize_tile_effect_guard_counterfactual_control",
    "summarize_tile_effect_guard_precision_review",
    "summarize_tile_effect_guarded_planner_connection",
    "summarize_tile_effect_model_repair_proposal",
    "summarize_tile_effect_model_repair_protected_probe_execution_gate",
    "summarize_tile_effect_model_repair_protected_probe_live_rank_guard_action_selection_guard",
    "summarize_tile_effect_model_repair_protected_probe_live_rank_guard_focused_runtime",
    "summarize_tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_admission",
    "summarize_tile_effect_model_repair_protected_probe_live_rank_guard_positive_action_selection_score_attribution",
    "summarize_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_focused_runtime_replay",
    "summarize_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_candidate_discovery",
    "summarize_tile_effect_model_repair_protected_probe_live_rank_guard_score_attributed_positive_rank_promotion_guard",
    "summarize_tile_effect_model_repair_protected_probe_outcome_posterior",
    "summarize_tile_effect_model_repair_protected_probe_rank_guard_runtime_guard",
    "summarize_tile_effect_model_repair_protected_probe_rank_guard_shadow",
    "summarize_tile_effect_model_repair_protected_probe_runtime_bridge_guard",
    "summarize_tile_effect_model_repair_protected_probe_runtime_bridge_outcome",
    "summarize_tile_effect_model_repair_protected_probe_runtime_executor",
    "summarize_tile_effect_model_repair_runtime_probe_proposal",
    "summarize_tile_effect_model_repair_write_audit",
    "summarize_tile_effect_model_repair_write_guard",
    "summarize_tile_effect_posterior",
    "summarize_tile_effect_runtime_trace_packets",
]
