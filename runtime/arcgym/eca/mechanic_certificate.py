"""MechanicCertificate registry: structure-keyed solver dispatch.

Phase L-3 (2026-05-31). Phase L-2 showed that dispatching a solver on a
palette-independent *firing certificate* (mechanic structure, not colour
cue) transfers a palette-shifted game to the right solver without
regressing the floor — demonstrated on the mirror mechanic (gtm0 0->1,
canonical 41/25 preserved).

This module generalises that one-off into a small framework so adding
the next mechanic's certificate (color_cycle, sand_filter, ...) is a
registry entry plus fixtures, NOT another ad-hoc `if` in the dispatch
chain (which would slide back toward per-game solvers).

A certificate is deliberately HIGH-PRECISION / lower-recall: in dispatch,
a false positive can steal another game's slot and break the floor (see
the L-2 v1 sp80 regression), whereas a false negative only misses a
transfer. So `evaluate` should reject aggressively and explain why.

Design contract per certificate:
  - name:        the primitive_id it certifies for dispatch
  - evaluate(state) -> CertificateVerdict (certified + human reason +
    precision tier). `state` is the mechanic's own perception output;
    the registry does NOT impose a shared perception.
  - latch:       whether "this is an X game" should stick for the
    episode once certified (the mechanic may transiently violate its
    own static signature mid-solve — e.g. mirror parity-break — so the
    instantaneous certificate must not be required every step). The
    latch STATE lives in the loop; this flag records the policy.
  - protected_note: games the certificate must NOT fire on (regression
    guardrails), recorded for the fixtures.

There is ZERO game-specific code here: certificates key on geometric /
structural invariants, never on coordinates, game ids, or palettes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


MECHANIC_CERTIFICATE_RUNTIME_TRACE_PACKET_SCHEMA_VERSION = (
    "eca_mechanic_certificate_runtime_trace_packet.v1"
)
MECHANIC_CERTIFICATE_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION = (
    "eca_mechanic_certificate_runtime_trace_sensor.v1"
)
MECHANIC_CERTIFICATE_POSTERIOR_SCHEMA_VERSION = (
    "eca_mechanic_certificate_posterior.v1"
)
MECHANIC_CERTIFICATE_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION = (
    "eca_mechanic_certificate_guarded_planner_connection.v1"
)


@dataclass(frozen=True)
class CertificateVerdict:
    """Result of evaluating a mechanic certificate on a perceived state."""

    certified: bool
    reason: str
    precision: str = "high"  # "high" | "medium"


class MechanicCertificate(Protocol):
    """A structure-keyed firing certificate for one solver/mechanic."""

    name: str
    latch: bool
    protected_note: str

    def evaluate(self, state: dict[str, Any] | None) -> CertificateVerdict:
        ...


class MirrorCertificate:
    """Certifies a genuine mirror_x cursor pair (Phase L-2 logic).

    positive: m0r0, gtm0 (palette-shift)
    negative: ft09 (16 cursor cells, asymmetric), sp80 (1-cell split blob)
    """

    name = "mirror_joint_cursor_bfs"
    latch = True
    protected_note = "must be FALSE on ft09 (color blobs) and sp80 (split blob)"

    def evaluate(self, state: dict[str, Any] | None) -> CertificateVerdict:
        if not state:
            return CertificateVerdict(False, "no mirror state")
        try:
            ca = state["cursor_a"]
            cb = state["cursor_b"]
            gw, _gh = state["grid_size"]
        except (KeyError, TypeError, ValueError):
            return CertificateVerdict(False, "malformed mirror state")
        symmetric = (int(ca[0]) + int(cb[0]) == int(gw) - 1)
        same_row = (int(ca[1]) == int(cb[1]))
        if not symmetric:
            return CertificateVerdict(
                False, f"pair not mirror-symmetric (x_a+x_b={ca[0]+cb[0]} != gw-1={gw-1})"
            )
        if not same_row:
            return CertificateVerdict(
                False, f"pair not same-row (y_a={ca[1]} != y_b={cb[1]})"
            )
        ncur = state.get("cursor_cell_count")
        if ncur is not None and int(ncur) != 2:
            return CertificateVerdict(
                False, f"cursor_cell_count={ncur} != 2 (not a genuine pair)"
            )
        return CertificateVerdict(
            True,
            f"symmetric same-row cursor pair, cells={ncur if ncur is not None else 'n/a'}",
        )


class ColorCycleCertificate:
    """Certifies a genuine colour-cycle constraint puzzle (Phase L-4).

    Phase L-4a is OBSERVATION-ONLY: this certificate is registered and
    its verdict/reason is logged, but it does NOT yet gate dispatch. The
    measured fact it encodes (Phase L-4 sweep): in the canonical suite
    `color_cycle_constraint_solver` already fires on ft09 ONLY (75/75
    steps, 6 WIN) and on no other game, so color_cycle is NOT currently
    over-broad. gtm0's historical mis-fire is already pre-empted upstream
    by the mirror certificate's early dispatch. This certificate is thus
    PREVENTIVE: it pins what "genuinely color_cycle" means structurally so
    a future transfer variant cannot quietly widen the dispatch.

    positive evidence (ft09, real): a puzzle-block grid yielding a small
      finite palette (2-4 colours), at least one template block to match,
      at least one clickable cell, and a non-empty solve plan with
      explicit constraints. These are palette-INDEPENDENT structural
      facts about a colour-cycle template-match puzzle, not a colour
      count on the raw frame.
    negative evidence: mirror games (m0r0/gtm0) produce NO color_cycle
      state at all (perception returns None -> rejected); a degenerate
      state with no templates, an empty plan, or a palette outside [2,4]
      is rejected. The cross-mechanic deferral (mirror wins when its own
      certificate latches) is enforced upstream at dispatch, not here.
    """

    name = "color_cycle_constraint_solver"
    latch = False  # template-match state is stable; no parity-break analogue
    protected_note = (
        "must be FALSE on mirror games (m0r0/gtm0: no color_cycle state) "
        "and on degenerate states (no templates / empty plan / palette "
        "outside [2,4])"
    )

    _MIN_PALETTE = 2
    _MAX_PALETTE = 4

    def evaluate(self, state: dict[str, Any] | None) -> CertificateVerdict:
        if not state:
            # Mirror games and non-puzzle frames land here: perception
            # produced no color_cycle constraint state at all.
            return CertificateVerdict(False, "no color_cycle state")
        palette = state.get("palette")
        if not isinstance(palette, (list, tuple)) or not palette:
            return CertificateVerdict(False, "no palette in color_cycle state")
        n_pal = len(palette)
        if not (self._MIN_PALETTE <= n_pal <= self._MAX_PALETTE):
            return CertificateVerdict(
                False,
                f"palette size {n_pal} outside cycle range "
                f"[{self._MIN_PALETTE},{self._MAX_PALETTE}]",
            )
        templates = state.get("template_count")
        if templates is None:
            templates = len(state.get("templates", []) or [])
        if int(templates) < 1:
            return CertificateVerdict(False, "no template block to match")
        clickable = int(state.get("clickable_count", 0) or 0)
        if clickable < 1:
            return CertificateVerdict(False, "no clickable cell")
        plan = state.get("plan")
        if not plan:
            return CertificateVerdict(False, "empty solve plan")
        constraints = int(state.get("constraint_count", 0) or 0)
        if constraints < 1:
            return CertificateVerdict(False, "no explicit constraints")
        return CertificateVerdict(
            True,
            f"palette={n_pal} colours, templates={templates}, "
            f"clickable={clickable}, plan_len={len(plan)}, "
            f"constraints={constraints}",
        )


class SandFilterCertificate:
    """Certifies a genuine sand-filter flow puzzle (Phase L-5).

    Phase L-5 is OBSERVATION-ONLY (same staged rollout as L-4): registered
    and logged, not yet gating dispatch. Measured fact (Phase L-5 sweep):
    `sand_filter_flow_solver` already fires on sp80 ONLY (20/100 steps,
    reaching its canonical L3 ceiling) and on no other game, so it is NOT
    currently over-broad. This certificate is PREVENTIVE: it pins what
    "genuinely sand_filter" means at the perception-OUTPUT layer (the
    perception has already resolved literal colours into filter/bowl/source
    ROLES), which is palette-independent even though the underlying
    perception keys on literal colours (filters 8/9, bowls 11/13,
    sources 4) — the same colour-coupling found for color_cycle in L-4b.

    positive evidence (sp80, real): mechanic_state_type == sand_filter_flow,
      a bounded set of filters (1-4), bowls (1-5) and sources (1-3), at
      least one solved target filter, and a non-empty operation plan. These
      are structural counts over resolved roles, not raw colours.
    negative evidence: no state (perception returned None on a non-puzzle
      frame), wrong mechanic_state_type, any role count zero or above its
      bound, no target filter, or an empty plan.
    """

    name = "sand_filter_flow_solver"
    latch = False  # filter/bowl/source layout is stable across the solve
    protected_note = (
        "must be FALSE when perception yields no sand_filter_flow state "
        "(non-puzzle frames) or degenerate counts (0 filters/bowls/sources, "
        "no target, empty plan)"
    )

    _MAX_FILTERS = 4
    _MAX_BOWLS = 5
    _MAX_SOURCES = 3

    def evaluate(self, state: dict[str, Any] | None) -> CertificateVerdict:
        if not state:
            return CertificateVerdict(False, "no sand_filter state")
        if state.get("mechanic_state_type") != "sand_filter_flow":
            return CertificateVerdict(
                False,
                f"mechanic_state_type={state.get('mechanic_state_type')!r} "
                "!= sand_filter_flow",
            )
        nf = int(state.get("filter_count", 0) or 0)
        nb = int(state.get("bowl_count", 0) or 0)
        ns = int(state.get("source_count", 0) or 0)
        if not (1 <= nf <= self._MAX_FILTERS):
            return CertificateVerdict(
                False, f"filter_count {nf} outside [1,{self._MAX_FILTERS}]"
            )
        if not (1 <= nb <= self._MAX_BOWLS):
            return CertificateVerdict(
                False, f"bowl_count {nb} outside [1,{self._MAX_BOWLS}]"
            )
        if not (1 <= ns <= self._MAX_SOURCES):
            return CertificateVerdict(
                False, f"source_count {ns} outside [1,{self._MAX_SOURCES}]"
            )
        targets = state.get("target_filters")
        if not targets:
            return CertificateVerdict(False, "no solved target filter")
        plan = state.get("operation_plan")
        if not plan:
            return CertificateVerdict(False, "empty operation plan")
        return CertificateVerdict(
            True,
            f"sand_filter flow: filters={nf}, bowls={nb}, sources={ns}, "
            f"targets={len(targets)}, plan_len={len(plan)}",
        )


class StrideMazeCertificate:
    """Certifies a genuine stride-marker maze (tu93 family, 2026-07-06).

    The perception (`_stride_marker_maze_state`) is already a strong gate:
    it only returns a state when a stride-6 lattice with an actor marker, a
    distinct target marker, AND a complete BFS path over walkable lattice
    cells exists. This certificate pins the structural invariants of that
    output so a lattice-shaped false positive on another mechanic cannot
    slip through (post-ka59-mirror-misfire discipline: detection alone is
    not screening).

    positive: tu93 (measured L1 204 -> 30 actions when the plan is used)
    negative: any frame where the state parser fails; degenerate lattices
      (tiny walkable set, actor==target colour, implausibly long plan)
    """

    name = "stride_marker_maze_bfs"
    latch = True
    protected_note = "perception requires stride-6 lattice + complete BFS path"

    _MIN_WALKABLE = 4
    _MAX_PLAN_LEN = 400
    _MAX_MARKERS = 2000

    def evaluate(self, state: dict[str, Any] | None) -> CertificateVerdict:
        if not state:
            return CertificateVerdict(False, "no stride_maze state")
        if int(state.get("stride", 0) or 0) != 6 or int(state.get("edge_offset", -1)) != 3:
            return CertificateVerdict(
                False,
                f"lattice geometry stride={state.get('stride')}/edge_offset="
                f"{state.get('edge_offset')} != 6/3",
            )
        walkable = state.get("walkable") or ()
        if len(walkable) < self._MIN_WALKABLE:
            return CertificateVerdict(
                False, f"walkable cells {len(walkable)} < {self._MIN_WALKABLE}"
            )
        targets = state.get("targets") or ()
        if len(targets) != 1:
            return CertificateVerdict(False, f"target count {len(targets)} != 1")
        if int(state.get("actor_value", -1)) == int(state.get("target_value", -2)):
            return CertificateVerdict(False, "actor and target share a colour value")
        plan_len = int(state.get("plan_len", 0) or 0)
        if not (1 <= plan_len <= self._MAX_PLAN_LEN):
            return CertificateVerdict(
                False, f"plan_len {plan_len} outside [1,{self._MAX_PLAN_LEN}]"
            )
        markers = int(state.get("marker_count", 0) or 0)
        if markers > self._MAX_MARKERS:
            return CertificateVerdict(False, f"marker_count {markers} implausible")
        return CertificateVerdict(
            True,
            f"stride-6 maze: walkable={len(walkable)}, plan_len={plan_len}, "
            f"markers={markers}",
        )


# Registry: primitive_id -> certificate. Add the next mechanic's
# certificate here to extend dispatch specificity without touching the
# selection chain's ad-hoc branches.
MECHANIC_CERTIFICATES: dict[str, MechanicCertificate] = {
    MirrorCertificate.name: MirrorCertificate(),
    ColorCycleCertificate.name: ColorCycleCertificate(),
    SandFilterCertificate.name: SandFilterCertificate(),
    StrideMazeCertificate.name: StrideMazeCertificate(),
}


def evaluate_certificate(
    name: str, state: dict[str, Any] | None
) -> CertificateVerdict:
    """Evaluate the named mechanic's certificate. Returns a not-certified
    verdict if no certificate is registered for that name."""
    cert = MECHANIC_CERTIFICATES.get(name)
    if cert is None:
        return CertificateVerdict(False, f"no certificate registered for {name}")
    return cert.evaluate(state)


def build_mechanic_certificate_runtime_trace_packet(
    before_frame: Any,
    after_frame: Any,
    *,
    game_id: str = "",
    state_hash: str = "",
    step: int = 0,
    action_name: str = "",
    action_payload: dict[str, Any] | None = None,
    score_delta: int = 0,
    mechanics: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Emit certificate verdict transitions for one public-frame action.

    This is diagnostic-only. It measures whether existing structural
    certificates are present before/after an action and whether a status change
    coincides with score. It never dispatches, ranks, or generates actions.
    """

    before = _final_2d_frame(before_frame)
    after = _final_2d_frame(after_frame)
    mechanic_names = tuple(mechanics or tuple(MECHANIC_CERTIFICATES.keys()))
    verdict_rows: list[dict[str, Any]] = []
    before_status_counts: Counter[str] = Counter()
    after_status_counts: Counter[str] = Counter()
    status_changed_count = 0
    certified_after_mechanics: list[str] = []
    rejected_after_mechanics: list[str] = []

    for mechanic in mechanic_names:
        if mechanic not in MECHANIC_CERTIFICATES:
            continue
        before_state = _certificate_state_from_frame(mechanic, before)
        after_state = _certificate_state_from_frame(mechanic, after)
        before_verdict = evaluate_certificate(mechanic, before_state)
        after_verdict = evaluate_certificate(mechanic, after_state)
        before_status = "certified" if before_verdict.certified else "rejected"
        after_status = "certified" if after_verdict.certified else "rejected"
        changed = before_status != after_status
        status_changed_count += int(changed)
        before_status_counts[before_status] += 1
        after_status_counts[after_status] += 1
        if after_status == "certified":
            certified_after_mechanics.append(mechanic)
        else:
            rejected_after_mechanics.append(mechanic)
        verdict_rows.append(
            {
                "mechanic": mechanic,
                "before_certified": bool(before_verdict.certified),
                "after_certified": bool(after_verdict.certified),
                "before_status": before_status,
                "after_status": after_status,
                "status_changed": bool(changed),
                "before_reason": str(before_verdict.reason or ""),
                "after_reason": str(after_verdict.reason or ""),
                "before_precision": str(before_verdict.precision or ""),
                "after_precision": str(after_verdict.precision or ""),
                "before_state_present": before_state is not None,
                "after_state_present": after_state is not None,
                "before_state_summary": _compact_certificate_state(
                    mechanic, before_state
                ),
                "after_state_summary": _compact_certificate_state(
                    mechanic, after_state
                ),
            }
        )

    return _json_safe(
        {
            "schema_version": MECHANIC_CERTIFICATE_RUNTIME_TRACE_PACKET_SCHEMA_VERSION,
            "sensor_family": "mechanic_certificate_dispatch",
            "game_id": str(game_id or "").strip().lower(),
            "state_hash": str(state_hash or ""),
            "step": int(step),
            "action_name": str(action_name or ""),
            "action_payload": dict(action_payload or {}),
            "score_delta": int(score_delta),
            "mechanic_count": int(len(verdict_rows)),
            "verdicts": verdict_rows,
            "before_status_counts": dict(sorted(before_status_counts.items())),
            "after_status_counts": dict(sorted(after_status_counts.items())),
            "status_changed_count": int(status_changed_count),
            "certified_after_count": int(len(certified_after_mechanics)),
            "rejected_after_count": int(len(rejected_after_mechanics)),
            "certified_after_mechanics": certified_after_mechanics,
            "rejected_after_mechanics": rejected_after_mechanics,
            "score_linked_status_change_candidate": bool(
                int(score_delta) > 0 and status_changed_count > 0
            ),
            "diagnostic_only": True,
            "planner_connection_enabled": False,
            "runtime_execution_enabled": False,
            "ranking_enabled": False,
            "action_generation_allowed": False,
        }
    )


def summarize_mechanic_certificate_runtime_trace_packets(
    packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, dict)]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    after_status_histogram: Counter[str] = Counter()
    status_transition_histogram: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    score_delta_sum = 0

    for packet in packet_rows:
        accepted = (
            str(packet.get("schema_version", "") or "")
            == MECHANIC_CERTIFICATE_RUNTIME_TRACE_PACKET_SCHEMA_VERSION
            and str(packet.get("sensor_family", "") or "")
            == "mechanic_certificate_dispatch"
        )
        counts["accepted_packet_count" if accepted else "rejected_packet_count"] += 1
        if not accepted:
            continue
        counts["packet_count"] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["packet_count"] += 1
        score_delta = _safe_int(packet.get("score_delta"))
        score_delta_sum += score_delta
        status_changed_count = _safe_int(packet.get("status_changed_count"))
        certified_after_count = _safe_int(packet.get("certified_after_count"))
        rejected_after_count = _safe_int(packet.get("rejected_after_count"))
        counts["status_changed_count"] += status_changed_count
        counts["certified_after_count"] += certified_after_count
        counts["rejected_after_count"] += rejected_after_count
        if status_changed_count > 0:
            counts["packet_with_status_change_count"] += 1
            game_counter["packet_with_status_change_count"] += 1
        if certified_after_count > 0:
            counts["packet_with_certified_after_count"] += 1
            game_counter["packet_with_certified_after_count"] += 1
        if bool(packet.get("score_linked_status_change_candidate")):
            counts["score_linked_status_change_candidate_count"] += 1
            game_counter["score_linked_status_change_candidate_count"] += 1
        for status, raw_count in dict(packet.get("after_status_counts") or {}).items():
            after_status_histogram[str(status)] += _safe_int(raw_count)
        for verdict in list(packet.get("verdicts") or []):
            if not isinstance(verdict, dict):
                continue
            mechanic = str(verdict.get("mechanic", "") or "unknown")
            mechanic_counter = by_mechanic.setdefault(mechanic, Counter())
            before_status = str(verdict.get("before_status", "") or "rejected")
            after_status = str(verdict.get("after_status", "") or "rejected")
            mechanic_counter[f"after::{after_status}"] += 1
            mechanic_counter["status_changed_count"] += int(
                bool(verdict.get("status_changed"))
            )
            status_transition_histogram[f"{mechanic}:{before_status}->{after_status}"] += 1
        if len(samples) < sample_limit:
            samples.append(
                {
                    "game_id": game_id,
                    "step": _safe_int(packet.get("step")),
                    "action_name": str(packet.get("action_name", "") or ""),
                    "score_delta": score_delta,
                    "status_changed_count": status_changed_count,
                    "certified_after_mechanics": list(
                        packet.get("certified_after_mechanics") or []
                    ),
                    "status_changed_mechanics": [
                        str(v.get("mechanic", "") or "")
                        for v in list(packet.get("verdicts") or [])
                        if isinstance(v, dict) and bool(v.get("status_changed"))
                    ],
                }
            )

    if counts["score_linked_status_change_candidate_count"] > 0:
        promotion_state = (
            "mechanic_certificate_runtime_trace_score_linked_status_change_observed"
        )
    elif counts["packet_with_status_change_count"] > 0:
        promotion_state = "mechanic_certificate_runtime_trace_status_change_observed"
    elif counts["packet_with_certified_after_count"] > 0:
        promotion_state = "mechanic_certificate_runtime_trace_certified_observed"
    elif counts["packet_count"] > 0:
        promotion_state = "mechanic_certificate_runtime_trace_connection_observed"
    else:
        promotion_state = "mechanic_certificate_runtime_trace_no_packets"

    return {
        "schema_version": MECHANIC_CERTIFICATE_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION,
        "diagnostic_only": True,
        "packet_count": counts["packet_count"],
        "accepted_packet_count": counts["accepted_packet_count"],
        "rejected_packet_count": counts["rejected_packet_count"],
        "status_changed_count": counts["status_changed_count"],
        "packet_with_status_change_count": counts["packet_with_status_change_count"],
        "certified_after_count": counts["certified_after_count"],
        "rejected_after_count": counts["rejected_after_count"],
        "packet_with_certified_after_count": counts[
            "packet_with_certified_after_count"
        ],
        "score_linked_status_change_candidate_count": counts[
            "score_linked_status_change_candidate_count"
        ],
        "score_delta_sum": int(score_delta_sum),
        "after_status_histogram": dict(sorted(after_status_histogram.items())),
        "status_transition_histogram": dict(
            sorted(status_transition_histogram.items())
        ),
        "by_game": {
            game: dict(sorted(counter.items()))
            for game, counter in sorted(by_game.items())
        },
        "by_mechanic": {
            mechanic: dict(sorted(counter.items()))
            for mechanic, counter in sorted(by_mechanic.items())
        },
        "samples": samples,
        "promotion_state": promotion_state,
        "runtime_trace_connection_observed": bool(counts["packet_count"] > 0),
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
    }


def summarize_mechanic_certificate_posterior(
    packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    entry_limit: int = 128,
) -> dict[str, Any]:
    packet_rows = [dict(packet) for packet in packets if isinstance(packet, dict)]
    groups: dict[str, Counter[str]] = {}
    accepted_count = 0
    rejected_count = 0

    for packet in packet_rows:
        accepted = (
            str(packet.get("schema_version", "") or "")
            == MECHANIC_CERTIFICATE_RUNTIME_TRACE_PACKET_SCHEMA_VERSION
            and str(packet.get("sensor_family", "") or "")
            == "mechanic_certificate_dispatch"
        )
        if not accepted:
            rejected_count += 1
            continue
        accepted_count += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        score_delta = _safe_int(packet.get("score_delta"))
        for verdict in list(packet.get("verdicts") or []):
            if not isinstance(verdict, dict):
                continue
            mechanic = str(verdict.get("mechanic", "") or "unknown")
            key = f"{game_id}:{mechanic}"
            counter = groups.setdefault(key, Counter())
            counter["observation_count"] += 1
            counter["score_delta_sum"] += score_delta
            after_status = str(verdict.get("after_status", "") or "rejected")
            if after_status == "certified":
                counter["after_certified_count"] += 1
            else:
                counter["after_rejected_count"] += 1
            if bool(verdict.get("status_changed")):
                counter["status_changed_count"] += 1
                if score_delta > 0:
                    counter["score_linked_status_change_count"] += 1
            if score_delta > 0:
                counter["score_positive_observation_count"] += 1
                if after_status == "certified":
                    counter["score_positive_certified_count"] += 1
                else:
                    counter["score_positive_rejected_count"] += 1

    entries: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    support_kind_counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}

    for key, counter in sorted(groups.items()):
        game_id, mechanic = key.split(":", 1) if ":" in key else ("unknown", key)
        state, support_kind = _mechanic_certificate_posterior_state(counter)
        state_counts[state] += 1
        support_kind_counts[support_kind] += 1
        game_counter = by_game.setdefault(game_id, Counter())
        mechanic_counter = by_mechanic.setdefault(mechanic, Counter())
        game_counter["posterior_entry_count"] += 1
        game_counter[f"{state}_count"] += 1
        mechanic_counter["posterior_entry_count"] += 1
        mechanic_counter[f"{state}_count"] += 1
        entries.append(
            {
                "posterior_key": key,
                "game_id": game_id,
                "mechanic": mechanic,
                "posterior_state": state,
                "support_kind": support_kind,
                "observation_count": counter["observation_count"],
                "after_certified_count": counter["after_certified_count"],
                "after_rejected_count": counter["after_rejected_count"],
                "status_changed_count": counter["status_changed_count"],
                "score_linked_status_change_count": counter[
                    "score_linked_status_change_count"
                ],
                "score_positive_observation_count": counter[
                    "score_positive_observation_count"
                ],
                "score_positive_certified_count": counter[
                    "score_positive_certified_count"
                ],
                "score_positive_rejected_count": counter[
                    "score_positive_rejected_count"
                ],
                "score_delta_sum": counter["score_delta_sum"],
            }
        )

    confirmed_games: Counter[str] = Counter()
    for entry in entries:
        if entry["posterior_state"] == "confirmed":
            confirmed_games[str(entry["game_id"])] += 1
    unique_confirmed_games = sorted(
        game for game, count in confirmed_games.items() if count == 1
    )
    competing_confirmed_games = sorted(
        game for game, count in confirmed_games.items() if count > 1
    )
    score_linked_confirmed_count = sum(
        1
        for entry in entries
        if entry["posterior_state"] == "confirmed"
        and (
            _safe_int(entry.get("score_linked_status_change_count")) > 0
            or _safe_int(entry.get("score_positive_certified_count")) > 0
        )
    )
    if score_linked_confirmed_count > 0 and competing_confirmed_games:
        promotion_state = (
            "mechanic_certificate_posterior_score_linked_confirmed_with_competing_roles"
        )
    elif score_linked_confirmed_count > 0:
        promotion_state = "mechanic_certificate_posterior_score_linked_confirmed"
    elif state_counts["confirmed"] > 0:
        promotion_state = "mechanic_certificate_posterior_confirmed"
    elif entries:
        promotion_state = "mechanic_certificate_posterior_observed"
    else:
        promotion_state = "mechanic_certificate_posterior_no_packets"

    return {
        "schema_version": MECHANIC_CERTIFICATE_POSTERIOR_SCHEMA_VERSION,
        "diagnostic_only": True,
        "packet_count": accepted_count,
        "accepted_packet_count": accepted_count,
        "rejected_packet_count": rejected_count,
        "posterior_entry_count": len(entries),
        "confirmed_count": state_counts["confirmed"],
        "falsified_count": state_counts["falsified"],
        "ambiguous_count": state_counts["ambiguous"],
        "insufficient_evidence_count": state_counts["insufficient_evidence"],
        "score_linked_confirmed_count": score_linked_confirmed_count,
        "posterior_state_counts": dict(sorted(state_counts.items())),
        "support_kind_counts": dict(sorted(support_kind_counts.items())),
        "by_game": {
            game: dict(sorted(counter.items()))
            for game, counter in sorted(by_game.items())
        },
        "by_mechanic": {
            mechanic: dict(sorted(counter.items()))
            for mechanic, counter in sorted(by_mechanic.items())
        },
        "unique_confirmed_game_count": len(unique_confirmed_games),
        "competing_confirmed_game_count": len(competing_confirmed_games),
        "unique_confirmed_games": unique_confirmed_games,
        "competing_confirmed_games": competing_confirmed_games,
        "entries": entries[: max(0, int(entry_limit))],
        "promotion_state": promotion_state,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
    }


def summarize_mechanic_certificate_guarded_planner_connection(
    packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    entry_limit: int = 128,
) -> dict[str, Any]:
    posterior = summarize_mechanic_certificate_posterior(
        packets, entry_limit=entry_limit
    )
    entries = [
        dict(entry)
        for entry in list(posterior.get("entries") or [])
        if isinstance(entry, dict)
    ]
    confirmed_by_game: Counter[str] = Counter()
    for entry in entries:
        if str(entry.get("posterior_state", "") or "") == "confirmed":
            confirmed_by_game[str(entry.get("game_id", "") or "unknown")] += 1

    counts: Counter[str] = Counter()
    admitted_entries: list[dict[str, Any]] = []
    blocked_entries: list[dict[str, Any]] = []
    for entry in entries:
        state = str(entry.get("posterior_state", "") or "")
        game_id = str(entry.get("game_id", "") or "unknown")
        if state == "confirmed" and confirmed_by_game.get(game_id, 0) == 1:
            decision = {
                "posterior_key": entry.get("posterior_key", ""),
                "game_id": game_id,
                "mechanic": entry.get("mechanic", ""),
                "support_kind": entry.get("support_kind", ""),
                "observation_count": _safe_int(entry.get("observation_count")),
                "after_certified_count": _safe_int(entry.get("after_certified_count")),
                "after_rejected_count": _safe_int(entry.get("after_rejected_count")),
                "score_linked_status_change_count": _safe_int(
                    entry.get("score_linked_status_change_count")
                ),
                "score_positive_certified_count": _safe_int(
                    entry.get("score_positive_certified_count")
                ),
                "guard_decision": "admitted",
                "guard_reason": "unique_confirmed_role",
            }
            admitted_entries.append(decision)
            counts["planner_connection_admitted_count"] += 1
            continue
        if state == "confirmed":
            reason = "blocked_competing_confirmed_roles"
            counts["blocked_competing_confirmed_count"] += 1
        elif state == "ambiguous":
            reason = "blocked_ambiguous_posterior"
            counts["blocked_ambiguous_count"] += 1
        elif state == "falsified":
            reason = "blocked_falsified_posterior"
            counts["blocked_falsified_count"] += 1
        elif state == "insufficient_evidence":
            reason = "blocked_insufficient_evidence"
            counts["blocked_insufficient_evidence_count"] += 1
        else:
            reason = "blocked_unknown_posterior_state"
            counts["blocked_other_count"] += 1
        blocked_entries.append(
            {
                "posterior_key": entry.get("posterior_key", ""),
                "game_id": game_id,
                "mechanic": entry.get("mechanic", ""),
                "posterior_state": state,
                "support_kind": entry.get("support_kind", ""),
                "guard_decision": "blocked",
                "guard_reason": reason,
            }
        )
        counts["planner_connection_blocked_count"] += 1

    if counts["planner_connection_admitted_count"] > 0 and counts[
        "blocked_competing_confirmed_count"
    ] > 0:
        promotion_state = (
            "mechanic_certificate_guarded_planner_connection_admitted_with_competing_blocks"
        )
    elif counts["planner_connection_admitted_count"] > 0:
        promotion_state = "mechanic_certificate_guarded_planner_connection_shadow_admitted"
    elif entries:
        promotion_state = "mechanic_certificate_guarded_planner_connection_all_blocked"
    else:
        promotion_state = "mechanic_certificate_guarded_planner_connection_no_entries"

    return {
        "schema_version": MECHANIC_CERTIFICATE_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION,
        "diagnostic_only": True,
        "packet_count": posterior.get("packet_count", 0),
        "posterior_entry_count": posterior.get("posterior_entry_count", 0),
        "confirmed_count": posterior.get("confirmed_count", 0),
        "falsified_count": posterior.get("falsified_count", 0),
        "ambiguous_count": posterior.get("ambiguous_count", 0),
        "insufficient_evidence_count": posterior.get(
            "insufficient_evidence_count", 0
        ),
        "score_linked_confirmed_count": posterior.get(
            "score_linked_confirmed_count", 0
        ),
        "unique_confirmed_game_count": posterior.get(
            "unique_confirmed_game_count", 0
        ),
        "competing_confirmed_game_count": posterior.get(
            "competing_confirmed_game_count", 0
        ),
        "competing_confirmed_games": list(
            posterior.get("competing_confirmed_games") or []
        ),
        "planner_connection_admitted_count": counts[
            "planner_connection_admitted_count"
        ],
        "planner_connection_blocked_count": counts[
            "planner_connection_blocked_count"
        ],
        "blocked_competing_confirmed_count": counts[
            "blocked_competing_confirmed_count"
        ],
        "blocked_ambiguous_count": counts["blocked_ambiguous_count"],
        "blocked_falsified_count": counts["blocked_falsified_count"],
        "blocked_insufficient_evidence_count": counts[
            "blocked_insufficient_evidence_count"
        ],
        "blocked_other_count": counts["blocked_other_count"],
        "admitted_entries": admitted_entries[: max(0, int(entry_limit))],
        "blocked_entries": blocked_entries[: max(0, int(entry_limit))],
        "promotion_state": promotion_state,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
    }


def _mechanic_certificate_posterior_state(counter: Counter[str]) -> tuple[str, str]:
    observations = counter["observation_count"]
    certified = counter["after_certified_count"]
    rejected = counter["after_rejected_count"]
    score_certified = counter["score_positive_certified_count"]
    score_status_change = counter["score_linked_status_change_count"]
    if observations <= 0:
        return "insufficient_evidence", "no_observations"
    if score_certified > 0 or score_status_change > 0:
        if certified > 0:
            if rejected > 0:
                return "confirmed", "score_positive_certified_mixed_trace"
            return "confirmed", "score_positive_certified"
    if certified > 0 and rejected == 0:
        return "confirmed", "stable_certified"
    if certified == 0 and rejected > 0:
        return "falsified", "rejected_only"
    if certified > 0 and rejected > 0:
        return "ambiguous", "mixed_certificate_status"
    return "insufficient_evidence", "no_effective_status"


def _certificate_state_from_frame(
    mechanic: str, frame: np.ndarray
) -> dict[str, Any] | None:
    try:
        from .primitive_synthesis import (
            _color_cycle_constraint_state,
            _mirror_joint_cursor_state,
            _sand_filter_flow_state,
        )
    except Exception:
        return None
    arr = _final_2d_frame(frame)
    try:
        if mechanic == "mirror_joint_cursor_bfs":
            return _mirror_joint_cursor_state(arr)
        if mechanic == "color_cycle_constraint_solver":
            return _color_cycle_constraint_state(arr)
        if mechanic == "sand_filter_flow_solver":
            return _sand_filter_flow_state(arr)
    except Exception:
        return None
    return None


def _final_2d_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    return arr


def _compact_certificate_state(
    mechanic: str, state: dict[str, Any] | None
) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    if mechanic == "mirror_joint_cursor_bfs":
        walkable = state.get("walkable")
        return _json_safe(
            {
                "cursor_a": state.get("cursor_a"),
                "cursor_b": state.get("cursor_b"),
                "grid_size": state.get("grid_size"),
                "cell": state.get("cell"),
                "origin": state.get("origin"),
                "cursor_cell_count": state.get("cursor_cell_count"),
                "walkable_count": len(walkable) if walkable is not None else 0,
            }
        )
    if mechanic == "color_cycle_constraint_solver":
        return _json_safe(
            {
                "palette_len": len(state.get("palette") or []),
                "template_count": state.get("template_count"),
                "clickable_count": state.get("clickable_count"),
                "constraint_count": state.get("constraint_count"),
                "plan_len": len(state.get("plan") or []),
                "cycle_operation_count": state.get("cycle_operation_count"),
                "mechanic_state_type": state.get("mechanic_state_type"),
                "goal_link_type": state.get("goal_link_type"),
            }
        )
    if mechanic == "sand_filter_flow_solver":
        return _json_safe(
            {
                "mechanic_state_type": state.get("mechanic_state_type"),
                "goal_link_type": state.get("goal_link_type"),
                "filter_count": state.get("filter_count"),
                "bowl_count": state.get("bowl_count"),
                "source_count": state.get("source_count"),
                "target_filter_count": len(state.get("target_filters") or []),
                "operation_plan_len": len(state.get("operation_plan") or []),
                "rotation_k": state.get("rotation_k"),
                "grid_size": state.get("grid_size"),
            }
        )
    return _json_safe(
        {
            str(key): value
            for key, value in list(state.items())[:16]
            if key not in {"walkable", "plan", "operation_plan", "constraints"}
        }
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return value


__all__ = [
    "CertificateVerdict",
    "MechanicCertificate",
    "MirrorCertificate",
    "ColorCycleCertificate",
    "SandFilterCertificate",
    "MECHANIC_CERTIFICATES",
    "MECHANIC_CERTIFICATE_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION",
    "MECHANIC_CERTIFICATE_POSTERIOR_SCHEMA_VERSION",
    "MECHANIC_CERTIFICATE_RUNTIME_TRACE_PACKET_SCHEMA_VERSION",
    "MECHANIC_CERTIFICATE_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION",
    "build_mechanic_certificate_runtime_trace_packet",
    "evaluate_certificate",
    "summarize_mechanic_certificate_guarded_planner_connection",
    "summarize_mechanic_certificate_posterior",
    "summarize_mechanic_certificate_runtime_trace_packets",
]
