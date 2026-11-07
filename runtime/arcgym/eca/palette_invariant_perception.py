"""Palette-invariant perception shadow (Phase L-6).

OBSERVATION-ONLY. This module runs ALONGSIDE the existing colour-keyed
perception and never replaces it. Its sole purpose is to MEASURE how much
each mechanic's perception depends on literal colours, by re-deriving the
certificate-relevant structure from palette-INDEPENDENT descriptors and
comparing the resulting verdict against the existing one.

The L-4b / L-5 finding: mirror transferred under a palette shift because
its perception keys on geometry, while color_cycle and sand_filter break
because their perception keys on literal colours (color_cycle: 4/5/6/8/9;
sand_filter: 8/9, 11/13, 4). The open question this audit answers: is the
ROLE structure (swatch / template / brush; filter / bowl / source)
recoverable from shape + relation alone, or is it genuinely colour-defined?

Design principle (per the L-6 directive): colour is kept as an ATTRIBUTE
(`dominant_color`, `color_rank_local`) but is NEVER the primary key for
identity. Roles are inferred from geometry and relations:
  - thin barrier / separator  -> filter-like
  - container / concavity      -> bowl-like
  - small repeated emitter     -> source-like
  - tiny symmetric pair        -> mirror cursor-like
  - small uniform block + multi-colour block + separator -> color_cycle-like

NOTHING here changes behaviour or dispatch; callers use the audit result
only for logging / reporting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Any

import numpy as np


PALETTE_INVARIANT_RUNTIME_TRACE_PACKET_SCHEMA_VERSION = (
    "eca_palette_invariant_runtime_trace_packet.v1"
)
PALETTE_INVARIANT_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION = (
    "eca_palette_invariant_runtime_trace_sensor.v1"
)
PALETTE_INVARIANT_POSTERIOR_SCHEMA_VERSION = (
    "eca_palette_invariant_perception_posterior.v1"
)
PALETTE_INVARIANT_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION = (
    "eca_palette_invariant_guarded_planner_connection.v1"
)


@dataclass(frozen=True)
class StructuralComponent:
    """A palette-independent description of one connected region.

    Colour is retained as an attribute (`dominant_color`,
    `color_rank_local`) but is not used as an identity key by any of the
    role predicates below.
    """

    bbox: tuple[int, int, int, int]  # (x0, y0, w, h) on the sampled grid
    area: int
    centroid: tuple[float, float]
    n_colors: int
    dominant_color: int
    color_rank_local: int  # rank of dominant colour by frequency (0 = bg-most)
    shape_signature: tuple[int, ...]  # palette-free occupancy hash per row
    is_thin: bool  # height == 1 or width == 1, length >= 2
    is_solid: bool  # bbox fully filled with a single colour

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]


def _background_value(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _sample_grid(frame: np.ndarray, *, cell: int) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError("frame must reduce to a 2-D grid")
    h, w = arr.shape
    gh, gw = h // cell, w // cell
    grid = np.zeros((gh, gw), dtype=int)
    for gy in range(gh):
        for gx in range(gw):
            grid[gy, gx] = int(arr[gy * cell, gx * cell])
    return grid


def extract_structural_components(
    frame: np.ndarray, *, cell: int = 2, max_components: int = 256
) -> list[StructuralComponent]:
    """Extract palette-independent connected components.

    Components are maximal 4-connected runs of non-background cells of the
    SAME colour (so a uniform swatch is one component and a multi-colour
    template is several adjacent ones). The descriptors are palette-free
    except for the retained colour attributes.
    """
    grid = _sample_grid(frame, cell=cell)
    bg = _background_value(grid)
    gh, gw = grid.shape
    # local colour-frequency ranking (palette-independent ordinal)
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    rank_of = {int(values[o]): i for i, o in enumerate(order)}

    seen = np.zeros((gh, gw), dtype=bool)
    out: list[StructuralComponent] = []
    for sy in range(gh):
        for sx in range(gw):
            if seen[sy, sx] or int(grid[sy, sx]) == bg:
                continue
            colour = int(grid[sy, sx])
            stack = [(sx, sy)]
            seen[sy, sx] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if (
                        0 <= nx < gw
                        and 0 <= ny < gh
                        and not seen[ny, nx]
                        and int(grid[ny, nx]) == colour
                    ):
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            x0, y0 = min(xs), min(ys)
            wdt = max(xs) - x0 + 1
            hgt = max(ys) - y0 + 1
            area = len(cells)
            occ = {(cx - x0, cy - y0) for cx, cy in cells}
            sig = tuple(
                sum(1 << dx for dx in range(wdt) if (dx, ry) in occ)
                for ry in range(hgt)
            )
            out.append(
                StructuralComponent(
                    bbox=(x0, y0, wdt, hgt),
                    area=area,
                    centroid=(float(sum(xs)) / area, float(sum(ys)) / area),
                    n_colors=1,
                    dominant_color=colour,
                    color_rank_local=int(rank_of.get(colour, 99)),
                    shape_signature=sig,
                    is_thin=(hgt == 1 or wdt == 1) and max(wdt, hgt) >= 2,
                    is_solid=(area == wdt * hgt),
                )
            )
            if len(out) >= max_components:
                return out
    return out


@dataclass(frozen=True)
class ShadowVerdict:
    """Palette-invariant shadow verdict for one mechanic on one frame.

    Phase L-7 makes this THREE-VALUED via `status`:
      "certified" -> high-precision positive (positive evidence, no negative)
      "rejected"  -> a negative-evidence rule fired (named competing structure)
      "unknown"   -> insufficient evidence either way
    `present` stays a bool (== certified) for backward compatibility with the
    L-6 audit. `negative_evidence` records the reject rule that fired.
    """

    present: bool
    reason: str
    features: dict[str, Any] = field(default_factory=dict)
    status: str = "unknown"  # "certified" | "rejected" | "unknown"
    negative_evidence: str = ""

    @staticmethod
    def certify(reason: str, features: dict[str, Any]) -> "ShadowVerdict":
        return ShadowVerdict(True, reason, features, "certified", "")

    @staticmethod
    def reject(reason: str, features: dict[str, Any], negative: str) -> "ShadowVerdict":
        return ShadowVerdict(False, reason, features, "rejected", negative)

    @staticmethod
    def unsure(reason: str, features: dict[str, Any]) -> "ShadowVerdict":
        return ShadowVerdict(False, reason, features, "unknown", "")


def _board_halves(comps: list[StructuralComponent]) -> int:
    """Count large comparable 'board half' components — the dual controllable
    regions a genuine mirror_x game has. Palette-free (area only)."""
    if not comps:
        return 0
    max_area = max(c.area for c in comps)
    big = sorted((c.area for c in comps if c.area >= 0.3 * max_area), reverse=True)
    if len(big) < 2:
        return len(big)
    # the top two must be comparable in mass (the two halves)
    return 2 if big[1] >= 0.6 * big[0] else 1


def shadow_mirror_verdict(frame: np.ndarray, *, cell: int = 1) -> ShadowVerdict:
    """Control mechanic, Phase L-7 three-valued. The cursor pair is two
    EQUAL-area, EQUAL-shape, same-colour SMALL solid blocks, mirror-symmetric
    across the vertical centre on the same row. NEGATIVE evidence (L-7): a
    genuine mirror_x game also has exactly TWO comparable board-half
    components (the dual controllable regions). ft09's symmetric colour blobs
    have only ONE large region -> rejected. sp80 has no clean pair -> rejected.
    """
    comps = extract_structural_components(frame, cell=cell)
    grid_w = max((c.bbox[0] + c.bbox[2] for c in comps), default=0)
    if not comps or not grid_w:
        return ShadowVerdict.unsure("no components", {})
    max_area = max(c.area for c in comps)
    halves = _board_halves(comps)
    cands = [c for c in comps if c.is_solid and c.area <= max(4, max_area // 4)]
    pair = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            if a.area != b.area or a.shape_signature != b.shape_signature:
                continue
            if a.dominant_color != b.dominant_color:
                continue
            if round(a.centroid[1]) != round(b.centroid[1]):
                continue
            ax, bx = a.centroid[0], b.centroid[0]
            if ax == bx:
                continue
            if abs((ax + bx) - (grid_w - 1)) <= 1.5:
                pair = (a, ax, bx)
                break
        if pair:
            break
    if pair is None:
        return ShadowVerdict.reject(
            f"no equal-area symmetric block pair among {len(cands)} candidates",
            {"board_halves": halves}, "no cursor pair",
        )
    a, ax, bx = pair
    feats = {
        "pair_x": (round(ax), round(bx)), "row": round(a.centroid[1]),
        "block_area": a.area, "board_halves": halves,
    }
    if halves != 2:
        # NEGATIVE evidence: symmetric pair without the dual-board context is
        # a colour-blob coincidence (ft09), not a mirror cursor pair.
        return ShadowVerdict.reject(
            f"symmetric pair but board_halves={halves} != 2 (no dual board)",
            feats, "missing dual-board context",
        )
    return ShadowVerdict.certify(
        f"equal-area({a.area}) symmetric same-row pair at "
        f"x={ax:.0f},{bx:.0f} with 2 board halves (w~{grid_w})",
        feats,
    )


def shadow_color_cycle_verdict(frame: np.ndarray, *, cell: int = 2) -> ShadowVerdict:
    """Palette-invariant color_cycle probe. A genuine colour-cycle puzzle has
    a ROW of small uniform swatch blocks (the palette legend / clickables),
    at least one small multi-colour block (the template to match), and a
    marker/applicator block — all by SHAPE and arrangement, not by colour 6.
    """
    comps = extract_structural_components(frame, cell=cell)
    # small blocks ~ 2-5 cells per side
    small = [c for c in comps if 1 <= c.width <= 6 and 1 <= c.height <= 6 and c.area >= 1]
    uniform = [c for c in small if c.is_solid]
    rows: dict[int, int] = {}
    for c in uniform:
        band = round(c.centroid[1])
        rows[band] = rows.get(band, 0) + 1
    swatch_row = max(rows.values(), default=0)
    template_like = sum(
        1 for c in small if c.height >= 2 and c.width >= 2
    )
    feats = {
        "swatch_row_max": swatch_row,
        "template_like": template_like,
        "uniform_small": len(uniform),
    }
    reason = (
        f"swatch_row_max={swatch_row}, template_like={template_like}, "
        f"uniform_small={len(uniform)}"
    )
    # Phase L-7 NEGATIVE evidence: a genuine colour-cycle puzzle has a DENSE
    # legend — many swatches in a row AND many template blocks. ft09/gtf0/gtf1
    # measure (10, 36); m0r0 (2, 2) and sp80 (1, 3) are far below and must be
    # rejected. Thresholds sit in the wide gap so the cut is robust.
    if swatch_row >= 4 and template_like >= 6:
        return ShadowVerdict.certify(reason, feats)
    if swatch_row >= 2 or template_like >= 4:
        return ShadowVerdict.reject(
            reason, feats,
            f"too sparse for a cycle legend (swatch_row_max={swatch_row} < 4 "
            f"or template_like={template_like} < 6)",
        )
    return ShadowVerdict.reject(reason, feats, "no swatch legend structure")


def shadow_sand_filter_verdict(frame: np.ndarray, *, cell: int = 4) -> ShadowVerdict:
    """Palette-invariant sand_filter probe. Roles by shape/relation:
      filter  = thin horizontal barrier (height 1, width >= 2)
      bowl    = container/concavity (wider-than-tall block low in the frame)
      source  = small emitter component high in the frame
    Colour is not used to assign roles.
    """
    comps = extract_structural_components(frame, cell=cell)
    grid_h = max((c.bbox[1] + c.bbox[3] for c in comps), default=1)
    thin_bars = [c for c in comps if c.is_thin and c.height == 1 and c.width >= 2]
    bowls = [
        c for c in comps
        if c.centroid[1] > grid_h * 0.5 and c.width >= 2 and c.width >= c.height
    ]
    sources = [c for c in comps if c.centroid[1] < grid_h / 3.0 and c.area <= 4]
    nf, nb, ns = len(thin_bars), len(bowls), len(sources)
    feats = {"thin_bars": nf, "bowls": nb, "sources": ns}
    reason = f"thin_bars={nf}, bowls={nb}, sources={ns} (grid_h~{grid_h})"
    if nf == 0 or nb == 0 or ns == 0:
        return ShadowVerdict.reject(reason, feats, "missing a sand role (filter/bowl/source)")
    # Phase L-7 NEGATIVE evidence: a genuine sand-filter puzzle has BOUNDED
    # role counts (the SandFilterCertificate bounds: filters<=4, bowls<=5,
    # sources<=3). ft09's colour grid reads as thin_bars=9/bowls=8/sources=18
    # -- far over bound -> a colour-field mis-read, not a flow puzzle.
    if nf > 4 or nb > 5 or ns > 3:
        return ShadowVerdict.reject(
            reason, feats,
            f"role counts exceed sand bounds (filters={nf}>4 or bowls={nb}>5 "
            f"or sources={ns}>3): colour-field mis-read",
        )
    return ShadowVerdict.certify(reason, feats)


SHADOW_PROBES = {
    "mirror_joint_cursor_bfs": shadow_mirror_verdict,
    "color_cycle_constraint_solver": shadow_color_cycle_verdict,
    "sand_filter_flow_solver": shadow_sand_filter_verdict,
}


@dataclass(frozen=True)
class ShadowAuditResult:
    mechanic: str
    existing_certified: bool
    shadow_present: bool
    agree: bool
    shadow_reason: str
    shadow_features: dict[str, Any]
    shadow_status: str = "unknown"  # certified | rejected | unknown (L-7)
    shadow_negative_evidence: str = ""


def audit_palette_invariance(
    frame: np.ndarray, mechanic: str, existing_certified: bool
) -> ShadowAuditResult | None:
    """Compare the existing colour-keyed verdict against the palette-invariant
    shadow probe for one mechanic. Returns None if the mechanic has no shadow
    probe. Pure measurement — no side effects, no behaviour change."""
    probe = SHADOW_PROBES.get(mechanic)
    if probe is None:
        return None
    verdict = probe(np.asarray(frame))
    return ShadowAuditResult(
        mechanic=mechanic,
        existing_certified=bool(existing_certified),
        shadow_present=bool(verdict.present),
        agree=(bool(existing_certified) == bool(verdict.present)),
        shadow_reason=verdict.reason,
        shadow_features=dict(verdict.features),
        shadow_status=verdict.status,
        shadow_negative_evidence=verdict.negative_evidence,
    )


def build_palette_invariant_runtime_trace_packet(
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
    """Emit palette-invariant role verdicts for one public transition.

    The packet is observational only. It records whether palette-free role
    probes are stable or change across a transition; it does not dispatch,
    rank, or execute any solver.
    """

    before = np.asarray(before_frame)
    after = np.asarray(after_frame)
    mechanic_names = tuple(mechanics or tuple(SHADOW_PROBES.keys()))
    verdict_rows: list[dict[str, Any]] = []
    after_status_counts: Counter[str] = Counter()
    before_status_counts: Counter[str] = Counter()
    status_changed_count = 0
    certified_after_mechanics: list[str] = []
    rejected_after_mechanics: list[str] = []
    unknown_after_mechanics: list[str] = []

    for mechanic in mechanic_names:
        if mechanic not in SHADOW_PROBES:
            continue
        before_verdict = _safe_shadow_verdict(mechanic, before)
        after_verdict = _safe_shadow_verdict(mechanic, after)
        before_status = str(before_verdict.status or "unknown")
        after_status = str(after_verdict.status or "unknown")
        changed = before_status != after_status
        status_changed_count += int(changed)
        before_status_counts[before_status] += 1
        after_status_counts[after_status] += 1
        if after_status == "certified":
            certified_after_mechanics.append(mechanic)
        elif after_status == "rejected":
            rejected_after_mechanics.append(mechanic)
        else:
            unknown_after_mechanics.append(mechanic)
        verdict_rows.append(
            {
                "mechanic": mechanic,
                "before_present": bool(before_verdict.present),
                "after_present": bool(after_verdict.present),
                "before_status": before_status,
                "after_status": after_status,
                "status_changed": bool(changed),
                "before_reason": str(before_verdict.reason or ""),
                "after_reason": str(after_verdict.reason or ""),
                "before_negative_evidence": str(
                    before_verdict.negative_evidence or ""
                ),
                "after_negative_evidence": str(after_verdict.negative_evidence or ""),
                "before_features": dict(before_verdict.features),
                "after_features": dict(after_verdict.features),
            }
        )

    return _json_safe(
        {
            "schema_version": PALETTE_INVARIANT_RUNTIME_TRACE_PACKET_SCHEMA_VERSION,
            "sensor_family": "palette_invariant_perception",
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
            "unknown_after_count": int(len(unknown_after_mechanics)),
            "certified_after_mechanics": certified_after_mechanics,
            "rejected_after_mechanics": rejected_after_mechanics,
            "unknown_after_mechanics": unknown_after_mechanics,
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


def summarize_palette_invariant_runtime_trace_packets(
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
            == PALETTE_INVARIANT_RUNTIME_TRACE_PACKET_SCHEMA_VERSION
            and str(packet.get("sensor_family", "") or "")
            == "palette_invariant_perception"
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
        unknown_after_count = _safe_int(packet.get("unknown_after_count"))
        counts["status_changed_count"] += status_changed_count
        counts["certified_after_count"] += certified_after_count
        counts["rejected_after_count"] += rejected_after_count
        counts["unknown_after_count"] += unknown_after_count
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
            before_status = str(verdict.get("before_status", "") or "unknown")
            after_status = str(verdict.get("after_status", "") or "unknown")
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
        promotion_state = "palette_invariant_runtime_trace_score_linked_status_change_observed"
    elif counts["packet_with_status_change_count"] > 0:
        promotion_state = "palette_invariant_runtime_trace_status_change_observed"
    elif counts["packet_with_certified_after_count"] > 0:
        promotion_state = "palette_invariant_runtime_trace_certified_roles_observed"
    elif counts["packet_count"] > 0:
        promotion_state = "palette_invariant_runtime_trace_connection_observed"
    else:
        promotion_state = "palette_invariant_runtime_trace_no_packets"

    return {
        "schema_version": PALETTE_INVARIANT_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION,
        "diagnostic_only": True,
        "packet_count": counts["packet_count"],
        "accepted_packet_count": counts["accepted_packet_count"],
        "rejected_packet_count": counts["rejected_packet_count"],
        "status_changed_count": counts["status_changed_count"],
        "packet_with_status_change_count": counts[
            "packet_with_status_change_count"
        ],
        "certified_after_count": counts["certified_after_count"],
        "rejected_after_count": counts["rejected_after_count"],
        "unknown_after_count": counts["unknown_after_count"],
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


def summarize_palette_invariant_posterior(
    packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    min_observations: int = 3,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Aggregate runtime role verdicts into a shadow-only mechanic posterior."""

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, dict)]
    counts: Counter[str] = Counter()
    by_key: dict[tuple[str, str], Counter[str]] = {}
    samples: list[dict[str, Any]] = []
    min_obs = max(1, int(min_observations))

    for packet in packet_rows:
        accepted = (
            str(packet.get("schema_version", "") or "")
            == PALETTE_INVARIANT_RUNTIME_TRACE_PACKET_SCHEMA_VERSION
            and str(packet.get("sensor_family", "") or "")
            == "palette_invariant_perception"
        )
        counts["accepted_packet_count" if accepted else "rejected_packet_count"] += 1
        if not accepted:
            continue
        counts["packet_count"] += 1
        game_id = str(packet.get("game_id", "") or "unknown")
        score_delta = _safe_int(packet.get("score_delta"))
        packet_score_linked = bool(packet.get("score_linked_status_change_candidate"))
        for verdict in list(packet.get("verdicts") or []):
            if not isinstance(verdict, dict):
                continue
            mechanic = str(verdict.get("mechanic", "") or "unknown")
            key = (game_id, mechanic)
            counter = by_key.setdefault(key, Counter())
            before_status = str(verdict.get("before_status", "") or "unknown")
            after_status = str(verdict.get("after_status", "") or "unknown")
            status_changed = bool(verdict.get("status_changed"))
            counter["observation_count"] += 1
            counter[f"after_{after_status}_count"] += 1
            counter["status_changed_count"] += int(status_changed)
            counter["score_delta_sum"] += score_delta
            counter["score_positive_observation_count"] += int(score_delta > 0)
            counter["score_positive_certified_count"] += int(
                score_delta > 0 and after_status == "certified"
            )
            counter["score_linked_status_change_count"] += int(
                packet_score_linked and status_changed
            )
            counter[f"transition::{before_status}->{after_status}"] += 1

    entries: list[dict[str, Any]] = []
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}
    posterior_state_counts: Counter[str] = Counter()
    support_kind_counts: Counter[str] = Counter()

    for (game_id, mechanic), counter in sorted(by_key.items()):
        obs = int(counter.get("observation_count", 0))
        certified = int(counter.get("after_certified_count", 0))
        rejected = int(counter.get("after_rejected_count", 0))
        unknown = int(counter.get("after_unknown_count", 0))
        score_linked = int(counter.get("score_linked_status_change_count", 0))
        score_positive_certified = int(counter.get("score_positive_certified_count", 0))
        if obs < min_obs:
            posterior_state = "insufficient_evidence"
            support_kind = "below_min_observations"
        elif score_linked > 0 and certified >= rejected:
            posterior_state = "confirmed"
            support_kind = "score_linked_status_change"
        elif score_positive_certified > 0 and certified >= rejected:
            posterior_state = "confirmed"
            support_kind = "score_positive_certified"
        elif score_linked > 0 or score_positive_certified > 0:
            posterior_state = "ambiguous"
            support_kind = "score_linked_mixed_shadow_status"
        elif certified >= min_obs and certified >= rejected:
            posterior_state = "confirmed"
            support_kind = "stable_certified"
        elif rejected >= min_obs and certified == 0:
            posterior_state = "falsified"
            support_kind = "rejected_only"
        elif certified > 0 and rejected > 0:
            posterior_state = "ambiguous"
            support_kind = "mixed_shadow_status"
        else:
            posterior_state = "insufficient_evidence"
            support_kind = "weak_or_unknown_status"

        posterior_state_counts[posterior_state] += 1
        support_kind_counts[support_kind] += 1
        game_counter = by_game.setdefault(game_id, Counter())
        mechanic_counter = by_mechanic.setdefault(mechanic, Counter())
        game_counter["posterior_entry_count"] += 1
        game_counter[f"{posterior_state}_count"] += 1
        mechanic_counter["posterior_entry_count"] += 1
        mechanic_counter[f"{posterior_state}_count"] += 1

        entry = {
            "posterior_key": f"{game_id}:{mechanic}",
            "game_id": game_id,
            "mechanic": mechanic,
            "posterior_state": posterior_state,
            "support_kind": support_kind,
            "observation_count": obs,
            "after_certified_count": certified,
            "after_rejected_count": rejected,
            "after_unknown_count": unknown,
            "status_changed_count": int(counter.get("status_changed_count", 0)),
            "score_linked_status_change_count": score_linked,
            "score_positive_observation_count": int(
                counter.get("score_positive_observation_count", 0)
            ),
            "score_positive_certified_count": score_positive_certified,
            "score_delta_sum": int(counter.get("score_delta_sum", 0)),
        }
        entries.append(entry)
        if len(samples) < sample_limit:
            samples.append(entry)

    unique_confirmed_games = sorted(
        game
        for game, counter in by_game.items()
        if int(counter.get("confirmed_count", 0)) == 1
    )
    competing_confirmed_games = sorted(
        game
        for game, counter in by_game.items()
        if int(counter.get("confirmed_count", 0)) > 1
    )

    if (
        posterior_state_counts["confirmed"]
        and support_kind_counts["score_linked_status_change"]
        and competing_confirmed_games
    ):
        promotion_state = (
            "palette_invariant_posterior_score_linked_confirmed_with_competing_roles"
        )
    elif posterior_state_counts["confirmed"] and support_kind_counts[
        "score_linked_status_change"
    ]:
        promotion_state = "palette_invariant_posterior_score_linked_confirmed"
    elif posterior_state_counts["confirmed"]:
        promotion_state = "palette_invariant_posterior_confirmed"
    elif posterior_state_counts["ambiguous"]:
        promotion_state = "palette_invariant_posterior_ambiguous_only"
    elif entries:
        promotion_state = "palette_invariant_posterior_no_confirmed_entries"
    else:
        promotion_state = "palette_invariant_posterior_no_packets"

    return {
        "schema_version": PALETTE_INVARIANT_POSTERIOR_SCHEMA_VERSION,
        "diagnostic_only": True,
        "packet_count": counts["packet_count"],
        "accepted_packet_count": counts["accepted_packet_count"],
        "rejected_packet_count": counts["rejected_packet_count"],
        "posterior_entry_count": len(entries),
        "confirmed_count": posterior_state_counts["confirmed"],
        "falsified_count": posterior_state_counts["falsified"],
        "ambiguous_count": posterior_state_counts["ambiguous"],
        "insufficient_evidence_count": posterior_state_counts[
            "insufficient_evidence"
        ],
        "score_linked_confirmed_count": support_kind_counts[
            "score_linked_status_change"
        ],
        "stable_certified_count": support_kind_counts["stable_certified"],
        "posterior_state_counts": dict(sorted(posterior_state_counts.items())),
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
        "entries": entries,
        "samples": samples,
        "promotion_state": promotion_state,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
    }


def summarize_palette_invariant_guarded_planner_connection(
    packets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
    *,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Shadow-only guard from palette posterior to planner-readiness evidence."""

    posterior = summarize_palette_invariant_posterior(packets)
    entries = [
        dict(entry)
        for entry in list(posterior.get("entries") or [])
        if isinstance(entry, dict)
    ]
    competing_games = set(posterior.get("competing_confirmed_games") or [])
    counts: Counter[str] = Counter()
    admitted_entries: list[dict[str, Any]] = []
    blocked_entries: list[dict[str, Any]] = []
    by_game: dict[str, Counter[str]] = {}
    by_mechanic: dict[str, Counter[str]] = {}

    for entry in entries:
        game_id = str(entry.get("game_id", "") or "unknown")
        mechanic = str(entry.get("mechanic", "") or "unknown")
        state = str(entry.get("posterior_state", "") or "insufficient_evidence")
        support_kind = str(entry.get("support_kind", "") or "")
        game_counter = by_game.setdefault(game_id, Counter())
        mechanic_counter = by_mechanic.setdefault(mechanic, Counter())
        if state == "confirmed" and game_id not in competing_games:
            decision = "admitted"
            reason = "unique_confirmed_role"
            counts["planner_connection_admitted_count"] += 1
            counts[f"admitted_support::{support_kind}"] += 1
            game_counter["planner_connection_admitted_count"] += 1
            mechanic_counter["planner_connection_admitted_count"] += 1
            admitted_entries.append(
                {
                    "posterior_key": str(entry.get("posterior_key", "") or ""),
                    "game_id": game_id,
                    "mechanic": mechanic,
                    "support_kind": support_kind,
                    "observation_count": _safe_int(entry.get("observation_count")),
                    "after_certified_count": _safe_int(
                        entry.get("after_certified_count")
                    ),
                    "after_rejected_count": _safe_int(
                        entry.get("after_rejected_count")
                    ),
                    "score_linked_status_change_count": _safe_int(
                        entry.get("score_linked_status_change_count")
                    ),
                    "guard_decision": decision,
                    "guard_reason": reason,
                }
            )
            continue

        if state == "confirmed" and game_id in competing_games:
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
            reason = "blocked_unrecognized_posterior_state"
            counts["blocked_other_count"] += 1
        counts["planner_connection_blocked_count"] += 1
        game_counter["planner_connection_blocked_count"] += 1
        mechanic_counter["planner_connection_blocked_count"] += 1
        if len(blocked_entries) < sample_limit:
            blocked_entries.append(
                {
                    "posterior_key": str(entry.get("posterior_key", "") or ""),
                    "game_id": game_id,
                    "mechanic": mechanic,
                    "posterior_state": state,
                    "support_kind": support_kind,
                    "guard_decision": "blocked",
                    "guard_reason": reason,
                }
            )

    admitted_count = counts["planner_connection_admitted_count"]
    blocked_count = counts["planner_connection_blocked_count"]
    if admitted_count and counts["blocked_competing_confirmed_count"]:
        promotion_state = (
            "palette_invariant_guarded_planner_connection_admitted_with_competing_blocks"
        )
    elif admitted_count:
        promotion_state = "palette_invariant_guarded_planner_connection_shadow_admitted"
    elif blocked_count:
        promotion_state = "palette_invariant_guarded_planner_connection_all_blocked"
    else:
        promotion_state = "palette_invariant_guarded_planner_connection_no_entries"

    return {
        "schema_version": PALETTE_INVARIANT_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION,
        "diagnostic_only": True,
        "posterior_source_schema_version": str(posterior.get("schema_version", "") or ""),
        "packet_count": _safe_int(posterior.get("packet_count")),
        "posterior_entry_count": len(entries),
        "confirmed_count": _safe_int(posterior.get("confirmed_count")),
        "falsified_count": _safe_int(posterior.get("falsified_count")),
        "ambiguous_count": _safe_int(posterior.get("ambiguous_count")),
        "insufficient_evidence_count": _safe_int(
            posterior.get("insufficient_evidence_count")
        ),
        "competing_confirmed_game_count": _safe_int(
            posterior.get("competing_confirmed_game_count")
        ),
        "unique_confirmed_game_count": _safe_int(
            posterior.get("unique_confirmed_game_count")
        ),
        "planner_connection_admitted_count": admitted_count,
        "planner_connection_blocked_count": blocked_count,
        "blocked_competing_confirmed_count": counts[
            "blocked_competing_confirmed_count"
        ],
        "blocked_ambiguous_count": counts["blocked_ambiguous_count"],
        "blocked_falsified_count": counts["blocked_falsified_count"],
        "blocked_insufficient_evidence_count": counts[
            "blocked_insufficient_evidence_count"
        ],
        "blocked_other_count": counts["blocked_other_count"],
        "admitted_support_kind_counts": {
            key.removeprefix("admitted_support::"): int(value)
            for key, value in sorted(counts.items())
            if key.startswith("admitted_support::")
        },
        "by_game": {
            game: dict(sorted(counter.items()))
            for game, counter in sorted(by_game.items())
        },
        "by_mechanic": {
            mechanic: dict(sorted(counter.items()))
            for mechanic, counter in sorted(by_mechanic.items())
        },
        "admitted_entries": admitted_entries[:sample_limit],
        "blocked_entries": blocked_entries,
        "promotion_state": promotion_state,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
    }


def _safe_shadow_verdict(mechanic: str, frame: np.ndarray) -> ShadowVerdict:
    probe = SHADOW_PROBES[mechanic]
    try:
        return probe(frame)
    except Exception as exc:
        return ShadowVerdict.unsure(
            f"{type(exc).__name__}: {exc}",
            {"error": f"{type(exc).__name__}: {exc}"},
        )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "PALETTE_INVARIANT_RUNTIME_TRACE_PACKET_SCHEMA_VERSION",
    "PALETTE_INVARIANT_RUNTIME_TRACE_SENSOR_SCHEMA_VERSION",
    "PALETTE_INVARIANT_POSTERIOR_SCHEMA_VERSION",
    "PALETTE_INVARIANT_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION",
    "StructuralComponent",
    "extract_structural_components",
    "ShadowVerdict",
    "ShadowAuditResult",
    "shadow_mirror_verdict",
    "shadow_color_cycle_verdict",
    "shadow_sand_filter_verdict",
    "SHADOW_PROBES",
    "audit_palette_invariance",
    "build_palette_invariant_runtime_trace_packet",
    "summarize_palette_invariant_runtime_trace_packets",
    "summarize_palette_invariant_posterior",
    "summarize_palette_invariant_guarded_planner_connection",
]
