"""Generic Semantic API for Task Adapters.

Core control (ControlIR, LocalController, aftercare, PSA, GCSTM) consumes
only these generic data structures.  Task-specific raw signals stay inside
adapters and are never referenced by core code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Semantic Event kinds (generic vocabulary) ─────────────────────────────

EVENT_AUX_STATE_MUTATION = "aux_state_mutation"
EVENT_GOAL_REGION_CANDIDATE = "goal_region_candidate"
EVENT_LOW_PROGRESS_CYCLE = "low_progress_cycle"
EVENT_INTERACTIVE_TILE_EFFECT = "interactive_tile_effect_hypothesis"
EVENT_ROUTE_TO_GOAL_UNKNOWN = "route_to_goal_unknown"
EVENT_ROUTE_PROGRESS_GAIN = "route_progress_gain"
EVENT_GOAL_APPROACH_GAIN = "goal_approach_gain"
EVENT_GOAL_ENTRY_CANDIDATE = "goal_entry_candidate"
EVENT_ROUTE_GATE_CANDIDATE = "route_gate_candidate"
EVENT_ACTION_BLOCK_CANDIDATE = "action_block_candidate"
EVENT_STATE_DISPLAY_CHANGE = "state_display_change"
# Semantics-lane events: abstract mechanics hints (no exact coords)
EVENT_GOAL_SLOT_BLOCKED = "goal_slot_blocked"                     # blocked by goal that needs state match
EVENT_STATE_MISMATCH = "state_mismatch"                           # current state doesn't match goal requirement
EVENT_STATE_TRANSITION_SITE_CANDIDATE = "state_transition_site_candidate"
EVENT_STATE_CHANGER_REACHABLE = "state_changer_reachable"         # a state-changing tile is reachable
EVENT_STATE_CHANGED_TOWARD_GOAL = "state_changed_toward_goal"     # state moved closer to goal requirement
EVENT_GOAL_PASSABLE = "goal_passable"                             # state now matches, goal should be passable
EVENT_SITE_APPROACH_GAIN = "site_approach_gain"                   # branch-internal movement got closer to transition site
EVENT_SITE_DISTANCE_REDUCED = "site_distance_reduced"             # abstract distance-to-site decreased
EVENT_BRANCH_INTERNAL_CYCLE = "branch_internal_cycle"             # stalled or oscillating while inside branch
EVENT_BRANCH_EXIT_REGRESSION = "branch_exit_regression"           # drifted back toward corridor before reaching site
EVENT_LATENT_TRANSITION_INFERRED = "latent_transition_inferred"   # contact at site → infer state changed (no display proof needed)
EVENT_POST_CONTACT_GATE_RETEST = "post_contact_gate_retest"       # after contact, retest the blocked gate route

# v15.1 — observation-first status / level-delta infrastructure.
# Names are deliberately neutral: the adapter emits raw scalar
# signals and raw diffs, and the LLM assigns meaning (step counter,
# lives, recovery item, death, etc.). No "vitality" / "death" labels
# are asserted at this layer — those would preempt interpretation.
EVENT_SCALAR_SIGNAL_CHANGE = "scalar_signal_change"               # a scalar count derived from the board changed
EVENT_SCALAR_SIGNAL_RESET = "scalar_signal_reset"                 # scalar jumped back to a prior observed max
EVENT_CRITICAL_SIGNAL_TRANSITION = "critical_signal_transition"   # one scalar reset while another simultaneously decreased
EVENT_LEVEL_SYMBOL_INVENTORY = "level_symbol_inventory"           # raw char-type inventory snapshot at level entry
EVENT_LEVEL_DELTA_CHARS = "level_delta_chars"                     # new/removed character types across level transition

# Entity-level observation events — schema for causal hypothesis layer.
# Still observation-first: no "enemy" / "item" / "trap" labels, only
# "an entity of some type was seen / contacted / disappeared".
EVENT_LEVEL_ENTERED = "level_entered"                             # new level_id became active
EVENT_ENTITY_CONTACT = "entity_contact"                           # player position coincided with a non-wall entity
EVENT_ENTITY_DISAPPEARED = "entity_disappeared"                   # a previously observed entity is no longer on board
EVENT_ENTITY_SEEN_FIRST_IN_LEVEL = "entity_seen_first_in_level"   # entity type first observed within current level

# Phase 4a — raw observation event reserved for Phase 4c adapter wiring.
# A scalar signal *increased* relative to the ring-buffer baseline
# (e.g. step counter restored after pickup). Distinct from
# scalar_signal_reset, which specifically denotes a jump back to a
# prior observed maximum; scalar_signal_up covers monotonic rises
# that are not necessarily returns-to-max.
EVENT_SCALAR_SIGNAL_UP = "scalar_signal_up"


@dataclass
class KnowledgeProvenance:
    """Describes where a piece of task knowledge came from."""
    mode: str = ""
    source_type: str = ""
    ref: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source_type": self.source_type,
            "ref": self.ref,
            "detail": self.detail,
        }


@dataclass
class SemanticEvent:
    """A single generic semantic event emitted by an adapter."""
    kind: str                           # one of EVENT_* constants
    confidence: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "payload": self.payload,
        }


@dataclass
class RegionCandidate:
    """A spatial region with a hypothesized role."""
    role: str                           # goal_region / route_gate / mutable_display / unknown
    bbox: tuple[int, int, int, int]     # (row_min, row_max, col_min, col_max)
    signature: str = ""                 # hash or short description
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "bbox": list(self.bbox),
            "signature": self.signature,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class RouteHypothesis:
    """A hypothesized route between regions."""
    from_region: str = ""
    to_region: str = ""
    direction: str = ""                 # up / down / left / right / unknown
    estimated_steps: int = -1
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_region": self.from_region,
            "to_region": self.to_region,
            "direction": self.direction,
            "estimated_steps": self.estimated_steps,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ProbeRecommendation:
    """A recommended probe mode and intent."""
    mode: str = ""                      # state_discovery / route_discovery / goal_conditioned_execution
    intent: str = ""                    # human-readable short description
    priority: float = 0.0
    suggested_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "intent": self.intent,
            "priority": round(self.priority, 3),
            "suggested_actions": self.suggested_actions,
            "metadata": dict(self.metadata),
        }


@dataclass
class AdapterRecoveryPlan:
    """Adapter-owned bounded recovery queue."""
    source: str = ""
    intent: str = ""
    actions: list[str] = field(default_factory=list)
    direct_load_on_queue_exhausted: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    knowledge_provenance: list[KnowledgeProvenance] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "intent": self.intent,
            "actions": list(self.actions),
            "direct_load_on_queue_exhausted": bool(self.direct_load_on_queue_exhausted),
            "metadata": dict(self.metadata),
            "knowledge_provenance": [item.to_dict() for item in self.knowledge_provenance],
        }


@dataclass
class AdapterPlannerPriors:
    """Optional adapter priors for generic routing/ranking."""
    corridor_col: int | None = None
    junction_row: int | None = None
    goal_row_max: int | None = None
    passage_cols: list[int] = field(default_factory=list)
    frontier_rows: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    knowledge_provenance: list[KnowledgeProvenance] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corridor_col": self.corridor_col,
            "junction_row": self.junction_row,
            "goal_row_max": self.goal_row_max,
            "passage_cols": list(self.passage_cols),
            "frontier_rows": list(self.frontier_rows),
            "metadata": dict(self.metadata),
            "knowledge_provenance": [item.to_dict() for item in self.knowledge_provenance],
        }


@dataclass
class AdapterSequenceSpec:
    """Named short sequence plus optional verify follow-up."""
    name: str = ""
    actions: list[str] = field(default_factory=list)
    contact_step_index: int | None = None
    verify_followup_name: str = ""
    verify_followup_actions: list[str] = field(default_factory=list)
    action_bias: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    knowledge_provenance: list[KnowledgeProvenance] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "actions": list(self.actions),
            "contact_step_index": self.contact_step_index,
            "verify_followup_name": self.verify_followup_name,
            "verify_followup_actions": list(self.verify_followup_actions),
            "action_bias": dict(self.action_bias),
            "metadata": dict(self.metadata),
            "knowledge_provenance": [item.to_dict() for item in self.knowledge_provenance],
        }


@dataclass
class AdapterScoreResult:
    """Adapter-owned notebook scoring payload."""
    adapter_name: str = ""
    score_kind: str = ""
    total_score: float = 0.0
    score_pct: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)
    knowledge_provenance: list[KnowledgeProvenance] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "score_kind": self.score_kind,
            "total_score": round(self.total_score, 3),
            "score_pct": round(self.score_pct, 3),
            "payload": dict(self.payload),
            "knowledge_provenance": [item.to_dict() for item in self.knowledge_provenance],
        }


@dataclass
class AdapterExecutionTicket:
    """Adapter-owned typed execution ticket for hidden-state control."""
    kind: str = ""
    phase: str = ""
    ownership: str = ""
    ttl_steps: int = 12
    stall_budget: int = 2
    route_preview: list[str] = field(default_factory=list)
    target_anchor: list[int] = field(default_factory=list)
    site_anchor: list[int] = field(default_factory=list)
    goal_anchor: list[int] = field(default_factory=list)
    provenance: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    knowledge_provenance: list[KnowledgeProvenance] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "phase": self.phase,
            "ownership": self.ownership,
            "ttl_steps": int(self.ttl_steps),
            "stall_budget": int(self.stall_budget),
            "route_preview": list(self.route_preview),
            "target_anchor": list(self.target_anchor),
            "site_anchor": list(self.site_anchor),
            "goal_anchor": list(self.goal_anchor),
            "provenance": self.provenance,
            "metadata": dict(self.metadata),
            "knowledge_provenance": [item.to_dict() for item in self.knowledge_provenance],
        }

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "AdapterExecutionTicket | None":
        if not isinstance(raw, dict) or not raw:
            return None

        def _normalize_anchor(value: Any) -> list[int]:
            if not isinstance(value, (list, tuple)):
                return []
            return [
                int(item)
                for item in list(value)
                if str(item).strip("-").isdigit()
            ]

        def _normalize_actions(value: Any) -> list[str]:
            return [
                str(action or "").strip().upper()
                for action in list(value or [])
                if str(action or "").strip()
            ]

        knowledge_items: list[KnowledgeProvenance] = []
        for item in list(raw.get("knowledge_provenance", []) or []):
            if not isinstance(item, dict):
                continue
            knowledge_items.append(
                KnowledgeProvenance(
                    mode=str(item.get("mode", "") or ""),
                    source_type=str(item.get("source_type", "") or ""),
                    ref=str(item.get("ref", "") or ""),
                    detail=str(item.get("detail", "") or ""),
                )
            )

        return cls(
            kind=str(raw.get("kind", "") or "").strip().lower(),
            phase=str(raw.get("phase", "") or "").strip().lower(),
            ownership=str(raw.get("ownership", "") or "").strip().lower(),
            ttl_steps=max(1, int(raw.get("ttl_steps", 12) or 12)),
            stall_budget=max(0, int(raw.get("stall_budget", 2) or 2)),
            route_preview=_normalize_actions(raw.get("route_preview", [])),
            target_anchor=_normalize_anchor(raw.get("target_anchor")),
            site_anchor=_normalize_anchor(raw.get("site_anchor")),
            goal_anchor=_normalize_anchor(raw.get("goal_anchor")),
            provenance=str(raw.get("provenance", "") or "").strip(),
            metadata=dict(raw.get("metadata", {}) or {}),
            knowledge_provenance=knowledge_items,
        )


@dataclass
class SemanticControlState:
    """Typed adapter-to-control boundary for generic modules."""
    mechanics_mode: str = ""
    player_bbox: tuple[int, int, int, int] | None = None
    stamp_region: tuple[int, int, int, int] | None = None
    goal_region: tuple[int, int, int, int] | None = None
    visited_rows: list[int] = field(default_factory=list)
    visited_cols: list[int] = field(default_factory=list)
    known_passage_cols: list[int] = field(default_factory=list)
    corridor_col: int | None = None
    semantic_progress: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanics_mode": self.mechanics_mode,
            "player_bbox": list(self.player_bbox) if self.player_bbox else None,
            "stamp_region": list(self.stamp_region) if self.stamp_region else None,
            "goal_region": list(self.goal_region) if self.goal_region else None,
            "visited_rows": list(self.visited_rows),
            "visited_cols": list(self.visited_cols),
            "known_passage_cols": list(self.known_passage_cols),
            "corridor_col": self.corridor_col,
            "semantic_progress": dict(self.semantic_progress),
        }


@dataclass
class SemanticObservation:
    """Complete semantic output from an adapter for one step."""
    adapter_name: str = ""
    events: list[SemanticEvent] = field(default_factory=list)
    region_candidates: list[RegionCandidate] = field(default_factory=list)
    route_hypotheses: list[RouteHypothesis] = field(default_factory=list)
    probe_recommendations: list[ProbeRecommendation] = field(default_factory=list)
    confidence: float = 0.0
    control_state: SemanticControlState | None = None
    private_debug_meta: dict[str, Any] | None = None
    knowledge_provenance: list[KnowledgeProvenance] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "adapter_name": self.adapter_name,
            "events": [e.to_dict() for e in self.events],
            "region_candidates": [r.to_dict() for r in self.region_candidates],
            "route_hypotheses": [r.to_dict() for r in self.route_hypotheses],
            "probe_recommendations": [p.to_dict() for p in self.probe_recommendations],
            "confidence": round(self.confidence, 3),
            "knowledge_provenance": [item.to_dict() for item in self.knowledge_provenance],
        }
        if self.control_state is not None:
            d["control_state"] = self.control_state.to_dict()
        return d

    def has_event(self, kind: str) -> bool:
        return any(e.kind == kind for e in self.events)

    def best_probe_mode(self) -> str:
        if not self.probe_recommendations:
            return ""
        return max(self.probe_recommendations, key=lambda p: p.priority).mode

    def render_compact(self) -> str:
        """Render a compact text block for the analyzer prompt."""
        lines: list[str] = []
        if self.events:
            lines.append("[SEMANTIC EVENTS]")
            for e in self.events:
                payload_str = ""
                if e.payload:
                    payload_str = ", " + ", ".join(
                        f"{k}={v}" for k, v in e.payload.items()
                        if not isinstance(v, dict)
                    )
                lines.append(f"  {e.kind}(conf={e.confidence:.2f}{payload_str})")
        if self.region_candidates:
            lines.append("[REGION CANDIDATES]")
            for r in self.region_candidates:
                lines.append(
                    f"  {r.role}: rows {r.bbox[0]}-{r.bbox[1]}, cols {r.bbox[2]}-{r.bbox[3]} "
                    f"(conf={r.confidence:.2f})"
                )
        if self.route_hypotheses:
            lines.append("[ROUTE HYPOTHESES]")
            for r in self.route_hypotheses:
                lines.append(
                    f"  {r.from_region} -> {r.to_region} ({r.direction}, "
                    f"~{r.estimated_steps} steps, conf={r.confidence:.2f})"
                )
        if self.probe_recommendations:
            best = max(self.probe_recommendations, key=lambda p: p.priority)
            lines.append(f"[RECOMMENDED PROBE MODE: {best.mode}]")
            lines.append(f"  intent: {best.intent}")
            if best.suggested_actions:
                lines.append(f"  suggested: {', '.join(best.suggested_actions)}")
        return "\n".join(lines)
