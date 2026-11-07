"""ECA-to-Control IR: Compile boundary from ECA semantics into executable control.

This module implements the ControlIR layer described in Section 16-18 of the
ECA Architecture Bottleneck Map (2026-04-02).

The ControlIR sits between ECA semantic state and the planner/runtime:

    ECA semantic state  ─┐
    GCSTM shadow evidence ─┤──► ControlIR ──► compact typed IR in analyzer prompt
    PSA subtype evidence ──┘                  bounded runtime hooks

Design principles:
    - Phase 1: observer-only export (emit into step metrics / episode summary)
    - Phase 2: advisory-only use (append compact IR text into analyzer prompt)
    - Never directly override the action queue
    - Keep the recovery branch readable

Ticket kinds (Section 16):
    - route_retention: post-shared route survival before no-target collapse
    - selected_target_promotion: bounded promotion from preserved candidate
    - dead_revisit_suppression: temporarily downrank preterminal_dead_revisit
    - bounded_second_chance: one-shot late conversion without resurface churn
    - continuation_attach: existing mechanism, secondary until critical-path live
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


# ─── Ticket Kinds ────────────────────────────────────────────────────────────

class TicketKind(str, Enum):
    """Enumeration of all ControlIR ticket kinds."""
    ROUTE_RETENTION = "route_retention"
    SELECTED_TARGET_PROMOTION = "selected_target_promotion"
    DEAD_REVISIT_SUPPRESSION = "dead_revisit_suppression"
    BOUNDED_SECOND_CHANCE = "bounded_second_chance"
    CONTINUATION_ATTACH = "continuation_attach"
    # Phase 3: v2 new tickets
    STALE_RETARGET = "stale_retarget"           # target is STALE → force retarget
    LOW_GAIN_DIV = "low_gain_div"               # low info-gain → diversify family
    SATURATION_RELIEF = "saturation_relief"     # candidate pool saturated → relief


class TicketState(str, Enum):
    """Lifecycle state of a ControlIR ticket."""
    ISSUED = "issued"
    ACTIVE = "active"
    ATTACHED = "attached"
    CONSUMED = "consumed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


_AUTO_ISSUE_KINDS: tuple[TicketKind, ...] = (
    TicketKind.ROUTE_RETENTION,
    TicketKind.SELECTED_TARGET_PROMOTION,
    TicketKind.DEAD_REVISIT_SUPPRESSION,
    TicketKind.BOUNDED_SECOND_CHANCE,
    TicketKind.STALE_RETARGET,
    TicketKind.LOW_GAIN_DIV,
    TicketKind.SATURATION_RELIEF,
)


# ─── Ticket Data ─────────────────────────────────────────────────────────────

@dataclass
class ControlIRTicket:
    """A single ControlIR ticket representing a bounded executable continuation artifact.

    Minimal ticket shape from Section 16:
    {
      "kind": "post_shared_route_retention",
      "source_stage": "POST_FINAL_SHARED_EXIT",
      "anchor_family": "shared_successor_family",
      "evidence": [...],
      "confidence": "tbd",
      "ttl_turns": 3,
      "promote_from": ["successor_slate", "protected_hedge"],
      "suppress_dispositions": ["preterminal_dead_revisit"],
      "success_signal": "selected_target_attached",
      "fallback": "expire_to_observability"
    }
    """
    kind: TicketKind
    state: TicketState = TicketState.ISSUED
    source_stage: str = ""
    anchor_family: str = ""
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    ttl_turns: int = 3
    remaining_turns: int = -1  # -1 means not yet activated
    issued_step: int = -1
    activated_step: int = -1
    resolved_step: int = -1
    promote_from: list[str] = field(default_factory=list)
    suppress_dispositions: list[str] = field(default_factory=list)
    success_signal: str = ""
    fallback: str = "expire_to_observability"
    resolution_reason: str = ""

    def __post_init__(self) -> None:
        if self.remaining_turns < 0:
            self.remaining_turns = self.ttl_turns

    def activate(self, step: int) -> None:
        """Transition from ISSUED to ACTIVE."""
        if self.state == TicketState.ISSUED:
            self.state = TicketState.ACTIVE
            self.activated_step = step
            self.remaining_turns = self.ttl_turns

    def tick(self) -> None:
        """Decrement remaining TTL by one turn."""
        if self.state == TicketState.ACTIVE:
            self.remaining_turns = max(0, self.remaining_turns - 1)
            if self.remaining_turns <= 0:
                self.state = TicketState.EXPIRED
                self.resolution_reason = "ttl_expired"

    def attach(self, step: int, reason: str = "") -> None:
        """Transition from ACTIVE to ATTACHED (success signal received)."""
        if self.state == TicketState.ACTIVE:
            self.state = TicketState.ATTACHED
            self.resolved_step = step
            self.resolution_reason = reason or self.success_signal

    def consume(self, step: int, reason: str = "") -> None:
        """Transition from ATTACHED to CONSUMED."""
        if self.state == TicketState.ATTACHED:
            self.state = TicketState.CONSUMED
            self.resolved_step = step
            self.resolution_reason = reason or "consumed"

    def cancel(self, step: int, reason: str = "") -> None:
        """Cancel the ticket (e.g., regression detected)."""
        if self.state in (TicketState.ISSUED, TicketState.ACTIVE, TicketState.ATTACHED):
            self.state = TicketState.CANCELLED
            self.resolved_step = step
            self.resolution_reason = reason or "cancelled"

    @property
    def is_live(self) -> bool:
        """Whether this ticket is still actionable."""
        return self.state in (TicketState.ISSUED, TicketState.ACTIVE, TicketState.ATTACHED)

    @property
    def is_terminal(self) -> bool:
        """Whether this ticket has reached a terminal state."""
        return self.state in (TicketState.CONSUMED, TicketState.EXPIRED, TicketState.CANCELLED)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for metrics/logging."""
        return {
            "kind": self.kind.value,
            "state": self.state.value,
            "source_stage": self.source_stage,
            "anchor_family": self.anchor_family,
            "evidence": list(self.evidence),
            "confidence": round(self.confidence, 4),
            "ttl_turns": self.ttl_turns,
            "remaining_turns": self.remaining_turns,
            "issued_step": self.issued_step,
            "activated_step": self.activated_step,
            "resolved_step": self.resolved_step,
            "promote_from": list(self.promote_from),
            "suppress_dispositions": list(self.suppress_dispositions),
            "success_signal": self.success_signal,
            "fallback": self.fallback,
            "resolution_reason": self.resolution_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControlIRTicket":
        """Deserialize from dictionary."""
        return cls(
            kind=TicketKind(data.get("kind", "route_retention")),
            state=TicketState(data.get("state", "issued")),
            source_stage=str(data.get("source_stage", "")),
            anchor_family=str(data.get("anchor_family", "")),
            evidence=list(data.get("evidence", [])),
            confidence=float(data.get("confidence", 0.0)),
            ttl_turns=int(data.get("ttl_turns", 3)),
            remaining_turns=int(data.get("remaining_turns", -1)),
            issued_step=int(data.get("issued_step", -1)),
            activated_step=int(data.get("activated_step", -1)),
            resolved_step=int(data.get("resolved_step", -1)),
            promote_from=list(data.get("promote_from", [])),
            suppress_dispositions=list(data.get("suppress_dispositions", [])),
            success_signal=str(data.get("success_signal", "")),
            fallback=str(data.get("fallback", "expire_to_observability")),
            resolution_reason=str(data.get("resolution_reason", "")),
        )


# ─── ControlIR Configuration ────────────────────────────────────────────────

@dataclass
class ControlIRConfig:
    """Configuration for the ControlIR layer."""
    enabled: bool = True
    mode: str = "observer"  # "observer" (Phase 1) or "advisory" (Phase 2)

    # Route retention
    route_retention_ttl: int = 3
    route_retention_min_confidence: float = 0.3

    # Selected target promotion
    promotion_ttl: int = 10
    promotion_min_candidate_pool_size: int = 1
    promotion_max_per_episode: int = 5

    # Dead revisit suppression
    suppression_ttl: int = 3
    suppression_max_active: int = 2
    suppression_requires_recent_live_evidence: bool = True

    # Bounded second chance
    second_chance_ttl: int = 8
    second_chance_max_per_episode: int = 3
    second_chance_cooldown_steps: int = 5

    # Continuation attach (secondary)
    continuation_attach_ttl: int = 4
    # Phase 3: stale_retarget
    stale_retarget_ttl: int = 5
    stale_retarget_stale_threshold_steps: int = 8   # turns without progress
    stale_retarget_max_per_episode: int = 4
    # Phase 3: low_gain_div
    low_gain_div_ttl: int = 4
    low_gain_div_min_repeat_steps: int = 5          # min steps same family repeated
    low_gain_div_max_active: int = 2
    # Phase 3: saturation_relief
    saturation_relief_ttl: int = 6
    saturation_relief_pool_saturation_ratio: float = 0.8  # fraction same family
    saturation_relief_min_pool_size: int = 4
    saturation_relief_max_per_episode: int = 3

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "ControlIRConfig":
        if not raw:
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", True)),
            mode=str(raw.get("mode", "observer")),
            route_retention_ttl=max(1, int(raw.get("route_retention_ttl", 3))),
            route_retention_min_confidence=float(raw.get("route_retention_min_confidence", 0.3)),
            promotion_ttl=max(1, int(raw.get("promotion_ttl", 10))),
            promotion_min_candidate_pool_size=max(0, int(raw.get("promotion_min_candidate_pool_size", 1))),
            promotion_max_per_episode=max(1, int(raw.get("promotion_max_per_episode", 5))),
            suppression_ttl=max(1, int(raw.get("suppression_ttl", 3))),
            suppression_max_active=max(1, int(raw.get("suppression_max_active", 2))),
            suppression_requires_recent_live_evidence=bool(
                raw.get("suppression_requires_recent_live_evidence", True)
            ),
            second_chance_ttl=max(1, int(raw.get("second_chance_ttl", 8))),
            second_chance_max_per_episode=max(1, int(raw.get("second_chance_max_per_episode", 3))),
            second_chance_cooldown_steps=max(0, int(raw.get("second_chance_cooldown_steps", 5))),
            continuation_attach_ttl=max(1, int(raw.get("continuation_attach_ttl", 4))),
            # Phase 3
            stale_retarget_ttl=max(1, int(raw.get("stale_retarget_ttl", 5))),
            stale_retarget_stale_threshold_steps=max(1, int(raw.get("stale_retarget_stale_threshold_steps", 8))),
            stale_retarget_max_per_episode=max(1, int(raw.get("stale_retarget_max_per_episode", 4))),
            low_gain_div_ttl=max(1, int(raw.get("low_gain_div_ttl", 4))),
            low_gain_div_min_repeat_steps=max(1, int(raw.get("low_gain_div_min_repeat_steps", 5))),
            low_gain_div_max_active=max(1, int(raw.get("low_gain_div_max_active", 2))),
            saturation_relief_ttl=max(1, int(raw.get("saturation_relief_ttl", 6))),
            saturation_relief_pool_saturation_ratio=float(raw.get("saturation_relief_pool_saturation_ratio", 0.8)),
            saturation_relief_min_pool_size=max(1, int(raw.get("saturation_relief_min_pool_size", 4))),
            saturation_relief_max_per_episode=max(1, int(raw.get("saturation_relief_max_per_episode", 3))),
        )


# ─── Evidence Collector ──────────────────────────────────────────────────────

@dataclass
class ControlIREvidence:
    """Collected evidence from ECA, GCSTM, and PSA for ticket issuance decisions."""
    # ECA semantic state
    post_final_shared_entry_count: int = 0
    post_final_shared_read_class: str = ""
    post_final_shared_after_exit_outcome_class: str = ""
    selected_target_empty: bool = False
    candidate_pool_size: int = 0
    candidate_pool_non_empty: bool = False
    candidate_pool_no_target_disposition: str = ""
    preterminal_dead_revisit_count: int = 0
    shared_rearm_live: bool = False
    shared_rerank_live: bool = False
    queue_exhaustion_detected: bool = False

    # GCSTM shadow evidence
    recently_live_anchor: bool = False
    recently_live_anchor_family: str = ""
    anchor_stability: float = 0.0
    dead_family_window: bool = False

    # PSA subtype evidence
    psa_subtype: str = ""
    psa_sweep_dead: bool = False
    last_viability_reason: str = ""
    last_viability_drop_reason: str = ""

    # Runtime context
    current_step: int = 0
    turns_since_shared_exit: int = -1
    phase_handoff_resurface_count: int = 0
    second_chance_kind_histogram: dict[str, int] = field(default_factory=dict)

     # Protected hedge / successor state
    protected_hedge_visible: bool = False
    successor_slate_non_empty: bool = False
    # Concrete candidate summaries for advisory text and Approach B boost
    # Format: "comp_id:score:source:family" (pipe-separated for top2)
    successor_slate_top2_summary: str = ""
    protected_hedge_summary: str = ""
    # Phase 3: TargetHealth-based evidence
    target_health: str = ""                        # TargetHealth enum value
    steps_since_target_progress: int = 0           # turns without board delta
    current_target_component_id: str = ""          # current selected target component
    current_target_source: str = ""                # current selected target source
    current_action_family: str = ""                # family of last action
    family_repeat_count: int = 0                   # consecutive same-family steps
    candidate_pool_dominant_family: str = ""       # most common family in pool
    candidate_pool_dominant_family_ratio: float = 0.0  # fraction of pool
    recent_target_shift: bool = False              # selected target changed recently
    turns_since_target_shift: int = -1             # age of the last target shift
    target_shift_from_component_id: str = ""       # previous target before the shift
    target_shift_from_family: str = ""             # previous family before the shift

    # Retained route anchor (after post_final_shared exit)
    retained_route_component_id: str = ""          # anchor component_id
    retained_route_family: str = ""                # anchor source family
    retained_route_source: str = ""                # anchor contract source (selected_target/protected_hedge)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─── ControlIR Registry ─────────────────────────────────────────────────────

class ControlIRRegistry:
    """Central registry for ControlIR tickets.

    Manages the lifecycle of all tickets and provides:
    - Ticket issuance based on evidence evaluation
    - Per-step tick/advance for TTL management
    - Observer-only export for step metrics
    - Advisory text generation for analyzer prompt
    """

    def __init__(self, config: ControlIRConfig | None = None) -> None:
        self.config = config or ControlIRConfig()
        self._tickets: list[ControlIRTicket] = []
        self._history: list[dict[str, Any]] = []

        # Episode-level counters
        self._issue_counts: dict[str, int] = {k.value: 0 for k in TicketKind}
        self._attach_counts: dict[str, int] = {k.value: 0 for k in TicketKind}
        self._consume_counts: dict[str, int] = {k.value: 0 for k in TicketKind}
        self._expire_counts: dict[str, int] = {k.value: 0 for k in TicketKind}
        self._cancel_counts: dict[str, int] = {k.value: 0 for k in TicketKind}
        self._issue_decision_histograms: dict[str, dict[str, int]] = {
            k.value: {} for k in _AUTO_ISSUE_KINDS
        }
        self._last_issue_decisions: dict[str, str] = {
            k.value: "" for k in _AUTO_ISSUE_KINDS
        }
        # Cooldown tracking
        self._last_second_chance_step: int = -999
        self._promotion_count: int = 0
        # Phase 3 counters
        self._stale_retarget_count: int = 0
        self._saturation_relief_count: int = 0

    def reset(self) -> None:
        """Reset all state for a new episode."""
        self._tickets.clear()
        self._history.clear()
        self._issue_counts = {k.value: 0 for k in TicketKind}
        self._attach_counts = {k.value: 0 for k in TicketKind}
        self._consume_counts = {k.value: 0 for k in TicketKind}
        self._expire_counts = {k.value: 0 for k in TicketKind}
        self._cancel_counts = {k.value: 0 for k in TicketKind}
        self._issue_decision_histograms = {
            k.value: {} for k in _AUTO_ISSUE_KINDS
        }
        self._last_issue_decisions = {
            k.value: "" for k in _AUTO_ISSUE_KINDS
        }
        self._last_second_chance_step = -999
        self._promotion_count = 0
        self._stale_retarget_count = 0
        self._saturation_relief_count = 0
    # ── Ticket Issuancee ──────────────────────────────────────────────────

    def issue_ticket(
        self,
        kind: TicketKind,
        *,
        step: int,
        source_stage: str = "",
        anchor_family: str = "",
        evidence: list[str] | None = None,
        confidence: float = 0.0,
        ttl_turns: int | None = None,
        promote_from: list[str] | None = None,
        suppress_dispositions: list[str] | None = None,
        success_signal: str = "",
        fallback: str = "expire_to_observability",
    ) -> ControlIRTicket | None:
        """Issue a new ticket if policy allows."""
        if not self.config.enabled:
            return None

        # Per-kind policy checks
        if kind == TicketKind.BOUNDED_SECOND_CHANCE:
            if self._issue_counts[kind.value] >= self.config.second_chance_max_per_episode:
                return None
            if (step - self._last_second_chance_step) < self.config.second_chance_cooldown_steps:
                return None

        if kind == TicketKind.SELECTED_TARGET_PROMOTION:
            if self._promotion_count >= self.config.promotion_max_per_episode:
                return None

        if kind == TicketKind.DEAD_REVISIT_SUPPRESSION:
            active_suppression = sum(
                1 for t in self._tickets
                if t.kind == TicketKind.DEAD_REVISIT_SUPPRESSION and t.is_live
            )
            if active_suppression >= self.config.suppression_max_active:
                return None

        # Determine TTL
        if ttl_turns is None:
            ttl_map = {
                TicketKind.ROUTE_RETENTION: self.config.route_retention_ttl,
                TicketKind.SELECTED_TARGET_PROMOTION: self.config.promotion_ttl,
                TicketKind.DEAD_REVISIT_SUPPRESSION: self.config.suppression_ttl,
                TicketKind.BOUNDED_SECOND_CHANCE: self.config.second_chance_ttl,
                TicketKind.CONTINUATION_ATTACH: self.config.continuation_attach_ttl,
                TicketKind.STALE_RETARGET: self.config.stale_retarget_ttl,
                TicketKind.LOW_GAIN_DIV: self.config.low_gain_div_ttl,
                TicketKind.SATURATION_RELIEF: self.config.saturation_relief_ttl,
            }
            ttl_turns = ttl_map.get(kind, 3)

        ticket = ControlIRTicket(
            kind=kind,
            state=TicketState.ISSUED,
            source_stage=source_stage,
            anchor_family=anchor_family,
            evidence=evidence or [],
            confidence=confidence,
            ttl_turns=ttl_turns,
            remaining_turns=ttl_turns,
            issued_step=step,
            promote_from=promote_from or [],
            suppress_dispositions=suppress_dispositions or [],
            success_signal=success_signal,
            fallback=fallback,
        )

        self._tickets.append(ticket)
        self._issue_counts[kind.value] = self._issue_counts.get(kind.value, 0) + 1

        if kind == TicketKind.BOUNDED_SECOND_CHANCE:
            self._last_second_chance_step = step
        if kind == TicketKind.SELECTED_TARGET_PROMOTION:
            self._promotion_count += 1

        return ticket

    # ── Evidence Evaluation & Auto-Issue ──────────────────────────────────

    def evaluate_and_issue(self, evidence: ControlIREvidence) -> list[ControlIRTicket]:
        """Evaluate current evidence and auto-issue appropriate tickets.

        This is the main entry point called each step. It examines the evidence
        and decides which tickets to issue based on the current bottleneck map.
        """
        self._last_issue_decisions = {k.value: "" for k in _AUTO_ISSUE_KINDS}

        if not self.config.enabled:
            for kind in _AUTO_ISSUE_KINDS:
                self._record_issue_decision(kind, "disabled")
            return []

        issued: list[ControlIRTicket] = []
        step = evidence.current_step

        # A. Route retention: post-shared follow-through instability
        route_retention_decision = self._route_retention_issue_decision(evidence)
        self._record_issue_decision(TicketKind.ROUTE_RETENTION, route_retention_decision)
        if route_retention_decision == "issued":
            t = self.issue_ticket(
                TicketKind.ROUTE_RETENTION,
                step=step,
                source_stage="POST_FINAL_SHARED_EXIT",
                anchor_family=evidence.recently_live_anchor_family or "shared_successor_family",
                evidence=self._build_route_retention_evidence(evidence),
                confidence=self._compute_route_retention_confidence(evidence),
                promote_from=["successor_slate", "protected_hedge"],
                suppress_dispositions=["preterminal_dead_revisit"],
                success_signal="selected_target_attached",
            )
            if t:
                issued.append(t)
            else:
                self._overwrite_last_issue_decision(TicketKind.ROUTE_RETENTION, "issue_ticket_rejected")

        # B. Selected target promotion: empty target with non-empty candidate pool
        promotion_decision = self._promotion_issue_decision(evidence)
        self._record_issue_decision(TicketKind.SELECTED_TARGET_PROMOTION, promotion_decision)
        if promotion_decision == "issued":
            t = self.issue_ticket(
                TicketKind.SELECTED_TARGET_PROMOTION,
                step=step,
                source_stage="LATE_EMPTY_TARGET",
                anchor_family=evidence.recently_live_anchor_family,
                evidence=self._build_promotion_evidence(evidence),
                confidence=self._compute_promotion_confidence(evidence),
                promote_from=["candidate_pool", "successor_slate"],
                success_signal="selected_target_non_empty",
            )
            if t:
                issued.append(t)
            else:
                self._overwrite_last_issue_decision(TicketKind.SELECTED_TARGET_PROMOTION, "issue_ticket_rejected")

        # C. Dead revisit suppression: monopoly of preterminal_dead_revisit
        suppression_decision = self._suppression_issue_decision(evidence)
        self._record_issue_decision(TicketKind.DEAD_REVISIT_SUPPRESSION, suppression_decision)
        if suppression_decision == "issued":
            t = self.issue_ticket(
                TicketKind.DEAD_REVISIT_SUPPRESSION,
                step=step,
                source_stage="LATE_NO_TARGET",
                evidence=self._build_suppression_evidence(evidence),
                confidence=self._compute_suppression_confidence(evidence),
                suppress_dispositions=["preterminal_dead_revisit"],
                success_signal="disposition_diversified",
            )
            if t:
                issued.append(t)
            else:
                self._overwrite_last_issue_decision(TicketKind.DEAD_REVISIT_SUPPRESSION, "issue_ticket_rejected")

        # D. Bounded second chance: one-shot late conversion
        second_chance_decision = self._second_chance_issue_decision(evidence)
        self._record_issue_decision(TicketKind.BOUNDED_SECOND_CHANCE, second_chance_decision)
        if second_chance_decision == "issued":
            t = self.issue_ticket(
                TicketKind.BOUNDED_SECOND_CHANCE,
                step=step,
                source_stage="LATE_CONVERSION_ATTEMPT",
                anchor_family=evidence.recently_live_anchor_family,
                evidence=self._build_second_chance_evidence(evidence),
                confidence=self._compute_second_chance_confidence(evidence),
                promote_from=["successor_slate", "protected_hedge"],
                success_signal="late_conversion_success",
            )
            if t:
                issued.append(t)
            else:
                self._overwrite_last_issue_decision(TicketKind.BOUNDED_SECOND_CHANCE, "issue_ticket_rejected")

        # E. Stale retarget: target health is STALE (Phase 3)
        stale_retarget_decision = self._stale_retarget_issue_decision(evidence)
        self._record_issue_decision(TicketKind.STALE_RETARGET, stale_retarget_decision)
        if stale_retarget_decision == "issued":
            t = self.issue_ticket(
                TicketKind.STALE_RETARGET,
                step=step,
                source_stage="STALE_TARGET_DETECTED",
                anchor_family=evidence.current_action_family,
                evidence=self._build_stale_retarget_evidence(evidence),
                confidence=self._compute_stale_retarget_confidence(evidence),
                promote_from=["candidate_pool", "base_pool"],
                suppress_dispositions=["stale_target"],
                success_signal="target_health_live",
            )
            if t:
                self._stale_retarget_count += 1
                issued.append(t)
            else:
                self._overwrite_last_issue_decision(TicketKind.STALE_RETARGET, "issue_ticket_rejected")

        # F. Low gain diversification: same family repeated without progress (Phase 3)
        low_gain_div_decision = self._low_gain_div_issue_decision(evidence)
        self._record_issue_decision(TicketKind.LOW_GAIN_DIV, low_gain_div_decision)
        if low_gain_div_decision == "issued":
            t = self.issue_ticket(
                TicketKind.LOW_GAIN_DIV,
                step=step,
                source_stage="LOW_INFO_GAIN_DETECTED",
                anchor_family=evidence.current_action_family,
                evidence=self._build_low_gain_div_evidence(evidence),
                confidence=self._compute_low_gain_div_confidence(evidence),
                suppress_dispositions=[evidence.current_action_family],
                success_signal="family_diversified",
            )
            if t:
                issued.append(t)
            else:
                self._overwrite_last_issue_decision(TicketKind.LOW_GAIN_DIV, "issue_ticket_rejected")

        # G. Saturation relief: candidate pool dominated by single family (Phase 3)
        saturation_relief_decision = self._saturation_relief_issue_decision(evidence)
        self._record_issue_decision(TicketKind.SATURATION_RELIEF, saturation_relief_decision)
        if saturation_relief_decision == "issued":
            t = self.issue_ticket(
                TicketKind.SATURATION_RELIEF,
                step=step,
                source_stage="POOL_SATURATION_DETECTED",
                anchor_family=evidence.candidate_pool_dominant_family,
                evidence=self._build_saturation_relief_evidence(evidence),
                confidence=self._compute_saturation_relief_confidence(evidence),
                suppress_dispositions=[evidence.candidate_pool_dominant_family],
                success_signal="pool_diversified",
            )
            if t:
                self._saturation_relief_count += 1
                issued.append(t)
            else:
                self._overwrite_last_issue_decision(TicketKind.SATURATION_RELIEF, "issue_ticket_rejected")

        return issued

    # ── Per-Step Advance ─────────────────────────────────────────────────

    def advance_step(self, step: int, *, resolution_signals: dict[str, bool] | None = None) -> None:
        """Advance all live tickets by one step.

        Args:
            step: Current step index.
            resolution_signals: Optional dict of signal_name -> bool indicating
                whether each success signal has been observed this step.
        """
        resolution_signals = resolution_signals or {}

        for ticket in self._tickets:
            if not ticket.is_live:
                continue

            # Activate issued tickets
            if ticket.state == TicketState.ISSUED:
                ticket.activate(step)

            # Check success signals
            if ticket.success_signal and resolution_signals.get(ticket.success_signal, False):
                ticket.attach(step, reason=ticket.success_signal)
                self._attach_counts[ticket.kind.value] = (
                    self._attach_counts.get(ticket.kind.value, 0) + 1
                )
                continue

            # Tick TTL
            if ticket.state == TicketState.ACTIVE:
                ticket.tick()
                if ticket.state == TicketState.EXPIRED:
                    self._expire_counts[ticket.kind.value] = (
                        self._expire_counts.get(ticket.kind.value, 0) + 1
                    )

    def consume_attached(self, step: int) -> list[ControlIRTicket]:
        """Consume all attached tickets and return them."""
        consumed = []
        for ticket in self._tickets:
            if ticket.state == TicketState.ATTACHED:
                ticket.consume(step)
                self._consume_counts[ticket.kind.value] = (
                    self._consume_counts.get(ticket.kind.value, 0) + 1
                )
                consumed.append(ticket)
        return consumed

    # ── Query ────────────────────────────────────────────────────────────

    def live_tickets(self, kind: TicketKind | None = None) -> list[ControlIRTicket]:
        """Return all currently live tickets, optionally filtered by kind."""
        tickets = [t for t in self._tickets if t.is_live]
        if kind is not None:
            tickets = [t for t in tickets if t.kind == kind]
        return tickets

    def active_suppression_dispositions(self) -> set[str]:
        """Return the set of dispositions currently being suppressed."""
        dispositions: set[str] = set()
        for ticket in self._tickets:
            if ticket.is_live and ticket.kind == TicketKind.DEAD_REVISIT_SUPPRESSION:
                dispositions.update(ticket.suppress_dispositions)
        return dispositions

    def active_promote_from_sources(self) -> set[str]:
        """Return the set of sources from which promotion is currently allowed."""
        sources: set[str] = set()
        for ticket in self._tickets:
            if ticket.is_live and ticket.kind in (
                TicketKind.ROUTE_RETENTION,
                TicketKind.SELECTED_TARGET_PROMOTION,
                TicketKind.BOUNDED_SECOND_CHANCE,
            ):
                sources.update(ticket.promote_from)
        return sources

    def has_live_ticket(self, kind: TicketKind) -> bool:
        """Check if there is at least one live ticket of the given kind."""
        return any(t.kind == kind and t.is_live for t in self._tickets)

    # ── Observer Export (Phase 1) ────────────────────────────────────────

    def build_step_metrics(self) -> dict[str, Any]:
        """Build ControlIR metrics for the current step (observer-only export)."""
        live = self.live_tickets()
        live_by_kind: dict[str, int] = {}
        for t in live:
            live_by_kind[t.kind.value] = live_by_kind.get(t.kind.value, 0) + 1

        return {
            "control_ir_enabled": self.config.enabled,
            "control_ir_mode": self.config.mode,
            "control_ir_live_ticket_count": len(live),
            "control_ir_live_tickets_by_kind": live_by_kind,
            "control_ir_active_suppression_dispositions": sorted(self.active_suppression_dispositions()),
            "control_ir_active_promote_from_sources": sorted(self.active_promote_from_sources()),
            "control_ir_issue_counts": dict(self._issue_counts),
            "control_ir_last_issue_decisions": dict(self._last_issue_decisions),
            "control_ir_issue_decision_histograms": {
                key: dict(value) for key, value in self._issue_decision_histograms.items()
            },
            "control_ir_attach_counts": dict(self._attach_counts),
            "control_ir_consume_counts": dict(self._consume_counts),
            "control_ir_expire_counts": dict(self._expire_counts),
            "control_ir_cancel_counts": dict(self._cancel_counts),
            "control_ir_live_tickets": [t.to_dict() for t in live],
        }

    def build_episode_summary(self) -> dict[str, Any]:
        """Build ControlIR section for episode summary."""
        all_tickets = [t.to_dict() for t in self._tickets]
        terminal_tickets = [t.to_dict() for t in self._tickets if t.is_terminal]

        # Compute effectiveness metrics
        total_issued = sum(self._issue_counts.values())
        total_attached = sum(self._attach_counts.values())
        total_consumed = sum(self._consume_counts.values())
        total_expired = sum(self._expire_counts.values())
        attach_rate = (total_attached / total_issued) if total_issued > 0 else 0.0
        consume_rate = (total_consumed / total_issued) if total_issued > 0 else 0.0

        return {
            "control_ir_enabled": self.config.enabled,
            "control_ir_mode": self.config.mode,
            "control_ir_total_issued": total_issued,
            "control_ir_total_attached": total_attached,
            "control_ir_total_consumed": total_consumed,
            "control_ir_total_expired": total_expired,
            "control_ir_total_cancelled": sum(self._cancel_counts.values()),
            "control_ir_attach_rate": round(attach_rate, 4),
            "control_ir_consume_rate": round(consume_rate, 4),
            "control_ir_issue_counts": dict(self._issue_counts),
            "control_ir_last_issue_decisions": dict(self._last_issue_decisions),
            "control_ir_issue_decision_histograms": {
                key: dict(value) for key, value in self._issue_decision_histograms.items()
            },
            "control_ir_attach_counts": dict(self._attach_counts),
            "control_ir_consume_counts": dict(self._consume_counts),
            "control_ir_expire_counts": dict(self._expire_counts),
            "control_ir_cancel_counts": dict(self._cancel_counts),
            "control_ir_ticket_history": all_tickets,
            "control_ir_terminal_tickets": terminal_tickets,
        }

    # ── Advisory Text (Phase 2) ──────────────────────────────────────────

    def build_advisory_text(self, last_evidence: "ControlIREvidence | None" = None) -> str:
        """Build compact ControlIR advisory text for the analyzer prompt.

        Only active in 'advisory' mode (Phase 2).
        Returns empty string in 'observer' mode (Phase 1).

        Args:
            last_evidence: Optional snapshot of the most recent evidence, used to
                include concrete component IDs and board positions in the text.
        """
        if not self.config.enabled or self.config.mode != "advisory":
            return ""

        live = self.live_tickets()
        if not live:
            return ""

        # Parse concrete candidate info from evidence summaries
        slate_entries: list[dict[str, str]] = []
        hedge_entry: dict[str, str] | None = None
        if last_evidence is not None:
            # Parse successor_slate_top2_summary: "comp_id:score:source:family | ..."
            if last_evidence.successor_slate_top2_summary:
                for raw in last_evidence.successor_slate_top2_summary.split(" | "):
                    parts = raw.strip().split(":")
                    if len(parts) >= 4:
                        slate_entries.append({
                            "comp_id": parts[0],
                            "score": parts[1],
                            "reason": parts[2],
                            "action_family": parts[3],
                        })
            # Parse protected_hedge_summary: "comp_id:score:source:family"
            if last_evidence.protected_hedge_summary:
                parts = last_evidence.protected_hedge_summary.strip().split(":")
                if len(parts) >= 4:
                    hedge_entry = {
                        "comp_id": parts[0],
                        "score": parts[1],
                        "reason": parts[2],
                        "action_family": parts[3],
                    }

        lines = [
            "[CONTROL IR ADVISORY]",
            "ECA internal state has detected a bottleneck. Use the guidance below to bias your next action.",
            "This is not a hard override — if the board/log contradicts this block, trust the board/log.",
            "",
        ]

        # Group by kind
        for kind in TicketKind:
            kind_tickets = [t for t in live if t.kind == kind]
            if not kind_tickets:
                continue

            lines.append(f"{kind.value}:")
            for i, ticket in enumerate(kind_tickets):
                lines.append(f"  [{i}] state={ticket.state.value} ttl_remaining={ticket.remaining_turns}")
                if ticket.anchor_family:
                    lines.append(f"      anchor_family={ticket.anchor_family}")
                if ticket.confidence > 0:
                    lines.append(f"      confidence={ticket.confidence:.3f}")

                # Add concrete guidance based on ticket kind
                if kind.value in ("selected_target_promotion", "route_retention", "bounded_second_chance"):
                    if kind.value == "bounded_second_chance":
                        lines.append("      GUIDANCE: Late conversion opportunity — one-shot attempt to recover progress.")
                    else:
                        lines.append("      GUIDANCE: No target is currently selected but candidates exist.")

                    if slate_entries:
                        for j, entry in enumerate(slate_entries):
                            lines.append(
                                f"      Preferred target [{j}]: {entry['comp_id']} "
                                f"(score={entry['score']}, reason={entry['reason']}, "
                                f"action_family={entry['action_family']})"
                            )
                    if hedge_entry:
                        lines.append(
                            f"      Protected hedge: {hedge_entry['comp_id']} "
                            f"(score={hedge_entry['score']}, reason={hedge_entry['reason']}, "
                            f"action_family={hedge_entry['action_family']})"
                        )

                    if kind.value == "bounded_second_chance":
                        lines.append("      ACTION: Do NOT invent a fresh target. Use the slate/hedge candidate above.")
                    else:
                        lines.append("      ACTION: Focus your next move on the preferred target above using its action_family.")

                elif kind.value == "dead_revisit_suppression":
                    if ticket.suppress_dispositions:
                        lines.append(f"      GUIDANCE: Suppress revisiting dead areas: {ticket.suppress_dispositions}")
                        lines.append("      ACTION: Avoid re-exploring areas already confirmed as dead. Seek new frontiers.")

            lines.append("")

        # Summary directives
        suppressed = self.active_suppression_dispositions()
        if suppressed:
            lines.append(f"active_suppression: {sorted(suppressed)}")

        return "\n".join(lines)

    # ── Private: Issuance Conditions ─────────────────────────────────────

    def _should_issue_route_retention(self, ev: ControlIREvidence) -> bool:
        return self._route_retention_issue_decision(ev) == "issued"

    def _route_retention_issue_decision(self, ev: ControlIREvidence) -> str:
        """Section 17.A: Post-shared follow-through instability."""
        # Don't issue if we already have a live route retention ticket
        if self.has_live_ticket(TicketKind.ROUTE_RETENTION):
            return "live_ticket_exists"
        # Shared entry must have happened and rearm must be live
        if ev.post_final_shared_entry_count <= 0:
            return "no_shared_entry"
        if ev.post_final_shared_read_class != "reached_rearm_live":
            return f"shared_read_class:{ev.post_final_shared_read_class or 'unknown'}"
        # Must be shortly after shared exit
        if ev.turns_since_shared_exit < 0:
            return "shared_exit_unobserved"
        if ev.turns_since_shared_exit > 5:
            return "shared_exit_too_old"
        # Selected target must be empty or fragile
        if not ev.selected_target_empty:
            return "selected_target_not_empty"
        return "issued"

    def _should_issue_promotion(self, ev: ControlIREvidence) -> bool:
        return self._promotion_issue_decision(ev) == "issued"

    def _promotion_issue_decision(self, ev: ControlIREvidence) -> str:
        """Section 17.B: Empty selected_target with non-empty candidate pool."""
        if self.has_live_ticket(TicketKind.SELECTED_TARGET_PROMOTION):
            return "live_ticket_exists"
        if not ev.selected_target_empty:
            return "selected_target_not_empty"
        if not ev.candidate_pool_non_empty:
            return "candidate_pool_empty"
        if ev.candidate_pool_size < self.config.promotion_min_candidate_pool_size:
            return f"candidate_pool_too_small:{ev.candidate_pool_size}"
        # Require some evidence of recent life
        if not (ev.recently_live_anchor or ev.protected_hedge_visible or ev.successor_slate_non_empty):
            return "no_recent_live_evidence"
        return "issued"

    def _should_issue_suppression(self, ev: ControlIREvidence) -> bool:
        return self._suppression_issue_decision(ev) == "issued"

    def _suppression_issue_decision(self, ev: ControlIREvidence) -> str:
        """Section 17.C: Monopoly of preterminal_dead_revisit."""
        if ev.candidate_pool_no_target_disposition != "preterminal_dead_revisit":
            return f"disposition:{ev.candidate_pool_no_target_disposition or 'none'}"
        if self.config.suppression_requires_recent_live_evidence:
            if not (ev.recently_live_anchor or ev.shared_rearm_live):
                return "no_recent_live_evidence"
        # Don't issue if already at max active suppression tickets
        active_count = sum(
            1 for t in self._tickets
            if t.kind == TicketKind.DEAD_REVISIT_SUPPRESSION and t.is_live
        )
        if active_count >= self.config.suppression_max_active:
            return "suppression_max_active"
        return "issued"

    def _should_issue_second_chance(self, ev: ControlIREvidence) -> bool:
        return self._second_chance_issue_decision(ev) == "issued"

    def _second_chance_issue_decision(self, ev: ControlIREvidence) -> str:
        """Section 17.D: Bounded second chance for late conversion."""
        if self.has_live_ticket(TicketKind.BOUNDED_SECOND_CHANCE):
            return "live_ticket_exists"
        # Must have empty target with candidates available
        if not ev.selected_target_empty:
            return "selected_target_not_empty"
        if not ev.candidate_pool_non_empty:
            return "candidate_pool_empty"
        # Must have recent shared success or protected hedge
        if not (ev.shared_rearm_live or ev.protected_hedge_visible):
            return "no_rearm_or_hedge"
        # Phase handoff resurface must be low (Section 9.E: mitigated)
        if ev.phase_handoff_resurface_count > 5:
            return f"phase_handoff_resurface_high:{ev.phase_handoff_resurface_count}"
        return "issued"

    def _record_issue_decision(self, kind: TicketKind, decision: str) -> None:
        self._last_issue_decisions[kind.value] = decision
        histogram = self._issue_decision_histograms.setdefault(kind.value, {})
        histogram[decision] = histogram.get(decision, 0) + 1

    def _overwrite_last_issue_decision(self, kind: TicketKind, decision: str) -> None:
        previous = self._last_issue_decisions.get(kind.value)
        self._last_issue_decisions[kind.value] = decision
        histogram = self._issue_decision_histograms.setdefault(kind.value, {})
        if previous:
            previous_count = histogram.get(previous, 0)
            if previous_count > 0:
                histogram[previous] = previous_count - 1
                if histogram[previous] <= 0:
                    histogram.pop(previous, None)
        histogram[decision] = histogram.get(decision, 0) + 1

    def _should_issue_stale_retarget(self, ev: ControlIREvidence) -> bool:
        """Phase 3.E: Target health is STALE for too many steps."""
        return self._stale_retarget_issue_decision(ev) == "issued"

    def _stale_retarget_issue_decision(self, ev: ControlIREvidence) -> str:
        if self.has_live_ticket(TicketKind.STALE_RETARGET):
            return "live_ticket_exists"
        if self._stale_retarget_count >= self.config.stale_retarget_max_per_episode:
            return "episode_max_reached"
        health = str(ev.target_health or "none")
        if health not in {"stale", "saturated"}:
            return f"target_health:{health}"
        if health == "saturated":
            if self.has_live_ticket(TicketKind.SATURATION_RELIEF):
                return "saturation_relief_active"
            if self._saturation_relief_issue_decision(ev) == "issued":
                return "saturation_relief_priority"
            if ev.steps_since_target_progress < self.config.stale_retarget_stale_threshold_steps:
                return f"saturated_streak_too_short:{ev.steps_since_target_progress}"
            return "issued"
        if ev.steps_since_target_progress < self.config.stale_retarget_stale_threshold_steps:
            return f"stale_streak_too_short:{ev.steps_since_target_progress}"
        return "issued"

    def _should_issue_low_gain_div(self, ev: ControlIREvidence) -> bool:
        """Phase 3.F: Same family repeated without progress."""
        return self._low_gain_div_issue_decision(ev) == "issued"

    def _low_gain_div_issue_decision(self, ev: ControlIREvidence) -> str:
        active_count = sum(
            1 for t in self._tickets
            if t.kind == TicketKind.LOW_GAIN_DIV and t.is_live
        )
        if active_count >= self.config.low_gain_div_max_active:
            return "max_active"
        if self.has_live_ticket(TicketKind.SATURATION_RELIEF):
            return "saturation_relief_active"
        if not ev.current_action_family:
            return "no_current_family"
        if ev.family_repeat_count < self.config.low_gain_div_min_repeat_steps:
            return f"repeat_too_short:{ev.family_repeat_count}"
        # Don't issue if board is making progress
        if ev.steps_since_target_progress == 0:
            return "progress_present"
        if ev.target_health == "saturated":
            if self._saturation_relief_issue_decision(ev) == "issued":
                return "saturation_relief_priority"
        return "issued"

    def _should_issue_saturation_relief(self, ev: ControlIREvidence) -> bool:
        """Phase 3.G: Candidate pool dominated by single family."""
        return self._saturation_relief_issue_decision(ev) == "issued"

    def _saturation_relief_issue_decision(self, ev: ControlIREvidence) -> str:
        if self.has_live_ticket(TicketKind.SATURATION_RELIEF):
            return "live_ticket_exists"
        if self._saturation_relief_count >= self.config.saturation_relief_max_per_episode:
            return "episode_max_reached"
        min_pool_size = self.config.saturation_relief_min_pool_size
        dominant_ratio_threshold = self.config.saturation_relief_pool_saturation_ratio
        if ev.target_health == "saturated":
            min_pool_size = max(2, min_pool_size - 1)
            dominant_ratio_threshold = max(0.6, dominant_ratio_threshold - 0.1)
        if ev.candidate_pool_size < min_pool_size:
            return f"candidate_pool_too_small:{ev.candidate_pool_size}"
        if not ev.candidate_pool_dominant_family:
            return "no_dominant_family"
        if ev.candidate_pool_dominant_family_ratio < dominant_ratio_threshold:
            return f"dominant_ratio_too_low:{ev.candidate_pool_dominant_family_ratio:.2f}"
        return "issued"

    # ── Private: Evidence Builders ────────────────────────────────────────────

    def _build_route_retention_evidence(self, ev: ControlIREvidence) -> list[str]:
        evidence = [
            f"post_final_shared_read_class={ev.post_final_shared_read_class}",
            f"selected_target_empty={ev.selected_target_empty}",
            f"turns_since_shared_exit={ev.turns_since_shared_exit}",
        ]
        if ev.candidate_pool_non_empty:
            evidence.append("candidate_pool_non_empty")
        if ev.recently_live_anchor:
            evidence.append(f"recently_live_anchor_family={ev.recently_live_anchor_family}")
        return evidence

    def _build_promotion_evidence(self, ev: ControlIREvidence) -> list[str]:
        evidence = [
            f"selected_target_empty={ev.selected_target_empty}",
            f"candidate_pool_size={ev.candidate_pool_size}",
        ]
        if ev.recently_live_anchor:
            evidence.append(f"recently_live_anchor_family={ev.recently_live_anchor_family}")
        if ev.protected_hedge_visible:
            evidence.append("protected_hedge_visible")
        if ev.successor_slate_non_empty:
            evidence.append("successor_slate_non_empty")
        return evidence

    def _build_suppression_evidence(self, ev: ControlIREvidence) -> list[str]:
        evidence = [
            f"candidate_pool_no_target_disposition={ev.candidate_pool_no_target_disposition}",
            f"preterminal_dead_revisit_count={ev.preterminal_dead_revisit_count}",
        ]
        if ev.recently_live_anchor:
            evidence.append("recently_live_anchor")
        if ev.shared_rearm_live:
            evidence.append("shared_rearm_live")
        return evidence

    def _build_second_chance_evidence(self, ev: ControlIREvidence) -> list[str]:
        evidence = [
            f"selected_target_empty={ev.selected_target_empty}",
            f"candidate_pool_non_empty={ev.candidate_pool_non_empty}",
        ]
        if ev.shared_rearm_live:
            evidence.append("shared_rearm_live")
        if ev.protected_hedge_visible:
            evidence.append("protected_hedge_visible")
        return evidence

    # ── Private: Confidence Computation ──────────────────────────────────

    def _compute_route_retention_confidence(self, ev: ControlIREvidence) -> float:
        """Compute confidence for route retention ticket."""
        conf = 0.3  # base
        if ev.shared_rearm_live:
            conf += 0.2
        if ev.shared_rerank_live:
            conf += 0.1
        if ev.recently_live_anchor:
            conf += 0.15
        if ev.candidate_pool_non_empty:
            conf += 0.1
        if ev.successor_slate_non_empty:
            conf += 0.1
        return min(1.0, conf)

    def _compute_promotion_confidence(self, ev: ControlIREvidence) -> float:
        """Compute confidence for selected target promotion ticket."""
        conf = 0.2  # base
        if ev.candidate_pool_size >= 3:
            conf += 0.2
        elif ev.candidate_pool_size >= 1:
            conf += 0.1
        if ev.recently_live_anchor:
            conf += 0.2
        if ev.protected_hedge_visible:
            conf += 0.15
        if ev.anchor_stability > 0.5:
            conf += 0.1
        return min(1.0, conf)

    def _compute_suppression_confidence(self, ev: ControlIREvidence) -> float:
        """Compute confidence for dead revisit suppression ticket."""
        conf = 0.25  # base
        if ev.recently_live_anchor:
            conf += 0.25
        if ev.shared_rearm_live:
            conf += 0.2
        if ev.preterminal_dead_revisit_count > 10:
            conf += 0.15
        return min(1.0, conf)

    def _compute_second_chance_confidence(self, ev: ControlIREvidence) -> float:
        """Compute confidence for bounded second chance ticket."""
        conf = 0.2  # base
        if ev.shared_rearm_live:
            conf += 0.2
        if ev.protected_hedge_visible:
            conf += 0.15
        if ev.successor_slate_non_empty:
            conf += 0.15
        if ev.candidate_pool_size >= 2:
            conf += 0.1
        return min(1.0, conf)

    # ── Phase 3: Evidence Builders ───────────────────────────────────────────

    def _build_stale_retarget_evidence(self, ev: ControlIREvidence) -> list[str]:
        evidence = [
            f"target_health={ev.target_health}",
            f"steps_since_target_progress={ev.steps_since_target_progress}",
        ]
        if ev.current_target_component_id:
            evidence.append(f"current_target_component_id={ev.current_target_component_id}")
        if ev.current_action_family:
            evidence.append(f"current_action_family={ev.current_action_family}")
        if ev.candidate_pool_size > 0:
            evidence.append(f"candidate_pool_size={ev.candidate_pool_size}")
        return evidence

    def _build_low_gain_div_evidence(self, ev: ControlIREvidence) -> list[str]:
        evidence = [
            f"current_action_family={ev.current_action_family}",
            f"family_repeat_count={ev.family_repeat_count}",
            f"steps_since_target_progress={ev.steps_since_target_progress}",
        ]
        if ev.current_target_component_id:
            evidence.append(f"current_target_component_id={ev.current_target_component_id}")
        return evidence

    def _build_saturation_relief_evidence(self, ev: ControlIREvidence) -> list[str]:
        evidence = [
            f"candidate_pool_dominant_family={ev.candidate_pool_dominant_family}",
            f"candidate_pool_dominant_family_ratio={ev.candidate_pool_dominant_family_ratio:.2f}",
            f"candidate_pool_size={ev.candidate_pool_size}",
        ]
        if ev.current_target_component_id:
            evidence.append(f"current_target_component_id={ev.current_target_component_id}")
        return evidence

    # ── Phase 3: Confidence Computation ─────────────────────────────────────

    def _compute_stale_retarget_confidence(self, ev: ControlIREvidence) -> float:
        conf = 0.3  # base
        if ev.steps_since_target_progress >= self.config.stale_retarget_stale_threshold_steps * 2:
            conf += 0.2
        if ev.candidate_pool_size >= 3:
            conf += 0.15
        if ev.recently_live_anchor:
            conf += 0.1
        return min(1.0, conf)

    def _compute_low_gain_div_confidence(self, ev: ControlIREvidence) -> float:
        conf = 0.25  # base
        if ev.family_repeat_count >= self.config.low_gain_div_min_repeat_steps * 2:
            conf += 0.2
        if ev.steps_since_target_progress >= 5:
            conf += 0.15
        return min(1.0, conf)

    def _compute_saturation_relief_confidence(self, ev: ControlIREvidence) -> float:
        conf = 0.25  # base
        if ev.candidate_pool_dominant_family_ratio >= 0.9:
            conf += 0.25
        elif ev.candidate_pool_dominant_family_ratio >= 0.8:
            conf += 0.15
        if ev.candidate_pool_size >= 8:
            conf += 0.1
        return min(1.0, conf)
