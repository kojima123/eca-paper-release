"""ControlIR Compiler: Compile ECA semantics into executable typed IR objects.

This is the core of ControlIR-first Runtime v1.  The compiler sits between
the ECA semantic layer and the Local Controller:

    ECA.observe  →  ControlIRCompiler.compile(...)  →  TicketStore.update(...)

Design principles:
    - Reads ECA state surfaces (same as ControlIRBridge) but produces
      *executable* IR objects, not advisory prose.
    - Stateless per call — all persistent state lives in TicketStore.
    - Deterministic: same evidence always produces the same IR set.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from arcgym.eca.control_ir import (
    ControlIRConfig,
    ControlIREvidence,
    ControlIRRegistry,
    ControlIRTicket,
    TicketKind,
    TicketState,
)
from arcgym.eca.control_ir_bridge import ControlIRBridge

if TYPE_CHECKING:
    pass  # ECARuntime is too large to import at module level

log = logging.getLogger(__name__)


# ─── Compiled IR Object ────────────────────────────────────────────────────

@dataclass
class CompiledIR:
    """A single compiled IR instruction for the Local Controller.

    This is a *typed* object that the controller can consume directly,
    without parsing prose.
    """
    kind: str  # matches TicketKind.value
    action: str  # "promote_target" | "suppress_disposition" | "retain_route" | "second_chance"
    target_component_id: str = ""
    target_score: float = 0.0
    target_source: str = ""
    target_action_family: str = ""
    suppress_dispositions: list[str] = field(default_factory=list)
    promote_from: list[str] = field(default_factory=list)
    confidence: float = 0.0
    ttl_remaining: int = 0
    ticket: ControlIRTicket | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind,
            "action": self.action,
            "target_component_id": self.target_component_id,
            "target_score": round(self.target_score, 4),
            "target_source": self.target_source,
            "target_action_family": self.target_action_family,
            "suppress_dispositions": list(self.suppress_dispositions),
            "promote_from": list(self.promote_from),
            "confidence": round(self.confidence, 4),
            "ttl_remaining": self.ttl_remaining,
        }
        if self.ticket is not None:
            d["ticket_state"] = self.ticket.state.value
        return d


@dataclass
class CompilationResult:
    """Result of a single compilation pass."""
    instructions: list[CompiledIR] = field(default_factory=list)
    evidence: ControlIREvidence | None = None
    newly_issued: list[ControlIRTicket] = field(default_factory=list)
    needs_analyzer_fallback: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction_count": len(self.instructions),
            "instructions": [ir.to_dict() for ir in self.instructions],
            "newly_issued_count": len(self.newly_issued),
            "needs_analyzer_fallback": self.needs_analyzer_fallback,
            "fallback_reason": self.fallback_reason,
        }


# ─── Candidate Entry (parsed from evidence summaries) ──────────────────────

@dataclass
class _CandidateEntry:
    comp_id: str = ""
    score: float = 0.0
    source: str = ""
    action_family: str = ""


def _parse_candidate_summary(raw: str) -> list[_CandidateEntry]:
    """Parse pipe-separated candidate summary into structured entries.

    Format: "comp_id:score:source:family | comp_id:score:source:family"
    """
    entries: list[_CandidateEntry] = []
    if not raw:
        return entries
    for segment in raw.split(" | "):
        parts = segment.strip().split(":")
        if len(parts) >= 4:
            try:
                score = float(parts[1])
            except (ValueError, IndexError):
                score = 0.0
            entries.append(_CandidateEntry(
                comp_id=parts[0],
                score=score,
                source=parts[2],
                action_family=parts[3],
            ))
    return entries


# ─── ControlIR Compiler ────────────────────────────────────────────────────

class ControlIRCompiler:
    """Compile ECA semantics into executable IR instructions.

    Lifecycle:
        compiler = ControlIRCompiler(config)
        # Each step:
        result = compiler.compile(runtime, step_index)
        # result.instructions → feed to LocalController
        # result.needs_analyzer_fallback → feed to harness
    """

    def __init__(self, config: ControlIRConfig | None = None) -> None:
        self.config = config or ControlIRConfig()
        # Delegate ticket management to the existing bridge/registry
        self._bridge = ControlIRBridge(self.config)
        self._last_result: CompilationResult | None = None

    @property
    def registry(self) -> ControlIRRegistry:
        return self._bridge.registry

    def reset(self) -> None:
        """Reset for a new episode."""
        self._bridge.reset()
        self._last_result = None

    def compile(self, runtime: Any, step_index: int) -> CompilationResult:
        """Run one compilation pass.

        1. Delegate evidence extraction + ticket issuance to bridge
        2. Compile live tickets into executable IR instructions
        3. Determine if analyzer fallback is needed
        """
        # Step the bridge (extract evidence, issue tickets, advance, consume)
        self._bridge.step(runtime, step_index)

        evidence = self._bridge._last_evidence
        newly_issued = list(self._bridge._last_issued or [])

        # Compile live tickets into IR instructions
        instructions = self._compile_live_tickets(evidence)

        # Determine analyzer fallback
        needs_fallback, fallback_reason = self._check_fallback_needed(
            instructions, evidence
        )

        result = CompilationResult(
            instructions=instructions,
            evidence=evidence,
            newly_issued=newly_issued,
            needs_analyzer_fallback=needs_fallback,
            fallback_reason=fallback_reason,
        )
        self._last_result = result
        return result

    def step_metrics(self) -> dict[str, Any]:
        """Return ControlIR metrics for the current step."""
        base = self._bridge.step_metrics()
        if self._last_result:
            base["control_ir_compilation"] = self._last_result.to_dict()
        return base

    def episode_summary(self) -> dict[str, Any]:
        """Return ControlIR section for episode summary."""
        return self._bridge.episode_summary()

    def advisory_text(self) -> str:
        """Return advisory text (backward compat with existing harness)."""
        return self._bridge.advisory_text()

    def get_recommended_components(self) -> list[str]:
        """Return recommended component IDs (backward compat)."""
        return self._bridge.get_recommended_components()

    # ── Private: Compile live tickets into IR ───────────────────────────

    def _compile_live_tickets(
        self, evidence: ControlIREvidence | None
    ) -> list[CompiledIR]:
        """Convert live tickets + evidence into executable IR instructions."""
        if evidence is None:
            return []

        instructions: list[CompiledIR] = []
        live_tickets = self.registry.live_tickets()

        # Parse candidate summaries once
        slate_candidates = _parse_candidate_summary(
            evidence.successor_slate_top2_summary
        )
        hedge_candidates = _parse_candidate_summary(
            evidence.protected_hedge_summary
        )
        # Pick best candidate across slate and hedge
        best = self._pick_best_candidate(slate_candidates, hedge_candidates)

        for ticket in live_tickets:
            ir = self._compile_single_ticket(ticket, evidence, best)
            if ir is not None:
                instructions.append(ir)

        return instructions

    def _compile_single_ticket(
        self,
        ticket: ControlIRTicket,
        evidence: ControlIREvidence,
        best_candidate: _CandidateEntry | None,
    ) -> CompiledIR | None:
        """Compile a single live ticket into an IR instruction."""
        kind = ticket.kind

        if kind == TicketKind.SELECTED_TARGET_PROMOTION:
            return self._compile_promotion(ticket, evidence, best_candidate)
        elif kind == TicketKind.ROUTE_RETENTION:
            return self._compile_route_retention(ticket, evidence, best_candidate)
        elif kind == TicketKind.DEAD_REVISIT_SUPPRESSION:
            return self._compile_suppression(ticket, evidence)
        elif kind == TicketKind.BOUNDED_SECOND_CHANCE:
            return self._compile_second_chance(ticket, evidence, best_candidate)
        # Phase 3 tickets
        elif kind == TicketKind.STALE_RETARGET:
            return self._compile_stale_retarget(ticket, evidence, best_candidate)
        elif kind == TicketKind.LOW_GAIN_DIV:
            return self._compile_low_gain_div(ticket, evidence)
        elif kind == TicketKind.SATURATION_RELIEF:
            return self._compile_saturation_relief(ticket, evidence, best_candidate)
        else:
            return None

    def _compile_promotion(
        self,
        ticket: ControlIRTicket,
        evidence: ControlIREvidence,
        best: _CandidateEntry | None,
    ) -> CompiledIR:
        ir = CompiledIR(
            kind=ticket.kind.value,
            action="promote_target",
            promote_from=list(ticket.promote_from),
            confidence=ticket.confidence,
            ttl_remaining=ticket.remaining_turns,
            ticket=ticket,
        )
        if best:
            ir.target_component_id = best.comp_id
            ir.target_score = best.score
            ir.target_source = best.source
            ir.target_action_family = best.action_family
        return ir

    def _compile_route_retention(
        self,
        ticket: ControlIRTicket,
        evidence: ControlIREvidence,
        best: _CandidateEntry | None,
    ) -> CompiledIR:
        ir = CompiledIR(
            kind=ticket.kind.value,
            action="retain_route",
            suppress_dispositions=list(ticket.suppress_dispositions),
            promote_from=list(ticket.promote_from),
            confidence=ticket.confidence,
            ttl_remaining=ticket.remaining_turns,
            ticket=ticket,
        )
        if best:
            ir.target_component_id = best.comp_id
            ir.target_score = best.score
            ir.target_source = best.source
            ir.target_action_family = best.action_family
        return ir

    def _compile_suppression(
        self,
        ticket: ControlIRTicket,
        evidence: ControlIREvidence,
    ) -> CompiledIR:
        return CompiledIR(
            kind=ticket.kind.value,
            action="suppress_disposition",
            suppress_dispositions=list(ticket.suppress_dispositions),
            confidence=ticket.confidence,
            ttl_remaining=ticket.remaining_turns,
            ticket=ticket,
        )

    def _compile_second_chance(
        self,
        ticket: ControlIRTicket,
        evidence: ControlIREvidence,
        best: _CandidateEntry | None,
    ) -> CompiledIR:
        ir = CompiledIR(
            kind=ticket.kind.value,
            action="second_chance",
            promote_from=list(ticket.promote_from),
            confidence=ticket.confidence,
            ttl_remaining=ticket.remaining_turns,
            ticket=ticket,
        )
        if best:
            ir.target_component_id = best.comp_id
            ir.target_score = best.score
            ir.target_source = best.source
            ir.target_action_family = best.action_family
        return ir

    def _compile_stale_retarget(
        self,
        ticket: ControlIRTicket,
        evidence: ControlIREvidence,
        best: _CandidateEntry | None,
    ) -> CompiledIR:
        """Phase 3: force retarget when current target is STALE."""
        ir = CompiledIR(
            kind=ticket.kind.value,
            action="force_retarget",
            promote_from=list(ticket.promote_from),
            confidence=ticket.confidence,
            ttl_remaining=ticket.remaining_turns,
            ticket=ticket,
        )
        if best:
            ir.target_component_id = best.comp_id
            ir.target_score = best.score
            ir.target_source = best.source
            ir.target_action_family = best.action_family
        return ir

    def _compile_low_gain_div(
        self,
        ticket: ControlIRTicket,
        evidence: ControlIREvidence,
    ) -> CompiledIR:
        """Phase 3: diversify action family when info-gain is low."""
        return CompiledIR(
            kind=ticket.kind.value,
            action="diversify_family",
            suppress_dispositions=list(ticket.suppress_dispositions),
            confidence=ticket.confidence,
            ttl_remaining=ticket.remaining_turns,
            ticket=ticket,
        )

    def _compile_saturation_relief(
        self,
        ticket: ControlIRTicket,
        evidence: ControlIREvidence,
        best: _CandidateEntry | None,
    ) -> CompiledIR:
        """Phase 3: relieve candidate pool saturation."""
        ir = CompiledIR(
            kind=ticket.kind.value,
            action="relief_pool",
            suppress_dispositions=list(ticket.suppress_dispositions),
            confidence=ticket.confidence,
            ttl_remaining=ticket.remaining_turns,
            ticket=ticket,
        )
        if best:
            ir.target_component_id = best.comp_id
            ir.target_score = best.score
            ir.target_source = best.source
            ir.target_action_family = best.action_family
        return ir

    # ── Private: Candidate selection ────────────────────────────────────

    def _pick_best_candidate(
        self,
        slate: list[_CandidateEntry],
        hedge: list[_CandidateEntry],
    ) -> _CandidateEntry | None:
        """Pick the best candidate from slate and hedge entries by score."""
        all_candidates = list(slate) + list(hedge)
        if not all_candidates:
            return None
        return max(all_candidates, key=lambda c: c.score)

    # ── Private: Analyzer fallback check ────────────────────────────────

    def _check_fallback_needed(
        self,
        instructions: list[CompiledIR],
        evidence: ControlIREvidence | None,
    ) -> tuple[bool, str]:
        """Determine if the Analyzer should be called as a fallback.

        Returns (needs_fallback, reason).

        Fallback conditions (Section 5 of the design):
        - No IR instructions and no target → unknown situation
        - Ticket conflicts detected
        - Unknown action family in best candidate
        - 2+ consecutive local failures (tracked externally)
        """
        if not instructions and evidence and evidence.selected_target_empty:
            # No IR and no target — unknown situation
            return True, "no_ir_no_target"

        if not instructions:
            # No IR but target exists — let runtime continue normally
            return False, ""

        # Check for ticket conflicts: multiple promote_target with different targets
        promote_targets = [
            ir for ir in instructions if ir.action == "promote_target"
        ]
        if len(promote_targets) > 1:
            targets = {ir.target_component_id for ir in promote_targets if ir.target_component_id}
            if len(targets) > 1:
                return True, "conflicting_promote_targets"

        # Check for unknown action family
        for ir in instructions:
            if ir.target_action_family and ir.target_action_family not in _KNOWN_ACTION_FAMILIES:
                return True, f"unknown_action_family:{ir.target_action_family}"

        return False, ""


# Known action families that MicroPlanner can handle locally
_KNOWN_ACTION_FAMILIES = frozenset({
    "move",
    "click",
    "color_fill",
    "drag",
    "copy",
    "flood_fill",
    "draw_line",
    "select",
    "reset",
})
