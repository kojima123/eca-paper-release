"""ControlIR Bridge: Integration layer between ECARuntime and ControlIR Registry.

This module provides the bridge that:
1. Extracts ControlIREvidence from ECARuntime state each step
2. Feeds evidence into ControlIRRegistry for ticket evaluation
3. Collects resolution signals from runtime state changes
4. Exports ControlIR metrics into step_metrics and episode_summary

Design: This is a thin adapter — it does not duplicate runtime logic.
It reads from runtime's existing surfaces and translates them into
the ControlIR evidence format.

Attribute name mapping (ECARuntime actual → ControlIR evidence field):
  post_final_shared_entry_count               → ev.post_final_shared_entry_count
  post_final_shared_runtime_reclaim_rearm_count → _infer_shared_read_class rearm check
  last_selected_target_summary                → ev.selected_target_empty
  last_candidate_pool                         → ev.candidate_pool_*
  last_candidate_pool_no_target_disposition   → ev.candidate_pool_no_target_disposition
  preterminal_dead_revisit_escape_attempt_count → ev.preterminal_dead_revisit_count
  last_post_final_shared_runtime_reclaim_rearmed → ev.shared_rearm_live
  last_post_final_shared_rerank_passed        → ev.shared_rerank_live
  queue_exhaustions                           → ev.queue_exhaustion_detected
  structural_time_tracker (no underscore)     → ev.anchor_stability / recently_live_anchor
  last_current_target_viability_reason        → ev.last_viability_reason
  last_current_target_viability_drop_reason   → ev.last_viability_drop_reason
  last_protected_hedge_contract["post_final_shared_entered_step_idx"] → turns_since_shared_exit
  _phase_handoff_resurface_count              → ev.phase_handoff_resurface_count
  last_protected_hedge_visible                → ev.protected_hedge_visible
  last_successor_slate                        → ev.successor_slate_non_empty
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from arcgym.eca.control_ir import (
    ControlIRConfig,
    ControlIREvidence,
    ControlIRRegistry,
    TicketKind,
)

if TYPE_CHECKING:
    pass  # ECARuntime is too large to import at module level


class ControlIRBridge:
    """Bridge between ECARuntime and ControlIRRegistry.

    Usage in runtime.py:
        self._control_ir_bridge = ControlIRBridge(config)
        # In observe():
        self._control_ir_bridge.step(self, step_index)
        # In record_step_metrics():
        metrics.update(self._control_ir_bridge.step_metrics())
        # In build_episode_summary():
        summary["control_ir"] = self._control_ir_bridge.episode_summary()
    """

    def __init__(self, config: ControlIRConfig | None = None) -> None:
        self.config = config or ControlIRConfig()
        self.registry = ControlIRRegistry(self.config)
        self._last_evidence: ControlIREvidence | None = None
        self._last_issued: list[Any] = []
        self._step_count: int = 0

    def reset(self) -> None:
        """Reset for a new episode."""
        self.registry.reset()
        self._last_evidence = None
        self._last_issued = []
        self._step_count = 0

    def step(self, runtime: Any, step_index: int) -> None:
        """Execute one ControlIR step: extract evidence, evaluate, advance.

        Args:
            runtime: The ECARuntime instance (duck-typed to avoid circular import).
            step_index: Current step index in the episode.
        """
        if not self.config.enabled:
            return

        self._step_count += 1

        # 1. Extract evidence from runtime state
        evidence = self._extract_evidence(runtime, step_index)
        self._last_evidence = evidence

        # 2. Evaluate and auto-issue tickets
        self._last_issued = self.registry.evaluate_and_issue(evidence)

        # 3. Collect resolution signals
        signals = self._collect_resolution_signals(runtime)

        # 4. Advance all live tickets
        self.registry.advance_step(step_index, resolution_signals=signals)

        # 5. Consume attached tickets
        self.registry.consume_attached(step_index)

    def step_metrics(self) -> dict[str, Any]:
        """Return ControlIR metrics for the current step."""
        if not self.config.enabled:
            return {"control_ir_enabled": False}
        return self.registry.build_step_metrics()

    def episode_summary(self) -> dict[str, Any]:
        """Return ControlIR section for episode summary."""
        if not self.config.enabled:
            return {"control_ir_enabled": False}
        return self.registry.build_episode_summary()

    def advisory_text(self) -> str:
        """Return ControlIR advisory text for analyzer prompt."""
        if not self.config.enabled:
            return ""
        return self.registry.build_advisory_text(last_evidence=self._last_evidence)

    def get_recommended_components(self) -> list[str]:
        """Return component IDs recommended by live ControlIR tickets.

        Used by Approach B: ECARuntime can call this to boost admission scores
        for ControlIR-recommended components during target selection.

        Returns:
            List of component_id strings (may be empty if no live tickets
            or no evidence available).
        """
        if not self.config.enabled:
            return []
        if not self.registry.live_tickets():
            return []
        if self._last_evidence is None:
            return []

        recommended: list[str] = []

        # Parse successor_slate_top2_summary: "comp_id:score:source:family | ..."
        slate_summary = self._last_evidence.successor_slate_top2_summary
        if slate_summary:
            for entry in slate_summary.split(" | "):
                parts = entry.strip().split(":")
                if parts and parts[0]:
                    recommended.append(parts[0])

        # Parse protected_hedge_summary: "comp_id:score:source:family"
        hedge_summary = self._last_evidence.protected_hedge_summary
        if hedge_summary:
            parts = hedge_summary.strip().split(":")
            if parts and parts[0] and parts[0] not in recommended:
                recommended.append(parts[0])

        return recommended

    # ── Evidence Extraction ──────────────────────────────────────────────

    def _extract_evidence(self, runtime: Any, step_index: int) -> ControlIREvidence:
        """Extract ControlIREvidence from ECARuntime state.

        This reads from runtime's existing surfaces without duplicating logic.
        Uses getattr with defaults to be resilient against missing attributes.

        All attribute names have been verified against ECARuntime.__init__ to
        ensure they exist as instance variables.
        """
        ev = ControlIREvidence()
        ev.current_step = step_index

        # ── ECA semantic state ───────────────────────────────────────────
        # post_final_shared_entry_count: public counter, no underscore prefix
        ev.post_final_shared_entry_count = int(
            getattr(runtime, "post_final_shared_entry_count", 0) or 0
        )

        # post_final_shared_read_class: not stored as instance var; infer it
        ev.post_final_shared_read_class = self._infer_shared_read_class(runtime)

        # after_exit_outcome_class: only computed in build_episode_summary;
        # use empty string as placeholder for live-step evidence
        ev.post_final_shared_after_exit_outcome_class = ""

        # ── Selected target state ────────────────────────────────────────
        # ECARuntime stores the selected target summary as last_selected_target_summary
        # (not _selected_target). None or empty dict means no target selected.
        selected_target = getattr(runtime, "last_selected_target_summary", None)
        if hasattr(runtime, "_selected_target_summary_effectively_empty") and callable(
            getattr(runtime, "_selected_target_summary_effectively_empty")
        ):
            try:
                selected_target_obj = None
                if selected_target and isinstance(selected_target, dict):
                    from arcgym.eca.types import SelectedTargetSummary

                    selected_target_obj = SelectedTargetSummary(
                        candidate_id=str(selected_target.get("candidate_id", "") or ""),
                        anchor_key=str(selected_target.get("anchor_key", "") or ""),
                        source=str(selected_target.get("source", "") or ""),
                        admission_score=float(selected_target.get("admission_score", 0.0) or 0.0),
                        suggested_probe_reason=str(selected_target.get("suggested_probe_reason", "") or ""),
                        click_budget=int(selected_target.get("click_budget", 0) or 0),
                        move_probe_budget=int(selected_target.get("move_probe_budget", 0) or 0),
                        rotate_cost=float(selected_target.get("rotate_cost", 0.0) or 0.0),
                        confidence=str(selected_target.get("confidence", "") or ""),
                        runtime_probe_ok=bool(selected_target.get("runtime_probe_ok", False)),
                        component_id=str(selected_target.get("component_id", "") or ""),
                        transfer_hint_bonus=float(selected_target.get("transfer_hint_bonus", 0.0) or 0.0),
                        second_chance_kind=str(selected_target.get("second_chance_kind", "") or ""),
                        recovery_after_state=str(selected_target.get("recovery_after_state", "") or ""),
                        recovery_after_reason=str(selected_target.get("recovery_after_reason", "") or ""),
                        transition_model_candidate_family=str(
                            selected_target.get("transition_model_candidate_family", "") or ""
                        ),
                        transition_model_backoff_level=str(
                            selected_target.get("transition_model_backoff_level", "") or ""
                        ),
                        transition_model_support=int(selected_target.get("transition_model_support", 0) or 0),
                        transition_model_p_contract=float(
                            selected_target.get("transition_model_p_contract", 0.0) or 0.0
                        ),
                        transition_model_p_frontier=float(
                            selected_target.get("transition_model_p_frontier", 0.0) or 0.0
                        ),
                        transition_model_p_turnover=float(
                            selected_target.get("transition_model_p_turnover", 0.0) or 0.0
                        ),
                        transition_model_p_dead_churn=float(
                            selected_target.get("transition_model_p_dead_churn", 0.0) or 0.0
                        ),
                        transition_model_p_slip=float(
                            selected_target.get("transition_model_p_slip", 0.0) or 0.0
                        ),
                        transition_model_quality_gain=float(
                            selected_target.get("transition_model_quality_gain", 0.0) or 0.0
                        ),
                        transition_model_predicted_anxiety=float(
                            selected_target.get("transition_model_predicted_anxiety", 0.0) or 0.0
                        ),
                    )
                ev.selected_target_empty = bool(
                    runtime._selected_target_summary_effectively_empty(selected_target_obj)
                )
            except Exception:
                ev.selected_target_empty = selected_target is None or not bool(selected_target)
        else:
            ev.selected_target_empty = selected_target is None or not bool(selected_target)

        # ── Candidate pool ───────────────────────────────────────────────
        candidate_pool = getattr(runtime, "last_candidate_pool", None) or []
        non_rejected = [c for c in candidate_pool if not bool(c.get("rejected", False))]
        ev.candidate_pool_size = len(non_rejected)
        ev.candidate_pool_non_empty = ev.candidate_pool_size > 0

        ev.candidate_pool_no_target_disposition = str(
            getattr(runtime, "last_candidate_pool_no_target_disposition", "") or ""
        )

        # ── Preterminal dead revisit count ───────────────────────────────
        ev.preterminal_dead_revisit_count = int(
            getattr(runtime, "preterminal_dead_revisit_escape_attempt_count", 0) or 0
        )

        # ── Shared rearm/rerank state ────────────────────────────────────
        # last_post_final_shared_runtime_reclaim_rearmed: bool flag set each step
        ev.shared_rearm_live = bool(
            getattr(runtime, "last_post_final_shared_runtime_reclaim_rearmed", False)
        )
        # last_post_final_shared_rerank_passed: bool flag set each step
        ev.shared_rerank_live = bool(
            getattr(runtime, "last_post_final_shared_rerank_passed", False)
        )

        # ── Queue exhaustion ─────────────────────────────────────────────
        # queue_exhaustions: public cumulative counter
        ev.queue_exhaustion_detected = bool(
            getattr(runtime, "queue_exhaustions", 0) or 0
        )

        # ── GCSTM shadow evidence ────────────────────────────────────────
        # structural_time_tracker: public attribute (no leading underscore)
        # Only initialized when ECAConfig.structural_time_tracker_enabled=True
        structural_tracker = getattr(runtime, "structural_time_tracker", None)
        if structural_tracker is not None:
            model = getattr(structural_tracker, "model", None)
            if model is not None:
                ev.anchor_stability = float(getattr(model, "anchor_stability", 0.0) or 0.0)
                ev.recently_live_anchor = bool(
                    getattr(model, "recently_live_anchor", False)
                )
                ev.recently_live_anchor_family = str(
                    getattr(model, "recently_live_anchor_family", "") or ""
                )
                ev.dead_family_window = bool(
                    getattr(model, "dead_family_window", False)
                )

        # ── PSA subtype / viability evidence ────────────────────────────
        # psa_subtype is only computed in build_episode_summary; use empty string
        ev.psa_subtype = ""
        ev.psa_sweep_dead = False

        # Use live viability reason/drop_reason from last_current_target_viability_*
        ev.last_viability_reason = str(
            getattr(runtime, "last_current_target_viability_reason", "") or ""
        )
        ev.last_viability_drop_reason = str(
            getattr(runtime, "last_current_target_viability_drop_reason", "") or ""
        )

        # ── Runtime context: turns since shared exit ─────────────────────
        # post_final_shared_entered_step_idx is stored inside the contract dict.
        # When inside the phase, the contract is active; after exit it is cleared.
        # We compute turns_since_shared_exit from the contract's entered_step_idx.
        contract = getattr(runtime, "last_protected_hedge_contract", None) or {}
        entered_step = int(contract.get("post_final_shared_entered_step_idx", -1) or -1)
        if entered_step >= 0:
            ev.turns_since_shared_exit = step_index - entered_step
        else:
            # Also check post_final_shared_exit_turn_count as a proxy:
            # if we have exits but no active contract, we are post-exit
            exit_count = int(getattr(runtime, "post_final_shared_exit_turn_count", 0) or 0)
            ev.turns_since_shared_exit = exit_count if exit_count > 0 else -1

        ev.phase_handoff_resurface_count = int(
            getattr(runtime, "_phase_handoff_resurface_count", 0) or 0
        )

        # second_chance_kind_histogram: only in build_episode_summary.
        # Use last_selected_target_summary.second_chance_kind as a per-step proxy.
        if selected_target and isinstance(selected_target, dict):
            sc_kind = str(selected_target.get("second_chance_kind", "") or "")
            if sc_kind:
                ev.second_chance_kind_histogram = {sc_kind: 1}

        # ── Protected hedge / successor state ────────────────────────────────
        # Check both last_protected_hedge_visible flag AND contract-based
        # visibility (build_protected_hedge_summary returns non-empty string
        # when a contract is active or refocus-eligible). This matches the
        # logic used in step_metrics for successor_slate_protected_hedge_visible.
        _hedge_flag = bool(getattr(runtime, "last_protected_hedge_visible", False))
        if not _hedge_flag:
            try:
                _hedge_flag = bool(runtime.build_protected_hedge_summary())
            except Exception:
                pass
        ev.protected_hedge_visible = _hedge_flag

        # last_successor_slate: public list attribute
        successor_slate = getattr(runtime, "last_successor_slate", None)
        ev.successor_slate_non_empty = bool(successor_slate)

        # Concrete summaries for advisory text and Approach B boost
        # Call runtime methods directly to get comp_id:score:source:family format
        try:
            ev.successor_slate_top2_summary = runtime.build_successor_slate_summary()
        except Exception:
            ev.successor_slate_top2_summary = ""
        try:
            ev.protected_hedge_summary = runtime.build_protected_hedge_summary()
        except Exception:
            ev.protected_hedge_summary = ""

        # ── Phase 3: TargetHealth v2 evidence ────────────────────────────────────
        # _last_target_health_result is populated by TargetHealthEvaluator
        # which now runs BEFORE bridge.step() (see runtime.py observe()).
        _th_result = getattr(runtime, "_last_target_health_result", None)
        if _th_result is not None:
            _th_health = getattr(_th_result, "health", None)
            ev.target_health = str(getattr(_th_health, "value", "") or "")
            ev.steps_since_target_progress = int(
                getattr(_th_result, "no_score_streak", 0) or 0
            )

        # ── Phase 3: CandidateExtractor v2 evidence ───────────────────────────
        # _last_base_pool is an ExtractionResult populated by CandidateExtractor
        # which now runs BEFORE bridge.step() (see runtime.py observe()).
        _base_pool = getattr(runtime, "_last_base_pool", None)
        if _base_pool is not None:
            _candidates = getattr(_base_pool, "candidates", []) or []
            _valid = [c for c in _candidates if not getattr(c, "reject_reason", "")]
            if _valid:
                from collections import Counter as _Counter
                _families = [
                    getattr(c, "family", "") for c in _valid
                    if getattr(c, "family", "")
                ]
                if _families:
                    _most_common = _Counter(_families).most_common(1)[0]
                    ev.candidate_pool_dominant_family = _most_common[0]
                    ev.candidate_pool_dominant_family_ratio = (
                        _most_common[1] / len(_valid)
                    )

        # ── Phase 3: Current action family evidence ──────────────────────────
        # Track consecutive same-family actions for LOW_GAIN_DIV detection.
        # Infer current family from the selected target summary.
        _sel = getattr(runtime, "last_selected_target_summary", None)
        if _sel and isinstance(_sel, dict):
            ev.current_target_component_id = str(
                _sel.get("component_id", "")
                or _sel.get("id", "")
                or ""
            )
            ev.current_target_source = str(
                _sel.get("source", "")
                or _sel.get("reason", "")
                or ""
            )
            _cur_family = str(
                _sel.get("action_family", "")
                or _sel.get("family", "")
                or _sel.get("transition_model_candidate_family", "")
                or getattr(runtime, "last_stable_action_family", "")
                or ""
            )
            if _cur_family:
                ev.current_action_family = _cur_family
                # Maintain consecutive repeat count in bridge instance state
                if not hasattr(self, "_last_action_family"):
                    self._last_action_family: str = ""
                    self._family_repeat_count: int = 0
                if _cur_family == self._last_action_family:
                    self._family_repeat_count += 1
                else:
                    self._last_action_family = _cur_family
                    self._family_repeat_count = 1
                ev.family_repeat_count = self._family_repeat_count
            else:
                if hasattr(self, "_family_repeat_count"):
                    self._family_repeat_count = 0
        else:
            _cur_family = str(getattr(runtime, "last_stable_action_family", "") or "")
            if _cur_family:
                ev.current_action_family = _cur_family
                if not hasattr(self, "_last_action_family"):
                    self._last_action_family = ""
                    self._family_repeat_count = 0
                if _cur_family == self._last_action_family:
                    self._family_repeat_count += 1
                else:
                    self._last_action_family = _cur_family
                    self._family_repeat_count = 1
                ev.family_repeat_count = self._family_repeat_count

        current_component_id = str(ev.current_target_component_id or "")
        current_family = str(ev.current_action_family or "")
        if not hasattr(self, "_last_target_component_id"):
            self._last_target_component_id = ""
            self._last_target_family = ""
            self._last_shift_from_component_id = ""
            self._last_shift_from_family = ""
            self._steps_since_target_shift = -1
        previous_component_id = str(getattr(self, "_last_target_component_id", "") or "")
        previous_family = str(getattr(self, "_last_target_family", "") or "")
        if current_component_id and previous_component_id and current_component_id != previous_component_id:
            self._last_shift_from_component_id = previous_component_id
            self._last_shift_from_family = previous_family
            self._steps_since_target_shift = 0
        elif current_component_id and int(getattr(self, "_steps_since_target_shift", -1) or -1) >= 0:
            self._steps_since_target_shift += 1
        else:
            self._steps_since_target_shift = -1
            if not current_component_id:
                self._last_shift_from_component_id = ""
                self._last_shift_from_family = ""
        _shift_age = getattr(self, "_steps_since_target_shift", -1)
        try:
            ev.turns_since_target_shift = int(_shift_age)
        except Exception:
            ev.turns_since_target_shift = -1
        ev.recent_target_shift = bool(
            current_component_id
            and ev.turns_since_target_shift >= 0
            and ev.turns_since_target_shift <= 1
            and str(getattr(self, "_last_shift_from_component_id", "") or "")
        )
        if ev.recent_target_shift:
            ev.target_shift_from_component_id = str(
                getattr(self, "_last_shift_from_component_id", "") or ""
            )
            ev.target_shift_from_family = str(
                getattr(self, "_last_shift_from_family", "") or ""
            )
        self._last_target_component_id = current_component_id
        self._last_target_family = current_family

        # Retained route anchor info for resolver bias
        retained_anchor = dict(getattr(runtime, "last_post_final_shared_retained_route_anchor", None) or {})
        if retained_anchor:
            ev.retained_route_component_id = str(retained_anchor.get("component_id", "") or "")
            ev.retained_route_family = str(retained_anchor.get("source_family", "") or "")
            ev.retained_route_source = str(
                retained_anchor.get("effective_local_contract_source", "") or ""
            )

        return ev

    def _infer_shared_read_class(self, runtime: Any) -> str:
        """Infer post_final_shared_read_class from available runtime state.

        Uses public attribute post_final_shared_runtime_reclaim_rearm_count
        (no leading underscore) which is a cumulative counter.
        """
        entry_count = int(getattr(runtime, "post_final_shared_entry_count", 0) or 0)
        if entry_count <= 0:
            return "unreached"
        # post_final_shared_runtime_reclaim_rearm_count: public cumulative counter
        rearm_count = int(
            getattr(runtime, "post_final_shared_runtime_reclaim_rearm_count", 0) or 0
        )
        if rearm_count > 0:
            return "reached_rearm_live"
        return "reached_no_rearm"

    def _collect_resolution_signals(self, runtime: Any) -> dict[str, bool]:
        """Collect resolution signals from runtime state changes.

        Uses last_selected_target_summary (not _selected_target) which is the
        public attribute storing the current selected target summary dict.
        """
        signals: dict[str, bool] = {}
        ev = self._last_evidence or ControlIREvidence()
        selected_target_non_empty = not bool(ev.selected_target_empty)

        # selected_target_attached / selected_target_non_empty:
        # use the same effectively-empty interpretation as evidence extraction
        signals["selected_target_attached"] = selected_target_non_empty
        signals["selected_target_non_empty"] = selected_target_non_empty

        # disposition_diversified: no longer dominated by preterminal_dead_revisit
        disposition = str(ev.candidate_pool_no_target_disposition or "")
        signals["disposition_diversified"] = (
            disposition != "" and disposition != "preterminal_dead_revisit"
        )

        # late_conversion_success: target was empty and is now non-empty
        signals["late_conversion_success"] = selected_target_non_empty

        current_family = str(ev.current_action_family or "")
        current_component_id = str(ev.current_target_component_id or "")
        stale_anchor_components = {
            self._ticket_evidence_value(t, "current_target_component_id")
            for t in self.registry.live_tickets(TicketKind.STALE_RETARGET)
        }
        stale_anchor_components.discard("")
        stale_component_shifted = bool(
            selected_target_non_empty
            and current_component_id
            and stale_anchor_components
            and all(current_component_id != anchor for anchor in stale_anchor_components)
        )
        # stale_retarget success: either the target recovered to LIVE or a
        # retargeted component replaced the old anchor under the same non-empty band.
        signals["target_health_live"] = bool(
            selected_target_non_empty
            and (
                str(ev.target_health or "") == "live"
                or stale_component_shifted
            )
        )

        # low_gain_div success: current family has moved away from the ticket's
        # anchor family. This remains a global signal, so require the current
        # family to differ from every live low_gain anchor.
        low_gain_anchor_families = {
            t.anchor_family
            for t in self.registry.live_tickets(TicketKind.LOW_GAIN_DIV)
            if str(t.anchor_family or "")
        }
        low_gain_anchor_components = {
            self._ticket_evidence_value(t, "current_target_component_id")
            for t in self.registry.live_tickets(TicketKind.LOW_GAIN_DIV)
        }
        low_gain_anchor_components.discard("")
        signals["family_diversified"] = bool(
            selected_target_non_empty
            and (
                (
                    current_family
                    and low_gain_anchor_families
                    and all(current_family != anchor for anchor in low_gain_anchor_families)
                )
                or (
                    current_component_id
                    and low_gain_anchor_components
                    and all(current_component_id != anchor for anchor in low_gain_anchor_components)
                )
            )
        )

        # saturation_relief success: candidate pool is no longer dominated by
        # the anchor family, either because the dominant family changed, the
        # dominance ratio dropped below threshold, or the pool dissolved.
        dominant_family = str(ev.candidate_pool_dominant_family or "")
        dominant_ratio = float(ev.candidate_pool_dominant_family_ratio or 0.0)
        saturation_anchor_families = {
            t.anchor_family
            for t in self.registry.live_tickets(TicketKind.SATURATION_RELIEF)
            if str(t.anchor_family or "")
        }
        saturation_anchor_components = {
            self._ticket_evidence_value(t, "current_target_component_id")
            for t in self.registry.live_tickets(TicketKind.SATURATION_RELIEF)
        }
        saturation_anchor_components.discard("")
        signals["pool_diversified"] = bool(
            saturation_anchor_families
            and (
                not ev.candidate_pool_non_empty
                or not dominant_family
                or dominant_ratio < self.config.saturation_relief_pool_saturation_ratio
                or all(dominant_family != anchor for anchor in saturation_anchor_families)
                or (
                    selected_target_non_empty
                    and current_component_id
                    and saturation_anchor_components
                    and all(current_component_id != anchor for anchor in saturation_anchor_components)
                )
            )
        )

        return signals

    @staticmethod
    def _ticket_evidence_value(ticket: ControlIRTicket, key: str) -> str:
        prefix = f"{key}="
        for item in list(ticket.evidence or []):
            text = str(item or "")
            if text.startswith(prefix):
                return text[len(prefix):]
        return ""
