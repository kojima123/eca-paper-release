"""Bounded hidden-state executor for exclusive semantic ownership."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arcgym.semantics.api import AdapterExecutionTicket

_ALLOWED_PHASES = (
    "branch_entry",
    "site_reaching",
    "site_actuation",
    "state_change_verification",
    "goal_route_execution",
)
_PHASE_ORDER = {phase: idx for idx, phase in enumerate(_ALLOWED_PHASES)}
_MOVE_DELTAS = {
    "ACTION1": (-5, 0),
    "ACTION2": (5, 0),
    "ACTION3": (0, -5),
    "ACTION4": (0, 5),
}
_PLAYER_BBOX_HEIGHT = 3
_PLAYER_BBOX_WIDTH = 5


@dataclass
class HiddenStateExecutorLease:
    """Normalized controller-facing lease for hidden-state ownership."""

    active: bool = False
    activation_id: int = 0
    kind: str = ""
    phase: str = ""
    ownership: str = ""
    route_preview: list[str] = field(default_factory=list)
    target_anchor: list[int] = field(default_factory=list)
    site_anchor: list[int] = field(default_factory=list)
    goal_anchor: list[int] = field(default_factory=list)
    provenance: str = ""
    route_source: str = ""
    ttl_remaining: int = 0
    stall_remaining: int = 0
    missing_compatible_ticket_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    abort_reason: str = ""
    completed_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": bool(self.active),
            "activation_id": int(self.activation_id),
            "kind": self.kind,
            "phase": self.phase,
            "ownership": self.ownership,
            "route_preview": list(self.route_preview),
            "target_anchor": list(self.target_anchor),
            "site_anchor": list(self.site_anchor),
            "goal_anchor": list(self.goal_anchor),
            "provenance": self.provenance,
            "route_source": self.route_source,
            "ttl_remaining": int(self.ttl_remaining),
            "stall_remaining": int(self.stall_remaining),
            "missing_compatible_ticket_count": int(self.missing_compatible_ticket_count),
            "metadata": dict(self.metadata),
            "abort_reason": self.abort_reason,
            "completed_reason": self.completed_reason,
        }


class HiddenStateExecutor:
    """Owns an exclusive hidden-state branch lane while a typed ticket is active."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._activation_id = 0
        self._kind = ""
        self._phase = ""
        self._ownership = ""
        self._ttl_remaining = 0
        self._stall_remaining = 0
        self._missing_compatible_ticket_count = 0
        self._route_preview: list[str] = []
        self._target_anchor: list[int] = []
        self._site_anchor: list[int] = []
        self._goal_anchor: list[int] = []
        self._provenance = ""
        self._route_source = ""
        self._metadata: dict[str, Any] = {}
        self._start_score = 0
        self._start_level_number = 0
        self._last_player_anchor: tuple[int, int] | None = None
        self._last_phase = ""
        self._last_abort_reason = ""
        self._last_completed_reason = ""
        self._activation_count = 0
        self._active_step_count = 0
        self._phase_transition_histogram: dict[str, int] = {}
        self._abort_count = 0
        self._abort_reason_histogram: dict[str, int] = {}
        self._completion_count = 0
        self._completion_reason_histogram: dict[str, int] = {}
        self._route_source_histogram: dict[str, int] = {}
        self._generic_preemption_count = 0

    def note_generic_preemption(self) -> None:
        if self._active:
            self._generic_preemption_count += 1

    def step(
        self,
        *,
        ticket: AdapterExecutionTicket | None,
        player_bbox: list[int] | tuple[int, int, int, int] | None,
        board_rows: list[str] | None,
        step_index: int,
        score: int,
        level_number: int,
    ) -> HiddenStateExecutorLease:
        _ = step_index
        normalized_ticket = self._normalize_ticket(ticket)
        current_anchor = self._player_anchor(player_bbox)
        board_rows = list(board_rows or [])

        if self._active and self._is_completion(score=score, level_number=level_number, ticket=normalized_ticket):
            return self._complete(reason="goal_resolved")

        if self._active:
            self._consume_route_head_for_new_step()
            if normalized_ticket is None:
                self._missing_compatible_ticket_count += 1
            elif self._should_ignore_competing_site_ticket(normalized_ticket):
                self._missing_compatible_ticket_count = 0
                self._seed_route_arrival_probe_if_needed()
            elif not self._is_compatible_transition(normalized_ticket):
                return self._abort("invalid_phase_transition")
            else:
                self._missing_compatible_ticket_count = 0
                preserve_route_preview = self._should_preserve_route_preview(normalized_ticket)
                if (
                    preserve_route_preview
                    and self._route_head_invalid_for_current_anchor(
                        board_rows=board_rows,
                        current_anchor=current_anchor,
                    )
                ):
                    preserve_route_preview = False
                route_reloaded = (
                    not preserve_route_preview
                    and bool(list(normalized_ticket.route_preview or []))
                    and self._phase == normalized_ticket.phase
                    and normalized_ticket.phase in {"state_change_verification", "goal_route_execution"}
                )
                phase_changed = normalized_ticket.phase != self._phase
                if phase_changed:
                    self._record_transition(self._phase, normalized_ticket.phase)
                if phase_changed or route_reloaded:
                    self._ttl_remaining = max(1, int(normalized_ticket.ttl_steps))
                    self._stall_remaining = max(1, int(normalized_ticket.stall_budget))
                self._phase = normalized_ticket.phase
                self._absorb_ticket(
                    normalized_ticket,
                    preserve_route_preview=preserve_route_preview,
                )

            if self._missing_compatible_ticket_count >= 2:
                return self._abort("ticket_canceled")
            if self._anchors_invalid(board_rows):
                return self._abort("invalid_anchor")

            progressed = self._progressed(current_anchor=current_anchor)
            if progressed:
                self._stall_remaining = max(0, int(self._stall_remaining))
            else:
                self._stall_remaining = max(0, int(self._stall_remaining) - 1)
                if self._stall_remaining <= 0:
                    return self._abort("stall_budget_exhausted")

            self._last_player_anchor = current_anchor
            self._last_phase = self._phase
            self._ttl_remaining = max(0, int(self._ttl_remaining) - 1)
            if self._ttl_remaining <= 0:
                return self._abort("ttl_exhausted")
            self._active_step_count += 1
            self._record_route_source()
            return self._lease(active=True)

        if normalized_ticket is None:
            return HiddenStateExecutorLease()

        self._activate(
            ticket=normalized_ticket,
            player_anchor=current_anchor,
            score=score,
            level_number=level_number,
        )
        if self._anchors_invalid(board_rows):
            return self._abort("invalid_anchor")
        self._ttl_remaining = max(0, int(self._ttl_remaining) - 1)
        if self._ttl_remaining <= 0:
            return self._abort("ttl_exhausted")
        self._active_step_count += 1
        self._record_route_source()
        return self._lease(active=True)

    def step_metrics(self) -> dict[str, Any]:
        return {
            "active": bool(self._active),
            "activation_id": int(self._activation_id),
            "phase": self._phase,
            "ttl_remaining": int(self._ttl_remaining),
            "stall_remaining": int(self._stall_remaining),
            "missing_compatible_ticket_count": int(self._missing_compatible_ticket_count),
            "activation_count": int(self._activation_count),
            "active_step_count": int(self._active_step_count),
            "phase_transition_histogram": dict(self._phase_transition_histogram),
            "abort_count": int(self._abort_count),
            "abort_reason_histogram": dict(self._abort_reason_histogram),
            "completion_count": int(self._completion_count),
            "completion_reason_histogram": dict(self._completion_reason_histogram),
            "route_source_histogram": dict(self._route_source_histogram),
            "generic_preemption_count": int(self._generic_preemption_count),
            "last_abort_reason": self._last_abort_reason,
            "last_completed_reason": self._last_completed_reason,
            "last_route_source": self._route_source,
            "route_preview": list(self._route_preview),
        }

    episode_summary = step_metrics

    def _normalize_ticket(self, ticket: AdapterExecutionTicket | None) -> AdapterExecutionTicket | None:
        if ticket is None:
            return None
        if ticket.kind != "hidden_state_branch":
            return None
        if ticket.ownership != "exclusive":
            return None
        if ticket.phase not in _PHASE_ORDER:
            return None
        return ticket

    def _activate(
        self,
        *,
        ticket: AdapterExecutionTicket,
        player_anchor: tuple[int, int] | None,
        score: int,
        level_number: int,
    ) -> None:
        self._active = True
        self._activation_id += 1
        self._activation_count += 1
        self._kind = ticket.kind
        self._phase = ticket.phase
        self._ownership = ticket.ownership
        self._ttl_remaining = max(1, int(ticket.ttl_steps))
        self._stall_remaining = max(1, int(ticket.stall_budget))
        self._missing_compatible_ticket_count = 0
        self._start_score = int(score)
        self._start_level_number = int(level_number)
        self._last_player_anchor = player_anchor
        self._last_phase = ticket.phase
        self._last_abort_reason = ""
        self._last_completed_reason = ""
        self._absorb_ticket(ticket)

    def _lease(
        self,
        *,
        active: bool,
        abort_reason: str = "",
        completed_reason: str = "",
    ) -> HiddenStateExecutorLease:
        return HiddenStateExecutorLease(
            active=active,
            activation_id=int(self._activation_id),
            kind=self._kind,
            phase=self._phase,
            ownership=self._ownership,
            route_preview=list(self._route_preview),
            target_anchor=list(self._target_anchor),
            site_anchor=list(self._site_anchor),
            goal_anchor=list(self._goal_anchor),
            provenance=self._provenance,
            route_source=self._route_source,
            ttl_remaining=int(self._ttl_remaining),
            stall_remaining=int(self._stall_remaining),
            missing_compatible_ticket_count=int(self._missing_compatible_ticket_count),
            metadata=dict(self._metadata),
            abort_reason=abort_reason,
            completed_reason=completed_reason,
        )

    def _complete(self, reason: str) -> HiddenStateExecutorLease:
        normalized_reason = str(reason or "completed").strip().lower()
        self._completion_count += 1
        self._completion_reason_histogram[normalized_reason] = (
            self._completion_reason_histogram.get(normalized_reason, 0) + 1
        )
        self._last_completed_reason = normalized_reason
        lease = self._lease(active=False, completed_reason=normalized_reason)
        self._clear_active_state()
        return lease

    def _abort(self, reason: str) -> HiddenStateExecutorLease:
        normalized_reason = str(reason or "aborted").strip().lower()
        self._abort_count += 1
        self._abort_reason_histogram[normalized_reason] = (
            self._abort_reason_histogram.get(normalized_reason, 0) + 1
        )
        self._last_abort_reason = normalized_reason
        lease = self._lease(active=False, abort_reason=normalized_reason)
        self._clear_active_state()
        return lease

    def _clear_active_state(self) -> None:
        self._active = False
        self._kind = ""
        self._phase = ""
        self._ownership = ""
        self._ttl_remaining = 0
        self._stall_remaining = 0
        self._missing_compatible_ticket_count = 0
        self._route_preview = []
        self._target_anchor = []
        self._site_anchor = []
        self._goal_anchor = []
        self._provenance = ""
        self._route_source = ""
        self._metadata = {}
        self._last_player_anchor = None
        self._last_phase = ""

    def _progressed(self, *, current_anchor: tuple[int, int] | None) -> bool:
        if current_anchor is None:
            return False
        return current_anchor != self._last_player_anchor

    def _record_transition(self, before: str, after: str) -> None:
        transition_key = f"{before or 'none'}->{after or 'none'}"
        self._phase_transition_histogram[transition_key] = (
            self._phase_transition_histogram.get(transition_key, 0) + 1
        )

    def _record_route_source(self) -> None:
        if not self._route_source:
            return
        self._route_source_histogram[self._route_source] = (
            self._route_source_histogram.get(self._route_source, 0) + 1
        )

    def _consume_route_head_for_new_step(self) -> None:
        if not self._route_preview:
            return
        self._route_preview = list(self._route_preview[1:])

    def _should_preserve_route_preview(self, ticket: AdapterExecutionTicket) -> bool:
        if not self._route_preview:
            return False
        if self._phase == "site_actuation" and ticket.phase == "site_actuation":
            return True
        site_route_phases = {"branch_entry", "site_reaching"}
        if self._phase in site_route_phases and ticket.phase in site_route_phases:
            current_target = [
                int(value)
                for value in list(self._target_anchor or [])[:2]
                if str(value).strip("-").isdigit()
            ]
            ticket_target = [
                int(value)
                for value in list(ticket.target_anchor or [])[:2]
                if str(value).strip("-").isdigit()
            ]
            if current_target and ticket_target and current_target != ticket_target:
                return False
            return True
        goal_route_phases = {"state_change_verification", "goal_route_execution"}
        if self._phase not in goal_route_phases or ticket.phase not in goal_route_phases:
            return False
        current_target = [
            int(value)
            for value in list(self._target_anchor or [])[:2]
            if str(value).strip("-").isdigit()
        ]
        ticket_target = [
            int(value)
            for value in list(ticket.target_anchor or [])[:2]
            if str(value).strip("-").isdigit()
        ]
        if current_target and ticket_target and current_target != ticket_target:
            return False
        return True

    def _route_head_invalid_for_current_anchor(
        self,
        *,
        board_rows: list[str],
        current_anchor: tuple[int, int] | None,
    ) -> bool:
        if not self._route_preview or current_anchor is None or not board_rows or not board_rows[0]:
            return False
        action_name = str(self._route_preview[0] or "").strip().upper()
        delta = _MOVE_DELTAS.get(action_name)
        if delta is None:
            return True
        next_anchor = (int(current_anchor[0]) + int(delta[0]), int(current_anchor[1]) + int(delta[1]))
        return not self._anchor_passable(board_rows, next_anchor)

    @staticmethod
    def _anchor_passable(board_rows: list[str], anchor: tuple[int, int]) -> bool:
        row, col = int(anchor[0]), int(anchor[1])
        if row < 0 or col < 0:
            return False
        if row + _PLAYER_BBOX_HEIGHT > len(board_rows):
            return False
        if col + _PLAYER_BBOX_WIDTH > len(board_rows[0]):
            return False
        for rr in range(row, row + _PLAYER_BBOX_HEIGHT):
            row_text = board_rows[rr]
            for cc in range(col, col + _PLAYER_BBOX_WIDTH):
                if cc >= len(row_text) or row_text[cc] == "q":
                    return False
        return True

    def _is_completion(
        self,
        *,
        score: int,
        level_number: int,
        ticket: AdapterExecutionTicket | None,
    ) -> bool:
        if not self._active:
            return False
        if self._phase == "goal_route_execution":
            if int(score) > int(self._start_score):
                return True
            if int(level_number) > int(self._start_level_number):
                return True
        if ticket is not None and bool(ticket.metadata.get("objective_resolved", False)):
            return True
        return False

    def _is_compatible_transition(self, ticket: AdapterExecutionTicket) -> bool:
        if not self._active:
            return True
        next_phase = str(ticket.phase or "").strip().lower()
        current_idx = _PHASE_ORDER.get(self._phase, -1)
        next_idx = _PHASE_ORDER.get(next_phase, -1)
        if current_idx < 0 or next_idx < 0:
            return False
        if next_idx == current_idx:
            return True
        if (
            current_idx == _PHASE_ORDER["site_reaching"]
            and next_phase == "branch_entry"
            and bool(ticket.metadata.get("allow_recovery_edge", False))
        ):
            return True
        if (
            self._phase == "site_actuation"
            and next_phase == "site_reaching"
            and bool(ticket.metadata.get("alternate_site_route_active", False))
        ):
            return True
        return next_idx == current_idx + 1

    def _should_ignore_competing_site_ticket(self, ticket: AdapterExecutionTicket) -> bool:
        if self._phase not in {"branch_entry", "site_reaching"}:
            return False
        if str(ticket.phase or "").strip().lower() not in {"branch_entry", "site_reaching", "site_actuation"}:
            return False
        if not self._current_site_route_owned():
            return False
        if not self._route_preview:
            if not self._next_arrival_probe_sequence():
                return False
        current_anchor = self._first_anchor(self._target_anchor) or self._first_anchor(self._site_anchor)
        ticket_anchor = self._first_anchor(ticket.target_anchor) or self._first_anchor(ticket.site_anchor)
        if current_anchor is None or ticket_anchor is None:
            return False
        if current_anchor == ticket_anchor:
            return False
        distance = abs(int(current_anchor[0]) - int(ticket_anchor[0])) + abs(int(current_anchor[1]) - int(ticket_anchor[1]))
        return distance > 8

    def _current_site_route_owned(self) -> bool:
        return bool(
            self._metadata.get("alternate_site_route_active", False)
            or self._metadata.get("resource_route_active", False)
            or "alternate_site" in str(self._route_source or "").strip().lower()
            or "resource_prefixed" in str(self._route_source or "").strip().lower()
        )

    def _seed_route_arrival_probe_if_needed(self) -> None:
        if self._route_preview:
            return
        if self._phase != "site_reaching":
            return
        sequence = self._next_arrival_probe_sequence()
        if not sequence:
            return
        index = int(self._metadata.get("route_arrival_probe_index", 0) or 0)
        self._route_preview = list(sequence)
        self._metadata["route_arrival_probe_index"] = index + 1
        self._metadata["route_arrival_probe_active"] = True
        self._ttl_remaining = max(int(self._ttl_remaining), len(sequence) + 1)
        self._stall_remaining = max(int(self._stall_remaining), 2)

    def _next_arrival_probe_sequence(self) -> list[str]:
        sequences = self._arrival_probe_sequences()
        if not sequences:
            return []
        try:
            index = int(self._metadata.get("route_arrival_probe_index", 0) or 0)
        except (TypeError, ValueError):
            index = 0
        if index < 0 or index >= len(sequences):
            return []
        return list(sequences[index])

    def _arrival_probe_sequences(self) -> list[list[str]]:
        raw_sequences = self._metadata.get("route_arrival_probe_sequences", [])
        if not isinstance(raw_sequences, (list, tuple)):
            return []
        sequences: list[list[str]] = []
        for raw_sequence in list(raw_sequences):
            if not isinstance(raw_sequence, (list, tuple)):
                continue
            sequence = [
                str(action or "").strip().upper()
                for action in list(raw_sequence)
                if str(action or "").strip().upper() in _MOVE_DELTAS
            ]
            if sequence:
                sequences.append(sequence)
        return sequences

    @staticmethod
    def _first_anchor(values: list[int] | tuple[int, ...] | None) -> tuple[int, int] | None:
        if not values or len(values) < 2:
            return None
        try:
            return int(values[0]), int(values[1])
        except (TypeError, ValueError):
            return None

    def _anchors_invalid(self, board_rows: list[str]) -> bool:
        if not board_rows or not board_rows[0]:
            return False
        board_height = len(board_rows)
        board_width = len(board_rows[0])
        for anchor in (self._target_anchor, self._site_anchor, self._goal_anchor):
            if len(anchor) < 2:
                continue
            row = int(anchor[0])
            col = int(anchor[1])
            if row < 0 or row >= board_height or col < 0 or col >= board_width:
                return True
        return False

    def _absorb_ticket(
        self,
        ticket: AdapterExecutionTicket,
        *,
        preserve_route_preview: bool = False,
    ) -> None:
        if not preserve_route_preview:
            self._route_preview = list(ticket.route_preview)
        self._target_anchor = list(ticket.target_anchor)
        self._site_anchor = list(ticket.site_anchor)
        self._goal_anchor = list(ticket.goal_anchor)
        self._provenance = str(ticket.provenance or "").strip()
        self._metadata = dict(ticket.metadata)
        self._route_source = str(
            ticket.metadata.get("route_source", "")
            or ticket.provenance
            or ticket.metadata.get("source", "")
            or ""
        ).strip()

    @staticmethod
    def _player_anchor(
        player_bbox: list[int] | tuple[int, int, int, int] | None,
    ) -> tuple[int, int] | None:
        if not player_bbox or len(player_bbox) != 4:
            return None
        return int(player_bbox[0]), int(player_bbox[2])
