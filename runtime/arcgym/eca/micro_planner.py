"""MicroPlanner: Local action queue generation for known action families.

Without this module, even when ControlIR produces correct IR instructions,
the system still needs the Analyzer LLM to convert them into action queues.
The MicroPlanner eliminates this dependency for known, high-frequency
action families by generating short (1-3 step) action queues locally.

The Analyzer is called only when:
    - MicroPlanner cannot build a queue (unknown family)
    - Ticket conflicts are detected
    - 2+ consecutive local failures
    - Candidate family entropy is too high

Input contract:
    - selected_target: dict with component_id, action_family, etc.
    - action_family: str (e.g., "click", "move", "color_fill")
    - allowed_actions: list of action names the environment accepts
    - blocked_actions: set of action names currently blocked
    - ttl: remaining turns for the active ticket
    - board_delta: optional dict describing recent board changes

Output:
    - A short action queue (1-3 actions) or None if the family is unknown.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from arcgym.eca.control_ir_compiler import CompiledIR
from arcgym.eca.candidate_resolver import ScoredCandidate, ResolutionResult

log = logging.getLogger(__name__)
_CLICK_ACTION_NAME = "ACTION6"
_MOVE_ACTION_NAMES = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}


def _normalized_action_set(actions: list[str] | set[str] | None) -> set[str]:
    return {
        str(action or "").strip().upper()
        for action in list(actions or [])
        if str(action or "").strip()
    }


# ─── Planned Action ────────────────────────────────────────────────────────

@dataclass
class PlannedAction:
    """A single action in a micro-planned queue."""
    action_name: str  # e.g., "click", "move", "color_fill"
    target_component_id: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "target_component_id": self.target_component_id,
            "parameters": dict(self.parameters),
            "reason": self.reason,
        }


@dataclass
class MicroPlanResult:
    """Result of a micro-planning pass."""
    actions: list[PlannedAction] = field(default_factory=list)
    planned: bool = False
    family: str = ""
    reason: str = ""
    needs_analyzer: bool = False
    analyzer_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "planned": self.planned,
            "family": self.family,
            "action_count": len(self.actions),
            "actions": [a.to_dict() for a in self.actions],
            "reason": self.reason,
            "needs_analyzer": self.needs_analyzer,
            "analyzer_reason": self.analyzer_reason,
        }


# ─── MicroPlanner Configuration ────────────────────────────────────────────

@dataclass
class MicroPlannerConfig:
    """Configuration for the MicroPlanner."""
    # Maximum actions per micro-plan
    max_actions_per_plan: int = 3
    # Maximum consecutive failures before triggering analyzer fallback
    max_consecutive_failures: int = 2
    # Whether to enable micro-planning (can be disabled for A/B testing)
    enabled: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "MicroPlannerConfig":
        if not raw:
            return cls()
        return cls(
            max_actions_per_plan=max(1, int(raw.get("max_actions_per_plan", 3))),
            max_consecutive_failures=max(1, int(raw.get("max_consecutive_failures", 2))),
            enabled=bool(raw.get("enabled", True)),
        )


# ─── Family Handlers ───────────────────────────────────────────────────────

class _FamilyHandler:
    """Base class for family-specific action queue builders."""

    family: str = ""

    def can_handle(
        self,
        action_family: str,
        allowed_actions: list[str],
        blocked_actions: set[str],
    ) -> bool:
        """Check if this handler can produce a queue for the given family."""
        if action_family != self.family:
            return False
        if self.family in blocked_actions:
            return False
        if self.family not in allowed_actions and allowed_actions:
            return False
        return True

    def build_queue(
        self,
        target: dict[str, Any],
        allowed_actions: list[str],
        blocked_actions: set[str],
        ttl: int,
        board_delta: dict[str, Any] | None = None,
    ) -> list[PlannedAction]:
        """Build a short action queue. Override in subclasses."""
        return []


class _ClickHandler(_FamilyHandler):
    """Handler for 'click' action family."""
    family = "click"

    def can_handle(
        self,
        action_family: str,
        allowed_actions: list[str],
        blocked_actions: set[str],
    ) -> bool:
        if action_family != self.family:
            return False
        blocked = _normalized_action_set(blocked_actions)
        if self.family.upper() in blocked or _CLICK_ACTION_NAME in blocked:
            return False
        allowed = _normalized_action_set(allowed_actions)
        return not allowed or self.family.upper() in allowed or _CLICK_ACTION_NAME in allowed

    def build_queue(
        self,
        target: dict[str, Any],
        allowed_actions: list[str],
        blocked_actions: set[str],
        ttl: int,
        board_delta: dict[str, Any] | None = None,
    ) -> list[PlannedAction]:
        comp_id = str(target.get("component_id", ""))
        position = target.get("position", target.get("coordinates", {}))
        return [PlannedAction(
            action_name="click",
            target_component_id=comp_id,
            parameters={"position": position} if position else {},
            reason=f"IR-directed click on {comp_id}",
        )]


class _MoveHandler(_FamilyHandler):
    """Handler for 'move' action family."""
    family = "move"

    def can_handle(
        self,
        action_family: str,
        allowed_actions: list[str],
        blocked_actions: set[str],
    ) -> bool:
        if action_family != self.family:
            return False
        blocked = _normalized_action_set(blocked_actions)
        if self.family.upper() in blocked:
            return False
        allowed = _normalized_action_set(allowed_actions)
        if not allowed or self.family.upper() in allowed:
            return True
        return bool(_MOVE_ACTION_NAMES & allowed)

    def build_queue(
        self,
        target: dict[str, Any],
        allowed_actions: list[str],
        blocked_actions: set[str],
        ttl: int,
        board_delta: dict[str, Any] | None = None,
    ) -> list[PlannedAction]:
        comp_id = str(target.get("component_id", ""))
        direction = target.get("direction", "")
        return [PlannedAction(
            action_name="move",
            target_component_id=comp_id,
            parameters={"direction": direction} if direction else {},
            reason=f"IR-directed move on {comp_id}",
        )]


class _AuxiliaryHandler(_FamilyHandler):
    """Handler for direct non-click/non-move ACTIONn controls."""
    family = "auxiliary"

    def can_handle(
        self,
        action_family: str,
        allowed_actions: list[str],
        blocked_actions: set[str],
    ) -> bool:
        return action_family == self.family and self.family not in blocked_actions

    def build_queue(
        self,
        target: dict[str, Any],
        allowed_actions: list[str],
        blocked_actions: set[str],
        ttl: int,
        board_delta: dict[str, Any] | None = None,
    ) -> list[PlannedAction]:
        _ = ttl, board_delta
        comp_id = str(target.get("component_id", ""))
        action_name = str(target.get("action_name", "") or "").strip().upper()
        if not action_name:
            match = re.search(r"(?:^|::)(?:ACTION)?(\d+)$", comp_id.strip(), re.IGNORECASE)
            if match:
                action_name = f"ACTION{int(match.group(1))}"
        if not re.match(r"^ACTION\d+$", action_name):
            return []
        if action_name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6", "RESET"}:
            return []
        normalized_allowed = _normalized_action_set(allowed_actions)
        normalized_blocked = _normalized_action_set(blocked_actions)
        if normalized_allowed and action_name not in normalized_allowed:
            return []
        if action_name in normalized_blocked:
            return []
        return [
            PlannedAction(
                action_name=action_name,
                target_component_id=comp_id,
                parameters={},
                reason=f"IR-directed auxiliary {action_name} on {comp_id}",
            )
        ]


class _ColorFillHandler(_FamilyHandler):
    """Handler for 'color_fill' action family."""
    family = "color_fill"

    def build_queue(
        self,
        target: dict[str, Any],
        allowed_actions: list[str],
        blocked_actions: set[str],
        ttl: int,
        board_delta: dict[str, Any] | None = None,
    ) -> list[PlannedAction]:
        comp_id = str(target.get("component_id", ""))
        color = target.get("color", target.get("fill_color", ""))
        position = target.get("position", target.get("coordinates", {}))
        actions: list[PlannedAction] = []
        if color and position:
            actions.append(PlannedAction(
                action_name="color_fill",
                target_component_id=comp_id,
                parameters={"color": color, "position": position},
                reason=f"IR-directed color_fill on {comp_id}",
            ))
        elif color:
            actions.append(PlannedAction(
                action_name="color_fill",
                target_component_id=comp_id,
                parameters={"color": color},
                reason=f"IR-directed color_fill on {comp_id}",
            ))
        return actions


class _DragHandler(_FamilyHandler):
    """Handler for 'drag' action family."""
    family = "drag"

    def build_queue(
        self,
        target: dict[str, Any],
        allowed_actions: list[str],
        blocked_actions: set[str],
        ttl: int,
        board_delta: dict[str, Any] | None = None,
    ) -> list[PlannedAction]:
        comp_id = str(target.get("component_id", ""))
        from_pos = target.get("from_position", target.get("source", {}))
        to_pos = target.get("to_position", target.get("destination", {}))
        params: dict[str, Any] = {}
        if from_pos:
            params["from_position"] = from_pos
        if to_pos:
            params["to_position"] = to_pos
        return [PlannedAction(
            action_name="drag",
            target_component_id=comp_id,
            parameters=params,
            reason=f"IR-directed drag on {comp_id}",
        )]


class _CopyHandler(_FamilyHandler):
    """Handler for 'copy' action family."""
    family = "copy"

    def build_queue(
        self,
        target: dict[str, Any],
        allowed_actions: list[str],
        blocked_actions: set[str],
        ttl: int,
        board_delta: dict[str, Any] | None = None,
    ) -> list[PlannedAction]:
        comp_id = str(target.get("component_id", ""))
        source = target.get("source", {})
        destination = target.get("destination", {})
        params: dict[str, Any] = {}
        if source:
            params["source"] = source
        if destination:
            params["destination"] = destination
        return [PlannedAction(
            action_name="copy",
            target_component_id=comp_id,
            parameters=params,
            reason=f"IR-directed copy on {comp_id}",
        )]


class _FloodFillHandler(_FamilyHandler):
    """Handler for 'flood_fill' action family."""
    family = "flood_fill"

    def build_queue(
        self,
        target: dict[str, Any],
        allowed_actions: list[str],
        blocked_actions: set[str],
        ttl: int,
        board_delta: dict[str, Any] | None = None,
    ) -> list[PlannedAction]:
        comp_id = str(target.get("component_id", ""))
        color = target.get("color", target.get("fill_color", ""))
        position = target.get("position", target.get("seed", {}))
        params: dict[str, Any] = {}
        if color:
            params["color"] = color
        if position:
            params["position"] = position
        return [PlannedAction(
            action_name="flood_fill",
            target_component_id=comp_id,
            parameters=params,
            reason=f"IR-directed flood_fill on {comp_id}",
        )]


class _DrawLineHandler(_FamilyHandler):
    """Handler for 'draw_line' action family."""
    family = "draw_line"

    def build_queue(
        self,
        target: dict[str, Any],
        allowed_actions: list[str],
        blocked_actions: set[str],
        ttl: int,
        board_delta: dict[str, Any] | None = None,
    ) -> list[PlannedAction]:
        comp_id = str(target.get("component_id", ""))
        start = target.get("start", target.get("from_position", {}))
        end = target.get("end", target.get("to_position", {}))
        color = target.get("color", "")
        params: dict[str, Any] = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if color:
            params["color"] = color
        return [PlannedAction(
            action_name="draw_line",
            target_component_id=comp_id,
            parameters=params,
            reason=f"IR-directed draw_line on {comp_id}",
        )]


class _SelectHandler(_FamilyHandler):
    """Handler for 'select' action family."""
    family = "select"

    def build_queue(
        self,
        target: dict[str, Any],
        allowed_actions: list[str],
        blocked_actions: set[str],
        ttl: int,
        board_delta: dict[str, Any] | None = None,
    ) -> list[PlannedAction]:
        comp_id = str(target.get("component_id", ""))
        position = target.get("position", target.get("coordinates", {}))
        return [PlannedAction(
            action_name="select",
            target_component_id=comp_id,
            parameters={"position": position} if position else {},
            reason=f"IR-directed select on {comp_id}",
        )]


class _ResetHandler(_FamilyHandler):
    """Handler for 'reset' action family."""
    family = "reset"

    def build_queue(
        self,
        target: dict[str, Any],
        allowed_actions: list[str],
        blocked_actions: set[str],
        ttl: int,
        board_delta: dict[str, Any] | None = None,
    ) -> list[PlannedAction]:
        return [PlannedAction(
            action_name="reset",
            parameters={},
            reason="IR-directed reset",
        )]


# ─── Handler Registry ──────────────────────────────────────────────────────

_ALL_HANDLERS: list[_FamilyHandler] = [
    _ClickHandler(),
    _MoveHandler(),
    _AuxiliaryHandler(),
    _ColorFillHandler(),
    _DragHandler(),
    _CopyHandler(),
    _FloodFillHandler(),
    _DrawLineHandler(),
    _SelectHandler(),
    _ResetHandler(),
]

_HANDLER_MAP: dict[str, _FamilyHandler] = {h.family: h for h in _ALL_HANDLERS}


# ─── MicroPlanner ──────────────────────────────────────────────────────────

class MicroPlanner:
    """Local action queue generator for known action families.

    Usage:
        planner = MicroPlanner(config)
        result = planner.plan(
            resolution=resolver_result,
            instructions=compilation_result.instructions,
            allowed_actions=["click", "move", "color_fill", ...],
            blocked_actions=set(),
            board_delta=None,
        )
        if result.planned:
            # Use result.actions to set the action queue
        elif result.needs_analyzer:
            # Fall back to Analyzer LLM
    """

    def __init__(self, config: MicroPlannerConfig | None = None) -> None:
        self.config = config or MicroPlannerConfig()
        self._consecutive_failures: int = 0
        self._total_planned: int = 0
        self._total_fallback: int = 0

    def reset(self) -> None:
        """Reset for a new episode."""
        self._consecutive_failures = 0
        self._total_planned = 0
        self._total_fallback = 0

    def plan(
        self,
        *,
        resolution: ResolutionResult | None = None,
        instructions: list[CompiledIR] | None = None,
        allowed_actions: list[str] | None = None,
        blocked_actions: set[str] | None = None,
        board_delta: dict[str, Any] | None = None,
    ) -> MicroPlanResult:
        """Run one micro-planning pass.

        Args:
            resolution: Result from CandidateResolver.
            instructions: Compiled IR instructions.
            allowed_actions: List of action names the environment accepts.
            blocked_actions: Set of action names currently blocked.
            board_delta: Optional dict describing recent board changes.

        Returns:
            MicroPlanResult with the planned actions or analyzer fallback flag.
        """
        result = MicroPlanResult()
        allowed = allowed_actions or []
        blocked = blocked_actions or set()

        if not self.config.enabled:
            result.needs_analyzer = True
            result.analyzer_reason = "micro_planner_disabled"
            return result

        # Check consecutive failure threshold
        if self._consecutive_failures >= self.config.max_consecutive_failures:
            result.needs_analyzer = True
            result.analyzer_reason = f"consecutive_failures={self._consecutive_failures}"
            self._total_fallback += 1
            return result

        # Determine the target and action family
        target_info, action_family = self._extract_target_info(
            resolution, instructions
        )
        if not action_family:
            result.needs_analyzer = True
            result.analyzer_reason = "no_action_family"
            self._total_fallback += 1
            return result

        result.family = action_family

        # Check if we have a handler for this family
        handler = _HANDLER_MAP.get(action_family)
        if handler is None:
            result.needs_analyzer = True
            result.analyzer_reason = f"unknown_family:{action_family}"
            self._total_fallback += 1
            return result

        # Check if the handler can produce a queue
        if not handler.can_handle(action_family, allowed, blocked):
            result.needs_analyzer = True
            result.analyzer_reason = f"family_blocked_or_not_allowed:{action_family}"
            self._total_fallback += 1
            return result

        # Determine TTL from instructions
        ttl = self._get_ttl(instructions)

        # Build the queue
        try:
            actions = handler.build_queue(
                target=target_info,
                allowed_actions=allowed,
                blocked_actions=blocked,
                ttl=ttl,
                board_delta=board_delta,
            )
        except Exception as e:
            log.warning("MicroPlanner handler error for %s: %s", action_family, e)
            actions = []

        if not actions:
            self._consecutive_failures += 1
            result.needs_analyzer = True
            result.analyzer_reason = f"handler_produced_empty_queue:{action_family}"
            self._total_fallback += 1
            return result

        # Trim to max actions
        actions = actions[:self.config.max_actions_per_plan]

        result.actions = actions
        result.planned = True
        result.reason = f"micro_planned:{action_family}:{len(actions)}_actions"
        self._consecutive_failures = 0
        self._total_planned += 1
        return result

    def record_outcome(self, success: bool) -> None:
        """Record the outcome of the last micro-planned queue.

        Call this after the environment processes the planned actions.
        """
        if success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_planned": self._total_planned,
            "total_fallback": self._total_fallback,
            "consecutive_failures": self._consecutive_failures,
            "success_rate": (
                self._total_planned / (self._total_planned + self._total_fallback)
                if (self._total_planned + self._total_fallback) > 0
                else 0.0
            ),
        }

    # ── Private helpers ─────────────────────────────────────────────────

    def _extract_target_info(
        self,
        resolution: ResolutionResult | None,
        instructions: list[CompiledIR] | None,
    ) -> tuple[dict[str, Any], str]:
        """Extract target info and action family from resolution/instructions."""
        target_info: dict[str, Any] = {}
        action_family = ""

        # Prefer resolution result
        if resolution and resolution.resolved and resolution.selected:
            sel = resolution.selected
            target_info = {
                "component_id": sel.component_id,
                "action_family": sel.action_family,
                "action_name": sel.action_name,
                "source": sel.source,
            }
            action_family = sel.action_family

        # Fallback to IR instructions
        if not action_family and instructions:
            for ir in instructions:
                if ir.target_action_family:
                    target_info = {
                        "component_id": ir.target_component_id,
                        "action_family": ir.target_action_family,
                        "source": ir.target_source,
                    }
                    action_family = ir.target_action_family
                    break

        return target_info, action_family

    def _get_ttl(self, instructions: list[CompiledIR] | None) -> int:
        """Get the minimum TTL from active instructions."""
        if not instructions:
            return 3  # default
        ttls = [ir.ttl_remaining for ir in instructions if ir.ttl_remaining > 0]
        return min(ttls) if ttls else 3
