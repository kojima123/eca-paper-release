"""Gameplay-only effect-to-hypothesis discovery for L2+ rules.

The store is intentionally game-agnostic: it observes local action effects,
classifies them by generic board-change shape, and emits short follow-up
candidates without reading task IDs, environment source, or adapter metadata.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from arcgym.eca.affordance_graph import AffordanceGraph
from arcgym.eca.effect_ledger import EffectLedger, EffectRecord


def _stabilize_lease_episode_budget_override() -> int | None:
    """Phase 8.F flag: env override for the stabilize-lease per-episode
    cap (default 4). Set `STABILIZE_LEASE_EPISODE_BUDGET=N` to admit up
    to N leases per episode. Default unset → preserves original cap.

    The 19-game breadth diagnostic showed vc33 opening 73 stabilize
    windows but admitting only 4 leases (33 rejected with reason
    `episode_budget`), keeping `stabilize_verify_closed_count` at 1.
    Raising the cap lets more high-quality windows convert into
    closure attempts; the per-window admission still gates on score
    quality / observed_count / followup budget so the lift is bounded.
    """
    raw = os.environ.get("STABILIZE_LEASE_EPISODE_BUDGET", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 0:
        return None
    return value


def _enable_goal_probe_l1_quota() -> bool:
    """Phase 8.D flag: relax the L2 gate on goal_probe_candidates when
    the L1 quota is engaged but the game is not scoring. Lets the
    affordance graph emit click-family candidates targeting components
    that have changed under quota-driven keyboard probing.

    Default off. Enable with `ENABLE_GOAL_PROBE_L1_QUOTA=1`.

    Engagement test: at least one record in the L1 effect ledger has
    `selected_source == "l1_action_family_quota"`. Without this gate,
    quota-engaged L1 games (ar25/g50t/re86/sp80/...) never exercise
    the goal_probe path because it's level_idx >= 2 only.
    """
    raw = os.environ.get("ENABLE_GOAL_PROBE_L1_QUOTA", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _enable_expanded_closure_sources() -> bool:
    """Phase 8.E-2 flag: expand the closure-eligible source set in
    record_effect so that score events from sources outside
    {goal_probe, rule_hypothesis, stabilize_verify} also reinforce
    rule hypotheses via _mark_selected_rule_closed.

    Default off. Enable with `ENABLE_EXPANDED_CLOSURE_SOURCES=1`.

    Background: phase 8.E followup found that 80% of score-bearing
    events across 25 games come from sources NOT in the legacy
    closure-eligible set (visual_stencil_composition,
    auxiliary_visual_arrangement, object_escape, local_click_fanout,
    level_rule_symbol_scout, provisional_base_pool). Expanding the
    set routes those events through _mark_selected_rule_closed so
    matching hypotheses are marked closed and earn the
    closure_status="closed" effect_bonus in candidate ranking.
    See docs/phase8d_goal_probe_l1_relaxation_2026-04-28.md.
    """
    raw = os.environ.get("ENABLE_EXPANDED_CLOSURE_SOURCES", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _enable_l1_quota_synthetic_anchor() -> bool:
    """Phase 8.A2 flag: relax the rule_discovery L1 gate so that
    move-family L1 quota emissions also pass through `record_effect`.

    Default off. Enable with `ENABLE_L1_QUOTA_SYNTHETIC_ANCHOR=1`.

    Background: zero-score breakdown found that sc25 / tr87 / tu93
    fire L1 quota actions hundreds of times but their quota emits
    only ACTION1-4 (move family), no ACTION5/7. The legacy gate
    `allow_l1_auxiliary_discovery` requires `family == "auxiliary"`,
    so move-family quota records bypass the rule_discovery effect
    ledger. This flag also accepts `family == "move"` when the
    source is `l1_action_family_quota`, restoring engagement
    parity with auxiliary-family quota games (ar25 etc.).

    The flag name predates the redesign (originally for a synthetic
    anchor in runtime.py); kept for continuity.
    """
    raw = os.environ.get("ENABLE_L1_QUOTA_SYNTHETIC_ANCHOR", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _enable_score_event_closure() -> bool:
    """Phase 8.E-2-b flag: read state.score_delta directly inside
    record_effect for closure_signal/closure_reason computation only,
    bypassing the score_delta propagation gap (Phase 8.E bug). Other
    uses of score_delta_value inside record_effect (goal_probe success
    counters, stabilize_verify lease note, etc.) keep using the buggy
    arg value so the load-bearing exploration heuristic in
    _local_effect_quality (which lives in runtime.py) is untouched.

    Default off. Enable with `ENABLE_SCORE_EVENT_CLOSURE=1`.

    Combined with ENABLE_EXPANDED_CLOSURE_SOURCES=1 (Phase 8.E-2),
    this should produce score-labeled closures across the 80% of
    score-bearing events that previously bypassed closure or got
    constraint-labeled closures.

    See docs/phase8e2_expanded_closure_sources_2026-04-29.md for the
    null-effect E-2-alone result that motivated this narrow fallback.
    """
    raw = os.environ.get("ENABLE_SCORE_EVENT_CLOSURE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


_LEGACY_CLOSURE_ELIGIBLE_SOURCES = {"goal_probe", "rule_hypothesis", "stabilize_verify"}
_EXPANDED_CLOSURE_ELIGIBLE_SOURCES = _LEGACY_CLOSURE_ELIGIBLE_SOURCES | {
    "visual_stencil_composition",
    "auxiliary_visual_arrangement",
    "object_escape",
    "local_click_fanout",
    "level_rule_symbol_scout",
    "provisional_base_pool",
}


_MOVE_ACTION_NAMES = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
_CLICK_ACTION_NAME = "ACTION6"
_RULE_ACTION_FAMILIES = {"click", "move", "auxiliary"}


@dataclass
class RuleHypothesis:
    hypothesis_id: str
    kind: str
    level_idx: int
    source_component_id: str
    source_anchor_key: str
    source_center_row: int
    source_center_col: int
    target_component_id: str
    target_anchor_key: str
    target_center_row: int
    target_center_col: int
    action_family: str
    action_name: str
    candidate_source: str
    relation_dr: int
    relation_dc: int
    target_motion_dr: int = 0
    target_motion_dc: int = 0
    target_value: int = 0
    target_previous_value: int = 0
    target_size: int = 0
    effect_count: int = 0
    verification_count: int = 0
    disconfirm_count: int = 0
    scoreless_success_count: int = 0
    scoreless_progress_count: int = 0
    followup_attempts: int = 0
    status: str = "active"
    score: float = 0.0
    created_step: int = -1
    last_step: int = -1
    expires_at: int = -1
    last_effect_kind: str = ""
    last_progress_signature: str = ""
    closure_status: str = "observed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuleDiscoveryConfig:
    ttl_steps: int = 24
    verification_budget: int = 3
    max_active: int = 16
    max_candidates: int = 4
    max_goal_probe_candidates: int = 3
    goal_probe_attempt_budget: int = 2
    visual_stencil_attempt_budget: int = 2
    stabilize_window_ttl_steps: int = 4
    stabilize_verify_attempt_budget: int = 2
    stabilize_closure_grace_steps: int = 8
    stabilize_lease_steps_max: int = 3
    stabilize_lease_nochange_patience: int = 1
    stabilize_lease_attempt_budget: int = 2
    stabilize_lease_episode_budget: int = 4
    stabilize_lease_admit_min_score: float = 1.35
    stabilize_aux_closer_move_budget: int = 2


@dataclass
class _ComponentDelta:
    component_id: str
    anchor_key: str
    center_row: int
    center_col: int
    previous_center_row: int
    previous_center_col: int
    value: int
    previous_value: int | None
    size: int
    previous_size: int | None
    change_kind: str
    bbox: dict[str, Any] = field(default_factory=dict)


@dataclass
class _EffectActuator:
    level_idx: int
    source_component_id: str
    source_anchor_key: str
    source_center_row: int
    source_center_col: int
    target_component_id: str
    target_anchor_key: str
    target_center_row: int
    target_center_col: int
    action_family: str
    action_name: str
    support_count: int = 0
    last_step: int = -1
    score: float = 0.0
    last_effect_kind: str = ""
    visual_repair_progress_count: int = 0
    visual_repair_regress_count: int = 0
    visual_repair_neutral_count: int = 0


@dataclass
class _StabilizeVerifyWindow:
    window_id: str
    level_idx: int
    source_component_id: str
    source_anchor_key: str
    source_center_row: int
    source_center_col: int
    relation_component_id: str
    relation_anchor_key: str
    relation_center_row: int
    relation_center_col: int
    action_family: str
    action_name: str
    selected_source: str
    linked_effect_id: str = ""
    state: str = "unstable"
    closure_status: str = "observed"
    opened_step: int = -1
    last_step: int = -1
    expires_at: int = -1
    board_change_count: int = 0
    delayed_count: int = 0
    issued_count: int = 0
    verify_attempts: int = 0
    settled_counted: bool = False
    observed_component_ids: list[str] = field(default_factory=list)
    observed_anchor_keys: list[str] = field(default_factory=list)
    lease_active: bool = False
    lease_started_step: int = -1
    lease_expires_at: int = -1
    lease_followup_issued_count: int = 0
    lease_nochange_count: int = 0
    lease_preempt_count: int = 0
    scoreless_verified_counted: bool = False
    lease_admitted: bool = False
    lease_rejected: bool = False
    lease_admit_score: float = 0.0
    lease_reject_reason: str = ""
    continuation_candidate_keys: list[str] = field(default_factory=list)
    continuation_issued_count: int = 0

    def key(self) -> str:
        return "|".join(
            [
                str(int(self.level_idx)),
                str(self.source_component_id or self.source_anchor_key),
                str(self.relation_component_id or self.relation_anchor_key),
                str(self.action_name or ""),
            ]
        )


@dataclass
class _VisualConstraintRecord:
    constraint_id: str
    level_idx: int
    anchor_key: str
    target_component_id: str
    target_anchor_key: str
    offset_row: int
    offset_col: int
    expected_char: str
    observed_char: str
    first_seen_step: int
    last_seen_step: int
    status: str = "unresolved"
    candidate_key: str = ""
    resolved_step: int = -1
    attempt_count: int = 0
    progress_count: int = 0
    regress_count: int = 0
    neutral_count: int = 0
    last_outcome_delta: int = 0
    last_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class _VisualComponent:
    component_id: str
    center_row: int
    center_col: int
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    size: int
    char_counts: dict[str, int]

    @property
    def height(self) -> int:
        return int(self.max_row) - int(self.min_row) + 1

    @property
    def width(self) -> int:
        return int(self.max_col) - int(self.min_col) + 1

    @property
    def unique_count(self) -> int:
        return len(self.char_counts)

    @property
    def dominant_char(self) -> str:
        if not self.char_counts:
            return ""
        return max(self.char_counts.items(), key=lambda item: (int(item[1]), str(item[0])))[0]


class RuleDiscoveryStore:
    """Small per-episode hypothesis store for gameplay-only rule discovery."""

    def __init__(self, config: RuleDiscoveryConfig | None = None) -> None:
        self.config = config or RuleDiscoveryConfig()
        self._hypotheses: dict[str, RuleHypothesis] = {}
        self.created_count = 0
        self.verified_count = 0
        self.disconfirmed_count = 0
        self.scoreless_churn_count = 0
        self.followup_issued_count = 0
        self.followup_success_count = 0
        self.goal_relation_probe_count = 0
        self.constraint_probe_count = 0
        self.source_burst_count = 0
        self.progress_lease_count = 0
        self.source_progress_suppressed_count = 0
        self.perimeter_effect_suppressed_count = 0
        self.invariant_violation_count = 0
        self.goal_probe_candidate_count = 0
        self.goal_probe_issued_count = 0
        self.goal_probe_success_count = 0
        self.goal_probe_board_change_success_count = 0
        self.goal_probe_score_success_count = 0
        self.goal_probe_level_success_count = 0
        self.goal_probe_constraint_progress_success_count = 0
        self.rule_closed_success_count = 0
        self.rule_closed_score_count = 0
        self.rule_closed_level_count = 0
        self.rule_closed_constraint_count = 0
        self.rule_closed_synthetic_count = 0
        self.closed_count = 0
        self.actuator_group_detected_count = 0
        self.actuator_group_completed_count = 0
        self.stabilize_verify_window_opened_count = 0
        self.stabilize_verify_window_settled_count = 0
        self.stabilize_verify_issued_count = 0
        self.stabilize_verify_closed_count = 0
        self.stabilize_verify_expired_count = 0
        self.stabilize_window_lease_started_count = 0
        self.stabilize_window_lease_followup_issued_count = 0
        self.stabilize_window_exact_closed_count = 0
        self.stabilize_window_chain_closed_count = 0
        self.stabilize_window_expired_nochange_count = 0
        self.stabilize_window_expired_budget_count = 0
        self.stabilize_window_scoreless_verified_count = 0
        self.stabilize_window_preempted_count = 0
        self.sv_lease_admitted_count = 0
        self.sv_lease_rejected_count = 0
        self.sv_lease_reject_reason_counts: dict[str, int] = {}
        self.sv_continuation_candidate_count = 0
        self.sv_continuation_issued_count = 0
        self.sv_continuation_board_change_count = 0
        self.sv_continuation_score_delta_count = 0
        self.sv_continuation_level_delta_count = 0
        self.sv_continuation_constraint_progress_count = 0
        self.sv_aux_closer_candidate_count = 0
        self.sv_aux_closer_issued_count = 0
        self.sv_expired_competing_progress_count = 0
        self.stabilize_verify_one_shot_guard_count = 0
        self.goal_probe_starvation_guard_count = 0
        self.kind_histogram: dict[str, int] = {}
        self.source_histogram: dict[str, int] = {}
        self.last_event: dict[str, Any] = {}
        self._probe_events: list[dict[str, Any]] = []
        self._last_components: list[dict[str, Any]] = []
        self._boundary_change_counts: dict[str, int] = {}
        self._boundary_change_sources: dict[str, set[str]] = {}
        self._volatile_components: set[str] = set()
        self.volatile_effect_suppressed_count = 0
        self.visual_stencil_candidate_count = 0
        self.visual_stencil_anchor_count = 0
        self.visual_stencil_attempt_count = 0
        self.visual_stencil_exhausted_count = 0
        self.visual_repair_actuator_candidate_count = 0
        self.visual_stencil_progress_count = 0
        self.visual_stencil_regress_count = 0
        self.visual_stencil_neutral_count = 0
        self.visual_stencil_zero_violation_scoreless_count = 0
        self.visual_stencil_zero_violation_after_progress_count = 0
        self.visual_stencil_near_solved_step_count = 0
        self.visual_stencil_near_solved_suppressed_count = 0
        self.visual_stencil_ledger_candidate_count = 0
        self._last_level_idx = 0
        self._last_board_rows: list[str] = []
        self._last_visual_stencil_violation_count: int | None = None
        self._min_visual_stencil_violation_count: int | None = None
        self._effect_actuators: dict[str, _EffectActuator] = {}
        self._visual_constraint_ledger: dict[str, _VisualConstraintRecord] = {}
        self._visual_constraint_candidate_index: dict[str, str] = {}
        self._visual_stencil_anchor_keys: set[str] = set()
        self._visual_stencil_candidate_keys: set[str] = set()
        self._visual_stencil_candidate_attempts: dict[str, int] = {}
        self._visual_stencil_component_index: dict[tuple[str, str], str] = {}
        self._visual_stencil_candidate_payloads: dict[str, dict[str, Any]] = {}
        self._visual_stencil_anchor_candidates: dict[str, list[str]] = {}
        self._progress_lease_hypothesis_id = ""
        self._source_scoreless_progress_counts: dict[str, int] = {}
        self._suppressed_scoreless_sources: set[str] = set()
        self._effect_ledger = EffectLedger()
        self._affordance_graph = AffordanceGraph()
        self._goal_probe_candidate_keys: set[str] = set()
        self._goal_probe_attempts: dict[str, int] = {}
        self._actuator_group_keys: set[str] = set()
        self._stabilize_windows: dict[str, _StabilizeVerifyWindow] = {}
        self._stabilize_window_archive: dict[str, _StabilizeVerifyWindow] = {}
        self._stabilize_candidate_attempts: dict[str, int] = {}
        self._stabilize_candidate_keys: set[str] = set()
        self._goal_probe_starvation_guard_steps: set[int] = set()
        self._stabilize_lease_preempted_steps: set[int] = set()
        self._stabilize_continuation_candidate_keys: set[str] = set()
        self._stabilize_one_shot_guard_steps: set[int] = set()

    def reset(self) -> None:
        self._hypotheses.clear()
        self.last_event = {}
        self._last_components = []
        self._boundary_change_counts = {}
        self._boundary_change_sources = {}
        self._volatile_components = set()
        self._last_level_idx = 0
        self._probe_events = []
        self._last_board_rows = []
        self._last_visual_stencil_violation_count = None
        self._min_visual_stencil_violation_count = None
        self._effect_actuators = {}
        self._visual_constraint_ledger = {}
        self._visual_constraint_candidate_index = {}
        self._visual_stencil_anchor_keys = set()
        self._visual_stencil_candidate_keys = set()
        self._visual_stencil_candidate_attempts = {}
        self._visual_stencil_component_index = {}
        self._visual_stencil_candidate_payloads = {}
        self._visual_stencil_anchor_candidates = {}
        self._progress_lease_hypothesis_id = ""
        self._source_scoreless_progress_counts = {}
        self._suppressed_scoreless_sources = set()
        self._effect_ledger.reset()
        self._goal_probe_candidate_keys = set()
        self._goal_probe_attempts = {}
        self._actuator_group_keys = set()
        self._stabilize_windows = {}
        self._stabilize_window_archive = {}
        self._stabilize_candidate_attempts = {}
        self._stabilize_candidate_keys = set()
        self._goal_probe_starvation_guard_steps = set()
        self._stabilize_lease_preempted_steps = set()
        self._stabilize_continuation_candidate_keys = set()
        self._stabilize_one_shot_guard_steps = set()

    def reset_level(self) -> None:
        self._hypotheses.clear()
        self.last_event = {"event": "level_reset"}
        self._last_components = []
        self._boundary_change_counts = {}
        self._boundary_change_sources = {}
        self._volatile_components = set()
        self._probe_events = []
        self._last_board_rows = []
        self._last_visual_stencil_violation_count = None
        self._min_visual_stencil_violation_count = None
        self._effect_actuators = {}
        self._visual_constraint_ledger = {}
        self._visual_constraint_candidate_index = {}
        self._visual_stencil_anchor_keys = set()
        self._visual_stencil_candidate_keys = set()
        self._visual_stencil_candidate_attempts = {}
        self._visual_stencil_component_index = {}
        self._visual_stencil_candidate_payloads = {}
        self._visual_stencil_anchor_candidates = {}
        self._progress_lease_hypothesis_id = ""
        self._source_scoreless_progress_counts = {}
        self._suppressed_scoreless_sources = set()
        self._effect_ledger.reset()
        self._goal_probe_candidate_keys = set()
        self._goal_probe_attempts = {}
        self._actuator_group_keys = set()
        self._stabilize_windows = {}
        self._stabilize_window_archive = {}
        self._stabilize_candidate_attempts = {}
        self._stabilize_candidate_keys = set()
        self._goal_probe_starvation_guard_steps = set()
        self._stabilize_lease_preempted_steps = set()
        self._stabilize_continuation_candidate_keys = set()
        self._stabilize_one_shot_guard_steps = set()

    @staticmethod
    def _component_center(entry: dict[str, Any]) -> tuple[int, int]:
        center = dict(entry.get("center") or {})
        if center:
            return int(center.get("row", 0) or 0), int(center.get("col", 0) or 0)
        bbox = dict(entry.get("bbox") or {})
        min_row = int(bbox.get("min_row", 0) or 0)
        max_row = int(bbox.get("max_row", min_row) or min_row)
        min_col = int(bbox.get("min_col", 0) or 0)
        max_col = int(bbox.get("max_col", min_col) or min_col)
        return (min_row + max_row) // 2, (min_col + max_col) // 2

    @classmethod
    def _component_anchor_key(cls, *, level_idx: int, entry: dict[str, Any]) -> str:
        row, col = cls._component_center(entry)
        return f"{int(level_idx)}||{int(row)}||{int(col)}"

    @classmethod
    def _component_signature(cls, entry: dict[str, Any]) -> tuple[Any, ...]:
        row, col = cls._component_center(entry)
        bbox = dict(entry.get("bbox") or {})
        return (
            int(entry.get("value", 0) or 0),
            int(entry.get("size", 0) or 0),
            int(row),
            int(col),
            int(bbox.get("min_row", 0) or 0),
            int(bbox.get("max_row", 0) or 0),
            int(bbox.get("min_col", 0) or 0),
            int(bbox.get("max_col", 0) or 0),
        )

    @staticmethod
    def _object_table_by_component(state: Any | None) -> dict[str, dict[str, Any]]:
        if state is None:
            return {}
        by_component: dict[str, dict[str, Any]] = {}
        for raw in list(getattr(state, "object_table", None) or []):
            if not isinstance(raw, dict):
                continue
            component_id = str(raw.get("component_id", "") or "").strip()
            if component_id:
                by_component[component_id] = dict(raw)
        return by_component

    @classmethod
    def _normalized_components(cls, state: Any | None) -> list[dict[str, Any]]:
        if state is None:
            return []
        level_idx = int(getattr(state, "level_number", 0) or 0)
        components: list[dict[str, Any]] = []
        for raw in list(getattr(state, "object_table", None) or []):
            if not isinstance(raw, dict):
                continue
            component_id = str(raw.get("component_id", "") or "").strip()
            if not component_id:
                continue
            row, col = cls._component_center(raw)
            components.append(
                {
                    "component_id": component_id,
                    "anchor_key": cls._component_anchor_key(level_idx=level_idx, entry=raw),
                    "center_row": int(row),
                    "center_col": int(col),
                    "value": int(raw.get("value", 0) or 0),
                    "size": int(raw.get("size", 0) or 0),
                    "bbox": dict(raw.get("bbox") or {}),
                }
            )
        return components

    @staticmethod
    def _normalized_board_rows(board_rows: list[str] | tuple[str, ...] | None) -> list[str]:
        rows: list[str] = []
        for raw in list(board_rows or []):
            row = str(raw if raw is not None else "").rstrip("\r\n")
            if row:
                rows.append(row)
        return rows

    @staticmethod
    def _dominant_background_char(board_rows: list[str]) -> str:
        counts: Counter[str] = Counter()
        for row in board_rows:
            counts.update(row)
        if not counts:
            return ""
        return counts.most_common(1)[0][0]

    @classmethod
    def _visual_components(cls, board_rows: list[str]) -> list[_VisualComponent]:
        if not board_rows:
            return []
        background = cls._dominant_background_char(board_rows)
        if not background:
            return []
        row_count = len(board_rows)
        visited: set[tuple[int, int]] = set()
        components: list[_VisualComponent] = []
        for row_idx, row in enumerate(board_rows):
            for col_idx, char in enumerate(row):
                if char == background or (row_idx, col_idx) in visited:
                    continue
                stack = [(row_idx, col_idx)]
                visited.add((row_idx, col_idx))
                cells: list[tuple[int, int]] = []
                counts: Counter[str] = Counter()
                while stack:
                    current_row, current_col = stack.pop()
                    current_char = board_rows[current_row][current_col]
                    cells.append((current_row, current_col))
                    counts[current_char] += 1
                    for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        next_row = current_row + delta_row
                        next_col = current_col + delta_col
                        if next_row < 0 or next_row >= row_count:
                            continue
                        if next_col < 0 or next_col >= len(board_rows[next_row]):
                            continue
                        if (next_row, next_col) in visited:
                            continue
                        if board_rows[next_row][next_col] == background:
                            continue
                        visited.add((next_row, next_col))
                        stack.append((next_row, next_col))
                if not cells:
                    continue
                min_row = min(row for row, _ in cells)
                max_row = max(row for row, _ in cells)
                min_col = min(col for _, col in cells)
                max_col = max(col for _, col in cells)
                components.append(
                    _VisualComponent(
                        component_id=f"visual_{min_row}_{min_col}_{max_row}_{max_col}",
                        center_row=(min_row + max_row) // 2,
                        center_col=(min_col + max_col) // 2,
                        min_row=min_row,
                        max_row=max_row,
                        min_col=min_col,
                        max_col=max_col,
                        size=len(cells),
                        char_counts=dict(counts),
                    )
                )
        return components

    @staticmethod
    def _visual_component_is_compact_solid(component: _VisualComponent) -> bool:
        if component.unique_count != 1:
            return False
        if component.size < 4:
            return False
        if component.height > 20 or component.width > 20:
            return False
        area = max(1, int(component.height) * int(component.width))
        return (float(component.size) / float(area)) >= 0.65

    @staticmethod
    def _visual_component_is_stencil_anchor(component: _VisualComponent) -> bool:
        if component.unique_count < 2:
            return False
        if component.size < 9:
            return False
        if component.height < 3 or component.width < 3:
            return False
        if component.height > 12 or component.width > 12:
            return False
        return True

    @staticmethod
    def _sample_visual_anchor_cell(
        *,
        board_rows: list[str],
        anchor: _VisualComponent,
        offset_row: int,
        offset_col: int,
    ) -> str:
        row_index = max(0, min(2, int(offset_row) + 1))
        col_index = max(0, min(2, int(offset_col) + 1))
        row_start = anchor.min_row + (row_index * anchor.height) // 3
        row_end = anchor.min_row + ((row_index + 1) * anchor.height) // 3
        col_start = anchor.min_col + (col_index * anchor.width) // 3
        col_end = anchor.min_col + ((col_index + 1) * anchor.width) // 3
        row_end = max(row_start + 1, row_end)
        col_end = max(col_start + 1, col_end)
        counts: Counter[str] = Counter()
        for row in range(row_start, min(row_end, len(board_rows))):
            for col in range(col_start, min(col_end, len(board_rows[row]))):
                counts[board_rows[row][col]] += 1
        if not counts:
            return ""
        return counts.most_common(1)[0][0]

    @staticmethod
    def _nearest_visual_solid(
        *,
        solids: list[_VisualComponent],
        target_row: int,
        target_col: int,
        max_distance: int,
    ) -> _VisualComponent | None:
        ranked: list[tuple[int, int, str, _VisualComponent]] = []
        for solid in solids:
            distance = abs(int(solid.center_row) - int(target_row)) + abs(
                int(solid.center_col) - int(target_col)
            )
            if distance > max_distance:
                continue
            ranked.append((distance, -int(solid.size), solid.component_id, solid))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return ranked[0][3]

    @staticmethod
    def _infer_visual_lattice_spacing(
        *,
        anchor: _VisualComponent,
        solids: list[_VisualComponent],
    ) -> int:
        distances: list[int] = []
        row_tolerance = max(3, anchor.width // 2 + 1)
        col_tolerance = max(3, anchor.height // 2 + 1)
        search_radius = max(18, max(anchor.height, anchor.width) * 4)
        for solid in solids:
            row_delta = abs(int(solid.center_row) - int(anchor.center_row))
            col_delta = abs(int(solid.center_col) - int(anchor.center_col))
            if row_delta == 0 and col_delta == 0:
                continue
            if row_delta + col_delta > search_radius:
                continue
            if col_delta <= row_tolerance and row_delta >= 2:
                distances.append(row_delta)
            if row_delta <= col_tolerance and col_delta >= 2:
                distances.append(col_delta)
        if distances:
            return max(2, min(distances))
        return max(3, max(anchor.height, anchor.width) + 2)

    def _visual_anchor_lattice_neighbor_count(
        self,
        *,
        anchor: _VisualComponent,
        solids: list[_VisualComponent],
    ) -> int:
        spacing = self._infer_visual_lattice_spacing(anchor=anchor, solids=solids)
        neighbor_count = 0
        for offset_row in (-1, 0, 1):
            for offset_col in (-1, 0, 1):
                if offset_row == 0 and offset_col == 0:
                    continue
                target_row = int(anchor.center_row) + int(offset_row) * int(spacing)
                target_col = int(anchor.center_col) + int(offset_col) * int(spacing)
                solid = self._nearest_visual_solid(
                    solids=solids,
                    target_row=target_row,
                    target_col=target_col,
                    max_distance=max(4, spacing // 2),
                )
                if solid is not None:
                    neighbor_count += 1
        return int(neighbor_count)

    @staticmethod
    def _mode_visual_block_size(solids: list[_VisualComponent]) -> tuple[int, int]:
        counts: Counter[tuple[int, int]] = Counter()
        for solid in solids:
            if solid.height < 2 or solid.width < 2:
                continue
            counts[(int(solid.height), int(solid.width))] += 1
        if not counts:
            return 0, 0
        return counts.most_common(1)[0][0]

    @staticmethod
    def _bbox_overlap_ratio(left: _VisualComponent, right: _VisualComponent) -> float:
        overlap_rows = max(0, min(left.max_row, right.max_row) - max(left.min_row, right.min_row) + 1)
        overlap_cols = max(0, min(left.max_col, right.max_col) - max(left.min_col, right.min_col) + 1)
        overlap = overlap_rows * overlap_cols
        if overlap <= 0:
            return 0.0
        left_area = max(1, left.height * left.width)
        right_area = max(1, right.height * right.width)
        return float(overlap) / float(min(left_area, right_area))

    @staticmethod
    def _visual_window_component(
        *,
        board_rows: list[str],
        min_row: int,
        min_col: int,
        height: int,
        width: int,
        component_prefix: str,
    ) -> _VisualComponent | None:
        if height <= 0 or width <= 0:
            return None
        if min_row < 0 or min_col < 0:
            return None
        max_row = min_row + height - 1
        max_col = min_col + width - 1
        if max_row >= len(board_rows):
            return None
        if any(max_col >= len(board_rows[row]) for row in range(min_row, max_row + 1)):
            return None
        counts: Counter[str] = Counter()
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                counts[board_rows[row][col]] += 1
        if not counts:
            return None
        return _VisualComponent(
            component_id=f"{component_prefix}_{min_row}_{min_col}_{max_row}_{max_col}",
            center_row=(min_row + max_row) // 2,
            center_col=(min_col + max_col) // 2,
            min_row=min_row,
            max_row=max_row,
            min_col=min_col,
            max_col=max_col,
            size=height * width,
            char_counts=dict(counts),
        )

    def _window_stencil_anchors(
        self,
        *,
        board_rows: list[str],
        visual_components: list[_VisualComponent],
        solids: list[_VisualComponent],
        existing_anchors: list[_VisualComponent],
    ) -> list[_VisualComponent]:
        block_height, block_width = self._mode_visual_block_size(solids)
        if block_height < 3 or block_width < 3 or block_height > 12 or block_width > 12:
            return []
        large_mixed = [
            component
            for component in visual_components
            if component.unique_count >= 2 and (component.height > 12 or component.width > 12)
        ]
        if not large_mixed:
            return []
        background = self._dominant_background_char(board_rows)
        ranked: list[tuple[float, int, int, _VisualComponent]] = []
        row_limit = len(board_rows) - block_height + 1
        col_limit = max((len(row) for row in board_rows), default=0) - block_width + 1
        for min_row in range(max(0, row_limit)):
            for min_col in range(max(0, col_limit)):
                window = self._visual_window_component(
                    board_rows=board_rows,
                    min_row=min_row,
                    min_col=min_col,
                    height=block_height,
                    width=block_width,
                    component_prefix="visual_window",
                )
                if window is None or window.unique_count < 2:
                    continue
                if not any(self._bbox_overlap_ratio(window, component) > 0.0 for component in large_mixed):
                    continue
                center_char = self._sample_visual_anchor_cell(
                    board_rows=board_rows,
                    anchor=window,
                    offset_row=0,
                    offset_col=0,
                )
                if not center_char or center_char == background:
                    continue
                if center_char == window.dominant_char:
                    continue
                total = max(1, int(window.size))
                dominant_count = max(window.char_counts.values()) if window.char_counts else 0
                center_count = int(window.char_counts.get(center_char, 0) or 0)
                background_count = int(window.char_counts.get(background, 0) or 0)
                non_background_ratio = float(total - background_count) / float(total)
                dominant_ratio = float(dominant_count) / float(total)
                if non_background_ratio < 0.35 or dominant_ratio > 0.85:
                    continue
                if center_count > max(4, int(0.35 * float(total))):
                    continue
                if any(self._bbox_overlap_ratio(window, anchor) >= 0.60 for anchor in existing_anchors):
                    continue
                score = (
                    1.5 * float(window.unique_count)
                    + non_background_ratio
                    - dominant_ratio
                    - 0.002 * float(min_row + min_col)
                )
                ranked.append((score, min_row, min_col, window))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected: list[_VisualComponent] = []
        for _, _, _, window in ranked:
            if any(self._bbox_overlap_ratio(window, item) >= 0.45 for item in selected):
                continue
            selected.append(window)
            if len(selected) >= 12:
                break
        return selected

    def _component_for_visual(self, visual: _VisualComponent) -> dict[str, Any]:
        ranked: list[tuple[int, int, str, dict[str, Any]]] = []
        for entry in self._last_components:
            component_id = str(entry.get("component_id", "") or "").strip()
            if not component_id:
                continue
            bbox = dict(entry.get("bbox") or {})
            min_row = int(bbox.get("min_row", entry.get("center_row", 0)) or 0)
            max_row = int(bbox.get("max_row", entry.get("center_row", 0)) or 0)
            min_col = int(bbox.get("min_col", entry.get("center_col", 0)) or 0)
            max_col = int(bbox.get("max_col", entry.get("center_col", 0)) or 0)
            overlap_rows = max(0, min(max_row, visual.max_row) - max(min_row, visual.min_row) + 1)
            overlap_cols = max(0, min(max_col, visual.max_col) - max(min_col, visual.min_col) + 1)
            overlap_area = overlap_rows * overlap_cols
            center_distance = abs(int(entry.get("center_row", 0) or 0) - int(visual.center_row)) + abs(
                int(entry.get("center_col", 0) or 0) - int(visual.center_col)
            )
            if overlap_area <= 0 and center_distance > max(6, max(visual.height, visual.width)):
                continue
            ranked.append((-overlap_area, center_distance, component_id, dict(entry)))
        if ranked:
            ranked.sort(key=lambda item: (item[0], item[1], item[2]))
            match = dict(ranked[0][3])
            return {
                "component_id": str(match.get("component_id", "") or "").strip(),
                "anchor_key": str(match.get("anchor_key", "") or "").strip()
                or f"{int(self._last_level_idx)}||{int(visual.center_row)}||{int(visual.center_col)}",
                "center_row": int(visual.center_row),
                "center_col": int(visual.center_col),
            }
        return {
            "component_id": visual.component_id,
            "anchor_key": f"{int(self._last_level_idx)}||{int(visual.center_row)}||{int(visual.center_col)}",
            "center_row": int(visual.center_row),
            "center_col": int(visual.center_col),
        }

    @staticmethod
    def _visual_relation_satisfied(
        *,
        observed_char: str,
        expected_char: str,
        center_char: str,
        background_char: str = "",
    ) -> bool:
        if not observed_char or not expected_char:
            return True
        expected = str(expected_char)
        if (
            expected == center_char
            or expected in {"0", "$", ".", " "}
            or (background_char and expected == background_char)
        ):
            return observed_char == center_char
        return observed_char != center_char

    def _visual_stencil_violation_count(self, board_rows: list[str]) -> int | None:
        if int(self._last_level_idx) < 2 or not board_rows:
            return None
        background_char = self._dominant_background_char(board_rows)
        visual_components = self._visual_components(board_rows)
        if not visual_components:
            return None
        solids = [
            component
            for component in visual_components
            if self._visual_component_is_compact_solid(component)
        ]
        anchors = [
            component
            for component in visual_components
            if self._visual_component_is_stencil_anchor(component)
        ]
        anchors.extend(
            self._window_stencil_anchors(
                board_rows=board_rows,
                visual_components=visual_components,
                solids=solids,
                existing_anchors=anchors,
            )
        )
        anchors = [
            anchor
            for anchor in anchors
            if self._visual_anchor_lattice_neighbor_count(anchor=anchor, solids=solids) >= 4
        ]
        if not solids or not anchors:
            return None
        violation_count = 0
        seen: set[tuple[str, int, int]] = set()
        for anchor in anchors:
            anchor_key = f"{int(self._last_level_idx)}||{anchor.min_row}:{anchor.max_row}||{anchor.min_col}:{anchor.max_col}"
            spacing = self._infer_visual_lattice_spacing(anchor=anchor, solids=solids)
            center_char = self._sample_visual_anchor_cell(
                board_rows=board_rows,
                anchor=anchor,
                offset_row=0,
                offset_col=0,
            )
            for offset_row in (-1, 0, 1):
                for offset_col in (-1, 0, 1):
                    if offset_row == 0 and offset_col == 0:
                        continue
                    expected_char = self._sample_visual_anchor_cell(
                        board_rows=board_rows,
                        anchor=anchor,
                        offset_row=offset_row,
                        offset_col=offset_col,
                    )
                    target_row = int(anchor.center_row) + int(offset_row) * int(spacing)
                    target_col = int(anchor.center_col) + int(offset_col) * int(spacing)
                    solid = self._nearest_visual_solid(
                        solids=solids,
                        target_row=target_row,
                        target_col=target_col,
                        max_distance=max(4, spacing // 2),
                    )
                    if solid is None:
                        solid = self._visual_window_component(
                            board_rows=board_rows,
                            min_row=target_row - (anchor.height // 2),
                            min_col=target_col - (anchor.width // 2),
                            height=anchor.height,
                            width=anchor.width,
                            component_prefix="visual_target_window",
                        )
                    if solid is None:
                        continue
                    if self._visual_relation_satisfied(
                        observed_char=solid.dominant_char,
                        expected_char=expected_char,
                        center_char=center_char,
                        background_char=background_char,
                    ):
                        continue
                    key = (anchor_key, int(offset_row), int(offset_col))
                    if key in seen:
                        continue
                    seen.add(key)
                    violation_count += 1
        return int(violation_count)

    @staticmethod
    def _effect_actuator_key(
        *,
        level_idx: int,
        source_component_id: str,
        source_anchor_key: str,
        target_component_id: str,
        target_anchor_key: str,
        action_family: str,
        action_name: str,
    ) -> str:
        source_key = str(source_component_id or source_anchor_key or "").strip()
        target_key = str(target_component_id or target_anchor_key or "").strip()
        return "||".join(
            [
                str(int(level_idx)),
                source_key,
                target_key,
                str(action_family or "").strip().lower(),
                str(action_name or "").strip().upper(),
            ]
        )

    @staticmethod
    def _visual_constraint_key(
        *,
        level_idx: int,
        anchor_key: str,
        target_component_id: str,
        target_anchor_key: str,
        offset_row: int,
        offset_col: int,
        expected_char: str,
    ) -> str:
        target_key = str(target_component_id or target_anchor_key or "").strip()
        return "||".join(
            [
                str(int(level_idx)),
                str(anchor_key or "").strip(),
                target_key,
                str(int(offset_row)),
                str(int(offset_col)),
                str(expected_char or ""),
            ]
        )

    def _upsert_visual_constraint_record(
        self,
        *,
        constraint_id: str,
        level_idx: int,
        anchor_key: str,
        target_component_id: str,
        target_anchor_key: str,
        offset_row: int,
        offset_col: int,
        expected_char: str,
        observed_char: str,
        candidate_key: str,
        payload: dict[str, Any],
        current_step: int,
    ) -> None:
        record = self._visual_constraint_ledger.get(constraint_id)
        if record is None:
            record = _VisualConstraintRecord(
                constraint_id=str(constraint_id),
                level_idx=int(level_idx),
                anchor_key=str(anchor_key or ""),
                target_component_id=str(target_component_id or ""),
                target_anchor_key=str(target_anchor_key or ""),
                offset_row=int(offset_row),
                offset_col=int(offset_col),
                expected_char=str(expected_char or ""),
                observed_char=str(observed_char or ""),
                first_seen_step=int(current_step),
                last_seen_step=int(current_step),
            )
            self._visual_constraint_ledger[constraint_id] = record
        record.status = "unresolved"
        record.resolved_step = -1
        record.last_seen_step = int(current_step)
        record.observed_char = str(observed_char or "")
        record.candidate_key = str(candidate_key or "")
        record.last_payload = dict(payload)
        if candidate_key:
            self._visual_constraint_candidate_index[str(candidate_key)] = str(constraint_id)

    def _resolve_visual_constraint_record(
        self,
        *,
        constraint_id: str,
        current_step: int,
    ) -> None:
        record = self._visual_constraint_ledger.get(str(constraint_id or ""))
        if record is None:
            return
        record.last_seen_step = int(current_step)
        if record.status != "resolved":
            record.status = "resolved"
            record.resolved_step = int(current_step)

    def _note_visual_constraint_attempt(self, *, candidate_key: str) -> None:
        constraint_id = self._visual_constraint_candidate_index.get(str(candidate_key or ""))
        if not constraint_id:
            return
        record = self._visual_constraint_ledger.get(constraint_id)
        if record is not None:
            record.attempt_count += 1

    def _note_visual_constraint_outcome(
        self,
        *,
        candidate_key: str,
        visual_violation_delta: int,
        board_changed: bool,
        current_step: int,
    ) -> str:
        constraint_id = self._visual_constraint_candidate_index.get(str(candidate_key or ""))
        if not constraint_id:
            return ""
        record = self._visual_constraint_ledger.get(constraint_id)
        if record is None:
            return ""
        delta = int(visual_violation_delta)
        record.last_outcome_delta = delta
        record.last_seen_step = int(current_step)
        if delta < 0:
            record.progress_count += 1
            self.visual_stencil_progress_count += 1
            return "progress"
        if delta > 0:
            record.regress_count += 1
            self.visual_stencil_regress_count += 1
            return "regress"
        if bool(board_changed):
            record.neutral_count += 1
            self.visual_stencil_neutral_count += 1
            return "neutral"
        record.neutral_count += 1
        self.visual_stencil_neutral_count += 1
        return "no_change"

    def _record_effect_actuators(
        self,
        *,
        level_idx: int,
        source_component_id: str,
        source_anchor_key: str,
        source_row: int,
        source_col: int,
        action_family: str,
        action_name: str,
        effect_kind: str,
        deltas: list[_ComponentDelta],
        current_step: int,
        visual_violation_delta: int = 0,
    ) -> None:
        if not str(source_component_id or source_anchor_key or "").strip():
            return
        normalized_family = str(action_family or "").strip().lower()
        normalized_action = str(action_name or "").strip().upper()
        if normalized_family not in _RULE_ACTION_FAMILIES:
            return
        for delta in deltas[:4]:
            target_component_id = str(delta.component_id or "").strip()
            target_anchor_key = str(delta.anchor_key or "").strip()
            if not str(target_component_id or target_anchor_key or "").strip():
                continue
            key = self._effect_actuator_key(
                level_idx=level_idx,
                source_component_id=source_component_id,
                source_anchor_key=source_anchor_key,
                target_component_id=target_component_id,
                target_anchor_key=target_anchor_key,
                action_family=normalized_family,
                action_name=normalized_action,
            )
            actuator = self._effect_actuators.get(key)
            if actuator is None:
                actuator = _EffectActuator(
                    level_idx=int(level_idx),
                    source_component_id=str(source_component_id or "").strip(),
                    source_anchor_key=str(source_anchor_key or "").strip(),
                    source_center_row=int(source_row),
                    source_center_col=int(source_col),
                    target_component_id=target_component_id,
                    target_anchor_key=target_anchor_key,
                    target_center_row=int(delta.center_row),
                    target_center_col=int(delta.center_col),
                    action_family=normalized_family,
                    action_name=normalized_action,
                )
                self._effect_actuators[key] = actuator
            actuator.support_count += 1
            actuator.last_step = int(current_step)
            actuator.last_effect_kind = str(effect_kind or "")
            actuator.source_center_row = int(source_row)
            actuator.source_center_col = int(source_col)
            actuator.target_center_row = int(delta.center_row)
            actuator.target_center_col = int(delta.center_col)
            actuator.target_anchor_key = target_anchor_key or actuator.target_anchor_key
            if int(visual_violation_delta) < 0:
                actuator.visual_repair_progress_count += abs(int(visual_violation_delta))
            elif int(visual_violation_delta) > 0:
                actuator.visual_repair_regress_count += int(visual_violation_delta)
            elif int(visual_violation_delta) == 0:
                actuator.visual_repair_neutral_count += 1
            actuator.score = round(
                max(
                    0.1,
                    min(
                        6.0,
                        3.0
                        + 0.45 * float(actuator.support_count)
                        + 0.35 * float(actuator.visual_repair_progress_count)
                        - 0.45 * float(actuator.visual_repair_regress_count)
                        - 0.08 * float(actuator.visual_repair_neutral_count)
                        + (0.35 if target_component_id != str(source_component_id or "").strip() else 0.0),
                    ),
                ),
                4,
            )

    def _best_effect_actuator_for_target(
        self,
        *,
        target_component_id: str,
        target_anchor_key: str,
        target_row: int,
        target_col: int,
        current_step: int,
    ) -> _EffectActuator | None:
        target_component = str(target_component_id or "").strip()
        target_anchor = str(target_anchor_key or "").strip()
        ranked: list[tuple[float, int, int, str, _EffectActuator]] = []
        for key, actuator in self._effect_actuators.items():
            if int(actuator.level_idx) != int(self._last_level_idx):
                continue
            source_component = str(actuator.source_component_id or "").strip()
            source_anchor = str(actuator.source_anchor_key or "").strip()
            if not str(source_component or source_anchor).strip():
                continue
            if source_component in self._volatile_components:
                continue
            component_match = bool(
                target_component and target_component == str(actuator.target_component_id or "")
            )
            anchor_match = bool(
                target_anchor and target_anchor == str(actuator.target_anchor_key or "")
            )
            distance = abs(int(actuator.target_center_row) - int(target_row)) + abs(
                int(actuator.target_center_col) - int(target_col)
            )
            if not component_match and not anchor_match and distance > 8:
                continue
            recency_penalty = max(0, int(current_step) - int(actuator.last_step)) * 0.01
            score = float(actuator.score) - recency_penalty
            score -= min(1.6, 0.55 * float(actuator.visual_repair_regress_count))
            score -= min(0.8, 0.06 * float(actuator.visual_repair_neutral_count))
            if component_match:
                score += 2.5
            if anchor_match:
                score += 1.25
            score += max(0.0, 1.0 - (float(distance) / 9.0))
            ranked.append(
                (
                    score,
                    int(actuator.support_count),
                    int(distance),
                    str(key),
                    actuator,
                )
            )
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        return ranked[0][4]

    def _note_visual_stencil_selected(
        self,
        *,
        component_id: str,
        anchor_key: str,
        current_step: int,
    ) -> str:
        component = str(component_id or "").strip()
        anchor = str(anchor_key or "").strip()
        candidate_key = self._visual_stencil_component_index.get((component, anchor))
        if not candidate_key and component:
            candidate_key = self._visual_stencil_component_index.get((component, ""))
        if not candidate_key and anchor:
            candidate_key = self._visual_stencil_component_index.get(("", anchor))
        if not candidate_key:
            return ""
        previous_count = int(self._visual_stencil_candidate_attempts.get(candidate_key, 0) or 0)
        next_count = previous_count + 1
        self._visual_stencil_candidate_attempts[candidate_key] = next_count
        self._note_visual_constraint_attempt(candidate_key=candidate_key)
        self.visual_stencil_attempt_count += 1
        if next_count == max(1, int(self.config.visual_stencil_attempt_budget)):
            self.visual_stencil_exhausted_count += 1
        self.last_event = {
            "event": "visual_stencil_attempt",
            "hypothesis_id": candidate_key,
            "count": int(next_count),
            "step_idx": int(current_step),
        }
        self._append_probe_event(dict(self.last_event))
        return candidate_key

    @staticmethod
    def _merge_visual_stencil_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not entries:
            return []
        grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        counts: dict[tuple[str, str, str, str], int] = {}
        for entry in entries:
            key = (
                str(entry.get("component_id", "") or ""),
                str(entry.get("anchor_key", "") or ""),
                str(entry.get("family", "") or ""),
                str(entry.get("action_name", "") or ""),
            )
            counts[key] = int(counts.get(key, 0) or 0) + 1
            existing = grouped.get(key)
            if existing is None or float(entry.get("base_score", 0.0) or 0.0) > float(
                existing.get("base_score", 0.0) or 0.0
            ):
                grouped[key] = dict(entry)
        merged: list[dict[str, Any]] = []
        for key, entry in grouped.items():
            demand_count = int(counts.get(key, 1) or 1)
            payload = dict(entry)
            payload["visual_repair_demand_count"] = demand_count
            payload["base_score"] = round(
                float(payload.get("base_score", 0.0) or 0.0)
                + min(1.0, 0.18 * float(max(0, demand_count - 1))),
                4,
            )
            merged.append(payload)
        merged.sort(
            key=lambda item: (
                -float(item.get("base_score", 0.0) or 0.0),
                -int(item.get("visual_repair_demand_count", 0) or 0),
                str(item.get("component_id", "") or ""),
                str(item.get("anchor_key", "") or ""),
            )
        )
        return merged

    def _visual_stencil_ledger_candidates(
        self,
        *,
        limit: int,
        current_step: int,
        existing_keys: set[tuple[str, str, str]],
        ignore_attempt_limit: bool = False,
        require_current_step: bool = False,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        ranked: list[tuple[float, int, str, _VisualConstraintRecord]] = []
        for constraint_id, record in self._visual_constraint_ledger.items():
            if int(record.level_idx) != int(self._last_level_idx):
                continue
            if str(record.status or "") != "unresolved":
                continue
            if require_current_step and int(record.last_seen_step) < int(current_step):
                continue
            payload = dict(record.last_payload or {})
            if not payload:
                continue
            component_id = str(payload.get("component_id", "") or "").strip()
            anchor_key = str(payload.get("anchor_key", "") or "").strip()
            action_name = str(payload.get("action_name", "") or "").strip().upper()
            if not component_id:
                continue
            candidate_key = str(payload.get("hypothesis_id", record.candidate_key) or "")
            attempt_limit = max(4, int(self.config.visual_stencil_attempt_budget)) + min(
                4,
                int(record.progress_count),
            )
            if (
                not ignore_attempt_limit
                and int(self._visual_stencil_candidate_attempts.get(candidate_key, 0) or 0) >= attempt_limit
            ):
                continue
            key = (component_id, anchor_key, action_name)
            if key in existing_keys:
                continue
            recency_penalty = max(0, int(current_step) - int(record.last_seen_step)) * 0.01
            score = (
                float(payload.get("base_score", 0.0) or 0.0)
                + 0.35
                + min(1.0, 0.20 * float(record.progress_count))
                - min(1.4, 0.45 * float(record.regress_count))
                - min(0.9, 0.10 * float(record.neutral_count))
                - min(0.8, 0.10 * float(record.attempt_count))
                - recency_penalty
            )
            if ignore_attempt_limit:
                score += 0.85
            ranked.append((score, -int(record.attempt_count), str(constraint_id), record))
        if not ranked:
            return []
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        entries: list[dict[str, Any]] = []
        for score, _, _, record in ranked:
            payload = dict(record.last_payload or {})
            component_id = str(payload.get("component_id", "") or "").strip()
            anchor_key = str(payload.get("anchor_key", "") or "").strip()
            action_name = str(payload.get("action_name", "") or "").strip().upper()
            key = (component_id, anchor_key, action_name)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            payload["stage"] = (
                "visual_stencil_ledger_repair"
                if str(payload.get("stage", "") or "") == "visual_stencil_repair_actuator"
                else "visual_stencil_ledger_adjust"
            )
            payload["visual_constraint_status"] = "unresolved"
            payload["visual_constraint_first_seen_step"] = int(record.first_seen_step)
            payload["visual_constraint_last_seen_step"] = int(record.last_seen_step)
            payload["visual_constraint_progress_count"] = int(record.progress_count)
            payload["visual_constraint_regress_count"] = int(record.regress_count)
            payload["visual_constraint_neutral_count"] = int(record.neutral_count)
            payload["visual_constraint_last_outcome_delta"] = int(record.last_outcome_delta)
            if ignore_attempt_limit:
                payload["visual_near_solved_reissue"] = True
            payload["base_score"] = round(max(0.0, float(score)), 4)
            candidate_key = str(payload.get("hypothesis_id", record.candidate_key) or "")
            if candidate_key:
                self._visual_stencil_candidate_payloads[candidate_key] = dict(payload)
                self._visual_constraint_candidate_index[candidate_key] = record.constraint_id
                self._visual_stencil_component_index[(component_id, anchor_key)] = candidate_key
                self._visual_stencil_component_index[(component_id, "")] = candidate_key
                if anchor_key:
                    self._visual_stencil_component_index[("", anchor_key)] = candidate_key
                stencil_anchor = str(payload.get("stencil_anchor_key", "") or "")
                if stencil_anchor:
                    self._visual_stencil_anchor_candidates.setdefault(stencil_anchor, []).append(candidate_key)
            self.visual_stencil_ledger_candidate_count += 1
            entries.append(payload)
            if len(entries) >= limit:
                break
        return entries

    def _visual_stencil_candidates(self, *, limit: int, current_step: int = 0) -> list[dict[str, Any]]:
        if int(self._last_level_idx) < 2 or limit <= 0:
            return []
        self._visual_stencil_component_index = {}
        self._visual_stencil_candidate_payloads = {}
        self._visual_stencil_anchor_candidates = {}
        rows = self._last_board_rows
        if not rows:
            return self._visual_stencil_ledger_candidates(
                limit=limit,
                current_step=current_step,
                existing_keys=set(),
            )
        background_char = self._dominant_background_char(rows)
        visual_components = self._visual_components(rows)
        if not visual_components:
            return self._visual_stencil_ledger_candidates(
                limit=limit,
                current_step=current_step,
                existing_keys=set(),
            )
        solids = [
            component
            for component in visual_components
            if self._visual_component_is_compact_solid(component)
        ]
        anchors = [
            component
            for component in visual_components
            if self._visual_component_is_stencil_anchor(component)
        ]
        anchors.extend(
            self._window_stencil_anchors(
                board_rows=rows,
                visual_components=visual_components,
                solids=solids,
                existing_anchors=anchors,
            )
        )
        anchors = [
            anchor
            for anchor in anchors
            if self._visual_anchor_lattice_neighbor_count(anchor=anchor, solids=solids) >= 4
        ]
        if not solids or not anchors:
            return self._visual_stencil_ledger_candidates(
                limit=limit,
                current_step=current_step,
                existing_keys=set(),
            )
        entries: list[dict[str, Any]] = []
        seen_components: set[str] = set()
        for anchor in sorted(
            anchors,
            key=lambda item: (
                item.min_row,
                item.min_col,
                item.component_id,
            ),
        ):
            anchor_key = f"{int(self._last_level_idx)}||{anchor.min_row}:{anchor.max_row}||{anchor.min_col}:{anchor.max_col}"
            if anchor_key not in self._visual_stencil_anchor_keys:
                self._visual_stencil_anchor_keys.add(anchor_key)
                self.visual_stencil_anchor_count += 1
            spacing = self._infer_visual_lattice_spacing(anchor=anchor, solids=solids)
            center_char = self._sample_visual_anchor_cell(
                board_rows=rows,
                anchor=anchor,
                offset_row=0,
                offset_col=0,
            )
            for offset_row in (-1, 0, 1):
                for offset_col in (-1, 0, 1):
                    if offset_row == 0 and offset_col == 0:
                        continue
                    expected_char = self._sample_visual_anchor_cell(
                        board_rows=rows,
                        anchor=anchor,
                        offset_row=offset_row,
                        offset_col=offset_col,
                    )
                    target_row = int(anchor.center_row) + int(offset_row) * int(spacing)
                    target_col = int(anchor.center_col) + int(offset_col) * int(spacing)
                    solid = self._nearest_visual_solid(
                        solids=solids,
                        target_row=target_row,
                        target_col=target_col,
                        max_distance=max(4, spacing // 2),
                    )
                    if solid is None:
                        solid = self._visual_window_component(
                            board_rows=rows,
                            min_row=target_row - (anchor.height // 2),
                            min_col=target_col - (anchor.width // 2),
                            height=anchor.height,
                            width=anchor.width,
                            component_prefix="visual_target_window",
                        )
                    if solid is None:
                        continue
                    observed_char = solid.dominant_char
                    target = self._component_for_visual(solid)
                    target_component_id = str(target.get("component_id", "") or "").strip()
                    target_anchor = str(target.get("anchor_key", "") or "")
                    target_center_row = int(target.get("center_row", solid.center_row) or 0)
                    target_center_col = int(target.get("center_col", solid.center_col) or 0)
                    constraint_id = self._visual_constraint_key(
                        level_idx=int(self._last_level_idx),
                        anchor_key=anchor_key,
                        target_component_id=target_component_id,
                        target_anchor_key=target_anchor,
                        offset_row=offset_row,
                        offset_col=offset_col,
                        expected_char=expected_char,
                    )
                    if self._visual_relation_satisfied(
                        observed_char=observed_char,
                        expected_char=expected_char,
                        center_char=center_char,
                        background_char=background_char,
                    ):
                        self._resolve_visual_constraint_record(
                            constraint_id=constraint_id,
                            current_step=current_step,
                        )
                        continue
                    if not target_component_id or target_component_id in seen_components:
                        continue
                    if target_component_id in self._volatile_components:
                        continue
                    previous_record = self._visual_constraint_ledger.get(constraint_id)
                    visual_constraint_score_adjustment = 0.0
                    visual_constraint_progress_count = 0
                    visual_constraint_regress_count = 0
                    visual_constraint_neutral_count = 0
                    if previous_record is not None:
                        visual_constraint_progress_count = int(previous_record.progress_count)
                        visual_constraint_regress_count = int(previous_record.regress_count)
                        visual_constraint_neutral_count = int(previous_record.neutral_count)
                        visual_constraint_score_adjustment = (
                            min(0.75, 0.16 * float(visual_constraint_progress_count))
                            - min(1.25, 0.42 * float(visual_constraint_regress_count))
                            - min(0.75, 0.10 * float(visual_constraint_neutral_count))
                        )
                    near_solved_direct_repair = self._visual_near_solved_active()
                    actuator = None
                    if not near_solved_direct_repair:
                        actuator = self._best_effect_actuator_for_target(
                            target_component_id=target_component_id,
                            target_anchor_key=target_anchor,
                            target_row=target_center_row,
                            target_col=target_center_col,
                            current_step=current_step,
                        )
                    if actuator is not None:
                        candidate_component_id = str(actuator.source_component_id or "").strip()
                        candidate_anchor = str(actuator.source_anchor_key or "").strip()
                        candidate_center_row = int(actuator.source_center_row)
                        candidate_center_col = int(actuator.source_center_col)
                        candidate_action = (
                            str(actuator.action_name or "").strip().upper() or "ACTION6"
                        )
                        candidate_family = (
                            "move" if candidate_action in _MOVE_ACTION_NAMES else "click"
                        )
                        candidate_key = (
                            f"{anchor_key}||{target_component_id}||{offset_row}||{offset_col}"
                            f"||actuator:{candidate_component_id or candidate_anchor}"
                        )
                        stage = "visual_stencil_repair_actuator"
                        base_score = (
                            5.65
                            + min(0.45, 0.08 * float(actuator.support_count))
                            + min(0.75, 0.12 * float(actuator.visual_repair_progress_count))
                            - min(0.95, 0.22 * float(actuator.visual_repair_regress_count))
                            - min(0.55, 0.04 * float(actuator.visual_repair_neutral_count))
                        )
                    else:
                        candidate_component_id = target_component_id
                        candidate_anchor = target_anchor
                        candidate_center_row = target_center_row
                        candidate_center_col = target_center_col
                        candidate_action = "ACTION6"
                        candidate_family = "click"
                        candidate_key = f"{anchor_key}||{target_component_id}||{offset_row}||{offset_col}"
                        stage = "visual_stencil_adjust"
                        base_score = 5.25
                    payload = {
                        "component_id": candidate_component_id,
                        "anchor_key": candidate_anchor,
                        "center_row": candidate_center_row,
                        "center_col": candidate_center_col,
                        "family": candidate_family,
                        "action_name": candidate_action,
                        "hypothesis_id": candidate_key,
                        "hypothesis_kind": "visual_stencil_constraint",
                        "hypothesis_status": "verified",
                        "stage": stage,
                        "expected_char": expected_char,
                        "observed_char": observed_char,
                        "stencil_anchor_key": anchor_key,
                        "repair_target_component_id": target_component_id,
                        "repair_target_anchor_key": target_anchor,
                        "repair_target_center_row": target_center_row,
                        "repair_target_center_col": target_center_col,
                        "base_score": (
                            base_score
                            + visual_constraint_score_adjustment
                            + (0.18 if expected_char != center_char else 0.0)
                        ),
                        "visual_constraint_progress_count": visual_constraint_progress_count,
                        "visual_constraint_regress_count": visual_constraint_regress_count,
                        "visual_constraint_neutral_count": visual_constraint_neutral_count,
                    }
                    if near_solved_direct_repair:
                        payload["visual_near_solved_direct_repair"] = True
                    if actuator is not None:
                        payload["repair_actuator_support_count"] = int(actuator.support_count)
                        payload["repair_actuator_last_step"] = int(actuator.last_step)
                        payload["repair_actuator_visual_progress_count"] = int(
                            actuator.visual_repair_progress_count
                        )
                        payload["repair_actuator_visual_regress_count"] = int(
                            actuator.visual_repair_regress_count
                        )
                        payload["repair_actuator_visual_neutral_count"] = int(
                            actuator.visual_repair_neutral_count
                        )
                    self._upsert_visual_constraint_record(
                        constraint_id=constraint_id,
                        level_idx=int(self._last_level_idx),
                        anchor_key=anchor_key,
                        target_component_id=target_component_id,
                        target_anchor_key=target_anchor,
                        offset_row=offset_row,
                        offset_col=offset_col,
                        expected_char=expected_char,
                        observed_char=observed_char,
                        candidate_key=candidate_key,
                        payload=payload,
                        current_step=current_step,
                    )
                    if (
                        int(self._visual_stencil_candidate_attempts.get(candidate_key, 0) or 0)
                        >= max(1, int(self.config.visual_stencil_attempt_budget))
                    ):
                        continue
                    seen_components.add(target_component_id)
                    if candidate_key not in self._visual_stencil_candidate_keys:
                        self._visual_stencil_candidate_keys.add(candidate_key)
                        self.visual_stencil_candidate_count += 1
                        if actuator is not None:
                            self.visual_repair_actuator_candidate_count += 1
                    self._visual_stencil_component_index[(candidate_component_id, candidate_anchor)] = candidate_key
                    self._visual_stencil_component_index[(candidate_component_id, "")] = candidate_key
                    self._visual_stencil_component_index[(target_component_id, target_anchor)] = candidate_key
                    self._visual_stencil_component_index[(target_component_id, "")] = candidate_key
                    if target_anchor:
                        self._visual_stencil_component_index[("", target_anchor)] = candidate_key
                    if candidate_anchor:
                        self._visual_stencil_component_index[("", candidate_anchor)] = candidate_key
                    self._visual_stencil_candidate_payloads[candidate_key] = dict(payload)
                    self._visual_stencil_anchor_candidates.setdefault(anchor_key, []).append(candidate_key)
                    entries.append(payload)
        merged = self._merge_visual_stencil_entries(entries)
        existing_keys = {
            (
                str(item.get("component_id", "") or ""),
                str(item.get("anchor_key", "") or ""),
                str(item.get("action_name", "") or ""),
            )
            for item in merged
        }
        if len(merged) < limit:
            merged.extend(
                self._visual_stencil_ledger_candidates(
                    limit=limit - len(merged),
                    current_step=current_step,
                    existing_keys=existing_keys,
                )
            )
        return merged[:limit]

    def _visual_near_solved_active(self) -> bool:
        if int(self._last_level_idx) < 2:
            return False
        if self._min_visual_stencil_violation_count is None:
            return False
        if self._last_visual_stencil_violation_count is None:
            return False
        if int(self._min_visual_stencil_violation_count) > 2:
            return False
        if int(self._last_visual_stencil_violation_count) > int(self._min_visual_stencil_violation_count) + 3:
            return False
        return any(item.status == "unresolved" for item in self._visual_constraint_ledger.values())

    def _visual_stencil_followup_packet(
        self,
        *,
        selected_component_id: str,
        selected_anchor_key: str,
        current_step: int,
    ) -> list[dict[str, Any]]:
        selected_component = str(selected_component_id or "").strip()
        selected_anchor = str(selected_anchor_key or "").strip()
        candidate_key = self._visual_stencil_component_index.get((selected_component, selected_anchor))
        if not candidate_key and selected_component:
            candidate_key = self._visual_stencil_component_index.get((selected_component, ""))
        if not candidate_key and selected_anchor:
            candidate_key = self._visual_stencil_component_index.get(("", selected_anchor))
        if not candidate_key:
            return []
        selected_payload = dict(self._visual_stencil_candidate_payloads.get(candidate_key) or {})
        if not selected_payload:
            return []
        anchor_key = str(selected_payload.get("stencil_anchor_key", "") or "")
        anchor_candidates = list(self._visual_stencil_anchor_candidates.get(anchor_key, []) or [])
        ranked_payloads: list[dict[str, Any]] = []
        for key in [candidate_key] + [item for item in anchor_candidates if item != candidate_key]:
            if (
                int(self._visual_stencil_candidate_attempts.get(key, 0) or 0)
                >= max(1, int(self.config.visual_stencil_attempt_budget))
            ):
                continue
            payload = dict(self._visual_stencil_candidate_payloads.get(key) or {})
            if not payload:
                continue
            if key != candidate_key:
                payload["stage"] = "visual_stencil_peer_adjust"
            ranked_payloads.append(payload)
            if len(ranked_payloads) >= 2:
                break
        if len(ranked_payloads) < 2:
            return []
        self.followup_issued_count += 1
        self.last_event = {
            "event": "visual_stencil_packet_issued",
            "hypothesis_id": candidate_key,
            "step_idx": int(current_step),
        }
        self._append_probe_event(dict(self.last_event))
        return ranked_payloads[:2]


    @staticmethod
    def _selected_value(selected_summary: Any, *names: str) -> Any:
        if isinstance(selected_summary, dict):
            for name in names:
                value = selected_summary.get(name)
                if value not in (None, ""):
                    return value
            return None
        for name in names:
            value = getattr(selected_summary, name, None)
            if value not in (None, ""):
                return value
        return None

    @classmethod
    def _selected_center(cls, selected_summary: Any) -> tuple[int, int]:
        row = cls._selected_value(selected_summary, "center_row", "row")
        col = cls._selected_value(selected_summary, "center_col", "col")
        if row in (None, "") or col in (None, ""):
            anchor_key = str(cls._selected_value(selected_summary, "anchor_key") or "").strip()
            parts = [part for part in anchor_key.split("||") if part != ""]
            if len(parts) >= 2:
                row = parts[-2]
                col = parts[-1]
        try:
            return int(row or 0), int(col or 0)
        except (TypeError, ValueError):
            return 0, 0

    @classmethod
    def _changed_components(cls, *, state: Any, previous_state: Any | None) -> list[_ComponentDelta]:
        current = cls._object_table_by_component(state)
        previous = cls._object_table_by_component(previous_state)
        deltas: list[_ComponentDelta] = []
        for component_id, current_entry in current.items():
            previous_entry = previous.get(component_id)
            if previous_entry is None:
                change_kind = "new_component"
            elif cls._component_signature(current_entry) != cls._component_signature(previous_entry):
                change_kind = "changed_component"
            else:
                continue
            row, col = cls._component_center(current_entry)
            prev_row, prev_col = cls._component_center(previous_entry or current_entry)
            delta = _ComponentDelta(
                component_id=component_id,
                anchor_key=cls._component_anchor_key(
                    level_idx=int(getattr(state, "level_number", 0) or 0),
                    entry=current_entry,
                ),
                center_row=int(row),
                center_col=int(col),
                previous_center_row=int(prev_row),
                previous_center_col=int(prev_col),
                value=int(current_entry.get("value", 0) or 0),
                previous_value=(
                    int(previous_entry.get("value", 0) or 0)
                    if previous_entry is not None
                    else None
                ),
                size=int(current_entry.get("size", 0) or 0),
                previous_size=(
                    int(previous_entry.get("size", 0) or 0)
                    if previous_entry is not None
                    else None
                ),
                change_kind=change_kind,
                bbox=dict(current_entry.get("bbox") or {}),
            )
            deltas.append(delta)
        return deltas

    @staticmethod
    def _classify_kind(
        *,
        delta: _ComponentDelta,
        source_component_id: str,
        source_row: int,
        source_col: int,
        quality: dict[str, Any],
    ) -> str:
        if bool(quality.get("component_count_delta")) or bool(quality.get("unresolved_components_delta")):
            return "unlock_or_route_candidate"
        if delta.component_id == source_component_id:
            return "local_state_change"
        moved = (
            delta.center_row != delta.previous_center_row
            or delta.center_col != delta.previous_center_col
        )
        if moved:
            return "paired_motion"
        same_place = (
            delta.center_row == delta.previous_center_row
            and delta.center_col == delta.previous_center_col
            and delta.size == (delta.previous_size if delta.previous_size is not None else delta.size)
        )
        if same_place and delta.previous_value is not None and delta.value != delta.previous_value:
            return "attribute_cycle"
        distance = abs(int(delta.center_row) - int(source_row)) + abs(int(delta.center_col) - int(source_col))
        if distance <= 1:
            return "local_state_change"
        return "remote_state_change"

    def _record_broad_effect_source(
        self,
        *,
        level_idx: int,
        source_component_id: str,
        source_anchor_key: str,
        source_row: int,
        source_col: int,
        action_family: str,
        action_name: str,
        candidate_source: str,
        effect_kind: str,
        current_step: int,
        candidate_count: int,
    ) -> None:
        if not str(source_component_id or "").strip():
            self.last_event = {
                "event": "broad_effect_skipped",
                "candidate_count": int(candidate_count),
                "reason": "missing_source",
                "step_idx": int(current_step),
            }
            return
        if self._is_zero_perimeter_point(row=source_row, col=source_col):
            self.perimeter_effect_suppressed_count += 1
            self.last_event = {
                "event": "broad_effect_skipped",
                "candidate_count": int(candidate_count),
                "reason": "perimeter_source",
                "step_idx": int(current_step),
            }
            return
        if str(candidate_source or "").strip() == "aligned_object_sweep":
            self.last_event = {
                "event": "broad_effect_skipped",
                "candidate_count": int(candidate_count),
                "reason": "aligned_sweep_continues",
                "step_idx": int(current_step),
            }
            return
        kind = "global_state_change"
        hypothesis_id = self._hypothesis_key(
            level_idx=level_idx,
            source_component_id=source_component_id,
            source_anchor_key=source_anchor_key,
            target_component_id=source_component_id,
            target_anchor_key=source_anchor_key,
            action_name=action_name,
            kind=kind,
        )
        hypothesis = self._hypotheses.get(hypothesis_id)
        if hypothesis is None:
            global_level_count = sum(
                1
                for item in self._hypotheses.values()
                if int(item.level_idx) == int(level_idx)
                and item.kind == kind
            )
            if global_level_count >= 5:
                self.last_event = {
                    "event": "broad_effect_skipped",
                    "candidate_count": int(candidate_count),
                    "reason": "global_level_total_cap",
                    "step_idx": int(current_step),
                }
                return
            active_global_count = sum(
                1
                for item in self._hypotheses.values()
                if int(item.level_idx) == int(level_idx)
                and item.kind == kind
                and item.status in {"active", "verified"}
            )
            if active_global_count >= 3:
                self.last_event = {
                    "event": "broad_effect_skipped",
                    "candidate_count": int(candidate_count),
                    "reason": "global_level_cap",
                    "step_idx": int(current_step),
                }
                return
            hypothesis = RuleHypothesis(
                hypothesis_id=hypothesis_id,
                kind=kind,
                level_idx=level_idx,
                source_component_id=source_component_id,
                source_anchor_key=source_anchor_key,
                source_center_row=int(source_row),
                source_center_col=int(source_col),
                target_component_id=source_component_id,
                target_anchor_key=source_anchor_key,
                target_center_row=int(source_row),
                target_center_col=int(source_col),
                action_family=action_family,
                action_name=action_name,
                candidate_source=candidate_source or "selected_target",
                relation_dr=0,
                relation_dc=0,
                created_step=int(current_step),
                expires_at=int(current_step) + self.config.ttl_steps,
            )
            self._hypotheses[hypothesis_id] = hypothesis
            self.created_count += 1
            self._bump(self.kind_histogram, kind)
        hypothesis.effect_count += 1
        hypothesis.last_step = int(current_step)
        hypothesis.expires_at = int(current_step) + self.config.ttl_steps
        hypothesis.last_effect_kind = str(effect_kind or "broad_effect")
        hypothesis.source_center_row = int(source_row)
        hypothesis.source_center_col = int(source_col)
        hypothesis.target_center_row = int(source_row)
        hypothesis.target_center_col = int(source_col)
        hypothesis.last_progress_signature = f"broad|{int(level_idx)}|{source_component_id}|{int(candidate_count)}"
        if hypothesis.verification_count <= 0:
            self.verified_count += 1
        hypothesis.verification_count += 1
        hypothesis.status = "verified"
        if str(hypothesis.closure_status or "") != "closed":
            hypothesis.closure_status = "verified"
        hypothesis.score = round(
            min(
                6.0,
                2.65
                + 0.25 * float(hypothesis.effect_count)
                + 0.35 * float(hypothesis.verification_count)
                + min(0.6, float(candidate_count) / 30.0),
            ),
            4,
        )
        self.last_event = {
            "event": "broad_effect_recorded",
            "hypothesis_id": hypothesis.hypothesis_id,
            "kind": kind,
            "candidate_count": int(candidate_count),
            "step_idx": int(current_step),
        }
        self._append_probe_event(dict(self.last_event))

    @staticmethod
    def _delta_payloads(deltas: list[_ComponentDelta]) -> list[dict[str, Any]]:
        return [
            {
                "component_id": str(delta.component_id or ""),
                "anchor_key": str(delta.anchor_key or ""),
                "center_row": int(delta.center_row),
                "center_col": int(delta.center_col),
                "value": int(delta.value),
                "size": int(delta.size),
                "change_kind": str(delta.change_kind or ""),
            }
            for delta in list(deltas or [])
        ]

    def _record_effect_ledger_entry(
        self,
        *,
        level_idx: int,
        current_step: int,
        source_component_id: str,
        source_anchor_key: str,
        source_row: int,
        source_col: int,
        selected_source: str,
        action_family: str,
        action_name: str,
        effect_kind: str,
        deltas: list[_ComponentDelta],
    ) -> EffectRecord | None:
        if self._is_zero_perimeter_point(row=int(source_row), col=int(source_col)):
            return None
        record = self._effect_ledger.record(
            level_idx=level_idx,
            step_idx=current_step,
            source_component_id=source_component_id,
            source_anchor_key=source_anchor_key,
            source_center_row=source_row,
            source_center_col=source_col,
            selected_source=selected_source,
            action_family=action_family,
            action_name=action_name,
            effect_kind=effect_kind,
            changed_components=self._delta_payloads(deltas),
        )
        if record is None:
            return None
        if str(selected_source or "").strip() == "aligned_object_sweep":
            group_key = f"{int(level_idx)}|{source_component_id or source_anchor_key}"
            if group_key not in self._actuator_group_keys:
                self._actuator_group_keys.add(group_key)
                self.actuator_group_detected_count += 1
        return record

    @staticmethod
    def _constraint_progress_signal(
        *,
        visual_violation_delta: int,
        quality: dict[str, Any],
    ) -> bool:
        if int(visual_violation_delta or 0) < 0:
            return True
        truthy_keys = {
            "constraint_progress",
            "visual_constraint_progress",
            "rule_closure_progress",
        }
        if any(bool(quality.get(key, False)) for key in truthy_keys):
            return True
        try:
            return int(quality.get("unresolved_components_delta", 0) or 0) < 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _quality_anchor_values(quality: dict[str, Any]) -> list[str]:
        anchors: list[str] = []
        seen: set[str] = set()
        for key in (
            "structural_focus_anchor",
            "self_location_focus",
            "previous_structural_focus_anchor",
        ):
            value = str((quality or {}).get(key, "") or "").strip()
            if not value or value == "(none)" or value in seen:
                continue
            anchors.append(value)
            seen.add(value)
        return anchors

    def _selected_matches_hypothesis(
        self,
        hypothesis: RuleHypothesis,
        *,
        component_id: str,
        anchor_key: str,
    ) -> bool:
        component_id = str(component_id or "").strip()
        anchor_key = str(anchor_key or "").strip()
        return bool(
            (component_id and component_id in {hypothesis.source_component_id, hypothesis.target_component_id})
            or (anchor_key and anchor_key in {hypothesis.source_anchor_key, hypothesis.target_anchor_key})
        )

    def _mark_selected_rule_closed(
        self,
        *,
        selected_component_id: str,
        selected_anchor_key: str,
        current_step: int,
        reason: str,
    ) -> int:
        closed = 0
        for hypothesis in self._hypotheses.values():
            if int(hypothesis.expires_at) < int(current_step):
                continue
            if str(hypothesis.status or "") not in {"active", "verified"}:
                continue
            if not self._selected_matches_hypothesis(
                hypothesis,
                component_id=selected_component_id,
                anchor_key=selected_anchor_key,
            ):
                continue
            if str(hypothesis.closure_status or "") != "closed":
                closed += 1
                self.closed_count += 1
            hypothesis.closure_status = "closed"
            hypothesis.status = "verified"
            hypothesis.verification_count = max(1, int(hypothesis.verification_count))
            hypothesis.last_step = int(current_step)
            hypothesis.expires_at = int(current_step) + self.config.ttl_steps
            hypothesis.score = round(min(7.5, max(float(hypothesis.score), 5.25) + 0.45), 4)
        if closed:
            self.last_event = {
                "event": "rule_closed",
                "reason": str(reason or "closure"),
                "closed_count": int(closed),
                "step_idx": int(current_step),
            }
            self._append_probe_event(dict(self.last_event))
        return closed

    def _ensure_closed_rule_hypothesis(
        self,
        *,
        level_idx: int,
        selected_component_id: str,
        selected_anchor_key: str,
        selected_row: int,
        selected_col: int,
        selected_source: str,
        action_family: str,
        action_name: str,
        current_step: int,
        reason: str,
    ) -> int:
        selected_component_id = str(selected_component_id or "").strip()
        selected_anchor_key = str(selected_anchor_key or "").strip()
        if not str(selected_component_id or selected_anchor_key).strip():
            return 0
        normalized_action = str(action_name or "ACTION6").strip().upper()
        normalized_family = str(action_family or "click").strip().lower()
        kind = "closed_goal_probe" if str(selected_source or "").strip() == "goal_probe" else "closed_actuator"
        hypothesis_id = self._hypothesis_key(
            level_idx=level_idx,
            source_component_id=selected_component_id,
            source_anchor_key=selected_anchor_key,
            target_component_id=selected_component_id,
            target_anchor_key=selected_anchor_key,
            action_name=normalized_action,
            kind=kind,
        )
        hypothesis = self._hypotheses.get(hypothesis_id)
        created = False
        if hypothesis is None:
            hypothesis = RuleHypothesis(
                hypothesis_id=hypothesis_id,
                kind=kind,
                level_idx=int(level_idx),
                source_component_id=selected_component_id,
                source_anchor_key=selected_anchor_key,
                source_center_row=int(selected_row),
                source_center_col=int(selected_col),
                target_component_id=selected_component_id,
                target_anchor_key=selected_anchor_key,
                target_center_row=int(selected_row),
                target_center_col=int(selected_col),
                action_family=normalized_family,
                action_name=normalized_action,
                candidate_source=str(selected_source or "closure"),
                relation_dr=0,
                relation_dc=0,
                effect_count=1,
                verification_count=1,
                status="verified",
                score=6.25,
                created_step=int(current_step),
                last_step=int(current_step),
                expires_at=int(current_step) + self.config.ttl_steps,
                last_effect_kind=str(reason or "closure"),
                closure_status="closed",
            )
            self._hypotheses[hypothesis_id] = hypothesis
            self.created_count += 1
            self._bump(self.kind_histogram, kind)
            created = True
        else:
            hypothesis.status = "verified"
            hypothesis.verification_count = max(1, int(hypothesis.verification_count))
            hypothesis.effect_count = max(1, int(hypothesis.effect_count))
            hypothesis.source_center_row = int(selected_row)
            hypothesis.source_center_col = int(selected_col)
            hypothesis.target_center_row = int(selected_row)
            hypothesis.target_center_col = int(selected_col)
            hypothesis.action_family = normalized_family
            hypothesis.action_name = normalized_action
            hypothesis.candidate_source = str(selected_source or hypothesis.candidate_source)
            hypothesis.last_step = int(current_step)
            hypothesis.expires_at = int(current_step) + self.config.ttl_steps
            hypothesis.last_effect_kind = str(reason or "closure")
            hypothesis.score = round(min(7.5, max(float(hypothesis.score), 6.25)), 4)
        if str(hypothesis.closure_status or "") != "closed":
            hypothesis.closure_status = "closed"
            self.closed_count += 1
            closed = 1
        elif created:
            self.closed_count += 1
            closed = 1
        else:
            closed = 0
        if closed:
            self.rule_closed_synthetic_count += 1
            self.last_event = {
                "event": "rule_closed_synthetic",
                "reason": str(reason or "closure"),
                "hypothesis_id": hypothesis_id,
                "source": str(selected_source or ""),
                "step_idx": int(current_step),
            }
            self._append_probe_event(dict(self.last_event))
        return int(closed)

    def _active_stabilize_windows(self, *, current_step: int, level_idx: int) -> list[_StabilizeVerifyWindow]:
        return [
            window
            for window in self._stabilize_windows.values()
            if int(window.level_idx) == int(level_idx)
            and int(window.expires_at) >= int(current_step)
            and str(window.state or "") not in {"closed", "expired"}
        ]

    def _archive_stabilize_window(self, window: _StabilizeVerifyWindow) -> None:
        if str(window.state or "") == "closed":
            return
        self._stabilize_window_archive[str(window.window_id)] = window

    def _prune_stabilize_window_archive(self, *, current_step: int) -> None:
        grace = max(0, int(self.config.stabilize_closure_grace_steps))
        self._stabilize_window_archive = {
            key: window
            for key, window in self._stabilize_window_archive.items()
            if str(window.state or "") != "closed"
            and int(current_step) <= max(int(window.last_step), int(window.expires_at)) + grace
        }

    @staticmethod
    def _append_unique_bounded(values: list[str], candidates: list[str], *, limit: int = 16) -> None:
        seen = {str(value or "").strip() for value in values if str(value or "").strip()}
        for candidate in candidates:
            value = str(candidate or "").strip()
            if not value or value in seen:
                continue
            values.append(value)
            seen.add(value)
            if len(values) >= limit:
                del values[:-limit]

    def _append_stabilize_window_observations(
        self,
        window: _StabilizeVerifyWindow,
        *,
        selected_component_id: str = "",
        selected_anchor_key: str = "",
        deltas: list[_ComponentDelta] | None = None,
    ) -> None:
        component_ids = [
            str(selected_component_id or ""),
            str(window.source_component_id or ""),
            str(window.relation_component_id or ""),
        ]
        anchor_keys = [
            str(selected_anchor_key or ""),
            str(window.source_anchor_key or ""),
            str(window.relation_anchor_key or ""),
        ]
        for delta in list(deltas or []):
            component_ids.append(str(delta.component_id or ""))
            anchor_keys.append(str(delta.anchor_key or ""))
        self._append_unique_bounded(window.observed_component_ids, component_ids)
        self._append_unique_bounded(window.observed_anchor_keys, anchor_keys)

    @staticmethod
    def _stabilize_window_component_ids(window: _StabilizeVerifyWindow) -> set[str]:
        return {
            value
            for value in [
                str(window.source_component_id or ""),
                str(window.relation_component_id or ""),
                *[str(item or "") for item in list(window.observed_component_ids or [])],
            ]
            if value
        }

    @staticmethod
    def _stabilize_window_anchor_keys(window: _StabilizeVerifyWindow) -> set[str]:
        return {
            value
            for value in [
                str(window.source_anchor_key or ""),
                str(window.relation_anchor_key or ""),
                *[str(item or "") for item in list(window.observed_anchor_keys or [])],
            ]
            if value
        }

    def _matching_stabilize_windows(
        self,
        *,
        selected_component_id: str,
        selected_anchor_key: str,
        current_step: int,
        level_idx: int,
    ) -> list[_StabilizeVerifyWindow]:
        selected_component_id = str(selected_component_id or "").strip()
        selected_anchor_key = str(selected_anchor_key or "").strip()
        matches: list[_StabilizeVerifyWindow] = []
        for window in self._active_stabilize_windows(current_step=current_step, level_idx=level_idx):
            if (
                selected_component_id
                and selected_component_id in self._stabilize_window_component_ids(window)
            ) or (
                selected_anchor_key
                and selected_anchor_key in self._stabilize_window_anchor_keys(window)
            ):
                matches.append(window)
        return sorted(matches, key=lambda item: (-int(item.last_step), str(item.window_id)))

    def _matching_recent_stabilize_windows(
        self,
        *,
        selected_component_id: str,
        selected_anchor_key: str,
        current_step: int,
        level_idx: int,
        require_issued: bool = False,
    ) -> list[_StabilizeVerifyWindow]:
        selected_component_id = str(selected_component_id or "").strip()
        selected_anchor_key = str(selected_anchor_key or "").strip()
        grace = max(0, int(self.config.stabilize_closure_grace_steps))
        matches: dict[str, _StabilizeVerifyWindow] = {}
        for window in list(self._stabilize_windows.values()) + list(self._stabilize_window_archive.values()):
            if int(window.level_idx) != int(level_idx):
                continue
            if str(window.state or "") == "closed":
                continue
            if require_issued and int(window.issued_count) <= 0:
                continue
            if int(current_step) > max(int(window.last_step), int(window.expires_at)) + grace:
                continue
            if (
                selected_component_id
                and selected_component_id in self._stabilize_window_component_ids(window)
            ) or (
                selected_anchor_key
                and selected_anchor_key in self._stabilize_window_anchor_keys(window)
            ):
                matches[str(window.window_id)] = window
        return sorted(matches.values(), key=lambda item: (-int(item.last_step), str(item.window_id)))

    def _latest_unstable_window(
        self,
        *,
        current_step: int,
        level_idx: int,
    ) -> _StabilizeVerifyWindow | None:
        candidates = [
            window
            for window in self._active_stabilize_windows(current_step=current_step, level_idx=level_idx)
            if str(window.state or "") == "unstable"
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (int(item.last_step), str(item.window_id)))

    def _settle_stabilize_window(self, window: _StabilizeVerifyWindow, *, current_step: int) -> None:
        window.state = "settled"
        window.last_step = int(current_step)
        window.expires_at = int(current_step) + max(1, int(self.config.stabilize_window_ttl_steps))
        if not window.settled_counted:
            window.settled_counted = True
            self.stabilize_verify_window_settled_count += 1
        self.last_event = {
            "event": "stabilize_window_settled",
            "window_id": window.window_id,
            "step_idx": int(current_step),
        }
        self._append_probe_event(dict(self.last_event))

    def _settle_unstable_windows(self, *, current_step: int, level_idx: int) -> None:
        for window in self._active_stabilize_windows(current_step=current_step, level_idx=level_idx):
            if str(window.state or "") != "unstable":
                continue
            self._settle_stabilize_window(window, current_step=current_step)

    def _active_stabilize_lease_window(
        self,
        *,
        current_step: int,
        level_idx: int,
    ) -> _StabilizeVerifyWindow | None:
        candidates = [
            window
            for window in self._active_stabilize_windows(current_step=current_step, level_idx=level_idx)
            if bool(window.lease_active)
            and int(window.lease_expires_at) >= int(current_step)
            and str(window.state or "") not in {"closed", "expired"}
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                int(item.lease_started_step),
                int(item.last_step),
                str(item.window_id),
            ),
        )

    def _level_has_stabilize_issue(self, *, level_idx: int) -> bool:
        return any(
            int(window.level_idx) == int(level_idx) and int(window.issued_count) > 0
            for window in list(self._stabilize_windows.values())
            + list(self._stabilize_window_archive.values())
        )

    def _stabilize_lease_admission(
        self,
        window: _StabilizeVerifyWindow,
        *,
        current_step: int,
    ) -> tuple[bool, float, str]:
        observed_count = len(
            {
                str(value or "").strip()
                for value in list(window.observed_component_ids or [])
                if str(value or "").strip()
            }
        )
        score = 0.0
        if int(window.board_change_count) >= 2:
            score += 1.0
        elif int(window.board_change_count) >= 1:
            score += 0.45
        if 0 < observed_count <= 6:
            score += 0.55
        elif observed_count > 10:
            score -= 1.0
        if str(window.relation_component_id or window.relation_anchor_key or "").strip():
            score += 0.35
        if str(window.relation_component_id or "") and str(window.relation_component_id or "") != str(
            window.source_component_id or ""
        ):
            score += 0.35
        if int(window.delayed_count) > 0:
            score += min(0.85, 0.35 + 0.18 * float(window.delayed_count))
        group_key = f"{int(window.level_idx)}|{str(window.source_component_id or window.source_anchor_key or '').strip()}"
        if group_key in self._actuator_group_keys or str(window.selected_source or "") == "aligned_object_sweep":
            score += 0.75
        if self._stabilize_window_has_constraint_neighbor(window):
            score += 0.45
        if int(window.lease_followup_issued_count) > 0:
            score -= min(0.8, 0.35 * float(window.lease_followup_issued_count))
        if int(window.lease_preempt_count) > 0:
            score -= min(0.55, 0.2 * float(window.lease_preempt_count))
        if int(current_step) - int(window.last_step) > max(1, int(self.config.stabilize_window_ttl_steps)):
            score -= 0.35
        reason = "admitted"
        # Phase 8.F: env override for per-episode cap. Default cap = 4
        # was the closure bottleneck on vc33 (73 windows opened,
        # 33 rejected purely by this gate). Per-window quality gates
        # (diffuse_changes / attempt_budget / weak_evidence) still
        # apply below, so raising this cap does not bypass quality.
        _episode_budget = _stabilize_lease_episode_budget_override()
        if _episode_budget is None:
            _episode_budget = int(self.config.stabilize_lease_episode_budget)
        if int(self.stabilize_window_lease_started_count) >= max(0, _episode_budget):
            reason = "episode_budget"
            return False, round(score, 4), reason
        if observed_count > 10:
            reason = "diffuse_changes"
        elif int(window.lease_followup_issued_count) >= max(
            1,
            int(self.config.stabilize_lease_attempt_budget),
        ):
            reason = "attempt_budget"
        elif score < float(self.config.stabilize_lease_admit_min_score):
            reason = "weak_evidence"
        return reason == "admitted", round(score, 4), reason

    def _note_stabilize_lease_admission(
        self,
        window: _StabilizeVerifyWindow,
        *,
        current_step: int,
    ) -> bool:
        admitted, score, reason = self._stabilize_lease_admission(
            window,
            current_step=current_step,
        )
        window.lease_admit_score = float(score)
        if admitted:
            if not bool(window.lease_admitted):
                self.sv_lease_admitted_count += 1
            window.lease_admitted = True
            window.lease_rejected = False
            window.lease_reject_reason = ""
            return True
        window.lease_reject_reason = str(reason or "rejected")
        if not bool(window.lease_rejected):
            self.sv_lease_rejected_count += 1
            self._bump(self.sv_lease_reject_reason_counts, window.lease_reject_reason)
        window.lease_rejected = True
        self.last_event = {
            "event": "stabilize_window_lease_rejected",
            "window_id": window.window_id,
            "reason": window.lease_reject_reason,
            "score": float(score),
            "step_idx": int(current_step),
        }
        self._append_probe_event(dict(self.last_event))
        return False

    def _stabilize_window_has_constraint_neighbor(self, window: _StabilizeVerifyWindow) -> bool:
        anchors = self._stabilize_window_anchor_keys(window)
        components = self._stabilize_window_component_ids(window)
        centers = [
            self._anchor_center(anchor)
            for anchor in anchors
            if str(anchor or "").strip()
        ]
        centers.extend(
            [
                (int(window.source_center_row), int(window.source_center_col)),
                (int(window.relation_center_row), int(window.relation_center_col)),
            ]
        )
        for record in self._visual_constraint_ledger.values():
            if int(record.level_idx) != int(window.level_idx):
                continue
            if str(record.status or "") == "resolved":
                continue
            if str(record.target_component_id or "") in components:
                return True
            if str(record.target_anchor_key or "") in anchors or str(record.anchor_key or "") in anchors:
                return True
            record_row, record_col = self._anchor_center(
                str(record.target_anchor_key or record.anchor_key or "")
            )
            if not (record_row or record_col):
                continue
            if any(abs(record_row - row) + abs(record_col - col) <= 5 for row, col in centers):
                return True
        return False

    def _start_stabilize_lease(self, window: _StabilizeVerifyWindow, *, current_step: int) -> None:
        for other in self._stabilize_windows.values():
            if str(other.window_id or "") != str(window.window_id or "") and bool(other.lease_active):
                other.lease_active = False
        if not bool(window.lease_active):
            self.stabilize_window_lease_started_count += 1
        window.lease_active = True
        window.lease_started_step = int(current_step)
        window.lease_expires_at = int(current_step) + max(1, int(self.config.stabilize_lease_steps_max))
        window.lease_nochange_count = 0
        window.state = "verifying"
        window.expires_at = max(int(window.expires_at), int(window.lease_expires_at))
        if not bool(window.scoreless_verified_counted):
            window.scoreless_verified_counted = True
            self.stabilize_window_scoreless_verified_count += 1
        self.last_event = {
            "event": "stabilize_window_lease_started",
            "window_id": window.window_id,
            "step_idx": int(current_step),
        }
        self._append_probe_event(dict(self.last_event))

    def _expire_stabilize_window(
        self,
        window: _StabilizeVerifyWindow,
        *,
        current_step: int,
        reason: str,
    ) -> None:
        if str(window.state or "") in {"closed", "expired"}:
            return
        window.state = "expired"
        window.lease_active = False
        window.last_step = int(current_step)
        self.stabilize_verify_expired_count += 1
        normalized_reason = str(reason or "").strip()
        if normalized_reason == "nochange":
            self.stabilize_window_expired_nochange_count += 1
        elif normalized_reason == "budget":
            self.stabilize_window_expired_budget_count += 1
        self._archive_stabilize_window(window)
        self.last_event = {
            "event": "stabilize_window_expired",
            "window_id": window.window_id,
            "reason": normalized_reason,
            "step_idx": int(current_step),
        }
        self._append_probe_event(dict(self.last_event))

    def _open_or_update_stabilize_window(
        self,
        *,
        level_idx: int,
        current_step: int,
        source_component_id: str,
        source_anchor_key: str,
        source_row: int,
        source_col: int,
        selected_source: str,
        action_family: str,
        action_name: str,
        deltas: list[_ComponentDelta],
        linked_effect_id: str,
        delayed_continuation: bool,
        retest: bool = False,
    ) -> None:
        trigger_sources = {
            "goal_probe",
            "rule_hypothesis",
            "aligned_object_sweep",
            "transition_frontier",
            "l1_action_family_quota",
        }
        selected_source = str(selected_source or "").strip()
        if selected_source not in trigger_sources and not delayed_continuation and not retest:
            return
        relation = deltas[0] if deltas else None
        relation_component_id = str(relation.component_id if relation else source_component_id or "")
        relation_anchor_key = str(relation.anchor_key if relation else source_anchor_key or "")
        relation_row = int(relation.center_row if relation else source_row)
        relation_col = int(relation.center_col if relation else source_col)
        if not str(source_component_id or source_anchor_key or relation_component_id or relation_anchor_key).strip():
            return
        window_key = "|".join(
            [
                str(int(level_idx)),
                str(source_component_id or source_anchor_key),
                str(relation_component_id or relation_anchor_key),
                str(action_name or "").strip().upper(),
            ]
        )
        window = self._stabilize_windows.get(window_key)
        if window is not None and str(window.state or "") == "closed":
            return
        if window is not None and retest and str(window.state or "") == "expired":
            return
        if window is None:
            window = _StabilizeVerifyWindow(
                window_id=window_key,
                level_idx=int(level_idx),
                source_component_id=str(source_component_id or ""),
                source_anchor_key=str(source_anchor_key or ""),
                source_center_row=int(source_row),
                source_center_col=int(source_col),
                relation_component_id=relation_component_id,
                relation_anchor_key=relation_anchor_key,
                relation_center_row=relation_row,
                relation_center_col=relation_col,
                action_family=str(action_family or "").strip().lower(),
                action_name=str(action_name or "").strip().upper(),
                selected_source=selected_source,
                linked_effect_id=str(linked_effect_id or ""),
                opened_step=int(current_step),
                last_step=int(current_step),
                expires_at=int(current_step) + max(1, int(self.config.stabilize_window_ttl_steps)),
            )
            self._stabilize_windows[window_key] = window
            self.stabilize_verify_window_opened_count += 1
        window.state = "unstable"
        window.closure_status = "verified" if retest else str(window.closure_status or "observed")
        window.last_step = int(current_step)
        window.expires_at = int(current_step) + max(1, int(self.config.stabilize_window_ttl_steps))
        window.board_change_count += 1
        if delayed_continuation:
            window.delayed_count += 1
        if linked_effect_id:
            window.linked_effect_id = str(linked_effect_id)
        if relation_component_id:
            window.relation_component_id = relation_component_id
            window.relation_anchor_key = relation_anchor_key
            window.relation_center_row = relation_row
            window.relation_center_col = relation_col
        self._append_stabilize_window_observations(
            window,
            selected_component_id=source_component_id,
            selected_anchor_key=source_anchor_key,
            deltas=deltas,
        )
        self.last_event = {
            "event": "stabilize_window_unstable",
            "window_id": window.window_id,
            "source": selected_source,
            "delayed": bool(delayed_continuation),
            "step_idx": int(current_step),
        }
        self._append_probe_event(dict(self.last_event))

    def _mark_stabilize_closed(
        self,
        *,
        windows: list[_StabilizeVerifyWindow],
        current_step: int,
        reason: str,
        closure_kind: str = "exact",
    ) -> None:
        for window in windows:
            if str(window.state or "") == "closed":
                continue
            window.state = "closed"
            window.closure_status = "closed"
            window.lease_active = False
            window.last_step = int(current_step)
            self.stabilize_verify_closed_count += 1
            if str(closure_kind or "").strip() == "chain":
                self.stabilize_window_chain_closed_count += 1
            else:
                self.stabilize_window_exact_closed_count += 1
            self.last_event = {
                "event": "stabilize_window_closed",
                "window_id": window.window_id,
                "reason": str(reason or "closure"),
                "closure_kind": str(closure_kind or "exact"),
                "step_idx": int(current_step),
            }
            self._append_probe_event(dict(self.last_event))

    def _note_stabilize_verify_selection(
        self,
        *,
        selected_component_id: str,
        selected_anchor_key: str,
        selected_window_id: str,
        selected_stage: str,
        selected_continuation_kind: str = "",
        current_step: int,
        level_idx: int,
        board_changed: bool,
        closed: bool,
        score_delta: float = 0.0,
        level_advanced: bool = False,
        constraint_progress: bool = False,
    ) -> list[_StabilizeVerifyWindow]:
        selected_window_id = str(selected_window_id or "").strip()
        windows = []
        if selected_window_id:
            windows = [
                window
                for window in self._active_stabilize_windows(
                    current_step=current_step,
                    level_idx=level_idx,
                )
                if str(window.window_id or "") == selected_window_id
            ]
        if not windows:
            windows = self._matching_stabilize_windows(
                selected_component_id=selected_component_id,
                selected_anchor_key=selected_anchor_key,
                current_step=current_step,
                level_idx=level_idx,
            )
        if not windows:
            return []
        self.stabilize_verify_issued_count += 1
        for window in windows[:1]:
            window.issued_count += 1
            window.verify_attempts += 1
            window.last_step = int(current_step)
            window.expires_at = int(current_step) + max(1, int(self.config.stabilize_window_ttl_steps))
            if str(selected_stage or "").strip() == "lease_followup":
                window.lease_followup_issued_count += 1
                window.continuation_issued_count += 1
                self.stabilize_window_lease_followup_issued_count += 1
                self.sv_continuation_issued_count += 1
                if str(selected_continuation_kind or "").strip() == "aux_click_pair" or str(
                    selected_continuation_kind or ""
                ).strip().startswith("aux_closer"):
                    self.sv_aux_closer_issued_count += 1
                if bool(board_changed):
                    self.sv_continuation_board_change_count += 1
                if abs(float(score_delta or 0.0)) > 1e-9:
                    self.sv_continuation_score_delta_count += 1
                if bool(level_advanced):
                    self.sv_continuation_level_delta_count += 1
                if bool(constraint_progress):
                    self.sv_continuation_constraint_progress_count += 1
            if closed:
                continue
            if board_changed:
                window.closure_status = "verified"
                if int(window.verify_attempts) >= max(1, int(self.config.stabilize_verify_attempt_budget)):
                    self._expire_stabilize_window(
                        window,
                        current_step=current_step,
                        reason="budget",
                    )
                elif int(window.lease_followup_issued_count) >= max(
                    1,
                    int(self.config.stabilize_lease_attempt_budget),
                ):
                    self._expire_stabilize_window(
                        window,
                        current_step=current_step,
                        reason="budget",
                    )
                elif not self._note_stabilize_lease_admission(
                    window,
                    current_step=current_step,
                ):
                    window.state = "unstable"
                    window.closure_status = "verified"
                else:
                    self._start_stabilize_lease(window, current_step=current_step)
            elif bool(window.lease_active):
                window.lease_nochange_count += 1
                if int(window.lease_nochange_count) >= max(
                    1,
                    int(self.config.stabilize_lease_nochange_patience),
                ):
                    self._expire_stabilize_window(
                        window,
                        current_step=current_step,
                        reason="nochange",
                    )
            elif int(window.verify_attempts) >= max(1, int(self.config.stabilize_verify_attempt_budget)):
                self._expire_stabilize_window(
                    window,
                    current_step=current_step,
                    reason="budget",
                )
        return windows

    @staticmethod
    def _anchor_center(anchor_key: str) -> tuple[int, int]:
        parts = [part for part in str(anchor_key or "").split("||") if part != ""]
        if len(parts) < 3:
            return 0, 0
        try:
            return int(parts[-2]), int(parts[-1])
        except (TypeError, ValueError):
            return 0, 0

    def _component_payload_for(
        self,
        *,
        component_id: str,
        anchor_key: str,
        fallback_row: int = 0,
        fallback_col: int = 0,
    ) -> tuple[str, str, int, int] | None:
        component_id = str(component_id or "").strip()
        anchor_key = str(anchor_key or "").strip()
        for item in list(self._last_components or []):
            item_component_id = str(item.get("component_id", "") or "").strip()
            item_anchor_key = str(item.get("anchor_key", "") or "").strip()
            if (component_id and item_component_id == component_id) or (
                anchor_key and item_anchor_key == anchor_key
            ):
                return (
                    item_component_id or component_id,
                    item_anchor_key or anchor_key,
                    int(item.get("center_row", 0) or 0),
                    int(item.get("center_col", 0) or 0),
                )
        if not (fallback_row or fallback_col) and anchor_key:
            fallback_row, fallback_col = self._anchor_center(anchor_key)
        if not str(component_id or anchor_key).strip():
            return None
        return component_id, anchor_key, int(fallback_row), int(fallback_col)

    def _note_stabilize_continuation_candidate(
        self,
        *,
        window: _StabilizeVerifyWindow,
        key: str,
        continuation_kind: str = "",
    ) -> None:
        normalized = str(key or "").strip()
        if not normalized:
            return
        if normalized not in self._stabilize_continuation_candidate_keys:
            self._stabilize_continuation_candidate_keys.add(normalized)
            self.sv_continuation_candidate_count += 1
            if str(continuation_kind or "").strip() == "aux_click_pair" or str(
                continuation_kind or ""
            ).strip().startswith("aux_closer"):
                self.sv_aux_closer_candidate_count += 1
        self._append_unique_bounded(window.continuation_candidate_keys, [normalized], limit=24)

    @staticmethod
    def _stabilize_window_is_auxiliary(window: _StabilizeVerifyWindow) -> bool:
        return bool(
            str(window.action_family or "").strip().lower() == "auxiliary"
            or str(window.source_component_id or "").strip().lower().startswith("auxiliary::")
            or str(window.source_anchor_key or "").strip().lower().startswith("auxiliary::")
        )

    def _auxiliary_closer_move_actions(self, *, row: int, col: int) -> list[str]:
        row_count = len(self._last_board_rows)
        col_count = max((len(row_text) for row_text in self._last_board_rows), default=0)
        center_row = (row_count - 1) / 2.0 if row_count > 0 else 8.0
        center_col = (col_count - 1) / 2.0 if col_count > 0 else 8.0
        vertical = "ACTION1" if float(row) >= center_row else "ACTION2"
        horizontal = "ACTION3" if float(col) >= center_col else "ACTION4"
        actions: list[str] = []
        for action in (vertical, horizontal):
            if action not in actions:
                actions.append(action)
            if len(actions) >= max(1, int(self.config.stabilize_aux_closer_move_budget)):
                break
        return actions

    def _stabilize_relay_neighbor_specs(
        self,
        window: _StabilizeVerifyWindow,
        *,
        current_step: int,
        limit: int,
    ) -> list[tuple[str, str, int, int, str, str, str, float]]:
        if limit <= 0:
            return []
        anchor_points: list[tuple[str, str, int, int]] = []
        seen_anchor: set[tuple[str, str]] = set()

        def add_anchor(component_id: str, anchor_key: str, row: int = 0, col: int = 0) -> None:
            payload = self._component_payload_for(
                component_id=component_id,
                anchor_key=anchor_key,
                fallback_row=row,
                fallback_col=col,
            )
            if payload is None:
                return
            resolved_component_id, resolved_anchor_key, resolved_row, resolved_col = payload
            synthetic_auxiliary_marker = bool(
                str(resolved_component_id or "").strip().lower().startswith(("auxiliary::", "anchor::auxiliary::"))
                or str(resolved_anchor_key or "").strip().lower().startswith("auxiliary::")
            )
            if synthetic_auxiliary_marker:
                return
            key = (resolved_component_id, resolved_anchor_key)
            if key in seen_anchor:
                return
            seen_anchor.add(key)
            anchor_points.append((resolved_component_id, resolved_anchor_key, resolved_row, resolved_col))

        for component_id in reversed(list(window.observed_component_ids or [])):
            add_anchor(component_id, "")
        for anchor_key in reversed(list(window.observed_anchor_keys or [])):
            add_anchor("", anchor_key)
        add_anchor(
            window.relation_component_id,
            window.relation_anchor_key,
            window.relation_center_row,
            window.relation_center_col,
        )
        specs: list[tuple[float, str, str, int, int, str, str, str]] = []
        seen_target: set[tuple[str, str, str, str]] = set()

        def add_spec(
            *,
            component_id: str,
            anchor_key: str,
            row: int,
            col: int,
            family: str,
            action_name: str,
            continuation_kind: str,
            score: float,
        ) -> None:
            if not str(component_id or anchor_key).strip():
                return
            if self._is_zero_perimeter_point(row=int(row), col=int(col)):
                return
            key = (
                str(component_id or ""),
                str(anchor_key or ""),
                str(family or "click"),
                str(action_name or "ACTION6"),
            )
            if key in seen_target:
                return
            seen_target.add(key)
            specs.append(
                (
                    float(score),
                    str(component_id or ""),
                    str(anchor_key or ""),
                    int(row),
                    int(col),
                    str(family or "click").strip().lower(),
                    str(action_name or "ACTION6").strip().upper(),
                    str(continuation_kind or "relay_anchor"),
                )
            )

        for component_id, anchor_key, row, col in anchor_points:
            if self._stabilize_window_is_auxiliary(window):
                for index, move_action in enumerate(
                    self._auxiliary_closer_move_actions(row=int(row), col=int(col))
                ):
                    add_spec(
                        component_id=component_id,
                        anchor_key=anchor_key,
                        row=row,
                        col=col,
                        family="move",
                        action_name=move_action,
                        continuation_kind=f"aux_closer_move_{index + 1}",
                        score=5.96 - 0.05 * float(index),
                    )
            actuator = self._best_effect_actuator_for_target(
                target_component_id=component_id,
                target_anchor_key=anchor_key,
                target_row=row,
                target_col=col,
                current_step=current_step,
            )
            if actuator is not None:
                source_key = str(actuator.source_component_id or actuator.source_anchor_key or "")
                window_source_key = str(window.source_component_id or window.source_anchor_key or "")
                if source_key and source_key != window_source_key:
                    add_spec(
                        component_id=actuator.source_component_id,
                        anchor_key=actuator.source_anchor_key,
                        row=actuator.source_center_row,
                        col=actuator.source_center_col,
                        family=actuator.action_family,
                        action_name=actuator.action_name,
                        continuation_kind="relay_actuator",
                        score=5.82 + min(0.35, 0.05 * float(actuator.support_count)),
                    )
            auxiliary_window = self._stabilize_window_is_auxiliary(window)
            add_spec(
                component_id=component_id,
                anchor_key=anchor_key,
                row=row,
                col=col,
                family="click",
                action_name="ACTION6",
                continuation_kind="aux_click_pair" if auxiliary_window else "relay_anchor",
                score=6.12 if auxiliary_window else 5.52,
            )
            anchor_snapshot = self._component_snapshot(component_id=component_id, anchor_key=anchor_key)
            anchor_value = int(anchor_snapshot.get("value", 0) or 0) if anchor_snapshot else 0
            anchor_size = int(anchor_snapshot.get("size", 0) or 0) if anchor_snapshot else 0
            ranked_neighbors: list[tuple[float, int, str, dict[str, Any]]] = []
            for entry in self._last_components:
                neighbor_component = str(entry.get("component_id", "") or "")
                neighbor_anchor = str(entry.get("anchor_key", "") or "")
                if not str(neighbor_component or neighbor_anchor).strip():
                    continue
                if neighbor_component == component_id or neighbor_anchor == anchor_key:
                    continue
                if neighbor_component == str(window.source_component_id or "") or neighbor_anchor == str(
                    window.source_anchor_key or ""
                ):
                    continue
                neighbor_row = int(entry.get("center_row", 0) or 0)
                neighbor_col = int(entry.get("center_col", 0) or 0)
                distance = abs(neighbor_row - row) + abs(neighbor_col - col)
                if distance <= 0 or distance > 8:
                    continue
                aligned = neighbor_row == row or neighbor_col == col
                near = distance <= 4
                same_value = bool(anchor_value and int(entry.get("value", 0) or 0) == anchor_value)
                same_size = bool(anchor_size and int(entry.get("size", 0) or 0) == anchor_size)
                if not (aligned or near or same_value or same_size):
                    continue
                score = (
                    5.20
                    + (0.32 if aligned else 0.0)
                    + (0.26 if near else 0.0)
                    + (0.22 if same_value else 0.0)
                    + (0.10 if same_size else 0.0)
                    - min(distance, 8) * 0.035
                )
                ranked_neighbors.append((score, distance, neighbor_component or neighbor_anchor, dict(entry)))
            ranked_neighbors.sort(key=lambda item: (-item[0], item[1], item[2]))
            for score, _distance, _key, entry in ranked_neighbors[:2]:
                add_spec(
                    component_id=str(entry.get("component_id", "") or ""),
                    anchor_key=str(entry.get("anchor_key", "") or ""),
                    row=int(entry.get("center_row", 0) or 0),
                    col=int(entry.get("center_col", 0) or 0),
                    family="click",
                    action_name="ACTION6",
                    continuation_kind="relay_neighbor",
                    score=score,
                )
        for record in self._visual_constraint_ledger.values():
            if int(record.level_idx) != int(window.level_idx) or str(record.status or "") == "resolved":
                continue
            if not any(
                str(record.target_component_id or "") == component_id
                or str(record.target_anchor_key or record.anchor_key or "") == anchor_key
                for component_id, anchor_key, _row, _col in anchor_points
            ):
                continue
            record_row, record_col = self._anchor_center(
                str(record.target_anchor_key or record.anchor_key or "")
            )
            add_spec(
                component_id=record.target_component_id,
                anchor_key=record.target_anchor_key,
                row=record_row,
                col=record_col,
                family="click",
                action_name="ACTION6",
                continuation_kind="constraint_frontier",
                score=5.74,
            )
        specs.sort(key=lambda item: (-item[0], item[5], item[1], item[2], item[6]))
        return [
            (component_id, anchor_key, row, col, family, action_name, continuation_kind, score)
            for score, component_id, anchor_key, row, col, family, action_name, continuation_kind in specs[:limit]
        ]

    def _stabilize_lease_candidates(self, *, limit: int, current_step: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        window = self._active_stabilize_lease_window(
            current_step=current_step,
            level_idx=int(self._last_level_idx),
        )
        if window is None:
            return []
        if int(window.lease_followup_issued_count) >= max(
            1,
            int(self.config.stabilize_lease_attempt_budget),
        ):
            return []
        if int(window.verify_attempts) >= max(1, int(self.config.stabilize_verify_attempt_budget)):
            return []
        entries: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()

        def add_candidate(
            *,
            component_id: str,
            anchor_key: str,
            row: int = 0,
            col: int = 0,
            base_score: float = 5.38,
            family: str = "",
            action_name: str = "",
            continuation_kind: str = "relay_anchor",
        ) -> None:
            if len(entries) >= limit:
                return
            payload = self._component_payload_for(
                component_id=component_id,
                anchor_key=anchor_key,
                fallback_row=row,
                fallback_col=col,
            )
            if payload is None:
                return
            resolved_component_id, resolved_anchor_key, resolved_row, resolved_col = payload
            synthetic_auxiliary_marker = bool(
                str(resolved_component_id or "").strip().lower().startswith(("auxiliary::", "anchor::auxiliary::"))
                or str(resolved_anchor_key or "").strip().lower().startswith("auxiliary::")
            )
            if synthetic_auxiliary_marker and str(continuation_kind or "") != "source_retest":
                return
            resolved_family = str(
                (
                    family
                    or (
                        window.action_family
                        if (
                            (resolved_component_id and resolved_component_id == str(window.source_component_id or ""))
                            or (resolved_anchor_key and resolved_anchor_key == str(window.source_anchor_key or ""))
                        )
                        else "click"
                    )
                    or "click"
                )
            ).strip().lower()
            resolved_action_name = str(
                (
                    action_name
                    or (
                        window.action_name
                        if (
                            (resolved_component_id and resolved_component_id == str(window.source_component_id or ""))
                            or (resolved_anchor_key and resolved_anchor_key == str(window.source_anchor_key or ""))
                        )
                        else "ACTION6"
                    )
                    or "ACTION6"
                )
            ).strip().upper()
            key = (resolved_component_id, resolved_anchor_key, resolved_family, resolved_action_name)
            if key in seen:
                return
            if self._is_zero_perimeter_point(row=int(resolved_row), col=int(resolved_col)):
                return
            seen.add(key)
            is_source = bool(
                resolved_component_id
                and resolved_component_id == str(window.source_component_id or "")
            ) or bool(
                resolved_anchor_key
                and resolved_anchor_key == str(window.source_anchor_key or "")
            )
            entries.append(
                {
                    "component_id": resolved_component_id,
                    "anchor_key": resolved_anchor_key,
                    "center_row": int(resolved_row),
                    "center_col": int(resolved_col),
                    "family": resolved_family,
                    "action_name": resolved_action_name,
                    "source": "stabilize_verify",
                    "stage": "lease_followup",
                    "continuation_kind": str(continuation_kind or "relay_anchor"),
                    "closure_status": str(window.closure_status or "verified"),
                    "linked_effect_id": str(window.linked_effect_id or ""),
                    "window_id": str(window.window_id or ""),
                    "lease_active": True,
                    "base_score": round(
                        float(base_score)
                        + min(0.35, 0.05 * float(window.board_change_count))
                        + min(0.25, 0.04 * float(window.delayed_count))
                        - 0.16 * float(window.lease_followup_issued_count),
                        4,
                    ),
                }
            )
            self._note_stabilize_continuation_candidate(
                window=window,
                key="|".join(
                    [
                        str(window.window_id or ""),
                        str(continuation_kind or "relay_anchor"),
                        resolved_component_id,
                        resolved_anchor_key,
                        resolved_action_name,
                    ]
                ),
                continuation_kind=str(continuation_kind or "relay_anchor"),
            )

        for (
            component_id,
            anchor_key,
            row,
            col,
            family,
            action_name,
            continuation_kind,
            base_score,
        ) in self._stabilize_relay_neighbor_specs(
            window,
            current_step=current_step,
            limit=max(1, limit),
        ):
            add_candidate(
                component_id=component_id,
                anchor_key=anchor_key,
                row=row,
                col=col,
                family=family,
                action_name=action_name,
                continuation_kind=continuation_kind,
                base_score=base_score,
            )
        for component_id in reversed(list(window.observed_component_ids or [])):
            add_candidate(
                component_id=component_id,
                anchor_key="",
                base_score=5.42,
                continuation_kind="relay_anchor",
            )
        for anchor_key in reversed(list(window.observed_anchor_keys or [])):
            add_candidate(
                component_id="",
                anchor_key=anchor_key,
                base_score=5.34,
                continuation_kind="relay_anchor",
            )
        add_candidate(
            component_id=window.relation_component_id,
            anchor_key=window.relation_anchor_key,
            row=window.relation_center_row,
            col=window.relation_center_col,
            base_score=5.28,
            continuation_kind="relation_anchor",
        )
        add_candidate(
            component_id=window.source_component_id,
            anchor_key=window.source_anchor_key,
            row=window.source_center_row,
            col=window.source_center_col,
            base_score=5.18,
            continuation_kind="source_retest",
        )
        return entries[:limit]

    def _stabilize_verify_candidates(self, *, limit: int, current_step: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        entries: list[dict[str, Any]] = []
        windows = sorted(
            self._active_stabilize_windows(current_step=current_step, level_idx=int(self._last_level_idx)),
            key=lambda item: (
                0 if str(item.state or "") == "settled" else 1,
                int(item.verify_attempts),
                -int(item.board_change_count),
                -int(item.delayed_count),
                -int(item.last_step),
                str(item.window_id),
            ),
        )
        for window in windows:
            if len(entries) >= limit:
                break
            if str(window.state or "") not in {"settled", "verifying"}:
                continue
            if int(window.verify_attempts) >= max(1, int(self.config.stabilize_verify_attempt_budget)):
                continue
            candidate_specs = [
                (
                    window.source_component_id,
                    window.source_anchor_key,
                    window.source_center_row,
                    window.source_center_col,
                    "settled_retest",
                    window.action_family,
                    window.action_name,
                    5.05,
                ),
                (
                    window.relation_component_id,
                    window.relation_anchor_key,
                    window.relation_center_row,
                    window.relation_center_col,
                    "relation_retest",
                    "click",
                    "ACTION6",
                    4.72,
                ),
            ]
            for component_id, anchor_key, row, col, stage, family, action_name, base_score in candidate_specs:
                if len(entries) >= limit:
                    break
                if not str(component_id or anchor_key).strip():
                    continue
                if stage == "relation_retest" and component_id == window.source_component_id and anchor_key == window.source_anchor_key:
                    continue
                if stage == "relation_retest" and (
                    str(component_id or "").strip().lower().startswith(("auxiliary::", "anchor::auxiliary::"))
                    or str(anchor_key or "").strip().lower().startswith("auxiliary::")
                ):
                    continue
                if self._is_zero_perimeter_point(row=int(row), col=int(col)):
                    continue
                key = f"{window.window_id}|{stage}|{component_id}|{anchor_key}"
                one_shot_boost = 0.0
                if (
                    int(window.issued_count) <= 0
                    and str(window.state or "") == "settled"
                    and int(current_step) not in self._stabilize_one_shot_guard_steps
                ):
                    one_shot_boost = 1.05
                    self._stabilize_one_shot_guard_steps.add(int(current_step))
                    self.stabilize_verify_one_shot_guard_count += 1
                payload = {
                    "component_id": str(component_id or ""),
                    "anchor_key": str(anchor_key or ""),
                    "center_row": int(row),
                    "center_col": int(col),
                    "family": str(family or "click").strip().lower(),
                    "action_name": str(action_name or "ACTION6").strip().upper(),
                    "source": "stabilize_verify",
                    "stage": stage,
                    "closure_status": str(window.closure_status or "verified"),
                    "linked_effect_id": str(window.linked_effect_id or ""),
                    "window_id": window.window_id,
                    "base_score": round(
                        float(base_score)
                        + min(0.45, 0.06 * float(window.board_change_count))
                        + min(0.35, 0.04 * float(window.delayed_count))
                        + one_shot_boost
                        - 0.18 * float(window.verify_attempts),
                        4,
                    ),
                }
                if key not in self._stabilize_candidate_keys:
                    self._stabilize_candidate_keys.add(key)
                entries.append(payload)
        return entries[:limit]

    def _goal_probe_candidates(self, *, limit: int, current_step: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        last_level = int(self._last_level_idx)
        min_level = 2
        if last_level < 2:
            if not _enable_goal_probe_l1_quota():
                return []
            l1_records = self._effect_ledger.recent(
                level_idx=last_level,
                current_step=int(current_step),
                ttl_steps=max(8, int(self.config.ttl_steps)),
            )
            quota_engaged = any(
                str(r.selected_source or "").strip() == "l1_action_family_quota"
                for r in l1_records
            )
            if not quota_engaged:
                return []
            min_level = 1
        records = [
            record
            for record in self._effect_ledger.recent(
                level_idx=last_level,
                current_step=int(current_step),
                ttl_steps=max(8, int(self.config.ttl_steps)),
            )
            if str(record.selected_source or "").strip() != "goal_probe"
        ]
        candidates = self._affordance_graph.goal_probe_candidates(
            records=records,
            current_components=list(self._last_components),
            level_idx=last_level,
            current_step=int(current_step),
            max_candidates=min(limit, int(self.config.max_goal_probe_candidates)),
            min_level_idx=min_level,
        )
        entries: list[dict[str, Any]] = []
        for candidate in candidates:
            if str(candidate.component_id or "").strip() in self._volatile_components:
                continue
            payload = candidate.to_dict()
            attempt_key = self._goal_probe_attempt_key(
                component_id=str(payload.get("component_id", "") or ""),
                anchor_key=str(payload.get("anchor_key", "") or ""),
            )
            if int(self._goal_probe_attempts.get(attempt_key, 0) or 0) >= max(
                1,
                int(self.config.goal_probe_attempt_budget),
            ):
                continue
            key = (
                str(payload.get("component_id", "") or ""),
                str(payload.get("anchor_key", "") or ""),
                str(payload.get("linked_effect_id", "") or ""),
            )
            if key not in self._goal_probe_candidate_keys:
                self._goal_probe_candidate_keys.add(key)
                self.goal_probe_candidate_count += 1
            entries.append(payload)
        return entries

    def _goal_probe_attempt_key(self, *, component_id: str, anchor_key: str) -> str:
        return "|".join(
            [
                str(int(self._last_level_idx)),
                str(component_id or "").strip(),
                str(anchor_key or "").strip(),
            ]
        )

    @staticmethod
    def _bump(histogram: dict[str, int], key: str) -> None:
        normalized = str(key or "").strip()
        if not normalized:
            normalized = "unknown"
        histogram[normalized] = int(histogram.get(normalized, 0) or 0) + 1

    def _append_probe_event(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("event", "probe")
        self._probe_events.append(payload)
        if len(self._probe_events) > 256:
            self._probe_events = self._probe_events[-256:]

    def _record_invariant_violation(self, *, reason: str, hypothesis: RuleHypothesis) -> None:
        self.invariant_violation_count += 1
        self.last_event = {
            "event": "invariant_violation",
            "reason": str(reason or "unknown"),
            "hypothesis_id": hypothesis.hypothesis_id,
            "kind": hypothesis.kind,
        }
        self._append_probe_event(dict(self.last_event))

    def _is_boundary_delta(self, delta: _ComponentDelta) -> bool:
        if not self._last_components:
            return False
        rows = [int(item.get("center_row", 0) or 0) for item in self._last_components]
        cols = [int(item.get("center_col", 0) or 0) for item in self._last_components]
        if not rows or not cols:
            return False
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        row_span = max_row - min_row
        col_span = max_col - min_col
        row = int(delta.center_row)
        col = int(delta.center_col)
        return (
            (row_span >= 8 and row in {min_row, max_row})
            or (col_span >= 8 and col in {min_col, max_col})
        )

    def _note_boundary_volatility(
        self,
        *,
        deltas: list[_ComponentDelta],
        source_component_id: str,
        source_anchor_key: str,
    ) -> None:
        source_key = str(source_component_id or source_anchor_key or "").strip()
        for delta in deltas:
            component_id = str(delta.component_id or "").strip()
            if not component_id or component_id == source_component_id:
                continue
            if not self._is_boundary_delta(delta):
                continue
            self._boundary_change_counts[component_id] = int(
                self._boundary_change_counts.get(component_id, 0) or 0
            ) + 1
            if source_key:
                self._boundary_change_sources.setdefault(component_id, set()).add(source_key)
            source_count = len(self._boundary_change_sources.get(component_id, set()))
            if self._boundary_change_counts[component_id] >= 4 and source_count >= 2:
                self._volatile_components.add(component_id)

    def _filter_volatile_deltas(self, deltas: list[_ComponentDelta]) -> list[_ComponentDelta]:
        if not deltas or not self._volatile_components:
            return list(deltas)
        kept = [
            delta
            for delta in deltas
            if str(delta.component_id or "").strip() not in self._volatile_components
        ]
        self.volatile_effect_suppressed_count += len(deltas) - len(kept)
        return kept

    def _prune(self, *, current_step: int) -> None:
        terminal_statuses = {"disconfirmed", "scoreless_churn"}
        self._hypotheses = {
            key: hypothesis
            for key, hypothesis in self._hypotheses.items()
            if int(hypothesis.expires_at) >= int(current_step)
            and str(hypothesis.status or "") not in terminal_statuses
            and not self._has_zero_perimeter_source(hypothesis)
            and str(hypothesis.source_component_id or "").strip() not in self._volatile_components
            and str(hypothesis.target_component_id or "").strip() not in self._volatile_components
        }
        if self._progress_lease_hypothesis_id and self._progress_lease_hypothesis_id not in self._hypotheses:
            self._progress_lease_hypothesis_id = ""
        kept_windows: dict[str, _StabilizeVerifyWindow] = {}
        for key, window in self._stabilize_windows.items():
            state = str(window.state or "")
            if state == "closed":
                kept_windows[key] = window
                continue
            if state == "expired":
                self._archive_stabilize_window(window)
                continue
            if bool(window.lease_active) and int(window.lease_expires_at) < int(current_step):
                self._expire_stabilize_window(
                    window,
                    current_step=current_step,
                    reason="budget",
                )
                continue
            if (
                state == "unstable"
                and int(window.expires_at) < int(current_step)
                and not self._level_has_stabilize_issue(level_idx=int(window.level_idx))
            ):
                self._settle_stabilize_window(window, current_step=current_step)
                kept_windows[key] = window
                continue
            if int(window.expires_at) < int(current_step):
                self._expire_stabilize_window(
                    window,
                    current_step=current_step,
                    reason="budget",
                )
                continue
            kept_windows[key] = window
        self._stabilize_windows = kept_windows
        self._prune_stabilize_window_archive(current_step=current_step)
        if len(self._hypotheses) <= self.config.max_active:
            return
        ranked = sorted(
            self._hypotheses.values(),
            key=lambda item: (
                -float(item.score),
                -int(item.verification_count),
                -int(item.last_step),
                str(item.hypothesis_id),
            ),
        )
        self._hypotheses = {item.hypothesis_id: item for item in ranked[: self.config.max_active]}
        if self._progress_lease_hypothesis_id and self._progress_lease_hypothesis_id not in self._hypotheses:
            self._progress_lease_hypothesis_id = ""

    def _hypothesis_key(
        self,
        *,
        level_idx: int,
        source_component_id: str,
        source_anchor_key: str,
        target_component_id: str,
        target_anchor_key: str,
        action_name: str,
        kind: str,
    ) -> str:
        source_key = source_component_id or source_anchor_key
        target_key = target_component_id or target_anchor_key
        return "||".join(
            [
                str(int(level_idx)),
                str(kind or "effect"),
                str(action_name or ""),
                str(source_key or ""),
                str(target_key or ""),
            ]
        )

    def _disconfirm_selected(
        self,
        *,
        selected_component_id: str,
        selected_anchor_key: str,
        current_step: int,
        force: bool = False,
    ) -> None:
        for hypothesis in self._hypotheses.values():
            if int(hypothesis.expires_at) < int(current_step):
                continue
            matches_target = (
                selected_component_id
                and selected_component_id == hypothesis.target_component_id
            ) or (
                selected_anchor_key
                and selected_anchor_key == hypothesis.target_anchor_key
            )
            matches_source = (
                selected_component_id
                and selected_component_id == hypothesis.source_component_id
            ) or (
                selected_anchor_key
                and selected_anchor_key == hypothesis.source_anchor_key
            )
            if not (matches_target or matches_source):
                continue
            hypothesis.disconfirm_count += 1
            hypothesis.score = round(max(0.0, hypothesis.score - 0.6), 4)
            hypothesis.last_step = int(current_step)
            if (
                force
                or hypothesis.disconfirm_count >= 2
                or hypothesis.followup_attempts >= self.config.verification_budget
            ):
                was_disconfirmed = hypothesis.status == "disconfirmed"
                hypothesis.status = "disconfirmed"
                if self._progress_lease_hypothesis_id == hypothesis.hypothesis_id:
                    self._progress_lease_hypothesis_id = ""
                if not was_disconfirmed:
                    self.disconfirmed_count += 1
            self.last_event = {
                "event": "disconfirm",
                "hypothesis_id": hypothesis.hypothesis_id,
                "step_idx": int(current_step),
            }
            return

    @staticmethod
    def _scoreless_churn_threshold(kind: str) -> int:
        thresholds = {
            "global_state_change": 2,
            "local_state_change": 2,
            "unlock_or_route_candidate": 3,
            "attribute_cycle": 4,
            "paired_motion": 4,
            "remote_state_change": 4,
        }
        return int(thresholds.get(str(kind or ""), 3))

    @staticmethod
    def _scoreless_progress_limit(kind: str) -> int:
        limits = {
            "global_state_change": 3,
            "local_state_change": 5,
            "attribute_cycle": 8,
            "remote_state_change": 10,
            "unlock_or_route_candidate": 12,
            "paired_motion": 14,
        }
        return int(limits.get(str(kind or ""), 8))

    @staticmethod
    def _progress_signature(delta: _ComponentDelta) -> str:
        return "|".join(
            [
                str(delta.component_id),
                str(int(delta.center_row)),
                str(int(delta.center_col)),
                str(int(delta.value)),
                str(int(delta.size)),
                str(delta.change_kind),
            ]
        )

    @staticmethod
    def _motion_actions(*, motion_dr: int, motion_dc: int) -> list[str]:
        actions: list[str] = []
        dr = int(motion_dr or 0)
        dc = int(motion_dc or 0)
        if abs(dr) >= abs(dc) and dr:
            actions.append("ACTION2" if dr > 0 else "ACTION1")
        if abs(dc) >= abs(dr) and dc:
            actions.append("ACTION4" if dc > 0 else "ACTION3")
        if not actions and dr:
            actions.append("ACTION2" if dr > 0 else "ACTION1")
        if not actions and dc:
            actions.append("ACTION4" if dc > 0 else "ACTION3")
        return actions[:2]

    @staticmethod
    def _opposite_move_action(action_name: str) -> str:
        opposites = {
            "ACTION1": "ACTION2",
            "ACTION2": "ACTION1",
            "ACTION3": "ACTION4",
            "ACTION4": "ACTION3",
        }
        return opposites.get(str(action_name or "").strip().upper(), "")

    @classmethod
    def _motion_probe_actions(cls, *, motion_dr: int, motion_dc: int) -> list[str]:
        primary = cls._motion_actions(motion_dr=motion_dr, motion_dc=motion_dc)
        ordered: list[str] = []
        for action_name in primary:
            if action_name and action_name not in ordered:
                ordered.append(action_name)
        for action_name in list(primary):
            opposite = cls._opposite_move_action(action_name)
            if opposite and opposite not in ordered:
                ordered.append(opposite)
        for action_name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if action_name not in ordered:
                ordered.append(action_name)
        return ordered

    @staticmethod
    def _source_action_name(hypothesis: RuleHypothesis) -> str:
        action_name = str(hypothesis.action_name or "").strip().upper()
        family = str(hypothesis.action_family or "").strip().lower()
        if family == "move" and action_name in _MOVE_ACTION_NAMES:
            return action_name
        if (
            family == "auxiliary"
            and action_name.startswith("ACTION")
            and action_name[6:].isdigit()
            and action_name not in _MOVE_ACTION_NAMES
            and action_name != _CLICK_ACTION_NAME
        ):
            return action_name
        return _CLICK_ACTION_NAME

    @classmethod
    def _source_family(cls, hypothesis: RuleHypothesis) -> str:
        action_name = cls._source_action_name(hypothesis)
        family = str(hypothesis.action_family or "").strip().lower()
        if family == "auxiliary" and action_name not in _MOVE_ACTION_NAMES and action_name != _CLICK_ACTION_NAME:
            return "auxiliary"
        return "move" if action_name in _MOVE_ACTION_NAMES else "click"

    @staticmethod
    def _source_key(*, component_id: str, anchor_key: str) -> str:
        return str(component_id or "").strip() or str(anchor_key or "").strip()

    @classmethod
    def _hypothesis_source_key(cls, hypothesis: RuleHypothesis) -> str:
        return cls._source_key(
            component_id=hypothesis.source_component_id,
            anchor_key=hypothesis.source_anchor_key,
        )

    @staticmethod
    def _is_zero_perimeter_point(*, row: int, col: int) -> bool:
        return int(row) <= 0 or int(col) <= 0

    @classmethod
    def _has_zero_perimeter_source(cls, hypothesis: RuleHypothesis) -> bool:
        return cls._is_zero_perimeter_point(
            row=hypothesis.source_center_row,
            col=hypothesis.source_center_col,
        )

    @classmethod
    def _is_bounded_click_unlock(cls, hypothesis: RuleHypothesis) -> bool:
        return (
            str(hypothesis.kind or "") == "unlock_or_route_candidate"
            and cls._source_action_name(hypothesis) == "ACTION6"
            and int(hypothesis.target_motion_dr or 0) == 0
            and int(hypothesis.target_motion_dc or 0) == 0
        )

    def _source_is_suppressed(self, hypothesis: RuleHypothesis) -> bool:
        source_key = self._hypothesis_source_key(hypothesis)
        return bool(source_key and source_key in self._suppressed_scoreless_sources)

    def _suppress_scoreless_source(
        self,
        *,
        hypothesis: RuleHypothesis,
        current_step: int,
        count: int,
    ) -> None:
        source_key = self._hypothesis_source_key(hypothesis)
        if not source_key:
            return
        already_suppressed = source_key in self._suppressed_scoreless_sources
        self._suppressed_scoreless_sources.add(source_key)
        if not already_suppressed:
            self.source_progress_suppressed_count += 1
        if self._progress_lease_hypothesis_id == hypothesis.hypothesis_id:
            self._progress_lease_hypothesis_id = ""
        for item in self._hypotheses.values():
            if self._hypothesis_source_key(item) == source_key and (
                self._is_bounded_click_unlock(item)
                or item.kind == "global_state_change"
            ):
                item.status = "scoreless_churn"
        self.scoreless_churn_count += 1
        self.disconfirmed_count += 1
        self.last_event = {
            "event": "scoreless_source_progress_suppressed",
            "hypothesis_id": hypothesis.hypothesis_id,
            "kind": hypothesis.kind,
            "source_key": source_key,
            "count": int(count),
            "step_idx": int(current_step),
        }
        self._append_probe_event(dict(self.last_event))

    def _component_snapshot(self, *, component_id: str, anchor_key: str) -> dict[str, Any]:
        normalized_component_id = str(component_id or "").strip()
        normalized_anchor_key = str(anchor_key or "").strip()
        for entry in self._last_components:
            if normalized_component_id and str(entry.get("component_id", "") or "") == normalized_component_id:
                return dict(entry)
            if normalized_anchor_key and str(entry.get("anchor_key", "") or "") == normalized_anchor_key:
                return dict(entry)
        return {}

    def _goal_relation_anchor(self, hypothesis: RuleHypothesis) -> dict[str, Any]:
        target = self._component_snapshot(
            component_id=hypothesis.target_component_id,
            anchor_key=hypothesis.target_anchor_key,
        )
        if not target:
            target = {
                "component_id": hypothesis.target_component_id,
                "anchor_key": hypothesis.target_anchor_key,
                "center_row": int(hypothesis.target_center_row),
                "center_col": int(hypothesis.target_center_col),
                "value": 0,
                "size": 0,
            }
        target_row = int(target.get("center_row", hypothesis.target_center_row) or 0)
        target_col = int(target.get("center_col", hypothesis.target_center_col) or 0)
        target_value = int(target.get("value", 0) or 0)
        target_size = int(target.get("size", 0) or 0)
        ranked: list[tuple[float, int, str, dict[str, Any]]] = []
        for entry in self._last_components:
            component_id = str(entry.get("component_id", "") or "")
            anchor_key = str(entry.get("anchor_key", "") or "")
            if component_id in {hypothesis.source_component_id, hypothesis.target_component_id}:
                continue
            if anchor_key in {hypothesis.source_anchor_key, hypothesis.target_anchor_key}:
                continue
            row = int(entry.get("center_row", 0) or 0)
            col = int(entry.get("center_col", 0) or 0)
            distance = abs(row - target_row) + abs(col - target_col)
            if distance <= 0:
                continue
            value = int(entry.get("value", 0) or 0)
            size = int(entry.get("size", 0) or 0)
            same_value = bool(target_value and value == target_value)
            same_size = bool(target_size and size == target_size)
            aligned = row == target_row or col == target_col
            near = distance <= 8
            score = (
                (2.4 if same_value else 0.0)
                + (0.9 if same_size else 0.0)
                + (0.8 if aligned else 0.0)
                + (0.45 if near else 0.0)
                - min(distance, 24) * 0.035
            )
            ranked.append((score, distance, component_id or anchor_key, dict(entry)))
        if not ranked:
            return {}
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        if ranked[0][0] <= -0.25:
            return {}
        return ranked[0][3]

    def _goal_relation_step(self, hypothesis: RuleHypothesis) -> dict[str, Any]:
        anchor = self._goal_relation_anchor(hypothesis)
        if not anchor:
            return {}
        target = self._component_snapshot(
            component_id=hypothesis.target_component_id,
            anchor_key=hypothesis.target_anchor_key,
        )
        target_row = int(target.get("center_row", hypothesis.target_center_row) or hypothesis.target_center_row)
        target_col = int(target.get("center_col", hypothesis.target_center_col) or hypothesis.target_center_col)
        goal_row = int(anchor.get("center_row", 0) or 0)
        goal_col = int(anchor.get("center_col", 0) or 0)
        actions = self._motion_probe_actions(
            motion_dr=int(goal_row) - int(target_row),
            motion_dc=int(goal_col) - int(target_col),
        )
        if not actions:
            return {}
        probe_index = max(0, int(hypothesis.followup_attempts) - 1)
        action_name = actions[probe_index % len(actions)]
        return {
            "component_id": hypothesis.target_component_id,
            "anchor_key": hypothesis.target_anchor_key,
            "center_row": target_row,
            "center_col": target_col,
            "action_name": action_name,
            "hypothesis_id": hypothesis.hypothesis_id,
            "hypothesis_kind": hypothesis.kind,
            "goal_component_id": str(anchor.get("component_id", "") or ""),
            "goal_anchor_key": str(anchor.get("anchor_key", "") or ""),
            "goal_center_row": goal_row,
            "goal_center_col": goal_col,
            "stage": "goal_relation_probe",
        }

    def _goal_verify_step(self, hypothesis: RuleHypothesis, *, stage: str) -> dict[str, Any]:
        anchor = self._goal_relation_anchor(hypothesis)
        if not anchor:
            return {}
        return {
            "component_id": str(anchor.get("component_id", "") or ""),
            "anchor_key": str(anchor.get("anchor_key", "") or ""),
            "center_row": int(anchor.get("center_row", 0) or 0),
            "center_col": int(anchor.get("center_col", 0) or 0),
            "action_name": "ACTION6",
            "hypothesis_id": hypothesis.hypothesis_id,
            "hypothesis_kind": hypothesis.kind,
            "goal_component_id": str(anchor.get("component_id", "") or ""),
            "goal_anchor_key": str(anchor.get("anchor_key", "") or ""),
            "stage": stage,
        }

    def _attribute_constraint_step(self, hypothesis: RuleHypothesis) -> dict[str, Any]:
        target = self._component_snapshot(
            component_id=hypothesis.target_component_id,
            anchor_key=hypothesis.target_anchor_key,
        )
        if not target:
            return {}
        anchor = self._goal_relation_anchor(hypothesis)
        if not anchor:
            return {}
        target_value = int(target.get("value", hypothesis.target_value) or hypothesis.target_value)
        goal_value = int(anchor.get("value", 0) or 0)
        if goal_value and target_value == goal_value:
            return {
                "component_id": hypothesis.target_component_id,
                "anchor_key": hypothesis.target_anchor_key,
                "center_row": int(target.get("center_row", hypothesis.target_center_row) or 0),
                "center_col": int(target.get("center_col", hypothesis.target_center_col) or 0),
                "action_name": "ACTION6",
                "hypothesis_id": hypothesis.hypothesis_id,
                "hypothesis_kind": hypothesis.kind,
                "goal_component_id": str(anchor.get("component_id", "") or ""),
                "goal_anchor_key": str(anchor.get("anchor_key", "") or ""),
                "stage": "attribute_constraint_verify",
            }
        return {
            "component_id": hypothesis.source_component_id,
            "anchor_key": hypothesis.source_anchor_key,
            "center_row": hypothesis.source_center_row,
            "center_col": hypothesis.source_center_col,
            "action_name": self._source_action_name(hypothesis),
            "hypothesis_id": hypothesis.hypothesis_id,
            "hypothesis_kind": hypothesis.kind,
            "goal_component_id": str(anchor.get("component_id", "") or ""),
            "goal_anchor_key": str(anchor.get("anchor_key", "") or ""),
            "target_value": target_value,
            "goal_value": goal_value,
            "stage": "attribute_constraint_actuate",
        }

    def _note_scoreless_success_selected(
        self,
        *,
        selected_component_id: str,
        selected_anchor_key: str,
        current_step: int,
        progress_signatures: list[str] | None = None,
    ) -> bool:
        signatures = [
            str(item or "").strip()
            for item in list(progress_signatures or [])
            if str(item or "").strip()
        ]
        for hypothesis in self._hypotheses.values():
            if int(hypothesis.expires_at) < int(current_step):
                continue
            if str(hypothesis.status or "") not in {"active", "verified"}:
                continue
            matches_target = (
                selected_component_id
                and selected_component_id == hypothesis.target_component_id
            ) or (
                selected_anchor_key
                and selected_anchor_key == hypothesis.target_anchor_key
            )
            matches_source = (
                selected_component_id
                and selected_component_id == hypothesis.source_component_id
            ) or (
                selected_anchor_key
                and selected_anchor_key == hypothesis.source_anchor_key
            )
            if not (matches_target or matches_source):
                continue
            fresh_signature = next(
                (item for item in signatures if item != str(hypothesis.last_progress_signature or "")),
                "",
            )
            if fresh_signature:
                hypothesis.last_progress_signature = fresh_signature
                hypothesis.scoreless_success_count = 0
                hypothesis.scoreless_progress_count += 1
                if hypothesis.kind != "global_state_change":
                    hypothesis.followup_attempts = max(0, int(hypothesis.followup_attempts) - 1)
                hypothesis.last_step = int(current_step)
                hypothesis.expires_at = int(current_step) + self.config.ttl_steps
                hypothesis.score = round(min(6.0, max(0.0, hypothesis.score) + 0.12), 4)
                if (
                    hypothesis.kind != "global_state_change"
                    and self._source_action_name(hypothesis) == "ACTION6"
                ):
                    if self._progress_lease_hypothesis_id != hypothesis.hypothesis_id:
                        self.progress_lease_count += 1
                    self._progress_lease_hypothesis_id = hypothesis.hypothesis_id
                if self._is_bounded_click_unlock(hypothesis) or hypothesis.kind == "global_state_change":
                    source_key = self._hypothesis_source_key(hypothesis)
                    if source_key:
                        source_count = self._source_scoreless_progress_counts.get(source_key, 0) + 1
                        self._source_scoreless_progress_counts[source_key] = source_count
                        if source_count >= 2:
                            self._suppress_scoreless_source(
                                hypothesis=hypothesis,
                                current_step=current_step,
                                count=source_count,
                            )
                            return True
                progress_limit = self._scoreless_progress_limit(hypothesis.kind)
                if hypothesis.scoreless_progress_count >= progress_limit:
                    hypothesis.status = "scoreless_churn"
                    if self._progress_lease_hypothesis_id == hypothesis.hypothesis_id:
                        self._progress_lease_hypothesis_id = ""
                    self.scoreless_churn_count += 1
                    self.disconfirmed_count += 1
                    self.last_event = {
                        "event": "scoreless_progress_budget_exhausted",
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "kind": hypothesis.kind,
                        "count": int(hypothesis.scoreless_progress_count),
                        "step_idx": int(current_step),
                    }
                    self._append_probe_event(dict(self.last_event))
                    return True
                self.last_event = {
                    "event": "scoreless_progress",
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "kind": hypothesis.kind,
                    "count": int(hypothesis.scoreless_progress_count),
                    "step_idx": int(current_step),
                }
                self._append_probe_event(dict(self.last_event))
                return False
            hypothesis.scoreless_success_count += 1
            hypothesis.last_step = int(current_step)
            hypothesis.score = round(max(0.0, hypothesis.score - 0.35), 4)
            threshold = self._scoreless_churn_threshold(hypothesis.kind)
            if hypothesis.scoreless_success_count >= threshold:
                hypothesis.status = "scoreless_churn"
                if self._progress_lease_hypothesis_id == hypothesis.hypothesis_id:
                    self._progress_lease_hypothesis_id = ""
                self.scoreless_churn_count += 1
                self.disconfirmed_count += 1
                self.last_event = {
                    "event": "scoreless_churn",
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "kind": hypothesis.kind,
                    "step_idx": int(current_step),
                }
                return True
            self.last_event = {
                "event": "scoreless_success",
                "hypothesis_id": hypothesis.hypothesis_id,
                "kind": hypothesis.kind,
                "count": int(hypothesis.scoreless_success_count),
                "step_idx": int(current_step),
            }
            return False
        return False

    def record_effect(
        self,
        *,
        state: Any,
        previous_state: Any | None,
        selected_summary: Any,
        action_name: str,
        family: str,
        board_changed: bool,
        score_delta: float,
        effect_kind: str,
        quality: dict[str, Any] | None = None,
        board_rows: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        current_step = int(getattr(state, "step_index", 0) or 0)
        level_idx = int(getattr(state, "level_number", 0) or 0)
        self._prune(current_step=current_step)
        self._last_level_idx = int(level_idx)
        self._last_components = self._normalized_components(state)
        normalized_rows = self._normalized_board_rows(board_rows)
        visual_violation_delta = 0
        if normalized_rows:
            self._last_board_rows = normalized_rows
            current_visual_violation_count = self._visual_stencil_violation_count(normalized_rows)
            if current_visual_violation_count is not None:
                if self._last_visual_stencil_violation_count is not None and level_idx >= 2:
                    visual_violation_delta = int(current_visual_violation_count) - int(
                        self._last_visual_stencil_violation_count
                    )
                self._last_visual_stencil_violation_count = int(current_visual_violation_count)
                if self._min_visual_stencil_violation_count is None:
                    self._min_visual_stencil_violation_count = int(current_visual_violation_count)
                else:
                    self._min_visual_stencil_violation_count = min(
                        int(self._min_visual_stencil_violation_count),
                        int(current_visual_violation_count),
                    )
                if (
                    level_idx >= 2
                    and int(current_visual_violation_count) == 0
                    and abs(float(score_delta or 0.0)) <= 1e-9
                ):
                    self.visual_stencil_zero_violation_scoreless_count += 1
                    if self.visual_stencil_progress_count > 0:
                        self.visual_stencil_zero_violation_after_progress_count += 1
        normalized_family_early = str(family or "").strip().lower()
        selected_source_early = str(self._selected_value(selected_summary, "source") or "").strip()
        allow_l1_auxiliary_discovery = bool(
            int(level_idx) == 1
            and (
                (
                    normalized_family_early == "auxiliary"
                    and selected_source_early in {"l1_action_family_quota", "stabilize_verify"}
                )
                or selected_source_early == "stabilize_verify"
                or (
                    _enable_l1_quota_synthetic_anchor()
                    and normalized_family_early == "move"
                    and selected_source_early == "l1_action_family_quota"
                )
            )
        )
        if level_idx < 2 and not allow_l1_auxiliary_discovery:
            if not board_changed:
                self._settle_unstable_windows(current_step=current_step, level_idx=level_idx)
            self._last_visual_stencil_violation_count = None
            self._min_visual_stencil_violation_count = None
            return

        source_component_id = str(
            self._selected_value(selected_summary, "component_id", "candidate_id") or ""
        ).strip()
        source_anchor_key = str(self._selected_value(selected_summary, "anchor_key") or "").strip()
        source_row, source_col = self._selected_center(selected_summary)
        source_entry = self._object_table_by_component(state).get(source_component_id)
        if source_entry is not None:
            source_row, source_col = self._component_center(source_entry)
        selected_source = str(self._selected_value(selected_summary, "source") or "").strip()
        selected_window_id = str(self._selected_value(selected_summary, "window_id") or "").strip()
        selected_stage = str(self._selected_value(selected_summary, "stage") or "").strip()
        selected_continuation_kind = str(
            self._selected_value(selected_summary, "continuation_kind") or ""
        ).strip()
        if selected_source:
            self._bump(self.source_histogram, selected_source)
        score_delta_value = float(score_delta or 0.0)
        if _enable_score_event_closure() and abs(score_delta_value) <= 1e-9:
            closure_score_delta = float(getattr(state, "score_delta", 0) or 0)
        else:
            closure_score_delta = score_delta_value
        quality = dict(quality or {})
        previous_level_idx = int(getattr(previous_state, "level_number", level_idx) or level_idx)
        level_advanced = int(level_idx) > int(previous_level_idx)
        constraint_progress = self._constraint_progress_signal(
            visual_violation_delta=visual_violation_delta,
            quality=quality,
        )
        closure_signal = (
            abs(closure_score_delta) > 1e-9
            or bool(level_advanced)
            or bool(constraint_progress)
        )
        selected_visual_candidate_key = ""
        if selected_source == "goal_probe":
            attempt_key = self._goal_probe_attempt_key(
                component_id=source_component_id,
                anchor_key=source_anchor_key,
            )
            self._goal_probe_attempts[attempt_key] = int(
                self._goal_probe_attempts.get(attempt_key, 0) or 0
            ) + 1
            self.goal_probe_issued_count += 1
            if board_changed or abs(score_delta_value) > 1e-9:
                self.goal_probe_success_count += 1
            if board_changed:
                self.goal_probe_board_change_success_count += 1
            if abs(score_delta_value) > 1e-9:
                self.goal_probe_score_success_count += 1
            if level_advanced:
                self.goal_probe_level_success_count += 1
            if constraint_progress:
                self.goal_probe_constraint_progress_success_count += 1
            if abs(score_delta_value) > 1e-9:
                self.actuator_group_completed_count += 1

        stabilize_windows: list[_StabilizeVerifyWindow] = []
        if selected_source == "stabilize_verify":
            stabilize_windows = self._note_stabilize_verify_selection(
                selected_component_id=source_component_id,
                selected_anchor_key=source_anchor_key,
                selected_window_id=selected_window_id,
                selected_stage=selected_stage,
                selected_continuation_kind=selected_continuation_kind,
                current_step=current_step,
                level_idx=level_idx,
                board_changed=bool(board_changed),
                closed=bool(closure_signal),
                score_delta=score_delta_value,
                level_advanced=bool(level_advanced),
                constraint_progress=bool(constraint_progress),
            )
        active_lease = self._active_stabilize_lease_window(
            current_step=current_step,
            level_idx=level_idx,
        )
        if (
            active_lease is not None
            and selected_source
            and str(active_lease.window_id or "") not in {
                str(window.window_id or "") for window in stabilize_windows
            }
            and int(current_step) not in self._stabilize_lease_preempted_steps
        ):
            active_lease.lease_preempt_count += 1
            self._stabilize_lease_preempted_steps.add(int(current_step))
            self.stabilize_window_preempted_count += 1

        closure_reason = (
            "score"
            if abs(closure_score_delta) > 1e-9
            else "level"
            if level_advanced
            else "constraint"
        )
        closure_eligible_sources = (
            _EXPANDED_CLOSURE_ELIGIBLE_SOURCES
            if _enable_expanded_closure_sources()
            else _LEGACY_CLOSURE_ELIGIBLE_SOURCES
        )
        if closure_signal and selected_source in closure_eligible_sources:
            self.rule_closed_success_count += 1
            if closure_reason == "score":
                self.rule_closed_score_count += 1
            elif closure_reason == "level":
                self.rule_closed_level_count += 1
            else:
                self.rule_closed_constraint_count += 1
            closed_count = self._mark_selected_rule_closed(
                selected_component_id=source_component_id,
                selected_anchor_key=source_anchor_key,
                current_step=current_step,
                reason=closure_reason,
            )
            if closed_count <= 0 and selected_source in {"goal_probe", "stabilize_verify"}:
                self._ensure_closed_rule_hypothesis(
                    level_idx=level_idx,
                    selected_component_id=source_component_id,
                    selected_anchor_key=source_anchor_key,
                    selected_row=source_row,
                    selected_col=source_col,
                    selected_source=selected_source,
                    action_family=str(family or "").strip().lower(),
                    action_name=str(action_name or "").strip().upper(),
                    current_step=current_step,
                    reason=closure_reason,
                )
            closure_windows = list(stabilize_windows)
            if selected_source in {"goal_probe", "rule_hypothesis"}:
                seen_window_ids = {str(window.window_id) for window in closure_windows}
                for window in self._matching_recent_stabilize_windows(
                    selected_component_id=source_component_id,
                    selected_anchor_key=source_anchor_key,
                    current_step=current_step,
                    level_idx=level_idx,
                    require_issued=True,
                ):
                    if str(window.window_id) in seen_window_ids:
                        continue
                    closure_windows.append(window)
                    seen_window_ids.add(str(window.window_id))
                for focus_anchor in self._quality_anchor_values(quality):
                    for window in self._matching_recent_stabilize_windows(
                        selected_component_id="",
                        selected_anchor_key=focus_anchor,
                        current_step=current_step,
                        level_idx=level_idx,
                        require_issued=True,
                    ):
                        if str(window.window_id) in seen_window_ids:
                            continue
                        closure_windows.append(window)
                        seen_window_ids.add(str(window.window_id))
            if closure_windows:
                closure_kind = "chain" if selected_window_id and selected_stage == "lease_followup" else "exact"
                self._mark_stabilize_closed(
                    windows=closure_windows,
                    current_step=current_step,
                    reason=closure_reason,
                    closure_kind=closure_kind,
                )

        if selected_source == "rule_hypothesis" and abs(score_delta_value) <= 1e-9:
            selected_visual_candidate_key = self._note_visual_stencil_selected(
                component_id=source_component_id,
                anchor_key=source_anchor_key,
                current_step=current_step,
            )

        if selected_source == "rule_hypothesis" and (board_changed or abs(score_delta_value) > 1e-9):
            self.followup_success_count += 1
            outcome = self._note_visual_constraint_outcome(
                candidate_key=selected_visual_candidate_key,
                visual_violation_delta=visual_violation_delta,
                board_changed=board_changed,
                current_step=current_step,
            )
            if not outcome:
                if visual_violation_delta < 0:
                    self.visual_stencil_progress_count += 1
                    outcome = "progress"
                elif visual_violation_delta > 0:
                    self.visual_stencil_regress_count += 1
                    outcome = "regress"
            if outcome == "progress":
                self.last_event = {
                    "event": "visual_stencil_progress",
                    "delta": int(visual_violation_delta),
                    "step_idx": int(current_step),
                }
                self._append_probe_event(dict(self.last_event))
            elif outcome == "regress":
                self.last_event = {
                    "event": "visual_stencil_regress",
                    "delta": int(visual_violation_delta),
                    "step_idx": int(current_step),
                }
                self._append_probe_event(dict(self.last_event))

        if abs(score_delta_value) > 1e-9:
            closed_hypotheses = {
                key: hypothesis
                for key, hypothesis in self._hypotheses.items()
                if not level_advanced
                and str(hypothesis.closure_status or "") == "closed"
                and int(hypothesis.expires_at) >= int(current_step)
                and int(hypothesis.level_idx) == int(level_idx)
            }
            self._hypotheses.clear()
            self._hypotheses.update(closed_hypotheses)
            self._progress_lease_hypothesis_id = ""
            self._source_scoreless_progress_counts = {}
            self._suppressed_scoreless_sources = set()
            self.last_event = {"event": "score_change_reset", "step_idx": current_step}
            return

        if not board_changed:
            self._settle_unstable_windows(current_step=current_step, level_idx=level_idx)
            if selected_source == "rule_hypothesis":
                self._note_visual_constraint_outcome(
                    candidate_key=selected_visual_candidate_key,
                    visual_violation_delta=0,
                    board_changed=False,
                    current_step=current_step,
                )
                self._disconfirm_selected(
                    selected_component_id=source_component_id,
                    selected_anchor_key=source_anchor_key,
                    current_step=current_step,
            )
            return

        normalized_family = str(family or "").strip().lower()
        if normalized_family not in _RULE_ACTION_FAMILIES:
            return
        normalized_action = str(action_name or "").strip().upper()
        deltas = self._changed_components(state=state, previous_state=previous_state)
        if not deltas and source_component_id:
            deltas = [
                _ComponentDelta(
                    component_id=source_component_id,
                    anchor_key=source_anchor_key,
                    center_row=int(source_row),
                    center_col=int(source_col),
                    previous_center_row=int(source_row),
                    previous_center_col=int(source_col),
                    value=0,
                    previous_value=None,
                    size=0,
                    previous_size=None,
                    change_kind="hash_only",
                )
            ]
        effect_record = self._record_effect_ledger_entry(
            level_idx=level_idx,
            current_step=current_step,
            source_component_id=source_component_id,
            source_anchor_key=source_anchor_key,
            source_row=source_row,
            source_col=source_col,
            selected_source=selected_source,
            action_family=normalized_family,
            action_name=normalized_action,
            effect_kind=effect_kind,
            deltas=deltas,
        )
        linked_effect_id = str(effect_record.effect_id if effect_record is not None else "")
        if len(deltas) > 8:
            if selected_source == "stabilize_verify" and stabilize_windows:
                for window in stabilize_windows[:1]:
                    if linked_effect_id:
                        window.linked_effect_id = linked_effect_id
                    self._append_stabilize_window_observations(
                        window,
                        selected_component_id=source_component_id,
                        selected_anchor_key=source_anchor_key,
                        deltas=deltas[:8],
                    )
            if not (selected_source == "stabilize_verify" and stabilize_windows):
                self._open_or_update_stabilize_window(
                    level_idx=level_idx,
                    current_step=current_step,
                    source_component_id=source_component_id,
                    source_anchor_key=source_anchor_key,
                    source_row=source_row,
                    source_col=source_col,
                    selected_source=selected_source,
                    action_family=normalized_family,
                    action_name=normalized_action,
                    deltas=deltas[:1],
                    linked_effect_id=linked_effect_id,
                    delayed_continuation=bool(quality.get("delayed_continuation", False)),
                    retest=selected_source == "stabilize_verify",
                )
            if selected_source == "rule_hypothesis":
                self._note_scoreless_success_selected(
                    selected_component_id=source_component_id,
                    selected_anchor_key=source_anchor_key,
                    current_step=current_step,
                    progress_signatures=[
                        f"broad|{int(level_idx)}|{source_component_id}|{len(deltas)}|{int(current_step)}"
                    ],
                )
            self._record_broad_effect_source(
                level_idx=level_idx,
                source_component_id=source_component_id,
                source_anchor_key=source_anchor_key,
                source_row=source_row,
                source_col=source_col,
                action_family=normalized_family,
                action_name=normalized_action,
                candidate_source=selected_source,
                effect_kind=effect_kind,
                current_step=current_step,
                candidate_count=len(deltas),
            )
            self._prune(current_step=current_step)
            return
        self._note_boundary_volatility(
            deltas=deltas,
            source_component_id=source_component_id,
            source_anchor_key=source_anchor_key,
        )
        deltas = self._filter_volatile_deltas(deltas)

        ranked = sorted(
            deltas,
            key=lambda item: (
                item.component_id != source_component_id,
                abs(int(item.center_row) - int(source_row)) + abs(int(item.center_col) - int(source_col)),
                str(item.component_id),
            ),
        )[:3]
        if selected_source == "stabilize_verify" and stabilize_windows:
            for window in stabilize_windows[:1]:
                if linked_effect_id:
                    window.linked_effect_id = linked_effect_id
                self._append_stabilize_window_observations(
                    window,
                    selected_component_id=source_component_id,
                    selected_anchor_key=source_anchor_key,
                    deltas=ranked,
                )
        if not (selected_source == "stabilize_verify" and stabilize_windows):
            if bool(quality.get("delayed_continuation", False)):
                latest_window = self._latest_unstable_window(
                    current_step=current_step,
                    level_idx=level_idx,
                )
                if latest_window is not None:
                    self._open_or_update_stabilize_window(
                        level_idx=level_idx,
                        current_step=current_step,
                        source_component_id=latest_window.source_component_id,
                        source_anchor_key=latest_window.source_anchor_key,
                        source_row=latest_window.source_center_row,
                        source_col=latest_window.source_center_col,
                        selected_source=latest_window.selected_source,
                        action_family=latest_window.action_family,
                        action_name=latest_window.action_name,
                        deltas=ranked,
                        linked_effect_id=linked_effect_id,
                        delayed_continuation=True,
                        retest=selected_source == "stabilize_verify",
                    )
                else:
                    self._open_or_update_stabilize_window(
                        level_idx=level_idx,
                        current_step=current_step,
                        source_component_id=source_component_id,
                        source_anchor_key=source_anchor_key,
                        source_row=source_row,
                        source_col=source_col,
                        selected_source=selected_source,
                        action_family=normalized_family,
                        action_name=normalized_action,
                        deltas=ranked,
                        linked_effect_id=linked_effect_id,
                        delayed_continuation=True,
                        retest=selected_source == "stabilize_verify",
                    )
            else:
                self._open_or_update_stabilize_window(
                    level_idx=level_idx,
                    current_step=current_step,
                    source_component_id=source_component_id,
                    source_anchor_key=source_anchor_key,
                    source_row=source_row,
                    source_col=source_col,
                    selected_source=selected_source,
                    action_family=normalized_family,
                    action_name=normalized_action,
                    deltas=ranked,
                    linked_effect_id=linked_effect_id,
                    delayed_continuation=False,
                    retest=selected_source == "stabilize_verify",
                )
        self._record_effect_actuators(
            level_idx=level_idx,
            source_component_id=source_component_id,
            source_anchor_key=source_anchor_key,
            source_row=source_row,
            source_col=source_col,
            action_family=normalized_family,
            action_name=normalized_action,
            effect_kind=effect_kind,
            deltas=ranked,
            current_step=current_step,
            visual_violation_delta=visual_violation_delta,
        )
        if selected_source == "rule_hypothesis":
            if visual_violation_delta > 0:
                self._disconfirm_selected(
                    selected_component_id=source_component_id,
                    selected_anchor_key=source_anchor_key,
                    current_step=current_step,
                    force=True,
                )
                self._prune(current_step=current_step)
                return
            self._note_scoreless_success_selected(
                selected_component_id=source_component_id,
                selected_anchor_key=source_anchor_key,
                current_step=current_step,
                progress_signatures=[self._progress_signature(delta) for delta in ranked],
            )
            structural_unlock = bool(quality.get("component_count_delta")) or bool(
                quality.get("unresolved_components_delta")
            )
            structural_unlock = structural_unlock or visual_violation_delta < 0
            if not structural_unlock:
                self._prune(current_step=current_step)
                return

        for delta in ranked:
            if self._is_zero_perimeter_point(row=source_row, col=source_col):
                self.perimeter_effect_suppressed_count += 1
                continue
            kind = self._classify_kind(
                delta=delta,
                source_component_id=source_component_id,
                source_row=source_row,
                source_col=source_col,
                quality=quality,
            )
            motion_dr = int(delta.center_row) - int(delta.previous_center_row)
            motion_dc = int(delta.center_col) - int(delta.previous_center_col)
            if (
                kind == "unlock_or_route_candidate"
                and normalized_action == "ACTION6"
                and motion_dr == 0
                and motion_dc == 0
                and self._source_key(
                    component_id=source_component_id,
                    anchor_key=source_anchor_key,
                )
                in self._suppressed_scoreless_sources
            ):
                continue
            hypothesis_id = self._hypothesis_key(
                level_idx=level_idx,
                source_component_id=source_component_id,
                source_anchor_key=source_anchor_key,
                target_component_id=delta.component_id,
                target_anchor_key=delta.anchor_key,
                action_name=normalized_action,
                kind=kind,
            )
            hypothesis = self._hypotheses.get(hypothesis_id)
            if hypothesis is None:
                hypothesis = RuleHypothesis(
                    hypothesis_id=hypothesis_id,
                    kind=kind,
                    level_idx=level_idx,
                    source_component_id=source_component_id,
                    source_anchor_key=source_anchor_key,
                    source_center_row=int(source_row),
                    source_center_col=int(source_col),
                    target_component_id=delta.component_id,
                    target_anchor_key=delta.anchor_key,
                    target_center_row=int(delta.center_row),
                    target_center_col=int(delta.center_col),
                    action_family=normalized_family,
                    action_name=normalized_action,
                    candidate_source=selected_source or "selected_target",
                    relation_dr=int(delta.center_row) - int(source_row),
                    relation_dc=int(delta.center_col) - int(source_col),
                    target_motion_dr=int(delta.center_row) - int(delta.previous_center_row),
                    target_motion_dc=int(delta.center_col) - int(delta.previous_center_col),
                    target_value=int(delta.value),
                    target_previous_value=int(delta.previous_value or 0),
                    target_size=int(delta.size),
                    created_step=current_step,
                    expires_at=current_step + self.config.ttl_steps,
                )
                self._hypotheses[hypothesis_id] = hypothesis
                self.created_count += 1
                self._bump(self.kind_histogram, kind)
            hypothesis.effect_count += 1
            hypothesis.last_step = current_step
            hypothesis.expires_at = current_step + self.config.ttl_steps
            hypothesis.last_effect_kind = str(effect_kind or "")
            if motion_dr or motion_dc:
                hypothesis.target_motion_dr = motion_dr
                hypothesis.target_motion_dc = motion_dc
            hypothesis.source_center_row = int(source_row)
            hypothesis.source_center_col = int(source_col)
            hypothesis.target_center_row = int(delta.center_row)
            hypothesis.target_center_col = int(delta.center_col)
            hypothesis.target_anchor_key = delta.anchor_key or hypothesis.target_anchor_key
            hypothesis.target_value = int(delta.value)
            hypothesis.target_previous_value = int(delta.previous_value or 0)
            hypothesis.target_size = int(delta.size)
            hypothesis.last_progress_signature = self._progress_signature(delta)
            verification_signal = hypothesis.effect_count >= 2 or kind in {
                "remote_state_change",
                "paired_motion",
                "attribute_cycle",
                "unlock_or_route_candidate",
            }
            if verification_signal:
                if hypothesis.verification_count <= 0:
                    self.verified_count += 1
                hypothesis.verification_count += 1
                hypothesis.status = "verified"
                if str(hypothesis.closure_status or "") != "closed":
                    hypothesis.closure_status = "verified"
            hypothesis.score = round(
                min(
                    6.0,
                    2.4
                    + 0.35 * float(hypothesis.effect_count)
                    + 0.45 * float(hypothesis.verification_count)
                    - 0.25 * float(hypothesis.followup_attempts)
                    - 0.45 * float(hypothesis.disconfirm_count),
                ),
                4,
            )
            self.last_event = {
                "event": "record_effect",
                "hypothesis_id": hypothesis_id,
                "kind": kind,
                "status": hypothesis.status,
                "step_idx": current_step,
            }
        self._prune(current_step=current_step)

    def candidate_entries(self, *, step_index: int | None = None, max_entries: int | None = None) -> list[dict[str, Any]]:
        current_step = int(step_index or 0)
        self._prune(current_step=current_step)
        limit = self.config.max_candidates if max_entries is None else int(max_entries)
        entries: list[dict[str, Any]] = self._visual_stencil_candidates(
            limit=limit,
            current_step=current_step,
        )
        if self._visual_near_solved_active():
            if len(entries) < limit:
                existing_keys = {
                    (
                        str(item.get("component_id", "") or ""),
                        str(item.get("anchor_key", "") or ""),
                        str(item.get("action_name", "") or ""),
                    )
                    for item in entries
                }
                entries.extend(
                    self._visual_stencil_ledger_candidates(
                        limit=limit - len(entries),
                        current_step=current_step,
                        existing_keys=existing_keys,
                        ignore_attempt_limit=True,
                        require_current_step=True,
                    )
                )
            suppressed = sum(
                1
                for item in self._hypotheses.values()
                if item.status in {"active", "verified"}
                and item.followup_attempts < self.config.verification_budget
            )
            self.visual_stencil_near_solved_step_count += 1
            self.visual_stencil_near_solved_suppressed_count += int(suppressed)
            if suppressed:
                self.last_event = {
                    "event": "visual_stencil_near_solved_focus",
                    "suppressed_count": int(suppressed),
                    "step_idx": int(current_step),
                }
            return entries[:limit]
        if len(entries) < limit:
            lease_entries = self._stabilize_lease_candidates(
                limit=min(2, limit - len(entries)),
                current_step=current_step,
            )
            lease_seen = {
                (
                    str(item.get("component_id", "") or ""),
                    str(item.get("anchor_key", "") or ""),
                    str(item.get("family", "") or ""),
                    str(item.get("action_name", "") or ""),
                )
                for item in entries
            }
            for item in lease_entries:
                key = (
                    str(item.get("component_id", "") or ""),
                    str(item.get("anchor_key", "") or ""),
                    str(item.get("family", "") or ""),
                    str(item.get("action_name", "") or ""),
                )
                if key in lease_seen:
                    continue
                entries.append(item)
                lease_seen.add(key)
                if len(entries) >= limit:
                    break
        if len(entries) < limit:
            stabilize_entries = self._stabilize_verify_candidates(
                limit=min(2, limit - len(entries)),
                current_step=current_step,
            )
            stabilize_seen = {
                (str(item.get("component_id", "") or ""), str(item.get("anchor_key", "") or ""))
                for item in entries
            }
            for item in stabilize_entries:
                key = (str(item.get("component_id", "") or ""), str(item.get("anchor_key", "") or ""))
                if key in stabilize_seen:
                    continue
                entries.append(item)
                stabilize_seen.add(key)
                if len(entries) >= limit:
                    break
        ranked = sorted(
            self._hypotheses.values(),
            key=lambda item: (
                0 if str(item.closure_status or "") == "closed" else 1,
                0 if item.status == "verified" else 1,
                0 if item.hypothesis_id == self._progress_lease_hypothesis_id else 1,
                1 if item.kind == "global_state_change" else 0,
                -int(item.scoreless_progress_count),
                int(item.followup_attempts),
                int(item.scoreless_success_count),
                -int(item.verification_count),
                -int(item.effect_count),
                -float(item.score),
                int(item.last_step),
                str(item.hypothesis_id),
            ),
        )
        seen: set[tuple[str, str]] = {
            (str(item.get("component_id", "") or ""), str(item.get("anchor_key", "") or ""))
            for item in entries
        }
        goal_probe_entries = self._goal_probe_candidates(
            limit=min(2, limit),
            current_step=current_step,
        )
        hypothesis_limit = limit
        if goal_probe_entries:
            hypothesis_limit = max(len(entries), max(1, limit - min(2, len(goal_probe_entries))))
            closed_or_stabilize_present = any(
                str(item.get("source", "") or "") == "stabilize_verify"
                or str(item.get("closure_status", "") or "") == "closed"
                for item in entries
            ) or any(str(item.closure_status or "") == "closed" for item in self._hypotheses.values())
            goal_probe_starved = (
                int(self.goal_probe_candidate_count) >= 12
                and int(self.goal_probe_issued_count) < 4
            )
            if not closed_or_stabilize_present and int(limit) >= 2 and goal_probe_starved:
                hypothesis_limit = min(hypothesis_limit, max(len(entries), limit - 1))
                if current_step not in self._goal_probe_starvation_guard_steps:
                    self._goal_probe_starvation_guard_steps.add(current_step)
                    self.goal_probe_starvation_guard_count += 1
                if goal_probe_entries:
                    goal_probe_entries[0]["base_score"] = round(
                        float(goal_probe_entries[0].get("base_score", 0.0) or 0.0) + 2.8,
                        4,
                    )
        for hypothesis in ranked:
            if len(entries) >= hypothesis_limit:
                break
            if hypothesis.status not in {"active", "verified"}:
                continue
            if hypothesis.followup_attempts >= self.config.verification_budget:
                continue
            if self._has_zero_perimeter_source(hypothesis):
                continue
            if (
                self._is_bounded_click_unlock(hypothesis)
                or hypothesis.kind == "global_state_change"
            ) and self._source_is_suppressed(hypothesis):
                continue
            if hypothesis.kind == "global_state_change" and int(hypothesis.followup_attempts) >= 2:
                continue
            actuator_kinds = {
                "attribute_cycle",
                "global_state_change",
                "paired_motion",
                "remote_state_change",
                "unlock_or_route_candidate",
            }
            if hypothesis.kind in actuator_kinds and hypothesis.source_component_id:
                source_action_name = self._source_action_name(hypothesis)
                source_family = self._source_family(hypothesis)
                primary = {
                    "component_id": hypothesis.source_component_id,
                    "anchor_key": hypothesis.source_anchor_key,
                    "center_row": hypothesis.source_center_row,
                    "center_col": hypothesis.source_center_col,
                    "family": source_family,
                    "action_name": source_action_name,
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "hypothesis_kind": hypothesis.kind,
                    "hypothesis_status": hypothesis.status,
                    "closure_status": hypothesis.closure_status,
                    "stage": "source_actuation",
                    "base_score": round(
                        (
                            3.55
                            if hypothesis.kind == "global_state_change"
                            else 4.35
                        )
                        + min(
                            0.45 if hypothesis.kind == "global_state_change" else 0.9,
                            hypothesis.score
                            * (0.08 if hypothesis.kind == "global_state_change" else 0.14),
                        ),
                        4,
                    ),
                }
                secondary = {
                    "component_id": hypothesis.target_component_id,
                    "anchor_key": hypothesis.target_anchor_key,
                    "center_row": hypothesis.target_center_row,
                    "center_col": hypothesis.target_center_col,
                    "family": "click",
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "hypothesis_kind": hypothesis.kind,
                    "hypothesis_status": hypothesis.status,
                    "closure_status": hypothesis.closure_status,
                    "stage": "target_verify",
                    "base_score": round(
                        (
                            3.10
                            if hypothesis.kind == "global_state_change"
                            else 3.65
                        )
                        + min(
                            0.30 if hypothesis.kind == "global_state_change" else 0.55,
                            hypothesis.score
                            * (0.05 if hypothesis.kind == "global_state_change" else 0.08),
                        ),
                        4,
                    ),
                }
            else:
                source_action_name = self._source_action_name(hypothesis)
                source_family = self._source_family(hypothesis)
                primary = {
                    "component_id": hypothesis.target_component_id,
                    "anchor_key": hypothesis.target_anchor_key,
                    "center_row": hypothesis.target_center_row,
                    "center_col": hypothesis.target_center_col,
                    "family": source_family if hypothesis.target_component_id == hypothesis.source_component_id else "click",
                    "action_name": (
                        source_action_name
                        if hypothesis.target_component_id == hypothesis.source_component_id
                        else "ACTION6"
                    ),
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "hypothesis_kind": hypothesis.kind,
                    "hypothesis_status": hypothesis.status,
                    "closure_status": hypothesis.closure_status,
                    "stage": "target_followup",
                    "base_score": round(4.1 + min(0.8, hypothesis.score * 0.12), 4),
                }
                secondary = {
                    "component_id": hypothesis.source_component_id,
                    "anchor_key": hypothesis.source_anchor_key,
                    "center_row": hypothesis.source_center_row,
                    "center_col": hypothesis.source_center_col,
                    "family": source_family,
                    "action_name": source_action_name,
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "hypothesis_kind": hypothesis.kind,
                    "hypothesis_status": hypothesis.status,
                    "closure_status": hypothesis.closure_status,
                    "stage": "source_retest",
                    "base_score": round(3.85 + min(0.6, hypothesis.score * 0.10), 4),
            }
            for item in (primary, secondary):
                if len(entries) >= min(limit, hypothesis_limit):
                    break
                if self._is_zero_perimeter_point(
                    row=int(item.get("center_row", 0) or 0),
                    col=int(item.get("center_col", 0) or 0),
                ):
                    continue
                key = (str(item["component_id"]), str(item["anchor_key"]))
                if item["component_id"] and key not in seen:
                    entries.append(item)
                    seen.add(key)
        for item in goal_probe_entries:
            if len(entries) >= limit:
                break
            key = (str(item.get("component_id", "") or ""), str(item.get("anchor_key", "") or ""))
            if key in seen:
                continue
            if self._is_zero_perimeter_point(
                row=int(item.get("center_row", 0) or 0),
                col=int(item.get("center_col", 0) or 0),
            ):
                continue
            entries.append(item)
            seen.add(key)
        return entries[:limit]

    def build_followup_packet(
        self,
        *,
        selected_component_id: str,
        selected_anchor_key: str = "",
        current_step: int,
    ) -> list[dict[str, Any]]:
        self._prune(current_step=current_step)
        selected_component_id = str(selected_component_id or "").strip()
        selected_anchor_key = str(selected_anchor_key or "").strip()
        if not selected_component_id and not selected_anchor_key:
            return []
        visual_packet = self._visual_stencil_followup_packet(
            selected_component_id=selected_component_id,
            selected_anchor_key=selected_anchor_key,
            current_step=current_step,
        )
        if visual_packet:
            return visual_packet
        matches: list[RuleHypothesis] = []
        for hypothesis in self._hypotheses.values():
            if hypothesis.followup_attempts >= self.config.verification_budget:
                continue
            if hypothesis.status not in {"active", "verified"}:
                continue
            if self._has_zero_perimeter_source(hypothesis):
                continue
            if (
                self._is_bounded_click_unlock(hypothesis)
                or hypothesis.kind == "global_state_change"
            ) and self._source_is_suppressed(hypothesis):
                continue
            if hypothesis.kind == "global_state_change" and int(hypothesis.followup_attempts) >= 2:
                continue
            target_match = (
                selected_component_id
                and selected_component_id == hypothesis.target_component_id
            ) or (
                selected_anchor_key
                and selected_anchor_key == hypothesis.target_anchor_key
            )
            source_match = (
                selected_component_id
                and selected_component_id == hypothesis.source_component_id
            ) or (
                selected_anchor_key
                and selected_anchor_key == hypothesis.source_anchor_key
            )
            if target_match or source_match:
                matches.append(hypothesis)
        if not matches:
            return self._visual_stencil_followup_packet(
                selected_component_id=selected_component_id,
                selected_anchor_key=selected_anchor_key,
                current_step=current_step,
            )
        hypothesis = max(
            matches,
            key=lambda item: (
                item.status == "verified",
                -int(item.followup_attempts),
                -int(item.scoreless_success_count),
                int(item.verification_count),
                int(item.effect_count),
                float(item.score),
                -int(item.last_step),
            ),
        )
        steps: list[dict[str, Any]] = []
        actuator_kinds = {
            "attribute_cycle",
            "global_state_change",
            "paired_motion",
            "remote_state_change",
            "unlock_or_route_candidate",
        }
        if hypothesis.kind in actuator_kinds and hypothesis.source_component_id:
            source_action_name = self._source_action_name(hypothesis)
            source_step = {
                "component_id": hypothesis.source_component_id,
                "anchor_key": hypothesis.source_anchor_key,
                "center_row": hypothesis.source_center_row,
                "center_col": hypothesis.source_center_col,
                "action_name": source_action_name,
                "hypothesis_id": hypothesis.hypothesis_id,
                "hypothesis_kind": hypothesis.kind,
                "stage": "source_actuation",
            }
            if hypothesis.kind == "paired_motion":
                if (
                    source_action_name == "ACTION6"
                    and (
                        int(hypothesis.followup_attempts) < 2
                        or int(hypothesis.scoreless_progress_count) > 0
                    )
                ):
                    verify_step = dict(source_step, stage="source_repeat")
                else:
                    goal_step = (
                        self._goal_relation_step(hypothesis)
                        if int(hypothesis.followup_attempts) >= 1
                        else {}
                    )
                    if goal_step:
                        verify_step = goal_step
                    else:
                        motion_actions = self._motion_probe_actions(
                            motion_dr=hypothesis.target_motion_dr,
                            motion_dc=hypothesis.target_motion_dc,
                        )
                        motion_action = (
                            motion_actions[int(hypothesis.followup_attempts) % len(motion_actions)]
                            if motion_actions
                            else ""
                        )
                        verify_step = (
                            {
                                "component_id": hypothesis.target_component_id,
                                "anchor_key": hypothesis.target_anchor_key,
                                "center_row": hypothesis.target_center_row,
                                "center_col": hypothesis.target_center_col,
                                "action_name": motion_action,
                                "hypothesis_id": hypothesis.hypothesis_id,
                                "hypothesis_kind": hypothesis.kind,
                                "stage": "motion_relation_probe",
                            }
                            if motion_action
                            else dict(source_step, stage="source_repeat")
                        )
            elif hypothesis.kind == "attribute_cycle":
                verify_step = self._attribute_constraint_step(hypothesis)
                if not verify_step:
                    verify_step = dict(source_step, stage="source_repeat")
            elif (
                hypothesis.kind in {"global_state_change", "remote_state_change", "unlock_or_route_candidate"}
                and source_action_name == "ACTION6"
                and (
                    int(hypothesis.followup_attempts) < 2
                    or int(hypothesis.scoreless_progress_count) > 0
                )
            ):
                verify_step = dict(source_step, stage="source_repeat")
            else:
                verify_step = {
                    "component_id": hypothesis.target_component_id,
                    "anchor_key": hypothesis.target_anchor_key,
                    "center_row": hypothesis.target_center_row,
                    "center_col": hypothesis.target_center_col,
                    "action_name": "ACTION6",
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "hypothesis_kind": hypothesis.kind,
                    "stage": "target_verify",
                }
            if (
                str(verify_step.get("stage", "") or "") == "target_verify"
                and str(verify_step.get("component_id", "") or "") == str(source_step.get("component_id", "") or "")
                and (
                    not str(verify_step.get("anchor_key", "") or "")
                    or not str(source_step.get("anchor_key", "") or "")
                    or str(verify_step.get("anchor_key", "") or "") == str(source_step.get("anchor_key", "") or "")
                )
            ):
                verify_step = dict(source_step, stage="source_repeat")
            steps = [source_step, verify_step]
        else:
            selected_is_source = selected_component_id == hypothesis.source_component_id
            source_action_name = self._source_action_name(hypothesis)
            first = {
                "component_id": hypothesis.source_component_id if selected_is_source else hypothesis.target_component_id,
                "anchor_key": hypothesis.source_anchor_key if selected_is_source else hypothesis.target_anchor_key,
                "center_row": hypothesis.source_center_row if selected_is_source else hypothesis.target_center_row,
                "center_col": hypothesis.source_center_col if selected_is_source else hypothesis.target_center_col,
                "action_name": source_action_name if selected_is_source else "ACTION6",
                "hypothesis_id": hypothesis.hypothesis_id,
                "hypothesis_kind": hypothesis.kind,
                "stage": "selected_probe",
            }
            second = {
                "component_id": hypothesis.target_component_id if selected_is_source else hypothesis.source_component_id,
                "anchor_key": hypothesis.target_anchor_key if selected_is_source else hypothesis.source_anchor_key,
                "center_row": hypothesis.target_center_row if selected_is_source else hypothesis.source_center_row,
                "center_col": hypothesis.target_center_col if selected_is_source else hypothesis.source_center_col,
                "action_name": "ACTION6" if selected_is_source else source_action_name,
                "hypothesis_id": hypothesis.hypothesis_id,
                "hypothesis_kind": hypothesis.kind,
                "stage": "relation_probe",
            }
            steps = [first, second]
        compact_steps: list[dict[str, Any]] = []
        for item in steps:
            if not str(item.get("component_id", "") or "").strip():
                continue
            if str(item.get("stage", "") or "") != "source_repeat" and self._is_zero_perimeter_point(
                row=int(item.get("center_row", 0) or 0),
                col=int(item.get("center_col", 0) or 0),
            ):
                continue
            if (
                str(item.get("stage", "") or "") != "source_repeat"
                and str(item.get("stage", "") or "") != "attribute_constraint_actuate"
                and any(
                    str(existing.get("component_id", "") or "") == str(item.get("component_id", "") or "")
                    for existing in compact_steps
                )
            ):
                continue
            compact_steps.append(item)
        steps = compact_steps
        if len(steps) < 2:
            return []
        stages = [str(item.get("stage", "") or "") for item in steps]
        if len(stages) >= 2 and stages[0] == "source_actuation" and stages[1] == "attribute_constraint_actuate":
            if str(steps[0].get("component_id", "") or "") != str(steps[1].get("component_id", "") or ""):
                self._record_invariant_violation(
                    reason="attribute_constraint_actuator_mismatch",
                    hypothesis=hypothesis,
                )
                return []
        hypothesis.followup_attempts += 1
        hypothesis.last_step = int(current_step)
        hypothesis.expires_at = int(current_step) + max(2, self.config.ttl_steps // 2)
        self.followup_issued_count += 1
        if any("goal" in str(item.get("stage", "") or "") for item in steps):
            self.goal_relation_probe_count += 1
        if any("constraint" in str(item.get("stage", "") or "") for item in steps):
            self.constraint_probe_count += 1
        if any(str(item.get("stage", "") or "") == "source_repeat" for item in steps):
            self.source_burst_count += 1
        self.last_event = {
            "event": "followup_issued",
            "hypothesis_id": hypothesis.hypothesis_id,
            "kind": hypothesis.kind,
            "stage": str(steps[-1].get("stage", "") or ""),
            "step_idx": int(current_step),
        }
        self._append_probe_event(dict(self.last_event))
        return steps[:2]

    def step_metrics(self) -> dict[str, Any]:
        active = [
            item
            for item in self._hypotheses.values()
            if str(item.status or "") in {"active", "verified"}
        ]
        verified = [item for item in active if str(item.status or "") == "verified"]
        closed = [item for item in active if str(item.closure_status or "") == "closed"]
        top = max(active, key=lambda item: float(item.score), default=None)
        metrics = {
            "rule_hypothesis_active_count": len(active),
            "rule_hypothesis_verified_active_count": len(verified),
            "rule_hypothesis_closed_active_count": len(closed),
            "rule_hypothesis_top_kind": str(top.kind if top else ""),
            "rule_hypothesis_top_score": round(float(top.score if top else 0.0), 4),
            "rule_hypothesis_last_event": str(self.last_event.get("event", "") or ""),
            "rule_hypothesis_scoreless_churn_count": int(self.scoreless_churn_count),
            "rule_hypothesis_goal_relation_probe_count": int(self.goal_relation_probe_count),
            "rule_hypothesis_constraint_probe_count": int(self.constraint_probe_count),
            "rule_hypothesis_source_burst_count": int(self.source_burst_count),
            "rule_hypothesis_progress_lease_count": int(self.progress_lease_count),
            "rule_hypothesis_source_progress_suppressed_count": int(
                self.source_progress_suppressed_count
            ),
            "rule_hypothesis_perimeter_effect_suppressed_count": int(
                self.perimeter_effect_suppressed_count
            ),
            "rule_hypothesis_invariant_violation_count": int(self.invariant_violation_count),
            "rule_hypothesis_volatile_effect_suppressed_count": int(
                self.volatile_effect_suppressed_count
            ),
            "rule_hypothesis_volatile_component_count": int(len(self._volatile_components)),
            "rule_hypothesis_visual_stencil_candidate_count": int(
                self.visual_stencil_candidate_count
            ),
            "rule_hypothesis_visual_stencil_anchor_count": int(
                self.visual_stencil_anchor_count
            ),
            "rule_hypothesis_visual_stencil_attempt_count": int(
                self.visual_stencil_attempt_count
            ),
            "rule_hypothesis_visual_stencil_exhausted_count": int(
                self.visual_stencil_exhausted_count
            ),
            "rule_hypothesis_visual_stencil_progress_count": int(
                self.visual_stencil_progress_count
            ),
            "rule_hypothesis_visual_stencil_regress_count": int(
                self.visual_stencil_regress_count
            ),
            "rule_hypothesis_visual_stencil_neutral_count": int(
                self.visual_stencil_neutral_count
            ),
            "rule_hypothesis_visual_stencil_zero_violation_scoreless_count": int(
                self.visual_stencil_zero_violation_scoreless_count
            ),
            "rule_hypothesis_visual_stencil_zero_violation_after_progress_count": int(
                self.visual_stencil_zero_violation_after_progress_count
            ),
            "rule_hypothesis_visual_stencil_near_solved_step_count": int(
                self.visual_stencil_near_solved_step_count
            ),
            "rule_hypothesis_visual_stencil_near_solved_suppressed_count": int(
                self.visual_stencil_near_solved_suppressed_count
            ),
            "rule_hypothesis_visual_stencil_ledger_candidate_count": int(
                self.visual_stencil_ledger_candidate_count
            ),
            "rule_hypothesis_visual_constraint_observed_count": int(
                len(self._visual_constraint_ledger)
            ),
            "rule_hypothesis_visual_constraint_unresolved_count": int(
                sum(1 for item in self._visual_constraint_ledger.values() if item.status == "unresolved")
            ),
            "rule_hypothesis_visual_constraint_resolved_count": int(
                sum(1 for item in self._visual_constraint_ledger.values() if item.status == "resolved")
            ),
            "rule_hypothesis_visual_constraint_regress_count": int(
                sum(int(item.regress_count) for item in self._visual_constraint_ledger.values())
            ),
            "rule_hypothesis_visual_constraint_neutral_count": int(
                sum(int(item.neutral_count) for item in self._visual_constraint_ledger.values())
            ),
            "rule_hypothesis_visual_stencil_current_violation_count": int(
                self._last_visual_stencil_violation_count or 0
            ),
            "rule_hypothesis_visual_stencil_min_violation_count": int(
                self._min_visual_stencil_violation_count or 0
            ),
            "rule_hypothesis_effect_actuator_count": int(len(self._effect_actuators)),
            "rule_hypothesis_visual_repair_actuator_candidate_count": int(
                self.visual_repair_actuator_candidate_count
            ),
            "rule_hypothesis_goal_probe_candidate_count": int(self.goal_probe_candidate_count),
            "rule_hypothesis_goal_probe_issued_count": int(self.goal_probe_issued_count),
            "rule_hypothesis_goal_probe_success_count": int(self.goal_probe_success_count),
            "rule_hypothesis_goal_probe_board_change_success_count": int(
                self.goal_probe_board_change_success_count
            ),
            "rule_hypothesis_goal_probe_score_success_count": int(
                self.goal_probe_score_success_count
            ),
            "rule_hypothesis_goal_probe_level_success_count": int(
                self.goal_probe_level_success_count
            ),
            "rule_hypothesis_goal_probe_constraint_progress_success_count": int(
                self.goal_probe_constraint_progress_success_count
            ),
            "rule_hypothesis_rule_closed_success_count": int(self.rule_closed_success_count),
            "rule_hypothesis_rule_closed_score_count": int(self.rule_closed_score_count),
            "rule_hypothesis_rule_closed_level_count": int(self.rule_closed_level_count),
            "rule_hypothesis_rule_closed_constraint_count": int(self.rule_closed_constraint_count),
            "rule_hypothesis_rule_closed_synthetic_count": int(self.rule_closed_synthetic_count),
            "rule_hypothesis_closed_count": int(self.closed_count),
            "rule_hypothesis_actuator_group_detected_count": int(
                self.actuator_group_detected_count
            ),
            "rule_hypothesis_actuator_group_completed_count": int(
                self.actuator_group_completed_count
            ),
            "rule_hypothesis_followup_issued_count": int(self.followup_issued_count),
            "rule_hypothesis_followup_success_count": int(self.followup_success_count),
            "stabilize_verify_window_opened_count": int(
                self.stabilize_verify_window_opened_count
            ),
            "stabilize_verify_window_settled_count": int(
                self.stabilize_verify_window_settled_count
            ),
            "stabilize_verify_issued_count": int(self.stabilize_verify_issued_count),
            "stabilize_verify_closed_count": int(self.stabilize_verify_closed_count),
            "stabilize_verify_expired_count": int(self.stabilize_verify_expired_count),
            "stabilize_window_lease_started_count": int(
                self.stabilize_window_lease_started_count
            ),
            "stabilize_window_lease_followup_issued_count": int(
                self.stabilize_window_lease_followup_issued_count
            ),
            "stabilize_window_exact_closed_count": int(
                self.stabilize_window_exact_closed_count
            ),
            "stabilize_window_chain_closed_count": int(
                self.stabilize_window_chain_closed_count
            ),
            "stabilize_window_expired_nochange_count": int(
                self.stabilize_window_expired_nochange_count
            ),
            "stabilize_window_expired_budget_count": int(
                self.stabilize_window_expired_budget_count
            ),
            "stabilize_window_scoreless_verified_count": int(
                self.stabilize_window_scoreless_verified_count
            ),
            "stabilize_window_preempted_count": int(
                self.stabilize_window_preempted_count
            ),
            "sv_lease_admitted_count": int(self.sv_lease_admitted_count),
            "sv_lease_rejected_count": int(self.sv_lease_rejected_count),
            "sv_lease_reject_reason_counts": dict(self.sv_lease_reject_reason_counts),
            "sv_continuation_candidate_count": int(self.sv_continuation_candidate_count),
            "sv_continuation_issued_count": int(self.sv_continuation_issued_count),
            "sv_continuation_board_change_count": int(
                self.sv_continuation_board_change_count
            ),
            "sv_continuation_score_delta_count": int(
                self.sv_continuation_score_delta_count
            ),
            "sv_continuation_level_delta_count": int(
                self.sv_continuation_level_delta_count
            ),
            "sv_continuation_constraint_progress_count": int(
                self.sv_continuation_constraint_progress_count
            ),
            "sv_aux_closer_candidate_count": int(self.sv_aux_closer_candidate_count),
            "sv_aux_closer_issued_count": int(self.sv_aux_closer_issued_count),
            "sv_exact_closed_count": int(self.stabilize_window_exact_closed_count),
            "sv_chain_closed_count": int(self.stabilize_window_chain_closed_count),
            "sv_expired_nochange_count": int(self.stabilize_window_expired_nochange_count),
            "sv_expired_budget_count": int(self.stabilize_window_expired_budget_count),
            "sv_expired_competing_progress_count": int(
                self.sv_expired_competing_progress_count
            ),
            "sv_preempted_count": int(self.stabilize_window_preempted_count),
            "stabilize_verify_one_shot_guard_count": int(
                self.stabilize_verify_one_shot_guard_count
            ),
            "goal_probe_starvation_guard_count": int(self.goal_probe_starvation_guard_count),
        }
        metrics.update(self._effect_ledger.summary())
        return metrics

    def episode_summary(self) -> dict[str, Any]:
        summary = {
            "rule_hypothesis_created_count": int(self.created_count),
            "rule_hypothesis_verified_count": int(self.verified_count),
            "rule_hypothesis_disconfirmed_count": int(self.disconfirmed_count),
            "rule_hypothesis_scoreless_churn_count": int(self.scoreless_churn_count),
            "rule_hypothesis_goal_relation_probe_count": int(self.goal_relation_probe_count),
            "rule_hypothesis_constraint_probe_count": int(self.constraint_probe_count),
            "rule_hypothesis_source_burst_count": int(self.source_burst_count),
            "rule_hypothesis_progress_lease_count": int(self.progress_lease_count),
            "rule_hypothesis_source_progress_suppressed_count": int(
                self.source_progress_suppressed_count
            ),
            "rule_hypothesis_perimeter_effect_suppressed_count": int(
                self.perimeter_effect_suppressed_count
            ),
            "rule_hypothesis_invariant_violation_count": int(self.invariant_violation_count),
            "rule_hypothesis_volatile_effect_suppressed_count": int(
                self.volatile_effect_suppressed_count
            ),
            "rule_hypothesis_volatile_component_count": int(len(self._volatile_components)),
            "rule_hypothesis_visual_stencil_candidate_count": int(
                self.visual_stencil_candidate_count
            ),
            "rule_hypothesis_visual_stencil_anchor_count": int(
                self.visual_stencil_anchor_count
            ),
            "rule_hypothesis_visual_stencil_attempt_count": int(
                self.visual_stencil_attempt_count
            ),
            "rule_hypothesis_visual_stencil_exhausted_count": int(
                self.visual_stencil_exhausted_count
            ),
            "rule_hypothesis_visual_stencil_progress_count": int(
                self.visual_stencil_progress_count
            ),
            "rule_hypothesis_visual_stencil_regress_count": int(
                self.visual_stencil_regress_count
            ),
            "rule_hypothesis_visual_stencil_neutral_count": int(
                self.visual_stencil_neutral_count
            ),
            "rule_hypothesis_visual_stencil_zero_violation_scoreless_count": int(
                self.visual_stencil_zero_violation_scoreless_count
            ),
            "rule_hypothesis_visual_stencil_zero_violation_after_progress_count": int(
                self.visual_stencil_zero_violation_after_progress_count
            ),
            "rule_hypothesis_visual_stencil_near_solved_step_count": int(
                self.visual_stencil_near_solved_step_count
            ),
            "rule_hypothesis_visual_stencil_near_solved_suppressed_count": int(
                self.visual_stencil_near_solved_suppressed_count
            ),
            "rule_hypothesis_visual_stencil_ledger_candidate_count": int(
                self.visual_stencil_ledger_candidate_count
            ),
            "rule_hypothesis_visual_constraint_observed_count": int(
                len(self._visual_constraint_ledger)
            ),
            "rule_hypothesis_visual_constraint_unresolved_count": int(
                sum(1 for item in self._visual_constraint_ledger.values() if item.status == "unresolved")
            ),
            "rule_hypothesis_visual_constraint_resolved_count": int(
                sum(1 for item in self._visual_constraint_ledger.values() if item.status == "resolved")
            ),
            "rule_hypothesis_visual_constraint_regress_count": int(
                sum(int(item.regress_count) for item in self._visual_constraint_ledger.values())
            ),
            "rule_hypothesis_visual_constraint_neutral_count": int(
                sum(int(item.neutral_count) for item in self._visual_constraint_ledger.values())
            ),
            "rule_hypothesis_visual_stencil_current_violation_count": int(
                self._last_visual_stencil_violation_count or 0
            ),
            "rule_hypothesis_visual_stencil_min_violation_count": int(
                self._min_visual_stencil_violation_count or 0
            ),
            "rule_hypothesis_effect_actuator_count": int(len(self._effect_actuators)),
            "rule_hypothesis_visual_repair_actuator_candidate_count": int(
                self.visual_repair_actuator_candidate_count
            ),
            "rule_hypothesis_goal_probe_candidate_count": int(self.goal_probe_candidate_count),
            "rule_hypothesis_goal_probe_issued_count": int(self.goal_probe_issued_count),
            "rule_hypothesis_goal_probe_success_count": int(self.goal_probe_success_count),
            "rule_hypothesis_goal_probe_board_change_success_count": int(
                self.goal_probe_board_change_success_count
            ),
            "rule_hypothesis_goal_probe_score_success_count": int(
                self.goal_probe_score_success_count
            ),
            "rule_hypothesis_goal_probe_level_success_count": int(
                self.goal_probe_level_success_count
            ),
            "rule_hypothesis_goal_probe_constraint_progress_success_count": int(
                self.goal_probe_constraint_progress_success_count
            ),
            "rule_hypothesis_rule_closed_success_count": int(self.rule_closed_success_count),
            "rule_hypothesis_rule_closed_score_count": int(self.rule_closed_score_count),
            "rule_hypothesis_rule_closed_level_count": int(self.rule_closed_level_count),
            "rule_hypothesis_rule_closed_constraint_count": int(self.rule_closed_constraint_count),
            "rule_hypothesis_rule_closed_synthetic_count": int(self.rule_closed_synthetic_count),
            "rule_hypothesis_closed_count": int(self.closed_count),
            "rule_hypothesis_actuator_group_detected_count": int(
                self.actuator_group_detected_count
            ),
            "rule_hypothesis_actuator_group_completed_count": int(
                self.actuator_group_completed_count
            ),
            "rule_hypothesis_followup_issued_count": int(self.followup_issued_count),
            "rule_hypothesis_followup_success_count": int(self.followup_success_count),
            "stabilize_verify_window_opened_count": int(
                self.stabilize_verify_window_opened_count
            ),
            "stabilize_verify_window_settled_count": int(
                self.stabilize_verify_window_settled_count
            ),
            "stabilize_verify_issued_count": int(self.stabilize_verify_issued_count),
            "stabilize_verify_closed_count": int(self.stabilize_verify_closed_count),
            "stabilize_verify_expired_count": int(self.stabilize_verify_expired_count),
            "stabilize_window_lease_started_count": int(
                self.stabilize_window_lease_started_count
            ),
            "stabilize_window_lease_followup_issued_count": int(
                self.stabilize_window_lease_followup_issued_count
            ),
            "stabilize_window_exact_closed_count": int(
                self.stabilize_window_exact_closed_count
            ),
            "stabilize_window_chain_closed_count": int(
                self.stabilize_window_chain_closed_count
            ),
            "stabilize_window_expired_nochange_count": int(
                self.stabilize_window_expired_nochange_count
            ),
            "stabilize_window_expired_budget_count": int(
                self.stabilize_window_expired_budget_count
            ),
            "stabilize_window_scoreless_verified_count": int(
                self.stabilize_window_scoreless_verified_count
            ),
            "stabilize_window_preempted_count": int(
                self.stabilize_window_preempted_count
            ),
            "sv_lease_admitted_count": int(self.sv_lease_admitted_count),
            "sv_lease_rejected_count": int(self.sv_lease_rejected_count),
            "sv_lease_reject_reason_counts": dict(self.sv_lease_reject_reason_counts),
            "sv_continuation_candidate_count": int(self.sv_continuation_candidate_count),
            "sv_continuation_issued_count": int(self.sv_continuation_issued_count),
            "sv_continuation_board_change_count": int(
                self.sv_continuation_board_change_count
            ),
            "sv_continuation_score_delta_count": int(
                self.sv_continuation_score_delta_count
            ),
            "sv_continuation_level_delta_count": int(
                self.sv_continuation_level_delta_count
            ),
            "sv_continuation_constraint_progress_count": int(
                self.sv_continuation_constraint_progress_count
            ),
            "sv_aux_closer_candidate_count": int(self.sv_aux_closer_candidate_count),
            "sv_aux_closer_issued_count": int(self.sv_aux_closer_issued_count),
            "sv_exact_closed_count": int(self.stabilize_window_exact_closed_count),
            "sv_chain_closed_count": int(self.stabilize_window_chain_closed_count),
            "sv_expired_nochange_count": int(self.stabilize_window_expired_nochange_count),
            "sv_expired_budget_count": int(self.stabilize_window_expired_budget_count),
            "sv_expired_competing_progress_count": int(
                self.sv_expired_competing_progress_count
            ),
            "sv_preempted_count": int(self.stabilize_window_preempted_count),
            "stabilize_verify_one_shot_guard_count": int(
                self.stabilize_verify_one_shot_guard_count
            ),
            "goal_probe_starvation_guard_count": int(self.goal_probe_starvation_guard_count),
            "rule_hypothesis_kind_histogram": dict(self.kind_histogram),
            "rule_hypothesis_source_histogram": dict(self.source_histogram),
            "rule_hypothesis_active_count": int(len(self._hypotheses)),
            "rule_hypothesis_last_event": dict(self.last_event),
            "rule_hypothesis_probe_event_count": int(len(self._probe_events)),
        }
        summary.update(self._effect_ledger.summary())
        return summary
