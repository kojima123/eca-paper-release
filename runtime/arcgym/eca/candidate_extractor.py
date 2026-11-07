"""CandidateExtractor: LLM-independent structured candidate pool builder.

Replaces the LLM-dependent candidate pool construction with a deterministic
extractor that reads from ECARuntime's existing surfaces:
  - last_successor_slate      : top-ranked successor candidates
  - last_protected_hedge_*    : protected hedge contract
  - structural_time_tracker   : recently-live anchor information
  - last_candidate_pool       : existing raw candidate pool

Output is a list of StructuredCandidate objects with:
  - component_id  : unique identifier
  - source        : where the candidate came from
  - family        : action family
  - base_score    : raw score from runtime
  - recent_live   : whether this candidate was recently live
  - is_hedge      : whether this is a protected hedge candidate
  - reject_reason : why this candidate was rejected (if any)

This module is observer-only in Phase 1: it builds and records the structured
pool but does not yet replace the existing candidate_pool in runtime.

Usage in runtime.py:
    self._candidate_extractor = CandidateExtractor()
    # In observe():
    base_pool = self._candidate_extractor.build_base_pool(self, step_index)
    self._last_base_pool = base_pool
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any


def _enable_l1_focus_retarget() -> bool:
    """Phase 8.C-1 flag: at L1 with score=0, allow object_escape to
    re-test components that previously produced a board change (i.e.,
    in `click_testing_stats["changed"]`). Default behavior excludes
    all tested components, which causes L1-stuck games to fan out
    across many unique components without ever revisiting a
    score-bearing candidate.

    Default off. Enable with `ENABLE_L1_FOCUS_RETARGET=1`.

    Background: 25-game zero-score breakdown found that scoring
    games (vc33, lp85) naturally repeat object_escape on a small
    set of components (top-5 ratio 76% for vc33), while L1-stuck
    games (sc25) test 15 unique components 1x each because
    object_escape excludes all tested components at
    `candidate_extractor.py:1091`. Reusing the "changed" set at L1
    restores focus / repetition.
    """
    raw = os.environ.get("ENABLE_L1_FOCUS_RETARGET", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _enable_l1_focus_retarget_global() -> bool:
    """Phase 8.C-1-global flag: when combined with
    `ENABLE_L1_FOCUS_RETARGET=1`, use the global "changed" set
    (aggregated across all state hashes via
    `runtime.last_changed_clicked_component_ids`) instead of the
    per-current-state `changed` set.

    Default off. Enable with `ENABLE_L1_FOCUS_RETARGET_GLOBAL=1`.

    Background: C-1 (per-state changed) was null-effect on sc25
    because state-evolving games lose their "changed" history when
    state advances. The global variant tracks "changed" across all
    state hashes, so a component that changed the board at any
    point remains re-eligible.
    """
    raw = os.environ.get("ENABLE_L1_FOCUS_RETARGET_GLOBAL", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# ─── Structured Candidate ─────────────────────────────────────────────────────

@dataclass
class StructuredCandidate:
    """A structured candidate with source metadata for LocalController use."""

    component_id: str = ""
    source: str = ""          # "successor_slate" | "protected_hedge" | "candidate_pool" | "object_table" | "transition_frontier" | "object_escape" | "aligned_object_sweep" | "level_rule_symbol_scout" | "grid_scan" | "local_click_fanout" | "local_reversible_click" | "rule_hypothesis" | "goal_probe" | "stabilize_verify"
    family: str = ""
    action_name: str = ""
    anchor_key: str = ""
    center_row: int = 0
    center_col: int = 0
    base_score: float = 0.0
    effect_bonus: float = 0.0
    recent_live: bool = False
    is_hedge: bool = False
    reject_reason: str = ""   # empty if not rejected
    shared_lineage: bool = False
    step_index: int = -1
    stage: str = ""
    closure_status: str = ""
    hypothesis_id: str = ""
    hypothesis_kind: str = ""
    linked_effect_id: str = ""
    window_id: str = ""
    continuation_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "source": self.source,
            "family": self.family,
            "action_name": self.action_name,
            "anchor_key": self.anchor_key,
            "center_row": self.center_row,
            "center_col": self.center_col,
            "base_score": round(self.base_score, 4),
            "effect_bonus": round(self.effect_bonus, 4),
            "recent_live": self.recent_live,
            "is_hedge": self.is_hedge,
            "reject_reason": self.reject_reason,
            "shared_lineage": self.shared_lineage,
            "step_index": self.step_index,
            "stage": self.stage,
            "closure_status": self.closure_status,
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_kind": self.hypothesis_kind,
            "linked_effect_id": self.linked_effect_id,
            "window_id": self.window_id,
            "continuation_kind": self.continuation_kind,
        }


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class CandidateExtractorConfig:
    """Configuration for CandidateExtractor."""

    # Maximum candidates to extract from each source
    max_from_successor_slate: int = 4
    max_from_candidate_pool: int = 8
    max_from_object_table: int = 6
    max_from_grid_scan: int = 4
    max_from_reversible_click: int = 2
    max_from_rule_hypothesis: int = 4
    max_from_aligned_object_sweep: int = 8
    max_total: int = 12

    # Score threshold below which candidates are considered low-quality
    min_base_score: float = 0.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "CandidateExtractorConfig":
        if not raw:
            return cls()
        return cls(
            max_from_successor_slate=int(raw.get("max_from_successor_slate", 4)),
            max_from_candidate_pool=int(raw.get("max_from_candidate_pool", 8)),
            max_from_object_table=int(raw.get("max_from_object_table", 6)),
            max_from_grid_scan=int(raw.get("max_from_grid_scan", 4)),
            max_from_reversible_click=int(raw.get("max_from_reversible_click", 2)),
            max_from_rule_hypothesis=int(raw.get("max_from_rule_hypothesis", 4)),
            max_from_aligned_object_sweep=int(raw.get("max_from_aligned_object_sweep", 8)),
            max_total=int(raw.get("max_total", 12)),
            min_base_score=float(raw.get("min_base_score", 0.0)),
        )


# ─── Extraction Result ────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Result of a single CandidateExtractor.build_base_pool() call."""

    candidates: list[StructuredCandidate] = field(default_factory=list)
    step_index: int = -1
    source_counts: dict[str, int] = field(default_factory=dict)
    family_counts: dict[str, int] = field(default_factory=dict)
    recently_live_family: str = ""
    hedge_family: str = ""
    grid_scan_pressure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "total": len(self.candidates),
            "source_counts": self.source_counts,
            "family_counts": self.family_counts,
            "recently_live_family": self.recently_live_family,
            "hedge_family": self.hedge_family,
            "grid_scan_pressure_reason": self.grid_scan_pressure_reason,
            "candidates": [c.to_dict() for c in self.candidates],
        }


# ─── Extractor ───────────────────────────────────────────────────────────────

class CandidateExtractor:
    """Builds a structured candidate pool from ECARuntime surfaces.

    This extractor is LLM-independent: it reads from runtime's existing
    attributes and translates them into StructuredCandidate objects.
    """

    def __init__(self, config: CandidateExtractorConfig | None = None) -> None:
        self.config = config or CandidateExtractorConfig()
        self._history: list[ExtractionResult] = []
        self._last_grid_scan_pressure_reason = ""

    def reset(self) -> None:
        """Reset for a new episode."""
        self._history.clear()
        self._last_grid_scan_pressure_reason = ""

    @property
    def history(self) -> list[ExtractionResult]:
        return self._history

    def _extract_family(self, entry: dict[str, Any]) -> str:
        return str(
            entry.get("transition_model_candidate_family", "")
            or entry.get("transition_model_lookup_candidate_family", "")
            or entry.get("preferred_action_family", "")
            or entry.get("rule_family", "")
            or entry.get("family", "")
            or entry.get("action_family", "")
            or ""
        )

    @staticmethod
    def _component_center(entry: dict[str, Any]) -> tuple[int, int]:
        center = dict(entry.get("center") or {})
        if center:
            return (
                int(center.get("row", 0) or 0),
                int(center.get("col", 0) or 0),
            )
        bbox = dict(entry.get("bbox") or {})
        min_row = int(bbox.get("min_row", 0) or 0)
        max_row = int(bbox.get("max_row", min_row) or min_row)
        min_col = int(bbox.get("min_col", 0) or 0)
        max_col = int(bbox.get("max_col", min_col) or min_col)
        return ((min_row + max_row) // 2, (min_col + max_col) // 2)

    @classmethod
    def _anchor_key(cls, *, level_number: int, component: dict[str, Any]) -> str:
        row, col = cls._component_center(component)
        return f"{int(level_number)}||{row}||{col}"

    @classmethod
    def _object_table_bootstrap_score(
        cls,
        *,
        component: dict[str, Any],
        board_rows: list[str],
        click_testing_stats: dict[str, set[str]],
    ) -> float:
        size = max(1, int(component.get("size", 0) or 0))
        bbox = dict(component.get("bbox") or {})
        min_row = int(bbox.get("min_row", 0) or 0)
        max_row = int(bbox.get("max_row", min_row) or min_row)
        min_col = int(bbox.get("min_col", 0) or 0)
        max_col = int(bbox.get("max_col", min_col) or min_col)
        bbox_area = max(1, (max_row - min_row + 1) * (max_col - min_col + 1))
        compactness = min(1.0, float(size) / float(bbox_area))
        row, col = cls._component_center(component)
        rows = len(board_rows)
        cols = max((len(str(item or "")) for item in board_rows), default=0)
        if rows <= 0:
            rows = max_row + 1
        if cols <= 0:
            cols = max_col + 1
        if rows <= 0 or cols <= 0:
            centrality = 0.0
        else:
            center_row = max(0.0, (float(rows) - 1.0) / 2.0)
            center_col = max(0.0, (float(cols) - 1.0) / 2.0)
            row_span = max(1.0, center_row if center_row > 0 else float(rows))
            col_span = max(1.0, center_col if center_col > 0 else float(cols))
            row_norm = abs(float(row) - center_row) / row_span
            col_norm = abs(float(col) - center_col) / col_span
            centrality = max(0.0, 1.0 - ((row_norm + col_norm) / 2.0))
        size_score = min(0.45, 0.06 * float(size))
        component_id = str(component.get("component_id", "") or "").strip().lower()
        tested_components = set(click_testing_stats.get("tested", set()))
        changed_components = set(click_testing_stats.get("changed", set()))
        no_change_components = set(click_testing_stats.get("no_change", set()))
        untested_bonus = 0.18 if component_id and component_id not in tested_components else 0.0
        tested_penalty = 0.20 if component_id and component_id in tested_components else 0.0
        changed_penalty = 0.06 if component_id and component_id in changed_components else 0.0
        no_change_penalty = 0.22 if component_id and component_id in no_change_components else 0.0
        score = 0.12 + size_score + 0.18 * compactness + 0.12 * centrality
        score += untested_bonus
        score -= tested_penalty + changed_penalty + no_change_penalty
        return round(max(0.01, score), 4)

    @staticmethod
    def _board_rows(runtime: Any) -> list[str]:
        rows = getattr(runtime, "last_board_rows", None)
        if isinstance(rows, list):
            return [str(row or "") for row in rows]
        return []

    @staticmethod
    def _level_number(runtime: Any) -> int:
        last_state = getattr(runtime, "last_state", None)
        for raw in (
            getattr(last_state, "level_number", None),
            getattr(runtime, "last_level_number", None),
        ):
            try:
                if raw is not None:
                    return int(raw)
            except (TypeError, ValueError):
                continue
        return 0

    @staticmethod
    def _available_action_names(runtime: Any) -> set[str]:
        raw_actions = getattr(runtime, "last_available_actions", None)
        if not isinstance(raw_actions, (list, tuple, set)):
            return set()
        names: set[str] = set()
        for raw_action in raw_actions:
            action = str(raw_action or "").strip().upper()
            if not action:
                continue
            if action.startswith("ACTION") and action[6:].isdigit():
                names.add(action)
            elif action.isdigit():
                names.add(f"ACTION{int(action)}")
        return names

    @classmethod
    def _available_candidate_families(cls, runtime: Any) -> set[str]:
        actions = cls._available_action_names(runtime)
        if not actions:
            return set()
        families: set[str] = set()
        if any(action in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"} for action in actions):
            families.add("move")
        if "ACTION6" in actions:
            families.add("click")
        if any(
            action.startswith("ACTION")
            and action not in {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6", "RESET"}
            for action in actions
        ):
            families.add("auxiliary")
        return families

    @classmethod
    def _candidate_family_allowed(cls, runtime: Any, family: str) -> bool:
        normalized = str(family or "").strip().lower()
        if not normalized:
            return True
        available = cls._available_candidate_families(runtime)
        return not available or normalized in available

    @classmethod
    def _filter_candidate_families_for_available_actions(
        cls,
        runtime: Any,
        families: list[str],
    ) -> list[str]:
        available = cls._available_candidate_families(runtime)
        if not available:
            return list(families)
        return [
            family
            for family in families
            if str(family or "").strip().lower() in available
        ]

    @staticmethod
    def _step_index(runtime: Any, *, fallback: int = 0) -> int:
        last_state = getattr(runtime, "last_state", None)
        raw = getattr(last_state, "step_index", None)
        if isinstance(raw, (int, float)):
            return int(raw)
        if isinstance(raw, str) and raw.strip():
            try:
                return int(float(raw.strip()))
            except (TypeError, ValueError):
                pass
        return int(fallback)

    @staticmethod
    def _has_viable_candidates(candidates: list[StructuredCandidate]) -> bool:
        return any(not str(candidate.reject_reason or "").strip() for candidate in candidates)

    @staticmethod
    def _click_component_testing_stats(runtime: Any) -> dict[str, set[str]]:
        tested: set[str] = set()
        changed: set[str] = set()
        no_change: set[str] = set()
        changed_global: set[str] = set()
        globally_clicked = getattr(runtime, "last_clicked_component_ids", None)
        if isinstance(globally_clicked, (set, list, tuple)):
            for component_id in globally_clicked:
                normalized = str(component_id or "").strip().lower()
                if normalized:
                    tested.add(normalized)
        # Phase 8.C-1-global: pull cross-state "changed" set from runtime
        globally_changed = getattr(runtime, "last_changed_clicked_component_ids", None)
        if isinstance(globally_changed, (set, list, tuple)):
            for component_id in globally_changed:
                normalized = str(component_id or "").strip().lower()
                if normalized:
                    changed_global.add(normalized)
        current_state_actions = getattr(getattr(runtime, "last_state", None), "current_state_actions", None)
        if not isinstance(current_state_actions, dict):
            return {
                "tested": tested,
                "changed": changed,
                "no_change": no_change,
                "changed_global": changed_global,
            }
        for action_key, record in current_state_actions.items():
            normalized_action_key = str(action_key or "").strip().lower()
            if not normalized_action_key.startswith("click_"):
                continue
            component_id = normalized_action_key.split("click_", 1)[1].strip().lower()
            if not component_id:
                continue
            tested.add(component_id)
            if bool((record or {}).get("changed")):
                changed.add(component_id)
            else:
                no_change.add(component_id)
        return {
            "tested": tested,
            "changed": changed,
            "no_change": no_change,
            "changed_global": changed_global,
        }

    @staticmethod
    def _selected_target_bootstrap_preference(runtime: Any) -> dict[str, Any]:
        selected_target_summary = dict(
            getattr(runtime, "last_selected_target_summary", None) or {}
        )
        component_id = str(
            selected_target_summary.get("component_id", "")
            or selected_target_summary.get("candidate_id", "")
            or ""
        ).strip()
        transition_family = str(
            selected_target_summary.get("transition_model_candidate_family", "")
            or selected_target_summary.get("action_family", "")
            or selected_target_summary.get("family", "")
            or ""
        ).strip().lower()
        health_result = getattr(runtime, "_last_target_health_result", None)
        raw_health = getattr(health_result, "health", "")
        if hasattr(raw_health, "value"):
            raw_health = getattr(raw_health, "value", "")
        health = str(raw_health or "").strip().lower()
        return {
            "component_id": component_id,
            "transition_family": transition_family,
            "health": health,
            "prefer_move_after_click_stale": bool(
                component_id
                and transition_family == "click"
                and health.endswith("stale")
            ),
        }

    @staticmethod
    def _object_table_candidate_families(
        *,
        component_id: str,
        click_testing_stats: dict[str, set[str]],
        preferred_component_id: str = "",
        prefer_move_after_click_stale: bool = False,
    ) -> list[str]:
        tested_components = set(click_testing_stats.get("tested", set()))
        no_change_components = set(click_testing_stats.get("no_change", set()))
        normalized_component_id = str(component_id or "").strip().lower()
        if not normalized_component_id:
            return ["click"]
        if (
            prefer_move_after_click_stale
            and normalized_component_id == str(preferred_component_id or "").strip().lower()
        ):
            return ["move", "click"]
        if normalized_component_id in no_change_components:
            return ["move", "click"]
        if normalized_component_id in tested_components:
            return ["move", "click"]
        return ["click"]

    @staticmethod
    def _bootstrap_focus_components(runtime: Any) -> list[dict[str, Any]]:
        raw = getattr(runtime, "last_bootstrap_focus_components", None)
        if not isinstance(raw, list):
            return []
        focus_components: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            component_id = str(entry.get("component_id", "") or "").strip()
            if not component_id or component_id in seen_ids:
                continue
            seen_ids.add(component_id)
            focus_components.append(dict(entry))
        return focus_components

    @staticmethod
    def _transition_frontier_components(runtime: Any, *, step_index: int) -> list[dict[str, Any]]:
        raw = getattr(runtime, "local_transition_frontier", None)
        if not isinstance(raw, list):
            return []
        entries: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            expires_at = int(item.get("expires_at", -10_000) or -10_000)
            if expires_at < int(step_index):
                continue
            component_id = str(item.get("component_id", "") or "").strip()
            if not component_id or component_id in seen_ids:
                continue
            seen_ids.add(component_id)
            entries.append(dict(item))
        return entries

    @staticmethod
    def _click_grid_scan_pressure(runtime: Any, *, step_index: int | None = None) -> bool:
        state = getattr(runtime, "last_state", None)
        step_index = CandidateExtractor._step_index(
            runtime,
            fallback=0 if step_index is None else int(step_index),
        )
        try:
            score = float(getattr(state, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        active_frontier = [
            item
            for item in list(getattr(runtime, "local_transition_frontier", None) or [])
            if isinstance(item, dict)
            and int(item.get("expires_at", -10_000) or -10_000) >= step_index
        ]
        if step_index >= 6 and score <= 0.0 and not active_frontier:
            return True
        if CandidateExtractor._level_rule_discovery_pressure_reason(
            runtime,
            step_index=step_index,
            active_frontier=active_frontier,
        ):
            return True
        same_family_streak = int(getattr(runtime, "_same_family_no_score_streak", 0) or 0)
        if same_family_streak >= 6:
            return True
        rows = getattr(runtime, "step_metrics", None)
        if isinstance(rows, list) and rows:
            last_row = dict(rows[-1] or {})
            last_family = str(
                last_row.get("local_effect_selected_family", "")
                or last_row.get("local_controller_selected_action_family", "")
                or ""
            ).strip().lower()
            last_probe_count = int(last_row.get("local_effect_family_probe_count", 0) or 0)
            last_score = float(last_row.get("local_effect_family_score", 0.0) or 0.0)
            last_score_change_count = int(last_row.get("local_effect_score_change_count", 0) or 0)
            if last_family == "click" and last_probe_count >= 4 and last_score <= -1.0 and last_score_change_count <= 0:
                return True
        event = getattr(runtime, "last_local_effect_family_event", None)
        if isinstance(event, dict):
            event_family = str(event.get("family", "") or "").strip().lower()
            event_probe_count = int(event.get("probe_count", 0) or 0)
            event_score = float(event.get("score", 0.0) or 0.0)
            event_score_change_count = int(event.get("score_change_count", 0) or 0)
            if (
                event_family == "click"
                and event_probe_count >= 4
                and event_score <= -1.0
                and event_score_change_count <= 0
            ):
                return True
        ledger = getattr(runtime, "local_effect_family_ledger", None)
        if not isinstance(ledger, dict):
            return False
        click_entry = dict(ledger.get("click") or {})
        if not click_entry:
            return False
        score_change_count = int(click_entry.get("score_change_count", 0) or 0)
        if score_change_count > 0:
            return False
        no_change_count = int(click_entry.get("no_change_count", 0) or 0)
        weak_count = int(click_entry.get("weak_effect_count", 0) or 0)
        probe_count = int(click_entry.get("probe_count", 0) or 0)
        score = float(click_entry.get("score", 0.0) or 0.0)
        if probe_count >= 4 and score <= -1.0:
            return True
        return bool(probe_count >= 4 and no_change_count + weak_count >= 3)

    @staticmethod
    def _delay_l1_grid_for_object_breadth(runtime: Any, *, step_index: int) -> bool:
        if CandidateExtractor._level_number(runtime) != 1:
            return False
        if int(step_index) >= 32:
            return False
        full_object_table = list(getattr(getattr(runtime, "last_state", None), "object_table", None) or [])
        if len(full_object_table) < 8:
            return False
        rows = [
            dict(row or {})
            for row in list(getattr(runtime, "step_metrics", None) or [])
            if isinstance(row, dict)
        ]
        object_table_selected = sum(
            1
            for row in rows
            if int(row.get("level", row.get("level_number", 0)) or 0) == 1
            and str(row.get("selected_target_source", "") or "").strip() == "object_table"
        )
        return bool(object_table_selected < 24)

    @staticmethod
    def _level_rule_discovery_pressure_reason(
        runtime: Any,
        *,
        step_index: int,
        active_frontier: list[dict[str, Any]] | None = None,
    ) -> str:
        if CandidateExtractor._level_number(runtime) < 2:
            return ""
        if int(step_index) < 8:
            return ""
        if not CandidateExtractor._board_rows(runtime):
            return ""

        state = getattr(runtime, "last_state", None)
        try:
            score_delta = float(getattr(state, "score_delta", 0.0) or 0.0)
        except (TypeError, ValueError):
            score_delta = 0.0
        if score_delta > 0.0:
            return ""

        raw_pool = getattr(runtime, "last_candidate_pool", None)
        if not raw_pool:
            return "level_rule_discovery_empty_candidate_pool"

        health_result = getattr(runtime, "_last_target_health_result", None)
        raw_health = getattr(health_result, "health", "") if health_result is not None else ""
        if hasattr(raw_health, "value"):
            raw_health = getattr(raw_health, "value", "")
        health = str(raw_health or "").strip().lower()
        if health in {"empty", "stale", "saturated", "blocked"}:
            return f"level_rule_discovery_target_{health}"

        same_family_streak = int(getattr(runtime, "_same_family_no_score_streak", 0) or 0)
        if same_family_streak >= 4:
            return "level_rule_discovery_same_family_stall"

        try:
            no_change_streak = int(getattr(state, "no_change_streak", 0) or 0)
        except (TypeError, ValueError):
            no_change_streak = 0
        try:
            repeated_state_count = int(getattr(state, "repeated_state_count", 0) or 0)
        except (TypeError, ValueError):
            repeated_state_count = 0
        if max(no_change_streak, repeated_state_count) >= 4:
            return "level_rule_discovery_state_stall"

        active = list(active_frontier or [])
        if not active:
            return "level_rule_discovery_no_frontier"
        saturated = 0
        for item in active:
            status = str(
                item.get("status", "")
                or item.get("state", "")
                or item.get("frontier_status", "")
                or item.get("reason", "")
                or ""
            ).strip().lower()
            if any(token in status for token in ("saturat", "stale", "exhaust", "dead", "blocked")):
                saturated += 1
        if active and saturated >= len(active):
            return "level_rule_discovery_saturated_frontier"
        return ""

    @staticmethod
    def _click_grid_scan_pressure_reason(runtime: Any, *, step_index: int | None = None) -> str:
        state = getattr(runtime, "last_state", None)
        step_index = CandidateExtractor._step_index(
            runtime,
            fallback=0 if step_index is None else int(step_index),
        )
        try:
            score = float(getattr(state, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        active_frontier = [
            item
            for item in list(getattr(runtime, "local_transition_frontier", None) or [])
            if isinstance(item, dict)
            and int(item.get("expires_at", -10_000) or -10_000) >= step_index
        ]
        if step_index >= 6 and score <= 0.0 and not active_frontier:
            return "zero_score_no_frontier"
        level_reason = CandidateExtractor._level_rule_discovery_pressure_reason(
            runtime,
            step_index=step_index,
            active_frontier=active_frontier,
        )
        if level_reason:
            return level_reason
        same_family_streak = int(getattr(runtime, "_same_family_no_score_streak", 0) or 0)
        if same_family_streak >= 6:
            return "same_family_no_score_streak"
        rows = getattr(runtime, "step_metrics", None)
        if isinstance(rows, list) and rows:
            last_row = dict(rows[-1] or {})
            last_family = str(
                last_row.get("local_effect_selected_family", "")
                or last_row.get("local_controller_selected_action_family", "")
                or ""
            ).strip().lower()
            last_probe_count = int(last_row.get("local_effect_family_probe_count", 0) or 0)
            last_score = float(last_row.get("local_effect_family_score", 0.0) or 0.0)
            last_score_change_count = int(last_row.get("local_effect_score_change_count", 0) or 0)
            if last_family == "click" and last_probe_count >= 4 and last_score <= -1.0 and last_score_change_count <= 0:
                return "previous_step_click_family_negative"
        event = getattr(runtime, "last_local_effect_family_event", None)
        if isinstance(event, dict):
            event_family = str(event.get("family", "") or "").strip().lower()
            event_probe_count = int(event.get("probe_count", 0) or 0)
            event_score = float(event.get("score", 0.0) or 0.0)
            event_score_change_count = int(event.get("score_change_count", 0) or 0)
            if (
                event_family == "click"
                and event_probe_count >= 4
                and event_score <= -1.0
                and event_score_change_count <= 0
            ):
                return "last_click_family_event_negative"
        ledger = getattr(runtime, "local_effect_family_ledger", None)
        if isinstance(ledger, dict):
            click_entry = dict(ledger.get("click") or {})
            score_change_count = int(click_entry.get("score_change_count", 0) or 0)
            no_change_count = int(click_entry.get("no_change_count", 0) or 0)
            weak_count = int(click_entry.get("weak_effect_count", 0) or 0)
            probe_count = int(click_entry.get("probe_count", 0) or 0)
            ledger_score = float(click_entry.get("score", 0.0) or 0.0)
            if score_change_count <= 0 and probe_count >= 4 and ledger_score <= -1.0:
                return "ledger_click_family_negative"
            if score_change_count <= 0 and probe_count >= 4 and no_change_count + weak_count >= 3:
                return "ledger_click_family_stuck"
        return ""

    @staticmethod
    def _grid_scan_base_score(*, pressure_reason: str, step_index: int) -> float:
        reason = str(pressure_reason or "").strip()
        if reason.startswith("level_rule_discovery"):
            return 2.72
        return 1.18

    def _grid_scan_candidates(
        self,
        runtime: Any,
        *,
        step_index: int,
        existing_keys: set[tuple[str, str]],
    ) -> list[StructuredCandidate]:
        pressure_reason = self._click_grid_scan_pressure_reason(
            runtime,
            step_index=step_index,
        )
        self._last_grid_scan_pressure_reason = pressure_reason
        if not pressure_reason:
            return []
        rows = CandidateExtractor._board_rows(runtime)
        if not rows:
            return []
        row_count = len(rows)
        col_count = max((len(row) for row in rows), default=0)
        if row_count <= 0 or col_count <= 0:
            return []
        frequencies: dict[str, int] = {}
        total = 0
        for row in rows:
            for char in str(row or ""):
                frequencies[char] = int(frequencies.get(char, 0) or 0) + 1
                total += 1
        if not frequencies or total <= 0:
            return []
        dominant = max(frequencies.items(), key=lambda item: item[1])[0]
        stride = max(3, min(8, max(3, min(row_count, col_count) // 12)))
        start = max(0, stride // 2)
        scored: list[tuple[float, int, int, str, float]] = []
        dominant_fallback_scored: list[tuple[float, int, int, str, float]] = []
        for row in range(start, row_count, stride):
            line = rows[row]
            if not line:
                continue
            for col in range(start, min(col_count, len(line)), stride):
                value = line[col]
                neighbor_values: set[str] = set()
                for dr in (-1, 0, 1):
                    nr = row + dr
                    if nr < 0 or nr >= row_count:
                        continue
                    nline = rows[nr]
                    for dc in (-1, 0, 1):
                        nc = col + dc
                        if nc < 0 or nc >= len(nline):
                            continue
                        neighbor_values.add(nline[nc])
                rarity = 1.0 - (float(frequencies.get(value, 0) or 0) / float(total))
                contrast = min(1.0, max(0.0, (len(neighbor_values) - 1) / 4.0))
                center_row = max(0.0, (float(row_count) - 1.0) / 2.0)
                center_col = max(0.0, (float(col_count) - 1.0) / 2.0)
                centrality = 0.0
                if center_row > 0 and center_col > 0:
                    centrality = max(
                        0.0,
                        1.0
                        - (
                            abs(float(row) - center_row) / max(1.0, center_row)
                            + abs(float(col) - center_col) / max(1.0, center_col)
                        )
                        / 2.0,
                    )
                component_id = f"grid_scan_{row}_{col}"
                anchor_key = f"grid||{row}||{col}"
                effect_bonus = self._grid_scan_effect_bonus(
                    runtime,
                    component_id=component_id,
                    anchor_key=anchor_key,
                )
                base = self._grid_scan_base_score(
                    pressure_reason=pressure_reason,
                    step_index=step_index,
                )
                if value == dominant:
                    if pressure_reason.startswith("level_rule_discovery"):
                        score = base - 0.10 + 0.10 * contrast + 0.04 * centrality
                        dominant_fallback_scored.append(
                            (round(score + effect_bonus, 4), row, col, value, effect_bonus)
                        )
                    continue
                score = base + 0.28 * rarity + 0.18 * contrast + 0.06 * centrality
                scored.append((round(score + effect_bonus, 4), row, col, value, effect_bonus))
        if not scored and pressure_reason.startswith("level_rule_discovery"):
            scored = dominant_fallback_scored
        if not scored:
            return []
        scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        candidates: list[StructuredCandidate] = []
        seen_points: set[tuple[int, int]] = set()
        for score, row, col, value, effect_bonus in scored:
            if len(candidates) >= max(1, int(self.config.max_from_grid_scan)):
                break
            component_id = f"grid_scan_{row}_{col}"
            key = (component_id, "click")
            if key in existing_keys or (row, col) in seen_points:
                continue
            seen_points.add((row, col))
            existing_keys.add(key)
            candidates.append(
                StructuredCandidate(
                    component_id=component_id,
                    source="grid_scan",
                    family="click",
                    anchor_key=f"grid||{row}||{col}",
                    center_row=int(row),
                    center_col=int(col),
                    base_score=float(score),
                    effect_bonus=float(effect_bonus),
                    stage=str(pressure_reason or ""),
                    step_index=step_index,
                )
            )
        return candidates

    def _level_rule_symbol_scout_candidates(
        self,
        runtime: Any,
        *,
        step_index: int,
        existing_keys: set[tuple[str, str]],
    ) -> list[StructuredCandidate]:
        pressure_reason = self._click_grid_scan_pressure_reason(
            runtime,
            step_index=step_index,
        )
        if not pressure_reason.startswith("level_rule_discovery"):
            return []
        rows = CandidateExtractor._board_rows(runtime)
        if not rows:
            return []
        row_count = len(rows)
        col_count = max((len(row) for row in rows), default=0)
        board_area = max(1, row_count * max(1, col_count))
        if row_count <= 0 or col_count <= 0:
            return []

        frequencies: dict[str, int] = {}
        total = 0
        for row in rows:
            for value in str(row or ""):
                frequencies[value] = int(frequencies.get(value, 0) or 0) + 1
                total += 1
        if not frequencies or total <= 0:
            return []
        dominant = max(frequencies.items(), key=lambda item: item[1])[0]

        recent_positions = (
            CandidateExtractor._recent_action6_positions(runtime, limit=24)
            + CandidateExtractor._recent_selected_anchor_positions(runtime, limit=96)
        )
        click_testing_stats = self._click_component_testing_stats(runtime)
        no_change_components = set(click_testing_stats.get("no_change", set()))
        changed_components = set(click_testing_stats.get("changed", set()))

        visited: set[tuple[int, int]] = set()
        scored: list[tuple[float, int, int, str, float]] = []
        max_char_frequency = max(12, int(round(float(total) * 0.05)))
        max_component_size = max(1, min(64, int(round(float(board_area) * 0.04))))
        max_bbox_area = max(1, min(96, int(round(float(board_area) * 0.08))))

        for row_index, row_text in enumerate(rows):
            for col_index, value in enumerate(row_text):
                point = (row_index, col_index)
                if point in visited:
                    continue
                visited.add(point)
                if not value or value.isspace() or value == dominant:
                    continue

                stack = [point]
                cells: list[tuple[int, int]] = []
                while stack:
                    current_row, current_col = stack.pop()
                    if current_row < 0 or current_row >= row_count:
                        continue
                    current_line = rows[current_row]
                    if current_col < 0 or current_col >= len(current_line):
                        continue
                    if current_line[current_col] != value:
                        continue
                    cells.append((current_row, current_col))
                    for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        next_point = (current_row + delta_row, current_col + delta_col)
                        if next_point in visited:
                            continue
                        next_row, next_col = next_point
                        if next_row < 0 or next_row >= row_count:
                            continue
                        next_line = rows[next_row]
                        if next_col < 0 or next_col >= len(next_line):
                            continue
                        if next_line[next_col] != value:
                            continue
                        visited.add(next_point)
                        stack.append(next_point)

                size = len(cells)
                if size <= 0:
                    continue
                char_frequency = int(frequencies.get(value, 0) or 0)
                if char_frequency > max_char_frequency and size > 2:
                    continue
                if size > max_component_size:
                    continue
                min_row = min(row for row, _ in cells)
                max_row = max(row for row, _ in cells)
                min_col = min(col for _, col in cells)
                max_col = max(col for _, col in cells)
                bbox_area = max(1, (max_row - min_row + 1) * (max_col - min_col + 1))
                if bbox_area > max_bbox_area:
                    continue
                center_row = int(round(sum(row for row, _ in cells) / float(size)))
                center_col = int(round(sum(col for _, col in cells) / float(size)))
                if any(
                    abs(center_row - recent_row) + abs(center_col - recent_col) <= 1
                    for recent_row, recent_col in recent_positions
                ):
                    continue

                neighbor_values: set[str] = set()
                for cell_row, cell_col in cells:
                    for delta_row in (-1, 0, 1):
                        next_row = cell_row + delta_row
                        if next_row < 0 or next_row >= row_count:
                            continue
                        next_line = rows[next_row]
                        for delta_col in (-1, 0, 1):
                            next_col = cell_col + delta_col
                            if next_col < 0 or next_col >= len(next_line):
                                continue
                            neighbor_values.add(next_line[next_col])
                compactness = min(1.0, float(size) / float(bbox_area))
                rarity = max(0.0, 1.0 - (float(char_frequency) / float(total)))
                scarcity = min(1.0, 8.0 / float(max(1, char_frequency)))
                size_focus = max(0.0, 1.0 - (float(size) / 24.0))
                contrast = min(1.0, max(0.0, (len(neighbor_values) - 1) / 5.0))
                center_board_row = max(0.0, (float(row_count) - 1.0) / 2.0)
                center_board_col = max(0.0, (float(col_count) - 1.0) / 2.0)
                centrality = 0.0
                if center_board_row > 0 and center_board_col > 0:
                    centrality = max(
                        0.0,
                        1.0
                        - (
                            abs(float(center_row) - center_board_row) / max(1.0, center_board_row)
                            + abs(float(center_col) - center_board_col) / max(1.0, center_board_col)
                        )
                        / 2.0,
                    )
                component_id = f"level_rule_symbol_scout_{center_row}_{center_col}_{ord(value[0])}"
                anchor_key = f"grid||{center_row}||{center_col}"
                effect_bonus = self._grid_scan_effect_bonus(
                    runtime,
                    component_id=component_id,
                    anchor_key=anchor_key,
                )
                normalized_component = component_id.lower()
                no_change_penalty = 0.22 if normalized_component in no_change_components else 0.0
                changed_penalty = 0.05 if normalized_component in changed_components else 0.0
                score = (
                    3.35
                    + 0.42 * scarcity
                    + 0.20 * rarity
                    + 0.18 * compactness
                    + 0.12 * size_focus
                    + 0.10 * contrast
                    + 0.05 * centrality
                    + effect_bonus
                    - no_change_penalty
                    - changed_penalty
                )
                scored.append((round(score, 4), center_row, center_col, value, effect_bonus))

        if not scored:
            return []
        scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        candidates: list[StructuredCandidate] = []
        seen_points: set[tuple[int, int]] = set()
        for score, row, col, value, effect_bonus in scored:
            if len(candidates) >= max(1, min(2, int(self.config.max_from_grid_scan or 1))):
                break
            component_id = f"level_rule_symbol_scout_{row}_{col}_{ord(value[0])}"
            key = (component_id, "click")
            if key in existing_keys or (row, col) in seen_points:
                continue
            seen_points.add((row, col))
            existing_keys.add(key)
            candidates.append(
                StructuredCandidate(
                    component_id=component_id,
                    source="level_rule_symbol_scout",
                    family="click",
                    anchor_key=f"grid||{row}||{col}",
                    center_row=int(row),
                    center_col=int(col),
                    base_score=float(score),
                    effect_bonus=float(effect_bonus),
                    stage=str(pressure_reason or ""),
                    step_index=step_index,
                )
            )
        return candidates

    @staticmethod
    def _object_escape_pressure(runtime: Any) -> bool:
        ledger = getattr(runtime, "local_effect_family_ledger", None)
        click_entry: dict[str, Any] = {}
        if isinstance(ledger, dict):
            click_entry = dict(ledger.get("click") or {})
        if not click_entry:
            event = getattr(runtime, "last_local_effect_family_event", None)
            if isinstance(event, dict) and str(event.get("family", "") or "").strip().lower() == "click":
                click_entry = dict(event)
        if not click_entry:
            return False
        score_change_count = int(click_entry.get("score_change_count", 0) or 0)
        if score_change_count > 0:
            return False
        probe_count = int(click_entry.get("probe_count", 0) or 0)
        weak_count = int(click_entry.get("weak_effect_count", 0) or 0)
        novel_count = int(click_entry.get("novel_hash_only_count", 0) or 0)
        repeated_count = int(click_entry.get("repeated_effect_count", 0) or 0)
        board_change_count = int(click_entry.get("board_change_no_score_count", 0) or 0)
        if probe_count < 6 or board_change_count < 4:
            return False
        return bool(weak_count + novel_count + repeated_count >= 3)

    @staticmethod
    def _compact_object_for_escape(component: dict[str, Any], *, row_count: int, col_count: int) -> bool:
        bbox = dict(component.get("bbox") or {})
        min_row = int(bbox.get("min_row", 0) or 0)
        max_row = int(bbox.get("max_row", min_row) or min_row)
        min_col = int(bbox.get("min_col", 0) or 0)
        max_col = int(bbox.get("max_col", min_col) or min_col)
        bbox_area = max(1, (max_row - min_row + 1) * (max_col - min_col + 1))
        board_area = max(1, int(row_count or 0) * int(col_count or 0))
        size = max(1, int(component.get("size", 0) or 0))
        compactness = min(1.0, float(size) / float(bbox_area))
        if max_row == min_row == 0:
            return False
        if bbox_area > 0.35 * float(board_area):
            return False
        if size > 0.25 * float(board_area):
            return False
        return bool(compactness >= 0.20)

    def _object_escape_candidates(
        self,
        runtime: Any,
        *,
        step_index: int,
        existing_keys: set[tuple[str, str]],
    ) -> list[StructuredCandidate]:
        if not self._object_escape_pressure(runtime):
            return []
        rows = self._board_rows(runtime)
        row_count = len(rows)
        col_count = max((len(row) for row in rows), default=0)
        if row_count <= 0 or col_count <= 0:
            return []
        full_object_table: list[dict[str, Any]] = list(
            getattr(getattr(runtime, "last_state", None), "object_table", None) or []
        )
        if not full_object_table:
            return []
        click_testing_stats = self._click_component_testing_stats(runtime)
        tested_components = set(click_testing_stats.get("tested", set()))
        # Phase 8.C-1-global: prefer global "changed" set when enabled,
        # else fall back to per-current-state changed set (C-1 legacy).
        if _enable_l1_focus_retarget_global():
            changed_components = set(click_testing_stats.get("changed_global", set()))
        else:
            changed_components = set(click_testing_stats.get("changed", set()))
        level_number = self._level_number(runtime)
        # Phase 8.C-1: at L1 with score=0, re-test components that
        # produced a board change (changed set) to restore focus/repetition
        # parity with naturally-scoring games. Inert components (in tested
        # but not in changed) remain excluded.
        l1_focus_active = (
            _enable_l1_focus_retarget()
            and int(level_number) == 1
            and int(getattr(getattr(runtime, "last_state", None), "score", 0) or 0) == 0
        )

        ranked: list[tuple[float, dict[str, Any]]] = []
        for entry in full_object_table:
            if not isinstance(entry, dict):
                continue
            component_id = str(entry.get("component_id", "") or "").strip()
            if not component_id:
                continue
            comp_lower = component_id.lower()
            if comp_lower in tested_components:
                if not (l1_focus_active and comp_lower in changed_components):
                    continue
            if not self._compact_object_for_escape(entry, row_count=row_count, col_count=col_count):
                continue
            bootstrap_score = self._object_table_bootstrap_score(
                component=entry,
                board_rows=rows,
                click_testing_stats=click_testing_stats,
            )
            ranked.append((float(bootstrap_score), dict(entry)))

        ranked.sort(
            key=lambda item: (
                -item[0],
                -int(item[1].get("size", 0) or 0),
                str(item[1].get("component_id", "") or ""),
            )
        )

        candidates: list[StructuredCandidate] = []
        max_candidates = max(1, min(4, int(self.config.max_from_object_table or 1)))
        for index, (score, entry) in enumerate(ranked[:max_candidates]):
            comp_id = str(entry.get("component_id", "") or "").strip()
            key = (comp_id, "click")
            if key in existing_keys:
                continue
            existing_keys.add(key)
            row, col = self._component_center(entry)
            anchor_key = self._anchor_key(level_number=level_number, component=entry)
            effect_bonus = self._effect_ledger_bonus(
                runtime,
                component_id=comp_id,
                family="click",
            )
            candidates.append(
                StructuredCandidate(
                    component_id=comp_id,
                    source="object_escape",
                    family="click",
                    anchor_key=anchor_key,
                    center_row=int(row),
                    center_col=int(col),
                    base_score=round(2.55 + float(score) - 0.04 * float(index) + effect_bonus, 4),
                    effect_bonus=float(effect_bonus),
                    step_index=step_index,
                )
            )
        return candidates

    @staticmethod
    def _size_bucket(size: int) -> int:
        size = max(1, int(size or 1))
        if size <= 2:
            return 1
        if size <= 5:
            return 2
        if size <= 12:
            return 3
        if size <= 32:
            return 4
        return 5

    @staticmethod
    def _compact_object_for_aligned_sweep(component: dict[str, Any], *, row_count: int, col_count: int) -> bool:
        bbox = dict(component.get("bbox") or {})
        min_row = int(bbox.get("min_row", 0) or 0)
        max_row = int(bbox.get("max_row", min_row) or min_row)
        min_col = int(bbox.get("min_col", 0) or 0)
        max_col = int(bbox.get("max_col", min_col) or min_col)
        bbox_area = max(1, (max_row - min_row + 1) * (max_col - min_col + 1))
        board_area = max(1, int(row_count or 0) * int(col_count or 0))
        size = max(1, int(component.get("size", 0) or 0))
        row, col = CandidateExtractor._component_center(component)
        if row <= 0 or col <= 0:
            return False
        if bbox_area > 0.18 * float(board_area):
            return False
        if size > 0.10 * float(board_area):
            return False
        compactness = min(1.0, float(size) / float(bbox_area))
        return bool(compactness >= 0.18)

    @staticmethod
    def _aligned_object_sweep_pressure(runtime: Any, *, step_index: int) -> bool:
        if CandidateExtractor._level_number(runtime) < 2:
            return False
        if int(step_index) < 8:
            return False
        rows = CandidateExtractor._board_rows(runtime)
        if not rows:
            return False
        if CandidateExtractor._level_rule_discovery_pressure_reason(runtime, step_index=step_index):
            return True
        if CandidateExtractor._object_escape_pressure(runtime):
            return True
        return int(getattr(runtime, "_same_family_no_score_streak", 0) or 0) >= 4

    def _aligned_object_sweep_candidates(
        self,
        runtime: Any,
        *,
        step_index: int,
        existing_keys: set[tuple[str, str]],
    ) -> list[StructuredCandidate]:
        if not self._aligned_object_sweep_pressure(runtime, step_index=step_index):
            return []
        rows = self._board_rows(runtime)
        row_count = len(rows)
        col_count = max((len(row) for row in rows), default=0)
        if row_count <= 0 or col_count <= 0:
            return []
        full_object_table: list[dict[str, Any]] = list(
            getattr(getattr(runtime, "last_state", None), "object_table", None) or []
        )
        if not full_object_table:
            return []

        click_testing_stats = self._click_component_testing_stats(runtime)
        tested_components = set(click_testing_stats.get("tested", set()))
        no_change_components = set(click_testing_stats.get("no_change", set()))
        recent_aligned_selection_count = self._recent_selected_source_count(
            runtime,
            source="aligned_object_sweep",
            limit=6,
        )
        alignment_lease_bonus = 0.0
        if recent_aligned_selection_count >= 2:
            alignment_lease_bonus = 2.25
        elif recent_aligned_selection_count == 1:
            alignment_lease_bonus = 1.65
        recent_positions = (
            self._recent_action6_positions(runtime, limit=24)
            + self._recent_selected_anchor_positions(runtime, limit=96)
        )
        level_number = self._level_number(runtime)
        groups: dict[tuple[str, int, str, int], list[tuple[int, dict[str, Any]]]] = {}
        for entry in full_object_table:
            if not isinstance(entry, dict):
                continue
            component_id = str(entry.get("component_id", "") or "").strip()
            if not component_id:
                continue
            if not self._compact_object_for_aligned_sweep(entry, row_count=row_count, col_count=col_count):
                continue
            row, col = self._component_center(entry)
            value = str(entry.get("value", "") or "")
            size_bucket = self._size_bucket(int(entry.get("size", 0) or 0))
            row_bin = int(round(float(row) / 2.0) * 2)
            col_bin = int(round(float(col) / 2.0) * 2)
            groups.setdefault(("row", row_bin, value, size_bucket), []).append((int(col), dict(entry)))
            groups.setdefault(("col", col_bin, value, size_bucket), []).append((int(row), dict(entry)))

        ranked_groups: list[tuple[float, tuple[str, int, str, int], list[tuple[int, dict[str, Any]]]]] = []
        for key, members in groups.items():
            deduped: dict[str, tuple[int, dict[str, Any]]] = {}
            for axis_pos, entry in members:
                component_id = str(entry.get("component_id", "") or "").strip()
                if component_id and component_id not in deduped:
                    deduped[component_id] = (axis_pos, entry)
            members = sorted(deduped.values(), key=lambda item: (item[0], str(item[1].get("component_id", "") or "")))
            if len(members) < 3:
                continue
            untested = sum(
                1
                for _, entry in members
                if str(entry.get("component_id", "") or "").strip().lower() not in tested_components
            )
            span = max(1, int(members[-1][0]) - int(members[0][0]))
            density = min(1.0, float(len(members)) / float(span + 1))
            score = 2.95 + min(0.35, 0.05 * float(len(members))) + 0.12 * density + 0.05 * float(untested)
            ranked_groups.append((round(score, 4), key, members))
        if not ranked_groups:
            return []
        ranked_groups.sort(key=lambda item: (-item[0], item[1][0], item[1][1], item[1][2], item[1][3]))

        candidates: list[StructuredCandidate] = []
        seen_components: set[str] = set()
        max_candidates = max(1, int(self.config.max_from_aligned_object_sweep or 1))
        for group_score, _, members in ranked_groups:
            for member_index, (_, entry) in enumerate(members):
                if len(candidates) >= max_candidates:
                    return candidates
                comp_id = str(entry.get("component_id", "") or "").strip()
                if not comp_id or comp_id in seen_components:
                    continue
                key = (comp_id, "click")
                if key in existing_keys:
                    continue
                normalized = comp_id.lower()
                row, col = self._component_center(entry)
                if any(abs(int(row) - int(r_row)) + abs(int(col) - int(r_col)) <= 1 for r_row, r_col in recent_positions):
                    continue
                tested_penalty = 0.14 if normalized in tested_components else 0.0
                no_change_penalty = 0.10 if normalized in no_change_components else 0.0
                effect_bonus = self._effect_ledger_bonus(runtime, component_id=comp_id, family="click")
                existing_keys.add(key)
                seen_components.add(comp_id)
                candidates.append(
                    StructuredCandidate(
                        component_id=comp_id,
                        source="aligned_object_sweep",
                        family="click",
                        anchor_key=self._anchor_key(level_number=level_number, component=entry),
                        center_row=int(row),
                        center_col=int(col),
                        base_score=round(
                            float(group_score)
                            + float(effect_bonus)
                            + float(alignment_lease_bonus)
                            - tested_penalty
                            - no_change_penalty
                            - 0.03 * float(member_index),
                            4,
                        ),
                        effect_bonus=round(effect_bonus, 4),
                        step_index=step_index,
                    )
                )
        return candidates

    def _transition_frontier_candidates(
        self,
        runtime: Any,
        *,
        step_index: int,
        existing_keys: set[tuple[str, str]],
    ) -> list[StructuredCandidate]:
        level_number = self._level_number(runtime)
        if level_number < 2:
            return []
        frontier_components = self._transition_frontier_components(
            runtime,
            step_index=step_index,
        )
        if not frontier_components:
            return []
        click_testing_stats = self._click_component_testing_stats(runtime)
        candidates: list[StructuredCandidate] = []
        for index, entry in enumerate(frontier_components):
            if len(candidates) >= max(1, int(self.config.max_from_object_table)):
                break
            comp_id = str(entry.get("component_id", "") or "").strip()
            if not comp_id:
                continue
            row, col = self._component_center(entry)
            frontier_score = float(entry.get("frontier_score", 0.0) or 0.0)
            families = self._object_table_candidate_families(
                component_id=comp_id,
                click_testing_stats=click_testing_stats,
            )
            families = self._filter_candidate_families_for_available_actions(runtime, families)
            for family_index, family in enumerate(families):
                if len(candidates) >= max(1, int(self.config.max_from_object_table)):
                    break
                key = (comp_id, family)
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                effect_bonus = self._effect_ledger_bonus(
                    runtime,
                    component_id=comp_id,
                    family=family,
                )
                if family == "move":
                    effect_bonus += 0.08
                anchor_key = str(entry.get("anchor_key", "") or "").strip()
                if not anchor_key:
                    anchor_key = self._anchor_key(level_number=level_number, component=entry)
                candidates.append(
                    StructuredCandidate(
                        component_id=comp_id,
                        source="transition_frontier",
                        family=family,
                        anchor_key=anchor_key,
                        center_row=int(row),
                        center_col=int(col),
                        base_score=round(
                            3.18
                            + min(0.35, max(0.0, frontier_score) * 0.25)
                            - 0.05 * float(index)
                            - 0.03 * float(family_index)
                            + effect_bonus,
                            4,
                        ),
                        effect_bonus=round(effect_bonus, 4),
                        step_index=step_index,
                    )
                )
        return candidates

    def _rule_hypothesis_candidates(
        self,
        runtime: Any,
        *,
        step_index: int,
        existing_keys: set[tuple[str, str]],
        family_filter: set[str] | None = None,
    ) -> list[StructuredCandidate]:
        provider = getattr(runtime, "rule_discovery_candidates", None)
        if not callable(provider):
            return []
        try:
            raw_candidates = list(provider(step_index=step_index) or [])
        except Exception:
            return []
        candidates: list[StructuredCandidate] = []
        for index, entry in enumerate(raw_candidates):
            if len(candidates) >= max(1, int(self.config.max_from_rule_hypothesis)):
                break
            if not isinstance(entry, dict):
                continue
            comp_id = str(entry.get("component_id", "") or "").strip()
            anchor_key = str(entry.get("anchor_key", "") or "").strip()
            family = str(entry.get("family", "") or "click").strip().lower()
            source = str(entry.get("source", "") or entry.get("candidate_source", "") or "rule_hypothesis").strip()
            if source not in {"rule_hypothesis", "goal_probe", "stabilize_verify"}:
                source = "rule_hypothesis"
            if not comp_id and source == "stabilize_verify" and anchor_key:
                comp_id = f"anchor::{anchor_key}"
            if not comp_id or not family:
                continue
            if not self._candidate_family_allowed(runtime, family):
                continue
            if family_filter is not None and family not in family_filter:
                continue
            key_family = family
            if source == "stabilize_verify":
                key_family = "|".join(
                    [
                        family,
                        "stabilize_verify",
                        str(entry.get("window_id", "") or ""),
                        str(entry.get("stage", "") or ""),
                    ]
                )
            key = (comp_id, key_family)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            candidates.append(
                StructuredCandidate(
                    component_id=comp_id,
                    source=source,
                    family=family,
                    action_name=str(entry.get("action_name", "") or "").strip().upper(),
                    anchor_key=anchor_key,
                    center_row=int(entry.get("center_row", 0) or 0),
                    center_col=int(entry.get("center_col", 0) or 0),
                    base_score=round(
                        float(entry.get("base_score", 4.0) or 4.0) - 0.03 * float(index),
                        4,
                    ),
                    effect_bonus=(
                        0.34
                        if source == "stabilize_verify"
                        else 0.22
                        if source == "goal_probe"
                        else 0.30
                        if str(entry.get("closure_status", "") or "") == "closed"
                        else 0.12
                    ),
                    step_index=step_index,
                    stage=str(entry.get("stage", "") or "").strip(),
                    closure_status=str(entry.get("closure_status", "") or "").strip(),
                    hypothesis_id=str(entry.get("hypothesis_id", "") or "").strip(),
                    hypothesis_kind=str(entry.get("hypothesis_kind", "") or "").strip(),
                    linked_effect_id=str(entry.get("linked_effect_id", "") or "").strip(),
                    window_id=str(entry.get("window_id", "") or "").strip(),
                    continuation_kind=str(entry.get("continuation_kind", "") or "").strip(),
                )
            )
        return candidates

    @staticmethod
    def _parse_action6_prompt_position(action_text: str) -> tuple[int, int] | None:
        """Parse BaseArcAgent ACTION6 display text as prompt-grid row/col."""
        match = re.search(r"ACTION6\(x=(-?\d+),\s*y=(-?\d+)", str(action_text or ""))
        if not match:
            return None
        try:
            row = int(match.group(1))
            col = int(match.group(2))
        except (TypeError, ValueError):
            return None
        return row, col

    @staticmethod
    def _recent_action6_positions(runtime: Any, *, limit: int = 12) -> list[tuple[int, int]]:
        state = getattr(runtime, "last_state", None)
        positions: list[tuple[int, int]] = []
        for step in reversed(list(getattr(state, "recent_history", None) or [])):
            if not isinstance(step, dict):
                continue
            action_text = str(step.get("action", "") or "")
            position = CandidateExtractor._parse_action6_prompt_position(action_text)
            if position is None:
                continue
            if position not in positions:
                positions.append(position)
            if len(positions) >= max(1, int(limit or 1)):
                break
        return positions

    @staticmethod
    def _parse_anchor_position(anchor_key: str) -> tuple[int, int] | None:
        parts = [part for part in str(anchor_key or "").strip().split("||") if part != ""]
        if len(parts) < 3:
            return None
        try:
            return int(parts[-2]), int(parts[-1])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _recent_selected_anchor_positions(runtime: Any, *, limit: int = 64) -> list[tuple[int, int]]:
        rows = getattr(runtime, "step_metrics", None)
        if not isinstance(rows, list):
            return []
        positions: list[tuple[int, int]] = []
        for row in reversed(rows[-max(1, int(limit or 1)):]):
            if not isinstance(row, dict):
                continue
            family = str(
                row.get("selected_target_transition_model_candidate_family", "")
                or row.get("local_controller_selected_action_family", "")
                or ""
            ).strip().lower()
            if family and family != "click":
                continue
            position = CandidateExtractor._parse_anchor_position(
                str(row.get("selected_target_anchor", "") or "")
            )
            if position is None:
                continue
            if position not in positions:
                positions.append(position)
        return positions

    @staticmethod
    def _recent_selected_source_count(runtime: Any, *, source: str, limit: int = 6) -> int:
        rows = getattr(runtime, "step_metrics", None)
        if not isinstance(rows, list):
            return 0
        normalized_source = str(source or "").strip()
        if not normalized_source:
            return 0
        count = 0
        for row in reversed(rows[-max(1, int(limit or 1)) :]):
            if not isinstance(row, dict):
                continue
            selected_source = str(row.get("selected_target_source", "") or "").strip()
            if selected_source == normalized_source:
                count += 1
        return count

    @staticmethod
    def _local_click_fanout_origin(runtime: Any) -> tuple[int, int] | None:
        state = getattr(runtime, "last_state", None)
        cluster: list[tuple[int, int]] = []
        for step in reversed(list(getattr(state, "recent_history", None) or [])):
            if not isinstance(step, dict):
                continue
            action_text = str(step.get("action", "") or "")
            match = re.search(r"ACTION6\(x=(-?\d+),\s*y=(-?\d+)", action_text)
            if not match:
                if cluster:
                    break
                continue
            if bool(step.get("no_state_change", False)):
                if cluster:
                    break
                continue
            position = CandidateExtractor._parse_action6_prompt_position(action_text)
            if position is None:
                continue
            cluster.append(position)
        if not cluster:
            return None
        return cluster[-1]

    @staticmethod
    def _last_action6_position(runtime: Any) -> tuple[int, int] | None:
        positions = CandidateExtractor._recent_action6_positions(runtime, limit=1)
        if positions:
            return positions[0]
        return None

    @staticmethod
    def _recent_no_change_action6_positions(runtime: Any, *, limit: int = 4) -> set[tuple[int, int]]:
        state = getattr(runtime, "last_state", None)
        positions: set[tuple[int, int]] = set()
        for step in reversed(list(getattr(state, "recent_history", None) or [])):
            if len(positions) >= max(1, int(limit or 1)):
                break
            if not isinstance(step, dict):
                continue
            action_text = str(step.get("action", "") or "")
            match = re.search(r"ACTION6\(x=(-?\d+),\s*y=(-?\d+)", action_text)
            if not match:
                continue
            if not bool(step.get("no_state_change", False)):
                continue
            position = CandidateExtractor._parse_action6_prompt_position(action_text)
            if position is None:
                continue
            positions.add(position)
        return positions

    @staticmethod
    def _recent_changed_action6_cluster(runtime: Any, *, limit: int = 12) -> list[tuple[int, int]]:
        state = getattr(runtime, "last_state", None)
        cluster_reversed: list[tuple[int, int]] = []
        for step in reversed(list(getattr(state, "recent_history", None) or [])):
            if len(cluster_reversed) >= max(1, int(limit or 1)):
                break
            if not isinstance(step, dict):
                continue
            action_text = str(step.get("action", "") or "")
            match = re.search(r"ACTION6\(x=(-?\d+),\s*y=(-?\d+)", action_text)
            if not match:
                if cluster_reversed:
                    break
                continue
            if bool(step.get("no_state_change", False)):
                if cluster_reversed:
                    break
                continue
            try:
                score_delta = float(step.get("score_delta", 0.0) or 0.0)
            except (TypeError, ValueError):
                score_delta = 0.0
            if abs(score_delta) > 1e-9:
                break
            position = CandidateExtractor._parse_action6_prompt_position(action_text)
            if position is None:
                continue
            cluster_reversed.append(position)
        return list(reversed(cluster_reversed))

    @staticmethod
    def _active_click_parity_positions(cluster: list[tuple[int, int]]) -> list[tuple[int, int]]:
        active: list[tuple[int, int]] = []
        active_set: set[tuple[int, int]] = set()
        for position in list(cluster or []):
            normalized = (int(position[0]), int(position[1]))
            if normalized in active_set:
                active_set.remove(normalized)
                active = [item for item in active if item != normalized]
            else:
                active_set.add(normalized)
                active.append(normalized)
        return active

    def _local_reversible_click_candidates(
        self,
        runtime: Any,
        *,
        step_index: int,
        existing_keys: set[tuple[str, str]],
    ) -> list[StructuredCandidate]:
        event = dict(getattr(runtime, "last_local_effect_event", {}) or {})
        if str(event.get("last_family", "") or "").strip().lower() != "click":
            return []
        if int(event.get("score_change_count", 0) or 0) > 0:
            return []
        progress_count = int(event.get("progress_effect_count", 0) or 0)
        board_change_count = int(event.get("board_change_no_score_count", 0) or 0)
        if progress_count <= 0 and board_change_count <= 0:
            return []
        cluster = self._recent_changed_action6_cluster(runtime, limit=12)
        unique_positions = list(dict.fromkeys(cluster))
        if len(unique_positions) < 4:
            return []
        # Once a reversible retest has happened, let fanout discover more
        # evidence before issuing another backtrack candidate.
        if len(unique_positions) < len(cluster):
            return []
        active_positions = self._active_click_parity_positions(cluster)
        if len(active_positions) < 4:
            return []
        rows = CandidateExtractor._board_rows(runtime)
        if not rows:
            return []
        max_row = len(rows) - 1
        max_col = max((len(row) for row in rows), default=0) - 1
        if max_row < 0 or max_col < 0:
            return []
        last_position = cluster[-1] if cluster else None
        recent_no_change_positions = self._recent_no_change_action6_positions(runtime, limit=4)
        centroid_row = sum(row for row, _ in active_positions) / float(len(active_positions))
        centroid_col = sum(col for _, col in active_positions) / float(len(active_positions))
        ranked = sorted(
            (
                position
                for position in active_positions
                if position != last_position
                and position not in recent_no_change_positions
            ),
            key=lambda position: (
                abs(float(position[0]) - centroid_row) + abs(float(position[1]) - centroid_col),
                position[0],
                position[1],
            ),
        )
        candidates: list[StructuredCandidate] = []
        for index, (row, col) in enumerate(ranked):
            if len(candidates) >= max(1, int(self.config.max_from_reversible_click)):
                break
            row = max(0, min(max_row, int(row)))
            col = max(0, min(max_col, int(col)))
            component_id = f"local_reversible_click_{row}_{col}"
            key = (component_id, "click")
            if key in existing_keys:
                continue
            existing_keys.add(key)
            anchor_key = f"grid||{row}||{col}"
            effect_bonus = self._grid_scan_effect_bonus(
                runtime,
                component_id=component_id,
                anchor_key=anchor_key,
            )
            candidates.append(
                StructuredCandidate(
                    component_id=component_id,
                    source="local_reversible_click",
                    family="click",
                    anchor_key=anchor_key,
                    center_row=int(row),
                    center_col=int(col),
                    base_score=round(2.70 - 0.05 * float(index) + effect_bonus, 4),
                    effect_bonus=float(effect_bonus),
                    step_index=step_index,
                )
            )
        return candidates

    def _local_click_fanout_candidates(
        self,
        runtime: Any,
        *,
        step_index: int,
        existing_keys: set[tuple[str, str]],
    ) -> list[StructuredCandidate]:
        event = dict(getattr(runtime, "last_local_effect_event", {}) or {})
        if str(event.get("last_family", "") or "").strip().lower() != "click":
            return []
        if int(event.get("score_change_count", 0) or 0) > 0:
            return []
        progress_count = int(event.get("progress_effect_count", 0) or 0)
        board_change_count = int(event.get("board_change_no_score_count", 0) or 0)
        if progress_count <= 0 and board_change_count <= 0:
            return []
        rows = CandidateExtractor._board_rows(runtime)
        if not rows:
            return []
        clicked = self._local_click_fanout_origin(runtime) or self._last_action6_position(runtime)
        if clicked is None:
            return []
        recent_clicked = set(self._recent_action6_positions(runtime, limit=16))
        base_row, base_col = clicked
        max_row = len(rows) - 1
        max_col = max((len(row) for row in rows), default=0) - 1
        if max_row < 0 or max_col < 0:
            return []
        offsets = [
            (8, 0),
            (0, 8),
            (8, 16),
            (16, 0),
            (8, 8),
            (0, 16),
            (-8, 0),
            (0, -8),
            (-8, -8),
            (-8, 8),
            (8, -8),
            (16, 8),
            (8, -16),
            (-16, 0),
            (0, -16),
        ]
        candidates: list[StructuredCandidate] = []
        seen_points: set[tuple[int, int]] = set()
        for index, (row_delta, col_delta) in enumerate(offsets):
            if len(candidates) >= max(1, int(self.config.max_from_grid_scan)):
                break
            row = max(0, min(max_row, int(base_row) + int(row_delta)))
            col = max(0, min(max_col, int(base_col) + int(col_delta)))
            if (row, col) == (base_row, base_col) or (row, col) in seen_points or (row, col) in recent_clicked:
                continue
            seen_points.add((row, col))
            component_id = f"local_click_fanout_{row}_{col}"
            key = (component_id, "click")
            if key in existing_keys:
                continue
            existing_keys.add(key)
            effect_bonus = self._grid_scan_effect_bonus(
                runtime,
                component_id=component_id,
                anchor_key=f"grid||{row}||{col}",
            )
            candidates.append(
                StructuredCandidate(
                    component_id=component_id,
                    source="local_click_fanout",
                    family="click",
                    anchor_key=f"grid||{row}||{col}",
                    center_row=int(row),
                    center_col=int(col),
                    base_score=round(2.15 - 0.03 * float(index) + effect_bonus, 4),
                    effect_bonus=float(effect_bonus),
                    step_index=step_index,
                )
            )
        return candidates

    @staticmethod
    def _grid_scan_effect_bonus(runtime: Any, *, component_id: str, anchor_key: str = "") -> float:
        resolver = getattr(runtime, "_local_effect_entry_for", None)
        entry: dict[str, Any] = {}
        if callable(resolver):
            try:
                entry = dict(resolver(component_id=component_id, anchor_key=anchor_key) or {})
            except Exception:
                entry = {}
        if not entry:
            ledger = getattr(runtime, "local_effect_ledger", None)
            if isinstance(ledger, dict):
                key = str(component_id or "").strip()
                entry = dict(ledger.get(key) or {})
                if not entry and anchor_key:
                    for raw_entry in ledger.values():
                        candidate_entry = dict(raw_entry or {})
                        if str(candidate_entry.get("anchor_key", "") or "") == str(anchor_key or ""):
                            entry = candidate_entry
                            break
        if not entry:
            return 0.0
        score_change_count = int(entry.get("score_change_count", 0) or 0)
        if score_change_count > 0:
            return 0.35
        no_change_count = int(entry.get("no_change_count", 0) or 0)
        weak_effect_count = int(entry.get("weak_effect_count", 0) or 0)
        repeated_effect_count = int(entry.get("repeated_effect_count", 0) or 0)
        board_change_count = int(entry.get("board_change_no_score_count", 0) or 0)
        probe_count = int(entry.get("probe_count", 0) or 0)
        if no_change_count > 0:
            return round(max(-0.85, -0.55 - 0.10 * float(no_change_count - 1)), 4)
        if weak_effect_count + repeated_effect_count > 0:
            return round(max(-0.70, -0.35 - 0.08 * float(weak_effect_count + repeated_effect_count - 1)), 4)
        if board_change_count > 0 and probe_count >= 2:
            return round(max(-0.55, -0.18 - 0.06 * float(board_change_count - 1)), 4)
        return 0.0

    @staticmethod
    def _effect_ledger_bonus(runtime: Any, *, component_id: str, family: str) -> float:
        ledger = getattr(runtime, "local_effect_ledger", None)
        if not isinstance(ledger, dict):
            return 0.0
        entry = dict(ledger.get(str(component_id or "").strip()) or {})
        if not entry:
            return 0.0
        family_counts = dict(entry.get("family_counts") or {})
        family_count = int(family_counts.get(str(family or "").strip(), 0) or 0)
        probe_count = int(entry.get("probe_count", 0) or 0)
        board_change_count = int(entry.get("board_change_no_score_count", 0) or 0)
        score_change_count = int(entry.get("score_change_count", 0) or 0)
        no_change_count = int(entry.get("no_change_count", 0) or 0)
        progress_effect_count = int(entry.get("progress_effect_count", 0) or 0)
        weak_effect_count = int(entry.get("weak_effect_count", 0) or 0)
        repeated_effect_count = int(entry.get("repeated_effect_count", 0) or 0)
        structural_delta_count = int(entry.get("structural_delta_count", 0) or 0)
        ledger_score = float(entry.get("score", 0.0) or 0.0)
        last_family = str(entry.get("last_family", "") or "")
        max_no_score_effect_probes = 4
        max_no_score_board_changes = 3
        if (
            score_change_count <= 0
            and probe_count >= max_no_score_effect_probes
            and board_change_count >= max_no_score_board_changes
            and structural_delta_count <= 0
        ):
            return -0.18
        if (
            score_change_count <= 0
            and probe_count >= 5
            and progress_effect_count >= 2
            and board_change_count >= 4
        ):
            return -0.16
        if score_change_count <= 0 and weak_effect_count >= 2 and structural_delta_count <= weak_effect_count:
            return -0.18
        bonus = 0.0
        if ledger_score > 0.0:
            bonus += min(0.18, 0.04 * ledger_score)
        if progress_effect_count > 0:
            bonus += min(0.16, 0.05 * progress_effect_count)
        if structural_delta_count > 0:
            bonus += min(0.12, 0.04 * structural_delta_count)
        if board_change_count > 0:
            bonus += min(0.08, 0.02 * board_change_count)
            if family and family != last_family:
                bonus += 0.04
        if score_change_count > 0:
            bonus += min(0.25, 0.15 * score_change_count)
        if family_count > 0:
            bonus += min(0.08, 0.02 * family_count)
        if weak_effect_count > 0 and score_change_count <= 0:
            bonus -= min(0.16, 0.03 * weak_effect_count)
        if repeated_effect_count > 0:
            bonus -= min(0.14, 0.04 * repeated_effect_count)
        if no_change_count >= 3 and family_count > 0:
            bonus -= min(0.18, 0.04 * no_change_count)
        return round(max(-0.18, min(0.35, bonus)), 4)

    @staticmethod
    def _effect_family_bonus(runtime: Any, *, family: str) -> float:
        ledger = getattr(runtime, "local_effect_family_ledger", None)
        if not isinstance(ledger, dict):
            return 0.0
        entry = dict(ledger.get(str(family or "").strip().lower()) or {})
        if not entry:
            return 0.0
        probe_count = int(entry.get("probe_count", 0) or 0)
        score_change_count = int(entry.get("score_change_count", 0) or 0)
        no_change_count = int(entry.get("no_change_count", 0) or 0)
        progress_effect_count = int(entry.get("progress_effect_count", 0) or 0)
        weak_effect_count = int(entry.get("weak_effect_count", 0) or 0)
        repeated_effect_count = int(entry.get("repeated_effect_count", 0) or 0)
        structural_delta_count = int(entry.get("structural_delta_count", 0) or 0)
        score = float(entry.get("score", 0.0) or 0.0)
        if probe_count <= 0:
            return 0.0
        bonus = 0.0
        if score > 0.0:
            bonus += min(0.10, 0.025 * score)
        if progress_effect_count > 0:
            bonus += min(0.16, 0.04 * progress_effect_count)
        if structural_delta_count > 0:
            bonus += min(0.10, 0.03 * structural_delta_count)
        if score_change_count > 0:
            bonus += min(0.20, 0.12 * score_change_count)
        if weak_effect_count > progress_effect_count:
            bonus -= min(0.18, 0.03 * (weak_effect_count - progress_effect_count))
        if repeated_effect_count > 0:
            bonus -= min(0.14, 0.04 * repeated_effect_count)
        if score_change_count <= 0 and progress_effect_count <= 0 and no_change_count >= 2:
            bonus -= min(0.16, 0.05 * no_change_count)
        return round(max(-0.20, min(0.28, bonus)), 4)

    def build_base_pool(self, runtime: Any, step_index: int) -> ExtractionResult:
        """Build a structured candidate pool from ECARuntime state.

        Args:
            runtime: ECARuntime instance (duck-typed).
            step_index: Current step index.

        Returns:
            ExtractionResult with structured candidates from all sources.
        """
        result = ExtractionResult(step_index=step_index)
        seen_keys: set[tuple[str, str]] = set()

        # ── 1. Recently-live anchor (highest priority) ─────────────────
        recently_live_family = ""
        try:
            model = getattr(runtime, "structural_time_tracker", None)
            if model is not None:
                recently_live = bool(getattr(model, "recently_live_anchor", False))
                recently_live_family = str(
                    getattr(model, "recently_live_anchor_family", "") or ""
                )
                result.recently_live_family = recently_live_family
                if recently_live and recently_live_family:
                    # Mark existing candidates from this family as recently_live
                    pass  # Applied below during pool construction
        except Exception:
            pass

        # ── 2. Successor slate ────────────────────────────────────────
        successor_slate: list[dict[str, Any]] = list(
            getattr(runtime, "last_successor_slate", None) or []
        )
        for entry in successor_slate[: self.config.max_from_successor_slate]:
            comp_id = str(entry.get("component_id", "") or entry.get("id", "") or "")
            family = self._extract_family(entry)
            seen_key = (comp_id, family)
            if (
                not comp_id
                or seen_key in seen_keys
                or not self._candidate_family_allowed(runtime, family)
            ):
                continue
            seen_keys.add(seen_key)
            base_score = float(entry.get("score", 0.0) or entry.get("base_score", 0.0) or 0.0)
            if base_score < self.config.min_base_score:
                continue
            cand = StructuredCandidate(
                component_id=comp_id,
                source="successor_slate",
                family=family,
                anchor_key=str(entry.get("anchor_key", "") or ""),
                center_row=int(entry.get("center_row", entry.get("row", 0)) or 0),
                center_col=int(entry.get("center_col", entry.get("col", 0)) or 0),
                base_score=base_score,
                recent_live=(family == recently_live_family and bool(recently_live_family)),
                step_index=step_index,
            )
            result.candidates.append(cand)

        # ── 3. Protected hedge ────────────────────────────────────────
        hedge_contract = getattr(runtime, "last_protected_hedge_contract", None)
        hedge_family = ""
        if hedge_contract and isinstance(hedge_contract, dict):
            comp_id = str(
                hedge_contract.get("component_id", "")
                or hedge_contract.get("anchor_component_id", "")
                or ""
            )
            hedge_family = self._extract_family(hedge_contract)
            result.hedge_family = hedge_family
            seen_key = (comp_id, hedge_family)
            if (
                comp_id
                and seen_key not in seen_keys
                and self._candidate_family_allowed(runtime, hedge_family)
            ):
                seen_keys.add(seen_key)
                base_score = float(hedge_contract.get("score", 0.0) or 0.0)
                cand = StructuredCandidate(
                    component_id=comp_id,
                    source="protected_hedge",
                    family=hedge_family,
                    anchor_key=str(hedge_contract.get("anchor_key", "") or ""),
                    center_row=int(((hedge_contract.get("center", {}) or {}).get("row", 0)) or 0),
                    center_col=int(((hedge_contract.get("center", {}) or {}).get("col", 0)) or 0),
                    base_score=base_score,
                    recent_live=(hedge_family == recently_live_family and bool(recently_live_family)),
                    is_hedge=True,
                    step_index=step_index,
                )
                result.candidates.append(cand)

        # ── 4. Existing candidate pool ────────────────────────────────
        # Gameplay-only click hypotheses are learned from observed effects.
        # Put click hypotheses before the raw candidate pool so same
        # component/family duplicates retain the rule_hypothesis source for
        # bounded follow-up packets. Move hypotheses are added after the raw
        # pool so navigation candidates do not get re-labeled as rule progress.
        result.candidates.extend(
            self._rule_hypothesis_candidates(
                runtime,
                step_index=step_index,
                existing_keys=seen_keys,
                family_filter={"click"},
            )
        )

        raw_pool: list[dict[str, Any]] = list(
            getattr(runtime, "last_candidate_pool", None) or []
        )
        for entry in raw_pool:
            if sum(1 for c in result.candidates if c.source == "candidate_pool") >= self.config.max_from_candidate_pool:
                break
            comp_id = str(entry.get("component_id", "") or entry.get("id", "") or "")
            family = self._extract_family(entry)
            seen_key = (comp_id, family)
            if (
                not comp_id
                or seen_key in seen_keys
                or not self._candidate_family_allowed(runtime, family)
            ):
                continue
            seen_keys.add(seen_key)
            base_score = float(entry.get("score", 0.0) or entry.get("base_score", 0.0) or 0.0)
            reject_reason = str(entry.get("rejected_reason", "") or entry.get("reject_reason", "") or "")
            if base_score < self.config.min_base_score and not reject_reason:
                continue
            cand = StructuredCandidate(
                component_id=comp_id,
                source="candidate_pool",
                family=family,
                anchor_key=str(entry.get("anchor_key", "") or ""),
                center_row=int(entry.get("center_row", entry.get("row", 0)) or 0),
                center_col=int(entry.get("center_col", entry.get("col", 0)) or 0),
                base_score=base_score,
                recent_live=(family == recently_live_family and bool(recently_live_family)),
                reject_reason=reject_reason,
                step_index=step_index,
            )
            result.candidates.append(cand)

        result.candidates.extend(
            self._rule_hypothesis_candidates(
                runtime,
                step_index=step_index,
                existing_keys=seen_keys,
            )
        )
        result.candidates.extend(
            self._aligned_object_sweep_candidates(
                runtime,
                step_index=step_index,
                existing_keys=seen_keys,
            )
        )

        # 5. Object-table bootstrap fallback
        if not self._has_viable_candidates(result.candidates):
            frontier_components = self._transition_frontier_components(
                runtime,
                step_index=step_index,
            )
            focus_components = self._bootstrap_focus_components(runtime)
            full_object_table: list[dict[str, Any]] = list(
                getattr(getattr(runtime, "last_state", None), "object_table", None) or []
            )
            selected_target_preference = self._selected_target_bootstrap_preference(runtime)
            object_table: list[dict[str, Any]] = []
            object_ids: set[str] = set()
            for entry in list(frontier_components) + (
                list(focus_components) if focus_components else list(full_object_table)
            ):
                if not isinstance(entry, dict):
                    continue
                component_id = str(entry.get("component_id", "") or "").strip()
                if not component_id or component_id in object_ids:
                    continue
                object_ids.add(component_id)
                object_table.append(dict(entry))
            if focus_components and bool(
                selected_target_preference.get("prefer_move_after_click_stale", False)
            ):
                preferred_component_id = str(
                    selected_target_preference.get("component_id", "") or ""
                ).strip()
                existing_ids = {
                    str(entry.get("component_id", "") or "").strip()
                    for entry in object_table
                    if isinstance(entry, dict)
                }
                if preferred_component_id and preferred_component_id not in existing_ids:
                    preferred_entry = next(
                        (
                            dict(entry)
                            for entry in full_object_table
                            if str(entry.get("component_id", "") or "").strip()
                            == preferred_component_id
                        ),
                        None,
                    )
                    if preferred_entry is not None:
                        object_table.append(preferred_entry)
            level_number = self._level_number(runtime)
            board_rows = self._board_rows(runtime)
            click_testing_stats = self._click_component_testing_stats(runtime)
            preferred_component_id = str(
                selected_target_preference.get("component_id", "") or ""
            ).strip()
            prefer_move_after_click_stale = bool(
                selected_target_preference.get("prefer_move_after_click_stale", False)
            )
            focus_rank_by_component = {
                str(entry.get("component_id", "") or "").strip(): index
                for index, entry in enumerate(focus_components)
                if str(entry.get("component_id", "") or "").strip()
            }
            focus_score_by_component = {
                str(entry.get("component_id", "") or "").strip(): float(entry.get("target_score", 0.0) or 0.0)
                for entry in focus_components
                if str(entry.get("component_id", "") or "").strip()
            }
            frontier_rank_by_component = {
                str(entry.get("component_id", "") or "").strip(): index
                for index, entry in enumerate(frontier_components)
                if str(entry.get("component_id", "") or "").strip()
            }
            frontier_score_by_component = {
                str(entry.get("component_id", "") or "").strip(): float(entry.get("frontier_score", 0.0) or 0.0)
                for entry in frontier_components
                if str(entry.get("component_id", "") or "").strip()
            }
            def bootstrap_score(entry: dict[str, Any]) -> float:
                component_id = str(entry.get("component_id", "") or "").strip()
                focus_rank = focus_rank_by_component.get(component_id)
                focus_score = float(focus_score_by_component.get(component_id, 0.0) or 0.0)
                focus_bonus = min(0.35, max(0.0, focus_score) * 0.07)
                if focus_rank is not None:
                    focus_bonus += max(0.0, 0.24 - 0.04 * float(focus_rank))
                if (
                    prefer_move_after_click_stale
                    and component_id
                    and component_id == preferred_component_id
                ):
                    focus_bonus += 0.16
                frontier_rank = frontier_rank_by_component.get(component_id)
                if frontier_rank is not None:
                    frontier_bonus = 0.30 + max(0.0, 0.12 - 0.03 * float(frontier_rank))
                    frontier_bonus += min(0.12, max(0.0, frontier_score_by_component.get(component_id, 0.0)) * 0.08)
                    focus_bonus += frontier_bonus
                return round(
                    self._object_table_bootstrap_score(
                        component=entry,
                        board_rows=board_rows,
                        click_testing_stats=click_testing_stats,
                    ) + focus_bonus,
                    4,
                )
            ranked_components = sorted(
                object_table,
                key=lambda entry: (
                    frontier_rank_by_component.get(
                        str(entry.get("component_id", "") or "").strip(),
                        10_000,
                    ),
                    0
                    if (
                        prefer_move_after_click_stale
                        and str(entry.get("component_id", "") or "").strip()
                        == preferred_component_id
                    )
                    else 1,
                    -bootstrap_score(entry),
                    -int(entry.get("size", 0) or 0),
                    str(entry.get("component_id", "") or ""),
                ),
            )
            added = 0
            for entry in ranked_components:
                if added >= self.config.max_from_object_table:
                    break
                comp_id = str(entry.get("component_id", "") or "").strip()
                if not comp_id:
                    continue
                row, col = self._component_center(entry)
                candidate_families = self._object_table_candidate_families(
                    component_id=comp_id,
                    click_testing_stats=click_testing_stats,
                    preferred_component_id=preferred_component_id,
                    prefer_move_after_click_stale=prefer_move_after_click_stale,
                )
                candidate_families = self._filter_candidate_families_for_available_actions(
                    runtime,
                    candidate_families,
                )
                for family in candidate_families:
                    if added >= self.config.max_from_object_table:
                        break
                    seen_key = (comp_id, family)
                    if seen_key in seen_keys:
                        continue
                    seen_keys.add(seen_key)
                    source = (
                        "transition_frontier"
                        if comp_id in frontier_rank_by_component
                        else "object_table"
                    )
                    family_bonus = 0.0
                    if family == "move":
                        family_bonus += 0.18
                        if str(comp_id).strip().lower() in set(click_testing_stats.get("no_change", set())):
                            family_bonus += 0.08
                        if prefer_move_after_click_stale and comp_id == preferred_component_id:
                            family_bonus += 0.18
                    elif family == "click" and str(comp_id).strip().lower() in set(click_testing_stats.get("tested", set())):
                        family_bonus -= 0.10
                        if prefer_move_after_click_stale and comp_id == preferred_component_id:
                            family_bonus -= 0.08
                    effect_bonus = self._effect_ledger_bonus(
                        runtime,
                        component_id=comp_id,
                        family=family,
                    )
                    effect_bonus = round(
                        effect_bonus + self._effect_family_bonus(runtime, family=family),
                        4,
                    )
                    cand = StructuredCandidate(
                        component_id=comp_id,
                        source=source,
                        family=family,
                        anchor_key=self._anchor_key(level_number=level_number, component=entry),
                        center_row=row,
                        center_col=col,
                        base_score=round(bootstrap_score(entry) + family_bonus + effect_bonus, 4),
                        effect_bonus=effect_bonus,
                        recent_live=(family == recently_live_family and bool(recently_live_family)),
                        step_index=step_index,
                    )
                    result.candidates.append(cand)
                    added += 1
        result.candidates.extend(
            self._local_reversible_click_candidates(
                runtime,
                step_index=step_index,
                existing_keys=seen_keys,
            )
        )
        result.candidates.extend(
            self._local_click_fanout_candidates(
                runtime,
                step_index=step_index,
                existing_keys=seen_keys,
            )
        )
        result.candidates.extend(
            self._object_escape_candidates(
                runtime,
                step_index=step_index,
                existing_keys=seen_keys,
            )
        )
        result.candidates.extend(
            self._transition_frontier_candidates(
                runtime,
                step_index=step_index,
                existing_keys=seen_keys,
            )
        )
        result.candidates.extend(
            self._level_rule_symbol_scout_candidates(
                runtime,
                step_index=step_index,
                existing_keys=seen_keys,
            )
        )
        result.candidates.extend(
            self._grid_scan_candidates(
                runtime,
                step_index=step_index,
                existing_keys=seen_keys,
            )
        )
        result.grid_scan_pressure_reason = self._last_grid_scan_pressure_reason
        result.candidates = [
            candidate
            for candidate in result.candidates
            if self._candidate_family_allowed(runtime, candidate.family)
        ]
        result.candidates = self._rebalance_candidates(result.candidates)
        result.source_counts = self._source_counts(result.candidates)
        result.family_counts = self._family_counts(result.candidates)

        self._history.append(result)
        return result

    def _rebalance_candidates(
        self,
        candidates: list[StructuredCandidate],
    ) -> list[StructuredCandidate]:
        """Prefer family breadth after preserving pinned candidates."""
        if len(candidates) <= 1:
            return list(candidates)

        selected: list[StructuredCandidate] = []
        selected_keys: set[tuple[str, str]] = set()
        selected_families: set[str] = set()

        for cand in candidates:
            if len(selected) >= self.config.max_total:
                break
            if not (cand.is_hedge or cand.recent_live):
                continue
            selected_key = (cand.component_id, cand.family)
            if selected_key in selected_keys:
                continue
            selected.append(cand)
            selected_keys.add(selected_key)
            if cand.family:
                selected_families.add(cand.family)

        # Preserve source breadth before filling by family. Rule-discovery,
        # stabilize, and visual-scan candidates are often all click-family;
        # without a source pass, late fallback probes can be trimmed out even
        # when they are the only fresh way to escape scoreless closure churn.
        priority_sources = (
            "stabilize_verify",
            "goal_probe",
            "rule_hypothesis",
            "transition_frontier",
            "object_escape",
            "level_rule_symbol_scout",
            "local_reversible_click",
            "local_click_fanout",
            "grid_scan",
        )
        for source in priority_sources:
            if len(selected) >= self.config.max_total:
                break
            source_candidates = [
                (index, cand)
                for index, cand in enumerate(candidates)
                if str(cand.source or "") == source
                and (cand.component_id, cand.family) not in selected_keys
            ]
            if not source_candidates:
                continue
            _, cand = max(
                source_candidates,
                key=lambda item: (
                    float(item[1].base_score or 0.0) + float(item[1].effect_bonus or 0.0),
                    1.0 if item[1].recent_live else 0.0,
                    1.0 if item[1].is_hedge else 0.0,
                    -int(item[0]),
                ),
            )
            selected_key = (cand.component_id, cand.family)
            if selected_key in selected_keys:
                continue
            selected.append(cand)
            selected_keys.add(selected_key)
            if cand.family:
                selected_families.add(cand.family)

        for cand in candidates:
            if len(selected) >= self.config.max_total:
                break
            selected_key = (cand.component_id, cand.family)
            if selected_key in selected_keys:
                continue
            if not cand.family or cand.family in selected_families:
                continue
            selected.append(cand)
            selected_keys.add(selected_key)
            selected_families.add(cand.family)

        for cand in candidates:
            if len(selected) >= self.config.max_total:
                break
            selected_key = (cand.component_id, cand.family)
            if selected_key in selected_keys:
                continue
            selected.append(cand)
            selected_keys.add(selected_key)

        return selected

    def _source_counts(self, candidates: list[StructuredCandidate]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cand in candidates:
            counts[cand.source] = int(counts.get(cand.source, 0) or 0) + 1
        return counts

    def _family_counts(self, candidates: list[StructuredCandidate]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cand in candidates:
            family = str(cand.family or "")
            if not family:
                continue
            counts[family] = int(counts.get(family, 0) or 0) + 1
        return counts

    def step_metrics(self) -> dict[str, Any]:
        """Return current step metrics for inclusion in runtime step_metrics."""
        if not self._history:
            return {
                "base_pool_size": 0,
                "base_pool_slate_count": 0,
                "base_pool_hedge_count": 0,
                "base_pool_pool_count": 0,
                "base_pool_object_table_count": 0,
                "base_pool_object_escape_count": 0,
                "base_pool_aligned_object_sweep_count": 0,
                "base_pool_rule_hypothesis_count": 0,
                "base_pool_goal_probe_count": 0,
                "base_pool_stabilize_verify_count": 0,
                "base_pool_transition_frontier_count": 0,
                "base_pool_level_rule_symbol_scout_count": 0,
                "base_pool_grid_scan_count": 0,
                "base_pool_local_click_fanout_count": 0,
                "base_pool_local_reversible_click_count": 0,
                "base_pool_grid_scan_pressure_reason": "",
                "base_pool_recently_live_family": "",
                "base_pool_hedge_family": "",
                "base_pool_unique_family_count": 0,
            }
        last = self._history[-1]
        return {
            "base_pool_size": len(last.candidates),
            "base_pool_slate_count": last.source_counts.get("successor_slate", 0),
            "base_pool_hedge_count": last.source_counts.get("protected_hedge", 0),
            "base_pool_pool_count": last.source_counts.get("candidate_pool", 0),
            "base_pool_object_table_count": last.source_counts.get("object_table", 0),
            "base_pool_object_escape_count": last.source_counts.get("object_escape", 0),
            "base_pool_aligned_object_sweep_count": last.source_counts.get("aligned_object_sweep", 0),
            "base_pool_rule_hypothesis_count": last.source_counts.get("rule_hypothesis", 0),
            "base_pool_goal_probe_count": last.source_counts.get("goal_probe", 0),
            "base_pool_stabilize_verify_count": last.source_counts.get("stabilize_verify", 0),
            "base_pool_transition_frontier_count": last.source_counts.get("transition_frontier", 0),
            "base_pool_level_rule_symbol_scout_count": last.source_counts.get("level_rule_symbol_scout", 0),
            "base_pool_grid_scan_count": last.source_counts.get("grid_scan", 0),
            "base_pool_local_click_fanout_count": last.source_counts.get("local_click_fanout", 0),
            "base_pool_local_reversible_click_count": last.source_counts.get("local_reversible_click", 0),
            "base_pool_grid_scan_pressure_reason": last.grid_scan_pressure_reason,
            "base_pool_recently_live_family": last.recently_live_family,
            "base_pool_hedge_family": last.hedge_family,
            "base_pool_unique_family_count": len(last.family_counts),
        }

    def episode_summary(self) -> dict[str, Any]:
        """Return episode-level summary for inclusion in runtime episode_summary."""
        if not self._history:
            return {
                "total_steps": 0,
                "avg_pool_size": 0.0,
                "max_pool_size": 0,
                "min_pool_size": 0,
                "steps_with_empty_pool": 0,
                "steps_with_object_table_fallback": 0,
                "steps_with_object_escape": 0,
                "steps_with_aligned_object_sweep": 0,
                "steps_with_rule_hypothesis": 0,
                "steps_with_goal_probe": 0,
                "steps_with_stabilize_verify": 0,
                "stabilize_verify_base_pool_count": 0,
                "steps_with_transition_frontier": 0,
                "steps_with_level_rule_symbol_scout": 0,
                "steps_with_grid_scan": 0,
                "steps_with_level_rule_discovery_grid_scan": 0,
                "grid_scan_pressure_reason_histogram": {},
                "avg_unique_family_count": 0.0,
                "steps_with_multi_family_pool": 0,
            }
        sizes = [len(r.candidates) for r in self._history]
        unique_families = [len(r.family_counts) for r in self._history]
        reason_histogram: dict[str, int] = {}
        for result in self._history:
            reason = str(result.grid_scan_pressure_reason or "").strip()
            if not reason:
                continue
            reason_histogram[reason] = int(reason_histogram.get(reason, 0) or 0) + 1
        return {
            "total_steps": len(self._history),
            "avg_pool_size": round(sum(sizes) / len(sizes), 2),
            "max_pool_size": max(sizes),
            "min_pool_size": min(sizes),
            "steps_with_empty_pool": sum(1 for s in sizes if s == 0),
            "steps_with_object_table_fallback": sum(
                1 for r in self._history if int(r.source_counts.get("object_table", 0) or 0) > 0
            ),
            "steps_with_object_escape": sum(
                1 for r in self._history if int(r.source_counts.get("object_escape", 0) or 0) > 0
            ),
            "steps_with_aligned_object_sweep": sum(
                1 for r in self._history if int(r.source_counts.get("aligned_object_sweep", 0) or 0) > 0
            ),
            "steps_with_rule_hypothesis": sum(
                1 for r in self._history if int(r.source_counts.get("rule_hypothesis", 0) or 0) > 0
            ),
            "steps_with_goal_probe": sum(
                1 for r in self._history if int(r.source_counts.get("goal_probe", 0) or 0) > 0
            ),
            "steps_with_stabilize_verify": sum(
                1 for r in self._history if int(r.source_counts.get("stabilize_verify", 0) or 0) > 0
            ),
            "stabilize_verify_base_pool_count": sum(
                int(r.source_counts.get("stabilize_verify", 0) or 0) for r in self._history
            ),
            "steps_with_transition_frontier": sum(
                1 for r in self._history if int(r.source_counts.get("transition_frontier", 0) or 0) > 0
            ),
            "steps_with_level_rule_symbol_scout": sum(
                1 for r in self._history if int(r.source_counts.get("level_rule_symbol_scout", 0) or 0) > 0
            ),
            "steps_with_grid_scan": sum(
                1 for r in self._history if int(r.source_counts.get("grid_scan", 0) or 0) > 0
            ),
            "steps_with_level_rule_discovery_grid_scan": sum(
                1
                for r in self._history
                if int(r.source_counts.get("grid_scan", 0) or 0) > 0
                and str(r.grid_scan_pressure_reason or "").startswith("level_rule_discovery")
            ),
            "grid_scan_pressure_reason_histogram": reason_histogram,
            "avg_unique_family_count": round(sum(unique_families) / len(unique_families), 2),
            "steps_with_multi_family_pool": sum(1 for n in unique_families if n >= 2),
        }
