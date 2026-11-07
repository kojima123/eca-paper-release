"""Generic affordance graph for gameplay-only goal probes.

This module converts effect records plus the current object table into a small
candidate set.  Edges are intentionally generic geometric relations such as
same-line and near-changed-component.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arcgym.eca.effect_ledger import EffectRecord


@dataclass
class GoalProbeCandidate:
    component_id: str
    anchor_key: str
    center_row: int
    center_col: int
    family: str = "click"
    action_name: str = "ACTION6"
    base_score: float = 0.0
    reason: str = ""
    effect_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "anchor_key": self.anchor_key,
            "center_row": int(self.center_row),
            "center_col": int(self.center_col),
            "family": self.family,
            "action_name": self.action_name,
            "base_score": round(float(self.base_score), 4),
            "source": "goal_probe",
            "stage": "goal_probe",
            "goal_probe_reason": self.reason,
            "linked_effect_id": self.effect_id,
        }


class AffordanceGraph:
    """Build bounded goal-probe candidates from generic component relations."""

    @staticmethod
    def _component_center(entry: dict[str, Any]) -> tuple[int, int]:
        if "center_row" in entry or "center_col" in entry:
            return int(entry.get("center_row", 0) or 0), int(entry.get("center_col", 0) or 0)
        center = dict(entry.get("center") or {})
        if center:
            return int(center.get("row", 0) or 0), int(center.get("col", 0) or 0)
        bbox = dict(entry.get("bbox") or {})
        min_row = int(bbox.get("min_row", 0) or 0)
        max_row = int(bbox.get("max_row", min_row) or min_row)
        min_col = int(bbox.get("min_col", 0) or 0)
        max_col = int(bbox.get("max_col", min_col) or min_col)
        return ((min_row + max_row) // 2, (min_col + max_col) // 2)

    @staticmethod
    def _anchor_key(*, level_idx: int, component: dict[str, Any]) -> str:
        row, col = AffordanceGraph._component_center(component)
        return f"{int(level_idx)}||{row}||{col}"

    @staticmethod
    def _compact_score(component: dict[str, Any]) -> float:
        bbox = dict(component.get("bbox") or {})
        min_row = int(bbox.get("min_row", 0) or 0)
        max_row = int(bbox.get("max_row", min_row) or min_row)
        min_col = int(bbox.get("min_col", 0) or 0)
        max_col = int(bbox.get("max_col", min_col) or min_col)
        bbox_area = max(1, (max_row - min_row + 1) * (max_col - min_col + 1))
        size = max(1, int(component.get("size", 0) or 0))
        return min(1.0, float(size) / float(bbox_area))

    @staticmethod
    def _component_id(component: dict[str, Any]) -> str:
        return str(component.get("component_id", "") or "").strip()

    def goal_probe_candidates(
        self,
        *,
        records: list[EffectRecord],
        current_components: list[dict[str, Any]],
        level_idx: int,
        current_step: int,
        max_candidates: int = 4,
        min_level_idx: int = 2,
    ) -> list[GoalProbeCandidate]:
        if not records or not current_components or int(level_idx) < int(min_level_idx):
            return []
        components = [dict(item) for item in current_components if isinstance(item, dict)]
        if not components:
            return []
        by_id = {self._component_id(item): item for item in components if self._component_id(item)}
        scored: dict[tuple[str, str], tuple[float, GoalProbeCandidate]] = {}
        for record in sorted(records, key=lambda item: int(item.step_idx), reverse=True):
            source_ids = {
                str(record.source_component_id or "").strip(),
                str(record.source_anchor_key or "").strip(),
            }
            changed_points = [
                (int(item.center_row), int(item.center_col), str(item.component_id or ""), str(item.anchor_key or ""))
                for item in record.changed_components
            ]
            if not changed_points:
                continue
            recency_bonus = max(0.0, 0.35 - 0.015 * float(max(0, int(current_step) - int(record.step_idx))))
            source_is_sweep = 1.0 if str(record.selected_source or "") == "aligned_object_sweep" else 0.0
            for component in components:
                component_id = self._component_id(component)
                if not component_id:
                    continue
                row, col = self._component_center(component)
                if row <= 0 or col <= 0:
                    continue
                anchor_key = self._anchor_key(level_idx=level_idx, component=component)
                if component_id in source_ids or anchor_key in source_ids:
                    continue
                best_relation_score = 0.0
                best_reason = ""
                for changed_row, changed_col, changed_id, changed_anchor in changed_points:
                    manhattan = abs(row - changed_row) + abs(col - changed_col)
                    if component_id == changed_id or anchor_key == changed_anchor:
                        relation_score = 1.35
                        reason = "changed_component"
                    elif row == changed_row or col == changed_col:
                        relation_score = max(0.45, 1.0 - min(0.55, manhattan / 80.0))
                        reason = "same_line_as_change"
                    elif manhattan <= 8:
                        relation_score = 0.62
                        reason = "near_change"
                    else:
                        continue
                    if relation_score > best_relation_score:
                        best_relation_score = relation_score
                        best_reason = reason
                if best_relation_score <= 0.0:
                    continue
                compact_bonus = 0.18 * self._compact_score(component)
                size = max(1, int(component.get("size", 0) or 0))
                size_penalty = min(0.45, 0.01 * max(0, size - 12))
                score = (
                    3.35
                    + best_relation_score
                    + compact_bonus
                    + recency_bonus
                    + 0.35 * source_is_sweep
                    - size_penalty
                )
                key = (component_id, anchor_key)
                candidate = GoalProbeCandidate(
                    component_id=component_id,
                    anchor_key=anchor_key,
                    center_row=row,
                    center_col=col,
                    base_score=score,
                    reason=best_reason,
                    effect_id=str(record.effect_id),
                )
                previous = scored.get(key)
                if previous is None or score > previous[0]:
                    scored[key] = (score, candidate)
        ranked = sorted(
            scored.values(),
            key=lambda item: (
                -float(item[0]),
                int(item[1].center_row),
                int(item[1].center_col),
                str(item[1].component_id),
            ),
        )
        return [item[1] for item in ranked[: max(1, int(max_candidates or 1))]]
