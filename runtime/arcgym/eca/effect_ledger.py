"""Append-only gameplay effect ledger for generic rule closure.

The ledger stores only observations produced by gameplay: selected target,
action, no-score board deltas, and normalized component locations.  It does
not know task ids, source-code facts, or adapter-private metadata.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EffectComponent:
    component_id: str
    anchor_key: str
    center_row: int
    center_col: int
    value: int = 0
    size: int = 0
    change_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EffectRecord:
    effect_id: str
    level_idx: int
    step_idx: int
    source_component_id: str
    source_anchor_key: str
    source_center_row: int
    source_center_col: int
    selected_source: str
    action_family: str
    action_name: str
    effect_kind: str
    changed_components: list[EffectComponent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["changed_components"] = [item.to_dict() for item in self.changed_components]
        return payload


class EffectLedger:
    """Bounded append-only effect history for one episode or level."""

    def __init__(self, *, max_records: int = 256) -> None:
        self.max_records = max(1, int(max_records or 1))
        self._records: list[EffectRecord] = []
        self.record_count = 0

    def reset(self) -> None:
        self._records = []
        self.record_count = 0

    @property
    def records(self) -> list[EffectRecord]:
        return list(self._records)

    def record(
        self,
        *,
        level_idx: int,
        step_idx: int,
        source_component_id: str,
        source_anchor_key: str,
        source_center_row: int,
        source_center_col: int,
        selected_source: str,
        action_family: str,
        action_name: str,
        effect_kind: str,
        changed_components: list[dict[str, Any]],
    ) -> EffectRecord | None:
        source_component_id = str(source_component_id or "").strip()
        source_anchor_key = str(source_anchor_key or "").strip()
        if not str(source_component_id or source_anchor_key).strip():
            return None
        components: list[EffectComponent] = []
        for raw in list(changed_components or []):
            if not isinstance(raw, dict):
                continue
            component_id = str(raw.get("component_id", "") or "").strip()
            anchor_key = str(raw.get("anchor_key", "") or "").strip()
            if not str(component_id or anchor_key).strip():
                continue
            try:
                center_row = int(raw.get("center_row", 0) or 0)
                center_col = int(raw.get("center_col", 0) or 0)
            except (TypeError, ValueError):
                continue
            components.append(
                EffectComponent(
                    component_id=component_id,
                    anchor_key=anchor_key,
                    center_row=center_row,
                    center_col=center_col,
                    value=int(raw.get("value", 0) or 0),
                    size=int(raw.get("size", 0) or 0),
                    change_kind=str(raw.get("change_kind", "") or ""),
                )
            )
        if not components:
            return None
        effect_id = "|".join(
            [
                str(int(level_idx)),
                str(int(step_idx)),
                source_component_id or source_anchor_key,
                str(action_name or "").strip().upper(),
                str(len(components)),
            ]
        )
        record = EffectRecord(
            effect_id=effect_id,
            level_idx=int(level_idx),
            step_idx=int(step_idx),
            source_component_id=source_component_id,
            source_anchor_key=source_anchor_key,
            source_center_row=int(source_center_row),
            source_center_col=int(source_center_col),
            selected_source=str(selected_source or "").strip(),
            action_family=str(action_family or "").strip().lower(),
            action_name=str(action_name or "").strip().upper(),
            effect_kind=str(effect_kind or "").strip(),
            changed_components=components,
        )
        self._records.append(record)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]
        self.record_count += 1
        return record

    def recent(self, *, level_idx: int, current_step: int, ttl_steps: int) -> list[EffectRecord]:
        min_step = int(current_step) - max(1, int(ttl_steps or 1))
        return [
            record
            for record in self._records
            if int(record.level_idx) == int(level_idx)
            and int(record.step_idx) >= min_step
            and int(record.step_idx) <= int(current_step)
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "effect_ledger_record_count": int(self.record_count),
            "effect_ledger_active_record_count": int(len(self._records)),
        }
