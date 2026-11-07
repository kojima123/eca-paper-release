"""Passive transition memory for affordance-aware ECA.

This is Phase T-1 of the affordance-aware planning line: collect compact,
frame-only transition events and duplicate statistics without changing
planning.  It consumes the existing ActionEffectRecord emitted by
PrimitiveSynthesisLoop.observe_transition and stores no environment internals.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np


def _payload_xy(payload: dict[str, Any] | None) -> tuple[int | None, int | None]:
    raw = dict(payload or {})
    try:
        x = int(raw["x"]) if raw.get("x") is not None else None
        y = int(raw["y"]) if raw.get("y") is not None else None
    except (TypeError, ValueError):
        return (None, None)
    return (x, y)


def _normalise_bbox(raw: Any) -> tuple[int, int, int, int] | None:
    if raw is None:
        return None
    if hasattr(raw, "min_y"):
        try:
            return (
                int(getattr(raw, "min_x")),
                int(getattr(raw, "min_y")),
                int(getattr(raw, "max_x")),
                int(getattr(raw, "max_y")),
            )
        except (TypeError, ValueError):
            return None
    if isinstance(raw, dict):
        try:
            return (
                int(raw["min_x"]),
                int(raw["min_y"]),
                int(raw["max_x"]),
                int(raw["max_y"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            return tuple(int(v) for v in raw)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    return None


def _colour_histogram_delta(pre_arr: np.ndarray, post_arr: np.ndarray) -> dict[str, int]:
    """Palette-keyed count delta, useful even when object segmentation fails."""
    try:
        pre_vals, pre_counts = np.unique(pre_arr, return_counts=True)
        post_vals, post_counts = np.unique(post_arr, return_counts=True)
    except Exception:
        return {}
    counts: Counter[int] = Counter()
    for val, count in zip(pre_vals.tolist(), pre_counts.tolist()):
        counts[int(val)] -= int(count)
    for val, count in zip(post_vals.tolist(), post_counts.tolist()):
        counts[int(val)] += int(count)
    return {
        str(key): int(value)
        for key, value in sorted(counts.items())
        if int(value) != 0
    }


def _palette_invariant_delta(pre_arr: np.ndarray, post_arr: np.ndarray) -> dict[str, Any]:
    """Shape-only changed-region summary, independent of concrete colours."""
    try:
        changed = np.asarray(pre_arr) != np.asarray(post_arr)
        raw_delta = int(np.count_nonzero(changed))
        if raw_delta <= 0:
            return {
                "changed_pixels": 0,
                "changed_fraction": 0.0,
                "bbox_area": 0,
                "bbox_fill": 0.0,
            }
        ys, xs = np.where(changed)
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        bbox_area = int((max_x - min_x + 1) * (max_y - min_y + 1))
        total = int(changed.size) or 1
        return {
            "changed_pixels": raw_delta,
            "changed_fraction": round(float(raw_delta) / float(total), 6),
            "bbox": (min_x, min_y, max_x, max_y),
            "bbox_area": bbox_area,
            "bbox_fill": round(float(raw_delta) / float(max(1, bbox_area)), 6),
        }
    except Exception:
        return {}


@dataclass(frozen=True)
class TransitionEvent:
    game_id: str | None
    level_id: int | None
    step: int
    pre_hash: str
    post_hash: str
    action_type: str
    x: int | None = None
    y: int | None = None
    changed: bool = False
    raw_delta_pixels: int = 0
    delta_bbox: tuple[int, int, int, int] | None = None
    score_delta: int = 0
    level_delta: int = 0
    object_delta: dict[str, Any] = field(default_factory=dict)
    palette_invariant_delta: dict[str, Any] = field(default_factory=dict)
    predicted_change_prob: float = 0.5
    surprise: float = 0.5
    psa_tags: list[str] = field(default_factory=list)
    duplicate_count_before: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TransitionMemory:
    """Bounded append-only transition event store with hash dedup stats."""

    def __init__(self, *, max_events: int = 512) -> None:
        self.max_events = max(1, int(max_events or 1))
        self._events: deque[TransitionEvent] = deque(maxlen=self.max_events)
        self._transition_counts: Counter[tuple[str, str, int | None, int | None, str]] = Counter()
        self._action_attempts: Counter[str] = Counter()
        self._action_changed: Counter[str] = Counter()
        self.total_events = 0
        self.duplicate_events = 0

    @property
    def events(self) -> list[TransitionEvent]:
        return list(self._events)

    def reset(self) -> None:
        self._events.clear()
        self._transition_counts.clear()
        self._action_attempts.clear()
        self._action_changed.clear()
        self.total_events = 0
        self.duplicate_events = 0

    def predicted_change_prob(self, action_type: str) -> float:
        action = str(action_type or "").strip().upper()
        attempts = int(self._action_attempts.get(action, 0))
        if attempts <= 0:
            return 0.5
        # Laplace smoothing keeps early surprise finite.
        changed = int(self._action_changed.get(action, 0))
        return float(changed + 1) / float(attempts + 2)

    def observe(
        self,
        record: Any,
        *,
        pre_frame: Any,
        post_frame: Any,
        game_id: str | None = None,
        level_id: int | None = None,
        psa_tags: Iterable[str] | None = None,
    ) -> TransitionEvent:
        pre_arr = np.asarray(pre_frame)
        post_arr = np.asarray(post_frame)
        action = str(getattr(record, "action_name", "") or "").strip().upper()
        x, y = _payload_xy(getattr(record, "action_payload", None))
        predicted = self.predicted_change_prob(action)
        changed = bool(int(getattr(record, "changed_pixel_count", 0) or 0) > 0)
        observed = 1.0 if changed else 0.0
        transition_key = (
            str(getattr(record, "pre_state_hash", "") or ""),
            action,
            x,
            y,
            str(getattr(record, "post_state_hash", "") or ""),
        )
        duplicate_count = int(self._transition_counts.get(transition_key, 0))
        event = TransitionEvent(
            game_id=str(game_id).strip() if game_id else None,
            level_id=int(level_id) if level_id is not None else None,
            step=int(getattr(record, "step_index", self.total_events) or self.total_events),
            pre_hash=str(getattr(record, "pre_state_hash", "") or ""),
            post_hash=str(getattr(record, "post_state_hash", "") or ""),
            action_type=action,
            x=x,
            y=y,
            changed=changed,
            raw_delta_pixels=int(getattr(record, "changed_pixel_count", 0) or 0),
            delta_bbox=_normalise_bbox(getattr(record, "changed_bbox", None)),
            score_delta=int(getattr(record, "score_delta", 0) or 0),
            level_delta=0,
            object_delta={
                "colour_count_delta": _colour_histogram_delta(pre_arr, post_arr),
                "segment_displacements": list(getattr(record, "segment_displacements", []) or []),
            },
            palette_invariant_delta=_palette_invariant_delta(pre_arr, post_arr),
            predicted_change_prob=round(float(predicted), 6),
            surprise=round(abs(float(predicted) - observed), 6),
            psa_tags=[str(tag) for tag in (psa_tags or []) if str(tag or "").strip()],
            duplicate_count_before=duplicate_count,
        )
        self._events.append(event)
        self._transition_counts[transition_key] += 1
        self._action_attempts[action] += 1
        if changed:
            self._action_changed[action] += 1
        self.total_events += 1
        if duplicate_count > 0:
            self.duplicate_events += 1
        return event

    def action_change_rates(self) -> dict[str, float]:
        return {
            action: round(
                float(self._action_changed.get(action, 0)) / float(max(1, attempts)),
                6,
            )
            for action, attempts in sorted(self._action_attempts.items())
        }

    def summary(self) -> dict[str, Any]:
        total = int(self.total_events)
        changed = int(sum(self._action_changed.values()))
        return {
            "enabled": True,
            "event_count": total,
            "active_event_count": int(len(self._events)),
            "transition_key_count": int(len(self._transition_counts)),
            "duplicate_count": int(self.duplicate_events),
            "duplicate_ratio": round(float(self.duplicate_events) / float(max(1, total)), 6),
            "changed_event_count": changed,
            "changed_action_ratio": round(float(changed) / float(max(1, total)), 6),
            "action_change_rates": self.action_change_rates(),
            "mean_surprise": round(
                sum(float(event.surprise) for event in self._events)
                / float(max(1, len(self._events))),
                6,
            ),
            "recent": [event.to_dict() for event in list(self._events)[-5:]],
        }
