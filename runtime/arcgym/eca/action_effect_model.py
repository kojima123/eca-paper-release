"""Frame-only action-effect memory and primitive classification.

This module is intentionally game-agnostic. It records transitions from
visible frames, public score/state fields, and the issued action. It does not
read environment internals or adapter-private state.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any

import numpy as np

from arcgym.perception.object_segments import connected_color_segments, match_segment_displacements
from .structural_rods import EffectEvent, build_effect_event


DIRECTIONAL_ACTIONS = frozenset({"ACTION1", "ACTION2", "ACTION3", "ACTION4"})
CLICK_ACTIONS = frozenset({"ACTION6"})
ACTION_DIRECTION_VECTORS = {
    "ACTION1": (0.0, -1.0),
    "ACTION2": (0.0, 1.0),
    "ACTION3": (-1.0, 0.0),
    "ACTION4": (1.0, 0.0),
}


@dataclass(frozen=True)
class ChangedBBox:
    min_y: int
    min_x: int
    max_y: int
    max_x: int

    @property
    def width(self) -> int:
        return int(self.max_x - self.min_x + 1)

    @property
    def height(self) -> int:
        return int(self.max_y - self.min_y + 1)

    @property
    def area(self) -> int:
        return int(self.width * self.height)

    def to_dict(self) -> dict[str, int]:
        payload = asdict(self)
        payload["width"] = self.width
        payload["height"] = self.height
        payload["area"] = self.area
        return payload


@dataclass(frozen=True)
class ActionEffectRecord:
    action_name: str
    step_index: int
    pre_state_hash: str
    post_state_hash: str
    frame_shape: tuple[int, ...]
    pre_score: int
    post_score: int
    score_delta: int
    pre_state: str
    post_state: str
    changed_pixel_count: int
    changed_fraction: float
    changed_bbox: ChangedBBox | None = None
    changed_values: list[dict[str, Any]] = field(default_factory=list)
    no_op: bool = False
    action_payload: dict[str, Any] = field(default_factory=dict)
    segment_displacements: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["changed_bbox"] = self.changed_bbox.to_dict() if self.changed_bbox else None
        return payload


@dataclass(frozen=True)
class ActionEffectStats:
    action_name: str
    attempts: int
    changed_count: int
    no_op_count: int
    score_event_count: int
    score_delta_total: int
    avg_changed_pixels: float
    avg_changed_fraction: float
    max_changed_fraction: float
    localized_change_rate: float

    @property
    def changed_rate(self) -> float:
        if self.attempts <= 0:
            return 0.0
        return float(self.changed_count) / float(self.attempts)

    @property
    def no_op_rate(self) -> float:
        if self.attempts <= 0:
            return 0.0
        return float(self.no_op_count) / float(self.attempts)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["changed_rate"] = self.changed_rate
        payload["no_op_rate"] = self.no_op_rate
        return payload


@dataclass(frozen=True)
class ActionEffectSummary:
    records_count: int
    changed_count: int
    no_op_count: int
    score_event_count: int
    score_delta_total: int
    overall_no_op_rate: float
    changed_actions: list[str]
    score_actions: list[str]
    action_stats: dict[str, ActionEffectStats]
    object_continuity: dict[str, Any] = field(default_factory=dict)
    target_candidates: list[dict[str, Any]] = field(default_factory=list)
    goal_transition_profile: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_count": self.records_count,
            "changed_count": self.changed_count,
            "no_op_count": self.no_op_count,
            "score_event_count": self.score_event_count,
            "score_delta_total": self.score_delta_total,
            "overall_no_op_rate": self.overall_no_op_rate,
            "changed_actions": list(self.changed_actions),
            "score_actions": list(self.score_actions),
            "action_stats": {
                action: stats.to_dict()
                for action, stats in sorted(self.action_stats.items())
            },
            "object_continuity": dict(self.object_continuity),
            "target_candidates": [dict(candidate) for candidate in self.target_candidates],
            "goal_transition_profile": dict(self.goal_transition_profile),
        }


@dataclass(frozen=True)
class PrimitiveClassification:
    primary_label: str
    confidence: float
    scores: dict[str, float]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _effect_family_evidence(summary: ActionEffectSummary) -> dict[str, Any]:
    stats = summary.action_stats
    dir_stats = [stats[name] for name in sorted(DIRECTIONAL_ACTIONS & set(stats))]
    click_stats = [stats[name] for name in sorted(CLICK_ACTIONS & set(stats))]
    non_direction_stats = [
        item
        for name, item in sorted(stats.items())
        if name not in DIRECTIONAL_ACTIONS and name not in CLICK_ACTIONS
    ]

    dir_attempts = sum(item.attempts for item in dir_stats)
    dir_changed = sum(item.changed_count for item in dir_stats)
    dir_localized = sum(item.localized_change_rate * item.attempts for item in dir_stats)
    dir_changed_rate = _safe_rate(dir_changed, dir_attempts)
    dir_localized_rate = _safe_rate(dir_localized, dir_attempts)

    click_attempts = sum(item.attempts for item in click_stats)
    click_changed = sum(item.changed_count for item in click_stats)
    click_score_events = sum(item.score_event_count for item in click_stats)
    click_changed_rate = _safe_rate(click_changed, click_attempts)
    click_score_rate = _safe_rate(click_score_events, click_attempts)
    click_avg_fraction = (
        sum(item.avg_changed_fraction * item.attempts for item in click_stats) / float(click_attempts)
        if click_attempts
        else 0.0
    )

    non_direction_attempts = sum(item.attempts for item in non_direction_stats)
    non_direction_changed = sum(item.changed_count for item in non_direction_stats)
    non_direction_localized = sum(item.localized_change_rate * item.attempts for item in non_direction_stats)
    non_direction_changed_rate = _safe_rate(non_direction_changed, non_direction_attempts)
    non_direction_localized_rate = _safe_rate(non_direction_localized, non_direction_attempts)

    changed_action_count = len(summary.changed_actions)
    action_count = len(stats)
    score_action_count = len(summary.score_actions)
    multi_action_score = (
        min(1.0, float(changed_action_count) / max(1.0, float(action_count)))
        if changed_action_count >= 2
        else 0.0
    )

    return {
        "directional_changed_rate": dir_changed_rate,
        "directional_localized_rate": dir_localized_rate,
        "click_changed_rate": click_changed_rate,
        "click_score_rate": click_score_rate,
        "click_avg_changed_fraction": click_avg_fraction,
        "non_direction_changed_rate": non_direction_changed_rate,
        "non_direction_localized_rate": non_direction_localized_rate,
        "changed_action_count": changed_action_count,
        "action_count": action_count,
        "score_action_count": score_action_count,
        "multi_action_score": multi_action_score,
        "non_direction_changed_actions": [
            name for name in summary.changed_actions if name not in DIRECTIONAL_ACTIONS
        ],
        "object_continuity": dict(summary.object_continuity),
        "target_candidates": [dict(candidate) for candidate in summary.target_candidates],
        "goal_transition_profile": dict(summary.goal_transition_profile),
    }


def infer_solver_archetypes(summary: ActionEffectSummary) -> dict[str, float]:
    """Infer game-agnostic solver-family hints from visible action effects.

    These scores are intentionally conservative diagnostics. They should guide
    primitive selection or probing, but they are not claims that a game has
    been solved and they do not use game ids, level ids, or environment
    internals.
    """

    evidence = _effect_family_evidence(summary)
    directional_changed_rate = float(evidence["directional_changed_rate"])
    directional_localized_rate = float(evidence["directional_localized_rate"])
    click_changed_rate = float(evidence["click_changed_rate"])
    click_score_rate = float(evidence["click_score_rate"])
    click_avg_fraction = float(evidence["click_avg_changed_fraction"])
    non_direction_changed_rate = float(evidence["non_direction_changed_rate"])
    non_direction_localized_rate = float(evidence["non_direction_localized_rate"])
    multi_action_score = float(evidence["multi_action_score"])
    object_continuity = dict(evidence.get("object_continuity") or {})
    object_like_continuity_score = float(
        object_continuity.get("object_like_continuity_score", 0.0) or 0.0
    )

    grid_navigation = min(1.0, 0.55 * directional_changed_rate + 0.45 * directional_localized_rate)
    click_toggle = min(
        1.0,
        0.55 * click_changed_rate + 0.25 * min(1.0, click_avg_fraction / 0.08) + 0.20 * click_score_rate,
    )
    push_object = min(
        1.0,
        0.35 * directional_localized_rate
        + 0.25 * multi_action_score
        + 0.20 * max(non_direction_localized_rate, click_changed_rate)
        + 0.20 * object_like_continuity_score,
    )
    attach_transport = min(
        1.0,
        0.30 * directional_changed_rate
        + 0.25 * non_direction_changed_rate
        + 0.15 * non_direction_localized_rate
        + 0.10 * multi_action_score
        + 0.20 * object_like_continuity_score,
    )
    broad_overlay = min(1.0, max(0.0, click_avg_fraction - 0.06) * 5.0 + 0.30 * click_score_rate)

    return {
        "grid_navigation": float(grid_navigation),
        "click_toggle": float(click_toggle),
        "push_object_candidate": float(push_object),
        "attach_transport_candidate": float(attach_transport),
        "broad_overlay_candidate": float(broad_overlay),
        "action_loop_probe": float(summary.overall_no_op_rate),
    }


def _normalize_action_name(action_name: Any) -> str:
    text = str(action_name or "").strip().upper()
    if text.startswith("GAMEACTION."):
        text = text.split(".", 1)[1]
    return text


def _as_frame_array(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.size <= 0:
        raise ValueError("frame must be non-empty")
    return np.ascontiguousarray(arr)


def _frame_hash(frame: np.ndarray) -> str:
    arr = np.ascontiguousarray(frame)
    hasher = hashlib.sha1()
    hasher.update(str(arr.shape).encode("ascii", errors="ignore"))
    hasher.update(str(arr.dtype).encode("ascii", errors="ignore"))
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def frame_state_hash(frame: Any) -> str:
    """Return the stable visible-state hash used by action-effect records."""

    return _frame_hash(_as_frame_array(frame))


def _changed_mask(pre_frame: np.ndarray, post_frame: np.ndarray) -> np.ndarray:
    if pre_frame.shape != post_frame.shape:
        raise ValueError(f"frame shape changed from {pre_frame.shape} to {post_frame.shape}")
    diff = pre_frame != post_frame
    if diff.ndim >= 3:
        return np.any(diff, axis=tuple(range(2, diff.ndim)))
    return diff


def _bbox_from_mask(mask: np.ndarray) -> ChangedBBox | None:
    ys, xs = np.nonzero(mask)
    if ys.size <= 0 or xs.size <= 0:
        return None
    return ChangedBBox(
        min_y=int(ys.min()),
        min_x=int(xs.min()),
        max_y=int(ys.max()),
        max_x=int(xs.max()),
    )


def _bbox_center(bbox: ChangedBBox) -> tuple[float, float]:
    return ((float(bbox.min_x) + float(bbox.max_x)) / 2.0, (float(bbox.min_y) + float(bbox.max_y)) / 2.0)


def _bbox_distance(a: ChangedBBox, b: ChangedBBox) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def _bbox_center_dict(bbox: ChangedBBox) -> dict[str, int]:
    x, y = _bbox_center(bbox)
    return {"x": int(round(x)), "y": int(round(y))}


def _target_candidate_bbox_allowed(record: ActionEffectRecord) -> bool:
    bbox = record.changed_bbox
    if bbox is None:
        return False
    frame_height = int(record.frame_shape[0]) if record.frame_shape else 1
    frame_width = int(record.frame_shape[1]) if len(record.frame_shape) >= 2 else frame_height
    frame_area = max(1, frame_width * frame_height)
    bbox_fraction = float(bbox.area) / float(frame_area)
    if record.changed_fraction > 0.12 or bbox_fraction > 0.08:
        return False
    touches_border = (
        int(bbox.min_x) <= 0
        or int(bbox.min_y) <= 0
        or int(bbox.max_x) >= max(0, frame_width - 1)
        or int(bbox.max_y) >= max(0, frame_height - 1)
    )
    if touches_border and (bbox.width <= 2 or bbox.height <= 2):
        return False
    return True


def _record_goal_likelihood(record: ActionEffectRecord) -> float:
    if int(record.score_delta) > 0:
        return 1.0
    state_changed = bool(record.pre_state and record.post_state and record.pre_state != record.post_state)
    terminal_no_score = (
        int(record.score_delta) <= 0
        and str(record.post_state).upper() in {"GAME_OVER", "LOST", "DEAD", "FAILED"}
        and str(record.pre_state).upper() != str(record.post_state).upper()
    )
    if terminal_no_score:
        return -0.75
    if state_changed:
        return 0.35
    if int(record.changed_pixel_count) > 0 and float(record.changed_fraction) <= 0.08:
        return 0.05
    return 0.0


def _direction_alignment(action_name: Any, dx: float, dy: float) -> dict[str, Any]:
    vector = ACTION_DIRECTION_VECTORS.get(_normalize_action_name(action_name))
    if vector is None:
        return {
            "axis_delta": 0.0,
            "perpendicular_delta": abs(float(dx)) + abs(float(dy)),
            "direction_match": False,
        }
    vx, vy = vector
    axis_delta = float(dx) * vx + float(dy) * vy
    perpendicular_delta = abs(float(dy)) if vx else abs(float(dx))
    return {
        "axis_delta": axis_delta,
        "perpendicular_delta": perpendicular_delta,
        "direction_match": bool(axis_delta >= 1.0 and axis_delta >= perpendicular_delta),
    }


def _direction_sign(dx: float, dy: float) -> tuple[int, int]:
    if abs(float(dx)) >= abs(float(dy)):
        if float(dx) > 0:
            return 1, 0
        if float(dx) < 0:
            return -1, 0
    else:
        if float(dy) > 0:
            return 0, 1
        if float(dy) < 0:
            return 0, -1
    return 0, 0


def _segment_motion_consistency_lookup(
    groups: list[dict[str, Any]],
) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    out: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for group in groups:
        direction = group.get("direction")
        if not isinstance(direction, (list, tuple)) or len(direction) < 2:
            continue
        try:
            key = (
                _normalize_action_name(group.get("action_name")),
                int(group.get("segment_color", 0) or 0),
                int(direction[0]),
                int(direction[1]),
            )
        except (TypeError, ValueError):
            continue
        out[key] = dict(group)
    return out


def _segment_displacement_dicts(
    pre_frame: np.ndarray,
    post_frame: np.ndarray,
    *,
    limit: int = 16,
) -> list[dict[str, Any]]:
    try:
        frame_height = int(pre_frame.shape[0]) if pre_frame.ndim >= 1 else 1
        frame_width = int(pre_frame.shape[1]) if pre_frame.ndim >= 2 else frame_height
        max_distance = max(4.0, 0.40 * float(max(frame_width, frame_height)))
        matches = match_segment_displacements(
            connected_color_segments(pre_frame, ignore_colors=(0,), min_size=2),
            connected_color_segments(post_frame, ignore_colors=(0,), min_size=2),
            max_distance=max_distance,
            max_size_delta_fraction=0.9,
        )
    except Exception:
        return []
    moving = [
        item.to_dict()
        for item in matches
        if abs(float(item.dx)) + abs(float(item.dy)) >= 1.0
    ]
    return moving[: int(max(1, limit))]


def _sample_changed_values(
    pre_frame: np.ndarray,
    post_frame: np.ndarray,
    mask: np.ndarray,
    *,
    limit: int = 16,
) -> list[dict[str, Any]]:
    ys, xs = np.nonzero(mask)
    values: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for y, x in zip(ys.tolist(), xs.tolist()):
        before = pre_frame[y, x]
        after = post_frame[y, x]
        before_value = before.tolist() if hasattr(before, "tolist") else before
        after_value = after.tolist() if hasattr(after, "tolist") else after
        key = (_freeze_jsonable(before_value), _freeze_jsonable(after_value))
        if key in seen:
            continue
        seen.add(key)
        values.append({"before": before_value, "after": after_value, "count_sampled": 1})
        if len(values) >= limit:
            break
    return values


def _freeze_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_jsonable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_jsonable(v)) for k, v in value.items()))
    if isinstance(value, np.generic):
        return value.item()
    return value


class ActionEffectTracker:
    """Append-only action-effect memory for one probe episode."""

    def __init__(self, *, max_records: int = 512) -> None:
        self.max_records = max(1, int(max_records or 1))
        self._records: list[ActionEffectRecord] = []
        self.record_count = 0

    @property
    def records(self) -> list[ActionEffectRecord]:
        return list(self._records)

    def reset(self) -> None:
        self._records = []
        self.record_count = 0

    def observe_transition(
        self,
        *,
        action_name: Any,
        pre_frame: Any,
        post_frame: Any,
        pre_score: int = 0,
        post_score: int = 0,
        pre_state: str = "",
        post_state: str = "",
        step_index: int | None = None,
        action_payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActionEffectRecord:
        pre_arr = _as_frame_array(pre_frame)
        post_arr = _as_frame_array(post_frame)
        mask = _changed_mask(pre_arr, post_arr)
        changed_pixel_count = int(np.count_nonzero(mask))
        total_pixels = max(1, int(mask.size))
        pre_hash = _frame_hash(pre_arr)
        post_hash = _frame_hash(post_arr)
        score_delta = int(post_score or 0) - int(pre_score or 0)
        state_changed = str(pre_state or "") != str(post_state or "")
        no_op = changed_pixel_count == 0 and score_delta == 0 and not state_changed
        record = ActionEffectRecord(
            action_name=_normalize_action_name(action_name),
            step_index=int(self.record_count if step_index is None else step_index),
            pre_state_hash=pre_hash,
            post_state_hash=post_hash,
            frame_shape=tuple(int(v) for v in pre_arr.shape),
            pre_score=int(pre_score or 0),
            post_score=int(post_score or 0),
            score_delta=score_delta,
            pre_state=str(pre_state or ""),
            post_state=str(post_state or ""),
            changed_pixel_count=changed_pixel_count,
            changed_fraction=float(changed_pixel_count) / float(total_pixels),
            changed_bbox=_bbox_from_mask(mask),
            changed_values=_sample_changed_values(pre_arr, post_arr, mask),
            no_op=bool(no_op),
            action_payload=dict(action_payload or {}),
            segment_displacements=_segment_displacement_dicts(pre_arr, post_arr),
            metadata=dict(metadata or {}),
        )
        self._records.append(record)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]
        self.record_count += 1
        return record

    def summary(self) -> ActionEffectSummary:
        return summarize_action_effect_records(self.records)


def effect_event_from_record(
    record: ActionEffectRecord,
    *,
    target_kind: str = "",
    objective_progress: bool = False,
    objective_progress_kind: str = "none",
    goal_progress: bool = False,
    goal_progress_kind: str = "none",
    goal_aligned_progress: bool = False,
    goal_aligned_progress_kind: str = "none",
    binding_source: str = "action_effect_record",
    route_connectivity_delta: dict[str, Any] | None = None,
) -> EffectEvent:
    """Normalize one transition into a game-independent effect event."""

    inferred_target_kind = str(target_kind or "")
    if not inferred_target_kind and record.segment_displacements:
        inferred_target_kind = "segment_displacement"
    return build_effect_event(
        action_name=str(record.action_name),
        score_delta=int(record.score_delta),
        changed_pixel_count=int(record.changed_pixel_count),
        no_op=bool(record.no_op),
        target_kind=inferred_target_kind,
        segment_motion=list(record.segment_displacements or []),
        route_connectivity_delta=dict(route_connectivity_delta or {}),
        binding_source=str(binding_source or "action_effect_record"),
        objective_progress=bool(objective_progress),
        objective_progress_kind=str(objective_progress_kind or "none"),
        goal_progress=bool(goal_progress),
        goal_progress_kind=str(goal_progress_kind or "none"),
        goal_aligned_progress=bool(goal_aligned_progress),
        goal_aligned_progress_kind=str(goal_aligned_progress_kind or "none"),
    )


def record_state_signatures(record: ActionEffectRecord) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return compact before/after signatures safe for rod payloads."""

    before = {
        "state_hash": str(record.pre_state_hash),
        "score": int(record.pre_score),
        "state": str(record.pre_state or ""),
        "frame_shape": list(record.frame_shape),
    }
    after = {
        "state_hash": str(record.post_state_hash),
        "score": int(record.post_score),
        "state": str(record.post_state or ""),
        "frame_shape": list(record.frame_shape),
        "changed_pixel_count": int(record.changed_pixel_count),
        "changed_fraction": round(float(record.changed_fraction), 6),
        "changed_bbox": record.changed_bbox.to_dict() if record.changed_bbox else None,
    }
    return before, after


def observed_effect_from_record(record: ActionEffectRecord, event: EffectEvent) -> dict[str, Any]:
    """Compact observed-effect payload for CausalTraceRod."""

    return {
        "effect_type": str(event.effect_type),
        "action_name": str(record.action_name),
        "score_delta": int(record.score_delta),
        "changed_pixel_count": int(record.changed_pixel_count),
        "no_op": bool(record.no_op),
        "segment_motion": [dict(item) for item in list(record.segment_displacements or [])],
        "changed_bbox": record.changed_bbox.to_dict() if record.changed_bbox else None,
        "changed_values": list(record.changed_values[:8]),
    }


def summarize_action_effect_records(records: list[ActionEffectRecord]) -> ActionEffectSummary:
    records = list(records)
    by_action: dict[str, list[ActionEffectRecord]] = defaultdict(list)
    for record in records:
        by_action[record.action_name].append(record)

    action_stats: dict[str, ActionEffectStats] = {}
    for action_name, action_records in by_action.items():
        attempts = len(action_records)
        changed_records = [record for record in action_records if record.changed_pixel_count > 0]
        score_records = [record for record in action_records if record.score_delta != 0]
        localized_records = [
            record
            for record in changed_records
            if record.changed_bbox is not None
            and record.changed_fraction <= 0.08
            and record.changed_bbox.area <= max(4, int(0.20 * np.prod(record.frame_shape[:2])))
        ]
        action_stats[action_name] = ActionEffectStats(
            action_name=action_name,
            attempts=attempts,
            changed_count=len(changed_records),
            no_op_count=sum(1 for record in action_records if record.no_op),
            score_event_count=len(score_records),
            score_delta_total=sum(record.score_delta for record in action_records),
            avg_changed_pixels=(
                sum(record.changed_pixel_count for record in action_records) / float(attempts)
                if attempts
                else 0.0
            ),
            avg_changed_fraction=(
                sum(record.changed_fraction for record in action_records) / float(attempts)
                if attempts
                else 0.0
            ),
            max_changed_fraction=max((record.changed_fraction for record in action_records), default=0.0),
            localized_change_rate=(
                float(len(localized_records)) / float(attempts)
                if attempts
                else 0.0
            ),
        )

    records_count = len(records)
    changed_count = sum(1 for record in records if record.changed_pixel_count > 0)
    no_op_count = sum(1 for record in records if record.no_op)
    score_event_count = sum(1 for record in records if record.score_delta != 0)
    return ActionEffectSummary(
        records_count=records_count,
        changed_count=changed_count,
        no_op_count=no_op_count,
        score_event_count=score_event_count,
        score_delta_total=sum(record.score_delta for record in records),
        overall_no_op_rate=(float(no_op_count) / float(records_count) if records_count else 0.0),
        changed_actions=sorted(
            action for action, stats in action_stats.items() if stats.changed_count > 0
        ),
        score_actions=sorted(
            action for action, stats in action_stats.items() if stats.score_event_count > 0
        ),
        action_stats=action_stats,
        object_continuity=infer_object_continuity(records),
        target_candidates=infer_target_candidates(records),
        goal_transition_profile=infer_goal_transition_profile(records),
    )


def infer_goal_transition_profile(records: list[ActionEffectRecord]) -> dict[str, Any]:
    """Summarize public goal-like transitions without game internals.

    This separates score-bearing transitions from ordinary local progress.
    It deliberately stays conservative: positive score is strong goal evidence,
    state changes without score are only weak evidence, and terminal/no-score
    transitions are risk evidence rather than progress evidence.
    """

    ordered = sorted(list(records), key=lambda record: record.step_index)
    positive_score_actions: set[str] = set()
    state_advance_actions: set[str] = set()
    terminal_no_score_actions: set[str] = set()
    local_progress_only_actions: set[str] = set()
    by_action: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "positive_score": 0,
            "state_advance": 0,
            "terminal_no_score": 0,
            "local_progress_only": 0,
        }
    )

    for record in ordered:
        action = str(record.action_name)
        state_changed = bool(record.pre_state and record.post_state and record.pre_state != record.post_state)
        terminal_no_score = (
            int(record.score_delta) <= 0
            and str(record.post_state).upper() in {"GAME_OVER", "LOST", "DEAD", "FAILED"}
            and str(record.pre_state).upper() != str(record.post_state).upper()
        )
        positive_score = int(record.score_delta) > 0
        local_progress_only = (
            not positive_score
            and not terminal_no_score
            and int(record.changed_pixel_count) > 0
            and float(record.changed_fraction) <= 0.08
        )
        state_advance = state_changed and not terminal_no_score and int(record.score_delta) >= 0

        if positive_score:
            positive_score_actions.add(action)
            by_action[action]["positive_score"] += 1
        if state_advance:
            state_advance_actions.add(action)
            by_action[action]["state_advance"] += 1
        if terminal_no_score:
            terminal_no_score_actions.add(action)
            by_action[action]["terminal_no_score"] += 1
        if local_progress_only:
            local_progress_only_actions.add(action)
            by_action[action]["local_progress_only"] += 1

    action_goal_scores: dict[str, float] = {}
    for action, counts in sorted(by_action.items()):
        score = (
            1.0 * counts["positive_score"]
            + 0.35 * counts["state_advance"]
            - 0.55 * counts["terminal_no_score"]
            - 0.08 * max(0, counts["local_progress_only"] - counts["positive_score"])
        )
        action_goal_scores[action] = round(max(-1.0, min(1.0, score)), 6)

    return {
        "positive_score_actions": sorted(positive_score_actions),
        "state_advance_actions": sorted(state_advance_actions),
        "terminal_no_score_actions": sorted(terminal_no_score_actions),
        "local_progress_only_actions": sorted(local_progress_only_actions),
        "positive_score_count": sum(1 for record in ordered if int(record.score_delta) > 0),
        "terminal_no_score_count": sum(
            1
            for record in ordered
            if int(record.score_delta) <= 0
            and str(record.post_state).upper() in {"GAME_OVER", "LOST", "DEAD", "FAILED"}
            and str(record.pre_state).upper() != str(record.post_state).upper()
        ),
        "action_goal_scores": action_goal_scores,
    }


def infer_object_continuity(records: list[ActionEffectRecord]) -> dict[str, Any]:
    """Summarize visible localized-change continuity across transitions.

    This is not object recognition. It is a frame-only signal that repeated
    localized bboxes move or persist through consecutive public states, which
    is useful for push/attach candidate selection.
    """

    ordered = sorted(list(records), key=lambda record: record.step_index)
    localized = [
        record
        for record in ordered
        if record.changed_bbox is not None
        and record.changed_pixel_count > 0
        and record.changed_fraction <= 0.08
    ]
    linked_pairs: list[tuple[ActionEffectRecord, ActionEffectRecord]] = []
    close_pairs = 0
    same_action_pairs = 0
    distances: list[float] = []
    for left, right in zip(localized, localized[1:]):
        if left.post_state_hash != right.pre_state_hash:
            continue
        linked_pairs.append((left, right))
        if left.action_name == right.action_name:
            same_action_pairs += 1
        if left.changed_bbox is not None and right.changed_bbox is not None:
            distance = _bbox_distance(left.changed_bbox, right.changed_bbox)
            distances.append(distance)
            frame_height = int(right.frame_shape[0]) if right.frame_shape else 1
            frame_width = int(right.frame_shape[1]) if len(right.frame_shape) >= 2 else frame_height
            local_radius = max(4.0, 0.20 * float(max(frame_width, frame_height)))
            if distance <= local_radius:
                close_pairs += 1

    localized_count = len(localized)
    linked_count = len(linked_pairs)
    segment_records = [
        record
        for record in ordered
        if record.segment_displacements
    ]
    segment_count = sum(len(record.segment_displacements) for record in segment_records)
    segment_distances = [
        float(item.get("distance", 0.0) or 0.0)
        for record in segment_records
        for item in record.segment_displacements
    ]
    directional_segment_records = [
        record for record in segment_records if record.action_name in DIRECTIONAL_ACTIONS
    ]
    consistency_groups = infer_segment_motion_consistency(ordered)
    repeated_consistent_groups = [
        group
        for group in consistency_groups
        if int(group.get("direction_match_count", 0) or 0) >= 2
        and float(group.get("direction_match_rate", 0.0) or 0.0) >= 0.5
    ]
    segment_motion_score = min(
        1.0,
        0.45 * _safe_rate(len(segment_records), len(ordered))
        + 0.35 * _safe_rate(len(directional_segment_records), max(1, len(segment_records)))
        + 0.20 * min(1.0, _safe_rate(segment_count, max(1, len(ordered)))),
    )
    return {
        "localized_transition_count": localized_count,
        "localized_transition_rate": _safe_rate(localized_count, len(ordered)),
        "linked_localized_pair_count": linked_count,
        "linked_localized_pair_rate": _safe_rate(linked_count, max(0, localized_count - 1)),
        "close_bbox_pair_count": close_pairs,
        "close_bbox_pair_rate": _safe_rate(close_pairs, linked_count),
        "same_action_pair_rate": _safe_rate(same_action_pairs, linked_count),
        "segment_displacement_record_count": len(segment_records),
        "segment_displacement_count": segment_count,
        "segment_displacement_record_rate": _safe_rate(len(segment_records), len(ordered)),
        "directional_segment_displacement_rate": _safe_rate(
            len(directional_segment_records),
            max(1, len(segment_records)),
        ),
        "segment_motion_actions": sorted({record.action_name for record in segment_records}),
        "segment_motion_consistency_groups": consistency_groups[:8],
        "repeated_consistent_segment_motion_group_count": len(repeated_consistent_groups),
        "consistent_segment_motion_score": min(
            1.0,
            0.50 * _safe_rate(len(repeated_consistent_groups), max(1, len(consistency_groups)))
            + 0.50 * min(
                1.0,
                sum(float(group.get("direction_match_count", 0) or 0) for group in repeated_consistent_groups)
                / max(1.0, float(len(ordered))),
            ),
        ),
        "mean_segment_displacement_distance": (
            float(sum(segment_distances)) / float(len(segment_distances))
            if segment_distances
            else 0.0
        ),
        "segment_motion_score": segment_motion_score,
        "mean_bbox_center_shift": (
            float(sum(distances)) / float(len(distances)) if distances else 0.0
        ),
        "object_like_continuity_score": min(
            1.0,
            0.45 * _safe_rate(localized_count, len(ordered))
            + 0.35 * _safe_rate(close_pairs, linked_count)
            + 0.20 * _safe_rate(linked_count, max(0, localized_count - 1))
            + 0.25 * segment_motion_score,
        ),
    }


def infer_segment_motion_consistency(records: list[ActionEffectRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for record in sorted(list(records), key=lambda item: item.step_index):
        action_name = _normalize_action_name(record.action_name)
        if action_name not in DIRECTIONAL_ACTIONS:
            continue
        for displacement in list(record.segment_displacements or []):
            if not isinstance(displacement, dict):
                continue
            dx = float(displacement.get("dx", 0.0) or 0.0)
            dy = float(displacement.get("dy", 0.0) or 0.0)
            alignment = _direction_alignment(action_name, dx, dy)
            if not bool(alignment["direction_match"]):
                continue
            direction = _direction_sign(dx, dy)
            color = int(displacement.get("color", 0) or 0)
            key = (action_name, color, direction[0], direction[1])
            group = groups.setdefault(
                key,
                {
                    "action_name": action_name,
                    "segment_color": color,
                    "direction": [direction[0], direction[1]],
                    "observation_count": 0,
                    "direction_match_count": 0,
                    "alignment_total": 0.0,
                    "distance_total": 0.0,
                    "first_step": int(record.step_index),
                    "last_step": int(record.step_index),
                },
            )
            group["observation_count"] = int(group["observation_count"]) + 1
            group["direction_match_count"] = int(group["direction_match_count"]) + 1
            group["alignment_total"] = float(group["alignment_total"]) + float(alignment["axis_delta"])
            group["distance_total"] = float(group["distance_total"]) + float(displacement.get("distance", 0.0) or 0.0)
            group["last_step"] = int(record.step_index)

    out: list[dict[str, Any]] = []
    for group in groups.values():
        count = max(1, int(group["observation_count"]))
        direction_count = int(group["direction_match_count"])
        out.append(
            {
                "action_name": str(group["action_name"]),
                "segment_color": int(group["segment_color"]),
                "direction": list(group["direction"]),
                "observation_count": count,
                "direction_match_count": direction_count,
                "direction_match_rate": round(_safe_rate(direction_count, count), 6),
                "mean_action_alignment": round(float(group["alignment_total"]) / float(count), 6),
                "mean_distance": round(float(group["distance_total"]) / float(count), 6),
                "first_step": int(group["first_step"]),
                "last_step": int(group["last_step"]),
            }
        )
    out.sort(
        key=lambda item: (
            int(item["direction_match_count"]),
            float(item["direction_match_rate"]),
            float(item["mean_action_alignment"]),
            int(item["last_step"]),
        ),
        reverse=True,
    )
    return out


def infer_target_candidates(records: list[ActionEffectRecord], *, limit: int = 12) -> list[dict[str, Any]]:
    """Extract frame-only target candidates from click points and changed bboxes."""

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    consistency_lookup = _segment_motion_consistency_lookup(infer_segment_motion_consistency(records))
    for record in sorted(list(records), key=lambda item: item.step_index, reverse=True):
        if (
            record.changed_bbox is not None
            and record.changed_pixel_count > 0
            and _target_candidate_bbox_allowed(record)
        ):
            center = _bbox_center_dict(record.changed_bbox)
            bbox_fraction = (
                float(record.changed_bbox.area)
                / float(max(1, int(np.prod(record.frame_shape[:2]))))
            )
            kind = "score_changed_bbox" if record.score_delta != 0 else "localized_changed_bbox"
            key = (kind, int(center["x"]), int(center["y"]), record.action_name)
            if key not in seen:
                seen.add(key)
                candidates.append(
                    {
                        "kind": kind,
                        "site": [int(center["x"]), int(center["y"])],
                        "action_name": record.action_name,
                        "step_index": int(record.step_index),
                        "score_delta": int(record.score_delta),
                        "goal_likelihood": round(_record_goal_likelihood(record), 6),
                        "changed_fraction": round(float(record.changed_fraction), 6),
                        "changed_bbox_area_fraction": round(float(bbox_fraction), 6),
                        "changed_bbox": record.changed_bbox.to_dict(),
                    }
                )
        for displacement in list(record.segment_displacements or []):
            if not isinstance(displacement, dict):
                continue
            post_centroid = displacement.get("post_centroid")
            pre_centroid = displacement.get("pre_centroid")
            if not isinstance(post_centroid, (list, tuple)) or len(post_centroid) < 2:
                continue
            try:
                x = int(round(float(post_centroid[0])))
                y = int(round(float(post_centroid[1])))
            except (TypeError, ValueError):
                continue
            kind = "score_segment_displacement" if record.score_delta != 0 else "segment_displacement"
            key = (kind, x, y, record.action_name)
            if key in seen:
                continue
            seen.add(key)
            goal_likelihood = round(_record_goal_likelihood(record), 6)
            if goal_likelihood >= 0.0:
                goal_likelihood = max(0.08, goal_likelihood)
            dx = float(displacement.get("dx", 0.0) or 0.0)
            dy = float(displacement.get("dy", 0.0) or 0.0)
            alignment = _direction_alignment(record.action_name, dx, dy)
            direction = _direction_sign(dx, dy)
            consistency = consistency_lookup.get(
                (_normalize_action_name(record.action_name), int(displacement.get("color", 0) or 0), direction[0], direction[1]),
                {},
            )
            candidate: dict[str, Any] = {
                "kind": kind,
                "site": [x, y],
                "action_name": record.action_name,
                "step_index": int(record.step_index),
                "score_delta": int(record.score_delta),
                "goal_likelihood": goal_likelihood,
                "changed_fraction": round(float(record.changed_fraction), 6),
                "relation_kind": "moved_segment_post_centroid",
                "segment_color": int(displacement.get("color", 0) or 0),
                "dx": dx,
                "dy": dy,
                "distance": float(displacement.get("distance", 0.0) or 0.0),
                "action_alignment": round(float(alignment["axis_delta"]), 6),
                "perpendicular_delta": round(float(alignment["perpendicular_delta"]), 6),
                "direction_match": bool(alignment["direction_match"]),
                "consistency_count": int(consistency.get("direction_match_count", 1 if alignment["direction_match"] else 0) or 0),
                "direction_match_rate": float(consistency.get("direction_match_rate", 1.0 if alignment["direction_match"] else 0.0) or 0.0),
            }
            if isinstance(pre_centroid, (list, tuple)) and len(pre_centroid) >= 2:
                try:
                    candidate["source_site"] = [
                        int(round(float(pre_centroid[0]))),
                        int(round(float(pre_centroid[1]))),
                    ]
                except (TypeError, ValueError):
                    pass
            candidates.append(candidate)
        if record.action_name in CLICK_ACTIONS:
            x_value = record.action_payload.get("x")
            y_value = record.action_payload.get("y")
            try:
                x = int(x_value)
                y = int(y_value)
            except (TypeError, ValueError):
                continue
            key = ("click_point", x, y, record.action_name)
            if key not in seen:
                seen.add(key)
                candidates.append(
                    {
                        "kind": "click_point",
                        "site": [x, y],
                        "action_name": record.action_name,
                        "step_index": int(record.step_index),
                        "score_delta": int(record.score_delta),
                        "goal_likelihood": round(_record_goal_likelihood(record), 6),
                        "changed_fraction": round(float(record.changed_fraction), 6),
                    }
                )

    def _rank(candidate: dict[str, Any]) -> tuple[float, int, int, int, float, int, float, float]:
        kind_bonus = 2 if candidate.get("kind") == "score_changed_bbox" else 0
        score_delta = int(candidate.get("score_delta", 0) or 0)
        goal_likelihood = float(candidate.get("goal_likelihood", 0.0) or 0.0)
        direction_match = 1 if bool(candidate.get("direction_match", False)) else 0
        consistency_count = int(candidate.get("consistency_count", 0) or 0)
        direction_match_rate = float(candidate.get("direction_match_rate", 0.0) or 0.0)
        action_alignment = float(candidate.get("action_alignment", 0.0) or 0.0)
        changed_fraction = float(candidate.get("changed_fraction", 0.0) or 0.0)
        step_index = int(candidate.get("step_index", 0) or 0)
        return (
            goal_likelihood,
            kind_bonus + min(3, max(0, score_delta)),
            direction_match,
            consistency_count,
            direction_match_rate,
            step_index,
            action_alignment,
            changed_fraction,
        )

    candidates.sort(key=_rank, reverse=True)
    return candidates[: max(0, int(limit))]


class PrimitiveClassifier:
    """Heuristic classifier over action-effect summaries.

    The output is a selector hint, not a solver. Later phases can replace these
    heuristics with learned or causal classifiers while keeping the same
    evidence contract.
    """

    def classify(self, summary: ActionEffectSummary | list[ActionEffectRecord]) -> PrimitiveClassification:
        if isinstance(summary, list):
            tracker = ActionEffectTracker(max_records=max(1, len(summary)))
            tracker._records = list(summary)
            tracker.record_count = len(summary)
            summary_obj = tracker.summary()
        else:
            summary_obj = summary

        evidence_values = _effect_family_evidence(summary_obj)
        dir_changed_rate = float(evidence_values["directional_changed_rate"])
        dir_localized_rate = float(evidence_values["directional_localized_rate"])
        click_changed_rate = float(evidence_values["click_changed_rate"])
        click_score_rate = float(evidence_values["click_score_rate"])
        click_avg_fraction = float(evidence_values["click_avg_changed_fraction"])
        changed_action_count = int(evidence_values["changed_action_count"])
        action_count = int(evidence_values["action_count"])
        non_direction_changed = list(evidence_values["non_direction_changed_actions"])
        score_action_count = int(evidence_values["score_action_count"])
        object_continuity = dict(evidence_values.get("object_continuity") or {})
        target_candidates = [dict(candidate) for candidate in evidence_values.get("target_candidates") or []]
        goal_transition_profile = dict(evidence_values.get("goal_transition_profile") or {})

        movement_score = min(1.0, 0.55 * dir_changed_rate + 0.45 * dir_localized_rate)
        click_toggle_score = min(
            1.0,
            0.55 * click_changed_rate + 0.25 * min(1.0, click_avg_fraction / 0.08) + 0.20 * click_score_rate,
        )
        score_action_score = min(1.0, float(score_action_count) / max(1.0, float(action_count)))
        multi_action_score = float(evidence_values["multi_action_score"])
        low_effect_score = summary_obj.overall_no_op_rate
        localized_transition_score = min(1.0, dir_localized_rate + 0.25 * click_changed_rate)
        solver_archetype_scores = infer_solver_archetypes(summary_obj)
        solver_archetype_label, solver_archetype_confidence = max(
            solver_archetype_scores.items(),
            key=lambda item: item[1],
        )

        scores = {
            "movement_game": movement_score,
            "click_toggle_game": click_toggle_score,
            "score_action_game": score_action_score,
            "multi_action_transition_game": multi_action_score,
            "localized_transition_game": localized_transition_score,
            "low_effect_or_noop": low_effect_score,
        }

        primary_label, confidence = max(scores.items(), key=lambda item: item[1])
        if summary_obj.records_count == 0:
            primary_label = "insufficient_evidence"
            confidence = 0.0
        elif confidence < 0.35:
            primary_label = "unknown_transition_game"

        return PrimitiveClassification(
            primary_label=primary_label,
            confidence=float(confidence),
            scores={key: float(value) for key, value in sorted(scores.items())},
            evidence={
                "records_count": summary_obj.records_count,
                "changed_actions": list(summary_obj.changed_actions),
                "score_actions": list(summary_obj.score_actions),
                "directional_changed_rate": dir_changed_rate,
                "directional_localized_rate": dir_localized_rate,
                "click_changed_rate": click_changed_rate,
                "click_avg_changed_fraction": click_avg_fraction,
                "non_direction_changed_actions": non_direction_changed,
                "overall_no_op_rate": summary_obj.overall_no_op_rate,
                "object_continuity": object_continuity,
                "target_candidates": target_candidates,
                "goal_transition_profile": goal_transition_profile,
                "solver_archetype_label": solver_archetype_label,
                "solver_archetype_confidence": solver_archetype_confidence,
                "solver_archetype_scores": {
                    key: float(value) for key, value in sorted(solver_archetype_scores.items())
                },
            },
        )


__all__ = [
    "ActionEffectRecord",
    "ActionEffectStats",
    "ActionEffectSummary",
    "ActionEffectTracker",
    "ChangedBBox",
    "PrimitiveClassification",
    "PrimitiveClassifier",
    "effect_event_from_record",
    "frame_state_hash",
    "observed_effect_from_record",
    "record_state_signatures",
    "infer_segment_motion_consistency",
    "infer_object_continuity",
    "infer_goal_transition_profile",
    "infer_solver_archetypes",
    "infer_target_candidates",
    "summarize_action_effect_records",
]
