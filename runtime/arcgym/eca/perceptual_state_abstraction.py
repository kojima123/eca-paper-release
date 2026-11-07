"""Perceptual / predicate / state abstraction candidates for Lane C.

The PSA layer turns visible action-effect evidence into typed state-variable,
predicate, operator, and goal-link candidates.  It intentionally does not
execute actions and does not know game adapters; ECA decides whether to try the
resulting mechanic hypothesis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from .action_effect_model import ActionEffectRecord


@dataclass(frozen=True)
class PSAAbstraction:
    abstraction_id: str
    family: str
    source_kind: str = "psa_induced"
    source_event_ids: tuple[str, ...] = ()
    state_variables: tuple[dict[str, Any], ...] = ()
    predicate_candidates: tuple[dict[str, Any], ...] = ()
    operator_candidates: tuple[dict[str, Any], ...] = ()
    goal_link_candidates: tuple[dict[str, Any], ...] = ()
    control_sites: tuple[dict[str, Any], ...] = ()
    controlled_objects: tuple[dict[str, Any], ...] = ()
    proposed_plan: tuple[dict[str, Any], ...] = ()
    evidence_frames: tuple[int, ...] = ()
    effect_confidence: float = 0.0
    predicate_confidence: float = 0.0
    contradiction_count: int = 0
    llm_role: str = "none"
    llm_direct_action: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        base = min(float(self.effect_confidence), float(self.predicate_confidence))
        if self.contradiction_count <= 0:
            return round(max(0.0, min(1.0, base)), 6)
        penalty = min(0.75, 0.20 * float(self.contradiction_count))
        return round(max(0.0, min(1.0, base - penalty)), 6)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = self.confidence
        return _sanitize(payload)


class PerceptualStateAbstractionInducer:
    """Infer conservative PSA candidates from frame-only effect records."""

    def build(
        self,
        *,
        records: Iterable[ActionEffectRecord] = (),
        frame_index: int | None = None,
        source_kind: str = "psa_induced",
    ) -> list[PSAAbstraction]:
        record_list = list(records)
        groups: dict[tuple[str, int, int], list[ActionEffectRecord]] = {}
        for record in record_list:
            if record.no_op or int(record.changed_pixel_count) <= 0:
                continue
            action_name = str(record.action_name or "")
            payload = dict(record.action_payload or {})
            if action_name != "ACTION6" or "x" not in payload or "y" not in payload:
                continue
            try:
                key = (action_name, int(payload["x"]), int(payload["y"]))
            except (TypeError, ValueError):
                continue
            groups.setdefault(key, []).append(record)

        abstractions: list[PSAAbstraction] = []
        for (action_name, x, y), items in sorted(groups.items()):
            latest = max(items, key=lambda item: int(item.step_index))
            support = len(items)
            score_events = sum(1 for item in items if int(item.score_delta) > 0)
            changed_total = sum(int(item.changed_pixel_count) for item in items)
            bbox = latest.changed_bbox.to_dict() if latest.changed_bbox is not None else {}
            state_variable = {
                "variable_id": f"psa:visible_state_hash:{action_name}:{x},{y}",
                "variable_type": "visible_state_hash_transition",
                "pre_state_hash": str(latest.pre_state_hash),
                "post_state_hash": str(latest.post_state_hash),
                "control_site": [int(x), int(y)],
                "changed_bbox": bbox,
                "confidence": round(min(0.95, 0.58 + 0.08 * min(3, support)), 6),
            }
            predicate_kind = "score_delta" if score_events > 0 else "visible_state_advance"
            predicate = {
                "predicate_id": f"psa:predicate:{predicate_kind}:{action_name}:{x},{y}",
                "predicate_kind": predicate_kind,
                "score_delta": int(max(0, latest.score_delta)),
                "post_state_hash": str(latest.post_state_hash),
                "confidence": round(
                    min(0.9, 0.55 + 0.12 * min(2, score_events) + 0.03 * min(3, support - 1)),
                    6,
                ),
            }
            operator = {
                "operator_id": f"psa:operator:repeat_effective_click:{action_name}:{x},{y}",
                "operator_family": "click_operator",
                "operation_kind": "repeat_effective_click",
                "action_name": action_name,
                "payload": {"x": int(x), "y": int(y)},
                "support_count": int(support),
                "changed_pixel_total": int(changed_total),
            }
            goal_link = {
                "goal_link_id": f"psa:goal_link:{predicate_kind}:{action_name}:{x},{y}",
                "goal_link_type": predicate_kind,
                "predicted_goal_delta": float(max(0, score_events)),
                "predicate_id": predicate["predicate_id"],
                "operator_id": operator["operator_id"],
                "confidence": predicate["confidence"],
            }
            abstractions.append(
                PSAAbstraction(
                    abstraction_id=f"psa:{action_name}:{x}:{y}:{latest.post_state_hash[:8]}",
                    family="cycle",
                    source_kind=str(source_kind or "psa_induced"),
                    source_event_ids=_source_event_ids(items, actions=[action_name]),
                    state_variables=(state_variable,),
                    predicate_candidates=(predicate,),
                    operator_candidates=(operator,),
                    goal_link_candidates=(goal_link,),
                    control_sites=({"x": int(x), "y": int(y), "action_name": action_name},),
                    controlled_objects=(
                        {
                            "kind": "changed_region",
                            "x": int(bbox.get("min_x", x)),
                            "y": int(bbox.get("min_y", y)),
                        },
                    ),
                    proposed_plan=({"action_name": action_name, "payload": {"x": int(x), "y": int(y)}},),
                    evidence_frames=() if frame_index is None else (int(frame_index),),
                    effect_confidence=round(
                        min(0.95, 0.58 + 0.08 * min(3, support) + 0.04 * min(4, changed_total // 4)),
                        6,
                    ),
                    predicate_confidence=float(predicate["confidence"]),
                    metadata={
                        "records_seen": len(record_list),
                        "support_count": int(support),
                        "score_event_count": int(score_events),
                        "changed_pixel_total": int(changed_total),
                    },
                )
            )
        abstractions.sort(key=lambda item: (-item.confidence, item.abstraction_id))
        return abstractions[:4]


def _source_event_ids(
    records: list[ActionEffectRecord],
    *,
    actions: Iterable[str],
) -> tuple[str, ...]:
    action_set = {str(action or "") for action in actions if str(action or "")}
    out: list[str] = []
    for record in records:
        if action_set and str(record.action_name) not in action_set:
            continue
        out.append(f"action_effect:{int(record.step_index)}:{str(record.action_name)}")
        if len(out) >= 8:
            break
    return tuple(out)


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


__all__ = [
    "PSAAbstraction",
    "PerceptualStateAbstractionInducer",
]
