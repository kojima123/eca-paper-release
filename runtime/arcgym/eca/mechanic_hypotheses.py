"""Mechanic-level hypotheses induced from visible state and probe evidence.

This module sits between frame/effect evidence and hand-written solver
families.  The first implementation deliberately targets cycle-state
constraints because ft09/lp85 already expose a compact state form that can be
compiled into a short plan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from .action_effect_model import ActionEffectRecord, summarize_action_effect_records
from .perceptual_state_abstraction import PSAAbstraction, PerceptualStateAbstractionInducer


ABSTRACTION_SOURCE_KINDS = {
    "psa_induced",
    "psa_symbolic_induced",
    "psa_llm_proposed",
    "eca_llm_interpreted",
    "ledger_confirmed",
    "mixed",
}


@dataclass(frozen=True)
class MechanicHypothesis:
    hypothesis_id: str
    family: str
    source_kind: str = "reader_fixed"
    source_event_ids: tuple[str, ...] = ()
    induced_variables: tuple[str, ...] = ()
    evidence_frames: list[int] = field(default_factory=list)
    controlled_objects: list[dict[str, Any]] = field(default_factory=list)
    control_sites: list[dict[str, Any]] = field(default_factory=list)
    current_state: dict[str, Any] = field(default_factory=dict)
    target_state: dict[str, Any] | None = None
    predicted_operator: dict[str, Any] | None = None
    goal_link_confidence: float = 0.0
    effect_confidence: float = 0.0
    contradiction_count: int = 0
    proposed_plan: list[dict[str, Any]] = field(default_factory=list)
    expected_goal_delta: float = 0.0
    source: str = "mechanic_hypothesis_inducer"
    status: str = "tentative"
    abort_conditions: list[str] = field(default_factory=list)
    falsification_conditions: tuple[str, ...] = ()
    compiler_family: str = ""
    executor_family: str = ""
    llm_role: str = "none"
    llm_direct_action: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        base = min(float(self.goal_link_confidence), float(self.effect_confidence))
        if self.contradiction_count <= 0:
            return round(max(0.0, min(1.0, base)), 6)
        penalty = min(0.75, 0.20 * float(self.contradiction_count))
        return round(max(0.0, min(1.0, base - penalty)), 6)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = self.confidence
        return _sanitize_payload(payload)


class MechanicHypothesisInducer:
    """Build conservative mechanic hypotheses from current visible evidence."""

    def build(
        self,
        *,
        cycle_states: Mapping[str, Mapping[str, Any] | None] | None = None,
        records: Iterable[ActionEffectRecord] = (),
        frame_index: int | None = None,
        rod_support: Mapping[str, Any] | None = None,
        enable_ledger_induction: bool = False,
        enable_psa_induction: bool = False,
        psa_abstractions: Iterable[PSAAbstraction] = (),
        external_emergent_hypotheses: Iterable["MechanicHypothesis"] = (),
    ) -> list[MechanicHypothesis]:
        record_list = list(records)
        explicit_psa_abstractions = list(psa_abstractions)
        hypotheses: list[MechanicHypothesis] = []
        for primitive_id, state in sorted((cycle_states or {}).items()):
            hypothesis = self._cycle_hypothesis(
                primitive_id=str(primitive_id),
                state=state,
                records=record_list,
                frame_index=frame_index,
                rod_support=rod_support or {},
            )
            if hypothesis is not None:
                hypotheses.append(hypothesis)
        if enable_ledger_induction:
            hypotheses.extend(
                self._ledger_induced_transition_hypotheses(
                    records=record_list,
                    frame_index=frame_index,
                    rod_support=rod_support or {},
                )
            )
        psa_candidates: list[PSAAbstraction] = []
        if enable_psa_induction:
            psa_candidates.extend(
                PerceptualStateAbstractionInducer().build(
                    records=record_list,
                    frame_index=frame_index,
                )
            )
        psa_candidates.extend(explicit_psa_abstractions)
        if psa_candidates:
            hypotheses.extend(
                self._psa_induced_transition_hypotheses(
                    abstractions=psa_candidates,
                    frame_index=frame_index,
                    rod_support=rod_support or {},
                )
            )
        # Gap 6 opt-in: callers may inject pre-built emergent_pattern_induced
        # hypotheses (e.g. from arcgym.eca.emergent_candidate_injector). They
        # are inserted alongside reader/ledger/PSA-derived ones and ranked by
        # the same confidence sort below. The injection happens here so the
        # external candidates are subject to the SAME downstream HypothesisPlan
        # Compiler / executor / OutcomeLedger flow as any other source_kind.
        for ext in external_emergent_hypotheses:
            if isinstance(ext, MechanicHypothesis):
                hypotheses.append(ext)
        hypotheses.sort(key=lambda item: (-item.confidence, item.family, item.hypothesis_id))
        return hypotheses

    def _cycle_hypothesis(
        self,
        *,
        primitive_id: str,
        state: Mapping[str, Any] | None,
        records: list[ActionEffectRecord],
        frame_index: int | None,
        rod_support: Mapping[str, Any],
    ) -> MechanicHypothesis | None:
        if not state:
            return None
        mechanic_family = str(state.get("mechanic_family", "") or "")
        if mechanic_family != "cycle_state_constraint":
            return None
        plan = _normalise_plan(state.get("operation_plan") or [])
        if not plan:
            return None
        mechanic_state_type = str(state.get("mechanic_state_type", "cycle_state") or "cycle_state")
        goal_link_type = str(state.get("goal_link_type", "unknown_goal_link") or "unknown_goal_link")
        action_names = [str(step.get("action_name", "") or "") for step in plan]
        goal_link_confidence = float(state.get("goal_link_confidence", 0.0) or 0.0)
        reader_confidence = float(state.get("reader_confidence", 0.0) or 0.0)
        effect_confidence = _effect_confidence(records, action_names, fallback=reader_confidence)
        contradiction_count = _rod_contradiction_count(rod_support, primitive_id)
        confidence_floor = min(goal_link_confidence, effect_confidence)
        status = "active" if confidence_floor >= 0.65 and contradiction_count == 0 else "tentative"
        if contradiction_count >= 2:
            status = "refuted"
        return MechanicHypothesis(
            hypothesis_id=f"{primitive_id}:{mechanic_state_type}:{goal_link_type}",
            family="cycle",
            source_kind="reader_fixed",
            source_event_ids=_source_event_ids(records, actions=action_names),
            induced_variables=_cycle_induced_variables(state),
            evidence_frames=[] if frame_index is None else [int(frame_index)],
            controlled_objects=_cycle_controlled_objects(state),
            control_sites=_cycle_control_sites(plan),
            current_state=_cycle_current_state(state),
            target_state=_cycle_target_state(state),
            predicted_operator=_cycle_predicted_operator(state, plan),
            goal_link_confidence=goal_link_confidence,
            effect_confidence=effect_confidence,
            contradiction_count=contradiction_count,
            proposed_plan=plan,
            expected_goal_delta=float(state.get("expected_goal_delta", 0.0) or 0.0),
            status=status,
            abort_conditions=[
                "unexpected_visible_effect",
                "target_state_stale",
                "goal_mismatch_not_decreasing",
                "operator_direction_contradicted",
            ],
            falsification_conditions=(
                "same_source_reader_absent",
                "operator_action_no_longer_changes_state",
                "target_state_shuffle_preserves_behavior",
                "stale_post_state_compiles_same_plan",
            ),
            compiler_family="cycle_hypothesis_plan_compiler",
            executor_family="mechanic_hypothesis_cycle_executor",
            evidence={
                "primitive_id": primitive_id,
                "source_kind": "reader_fixed",
                "mechanic_state_type": mechanic_state_type,
                "goal_link_type": goal_link_type,
                "required_primitive_family": str(state.get("required_primitive_family", "") or ""),
                "cycle_operation_kind": str(state.get("cycle_operation_kind", "") or ""),
                "reader_confidence": reader_confidence,
                "records_seen": len(records),
            },
        )

    def _ledger_induced_transition_hypotheses(
        self,
        *,
        records: list[ActionEffectRecord],
        frame_index: int | None,
        rod_support: Mapping[str, Any],
    ) -> list[MechanicHypothesis]:
        del rod_support  # reserved for posterior suppression in Phase 166.
        groups: dict[tuple[str, int, int], list[ActionEffectRecord]] = {}
        for record in records:
            if record.no_op or record.changed_pixel_count <= 0:
                continue
            action = str(record.action_name or "")
            payload = dict(record.action_payload or {})
            if action != "ACTION6" or "x" not in payload or "y" not in payload:
                continue
            key = (action, int(payload["x"]), int(payload["y"]))
            groups.setdefault(key, []).append(record)

        hypotheses: list[MechanicHypothesis] = []
        for (action, x, y), items in sorted(groups.items()):
            latest = max(items, key=lambda item: int(item.step_index))
            support = len(items)
            score_events = sum(1 for item in items if int(item.score_delta) != 0)
            changed_total = sum(int(item.changed_pixel_count) for item in items)
            effect_confidence = min(
                0.95,
                0.58 + 0.08 * min(3, support) + 0.04 * min(4, changed_total // 4),
            )
            goal_link_confidence = min(
                0.82,
                0.55 + 0.12 * min(2, score_events) + 0.03 * min(3, support - 1),
            )
            bbox = latest.changed_bbox.to_dict() if latest.changed_bbox is not None else {}
            source_event_ids = _source_event_ids(items, actions=[action])
            hypotheses.append(
                MechanicHypothesis(
                    hypothesis_id=f"ledger:{action}:{x}:{y}:{latest.post_state_hash[:8]}",
                    family="cycle",
                    source_kind="ledger_induced",
                    source_event_ids=source_event_ids,
                    induced_variables=(
                        "control_site",
                        "pre_state_hash",
                        "post_state_hash",
                        "operator_delta",
                        "effect_signature",
                    ),
                    evidence_frames=[] if frame_index is None else [int(frame_index)],
                    controlled_objects=[
                        {
                            "kind": "changed_region",
                            "x": int(bbox.get("min_x", x)),
                            "y": int(bbox.get("min_y", y)),
                        }
                    ],
                    control_sites=[{"x": int(x), "y": int(y), "action_name": action}],
                    current_state={
                        "mechanic_state_type": "ledger_induced_click_transition",
                        "pre_state_hash": latest.pre_state_hash,
                        "action_name": action,
                        "control_site": [int(x), int(y)],
                    },
                    target_state={
                        "goal_link_type": "ledger_visible_state_advance",
                        "post_state_hash": latest.post_state_hash,
                        "score_delta": int(latest.score_delta),
                        "changed_bbox": bbox,
                    },
                    predicted_operator={
                        "operator_family": "ledger_transition_replay",
                        "operation_kind": "repeat_effective_click",
                        "action_names": [action],
                        "operation_count": 1,
                    },
                    goal_link_confidence=goal_link_confidence,
                    effect_confidence=effect_confidence,
                    contradiction_count=0,
                    proposed_plan=[{"action_name": action, "payload": {"x": int(x), "y": int(y)}}],
                    expected_goal_delta=float(max(0, score_events)),
                    status="active",
                    abort_conditions=[
                        "same_click_no_longer_changes_state",
                        "post_state_hash_repeats_without_progress",
                        "stale_post_state_compiles_same_plan",
                    ],
                    falsification_conditions=(
                        "ledger_event_removed",
                        "same_site_click_noop",
                        "shuffled_control_site_preserves_behavior",
                        "stale_post_state_compiles_same_plan",
                    ),
                    compiler_family="ledger_transition_hypothesis_plan_compiler",
                    executor_family="mechanic_hypothesis_cycle_executor",
                    evidence={
                        "primitive_id": "ledger_induced_transition",
                        "source_kind": "ledger_induced",
                        "mechanic_state_type": "ledger_induced_click_transition",
                        "goal_link_type": "ledger_visible_state_advance",
                        "records_seen": len(records),
                        "support_count": support,
                        "score_event_count": score_events,
                        "changed_pixel_total": changed_total,
                    },
                )
            )
        return hypotheses[:4]

    def _psa_induced_transition_hypotheses(
        self,
        *,
        abstractions: list[PSAAbstraction],
        frame_index: int | None,
        rod_support: Mapping[str, Any],
    ) -> list[MechanicHypothesis]:
        del rod_support  # reserved for PSA posterior in a later phase.
        hypotheses: list[MechanicHypothesis] = []
        for abstraction in abstractions:
            if abstraction.family not in {"cycle", "coverage", "reflection"} or not abstraction.proposed_plan:
                continue
            variable = abstraction.state_variables[0] if abstraction.state_variables else {}
            predicate = abstraction.predicate_candidates[0] if abstraction.predicate_candidates else {}
            operator = abstraction.operator_candidates[0] if abstraction.operator_candidates else {}
            goal_link = abstraction.goal_link_candidates[0] if abstraction.goal_link_candidates else {}
            control_site = abstraction.control_sites[0] if abstraction.control_sites else {}
            metadata = dict(abstraction.metadata or {})
            repair_id = str(
                metadata.get("psa_cycle_goal_link_repair_id", "")
                or metadata.get("psa_structural_goal_link_repair_id", "")
                or goal_link.get("repair_id", "")
                or ""
            )
            goal_link_id = str(
                metadata.get("psa_cycle_goal_link_id", "")
                or metadata.get("psa_structural_goal_link_id", "")
                or goal_link.get("goal_link_id", "")
                or ""
            )
            protected_plan_id = str(
                metadata.get("psa_cycle_protected_plan_id", "")
                or metadata.get("psa_structural_protected_plan_id", "")
                or goal_link.get("protected_plan_id", "")
                or operator.get("protected_plan_id", "")
                or ""
            )
            protected_plan_len = int(
                metadata.get("visible_cycle_protected_plan_len", 0)
                or metadata.get("visible_structural_protected_plan_len", 0)
                or len(abstraction.proposed_plan)
                or 0
            )
            repair_control_mode = str(metadata.get("psa_cycle_goal_link_repair_control_mode", "") or "")
            force_wrong_site = metadata.get("psa_cycle_force_wrong_site_after_first_step")
            if not isinstance(force_wrong_site, Mapping):
                force_wrong_site = {}
            force_stale_hash = str(metadata.get("psa_cycle_force_required_current_state_hash", "") or "")
            try:
                x = int(control_site.get("x"))
                y = int(control_site.get("y"))
            except (AttributeError, TypeError, ValueError):
                continue
            first_step = abstraction.proposed_plan[0] if abstraction.proposed_plan else {}
            first_step = first_step if isinstance(first_step, Mapping) else {}
            action_name = str(
                first_step.get("action_name", "")
                or control_site.get("action_name", "")
                or operator.get("action_name", "")
                or "ACTION6"
            )
            score_delta = int(predicate.get("score_delta", 0) or 0)
            source_kind = str(abstraction.source_kind or "psa_induced")
            mechanic_state_type = str(
                variable.get("mechanic_state_type", "")
                or (
                    "psa_visible_state_variable"
                    if abstraction.family == "cycle"
                    else f"psa_{abstraction.family}_state_variable"
                )
            )
            hypotheses.append(
                MechanicHypothesis(
                    hypothesis_id=f"{abstraction.abstraction_id}:state_variable",
                    family=str(abstraction.family or "cycle"),
                    source_kind=source_kind,
                    source_event_ids=tuple(abstraction.source_event_ids),
                    induced_variables=(
                        "psa_state_variable",
                        "psa_predicate_candidate",
                        "psa_operator_candidate",
                        "psa_goal_link_candidate",
                    ),
                    evidence_frames=(
                        list(abstraction.evidence_frames)
                        if abstraction.evidence_frames
                        else ([] if frame_index is None else [int(frame_index)])
                    ),
                    controlled_objects=[dict(item) for item in abstraction.controlled_objects],
                    control_sites=[dict(item) for item in abstraction.control_sites],
                    current_state={
                        "mechanic_state_type": mechanic_state_type,
                        "state_variable_id": str(variable.get("variable_id", "") or ""),
                        "variable_type": str(variable.get("variable_type", "") or ""),
                        "pre_state_hash": str(variable.get("pre_state_hash", "") or ""),
                        "action_name": action_name,
                        "control_site": [int(x), int(y)],
                    },
                    target_state={
                        "goal_link_type": str(goal_link.get("goal_link_type", "") or "visible_state_advance"),
                        "goal_link_id": goal_link_id,
                        "predicate_id": str(predicate.get("predicate_id", "") or ""),
                        "post_state_hash": str(variable.get("post_state_hash", "") or predicate.get("post_state_hash", "") or ""),
                        "score_delta": score_delta,
                        "changed_bbox": _sanitize_payload(variable.get("changed_bbox", {})),
                    },
                    predicted_operator={
                        "operator_family": str(operator.get("operator_family", "") or "click_operator"),
                        "operation_kind": str(operator.get("operation_kind", "") or "repeat_effective_click"),
                        "operator_id": str(operator.get("operator_id", "") or ""),
                        "repair_id": repair_id,
                        "psa_structural_repair_posterior_key": str(
                            metadata.get("psa_structural_repair_posterior_key", "") or ""
                        ),
                        "goal_link_id": goal_link_id,
                        "protected_plan_id": protected_plan_id,
                        "repair_control_mode": repair_control_mode,
                        "force_wrong_site_after_first_step": dict(force_wrong_site),
                        "force_required_current_state_hash": force_stale_hash,
                        "action_names": [action_name],
                        "operation_count": 1,
                    },
                    goal_link_confidence=float(abstraction.predicate_confidence),
                    effect_confidence=float(abstraction.effect_confidence),
                    contradiction_count=int(abstraction.contradiction_count),
                    proposed_plan=[dict(step) for step in abstraction.proposed_plan],
                    expected_goal_delta=float(goal_link.get("predicted_goal_delta", 0.0) or 0.0),
                    status="active" if abstraction.confidence >= 0.55 else "tentative",
                    abort_conditions=[
                        "psa_operator_no_longer_changes_state",
                        "psa_predicate_state_stale",
                        "psa_goal_delta_not_observed",
                    ],
                    falsification_conditions=(
                        "psa_state_variable_removed",
                        "psa_control_site_shuffled",
                        "psa_operator_reversed",
                        "stale_post_state_compiles_same_plan",
                    ),
                    compiler_family="psa_state_abstraction_plan_compiler",
                    executor_family="mechanic_hypothesis_cycle_executor",
                    llm_role=str(abstraction.llm_role or "none"),
                    llm_direct_action=bool(abstraction.llm_direct_action),
                    evidence={
                        "primitive_id": "psa_state_abstraction",
                        "source_kind": source_kind,
                        "abstraction_id": abstraction.abstraction_id,
                        "pre_repair_source_kind": str(metadata.get("pre_repair_source_kind", "") or ""),
                        "repair_id": repair_id,
                        "goal_link_id": goal_link_id,
                        "protected_plan_id": protected_plan_id,
                        "protected_plan_len": protected_plan_len,
                        "repair_control_mode": repair_control_mode,
                        "force_wrong_site_after_first_step": dict(force_wrong_site),
                        "force_required_current_state_hash": force_stale_hash,
                        "mechanic_state_type": mechanic_state_type,
                        "goal_link_type": str(goal_link.get("goal_link_type", "") or ""),
                        "records_seen": int(metadata.get("records_seen", 0) or 0),
                        "support_count": int(metadata.get("support_count", 0) or 0),
                        "score_event_count": int(metadata.get("score_event_count", 0) or 0),
                        "llm_role": str(abstraction.llm_role or "none"),
                        "llm_direct_action": bool(abstraction.llm_direct_action),
                    },
                )
            )
        return hypotheses[:4]


class HypothesisPlanCompiler:
    """Compile induced mechanic hypotheses into one guarded macro step."""

    def compile(
        self,
        hypotheses: Iterable[MechanicHypothesis],
        *,
        available_actions: Iterable[str],
        min_confidence: float = 0.55,
    ) -> dict[str, Any] | None:
        available = {str(action or "") for action in available_actions}
        for hypothesis in sorted(hypotheses, key=lambda item: (-item.confidence, item.hypothesis_id)):
            if hypothesis.status == "refuted":
                continue
            if hypothesis.llm_direct_action:
                continue
            if hypothesis.confidence < _protected_probe_min_confidence(
                hypothesis,
                default_min_confidence=float(min_confidence),
            ):
                continue
            if hypothesis.family not in {"cycle", "coverage", "reflection", "emergent"}:
                continue
            for step in hypothesis.proposed_plan:
                action_name = str(step.get("action_name", "") or "")
                payload = step.get("payload") if isinstance(step.get("payload"), Mapping) else {}
                if action_name not in available:
                    continue
                source_primitive = str(hypothesis.evidence.get("primitive_id", "") or "")
                repair_id = str(hypothesis.evidence.get("repair_id", "") or "")
                goal_link_id = str(hypothesis.evidence.get("goal_link_id", "") or "")
                protected_plan_id = str(hypothesis.evidence.get("protected_plan_id", "") or "")
                return {
                    "primitive_id": "mechanic_hypothesis_cycle_executor",
                    "source_primitive_id": source_primitive,
                    "source_kind": hypothesis.source_kind,
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "posterior_key": mechanic_hypothesis_posterior_key(hypothesis),
                    "repair_id": repair_id,
                    "goal_link_id": goal_link_id,
                    "protected_plan_id": protected_plan_id,
                    "protected_plan_len": int(hypothesis.evidence.get("protected_plan_len", 0) or len(hypothesis.proposed_plan)),
                    "protected_plan_step_index": 0,
                    "family": hypothesis.family,
                    "llm_role": hypothesis.llm_role,
                    "llm_direct_action": bool(hypothesis.llm_direct_action),
                    "step": (
                        action_name,
                        {key: int(value) for key, value in dict(payload).items() if key in {"x", "y"}},
                    ),
                    "hypothesis": hypothesis,
                    "reason": (
                        "mechanic_hypothesis_cycle_executor compiled induced "
                        f"{hypothesis.hypothesis_id}"
                    ),
                }
        return None


def _protected_probe_min_confidence(
    hypothesis: MechanicHypothesis,
    *,
    default_min_confidence: float,
) -> float:
    """Return the compiler threshold for a protected hypothesis probe.

    LLM-derived candidates are still schema constrained and ledger verified.
    Allow them to enter the protected probe executor at 0.50 when they carry
    concrete source evidence; score-bearing adoption still depends on the
    later OutcomeLedger result.
    """

    if (
        str(hypothesis.llm_role or "none") != "none"
        and hypothesis.source_event_ids
        and not bool(hypothesis.llm_direct_action)
    ):
        return min(float(default_min_confidence), 0.50)
    return float(default_min_confidence)


def mechanic_hypothesis_posterior_key(hypothesis: MechanicHypothesis) -> str:
    """Stable key for outcome posterior gating.

    Ledger-induced hypotheses should generalise over post-state hashes so the
    same effective site can be confirmed or suppressed across repeated probes.
    Reader-fixed hypotheses remain keyed by their explicit reader hypothesis.
    """

    source_kind = str(hypothesis.source_kind or "unknown")
    family = str(hypothesis.family or "unknown")
    if source_kind == "ledger_induced" or source_kind in ABSTRACTION_SOURCE_KINDS:
        action_name = ""
        x = None
        y = None
        if hypothesis.control_sites:
            site = hypothesis.control_sites[0]
            if isinstance(site, Mapping):
                action_name = str(site.get("action_name", "") or "")
                try:
                    x = int(site.get("x"))
                    y = int(site.get("y"))
                except (TypeError, ValueError):
                    x = None
                    y = None
        if not action_name:
            action_name = str(hypothesis.current_state.get("action_name", "") or "")
        if x is None or y is None:
            site = hypothesis.current_state.get("control_site")
            if isinstance(site, (list, tuple)) and len(site) >= 2:
                try:
                    x = int(site[0])
                    y = int(site[1])
                except (TypeError, ValueError):
                    x = None
                    y = None
        operator = dict(hypothesis.predicted_operator or {})
        operator_kind = str(
            operator.get("operation_kind", "") or operator.get("operator_family", "") or "unknown_operator"
        )
        if action_name and x is not None and y is not None:
            return f"{source_kind}|{family}|{action_name}|{x},{y}|{operator_kind}"
    return f"{source_kind}|{family}|{hypothesis.hypothesis_id}"


def _normalise_plan(raw_plan: Iterable[Any]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for step in raw_plan:
        action_name = ""
        payload: Mapping[str, Any] = {}
        if isinstance(step, Mapping):
            action_name = str(step.get("action_name") or step.get("action") or "")
            payload = step.get("payload") if isinstance(step.get("payload"), Mapping) else step
        elif isinstance(step, (list, tuple)) and step:
            action_name = str(step[0] or "")
            if len(step) > 1 and isinstance(step[1], Mapping):
                payload = step[1]
        if not action_name:
            continue
        compact_payload: dict[str, int] = {}
        for key in ("x", "y"):
            if key in payload:
                compact_payload[key] = int(payload[key])
        plan.append({"action_name": action_name, "payload": compact_payload})
    return plan


def _effect_confidence(
    records: list[ActionEffectRecord],
    action_names: Iterable[str],
    *,
    fallback: float,
) -> float:
    actions = {str(action or "") for action in action_names if str(action or "")}
    if not records or not actions:
        return max(0.0, min(1.0, float(fallback)))
    summary = summarize_action_effect_records(records)
    attempts = 0
    changed = 0
    score_events = 0
    for action in actions:
        stats = summary.action_stats.get(action)
        if stats is None:
            continue
        attempts += int(stats.attempts)
        changed += int(stats.changed_count)
        score_events += int(stats.score_event_count)
    if attempts <= 0:
        return max(0.0, min(1.0, 0.5 * float(fallback)))
    changed_rate = float(changed) / float(attempts)
    score_rate = float(score_events) / float(attempts)
    return max(0.0, min(1.0, 0.70 * changed_rate + 0.20 * score_rate + 0.10 * float(fallback)))


def _rod_contradiction_count(rod_support: Mapping[str, Any], primitive_id: str) -> int:
    by_primitive = rod_support.get("by_primitive")
    if not isinstance(by_primitive, Mapping):
        return 0
    raw = by_primitive.get(primitive_id)
    if not isinstance(raw, Mapping):
        return 0
    return int(raw.get("contradicted", 0) or 0)


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


def _cycle_induced_variables(state: Mapping[str, Any]) -> tuple[str, ...]:
    variables = ["current_state", "target_state", "operator_delta"]
    mechanic_state_type = str(state.get("mechanic_state_type", "") or "")
    if mechanic_state_type:
        variables.append(mechanic_state_type)
    if "constraints" in state:
        variables.append("constraint_set")
    if "cycle_len" in state:
        variables.append("cycle_index")
    if "selected_index" in state or "target_index" in state:
        variables.append("editable_slot_index")
    return tuple(dict.fromkeys(variables))


def _cycle_controlled_objects(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in ("current_pos", "target_pos", "active_piece", "selected_center", "target_center"):
        value = state.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            out.append({"kind": key, "x": int(value[0]), "y": int(value[1])})
    for key in ("solid", "brush"):
        value = state.get(key)
        if isinstance(value, Mapping):
            for pos in list(value.keys())[:8]:
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    out.append({"kind": key, "x": int(pos[0]), "y": int(pos[1])})
    return out


def _cycle_control_sites(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for step in plan:
        payload = step.get("payload") if isinstance(step.get("payload"), Mapping) else {}
        if "x" not in payload or "y" not in payload:
            continue
        site = (int(payload["x"]), int(payload["y"]))
        if site in seen:
            continue
        seen.add(site)
        sites.append({"x": site[0], "y": site[1], "action_name": str(step.get("action_name", "") or "")})
    return sites


def _cycle_current_state(state: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "mechanic_state_type",
        "cycle_operation_kind",
        "cycle_len",
        "current_pos",
        "selected_index",
        "selected_slot_center",
        "remaining_clicks",
        "constraint_count",
        "clickable_count",
        "mismatch_count",
        "satisfied_count",
        "action_role",
    )
    return {key: _sanitize_payload(state[key]) for key in keys if key in state}


def _cycle_target_state(state: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "goal_link_type",
        "target_pos",
        "target_index",
        "target_slot_center",
        "desired_signatures",
        "constraints",
        "template_count",
    )
    return {key: _sanitize_payload(state[key]) for key in keys if key in state}


def _cycle_predicted_operator(
    state: Mapping[str, Any],
    plan: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "operator_family": "cycle_operation",
        "operation_kind": str(state.get("cycle_operation_kind", "") or ""),
        "required_primitive_family": str(state.get("required_primitive_family", "") or ""),
        "action_names": [str(step.get("action_name", "") or "") for step in plan],
        "operation_count": len(plan),
    }


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_payload(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


__all__ = [
    "ABSTRACTION_SOURCE_KINDS",
    "HypothesisPlanCompiler",
    "MechanicHypothesis",
    "MechanicHypothesisInducer",
    "mechanic_hypothesis_posterior_key",
]
