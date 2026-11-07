"""Drive and structural-anxiety calculations for the lightweight ECA runtime."""
from __future__ import annotations

from typing import Any

from arcgym.eca.types import (
    AnxietySnapshot,
    DriveWeights,
    ECAState,
    SelfBodyCoordinate,
    SelfLocation,
    SelfRuleCoordinate,
    StructuralAnxietySignal,
    StructuralDecomposition,
    StructuralDeficitVector,
    StructuralElementMeta,
    StructuralGraphEdge,
    StructuralGraphNode,
    StructuralLocalWindow,
    StructuralPlan,
    StructuralRequirement,
)


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _normalize_status(raw: Any, *, fallback: str = "H") -> str:
    value = str(raw or "").strip().upper()
    mapping = {
        "KNOWN": "K",
        "HYPOTHESIS": "H",
        "OPEN": "H",
        "UNKNOWN": "U",
        "UNRESOLVED": "U",
        "CONTRADICTION": "C",
        "PENDING": "P",
        "TEST": "P",
    }
    if value in {"K", "H", "U", "C", "P"}:
        return value
    return mapping.get(value, fallback)


def _resource_pressure(resource: dict[str, float]) -> float:
    if not resource:
        return 0.0
    remaining_actions = float(resource.get("remaining_actions_ratio", 1.0) or 0.0)
    queue_headroom = float(resource.get("queue_headroom", 1.0) or 0.0)
    analyzer_budget = float(resource.get("analyzer_headroom", 1.0) or 0.0)
    return _clamp01(
        0.45 * (1.0 - remaining_actions)
        + 0.30 * (1.0 - queue_headroom)
        + 0.25 * (1.0 - analyzer_budget)
    )


def _status_distribution(decomposition: StructuralDecomposition) -> dict[str, float]:
    histogram = decomposition.status_histogram()
    total = max(1, sum(histogram.values()))
    return {
        key: round(_safe_ratio(value, total), 6)
        for key, value in histogram.items()
    }


def _structural_mode_weights(mode: str) -> dict[str, float]:
    normalized_mode = str(mode or "execute").strip().lower()
    table = {
        "execute": {"miss": 0.22, "conf": 0.22, "unk": 0.10, "reach": 0.22, "recur": 0.14, "loc": 0.10},
        "explore": {"miss": 0.12, "conf": 0.12, "unk": 0.34, "reach": 0.14, "recur": 0.12, "loc": 0.16},
        "verify": {"miss": 0.10, "conf": 0.34, "unk": 0.14, "reach": 0.12, "recur": 0.16, "loc": 0.14},
        "plan": {"miss": 0.30, "conf": 0.18, "unk": 0.14, "reach": 0.12, "recur": 0.14, "loc": 0.12},
        "refactor": {"miss": 0.24, "conf": 0.18, "unk": 0.10, "reach": 0.12, "recur": 0.24, "loc": 0.12},
        "relocalize": {"miss": 0.08, "conf": 0.16, "unk": 0.12, "reach": 0.18, "recur": 0.16, "loc": 0.30},
        "retreat": {"miss": 0.10, "conf": 0.10, "unk": 0.08, "reach": 0.40, "recur": 0.18, "loc": 0.14},
        "sleep": {"miss": 0.12, "conf": 0.12, "unk": 0.12, "reach": 0.12, "recur": 0.12, "loc": 0.40},
        "consolidate": {"miss": 0.18, "conf": 0.12, "unk": 0.08, "reach": 0.16, "recur": 0.18, "loc": 0.28},
    }
    weights = dict(table.get(normalized_mode, table["execute"]))
    total = sum(weights.values())
    if total <= 0:
        return dict(table["execute"])
    return {key: round(float(value) / total, 6) for key, value in weights.items()}


def _route_from_deficits(
    *,
    deficits: StructuralDeficitVector,
    reachability: float,
    plan_active: bool,
) -> tuple[str, str]:
    if float(deficits.loc) >= 0.58:
        return "relocalize", "self_location_confidence_low"
    if float(deficits.reach) >= 0.60:
        return "retreat", "reachable_actions_do_not_cover_required_scope"
    if float(deficits.conf) >= 0.55:
        return "verify", "local_contradiction_pressure"
    if float(deficits.unk) >= 0.55 and float(reachability) >= 0.45:
        return "explore", "frontier_unknown_pressure"
    if float(deficits.miss) >= 0.55 and float(deficits.recur) >= 0.45:
        return "refactor", "repeated_structural_gap"
    if float(deficits.miss) >= 0.55:
        return "plan", "local_structural_requirement_missing"
    if float(deficits.recur) >= 0.55 and not plan_active:
        return "refactor", "recurring_stagnation_without_live_plan"
    if float(deficits.recur) <= 0.20 and float(deficits.miss) <= 0.20 and plan_active:
        return "consolidate", "local_contract_is_stable"
    return "execute", "continue_low_regret_local_contract"


def _build_structural_decomposition(
    *,
    state: ECAState,
    context: dict[str, Any],
) -> StructuralDecomposition:
    nodes: list[StructuralGraphNode] = []
    edges: list[StructuralGraphEdge] = []
    meta: dict[str, StructuralElementMeta] = {}

    def add_node(
        node_id: str,
        *,
        label: str,
        node_type: str,
        conf: float,
        status: str,
        salience: float = 0.0,
        last_tested: int = -1,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        if node_id in meta:
            return
        nodes.append(StructuralGraphNode(node_id=node_id, label=label))
        meta[node_id] = StructuralElementMeta(
            type=node_type,
            conf=_clamp01(conf),
            status=_normalize_status(status),
            salience=_clamp01(salience),
            last_tested=int(last_tested),
            attrs=dict(attrs or {}),
        )

    def add_edge(
        edge_id: str,
        *,
        source: str,
        target: str,
        relation: str,
        edge_type: str = "constraint",
        conf: float = 1.0,
        status: str = "K",
        salience: float = 0.0,
        last_tested: int = -1,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        if edge_id in meta:
            return
        edges.append(
            StructuralGraphEdge(
                edge_id=edge_id,
                source=source,
                target=target,
                relation=relation,
            )
        )
        meta[edge_id] = StructuralElementMeta(
            type=edge_type,
            conf=_clamp01(conf),
            status=_normalize_status(status),
            salience=_clamp01(salience),
            last_tested=int(last_tested),
            attrs=dict(attrs or {}),
        )

    step_idx = int(state.step_index)
    add_node(
        "self::agent",
        label="agent",
        node_type="self",
        conf=1.0,
        status="K",
        salience=1.0,
        last_tested=step_idx,
        attrs={"mode": str(context.get("mode", "") or "")},
    )
    state_node_id = f"state::{state.state_hash}"
    add_node(
        state_node_id,
        label=f"state@{state.step_index}",
        node_type="state",
        conf=1.0,
        status="K",
        salience=0.95,
        last_tested=step_idx,
        attrs={"score": int(state.score), "level": int(state.level_number)},
    )
    add_edge(
        "edge::self_observes_state",
        source="self::agent",
        target=state_node_id,
        relation="observes",
        edge_type="constraint",
        conf=1.0,
        status="K",
        salience=1.0,
        last_tested=step_idx,
    )

    current_question = str(context.get("current_structure_question", "") or "")
    active_hypothesis = str(context.get("active_hypothesis", "") or current_question)
    if active_hypothesis:
        rule_node_id = "rule::active_hypothesis"
        add_node(
            rule_node_id,
            label=active_hypothesis[:96],
            node_type="rule",
            conf=0.68 if current_question else 0.55,
            status="H" if state.plan_active else "P",
            salience=0.92,
            last_tested=step_idx if state.plan_active else -1,
            attrs={"question": current_question},
        )
        add_edge(
            "edge::state_needs_rule",
            source=state_node_id,
            target=rule_node_id,
            relation="explained_by",
            edge_type="rule",
            conf=0.72,
            status="H",
            salience=0.86,
            last_tested=step_idx,
        )

    required_action_family = str(context.get("required_action_family", "") or "")
    if required_action_family:
        constraint_node_id = "constraint::required_family"
        add_node(
            constraint_node_id,
            label=f"required_family={required_action_family}",
            node_type="constraint",
            conf=0.95,
            status="K",
            salience=0.80,
            last_tested=step_idx,
        )
        add_edge(
            "edge::rule_constrained_by_family",
            source="rule::active_hypothesis" if active_hypothesis else state_node_id,
            target=constraint_node_id,
            relation="constrained_by",
            edge_type="constraint",
            conf=0.90,
            status="K",
            salience=0.78,
            last_tested=step_idx,
        )

    selected_target_id = str(context.get("selected_target_id", "") or "")
    selected_target_anchor = str(context.get("selected_target_anchor", "") or "")
    if selected_target_id or selected_target_anchor:
        target_node_id = "object::selected_target"
        add_node(
            target_node_id,
            label=selected_target_id or selected_target_anchor,
            node_type="object",
            conf=0.74,
            status="P",
            salience=0.90,
            last_tested=step_idx,
            attrs={"anchor": selected_target_anchor},
        )
        add_edge(
            "edge::rule_focuses_selected_target",
            source="rule::active_hypothesis" if active_hypothesis else state_node_id,
            target=target_node_id,
            relation="focuses",
            edge_type="constraint",
            conf=0.78,
            status="P",
            salience=0.88,
            last_tested=step_idx,
        )

    protected_hedge_summary = str(context.get("protected_hedge_summary", "") or "")
    if bool(context.get("protected_hedge_visible")) and protected_hedge_summary:
        hedge_node_id = "object::protected_hedge"
        add_node(
            hedge_node_id,
            label=protected_hedge_summary[:96],
            node_type="object",
            conf=0.70,
            status="P",
            salience=0.84,
            last_tested=step_idx,
        )
        add_edge(
            "edge::rule_preserves_hedge",
            source="rule::active_hypothesis" if active_hypothesis else state_node_id,
            target=hedge_node_id,
            relation="preserves",
            edge_type="constraint",
            conf=0.72,
            status="P",
            salience=0.82,
            last_tested=step_idx,
        )

    max_object_size = max((int(item.get("size", 0) or 0) for item in state.object_table), default=1)
    for index, item in enumerate((state.object_table or [])[:4]):
        component_id = str(item.get("component_id", "") or f"obj_{index}")
        node_id = f"object::{component_id}"
        size = int(item.get("size", 0) or 0)
        add_node(
            node_id,
            label=component_id,
            node_type="object",
            conf=0.65 + 0.30 * _clamp01(_safe_ratio(size, max_object_size)),
            status="K",
            salience=0.35 + 0.45 * _clamp01(_safe_ratio(size, max_object_size)),
            last_tested=step_idx,
            attrs={"size": size, "value": int(item.get("value", 0) or 0)},
        )
        add_edge(
            edge_id=f"edge::state_contains::{component_id}",
            source=state_node_id,
            target=node_id,
            relation="contains",
            edge_type="state",
            conf=1.0,
            status="K",
            salience=0.42,
            last_tested=step_idx,
        )

    for index, hypothesis in enumerate((state.structural_hypotheses or [])[:3]):
        node_id = f"rule::hypothesis::{index}"
        confidence = float(hypothesis.get("confidence", 0.0) or 0.0)
        add_node(
            node_id,
            label=str(hypothesis.get("description", "") or f"hypothesis_{index}")[:96],
            node_type="rule",
            conf=confidence,
            status=_normalize_status(hypothesis.get("status"), fallback="H"),
            salience=0.48 + 0.32 * _clamp01(confidence),
            last_tested=step_idx if confidence > 0.0 else -1,
            attrs={"kind": str(hypothesis.get("kind", "") or "")},
        )
        add_edge(
            edge_id=f"edge::state_supports_hypothesis::{index}",
            source=state_node_id,
            target=node_id,
            relation="supports",
            edge_type="rule",
            conf=max(0.20, confidence),
            status="H",
            salience=0.56,
            last_tested=step_idx,
        )

    frontier_unknown_ids: list[str] = []
    frontier_unknown_count = min(
        3,
        max(
            int(context.get("frontier_unknown_count", 0) or 0),
            1 if str(context.get("frontier_status", "") or "") in {"open", "probing"} else 0,
            1 if state.unresolved_components > 0 and current_question else 0,
        ),
    )
    for index in range(frontier_unknown_count):
        node_id = f"frontier::unknown::{index}"
        frontier_unknown_ids.append(node_id)
        add_node(
            node_id,
            label=f"frontier_unknown_{index}",
            node_type="object",
            conf=0.25,
            status="U",
            salience=0.52,
            last_tested=-1,
            attrs={"frontier_id": str(context.get("frontier_id", "") or "")},
        )
        add_edge(
            edge_id=f"edge::frontier_unknown::{index}",
            source="rule::active_hypothesis" if active_hypothesis else state_node_id,
            target=node_id,
            relation="tests_next",
            edge_type="constraint",
            conf=0.40,
            status="U",
            salience=0.58,
            last_tested=-1,
        )

    contradiction_pressure = bool(context.get("control_latch_active")) or (
        bool(state.plan_active) and state.no_change_streak >= 2
    )
    if contradiction_pressure:
        contradiction_node_id = "constraint::local_contradiction"
        add_node(
            contradiction_node_id,
            label="local_contradiction",
            node_type="constraint",
            conf=0.80,
            status="C",
            salience=0.90,
            last_tested=step_idx,
            attrs={"reason": str(context.get("contradiction_reason", "") or "")},
        )
        add_edge(
            "edge::contradicts_active_rule",
            source=contradiction_node_id,
            target="rule::active_hypothesis" if active_hypothesis else state_node_id,
            relation="contradicts",
            edge_type="constraint",
            conf=0.82,
            status="C",
            salience=0.90,
            last_tested=step_idx,
        )

    return StructuralDecomposition(
        nodes=nodes,
        edges=edges,
        meta=meta,
        frontier_unknown_ids=frontier_unknown_ids,
    )


def _build_structural_plan(
    *,
    state: ECAState,
    context: dict[str, Any],
    reachability: float,
) -> StructuralPlan:
    current_question = str(context.get("current_structure_question", "") or "")
    focus = str(context.get("focus", "") or current_question or "(unfocused)")
    no_target_allowed = bool(context.get("no_target_contract_allowed", False))
    selected_target_present = bool(context.get("selected_target_id") or context.get("protected_hedge_visible"))
    frontier_open = bool(str(context.get("frontier_status", "") or "") in {"open", "probing"})
    frontier_unknown_count = int(context.get("frontier_unknown_count", 0) or 0)
    required_action_family = str(context.get("required_action_family", "") or "")

    requirements = [
        StructuralRequirement(
            requirement_id="local_focus_contract",
            scope=focus,
            predicate="A single local structure question must stay aligned with one reachable local contract.",
            priority=1.0,
            repair_ops=["relocalize_focus", "rebuild_local_plan"],
            satisfied=bool(current_question and (selected_target_present or no_target_allowed)),
        ),
        StructuralRequirement(
            requirement_id="reachable_local_probe",
            scope=focus,
            predicate="The active local requirement must admit at least one reachable probe under current body constraints.",
            priority=0.95,
            repair_ops=["relocalize_focus", "rebuild_local_plan", "retreat"],
            satisfied=bool(reachability >= 0.40),
        ),
        StructuralRequirement(
            requirement_id="recent_steps_explained",
            scope=current_question or focus,
            predicate="The active hypothesis should explain the recent 1-3 local steps without unresolved contradiction.",
            priority=0.90,
            repair_ops=["promote_unknown_to_hypothesis", "demote_hypothesis", "mark_contradiction", "split_rule"],
            satisfied=bool(state.no_change_streak <= 1 and state.repeated_state_count <= 1),
        ),
        StructuralRequirement(
            requirement_id="frontier_unknown_bounded",
            scope=str(context.get("frontier_id", "") or focus),
            predicate="Frontier unknown pressure near the active scope should stay bounded before long execution continues.",
            priority=0.78 if frontier_open or frontier_unknown_count > 0 else 0.55,
            repair_ops=["add_unknown_placeholder", "promote_unknown_to_hypothesis", "rebuild_local_plan"],
            satisfied=bool(frontier_unknown_count <= 1 or state.unresolved_components <= 1),
        ),
        StructuralRequirement(
            requirement_id="body_constraints_respected",
            scope=focus,
            predicate="The main local plan must respect current action-family and reachability constraints.",
            priority=0.82 if required_action_family else 0.60,
            repair_ops=["rebuild_local_plan", "retreat"],
            satisfied=bool(not required_action_family or reachability >= 0.35),
        ),
    ]
    return StructuralPlan(requirements=requirements)


def _estimate_self_location(
    *,
    state: ECAState,
    decomposition: StructuralDecomposition,
    context: dict[str, Any],
    reachability: float,
) -> SelfLocation:
    no_target_streak = int(context.get("no_target_contract_streak", 0) or 0)
    dead_tail_ratio = float(context.get("candidate_pool_dead_tail_ratio", 0.0) or 0.0)
    frontier_status = str(context.get("frontier_status", "") or "")
    current_psa_state = str(context.get("current_target_psa_state", "") or "")
    target_anchor_locked = bool(context.get("target_anchor_locked", False))

    if current_psa_state == "VERIFY_ONCE" or target_anchor_locked:
        mode = "verify"
    elif no_target_streak >= 2:
        mode = "relocalize"
    elif dead_tail_ratio >= 0.72 and int(context.get("candidate_pool_live_escape_candidate_count", 0) or 0) <= 0:
        mode = "retreat"
    elif state.repeated_state_count >= 2 and state.no_change_streak >= 2:
        mode = "refactor"
    elif frontier_status in {"open", "probing"} or (state.unresolved_components > 0 and not state.plan_active):
        mode = "explore"
    elif state.plan_active or state.queue_remaining > 0:
        mode = "execute"
    else:
        mode = "explore"

    affordances = ["move_probe"]
    if float(state.click_coverage or 0.0) < 1.0:
        affordances.append("click_probe")
    if state.queue_remaining > 0:
        affordances.append("queued_execute")
    if no_target_streak > 0:
        affordances.append("relocalize_focus")
    if int(context.get("candidate_pool_live_escape_candidate_count", 0) or 0) > 0:
        affordances.append("expanded_focus_recovery")
    if bool(context.get("protected_hedge_visible")):
        affordances.append("protected_hedge_probe")

    remaining_actions_ratio = _safe_ratio(state.remaining_actions, max(1, state.max_actions))
    queue_headroom = 1.0 - _clamp01(_safe_ratio(state.queue_remaining, max(1, state.max_actions)))
    analyzer_headroom = 1.0 - _clamp01(_safe_ratio(state.analyzer_calls, max(1, state.step_index + 1)))
    resource = {
        "remaining_actions_ratio": round(remaining_actions_ratio, 6),
        "queue_headroom": round(queue_headroom, 6),
        "analyzer_headroom": round(analyzer_headroom, 6),
    }
    risk = _clamp01(
        0.35 * _clamp01(_safe_ratio(state.no_change_streak, 3.0))
        + 0.25 * _clamp01(_safe_ratio(state.repeated_state_count, 3.0))
        + 0.20 * _clamp01(dead_tail_ratio)
        + 0.20 * _resource_pressure(resource)
    )
    distribution = _status_distribution(decomposition)
    loc_conf = _clamp01(
        1.0
        - (
            0.35 * _clamp01(_safe_ratio(no_target_streak, 3.0))
            + 0.25 * float(context.get("frontier_uncertainty", 0.0) or 0.0)
            + 0.20 * _clamp01(_safe_ratio(state.repeated_state_count, 3.0))
            + 0.20 * _clamp01(dead_tail_ratio)
        )
    )
    focus = str(context.get("focus", "") or context.get("current_structure_question", "") or "")
    frontier = list(decomposition.frontier_unknown_ids[:3])
    active_hypothesis = str(context.get("active_hypothesis", "") or context.get("current_structure_question", "") or "")
    return SelfLocation(
        body=SelfBodyCoordinate(
            mode=mode,
            affordances=affordances,
            resource=resource,
            risk=risk,
            reachability=reachability,
        ),
        rule=SelfRuleCoordinate(
            focus=focus,
            kuhcp_distribution=distribution,
            frontier=frontier,
            active_hypothesis=active_hypothesis,
            loc_conf=loc_conf,
        ),
    )


def _build_local_window(
    *,
    decomposition: StructuralDecomposition,
    plan: StructuralPlan,
    self_location: SelfLocation,
    context: dict[str, Any],
) -> StructuralLocalWindow:
    focus = str(self_location.rule.focus or context.get("current_structure_question", "") or "")
    scope_ids = [
        item.requirement_id
        for item in plan.requirements
        if focus and focus in str(item.scope or "")
    ]
    if not scope_ids:
        scope_ids = [item.requirement_id for item in plan.requirements[:2]]
    node_ids = ["self::agent"]
    state_nodes = [item.node_id for item in decomposition.nodes if item.node_id.startswith("state::")]
    if state_nodes:
        node_ids.append(state_nodes[0])
    for candidate_id in (
        "rule::active_hypothesis",
        "object::selected_target",
        "object::protected_hedge",
    ):
        if candidate_id in decomposition.meta:
            node_ids.append(candidate_id)
    node_ids.extend(decomposition.frontier_unknown_ids[:2])
    deduped_node_ids = list(dict.fromkeys(node_ids))
    reachable_node_ids: list[str] = []
    if float(self_location.body.reachability) >= 0.35:
        reachable_node_ids = [
            node_id
            for node_id in deduped_node_ids
            if not node_id.startswith("frontier::unknown") or float(self_location.body.reachability) >= 0.55
        ]
    summary = (
        f"focus={focus or '(none)'} "
        f"scope={len(scope_ids)} reachable={len(reachable_node_ids)} "
        f"frontier_unknown={len(decomposition.frontier_unknown_ids)}"
    )
    return StructuralLocalWindow(
        focus=focus,
        scope_requirement_ids=scope_ids,
        node_ids=deduped_node_ids,
        reachable_node_ids=reachable_node_ids,
        frontier_unknown_ids=list(decomposition.frontier_unknown_ids),
        summary=summary,
    )


def _compute_structural_deficits(
    *,
    state: ECAState,
    decomposition: StructuralDecomposition,
    plan: StructuralPlan,
    self_location: SelfLocation,
    context: dict[str, Any],
) -> StructuralDeficitVector:
    total_priority = sum(float(item.priority or 0.0) for item in plan.requirements)
    unsatisfied_priority = sum(
        float(item.priority or 0.0)
        for item in plan.requirements
        if item.satisfied is False
    )
    contradiction_nodes = sum(
        1 for item in decomposition.meta.values() if str(item.status or "").upper() == "C"
    )
    frontier_unknown_count = len(decomposition.frontier_unknown_ids)
    unresolved_ratio = _clamp01(_safe_ratio(state.unresolved_components, max(1, state.component_count)))
    no_target_streak = int(context.get("no_target_contract_streak", 0) or 0)
    dead_tail_ratio = float(context.get("candidate_pool_dead_tail_ratio", 0.0) or 0.0)
    queue_exhaustions = int(context.get("queue_exhaustions", 0) or 0)

    miss = _clamp01(_safe_ratio(unsatisfied_priority, max(total_priority, 0.01)))
    conf = _clamp01(
        0.55 * _clamp01(_safe_ratio(contradiction_nodes, 2.0))
        + 0.25 * (_clamp01(_safe_ratio(state.no_change_streak, 3.0)) if state.plan_active else 0.0)
        + 0.20 * (1.0 if bool(context.get("control_latch_active")) else 0.0)
    )
    unk = _clamp01(
        0.45 * _clamp01(_safe_ratio(frontier_unknown_count, 3.0))
        + 0.35 * unresolved_ratio
        + 0.20 * float(context.get("frontier_uncertainty", 0.0) or 0.0)
    )
    reach = _clamp01(
        0.70 * (1.0 - float(self_location.body.reachability or 0.0))
        + 0.30 * (1.0 if no_target_streak > 0 and not bool(context.get("protected_hedge_visible")) else 0.0)
    )
    recur = _clamp01(
        0.35 * _clamp01(_safe_ratio(state.repeated_state_count, 3.0))
        + 0.25 * _clamp01(_safe_ratio(state.no_change_streak, 3.0))
        + 0.20 * _clamp01(_safe_ratio(no_target_streak, 3.0))
        + 0.20 * _clamp01(_safe_ratio(queue_exhaustions, max(1, state.step_index + 1)))
    )
    loc = _clamp01(1.0 - float(self_location.rule.loc_conf or 0.0))

    if dead_tail_ratio >= 0.65 and int(context.get("candidate_pool_live_escape_candidate_count", 0) or 0) <= 0:
        reach = _clamp01(reach + 0.10)
        recur = _clamp01(recur + 0.10)

    return StructuralDeficitVector(
        miss=miss,
        conf=conf,
        unk=unk,
        reach=reach,
        recur=recur,
        loc=loc,
    )


def _compute_structural_signal(
    *,
    deficits: StructuralDeficitVector,
    self_location: SelfLocation,
    plan_active: bool,
) -> StructuralAnxietySignal:
    weights = _structural_mode_weights(self_location.body.mode)
    total = (
        float(weights.get("miss", 0.0)) * float(deficits.miss)
        + float(weights.get("conf", 0.0)) * float(deficits.conf)
        + float(weights.get("unk", 0.0)) * float(deficits.unk)
        + float(weights.get("reach", 0.0)) * float(deficits.reach)
        + float(weights.get("recur", 0.0)) * float(deficits.recur)
        + float(weights.get("loc", 0.0)) * float(deficits.loc)
    )
    dominant_type = max(
        ("miss", "conf", "unk", "reach", "recur", "loc"),
        key=lambda key: float(getattr(deficits, key)),
    )
    route, route_reason = _route_from_deficits(
        deficits=deficits,
        reachability=float(self_location.body.reachability or 0.0),
        plan_active=bool(plan_active),
    )
    urgency = _clamp01(
        0.60 * float(total)
        + 0.40 * max(
            float(deficits.miss),
            float(deficits.conf),
            float(deficits.unk),
            float(deficits.reach),
            float(deficits.recur),
            float(deficits.loc),
        )
    )
    return StructuralAnxietySignal(
        deficits=deficits,
        dominant_type=dominant_type,
        urgency=urgency,
        route=route,
        total=_clamp01(total),
        weights=weights,
        route_reason=route_reason,
    )


def compute_anxiety_snapshot(
    *,
    state: ECAState,
    weights: DriveWeights,
    plan_generation: int,
    high_threshold: float,
    structural_context: dict[str, Any] | None = None,
) -> AnxietySnapshot:
    repeated_ratio = _clamp01(max(0, state.repeated_state_count - 1) / 3.0)
    no_change_ratio = _clamp01(state.no_change_streak / 3.0)
    stagnant_ratio = _clamp01(state.stagnant_steps / 6.0)
    same_action_ratio = _clamp01(max(0, state.same_action_streak - 1) / 3.0)
    unresolved_ratio = _clamp01(state.unresolved_components / max(1, state.component_count))
    complexity_ratio = _clamp01(state.component_count / 10.0)
    coverage_gap = 1.0 - _clamp01(state.click_coverage)
    known_effect_gap = 1.0 - _clamp01(state.known_effect_ratio)
    analyzer_density = _clamp01(state.analyzer_calls / max(1.0, state.step_index / 3.0 if state.step_index else 1.0))
    interrupt_pressure = _clamp01(state.interrupt_count / 4.0)
    plan_progress_ratio = _clamp01(state.plan_index / max(1, state.plan_total))
    remaining_pressure = _clamp01(1.0 - (state.remaining_actions / max(1, state.max_actions)))

    world = _clamp01(
        0.35 * no_change_ratio
        + 0.25 * repeated_ratio
        + 0.25 * known_effect_gap
        + 0.15 * same_action_ratio
    )
    structure = _clamp01(
        0.45 * unresolved_ratio
        + 0.30 * coverage_gap
        + 0.15 * complexity_ratio
        + 0.10 * _clamp01(len(state.structural_hypotheses) / 3.0)
    )
    self_drive = _clamp01(
        0.50 * no_change_ratio * (1.0 if state.plan_active else 0.0)
        + 0.25 * stagnant_ratio * (1.0 if state.current_plan else 0.0)
        + 0.15 * plan_progress_ratio
        + 0.10 * repeated_ratio
    )
    progress = _clamp01(
        0.45 * stagnant_ratio
        + 0.35 * no_change_ratio
        + 0.20 * repeated_ratio
    )
    control = _clamp01(
        0.40 * analyzer_density
        + 0.35 * interrupt_pressure
        + 0.15 * remaining_pressure
        + 0.10 * same_action_ratio
    )

    breakdown = {
        "world": world,
        "structure": structure,
        "self": self_drive,
        "progress": progress,
        "control": control,
    }
    weight_dict = weights.normalized().to_dict()
    total = sum(weight_dict[key] * breakdown[key] for key in breakdown)
    top_drive = max(breakdown, key=breakdown.get)

    context = dict(structural_context or {})
    context.setdefault("current_structure_question", "")
    context.setdefault("active_hypothesis", context.get("current_structure_question", ""))
    context.setdefault("focus", context.get("selected_target_anchor", "") or context.get("current_structure_question", ""))
    context.setdefault("frontier_unknown_count", min(3, max(0, state.unresolved_components)))
    context.setdefault("queue_exhaustions", 0)
    context.setdefault("candidate_pool_dead_tail_ratio", 0.0)
    context.setdefault("candidate_pool_live_escape_candidate_count", 0)
    context.setdefault("no_target_contract_streak", 0)
    context.setdefault("frontier_uncertainty", 0.0)

    reachability = _clamp01(
        0.55
        + (0.20 if context.get("selected_target_id") or context.get("protected_hedge_visible") else 0.0)
        - 0.25 * float(context.get("candidate_pool_dead_tail_ratio", 0.0) or 0.0)
        - 0.15 * _clamp01(_safe_ratio(int(context.get("no_target_contract_streak", 0) or 0), 3.0))
        - 0.10 * remaining_pressure
    )
    decomposition = _build_structural_decomposition(state=state, context=context)
    plan = _build_structural_plan(state=state, context=context, reachability=reachability)
    self_location = _estimate_self_location(
        state=state,
        decomposition=decomposition,
        context=context,
        reachability=reachability,
    )
    local_window = _build_local_window(
        decomposition=decomposition,
        plan=plan,
        self_location=self_location,
        context=context,
    )
    deficits = _compute_structural_deficits(
        state=state,
        decomposition=decomposition,
        plan=plan,
        self_location=self_location,
        context=context,
    )
    structural_signal = _compute_structural_signal(
        deficits=deficits,
        self_location=self_location,
        plan_active=state.plan_active,
    )

    return AnxietySnapshot(
        step_index=state.step_index,
        level_number=state.level_number,
        score=state.score,
        score_delta=state.score_delta,
        state_hash=state.state_hash,
        total=float(total),
        top_drive=top_drive,
        breakdown=breakdown,
        weights=weight_dict,
        plan_generation=plan_generation,
        episode_id=f"plan-{plan_generation}",
        queue_remaining=state.queue_remaining,
        plan_index=state.plan_index,
        plan_total=state.plan_total,
        no_change_streak=state.no_change_streak,
        stagnant_steps=state.stagnant_steps,
        repeated_state_count=state.repeated_state_count,
        unresolved_components=state.unresolved_components,
        analyzer_calls=state.analyzer_calls,
        interrupt_count=state.interrupt_count,
        plan_active=state.plan_active,
        high_anxiety=bool(total >= high_threshold),
        structural_decomposition=decomposition,
        structural_plan=plan,
        self_location=self_location,
        local_window=local_window,
        structural_signal=structural_signal,
    )
