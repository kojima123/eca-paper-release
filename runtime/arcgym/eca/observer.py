"""Observers that project the agent loop into an ECA state.

Extended in v0.2 to integrate the Goal-Conditioned Structural Time Model.
The new `build_goal_conditioned_state()` function constructs a full
`GoalConditionedState` (z_t) from the raw grid observation, including:
  - StructuralModel (m_t): objects, relations, invariants, affordance
  - GoalHypothesis (g_t): online-generated DSL hypothesis
  - goal_progress (p_t), uncertainty (u_t)
  - internal_time (tau): cumulative meaningful structural transitions

The original `build_eca_state()` is preserved unchanged for backward
compatibility with the existing ECA runtime.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from arcgym.eca.types import ECAState, Hypothesis
from arcgym.eca.structural_model import (
    GoalConditionedState,
    GoalHypothesis,
    StructuralModel,
    OscillationDetector,
    GoalDistanceTracker,
    build_structural_model_from_grid,
    compute_structural_diff,
    compute_D_O,
    compute_D_I,
    compute_structural_eca,
    compute_structural_psa,
    compute_adaptive_epsilon,
    generate_goal_hypothesis,
    update_internal_time,
)
from arcgym.eca.d_flow import (
    DFlowResult,
    run_d_flow_pipeline,
)
from arcgym.utils.grid_utils import find_connected_components, hash_grid_state


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _extract_grid(observation: dict[str, Any] | None) -> list[list[int]]:
    obs = observation or {}
    frame = obs.get("frame") or []
    if not frame:
        return []
    return [list(row) for row in frame[-1]]


def _build_object_table(grid: list[list[int]]) -> list[dict[str, Any]]:
    if not grid:
        return []
    counts = Counter(cell for row in grid for cell in row)
    background = counts.most_common(1)[0][0]
    component_map = find_connected_components(grid)
    grouped: dict[int, dict[str, Any]] = {}
    for (row, col), component_id in component_map.items():
        entry = grouped.setdefault(
            component_id,
            {
                "component_id": f"comp_{component_id}",
                "value": int(grid[row][col]),
                "size": 0,
                "bbox": {"min_row": row, "max_row": row, "min_col": col, "max_col": col},
            },
        )
        entry["size"] += 1
        bbox = entry["bbox"]
        bbox["min_row"] = min(bbox["min_row"], row)
        bbox["max_row"] = max(bbox["max_row"], row)
        bbox["min_col"] = min(bbox["min_col"], col)
        bbox["max_col"] = max(bbox["max_col"], col)

    objects = [
        entry
        for entry in grouped.values()
        if entry["value"] != background or entry["size"] < (len(grid) * len(grid[0]) * 0.8)
    ]
    objects.sort(key=lambda item: (-item["size"], item["value"], item["component_id"]))
    return objects


def _affordance_hot_regions(
    model: StructuralModel,
    *,
    prev_model: StructuralModel | None = None,
    limit: int = 8,
    min_score: float = 0.25,
) -> list[dict[str, Any]]:
    aff = model.affordance
    if not aff.data or aff.rows <= 0 or aff.cols <= 0:
        return []
    prev_aff = None if prev_model is None else prev_model.affordance
    prev_compatible = (
        prev_aff is not None
        and bool(prev_aff.data)
        and prev_aff.rows == aff.rows
        and prev_aff.cols == aff.cols
    )
    rows: list[dict[str, Any]] = []
    for r in range(aff.rows):
        for c in range(aff.cols):
            current = float(aff.get(r, c))
            previous = float(prev_aff.get(r, c)) if prev_compatible and prev_aff is not None else 0.0
            delta = abs(current - previous) if prev_compatible else current
            score = max(current, delta)
            if score < min_score:
                continue
            rows.append(
                {
                    "row": int(r),
                    "col": int(c),
                    "affordance_prior": round(current, 6),
                    "d_a_prior": round(score, 6),
                    "d_a_posterior_delta": round(delta, 6),
                    "spatial_confidence": round(score, 6),
                    "prior_source": "structural_time_affordance",
                }
            )
    rows.sort(
        key=lambda item: (
            -float(item.get("d_a_prior", 0.0) or 0.0),
            int(item.get("row", 0) or 0),
            int(item.get("col", 0) or 0),
        )
    )
    return rows[: max(0, int(limit))]


def build_eca_state(
    *,
    agent: Any,
    observation: dict[str, Any] | None,
    level_number: int,
    max_actions: int,
    analyzer_calls: int,
    interrupt_count: int,
) -> ECAState:
    """Build ECAState from the agent loop (original, preserved for compatibility)."""
    view = agent.export_eca_view()
    recent_history = view.get("recent_history") or []
    state_action_memory = view.get("state_action_memory") or {}
    plan_status = view.get("plan_status") or {}

    grid = _extract_grid(observation)
    state_hash = hash_grid_state(grid)
    object_table = _build_object_table(grid)
    component_count = len(object_table)
    unique_colors = len({entry["value"] for entry in object_table}) if object_table else 0

    current_state_actions = dict(state_action_memory.get(state_hash, {}))
    current_state_known_actions = len(current_state_actions)
    current_state_changed_actions = sum(1 for record in current_state_actions.values() if record.get("changed"))
    known_effect_ratio = (
        current_state_changed_actions / max(1, current_state_known_actions)
        if current_state_known_actions
        else 0.0
    )

    clicked_components = {
        action_key.split("click_", 1)[1]
        for state_rows in state_action_memory.values()
        for action_key in state_rows.keys()
        if action_key.startswith("click_")
    }
    click_coverage = len(clicked_components) / max(1, component_count) if component_count else 0.0
    unresolved_components = max(0, component_count - len(clicked_components))

    no_change_streak = 0
    stagnant_steps = 0
    repeated_state_count = 0
    same_action_streak = 0
    recent_failures = 0
    latest_action = None
    current_score = int((observation or {}).get("score", 0) or 0)

    for step in reversed(recent_history):
        pre_grid = step.get("grid_raw") or []
        if hash_grid_state(pre_grid) == state_hash:
            repeated_state_count += 1
        if step.get("score", current_score) == current_score:
            stagnant_steps += 1
        else:
            break

    for step in reversed(recent_history):
        action_name = step.get("action")
        if latest_action is None:
            latest_action = action_name
        if step.get("no_state_change"):
            no_change_streak += 1
            recent_failures += 1
        else:
            break

    for step in reversed(recent_history):
        action_name = step.get("action")
        if action_name and action_name == latest_action:
            same_action_streak += 1
        else:
            break

    hypotheses: list[Hypothesis] = []
    if unresolved_components > 0:
        hypotheses.append(
            Hypothesis(
                kind="object_role",
                description=f"{unresolved_components} components have not been causally isolated yet.",
                confidence=_clamp01(unresolved_components / max(1, component_count)),
            )
        )
    if stagnant_steps >= 3:
        hypotheses.append(
            Hypothesis(
                kind="phase_change",
                description="The board is stagnant; a timer, phase change, or gating mechanic may be missing.",
                confidence=_clamp01(stagnant_steps / 6.0),
            )
        )
    if no_change_streak >= 2:
        hypotheses.append(
            Hypothesis(
                kind="causal_mismatch",
                description="The current plan predicts change, but recent actions are not moving the board.",
                confidence=_clamp01(no_change_streak / 4.0),
            )
        )

    queue_remaining = int(plan_status.get("queue_remaining", 0))
    plan_total = int(plan_status.get("plan_total", 0))
    plan_index = int(plan_status.get("plan_index", 0))
    current_plan = str(plan_status.get("current_plan", "") or "")
    plan_reasoning = str(plan_status.get("plan_reasoning", "") or "")

    return ECAState(
        game_id=view.get("game_id"),
        step_index=int(view.get("action_counter", 0)),
        level_number=level_number,
        score=current_score,
        score_delta=int((observation or {}).get("score_delta", 0) or 0),
        arc_state=str((observation or {}).get("state", "UNKNOWN")),
        state_hash=state_hash,
        component_count=component_count,
        unresolved_components=unresolved_components,
        unique_colors=unique_colors,
        object_table=object_table,
        structural_hypotheses=[hyp.to_dict() for hyp in hypotheses],
        queue_remaining=queue_remaining,
        plan_index=plan_index,
        plan_total=plan_total,
        current_plan=current_plan,
        plan_reasoning=plan_reasoning,
        analyzer_calls=analyzer_calls,
        interrupt_count=interrupt_count,
        remaining_actions=max(0, max_actions - int(view.get("action_counter", 0))),
        max_actions=max_actions,
        no_change_streak=no_change_streak,
        recent_failures=min(recent_failures, len(recent_history)),
        stagnant_steps=stagnant_steps,
        repeated_state_count=repeated_state_count,
        same_action_streak=same_action_streak,
        current_state_known_actions=current_state_known_actions,
        current_state_changed_actions=current_state_changed_actions,
        click_coverage=_clamp01(click_coverage),
        known_effect_ratio=_clamp01(known_effect_ratio),
        plan_active=bool(current_plan or queue_remaining > 0 or plan_total > 0),
        recent_history=recent_history,
        current_state_actions=current_state_actions,
    )


# ---------------------------------------------------------------------------
# v0.2: Goal-Conditioned Structural Time Model integration
# ---------------------------------------------------------------------------

class StructuralTimeTracker:
    """Stateful tracker for the internal structural time model (v0.2).

    Maintains the current GoalConditionedState (z_t) and internal time (tau)
    across steps. This is the primary integration point for the new model.

    Usage:
        tracker = StructuralTimeTracker()
        # On each step after receiving observation:
        z_t = tracker.update(observation, action_history)
        eca = tracker.get_eca()
        psa = tracker.get_psa()
    """

    def __init__(self) -> None:
        self._current_state: GoalConditionedState = GoalConditionedState()
        self._prev_model: StructuralModel | None = None
        self._rollout_history: list[dict[str, float]] = []
        self._action_history: list[str] = []
        self._step_count: int = 0
        self._hypothesis_counter: int = 0
        self._last_d_flow_result: DFlowResult | None = None
        self._last_utility_margin: float = float("inf")
        self._last_affordance_hot_regions: list[dict[str, Any]] = []
        # v0.2.1: oscillation detection and goal-distance tracking
        self._oscillation_detector: OscillationDetector = OscillationDetector()
        self._goal_distance_tracker: GoalDistanceTracker = GoalDistanceTracker()
        self._last_oscillation_suppressed: bool = False

    def update(
        self,
        observation: dict[str, Any] | None,
        action: str = "",
        *,
        noise_floor: float = 0.02,
    ) -> GoalConditionedState:
        """Update the structural state from a new observation.

        Args:
            observation: raw observation dict from the environment
            action: the action that was taken to reach this observation
            noise_floor: estimated noise floor for adaptive epsilon

        Returns:
            Updated GoalConditionedState z_t
        """
        grid = _extract_grid(observation)
        external_score = float((observation or {}).get("score", 0) or 0)

        # Build new structural model m_t
        new_model = build_structural_model_from_grid(
            grid,
            step_index=self._step_count,
            prev_model=self._prev_model,
        )
        self._last_affordance_hot_regions = _affordance_hot_regions(
            new_model,
            prev_model=self._prev_model,
            limit=8,
        )

        # Compute structural difference d_m (base components)
        diff_result = {"d_m": 0.0, "D_O": 0.0, "D_R": 0.0, "D_I": 0.0, "D_A": 0.0}
        d_flow_result: DFlowResult | None = None
        if self._prev_model is not None:
            diff_result = compute_structural_diff(self._prev_model, new_model)
            # Run D_flow pipeline (Tier 0/1/2) to refine D_O estimate
            d_flow_result = run_d_flow_pipeline(
                self._prev_model,
                new_model,
                d_core=diff_result["D_O"],
                d_i=diff_result["D_I"],
                uncertainty=self._current_state.uncertainty,
                utility_margin=self._last_utility_margin,
            )
            # Replace D_O with D_flow-blended estimate when available
            if d_flow_result.tier_used >= 1:
                diff_result = dict(diff_result)
                diff_result["D_O"] = d_flow_result.d_o_blended
                # Recompute d_m with updated D_O
                w_o, w_r, w_i, w_a = 0.35, 0.25, 0.25, 0.15
                diff_result["d_m"] = _clamp01(
                    (w_o * diff_result["D_O"]
                     + w_r * diff_result["D_R"]
                     + w_i * diff_result["D_I"]
                     + w_a * diff_result["D_A"]) / (w_o + w_r + w_i + w_a)
                )
            self._last_d_flow_result = d_flow_result

        d_m = diff_result["d_m"]

        # Update internal time tau (with oscillation-aware phi)
        current_uncertainty = self._current_state.uncertainty
        is_osc = self._oscillation_detector.is_oscillating()
        if is_osc and not self._goal_distance_tracker.is_making_progress():
            # Suppress phi when oscillating without progress
            epsilon = compute_adaptive_epsilon(current_uncertainty, noise_floor)
            phi_val = 0.0
            new_tau = self._current_state.internal_time
            self._last_oscillation_suppressed = True
        else:
            new_tau, phi_val = update_internal_time(
                self._current_state.internal_time,
                d_m,
                current_uncertainty,
                noise_floor,
            )
            self._last_oscillation_suppressed = False

        # Generate/update goal hypothesis g_t
        self._hypothesis_counter += 1
        new_hypothesis = generate_goal_hypothesis(
            new_model,
            self._action_history,
            prev_hypothesis=self._current_state.goal_hypothesis,
            hypothesis_id=f"h{self._hypothesis_counter}",
        )

        # Estimate goal progress p_t (heuristic: based on hypothesis confidence
        # and reduction in uncertainty)
        new_progress = _clamp01(
            0.6 * new_hypothesis.confidence
            + 0.4 * self._current_state.goal_progress
        )

        # Estimate uncertainty u_t (decreases as hypothesis confidence grows)
        new_uncertainty = _clamp01(1.0 - new_hypothesis.confidence)

        # Build new state z_t
        new_state = GoalConditionedState(
            structural_model=new_model,
            goal_hypothesis=new_hypothesis,
            goal_progress=new_progress,
            uncertainty=new_uncertainty,
            external_score=external_score,
            internal_time=new_tau,
        )

        # Record rollout history for PSA computation
        eca_result = compute_structural_eca(new_state)
        history_entry: dict[str, Any] = {
            "d_m": d_m,
            "D_O": diff_result.get("D_O", 0.0),
            "D_R": diff_result.get("D_R", 0.0),
            "D_I": diff_result.get("D_I", 0.0),
            "D_A": diff_result.get("D_A", 0.0),
            "phi_val": phi_val,
            "eca": eca_result["eca"],
            "oscillation_suppressed": self._last_oscillation_suppressed,
        }
        if d_flow_result is not None:
            history_entry["d_flow_tier"] = d_flow_result.tier_used
            history_entry["q_flow"] = d_flow_result.q_flow
            history_entry["rho"] = d_flow_result.residual.rho
        self._rollout_history.append(history_entry)
        # Keep only recent history (last 10 steps)
        if len(self._rollout_history) > 10:
            self._rollout_history = self._rollout_history[-10:]

        # Update state
        self._prev_model = new_model
        self._current_state = new_state
        self._step_count += 1
        if action:
            self._action_history.append(action)
            self._oscillation_detector.record(action)
            if len(self._action_history) > 50:
                self._action_history = self._action_history[-50:]

        return new_state

    def get_eca(self) -> dict[str, float]:
        """Get current structural anxiety (ECA) breakdown."""
        return compute_structural_eca(self._current_state)

    def get_psa(
        self,
        tau_budget: float = 2.0,
        stagnation_window: int = 3,
    ) -> dict[str, float]:
        """Get future structural anxiety (PSA) using tau-budget rollout."""
        return compute_structural_psa(
            self._current_state,
            self._rollout_history,
            tau_budget=tau_budget,
            stagnation_window=stagnation_window,
        )

    def get_state(self) -> GoalConditionedState:
        """Return the current goal-conditioned state z_t."""
        return self._current_state

    def get_structural_diff_summary(self) -> dict[str, float]:
        """Return the most recent structural diff breakdown."""
        if not self._rollout_history:
            return {"d_m": 0.0, "D_O": 0.0, "D_R": 0.0, "D_I": 0.0, "D_A": 0.0, "phi_val": 0.0}
        last = self._rollout_history[-1]
        return {
            "d_m": last.get("d_m", 0.0),
            "D_O": last.get("D_O", 0.0),
            "D_R": last.get("D_R", 0.0),
            "D_I": last.get("D_I", 0.0),
            "D_A": last.get("D_A", 0.0),
            "phi_val": last.get("phi_val", 0.0),
        }

    def get_affordance_hot_regions(self, *, limit: int = 8) -> list[dict[str, Any]]:
        """Return generic D_A-style spatial priors from the latest heatmap."""

        if limit <= 0:
            return []
        return [dict(item) for item in self._last_affordance_hot_regions[: int(limit)]]

    def set_utility_margin(self, margin: float) -> None:
        """Update the top-1 vs top-2 action utility margin for Tier 2 triggering.

        Call this before `update()` when the utility margin is known from
        the policy layer (e.g., after scoring candidate actions).

        Args:
            margin: U(a_1) - U(a_2), the gap between best and second-best action.
        """
        self._last_utility_margin = max(0.0, float(margin))

    def get_d_flow_result(self) -> DFlowResult | None:
        """Return the most recent D_flow pipeline result (or None if not yet run)."""
        return self._last_d_flow_result

    def get_d_flow_summary(self) -> dict[str, Any]:
        """Return a compact summary of the most recent D_flow result."""
        if self._last_d_flow_result is None:
            return {"tier_used": 0, "q_flow": 0.0, "d_flow": 0.0, "d_o_blended": 0.0}
        r = self._last_d_flow_result
        summary = {
            "tier_used": r.tier_used,
            "q_flow": round(r.q_flow, 4),
            "d_flow": round(r.d_flow, 4),
            "d_o_blended": round(r.d_o_blended, 4),
            "rho": round(r.residual.rho, 4),
            "delta_n": r.residual.delta_n,
        }
        if r.sinkhorn is not None:
            summary["split_detected"] = r.sinkhorn.split_detected
            summary["merge_detected"] = r.sinkhorn.merge_detected
            summary["birth_detected"] = r.sinkhorn.birth_detected
            summary["death_detected"] = r.sinkhorn.death_detected
        return summary

    def record_goal_distance(self, distance: float) -> None:
        """Record the current distance to goal for progress tracking.

        Call this from the policy layer when the goal distance is known.
        """
        self._goal_distance_tracker.record_distance(distance)

    def is_oscillating(self) -> bool:
        """Check if the agent is currently oscillating."""
        return self._oscillation_detector.is_oscillating()

    def get_oscillation_info(self) -> dict:
        """Return oscillation diagnostic info."""
        return self._oscillation_detector.to_dict()

    def get_escape_actions(self) -> list[str]:
        """Return escape actions to break oscillation."""
        return self._oscillation_detector.get_escape_actions()

    def get_progress_ratio(self) -> float:
        """Return goal distance progress ratio."""
        return self._goal_distance_tracker.get_progress_ratio()

    def reset(self) -> None:
        """Reset the tracker state (e.g., on new game/level)."""
        self._current_state = GoalConditionedState()
        self._prev_model = None
        self._rollout_history = []
        self._action_history = []
        self._step_count = 0
        self._hypothesis_counter = 0
        self._last_d_flow_result = None
        self._last_utility_margin = float("inf")
        self._last_affordance_hot_regions = []
        self._oscillation_detector.reset()
        self._goal_distance_tracker.reset()
        self._last_oscillation_suppressed = False

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a compact summary dict for logging/debugging."""
        eca = self.get_eca()
        psa = self.get_psa()
        state = self._current_state
        diff = self.get_structural_diff_summary()
        d_flow_summary = self.get_d_flow_summary()
        return {
            "step": self._step_count,
            "internal_time_tau": round(state.internal_time, 4),
            "goal_progress": round(state.goal_progress, 4),
            "uncertainty": round(state.uncertainty, 4),
            "external_score": round(state.external_score, 4),
            "eca": round(eca["eca"], 4),
            "psa": round(psa["psa"], 4),
            "d_m": round(diff["d_m"], 4),
            "D_A": round(diff.get("D_A", 0.0), 4),
            "phi_val": round(diff["phi_val"], 4),
            "affordance_hot_region_count": len(self._last_affordance_hot_regions),
            "affordance_hot_regions": self.get_affordance_hot_regions(limit=5),
            "n_objects": state.structural_model.invariants.n_objects,
            "hypothesis_id": state.goal_hypothesis.hypothesis_id,
            "hypothesis_confidence": round(state.goal_hypothesis.confidence, 4),
            "termination_predicate": state.goal_hypothesis.termination_predicate,
            # D_flow diagnostics
            "d_flow_tier": d_flow_summary.get("tier_used", 0),
            "q_flow": d_flow_summary.get("q_flow", 0.0),
            "d_flow": d_flow_summary.get("d_flow", 0.0),
            "d_o_blended": d_flow_summary.get("d_o_blended", 0.0),
            "rho": d_flow_summary.get("rho", 0.0),
            # Oscillation diagnostics
            "is_oscillating": self._oscillation_detector.is_oscillating(),
            "oscillation_severity": self._oscillation_detector.get_oscillation_severity(),
            "oscillation_suppressed": self._last_oscillation_suppressed,
            "goal_progress_ratio": self._goal_distance_tracker.get_progress_ratio(),
        }


def build_goal_conditioned_state(
    *,
    observation: dict[str, Any] | None,
    tracker: StructuralTimeTracker,
    action: str = "",
) -> GoalConditionedState:
    """Convenience function to update and return the goal-conditioned state.

    This is the primary entry point for integrating the v0.2 structural
    time model into the agent loop.

    Args:
        observation: raw observation dict from the environment
        tracker: stateful StructuralTimeTracker instance
        action: the action taken to reach this observation

    Returns:
        Updated GoalConditionedState z_t
    """
    return tracker.update(observation, action)
