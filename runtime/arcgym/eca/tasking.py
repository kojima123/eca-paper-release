"""Tasking helpers that convert anxiety into analyzer work."""
from __future__ import annotations

from arcgym.eca.types import AnxietySnapshot, ECAState


def preferred_plan_horizon(
    snapshot: AnxietySnapshot,
    *,
    structural_guidance_enabled: bool = True,
    structural_guidance_mode: str = "active",
) -> str:
    if structural_guidance_enabled:
        route = str((snapshot.structural_signal.route or "")).strip().lower()
        if structural_guidance_mode == "shadow" and route in {
            "relocalize", "explore", "verify", "retreat", "plan", "refactor",
        }:
            route = ""
        if route in {"relocalize", "verify", "retreat"}:
            return "1-2"
        if route in {"explore", "execute", "consolidate"}:
            return "1-3"
        if route in {"plan", "refactor"}:
            return "2-4"
    if snapshot.top_drive in {"self", "progress"} or snapshot.breakdown.get("world", 0.0) >= 0.70:
        return "1-3"
    if snapshot.top_drive == "structure":
        return "2-4"
    return "3-5"


def build_structural_task(
    *,
    snapshot: AnxietySnapshot | None,
    state: ECAState | None,
    trigger: str,
    structural_guidance_enabled: bool = True,
    structural_guidance_mode: str = "active",
    structural_guidance_shadow_reason: str = "",
) -> str:
    if snapshot is None or state is None:
        return (
            "Bootstrap a world-model probe. Infer the board objects, identify the most likely controllable "
            "component, and produce a short action sequence that can falsify one structural hypothesis."
        )

    route = str(snapshot.structural_signal.route or "").strip().lower()
    dominant = str(snapshot.structural_signal.dominant_type or "").strip().lower()
    focus = str(snapshot.self_location.rule.focus or "").strip()
    local_window = str(snapshot.local_window.summary or "").strip()
    route_context = (
        f"PSA route={route or 'execute'}, dominant_deficit={dominant or 'none'}, "
        f"focus={focus or '(none)'}, local_window={local_window or '(none)'}."
    )

    shadow_eligible = structural_guidance_enabled and structural_guidance_mode == "shadow" and route in {
        "relocalize", "explore", "verify", "retreat", "plan", "refactor",
    }
    if shadow_eligible:
        shadow_contract = "current visible preserved alternative"
        if structural_guidance_shadow_reason == "protected_hedge":
            shadow_contract = "current protected hedge"
        elif structural_guidance_shadow_reason == "late_phase_commit":
            shadow_contract = "current active contract"
        # v6 Patch 3: in late-phase commit, override all shadow routes to consolidate
        if structural_guidance_shadow_reason == "late_phase_commit":
            return (
                f"LATE PHASE — commit to the current contract. {route_context} "
                f"Keep the {shadow_contract} and do NOT explore, relocalize, or rewrite. "
                "Use the shortest possible action sequence that directly tests or extends "
                "the current hypothesis. Every remaining action must push toward scoring."
            )
        shadow_prefix = (
            f"Preserve the {shadow_contract} as the primary contract and visible fallback. "
            "Do not rewrite the local contract."
        )
        if route == "relocalize":
            return (
                f"Probe whether relocalization is actually warranted. {route_context} "
                f"{shadow_prefix} Use one bounded move-family probe to test whether "
                "relocalization evidence is strong enough to justify a later contract change."
            )
        if route == "explore":
            return (
                f"Probe one nearby frontier without overriding the current contract. {route_context} "
                f"{shadow_prefix} Use one bounded move-family probe near the current focus "
                "to test whether the unknown boundary is worth a later contract switch."
            )
        if route == "verify":
            return (
                f"Verify a local contradiction within the current contract. {route_context} "
                f"{shadow_prefix} Pick one conflicting assumption and produce the shortest "
                "discriminative test that can confirm or reject it without widening."
            )
        if route == "retreat":
            return (
                f"Check whether repositioning is needed within the current contract. {route_context} "
                f"{shadow_prefix} Use one short repositioning move only if it improves "
                "reachability without abandoning the current hedge."
            )
        if route == "plan":
            return (
                f"Identify a missing local constraint without rewriting the contract. {route_context} "
                f"{shadow_prefix} Produce one short plan that fills the gap "
                "inside the current target without widening to a new component."
            )
        if route == "refactor":
            return (
                f"Check whether the current local pattern is genuinely stale. {route_context} "
                f"{shadow_prefix} Only terminate a churn pattern if evidence directly contradicts "
                "the current hedge; otherwise continue with the current contract."
            )
    if structural_guidance_enabled and route == "relocalize":
        return (
            f"Relocalize the local contract. {route_context} "
            "Re-anchor on one concrete focus region, restate the active hypothesis in that region only, "
            "and emit the smallest move-family probe that can restore local orientation."
        )
    if structural_guidance_enabled and route == "explore":
        return (
            f"Explore the live frontier. {route_context} "
            "Probe one reachable unknown boundary near the current focus. Prefer the shortest local action sequence "
            "that can turn an unknown frontier into either a live hypothesis or a rejected branch."
        )
    if structural_guidance_enabled and route == "verify":
        return (
            f"Verify a local contradiction. {route_context} "
            "Pick one conflicting assumption in the current local contract, state the falsifiable rule, "
            "and produce the shortest discriminative test that can confirm or reject it."
        )
    if structural_guidance_enabled and route == "plan":
        return (
            f"Rebuild the local structural plan. {route_context} "
            "Identify the missing local constraint or dependency, then produce one short plan that fills that gap "
            "without widening to a new component or frontier."
        )
    if structural_guidance_enabled and route == "refactor":
        return (
            f"Refactor recurring local churn. {route_context} "
            "The same structural miss is repeating in roughly the same self-location. "
            "Terminate the stale local pattern, replace it with one cleaner local contract, and test only that rewrite."
        )
    if structural_guidance_enabled and route == "retreat":
        return (
            f"Reposition away from an unreachable local contract. {route_context} "
            "The current demand is visible but poorly reachable from the present body/tool state. "
            "Use a short repositioning move that improves reachability before attempting another detailed local test."
        )
    if structural_guidance_enabled and route == "consolidate":
        return (
            f"Consolidate the current local contract. {route_context} "
            "Keep the same local target and use a short low-regret continuation that strengthens the current hypothesis "
            "without reopening a broad search."
        )

    if snapshot.top_drive == "world":
        return (
            f"World-model repair: {route_context} recent actions are not producing the expected effect "
            f"(no_change_streak={state.no_change_streak}, repeated_state_count={state.repeated_state_count}). "
            "Find the smallest causal probe that can verify which component or movement actually changes the board."
        )
    if snapshot.top_drive == "structure":
        return (
            f"Structure-memory repair: {route_context} {state.unresolved_components}/{max(1, state.component_count)} components "
            "remain unresolved. Infer object roles, barriers, corridors, or trigger components before committing to a long plan."
        )
    if snapshot.top_drive == "self":
        return (
            f"Plan repair: {route_context} the current plan is losing support at step {state.plan_index}/{max(1, state.plan_total)}. "
            "Explain what assumption broke, then replace it with a shorter plan that directly tests the contradiction."
        )
    if snapshot.top_drive == "progress":
        return (
            f"Loop escape: {route_context} the run is stagnant for {state.stagnant_steps} steps with repeated-state pressure "
            f"{state.repeated_state_count}. Break the loop with a minimal exploratory branch that reaches a genuinely new board regime."
        )
    return (
        f"Control stabilization: {route_context} analyzer_calls={state.analyzer_calls}, interrupts={state.interrupt_count}. "
        "Return a plan that is short, robust, and unlikely to thrash if the board response is ambiguous."
    )
