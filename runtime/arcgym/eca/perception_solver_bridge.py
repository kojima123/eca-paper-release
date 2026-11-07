"""Observer bridge from ECA runtime to perception-driven game solvers.

By default the bridge is non-authoritative: it asks the existing
``*_perception_driven`` adapters for a plan from the current rendered frame
and records whether a runtime solver was available. If explicitly switched to
``guarded`` mode, ECA may consume a short, allowed-action-checked prefix of
that advice as a local controller handoff. This bridge is deployability
infrastructure; guarded adoption is not evidence that ECA independently
discovered the plan.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from arcgym.eca.target_binding_repair_gate import (
    RepairDecision,
    TargetBindingRepairGate,
    _selective_high_risk_threshold_from_env,
    _selective_mode_from_env,
    _selective_ood_max_from_env,
    _selective_stuck_threshold_from_env,
    load_default_repair_gate,
    repair_gate_enabled,
    repair_gate_mode,
)


PERCEPTION_SOLVER_REGISTRY: dict[str, str] = {
    "wa30": "arcgym.adapters.wa30_perception_driven",
    "tu93": "arcgym.adapters.tu93_perception_driven",
    "m0r0": "arcgym.adapters.m0r0_perception_driven",
    "ka59": "arcgym.adapters.ka59_perception_driven",
    "r11l": "arcgym.adapters.r11l_perception_driven",
    "s5i5": "arcgym.adapters.s5i5_perception_driven",
    "tn36": "arcgym.adapters.tn36_perception_driven",
    "cn04": "arcgym.adapters.cn04_perception_driven",
    "ar25": "arcgym.adapters.ar25_perception_driven",
    "su15": "arcgym.adapters.su15_perception_driven",
    "tr87": "arcgym.adapters.tr87_perception_driven",
    "sc25": "arcgym.adapters.sc25_perception_driven",
    "sp80": "arcgym.adapters.sp80_perception_driven",
    "re86": "arcgym.adapters.re86_perception_driven",
    "dc22": "arcgym.adapters.dc22_perception_driven",
    "lf52": "arcgym.adapters.lf52_perception_driven",
    "bp35": "arcgym.adapters.bp35_perception_driven",
    "g50t": "arcgym.adapters.g50t_perception_driven",
    "sk48": "arcgym.adapters.sk48_perception_driven",
    "cd82": "arcgym.adapters.cd82_perception_driven",
    "ls20": "arcgym.adapters.ls20_perception_driven",
    "ft09": "arcgym.adapters.ft09_perception_driven",
    "vc33": "arcgym.adapters.vc33_perception_driven",
    "lp85": "arcgym.adapters.lp85_perception_driven",
    "sb26": "arcgym.adapters.sb26_perception_driven",
}

R11L_SCORE_PROGRESS_TEMPLATE_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"score": 1, "kind": "inactive_piece", "color": 15, "support": 2},
    {"score": 1, "kind": "inactive_piece", "color": 12, "support": 1},
    {"score": 2, "kind": "inactive_piece", "color": 15, "support": 2},
    {"score": 2, "kind": "inactive_piece", "color": 14, "support": 1},
    {"score": 3, "kind": "inactive_piece", "color": 14, "support": 2},
    {
        "score": 2,
        "kind": "target_centroid_destination_jitter",
        "color": 14,
        "target_dx": -4,
        "target_dy": 2,
        "jitter_dx": 1,
        "jitter_dy": -1,
        "support": 1,
    },
    {
        "score": 3,
        "kind": "target_centroid_destination_jitter",
        "color": 11,
        "target_dx": 3,
        "target_dy": -6,
        "jitter_dx": 0,
        "jitter_dy": -1,
        "support": 1,
    },
    {
        "score": 4,
        "kind": "target_centroid_destination_jitter",
        "color": 14,
        "target_dx": -1,
        "target_dy": -4,
        "jitter_dx": 0,
        "jitter_dy": 1,
        "support": 1,
    },
    {
        "score": 5,
        "kind": "target_centroid_destination_jitter",
        "color": 11,
        "target_dx": 1,
        "target_dy": -1,
        "jitter_dx": -1,
        "jitter_dy": 0,
        "support": 1,
    },
    {
        "score": 5,
        "kind": "target_centroid_destination_jitter",
        "color": 11,
        "target_dx": -2,
        "target_dy": 5,
        "jitter_dx": -1,
        "jitter_dy": 1,
        "support": 1,
    },
    {
        "score": 5,
        "kind": "target_centroid_destination_jitter",
        "color": 11,
        "target_dx": 5,
        "target_dy": -6,
        "jitter_dx": 1,
        "jitter_dy": 0,
        "support": 1,
    },
    {
        "score": 5,
        "kind": "target_centroid_destination_jitter",
        "color": 11,
        "target_dx": 12,
        "target_dy": 0,
        "jitter_dx": 1,
        "jitter_dy": 1,
        "support": 1,
    },
)

R11L_SCORE_PROGRESS_SELECTED_TARGET_OFFSET_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"score": 3, "dx": 0, "dy": 0, "support": 2},
    {"score": 5, "dx": 0, "dy": 0, "support": 4},
    {"score": 1, "dx": -2, "dy": 33, "support": 1},
    {"score": 3, "dx": 30, "dy": 19, "support": 1},
    {"score": 4, "dx": -2, "dy": 39, "support": 1},
    {"score": 4, "dx": -6, "dy": 15, "support": 1},
    {"score": 4, "dx": 13, "dy": -30, "support": 1},
    {"score": 4, "dx": 33, "dy": -8, "support": 1},
    {"score": 4, "dx": 35, "dy": -8, "support": 1},
    {"score": 5, "dx": -19, "dy": 21, "support": 1},
)


def _normalize_mode(value: str | None) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"1", "true", "yes", "on", "observer"}:
        return "observer"
    if mode in {"advisory", "advise"}:
        return "advisory"
    if mode in {"guarded", "execute", "execution", "exec"}:
        return "guarded"
    if mode in {"0", "false", "no", "off", ""}:
        return "off"
    return mode


def solver_bridge_mode() -> str:
    """Return the requested bridge mode from env."""

    return _normalize_mode(os.environ.get("V2_ECA_SOLVER_BRIDGE"))


def solver_bridge_enabled() -> bool:
    return solver_bridge_mode() in {"observer", "advisory", "guarded"}


def _selected_games_from_env() -> set[str]:
    raw = str(os.environ.get("V2_ECA_SOLVER_BRIDGE_GAMES", "") or "").strip()
    if not raw:
        return set()
    return {part.strip().lower() for part in raw.replace(";", ",").split(",") if part.strip()}


def _max_preview_from_env(default: int = 5) -> int:
    try:
        return max(0, int(os.environ.get("V2_ECA_SOLVER_BRIDGE_MAX_PREVIEW", default)))
    except (TypeError, ValueError):
        return default


def _max_execution_actions_from_env(default: int = 1) -> int:
    try:
        return max(1, int(os.environ.get("V2_ECA_SOLVER_BRIDGE_MAX_EXEC_ACTIONS", default)))
    except (TypeError, ValueError):
        return default


def _execution_tiers_from_env() -> set[str]:
    raw = str(
        os.environ.get(
            "V2_ECA_SOLVER_BRIDGE_EXEC_TIERS",
            "tier_a_runtime_solver,frame_conditioned_per_game_solver,runtime_solver_observed",
        )
        or ""
    ).strip()
    return {
        part.strip().lower()
        for part in raw.replace(";", ",").split(",")
        if part.strip()
    }


def _repair_execution_mode_from_env() -> str:
    mode = str(os.environ.get("V2_ECA_TARGET_BINDING_REPAIR_EXECUTE", "") or "").strip().lower()
    if mode in {"guarded", "execute", "execution", "1", "true", "yes", "on"}:
        return "guarded"
    return "off"


def _repair_execution_games_from_env() -> set[str]:
    raw = str(os.environ.get("V2_ECA_TARGET_BINDING_REPAIR_EXECUTE_GAMES", "") or "").strip()
    if not raw:
        return set()
    return {part.strip().lower() for part in raw.replace(";", ",").split(",") if part.strip()}


def _repair_prior_execution_mode_from_env() -> str:
    mode = str(os.environ.get("V2_ECA_REPAIR_PRIOR_EXECUTE", "") or "").strip().lower()
    if mode in {"guarded", "execute", "execution", "1", "true", "yes", "on"}:
        return "guarded"
    return "off"


def _repair_prior_execution_games_from_env() -> set[str]:
    raw = str(os.environ.get("V2_ECA_REPAIR_PRIOR_EXECUTE_GAMES", "") or "").strip()
    if not raw:
        return set()
    return {part.strip().lower() for part in raw.replace(";", ",").split(",") if part.strip()}


def _repair_prior_execution_min_score_from_env(default: float = 0.95) -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("V2_ECA_REPAIR_PRIOR_EXECUTE_MIN_SCORE", default))))
    except (TypeError, ValueError):
        return default


def _repair_prior_dynamic_threshold_enabled() -> bool:
    raw = str(os.environ.get("V2_ECA_REPAIR_PRIOR_DYNAMIC_THRESHOLD", "1") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _repair_prior_execution_floor_score_from_env(default: float = 0.50) -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("V2_ECA_REPAIR_PRIOR_EXECUTE_FLOOR_SCORE", default))))
    except (TypeError, ValueError):
        return default


def _repair_prior_effect_credit_window_from_env(default: int = 8) -> int:
    try:
        return max(0, int(os.environ.get("V2_ECA_REPAIR_PRIOR_EFFECT_CREDIT_WINDOW", default)))
    except (TypeError, ValueError):
        return default


def _repair_prior_effect_history_max_from_env(default: int = 32) -> int:
    try:
        return max(0, int(os.environ.get("V2_ECA_REPAIR_PRIOR_EFFECT_HISTORY_MAX", default)))
    except (TypeError, ValueError):
        return default


def _repair_prior_chain_value_seed_path_from_env() -> str:
    return str(os.environ.get("V2_ECA_REPAIR_PRIOR_CHAIN_VALUE_SEED_PATH", "") or "").strip()


def _repair_prior_chain_value_seed_min_support_from_env(default: int = 1) -> int:
    try:
        return max(1, int(os.environ.get("V2_ECA_REPAIR_PRIOR_CHAIN_VALUE_SEED_MIN_SUPPORT", default)))
    except (TypeError, ValueError):
        return default


def _repair_prior_chain_value_seed_max_rules_from_env(default: int = 64) -> int:
    try:
        return max(0, int(os.environ.get("V2_ECA_REPAIR_PRIOR_CHAIN_VALUE_SEED_MAX_RULES", default)))
    except (TypeError, ValueError):
        return default


def _repair_prior_chain_value_seed_key_mode_from_env() -> str:
    raw = str(
        os.environ.get("V2_ECA_REPAIR_PRIOR_CHAIN_VALUE_SEED_KEY_MODE", "all") or "all"
    ).strip().lower()
    if raw in {"context", "context_only", "context-only"}:
        return "context_only"
    if raw in {"context_backed", "context-backed", "backed_context"}:
        return "context_backed"
    if raw in {"coarse", "coarse_only", "coarse-only"}:
        return "coarse_only"
    return "all"


def _repair_prior_chain_value_runtime_feedback_enabled() -> bool:
    raw = str(
        os.environ.get("V2_ECA_REPAIR_PRIOR_CHAIN_VALUE_RUNTIME_FEEDBACK", "1") or ""
    ).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _repair_prior_seed_stale_reject_recovery_enabled() -> bool:
    raw = str(
        os.environ.get("V2_ECA_REPAIR_PRIOR_SEED_STALE_REJECT_RECOVERY", "0") or ""
    ).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _repair_prior_seed_stale_reject_min_repeats_from_env(default: int = 8) -> int:
    try:
        return max(1, int(os.environ.get("V2_ECA_REPAIR_PRIOR_SEED_STALE_REJECT_MIN_REPEATS", default)))
    except (TypeError, ValueError):
        return default


def _repair_prior_seed_stale_reject_floor_score_from_env(default: float = 0.65) -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("V2_ECA_REPAIR_PRIOR_SEED_STALE_REJECT_FLOOR_SCORE", default))))
    except (TypeError, ValueError):
        return default


def _repair_prior_context_frontier_lease_enabled() -> bool:
    raw = str(
        os.environ.get("V2_ECA_REPAIR_PRIOR_CONTEXT_FRONTIER_LEASE", "0") or ""
    ).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _repair_prior_context_frontier_lease_ttl_from_env(default: int = 4) -> int:
    try:
        return max(1, int(os.environ.get("V2_ECA_REPAIR_PRIOR_CONTEXT_FRONTIER_LEASE_TTL", default)))
    except (TypeError, ValueError):
        return default


def _repair_prior_context_frontier_lease_max_rank_from_env(default: int = 8) -> int:
    try:
        return max(1, int(os.environ.get("V2_ECA_REPAIR_PRIOR_CONTEXT_FRONTIER_LEASE_MAX_RANK", default)))
    except (TypeError, ValueError):
        return default


def _repair_prior_context_family_backoff_enabled() -> bool:
    raw = str(
        os.environ.get("V2_ECA_REPAIR_PRIOR_CONTEXT_FAMILY_BACKOFF", "0") or ""
    ).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _repair_prior_chain_continuation_enabled() -> bool:
    raw = str(
        os.environ.get("V2_ECA_REPAIR_PRIOR_CHAIN_CONTINUATION", "0") or ""
    ).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _repair_prior_chain_continuation_ttl_from_env(default: int = 3) -> int:
    try:
        return max(1, int(os.environ.get("V2_ECA_REPAIR_PRIOR_CHAIN_CONTINUATION_TTL", default)))
    except (TypeError, ValueError):
        return default


def _repair_prior_chain_continuation_bands_from_env(
    default: str = "visible_object",
) -> set[str]:
    raw = str(
        os.environ.get("V2_ECA_REPAIR_PRIOR_CHAIN_CONTINUATION_BANDS", default)
        or default
    )
    return {
        part.strip()
        for part in raw.replace(";", ",").split(",")
        if part.strip()
    }


def _repair_prior_chain_slot_binding_enabled() -> bool:
    raw = str(
        os.environ.get("V2_ECA_REPAIR_PRIOR_CHAIN_SLOT_BINDING", "1") or ""
    ).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _repair_prior_chain_slot_binding_max_distance_from_env(default: int = 12) -> int:
    try:
        return max(0, int(os.environ.get("V2_ECA_REPAIR_PRIOR_CHAIN_SLOT_BINDING_MAX_DISTANCE", default)))
    except (TypeError, ValueError):
        return default


def _repair_prior_chain_slot_ledger_enabled() -> bool:
    raw = str(
        os.environ.get("V2_ECA_REPAIR_PRIOR_CHAIN_SLOT_LEDGER", "1") or ""
    ).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _repair_prior_chain_slot_ledger_ttl_from_env(default: int = 128) -> int:
    try:
        return max(1, int(os.environ.get("V2_ECA_REPAIR_PRIOR_CHAIN_SLOT_LEDGER_TTL", default)))
    except (TypeError, ValueError):
        return default


def _repair_prior_objective_slot_enabled() -> bool:
    raw = str(
        os.environ.get("V2_ECA_REPAIR_PRIOR_OBJECTIVE_SLOT", "1") or ""
    ).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _repair_prior_objective_slot_ranking_enabled() -> bool:
    raw = str(
        os.environ.get("V2_ECA_REPAIR_PRIOR_OBJECTIVE_SLOT_RANKING", "1") or ""
    ).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _repair_prior_objective_slot_candidate_binding_enabled() -> bool:
    raw = str(
        os.environ.get("V2_ECA_REPAIR_PRIOR_OBJECTIVE_SLOT_CANDIDATE_BINDING", "1") or ""
    ).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _repair_prior_objective_slot_candidate_binding_max_distance_from_env(
    default: int = 12,
) -> int:
    try:
        return max(
            0,
            int(
                os.environ.get(
                    "V2_ECA_REPAIR_PRIOR_OBJECTIVE_SLOT_CANDIDATE_BINDING_MAX_DISTANCE",
                    default,
                )
            ),
        )
    except (TypeError, ValueError):
        return default


def _repair_prior_objective_slot_candidate_exposure_enabled() -> bool:
    raw = str(
        os.environ.get("V2_ECA_REPAIR_PRIOR_OBJECTIVE_SLOT_CANDIDATE_EXPOSURE", "0") or ""
    ).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _repair_prior_objective_slot_exposure_admission_mode_from_env() -> str:
    raw = str(
        os.environ.get(
            "V2_ECA_REPAIR_PRIOR_OBJECTIVE_SLOT_EXPOSURE_ADMISSION",
            "shadow",
        )
        or ""
    ).strip().lower()
    if raw in {"1", "true", "yes", "on", "guard", "guarded"}:
        return "guarded"
    if raw in {"shadow", "observe", "observer", "metric", "metrics"}:
        return "shadow"
    return "off"


def _repair_prior_objective_slot_exposure_admission_max_samples_from_env(
    default: int = 8,
) -> int:
    try:
        return max(
            0,
            int(
                os.environ.get(
                    "V2_ECA_REPAIR_PRIOR_OBJECTIVE_SLOT_EXPOSURE_ADMISSION_MAX_SAMPLES",
                    default,
                )
            ),
        )
    except (TypeError, ValueError):
        return default


def _repair_prior_objective_slot_exposure_probe_mode_from_env() -> str:
    raw = str(
        os.environ.get(
            "V2_ECA_REPAIR_PRIOR_OBJECTIVE_SLOT_EXPOSURE_PROBE",
            "shadow",
        )
        or ""
    ).strip().lower()
    if raw in {"1", "true", "yes", "on", "guard", "guarded"}:
        return "guarded"
    if raw in {"shadow", "observe", "observer", "metric", "metrics"}:
        return "shadow"
    return "off"


def _repair_prior_objective_slot_exposure_probe_execute_enabled() -> bool:
    raw = str(
        os.environ.get(
            "V2_ECA_REPAIR_PRIOR_OBJECTIVE_SLOT_EXPOSURE_PROBE_EXECUTE",
            "0",
        )
        or ""
    ).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _repair_prior_objective_slot_exposure_probe_min_score_from_env(
    default: float = 0.70,
) -> float:
    try:
        return max(
            0.0,
            min(
                1.0,
                float(
                    os.environ.get(
                        "V2_ECA_REPAIR_PRIOR_OBJECTIVE_SLOT_EXPOSURE_PROBE_MIN_SCORE",
                        default,
                    )
                ),
            ),
        )
    except (TypeError, ValueError):
        return default


def _repair_prior_objective_slot_exposure_probe_max_per_episode_from_env(
    default: int = 1,
) -> int:
    try:
        return max(
            0,
            int(
                os.environ.get(
                    "V2_ECA_REPAIR_PRIOR_OBJECTIVE_SLOT_EXPOSURE_PROBE_MAX_PER_EPISODE",
                    default,
                )
            ),
        )
    except (TypeError, ValueError):
        return default


def _repair_distillation_max_examples_from_env(default: int = 64) -> int:
    try:
        return max(0, int(os.environ.get("V2_ECA_REPAIR_DISTILLATION_MAX_EXAMPLES", default)))
    except (TypeError, ValueError):
        return default


def _teacher_coherence_max_samples_from_env(default: int = 32) -> int:
    """Phase 47 — bounded ring of recorded coherence comparisons."""

    try:
        return max(0, int(os.environ.get("V2_ECA_TEACHER_COHERENCE_MAX_SAMPLES", default)))
    except (TypeError, ValueError):
        return default


def _shadow_injection_mode_from_env(default: str = "off") -> str:
    """Phase 48 — passive shadow-injection telemetry mode.

    ``off``      : no shadow tracking (default).
    ``passive``  : record per-decision (eca_followed, score_at_decision,
                   payload_distance) at gate-fire time and drain via a
                   K-frame post-hoc score-lift attribution. No replay
                   environment is created and no action is mutated.
    """

    raw = str(os.environ.get("V2_ECA_REPAIR_GATE_SHADOW_INJECTION", default) or "").strip().lower()
    if raw not in {"off", "passive"}:
        return "off"
    return raw


def _shadow_injection_horizon_from_env(default: int = 5) -> int:
    """Phase 48 — K horizon for post-hoc score-lift attribution.

    Default 5 matches Phase 44's K-step harness, where lp85 first
    showed 100% causal lift.
    """

    try:
        return max(1, int(os.environ.get("V2_ECA_SHADOW_INJECTION_HORIZON", default)))
    except (TypeError, ValueError):
        return default


def _shadow_injection_max_samples_from_env(default: int = 64) -> int:
    """Phase 48 — bounded ring of recorded shadow decisions (post-drain)."""

    try:
        return max(0, int(os.environ.get("V2_ECA_SHADOW_INJECTION_MAX_SAMPLES", default)))
    except (TypeError, ValueError):
        return default


def _adapter_handoff_reset_on_empty_from_env() -> bool:
    """Phase 68 — when handoff cache is exhausted AND adapter returns
    empty plan, emit RESET so the adapter can re-perceive from a clean
    canonical state. Default off; opt-in for Pattern B games where
    cached plans don't reach a win state and local controller fallback
    actions corrupt puzzle state between adapter clicks.
    """

    raw = str(os.environ.get("V2_ECA_ADAPTER_HANDOFF_RESET_ON_EMPTY", "off") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _adapter_handoff_reset_grace_from_env(default: int = 30) -> int:
    """Phase 69 — frames-since-last-score-event guard before RESET.

    When score advanced within this many frames, don't fire RESET.
    Prevents losing progress on a level-transition where adapter
    briefly emits empty plans during animation. Default 30 frames.
    """

    try:
        return max(0, int(os.environ.get("V2_ECA_ADAPTER_HANDOFF_RESET_GRACE", default)))
    except (TypeError, ValueError):
        return default


def _adapter_handoff_games_from_env() -> frozenset[str]:
    """Phase 58 — per-game adapter handoff whitelist.

    When ``game_id`` is in this set, the bridge offers
    ``adapter_handoff_action`` regardless of bridge mode (provided
    the perception adapter has a non-empty plan). The runtime
    layer can then override ECA's authoritative action with the
    adapter's first plan entry — the actual guarded action mutation.

    Default: empty set (handoff disabled). Set via
    ``V2_ECA_ADAPTER_HANDOFF_GAMES=lp85,sk48,ft09`` or similar
    based on Phase 57 K=100 causal evidence.
    """

    raw = str(os.environ.get("V2_ECA_ADAPTER_HANDOFF_GAMES", "") or "")
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return frozenset(parts)


def _visible_click_candidate_max_from_env(default: int = 64) -> int:
    try:
        return max(0, int(os.environ.get("V2_ECA_VISIBLE_CLICK_CANDIDATE_MAX", default)))
    except (TypeError, ValueError):
        return default


def _score_progress_template_candidates_enabled() -> bool:
    raw = str(os.environ.get("V2_ECA_SCORE_PROGRESS_TEMPLATE_CANDIDATES", "1") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _frame_from_observation(observation: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(observation, dict):
        return None
    frame = observation.get("frame")
    if frame is None:
        return None
    if isinstance(frame, list) and frame:
        frame = frame[-1]
    return np.asarray(frame)


def _action_name(action: Any) -> str:
    raw = getattr(action, "name", None)
    if raw:
        return str(raw).strip().upper()
    raw = getattr(action, "value", None)
    if raw:
        return str(raw).strip().upper()
    return str(action or "").strip().upper()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _bucket_int(value: int, *, width: int = 8) -> str:
    try:
        val = int(value)
    except (TypeError, ValueError):
        val = 0
    if width <= 0:
        return str(val)
    start = (val // int(width)) * int(width)
    return f"{start}..{start + int(width) - 1}"


def _score_bucket(value: Any, *, width: int = 10) -> str:
    try:
        scaled = int(round(float(value) * 100.0))
    except (TypeError, ValueError):
        scaled = 0
    return _bucket_int(scaled, width=width)


def _operator_from_action(action: Any) -> str:
    action_name = _action_name(action)
    if action_name == "ACTION6":
        return "click"
    if action_name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
        return "move"
    if action_name == "ACTION5":
        return "auxiliary"
    if action_name:
        return action_name.lower()
    return ""


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    return str(value)


def _payload_for_compare(payload: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(payload or {})
    if not raw:
        return {}
    if isinstance(raw.get("parameters"), dict):
        raw = dict(raw.get("parameters") or {})
    return {
        str(key): _jsonable_value(value)
        for key, value in raw.items()
        if str(key) not in {"data_keys", "reasoning", "action"}
    }


def _click_point_from_payload(payload: dict[str, Any] | None) -> tuple[int, int] | None:
    raw = dict(payload or {})
    if isinstance(raw.get("parameters"), dict):
        raw = dict(raw.get("parameters") or {})
    if "x" in raw and "y" in raw:
        try:
            return (int(raw["x"]), int(raw["y"]))
        except (TypeError, ValueError):
            return None
    if "col" in raw and "row" in raw:
        try:
            return (int(raw["col"]), int(raw["row"]))
        except (TypeError, ValueError):
            return None
    return None


def _candidate_distance(
    point: tuple[int, int] | None,
    candidate: dict[str, Any],
) -> int | None:
    if point is None:
        return None
    try:
        return abs(int(point[0]) - int(candidate.get("x"))) + abs(
            int(point[1]) - int(candidate.get("y"))
        )
    except (TypeError, ValueError):
        return None


def _rank_visible_click_candidate(
    candidates: list[dict[str, Any]],
    point: tuple[int, int] | None,
) -> dict[str, Any]:
    if point is None or not candidates:
        return {
            "rank": -1,
            "distance": -1,
            "kind": "",
            "candidate_id": "",
            "in_candidates": False,
        }
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        distance = _candidate_distance(point, candidate)
        if distance is None:
            continue
        ranked.append((int(distance), index, candidate))
    if not ranked:
        return {
            "rank": -1,
            "distance": -1,
            "kind": "",
            "candidate_id": "",
            "in_candidates": False,
        }
    ranked.sort(key=lambda item: (item[0], item[1]))
    distance, index, candidate = ranked[0]
    return {
        "rank": int(index),
        "distance": int(distance),
        "kind": str(candidate.get("kind", "") or ""),
        "candidate_band": str(candidate.get("candidate_band", "") or ""),
        "candidate_id": str(candidate.get("candidate_id", "") or ""),
        "x": int(candidate.get("x", 0) or 0),
        "y": int(candidate.get("y", 0) or 0),
        "in_candidates": bool(distance == 0),
    }


def _frame_change_summary(
    pre_frame: np.ndarray | None,
    post_frame: np.ndarray | None,
    *,
    click_point: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if pre_frame is None or post_frame is None:
        return {
            "frame_compared": False,
            "changed_pixel_count": 0,
            "changed_fraction": 0.0,
            "changed_bbox": None,
            "click_to_changed_bbox_distance": -1,
            "click_near_changed_bbox": False,
        }
    pre = np.asarray(pre_frame)
    post = np.asarray(post_frame)
    if pre.shape != post.shape:
        return {
            "frame_compared": False,
            "shape_mismatch": True,
            "pre_shape": list(pre.shape),
            "post_shape": list(post.shape),
            "changed_pixel_count": 0,
            "changed_fraction": 0.0,
            "changed_bbox": None,
            "click_to_changed_bbox_distance": -1,
            "click_near_changed_bbox": False,
        }
    if pre.ndim >= 3:
        changed = np.any(pre != post, axis=tuple(range(2, pre.ndim)))
    else:
        changed = pre != post
    changed_count = int(np.count_nonzero(changed))
    total = int(changed.size) if changed.size else 0
    bbox = None
    click_distance = -1
    click_near = False
    if changed_count > 0:
        ys, xs = np.where(changed)
        min_y, max_y = int(ys.min()), int(ys.max())
        min_x, max_x = int(xs.min()), int(xs.max())
        bbox = {
            "min_y": min_y,
            "min_x": min_x,
            "max_y": max_y,
            "max_x": max_x,
            "height": int(max_y - min_y + 1),
            "width": int(max_x - min_x + 1),
            "area": int((max_y - min_y + 1) * (max_x - min_x + 1)),
        }
        if click_point is not None:
            click_x, click_y = int(click_point[0]), int(click_point[1])
            dx = 0
            if click_x < min_x:
                dx = min_x - click_x
            elif click_x > max_x:
                dx = click_x - max_x
            dy = 0
            if click_y < min_y:
                dy = min_y - click_y
            elif click_y > max_y:
                dy = click_y - max_y
            click_distance = int(dx + dy)
            click_near = bool(click_distance <= 8)
    return {
        "frame_compared": True,
        "changed_pixel_count": int(changed_count),
        "changed_fraction": round(float(changed_count) / float(total), 6) if total else 0.0,
        "changed_bbox": bbox,
        "click_to_changed_bbox_distance": int(click_distance),
        "click_near_changed_bbox": bool(click_near),
    }


def _normalize_action_entry(entry: Any) -> dict[str, Any]:
    if isinstance(entry, tuple) and entry:
        action = entry[0]
        data = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
    elif isinstance(entry, dict):
        action = entry.get("action")
        data = {k: v for k, v in entry.items() if k != "action"}
    else:
        action = entry
        data = {}

    normalized: dict[str, Any] = {"action": _action_name(action)}
    parameters = {str(key): _jsonable_value(value) for key, value in data.items()}
    for key in ("x", "y", "row", "col"):
        if key in parameters:
            try:
                normalized[key] = int(parameters[key])
            except (TypeError, ValueError):
                normalized[key] = parameters[key]
    if parameters:
        normalized["parameters"] = parameters
        normalized["data_keys"] = sorted(parameters.keys())
    return normalized


def _tier_from_source(source: str, *, available: bool) -> str:
    if not available:
        return ""
    normalized = source.strip().lower()
    if "primitive_binding" in normalized:
        return "frame_conditioned_game_primitive"
    if any(token in normalized for token in ("fallback", "hardcoded", "precomputed", "replay")):
        return "tier_c_fixed_plan"
    if any(token in normalized for token in ("frame_conditioned", "state_policy", "per_game")):
        return "frame_conditioned_per_game_solver"
    if any(token in normalized for token in ("tier_a", "bfs", "astar", "a_star")):
        return "tier_a_runtime_solver"
    if any(token in normalized for token in ("runtime", "solver", "rule", "program")):
        return "runtime_solver_observed"
    return "runtime_solver_observed"


def _plan_diagnostics_from_module(module: Any | None) -> dict[str, Any]:
    if module is None:
        return {}
    diagnostics = getattr(module, "LAST_PLAN_DIAGNOSTICS", {})
    if not isinstance(diagnostics, Mapping):
        return {}
    return {
        str(key): _jsonable_value(value)
        for key, value in diagnostics.items()
        if str(key)
    }


@dataclass
class PerceptionSolverBridge:
    """Call a game's perception-driven solver in ECA observer mode."""

    game_id: str | None
    max_preview: int | None = None
    mode: str = field(default_factory=solver_bridge_mode)

    def __post_init__(self) -> None:
        self.game_id_normalized = str(self.game_id or "").strip().lower()
        self.module_name = PERCEPTION_SOLVER_REGISTRY.get(self.game_id_normalized, "")
        self.max_preview = _max_preview_from_env() if self.max_preview is None else int(self.max_preview)
        self.enabled = bool(self.mode in {"observer", "advisory", "guarded"} and self.module_name)
        selected_games = _selected_games_from_env()
        if selected_games and self.game_id_normalized not in selected_games:
            self.enabled = False
        self.module: Any | None = None
        self.import_error = ""
        self.observed_steps = 0
        self.available_steps = 0
        self.error_steps = 0
        self.plan_action_total = 0
        self.source_histogram: Counter[str] = Counter()
        self.generalisation_tier_histogram: Counter[str] = Counter()
        self.guard_eval_count = 0
        self.guard_adopted_count = 0
        self.guard_rejected_count = 0
        self.guard_reject_reason_histogram: Counter[str] = Counter()
        self.last_guard_decision: dict[str, Any] = {}
        self.last_plan: list[dict[str, Any]] = []
        self.last_advice = self._empty_advice()
        self._wa30_fill_zones_cache: Any | None = None
        self._wa30_last_score = -1
        # ── Phase 42: Target-binding repair gate (advisory only) ──────
        self._current_proposal: dict[str, Any] = {}
        self._current_proposal_features: dict[str, Any] = {}
        self.repair_gate_mode_value = repair_gate_mode()
        self.repair_gate_active = bool(
            self.enabled
            and self.mode == "advisory"
            and repair_gate_enabled()
        )
        # Phase 46 — selective gating env config. Read up front so the
        # bridge can pass the same stuck threshold to its own ledger.
        self.repair_gate_selective_mode_value = _selective_mode_from_env()
        self.repair_gate_selective_stuck_threshold = (
            _selective_stuck_threshold_from_env()
        )
        self.repair_gate_selective_high_risk_threshold = (
            _selective_high_risk_threshold_from_env()
        )
        self.repair_gate_selective_ood_max = _selective_ood_max_from_env()
        self.repair_gate: TargetBindingRepairGate | None = (
            load_default_repair_gate(
                mode=self.repair_gate_mode_value,
                selective_mode=self.repair_gate_selective_mode_value,
                stuck_threshold=self.repair_gate_selective_stuck_threshold,
                selective_high_risk_threshold=(
                    self.repair_gate_selective_high_risk_threshold
                ),
                selective_ood_max=self.repair_gate_selective_ood_max,
            )
            if self.repair_gate_active
            else None
        )
        self.repair_gate_eval_count = 0
        self.repair_gate_recommended_count = 0
        self.repair_gate_payload_repaired_count = 0
        self.repair_gate_objective_repaired_count = 0
        # Phase 55 — score-aware suppression diagnostics.
        self.repair_gate_score_aware_engaged_count = 0
        self.repair_gate_score_aware_evaluations_with_baseline = 0
        # Phase 56 — static game-baseline parity diagnostic.
        self.repair_gate_game_baseline_parity_engaged_count = 0
        self.repair_gate_action_compare_count = 0
        self.repair_gate_action_agreement_count = 0
        self.repair_gate_action_after_repair_agreement_count = 0
        self.repair_gate_payload_compare_count = 0
        self.repair_gate_payload_agreement_count = 0
        self.repair_gate_payload_after_repair_agreement_count = 0
        self.repair_execution_mode_value = _repair_execution_mode_from_env()
        self.repair_execution_games = _repair_execution_games_from_env()
        self.repair_guard_eval_count = 0
        self.repair_guard_adopted_count = 0
        self.repair_guard_rejected_count = 0
        self.repair_guard_reject_reason_histogram: Counter[str] = Counter()
        self.last_repair_guard_decision: dict[str, Any] = {}
        self.repair_distillation_max_examples = _repair_distillation_max_examples_from_env()
        self.repair_distillation_example_count = 0
        self.repair_distillation_executed_count = 0
        self.repair_distillation_samples: list[dict[str, Any]] = []
        self.repair_distillation_action_pair_histogram: Counter[str] = Counter()
        self.repair_distillation_target_translation_status_histogram: Counter[str] = Counter()
        self.repair_distillation_label_histogram: Counter[str] = Counter()
        self._last_visible_click_candidates: list[dict[str, Any]] = []
        self._visible_click_candidate_memory: dict[tuple[int, int], dict[str, Any]] = {}
        self._visible_click_candidate_memory_score: int | None = None
        self._teacher_assisted_click_history: list[tuple[int, int]] = []
        self._proposal_click_history: list[tuple[int, int]] = []
        self._last_repair_distillation_teacher_point: tuple[int, int] | None = None
        self._last_repair_distillation_teacher_point_eval_index = 0
        # ── Phase 47: Teacher coherence tracker (advisory only) ───────
        # We compare each frame's `teacher_first_action` against what
        # the teacher predicted as the *next* action after the ECA-
        # chosen action one step earlier. Two predictions are tracked:
        # ``after_eca`` (assumes ECA's actually-chosen action was taken)
        # and ``after_teacher`` (assumes teacher's first_action was
        # taken). The latter is the teacher's *self-consistency* under
        # its own plan and is independent of ECA actions.
        self._last_predicted_next_action_after_eca: dict[str, Any] | None = None
        self._last_predicted_next_action_after_teacher: dict[str, Any] | None = None
        self._last_teacher_first_action: str = ""
        self._last_teacher_first_payload: dict[str, Any] = {}
        self._last_eca_action_for_coherence: str = ""
        self._last_eca_followed_teacher: bool | None = None
        self.teacher_coherence_compare_count = 0
        self.teacher_coherence_match_count = 0
        self.teacher_coherence_mismatch_count = 0
        self.teacher_coherence_payload_compare_count = 0
        self.teacher_coherence_payload_match_count = 0
        self.teacher_coherence_compare_when_eca_followed = 0
        self.teacher_coherence_match_when_eca_followed = 0
        self.teacher_coherence_compare_when_eca_diverged = 0
        self.teacher_coherence_match_when_eca_diverged = 0
        self.teacher_self_stability_compare_count = 0
        self.teacher_self_stability_match_count = 0
        self.teacher_coherence_samples: list[dict[str, Any]] = []
        self.teacher_coherence_max_samples = _teacher_coherence_max_samples_from_env()
        # ── Phase 48: Shadow injection telemetry (advisory only) ──────
        # When the gate fires, we record a "pending shadow decision"
        # capturing (eval_idx, score_at_decision, eca_followed,
        # payload_distance). On each subsequent observe(), we drain
        # decisions older than ``shadow_injection_horizon`` frames and
        # bin them into followed/diverged categories with score lift.
        # No replay env is created; no action is mutated.
        self.shadow_injection_mode_value = _shadow_injection_mode_from_env()
        self.shadow_injection_horizon = _shadow_injection_horizon_from_env()
        self.shadow_injection_max_samples = _shadow_injection_max_samples_from_env()
        self.shadow_injection_active = bool(
            self.repair_gate_active
            and self.shadow_injection_mode_value == "passive"
        )
        self._pending_shadow_decisions: list[dict[str, Any]] = []
        self.shadow_decision_count = 0
        self.shadow_decision_eca_followed_count = 0
        self.shadow_decision_eca_diverged_count = 0
        self.shadow_drained_followed_count = 0
        self.shadow_drained_diverged_count = 0
        self._shadow_lift_followed_sum = 0.0
        self._shadow_lift_diverged_sum = 0.0
        self.shadow_lift_followed_positive_count = 0
        self.shadow_lift_diverged_positive_count = 0
        self._shadow_payload_distance_sum_followed = 0.0
        self._shadow_payload_distance_count_followed = 0
        self._shadow_payload_distance_sum_diverged = 0.0
        self._shadow_payload_distance_count_diverged = 0
        self.shadow_injection_samples: list[dict[str, Any]] = []
        # ── Phase 58: Adapter handoff (actual action mutation) ──────
        # When game_id is in HANDOFF_GAMES, the bridge offers the
        # adapter's first plan entry as a guarded override action.
        # Phase 57 K=100 causal evidence: lp85/sk48/ft09 hit 100%
        # causal lift -- handoff at any frame would deterministically
        # advance score on these games. Default empty set.
        self.adapter_handoff_games = _adapter_handoff_games_from_env()
        self.adapter_handoff_active = bool(
            self.game_id_normalized
            and self.game_id_normalized in self.adapter_handoff_games
        )
        self.adapter_handoff_eval_count = 0
        self.adapter_handoff_adopted_count = 0
        self.adapter_handoff_rejected_count = 0
        self.adapter_handoff_reject_reason_histogram: Counter[str] = Counter()
        self.last_adapter_handoff_decision: dict[str, Any] = {}
        # Phase 63: plan-state tracking. Pattern A adapters (cn04,
        # ar25, sc25, etc.) regenerate identical plans from scratch
        # each frame because their `perceive_and_plan` always returns
        # "the full L1 plan from this state". Without tracking, the
        # bridge would hand off plan[0] every frame and the adapter
        # would loop on the same action. Tracking advances through
        # the cached plan, refreshing on score change (level transition).
        self._handoff_cached_plan: list[dict[str, Any]] = []
        self._handoff_cached_plan_index: int = 0
        self._handoff_cached_plan_score: int = -1
        self.adapter_handoff_plan_advance_count = 0
        self.adapter_handoff_plan_refresh_count = 0
        self.adapter_handoff_plan_exhaust_count = 0
        # Phase 68: opt-in RESET-on-empty for Pattern B games.
        self.adapter_handoff_reset_on_empty = _adapter_handoff_reset_on_empty_from_env()
        self.adapter_handoff_reset_emitted_count = 0
        # Phase 69: post-score-event grace period before RESET.
        # When score advances, capture the observed_steps so we can
        # gate RESET on "no recent score progress" within the grace
        # window. Prevents RESET from rolling back a recent level-up.
        self.adapter_handoff_reset_grace_steps = _adapter_handoff_reset_grace_from_env()
        self._adapter_handoff_last_score_event_step = -1
        self.adapter_handoff_reset_grace_blocked_count = 0
        self._last_repair_decision_eval_index = 0
        self._last_repair_guard_consumed_eval_index = 0
        self._repair_gate_high_risk_sum = 0.0
        self._repair_gate_high_risk_count = 0
        # Phase 45 — OOD diagnostic metrics
        self.repair_gate_ood_eval_count = 0
        self.repair_gate_ood_high_count = 0
        self._repair_gate_ood_fraction_sum = 0.0
        self.repair_gate_ood_backoff_engaged_count = 0
        # Phase 46 — selective gating diagnostics + stuck-step tracker.
        # ``_stuck_score_history`` keeps the last ``stuck_threshold + 2``
        # observed scores; ``_compute_stuck_steps`` returns the run of
        # consecutive trailing observations with no score gain. The +2
        # padding guarantees the window can always emit
        # ``stuck_threshold`` consecutive identical-score frames after
        # an initial gap.
        from collections import deque  # local import — std lib
        stuck_window = max(int(self.repair_gate_selective_stuck_threshold), 1) + 2
        self._stuck_score_history: deque[int] = deque(maxlen=stuck_window)
        self._stuck_steps_current = 0
        self._stuck_steps_sum = 0
        self._stuck_steps_obs_count = 0
        self.repair_gate_selective_eligible_count = 0
        self.repair_gate_selective_blocked_count = 0
        self.repair_gate_selective_blocked_reason_histogram: Counter[str] = Counter()
        self.last_repair_decision: dict[str, Any] = {}
        self._last_frame: np.ndarray | None = None
        self._last_score = 0
        self._pending_repair_effect: dict[str, Any] = {}
        self.repair_effect_resolved_count = 0
        self.repair_effect_visible_change_count = 0
        self.repair_effect_score_progress_count = 0
        self.repair_prior_observer_eval_count = 0
        self.repair_prior_observer_teacher_contained_count = 0
        self.repair_prior_observer_candidate_count_sum = 0
        self.repair_prior_observer_mrr_sum = 0.0
        self.repair_prior_observer_top1_count = 0
        self.repair_prior_observer_top3_count = 0
        self.repair_prior_observer_top5_count = 0
        self.repair_prior_observer_top10_count = 0
        self.repair_prior_observer_top1_when_contained_count = 0
        self.repair_prior_observer_top3_when_contained_count = 0
        self.repair_prior_observer_top5_when_contained_count = 0
        self.repair_prior_observer_top10_when_contained_count = 0
        self.last_repair_prior_observer_prediction: dict[str, Any] = {}
        self.repair_prior_action_proposal_count = 0
        self.last_repair_prior_action_proposal: dict[str, Any] = {}
        self._last_repair_prior_ranked: list[dict[str, Any]] = []
        self.repair_prior_execution_mode_value = _repair_prior_execution_mode_from_env()
        self.repair_prior_execution_games = _repair_prior_execution_games_from_env()
        self.repair_prior_guard_eval_count = 0
        self.repair_prior_guard_adopted_count = 0
        self.repair_prior_guard_rejected_count = 0
        self.repair_prior_guard_reject_reason_histogram: Counter[str] = Counter()
        self.last_repair_prior_guard_decision: dict[str, Any] = {}
        self._repair_prior_guard_executed_sites: set[tuple[int, int]] = set()
        self._pending_repair_prior_effect: dict[str, Any] = {}
        self.repair_prior_effect_resolved_count = 0
        self.repair_prior_effect_visible_change_count = 0
        self.repair_prior_effect_score_progress_count = 0
        self.last_repair_prior_effect_outcome: dict[str, Any] = {}
        self.repair_prior_effect_chain_credit_count = 0
        self.repair_prior_effect_history: list[dict[str, Any]] = []
        self.repair_prior_chain_value_counts: dict[tuple[str, ...], Counter[str]] = {}
        self.repair_prior_chain_value_runtime_feedback_enabled = (
            _repair_prior_chain_value_runtime_feedback_enabled()
        )
        self.repair_prior_chain_value_seed_path = _repair_prior_chain_value_seed_path_from_env()
        self.repair_prior_chain_value_seed_key_mode = (
            _repair_prior_chain_value_seed_key_mode_from_env()
        )
        self.repair_prior_chain_value_seed_rule_count = 0
        self.repair_prior_chain_value_seed_error = ""
        self.repair_prior_chain_value_rerank_eval_count = 0
        self.repair_prior_chain_value_adjusted_row_count = 0
        self.repair_prior_chain_value_seed_adjusted_row_count = 0
        self.repair_prior_chain_value_top_supported_count = 0
        self.repair_prior_chain_value_top_seed_supported_count = 0
        self.repair_prior_chain_value_top_changed_count = 0
        self.repair_prior_chain_value_top_score_lift_count = 0
        self.repair_prior_chain_value_top_seed_score_lift_count = 0
        self.repair_prior_chain_value_top_score_lift_sum = 0.0
        self.repair_prior_chain_value_top_seed_score_lift_sum = 0.0
        self.repair_prior_chain_value_top_guard_threshold_cross_count = 0
        self.repair_prior_chain_value_top_seed_guard_threshold_cross_count = 0
        self.repair_prior_chain_value_guard_selected_supported_count = 0
        self.repair_prior_chain_value_guard_selected_seed_supported_count = 0
        self.repair_prior_chain_value_guard_selected_score_lift_count = 0
        self.repair_prior_chain_value_guard_selected_seed_score_lift_count = 0
        self.repair_prior_chain_value_guard_selected_threshold_cross_count = 0
        self.repair_prior_chain_value_guard_selected_seed_threshold_cross_count = 0
        self.repair_prior_context_frontier_lease_enabled = (
            _repair_prior_context_frontier_lease_enabled()
        )
        self.repair_prior_context_frontier_lease_ttl = (
            _repair_prior_context_frontier_lease_ttl_from_env()
        )
        self.repair_prior_context_frontier_lease_max_rank = (
            _repair_prior_context_frontier_lease_max_rank_from_env()
        )
        self.repair_prior_context_family_backoff_enabled = (
            _repair_prior_context_family_backoff_enabled()
        )
        self.repair_prior_context_frontier_seen_count = 0
        self.repair_prior_context_frontier_leased_count = 0
        self.repair_prior_context_frontier_injected_count = 0
        self.repair_prior_context_frontier_family_backoff_count = 0
        self.repair_prior_context_frontier_family_backoff_selected_count = 0
        self.repair_prior_context_frontier_selected_count = 0
        self.repair_prior_context_frontier_expired_count = 0
        self.repair_prior_context_frontier_best_rank = 0
        self.repair_prior_unsupported_top_over_context_count = 0
        self._repair_prior_context_frontier_lease: dict[str, Any] = {}
        self._repair_prior_context_family_lease: dict[str, Any] = {}
        self.repair_prior_chain_continuation_enabled = (
            _repair_prior_chain_continuation_enabled()
        )
        self.repair_prior_chain_continuation_ttl = (
            _repair_prior_chain_continuation_ttl_from_env()
        )
        self.repair_prior_chain_continuation_bands = (
            _repair_prior_chain_continuation_bands_from_env()
        )
        self.repair_prior_chain_slot_binding_enabled = (
            _repair_prior_chain_slot_binding_enabled()
        )
        self.repair_prior_chain_slot_binding_max_distance = (
            _repair_prior_chain_slot_binding_max_distance_from_env()
        )
        self.repair_prior_chain_slot_ledger_enabled = (
            _repair_prior_chain_slot_ledger_enabled()
        )
        self.repair_prior_chain_slot_ledger_ttl = (
            _repair_prior_chain_slot_ledger_ttl_from_env()
        )
        self.repair_prior_objective_slot_enabled = (
            _repair_prior_objective_slot_enabled()
        )
        self.repair_prior_objective_slot_ranking_enabled = (
            _repair_prior_objective_slot_ranking_enabled()
        )
        self.repair_prior_objective_slot_candidate_binding_enabled = (
            _repair_prior_objective_slot_candidate_binding_enabled()
        )
        self.repair_prior_objective_slot_candidate_binding_max_distance = (
            _repair_prior_objective_slot_candidate_binding_max_distance_from_env()
        )
        self.repair_prior_objective_slot_candidate_exposure_enabled = (
            _repair_prior_objective_slot_candidate_exposure_enabled()
        )
        self.repair_prior_objective_slot_exposure_admission_mode = (
            _repair_prior_objective_slot_exposure_admission_mode_from_env()
        )
        self.repair_prior_objective_slot_exposure_admission_max_samples = (
            _repair_prior_objective_slot_exposure_admission_max_samples_from_env()
        )
        self.repair_prior_objective_slot_exposure_probe_mode = (
            _repair_prior_objective_slot_exposure_probe_mode_from_env()
        )
        self.repair_prior_objective_slot_exposure_probe_execute_enabled = (
            _repair_prior_objective_slot_exposure_probe_execute_enabled()
        )
        self.repair_prior_objective_slot_exposure_probe_min_score = (
            _repair_prior_objective_slot_exposure_probe_min_score_from_env()
        )
        self.repair_prior_objective_slot_exposure_probe_max_per_episode = (
            _repair_prior_objective_slot_exposure_probe_max_per_episode_from_env()
        )
        self.repair_prior_chain_continuation_open_count = 0
        self.repair_prior_chain_continuation_active_count = 0
        self.repair_prior_chain_continuation_supported_count = 0
        self.repair_prior_chain_continuation_selected_count = 0
        self.repair_prior_chain_continuation_stale_reject_recovery_count = 0
        self.repair_prior_chain_slot_binding_supported_count = 0
        self.repair_prior_chain_slot_binding_rejected_count = 0
        self.repair_prior_chain_slot_binding_selected_count = 0
        self.repair_prior_chain_slot_binding_best_distance = -1
        self.repair_prior_chain_slot_ledger_open_count = 0
        self.repair_prior_chain_slot_ledger_supported_count = 0
        self.repair_prior_chain_slot_ledger_selected_count = 0
        self.repair_prior_chain_slot_ledger_closed_count = 0
        self.repair_prior_chain_slot_ledger_expired_count = 0
        self.repair_prior_chain_slot_ledger_consumed_reject_count = 0
        self.repair_prior_chain_slot_ledger_delayed_credit_count = 0
        self.repair_prior_chain_slot_ledger_credited_count = 0
        self.repair_prior_chain_slot_ledger_late_credit_count = 0
        self.repair_prior_objective_slot_open_count = 0
        self.repair_prior_objective_slot_supported_count = 0
        self.repair_prior_objective_slot_selected_count = 0
        self.repair_prior_objective_slot_credited_count = 0
        self.repair_prior_objective_slot_ranked_count = 0
        self.repair_prior_objective_slot_top_ranked_count = 0
        self.repair_prior_objective_slot_best_priority = 0.0
        self.repair_prior_objective_slot_active_max_count = 0
        self.repair_prior_objective_slot_multi_active_eval_count = 0
        self.repair_prior_objective_slot_candidate_conflict_count = 0
        self.repair_prior_objective_slot_top_changed_count = 0
        self.repair_prior_objective_slot_local_score_override_count = 0
        self.repair_prior_objective_slot_candidate_bound_count = 0
        self.repair_prior_objective_slot_candidate_binding_rejected_count = 0
        self.repair_prior_objective_slot_candidate_binding_best_distance = -1
        self.repair_prior_objective_slot_candidate_exposed_count = 0
        self.repair_prior_objective_slot_candidate_exposure_slot_count = 0
        self.repair_prior_objective_slot_candidate_exposure_existing_slot_count = 0
        self.repair_prior_objective_slot_candidate_bound_slot_max_count = 0
        self.repair_prior_objective_slot_candidate_bound_slot_histogram: Counter[str] = Counter()
        self.repair_prior_objective_slot_exposed_selected_count = 0
        self.repair_prior_objective_slot_exposed_visible_change_count = 0
        self.repair_prior_objective_slot_exposed_score_progress_count = 0
        self.repair_prior_objective_slot_exposed_no_visible_change_count = 0
        self.repair_prior_objective_slot_exposed_stale_reject_count = 0
        self.repair_prior_objective_slot_exposed_reject_reason_histogram: Counter[str] = Counter()
        self.repair_prior_objective_slot_exposure_admission_eval_count = 0
        self.repair_prior_objective_slot_exposure_admission_eligible_count = 0
        self.repair_prior_objective_slot_exposure_admission_admitted_count = 0
        self.repair_prior_objective_slot_exposure_admission_blocked_count = 0
        self.repair_prior_objective_slot_exposure_admission_reason_histogram: Counter[str] = Counter()
        self.repair_prior_objective_slot_exposure_admission_samples: list[dict[str, Any]] = []
        self.repair_prior_objective_slot_exposure_probe_eval_count = 0
        self.repair_prior_objective_slot_exposure_probe_eligible_count = 0
        self.repair_prior_objective_slot_exposure_probe_admitted_count = 0
        self.repair_prior_objective_slot_exposure_probe_blocked_count = 0
        self.repair_prior_objective_slot_exposure_probe_visible_change_count = 0
        self.repair_prior_objective_slot_exposure_probe_score_progress_count = 0
        self.repair_prior_objective_slot_exposure_probe_no_visible_change_count = 0
        self.repair_prior_objective_slot_exposure_probe_sandboxed_visible_change_count = 0
        self.repair_prior_objective_slot_exposure_probe_reason_histogram: Counter[str] = Counter()
        self.repair_prior_objective_slot_exposure_probe_samples: list[dict[str, Any]] = []
        self._repair_prior_chain_continuation: dict[str, Any] = {}
        self._repair_prior_chain_slot_ledger: dict[str, Any] = {}
        self._repair_prior_chain_slot_ledgers: dict[str, dict[str, Any]] = {}
        self._repair_prior_objective_slot_ledgers: dict[str, dict[str, Any]] = {}
        self.repair_prior_seed_stale_reject_recovery_enabled = (
            _repair_prior_seed_stale_reject_recovery_enabled()
        )
        self.repair_prior_seed_stale_reject_recovery_count = 0
        self.repair_prior_seed_stale_reject_repeat_max = 0
        self._repair_prior_seed_stale_reject_counts: Counter[tuple[int, int, str]] = Counter()
        self._load_repair_prior_chain_value_seed()
        if self.enabled:
            try:
                self.module = importlib.import_module(self.module_name)
            except Exception as exc:  # pragma: no cover - defensive runtime path
                self.import_error = str(exc)
                self.enabled = False
                self.last_advice = self._empty_advice(error=self.import_error)

    def _empty_advice(self, *, error: str = "") -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "available": False,
            "game_id": self.game_id_normalized,
            "module": self.module_name,
            "plan_source": "",
            "generalisation_tier": "",
            "binding_id": "",
            "primitive_name": "",
            "adapter_free": False,
            "plan_diagnostics": {},
            "action_count": 0,
            "first_action": "",
            "preview": [],
            "error": error,
        }

    def set_current_proposal(
        self,
        proposal: dict[str, Any] | None,
        *,
        features: dict[str, Any] | None = None,
    ) -> None:
        """Cache the latest ECA ControlIR proposal for the repair gate.

        Phase 42 advisory only: the cached proposal is compared against
        teacher advice the next time :meth:`observe` runs. Mutating the
        proposal here does not feed back into ECA's authoritative
        action path.
        """

        if not isinstance(proposal, dict):
            self._current_proposal = {}
            self._current_proposal_features = {}
            return
        self._current_proposal = dict(proposal)
        self._current_proposal_features = dict(features or {})

    def _visible_click_candidates_from_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Return compact frame-only click candidates for repair distillation."""

        if self.game_id_normalized != "r11l":
            return []
        max_candidates = _visible_click_candidate_max_from_env()
        if max_candidates <= 0:
            return []
        try:
            from arcgym.perception.r11l_perception import perceive_r11l

            perceived = perceive_r11l(frame)
        except Exception:
            return []

        rows: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()

        def _colors(value: Any) -> list[int]:
            if value is None:
                return []
            if isinstance(value, (set, list, tuple)):
                out: list[int] = []
                for item in value:
                    try:
                        out.append(int(item))
                    except (TypeError, ValueError):
                        continue
                return sorted(set(out))
            try:
                return [int(value)]
            except (TypeError, ValueError):
                return []

        def _add_xy(
            kind: str,
            x: int,
            y: int,
            *,
            color: Any = None,
            colors: Any = None,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            if not (0 <= int(x) < 64 and 0 <= int(y) < 64):
                return
            key = (x, y)
            if key in seen:
                return
            seen.add(key)
            row = {
                "kind": str(kind),
                "x": int(x),
                "y": int(y),
                "color": _jsonable_value(color),
                "colors": _colors(colors if colors is not None else color),
            }
            if metadata:
                for meta_key, meta_value in metadata.items():
                    row[str(meta_key)] = _jsonable_value(meta_value)
            rows.append(row)

        def _add(
            kind: str,
            pos: Any,
            *,
            offset: int,
            color: Any = None,
            colors: Any = None,
        ) -> None:
            if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                return
            try:
                x = int(pos[0]) + int(offset)
                y = int(pos[1]) + int(offset)
            except (TypeError, ValueError):
                return
            _add_xy(kind, x, y, color=color, colors=colors)

        def _candidate_priority(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
            kind = str(row.get("kind", "") or "")
            try:
                raw_template_index = row.get("candidate_template_index", 9999)
                template_index = 9999 if raw_template_index in (None, "") else int(raw_template_index)
            except (TypeError, ValueError):
                template_index = 9999
            priority_by_kind = {
                "active_piece": 0,
                "inactive_piece": 0,
                "canvas": 0,
                "target": 0,
                "signature_piece": 0,
                "score_progress_template_inactive_piece": 1,
                "score_progress_template_candidate": 1,
                "score_progress_selected_target_offset": 1,
                "target_centroid_destination": 2,
                "target_centroid_destination_jitter": 2,
                "signature_centroid_destination": 2,
                "signature_centroid_destination_jitter": 2,
                "target_destination": 3,
                "signature_waypoint": 3,
                "target_neighborhood": 4,
                "signature_waypoint_neighborhood": 4,
            }
            return (
                int(priority_by_kind.get(kind, 5)),
                int(template_index),
                int(row.get("y", 0) or 0),
                int(row.get("x", 0) or 0),
            )

        def _candidate_band(row: Mapping[str, Any]) -> str:
            kind = str(row.get("kind", "") or "")
            if kind in {
                "active_piece",
                "inactive_piece",
                "canvas",
                "target",
                "signature_piece",
                "signature_waypoint",
            }:
                return "visible_object"
            if kind in {
                "score_progress_template_inactive_piece",
                "score_progress_template_candidate",
                "score_progress_selected_target_offset",
            }:
                return "progress_template"
            return "dense_cloud"

        def _band_budget(env_name: str, default: int) -> int:
            try:
                return max(0, int(os.environ.get(env_name, default)))
            except (TypeError, ValueError):
                return max(0, int(default))

        def _select_candidate_bands(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            visible_budget = min(
                max_candidates,
                _band_budget("V2_ECA_VISIBLE_CLICK_VISIBLE_OBJECT_BUDGET", 18),
            )
            progress_budget = min(
                max_candidates,
                _band_budget("V2_ECA_VISIBLE_CLICK_PROGRESS_TEMPLATE_BUDGET", 24),
            )
            bands: dict[str, list[dict[str, Any]]] = {
                "visible_object": [],
                "progress_template": [],
                "dense_cloud": [],
            }
            for row in candidate_rows:
                decorated = dict(row)
                decorated["candidate_band"] = _candidate_band(decorated)
                bands.setdefault(str(decorated["candidate_band"]), []).append(decorated)
            for band_rows in bands.values():
                band_rows.sort(key=_candidate_priority)

            selected: list[dict[str, Any]] = []
            selected_keys: set[tuple[str, int, int]] = set()

            def take_from_band(band_name: str, budget: int) -> None:
                nonlocal selected
                if budget <= 0 or len(selected) >= max_candidates:
                    return
                remaining_budget = min(int(budget), max_candidates - len(selected))
                taken = 0
                for row in bands.get(band_name, []):
                    key = (str(row.get("kind", "")), int(row.get("x", 0)), int(row.get("y", 0)))
                    if key in selected_keys:
                        continue
                    selected.append(dict(row))
                    selected_keys.add(key)
                    taken += 1
                    if taken >= remaining_budget or len(selected) >= max_candidates:
                        break

            take_from_band("visible_object", visible_budget)
            take_from_band("progress_template", progress_budget)
            take_from_band("dense_cloud", max_candidates - len(selected))

            if len(selected) < max_candidates:
                overflow: list[dict[str, Any]] = []
                for band_name in ("visible_object", "progress_template", "dense_cloud"):
                    overflow.extend(bands.get(band_name, []))
                overflow.sort(
                    key=lambda row: (
                        {"visible_object": 0, "progress_template": 1, "dense_cloud": 2}.get(
                            str(row.get("candidate_band", "")),
                            3,
                        ),
                        *_candidate_priority(row),
                    )
                )
                for row in overflow:
                    key = (str(row.get("kind", "")), int(row.get("x", 0)), int(row.get("y", 0)))
                    if key in selected_keys:
                        continue
                    selected.append(dict(row))
                    selected_keys.add(key)
                    if len(selected) >= max_candidates:
                        break

            return selected[:max_candidates]

        def _add_centroid_destination_candidates(
            anchor_x: int,
            anchor_y: int,
            *,
            kind: str,
            color: Any = None,
            colors: Any = None,
        ) -> None:
            def add_top_left(pos: tuple[int, int]) -> None:
                click_x = int(pos[0]) + 2
                click_y = int(pos[1]) + 2
                base_metadata = {
                    "candidate_origin_anchor_x": int(anchor_x),
                    "candidate_origin_anchor_y": int(anchor_y),
                    "candidate_target_dx": int(click_x) - (int(anchor_x) + 3),
                    "candidate_target_dy": int(click_y) - (int(anchor_y) + 3),
                    "candidate_jitter_dx": 0,
                    "candidate_jitter_dy": 0,
                }
                _add_xy(
                    kind,
                    click_x,
                    click_y,
                    color=color,
                    colors=colors,
                    metadata=base_metadata,
                )
                for jy in (-1, 0, 1):
                    for jx in (-1, 0, 1):
                        if jx == 0 and jy == 0:
                            continue
                        jitter_metadata = dict(base_metadata)
                        jitter_metadata["candidate_jitter_dx"] = int(jx)
                        jitter_metadata["candidate_jitter_dy"] = int(jy)
                        jitter_metadata["candidate_target_dx"] = int(click_x + jx) - (
                            int(anchor_x) + 3
                        )
                        jitter_metadata["candidate_target_dy"] = int(click_y + jy) - (
                            int(anchor_y) + 3
                        )
                        _add_xy(
                            f"{kind}_jitter",
                            click_x + jx,
                            click_y + jy,
                            color=color,
                            colors=colors,
                            metadata=jitter_metadata,
                        )

            # Mirrors the generic centroid-placement shapes used by r11l:
            # 2-piece pairs, 3-piece lines/triangles, and 4-piece boxes/lines.
            for delta in range(5, 9):
                add_top_left((anchor_x, anchor_y - delta))
                add_top_left((anchor_x, anchor_y + delta))
                add_top_left((anchor_x - delta, anchor_y))
                add_top_left((anchor_x + delta, anchor_y))
            for delta in range(4, 7):
                add_top_left((anchor_x - delta, anchor_y - delta))
                add_top_left((anchor_x + delta, anchor_y + delta))
                add_top_left((anchor_x - delta, anchor_y + delta))
                add_top_left((anchor_x + delta, anchor_y - delta))
                add_top_left((anchor_x, anchor_y - delta))
                add_top_left((anchor_x, anchor_y))
                add_top_left((anchor_x, anchor_y + delta))
                add_top_left((anchor_x - delta, anchor_y))
                add_top_left((anchor_x + delta, anchor_y))
                add_top_left((anchor_x - delta, anchor_y + delta))
                add_top_left((anchor_x + delta, anchor_y))
            for delta in range(3, 8):
                add_top_left((anchor_x - delta, anchor_y - delta))
                add_top_left((anchor_x + delta, anchor_y - delta))
                add_top_left((anchor_x - delta, anchor_y + delta))
                add_top_left((anchor_x + delta, anchor_y + delta))
            for dx in range(3, 6):
                for dy in range(3, 6):
                    add_top_left((anchor_x - dx, anchor_y - dy))
                    add_top_left((anchor_x + dx, anchor_y - dy))
                    add_top_left((anchor_x - dx, anchor_y + dy))
                    add_top_left((anchor_x + dx, anchor_y + dy))
            for delta in range(3, 6):
                add_top_left((anchor_x, anchor_y - 3 * delta))
                add_top_left((anchor_x, anchor_y - delta))
                add_top_left((anchor_x, anchor_y + delta))
                add_top_left((anchor_x, anchor_y + 3 * delta))
                add_top_left((anchor_x - 3 * delta, anchor_y))
                add_top_left((anchor_x - delta, anchor_y))
                add_top_left((anchor_x + delta, anchor_y))
                add_top_left((anchor_x + 3 * delta, anchor_y))

        def _add_score_progress_template_candidates() -> None:
            if not _score_progress_template_candidates_enabled():
                return
            score_now = int(self._last_score)
            target_rows = sorted(
                [t for t in list(perceived.get("targets") or []) if isinstance(t, dict)],
                key=lambda item: tuple(item.get("pos") or (99, 99)),
            )
            inactive_rows = sorted(
                [p for p in list(perceived.get("inactive_pieces") or []) if isinstance(p, dict)],
                key=lambda item: tuple(item.get("pos") or (99, 99)),
            )
            for template_index, template in enumerate(R11L_SCORE_PROGRESS_TEMPLATE_CANDIDATES):
                template_score = int(template.get("score", -999) or -999)
                if template_score != score_now:
                    continue
                template_kind = str(template.get("kind", "") or "")
                template_color = template.get("color")
                if template_kind == "inactive_piece":
                    for piece in inactive_rows:
                        if template_color not in (None, "") and int(piece.get("color", -999) or -999) != int(template_color):
                            continue
                        pos = piece.get("pos")
                        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                            continue
                        try:
                            x = int(pos[0]) + 2
                            y = int(pos[1]) + 2
                        except (TypeError, ValueError):
                            continue
                        _add_xy(
                            "score_progress_template_inactive_piece",
                            x,
                            y,
                            color=piece.get("color"),
                            metadata={
                                "candidate_template_score": int(template_score),
                                "candidate_template_index": int(template_index),
                                "candidate_template_support": int(template.get("support", 0) or 0),
                                "candidate_template_kind": template_kind,
                            },
                        )
                    continue
                if "target_dx" not in template or "target_dy" not in template:
                    continue
                for target in target_rows:
                    if template_color not in (None, ""):
                        target_colors = _colors(target.get("colors", target.get("color")))
                        if int(template_color) not in target_colors:
                            continue
                    pos = target.get("pos")
                    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                        continue
                    try:
                        anchor_x = int(pos[0])
                        anchor_y = int(pos[1])
                        target_dx = int(template.get("target_dx", 0) or 0)
                        target_dy = int(template.get("target_dy", 0) or 0)
                        x = anchor_x + 3 + target_dx
                        y = anchor_y + 3 + target_dy
                    except (TypeError, ValueError):
                        continue
                    _add_xy(
                        "score_progress_template_candidate",
                        x,
                        y,
                        color=template_color,
                        colors=target.get("colors", target.get("color")),
                        metadata={
                            "candidate_origin_anchor_x": int(anchor_x),
                            "candidate_origin_anchor_y": int(anchor_y),
                            "candidate_target_dx": int(target_dx),
                            "candidate_target_dy": int(target_dy),
                            "candidate_jitter_dx": int(template.get("jitter_dx", 0) or 0),
                            "candidate_jitter_dy": int(template.get("jitter_dy", 0) or 0),
                            "candidate_template_score": int(template_score),
                            "candidate_template_index": int(template_index),
                            "candidate_template_support": int(template.get("support", 0) or 0),
                            "candidate_template_kind": template_kind,
                        },
                    )

        def _add_score_progress_selected_target_offset_candidates() -> None:
            if not _score_progress_template_candidates_enabled():
                return
            score_now = int(self._last_score)
            features = dict(self._current_proposal_features or {})
            try:
                center_col = int(features.get("selected_target_center_col", 0) or 0)
                center_row = int(features.get("selected_target_center_row", 0) or 0)
            except (TypeError, ValueError):
                return
            if center_col <= 0 and center_row <= 0:
                return
            for template_index, template in enumerate(
                R11L_SCORE_PROGRESS_SELECTED_TARGET_OFFSET_CANDIDATES
            ):
                try:
                    template_score = int(template.get("score", -999) or -999)
                    dx = int(template.get("dx", 0) or 0)
                    dy = int(template.get("dy", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if template_score != score_now:
                    continue
                _add_xy(
                    "score_progress_selected_target_offset",
                    center_col + dx,
                    center_row + dy,
                    metadata={
                        "candidate_selected_target_center_col": int(center_col),
                        "candidate_selected_target_center_row": int(center_row),
                        "candidate_selected_target_dx": int(dx),
                        "candidate_selected_target_dy": int(dy),
                        "candidate_template_score": int(template_score),
                        "candidate_template_index": int(template_index),
                        "candidate_template_support": int(template.get("support", 0) or 0),
                        "candidate_template_kind": "selected_target_offset",
                    },
                )

        _add_score_progress_template_candidates()
        _add_score_progress_selected_target_offset_candidates()

        _add(
            "active_piece",
            perceived.get("active_piece"),
            offset=2,
            color=perceived.get("active_piece_color"),
        )
        for piece in sorted(
            [p for p in list(perceived.get("inactive_pieces") or []) if isinstance(p, dict)],
            key=lambda item: tuple(item.get("pos") or (99, 99)),
        ):
            _add("inactive_piece", piece.get("pos"), offset=2, color=piece.get("color"))
        for canvas in sorted(
            [c for c in list(perceived.get("canvases") or []) if isinstance(c, dict)],
            key=lambda item: tuple(item.get("pos") or (99, 99)),
        ):
            _add(
                "canvas",
                canvas.get("pos"),
                offset=2,
                color=canvas.get("piece_color", canvas.get("color")),
                colors=canvas.get("colors"),
            )
        for target in sorted(
            [t for t in list(perceived.get("targets") or []) if isinstance(t, dict)],
            key=lambda item: tuple(item.get("pos") or (99, 99)),
        ):
            _add(
                "target",
                target.get("pos"),
                offset=3,
                color=target.get("color"),
                colors=target.get("colors"),
            )
            pos = target.get("pos")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                try:
                    target_x = int(pos[0])
                    target_y = int(pos[1])
                    center_x = target_x + 3
                    center_y = target_y + 3
                except (TypeError, ValueError):
                    continue
                _add_centroid_destination_candidates(
                    target_x,
                    target_y,
                    kind="target_centroid_destination",
                    color=target.get("color"),
                    colors=target.get("colors"),
                )
                # r11l click destinations are not always visible sprites:
                # the solver often clicks inferred piece destination centers
                # around a target/canvas centroid. These offsets are a
                # frame-derived candidate set, not adapter advice.
                for delta in range(3, 9):
                    offsets = (
                        (-delta - 1, -1),
                        (delta - 1, -1),
                        (-1, -delta - 1),
                        (-1, delta - 1),
                        (-delta - 1, -delta - 1),
                        (delta - 1, delta - 1),
                        (-delta - 1, delta - 1),
                        (delta - 1, -delta - 1),
                    )
                    for dx, dy in offsets:
                        _add_xy(
                            "target_destination",
                            center_x + dx,
                            center_y + dy,
                            color=target.get("color"),
                            colors=target.get("colors"),
                        )
                dense_offsets: list[tuple[int, int]] = []
                for dy in range(-8, 9):
                    for dx in range(-8, 9):
                        if abs(dx) <= 2 and abs(dy) <= 2:
                            continue
                        dense_offsets.append((dx, dy))
                dense_offsets.sort(key=lambda item: (max(abs(item[0]), abs(item[1])), abs(item[0]) + abs(item[1]), item))
                for dx, dy in dense_offsets:
                    _add_xy(
                        "target_neighborhood",
                        center_x + dx,
                        center_y + dy,
                        color=target.get("color"),
                        colors=target.get("colors"),
                    )

        signature = str(perceived.get("level_signature", "") or "")
        if signature:
            try:
                from arcgym.adapters.r11l_perception_driven import _ABSORBER_SPECS
            except Exception:
                _ABSORBER_SPECS = {}
            spec = dict(_ABSORBER_SPECS.get(signature, {}) or {})
            for pos in list(spec.get("pieces") or []):
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    try:
                        _add_xy("signature_piece", int(pos[0]) + 2, int(pos[1]) + 2)
                    except (TypeError, ValueError):
                        continue
            for group in list(spec.get("groups") or []):
                if not isinstance(group, dict):
                    continue
                for waypoint in list(group.get("waypoints") or []):
                    if (
                        not isinstance(waypoint, (list, tuple))
                        or len(waypoint) < 2
                        or not isinstance(waypoint[0], (list, tuple))
                        or not isinstance(waypoint[1], (list, tuple))
                    ):
                        continue
                    try:
                        wx, wy = int(waypoint[0][0]), int(waypoint[0][1])
                        ww, wh = int(waypoint[1][0]), int(waypoint[1][1])
                    except (TypeError, ValueError):
                        continue
                    center_x = wx + max(0, ww // 2)
                    center_y = wy + max(0, wh // 2)
                    _add_xy("signature_waypoint", center_x, center_y)
                    _add_centroid_destination_candidates(
                        wx,
                        wy,
                        kind="signature_centroid_destination",
                    )
                    for dy in range(-10, 11):
                        for dx in range(-10, 11):
                            if abs(dx) <= 1 and abs(dy) <= 1:
                                continue
                        _add_xy("signature_waypoint_neighborhood", center_x + dx, center_y + dy)

        selected_rows = _select_candidate_bands(rows)
        for index, row in enumerate(selected_rows):
            row["candidate_id"] = (
                f"{self.game_id_normalized}:{row['kind']}:{index}:{row['x']}:{row['y']}"
            )
        return selected_rows

    def _candidate_memory_allowed(self, candidate: dict[str, Any]) -> bool:
        kind = str(candidate.get("kind", "") or "")
        return kind in {
            "active_piece",
            "inactive_piece",
            "canvas",
            "target",
            "target_destination",
            "target_centroid_destination",
            "target_centroid_destination_jitter",
            "signature_piece",
            "signature_waypoint",
            "signature_centroid_destination",
            "signature_centroid_destination_jitter",
        }

    def _remember_click_candidates(self, candidates: list[dict[str, Any]], *, score: int) -> None:
        if self._visible_click_candidate_memory_score != int(score):
            self._visible_click_candidate_memory = {}
            self._visible_click_candidate_memory_score = int(score)
        for candidate in candidates:
            if not self._candidate_memory_allowed(candidate):
                continue
            try:
                key = (int(candidate.get("x")), int(candidate.get("y")))
            except (TypeError, ValueError):
                continue
            previous = dict(self._visible_click_candidate_memory.get(key) or {})
            seen_count = int(previous.get("memory_seen_count", 0) or 0) + 1
            first_step = int(previous.get("memory_first_step", self.observed_steps) or self.observed_steps)
            row = dict(candidate)
            row["memory_seen_count"] = int(seen_count)
            row["memory_first_step"] = int(first_step)
            row["memory_last_step"] = int(self.observed_steps)
            row["memory_score"] = int(score)
            self._visible_click_candidate_memory[key] = row

    def _with_click_candidate_memory(
        self,
        candidates: list[dict[str, Any]],
        *,
        score: int,
    ) -> list[dict[str, Any]]:
        max_candidates = _visible_click_candidate_max_from_env()
        if max_candidates <= 0:
            return []
        executed = set(self._teacher_assisted_click_history)
        proposed = set(self._proposal_click_history)
        rows: list[dict[str, Any]] = []
        seen_xy: set[tuple[int, int]] = set()

        def _decorate(candidate: dict[str, Any], *, source: str) -> dict[str, Any] | None:
            try:
                x = int(candidate.get("x"))
                y = int(candidate.get("y"))
            except (TypeError, ValueError):
                return None
            row = dict(candidate)
            row["candidate_source"] = str(source)
            row["candidate_previously_executed"] = bool((x, y) in executed)
            row["candidate_previously_proposed"] = bool((x, y) in proposed)
            row.setdefault("memory_seen_count", 0)
            row.setdefault("memory_score", int(score))
            row["memory_age"] = max(0, int(self.observed_steps) - int(row.get("memory_last_step", self.observed_steps) or self.observed_steps))
            return row

        for candidate in candidates:
            decorated = _decorate(candidate, source="current")
            if decorated is None:
                continue
            key = (int(decorated["x"]), int(decorated["y"]))
            if key in seen_xy:
                continue
            seen_xy.add(key)
            rows.append(decorated)

        memory_rows = sorted(
            [
                dict(candidate)
                for candidate in self._visible_click_candidate_memory.values()
                if int(candidate.get("memory_score", score) or score) == int(score)
            ],
            key=lambda row: (
                -int(row.get("memory_last_step", 0) or 0),
                -int(row.get("memory_seen_count", 0) or 0),
                str(row.get("kind", "") or ""),
                int(row.get("x", 0) or 0),
                int(row.get("y", 0) or 0),
            ),
        )
        for candidate in memory_rows:
            decorated = _decorate(candidate, source="memory")
            if decorated is None:
                continue
            key = (int(decorated["x"]), int(decorated["y"]))
            if key in seen_xy:
                continue
            seen_xy.add(key)
            decorated["kind"] = f"memory_{decorated.get('kind', '')}"
            rows.append(decorated)
            if len(rows) >= max_candidates:
                break

        for index, row in enumerate(rows[:max_candidates]):
            row["candidate_id"] = (
                f"{self.game_id_normalized}:{row.get('kind', '')}:{index}:{row['x']}:{row['y']}"
            )
        return rows[:max_candidates]

    def _compute_stuck_steps(self, current_score: int) -> int:
        """Phase 46 — return consecutive frames with no score gain.

        Pushes ``current_score`` into the rolling history then walks
        backwards through the buffer counting the trailing run of
        observations whose score never exceeded the latest. The result
        equals the number of frames in the ``stuck_threshold + 2``
        window in which no positive score progress has happened. When
        the buffer is shorter than the window, returns the run length
        within the available history.
        """

        try:
            score = int(current_score)
        except (TypeError, ValueError):
            score = 0
        self._stuck_score_history.append(score)
        history = list(self._stuck_score_history)
        if not history:
            self._stuck_steps_current = 0
            return 0
        latest = history[-1]
        run = 0
        # Walk backwards from the most recent frame; count how many
        # earlier frames already had ``score >= latest`` (no progress
        # since). Each such frame extends the stuck run by one.
        for prior in reversed(history[:-1]):
            if int(prior) >= int(latest):
                run += 1
            else:
                break
        self._stuck_steps_current = int(run)
        self._stuck_steps_sum += int(run)
        self._stuck_steps_obs_count += 1
        return int(run)

    def _repair_gate_features(
        self,
        *,
        teacher_action: str,
        advice: dict[str, Any] | None,
    ) -> dict[str, Any]:
        features = dict(self._current_proposal_features or {})
        eca_action = _action_name((self._current_proposal or {}).get("action", ""))
        teacher_operator = _operator_from_action(teacher_action)
        control_operator = str(features.get("control_operator", "") or "").strip()
        if not control_operator:
            control_operator = _operator_from_action(eca_action)
            if control_operator:
                features["control_operator"] = control_operator
        if control_operator or teacher_operator:
            features["operator_pair"] = f"{control_operator}->{teacher_operator}"
        if eca_action:
            features.setdefault("eca_action", eca_action)
        if teacher_action:
            features["adapter_action"] = teacher_action
            features["action_pair"] = f"{eca_action}->{teacher_action}"
        features["plan_source"] = str((advice or {}).get("plan_source", "") or "")
        features.setdefault(
            "solver_family",
            str((advice or {}).get("generalisation_tier", "") or ""),
        )
        features.setdefault("advice_used", "false")
        return features

    def _record_repair_gate_agreement(
        self,
        *,
        decision: RepairDecision,
        teacher_action: str,
        teacher_payload: dict[str, Any],
    ) -> None:
        teacher_action_name = _action_name(teacher_action)
        proposal_action_name = _action_name((self._current_proposal or {}).get("action", ""))
        repaired_action_name = (
            _action_name(decision.repaired_action)
            if decision.repair_recommended and decision.repaired_action
            else proposal_action_name
        )
        if teacher_action_name:
            self.repair_gate_action_compare_count += 1
            if proposal_action_name == teacher_action_name:
                self.repair_gate_action_agreement_count += 1
            if repaired_action_name == teacher_action_name:
                self.repair_gate_action_after_repair_agreement_count += 1

        teacher_cmp = _payload_for_compare(teacher_payload)
        if teacher_cmp:
            proposal_cmp = _payload_for_compare(
                dict((self._current_proposal or {}).get("payload") or {})
            )
            repaired_cmp = (
                _payload_for_compare(decision.repaired_payload)
                if decision.repair_recommended and decision.repaired_payload
                else proposal_cmp
            )
            self.repair_gate_payload_compare_count += 1
            if proposal_cmp == teacher_cmp:
                self.repair_gate_payload_agreement_count += 1
            if repaired_cmp == teacher_cmp:
                self.repair_gate_payload_after_repair_agreement_count += 1

    def _append_repair_distillation_sample(
        self,
        *,
        decision: RepairDecision,
        teacher_action: str,
        teacher_payload: dict[str, Any],
        features: dict[str, Any],
    ) -> None:
        if not decision.repair_recommended:
            return
        self.repair_distillation_example_count += 1

        proposal_action = _action_name((self._current_proposal or {}).get("action", ""))
        repaired_action = _action_name(decision.repaired_action) or proposal_action
        teacher_action_name = _action_name(teacher_action)
        teacher_cmp = _payload_for_compare(teacher_payload)
        proposal_cmp = _payload_for_compare(dict((self._current_proposal or {}).get("payload") or {}))
        repaired_cmp = (
            _payload_for_compare(decision.repaired_payload)
            if decision.repaired_payload
            else proposal_cmp
        )
        teacher_point = _click_point_from_payload(teacher_cmp)
        proposal_point = _click_point_from_payload(proposal_cmp)
        visible_candidates = [dict(candidate) for candidate in self._last_visible_click_candidates]
        teacher_match = _rank_visible_click_candidate(visible_candidates, teacher_point)
        proposal_match = _rank_visible_click_candidate(visible_candidates, proposal_point)
        teacher_history = set(self._teacher_assisted_click_history)
        remaining_visible_candidate_count = sum(
            1
            for candidate in visible_candidates
            if (int(candidate.get("x", -999)), int(candidate.get("y", -999))) not in teacher_history
        )
        teacher_previously_executed = bool(teacher_point is not None and teacher_point in teacher_history)
        kept_features = {
            key: _jsonable_value(features.get(key))
            for key in (
                "phase",
                "level_number",
                "score",
                "score_delta",
                "step_index_bucket",
                "queue_remaining_bucket",
                "control_solver_family",
                "solver_family",
                "control_operator",
                "operator_pair",
                "target_translation_status",
                "eca_action",
                "adapter_action",
                "action_pair",
                "plan_source",
                "selected_target_source",
                "selected_target_component_id",
                "selected_target_anchor",
                "selected_target_action_family",
                "selected_target_action_name",
                "selected_target_center_row",
                "selected_target_center_col",
                "selected_target_rank",
                "selected_target_resolver_score_bucket",
                "selected_target_recent_churn_selected_count",
                "selected_target_recent_churn_unique_target_count",
                "selected_target_recent_churn_no_score_change_count",
                "selected_target_recent_churn_score_gain_count",
                "candidate_count",
            )
            if key in features
        }
        kept_features.update(
            {
                "visible_click_candidate_count": int(len(visible_candidates)),
                "remaining_visible_click_candidate_count": int(remaining_visible_candidate_count),
                "tried_click_sites_count": int(len(teacher_history)),
                "proposal_click_history_count": int(len(self._proposal_click_history)),
                "click_candidate_memory_count": int(len(self._visible_click_candidate_memory)),
                "teacher_site_previously_executed": str(bool(teacher_previously_executed)).lower(),
                "teacher_visible_click_candidate_rank": int(teacher_match.get("rank", -1)),
                "teacher_visible_click_candidate_distance": int(teacher_match.get("distance", -1)),
                "teacher_visible_click_candidate_kind": str(teacher_match.get("kind", "") or ""),
                "teacher_visible_click_candidate_band": str(
                    teacher_match.get("candidate_band", "") or ""
                ),
                "proposal_visible_click_candidate_rank": int(proposal_match.get("rank", -1)),
                "proposal_visible_click_candidate_distance": int(proposal_match.get("distance", -1)),
                "proposal_visible_click_candidate_kind": str(proposal_match.get("kind", "") or ""),
                "proposal_visible_click_candidate_band": str(
                    proposal_match.get("candidate_band", "") or ""
                ),
            }
        )
        labels = {
            "action_agrees_before": bool(proposal_action and proposal_action == teacher_action_name),
            "action_agrees_after": bool(repaired_action and repaired_action == teacher_action_name),
            "payload_compared": bool(teacher_cmp),
            "payload_agrees_before": bool(teacher_cmp and proposal_cmp == teacher_cmp),
            "payload_agrees_after": bool(teacher_cmp and repaired_cmp == teacher_cmp),
            "teacher_assisted_executed": False,
            "teacher_site_in_visible_candidates": bool(teacher_match.get("in_candidates")),
            "teacher_site_previously_executed": bool(teacher_previously_executed),
        }
        self._last_repair_distillation_teacher_point = teacher_point
        self._last_repair_distillation_teacher_point_eval_index = int(self.repair_gate_eval_count)
        if proposal_point is not None:
            self._proposal_click_history.append(proposal_point)
        action_pair = str(kept_features.get("action_pair", "") or "")
        if action_pair:
            self.repair_distillation_action_pair_histogram[action_pair] += 1
        target_translation_status = str(
            kept_features.get("target_translation_status", "") or ""
        )
        if target_translation_status:
            self.repair_distillation_target_translation_status_histogram[target_translation_status] += 1
        for key, value in labels.items():
            if value:
                self.repair_distillation_label_histogram[key] += 1
        if len(self.repair_distillation_samples) >= int(self.repair_distillation_max_examples):
            return
        self.repair_distillation_samples.append(
            {
                "schema_version": "eca_repair_distillation_example.v1",
                "example_id": f"{self.game_id_normalized}:{self.repair_gate_eval_count}",
                "repair_gate_eval_index": int(self.repair_gate_eval_count),
                "game_id": self.game_id_normalized,
                "proposal": {
                    "action": proposal_action,
                    "payload": proposal_cmp,
                },
                "teacher": {
                    "action": teacher_action_name,
                    "payload": teacher_cmp,
                },
                "repair": {
                    "action": repaired_action,
                    "payload": repaired_cmp,
                    "reason": str(decision.reason or ""),
                    "objective_mismatch_risk": round(float(decision.objective_mismatch_risk), 6),
                    "payload_mismatch_risk": round(float(decision.payload_mismatch_risk), 6),
                    "action_mismatch_risk": round(float(decision.action_mismatch_risk), 6),
                    "scoreless_mismatch_risk": round(float(decision.scoreless_mismatch_risk), 6),
                    "payload_disagree": bool(decision.payload_disagree),
                },
                "features": kept_features,
                "labels": labels,
                "visible_click_candidates": visible_candidates,
                "teacher_site_candidate_match": teacher_match,
                "proposal_site_candidate_match": proposal_match,
            }
        )

    def _mark_latest_repair_distillation_executed(self) -> None:
        self.repair_distillation_executed_count += 1
        self._last_repair_guard_consumed_eval_index = int(self._last_repair_decision_eval_index)
        self.repair_distillation_label_histogram["teacher_assisted_executed"] += 1
        if (
            self._last_repair_distillation_teacher_point is not None
            and int(self._last_repair_distillation_teacher_point_eval_index)
            == int(self._last_repair_decision_eval_index)
        ):
            self._teacher_assisted_click_history.append(self._last_repair_distillation_teacher_point)
        if self.repair_distillation_samples:
            latest = self.repair_distillation_samples[-1]
            if int(latest.get("repair_gate_eval_index", 0) or 0) == int(
                self._last_repair_decision_eval_index
            ):
                latest.setdefault("labels", {})["teacher_assisted_executed"] = True
                latest["repair_guard"] = {
                    "adopted": True,
                    "reason": "accepted",
                }

    def _start_pending_repair_effect(
        self,
        *,
        action_name: str,
        parameters: dict[str, Any],
    ) -> None:
        if not self.repair_distillation_samples:
            self._pending_repair_effect = {}
            return
        sample_index = len(self.repair_distillation_samples) - 1
        latest = self.repair_distillation_samples[sample_index]
        if int(latest.get("repair_gate_eval_index", 0) or 0) != int(
            self._last_repair_decision_eval_index
        ):
            self._pending_repair_effect = {}
            return
        click_point = _click_point_from_payload(parameters)
        self._pending_repair_effect = {
            "sample_index": int(sample_index),
            "repair_gate_eval_index": int(self._last_repair_decision_eval_index),
            "action": str(action_name or ""),
            "parameters": dict(parameters or {}),
            "click_point": list(click_point) if click_point is not None else [],
            "pre_score": int(self._last_score),
            "pre_observed_step": int(self.observed_steps),
            "pre_frame": np.array(self._last_frame, copy=True)
            if self._last_frame is not None
            else None,
        }
        latest["effect_pending"] = {
            "action": str(action_name or ""),
            "has_pre_frame": bool(self._last_frame is not None),
            "pre_score": int(self._last_score),
        }

    def _resolve_pending_repair_effect(
        self,
        post_frame: np.ndarray,
        *,
        post_score: int,
    ) -> None:
        pending = dict(self._pending_repair_effect or {})
        if not pending:
            return
        self._pending_repair_effect = {}
        sample_index = int(pending.get("sample_index", -1))
        if sample_index < 0 or sample_index >= len(self.repair_distillation_samples):
            return
        sample = self.repair_distillation_samples[sample_index]
        if int(sample.get("repair_gate_eval_index", 0) or 0) != int(
            pending.get("repair_gate_eval_index", -1)
        ):
            return
        click_point_raw = list(pending.get("click_point") or [])
        click_point = (
            (int(click_point_raw[0]), int(click_point_raw[1]))
            if len(click_point_raw) >= 2
            else None
        )
        pre_score = int(pending.get("pre_score", 0) or 0)
        score_delta = int(post_score) - int(pre_score)
        change = _frame_change_summary(
            pending.get("pre_frame"),
            post_frame,
            click_point=click_point,
        )
        visible_change = bool(int(change.get("changed_pixel_count", 0) or 0) > 0)
        score_progress = bool(score_delta > 0)
        outcome_kind = (
            "score_progress"
            if score_progress
            else "visible_change"
            if visible_change
            else "no_visible_change"
        )
        outcome = {
            "schema_version": "eca_repair_action_effect_outcome.v1",
            "resolved": True,
            "outcome_kind": outcome_kind,
            "action": str(pending.get("action", "") or ""),
            "parameters": _jsonable_value(dict(pending.get("parameters") or {})),
            "pre_score": int(pre_score),
            "post_score": int(post_score),
            "score_delta": int(score_delta),
            "pre_observed_step": int(pending.get("pre_observed_step", 0) or 0),
            "post_observed_step": int(self.observed_steps),
            **change,
        }
        sample.pop("effect_pending", None)
        sample["outcome"] = outcome
        labels = sample.setdefault("labels", {})
        labels["repair_effect_resolved"] = True
        labels["repair_effect_visible_change"] = bool(visible_change)
        labels["repair_effect_score_progress"] = bool(score_progress)
        labels["repair_effect_click_near_change"] = bool(change.get("click_near_changed_bbox"))
        self.repair_effect_resolved_count += 1
        self.repair_distillation_label_histogram["repair_effect_resolved"] += 1
        if visible_change:
            self.repair_effect_visible_change_count += 1
            self.repair_distillation_label_histogram["repair_effect_visible_change"] += 1
        if score_progress:
            self.repair_effect_score_progress_count += 1
            self.repair_distillation_label_histogram["repair_effect_score_progress"] += 1

    def _start_pending_repair_prior_effect(
        self,
        *,
        action_name: str,
        parameters: dict[str, Any],
        proposal: dict[str, Any],
    ) -> None:
        click_point = _click_point_from_payload(parameters)
        self._pending_repair_prior_effect = {
            "action": str(action_name or ""),
            "parameters": dict(parameters or {}),
            "proposal": dict(proposal or {}),
            "click_point": list(click_point) if click_point is not None else [],
            "pre_score": int(self._last_score),
            "pre_observed_step": int(self.observed_steps),
            "pre_frame": np.array(self._last_frame, copy=True)
            if self._last_frame is not None
            else None,
        }

    def _resolve_pending_repair_prior_effect(
        self,
        post_frame: np.ndarray,
        *,
        post_score: int,
    ) -> None:
        pending = dict(self._pending_repair_prior_effect or {})
        if not pending:
            return
        self._pending_repair_prior_effect = {}
        click_point_raw = list(pending.get("click_point") or [])
        click_point = (
            (int(click_point_raw[0]), int(click_point_raw[1]))
            if len(click_point_raw) >= 2
            else None
        )
        pre_score = int(pending.get("pre_score", 0) or 0)
        score_delta = int(post_score) - int(pre_score)
        change = _frame_change_summary(
            pending.get("pre_frame"),
            post_frame,
            click_point=click_point,
        )
        visible_change = bool(int(change.get("changed_pixel_count", 0) or 0) > 0)
        score_progress = bool(score_delta > 0)
        outcome_kind = (
            "score_progress"
            if score_progress
            else "visible_change"
            if visible_change
            else "no_visible_change"
        )
        outcome = {
            "schema_version": "eca_repair_prior_action_effect_outcome.v1",
            "resolved": True,
            "outcome_kind": outcome_kind,
            "action": str(pending.get("action", "") or ""),
            "parameters": _jsonable_value(dict(pending.get("parameters") or {})),
            "proposal": _jsonable_value(dict(pending.get("proposal") or {})),
            "pre_score": int(pre_score),
            "post_score": int(post_score),
            "score_delta": int(score_delta),
            "pre_observed_step": int(pending.get("pre_observed_step", 0) or 0),
            "post_observed_step": int(self.observed_steps),
            **change,
        }
        credit_rows: list[dict[str, Any]] = []
        if score_progress:
            window = _repair_prior_effect_credit_window_from_env()
            if window > 0:
                recent = self.repair_prior_effect_history[-window:]
                for history_row in reversed(recent):
                    if bool(history_row.get("delayed_score_progress_credit")):
                        continue
                    if str(history_row.get("outcome_kind", "") or "") != "visible_change":
                        continue
                    if int(history_row.get("score_delta", 0) or 0) != 0:
                        continue
                    history_row["delayed_score_progress_credit"] = True
                    self._record_repair_prior_chain_delayed_credit(history_row)
                    history_proposal = dict(history_row.get("proposal") or {})
                    if bool(history_proposal.get("chain_slot_ledger_support", False)):
                        self.repair_prior_chain_slot_ledger_delayed_credit_count += 1
                        self._credit_repair_prior_chain_slot_ledger(
                            slot_id=str(history_proposal.get("chain_slot_ledger_id", "") or "")
                        )
                    elif (
                        objective_slot_id := str(
                            history_proposal.get("objective_slot_id", "") or ""
                        )
                    ):
                        objective_ledger = self._repair_prior_objective_slot_ledgers.get(
                            objective_slot_id
                        )
                        if isinstance(objective_ledger, dict):
                            objective_ledger["delayed_credit_count"] = int(
                                objective_ledger.get("delayed_credit_count", 0) or 0
                            ) + 1
                            objective_ledger["credited_step"] = int(self.observed_steps)
                            self.repair_prior_objective_slot_credited_count += 1
                    compact = {
                        "action": str(history_row.get("action", "") or ""),
                        "parameters": dict(history_row.get("parameters") or {}),
                        "candidate": dict(
                            history_proposal.get("candidate") or {}
                        ),
                        "chain_slot_ledger_id": str(
                            history_proposal.get("chain_slot_ledger_id", "") or ""
                        ),
                        "chain_slot_ledger_support": bool(
                            history_proposal.get("chain_slot_ledger_support", False)
                        ),
                        "objective_slot_id": str(
                            history_proposal.get("objective_slot_id", "") or ""
                        ),
                        "objective_slot_support": bool(
                            history_proposal.get("objective_slot_support", False)
                        ),
                        "pre_observed_step": int(history_row.get("pre_observed_step", 0) or 0),
                        "post_observed_step": int(history_row.get("post_observed_step", 0) or 0),
                    }
                    credit_rows.append(compact)
            self.repair_prior_effect_chain_credit_count += len(credit_rows)
        if credit_rows:
            outcome["credited_prior_visible_change_count"] = int(len(credit_rows))
            outcome["credited_prior_visible_changes"] = credit_rows[:8]
        pending_proposal = dict(pending.get("proposal") or {})
        if self._repair_prior_proposal_is_objective_slot_exposure(pending_proposal):
            outcome["objective_slot_exposure_candidate"] = True
            if visible_change:
                self.repair_prior_objective_slot_exposed_visible_change_count += 1
            if score_progress:
                self.repair_prior_objective_slot_exposed_score_progress_count += 1
            if not visible_change and not score_progress:
                self.repair_prior_objective_slot_exposed_no_visible_change_count += 1
        probe_candidate = bool(pending_proposal.get("objective_slot_exposure_probe"))
        probe_sandboxed = bool(probe_candidate and visible_change and not score_progress)
        if probe_candidate:
            outcome["objective_slot_exposure_probe_candidate"] = True
            if visible_change:
                self.repair_prior_objective_slot_exposure_probe_visible_change_count += 1
            if score_progress:
                self.repair_prior_objective_slot_exposure_probe_score_progress_count += 1
            if not visible_change and not score_progress:
                self.repair_prior_objective_slot_exposure_probe_no_visible_change_count += 1
            if probe_sandboxed:
                self.repair_prior_objective_slot_exposure_probe_sandboxed_visible_change_count += 1
                outcome["objective_slot_exposure_probe_sandboxed"] = True
        self.last_repair_prior_effect_outcome = outcome
        if visible_change and not probe_sandboxed:
            self._open_repair_prior_chain_continuation(
                proposal=pending_proposal,
                outcome=outcome,
            )
        if bool(pending_proposal.get("chain_slot_ledger_support", False)):
            if score_progress:
                self._close_repair_prior_chain_slot_ledger(
                    status="closed",
                    slot_id=str(
                        pending_proposal.get("chain_slot_ledger_id", "") or ""
                    ),
                )
        if not probe_sandboxed:
            self._record_repair_prior_chain_attempt(
                outcome,
                visible_change=visible_change,
                score_progress=score_progress,
            )
        history_max = _repair_prior_effect_history_max_from_env()
        if history_max > 0:
            self.repair_prior_effect_history.append(dict(outcome))
            if len(self.repair_prior_effect_history) > history_max:
                self.repair_prior_effect_history = self.repair_prior_effect_history[-history_max:]
        self.repair_prior_effect_resolved_count += 1
        if visible_change:
            self.repair_prior_effect_visible_change_count += 1
        if score_progress:
            self.repair_prior_effect_score_progress_count += 1

    def _record_repair_prior_observer_agreement(
        self,
        *,
        features: dict[str, Any],
        teacher_action: str,
        teacher_payload: dict[str, Any],
    ) -> None:
        if _action_name(teacher_action) != "ACTION6":
            return
        teacher_point = _click_point_from_payload(teacher_payload)
        if teacher_point is None:
            return
        prediction = self._rank_repair_prior_candidates(
            features=features,
            teacher_action=teacher_action,
            teacher_payload=teacher_payload,
            example_id=f"{self.game_id_normalized}:observer_prior:{self.repair_gate_eval_count + 1}",
        )
        if not prediction:
            self.last_repair_prior_observer_prediction = {
                "available": False,
                "error": "prior_unavailable",
            }
            return

        ranked = [dict(row) for row in list(prediction.get("ranked") or [])]
        positive_rank = -1
        for index, row in enumerate(ranked, start=1):
            if bool(row.get("label")):
                positive_rank = int(index)
                break
        positive_exists = bool(positive_rank > 0)
        self.repair_prior_observer_eval_count += 1
        self.repair_prior_observer_candidate_count_sum += int(
            prediction.get("candidate_count", len(ranked)) or len(ranked)
        )
        if positive_exists:
            self.repair_prior_observer_teacher_contained_count += 1
            self.repair_prior_observer_mrr_sum += 1.0 / float(positive_rank)
            if positive_rank <= 1:
                self.repair_prior_observer_top1_count += 1
                self.repair_prior_observer_top1_when_contained_count += 1
            if positive_rank <= 3:
                self.repair_prior_observer_top3_count += 1
                self.repair_prior_observer_top3_when_contained_count += 1
            if positive_rank <= 5:
                self.repair_prior_observer_top5_count += 1
                self.repair_prior_observer_top5_when_contained_count += 1
            if positive_rank <= 10:
                self.repair_prior_observer_top10_count += 1
                self.repair_prior_observer_top10_when_contained_count += 1

        top = dict(prediction.get("top") or {})
        top_candidate = dict(top.get("candidate") or {})
        self.last_repair_prior_observer_prediction = {
            "available": True,
            "candidate_count": int(prediction.get("candidate_count", len(ranked)) or len(ranked)),
            "teacher_point": [int(teacher_point[0]), int(teacher_point[1])],
            "teacher_contained": bool(positive_exists),
            "teacher_rank": int(positive_rank),
            "top_score": round(float(top.get("score", 0.0) or 0.0), 6) if top else 0.0,
            "top_label": bool(top.get("label")) if top else False,
            "top_teacher_distance": int(top.get("teacher_distance", -1) or -1) if top else -1,
            "top_candidate": {
                "kind": str(top_candidate.get("kind", "") or ""),
                "band": str(top_candidate.get("candidate_band", "") or ""),
                "source": str(top_candidate.get("candidate_source", "") or ""),
                "x": int(top_candidate.get("x", 0) or 0),
                "y": int(top_candidate.get("y", 0) or 0),
            },
            "top_reasons": list(top.get("prior_reasons") or []) if top else [],
        }

    def _rank_repair_prior_candidates(
        self,
        *,
        features: dict[str, Any] | None = None,
        teacher_action: str = "",
        teacher_payload: dict[str, Any] | None = None,
        example_id: str = "",
    ) -> dict[str, Any]:
        visible_candidates = [dict(candidate) for candidate in self._last_visible_click_candidates]
        visible_candidates.extend(
            self._repair_prior_objective_slot_exposure_candidates(visible_candidates)
        )
        if not visible_candidates:
            return {}
        prior_features = dict(features or {})
        if "score" not in prior_features:
            prior_features["score"] = int(self._last_score)
        teacher = {}
        if _action_name(teacher_action):
            teacher = {
                "action": _action_name(teacher_action),
                "payload": dict(teacher_payload or {}),
            }
        example = {
            "schema_version": "eca_repair_distillation_example.v1",
            "example_id": str(
                example_id
                or f"{self.game_id_normalized}:repair_prior:{self.observed_steps}"
            ),
            "game_id": self.game_id_normalized,
            "features": prior_features,
            "proposal": dict(self._current_proposal or {}),
            "teacher": teacher,
            "visible_click_candidates": visible_candidates,
            "labels": {"teacher_site_in_visible_candidates": False},
        }
        try:
            from arcgym.eca.adapter_mismatch_learner import (
                rank_repair_candidates_by_progress_prior,
            )

            return dict(rank_repair_candidates_by_progress_prior(example))
        except Exception as exc:  # pragma: no cover - diagnostic path
            return {"available": False, "error": str(exc)}

    def _proposal_from_repair_prior_row(
        self,
        row: dict[str, Any],
        *,
        candidate_count: int,
    ) -> dict[str, Any]:
        candidate = dict(row.get("candidate") or {})
        try:
            x = int(candidate.get("x"))
            y = int(candidate.get("y"))
        except (TypeError, ValueError):
            return {
                "available": False,
                "source": "teacher_free_repair_prior",
                "execution_enabled": False,
                "reason": "missing_click_candidate",
            }
        def _row_int(name: str, default: int = 0) -> int:
            value = row.get(name, default)
            if value is None or value == "":
                value = default
            return int(value)

        return {
            "available": True,
            "source": "teacher_free_repair_prior",
            "execution_enabled": False,
            "action": "ACTION6",
            "parameters": {"x": int(x), "y": int(y)},
            "candidate_count": int(candidate_count),
            "candidate_index": int(row.get("candidate_index", 0) or 0),
            "top_score": round(float(row.get("score", 0.0) or 0.0), 6),
            "base_prior_score": round(
                float(row.get("base_prior_score", row.get("score", 0.0)) or 0.0),
                6,
            ),
            "chain_value_support": int(row.get("chain_value_support", 0) or 0),
            "chain_value_seed_support": int(row.get("chain_value_seed_support", 0) or 0),
            "chain_value_context_support": int(row.get("chain_value_context_support", 0) or 0),
            "chain_value_context_family_backoff": bool(
                row.get("chain_value_context_family_backoff", False)
            ),
            "chain_continuation_support": int(row.get("chain_continuation_support", 0) or 0),
            "chain_continuation_ttl": int(row.get("chain_continuation_ttl", 0) or 0),
            "chain_continuation_source_kind": str(
                row.get("chain_continuation_source_kind", "") or ""
            ),
            "chain_slot_binding_support": bool(row.get("chain_slot_binding_support", False)),
            "chain_slot_binding_distance": _row_int("chain_slot_binding_distance", -1),
            "chain_slot_ledger_support": bool(row.get("chain_slot_ledger_support", False)),
            "chain_slot_ledger_id": str(row.get("chain_slot_ledger_id", "") or ""),
            "chain_slot_ledger_status": str(row.get("chain_slot_ledger_status", "") or ""),
            "chain_slot_ledger_distance": _row_int("chain_slot_ledger_distance", -1),
            "chain_slot_candidate_key": list(row.get("chain_slot_candidate_key") or []),
            "objective_slot_support": bool(row.get("objective_slot_support", False)),
            "objective_slot_id": str(row.get("objective_slot_id", "") or ""),
            "objective_slot_rank": int(row.get("objective_slot_rank", -1) or -1),
            "objective_slot_priority": round(
                float(row.get("objective_slot_priority", 0.0) or 0.0),
                6,
            ),
            "objective_slot_binding_distance": int(
                _row_int("objective_slot_binding_distance", -1)
            ),
            "objective_slot_binding_source": str(
                row.get("objective_slot_binding_source", "") or ""
            ),
            "context_frontier_leased": bool(row.get("context_frontier_leased", False)),
            "context_frontier_lease_ttl": int(row.get("context_frontier_lease_ttl", 0) or 0),
            "context_frontier_lease_injected": bool(
                row.get("context_frontier_lease_injected", False)
            ),
            "chain_setup_value": round(float(row.get("chain_setup_value", 0.0) or 0.0), 6),
            "chain_immediate_value": round(float(row.get("chain_immediate_value", 0.0) or 0.0), 6),
            "chain_value": round(float(row.get("chain_value", 0.0) or 0.0), 6),
            "candidate": {
                "kind": str(candidate.get("kind", "") or ""),
                "band": str(candidate.get("candidate_band", "") or ""),
                "source": str(candidate.get("candidate_source", "") or ""),
                "objective_slot_id_hint": str(
                    candidate.get("objective_slot_id_hint", "") or ""
                ),
                "objective_slot_exposure_rank": int(
                    candidate.get("objective_slot_exposure_rank", -1) or -1
                ),
                "objective_slot_global_rank_hint": int(
                    candidate.get("objective_slot_global_rank_hint", -1) or -1
                ),
                "higher_ranked_objective_slot_covered_count": int(
                    candidate.get("higher_ranked_objective_slot_covered_count", 0) or 0
                ),
                "x": int(x),
                "y": int(y),
                "candidate_index": int(row.get("candidate_index", 0) or 0),
                "candidate_previously_executed": bool(
                    candidate.get("candidate_previously_executed")
                ),
                "candidate_previously_proposed": bool(
                    candidate.get("candidate_previously_proposed")
                ),
            },
            "reasons": list(row.get("prior_reasons") or []),
            "score_context": int(self._last_score),
        }

    def _repair_prior_proposal_is_objective_slot_exposure(
        self,
        proposal: Mapping[str, Any],
    ) -> bool:
        candidate = dict(proposal.get("candidate") or {})
        return (
            str(candidate.get("source", candidate.get("candidate_source", "")) or "")
            == "objective_slot_exposure"
            or str(candidate.get("objective_slot_id_hint", "") or "") != ""
            or str(proposal.get("objective_slot_binding_source", "") or "")
            == "objective_slot_exposure"
        )

    def _record_repair_prior_objective_slot_exposed_reject(
        self,
        reason: str,
        proposal: Mapping[str, Any],
    ) -> None:
        if not self._repair_prior_proposal_is_objective_slot_exposure(proposal):
            return
        reason = str(reason or "unknown")
        self.repair_prior_objective_slot_exposed_reject_reason_histogram[reason] += 1
        if reason in {
            "prior_score_below_threshold",
            "prior_site_already_executed",
            "candidate_previously_seen",
        }:
            self.repair_prior_objective_slot_exposed_stale_reject_count += 1

    def _record_repair_prior_objective_slot_exposure_admission_sample(
        self,
        proposal: Mapping[str, Any],
        decision: Mapping[str, Any],
        *,
        min_score: float,
    ) -> None:
        max_samples = int(self.repair_prior_objective_slot_exposure_admission_max_samples)
        if max_samples <= 0:
            return
        if len(self.repair_prior_objective_slot_exposure_admission_samples) >= max_samples:
            return
        candidate = dict(proposal.get("candidate") or {})
        sample = {
            "step": int(self.observed_steps),
            "mode": str(decision.get("mode", "") or ""),
            "eligible": bool(decision.get("eligible", False)),
            "reason": str(decision.get("reason", "") or ""),
            "score": round(float(decision.get("score", proposal.get("top_score", 0.0)) or 0.0), 6),
            "min_score": round(float(min_score), 6),
            "candidate": {
                "kind": str(candidate.get("kind", "") or ""),
                "band": str(candidate.get("band", "") or ""),
                "source": str(candidate.get("source", "") or ""),
                "x": int(candidate.get("x", 0) or 0),
                "y": int(candidate.get("y", 0) or 0),
            },
            "objective_slot_id": str(
                decision.get("objective_slot_id", proposal.get("objective_slot_id", ""))
                or ""
            ),
            "objective_slot_rank": int(
                proposal.get("objective_slot_rank", -1) or -1
            ),
            "objective_slot_exposure_rank": int(
                candidate.get("objective_slot_exposure_rank", -1) or -1
            ),
            "objective_slot_global_rank_hint": int(
                candidate.get("objective_slot_global_rank_hint", -1) or -1
            ),
            "higher_ranked_objective_slot_covered_count": int(
                candidate.get("higher_ranked_objective_slot_covered_count", 0) or 0
            ),
            "objective_slot_priority": round(
                float(proposal.get("objective_slot_priority", 0.0) or 0.0),
                6,
            ),
            "objective_slot_binding_distance": int(
                proposal.get("objective_slot_binding_distance", -1) or -1
            ),
            "objective_slot_status": str(decision.get("objective_slot_status", "") or ""),
            "objective_slot_delayed_credit_count": int(
                decision.get("objective_slot_delayed_credit_count", 0) or 0
            ),
            "evidence": {
                "chain_value_context_support": int(
                    proposal.get("chain_value_context_support", 0) or 0
                ),
                "chain_value_seed_support": int(
                    proposal.get("chain_value_seed_support", 0) or 0
                ),
                "chain_continuation_support": int(
                    proposal.get("chain_continuation_support", 0) or 0
                ),
                "chain_slot_ledger_support": bool(
                    proposal.get("chain_slot_ledger_support", False)
                ),
                "context_frontier_leased": bool(
                    proposal.get("context_frontier_leased", False)
                ),
                "context_family_backoff": bool(
                    proposal.get("chain_value_context_family_backoff", False)
                ),
                "objective_slot_value_evidence": bool(
                    decision.get("objective_slot_value_evidence", False)
                ),
            },
            "reasons": list(proposal.get("reasons") or []),
        }
        self.repair_prior_objective_slot_exposure_admission_samples.append(sample)

    def _repair_prior_objective_slot_exposure_admission_decision(
        self,
        proposal: Mapping[str, Any],
        *,
        min_score: float,
    ) -> dict[str, Any]:
        if not self._repair_prior_proposal_is_objective_slot_exposure(proposal):
            return {"eligible": False, "reason": "not_exposure_candidate"}
        self.repair_prior_objective_slot_exposure_admission_eval_count += 1
        mode = str(self.repair_prior_objective_slot_exposure_admission_mode or "off")
        if mode == "off":
            reason = "admission_off"
            self.repair_prior_objective_slot_exposure_admission_blocked_count += 1
            self.repair_prior_objective_slot_exposure_admission_reason_histogram[reason] += 1
            decision = {"eligible": False, "reason": reason, "mode": mode}
            self._record_repair_prior_objective_slot_exposure_admission_sample(
                proposal,
                decision,
                min_score=min_score,
            )
            return decision
        reasons: list[str] = []
        score = float(proposal.get("top_score", 0.0) or 0.0)
        if score < float(min_score):
            reasons.append("score_below_threshold")
        if not bool(proposal.get("objective_slot_support", False)):
            reasons.append("no_objective_slot_support")
        candidate = dict(proposal.get("candidate") or {})
        exposure_rank = int(candidate.get("objective_slot_exposure_rank", 9999) or 9999)
        is_top_objective_slot = int(proposal.get("objective_slot_rank", 9999) or 9999) == 1
        is_top_uncovered_objective_slot = bool(exposure_rank == 1)
        if not is_top_objective_slot and not is_top_uncovered_objective_slot:
            reasons.append("not_top_objective_slot")
        objective_slot_id = str(proposal.get("objective_slot_id", "") or "")
        if not objective_slot_id:
            objective_slot_id = str(candidate.get("objective_slot_id_hint", "") or "")
        objective_slot_ledger = (
            self._repair_prior_objective_slot_ledgers.get(objective_slot_id)
            if objective_slot_id
            else None
        )
        objective_slot_status = ""
        objective_slot_delayed_credit_count = 0
        if isinstance(objective_slot_ledger, dict):
            objective_slot_status = str(objective_slot_ledger.get("status", "") or "")
            objective_slot_delayed_credit_count = int(
                objective_slot_ledger.get("delayed_credit_count", 0) or 0
            )
        objective_slot_value_evidence = bool(objective_slot_delayed_credit_count > 0)
        independent_evidence = (
            int(proposal.get("chain_value_context_support", 0) or 0) > 0
            or int(proposal.get("chain_value_seed_support", 0) or 0) > 0
            or int(proposal.get("chain_continuation_support", 0) or 0) > 0
            or bool(proposal.get("chain_slot_ledger_support", False))
            or bool(proposal.get("context_frontier_leased", False))
            or bool(proposal.get("chain_value_context_family_backoff", False))
            or objective_slot_value_evidence
        )
        if not independent_evidence:
            reasons.append("missing_independent_value_evidence")
        eligible = not reasons
        reason = "eligible" if eligible else "|".join(reasons)
        self.repair_prior_objective_slot_exposure_admission_reason_histogram[reason] += 1
        if eligible:
            self.repair_prior_objective_slot_exposure_admission_eligible_count += 1
        else:
            self.repair_prior_objective_slot_exposure_admission_blocked_count += 1
        decision = {
            "eligible": bool(eligible),
            "reason": reason,
            "mode": mode,
            "score": round(float(score), 6),
            "independent_evidence": bool(independent_evidence),
            "objective_slot_id": str(objective_slot_id),
            "objective_slot_status": str(objective_slot_status),
            "objective_slot_delayed_credit_count": int(
                objective_slot_delayed_credit_count
            ),
            "objective_slot_value_evidence": bool(objective_slot_value_evidence),
            "top_uncovered_objective_slot": bool(is_top_uncovered_objective_slot),
        }
        self._record_repair_prior_objective_slot_exposure_admission_sample(
            proposal,
            decision,
            min_score=min_score,
        )
        return decision

    def _record_repair_prior_objective_slot_exposure_probe_sample(
        self,
        proposal: Mapping[str, Any],
        probe: Mapping[str, Any],
        *,
        admission: Mapping[str, Any],
    ) -> None:
        max_samples = int(self.repair_prior_objective_slot_exposure_admission_max_samples)
        if max_samples <= 0:
            return
        if len(self.repair_prior_objective_slot_exposure_probe_samples) >= max_samples:
            return
        candidate = dict(proposal.get("candidate") or {})
        sample = {
            "step": int(self.observed_steps),
            "mode": str(probe.get("mode", "") or ""),
            "eligible": bool(probe.get("eligible", False)),
            "reason": str(probe.get("reason", "") or ""),
            "admission_reason": str(admission.get("reason", "") or ""),
            "score": round(float(probe.get("score", proposal.get("top_score", 0.0)) or 0.0), 6),
            "probe_min_score": round(
                float(self.repair_prior_objective_slot_exposure_probe_min_score),
                6,
            ),
            "candidate": {
                "kind": str(candidate.get("kind", "") or ""),
                "band": str(candidate.get("band", "") or ""),
                "source": str(candidate.get("source", "") or ""),
                "x": int(candidate.get("x", 0) or 0),
                "y": int(candidate.get("y", 0) or 0),
            },
            "objective_slot_id": str(
                probe.get("objective_slot_id", proposal.get("objective_slot_id", ""))
                or ""
            ),
            "objective_slot_rank": int(proposal.get("objective_slot_rank", -1) or -1),
            "objective_slot_exposure_rank": int(
                candidate.get("objective_slot_exposure_rank", -1) or -1
            ),
            "objective_slot_global_rank_hint": int(
                candidate.get("objective_slot_global_rank_hint", -1) or -1
            ),
            "higher_ranked_objective_slot_covered_count": int(
                candidate.get("higher_ranked_objective_slot_covered_count", 0) or 0
            ),
            "objective_slot_priority": round(
                float(proposal.get("objective_slot_priority", 0.0) or 0.0),
                6,
            ),
            "evidence": {
                "chain_value_context_support": int(
                    proposal.get("chain_value_context_support", 0) or 0
                ),
                "chain_value_seed_support": int(
                    proposal.get("chain_value_seed_support", 0) or 0
                ),
                "chain_continuation_support": int(
                    proposal.get("chain_continuation_support", 0) or 0
                ),
                "chain_slot_ledger_support": bool(
                    proposal.get("chain_slot_ledger_support", False)
                ),
                "context_frontier_leased": bool(
                    proposal.get("context_frontier_leased", False)
                ),
                "context_family_backoff": bool(
                    proposal.get("chain_value_context_family_backoff", False)
                ),
                "objective_slot_value_evidence": bool(
                    admission.get("objective_slot_value_evidence", False)
                ),
            },
        }
        self.repair_prior_objective_slot_exposure_probe_samples.append(sample)

    def _repair_prior_objective_slot_exposure_probe_decision(
        self,
        proposal: Mapping[str, Any],
        *,
        admission: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not self._repair_prior_proposal_is_objective_slot_exposure(proposal):
            return {"eligible": False, "reason": "not_exposure_candidate"}
        self.repair_prior_objective_slot_exposure_probe_eval_count += 1
        mode = str(self.repair_prior_objective_slot_exposure_probe_mode or "off")
        score = float(proposal.get("top_score", 0.0) or 0.0)
        candidate = dict(proposal.get("candidate") or {})
        reasons: list[str] = []
        if mode == "off":
            reasons.append("probe_off")
        if bool(admission.get("eligible", False)):
            reasons.append("admission_already_eligible")
        if (
            int(self.repair_prior_objective_slot_exposure_probe_admitted_count)
            >= int(self.repair_prior_objective_slot_exposure_probe_max_per_episode)
        ):
            reasons.append("probe_budget_exhausted")
        if score < float(self.repair_prior_objective_slot_exposure_probe_min_score):
            reasons.append("probe_score_below_floor")
        if int(candidate.get("objective_slot_exposure_rank", 9999) or 9999) != 1:
            reasons.append("not_top_uncovered_objective_slot")
        admission_reason = str(admission.get("reason", "") or "")
        if "missing_independent_value_evidence" not in admission_reason:
            reasons.append("not_missing_evidence")
        if "no_objective_slot_support" in admission_reason:
            reasons.append("no_objective_slot_support")
        if "not_top_objective_slot" in admission_reason:
            reasons.append("not_top_objective_slot")
        if bool(candidate.get("candidate_previously_executed")) or bool(
            candidate.get("candidate_previously_proposed")
        ):
            reasons.append("candidate_previously_seen")
        try:
            x = int(candidate.get("x"))
            y = int(candidate.get("y"))
            if (x, y) in self._repair_prior_guard_executed_sites:
                reasons.append("probe_site_already_executed")
        except (TypeError, ValueError):
            reasons.append("probe_invalid_xy")
        eligible = not reasons
        reason = "eligible" if eligible else "|".join(reasons)
        self.repair_prior_objective_slot_exposure_probe_reason_histogram[reason] += 1
        if eligible:
            self.repair_prior_objective_slot_exposure_probe_eligible_count += 1
        else:
            self.repair_prior_objective_slot_exposure_probe_blocked_count += 1
        decision = {
            "eligible": bool(eligible),
            "reason": reason,
            "mode": mode,
            "score": round(float(score), 6),
            "probe_min_score": round(
                float(self.repair_prior_objective_slot_exposure_probe_min_score),
                6,
            ),
            "objective_slot_id": str(proposal.get("objective_slot_id", "") or ""),
        }
        self._record_repair_prior_objective_slot_exposure_probe_sample(
            proposal,
            decision,
            admission=admission,
        )
        return decision

    def _proposal_from_repair_prior_prediction(
        self,
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        ranked = [dict(row) for row in list(prediction.get("ranked") or [])]
        top = dict(prediction.get("top") or {})
        return self._proposal_from_repair_prior_row(
            top,
            candidate_count=int(prediction.get("candidate_count", len(ranked)) or len(ranked)),
        )

    def _refresh_repair_prior_action_proposal(
        self,
        *,
        features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prediction = self._rank_repair_prior_candidates(
            features=features,
            example_id=f"{self.game_id_normalized}:repair_prior_action:{self.observed_steps}",
        )
        if not prediction or str(prediction.get("error", "") or ""):
            self.last_repair_prior_action_proposal = {
                "available": False,
                "source": "teacher_free_repair_prior",
                "execution_enabled": False,
                "error": str((prediction or {}).get("error", "") or ""),
            }
            self._last_repair_prior_ranked = []
            return dict(self.last_repair_prior_action_proposal)
        self._last_repair_prior_ranked = [
            dict(row) for row in list(prediction.get("ranked") or [])
        ]
        self._last_repair_prior_ranked = self._rerank_repair_prior_with_chain_value(
            self._last_repair_prior_ranked
        )
        prediction["ranked"] = list(self._last_repair_prior_ranked)
        prediction["top"] = dict(self._last_repair_prior_ranked[0]) if self._last_repair_prior_ranked else {}
        proposal = self._proposal_from_repair_prior_prediction(prediction)
        self.last_repair_prior_action_proposal = dict(proposal)
        if bool(proposal.get("available")):
            self.repair_prior_action_proposal_count += 1
        return dict(self.last_repair_prior_action_proposal)

    def _repair_prior_chain_key_from_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        score: int | None = None,
    ) -> tuple[str, str, str]:
        return (
            str(int(self._last_score if score is None else score)),
            str(candidate.get("kind", "") or ""),
            str(candidate.get("band", candidate.get("candidate_band", "")) or ""),
        )

    def _repair_prior_chain_context_key_from_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        score: int | None = None,
        prior_score: Any = None,
    ) -> tuple[str, ...] | None:
        try:
            x = int(candidate.get("x"))
            y = int(candidate.get("y"))
        except (TypeError, ValueError):
            return None
        return (
            str(int(self._last_score if score is None else score)),
            str(candidate.get("kind", "") or ""),
            str(candidate.get("band", candidate.get("candidate_band", "")) or ""),
            _score_bucket(0.0 if prior_score is None else prior_score),
            _bucket_int(x, width=8),
            _bucket_int(y, width=8),
        )

    def _repair_prior_chain_index_context_key_from_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        score: int | None = None,
        prior_score: Any = None,
    ) -> tuple[str, ...] | None:
        try:
            candidate_index = int(candidate.get("candidate_index"))
        except (TypeError, ValueError):
            return None
        if candidate_index < 0:
            return None
        return (
            str(int(self._last_score if score is None else score)),
            str(candidate.get("kind", "") or ""),
            str(candidate.get("band", candidate.get("candidate_band", "")) or ""),
            _score_bucket(0.0 if prior_score is None else prior_score),
            _bucket_int(candidate_index, width=4),
        )

    def _repair_prior_chain_keys_from_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        score: int | None = None,
        prior_score: Any = None,
    ) -> list[tuple[str, ...]]:
        keys: list[tuple[str, ...]] = []
        index_context_key = self._repair_prior_chain_index_context_key_from_candidate(
            candidate,
            score=score,
            prior_score=prior_score,
        )
        if index_context_key is not None and all(index_context_key):
            keys.append(index_context_key)
        context_key = self._repair_prior_chain_context_key_from_candidate(
            candidate,
            score=score,
            prior_score=prior_score,
        )
        if context_key is not None and all(context_key):
            keys.append(context_key)
        keys.append(self._repair_prior_chain_key_from_candidate(candidate, score=score))
        return keys

    def _repair_prior_chain_key_from_outcome(
        self,
        outcome: Mapping[str, Any],
    ) -> tuple[str, str, str]:
        proposal = dict(outcome.get("proposal") or {})
        candidate = dict(proposal.get("candidate") or outcome.get("candidate") or {})
        score = _as_int(outcome.get("pre_score"), _as_int(proposal.get("score_context"), self._last_score))
        return self._repair_prior_chain_key_from_candidate(candidate, score=score)

    def _repair_prior_chain_keys_from_outcome(
        self,
        outcome: Mapping[str, Any],
    ) -> list[tuple[str, ...]]:
        proposal = dict(outcome.get("proposal") or {})
        candidate = dict(proposal.get("candidate") or outcome.get("candidate") or {})
        score = _as_int(outcome.get("pre_score"), _as_int(proposal.get("score_context"), self._last_score))
        return self._repair_prior_chain_keys_from_candidate(
            candidate,
            score=score,
            prior_score=proposal.get("top_score", proposal.get("base_prior_score")),
        )

    def _load_repair_prior_chain_value_seed(self) -> None:
        path_raw = str(self.repair_prior_chain_value_seed_path or "").strip()
        if not path_raw:
            return
        try:
            path = Path(path_raw)
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.repair_prior_chain_value_seed_error = str(exc)
            return
        try:
            self.repair_prior_chain_value_seed_rule_count = (
                self._apply_repair_prior_chain_value_seed(data)
            )
        except Exception as exc:  # pragma: no cover - defensive runtime path
            self.repair_prior_chain_value_seed_rule_count = 0
            self.repair_prior_chain_value_seed_error = str(exc)

    def _apply_repair_prior_chain_value_seed(self, data: Mapping[str, Any]) -> int:
        min_support = _repair_prior_chain_value_seed_min_support_from_env()
        max_rules = _repair_prior_chain_value_seed_max_rules_from_env()
        if max_rules <= 0:
            return 0
        applied = 0
        seen_keys: set[tuple[str, ...]] = set()

        rows = list(data.get("top_chain_value_rules") or [])
        for row in rows:
            if applied >= max_rules:
                break
            if not isinstance(row, Mapping):
                continue
            feature_values = dict(row.get("feature_values") or {})
            group = str(row.get("feature_group", "") or "")
            if not {"score", "candidate_kind", "candidate_band"}.issubset(feature_values):
                continue
            if group and not (
                group.endswith("score|candidate_kind|candidate_band")
                or group.endswith(
                    "score|candidate_kind|candidate_band|prior_score_bucket|x_bucket|y_bucket"
                )
                or group.endswith(
                    "score|candidate_kind|candidate_band|prior_score_bucket|candidate_index_bucket"
                )
            ):
                continue
            row_game = str(feature_values.get("game_id", "") or "").strip().lower()
            if row_game and row_game != self.game_id_normalized:
                continue
            support = _as_int(row.get("support"), 0)
            if support < min_support:
                continue
            key: tuple[str, ...]
            if {"prior_score_bucket", "x_bucket", "y_bucket"}.issubset(feature_values):
                key = (
                    str(feature_values.get("score", "") or ""),
                    str(feature_values.get("candidate_kind", "") or ""),
                    str(feature_values.get("candidate_band", "") or ""),
                    str(feature_values.get("prior_score_bucket", "") or ""),
                    str(feature_values.get("x_bucket", "") or ""),
                    str(feature_values.get("y_bucket", "") or ""),
                )
            elif {"prior_score_bucket", "candidate_index_bucket"}.issubset(feature_values):
                key = (
                    str(feature_values.get("score", "") or ""),
                    str(feature_values.get("candidate_kind", "") or ""),
                    str(feature_values.get("candidate_band", "") or ""),
                    str(feature_values.get("prior_score_bucket", "") or ""),
                    str(feature_values.get("candidate_index_bucket", "") or ""),
                )
            else:
                key = (
                    str(feature_values.get("score", "") or ""),
                    str(feature_values.get("candidate_kind", "") or ""),
                    str(feature_values.get("candidate_band", "") or ""),
                )
            if not self._repair_prior_chain_seed_key_allowed(key):
                continue
            if not all(key) or key in seen_keys:
                continue
            self._seed_repair_prior_chain_counter(key, row, support=support)
            seen_keys.add(key)
            applied += 1

        rows = list(data.get("repair_prior_chain_value_top_rules") or [])
        for row in rows:
            if applied >= max_rules:
                break
            if not isinstance(row, Mapping):
                continue
            key_parts = list(row.get("key") or [])
            if len(key_parts) not in {3, 5, 6}:
                continue
            support = _as_int(row.get("attempt"), 0)
            if support < min_support:
                continue
            key = tuple(str(part) for part in key_parts)
            if not self._repair_prior_chain_seed_key_allowed(key):
                continue
            if not all(key) or key in seen_keys:
                continue
            self._seed_repair_prior_chain_counter(key, row, support=support)
            seen_keys.add(key)
            applied += 1
        return applied

    def _repair_prior_chain_seed_key_allowed(self, key: tuple[str, ...]) -> bool:
        mode = str(self.repair_prior_chain_value_seed_key_mode or "all")
        is_context = len(key) > 3
        if mode == "context_only":
            return is_context
        if mode == "coarse_only":
            return not is_context
        return True

    def _seed_repair_prior_chain_counter(
        self,
        key: tuple[str, ...],
        row: Mapping[str, Any],
        *,
        support: int,
    ) -> None:
        counter = self.repair_prior_chain_value_counts.setdefault(key, Counter())
        counter["attempt"] += int(support)
        counter["visible_change"] += _as_int(
            row.get("visible_change_count", row.get("visible_change")),
            0,
        )
        counter["score_progress"] += _as_int(
            row.get("score_progress_count", row.get("score_progress")),
            0,
        )
        counter["delayed_credit"] += _as_int(
            row.get("delayed_credit_count", row.get("delayed_credit")),
            0,
        )
        chain_positive = _as_int(
            row.get("chain_positive_count", row.get("chain_positive")),
            0,
        )
        if chain_positive <= 0:
            chain_positive = (
                _as_int(row.get("score_progress_count", row.get("score_progress")), 0)
                + _as_int(row.get("delayed_credit_count", row.get("delayed_credit")), 0)
            )
        counter["chain_positive"] += chain_positive
        counter["seed"] += int(support)

    def _record_repair_prior_chain_attempt(
        self,
        outcome: Mapping[str, Any],
        *,
        visible_change: bool,
        score_progress: bool,
    ) -> None:
        if not self.repair_prior_chain_value_runtime_feedback_enabled:
            return
        for key in self._repair_prior_chain_keys_from_outcome(outcome):
            counter = self.repair_prior_chain_value_counts.setdefault(key, Counter())
            counter["attempt"] += 1
            if visible_change:
                counter["visible_change"] += 1
            if score_progress:
                counter["score_progress"] += 1
                counter["chain_positive"] += 1

    def _record_repair_prior_chain_delayed_credit(
        self,
        outcome: Mapping[str, Any],
    ) -> None:
        if not self.repair_prior_chain_value_runtime_feedback_enabled:
            return
        for key in self._repair_prior_chain_keys_from_outcome(outcome):
            counter = self.repair_prior_chain_value_counts.setdefault(key, Counter())
            counter["delayed_credit"] += 1
            counter["chain_positive"] += 1

    def _repair_prior_chain_value_for_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        prior_score: Any = None,
    ) -> dict[str, Any]:
        keys = self._repair_prior_chain_keys_from_candidate(
            candidate,
            prior_score=prior_score,
        )
        key = keys[0]
        counter = None
        matched_key = key
        context_support_override: int | None = None
        mode = str(self.repair_prior_chain_value_seed_key_mode or "all")
        candidate_keys = list(keys)
        if mode == "context_backed":
            context_keys = [
                candidate_key for candidate_key in candidate_keys if len(candidate_key) > 3
            ]
            coarse_keys = [
                candidate_key for candidate_key in candidate_keys if len(candidate_key) == 3
            ]
            for candidate_key in context_keys:
                found = self.repair_prior_chain_value_counts.get(candidate_key)
                if found:
                    counter = Counter(found)
                    matched_key = candidate_key
                    context_support_override = int(counter.get("attempt", 0) or 0)
                    break
            if counter is None:
                if (
                    self.repair_prior_context_family_backoff_enabled
                    and coarse_keys
                    and tuple(coarse_keys[0])
                    == tuple(self._repair_prior_context_family_lease.get("key") or ())
                ):
                    coarse = self.repair_prior_chain_value_counts.get(coarse_keys[0])
                    if coarse:
                        counter = Counter(coarse)
                        matched_key = coarse_keys[0]
                        context_support_override = 0
                if counter is None:
                    return {
                        "support": 0,
                        "seed_support": 0,
                        "context_support": 0,
                        "setup_value": 0.0,
                        "immediate_value": 0.0,
                        "chain_value": 0.0,
                        "key": list(key),
                    }
                family_backoff = True
            else:
                family_backoff = False
            if counter is None:
                return {
                    "support": 0,
                    "seed_support": 0,
                    "context_support": 0,
                    "setup_value": 0.0,
                    "immediate_value": 0.0,
                    "chain_value": 0.0,
                    "key": list(key),
                }
            for coarse_key in coarse_keys:
                coarse = self.repair_prior_chain_value_counts.get(coarse_key)
                if not coarse:
                    continue
                for name, value in coarse.items():
                    counter[name] += int(value or 0)
                break
        else:
            family_backoff = False
            for candidate_key in candidate_keys:
                found = self.repair_prior_chain_value_counts.get(candidate_key)
                if found:
                    counter = found
                    matched_key = candidate_key
                    break
        if not counter:
            return {
                "support": 0,
                "seed_support": 0,
                "context_support": 0,
                "setup_value": 0.0,
                "immediate_value": 0.0,
                "chain_value": 0.0,
                "key": list(key),
            }
        attempts = int(counter.get("attempt", 0) or 0)
        if attempts <= 0:
            return {
                "support": 0,
                "seed_support": 0,
                "context_support": 0,
                "setup_value": 0.0,
                "immediate_value": 0.0,
                "chain_value": 0.0,
                "key": list(matched_key),
            }
        setup = (float(counter.get("delayed_credit", 0) or 0) + 0.5) / (float(attempts) + 1.0)
        immediate = (float(counter.get("score_progress", 0) or 0) + 0.5) / (float(attempts) + 1.0)
        chain = (float(counter.get("chain_positive", 0) or 0) + 0.5) / (float(attempts) + 1.0)
        return {
            "support": attempts,
            "seed_support": int(counter.get("seed", 0) or 0),
            "context_support": (
                int(context_support_override)
                if context_support_override is not None
                else attempts if len(matched_key) > 3 else 0
            ),
            "setup_value": round(setup, 6),
            "immediate_value": round(immediate, 6),
            "chain_value": round(chain, 6),
            "key": list(matched_key),
            "context_family_backoff": bool(family_backoff),
        }

    def _repair_prior_candidate_consumed(self, candidate: Mapping[str, Any]) -> bool:
        if bool(candidate.get("candidate_previously_executed")) or bool(
            candidate.get("candidate_previously_proposed")
        ):
            return True
        try:
            x = int(candidate.get("x"))
            y = int(candidate.get("y"))
        except (TypeError, ValueError):
            return False
        return (x, y) in self._repair_prior_guard_executed_sites

    def _repair_prior_candidate_lease_key(
        self,
        candidate: Mapping[str, Any],
    ) -> tuple[int, int, str, str] | None:
        try:
            x = int(candidate.get("x"))
            y = int(candidate.get("y"))
        except (TypeError, ValueError):
            return None
        return (
            int(x),
            int(y),
            str(candidate.get("kind", "") or ""),
            str(candidate.get("band", candidate.get("candidate_band", "")) or ""),
        )

    def _repair_prior_chain_slot_id(self, bbox: Mapping[str, Any]) -> str:
        def _value(name: str) -> int:
            try:
                return int(bbox.get(name))
            except (TypeError, ValueError):
                return -1

        return (
            f"bbox:{_value('min_x')}:"
            f"{_value('min_y')}:"
            f"{_value('max_x')}:"
            f"{_value('max_y')}"
        )

    def _repair_prior_objective_slot_id(
        self,
        *,
        candidate: Mapping[str, Any],
        bbox: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> str:
        if not self.repair_prior_objective_slot_enabled:
            return ""

        def _int_value(source: Mapping[str, Any], name: str, default: int = -1) -> int:
            try:
                return int(source.get(name))
            except (TypeError, ValueError):
                return int(default)

        source_kind = str(candidate.get("kind", "") or "")
        source_band = str(candidate.get("band", candidate.get("candidate_band", "")) or "")
        try:
            score_context = int(outcome.get("pre_score", self._last_score) or 0)
        except (TypeError, ValueError):
            score_context = int(self._last_score)
        width = max(0, _int_value(bbox, "max_x") - _int_value(bbox, "min_x") + 1)
        height = max(0, _int_value(bbox, "max_y") - _int_value(bbox, "min_y") + 1)
        area = max(0, int(width) * int(height))
        if area <= 1:
            effect_shape = "point"
        elif area <= 9:
            effect_shape = "small"
        elif width >= 2 * max(1, height):
            effect_shape = "wide"
        elif height >= 2 * max(1, width):
            effect_shape = "tall"
        else:
            effect_shape = "block"
        return (
            "objective:"
            f"score:{int(score_context)}:"
            f"source:{source_kind}:"
            f"band:{source_band}:"
            f"effect:{effect_shape}"
        )

    def _open_repair_prior_chain_continuation(
        self,
        *,
        proposal: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> None:
        if not self.repair_prior_chain_continuation_enabled:
            return
        candidate = dict(proposal.get("candidate") or {})
        kind = str(candidate.get("kind", "") or "")
        if kind not in {"active_piece", "inactive_piece"}:
            return
        has_context = (
            int(proposal.get("chain_value_context_support", 0) or 0) > 0
            or bool(proposal.get("chain_value_context_family_backoff", False))
            or bool(proposal.get("context_frontier_leased", False))
        )
        if not has_context:
            return
        changed_bbox = dict(outcome.get("changed_bbox") or {})

        def _bbox_int(name: str) -> int:
            try:
                return int(changed_bbox.get(name))
            except (TypeError, ValueError):
                return -1

        def _candidate_int(name: str) -> int:
            try:
                return int(candidate.get(name))
            except (TypeError, ValueError):
                return -1
        normalized_bbox = (
            {
                "min_x": _bbox_int("min_x"),
                "min_y": _bbox_int("min_y"),
                "max_x": _bbox_int("max_x"),
                "max_y": _bbox_int("max_y"),
            }
            if changed_bbox
            else {}
        )
        slot_id = self._repair_prior_chain_slot_id(normalized_bbox) if normalized_bbox else ""
        objective_slot_id = (
            self._repair_prior_objective_slot_id(
                candidate=candidate,
                bbox=normalized_bbox,
                outcome=outcome,
            )
            if normalized_bbox
            else ""
        )

        self._repair_prior_chain_continuation = {
            "ttl": int(self.repair_prior_chain_continuation_ttl),
            "slot_id": str(slot_id),
            "objective_slot_id": str(objective_slot_id),
            "source_kind": kind,
            "source_candidate": dict(candidate),
            "source_x": _candidate_int("x"),
            "source_y": _candidate_int("y"),
            "changed_bbox": normalized_bbox,
            "opened_step": int(self.observed_steps),
            "outcome_kind": str(outcome.get("outcome_kind", "") or ""),
        }
        if self.repair_prior_chain_slot_ledger_enabled and slot_id:
            prior_ledger = dict(self._repair_prior_chain_slot_ledgers.get(str(slot_id)) or {})
            ledger = {
                "slot_id": str(slot_id),
                "objective_slot_id": str(objective_slot_id),
                "status": "open",
                "ttl": int(self.repair_prior_chain_slot_ledger_ttl),
                "changed_bbox": dict(normalized_bbox),
                "source_kind": str(kind),
                "opened_step": int(prior_ledger.get("opened_step", self.observed_steps) or self.observed_steps),
                "reopened_step": int(self.observed_steps),
                "last_ttl_step": int(self.observed_steps),
                "selected_candidate_keys": [
                    list(item) for item in list(prior_ledger.get("selected_candidate_keys") or [])
                ],
                "consumed_candidate_keys": [
                    list(item) for item in list(prior_ledger.get("consumed_candidate_keys") or [])
                ],
                "delayed_credit_count": int(prior_ledger.get("delayed_credit_count", 0) or 0),
            }
            self._repair_prior_chain_slot_ledger = ledger
            self._repair_prior_chain_slot_ledgers[str(slot_id)] = ledger
            if objective_slot_id:
                self._repair_prior_objective_slot_ledgers[str(objective_slot_id)] = ledger
                self.repair_prior_objective_slot_open_count += 1
            self.repair_prior_chain_slot_ledger_open_count += 1
        self.repair_prior_chain_continuation_open_count += 1

    def _repair_prior_chain_slot_ledger_for_id(
        self,
        slot_id: str,
    ) -> dict[str, Any]:
        slot_id = str(slot_id or "")
        if slot_id:
            ledger = self._repair_prior_chain_slot_ledgers.get(slot_id)
            if isinstance(ledger, dict):
                return ledger
        current = self._repair_prior_chain_slot_ledger
        if not isinstance(current, dict) or not current:
            return {}
        current_slot_id = str(current.get("slot_id", "") or "")
        if not slot_id or current_slot_id == slot_id:
            return current
        return {}

    def _repair_prior_chain_slot_ledger_active_count(self) -> int:
        return sum(
            1
            for ledger in self._repair_prior_chain_slot_ledgers.values()
            if str(ledger.get("status", "") or "") in {"open", "pursuing"}
        )

    def _repair_prior_objective_slot_priority(
        self,
        ledger: Mapping[str, Any],
    ) -> float:
        status = str(ledger.get("status", "") or "")
        if status not in {"open", "pursuing"}:
            return 0.0
        priority = 0.50
        if status == "pursuing":
            priority += 0.16
        if list(ledger.get("selected_candidate_keys") or []):
            priority += 0.12
        if int(ledger.get("delayed_credit_count", 0) or 0) > 0:
            priority += 0.18
        try:
            age = max(0, int(self.observed_steps) - int(ledger.get("opened_step", self.observed_steps) or self.observed_steps))
        except (TypeError, ValueError):
            age = 0
        priority -= min(0.18, 0.01 * float(age))
        return round(max(0.0, min(1.0, priority)), 6)

    def _repair_prior_active_objective_slot_rankings(self) -> list[dict[str, Any]]:
        if not (
            self.repair_prior_objective_slot_enabled
            and self.repair_prior_objective_slot_ranking_enabled
        ):
            return []
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for objective_slot_id, ledger in self._repair_prior_objective_slot_ledgers.items():
            objective_slot_id = str(objective_slot_id or "")
            if not objective_slot_id or objective_slot_id in seen:
                continue
            seen.add(objective_slot_id)
            priority = self._repair_prior_objective_slot_priority(ledger)
            if priority <= 0.0:
                continue
            rows.append(
                {
                    "objective_slot_id": objective_slot_id,
                    "priority": float(priority),
                    "status": str(ledger.get("status", "") or ""),
                    "opened_step": int(ledger.get("opened_step", 0) or 0),
                    "selected_count": len(list(ledger.get("selected_candidate_keys") or [])),
                    "delayed_credit_count": int(ledger.get("delayed_credit_count", 0) or 0),
                }
            )
        rows.sort(
            key=lambda row: (
                -float(row.get("priority", 0.0) or 0.0),
                -int(row.get("delayed_credit_count", 0) or 0),
                -int(row.get("selected_count", 0) or 0),
                int(row.get("opened_step", 0) or 0),
                str(row.get("objective_slot_id", "") or ""),
            )
        )
        return rows

    def _repair_prior_objective_slot_rank_for_id(
        self,
        objective_slot_id: str,
    ) -> tuple[int, float]:
        objective_slot_id = str(objective_slot_id or "")
        if not objective_slot_id:
            return -1, 0.0
        for index, row in enumerate(self._repair_prior_active_objective_slot_rankings()):
            if str(row.get("objective_slot_id", "") or "") == objective_slot_id:
                return index + 1, float(row.get("priority", 0.0) or 0.0)
        return -1, 0.0

    def _repair_prior_objective_slot_active_count(self) -> int:
        return len(self._repair_prior_active_objective_slot_rankings())

    def _project_repair_prior_objective_slot_binding(
        self,
        candidate: Mapping[str, Any],
        ranked_slots: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        try:
            x = int(candidate.get("x"))
            y = int(candidate.get("y"))
        except (TypeError, ValueError):
            return None
        slots = (
            list(ranked_slots)
            if ranked_slots is not None
            else self._repair_prior_active_objective_slot_rankings()
        )
        if len(slots) < 2:
            return None
        hinted_objective_slot_id = str(candidate.get("objective_slot_id_hint", "") or "")
        if hinted_objective_slot_id:
            hinted_ledger = self._repair_prior_objective_slot_ledgers.get(
                hinted_objective_slot_id
            )
            if isinstance(hinted_ledger, dict):
                hinted_distance = self._repair_prior_point_distance_to_bbox(
                    x=x,
                    y=y,
                    bbox=dict(hinted_ledger.get("changed_bbox") or {}),
                )
                hinted_rank, hinted_priority = self._repair_prior_objective_slot_rank_for_id(
                    hinted_objective_slot_id
                )
                if (
                    hinted_distance is not None
                    and int(hinted_distance)
                    <= int(self.repair_prior_objective_slot_candidate_binding_max_distance)
                    and int(hinted_rank) > 0
                ):
                    return {
                        "objective_slot_id": hinted_objective_slot_id,
                        "objective_slot_rank": int(hinted_rank),
                        "objective_slot_priority": float(hinted_priority),
                        "objective_slot_binding_distance": int(hinted_distance),
                    }
        best: tuple[int, int, float, str] | None = None
        for slot in slots:
            objective_slot_id = str(slot.get("objective_slot_id", "") or "")
            ledger = self._repair_prior_objective_slot_ledgers.get(objective_slot_id)
            if not isinstance(ledger, dict):
                continue
            distance = self._repair_prior_point_distance_to_bbox(
                x=x,
                y=y,
                bbox=dict(ledger.get("changed_bbox") or {}),
            )
            if distance is None:
                continue
            try:
                rank = int(
                    self._repair_prior_objective_slot_rank_for_id(objective_slot_id)[0]
                )
                priority = float(slot.get("priority", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if distance > int(self.repair_prior_objective_slot_candidate_binding_max_distance):
                continue
            item = (int(distance), int(rank), -float(priority), objective_slot_id)
            if best is None or item < best:
                best = item
        if best is None:
            return None
        distance, rank, negative_priority, objective_slot_id = best
        return {
            "objective_slot_id": objective_slot_id,
            "objective_slot_rank": int(rank),
            "objective_slot_priority": -float(negative_priority),
            "objective_slot_binding_distance": int(distance),
        }

    def _repair_prior_objective_slot_exposure_candidates(
        self,
        visible_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not (
            self.repair_prior_objective_slot_enabled
            and self.repair_prior_objective_slot_ranking_enabled
            and self.repair_prior_objective_slot_candidate_binding_enabled
            and self.repair_prior_objective_slot_candidate_exposure_enabled
        ):
            return []
        ranked_slots = self._repair_prior_active_objective_slot_rankings()
        if len(ranked_slots) < 2:
            return []
        existing_keys = {
            key
            for candidate in visible_candidates
            if (key := self._repair_prior_candidate_lease_key(candidate)) is not None
        }
        covered_objective_slot_ids = {
            str(projected.get("objective_slot_id", "") or "")
            for candidate in visible_candidates
            if (
                projected := self._project_repair_prior_objective_slot_binding(
                    candidate,
                    ranked_slots,
                )
            )
            is not None
        }
        exposed: list[dict[str, Any]] = []
        exposure_rank = 0
        for slot_index, slot in enumerate(ranked_slots, start=1):
            objective_slot_id = str(slot.get("objective_slot_id", "") or "")
            if objective_slot_id in covered_objective_slot_ids:
                self.repair_prior_objective_slot_candidate_exposure_existing_slot_count += 1
                continue
            ledger = self._repair_prior_objective_slot_ledgers.get(objective_slot_id)
            if not isinstance(ledger, dict):
                continue
            bbox = dict(ledger.get("changed_bbox") or {})
            if not bbox:
                continue

            candidate_kind = str(ledger.get("source_kind", "") or "objective_slot_anchor")
            candidate_band = "objective_slot"
            try:
                x = (int(bbox.get("min_x")) + int(bbox.get("max_x"))) // 2
                y = (int(bbox.get("min_y")) + int(bbox.get("max_y"))) // 2
            except (TypeError, ValueError):
                continue
            exposure_rank += 1
            candidate = {
                "kind": str(candidate_kind),
                "band": str(candidate_band),
                "candidate_band": str(candidate_band),
                "candidate_source": "objective_slot_exposure",
                "source": "objective_slot_exposure",
                "x": int(x),
                "y": int(y),
                "objective_slot_id_hint": str(objective_slot_id),
                "objective_slot_exposure_rank": int(exposure_rank),
                "objective_slot_global_rank_hint": int(slot_index),
                "higher_ranked_objective_slot_covered_count": max(0, int(slot_index) - 1),
                "candidate_previously_executed": False,
                "candidate_previously_proposed": False,
            }
            key = self._repair_prior_candidate_lease_key(candidate)
            if key is not None and key in existing_keys:
                self.repair_prior_objective_slot_candidate_exposure_existing_slot_count += 1
                continue
            if key is not None:
                existing_keys.add(key)
                covered_objective_slot_ids.add(objective_slot_id)
            exposed.append(candidate)
            self.repair_prior_objective_slot_candidate_exposed_count += 1
            self.repair_prior_objective_slot_candidate_exposure_slot_count += 1
        return exposed

    def _repair_prior_objective_slot_binding_for_candidate(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not (
            self.repair_prior_objective_slot_enabled
            and self.repair_prior_objective_slot_ranking_enabled
            and self.repair_prior_objective_slot_candidate_binding_enabled
        ):
            return None
        ranked_slots = self._repair_prior_active_objective_slot_rankings()
        if len(ranked_slots) < 2:
            self.repair_prior_objective_slot_candidate_binding_rejected_count += 1
            return None
        projected = self._project_repair_prior_objective_slot_binding(
            candidate,
            ranked_slots,
        )
        if projected is None:
            self.repair_prior_objective_slot_candidate_binding_rejected_count += 1
            return None
        raw_distance = projected.get("objective_slot_binding_distance", -1)
        if raw_distance is None or raw_distance == "":
            raw_distance = -1
        distance = int(raw_distance)
        objective_slot_id = str(projected.get("objective_slot_id", "") or "")
        self.repair_prior_objective_slot_candidate_bound_count += 1
        if (
            int(self.repair_prior_objective_slot_candidate_binding_best_distance) < 0
            or int(distance) < int(self.repair_prior_objective_slot_candidate_binding_best_distance)
        ):
            self.repair_prior_objective_slot_candidate_binding_best_distance = int(distance)
        return {
            "objective_slot_support": True,
            "objective_slot_id": objective_slot_id,
            "objective_slot_rank": int(projected.get("objective_slot_rank", -1) or -1),
            "objective_slot_priority": round(
                float(projected.get("objective_slot_priority", 0.0) or 0.0),
                6,
            ),
            "objective_slot_binding_distance": int(distance),
            "objective_slot_binding_source": "active_objective_slot",
        }

    def _repair_prior_point_distance_to_bbox(
        self,
        *,
        x: int,
        y: int,
        bbox: Mapping[str, Any],
    ) -> int | None:
        try:
            min_x = int(bbox.get("min_x"))
            min_y = int(bbox.get("min_y"))
            max_x = int(bbox.get("max_x"))
            max_y = int(bbox.get("max_y"))
        except (TypeError, ValueError):
            return None
        if min_x < 0 or min_y < 0 or max_x < min_x or max_y < min_y:
            return None
        dx = 0
        if x < min_x:
            dx = min_x - x
        elif x > max_x:
            dx = x - max_x
        dy = 0
        if y < min_y:
            dy = min_y - y
        elif y > max_y:
            dy = y - max_y
        return max(int(dx), int(dy))

    def _repair_prior_chain_slot_ledger_support(
        self,
        candidate: Mapping[str, Any],
        *,
        slot_distance: int,
    ) -> dict[str, Any]:
        if not self.repair_prior_chain_slot_ledger_enabled:
            return {
                "chain_slot_ledger_support": False,
                "chain_slot_ledger_id": "",
                "chain_slot_ledger_status": "",
            }
        continuation_slot_id = str(
            self._repair_prior_chain_continuation.get("slot_id", "") or ""
        )
        ledger = self._repair_prior_chain_slot_ledger_for_id(continuation_slot_id)
        ledger_status = str(ledger.get("status", "") or "")
        if ledger_status not in {"open", "pursuing"}:
            return {
                "chain_slot_ledger_support": False,
                "chain_slot_ledger_id": "",
                "chain_slot_ledger_status": ledger_status,
            }
        slot_id = str(ledger.get("slot_id", "") or "")
        if not slot_id or slot_id != continuation_slot_id:
            return {
                "chain_slot_ledger_support": False,
                "chain_slot_ledger_id": slot_id,
                "chain_slot_ledger_status": "mismatched",
            }
        candidate_key_raw = self._repair_prior_candidate_lease_key(candidate)
        candidate_key = list(candidate_key_raw) if candidate_key_raw is not None else []
        consumed_keys = [list(item) for item in list(ledger.get("consumed_candidate_keys") or [])]
        if candidate_key and candidate_key in consumed_keys:
            self.repair_prior_chain_slot_ledger_consumed_reject_count += 1
            return {
                "chain_slot_ledger_support": False,
                "chain_slot_ledger_id": slot_id,
                "chain_slot_ledger_status": "consumed_candidate",
                "chain_slot_candidate_key": candidate_key,
            }
        ledger["last_supported_step"] = int(self.observed_steps)
        ledger["ttl"] = max(
            int(ledger.get("ttl", 0) or 0),
            int(self.repair_prior_chain_slot_ledger_ttl),
        )
        objective_slot_id = str(ledger.get("objective_slot_id", "") or "")
        objective_slot_rank, objective_slot_priority = (
            self._repair_prior_objective_slot_rank_for_id(objective_slot_id)
        )
        if objective_slot_id:
            self.repair_prior_objective_slot_supported_count += 1
            if objective_slot_rank > 0:
                self.repair_prior_objective_slot_ranked_count += 1
                if objective_slot_rank == 1:
                    self.repair_prior_objective_slot_top_ranked_count += 1
                self.repair_prior_objective_slot_best_priority = max(
                    float(self.repair_prior_objective_slot_best_priority),
                    float(objective_slot_priority),
                )
        self.repair_prior_chain_slot_ledger_supported_count += 1
        return {
            "chain_slot_ledger_support": True,
            "chain_slot_ledger_id": slot_id,
            "chain_slot_ledger_status": ledger_status,
            "chain_slot_candidate_key": candidate_key,
            "chain_slot_ledger_distance": int(slot_distance),
            "objective_slot_id": objective_slot_id,
            "objective_slot_support": bool(objective_slot_id),
            "objective_slot_rank": int(objective_slot_rank),
            "objective_slot_priority": round(float(objective_slot_priority), 6),
        }

    def _record_repair_prior_chain_slot_ledger_selection(
        self,
        proposal: Mapping[str, Any],
    ) -> None:
        if not self.repair_prior_chain_slot_ledger_enabled:
            return
        if not bool(proposal.get("chain_slot_ledger_support", False)):
            return
        slot_id = str(proposal.get("chain_slot_ledger_id", "") or "")
        ledger = self._repair_prior_chain_slot_ledger_for_id(slot_id)
        if str(ledger.get("status", "") or "") not in {"open", "pursuing"}:
            return
        candidate_key = list(proposal.get("chain_slot_candidate_key") or [])
        selected_keys = [list(item) for item in list(ledger.get("selected_candidate_keys") or [])]
        consumed_keys = [list(item) for item in list(ledger.get("consumed_candidate_keys") or [])]
        if candidate_key and candidate_key not in selected_keys:
            selected_keys.append(candidate_key)
        if candidate_key and candidate_key not in consumed_keys:
            consumed_keys.append(candidate_key)
        ledger["selected_candidate_keys"] = selected_keys
        ledger["consumed_candidate_keys"] = consumed_keys
        ledger["status"] = "pursuing"
        ledger["last_selected_step"] = int(self.observed_steps)
        if str(ledger.get("objective_slot_id", "") or ""):
            self.repair_prior_objective_slot_selected_count += 1

    def _close_repair_prior_chain_slot_ledger(
        self,
        *,
        status: str,
        slot_id: str = "",
    ) -> None:
        ledger = self._repair_prior_chain_slot_ledger_for_id(slot_id)
        if not ledger:
            return
        prior_status = str(ledger.get("status", "") or "")
        closed_statuses = {"closed", "credited", "resolved"}
        if prior_status in closed_statuses:
            return
        if prior_status == "expired" and status == "expired":
            return
        if prior_status == "expired" and status != "expired":
            self.repair_prior_chain_slot_ledger_late_credit_count += 1
        ledger["status"] = str(status)
        ledger["closed_step"] = int(self.observed_steps)
        if status == "closed":
            self.repair_prior_chain_slot_ledger_closed_count += 1
        elif status == "credited":
            self.repair_prior_chain_slot_ledger_closed_count += 1
            self.repair_prior_chain_slot_ledger_credited_count += 1
            if str(ledger.get("objective_slot_id", "") or ""):
                self.repair_prior_objective_slot_credited_count += 1
        elif status == "expired":
            self.repair_prior_chain_slot_ledger_expired_count += 1

    def _credit_repair_prior_chain_slot_ledger(self, *, slot_id: str) -> None:
        ledger = self._repair_prior_chain_slot_ledger_for_id(slot_id)
        if not ledger:
            return
        ledger["delayed_credit_count"] = int(ledger.get("delayed_credit_count", 0) or 0) + 1
        ledger["credited_step"] = int(self.observed_steps)
        self._close_repair_prior_chain_slot_ledger(status="credited", slot_id=slot_id)

    def _tick_repair_prior_chain_slot_ledgers(self) -> None:
        if not self.repair_prior_chain_slot_ledger_enabled:
            return
        current_step = int(self.observed_steps)
        for slot_id, ledger in list(self._repair_prior_chain_slot_ledgers.items()):
            if str(ledger.get("status", "") or "") not in {"open", "pursuing"}:
                continue
            if int(ledger.get("last_ttl_step", -1) or -1) == current_step:
                continue
            ledger["last_ttl_step"] = current_step
            ttl = int(ledger.get("ttl", self.repair_prior_chain_slot_ledger_ttl) or 0) - 1
            ledger["ttl"] = ttl
            if ttl <= 0:
                self._close_repair_prior_chain_slot_ledger(
                    status="expired",
                    slot_id=str(slot_id),
                )

    def _repair_prior_chain_continuation_support_for_candidate(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not self.repair_prior_chain_continuation_enabled:
            return None
        ttl = int(self._repair_prior_chain_continuation.get("ttl", 0) or 0)
        if ttl <= 0:
            return None
        band = str(candidate.get("candidate_band", "") or "")
        if self.repair_prior_chain_continuation_bands and band not in self.repair_prior_chain_continuation_bands:
            return None
        kind = str(candidate.get("kind", "") or "")
        if kind.startswith("memory_"):
            kind = kind[len("memory_") :]
        if kind not in {
            "target",
            "target_destination",
            "target_neighborhood",
            "target_centroid_destination",
            "target_centroid_destination_jitter",
            "score_progress_template_candidate",
            "score_progress_selected_target_offset",
            "signature_centroid_destination",
            "signature_centroid_destination_jitter",
        }:
            return None
        slot_distance: int | None = None
        if self.repair_prior_chain_slot_binding_enabled:
            try:
                x = int(candidate.get("x"))
                y = int(candidate.get("y"))
            except (TypeError, ValueError):
                self.repair_prior_chain_slot_binding_rejected_count += 1
                return None
            slot_distance = self._repair_prior_point_distance_to_bbox(
                x=x,
                y=y,
                bbox=dict(self._repair_prior_chain_continuation.get("changed_bbox") or {}),
            )
            if slot_distance is None:
                self.repair_prior_chain_slot_binding_rejected_count += 1
                return None
            if slot_distance > int(self.repair_prior_chain_slot_binding_max_distance):
                self.repair_prior_chain_slot_binding_rejected_count += 1
                return None
            self.repair_prior_chain_slot_binding_supported_count += 1
            if (
                int(self.repair_prior_chain_slot_binding_best_distance) < 0
                or int(slot_distance) < int(self.repair_prior_chain_slot_binding_best_distance)
            ):
                self.repair_prior_chain_slot_binding_best_distance = int(slot_distance)
        ledger_support = self._repair_prior_chain_slot_ledger_support(
            candidate,
            slot_distance=int(slot_distance) if slot_distance is not None else -1,
        )
        if (
            self.repair_prior_chain_slot_ledger_enabled
            and not bool(ledger_support.get("chain_slot_ledger_support", False))
        ):
            return None
        return {
            "slot_binding_support": bool(self.repair_prior_chain_slot_binding_enabled),
            "slot_binding_distance": int(slot_distance) if slot_distance is not None else -1,
            **ledger_support,
        }

    def _repair_prior_chain_continuation_supports_candidate(
        self,
        candidate: Mapping[str, Any],
    ) -> bool:
        return self._repair_prior_chain_continuation_support_for_candidate(candidate) is not None

    def _tick_repair_prior_chain_continuation(self) -> None:
        if not self._repair_prior_chain_continuation:
            return
        ttl = int(self._repair_prior_chain_continuation.get("ttl", 0) or 0) - 1
        if ttl <= 0:
            self._repair_prior_chain_continuation = {}
        else:
            self._repair_prior_chain_continuation["ttl"] = ttl

    def _repair_prior_context_frontier_snapshot(
        self,
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        snapshot = dict(row)
        snapshot["candidate"] = dict(snapshot.get("candidate") or {})
        snapshot["prior_reasons"] = list(snapshot.get("prior_reasons") or [])
        return snapshot

    def _apply_repair_prior_context_frontier_lease(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        context_rows: list[tuple[int, dict[str, Any], tuple[int, int, str, str]]] = []
        for index, row in enumerate(rows):
            if bool(row.get("chain_value_candidate_consumed", False)):
                continue
            if int(row.get("chain_value_context_support", 0) or 0) <= 0:
                continue
            key = self._repair_prior_candidate_lease_key(dict(row.get("candidate") or {}))
            if key is None:
                continue
            context_rows.append((index, row, key))

        if context_rows:
            self.repair_prior_context_frontier_seen_count += 1
            best_rank = int(context_rows[0][0]) + 1
            if (
                self.repair_prior_context_frontier_best_rank <= 0
                or best_rank < self.repair_prior_context_frontier_best_rank
            ):
                self.repair_prior_context_frontier_best_rank = best_rank
            if int(rows[0].get("chain_value_context_support", 0) or 0) <= 0:
                self.repair_prior_unsupported_top_over_context_count += 1
            family_candidate = dict(context_rows[0][1].get("candidate") or {})
            family_key = self._repair_prior_chain_key_from_candidate(family_candidate)
            if all(family_key):
                self._repair_prior_context_family_lease = {
                    "key": tuple(family_key),
                    "ttl": int(self.repair_prior_context_frontier_lease_ttl),
                }
        elif self._repair_prior_context_family_lease:
            family_ttl = int(self._repair_prior_context_family_lease.get("ttl", 0) or 0) - 1
            if family_ttl <= 0:
                self._repair_prior_context_family_lease = {}
            else:
                self._repair_prior_context_family_lease["ttl"] = family_ttl

        if not self.repair_prior_context_frontier_lease_enabled:
            return

        ttl = int(self._repair_prior_context_frontier_lease.get("ttl", 0) or 0)
        active_key_raw = self._repair_prior_context_frontier_lease.get("key")
        active_key = tuple(active_key_raw) if active_key_raw else None
        found_active = False

        if active_key is not None:
            for _index, row, key in context_rows:
                if tuple(key) == active_key:
                    row["context_frontier_leased"] = True
                    row["context_frontier_lease_ttl"] = max(
                        ttl,
                        int(self.repair_prior_context_frontier_lease_ttl),
                    )
                    found_active = True
                    break

        if active_key is not None and not found_active:
            ttl -= 1
            snapshot = dict(self._repair_prior_context_frontier_lease.get("row") or {})
            snapshot_candidate = dict(snapshot.get("candidate") or {})
            if ttl <= 0 or not snapshot or self._repair_prior_candidate_consumed(snapshot_candidate):
                self._repair_prior_context_frontier_lease = {}
                self.repair_prior_context_frontier_expired_count += 1
            else:
                self._repair_prior_context_frontier_lease["ttl"] = ttl
                injected = self._repair_prior_context_frontier_snapshot(snapshot)
                injected["context_frontier_leased"] = True
                injected["context_frontier_lease_ttl"] = ttl
                injected["context_frontier_lease_injected"] = True
                injected.setdefault("prior_reasons", [])
                injected["prior_reasons"] = list(injected.get("prior_reasons") or []) + [
                    "context_frontier_lease"
                ]
                rows.append(injected)
                self.repair_prior_context_frontier_injected_count += 1
                return

        if found_active:
            return

        max_rank = int(self.repair_prior_context_frontier_lease_max_rank)
        for index, row, key in context_rows:
            if index >= max_rank:
                continue
            ttl_new = int(self.repair_prior_context_frontier_lease_ttl)
            self._repair_prior_context_frontier_lease = {
                "key": tuple(key),
                "ttl": ttl_new,
                "row": self._repair_prior_context_frontier_snapshot(row),
            }
            row["context_frontier_leased"] = True
            row["context_frontier_lease_ttl"] = ttl_new
            self.repair_prior_context_frontier_leased_count += 1
            return

    def _rerank_repair_prior_with_chain_value(
        self,
        ranked: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if ranked:
            self.repair_prior_chain_value_rerank_eval_count += 1
        original_top_key = (
            self._repair_prior_chain_key_from_candidate(
                dict((ranked[0] or {}).get("candidate") or {})
            )
            if ranked
            else ("", "", "")
        )
        adjusted_rows: list[dict[str, Any]] = []
        continuation_active = bool(self._repair_prior_chain_continuation)
        if continuation_active:
            self.repair_prior_chain_continuation_active_count += 1
        active_objective_slot_count = len(self._repair_prior_active_objective_slot_rankings())
        self.repair_prior_objective_slot_active_max_count = max(
            int(self.repair_prior_objective_slot_active_max_count),
            int(active_objective_slot_count),
        )
        if ranked and active_objective_slot_count > 1:
            self.repair_prior_objective_slot_multi_active_eval_count += 1
        for row in ranked:
            out = dict(row)
            candidate = dict(out.get("candidate") or {})
            candidate["candidate_index"] = int(out.get("candidate_index", -1) or -1)
            base_score = float(out.get("score", 0.0) or 0.0)
            value = self._repair_prior_chain_value_for_candidate(
                candidate,
                prior_score=base_score,
            )
            support = int(value.get("support", 0) or 0)
            consumed = self._repair_prior_candidate_consumed(candidate)
            out["chain_value_candidate_consumed"] = bool(consumed)
            if support > 0:
                seed_support = int(value.get("seed_support", 0) or 0)
                context_support = int(value.get("context_support", 0) or 0)
                setup = float(value.get("setup_value", 0.0) or 0.0)
                immediate = float(value.get("immediate_value", 0.0) or 0.0)
                chain = float(value.get("chain_value", 0.0) or 0.0)
                family_backoff = bool(value.get("context_family_backoff", False))
                adjustment = 0.0
                if chain > 0.5 and not consumed:
                    adjustment += 0.12 * max(0.0, setup - 0.5)
                    adjustment += 0.24 * max(0.0, immediate - 0.5)
                    adjustment += 0.08 * max(0.0, chain - 0.5)
                if adjustment > 0.0:
                    self.repair_prior_chain_value_adjusted_row_count += 1
                    if seed_support > 0:
                        self.repair_prior_chain_value_seed_adjusted_row_count += 1
                out["base_prior_score"] = round(base_score, 6)
                out["chain_setup_value"] = round(setup, 6)
                out["chain_immediate_value"] = round(immediate, 6)
                out["chain_value"] = round(chain, 6)
                out["chain_value_support"] = int(support)
                out["chain_value_seed_support"] = int(seed_support)
                out["chain_value_context_support"] = int(context_support)
                out["chain_value_context_family_backoff"] = bool(family_backoff)
                if family_backoff:
                    self.repair_prior_context_frontier_family_backoff_count += 1
                out["score"] = round(max(0.0, min(1.0, base_score + adjustment)), 6)
                out.setdefault("prior_reasons", [])
                out["prior_reasons"] = list(out.get("prior_reasons") or []) + [
                    "runtime_chain_value"
                ]
            continuation_support = (
                None
                if consumed
                else self._repair_prior_chain_continuation_support_for_candidate(candidate)
            )
            if continuation_support is not None:
                out["chain_continuation_support"] = 1
                out["chain_continuation_ttl"] = int(
                    self._repair_prior_chain_continuation.get("ttl", 0) or 0
                )
                out["chain_continuation_source_kind"] = str(
                    self._repair_prior_chain_continuation.get("source_kind", "") or ""
                )
                out["chain_slot_binding_support"] = bool(
                    continuation_support.get("slot_binding_support", False)
                )
                out["chain_slot_binding_distance"] = int(
                    continuation_support.get("slot_binding_distance", -1) or -1
                )
                out["chain_slot_ledger_support"] = bool(
                    continuation_support.get("chain_slot_ledger_support", False)
                )
                out["chain_slot_ledger_id"] = str(
                    continuation_support.get("chain_slot_ledger_id", "") or ""
                )
                out["chain_slot_ledger_status"] = str(
                    continuation_support.get("chain_slot_ledger_status", "") or ""
                )
                out["chain_slot_ledger_distance"] = int(
                    continuation_support.get("chain_slot_ledger_distance", -1) or -1
                )
                out["chain_slot_candidate_key"] = list(
                    continuation_support.get("chain_slot_candidate_key") or []
                )
                out["objective_slot_support"] = bool(
                    continuation_support.get("objective_slot_support", False)
                )
                out["objective_slot_id"] = str(
                    continuation_support.get("objective_slot_id", "") or ""
                )
                out["objective_slot_rank"] = int(
                    continuation_support.get("objective_slot_rank", -1) or -1
                )
                out["objective_slot_priority"] = round(
                    float(continuation_support.get("objective_slot_priority", 0.0) or 0.0),
                    6,
                )
                out.setdefault("prior_reasons", [])
                out["prior_reasons"] = list(out.get("prior_reasons") or []) + [
                    "chain_continuation"
                ]
                if bool(continuation_support.get("slot_binding_support", False)):
                    out["prior_reasons"] = list(out.get("prior_reasons") or []) + [
                        "chain_slot_binding"
                    ]
                if bool(continuation_support.get("chain_slot_ledger_support", False)):
                    out["prior_reasons"] = list(out.get("prior_reasons") or []) + [
                        "chain_slot_ledger"
                    ]
                if bool(continuation_support.get("objective_slot_support", False)):
                    out["prior_reasons"] = list(out.get("prior_reasons") or []) + [
                        "objective_slot"
                    ]
                    if int(continuation_support.get("objective_slot_rank", -1) or -1) == 1:
                        out["prior_reasons"] = list(out.get("prior_reasons") or []) + [
                            "objective_slot_top_rank"
                        ]
                self.repair_prior_chain_continuation_supported_count += 1
            if not bool(out.get("objective_slot_support", False)):
                objective_binding = self._repair_prior_objective_slot_binding_for_candidate(
                    candidate
                )
                if objective_binding is not None:
                    out.update(objective_binding)
                    out.setdefault("prior_reasons", [])
                    out["prior_reasons"] = list(out.get("prior_reasons") or []) + [
                        "objective_slot_binding"
                    ]
                    if int(objective_binding.get("objective_slot_rank", -1) or -1) == 1:
                        out["prior_reasons"] = list(out.get("prior_reasons") or []) + [
                            "objective_slot_top_rank"
                        ]
            adjusted_rows.append(out)
        self._apply_repair_prior_context_frontier_lease(adjusted_rows)
        pre_sort_rows = list(adjusted_rows)
        objective_rows = [
            row for row in pre_sort_rows if bool(row.get("objective_slot_support", False))
        ]
        objective_slot_ids = {
            str(row.get("objective_slot_id", "") or "")
            for row in objective_rows
            if str(row.get("objective_slot_id", "") or "")
        }
        if objective_slot_ids:
            self.repair_prior_objective_slot_candidate_bound_slot_max_count = max(
                int(self.repair_prior_objective_slot_candidate_bound_slot_max_count),
                len(objective_slot_ids),
            )
            for objective_slot_id in sorted(objective_slot_ids):
                self.repair_prior_objective_slot_candidate_bound_slot_histogram[
                    str(objective_slot_id)
                ] += 1
        if len(objective_slot_ids) > 1:
            self.repair_prior_objective_slot_candidate_conflict_count += 1
        pre_sort_top_site_key = (
            self._repair_prior_candidate_lease_key(
                dict((pre_sort_rows[0] or {}).get("candidate") or {})
            )
            if pre_sort_rows
            else None
        )
        adjusted_rows.sort(
            key=lambda row: (
                bool(row.get("chain_value_candidate_consumed", False)),
                not bool(row.get("context_frontier_leased", False)),
                not bool(row.get("chain_continuation_support", False)),
                int(row.get("objective_slot_rank", 9999) or 9999)
                if bool(row.get("objective_slot_support", False))
                else 9999,
                -float(row.get("objective_slot_priority", 0.0) or 0.0),
                -float(row.get("score", 0.0) or 0.0),
                -float(row.get("chain_immediate_value", 0.0) or 0.0),
                -float(row.get("chain_setup_value", 0.0) or 0.0),
                int(row.get("candidate_index", 0) or 0),
            )
        )
        if adjusted_rows:
            top = adjusted_rows[0]
            sorted_top_site_key = self._repair_prior_candidate_lease_key(
                dict(top.get("candidate") or {})
            )
            if (
                objective_rows
                and pre_sort_top_site_key is not None
                and sorted_top_site_key is not None
                and sorted_top_site_key != pre_sort_top_site_key
                and bool(top.get("objective_slot_support", False))
            ):
                self.repair_prior_objective_slot_top_changed_count += 1
                top_score = float(top.get("score", 0.0) or 0.0)
                if any(
                    self._repair_prior_candidate_lease_key(
                        dict(row.get("candidate") or {})
                    )
                    != sorted_top_site_key
                    and float(row.get("score", 0.0) or 0.0) > top_score
                    for row in pre_sort_rows
                ):
                    self.repair_prior_objective_slot_local_score_override_count += 1
            if int(top.get("chain_value_support", 0) or 0) > 0:
                self.repair_prior_chain_value_top_supported_count += 1
            if int(top.get("chain_value_seed_support", 0) or 0) > 0:
                self.repair_prior_chain_value_top_seed_supported_count += 1
            base_top_score = float(top.get("base_prior_score", top.get("score", 0.0)) or 0.0)
            final_top_score = float(top.get("score", 0.0) or 0.0)
            top_lift = max(0.0, final_top_score - base_top_score)
            if top_lift > 0.0:
                self.repair_prior_chain_value_top_score_lift_count += 1
                self.repair_prior_chain_value_top_score_lift_sum += float(top_lift)
                if int(top.get("chain_value_seed_support", 0) or 0) > 0:
                    self.repair_prior_chain_value_top_seed_score_lift_count += 1
                    self.repair_prior_chain_value_top_seed_score_lift_sum += float(top_lift)
                guard_min = self._repair_prior_guard_min_score()
                if base_top_score < guard_min <= final_top_score:
                    self.repair_prior_chain_value_top_guard_threshold_cross_count += 1
                    if int(top.get("chain_value_seed_support", 0) or 0) > 0:
                        self.repair_prior_chain_value_top_seed_guard_threshold_cross_count += 1
            final_top_key = self._repair_prior_chain_key_from_candidate(
                dict(top.get("candidate") or {})
            )
            if original_top_key != ("", "", "") and final_top_key != original_top_key:
                self.repair_prior_chain_value_top_changed_count += 1
        if continuation_active:
            self._tick_repair_prior_chain_continuation()
        return adjusted_rows

    def _repair_prior_chain_value_top_rules(self, *, limit: int = 8) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key, counter in self.repair_prior_chain_value_counts.items():
            attempts = int(counter.get("attempt", 0) or 0)
            if attempts <= 0:
                continue
            rows.append(
                {
                    "key": list(key),
                    "attempt": attempts,
                    "visible_change": int(counter.get("visible_change", 0) or 0),
                    "score_progress": int(counter.get("score_progress", 0) or 0),
                    "delayed_credit": int(counter.get("delayed_credit", 0) or 0),
                    "chain_positive": int(counter.get("chain_positive", 0) or 0),
                    "seed": int(counter.get("seed", 0) or 0),
                    "setup_value": round(
                        (float(counter.get("delayed_credit", 0) or 0) + 0.5)
                        / (float(attempts) + 1.0),
                        6,
                    ),
                    "immediate_value": round(
                        (float(counter.get("score_progress", 0) or 0) + 0.5)
                        / (float(attempts) + 1.0),
                        6,
                    ),
                    "chain_value": round(
                        (float(counter.get("chain_positive", 0) or 0) + 0.5)
                        / (float(attempts) + 1.0),
                        6,
                    ),
                }
            )
        rows.sort(
            key=lambda row: (
                -float(row.get("chain_value", 0.0) or 0.0),
                -float(row.get("immediate_value", 0.0) or 0.0),
                -int(row.get("attempt", 0) or 0),
                str(row.get("key", "")),
            )
        )
        return rows[:limit]

    def _evaluate_repair_gate(
        self,
        *,
        advice: dict[str, Any] | None,
        plan: list[dict[str, Any]] | None,
    ) -> RepairDecision | None:
        if not self.repair_gate_active or self.repair_gate is None:
            return None
        teacher_first = (plan or [])[0] if plan else {}
        teacher_action = (
            (teacher_first or {}).get("action", "")
            or (advice or {}).get("first_action", "")
        )
        teacher_advice_payload = {
            "action": teacher_action,
            "payload": (teacher_first or {}).get("parameters", {}) or {},
            "target_kind": _operator_from_action(teacher_action),
        }
        features = self._repair_gate_features(
            teacher_action=str(teacher_action or ""),
            advice=advice,
        )
        self._record_repair_prior_observer_agreement(
            features=features,
            teacher_action=str(teacher_action or ""),
            teacher_payload=dict(teacher_advice_payload.get("payload") or {}),
        )
        try:
            decision = self.repair_gate.evaluate_repair(
                self._current_proposal or None,
                teacher_advice_payload,
                features=features,
                stuck_steps=int(self._stuck_steps_current),
                game_id=self.game_id_normalized,
                current_score=int(self._last_score),
            )
        except Exception:  # pragma: no cover - defensive
            return None

        self.repair_gate_eval_count += 1
        self._record_repair_gate_agreement(
            decision=decision,
            teacher_action=str(teacher_action or ""),
            teacher_payload=dict(teacher_advice_payload.get("payload") or {}),
        )
        # Phase 45 — record OOD diagnostic per evaluation. Tracked on
        # every evaluation regardless of repair_recommended so the
        # diagnostic surfaces both Q1 over-fire (rule matches, score-
        # uncorrelated) and Q2 under-fire (no rule matches, real
        # disagreement invisible).
        self.repair_gate_ood_eval_count += 1
        self._repair_gate_ood_fraction_sum += float(decision.ood_fraction)
        if float(decision.ood_fraction) >= float(
            getattr(self.repair_gate, "ood_threshold", 0.5)
        ):
            self.repair_gate_ood_high_count += 1
        if bool(decision.ood_backoff_engaged):
            self.repair_gate_ood_backoff_engaged_count += 1
        # Phase 55 — score-aware diagnostics. Always tracked even
        # when score_aware suppression is off, so that off vs on
        # comparisons can be made post-hoc by replaying summaries.
        if int(getattr(decision, "score_baseline", -1)) >= 0:
            self.repair_gate_score_aware_evaluations_with_baseline += 1
        if bool(getattr(decision, "score_aware_engaged", False)):
            self.repair_gate_score_aware_engaged_count += 1
        if bool(getattr(decision, "game_baseline_parity_engaged", False)):
            self.repair_gate_game_baseline_parity_engaged_count += 1
        # Phase 46 — selective gating diagnostics. We always track
        # eligible / blocked counts so the bridge can compare the raw
        # rec rate against the selective-eligible rate offline. Note:
        # ``selective_eligible`` is set on the decision regardless of
        # whether selective_mode is "off"; the recommendation itself
        # is only narrowed when selective_mode is on.
        if decision.selective_eligible:
            self.repair_gate_selective_eligible_count += 1
        elif decision.selective_blocked_reason:
            self.repair_gate_selective_blocked_count += 1
            self.repair_gate_selective_blocked_reason_histogram[
                str(decision.selective_blocked_reason)
            ] += 1
        if decision.repair_recommended:
            self.repair_gate_recommended_count += 1
            if decision.objective_mismatch_risk >= self.repair_gate.objective_threshold:
                self.repair_gate_objective_repaired_count += 1
            if (
                decision.payload_mismatch_risk >= self.repair_gate.payload_threshold
                and bool(decision.repaired_payload)
            ):
                self.repair_gate_payload_repaired_count += 1
            high = max(decision.objective_mismatch_risk, decision.payload_mismatch_risk)
            self._repair_gate_high_risk_sum += float(high)
            self._repair_gate_high_risk_count += 1
            self._append_repair_distillation_sample(
                decision=decision,
                teacher_action=str(teacher_action or ""),
                teacher_payload=dict(teacher_advice_payload.get("payload") or {}),
                features=features,
            )
        self.last_repair_decision = decision.to_dict()
        self._last_repair_decision_eval_index = int(self.repair_gate_eval_count)
        # Phase 48 — queue passive shadow-injection record for K-frame
        # post-hoc score-lift attribution. Only records when gate fires.
        if decision.repair_recommended and self.shadow_injection_active:
            try:
                self._record_shadow_decision(
                    decision=decision,
                    score_at_decision=int(self._last_score),
                )
            except Exception:  # pragma: no cover - diagnostic only
                pass
        return decision

    def _plan_from_module(self, frame: np.ndarray, observation: dict[str, Any] | None) -> list[Any]:
        if self.module is None:
            return []
        if self.game_id_normalized == "wa30":
            score = int((observation or {}).get("score", 0) or 0)
            if score != self._wa30_last_score or self._wa30_fill_zones_cache is None:
                from arcgym.perception.wa30_perception import perceive_wa30

                perceived = perceive_wa30(frame)
                self._wa30_fill_zones_cache = perceived.get("fill_zone_cells")
                self._wa30_last_score = score
            return list(
                self.module.perceive_and_plan(
                    frame,
                    fill_zones_cache=self._wa30_fill_zones_cache,
                )
                or []
            )
        return list(self.module.perceive_and_plan(frame) or [])

    def observe(self, observation: dict[str, Any] | None) -> dict[str, Any]:
        if not self.enabled:
            self.last_plan = []
            self.last_advice = self._empty_advice(error=self.import_error)
            return dict(self.last_advice)

        self.observed_steps += 1
        frame = _frame_from_observation(observation)
        if frame is None:
            self.error_steps += 1
            self.last_plan = []
            self._last_visible_click_candidates = []
            self.last_advice = self._empty_advice(error="missing_frame")
            return dict(self.last_advice)

        score_now = 0
        try:
            score_now = int((observation or {}).get("score", 0) or 0)
        except (TypeError, ValueError):
            score_now = 0
        # Phase 46 — update the stuck-step ledger BEFORE any repair-gate
        # evaluation runs so the gate sees the freshest stuck count.
        self._compute_stuck_steps(score_now)
        # Phase 48 — drain pending shadow decisions whose K-frame
        # horizon has elapsed. Done before this frame's gate eval so
        # any new decision queued this frame doesn't drain itself.
        if self.shadow_injection_active:
            try:
                self._drain_shadow_decisions(score_now=score_now)
            except Exception:  # pragma: no cover - diagnostic only
                pass
        self._resolve_pending_repair_effect(frame, post_score=score_now)
        self._resolve_pending_repair_prior_effect(frame, post_score=score_now)
        self._tick_repair_prior_chain_slot_ledgers()
        self._last_frame = np.array(frame, copy=True)
        # Phase 69: track score events for handoff RESET grace logic.
        if int(score_now) > int(self._last_score):
            self._adapter_handoff_last_score_event_step = int(self.observed_steps)
        self._last_score = int(score_now)
        frame_candidates = self._visible_click_candidates_from_frame(frame)
        self._remember_click_candidates(frame_candidates, score=score_now)
        self._last_visible_click_candidates = self._with_click_candidate_memory(
            frame_candidates,
            score=score_now,
        )
        self._refresh_repair_prior_action_proposal(
            features=dict(self._current_proposal_features or {}),
        )
        try:
            plan = self._plan_from_module(frame, observation)
        except Exception as exc:
            self.error_steps += 1
            self.last_plan = []
            self.last_advice = self._empty_advice(error=str(exc))
            return dict(self.last_advice)

        self.last_plan = [_normalize_action_entry(entry) for entry in plan]
        preview = list(self.last_plan[: int(self.max_preview or 0)])
        first_action = str((preview[0] or {}).get("action", "") if preview else "")
        source = str(getattr(self.module, "LAST_PLAN_SOURCE", "") or "")
        diagnostics = _plan_diagnostics_from_module(self.module)
        available = bool(plan)
        tier = str(diagnostics.get("generalisation_tier", "") or "")
        if not tier:
            tier = _tier_from_source(source, available=available)
        if available:
            self.available_steps += 1
            self.plan_action_total += len(plan)
        if source:
            self.source_histogram[source] += 1
        if tier:
            self.generalisation_tier_histogram[tier] += 1

        self.last_advice = {
            "enabled": True,
            "mode": self.mode,
            "available": available,
            "game_id": self.game_id_normalized,
            "module": self.module_name,
            "plan_source": source,
            "generalisation_tier": tier,
            "binding_id": str(diagnostics.get("binding_id", "") or ""),
            "primitive_name": str(diagnostics.get("primitive_name", "") or ""),
            "adapter_free": bool(diagnostics.get("adapter_free", False)),
            "plan_diagnostics": diagnostics,
            "action_count": len(plan),
            "first_action": first_action,
            "preview": preview,
            "error": "",
        }
        # Phase 47: teacher coherence — measure whether the teacher's
        # frame-N first_action matches the action it predicted as
        # "next" one step earlier. Two predictions are compared:
        # (1) after_eca: assumes ECA's recorded action was taken
        # (2) after_teacher: assumes teacher's own first_action was
        #     taken (teacher self-stability — independent of ECA)
        if available:
            try:
                self._track_teacher_coherence(plan=self.last_plan)
            except Exception:  # pragma: no cover - diagnostic only
                # Coherence is purely advisory; never let it break observe().
                pass
        # Phase 42: advisory-only repair gate evaluation.
        if self.repair_gate_active and available:
            self._evaluate_repair_gate(advice=self.last_advice, plan=self.last_plan)
        return dict(self.last_advice)

    def _reject_guarded_actions(self, reason: str) -> list[dict[str, Any]]:
        self.guard_rejected_count += 1
        self.guard_reject_reason_histogram[reason] += 1
        self.last_guard_decision = {
            "adopted": False,
            "reason": reason,
            "action_count": 0,
        }
        return []

    def guarded_actions(
        self,
        *,
        allowed_actions: list[str] | set[str] | None,
        max_actions: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return a guarded executable prefix of the latest solver plan."""

        self.guard_eval_count += 1
        if self.mode != "guarded":
            return self._reject_guarded_actions("mode_not_guarded")
        if not bool((self.last_advice or {}).get("available")):
            return self._reject_guarded_actions("no_solver_plan")
        if str((self.last_advice or {}).get("error", "") or ""):
            return self._reject_guarded_actions("solver_error")

        tier = str((self.last_advice or {}).get("generalisation_tier", "") or "").lower()
        allowed_tiers = _execution_tiers_from_env()
        if allowed_tiers and tier not in allowed_tiers:
            return self._reject_guarded_actions("tier_not_allowed")

        allowed = {
            str(action or "").strip().upper()
            for action in list(allowed_actions or [])
            if str(action or "").strip()
        }
        if not allowed:
            return self._reject_guarded_actions("missing_allowed_actions")

        limit = max(1, int(max_actions or _max_execution_actions_from_env()))
        selected: list[dict[str, Any]] = []
        for action in list(self.last_plan or [])[:limit]:
            action_name = str(action.get("action", "") or "").strip().upper()
            if not action_name:
                return self._reject_guarded_actions("missing_action_name")
            if action_name not in allowed:
                return self._reject_guarded_actions("action_not_allowed")
            parameters = dict(action.get("parameters", {}) or {})
            if action_name == "ACTION6" and not {"x", "y"}.issubset(parameters.keys()):
                return self._reject_guarded_actions("click_missing_xy")
            selected.append(
                {
                    "action": action_name,
                    "parameters": parameters,
                }
            )

        if not selected:
            return self._reject_guarded_actions("empty_guarded_prefix")

        self.guard_adopted_count += 1
        self.last_guard_decision = {
            "adopted": True,
            "reason": "accepted",
            "action_count": len(selected),
            "first_action": str(selected[0].get("action", "") or ""),
            "plan_source": str((self.last_advice or {}).get("plan_source", "") or ""),
            "generalisation_tier": str((self.last_advice or {}).get("generalisation_tier", "") or ""),
        }
        return selected

    def _reject_repair_guarded_action(self, reason: str) -> dict[str, Any]:
        self.repair_guard_rejected_count += 1
        self.repair_guard_reject_reason_histogram[reason] += 1
        self.last_repair_guard_decision = {
            "adopted": False,
            "reason": reason,
        }
        return {}

    def repair_guarded_action(
        self,
        *,
        allowed_actions: list[str] | set[str] | None,
    ) -> dict[str, Any]:
        """Return one guarded repaired action from the advisory gate.

        This is an explicit experiment path. It only activates when
        ``V2_ECA_TARGET_BINDING_REPAIR_EXECUTE=guarded`` and the game is
        present in ``V2_ECA_TARGET_BINDING_REPAIR_EXECUTE_GAMES``.
        """

        self.repair_guard_eval_count += 1
        if self.mode != "advisory":
            return self._reject_repair_guarded_action("bridge_mode_not_advisory")
        if self.repair_execution_mode_value != "guarded":
            return self._reject_repair_guarded_action("repair_execution_disabled")
        if not self.repair_gate_active:
            return self._reject_repair_guarded_action("repair_gate_inactive")
        if not self.repair_execution_games or self.game_id_normalized not in self.repair_execution_games:
            return self._reject_repair_guarded_action("game_not_allowlisted")

        decision = dict(self.last_repair_decision or {})
        if not bool(decision.get("repair_recommended")):
            return self._reject_repair_guarded_action("repair_not_recommended")
        if self._last_repair_decision_eval_index <= 0:
            return self._reject_repair_guarded_action("missing_live_repair_decision")
        if self._last_repair_decision_eval_index <= self._last_repair_guard_consumed_eval_index:
            return self._reject_repair_guarded_action("repair_decision_already_consumed")
        action_name = _action_name(decision.get("repaired_action"))
        if not action_name:
            return self._reject_repair_guarded_action("missing_repaired_action")

        allowed = {
            str(action or "").strip().upper()
            for action in list(allowed_actions or [])
            if str(action or "").strip()
        }
        if not allowed:
            return self._reject_repair_guarded_action("missing_allowed_actions")
        if action_name not in allowed:
            return self._reject_repair_guarded_action("action_not_allowed")

        parameters = dict(decision.get("repaired_payload") or {})
        if action_name == "ACTION6" and not {"x", "y"}.issubset(parameters.keys()):
            return self._reject_repair_guarded_action("click_missing_xy")

        self.repair_guard_adopted_count += 1
        self.last_repair_guard_decision = {
            "adopted": True,
            "reason": "accepted",
            "action": action_name,
            "has_payload": bool(parameters),
        }
        self._mark_latest_repair_distillation_executed()
        self._start_pending_repair_effect(
            action_name=action_name,
            parameters=parameters,
        )
        return {"action": action_name, "parameters": parameters}

    # ──────────────────────────────────────────────────────────────
    # Phase 58 — Adapter handoff (per-game guarded action override)
    # ──────────────────────────────────────────────────────────────
    def _reject_adapter_handoff(self, reason: str) -> dict[str, Any]:
        self.adapter_handoff_rejected_count += 1
        self.adapter_handoff_reject_reason_histogram[reason] += 1
        self.last_adapter_handoff_decision = {
            "adopted": False,
            "reason": reason,
            "action": "",
        }
        return {}

    def adapter_handoff_action(
        self,
        *,
        allowed_actions: list[str] | set[str] | None,
    ) -> dict[str, Any]:
        """Return the perception adapter's first plan entry as a
        guarded override action when ``game_id`` is in the Phase 58
        handoff whitelist.

        This bypasses the global bridge mode check (works in
        ``advisory`` or ``observer``), since the handoff set is the
        per-game promotion target population identified by Phase 57's
        K=100 causal lift measurement (lp85/sk48/ft09 at 100%).

        Returns ``{"action": <name>, "parameters": {...}}`` when
        accepted; ``{}`` when rejected with reason recorded in the
        histogram.

        Note: this is the FIRST mechanism in the Phase 41-57
        sequence that actually mutates ECA's authoritative action
        path. The advisory contract from Phase 41 explicitly
        permits per-game promotion; the handoff whitelist is the
        opt-in.
        """

        self.adapter_handoff_eval_count += 1
        if not self.adapter_handoff_active:
            return self._reject_adapter_handoff("game_not_in_handoff_set")

        # Phase 68: when reset_on_empty enabled, prefer cache-driven
        # continuation over the early no_adapter_plan reject. The
        # `available` flag is False whenever the adapter returned
        # an empty plan this frame, but our cached plan may still
        # have actions to execute. We check cache state below.
        plan = list(self.last_plan or [])
        last_advice_available = bool((self.last_advice or {}).get("available"))
        last_advice_error = str((self.last_advice or {}).get("error", "") or "")

        if last_advice_error:
            return self._reject_adapter_handoff("adapter_error")

        # When the adapter is unavailable AND we are NOT running
        # with reset-on-empty AND the cache is exhausted, retain
        # the original Phase 58 reject reason for downstream
        # diagnostic continuity.
        cache_has_remaining_now = (
            self._handoff_cached_plan
            and self._handoff_cached_plan_index < len(self._handoff_cached_plan)
        )
        if (
            not last_advice_available
            and not cache_has_remaining_now
            and not self.adapter_handoff_reset_on_empty
        ):
            return self._reject_adapter_handoff("no_adapter_plan")

        # Phase 63 + 65: plan-state tracking with cached-plan
        # continuation across empty-plan frames. Maintain a cached
        # plan and advance through it rather than always taking
        # freshly-regenerated last_plan[0]. Phase 65 extension:
        # when last_plan is empty but score hasn't changed AND the
        # cached plan still has entries, continue using the cached
        # plan. This addresses Pattern B (ka59/lf52/m0r0/re86/tn36/
        # tr87/bp35) where the adapter emits a plan only on some
        # frames; the cached plan can carry through the empty
        # frames without restarting.
        score_now = int(self._last_score)
        cache_has_remaining = (
            self._handoff_cached_plan
            and self._handoff_cached_plan_index < len(self._handoff_cached_plan)
        )

        if score_now != self._handoff_cached_plan_score:
            # Score changed (or first-ever fire) — refresh cache.
            # Requires a non-empty current plan; without one, we
            # have no source to populate the cache from.
            if not plan:
                return self._reject_adapter_handoff("empty_plan_at_refresh")
            self._handoff_cached_plan = list(plan)
            self._handoff_cached_plan_index = 0
            self._handoff_cached_plan_score = score_now
            self.adapter_handoff_plan_refresh_count += 1
        elif cache_has_remaining:
            # Phase 65: even if last_plan is empty this frame, the
            # cached plan still has remaining entries we can use.
            # No-op — continue using cache.
            pass
        elif plan:
            # Cached plan exhausted AND new plan available — refresh.
            self._handoff_cached_plan = list(plan)
            self._handoff_cached_plan_index = 0
            self.adapter_handoff_plan_exhaust_count += 1
        else:
            # Cache exhausted AND no new plan.
            # Phase 68: if reset-on-empty is enabled, emit RESET to
            # let the game return to a canonical state that the
            # adapter can re-perceive. RESET is a meta-action that
            # is not normally listed in ``allowed_actions`` (which
            # comes from the env's available_actions), so we
            # intentionally bypass that check for RESET only.
            if self.adapter_handoff_reset_on_empty:
                # Phase 69: grace period guard. If score advanced
                # within ``reset_grace_steps`` frames, we are likely
                # in a level-transition animation. RESET would roll
                # back the level gain, so block it.
                grace = int(self.adapter_handoff_reset_grace_steps)
                if (
                    grace > 0
                    and self._adapter_handoff_last_score_event_step >= 0
                    and (int(self.observed_steps) - int(self._adapter_handoff_last_score_event_step)) <= grace
                ):
                    self.adapter_handoff_reset_grace_blocked_count += 1
                    return self._reject_adapter_handoff("reset_blocked_post_score_event")
                self.adapter_handoff_reset_emitted_count += 1
                self.adapter_handoff_adopted_count += 1
                self.last_adapter_handoff_decision = {
                    "adopted": True,
                    "reason": "reset_on_empty",
                    "action": "RESET",
                    "has_payload": False,
                }
                return {"action": "RESET", "parameters": {}}
            return self._reject_adapter_handoff("empty_plan")

        if not self._handoff_cached_plan:
            return self._reject_adapter_handoff("cached_plan_empty")

        first = self._handoff_cached_plan[self._handoff_cached_plan_index] or {}
        action_name = str(first.get("action", "") or "").strip().upper()
        if not action_name:
            return self._reject_adapter_handoff("missing_action_name")

        allowed = {
            str(action or "").strip().upper()
            for action in list(allowed_actions or [])
            if str(action or "").strip()
        }
        if not allowed:
            return self._reject_adapter_handoff("missing_allowed_actions")
        if action_name not in allowed:
            return self._reject_adapter_handoff("action_not_allowed")

        parameters = dict(first.get("parameters", {}) or {})
        if action_name == "ACTION6" and not {"x", "y"}.issubset(parameters.keys()):
            return self._reject_adapter_handoff("click_missing_xy")

        self.adapter_handoff_adopted_count += 1
        # Phase 63: advance the cached plan index since this action
        # was adopted. Next frame will return the next plan entry.
        self._handoff_cached_plan_index += 1
        self.adapter_handoff_plan_advance_count += 1
        self.last_adapter_handoff_decision = {
            "adopted": True,
            "reason": "accepted",
            "action": action_name,
            "has_payload": bool(parameters),
            "cached_plan_index": int(self._handoff_cached_plan_index),
            "cached_plan_len": int(len(self._handoff_cached_plan)),
        }
        return {"action": action_name, "parameters": parameters}

    def _reject_repair_prior_guarded_action(self, reason: str) -> dict[str, Any]:
        self.repair_prior_guard_rejected_count += 1
        self.repair_prior_guard_reject_reason_histogram[reason] += 1
        self.last_repair_prior_guard_decision = {
            "adopted": False,
            "reason": reason,
        }
        return {}

    def _repair_prior_seed_stale_reject_repeat_count(
        self,
        *,
        x: int,
        y: int,
        candidate: Mapping[str, Any],
    ) -> int:
        key = (int(x), int(y), str(candidate.get("kind", "") or ""))
        self._repair_prior_seed_stale_reject_counts[key] += 1
        repeat_count = int(self._repair_prior_seed_stale_reject_counts[key])
        self.repair_prior_seed_stale_reject_repeat_max = max(
            int(self.repair_prior_seed_stale_reject_repeat_max),
            repeat_count,
        )
        return repeat_count

    def repair_prior_guarded_action(
        self,
        *,
        allowed_actions: list[str] | set[str] | None,
    ) -> dict[str, Any]:
        """Return one guarded action from the teacher-free repair prior.

        This is an explicit experiment path, separate from teacher-assisted
        repair. It never reads adapter payloads; it only consumes the current
        ``repair_prior_action_proposal`` surface.
        """

        self.repair_prior_guard_eval_count += 1
        if self.mode != "advisory":
            return self._reject_repair_prior_guarded_action("bridge_mode_not_advisory")
        if self.repair_prior_execution_mode_value != "guarded":
            return self._reject_repair_prior_guarded_action("prior_execution_disabled")
        if not self.repair_prior_execution_games or self.game_id_normalized not in self.repair_prior_execution_games:
            return self._reject_repair_prior_guarded_action("game_not_allowlisted")

        allowed = {
            str(action or "").strip().upper()
            for action in list(allowed_actions or [])
            if str(action or "").strip()
        }
        if not allowed:
            return self._reject_repair_prior_guarded_action("missing_allowed_actions")
        ranked = [dict(row) for row in list(self._last_repair_prior_ranked or [])]
        if not ranked:
            proposal = dict(self.last_repair_prior_action_proposal or {})
            if bool(proposal.get("available")):
                ranked = [
                    {
                        "candidate": {
                            **dict(proposal.get("candidate") or {}),
                            "candidate_band": str(
                                dict(proposal.get("candidate") or {}).get("band", "") or ""
                            ),
                        },
                        "score": float(proposal.get("top_score", 0.0) or 0.0),
                        "prior_reasons": list(proposal.get("reasons") or []),
                    }
                ]
        if not ranked:
            return self._reject_repair_prior_guarded_action("prior_proposal_unavailable")

        min_score = self._repair_prior_guard_min_score()
        reject_reason = "prior_no_guard_eligible_candidate"
        proposal: dict[str, Any] = {}
        action_name = ""
        x = y = 0
        candidate: dict[str, Any] = {}
        candidate_count = len(ranked)
        seed_stale_reject_recovered = False
        seed_stale_reject_repeat_count = 0
        objective_slot_exposure_probe_selected = False
        for row in ranked:
            proposal = self._proposal_from_repair_prior_row(
                row,
                candidate_count=candidate_count,
            )
            if not bool(proposal.get("available")):
                reject_reason = str(proposal.get("reason", "") or "prior_proposal_unavailable")
                continue
            action_name = _action_name(proposal.get("action"))
            if action_name != "ACTION6":
                reject_reason = "prior_action_not_click"
                continue
            if action_name not in allowed:
                return self._reject_repair_prior_guarded_action("action_not_allowed")
            parameters = dict(proposal.get("parameters") or {})
            if not {"x", "y"}.issubset(parameters.keys()):
                reject_reason = "click_missing_xy"
                continue
            try:
                x = int(parameters["x"])
                y = int(parameters["y"])
            except (TypeError, ValueError):
                reject_reason = "click_invalid_xy"
                continue
            candidate = dict(proposal.get("candidate") or {})
            if str(candidate.get("band", "") or "") != "visible_object":
                admission = (
                    self._repair_prior_objective_slot_exposure_admission_decision(
                        proposal,
                        min_score=min_score,
                    )
                )
                if (
                    bool(admission.get("eligible", False))
                    and str(admission.get("mode", "") or "") == "guarded"
                ):
                    self.repair_prior_objective_slot_exposure_admission_admitted_count += 1
                    proposal["objective_slot_exposure_admission"] = dict(admission)
                else:
                    probe = (
                        self._repair_prior_objective_slot_exposure_probe_decision(
                            proposal,
                            admission=admission,
                        )
                    )
                    if (
                        bool(probe.get("eligible", False))
                        and str(probe.get("mode", "") or "") == "guarded"
                        and bool(
                            self.repair_prior_objective_slot_exposure_probe_execute_enabled
                        )
                    ):
                        self.repair_prior_objective_slot_exposure_probe_admitted_count += 1
                        objective_slot_exposure_probe_selected = True
                        proposal["objective_slot_exposure_admission"] = dict(admission)
                        proposal["objective_slot_exposure_probe"] = dict(probe)
                    else:
                        if bool(admission.get("eligible", False)):
                            self.repair_prior_objective_slot_exposure_admission_blocked_count += 1
                        if bool(probe.get("eligible", False)):
                            self.repair_prior_objective_slot_exposure_probe_blocked_count += 1
                        reject_reason = "objective_slot_exposure_admission_blocked"
                        if bool(probe.get("eligible", False)):
                            reject_reason = "objective_slot_exposure_probe_blocked"
                        if not self._repair_prior_proposal_is_objective_slot_exposure(proposal):
                            reject_reason = "candidate_band_not_visible_object"
                        self._record_repair_prior_objective_slot_exposed_reject(
                            reject_reason,
                            proposal,
                        )
                        continue
            if int(proposal.get("score_context", self._last_score) or self._last_score) != 0:
                self._record_repair_prior_objective_slot_exposed_reject(
                    "score_context_not_zero",
                    proposal,
                )
                return self._reject_repair_prior_guarded_action("score_context_not_zero")
            if (
                not objective_slot_exposure_probe_selected
                and float(proposal.get("top_score", 0.0) or 0.0) < min_score
            ):
                seed_stale_reject_repeat_count = (
                    self._repair_prior_seed_stale_reject_repeat_count(
                        x=x,
                        y=y,
                        candidate=candidate,
                    )
                )
                score = float(proposal.get("top_score", 0.0) or 0.0)
                seed_support = int(proposal.get("chain_value_seed_support", 0) or 0)
                continuation_support = int(
                    proposal.get("chain_continuation_support", 0) or 0
                )
                min_repeats = _repair_prior_seed_stale_reject_min_repeats_from_env()
                floor_score = _repair_prior_seed_stale_reject_floor_score_from_env()
                if (
                    self.repair_prior_seed_stale_reject_recovery_enabled
                    and (seed_support > 0 or continuation_support > 0)
                    and score >= floor_score
                    and seed_stale_reject_repeat_count >= min_repeats
                ):
                    seed_stale_reject_recovered = True
                    if continuation_support > 0 and seed_support <= 0:
                        self.repair_prior_chain_continuation_stale_reject_recovery_count += 1
                else:
                    reject_reason = "prior_score_below_threshold"
                    self._record_repair_prior_objective_slot_exposed_reject(
                        reject_reason,
                        proposal,
                    )
                    break
            if (x, y) in self._repair_prior_guard_executed_sites:
                reject_reason = "prior_site_already_executed"
                self._record_repair_prior_objective_slot_exposed_reject(
                    reject_reason,
                    proposal,
                )
                continue
            if bool(candidate.get("candidate_previously_executed")) or bool(candidate.get("candidate_previously_proposed")):
                reject_reason = "candidate_previously_seen"
                self._record_repair_prior_objective_slot_exposed_reject(
                    reject_reason,
                    proposal,
                )
                continue
            break
        else:
            return self._reject_repair_prior_guarded_action(reject_reason)

        if not proposal or not action_name:
            return self._reject_repair_prior_guarded_action(reject_reason)
        if reject_reason == "prior_score_below_threshold" and float(proposal.get("top_score", 0.0) or 0.0) < min_score:
            return self._reject_repair_prior_guarded_action(reject_reason)

        final_score = float(proposal.get("top_score", 0.0) or 0.0)
        base_score = float(proposal.get("base_prior_score", final_score) or final_score)
        score_lift = max(0.0, final_score - base_score)
        support = int(proposal.get("chain_value_support", 0) or 0)
        seed_support = int(proposal.get("chain_value_seed_support", 0) or 0)
        context_support = int(proposal.get("chain_value_context_support", 0) or 0)
        continuation_support = int(proposal.get("chain_continuation_support", 0) or 0)
        slot_binding_support = bool(proposal.get("chain_slot_binding_support", False))
        slot_ledger_support = bool(proposal.get("chain_slot_ledger_support", False))
        context_frontier_leased = bool(proposal.get("context_frontier_leased", False))
        context_family_backoff = bool(
            proposal.get("chain_value_context_family_backoff", False)
        )
        if support > 0:
            self.repair_prior_chain_value_guard_selected_supported_count += 1
        if seed_support > 0:
            self.repair_prior_chain_value_guard_selected_seed_supported_count += 1
        if context_frontier_leased:
            self.repair_prior_context_frontier_selected_count += 1
        if context_family_backoff:
            self.repair_prior_context_frontier_family_backoff_selected_count += 1
        if continuation_support > 0:
            self.repair_prior_chain_continuation_selected_count += 1
            if slot_binding_support:
                self.repair_prior_chain_slot_binding_selected_count += 1
            if slot_ledger_support:
                self.repair_prior_chain_slot_ledger_selected_count += 1
                self._record_repair_prior_chain_slot_ledger_selection(proposal)
        if score_lift > 0.0:
            self.repair_prior_chain_value_guard_selected_score_lift_count += 1
            if seed_support > 0:
                self.repair_prior_chain_value_guard_selected_seed_score_lift_count += 1
        if base_score < min_score <= final_score:
            self.repair_prior_chain_value_guard_selected_threshold_cross_count += 1
            if seed_support > 0:
                self.repair_prior_chain_value_guard_selected_seed_threshold_cross_count += 1
        if seed_stale_reject_recovered:
            self.repair_prior_seed_stale_reject_recovery_count += 1
        objective_slot_exposure_selected = self._repair_prior_proposal_is_objective_slot_exposure(
            proposal
        )
        if objective_slot_exposure_selected:
            self.repair_prior_objective_slot_exposed_selected_count += 1

        self.repair_prior_guard_adopted_count += 1
        self._repair_prior_guard_executed_sites.add((x, y))
        self._proposal_click_history.append((x, y))
        self.last_repair_prior_guard_decision = {
            "adopted": True,
            "reason": "accepted",
            "action": action_name,
            "has_payload": True,
            "candidate_band": str(candidate.get("band", "") or ""),
            "candidate_kind": str(candidate.get("kind", "") or ""),
            "top_score": round(final_score, 6),
            "base_prior_score": round(base_score, 6),
            "score_lift": round(score_lift, 6),
            "chain_value_support": support,
            "chain_value_seed_support": seed_support,
            "chain_value_context_support": context_support,
            "chain_continuation_support": continuation_support,
            "chain_continuation_ttl": int(
                proposal.get("chain_continuation_ttl", 0) or 0
            ),
            "chain_continuation_source_kind": str(
                proposal.get("chain_continuation_source_kind", "") or ""
            ),
            "chain_slot_binding_support": bool(slot_binding_support),
            "chain_slot_binding_distance": int(
                proposal.get("chain_slot_binding_distance", -1) or -1
            ),
            "chain_slot_ledger_support": bool(slot_ledger_support),
            "chain_slot_ledger_id": str(proposal.get("chain_slot_ledger_id", "") or ""),
            "chain_slot_ledger_status": str(
                proposal.get("chain_slot_ledger_status", "") or ""
            ),
            "chain_slot_ledger_distance": int(
                proposal.get("chain_slot_ledger_distance", -1) or -1
            ),
            "objective_slot_support": bool(
                proposal.get("objective_slot_support", False)
            ),
            "objective_slot_id": str(proposal.get("objective_slot_id", "") or ""),
            "objective_slot_rank": int(proposal.get("objective_slot_rank", -1) or -1),
            "objective_slot_priority": round(
                float(proposal.get("objective_slot_priority", 0.0) or 0.0),
                6,
            ),
            "objective_slot_exposure_candidate": bool(objective_slot_exposure_selected),
            "objective_slot_exposure_admission": dict(
                proposal.get("objective_slot_exposure_admission") or {}
            ),
            "objective_slot_exposure_probe": dict(
                proposal.get("objective_slot_exposure_probe") or {}
            ),
            "objective_slot_exposure_probe_candidate": bool(
                objective_slot_exposure_probe_selected
            ),
            "chain_value_context_family_backoff": bool(context_family_backoff),
            "context_frontier_leased": bool(context_frontier_leased),
            "context_frontier_lease_ttl": int(
                proposal.get("context_frontier_lease_ttl", 0) or 0
            ),
            "context_frontier_lease_injected": bool(
                proposal.get("context_frontier_lease_injected", False)
            ),
            "chain_setup_value": round(float(proposal.get("chain_setup_value", 0.0) or 0.0), 6),
            "chain_immediate_value": round(float(proposal.get("chain_immediate_value", 0.0) or 0.0), 6),
            "chain_value": round(float(proposal.get("chain_value", 0.0) or 0.0), 6),
            "min_score": round(float(min_score), 6),
            "seed_stale_reject_recovered": bool(seed_stale_reject_recovered),
            "seed_stale_reject_repeat_count": int(seed_stale_reject_repeat_count),
        }
        self._start_pending_repair_prior_effect(
            action_name=action_name,
            parameters={"x": x, "y": y},
            proposal=proposal,
        )
        return {"action": action_name, "parameters": {"x": x, "y": y}}

    def _repair_prior_guard_min_score(self) -> float:
        base = _repair_prior_execution_min_score_from_env()
        if not _repair_prior_dynamic_threshold_enabled():
            return base
        if int(self._last_score) != 0:
            return base
        if int(self.repair_prior_effect_score_progress_count) > 0:
            return base
        visible = int(self.repair_prior_effect_visible_change_count)
        if visible >= 2:
            return min(base, _repair_prior_execution_floor_score_from_env())
        if visible >= 1:
            return min(base, 0.80)
        return base

    def _repair_prior_observer_metrics(
        self,
        *,
        include_effect_details: bool = False,
    ) -> dict[str, Any]:
        eval_count = int(self.repair_prior_observer_eval_count)
        contained = int(self.repair_prior_observer_teacher_contained_count)

        def _rate(value: int, denom: int) -> float:
            return round(float(value) / float(denom), 6) if denom else 0.0
        prior_effect_count = int(self.repair_prior_effect_resolved_count)

        metrics = {
            "repair_prior_observer_eval_count": eval_count,
            "repair_prior_observer_teacher_contained_count": contained,
            "repair_prior_observer_teacher_containment_rate": _rate(contained, eval_count),
            "repair_prior_observer_candidate_count_avg": round(
                float(self.repair_prior_observer_candidate_count_sum) / float(eval_count),
                6,
            )
            if eval_count
            else 0.0,
            "repair_prior_observer_top1_count": int(self.repair_prior_observer_top1_count),
            "repair_prior_observer_top3_count": int(self.repair_prior_observer_top3_count),
            "repair_prior_observer_top5_count": int(self.repair_prior_observer_top5_count),
            "repair_prior_observer_top10_count": int(self.repair_prior_observer_top10_count),
            "repair_prior_observer_top1_rate": _rate(
                int(self.repair_prior_observer_top1_count),
                eval_count,
            ),
            "repair_prior_observer_top3_rate": _rate(
                int(self.repair_prior_observer_top3_count),
                eval_count,
            ),
            "repair_prior_observer_top5_rate": _rate(
                int(self.repair_prior_observer_top5_count),
                eval_count,
            ),
            "repair_prior_observer_top10_rate": _rate(
                int(self.repair_prior_observer_top10_count),
                eval_count,
            ),
            "repair_prior_observer_top1_when_contained_rate": _rate(
                int(self.repair_prior_observer_top1_when_contained_count),
                contained,
            ),
            "repair_prior_observer_top3_when_contained_rate": _rate(
                int(self.repair_prior_observer_top3_when_contained_count),
                contained,
            ),
            "repair_prior_observer_top5_when_contained_rate": _rate(
                int(self.repair_prior_observer_top5_when_contained_count),
                contained,
            ),
            "repair_prior_observer_top10_when_contained_rate": _rate(
                int(self.repair_prior_observer_top10_when_contained_count),
                contained,
            ),
            "repair_prior_observer_mrr": round(
                float(self.repair_prior_observer_mrr_sum) / float(eval_count),
                6,
            )
            if eval_count
            else 0.0,
            "repair_prior_observer_last_prediction": dict(
                self.last_repair_prior_observer_prediction or {}
            ),
            "repair_prior_action_proposal_count": int(
                self.repair_prior_action_proposal_count
            ),
            "repair_prior_action_proposal_available": bool(
                (self.last_repair_prior_action_proposal or {}).get("available")
            ),
            "repair_prior_action_proposal": dict(
                self.last_repair_prior_action_proposal or {}
            ),
            "repair_prior_execution_mode": str(self.repair_prior_execution_mode_value or ""),
            "repair_prior_guard_eval_count": int(self.repair_prior_guard_eval_count),
            "repair_prior_guard_adopted_count": int(self.repair_prior_guard_adopted_count),
            "repair_prior_guard_rejected_count": int(self.repair_prior_guard_rejected_count),
            "repair_prior_guard_reject_reason_histogram": dict(
                self.repair_prior_guard_reject_reason_histogram
            ),
            "repair_prior_guard_last_decision": dict(
                self.last_repair_prior_guard_decision or {}
            ),
            "repair_prior_effect_resolved_count": prior_effect_count,
            "repair_prior_effect_visible_change_count": int(
                self.repair_prior_effect_visible_change_count
            ),
            "repair_prior_effect_score_progress_count": int(
                self.repair_prior_effect_score_progress_count
            ),
            "repair_prior_effect_visible_change_rate": _rate(
                int(self.repair_prior_effect_visible_change_count),
                prior_effect_count,
            ),
            "repair_prior_effect_score_progress_rate": _rate(
                int(self.repair_prior_effect_score_progress_count),
                prior_effect_count,
            ),
            "repair_prior_effect_chain_credit_count": int(
                self.repair_prior_effect_chain_credit_count
            ),
            "repair_prior_chain_value_rule_count": int(
                len(self.repair_prior_chain_value_counts)
            ),
            "repair_prior_chain_value_context_rule_count": int(
                sum(1 for key in self.repair_prior_chain_value_counts if len(key) > 3)
            ),
            "repair_prior_chain_value_runtime_feedback_enabled": bool(
                self.repair_prior_chain_value_runtime_feedback_enabled
            ),
            "repair_prior_chain_value_seed_path": str(
                self.repair_prior_chain_value_seed_path or ""
            ),
            "repair_prior_chain_value_seed_key_mode": str(
                self.repair_prior_chain_value_seed_key_mode or "all"
            ),
            "repair_prior_chain_value_seed_rule_count": int(
                self.repair_prior_chain_value_seed_rule_count
            ),
            "repair_prior_chain_value_seed_error": str(
                self.repair_prior_chain_value_seed_error or ""
            ),
            "repair_prior_chain_value_rerank_eval_count": int(
                self.repair_prior_chain_value_rerank_eval_count
            ),
            "repair_prior_chain_value_adjusted_row_count": int(
                self.repair_prior_chain_value_adjusted_row_count
            ),
            "repair_prior_chain_value_seed_adjusted_row_count": int(
                self.repair_prior_chain_value_seed_adjusted_row_count
            ),
            "repair_prior_chain_value_top_supported_count": int(
                self.repair_prior_chain_value_top_supported_count
            ),
            "repair_prior_chain_value_top_seed_supported_count": int(
                self.repair_prior_chain_value_top_seed_supported_count
            ),
            "repair_prior_chain_value_top_changed_count": int(
                self.repair_prior_chain_value_top_changed_count
            ),
            "repair_prior_chain_value_top_score_lift_count": int(
                self.repair_prior_chain_value_top_score_lift_count
            ),
            "repair_prior_chain_value_top_seed_score_lift_count": int(
                self.repair_prior_chain_value_top_seed_score_lift_count
            ),
            "repair_prior_chain_value_top_score_lift_avg": round(
                float(self.repair_prior_chain_value_top_score_lift_sum)
                / float(self.repair_prior_chain_value_top_score_lift_count),
                6,
            )
            if self.repair_prior_chain_value_top_score_lift_count
            else 0.0,
            "repair_prior_chain_value_top_seed_score_lift_avg": round(
                float(self.repair_prior_chain_value_top_seed_score_lift_sum)
                / float(self.repair_prior_chain_value_top_seed_score_lift_count),
                6,
            )
            if self.repair_prior_chain_value_top_seed_score_lift_count
            else 0.0,
            "repair_prior_chain_value_top_guard_threshold_cross_count": int(
                self.repair_prior_chain_value_top_guard_threshold_cross_count
            ),
            "repair_prior_chain_value_top_seed_guard_threshold_cross_count": int(
                self.repair_prior_chain_value_top_seed_guard_threshold_cross_count
            ),
            "repair_prior_chain_value_guard_selected_supported_count": int(
                self.repair_prior_chain_value_guard_selected_supported_count
            ),
            "repair_prior_chain_value_guard_selected_seed_supported_count": int(
                self.repair_prior_chain_value_guard_selected_seed_supported_count
            ),
            "repair_prior_chain_value_guard_selected_score_lift_count": int(
                self.repair_prior_chain_value_guard_selected_score_lift_count
            ),
            "repair_prior_chain_value_guard_selected_seed_score_lift_count": int(
                self.repair_prior_chain_value_guard_selected_seed_score_lift_count
            ),
            "repair_prior_chain_value_guard_selected_threshold_cross_count": int(
                self.repair_prior_chain_value_guard_selected_threshold_cross_count
            ),
            "repair_prior_chain_value_guard_selected_seed_threshold_cross_count": int(
                self.repair_prior_chain_value_guard_selected_seed_threshold_cross_count
            ),
            "repair_prior_context_frontier_lease_enabled": bool(
                self.repair_prior_context_frontier_lease_enabled
            ),
            "repair_prior_context_frontier_lease_ttl": int(
                self.repair_prior_context_frontier_lease_ttl
            ),
            "repair_prior_context_frontier_lease_max_rank": int(
                self.repair_prior_context_frontier_lease_max_rank
            ),
            "repair_prior_context_frontier_seen_count": int(
                self.repair_prior_context_frontier_seen_count
            ),
            "repair_prior_context_frontier_leased_count": int(
                self.repair_prior_context_frontier_leased_count
            ),
            "repair_prior_context_frontier_injected_count": int(
                self.repair_prior_context_frontier_injected_count
            ),
            "repair_prior_context_frontier_family_backoff_enabled": bool(
                self.repair_prior_context_family_backoff_enabled
            ),
            "repair_prior_context_frontier_family_backoff_count": int(
                self.repair_prior_context_frontier_family_backoff_count
            ),
            "repair_prior_context_frontier_family_backoff_selected_count": int(
                self.repair_prior_context_frontier_family_backoff_selected_count
            ),
            "repair_prior_context_frontier_selected_count": int(
                self.repair_prior_context_frontier_selected_count
            ),
            "repair_prior_context_frontier_expired_count": int(
                self.repair_prior_context_frontier_expired_count
            ),
            "repair_prior_context_frontier_best_rank": int(
                self.repair_prior_context_frontier_best_rank
            ),
            "repair_prior_chain_continuation_enabled": bool(
                self.repair_prior_chain_continuation_enabled
            ),
            "repair_prior_chain_continuation_ttl": int(
                self.repair_prior_chain_continuation_ttl
            ),
            "repair_prior_chain_continuation_bands": sorted(
                str(band) for band in self.repair_prior_chain_continuation_bands
            ),
            "repair_prior_chain_slot_binding_enabled": bool(
                self.repair_prior_chain_slot_binding_enabled
            ),
            "repair_prior_chain_slot_binding_max_distance": int(
                self.repair_prior_chain_slot_binding_max_distance
            ),
            "repair_prior_chain_slot_ledger_enabled": bool(
                self.repair_prior_chain_slot_ledger_enabled
            ),
            "repair_prior_chain_slot_ledger_ttl": int(
                self.repair_prior_chain_slot_ledger_ttl
            ),
            "repair_prior_chain_slot_ledger_active_count": int(
                self._repair_prior_chain_slot_ledger_active_count()
            ),
            "repair_prior_chain_continuation_open_count": int(
                self.repair_prior_chain_continuation_open_count
            ),
            "repair_prior_chain_continuation_active_count": int(
                self.repair_prior_chain_continuation_active_count
            ),
            "repair_prior_chain_continuation_supported_count": int(
                self.repair_prior_chain_continuation_supported_count
            ),
            "repair_prior_chain_continuation_selected_count": int(
                self.repair_prior_chain_continuation_selected_count
            ),
            "repair_prior_chain_continuation_stale_reject_recovery_count": int(
                self.repair_prior_chain_continuation_stale_reject_recovery_count
            ),
            "repair_prior_chain_slot_binding_supported_count": int(
                self.repair_prior_chain_slot_binding_supported_count
            ),
            "repair_prior_chain_slot_binding_rejected_count": int(
                self.repair_prior_chain_slot_binding_rejected_count
            ),
            "repair_prior_chain_slot_binding_selected_count": int(
                self.repair_prior_chain_slot_binding_selected_count
            ),
            "repair_prior_chain_slot_binding_best_distance": int(
                self.repair_prior_chain_slot_binding_best_distance
            ),
            "repair_prior_chain_slot_ledger_open_count": int(
                self.repair_prior_chain_slot_ledger_open_count
            ),
            "repair_prior_chain_slot_ledger_supported_count": int(
                self.repair_prior_chain_slot_ledger_supported_count
            ),
            "repair_prior_chain_slot_ledger_selected_count": int(
                self.repair_prior_chain_slot_ledger_selected_count
            ),
            "repair_prior_chain_slot_ledger_closed_count": int(
                self.repair_prior_chain_slot_ledger_closed_count
            ),
            "repair_prior_chain_slot_ledger_expired_count": int(
                self.repair_prior_chain_slot_ledger_expired_count
            ),
            "repair_prior_chain_slot_ledger_consumed_reject_count": int(
                self.repair_prior_chain_slot_ledger_consumed_reject_count
            ),
            "repair_prior_chain_slot_ledger_delayed_credit_count": int(
                self.repair_prior_chain_slot_ledger_delayed_credit_count
            ),
            "repair_prior_chain_slot_ledger_credited_count": int(
                self.repair_prior_chain_slot_ledger_credited_count
            ),
            "repair_prior_chain_slot_ledger_late_credit_count": int(
                self.repair_prior_chain_slot_ledger_late_credit_count
            ),
            "repair_prior_objective_slot_enabled": bool(
                self.repair_prior_objective_slot_enabled
            ),
            "repair_prior_objective_slot_ranking_enabled": bool(
                self.repair_prior_objective_slot_ranking_enabled
            ),
            "repair_prior_objective_slot_candidate_binding_enabled": bool(
                self.repair_prior_objective_slot_candidate_binding_enabled
            ),
            "repair_prior_objective_slot_candidate_exposure_enabled": bool(
                self.repair_prior_objective_slot_candidate_exposure_enabled
            ),
            "repair_prior_objective_slot_exposure_admission_mode": str(
                self.repair_prior_objective_slot_exposure_admission_mode
            ),
            "repair_prior_objective_slot_candidate_binding_max_distance": int(
                self.repair_prior_objective_slot_candidate_binding_max_distance
            ),
            "repair_prior_objective_slot_active_count": int(
                self._repair_prior_objective_slot_active_count()
            ),
            "repair_prior_objective_slot_open_count": int(
                self.repair_prior_objective_slot_open_count
            ),
            "repair_prior_objective_slot_supported_count": int(
                self.repair_prior_objective_slot_supported_count
            ),
            "repair_prior_objective_slot_selected_count": int(
                self.repair_prior_objective_slot_selected_count
            ),
            "repair_prior_objective_slot_credited_count": int(
                self.repair_prior_objective_slot_credited_count
            ),
            "repair_prior_objective_slot_ranked_count": int(
                self.repair_prior_objective_slot_ranked_count
            ),
            "repair_prior_objective_slot_top_ranked_count": int(
                self.repair_prior_objective_slot_top_ranked_count
            ),
            "repair_prior_objective_slot_best_priority": round(
                float(self.repair_prior_objective_slot_best_priority),
                6,
            ),
            "repair_prior_objective_slot_active_max_count": int(
                self.repair_prior_objective_slot_active_max_count
            ),
            "repair_prior_objective_slot_multi_active_eval_count": int(
                self.repair_prior_objective_slot_multi_active_eval_count
            ),
            "repair_prior_objective_slot_candidate_conflict_count": int(
                self.repair_prior_objective_slot_candidate_conflict_count
            ),
            "repair_prior_objective_slot_top_changed_count": int(
                self.repair_prior_objective_slot_top_changed_count
            ),
            "repair_prior_objective_slot_local_score_override_count": int(
                self.repair_prior_objective_slot_local_score_override_count
            ),
            "repair_prior_objective_slot_candidate_bound_count": int(
                self.repair_prior_objective_slot_candidate_bound_count
            ),
            "repair_prior_objective_slot_candidate_binding_rejected_count": int(
                self.repair_prior_objective_slot_candidate_binding_rejected_count
            ),
            "repair_prior_objective_slot_candidate_binding_best_distance": int(
                self.repair_prior_objective_slot_candidate_binding_best_distance
            ),
            "repair_prior_objective_slot_candidate_exposed_count": int(
                self.repair_prior_objective_slot_candidate_exposed_count
            ),
            "repair_prior_objective_slot_candidate_exposure_slot_count": int(
                self.repair_prior_objective_slot_candidate_exposure_slot_count
            ),
            "repair_prior_objective_slot_candidate_exposure_existing_slot_count": int(
                self.repair_prior_objective_slot_candidate_exposure_existing_slot_count
            ),
            "repair_prior_objective_slot_candidate_bound_slot_max_count": int(
                self.repair_prior_objective_slot_candidate_bound_slot_max_count
            ),
            "repair_prior_objective_slot_candidate_bound_slot_histogram": dict(
                self.repair_prior_objective_slot_candidate_bound_slot_histogram
            ),
            "repair_prior_objective_slot_exposed_selected_count": int(
                self.repair_prior_objective_slot_exposed_selected_count
            ),
            "repair_prior_objective_slot_exposed_visible_change_count": int(
                self.repair_prior_objective_slot_exposed_visible_change_count
            ),
            "repair_prior_objective_slot_exposed_score_progress_count": int(
                self.repair_prior_objective_slot_exposed_score_progress_count
            ),
            "repair_prior_objective_slot_exposed_no_visible_change_count": int(
                self.repair_prior_objective_slot_exposed_no_visible_change_count
            ),
            "repair_prior_objective_slot_exposed_stale_reject_count": int(
                self.repair_prior_objective_slot_exposed_stale_reject_count
            ),
            "repair_prior_objective_slot_exposed_reject_reason_histogram": dict(
                self.repair_prior_objective_slot_exposed_reject_reason_histogram
            ),
            "repair_prior_objective_slot_exposure_admission_eval_count": int(
                self.repair_prior_objective_slot_exposure_admission_eval_count
            ),
            "repair_prior_objective_slot_exposure_admission_eligible_count": int(
                self.repair_prior_objective_slot_exposure_admission_eligible_count
            ),
            "repair_prior_objective_slot_exposure_admission_admitted_count": int(
                self.repair_prior_objective_slot_exposure_admission_admitted_count
            ),
            "repair_prior_objective_slot_exposure_admission_blocked_count": int(
                self.repair_prior_objective_slot_exposure_admission_blocked_count
            ),
            "repair_prior_objective_slot_exposure_admission_reason_histogram": dict(
                self.repair_prior_objective_slot_exposure_admission_reason_histogram
            ),
            "repair_prior_objective_slot_exposure_admission_samples": [
                dict(row)
                for row in self.repair_prior_objective_slot_exposure_admission_samples
            ],
            "repair_prior_objective_slot_exposure_probe_mode": str(
                self.repair_prior_objective_slot_exposure_probe_mode
            ),
            "repair_prior_objective_slot_exposure_probe_execute_enabled": bool(
                self.repair_prior_objective_slot_exposure_probe_execute_enabled
            ),
            "repair_prior_objective_slot_exposure_probe_min_score": round(
                float(self.repair_prior_objective_slot_exposure_probe_min_score),
                6,
            ),
            "repair_prior_objective_slot_exposure_probe_max_per_episode": int(
                self.repair_prior_objective_slot_exposure_probe_max_per_episode
            ),
            "repair_prior_objective_slot_exposure_probe_eval_count": int(
                self.repair_prior_objective_slot_exposure_probe_eval_count
            ),
            "repair_prior_objective_slot_exposure_probe_eligible_count": int(
                self.repair_prior_objective_slot_exposure_probe_eligible_count
            ),
            "repair_prior_objective_slot_exposure_probe_admitted_count": int(
                self.repair_prior_objective_slot_exposure_probe_admitted_count
            ),
            "repair_prior_objective_slot_exposure_probe_blocked_count": int(
                self.repair_prior_objective_slot_exposure_probe_blocked_count
            ),
            "repair_prior_objective_slot_exposure_probe_visible_change_count": int(
                self.repair_prior_objective_slot_exposure_probe_visible_change_count
            ),
            "repair_prior_objective_slot_exposure_probe_score_progress_count": int(
                self.repair_prior_objective_slot_exposure_probe_score_progress_count
            ),
            "repair_prior_objective_slot_exposure_probe_no_visible_change_count": int(
                self.repair_prior_objective_slot_exposure_probe_no_visible_change_count
            ),
            "repair_prior_objective_slot_exposure_probe_sandboxed_visible_change_count": int(
                self.repair_prior_objective_slot_exposure_probe_sandboxed_visible_change_count
            ),
            "repair_prior_objective_slot_exposure_probe_reason_histogram": dict(
                self.repair_prior_objective_slot_exposure_probe_reason_histogram
            ),
            "repair_prior_objective_slot_exposure_probe_samples": [
                dict(row)
                for row in self.repair_prior_objective_slot_exposure_probe_samples
            ],
            "repair_prior_unsupported_top_over_context_count": int(
                self.repair_prior_unsupported_top_over_context_count
            ),
            "repair_prior_seed_stale_reject_recovery_enabled": bool(
                self.repair_prior_seed_stale_reject_recovery_enabled
            ),
            "repair_prior_seed_stale_reject_recovery_count": int(
                self.repair_prior_seed_stale_reject_recovery_count
            ),
            "repair_prior_seed_stale_reject_repeat_max": int(
                self.repair_prior_seed_stale_reject_repeat_max
            ),
        }
        if include_effect_details:
            metrics["repair_prior_effect_last_outcome"] = dict(
                self.last_repair_prior_effect_outcome or {}
            )
            metrics["repair_prior_effect_samples"] = [
                dict(row) for row in self.repair_prior_effect_history
            ]
            metrics["repair_prior_effect_recent_outcomes"] = [
                dict(row) for row in self.repair_prior_effect_history[-8:]
            ]
            metrics["repair_prior_chain_value_top_rules"] = (
                self._repair_prior_chain_value_top_rules()
            )
        return metrics

    # ──────────────────────────────────────────────────────────────
    # Phase 47 — teacher coherence tracking
    # ──────────────────────────────────────────────────────────────
    def _track_teacher_coherence(self, *, plan: list[dict[str, Any]]) -> None:
        """Compare current frame's first_action against last frame's
        predicted "next" action under both ECA-followed and teacher-
        followed branches. Update bounded sample ring + counters.

        Stateless contract:
        * Reads ``self._last_predicted_next_action_after_eca`` and
          ``self._last_predicted_next_action_after_teacher`` set by
          the *previous* observe() call.
        * Reads ``self._current_proposal`` (ECA action that ran on
          frame N-1, since runtime caches just before observe).
        * Writes new predictions for frame N+1 to the same fields.
        * Never raises into observe(); caller wraps in try/except.
        """

        if not plan:
            self._last_predicted_next_action_after_eca = None
            self._last_predicted_next_action_after_teacher = None
            self._last_teacher_first_action = ""
            self._last_teacher_first_payload = {}
            self._last_eca_action_for_coherence = ""
            self._last_eca_followed_teacher = None
            return

        teacher_first_entry = dict(plan[0] or {})
        teacher_first_action = _action_name(teacher_first_entry.get("action", ""))
        teacher_first_payload = _payload_for_compare(
            dict(teacher_first_entry.get("parameters") or {})
        )

        # ECA action proposal for the CURRENT frame (recorded by
        # runtime before observe()). We treat this as the action ECA
        # is *about to* commit, which on the next frame becomes the
        # action that was actually taken between frame N-1 and N.
        eca_proposal_action = _action_name(
            (self._current_proposal or {}).get("action", "")
        )

        # ── PHASE 1: compare current teacher_first against last frame's
        # predictions ────────────────────────────────────────────────
        last_after_eca = self._last_predicted_next_action_after_eca
        last_after_teacher = self._last_predicted_next_action_after_teacher

        sample_record: dict[str, Any] | None = None

        if last_after_eca is not None:
            self.teacher_coherence_compare_count += 1
            predicted_action = _action_name(last_after_eca.get("action", ""))
            predicted_payload = _payload_for_compare(
                dict(last_after_eca.get("parameters") or {})
            )
            action_match = bool(
                teacher_first_action
                and predicted_action
                and teacher_first_action == predicted_action
            )
            if action_match:
                self.teacher_coherence_match_count += 1
            else:
                self.teacher_coherence_mismatch_count += 1

            # Payload-level comparison (only if both have payloads)
            payload_match = False
            if predicted_payload and teacher_first_payload:
                self.teacher_coherence_payload_compare_count += 1
                payload_match = predicted_payload == teacher_first_payload
                if payload_match:
                    self.teacher_coherence_payload_match_count += 1

            # Subdivide by whether ECA followed teacher last frame
            if self._last_eca_followed_teacher is True:
                self.teacher_coherence_compare_when_eca_followed += 1
                if action_match:
                    self.teacher_coherence_match_when_eca_followed += 1
            elif self._last_eca_followed_teacher is False:
                self.teacher_coherence_compare_when_eca_diverged += 1
                if action_match:
                    self.teacher_coherence_match_when_eca_diverged += 1

            sample_record = {
                "predicted_after_eca_action": predicted_action,
                "predicted_after_eca_payload": predicted_payload,
                "actual_teacher_first_action": teacher_first_action,
                "actual_teacher_first_payload": teacher_first_payload,
                "action_match": bool(action_match),
                "payload_match": bool(payload_match),
                "eca_followed_teacher_last": self._last_eca_followed_teacher,
            }

        if last_after_teacher is not None:
            self.teacher_self_stability_compare_count += 1
            predicted_action = _action_name(last_after_teacher.get("action", ""))
            stability_match = bool(
                teacher_first_action
                and predicted_action
                and teacher_first_action == predicted_action
            )
            if stability_match:
                self.teacher_self_stability_match_count += 1
            if sample_record is not None:
                sample_record["predicted_after_teacher_action"] = predicted_action
                sample_record["self_stability_match"] = bool(stability_match)

        if sample_record is not None and self.teacher_coherence_max_samples > 0:
            self.teacher_coherence_samples.append(sample_record)
            if len(self.teacher_coherence_samples) > self.teacher_coherence_max_samples:
                # Bounded ring — drop oldest first.
                self.teacher_coherence_samples = self.teacher_coherence_samples[
                    -self.teacher_coherence_max_samples :
                ]

        # ── PHASE 2: compute predictions for next frame ──────────────
        # Determine if ECA is following teacher this frame (for next
        # frame's "ECA followed?" subdivision).
        eca_followed = bool(
            eca_proposal_action
            and teacher_first_action
            and eca_proposal_action == teacher_first_action
        )

        # plan[1] is teacher's predicted "if I take plan[0], next is
        # plan[1]". This is teacher's self-consistent next action.
        if len(plan) >= 2:
            teacher_next = dict(plan[1] or {})
        else:
            teacher_next = {}

        # ``after_teacher``: teacher self-stability — the teacher's
        # plan continues with plan[1] regardless of ECA's choice.
        self._last_predicted_next_action_after_teacher = (
            dict(teacher_next) if teacher_next else None
        )

        # ``after_eca``: prediction conditioned on ECA's actual choice.
        # If ECA followed teacher's first action, the next teacher
        # advice should match plan[1]. If ECA diverged, the next
        # teacher advice could legitimately re-plan, but we still
        # measure it against plan[1] (or plan[0] if ECA chose nothing
        # / NOOP) as a baseline. Empirically interesting: when ECA
        # diverges, does teacher recover with same advice?
        if eca_followed:
            self._last_predicted_next_action_after_eca = (
                dict(teacher_next) if teacher_next else None
            )
        else:
            # When ECA did NOT follow, the next teacher advice ideally
            # re-emits the same first_action (since the world hasn't
            # advanced according to teacher's plan). Predict plan[0].
            self._last_predicted_next_action_after_eca = dict(teacher_first_entry)

        self._last_teacher_first_action = teacher_first_action
        self._last_teacher_first_payload = dict(teacher_first_payload)
        self._last_eca_action_for_coherence = eca_proposal_action
        self._last_eca_followed_teacher = eca_followed

    def teacher_coherence_metrics(self) -> dict[str, Any]:
        """Return the Phase 47 coherence aggregate (rates + samples)."""

        total = int(self.teacher_coherence_compare_count)
        match = int(self.teacher_coherence_match_count)
        rate = float(match) / float(total) if total else 0.0

        followed_total = int(self.teacher_coherence_compare_when_eca_followed)
        followed_match = int(self.teacher_coherence_match_when_eca_followed)
        followed_rate = (
            float(followed_match) / float(followed_total) if followed_total else 0.0
        )

        diverged_total = int(self.teacher_coherence_compare_when_eca_diverged)
        diverged_match = int(self.teacher_coherence_match_when_eca_diverged)
        diverged_rate = (
            float(diverged_match) / float(diverged_total) if diverged_total else 0.0
        )

        payload_total = int(self.teacher_coherence_payload_compare_count)
        payload_match = int(self.teacher_coherence_payload_match_count)
        payload_rate = (
            float(payload_match) / float(payload_total) if payload_total else 0.0
        )

        stability_total = int(self.teacher_self_stability_compare_count)
        stability_match = int(self.teacher_self_stability_match_count)
        stability_rate = (
            float(stability_match) / float(stability_total) if stability_total else 0.0
        )

        return {
            "teacher_coherence_compare_count": total,
            "teacher_coherence_match_count": match,
            "teacher_coherence_mismatch_count": int(self.teacher_coherence_mismatch_count),
            "teacher_coherence_rate": round(rate, 6),
            "teacher_coherence_when_eca_followed_compare": followed_total,
            "teacher_coherence_when_eca_followed_match": followed_match,
            "teacher_coherence_when_eca_followed_rate": round(followed_rate, 6),
            "teacher_coherence_when_eca_diverged_compare": diverged_total,
            "teacher_coherence_when_eca_diverged_match": diverged_match,
            "teacher_coherence_when_eca_diverged_rate": round(diverged_rate, 6),
            "teacher_coherence_payload_compare_count": payload_total,
            "teacher_coherence_payload_match_count": payload_match,
            "teacher_coherence_payload_rate": round(payload_rate, 6),
            "teacher_self_stability_compare_count": stability_total,
            "teacher_self_stability_match_count": stability_match,
            "teacher_self_stability_rate": round(stability_rate, 6),
            "teacher_coherence_samples": [
                dict(row) for row in self.teacher_coherence_samples
            ],
        }

    # ──────────────────────────────────────────────────────────────
    # Phase 48 — shadow injection telemetry
    # ──────────────────────────────────────────────────────────────
    def _record_shadow_decision(
        self,
        *,
        decision: RepairDecision,
        score_at_decision: int,
    ) -> None:
        """Queue a shadow decision for post-hoc K-frame score lift attribution.

        Called immediately after the gate fires. ``decision.repaired_action``
        is the gate's recommendation; ``self._current_proposal`` holds
        the action ECA is about to commit (already cached by the runtime
        before observe()). We record:
        * eca_followed: ECA's action == gate-recommended action
        * payload_distance: |x_eca - x_gate| + |y_eca - y_gate| for clicks
        * score_at_decision: snapshot for K-frame lift computation

        Pending decisions drain via ``_drain_shadow_decisions``.
        """

        if not self.shadow_injection_active:
            return
        eca_action = str((self._current_proposal or {}).get("action", "") or "")
        eca_payload = dict((self._current_proposal or {}).get("payload") or {})
        repaired_action = str(decision.repaired_action or "")
        repaired_payload = dict(decision.repaired_payload or {})
        eca_followed = bool(repaired_action) and eca_action == repaired_action
        # Payload distance only meaningful for click-class actions where
        # both decisions carry numeric (x, y). For non-click pairs we
        # store ``None`` to keep histogram clean.
        payload_distance: float | None = None
        try:
            x_eca = eca_payload.get("x")
            y_eca = eca_payload.get("y")
            x_gate = repaired_payload.get("x")
            y_gate = repaired_payload.get("y")
            if all(v is not None for v in (x_eca, y_eca, x_gate, y_gate)):
                payload_distance = float(abs(int(x_eca) - int(x_gate)) + abs(int(y_eca) - int(y_gate)))
        except (TypeError, ValueError):
            payload_distance = None
        record: dict[str, Any] = {
            "eval_idx": int(self.repair_gate_eval_count),
            "frame_idx": int(self.observed_steps),
            "score_at_decision": int(score_at_decision),
            "eca_action": eca_action,
            "repaired_action": repaired_action,
            "eca_followed": eca_followed,
            "payload_distance": payload_distance,
            "objective_risk": float(decision.objective_mismatch_risk),
            "payload_risk": float(decision.payload_mismatch_risk),
            "ood_fraction": float(decision.ood_fraction),
        }
        self._pending_shadow_decisions.append(record)
        self.shadow_decision_count += 1
        if eca_followed:
            self.shadow_decision_eca_followed_count += 1
        else:
            self.shadow_decision_eca_diverged_count += 1

    def _drain_shadow_decisions(self, *, score_now: int) -> None:
        """Drain decisions older than ``shadow_injection_horizon`` frames.

        Each drained decision contributes (score_now - score_at_decision)
        to either ``_shadow_lift_followed_sum`` or
        ``_shadow_lift_diverged_sum``. Bounded sample ring keeps the
        most recent ``shadow_injection_max_samples`` records.
        """

        if not self.shadow_injection_active or not self._pending_shadow_decisions:
            return
        horizon = int(self.shadow_injection_horizon)
        current_frame = int(self.observed_steps)
        still_pending: list[dict[str, Any]] = []
        for record in self._pending_shadow_decisions:
            age = current_frame - int(record.get("frame_idx", 0))
            if age < horizon:
                still_pending.append(record)
                continue
            lift = float(score_now) - float(record.get("score_at_decision", 0))
            followed = bool(record.get("eca_followed"))
            if followed:
                self.shadow_drained_followed_count += 1
                self._shadow_lift_followed_sum += lift
                if lift > 0:
                    self.shadow_lift_followed_positive_count += 1
            else:
                self.shadow_drained_diverged_count += 1
                self._shadow_lift_diverged_sum += lift
                if lift > 0:
                    self.shadow_lift_diverged_positive_count += 1
            payload_distance = record.get("payload_distance")
            if payload_distance is not None:
                if followed:
                    self._shadow_payload_distance_sum_followed += float(payload_distance)
                    self._shadow_payload_distance_count_followed += 1
                else:
                    self._shadow_payload_distance_sum_diverged += float(payload_distance)
                    self._shadow_payload_distance_count_diverged += 1
            sample_record = dict(record)
            sample_record["score_at_drain"] = int(score_now)
            sample_record["lift"] = float(lift)
            if self.shadow_injection_max_samples > 0:
                self.shadow_injection_samples.append(sample_record)
                if len(self.shadow_injection_samples) > self.shadow_injection_max_samples:
                    self.shadow_injection_samples = self.shadow_injection_samples[
                        -self.shadow_injection_max_samples :
                    ]
        self._pending_shadow_decisions = still_pending

    def shadow_injection_metrics(self) -> dict[str, Any]:
        """Return the Phase 48 shadow-injection aggregate (rates + lift)."""

        followed_drained = int(self.shadow_drained_followed_count)
        diverged_drained = int(self.shadow_drained_diverged_count)
        followed_lift_avg = (
            float(self._shadow_lift_followed_sum) / float(followed_drained)
            if followed_drained
            else 0.0
        )
        diverged_lift_avg = (
            float(self._shadow_lift_diverged_sum) / float(diverged_drained)
            if diverged_drained
            else 0.0
        )
        followed_lift_positive_rate = (
            float(self.shadow_lift_followed_positive_count) / float(followed_drained)
            if followed_drained
            else 0.0
        )
        diverged_lift_positive_rate = (
            float(self.shadow_lift_diverged_positive_count) / float(diverged_drained)
            if diverged_drained
            else 0.0
        )
        followed_payload_avg = (
            float(self._shadow_payload_distance_sum_followed)
            / float(self._shadow_payload_distance_count_followed)
            if self._shadow_payload_distance_count_followed
            else 0.0
        )
        diverged_payload_avg = (
            float(self._shadow_payload_distance_sum_diverged)
            / float(self._shadow_payload_distance_count_diverged)
            if self._shadow_payload_distance_count_diverged
            else 0.0
        )
        return {
            "shadow_injection_mode": self.shadow_injection_mode_value,
            "shadow_injection_active": bool(self.shadow_injection_active),
            "shadow_injection_horizon": int(self.shadow_injection_horizon),
            "shadow_decision_count": int(self.shadow_decision_count),
            "shadow_decision_eca_followed_count": int(self.shadow_decision_eca_followed_count),
            "shadow_decision_eca_diverged_count": int(self.shadow_decision_eca_diverged_count),
            "shadow_drained_followed_count": followed_drained,
            "shadow_drained_diverged_count": diverged_drained,
            "shadow_lift_followed_sum": round(float(self._shadow_lift_followed_sum), 6),
            "shadow_lift_diverged_sum": round(float(self._shadow_lift_diverged_sum), 6),
            "shadow_lift_followed_avg": round(followed_lift_avg, 6),
            "shadow_lift_diverged_avg": round(diverged_lift_avg, 6),
            "shadow_lift_followed_positive_count": int(self.shadow_lift_followed_positive_count),
            "shadow_lift_diverged_positive_count": int(self.shadow_lift_diverged_positive_count),
            "shadow_lift_followed_positive_rate": round(followed_lift_positive_rate, 6),
            "shadow_lift_diverged_positive_rate": round(diverged_lift_positive_rate, 6),
            "shadow_payload_distance_followed_avg": round(followed_payload_avg, 6),
            "shadow_payload_distance_diverged_avg": round(diverged_payload_avg, 6),
            "shadow_payload_distance_followed_count": int(self._shadow_payload_distance_count_followed),
            "shadow_payload_distance_diverged_count": int(self._shadow_payload_distance_count_diverged),
            "shadow_pending_count": int(len(self._pending_shadow_decisions)),
            "shadow_injection_samples": [
                dict(row) for row in self.shadow_injection_samples
            ],
        }

    def step_metrics(self) -> dict[str, Any]:
        advice = dict(self.last_advice or self._empty_advice())
        avg_high_risk = (
            float(self._repair_gate_high_risk_sum) / float(self._repair_gate_high_risk_count)
            if self._repair_gate_high_risk_count
            else 0.0
        )
        ood_fraction_avg = (
            float(self._repair_gate_ood_fraction_sum) / float(self.repair_gate_ood_eval_count)
            if self.repair_gate_ood_eval_count
            else 0.0
        )
        ood_high_rate = (
            float(self.repair_gate_ood_high_count) / float(self.repair_gate_ood_eval_count)
            if self.repair_gate_ood_eval_count
            else 0.0
        )
        action_rate = (
            float(self.repair_gate_action_agreement_count)
            / float(self.repair_gate_action_compare_count)
            if self.repair_gate_action_compare_count
            else 0.0
        )
        action_after_rate = (
            float(self.repair_gate_action_after_repair_agreement_count)
            / float(self.repair_gate_action_compare_count)
            if self.repair_gate_action_compare_count
            else 0.0
        )
        payload_rate = (
            float(self.repair_gate_payload_agreement_count)
            / float(self.repair_gate_payload_compare_count)
            if self.repair_gate_payload_compare_count
            else 0.0
        )
        payload_after_rate = (
            float(self.repair_gate_payload_after_repair_agreement_count)
            / float(self.repair_gate_payload_compare_count)
            if self.repair_gate_payload_compare_count
            else 0.0
        )
        effect_score_progress_rate = (
            float(self.repair_effect_score_progress_count)
            / float(self.repair_effect_resolved_count)
            if self.repair_effect_resolved_count
            else 0.0
        )
        metrics = {
            "solver_bridge_enabled": bool(advice.get("enabled")),
            "solver_bridge_mode": str(advice.get("mode", "")),
            "solver_bridge_available": bool(advice.get("available")),
            "solver_bridge_action_count": int(advice.get("action_count", 0) or 0),
            "solver_bridge_plan_source": str(advice.get("plan_source", "") or ""),
            "solver_bridge_generalisation_tier": str(advice.get("generalisation_tier", "") or ""),
            "solver_bridge_first_action": str(advice.get("first_action", "") or ""),
            "solver_bridge_preview": list(advice.get("preview") or []),
            "solver_bridge_error": str(advice.get("error", "") or ""),
            "solver_bridge_guard_eval_count": int(self.guard_eval_count),
            "solver_bridge_guard_adopted_count": int(self.guard_adopted_count),
            "solver_bridge_guard_rejected_count": int(self.guard_rejected_count),
            "solver_bridge_guard_last_decision": dict(self.last_guard_decision or {}),
            "repair_gate_active": bool(self.repair_gate_active),
            "repair_gate_mode": str(self.repair_gate_mode_value or ""),
            "repair_gate_eval_count": int(self.repair_gate_eval_count),
            "repair_gate_recommended_count": int(self.repair_gate_recommended_count),
            "repair_gate_payload_repaired_count": int(self.repair_gate_payload_repaired_count),
            "repair_gate_objective_repaired_count": int(self.repair_gate_objective_repaired_count),
            "repair_gate_high_risk_avg": round(float(avg_high_risk), 6),
            "repair_gate_last_decision": dict(self.last_repair_decision or {}),
            "repair_gate_action_compare_count": int(self.repair_gate_action_compare_count),
            "repair_gate_action_agreement_count": int(self.repair_gate_action_agreement_count),
            "repair_gate_action_after_repair_agreement_count": int(
                self.repair_gate_action_after_repair_agreement_count
            ),
            "repair_gate_action_agreement_rate": round(float(action_rate), 6),
            "repair_gate_action_after_repair_agreement_rate": round(float(action_after_rate), 6),
            "repair_gate_action_agreement_lift": round(float(action_after_rate - action_rate), 6),
            "repair_gate_payload_compare_count": int(self.repair_gate_payload_compare_count),
            "repair_gate_payload_agreement_count": int(self.repair_gate_payload_agreement_count),
            "repair_gate_payload_after_repair_agreement_count": int(
                self.repair_gate_payload_after_repair_agreement_count
            ),
            "repair_gate_payload_agreement_rate": round(float(payload_rate), 6),
            "repair_gate_payload_after_repair_agreement_rate": round(float(payload_after_rate), 6),
            "repair_gate_payload_agreement_lift": round(float(payload_after_rate - payload_rate), 6),
            "repair_gate_ood_eval_count": int(self.repair_gate_ood_eval_count),
            "repair_gate_ood_high_count": int(self.repair_gate_ood_high_count),
            "repair_gate_ood_high_rate": round(float(ood_high_rate), 6),
            "repair_gate_ood_fraction_avg": round(float(ood_fraction_avg), 6),
            "repair_gate_ood_backoff_engaged_count": int(
                self.repair_gate_ood_backoff_engaged_count
            ),
            "repair_gate_selective_mode_value": str(
                self.repair_gate_selective_mode_value or "off"
            ),
            "repair_gate_selective_stuck_threshold": int(
                self.repair_gate_selective_stuck_threshold
            ),
            "repair_gate_selective_high_risk_threshold": round(
                float(self.repair_gate_selective_high_risk_threshold), 6
            ),
            "repair_gate_selective_ood_max": round(
                float(self.repair_gate_selective_ood_max), 6
            ),
            "repair_gate_selective_eligible_count": int(
                self.repair_gate_selective_eligible_count
            ),
            "repair_gate_selective_blocked_count": int(
                self.repair_gate_selective_blocked_count
            ),
            "repair_gate_selective_blocked_reason_histogram": dict(
                self.repair_gate_selective_blocked_reason_histogram
            ),
            "repair_gate_score_aware_engaged_count": int(
                self.repair_gate_score_aware_engaged_count
            ),
            "repair_gate_score_aware_evals_with_baseline": int(
                self.repair_gate_score_aware_evaluations_with_baseline
            ),
            "repair_gate_game_baseline_parity_engaged_count": int(
                self.repair_gate_game_baseline_parity_engaged_count
            ),
            "adapter_handoff_active": bool(self.adapter_handoff_active),
            "adapter_handoff_games": sorted(self.adapter_handoff_games),
            "adapter_handoff_eval_count": int(self.adapter_handoff_eval_count),
            "adapter_handoff_adopted_count": int(self.adapter_handoff_adopted_count),
            "adapter_handoff_rejected_count": int(self.adapter_handoff_rejected_count),
            "adapter_handoff_reject_reason_histogram": dict(
                self.adapter_handoff_reject_reason_histogram
            ),
            "adapter_handoff_last_decision": dict(self.last_adapter_handoff_decision or {}),
            "adapter_handoff_plan_advance_count": int(self.adapter_handoff_plan_advance_count),
            "adapter_handoff_plan_refresh_count": int(self.adapter_handoff_plan_refresh_count),
            "adapter_handoff_plan_exhaust_count": int(self.adapter_handoff_plan_exhaust_count),
            "adapter_handoff_reset_on_empty": bool(self.adapter_handoff_reset_on_empty),
            "adapter_handoff_reset_emitted_count": int(self.adapter_handoff_reset_emitted_count),
            "adapter_handoff_reset_grace_steps": int(self.adapter_handoff_reset_grace_steps),
            "adapter_handoff_reset_grace_blocked_count": int(self.adapter_handoff_reset_grace_blocked_count),
            "repair_gate_stuck_steps_avg": round(
                float(self._stuck_steps_sum) / float(self._stuck_steps_obs_count)
                if self._stuck_steps_obs_count
                else 0.0,
                6,
            ),
            "repair_gate_stuck_steps_current": int(self._stuck_steps_current),
            "repair_execution_mode": str(self.repair_execution_mode_value or ""),
            "repair_guard_eval_count": int(self.repair_guard_eval_count),
            "repair_guard_adopted_count": int(self.repair_guard_adopted_count),
            "repair_guard_rejected_count": int(self.repair_guard_rejected_count),
            "repair_guard_reject_reason_histogram": dict(self.repair_guard_reject_reason_histogram),
            "repair_guard_last_decision": dict(self.last_repair_guard_decision or {}),
            "repair_distillation_example_count": int(self.repair_distillation_example_count),
            "repair_distillation_stored_count": int(len(self.repair_distillation_samples)),
            "repair_distillation_executed_count": int(self.repair_distillation_executed_count),
            "repair_distillation_action_pair_histogram": dict(
                self.repair_distillation_action_pair_histogram
            ),
            "repair_distillation_target_translation_status_histogram": dict(
                self.repair_distillation_target_translation_status_histogram
            ),
            "repair_distillation_label_histogram": dict(self.repair_distillation_label_histogram),
            "repair_effect_resolved_count": int(self.repair_effect_resolved_count),
            "repair_effect_visible_change_count": int(self.repair_effect_visible_change_count),
            "repair_effect_score_progress_count": int(self.repair_effect_score_progress_count),
            "repair_effect_score_progress_rate": round(float(effect_score_progress_rate), 6),
            "visible_click_candidate_count": int(len(self._last_visible_click_candidates)),
            "click_candidate_memory_count": int(len(self._visible_click_candidate_memory)),
            "teacher_assisted_click_history_count": int(len(self._teacher_assisted_click_history)),
        }
        metrics.update(self._repair_prior_observer_metrics())
        return metrics

    def episode_summary(self) -> dict[str, Any]:
        avg_high_risk = (
            float(self._repair_gate_high_risk_sum) / float(self._repair_gate_high_risk_count)
            if self._repair_gate_high_risk_count
            else 0.0
        )
        ood_fraction_avg = (
            float(self._repair_gate_ood_fraction_sum) / float(self.repair_gate_ood_eval_count)
            if self.repair_gate_ood_eval_count
            else 0.0
        )
        ood_high_rate = (
            float(self.repair_gate_ood_high_count) / float(self.repair_gate_ood_eval_count)
            if self.repair_gate_ood_eval_count
            else 0.0
        )
        action_rate = (
            float(self.repair_gate_action_agreement_count)
            / float(self.repair_gate_action_compare_count)
            if self.repair_gate_action_compare_count
            else 0.0
        )
        action_after_rate = (
            float(self.repair_gate_action_after_repair_agreement_count)
            / float(self.repair_gate_action_compare_count)
            if self.repair_gate_action_compare_count
            else 0.0
        )
        payload_rate = (
            float(self.repair_gate_payload_agreement_count)
            / float(self.repair_gate_payload_compare_count)
            if self.repair_gate_payload_compare_count
            else 0.0
        )
        payload_after_rate = (
            float(self.repair_gate_payload_after_repair_agreement_count)
            / float(self.repair_gate_payload_compare_count)
            if self.repair_gate_payload_compare_count
            else 0.0
        )
        effect_score_progress_rate = (
            float(self.repair_effect_score_progress_count)
            / float(self.repair_effect_resolved_count)
            if self.repair_effect_resolved_count
            else 0.0
        )
        summary = {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "game_id": self.game_id_normalized,
            "module": self.module_name,
            "observed_steps": int(self.observed_steps),
            "available_steps": int(self.available_steps),
            "error_steps": int(self.error_steps),
            "availability_rate": round(
                float(self.available_steps) / float(self.observed_steps),
                6,
            )
            if self.observed_steps
            else 0.0,
            "average_action_count_when_available": round(
                float(self.plan_action_total) / float(self.available_steps),
                6,
            )
            if self.available_steps
            else 0.0,
            "plan_source_histogram": dict(self.source_histogram),
            "generalisation_tier_histogram": dict(self.generalisation_tier_histogram),
            "guard_eval_count": int(self.guard_eval_count),
            "guard_adopted_count": int(self.guard_adopted_count),
            "guard_rejected_count": int(self.guard_rejected_count),
            "guard_reject_reason_histogram": dict(self.guard_reject_reason_histogram),
            "last_guard_decision": dict(self.last_guard_decision or {}),
            "last_advice": dict(self.last_advice or {}),
            "repair_gate_active": bool(self.repair_gate_active),
            "repair_gate_mode": str(self.repair_gate_mode_value or ""),
            "repair_gate_eval_count": int(self.repair_gate_eval_count),
            "repair_gate_recommended_count": int(self.repair_gate_recommended_count),
            "repair_gate_payload_repaired_count": int(self.repair_gate_payload_repaired_count),
            "repair_gate_objective_repaired_count": int(self.repair_gate_objective_repaired_count),
            "repair_gate_high_risk_avg": round(float(avg_high_risk), 6),
            "repair_gate_last_decision": dict(self.last_repair_decision or {}),
            "repair_gate_action_compare_count": int(self.repair_gate_action_compare_count),
            "repair_gate_action_agreement_count": int(self.repair_gate_action_agreement_count),
            "repair_gate_action_after_repair_agreement_count": int(
                self.repair_gate_action_after_repair_agreement_count
            ),
            "repair_gate_action_agreement_rate": round(float(action_rate), 6),
            "repair_gate_action_after_repair_agreement_rate": round(float(action_after_rate), 6),
            "repair_gate_action_agreement_lift": round(float(action_after_rate - action_rate), 6),
            "repair_gate_payload_compare_count": int(self.repair_gate_payload_compare_count),
            "repair_gate_payload_agreement_count": int(self.repair_gate_payload_agreement_count),
            "repair_gate_payload_after_repair_agreement_count": int(
                self.repair_gate_payload_after_repair_agreement_count
            ),
            "repair_gate_payload_agreement_rate": round(float(payload_rate), 6),
            "repair_gate_payload_after_repair_agreement_rate": round(float(payload_after_rate), 6),
            "repair_gate_payload_agreement_lift": round(float(payload_after_rate - payload_rate), 6),
            "repair_gate_ood_eval_count": int(self.repair_gate_ood_eval_count),
            "repair_gate_ood_high_count": int(self.repair_gate_ood_high_count),
            "repair_gate_ood_high_rate": round(float(ood_high_rate), 6),
            "repair_gate_ood_fraction_avg": round(float(ood_fraction_avg), 6),
            "repair_gate_ood_backoff_engaged_count": int(
                self.repair_gate_ood_backoff_engaged_count
            ),
            "repair_gate_selective_mode_value": str(
                self.repair_gate_selective_mode_value or "off"
            ),
            "repair_gate_selective_stuck_threshold": int(
                self.repair_gate_selective_stuck_threshold
            ),
            "repair_gate_selective_high_risk_threshold": round(
                float(self.repair_gate_selective_high_risk_threshold), 6
            ),
            "repair_gate_selective_ood_max": round(
                float(self.repair_gate_selective_ood_max), 6
            ),
            "repair_gate_selective_eligible_count": int(
                self.repair_gate_selective_eligible_count
            ),
            "repair_gate_selective_blocked_count": int(
                self.repair_gate_selective_blocked_count
            ),
            "repair_gate_selective_blocked_reason_histogram": dict(
                self.repair_gate_selective_blocked_reason_histogram
            ),
            "repair_gate_score_aware_engaged_count": int(
                self.repair_gate_score_aware_engaged_count
            ),
            "repair_gate_score_aware_evals_with_baseline": int(
                self.repair_gate_score_aware_evaluations_with_baseline
            ),
            "repair_gate_game_baseline_parity_engaged_count": int(
                self.repair_gate_game_baseline_parity_engaged_count
            ),
            "adapter_handoff_active": bool(self.adapter_handoff_active),
            "adapter_handoff_games": sorted(self.adapter_handoff_games),
            "adapter_handoff_eval_count": int(self.adapter_handoff_eval_count),
            "adapter_handoff_adopted_count": int(self.adapter_handoff_adopted_count),
            "adapter_handoff_rejected_count": int(self.adapter_handoff_rejected_count),
            "adapter_handoff_reject_reason_histogram": dict(
                self.adapter_handoff_reject_reason_histogram
            ),
            "adapter_handoff_last_decision": dict(self.last_adapter_handoff_decision or {}),
            "adapter_handoff_plan_advance_count": int(self.adapter_handoff_plan_advance_count),
            "adapter_handoff_plan_refresh_count": int(self.adapter_handoff_plan_refresh_count),
            "adapter_handoff_plan_exhaust_count": int(self.adapter_handoff_plan_exhaust_count),
            "adapter_handoff_reset_on_empty": bool(self.adapter_handoff_reset_on_empty),
            "adapter_handoff_reset_emitted_count": int(self.adapter_handoff_reset_emitted_count),
            "adapter_handoff_reset_grace_steps": int(self.adapter_handoff_reset_grace_steps),
            "adapter_handoff_reset_grace_blocked_count": int(self.adapter_handoff_reset_grace_blocked_count),
            "repair_gate_stuck_steps_avg": round(
                float(self._stuck_steps_sum) / float(self._stuck_steps_obs_count)
                if self._stuck_steps_obs_count
                else 0.0,
                6,
            ),
            "repair_gate_stuck_steps_current": int(self._stuck_steps_current),
            "repair_execution_mode": str(self.repair_execution_mode_value or ""),
            "repair_guard_eval_count": int(self.repair_guard_eval_count),
            "repair_guard_adopted_count": int(self.repair_guard_adopted_count),
            "repair_guard_rejected_count": int(self.repair_guard_rejected_count),
            "repair_guard_reject_reason_histogram": dict(self.repair_guard_reject_reason_histogram),
            "repair_guard_last_decision": dict(self.last_repair_guard_decision or {}),
            "repair_distillation_example_count": int(self.repair_distillation_example_count),
            "repair_distillation_stored_count": int(len(self.repair_distillation_samples)),
            "repair_distillation_executed_count": int(self.repair_distillation_executed_count),
            "repair_distillation_action_pair_histogram": dict(
                self.repair_distillation_action_pair_histogram
            ),
            "repair_distillation_target_translation_status_histogram": dict(
                self.repair_distillation_target_translation_status_histogram
            ),
            "repair_distillation_label_histogram": dict(self.repair_distillation_label_histogram),
            "repair_effect_resolved_count": int(self.repair_effect_resolved_count),
            "repair_effect_visible_change_count": int(self.repair_effect_visible_change_count),
            "repair_effect_score_progress_count": int(self.repair_effect_score_progress_count),
            "repair_effect_score_progress_rate": round(float(effect_score_progress_rate), 6),
            "visible_click_candidate_count": int(len(self._last_visible_click_candidates)),
            "click_candidate_memory_count": int(len(self._visible_click_candidate_memory)),
            "teacher_assisted_click_history_count": int(len(self._teacher_assisted_click_history)),
            "repair_distillation_samples": list(self.repair_distillation_samples),
        }
        summary.update(self._repair_prior_observer_metrics(include_effect_details=True))
        # Phase 47 — teacher coherence diagnostics.
        summary.update(self.teacher_coherence_metrics())
        # Phase 48 — shadow injection telemetry.
        summary.update(self.shadow_injection_metrics())
        return summary
