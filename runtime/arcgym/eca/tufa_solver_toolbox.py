"""Certified ECA solver plans for the Tufa duck harness (host-side, data only).

The Tufa python sandbox runs in a separate OS process behind an import
allowlist, so callables cannot be injected into it. Instead this module
computes full solution plans privately on the HOST side (inside the solver
process, where ``arcgym`` is importable).  The ECA-lite policy either keeps
them in shadow telemetry or projects an explicit-family, grounded, bounded
prefix into ``eca_advice["solver_plans"]``.

Gate design (postmortem rule: a certificate needs detection AND arbitration):

1. detection   - the family's frame-only perception must parse the frame
2. screening   - the family's MechanicCertificate must certify, where one
                 exists; families without a certificate are reported with
                 ``certified=False`` so the prompt layer can stay silent
3. arbitration - at most one family survives on a frame
4. projection  - only an allowlisted, revocable prefix may reach the LLM
5. verification - every executed step must preserve certification and shrink
                 the private remaining plan; otherwise pure Tufa resumes

Every entry point is exception-safe: a missing/broken arcgym import or a
solver crash degrades to an empty plan list, never an exception escaping
into the harness turn loop.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "arcgym.tufa_solver_toolbox.v1"
POLICY_SCHEMA_VERSION = "arcgym.tufa_toolbox_policy.v1"
LEDGER_SCHEMA_VERSION = "arcgym.tufa_family_outcome_ledger.v1"


FAMILY_ADMISSION: dict[str, dict[str, Any]] = {
    "sand_filter": {
        "transfer_status": "variant_trajectory_verified",
        "evidence": (
            "palette-variant transfer plus canonical L1-L2 cascade graph "
            "replay with exact one-step verification"
        ),
        "initial_successes": 2,
    },
    "mirror": {
        "transfer_status": "variant_partial",
        "evidence": (
            "palette/layout variants, canonical L1-L3 exact replay, and "
            "all25 execution screen with no off-family certification"
        ),
        "initial_successes": 0,
    },
    "color_cycle": {
        "transfer_status": "variant_trajectory_verified",
        "evidence": "palette-role variants plus end-to-end canonical replay",
        "initial_successes": 0,
    },
    "stride_maze": {
        "transfer_status": "variant_partial",
        "evidence": "palette/layout variants plus exact actor-state replay",
        "initial_successes": 0,
    },
    "program_execution": {
        "transfer_status": "canonical_exact_verified",
        "evidence": (
            "polarity/layout variants and canonical L1-L5 translation, scale, "
            "rotation, and shape trace replay with premature RUN rejection"
        ),
        "initial_successes": 0,
    },
    "linked_extent": {
        "transfer_status": "canonical_exact_verified",
        "evidence": (
            "canonical L1-L2 closed-loop replay, exact rebound successor "
            "digests, verified L1 schema inheritance, and L3 shrink fallback"
        ),
        "initial_successes": 0,
    },
    "rule_rewrite": {
        "transfer_status": "canonical_exact_verified",
        "evidence": (
            "selected-glyph probe scope, multi-cycle schema inheritance, "
            "and canonical L1-L6 direct/tree alter-rules exact replay to WIN"
        ),
        "initial_successes": 0,
    },
    "reflection_orbit": {
        "transfer_status": "metamorphic_exact_verified",
        "evidence": (
            "palette-remapped exact replay, canonical L1-L3 score closure, "
            "and zero off-family detections on the all25 initial screen"
        ),
        "initial_successes": 0,
    },
    "anchor_overlap": {
        "transfer_status": "variant_trajectory_verified",
        "evidence": (
            "palette-remapped L1 and canonical L1-L2 exact rigid replay, "
            "with zero off-family certification on the all25 initial screen"
        ),
        "initial_successes": 0,
    },
    "region_paint": {
        "transfer_status": "variant_trajectory_verified",
        "evidence": (
            "palette/layout transfer and canonical L1-L6 exact region replay "
            "to WIN, with zero all25 off-family certification"
        ),
        "initial_successes": 0,
    },
    "shape_marker": {
        "transfer_status": "metamorphic_exact_verified",
        "evidence": (
            "palette/mirror transfer and canonical L1-L4 exact shape-marker "
            "replay with fail-closed static-role checks"
        ),
        "initial_successes": 0,
    },
    "hierarchical_slots": {
        "transfer_status": "metamorphic_exact_verified",
        "evidence": (
            "palette/layout transfer and canonical L1-L8 frame-graph replay "
            "to WIN with exact one-step verification"
        ),
        "initial_successes": 0,
    },
    "recorded_replay": {
        "transfer_status": "metamorphic_exact_verified",
        "evidence": (
            "palette/layout transfer and canonical L1-L3 concurrent replay "
            "with exact actor/ghost transition verification"
        ),
        "initial_successes": 0,
    },
    "cycle_permutation": {
        "transfer_status": "metamorphic_exact_verified",
        "evidence": (
            "palette/layout transfer and canonical L1-L3 visible-history "
            "permutation replay with conservation checks"
        ),
        "initial_successes": 0,
    },
    "control_marker": {
        "transfer_status": "gate_verified_solver_unproven",
        "evidence": (
            "frame-only congruent-piece gate with a unique "
            "drawn control marker; abstains on the other 24 "
            "canonical games"
        ),
        "initial_successes": 0,
    },
}


@dataclass
class FamilyOutcomeLedger:
    """Episode-local positive and negative attribution for family leases."""

    game: str = ""
    entries: list[dict[str, Any]] = field(default_factory=list)
    successes: dict[str, int] = field(default_factory=dict)
    active: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for family, admission in FAMILY_ADMISSION.items():
            self.successes.setdefault(
                family, int(admission.get("initial_successes", 0) or 0)
            )

    def lease_limit(self, family: str, full_plan_len: int) -> tuple[int, str]:
        success_count = max(0, int(self.successes.get(family, 0) or 0))
        if success_count <= 0:
            return min(1, max(1, int(full_plan_len))), "one"
        if success_count == 1:
            return min(2, max(1, int(full_plan_len))), "two"
        if success_count == 2:
            return min(4, max(1, int(full_plan_len))), "four"
        return max(1, int(full_plan_len)), "verified_full"

    def begin(
        self,
        *,
        family: str,
        level: int,
        certificate_reason: str,
        full_plan_len: int,
        action_count: int,
        compute_ms: float,
    ) -> dict[str, Any]:
        if self.active and (
            self.active.get("family") != family
            or int(self.active.get("level", -1)) != int(level)
        ):
            self.close(
                action_count=action_count,
                level_transition=False,
                game_over=False,
                fallback_reason="family_or_level_replaced",
            )
        if self.active is None:
            limit, stage = self.lease_limit(family, full_plan_len)
            self.active = {
                "family": family,
                "game": self.game,
                "level": int(level),
                "certificate_reason": str(certificate_reason),
                "full_plan_length": int(full_plan_len),
                "executed_eca_actions": 0,
                "level_transition": False,
                "game_over": False,
                "actions_to_score": None,
                "compute_milliseconds": float(compute_ms),
                "llm_turns_skipped": 0,
                "fallback_reason": "",
                "started_action_count": int(action_count),
                "lease_limit": int(limit),
                "lease_stage": stage,
                "verified_renewal_count": 0,
                "last_verified_action_count": -1,
                "current_private_plan_length": int(full_plan_len),
                "score_owner": False,
            }
        else:
            self.active["compute_milliseconds"] = round(
                float(self.active.get("compute_milliseconds", 0.0))
                + float(compute_ms),
                3,
            )
            self.active["current_private_plan_length"] = int(full_plan_len)
        return self.active

    def record_execution(self, *, llm_turn_skipped: bool = True) -> None:
        if not self.active:
            return
        self.active["executed_eca_actions"] = int(
            self.active.get("executed_eca_actions", 0)
        ) + 1
        if llm_turn_skipped:
            self.active["llm_turns_skipped"] = int(
                self.active.get("llm_turns_skipped", 0)
            ) + 1

    def renew_verified_step(
        self,
        *,
        full_plan_len: int,
        action_count: int,
        certificate_reason: str,
    ) -> dict[str, Any] | None:
        """Authorize exactly one more action after fresh host verification."""

        if not self.active:
            return None
        if int(action_count) <= int(
            self.active.get("last_verified_action_count", -1) or -1
        ):
            return self.active
        executed = int(self.active.get("executed_eca_actions", 0) or 0)
        self.active["lease_limit"] = max(
            int(self.active.get("lease_limit", 1) or 1),
            executed + 1,
        )
        self.active["lease_stage"] = "verified_stepwise"
        self.active["verified_renewal_count"] = int(
            self.active.get("verified_renewal_count", 0) or 0
        ) + 1
        self.active["last_verified_action_count"] = int(action_count)
        self.active["current_private_plan_length"] = int(full_plan_len)
        self.active["last_verification_reason"] = str(certificate_reason)
        return self.active

    def close(
        self,
        *,
        action_count: int,
        level_transition: bool,
        game_over: bool,
        fallback_reason: str = "",
    ) -> dict[str, Any] | None:
        if not self.active:
            return None
        item = dict(self.active)
        item["level_transition"] = bool(level_transition)
        item["game_over"] = bool(game_over)
        item["fallback_reason"] = str(fallback_reason or "")
        if level_transition:
            item["actions_to_score"] = max(
                0, int(action_count) - int(item.get("started_action_count", action_count))
            )
            item["score_owner"] = True
            family = str(item.get("family", ""))
            self.successes[family] = int(self.successes.get(family, 0) or 0) + 1
        item.pop("started_action_count", None)
        self.entries.append(item)
        self.active = None
        return item

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "game": self.game,
            "successes": dict(sorted(self.successes.items())),
            "active": copy.deepcopy(self.active),
            "entries": copy.deepcopy(self.entries),
        }


_CACHE_LOCK = threading.Lock()
_RESULT_CACHE: "OrderedDict[str, tuple[dict[str, Any], dict[str, Any] | None]]" = OrderedDict()
_PERCEPTION_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_FAMILY_WORKER_LOCK = threading.Lock()
_FAMILY_WORKERS: set[threading.Thread] = set()
_MAX_LIVE_FAMILY_WORKERS = 2


def clear_solver_caches() -> None:
    with _CACHE_LOCK:
        _RESULT_CACHE.clear()
        _PERCEPTION_CACHE.clear()


def _ensure_arcgym_importable() -> None:
    """Make ``arcgym`` importable from wherever this module was vendor-copied.

    The fork's pyproject installs ``rgb-agent`` only — ``arcgym`` is never
    registered as a package, so the harness process cannot import it unless
    the repo root is already on sys.path. Resolution order: explicit
    ``ECA_TUFA_ARCGYM_ROOT``, the repo roots implied by both copy locations
    of this file, the cwd, and the Kaggle source-bundle layouts.
    """
    try:
        import arcgym  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    candidates: list[Path] = []
    env_root = os.environ.get("ECA_TUFA_ARCGYM_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    here = Path(__file__).resolve()
    # arcgym-side copy:  <repo>/arcgym/eca/<this file>          -> parents[2]
    # vendored copy:     <repo>/third_party/tufa_arc3_inference/
    #                    inference/agent/<this file>            -> parents[4]
    for up in (2, 4):
        if len(here.parents) > up:
            candidates.append(here.parents[up])
    cwd = Path.cwd()
    candidates.extend([cwd, cwd / "src" / "arc-agi3-rgb"])
    for base in (Path("/kaggle/working"), Path("/kaggle/input")):
        try:
            if base.exists():
                candidates.extend(base.glob("src/arc-agi3-rgb"))
                candidates.extend(base.glob("*/src/arc-agi3-rgb"))
        except Exception:
            continue
    for cand in candidates:
        try:
            if (cand / "arcgym" / "__init__.py").is_file():
                sys.path.insert(0, str(cand))
                import arcgym  # noqa: F401

                return
        except Exception:
            continue
    raise ModuleNotFoundError(
        "arcgym is not importable from the harness process; "
        "set ECA_TUFA_ARCGYM_ROOT to the arc-agi3-rgb repo root"
    )

# Kept in sync with the Tufa harness action vocabulary
# (inference/agent/action_names.py). Duplicated here because this module is
# vendor-copied into the Tufa checkout and must not depend on either side's
# package layout.
ENGINE_TO_MODEL_ACTION = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "ACTION6": "MOUSE",
    "RESET": "RESET",
}
MODEL_TO_ENGINE_ACTION = {value: key for key, value in ENGINE_TO_MODEL_ACTION.items()}

_MOVE_ACTIONS = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
KNOWN_FAMILIES = (
    "mirror",
    "color_cycle",
    "stride_maze",
    "sand_filter",
    "program_execution",
    "linked_extent",
    "rule_rewrite",
    "reflection_orbit",
    "anchor_overlap",
    "region_paint",
    "shape_marker",
    "hierarchical_slots",
    "recorded_replay",
    "cycle_permutation",
    "control_marker",
)
RUNTIME_PROFILES = ("pure", "shadow", "family_lease", "portfolio_lease")


def _bounded_int(raw: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def resolve_toolbox_policy(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Resolve the pure-first ECA-lite runtime contract.

    Merely enabling the toolbox now means *shadow evaluation*.  Plans reach
    the agent only when the operator explicitly selects ``advisory`` or
    ``prefix`` and names the allowed families.  ``legacy`` is retained solely
    to reproduce v1.2-v1.6 runs that exposed complete plans.
    """
    env = os.environ if environ is None else environ
    profile = str(env.get("ECA_TUFA_PROFILE", "") or "").strip().lower()
    if profile not in {*RUNTIME_PROFILES, ""}:
        profile = ""
    mode = str(env.get("ECA_TUFA_TOOLBOX_MODE", "shadow") or "shadow").strip().lower()
    if profile in {"pure", "shadow"}:
        mode = "shadow"
    elif profile in {"family_lease", "portfolio_lease"}:
        mode = "prefix"
    if mode not in {"shadow", "advisory", "prefix", "legacy"}:
        mode = "shadow"
    requested: list[str] = []
    unknown: list[str] = []
    for token in str(env.get("ECA_TUFA_TOOLBOX_FAMILIES", "") or "").split(","):
        family = token.strip().lower()
        if not family or family in requested or family in unknown:
            continue
        if family in KNOWN_FAMILIES:
            requested.append(family)
        else:
            unknown.append(family)
    disabled = {
        token.strip().lower()
        for token in str(env.get("ECA_TUFA_DISABLE_FAMILIES", "") or "").split(",")
        if token.strip().lower() in KNOWN_FAMILIES
    }
    requested_disabled = [family for family in requested if family in disabled]
    requested = [family for family in requested if family not in disabled]
    # Legacy keeps the historical all-family behaviour when no list is given.
    # A family lease is intentionally singular: arbitration is not a substitute
    # for deciding which solver is allowed to spend CPU or control actions.
    profile_error = ""
    if profile == "family_lease" and requested_disabled:
        profile_error = "family_lease family is disabled by ECA_TUFA_DISABLE_FAMILIES"
        enabled = []
    elif profile == "family_lease" and len(requested) != 1:
        profile_error = "family_lease requires exactly one known toolbox family"
        enabled: list[str] = []
    elif profile == "portfolio_lease" and not requested:
        profile_error = "portfolio_lease requires at least one known toolbox family"
        enabled = []
    elif profile == "portfolio_lease" and unknown:
        profile_error = (
            "portfolio_lease contains unknown toolbox families: "
            + ",".join(unknown)
        )
        enabled = []
    elif profile == "family_lease" and requested[0] not in FAMILY_ADMISSION:
        profile_error = (
            "family_lease requires an evidence-scoped toolbox family; "
            f"{requested[0]} is advisory-only"
        )
        enabled: list[str] = []
    elif profile == "portfolio_lease" and any(
        family not in FAMILY_ADMISSION for family in requested
    ):
        rejected = [family for family in requested if family not in FAMILY_ADMISSION]
        profile_error = (
            "portfolio_lease requires evidence-scoped toolbox families; "
            + ",".join(rejected)
            + " are advisory-only"
        )
        enabled = []
    else:
        enabled = (
            [family for family in KNOWN_FAMILIES if family not in disabled]
            if mode == "legacy" and not requested
            else requested
        )
    requested_prefix_max_actions = _bounded_int(
        env.get("ECA_TUFA_PREFIX_MAX_ACTIONS", "1"), 1, 1, 8
    )
    prefix_max_actions = (
        1
        if profile in {"family_lease", "portfolio_lease"}
        else requested_prefix_max_actions
    )
    lease_default = (
        4 if profile in {"family_lease", "portfolio_lease"} else prefix_max_actions
    )
    lease_max_actions = _bounded_int(
        env.get("ECA_TUFA_LEASE_MAX_ACTIONS", str(lease_default)),
        lease_default,
        1,
        64,
    )
    default_strikes = 2 if mode == "legacy" else 1
    suppression_strikes = _bounded_int(
        env.get("ECA_TUFA_SUPPRESSION_STRIKES", str(default_strikes)),
        default_strikes,
        1,
        4,
    )
    shadow_sample_every_actions = _bounded_int(
        env.get("ECA_TUFA_SHADOW_SAMPLE_EVERY_ACTIONS", "16"), 16, 1, 512
    )
    family_timeout_default = (
        3000 if profile in {"family_lease", "portfolio_lease"} else 750
    )
    family_timeout_ms = _bounded_int(
        env.get("ECA_TUFA_FAMILY_TIMEOUT_MS", str(family_timeout_default)),
        family_timeout_default,
        10,
        10000,
    )
    toolbox_budget_ms = _bounded_int(
        env.get("ECA_TUFA_TOOLBOX_BUDGET_MS", "2500"), 2500, 10, 30000
    )
    family_isolation = str(
        env.get("ECA_TUFA_FAMILY_ISOLATION", "thread") or "thread"
    ).strip().lower()
    if family_isolation not in {"thread", "process"}:
        family_isolation = "thread"
    cache_entries = _bounded_int(
        env.get("ECA_TUFA_CACHE_ENTRIES", "256"), 256, 0, 4096
    )
    shadow_round_robin = str(
        env.get("ECA_TUFA_SHADOW_ROUND_ROBIN", "1") or "1"
    ).strip().lower() in {"1", "true", "yes", "on"}
    mechanic_level_transfer = str(
        env.get(
            "ECA_TUFA_ENABLE_MECHANIC_LEVEL_TRANSFER",
                "1" if profile in {"family_lease", "portfolio_lease"} else "0",
        )
        or "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    mechanic_schema_advisory = str(
        env.get("ECA_TUFA_ENABLE_MECHANIC_SCHEMA_ADVISORY", "0") or "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    mechanic_schema_max_items = _bounded_int(
        env.get("ECA_TUFA_MECHANIC_SCHEMA_MAX_ITEMS", "4"), 4, 1, 8
    )
    toolbox_enabled = (
        profile in {"shadow", "family_lease", "portfolio_lease"}
        if profile
        else str(env.get("ECA_TUFA_ENABLE_TOOLBOX", "0") or "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    pure_baseline = (
        True
        if profile
        else str(env.get("ECA_TUFA_PURE_BASELINE", "0") or "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    enable_prior = (
        False
        if profile
        else str(env.get("ECA_TUFA_ENABLE_PRIOR", "1") or "1").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    prefix_executor = (
        profile in {"family_lease", "portfolio_lease"}
        if profile
        else str(env.get("ECA_TUFA_ENABLE_PREFIX_EXECUTOR", "0") or "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    expose = (
        mode in {"advisory", "prefix", "legacy"}
        and bool(enabled)
        and not profile_error
    )
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "profile": profile or "custom",
        "profile_error": profile_error,
        "pure_baseline": pure_baseline,
        "enable_prior": enable_prior,
        "toolbox_enabled": toolbox_enabled,
        "prefix_executor": prefix_executor,
        "mode": mode,
        "enabled_families": enabled,
        "disabled_families": sorted(disabled),
        "unknown_families": unknown,
        "prefix_max_actions": prefix_max_actions,
        "lease_max_actions": lease_max_actions,
        "suppression_strikes": suppression_strikes,
        "shadow_sample_every_actions": shadow_sample_every_actions,
        "family_timeout_ms": family_timeout_ms,
        "toolbox_budget_ms": toolbox_budget_ms,
        "family_isolation": family_isolation,
        "cache_entries": cache_entries,
        "shadow_round_robin": shadow_round_robin,
        "adaptive_lease": profile in {"family_lease", "portfolio_lease"},
        "mechanic_level_transfer": mechanic_level_transfer,
        "mechanic_schema_advisory": mechanic_schema_advisory,
        "mechanic_schema_max_items": mechanic_schema_max_items,
        "family_admission": copy.deepcopy(FAMILY_ADMISSION),
        "expose_to_agent": expose,
        "auto_execute": mode == "prefix",
        "immediate_fallback": mode != "legacy",
    }


def shadow_sample_due(
    policy: Mapping[str, Any],
    *,
    level: int,
    action_count: int,
    last_level: int,
    last_action_count: int,
) -> bool:
    """Bound shadow CPU cost without changing any model-facing state."""
    if str(policy.get("mode", "shadow")) != "shadow":
        return True
    if int(level) != int(last_level) or int(last_action_count) < 0:
        return True
    every = _bounded_int(
        policy.get("shadow_sample_every_actions", 16), 16, 1, 512
    )
    return int(action_count) - int(last_action_count) >= every


def project_solver_plans(
    raw_result: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    blocked_families: Mapping[str, str] | None = None,
    verified_continuations: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the least-privilege toolbox view that may reach the LLM.

    Shadow mode returns no mechanic data. ECA-lite modes expose exactly one
    authorized plan from an explicit family allowlist and replace its full
    solution with a short revocable prefix. Detected, absent, revoked, and
    otherwise uncertified family evidence remains host-private telemetry.
    """
    policy_data = copy.deepcopy(dict(policy))
    mode = str(policy.get("mode", "shadow") or "shadow")
    if mode == "legacy":
        view = copy.deepcopy(dict(raw_result))
        view["policy"] = policy_data
        view["delivery"] = str(view.get("delivery", "advertise") or "advertise")
        return view

    view: dict[str, Any] = {
        "schema_version": str(raw_result.get("schema_version", SCHEMA_VERSION)),
        "available": bool(raw_result.get("available", True)),
        "delivery": "shadow",
        "policy": {
            "schema_version": policy_data.get("schema_version", POLICY_SCHEMA_VERSION),
            "profile": policy_data.get("profile", "custom"),
            "mode": mode,
            "prefix_max_actions": policy_data.get("prefix_max_actions", 1),
        },
        "plans": [],
        "runtime_mechanics": [],
        "skipped": [],
        "observations": {
            "confidence_note": (
                "Only a unique currently authorized family may cross the model boundary"
            ),
            "families": [],
        },
        "exposed_families": [],
    }
    allowed = {str(f) for f in policy.get("enabled_families", [])}
    blocked = {str(k): str(v) for k, v in (blocked_families or {}).items()}
    continuations = {
        str(k): str(v) for k, v in (verified_continuations or {}).items()
    }
    # Every non-legacy model-facing lease is exactly one action. The host keeps
    # the residual plan and recomputes after observing the real transition.
    max_actions = 1
    expose = bool(policy.get("expose_to_agent")) and mode in {"advisory", "prefix"}
    if expose and isinstance(raw_result.get("measured_hazards"), Mapping):
        view["measured_hazards"] = copy.deepcopy(
            dict(raw_result["measured_hazards"])
        )
    candidates: list[dict[str, Any]] = []
    if expose:
        for raw_plan in raw_result.get("plans", []):
            if not isinstance(raw_plan, Mapping):
                continue
            family = str(raw_plan.get("family", ""))
            if family not in allowed or family in blocked:
                continue
            current_authority = bool(
                raw_plan.get("authorized", raw_plan.get("certified", False))
            )
            continuity_authority = family in continuations
            if not current_authority and not continuity_authority:
                continue
            plan = copy.deepcopy(dict(raw_plan))
            full_actions = [
                copy.deepcopy(step)
                for step in plan.get("actions", [])
                if isinstance(step, Mapping)
            ]
            prefix = full_actions[:max_actions]
            if not prefix:
                continue
            plan["actions"] = prefix
            private_plan_length = max(
                len(full_actions),
                int(plan.get("private_plan_length", 0) or 0),
            )
            plan["private_plan_length"] = private_plan_length
            plan["full_action_count"] = private_plan_length
            plan["action_count"] = len(prefix)
            plan["bounded_prefix"] = True
            plan["requires_stepwise_verification"] = True
            plan["authorized"] = True
            if continuity_authority and not plan.get("certified"):
                plan["continuity_verified"] = True
                plan["continuity_reason"] = continuations[family]
            candidates.append(plan)

    # A portfolio is model-visible only after arbitration leaves one owner.
    # Multiple surviving candidates are treated as an authority bug and fail
    # closed even when they happen to advertise the same first action.
    if len(candidates) == 1:
        view["delivery"] = "advisory"
        view["plans"] = candidates
        view["exposed_families"] = [str(candidates[0].get("family", ""))]
    return view


def _stable_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_data(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, np.ndarray):
        return _stable_data(value.tolist())
    if isinstance(value, (set, frozenset)):
        items = [_stable_data(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_stable_data(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        _stable_data(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _family_verification_state(
    family: str,
    state: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if family == "mirror":
        static = {
            key: state.get(key)
            for key in ("walkable", "hazards", "cell", "origin", "grid_size")
        }
        dynamic = {"cursor_a": state.get("cursor_a"), "cursor_b": state.get("cursor_b")}
    elif family == "stride_maze":
        static = {
            key: state.get(key)
            for key in (
                "walkable",
                "targets",
                "forbidden",
                "stride",
                "edge_offset",
                "target_value",
            )
        }
        dynamic = {"actor_pos": state.get("actor_pos")}
    elif family == "color_cycle":
        static = {
            "palette_size": len(state.get("palette") or ()),
            "templates": state.get("templates"),
            "clickable_positions": sorted(
                [tuple(item.get("pos") or ()) for item in state.get("solid") or ()]
                + [tuple(item.get("pos") or ()) for item in state.get("brush") or ()]
            ),
        }
        dynamic = {
            "solid": state.get("solid"),
            "brush": state.get("brush"),
        }
    else:
        return {}
    return {
        "family": family,
        "static_signature": _stable_digest(static),
        "dynamic_signature": _stable_digest(dynamic),
        "plan_hash": _stable_digest(list(actions)),
        "plan_len": len(actions),
    }


def _family_exact_prediction(
    family: str,
    state: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    verification_state: Mapping[str, Any],
    expected_change: Mapping[str, Any],
) -> dict[str, Any] | None:
    if family not in {"mirror", "stride_maze", "color_cycle"} or not actions:
        return None
    prediction: dict[str, Any] = {
        "verifier": f"{family}_state_transition_v1",
        "family": family,
        "static_signature": str(verification_state.get("static_signature", "")),
        "dynamic_signature_before": str(
            verification_state.get("dynamic_signature", "")
        ),
        "expected_remaining_plan_hash": _stable_digest(list(actions[1:])),
        "expected_remaining_plan_len": max(0, len(actions) - 1),
    }
    if family == "mirror":
        prediction["expected_dynamic_signature"] = _stable_digest(
            {
                "cursor_a": (
                    int(expected_change["cursor_a_after"]["col"]),
                    int(expected_change["cursor_a_after"]["row"]),
                ),
                "cursor_b": (
                    int(expected_change["cursor_b_after"]["col"]),
                    int(expected_change["cursor_b_after"]["row"]),
                ),
            }
        )
    elif family == "stride_maze":
        prediction["expected_dynamic_signature"] = _stable_digest(
            {
                "actor_pos": (
                    int(expected_change["actor_after"]["col"]),
                    int(expected_change["actor_after"]["row"]),
                )
            }
        )
    return prediction


def grounded_transition_matches(
    expected_change: Mapping[str, Any] | None,
    current_grounding: Mapping[str, Any] | None,
    *,
    current_frame: Any = None,
) -> bool | None:
    """Compare a prior evidence-card prediction with newly parsed entities.

    Returns ``True`` for a grounded match, ``False`` for a grounded
    contradiction, and ``None`` when this family does not provide an exact
    entity-position prediction (those cases still require recertification).
    """
    if not isinstance(expected_change, Mapping) or not isinstance(
        current_grounding, Mapping
    ):
        return None
    runtime_exact = expected_change.get("runtime_mechanic_exact")
    if isinstance(runtime_exact, Mapping):
        if current_frame is None:
            return False
        try:
            from arcgym.eca.tufa_mechanics.bridge import verify_runtime_exact

            return verify_runtime_exact(runtime_exact, current_frame)
        except Exception:
            return False
    for family in ("mirror", "stride_maze", "color_cycle"):
        exact = expected_change.get(f"{family}_exact")
        if not isinstance(exact, Mapping):
            continue
        if str(exact.get("verifier", "")) != f"{family}_state_transition_v1":
            return False
        current_state = current_grounding.get("verification_state")
        if not isinstance(current_state, Mapping):
            return False
        if str(current_state.get("family", "")) != family:
            return False
        if str(current_state.get("static_signature", "")) != str(
            exact.get("static_signature", "")
        ):
            return False
        if int(current_state.get("plan_len", -1)) != int(
            exact.get("expected_remaining_plan_len", -2)
        ):
            return False
        if str(current_state.get("plan_hash", "")) != str(
            exact.get("expected_remaining_plan_hash", "")
        ):
            return False
        current_dynamic = str(current_state.get("dynamic_signature", ""))
        if not current_dynamic or current_dynamic == str(
            exact.get("dynamic_signature_before", "")
        ):
            return False
        expected_dynamic = str(exact.get("expected_dynamic_signature", "") or "")
        if expected_dynamic and current_dynamic != expected_dynamic:
            return False
        return True
    sand_expected = expected_change.get("sand_filter_exact")
    if isinstance(sand_expected, Mapping):
        verifier = str(sand_expected.get("verifier", "") or "")
        if verifier == "sand_filter_level_transition_v1":
            # This matcher is called only while the host is still on the same
            # level. A submit action that did not transition is a contradiction.
            return False
        if verifier != "sand_filter_exact_frame_v1":
            return False
        current_state = current_grounding.get("verification_state")
        if not isinstance(current_state, Mapping):
            return False
        try:
            current_potential = int(current_state.get("flow_potential", -1))
            expected_potential = int(sand_expected.get("flow_potential_after", -2))
            before_potential = int(sand_expected.get("flow_potential_before", -1))
        except (TypeError, ValueError):
            return False
        if current_potential != expected_potential or current_potential >= before_potential:
            return False
        arr = _as_2d_frame(current_frame)
        shape = list(sand_expected.get("frame_shape") or ())
        if arr is None or len(shape) != 2 or list(arr.shape) != [int(shape[0]), int(shape[1])]:
            return False
        actual_hash = _masked_frame_hash(
            arr,
            [
                item
                for item in sand_expected.get("hash_exclusions", [])
                if isinstance(item, Mapping)
            ],
        )
        return actual_hash == str(sand_expected.get("expected_frame_hash", ""))
    entities = {
        str(item.get("role")): item
        for item in current_grounding.get("entities", [])
        if isinstance(item, Mapping) and item.get("role")
    }
    comparisons = (
        ("cursor_a_after", "cursor_a"),
        ("cursor_b_after", "cursor_b"),
        ("actor_after", "actor"),
    )
    checked = 0
    for expected_key, role in comparisons:
        expected = expected_change.get(expected_key)
        if not isinstance(expected, Mapping):
            continue
        actual = entities.get(role)
        if not isinstance(actual, Mapping):
            return False
        checked += 1
        if int(actual.get("row", -1)) != int(expected.get("row", -2)):
            return False
        if int(actual.get("col", -1)) != int(expected.get("col", -2)):
            return False
    return True if checked else None


def grounded_transition_diagnostic(
    expected_change: Mapping[str, Any] | None,
    current_grounding: Mapping[str, Any] | None,
    *,
    current_frame: Any = None,
) -> dict[str, Any]:
    """Return exact-verifier evidence suitable for repair attribution."""

    if isinstance(expected_change, Mapping):
        runtime_exact = expected_change.get("runtime_mechanic_exact")
        if isinstance(runtime_exact, Mapping):
            if current_frame is None:
                return {
                    "accepted": False,
                    "classification": "missing_current_frame",
                    "toolbox_family": str(
                        runtime_exact.get("toolbox_family", "") or ""
                    ),
                }
            try:
                from arcgym.eca.tufa_mechanics.bridge import (
                    verify_runtime_exact_diagnostic,
                )

                return verify_runtime_exact_diagnostic(runtime_exact, current_frame)
            except Exception as exc:
                return {
                    "accepted": False,
                    "classification": "verifier_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
    matched = grounded_transition_matches(
        expected_change,
        current_grounding,
        current_frame=current_frame,
    )
    return {
        "accepted": matched,
        "classification": (
            "exact_match"
            if matched is True
            else "true_reject"
            if matched is False
            else "not_comparable"
        ),
    }


def grounded_progress_matches(
    expected_change: Mapping[str, Any] | None,
    current_grounding: Mapping[str, Any] | None,
) -> bool | None:
    """Verify a family-provided monotone replanning measure.

    This complements remaining-plan shrink for mechanics that intentionally
    reveal and regenerate only one action at a time.
    """

    if not isinstance(expected_change, Mapping) or not isinstance(
        current_grounding, Mapping
    ):
        return None
    expected = expected_change.get("progress_measure")
    current = current_grounding.get("progress_measure")
    if not isinstance(expected, Mapping) or not isinstance(current, Mapping):
        return None
    metric = str(expected.get("metric", "") or "")
    if not metric or str(current.get("metric", "") or "") != metric:
        return False
    try:
        before = float(expected["before"])
        expected_after = float(expected["expected_after"])
        rebound = float(current["before"])
    except (KeyError, TypeError, ValueError):
        return False
    if bool(expected.get("strict", True)):
        return rebound < before and rebound <= expected_after
    return rebound <= expected_after


def _as_2d_frame(frame: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(frame)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


def _engine_valid_actions(valid_actions: Iterable[Any] | None) -> set[str]:
    """Normalize a mixed model/engine action list to engine names.

    ``None`` means "unknown" and is treated as fully permissive so the
    toolbox stays useful when the harness does not thread valid_actions.
    """
    if valid_actions is None:
        return set(ENGINE_TO_MODEL_ACTION)
    out: set[str] = set()
    for item in valid_actions:
        raw = str(item or "").strip().upper()
        if not raw:
            continue
        if raw in ENGINE_TO_MODEL_ACTION:
            out.add(raw)
        elif raw in MODEL_TO_ENGINE_ACTION:
            out.add(MODEL_TO_ENGINE_ACTION[raw])
    return out


def _to_model_steps(steps: Sequence[Any]) -> list[dict[str, Any]] | None:
    """Convert engine-plan steps to Tufa model payloads.

    Accepts ``["ACTION1", ...]`` and ``[("ACTION6", {"x":..,"y":..}), ...]``.
    Click payloads use frame pixel coords where x is the column and y is the
    row; Tufa's MOUSE payload is {"row", "col"}.
    """
    out: list[dict[str, Any]] = []
    for item in steps:
        if isinstance(item, str):
            action = ENGINE_TO_MODEL_ACTION.get(item.strip().upper())
            if action is None:
                return None
            out.append({"action": action})
            continue
        try:
            action_raw, payload = item
        except Exception:
            return None
        action = ENGINE_TO_MODEL_ACTION.get(str(action_raw).strip().upper())
        if action is None:
            return None
        entry: dict[str, Any] = {"action": action}
        if action == "MOUSE":
            if not isinstance(payload, Mapping):
                return None
            entry["row"] = int(payload.get("y", 0) or 0)
            entry["col"] = int(payload.get("x", 0) or 0)
        out.append(entry)
    return out or None


def _verdict_fields(verdict: Any) -> tuple[bool, str]:
    certified = bool(getattr(verdict, "certified", False))
    reason = str(getattr(verdict, "reason", "") or "")
    return certified, reason


def _obs(family: str, status: str, reason: str, **detail: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"family": family, "status": status, "reason": reason}
    if detail:
        out["detail"] = detail
    return out


def _rc(pos: Sequence[Any]) -> dict[str, int]:
    """Convert internal (x, y) coordinates to the agent's (row, col)."""
    return {"row": int(pos[1]), "col": int(pos[0])}


def _masked_frame_hash(
    frame: Any, exclusions: Sequence[Mapping[str, Any]] = ()
) -> str | None:
    arr = _as_2d_frame(frame)
    if arr is None:
        return None
    mask = np.ones(arr.shape, dtype=bool)
    for rect in exclusions:
        try:
            row0 = max(0, min(arr.shape[0], int(rect.get("row0", 0))))
            row1 = max(row0, min(arr.shape[0], int(rect.get("row1", row0))))
            col0 = max(0, min(arr.shape[1], int(rect.get("col0", 0))))
            col1 = max(col0, min(arr.shape[1], int(rect.get("col1", col0))))
        except (TypeError, ValueError):
            return None
        mask[row0:row1, col0:col1] = False
    payload = np.ascontiguousarray(arr[mask]).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _component_pixel_box(
    item: Mapping[str, Any], state: Mapping[str, Any], shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    grid_size = list(state.get("grid_size") or ())
    if len(grid_size) < 2:
        return None
    try:
        grid_w, grid_h = int(grid_size[0]), int(grid_size[1])
        x, y = int(item.get("x", 0)), int(item.get("y", 0))
        w, h = int(item.get("w", 0)), int(item.get("h", 0))
    except (TypeError, ValueError):
        return None
    if min(grid_w, grid_h, w, h) <= 0:
        return None
    frame_h, frame_w = shape
    col0 = int(round(x * frame_w / grid_w))
    col1 = int(round((x + w) * frame_w / grid_w))
    row0 = int(round(y * frame_h / grid_h))
    row1 = int(round((y + h) * frame_h / grid_h))
    if not (0 <= row0 < row1 <= frame_h and 0 <= col0 < col1 <= frame_w):
        return None
    return row0, row1, col0, col1


def _local_background_value(
    arr: np.ndarray, box: tuple[int, int, int, int]
) -> Any | None:
    row0, row1, col0, col1 = box
    pieces = []
    if row0 > 0:
        pieces.append(arr[row0 - 1 : row0, col0:col1])
    if row1 < arr.shape[0]:
        pieces.append(arr[row1 : row1 + 1, col0:col1])
    if col0 > 0:
        pieces.append(arr[row0:row1, col0 - 1 : col0])
    if col1 < arr.shape[1]:
        pieces.append(arr[row0:row1, col1 : col1 + 1])
    if not pieces:
        return None
    values, counts = np.unique(np.concatenate([part.ravel() for part in pieces]), return_counts=True)
    return values[int(np.argmax(counts))]


def _refine_component_pixel_box(
    arr: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    padding: int,
) -> tuple[int, int, int, int] | None:
    """Snap a sampled logical box to the visible non-background rectangle."""
    row0, row1, col0, col1 = box
    er0, er1 = max(0, row0 - padding), min(arr.shape[0], row1 + padding)
    ec0, ec1 = max(0, col0 - padding), min(arr.shape[1], col1 + padding)
    expanded = arr[er0:er1, ec0:ec1]
    if expanded.size == 0:
        return None
    border = np.concatenate(
        [expanded[0, :], expanded[-1, :], expanded[:, 0], expanded[:, -1]]
    )
    values, counts = np.unique(border, return_counts=True)
    background = values[int(np.argmax(counts))]
    ys, xs = np.where(expanded != background)
    if not len(xs):
        return None
    refined = (
        er0 + int(ys.min()),
        er0 + int(ys.max()) + 1,
        ec0 + int(xs.min()),
        ec0 + int(xs.max()) + 1,
    )
    # Refuse a region that expanded far beyond the reader's component guess;
    # that means another nearby object contaminated the candidate window.
    rr0, rr1, rc0, rc1 = refined
    if rr1 - rr0 > (row1 - row0) + 2 * padding:
        return None
    if rc1 - rc0 > (col1 - col0) + 2 * padding:
        return None
    return refined


def _sand_filter_exact_prediction(
    state: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    frame: Any,
) -> dict[str, Any] | None:
    """Predict one sand-filter host action at pixel precision.

    The reader can switch between its 16x16 and 20x20 logical grids after a
    move, so logical entity coordinates are not a stable transition oracle.
    Instead, movement leases carry a hash of the entire predicted next frame,
    excluding only the two-pixel action counter in the upper-right corner.
    That makes source/bowl and non-operated-component invariance structural.
    """
    if not actions:
        return None
    first = actions[0]
    action = str(first.get("action", "") or "").upper()
    before_potential = int(state.get("operation_plan_len", 0) or 0)
    if action == "SPACE" and before_potential == 1:
        return {
            "verifier": "sand_filter_level_transition_v1",
            "requires_level_transition": True,
            "flow_potential_before": before_potential,
        }
    deltas = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
    }
    delta = deltas.get(action)
    if delta is None:
        return None
    if int(state.get("rotation_k", 0) or 0) == 2:
        delta = (-delta[0], -delta[1])
    arr = _as_2d_frame(frame)
    if arr is None:
        return None
    selected: list[tuple[str, Mapping[str, Any]]] = []
    for kind, items in (
        ("filter", state.get("filters") or ()),
        ("reflector", state.get("reflectors") or ()),
    ):
        selected.extend(
            (kind, item)
            for item in items
            if isinstance(item, Mapping) and bool(item.get("selected", False))
        )
    if len(selected) != 1:
        return None
    component_kind, component = selected[0]
    box = _component_pixel_box(component, state, arr.shape)
    grid_size = list(state.get("grid_size") or ())
    if box is None or len(grid_size) < 2:
        return None
    try:
        padding = max(
            2,
            int(
                np.ceil(
                    max(
                        arr.shape[1] / int(grid_size[0]),
                        arr.shape[0] / int(grid_size[1]),
                    )
                )
            ),
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    refined = _refine_component_pixel_box(arr, box, padding=padding)
    if refined is not None:
        box = refined
    try:
        row0, row1, col0, col1 = box
        logical_w = max(1, int(component.get("w", 1) or 1))
        logical_h = max(1, int(component.get("h", 1) or 1))
        pixel_step = max(
            1,
            int(
                round(
                    max(
                        (col1 - col0) / logical_w,
                        (row1 - row0) / logical_h,
                    )
                )
            ),
        )
        col_delta = int(delta[0] * pixel_step)
        row_delta = int(delta[1] * pixel_step)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    target = (
        row0 + row_delta,
        row1 + row_delta,
        col0 + col_delta,
        col1 + col_delta,
    )
    tr0, tr1, tc0, tc1 = target
    if not (0 <= tr0 < tr1 <= arr.shape[0] and 0 <= tc0 < tc1 <= arr.shape[1]):
        return None
    background = _local_background_value(arr, box)
    if background is None:
        return None
    expected = arr.copy()
    crop = arr[row0:row1, col0:col1].copy()
    expected[row0:row1, col0:col1] = background
    expected[tr0:tr1, tc0:tc1] = crop
    # The ARC engine increments a tiny action counter at the upper-right; it
    # is unrelated to the mechanic and is the only excluded dynamic region.
    exclusions = [
        {
            "row0": 0,
            "row1": min(1, arr.shape[0]),
            "col0": max(0, arr.shape[1] - 8),
            "col1": arr.shape[1],
        }
    ]
    expected_hash = _masked_frame_hash(expected, exclusions)
    if expected_hash is None or np.array_equal(expected, arr):
        return None
    return {
        "verifier": "sand_filter_exact_frame_v1",
        "component_kind": component_kind,
        "action": action,
        "component_box_before": [row0, row1, col0, col1],
        "component_box_after": [tr0, tr1, tc0, tc1],
        "expected_frame_hash": expected_hash,
        "hash_exclusions": exclusions,
        "frame_shape": [int(arr.shape[0]), int(arr.shape[1])],
        "flow_potential_before": before_potential,
        "flow_potential_after": max(0, before_potential - 1),
    }


def _grounded_observation(
    family: str,
    state: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    *,
    frame: Any = None,
) -> dict[str, Any]:
    """Translate solver internals into an LLM-checkable evidence card.

    Family labels alone are too abstract for the tool agent.  Each card ties
    the claim to visible roles/coordinates, predicts the next local change,
    states what should remain invariant, and names an immediate falsifier.
    """
    first = copy.deepcopy(dict(actions[0])) if actions else {}
    card: dict[str, Any] = {
        "coordinate_convention": "row,col; grid coordinates are labelled logical_grid",
        "entities": [],
        "proposed_first_action": first,
        "expected_change": {},
        "invariants": [],
        "falsifiers": [],
        "confidence_boundary": (
            "certificate supports mechanic detection only; the next observed "
            "transition must still match this card"
        ),
    }
    if family == "mirror":
        a = tuple(state.get("cursor_a") or (0, 0))
        b = tuple(state.get("cursor_b") or (0, 0))
        card["entities"] = [
            {"role": "cursor_a", "space": "logical_grid", **_rc(a)},
            {"role": "cursor_b", "space": "logical_grid", **_rc(b)},
        ]
        delta = {
            "UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)
        }.get(str(first.get("action", "")), (0, 0))
        walkable = set(state.get("walkable") or ())
        intended_a = (int(a[0]) + delta[0], int(a[1]) + delta[1])
        intended_b = (int(b[0]) - delta[0], int(b[1]) + delta[1])
        after_a = intended_a if intended_a in walkable else a
        after_b = intended_b if intended_b in walkable else b
        card["expected_change"] = {
            "cursor_a_after": {"space": "logical_grid", **_rc(after_a)},
            "cursor_b_after": {"space": "logical_grid", **_rc(after_b)},
            "rule": "horizontal inputs move the two cursors oppositely; vertical inputs move both together",
        }
        card["invariants"] = ["walkable and hazard cells remain fixed"]
        card["falsifiers"] = [
            "either visible cursor fails to occupy its predicted logical-grid cell",
            "a non-cursor object changes unexpectedly",
        ]
    elif family == "stride_maze":
        actor = tuple(state.get("actor_pos") or (0, 0))
        targets = [tuple(pos) for pos in state.get("targets") or []]
        card["entities"] = [
            {"role": "actor", "space": "frame", **_rc(actor)},
            *[
                {"role": "target", "space": "frame", **_rc(pos)}
                for pos in targets[:3]
            ],
        ]
        stride = int(state.get("stride", 1) or 1)
        delta = {
            "UP": (0, -stride), "DOWN": (0, stride),
            "LEFT": (-stride, 0), "RIGHT": (stride, 0),
        }.get(str(first.get("action", "")), (0, 0))
        intended = (int(actor[0]) + delta[0], int(actor[1]) + delta[1])
        # The stride reader stores corridor marker coordinates in ``walkable``;
        # actor centers lie between those markers. The BFS already proved the
        # first edge, so the exact actor prediction is the stride displacement.
        after = intended
        card["expected_change"] = {
            "actor_after": {"space": "frame", **_rc(after)},
            "stride_pixels": stride,
        }
        card["invariants"] = ["target and forbidden markers remain fixed"]
        card["falsifiers"] = [
            "actor does not move to the predicted frame coordinate",
            "the step changes a marker instead of moving the actor",
        ]
    elif family == "color_cycle":
        site = {"row": int(first.get("row", 0)), "col": int(first.get("col", 0))}
        card["entities"] = [{"role": "click_site", "space": "frame", **site}]
        card["expected_change"] = {
            "local_region": site,
            "rule": "the clicked swatch advances one palette state toward template constraints",
        }
        card["invariants"] = ["block geometry and template locations remain fixed"]
        card["falsifiers"] = [
            "no local colour change occurs at the click site",
            "the click changes board geometry or an unrelated region",
        ]
    elif family == "sand_filter":
        filters = [dict(item) for item in state.get("filters") or [] if isinstance(item, Mapping)]
        targets = [
            dict(item) for item in state.get("target_filters") or [] if isinstance(item, Mapping)
        ]
        card["coordinate_convention"] = (
            "filter entities use logical_grid row,col; mouse actions use frame row,col"
        )
        card["entities"] = [
            {"role": "filter", "space": "logical_grid", **_rc((f.get("x", 0), f.get("y", 0)))}
            for f in filters[:4]
        ] + [
            {"role": "filter_target", "space": "logical_grid", **_rc((t.get("x", 0), t.get("y", 0)))}
            for t in targets[:4]
        ]
        exact_prediction = _sand_filter_exact_prediction(state, actions, frame)
        card["expected_change"] = {
            "first_action": first,
            "rule": "one routing component changes toward the listed target arrangement",
        }
        if exact_prediction is not None:
            card["expected_change"]["sand_filter_exact"] = exact_prediction
        card["verification_state"] = {
            "family": "sand_filter",
            "flow_potential": int(state.get("operation_plan_len", 0) or 0),
        }
        card["invariants"] = ["sources and bowls remain fixed"]
        card["falsifiers"] = [
            "no routing component changes after the first action",
            "a source or bowl moves or disappears",
        ]
    if family in {"mirror", "stride_maze", "color_cycle"}:
        verification_state = _family_verification_state(family, state, actions)
        card["verification_state"] = verification_state
        exact_prediction = _family_exact_prediction(
            family,
            state,
            actions,
            verification_state,
            card["expected_change"],
        )
        if exact_prediction is not None:
            card["expected_change"][f"{family}_exact"] = exact_prediction
    return card


_MIRROR_MEASURED_STORE: "OrderedDict[str, dict]" = OrderedDict()
_MIRROR_MEASURED_ERRORS = 0
_MIRROR_MEASURED_ERROR_LOCK = threading.Lock()
_MIRROR_MEASURED_MAX = 64
_MIRROR_MEASURED_TOPOLOGIES_MAX = 8
_MIRROR_MEASURED_TRANSITIONS_MAX = 512
_MIRROR_MEASURED_FRAMES_MAX = 512


def _mirror_measured_note_error() -> None:
    """Count a swallowed A4 failure without weakening fail-closed behavior."""
    global _MIRROR_MEASURED_ERRORS
    try:
        with _MIRROR_MEASURED_ERROR_LOCK:
            _MIRROR_MEASURED_ERRORS += 1
    except Exception:
        pass


def _mirror_measured_topology(
    state: Mapping[str, Any],
) -> tuple[Any, ...] | None:
    """Return a stable raw-perception topology; never invent geometry."""
    try:
        origin = tuple(int(v) for v in state.get("origin", ()))
        grid_size = tuple(int(v) for v in state.get("grid_size", ()))
        if len(origin) != 2 or len(grid_size) != 2:
            return None
        walkable = tuple(
            sorted(tuple(int(v) for v in cell) for cell in state["walkable"])
        )
        hazards = tuple(
            sorted(
                tuple(int(v) for v in cell)
                for cell in (state.get("hazards") or state.get("death") or ())
            )
        )
        return (
            origin,
            int(state.get("cell", 0) or 0),
            grid_size,
            int(state.get("cursor_value", -1)),
            int(state.get("walkable_value", -1)),
            walkable,
            hazards,
        )
    except Exception:
        _mirror_measured_note_error()
        return None


def _mirror_measured_snapshot(
    key: str, topology: tuple[Any, ...], bucket: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Copy the bounded payload before the solver consumes it."""
    try:
        pending_records = set(bucket.get("pending_walls") or ())
        confirmed_records = set(bucket.get("confirmed_walls") or ())
        return {
            "key": key,
            "topology": topology,
            "confirmations": int(bucket.get("confirmations", 0) or 0),
            "updates": int(bucket.get("updates", 0) or 0),
            "provable_stays": int(bucket.get("provable_stays", 0) or 0),
            "joint_transition_count": len(
                bucket.get("joint_transitions") or ()
            ),
            "safe_settles": frozenset(bucket.get("safe_settles") or ()),
            "pending_components": frozenset(pending_records),
            "confirmed_components": frozenset(confirmed_records),
            "pending_walls": frozenset(cell for _label, cell in pending_records),
            "confirmed_walls": frozenset(
                cell for _label, cell in confirmed_records
            ),
        }
    except Exception:
        _mirror_measured_note_error()
        return None


def _mirror_measured_walls_update(
    arr: np.ndarray,
    mechanic_context: Mapping[str, Any],
    raw_state: Mapping[str, Any],
    _ps: Any,
) -> dict[str, Any] | None:
    """Learn only uniquely attributable component wall refutations.

    The full previous/current cursor tuples are matched against the mirror
    action matrix.  A wall candidate is recorded only when exactly one
    assignment exists, at least one partner completed its intended motion,
    and a pinned component's perceived-walkable destination is absent.
    Whole-tuple no-ops, resets, teleports, merges, swaps, and geometry drift
    abstain.  Two distinct action-correlated mirror transitions on the same
    raw topology are required before pending walls become consumable.
    """
    try:
        context_frame = mechanic_context.get("frame")
        if context_frame is None or "level" not in mechanic_context:
            return None
        if "score" not in mechanic_context:
            return None
        solved = np.asarray(arr)
        current = np.asarray(context_frame)
        if solved.ndim == 3:
            solved = solved[-1]
        if current.ndim == 3:
            current = current[-1]
        if (
            solved.ndim != 2
            or current.ndim != 2
            or solved.shape != current.shape
            or not np.array_equal(solved, current)
        ):
            return None
        current_state = _ps._mirror_joint_cursor_state(current)
        current_topology = _mirror_measured_topology(current_state or {})
        supplied_topology = _mirror_measured_topology(raw_state)
        if current_state is None or current_topology is None:
            return None
        if supplied_topology != current_topology:
            return None
        if (
            tuple(current_state.get("cursor_a", ()))
            != tuple(raw_state.get("cursor_a", ()))
            or tuple(current_state.get("cursor_b", ()))
            != tuple(raw_state.get("cursor_b", ()))
            or int(current_state.get("cursor_cell_count", 0) or 0) != 2
        ):
            return None
        level = int(mechanic_context["level"])
        score = int(mechanic_context["score"])
        base_key = _stride_measured_key(mechanic_context)
        if base_key is None:
            return None
        key = f"mirror:{base_key}"
        h_current = _stride_frame_hash(current)
        previous_frame = mechanic_context.get("previous_frame")
        h_previous = _stride_frame_hash(previous_frame)
        if h_current is None:
            return None

        with _STRIDE_MEASURED_LOCK:
            store = _MIRROR_MEASURED_STORE.get(key)
            if store is None:
                store = {"topologies": OrderedDict()}
                _MIRROR_MEASURED_STORE[key] = store
                while len(_MIRROR_MEASURED_STORE) > _MIRROR_MEASURED_MAX:
                    _MIRROR_MEASURED_STORE.popitem(last=False)
            _MIRROR_MEASURED_STORE.move_to_end(key)
            topologies = store.setdefault("topologies", OrderedDict())
            bucket = topologies.get(current_topology)
            if bucket is None:
                bucket = {
                    "frame_progress": OrderedDict(),
                    "joint_transitions": OrderedDict(),
                    "safe_settles": set(),
                    "pending_walls": set(),
                    "confirmed_walls": set(),
                    "confirmations": 0,
                    "updates": 0,
                    "provable_stays": 0,
                }
                topologies[current_topology] = bucket
                while len(topologies) > _MIRROR_MEASURED_TOPOLOGIES_MAX:
                    topologies.popitem(last=False)
            topologies.move_to_end(current_topology)
            # A visibly occupied cursor cell is safe-in-fact even when this
            # solve has no attributable transition (fresh-life rejoin,
            # duplicate, or missing previous context).  Retract stale wall
            # records before exposing a snapshot for planning.
            current_cells = {
                tuple(int(v) for v in current_state["cursor_a"]),
                tuple(int(v) for v in current_state["cursor_b"]),
            }
            safe_now = bucket.setdefault("safe_settles", set())
            safe_now.update(current_cells)
            pending_now = bucket.setdefault("pending_walls", set())
            confirmed_now = bucket.setdefault("confirmed_walls", set())
            pending_now.difference_update(
                record for record in set(pending_now)
                if record[1] in current_cells
            )
            confirmed_now.difference_update(
                record for record in set(confirmed_now)
                if record[1] in current_cells
            )
            progress = bucket.setdefault("frame_progress", OrderedDict())
            previous_progress = (
                progress.get(h_previous) if h_previous is not None else None
            )
            progress[h_current] = (level, score)
            progress.move_to_end(h_current)
            while len(progress) > _MIRROR_MEASURED_FRAMES_MAX:
                progress.popitem(last=False)
            initial_snapshot = _mirror_measured_snapshot(
                key, current_topology, bucket
            )
            # PATCH 17 (mirror death ADVICE, sec 78 boundary applied):
            # mirror deaths are MOVEMENT-class and production strips the
            # fatal action, so positional attribution would mark the
            # safe pre-death cell. Record only PAIR-CONTEXT evidence
            # ("a life ended right after the pair stood here") and
            # deliver it as advice; never into forbidden.
            try:
                if os.environ.get(
                    "ECA_TUFA_MIRROR_DEATH_ADVICE", "0"
                ).strip() == "1":
                    hist_pd = list(mechanic_context.get("history") or ())
                    seed_shaped = (
                        len(hist_pd) == 1
                        and str(
                            hist_pd[0].get("action") or ""
                        ).strip() == ""
                        and mechanic_context.get("previous_frame") is None
                    )
                    chain_knows = h_current in _STRIDE_FRAME_TO_EP
                    last_pair = store.get("last_seen_pair")
                    if (
                        seed_shaped
                        and chain_knows
                        and last_pair
                        and last_pair[1] != h_current
                    ):
                        seen_pd = store.setdefault(
                            "pair_death_seen", set()
                        )
                        if last_pair[1] not in seen_pd:
                            seen_pd.add(last_pair[1])
                            votes_pd = store.setdefault(
                                "pair_death_votes", {}
                            )
                            pk = tuple(last_pair[0])
                            votes_pd[pk] = int(
                                votes_pd.get(pk, 0)
                            ) + 1
                            if votes_pd[pk] >= 2:
                                store.setdefault(
                                    "pair_death_confirmed", set()
                                ).add(pk)
                    store["last_seen_pair"] = (
                        tuple(sorted(current_cells)), h_current
                    )
            except Exception:
                _mirror_measured_note_error()

        # Recording the current frame above lets the next observed action
        # prove score continuity.  No transition is learned without a prior
        # solve of exactly the previous frame at the same level and score.
        if previous_frame is None or h_previous is None:
            return initial_snapshot
        if previous_progress != (level, score):
            return initial_snapshot
        previous = np.asarray(previous_frame)
        if previous.ndim == 3:
            previous = previous[-1]
        if previous.ndim != 2 or previous.shape != current.shape:
            return initial_snapshot

        history = list(mechanic_context.get("history") or ())
        if len(history) < 2:
            return initial_snapshot
        before_row, current_row = history[-2], history[-1]
        if not isinstance(before_row, Mapping) or not isinstance(
            current_row, Mapping
        ):
            return initial_snapshot
        if (
            _stride_frame_hash(before_row.get("frame")) != h_previous
            or _stride_frame_hash(current_row.get("frame")) != h_current
        ):
            return initial_snapshot
        if "level" not in before_row or "level" not in current_row:
            return initial_snapshot
        if int(before_row["level"]) != level or int(current_row["level"]) != level:
            return initial_snapshot

        action_raw = str(
            mechanic_context.get("previous_action") or ""
        ).strip().upper()
        action = MODEL_TO_ENGINE_ACTION.get(action_raw, action_raw)
        row_action_raw = str(current_row.get("action") or "").strip().upper()
        row_action = MODEL_TO_ENGINE_ACTION.get(row_action_raw, row_action_raw)
        deltas = {
            "ACTION1": (0, -1),
            "ACTION2": (0, 1),
            "ACTION3": (-1, 0),
            "ACTION4": (1, 0),
        }
        if action not in deltas or row_action != action:
            return initial_snapshot

        previous_state = _ps._mirror_joint_cursor_state(previous)
        if previous_state is None:
            return initial_snapshot
        if int(previous_state.get("cursor_cell_count", 0) or 0) != 2:
            return initial_snapshot
        if _mirror_measured_topology(previous_state) != current_topology:
            return initial_snapshot
        previous_a = tuple(int(v) for v in previous_state["cursor_a"])
        previous_b = tuple(int(v) for v in previous_state["cursor_b"])
        current_a = tuple(int(v) for v in current_state["cursor_a"])
        current_b = tuple(int(v) for v in current_state["cursor_b"])
        if len({previous_a, previous_b}) != 2 or len({current_a, current_b}) != 2:
            return initial_snapshot
        dx, dy = deltas[action]
        intended_a = (previous_a[0] + dx, previous_a[1] + dy)
        intended_b = (previous_b[0] - dx, previous_b[1] + dy)

        # Collision/terminal frames do not identify a wall.  In particular,
        # never turn a failed merge or adjacent swap into a whole-tuple block.
        if intended_a == intended_b:
            return initial_snapshot
        if intended_a == previous_b or intended_b == previous_a:
            return initial_snapshot
        if (
            abs(previous_a[0] - previous_b[0]) == 1
            and previous_a[1] == previous_b[1]
            and (intended_a == previous_b or intended_b == previous_a)
        ):
            return initial_snapshot

        assignments = set()
        for next_a, next_b in (
            (current_a, current_b),
            (current_b, current_a),
        ):
            if next_a in {previous_a, intended_a} and next_b in {
                previous_b,
                intended_b,
            }:
                assignments.add((next_a, next_b))
        if len(assignments) != 1:
            return initial_snapshot
        next_a, next_b = next(iter(assignments))
        moved_a = next_a == intended_a and next_a != previous_a
        moved_b = next_b == intended_b and next_b != previous_b
        stayed_a = next_a == previous_a
        stayed_b = next_b == previous_b
        if not (moved_a or moved_b):
            # The whole tuple stayed: action loss/reset/dual blocking cannot
            # be attributed to either destination, so learn nothing.
            return initial_snapshot
        if not ((moved_a or stayed_a) and (moved_b or stayed_b)):
            return initial_snapshot

        observed = {next_a, next_b}
        perceived_walkable = set(previous_state["walkable"]) & set(
            current_state["walkable"]
        )
        wall_candidates: list[tuple[str, tuple[int, int]]] = []
        if (
            stayed_a
            and intended_a not in observed
            and intended_a in perceived_walkable
        ):
            wall_candidates.append(("a", intended_a))
        if (
            stayed_b
            and intended_b not in observed
            and intended_b in perceived_walkable
        ):
            wall_candidates.append(("b", intended_b))
        if len(wall_candidates) > 1:
            # Defensive whole-tuple-poisoning guard.  With a moving partner
            # this is unreachable, but never encode both components.
            return initial_snapshot

        transition_id = (
            h_previous,
            h_current,
            action,
            previous_a,
            previous_b,
            next_a,
            next_b,
        )
        with _STRIDE_MEASURED_LOCK:
            transitions = bucket.setdefault(
                "joint_transitions", OrderedDict()
            )
            if transition_id in transitions:
                return _mirror_measured_snapshot(key, current_topology, bucket)
            transitions[transition_id] = {
                "action": action,
                "origin": (previous_a, previous_b),
                "intended": (intended_a, intended_b),
                "successor": (next_a, next_b),
            }
            transitions.move_to_end(transition_id)
            while len(transitions) > _MIRROR_MEASURED_TRANSITIONS_MAX:
                transitions.popitem(last=False)

            safe = bucket.setdefault("safe_settles", set())
            safe.update({previous_a, previous_b, next_a, next_b})
            pending = bucket.setdefault("pending_walls", set())
            confirmed = bucket.setdefault("confirmed_walls", set())
            pending.difference_update(
                record for record in set(pending) if record[1] in safe
            )
            confirmed.difference_update(
                record for record in set(confirmed) if record[1] in safe
            )
            for record in wall_candidates:
                if record[1] not in safe:
                    pending.add(record)

            bucket["updates"] = int(bucket.get("updates", 0) or 0) + 1
            bucket["confirmations"] = int(
                bucket.get("confirmations", 0) or 0
            ) + 1
            bucket["provable_stays"] = int(
                bucket.get("provable_stays", 0) or 0
            ) + len(wall_candidates)
            if int(bucket["confirmations"]) >= 2:
                confirmed.update(
                    record for record in pending if record[1] not in safe
                )
                pending.clear()
            return _mirror_measured_snapshot(key, current_topology, bucket)
    except Exception:
        _mirror_measured_note_error()
        return None


def _solve_mirror_legacy(
    arr: np.ndarray, engine_valid: set[str],
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from arcgym.eca import primitive_synthesis as _ps
    from arcgym.eca.mechanic_certificate import MirrorCertificate

    moves = engine_valid & _MOVE_ACTIONS
    if not moves:
        return _obs("mirror", "unavailable", "movement actions not in valid set"), None
    try:
        mirror_measured_on = os.environ.get(
            "ECA_TUFA_MIRROR_MEASURED_WALLS", "0"
        ).strip() == "1"
    except Exception:
        _mirror_measured_note_error()
        mirror_measured_on = False
    state = _ps._mirror_joint_cursor_state(arr)
    if state is None:
        return _obs("mirror", "absent", "no mirror cursor-pair structure in frame"), None
    certified, reason = _verdict_fields(MirrorCertificate().evaluate(state))
    try:
        mirror_death_advice_on = os.environ.get(
            "ECA_TUFA_MIRROR_DEATH_ADVICE", "0"
        ).strip() == "1"
    except Exception:
        _mirror_measured_note_error()
        mirror_death_advice_on = False
    measured = None
    if (mirror_measured_on or mirror_death_advice_on) and mechanic_context:
        try:
            measured = _mirror_measured_walls_update(
                arr, mechanic_context, state, _ps
            )
        except Exception:
            _mirror_measured_note_error()
            measured = None
    plan_state = state
    if mirror_measured_on and certified and measured:
        try:
            measured_walls = set(measured.get("confirmed_walls") or ())
            measured_walls.difference_update(
                {tuple(state["cursor_a"]), tuple(state["cursor_b"])}
            )
            effective_walkable = frozenset(
                set(state["walkable"]) - measured_walls
            )
            if effective_walkable != state["walkable"]:
                plan_state = dict(state)
                plan_state["walkable"] = effective_walkable
        except Exception:
            _mirror_measured_note_error()
            plan_state = state
    hazards = plan_state.get("hazards") or plan_state.get("death") or frozenset()
    plan = _ps._mirror_joint_cursor_bfs_plan(
        plan_state["cursor_a"],
        plan_state["cursor_b"],
        plan_state["walkable"],
        moves,
        hazards=frozenset(hazards),
    )
    status = "certified" if certified else "detected"
    observation = _obs(
        "mirror",
        status,
        reason,
        walkable_count=len(state.get("walkable") or ()),
        hazard_count=len(hazards),
        plan_found=bool(plan),
        plan_len=len(plan or ()),
    )
    if not plan:
        observation["reason"] = reason + "; no collision path found (BFS exhausted)"
        return observation, None
    actions = _to_model_steps(plan)
    if actions is None:
        return observation, None
    return observation, {
        "family": "mirror",
        "certified": certified,
        "certificate_reason": reason,
        "actions": actions,
        "grounded_observation": _grounded_observation(
            "mirror", plan_state, actions
        ),
        "detail": {
            "cursor_a": [int(v) for v in state["cursor_a"]],
            "cursor_b": [int(v) for v in state["cursor_b"]],
            "hazard_count": len(hazards),
            "verification_binding": {
                "origin": [int(v) for v in state.get("origin", (0, 0))],
                "cell": int(state.get("cell", 1) or 1),
                "grid_size": [int(v) for v in state.get("grid_size", (0, 0))],
                "cursor_value": int(state.get("cursor_value", -1)),
                "walkable_value": int(state.get("walkable_value", -1)),
            },
        },
    }


_CC_MEASURED_ERRORS = 0


def _cc_measured_note_error() -> None:
    global _CC_MEASURED_ERRORS
    try:
        _CC_MEASURED_ERRORS += 1
    except Exception:
        pass


def _cc_parse_click(action: str) -> tuple[int, int] | None:
    """Parse the production click display "MOUSE(row=R, col=C)"."""
    try:
        raw = str(action or "").strip()
        if not raw.upper().startswith("MOUSE"):
            return None
        inner = raw[raw.index("(") + 1:raw.rindex(")")]
        parts = dict(
            item.strip().split("=", 1) for item in inner.split(",")
        )
        return (int(parts["col"]), int(parts["row"]))
    except Exception:
        return None


def _cc_measured_update(
    mechanic_context: Mapping[str, Any],
) -> dict | None:
    """B-passive observer (sec 80): feed the click-effect learner from
    the LLM's OWN executed clicks, delivered through the same context
    channel the A-series consumes. Observations dedup by transition
    identity (the sliding window re-presents each turn on every solve,
    and a double-counted click would fake the two-observation stencil
    evidence). Fail-closed."""
    try:
        base_key = _stride_measured_key(mechanic_context)
        if base_key is None:
            return None
        key = f"cc:{base_key}"
        with _STRIDE_MEASURED_LOCK:
            store = _MIRROR_MEASURED_STORE.get(key)
            if store is None:
                store = {"seen": set()}
                _MIRROR_MEASURED_STORE[key] = store
                while len(_MIRROR_MEASURED_STORE) > _MIRROR_MEASURED_MAX:
                    _MIRROR_MEASURED_STORE.popitem(last=False)
            _MIRROR_MEASURED_STORE.move_to_end(key)
        hist = list(mechanic_context.get("history") or ())
        if len(hist) == 1 and str(
            hist[0].get("action") or ""
        ).strip() == "":
            # fresh-life seed: the click budget refilled -- reset the
            # advertised-plan-click counter (sec 83 strike guard)
            with _STRIDE_MEASURED_LOCK:
                store["plan_clicks_life"] = 0
                store["life_goals"] = {}
        if len(hist) < 2:
            return store
        click = _cc_parse_click(hist[-1].get("action"))
        if click is None:
            return store
        h_prev = _stride_frame_hash(hist[-2].get("frame"))
        h_cur = _stride_frame_hash(hist[-1].get("frame"))
        if h_prev is None or h_cur is None or h_prev == h_cur:
            return store
        tid = (h_prev, h_cur)
        with _STRIDE_MEASURED_LOCK:
            if tid in store["seen"]:
                return store
            store["seen"].add(tid)
        try:
            from arcgym.eca.effect_algebra import ClickEffectLearner
        except Exception:
            _cc_measured_note_error()
            return store
        with _STRIDE_MEASURED_LOCK:
            learner = store.get("learner")
            if learner is None:
                learner = ClickEffectLearner()
                store["learner"] = learner
        before = np.asarray(hist[-2].get("frame"))
        after = np.asarray(hist[-1].get("frame"))
        if before.ndim == 3:
            before = before[-1]
        if after.ndim == 3:
            after = after[-1]
        with _STRIDE_MEASURED_LOCK:
            learner.observe(before, after, click[0], click[1])
        return store
    except Exception:
        _cc_measured_note_error()
        return None


def _cc_state_from_measurement(
    arr: np.ndarray, store: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Synthesize a color_cycle constraint state from PASSIVELY learned
    click algebra when the frame perception is absent. The passive port
    of the proven executive's solve stage (template_goal_executive):
    confirmed targets -> cycle -> TemplateGoalInducer -> goal variants
    -> cheapest solvable program. Requires real example votes (no
    fallback-only goals) and a palette-bounded cycle so the EXISTING
    ColorCycleCertificate stays the final gate. Fail-closed."""
    try:
        if not store:
            return None
        learner = store.get("learner")
        if learner is None:
            return None
        try:
            from arcgym.eca.template_goal_induction import (
                TemplateGoalInducer,
            )
        except Exception:
            _cc_measured_note_error()
            return None
        board = np.asarray(arr)
        if board.ndim == 3:
            board = board[-1]
        if board.ndim != 2:
            return None
        with _STRIDE_MEASURED_LOCK:
            # the executive gate the first draft dropped: a genuine
            # cycle family must be LINEAR-certified (clicks follow a
            # consistent value algebra). Non-cycle click games (piece
            # moves / selections) produce stencil-shaped junk that
            # passed everything else (measured: r11l fired 6 false
            # synthesized certifications without this gate).
            if not learner.linear_certified():
                return None
            stencils = learner.confirmed_targets()
            targets_now = sorted(stencils)
            if len(targets_now) < 2:
                return None
            cycle = learner.cycle() or learner.value_cycle_over_cells(
                targets_now
            )
        if not cycle or not (2 <= len(cycle) <= 4):
            return None
        inducer = TemplateGoalInducer(board, targets_now, list(cycle))
        if not inducer.ok:
            return None
        # arrangement regularity gate: cycle-template puzzles keep
        # their clickable swatches on a dense regular lattice (the
        # perception itself samples a grid). Arbitrary click spots
        # that happen to toggle values (selection highlights are a
        # genuine 2-cycle -- measured on r11l) form SPARSE irregular
        # arrangements and are rejected here, not by the algebra.
        try:
            fill = float(len(inducer.offsets)) / float(
                max(1, int(inducer.rows) * int(inducer.cols))
            )
        except Exception:
            return None
        if len(targets_now) < 4 or fill < 0.6:
            return None
        # sec 82: learned-anchored template reader -- the goal pattern
        # is usually SHOWN somewhere on the board as a scaled copy of
        # the learned arrangement (the ft09 template icon). Geometry
        # comes from LEARNING (the confirmed-site offsets), only the
        # values are read from raw pixels: scan every anchor and
        # spacing for an 8-cell group matching the arrangement shape
        # whose values are palette members with >=2 distinct values,
        # excluding the learned sites themselves. Unique-region
        # patterns rank first (a template appears once).
        reader_goals = []
        try:
            step_x = max(1, int(getattr(inducer, "step_x", 1)))
            step_y = max(1, int(getattr(inducer, "step_y", 1)))
            norm = sorted(
                (int(o[0]) // step_x, int(o[1]) // step_y)
                for o in inducer.offsets
            )
            site_cells = {tuple(t) for t in targets_now}
            # exclude every MUTABLE cell (the union of learned click
            # stencils): a candidate read from the click-mutated area
            # chases its own tail -- executing its plan changes the
            # very region it was read from, so it vanishes from the
            # next scan and the achieved-without-score rejection never
            # fires (measured: rejected stayed 0 while 1-click plans
            # churned). The template must live in STATIC board area.
            mutable = set(site_cells)
            try:
                for _stencil in stencils.values():
                    mutable.update(
                        (int(cx), int(cy)) for cx, cy in _stencil
                    )
            except Exception:
                pass
            h, w = board.shape
            max_c = max(c for c, _r in norm)
            max_r = max(r for _c, r in norm)
            # the template may be drawn in OTHER colours (e.g. one
            # palette colour against a neutral) and FLIPPED relative
            # to the click arrangement (measured on ft09: spacing 5,
            # colours {5,8}/{5,9}, fx/r180) -- so match the SHAPE as
            # a binary pattern under all 8 D4 transforms and map each
            # of the two binary assignments onto the two-value cycle.
            d4 = (
                lambda c, r: (c, r),
                lambda c, r: (r, c),
                lambda c, r: (max_c - c, r),
                lambda c, r: (c, max_r - r),
                lambda c, r: (max_c - c, max_r - r),
                lambda c, r: (max_r - r, c),
                lambda c, r: (r, max_c - c),
                lambda c, r: (max_r - r, max_c - c),
            )
            norm_set = set(norm)
            bins: dict = {}
            for t in range(1, 9):
                for y0 in range(0, h - max_r * t):
                    for x0 in range(0, w - max_c * t):
                        cells = [
                            (x0 + c * t, y0 + r * t) for c, r in norm
                        ]
                        if any(cell in mutable for cell in cells):
                            continue
                        vals = [int(board[cy, cx]) for cx, cy in cells]
                        if len(set(vals)) != 2:
                            continue
                        v0 = sorted(set(vals))[0]
                        key = frozenset(
                            norm[i] for i in range(len(norm))
                            if vals[i] == v0
                        )
                        bins[key] = bins.get(key, 0) + 1
            cyc2 = sorted(int(v) for v in cycle)[:2]
            goals_seen = set()
            for key, _count in sorted(
                bins.items(), key=lambda kv: kv[1]
            ):
                for tf in d4:
                    mapped = {tf(c, r) for (c, r) in key}
                    if not mapped <= norm_set:
                        continue
                    for lo, hi in ((cyc2[0], cyc2[-1]),
                                   (cyc2[-1], cyc2[0])):
                        goal = {
                            (c * step_x, r * step_y):
                            (lo if (c, r) in mapped else hi)
                            for (c, r) in norm
                        }
                        gid = tuple(sorted(goal.items()))
                        if gid in goals_seen:
                            continue
                        goals_seen.add(gid)
                        reader_goals.append(goal)
                        if len(reader_goals) >= 24:
                            break
                    if len(reader_goals) >= 24:
                        break
                if len(reader_goals) >= 24:
                    break
        except Exception:
            _cc_measured_note_error()
            reader_goals = []
        votes = inducer.example_votes()
        if not votes and not reader_goals:
            return None
        fallback = dict(votes.most_common()[0][0]) if votes else {}
        # passive replacement for the executive ACTIVE free-cell
        # enumeration: goals may leave <=3 offsets undetermined, so
        # expand every variant into its bounded completions and let
        # each completion be solved/rejected individually.
        from itertools import product as _product
        candidates = list(reader_goals)
        for required in (
            inducer.goal_variants(fallback) if votes else []
        ):
            base = dict(required)
            free = [
                off for off in inducer.offsets if off not in base
            ]
            if len(free) > 3:
                candidates.append(base)
                continue
            if not free:
                candidates.append(base)
                continue
            for combo in _product(list(cycle), repeat=len(free)):
                filled = dict(base)
                for off, val in zip(free, combo):
                    filled[off] = int(val)
                candidates.append(filled)
            if len(candidates) > 64:
                break
        scored = []
        for required in candidates[:64]:
            absolute = inducer.absolute(required)
            if not absolute:
                continue
            # achieved-without-score refutation (sec 80 / B design):
            # a variant the CURRENT board already satisfies cannot be
            # the level goal (we would have scored), so reject it
            # PERMANENTLY (store-persisted: a fresh life resets the
            # board but must not resurrect refuted goals) -- this is
            # what lets successive solves rotate through variants.
            try:
                goal_id = tuple(sorted(
                    ((int(x), int(y)), int(v))
                    for (x, y), v in absolute.items()
                ))
                with _STRIDE_MEASURED_LOCK:
                    rejected = store.setdefault("rejected_goals", set())                         if isinstance(store, dict) else set()
                    if goal_id in rejected:
                        continue
                if all(
                    int(board[int(y), int(x)]) == int(v)
                    for (x, y), v in absolute.items()
                ):
                    with _STRIDE_MEASURED_LOCK:
                        if isinstance(store, dict):
                            store.setdefault(
                                "rejected_goals", set()
                            ).add(goal_id)
                    continue
            except Exception:
                continue
            with _STRIDE_MEASURED_LOCK:
                program = learner.solve(
                    inducer.refreshed(board),
                    {
                        "kind": "pattern",
                        "cells": sorted(absolute),
                        "values": absolute,
                    },
                )
            if program:
                scored.append((len(program), program, absolute))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0])
        _plen, program, absolute = scored[0]
        # sec 83 strike guard: production suppresses a family for the
        # level after ONE advertised-plan-followed-by-game-over
        # (SUPPRESSION_STRIKES=1). Never advertise more plan clicks in
        # a life than a conservative cap -- the budget is unknown a
        # priori, and a silent toolbox costs nothing (the LLM keeps
        # exploring) while an advertised plan running into a budget
        # death kills the family for the level.
        try:
            cap = int(os.environ.get(
                "ECA_TUFA_CC_PLAN_CLICK_CAP", "12"
            ) or 12)
        except Exception:
            cap = 12
        chosen_id = tuple(sorted(
            ((int(x), int(y)), int(v))
            for (x, y), v in absolute.items()
        ))
        with _STRIDE_MEASURED_LOCK:
            life_goals = (
                store.setdefault("life_goals", {})
                if isinstance(store, dict) else {}
            )
            used = sum(int(v) for v in life_goals.values())
            already = chosen_id in life_goals
        # the harness re-solves every turn, so the SAME goal is
        # re-emitted many times per advertisement -- only the FIRST
        # emission of each distinct goal spends cap budget.
        if not already and used + len(program) > cap:
            return None
        with _STRIDE_MEASURED_LOCK:
            if isinstance(store, dict) and not already:
                life_goals[chosen_id] = len(program)
        plan = [
            ("ACTION6", {"x": int(c["x"]), "y": int(c["y"])})
            for c in program
        ]
        solid = [
            {"pos": (int(x), int(y)),
             "value": int(board[int(y), int(x)])}
            for x, y in targets_now
        ]
        constraints = [
            {"cell": [int(x), int(y)], "value": int(v)}
            for (x, y), v in sorted(absolute.items())
        ]
        return {
            "mechanic_family": "cycle_state_constraint",
            "mechanic_state_type": "color_cycle_template",
            "goal_link_type": "template_color_match",
            "required_primitive_family": "cycle_operation_solver",
            "palette": [int(v) for v in cycle],
            "templates": [
                {"goal": sorted(
                    (list(k), int(v)) for k, v in absolute.items()
                )}
            ],
            "solid": solid,
            "brush": [],
            "plan": list(plan),
            "operation_plan": list(plan),
            "cycle_operation_kind": "palette_cell_click",
            "cycle_operation_count": len(plan),
            "cycle_operation_confidence_threshold": 0.55,
            "constraints": constraints,
            "constraint_count": len(constraints),
            "clickable_count": len(solid),
            "template_count": 1,
            "expected_goal_delta": float(max(1, len(constraints))),
            "reader_confidence": 0.6,
            "goal_link_confidence": 0.6,
            "measured_synthesis": True,
        }
    except Exception:
        _cc_measured_note_error()
        return None


def _solve_color_cycle(
    arr: np.ndarray,
    engine_valid: set[str],
    role_cache: dict[str, Any] | None,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from arcgym.eca import primitive_synthesis as _ps
    from arcgym.eca.mechanic_certificate import ColorCycleCertificate

    if "ACTION6" not in engine_valid:
        return _obs("color_cycle", "unavailable", "click action not in valid set"), None
    cc_measured = None
    if mechanic_context is not None and os.environ.get(
        "ECA_TUFA_COLOR_CYCLE_MEASURED", "0"
    ).strip() == "1":
        cc_measured = _cc_measured_update(mechanic_context)
    state = _ps._color_cycle_constraint_state(
        arr,
        role_cache,
        force_palette_robust=True,
    )
    if state is None and cc_measured is not None:
        state = _cc_state_from_measurement(arr, cc_measured)
    if state is None:
        return _obs("color_cycle", "absent", "no colour-cycle constraint grid in frame"), None
    certified, reason = _verdict_fields(ColorCycleCertificate().evaluate(state))
    plan = state.get("plan")
    observation = _obs(
        "color_cycle",
        "certified" if certified else "detected",
        reason,
        constraint_count=int(state.get("constraint_count", 0) or 0),
        clickable_count=int(state.get("clickable_count", 0) or 0),
        plan_found=bool(plan),
        plan_len=len(plan or ()),
    )
    if not plan:
        observation["reason"] = reason + "; constraint system infeasible (no click plan)"
        return observation, None
    actions = _to_model_steps(plan)
    if actions is None:
        return observation, None
    return observation, {
        "family": "color_cycle",
        "certified": certified,
        "certificate_reason": reason,
        "actions": actions,
        "grounded_observation": _grounded_observation("color_cycle", state, actions),
        "detail": {
            "constraint_count": int(state.get("constraint_count", 0) or 0),
            "clickable_count": int(state.get("clickable_count", 0) or 0),
        },
    }


_STRIDE_MEASURED_STORE: "OrderedDict[str, dict]" = OrderedDict()
_STRIDE_MEASURED_LOCK = threading.Lock()
_STRIDE_MEASURED_MAX = 64


_STRIDE_FRAME_TO_EP: "OrderedDict[str, str]" = OrderedDict()
_STRIDE_FRAME_TO_EP_MAX = 512


def _stride_frame_hash(frame: Any) -> str | None:
    try:
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.ndim == 3:
            arr = arr[-1]
        if arr.ndim != 2 or arr.size == 0:
            return None
        return hashlib.sha1(arr.tobytes()).hexdigest()[:16]
    except Exception:
        return None


def _stride_measured_key(mechanic_context: Mapping[str, Any]) -> str | None:
    """Episode identity via FRAME-HASH CHAINING: the harness passes a
    sliding 64-entry history, so any fixed history index changes as
    the window slides (measured live: the store shattered into 51
    keys past 64 steps). Every observed frame hash maps to a stable
    episode id instead; the previous frame links the chain, and a
    reset re-joins through the recorded initial frame."""
    try:
        h_cur = _stride_frame_hash(mechanic_context.get("frame"))
        if h_cur is None:
            return None
        h_prev = _stride_frame_hash(mechanic_context.get("previous_frame"))
        with _STRIDE_MEASURED_LOCK:
            # CURRENT frame first: a reset lands back on a frame the
            # chain already knows, and that identity must never be
            # overridden by whatever episode the previous life's last
            # frame drifted into (measured live: the reset frame was
            # captured by a 3-cell fork and synthesis went silent).
            epid = _STRIDE_FRAME_TO_EP.get(h_cur)
            if epid is None and h_prev is not None:
                epid = _STRIDE_FRAME_TO_EP.get(h_prev)
            if epid is None:
                # multi-action gaps (a lease executing its plan) leave
                # frames the chain never saw; re-join through the most
                # recent known frame in the harness history instead of
                # forking a new episode.
                hist = mechanic_context.get("history") or ()
                for entry in reversed(list(hist)[-64:]):
                    h_e = _stride_frame_hash(
                        entry.get("frame")
                        if isinstance(entry, Mapping)
                        else None
                    )
                    if h_e is not None and h_e in _STRIDE_FRAME_TO_EP:
                        epid = _STRIDE_FRAME_TO_EP[h_e]
                        break
            if epid is None:
                epid = h_cur
            if h_prev is not None and h_prev not in _STRIDE_FRAME_TO_EP:
                _STRIDE_FRAME_TO_EP[h_prev] = epid
            _STRIDE_FRAME_TO_EP[h_cur] = epid
            while len(_STRIDE_FRAME_TO_EP) > _STRIDE_FRAME_TO_EP_MAX:
                _STRIDE_FRAME_TO_EP.popitem(last=False)
        # LEVEL scoping: frame-hash chaining links straight through a
        # level transition (the transition frame's previous frame is
        # known), so without this suffix the L1 measurement store
        # would leak into L2+ synthesis. The harness threads `level`
        # in every mechanic_context.
        lvl = mechanic_context.get("level")
        if lvl is not None:
            return f"{epid}:L{lvl}"
        return epid
    except Exception:
        return None




def _stride_death_scan(
    mechanic_context: Mapping[str, Any], store: dict
) -> None:
    """A3 v2 (sec 77 DEFECT A): PRODUCTION-shape death detection.

    The harness epoch slicer delivers only current-level entries,
    restarts the epoch at any RESET, and erases the boundary action --
    so neither a level regression nor the pre-death transition is ever
    visible in history. What production DOES deliver after a death is
    a single-entry epoch seed whose frame the episode chain has seen
    before while the store already holds later progress. Detect that,
    and attribute the death to the LAST OBSERVED actor position (the
    settle-then-die class that owns the real tu93 evidence). Voluntary
    resets scatter votes across positions; genuine lethal settles
    recur -- the two-vote threshold separates them, and votes dedup by
    the pre-death frame hash so one death is one vote. Fail-open."""
    try:
        hist = list(mechanic_context.get("history") or ())
        if len(hist) != 1:
            return
        if str(hist[0].get("action") or "").strip() != "":
            return
        if mechanic_context.get("previous_frame") is not None:
            return
        h_cur = _stride_frame_hash(mechanic_context.get("frame"))
        if h_cur is None:
            return
        with _STRIDE_MEASURED_LOCK:
            chain_knows = h_cur in _STRIDE_FRAME_TO_EP
            last = store.get("last_seen")
        if not chain_knows or not last:
            # first-ever visit to this frame (fresh level start) or no
            # recorded progress: no death evidence.
            return
        last_pos, last_hash = last
        if last_hash == h_cur:
            return
        dst = (int(last_pos[0]), int(last_pos[1]))
        with _STRIDE_MEASURED_LOCK:
            seen = store.setdefault("death_seen", set())
            if last_hash in seen:
                return
            seen.add(last_hash)
            votes = store.setdefault("death_votes", {})
            votes[dst] = int(votes.get(dst, 0)) + 1
            if votes[dst] >= 2:
                conf = store.setdefault("death_forbidden", set())
                conf.add(dst)
                # two recurrences outrank walkable evidence:
                # settle-then-die cells (tu93 locks) carry safe
                # settles legitimately, so the R28 rule applies
                # only to the one-vote stage (by never forbidding
                # there at all).
                store.get("walkable", set()).discard(dst)
    except Exception:
        pass




def _stride_measured_hazards_view(
    frame: Any, level: Any
) -> dict[str, Any] | None:
    """Read-only advice view of the stride lane's measured store for
    the current (episode, level): twice-confirmed lethal cells only.
    Never writes the chain map. Least-privilege: coordinates only."""
    try:
        h = _stride_frame_hash(frame)
        if h is None:
            return None
        with _STRIDE_MEASURED_LOCK:
            epid = _STRIDE_FRAME_TO_EP.get(h)
            if epid is None:
                return None
            key = (
                f"{epid}:L{int(level)}" if level is not None else epid
            )
            store = _STRIDE_MEASURED_STORE.get(key)
            if not store:
                return None
            lethal = sorted(store.get("death_forbidden") or ())
        if not lethal:
            return None
        return {
            "lethal": [(int(x), int(y)) for x, y in lethal],
        }
    except Exception:
        return None


def _mirror_death_view(
    frame: Any, level: Any
) -> list | None:
    """Read-only advice view of twice-confirmed mirror pair-death
    contexts for the current (episode, level). Coordinates only."""
    try:
        h = _stride_frame_hash(frame)
        if h is None:
            return None
        with _STRIDE_MEASURED_LOCK:
            epid = _STRIDE_FRAME_TO_EP.get(h)
            if epid is None:
                return None
            key = (
                f"mirror:{epid}:L{int(level)}"
                if level is not None
                else f"mirror:{epid}"
            )
            store = _MIRROR_MEASURED_STORE.get(key)
            if not store:
                return None
            confirmed = sorted(store.get("pair_death_confirmed") or ())
        if not confirmed:
            return None
        return [
            [[int(x), int(y)] for x, y in pair] for pair in confirmed
        ]
    except Exception:
        _mirror_measured_note_error()
        return None


def _stride_measured_walkable_update(
    mechanic_context: Mapping[str, Any],
) -> dict | None:
    """Accumulate MEASURED walkable cells for this episode-level from
    the latest observed transition (same diff channel as the actor
    supplier): a successful one-stride move marks the source node,
    the lead edge cell, and the destination node walkable-IN-FACT --
    repairing single-color walkable hypotheses (two-color floors)
    with evidence instead of guesses. Store keyed by the epoch's
    first frame (level-scoped, 28-parallel safe). Fail-closed."""
    key = _stride_measured_key(mechanic_context)
    if key is None:
        return None
    with _STRIDE_MEASURED_LOCK:
        store = _STRIDE_MEASURED_STORE.get(key)
        if store is None:
            store = {"walkable": set()}
            _STRIDE_MEASURED_STORE[key] = store
            while len(_STRIDE_MEASURED_STORE) > _STRIDE_MEASURED_MAX:
                _STRIDE_MEASURED_STORE.popitem(last=False)
        _STRIDE_MEASURED_STORE.move_to_end(key)
    try:
        if os.environ.get(
            "ECA_TUFA_STRIDE_DEATH_FORBIDDEN", "0"
        ).strip() == "1":
            _stride_death_scan(mechanic_context, store)
        # NEGATIVE learning: an executed move that left the frame
        # bit-identical is direct evidence that this direction does
        # not fire from the current cell. Without it a synthesized
        # plan whose final hop is wrong re-fires the same dead move
        # forever (measured live: ACTION4 x44 against a wall).
        h_cur = _stride_frame_hash(mechanic_context.get("frame"))
        h_prev = _stride_frame_hash(mechanic_context.get("previous_frame"))
        prev_action = str(
            mechanic_context.get("previous_action") or ""
        ).upper()
        prev_engine = MODEL_TO_ENGINE_ACTION.get(prev_action, prev_action)
        deltas_neg = {
            "ACTION1": (0, -6),
            "ACTION2": (0, 6),
            "ACTION3": (-6, 0),
            "ACTION4": (6, 0),
        }
        if prev_engine in deltas_neg and mechanic_context.get(
            "previous_frame"
        ) is not None:
            # frame hashes cannot detect a dead move here: a step
            # counter repaints every action. Compare the MARKER actor
            # position across the transition instead (same vocabulary
            # the synthesis anchors on, counter-immune).
            try:
                from arcgym.eca import primitive_synthesis as _psn
                arr_now = np.asarray(mechanic_context.get("frame"))
                arr_prev = np.asarray(mechanic_context.get("previous_frame"))
                m_now = _psn._marker_blocks(arr_now, block_size=3)
                m_prev = _psn._marker_blocks(arr_prev, block_size=3)
                a_now = (
                    _psn._stride_marker_actor(m_now, actor_hint=None, block_size=3)
                    if m_now
                    else None
                )
                a_prev = (
                    _psn._stride_marker_actor(m_prev, actor_hint=None, block_size=3)
                    if m_prev
                    else None
                )
                if a_now is not None:
                    with _STRIDE_MEASURED_LOCK:
                        store["last_seen"] = (
                            (int(a_now["x"]), int(a_now["y"])),
                            h_cur,
                        )
                if (
                    a_now is not None
                    and a_prev is not None
                    and int(a_now["x"]) == int(a_prev["x"])
                    and int(a_now["y"]) == int(a_prev["y"])
                ):
                    a_pos = (int(a_now["x"]), int(a_now["y"]))
                    with _STRIDE_MEASURED_LOCK:
                        store.setdefault("refuted", set()).add(
                            (a_pos, deltas_neg[prev_engine])
                        )
            except Exception:
                pass
        hint = _stride_actor_hint_from_observation(mechanic_context)
        if hint is None:
            return store
        history = mechanic_context.get("history") or []
        action_raw = str(history[-1].get("action") or "").strip().upper()
        engine = MODEL_TO_ENGINE_ACTION.get(action_raw, action_raw)
        deltas = {
            "ACTION1": (0, -6),
            "ACTION2": (0, 6),
            "ACTION3": (-6, 0),
            "ACTION4": (6, 0),
        }
        if engine not in deltas:
            return store
        ddx, ddy = deltas[engine]
        dst_c = (int(hint[0]), int(hint[1]))
        src_c = (dst_c[0] - ddx, dst_c[1] - ddy)
        mid_c = ((src_c[0] + dst_c[0]) // 2, (src_c[1] + dst_c[1]) // 2)
        src_tl = (src_c[0] - 1, src_c[1] - 1)
        dst_tl = (dst_c[0] - 1, dst_c[1] - 1)
        with _STRIDE_MEASURED_LOCK:
            for cx, cy in (src_c, mid_c, dst_c):
                store["walkable"].add((cx - 1, cy - 1))
            store.setdefault("edges", set()).add((src_tl, dst_tl))
        return store
    except Exception:
        return store


def _stride_state_from_measurement(
    arr: np.ndarray,
    actor_hint: tuple[int, int] | None,
    measured: dict | None,
    _ps: Any,
) -> dict[str, Any] | None:
    """Synthesize a stride state from MEASUREMENT when the frame
    perception fails entirely (e.g. its single-color walkable
    hypothesis collapses): actor from the observation hint, walkable
    from measured cells, targets from the perception's own
    rare-color block rule. Returns a state only when a real plan
    exists (the perception's own solvability contract); the caller's
    MechanicCertificate stays the final gate. Fail-closed."""
    try:
        if not measured:
            return None
        walk = set(measured.get("walkable") or ())
        if len(walk) < 3:
            return None
        markers = _ps._marker_blocks(arr, block_size=3)
        pure = _ps._pure_blocks(arr, block_size=3)
        if not pure:
            return None
        from collections import Counter as _Counter
        value_counts = _Counter(int(v) for v in np.asarray(arr).flatten().tolist())
        actor = (
            _ps._stride_marker_actor(markers, actor_hint=actor_hint, block_size=3)
            if markers
            else None
        )
        if actor is not None:
            actor_pos = (int(actor["x"]), int(actor["y"]))
            actor_values = {
                int(actor["primary_value"]),
                int(actor["secondary_value"]),
            }
        elif actor_hint is not None:
            actor_pos = (int(actor_hint[0]) - 1, int(actor_hint[1]) - 1)
            actor_values = set()
        else:
            # no observation hint (e.g. first frame of a new life) and
            # no marker-derived actor: nothing to anchor the lattice on.
            return None
        # GRID SNAPPING: measured coordinates inherit the supplier's
        # 1-2px window offset; snap onto the actor-anchored lattice.
        def _snap(c):
            return (
                actor_pos[0] + int(round((c[0] - actor_pos[0]) / 6.0)) * 6,
                actor_pos[1] + int(round((c[1] - actor_pos[1]) / 6.0)) * 6,
            )
        edges_raw = set(measured.get("edges") or ())
        refuted = {
            (_snap(p), dd) for p, dd in (measured.get("refuted") or ())
        }
        # measured EDGE graph: each recorded transition is direct
        # evidence that the move src->dst executes; symmetric mazes
        # justify the reverse direction too (a wrong reverse costs
        # one refused step at execution, never a false certificate
        # by itself -- the certificate still gates the whole state).
        adj: dict = {}
        cells: set = {actor_pos}
        for s_raw, d_raw in edges_raw:
            s, d = _snap(s_raw), _snap(d_raw)
            if s == d:
                continue
            adj.setdefault(s, set()).add(d)
            adj.setdefault(d, set()).add(s)
            # the caller re-plans with the standard lattice BFS, which
            # requires the mid (lead-edge) cell of every hop -- each
            # measured edge certifies exactly its own mid cell.
            cells.add(s)
            cells.add(d)
            cells.add(((s[0] + d[0]) // 2, (s[1] + d[1]) // 2))
        targets = [
            (int(b["x"]), int(b["y"]))
            for b in pure
            if int(b["value"]) not in actor_values
            and value_counts[int(b["value"])] <= 36
        ]
        if not targets:
            return None
        forbidden = (
            _ps._stride_marker_forbidden_cells(markers, actor=actor, stride=6)
            if actor is not None
            else frozenset()
        )
        death_conf = {
            _snap(c) for c in (measured.get("death_forbidden") or ())
        }
        if death_conf:
            forbidden = frozenset(set(forbidden) | death_conf)
        reroute_on = os.environ.get(
            "ECA_TUFA_STRIDE_DEATH_REROUTE", "0"
        ).strip() == "1"
        name_of = {(0, -6): "ACTION1", (0, 6): "ACTION2",
                   (-6, 0): "ACTION3", (6, 0): "ACTION4"}

        def _edge_bfs(start, goal):
            from collections import deque as _dq
            if start == goal:
                return None
            seen = {start}
            q = _dq([(start, [])])
            while q:
                cur, pth = q.popleft()
                neighbours = set(adj.get(cur, ()))
                # final-hop exemption: stepping ONTO the rare-color
                # target needs no recorded edge (it only becomes
                # observable by clearing the level) -- adjacency on
                # the lattice suffices; everything else must ride
                # measured edges.
                for dd, nm in name_of.items():
                    cand = (cur[0] + dd[0], cur[1] + dd[1])
                    if cand == goal:
                        neighbours.add(cand)
                for nxt in neighbours:
                    if reroute_on and nxt in death_conf:
                        continue
                    if nxt in seen:
                        continue
                    dd = (nxt[0] - cur[0], nxt[1] - cur[1])
                    if dd not in name_of:
                        continue
                    if (cur, dd) in refuted:
                        continue
                    seen.add(nxt)
                    npth = pth + [name_of[dd]]
                    if nxt == goal:
                        return npth
                    q.append((nxt, npth))
            return None

        for target in targets:
            snapped_target = _snap(tuple(target))
            for node in list(adj.keys()) + [actor_pos]:
                dd = (snapped_target[0] - node[0], snapped_target[1] - node[1])
                if dd in name_of and (node, dd) not in refuted:
                    cells.add(
                        ((node[0] + snapped_target[0]) // 2,
                         (node[1] + snapped_target[1]) // 2)
                    )
                    cells.add(snapped_target)
            path = _edge_bfs(actor_pos, snapped_target)
            if reroute_on:
                reroute_cells = set(cells)
            if path and frozenset(forbidden):
                cells = []
                cx, cy = actor_pos
                ok = True
                for nm in path:
                    dd = {v: k for k, v in name_of.items()}[nm]
                    cx, cy = cx + dd[0], cy + dd[1]
                    if (cx, cy) in forbidden:
                        ok = False
                        break
                if not ok:
                    path = None
            if reroute_on and path:
                cells = reroute_cells
            if path:
                return {
                    "actor_pos": actor_pos,
                    "targets": [tuple(target)],
                    "walkable": frozenset(cells),
                    "forbidden": frozenset(forbidden),
                    "stride": 6,
                    "edge_offset": 3,
                    "actor_value": int(actor["primary_value"]) if actor else -1,
                    "marker_count": len(markers),
                    "target_value": -1,
                    "walkable_value": -1,
                    "plan_len": len(path),
                    "measured_synthesis": True,
                }
        return None
    except Exception:
        return None


def _stride_actor_hint_from_observation(
    mechanic_context: Mapping[str, Any],
) -> tuple[int, int] | None:
    """Derive the stride actor from causal observation (audit
    section 61/62): the 3x3 block whose CENTER pixel (rotation fixed
    point) translated by one stride in the direction of the LAST
    EXECUTED action, previous frame -> current frame. Returns the
    destination block's center coordinate, or None unless the
    candidate is UNIQUE -- ambiguity (moving same-shape objects,
    board resets) yields no hint and the perception falls back to
    its own heuristic. Fail-closed on every error."""
    try:
        prev = mechanic_context.get("previous_frame")
        cur = mechanic_context.get("frame")
        history = mechanic_context.get("history") or []
        if prev is None or cur is None or not history:
            return None
        action_raw = str(history[-1].get("action") or "").strip().upper()
        engine = MODEL_TO_ENGINE_ACTION.get(action_raw, action_raw)
        deltas = {
            "ACTION1": (0, -6),
            "ACTION2": (0, 6),
            "ACTION3": (-6, 0),
            "ACTION4": (6, 0),
        }
        if engine not in deltas:
            return None
        ddx, ddy = deltas[engine]
        before = np.asarray(prev)
        after = np.asarray(cur)
        if before.ndim == 3:
            before = before[-1]
        if after.ndim == 3:
            after = after[-1]
        if before.shape != after.shape or before.ndim != 2:
            return None
        ys, xs = np.where(before != after)
        if ys.size == 0 or ys.size > 400:
            return None
        cand: set[tuple[int, int]] = set()
        for y, x in zip(ys.tolist(), xs.tolist()):
            for oy in range(3):
                for ox in range(3):
                    cand.add((x - ox, y - oy))
        h, w = before.shape
        hits: list[tuple[int, int]] = []
        for sx, sy in cand:
            dx0, dy0 = sx + ddx, sy + ddy
            if min(sx, sy, dx0, dy0) < 0:
                continue
            if sy + 3 > h or sx + 3 > w or dy0 + 3 > h or dx0 + 3 > w:
                continue
            c_src = before[sy + 1, sx + 1]
            if after[dy0 + 1, dx0 + 1] != c_src:
                continue
            if after[sy + 1, sx + 1] == c_src:
                continue
            if before[dy0 + 1, dx0 + 1] == c_src:
                continue
            hits.append((dx0, dy0))
        deduped: list[tuple[int, int]] = []
        for pt in hits:
            if not any(
                abs(pt[0] - q[0]) <= 2 and abs(pt[1] - q[1]) <= 2
                for q in deduped
            ):
                deduped.append(pt)
        if len(deduped) != 1:
            return None
        dx0, dy0 = deduped[0]
        return (int(dx0) + 1, int(dy0) + 1)
    except Exception:
        return None


def _solve_stride_maze_legacy(
    arr: np.ndarray, engine_valid: set[str],
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from arcgym.eca import primitive_synthesis as _ps
    from arcgym.eca.mechanic_certificate import StrideMazeCertificate

    moves = engine_valid & _MOVE_ACTIONS
    if not moves:
        return _obs("stride_maze", "unavailable", "movement actions not in valid set"), None
    actor_hint = None
    if mechanic_context:
        raw_hint = mechanic_context.get("stride_actor_hint")
        try:
            if raw_hint is not None:
                actor_hint = (int(raw_hint[0]), int(raw_hint[1]))
        except (TypeError, ValueError, IndexError):
            actor_hint = None
    if actor_hint is None and mechanic_context and os.environ.get(
        "ECA_TUFA_STRIDE_ACTOR_HINT_FROM_OBS", "0"
    ).strip() == "1":
        actor_hint = _stride_actor_hint_from_observation(mechanic_context)
    measured = None
    if mechanic_context and os.environ.get(
        "ECA_TUFA_STRIDE_MEASURED_WALKABLE", "0"
    ).strip() == "1":
        measured = _stride_measured_walkable_update(mechanic_context)
    state = _ps._stride_marker_maze_state(arr, actor_hint=actor_hint)
    if state is not None and measured and measured.get("walkable"):
        merged = frozenset(set(state["walkable"]) | set(measured["walkable"]))
        if merged != state["walkable"]:
            state = dict(state)
            state["walkable"] = merged
    if (
        state is not None
        and measured
        and measured.get("death_forbidden")
        and os.environ.get(
            "ECA_TUFA_STRIDE_DEATH_FORBIDDEN", "0"
        ).strip() == "1"
    ):
        state = dict(state)
        state["forbidden"] = frozenset(
            set(state.get("forbidden") or frozenset())
            | set(measured["death_forbidden"])
        )
        state["walkable"] = frozenset(
            set(state["walkable"]) - set(measured["death_forbidden"])
        )
    if state is None and measured is not None:
        state = _stride_state_from_measurement(arr, actor_hint, measured, _ps)
    if state is None:
        return _obs(
            "stride_maze", "absent",
            "no stride-6 lattice maze structure in frame",
            actor_hint_used=bool(actor_hint is not None),
        ), None
    certified, reason = _verdict_fields(StrideMazeCertificate().evaluate(state))
    targets = list(state.get("targets") or [])
    forbidden = frozenset(state.get("forbidden") or frozenset())
    observation = _obs(
        "stride_maze",
        "certified" if certified else "detected",
        reason,
        walkable_count=len(state.get("walkable") or ()),
        target_count=len(targets),
        forbidden_count=len(forbidden),
        actor_hint_used=bool(actor_hint is not None),
    )
    if not targets:
        observation["reason"] = reason + "; lattice parsed but no target marker"
        return observation, None
    path = None
    for target in targets:
        path = _ps._stride_marker_maze_bfs_path(
            state["actor_pos"],
            tuple(target),
            state["walkable"],
            moves,
            stride=int(state["stride"]),
            edge_offset=int(state["edge_offset"]),
            forbidden=forbidden,
        )
        if path:
            break
    detail = observation.setdefault("detail", {})
    detail["plan_found"] = bool(path)
    detail["plan_len"] = len(path or ())
    if not path:
        observation["reason"] = reason + "; no lattice path to any target (BFS exhausted)"
        return observation, None
    actions = _to_model_steps(path)
    if actions is None:
        return observation, None
    return observation, {
        "family": "stride_maze",
        "certified": certified,
        "certificate_reason": reason,
        "actions": actions,
        "grounded_observation": _grounded_observation("stride_maze", state, actions),
        "detail": {
            "actor_pos": [int(v) for v in state["actor_pos"]],
            "target_count": len(targets),
            "forbidden_count": len(forbidden),
        },
    }


def _solve_sand_filter_legacy(
    arr: np.ndarray, engine_valid: set[str]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from arcgym.eca import primitive_synthesis as _ps
    from arcgym.eca.mechanic_certificate import SandFilterCertificate

    state = _ps._sand_filter_flow_state(arr)
    if state is None:
        return _obs("sand_filter", "absent", "no sand-filter flow structure in frame"), None
    certified, reason = _verdict_fields(SandFilterCertificate().evaluate(state))
    plan = _ps._sand_filter_flow_plan(state, engine_valid)
    observation = _obs(
        "sand_filter",
        "certified" if certified else "detected",
        reason,
        filter_count=int(state.get("filter_count", 0) or 0),
        bowl_count=int(state.get("bowl_count", 0) or 0),
        source_count=int(state.get("source_count", 0) or 0),
        plan_found=bool(plan),
        plan_len=len(plan or ()),
    )
    if not plan:
        observation["reason"] = reason + "; no executable operation plan"
        return observation, None
    actions = _to_model_steps(plan)
    if actions is None:
        return observation, None
    return observation, {
        "family": "sand_filter",
        "certified": certified,
        "certificate_reason": reason,
        "actions": actions,
        "grounded_observation": _grounded_observation(
            "sand_filter", state, actions, frame=arr
        ),
        "detail": {
            "filter_count": int(state.get("filter_count", 0) or 0),
        },
    }


def _runtime_artifact(
    family: str,
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        from arcgym.eca.tufa_mechanics.bridge import compute_runtime_artifact

        return compute_runtime_artifact(
            family,
            arr,
            level=int(level),
            valid_actions=engine_valid,
            context=mechanic_context,
        )
    except Exception:
        return None


def _runtime_observation(
    family: str, artifact: Mapping[str, Any]
) -> dict[str, Any]:
    runtime = artifact.get("runtime_mechanic")
    runtime = dict(runtime) if isinstance(runtime, Mapping) else {}
    plan = artifact.get("plan")
    plan = plan if isinstance(plan, Mapping) else None
    certified = bool(plan and plan.get("certified"))
    reason = (
        str(plan.get("certificate_reason", ""))
        if plan
        else "frame-bound mechanic schema available without an executable plan"
    )
    observation = _obs(
        family,
        "certified" if certified else "detected",
        reason,
        mechanic_family=str(runtime.get("mechanic_family", "")),
        plan_found=bool(plan),
        plan_len=len(plan.get("actions", [])) if plan else 0,
        rollout=str(runtime.get("rollout", "")),
    )
    observation["_runtime_mechanic"] = runtime
    return observation


def _solve_runtime_only(
    family: str,
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    artifact = _runtime_artifact(
        family,
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )
    if artifact is None:
        return _obs(family, "absent", "runtime mechanic module unavailable"), None
    observation = _runtime_observation(family, artifact)
    plan = artifact.get("plan")
    return observation, dict(plan) if isinstance(plan, Mapping) else None


def _runtime_blocks_legacy_fallback(artifact: Mapping[str, Any]) -> bool:
    runtime = artifact.get("runtime_mechanic")
    if not isinstance(runtime, Mapping):
        return False
    binding = runtime.get("current_binding")
    if not isinstance(binding, Mapping):
        return False
    return binding.get("status") == "unsupported_dynamic_hazards"


def _solve_mirror(
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    artifact = _runtime_artifact(
        "mirror",
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )
    if artifact is not None:
        plan = artifact.get("plan")
        if isinstance(plan, Mapping) and plan.get("certified"):
            return _runtime_observation("mirror", artifact), dict(plan)
    observation, plan = _solve_mirror_legacy(
        arr, engine_valid, mechanic_context=mechanic_context
    )
    if artifact is not None:
        runtime = artifact.get("runtime_mechanic")
        if isinstance(runtime, Mapping):
            observation["_runtime_mechanic"] = dict(runtime)
    return observation, plan


def _solve_stride_maze(
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    artifact = _runtime_artifact(
        "stride_maze",
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )
    if artifact is not None:
        plan = artifact.get("plan")
        if isinstance(plan, Mapping) and plan.get("certified"):
            return _runtime_observation("stride_maze", artifact), dict(plan)
        if _runtime_blocks_legacy_fallback(artifact):
            return _runtime_observation("stride_maze", artifact), None
    observation, plan = _solve_stride_maze_legacy(
        arr, engine_valid, mechanic_context=mechanic_context
    )
    if artifact is not None:
        runtime = artifact.get("runtime_mechanic")
        if isinstance(runtime, Mapping):
            observation["_runtime_mechanic"] = dict(runtime)
    return observation, plan


def _solve_sand_filter(
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    artifact = _runtime_artifact(
        "sand_filter",
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )
    if artifact is not None:
        plan = artifact.get("plan")
        if isinstance(plan, Mapping) and plan.get("certified"):
            return _runtime_observation("sand_filter", artifact), dict(plan)
    observation, plan = _solve_sand_filter_legacy(arr, engine_valid)
    if artifact is not None:
        runtime = artifact.get("runtime_mechanic")
        if isinstance(runtime, Mapping):
            observation["_runtime_mechanic"] = dict(runtime)
    return observation, plan


def _solve_program_execution(
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return _solve_runtime_only(
        "program_execution",
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )


def _solve_linked_extent(
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return _solve_runtime_only(
        "linked_extent",
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )


def _solve_rule_rewrite(
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return _solve_runtime_only(
        "rule_rewrite",
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )


def _solve_reflection_orbit(
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return _solve_runtime_only(
        "reflection_orbit",
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )


def _solve_anchor_overlap(
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return _solve_runtime_only(
        "anchor_overlap",
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )


def _solve_region_paint(
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return _solve_runtime_only(
        "region_paint",
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )


def _solve_shape_marker(
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return _solve_runtime_only(
        "shape_marker",
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )


def _solve_hierarchical_slots(
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return _solve_runtime_only(
        "hierarchical_slots",
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )


def _solve_recorded_replay(
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return _solve_runtime_only(
        "recorded_replay",
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )


def _solve_cycle_permutation(
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return _solve_runtime_only(
        "cycle_permutation",
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )


def _solve_control_marker(
    arr: np.ndarray,
    engine_valid: set[str],
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return _solve_runtime_only(
        "control_marker",
        arr,
        engine_valid,
        level=level,
        mechanic_context=mechanic_context,
    )


_FAMILY_SOLVERS = (
    ("mirror", _solve_mirror),
    ("color_cycle", _solve_color_cycle),
    ("stride_maze", _solve_stride_maze),
    ("sand_filter", _solve_sand_filter),
    ("program_execution", _solve_program_execution),
    ("linked_extent", _solve_linked_extent),
    ("rule_rewrite", _solve_rule_rewrite),
    ("reflection_orbit", _solve_reflection_orbit),
    ("anchor_overlap", _solve_anchor_overlap),
    ("region_paint", _solve_region_paint),
    ("shape_marker", _solve_shape_marker),
    ("hierarchical_slots", _solve_hierarchical_slots),
    ("recorded_replay", _solve_recorded_replay),
    ("cycle_permutation", _solve_cycle_permutation),
    ("control_marker", _solve_control_marker),
)

# Arbitration order for multi-certified frames (v1.6). Measured false
# positives: mirror misfired on ka59 AND sp80, color_cycle misfired on
# sp80; sand_filter and stride_maze have no recorded false positive. So
# when several certificates fire on one frame — b6 telemetry caught
# sand_filter+mirror+color_cycle simultaneously certified on sp80 — we
# keep only the most structurally specific family and demote the rest.
_ARBITRATION_ORDER = (
    "sand_filter",
    "program_execution",
    "linked_extent",
    "rule_rewrite",
    "reflection_orbit",
    "anchor_overlap",
    "region_paint",
    "shape_marker",
    "hierarchical_slots",
    "recorded_replay",
    "cycle_permutation",
    "stride_maze",
    "color_cycle",
    "mirror",
)


def _arbitrate_certified(result: dict[str, Any]) -> None:
    """Fail closed when independently certified mechanics predict a conflict."""
    plans = [p for p in result.get("plans", []) if isinstance(p, dict)]
    certified = [p for p in plans if p.get("authorized", p.get("certified"))]
    if len(certified) <= 1:
        return
    signatures: dict[str, str] = {}
    for plan in certified:
        family = str(plan.get("family", "") or "")
        actions = [item for item in plan.get("actions", []) if isinstance(item, Mapping)]
        grounding = plan.get("grounded_observation") or {}
        expected = grounding.get("expected_change") or {}
        exact = expected.get("runtime_mechanic_exact")
        if not isinstance(exact, Mapping):
            exact = expected.get(f"{family}_exact")
        signatures[family] = _stable_digest(
            {
                "first_action": actions[0] if actions else None,
                "exact_prediction": exact if isinstance(exact, Mapping) else expected,
            }
        )
    if len(set(signatures.values())) > 1:
        families = sorted(signatures)
        for plan in certified:
            plan["certified"] = False
            plan["authorized"] = False
            plan["advisory_only"] = True
            plan["certificate_reason"] = (
                "demoted by fail-closed arbitration: certified mechanics "
                "disagree on the first action or exact prediction"
            )
        result["arbitration"] = {
            "status": "conflict_abstain",
            "families": families,
            "signatures": signatures,
        }
        obs = result.get("observations")
        if isinstance(obs, dict):
            for entry in obs.get("families", []):
                if isinstance(entry, dict) and entry.get("family") in signatures:
                    entry["status"] = "detected"
                    entry["reason"] = (
                        str(entry.get("reason", ""))
                        + "; conflicting certified mechanic; execution withheld"
                    )
        return
    rank = {family: i for i, family in enumerate(_ARBITRATION_ORDER)}
    winner = min(certified, key=lambda p: rank.get(str(p.get("family")), len(rank)))
    demoted = []
    for plan in certified:
        if plan is winner:
            continue
        plan["certified"] = False
        plan["certificate_reason"] = (
            f"demoted by arbitration: {winner.get('family')} is the more "
            "structurally specific certified mechanic on this frame"
        )
        demoted.append(str(plan.get("family")))
    obs = result.get("observations")
    if isinstance(obs, dict):
        for entry in obs.get("families", []):
            if isinstance(entry, dict) and entry.get("family") in demoted:
                entry["status"] = "detected"
                entry["reason"] = (
                    str(entry.get("reason", ""))
                    + f"; demoted by arbitration in favour of {winner.get('family')}"
                )
    result["arbitration"] = {
        "status": "equivalent_prediction_priority",
        "winner": str(winner.get("family", "")),
        "families": sorted(signatures),
    }


def _runtime_frame_hash(arr: np.ndarray) -> str:
    exclusions = [
        {
            "row0": 0,
            "row1": min(1, int(arr.shape[0])),
            "col0": max(0, int(arr.shape[1]) - 8),
            "col1": int(arr.shape[1]),
        }
    ]
    return str(_masked_frame_hash(arr, exclusions) or _stable_digest(arr))


def _cache_get(
    key: str,
) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
    with _CACHE_LOCK:
        value = _RESULT_CACHE.get(key)
        if value is None:
            return None
        _RESULT_CACHE.move_to_end(key)
        return copy.deepcopy(value)


def _cache_put(
    key: str,
    value: tuple[dict[str, Any], dict[str, Any] | None],
    *,
    max_entries: int,
) -> None:
    if max_entries <= 0:
        return
    with _CACHE_LOCK:
        _RESULT_CACHE[key] = copy.deepcopy(value)
        _RESULT_CACHE.move_to_end(key)
        entry = value[1]
        if isinstance(entry, Mapping) and isinstance(entry.get("grounded_observation"), Mapping):
            _PERCEPTION_CACHE[key] = copy.deepcopy(dict(entry["grounded_observation"]))
            _PERCEPTION_CACHE.move_to_end(key)
        while len(_RESULT_CACHE) > max_entries:
            stale, _ = _RESULT_CACHE.popitem(last=False)
            _PERCEPTION_CACHE.pop(stale, None)
        while len(_PERCEPTION_CACHE) > max_entries:
            _PERCEPTION_CACHE.popitem(last=False)


def _invoke_family_solver(
    family: str,
    solver: Any,
    arr: np.ndarray,
    engine_valid: set[str],
    role_cache: dict[str, Any] | None,
    mechanic_context: Mapping[str, Any] | None = None,
    level: int = 0,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        parameters = inspect.signature(solver).parameters.values()
        accepts_kwargs = any(
            item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters
        )
        names = {item.name for item in parameters}
    except (TypeError, ValueError):
        accepts_kwargs = False
        names = set()
    call_kwargs: dict[str, Any] = {}
    if accepts_kwargs or "mechanic_context" in names:
        call_kwargs["mechanic_context"] = mechanic_context
    if accepts_kwargs or "level" in names:
        call_kwargs["level"] = int(level)
    if family == "color_cycle":
        return solver(arr, engine_valid, role_cache, **call_kwargs)
    return solver(arr, engine_valid, **call_kwargs)


def _family_process_target(
    output: Any,
    family: str,
    arr: np.ndarray,
    engine_valid: set[str],
    role_cache: dict[str, Any] | None,
    mechanic_context: Mapping[str, Any] | None,
    level: int,
) -> None:
    try:
        solver = dict(_FAMILY_SOLVERS)[family]
        observation, entry = _invoke_family_solver(
            family,
            solver,
            arr,
            engine_valid,
            role_cache,
            mechanic_context,
            level,
        )
        output.put((observation, entry, ""))
    except BaseException as exc:
        output.put((None, None, f"{type(exc).__name__}: {exc}"))


def _run_family_solver_process(
    family: str,
    arr: np.ndarray,
    engine_valid: set[str],
    role_cache: dict[str, Any] | None,
    timeout_ms: float,
    mechanic_context: Mapping[str, Any] | None,
    level: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, BaseException | None, bool]:
    context = mp.get_context("spawn")
    output = context.Queue(maxsize=1)
    process = context.Process(
        target=_family_process_target,
        args=(
            output,
            family,
            arr,
            engine_valid,
            role_cache,
            mechanic_context,
            int(level),
        ),
        name=f"eca-toolbox-{family}",
        daemon=True,
    )
    process.start()
    process.join(max(0.001, float(timeout_ms) / 1000.0))
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(1.0)
        output.close()
        return None, None, None, True
    try:
        observation, entry, error_text = output.get(timeout=0.1)
    except queue.Empty:
        observation, entry, error_text = None, None, "worker returned no result"
    finally:
        output.close()
    error = RuntimeError(error_text) if error_text else None
    return observation, entry, error, False


def _run_family_solver(
    family: str,
    solver: Any,
    arr: np.ndarray,
    engine_valid: set[str],
    role_cache: dict[str, Any] | None,
    timeout_ms: float,
    mechanic_context: Mapping[str, Any] | None = None,
    level: int = 0,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, BaseException | None, bool]:
    if os.environ.get("ECA_TUFA_FAMILY_ISOLATION", "thread").strip().lower() == "process":
        return _run_family_solver_process(
            family,
            arr,
            engine_valid,
            role_cache,
            timeout_ms,
            mechanic_context,
            level,
        )
    output: "queue.Queue[tuple[Any, Any, BaseException | None]]" = queue.Queue(maxsize=1)
    effective_timeout_ms = max(1.0, float(timeout_ms))

    def target() -> None:
        try:
            observation, entry = _invoke_family_solver(
                family,
                solver,
                arr,
                engine_valid,
                role_cache,
                mechanic_context,
                level,
            )
            output.put((observation, entry, None))
        except BaseException as exc:  # isolated worker must report every failure
            output.put((None, None, exc))
        finally:
            with _FAMILY_WORKER_LOCK:
                _FAMILY_WORKERS.discard(threading.current_thread())

    worker = threading.Thread(
        target=target, name=f"eca-toolbox-{family}", daemon=True
    )
    admission_raw = os.environ.get("ECA_TUFA_FAMILY_ADMISSION_QUEUE_MS", "0")
    try:
        admission_wait_ms = max(0.0, float(admission_raw or 0))
    except (TypeError, ValueError):
        admission_wait_ms = 0.0
    admission_deadline = time.monotonic() + admission_wait_ms / 1000.0
    admitted = False
    while True:
        with _FAMILY_WORKER_LOCK:
            stale = {item for item in _FAMILY_WORKERS if not item.is_alive()}
            _FAMILY_WORKERS.difference_update(stale)
            if len(_FAMILY_WORKERS) < _MAX_LIVE_FAMILY_WORKERS:
                _FAMILY_WORKERS.add(worker)
                worker.start()
                admitted = True
                break
        if time.monotonic() >= admission_deadline:
            break
        time.sleep(0.004)
    if not admitted:
        return None, None, TimeoutError("family worker capacity exhausted"), True
    worker.join(max(0.001, effective_timeout_ms / 1000.0))
    if worker.is_alive():
        return None, None, None, True
    try:
        observation, entry, error = output.get_nowait()
    except queue.Empty:
        return None, None, RuntimeError("family worker returned no result"), False
    return observation, entry, error, False


def compute_solver_plans(
    frame: Any,
    *,
    level: int = 0,
    mechanic_context: Mapping[str, Any] | None = None,
    valid_actions: Iterable[Any] | None = None,
    budget_ms: float = 2500.0,
    role_cache: dict[str, Any] | None = None,
    families: Iterable[str] | None = None,
    suppress_families: Iterable[str] | None = None,
    family_timeout_ms: float = 750.0,
    cache_entries: int = 256,
    round_robin_offset: int = 0,
    max_families: int | None = None,
) -> dict[str, Any]:
    """Run selected family gates/solvers against one frame and return plain data.

    Never raises. ``plans`` holds one entry per family whose perception
    parsed AND whose solver produced a full plan; ``certified`` marks the
    entries that also passed their MechanicCertificate.

    ``families`` is applied before solver iteration, so a one-family lease
    does not pay the CPU cost of the other readers. ``None`` retains the
    historical all-family audit behaviour.

    ``suppress_families`` is the host-side kill switch: families whose
    advertised plan was followed by a game-over are excluded here (recorded
    under ``skipped``) so a misfiring certificate cannot burn the rest of
    the session re-offering a plan that kills the run.
    """
    started = time.monotonic()
    budget_guard_ms = min(250.0, max(1.0, float(budget_ms) * 0.10))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "plans": [],
        "runtime_mechanics": [],
        "skipped": [],
        "errors": [],
        "family_elapsed_ms": {},
        "cache": {"hits": 0, "misses": 0, "perception_entries": 0},
        "worker": {
            "isolation": os.environ.get(
                "ECA_TUFA_FAMILY_ISOLATION", "thread"
            ).strip().lower(),
            "live_before": 0,
            "live_after": 0,
            "max_live": _MAX_LIVE_FAMILY_WORKERS,
            "terminated_processes": 0,
        },
        "budget": {"limit_ms": float(budget_ms), "guard_ms": budget_guard_ms},
    }
    with _FAMILY_WORKER_LOCK:
        result["worker"]["live_before"] = sum(
            1 for item in _FAMILY_WORKERS if item.is_alive()
        )
    arr = _as_2d_frame(frame)
    if arr is None:
        result["available"] = False
        result["errors"].append({"family": "*", "error": "frame is not a 2D/3D grid"})
        result["elapsed_ms"] = round((time.monotonic() - started) * 1000.0, 3)
        return result
    try:
        _ensure_arcgym_importable()
    except Exception as exc:  # noqa: BLE001 - degrade, never break the turn
        result["available"] = False
        result["errors"].append({"family": "*", "error": f"{type(exc).__name__}: {exc}"})
        result["elapsed_ms"] = round((time.monotonic() - started) * 1000.0, 3)
        return result
    try:
        from arcgym.eca.tufa_mechanics.base import coerce_mechanic_context

        context = coerce_mechanic_context(
            arr,
            level=int(level),
            context=mechanic_context,
        )
        context_input = context.module_input()
        context_hash = _stable_digest(context.cache_payload())
        result["mechanic_context"] = context.public_summary()
    except Exception:
        context_input = mechanic_context
        context_hash = _stable_digest(
            {"level": int(level), "context_available": bool(mechanic_context)}
        )
    engine_valid = _engine_valid_actions(valid_actions)
    selected_order = (
        []
        if families is None
        else list(
            dict.fromkeys(
                str(f).strip().lower() for f in families if str(f).strip()
            )
        )
    )
    selected = None if families is None else set(selected_order)
    suppressed = {str(f).strip() for f in (suppress_families or ()) if str(f).strip()}
    solver_rows = list(_FAMILY_SOLVERS)
    if selected_order:
        priority = {family: index for index, family in enumerate(selected_order)}
        solver_rows.sort(key=lambda item: priority.get(item[0], len(priority)))
    eligible = [
        family
        for family, _solver in solver_rows
        if (selected is None or family in selected) and family not in suppressed
    ]
    scheduled = set(eligible)
    if max_families is not None and eligible:
        count = max(1, min(len(eligible), int(max_families)))
        offset = int(round_robin_offset) % len(eligible)
        rotated = eligible[offset:] + eligible[:offset]
        scheduled = set(rotated[:count])
    frame_hash = _runtime_frame_hash(arr)
    valid_hash = _stable_digest(sorted(engine_valid))
    role_hash = _stable_digest(
        {"role_cache": role_cache or {}, "mechanic_context": context_hash}
    )
    # Plan delivery mode (observations are always produced):
    #   advertise - plans in sandbox data AND certified call-to-action in the
    #               prompt (v1.2-v1.4 behaviour, default)
    #   silent    - plans stay readable in sandbox data; the prompt carries
    #               observations only (pull, not push)
    #   none      - plans are withheld entirely; the LLM gets structured
    #               observations + uncertainty and plans by itself
    mode = os.environ.get("ECA_TUFA_PLAN_DELIVERY", "advertise").strip().lower()
    if mode not in {"advertise", "silent", "none"}:
        mode = "advertise"
    result["delivery"] = mode
    observations: list[dict[str, Any]] = []
    result["observations"] = {
        "confidence_note": (
            "certified = perception passed a mechanic certificate; "
            "detected = structure parsed but certificate rejected or no plan exists; "
            "absent = this mechanic's structure is not in the frame"
        ),
        "families": observations,
    }

    for family, solver in solver_rows:
        if selected is not None and family not in selected:
            result["skipped"].append({"family": family, "reason": "not_allowlisted"})
            continue
        if family in suppressed:
            result["skipped"].append({"family": family, "reason": "suppressed_after_game_over"})
            observations.append(
                _obs(family, "suppressed", "advertised plan preceded a game over on this level")
            )
            continue
        if family not in scheduled:
            result["skipped"].append({"family": family, "reason": "round_robin_deferred"})
            continue
        elapsed_before_ms = (time.monotonic() - started) * 1000.0
        remaining_budget_ms = float(budget_ms) - elapsed_before_ms
        if remaining_budget_ms <= budget_guard_ms:
            result["skipped"].append({"family": family, "reason": "budget_exceeded"})
            observations.append(_obs(family, "skipped", "compute budget exhausted"))
            continue
        cache_key = f"{family}:{frame_hash}:{valid_hash}:{role_hash}"
        # sec 77 DEFECT B: a cache hit skips the family solver entirely,
        # silently skipping the A-series learning exactly on the two
        # recurring-frame moments that matter (blocked-move no-ops and
        # post-death splashes). When a learning flag is live AND a
        # context is present, bypass the cache so legacy always runs.
        try:
            _learning_ctx_live = bool(
                mechanic_context is not None
                and (
                    os.environ.get(
                        "ECA_TUFA_STRIDE_ACTOR_HINT_FROM_OBS", "0"
                    ).strip() == "1"
                    or os.environ.get(
                        "ECA_TUFA_STRIDE_MEASURED_WALKABLE", "0"
                    ).strip() == "1"
                    or os.environ.get(
                        "ECA_TUFA_STRIDE_DEATH_FORBIDDEN", "0"
                    ).strip() == "1"
                    or os.environ.get(
                        "ECA_TUFA_MIRROR_MEASURED_WALLS", "0"
                    ).strip() == "1"
                    or os.environ.get(
                        "ECA_TUFA_COLOR_CYCLE_MEASURED", "0"
                    ).strip() == "1"
                )
            )
        except Exception:
            _learning_ctx_live = False
        cached = (
            _cache_get(cache_key)
            if int(cache_entries) > 0 and not _learning_ctx_live
            else None
        )
        if cached is not None:
            observation, entry = cached
            result["cache"]["hits"] += 1
            result["family_elapsed_ms"][family] = 0.0
        else:
            result["cache"]["misses"] += 1
            family_started = time.monotonic()
            observation, entry, error, timed_out = _run_family_solver(
                family,
                solver,
                arr,
                engine_valid,
                role_cache,
                min(
                    float(family_timeout_ms),
                    max(1.0, remaining_budget_ms - budget_guard_ms),
                ),
                context_input,
                int(level),
            )
            elapsed = round((time.monotonic() - family_started) * 1000.0, 3)
            result["family_elapsed_ms"][family] = elapsed
            if timed_out:
                if result["worker"]["isolation"] == "process":
                    result["worker"]["terminated_processes"] += 1
                result["skipped"].append({"family": family, "reason": "family_timeout"})
                observations.append(_obs(family, "timeout", "per-family hard timeout"))
                continue
            if error is not None:
                exc = error
                result["errors"].append(
                    {"family": family, "error": f"{type(exc).__name__}: {exc}"}
                )
                observations.append(_obs(family, "error", type(exc).__name__))
                continue
            assert observation is not None
            _cache_put(
                cache_key,
                (observation, entry),
                max_entries=max(0, int(cache_entries)),
            )
        if observation is None:
            exc = RuntimeError("family worker returned an empty observation")
            result["errors"].append({"family": family, "error": f"{type(exc).__name__}: {exc}"})
            observations.append(_obs(family, "error", type(exc).__name__))
            continue
        runtime_mechanic = observation.pop("_runtime_mechanic", None)
        if isinstance(runtime_mechanic, Mapping):
            result["runtime_mechanics"].append(copy.deepcopy(dict(runtime_mechanic)))
        observations.append(observation)
        if entry is None or mode == "none":
            continue
        entry["action_count"] = len(entry.get("actions") or [])
        result["plans"].append(entry)

    _arbitrate_certified(result)
    if os.environ.get(
        "ECA_TUFA_STRIDE_DEATH_FORBIDDEN", "0"
    ).strip() == "1":
        _hz = _stride_measured_hazards_view(frame, level)
        if _hz:
            result["measured_hazards"] = _hz
    if os.environ.get(
        "ECA_TUFA_MIRROR_DEATH_ADVICE", "0"
    ).strip() == "1":
        _pd = _mirror_death_view(frame, level)
        if _pd:
            _hz2 = result.get("measured_hazards")
            if not isinstance(_hz2, dict):
                _hz2 = {}
                result["measured_hazards"] = _hz2
            _hz2["pair_deaths"] = _pd
    with _CACHE_LOCK:
        result["cache"]["perception_entries"] = len(_PERCEPTION_CACHE)
    result["elapsed_ms"] = round((time.monotonic() - started) * 1000.0, 3)
    with _FAMILY_WORKER_LOCK:
        result["worker"]["live_after"] = sum(
            1 for item in _FAMILY_WORKERS if item.is_alive()
        )
    return result


def render_solver_plans_snippet(solver_plans: Mapping[str, Any] | None) -> str:
    """Render only one currently authorized family into the model prompt."""
    if not isinstance(solver_plans, Mapping):
        return ""
    delivery = str(solver_plans.get("delivery", "advertise") or "advertise")
    if delivery in {"shadow", "none", "silent"}:
        return ""
    hazard_line = ""
    _hz = solver_plans.get("measured_hazards")
    if isinstance(_hz, Mapping) and _hz.get("pair_deaths"):
        _pairs = "; ".join(
            " & ".join(f"(row={int(y)}, col={int(x)})" for x, y in pair)
            for pair in list(_hz["pair_deaths"])[:4]
        )
        hazard_line = (
            "[ECA MEASURED HAZARDS] A life ended TWICE right after the "
            "cursor pair stood at " + _pairs + " (measured on this "
            "level). The fatal move was made from that position - "
            "approach differently instead of repeating it. "
        )
    if isinstance(_hz, Mapping) and _hz.get("lethal"):
        _cells = ", ".join(
            f"(row={int(y)}, col={int(x)})"
            for x, y in list(_hz["lethal"])[:8]
        )
        hazard_line = (
            "[ECA MEASURED HAZARDS] Cells that ended a life TWICE "
            "(measured on this level, not guessed): " + _cells + ". "
            "Moving onto one of them loses the run - plan routes "
            "around them and do not re-test them."
        )
    plans = [p for p in solver_plans.get("plans", []) if isinstance(p, Mapping)]
    authorized = [
        plan
        for plan in plans
        if bool(plan.get("authorized", plan.get("certified", False)))
    ]
    if len(authorized) != 1:
        return hazard_line
    plan = authorized[0]
    family = str(plan.get("family", "") or "")
    if not family:
        return hazard_line
    lines: list[str] = []
    mechanic_memory = solver_plans.get("mechanic_schema_advisory")
    if isinstance(mechanic_memory, Mapping):
        schema_rows = [
            item
            for item in mechanic_memory.get("schemas", [])
            if isinstance(item, Mapping)
            and str(item.get("source_family", "") or "") == family
            and str(item.get("status", "") or "") == "verified"
            and str((item.get("current_binding") or {}).get("status", "") or "")
            == "bound_verified"
        ][:1]
        if schema_rows:
            summaries = []
            probe_examples = []
            for item in schema_rows:
                operator = item.get("operator") or {}
                binding = item.get("current_binding") or {}
                role_bindings = []
                for role in binding.get("role_bindings", []):
                    if not isinstance(role, Mapping) or not role.get("role"):
                        continue
                    if "row" in role and "col" in role:
                        interaction = str(role.get("interaction", "") or "")
                        prefix = f"{role.get('role')}"
                        if interaction:
                            prefix += f":{interaction}"
                        role_bindings.append(
                            f"{prefix}@({int(role['row'])},{int(role['col'])})"
                            + (
                                f" visible={role.get('visible_symbol')}"
                                if role.get("visible_symbol")
                                else ""
                            )
                        )
                        if interaction == "MOUSE":
                            row = int(role["row"])
                            col = int(role["col"])
                            symbol = str(role.get("visible_symbol", "") or "")
                            action_code = (
                                "action([{'action':'MOUSE','row':"
                                f"{row},'col':{col}}}])"
                            )
                            if symbol:
                                probe_examples.append(
                                    "lines=current_frame.ascii.splitlines(); "
                                    f"print(lines[{row}][{col}]); "
                                    f"{action_code} if lines[{row}][{col}]=="
                                    f"{symbol!r} else None"
                                )
                            else:
                                probe_examples.append(action_code)
                    else:
                        role_bindings.append(str(role.get("role")))
                summaries.append(
                    f"{item.get('mechanic_id')} [{item.get('status')}] "
                    f"operator={operator.get('transition', operator.get('effect', operator.get('kind', 'unknown')))}; "
                    f"binding={binding.get('status', 'unbound')}; "
                    f"roles={','.join(role_bindings) if role_bindings else 'unbound'}; "
                    f"goal={item.get('goal_relation', 'unknown')}"
                )
            lines.append(
                "ECA cross-level mechanic memory (structured advisory only; "
                "not action authorization): " + " | ".join(summaries) + ". "
                "These role bindings were re-detected on the current level, "
                "not copied from an older layout. Inspect bound roles before "
                "scanning unrelated objects; when the visible role and operator "
                "agree, use that action family at the bound role as the first "
                "probe, then verify the observed transition. A binding marked "
                "with an interaction such as `click_site:MOUSE@(row,col)` is "
                "the exact interaction point; do not replace it with the "
                "component centroid."
                + (
                    " Confirm an exact cell with `current_frame.ascii.splitlines()`: "
                    "segmentation `boundary` lists contour corners, not every cell. "
                    "Use this one-step probe form: `"
                    + "`; `".join(probe_examples[:4])
                    + "` ."
                    if probe_examples
                    else ""
                )
            )
    if delivery == "advisory":
        parts = (
            f"{family} ({int(plan.get('action_count', 0) or 0)}-step prefix "
            f"of {int(plan.get('full_action_count', 0) or 0)})"
        )
        lines.append(
            "ECA-lite revocable candidate prefix available: "
            + parts
            + ". This is family-gated advice, not proof. Inspect "
            "each plan's `grounded_observation`: first match its role-labelled "
            "row/col entities to the visible board. Then execute at most its first action "
            "with `action([plan['actions'][0]])`, then stop and re-read the refreshed "
            "advice and compare the observed transition with `expected_change` and "
            "`falsifiers`. Never batch the prefix or infer the withheld suffix. If the family "
            "or prefix disappears, immediately continue with your own pure strategy."
        )
    if delivery == "advertise":
        parts = (
            f"{family} ({int(plan.get('action_count', 0) or 0)} actions)"
        )
        lines.append(
            "ECA authorized one-step solver lease available: "
            + parts
            + ". In the python tool, `eca_advice['solver_plans']['plans']` contains exactly "
            "one authorized prefix. Execute only its first action, e.g. "
            "`plan = eca_advice['solver_plans']['plans'][0]; "
            "action([plan['actions'][0]])`, then stop and inspect the new frame. "
            "If that step contradicts what you observe, the mechanic detector "
            "misfired on this game: ignore the plan entirely and do not re-read it. "
            "Likewise, if you executed a plan and the game then ended or the board "
            "contradicted it mid-way, treat that family as misfired for this game."
        )
    if hazard_line:
        lines.insert(0, hazard_line)
    return " ".join(lines)
