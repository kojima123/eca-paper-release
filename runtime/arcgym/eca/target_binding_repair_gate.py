"""Phase 42 — Target-binding repair gate.

The gate observes ECA's current ControlIR proposal alongside the
teacher's perception-driven advice and asks the Phase 41 mismatch
report whether the proposal is likely to disagree with the teacher
on objective- or payload-binding terms. When risk is high the gate
emits an advisory ``RepairDecision`` recommending the agent prefer
a teacher-aligned site/operator.

Phase 42 is **advisory-only by default** (see Phase 41 spec:
"Start in advisory mode only. Do not count it as adapter-free
score."). The gate does NOT mutate ECA's authoritative action; it
only publishes recommendations and emits metrics. Promotion to
``guarded`` is a Phase 43+ decision.

This module is intentionally self-contained: it reads the Phase 41
``top_risk_rules`` directly from the JSON report (each rule has a
``feature``, ``value``, and ``positive_rate``). The actual
``MismatchFrequencyModel`` lives in :mod:`adapter_mismatch_learner`
and remains the source-of-truth trainer; the gate consumes only the
distilled rules so it has no runtime dependency on the learner.

Modes (controlled by ``V2_ECA_TARGET_BINDING_REPAIR``):

* ``off`` (default when env unset/empty/0/false): every call returns
  a no-op decision.
* ``advisory`` (recommended): full risk evaluation, repair flagged in
  the decision, no action mutation.
* ``aggressive``: same as advisory but lowers thresholds; reserved
  for future shadow experiments. Still NOT authoritative.

Public API:

* :class:`RepairDecision`
* :class:`TargetBindingRepairGate`
* :func:`repair_gate_mode`
* :func:`repair_gate_enabled`
* :func:`load_default_repair_gate`
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Mapping


_DEFAULT_REPORT_BASENAME = "phase49b_adapter_mismatch_learning_report_25g_wide_20260509.json"
# Legacy fallback: Phase 41 corpus (7 games, narrow vocab). Searched after
# the default so existing env overrides keep working but new installs
# pick up the wider 25-game corpus. Removed once Phase 49b is verified
# in the canonical 25-game sweep.
_LEGACY_REPORT_BASENAMES = ("phase41_adapter_mismatch_learning_report_20260508.json",)

_LABELS = (
    "objective_mismatch",
    "payload_mismatch",
    "action_mismatch",
    "scoreless_mismatch",
)


def _normalize_mode(value: str | None) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"1", "true", "yes", "on", "advisory", "advise"}:
        return "advisory"
    if mode in {"aggressive", "guarded-shadow", "shadow"}:
        return "aggressive"
    if mode in {"0", "false", "no", "off", ""}:
        return "off"
    return "off"


def repair_gate_mode() -> str:
    """Return the requested repair-gate mode from env."""

    return _normalize_mode(os.environ.get("V2_ECA_TARGET_BINDING_REPAIR"))


def repair_gate_enabled() -> bool:
    return repair_gate_mode() in {"advisory", "aggressive"}


def _ood_backoff_enabled() -> bool:
    """Return True iff the OOD-backoff env override is enabled."""

    raw = os.environ.get("V2_ECA_REPAIR_GATE_OOD_BACKOFF") or ""
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _ood_threshold_from_env(default: float = 0.5) -> float:
    """Return the OOD threshold from env, falling back to the default."""

    raw = os.environ.get("V2_ECA_REPAIR_GATE_OOD_THRESHOLD")
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _selective_mode_from_env() -> str:
    """Return the Phase 46 selective-gating mode from env.

    Modes:
      * ``off`` (default) — Phase 42/45 behaviour, no selective gating.
      * ``selective`` — fire only when ECA stuck AND high risk AND low OOD.
      * ``aggressive_selective`` — same as selective but uses a lower
        risk threshold (0.7) so it fires more often during shadow.
    """

    raw = os.environ.get("V2_ECA_REPAIR_GATE_SELECTIVE") or ""
    value = str(raw).strip().lower()
    if value in {"selective", "tight", "tight_selective"}:
        return "selective"
    if value in {"aggressive_selective", "aggressive-selective", "aggressive"}:
        return "aggressive_selective"
    if value in {"", "0", "false", "no", "off"}:
        return "off"
    return "off"


def _selective_stuck_threshold_from_env(default: int = 8) -> int:
    """Return the Phase 46 stuck-threshold K_stuck from env.

    Used by the bridge + gate. Defaults to 8 frames. Clamped to >=1.
    """

    raw = os.environ.get("V2_ECA_REPAIR_GATE_SELECTIVE_STUCK_THRESHOLD")
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return int(default)
    return max(1, value)


def _selective_high_risk_threshold_from_env(default: float = 0.9) -> float:
    """Return the Phase 46 selective high-risk threshold from env."""

    raw = os.environ.get("V2_ECA_REPAIR_GATE_SELECTIVE_HIGH_RISK")
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


# ──────────────────────────────────────────────────────────────────
# Phase 55 — per-game adapter baseline + score-aware suppression.
# When ECA's live score >= adapter ceiling for the current game, the
# gate suppresses recommendations entirely (Phase 53 K-step replay
# showed gate is HARMFUL on cd82/sb26 where ECA already wins). This
# is a cheap, advisory-only narrowing of the diagnostic signal —
# never mutates ECA's authoritative action path.
# Source of truth for ceilings: docs/CLAUDE.md "ARC-AGI-3 game catalog"
# adapter-included scores (Phase 37, 2026-05-05).
# ──────────────────────────────────────────────────────────────────
_PER_GAME_ADAPTER_BASELINE: dict[str, int] = {
    "ar25": 8, "bp35": 3, "cd82": 6, "cn04": 6, "dc22": 3,
    "ft09": 6, "g50t": 1, "ka59": 5, "lf52": 4, "lp85": 8,
    "ls20": 7, "m0r0": 2, "r11l": 6, "re86": 2, "s5i5": 2,
    "sb26": 8, "sc25": 5, "sk48": 1, "sp80": 2, "su15": 2,
    "tn36": 5, "tr87": 6, "tu93": 1, "vc33": 7, "wa30": 2,
}

# Phase 56 — per-game ECA-only baseline (Phase 39 sum 25/25, last
# verified 2026-04-30 via canonical sweep). Used for static
# game-baseline parity suppression: when eca_baseline >=
# adapter_baseline, the gate's recommendations cannot improve score
# (Phase 53 K-step replay confirmed). Suppress unconditionally.
_PER_GAME_ECA_BASELINE: dict[str, int] = {
    "ar25": 0, "bp35": 0, "cd82": 6, "cn04": 0, "dc22": 0,
    "ft09": 4, "g50t": 0, "ka59": 0, "lf52": 0, "lp85": 1,
    "ls20": 4, "m0r0": 0, "r11l": 0, "re86": 0, "s5i5": 0,
    "sb26": 8, "sc25": 0, "sk48": 0, "sp80": 0, "su15": 0,
    "tn36": 0, "tr87": 0, "tu93": 0, "vc33": 2, "wa30": 0,
}


def _score_aware_suppression_from_env() -> bool:
    """Phase 55 — opt-in to suppress gate fires when ECA score >= baseline.

    Default OFF for backward compatibility. When ON, the gate's
    selective_blocked_reason becomes ``score_above_baseline`` for
    games where ECA has already met or exceeded the per-game
    adapter ceiling.
    """

    raw = str(os.environ.get("V2_ECA_REPAIR_GATE_SCORE_AWARE", "off") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _per_game_adapter_baseline(game_id: str | None) -> int | None:
    """Return the known adapter ceiling for ``game_id`` or ``None``."""

    if not game_id:
        return None
    key = str(game_id or "").strip().lower()
    return _PER_GAME_ADAPTER_BASELINE.get(key)


def _per_game_eca_baseline(game_id: str | None) -> int | None:
    """Phase 56 — return the known ECA-only baseline for ``game_id``."""

    if not game_id:
        return None
    key = str(game_id or "").strip().lower()
    return _PER_GAME_ECA_BASELINE.get(key)


def _eca_baseline_meets_adapter(game_id: str | None) -> bool:
    """Phase 56 — True iff ECA-only baseline >= adapter ceiling.

    Games meeting this condition (currently sb26 + cd82) are
    "ECA-optimal": Phase 53 K-step replay measured the gate as
    HARMFUL on every frame for these games. Static suppression
    eliminates the harmful fires without affecting any other game.
    """

    eca = _per_game_eca_baseline(game_id)
    adapter = _per_game_adapter_baseline(game_id)
    if eca is None or adapter is None:
        return False
    return int(eca) >= int(adapter)


def _game_baseline_parity_suppression_from_env() -> bool:
    """Phase 56 — opt-in to suppress gate fires for ECA-optimal games.

    Default OFF for backward compatibility. When ON, the gate
    suppresses every fire on games where ECA's baseline already
    matches the adapter ceiling.
    """

    raw = str(
        os.environ.get("V2_ECA_REPAIR_GATE_GAME_BASELINE_PARITY", "off") or ""
    ).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _selective_ood_max_from_env(default: float = 0.5) -> float:
    """Return the Phase 46 selective OOD-max threshold from env."""

    raw = os.environ.get("V2_ECA_REPAIR_GATE_SELECTIVE_OOD_MAX")
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _default_report_search_paths() -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    candidates: list[Path] = []
    env_path = os.environ.get("V2_ECA_REPAIR_GATE_REPORT")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(repo_root / "tests" / _DEFAULT_REPORT_BASENAME)
    for legacy in _LEGACY_REPORT_BASENAMES:
        candidates.append(repo_root / "tests" / legacy)
    return candidates


@dataclass
class RepairDecision:
    """Advisory decision emitted by the repair gate."""

    repair_recommended: bool = False
    objective_mismatch_risk: float = 0.0
    payload_mismatch_risk: float = 0.0
    action_mismatch_risk: float = 0.0
    scoreless_mismatch_risk: float = 0.0
    repaired_action: str = ""
    repaired_payload: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    mode: str = "off"
    payload_disagree: bool = False
    ood_fraction: float = 0.0
    ood_features: list[str] = field(default_factory=list)
    ood_backoff_engaged: bool = False
    # Phase 46 — selective gating diagnostic fields.
    selective_mode: str = "off"
    selective_eligible: bool = False
    selective_blocked_reason: str = ""
    selective_stuck_steps: int = 0
    # Phase 55 — score-aware suppression diagnostic fields.
    score_aware_engaged: bool = False
    score_baseline: int = -1
    score_at_decision: int = -1
    # Phase 56 — static game-baseline parity diagnostic field.
    game_baseline_parity_engaged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_recommended": bool(self.repair_recommended),
            "objective_mismatch_risk": float(self.objective_mismatch_risk),
            "payload_mismatch_risk": float(self.payload_mismatch_risk),
            "action_mismatch_risk": float(self.action_mismatch_risk),
            "scoreless_mismatch_risk": float(self.scoreless_mismatch_risk),
            "repaired_action": str(self.repaired_action or ""),
            "repaired_payload": dict(self.repaired_payload or {}),
            "reason": str(self.reason or ""),
            "mode": str(self.mode or ""),
            "payload_disagree": bool(self.payload_disagree),
            "ood_fraction": float(self.ood_fraction),
            "ood_features": list(self.ood_features or []),
            "ood_backoff_engaged": bool(self.ood_backoff_engaged),
            "selective_mode": str(self.selective_mode or "off"),
            "selective_eligible": bool(self.selective_eligible),
            "selective_blocked_reason": str(self.selective_blocked_reason or ""),
            "selective_stuck_steps": int(self.selective_stuck_steps),
            "score_aware_engaged": bool(self.score_aware_engaged),
            "score_baseline": int(self.score_baseline),
            "score_at_decision": int(self.score_at_decision),
            "game_baseline_parity_engaged": bool(self.game_baseline_parity_engaged),
        }


def _action_name(value: Any) -> str:
    if value is None:
        return ""
    raw = getattr(value, "name", None)
    if raw:
        return str(raw).strip().upper()
    raw = getattr(value, "value", None)
    if raw:
        return str(raw).strip().upper()
    return str(value).strip().upper()


def _coerce_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if value is None:
        return {}
    if isinstance(value, (list, tuple)):
        return {str(i): v for i, v in enumerate(value)}
    return {"value": value}


def _coerce_proposal(proposal: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(proposal, Mapping):
        return {"action": "", "payload": {}, "target_kind": ""}
    payload_raw = (
        proposal.get("payload")
        or proposal.get("parameters")
        or proposal.get("data")
        or {}
    )
    return {
        "action": _action_name(proposal.get("action")),
        "payload": _coerce_payload(payload_raw),
        "target_kind": str(proposal.get("target_kind") or proposal.get("kind") or "").strip().lower(),
    }


def _coerce_advice(advice: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(advice, Mapping):
        return {"action": "", "payload": {}, "target_kind": ""}
    payload_raw = (
        advice.get("payload")
        or advice.get("parameters")
        or advice.get("data")
        or {}
    )
    action = advice.get("action") or advice.get("first_action") or ""
    return {
        "action": _action_name(action),
        "payload": _coerce_payload(payload_raw),
        "target_kind": str(advice.get("target_kind") or advice.get("kind") or "").strip().lower(),
    }


def _build_features(
    proposal: Mapping[str, Any],
    advice: Mapping[str, Any],
    extra_features: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Build a flat string-string feature dict from proposal + advice + extras.

    Surfaces the same keys produced by Phase 41 examples so the
    rule-based scorer can match them directly.
    """

    p = _coerce_proposal(proposal)
    a = _coerce_advice(advice)
    eca_action = p["action"]
    teacher_action = a["action"]
    features: dict[str, str] = {
        "proposed_action": eca_action,
        "teacher_action": teacher_action,
        "eca_action": eca_action,
        "adapter_action": teacher_action,
        "action_pair": f"{eca_action}->{teacher_action}" if (eca_action or teacher_action) else "",
        "proposed_target_kind": p["target_kind"],
        "teacher_target_kind": a["target_kind"],
    }
    if isinstance(extra_features, Mapping):
        for key, value in extra_features.items():
            if value is None:
                continue
            features[str(key)] = str(value)
    return features


@dataclass
class TargetBindingRepairGate:
    """Phase 42 advisory repair gate over ECA target binding.

    Reads ``top_risk_rules`` per label from the Phase 41 report. For
    each label, the risk score is the maximum ``positive_rate`` of any
    rule whose ``(feature, value)`` matches the current example.
    Priors are retained for reporting/load compatibility, but they do
    not trigger repair by themselves.

    Phase 45 adds ``feature_coverage`` — for each feature key, the SET
    of values seen in the trained rule corpus. ``_compute_ood_score``
    flags features whose current value isn't in the trained vocabulary;
    ``ood_fraction`` aggregates the share of features that are OOD.
    OOD is recorded on every decision as a diagnostic flag (advisory
    contract intact). When ``V2_ECA_REPAIR_GATE_OOD_BACKOFF=1`` is set
    AND the gate would otherwise return low-risk-no-rec because no rules
    matched, but ``ood_fraction`` exceeds ``ood_threshold``, the gate
    treats "OOD with disagreement" as a we-don't-know-but-disagreement-
    is-real signal and recommends repair.
    """

    rules_by_label: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    label_priors: dict[str, float] = field(default_factory=dict)
    objective_threshold: float = 0.8
    payload_threshold: float = 0.8
    mode: str = field(default_factory=repair_gate_mode)
    feature_coverage: dict[str, set[str]] = field(default_factory=dict)
    ood_threshold: float = field(default_factory=_ood_threshold_from_env)
    ood_backoff_enabled: bool = field(default_factory=_ood_backoff_enabled)
    # Phase 46 — selective gating fields.
    selective_mode: str = field(default_factory=_selective_mode_from_env)
    stuck_threshold: int = field(default_factory=_selective_stuck_threshold_from_env)
    selective_high_risk_threshold: float = field(
        default_factory=_selective_high_risk_threshold_from_env
    )
    selective_ood_max: float = field(default_factory=_selective_ood_max_from_env)
    # Phase 55 — score-aware suppression.
    score_aware_enabled: bool = field(default_factory=_score_aware_suppression_from_env)
    # Phase 56 — static game-baseline parity suppression.
    game_baseline_parity_enabled: bool = field(
        default_factory=_game_baseline_parity_suppression_from_env
    )

    def __post_init__(self) -> None:
        self.mode = _normalize_mode(self.mode or "")
        if self.mode == "aggressive":
            # Aggressive lowers thresholds modestly so the gate fires more
            # often during shadow experiments while staying advisory.
            self.objective_threshold = min(self.objective_threshold, 0.6)
            self.payload_threshold = min(self.payload_threshold, 0.6)
        # Phase 46 — normalise selective_mode, and lower the high-risk
        # threshold when in aggressive_selective mode.
        sel_mode = str(self.selective_mode or "off").strip().lower()
        if sel_mode in {"selective", "tight", "tight_selective"}:
            self.selective_mode = "selective"
        elif sel_mode in {
            "aggressive_selective",
            "aggressive-selective",
            "aggressive",
        }:
            self.selective_mode = "aggressive_selective"
            # aggressive_selective uses 0.7 risk threshold by default; an
            # explicit threshold from env still wins (factory ran first).
            if self.selective_high_risk_threshold >= 0.9 - 1e-6:
                self.selective_high_risk_threshold = 0.7
        else:
            self.selective_mode = "off"
        # Normalise feature_coverage to a dict of sets if rules supply it
        # via dict-of-list etc.; missing → infer from rules_by_label.
        if not self.feature_coverage:
            self.feature_coverage = self._infer_feature_coverage_from_rules()
        else:
            self.feature_coverage = {
                str(k): set(str(v) for v in (vals or []))
                for k, vals in self.feature_coverage.items()
            }

    def _infer_feature_coverage_from_rules(self) -> dict[str, set[str]]:
        """Build feature_coverage from the rules' (feature, value) pairs.

        This is the trained vocabulary the gate knows about — anything
        outside this set is OOD.
        """

        coverage: dict[str, set[str]] = {}
        for rules in (self.rules_by_label or {}).values():
            if not rules:
                continue
            for rule in rules:
                if not isinstance(rule, Mapping):
                    continue
                feat = str(rule.get("feature") or "")
                val = str(rule.get("value") or "")
                if not feat:
                    continue
                coverage.setdefault(feat, set()).add(val)
        return coverage

    @property
    def enabled(self) -> bool:
        return self.mode in {"advisory", "aggressive"}

    def _label_risk(self, label: str, features: Mapping[str, str]) -> float:
        rules = self.rules_by_label.get(label) or []
        max_rate = 0.0
        for rule in rules:
            feat = str(rule.get("feature") or "")
            val = str(rule.get("value") or "")
            if not feat:
                continue
            if features.get(feat, "") == val:
                rate = float(rule.get("positive_rate") or 0.0)
                if rate > max_rate:
                    max_rate = rate
        return max_rate

    def _any_rule_matches(self, features: Mapping[str, str]) -> bool:
        """Return True iff at least one rule (across all labels) matches.

        Used by OOD-backoff: distinguishes "low risk because rules say so"
        from "low risk because no rule matched at all".
        """

        for rules in (self.rules_by_label or {}).values():
            for rule in rules or []:
                if not isinstance(rule, Mapping):
                    continue
                feat = str(rule.get("feature") or "")
                val = str(rule.get("value") or "")
                if not feat:
                    continue
                if features.get(feat, "") == val:
                    return True
        return False

    def _compute_ood_score(self, features: Mapping[str, str]) -> dict[str, Any]:
        """Compute per-feature OOD flags + aggregate fraction.

        For each feature key that is BOTH (a) populated in the current
        ``features`` dict (non-empty value) AND (b) present in the
        trained ``feature_coverage`` vocabulary, check whether the
        current value is in the trained value set. Features absent from
        the trained vocabulary are skipped (we can't say anything about
        them). Returns:

        * ``per_feature``: ``{feat_key: bool_is_ood}`` only for features
          that were evaluable.
        * ``ood_features``: list of feat_keys that are OOD.
        * ``evaluable_count``: number of features actually checked.
        * ``ood_count``: number of OOD features.
        * ``ood_fraction``: ``ood_count / evaluable_count`` (0.0 if
          ``evaluable_count == 0``).
        """

        per_feature: dict[str, bool] = {}
        ood_features: list[str] = []
        evaluable = 0
        ood_count = 0
        coverage = self.feature_coverage or {}
        if not coverage:
            return {
                "per_feature": per_feature,
                "ood_features": ood_features,
                "evaluable_count": 0,
                "ood_count": 0,
                "ood_fraction": 0.0,
            }
        for feat_key, vocab in coverage.items():
            if not vocab:
                continue
            value = features.get(feat_key, "")
            if value == "":
                # Feature not populated in this evaluation — skip,
                # because OOD only makes sense for observed values.
                continue
            evaluable += 1
            is_ood = value not in vocab
            per_feature[feat_key] = bool(is_ood)
            if is_ood:
                ood_count += 1
                ood_features.append(feat_key)
        ood_fraction = (
            float(ood_count) / float(evaluable) if evaluable else 0.0
        )
        return {
            "per_feature": per_feature,
            "ood_features": ood_features,
            "evaluable_count": int(evaluable),
            "ood_count": int(ood_count),
            "ood_fraction": float(ood_fraction),
        }

    def _selective_eligible(
        self,
        *,
        objective_risk: float,
        payload_risk: float,
        actions_disagree: bool,
        payloads_disagree: bool,
        ood_fraction: float,
        stuck_steps: int,
    ) -> tuple[bool, str]:
        """Phase 46 — selective gating eligibility.

        Returns ``(eligible, blocked_reason)``. ``eligible`` is True iff
        ALL the following hold:

        * ECA is stuck: ``stuck_steps >= stuck_threshold``.
        * Risk is high: ``max(objective, payload) >= selective_high_risk_threshold``.
        * OOD is low: ``ood_fraction <= selective_ood_max``.
        * Either actions or payloads disagree.

        When NOT eligible, ``blocked_reason`` is one of:
        ``stuck_too_short`` / ``risk_too_low`` / ``ood_too_high`` /
        ``not_eligible`` (no disagreement). When eligible, returns
        ``(True, "")``.
        """

        if int(stuck_steps) < int(self.stuck_threshold):
            return False, "stuck_too_short"
        if not (actions_disagree or payloads_disagree):
            return False, "not_eligible"
        max_risk = max(float(objective_risk), float(payload_risk))
        if max_risk < float(self.selective_high_risk_threshold):
            return False, "risk_too_low"
        if float(ood_fraction) > float(self.selective_ood_max):
            return False, "ood_too_high"
        return True, ""

    def evaluate_repair(
        self,
        eca_proposal: Mapping[str, Any] | None,
        teacher_advice: Mapping[str, Any] | None,
        features: Mapping[str, Any] | None = None,
        *,
        stuck_steps: int = 0,
        game_id: str | None = None,
        current_score: int | None = None,
    ) -> RepairDecision:
        if not self.enabled:
            return RepairDecision(mode=self.mode, reason="gate_disabled")

        proposal = _coerce_proposal(eca_proposal)
        advice = _coerce_advice(teacher_advice)
        if not advice.get("action"):
            return RepairDecision(mode=self.mode, reason="no_teacher_advice")

        feature_dict = _build_features(proposal, advice, features)
        objective_risk = self._label_risk("objective_mismatch", feature_dict)
        payload_risk = self._label_risk("payload_mismatch", feature_dict)
        action_risk = self._label_risk("action_mismatch", feature_dict)
        scoreless_risk = self._label_risk("scoreless_mismatch", feature_dict)

        objective_high = objective_risk >= self.objective_threshold
        payload_high = payload_risk >= self.payload_threshold
        actions_disagree = bool(
            proposal.get("action")
            and advice.get("action")
            and proposal.get("action") != advice.get("action")
        )
        teacher_payload = dict(advice.get("payload") or {})
        proposal_payload = dict(proposal.get("payload") or {})
        payloads_disagree = bool(teacher_payload) and proposal_payload != teacher_payload

        # Phase 45 — OOD diagnostic over feature_coverage.
        ood = self._compute_ood_score(feature_dict)
        ood_fraction = float(ood.get("ood_fraction", 0.0))
        ood_features_list = list(ood.get("ood_features") or [])
        ood_high = ood_fraction >= float(self.ood_threshold)
        any_rule_matches = self._any_rule_matches(feature_dict)

        repair_recommended = bool(
            (objective_high or payload_high)
            and (actions_disagree or payloads_disagree)
        )

        # OOD-backoff: when the gate would otherwise NOT recommend (low
        # risk) AND no rule matched the example AND OOD is high AND the
        # actions disagree, treat it as "we don't know but disagreement
        # is real" and recommend repair. Advisory contract intact —
        # this only changes the recommend bit, never mutates ECA's
        # action authoritatively.
        ood_backoff_engaged = False
        if (
            self.ood_backoff_enabled
            and not repair_recommended
            and not any_rule_matches
            and ood_high
            and (actions_disagree or payloads_disagree)
        ):
            repair_recommended = True
            ood_backoff_engaged = True

        # Phase 46 — selective gating layer. Computed AFTER existing
        # logic so the diagnostic always records what the prior phases
        # would have produced. When ``selective_mode`` is on, the
        # selective filter narrows ``repair_recommended`` from the raw
        # 88.5%-rate signal to the smaller subset where ECA is truly
        # stuck AND risk is decisively high AND OOD is consistent. The
        # blocked path still records ``selective_blocked_reason`` so the
        # bridge can build histograms for next-step analysis.
        selective_eligible, selective_blocked_reason = self._selective_eligible(
            objective_risk=objective_risk,
            payload_risk=payload_risk,
            actions_disagree=actions_disagree,
            payloads_disagree=payloads_disagree,
            ood_fraction=ood_fraction,
            stuck_steps=int(stuck_steps),
        )
        if self.selective_mode in {"selective", "aggressive_selective"}:
            if repair_recommended and not selective_eligible:
                repair_recommended = False

        # Phase 55 — score-aware suppression. When enabled AND ECA's
        # current score has met or exceeded the per-game adapter
        # ceiling, the gate suppresses the recommendation entirely
        # (Phase 53 K-step replay showed gate is harmful where ECA
        # already wins). Always advisory; never mutates ECA's path.
        score_above_baseline = False
        score_baseline_used: int | None = None
        if self.score_aware_enabled and current_score is not None:
            baseline = _per_game_adapter_baseline(game_id)
            if baseline is not None:
                score_baseline_used = int(baseline)
                if int(current_score) >= int(baseline):
                    score_above_baseline = True
                    if repair_recommended:
                        repair_recommended = False
                    if not selective_blocked_reason:
                        selective_blocked_reason = "score_above_baseline"

        # Phase 56 — static game-baseline parity suppression. When
        # enabled AND the game is "ECA-optimal" (ECA-only baseline
        # already matches adapter ceiling), suppress every fire on
        # this game unconditionally. This generalises Phase 55's
        # live-score check: cd82/sb26 have eca==adapter==WIN, so
        # the gate cannot improve score on any frame (Phase 53
        # K=10 measured -0.83 / -0.62 lift_avg).
        game_baseline_parity_engaged = False
        if self.game_baseline_parity_enabled and _eca_baseline_meets_adapter(game_id):
            game_baseline_parity_engaged = True
            if repair_recommended:
                repair_recommended = False
            if not selective_blocked_reason:
                selective_blocked_reason = "game_baseline_parity"

        reason_parts: list[str] = []
        if objective_high:
            reason_parts.append(f"objective_risk={objective_risk:.3f}>={self.objective_threshold:.2f}")
        if payload_high:
            reason_parts.append(f"payload_risk={payload_risk:.3f}>={self.payload_threshold:.2f}")
        if payloads_disagree:
            reason_parts.append("payloads_disagree")
        if not actions_disagree and not payloads_disagree:
            reason_parts.append("actions_match")
        if ood_high:
            reason_parts.append(
                f"ood_high (frac={ood_fraction:.3f}>={self.ood_threshold:.2f})"
            )
        if ood_backoff_engaged:
            reason_parts.append("ood_backoff_engaged")
        if self.selective_mode in {"selective", "aggressive_selective"}:
            reason_parts.append(f"selective_mode={self.selective_mode}")
            if selective_eligible:
                reason_parts.append(
                    f"selective_eligible (stuck={int(stuck_steps)}>={int(self.stuck_threshold)})"
                )
            elif selective_blocked_reason:
                reason_parts.append(
                    f"selective_blocked={selective_blocked_reason}"
                )
        if score_above_baseline:
            reason_parts.append(
                f"score_above_baseline (score={int(current_score)}>={score_baseline_used})"
            )
        if game_baseline_parity_engaged:
            reason_parts.append(
                f"game_baseline_parity (eca_baseline>=adapter_baseline for {game_id})"
            )
        if not reason_parts:
            reason_parts.append("low_risk")

        return RepairDecision(
            repair_recommended=repair_recommended,
            objective_mismatch_risk=objective_risk,
            payload_mismatch_risk=payload_risk,
            action_mismatch_risk=action_risk,
            scoreless_mismatch_risk=scoreless_risk,
            repaired_action=advice.get("action", "") if repair_recommended else "",
            repaired_payload=teacher_payload if repair_recommended and teacher_payload else {},
            reason="; ".join(reason_parts),
            mode=self.mode,
            payload_disagree=payloads_disagree,
            ood_fraction=ood_fraction,
            ood_features=ood_features_list,
            ood_backoff_engaged=ood_backoff_engaged,
            selective_mode=str(self.selective_mode or "off"),
            selective_eligible=bool(selective_eligible),
            selective_blocked_reason=str(selective_blocked_reason or ""),
            selective_stuck_steps=int(stuck_steps),
            score_aware_engaged=bool(score_above_baseline),
            score_baseline=int(score_baseline_used) if score_baseline_used is not None else -1,
            score_at_decision=int(current_score) if current_score is not None else -1,
            game_baseline_parity_engaged=bool(game_baseline_parity_engaged),
        )


def _load_rules_from_report(report_path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    """Extract ``rules_by_label`` and ``label_priors`` from a Phase 41 report JSON."""

    rules: dict[str, list[dict[str, Any]]] = {}
    priors: dict[str, float] = {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return rules, priors

    models = report.get("models") or {}
    if not isinstance(models, Mapping):
        return rules, priors

    for label in _LABELS:
        entry = models.get(label)
        if not isinstance(entry, Mapping):
            continue
        # Top risk rules carry per-feature value lifts above the prior.
        top_risk = entry.get("top_risk_rules") or []
        if isinstance(top_risk, list):
            rules[label] = [dict(r) for r in top_risk if isinstance(r, Mapping)]
        in_sample = entry.get("in_sample") or {}
        if isinstance(in_sample, Mapping):
            try:
                priors[label] = float(in_sample.get("positive_rate") or 0.0)
            except (TypeError, ValueError):
                priors[label] = 0.0
    return rules, priors


def _build_feature_coverage_from_rules(
    rules_by_label: Mapping[str, list[dict[str, Any]]],
) -> dict[str, set[str]]:
    """Phase 45 — collect the trained ``(feature, value)`` vocabulary."""

    coverage: dict[str, set[str]] = {}
    for rules in (rules_by_label or {}).values():
        for rule in rules or []:
            if not isinstance(rule, Mapping):
                continue
            feat = str(rule.get("feature") or "")
            val = str(rule.get("value") or "")
            if not feat:
                continue
            coverage.setdefault(feat, set()).add(val)
    return coverage


def load_default_repair_gate(
    *,
    mode: str | None = None,
    objective_threshold: float | None = None,
    payload_threshold: float | None = None,
    selective_mode: str | None = None,
    stuck_threshold: int | None = None,
    selective_high_risk_threshold: float | None = None,
    selective_ood_max: float | None = None,
) -> TargetBindingRepairGate:
    """Construct a repair gate from the bundled Phase 41 report.

    Falls back to an empty rule-set if the report file is missing;
    the gate will still respect ``mode=off`` and emit no-op
    decisions.
    """

    rules: dict[str, list[dict[str, Any]]] = {}
    priors: dict[str, float] = {}
    for path in _default_report_search_paths():
        try:
            if path.is_file():
                rules, priors = _load_rules_from_report(path)
                if rules:
                    break
        except Exception:  # pragma: no cover - defensive
            continue

    feature_coverage = _build_feature_coverage_from_rules(rules)

    kwargs: dict[str, Any] = {
        "rules_by_label": rules,
        "label_priors": priors,
        "mode": mode or repair_gate_mode(),
        "feature_coverage": feature_coverage,
    }
    if objective_threshold is not None:
        kwargs["objective_threshold"] = float(objective_threshold)
    if payload_threshold is not None:
        kwargs["payload_threshold"] = float(payload_threshold)
    if selective_mode is not None:
        kwargs["selective_mode"] = str(selective_mode)
    if stuck_threshold is not None:
        kwargs["stuck_threshold"] = int(stuck_threshold)
    if selective_high_risk_threshold is not None:
        kwargs["selective_high_risk_threshold"] = float(selective_high_risk_threshold)
    if selective_ood_max is not None:
        kwargs["selective_ood_max"] = float(selective_ood_max)
    return TargetBindingRepairGate(**kwargs)
