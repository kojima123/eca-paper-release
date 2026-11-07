"""Offline learner for ECA/adapter mismatch diagnostics.

This module is intentionally lightweight. It does not train an action policy.
It learns where ECA's current ControlIR/action choice tends to disagree with a
per-game adapter teacher, so those disagreements can become concrete training
targets for later target-binding work.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .perturbation_benchmark import (
    _as_int,
    _eca_objective_from_control_target,
    _normalize_action_name,
    _objective_operator_matches,
    _safe_div,
    _teacher_objective_from_advice,
)


LABEL_KEYS = (
    "action_mismatch",
    "payload_mismatch",
    "objective_mismatch",
    "scoreless_mismatch",
)

FEATURE_KEYS = (
    "game_id",
    "mode",
    "variant",
    "phase",
    "primitive",
    "solver_family",
    "control_solver_family",
    "control_operator",
    "target_translation_status",
    "eca_action",
    "adapter_action",
    "action_pair",
    "operator_pair",
    "plan_source",
    "advice_used",
)

TRANSFER_FEATURE_KEYS = (
    "phase",
    "primitive",
    "solver_family",
    "control_solver_family",
    "control_operator",
    "target_translation_status",
    "eca_action",
    "adapter_action",
    "action_pair",
    "operator_pair",
)

REPAIR_STUDENT_FEATURE_KEY_GROUPS = (
    (
        "game_id",
        "selected_target_component_id",
        "proposal_visible_click_candidate_kind",
        "proposal_visible_click_candidate_rank",
        "remaining_visible_click_candidate_count",
    ),
    (
        "game_id",
        "selected_target_anchor",
        "proposal_visible_click_candidate_kind",
        "proposal_visible_click_candidate_distance",
        "tried_click_sites_count",
    ),
    (
        "game_id",
        "selected_target_component_id",
        "visible_click_candidate_count",
        "tried_click_sites_count",
    ),
    (
        "game_id",
        "selected_target_component_id",
        "selected_target_anchor",
        "target_translation_status",
    ),
    (
        "game_id",
        "selected_target_source",
        "selected_target_center_row",
        "selected_target_center_col",
    ),
    (
        "game_id",
        "selected_target_component_id",
    ),
    (
        "game_id",
        "selected_target_anchor",
    ),
    (
        "game_id",
        "selected_target_component_id",
        "selected_target_recent_churn_selected_count",
        "score",
    ),
    (
        "game_id",
        "selected_target_component_id",
        "step_index_bucket",
    ),
    (
        "game_id",
        "selected_target_anchor",
        "selected_target_recent_churn_selected_count",
    ),
    (
        "action_pair",
        "target_translation_status",
        "control_operator",
        "operator_pair",
    ),
    ("action_pair", "target_translation_status"),
    ("action_pair",),
    ("target_translation_status", "control_operator"),
    ("control_operator",),
)

REPAIR_CANDIDATE_RANKER_FEATURE_KEY_GROUPS = (
    (
        "game_id",
        "score",
        "candidate_band",
        "candidate_kind_base",
        "candidate_source",
    ),
    (
        "game_id",
        "score",
        "candidate_kind_base",
        "candidate_source",
        "candidate_previously_executed",
        "candidate_dx_bucket",
        "candidate_dy_bucket",
    ),
    (
        "game_id",
        "score",
        "candidate_kind_base",
        "candidate_source",
        "candidate_memory_seen_bucket",
    ),
    (
        "game_id",
        "score",
        "candidate_kind_base",
        "candidate_dx",
        "candidate_dy",
        "selected_target_component_id",
    ),
    (
        "game_id",
        "score",
        "candidate_kind_base",
        "candidate_dx",
        "candidate_dy",
    ),
    (
        "game_id",
        "candidate_kind_base",
        "candidate_dx",
        "candidate_dy",
    ),
    (
        "candidate_kind_base",
        "candidate_dx",
        "candidate_dy",
    ),
    (
        "candidate_kind_base",
        "candidate_color",
        "candidate_manhattan_bucket",
    ),
    (
        "candidate_kind_base",
        "candidate_distance_to_proposal_bucket",
    ),
    ("candidate_band", "candidate_kind_base"),
    ("candidate_kind_base",),
)

REPAIR_PRIOR_CHAIN_VALUE_FEATURE_KEY_GROUPS = (
    (
        "game_id",
        "score",
        "candidate_kind",
        "candidate_band",
        "prior_score_bucket",
        "x_bucket",
        "y_bucket",
    ),
    (
        "score",
        "candidate_kind",
        "candidate_band",
        "prior_score_bucket",
        "x_bucket",
        "y_bucket",
    ),
    (
        "score",
        "candidate_kind",
        "candidate_band",
        "prior_score_bucket",
        "candidate_index_bucket",
    ),
    ("game_id", "score", "candidate_kind", "candidate_band"),
    ("game_id", "score", "candidate_kind"),
    ("score", "candidate_kind", "candidate_band"),
    ("score", "candidate_kind"),
    ("candidate_kind", "candidate_band"),
    ("candidate_kind",),
    ("candidate_band",),
)

REPAIR_CANDIDATE_PROTOTYPE_FEATURE_KEY_GROUPS = (
    (
        "game_id",
        "score",
        "candidate_band",
        "candidate_kind_base",
        "candidate_color",
    ),
    (
        "game_id",
        "score",
        "candidate_kind_base",
        "candidate_color",
        "candidate_target_dx_bucket",
        "candidate_target_dy_bucket",
    ),
    (
        "score",
        "candidate_kind_base",
        "candidate_color",
        "candidate_target_dx_bucket",
        "candidate_target_dy_bucket",
    ),
    (
        "score",
        "candidate_kind_base",
        "candidate_target_dx_bucket",
        "candidate_target_dy_bucket",
    ),
    (
        "game_id",
        "score",
        "candidate_kind_base",
        "candidate_color",
    ),
    (
        "game_id",
        "score",
        "candidate_kind_base",
    ),
    (
        "score",
        "candidate_kind_base",
        "candidate_color",
    ),
    (
        "score",
        "candidate_kind_base",
    ),
    (
        "candidate_kind_base",
        "candidate_color",
    ),
    ("candidate_band", "candidate_kind_base"),
    ("candidate_kind_base",),
)


def _control_contract(control_target: Any) -> Mapping[str, Any]:
    if not isinstance(control_target, Mapping):
        return {}
    contract = control_target.get("goal_target_contract")
    if isinstance(contract, Mapping):
        return contract
    return {}


def _objective_mismatch(eca_objective: Mapping[str, Any], teacher_objective: Mapping[str, Any]) -> bool:
    if not teacher_objective:
        return False
    if not _objective_operator_matches(eca_objective, teacher_objective):
        return True
    eca_site = eca_objective.get("site")
    teacher_site = teacher_objective.get("site")
    if isinstance(eca_site, list) and isinstance(teacher_site, list):
        return eca_site != teacher_site
    if teacher_site is not None and eca_site is None:
        return True
    return False


def extract_adapter_mismatch_examples(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build supervised mismatch examples from perturbation rows."""
    examples: list[dict[str, Any]] = []
    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown")
        variant = str(row.get("variant", "") or "unknown")
        mode = str(row.get("adapter_advice_mode", "") or "none")
        final_score = _as_int(row.get("final_score"), 0)
        trace = row.get("trace")
        if not isinstance(trace, list):
            continue
        for item in trace:
            if not isinstance(item, Mapping):
                continue
            advice = item.get("adapter_advice")
            if not isinstance(advice, Mapping) or not bool(advice.get("available")):
                continue
            control_target = item.get("control_target")
            contract = _control_contract(control_target)
            eca_action = _normalize_action_name(advice.get("eca_action") or item.get("action"))
            adapter_action = _normalize_action_name(advice.get("action"))
            teacher_objective = _teacher_objective_from_advice(advice)
            eca_objective = _eca_objective_from_control_target(control_target, eca_action)
            eca_operator = str(eca_objective.get("operator", "") or "")
            teacher_operator = str(teacher_objective.get("operator", "") or "")
            labels = {
                "action_mismatch": bool("action_agreed" in advice and not bool(advice.get("action_agreed"))),
                "payload_mismatch": bool("payload_agreed" in advice and not bool(advice.get("payload_agreed"))),
                "objective_mismatch": _objective_mismatch(eca_objective, teacher_objective),
                "scoreless_mismatch": final_score <= 0
                and bool(
                    ("action_agreed" in advice and not bool(advice.get("action_agreed")))
                    or ("payload_agreed" in advice and not bool(advice.get("payload_agreed")))
                    or _objective_mismatch(eca_objective, teacher_objective)
                ),
            }
            features = {
                "game_id": game_id,
                "mode": mode,
                "variant": variant,
                "phase": str(item.get("phase", "") or ""),
                "primitive": str(item.get("primitive", "") or ""),
                "solver_family": str(item.get("solver_family", "") or ""),
                "control_solver_family": str(item.get("control_solver_family", "") or ""),
                "control_operator": eca_operator,
                "target_translation_status": str(
                    contract.get("target_translation_status", "") or ""
                ),
                "eca_action": eca_action,
                "adapter_action": adapter_action,
                "action_pair": f"{eca_action}->{adapter_action}",
                "operator_pair": f"{eca_operator}->{teacher_operator}",
                "plan_source": str(advice.get("plan_source", "") or ""),
                "advice_used": "true" if bool(advice.get("used")) else "false",
            }
            examples.append(
                {
                    "schema_version": "eca_adapter_mismatch_example.v1",
                    "game_id": game_id,
                    "variant": variant,
                    "mode": mode,
                    "step": _as_int(item.get("step"), 0),
                    "features": features,
                    "labels": labels,
                    "eca_objective": eca_objective,
                    "teacher_objective": teacher_objective,
                }
            )
    return examples


def _feature_pairs(
    example: Mapping[str, Any],
    *,
    feature_keys: Iterable[str] = FEATURE_KEYS,
) -> list[tuple[str, str]]:
    features = dict(example.get("features") or {})
    pairs: list[tuple[str, str]] = []
    for key in feature_keys:
        value = str(features.get(key, "") or "")
        if value:
            pairs.append((key, value))
    return pairs


def _label(example: Mapping[str, Any], label_key: str) -> bool:
    labels = dict(example.get("labels") or {})
    return bool(labels.get(label_key))


@dataclass
class MismatchFrequencyModel:
    """Smoothed categorical frequency model for one mismatch label."""

    label_key: str
    feature_keys: tuple[str, ...] = TRANSFER_FEATURE_KEYS
    alpha: float = 1.0
    positive_count: int = 0
    example_count: int = 0
    feature_counts: dict[tuple[str, str], Counter[str]] = field(default_factory=dict)

    @property
    def positive_rate(self) -> float:
        return _safe_div(self.positive_count, self.example_count)

    def fit(self, examples: Iterable[Mapping[str, Any]]) -> "MismatchFrequencyModel":
        for example in examples:
            positive = _label(example, self.label_key)
            self.example_count += 1
            if positive:
                self.positive_count += 1
            for pair in _feature_pairs(example, feature_keys=self.feature_keys):
                stats = self.feature_counts.setdefault(pair, Counter())
                stats["total"] += 1
                if positive:
                    stats["positive"] += 1
        return self

    def _feature_rate(self, pair: tuple[str, str]) -> float:
        stats = self.feature_counts.get(pair)
        if not stats:
            return self.positive_rate
        total = int(stats.get("total", 0))
        positive = int(stats.get("positive", 0))
        prior_positive = self.positive_rate * self.alpha
        return _safe_div(positive + prior_positive, total + self.alpha)

    def predict_proba(self, example: Mapping[str, Any]) -> float:
        pairs = _feature_pairs(example, feature_keys=self.feature_keys)
        if not pairs:
            return self.positive_rate
        rates = [self._feature_rate(pair) for pair in pairs if pair in self.feature_counts]
        if not rates:
            return self.positive_rate
        return round(sum(rates) / len(rates), 6)

    def _rules(self, *, min_count: int = 8) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        baseline = self.positive_rate
        for (feature, value), stats in self.feature_counts.items():
            total = int(stats.get("total", 0))
            if total < min_count:
                continue
            positive = int(stats.get("positive", 0))
            rate = _safe_div(positive, total)
            rows.append(
                {
                    "feature": feature,
                    "value": value,
                    "count": total,
                    "positive_count": positive,
                    "positive_rate": rate,
                    "lift": round(rate - baseline, 6),
                }
            )
        return rows

    def top_rules(self, *, min_count: int = 8, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._rules(min_count=min_count)
        return sorted(rows, key=lambda row: (-abs(float(row["lift"])), -int(row["count"]), row["feature"]))[:limit]

    def top_risk_rules(self, *, min_count: int = 8, limit: int = 20) -> list[dict[str, Any]]:
        rows = [row for row in self._rules(min_count=min_count) if float(row["lift"]) > 0.0]
        return sorted(rows, key=lambda row: (-float(row["lift"]), -int(row["count"]), row["feature"]))[:limit]

    def top_safe_rules(self, *, min_count: int = 8, limit: int = 20) -> list[dict[str, Any]]:
        rows = [row for row in self._rules(min_count=min_count) if float(row["lift"]) < 0.0]
        return sorted(rows, key=lambda row: (float(row["lift"]), -int(row["count"]), row["feature"]))[:limit]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _counter_entropy(counter: Counter[str]) -> float:
    total = sum(int(value) for value in counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counter.values():
        p = float(value) / float(total)
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 6)


def _repair_student_target(example: Mapping[str, Any]) -> dict[str, Any]:
    repair = dict(example.get("repair") or {})
    teacher = dict(example.get("teacher") or {})
    action = str(repair.get("action", "") or teacher.get("action", "") or "")
    payload = repair.get("payload")
    if not isinstance(payload, Mapping) or not payload:
        payload = teacher.get("payload")
    return {
        "action": action,
        "payload": dict(payload or {}),
    }


def _repair_student_feature_key(
    example: Mapping[str, Any],
    keys: Iterable[str],
) -> tuple[tuple[str, str], ...] | None:
    features = dict(example.get("features") or {})
    parts: list[tuple[str, str]] = []
    for key in keys:
        value = str(example.get(key, "") or "") if key == "game_id" else str(features.get(key, "") or "")
        if not value:
            return None
        parts.append((str(key), value))
    return tuple(parts)


def _repair_click_point(payload: Any) -> tuple[int, int] | None:
    if not isinstance(payload, Mapping):
        return None
    raw = dict(payload)
    if isinstance(raw.get("parameters"), Mapping):
        raw = dict(raw.get("parameters") or {})
    if "x" in raw and "y" in raw:
        try:
            return (int(raw["x"]), int(raw["y"]))
        except (TypeError, ValueError):
            return None
    return None


@dataclass
class RepairPayloadStudent:
    """Frequency student for teacher-free repair target shadow scoring."""

    feature_key_groups: tuple[tuple[str, ...], ...] = REPAIR_STUDENT_FEATURE_KEY_GROUPS
    target_counts: dict[tuple[int, tuple[tuple[str, str], ...]], Counter[str]] = field(
        default_factory=dict
    )
    default_counts: Counter[str] = field(default_factory=Counter)

    def fit(self, examples: Iterable[Mapping[str, Any]]) -> "RepairPayloadStudent":
        for example in examples:
            target = _repair_student_target(example)
            if not target.get("action"):
                continue
            target_key = _canonical_json(target)
            self.default_counts[target_key] += 1
            for index, keys in enumerate(self.feature_key_groups):
                feature_key = _repair_student_feature_key(example, keys)
                if feature_key is None:
                    continue
                self.target_counts.setdefault((index, feature_key), Counter())[target_key] += 1
        return self

    def _decode_target(self, target_key: str) -> dict[str, Any]:
        try:
            loaded = json.loads(target_key)
        except Exception:
            return {}
        return dict(loaded) if isinstance(loaded, Mapping) else {}

    def _best_target(self, counter: Counter[str]) -> tuple[str, int]:
        if not counter:
            return "", 0
        target_key, support = sorted(
            counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[0]
        return str(target_key), int(support)

    def predict(self, example: Mapping[str, Any]) -> dict[str, Any]:
        for index, keys in enumerate(self.feature_key_groups):
            feature_key = _repair_student_feature_key(example, keys)
            if feature_key is None:
                continue
            counter = self.target_counts.get((index, feature_key))
            if not counter:
                continue
            target_key, support = self._best_target(counter)
            return {
                "available": True,
                "target": self._decode_target(target_key),
                "support": support,
                "feature_keys": list(keys),
            }
        target_key, support = self._best_target(self.default_counts)
        if not target_key:
            return {"available": False, "target": {}, "support": 0, "feature_keys": []}
        return {
            "available": True,
            "target": self._decode_target(target_key),
            "support": support,
            "feature_keys": ["default"],
        }

    def top_rules(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (index, feature_key), counter in self.target_counts.items():
            target_key, support = self._best_target(counter)
            rows.append(
                {
                    "feature_keys": list(self.feature_key_groups[index]),
                    "feature_values": dict(feature_key),
                    "support": int(support),
                    "target": self._decode_target(target_key),
                    "target_entropy": _counter_entropy(counter),
                    "distinct_targets": len(counter),
                }
            )
        rows.sort(
            key=lambda row: (
                -int(row["support"]),
                float(row["target_entropy"]),
                _canonical_json(row["feature_values"]),
            )
        )
        return rows[:limit]


def _auc(labels: list[bool], scores: list[float]) -> float:
    positives = [(score, idx) for idx, (label, score) in enumerate(zip(labels, scores)) if label]
    negatives = [(score, idx) for idx, (label, score) in enumerate(zip(labels, scores)) if not label]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            total += 1
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return _safe_div(wins, total)


def _brier(labels: list[bool], scores: list[float]) -> float:
    if not labels:
        return 0.0
    total = 0.0
    for label, score in zip(labels, scores):
        target = 1.0 if label else 0.0
        total += (float(score) - target) ** 2
    return round(total / len(labels), 6)


def _top_fraction_positive_rate(labels: list[bool], scores: list[float], fraction: float = 0.10) -> float:
    if not labels:
        return 0.0
    take = max(1, int(math.ceil(len(labels) * fraction)))
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)[:take]
    return _safe_div(sum(1 for _score, label in ranked if label), len(ranked))


def _evaluate_predictions(examples: list[Mapping[str, Any]], label_key: str, scores: list[float]) -> dict[str, Any]:
    labels = [_label(example, label_key) for example in examples]
    positive_count = sum(1 for label in labels if label)
    return {
        "example_count": len(examples),
        "positive_count": positive_count,
        "positive_rate": _safe_div(positive_count, len(examples)),
        "auc": _auc(labels, scores),
        "brier": _brier(labels, scores),
        "top_decile_positive_rate": _top_fraction_positive_rate(labels, scores),
    }


def evaluate_mismatch_frequency_model(
    examples: Iterable[Mapping[str, Any]],
    *,
    label_key: str,
    feature_keys: Iterable[str] = TRANSFER_FEATURE_KEYS,
) -> dict[str, Any]:
    example_list = [dict(example) for example in examples]
    key_tuple = tuple(feature_keys)
    model = MismatchFrequencyModel(label_key=label_key, feature_keys=key_tuple).fit(example_list)
    in_sample_scores = [model.predict_proba(example) for example in example_list]

    games = sorted({str(example.get("game_id", "") or "unknown") for example in example_list})
    loo_examples: list[Mapping[str, Any]] = []
    loo_scores: list[float] = []
    for game in games:
        train = [example for example in example_list if str(example.get("game_id", "") or "unknown") != game]
        test = [example for example in example_list if str(example.get("game_id", "") or "unknown") == game]
        if not train or not test:
            continue
        fold_model = MismatchFrequencyModel(label_key=label_key, feature_keys=key_tuple).fit(train)
        loo_examples.extend(test)
        loo_scores.extend(fold_model.predict_proba(example) for example in test)

    return {
        "label_key": label_key,
        "feature_keys": list(key_tuple),
        "in_sample": _evaluate_predictions(example_list, label_key, in_sample_scores),
        "leave_one_game_out": _evaluate_predictions(loo_examples, label_key, loo_scores),
        "top_rules": model.top_rules(),
        "top_risk_rules": model.top_risk_rules(),
        "top_safe_rules": model.top_safe_rules(),
    }


def build_adapter_mismatch_learning_report(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    examples = extract_adapter_mismatch_examples(rows)
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        game_id = str(example.get("game_id", "") or "unknown")
        by_game[game_id]["example_count"] += 1
        for label_key in LABEL_KEYS:
            if _label(example, label_key):
                by_game[game_id][label_key] += 1

    return {
        "schema_version": "eca_adapter_mismatch_learning_report.v1",
        "example_count": len(examples),
        "feature_keys": list(FEATURE_KEYS),
        "transfer_feature_keys": list(TRANSFER_FEATURE_KEYS),
        "label_keys": list(LABEL_KEYS),
        "by_game": {
            game_id: {
                "example_count": int(stats.get("example_count", 0)),
                **{
                    f"{label_key}_rate": _safe_div(
                        int(stats.get(label_key, 0)),
                        int(stats.get("example_count", 0)),
                    )
                    for label_key in LABEL_KEYS
                },
            }
            for game_id, stats in sorted(by_game.items())
        },
        "models": {
            label_key: evaluate_mismatch_frequency_model(examples, label_key=label_key)
            for label_key in LABEL_KEYS
        },
        "diagnostic_models": {
            label_key: evaluate_mismatch_frequency_model(
                examples,
                label_key=label_key,
                feature_keys=FEATURE_KEYS,
            )
            for label_key in LABEL_KEYS
        },
    }


def extract_repair_distillation_examples(
    episode_summaries: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract compact teacher-assisted repair examples from episode summaries.

    These examples are not score evidence. They are supervision targets for
    later ECA-side target-binding/ranking work: proposal vs teacher, repair
    decision, and before/after agreement labels.
    """

    examples: list[dict[str, Any]] = []
    for summary in episode_summaries:
        if not isinstance(summary, Mapping):
            continue
        bridge_summary = summary.get("perception_solver_bridge")
        if not isinstance(bridge_summary, Mapping):
            bridge_summary = summary
        samples = bridge_summary.get("repair_distillation_samples")
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            if str(sample.get("schema_version", "") or "") != "eca_repair_distillation_example.v1":
                continue
            examples.append(dict(sample))
    return examples


def _repair_prior_effect_key(outcome: Mapping[str, Any]) -> str:
    params = dict(outcome.get("parameters") or {})
    return _canonical_json(
        {
            "action": str(outcome.get("action", "") or ""),
            "x": params.get("x"),
            "y": params.get("y"),
            "pre": outcome.get("pre_observed_step"),
            "post": outcome.get("post_observed_step"),
            "delta": outcome.get("score_delta"),
        }
    )


def _repair_prior_effect_example_from_outcome(
    outcome: Mapping[str, Any],
    *,
    game_id: str,
    source: str = "executed_prior_effect",
    parent_outcome: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    proposal = dict(outcome.get("proposal") or {})
    candidate = dict(proposal.get("candidate") or outcome.get("candidate") or {})
    parameters = dict(outcome.get("parameters") or {})
    pre_score = _as_int(outcome.get("pre_score"), _as_int(proposal.get("score_context"), 0))
    score_delta = _as_int(outcome.get("score_delta"), 0)
    score_progress = bool(score_delta > 0 or str(outcome.get("outcome_kind", "") or "") == "score_progress")
    delayed_credit = bool(outcome.get("delayed_score_progress_credit"))
    if source == "credited_prior_visible_change":
        delayed_credit = True
    features = {
        "game_id": str(game_id or ""),
        "score": str(pre_score),
        "candidate_kind": str(candidate.get("kind", "") or ""),
        "candidate_band": str(candidate.get("band", candidate.get("candidate_band", "")) or ""),
        "candidate_source": str(candidate.get("source", candidate.get("candidate_source", "")) or ""),
        "candidate_index_bucket": _bucket_int(_as_int(proposal.get("candidate_index"), -1), width=4)
        if _as_int(proposal.get("candidate_index"), -1) >= 0
        else "",
        "prior_score_bucket": _bucket_int(int(round(float(proposal.get("top_score", 0.0) or 0.0) * 100)), width=10),
        "x_bucket": _bucket_int(_as_int(parameters.get("x"), 0), width=8),
        "y_bucket": _bucket_int(_as_int(parameters.get("y"), 0), width=8),
    }
    labels = {
        "visible_change": bool(
            str(outcome.get("outcome_kind", "") or "") in {"visible_change", "score_progress"}
            or _as_int(outcome.get("changed_pixel_count"), 0) > 0
        ),
        "score_progress": score_progress,
        "delayed_score_progress_credit": delayed_credit,
        "chain_positive": bool(score_progress or delayed_credit),
    }
    parent = dict(parent_outcome or {})
    return {
        "schema_version": "eca_repair_prior_chain_value_example.v1",
        "game_id": str(game_id or ""),
        "source": source,
        "action": str(outcome.get("action", "") or ""),
        "parameters": parameters,
        "candidate": candidate,
        "features": features,
        "labels": labels,
        "outcome": {
            "outcome_kind": str(outcome.get("outcome_kind", "") or ""),
            "score_delta": int(score_delta),
            "pre_score": int(pre_score),
            "post_score": _as_int(outcome.get("post_score"), pre_score + score_delta),
            "pre_observed_step": _as_int(outcome.get("pre_observed_step"), 0),
            "post_observed_step": _as_int(outcome.get("post_observed_step"), 0),
        },
        "parent_score_progress_parameters": dict(parent.get("parameters") or {}),
    }


def extract_repair_prior_chain_value_examples(
    episode_summaries: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract teacher-free prior execution outcomes for chain-value learning.

    These examples are action-effect evidence. Unlike repair distillation
    examples, they do not use adapter payloads as labels; positives come from
    score progress or delayed credit assigned to preceding visible-change
    actions.
    """

    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for summary in episode_summaries:
        if not isinstance(summary, Mapping):
            continue
        bridge_summary = summary.get("perception_solver_bridge")
        if not isinstance(bridge_summary, Mapping):
            bridge_summary = summary
        game_id = str(bridge_summary.get("game_id", "") or summary.get("game_id", "") or "unknown")
        raw_outcomes = bridge_summary.get("repair_prior_effect_samples")
        if not isinstance(raw_outcomes, list) or not raw_outcomes:
            raw_outcomes = bridge_summary.get("repair_prior_effect_recent_outcomes")
        outcomes = [
            outcome
            for outcome in list(raw_outcomes or [])
            if isinstance(outcome, Mapping)
        ]
        last = bridge_summary.get("repair_prior_effect_last_outcome")
        if isinstance(last, Mapping):
            outcomes.append(last)
        for outcome in outcomes:
            if str(outcome.get("schema_version", "") or "") != "eca_repair_prior_action_effect_outcome.v1":
                continue
            key = _repair_prior_effect_key(outcome)
            if key not in seen:
                seen.add(key)
                examples.append(
                    _repair_prior_effect_example_from_outcome(
                        outcome,
                        game_id=game_id,
                    )
                )
            for credited in list(outcome.get("credited_prior_visible_changes") or []):
                if not isinstance(credited, Mapping):
                    continue
                credit_outcome = {
                    **dict(credited),
                    "action": str(credited.get("action", "") or "ACTION6"),
                    "outcome_kind": "visible_change",
                    "score_delta": 0,
                    "pre_score": _as_int(outcome.get("pre_score"), 0),
                    "post_score": _as_int(outcome.get("pre_score"), 0),
                    "proposal": {
                        "candidate": dict(credited.get("candidate") or {}),
                        "parameters": dict(credited.get("parameters") or {}),
                    },
                    "delayed_score_progress_credit": True,
                }
                credit_base_key = _repair_prior_effect_key(credit_outcome)
                if credit_base_key in seen:
                    continue
                credit_key = "credit:" + credit_base_key
                if credit_key in seen:
                    continue
                seen.add(credit_key)
                examples.append(
                    _repair_prior_effect_example_from_outcome(
                        credit_outcome,
                        game_id=game_id,
                        source="credited_prior_visible_change",
                        parent_outcome=outcome,
                    )
                )
    return examples


def _repair_prior_chain_feature_key(
    features: Mapping[str, Any],
    keys: Iterable[str],
) -> tuple[tuple[str, str], ...] | None:
    parts: list[tuple[str, str]] = []
    for key in keys:
        value = str(features.get(key, "") or "")
        if not value:
            return None
        parts.append((str(key), value))
    return tuple(parts)


def _repair_prior_chain_rule_rows(
    examples: list[Mapping[str, Any]],
    *,
    feature_key_groups: tuple[tuple[str, ...], ...],
    limit: int = 30,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, tuple[tuple[str, str], ...]], Counter[str]] = {}
    for example in examples:
        features = dict(example.get("features") or {})
        labels = dict(example.get("labels") or {})
        for keys in feature_key_groups:
            feature_key = _repair_prior_chain_feature_key(features, keys)
            if feature_key is None:
                continue
            bucket_key = ("|".join(keys), feature_key)
            counter = buckets.setdefault(bucket_key, Counter())
            counter["support"] += 1
            if bool(labels.get("chain_positive")):
                counter["chain_positive"] += 1
            if bool(labels.get("score_progress")):
                counter["score_progress"] += 1
            if bool(labels.get("delayed_score_progress_credit")):
                counter["delayed_credit"] += 1
            if bool(labels.get("visible_change")):
                counter["visible_change"] += 1
    rows: list[dict[str, Any]] = []
    for (group_name, feature_key), counter in buckets.items():
        support = int(counter.get("support", 0))
        if support <= 0:
            continue
        chain_positive = int(counter.get("chain_positive", 0))
        rows.append(
            {
                "feature_group": group_name,
                "feature_values": dict(feature_key),
                "support": support,
                "chain_positive_count": chain_positive,
                "score_progress_count": int(counter.get("score_progress", 0)),
                "delayed_credit_count": int(counter.get("delayed_credit", 0)),
                "visible_change_count": int(counter.get("visible_change", 0)),
                "chain_positive_rate": _safe_div(chain_positive, support),
                "score_progress_rate": _safe_div(
                    int(counter.get("score_progress", 0)),
                    support,
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row.get("chain_positive_rate", 0.0) or 0.0),
            -int(row.get("chain_positive_count", 0) or 0),
            -int(row.get("support", 0) or 0),
            _canonical_json(row.get("feature_values") or {}),
        )
    )
    return rows[:limit]


def evaluate_repair_prior_chain_value(
    examples: Iterable[Mapping[str, Any]],
    *,
    feature_key_groups: tuple[tuple[str, ...], ...] = REPAIR_PRIOR_CHAIN_VALUE_FEATURE_KEY_GROUPS,
) -> dict[str, Any]:
    example_list = [dict(example) for example in examples]
    total = len(example_list)
    label_totals = Counter()
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    by_score: dict[str, Counter[str]] = defaultdict(Counter)
    by_kind: Counter[str] = Counter()
    for example in example_list:
        labels = dict(example.get("labels") or {})
        features = dict(example.get("features") or {})
        game_id = str(example.get("game_id", "") or "unknown")
        score = str(features.get("score", "") or "unknown")
        kind = str(features.get("candidate_kind", "") or "unknown")
        by_game[game_id]["example_count"] += 1
        by_score[score]["example_count"] += 1
        by_kind[kind] += 1
        for key in (
            "visible_change",
            "score_progress",
            "delayed_score_progress_credit",
            "chain_positive",
        ):
            if bool(labels.get(key)):
                label_totals[key] += 1
                by_game[game_id][key] += 1
                by_score[score][key] += 1
    chain_positive = int(label_totals.get("chain_positive", 0))
    score_progress = int(label_totals.get("score_progress", 0))
    delayed_credit = int(label_totals.get("delayed_score_progress_credit", 0))
    return {
        "schema_version": "eca_repair_prior_chain_value_eval.v1",
        "example_count": total,
        "feature_key_groups": [list(group) for group in feature_key_groups],
        "visible_change_count": int(label_totals.get("visible_change", 0)),
        "score_progress_count": score_progress,
        "delayed_credit_count": delayed_credit,
        "chain_positive_count": chain_positive,
        "visible_change_rate": _safe_div(int(label_totals.get("visible_change", 0)), total),
        "score_progress_rate": _safe_div(score_progress, total),
        "delayed_credit_rate": _safe_div(delayed_credit, total),
        "chain_positive_rate": _safe_div(chain_positive, total),
        "by_game": {
            game_id: {
                "example_count": int(stats.get("example_count", 0)),
                "chain_positive_rate": _safe_div(
                    int(stats.get("chain_positive", 0)),
                    int(stats.get("example_count", 0)),
                ),
                "score_progress_rate": _safe_div(
                    int(stats.get("score_progress", 0)),
                    int(stats.get("example_count", 0)),
                ),
            }
            for game_id, stats in sorted(by_game.items())
        },
        "by_score": {
            score: {
                "example_count": int(stats.get("example_count", 0)),
                "chain_positive_count": int(stats.get("chain_positive", 0)),
                "score_progress_count": int(stats.get("score_progress", 0)),
                "delayed_credit_count": int(stats.get("delayed_score_progress_credit", 0)),
                "chain_positive_rate": _safe_div(
                    int(stats.get("chain_positive", 0)),
                    int(stats.get("example_count", 0)),
                ),
            }
            for score, stats in sorted(by_score.items())
        },
        "candidate_kind_histogram": dict(by_kind),
        "top_chain_value_rules": _repair_prior_chain_rule_rows(
            example_list,
            feature_key_groups=feature_key_groups,
        ),
    }


def build_repair_distillation_report(
    episode_summaries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize repair distillation examples without treating them as wins."""

    summary_list = [dict(summary) for summary in episode_summaries if isinstance(summary, Mapping)]
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    by_action_pair: Counter[str] = Counter()
    by_translation_status: Counter[str] = Counter()
    stored_example_count = 0

    def _add_sample(sample: Mapping[str, Any], *, default_game_id: str = "unknown") -> None:
        nonlocal stored_example_count
        game_id = str(sample.get("game_id", "") or default_game_id or "unknown")
        labels = dict(sample.get("labels") or {})
        features = dict(sample.get("features") or {})
        by_game[game_id]["example_count"] += 1
        stored_example_count += 1
        if bool(labels.get("teacher_assisted_executed")):
            by_game[game_id]["teacher_assisted_executed"] += 1
        for key in (
            "action_agrees_before",
            "action_agrees_after",
            "payload_compared",
            "payload_agrees_before",
            "payload_agrees_after",
        ):
            if bool(labels.get(key)):
                by_game[game_id][key] += 1
        action_pair = str(features.get("action_pair", "") or "")
        if action_pair:
            by_action_pair[action_pair] += 1
        translation_status = str(features.get("target_translation_status", "") or "")
        if translation_status:
            by_translation_status[translation_status] += 1

    for summary in summary_list:
        bridge_summary = summary.get("perception_solver_bridge")
        if not isinstance(bridge_summary, Mapping):
            bridge_summary = summary
        game_id = str(bridge_summary.get("game_id", "") or "unknown")
        samples = [
            sample
            for sample in list(bridge_summary.get("repair_distillation_samples") or [])
            if isinstance(sample, Mapping)
            and str(sample.get("schema_version", "") or "") == "eca_repair_distillation_example.v1"
        ]
        aggregate_count = int(bridge_summary.get("repair_distillation_example_count", 0) or 0)
        if aggregate_count <= 0:
            for sample in samples:
                _add_sample(sample, default_game_id=game_id)
            continue

        stored_example_count += len(samples)
        by_game[game_id]["example_count"] += aggregate_count
        by_game[game_id]["teacher_assisted_executed"] += int(
            bridge_summary.get("repair_distillation_executed_count", 0) or 0
        )
        label_hist = Counter(dict(bridge_summary.get("repair_distillation_label_histogram", {}) or {}))
        for key in (
            "action_agrees_before",
            "action_agrees_after",
            "payload_compared",
            "payload_agrees_before",
            "payload_agrees_after",
        ):
            by_game[game_id][key] += int(label_hist.get(key, 0) or 0)
        by_action_pair.update(
            {
                str(key): int(value)
                for key, value in dict(
                    bridge_summary.get("repair_distillation_action_pair_histogram", {}) or {}
                ).items()
                if str(key)
            }
        )
        by_translation_status.update(
            {
                str(key): int(value)
                for key, value in dict(
                    bridge_summary.get(
                        "repair_distillation_target_translation_status_histogram", {}
                    )
                    or {}
                ).items()
                if str(key)
            }
        )
        if not by_action_pair or not by_translation_status:
            for sample in samples:
                features = dict(sample.get("features") or {})
                action_pair = str(features.get("action_pair", "") or "")
                if action_pair:
                    by_action_pair[action_pair] += 1
                translation_status = str(features.get("target_translation_status", "") or "")
                if translation_status:
                    by_translation_status[translation_status] += 1

    total = sum(int(stats.get("example_count", 0) or 0) for stats in by_game.values())
    totals = Counter()
    for stats in by_game.values():
        totals.update(stats)
    return {
        "schema_version": "eca_repair_distillation_report.v1",
        "example_count": total,
        "stored_example_count": stored_example_count,
        "teacher_assisted_executed_count": int(totals.get("teacher_assisted_executed", 0)),
        "action_agreement_before_rate": _safe_div(int(totals.get("action_agrees_before", 0)), total),
        "action_agreement_after_rate": _safe_div(int(totals.get("action_agrees_after", 0)), total),
        "payload_compared_count": int(totals.get("payload_compared", 0)),
        "payload_agreement_before_rate": _safe_div(
            int(totals.get("payload_agrees_before", 0)),
            int(totals.get("payload_compared", 0)),
        ),
        "payload_agreement_after_rate": _safe_div(
            int(totals.get("payload_agrees_after", 0)),
            int(totals.get("payload_compared", 0)),
        ),
        "by_game": {
            game_id: {
                "example_count": int(stats.get("example_count", 0)),
                "teacher_assisted_executed_count": int(
                    stats.get("teacher_assisted_executed", 0)
                ),
                "action_agreement_before_rate": _safe_div(
                    int(stats.get("action_agrees_before", 0)),
                    int(stats.get("example_count", 0)),
                ),
                "action_agreement_after_rate": _safe_div(
                    int(stats.get("action_agrees_after", 0)),
                    int(stats.get("example_count", 0)),
                ),
                "payload_compared_count": int(stats.get("payload_compared", 0)),
                "payload_agreement_before_rate": _safe_div(
                    int(stats.get("payload_agrees_before", 0)),
                    int(stats.get("payload_compared", 0)),
                ),
                "payload_agreement_after_rate": _safe_div(
                    int(stats.get("payload_agrees_after", 0)),
                    int(stats.get("payload_compared", 0)),
                ),
            }
            for game_id, stats in sorted(by_game.items())
        },
        "top_action_pairs": [
            {"action_pair": pair, "count": int(count)}
            for pair, count in by_action_pair.most_common(20)
        ],
        "top_target_translation_statuses": [
            {"target_translation_status": status, "count": int(count)}
            for status, count in by_translation_status.most_common(20)
        ],
    }


def discover_repair_distillation_summary_paths(
    roots: Iterable[str | Path],
    *,
    limit: int | None = None,
) -> list[Path]:
    """Discover `eca_episode_summary.json` files under run roots."""

    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        path = Path(root)
        candidates: list[Path] = []
        if path.is_file():
            candidates = [path]
        elif (path / "eca_episode_summary.json").is_file():
            candidates = [path / "eca_episode_summary.json"]
        elif path.exists():
            candidates = sorted(path.rglob("eca_episode_summary.json"))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(candidate)
            if limit is not None and len(paths) >= int(limit):
                return paths
    return paths


def load_repair_distillation_episode_summaries(
    roots: Iterable[str | Path],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in discover_repair_distillation_summary_paths(roots, limit=limit):
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(loaded, Mapping):
            summary = dict(loaded)
            summary["_source_path"] = str(path)
            summaries.append(summary)
    return summaries


def build_repair_distillation_report_from_roots(
    roots: Iterable[str | Path],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    summaries = load_repair_distillation_episode_summaries(roots, limit=limit)
    report = build_repair_distillation_report(summaries)
    report["source_summary_count"] = len(summaries)
    report["source_paths"] = [
        str(summary.get("_source_path", "") or "")
        for summary in summaries
        if str(summary.get("_source_path", "") or "")
    ]
    return report


def write_repair_distillation_report(
    roots: Iterable[str | Path],
    output: str | Path,
    *,
    limit: int | None = None,
) -> Path:
    report = build_repair_distillation_report_from_roots(roots, limit=limit)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _repair_student_shadow_metrics(
    examples: list[Mapping[str, Any]],
    predictions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    available = 0
    action_agree = 0
    payload_compared = 0
    payload_agree = 0
    exact_agree = 0
    support_total = 0
    feature_key_histogram: Counter[str] = Counter()
    for example, prediction in zip(examples, predictions):
        labels = dict(example.get("labels") or {})
        target = _repair_student_target(example)
        if bool(labels.get("payload_compared")) or target.get("payload"):
            payload_compared += 1
        if not bool(prediction.get("available")):
            continue
        available += 1
        support_total += int(prediction.get("support", 0) or 0)
        feature_key_histogram["|".join(list(prediction.get("feature_keys") or []))] += 1
        predicted_target = dict(prediction.get("target") or {})
        if str(predicted_target.get("action", "") or "") == str(target.get("action", "") or ""):
            action_agree += 1
        if dict(predicted_target.get("payload") or {}) == dict(target.get("payload") or {}):
            payload_agree += 1
        if (
            str(predicted_target.get("action", "") or "") == str(target.get("action", "") or "")
            and dict(predicted_target.get("payload") or {}) == dict(target.get("payload") or {})
        ):
            exact_agree += 1

    total = len(examples)
    return {
        "example_count": total,
        "coverage_count": available,
        "coverage_rate": _safe_div(available, total),
        "action_agreement_rate": _safe_div(action_agree, available),
        "payload_compared_count": payload_compared,
        "payload_agreement_rate": _safe_div(payload_agree, payload_compared),
        "exact_target_agreement_rate": _safe_div(exact_agree, available),
        "average_support": _safe_div(support_total, available),
        "feature_key_histogram": dict(feature_key_histogram),
    }


def _repair_student_feature_ambiguity(
    examples: list[Mapping[str, Any]],
    feature_key_groups: tuple[tuple[str, ...], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for keys in feature_key_groups:
        buckets: dict[tuple[tuple[str, str], ...], Counter[str]] = {}
        missing_count = 0
        for example in examples:
            feature_key = _repair_student_feature_key(example, keys)
            if feature_key is None:
                missing_count += 1
                continue
            target_key = _canonical_json(_repair_student_target(example))
            buckets.setdefault(feature_key, Counter())[target_key] += 1
        ambiguous = [counter for counter in buckets.values() if len(counter) > 1]
        rows.append(
            {
                "feature_keys": list(keys),
                "bucket_count": len(buckets),
                "missing_count": int(missing_count),
                "ambiguous_bucket_count": len(ambiguous),
                "ambiguous_bucket_rate": _safe_div(len(ambiguous), len(buckets)),
                "max_distinct_targets": max([len(counter) for counter in buckets.values()] or [0]),
                "mean_target_entropy": round(
                    _safe_div(
                        sum(_counter_entropy(counter) for counter in buckets.values()),
                        len(buckets),
                    ),
                    6,
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["ambiguous_bucket_rate"]),
            -int(row["max_distinct_targets"]),
            int(row["missing_count"]),
        )
    )
    return rows


def _bucket_int(value: int, *, width: int = 4) -> str:
    if width <= 1:
        return str(int(value))
    lower = int(math.floor(float(value) / float(width)) * width)
    upper = lower + width - 1
    return f"{lower}..{upper}"


def _score_key_from_features(
    features: Mapping[str, Any],
    *,
    default: str = "unknown",
) -> str:
    value = features.get("score")
    if value in (None, ""):
        return str(default)
    return str(value)


def _candidate_scalar_color(candidate: Mapping[str, Any]) -> str:
    color = candidate.get("color")
    if color not in (None, ""):
        return str(color)
    colors = candidate.get("colors")
    if isinstance(colors, list) and colors:
        return ",".join(str(item) for item in colors)
    return ""


def _candidate_kind_base(candidate: Mapping[str, Any]) -> str:
    kind = str(candidate.get("kind", "") or "")
    if kind.startswith("memory_"):
        return kind[len("memory_") :]
    return kind


def _repair_candidate_feature_row(
    example: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, str]:
    features = dict(example.get("features") or {})
    teacher = dict(example.get("teacher") or {})
    proposal = dict(example.get("proposal") or {})
    teacher_point = _repair_click_point(teacher.get("payload"))
    proposal_point = _repair_click_point(proposal.get("payload"))
    try:
        x = int(candidate.get("x"))
        y = int(candidate.get("y"))
    except (TypeError, ValueError):
        x, y = 0, 0
    center_col = _as_int(features.get("selected_target_center_col"), 0)
    center_row = _as_int(features.get("selected_target_center_row"), 0)
    dx = x - center_col if center_col else 0
    dy = y - center_row if center_row else 0
    manhattan = abs(dx) + abs(dy)
    proposal_distance = (
        abs(x - proposal_point[0]) + abs(y - proposal_point[1])
        if proposal_point is not None
        else -1
    )
    teacher_distance = (
        abs(x - teacher_point[0]) + abs(y - teacher_point[1])
        if teacher_point is not None
        else -1
    )
    memory_seen = _as_int(candidate.get("memory_seen_count"), 0)
    memory_age = _as_int(candidate.get("memory_age"), 0)
    target_dx_value = candidate.get("candidate_target_dx")
    target_dy_value = candidate.get("candidate_target_dy")
    jitter_dx_value = candidate.get("candidate_jitter_dx")
    jitter_dy_value = candidate.get("candidate_jitter_dy")
    target_dx = (
        _as_int(target_dx_value, 0)
        if target_dx_value not in (None, "")
        else None
    )
    target_dy = (
        _as_int(target_dy_value, 0)
        if target_dy_value not in (None, "")
        else None
    )
    jitter_dx = (
        _as_int(jitter_dx_value, 0)
        if jitter_dx_value not in (None, "")
        else None
    )
    jitter_dy = (
        _as_int(jitter_dy_value, 0)
        if jitter_dy_value not in (None, "")
        else None
    )
    return {
        "game_id": str(example.get("game_id", "") or ""),
        "score": _score_key_from_features(features, default=""),
        "phase": str(features.get("phase", "") or ""),
        "selected_target_component_id": str(
            features.get("selected_target_component_id", "") or ""
        ),
        "selected_target_anchor": str(features.get("selected_target_anchor", "") or ""),
        "target_translation_status": str(
            features.get("target_translation_status", "") or ""
        ),
        "candidate_kind": str(candidate.get("kind", "") or ""),
        "candidate_kind_base": _candidate_kind_base(candidate),
        "candidate_band": str(candidate.get("candidate_band", "") or ""),
        "candidate_source": str(candidate.get("candidate_source", "") or ""),
        "candidate_color": _candidate_scalar_color(candidate),
        "candidate_x": str(x),
        "candidate_y": str(y),
        "candidate_previously_executed": str(
            bool(candidate.get("candidate_previously_executed"))
        ).lower(),
        "candidate_previously_proposed": str(
            bool(candidate.get("candidate_previously_proposed"))
        ).lower(),
        "candidate_dx": str(dx),
        "candidate_dy": str(dy),
        "candidate_dx_bucket": _bucket_int(dx),
        "candidate_dy_bucket": _bucket_int(dy),
        "candidate_manhattan_bucket": _bucket_int(manhattan, width=8),
        "candidate_target_dx": "" if target_dx is None else str(target_dx),
        "candidate_target_dy": "" if target_dy is None else str(target_dy),
        "candidate_target_dx_bucket": ""
        if target_dx is None
        else _bucket_int(target_dx),
        "candidate_target_dy_bucket": ""
        if target_dy is None
        else _bucket_int(target_dy),
        "candidate_jitter_dx": "" if jitter_dx is None else str(jitter_dx),
        "candidate_jitter_dy": "" if jitter_dy is None else str(jitter_dy),
        "candidate_index": str(int(index)),
        "candidate_index_bucket": _bucket_int(int(index), width=16),
        "candidate_memory_seen_bucket": _bucket_int(memory_seen, width=4),
        "candidate_memory_age_bucket": _bucket_int(memory_age, width=8),
        "candidate_distance_to_proposal_bucket": _bucket_int(proposal_distance, width=8)
        if proposal_distance >= 0
        else "",
        "candidate_matches_proposal": str(bool(proposal_distance == 0)).lower(),
        "teacher_distance": str(teacher_distance),
    }


def _repair_candidate_feature_key(
    feature_row: Mapping[str, Any],
    keys: Iterable[str],
) -> tuple[tuple[str, str], ...] | None:
    parts: list[tuple[str, str]] = []
    for key in keys:
        value = str(feature_row.get(key, "") or "")
        if not value:
            return None
        parts.append((str(key), value))
    return tuple(parts)


def _repair_candidate_examples_from_sample(
    example: Mapping[str, Any],
    *,
    label_mode: str = "teacher_site",
) -> dict[str, Any]:
    teacher = dict(example.get("teacher") or {})
    labels = dict(example.get("labels") or {})
    teacher_point = _repair_click_point(teacher.get("payload"))
    mode = str(label_mode or "teacher_site").strip().lower()
    require_score_progress = mode in {
        "score_progress",
        "teacher_score_progress",
        "score_progress_value",
    }
    require_delayed_score_progress = mode in {
        "score_progress_h5",
        "delayed_score_progress",
        "delayed_score_progress_value",
    }
    candidates = [
        dict(candidate)
        for candidate in list(example.get("visible_click_candidates") or [])
        if isinstance(candidate, Mapping)
    ]
    rows: list[dict[str, Any]] = []
    positive_indices: list[int] = []
    for index, candidate in enumerate(candidates):
        try:
            x = int(candidate.get("x"))
            y = int(candidate.get("y"))
        except (TypeError, ValueError):
            continue
        teacher_match = bool(teacher_point is not None and (x, y) == teacher_point)
        is_positive = bool(
            teacher_match
            and (
                not require_score_progress
                or bool(labels.get("repair_effect_score_progress"))
            )
            and (
                not require_delayed_score_progress
                or bool(labels.get("repair_effect_score_progress_h5"))
            )
        )
        if is_positive:
            positive_indices.append(len(rows))
        rows.append(
            {
                "candidate": dict(candidate),
                "features": _repair_candidate_feature_row(
                    example,
                    candidate,
                    index=index,
                ),
                "label": bool(is_positive),
            }
        )
    return {
        "example_id": str(example.get("example_id", "") or ""),
        "game_id": str(example.get("game_id", "") or ""),
        "label_mode": mode,
        "teacher_point": list(teacher_point) if teacher_point is not None else [],
        "candidate_count": len(rows),
        "positive_indices": positive_indices,
        "positive_exists": bool(positive_indices),
        "rows": rows,
    }


def _repair_candidate_progress_prior(feature_row: Mapping[str, Any]) -> dict[str, Any]:
    """Teacher-free candidate priority used as a shadow diagnostic.

    The prior is deliberately simple: it encodes generic repair preferences
    over visible candidate families, tried/proposed state, and coarse r11l
    phase order. It does not inspect the teacher payload.
    """

    kind = str(feature_row.get("candidate_kind_base", "") or "")
    band = str(feature_row.get("candidate_band", "") or "")
    score_bucket = _as_int(feature_row.get("score"), 0)
    candidate_index = _as_int(feature_row.get("candidate_index"), -1)
    previously_executed = str(feature_row.get("candidate_previously_executed", "")).lower() == "true"
    previously_proposed = str(feature_row.get("candidate_previously_proposed", "")).lower() == "true"
    source = str(feature_row.get("candidate_source", "") or "")

    base_by_kind = {
        "score_progress_selected_target_offset": 0.86,
        "score_progress_template_candidate": 0.80,
        "score_progress_template_inactive_piece": 0.72,
        "target_centroid_destination_jitter": 0.72,
        "target_centroid_destination": 0.68,
        "inactive_piece": 0.62,
        "target_neighborhood": 0.48,
        "signature_absorber_waypoint": 0.46,
        "signature_centroid_destination": 0.44,
        "target_destination": 0.42,
        "target": 0.24,
        "canvas": 0.18,
        "active_piece": 0.12,
    }
    value = float(base_by_kind.get(kind, 0.30))
    reasons = [f"kind:{kind or 'unknown'}"]

    if band == "progress_template":
        value += 0.10
        reasons.append("progress_template_band")
    elif band == "visible_object":
        value += 0.02
        reasons.append("visible_object_band")
    elif band == "dense_cloud":
        value -= 0.03
        reasons.append("dense_cloud_band_penalty")

    if score_bucket <= 0:
        if band == "visible_object":
            value += 0.28
            reasons.append("score0_visible_bootstrap")
        if kind in {"active_piece", "inactive_piece", "target"}:
            value += 0.10
            reasons.append("score0_object_site")
        if band == "dense_cloud":
            value -= 0.20
            reasons.append("score0_dense_deprioritized")

    if score_bucket >= 4:
        if kind in {"target_centroid_destination", "target_centroid_destination_jitter"}:
            value += 0.16
            reasons.append("late_target_centroid")
        elif kind in {
            "score_progress_selected_target_offset",
            "score_progress_template_candidate",
        }:
            value += 0.08
            reasons.append("late_progress_template")
        elif kind == "inactive_piece":
            value -= 0.18
            reasons.append("late_inactive_penalty")
    elif 1 <= score_bucket <= 3:
        if kind in {"inactive_piece", "score_progress_template_inactive_piece"}:
            value += 0.12
            reasons.append("early_inactive_piece")
        elif kind in {"target_centroid_destination", "target_centroid_destination_jitter"}:
            value += 0.04
            reasons.append("early_target_centroid")
        elif kind == "score_progress_selected_target_offset":
            value += 0.06
            reasons.append("early_selected_target_template")

    if previously_executed:
        value -= 0.25
        reasons.append("already_executed_penalty")
    if previously_proposed:
        value -= 0.08
        reasons.append("already_proposed_penalty")
    if source == "memory":
        value -= 0.02
        reasons.append("memory_slight_penalty")

    # The candidate generator is ordered by candidate family and target phase.
    # This coarse phase prior is a diagnostic bridge toward a learned
    # candidate-value model; it must not be interpreted as a deployable policy.
    ideal_index_by_score = {
        1: 120,
        2: 205,
        3: 430,
        4: 610,
        5: 640,
    }
    if candidate_index >= 0 and score_bucket in ideal_index_by_score:
        ideal = ideal_index_by_score[score_bucket]
        closeness = max(0.0, 1.0 - abs(float(candidate_index - ideal)) / 260.0)
        value += 0.20 * closeness
        reasons.append("phase_order_prior")

    return {
        "score": round(max(0.0, min(1.0, value)), 6),
        "feature_keys": ["candidate_progress_prior"],
        "reasons": reasons,
    }


def _rank_repair_candidates_by_progress_prior(
    example: Mapping[str, Any],
    *,
    label_mode: str = "teacher_site",
) -> dict[str, Any]:
    bundle = _repair_candidate_examples_from_sample(example, label_mode=label_mode)
    ranked: list[dict[str, Any]] = []
    for index, row in enumerate(list(bundle.get("rows") or [])):
        feature_row = dict(row.get("features") or {})
        prior = _repair_candidate_progress_prior(feature_row)
        ranked.append(
            {
                "candidate_index": int(index),
                "candidate": dict(row.get("candidate") or {}),
                "label": bool(row.get("label")),
                "score": float(prior.get("score", 0.0) or 0.0),
                "support": 0,
                "positive_support": 0,
                "feature_keys": list(prior.get("feature_keys") or []),
                "prior_reasons": list(prior.get("reasons") or []),
                "teacher_distance": _as_int(feature_row.get("teacher_distance"), -1),
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row.get("score", 0.0) or 0.0),
            int(row.get("candidate_index", 0) or 0),
        )
    )
    top = dict(ranked[0]) if ranked else {}
    return {
        "example_id": str(bundle.get("example_id", "") or ""),
        "label_mode": str(bundle.get("label_mode", "") or ""),
        "candidate_count": int(bundle.get("candidate_count", 0) or 0),
        "positive_exists": bool(bundle.get("positive_exists")),
        "positive_indices": list(bundle.get("positive_indices") or []),
        "top": top,
        "ranked": ranked,
    }


def rank_repair_candidates_by_progress_prior(
    example: Mapping[str, Any],
    *,
    label_mode: str = "teacher_site",
) -> dict[str, Any]:
    """Public wrapper for teacher-free repair candidate prior diagnostics."""

    return _rank_repair_candidates_by_progress_prior(
        example,
        label_mode=label_mode,
    )


@dataclass
class RepairCandidateRanker:
    """Frequency ranker over frame-derived repair candidates.

    This is intentionally a shadow evaluator. It learns reusable candidate
    features, then ranks the current candidate set without reading the teacher
    payload.
    """

    feature_key_groups: tuple[tuple[str, ...], ...] = REPAIR_CANDIDATE_RANKER_FEATURE_KEY_GROUPS
    label_mode: str = "teacher_site"
    counts: dict[tuple[int, tuple[tuple[str, str], ...]], Counter[str]] = field(
        default_factory=dict
    )
    default_counts: Counter[str] = field(default_factory=Counter)

    def fit(self, examples: Iterable[Mapping[str, Any]]) -> "RepairCandidateRanker":
        for example in examples:
            bundle = _repair_candidate_examples_from_sample(
                example,
                label_mode=self.label_mode,
            )
            for row in list(bundle.get("rows") or []):
                label = "positive" if bool(row.get("label")) else "negative"
                self.default_counts[label] += 1
                feature_row = dict(row.get("features") or {})
                for index, keys in enumerate(self.feature_key_groups):
                    feature_key = _repair_candidate_feature_key(feature_row, keys)
                    if feature_key is None:
                        continue
                    self.counts.setdefault((index, feature_key), Counter())[label] += 1
        return self

    def _score_feature_row(self, feature_row: Mapping[str, Any]) -> dict[str, Any]:
        for index, keys in enumerate(self.feature_key_groups):
            feature_key = _repair_candidate_feature_key(feature_row, keys)
            if feature_key is None:
                continue
            counter = self.counts.get((index, feature_key))
            if not counter:
                continue
            pos = int(counter.get("positive", 0))
            neg = int(counter.get("negative", 0))
            total = pos + neg
            score = (float(pos) + 0.5) / (float(total) + 1.0)
            return {
                "score": round(score, 6),
                "support": total,
                "positive_support": pos,
                "feature_keys": list(keys),
            }
        pos = int(self.default_counts.get("positive", 0))
        neg = int(self.default_counts.get("negative", 0))
        total = pos + neg
        if total <= 0:
            return {
                "score": 0.0,
                "support": 0,
                "positive_support": 0,
                "feature_keys": [],
            }
        return {
            "score": round((float(pos) + 0.5) / (float(total) + 1.0), 6),
            "support": total,
            "positive_support": pos,
            "feature_keys": ["default"],
        }

    def rank(self, example: Mapping[str, Any]) -> dict[str, Any]:
        bundle = _repair_candidate_examples_from_sample(
            example,
            label_mode=self.label_mode,
        )
        ranked: list[dict[str, Any]] = []
        for index, row in enumerate(list(bundle.get("rows") or [])):
            feature_row = dict(row.get("features") or {})
            scored = self._score_feature_row(feature_row)
            prior = _repair_candidate_progress_prior(feature_row)
            candidate = dict(row.get("candidate") or {})
            ranked.append(
                {
                    "candidate_index": int(index),
                    "candidate": candidate,
                    "label": bool(row.get("label")),
                    "score": float(scored.get("score", 0.0) or 0.0),
                    "support": int(scored.get("support", 0) or 0),
                    "positive_support": int(scored.get("positive_support", 0) or 0),
                    "feature_keys": list(scored.get("feature_keys") or []),
                    "prior_score": float(prior.get("score", 0.0) or 0.0),
                    "prior_reasons": list(prior.get("reasons") or []),
                    "teacher_distance": _as_int(feature_row.get("teacher_distance"), -1),
                }
            )
        ranked.sort(
            key=lambda row: (
                -float(row.get("score", 0.0) or 0.0),
                -int(row.get("positive_support", 0) or 0),
                -int(row.get("support", 0) or 0),
                int(row.get("candidate_index", 0) or 0),
            )
        )
        top = dict(ranked[0]) if ranked else {}
        return {
            "example_id": str(bundle.get("example_id", "") or ""),
            "label_mode": str(bundle.get("label_mode", "") or ""),
            "candidate_count": int(bundle.get("candidate_count", 0) or 0),
            "positive_exists": bool(bundle.get("positive_exists")),
            "positive_indices": list(bundle.get("positive_indices") or []),
            "top": top,
            "ranked": ranked,
        }

    def top_rules(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (index, feature_key), counter in self.counts.items():
            pos = int(counter.get("positive", 0))
            neg = int(counter.get("negative", 0))
            total = pos + neg
            if total <= 0:
                continue
            rows.append(
                {
                    "feature_keys": list(self.feature_key_groups[index]),
                    "feature_values": dict(feature_key),
                    "support": total,
                    "positive_support": pos,
                    "negative_support": neg,
                    "positive_rate": _safe_div(pos, total),
                }
            )
        rows.sort(
            key=lambda row: (
                -float(row["positive_rate"]),
                -int(row["positive_support"]),
                -int(row["support"]),
                _canonical_json(row["feature_values"]),
            )
        )
        return rows[:limit]


@dataclass
class RepairCandidatePrototypeRanker:
    """Prototype ranker over teacher-labeled candidate coordinates.

    This remains a shadow evaluator. It compresses observed positive candidate
    rows into reusable score/kind/color prototypes, then ranks future candidate
    sets by proximity to those prototypes plus the teacher-free progress prior.
    """

    feature_key_groups: tuple[tuple[str, ...], ...] = REPAIR_CANDIDATE_PROTOTYPE_FEATURE_KEY_GROUPS
    label_mode: str = "teacher_site"
    prototypes: dict[tuple[int, tuple[tuple[str, str], ...]], list[dict[str, float]]] = field(
        default_factory=dict
    )

    def fit(self, examples: Iterable[Mapping[str, Any]]) -> "RepairCandidatePrototypeRanker":
        for example in examples:
            bundle = _repair_candidate_examples_from_sample(
                example,
                label_mode=self.label_mode,
            )
            for row in list(bundle.get("rows") or []):
                if not bool(row.get("label")):
                    continue
                feature_row = dict(row.get("features") or {})
                prototype = {
                    "x": float(_as_int(feature_row.get("candidate_x"), 0)),
                    "y": float(_as_int(feature_row.get("candidate_y"), 0)),
                    "dx": float(_as_int(feature_row.get("candidate_dx"), 0)),
                    "dy": float(_as_int(feature_row.get("candidate_dy"), 0)),
                    "index": float(_as_int(feature_row.get("candidate_index"), 0)),
                    "target_dx": float(_as_int(feature_row.get("candidate_target_dx"), 0))
                    if str(feature_row.get("candidate_target_dx", "") or "")
                    else None,
                    "target_dy": float(_as_int(feature_row.get("candidate_target_dy"), 0))
                    if str(feature_row.get("candidate_target_dy", "") or "")
                    else None,
                }
                for index, keys in enumerate(self.feature_key_groups):
                    feature_key = _repair_candidate_feature_key(feature_row, keys)
                    if feature_key is None:
                        continue
                    self.prototypes.setdefault((index, feature_key), []).append(prototype)
        return self

    def _score_feature_row(self, feature_row: Mapping[str, Any]) -> dict[str, Any]:
        candidate = {
            "x": float(_as_int(feature_row.get("candidate_x"), 0)),
            "y": float(_as_int(feature_row.get("candidate_y"), 0)),
            "dx": float(_as_int(feature_row.get("candidate_dx"), 0)),
            "dy": float(_as_int(feature_row.get("candidate_dy"), 0)),
            "index": float(_as_int(feature_row.get("candidate_index"), 0)),
            "target_dx": float(_as_int(feature_row.get("candidate_target_dx"), 0))
            if str(feature_row.get("candidate_target_dx", "") or "")
            else None,
            "target_dy": float(_as_int(feature_row.get("candidate_target_dy"), 0))
            if str(feature_row.get("candidate_target_dy", "") or "")
            else None,
        }
        prior = _repair_candidate_progress_prior(feature_row)
        prior_score = float(prior.get("score", 0.0) or 0.0)
        for index, keys in enumerate(self.feature_key_groups):
            feature_key = _repair_candidate_feature_key(feature_row, keys)
            if feature_key is None:
                continue
            prototypes = self.prototypes.get((index, feature_key))
            if not prototypes:
                continue
            distances: list[float] = []
            for proto in prototypes:
                target_distance = 0.0
                if (
                    candidate["target_dx"] is not None
                    and candidate["target_dy"] is not None
                    and proto.get("target_dx") is not None
                    and proto.get("target_dy") is not None
                ):
                    target_distance = (
                        abs(float(candidate["target_dx"]) - float(proto["target_dx"]))
                        + abs(float(candidate["target_dy"]) - float(proto["target_dy"]))
                    ) / 18.0
                distances.append(
                    (abs(candidate["x"] - proto["x"]) + abs(candidate["y"] - proto["y"])) / 40.0
                    + (abs(candidate["dx"] - proto["dx"]) + abs(candidate["dy"] - proto["dy"])) / 56.0
                    + abs(candidate["index"] - proto["index"]) / 420.0
                    + target_distance
                )
            best_distance = min(distances)
            prototype_score = 1.0 / (1.0 + best_distance)
            score = 0.75 * prototype_score + 0.25 * prior_score
            return {
                "score": round(max(0.0, min(1.0, score)), 6),
                "support": len(prototypes),
                "positive_support": len(prototypes),
                "feature_keys": list(keys),
                "prototype_distance": round(best_distance, 6),
            }
        return {
            "score": round(0.25 * prior_score, 6),
            "support": 0,
            "positive_support": 0,
            "feature_keys": ["candidate_progress_prior_fallback"],
            "prototype_distance": None,
        }

    def rank(self, example: Mapping[str, Any]) -> dict[str, Any]:
        bundle = _repair_candidate_examples_from_sample(
            example,
            label_mode=self.label_mode,
        )
        ranked: list[dict[str, Any]] = []
        for index, row in enumerate(list(bundle.get("rows") or [])):
            feature_row = dict(row.get("features") or {})
            scored = self._score_feature_row(feature_row)
            candidate = dict(row.get("candidate") or {})
            ranked.append(
                {
                    "candidate_index": int(index),
                    "candidate": candidate,
                    "label": bool(row.get("label")),
                    "score": float(scored.get("score", 0.0) or 0.0),
                    "support": int(scored.get("support", 0) or 0),
                    "positive_support": int(scored.get("positive_support", 0) or 0),
                    "feature_keys": list(scored.get("feature_keys") or []),
                    "prototype_distance": scored.get("prototype_distance"),
                    "teacher_distance": _as_int(feature_row.get("teacher_distance"), -1),
                }
            )
        ranked.sort(
            key=lambda row: (
                -float(row.get("score", 0.0) or 0.0),
                -int(row.get("positive_support", 0) or 0),
                int(row.get("candidate_index", 0) or 0),
            )
        )
        top = dict(ranked[0]) if ranked else {}
        return {
            "example_id": str(bundle.get("example_id", "") or ""),
            "label_mode": str(bundle.get("label_mode", "") or ""),
            "candidate_count": int(bundle.get("candidate_count", 0) or 0),
            "positive_exists": bool(bundle.get("positive_exists")),
            "positive_indices": list(bundle.get("positive_indices") or []),
            "top": top,
            "ranked": ranked,
        }

    def top_prototypes(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (index, feature_key), prototypes in self.prototypes.items():
            if not prototypes:
                continue
            rows.append(
                {
                    "feature_keys": list(self.feature_key_groups[index]),
                    "feature_values": dict(feature_key),
                    "support": len(prototypes),
                    "mean_x": round(_safe_div(sum(proto["x"] for proto in prototypes), len(prototypes)), 6),
                    "mean_y": round(_safe_div(sum(proto["y"] for proto in prototypes), len(prototypes)), 6),
                    "mean_index": round(
                        _safe_div(sum(proto["index"] for proto in prototypes), len(prototypes)),
                        6,
                    ),
                }
            )
        rows.sort(
            key=lambda row: (
                -int(row["support"]),
                _canonical_json(row["feature_values"]),
            )
        )
        return rows[:limit]


def _repair_candidate_ranker_metrics(
    examples: list[Mapping[str, Any]],
    predictions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(examples)
    candidate_episode_count = 0
    positive_episode_count = 0
    top1_exact = 0
    top1_exact_when_contained = 0
    top1_distance_sum = 0
    top1_distance_count = 0
    topk_values = (3, 5, 10)
    topk_exact: Counter[str] = Counter()
    topk_exact_when_contained: Counter[str] = Counter()
    mrr_sum = 0.0
    feature_key_histogram: Counter[str] = Counter()
    by_score: dict[str, Counter[str]] = defaultdict(Counter)

    for example, prediction in zip(examples, predictions):
        features = dict(example.get("features") or {})
        score_key = _score_key_from_features(features)
        if int(prediction.get("candidate_count", 0) or 0) <= 0:
            continue
        candidate_episode_count += 1
        by_score[score_key]["candidate_episode_count"] += 1
        positive_exists = bool(prediction.get("positive_exists"))
        if positive_exists:
            positive_episode_count += 1
            by_score[score_key]["positive_episode_count"] += 1
        ranked = list(prediction.get("ranked") or [])
        top = dict(prediction.get("top") or {})
        if top:
            feature_key_histogram["|".join(list(top.get("feature_keys") or []))] += 1
            distance = _as_int(top.get("teacher_distance"), -1)
            if distance >= 0:
                top1_distance_sum += distance
                top1_distance_count += 1
                by_score[score_key]["top1_distance_sum"] += distance
                by_score[score_key]["top1_distance_count"] += 1
            if bool(top.get("label")):
                top1_exact += 1
                by_score[score_key]["top1_exact"] += 1
                if positive_exists:
                    top1_exact_when_contained += 1
        for topk in topk_values:
            if any(bool(row.get("label")) for row in ranked[:topk]):
                topk_exact[str(topk)] += 1
                if positive_exists:
                    topk_exact_when_contained[str(topk)] += 1
        for rank_index, row in enumerate(ranked, start=1):
            if bool(row.get("label")):
                mrr_sum += 1.0 / float(rank_index)
                by_score[score_key]["mrr_sum_scaled"] += int(round((1.0 / float(rank_index)) * 1000000))
                break

    return {
        "example_count": total,
        "candidate_episode_count": int(candidate_episode_count),
        "positive_episode_count": int(positive_episode_count),
        "candidate_containment_rate": _safe_div(positive_episode_count, candidate_episode_count),
        "top1_exact_rate": _safe_div(top1_exact, candidate_episode_count),
        "top1_exact_when_contained_rate": _safe_div(
            top1_exact_when_contained,
            positive_episode_count,
        ),
        "topk_exact_rates": {
            f"top{topk}": _safe_div(int(topk_exact.get(str(topk), 0)), candidate_episode_count)
            for topk in topk_values
        },
        "topk_exact_when_contained_rates": {
            f"top{topk}": _safe_div(
                int(topk_exact_when_contained.get(str(topk), 0)),
                positive_episode_count,
            )
            for topk in topk_values
        },
        "top1_teacher_distance_avg": _safe_div(top1_distance_sum, top1_distance_count),
        "mrr": round(_safe_div(mrr_sum, positive_episode_count), 6),
        "feature_key_histogram": dict(feature_key_histogram),
        "by_score": {
            score: {
                "candidate_episode_count": int(stats.get("candidate_episode_count", 0)),
                "candidate_containment_rate": _safe_div(
                    int(stats.get("positive_episode_count", 0)),
                    int(stats.get("candidate_episode_count", 0)),
                ),
                "top1_exact_rate": _safe_div(
                    int(stats.get("top1_exact", 0)),
                    int(stats.get("candidate_episode_count", 0)),
                ),
                "top1_teacher_distance_avg": _safe_div(
                    int(stats.get("top1_distance_sum", 0)),
                    int(stats.get("top1_distance_count", 0)),
                ),
                "mrr": round(
                    _safe_div(
                        int(stats.get("mrr_sum_scaled", 0)) / 1000000.0,
                        int(stats.get("positive_episode_count", 0)),
                    ),
                    6,
                ),
            }
            for score, stats in sorted(by_score.items(), key=lambda item: str(item[0]))
        },
    }


def _repair_candidate_chronological_holdout(
    examples: list[Mapping[str, Any]],
    *,
    feature_key_groups: tuple[tuple[str, ...], ...],
    label_mode: str = "teacher_site",
    train_fraction: float = 0.7,
) -> dict[str, Any]:
    if len(examples) < 2:
        return {"skipped": True, "reason": "need_at_least_two_examples"}
    split_index = int(math.floor(float(len(examples)) * float(train_fraction)))
    split_index = max(1, min(len(examples) - 1, split_index))
    train = examples[:split_index]
    test = examples[split_index:]
    frequency_model = RepairCandidateRanker(
        feature_key_groups=feature_key_groups,
        label_mode=label_mode,
    ).fit(train)
    prototype_model = RepairCandidatePrototypeRanker(label_mode=label_mode).fit(train)
    prior_predictions = [
        _rank_repair_candidates_by_progress_prior(example, label_mode=label_mode)
        for example in test
    ]
    frequency_predictions = [frequency_model.rank(example) for example in test]
    prototype_predictions = [prototype_model.rank(example) for example in test]
    return {
        "train_count": len(train),
        "test_count": len(test),
        "train_fraction": train_fraction,
        "label_mode": label_mode,
        "prior_only": _repair_candidate_ranker_metrics(test, prior_predictions),
        "frequency": _repair_candidate_ranker_metrics(test, frequency_predictions),
        "prototype": _repair_candidate_ranker_metrics(test, prototype_predictions),
    }


def _repair_candidate_stratified_holdout(
    examples: list[Mapping[str, Any]],
    *,
    feature_key_groups: tuple[tuple[str, ...], ...],
    label_mode: str = "teacher_site",
    train_fraction: float = 0.7,
) -> dict[str, Any]:
    by_score: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for example in examples:
        features = dict(example.get("features") or {})
        by_score[_score_key_from_features(features)].append(example)

    train: list[Mapping[str, Any]] = []
    test: list[Mapping[str, Any]] = []
    score_splits: dict[str, dict[str, int]] = {}
    for score, rows in sorted(by_score.items(), key=lambda item: str(item[0])):
        if len(rows) < 2:
            train.extend(rows)
            score_splits[score] = {"train_count": len(rows), "test_count": 0}
            continue
        split_index = int(math.floor(float(len(rows)) * float(train_fraction)))
        split_index = max(1, min(len(rows) - 1, split_index))
        train.extend(rows[:split_index])
        test.extend(rows[split_index:])
        score_splits[score] = {
            "train_count": split_index,
            "test_count": len(rows) - split_index,
        }

    if not train or not test:
        return {"skipped": True, "reason": "need_train_and_test_examples"}

    frequency_model = RepairCandidateRanker(
        feature_key_groups=feature_key_groups,
        label_mode=label_mode,
    ).fit(train)
    prototype_model = RepairCandidatePrototypeRanker(label_mode=label_mode).fit(train)
    prior_predictions = [
        _rank_repair_candidates_by_progress_prior(example, label_mode=label_mode)
        for example in test
    ]
    frequency_predictions = [frequency_model.rank(example) for example in test]
    prototype_predictions = [prototype_model.rank(example) for example in test]
    return {
        "train_count": len(train),
        "test_count": len(test),
        "train_fraction": train_fraction,
        "label_mode": label_mode,
        "score_splits": score_splits,
        "prior_only": _repair_candidate_ranker_metrics(test, prior_predictions),
        "frequency": _repair_candidate_ranker_metrics(test, frequency_predictions),
        "prototype": _repair_candidate_ranker_metrics(test, prototype_predictions),
    }


def _repair_effect_outcome_diagnostics(
    examples: list[Mapping[str, Any]],
) -> dict[str, Any]:
    resolved_count = 0
    visible_change_count = 0
    score_progress_count = 0
    click_near_change_count = 0
    score_delta_sum = 0
    outcome_kind_histogram: Counter[str] = Counter()
    score_progress_teacher_in_candidates = 0
    score_progress_teacher_kind_histogram: Counter[str] = Counter()
    by_score: dict[str, Counter[str]] = defaultdict(Counter)

    for example in examples:
        labels = dict(example.get("labels") or {})
        outcome = dict(example.get("outcome") or {})
        features = dict(example.get("features") or {})
        score_key = _score_key_from_features(features)
        if not bool(labels.get("repair_effect_resolved")):
            continue
        resolved_count += 1
        by_score[score_key]["resolved_count"] += 1
        outcome_kind = str(outcome.get("outcome_kind", "") or "unknown")
        outcome_kind_histogram[outcome_kind] += 1
        score_delta = _as_int(outcome.get("score_delta"), 0)
        score_delta_sum += score_delta
        by_score[score_key]["score_delta_sum"] += score_delta
        if bool(labels.get("repair_effect_visible_change")):
            visible_change_count += 1
            by_score[score_key]["visible_change_count"] += 1
        if bool(labels.get("repair_effect_click_near_change")):
            click_near_change_count += 1
            by_score[score_key]["click_near_change_count"] += 1
        if bool(labels.get("repair_effect_score_progress")):
            score_progress_count += 1
            by_score[score_key]["score_progress_count"] += 1
            match = dict(example.get("teacher_site_candidate_match") or {})
            if bool(match.get("in_candidates")):
                score_progress_teacher_in_candidates += 1
                kind = str(match.get("kind", "") or "unknown")
                score_progress_teacher_kind_histogram[kind] += 1

    return {
        "resolved_count": int(resolved_count),
        "visible_change_count": int(visible_change_count),
        "score_progress_count": int(score_progress_count),
        "click_near_change_count": int(click_near_change_count),
        "visible_change_rate": _safe_div(visible_change_count, resolved_count),
        "score_progress_rate": _safe_div(score_progress_count, resolved_count),
        "click_near_change_rate": _safe_div(click_near_change_count, resolved_count),
        "score_delta_avg": _safe_div(score_delta_sum, resolved_count),
        "score_progress_teacher_containment_rate": _safe_div(
            score_progress_teacher_in_candidates,
            score_progress_count,
        ),
        "outcome_kind_histogram": dict(outcome_kind_histogram),
        "score_progress_teacher_kind_histogram": dict(score_progress_teacher_kind_histogram),
        "by_score": {
            score: {
                "resolved_count": int(stats.get("resolved_count", 0)),
                "visible_change_rate": _safe_div(
                    int(stats.get("visible_change_count", 0)),
                    int(stats.get("resolved_count", 0)),
                ),
                "score_progress_rate": _safe_div(
                    int(stats.get("score_progress_count", 0)),
                    int(stats.get("resolved_count", 0)),
                ),
                "click_near_change_rate": _safe_div(
                    int(stats.get("click_near_change_count", 0)),
                    int(stats.get("resolved_count", 0)),
                ),
                "score_delta_avg": _safe_div(
                    int(stats.get("score_delta_sum", 0)),
                    int(stats.get("resolved_count", 0)),
                ),
            }
            for score, stats in sorted(by_score.items(), key=lambda item: str(item[0]))
        },
    }


def _repair_score_delta(example: Mapping[str, Any]) -> int:
    outcome = dict(example.get("outcome") or {})
    return _as_int(outcome.get("score_delta"), 0)


def _with_delayed_progress_labels(
    examples: list[Mapping[str, Any]],
    *,
    horizon: int = 5,
) -> list[dict[str, Any]]:
    rows = [dict(example) for example in examples]
    for index, example in enumerate(rows):
        labels = dict(example.get("labels") or {})
        best_distance = -1
        best_delta = 0
        end = min(len(rows), index + max(0, int(horizon)) + 1)
        for future_index in range(index, end):
            delta = _repair_score_delta(rows[future_index])
            future_labels = dict(rows[future_index].get("labels") or {})
            if delta > 0 or bool(future_labels.get("repair_effect_score_progress")):
                best_distance = future_index - index
                best_delta = max(1, int(delta))
                break
        labels["repair_effect_score_progress_h5"] = bool(best_distance >= 0)
        if best_distance >= 0:
            labels["repair_effect_progress_credit_distance"] = int(best_distance)
            labels["repair_effect_progress_credit_score_delta"] = int(best_delta)
        example["labels"] = labels
    return rows


def _repair_positive_offset_templates(
    examples: list[Mapping[str, Any]],
    *,
    label_mode: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[tuple[str, str], ...], Counter[str]] = {}
    for example in examples:
        bundle = _repair_candidate_examples_from_sample(example, label_mode=label_mode)
        features = dict(example.get("features") or {})
        sample_score = _score_key_from_features(features)
        for row in list(bundle.get("rows") or []):
            if not bool(row.get("label")):
                continue
            feature_row = dict(row.get("features") or {})
            key = (
                ("score", sample_score),
                ("candidate_kind_base", str(feature_row.get("candidate_kind_base", "") or "")),
                ("candidate_color", str(feature_row.get("candidate_color", "") or "")),
                ("candidate_target_dx", str(feature_row.get("candidate_target_dx", "") or "")),
                ("candidate_target_dy", str(feature_row.get("candidate_target_dy", "") or "")),
                ("candidate_jitter_dx", str(feature_row.get("candidate_jitter_dx", "") or "")),
                ("candidate_jitter_dy", str(feature_row.get("candidate_jitter_dy", "") or "")),
            )
            counter = buckets.setdefault(key, Counter())
            counter["support"] += 1
            counter[f"score:{sample_score}"] += 1
            if str(feature_row.get("candidate_source", "") or "") == "memory":
                counter["memory_source"] += 1
    rows: list[dict[str, Any]] = []
    for key, counter in buckets.items():
        rows.append(
            {
                "template": dict(key),
                "support": int(counter.get("support", 0)),
                "memory_source_count": int(counter.get("memory_source", 0)),
                "score_histogram": {
                    str(name).split(":", 1)[1]: int(count)
                    for name, count in sorted(counter.items())
                    if str(name).startswith("score:")
                },
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row.get("support", 0) or 0),
            _canonical_json(row.get("template") or {}),
        )
    )
    return rows[:limit]


def _repair_example_has_teacher_template_label(
    example: Mapping[str, Any],
    *,
    label_mode: str,
) -> bool:
    labels = dict(example.get("labels") or {})
    mode = str(label_mode or "teacher_site").strip().lower()
    if mode in {"score_progress", "teacher_score_progress", "score_progress_value"}:
        return bool(labels.get("repair_effect_score_progress"))
    if mode in {
        "score_progress_h5",
        "delayed_score_progress",
        "delayed_score_progress_value",
    }:
        return bool(labels.get("repair_effect_score_progress_h5"))
    return True


def _repair_teacher_candidate_match(
    example: Mapping[str, Any],
) -> dict[str, Any]:
    teacher = dict(example.get("teacher") or {})
    teacher_point = _repair_click_point(teacher.get("payload"))
    if teacher_point is None:
        return {
            "teacher_point": [],
            "exact": False,
            "nearest_distance": -1,
            "candidate_kind_base": "",
            "candidate_source": "",
        }
    nearest_distance: int | None = None
    nearest_candidate: dict[str, Any] = {}
    exact_candidate: dict[str, Any] = {}
    for candidate in list(example.get("visible_click_candidates") or []):
        if not isinstance(candidate, Mapping):
            continue
        try:
            x = int(candidate.get("x"))
            y = int(candidate.get("y"))
        except (TypeError, ValueError):
            continue
        distance = abs(x - teacher_point[0]) + abs(y - teacher_point[1])
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = int(distance)
            nearest_candidate = dict(candidate)
        if distance == 0:
            exact_candidate = dict(candidate)
            break
    candidate = exact_candidate or nearest_candidate
    return {
        "teacher_point": [int(teacher_point[0]), int(teacher_point[1])],
        "exact": bool(exact_candidate),
        "nearest_distance": int(nearest_distance if nearest_distance is not None else -1),
        "candidate_kind_base": _candidate_kind_base(candidate) if candidate else "",
        "candidate_source": str(candidate.get("candidate_source", "") or "") if candidate else "",
    }


def _repair_teacher_offset_templates(
    examples: list[Mapping[str, Any]],
    *,
    label_mode: str,
    limit: int = 20,
    only_missing: bool = False,
) -> list[dict[str, Any]]:
    """Summarize positive teacher click offsets, including candidate misses.

    Candidate rankers can only learn from positives that already exist inside
    the candidate set. This diagnostic exposes score-progress teacher clicks
    that were not generated as visible candidates, so generator coverage can be
    improved before tuning the ranker.
    """

    buckets: dict[tuple[tuple[str, str], ...], Counter[str]] = {}
    for example in examples:
        if not _repair_example_has_teacher_template_label(example, label_mode=label_mode):
            continue
        features = dict(example.get("features") or {})
        teacher = dict(example.get("teacher") or {})
        teacher_point = _repair_click_point(teacher.get("payload"))
        if teacher_point is None:
            continue
        center_col = _as_int(features.get("selected_target_center_col"), 0)
        center_row = _as_int(features.get("selected_target_center_row"), 0)
        teacher_dx = teacher_point[0] - center_col if center_col else 0
        teacher_dy = teacher_point[1] - center_row if center_row else 0
        score_key = _score_key_from_features(features)
        match = _repair_teacher_candidate_match(example)
        exact = bool(match.get("exact"))
        if only_missing and exact:
            continue
        nearest_distance = _as_int(match.get("nearest_distance"), -1)
        key = (
            ("score", score_key),
            ("teacher_dx", str(teacher_dx)),
            ("teacher_dy", str(teacher_dy)),
        )
        counter = buckets.setdefault(key, Counter())
        counter["support"] += 1
        counter["contained" if exact else "missing"] += 1
        if nearest_distance >= 0:
            counter["nearest_distance_sum"] += nearest_distance
        counter[f"nearest_distance:{nearest_distance}"] += 1
        kind = str(match.get("candidate_kind_base", "") or "")
        if kind:
            counter[f"nearest_kind:{kind}"] += 1
    rows: list[dict[str, Any]] = []
    for key, counter in buckets.items():
        support = int(counter.get("support", 0))
        contained = int(counter.get("contained", 0))
        missing = int(counter.get("missing", 0))
        rows.append(
            {
                "template": dict(key),
                "support": support,
                "exact_candidate_containment_count": contained,
                "missing_candidate_count": missing,
                "candidate_containment_rate": _safe_div(contained, support),
                "nearest_distance_avg": _safe_div(
                    int(counter.get("nearest_distance_sum", 0)),
                    support,
                ),
                "nearest_distance_histogram": {
                    str(name).split(":", 1)[1]: int(count)
                    for name, count in sorted(counter.items())
                    if str(name).startswith("nearest_distance:")
                },
                "nearest_kind_histogram": {
                    str(name).split(":", 1)[1]: int(count)
                    for name, count in sorted(counter.items())
                    if str(name).startswith("nearest_kind:")
                },
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row.get("missing_candidate_count", 0) or 0),
            -int(row.get("support", 0) or 0),
            _canonical_json(row.get("template") or {}),
        )
    )
    return rows[:limit]


def _evaluate_repair_candidate_value(
    examples: list[Mapping[str, Any]],
    *,
    feature_key_groups: tuple[tuple[str, ...], ...],
    label_mode: str,
    positive_definition: str,
) -> dict[str, Any]:
    frequency_model = RepairCandidateRanker(
        feature_key_groups=feature_key_groups,
        label_mode=label_mode,
    ).fit(examples)
    prototype_model = RepairCandidatePrototypeRanker(label_mode=label_mode).fit(examples)
    prior_predictions = [
        _rank_repair_candidates_by_progress_prior(example, label_mode=label_mode)
        for example in examples
    ]
    frequency_predictions = [frequency_model.rank(example) for example in examples]
    prototype_predictions = [prototype_model.rank(example) for example in examples]
    return {
        "label_mode": label_mode,
        "positive_definition": positive_definition,
        "prior_only": _repair_candidate_ranker_metrics(examples, prior_predictions),
        "in_sample": _repair_candidate_ranker_metrics(examples, frequency_predictions),
        "prototype_in_sample": _repair_candidate_ranker_metrics(
            examples,
            prototype_predictions,
        ),
        "top_offset_templates": _repair_positive_offset_templates(
            examples,
            label_mode=label_mode,
        ),
        "teacher_offset_templates": _repair_teacher_offset_templates(
            examples,
            label_mode=label_mode,
        ),
        "missing_teacher_offset_templates": _repair_teacher_offset_templates(
            examples,
            label_mode=label_mode,
            only_missing=True,
        ),
        "chronological_holdout": _repair_candidate_chronological_holdout(
            examples,
            feature_key_groups=feature_key_groups,
            label_mode=label_mode,
        ),
        "stratified_holdout": _repair_candidate_stratified_holdout(
            examples,
            feature_key_groups=feature_key_groups,
            label_mode=label_mode,
        ),
        "top_rules": frequency_model.top_rules(),
        "top_prototypes": prototype_model.top_prototypes(),
    }


def _evaluate_repair_candidate_score_progress_value(
    examples: list[Mapping[str, Any]],
    *,
    feature_key_groups: tuple[tuple[str, ...], ...],
) -> dict[str, Any]:
    return _evaluate_repair_candidate_value(
        examples,
        feature_key_groups=feature_key_groups,
        label_mode="score_progress",
        positive_definition="teacher_site_candidate_and_repair_effect_score_progress",
    )


def _evaluate_repair_candidate_delayed_score_progress_value(
    examples: list[Mapping[str, Any]],
    *,
    feature_key_groups: tuple[tuple[str, ...], ...],
    horizon: int = 5,
) -> dict[str, Any]:
    delayed_examples = _with_delayed_progress_labels(examples, horizon=horizon)
    report = _evaluate_repair_candidate_value(
        delayed_examples,
        feature_key_groups=feature_key_groups,
        label_mode="score_progress_h5",
        positive_definition=(
            "teacher_site_candidate_and_repair_effect_score_progress_within_"
            f"{int(horizon)}_repair_steps"
        ),
    )
    report["horizon"] = int(horizon)
    progress_credit_count = sum(
        1
        for example in delayed_examples
        if bool(dict(example.get("labels") or {}).get("repair_effect_score_progress_h5"))
    )
    report["progress_credit_example_count"] = int(progress_credit_count)
    return report


def evaluate_repair_candidate_ranker(
    examples: Iterable[Mapping[str, Any]],
    *,
    feature_key_groups: tuple[tuple[str, ...], ...] = REPAIR_CANDIDATE_RANKER_FEATURE_KEY_GROUPS,
    include_leave_one_out: bool = True,
) -> dict[str, Any]:
    example_list = [dict(example) for example in examples]
    model = RepairCandidateRanker(feature_key_groups=feature_key_groups).fit(example_list)
    prototype_model = RepairCandidatePrototypeRanker().fit(example_list)
    in_sample_predictions = [model.rank(example) for example in example_list]
    prototype_in_sample_predictions = [
        prototype_model.rank(example)
        for example in example_list
    ]
    prior_predictions = [
        _rank_repair_candidates_by_progress_prior(example)
        for example in example_list
    ]

    loo_predictions: list[dict[str, Any]] = []
    prototype_loo_predictions: list[dict[str, Any]] = []
    if include_leave_one_out:
        for index, example in enumerate(example_list):
            train = [other for other_index, other in enumerate(example_list) if other_index != index]
            if not train:
                empty_prediction = {
                    "candidate_count": 0,
                    "positive_exists": False,
                    "top": {},
                    "ranked": [],
                }
                loo_predictions.append(empty_prediction)
                prototype_loo_predictions.append(empty_prediction)
                continue
            fold_model = RepairCandidateRanker(feature_key_groups=feature_key_groups).fit(train)
            loo_predictions.append(fold_model.rank(example))
            fold_prototype_model = RepairCandidatePrototypeRanker().fit(train)
            prototype_loo_predictions.append(fold_prototype_model.rank(example))

    report = {
        "schema_version": "eca_repair_candidate_ranker_eval.v1",
        "example_count": len(example_list),
        "feature_key_groups": [list(group) for group in feature_key_groups],
        "prototype_feature_key_groups": [
            list(group)
            for group in REPAIR_CANDIDATE_PROTOTYPE_FEATURE_KEY_GROUPS
        ],
        "effect_outcome_diagnostics": _repair_effect_outcome_diagnostics(example_list),
        "visible_candidate_diagnostics": _repair_visible_candidate_diagnostics(example_list),
        "score_progress_value": _evaluate_repair_candidate_score_progress_value(
            example_list,
            feature_key_groups=feature_key_groups,
        ),
        "delayed_score_progress_value": _evaluate_repair_candidate_delayed_score_progress_value(
            example_list,
            feature_key_groups=feature_key_groups,
            horizon=5,
        ),
        "prior_only": _repair_candidate_ranker_metrics(example_list, prior_predictions),
        "prototype_in_sample": _repair_candidate_ranker_metrics(
            example_list,
            prototype_in_sample_predictions,
        ),
        "in_sample": _repair_candidate_ranker_metrics(example_list, in_sample_predictions),
        "chronological_holdout": _repair_candidate_chronological_holdout(
            example_list,
            feature_key_groups=feature_key_groups,
        ),
        "stratified_holdout": _repair_candidate_stratified_holdout(
            example_list,
            feature_key_groups=feature_key_groups,
        ),
        "top_rules": model.top_rules(),
        "top_prototypes": prototype_model.top_prototypes(),
    }
    if include_leave_one_out:
        report["leave_one_out"] = _repair_candidate_ranker_metrics(example_list, loo_predictions)
        report["prototype_leave_one_out"] = _repair_candidate_ranker_metrics(
            example_list,
            prototype_loo_predictions,
        )
    else:
        report["leave_one_out"] = {"skipped": True}
        report["prototype_leave_one_out"] = {"skipped": True}
    return report


def _repair_visible_candidate_diagnostics(
    examples: list[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_examples = 0
    teacher_in_visible = 0
    proposal_in_visible = 0
    proposal_teacher_same_candidate = 0
    teacher_previously_executed = 0
    teacher_distances: list[int] = []
    proposal_distances: list[int] = []
    teacher_ranks: list[int] = []
    candidate_counts: list[int] = []
    remaining_counts: list[int] = []
    teacher_kind_histogram: Counter[str] = Counter()
    proposal_kind_histogram: Counter[str] = Counter()
    teacher_band_histogram: Counter[str] = Counter()
    proposal_band_histogram: Counter[str] = Counter()
    candidate_source_histogram: Counter[str] = Counter()
    candidate_band_histogram: Counter[str] = Counter()
    positive_candidate_source_histogram: Counter[str] = Counter()
    positive_candidate_band_histogram: Counter[str] = Counter()
    previously_executed_candidate_count = 0
    previously_proposed_candidate_count = 0
    by_score: dict[str, Counter[str]] = defaultdict(Counter)

    for example in examples:
        features = dict(example.get("features") or {})
        labels = dict(example.get("labels") or {})
        teacher_match = dict(example.get("teacher_site_candidate_match") or {})
        proposal_match = dict(example.get("proposal_site_candidate_match") or {})
        candidate_count = _as_int(
            features.get("visible_click_candidate_count"),
            len(list(example.get("visible_click_candidates") or [])),
        )
        if candidate_count <= 0:
            continue
        visible_candidates = [
            dict(candidate)
            for candidate in list(example.get("visible_click_candidates") or [])
            if isinstance(candidate, Mapping)
        ]
        for candidate in visible_candidates:
            source = str(candidate.get("candidate_source", "") or "unknown")
            band = str(candidate.get("candidate_band", "") or "unknown")
            candidate_source_histogram[source] += 1
            candidate_band_histogram[band] += 1
            if bool(candidate.get("candidate_previously_executed")):
                previously_executed_candidate_count += 1
            if bool(candidate.get("candidate_previously_proposed")):
                previously_proposed_candidate_count += 1
        candidate_examples += 1
        score_key = _score_key_from_features(features)
        by_score[score_key]["candidate_example_count"] += 1
        candidate_counts.append(candidate_count)
        remaining_counts.append(_as_int(features.get("remaining_visible_click_candidate_count"), 0))

        teacher_distance = _as_int(
            teacher_match.get("distance"),
            _as_int(features.get("teacher_visible_click_candidate_distance"), -1),
        )
        proposal_distance = _as_int(
            proposal_match.get("distance"),
            _as_int(features.get("proposal_visible_click_candidate_distance"), -1),
        )
        teacher_rank = _as_int(
            teacher_match.get("rank"),
            _as_int(features.get("teacher_visible_click_candidate_rank"), -1),
        )
        if teacher_distance >= 0:
            teacher_distances.append(teacher_distance)
        if proposal_distance >= 0:
            proposal_distances.append(proposal_distance)
        if teacher_rank >= 0:
            teacher_ranks.append(teacher_rank)

        teacher_in = bool(labels.get("teacher_site_in_visible_candidates")) or teacher_distance == 0
        proposal_in = bool(proposal_match.get("in_candidates")) or proposal_distance == 0
        if teacher_in:
            teacher_in_visible += 1
            by_score[score_key]["teacher_in_visible_candidate"] += 1
        if proposal_in:
            proposal_in_visible += 1
            by_score[score_key]["proposal_in_visible_candidate"] += 1
        if bool(labels.get("teacher_site_previously_executed")):
            teacher_previously_executed += 1
            by_score[score_key]["teacher_site_previously_executed"] += 1
        if teacher_distance >= 0:
            by_score[score_key]["teacher_distance_sum"] += int(teacher_distance)
            by_score[score_key]["teacher_distance_count"] += 1

        teacher_kind = str(
            teacher_match.get("kind")
            or features.get("teacher_visible_click_candidate_kind")
            or ""
        )
        proposal_kind = str(
            proposal_match.get("kind")
            or features.get("proposal_visible_click_candidate_kind")
            or ""
        )
        teacher_candidate_id = str(teacher_match.get("candidate_id", "") or "")
        proposal_candidate_id = str(proposal_match.get("candidate_id", "") or "")
        teacher_candidate = {}
        proposal_candidate = {}
        if teacher_candidate_id or proposal_candidate_id:
            for candidate in visible_candidates:
                candidate_id = str(candidate.get("candidate_id", "") or "")
                if teacher_candidate_id and candidate_id == teacher_candidate_id:
                    teacher_candidate = dict(candidate)
                if proposal_candidate_id and candidate_id == proposal_candidate_id:
                    proposal_candidate = dict(candidate)
        if teacher_kind:
            teacher_kind_histogram[teacher_kind] += 1
        if proposal_kind:
            proposal_kind_histogram[proposal_kind] += 1
        teacher_band = str(
            teacher_match.get("candidate_band")
            or teacher_candidate.get("candidate_band")
            or features.get("teacher_visible_click_candidate_band")
            or ""
        )
        proposal_band = str(
            proposal_match.get("candidate_band")
            or proposal_candidate.get("candidate_band")
            or features.get("proposal_visible_click_candidate_band")
            or ""
        )
        if teacher_band:
            teacher_band_histogram[teacher_band] += 1
        if proposal_band:
            proposal_band_histogram[proposal_band] += 1
        if bool(teacher_match.get("in_candidates")):
            candidate = dict(teacher_candidate)
            if candidate:
                positive_candidate_source_histogram[
                    str(candidate.get("candidate_source", "") or "unknown")
                ] += 1
                positive_candidate_band_histogram[
                    str(candidate.get("candidate_band", "") or "unknown")
                ] += 1

        if teacher_candidate_id and teacher_candidate_id == proposal_candidate_id:
            proposal_teacher_same_candidate += 1

    return {
        "candidate_example_count": int(candidate_examples),
        "teacher_in_visible_candidate_rate": _safe_div(teacher_in_visible, candidate_examples),
        "proposal_in_visible_candidate_rate": _safe_div(proposal_in_visible, candidate_examples),
        "proposal_teacher_same_candidate_rate": _safe_div(
            proposal_teacher_same_candidate,
            candidate_examples,
        ),
        "teacher_site_previously_executed_rate": _safe_div(
            teacher_previously_executed,
            candidate_examples,
        ),
        "teacher_nearest_distance_avg": _safe_div(sum(teacher_distances), len(teacher_distances)),
        "proposal_nearest_distance_avg": _safe_div(sum(proposal_distances), len(proposal_distances)),
        "teacher_visible_candidate_rank_avg": _safe_div(sum(teacher_ranks), len(teacher_ranks)),
        "teacher_rank_le_1_rate": _safe_div(
            sum(1 for rank in teacher_ranks if rank <= 1),
            len(teacher_ranks),
        ),
        "visible_click_candidate_count_avg": _safe_div(sum(candidate_counts), len(candidate_counts)),
        "remaining_visible_click_candidate_count_avg": _safe_div(
            sum(remaining_counts),
            len(remaining_counts),
        ),
        "teacher_candidate_kind_histogram": dict(teacher_kind_histogram),
        "proposal_candidate_kind_histogram": dict(proposal_kind_histogram),
        "teacher_candidate_band_histogram": dict(teacher_band_histogram),
        "proposal_candidate_band_histogram": dict(proposal_band_histogram),
        "candidate_source_histogram": dict(candidate_source_histogram),
        "candidate_band_histogram": dict(candidate_band_histogram),
        "positive_candidate_source_histogram": dict(positive_candidate_source_histogram),
        "positive_candidate_band_histogram": dict(positive_candidate_band_histogram),
        "previously_executed_candidate_rate": _safe_div(
            previously_executed_candidate_count,
            sum(candidate_source_histogram.values()),
        ),
        "previously_proposed_candidate_rate": _safe_div(
            previously_proposed_candidate_count,
            sum(candidate_source_histogram.values()),
        ),
        "by_score": {
            score: {
                "candidate_example_count": int(stats.get("candidate_example_count", 0)),
                "teacher_in_visible_candidate_rate": _safe_div(
                    int(stats.get("teacher_in_visible_candidate", 0)),
                    int(stats.get("candidate_example_count", 0)),
                ),
                "proposal_in_visible_candidate_rate": _safe_div(
                    int(stats.get("proposal_in_visible_candidate", 0)),
                    int(stats.get("candidate_example_count", 0)),
                ),
                "teacher_site_previously_executed_rate": _safe_div(
                    int(stats.get("teacher_site_previously_executed", 0)),
                    int(stats.get("candidate_example_count", 0)),
                ),
                "teacher_nearest_distance_avg": _safe_div(
                    int(stats.get("teacher_distance_sum", 0)),
                    int(stats.get("teacher_distance_count", 0)),
                ),
            }
            for score, stats in sorted(by_score.items(), key=lambda item: str(item[0]))
        },
    }


def evaluate_repair_payload_student(
    examples: Iterable[Mapping[str, Any]],
    *,
    feature_key_groups: tuple[tuple[str, ...], ...] = REPAIR_STUDENT_FEATURE_KEY_GROUPS,
) -> dict[str, Any]:
    example_list = [dict(example) for example in examples]
    model = RepairPayloadStudent(feature_key_groups=feature_key_groups).fit(example_list)
    in_sample_predictions = [model.predict(example) for example in example_list]

    loo_predictions: list[dict[str, Any]] = []
    for index, example in enumerate(example_list):
        train = [other for other_index, other in enumerate(example_list) if other_index != index]
        if not train:
            loo_predictions.append(
                {"available": False, "target": {}, "support": 0, "feature_keys": []}
            )
            continue
        fold_model = RepairPayloadStudent(feature_key_groups=feature_key_groups).fit(train)
        loo_predictions.append(fold_model.predict(example))

    label_totals = Counter()
    for example in example_list:
        labels = dict(example.get("labels") or {})
        for key in (
            "action_agrees_before",
            "payload_compared",
            "payload_agrees_before",
        ):
            if bool(labels.get(key)):
                label_totals[key] += 1
    return {
        "schema_version": "eca_repair_payload_student_eval.v1",
        "example_count": len(example_list),
        "feature_key_groups": [list(group) for group in feature_key_groups],
        "baseline": {
            "action_agreement_before_rate": _safe_div(
                int(label_totals.get("action_agrees_before", 0)),
                len(example_list),
            ),
            "payload_compared_count": int(label_totals.get("payload_compared", 0)),
            "payload_agreement_before_rate": _safe_div(
                int(label_totals.get("payload_agrees_before", 0)),
                int(label_totals.get("payload_compared", 0)),
            ),
        },
        "in_sample": _repair_student_shadow_metrics(example_list, in_sample_predictions),
        "leave_one_out": _repair_student_shadow_metrics(example_list, loo_predictions),
        "feature_ambiguity": _repair_student_feature_ambiguity(
            example_list,
            feature_key_groups,
        ),
        "visible_candidate_diagnostics": _repair_visible_candidate_diagnostics(example_list),
        "top_rules": model.top_rules(),
    }


def _aggregate_repair_distillation_count(summaries: Iterable[Mapping[str, Any]]) -> int:
    total = 0
    for summary in summaries:
        bridge_summary = summary.get("perception_solver_bridge")
        if not isinstance(bridge_summary, Mapping):
            bridge_summary = summary
        count = int(bridge_summary.get("repair_distillation_example_count", 0) or 0)
        if count:
            total += count
            continue
        samples = bridge_summary.get("repair_distillation_samples")
        if isinstance(samples, list):
            total += len(samples)
    return total


def build_repair_student_shadow_report(
    episode_summaries: Iterable[Mapping[str, Any]],
    *,
    feature_key_groups: tuple[tuple[str, ...], ...] = REPAIR_STUDENT_FEATURE_KEY_GROUPS,
) -> dict[str, Any]:
    summary_list = [dict(summary) for summary in episode_summaries if isinstance(summary, Mapping)]
    examples = extract_repair_distillation_examples(summary_list)
    eval_report = evaluate_repair_payload_student(
        examples,
        feature_key_groups=feature_key_groups,
    )
    eval_report["schema_version"] = "eca_repair_student_shadow_report.v1"
    eval_report["stored_example_count"] = len(examples)
    eval_report["aggregate_example_count"] = _aggregate_repair_distillation_count(summary_list)
    eval_report["source_summary_count"] = len(summary_list)
    return eval_report


def build_repair_candidate_ranker_report(
    episode_summaries: Iterable[Mapping[str, Any]],
    *,
    feature_key_groups: tuple[tuple[str, ...], ...] = REPAIR_CANDIDATE_RANKER_FEATURE_KEY_GROUPS,
    include_leave_one_out: bool = True,
) -> dict[str, Any]:
    summary_list = [dict(summary) for summary in episode_summaries if isinstance(summary, Mapping)]
    examples = extract_repair_distillation_examples(summary_list)
    report = evaluate_repair_candidate_ranker(
        examples,
        feature_key_groups=feature_key_groups,
        include_leave_one_out=include_leave_one_out,
    )
    report["schema_version"] = "eca_repair_candidate_ranker_report.v1"
    report["stored_example_count"] = len(examples)
    report["aggregate_example_count"] = _aggregate_repair_distillation_count(summary_list)
    report["source_summary_count"] = len(summary_list)
    return report


def build_repair_prior_chain_value_report(
    episode_summaries: Iterable[Mapping[str, Any]],
    *,
    feature_key_groups: tuple[tuple[str, ...], ...] = REPAIR_PRIOR_CHAIN_VALUE_FEATURE_KEY_GROUPS,
) -> dict[str, Any]:
    summary_list = [dict(summary) for summary in episode_summaries if isinstance(summary, Mapping)]
    examples = extract_repair_prior_chain_value_examples(summary_list)
    report = evaluate_repair_prior_chain_value(
        examples,
        feature_key_groups=feature_key_groups,
    )
    report["schema_version"] = "eca_repair_prior_chain_value_report.v1"
    report["stored_example_count"] = len(examples)
    report["source_summary_count"] = len(summary_list)
    return report


def build_repair_student_shadow_report_from_roots(
    roots: Iterable[str | Path],
    *,
    limit: int | None = None,
    feature_key_groups: tuple[tuple[str, ...], ...] = REPAIR_STUDENT_FEATURE_KEY_GROUPS,
) -> dict[str, Any]:
    summaries = load_repair_distillation_episode_summaries(roots, limit=limit)
    report = build_repair_student_shadow_report(
        summaries,
        feature_key_groups=feature_key_groups,
    )
    report["source_paths"] = [
        str(summary.get("_source_path", "") or "")
        for summary in summaries
        if str(summary.get("_source_path", "") or "")
    ]
    return report


def build_repair_candidate_ranker_report_from_roots(
    roots: Iterable[str | Path],
    *,
    limit: int | None = None,
    feature_key_groups: tuple[tuple[str, ...], ...] = REPAIR_CANDIDATE_RANKER_FEATURE_KEY_GROUPS,
    include_leave_one_out: bool = True,
) -> dict[str, Any]:
    summaries = load_repair_distillation_episode_summaries(roots, limit=limit)
    report = build_repair_candidate_ranker_report(
        summaries,
        feature_key_groups=feature_key_groups,
        include_leave_one_out=include_leave_one_out,
    )
    report["source_paths"] = [
        str(summary.get("_source_path", "") or "")
        for summary in summaries
        if str(summary.get("_source_path", "") or "")
    ]
    return report


def build_repair_prior_chain_value_report_from_roots(
    roots: Iterable[str | Path],
    *,
    limit: int | None = None,
    feature_key_groups: tuple[tuple[str, ...], ...] = REPAIR_PRIOR_CHAIN_VALUE_FEATURE_KEY_GROUPS,
) -> dict[str, Any]:
    summaries = load_repair_distillation_episode_summaries(roots, limit=limit)
    report = build_repair_prior_chain_value_report(
        summaries,
        feature_key_groups=feature_key_groups,
    )
    report["source_paths"] = [
        str(summary.get("_source_path", "") or "")
        for summary in summaries
        if str(summary.get("_source_path", "") or "")
    ]
    return report


def write_repair_student_shadow_report(
    roots: Iterable[str | Path],
    output: str | Path,
    *,
    limit: int | None = None,
) -> Path:
    report = build_repair_student_shadow_report_from_roots(roots, limit=limit)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def write_repair_prior_chain_value_report(
    roots: Iterable[str | Path],
    output: str | Path,
    *,
    limit: int | None = None,
) -> Path:
    report = build_repair_prior_chain_value_report_from_roots(roots, limit=limit)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def write_repair_candidate_ranker_report(
    roots: Iterable[str | Path],
    output: str | Path,
    *,
    limit: int | None = None,
    include_leave_one_out: bool = True,
) -> Path:
    report = build_repair_candidate_ranker_report_from_roots(
        roots,
        limit=limit,
        include_leave_one_out=include_leave_one_out,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
