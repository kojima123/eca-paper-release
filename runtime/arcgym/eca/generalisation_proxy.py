"""Offline perturbation and unknown-transfer proxy for ECA modeling data.

This is intentionally a proxy, not a real environment perturbation harness. It
measures whether the learned policy bundle can re-score mid-trajectory states,
recovery-like states, and held-out games from frame-conditioned logs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from arcgym.eca.policy_bundle import (
    collect_policy_examples,
    evaluate_policy_bundle,
    predict_action,
    predict_mode,
    train_policy_bundle,
)


RECOVERY_MODES = {"recover_branch", "reclaim", "rotate", "hedge", "budget_conserve"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _game_id(example: Mapping[str, Any]) -> str:
    return str(example.get("_game_id", "") or "unknown")


def _split_examples_by_game(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in examples:
        out[_game_id(row)].append(row)
    return dict(sorted(out.items()))


def _filter_game(rows: list[dict[str, Any]], game: str, *, include: bool) -> list[dict[str, Any]]:
    if include:
        return [row for row in rows if _game_id(row) == game]
    return [row for row in rows if _game_id(row) != game]


def _train_from_example_dict(examples: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return train_policy_bundle(
        candidate_examples=examples.get("candidate_examples") or [],
        risk_examples=examples.get("risk_examples") or [],
        mode_examples=examples.get("mode_examples") or [],
        action_examples=examples.get("action_examples") or [],
        metadata={"source": "generalisation_proxy"},
    )


def _subset_examples(
    examples: Mapping[str, list[dict[str, Any]]],
    game: str,
    *,
    include: bool,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "candidate_examples": _filter_game(list(examples.get("candidate_examples") or []), game, include=include),
        "risk_examples": _filter_game(list(examples.get("risk_examples") or []), game, include=include),
        "mode_examples": _filter_game(list(examples.get("mode_examples") or []), game, include=include),
        "action_examples": _filter_game(list(examples.get("action_examples") or []), game, include=include),
    }


def _mode_features(row: Mapping[str, Any]) -> dict[str, Any]:
    return dict(row.get("features") or {})


def _is_mid_state(row: Mapping[str, Any]) -> bool:
    return _as_int(row.get("step_id"), 0) > 0


def _is_recovery_like_mode_example(row: Mapping[str, Any]) -> bool:
    features = _mode_features(row)
    return bool(
        _as_bool(features.get("recovery_in_window"))
        or _as_bool(features.get("selected_target_empty_with_candidate_pool"))
        or str(features.get("no_target_disposition", "") or "")
        or _as_int(features.get("no_change_streak"), 0) >= 3
    )


def _is_wrong_branch_window(row: Mapping[str, Any]) -> bool:
    return _as_bool(_mode_features(row).get("recovery_in_window"))


def _family_accuracy(policy_bundle: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    correct = 0
    for row in rows:
        gold = str((row.get("label") or {}).get("action_family", "") or "")
        if not gold or gold == "unknown":
            continue
        total += 1
        pred = predict_action(policy_bundle, row)
        if str(pred.get("action_family", "") or "") == gold:
            correct += 1
    return {
        "example_count": len(rows),
        "family_total": total,
        "family_accuracy": _safe_div(correct, total),
    }


def _recovery_mode_rate(policy_bundle: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected: Counter[str] = Counter()
    recovery_count = 0
    for row in rows:
        pred = predict_mode(policy_bundle, row)
        mode = str(pred.get("selected_mode", "") or "")
        selected[mode] += 1
        if mode in RECOVERY_MODES:
            recovery_count += 1
    return {
        "example_count": len(rows),
        "recovery_mode_rate": _safe_div(recovery_count, len(rows)),
        "prediction_counts": dict(sorted(selected.items())),
    }


def build_perturbation_proxy_report(
    policy_bundle: Mapping[str, Any],
    examples: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    mode_rows = list(examples.get("mode_examples") or [])
    action_rows = list(examples.get("action_examples") or [])
    mid_state_actions = [row for row in action_rows if _is_mid_state(row)]
    recovery_like_modes = [row for row in mode_rows if _is_recovery_like_mode_example(row)]
    wrong_branch_modes = [row for row in mode_rows if _is_wrong_branch_window(row)]
    return {
        "schema_version": "eca_perturbation_proxy_report.v1",
        "description": "offline proxy over mid-state and recovery-like logged states; not injected environment perturbation",
        "mid_state_replan": _family_accuracy(policy_bundle, mid_state_actions),
        "off_plan_recovery_proxy": _recovery_mode_rate(policy_bundle, recovery_like_modes),
        "wrong_branch_recovery_proxy": _recovery_mode_rate(policy_bundle, wrong_branch_modes),
    }


def build_unknown_transfer_report(
    examples: Mapping[str, list[dict[str, Any]]],
    *,
    holdout_game: str | None = None,
) -> dict[str, Any]:
    games = sorted(
        {
            _game_id(row)
            for key in ("candidate_examples", "risk_examples", "mode_examples", "action_examples")
            for row in list(examples.get(key) or [])
        }
    )
    if holdout_game:
        games = [holdout_game] if holdout_game in games else []

    rows: list[dict[str, Any]] = []
    for game in games:
        train = _subset_examples(examples, game, include=False)
        test = _subset_examples(examples, game, include=True)
        train_count = sum(len(train[key]) for key in train)
        test_count = sum(len(test[key]) for key in test)
        if train_count <= 0 or test_count <= 0:
            rows.append(
                {
                    "holdout_game": game,
                    "status": "skipped",
                    "reason": "empty_train_or_test_split",
                    "train_example_count": train_count,
                    "test_example_count": test_count,
                }
            )
            continue
        bundle = _train_from_example_dict(train)
        eval_report = evaluate_policy_bundle(
            bundle,
            candidate_examples=test["candidate_examples"],
            risk_examples=test["risk_examples"],
            mode_examples=test["mode_examples"],
            action_examples=test["action_examples"],
        )
        rows.append(
            {
                "holdout_game": game,
                "status": "ok",
                "train_example_count": train_count,
                "test_example_count": test_count,
                "eval": eval_report,
            }
        )

    evaluated = [row for row in rows if row.get("status") == "ok"]
    return {
        "schema_version": "eca_unknown_transfer_report.v1",
        "holdout_mode": "single_game" if holdout_game else "leave_one_game_out",
        "game_count": len(games),
        "evaluated_game_count": len(evaluated),
        "rows": rows,
    }


def build_generalisation_proxy_report(
    roots: Iterable[Path],
    *,
    limit: int | None = None,
    holdout_game: str | None = None,
) -> dict[str, Any]:
    collected = collect_policy_examples(roots, limit=limit)
    examples = {
        "candidate_examples": list(collected.get("candidate_examples") or []),
        "risk_examples": list(collected.get("risk_examples") or []),
        "mode_examples": list(collected.get("mode_examples") or []),
        "action_examples": list(collected.get("action_examples") or []),
    }
    policy_bundle = _train_from_example_dict(examples)
    full_eval = evaluate_policy_bundle(
        policy_bundle,
        candidate_examples=examples["candidate_examples"],
        risk_examples=examples["risk_examples"],
        mode_examples=examples["mode_examples"],
        action_examples=examples["action_examples"],
    )
    game_example_counts = {
        key: {game: len(rows) for game, rows in _split_examples_by_game(list(examples[key])).items()}
        for key in examples
    }
    return {
        "schema_version": "eca_generalisation_proxy_report.v1",
        "run_dirs": list(collected.get("run_dirs") or []),
        "game_counts": dict(collected.get("game_counts") or {}),
        "game_example_counts": game_example_counts,
        "policy_eval_in_sample": full_eval,
        "perturbation_proxy": build_perturbation_proxy_report(policy_bundle, examples),
        "unknown_transfer": build_unknown_transfer_report(
            examples,
            holdout_game=holdout_game,
        ),
    }
