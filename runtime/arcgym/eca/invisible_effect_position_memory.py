"""Invisible-effect position memory diagnostic sensor.

This sensor records interaction evidence for positions whose effect cannot be
trusted from static appearance alone. It consumes moving-sprite runtime trace
packets and stays diagnostic/shadow-only: no planner connection, ranking, or
action generation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


INVISIBLE_EFFECT_POSITION_MEMORY_DIAGNOSTIC_SCHEMA_VERSION = (
    "eca_invisible_effect_position_memory_diagnostic_sensor.v1"
)
INVISIBLE_EFFECT_POSITION_MEMORY_POSTERIOR_SCHEMA_VERSION = (
    "eca_invisible_effect_position_memory_posterior.v1"
)
INVISIBLE_EFFECT_POSITION_MEMORY_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION = (
    "eca_invisible_effect_position_memory_guarded_planner_connection.v1"
)


def summarize_invisible_effect_position_memory_packets(
    packets: Iterable[Mapping[str, Any]] | Any,
    *,
    repeated_noop_threshold: int = 2,
    delayed_effect_window: int = 3,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Summarize hidden-block / delayed-effect position evidence.

    A stable no-op is an action with no sprite movement, no score change, and
    unchanged component count. Repeated stable no-ops at the same site are the
    first diagnostic sign that static perception may be missing a wall/effect.
    A later effect near the same site is tracked separately as delayed-effect
    evidence. Both remain observational.
    """

    packet_rows = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    repeat_min = max(1, int(repeated_noop_threshold))
    delayed_window = max(1, int(delayed_effect_window))

    by_game: dict[str, Counter[str]] = {}
    by_position_key: dict[str, Counter[str]] = defaultdict(Counter)
    noops_by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    samples: list[dict[str, Any]] = []
    delayed_seen: set[tuple[str, str, int]] = set()

    counts: Counter[str] = Counter()
    rows_by_key: dict[str, dict[str, Any]] = {}

    for packet in sorted(packet_rows, key=_packet_sort_key):
        game_id = _game_id(packet)
        game_counter = by_game.setdefault(game_id, Counter())
        game_counter["packet_count"] += 1
        counts["packet_count"] += 1

        step = _safe_int(packet.get("step"))
        action_name = str(packet.get("action_name", "") or "")
        move_count = _safe_int(packet.get("move_count"))
        score_delta = _safe_int(packet.get("score_delta"))
        before_count = _safe_int(packet.get("before_component_count"))
        after_count = _safe_int(packet.get("after_component_count"))
        component_count_known = before_count > 0 or after_count > 0
        component_count_changed = component_count_known and before_count != after_count
        effectful = move_count > 0 or score_delta > 0 or component_count_changed
        site = _site_from_packet(packet)
        position_key = _position_key(packet, site=site)

        if move_count == 0:
            counts["no_move_packet_count"] += 1
            game_counter["no_move_packet_count"] += 1

        stable_noop = (
            move_count == 0
            and score_delta == 0
            and not component_count_changed
            and bool(action_name)
        )
        if stable_noop:
            counts["stable_noop_packet_count"] += 1
            game_counter["stable_noop_packet_count"] += 1
            if site is None:
                counts["ambiguous_no_position_count"] += 1
                game_counter["ambiguous_no_position_count"] += 1
            else:
                counts["invisible_block_candidate_count"] += 1
                game_counter["invisible_block_candidate_count"] += 1
            key_counter = by_position_key[position_key]
            key_counter["noop_observation_count"] += 1
            key_counter[f"action::{action_name}"] += 1
            key_counter[f"game::{game_id}"] += 1
            record = rows_by_key.setdefault(
                position_key,
                {
                    "position_key": position_key,
                    "game_id": game_id,
                    "site": list(site) if site is not None else None,
                    "action_names": [],
                    "first_step": step,
                    "last_step": step,
                    "noop_observation_count": 0,
                    "delayed_effect_observed": False,
                    "posterior_state": "single_noop_observation",
                },
            )
            if action_name not in record["action_names"]:
                record["action_names"].append(action_name)
            record["first_step"] = min(_safe_int(record["first_step"]), step)
            record["last_step"] = max(_safe_int(record["last_step"]), step)
            record["noop_observation_count"] += 1
            noops_by_game[game_id].append(
                {
                    "position_key": position_key,
                    "step": step,
                    "site": site,
                    "action_name": action_name,
                }
            )
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "kind": "stable_noop",
                        "game_id": game_id,
                        "step": step,
                        "position_key": position_key,
                        "site": list(site) if site is not None else None,
                        "action_name": action_name,
                    }
                )

        if effectful:
            counts["effectful_packet_count"] += 1
            game_counter["effectful_packet_count"] += 1
            for prior in noops_by_game.get(game_id, []):
                if step <= _safe_int(prior.get("step")):
                    continue
                if step - _safe_int(prior.get("step")) > delayed_window:
                    continue
                if not _same_site_or_unknown(site, prior.get("site")):
                    continue
                delayed_key = (game_id, str(prior.get("position_key")), step)
                if delayed_key in delayed_seen:
                    continue
                delayed_seen.add(delayed_key)
                counts["delayed_effect_position_candidate_count"] += 1
                game_counter["delayed_effect_position_candidate_count"] += 1
                position_key_prior = str(prior.get("position_key"))
                rows_by_key.setdefault(
                    position_key_prior,
                    {
                        "position_key": position_key_prior,
                        "game_id": game_id,
                        "site": (
                            list(prior["site"])
                            if isinstance(prior.get("site"), tuple)
                            else None
                        ),
                        "action_names": [],
                        "first_step": _safe_int(prior.get("step")),
                        "last_step": _safe_int(prior.get("step")),
                        "noop_observation_count": 0,
                        "delayed_effect_observed": False,
                        "posterior_state": "single_noop_observation",
                    },
                )
                rows_by_key[position_key_prior]["delayed_effect_observed"] = True
                if len(samples) < sample_limit:
                    samples.append(
                        {
                            "kind": "delayed_effect_after_noop",
                            "game_id": game_id,
                            "noop_step": _safe_int(prior.get("step")),
                            "effect_step": step,
                            "position_key": position_key_prior,
                            "effect_move_count": move_count,
                            "effect_score_delta": score_delta,
                        }
                    )

    repeated_key_count = 0
    for key, record in rows_by_key.items():
        noop_count = _safe_int(record.get("noop_observation_count"))
        repeated = noop_count >= repeat_min
        delayed = bool(record.get("delayed_effect_observed"))
        if repeated:
            repeated_key_count += 1
        if delayed:
            record["posterior_state"] = "delayed_effect_candidate"
        elif repeated:
            record["posterior_state"] = "repeated_noop_candidate"
        else:
            record["posterior_state"] = "single_noop_observation"
        by_position_key[key]["repeated_noop_candidate"] = int(repeated)
        by_position_key[key]["delayed_effect_candidate"] = int(delayed)

    counts["repeated_invisible_position_count"] = repeated_key_count
    counts["position_memory_key_count"] = len(rows_by_key)

    if counts["delayed_effect_position_candidate_count"] > 0:
        promotion_state = "invisible_effect_position_memory_delayed_effect_observed"
    elif counts["repeated_invisible_position_count"] > 0:
        promotion_state = "invisible_effect_position_memory_repeated_noop_observed"
    elif counts["stable_noop_packet_count"] > 0:
        promotion_state = "invisible_effect_position_memory_noop_observed"
    elif packet_rows:
        promotion_state = "invisible_effect_position_memory_no_candidates"
    else:
        promotion_state = "invisible_effect_position_memory_no_packets"

    return {
        "schema_version": INVISIBLE_EFFECT_POSITION_MEMORY_DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_only": True,
        "shadow_only": True,
        "packet_count": counts["packet_count"],
        "no_move_packet_count": counts["no_move_packet_count"],
        "stable_noop_packet_count": counts["stable_noop_packet_count"],
        "effectful_packet_count": counts["effectful_packet_count"],
        "invisible_block_candidate_count": counts["invisible_block_candidate_count"],
        "ambiguous_no_position_count": counts["ambiguous_no_position_count"],
        "repeated_invisible_position_count": counts[
            "repeated_invisible_position_count"
        ],
        "delayed_effect_position_candidate_count": counts[
            "delayed_effect_position_candidate_count"
        ],
        "position_memory_key_count": counts["position_memory_key_count"],
        "repeated_noop_threshold": repeat_min,
        "delayed_effect_window": delayed_window,
        "promotion_state": promotion_state,
        "by_game": {
            game: dict(sorted(counter.items()))
            for game, counter in sorted(by_game.items())
        },
        "rows": sorted(
            rows_by_key.values(),
            key=lambda row: (
                str(row.get("game_id", "")),
                str(row.get("position_key", "")),
            ),
        ),
        "samples": samples,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
    }


def summarize_invisible_effect_position_memory_posterior(
    packets_or_diagnostic_summary: Iterable[Mapping[str, Any]] | Mapping[str, Any] | Any,
    *,
    repeated_noop_threshold: int = 2,
    delayed_effect_window: int = 3,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Build a posterior over site-resolved vs ambiguous position memory rows."""

    diagnostic = _diagnostic_summary_from_input(
        packets_or_diagnostic_summary,
        repeated_noop_threshold=repeated_noop_threshold,
        delayed_effect_window=delayed_effect_window,
    )
    rows = [
        dict(row)
        for row in list(diagnostic.get("rows") or [])
        if isinstance(row, Mapping)
    ]
    by_game: dict[str, Counter[str]] = {}
    state_counts: Counter[str] = Counter()
    entries: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for row in rows:
        game_id = str(row.get("game_id", "") or "unknown").strip().lower()
        game_counter = by_game.setdefault(game_id, Counter())
        site = _site_from_value(row.get("site"))
        row_state = str(row.get("posterior_state", "") or "")
        if site is None:
            posterior_state = "ambiguous_action_only"
            counts["ambiguous_action_only_entry_count"] += 1
            game_counter["ambiguous_action_only_entry_count"] += 1
        elif row_state == "delayed_effect_candidate":
            posterior_state = "site_resolved_delayed_effect_candidate"
            counts["site_resolved_delayed_effect_candidate_count"] += 1
            counts["admissible_site_resolved_entry_count"] += 1
            game_counter["site_resolved_delayed_effect_candidate_count"] += 1
            game_counter["admissible_site_resolved_entry_count"] += 1
        elif row_state == "repeated_noop_candidate":
            posterior_state = "site_resolved_repeated_noop_candidate"
            counts["site_resolved_repeated_noop_candidate_count"] += 1
            counts["admissible_site_resolved_entry_count"] += 1
            game_counter["site_resolved_repeated_noop_candidate_count"] += 1
            game_counter["admissible_site_resolved_entry_count"] += 1
        else:
            posterior_state = "site_resolved_single_noop_observation"
            counts["site_resolved_single_noop_observation_count"] += 1
            game_counter["site_resolved_single_noop_observation_count"] += 1
        if site is not None:
            counts["site_resolved_entry_count"] += 1
            game_counter["site_resolved_entry_count"] += 1
        state_counts[posterior_state] += 1
        game_counter["posterior_entry_count"] += 1
        entry = {
            "position_key": str(row.get("position_key", "") or ""),
            "game_id": game_id,
            "site": list(site) if site is not None else None,
            "action_names": list(row.get("action_names") or []),
            "noop_observation_count": _safe_int(row.get("noop_observation_count")),
            "delayed_effect_observed": bool(row.get("delayed_effect_observed")),
            "diagnostic_state": row_state,
            "posterior_state": posterior_state,
            "admissible_for_guarded_connection": bool(
                site is not None
                and posterior_state
                in {
                    "site_resolved_delayed_effect_candidate",
                    "site_resolved_repeated_noop_candidate",
                }
            ),
        }
        entries.append(entry)
        if len(samples) < sample_limit:
            samples.append(dict(entry))

    counts["posterior_entry_count"] = len(entries)
    counts["blocked_ambiguous_entry_count"] = counts[
        "ambiguous_action_only_entry_count"
    ]
    if counts["site_resolved_entry_count"] and counts["ambiguous_action_only_entry_count"]:
        promotion_state = (
            "invisible_effect_position_memory_posterior_site_resolved_with_ambiguous_residual"
        )
    elif counts["site_resolved_entry_count"]:
        promotion_state = "invisible_effect_position_memory_posterior_site_resolved"
    elif counts["ambiguous_action_only_entry_count"]:
        promotion_state = "invisible_effect_position_memory_posterior_ambiguous_only"
    elif rows:
        promotion_state = "invisible_effect_position_memory_posterior_no_admissible_entries"
    else:
        promotion_state = "invisible_effect_position_memory_posterior_no_rows"

    return {
        "schema_version": INVISIBLE_EFFECT_POSITION_MEMORY_POSTERIOR_SCHEMA_VERSION,
        "diagnostic_only": True,
        "shadow_only": True,
        "packet_count": _safe_int(diagnostic.get("packet_count")),
        "posterior_entry_count": counts["posterior_entry_count"],
        "site_resolved_entry_count": counts["site_resolved_entry_count"],
        "ambiguous_action_only_entry_count": counts[
            "ambiguous_action_only_entry_count"
        ],
        "blocked_ambiguous_entry_count": counts["blocked_ambiguous_entry_count"],
        "admissible_site_resolved_entry_count": counts[
            "admissible_site_resolved_entry_count"
        ],
        "site_resolved_single_noop_observation_count": counts[
            "site_resolved_single_noop_observation_count"
        ],
        "site_resolved_repeated_noop_candidate_count": counts[
            "site_resolved_repeated_noop_candidate_count"
        ],
        "site_resolved_delayed_effect_candidate_count": counts[
            "site_resolved_delayed_effect_candidate_count"
        ],
        "state_counts": dict(sorted(state_counts.items())),
        "by_game": {
            game: dict(sorted(counter.items()))
            for game, counter in sorted(by_game.items())
        },
        "entries": sorted(
            entries,
            key=lambda item: (
                str(item.get("game_id", "")),
                str(item.get("position_key", "")),
            ),
        ),
        "samples": samples,
        "promotion_state": promotion_state,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
        "next_planner_contract": {
            "required_before_ranking": (
                "guarded planner connection plus controls; ambiguous action-only "
                "entries remain blocked"
            ),
            "allowed_connection": "diagnostic_to_posterior_only",
            "action_generation_allowed": False,
        },
    }


def summarize_invisible_effect_position_memory_guarded_planner_connection(
    packets_or_posterior: Iterable[Mapping[str, Any]] | Mapping[str, Any] | Any,
    *,
    repeated_noop_threshold: int = 2,
    delayed_effect_window: int = 3,
    sample_limit: int = 16,
) -> dict[str, Any]:
    """Shadow guard for future invisible-effect position-memory planner use."""

    posterior = _posterior_summary_from_input(
        packets_or_posterior,
        repeated_noop_threshold=repeated_noop_threshold,
        delayed_effect_window=delayed_effect_window,
    )
    entries = [
        dict(entry)
        for entry in list(posterior.get("entries") or [])
        if isinstance(entry, Mapping)
    ]
    counts: Counter[str] = Counter()
    by_game: dict[str, Counter[str]] = {}
    samples: list[dict[str, Any]] = []

    for entry in entries:
        game_id = str(entry.get("game_id", "") or "unknown").strip().lower()
        game_counter = by_game.setdefault(game_id, Counter())
        state = str(entry.get("posterior_state", "") or "")
        admitted = bool(entry.get("admissible_for_guarded_connection"))
        if admitted:
            verdict = "admitted_shadow_only"
            counts["planner_connection_admitted_count"] += 1
            counts["admitted_site_resolved_count"] += 1
            game_counter["planner_connection_admitted_count"] += 1
            game_counter["admitted_site_resolved_count"] += 1
        else:
            verdict = "blocked"
            counts["planner_connection_blocked_count"] += 1
            game_counter["planner_connection_blocked_count"] += 1
            if state == "ambiguous_action_only":
                counts["blocked_ambiguous_action_only_count"] += 1
                game_counter["blocked_ambiguous_action_only_count"] += 1
            elif state == "site_resolved_single_noop_observation":
                counts["blocked_single_noop_insufficient_evidence_count"] += 1
                game_counter["blocked_single_noop_insufficient_evidence_count"] += 1
            else:
                counts["blocked_other_count"] += 1
                game_counter["blocked_other_count"] += 1
        game_counter["posterior_entry_count"] += 1
        if len(samples) < sample_limit:
            samples.append(
                {
                    "position_key": str(entry.get("position_key", "") or ""),
                    "game_id": game_id,
                    "site": entry.get("site"),
                    "posterior_state": state,
                    "verdict": verdict,
                }
            )

    counts["posterior_entry_count"] = len(entries)
    if counts["planner_connection_admitted_count"] > 0:
        promotion_state = (
            "invisible_effect_position_memory_guarded_planner_connection_admitted_shadow_only"
        )
    elif counts["planner_connection_blocked_count"] > 0:
        promotion_state = (
            "invisible_effect_position_memory_guarded_planner_connection_blocked_insufficient_evidence"
        )
    else:
        promotion_state = (
            "invisible_effect_position_memory_guarded_planner_connection_no_entries"
        )

    return {
        "schema_version": (
            INVISIBLE_EFFECT_POSITION_MEMORY_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION
        ),
        "diagnostic_only": True,
        "shadow_only": True,
        "packet_count": _safe_int(posterior.get("packet_count")),
        "posterior_entry_count": counts["posterior_entry_count"],
        "planner_connection_admitted_count": counts[
            "planner_connection_admitted_count"
        ],
        "planner_connection_blocked_count": counts[
            "planner_connection_blocked_count"
        ],
        "admitted_site_resolved_count": counts["admitted_site_resolved_count"],
        "blocked_ambiguous_action_only_count": counts[
            "blocked_ambiguous_action_only_count"
        ],
        "blocked_single_noop_insufficient_evidence_count": counts[
            "blocked_single_noop_insufficient_evidence_count"
        ],
        "blocked_other_count": counts["blocked_other_count"],
        "posterior_promotion_state": str(posterior.get("promotion_state", "") or ""),
        "promotion_state": promotion_state,
        "by_game": {
            game: dict(sorted(counter.items()))
            for game, counter in sorted(by_game.items())
        },
        "samples": samples,
        "planner_connection_enabled": False,
        "runtime_execution_enabled": False,
        "ranking_enabled": False,
        "action_generation_allowed": False,
        "next_runtime_contract": {
            "required_before_execution": "explicit opt-in after all25 controls",
            "allowed_connection": "shadow_guard_only",
            "action_generation_allowed": False,
        },
    }


def _diagnostic_summary_from_input(
    packets_or_summary: Iterable[Mapping[str, Any]] | Mapping[str, Any] | Any,
    *,
    repeated_noop_threshold: int,
    delayed_effect_window: int,
) -> dict[str, Any]:
    if (
        isinstance(packets_or_summary, Mapping)
        and str(packets_or_summary.get("schema_version", "") or "")
        == INVISIBLE_EFFECT_POSITION_MEMORY_DIAGNOSTIC_SCHEMA_VERSION
    ):
        return dict(packets_or_summary)
    return summarize_invisible_effect_position_memory_packets(
        packets_or_summary,
        repeated_noop_threshold=repeated_noop_threshold,
        delayed_effect_window=delayed_effect_window,
    )


def _posterior_summary_from_input(
    packets_or_summary: Iterable[Mapping[str, Any]] | Mapping[str, Any] | Any,
    *,
    repeated_noop_threshold: int,
    delayed_effect_window: int,
) -> dict[str, Any]:
    if (
        isinstance(packets_or_summary, Mapping)
        and str(packets_or_summary.get("schema_version", "") or "")
        == INVISIBLE_EFFECT_POSITION_MEMORY_POSTERIOR_SCHEMA_VERSION
    ):
        return dict(packets_or_summary)
    return summarize_invisible_effect_position_memory_posterior(
        packets_or_summary,
        repeated_noop_threshold=repeated_noop_threshold,
        delayed_effect_window=delayed_effect_window,
    )


def _packet_sort_key(packet: Mapping[str, Any]) -> tuple[str, int]:
    return (_game_id(packet), _safe_int(packet.get("step")))


def _game_id(packet: Mapping[str, Any]) -> str:
    return str(packet.get("game_id", "") or "unknown").strip().lower()


def _site_from_packet(packet: Mapping[str, Any]) -> tuple[int, int] | None:
    payload = packet.get("action_payload")
    if not isinstance(payload, Mapping):
        payload = {}
    for key in (
        "selected_site",
        "site",
        "target_site",
        "click_site",
        "attempted_site",
        "blocked_site",
        "position_memory_anchor",
        "actor_pos",
        "cell",
        "position",
    ):
        site = _site_from_value(payload.get(key))
        if site is not None:
            return site
    for key in (
        "selected_site",
        "site",
        "target_site",
        "click_site",
        "attempted_site",
        "blocked_site",
        "position_memory_anchor",
        "actor_pos",
        "cell",
    ):
        site = _site_from_value(packet.get(key))
        if site is not None:
            return site
    return None


def _site_from_value(value: Any) -> tuple[int, int] | None:
    if isinstance(value, Mapping):
        x = value.get("x", value.get("col", value.get("column")))
        y = value.get("y", value.get("row"))
        if x is None or y is None:
            return None
        return (_safe_int(x), _safe_int(y))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (_safe_int(value[0]), _safe_int(value[1]))
    return None


def _position_key(packet: Mapping[str, Any], *, site: tuple[int, int] | None) -> str:
    game = _game_id(packet)
    action = str(packet.get("action_name", "") or "")
    if site is not None:
        return f"{game}|site={site[0]},{site[1]}|action={action}"
    state_hash = str(packet.get("state_hash", "") or "")
    return f"{game}|state={state_hash}|action={action}"


def _same_site_or_unknown(
    current_site: tuple[int, int] | None,
    prior_site: Any,
) -> bool:
    if current_site is None or prior_site is None:
        return True
    return current_site == prior_site


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "INVISIBLE_EFFECT_POSITION_MEMORY_DIAGNOSTIC_SCHEMA_VERSION",
    "INVISIBLE_EFFECT_POSITION_MEMORY_GUARDED_PLANNER_CONNECTION_SCHEMA_VERSION",
    "INVISIBLE_EFFECT_POSITION_MEMORY_POSTERIOR_SCHEMA_VERSION",
    "summarize_invisible_effect_position_memory_guarded_planner_connection",
    "summarize_invisible_effect_position_memory_packets",
    "summarize_invisible_effect_position_memory_posterior",
]
