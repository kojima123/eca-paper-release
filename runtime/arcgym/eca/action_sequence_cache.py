"""ActionSequenceCache: (state, action_history) -> next_action_outcome.

Phase I of the closed-loop architecture. The probe layer (Phase 1-G)
learned per-(state, click_position) outcomes, which only engages on
~5/25 games. This layer goes one abstraction up: it observes which
actions tend to produce score events after specific recent histories,
across all action types (ACTION1-6, not just click).

Phase I-1: observation-only. The cache records every transition's
(pre_state_key, last_K_actions, action_taken, score_delta) but does
NOT yet influence selection. Used to (a) verify the cache populates
on realistic sweeps, (b) measure cross-state and cross-game key
collisions.

Phase I-2: add a query method that returns an action bias score
distribution per (state, history). Then optionally weight upstream
proposals.

Implementation choices:
- history depth K is small (default 2) to keep key space tractable.
- Entries are bounded to avoid runaway memory; oldest LRU evicted.
- Score is a sum (delta) and count, so callers can compute average
  delta per (state, history, action).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any


class ActionSequenceCache:
    """Per-(state_key, recent action history) -> action outcome stats."""

    def __init__(
        self,
        *,
        history_depth: int = 2,
        max_entries: int = 4096,
    ) -> None:
        self._history_depth = max(0, int(history_depth))
        self._max_entries = max(16, int(max_entries))
        # key: (state_key, tuple_of_last_K_action_names)
        # value: dict[next_action_name] -> (count, total_score_delta)
        self._table: OrderedDict[
            tuple[str, tuple[str, ...]], dict[str, tuple[int, int]]
        ] = OrderedDict()
        self._observe_count: int = 0
        self._positive_observe_count: int = 0

    def observe(
        self,
        *,
        state_key: str,
        action_history: tuple[str, ...] | list[str],
        action_taken: str,
        score_delta: int,
    ) -> None:
        """Record a single (state, history) -> (action, delta) sample.

        Truncates history to self._history_depth. Updates LRU order
        and evicts oldest if cap exceeded.
        """
        sk = str(state_key)
        hist = tuple(action_history)[-self._history_depth:] if self._history_depth > 0 else ()
        key = (sk, hist)
        bucket = self._table.get(key)
        if bucket is None:
            bucket = {}
            self._table[key] = bucket
        else:
            # LRU touch
            self._table.move_to_end(key)
        prev = bucket.get(str(action_taken), (0, 0))
        bucket[str(action_taken)] = (
            int(prev[0]) + 1,
            int(prev[1]) + int(score_delta),
        )
        self._observe_count += 1
        if int(score_delta) > 0:
            self._positive_observe_count += 1
        # Evict oldest if over cap
        while len(self._table) > self._max_entries:
            self._table.popitem(last=False)

    def query(
        self,
        *,
        state_key: str,
        action_history: tuple[str, ...] | list[str],
    ) -> dict[str, float]:
        """Return action -> average score delta for this (state, history).

        Returns empty dict if no entries match. Average is total_score_delta
        / count per action. Negative average is possible (= action led to
        no_score / noop on average).
        """
        sk = str(state_key)
        hist = tuple(action_history)[-self._history_depth:] if self._history_depth > 0 else ()
        bucket = self._table.get((sk, hist))
        if not bucket:
            return {}
        return {
            action: (float(total) / max(1, int(count)))
            for action, (count, total) in bucket.items()
        }

    def query_with_counts(
        self,
        *,
        state_key: str,
        action_history: tuple[str, ...] | list[str],
    ) -> dict[str, tuple[int, int]]:
        """Return action -> (count, total_delta) for this (state, history).

        Phase I-2 needs both count and total to filter weak evidence.
        Returns empty dict if no entries match.
        """
        sk = str(state_key)
        hist = tuple(action_history)[-self._history_depth:] if self._history_depth > 0 else ()
        bucket = self._table.get((sk, hist))
        if not bucket:
            return {}
        return {action: tuple(stats) for action, stats in bucket.items()}

    def size(self) -> int:
        return len(self._table)

    def summary(self) -> dict[str, Any]:
        """Aggregate summary across the cache. Useful for audits."""
        total_actions = 0
        positive_buckets = 0
        bucket_size_distribution: list[int] = []
        for bucket in self._table.values():
            total_actions += sum(c for (c, _) in bucket.values())
            if any(t > 0 for (_, t) in bucket.values()):
                positive_buckets += 1
            bucket_size_distribution.append(len(bucket))
        avg_bucket_size = (
            sum(bucket_size_distribution) / max(1, len(bucket_size_distribution))
        )
        return {
            "entries": len(self._table),
            "observe_count": self._observe_count,
            "positive_observe_count": self._positive_observe_count,
            "total_actions_recorded": total_actions,
            "positive_buckets": positive_buckets,
            "avg_actions_per_bucket": round(avg_bucket_size, 2),
            "history_depth": self._history_depth,
            "max_entries": self._max_entries,
        }

    def clear(self) -> None:
        self._table.clear()
        self._observe_count = 0
        self._positive_observe_count = 0


__all__ = ["ActionSequenceCache"]
