"""ConfidenceCache for the closed-loop hypothesis selector.

Stores per-(state_cluster, hypothesis_id) confidence in [0.0, 1.0].
Updated by observe_transition based on outcome. Read by
HypothesisSelector to bias action selection.

This is Layer C of the closed-loop architecture described in
docs/closed_loop_architecture_design_2026-05-27.md.

Default-off; the loop must explicitly call update_score_event /
update_no_score / update_noop to populate the cache.
"""
from __future__ import annotations

from typing import Any


class ConfidenceCache:
    """Per-key confidence in [0.0, 1.0] with exponential update.

    Update rules:
      score_event (delta > 0): conf += alpha * (1 - conf)
      no_score (changed > 0, delta == 0): conf -= decay * conf
      noop (changed == 0, delta == 0): conf -= beta * conf
    """

    DEFAULT_PRIOR: float = 0.5

    def __init__(
        self,
        *,
        alpha: float = 0.2,
        beta: float = 0.05,
        decay: float = 0.01,
        prior: float = DEFAULT_PRIOR,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.decay = float(decay)
        self.prior = float(prior)
        self._cache: dict[tuple[str, str], float] = {}
        self._update_event_count: int = 0
        self._update_no_score_count: int = 0
        self._update_noop_count: int = 0

    def get(
        self, state_key: str, hyp_id: str, default: float | None = None
    ) -> float:
        return self._cache.get((str(state_key), str(hyp_id)), default if default is not None else self.prior)

    def update_score_event(self, state_key: str, hyp_id: str) -> None:
        k = (str(state_key), str(hyp_id))
        c = self._cache.get(k, self.prior)
        self._cache[k] = min(1.0, c + self.alpha * (1.0 - c))
        self._update_event_count += 1

    def update_no_score(self, state_key: str, hyp_id: str) -> None:
        k = (str(state_key), str(hyp_id))
        c = self._cache.get(k, self.prior)
        self._cache[k] = max(0.0, c - self.decay * c)
        self._update_no_score_count += 1

    def update_noop(self, state_key: str, hyp_id: str) -> None:
        k = (str(state_key), str(hyp_id))
        c = self._cache.get(k, self.prior)
        self._cache[k] = max(0.0, c - self.beta * c)
        self._update_noop_count += 1

    def size(self) -> int:
        return len(self._cache)

    def summary(self) -> dict[str, Any]:
        if not self._cache:
            return {
                "size": 0, "max": 0.0, "min": 0.0, "mean": 0.0,
                "updates_score": self._update_event_count,
                "updates_no_score": self._update_no_score_count,
                "updates_noop": self._update_noop_count,
            }
        vals = list(self._cache.values())
        return {
            "size": len(vals),
            "max": float(max(vals)),
            "min": float(min(vals)),
            "mean": float(sum(vals) / len(vals)),
            "updates_score": self._update_event_count,
            "updates_no_score": self._update_no_score_count,
            "updates_noop": self._update_noop_count,
        }

    def clear(self) -> None:
        self._cache.clear()
        self._update_event_count = 0
        self._update_no_score_count = 0
        self._update_noop_count = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize cache + counters for persistence."""
        return {
            "version": 1,
            "alpha": self.alpha,
            "beta": self.beta,
            "decay": self.decay,
            "prior": self.prior,
            "entries": [
                {"state": s, "hyp": h, "conf": float(v)}
                for (s, h), v in self._cache.items()
            ],
            "updates_score": self._update_event_count,
            "updates_no_score": self._update_no_score_count,
            "updates_noop": self._update_noop_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfidenceCache":
        """Reconstruct cache from to_dict output. Merges any 'entries'.

        Hyperparameters fall back to constructor defaults if missing.
        """
        c = cls(
            alpha=float(data.get("alpha", 0.2)),
            beta=float(data.get("beta", 0.05)),
            decay=float(data.get("decay", 0.01)),
            prior=float(data.get("prior", cls.DEFAULT_PRIOR)),
        )
        for e in data.get("entries", []) or []:
            try:
                s = str(e.get("state", ""))
                h = str(e.get("hyp", ""))
                v = float(e.get("conf", c.prior))
                if s and h:
                    c._cache[(s, h)] = max(0.0, min(1.0, v))
            except (TypeError, ValueError):
                continue
        c._update_event_count = int(data.get("updates_score", 0) or 0)
        c._update_no_score_count = int(data.get("updates_no_score", 0) or 0)
        c._update_noop_count = int(data.get("updates_noop", 0) or 0)
        return c

    def save_to(self, path: Any) -> None:
        """Persist cache to a JSON file (best-effort, no exception
        propagation on disk errors)."""
        import json
        import os as _os
        try:
            _os.makedirs(_os.path.dirname(str(path)) or ".", exist_ok=True)
            with open(str(path), "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh)
        except Exception:
            pass

    @classmethod
    def load_from(cls, path: Any) -> "ConfidenceCache | None":
        """Load a previously-saved cache. Returns None on any failure
        (file missing, JSON malformed, schema mismatch)."""
        import json
        try:
            with open(str(path), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return None
            return cls.from_dict(data)
        except Exception:
            return None

    def merge_from(self, other: "ConfidenceCache") -> None:
        """Merge entries from another cache. On key collision the
        average is used. Counters accumulate."""
        for (s, h), v in other._cache.items():
            existing = self._cache.get((s, h))
            if existing is None:
                self._cache[(s, h)] = v
            else:
                self._cache[(s, h)] = max(0.0, min(1.0, 0.5 * (existing + v)))
        self._update_event_count += other._update_event_count
        self._update_no_score_count += other._update_no_score_count
        self._update_noop_count += other._update_noop_count


__all__ = ["ConfidenceCache"]
