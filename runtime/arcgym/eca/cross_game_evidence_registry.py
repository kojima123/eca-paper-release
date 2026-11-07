"""Cross-game emergent evidence registry.

Persistent JSON-backed store of rod signatures that have been
score-correlated across multiple games. Used by Method A's
emergent affordance pipeline to boost confidence for signatures seen
in 2+ games (= structurally-meaningful mechanic candidates).

See docs/cross_game_emergent_evidence_design_2026-05-22.md.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "cross_game_emergent_evidence_registry.v1"


class CrossGameRodSignatureRegistry:
    """Persistent registry of score-correlated rod signatures.

    Each entry maps a rod_pattern_mining signature to:
      - the set of game_ids where it was score-correlated
      - total observed score events
      - first/last seen timestamps

    boost_factor_for_signature() returns a confidence multiplier:
      n=0 (not in registry): 1.0
      n=1 (one game):        1.0 (no boost — single-game match is weak)
      n>=2 (multi-game):     1.0 + 0.15 * (n - 1), capped at 2.0
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path: Path | None = Path(path) if path else None
        # In-memory store: signature -> entry dict
        self._entries: dict[str, dict[str, Any]] = {}
        if self._path is not None and self._path.is_file():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        for entry in data.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            sig = str(entry.get("signature") or "")
            if not sig:
                continue
            games = entry.get("games_with_score_correlation") or []
            if not isinstance(games, list):
                continue
            normalized_games = sorted(
                {str(g) for g in games if str(g or "")}
            )
            self._entries[sig] = {
                "signature": sig,
                "games_with_score_correlation": normalized_games,
                "total_score_events_observed": int(
                    entry.get("total_score_events_observed", 0) or 0
                ),
                "first_seen_at": str(entry.get("first_seen_at") or ""),
                "last_updated_at": str(entry.get("last_updated_at") or ""),
            }

    def add_score_correlated_signature(
        self, signature: str, game_id: str
    ) -> None:
        """Record that `signature` was near a score event in `game_id`.

        Idempotent: re-adding the same (signature, game_id) increments
        the event counter and updates last_updated_at without
        duplicating game_id in the set.
        """
        sig = str(signature or "").strip()
        gid = str(game_id or "").strip()
        if not sig or not gid:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        entry = self._entries.get(sig)
        if entry is None:
            self._entries[sig] = {
                "signature": sig,
                "games_with_score_correlation": [gid],
                "total_score_events_observed": 1,
                "first_seen_at": now_iso,
                "last_updated_at": now_iso,
            }
            return
        games = set(entry.get("games_with_score_correlation") or [])
        games.add(gid)
        entry["games_with_score_correlation"] = sorted(games)
        entry["total_score_events_observed"] = (
            int(entry.get("total_score_events_observed", 0)) + 1
        )
        entry["last_updated_at"] = now_iso

    def signature_score_games(self, signature: str) -> set[str]:
        """Return set of game_ids where this signature was
        score-correlated. Empty set if signature not in registry."""
        sig = str(signature or "").strip()
        entry = self._entries.get(sig)
        if entry is None:
            return set()
        return set(entry.get("games_with_score_correlation") or [])

    def boost_factor_for_signature(self, signature: str) -> float:
        """Confidence multiplier per signature.

        Formula:
          n=0 (not present):      1.0  (= no change)
          n=1 (single game):      1.0  (single-game evidence is too weak)
          n>=2 (multi-game):      1.0 + 0.15 * (n - 1)
                                  capped at 2.0
        """
        games = self.signature_score_games(signature)
        n = len(games)
        if n < 2:
            return 1.0
        boost = 1.0 + 0.15 * (n - 1)
        return min(2.0, boost)

    def to_payload(self) -> dict[str, Any]:
        """Serialize to JSON-ready dict."""
        return {
            "schema_version": SCHEMA_VERSION,
            "entries": list(self._entries.values()),
        }

    def save(self, path: Path | None = None) -> None:
        target = Path(path) if path is not None else self._path
        if target is None:
            raise ValueError(
                "save() requires a path; constructor path was not set"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, signature: str) -> bool:
        return str(signature or "") in self._entries
