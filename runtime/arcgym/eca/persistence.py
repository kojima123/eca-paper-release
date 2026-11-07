"""Persistence helpers for ECA snapshots and bundles."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import NightBundle


def append_jsonl(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def load_bundle(path: Path | None, *, game_id: str | None = None) -> NightBundle:
    if path is None or not path.exists():
        return NightBundle(game_id=game_id)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return NightBundle.from_mapping(raw, game_id=game_id)


def save_bundle(path: Path | None, bundle: NightBundle) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
