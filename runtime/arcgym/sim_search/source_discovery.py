"""Runtime discovery of an ARC-AGI-3 game's .py source file.

The Kaggle rerun environment (and our local checkouts) ship the per-game
source files. We look for ``<gid>.py`` where ``gid`` is the short game id
(``game_id.split('-')[0]``), first via explicit hints, then via glob
patterns over the usual roots. The game class is identified by the
``class X(ARCBaseGame)`` signature inside the file.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

_CLASS_RE = re.compile(r"class\s+(\w+)\s*\(\s*ARCBaseGame")

# Glob patterns tried in order, formatted with the short game id.
_DEFAULT_PATTERNS: tuple[str, ...] = (
    "/kaggle/input/**/{gid}.py",
    "/tmp/**/{gid}.py",
    "environment_files/**/{gid}.py",
    "**/environment_files/**/{gid}.py",
    "**/game_sources/**/{gid}.py",
)


def base_game_id(game_id: str) -> str:
    return str(game_id or "").split("-", 1)[0].strip().lower()


def _class_name_from_source(path: str) -> str | None:
    """Find the ``class X(ARCBaseGame)`` name. Game files often open with a
    large sprites dict, so the whole file is scanned (not just the head)."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
    except OSError:
        return None
    match = _CLASS_RE.search(content)
    if match:
        return match.group(1)
    if "ARCBaseGame" in content:
        # Imported but declared in an unexpected form; fall back to the
        # conventional ``Gid`` class name derived from the file name.
        gid = Path(path).stem
        return gid[:1].upper() + gid[1:]
    return None


def _candidate_roots() -> list[str]:
    roots: list[str] = []
    env_roots = os.environ.get("SIM_SEARCH_SOURCE_ROOTS", "")
    if env_roots:
        roots.extend(r for r in env_roots.split(os.pathsep) if r.strip())
    return roots


def find_game_source(
    game_id: str,
    arc_env: object | None = None,
    extra_roots: list[str] | None = None,
) -> tuple[str | None, str | None]:
    """Locate the game source file for ``game_id``.

    Returns ``(path, class_name)``; ``(None, None)`` when no source is
    discoverable — the caller then falls back to the frame-only stack.
    """
    gid = base_game_id(game_id)
    if not gid:
        return None, None

    candidates: list[str] = []

    # 1. arc_env hint (the toolkit exposes environment_info.local_dir when
    #    the game runs from a local checkout).
    local_dir = getattr(getattr(arc_env, "environment_info", None), "local_dir", None)
    if local_dir:
        candidates.append(str(Path(str(local_dir)) / f"{gid}.py"))

    # 2. explicit roots (env var + caller-provided), searched recursively.
    roots = _candidate_roots()
    if extra_roots:
        roots.extend(extra_roots)
    for root in roots:
        candidates.extend(
            sorted(glob.glob(str(Path(root) / "**" / f"{gid}.py"), recursive=True))
        )

    # 3. default glob patterns (Kaggle input, /tmp, repo-relative).
    for pattern in _DEFAULT_PATTERNS:
        try:
            candidates.extend(sorted(glob.glob(pattern.format(gid=gid), recursive=True)))
        except (OSError, ValueError):
            continue

    seen: set[str] = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if not os.path.isfile(path):
            continue
        cls_name = _class_name_from_source(path)
        if cls_name:
            return path, cls_name
    return None, None


__all__ = ["base_game_id", "find_game_source"]
