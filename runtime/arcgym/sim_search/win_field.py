"""Win-field extraction from a game's source text.

Many games gate level completion on a scalar attribute checked right
before ``self.next_level()``. Extracting that field name (and whether
the win wants it maximised or minimised) gives the counter-A* cascade
rung a dense priority signal where plain BFS drowns.
"""
from __future__ import annotations

import re
from pathlib import Path

_SELF_FIELD_RE = re.compile(r"self\.(\w+)")


def extract_win_field(source_path: str) -> tuple[str | None, int]:
    """Return ``(field_name, direction)``; direction +1 = maximise,
    -1 = minimise, 0/None = not found.

    Heuristic: find ``self.next_level()`` call sites; scan up to 7 lines
    above each for an ``if``/``elif`` condition mentioning a ``self.X``
    scalar; infer direction from the comparison operator next to the
    field anywhere in the source.
    """
    try:
        text = Path(source_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None, 0
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "self.next_level()" not in line:
            continue
        for back in range(1, 8):
            j = idx - back
            if j < 0:
                break
            stripped = lines[j].strip()
            if not (stripped.startswith("if ") or stripped.startswith("elif ")):
                continue
            match = _SELF_FIELD_RE.search(stripped)
            if not match:
                continue
            field = match.group(1)
            if field in ("next_level", "_state", "_score"):
                continue
            # Method calls (``self.foo(...)``) are not counters.
            if re.search(rf"self\.{field}\s*\(", stripped):
                continue
            direction = 0
            if re.search(rf"self\.{field}\s*(>=|>)", text):
                direction = 1
            elif re.search(rf"self\.{field}\s*(<=|<)", text):
                direction = -1
            return field, (direction or 1)
    return None, 0


__all__ = ["extract_win_field"]
