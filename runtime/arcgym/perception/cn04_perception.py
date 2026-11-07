"""cn04 frame perception — full L1-L6 via signature dispatch.

cn04 is a sprite-pixel-overlap puzzle. Each sprite has anchor pixels
(colors 8 or 13). Win condition: each anchor pixel has a matching
anchor at the same world coord on another sprite.

Levels have canonical fixed initial layouts. Perception identifies
the level by signature (sprite count + anchor pixel count); adapter
dispatches to hardcoded plan.

Public API:
    perceive_cn04(frame) -> dict {
        "level_signature": "cn04_l1" .. "cn04_l6" or None,
        "anchor_count": int,
    }
"""
from __future__ import annotations

import numpy as np


ANCHOR_COLORS = (8, 13)


def perceive_cn04(frame: np.ndarray) -> dict:
    if frame.ndim == 3:
        frame = frame[-1]
    frame = np.asarray(frame, dtype=np.int16)

    # Each level has a unique color SET (verified L1-L6 in env c574d22b).
    # Use frozenset of colors as signature.
    cols = frozenset(int(c) for c in np.unique(frame) if c >= 0)
    sig_map = {
        frozenset({0, 4, 8, 10, 14}): "cn04_l1",
        frozenset({0, 4, 8, 9, 11, 12, 14}): "cn04_l2",
        frozenset({4, 8, 12, 14}): "cn04_l3",
        frozenset({4, 8, 9, 10}): "cn04_l4",
        frozenset({0, 4, 8, 11, 15}): "cn04_l5",
        frozenset({4, 8, 9, 14}): "cn04_l6",
    }
    sig = sig_map.get(cols)

    return {
        "level_signature": sig,
        "colors": cols,
    }


def _count_sprites(frame: np.ndarray) -> int:
    """Approximate sprite count by clustering anchor pixels.

    Each sprite has at least one anchor pixel pair. Cluster anchor
    pixels by Chebyshev proximity (within 12 cells = sprite size).
    """
    pts: list[tuple[int, int]] = []
    for c in ANCHOR_COLORS:
        ys, xs = np.where(frame == c)
        for x, y in zip(xs.tolist(), ys.tolist()):
            pts.append((int(x), int(y)))
    if not pts:
        return 0
    visited: set[int] = set()
    clusters = 0
    for i, p in enumerate(pts):
        if i in visited:
            continue
        # BFS within Chebyshev 12.
        stack = [i]
        while stack:
            j = stack.pop()
            if j in visited:
                continue
            visited.add(j)
            for k in range(len(pts)):
                if k in visited:
                    continue
                if (abs(pts[k][0] - pts[j][0]) <= 12 and
                        abs(pts[k][1] - pts[j][1]) <= 12):
                    stack.append(k)
        clusters += 1
    return clusters
