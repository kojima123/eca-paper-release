"""Grid helpers used by the ECA runtime: a state hash, 4-connected components, an ASCII ramp."""
from __future__ import annotations

import hashlib
from collections import deque
from typing import Dict, List, Sequence, Tuple

# The common 70-character ASCII density ramp (dark to light), with "{}" written as "EG";
# kept character for character because rendered boards are parsed back with it.
_ASCII_PALETTE = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1EG[]?-_+~<>i!lI;:,\"^`'. "


def hash_grid_state(grid: List[List[int]]) -> str:
    """12-hex-digit MD5 of the grid's Python repr (stable across runs)."""
    return hashlib.md5(repr(grid).encode()).hexdigest()[:12]


def find_connected_components(grid: Sequence[Sequence[int]]) -> Dict[Tuple[int, int], int]:
    """Label 4-connected same-value regions; ids are given in row-major order of each region's first cell."""
    labels: Dict[Tuple[int, int], int] = {}
    if not grid:
        return labels
    h, w = len(grid), len(grid[0])
    next_id = 0
    for r0 in range(h):
        for c0 in range(w):
            if (r0, c0) in labels:
                continue
            value = grid[r0][c0]
            labels[(r0, c0)] = next_id
            todo = deque([(r0, c0)])
            while todo:
                r, c = todo.popleft()
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in labels and grid[rr][cc] == value:
                        labels[(rr, cc)] = next_id
                        todo.append((rr, cc))
            next_id += 1
    return labels
