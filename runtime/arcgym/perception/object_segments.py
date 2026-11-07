"""Frame-only connected component extraction for object-like regions."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ObjectSegment:
    color: int
    size: int
    bbox: BBox
    centroid: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "color": int(self.color),
            "size": int(self.size),
            "bbox": [int(value) for value in self.bbox],
            "centroid": [round(float(self.centroid[0]), 3), round(float(self.centroid[1]), 3)],
        }


@dataclass(frozen=True)
class SegmentDisplacement:
    color: int
    pre_size: int
    post_size: int
    pre_bbox: BBox
    post_bbox: BBox
    pre_centroid: tuple[float, float]
    post_centroid: tuple[float, float]
    dx: float
    dy: float
    distance: float
    size_delta: int
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "color": int(self.color),
            "pre_size": int(self.pre_size),
            "post_size": int(self.post_size),
            "pre_bbox": [int(value) for value in self.pre_bbox],
            "post_bbox": [int(value) for value in self.post_bbox],
            "pre_centroid": [
                round(float(self.pre_centroid[0]), 3),
                round(float(self.pre_centroid[1]), 3),
            ],
            "post_centroid": [
                round(float(self.post_centroid[0]), 3),
                round(float(self.post_centroid[1]), 3),
            ],
            "dx": round(float(self.dx), 3),
            "dy": round(float(self.dy), 3),
            "distance": round(float(self.distance), 3),
            "size_delta": int(self.size_delta),
            "confidence": round(float(self.confidence), 3),
        }


def connected_color_segments(
    frame: np.ndarray,
    *,
    ignore_colors: Iterable[int] = (0,),
    min_size: int = 1,
) -> tuple[ObjectSegment, ...]:
    """Return 4-connected same-color components from a 2D rendered frame."""

    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D frame, got shape={arr.shape!r}")
    height, width = int(arr.shape[0]), int(arr.shape[1])
    ignored = {int(color) for color in ignore_colors}
    visited = np.zeros((height, width), dtype=bool)
    segments: list[ObjectSegment] = []

    for y in range(height):
        for x in range(width):
            if visited[y, x]:
                continue
            color = int(arr[y, x])
            if color in ignored:
                visited[y, x] = True
                continue
            cells = _flood_same_color(arr, visited, x, y, color)
            if len(cells) < int(min_size):
                continue
            segments.append(_segment_from_cells(color, cells))

    segments.sort(key=lambda item: (item.color, item.bbox[1], item.bbox[0], item.size))
    return tuple(segments)


def segment_dicts(
    frame: np.ndarray,
    *,
    ignore_colors: Iterable[int] = (0,),
    min_size: int = 1,
) -> list[dict[str, object]]:
    return [
        segment.to_dict()
        for segment in connected_color_segments(
            frame,
            ignore_colors=ignore_colors,
            min_size=min_size,
        )
    ]


def match_segment_displacements(
    pre_segments: Iterable[ObjectSegment | dict[str, object]],
    post_segments: Iterable[ObjectSegment | dict[str, object]],
    *,
    colors: Iterable[int] | None = None,
    max_distance: float = 16.0,
    max_size_delta_fraction: float = 0.75,
) -> tuple[SegmentDisplacement, ...]:
    """Greedily match same-color segments across two frames and report motion.

    The matcher is intentionally generic and conservative: it does not know game
    ids or roles, and it only pairs components with compatible color, centroid
    distance, and size. Higher-level adapters can then interpret the displacement
    as action-effect evidence.
    """

    color_filter = None if colors is None else {int(color) for color in colors}
    pre = [
        segment
        for segment in (_coerce_segment(item) for item in pre_segments)
        if segment is not None and (color_filter is None or int(segment.color) in color_filter)
    ]
    post = [
        segment
        for segment in (_coerce_segment(item) for item in post_segments)
        if segment is not None and (color_filter is None or int(segment.color) in color_filter)
    ]
    max_distance = max(0.0, float(max_distance))
    max_size_delta_fraction = max(0.0, float(max_size_delta_fraction))
    pre_ranks = _color_order_ranks(pre)
    post_ranks = _color_order_ranks(post)

    candidates: list[tuple[float, int, int, SegmentDisplacement]] = []
    for pre_index, pre_segment in enumerate(pre):
        for post_index, post_segment in enumerate(post):
            if int(pre_segment.color) != int(post_segment.color):
                continue
            dx = float(post_segment.centroid[0]) - float(pre_segment.centroid[0])
            dy = float(post_segment.centroid[1]) - float(pre_segment.centroid[1])
            distance = float(hypot(dx, dy))
            if distance > max_distance:
                continue
            size_delta = abs(int(post_segment.size) - int(pre_segment.size))
            max_size_delta = max(1.0, float(pre_segment.size) * max_size_delta_fraction)
            if float(size_delta) > max_size_delta:
                continue
            rank_delta = abs(pre_ranks.get(pre_index, 0) - post_ranks.get(post_index, 0))
            distance_score = 1.0 if max_distance == 0.0 else max(0.0, 1.0 - distance / max_distance)
            size_score = max(0.0, 1.0 - float(size_delta) / max_size_delta)
            confidence = round((distance_score + size_score) / 2.0, 6)
            candidates.append(
                (
                    distance
                    + float(size_delta) / max(1.0, float(pre_segment.size))
                    + float(rank_delta) * max(1.0, max_distance * 2.0),
                    pre_index,
                    post_index,
                    SegmentDisplacement(
                        color=int(pre_segment.color),
                        pre_size=int(pre_segment.size),
                        post_size=int(post_segment.size),
                        pre_bbox=pre_segment.bbox,
                        post_bbox=post_segment.bbox,
                        pre_centroid=pre_segment.centroid,
                        post_centroid=post_segment.centroid,
                        dx=dx,
                        dy=dy,
                        distance=distance,
                        size_delta=int(size_delta),
                        confidence=float(confidence),
                    ),
                )
            )

    candidates.sort(key=lambda item: item[0])
    used_pre: set[int] = set()
    used_post: set[int] = set()
    matches: list[SegmentDisplacement] = []
    for _score, pre_index, post_index, displacement in candidates:
        if pre_index in used_pre or post_index in used_post:
            continue
        used_pre.add(pre_index)
        used_post.add(post_index)
        matches.append(displacement)
    matches.sort(key=lambda item: (item.color, item.pre_bbox[1], item.pre_bbox[0], item.distance))
    return tuple(matches)


def _flood_same_color(
    arr: np.ndarray,
    visited: np.ndarray,
    x0: int,
    y0: int,
    color: int,
) -> list[Coord]:
    height, width = int(arr.shape[0]), int(arr.shape[1])
    queue: deque[Coord] = deque([(int(x0), int(y0))])
    visited[y0, x0] = True
    cells: list[Coord] = []

    while queue:
        x, y = queue.popleft()
        cells.append((x, y))
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if visited[ny, nx] or int(arr[ny, nx]) != color:
                continue
            visited[ny, nx] = True
            queue.append((nx, ny))
    return cells


def _segment_from_cells(color: int, cells: list[Coord]) -> ObjectSegment:
    xs = [cell[0] for cell in cells]
    ys = [cell[1] for cell in cells]
    size = len(cells)
    return ObjectSegment(
        color=int(color),
        size=int(size),
        bbox=(min(xs), min(ys), max(xs) + 1, max(ys) + 1),
        centroid=(sum(xs) / size, sum(ys) / size),
    )


def _coerce_segment(item: ObjectSegment | dict[str, object]) -> ObjectSegment | None:
    if isinstance(item, ObjectSegment):
        return item
    if not isinstance(item, dict):
        return None
    try:
        bbox = item.get("bbox")
        centroid = item.get("centroid")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
            return None
        return ObjectSegment(
            color=int(item["color"]),
            size=int(item["size"]),
            bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            centroid=(float(centroid[0]), float(centroid[1])),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _color_order_ranks(segments: list[ObjectSegment]) -> dict[int, int]:
    ranks: dict[int, int] = {}
    by_color: dict[int, list[tuple[int, ObjectSegment]]] = {}
    for index, segment in enumerate(segments):
        by_color.setdefault(int(segment.color), []).append((index, segment))
    for items in by_color.values():
        items.sort(
            key=lambda pair: (
                float(pair[1].centroid[1]),
                float(pair[1].centroid[0]),
                pair[1].bbox,
            )
        )
        for rank, (index, _segment) in enumerate(items):
            ranks[index] = rank
    return ranks


__all__ = [
    "ObjectSegment",
    "SegmentDisplacement",
    "connected_color_segments",
    "match_segment_displacements",
    "segment_dicts",
]
