"""Frame-only click site discovery for ECA primitive synthesis.

This module does not execute clicks. It proposes candidate sites that can be
verified by the action-effect ledger. The goal is to turn "click somewhere"
into concrete, auditable hypotheses without using game ids or environment
internals.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ClickSiteCandidate:
    site: tuple[int, int]
    source: str
    confidence: float
    reason: str
    value: Any = None
    area: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": "discovered_click_site",
            "site": [int(self.site[0]), int(self.site[1])],
            "action_name": "ACTION6",
            "operator": "click",
            "source": self.source,
            "confidence": round(max(0.0, min(1.0, float(self.confidence))), 6),
            "reason": self.reason,
            "actionable": False,
            "tested": False,
        }
        if self.value is not None:
            payload["value"] = _json_safe_value(self.value)
        if self.area > 0:
            payload["area"] = int(self.area)
        return payload


class ClickSiteDiscovery:
    """Generate candidate click sites from visible frame structure."""

    def __init__(self, *, max_candidates: int = 24) -> None:
        self.max_candidates = max(1, int(max_candidates or 1))

    def discover(
        self,
        frame: Any,
        *,
        failed_sites: Iterable[tuple[int, int]] = (),
        tested_sites: Iterable[tuple[int, int]] = (),
        changed_bboxes: Iterable[Mapping[str, Any]] = (),
        source_stats: Mapping[str, Mapping[str, int]] | None = None,
    ) -> list[dict[str, Any]]:
        arr = np.asarray(frame)
        if arr.ndim < 2 or arr.size <= 0:
            return []
        height = int(arr.shape[0])
        width = int(arr.shape[1])
        labels = _label_grid(arr)
        failed = {tuple(map(int, site)) for site in failed_sites}
        tested = {tuple(map(int, site)) for site in tested_sites}

        candidates: list[ClickSiteCandidate] = []
        candidates.extend(
            self._changed_bbox_neighbors(
                changed_bboxes,
                width=width,
                height=height,
            )
        )
        candidates.extend(self._object_centers(labels, width=width, height=height))
        candidates.extend(self._rare_color_centers(labels, width=width, height=height))
        candidates.extend(self._symmetry_sites(tested | failed, width=width, height=height))
        candidates.extend(self._grid_samples(width=width, height=height))

        scored: list[tuple[float, int, ClickSiteCandidate]] = []
        seen: set[tuple[int, int]] = set()
        stats = dict(source_stats or {})
        for index, candidate in enumerate(candidates):
            site = candidate.site
            if site in seen:
                continue
            seen.add(site)
            if not _inside(site, width=width, height=height):
                continue
            if _near_any(site, failed, radius=2):
                penalty = 0.45
            elif _near_any(site, tested, radius=1):
                penalty = 0.25
            else:
                penalty = 0.0
            border_penalty = 0.20 if _near_border(site, width=width, height=height) else 0.0
            source_adjustment = _source_adjustment(candidate.source, stats)
            score = max(0.0, float(candidate.confidence) + source_adjustment - penalty - border_penalty)
            if score <= 0.02:
                continue
            scored.append((-score, index, candidate))

        scored.sort(key=lambda item: (item[0], item[1]))
        return [candidate.to_dict() for _score, _index, candidate in scored[: self.max_candidates]]

    def _changed_bbox_neighbors(
        self,
        bboxes: Iterable[Mapping[str, Any]],
        *,
        width: int,
        height: int,
    ) -> list[ClickSiteCandidate]:
        out: list[ClickSiteCandidate] = []
        for bbox in list(bboxes)[-8:]:
            if not isinstance(bbox, Mapping):
                continue
            try:
                min_x = int(bbox.get("min_x"))
                max_x = int(bbox.get("max_x"))
                min_y = int(bbox.get("min_y"))
                max_y = int(bbox.get("max_y"))
            except (TypeError, ValueError):
                continue
            cx = max(0, min(width - 1, int(round((min_x + max_x) / 2.0))))
            cy = max(0, min(height - 1, int(round((min_y + max_y) / 2.0))))
            for dx, dy in ((0, 0), (3, 0), (-3, 0), (0, 3), (0, -3)):
                site = (max(0, min(width - 1, cx + dx)), max(0, min(height - 1, cy + dy)))
                out.append(
                    ClickSiteCandidate(
                        site=site,
                        source="changed_bbox_neighbor",
                        confidence=0.70 if (dx, dy) == (0, 0) else 0.52,
                        reason="near previously localized changed bbox",
                    )
                )
        return out

    def _object_centers(self, labels: np.ndarray, *, width: int, height: int) -> list[ClickSiteCandidate]:
        background = _background_label(labels)
        min_area = 2
        max_area = max(8, int(width * height * 0.12))
        out: list[ClickSiteCandidate] = []
        for value in _rare_labels(labels, background=background, limit=12):
            mask = labels == value
            for component in _components(mask, limit=8):
                area = len(component)
                if area < min_area or area > max_area:
                    continue
                xs = [point[0] for point in component]
                ys = [point[1] for point in component]
                bbox_width = max(xs) - min(xs) + 1
                bbox_height = max(ys) - min(ys) + 1
                if (bbox_width <= 2 or bbox_height <= 2) and _touches_border(xs, ys, width=width, height=height):
                    continue
                site = (int(round(sum(xs) / area)), int(round(sum(ys) / area)))
                out.append(
                    ClickSiteCandidate(
                        site=site,
                        source="object_center",
                        confidence=0.62,
                        reason="center of compact non-background component",
                        value=value,
                        area=area,
                    )
                )
        return out

    def _rare_color_centers(self, labels: np.ndarray, *, width: int, height: int) -> list[ClickSiteCandidate]:
        background = _background_label(labels)
        out: list[ClickSiteCandidate] = []
        for value in _rare_labels(labels, background=background, limit=8):
            ys, xs = np.nonzero(labels == value)
            if xs.size <= 0:
                continue
            area = int(xs.size)
            if area > max(4, int(width * height * 0.18)):
                continue
            site = (int(round(float(xs.mean()))), int(round(float(ys.mean()))))
            out.append(
                ClickSiteCandidate(
                    site=site,
                    source="rare_color_center",
                    confidence=0.46,
                    reason="center of rare visible value",
                    value=value,
                    area=area,
                )
            )
        return out

    def _symmetry_sites(
        self,
        sites: set[tuple[int, int]],
        *,
        width: int,
        height: int,
    ) -> list[ClickSiteCandidate]:
        out: list[ClickSiteCandidate] = []
        for x, y in list(sites)[-12:]:
            for site, reason in (
                ((width - 1 - x, y), "horizontal symmetry of tested click"),
                ((x, height - 1 - y), "vertical symmetry of tested click"),
            ):
                if _inside(site, width=width, height=height) and site != (x, y):
                    out.append(
                        ClickSiteCandidate(
                            site=site,
                            source="symmetry",
                            confidence=0.40,
                            reason=reason,
                        )
                    )
        return out

    def _grid_samples(self, *, width: int, height: int) -> list[ClickSiteCandidate]:
        xs = _sample_axis(width)
        ys = _sample_axis(height)
        out: list[ClickSiteCandidate] = []
        for y in ys:
            for x in xs:
                out.append(
                    ClickSiteCandidate(
                        site=(x, y),
                        source="grid_sample",
                        confidence=0.18,
                        reason="coarse unverified click sample",
                    )
                )
        return out


def _label_grid(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.int64, copy=False)
    flat = arr.reshape((-1, int(np.prod(arr.shape[2:]))))
    labels = np.zeros((flat.shape[0],), dtype=np.int64)
    mapping: dict[tuple[Any, ...], int] = {}
    next_label = 0
    for index, row in enumerate(flat.tolist()):
        key = tuple(row)
        if key not in mapping:
            mapping[key] = next_label
            next_label += 1
        labels[index] = mapping[key]
    return labels.reshape(arr.shape[:2])


def _background_label(labels: np.ndarray) -> int:
    counts = Counter(int(value) for value in labels.flatten().tolist())
    if not counts:
        return 0
    return int(counts.most_common(1)[0][0])


def _rare_labels(labels: np.ndarray, *, background: int, limit: int) -> list[int]:
    counts = Counter(int(value) for value in labels.flatten().tolist())
    items = [(count, value) for value, count in counts.items() if int(value) != int(background)]
    items.sort(key=lambda item: (item[0], item[1]))
    return [int(value) for _count, value in items[: max(0, int(limit))]]


def _components(mask: np.ndarray, *, limit: int) -> list[list[tuple[int, int]]]:
    height, width = mask.shape[:2]
    visited = np.zeros((height, width), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not bool(mask[y, x]):
                continue
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y, x] = True
            component: list[tuple[int, int]] = []
            while queue:
                cx, cy = queue.popleft()
                component.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if visited[ny, nx] or not bool(mask[ny, nx]):
                        continue
                    visited[ny, nx] = True
                    queue.append((nx, ny))
            out.append(component)
            if len(out) >= limit:
                return out
    return out


def _sample_axis(size: int) -> list[int]:
    if size <= 1:
        return [0]
    points = {size // 2}
    for fraction in (0.25, 0.5, 0.75):
        points.add(max(0, min(size - 1, int(round((size - 1) * fraction)))))
    return sorted(points)


def _inside(site: tuple[int, int], *, width: int, height: int) -> bool:
    x, y = site
    return 0 <= x < width and 0 <= y < height


def _near_border(site: tuple[int, int], *, width: int, height: int) -> bool:
    x, y = site
    return x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1


def _touches_border(xs: list[int], ys: list[int], *, width: int, height: int) -> bool:
    return min(xs) <= 0 or min(ys) <= 0 or max(xs) >= width - 1 or max(ys) >= height - 1


def _near_any(site: tuple[int, int], others: set[tuple[int, int]], *, radius: int) -> bool:
    x, y = site
    for ox, oy in others:
        if abs(x - ox) + abs(y - oy) <= radius:
            return True
    return False


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _source_adjustment(source: str, source_stats: Mapping[str, Mapping[str, int]]) -> float:
    stats = source_stats.get(str(source), {}) if isinstance(source_stats, Mapping) else {}
    if not isinstance(stats, Mapping):
        return 0.0
    attempts = int(stats.get("attempts", 0) or 0)
    successes = int(stats.get("successes", 0) or 0)
    failures = int(stats.get("failures", 0) or 0)
    if attempts <= 0:
        return 0.0
    success_rate = float(successes) / float(max(1, attempts))
    failure_rate = float(failures) / float(max(1, attempts))
    if successes > 0:
        return min(0.25, 0.12 + success_rate * 0.18)
    if attempts >= 3 and failure_rate >= 0.75:
        return -0.35
    if attempts >= 2 and failure_rate >= 0.50:
        return -0.18
    return 0.0


__all__ = ["ClickSiteCandidate", "ClickSiteDiscovery"]
