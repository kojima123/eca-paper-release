"""Action-effect conditioned click affordance induction.

This module only learns from observed click transitions. It does not sample the
whole frame looking for possible click sites; that broader fallback remains in
``ClickSiteDiscovery``. The goal here is to preserve relations that were already
shown to have an effect and suppress regions that repeatedly no-op.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ClickAffordanceCandidate:
    site: tuple[int, int]
    source: str
    confidence: float
    reason: str
    relation_kind: str
    attempts: int
    successes: int
    failures: int
    follow_action_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": "click_affordance",
            "site": [int(self.site[0]), int(self.site[1])],
            "action_name": "ACTION6",
            "operator": "click",
            "source": str(self.source),
            "confidence": round(max(0.0, min(1.0, float(self.confidence))), 6),
            "reason": str(self.reason),
            "relation_kind": str(self.relation_kind),
            "attempts": int(self.attempts),
            "successes": int(self.successes),
            "failures": int(self.failures),
            "actionable": True,
            "tested": self.attempts > 0,
        }
        if self.follow_action_name:
            payload["follow_action_name"] = str(self.follow_action_name)
        return payload


class ClickAffordanceInducer:
    """Learn click candidates from previous click effects."""

    def __init__(self, *, max_candidates: int = 8, max_observations: int = 96) -> None:
        self.max_candidates = max(1, int(max_candidates or 1))
        self._observations = deque(maxlen=max(1, int(max_observations or 1)))
        self._active_click_contexts = deque(maxlen=8)
        self._site_attempts: Counter[tuple[int, int]] = Counter()
        self._site_successes: Counter[tuple[int, int]] = Counter()
        self._site_failures: Counter[tuple[int, int]] = Counter()
        self._relation_attempts: Counter[tuple[str, tuple[int, int]]] = Counter()
        self._relation_successes: Counter[tuple[str, tuple[int, int]]] = Counter()
        self._relation_failures: Counter[tuple[str, tuple[int, int]]] = Counter()
        self._verified_attempts: Counter[tuple[str, tuple[int, int]]] = Counter()
        self._verified_successes: Counter[tuple[str, tuple[int, int]]] = Counter()
        self._verified_failures: Counter[tuple[str, tuple[int, int]]] = Counter()
        self._chain_attempts: Counter[tuple[tuple[int, int], str]] = Counter()
        self._chain_successes: Counter[tuple[tuple[int, int], str]] = Counter()
        self._chain_failures: Counter[tuple[tuple[int, int], str]] = Counter()
        self._dead_sites: Counter[tuple[int, int]] = Counter()
        self._effect_bboxes = deque(maxlen=32)

    def reset(self) -> None:
        self._observations.clear()
        self._active_click_contexts.clear()
        self._site_attempts.clear()
        self._site_successes.clear()
        self._site_failures.clear()
        self._relation_attempts.clear()
        self._relation_successes.clear()
        self._relation_failures.clear()
        self._verified_attempts.clear()
        self._verified_successes.clear()
        self._verified_failures.clear()
        self._chain_attempts.clear()
        self._chain_successes.clear()
        self._chain_failures.clear()
        self._dead_sites.clear()
        self._effect_bboxes.clear()

    def observe(
        self,
        record: Any,
        pre_frame: Any,
        post_frame: Any,
        *,
        pending_target: Mapping[str, Any] | None = None,
    ) -> None:
        action_name = str(getattr(record, "action_name", "") or "")
        if action_name != "ACTION6":
            self._observe_follow_action(record)
            return
        site = _payload_site(getattr(record, "action_payload", {}) or {})
        if site is None:
            return
        arr = np.asarray(pre_frame)
        height, width = _frame_size(arr, getattr(record, "frame_shape", None))
        if not _inside(site, width=width, height=height):
            return
        relation_kind = _relation_kind(record, site)
        changed = int(getattr(record, "changed_pixel_count", 0) or 0) > 0
        score_delta = int(getattr(record, "score_delta", 0) or 0)
        no_op = bool(getattr(record, "no_op", False))
        progressed = bool(score_delta > 0 or (changed and not no_op))
        relation_key = (relation_kind, site)
        self._site_attempts[site] += 1
        self._relation_attempts[relation_key] += 1
        if progressed:
            self._site_successes[site] += 1
            self._relation_successes[relation_key] += 1
        else:
            self._site_failures[site] += 1
            self._relation_failures[relation_key] += 1
            self._dead_sites[site] += 1
        bbox = _bbox_dict(getattr(record, "changed_bbox", None))
        if bbox and progressed:
            self._effect_bboxes.append({"site": site, "bbox": bbox, "relation_kind": relation_kind})
        self._active_click_contexts.append(
            {
                "site": site,
                "relation_kind": relation_kind,
                "ttl": 4,
                "click_progressed": bool(progressed),
            }
        )
        self._observations.append(
            {
                "site": [int(site[0]), int(site[1])],
                "relation_kind": relation_kind,
                "changed": bool(changed),
                "score_delta": int(score_delta),
                "no_op": bool(no_op),
                "progressed": bool(progressed),
                "changed_bbox": bbox,
                "pending_kind": _pending_kind(pending_target),
            }
        )

    def _observe_follow_action(self, record: Any) -> None:
        if not self._active_click_contexts:
            return
        action_name = str(getattr(record, "action_name", "") or "")
        if not action_name:
            return
        score_delta = int(getattr(record, "score_delta", 0) or 0)
        progressed = score_delta > 0
        next_contexts = deque(maxlen=8)
        for context in list(self._active_click_contexts):
            site = context.get("site")
            if not isinstance(site, tuple) or len(site) < 2:
                continue
            chain_key = (site, action_name)
            self._chain_attempts[chain_key] += 1
            if progressed:
                self._chain_successes[chain_key] += 1
            else:
                self._chain_failures[chain_key] += 1
            ttl = int(context.get("ttl", 0) or 0) - 1
            if ttl > 0 and not progressed:
                new_context = dict(context)
                new_context["ttl"] = ttl
                next_contexts.append(new_context)
        self._active_click_contexts = next_contexts

    def observe_verification(
        self,
        candidate: Mapping[str, Any],
        *,
        progressed: bool,
    ) -> None:
        if not isinstance(candidate, Mapping):
            return
        if str(candidate.get("kind", "") or "") != "click_affordance":
            return
        site = candidate.get("site")
        if not isinstance(site, (list, tuple)) or len(site) < 2:
            return
        try:
            click_site = (int(site[0]), int(site[1]))
        except (TypeError, ValueError):
            return
        relation_kind = str(candidate.get("relation_kind", "") or "")
        if not relation_kind:
            source = str(candidate.get("discovery_source", "") or "")
            relation_kind = source.removeprefix("induced_") if source.startswith("induced_") else "unknown"
        relation_key = (relation_kind, click_site)
        self._verified_attempts[relation_key] += 1
        if progressed:
            self._verified_successes[relation_key] += 1
        else:
            self._verified_failures[relation_key] += 1

    def propose(
        self,
        frame: Any,
        *,
        failed_sites: Iterable[tuple[int, int]] = (),
        tested_sites: Iterable[tuple[int, int]] = (),
    ) -> list[dict[str, Any]]:
        arr = np.asarray(frame)
        height, width = _frame_size(arr, None)
        failed = {tuple(map(int, site)) for site in failed_sites}
        tested = {tuple(map(int, site)) for site in tested_sites}
        scored: list[tuple[float, int, ClickAffordanceCandidate]] = []
        seen: set[tuple[int, int]] = set()
        for index, (chain_key, successes) in enumerate(self._chain_successes.most_common()):
            site, follow_action = chain_key
            if site in seen or not _inside(site, width=width, height=height):
                continue
            attempts = int(self._chain_attempts[chain_key])
            failures = int(self._chain_failures[chain_key])
            if attempts <= 0 or successes <= 0:
                continue
            confidence = _chain_confidence(
                attempts=attempts,
                successes=int(successes),
                failures=failures,
                near_failed=_near_any(site, failed, radius=2),
                tested=site in tested,
            )
            if confidence <= 0.05:
                continue
            seen.add(site)
            scored.append(
                (
                    -confidence,
                    index,
                    ClickAffordanceCandidate(
                        site=site,
                        source="induced_selector_effect",
                        confidence=confidence,
                        reason="click site enabled a later score-bearing action",
                        relation_kind="selector_effect",
                        attempts=attempts,
                        successes=int(successes),
                        failures=failures,
                        follow_action_name=follow_action,
                    ),
                )
            )
        for index, (relation_key, successes) in enumerate(self._relation_successes.most_common()):
            relation_kind, site = relation_key
            if site in seen or not _inside(site, width=width, height=height):
                continue
            seen.add(site)
            attempts = int(self._relation_attempts[relation_key])
            failures = int(self._relation_failures[relation_key])
            if attempts <= 0 or successes <= 0:
                continue
            verified_attempts = int(self._verified_attempts[relation_key])
            verified_successes = int(self._verified_successes[relation_key])
            verified_failures = int(self._verified_failures[relation_key])
            class_verified_attempts = sum(
                int(count)
                for (kind, _site), count in self._verified_attempts.items()
                if kind == relation_kind
            )
            class_verified_successes = sum(
                int(count)
                for (kind, _site), count in self._verified_successes.items()
                if kind == relation_kind
            )
            class_verified_failures = sum(
                int(count)
                for (kind, _site), count in self._verified_failures.items()
                if kind == relation_kind
            )
            confidence = _relation_confidence(
                relation_kind=relation_kind,
                attempts=attempts,
                successes=int(successes),
                failures=failures,
                verified_attempts=verified_attempts,
                verified_successes=verified_successes,
                verified_failures=verified_failures,
                class_verified_attempts=class_verified_attempts,
                class_verified_successes=class_verified_successes,
                class_verified_failures=class_verified_failures,
                near_failed=_near_any(site, failed, radius=2),
                tested=site in tested,
                dead_count=int(self._dead_sites[site]),
            )
            if confidence <= 0.05:
                continue
            scored.append(
                (
                    -confidence,
                    index,
                    ClickAffordanceCandidate(
                        site=site,
                        source=f"induced_{relation_kind}",
                        confidence=confidence,
                        reason="click site previously caused visible action-effect progress",
                        relation_kind=relation_kind,
                        attempts=attempts,
                        successes=int(successes),
                        failures=failures,
                    ),
                )
            )
        scored.sort(key=lambda item: (item[0], item[1]))
        return [candidate.to_dict() for _score, _index, candidate in scored[: self.max_candidates]]

    def summary(self, *, enabled: bool = False) -> dict[str, Any]:
        relation_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"attempts": 0, "successes": 0, "failures": 0})
        verified_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"attempts": 0, "successes": 0, "failures": 0})
        chain_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"attempts": 0, "successes": 0, "failures": 0})
        for (relation_kind, _site), attempts in self._relation_attempts.items():
            relation_stats[relation_kind]["attempts"] += int(attempts)
        for (relation_kind, _site), successes in self._relation_successes.items():
            relation_stats[relation_kind]["successes"] += int(successes)
        for (relation_kind, _site), failures in self._relation_failures.items():
            relation_stats[relation_kind]["failures"] += int(failures)
        for (relation_kind, _site), attempts in self._verified_attempts.items():
            verified_stats[relation_kind]["attempts"] += int(attempts)
        for (relation_kind, _site), successes in self._verified_successes.items():
            verified_stats[relation_kind]["successes"] += int(successes)
        for (relation_kind, _site), failures in self._verified_failures.items():
            verified_stats[relation_kind]["failures"] += int(failures)
        for (_site, action_name), attempts in self._chain_attempts.items():
            chain_stats[action_name]["attempts"] += int(attempts)
        for (_site, action_name), successes in self._chain_successes.items():
            chain_stats[action_name]["successes"] += int(successes)
        for (_site, action_name), failures in self._chain_failures.items():
            chain_stats[action_name]["failures"] += int(failures)
        return {
            "enabled": bool(enabled),
            "observation_count": len(self._observations),
            "attempt_count": int(sum(self._site_attempts.values())),
            "success_count": int(sum(self._site_successes.values())),
            "failure_count": int(sum(self._site_failures.values())),
            "dead_sites": [list(site) for site, _count in self._dead_sites.most_common(12)],
            "effective_sites": [list(site) for site, _count in self._site_successes.most_common(12)],
            "relation_stats": {key: dict(value) for key, value in sorted(relation_stats.items())},
            "verified_relation_stats": {key: dict(value) for key, value in sorted(verified_stats.items())},
            "chain_stats": {key: dict(value) for key, value in sorted(chain_stats.items())},
            "recent": list(self._observations)[-8:],
        }


def _payload_site(payload: Mapping[str, Any]) -> tuple[int, int] | None:
    try:
        return (int(payload.get("x")), int(payload.get("y")))
    except (TypeError, ValueError):
        return None


def _frame_size(frame: np.ndarray, shape: Any) -> tuple[int, int]:
    if frame.ndim >= 2:
        return int(frame.shape[0]), int(frame.shape[1])
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    return 0, 0


def _bbox_dict(bbox: Any) -> dict[str, int] | None:
    if bbox is None:
        return None
    if isinstance(bbox, Mapping):
        raw = bbox
    elif hasattr(bbox, "to_dict"):
        raw = bbox.to_dict()
    else:
        raw = {
            "min_y": getattr(bbox, "min_y", None),
            "min_x": getattr(bbox, "min_x", None),
            "max_y": getattr(bbox, "max_y", None),
            "max_x": getattr(bbox, "max_x", None),
        }
    try:
        return {
            "min_y": int(raw["min_y"]),
            "min_x": int(raw["min_x"]),
            "max_y": int(raw["max_y"]),
            "max_x": int(raw["max_x"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _relation_kind(record: Any, site: tuple[int, int]) -> str:
    score_delta = int(getattr(record, "score_delta", 0) or 0)
    if score_delta > 0:
        return "score_click"
    if bool(getattr(record, "no_op", False)) or int(getattr(record, "changed_pixel_count", 0) or 0) <= 0:
        return "dead_click_zone"
    bbox = _bbox_dict(getattr(record, "changed_bbox", None))
    if not bbox:
        return "broad_click_effect"
    center = (
        int(round((bbox["min_x"] + bbox["max_x"]) / 2.0)),
        int(round((bbox["min_y"] + bbox["max_y"]) / 2.0)),
    )
    width = int(bbox["max_x"] - bbox["min_x"] + 1)
    height = int(bbox["max_y"] - bbox["min_y"] + 1)
    near_radius = max(2, int(round((width + height) / 2.0)))
    return "direct_click_effect" if _manhattan(site, center) <= near_radius else "remote_toggle_effect"


def _relation_confidence(
    *,
    relation_kind: str,
    attempts: int,
    successes: int,
    failures: int,
    verified_attempts: int,
    verified_successes: int,
    verified_failures: int,
    class_verified_attempts: int,
    class_verified_successes: int,
    class_verified_failures: int,
    near_failed: bool,
    tested: bool,
    dead_count: int,
) -> float:
    base_by_kind = {
        "score_click": 0.88,
        "direct_click_effect": 0.70,
        "remote_toggle_effect": 0.62,
        "broad_click_effect": 0.50,
    }
    base = float(base_by_kind.get(relation_kind, 0.0))
    if base <= 0.0:
        return 0.0
    if verified_attempts >= 3 and verified_successes <= 0:
        return 0.0
    if (
        relation_kind != "score_click"
        and class_verified_attempts >= 4
        and class_verified_successes <= 0
        and class_verified_failures >= 4
    ):
        return 0.0
    reliability = float(successes) / float(max(1, attempts))
    confidence = base + min(0.18, 0.08 * successes) + 0.22 * reliability
    confidence -= min(0.35, 0.10 * failures)
    confidence -= min(0.30, 0.12 * dead_count)
    if verified_successes > 0:
        confidence += min(0.22, 0.12 * verified_successes)
    if verified_failures > 0:
        confidence -= min(0.55, 0.18 * verified_failures)
    if near_failed:
        confidence -= 0.20
    if tested and successes <= failures:
        confidence -= 0.16
    return max(0.0, min(1.0, confidence))


def _chain_confidence(
    *,
    attempts: int,
    successes: int,
    failures: int,
    near_failed: bool,
    tested: bool,
) -> float:
    if attempts <= 0 or successes <= 0:
        return 0.0
    reliability = float(successes) / float(max(1, attempts))
    confidence = 0.82 + min(0.20, 0.10 * successes) + 0.18 * reliability
    confidence -= min(0.35, 0.08 * failures)
    if near_failed:
        confidence -= 0.12
    if tested and successes <= failures:
        confidence -= 0.10
    return max(0.0, min(1.0, confidence))


def _inside(site: tuple[int, int], *, width: int, height: int) -> bool:
    return 0 <= int(site[0]) < int(width) and 0 <= int(site[1]) < int(height)


def _near_any(site: tuple[int, int], sites: set[tuple[int, int]], *, radius: int) -> bool:
    return any(_manhattan(site, other) <= radius for other in sites)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _pending_kind(pending_target: Mapping[str, Any] | None) -> str:
    if not isinstance(pending_target, Mapping):
        return ""
    candidate = pending_target.get("candidate")
    if isinstance(candidate, Mapping):
        return str(candidate.get("kind", "") or "")
    return ""
