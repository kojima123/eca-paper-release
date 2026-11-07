"""MechanicClassifier: classify ARC games into mechanic families.

Phase J (hierarchical mechanic categorization, 2026-05-29).

The probe layer (Phase 1-I) plateaus at 40/25 because:
- Probe only engages on click puzzles where upstream lacks structural
  preference (5/25 games)
- For move-family games (wa30, m0r0 etc.), probe never engages
- For rule-table games (tr87, lp85), upstream already has good priors

This module introduces a higher abstraction: classify games into
mechanic families based on initial-frame features + action vocabulary.
Downstream consumers (Phase J-2+) can use the family to bias primitive
selection, gate optional layers, or load family-specific knowledge.

Phase J-1 (this commit): classifier + observation-only integration.
No behavioral changes. Used to audit classification accuracy across
the 25-game suite before committing to a routing strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


MechanicFamily = Literal[
    "click_only",          # vocab: ACTION6 only (no structural sig)
    "click_targets",       # click + many small objects (lp85, r11l, su15, tn36)
    "move_only",           # vocab: no ACTION6
    "sprite_rotation",     # large dominant component, few comps (lf52, m0r0, tr87, vc33)
    "small_object_grid",   # many tiny components (re86)
    "mixed",               # vocab: both, no specific sub-pattern
    "unknown",
]


# Phase J-2 (2026-05-29) subfamily detection thresholds.
# Derived from the 25-game audit: see
# docs/phase_j1_mechanic_classifier_2026-05-29.md.
_CLICK_TARGETS_MIN_COMPONENTS = 15
_SPRITE_ROTATION_MIN_LARGEST = 1500
_SPRITE_ROTATION_MAX_COMPONENTS = 5
_SMALL_OBJECT_GRID_MIN_COMPONENTS = 10
_SMALL_OBJECT_GRID_MAX_LARGEST = 300


@dataclass(frozen=True)
class MechanicClassification:
    """Frozen classification record."""

    family: MechanicFamily
    confidence: float
    features: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


def _normalize_action(name: Any) -> str:
    """Normalize action name to ACTIONN form (mirrors agent helper)."""
    text = str(getattr(name, "name", name) or "").strip().upper()
    if text.startswith("GAMEACTION."):
        text = text.split(".", 1)[1]
    if text.isdigit():
        return f"ACTION{int(text)}"
    return text


def _extract_frame_features(frame: Any) -> dict[str, Any]:
    """Extract coarse features from a single 2D frame.

    Returns:
      - frame_height
      - frame_width
      - n_distinct_colors (excluding background)
      - dominant_color
      - bg_fraction (= largest color's share of pixels)
      - non_bg_pixel_count
      - n_components (4-connectivity, non-bg)
      - largest_component_size
      - smallest_component_size
    """
    try:
        arr = np.asarray(frame)
    except Exception:
        return {"error": "cannot_convert_to_array"}
    if arr.ndim < 2:
        return {"error": "scalar_or_1d"}
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h <= 0 or w <= 0:
        return {"error": "empty_dims"}
    flat = arr.ravel().astype(int)
    if flat.size == 0:
        return {"error": "empty_flat"}
    bins = [0] * 16
    for v in flat:
        if 0 <= v < 16:
            bins[v] += 1
    total = max(1, int(sum(bins)))
    bg = max(range(16), key=lambda i: bins[i])
    bg_fraction = bins[bg] / total
    non_bg_pixels = total - bins[bg]
    distinct_colors = sum(1 for c in bins if c > 0)
    distinct_non_bg_colors = sum(
        1 for color, count in enumerate(bins) if color != bg and count > 0
    )
    # Connected components (4-connectivity) on non-bg mask
    mask = (arr != bg).astype(np.int32)
    visited = np.zeros_like(mask, dtype=bool)
    component_sizes: list[int] = []
    stack: list[tuple[int, int]] = []
    for sr in range(h):
        for sc in range(w):
            if mask[sr, sc] == 0 or visited[sr, sc]:
                continue
            stack.append((sr, sc))
            visited[sr, sc] = True
            size = 0
            while stack:
                r, c = stack.pop()
                size += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and mask[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            component_sizes.append(size)
    largest = max(component_sizes) if component_sizes else 0
    smallest = min(component_sizes) if component_sizes else 0
    return {
        "frame_height": h,
        "frame_width": w,
        "n_distinct_colors": distinct_colors,
        "n_distinct_non_bg_colors": distinct_non_bg_colors,
        "dominant_color": bg,
        "bg_fraction": round(float(bg_fraction), 4),
        "non_bg_pixel_count": int(non_bg_pixels),
        "n_components": len(component_sizes),
        "largest_component_size": largest,
        "smallest_component_size": smallest,
    }


def _classify_from_actions(
    available_actions: list[str],
) -> tuple[MechanicFamily, float, str]:
    """Classify family from action vocabulary alone."""
    if not available_actions:
        return "unknown", 0.0, "no_available_actions"
    normalized = [_normalize_action(a) for a in available_actions]
    has_click = "ACTION6" in normalized
    move_actions = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}
    has_move = any(a in move_actions for a in normalized)
    if has_click and not has_move:
        return "click_only", 0.85, "vocabulary_click_only"
    if has_move and not has_click:
        return "move_only", 0.85, "vocabulary_move_only"
    if has_click and has_move:
        return "mixed", 0.5, "vocabulary_both_available"
    return "unknown", 0.0, "no_known_action"


def _refine_with_features(
    base_family: MechanicFamily,
    base_confidence: float,
    base_reason: str,
    features: dict[str, Any],
) -> tuple[MechanicFamily, float, str]:
    """Phase J-2: refine the vocabulary-based family using structural
    features. Returns (family, confidence, reason).

    Subfamily rules derived from the 25-game audit (Phase J-1):

    - click_targets: many components (>= 15)
    - sprite_rotation: single dominant large component
    - small_object_grid: many tiny components

    Only refines when feature thresholds are met. Otherwise keeps the
    vocabulary-based family.
    """
    if "error" in features:
        return base_family, base_confidence, base_reason
    n_components = int(features.get("n_components", 0) or 0)
    largest = int(features.get("largest_component_size", 0) or 0)

    # Subfamily detection. Vocabulary gates the subfamily so that
    # click-family games go to click_targets and move-family go to
    # small_object_grid even when structural numbers overlap.

    # 1. sprite_rotation: single dominant large component regardless of
    #    vocab. This pattern is strong enough to override vocab.
    if (
        largest >= _SPRITE_ROTATION_MIN_LARGEST
        and n_components <= _SPRITE_ROTATION_MAX_COMPONENTS
    ):
        return (
            "sprite_rotation",
            0.7,
            f"structural_sprite_rotation(largest={largest}, n={n_components})",
        )

    # 2. click_targets: click-family vocab + many components.
    if base_family in {"click_only", "mixed"} and (
        n_components >= _CLICK_TARGETS_MIN_COMPONENTS
    ):
        return (
            "click_targets",
            0.75,
            f"structural_click_targets(n={n_components})",
        )

    # 3. small_object_grid: move-family vocab + many tiny components.
    if base_family in {"move_only", "mixed"} and (
        n_components >= _SMALL_OBJECT_GRID_MIN_COMPONENTS
        and largest <= _SMALL_OBJECT_GRID_MAX_LARGEST
    ):
        return (
            "small_object_grid",
            0.7,
            f"structural_small_object_grid(n={n_components}, max={largest})",
        )

    # No subfamily match
    return base_family, base_confidence, base_reason


def classify(
    *,
    initial_frame: Any,
    available_actions: list[str] | None = None,
) -> MechanicClassification:
    """Public entry point: classify a game from one initial frame +
    action vocabulary.

    Phase J-2: combines vocabulary heuristic with structural feature
    refinement to detect subfamilies. Falls back to the vocabulary
    classification when no subfamily threshold is met.
    """
    avail = list(available_actions or [])
    base_family, base_conf, base_reason = _classify_from_actions(avail)
    features = _extract_frame_features(initial_frame)
    family, confidence, reason = _refine_with_features(
        base_family=base_family,
        base_confidence=base_conf,
        base_reason=base_reason,
        features=features,
    )
    return MechanicClassification(
        family=family,
        confidence=confidence,
        features=features,
        reason=reason,
    )


__all__ = [
    "MechanicClassifier",
    "MechanicClassification",
    "MechanicFamily",
    "classify",
]


class MechanicClassifier:
    """Object-style wrapper around the classify() function."""

    def __init__(self) -> None:
        # Reserved for future per-instance state (e.g. memoization).
        pass

    def classify(
        self,
        *,
        initial_frame: Any,
        available_actions: list[str] | None = None,
    ) -> MechanicClassification:
        return classify(
            initial_frame=initial_frame,
            available_actions=available_actions,
        )
