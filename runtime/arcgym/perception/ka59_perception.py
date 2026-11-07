"""ka59 perception — L1-L5 via signature dispatch."""
from __future__ import annotations
import numpy as np

from arcgym.perception.object_segments import segment_dicts


def classify_ka59_segments(segments: list[dict[str, object]]) -> dict:
    """Classify raw color components into coarse ka59 object-role candidates."""

    selectable: list[dict[str, object]] = []
    target_regions: list[dict[str, object]] = []
    barrier_regions: list[dict[str, object]] = []
    room_regions: list[dict[str, object]] = []
    field_regions: list[dict[str, object]] = []

    for segment in segments:
        color = int(segment.get("color", -1) or -1)
        size = int(segment.get("size", 0) or 0)
        bbox = list(segment.get("bbox", []) or [])
        centroid = list(segment.get("centroid", []) or [])
        if len(centroid) >= 2:
            site = [int(round(float(centroid[0]))), int(round(float(centroid[1])))]
        else:
            site = []
        item = {**segment, "site": site}
        if color == 14 and 4 <= size <= 32:
            selectable.append({**item, "role": "selectable_or_cursor"})
        elif color == 4 and 8 <= size <= 40 and not (len(bbox) >= 4 and int(bbox[3]) >= 64):
            target_regions.append({**item, "role": "target_region"})
        elif color == 15:
            barrier_regions.append({**item, "role": "barrier_or_push_wall"})
        elif color == 1 and size >= 100:
            room_regions.append({**item, "role": "room_or_goal_container"})
        elif color == 2 and size >= 100:
            field_regions.append({**item, "role": "field_or_background"})

    return {
        "selectable_candidates": selectable,
        "target_regions": target_regions,
        "barrier_regions": barrier_regions,
        "room_regions": room_regions,
        "field_regions": field_regions,
        "candidate_summary": {
            "selectable_count": len(selectable),
            "target_region_count": len(target_regions),
            "barrier_region_count": len(barrier_regions),
            "room_region_count": len(room_regions),
        },
    }


def perceive_ka59(frame: np.ndarray) -> dict:
    if frame.ndim == 3:
        frame = frame[-1]
    frame = np.asarray(frame, dtype=np.int16)
    cols = frozenset(int(c) for c in np.unique(frame) if c >= 0)
    c1 = int((frame == 1).sum())
    sigs = {
        (frozenset({0, 1, 2, 4, 5, 14, 15}), 607): "ka59_l1",
        (frozenset({0, 1, 2, 4, 5, 14, 15}), 2117): "ka59_l2",
        (frozenset({0, 1, 2, 4, 11, 14}), 2721): "ka59_l3",
        (frozenset({0, 1, 2, 4, 5, 11, 14, 15}), 1786): "ka59_l4",
        (frozenset({0, 1, 2, 4, 12, 13, 14, 15}), 3488): "ka59_l5",
    }
    segments = segment_dicts(frame, ignore_colors={0}, min_size=4)
    roles = classify_ka59_segments(segments)
    return {
        "level_signature": sigs.get((cols, c1)),
        "object_segments": segments,
        **roles,
    }


__all__ = ["classify_ka59_segments", "perceive_ka59"]
