"""Runtime wrapper for the generic side-scroll fall bootstrap.

The underlying bootstrap performs source-free RESET+replay discovery and
returns a score-confirmed prefix. This wrapper only drains that prefix into the
current C-axis live episode. It is default-off and does not branch on game id.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from arcgym.eca.agent_identifier import _frame_components


ActionSpec = tuple[Any, ...]
LAST_FOLLOWUP_DIAGNOSTICS: dict[str, Any] = {}


@dataclass(frozen=True)
class _SideFallLattice:
    xs: tuple[int, ...]
    upper_y: int
    base_y: int
    stride: int
    destructible_color: int
    hazard_color: int


@dataclass(frozen=True)
class _SideFallToggleBridge:
    player_x: int
    player_y: int
    stride: int
    destructible_color: int
    toggle_color: int
    destructible_y: int


@dataclass(frozen=True)
class _SideFallGravitySwitchLadder:
    wall_x: int
    bridge_x: int
    top_switch_y: int
    upper_switch_y: int
    middle_switch_y: int
    lower_switch_y: int
    stride: int
    wall_color: int
    hazard_color: int
    platform_color: int
    switch_color: int


@dataclass(frozen=True)
class _SideFallTwinSwitchGemChamber:
    switch_x: int
    switch_y: int
    player_x: int
    player_y: int
    stride: int
    switch_color: int
    destructible_color: int
    hazard_color: int


@dataclass(frozen=True)
class _SideFallOffscreenSwitchWalkway:
    player_x: int
    player_y: int
    visible_switch_x: int
    visible_switch_y: int
    stride: int
    switch_color: int
    toggle_color: int
    hazard_color: int


def _frame2d(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=np.int16)
    return arr.astype(np.int16, copy=False)


def _bg(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(counts))]) if len(vals) else 0


def _round_int(value: Any) -> int:
    return int(round(float(value or 0.0)))


def _round_half_up(value: Any) -> int:
    return int(np.floor(float(value or 0.0) + 0.5))


def _row_xs(comps: list[dict[str, Any]], y: int) -> tuple[int, ...]:
    xs = sorted(_round_int(comp.get("cx", 0.0)) for comp in comps if _round_int(comp.get("cy", 0.0)) == y)
    return tuple(xs)


def _regular_stride(xs: tuple[int, ...]) -> int:
    if len(xs) < 3:
        return 0
    diffs = [b - a for a, b in zip(xs, xs[1:])]
    if min(diffs) <= 0:
        return 0
    stride = int(round(float(np.median(diffs))))
    if stride < 4 or stride > 10:
        return 0
    if any(abs(diff - stride) > 1 for diff in diffs):
        return 0
    return stride


def _detect_two_row_side_fall_lattice(frame: Any) -> _SideFallLattice | None:
    """Read a side-fall destructible lattice from the current rendered frame.

    The gate intentionally uses only visible component geometry. It looks for
    two aligned rows of small destructible-like components plus a separate
    repeated hazard row above them; no game id, source state, or adapter plan is
    consulted.
    """
    arr = _frame2d(frame)
    if arr.size == 0:
        return None

    comps = list(_frame_components(arr))
    by_color: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for comp in comps:
        color = int(comp.get("color", -1))
        n = int(comp.get("n", 0) or 0)
        cx = float(comp.get("cx", 0.0) or 0.0)
        cy = float(comp.get("cy", 0.0) or 0.0)
        if 8 <= n <= 36 and 2.0 <= cx <= arr.shape[1] - 3 and 2.0 <= cy <= arr.shape[0] - 3:
            by_color[color].append(comp)

    for color, color_comps in by_color.items():
        rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for comp in color_comps:
            rows[_round_int(comp.get("cy", 0.0))].append(comp)
        if len(rows) < 2:
            continue
        y_values = sorted(rows)
        for upper_y, base_y in zip(y_values, y_values[1:]):
            upper_xs = _row_xs(rows[upper_y], upper_y)
            base_xs = _row_xs(rows[base_y], base_y)
            if len(upper_xs) != len(base_xs) or len(base_xs) != 7:
                continue
            stride = _regular_stride(base_xs)
            if stride == 0:
                continue
            if upper_xs != base_xs:
                continue
            if abs((base_y - upper_y) - stride) > 1:
                continue

            hazard_color = -1
            for other_color, other_comps in by_color.items():
                if other_color == color:
                    continue
                hazard_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for comp in other_comps:
                    if int(comp.get("n", 0) or 0) < 10:
                        continue
                    hy = _round_int(comp.get("cy", 0.0))
                    if hy < upper_y:
                        hazard_rows[hy].append(comp)
                if any(len(items) >= 3 for items in hazard_rows.values()):
                    hazard_color = int(other_color)
                    break
            if hazard_color < 0:
                continue

            return _SideFallLattice(
                xs=base_xs,
                upper_y=int(upper_y),
                base_y=int(base_y),
                stride=int(stride),
                destructible_color=int(color),
                hazard_color=int(hazard_color),
            )
    return None


def _detect_toggle_bridge_lattice(frame: Any) -> _SideFallToggleBridge | None:
    """Read a compact toggle-bridge side-fall layout from visible geometry.

    This is the next side-fall family after the two-row destructible lattice:
    a small player, a row of three same-colour destructibles one stride above
    the player row, and repeated same-colour toggle clusters that can be
    clicked into temporary supports.
    """
    arr = _frame2d(frame)
    if arr.size == 0:
        return None

    bg = int(_bg(arr))
    comps = list(_frame_components(arr))
    player_candidates = [
        comp
        for comp in comps
        if int(comp.get("color", -1)) not in (bg, 0, 3, 10)
        and 4 <= int(comp.get("n", 0) or 0) <= 12
        and 12.0 <= float(comp.get("cy", 0.0) or 0.0) <= arr.shape[0] - 6
    ]
    if not player_candidates:
        return None
    player = max(
        player_candidates,
        key=lambda comp: (
            float(comp.get("cy", 0.0) or 0.0),
            -abs(float(comp.get("cx", 0.0) or 0.0) - (arr.shape[1] / 2.0)),
        ),
    )
    player_x = _round_half_up(player.get("cx", 0.0))
    player_y = _round_half_up(player.get("cy", 0.0))

    by_color: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for comp in comps:
        color = int(comp.get("color", -1))
        n = int(comp.get("n", 0) or 0)
        if color in (bg, 0, 3, 10) or n < 1:
            continue
        by_color[color].append(comp)

    destructible_choice: tuple[int, int, tuple[int, ...], int] | None = None
    for color, color_comps in by_color.items():
        if color == int(player.get("color", -1)):
            continue
        rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for comp in color_comps:
            n = int(comp.get("n", 0) or 0)
            if 15 <= n <= 40:
                rows[_round_half_up(comp.get("cy", 0.0))].append(comp)
        for y, row_comps in rows.items():
            if y >= player_y or len(row_comps) < 3:
                continue
            xs = tuple(sorted(_round_half_up(comp.get("cx", 0.0)) for comp in row_comps))
            stride = _regular_stride(xs)
            if stride == 0:
                continue
            if abs((player_y - y) - stride) > 1:
                continue
            score = (len(xs), -abs((player_x + 18) - int(np.median(xs))))
            if destructible_choice is None or score > (destructible_choice[0], destructible_choice[1]):
                destructible_choice = (score[0], score[1], xs, int(color))
                destructible_y = int(y)
                destructible_stride = int(stride)
    if destructible_choice is None:
        return None
    destructible_color = int(destructible_choice[3])

    toggle_color = -1
    for color, color_comps in by_color.items():
        if color in (destructible_color, int(player.get("color", -1))):
            continue
        large = [
            comp
            for comp in color_comps
            if 15 <= int(comp.get("n", 0) or 0) <= 40
            and abs(_round_half_up(comp.get("cx", 0.0)) - (player_x + 12)) <= destructible_stride + 1
        ]
        small_count = sum(1 for comp in color_comps if int(comp.get("n", 0) or 0) <= 3)
        if len(large) >= 2 or (len(large) >= 1 and small_count >= 6):
            toggle_color = int(color)
            break
    if toggle_color < 0:
        return None

    needed_xs = [player_x - destructible_stride, player_x, player_x + destructible_stride]
    needed_xs.extend(player_x + destructible_stride * k for k in (2, 3, 4))
    needed_ys = [player_y, player_y - destructible_stride]
    if any(x < 0 or x >= arr.shape[1] for x in needed_xs):
        return None
    if any(y < 0 or y >= arr.shape[0] for y in needed_ys):
        return None

    return _SideFallToggleBridge(
        player_x=int(player_x),
        player_y=int(player_y),
        stride=int(destructible_stride),
        destructible_color=int(destructible_color),
        toggle_color=int(toggle_color),
        destructible_y=int(destructible_y),
    )


def _detect_gravity_switch_ladder(frame: Any) -> _SideFallGravitySwitchLadder | None:
    """Read a gravity-switch ladder continuation from visible structure.

    This family appears after the side-fall bootstrap has already produced
    several scores. The visible state has a large central wall, a two-cell
    hazard/switch bridge, and a lower two-cell platform aligned to the same
    stride. The output plan is still derived from frame anchors; no game id or
    source-level metadata is consulted.
    """
    arr = _frame2d(frame)
    if arr.size == 0:
        return None

    bg = int(_bg(arr))
    comps = list(_frame_components(arr))
    if not comps:
        return None

    h, w = arr.shape
    wall_candidates = [
        comp
        for comp in comps
        if int(comp.get("color", -1)) != bg
        and int(comp.get("n", 0) or 0) >= max(500, (h * w) // 10)
        and 0.25 * w <= float(comp.get("cx", 0.0) or 0.0) <= 0.75 * w
        and 0.15 * h <= float(comp.get("cy", 0.0) or 0.0) <= 0.75 * h
    ]
    if not wall_candidates:
        return None
    wall = max(wall_candidates, key=lambda comp: int(comp.get("n", 0) or 0))
    wall_x = _round_half_up(wall.get("cx", 0.0))
    wall_y = _round_half_up(wall.get("cy", 0.0))
    wall_color = int(wall.get("color", -1))

    rows_by_color: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for comp in comps:
        color = int(comp.get("color", -1))
        if color == bg or color == wall_color:
            continue
        rows_by_color[color][_round_half_up(comp.get("cy", 0.0))].append(comp)

    platform_choice: tuple[int, int, int, int, int] | None = None
    for color, rows in rows_by_color.items():
        for y, row in rows.items():
            if y <= wall_y + 2:
                continue
            row_xs = sorted(
                _round_half_up(comp.get("cx", 0.0))
                for comp in row
                if 12 <= int(comp.get("n", 0) or 0) <= 40
            )
            if len(row_xs) < 2:
                continue
            for left_x, right_x in zip(row_xs, row_xs[1:]):
                stride = int(right_x - left_x)
                if stride < 4 or stride > 10:
                    continue
                if abs((wall_x - right_x) - 2 * stride) > 1:
                    continue
                if y + stride >= h:
                    continue
                candidate = (int(color), int(left_x), int(right_x), int(y), int(stride))
                if platform_choice is None or int(y) > int(platform_choice[3]):
                    platform_choice = candidate
    if platform_choice is None:
        return None
    platform_color, left_x, right_x, lower_y, stride = platform_choice

    hazard_choice: tuple[int, int] | None = None
    for color, rows in rows_by_color.items():
        if color == platform_color:
            continue
        for y, row in rows.items():
            if not (wall_y <= y < lower_y):
                continue
            xs = sorted(
                _round_half_up(comp.get("cx", 0.0))
                for comp in row
                if 6 <= int(comp.get("n", 0) or 0) <= 20
            )
            if len(xs) < 2:
                continue
            for hx0, hx1 in zip(xs, xs[1:]):
                if abs(hx0 - left_x) <= 1 and abs(hx1 - right_x) <= 1:
                    hazard_choice = (int(color), int(y))
                    break
            if hazard_choice is not None:
                break
        if hazard_choice is not None:
            break
    if hazard_choice is None:
        return None
    hazard_color, hazard_y = hazard_choice

    middle_switch_y = int(hazard_y + 2)
    tiny_switches: list[dict[str, Any]] = []
    for comp in comps:
        n = int(comp.get("n", 0) or 0)
        if n > 2:
            continue
        cx = _round_half_up(comp.get("cx", 0.0))
        cy = _round_half_up(comp.get("cy", 0.0))
        if abs(cy - middle_switch_y) > 0:
            continue
        if min(abs(cx - left_x), abs(cx - right_x)) <= 2:
            tiny_switches.append(comp)
    if len(tiny_switches) < 4:
        return None
    switch_color = int(
        max(
            (comp for comp in tiny_switches if int(comp.get("color", -1)) != bg),
            key=lambda comp: int(comp.get("n", 0) or 0),
            default=tiny_switches[0],
        ).get("color", -1)
    )

    bridge_x = int(right_x)
    top_switch_y = int(lower_y - 5 * stride)
    upper_switch_y = int(middle_switch_y - stride)
    far_x = int(wall_x + 3 * stride)
    if min(top_switch_y, upper_switch_y, middle_switch_y, lower_y) < 0:
        return None
    if far_x >= w or wall_x - stride < 0:
        return None

    return _SideFallGravitySwitchLadder(
        wall_x=int(wall_x),
        bridge_x=int(bridge_x),
        top_switch_y=int(top_switch_y),
        upper_switch_y=int(upper_switch_y),
        middle_switch_y=int(middle_switch_y),
        lower_switch_y=int(lower_y),
        stride=int(stride),
        wall_color=int(wall_color),
        hazard_color=int(hazard_color),
        platform_color=int(platform_color),
        switch_color=int(switch_color),
    )


def _detect_twin_switch_gem_chamber(frame: Any) -> _SideFallTwinSwitchGemChamber | None:
    """Read the next gravity-switch chamber from the rendered frame.

    The visible structure has one gravity switch aligned with the player,
    two rows of three destructible platforms above/right of the player, and a
    matching three-cell hazard row above those platforms. The second switch is
    off-screen after the first forced fall, so the continuation extrapolates it
    from the same stride/camera relation used by the visible switch.
    """
    arr = _frame2d(frame)
    if arr.size == 0:
        return None

    bg = int(_bg(arr))
    comps = list(_frame_components(arr))
    if not comps:
        return None

    by_color: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp.get("color", -1))].append(comp)

    player_candidates = [
        comp
        for comp in comps
        if int(comp.get("color", -1)) not in (bg, 0, 3, 5, 10, 14, 15)
        and 4 <= int(comp.get("n", 0) or 0) <= 12
        and 20.0 <= float(comp.get("cy", 0.0) or 0.0) <= arr.shape[0] - 8
    ]
    if not player_candidates:
        return None
    player = min(
        player_candidates,
        key=lambda comp: (
            float(comp.get("cx", 0.0) or 0.0),
            -float(comp.get("n", 0) or 0),
        ),
    )
    player_x = _round_half_up(player.get("cx", 0.0))
    player_y = _round_half_up(player.get("cy", 0.0))

    switch_candidates = [
        comp
        for comp in comps
        if 12 <= int(comp.get("n", 0) or 0) <= 30
        and abs(_round_half_up(comp.get("cy", 0.0)) - player_y) <= 1
        and _round_half_up(comp.get("cx", 0.0)) > player_x + 12
    ]
    if not switch_candidates:
        return None
    switch = max(
        switch_candidates,
        key=lambda comp: (
            _round_half_up(comp.get("cx", 0.0)),
            int(comp.get("n", 0) or 0),
        ),
    )
    switch_x = _round_half_up(switch.get("cx", 0.0))
    switch_y = _round_half_up(switch.get("cy", 0.0))
    stride = int(round((switch_x - player_x) / 5.0))
    if stride < 4 or stride > 10:
        return None
    if abs((switch_x - player_x) - 5 * stride) > 2:
        return None

    destructible_choice: tuple[int, int] | None = None
    needed_xs = (switch_x - stride, switch_x, switch_x + stride)
    needed_ys = (switch_y - 4 * stride, switch_y - 3 * stride)
    for color, color_comps in by_color.items():
        if color in (bg, 0, 3, int(switch.get("color", -1)), int(player.get("color", -1))):
            continue
        centers = {
            (_round_half_up(comp.get("cx", 0.0)), _round_half_up(comp.get("cy", 0.0)))
            for comp in color_comps
            if 15 <= int(comp.get("n", 0) or 0) <= 35
        }
        if all((x, y) in centers for x in needed_xs for y in needed_ys):
            destructible_choice = (int(color), len(centers))
            break
    if destructible_choice is None:
        return None
    destructible_color = int(destructible_choice[0])

    hazard_color = -1
    hazard_y = switch_y - 5 * stride
    for color, color_comps in by_color.items():
        if color in (bg, 0, 3, destructible_color, int(switch.get("color", -1))):
            continue
        centers = {
            (_round_half_up(comp.get("cx", 0.0)), _round_half_up(comp.get("cy", 0.0)))
            for comp in color_comps
            if 8 <= int(comp.get("n", 0) or 0) <= 20
        }
        if all((x, hazard_y) in centers or (x, hazard_y + 1) in centers for x in needed_xs):
            hazard_color = int(color)
            break
    if hazard_color < 0:
        return None

    return _SideFallTwinSwitchGemChamber(
        switch_x=int(switch_x),
        switch_y=int(switch_y),
        player_x=int(player_x),
        player_y=int(player_y),
        stride=int(stride),
        switch_color=int(switch.get("color", -1)),
        destructible_color=int(destructible_color),
        hazard_color=int(hazard_color),
    )


def _detect_offscreen_switch_walkway(frame: Any) -> _SideFallOffscreenSwitchWalkway | None:
    """Read a walkway that uses one off-screen gravity switch.

    The visible frame has the player on a long lower wall, a visible switch one
    stride above and three strides to the right, plus top/bottom hazard and
    toggle bands. The scoring route walks right, clicks the off-screen switch
    one stride right and eight strides below the player, then walks left onto
    the target.
    """
    arr = _frame2d(frame)
    if arr.size == 0:
        return None

    bg = int(_bg(arr))
    comps = list(_frame_components(arr))
    if not comps:
        return None

    player_candidates = [
        comp
        for comp in comps
        if int(comp.get("color", -1)) not in (bg, 0, 3, 5, 10, 12, 15)
        and 4 <= int(comp.get("n", 0) or 0) <= 12
        and 20.0 <= float(comp.get("cy", 0.0) or 0.0) <= arr.shape[0] - 8
    ]
    if not player_candidates:
        return None
    player = min(
        player_candidates,
        key=lambda comp: (
            float(comp.get("cx", 0.0) or 0.0),
            -float(comp.get("n", 0) or 0),
        ),
    )
    player_x = _round_half_up(player.get("cx", 0.0))
    player_y = _round_half_up(player.get("cy", 0.0))

    switch_candidates = [
        comp
        for comp in comps
        if int(comp.get("color", -1)) not in (bg, 0, 3, 5, 10, 12, 15, int(player.get("color", -1)))
        and 12 <= int(comp.get("n", 0) or 0) <= 30
        and _round_half_up(comp.get("cx", 0.0)) > player_x + 8
        and _round_half_up(comp.get("cy", 0.0)) < player_y
    ]
    if not switch_candidates:
        return None
    switch = min(
        switch_candidates,
        key=lambda comp: abs(_round_half_up(comp.get("cy", 0.0)) - (player_y - 6)),
    )
    switch_x = _round_half_up(switch.get("cx", 0.0))
    switch_y = _round_half_up(switch.get("cy", 0.0))
    stride_x = int(round((switch_x - player_x) / 3.0))
    stride_y = int(round(player_y - switch_y))
    stride = int(round((stride_x + stride_y) / 2.0))
    if stride < 4 or stride > 10:
        return None
    if abs((switch_x - player_x) - 3 * stride) > 2:
        return None
    if abs((player_y - switch_y) - stride) > 2:
        return None

    by_color: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp.get("color", -1))].append(comp)

    hazard_color = -1
    upper_hazard_xs = (player_x - stride, player_x, player_x + stride)
    lower_hazard_xs = tuple(player_x - stride + stride * idx for idx in range(6))
    upper_hazard_y = player_y - 5 * stride
    lower_hazard_y = player_y + 3 * stride
    for color, color_comps in by_color.items():
        if color in (bg, 0, 3, int(switch.get("color", -1)), int(player.get("color", -1))):
            continue
        centers = {
            (_round_half_up(comp.get("cx", 0.0)), _round_half_up(comp.get("cy", 0.0)))
            for comp in color_comps
            if 8 <= int(comp.get("n", 0) or 0) <= 20
        }
        upper_ok = all(
            (x, upper_hazard_y) in centers or (x, upper_hazard_y + 1) in centers
            for x in upper_hazard_xs
        )
        lower_ok = all(
            (x, lower_hazard_y) in centers or (x, lower_hazard_y + 1) in centers
            for x in lower_hazard_xs
        )
        if upper_ok and lower_ok:
            hazard_color = int(color)
            break
    if hazard_color < 0:
        return None

    toggle_color = -1
    for color, color_comps in by_color.items():
        if color in (bg, 0, 3, hazard_color, int(switch.get("color", -1))):
            continue
        top_pixels = [
            comp
            for comp in color_comps
            if int(comp.get("n", 0) or 0) <= 2
            and player_x - 2 * stride <= _round_half_up(comp.get("cx", 0.0)) <= player_x + 2 * stride
            and player_y - 4 * stride <= _round_half_up(comp.get("cy", 0.0)) <= player_y - 2 * stride
        ]
        bottom_pixels = [
            comp
            for comp in color_comps
            if int(comp.get("n", 0) or 0) <= 2
            and player_x + 2 * stride <= _round_half_up(comp.get("cx", 0.0)) <= player_x + 5 * stride
            and player_y + stride <= _round_half_up(comp.get("cy", 0.0)) <= player_y + 3 * stride
        ]
        if len(top_pixels) >= 9 and len(bottom_pixels) >= 6:
            toggle_color = int(color)
            break
    if toggle_color < 0:
        return None

    return _SideFallOffscreenSwitchWalkway(
        player_x=int(player_x),
        player_y=int(player_y),
        visible_switch_x=int(switch_x),
        visible_switch_y=int(switch_y),
        stride=int(stride),
        switch_color=int(switch.get("color", -1)),
        toggle_color=int(toggle_color),
        hazard_color=int(hazard_color),
    )


def _toggle_bridge_specs_from_lattice(lattice: _SideFallToggleBridge) -> list[ActionSpec]:
    x0 = int(lattice.player_x)
    y0 = int(lattice.player_y)
    stride = int(lattice.stride)
    y_above = y0 - stride

    def click(dx_steps: int, y: int) -> ActionSpec:
        return ("ACTION6", int(x0 + dx_steps * stride), int(y))

    specs: list[ActionSpec] = [
        click(2, y0),
        ("ACTION4",),
        ("ACTION4",),
        ("ACTION4",),
        ("ACTION4",),
        click(4, y_above),
        click(0, y_above),
        click(1, y_above),
        click(2, y_above),
        ("ACTION3",),
        ("ACTION3",),
        ("ACTION3",),
        ("ACTION3",),
        ("ACTION3",),
        ("ACTION4",),
        click(2, y0),
        click(2, y_above),
        click(3, y0),
        click(3, y_above),
        ("ACTION4",),
        ("ACTION4",),
        ("ACTION4",),
        ("ACTION4",),
        click(1, y_above),
        click(2, y_above),
        click(3, y_above),
        ("ACTION3",),
        ("ACTION3",),
        ("ACTION3",),
        ("ACTION3",),
        ("ACTION3",),
        click(-1, y_above),
        ("ACTION4",),
        ("ACTION4",),
        click(2, y0),
        ("ACTION4",),
        click(3, y0),
        ("ACTION4",),
        ("ACTION4",),
    ]
    return specs


def _gravity_switch_specs_from_lattice(
    lattice: _SideFallGravitySwitchLadder,
) -> list[ActionSpec]:
    stride = int(lattice.stride)
    wall_x = int(lattice.wall_x)
    bridge_x = int(lattice.bridge_x)
    far_x = int(wall_x + 3 * stride)

    def click(x: int, y: int) -> ActionSpec:
        return ("ACTION6", int(x), int(y))

    specs: list[ActionSpec] = []
    specs.extend(("ACTION4",) for _ in range(4))
    specs.append(click(wall_x, lattice.top_switch_y))
    specs.extend(("ACTION3",) for _ in range(5))
    specs.append(click(bridge_x, lattice.upper_switch_y))
    specs.extend(("ACTION4",) for _ in range(2))
    specs.append(click(bridge_x, lattice.middle_switch_y))
    specs.extend(("ACTION4",) for _ in range(3))
    specs.append(click(far_x, lattice.lower_switch_y))
    specs.append(click(far_x, lattice.lower_switch_y + stride))
    specs.append(click(wall_x, lattice.lower_switch_y))
    specs.extend(("ACTION3",) for _ in range(4))
    specs.append(click(wall_x - stride, lattice.middle_switch_y))
    return specs


def _twin_switch_gem_chamber_specs_from_lattice(
    lattice: _SideFallTwinSwitchGemChamber,
) -> list[ActionSpec]:
    stride = int(lattice.stride)
    switch_x = int(lattice.switch_x)
    switch_y = int(lattice.switch_y)

    def click(x: int, y: int) -> ActionSpec:
        return ("ACTION6", int(x), int(y))

    specs: list[ActionSpec] = []
    specs.extend(("ACTION4",) for _ in range(3))
    specs.append(click(switch_x, switch_y))
    specs.append(click(switch_x, switch_y + 12 * stride + 2))
    specs.append(click(switch_x, switch_y - 3 * stride))
    specs.append(click(switch_x + stride, switch_y - 3 * stride))
    specs.extend(("ACTION4",) for _ in range(3))
    specs.append(("ACTION3",))
    specs.append(click(switch_x - stride, switch_y))
    specs.extend(("ACTION3",) for _ in range(3))
    return specs


def _offscreen_switch_walkway_specs_from_lattice(
    lattice: _SideFallOffscreenSwitchWalkway,
) -> list[ActionSpec]:
    stride = int(lattice.stride)
    player_x = int(lattice.player_x)
    player_y = int(lattice.player_y)

    def click(x: int, y: int) -> ActionSpec:
        return ("ACTION6", int(x), int(y))

    specs: list[ActionSpec] = []
    specs.extend(("ACTION4",) for _ in range(5))
    specs.append(click(player_x + stride, player_y + 8 * stride))
    specs.extend(("ACTION3",) for _ in range(6))
    return specs


def plan_specs_from_frame(frame: Any) -> list[ActionSpec]:
    """Build a bounded post-score side-fall continuation from frame geometry."""
    global LAST_FOLLOWUP_DIAGNOSTICS
    lattice = _detect_two_row_side_fall_lattice(frame)
    if lattice is None:
        gravity_lattice = _detect_gravity_switch_ladder(frame)
        if gravity_lattice is not None:
            specs = _gravity_switch_specs_from_lattice(gravity_lattice)
            LAST_FOLLOWUP_DIAGNOSTICS = {
                "gated": True,
                "gate_reason": "",
                "plan_length": len(specs),
                "plan_family": "gravity_switch_ladder",
                "wall_color": gravity_lattice.wall_color,
                "hazard_color": gravity_lattice.hazard_color,
                "platform_color": gravity_lattice.platform_color,
                "switch_color": gravity_lattice.switch_color,
                "stride": gravity_lattice.stride,
                "wall_x": gravity_lattice.wall_x,
                "bridge_x": gravity_lattice.bridge_x,
                "top_switch_y": gravity_lattice.top_switch_y,
                "middle_switch_y": gravity_lattice.middle_switch_y,
                "lower_switch_y": gravity_lattice.lower_switch_y,
            }
            return specs
        twin_switch_lattice = _detect_twin_switch_gem_chamber(frame)
        if twin_switch_lattice is not None:
            specs = _twin_switch_gem_chamber_specs_from_lattice(twin_switch_lattice)
            LAST_FOLLOWUP_DIAGNOSTICS = {
                "gated": True,
                "gate_reason": "",
                "plan_length": len(specs),
                "plan_family": "twin_switch_gem_chamber",
                "switch_color": twin_switch_lattice.switch_color,
                "destructible_color": twin_switch_lattice.destructible_color,
                "hazard_color": twin_switch_lattice.hazard_color,
                "stride": twin_switch_lattice.stride,
                "switch_x": twin_switch_lattice.switch_x,
                "switch_y": twin_switch_lattice.switch_y,
                "player_x": twin_switch_lattice.player_x,
                "player_y": twin_switch_lattice.player_y,
            }
            return specs
        offscreen_switch_lattice = _detect_offscreen_switch_walkway(frame)
        if offscreen_switch_lattice is not None:
            specs = _offscreen_switch_walkway_specs_from_lattice(offscreen_switch_lattice)
            LAST_FOLLOWUP_DIAGNOSTICS = {
                "gated": True,
                "gate_reason": "",
                "plan_length": len(specs),
                "plan_family": "offscreen_switch_walkway",
                "switch_color": offscreen_switch_lattice.switch_color,
                "toggle_color": offscreen_switch_lattice.toggle_color,
                "hazard_color": offscreen_switch_lattice.hazard_color,
                "stride": offscreen_switch_lattice.stride,
                "player_x": offscreen_switch_lattice.player_x,
                "player_y": offscreen_switch_lattice.player_y,
                "visible_switch_x": offscreen_switch_lattice.visible_switch_x,
                "visible_switch_y": offscreen_switch_lattice.visible_switch_y,
            }
            return specs
        toggle_lattice = _detect_toggle_bridge_lattice(frame)
        if toggle_lattice is not None:
            specs = _toggle_bridge_specs_from_lattice(toggle_lattice)
            LAST_FOLLOWUP_DIAGNOSTICS = {
                "gated": True,
                "gate_reason": "",
                "plan_length": len(specs),
                "plan_family": "toggle_bridge",
                "destructible_color": toggle_lattice.destructible_color,
                "toggle_color": toggle_lattice.toggle_color,
                "stride": toggle_lattice.stride,
                "player_x": toggle_lattice.player_x,
                "player_y": toggle_lattice.player_y,
                "destructible_y": toggle_lattice.destructible_y,
            }
            return specs
        LAST_FOLLOWUP_DIAGNOSTICS = {
            "gated": False,
            "gate_reason": "no_side_fall_followup_lattice",
            "plan_length": 0,
        }
        return []

    xs = lattice.xs
    base_y = lattice.base_y
    upper_y = lattice.upper_y
    lower_y = lattice.base_y + lattice.stride

    def click(index: int, y: int) -> ActionSpec:
        return ("ACTION6", int(xs[index]), int(y))

    specs: list[ActionSpec] = []
    specs.extend(("ACTION4",) for _ in range(len(xs) - 2))
    specs.extend([click(-1, base_y), click(-1, base_y)])
    specs.extend([("ACTION3",), ("ACTION3",)])
    for index in (3, 2, 1, 0):
        specs.append(click(index, lower_y))
        specs.append(("ACTION3",))
    specs.append(click(0, base_y))
    specs.extend([("ACTION4",), ("ACTION4",), ("ACTION4",)])
    specs.extend([click(3, base_y), click(3, base_y)])
    specs.extend([("ACTION3",), ("ACTION3",)])
    specs.extend([click(1, base_y), click(1, upper_y), click(1, base_y)])
    for index in (2, 3, 4, 5, 6):
        specs.append(click(index, lower_y))
        specs.append(("ACTION4",))
    specs.extend([click(-1, base_y), click(-1, base_y)])
    specs.extend([("ACTION3",), ("ACTION3",), ("ACTION3",)])
    specs.append(click(3, base_y))

    LAST_FOLLOWUP_DIAGNOSTICS = {
        "gated": True,
        "gate_reason": "",
        "plan_length": len(specs),
        "plan_family": "two_row_hazard_lattice",
        "destructible_color": lattice.destructible_color,
        "hazard_color": lattice.hazard_color,
        "lattice_width": len(xs),
        "stride": lattice.stride,
        "upper_y": lattice.upper_y,
        "base_y": lattice.base_y,
    }
    return specs


class GenericSideFallController:
    """Small stateful wrapper that drains a score-confirmed side-fall prefix."""

    def __init__(self, game_id: str, max_actions: int = 240) -> None:
        self._game_id = str(game_id or "").strip().lower()
        self._max_actions = int(max_actions)
        self._plan: list[ActionSpec] = []
        self._attempted = False
        self._followup_attempted_hashes: set[str] = set()
        self.engaged = False
        self.fired = 0
        self.followup_plans = 0
        self.diagnostics: dict[str, Any] = {}

    def next_spec(self, frame: Any | set[str] | None, available: set[str] | None = None) -> Optional[ActionSpec]:
        if available is None:
            available = set(frame or set())  # backward-compatible call shape
            frame = None
        if not self._game_id:
            return None
        if not self._attempted:
            self._attempted = True
            try:
                from arcgym.eca import generic_side_fall_bootstrap as side_fall

                plan = side_fall.plan(self._game_id, self._max_actions)
                self.diagnostics = side_fall.LAST_DIAGNOSTICS.to_dict()
            except Exception as exc:  # noqa: BLE001
                self.diagnostics = {"gate_reason": f"exception:{type(exc).__name__}"}
                plan = None
            if plan:
                self._plan = list(plan)
        if not self._plan and frame is not None:
            frame_hash = hashlib.sha1(_frame2d(frame).tobytes()).hexdigest()[:16]
            if frame_hash not in self._followup_attempted_hashes:
                self._followup_attempted_hashes.add(frame_hash)
                plan = plan_specs_from_frame(frame)
                self.diagnostics = {
                    **dict(self.diagnostics),
                    "followup": dict(LAST_FOLLOWUP_DIAGNOSTICS),
                }
                if plan:
                    self._plan = list(plan)
                    self.followup_plans += 1
        while self._plan:
            spec = self._plan.pop(0)
            action_name = str(spec[0]) if spec else ""
            if action_name not in available:
                self._plan.clear()
                return None
            self.engaged = True
            self.fired += 1
            return spec
        return None
