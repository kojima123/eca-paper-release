"""Control-marker piece family (ka59) — the board DRAWS which piece the
keys drive.

Ported from the parallel session's per-cell reading (their `ka59 = 1`
arc), but onto the axis that converts: a mechanic FAMILY runs on the
executor, whose actions own 87% of the suite's play and every stable
full clear, whereas the retrodict lane that first found this reading
delivered 37 certified plans on canonical and converted zero levels
(docs/s16_retrodict_lane_regression_20260817.md).

What the frame gives, measured on ka59's FIRST frame with no window
and no induction (docs/design_port_r30_words_as_families_20260817.md):

    two congruent 3x3 rings of colour 14 with a centre hole;
    the centre cell IS the control marker;
    centre 0 = this piece is under control, centre 4 = it is not.

What the frame does NOT give:

* the push distance. A move whose destination enters another piece
  displaces THAT piece — on ka59 by five strides, through a wall the
  mover itself cannot cross — and the pusher stays put. This module
  proposes the piece PITCH and lets the executor's per-step lease
  revoke a wrong guess; that is the toolbox's existing contract and it
  costs nothing when the guess is right.
* which colours are background. The reading only fires under the
  `dominant + static` hypothesis (dominant-only returns None on ka59),
  so both variants are tried in the parallel session's order —
  dominant first, then dominant plus each further colour in board
  order. A module that tried only the dominant colour would be
  silently inert.

Fail-closed everywhere: an unread frame, an ambiguous family, or a
marker that is not carried by exactly one piece all return
``supported: False`` and no plan.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .base import MechanicContext

FAMILY_ID = "control_marker_pieces"
_MOVE_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
# board deltas the engine's move actions produce, in (dx, dy)
_MOVE_DELTAS = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}
# The bridge renders a plan's engine action into the MODEL name
# (ACTION4 -> RIGHT) and then feeds that back into predict_transition.
# A module that only knows engine names rejects its own plan, and the
# bridge ANDs prediction acceptance into `certified` — measured
# 2026-08-17: status=detected, plan_len=1, certified=false, with the
# plan reading [{'action': 'RIGHT'}]. Accept both spellings.
_ACTION_ALIASES = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
    "MOUSE": "ACTION6",
}


def _engine_action(action: Any) -> str:
    name = str(getattr(action, "name", action) or "").split(" ")[0].upper()
    if isinstance(action, Mapping):
        name = str(action.get("action", "") or "").split(" ")[0].upper()
    # The production framework records payload-bearing clicks as
    # ``MOUSE(row=31, col=43)`` in HistoryEntry.  Treat that display form as
    # the same observed ACTION6 vocabulary as the raw engine/model spellings.
    if name.startswith("MOUSE("):
        name = "MOUSE"
    return _ACTION_ALIASES.get(name, name)
_MIN_PIECES = 2
_MAX_BACKGROUND_VARIANTS = 4
# A body is what makes the pieces RECOGNISABLE as congruent shapes.
# Measured 2026-08-17: without this floor the reading fired on r11l,
# whose 12 one-cell components trivially "differ in exactly one
# offset" because the body is EMPTY. ka59's body is 8 cells.
_MIN_BODY_CELLS = 4


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _grid(frame: Any) -> list[list[int]] | None:
    board = getattr(frame, "grid", frame)
    if isinstance(board, list) and board and isinstance(board[0], (list, np.ndarray)):
        pass
    elif isinstance(board, list) and board and isinstance(board[0], (int, float)):
        return None
    try:
        arr = np.asarray(board)
        if arr.ndim == 3:
            arr = arr[-1]
        if arr.ndim != 2 or arr.size == 0:
            return None
        return [[int(c) for c in row] for row in arr]
    except Exception:
        return None


def _background_variants(board: list[list[int]]) -> list[set[int]]:
    """Both hypotheses, in the parallel session's order.

    Their `_background_variants` emits dominant-only FIRST and then
    dominant + static, and the order is the caller's tie-break. ka59
    needs the second one — dominant-only returns None — which is why
    this is a list and not a single guess.
    """
    census = Counter(c for row in board for c in row)
    if not census:
        return []
    ordered = [colour for colour, _n in census.most_common()]
    dominant = {ordered[0]}
    variants = [dominant]
    for colour in ordered[1:_MAX_BACKGROUND_VARIANTS]:
        variants.append(dominant | {colour})
    return variants


def _components(board: list[list[int]], background: set[int]) -> list[dict]:
    """4-connected non-background components as offset->colour maps."""
    height, width = len(board), len(board[0])
    seen = [[False] * width for _ in range(height)]
    out: list[dict] = []
    for y in range(height):
        for x in range(width):
            if seen[y][x] or board[y][x] in background:
                continue
            stack = [(x, y)]
            seen[y][x] = True
            cells: list[tuple[int, int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy, board[cy][cx]))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if (0 <= nx < width and 0 <= ny < height
                            and not seen[ny][nx]
                            and board[ny][nx] not in background):
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            ox, oy = min(xs), min(ys)
            out.append({
                "origin": (ox, oy),
                "size": (max(xs) - ox + 1, max(ys) - oy + 1),
                "cells": {(cx - ox, cy - oy): value for cx, cy, value in cells},
            })
    return out


def _control_family(board: list[list[int]], background: set[int]):
    """Congruent pieces differing in EXACTLY ONE offset = the marker.

    Returns ``(body, marker_offset, count)`` where ``body`` is the
    shared cell map excluding the marker. Fails closed on anything
    ambiguous: fewer than two congruent pieces, more than one differing
    offset, or a marker value that is not unique to a single piece.
    """
    comps = _components(board, background)
    groups: dict[tuple, list[dict]] = {}
    for comp in comps:
        key = (comp["size"], tuple(sorted(comp["cells"])))
        groups.setdefault(key, []).append(comp)
    for key, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) < _MIN_PIECES:
            continue
        offsets = sorted(members[0]["cells"])
        differing = [
            off for off in offsets
            if len({m["cells"][off] for m in members}) > 1
        ]
        if len(differing) != 1:
            continue
        marker = differing[0]
        body = tuple(sorted(
            (off, members[0]["cells"][off])
            for off in offsets if off != marker
        ))
        if len(body) < _MIN_BODY_CELLS:
            continue
        values = Counter(m["cells"][marker] for m in members)
        # The selected value is the MINORITY: with >=3 pieces exactly
        # one carries it and the reading is unambiguous. With exactly 2
        # pieces both values are carried once, so one frame cannot say
        # which means "under control" — r30 resolves that over a whole
        # window. The frame-only honest answer is to report the
        # ambiguity and let the caller resolve it from the previous
        # transition (which piece actually moved).
        minority = [v for v, n in values.items() if n == 1]
        if len(minority) == 1:
            return body, marker, len(members), members, minority[0], False
        if len(values) == len(members) == 2:
            return body, marker, len(members), members, None, True
        continue
    return None


def _piece_origins(board: list[list[int]], marker: tuple[int, int],
                   template: tuple | None) -> list[tuple[int, int]] | None:
    """Piece origins from EITHER reader.

    The ambiguity resolver used the component reader only, so on the
    very frames where the template reader was needed (pieces touching,
    contact face redrawn) the resolution failed and the whole state
    went unread — measured on ka59: the template located both pieces
    correctly while bind_current still reported unbound.
    """
    for background in _background_variants(board):
        found = _control_family(board, background)
        if found is None:
            continue
        _body, prev_marker, _count, members, _sel, _amb = found
        if prev_marker == marker:
            return [tuple(m["origin"]) for m in members]
    if template is not None:
        body, tmarker, count = template
        if tmarker == marker:
            body_colours = {v for _o, v in body}
            seen = {int(v) for row in board for v in row}
            for allow in [None] + [{c} for c in sorted(seen - body_colours)]:
                members = _locate_template(board, body, marker, count,
                                           allow=allow)
                if members:
                    return [tuple(m["origin"]) for m in members]
    return None


def _moved_piece_index(previous: Any, pieces: list[dict],
                       marker: tuple[int, int],
                       template: tuple | None = None) -> int | None:
    """Which piece changed origin since the previous frame.

    This is how a 2-piece board resolves "which marker value means
    under control" WITHOUT hardcoding a palette: the piece that the
    last action actually moved is the controlled one. Frame-grounded,
    so it survives a palette shift; returns None when the previous
    frame is missing or nothing moved (fail closed).
    """
    prev_board = _grid(previous)
    if prev_board is None:
        return None
    prev_origins = _piece_origins(prev_board, marker, template)
    if prev_origins is None or len(prev_origins) != len(pieces):
        return None
    known = set(prev_origins)
    moved = [i for i, piece in enumerate(pieces)
             if tuple(piece["origin"]) not in known]
    return moved[0] if len(moved) == 1 else None



def _locate_template(board: list[list[int]], body: tuple,
                     marker: tuple[int, int],
                     count: int,
                     allow: set[int] | None = None) -> list[dict] | None:
    """Find the known piece shape by TEMPLATE MATCH, not components.

    Measured 2026-08-17: component-based reading found the ka59 pieces
    on frame 1 and NEVER AGAIN — one move puts the two pieces adjacent
    and 4-connectivity merges them into a single component, so "two
    congruent pieces" vanishes exactly when the interesting things
    start happening. The parallel session hit the same wall and solved
    it the same way: match the known extents and tolerate occluders.

    ``body`` is the offset->colour map WITHOUT the marker cell, so a
    piece is located wherever every body cell matches; the marker cell
    is read off rather than required, which is what lets the two
    pieces be distinguished while touching.
    """
    if not board or not board[0] or not body:
        return None
    height, width = len(board), len(board[0])
    offsets = [off for off, _v in body]
    w = max(ox for ox, _oy in offsets) + 1
    h = max(oy for _ox, oy in offsets) + 1
    # CONTACT TOLERANCE. Measured cell by cell on ka59 (2026-08-17):
    # when the controlled piece comes to rest against another piece,
    # the touching face is REDRAWN in the marker colour —
    #     row30  14 14  0      (the right column of the moving piece)
    #     row31  14  0  0
    #     row32  14 14  0
    # so a strict body match loses the piece exactly when contact (the
    # mechanic's whole point) happens. The parallel session hit this
    # too and solved it the same way: do not guess the deviating
    # colour, COLLECT the candidates and test them one at a time.
    hits: list[dict] = []
    tolerated = allow if allow is not None else set()
    for oy in range(height - h + 1):
        for ox in range(width - w + 1):
            ok = True
            for (dx, dy), value in body:
                cell = board[oy + dy][ox + dx]
                if cell != value and cell not in tolerated:
                    ok = False
                    break
            if not ok:
                continue
            mx, my = ox + marker[0], oy + marker[1]
            if not (0 <= my < height and 0 <= mx < width):
                continue
            deviations = sum(
                1 for (dx, dy), value in body
                if board[oy + dy][ox + dx] != value
            )
            hits.append({
                "origin": (ox, oy),
                "size": (w, h),
                "deviations": deviations,
                "cells": {**{off: value for off, value in body},
                          marker: int(board[my][mx])},
            })
    # Tolerance widens matching, so a contact-deformed board yields
    # MORE placements than there are pieces (measured on ka59: with
    # the contact colour allowed, x=24 and x=25 both match). Two
    # physical facts disambiguate without any replay machinery:
    # pieces cannot overlap, and the truest reading is the one that
    # needs the fewest tolerated cells.
    hits.sort(key=lambda hit: (hit["deviations"], hit["origin"]))
    chosen: list[dict] = []
    for hit in hits:
        ox, oy = hit["origin"]
        if any(abs(ox - cx) < w and abs(oy - cy) < h
               for cx, cy in (c["origin"] for c in chosen)):
            continue
        chosen.append(hit)
    if len(chosen) != count:
        return None
    chosen.sort(key=lambda hit: (hit["origin"][1], hit["origin"][0]))
    return chosen


def _template_from_history(context: Any):
    """The earliest frame in the window where the pieces WERE
    separable gives the template; later frames are then located by
    matching it. Without this the family reads one frame per episode."""
    for board, _action in _history_boards(context):
        for background in _background_variants(board):
            found = _control_family(board, background)
            if found is not None:
                body, marker, count, _members, _sel, _amb = found
                return body, marker, count
    return None


def _template_payload(template: tuple | None) -> dict[str, Any] | None:
    if template is None:
        return None
    body, marker, count = template
    return {
        "body": [
            [int(offset[0]), int(offset[1]), int(value)]
            for offset, value in body
        ],
        "marker_offset": [int(marker[0]), int(marker[1])],
        "piece_count": int(count),
    }


def _template_from_payload(payload: Any) -> tuple | None:
    if not isinstance(payload, Mapping):
        return None
    try:
        body = tuple(
            ((int(row[0]), int(row[1])), int(row[2]))
            for row in payload.get("body", ())
        )
        marker = tuple(int(value) for value in payload["marker_offset"])
        count = int(payload["piece_count"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if not body or len(marker) != 2 or count < 2:
        return None
    return body, marker, count


def _observed_stride(previous: Any, pieces: list[dict],
                     marker: tuple[int, int],
                     template: tuple | None = None) -> int | None:
    """How far a piece actually moved between the two frames.

    Returns the Chebyshev displacement of the single piece whose origin
    changed, or None when the previous frame is missing, unreadable, or
    more than one piece moved (fail closed rather than average).
    """
    prev_board = _grid(previous)
    if prev_board is None:
        return None
    prev_origins = _piece_origins(prev_board, marker, template)
    if prev_origins is None or len(prev_origins) != len(pieces):
        return None
    now_origins = sorted(tuple(p["origin"]) for p in pieces)
    moved = [(a, b) for a, b in zip(sorted(prev_origins), now_origins)
             if a != b]
    if len(moved) != 1:
        return None
    (ax, ay), (bx, by) = moved[0]
    step = max(abs(bx - ax), abs(by - ay))
    return step or None


def _read_state(frame: Any,
                previous_frame: Any = None,
                template: tuple | None = None,
                selected_value: int | None = None,
                *,
                include_rings: bool = True) -> dict[str, Any] | None:
    board = _grid(frame)
    if board is None or not board or not board[0]:
        return None
    candidates = []
    for background in _background_variants(board):
        found = _control_family(board, background)
        if found is not None:
            candidates.append(found)
            break
    if not candidates and template is not None:
        # components merged or a face was redrawn on contact: locate
        # the known shape, widening the tolerated colour set one
        # candidate at a time (never guessed — the candidates are the
        # colours the board actually shows inside the piece extents).
        body, marker, count = template
        body_colours = {value for _off, value in body}
        seen = {int(v) for row in board for v in row}
        members = None
        for allow in [None] + [{c} for c in sorted(seen - body_colours)]:
            members = _locate_template(board, body, marker, count,
                                       allow=allow)
            if members:
                break
        if members:
            values = Counter(m["cells"][marker] for m in members)
            minority = [v for v, n in values.items() if n == 1]
            if len(minority) == 1:
                candidates.append((body, marker, count, members,
                                   minority[0], False))
            elif len(values) == len(members) == 2:
                candidates.append((body, marker, count, members, None, True))
    learned_selected_value = selected_value
    for found in candidates:
        body, marker, count, members, frame_selected_value, ambiguous = found
        resolved_selected_value = frame_selected_value
        pieces = []
        for member in members:
            ox, oy = member["origin"]
            pieces.append({
                "origin": [int(ox), int(oy)],
                "size": [int(member["size"][0]), int(member["size"][1])],
                "marker_value": int(member["cells"][marker]),
                "selected": bool(resolved_selected_value is not None
                                 and member["cells"][marker]
                                 == resolved_selected_value),
                "selection_source": ("minority_marker_value"
                                     if resolved_selected_value is not None
                                     else "unresolved"),
            })
        pieces.sort(key=lambda p: (p["origin"][1], p["origin"][0]))
        observed_stride = _observed_stride(previous_frame, pieces, marker,
                                           template=template)
        if ambiguous and learned_selected_value is not None:
            # a LEARNED value beats the moved-piece heuristic, which is
            # wrong on a push (the controlled piece stays)
            resolved_selected_value = learned_selected_value
            for piece in pieces:
                piece["selected"] = (
                    piece["marker_value"] == resolved_selected_value
                )
                piece["selection_source"] = "learned_marker_value"
            if sum(1 for p in pieces if p["selected"]) == 1:
                ambiguous = False
        if ambiguous:
            # two pieces, two marker values: the frame alone cannot say
            # which value means "under control". Resolve it from the
            # previous transition, or fail closed.
            idx = _moved_piece_index(previous_frame, pieces, marker,
                                     template=template)
            if idx is None:
                return None
            for other, piece in enumerate(pieces):
                piece["selected"] = other == idx
                piece["selection_source"] = "previous_transition"
            resolved_selected_value = pieces[idx]["marker_value"]
        controlled = [p for p in pieces if p["selected"]]
        if len(controlled) != 1:
            return None
        xs = sorted({p["origin"][0] for p in pieces})
        ys = sorted({p["origin"][1] for p in pieces})
        # pitch = the smallest positive gap between piece origins; the
        # push distance is proposed from it and verified by the lease.
        gaps = [b - a for a, b in zip(xs, xs[1:])] + \
               [b - a for a, b in zip(ys, ys[1:])]
        pitch = min([g for g in gaps if g > 0], default=0)
        rings: list[dict[str, Any]] = []
        if include_rings:
            dominant_colour = Counter(
                int(value) for row in board for value in row
            ).most_common(1)[0][0]
            rings = _find_rings(
                board, tuple(pieces[0]["size"]),
                {int(v) for _o, v in body} | {int(dominant_colour)},
                piece_origins=[tuple(p["origin"]) for p in pieces])
        return {
            "background": sorted(int(c) for c in background),
            "marker_offset": [int(marker[0]), int(marker[1])],
            "selected_marker_value": int(resolved_selected_value),
            "piece_size": pieces[0]["size"],
            "pieces": pieces,
            "controlled_index": pieces.index(controlled[0]),
            "pitch": int(pitch),
            # The stride a move actually produced, read off the last
            # transition. Measured 2026-08-17: guessing it (pitch //
            # pieces) got the family certified twice on ka59 and then
            # REVOKED by the lease on a grounded_transition_mismatch.
            # The previous frame is already being read for the
            # selection, so the true displacement is free — and a
            # measured stride is the difference between a plan that
            # survives verification and one that does not.
            "observed_stride": (None if observed_stride is None
                                else int(observed_stride)),
            # Exclude the repeated BODY and the one dominant canvas colour.
            # The gate's multi-colour background hypothesis cannot be reused:
            # it includes ka59's colour-4 target rings.  The unselected marker
            # also shares that palette, so excluding marker values erases the
            # real goal.  Ring structure is the remaining guard.
            "rings": rings,
            "body_cells": len(body),
        }
    return None


def _binding_key(state: Mapping[str, Any]) -> str:
    return hashlib.sha1(
        json.dumps(state, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def bind_current(
    frame: Any,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the congruent piece family and which piece is under control."""
    previous = None
    if isinstance(context, MechanicContext):
        previous = context.previous_frame
    elif isinstance(context, Mapping):
        previous = context.get("previous_frame")
    template = _template_from_history(context)
    learned = None
    if template is not None:
        learned = _learn_selected_value(context, template[1], template)
    state = _read_state(frame, previous_frame=previous,
                        template=template, selected_value=learned)
    if state is None:
        return {
            "schema_version": "tufa.control_marker.binding.v1",
            "mechanic_id": FAMILY_ID,
            "family_id": FAMILY_ID,
            "status": "unbound",
            "supported": False,
            "reason": "no_unique_control_marker_family",
        }
    return _json({
        "schema_version": "tufa.control_marker.binding.v1",
        "mechanic_id": FAMILY_ID,
        "family_id": FAMILY_ID,
        "status": "bound",
        "supported": True,
        "binding_key": _binding_key(state),
        "state": state,
    })


def compile_schema(
    frame: Any,
    level: Any,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a coordinate-free schema; ``level`` is metadata only."""
    binding = bind_current(frame, context=context)
    state = binding.get("state") or {}
    return _json({
        "schema_id": "tufa.control_marker_pieces.v1",
        "schema_version": "tufa.control_marker.schema.v1",
        "mechanic_id": FAMILY_ID,
        "family_id": FAMILY_ID,
        "level": str(level),
        "supported": bool(binding.get("supported")),
        # The bridge ANDs `schema["roles"]` and `required_bindings`
        # into `supported`, and `supported` into `certified`. A schema
        # without them is bound, plans, and can never certify —
        # measured 2026-08-17 (supported=False while binding.supported
        # and schema.supported were both True).
        "roles": [
            "congruent_piece_family",
            "drawn_control_marker",
            "piece_lattice_pitch",
        ],
        "required_bindings": [
            "congruent_piece_family",
            "drawn_control_marker",
            "controlled_piece_identity",
        ],
        "marker_offset": state.get("marker_offset"),
        "selected_marker_value": state.get("selected_marker_value"),
        "piece_count": len(state.get("pieces") or []),
        "piece_size": state.get("piece_size"),
        "pitch": state.get("pitch"),
        "binding": binding,
    })


def _allowed(valid_actions: Iterable[Any] | None) -> set[str]:
    if valid_actions is None:
        return set(_MOVE_ACTIONS)
    names = set()
    for item in valid_actions:
        name = _engine_action(item)
        if name in _MOVE_DELTAS or name == "ACTION6":
            names.add(name)
    return names


def _blocked(reason: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    return _json({
        "schema_version": "tufa.control_marker.plan.v1",
        "mechanic_id": FAMILY_ID,
        "family_id": FAMILY_ID,
        "certified": False,
        "supported": False,
        "actions": [],
        "reason": reason,
        "binding": dict(binding),
    })



def _history_boards(context: Any) -> list[tuple[list, str]]:
    """(board, action) pairs from the episode history the toolbox
    already delivers. No probe window is needed: the family learns
    from whatever the agent has played."""
    if isinstance(context, MechanicContext):
        entries = list(context.history or ())
    elif isinstance(context, Mapping):
        entries = list(context.get("history") or ())
    else:
        return []
    out = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        board = _grid(entry.get("frame"))
        if board is None:
            continue
        out.append((board, _engine_action(entry.get("action", ""))))
    return out




def _components_of_colour(board: list[list[int]], colour: int) -> list[dict]:
    """4-connected components of ONE colour.

    Ring detection must not depend on which background hypothesis the
    piece gate happened to use: measured 2026-08-17, when the gate
    fired under `dominant + 4` the ring colour became "background" and
    both rings vanished from the state. A ring is a frame of a single
    colour, so look for it per colour and the coupling disappears.
    """
    height, width = len(board), len(board[0])
    seen = [[False] * width for _ in range(height)]
    out: list[dict] = []
    for y in range(height):
        for x in range(width):
            if seen[y][x] or board[y][x] != colour:
                continue
            stack = [(x, y)]
            seen[y][x] = True
            cells = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if (0 <= nx < width and 0 <= ny < height
                            and not seen[ny][nx]
                            and board[ny][nx] == colour):
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            out.append({"origin": (min(xs), min(ys)),
                        "size": (max(xs) - min(xs) + 1,
                                 max(ys) - min(ys) + 1),
                        "count": len(cells)})
    return out


def _find_rings(board: list[list[int]], piece_size: tuple[int, int],
                exclude: set[int],
                piece_origins: list[tuple[int, int]] | None = None
                ) -> list[dict]:
    """Closed frames whose hole is exactly one piece, scanned DIRECTLY.

    Measured on ka59 (2026-08-17): two 5x5 frames with a 3x3 floor
    hole, at (44,26) and (11,32). The parallel session's record names
    the win as "both pieces inside their respective rings", which is
    why a piece-to-piece objective cannot score here.

    Connected components were the first attempt and they lose a ring
    the moment a PIECE rests on it (measured: the east ring vanished
    while a piece covered one corner cell) — the same occlusion lesson
    the piece reader already learned. A frame has a known size, so scan
    for it directly and treat piece-covered cells as satisfied rather
    than as evidence against.
    """
    pw, ph = piece_size
    w, h = pw + 2, ph + 2
    height, width = len(board), len(board[0])
    covered: set[tuple[int, int]] = set()
    for ox, oy in (piece_origins or []):
        for dy in range(ph):
            for dx in range(pw):
                covered.add((ox + dx, oy + dy))
    rings: list[dict] = []
    for oy in range(height - h + 1):
        for ox in range(width - w + 1):
            frame_cells = [(ox + dx, oy + dy)
                           for dy in range(h) for dx in range(w)
                           if dx in (0, w - 1) or dy in (0, h - 1)]
            values = {board[y][x] for x, y in frame_cells
                      if (x, y) not in covered}
            values -= set(exclude)
            if len(values) != 1:
                continue
            frame_colour = next(iter(values))
            if any(board[y][x] != frame_colour and (x, y) not in covered
                   for x, y in frame_cells):
                continue
            interior = [(ox + 1 + dx, oy + 1 + dy)
                        for dy in range(ph) for dx in range(pw)]
            inside = {board[y][x] for x, y in interior
                      if (x, y) not in covered}
            if len(inside) > 1 or frame_colour in inside:
                continue
            rings.append({
                "origin": [int(ox), int(oy)],
                "size": [int(w), int(h)],
                "hole_origin": [int(ox) + 1, int(oy) + 1],
                "frame_colour": int(frame_colour),
                "hole_colour": (int(next(iter(inside))) if inside else None),
            })
    # A piece resting on a uniform traversable floor can make the surrounding
    # 5x5 patch look exactly like one occupied ring: the piece covers the
    # interior and the floor supplies a monochrome perimeter.  The family
    # objective, however, is drawn as a repeated same-colour target set (one
    # target per congruent piece).  Keep only frame colours with enough
    # repeated rings to house the whole family; this removes the transient
    # single floor halo without palette or coordinate rules.
    required = max(1, len(piece_origins or ()))
    frame_colour_counts = Counter(ring["frame_colour"] for ring in rings)
    rings = [
        ring for ring in rings
        if frame_colour_counts[ring["frame_colour"]] >= required
    ]
    rings.sort(key=lambda r: (r["hole_origin"][1], r["hole_origin"][0]))
    return rings


def _containment_cost(pieces: list[dict], rings: list[dict]) -> int:
    """Assignment-sum distance from pieces to ring holes.

    The parallel session's record is explicit that a per-actor MINIMUM
    reads "one housed = done" while the win needs ALL pieces housed —
    so the cost is a sum over an assignment, and it reaches 0 only when
    every piece sits in a hole.
    """
    if not rings or not pieces:
        return 0
    import itertools
    holes = [tuple(r["hole_origin"]) for r in rings]
    if len(holes) < len(pieces):
        # Fewer holes than pieces: an assignment does not exist, and
        # permutations() would silently return NOTHING, leaving the
        # cost at 0 — i.e. "already won" (measured 2026-08-17 with one
        # ring occluded). Fall back to the best per-piece distance
        # sum, which is still a real gradient and never reads as won.
        return int(sum(min(abs(p["origin"][0] - hx) + abs(p["origin"][1] - hy)
                           for hx, hy in holes) for p in pieces))
    best = None
    for perm in itertools.permutations(range(len(holes)), len(pieces)):
        total = 0
        for piece, hole_index in zip(pieces, perm):
            px, py = piece["origin"]
            hx, hy = holes[hole_index]
            total += abs(px - hx) + abs(py - hy)
        if best is None or total < best:
            best = total
    return int(best if best is not None else 0)


def _learn_selected_value(context: Any, marker: tuple[int, int],
                          template: tuple | None) -> int | None:
    """Which marker VALUE means "under control", learned from the window.

    The moved-piece heuristic is right for a move and WRONG for a push:
    on a push the controlled piece stays and the other one travels
    (measured on ka59 — the resolver reported the pushed piece as
    selected). The value itself is stable across the episode, so learn
    it once from a transition that cannot be a push: exactly one piece
    moved AND its destination did not touch another piece.
    """
    boards = _history_boards(context)
    prev_board = None
    for (before_board, _a0), (after_board, action) in zip(boards, boards[1:]):
        if action not in _MOVE_DELTAS:
            prev_board = before_board
            continue
        ob = _piece_origins(before_board, marker, template)
        oa = _piece_origins(after_board, marker, template)
        if ob is None or oa is None or len(ob) != len(oa) or len(ob) < 2:
            prev_board = before_board
            continue
        ob_s, oa_s = sorted(ob), sorted(oa)
        moved = [i for i, (a, b) in enumerate(zip(ob_s, oa_s)) if a != b]
        if len(moved) != 1:
            prev_board = before_board
            continue
        idx = moved[0]
        start, end = ob_s[idx], oa_s[idx]
        # a push moves a piece that the MOVER walked into, so the
        # travelling piece would have been adjacent to another piece
        # before it moved; a plain move is not.
        others = [o for j, o in enumerate(ob_s) if j != idx]
        w = h = 3
        found = _control_family_size(before_board, marker, template)
        if found:
            w, h = found
        touching = any(abs(start[0] - o[0]) <= w and abs(start[1] - o[1]) <= h
                       for o in others)
        if touching:
            prev_board = before_board
            continue
        state = _read_state(
            before_board,
            previous_frame=prev_board,
            template=template,
            include_rings=False,
        )
        if state is None:
            # read the marker value directly off the board
            board = before_board
            mx, my = start[0] + marker[0], start[1] + marker[1]
            if 0 <= my < len(board) and 0 <= mx < len(board[0]):
                return int(board[my][mx])
            prev_board = before_board
            continue
        for piece in state["pieces"]:
            if tuple(piece["origin"]) == start:
                return int(piece["marker_value"])
        prev_board = before_board
    return None


def _learn_click_transfer(context: Any, marker: tuple[int, int],
                          template: tuple | None,
                          selected_value: int | None) -> bool:
    """Whether an observed click transferred control between pieces.

    ACTION6 is not assumed to mean selection.  The evidence is a click
    transition where every piece stays at the same origin while the piece
    carrying the already-learned selected marker value changes.  This keeps
    the click vocabulary frame- and history-grounded rather than ka59-coded.
    """
    if selected_value is None:
        return False
    boards = _history_boards(context)
    previous_board = None
    for (before_board, _a0), (after_board, action) in zip(boards, boards[1:]):
        if action != "ACTION6":
            previous_board = before_board
            continue
        before = _read_state(
            before_board,
            previous_frame=previous_board,
            template=template,
            selected_value=selected_value,
            include_rings=False,
        )
        after = _read_state(
            after_board,
            previous_frame=before_board,
            template=template,
            selected_value=selected_value,
            include_rings=False,
        )
        previous_board = before_board
        if before is None or after is None:
            continue
        before_pieces = before.get("pieces") or []
        after_pieces = after.get("pieces") or []
        before_origins = sorted(tuple(p["origin"]) for p in before_pieces)
        after_origins = sorted(tuple(p["origin"]) for p in after_pieces)
        if before_origins != after_origins:
            continue
        before_controlled = tuple(
            before_pieces[int(before["controlled_index"])]["origin"]
        )
        after_controlled = tuple(
            after_pieces[int(after["controlled_index"])]["origin"]
        )
        if before_controlled != after_controlled:
            return True
    return False


def _control_family_size(board: list[list[int]], marker: tuple[int, int],
                         template: tuple | None):
    if template is not None:
        body, tmarker, _count = template
        if tmarker == marker and body:
            offsets = [off for off, _v in body]
            return (max(ox for ox, _oy in offsets) + 1,
                    max(oy for _ox, oy in offsets) + 1)
    return None


def _learn_dynamics(context: Any, marker: tuple[int, int]) -> dict[str, Any]:
    """Fit what a move DOES, from observed transitions only.

    Mirrors the parallel session's `fit_controlled_pieces`: an action
    that moved only the controlled piece teaches `moves[action]` and
    which colours the piece traversed (`passable`); an action that
    moved a DIFFERENT piece while the controlled one stayed teaches
    `pushes[action]`. Nothing is assumed — an action with no evidence
    stays unknown, and unknown actions are never proposed.
    """
    moves: dict[str, list[int]] = {}
    pushes: dict[str, list[int]] = {}
    passable: set[int] = set()
    blocked: set[str] = set()
    grounded_moves: set[str] = set()
    boards = _history_boards(context)
    template = _template_from_history(context)
    prev_board = None
    for (before_board, _a0), (after_board, action) in zip(boards, boards[1:]):
        if action not in _MOVE_DELTAS:
            continue
        # Geometry is enough to learn dynamics and does not need the
        # selection resolved, so read ORIGINS with both readers (the
        # component reader loses the pieces the moment they touch).
        ob = _piece_origins(before_board, marker, template)
        oa = _piece_origins(after_board, marker, template)
        if ob is None or oa is None or len(ob) != len(oa):
            prev_board = before_board
            continue
        ob = sorted(ob)
        oa_sorted = sorted(oa)
        deltas = [(b[0] - a[0], b[1] - a[1])
                  for a, b in zip(ob, oa_sorted)]
        movedi = [i for i, d in enumerate(deltas) if d != (0, 0)]
        # Chain the window: each board's selection is resolved from its
        # OWN predecessor, which is the only way a 2-piece board can
        # say which piece the keys drive. Without the chain every
        # `before` is unresolved and a push is indistinguishable from a
        # move (measured: pushes stayed empty while moves learned).
        before = _read_state(
            before_board,
            previous_frame=prev_board,
            template=template,
            include_rings=False,
        )
        after = None
        if before is None:
            if not movedi:
                blocked.add(action)
            elif len(movedi) == 1:
                # exactly one piece moved: without a resolved selection
                # this is either a move or a push, so record it as the
                # weaker fact (a displacement of SOME piece) only when
                # no stronger reading exists for this action.
                moves.setdefault(action, list(deltas[movedi[0]]))
            prev_board = before_board
            continue
        ctl = int(before["controlled_index"])
        if not movedi:
            blocked.add(action)
        elif movedi == [ctl]:
            moves.setdefault(action, list(deltas[ctl]))
            grounded_moves.add(action)
            w, h = before["piece_size"]
            ox, oy = ob[ctl]
            dx, dy = deltas[ctl]
            for yy in range(h):
                for xx in range(w):
                    cx, cy = ox + dx + xx, oy + dy + yy
                    if 0 <= cy < len(before_board) and 0 <= cx < len(before_board[0]):
                        passable.add(int(before_board[cy][cx]))
        elif ctl not in movedi and len(movedi) == 1:
            pushes.setdefault(action, list(deltas[movedi[0]]))
        prev_board = before_board
    # Cardinal action semantics are fixed by the engine; what this family has
    # to learn is the piece stride.  Requiring a separate natural observation
    # for every direction starves short levels: ka59 measured RIGHT exactly,
    # then the family abstained before the immediately-needed UP because that
    # direction had not yet appeared in history.  Complete the other cardinal
    # deltas only from a grounded controlled-piece move, and only when every
    # such observation agrees on one positive stride.  Weak "some piece
    # moved" evidence and pushes cannot seed this completion.
    grounded_strides: list[int] = []
    for action in sorted(grounded_moves):
        dx, dy = moves[action]
        ux, uy = _MOVE_DELTAS[action]
        stride = abs(dx) + abs(dy)
        if (stride <= 0
                or (dx, dy) != (ux * stride, uy * stride)):
            grounded_strides = []
            break
        grounded_strides.append(stride)
    inferred_moves: list[str] = []
    if grounded_strides and len(set(grounded_strides)) == 1:
        stride = grounded_strides[0]
        for action, (ux, uy) in _MOVE_DELTAS.items():
            if action not in moves:
                moves[action] = [ux * stride, uy * stride]
                inferred_moves.append(action)
    return {"moves": moves, "pushes": pushes,
            "passable": sorted(passable), "blocked": sorted(blocked),
            "inferred_moves": inferred_moves}


def _control_family_any(board: list[list[int]]) -> list[tuple[int, int]] | None:
    """Piece origins under any background hypothesis (geometry only)."""
    for background in _background_variants(board):
        found = _control_family(board, background)
        if found is None:
            continue
        _body, _marker, _count, members, _sel, _amb = found
        return [tuple(m["origin"]) for m in members]
    return None


def _step_for(state: Mapping[str, Any]) -> int:
    """The measured stride when one exists, else a bounded guess.

    A guess is only used before any move has been observed; once the
    episode has produced one transition the stride is a measurement,
    so the prediction the lease checks is exact.
    """
    observed = state.get("observed_stride")
    if observed:
        return max(1, int(observed))
    pieces = state.get("pieces") or []
    return max(1, int(state.get("pitch") or 1) // max(1, len(pieces) or 1))


def _housing_cost(state: Mapping[str, Any]) -> int:
    """Distance to the WIN, which is containment — not separation.

    CORRECTED 2026-08-17 after reading the parallel session's ka59
    record: the win is "every piece inside its own ring", and their
    earlier arc measured one piece fully housed at score 0 because the
    OTHER piece was still out. A piece-to-piece objective therefore
    points at the wrong target no matter how good the dynamics model
    is. When no ring is visible the module has no objective to offer
    and says so (cost 0 => propose_plan finds no improving move and
    abstains) rather than optimising something incidental.
    """
    pieces = state.get("pieces") or []
    rings = state.get("rings") or []
    if not pieces or not rings:
        return 0
    return _containment_cost(pieces, rings)


def _move_destination_supported(
    board: list[list[int]] | None,
    state: Mapping[str, Any],
    piece_index: int,
    delta: tuple[int, int] | list[int],
    dynamics: Mapping[str, Any],
) -> bool:
    """Whether a measured-stride move has grounded traversable cells.

    A learned stride does not imply that every canvas colour is passable.
    Plain moves may enter colours observed under an earlier successful move,
    plus the visible target-ring cells.  Entering another piece remains a
    push candidate: ka59's push keeps the mover fixed and may cross walls.
    """
    if board is None or not board or not board[0]:
        return False
    pieces = list(state.get("pieces") or [])
    if not 0 <= int(piece_index) < len(pieces):
        return False
    piece = pieces[int(piece_index)]
    ox, oy = (int(value) for value in piece["origin"])
    width, height = (int(value) for value in piece["size"])
    dx, dy = (int(value) for value in delta)
    target = (ox + dx, oy + dy)
    board_height, board_width = len(board), len(board[0])
    if (target[0] < 0 or target[1] < 0
            or target[0] + width > board_width
            or target[1] + height > board_height):
        return False

    def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return abs(a[0] - b[0]) < width and abs(a[1] - b[1]) < height

    other_origins = [
        tuple(int(value) for value in other["origin"])
        for index, other in enumerate(pieces)
        if index != int(piece_index)
    ]
    if any(_overlaps(target, other) for other in other_origins):
        return True

    allowed = {int(value) for value in dynamics.get("passable") or ()}
    for ring in state.get("rings") or ():
        for key in ("frame_colour", "hole_colour"):
            value = ring.get(key)
            if value is not None:
                allowed.add(int(value))
    old_cells = {
        (ox + local_x, oy + local_y)
        for local_y in range(height)
        for local_x in range(width)
    }
    for local_y in range(height):
        for local_x in range(width):
            cell = (target[0] + local_x, target[1] + local_y)
            if cell in old_cells:
                continue
            if int(board[cell[1]][cell[0]]) not in allowed:
                return False
    return True


def _prediction_from_precomputed(
    state: Mapping[str, Any],
    action: Any,
    *,
    board: list[list[int]] | None,
    template: tuple | None,
    dynamics: Mapping[str, Any],
    click_transfer: bool,
) -> dict[str, Any]:
    """Build the exact verifier prediction without rescanning history."""
    name = _engine_action(action)
    pieces = state["pieces"]
    idx = int(state["controlled_index"])
    if name == "ACTION6":
        if not click_transfer:
            return _json({
                "schema_version": "tufa.control_marker.prediction.v1",
                "mechanic_id": FAMILY_ID,
                "family_id": FAMILY_ID,
                "accepted": False,
                "reason": "no_observed_control_transfer",
            })
        if not isinstance(action, Mapping):
            return _json({
                "schema_version": "tufa.control_marker.prediction.v1",
                "mechanic_id": FAMILY_ID,
                "family_id": FAMILY_ID,
                "accepted": False,
                "reason": "missing_click_payload",
            })
        try:
            click_x = int(action.get("x", action.get("col")))
            click_y = int(action.get("y", action.get("row")))
        except (TypeError, ValueError):
            return _json({
                "schema_version": "tufa.control_marker.prediction.v1",
                "mechanic_id": FAMILY_ID,
                "family_id": FAMILY_ID,
                "accepted": False,
                "reason": "missing_click_payload",
            })
        targets = []
        for piece_index, piece in enumerate(pieces):
            ox, oy = piece["origin"]
            width, height = piece["size"]
            if ox <= click_x < ox + width and oy <= click_y < oy + height:
                targets.append(piece_index)
        if len(targets) != 1 or targets[0] == idx:
            return _json({
                "schema_version": "tufa.control_marker.prediction.v1",
                "mechanic_id": FAMILY_ID,
                "family_id": FAMILY_ID,
                "accepted": False,
                "reason": (
                    "click_does_not_name_one_other_piece"
                    if len(targets) != 1
                    else "piece_already_controlled"
                ),
            })
        target_index = targets[0]
        return _json({
            "schema_version": "tufa.control_marker.prediction.v1",
            "mechanic_id": FAMILY_ID,
            "family_id": FAMILY_ID,
            "accepted": True,
            "expected_change": {
                "controlled_origin": pieces[target_index]["origin"],
                "marker_offset": state["marker_offset"],
                "selected_marker_value": state["selected_marker_value"],
                "selection_transfer": True,
                "piece_template": _template_payload(template),
            },
            "reason": "observed click transfer to the named piece",
        })
    if name not in _MOVE_DELTAS:
        return _json({
            "schema_version": "tufa.control_marker.prediction.v1",
            "mechanic_id": FAMILY_ID,
            "family_id": FAMILY_ID,
            "accepted": False,
            "reason": "unbound_action",
        })
    moves = dynamics.get("moves") or {}
    pushes = dynamics.get("pushes") or {}
    if name not in moves:
        return _json({
            "schema_version": "tufa.control_marker.prediction.v1",
            "mechanic_id": FAMILY_ID,
            "family_id": FAMILY_ID,
            "accepted": False,
            "reason": "no_observed_displacement_for_action",
        })
    ox, oy = pieces[idx]["origin"]
    dx, dy = moves[name]
    w, h = state["piece_size"]
    target = (ox + dx, oy + dy)

    if not _move_destination_supported(
        board, state, idx, (dx, dy), dynamics
    ):
        return _json({
            "schema_version": "tufa.control_marker.prediction.v1",
            "mechanic_id": FAMILY_ID,
            "family_id": FAMILY_ID,
            "accepted": False,
            "reason": "destination_not_grounded_passable",
        })

    def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return abs(a[0] - b[0]) < w and abs(a[1] - b[1]) < h

    others = [tuple(p["origin"]) for i, p in enumerate(pieces) if i != idx]
    pushing = any(_overlaps(target, other) for other in others)
    if pushing:
        return _json({
            "schema_version": "tufa.control_marker.prediction.v1",
            "mechanic_id": FAMILY_ID,
            "family_id": FAMILY_ID,
            "accepted": True,
            "expected_change": {
                "controlled_origin": [ox, oy],
                "marker_offset": state["marker_offset"],
                "selected_marker_value": state["selected_marker_value"],
                "piece_template": _template_payload(template),
            },
            "push_predicted": True,
            "push_delta": pushes.get(name),
            "reason": "entering another piece: the mover stays",
        })
    return _json({
        "schema_version": "tufa.control_marker.prediction.v1",
        "mechanic_id": FAMILY_ID,
        "family_id": FAMILY_ID,
        "accepted": True,
        "expected_change": {
            "controlled_origin": [target[0], target[1]],
            "marker_offset": state["marker_offset"],
            "selected_marker_value": state["selected_marker_value"],
            "piece_template": _template_payload(template),
        },
        "push_predicted": False,
        "reason": (
            "measured shared cardinal stride, destination clear of pieces"
            if name in set(dynamics.get("inferred_moves") or ())
            else "observed displacement, destination clear of pieces"
        ),
    })


def propose_plan(
    frame: Any,
    valid_actions: Iterable[Any] | None = None,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One move, or one learned selection click, toward containment.

    Deliberately shallow: the push distance is a PROPOSAL verified by
    the executor's lease, so a single certified step per call keeps a
    wrong pitch guess cheap (one revocation instead of a dead plan).  A
    click is proposed only after history has shown that ACTION6 transfers
    control between stationary congruent pieces.
    """
    binding = bind_current(frame, context=context)
    if not binding.get("supported"):
        return _blocked("no_unique_control_marker_family", binding)
    state = binding["state"]
    pieces = state["pieces"]
    idx = int(state["controlled_index"])
    if len(pieces) < 2:
        return _blocked("single_piece_no_target", binding)
    allowed = _allowed(valid_actions)
    marker = tuple(state["marker_offset"])
    dynamics = _learn_dynamics(context, marker)
    template = _template_from_history(context)
    board = _grid(frame)
    click_transfer = _learn_click_transfer(
        context,
        marker,
        template,
        state.get("selected_marker_value"),
    )
    moves = dynamics["moves"]
    if not moves:
        # Nothing has been observed yet, so no displacement is known.
        # Proposing a guess here is what the lease revoked on ka59
        # (measured 2026-08-17): abstain instead and let the agent
        # generate the evidence.
        return _blocked("no_observed_displacement_yet", binding)
    if not click_transfer:
        # Housing the currently controlled piece before the selection
        # mechanism is known strands the family halfway through its all-piece
        # objective and suppresses the external action that would have taught
        # the click.  Discovery therefore waits until both required controls
        # (move and transfer) have appeared in public history.
        return _blocked("no_observed_control_transfer", binding)
    best: tuple[int, int, str] | None = None
    best_key: tuple[int, bool, str, int] | None = None
    base = _housing_cost(state)
    for piece_index, piece in enumerate(pieces):
        cx, cy = piece["origin"]
        for action in _MOVE_ACTIONS:
            if action not in allowed or action not in moves:
                continue
            if action in dynamics["blocked"] and action not in moves:
                continue
            dx, dy = moves[action]
            if not _move_destination_supported(
                board, state, piece_index, (dx, dy), dynamics
            ):
                continue
            moved = [dict(p) for p in pieces]
            moved[piece_index] = {
                **moved[piece_index],
                "origin": [cx + dx, cy + dy],
            }
            total = _containment_cost(moved, state.get("rings") or [])
            key = (total, piece_index != idx, action, piece_index)
            if best_key is None or key < best_key:
                best_key = key
                best = (total, piece_index, action)
    if best is None or best[0] >= base:
        return _blocked("no_containment_reducing_move", binding)

    target_index = best[1]
    if target_index == idx:
        action_step: Any = best[2]
        objective_after = best[0]
        certificate_reason = (
            "congruent piece family with a unique drawn control marker; "
            "the observed move reduces summed ring-containment distance"
        )
    else:
        if "ACTION6" not in allowed:
            return _blocked("selection_click_not_available", binding)
        target_piece = pieces[target_index]
        tx, ty = target_piece["origin"]
        tw, th = target_piece["size"]
        action_step = {
            "action": "ACTION6",
            "x": int(tx) + int(tw) // 2,
            "y": int(ty) + int(th) // 2,
        }
        objective_after = base
        certificate_reason = (
            "an observed ACTION6 transition transferred control between "
            "stationary congruent pieces; the selected piece has an observed "
            "move that reduces summed ring-containment distance"
        )
    is_control_transfer = target_index != idx
    progress_measure = {
        "metric": "ring_containment_sum",
        "before": int(base),
        "expected_after": int(base if is_control_transfer else max(0, base - 1)),
        "strict": not is_control_transfer,
    }
    first_prediction = _prediction_from_precomputed(
        state,
        action_step,
        board=board,
        template=template,
        dynamics=dynamics,
        click_transfer=click_transfer,
    )
    return _json({
        "schema_version": "tufa.control_marker.plan.v1",
        "mechanic_id": FAMILY_ID,
        "family_id": FAMILY_ID,
        "certified": True,
        "supported": True,
        "actions": [action_step],
        "first_prediction": first_prediction,
        "progress_measure": progress_measure,
        "objective": {"kind": "ring_containment_sum",
                      "before": base, "after": objective_after,
                      "lookahead_after": best[0]},
        "certificate_reason": certificate_reason,
        "binding": binding,
    })


def predict_transition(
    frame: Any,
    action: Any,
    context: MechanicContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict a learned control transfer or the controlled displacement.

    The push is NOT predicted: its distance is unknown from one frame,
    and claiming it would make the lease revoke a correct plan. The
    honest prediction is the mover's step, and the executor's
    verification decides.
    """
    binding = bind_current(frame, context=context)
    if not binding.get("supported"):
        return _json({
            "schema_version": "tufa.control_marker.prediction.v1",
            "mechanic_id": FAMILY_ID,
            "family_id": FAMILY_ID,
            "accepted": False,
            "reason": "no_unique_control_marker_family",
        })
    state = binding["state"]
    name = _engine_action(action)
    marker = tuple(state["marker_offset"])
    template = _template_from_history(context)
    click_transfer = False
    dynamics: Mapping[str, Any] = {}
    if name == "ACTION6":
        click_transfer = _learn_click_transfer(
            context,
            marker,
            template,
            state.get("selected_marker_value"),
        )
    elif name in _MOVE_DELTAS:
        dynamics = _learn_dynamics(context, marker)
    return _prediction_from_precomputed(
        state,
        action,
        board=_grid(frame),
        template=template,
        dynamics=dynamics,
        click_transfer=click_transfer,
    )


def verify_transition(expected: dict[str, Any], current_frame: Any) -> bool:
    """Fail closed unless the controlled piece landed where predicted."""
    if not isinstance(expected, Mapping):
        return False
    change = expected.get("expected_change")
    if not isinstance(change, Mapping):
        return False
    target = change.get("controlled_origin")
    if not isinstance(target, (list, tuple)) or len(target) != 2:
        return False
    selected_value = change.get("selected_marker_value")
    try:
        selected_value = int(selected_value)
    except (TypeError, ValueError):
        return False
    template = _template_from_payload(change.get("piece_template"))
    state = _read_state(
        current_frame,
        template=template,
        selected_value=selected_value,
    )
    if state is None:
        return False
    pieces = state.get("pieces") or []
    idx = int(state.get("controlled_index") or 0)
    if idx >= len(pieces):
        return False
    if state.get("marker_offset") != change.get("marker_offset"):
        return False
    return list(pieces[idx]["origin"]) == [int(target[0]), int(target[1])]
