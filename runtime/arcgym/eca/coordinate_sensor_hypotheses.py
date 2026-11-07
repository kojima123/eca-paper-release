"""Coordinate-relation hypothesis verifier for ECA sensor packets.

The verifier treats LLM output as a narrow enum-valued relation hint.  It does
not accept action coordinates or plans from the LLM; candidate action sites are
derived from the ECA sensor packet and can then be protected-probed by ECA.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import json
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


ALLOWED_COORDINATE_HYPOTHESES = frozenset(
    {
        "absolute_board",
        "centroid_overlap",
        "object_local_offset",
        "lattice_stride",
        "reflected_coordinate",
        "relative_delta",
        "target_overlap",
        "missing_piece_position",
        "same_color_group_centroid",
    }
)

SUPPORTED_MISSING_PIECE_HYPOTHESES = frozenset(
    {
        "centroid_overlap",
        "target_overlap",
        "missing_piece_position",
        "same_color_group_centroid",
    }
)
COORDINATE_SENSOR_NAMES = frozenset(
    {
        "component_centroids",
        "component_bboxes",
        "canvas_delta",
        "target_overlap",
        "score_delta",
    }
)

DIRECT_ACTION_KEYS = frozenset(
    {
        "action",
        "actions",
        "action_name",
        "candidate_site",
        "candidate_sites",
        "click",
        "click_site",
        "execute",
        "plan",
        "site",
        "step",
        "x",
        "y",
    }
)

PACKET_ID_KEYS = frozenset({"game", "game_id", "level", "level_id"})
COORDINATE_HYPOTHESIS_SCHEMA_VERSION = "eca_coordinate_hypothesis.v1"
CoordinateHypothesisProvider = Callable[..., Any]

_R11L_GRID_SIZE = 64
_R11L_STRUCTURED_BFS_GRID_SIZE = 60
_R11L_PIECE_W = 5
_R11L_PIECE_H = 5
_R11L_CANVAS_W = 5
_R11L_CANVAS_H = 5
_R11L_JOINT_ASSIGNMENT_MAX_CANVASES = 8
_R11L_JOINT_ASSIGNMENT_MAX_TARGET_OPTIONS_PER_CANVAS = 8
_R11L_JOINT_ASSIGNMENT_MAX_SEARCH_STATES = 2000
_R11L_CANVAS_PATTERN = (
    (1, 0),
    (2, 0),
    (3, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (3, 2),
    (4, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 3),
    (4, 3),
    (1, 4),
    (2, 4),
    (3, 4),
)
_R11L_PIECE_PATTERN = (
    (2, 0),
    (1, 1),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (3, 2),
    (4, 2),
    (1, 3),
    (2, 3),
    (3, 3),
    (2, 4),
)


@dataclass(frozen=True)
class CoordinateHypothesisResult:
    accepted: bool
    posterior: str
    candidate_sites: tuple[tuple[int, int], ...] = ()
    rejected_reasons: tuple[str, ...] = ()
    direct_llm_action_blocked_count: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_sites"] = [list(site) for site in self.candidate_sites]
        return _sanitize(payload)


@dataclass(frozen=True)
class CoordinateHypothesisBridgeResult:
    """Provider bridge result before any production selector integration."""

    hypothesis: dict[str, Any] | None = None
    verification: CoordinateHypothesisResult | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _sanitize(asdict(self))


class CoordinateHypothesisBridge:
    """Call a source-swappable provider and verify its coordinate hypothesis.

    This class owns the LLM boundary for the H-axis prototype.  Providers see
    only the compact sensor packet and a JSON schema.  The bridge accepts at
    most enum-valued coordinate relation hypotheses, then delegates all action
    candidate generation to :func:`verify_coordinate_hypothesis`.
    """

    def __init__(
        self,
        provider: CoordinateHypothesisProvider | None = None,
        *,
        provider_source: str = "",
        provider_path: str = "",
    ) -> None:
        self.provider = provider
        self.provider_source = str(provider_source or ("constructor" if provider is not None else "none"))
        self.provider_path = str(provider_path or "")

    def build(
        self,
        *,
        enabled: bool,
        packet: Mapping[str, Any],
        current_state_hash: str,
        max_hypotheses: int = 3,
    ) -> CoordinateHypothesisBridgeResult:
        diagnostics: dict[str, Any] = {
            "enabled": bool(enabled),
            "provider_available": self.provider is not None,
            "provider_called": False,
            "provider_source": self.provider_source,
            "provider_path": self.provider_path,
            "schema_enforced": True,
            "accepted_schema_versions": [COORDINATE_HYPOTHESIS_SCHEMA_VERSION],
            "prompt_packet_role": "coordinate_hypothesis_generation",
            "prompt_packet_schema_version": COORDINATE_HYPOTHESIS_SCHEMA_VERSION,
            "prompt_packet_hash": "",
            "prompt_packet_audit": {},
            "raw_response_kind": "",
            "raw_response_preview": None,
            "json_response_parsed": False,
            "json_response_parse_error": "",
            "parsed_response_keys": [],
            "raw_hypothesis_count": 0,
            "coordinate_hypothesis_accepted_count": 0,
            "coordinate_hypothesis_rejected_count": 0,
            "direct_llm_action_blocked_count": 0,
            "candidate_sites_generated_from_hypothesis_count": 0,
            "rejection_reasons": {},
            "error": "",
        }
        empty_verification = CoordinateHypothesisResult(
            accepted=False,
            posterior="rejected",
            rejected_reasons=("no_hypothesis",),
            diagnostics={"candidate_generation_attempted": False},
        )
        if not enabled or self.provider is None:
            diagnostics["rejection_reasons"] = {"provider_not_called": 1}
            return CoordinateHypothesisBridgeResult(
                verification=empty_verification,
                diagnostics=diagnostics,
            )

        llm_packet = packet_for_llm(packet)
        prompt_packet = build_coordinate_hypothesis_prompt_packet(
            llm_packet,
            max_hypotheses=max_hypotheses,
        )
        prompt_audit = coordinate_hypothesis_prompt_packet_audit(prompt_packet)
        diagnostics["prompt_packet_hash"] = prompt_audit["prompt_packet_hash"]
        diagnostics["prompt_packet_audit"] = prompt_audit
        diagnostics["provider_called"] = True
        try:
            raw = self.provider(
                sensor_packet=llm_packet,
                current_state_hash=str(current_state_hash or ""),
                prompt_packet=prompt_packet,
            )
        except Exception as exc:  # pragma: no cover - defensive diagnostic path.
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            return CoordinateHypothesisBridgeResult(
                verification=empty_verification,
                diagnostics=diagnostics,
            )

        diagnostics["raw_response_preview"] = _payload_preview(raw)
        parsed = _parse_coordinate_provider_response(raw, diagnostics)
        items = _extract_coordinate_hypothesis_items(parsed)[: max(1, int(max_hypotheses or 1))]
        diagnostics["raw_hypothesis_count"] = int(len(items))
        rejection_counts: dict[str, int] = {}
        selected_hypothesis: dict[str, Any] | None = None
        selected_verification: CoordinateHypothesisResult | None = None
        last_rejected_verification: CoordinateHypothesisResult | None = None
        for item in items:
            hypothesis, reason = _coerce_coordinate_hypothesis(item)
            if hypothesis is None:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            verification = verify_coordinate_hypothesis(
                packet,
                hypothesis,
                current_state_hash=current_state_hash,
            )
            diagnostics["direct_llm_action_blocked_count"] += int(
                verification.direct_llm_action_blocked_count
            )
            if verification.candidate_sites:
                diagnostics["candidate_sites_generated_from_hypothesis_count"] += len(
                    verification.candidate_sites
                )
            if verification.accepted:
                selected_hypothesis = hypothesis
                selected_verification = verification
                diagnostics["coordinate_hypothesis_accepted_count"] += 1
                break
            last_rejected_verification = verification
            diagnostics["coordinate_hypothesis_rejected_count"] += 1
            for reason in verification.rejected_reasons:
                rejection_counts[str(reason)] = rejection_counts.get(str(reason), 0) + 1

        if selected_verification is None:
            selected_verification = last_rejected_verification or empty_verification
        diagnostics["rejection_reasons"] = dict(sorted(rejection_counts.items()))
        return CoordinateHypothesisBridgeResult(
            hypothesis=selected_hypothesis,
            verification=selected_verification,
            diagnostics=_sanitize(diagnostics),
        )


def build_r11l_l1_coordinate_sensor_packet(
    frame: Any,
    *,
    state_hash: str = "",
    step: int = 0,
    last_action: Mapping[str, Any] | None = None,
    effect: Mapping[str, Any] | None = None,
    failed_probe_history: Iterable[Mapping[str, Any]] = (),
    progressed_probe_history: Iterable[Mapping[str, Any]] = (),
    remembered_targets: Iterable[Mapping[str, Any]] = (),
    include_eca_only_context: bool = False,
    include_source_ids: bool = False,
    game_id: str = "",
    level_id: str = "",
) -> dict[str, Any]:
    """Build a compact coordinate sensor packet from an observed r11l frame.

    This helper is r11l-focused because Phase309 diagnoses the r11l L1 false
    negative, but the emitted packet is the generic schema consumed by the
    verifier: components, bboxes, centroids, recent action/effect, and sensor
    availability.  The optional game/level fields are intentionally stripped by
    :func:`packet_for_llm`.
    """

    from arcgym.perception.r11l_perception import perceive_r11l

    arr = np.asarray(frame, dtype=np.int16)
    perceived = perceive_r11l(arr)
    components: list[dict[str, Any]] = []

    def add_component(
        *,
        component_id: str,
        kind: str,
        role: str,
        pos: Sequence[int],
        size: tuple[int, int],
        color: Any = None,
        colors: Any = None,
        moved_recently: bool = False,
    ) -> None:
        x, y = int(pos[0]), int(pos[1])
        w, h = int(size[0]), int(size[1])
        row: dict[str, Any] = {
            "id": component_id,
            "kind": kind,
            "role": role,
            "bbox": [x, y, x + w - 1, y + h - 1],
            "centroid": [x + w // 2, y + h // 2],
            "moved_recently": bool(moved_recently),
        }
        if color is not None:
            row["color"] = int(color)
        clean_colors = _clean_colors(colors)
        if clean_colors:
            row["colors"] = sorted(clean_colors)
        components.append(row)

    active = perceived.get("active_piece")
    if active is not None:
        color = perceived.get("active_piece_color")
        add_component(
            component_id=f"piece:active:{int(active[0])},{int(active[1])}",
            kind="piece",
            role="active_piece",
            pos=active,
            size=(5, 5),
            color=color,
            colors={color} if color is not None else (),
            moved_recently=True,
        )
    for index, piece in enumerate(list(perceived.get("inactive_pieces") or [])):
        if not isinstance(piece, Mapping):
            continue
        pos = piece.get("pos")
        if not _is_xy(pos):
            continue
        color = piece.get("color")
        add_component(
            component_id=f"piece:inactive:{index}:{int(pos[0])},{int(pos[1])}",
            kind="piece",
            role="inactive_piece",
            pos=pos,
            size=(5, 5),
            color=color,
            colors={color} if color is not None else (),
            moved_recently=False,
        )
    for index, canvas in enumerate(list(perceived.get("canvases") or [])):
        if not isinstance(canvas, Mapping):
            continue
        pos = canvas.get("pos")
        if not _is_xy(pos):
            continue
        color = canvas.get("piece_color", canvas.get("color"))
        add_component(
            component_id=f"canvas:{index}:{int(pos[0])},{int(pos[1])}",
            kind="canvas",
            role="canvas",
            pos=pos,
            size=(5, 5),
            color=color,
            colors=canvas.get("colors", {color} if color is not None else ()),
        )
    for index, target in enumerate(list(perceived.get("targets") or [])):
        if not isinstance(target, Mapping):
            continue
        pos = target.get("pos")
        if not _is_xy(pos):
            continue
        add_component(
            component_id=f"target:{index}:{int(pos[0])},{int(pos[1])}",
            kind="target",
            role="target",
            pos=pos,
            size=(7, 7),
            color=target.get("color"),
            colors=target.get("colors"),
        )
    visible_target_color_keys = {
        tuple(_component_colors(row))
        for row in components
        if row.get("kind") == "target" and row.get("role") == "target" and _component_colors(row)
    }
    for index, target in enumerate(list(remembered_targets or [])):
        if not isinstance(target, Mapping):
            continue
        colors = _clean_colors(target.get("colors"))
        color = target.get("color")
        if not colors and color is not None:
            colors = {int(color)}
        color_key = tuple(sorted(colors))
        if not color_key or color_key in visible_target_color_keys:
            continue
        pos = target.get("pos")
        size = target.get("size", (7, 7))
        bbox = target.get("bbox")
        if _is_bbox(bbox):
            pos = (int(bbox[0]), int(bbox[1]))
            size = (int(bbox[2]) - int(bbox[0]) + 1, int(bbox[3]) - int(bbox[1]) + 1)
        if not _is_xy(pos):
            continue
        add_component(
            component_id=f"target:remembered:{index}:{int(pos[0])},{int(pos[1])}",
            kind="target",
            role="remembered_target",
            pos=pos,
            size=(int(size[0]), int(size[1])) if _is_xy(size) else (7, 7),
            color=next(iter(color_key), None),
            colors=color_key,
        )
        visible_target_color_keys.add(color_key)

    packet: dict[str, Any] = {
        "schema_version": "eca_coordinate_sensor_packet.v1",
        "state_hash": str(state_hash or _frame_hash(arr)),
        "step": int(step),
        "components": components,
        "last_action": dict(last_action or {}),
        "effect": dict(effect or {}),
        "failed_probe_history": [dict(row) for row in failed_probe_history],
        "progressed_probe_history": [dict(row) for row in progressed_probe_history],
        "sensors": {
            "component_centroids": True,
            "component_bboxes": True,
            "canvas_delta": "canvas_delta" in dict(effect or {}),
            "target_overlap": (
                "target_overlap_before" in dict(effect or {})
                and "target_overlap_after" in dict(effect or {})
            ),
            "score_delta": "score_delta" in dict(effect or {}),
        },
    }
    if include_source_ids:
        if game_id:
            packet["game_id"] = str(game_id)
        if level_id:
            packet["level_id"] = str(level_id)
    if include_eca_only_context:
        piece_obstacle_cells = sorted(
            [list(cell) for cell in perceived.get("wakneh_cells", set())],
            key=lambda item: (int(item[1]), int(item[0])),
        )
        canvas_obstacle_cells = sorted(
            [list(cell) for cell in perceived.get("defgjl_cells", set())],
            key=lambda item: (int(item[1]), int(item[0])),
        )
        packet["eca_only_context"] = {
            "obstacle_cells": piece_obstacle_cells,
            "piece_obstacle_cells": piece_obstacle_cells,
            "canvas_obstacle_cells": canvas_obstacle_cells,
            "obstacle_cell_source": "perception:wakneh_piece_obstacles",
            "canvas_obstacle_cell_source": "perception:defgjl_canvas_obstacles",
        }
    return _sanitize(packet)


def packet_for_llm(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return the packet fields allowed to cross the LLM boundary."""

    clean: dict[str, Any] = {}
    for key, value in packet.items():
        if str(key) in PACKET_ID_KEYS:
            continue
        if str(key) in {"known_scoring_move", "oracle_actions", "source_hint", "eca_only_context"}:
            continue
        clean[str(key)] = value
    return _sanitize(clean)


def build_coordinate_hypothesis_prompt_packet(
    packet: Mapping[str, Any],
    *,
    max_hypotheses: int = 3,
) -> dict[str, Any]:
    """Build the provider prompt packet for enum-only coordinate hypotheses."""

    llm_packet = packet_for_llm(packet)
    available_sensor_names = _available_coordinate_sensor_names(llm_packet)
    return _sanitize(
        {
            "role": "coordinate_hypothesis_generation",
            "expected_schema_version": COORDINATE_HYPOTHESIS_SCHEMA_VERSION,
            "system_prompt": (
                "You read compact ECA structural sensor packets. Do not ask "
                "for raw frames and do not output actions, click coordinates, "
                "or plans. Return only enum-valued coordinate relation "
                "hypotheses using the required JSON schema. Return at most "
                "one hypothesis. If the packet is insufficient, return an "
                "empty coordinate_hypotheses array. "
                "/no_think"
            ),
            "user_payload": {
                "sensor_packet": llm_packet,
                "allowed_coordinate_hypotheses": sorted(
                    SUPPORTED_MISSING_PIECE_HYPOTHESES
                ),
                "currently_supported_coordinate_hypotheses": sorted(
                    SUPPORTED_MISSING_PIECE_HYPOTHESES
                ),
                "supported_candidate_generation": ["solve_missing_piece_position"],
                "known_sensor_names": available_sensor_names,
                "instruction": (
                    "For solve_missing_piece_position, choose a supported "
                    "coordinate_hypothesis and choose required_sensors only "
                    "from known_sensor_names. A minimal sufficient sensor set "
                    "for same-color centroid reasoning is component_centroids "
                    "plus component_bboxes. Prefer the "
                    "minimum_valid_response_template values when they fit the "
                    "packet; do not invent placeholder actors or relations."
                ),
                "minimum_valid_response_template": {
                    "coordinate_hypotheses": [
                        {
                            "schema_version": COORDINATE_HYPOTHESIS_SCHEMA_VERSION,
                            "coordinate_hypothesis": "centroid_overlap",
                            "actors": ["piece_group", "canvas", "target"],
                            "relation": "canvas_centroid_tracks_piece_group_centroid",
                            "goal_test": "canvas_overlaps_target",
                            "candidate_generation": "solve_missing_piece_position",
                            "confidence": 0.5,
                            "required_sensors": ["component_centroids", "component_bboxes"],
                        }
                    ]
                },
                "forbidden_fields": sorted(DIRECT_ACTION_KEYS),
                "max_hypotheses": max(1, int(max_hypotheses or 1)),
            },
            "response_schema": coordinate_hypothesis_response_schema(
                max_hypotheses=max_hypotheses,
                sensor_names=available_sensor_names,
            ),
            "forbidden_fields": sorted(DIRECT_ACTION_KEYS),
        }
    )


def coordinate_hypothesis_response_schema(
    *,
    max_hypotheses: int = 3,
    sensor_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """JSON schema for H-axis provider responses."""

    allowed_sensors = sorted(
        {
            str(name)
            for name in (sensor_names if sensor_names is not None else COORDINATE_SENSOR_NAMES)
            if str(name) in COORDINATE_SENSOR_NAMES
        }
    )
    if not allowed_sensors:
        allowed_sensors = sorted(COORDINATE_SENSOR_NAMES)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["coordinate_hypotheses"],
        "properties": {
            "coordinate_hypotheses": {
                "type": "array",
                "maxItems": max(1, int(max_hypotheses or 1)),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "schema_version",
                        "coordinate_hypothesis",
                        "actors",
                        "relation",
                        "goal_test",
                        "candidate_generation",
                        "confidence",
                        "required_sensors",
                    ],
                    "properties": {
                        "schema_version": {
                            "type": "string",
                            "enum": [COORDINATE_HYPOTHESIS_SCHEMA_VERSION],
                        },
                        "coordinate_hypothesis": {
                            "type": "string",
                            "enum": sorted(SUPPORTED_MISSING_PIECE_HYPOTHESES),
                        },
                        "actors": {
                            "type": "array",
                            "maxItems": 3,
                            "uniqueItems": True,
                            "items": {
                                "type": "string",
                                "enum": ["canvas", "piece_group", "target"],
                            },
                        },
                        "relation": {
                            "type": "string",
                            "enum": ["canvas_centroid_tracks_piece_group_centroid"],
                        },
                        "goal_test": {
                            "type": "string",
                            "enum": ["canvas_overlaps_target"],
                        },
                        "candidate_generation": {
                            "type": "string",
                            "enum": ["solve_missing_piece_position"],
                        },
                        "confidence": {"type": "number"},
                        "required_sensors": {
                            "type": "array",
                            "maxItems": max(1, len(allowed_sensors)),
                            "uniqueItems": True,
                            "items": {
                                "type": "string",
                                "enum": allowed_sensors,
                            },
                        },
                    },
                },
            }
        },
    }


def _available_coordinate_sensor_names(packet: Mapping[str, Any]) -> list[str]:
    sensors = dict(packet.get("sensors") or {})
    names = [
        str(name)
        for name in sorted(COORDINATE_SENSOR_NAMES)
        if bool(sensors.get(str(name), False))
    ]
    return names or sorted(COORDINATE_SENSOR_NAMES)


def coordinate_hypothesis_prompt_packet_audit(packet: Mapping[str, Any]) -> dict[str, Any]:
    payload = _sanitize(packet)
    user_payload = payload.get("user_payload", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(user_payload, Mapping):
        user_payload = {}
    sensor_packet = user_payload.get("sensor_packet", {})
    if not isinstance(sensor_packet, Mapping):
        sensor_packet = {}
    packet_json = _canonical_json(payload)
    user_payload_json = _canonical_json(user_payload)
    sensor_packet_json = _canonical_json(sensor_packet)
    return {
        "prompt_packet_hash": "sha256:" + hashlib.sha256(packet_json.encode("utf-8")).hexdigest(),
        "role": str(payload.get("role", "") or ""),
        "expected_schema_version": str(payload.get("expected_schema_version", "") or ""),
        "system_prompt_chars": len(str(payload.get("system_prompt", "") or "")),
        "packet_json_chars": len(packet_json),
        "user_payload_json_chars": len(user_payload_json),
        "sensor_packet_json_chars": len(sensor_packet_json),
        "sensor_packet_keys": sorted(str(key) for key in sensor_packet.keys()),
        "raw_frame_present": _contains_key(sensor_packet, {"frame", "raw_frame", "pixels"}),
        "game_or_level_id_present": _contains_key(sensor_packet, PACKET_ID_KEYS),
        "known_scoring_move_present": _contains_key(
            sensor_packet,
            {"known_scoring_move", "oracle_actions", "source_hint"},
        ),
        "forbidden_fields": [str(item) for item in payload.get("forbidden_fields", [])],
    }


def resolve_coordinate_hypothesis_provider_path(path: str | None) -> CoordinateHypothesisProvider | None:
    normalized = str(path or "").strip()
    if not normalized:
        return None
    if ":" in normalized:
        module_name, attr_path = normalized.split(":", 1)
    else:
        module_name, sep, attr_path = normalized.rpartition(".")
        if not sep:
            raise ValueError(
                "coordinate hypothesis provider path must be 'module:callable' or 'module.callable'"
            )
    module = importlib.import_module(module_name.strip())
    target: Any = module
    for attr in attr_path.strip().split("."):
        if not attr:
            raise ValueError(f"invalid coordinate hypothesis provider attribute path: {normalized}")
        target = getattr(target, attr)
    if not callable(target):
        raise TypeError(f"coordinate hypothesis provider is not callable: {normalized}")
    return target


def verify_coordinate_hypothesis(
    packet: Mapping[str, Any],
    hypothesis: Mapping[str, Any] | None,
    *,
    current_state_hash: str,
    board_size: tuple[int, int] = (64, 64),
) -> CoordinateHypothesisResult:
    """Validate a coordinate hypothesis and generate ECA-owned candidates."""

    rejected: list[str] = []
    if not hypothesis:
        return CoordinateHypothesisResult(
            accepted=False,
            posterior="rejected",
            rejected_reasons=("no_hypothesis",),
            diagnostics={"candidate_generation_attempted": False},
        )

    direct_keys = _find_direct_action_keys(hypothesis)
    if direct_keys:
        return CoordinateHypothesisResult(
            accepted=False,
            posterior="rejected",
            rejected_reasons=("direct_coordinate_output",),
            direct_llm_action_blocked_count=1,
            diagnostics={"blocked_direct_keys": direct_keys},
        )

    packet_hash = str(packet.get("state_hash", "") or "")
    if packet_hash != str(current_state_hash or ""):
        rejected.append("stale_state_hash")

    coordinate_hypothesis = str(hypothesis.get("coordinate_hypothesis", "") or "")
    if coordinate_hypothesis not in ALLOWED_COORDINATE_HYPOTHESES:
        rejected.append("unknown_coordinate_hypothesis")

    candidate_generation = str(hypothesis.get("candidate_generation", "") or "")
    if candidate_generation != "solve_missing_piece_position":
        rejected.append("unsupported_candidate_generation")

    relation = str(hypothesis.get("relation", "") or "")
    if relation != "canvas_centroid_tracks_piece_group_centroid":
        rejected.append("unsupported_relation")

    goal_test = str(hypothesis.get("goal_test", "") or "")
    if goal_test != "canvas_overlaps_target":
        rejected.append("unsupported_goal_test")

    if coordinate_hypothesis not in SUPPORTED_MISSING_PIECE_HYPOTHESES:
        rejected.append("unsupported_relation_for_missing_piece_position")

    required_sensors = tuple(str(item) for item in hypothesis.get("required_sensors", ()) or ())
    available_sensors = dict(packet.get("sensors") or {})
    for sensor in required_sensors:
        if not bool(available_sensors.get(sensor, False)):
            rejected.append(f"missing_required_sensor:{sensor}")

    if rejected:
        return CoordinateHypothesisResult(
            accepted=False,
            posterior="rejected",
            rejected_reasons=tuple(dict.fromkeys(rejected)),
            diagnostics={"candidate_generation_attempted": False},
        )

    candidates, candidate_diagnostics = _generate_missing_piece_position_candidates(
        packet,
        board_size=board_size,
    )
    if not candidates:
        return CoordinateHypothesisResult(
            accepted=False,
            posterior="falsified",
            rejected_reasons=("no_same_color_piece_canvas_target_candidate",),
            diagnostics=candidate_diagnostics,
        )
    return CoordinateHypothesisResult(
        accepted=True,
        posterior="candidate_generated",
        candidate_sites=tuple(candidates),
        diagnostics=candidate_diagnostics,
    )


def coordinate_hypothesis_preflight(
    packet: Mapping[str, Any],
    *,
    require_candidate_sites: bool = False,
    board_size: tuple[int, int] = (64, 64),
) -> dict[str, Any]:
    """Check whether a packet is structurally worth sending to the provider."""

    components = [row for row in list(packet.get("components") or []) if isinstance(row, Mapping)]
    active_pieces = [row for row in components if row.get("kind") == "piece" and row.get("role") == "active_piece"]
    pieces = [row for row in components if row.get("kind") == "piece"]
    canvases = [row for row in components if row.get("kind") == "canvas"]
    targets = [row for row in components if row.get("kind") == "target"]

    active_with_same_color_fixed_piece = 0
    same_color_canvas_matches = 0
    same_color_target_matches = 0
    triples_evaluated = 0
    for active in active_pieces:
        active_colors = _component_colors(active)
        if not active_colors or _bbox(active) is None:
            continue
        fixed_centroids = [
            _centroid(piece)
            for piece in pieces
            if str(piece.get("id", "")) != str(active.get("id", ""))
            and _colors_overlap(active_colors, _component_colors(piece))
        ]
        fixed_centroids = [center for center in fixed_centroids if center is not None]
        if not fixed_centroids:
            continue
        active_with_same_color_fixed_piece += 1
        matching_canvases = [
            canvas
            for canvas in canvases
            if _bbox(canvas) is not None and _colors_overlap(active_colors, _component_colors(canvas))
        ]
        matching_targets = [
            target
            for target in targets
            if _bbox(target) is not None and _colors_overlap(active_colors, _component_colors(target))
        ]
        same_color_canvas_matches += len(matching_canvases)
        same_color_target_matches += len(matching_targets)
        triples_evaluated += len(matching_canvases) * len(matching_targets)

    diagnostics: dict[str, Any] = {
        "enabled": True,
        "passed": bool(triples_evaluated > 0),
        "block_reason": "" if triples_evaluated > 0 else "no_same_color_piece_canvas_target_precondition",
        "active_piece_count": int(len(active_pieces)),
        "piece_count": int(len(pieces)),
        "canvas_count": int(len(canvases)),
        "target_count": int(len(targets)),
        "active_with_same_color_fixed_piece_count": int(active_with_same_color_fixed_piece),
        "same_color_canvas_match_count": int(same_color_canvas_matches),
        "same_color_target_match_count": int(same_color_target_matches),
        "same_color_piece_canvas_target_triples_evaluated": int(triples_evaluated),
        "require_candidate_sites": bool(require_candidate_sites),
    }
    if not diagnostics["passed"]:
        return diagnostics

    if require_candidate_sites:
        candidates, candidate_diagnostics = _generate_missing_piece_position_candidates(
            packet,
            board_size=board_size,
            allow_structured_bfs=False,
        )
        diagnostics["candidate_sites_generated_from_preflight_count"] = int(len(candidates))
        diagnostics["candidate_sites"] = [[int(site[0]), int(site[1])] for site in candidates]
        diagnostics["candidate_generation_preflight"] = candidate_diagnostics
        if not candidates:
            diagnostics["passed"] = False
            diagnostics["block_reason"] = "no_candidate_sites_from_coordinate_preflight"
    return diagnostics


def _generate_missing_piece_position_candidates(
    packet: Mapping[str, Any],
    *,
    board_size: tuple[int, int],
    allow_structured_bfs: bool = True,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    components = [row for row in list(packet.get("components") or []) if isinstance(row, Mapping)]
    eca_context = dict(packet.get("eca_only_context") or {})
    piece_obstacle_cells = _cells_from_context(
        eca_context,
        "piece_obstacle_cells",
        fallback_key="obstacle_cells",
    )
    canvas_obstacle_cells = _cells_from_context(eca_context, "canvas_obstacle_cells")
    group_focus = dict(eca_context.get("coordinate_group_focus") or {})
    target_preselection = dict(eca_context.get("coordinate_target_preselection") or {})
    immediate_flood_guard = dict(eca_context.get("coordinate_immediate_candidate_flood_guard") or {})
    allow_target_preselection = bool(target_preselection.get("enabled"))
    flood_guard_enabled = bool(immediate_flood_guard.get("enabled"))
    locked_satisfied_canvas_ids = {
        str(value or "")
        for value in list(group_focus.get("locked_satisfied_canvas_ids") or [])
        if str(value or "")
    }
    terminal_immediate_target_diversity = bool(
        group_focus.get("terminal_immediate_target_diversity_enabled")
    )
    terminal_target_color_coverage_rank = bool(
        group_focus.get("terminal_target_color_coverage_rank_enabled")
    )
    terminal_immediate_move_click_empty_guard = bool(
        group_focus.get("terminal_immediate_move_click_empty_guard_enabled")
    )
    post_score1_score_bearing_immediate_guard = bool(
        group_focus.get("post_score1_score_bearing_immediate_guard_enabled")
    )
    post_score1_stale_target_keys = [
        str(value or "")
        for value in list(group_focus.get("post_score1_score_bearing_stale_target_keys") or [])
        if str(value or "")
    ]
    preferred_target_keys = [
        str(value or "")
        for value in list(group_focus.get("preferred_target_keys") or [])
        if str(value or "")
    ]
    suppressed_target_keys = [
        str(value or "")
        for value in list(group_focus.get("suppressed_target_keys") or [])
        if str(value or "")
    ]
    try:
        flood_guard_threshold = max(1, int(immediate_flood_guard.get("threshold", 64) or 64))
    except (TypeError, ValueError):
        flood_guard_threshold = 64
    active_pieces = [row for row in components if row.get("kind") == "piece" and row.get("role") == "active_piece"]
    pieces = [row for row in components if row.get("kind") == "piece"]
    canvases = [row for row in components if row.get("kind") == "canvas"]
    targets = [row for row in components if row.get("kind") == "target"]
    remaining_target_recanonicalization = bool(
        group_focus.get("remaining_target_recanonicalization_enabled")
    )
    try:
        remaining_target_recanonicalization_max_distance = max(
            0,
            int(group_focus.get("remaining_target_recanonicalization_max_distance", 8) or 8),
        )
    except (TypeError, ValueError):
        remaining_target_recanonicalization_max_distance = 8
    recanonicalization_diag = {
        "enabled": bool(remaining_target_recanonicalization),
        "max_centroid_distance": int(remaining_target_recanonicalization_max_distance),
        "preferred_remap_count": 0,
        "suppressed_remap_count": 0,
    }
    if remaining_target_recanonicalization:
        preferred_target_keys, preferred_recanonicalization_diag = (
            _canonicalize_target_focus_keys_to_current_targets(
                preferred_target_keys,
                targets,
                max_centroid_distance=remaining_target_recanonicalization_max_distance,
            )
        )
        suppressed_target_keys, suppressed_recanonicalization_diag = (
            _canonicalize_target_focus_keys_to_current_targets(
                suppressed_target_keys,
                targets,
                max_centroid_distance=remaining_target_recanonicalization_max_distance,
            )
        )
        recanonicalization_diag = {
            **recanonicalization_diag,
            "preferred": preferred_recanonicalization_diag,
            "suppressed": suppressed_recanonicalization_diag,
            "preferred_remap_count": int(
                preferred_recanonicalization_diag.get("remap_count", 0) or 0
            ),
            "suppressed_remap_count": int(
                suppressed_recanonicalization_diag.get("remap_count", 0) or 0
            ),
        }
    board_w, board_h = int(board_size[0]), int(board_size[1])
    candidate_scores: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    candidate_metadata: dict[tuple[int, int], dict[str, Any]] = {}
    triples_evaluated = 0
    locked_satisfied_canvas_immediate_skip_count = 0
    terminal_immediate_suppressed_target_skip_count = 0
    terminal_immediate_non_preferred_target_skip_count = 0
    terminal_immediate_move_click_occupied_skip_count = 0
    post_score1_score_bearing_immediate_skip_count = 0
    all_piece_tops = [
        top for top in (_bbox_top_left_from_component(piece) for piece in pieces)
        if top is not None
    ]

    for active in active_pieces:
        active_colors = _component_colors(active)
        active_bbox = _bbox(active)
        if active_bbox is None:
            continue
        active_w = active_bbox[2] - active_bbox[0] + 1
        active_h = active_bbox[3] - active_bbox[1] + 1
        for canvas in canvases:
            if str(canvas.get("id", "") or "") in locked_satisfied_canvas_ids:
                locked_satisfied_canvas_immediate_skip_count += 1
                continue
            canvas_colors = _component_colors(canvas)
            if not _colors_overlap(active_colors, canvas_colors):
                continue
            canvas_bbox = _bbox(canvas)
            if canvas_bbox is None:
                continue
            canvas_w = canvas_bbox[2] - canvas_bbox[0] + 1
            canvas_h = canvas_bbox[3] - canvas_bbox[1] + 1
            canvas_centroid = (
                (canvas_bbox[0] + canvas_bbox[2]) // 2,
                (canvas_bbox[1] + canvas_bbox[3]) // 2,
            )
            canvas_target_relations: list[dict[str, Any]] = []
            for relation_target in targets:
                relation_target_bbox = _bbox(relation_target)
                if relation_target_bbox is None:
                    continue
                relation_target_colors = _component_colors(relation_target)
                if not _colors_overlap(canvas_colors, relation_target_colors):
                    continue
                relation_overlap = _bbox_overlap_area(canvas_bbox, relation_target_bbox)
                relation_key = _target_focus_key(relation_target)
                canvas_target_relations.append(
                    {
                        "target_focus_key": str(relation_key or ""),
                        "target_bbox": relation_target_bbox,
                        "current_overlap_area": int(relation_overlap),
                    }
                )
            current_canvas_satisfied_pair_count = sum(
                1
                for relation in canvas_target_relations
                if int(relation.get("current_overlap_area", 0) or 0) > 0
            )
            current_canvas_overlap_area_sum = sum(
                int(relation.get("current_overlap_area", 0) or 0)
                for relation in canvas_target_relations
            )
            group_pieces = [
                piece
                for piece in pieces
                if _colors_overlap(active_colors, _component_colors(piece))
            ]
            fixed_centroids = [
                _centroid(piece)
                for piece in group_pieces
                if str(piece.get("id", "")) != str(active.get("id", ""))
            ]
            fixed_centroids = [center for center in fixed_centroids if center is not None]
            if not fixed_centroids:
                continue
            piece_count = len(fixed_centroids) + 1
            for target in targets:
                if not _colors_overlap(active_colors, _component_colors(target)):
                    continue
                target_bbox = _bbox(target)
                if target_bbox is None:
                    continue
                target_key = _target_focus_key(target)
                if (
                    terminal_immediate_target_diversity
                    and target_key
                    and _target_focus_key_matches_any(target_key, suppressed_target_keys)
                ):
                    terminal_immediate_suppressed_target_skip_count += 1
                    continue
                if (
                    terminal_immediate_target_diversity
                    and preferred_target_keys
                    and target_key
                    and _target_focus_key_index(target_key, preferred_target_keys) is None
                ):
                    terminal_immediate_non_preferred_target_skip_count += 1
                    continue
                target_colors = _component_colors(target)
                target_color_coverage_count = int(len(canvas_colors & target_colors))
                target_color_missing_count = int(len(target_colors - canvas_colors))
                triples_evaluated += 1
                for y in range(board_h):
                    top_y = y - active_h // 2
                    if top_y < 0 or top_y + active_h > board_h:
                        continue
                    for x in range(board_w):
                        top_x = x - active_w // 2
                        if top_x < 0 or top_x + active_w > board_w:
                            continue
                        if piece_obstacle_cells and _rect_touches_cells(
                            top_x,
                            top_y,
                            active_w,
                            active_h,
                            piece_obstacle_cells,
                        ):
                            continue
                        sum_x = sum(center[0] for center in fixed_centroids) + x
                        sum_y = sum(center[1] for center in fixed_centroids) + y
                        canvas_x = sum_x // piece_count - canvas_w // 2
                        canvas_y = sum_y // piece_count - canvas_h // 2
                        projected_canvas = (
                            canvas_x,
                            canvas_y,
                            canvas_x + canvas_w - 1,
                            canvas_y + canvas_h - 1,
                        )
                        overlap_area = _bbox_overlap_area(projected_canvas, target_bbox)
                        if overlap_area > 0:
                            site = (int(x), int(y))
                            if (
                                terminal_immediate_move_click_empty_guard
                                and _click_inside_piece_tops(site, all_piece_tops)
                            ):
                                terminal_immediate_move_click_occupied_skip_count += 1
                                continue
                            target_center = (
                                (target_bbox[0] + target_bbox[2]) // 2,
                                (target_bbox[1] + target_bbox[3]) // 2,
                            )
                            projected_canvas_centroid = (
                                (projected_canvas[0] + projected_canvas[2]) // 2,
                                (projected_canvas[1] + projected_canvas[3]) // 2,
                            )
                            projected_satisfied_target_keys: list[str] = []
                            projected_canvas_satisfied_pair_count = 0
                            projected_canvas_overlap_area_sum = 0
                            for relation in canvas_target_relations:
                                relation_bbox = relation.get("target_bbox")
                                if not isinstance(relation_bbox, tuple) or len(relation_bbox) != 4:
                                    continue
                                relation_projected_overlap = _bbox_overlap_area(
                                    projected_canvas,
                                    relation_bbox,
                                )
                                projected_canvas_overlap_area_sum += int(
                                    relation_projected_overlap
                                )
                                if relation_projected_overlap > 0:
                                    projected_canvas_satisfied_pair_count += 1
                                    relation_key = str(
                                        relation.get("target_focus_key", "") or ""
                                    )
                                    if relation_key:
                                        projected_satisfied_target_keys.append(relation_key)
                            current_overlap_area = _bbox_overlap_area(
                                canvas_bbox,
                                target_bbox,
                            )
                            predicted_requested_target_satisfied_delta = int(
                                overlap_area > 0
                            ) - int(current_overlap_area > 0)
                            predicted_canvas_satisfied_pair_delta = int(
                                projected_canvas_satisfied_pair_count
                                - current_canvas_satisfied_pair_count
                            )
                            predicted_canvas_overlap_area_delta = int(
                                projected_canvas_overlap_area_sum
                                - current_canvas_overlap_area_sum
                            )
                            if (
                                post_score1_score_bearing_immediate_guard
                                and target_key
                                and _target_focus_key_matches_any(
                                    target_key,
                                    post_score1_stale_target_keys,
                                )
                                and predicted_requested_target_satisfied_delta <= 0
                                and predicted_canvas_satisfied_pair_delta <= 0
                            ):
                                post_score1_score_bearing_immediate_skip_count += 1
                                continue
                            distance = abs(int(x) - target_center[0]) + abs(int(y) - target_center[1])
                            previous = candidate_scores.get(site)
                            if terminal_target_color_coverage_rank:
                                score = (
                                    int(overlap_area),
                                    int(target_color_coverage_count),
                                    -int(target_color_missing_count),
                                    -int(distance),
                                )
                            else:
                                score = (int(overlap_area), 0, 0, -int(distance))
                            if previous is None or score > previous:
                                candidate_scores[site] = score
                                candidate_metadata[site] = {
                                    "source": "immediate_missing_piece_position",
                                    "site": [int(site[0]), int(site[1])],
                                    "canvas_id": str(canvas.get("id", "") or ""),
                                    "target_id": str(target.get("id", "") or ""),
                                    "target_focus_key": str(target_key or ""),
                                    "overlap_area": int(overlap_area),
                                    "distance": int(distance),
                                    "current_canvas_bbox": [
                                        int(canvas_bbox[0]),
                                        int(canvas_bbox[1]),
                                        int(canvas_bbox[2]),
                                        int(canvas_bbox[3]),
                                    ],
                                    "current_canvas_centroid": [
                                        int(canvas_centroid[0]),
                                        int(canvas_centroid[1]),
                                    ],
                                    "projected_canvas_bbox": [
                                        int(projected_canvas[0]),
                                        int(projected_canvas[1]),
                                        int(projected_canvas[2]),
                                        int(projected_canvas[3]),
                                    ],
                                    "projected_canvas_centroid": [
                                        int(projected_canvas_centroid[0]),
                                        int(projected_canvas_centroid[1]),
                                    ],
                                    "target_bbox": [
                                        int(target_bbox[0]),
                                        int(target_bbox[1]),
                                        int(target_bbox[2]),
                                        int(target_bbox[3]),
                                    ],
                                    "target_centroid": [
                                        int(target_center[0]),
                                        int(target_center[1]),
                                    ],
                                    "current_overlap_area": int(current_overlap_area),
                                    "predicted_requested_target_satisfied_delta": int(
                                        predicted_requested_target_satisfied_delta
                                    ),
                                    "current_canvas_satisfied_pair_count": int(
                                        current_canvas_satisfied_pair_count
                                    ),
                                    "projected_canvas_satisfied_pair_count": int(
                                        projected_canvas_satisfied_pair_count
                                    ),
                                    "predicted_canvas_satisfied_pair_delta": int(
                                        predicted_canvas_satisfied_pair_delta
                                    ),
                                    "current_canvas_overlap_area_sum": int(
                                        current_canvas_overlap_area_sum
                                    ),
                                    "projected_canvas_overlap_area_sum": int(
                                        projected_canvas_overlap_area_sum
                                    ),
                                    "predicted_canvas_overlap_area_delta": int(
                                        predicted_canvas_overlap_area_delta
                                    ),
                                    "projected_satisfied_target_keys": list(
                                        dict.fromkeys(projected_satisfied_target_keys)
                                    )[:8],
                                    "target_color_coverage_count": int(
                                        target_color_coverage_count
                                    ),
                                    "target_color_missing_count": int(
                                        target_color_missing_count
                                    ),
                                    "target_color_complete": bool(
                                        target_colors and target_colors.issubset(canvas_colors)
                                    ),
                                }

    immediate_ordered = sorted(
        candidate_scores,
        key=lambda site: (
            -candidate_scores[site][0],
            -candidate_scores[site][1],
            candidate_scores[site][2],
            candidate_scores[site][3],
            site[1],
            site[0],
        ),
    )
    flood_guard_triggered = bool(
        flood_guard_enabled and len(immediate_ordered) >= flood_guard_threshold
    )
    if immediate_ordered and not flood_guard_triggered:
        structured_ordered: list[tuple[int, int]] = []
        structured_diagnostics: dict[str, Any] = {
            "attempted": False,
            "skip_reason": "immediate_candidates_available",
        }
    else:
        structured_ordered, structured_diagnostics = _generate_structured_group_completion_candidates(
            components,
            piece_obstacle_cells=piece_obstacle_cells,
            canvas_obstacle_cells=canvas_obstacle_cells,
            board_size=board_size,
            allow_structured_bfs=allow_structured_bfs,
            group_focus=group_focus,
            allow_target_preselection=allow_target_preselection,
        )
    flood_guard_applied = bool(flood_guard_triggered and structured_ordered)
    if flood_guard_applied:
        ordered = list(structured_ordered)
        for site in immediate_ordered:
            if site not in ordered:
                ordered.append(site)
    else:
        ordered = list(immediate_ordered)
    for site in structured_ordered:
        if site not in ordered:
            ordered.append(site)
    diagnostics = {
        "candidate_generation_attempted": True,
        "same_color_piece_canvas_target_triples_evaluated": int(triples_evaluated),
        "candidate_sites_generated_from_hypothesis_count": int(len(ordered)),
        "immediate_candidate_sites_generated_count": int(len(immediate_ordered)),
        "structured_candidate_sites_generated_count": int(len(structured_ordered)),
        "immediate_candidate_site_metadata": [
            dict(candidate_metadata.get(site) or {"site": [int(site[0]), int(site[1])]})
            for site in immediate_ordered
        ],
        "eca_only_obstacle_cell_count": int(len(piece_obstacle_cells)),
        "piece_obstacle_cell_count": int(len(piece_obstacle_cells)),
        "canvas_obstacle_cell_count": int(len(canvas_obstacle_cells)),
        "coordinate_group_focus": _coordinate_group_focus_diagnostics(group_focus),
        "locked_satisfied_canvas_immediate_skip_count": int(
            locked_satisfied_canvas_immediate_skip_count
        ),
        "terminal_immediate_target_diversity_enabled": bool(
            terminal_immediate_target_diversity
        ),
        "terminal_target_color_coverage_rank_enabled": bool(
            terminal_target_color_coverage_rank
        ),
        "terminal_immediate_move_click_empty_guard_enabled": bool(
            terminal_immediate_move_click_empty_guard
        ),
        "terminal_immediate_move_click_occupied_skip_count": int(
            terminal_immediate_move_click_occupied_skip_count
        ),
        "post_score1_score_bearing_immediate_guard_enabled": bool(
            post_score1_score_bearing_immediate_guard
        ),
        "post_score1_score_bearing_stale_target_key_count": int(
            len(post_score1_stale_target_keys)
        ),
        "post_score1_score_bearing_immediate_skip_count": int(
            post_score1_score_bearing_immediate_skip_count
        ),
        "remaining_target_recanonicalization": recanonicalization_diag,
        "terminal_immediate_suppressed_target_skip_count": int(
            terminal_immediate_suppressed_target_skip_count
        ),
        "terminal_immediate_non_preferred_target_skip_count": int(
            terminal_immediate_non_preferred_target_skip_count
        ),
        "coordinate_target_preselection": {
            "enabled": bool(allow_target_preselection),
        },
        "coordinate_immediate_candidate_flood_guard": {
            "enabled": bool(flood_guard_enabled),
            "threshold": int(flood_guard_threshold),
            "immediate_candidate_count": int(len(immediate_ordered)),
            "triggered": bool(flood_guard_triggered),
            "applied": bool(flood_guard_applied),
            "structured_candidate_count": int(len(structured_ordered)),
            "ordered_mode": "structured_first" if flood_guard_applied else "immediate_first",
            "source": str(immediate_flood_guard.get("source", "") or ""),
        },
        "structured_group_completion": structured_diagnostics,
    }
    return ordered, diagnostics


def _cells_from_context(
    context: Mapping[str, Any],
    key: str,
    *,
    fallback_key: str = "",
) -> set[tuple[int, int]]:
    raw = list(context.get(key) or [])
    if not raw and fallback_key:
        raw = list(context.get(fallback_key) or [])
    return {(int(cell[0]), int(cell[1])) for cell in raw if _is_xy(cell)}


def _coordinate_group_focus_diagnostics(group_focus: Mapping[str, Any] | None) -> dict[str, Any]:
    focus = dict(group_focus or {})
    preferred = [str(value) for value in list(focus.get("preferred_target_keys") or [])]
    suppressed = [str(value) for value in list(focus.get("suppressed_target_keys") or [])]
    detour_no_score_suppressed = [
        str(value)
        for value in list(focus.get("detour_no_score_suppressed_target_keys") or [])
    ]
    completed_no_score_suppressed = [
        str(value)
        for value in list(focus.get("completed_no_score_suppressed_target_keys") or [])
    ]
    attempted_sites = [
        value
        for value in list(focus.get("attempted_candidate_sites") or [])
        if _is_xy(value)
    ]
    locked_canvas_ids = [
        str(value) for value in list(focus.get("locked_satisfied_canvas_ids") or [])
    ]
    locked_target_keys = [
        str(value) for value in list(focus.get("locked_satisfied_target_keys") or [])
    ]
    return {
        "enabled": bool(focus.get("enabled")),
        "strict_preferred_targets": bool(focus.get("strict_preferred_targets")),
        "multi_target_assignment_sweep_enabled": bool(
            focus.get("multi_target_assignment_sweep_enabled")
        ),
        "joint_target_assignment_enabled": bool(
            focus.get("joint_target_assignment_enabled")
        ),
        "joint_target_execution_plan_enabled": bool(
            focus.get("joint_target_execution_plan_enabled")
        ),
        "joint_target_progress_attribution_enabled": bool(
            focus.get("joint_target_progress_attribution_enabled")
        ),
        "lock_aware_target_canonicalization_enabled": bool(
            focus.get("lock_aware_target_canonicalization_enabled")
        ),
        "remaining_target_recanonicalization_enabled": bool(
            focus.get("remaining_target_recanonicalization_enabled")
        ),
        "remaining_target_recanonicalization": dict(
            focus.get("remaining_target_recanonicalization") or {}
        ),
        "locked_satisfied_canvas_unsatisfied_target_retry_enabled": bool(
            focus.get("locked_satisfied_canvas_unsatisfied_target_retry_enabled")
        ),
        "terminal_no_progress_target_suppression_enabled": bool(
            focus.get("terminal_no_progress_target_suppression_enabled")
        ),
        "detour_no_score_target_suppression_enabled": bool(
            focus.get("detour_no_score_target_suppression_enabled")
        ),
        "completed_no_score_target_suppression_enabled": bool(
            focus.get("completed_no_score_target_suppression_enabled")
        ),
        "terminal_structured_preflight_fallback_enabled": bool(
            focus.get("terminal_structured_preflight_fallback_enabled")
        ),
        "terminal_attempted_site_alternatives_enabled": bool(
            focus.get("terminal_attempted_site_alternatives_enabled")
        ),
        "terminal_execution_order_alternatives_enabled": bool(
            focus.get("terminal_execution_order_alternatives_enabled")
        ),
        "terminal_execution_order_full_permutations_enabled": bool(
            focus.get("terminal_execution_order_full_permutations_enabled")
        ),
        "terminal_obstacle_ownership_diagnostics_enabled": bool(
            focus.get("terminal_obstacle_ownership_diagnostics_enabled")
        ),
        "terminal_self_overlap_detour_enabled": bool(
            focus.get("terminal_self_overlap_detour_enabled")
        ),
        "terminal_execution_order_canvas_safety_enabled": bool(
            focus.get("terminal_execution_order_canvas_safety_enabled")
        ),
        "terminal_joint_action_budget_guard_enabled": bool(
            focus.get("terminal_joint_action_budget_guard_enabled")
        ),
        "terminal_joint_single_target_after_score_guard_enabled": bool(
            focus.get("terminal_joint_single_target_after_score_guard_enabled")
        ),
        "terminal_completion_coverage_guard_enabled": bool(
            focus.get("terminal_completion_coverage_guard_enabled")
        ),
        "terminal_completion_best_incomplete_telemetry_enabled": bool(
            focus.get("terminal_completion_best_incomplete_telemetry_enabled")
        ),
        "terminal_completion_best_incomplete_fallback_enabled": bool(
            focus.get("terminal_completion_best_incomplete_fallback_enabled")
        ),
        "terminal_target_color_coverage_rank_enabled": bool(
            focus.get("terminal_target_color_coverage_rank_enabled")
        ),
        "terminal_exact_color_target_match_enabled": bool(
            focus.get("terminal_exact_color_target_match_enabled")
        ),
        "terminal_canvas_primary_color_grouping_enabled": bool(
            focus.get("terminal_canvas_primary_color_grouping_enabled")
        ),
        "structured_group_target_association_sensor_enabled": bool(
            focus.get("structured_group_target_association_sensor_enabled")
        ),
        "remaining_target_group_fallback_enabled": bool(
            focus.get("remaining_target_group_fallback_enabled")
        ),
        "single_piece_target_completion_enabled": bool(
            focus.get("single_piece_target_completion_enabled")
        ),
        "support_piece_transition_sensor_enabled": bool(
            focus.get("support_piece_transition_sensor_enabled")
        ),
        "support_reassignment_counterfactual_sensor_enabled": bool(
            focus.get("support_reassignment_counterfactual_sensor_enabled")
        ),
        "target_color_support_readiness_sensor_enabled": bool(
            focus.get("target_color_support_readiness_sensor_enabled")
        ),
        "target_color_support_blocked_ambiguous_guard_enabled": bool(
            focus.get("target_color_support_blocked_ambiguous_guard_enabled")
        ),
        "target_color_membership_ambiguity_sensor_enabled": bool(
            focus.get("target_color_membership_ambiguity_sensor_enabled")
        ),
        "kinematic_canvas_coupling_sensor_enabled": bool(
            focus.get("kinematic_canvas_coupling_sensor_enabled")
        ),
        "kinematic_membership_rank_guard_enabled": bool(
            focus.get("kinematic_membership_rank_guard_enabled")
        ),
        "terminal_immediate_move_click_empty_guard_enabled": bool(
            focus.get("terminal_immediate_move_click_empty_guard_enabled")
        ),
        "post_score1_score_bearing_immediate_guard_enabled": bool(
            focus.get("post_score1_score_bearing_immediate_guard_enabled")
        ),
        "post_score1_score_bearing_stale_target_key_count": int(
            len(
                [
                    value
                    for value in list(
                        focus.get("post_score1_score_bearing_stale_target_keys") or []
                    )
                    if str(value or "")
                ]
            )
        ),
        "terminal_completion_best_incomplete_min_coverage": int(
            focus.get("terminal_completion_best_incomplete_min_coverage", 0) or 0
        ),
        "terminal_failed_commitment_retirement_enabled": bool(
            focus.get("terminal_failed_commitment_retirement_enabled")
        ),
        "terminal_local_verifier_after_max_calls_enabled": bool(
            focus.get("terminal_local_verifier_after_max_calls_enabled")
        ),
        "terminal_immediate_target_diversity_enabled": bool(
            focus.get("terminal_immediate_target_diversity_enabled")
        ),
        "terminal_structured_target_diversity_enabled": bool(
            focus.get("terminal_structured_target_diversity_enabled")
        ),
        "terminal_structured_action_budget_guard_enabled": bool(
            focus.get("terminal_structured_action_budget_guard_enabled")
        ),
        "terminal_structured_action_budget_limit": int(
            focus.get("terminal_structured_action_budget_limit", 0) or 0
        ),
        "terminal_structured_action_budget_remaining": int(
            focus.get("terminal_structured_action_budget_remaining", 0) or 0
        ),
        "attempted_candidate_site_count": int(len(attempted_sites)),
        "large_group_destination_templates_enabled": bool(
            focus.get("large_group_destination_templates_enabled")
        ),
        "large_group_destination_templates_configured": bool(
            focus.get("large_group_destination_templates_configured")
        ),
        "large_group_min_score_successes": int(
            focus.get("large_group_min_score_successes", 0) or 0
        ),
        "large_group_score_success_count": int(
            focus.get("large_group_score_success_count", 0) or 0
        ),
        "preferred_target_key_count": int(len(preferred)),
        "suppressed_target_key_count": int(len(suppressed)),
        "detour_no_score_suppressed_target_key_count": int(
            len(detour_no_score_suppressed)
        ),
        "completed_no_score_suppressed_target_key_count": int(
            len(completed_no_score_suppressed)
        ),
        "preferred_target_keys": preferred[:8],
        "suppressed_target_keys": suppressed[:8],
        "detour_no_score_suppressed_target_keys": detour_no_score_suppressed[:8],
        "completed_no_score_suppressed_target_keys": completed_no_score_suppressed[:8],
        "satisfied_pair_lock_enabled": bool(focus.get("satisfied_pair_lock_enabled")),
        "locked_satisfied_canvas_id_count": int(len(locked_canvas_ids)),
        "locked_satisfied_target_key_count": int(len(locked_target_keys)),
        "locked_satisfied_canvas_ids": locked_canvas_ids[:8],
        "locked_satisfied_target_keys": locked_target_keys[:8],
    }


def _target_focus_sort_key(
    target: Mapping[str, Any],
    group_focus: Mapping[str, Any] | None,
) -> tuple[int, int]:
    key = _target_focus_key(target)
    focus = dict(group_focus or {})
    preferred = [str(value) for value in list(focus.get("preferred_target_keys") or [])]
    suppressed = {str(value) for value in list(focus.get("suppressed_target_keys") or [])}
    preferred_index = _target_focus_key_index(key, preferred)
    if preferred_index is not None:
        return 0, int(preferred_index)
    if key and _target_focus_key_matches_any(key, suppressed):
        return 2, 0
    return 1, 0


def _target_focus_key(target: Mapping[str, Any]) -> str:
    box = _bbox(target)
    colors = sorted(_component_colors(target))
    if box is None or not colors:
        return ""
    color_key = ",".join(str(int(color)) for color in colors)
    return (
        f"target:{color_key}:"
        f"{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}"
    )


def _target_focus_key_index(key: str, candidates: Sequence[str]) -> int | None:
    for index, candidate in enumerate(list(candidates or [])):
        if _target_focus_keys_match(str(key or ""), str(candidate or "")):
            return int(index)
    return None


def _target_focus_key_matches_any(key: str, candidates: Iterable[str]) -> bool:
    return any(_target_focus_keys_match(str(key or ""), str(candidate or "")) for candidate in candidates)


def _target_focus_keys_match(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    left_parsed = _parse_target_focus_key(left)
    right_parsed = _parse_target_focus_key(right)
    if not left_parsed or not right_parsed:
        return False
    left_colors, left_bbox = left_parsed
    right_colors, right_bbox = right_parsed
    if not bool(set(left_colors) & set(right_colors)):
        return False
    if _bbox_overlap_area(left_bbox, right_bbox) > 0:
        return True
    left_centroid = ((left_bbox[0] + left_bbox[2]) // 2, (left_bbox[1] + left_bbox[3]) // 2)
    right_centroid = ((right_bbox[0] + right_bbox[2]) // 2, (right_bbox[1] + right_bbox[3]) // 2)
    return abs(left_centroid[0] - right_centroid[0]) + abs(left_centroid[1] - right_centroid[1]) <= 4


def _parse_target_focus_key(key: str) -> tuple[tuple[int, ...], tuple[int, int, int, int]] | None:
    parts = str(key or "").split(":")
    if len(parts) != 3 or parts[0] != "target":
        return None
    try:
        colors = tuple(sorted(int(part) for part in parts[1].split(",") if part != ""))
        coords = tuple(int(part) for part in parts[2].split(","))
    except (TypeError, ValueError):
        return None
    if not colors or len(coords) != 4:
        return None
    return colors, (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))


def _canonicalize_target_focus_keys_to_current_targets(
    keys: Sequence[str],
    targets: Sequence[Mapping[str, Any]],
    *,
    max_centroid_distance: int,
) -> tuple[list[str], dict[str, Any]]:
    out: list[str] = []
    remaps: list[dict[str, Any]] = []
    parsed_targets: list[tuple[str, tuple[int, ...], tuple[int, int, int, int]]] = []
    for target in targets:
        target_key = _target_focus_key(target)
        parsed = _parse_target_focus_key(target_key)
        if not target_key or parsed is None:
            continue
        parsed_targets.append((target_key, parsed[0], parsed[1]))

    def append_key(key: str) -> None:
        if key and not _target_focus_key_matches_any(key, out):
            out.append(key)

    for raw_key in list(keys or []):
        key = str(raw_key or "")
        if not key:
            continue
        parsed_key = _parse_target_focus_key(key)
        if parsed_key is None:
            append_key(key)
            continue
        source_colors, source_bbox = parsed_key
        source_centroid = (
            (int(source_bbox[0]) + int(source_bbox[2])) // 2,
            (int(source_bbox[1]) + int(source_bbox[3])) // 2,
        )
        best_key = ""
        best_score: tuple[int, int, int, str] | None = None
        for target_key, target_colors, target_bbox in parsed_targets:
            if not bool(set(source_colors) & set(target_colors)):
                continue
            target_centroid = (
                (int(target_bbox[0]) + int(target_bbox[2])) // 2,
                (int(target_bbox[1]) + int(target_bbox[3])) // 2,
            )
            distance = abs(source_centroid[0] - target_centroid[0]) + abs(
                source_centroid[1] - target_centroid[1]
            )
            overlap = _bbox_overlap_area(source_bbox, target_bbox)
            exact_or_compatible = bool(
                key == target_key or _target_focus_keys_match(key, target_key)
            )
            if not exact_or_compatible and distance > max_centroid_distance:
                continue
            score = (
                0 if exact_or_compatible else 1,
                int(distance),
                -int(overlap),
                str(target_key),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_key = target_key
        if not best_key:
            append_key(key)
            continue
        append_key(best_key)
        if best_key != key:
            remaps.append({"from": key, "to": best_key})

    return out, {
        "input_count": int(len([key for key in list(keys or []) if str(key or "")])),
        "output_count": int(len(out)),
        "remap_count": int(len(remaps)),
        "remaps": remaps[:8],
    }


def _generate_structured_group_completion_candidates(
    components: Sequence[Mapping[str, Any]],
    *,
    piece_obstacle_cells: set[tuple[int, int]],
    canvas_obstacle_cells: set[tuple[int, int]],
    board_size: tuple[int, int],
    allow_structured_bfs: bool,
    group_focus: Mapping[str, Any] | None = None,
    allow_target_preselection: bool = False,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Generate first-step sites for multi-piece canvas-centroid completion.

    The direct missing-piece generator only accepts sites that immediately make
    a canvas overlap its target after moving the current active piece.  r11l L2+
    often needs a sequence of piece destinations first.  This helper keeps the
    same ECA-owned verifier boundary, but enumerates structured destination
    sets from visible components and returns the next executable click site.
    """

    active = next(
        (row for row in components if row.get("kind") == "piece" and row.get("role") == "active_piece"),
        None,
    )
    pieces = [row for row in components if row.get("kind") == "piece"]
    canvases = [row for row in components if row.get("kind") == "canvas"]
    targets = [row for row in components if row.get("kind") == "target"]
    focus = dict(group_focus or {})
    preferred_target_keys = [
        str(value)
        for value in list(focus.get("preferred_target_keys") or [])
        if str(value or "")
    ]
    suppressed_target_keys = [
        str(value or "")
        for value in list(focus.get("suppressed_target_keys") or [])
        if str(value or "")
    ]
    detour_no_score_target_suppression = bool(
        focus.get("detour_no_score_target_suppression_enabled")
    )
    detour_no_score_suppressed_target_keys = [
        str(value or "")
        for value in list(focus.get("detour_no_score_suppressed_target_keys") or [])
        if str(value or "")
    ]
    completed_no_score_target_suppression = bool(
        focus.get("completed_no_score_target_suppression_enabled")
    )
    completed_no_score_suppressed_target_keys = [
        str(value or "")
        for value in list(focus.get("completed_no_score_suppressed_target_keys") or [])
        if str(value or "")
    ]
    remaining_target_recanonicalization = bool(
        focus.get("remaining_target_recanonicalization_enabled")
    )
    try:
        remaining_target_recanonicalization_max_distance = max(
            0,
            int(focus.get("remaining_target_recanonicalization_max_distance", 8) or 8),
        )
    except (TypeError, ValueError):
        remaining_target_recanonicalization_max_distance = 8
    recanonicalization_diag = {
        "enabled": bool(remaining_target_recanonicalization),
        "max_centroid_distance": int(remaining_target_recanonicalization_max_distance),
        "preferred_remap_count": 0,
        "suppressed_remap_count": 0,
    }
    if remaining_target_recanonicalization:
        preferred_target_keys, preferred_recanonicalization_diag = (
            _canonicalize_target_focus_keys_to_current_targets(
                preferred_target_keys,
                targets,
                max_centroid_distance=remaining_target_recanonicalization_max_distance,
            )
        )
        suppressed_target_keys, suppressed_recanonicalization_diag = (
            _canonicalize_target_focus_keys_to_current_targets(
                suppressed_target_keys,
                targets,
                max_centroid_distance=remaining_target_recanonicalization_max_distance,
            )
        )
        detour_no_score_suppressed_target_keys, detour_recanonicalization_diag = (
            _canonicalize_target_focus_keys_to_current_targets(
                detour_no_score_suppressed_target_keys,
                targets,
                max_centroid_distance=remaining_target_recanonicalization_max_distance,
            )
        )
        completed_no_score_suppressed_target_keys, completed_recanonicalization_diag = (
            _canonicalize_target_focus_keys_to_current_targets(
                completed_no_score_suppressed_target_keys,
                targets,
                max_centroid_distance=remaining_target_recanonicalization_max_distance,
            )
        )
        recanonicalization_diag = {
            **recanonicalization_diag,
            "preferred": preferred_recanonicalization_diag,
            "suppressed": suppressed_recanonicalization_diag,
            "detour_no_score_suppressed": detour_recanonicalization_diag,
            "completed_no_score_suppressed": completed_recanonicalization_diag,
            "preferred_remap_count": int(
                preferred_recanonicalization_diag.get("remap_count", 0) or 0
            ),
            "suppressed_remap_count": int(
                suppressed_recanonicalization_diag.get("remap_count", 0) or 0
            ),
            "detour_no_score_suppressed_remap_count": int(
                detour_recanonicalization_diag.get("remap_count", 0) or 0
            ),
            "completed_no_score_suppressed_remap_count": int(
                completed_recanonicalization_diag.get("remap_count", 0) or 0
            ),
        }
        focus = {
            **focus,
            "preferred_target_keys": list(preferred_target_keys),
            "suppressed_target_keys": list(suppressed_target_keys),
            "detour_no_score_suppressed_target_keys": list(
                detour_no_score_suppressed_target_keys
            ),
            "completed_no_score_suppressed_target_keys": list(
                completed_no_score_suppressed_target_keys
            ),
            "remaining_target_recanonicalization": recanonicalization_diag,
        }
    focus_diag = _coordinate_group_focus_diagnostics(focus)
    strict_preferred_targets = bool(focus.get("strict_preferred_targets")) and bool(preferred_target_keys)
    multi_target_sweep = bool(focus.get("multi_target_assignment_sweep_enabled"))
    joint_target_assignment = bool(focus.get("joint_target_assignment_enabled"))
    joint_target_execution_plan = bool(focus.get("joint_target_execution_plan_enabled"))
    joint_target_progress_attribution = bool(
        focus.get("joint_target_progress_attribution_enabled")
    )
    lock_aware_target_canonicalization = bool(
        focus.get("lock_aware_target_canonicalization_enabled")
    )
    terminal_attempted_site_alternatives = bool(
        focus.get("terminal_attempted_site_alternatives_enabled")
    )
    terminal_execution_order_alternatives = bool(
        focus.get("terminal_execution_order_alternatives_enabled")
    )
    terminal_execution_order_full_permutations = bool(
        focus.get("terminal_execution_order_full_permutations_enabled")
    )
    terminal_obstacle_ownership_diagnostics = bool(
        focus.get("terminal_obstacle_ownership_diagnostics_enabled")
    )
    terminal_self_overlap_detour = bool(
        focus.get("terminal_self_overlap_detour_enabled")
    )
    terminal_execution_order_canvas_safety = bool(
        focus.get("terminal_execution_order_canvas_safety_enabled")
    )
    terminal_structured_target_diversity = bool(
        focus.get("terminal_structured_target_diversity_enabled")
    )
    terminal_structured_action_budget_guard = bool(
        focus.get("terminal_structured_action_budget_guard_enabled")
    )
    terminal_joint_action_budget_guard = bool(
        focus.get("terminal_joint_action_budget_guard_enabled")
    )
    terminal_completion_coverage_guard_configured = bool(
        focus.get("terminal_completion_coverage_guard_enabled")
    )
    try:
        terminal_completion_score_success_count = max(
            0, int(focus.get("large_group_score_success_count", 0) or 0)
        )
    except (TypeError, ValueError):
        terminal_completion_score_success_count = 0
    try:
        terminal_completion_min_score_successes = max(
            0, int(focus.get("large_group_min_score_successes", 0) or 0)
        )
    except (TypeError, ValueError):
        terminal_completion_min_score_successes = 0
    terminal_completion_coverage_guard = bool(
        terminal_completion_coverage_guard_configured
        and (
            terminal_completion_min_score_successes <= 0
            or terminal_completion_score_success_count
            >= terminal_completion_min_score_successes
        )
    )
    terminal_joint_single_target_after_score_guard_configured = bool(
        focus.get("terminal_joint_single_target_after_score_guard_enabled")
        and terminal_completion_score_success_count >= 1
        and not terminal_completion_coverage_guard
    )
    terminal_joint_single_target_after_score_ceiling_enabled = bool(
        focus.get("terminal_joint_single_target_after_score_ceiling_enabled")
    )
    try:
        terminal_joint_single_target_after_score_max_score_successes = max(
            0,
            int(
                focus.get(
                    "terminal_joint_single_target_after_score_max_score_successes",
                    3,
                )
                or 3
            ),
        )
    except (TypeError, ValueError):
        terminal_joint_single_target_after_score_max_score_successes = 3
    terminal_joint_single_target_after_score_ceiling_blocked = bool(
        terminal_joint_single_target_after_score_ceiling_enabled
        and terminal_joint_single_target_after_score_max_score_successes > 0
        and terminal_completion_score_success_count
        >= terminal_joint_single_target_after_score_max_score_successes
    )
    terminal_joint_single_target_after_score_guard = bool(
        terminal_joint_single_target_after_score_guard_configured
        and not terminal_joint_single_target_after_score_ceiling_blocked
    )
    terminal_completion_best_incomplete_telemetry = bool(
        focus.get("terminal_completion_best_incomplete_telemetry_enabled")
    )
    terminal_completion_best_incomplete_fallback_configured = bool(
        focus.get("terminal_completion_best_incomplete_fallback_enabled")
    )
    try:
        terminal_completion_best_incomplete_fallback_application_count = max(
            0,
            int(
                focus.get(
                    "terminal_completion_best_incomplete_fallback_application_count",
                    0,
                )
                or 0
            ),
        )
    except (TypeError, ValueError):
        terminal_completion_best_incomplete_fallback_application_count = 0
    try:
        terminal_completion_best_incomplete_max_applications = max(
            0,
            int(
                focus.get(
                    "terminal_completion_best_incomplete_max_applications",
                    1,
                )
                or 1
            ),
        )
    except (TypeError, ValueError):
        terminal_completion_best_incomplete_max_applications = 1
    terminal_completion_best_incomplete_fallback = bool(
        terminal_completion_best_incomplete_fallback_configured
        and (
            terminal_completion_best_incomplete_max_applications <= 0
            or terminal_completion_best_incomplete_fallback_application_count
            < terminal_completion_best_incomplete_max_applications
        )
    )
    try:
        terminal_completion_best_incomplete_min_coverage = max(
            1,
            int(focus.get("terminal_completion_best_incomplete_min_coverage", 2) or 2),
        )
    except (TypeError, ValueError):
        terminal_completion_best_incomplete_min_coverage = 2
    terminal_completion_track_best_incomplete = bool(
        terminal_completion_best_incomplete_telemetry
        or terminal_completion_best_incomplete_fallback
    )
    terminal_target_color_coverage_rank = bool(
        focus.get("terminal_target_color_coverage_rank_enabled")
    )
    terminal_exact_color_target_match = bool(
        focus.get("terminal_exact_color_target_match_enabled")
    )
    terminal_canvas_primary_color_grouping = bool(
        focus.get("terminal_canvas_primary_color_grouping_enabled")
    )
    structured_group_target_association_sensor = bool(
        focus.get("structured_group_target_association_sensor_enabled")
    )
    remaining_target_group_fallback = bool(
        focus.get("remaining_target_group_fallback_enabled")
    )
    single_piece_target_completion = bool(
        focus.get("single_piece_target_completion_enabled")
    )
    single_piece_target_family_sensor = bool(
        focus.get("single_piece_target_family_sensor_enabled")
    )
    support_piece_transition_sensor = bool(
        focus.get("support_piece_transition_sensor_enabled")
    )
    support_reassignment_counterfactual_sensor = bool(
        focus.get("support_reassignment_counterfactual_sensor_enabled")
    )
    target_color_support_readiness_sensor = bool(
        focus.get("target_color_support_readiness_sensor_enabled")
    )
    target_color_support_readiness_negative_guard = bool(
        focus.get("target_color_support_readiness_negative_guard_enabled")
    )
    target_color_support_blocked_ambiguous_guard = bool(
        focus.get("target_color_support_blocked_ambiguous_guard_enabled")
    )
    target_color_membership_ambiguity_sensor = bool(
        focus.get("target_color_membership_ambiguity_sensor_enabled")
    )
    target_membership_causal_posterior_rank_guard = bool(
        focus.get("target_membership_causal_posterior_rank_guard_enabled")
    )
    target_membership_causal_posterior_conservative_demote_guard = bool(
        focus.get(
            "target_membership_causal_posterior_conservative_demote_guard_enabled"
        )
    )
    target_membership_causal_posterior_sensor = bool(
        focus.get("target_membership_causal_posterior_sensor_enabled")
        or target_membership_causal_posterior_rank_guard
    )
    kinematic_canvas_coupling_sensor = bool(
        focus.get("kinematic_canvas_coupling_sensor_enabled")
    )
    kinematic_canvas_coupling_target_instance_strict = bool(
        focus.get("kinematic_canvas_coupling_target_instance_strict_enabled")
    )
    kinematic_membership_rank_guard = bool(
        focus.get("kinematic_membership_rank_guard_enabled")
    )
    kinematic_membership_ambiguous_nonprimary_reject = bool(
        focus.get("kinematic_membership_ambiguous_nonprimary_reject_enabled")
    )
    locked_satisfied_canvas_unsatisfied_target_retry = bool(
        focus.get("locked_satisfied_canvas_unsatisfied_target_retry_enabled")
    )
    terminal_completion_required_target_count = (
        len([key for key in preferred_target_keys if str(key or "")])
        if terminal_completion_coverage_guard
        else 0
    )
    try:
        terminal_structured_action_budget_remaining = max(
            0,
            int(focus.get("terminal_structured_action_budget_remaining", 0) or 0),
        )
    except (TypeError, ValueError):
        terminal_structured_action_budget_remaining = 0
    try:
        terminal_structured_action_budget_limit = max(
            0,
            int(focus.get("terminal_structured_action_budget_limit", 0) or 0),
        )
    except (TypeError, ValueError):
        terminal_structured_action_budget_limit = 0
    attempted_candidate_sites = {
        (int(value[0]), int(value[1]))
        for value in list(focus.get("attempted_candidate_sites") or [])
        if _is_xy(value)
    }
    large_group_destination_templates = bool(
        focus.get("large_group_destination_templates_enabled")
    )
    locked_satisfied_canvas_ids = {
        str(value or "")
        for value in list(focus.get("locked_satisfied_canvas_ids") or [])
        if str(value or "")
    }
    locked_satisfied_target_keys = [
        str(value or "")
        for value in list(focus.get("locked_satisfied_target_keys") or [])
        if str(value or "")
    ]
    diagnostics: dict[str, Any] = {
        "attempted": True,
        "active_piece_present": bool(active is not None),
        "piece_count": int(len(pieces)),
        "canvas_count": int(len(canvases)),
        "target_count": int(len(targets)),
        "groups_evaluated": 0,
        "destination_sets_evaluated": 0,
        "safe_destination_plan_count": 0,
        "first_action_site_count": 0,
        "multi_target_assignment_sweep_enabled": bool(multi_target_sweep),
        "joint_target_assignment_enabled": bool(joint_target_assignment),
        "joint_target_execution_plan_enabled": bool(joint_target_execution_plan),
        "joint_target_progress_attribution_enabled": bool(joint_target_progress_attribution),
        "lock_aware_target_canonicalization_enabled": bool(
            lock_aware_target_canonicalization
        ),
        "terminal_attempted_site_alternatives_enabled": bool(
            terminal_attempted_site_alternatives
        ),
        "terminal_execution_order_alternatives_enabled": bool(
            terminal_execution_order_alternatives
        ),
        "terminal_execution_order_full_permutations_enabled": bool(
            terminal_execution_order_full_permutations
        ),
        "terminal_obstacle_ownership_diagnostics_enabled": bool(
            terminal_obstacle_ownership_diagnostics
        ),
        "terminal_self_overlap_detour_enabled": bool(
            terminal_self_overlap_detour
        ),
        "terminal_execution_order_canvas_safety_enabled": bool(
            terminal_execution_order_canvas_safety
        ),
        "terminal_execution_order_canvas_safety_skip_count": 0,
        "terminal_structured_target_diversity_enabled": bool(
            terminal_structured_target_diversity
        ),
        "terminal_structured_suppressed_target_skip_count": 0,
        "terminal_structured_non_preferred_target_skip_count": 0,
        "terminal_structured_action_budget_guard_enabled": bool(
            terminal_structured_action_budget_guard
        ),
        "terminal_structured_action_budget_limit": int(
            terminal_structured_action_budget_limit
        ),
        "terminal_structured_action_budget_remaining": int(
            terminal_structured_action_budget_remaining
        ),
        "terminal_structured_action_budget_skip_count": 0,
        "terminal_joint_action_budget_guard_enabled": bool(
            terminal_joint_action_budget_guard
        ),
        "terminal_joint_action_budget_skip_count": 0,
        "terminal_joint_single_target_after_score_guard_enabled": bool(
            terminal_joint_single_target_after_score_guard
        ),
        "terminal_joint_single_target_after_score_guard_configured": bool(
            terminal_joint_single_target_after_score_guard_configured
        ),
        "terminal_joint_single_target_after_score_ceiling_enabled": bool(
            terminal_joint_single_target_after_score_ceiling_enabled
        ),
        "terminal_joint_single_target_after_score_max_score_successes": int(
            terminal_joint_single_target_after_score_max_score_successes
        ),
        "terminal_joint_single_target_after_score_ceiling_blocked": bool(
            terminal_joint_single_target_after_score_ceiling_blocked
        ),
        "terminal_joint_single_target_after_score_replacement_count": 0,
        "terminal_joint_single_target_after_score_original_target_keys": [],
        "terminal_joint_single_target_after_score_replacement_target_keys": [],
        "terminal_completion_coverage_guard_enabled": bool(
            terminal_completion_coverage_guard
        ),
        "terminal_completion_coverage_guard_configured": bool(
            terminal_completion_coverage_guard_configured
        ),
        "terminal_completion_score_success_count": int(
            terminal_completion_score_success_count
        ),
        "terminal_completion_min_score_successes": int(
            terminal_completion_min_score_successes
        ),
        "terminal_completion_required_target_count": int(
            terminal_completion_required_target_count
        ),
        "terminal_completion_coverage_skip_count": 0,
        "terminal_completion_best_incomplete_telemetry_enabled": bool(
            terminal_completion_best_incomplete_telemetry
        ),
        "terminal_completion_best_incomplete_fallback_enabled": bool(
            terminal_completion_best_incomplete_fallback
        ),
        "terminal_completion_best_incomplete_fallback_configured": bool(
            terminal_completion_best_incomplete_fallback_configured
        ),
        "terminal_completion_best_incomplete_fallback_application_count": int(
            terminal_completion_best_incomplete_fallback_application_count
        ),
        "terminal_completion_best_incomplete_max_applications": int(
            terminal_completion_best_incomplete_max_applications
        ),
        "terminal_completion_best_incomplete_fallback_block_reason": (
            "max_applications"
            if (
                terminal_completion_best_incomplete_fallback_configured
                and not terminal_completion_best_incomplete_fallback
            )
            else ""
        ),
        "terminal_completion_best_incomplete_min_coverage": int(
            terminal_completion_best_incomplete_min_coverage
        ),
        "terminal_completion_best_incomplete_fallback_applied": False,
        "terminal_completion_best_incomplete_coverage_count": 0,
        "terminal_completion_best_incomplete_required_target_count": int(
            terminal_completion_required_target_count
        ),
        "terminal_completion_best_incomplete_assignment_count": 0,
        "terminal_completion_best_incomplete_execution_site_count": 0,
        "terminal_completion_best_incomplete_target_keys": [],
        "terminal_completion_best_incomplete_first_site": [],
        "terminal_target_color_coverage_rank_enabled": bool(
            terminal_target_color_coverage_rank
        ),
        "terminal_exact_color_target_match_enabled": bool(
            terminal_exact_color_target_match
        ),
        "terminal_exact_color_target_filter_applied_count": 0,
        "terminal_exact_color_target_filter_removed_count": 0,
        "terminal_canvas_primary_color_grouping_enabled": bool(
            terminal_canvas_primary_color_grouping
        ),
        "terminal_canvas_primary_color_grouping_applied_count": 0,
        "terminal_canvas_primary_color_grouping_filtered_piece_count": 0,
        "structured_group_target_association_sensor": {
            "enabled": bool(structured_group_target_association_sensor),
            "row_count": 0,
            "no_matching_target_count": 0,
            "group_too_small_count": 0,
            "max_initial_matching_target_count": 0,
            "max_final_matching_target_count": 0,
            "max_all_matching_group_piece_count": 0,
            "max_final_group_piece_count": 0,
            "rows": [],
        },
        "remaining_target_group_fallback_enabled": bool(
            remaining_target_group_fallback
        ),
        "remaining_target_group_fallback_applied_count": 0,
        "remaining_target_group_fallback_restored_piece_count": 0,
        "single_piece_target_completion_enabled": bool(
            single_piece_target_completion
        ),
        "single_piece_target_completion_candidate_count": 0,
        "single_piece_target_family_sensor": {
            "enabled": bool(single_piece_target_family_sensor),
            "row_count": 0,
            "route_candidate_count": 0,
            "score_bearing_projection_candidate_count": 0,
            "projected_canvas_misses_target_count": 0,
            "same_color_family_under_min_count": 0,
            "support_deficit_count": 0,
            "support_deficit_external_piece_count_max": 0,
            "support_deficit_target_color_external_piece_count": 0,
            "execution_plan_candidate_count": 0,
            "active_piece_in_group_count": 0,
            "inactive_piece_only_count": 0,
            "score_bearing_projection_failure_counts": {},
            "destination_failure_counts": {},
            "execution_failure_counts": {},
            "target_keys": [],
            "rows": [],
        },
        "support_piece_transition_sensor": {
            "enabled": bool(support_piece_transition_sensor),
            "row_count": 0,
            "score_bearing_multi_piece_group_count": 0,
            "max_support_piece_count": 0,
            "max_inactive_piece_count": 0,
            "target_keys": [],
            "rows": [],
        },
        "support_reassignment_counterfactual_sensor": {
            "enabled": bool(support_reassignment_counterfactual_sensor),
            "row_count": 0,
            "support_deficit_row_count": 0,
            "counterfactual_candidate_count": 0,
            "score_bearing_counterfactual_count": 0,
            "direct_execution_blocked_count": 0,
            "support_activation_probe_eligible_count": 0,
            "support_activation_probe_blocked_count": 0,
            "max_projected_canvas_target_overlap": 0,
            "target_keys": [],
            "rows": [],
        },
        "target_color_support_readiness_sensor": {
            "enabled": bool(target_color_support_readiness_sensor),
            "row_count": 0,
            "ready_target_count": 0,
            "not_ready_target_count": 0,
            "max_target_color_piece_count": 0,
            "max_active_target_color_piece_count": 0,
            "target_keys": [],
            "rows": [],
        },
        "target_color_support_readiness_negative_guard": {
            "enabled": bool(target_color_support_readiness_negative_guard),
            "candidate_count_before": 0,
            "rejected_count": 0,
            "missing_readiness_row_count": 0,
            "target_keys": [],
            "rows": [],
        },
        "target_color_support_blocked_ambiguous_guard": {
            "enabled": bool(target_color_support_blocked_ambiguous_guard),
            "candidate_count_before": 0,
            "rejected_count": 0,
            "missing_readiness_row_count": 0,
            "target_keys": [],
            "rows": [],
        },
        "target_color_membership_ambiguity_sensor": {
            "enabled": bool(target_color_membership_ambiguity_sensor),
            "row_count": 0,
            "ambiguous_target_count": 0,
            "kinematic_evidence_required_count": 0,
            "max_visible_target_color_piece_count": 0,
            "max_active_target_color_piece_count": 0,
            "target_keys": [],
            "rows": [],
        },
        "target_membership_causal_posterior_sensor": {
            "enabled": bool(target_membership_causal_posterior_sensor),
            "row_count": 0,
            "confirmed_count": 0,
            "falsified_count": 0,
            "ambiguous_count": 0,
            "insufficient_evidence_count": 0,
            "strong_ready_count": 0,
            "weak_ready_count": 0,
            "probe_required_count": 0,
            "blocked_count": 0,
            "rank_signal_counts": {},
            "target_keys": [],
            "rows": [],
        },
        "target_membership_causal_posterior_rank_guard": {
            "enabled": bool(target_membership_causal_posterior_rank_guard),
            "conservative_demote_guard_enabled": bool(
                target_membership_causal_posterior_conservative_demote_guard
            ),
            "candidate_count": 0,
            "boost_count": 0,
            "demote_count": 0,
            "demote_suppressed_no_confirmed_count": 0,
            "confirmed_count": 0,
            "falsified_count": 0,
            "ambiguous_count": 0,
            "not_ready_count": 0,
            "missing_posterior_count": 0,
            "score_histogram": {},
            "reason_counts": {},
            "target_keys": [],
            "rows": [],
        },
        "kinematic_canvas_coupling_sensor": {
            "enabled": bool(kinematic_canvas_coupling_sensor),
            "target_instance_strict": bool(
                kinematic_canvas_coupling_target_instance_strict
            ),
            "row_count": 0,
            "confirmed_count": 0,
            "falsified_count": 0,
            "ambiguous_count": 0,
            "insufficient_evidence_count": 0,
            "target_keys": [],
            "rows": [],
        },
        "kinematic_membership_rank_guard": {
            "enabled": bool(kinematic_membership_rank_guard),
            "candidate_count": 0,
            "boost_count": 0,
            "demote_count": 0,
            "exact_confirmed_count": 0,
            "target_confirmed_count": 0,
            "exact_falsified_count": 0,
            "blocked_ambiguous_count": 0,
            "score_histogram": {},
            "rows": [],
        },
        "kinematic_membership_ambiguous_nonprimary_reject": {
            "enabled": bool(kinematic_membership_ambiguous_nonprimary_reject),
            "candidate_count_before": 0,
            "singleton_candidate_count": 0,
            "rejected_count": 0,
            "nonprimary_rejected_count": 0,
            "target_keys": [],
            "rows": [],
        },
        "attempted_candidate_site_count": int(len(attempted_candidate_sites)),
        "attempted_site_candidate_count": 0,
        "terminal_attempted_site_alternative_selected": False,
        "terminal_execution_order_alternative_selected": False,
        "large_group_destination_templates_enabled": bool(large_group_destination_templates),
        "locked_satisfied_target_skip_count": 0,
        "locked_satisfied_canvas_skip_count": 0,
        "locked_satisfied_canvas_unsatisfied_target_skip_count": 0,
        "locked_satisfied_canvas_unsatisfied_target_retry_count": 0,
        "locked_satisfied_canvas_unsatisfied_target_keys": [],
        "joint_assignment_canonical_target_conflict_count": 0,
        "target_options_evaluated": 0,
        "selected_group": {},
        "joint_target_assignment": {
            "enabled": bool(joint_target_assignment),
            "execution_plan_enabled": bool(joint_target_execution_plan),
            "progress_attribution_enabled": bool(joint_target_progress_attribution),
            "attempted": False,
            "canvas_count_considered": 0,
            "pair_candidate_count": 0,
            "search_state_count": 0,
            "selected_assignment_count": 0,
            "selected_execution_site_count": 0,
            "selected_target_keys": [],
            "selected_canvas_ids": [],
            "block_reason": "",
            "best_incomplete_telemetry_enabled": bool(
                terminal_completion_best_incomplete_telemetry
            ),
            "best_incomplete_fallback_enabled": bool(
                terminal_completion_best_incomplete_fallback
            ),
            "best_incomplete_fallback_configured": bool(
                terminal_completion_best_incomplete_fallback_configured
            ),
            "best_incomplete_fallback_application_count": int(
                terminal_completion_best_incomplete_fallback_application_count
            ),
            "best_incomplete_max_applications": int(
                terminal_completion_best_incomplete_max_applications
            ),
            "best_incomplete_fallback_block_reason": (
                "max_applications"
                if (
                    terminal_completion_best_incomplete_fallback_configured
                    and not terminal_completion_best_incomplete_fallback
                )
                else ""
            ),
            "best_incomplete_min_coverage": int(
                terminal_completion_best_incomplete_min_coverage
            ),
            "best_incomplete_fallback_applied": False,
            "best_incomplete_coverage_count": 0,
            "best_incomplete_required_target_count": int(
                terminal_completion_required_target_count
            ),
            "best_incomplete_assignment_count": 0,
            "best_incomplete_execution_site_count": 0,
            "best_incomplete_target_keys": [],
            "best_incomplete_first_site": [],
            "target_color_coverage_rank_enabled": bool(
                terminal_target_color_coverage_rank
            ),
            "single_target_after_score_guard_enabled": bool(
                terminal_joint_single_target_after_score_guard
            ),
            "single_target_after_score_replacement_count": 0,
            "single_target_after_score_original_target_keys": [],
            "single_target_after_score_replacement_target_keys": [],
        },
        "skip_reasons": {},
        "coordinate_group_focus": focus_diag,
        "remaining_target_recanonicalization": recanonicalization_diag,
        "detour_no_score_target_suppression": {
            "enabled": bool(detour_no_score_target_suppression),
            "target_keys": list(detour_no_score_suppressed_target_keys[:8]),
            "skip_count": 0,
            "rows": [],
        },
        "completed_no_score_target_suppression": {
            "enabled": bool(completed_no_score_target_suppression),
            "target_keys": list(completed_no_score_suppressed_target_keys[:8]),
            "skip_count": 0,
            "rows": [],
        },
        "target_preselection_allowed": bool(allow_target_preselection),
    }

    def kinematic_coupling_rows_for_target(
        target_key: str,
        target_colors: set[int],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw in list(focus.get("kinematic_canvas_coupling_evidence") or []):
            if not isinstance(raw, Mapping):
                continue
            raw_key = str(raw.get("target_key", "") or raw.get("target_focus_key", "") or "")
            raw_colors = {
                int(value)
                for value in list(raw.get("target_colors") or [])
                if isinstance(value, int)
                or (isinstance(value, str) and value.strip().lstrip("-").isdigit())
            }
            key_matches = bool(
                target_key and raw_key and _target_focus_keys_match(target_key, raw_key)
            )
            color_matches = bool(target_colors and raw_colors and target_colors & raw_colors)
            if kinematic_canvas_coupling_target_instance_strict and target_key and raw_key:
                if key_matches:
                    rows.append(dict(raw))
            elif key_matches or color_matches:
                rows.append(dict(raw))
        return rows

    def kinematic_rank_guard_score(record: Mapping[str, Any]) -> tuple[int, str]:
        if not kinematic_membership_rank_guard:
            return 0, "disabled"
        target_key = str(record.get("target_focus_key", "") or "")
        target_colors = {
            int(value)
            for colors in list(record.get("group_piece_colors") or [])
            if isinstance(colors, (list, tuple))
            for value in list(colors or [])
            if isinstance(value, int)
            or (isinstance(value, str) and value.strip().lstrip("-").isdigit())
        }
        if not target_colors:
            parsed = _parse_target_focus_key(target_key)
            if parsed is not None:
                target_colors = {int(value) for value in parsed[0]}
        rows = kinematic_coupling_rows_for_target(target_key, target_colors)
        site = record.get("site")
        site_xy = (
            (int(site[0]), int(site[1]))
            if isinstance(site, (list, tuple)) and len(site) >= 2
            else None
        )
        exact_rows: list[dict[str, Any]] = []
        if site_xy is not None:
            for row in rows:
                raw_site = row.get("selected_site")
                if not isinstance(raw_site, (list, tuple)) or len(raw_site) < 2:
                    continue
                try:
                    if (int(raw_site[0]), int(raw_site[1])) == site_xy:
                        exact_rows.append(dict(row))
                except (TypeError, ValueError):
                    continue
        if any(str(row.get("status", "") or "") == "confirmed" for row in exact_rows):
            return -3, "exact_confirmed"
        if any(str(row.get("status", "") or "") == "falsified" for row in exact_rows):
            return 4, "exact_falsified"
        if any(str(row.get("status", "") or "") == "confirmed" for row in rows):
            return -1, "target_confirmed"
        target_color_piece_count = int(record.get("piece_count", 0) or 0)
        if target_color_piece_count > 2:
            return 3, "blocked_ambiguous"
        return 0, "neutral"

    def apply_kinematic_rank_guard_metadata(records: list[dict[str, Any]]) -> None:
        if not kinematic_membership_rank_guard:
            return
        sensor = diagnostics["kinematic_membership_rank_guard"]
        for record in records:
            score, reason = kinematic_rank_guard_score(record)
            record["kinematic_membership_rank_score"] = int(score)
            record["kinematic_membership_rank_reason"] = str(reason)
            sensor["candidate_count"] = int(sensor.get("candidate_count", 0) or 0) + 1
            counts = dict(sensor.get("score_histogram") or {})
            counts[str(score)] = int(counts.get(str(score), 0) or 0) + 1
            sensor["score_histogram"] = counts
            if score < 0:
                sensor["boost_count"] = int(sensor.get("boost_count", 0) or 0) + 1
            elif score > 0:
                sensor["demote_count"] = int(sensor.get("demote_count", 0) or 0) + 1
            if reason == "exact_confirmed":
                sensor["exact_confirmed_count"] = int(
                    sensor.get("exact_confirmed_count", 0) or 0
                ) + 1
            elif reason == "target_confirmed":
                sensor["target_confirmed_count"] = int(
                    sensor.get("target_confirmed_count", 0) or 0
                ) + 1
            elif reason == "exact_falsified":
                sensor["exact_falsified_count"] = int(
                    sensor.get("exact_falsified_count", 0) or 0
                ) + 1
            elif reason == "blocked_ambiguous":
                sensor["blocked_ambiguous_count"] = int(
                    sensor.get("blocked_ambiguous_count", 0) or 0
                ) + 1
            rows = list(sensor.get("rows") or [])
            if len(rows) < 8 and score != 0:
                rows.append(
                    {
                        "target_focus_key": str(record.get("target_focus_key", "") or ""),
                        "site": [
                            int(record["site"][0]),
                            int(record["site"][1]),
                        ]
                        if _is_xy(record.get("site"))
                        else [],
                        "rank_score": int(score),
                        "rank_reason": str(reason),
                    }
                )
                sensor["rows"] = rows

    def kinematic_rank_guard_sort_key(record: Mapping[str, Any]) -> int:
        if not kinematic_membership_rank_guard:
            return 0
        return int(record.get("kinematic_membership_rank_score", 0) or 0)

    def causal_posterior_row_for_target(target_key: str) -> dict[str, Any] | None:
        sensor = diagnostics["target_membership_causal_posterior_sensor"]
        for row in list(sensor.get("rows") or []):
            if not isinstance(row, Mapping):
                continue
            row_key = str(row.get("target_focus_key", "") or "")
            if target_key and row_key and _target_focus_keys_match(target_key, row_key):
                return dict(row)
        return None

    def target_membership_causal_posterior_rank_guard_score(
        record: Mapping[str, Any],
    ) -> tuple[int, str]:
        if not target_membership_causal_posterior_rank_guard:
            return 0, "disabled"
        target_key = str(record.get("target_focus_key", "") or "")
        row = causal_posterior_row_for_target(target_key)
        if row is None:
            return 0, "missing_posterior"
        rank_signal = str(row.get("rank_signal", "") or "")
        if rank_signal == "boost_existing_candidates":
            return -2, "causal_confirmed"
        if rank_signal == "demote_falsified_sites":
            return 4, "causal_falsified"
        if rank_signal == "demote_until_confirmed":
            return 3, "causal_ambiguous"
        if rank_signal == "reject_if_readiness_guard_enabled":
            return 4, "causal_not_ready"
        return 0, "causal_neutral"

    def apply_target_membership_causal_posterior_rank_guard_metadata(
        records: list[dict[str, Any]],
    ) -> None:
        if not target_membership_causal_posterior_rank_guard:
            return
        sensor = diagnostics["target_membership_causal_posterior_rank_guard"]
        scored_records: list[tuple[dict[str, Any], int, str, int, str]] = []
        for record in records:
            base_score, base_reason = (
                target_membership_causal_posterior_rank_guard_score(record)
            )
            scored_records.append(
                (record, int(base_score), str(base_reason), int(base_score), str(base_reason))
            )
        has_confirmed_candidate = any(
            item[2] == "causal_confirmed" for item in scored_records
        )
        if (
            target_membership_causal_posterior_conservative_demote_guard
            and not has_confirmed_candidate
        ):
            updated: list[tuple[dict[str, Any], int, str, int, str]] = []
            for record, score, reason, base_score, base_reason in scored_records:
                if reason in {"causal_ambiguous", "causal_not_ready"}:
                    score = 0
                    reason = f"{reason}_suppressed_no_confirmed"
                    sensor["demote_suppressed_no_confirmed_count"] = int(
                        sensor.get("demote_suppressed_no_confirmed_count", 0) or 0
                    ) + 1
                updated.append((record, score, reason, base_score, base_reason))
            scored_records = updated
        for record, score, reason, base_score, base_reason in scored_records:
            record["target_membership_causal_posterior_rank_score"] = int(score)
            record["target_membership_causal_posterior_rank_reason"] = str(reason)
            record["target_membership_causal_posterior_base_rank_score"] = int(
                base_score
            )
            record["target_membership_causal_posterior_base_rank_reason"] = str(
                base_reason
            )
            target_key = str(record.get("target_focus_key", "") or "")
            posterior = causal_posterior_row_for_target(target_key) or {}
            record["target_membership_causal_state"] = str(
                posterior.get("causal_state", "") or ""
            )
            record["target_membership_planner_disposition"] = str(
                posterior.get("planner_disposition", "") or ""
            )
            record["target_membership_rank_signal"] = str(
                posterior.get("rank_signal", "") or ""
            )
            sensor["candidate_count"] = int(sensor.get("candidate_count", 0) or 0) + 1
            counts = dict(sensor.get("score_histogram") or {})
            counts[str(score)] = int(counts.get(str(score), 0) or 0) + 1
            sensor["score_histogram"] = counts
            reasons = dict(sensor.get("reason_counts") or {})
            reasons[str(reason)] = int(reasons.get(str(reason), 0) or 0) + 1
            sensor["reason_counts"] = dict(sorted(reasons.items()))
            if score < 0:
                sensor["boost_count"] = int(sensor.get("boost_count", 0) or 0) + 1
            elif score > 0:
                sensor["demote_count"] = int(sensor.get("demote_count", 0) or 0) + 1
            if reason == "causal_confirmed":
                sensor["confirmed_count"] = int(sensor.get("confirmed_count", 0) or 0) + 1
            elif reason == "causal_falsified":
                sensor["falsified_count"] = int(sensor.get("falsified_count", 0) or 0) + 1
            elif reason in {
                "causal_ambiguous",
                "causal_ambiguous_suppressed_no_confirmed",
            }:
                sensor["ambiguous_count"] = int(sensor.get("ambiguous_count", 0) or 0) + 1
            elif reason in {
                "causal_not_ready",
                "causal_not_ready_suppressed_no_confirmed",
            }:
                sensor["not_ready_count"] = int(sensor.get("not_ready_count", 0) or 0) + 1
            elif reason == "missing_posterior":
                sensor["missing_posterior_count"] = int(
                    sensor.get("missing_posterior_count", 0) or 0
                ) + 1
            keys = list(sensor.get("target_keys") or [])
            if target_key and not _target_focus_key_matches_any(target_key, keys):
                keys.append(target_key)
                sensor["target_keys"] = keys[:8]
            rows = list(sensor.get("rows") or [])
            if len(rows) < 8 and (score != 0 or reason == "missing_posterior"):
                site = record.get("site")
                rows.append(
                    {
                        "target_focus_key": target_key,
                        "site": [
                            int(site[0]),
                            int(site[1]),
                        ]
                        if _is_xy(site)
                        else [],
                        "rank_score": int(score),
                        "rank_reason": str(reason),
                        "base_rank_score": int(base_score),
                        "base_rank_reason": str(base_reason),
                        "causal_state": str(
                            posterior.get("causal_state", "") or ""
                        ),
                        "planner_disposition": str(
                            posterior.get("planner_disposition", "") or ""
                        ),
                        "rank_signal": str(posterior.get("rank_signal", "") or ""),
                    }
                )
                sensor["rows"] = rows

    def target_membership_causal_posterior_rank_guard_sort_key(
        record: Mapping[str, Any],
    ) -> int:
        if not target_membership_causal_posterior_rank_guard:
            return 0
        return int(record.get("target_membership_causal_posterior_rank_score", 0) or 0)

    def reject_kinematic_ambiguous_nonprimary_records(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not kinematic_membership_ambiguous_nonprimary_reject:
            return records
        sensor = diagnostics["kinematic_membership_ambiguous_nonprimary_reject"]
        sensor["candidate_count_before"] = int(
            sensor.get("candidate_count_before", 0) or 0
        ) + int(len(records))
        if len(records) == 1:
            sensor["singleton_candidate_count"] = int(
                sensor.get("singleton_candidate_count", 0) or 0
            ) + 1
        if len(records) != 1:
            return records
        record = records[0]
        reason = str(record.get("kinematic_membership_rank_reason", "") or "")
        rank = tuple(record.get("target_focus_rank", (0, 0)) or (0, 0))
        try:
            nonprimary_rank = int(rank[1])
        except (TypeError, ValueError, IndexError):
            nonprimary_rank = 0
        if reason != "blocked_ambiguous" or nonprimary_rank <= 0:
            return records
        target_key = str(record.get("target_focus_key", "") or "")
        sensor["rejected_count"] = int(sensor.get("rejected_count", 0) or 0) + 1
        sensor["nonprimary_rejected_count"] = int(
            sensor.get("nonprimary_rejected_count", 0) or 0
        ) + 1
        keys = list(sensor.get("target_keys") or [])
        if target_key and not _target_focus_key_matches_any(target_key, keys):
            keys.append(target_key)
            sensor["target_keys"] = keys[:8]
        rows = list(sensor.get("rows") or [])
        if len(rows) < 8:
            site = record.get("site")
            rows.append(
                {
                    "target_focus_key": target_key,
                    "site": [
                        int(site[0]),
                        int(site[1]),
                    ]
                    if _is_xy(site)
                    else [],
                    "rank_reason": reason,
                    "target_focus_rank": [
                        int(value)
                        for value in list(record.get("target_focus_rank") or [])
                        if isinstance(value, int)
                        or (
                            isinstance(value, str)
                            and value.strip().lstrip("-").isdigit()
                        )
                    ],
                    "block_reason": "kinematic_membership_ambiguous_nonprimary_singleton",
                }
            )
            sensor["rows"] = rows
        return []

    def reject_target_readiness_negative_records(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not target_color_support_readiness_negative_guard:
            return records
        sensor = diagnostics["target_color_support_readiness_negative_guard"]
        sensor["candidate_count_before"] = int(
            sensor.get("candidate_count_before", 0) or 0
        ) + int(len(records))
        kept: list[dict[str, Any]] = []
        for record in records:
            target_key = str(record.get("target_focus_key", "") or "")
            readiness = readiness_row_for_target(target_key)
            if readiness is None:
                sensor["missing_readiness_row_count"] = int(
                    sensor.get("missing_readiness_row_count", 0) or 0
                ) + 1
                kept.append(record)
                continue
            if bool(readiness.get("target_color_support_ready")):
                kept.append(record)
                continue
            sensor["rejected_count"] = int(sensor.get("rejected_count", 0) or 0) + 1
            keys = list(sensor.get("target_keys") or [])
            if target_key and not _target_focus_key_matches_any(target_key, keys):
                keys.append(target_key)
                sensor["target_keys"] = keys[:8]
            rows = list(sensor.get("rows") or [])
            if len(rows) < 8:
                site = record.get("site")
                rows.append(
                    {
                        "target_focus_key": target_key,
                        "site": [
                            int(site[0]),
                            int(site[1]),
                        ]
                        if _is_xy(site)
                        else [],
                        "readiness_block_reason": str(
                            readiness.get("readiness_block_reason", "") or ""
                        ),
                        "target_color_piece_count": int(
                            readiness.get("target_color_piece_count", 0) or 0
                        ),
                        "active_target_color_piece_count": int(
                            readiness.get("active_target_color_piece_count", 0) or 0
                        ),
                        "readiness_strength": str(
                            readiness.get("readiness_strength", "") or ""
                        ),
                        "block_reason": "target_color_support_readiness_false",
                    }
                )
                sensor["rows"] = rows
        return kept

    def reject_target_readiness_blocked_ambiguous_records(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not target_color_support_blocked_ambiguous_guard:
            return records
        sensor = diagnostics["target_color_support_blocked_ambiguous_guard"]
        sensor["candidate_count_before"] = int(
            sensor.get("candidate_count_before", 0) or 0
        ) + int(len(records))
        kept: list[dict[str, Any]] = []
        for record in records:
            target_key = str(record.get("target_focus_key", "") or "")
            readiness = readiness_row_for_target(target_key)
            if readiness is None:
                sensor["missing_readiness_row_count"] = int(
                    sensor.get("missing_readiness_row_count", 0) or 0
                ) + 1
                kept.append(record)
                continue
            readiness_strength = str(readiness.get("readiness_strength", "") or "")
            confirmed_count = int(
                readiness.get("kinematic_membership_confirmed_count", 0) or 0
            )
            if readiness_strength != "blocked_ambiguous" or confirmed_count > 0:
                kept.append(record)
                continue
            sensor["rejected_count"] = int(sensor.get("rejected_count", 0) or 0) + 1
            keys = list(sensor.get("target_keys") or [])
            if target_key and not _target_focus_key_matches_any(target_key, keys):
                keys.append(target_key)
                sensor["target_keys"] = keys[:8]
            rows = list(sensor.get("rows") or [])
            if len(rows) < 8:
                site = record.get("site")
                rows.append(
                    {
                        "target_focus_key": target_key,
                        "site": [
                            int(site[0]),
                            int(site[1]),
                        ]
                        if _is_xy(site)
                        else [],
                        "rank_reason": str(
                            record.get("kinematic_membership_rank_reason", "") or ""
                        ),
                        "target_color_piece_count": int(
                            readiness.get("target_color_piece_count", 0) or 0
                        ),
                        "active_target_color_piece_count": int(
                            readiness.get("active_target_color_piece_count", 0) or 0
                        ),
                        "kinematic_membership_confirmed_count": confirmed_count,
                        "readiness_strength": readiness_strength,
                        "block_reason": "target_color_support_blocked_ambiguous",
                    }
                )
                sensor["rows"] = rows
        return kept

    def populate_kinematic_canvas_coupling_sensor() -> None:
        if not kinematic_canvas_coupling_sensor:
            return
        sensor = diagnostics["kinematic_canvas_coupling_sensor"]
        rows: list[dict[str, Any]] = []
        for raw in list(focus.get("kinematic_canvas_coupling_evidence") or []):
            if not isinstance(raw, Mapping):
                continue
            target_key = str(
                raw.get("target_key", "") or raw.get("target_focus_key", "") or ""
            )
            status = str(raw.get("status", "") or "insufficient_evidence")
            row = {
                "status": status,
                "target_key": target_key,
                "target_focus_key": target_key,
                "target_colors": [
                    int(value)
                    for value in list(raw.get("target_colors") or [])
                    if isinstance(value, int)
                    or (isinstance(value, str) and value.strip().lstrip("-").isdigit())
                ][:8],
                "selected_site": [
                    int(value)
                    for value in list(raw.get("selected_site") or [])
                    if isinstance(value, int)
                    or (isinstance(value, str) and value.strip().lstrip("-").isdigit())
                ][:2],
                "active_piece_colors": [
                    int(value)
                    for value in list(raw.get("active_piece_colors") or [])
                    if isinstance(value, int)
                    or (isinstance(value, str) and value.strip().lstrip("-").isdigit())
                ][:8],
                "moved_canvas_colors": [
                    int(value)
                    for value in list(raw.get("moved_canvas_colors") or [])
                    if isinstance(value, int)
                    or (isinstance(value, str) and value.strip().lstrip("-").isdigit())
                ][:8],
                "canvas_delta": [
                    int(value)
                    for value in list(raw.get("canvas_delta") or [])
                    if isinstance(value, int)
                    or (isinstance(value, str) and value.strip().lstrip("-").isdigit())
                ][:2],
                "overlap_delta": int(raw.get("overlap_delta", 0) or 0),
                "distance_delta": int(raw.get("distance_delta", 0) or 0),
                "score_delta": int(raw.get("score_delta", 0) or 0),
                "posterior_key": str(raw.get("posterior_key", "") or ""),
            }
            rows.append(row)
        sensor["row_count"] = int(len(rows))
        sensor["confirmed_count"] = int(
            sum(1 for row in rows if str(row.get("status", "") or "") == "confirmed")
        )
        sensor["falsified_count"] = int(
            sum(1 for row in rows if str(row.get("status", "") or "") == "falsified")
        )
        sensor["ambiguous_count"] = int(
            sum(1 for row in rows if str(row.get("status", "") or "") == "ambiguous")
        )
        sensor["insufficient_evidence_count"] = int(
            sum(
                1
                for row in rows
                if str(row.get("status", "") or "") == "insufficient_evidence"
            )
        )
        keys: list[str] = []
        for row in rows:
            key = str(row.get("target_key", "") or "")
            if key and not _target_focus_key_matches_any(key, keys):
                keys.append(key)
        sensor["target_keys"] = keys[:8]
        sensor["rows"] = rows[:8]

    def populate_target_color_membership_ambiguity_sensor() -> None:
        if not target_color_membership_ambiguity_sensor:
            return
        sensor = diagnostics["target_color_membership_ambiguity_sensor"]
        rows: list[dict[str, Any]] = []
        required_piece_count = 2
        for target in sorted(targets, key=lambda row: _target_focus_sort_key(row, focus)):
            target_colors = _component_colors(target)
            target_key = _target_focus_key(target)
            if not target_colors or not target_key:
                continue
            matching_pieces = [
                piece
                for piece in pieces
                if _colors_overlap(_component_colors(piece), target_colors)
            ]
            active_matching = [
                piece
                for piece in matching_pieces
                if str(piece.get("role", "") or "") == "active_piece"
            ]
            matching_canvases = [
                canvas
                for canvas in canvases
                if _colors_overlap(_component_colors(canvas), target_colors)
            ]
            coupling_rows = kinematic_coupling_rows_for_target(
                target_key,
                target_colors,
            )
            confirmed_coupling_count = sum(
                1
                for row in coupling_rows
                if str(row.get("status", "") or "") == "confirmed"
            )
            visible_piece_count = int(len(matching_pieces))
            ambiguous = bool(
                visible_piece_count > required_piece_count
                and confirmed_coupling_count <= 0
            )
            reason = (
                "visible_target_color_piece_count_exceeds_min_without_kinematic_evidence"
                if ambiguous
                else ""
            )
            rows.append(
                {
                    "target_focus_key": target_key,
                    "target_colors": sorted(int(value) for value in target_colors),
                    "target_bbox": [
                        int(value) for value in (_bbox(target) or ())
                    ],
                    "required_target_color_piece_count": int(required_piece_count),
                    "visible_target_color_piece_count": visible_piece_count,
                    "active_target_color_piece_count": int(len(active_matching)),
                    "target_color_canvas_count": int(len(matching_canvases)),
                    "kinematic_evidence_available": bool(coupling_rows),
                    "kinematic_evidence_required": bool(
                        visible_piece_count > required_piece_count
                    ),
                    "kinematic_membership_confirmed_count": int(
                        confirmed_coupling_count
                    ),
                    "membership_ambiguous": bool(ambiguous),
                    "ambiguity_reason": reason,
                    "target_color_piece_ids": [
                        str(piece.get("id", "") or "") for piece in matching_pieces[:8]
                    ],
                    "active_target_color_piece_ids": [
                        str(piece.get("id", "") or "") for piece in active_matching[:8]
                    ],
                }
            )
        sensor["row_count"] = int(len(rows))
        sensor["ambiguous_target_count"] = int(
            sum(1 for row in rows if bool(row.get("membership_ambiguous")))
        )
        sensor["kinematic_evidence_required_count"] = int(
            sum(1 for row in rows if bool(row.get("kinematic_evidence_required")))
        )
        sensor["max_visible_target_color_piece_count"] = max(
            [
                int(row.get("visible_target_color_piece_count", 0) or 0)
                for row in rows
            ]
            or [0]
        )
        sensor["max_active_target_color_piece_count"] = max(
            [
                int(row.get("active_target_color_piece_count", 0) or 0)
                for row in rows
            ]
            or [0]
        )
        keys: list[str] = []
        for row in rows:
            key = str(row.get("target_focus_key", "") or "")
            if key and not _target_focus_key_matches_any(key, keys):
                keys.append(key)
        sensor["target_keys"] = keys[:8]
        sensor["rows"] = rows[:8]

    populate_kinematic_canvas_coupling_sensor()
    populate_target_color_membership_ambiguity_sensor()
    if active is None and not allow_target_preselection:
        diagnostics["skip_reasons"]["missing_active_piece"] = 1
        return [], diagnostics

    active_id = str(active.get("id", "") or "") if active is not None else ""
    active_colors = _component_colors(active or {})

    def canvas_focus_rank(canvas: Mapping[str, Any]) -> tuple[int, int]:
        canvas_colors = _component_colors(canvas)
        best = (3, 0)
        for target in targets:
            if not _colors_overlap(canvas_colors, _component_colors(target)):
                continue
            rank = _target_focus_sort_key(target, focus)
            if rank < best:
                best = rank
        return best

    ordered_canvases = sorted(
        canvases,
        key=lambda row: (
            canvas_focus_rank(row),
            0 if _colors_overlap(active_colors, _component_colors(row)) else 1,
            _centroid(row) or (999, 999),
        ),
    )

    all_piece_tops = [_bbox_top_left_from_component(piece) for piece in pieces]
    all_piece_tops = [pos for pos in all_piece_tops if pos is not None]
    ordered_sites: list[tuple[int, int]] = []

    def add_skip(reason: str) -> None:
        skips = diagnostics["skip_reasons"]
        skips[reason] = int(skips.get(reason, 0) or 0) + 1

    def populate_target_color_support_readiness_sensor() -> None:
        if not target_color_support_readiness_sensor:
            return
        sensor = diagnostics["target_color_support_readiness_sensor"]
        rows: list[dict[str, Any]] = []
        for target in sorted(targets, key=lambda row: _target_focus_sort_key(row, focus)):
            target_colors = _component_colors(target)
            target_key = _target_focus_key(target)
            if not target_colors or not target_key:
                continue
            matching_pieces = [
                piece
                for piece in pieces
                if _colors_overlap(_component_colors(piece), target_colors)
            ]
            active_matching = [
                piece
                for piece in matching_pieces
                if str(piece.get("role", "") or "") == "active_piece"
            ]
            inactive_matching = [
                piece
                for piece in matching_pieces
                if str(piece.get("role", "") or "") != "active_piece"
            ]
            matching_canvases = [
                canvas
                for canvas in canvases
                if _colors_overlap(_component_colors(canvas), target_colors)
            ]
            coupling_rows = kinematic_coupling_rows_for_target(
                target_key,
                target_colors,
            )
            confirmed_coupling_count = int(
                sum(
                    1
                    for row in coupling_rows
                    if str(row.get("status", "") or "") == "confirmed"
                )
            )
            piece_count = int(len(matching_pieces))
            active_count = int(len(active_matching))
            readiness_min_piece_count = 2
            support_ready = bool(
                piece_count >= readiness_min_piece_count and active_count >= 1
            )
            kinematic_membership_required = bool(
                target_color_membership_ambiguity_sensor
                and piece_count > readiness_min_piece_count
            )
            if support_ready and confirmed_coupling_count > 0:
                readiness_strength = "confirmed_coupled"
            elif support_ready and kinematic_membership_required:
                readiness_strength = "blocked_ambiguous"
            elif support_ready:
                readiness_strength = "weak_color_count"
            else:
                readiness_strength = ""
            if support_ready:
                block_reason = ""
            elif active_count <= 0:
                block_reason = "no_active_target_color_piece"
            else:
                block_reason = "insufficient_target_color_piece_count"
            target_bbox = _bbox(target)
            rows.append(
                {
                    "target_focus_key": target_key,
                    "target_colors": sorted(int(value) for value in target_colors),
                    "target_bbox": [int(value) for value in target_bbox]
                    if target_bbox is not None
                    else [],
                    "required_target_color_piece_count": int(readiness_min_piece_count),
                    "target_color_piece_count": piece_count,
                    "active_target_color_piece_count": active_count,
                    "inactive_target_color_piece_count": int(len(inactive_matching)),
                    "target_color_canvas_count": int(len(matching_canvases)),
                    "target_color_support_ready": bool(support_ready),
                    "readiness_block_reason": block_reason,
                    "kinematic_membership_required": bool(
                        kinematic_membership_required
                    ),
                    "kinematic_membership_confirmed_count": int(
                        confirmed_coupling_count
                    ),
                    "readiness_strength": readiness_strength,
                    "target_color_piece_ids": [
                        str(piece.get("id", "") or "") for piece in matching_pieces[:8]
                    ],
                    "active_target_color_piece_ids": [
                        str(piece.get("id", "") or "") for piece in active_matching[:8]
                    ],
                }
            )
        sensor["row_count"] = int(len(rows))
        sensor["ready_target_count"] = int(
            sum(1 for row in rows if bool(row.get("target_color_support_ready")))
        )
        sensor["not_ready_target_count"] = int(
            sum(1 for row in rows if not bool(row.get("target_color_support_ready")))
        )
        sensor["max_target_color_piece_count"] = max(
            [int(row.get("target_color_piece_count", 0) or 0) for row in rows] or [0]
        )
        sensor["max_active_target_color_piece_count"] = max(
            [int(row.get("active_target_color_piece_count", 0) or 0) for row in rows]
            or [0]
        )
        keys: list[str] = []
        for row in rows:
            key = str(row.get("target_focus_key", "") or "")
            if key and not _target_focus_key_matches_any(key, keys):
                keys.append(key)
        sensor["target_keys"] = keys[:8]
        sensor["rows"] = rows[:8]

    populate_target_color_support_readiness_sensor()

    def readiness_row_for_target(target_key: str) -> dict[str, Any] | None:
        sensor = diagnostics["target_color_support_readiness_sensor"]
        for row in list(sensor.get("rows") or []):
            if not isinstance(row, Mapping):
                continue
            row_key = str(row.get("target_focus_key", "") or "")
            if target_key and row_key and _target_focus_keys_match(target_key, row_key):
                return dict(row)
        return None

    def ambiguity_row_for_target(target_key: str) -> dict[str, Any] | None:
        sensor = diagnostics["target_color_membership_ambiguity_sensor"]
        for row in list(sensor.get("rows") or []):
            if not isinstance(row, Mapping):
                continue
            row_key = str(row.get("target_focus_key", "") or "")
            if target_key and row_key and _target_focus_keys_match(target_key, row_key):
                return dict(row)
        return None

    def populate_target_membership_causal_posterior_sensor() -> None:
        if not target_membership_causal_posterior_sensor:
            return
        sensor = diagnostics["target_membership_causal_posterior_sensor"]
        rows: list[dict[str, Any]] = []
        for target in sorted(targets, key=lambda row: _target_focus_sort_key(row, focus)):
            target_colors = _component_colors(target)
            target_key = _target_focus_key(target)
            if not target_colors or not target_key:
                continue
            readiness = readiness_row_for_target(target_key) or {}
            ambiguity = ambiguity_row_for_target(target_key) or {}
            coupling_rows = kinematic_coupling_rows_for_target(
                target_key,
                target_colors,
            )
            confirmed_count = int(
                sum(
                    1
                    for row in coupling_rows
                    if str(row.get("status", "") or "") == "confirmed"
                )
            )
            falsified_count = int(
                sum(
                    1
                    for row in coupling_rows
                    if str(row.get("status", "") or "") == "falsified"
                )
            )
            ambiguous_count = int(
                sum(
                    1
                    for row in coupling_rows
                    if str(row.get("status", "") or "") == "ambiguous"
                )
            )
            support_ready = bool(readiness.get("target_color_support_ready"))
            membership_ambiguous = bool(
                ambiguity.get("membership_ambiguous")
                or str(readiness.get("readiness_strength", "") or "")
                == "blocked_ambiguous"
            )
            readiness_strength = str(readiness.get("readiness_strength", "") or "")
            if confirmed_count > 0:
                causal_state = "confirmed"
                planner_disposition = "strong_ready"
                rank_signal = "boost_existing_candidates"
            elif falsified_count > 0:
                causal_state = "falsified"
                planner_disposition = "probe_alternative"
                rank_signal = "demote_falsified_sites"
            elif membership_ambiguous:
                causal_state = "ambiguous"
                planner_disposition = "blocked_ambiguous"
                rank_signal = "demote_until_confirmed"
            elif support_ready:
                causal_state = "insufficient_evidence"
                planner_disposition = "weak_ready"
                rank_signal = "neutral"
            else:
                causal_state = "insufficient_evidence"
                planner_disposition = "blocked_not_ready"
                rank_signal = "reject_if_readiness_guard_enabled"
            target_bbox = _bbox(target)
            rows.append(
                {
                    "target_focus_key": target_key,
                    "target_colors": sorted(int(value) for value in target_colors),
                    "target_bbox": [int(value) for value in target_bbox]
                    if target_bbox is not None
                    else [],
                    "causal_state": causal_state,
                    "planner_disposition": planner_disposition,
                    "rank_signal": rank_signal,
                    "candidate_set_policy": "rank_or_reject_existing_only",
                    "action_generation_allowed": False,
                    "direct_action_site_allowed": False,
                    "target_color_support_ready": support_ready,
                    "readiness_strength": readiness_strength,
                    "readiness_block_reason": str(
                        readiness.get("readiness_block_reason", "") or ""
                    ),
                    "membership_ambiguous": bool(membership_ambiguous),
                    "kinematic_membership_required": bool(
                        readiness.get("kinematic_membership_required")
                        or ambiguity.get("kinematic_evidence_required")
                    ),
                    "kinematic_membership_confirmed_count": int(confirmed_count),
                    "kinematic_membership_falsified_count": int(falsified_count),
                    "kinematic_membership_ambiguous_count": int(ambiguous_count),
                    "target_color_piece_count": int(
                        readiness.get(
                            "target_color_piece_count",
                            ambiguity.get("visible_target_color_piece_count", 0),
                        )
                        or 0
                    ),
                    "active_target_color_piece_count": int(
                        readiness.get(
                            "active_target_color_piece_count",
                            ambiguity.get("active_target_color_piece_count", 0),
                        )
                        or 0
                    ),
                }
            )
        sensor["row_count"] = int(len(rows))
        sensor["confirmed_count"] = int(
            sum(1 for row in rows if row.get("causal_state") == "confirmed")
        )
        sensor["falsified_count"] = int(
            sum(1 for row in rows if row.get("causal_state") == "falsified")
        )
        sensor["ambiguous_count"] = int(
            sum(1 for row in rows if row.get("causal_state") == "ambiguous")
        )
        sensor["insufficient_evidence_count"] = int(
            sum(
                1
                for row in rows
                if row.get("causal_state") == "insufficient_evidence"
            )
        )
        sensor["strong_ready_count"] = int(
            sum(1 for row in rows if row.get("planner_disposition") == "strong_ready")
        )
        sensor["weak_ready_count"] = int(
            sum(1 for row in rows if row.get("planner_disposition") == "weak_ready")
        )
        sensor["probe_required_count"] = int(
            sum(
                1
                for row in rows
                if row.get("planner_disposition")
                in {"probe_alternative", "blocked_ambiguous"}
            )
        )
        sensor["blocked_count"] = int(
            sum(
                1
                for row in rows
                if row.get("planner_disposition")
                in {"blocked_ambiguous", "blocked_not_ready"}
            )
        )
        signal_counts: dict[str, int] = {}
        keys: list[str] = []
        for row in rows:
            signal = str(row.get("rank_signal", "") or "")
            signal_counts[signal] = int(signal_counts.get(signal, 0) or 0) + 1
            key = str(row.get("target_focus_key", "") or "")
            if key and not _target_focus_key_matches_any(key, keys):
                keys.append(key)
        sensor["rank_signal_counts"] = dict(sorted(signal_counts.items()))
        sensor["target_keys"] = keys[:8]
        sensor["rows"] = rows[:8]

    populate_target_membership_causal_posterior_sensor()

    def append_group_target_association_row(row: Mapping[str, Any]) -> None:
        if not structured_group_target_association_sensor:
            return
        sensor = diagnostics["structured_group_target_association_sensor"]
        sensor["row_count"] = int(sensor.get("row_count", 0) or 0) + 1
        reason = str(row.get("skip_reason", "") or "")
        if reason == "no_matching_target":
            sensor["no_matching_target_count"] = int(
                sensor.get("no_matching_target_count", 0) or 0
            ) + 1
        if reason == "group_too_small":
            sensor["group_too_small_count"] = int(
                sensor.get("group_too_small_count", 0) or 0
            ) + 1
        for source_key, dest_key in (
            ("initial_matching_target_count", "max_initial_matching_target_count"),
            ("final_matching_target_count", "max_final_matching_target_count"),
            ("all_matching_group_piece_count", "max_all_matching_group_piece_count"),
            ("final_group_piece_count", "max_final_group_piece_count"),
        ):
            sensor[dest_key] = max(
                int(sensor.get(dest_key, 0) or 0),
                int(row.get(source_key, 0) or 0),
            )
        rows = list(sensor.get("rows") or [])
        if len(rows) < 8:
            rows.append(dict(row))
            sensor["rows"] = rows

    def append_single_piece_target_family_row(row: Mapping[str, Any]) -> None:
        if support_reassignment_counterfactual_sensor:
            counter_sensor = diagnostics["support_reassignment_counterfactual_sensor"]
            counter_sensor["row_count"] = int(
                counter_sensor.get("row_count", 0) or 0
            ) + 1
            support_deficit = row.get("support_deficit")
            if isinstance(support_deficit, Mapping):
                counter_sensor["support_deficit_row_count"] = int(
                    counter_sensor.get("support_deficit_row_count", 0) or 0
                ) + 1
            candidates = [
                dict(candidate)
                for candidate in list(
                    row.get("support_reassignment_counterfactual_candidates") or []
                )
                if isinstance(candidate, Mapping)
            ]
            counter_sensor["counterfactual_candidate_count"] = int(
                counter_sensor.get("counterfactual_candidate_count", 0) or 0
            ) + int(len(candidates))
            counter_sensor["score_bearing_counterfactual_count"] = int(
                counter_sensor.get("score_bearing_counterfactual_count", 0) or 0
            ) + sum(
                1
                for candidate in candidates
                if bool(candidate.get("score_bearing_counterfactual_candidate"))
            )
            direct_blocked_count = sum(
                1 for candidate in candidates if bool(candidate.get("direct_execution_blocked"))
            )
            activation_eligible_count = sum(
                1
                for candidate in candidates
                if bool(candidate.get("support_activation_probe_eligible"))
            )
            counter_sensor["direct_execution_blocked_count"] = int(
                counter_sensor.get("direct_execution_blocked_count", 0) or 0
            ) + int(direct_blocked_count)
            counter_sensor["support_activation_probe_eligible_count"] = int(
                counter_sensor.get("support_activation_probe_eligible_count", 0) or 0
            ) + int(activation_eligible_count)
            counter_sensor["support_activation_probe_blocked_count"] = int(
                counter_sensor.get("support_activation_probe_blocked_count", 0) or 0
            ) + int(max(0, len(candidates) - activation_eligible_count))
            max_overlap = max(
                [
                    int(candidate.get("projected_canvas_target_overlap", 0) or 0)
                    for candidate in candidates
                ]
                or [0]
            )
            counter_sensor["max_projected_canvas_target_overlap"] = max(
                int(counter_sensor.get("max_projected_canvas_target_overlap", 0) or 0),
                int(max_overlap),
            )
            target_key = str(row.get("target_focus_key", "") or "")
            if target_key:
                keys = list(counter_sensor.get("target_keys") or [])
                if not _target_focus_key_matches_any(target_key, keys):
                    keys.append(target_key)
                counter_sensor["target_keys"] = keys[:8]
            rows = list(counter_sensor.get("rows") or [])
            if len(rows) < 8:
                rows.append(
                    {
                        "target_id": str(row.get("target_id", "") or ""),
                        "target_focus_key": target_key,
                        "piece_id": str(row.get("piece_id", "") or ""),
                        "same_color_family_piece_count": int(
                            row.get("same_color_family_piece_count", 0) or 0
                        ),
                        "support_deficit": dict(row.get("support_deficit") or {}),
                        "execution_guard": dict(
                            row.get("support_reassignment_execution_guard") or {}
                        ),
                        "counterfactual_candidate_count": int(len(candidates)),
                        "score_bearing_counterfactual_count": sum(
                            1
                            for candidate in candidates
                            if bool(
                                candidate.get("score_bearing_counterfactual_candidate")
                            )
                        ),
                        "max_projected_canvas_target_overlap": int(max_overlap),
                        "candidates": candidates[:6],
                    }
                )
                counter_sensor["rows"] = rows
        if not single_piece_target_family_sensor:
            return
        sensor = diagnostics["single_piece_target_family_sensor"]
        sensor["row_count"] = int(sensor.get("row_count", 0) or 0) + 1
        if bool(row.get("route_candidate")):
            sensor["route_candidate_count"] = int(
                sensor.get("route_candidate_count", 0) or 0
            ) + 1
        if bool(row.get("same_color_family_under_structured_min")):
            sensor["same_color_family_under_min_count"] = int(
                sensor.get("same_color_family_under_min_count", 0) or 0
            ) + 1
        support_deficit = row.get("support_deficit")
        if isinstance(support_deficit, Mapping):
            if bool(support_deficit.get("same_color_family_under_min")):
                sensor["support_deficit_count"] = int(
                    sensor.get("support_deficit_count", 0) or 0
                ) + 1
            sensor["support_deficit_external_piece_count_max"] = max(
                int(sensor.get("support_deficit_external_piece_count_max", 0) or 0),
                int(support_deficit.get("non_matching_piece_count", 0) or 0),
            )
            sensor["support_deficit_target_color_external_piece_count"] = int(
                sensor.get("support_deficit_target_color_external_piece_count", 0) or 0
            ) + int(support_deficit.get("target_color_external_piece_count", 0) or 0)
        if bool(row.get("score_bearing_projection_candidate")):
            sensor["score_bearing_projection_candidate_count"] = int(
                sensor.get("score_bearing_projection_candidate_count", 0) or 0
            ) + 1
        projection_failure = str(
            row.get("score_bearing_projection_failed_reason", "") or ""
        )
        if projection_failure:
            counts = dict(sensor.get("score_bearing_projection_failure_counts") or {})
            counts[projection_failure] = int(counts.get(projection_failure, 0) or 0) + 1
            sensor["score_bearing_projection_failure_counts"] = counts
            if projection_failure == "projected_canvas_misses_target":
                sensor["projected_canvas_misses_target_count"] = int(
                    sensor.get("projected_canvas_misses_target_count", 0) or 0
                ) + 1
        if int(row.get("execution_site_count", 0) or 0) > 0:
            sensor["execution_plan_candidate_count"] = int(
                sensor.get("execution_plan_candidate_count", 0) or 0
            ) + 1
        if bool(row.get("active_piece_in_group")):
            sensor["active_piece_in_group_count"] = int(
                sensor.get("active_piece_in_group_count", 0) or 0
            ) + 1
        else:
            sensor["inactive_piece_only_count"] = int(
                sensor.get("inactive_piece_only_count", 0) or 0
            ) + 1
        target_key = str(row.get("target_focus_key", "") or "")
        if target_key:
            keys = list(sensor.get("target_keys") or [])
            if not _target_focus_key_matches_any(target_key, keys):
                keys.append(target_key)
            sensor["target_keys"] = keys[:8]
        destination_failure = str(row.get("destination_failed_reason", "") or "")
        if destination_failure:
            counts = dict(sensor.get("destination_failure_counts") or {})
            counts[destination_failure] = int(counts.get(destination_failure, 0) or 0) + 1
            sensor["destination_failure_counts"] = counts
        execution_failure = str(row.get("execution_failed_reason", "") or "")
        if execution_failure:
            counts = dict(sensor.get("execution_failure_counts") or {})
            counts[execution_failure] = int(counts.get(execution_failure, 0) or 0) + 1
            sensor["execution_failure_counts"] = counts
        rows = list(sensor.get("rows") or [])
        if len(rows) < 8:
            rows.append(dict(row))
            sensor["rows"] = rows

    def projected_canvas_payload(
        canvas_box: tuple[int, int, int, int] | None,
        target_box: tuple[int, int, int, int] | None,
        dest_plan: Sequence[tuple[int, int]],
    ) -> dict[str, Any]:
        if canvas_box is None or target_box is None or not dest_plan:
            return {
                "projected_canvas_bbox": [],
                "projected_canvas_target_overlap": 0,
                "projected_canvas_target_overlap_delta": 0,
            }
        piece_centroids = [
            (
                int(pos[0]) + _R11L_PIECE_W // 2,
                int(pos[1]) + _R11L_PIECE_H // 2,
            )
            for pos in dest_plan
        ]
        centroid_x = sum(pos[0] for pos in piece_centroids) // len(piece_centroids)
        centroid_y = sum(pos[1] for pos in piece_centroids) // len(piece_centroids)
        canvas_w = int(canvas_box[2]) - int(canvas_box[0]) + 1
        canvas_h = int(canvas_box[3]) - int(canvas_box[1]) + 1
        projected_canvas = (
            int(centroid_x) - canvas_w // 2,
            int(centroid_y) - canvas_h // 2,
            int(centroid_x) - canvas_w // 2 + canvas_w - 1,
            int(centroid_y) - canvas_h // 2 + canvas_h - 1,
        )
        projected_overlap = int(_bbox_overlap_area(projected_canvas, target_box))
        current_overlap = int(_bbox_overlap_area(canvas_box, target_box))
        return {
            "projected_canvas_bbox": [
                int(projected_canvas[0]),
                int(projected_canvas[1]),
                int(projected_canvas[2]),
                int(projected_canvas[3]),
            ],
            "projected_canvas_target_overlap": int(projected_overlap),
            "projected_canvas_target_overlap_delta": int(
                projected_overlap - current_overlap
            ),
        }

    def append_support_piece_transition_row(payload: Mapping[str, Any]) -> None:
        if not support_piece_transition_sensor:
            return
        sensor = diagnostics["support_piece_transition_sensor"]
        piece_count = int(payload.get("piece_count", 0) or 0)
        support_piece_count = max(0, piece_count - 1)
        roles = [
            str(value or "")
            for value in list(payload.get("group_piece_roles") or [])
            if str(value or "")
        ]
        inactive_piece_count = sum(1 for role in roles if role == "inactive_piece")
        row = {
            "target_id": str(payload.get("target_id", "") or ""),
            "target_focus_key": str(payload.get("target_focus_key", "") or ""),
            "piece_count": int(piece_count),
            "support_piece_count": int(support_piece_count),
            "active_piece_in_group": bool(payload.get("active_piece_in_group")),
            "inactive_piece_count": int(inactive_piece_count),
            "group_piece_ids": [
                str(value or "")
                for value in list(payload.get("group_piece_ids") or [])
                if str(value or "")
            ][:8],
            "group_piece_roles": roles[:8],
            "group_piece_colors": [
                [int(color) for color in list(colors or [])]
                for colors in list(payload.get("group_piece_colors") or [])[:8]
                if isinstance(colors, (list, tuple))
            ],
            "destination_plan": [
                [int(pos[0]), int(pos[1])]
                for pos in list(payload.get("destination_plan") or [])
                if _is_xy(pos)
            ],
            "execution_sites": [
                [int(pos[0]), int(pos[1])]
                for pos in list(payload.get("execution_sites") or [])
                if _is_xy(pos)
            ],
            "execution_site_count": int(payload.get("execution_site_count", 0) or 0),
            "projected_canvas_bbox": [
                int(value) for value in list(payload.get("projected_canvas_bbox") or [])
            ][:4],
            "projected_canvas_target_overlap": int(
                payload.get("projected_canvas_target_overlap", 0) or 0
            ),
            "score_bearing_projection_candidate": bool(
                payload.get("score_bearing_projection_candidate")
            ),
        }
        sensor["row_count"] = int(sensor.get("row_count", 0) or 0) + 1
        sensor["max_support_piece_count"] = max(
            int(sensor.get("max_support_piece_count", 0) or 0),
            int(support_piece_count),
        )
        sensor["max_inactive_piece_count"] = max(
            int(sensor.get("max_inactive_piece_count", 0) or 0),
            int(inactive_piece_count),
        )
        if piece_count >= 2 and bool(row["score_bearing_projection_candidate"]):
            sensor["score_bearing_multi_piece_group_count"] = int(
                sensor.get("score_bearing_multi_piece_group_count", 0) or 0
            ) + 1
        target_key = str(row.get("target_focus_key", "") or "")
        if target_key:
            keys = list(sensor.get("target_keys") or [])
            if not _target_focus_key_matches_any(target_key, keys):
                keys.append(target_key)
            sensor["target_keys"] = keys[:8]
        rows = list(sensor.get("rows") or [])
        if len(rows) < 8:
            rows.append(row)
            sensor["rows"] = rows

    def selected_group_payload(record: Mapping[str, Any]) -> dict[str, Any]:
        target_focus_rank = tuple(record.get("target_focus_rank", (3, 0)) or (3, 0))
        site = (int(record["site"][0]), int(record["site"][1]))
        payload = {
            "canvas_id": str(record.get("canvas_id", "") or ""),
            "target_id": str(record.get("target_id", "") or ""),
            "target_focus_key": str(record.get("target_focus_key", "") or ""),
            "target_focus_rank": [int(target_focus_rank[0]), int(target_focus_rank[1])],
            "piece_count": int(record.get("piece_count", 0) or 0),
            "active_piece_in_group": bool(record.get("active_piece_in_group")),
            "first_action_site": [
                int(record["site"][0]),
                int(record["site"][1]),
            ],
            "destination_plan": [
                [int(pos[0]), int(pos[1])] for pos in list(record.get("destination_plan") or [])
            ],
            "execution_sites": [
                [int(pos[0]), int(pos[1])] for pos in list(record.get("execution_sites") or [])
            ],
            "execution_site_count": int(record.get("execution_site_count", 0) or 0),
            "target_color_coverage_count": int(
                record.get("target_color_coverage_count", 0) or 0
            ),
            "target_color_missing_count": int(
                record.get("target_color_missing_count", 0) or 0
            ),
            "target_color_complete": bool(record.get("target_color_complete")),
            "target_color_exact_match": bool(
                record.get("target_color_exact_match")
            ),
            "group_piece_ids": [
                str(value or "")
                for value in list(record.get("group_piece_ids") or [])
                if str(value or "")
            ][:8],
            "group_piece_roles": [
                str(value or "")
                for value in list(record.get("group_piece_roles") or [])
                if str(value or "")
            ][:8],
            "group_piece_colors": [
                [int(color) for color in list(colors or [])]
                for colors in list(record.get("group_piece_colors") or [])[:8]
                if isinstance(colors, (list, tuple))
            ],
            "group_piece_tops": [
                [int(pos[0]), int(pos[1])]
                for pos in list(record.get("group_piece_tops") or [])
                if _is_xy(pos)
            ],
            "current_canvas_bbox": [
                int(value) for value in list(record.get("current_canvas_bbox") or [])
            ][:4],
            "target_bbox": [
                int(value) for value in list(record.get("target_bbox") or [])
            ][:4],
            "projected_canvas_bbox": [
                int(value) for value in list(record.get("projected_canvas_bbox") or [])
            ][:4],
            "projected_canvas_target_overlap": int(
                record.get("projected_canvas_target_overlap", 0) or 0
            ),
            "projected_canvas_target_overlap_delta": int(
                record.get("projected_canvas_target_overlap_delta", 0) or 0
            ),
            "score_bearing_projection_candidate": bool(
                record.get("score_bearing_projection_candidate")
            ),
            "kinematic_membership_rank_score": int(
                record.get("kinematic_membership_rank_score", 0) or 0
            ),
            "kinematic_membership_rank_reason": str(
                record.get("kinematic_membership_rank_reason", "") or ""
            ),
            "target_membership_causal_posterior_rank_score": int(
                record.get("target_membership_causal_posterior_rank_score", 0) or 0
            ),
            "target_membership_causal_posterior_rank_reason": str(
                record.get("target_membership_causal_posterior_rank_reason", "") or ""
            ),
            "target_membership_causal_state": str(
                record.get("target_membership_causal_state", "") or ""
            ),
            "target_membership_planner_disposition": str(
                record.get("target_membership_planner_disposition", "") or ""
            ),
            "target_membership_rank_signal": str(
                record.get("target_membership_rank_signal", "") or ""
            ),
            "plan_diagnostics": dict(record.get("plan_diagnostics") or {}),
            "execution_diagnostics": dict(record.get("execution_diagnostics") or {}),
        }
        if terminal_attempted_site_alternatives:
            payload["terminal_attempted_site_alternatives_enabled"] = True
            payload["selected_site_was_attempted"] = bool(site in attempted_candidate_sites)
        append_support_piece_transition_row(payload)
        return payload

    def attempted_site_rank(record: Mapping[str, Any]) -> int:
        if not terminal_attempted_site_alternatives:
            return 0
        site = record.get("site")
        if not _is_xy(site):
            return 0
        return 1 if (int(site[0]), int(site[1])) in attempted_candidate_sites else 0

    def target_color_rank(record: Mapping[str, Any]) -> tuple[int, int]:
        if terminal_exact_color_target_match:
            return (0 if bool(record.get("target_color_exact_match")) else 1, 0)
        if not terminal_target_color_coverage_rank:
            return 0, 0
        return (
            int(record.get("target_color_missing_count", 0) or 0),
            -int(record.get("target_color_coverage_count", 0) or 0),
        )

    def candidate_records_for_canvas(canvas: Mapping[str, Any]) -> list[dict[str, Any]]:
        canvas_box = _bbox(canvas)
        sensor_row: dict[str, Any] = {}
        if structured_group_target_association_sensor:
            sensor_row = {
                "canvas_id": str(canvas.get("id", "") or ""),
                "canvas_colors": sorted(_component_colors(canvas)),
                "canvas_bbox": [int(value) for value in canvas_box]
                if canvas_box is not None
                else [],
                "locked_satisfied_canvas": bool(
                    str(canvas.get("id", "") or "") in locked_satisfied_canvas_ids
                ),
                "initial_matching_target_count": 0,
                "after_lock_target_count": 0,
                "after_suppression_target_count": 0,
                "after_preferred_target_count": 0,
                "after_exact_target_count": 0,
                "final_matching_target_count": 0,
                "all_matching_group_piece_count": 0,
                "primary_group_piece_count": 0,
                "final_group_piece_count": 0,
                "candidate_record_count": 0,
                "primary_grouping_applied": False,
                "skip_reason": "",
            }
        if str(canvas.get("id", "") or "") in locked_satisfied_canvas_ids:
            diagnostics["locked_satisfied_canvas_skip_count"] += 1
            canvas_colors_for_lock = _component_colors(canvas)
            unsatisfied_target_keys = [
                _target_focus_key(target)
                for target in targets
                if (
                    _bbox(target) is not None
                    and _colors_overlap(canvas_colors_for_lock, _component_colors(target))
                    and not _target_focus_key_matches_any(
                        _target_focus_key(target),
                        locked_satisfied_target_keys,
                    )
                )
            ]
            unsatisfied_target_keys = [
                str(key or "") for key in unsatisfied_target_keys if str(key or "")
            ]
            if unsatisfied_target_keys:
                diagnostics[
                    "locked_satisfied_canvas_unsatisfied_target_skip_count"
                ] += int(len(unsatisfied_target_keys))
                current_keys = list(
                    diagnostics.get("locked_satisfied_canvas_unsatisfied_target_keys")
                    or []
                )
                for key in unsatisfied_target_keys:
                    if not _target_focus_key_matches_any(key, current_keys):
                        current_keys.append(key)
                diagnostics["locked_satisfied_canvas_unsatisfied_target_keys"] = current_keys[:8]
            if not (
                locked_satisfied_canvas_unsatisfied_target_retry
                and unsatisfied_target_keys
            ):
                add_skip("locked_satisfied_canvas")
                if sensor_row:
                    sensor_row["skip_reason"] = "locked_satisfied_canvas"
                    append_group_target_association_row(sensor_row)
                return []
            diagnostics["locked_satisfied_canvas_unsatisfied_target_retry_count"] += 1
        canvas_colors = _component_colors(canvas)
        if not canvas_colors:
            add_skip("canvas_without_color")
            if sensor_row:
                sensor_row["skip_reason"] = "canvas_without_color"
                append_group_target_association_row(sensor_row)
            return []
        matching_targets = [
            target
            for target in targets
            if _bbox(target) is not None and _colors_overlap(canvas_colors, _component_colors(target))
        ]
        if sensor_row:
            sensor_row["initial_matching_target_count"] = int(len(matching_targets))
            sensor_row["initial_matching_target_keys"] = [
                _target_focus_key(target)
                for target in matching_targets[:8]
                if _target_focus_key(target)
            ]
        if lock_aware_target_canonicalization and locked_satisfied_target_keys:
            unlocked_targets = [
                target
                for target in matching_targets
                if not _target_focus_key_matches_any(
                    _target_focus_key(target),
                    locked_satisfied_target_keys,
                )
            ]
            diagnostics["locked_satisfied_target_skip_count"] += int(
                len(matching_targets) - len(unlocked_targets)
            )
            matching_targets = unlocked_targets
        if sensor_row:
            sensor_row["after_lock_target_count"] = int(len(matching_targets))
        if detour_no_score_target_suppression and detour_no_score_suppressed_target_keys:
            before_detour_suppression_count = len(matching_targets)
            unsuppressed_targets = [
                target
                for target in matching_targets
                if not (
                    _target_focus_key(target)
                    and _target_focus_key_matches_any(
                        _target_focus_key(target),
                        detour_no_score_suppressed_target_keys,
                    )
                )
            ]
            skipped = int(before_detour_suppression_count - len(unsuppressed_targets))
            if skipped > 0:
                detour_sensor = diagnostics["detour_no_score_target_suppression"]
                detour_sensor["skip_count"] = int(
                    detour_sensor.get("skip_count", 0) or 0
                ) + skipped
                rows = list(detour_sensor.get("rows") or [])
                for target in matching_targets:
                    target_key = _target_focus_key(target)
                    if not (
                        target_key
                        and _target_focus_key_matches_any(
                            target_key,
                            detour_no_score_suppressed_target_keys,
                        )
                    ):
                        continue
                    if len(rows) >= 8:
                        break
                    rows.append(
                        {
                            "canvas_id": str(canvas.get("id", "") or ""),
                            "target_focus_key": str(target_key),
                            "suppressed_by": list(
                                detour_no_score_suppressed_target_keys[:8]
                            ),
                            "block_reason": "detour_no_score_target_suppressed",
                        }
                    )
                detour_sensor["rows"] = rows
            matching_targets = unsuppressed_targets
            if sensor_row:
                sensor_row["detour_no_score_suppressed_target_count"] = int(skipped)
                sensor_row["after_detour_no_score_suppression_target_count"] = int(
                    len(matching_targets)
                )
        if completed_no_score_target_suppression and completed_no_score_suppressed_target_keys:
            before_completed_suppression_count = len(matching_targets)
            unsuppressed_targets = [
                target
                for target in matching_targets
                if not (
                    _target_focus_key(target)
                    and _target_focus_key_matches_any(
                        _target_focus_key(target),
                        completed_no_score_suppressed_target_keys,
                    )
                )
            ]
            skipped = int(
                before_completed_suppression_count - len(unsuppressed_targets)
            )
            if skipped > 0:
                completed_sensor = diagnostics["completed_no_score_target_suppression"]
                completed_sensor["skip_count"] = int(
                    completed_sensor.get("skip_count", 0) or 0
                ) + skipped
                rows = list(completed_sensor.get("rows") or [])
                for target in matching_targets:
                    target_key = _target_focus_key(target)
                    if not (
                        target_key
                        and _target_focus_key_matches_any(
                            target_key,
                            completed_no_score_suppressed_target_keys,
                        )
                    ):
                        continue
                    if len(rows) >= 8:
                        break
                    rows.append(
                        {
                            "canvas_id": str(canvas.get("id", "") or ""),
                            "target_focus_key": str(target_key),
                            "suppressed_by": list(
                                completed_no_score_suppressed_target_keys[:8]
                            ),
                            "block_reason": "completed_no_score_target_suppressed",
                        }
                    )
                completed_sensor["rows"] = rows
            matching_targets = unsuppressed_targets
            if sensor_row:
                sensor_row["completed_no_score_suppressed_target_count"] = int(
                    skipped
                )
                sensor_row[
                    "after_completed_no_score_suppression_target_count"
                ] = int(len(matching_targets))
        if terminal_structured_target_diversity:
            before_suppression_count = len(matching_targets)
            unsuppressed_targets = [
                target
                for target in matching_targets
                if not (
                    _target_focus_key(target)
                    and _target_focus_key_matches_any(
                        _target_focus_key(target),
                        suppressed_target_keys,
                    )
                )
            ]
            diagnostics["terminal_structured_suppressed_target_skip_count"] += int(
                len(matching_targets) - len(unsuppressed_targets)
            )
            matching_targets = unsuppressed_targets
            if sensor_row:
                sensor_row["suppressed_target_count"] = int(
                    before_suppression_count - len(matching_targets)
                )
                sensor_row["after_suppression_target_count"] = int(len(matching_targets))
            if preferred_target_keys:
                before_preferred_count = len(matching_targets)
                preferred_targets = [
                    target
                    for target in matching_targets
                    if (
                        _target_focus_key(target)
                        and _target_focus_key_index(
                            _target_focus_key(target),
                            preferred_target_keys,
                        )
                        is not None
                    )
                ]
                diagnostics["terminal_structured_non_preferred_target_skip_count"] += int(
                    len(matching_targets) - len(preferred_targets)
                )
                matching_targets = preferred_targets
                if sensor_row:
                    sensor_row["non_preferred_target_count"] = int(
                        before_preferred_count - len(matching_targets)
                    )
        if sensor_row and (
            not terminal_structured_target_diversity
            or (
                int(sensor_row.get("after_suppression_target_count", 0) or 0) == 0
                and matching_targets
            )
        ):
            sensor_row["after_suppression_target_count"] = int(len(matching_targets))
        if sensor_row and (
            not preferred_target_keys
            or (
                int(sensor_row.get("after_preferred_target_count", 0) or 0) == 0
                and matching_targets
            )
        ):
            sensor_row["after_preferred_target_count"] = int(len(matching_targets))
        if strict_preferred_targets:
            matching_targets = [
                target
                for target in matching_targets
                if _target_focus_key_index(_target_focus_key(target), preferred_target_keys) is not None
            ]
            if sensor_row:
                sensor_row["after_preferred_target_count"] = int(len(matching_targets))
        if not matching_targets:
            reason = "no_matching_preferred_target" if strict_preferred_targets else "no_matching_target"
            add_skip(reason)
            if sensor_row:
                sensor_row["skip_reason"] = reason
                sensor_row["final_matching_target_count"] = 0
                append_group_target_association_row(sensor_row)
            return []
        if terminal_exact_color_target_match:
            exact_targets = [
                target
                for target in matching_targets
                if _component_colors(target) == canvas_colors
            ]
            if exact_targets:
                diagnostics["terminal_exact_color_target_filter_applied_count"] += 1
                diagnostics["terminal_exact_color_target_filter_removed_count"] += int(
                    len(matching_targets) - len(exact_targets)
                )
                matching_targets = exact_targets
        if sensor_row:
            sensor_row["after_exact_target_count"] = int(len(matching_targets))
            sensor_row["final_matching_target_count"] = int(len(matching_targets))
            sensor_row["final_matching_target_keys"] = [
                _target_focus_key(target)
                for target in matching_targets[:8]
                if _target_focus_key(target)
            ]
        matching_targets.sort(
            key=lambda target: (
                _target_focus_sort_key(target, focus),
                0 if terminal_exact_color_target_match and _component_colors(target) == canvas_colors else 1,
                -len(canvas_colors & _component_colors(target)),
                _centroid(target) or (999, 999),
            )
        )
        all_matching_group_pieces = [
            piece
            for piece in pieces
            if _bbox(piece) is not None and _colors_overlap(canvas_colors, _component_colors(piece))
        ]
        if sensor_row:
            sensor_row["all_matching_group_piece_count"] = int(
                len(all_matching_group_pieces)
            )
            sensor_row["all_matching_group_piece_ids"] = [
                str(piece.get("id", "") or "")
                for piece in all_matching_group_pieces[:8]
            ]
        group_pieces = list(all_matching_group_pieces)
        primary_grouping_applied = False
        if terminal_canvas_primary_color_grouping:
            primary_canvas_color = canvas.get("color")
            primary_colors: set[int] = set()
            try:
                if primary_canvas_color is not None:
                    primary_colors.add(int(primary_canvas_color))
            except (TypeError, ValueError):
                primary_colors = set()
            if primary_colors:
                primary_group_pieces = [
                    piece
                    for piece in group_pieces
                    if bool(primary_colors & _component_colors(piece))
                ]
                if primary_group_pieces:
                    primary_grouping_applied = True
                    diagnostics[
                        "terminal_canvas_primary_color_grouping_applied_count"
                    ] += 1
                    diagnostics[
                        "terminal_canvas_primary_color_grouping_filtered_piece_count"
                    ] += int(len(group_pieces) - len(primary_group_pieces))
                    group_pieces = primary_group_pieces
        if sensor_row:
            sensor_row["primary_grouping_applied"] = bool(primary_grouping_applied)
            sensor_row["primary_group_piece_count"] = int(len(group_pieces))
        if (
            remaining_target_group_fallback
            and primary_grouping_applied
            and len(group_pieces) < 2
            and len(all_matching_group_pieces) >= 2
            and matching_targets
        ):
            diagnostics["remaining_target_group_fallback_applied_count"] += 1
            diagnostics["remaining_target_group_fallback_restored_piece_count"] += int(
                len(all_matching_group_pieces) - len(group_pieces)
            )
            group_pieces = list(all_matching_group_pieces)
        if sensor_row:
            sensor_row["final_group_piece_count"] = int(len(group_pieces))
            sensor_row["final_group_piece_ids"] = [
                str(piece.get("id", "") or "") for piece in group_pieces[:8]
            ]
        if (
            single_piece_target_family_sensor
            or support_reassignment_counterfactual_sensor
        ) and len(group_pieces) == 1:
            piece = group_pieces[0]
            piece_top = _bbox_top_left_from_component(piece)
            piece_box = _bbox(piece)
            canvas_box_for_sensor = _bbox(canvas)
            obstacle_piece_tops_for_sensor = [
                pos for pos in all_piece_tops if pos != piece_top
            ]
            for target in matching_targets[:8]:
                target_key = _target_focus_key(target)
                target_box = _bbox(target)
                target_top = _bbox_top_left_from_component(target)
                target_colors = _component_colors(target)
                non_matching_support_pieces: list[dict[str, Any]] = []
                non_matching_support_piece_options: list[
                    tuple[Mapping[str, Any], dict[str, Any]]
                ] = []
                target_centroid = _centroid(target)
                for support_piece in pieces:
                    support_piece_id = str(support_piece.get("id", "") or "")
                    if any(
                        support_piece_id == str(group_piece.get("id", "") or "")
                        for group_piece in all_matching_group_pieces
                    ):
                        continue
                    support_box = _bbox(support_piece)
                    support_centroid = _centroid(support_piece)
                    support_colors = _component_colors(support_piece)
                    if support_box is None or support_centroid is None:
                        continue
                    distance_to_target = (
                        abs(int(support_centroid[0]) - int(target_centroid[0]))
                        + abs(int(support_centroid[1]) - int(target_centroid[1]))
                        if target_centroid is not None
                        else 10**9
                    )
                    support_payload = {
                        "piece_id": support_piece_id,
                        "role": str(support_piece.get("role", "") or ""),
                        "colors": sorted(int(value) for value in support_colors),
                        "bbox": [int(value) for value in support_box],
                        "centroid": [
                            int(support_centroid[0]),
                            int(support_centroid[1]),
                        ],
                        "overlaps_canvas_color": bool(
                            canvas_colors & support_colors
                        ),
                        "overlaps_target_color": bool(
                            target_colors & support_colors
                        ),
                        "distance_to_target_centroid": int(distance_to_target),
                    }
                    non_matching_support_pieces.append(support_payload)
                    non_matching_support_piece_options.append(
                        (support_piece, support_payload)
                    )
                non_matching_support_piece_options.sort(
                    key=lambda item: (
                        0 if bool(item[1].get("overlaps_target_color")) else 1,
                        int(
                            item[1].get("distance_to_target_centroid", 10**9)
                            or 10**9
                        ),
                        str(item[1].get("piece_id", "") or ""),
                    )
                )
                non_matching_support_pieces = [
                    dict(payload)
                    for _, payload in non_matching_support_piece_options
                ]
                row: dict[str, Any] = {
                    "canvas_id": str(canvas.get("id", "") or ""),
                    "canvas_bbox": [
                        int(value) for value in canvas_box_for_sensor
                    ]
                    if canvas_box_for_sensor is not None
                    else [],
                    "piece_id": str(piece.get("id", "") or ""),
                    "piece_role": str(piece.get("role", "") or ""),
                    "piece_bbox": [int(value) for value in piece_box]
                    if piece_box is not None
                    else [],
                    "piece_top_left": [int(piece_top[0]), int(piece_top[1])]
                    if piece_top is not None
                    else [],
                    "target_id": str(target.get("id", "") or ""),
                    "target_focus_key": str(target_key or ""),
                    "target_bbox": [int(value) for value in target_box]
                    if target_box is not None
                    else [],
                    "same_color_family_piece_count": int(
                        len(all_matching_group_pieces)
                    ),
                    "same_color_family_piece_ids": [
                        str(group_piece.get("id", "") or "")
                        for group_piece in all_matching_group_pieces[:8]
                    ],
                    "structured_completion_min_piece_count": 2,
                    "same_color_family_under_structured_min": bool(
                        len(all_matching_group_pieces) < 2
                    ),
                    "missing_piece_count_to_structured_min": int(
                        max(0, 2 - len(all_matching_group_pieces))
                    ),
                    "support_deficit": {
                        "required_same_color_piece_count": 2,
                        "same_color_family_piece_count": int(
                            len(all_matching_group_pieces)
                        ),
                        "missing_same_color_piece_count": int(
                            max(0, 2 - len(all_matching_group_pieces))
                        ),
                        "same_color_family_under_min": bool(
                            len(all_matching_group_pieces) < 2
                        ),
                        "non_matching_piece_count": int(
                            len(non_matching_support_pieces)
                        ),
                        "target_color_external_piece_count": int(
                            sum(
                                1
                                for item in non_matching_support_pieces
                                if bool(item.get("overlaps_target_color"))
                            )
                        ),
                        "canvas_color_external_piece_count": int(
                            sum(
                                1
                                for item in non_matching_support_pieces
                                if bool(item.get("overlaps_canvas_color"))
                            )
                        ),
                        "nearest_non_matching_pieces": non_matching_support_pieces[:6],
                        "requires_color_or_family_change": bool(
                            len(all_matching_group_pieces) < 2
                            and not any(
                                bool(item.get("overlaps_target_color"))
                                or bool(item.get("overlaps_canvas_color"))
                                for item in non_matching_support_pieces
                            )
                        ),
                    },
                    "target_top_left": [int(target_top[0]), int(target_top[1])]
                    if target_top is not None
                    else [],
                    "active_piece_in_group": bool(
                        str(piece.get("id", "") or "") == active_id
                    ),
                    "route_candidate": False,
                    "destination_plan_found": False,
                    "destination_failed_reason": "",
                    "execution_site_count": 0,
                    "execution_sites": [],
                    "execution_failed_reason": "",
                    "current_canvas_target_overlap": int(
                        _bbox_overlap_area(canvas_box_for_sensor, target_box)
                    )
                    if canvas_box_for_sensor is not None and target_box is not None
                    else 0,
                }
                if piece_top is None or target_top is None:
                    row["destination_failed_reason"] = "missing_piece_or_target_top"
                    append_single_piece_target_family_row(row)
                    continue
                if support_reassignment_counterfactual_sensor:
                    counterfactual_candidates: list[dict[str, Any]] = []
                    for support_piece, support_payload in (
                        non_matching_support_piece_options[:6]
                    ):
                        support_top = _bbox_top_left_from_component(support_piece)
                        if support_top is None:
                            continue
                        support_color_compatible = bool(
                            support_payload.get("overlaps_target_color")
                            or support_payload.get("overlaps_canvas_color")
                        )
                        support_activation_probe_eligible = bool(
                            support_color_compatible
                            and not bool(
                                row["support_deficit"].get(
                                    "requires_color_or_family_change"
                                )
                            )
                        )
                        if support_activation_probe_eligible:
                            support_activation_probe_block_reason = ""
                        elif not support_color_compatible:
                            support_activation_probe_block_reason = (
                                "support_piece_has_no_target_or_canvas_color_overlap"
                            )
                        else:
                            support_activation_probe_block_reason = (
                                "support_deficit_requires_color_or_family_change"
                            )
                        support_obstacle_tops = [
                            pos
                            for pos in all_piece_tops
                            if pos not in {piece_top, support_top}
                        ]
                        support_dest_plan, support_plan_diag = (
                            _solve_structured_group_destinations(
                                [piece_top, support_top],
                                target_top,
                                piece_obstacle_cells=piece_obstacle_cells,
                                canvas_obstacle_cells=canvas_obstacle_cells,
                                obstacle_piece_tops=support_obstacle_tops,
                                board_size=board_size,
                                allow_bfs_fallback=False,
                                allow_large_group_templates=False,
                                allow_single_piece_target=False,
                            )
                        )
                        candidate: dict[str, Any] = {
                            "support_piece_id": str(
                                support_payload.get("piece_id", "") or ""
                            ),
                            "support_piece_role": str(
                                support_payload.get("role", "") or ""
                            ),
                            "support_piece_colors": [
                                int(value)
                                for value in list(
                                    support_payload.get("colors") or []
                                )
                            ],
                            "support_overlaps_target_color": bool(
                                support_payload.get("overlaps_target_color")
                            ),
                            "support_overlaps_canvas_color": bool(
                                support_payload.get("overlaps_canvas_color")
                            ),
                            "support_piece_top_left": [
                                int(support_top[0]),
                                int(support_top[1]),
                            ],
                            "direct_execution_blocked": True,
                            "direct_execution_block_reason": (
                                "counterfactual_candidate_requires_verified_support_assignment"
                            ),
                            "support_activation_probe_eligible": bool(
                                support_activation_probe_eligible
                            ),
                            "support_activation_probe_block_reason": str(
                                support_activation_probe_block_reason
                            ),
                            "destination_plan_found": bool(
                                support_dest_plan is not None
                            ),
                            "destination_failed_reason": str(
                                support_plan_diag.get("failed_reason", "") or ""
                            ),
                            "score_bearing_counterfactual_candidate": False,
                            "projected_canvas_target_overlap": 0,
                            "execution_site_count": 0,
                            "execution_sites": [],
                        }
                        if support_dest_plan is not None:
                            candidate["destination_plan"] = [
                                [int(pos[0]), int(pos[1])]
                                for pos in support_dest_plan
                            ]
                            projection_payload = projected_canvas_payload(
                                canvas_box_for_sensor,
                                target_box,
                                support_dest_plan,
                            )
                            candidate.update(projection_payload)
                            candidate[
                                "score_bearing_counterfactual_candidate"
                            ] = bool(
                                int(
                                    projection_payload.get(
                                        "projected_canvas_target_overlap", 0
                                    )
                                    or 0
                                )
                                > 0
                            )
                            support_execution_sites, support_execution_diag = (
                                _execution_sites_for_destination_plan(
                                    [piece, support_piece],
                                    [piece_top, support_top],
                                    support_dest_plan,
                                    active_id=active_id,
                                    active_top=_bbox_top_left_from_component(active),
                                    all_piece_tops=all_piece_tops,
                                    all_pieces=pieces,
                                    piece_obstacle_cells=piece_obstacle_cells,
                                    canvas_obstacle_cells=canvas_obstacle_cells,
                                )
                            )
                            candidate["execution_site_count"] = int(
                                len(support_execution_sites)
                            )
                            candidate["execution_sites"] = [
                                [int(pos[0]), int(pos[1])]
                                for pos in support_execution_sites[:8]
                            ]
                            candidate["execution_failed_reason"] = str(
                                support_execution_diag.get("failed_reason", "")
                                or ""
                            )
                        counterfactual_candidates.append(candidate)
                    activation_eligible_count = sum(
                        1
                        for candidate in counterfactual_candidates
                        if bool(candidate.get("support_activation_probe_eligible"))
                    )
                    row[
                        "support_reassignment_counterfactual_candidates"
                    ] = counterfactual_candidates
                    row[
                        "support_reassignment_counterfactual_candidate_count"
                    ] = int(len(counterfactual_candidates))
                    row[
                        "score_bearing_support_reassignment_counterfactual_count"
                    ] = int(
                        sum(
                            1
                            for candidate in counterfactual_candidates
                            if bool(
                                candidate.get(
                                    "score_bearing_counterfactual_candidate"
                                )
                            )
                        )
                    )
                    row["support_reassignment_execution_guard"] = {
                        "direct_execution_blocked_count": int(
                            len(counterfactual_candidates)
                        ),
                        "support_activation_probe_eligible_count": int(
                            activation_eligible_count
                        ),
                        "support_activation_probe_blocked_count": int(
                            max(0, len(counterfactual_candidates) - activation_eligible_count)
                        ),
                        "requires_unverified_color_or_family_change": bool(
                            row["support_deficit"].get(
                                "requires_color_or_family_change"
                            )
                        ),
                    }
                dest_plan, plan_diag = _solve_structured_group_destinations(
                    [piece_top],
                    target_top,
                    piece_obstacle_cells=piece_obstacle_cells,
                    canvas_obstacle_cells=canvas_obstacle_cells,
                    obstacle_piece_tops=obstacle_piece_tops_for_sensor,
                    board_size=board_size,
                    allow_bfs_fallback=False,
                    allow_large_group_templates=False,
                    allow_single_piece_target=True,
                )
                row["destination_diagnostics"] = dict(plan_diag)
                if dest_plan is None:
                    row["destination_failed_reason"] = str(
                        plan_diag.get("failed_reason", "") or "no_destination_plan"
                    )
                    append_single_piece_target_family_row(row)
                    continue
                row["destination_plan_found"] = True
                row["destination_plan"] = [
                    [int(pos[0]), int(pos[1])] for pos in dest_plan
                ]
                if (
                    piece_box is not None
                    and canvas_box_for_sensor is not None
                    and target_box is not None
                    and dest_plan
                ):
                    piece_centroid = (
                        (int(piece_box[0]) + int(piece_box[2])) // 2,
                        (int(piece_box[1]) + int(piece_box[3])) // 2,
                    )
                    piece_w = int(piece_box[2]) - int(piece_box[0]) + 1
                    piece_h = int(piece_box[3]) - int(piece_box[1]) + 1
                    dest_top_for_projection = dest_plan[0]
                    dest_centroid = (
                        int(dest_top_for_projection[0]) + piece_w // 2,
                        int(dest_top_for_projection[1]) + piece_h // 2,
                    )
                    delta = (
                        int(dest_centroid[0]) - int(piece_centroid[0]),
                        int(dest_centroid[1]) - int(piece_centroid[1]),
                    )
                    projected_canvas = (
                        int(canvas_box_for_sensor[0]) + int(delta[0]),
                        int(canvas_box_for_sensor[1]) + int(delta[1]),
                        int(canvas_box_for_sensor[2]) + int(delta[0]),
                        int(canvas_box_for_sensor[3]) + int(delta[1]),
                    )
                    projected_overlap = int(
                        _bbox_overlap_area(projected_canvas, target_box)
                    )
                    current_overlap = int(
                        row.get("current_canvas_target_overlap", 0) or 0
                    )
                    row["score_bearing_projection"] = {
                        "model": "translate_canvas_by_single_piece_centroid_delta",
                        "piece_centroid_before": [
                            int(piece_centroid[0]),
                            int(piece_centroid[1]),
                        ],
                        "piece_centroid_after": [
                            int(dest_centroid[0]),
                            int(dest_centroid[1]),
                        ],
                        "piece_delta": [int(delta[0]), int(delta[1])],
                        "projected_canvas_bbox": [
                            int(projected_canvas[0]),
                            int(projected_canvas[1]),
                            int(projected_canvas[2]),
                            int(projected_canvas[3]),
                        ],
                        "projected_canvas_target_overlap": int(projected_overlap),
                        "current_canvas_target_overlap": int(current_overlap),
                        "projected_canvas_target_overlap_delta": int(
                            projected_overlap - current_overlap
                        ),
                    }
                    row["projected_canvas_target_overlap"] = int(projected_overlap)
                    row["projected_canvas_target_overlap_delta"] = int(
                        projected_overlap - current_overlap
                    )
                    row["score_bearing_projection_candidate"] = bool(
                        projected_overlap > 0
                    )
                    if projected_overlap <= 0:
                        row["score_bearing_projection_failed_reason"] = (
                            "projected_canvas_misses_target"
                        )
                first_site = _first_action_site_for_destination_plan(
                    [piece],
                    [piece_top],
                    dest_plan,
                    active_id=active_id,
                    active_top=_bbox_top_left_from_component(active),
                    all_piece_tops=all_piece_tops,
                )
                row["first_action_site"] = [
                    int(first_site[0]),
                    int(first_site[1]),
                ] if first_site is not None else []
                execution_sites, execution_diag = _execution_sites_for_destination_plan(
                    [piece],
                    [piece_top],
                    dest_plan,
                    active_id=active_id,
                    active_top=_bbox_top_left_from_component(active),
                    all_piece_tops=all_piece_tops,
                    all_pieces=pieces,
                    piece_obstacle_cells=piece_obstacle_cells,
                    canvas_obstacle_cells=canvas_obstacle_cells,
                )
                row["execution_site_count"] = int(len(execution_sites))
                row["execution_sites"] = [
                    [int(pos[0]), int(pos[1])] for pos in execution_sites
                ]
                row["execution_diagnostics"] = dict(execution_diag)
                if not execution_sites:
                    row["execution_failed_reason"] = str(
                        execution_diag.get("failed_reason", "") or "no_execution_sites"
                    )
                row["route_candidate"] = bool(dest_plan and execution_sites)
                append_single_piece_target_family_row(row)
        if len(group_pieces) < 2 and not (
            single_piece_target_completion and len(group_pieces) == 1
        ):
            add_skip("group_too_small")
            if sensor_row:
                sensor_row["skip_reason"] = "group_too_small"
                append_group_target_association_row(sensor_row)
            return []
        if single_piece_target_completion and len(group_pieces) == 1:
            diagnostics["single_piece_target_completion_candidate_count"] += int(
                len(matching_targets)
            )
        group_pieces.sort(
            key=lambda piece: (
                ((_centroid(piece) or (0, 0))[0] ** 2 + (_centroid(piece) or (0, 0))[1] ** 2) ** 0.5,
                str(piece.get("id", "") or ""),
            )
        )
        group_piece_tops = [_bbox_top_left_from_component(piece) for piece in group_pieces]
        if any(pos is None for pos in group_piece_tops):
            add_skip("group_piece_without_bbox")
            return []
        group_piece_tops = [pos for pos in group_piece_tops if pos is not None]
        obstacle_piece_tops = [pos for pos in all_piece_tops if pos not in group_piece_tops]
        diagnostics["groups_evaluated"] += 1
        if joint_target_assignment:
            target_options = matching_targets[:_R11L_JOINT_ASSIGNMENT_MAX_TARGET_OPTIONS_PER_CANVAS]
        else:
            target_options = matching_targets if multi_target_sweep else matching_targets[:1]
        records: list[dict[str, Any]] = []
        for target in target_options:
            diagnostics["target_options_evaluated"] += 1
            target_focus_key = _target_focus_key(target)
            target_focus_rank = _target_focus_sort_key(target, focus)
            target_colors = _component_colors(target)
            target_color_coverage_count = int(len(canvas_colors & target_colors))
            target_color_missing_count = int(len(target_colors - canvas_colors))
            target_color_complete = bool(
                target_colors and target_colors.issubset(canvas_colors)
            )
            target_color_exact_match = bool(target_colors == canvas_colors)
            target_top = _bbox_top_left_from_component(target)
            if target_top is None:
                add_skip("target_without_bbox")
                continue
            canvas_box = _bbox(canvas)
            target_box = _bbox(target)
            if canvas_box is not None and target_box is not None and _bbox_overlap_area(canvas_box, target_box) > 0:
                add_skip("canvas_already_overlaps_target")
                continue
            dest_plan, plan_diag = _solve_structured_group_destinations(
                group_piece_tops,
                target_top,
                piece_obstacle_cells=piece_obstacle_cells,
                canvas_obstacle_cells=canvas_obstacle_cells,
                obstacle_piece_tops=obstacle_piece_tops,
                board_size=board_size,
                allow_bfs_fallback=allow_structured_bfs,
                allow_large_group_templates=large_group_destination_templates,
                allow_single_piece_target=single_piece_target_completion,
            )
            diagnostics["destination_sets_evaluated"] += int(
                plan_diag.get("destination_sets_evaluated", 0) or 0
            )
            if dest_plan is None:
                add_skip(str(plan_diag.get("failed_reason", "") or "no_safe_destination_plan"))
                continue
            diagnostics["safe_destination_plan_count"] += 1
            site = _first_action_site_for_destination_plan(
                group_pieces,
                group_piece_tops,
                dest_plan,
                active_id=active_id,
                active_top=_bbox_top_left_from_component(active),
                all_piece_tops=all_piece_tops,
            )
            if site is None:
                add_skip("no_executable_first_action_site")
                continue
            execution_sites, execution_diag = _execution_sites_for_destination_plan(
                group_pieces,
                group_piece_tops,
                dest_plan,
                active_id=active_id,
                active_top=_bbox_top_left_from_component(active),
                all_piece_tops=all_piece_tops,
                all_pieces=pieces,
                piece_obstacle_cells=piece_obstacle_cells,
                canvas_obstacle_cells=canvas_obstacle_cells,
                avoid_first_sites=(
                    attempted_candidate_sites
                    if terminal_execution_order_alternatives
                    else set()
                ),
                allow_full_order_permutations=terminal_execution_order_full_permutations,
                enable_obstacle_ownership_diagnostics=(
                    terminal_obstacle_ownership_diagnostics
                ),
                enable_self_overlap_detour=terminal_self_overlap_detour,
            )
            if execution_sites:
                execution_first_site = (
                    int(execution_sites[0][0]),
                    int(execution_sites[0][1]),
                )
                if execution_first_site != site:
                    execution_diag = {
                        **execution_diag,
                        "first_site_reordered_from": [int(site[0]), int(site[1])],
                        "first_site_reordered_to": [
                            int(execution_first_site[0]),
                            int(execution_first_site[1]),
                        ],
                    }
                    site = execution_first_site
            if active is None and not execution_sites:
                add_skip("no_target_preselection_execution_plan")
                continue
            if terminal_execution_order_canvas_safety and execution_sites:
                order_canvas_safe, order_canvas_diag = _r11l_execution_order_canvas_safe(
                    group_piece_tops,
                    dest_plan,
                    execution_diag.get("execution_order") or [],
                    canvas_obstacle_cells=canvas_obstacle_cells,
                )
                execution_diag = {
                    **execution_diag,
                    "canvas_safety": order_canvas_diag,
                }
                if not order_canvas_safe:
                    diagnostics["terminal_execution_order_canvas_safety_skip_count"] += 1
                    diagnostics[
                        "terminal_execution_order_canvas_safety_last_rejection"
                    ] = order_canvas_diag
                    add_skip("terminal_execution_order_canvas_unsafe")
                    continue
            if (
                terminal_structured_action_budget_guard
                and terminal_structured_action_budget_limit > 0
                and len(execution_sites) > terminal_structured_action_budget_remaining
            ):
                diagnostics["terminal_structured_action_budget_skip_count"] += 1
                add_skip("terminal_structured_action_budget_exceeded")
                continue
            projected_canvas: tuple[int, int, int, int] | None = None
            projected_canvas_target_overlap = 0
            projected_canvas_target_overlap_delta = 0
            if canvas_box is not None and target_box is not None and dest_plan:
                piece_centroids = [
                    (
                        int(pos[0]) + _R11L_PIECE_W // 2,
                        int(pos[1]) + _R11L_PIECE_H // 2,
                    )
                    for pos in dest_plan
                ]
                centroid_x = sum(pos[0] for pos in piece_centroids) // len(piece_centroids)
                centroid_y = sum(pos[1] for pos in piece_centroids) // len(piece_centroids)
                canvas_w = int(canvas_box[2]) - int(canvas_box[0]) + 1
                canvas_h = int(canvas_box[3]) - int(canvas_box[1]) + 1
                projected_canvas = (
                    int(centroid_x) - canvas_w // 2,
                    int(centroid_y) - canvas_h // 2,
                    int(centroid_x) - canvas_w // 2 + canvas_w - 1,
                    int(centroid_y) - canvas_h // 2 + canvas_h - 1,
                )
                projected_canvas_target_overlap = int(
                    _bbox_overlap_area(projected_canvas, target_box)
                )
                projected_canvas_target_overlap_delta = int(
                    projected_canvas_target_overlap
                    - _bbox_overlap_area(canvas_box, target_box)
                )
            records.append(
                {
                    "canvas_id": str(canvas.get("id", "") or ""),
                    "target_id": str(target.get("id", "") or ""),
                    "target_focus_key": str(target_focus_key or ""),
                    "target_focus_rank": tuple(target_focus_rank),
                    "target_color_coverage_count": int(target_color_coverage_count),
                    "target_color_missing_count": int(target_color_missing_count),
                    "target_color_complete": bool(target_color_complete),
                    "target_color_exact_match": bool(target_color_exact_match),
                    "piece_count": int(len(group_piece_tops)),
                    "group_piece_ids": [
                        str(piece.get("id", "") or "") for piece in group_pieces
                    ],
                    "group_piece_roles": [
                        str(piece.get("role", "") or "") for piece in group_pieces
                    ],
                    "group_piece_colors": [
                        sorted(int(value) for value in _component_colors(piece))
                        for piece in group_pieces
                    ],
                    "active_piece_in_group": bool(
                        any(str(piece.get("id", "") or "") == active_id for piece in group_pieces)
                    ),
                    "site": (int(site[0]), int(site[1])),
                    "group_piece_tops": [(int(pos[0]), int(pos[1])) for pos in group_piece_tops],
                    "destination_plan": [(int(pos[0]), int(pos[1])) for pos in dest_plan],
                    "current_canvas_bbox": [
                        int(canvas_box[0]),
                        int(canvas_box[1]),
                        int(canvas_box[2]),
                        int(canvas_box[3]),
                    ]
                    if canvas_box is not None
                    else [],
                    "target_bbox": [
                        int(target_box[0]),
                        int(target_box[1]),
                        int(target_box[2]),
                        int(target_box[3]),
                    ]
                    if target_box is not None
                    else [],
                    "projected_canvas_bbox": [
                        int(projected_canvas[0]),
                        int(projected_canvas[1]),
                        int(projected_canvas[2]),
                        int(projected_canvas[3]),
                    ]
                    if projected_canvas is not None
                    else [],
                    "projected_canvas_target_overlap": int(
                        projected_canvas_target_overlap
                    ),
                    "projected_canvas_target_overlap_delta": int(
                        projected_canvas_target_overlap_delta
                    ),
                    "score_bearing_projection_candidate": bool(
                        projected_canvas_target_overlap > 0
                    ),
                    "execution_sites": [
                        (int(pos[0]), int(pos[1])) for pos in execution_sites
                    ],
                    "execution_site_count": int(len(execution_sites)),
                    "plan_diagnostics": plan_diag,
                    "execution_diagnostics": execution_diag,
                }
            )
        apply_target_membership_causal_posterior_rank_guard_metadata(records)
        apply_kinematic_rank_guard_metadata(records)
        records = reject_kinematic_ambiguous_nonprimary_records(records)
        records = reject_target_readiness_negative_records(records)
        records = reject_target_readiness_blocked_ambiguous_records(records)
        records.sort(
            key=lambda record: (
                attempted_site_rank(record),
                target_membership_causal_posterior_rank_guard_sort_key(record),
                kinematic_rank_guard_sort_key(record),
                tuple(record.get("target_focus_rank", (3, 0)) or (3, 0)),
                target_color_rank(record),
                0 if bool(record.get("active_piece_in_group")) else 1,
                int(record.get("execution_site_count", 0) or 0),
                int(record["site"][1]),
                int(record["site"][0]),
            )
        )
        if terminal_attempted_site_alternatives:
            diagnostics["attempted_site_candidate_count"] += sum(
                1 for record in records if attempted_site_rank(record) > 0
            )
        if sensor_row:
            sensor_row["candidate_record_count"] = int(len(records))
            if not records:
                sensor_row["skip_reason"] = str(sensor_row.get("skip_reason", "") or "no_candidate_record")
            append_group_target_association_row(sensor_row)
        return records

    if joint_target_assignment:
        joint_diag = dict(diagnostics["joint_target_assignment"])
        joint_diag["attempted"] = True
        per_canvas_records: list[list[dict[str, Any]]] = []
        for canvas in ordered_canvases[:_R11L_JOINT_ASSIGNMENT_MAX_CANVASES]:
            records = candidate_records_for_canvas(canvas)
            if records:
                per_canvas_records.append(records)
        pair_candidate_count = sum(len(records) for records in per_canvas_records)
        joint_diag["canvas_count_considered"] = int(len(per_canvas_records))
        joint_diag["pair_candidate_count"] = int(pair_candidate_count)

        best_assignment: list[dict[str, Any]] = []
        best_score = (-1, -999999, -999999, -999999, -999999, -999999)
        best_single_target_assignment: list[dict[str, Any]] = []
        best_single_target_score = (-1, -999999, -999999, -999999, -999999, -999999)
        best_incomplete_assignment: list[dict[str, Any]] = []
        best_incomplete_score = (-1, -1, -999999, -999999, -999999, -999999)
        search_state_count = 0

        def assignment_score(
            records: Sequence[Mapping[str, Any]]
        ) -> tuple[int, int, int, int, int, int]:
            target_rank_sum = 0
            execution_cost = 0
            active_bonus = 0
            fresh_site_count = 0
            causal_posterior_guard_bonus = 0
            kinematic_guard_bonus = 0
            for record in records:
                rank = tuple(record.get("target_focus_rank", (3, 0)) or (3, 0))
                target_rank_sum += int(rank[0]) * 100 + int(rank[1])
                if terminal_target_color_coverage_rank:
                    target_rank_sum += (
                        int(record.get("target_color_missing_count", 0) or 0) * 10
                        - int(record.get("target_color_coverage_count", 0) or 0)
                    )
                execution_cost += int(record.get("execution_site_count", 0) or 0)
                active_bonus += 1 if bool(record.get("active_piece_in_group")) else 0
                fresh_site_count += 1 if attempted_site_rank(record) == 0 else 0
                causal_posterior_guard_bonus += -int(
                    record.get("target_membership_causal_posterior_rank_score", 0) or 0
                )
                kinematic_guard_bonus += -int(
                    record.get("kinematic_membership_rank_score", 0) or 0
                )
            if terminal_attempted_site_alternatives:
                return (
                    int(fresh_site_count),
                    int(len(records)),
                    int(causal_posterior_guard_bonus + kinematic_guard_bonus),
                    -int(target_rank_sum),
                    int(active_bonus),
                    -int(execution_cost),
                )
            return (
                int(len(records)),
                int(causal_posterior_guard_bonus),
                int(kinematic_guard_bonus),
                -int(target_rank_sum),
                int(active_bonus),
                -int(execution_cost),
            )

        def assignment_execution_cost(records: Sequence[Mapping[str, Any]]) -> int:
            return sum(int(record.get("execution_site_count", 0) or 0) for record in records)

        def assignment_target_coverage_count(records: Sequence[Mapping[str, Any]]) -> int:
            keys: list[str] = []
            for record in records:
                key = str(record.get("target_focus_key", "") or "")
                if key and not _target_focus_key_matches_any(key, keys):
                    keys.append(key)
            return int(len(keys))

        def assignment_target_keys(records: Sequence[Mapping[str, Any]]) -> list[str]:
            keys: list[str] = []
            for record in records:
                key = str(record.get("target_focus_key", "") or "")
                if key and not _target_focus_key_matches_any(key, keys):
                    keys.append(key)
            return keys

        def assignment_completion_coverage_ok(records: Sequence[Mapping[str, Any]]) -> bool:
            if not terminal_completion_coverage_guard:
                return True
            if terminal_completion_required_target_count <= 1:
                return True
            return assignment_target_coverage_count(records) >= terminal_completion_required_target_count

        def update_best_incomplete_assignment(records: Sequence[Mapping[str, Any]]) -> None:
            nonlocal best_incomplete_assignment, best_incomplete_score
            if not (
                terminal_completion_track_best_incomplete
                and terminal_completion_coverage_guard
                and records
            ):
                return
            coverage = assignment_target_coverage_count(records)
            if coverage >= terminal_completion_required_target_count:
                return
            score = (
                int(coverage),
                int(len(records)),
                *assignment_score(records)[2:],
            )
            if score > best_incomplete_score:
                best_incomplete_score = score
                best_incomplete_assignment = [dict(record) for record in records]

        def update_best_single_target_assignment(
            records: Sequence[Mapping[str, Any]]
        ) -> None:
            nonlocal best_single_target_assignment, best_single_target_score
            if not terminal_joint_single_target_after_score_guard:
                return
            if not records or assignment_target_coverage_count(records) != 1:
                return
            score = assignment_score(records)
            if score > best_single_target_score:
                best_single_target_score = score
                best_single_target_assignment = [dict(record) for record in records]

        def compatible_with_selected(
            record: Mapping[str, Any],
            selected: Sequence[Mapping[str, Any]],
        ) -> bool:
            record_sources = {
                (int(pos[0]), int(pos[1]))
                for pos in list(record.get("group_piece_tops") or [])
                if isinstance(pos, (list, tuple)) and len(pos) >= 2
            }
            record_destinations = {
                (int(pos[0]), int(pos[1]))
                for pos in list(record.get("destination_plan") or [])
                if isinstance(pos, (list, tuple)) and len(pos) >= 2
            }
            for other in selected:
                other_sources = {
                    (int(pos[0]), int(pos[1]))
                    for pos in list(other.get("group_piece_tops") or [])
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2
                }
                other_destinations = {
                    (int(pos[0]), int(pos[1]))
                    for pos in list(other.get("destination_plan") or [])
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2
                }
                if record_destinations & other_destinations:
                    return False
                if record_destinations & other_sources:
                    return False
                if other_destinations & record_sources:
                    return False
            return True

        def search(index: int, selected: list[dict[str, Any]], used_targets: set[str]) -> None:
            nonlocal best_assignment, best_score, search_state_count
            if search_state_count >= _R11L_JOINT_ASSIGNMENT_MAX_SEARCH_STATES:
                return
            search_state_count += 1
            if (
                terminal_joint_action_budget_guard
                and terminal_structured_action_budget_limit > 0
                and assignment_execution_cost(selected) > terminal_structured_action_budget_remaining
            ):
                diagnostics["terminal_joint_action_budget_skip_count"] += 1
                return
            if assignment_completion_coverage_ok(selected):
                score = assignment_score(selected)
                if score > best_score:
                    best_score = score
                    best_assignment = list(selected)
                update_best_single_target_assignment(selected)
            if index >= len(per_canvas_records):
                if selected and not assignment_completion_coverage_ok(selected):
                    diagnostics["terminal_completion_coverage_skip_count"] += 1
                    update_best_incomplete_assignment(selected)
                return
            if len(selected) + (len(per_canvas_records) - index) < best_score[0]:
                return
            search(index + 1, selected, used_targets)
            for record in per_canvas_records[index]:
                target_key = str(record.get("target_focus_key", "") or "")
                used_target_conflict = bool(
                    target_key
                    and (
                        target_key in used_targets
                        or (
                            lock_aware_target_canonicalization
                            and _target_focus_key_matches_any(target_key, used_targets)
                        )
                    )
                )
                if used_target_conflict:
                    if lock_aware_target_canonicalization and target_key not in used_targets:
                        diagnostics["joint_assignment_canonical_target_conflict_count"] += 1
                    continue
                if not compatible_with_selected(record, selected):
                    continue
                selected.append(record)
                if target_key:
                    used_targets.add(target_key)
                search(index + 1, selected, used_targets)
                if target_key:
                    used_targets.remove(target_key)
                selected.pop()

        if per_canvas_records:
            search(0, [], set())
        best_incomplete_fallback_applied = False
        if (
            terminal_completion_best_incomplete_fallback
            and not best_assignment
            and best_incomplete_assignment
            and assignment_target_coverage_count(best_incomplete_assignment)
            >= terminal_completion_best_incomplete_min_coverage
        ):
            best_assignment = [dict(record) for record in best_incomplete_assignment]
            best_incomplete_fallback_applied = True
        single_target_after_score_replacement_applied = False
        if (
            terminal_joint_single_target_after_score_guard
            and best_assignment
            and assignment_target_coverage_count(best_assignment) > 1
            and best_single_target_assignment
        ):
            original_keys = assignment_target_keys(best_assignment)
            replacement_keys = assignment_target_keys(best_single_target_assignment)
            best_assignment = [dict(record) for record in best_single_target_assignment]
            single_target_after_score_replacement_applied = True
            diagnostics["terminal_joint_single_target_after_score_replacement_count"] += 1
            diagnostics[
                "terminal_joint_single_target_after_score_original_target_keys"
            ] = [str(key) for key in original_keys[:8]]
            diagnostics[
                "terminal_joint_single_target_after_score_replacement_target_keys"
            ] = [str(key) for key in replacement_keys[:8]]
        joint_diag["search_state_count"] = int(search_state_count)
        joint_diag["budget_guard_enabled"] = bool(terminal_joint_action_budget_guard)
        joint_diag["budget_skip_count"] = int(
            diagnostics.get("terminal_joint_action_budget_skip_count", 0) or 0
        )
        joint_diag["single_target_after_score_guard_enabled"] = bool(
            terminal_joint_single_target_after_score_guard
        )
        joint_diag["single_target_after_score_guard_configured"] = bool(
            terminal_joint_single_target_after_score_guard_configured
        )
        joint_diag["single_target_after_score_ceiling_enabled"] = bool(
            terminal_joint_single_target_after_score_ceiling_enabled
        )
        joint_diag["single_target_after_score_max_score_successes"] = int(
            terminal_joint_single_target_after_score_max_score_successes
        )
        joint_diag["single_target_after_score_ceiling_blocked"] = bool(
            terminal_joint_single_target_after_score_ceiling_blocked
        )
        joint_diag["single_target_after_score_replacement_count"] = int(
            diagnostics.get(
                "terminal_joint_single_target_after_score_replacement_count", 0
            )
            or 0
        )
        if single_target_after_score_replacement_applied:
            joint_diag["single_target_after_score_original_target_keys"] = list(
                diagnostics.get(
                    "terminal_joint_single_target_after_score_original_target_keys", []
                )
                or []
            )
            joint_diag["single_target_after_score_replacement_target_keys"] = list(
                diagnostics.get(
                    "terminal_joint_single_target_after_score_replacement_target_keys",
                    [],
                )
                or []
            )
        joint_diag["completion_coverage_guard_enabled"] = bool(
            terminal_completion_coverage_guard
        )
        joint_diag["completion_required_target_count"] = int(
            terminal_completion_required_target_count
        )
        joint_diag["completion_coverage_skip_count"] = int(
            diagnostics.get("terminal_completion_coverage_skip_count", 0) or 0
        )
        diagnostics["terminal_completion_best_incomplete_fallback_applied"] = bool(
            best_incomplete_fallback_applied
        )
        joint_diag["best_incomplete_fallback_applied"] = bool(
            best_incomplete_fallback_applied
        )
        if best_incomplete_assignment:
            incomplete_execution_site_count = assignment_execution_cost(
                best_incomplete_assignment
            )
            incomplete_keys = assignment_target_keys(best_incomplete_assignment)
            incomplete_first_site = list(best_incomplete_assignment[0].get("site") or [])
            diagnostics["terminal_completion_best_incomplete_coverage_count"] = int(
                len(incomplete_keys)
            )
            diagnostics["terminal_completion_best_incomplete_assignment_count"] = int(
                len(best_incomplete_assignment)
            )
            diagnostics["terminal_completion_best_incomplete_execution_site_count"] = int(
                incomplete_execution_site_count
            )
            diagnostics["terminal_completion_best_incomplete_target_keys"] = [
                str(key) for key in incomplete_keys[:8]
            ]
            diagnostics["terminal_completion_best_incomplete_first_site"] = [
                int(incomplete_first_site[0]),
                int(incomplete_first_site[1]),
            ] if len(incomplete_first_site) >= 2 else []
            joint_diag["best_incomplete_coverage_count"] = int(len(incomplete_keys))
            joint_diag["best_incomplete_assignment_count"] = int(
                len(best_incomplete_assignment)
            )
            joint_diag["best_incomplete_execution_site_count"] = int(
                incomplete_execution_site_count
            )
            joint_diag["best_incomplete_target_keys"] = [
                str(key) for key in incomplete_keys[:8]
            ]
            joint_diag["best_incomplete_first_site"] = [
                int(incomplete_first_site[0]),
                int(incomplete_first_site[1]),
            ] if len(incomplete_first_site) >= 2 else []
        if best_assignment:
            best_assignment.sort(
                key=lambda record: (
                    attempted_site_rank(record),
                    target_membership_causal_posterior_rank_guard_sort_key(record),
                    kinematic_rank_guard_sort_key(record),
                    tuple(record.get("target_focus_rank", (3, 0)) or (3, 0)),
                    target_color_rank(record),
                    0 if bool(record.get("active_piece_in_group")) else 1,
                    int(record.get("execution_site_count", 0) or 0),
                    int(record["site"][1]),
                    int(record["site"][0]),
                )
            )
            ordered_sites.extend([record["site"] for record in best_assignment])
            selected_payload = selected_group_payload(best_assignment[0])
            if terminal_attempted_site_alternatives and best_assignment:
                diagnostics["terminal_attempted_site_alternative_selected"] = bool(
                    attempted_site_rank(best_assignment[0]) == 0
                    and int(diagnostics.get("attempted_site_candidate_count", 0) or 0) > 0
                )
            if terminal_execution_order_alternatives and best_assignment:
                execution_diag = dict(best_assignment[0].get("execution_diagnostics") or {})
                diagnostics["terminal_execution_order_alternative_selected"] = bool(
                    execution_diag.get("first_site_reordered_to")
                )
            if joint_target_progress_attribution and len(best_assignment) > 1:
                selected_payload["joint_assignment_progress_attribution_enabled"] = True
                selected_payload["joint_assignment_target_keys"] = [
                    str(record.get("target_focus_key", "") or "")
                    for record in best_assignment[:8]
                ]
                selected_payload["joint_assignment_canvas_ids"] = [
                    str(record.get("canvas_id", "") or "") for record in best_assignment[:8]
                ]
            if joint_target_execution_plan and len(best_assignment) > 1:
                joint_execution_sites: list[tuple[int, int]] = []
                for record in best_assignment:
                    for site in list(record.get("execution_sites") or []):
                        if isinstance(site, (list, tuple)) and len(site) >= 2:
                            joint_execution_sites.append((int(site[0]), int(site[1])))
                if joint_execution_sites and joint_execution_sites[0] == best_assignment[0]["site"]:
                    selected_payload["execution_sites"] = [
                        [int(site[0]), int(site[1])] for site in joint_execution_sites
                    ]
                    selected_payload["execution_site_count"] = int(len(joint_execution_sites))
                    selected_payload["joint_assignment_execution_enabled"] = True
                    selected_payload["joint_assignment_group_count"] = int(len(best_assignment))
                    selected_payload["joint_assignment_target_keys"] = [
                        str(record.get("target_focus_key", "") or "")
                        for record in best_assignment[:8]
                    ]
                    selected_payload["joint_assignment_canvas_ids"] = [
                        str(record.get("canvas_id", "") or "") for record in best_assignment[:8]
                    ]
            diagnostics["selected_group"] = selected_payload
            joint_diag["selected_assignment_count"] = int(len(best_assignment))
            joint_diag["selected_execution_site_count"] = int(
                selected_payload.get("execution_site_count", 0) or 0
            )
            joint_diag["selected_target_keys"] = [
                str(record.get("target_focus_key", "") or "") for record in best_assignment[:8]
            ]
            joint_diag["selected_canvas_ids"] = [
                str(record.get("canvas_id", "") or "") for record in best_assignment[:8]
            ]
        else:
            if int(diagnostics.get("terminal_completion_coverage_skip_count", 0) or 0) > 0:
                joint_diag["block_reason"] = "terminal_completion_coverage_guard"
                add_skip("terminal_completion_coverage_guard")
            else:
                joint_diag["block_reason"] = "no_joint_assignment_candidate"
        diagnostics["joint_target_assignment"] = joint_diag
    else:
        for canvas in ordered_canvases:
            records = candidate_records_for_canvas(canvas)
            if not records:
                continue
            selected = records[0]
            ordered_sites.append(selected["site"])
            if not diagnostics["selected_group"]:
                diagnostics["selected_group"] = selected_group_payload(selected)
                if terminal_attempted_site_alternatives:
                    diagnostics["terminal_attempted_site_alternative_selected"] = bool(
                        attempted_site_rank(selected) == 0
                        and int(diagnostics.get("attempted_site_candidate_count", 0) or 0) > 0
                    )
                if terminal_execution_order_alternatives:
                    execution_diag = dict(selected.get("execution_diagnostics") or {})
                    diagnostics["terminal_execution_order_alternative_selected"] = bool(
                        execution_diag.get("first_site_reordered_to")
                    )
            break

    deduped: list[tuple[int, int]] = []
    for site in ordered_sites:
        if site not in deduped:
            deduped.append(site)
    diagnostics["first_action_site_count"] = int(len(deduped))
    return deduped, diagnostics


def _bbox_top_left_from_component(row: Mapping[str, Any] | None) -> tuple[int, int] | None:
    box = _bbox(row or {})
    if box is None:
        return None
    return int(box[0]), int(box[1])


def _generate_structured_destination_sets(
    n: int,
    target_x: int,
    target_y: int,
    *,
    board_size: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    board_w, board_h = int(board_size[0]), int(board_size[1])
    max_x = board_w - _R11L_PIECE_W
    max_y = board_h - _R11L_PIECE_H
    sets: list[list[tuple[int, int]]] = []

    def in_bounds(pos: tuple[int, int]) -> bool:
        return 0 <= int(pos[0]) <= max_x and 0 <= int(pos[1]) <= max_y

    def pairwise_ok(positions: Sequence[tuple[int, int]]) -> bool:
        for i, a in enumerate(positions):
            for b in positions[i + 1 :]:
                if abs(int(a[0]) - int(b[0])) <= 2 and abs(int(a[1]) - int(b[1])) <= 2:
                    return False
        return True

    if n == 2:
        for dy in (5, 6, 7, 8):
            sets.append([(target_x, target_y - dy), (target_x, target_y + dy)])
        for dx in (5, 6, 7, 8):
            sets.append([(target_x - dx, target_y), (target_x + dx, target_y)])
        for d in (4, 5, 6):
            sets.append([(target_x - d, target_y - d), (target_x + d, target_y + d)])
            sets.append([(target_x - d, target_y + d), (target_x + d, target_y - d)])
    elif n == 3:
        for dy in (4, 5, 6):
            sets.append([(target_x, target_y - dy), (target_x, target_y), (target_x, target_y + dy)])
        for dx in (4, 5, 6):
            sets.append([(target_x - dx, target_y), (target_x, target_y), (target_x + dx, target_y)])
        for d in (4, 5, 6):
            sets.append([(target_x, target_y - d), (target_x - d, target_y + d), (target_x + d, target_y)])
    elif n == 4:
        for d in (3, 4, 5, 6, 7):
            sets.append(
                [
                    (target_x - d, target_y - d),
                    (target_x + d, target_y - d),
                    (target_x - d, target_y + d),
                    (target_x + d, target_y + d),
                ]
            )
        for dx in (3, 4, 5):
            for dy in (3, 4, 5):
                sets.append(
                    [
                        (target_x - dx, target_y - dy),
                        (target_x + dx, target_y - dy),
                        (target_x - dx, target_y + dy),
                        (target_x + dx, target_y + dy),
                    ]
                )
        for d in (3, 4, 5):
            sets.append(
                [
                    (target_x, target_y - 3 * d),
                    (target_x, target_y - d),
                    (target_x, target_y + d),
                    (target_x, target_y + 3 * d),
                ]
            )
            sets.append(
                [
                    (target_x - 3 * d, target_y),
                    (target_x - d, target_y),
                    (target_x + d, target_y),
                    (target_x + 3 * d, target_y),
                ]
            )
    elif n == 5:
        for d in (4, 5, 6, 7, 8):
            sets.append(
                [
                    (target_x, target_y),
                    (target_x - d, target_y),
                    (target_x + d, target_y),
                    (target_x, target_y - d),
                    (target_x, target_y + d),
                ]
            )
            sets.append(
                [
                    (target_x, target_y),
                    (target_x - d, target_y - d),
                    (target_x + d, target_y - d),
                    (target_x - d, target_y + d),
                    (target_x + d, target_y + d),
                ]
            )
        for d in (3, 4, 5):
            sets.append(
                [
                    (target_x - 2 * d, target_y),
                    (target_x - d, target_y),
                    (target_x, target_y),
                    (target_x + d, target_y),
                    (target_x + 2 * d, target_y),
                ]
            )
            sets.append(
                [
                    (target_x, target_y - 2 * d),
                    (target_x, target_y - d),
                    (target_x, target_y),
                    (target_x, target_y + d),
                    (target_x, target_y + 2 * d),
                ]
            )
    elif n == 6:
        for dx in (4, 5, 6, 7, 8):
            for dy in (4, 5, 6, 7, 8):
                sets.append(
                    [
                        (target_x - dx, target_y - dy),
                        (target_x + dx, target_y - dy),
                        (target_x - dx, target_y),
                        (target_x + dx, target_y),
                        (target_x - dx, target_y + dy),
                        (target_x + dx, target_y + dy),
                    ]
                )
                sets.append(
                    [
                        (target_x - dx, target_y - dy),
                        (target_x, target_y - dy),
                        (target_x + dx, target_y - dy),
                        (target_x - dx, target_y + dy),
                        (target_x, target_y + dy),
                        (target_x + dx, target_y + dy),
                    ]
                )
        for d in (4, 5, 6):
            sets.append(
                [
                    (target_x - 3 * d, target_y),
                    (target_x - 2 * d, target_y),
                    (target_x - d, target_y),
                    (target_x + d, target_y),
                    (target_x + 2 * d, target_y),
                    (target_x + 3 * d, target_y),
                ]
            )
            sets.append(
                [
                    (target_x, target_y - 3 * d),
                    (target_x, target_y - 2 * d),
                    (target_x, target_y - d),
                    (target_x, target_y + d),
                    (target_x, target_y + 2 * d),
                    (target_x, target_y + 3 * d),
                ]
            )

    valid: list[list[tuple[int, int]]] = []
    for positions in sets:
        if not all(in_bounds(pos) for pos in positions):
            continue
        if not pairwise_ok(positions):
            continue
        sx = sum(int(pos[0]) + _R11L_PIECE_W // 2 for pos in positions)
        sy = sum(int(pos[1]) + _R11L_PIECE_H // 2 for pos in positions)
        canvas_x = sx // n - _R11L_CANVAS_W // 2
        canvas_y = sy // n - _R11L_CANVAS_H // 2
        if canvas_x == target_x and canvas_y == target_y:
            valid.append([(int(pos[0]), int(pos[1])) for pos in positions])
    return valid


def _solve_structured_group_destinations(
    pieces: Sequence[tuple[int, int]],
    target_top_left: tuple[int, int],
    *,
    piece_obstacle_cells: set[tuple[int, int]],
    canvas_obstacle_cells: set[tuple[int, int]],
    obstacle_piece_tops: Sequence[tuple[int, int]],
    board_size: tuple[int, int],
    allow_bfs_fallback: bool,
    allow_large_group_templates: bool = False,
    allow_single_piece_target: bool = False,
) -> tuple[list[tuple[int, int]] | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "piece_count": int(len(pieces)),
        "large_group_templates_allowed": bool(allow_large_group_templates),
        "single_piece_target_allowed": bool(allow_single_piece_target),
        "destination_sets_evaluated": 0,
        "permutations_evaluated": 0,
        "bfs_fallback_attempted": False,
        "bfs_fallback_found": False,
        "bfs_fallback_states_visited": 0,
        "failed_reason": "",
    }
    n = len(pieces)
    max_supported = 6 if allow_large_group_templates else 4
    if n == 1 and allow_single_piece_target:
        dest = (int(target_top_left[0]), int(target_top_left[1]))
        diagnostics["destination_sets_evaluated"] = 1
        diagnostics["permutations_evaluated"] = 1
        if not _r11l_piece_safe(dest, piece_obstacle_cells):
            diagnostics["failed_reason"] = "single_piece_destination_unsafe"
            return None, diagnostics
        if not _r11l_canvas_safe(dest, canvas_obstacle_cells):
            diagnostics["failed_reason"] = "single_piece_canvas_destination_unsafe"
            return None, diagnostics
        if not _destinations_clear_obstacle_piece_clicks([dest], obstacle_piece_tops):
            diagnostics["failed_reason"] = "single_piece_destination_click_obstructed"
            return None, diagnostics
        return [dest], diagnostics
    if n < 2 or n > max_supported:
        diagnostics["failed_reason"] = "unsupported_piece_count"
        return None, diagnostics
    destination_sets = _generate_structured_destination_sets(
        n,
        int(target_top_left[0]),
        int(target_top_left[1]),
        board_size=board_size,
    )
    diagnostics["destination_sets_evaluated"] = int(len(destination_sets))
    for destinations in destination_sets:
        result, perm_count = _find_safe_structured_permutation(
            list(pieces),
            destinations,
            piece_obstacle_cells=piece_obstacle_cells,
            canvas_obstacle_cells=canvas_obstacle_cells,
        )
        diagnostics["permutations_evaluated"] += int(perm_count)
        if result is not None and _destinations_clear_obstacle_piece_clicks(
            result,
            obstacle_piece_tops,
        ):
            return result, diagnostics
    if not allow_bfs_fallback:
        diagnostics["failed_reason"] = "no_safe_structured_destination_set_without_bfs"
        return None, diagnostics
    diagnostics["bfs_fallback_attempted"] = True
    bfs_result, bfs_diag = _bfs_structured_canvas_path(
        list(pieces),
        target_top_left,
        piece_obstacle_cells=piece_obstacle_cells,
        canvas_obstacle_cells=canvas_obstacle_cells,
        board_size=board_size,
    )
    diagnostics["bfs_fallback_states_visited"] = int(bfs_diag.get("states_visited", 0) or 0)
    if bfs_result is not None and _destinations_clear_obstacle_piece_clicks(
        bfs_result,
        obstacle_piece_tops,
    ):
        diagnostics["bfs_fallback_found"] = True
        return bfs_result, diagnostics
    diagnostics["failed_reason"] = "no_safe_structured_destination_set"
    return None, diagnostics


def _find_safe_structured_permutation(
    pieces: list[tuple[int, int]],
    destinations: list[tuple[int, int]],
    *,
    piece_obstacle_cells: set[tuple[int, int]],
    canvas_obstacle_cells: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]] | None, int]:
    from itertools import permutations

    n = len(pieces)
    evaluated = 0
    for perm in permutations(range(n)):
        evaluated += 1
        ok = True
        for moved_count in range(1, n + 1):
            sx = sum(destinations[perm[i]][0] + _R11L_PIECE_W // 2 for i in range(moved_count))
            sx += sum(pieces[i][0] + _R11L_PIECE_W // 2 for i in range(moved_count, n))
            sy = sum(destinations[perm[i]][1] + _R11L_PIECE_H // 2 for i in range(moved_count))
            sy += sum(pieces[i][1] + _R11L_PIECE_H // 2 for i in range(moved_count, n))
            canvas_pos = (
                sx // n - _R11L_CANVAS_W // 2,
                sy // n - _R11L_CANVAS_H // 2,
            )
            if not _r11l_canvas_safe(canvas_pos, canvas_obstacle_cells):
                ok = False
                break
            dest = destinations[perm[moved_count - 1]]
            if not _r11l_piece_safe(dest, piece_obstacle_cells):
                ok = False
                break
        if ok:
            return [destinations[perm[i]] for i in range(n)], evaluated
    return None, evaluated


def _bfs_structured_canvas_path(
    pieces: list[tuple[int, int]],
    target_top_left: tuple[int, int],
    *,
    piece_obstacle_cells: set[tuple[int, int]],
    canvas_obstacle_cells: set[tuple[int, int]],
    board_size: tuple[int, int],
) -> tuple[list[tuple[int, int]] | None, dict[str, Any]]:
    n = len(pieces)
    diagnostics = {"states_visited": 0}
    if n < 2:
        return None, diagnostics
    board_w = min(int(board_size[0]), _R11L_STRUCTURED_BFS_GRID_SIZE)
    board_h = min(int(board_size[1]), _R11L_STRUCTURED_BFS_GRID_SIZE)
    max_x = board_w - _R11L_PIECE_W
    max_y = board_h - _R11L_PIECE_H
    init_sx = sum(pos[0] + _R11L_PIECE_W // 2 for pos in pieces)
    init_sy = sum(pos[1] + _R11L_PIECE_H // 2 for pos in pieces)
    target_x, target_y = int(target_top_left[0]), int(target_top_left[1])
    target_sx_min = n * (target_x + _R11L_CANVAS_W // 2)
    target_sx_max = target_sx_min + n - 1
    target_sy_min = n * (target_y + _R11L_CANVAS_H // 2)
    target_sy_max = target_sy_min + n - 1

    def canvas_safe_from_sum(total_x: int, total_y: int) -> bool:
        canvas_pos = (
            int(total_x) // n - _R11L_CANVAS_W // 2,
            int(total_y) // n - _R11L_CANVAS_H // 2,
        )
        return _r11l_canvas_safe(canvas_pos, canvas_obstacle_cells)

    if not canvas_safe_from_sum(init_sx, init_sy):
        return None, diagnostics

    start = (0, init_sx, init_sy)
    visited: dict[tuple[int, int, int], tuple[tuple[int, int, int], tuple[int, int]] | None] = {start: None}
    queue: deque[tuple[int, int, int]] = deque([start])
    goal: tuple[int, int, int] | None = None
    while queue:
        moved_count, total_x, total_y = queue.popleft()
        diagnostics["states_visited"] += 1
        if moved_count == n:
            if (
                target_sx_min <= total_x <= target_sx_max
                and target_sy_min <= total_y <= target_sy_max
            ):
                goal = (moved_count, total_x, total_y)
                break
            continue

        old_pos = pieces[moved_count]
        old_center_x = old_pos[0] + _R11L_PIECE_W // 2
        old_center_y = old_pos[1] + _R11L_PIECE_H // 2
        if moved_count == n - 1:
            for wanted_x in range(target_sx_min, target_sx_max + 1):
                new_x = wanted_x - (total_x - old_center_x) - _R11L_PIECE_W // 2
                if not (0 <= new_x <= max_x):
                    continue
                for wanted_y in range(target_sy_min, target_sy_max + 1):
                    new_y = wanted_y - (total_y - old_center_y) - _R11L_PIECE_H // 2
                    if not (0 <= new_y <= max_y):
                        continue
                    new_pos = (int(new_x), int(new_y))
                    if not _r11l_piece_safe(new_pos, piece_obstacle_cells):
                        continue
                    new_state = (moved_count + 1, int(wanted_x), int(wanted_y))
                    if new_state in visited:
                        continue
                    visited[new_state] = ((moved_count, total_x, total_y), new_pos)
                    queue.append(new_state)
        else:
            for new_x in range(max_x + 1):
                new_center_x = new_x + _R11L_PIECE_W // 2
                next_total_x = total_x - old_center_x + new_center_x
                canvas_x = next_total_x // n - _R11L_CANVAS_W // 2
                if not (0 <= canvas_x <= board_w - _R11L_CANVAS_W):
                    continue
                for new_y in range(max_y + 1):
                    new_pos = (int(new_x), int(new_y))
                    if not _r11l_piece_safe(new_pos, piece_obstacle_cells):
                        continue
                    new_center_y = new_y + _R11L_PIECE_H // 2
                    next_total_y = total_y - old_center_y + new_center_y
                    if not canvas_safe_from_sum(next_total_x, next_total_y):
                        continue
                    new_state = (moved_count + 1, int(next_total_x), int(next_total_y))
                    if new_state in visited:
                        continue
                    visited[new_state] = ((moved_count, total_x, total_y), new_pos)
                    queue.append(new_state)

    if goal is None:
        return None, diagnostics
    path: list[tuple[int, int]] = []
    state: tuple[int, int, int] | None = goal
    while state is not None:
        entry = visited.get(state)
        if entry is None:
            break
        parent, new_pos = entry
        path.append(new_pos)
        state = parent
    path.reverse()
    if len(path) != n:
        return None, diagnostics
    return path, diagnostics


def _r11l_canvas_safe(pos: tuple[int, int], obstacle_cells: set[tuple[int, int]]) -> bool:
    x, y = int(pos[0]), int(pos[1])
    if not (0 <= x <= _R11L_GRID_SIZE - _R11L_CANVAS_W and 0 <= y <= _R11L_GRID_SIZE - _R11L_CANVAS_H):
        return False
    return not any((x + dx, y + dy) in obstacle_cells for dx, dy in _R11L_CANVAS_PATTERN)


def _r11l_piece_safe(pos: tuple[int, int], obstacle_cells: set[tuple[int, int]]) -> bool:
    x, y = int(pos[0]), int(pos[1])
    if not (0 <= x <= _R11L_GRID_SIZE - _R11L_PIECE_W and 0 <= y <= _R11L_GRID_SIZE - _R11L_PIECE_H):
        return False
    return not any((x + dx, y + dy) in obstacle_cells for dx, dy in _R11L_PIECE_PATTERN)


def _destinations_clear_obstacle_piece_clicks(
    destinations: Sequence[tuple[int, int]],
    obstacle_piece_tops: Sequence[tuple[int, int]],
) -> bool:
    for dest in destinations:
        click = (int(dest[0]) + _R11L_PIECE_W // 2, int(dest[1]) + _R11L_PIECE_H // 2)
        for obstacle in obstacle_piece_tops:
            if obstacle in destinations:
                continue
            if (
                int(obstacle[0]) <= click[0] < int(obstacle[0]) + _R11L_PIECE_W
                and int(obstacle[1]) <= click[1] < int(obstacle[1]) + _R11L_PIECE_H
            ):
                return False
    return True


def _first_action_site_for_destination_plan(
    group_pieces: Sequence[Mapping[str, Any]],
    group_piece_tops: Sequence[tuple[int, int]],
    destinations: Sequence[tuple[int, int]],
    *,
    active_id: str,
    active_top: tuple[int, int] | None,
    all_piece_tops: Sequence[tuple[int, int]],
) -> tuple[int, int] | None:
    if not group_piece_tops or not destinations:
        return None
    active_index: int | None = None
    if active_top is not None:
        for index, piece in enumerate(group_pieces):
            if str(piece.get("id", "") or "") == active_id and index < len(destinations):
                active_index = index
                dest = destinations[index]
                site = (int(dest[0]) + _R11L_PIECE_W // 2, int(dest[1]) + _R11L_PIECE_H // 2)
                if not _click_inside_piece_tops(site, all_piece_tops):
                    return site
                break
    for index, piece_top in enumerate(group_piece_tops):
        if active_index is not None and index == active_index:
            continue
        site = (
            int(piece_top[0]) + _R11L_PIECE_W // 2,
            int(piece_top[1]) + _R11L_PIECE_H // 2,
        )
        if _piece_top_at_site(site, all_piece_tops) == (int(piece_top[0]), int(piece_top[1])):
            return site
    return None


def _r11l_execution_order_canvas_safe(
    group_piece_tops: Sequence[tuple[int, int]],
    destinations: Sequence[tuple[int, int]],
    execution_order: Sequence[int],
    *,
    canvas_obstacle_cells: set[tuple[int, int]],
) -> tuple[bool, dict[str, Any]]:
    current_tops = [(int(x), int(y)) for x, y in group_piece_tops]
    destination_tops = [(int(x), int(y)) for x, y in destinations]
    n = len(current_tops)
    diagnostics: dict[str, Any] = {
        "checked": True,
        "piece_count": int(n),
        "destination_count": int(len(destination_tops)),
        "execution_order": [],
        "safe": False,
        "failed_reason": "",
        "safe_prefix_count": 0,
        "canvas_positions": [],
    }
    if n == 0 or n != len(destination_tops):
        diagnostics["failed_reason"] = "piece_destination_count_mismatch"
        return False, diagnostics
    try:
        order = [int(index) for index in list(execution_order)]
    except (TypeError, ValueError):
        diagnostics["failed_reason"] = "invalid_execution_order"
        return False, diagnostics
    if not order:
        order = list(range(n))
    if sorted(order) != list(range(n)):
        diagnostics["execution_order"] = [int(index) for index in order]
        diagnostics["failed_reason"] = "execution_order_not_permutation"
        return False, diagnostics
    diagnostics["execution_order"] = [int(index) for index in order]

    for step_index, piece_index in enumerate(order):
        current_tops[int(piece_index)] = destination_tops[int(piece_index)]
        sx = sum(int(pos[0]) + _R11L_PIECE_W // 2 for pos in current_tops)
        sy = sum(int(pos[1]) + _R11L_PIECE_H // 2 for pos in current_tops)
        canvas_pos = (
            sx // n - _R11L_CANVAS_W // 2,
            sy // n - _R11L_CANVAS_H // 2,
        )
        diagnostics["canvas_positions"].append(
            [int(canvas_pos[0]), int(canvas_pos[1])]
        )
        if not _r11l_canvas_safe(canvas_pos, canvas_obstacle_cells):
            diagnostics.update(
                {
                    "failed_reason": "canvas_obstacle_collision",
                    "failed_step_index": int(step_index),
                    "failed_piece_index": int(piece_index),
                    "failed_canvas_top_left": [
                        int(canvas_pos[0]),
                        int(canvas_pos[1]),
                    ],
                }
            )
            return False, diagnostics
        diagnostics["safe_prefix_count"] = int(step_index + 1)

    diagnostics["safe"] = True
    return True, diagnostics


def _execution_sites_for_destination_plan(
    group_pieces: Sequence[Mapping[str, Any]],
    group_piece_tops: Sequence[tuple[int, int]],
    destinations: Sequence[tuple[int, int]],
    *,
    active_id: str,
    active_top: tuple[int, int] | None,
    all_piece_tops: Sequence[tuple[int, int]],
    all_pieces: Sequence[Mapping[str, Any]] | None = None,
    piece_obstacle_cells: set[tuple[int, int]] | None = None,
    canvas_obstacle_cells: set[tuple[int, int]] | None = None,
    avoid_first_sites: set[tuple[int, int]] | None = None,
    allow_full_order_permutations: bool = False,
    enable_obstacle_ownership_diagnostics: bool = False,
    enable_self_overlap_detour: bool = False,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "piece_count": int(len(group_piece_tops)),
        "destination_count": int(len(destinations)),
        "active_piece_in_group": False,
        "failed_reason": "",
        "execution_order": [],
        "order_alternative_count": 0,
        "full_order_permutations_enabled": bool(allow_full_order_permutations),
        "full_order_permutation_count": 0,
        "obstacle_ownership_diagnostics_enabled": bool(
            enable_obstacle_ownership_diagnostics
        ),
        "self_overlap_detour_enabled": bool(enable_self_overlap_detour),
        "self_overlap_detour_count": 0,
        "first_site_reordered_to": [],
    }
    if not group_piece_tops or not destinations:
        diagnostics["failed_reason"] = "empty_destination_plan"
        return [], diagnostics
    if len(group_piece_tops) != len(destinations):
        diagnostics["failed_reason"] = "piece_destination_count_mismatch"
        return [], diagnostics

    current_group_tops = [(int(x), int(y)) for x, y in group_piece_tops]
    current_all_tops = [(int(x), int(y)) for x, y in all_piece_tops]
    destination_tops = [(int(x), int(y)) for x, y in destinations]
    current_active_top = (
        (int(active_top[0]), int(active_top[1]))
        if active_top is not None
        else None
    )
    piece_obstacle_cells = set(piece_obstacle_cells or set())
    canvas_obstacle_cells = set(canvas_obstacle_cells or set())
    group_ids = {str(piece.get("id", "") or "") for piece in group_pieces}
    group_index_by_id = {
        str(piece.get("id", "") or ""): int(index)
        for index, piece in enumerate(group_pieces)
        if str(piece.get("id", "") or "")
    }

    def initial_piece_entries() -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if all_pieces is not None:
            for idx, piece in enumerate(all_pieces):
                top = _bbox_top_left_from_component(piece)
                if top is None:
                    continue
                piece_id = str(piece.get("id", "") or "")
                entries.append(
                    {
                        "index": int(idx),
                        "id": piece_id,
                        "role": str(piece.get("role", "") or ""),
                        "colors": sorted(int(color) for color in _component_colors(piece)),
                        "top_left": (int(top[0]), int(top[1])),
                        "bbox": list(_bbox(piece) or ()),
                        "in_group": bool(piece_id and piece_id in group_ids),
                        "group_index": int(group_index_by_id[piece_id])
                        if piece_id in group_index_by_id
                        else -1,
                        "is_active": bool(piece_id and piece_id == active_id),
                    }
                )
        else:
            for idx, top in enumerate(all_piece_tops):
                entries.append(
                    {
                        "index": int(idx),
                        "id": "",
                        "role": "",
                        "colors": [],
                        "top_left": (int(top[0]), int(top[1])),
                        "bbox": [],
                        "in_group": bool(
                            (int(top[0]), int(top[1])) in set(current_group_tops)
                        ),
                        "group_index": -1,
                        "is_active": bool(
                            current_active_top == (int(top[0]), int(top[1]))
                        ),
                    }
                )
        return entries

    def entry_at_site(
        site: tuple[int, int],
        entries: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any] | None:
        x, y = int(site[0]), int(site[1])
        for entry in entries:
            raw_top = entry.get("top_left")
            if not isinstance(raw_top, (list, tuple)) or len(raw_top) < 2:
                continue
            top = (int(raw_top[0]), int(raw_top[1]))
            if top[0] <= x < top[0] + _R11L_PIECE_W and top[1] <= y < top[1] + _R11L_PIECE_H:
                return entry
        return None

    def compact_entry(entry: Mapping[str, Any] | None) -> dict[str, Any]:
        if not enable_obstacle_ownership_diagnostics or entry is None:
            return {}
        raw_top = entry.get("top_left")
        top = (
            [int(raw_top[0]), int(raw_top[1])]
            if isinstance(raw_top, (list, tuple)) and len(raw_top) >= 2
            else []
        )
        raw_bbox = list(entry.get("bbox") or [])
        return {
            "id": str(entry.get("id", "") or ""),
            "role": str(entry.get("role", "") or ""),
            "colors": [int(color) for color in list(entry.get("colors") or [])],
            "top_left": top,
            "bbox": [int(value) for value in raw_bbox[:4]],
            "in_group": bool(entry.get("in_group")),
            "group_index": int(entry.get("group_index", -1) or -1),
            "is_active": bool(entry.get("is_active")),
        }

    def canvas_safe_for_tops(candidate_group_tops: Sequence[tuple[int, int]]) -> bool:
        if not canvas_obstacle_cells:
            return True
        n = len(candidate_group_tops)
        if n <= 0:
            return False
        sx = sum(int(pos[0]) + _R11L_PIECE_W // 2 for pos in candidate_group_tops)
        sy = sum(int(pos[1]) + _R11L_PIECE_H // 2 for pos in candidate_group_tops)
        canvas_pos = (
            sx // n - _R11L_CANVAS_W // 2,
            sy // n - _R11L_CANVAS_H // 2,
        )
        return _r11l_canvas_safe(canvas_pos, canvas_obstacle_cells)

    def find_self_overlap_detour_top(
        *,
        piece_index: int,
        piece_top: tuple[int, int],
        destination: tuple[int, int],
        local_group_tops: Sequence[tuple[int, int]],
        local_all_tops: Sequence[tuple[int, int]],
    ) -> tuple[int, int] | None:
        if not enable_self_overlap_detour:
            return None
        offsets = [
            (-7, 0),
            (7, 0),
            (0, -7),
            (0, 7),
            (-10, 0),
            (10, 0),
            (0, -10),
            (0, 10),
            (-7, -7),
            (-7, 7),
            (7, -7),
            (7, 7),
        ]
        final_click = (
            int(destination[0]) + _R11L_PIECE_W // 2,
            int(destination[1]) + _R11L_PIECE_H // 2,
        )
        for dx, dy in offsets:
            detour = (int(piece_top[0]) + int(dx), int(piece_top[1]) + int(dy))
            if not _r11l_piece_safe(detour, piece_obstacle_cells):
                continue
            detour_click = (
                int(detour[0]) + _R11L_PIECE_W // 2,
                int(detour[1]) + _R11L_PIECE_H // 2,
            )
            if _click_inside_piece_tops(detour_click, local_all_tops):
                continue
            tops_after_detour = [
                detour if pos == piece_top else pos for pos in local_all_tops
            ]
            if _click_inside_piece_tops(final_click, tops_after_detour):
                continue
            group_after_detour = list(local_group_tops)
            group_after_detour[int(piece_index)] = detour
            if not canvas_safe_for_tops(group_after_detour):
                continue
            group_after_final = list(group_after_detour)
            group_after_final[int(piece_index)] = destination
            if not canvas_safe_for_tops(group_after_final):
                continue
            return detour
        return None

    active_index: int | None = None
    for index, piece in enumerate(group_pieces):
        if str(piece.get("id", "") or "") == active_id:
            active_index = index
            break
    if active_index is not None and active_index >= len(current_group_tops):
        active_index = None
    diagnostics["active_piece_in_group"] = bool(active_index is not None)

    base_order: list[int] = []
    if active_index is not None:
        base_order.append(active_index)
    base_order.extend(index for index in range(len(current_group_tops)) if index != active_index)
    avoid_first_sites = set(avoid_first_sites or set())
    orders: list[list[int]] = [list(base_order)]
    if avoid_first_sites and len(current_group_tops) > 1:
        for index in range(len(current_group_tops)):
            alt = [index]
            alt.extend(other for other in range(len(current_group_tops)) if other != index)
            if alt not in orders:
                orders.append(alt)
    if allow_full_order_permutations and 1 < len(current_group_tops) <= 4:
        from itertools import permutations

        for perm in permutations(range(len(current_group_tops))):
            alt = [int(index) for index in perm]
            if alt not in orders:
                orders.append(alt)
    diagnostics["order_alternative_count"] = int(max(0, len(orders) - 1))
    diagnostics["full_order_permutation_count"] = int(
        max(0, len(orders) - 1)
        if allow_full_order_permutations and 1 < len(current_group_tops) <= 4
        else 0
    )

    failures: list[dict[str, Any]] = []

    def simulate_order(order: Sequence[int]) -> tuple[list[tuple[int, int]], dict[str, Any]]:
        local_group_tops = [(int(x), int(y)) for x, y in group_piece_tops]
        local_all_tops = [(int(x), int(y)) for x, y in all_piece_tops]
        local_entries = initial_piece_entries()
        local_active_top = (
            (int(active_top[0]), int(active_top[1]))
            if active_top is not None
            else None
        )
        local_diag = {
            "execution_order": [int(index) for index in order],
            "failed_reason": "",
        }
        local_sites: list[tuple[int, int]] = []
        for index in order:
            piece_top = local_group_tops[index]
            destination = destination_tops[index]
            if piece_top == destination:
                continue
            if local_active_top != piece_top:
                select_site = (
                    int(piece_top[0]) + _R11L_PIECE_W // 2,
                    int(piece_top[1]) + _R11L_PIECE_H // 2,
                )
                if _piece_top_at_site(select_site, local_all_tops) != piece_top:
                    local_diag.update(
                        {
                            "failed_reason": "selection_site_does_not_select_piece",
                            "failed_index": int(index),
                            "failed_site": [int(select_site[0]), int(select_site[1])],
                        }
                    )
                    return [], local_diag
                local_sites.append(select_site)
                local_active_top = piece_top

            move_site = (
                int(destination[0]) + _R11L_PIECE_W // 2,
                int(destination[1]) + _R11L_PIECE_H // 2,
            )
            piece_top_for_update = piece_top
            if _click_inside_piece_tops(move_site, local_all_tops):
                blocking_entry = entry_at_site(move_site, local_entries)
                blocking_group_index = (
                    int(blocking_entry.get("group_index", -1) or -1)
                    if blocking_entry is not None
                    else -1
                )
                if blocking_group_index == int(index):
                    detour_top = find_self_overlap_detour_top(
                        piece_index=int(index),
                        piece_top=piece_top,
                        destination=destination,
                        local_group_tops=local_group_tops,
                        local_all_tops=local_all_tops,
                    )
                    if detour_top is not None:
                        detour_site = (
                            int(detour_top[0]) + _R11L_PIECE_W // 2,
                            int(detour_top[1]) + _R11L_PIECE_H // 2,
                        )
                        local_sites.append(detour_site)
                        for entry in local_entries:
                            raw_top = entry.get("top_left")
                            if (
                                isinstance(raw_top, (list, tuple))
                                and len(raw_top) >= 2
                                and (int(raw_top[0]), int(raw_top[1])) == piece_top
                            ):
                                entry["top_left"] = (
                                    int(detour_top[0]),
                                    int(detour_top[1]),
                                )
                                if entry.get("bbox"):
                                    entry["bbox"] = [
                                        int(detour_top[0]),
                                        int(detour_top[1]),
                                        int(detour_top[0]) + _R11L_PIECE_W - 1,
                                        int(detour_top[1]) + _R11L_PIECE_H - 1,
                                    ]
                                break
                        local_all_tops = [
                            detour_top if top == piece_top else top
                            for top in local_all_tops
                        ]
                        local_group_tops[index] = detour_top
                        local_active_top = detour_top
                        piece_top_for_update = detour_top
                        local_diag["self_overlap_detour_count"] = int(
                            local_diag.get("self_overlap_detour_count", 0) or 0
                        ) + 1
                        local_diag["self_overlap_detour"] = {
                            "piece_index": int(index),
                            "from_top_left": [int(piece_top[0]), int(piece_top[1])],
                            "detour_top_left": [
                                int(detour_top[0]),
                                int(detour_top[1]),
                            ],
                            "detour_site": [
                                int(detour_site[0]),
                                int(detour_site[1]),
                            ],
                            "final_destination_top_left": [
                                int(destination[0]),
                                int(destination[1]),
                            ],
                            "final_site": [
                                int(move_site[0]),
                                int(move_site[1]),
                            ],
                        }
                    else:
                        detour_top = None
                else:
                    detour_top = None
                if detour_top is not None:
                    pass
                else:
                    local_diag.update(
                        {
                            "failed_reason": "move_site_hits_existing_piece",
                            "failed_index": int(index),
                            "failed_site": [int(move_site[0]), int(move_site[1])],
                        }
                    )
                    if enable_obstacle_ownership_diagnostics:
                        local_diag["blocking_piece"] = compact_entry(blocking_entry)
                    return [], local_diag
            if not _click_inside_piece_tops(move_site, local_all_tops):
                local_sites.append(move_site)
            else:
                local_diag.update(
                    {
                        "failed_reason": "move_site_hits_existing_piece",
                        "failed_index": int(index),
                        "failed_site": [int(move_site[0]), int(move_site[1])],
                    }
                )
                if enable_obstacle_ownership_diagnostics:
                    local_diag["blocking_piece"] = compact_entry(
                        entry_at_site(move_site, local_entries)
                    )
                return [], local_diag

            for entry in local_entries:
                raw_top = entry.get("top_left")
                if (
                    isinstance(raw_top, (list, tuple))
                    and len(raw_top) >= 2
                    and (int(raw_top[0]), int(raw_top[1])) == piece_top_for_update
                ):
                    entry["top_left"] = (int(destination[0]), int(destination[1]))
                    if entry.get("bbox"):
                        entry["bbox"] = [
                            int(destination[0]),
                            int(destination[1]),
                            int(destination[0]) + _R11L_PIECE_W - 1,
                            int(destination[1]) + _R11L_PIECE_H - 1,
                        ]
                    break
            local_all_tops = [
                destination if top == piece_top_for_update else top
                for top in local_all_tops
            ]
            local_group_tops[index] = destination
            local_active_top = destination

        if not local_sites:
            local_diag["failed_reason"] = "no_required_moves"
            return [], local_diag
        local_diag["site_count"] = int(len(local_sites))
        return local_sites, local_diag

    best_sites: list[tuple[int, int]] = []
    best_diag: dict[str, Any] = {}
    for order in orders:
        sites, order_diag = simulate_order(order)
        if not sites:
            failures.append(order_diag)
            continue
        first = (int(sites[0][0]), int(sites[0][1]))
        if not best_sites:
            best_sites = sites
            best_diag = order_diag
        if first not in avoid_first_sites:
            best_sites = sites
            best_diag = order_diag
            break
    if not best_sites:
        if failures:
            diagnostics.update(failures[0])
        return [], diagnostics
    diagnostics.update(best_diag)
    if best_sites and base_order != list(diagnostics.get("execution_order") or []):
        diagnostics["first_site_reordered_to"] = [
            int(best_sites[0][0]),
            int(best_sites[0][1]),
        ]
    diagnostics["site_count"] = int(len(best_sites))
    return best_sites, diagnostics


def _click_inside_piece_tops(
    site: tuple[int, int],
    piece_tops: Sequence[tuple[int, int]],
) -> bool:
    x, y = int(site[0]), int(site[1])
    return any(
        int(px) <= x < int(px) + _R11L_PIECE_W and int(py) <= y < int(py) + _R11L_PIECE_H
        for px, py in piece_tops
    )


def _piece_top_at_site(
    site: tuple[int, int],
    piece_tops: Sequence[tuple[int, int]],
) -> tuple[int, int] | None:
    x, y = int(site[0]), int(site[1])
    for px, py in piece_tops:
        top = (int(px), int(py))
        if top[0] <= x < top[0] + _R11L_PIECE_W and top[1] <= y < top[1] + _R11L_PIECE_H:
            return top
    return None


def _find_direct_action_keys(value: Any, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_s = str(key)
            path = f"{prefix}.{key_s}" if prefix else key_s
            if key_s in DIRECT_ACTION_KEYS:
                found.append(path)
            found.extend(_find_direct_action_keys(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_direct_action_keys(item, prefix=f"{prefix}[{index}]"))
    return found


def _parse_coordinate_provider_response(
    raw: Any,
    diagnostics: dict[str, Any],
) -> Any:
    diagnostics["raw_response_kind"] = type(raw).__name__
    if not isinstance(raw, (str, bytes, bytearray)):
        return raw
    try:
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        parsed = json.loads(text)
    except Exception as exc:
        diagnostics["json_response_parse_error"] = f"{type(exc).__name__}: {exc}"
        return None
    diagnostics["json_response_parsed"] = True
    if isinstance(parsed, Mapping):
        diagnostics["parsed_response_keys"] = sorted(str(key) for key in parsed.keys())
    return parsed


def _extract_coordinate_hypothesis_items(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        for key in ("coordinate_hypotheses", "hypotheses", "candidates"):
            value = raw.get(key)
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
                return list(value)
        if "coordinate_hypothesis" in raw:
            return [raw]
        return []
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        return list(raw)
    return []


def _coerce_coordinate_hypothesis(item: Any) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(item, Mapping):
        return None, "non_mapping_hypothesis"
    row = _sanitize(item)
    if not isinstance(row, Mapping):
        return None, "non_mapping_hypothesis"
    schema_version = str(row.get("schema_version", "") or "")
    if schema_version != COORDINATE_HYPOTHESIS_SCHEMA_VERSION:
        return None, "schema_version_mismatch"
    if _find_direct_action_keys(row):
        return dict(row), ""
    if str(row.get("coordinate_hypothesis", "") or "") not in ALLOWED_COORDINATE_HYPOTHESES:
        return dict(row), ""
    if str(row.get("candidate_generation", "") or "") != "solve_missing_piece_position":
        return dict(row), ""
    return dict(row), ""


def _payload_preview(value: Any, *, max_chars: int = 1200) -> Any:
    sanitized = _sanitize(value)
    try:
        text = json.dumps(sanitized, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError):
        text = str(sanitized)
    if len(text) <= int(max_chars):
        return sanitized
    return {
        "truncated": True,
        "char_count": len(text),
        "preview": text[: int(max_chars)],
    }


def _frame_hash(frame: np.ndarray) -> str:
    arr = np.asarray(frame, dtype=np.int16)
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    return f"sha256:{digest[:16]}"


def _clean_colors(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (set, list, tuple)):
        raw = value
    else:
        raw = (value,)
    colors: set[int] = set()
    for item in raw:
        try:
            colors.add(int(item))
        except (TypeError, ValueError):
            continue
    return colors


def _component_colors(component: Mapping[str, Any]) -> set[int]:
    colors = _clean_colors(component.get("colors"))
    colors.update(_clean_colors(component.get("color")))
    return colors


def _colors_overlap(left: set[int], right: set[int]) -> bool:
    return bool(left and right and (left & right))


def _bbox(component: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    raw = component.get("bbox")
    if not isinstance(raw, Sequence) or len(raw) != 4:
        return None
    try:
        return int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
    except (TypeError, ValueError):
        return None


def _centroid(component: Mapping[str, Any]) -> tuple[int, int] | None:
    raw = component.get("centroid")
    if not _is_xy(raw):
        return None
    return int(raw[0]), int(raw[1])


def _is_xy(value: Any) -> bool:
    if not isinstance(value, Sequence) or len(value) < 2:
        return False
    try:
        int(value[0])
        int(value[1])
    except (TypeError, ValueError):
        return False
    return True


def _is_bbox(value: Any) -> bool:
    if not isinstance(value, Sequence) or len(value) < 4:
        return False
    try:
        int(value[0])
        int(value[1])
        int(value[2])
        int(value[3])
    except (TypeError, ValueError):
        return False
    return True


def _bboxes_overlap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def _bbox_overlap_area(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
    if not _bboxes_overlap(left, right):
        return 0
    width = min(left[2], right[2]) - max(left[0], right[0]) + 1
    height = min(left[3], right[3]) - max(left[1], right[1]) + 1
    return max(0, int(width)) * max(0, int(height))


def _rect_touches_cells(
    x: int,
    y: int,
    w: int,
    h: int,
    cells: set[tuple[int, int]],
) -> bool:
    for yy in range(int(y), int(y) + int(h)):
        for xx in range(int(x), int(x) + int(w)):
            if (xx, yy) in cells:
                return True
    return False


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, set):
        return sorted(_sanitize(item) for item in value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_sanitize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _contains_key(value: Any, keys: Iterable[str]) -> bool:
    wanted = {str(key) for key in keys}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in wanted:
                return True
            if _contains_key(item, wanted):
                return True
    elif isinstance(value, list):
        return any(_contains_key(item, wanted) for item in value)
    return False


__all__ = [
    "ALLOWED_COORDINATE_HYPOTHESES",
    "COORDINATE_SENSOR_NAMES",
    "COORDINATE_HYPOTHESIS_SCHEMA_VERSION",
    "CoordinateHypothesisBridge",
    "CoordinateHypothesisBridgeResult",
    "CoordinateHypothesisResult",
    "build_coordinate_hypothesis_prompt_packet",
    "build_r11l_l1_coordinate_sensor_packet",
    "coordinate_hypothesis_preflight",
    "coordinate_hypothesis_prompt_packet_audit",
    "coordinate_hypothesis_response_schema",
    "packet_for_llm",
    "resolve_coordinate_hypothesis_provider_path",
    "verify_coordinate_hypothesis",
]
