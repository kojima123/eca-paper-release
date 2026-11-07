"""Cross-level executable mechanic memory for the Tufa host.

This module compiles existing ECA evidence into one coordinate-free schema
format. A schema has two projections: verified schemas may authorize a
re-bound host planner on later levels, while supported schemas produce a
bounded advisory packet for composition-heavy reasoning.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from .action_effect_learner import ActionEffectLearner, _bbox_bucket, _component_at
from .palette_invariant_perception import extract_structural_components
from .tufa_mechanics.registry import runtime_mechanic_spec


SCHEMA_VERSION = "eca_executable_mechanic_schema.v1"
MEMORY_VERSION = "eca_episode_mechanic_memory.v1"
ADVISORY_VERSION = "eca_mechanic_schema_advisory.v1"
_ASCII_COLOR_CHARS = "WwgGcBMPRbSYOrNp"

OBSERVED = "observed"
SUPPORTED = "supported"
VERIFIED = "verified"

_STATUS_RANK = {OBSERVED: 0, SUPPORTED: 1, VERIFIED: 2}
_FAMILY_GOALS = {
    "mirror": "bring the coupled cursor pair into the terminal contact relation",
    "stride_maze": "route the controllable actor to a visible target",
    "color_cycle": "satisfy visible template colour constraints",
    "sand_filter": "configure routing components, then submit the flow",
    "program_execution": "configure a visible program so its execution matches the target state",
    "linked_extent": "change linked extents until carried targets align with indicators",
    "rule_rewrite": "rotate visible rule entries until the rewrite matches the consequence",
}
_FAMILY_OPERATORS = {
    "mirror": {
        "kind": "coupled_motion",
        "action_family": "directional",
        "transition": "horizontal inputs move paired actors oppositely; vertical inputs move them together",
    },
    "stride_maze": {
        "kind": "lattice_motion",
        "action_family": "directional",
        "transition": "actor moves one inferred lattice stride unless the route is blocked",
    },
    "color_cycle": {
        "kind": "local_state_successor",
        "action_family": "click",
        "transition": "clicked state advances in its inferred palette cycle",
    },
    "sand_filter": {
        "kind": "routing_transform",
        "action_family": "mixed",
        "transition": "one routing component changes toward the target arrangement; submit evaluates flow",
    },
    "program_execution": {
        "kind": "bit_program_execution",
        "action_family": "click_then_submit",
        "transition": "one visible bit toggles or the configured rows execute deterministically",
    },
    "linked_extent": {
        "kind": "linked_extent_growth",
        "action_family": "click",
        "transition": "one bound extent changes and its linked dependants translate",
    },
    "rule_rewrite": {
        "kind": "rule_table_rewrite",
        "action_family": "mixed",
        "transition": "one selected rule symbol rotates in the inferred modular alphabet",
    },
}


def _family_defaults(family: str) -> tuple[dict[str, Any], str] | None:
    """Resolve fallback metadata without maintaining a second family registry."""

    if family in _FAMILY_OPERATORS:
        return copy.deepcopy(_FAMILY_OPERATORS[family]), _FAMILY_GOALS[family]
    spec = runtime_mechanic_spec(family)
    if spec is None:
        return None
    return (
        {
            "kind": spec.mechanic_family,
            "action_family": "runtime_defined",
            "transition": "apply the current runtime schema after rebinding every role",
        },
        f"satisfy the visible goal relation for {spec.mechanic_family}",
    )


def _stable_digest(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): normalize(val)
                for key, val in sorted(item.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        if isinstance(item, (set, frozenset)):
            values = [normalize(val) for val in item]
            return sorted(values, key=lambda val: json.dumps(val, sort_keys=True))
        if isinstance(item, np.generic):
            return item.item()
        return item

    encoded = json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _mirror_positions_after(
    cursor_a: tuple[int, int],
    cursor_b: tuple[int, int],
    action: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    ax, ay = cursor_a
    bx, by = cursor_b
    movement = str(action).upper()
    if movement == "UP":
        return (ax, ay - 1), (bx, by - 1)
    if movement == "DOWN":
        return (ax, ay + 1), (bx, by + 1)
    if movement == "LEFT":
        return (ax - 1, ay), (bx + 1, by)
    if movement == "RIGHT":
        return (ax + 1, ay), (bx - 1, by)
    return None


def _logical_cell_is_value(
    frame: np.ndarray,
    *,
    origin: tuple[int, int],
    cell: int,
    position: tuple[int, int],
    value: int,
) -> bool:
    x, y = position
    ox, oy = origin
    row0, col0 = oy + y * cell, ox + x * cell
    row1, col1 = row0 + cell, col0 + cell
    if row0 < 0 or col0 < 0 or row1 > frame.shape[0] or col1 > frame.shape[1]:
        return False
    patch = frame[row0:row1, col0:col1]
    return bool(patch.size and np.all(patch == int(value)))


@dataclass
class ExecutableMechanicSchema:
    schema_id: str
    mechanic_id: str
    source_family: str
    status: str
    confidence: float
    learned_level: int
    roles: list[str]
    preconditions: list[str]
    operator: dict[str, Any]
    invariants: list[str]
    goal_relation: str
    binding_recipe: dict[str, Any]
    falsifiers: list[str]
    required_bindings: list[str]
    evidence: dict[str, Any] = field(default_factory=dict)
    _evidence_keys: set[str] = field(default_factory=set, repr=False)

    def promote(self, status: str) -> None:
        if _STATUS_RANK.get(status, -1) > _STATUS_RANK.get(self.status, -1):
            self.status = status
        support = int(self.evidence.get("support_count", 0) or 0)
        score = int(self.evidence.get("score_closure_count", 0) or 0)
        exact = int(self.evidence.get("exact_transition_count", 0) or 0)
        self.confidence = min(
            1.0,
            max(
                self.confidence,
                0.45 + 0.12 * min(3, support) + 0.18 * min(1, score) + 0.08 * min(2, exact),
            ),
        )

    def add_evidence(
        self,
        key: str,
        *,
        level: int,
        prediction_available: bool = False,
        observed_transition: bool = False,
        exact_transition: bool = False,
        score_closure: bool = False,
    ) -> None:
        is_new = key not in self._evidence_keys
        if is_new:
            self._evidence_keys.add(key)
            self.evidence["support_count"] = int(
                self.evidence.get("support_count", 0) or 0
            ) + 1
        levels = {int(value) for value in self.evidence.get("levels", [])}
        levels.add(int(level))
        self.evidence["levels"] = sorted(levels)
        if prediction_available and is_new:
            self.evidence["prediction_available_count"] = int(
                self.evidence.get("prediction_available_count", 0) or 0
            ) + 1
        if observed_transition and is_new:
            self.evidence["observed_transition_count"] = int(
                self.evidence.get("observed_transition_count", 0) or 0
            ) + 1
        if exact_transition and is_new:
            self.evidence["exact_transition_count"] = int(
                self.evidence.get("exact_transition_count", 0) or 0
            ) + 1
        if score_closure and is_new:
            self.evidence["score_closure_count"] = int(
                self.evidence.get("score_closure_count", 0) or 0
            ) + 1
            self.promote(VERIFIED)
        elif int(self.evidence.get("support_count", 0) or 0) >= 1:
            self.promote(SUPPORTED)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_evidence_keys", None)
        data["confidence"] = round(float(self.confidence), 4)
        return data


class EpisodeMechanicMemory:
    """Game-scoped memory that survives level changes and resets per game."""

    def __init__(self, *, game_key: str = "", max_advisory_schemas: int = 4) -> None:
        self.game_key = str(game_key)
        self.max_advisory_schemas = max(1, int(max_advisory_schemas))
        self._schemas: dict[str, ExecutableMechanicSchema] = {}
        self._family_schema_ids: dict[str, str] = {}
        self._residual_plans: dict[str, dict[str, Any]] = {}
        self._verified_residual_levels: dict[str, int] = {}
        self._runtime_bindings: dict[str, dict[str, Any]] = {}
        self._action_effect = ActionEffectLearner()
        self._last_level = 1
        self._transition_count = 0
        self._level_transition_count = 0

    @property
    def schemas(self) -> list[ExecutableMechanicSchema]:
        return sorted(
            self._schemas.values(),
            key=lambda item: (
                -_STATUS_RANK.get(item.status, -1),
                -float(item.confidence),
                item.schema_id,
            ),
        )

    def ingest_toolbox_result(
        self, result: Mapping[str, Any] | None, *, level: int
    ) -> None:
        if not isinstance(result, Mapping):
            return
        # Runtime modules own the canonical transferable schema. Ingest them
        # before plans so every registered family uses its compiled schema
        # instead of a second hand-maintained family table.
        for row in result.get("runtime_mechanics", []):
            if not isinstance(row, Mapping):
                continue
            family = str(row.get("toolbox_family", "") or "")
            if _family_defaults(family) is None:
                continue
            binding = row.get("current_binding")
            if isinstance(binding, Mapping):
                self._runtime_bindings[family] = copy.deepcopy(dict(binding))
            if not row.get("supported"):
                continue
            schema = self._schema_from_runtime_mechanic(row, level=level)
            if schema is None:
                continue
            evidence_key = _stable_digest(
                {
                    "family": family,
                    "level": int(level),
                    "schema": row.get("schema"),
                    "binding": binding,
                }
            )
            schema.add_evidence(
                evidence_key,
                level=int(level),
                prediction_available=bool(row.get("prediction_available")),
            )
            self._family_schema_ids[family] = schema.schema_id
        for plan in result.get("plans", []):
            if not isinstance(plan, Mapping) or not plan.get("certified"):
                continue
            family = str(plan.get("family", "") or "")
            if _family_defaults(family) is None:
                continue
            schema = self._schema_from_toolbox_plan(plan, level=level)
            certificate = str(plan.get("certificate_reason", "") or "")
            grounding = plan.get("grounded_observation") or {}
            verification = grounding.get("verification_state") or {}
            evidence_key = _stable_digest(
                {
                    "family": family,
                    "level": int(level),
                    "static": verification.get("static_signature", ""),
                    "certificate": certificate,
                }
            )
            schema.evidence["certificate_reason"] = certificate
            schema.add_evidence(evidence_key, level=level)
            self._family_schema_ids[family] = schema.schema_id
            remembered = copy.deepcopy(dict(plan))
            remembered["_memory_level"] = int(level)
            self._residual_plans[family] = remembered
            self._verified_residual_levels.pop(family, None)

    def observe_transition(
        self,
        *,
        action_name: str,
        action_payload: Mapping[str, Any] | None,
        before_frame: Any,
        after_frame: Any,
        level_before: int,
        level_after: int,
        score_delta: int,
        source_family: str = "",
    ) -> None:
        before = np.asarray(before_frame)
        after = np.asarray(after_frame)
        self._transition_count += 1
        self._action_effect.observe(
            action_name,
            dict(action_payload or {}),
            before,
            after,
        )
        self._compile_click_operators(level=int(level_before))
        family = str(source_family or "")
        self._advance_residual_plan(
            family=family,
            action_name=action_name,
            action_payload=action_payload,
            after_frame=after,
            level=int(level_before),
            score_delta=int(score_delta),
        )
        schema = self._family_schema(family)
        if schema is not None:
            transition_key = _stable_digest(
                {
                    "schema": schema.schema_id,
                    "level": int(level_before),
                    "action": str(action_name),
                    "before": _stable_digest(before.tolist()),
                    "after": _stable_digest(after.tolist()),
                }
            )
            schema.add_evidence(
                transition_key,
                level=int(level_before),
                observed_transition=True,
                score_closure=int(score_delta) > 0,
            )
        elif int(score_delta) > 0 and str(action_name).upper() == "ACTION6":
            for click_schema in self._matching_click_schemas(before, action_payload):
                click_schema.add_evidence(
                    f"score:{self._transition_count}",
                    level=int(level_before),
                    observed_transition=True,
                    score_closure=True,
                )
        if int(level_after) != int(level_before):
            self._level_transition_count += 1
        self._last_level = int(level_after)

    def record_verified_transition(
        self,
        *,
        family: str,
        level: int,
        evidence_key: str,
        diagnostic: Mapping[str, Any] | None = None,
        inherited: bool = False,
    ) -> None:
        """Record a post-action successor only after the host verifier matched."""

        schema = self._family_schema(str(family or ""))
        if schema is None:
            return
        schema.add_evidence(
            f"verified:{evidence_key}",
            level=int(level),
            exact_transition=True,
        )
        self.record_verifier_outcome(
            family=family,
            level=level,
            evidence_key=evidence_key,
            accepted=True,
            diagnostic=diagnostic,
        )
        if inherited:
            self.record_inheritance_outcome(
                family=family,
                level=level,
                evidence_key=evidence_key,
                accepted=True,
                reason=str((diagnostic or {}).get("classification", "exact_match")),
            )

    def record_inheritance_offer(
        self, *, family: str, level: int, evidence_key: str
    ) -> None:
        schema = self._family_schema(str(family or ""))
        if schema is None:
            return
        self._increment_counter_once(
            schema,
            marker=f"inheritance_offer:{evidence_key}",
            counter="inherited_schema_offered_count",
        )
        schema.evidence["last_inherited_level"] = int(level)

    def record_inheritance_outcome(
        self,
        *,
        family: str,
        level: int,
        evidence_key: str,
        accepted: bool,
        reason: str,
    ) -> None:
        schema = self._family_schema(str(family or ""))
        if schema is None:
            return
        counter = (
            "inherited_schema_used_count"
            if accepted
            else "inherited_schema_refuted_count"
        )
        if not self._increment_counter_once(
            schema,
            marker=f"inheritance_outcome:{evidence_key}",
            counter=counter,
        ):
            return
        outcomes = list(schema.evidence.get("inheritance_outcomes", []))
        outcomes.append(
            {
                "level": int(level),
                "accepted": bool(accepted),
                "reason": str(reason),
            }
        )
        schema.evidence["inheritance_outcomes"] = outcomes[-16:]

    def record_verifier_outcome(
        self,
        *,
        family: str,
        level: int,
        evidence_key: str,
        accepted: bool,
        diagnostic: Mapping[str, Any] | None,
    ) -> None:
        schema = self._family_schema(str(family or ""))
        if schema is None:
            return
        detail = dict(diagnostic or {})
        classification = str(
            detail.get("classification")
            or ("exact_match" if accepted else "true_reject")
        )
        if accepted:
            counter = "verifier_match_count"
        elif classification == "world_model_residual":
            counter = "verifier_model_residual_count"
        else:
            counter = "verifier_true_reject_count"
        if not self._increment_counter_once(
            schema,
            marker=f"verifier_outcome:{evidence_key}",
            counter=counter,
        ):
            return
        evidence = detail.get("evidence")
        repaired = bool(
            classification == "model_repaired_match"
            or (
                isinstance(evidence, Mapping)
                and evidence.get("schema_repair_applied")
            )
        )
        if repaired:
            schema.evidence["schema_repair_count"] = int(
                schema.evidence.get("schema_repair_count", 0) or 0
            ) + 1
        outcomes = list(schema.evidence.get("verifier_outcomes", []))
        outcomes.append(
            {
                "level": int(level),
                "accepted": bool(accepted),
                "classification": classification,
                "repaired": repaired,
                "rejected_reasons": [
                    str(value) for value in detail.get("rejected_reasons", [])
                ][:8],
            }
        )
        schema.evidence["verifier_outcomes"] = outcomes[-16:]

    @staticmethod
    def _increment_counter_once(
        schema: ExecutableMechanicSchema,
        *,
        marker: str,
        counter: str,
    ) -> bool:
        if marker in schema._evidence_keys:
            return False
        schema._evidence_keys.add(marker)
        schema.evidence[counter] = int(schema.evidence.get(counter, 0) or 0) + 1
        return True

    def recover_verified_residual_plans(
        self, result: Mapping[str, Any], *, frame: Any, level: int
    ) -> tuple[dict[str, Any], list[str]]:
        """Restore a frame-verified residual when fresh planning goes blind."""
        projected = copy.deepcopy(dict(result))
        plans = [item for item in projected.get("plans", []) if isinstance(item, dict)]
        certified_present = {
            str(item.get("family", ""))
            for item in plans
            if item.get("certified")
        }
        recovered: list[str] = []
        for family, remembered in sorted(self._residual_plans.items()):
            if family in certified_present or family != "mirror":
                continue
            schema = self._family_schema(family)
            if schema is None or schema.status != VERIFIED:
                continue
            if int(level) <= int(schema.learned_level):
                continue
            if int(remembered.get("_memory_level", -1)) != int(level):
                continue
            if int(self._verified_residual_levels.get(family, -1)) != int(level):
                continue
            actions = [
                copy.deepcopy(item)
                for item in remembered.get("actions", [])
                if isinstance(item, Mapping)
            ]
            if not actions:
                continue
            plans = [
                item for item in plans if str(item.get("family", "")) != family
            ]
            plan = copy.deepcopy(remembered)
            plan.pop("_memory_level", None)
            plan["actions"] = actions
            plan["action_count"] = len(actions)
            plan["certified"] = False
            plan["continuity_recovered"] = True
            plan["certificate_reason"] = (
                "verified cross-level residual: current static and dynamic "
                "signatures match the prior exact transition"
            )
            plan["inherited_schema"] = {
                "schema_id": schema.schema_id,
                "learned_level": int(schema.learned_level),
                "bound_level": int(level),
                "confidence": round(float(schema.confidence), 4),
                "authorization": "exact residual transition verified on current frame",
            }
            plans.append(plan)
            recovered.append(family)

            observations = projected.get("observations")
            if isinstance(observations, dict):
                for item in observations.get("families", []):
                    if isinstance(item, dict) and item.get("family") == family:
                        item["status"] = "continuity_recovered"
                        item["reason"] = plan["certificate_reason"]
                        break
        projected["plans"] = plans
        projected["recovered_residual_families"] = sorted(set(recovered))
        return projected, sorted(set(recovered))

    def annotate_inherited_plans(
        self, result: Mapping[str, Any], *, level: int
    ) -> tuple[dict[str, Any], list[str]]:
        projected = copy.deepcopy(dict(result))
        inherited: list[str] = []
        for plan in projected.get("plans", []):
            if not isinstance(plan, dict) or not plan.get("certified"):
                continue
            family = str(plan.get("family", "") or "")
            schema = self._family_schema(family)
            if schema is None or schema.status != VERIFIED:
                continue
            if int(level) <= int(schema.learned_level):
                continue
            grounding = plan.get("grounded_observation") or {}
            expected = grounding.get("expected_change") or {}
            family_exact = expected.get(f"{family}_exact")
            runtime_exact = expected.get("runtime_mechanic_exact")
            has_runtime_exact = bool(
                isinstance(runtime_exact, Mapping)
                and str(runtime_exact.get("toolbox_family", "")) == family
            )
            if not isinstance(family_exact, Mapping) and not has_runtime_exact:
                continue
            plan["inherited_schema"] = {
                "schema_id": schema.schema_id,
                "learned_level": int(schema.learned_level),
                "bound_level": int(level),
                "confidence": round(float(schema.confidence), 4),
                "authorization": "current-level certificate plus exact verifier",
            }
            inherited.append(family)
        projected["inherited_families"] = sorted(set(inherited))
        return projected, sorted(set(inherited))

    def advisory_packet(
        self,
        *,
        frame: Any,
        level: int,
        current_result: Mapping[str, Any] | None = None,
        authorized_families: Iterable[str] | None = None,
        require_verified_binding: bool = False,
    ) -> dict[str, Any]:
        allowed = (
            {str(family) for family in authorized_families}
            if authorized_families is not None
            else None
        )
        current_plans = {
            str(plan.get("family", "")): plan
            for plan in (current_result or {}).get("plans", [])
            if isinstance(plan, Mapping) and plan.get("certified")
        }
        rows: list[dict[str, Any]] = []
        for schema in self.schemas:
            if allowed is not None and schema.source_family not in allowed:
                continue
            if schema.status not in {SUPPORTED, VERIFIED}:
                continue
            binding = self._binding_for_schema(
                schema, frame=frame, level=level, current_plan=current_plans.get(schema.source_family)
            )
            if require_verified_binding and (
                schema.status != VERIFIED
                or str(binding.get("status", "") or "") != "bound_verified"
            ):
                continue
            rows.append(
                {
                    "schema_id": schema.schema_id,
                    "mechanic_id": schema.mechanic_id,
                    "source_family": schema.source_family,
                    "status": schema.status,
                    "confidence": round(float(schema.confidence), 4),
                    "learned_level": int(schema.learned_level),
                    "roles": list(schema.roles),
                    "operator": copy.deepcopy(schema.operator),
                    "invariants": list(schema.invariants),
                    "goal_relation": schema.goal_relation,
                    "required_bindings": list(schema.required_bindings),
                    "current_binding": binding,
                    "falsifiers": list(schema.falsifiers),
                }
            )
            if len(rows) >= self.max_advisory_schemas:
                break
        return {
            "schema_version": ADVISORY_VERSION,
            "level": int(level),
            "schemas": rows,
            "schema_count": len(rows),
            "authority": "advisory_only; host lease still requires a current exact verifier",
        }

    def runtime_context(self, *, level: int) -> dict[str, Any]:
        """Return coordinate-free episode knowledge for the next mechanic pass.

        Current bindings and residual action lists deliberately stay private to
        the current level. Runtime modules receive only schemas and evidence;
        they must bind every role again from the supplied current frame.
        """

        return {
            "schema_version": MEMORY_VERSION,
            "level": int(level),
            "level_transition_count": int(self._level_transition_count),
            "schemas": [schema.to_dict() for schema in self.schemas],
            "authority": "schema_evidence_only; current roles require rebinding",
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": MEMORY_VERSION,
            "game_key": self.game_key,
            "last_level": int(self._last_level),
            "transition_count": int(self._transition_count),
            "level_transition_count": int(self._level_transition_count),
            "residual_plans": {
                family: {
                    "level": int(plan.get("_memory_level", -1)),
                    "action_count": len(plan.get("actions", [])),
                    "frame_verified": int(
                        self._verified_residual_levels.get(family, -1)
                    )
                    == int(plan.get("_memory_level", -2)),
                }
                for family, plan in sorted(self._residual_plans.items())
            },
            "runtime_binding_families": sorted(self._runtime_bindings),
            "schemas": [schema.to_dict() for schema in self.schemas],
        }

    def _schema_from_runtime_mechanic(
        self, row: Mapping[str, Any], *, level: int
    ) -> ExecutableMechanicSchema | None:
        family = str(row.get("toolbox_family", "") or "")
        raw = row.get("schema")
        defaults = _family_defaults(family)
        if defaults is None or not isinstance(raw, Mapping):
            return None
        fallback_operator, fallback_goal = defaults
        schema_id = f"family:{family}"
        existing = self._schemas.get(schema_id)
        if existing is not None:
            operator = raw.get("operator")
            if isinstance(operator, Mapping):
                existing.operator = copy.deepcopy(dict(operator))
            return existing
        roles = [str(value) for value in raw.get("roles", []) if str(value)]
        required = [
            str(value) for value in raw.get("required_bindings", []) if str(value)
        ]
        operator = raw.get("operator")
        if not isinstance(operator, Mapping):
            operator = fallback_operator
        schema = ExecutableMechanicSchema(
            schema_id=schema_id,
            mechanic_id=str(
                raw.get("mechanic_id")
                or row.get("mechanic_family")
                or fallback_operator["kind"]
            ),
            source_family=family,
            status=OBSERVED,
            confidence=0.45,
            learned_level=int(level),
            roles=roles,
            preconditions=[
                str(value) for value in raw.get("preconditions", []) if str(value)
            ],
            operator=copy.deepcopy(dict(operator)),
            invariants=[
                str(value) for value in raw.get("invariants", []) if str(value)
            ],
            goal_relation=str(
                raw.get("goal_relation") or fallback_goal
            ),
            binding_recipe={
                **copy.deepcopy(dict(raw.get("binding_recipe") or {})),
                "requires_current_certificate": True,
                "uses_game_id": False,
                "uses_absolute_coordinates": False,
            },
            falsifiers=[
                str(value) for value in raw.get("falsifiers", []) if str(value)
            ],
            required_bindings=required
            or ["roles", "current_state", "goal", "transition_model"],
            evidence={
                "support_count": 0,
                "prediction_available_count": 0,
                "observed_transition_count": 0,
                "exact_transition_count": 0,
                "score_closure_count": 0,
                "verifier_match_count": 0,
                "verifier_true_reject_count": 0,
                "verifier_model_residual_count": 0,
                "schema_repair_count": 0,
                "inherited_schema_offered_count": 0,
                "inherited_schema_used_count": 0,
                "inherited_schema_refuted_count": 0,
                "levels": [],
            },
        )
        self._schemas[schema_id] = schema
        return schema

    def _advance_residual_plan(
        self,
        *,
        family: str,
        action_name: str,
        action_payload: Mapping[str, Any] | None,
        after_frame: np.ndarray,
        level: int,
        score_delta: int,
    ) -> None:
        remembered = self._residual_plans.get(family)
        if remembered is None:
            return
        actions = [item for item in remembered.get("actions", []) if isinstance(item, Mapping)]
        if not actions or int(remembered.get("_memory_level", -1)) != int(level):
            self._residual_plans.pop(family, None)
            self._verified_residual_levels.pop(family, None)
            return
        first = actions[0]
        expected_engine = {
            "UP": "ACTION1",
            "DOWN": "ACTION2",
            "LEFT": "ACTION3",
            "RIGHT": "ACTION4",
            "SPACE": "ACTION5",
            "MOUSE": "ACTION6",
        }.get(str(first.get("action", "")).upper(), "")
        matches = expected_engine == str(action_name).upper()
        if matches and expected_engine == "ACTION6":
            payload = dict(action_payload or {})
            matches = int(payload.get("x", -1)) == int(first.get("col", -2)) and int(
                payload.get("y", -1)
            ) == int(first.get("row", -2))
        if not matches:
            self._residual_plans.pop(family, None)
            self._verified_residual_levels.pop(family, None)
            return
        if int(score_delta) > 0:
            self._residual_plans.pop(family, None)
            self._verified_residual_levels.pop(family, None)
            return

        residual = [copy.deepcopy(dict(item)) for item in actions[1:]]
        grounding = remembered.get("grounded_observation") or {}
        expected = grounding.get("expected_change") or {}
        exact = expected.get(f"{family}_exact")
        if family != "mirror" or not residual or not isinstance(exact, Mapping):
            self._residual_plans.pop(family, None)
            self._verified_residual_levels.pop(family, None)
            return

        try:
            from arcgym.eca import primitive_synthesis as synthesis
            from arcgym.eca.tufa_solver_toolbox import (
                _family_verification_state,
                _grounded_observation,
            )

            state = synthesis._mirror_joint_cursor_state(after_frame)
            if state is not None:
                parsed_a = tuple(int(v) for v in state["cursor_a"])
                parsed_b = tuple(int(v) for v in state["cursor_b"])
                expected_change = grounding.get("expected_change") or {}
                expected_a = (
                    int(expected_change["cursor_a_after"]["col"]),
                    int(expected_change["cursor_a_after"]["row"]),
                )
                expected_b = (
                    int(expected_change["cursor_b_after"]["col"]),
                    int(expected_change["cursor_b_after"]["row"]),
                )
                if {parsed_a, parsed_b} == {expected_a, expected_b} and (
                    parsed_a != expected_a or parsed_b != expected_b
                ):
                    cursor_a, cursor_b = expected_a, expected_b
                    corrected_state = dict(state)
                    corrected_state["cursor_a"] = cursor_a
                    corrected_state["cursor_b"] = cursor_b
                    verification = _family_verification_state(
                        family, corrected_state, residual
                    )
                    next_grounding = _grounded_observation(
                        family, corrected_state, residual
                    )
                else:
                    verification = _family_verification_state(
                        family, state, residual
                    )
                    cursor_a, cursor_b = parsed_a, parsed_b
                    next_grounding = _grounded_observation(family, state, residual)
            else:
                expected_change = grounding.get("expected_change") or {}
                cursor_a = (
                    int(expected_change["cursor_a_after"]["col"]),
                    int(expected_change["cursor_a_after"]["row"]),
                )
                cursor_b = (
                    int(expected_change["cursor_b_after"]["col"]),
                    int(expected_change["cursor_b_after"]["row"]),
                )
                detail = remembered.get("detail") or {}
                binding = detail.get("verification_binding") or {}
                origin_values = list(binding.get("origin") or ())
                cell = int(binding.get("cell", 0) or 0)
                cursor_value = int(binding.get("cursor_value", -1))
                walkable_value = int(binding.get("walkable_value", -1))
                old_a = tuple(int(v) for v in detail.get("cursor_a", ()))
                old_b = tuple(int(v) for v in detail.get("cursor_b", ()))
                if (
                    len(origin_values) != 2
                    or cell <= 0
                    or len(old_a) != 2
                    or len(old_b) != 2
                ):
                    raise ValueError("mirror footprint binding unavailable")
                origin = (int(origin_values[0]), int(origin_values[1]))
                expected_positions = {cursor_a, cursor_b}
                footprint_valid = all(
                    _logical_cell_is_value(
                        after_frame,
                        origin=origin,
                        cell=cell,
                        position=position,
                        value=cursor_value,
                    )
                    for position in expected_positions
                ) and all(
                    _logical_cell_is_value(
                        after_frame,
                        origin=origin,
                        cell=cell,
                        position=position,
                        value=walkable_value,
                    )
                    for position in {old_a, old_b} - expected_positions
                )
                if not footprint_valid:
                    raise ValueError("mirror merged footprint mismatch")
                verification = {
                    "family": family,
                    "static_signature": str(exact.get("static_signature", "")),
                    "dynamic_signature": _stable_digest(
                        {"cursor_a": cursor_a, "cursor_b": cursor_b}
                    ),
                    "plan_hash": _stable_digest(residual),
                    "plan_len": len(residual),
                }
                next_grounding = self._mirror_residual_grounding(
                    remembered,
                    residual=residual,
                    cursor_a=cursor_a,
                    cursor_b=cursor_b,
                    static_signature=str(exact.get("static_signature", "")),
                )
            valid = (
                str(verification.get("static_signature", ""))
                == str(exact.get("static_signature", ""))
                and str(verification.get("dynamic_signature", ""))
                == str(exact.get("expected_dynamic_signature", ""))
                and str(verification.get("dynamic_signature", ""))
                != str(exact.get("dynamic_signature_before", ""))
                and int(verification.get("plan_len", -1))
                == int(exact.get("expected_remaining_plan_len", -2))
                and str(verification.get("plan_hash", ""))
                == str(exact.get("expected_remaining_plan_hash", ""))
            )
            if not valid:
                raise ValueError("mirror residual verification failed")
            remembered["actions"] = residual
            remembered["action_count"] = len(residual)
            remembered["certified"] = False
            remembered["grounded_observation"] = next_grounding
            detail = remembered.setdefault("detail", {})
            detail["cursor_a"] = [int(v) for v in cursor_a]
            detail["cursor_b"] = [int(v) for v in cursor_b]
            self._verified_residual_levels[family] = int(level)
        except Exception:
            self._residual_plans.pop(family, None)
            self._verified_residual_levels.pop(family, None)

    def _mirror_residual_grounding(
        self,
        remembered: Mapping[str, Any],
        *,
        residual: list[dict[str, Any]],
        cursor_a: tuple[int, int],
        cursor_b: tuple[int, int],
        static_signature: str,
    ) -> dict[str, Any]:
        first = residual[0]
        after = _mirror_positions_after(cursor_a, cursor_b, str(first.get("action", "")))
        if after is None:
            raise ValueError("residual mirror action is not directional")
        next_a, next_b = after
        grounding = copy.deepcopy(remembered.get("grounded_observation") or {})
        grounding["entities"] = [
            {
                "role": "cursor_a",
                "space": "logical_grid",
                "row": int(cursor_a[1]),
                "col": int(cursor_a[0]),
            },
            {
                "role": "cursor_b",
                "space": "logical_grid",
                "row": int(cursor_b[1]),
                "col": int(cursor_b[0]),
            },
        ]
        expected_change = {
            "cursor_a_after": {
                "space": "logical_grid",
                "row": int(next_a[1]),
                "col": int(next_a[0]),
            },
            "cursor_b_after": {
                "space": "logical_grid",
                "row": int(next_b[1]),
                "col": int(next_b[0]),
            },
            "rule": (
                "horizontal inputs move the two cursors oppositely; "
                "vertical inputs move both together"
            ),
        }
        verification = {
            "family": "mirror",
            "static_signature": static_signature,
            "dynamic_signature": _stable_digest(
                {"cursor_a": cursor_a, "cursor_b": cursor_b}
            ),
            "plan_hash": _stable_digest(residual),
            "plan_len": len(residual),
        }
        expected_change["mirror_exact"] = {
            "verifier": "mirror_state_transition_v1",
            "family": "mirror",
            "static_signature": static_signature,
            "dynamic_signature_before": verification["dynamic_signature"],
            "expected_remaining_plan_hash": _stable_digest(residual[1:]),
            "expected_remaining_plan_len": max(0, len(residual) - 1),
            "expected_dynamic_signature": _stable_digest(
                {"cursor_a": next_a, "cursor_b": next_b}
            ),
        }
        grounding["proposed_first_action"] = copy.deepcopy(first)
        grounding["expected_change"] = expected_change
        grounding["verification_state"] = verification
        return grounding

    def _schema_from_toolbox_plan(
        self, plan: Mapping[str, Any], *, level: int
    ) -> ExecutableMechanicSchema:
        family = str(plan.get("family", ""))
        schema_id = f"family:{family}"
        existing = self._schemas.get(schema_id)
        if existing is not None:
            return existing
        defaults = _family_defaults(family)
        if defaults is None:
            raise ValueError(f"unregistered toolbox family: {family}")
        fallback_operator, fallback_goal = defaults
        grounding = plan.get("grounded_observation") or {}
        roles = sorted(
            {
                str(item.get("role", ""))
                for item in grounding.get("entities", [])
                if isinstance(item, Mapping) and item.get("role")
            }
        )
        action_names = sorted(
            {
                str(item.get("action", ""))
                for item in plan.get("actions", [])
                if isinstance(item, Mapping) and item.get("action")
            }
        )
        operator = copy.deepcopy(fallback_operator)
        operator["observed_action_vocabulary"] = action_names
        schema = ExecutableMechanicSchema(
            schema_id=schema_id,
            mechanic_id=operator["kind"],
            source_family=family,
            status=OBSERVED,
            confidence=0.45,
            learned_level=int(level),
            roles=roles,
            preconditions=[
                f"current frame is certified by the {family} structural reader",
                "all action parameters are rebound from the current frame",
            ],
            operator=operator,
            invariants=[str(value) for value in grounding.get("invariants", [])],
            goal_relation=fallback_goal,
            binding_recipe={
                "reader": family,
                "requires_current_certificate": True,
                "requires_exact_transition": True,
                "uses_game_id": False,
                "uses_absolute_coordinates": False,
            },
            falsifiers=[str(value) for value in grounding.get("falsifiers", [])],
            required_bindings=["roles", "current_state", "goal", "transition_model"],
            evidence={
                "support_count": 0,
                "prediction_available_count": 0,
                "observed_transition_count": 0,
                "exact_transition_count": 0,
                "score_closure_count": 0,
                "verifier_match_count": 0,
                "verifier_true_reject_count": 0,
                "verifier_model_residual_count": 0,
                "schema_repair_count": 0,
                "inherited_schema_offered_count": 0,
                "inherited_schema_used_count": 0,
                "inherited_schema_refuted_count": 0,
                "levels": [],
            },
        )
        self._schemas[schema_id] = schema
        return schema

    def _compile_click_operators(self, *, level: int) -> None:
        for operator in self._action_effect.admitted_operators():
            signature = operator.signature
            schema_id = (
                f"click:{operator.effect_class}:"
                f"{signature.bbox_bucket[0]}x{signature.bbox_bucket[1]}:"
                f"unique={int(signature.unique_on_frame)}"
            )
            schema = self._schemas.get(schema_id)
            if schema is None:
                schema = ExecutableMechanicSchema(
                    schema_id=schema_id,
                    mechanic_id="click_effect",
                    source_family="action_effect",
                    status=OBSERVED,
                    confidence=0.45,
                    learned_level=int(level),
                    roles=["clickable_component", "changed_region"],
                    preconditions=["a structural component matches the learned shape bucket"],
                    operator={
                        "kind": "click_effect",
                        "action_family": "click",
                        "action": "MOUSE",
                        "effect": operator.effect_class,
                    },
                    invariants=["unrelated stable components should remain unchanged"],
                    goal_relation="use the observed click effect as a composable state-changing operator",
                    binding_recipe={
                        "bbox_bucket": list(signature.bbox_bucket),
                        "unique_on_frame": bool(signature.unique_on_frame),
                        "uses_color_identity": False,
                        "uses_absolute_coordinates": False,
                    },
                    falsifiers=["a rebound click produces no matching effect"],
                    required_bindings=["clickable_component", "current_site", "desired_effect"],
                    evidence={
                        "support_count": 0,
                        "exact_transition_count": 0,
                        "score_closure_count": 0,
                        "verifier_match_count": 0,
                        "verifier_true_reject_count": 0,
                        "verifier_model_residual_count": 0,
                        "schema_repair_count": 0,
                        "inherited_schema_offered_count": 0,
                        "inherited_schema_used_count": 0,
                        "inherited_schema_refuted_count": 0,
                        "levels": [],
                    },
                )
                self._schemas[schema_id] = schema
            for index in range(int(operator.reproductions)):
                schema.add_evidence(
                    f"action-effect:{schema_id}:{index}", level=int(level), exact_transition=True
                )

    def _family_schema(self, family: str) -> ExecutableMechanicSchema | None:
        schema_id = self._family_schema_ids.get(str(family))
        return self._schemas.get(schema_id) if schema_id else None

    def _matching_click_schemas(
        self, frame: np.ndarray, payload: Mapping[str, Any] | None
    ) -> list[ExecutableMechanicSchema]:
        try:
            x = int((payload or {}).get("x"))
            y = int((payload or {}).get("y"))
        except (TypeError, ValueError):
            return []
        component = _component_at(frame, x, y)
        if component is None:
            return []
        value, cells = component
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        bucket = (_bbox_bucket(max(xs) - min(xs) + 1), _bbox_bucket(max(ys) - min(ys) + 1))
        matching: list[ExecutableMechanicSchema] = []
        for operator in self._action_effect.admitted_operators():
            if int(operator.signature.class_value) != int(value):
                continue
            if tuple(operator.signature.bbox_bucket) != bucket:
                continue
            schema_id = (
                f"click:{operator.effect_class}:{bucket[0]}x{bucket[1]}:"
                f"unique={int(operator.signature.unique_on_frame)}"
            )
            schema = self._schemas.get(schema_id)
            if schema is not None:
                matching.append(schema)
        return matching

    def _binding_for_schema(
        self,
        schema: ExecutableMechanicSchema,
        *,
        frame: Any,
        level: int,
        current_plan: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if current_plan is not None:
            grounding = current_plan.get("grounded_observation") or {}
            expected = grounding.get("expected_change") or {}
            exact = isinstance(
                expected.get(f"{schema.source_family}_exact"), Mapping
            ) or isinstance(expected.get("runtime_mechanic_exact"), Mapping)
            role_bindings = []
            for item in grounding.get("entities", []):
                if not isinstance(item, Mapping) or not item.get("role"):
                    continue
                binding = {
                    "role": str(item.get("role")),
                    "space": str(item.get("space", "frame")),
                }
                if "row" in item and "col" in item:
                    binding["row"] = int(item["row"])
                    binding["col"] = int(item["col"])
                    try:
                        visible_value = int(
                            np.asarray(frame)[binding["row"], binding["col"]]
                        )
                    except (IndexError, TypeError, ValueError):
                        visible_value = -1
                    if 0 <= visible_value < len(_ASCII_COLOR_CHARS):
                        binding["visible_symbol"] = _ASCII_COLOR_CHARS[visible_value]
                        binding["visual_validation"] = "ascii_cell_exact"
                    if binding["role"] == "click_site":
                        binding["interaction"] = "MOUSE"
                        binding["coordinate_semantics"] = "interaction_point"
                role_bindings.append(binding)
                if len(role_bindings) >= 8:
                    break
            return {
                "status": "bound_verified" if exact else "bound_supported",
                "bound_level": int(level),
                "current_plan_actions": len(current_plan.get("actions", [])),
                "exact_transition_available": exact,
                "role_bindings": role_bindings,
                "requires_probe": not exact,
            }
        runtime_binding = self._runtime_bindings.get(schema.source_family)
        if isinstance(runtime_binding, Mapping):
            binding = copy.deepcopy(dict(runtime_binding))
            binding.setdefault("status", "bound_supported")
            binding.setdefault("bound_level", int(level))
            binding.setdefault("requires_probe", True)
            return binding
        if schema.source_family != "action_effect":
            return {
                "status": "unbound",
                "bound_level": int(level),
                "requires_probe": True,
            }
        recipe = schema.binding_recipe
        bucket = tuple(int(value) for value in recipe.get("bbox_bucket", []))
        candidates: list[dict[str, int]] = []
        try:
            components = extract_structural_components(np.asarray(frame), cell=1)
        except Exception:
            components = []
        for component in components:
            if (_bbox_bucket(component.width), _bbox_bucket(component.height)) != bucket:
                continue
            candidates.append(
                {
                    "row": int(round(component.centroid[1])),
                    "col": int(round(component.centroid[0])),
                }
            )
            if len(candidates) >= 4:
                break
        return {
            "status": "candidate_sites" if candidates else "unbound",
            "bound_level": int(level),
            "candidate_sites": candidates,
            "requires_probe": True,
        }


__all__ = [
    "ADVISORY_VERSION",
    "EpisodeMechanicMemory",
    "ExecutableMechanicSchema",
    "MEMORY_VERSION",
    "OBSERVED",
    "SCHEMA_VERSION",
    "SUPPORTED",
    "VERIFIED",
]
