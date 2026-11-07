"""Lazy registry for independently implemented Tufa mechanic families."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

from .base import module_protocol_errors


@dataclass(frozen=True)
class RuntimeMechanicSpec:
    toolbox_family: str
    mechanic_family: str
    module_name: str
    rollout: str


_SPECS = (
    RuntimeMechanicSpec(
        toolbox_family="control_marker",
        mechanic_family="control_marker_pieces",
        module_name="arcgym.eca.tufa_mechanics.control_marker",
        rollout="repair",
    ),
    RuntimeMechanicSpec(
        toolbox_family="mirror",
        mechanic_family="coupled_mirror_navigation",
        module_name="arcgym.eca.tufa_mechanics.mirror_joint",
        rollout="repair",
    ),
    RuntimeMechanicSpec(
        toolbox_family="sand_filter",
        mechanic_family="filter_flow_routing",
        module_name="arcgym.eca.tufa_mechanics.filter_flow",
        rollout="repair",
    ),
    RuntimeMechanicSpec(
        toolbox_family="stride_maze",
        mechanic_family="stride_hazard_navigation",
        module_name="arcgym.eca.tufa_mechanics.stride_hazard",
        rollout="repair",
    ),
    RuntimeMechanicSpec(
        toolbox_family="program_execution",
        mechanic_family="bit_program_execution",
        module_name="arcgym.eca.tufa_mechanics.program_execution",
        rollout="repair",
    ),
    RuntimeMechanicSpec(
        toolbox_family="linked_extent",
        mechanic_family="linked_extent_growth",
        module_name="arcgym.eca.tufa_mechanics.linked_extent",
        rollout="repair",
    ),
    RuntimeMechanicSpec(
        toolbox_family="rule_rewrite",
        mechanic_family="rule_table_rewrite",
        module_name="arcgym.eca.tufa_mechanics.rule_rewrite",
        rollout="repair",
    ),
    RuntimeMechanicSpec(
        toolbox_family="reflection_orbit",
        mechanic_family="reflection_orbit_coverage",
        module_name="arcgym.eca.tufa_mechanics.reflection_orbit",
        rollout="repair",
    ),
    RuntimeMechanicSpec(
        toolbox_family="anchor_overlap",
        mechanic_family="anchor_overlap_assembly",
        module_name="arcgym.eca.tufa_mechanics.anchor_overlap",
        rollout="repair",
    ),
    RuntimeMechanicSpec(
        toolbox_family="region_paint",
        mechanic_family="region_paint_program",
        module_name="arcgym.eca.tufa_mechanics.region_paint",
        rollout="repair",
    ),
    RuntimeMechanicSpec(
        toolbox_family="shape_marker",
        mechanic_family="shape_marker_coverage",
        module_name="arcgym.eca.tufa_mechanics.shape_marker",
        rollout="repair",
    ),
    RuntimeMechanicSpec(
        toolbox_family="hierarchical_slots",
        mechanic_family="hierarchical_slot_sequence",
        module_name="arcgym.eca.tufa_mechanics.hierarchical_slots",
        rollout="repair",
    ),
    RuntimeMechanicSpec(
        toolbox_family="recorded_replay",
        mechanic_family="recorded_path_replay",
        module_name="arcgym.eca.tufa_mechanics.recorded_replay",
        rollout="repair",
    ),
    RuntimeMechanicSpec(
        toolbox_family="cycle_permutation",
        mechanic_family="cycle_permutation_alignment",
        module_name="arcgym.eca.tufa_mechanics.cycle_permutation",
        rollout="advisory",
    ),
)

_BY_TOOLBOX_FAMILY = {item.toolbox_family: item for item in _SPECS}


def runtime_mechanic_spec(toolbox_family: str) -> RuntimeMechanicSpec | None:
    return _BY_TOOLBOX_FAMILY.get(str(toolbox_family or "").strip())


def available_runtime_families() -> tuple[str, ...]:
    return tuple(item.toolbox_family for item in _SPECS)


def load_runtime_mechanic(toolbox_family: str) -> ModuleType | None:
    spec = runtime_mechanic_spec(toolbox_family)
    if spec is None:
        return None
    try:
        module = importlib.import_module(spec.module_name)
    except (ImportError, ModuleNotFoundError):
        return None
    if module_protocol_errors(module):
        return None
    if str(getattr(module, "FAMILY_ID", "")) != spec.mechanic_family:
        return None
    return module


__all__ = [
    "RuntimeMechanicSpec",
    "available_runtime_families",
    "load_runtime_mechanic",
    "runtime_mechanic_spec",
]
