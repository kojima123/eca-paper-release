"""Goal-Conditioned Structural Time Model (v0.2) for ARC-AGI-3.

This module implements the core data structures and algorithms described in the
"Goal-Conditioned Structural Time Model" specification (v0.1/v0.2).

Key concepts:
- z_t: goal-conditioned latent structural state (NOT a scalar score)
- m_t: current structural model (objects, relations, invariants, affordances)
- g_t: goal hypothesis (online-generated DSL, not a pre-enumerated list)
- p_t: goal progress estimate
- u_t: goal hypothesis uncertainty
- d_m: structural difference (composite of D_O, D_R, D_I, D_A)
- tau: internal time (cumulative meaningful structural transitions)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Object-level features
# ---------------------------------------------------------------------------

@dataclass
class ObjectFeature:
    """Per-object feature set for structural model m_t.

    Attributes:
        obj_id: unique identifier within a frame
        frame_id: frame index (for multi-frame support)
        mask: 2-D binary mask as a flat list of (row, col) tuples
        bbox: (row_min, col_min, row_max, col_max)
        centroid: (row, col) float centroid
        area: number of cells
        color_hist: dict mapping color int -> cell count
        euler: topological Euler number (holes indicator)
        sym: symmetry bitmask (bit0=LR, bit1=UD, bit2=ROT90, bit3=ROT180)
    """
    obj_id: str
    frame_id: int = 0
    mask: list[tuple[int, int]] = field(default_factory=list)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    centroid: tuple[float, float] = (0.0, 0.0)
    area: int = 0
    color_hist: dict[int, int] = field(default_factory=dict)
    euler: int = 0
    sym: int = 0  # bitmask

    def to_dict(self) -> dict[str, Any]:
        return {
            "obj_id": self.obj_id,
            "frame_id": self.frame_id,
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "area": self.area,
            "color_hist": self.color_hist,
            "euler": self.euler,
            "sym": self.sym,
        }


@dataclass
class RelationEdge:
    """Directed relation edge between two objects.

    Relation types: adjacent, overlap, contains, direction, distance_bucket,
    isomorphic, symmetric_pair.
    """
    src_id: str
    dst_id: str
    relation: str  # e.g. "adjacent", "contains", "overlap"
    attrs: dict[str, Any] = field(default_factory=dict)

    def edge_key(self) -> str:
        return f"{self.src_id}|{self.relation}|{self.dst_id}"


@dataclass
class InvariantFeature:
    """Global invariant features of the current grid state.

    These are computed over the whole grid rather than individual objects,
    providing a robust fallback when object matching is unstable.
    """
    n_objects: int = 0
    n_connected_components: int = 0
    color_mass: dict[int, int] = field(default_factory=dict)  # color -> total cells
    symmetry_signature: int = 0  # bitmask of global symmetries
    border_touch_count: int = 0  # objects touching the border
    occupancy_ratio: float = 0.0  # filled cells / total cells
    frame_stability: float = 1.0  # similarity to previous frame [0,1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_objects": self.n_objects,
            "n_connected_components": self.n_connected_components,
            "color_mass": self.color_mass,
            "symmetry_signature": self.symmetry_signature,
            "border_touch_count": self.border_touch_count,
            "occupancy_ratio": round(self.occupancy_ratio, 6),
            "frame_stability": round(self.frame_stability, 6),
        }


@dataclass
class AffordanceHeatmap:
    """Action affordance heatmap for hidden-coordinate actions (e.g. ACTION6).

    Stores a 2-D probability map over grid positions indicating which
    coordinates are likely to be effective for the current action type.
    """
    rows: int = 0
    cols: int = 0
    data: list[list[float]] = field(default_factory=list)  # rows x cols

    def get(self, r: int, c: int) -> float:
        if not self.data or r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return 0.0
        return float(self.data[r][c])

    def flat(self) -> list[float]:
        if not self.data:
            return []
        return [v for row in self.data for v in row]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "data": self.data,
        }


# ---------------------------------------------------------------------------
# Structural model z_t
# ---------------------------------------------------------------------------

@dataclass
class StructuralModel:
    """Current structural model m_t = (O_t, R_t, I_t, A_t).

    This is the core internal representation of the grid state, expressed
    as objects, relations, global invariants, and action affordances.
    """
    objects: list[ObjectFeature] = field(default_factory=list)
    relations: list[RelationEdge] = field(default_factory=list)
    invariants: InvariantFeature = field(default_factory=InvariantFeature)
    affordance: AffordanceHeatmap = field(default_factory=AffordanceHeatmap)
    step_index: int = 0

    def object_by_id(self, obj_id: str) -> ObjectFeature | None:
        for o in self.objects:
            if o.obj_id == obj_id:
                return o
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "n_objects": len(self.objects),
            "invariants": self.invariants.to_dict(),
            "affordance_shape": [self.affordance.rows, self.affordance.cols],
        }


@dataclass
class GoalHypothesis:
    """Online-generated goal hypothesis g_t using a small DSL.

    Rather than pre-enumerating game-specific goal labels, this hypothesis
    is generated online from observed object/relation vocabulary.

    Fields:
        termination_predicate: description of the win/completion condition
        target_substructure: description of the target structural pattern
        invariant_constraints: list of invariant conditions that must hold
        policy_sketch: high-level action strategy
        confidence: hypothesis confidence score in [0, 1]
        hypothesis_id: unique identifier for tracking
    """
    termination_predicate: str = ""
    target_substructure: str = ""
    invariant_constraints: list[str] = field(default_factory=list)
    policy_sketch: str = ""
    confidence: float = 0.5
    hypothesis_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "termination_predicate": self.termination_predicate,
            "target_substructure": self.target_substructure,
            "invariant_constraints": self.invariant_constraints,
            "policy_sketch": self.policy_sketch,
            "confidence": round(self.confidence, 6),
        }


@dataclass
class GoalConditionedState:
    """Full latent structural state z_t = (m_t, g_t, p_t, u_t).

    This replaces the naive z_t = score formulation.
    The external score is kept separately as an auxiliary signal.
    """
    structural_model: StructuralModel = field(default_factory=StructuralModel)
    goal_hypothesis: GoalHypothesis = field(default_factory=GoalHypothesis)
    goal_progress: float = 0.0   # p_t: estimated completion probability [0,1]
    uncertainty: float = 1.0     # u_t: hypothesis entropy / undifferentiation [0,1]
    # External score kept separate — NOT part of z_t
    external_score: float = 0.0
    internal_time: float = 0.0   # tau: cumulative meaningful structural transitions

    def to_dict(self) -> dict[str, Any]:
        return {
            "structural_model": self.structural_model.to_dict(),
            "goal_hypothesis": self.goal_hypothesis.to_dict(),
            "goal_progress": round(self.goal_progress, 6),
            "uncertainty": round(self.uncertainty, 6),
            "external_score": round(self.external_score, 6),
            "internal_time": round(self.internal_time, 6),
        }


# ---------------------------------------------------------------------------
# Structural difference d_m
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def _object_match_cost(a: ObjectFeature, b: ObjectFeature) -> float:
    """Compute matching cost between two objects for Hungarian assignment.

    Lower cost = better match. Uses centroid distance + area ratio + color overlap.
    """
    # Centroid distance (normalized by grid diagonal estimate)
    dr = a.centroid[0] - b.centroid[0]
    dc = a.centroid[1] - b.centroid[1]
    centroid_dist = math.sqrt(dr * dr + dc * dc) / 30.0  # normalize by ~30 cells

    # Area ratio mismatch
    max_area = max(a.area, b.area, 1)
    area_diff = abs(a.area - b.area) / max_area

    # Color histogram overlap (1 - overlap = cost)
    all_colors = set(a.color_hist) | set(b.color_hist)
    if all_colors:
        overlap = sum(
            min(a.color_hist.get(c, 0), b.color_hist.get(c, 0))
            for c in all_colors
        )
        total = max(sum(a.color_hist.values()), sum(b.color_hist.values()), 1)
        color_cost = 1.0 - _safe_div(overlap, total)
    else:
        color_cost = 0.0

    return _clamp01(0.5 * centroid_dist + 0.3 * area_diff + 0.2 * color_cost)


def _hungarian_match(
    objs_a: list[ObjectFeature],
    objs_b: list[ObjectFeature],
    cost_threshold: float = 0.7,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Simple greedy Hungarian-style matching (O(n^2), sufficient for small n).

    Returns:
        matched: list of (idx_a, idx_b) pairs
        unmatched_a: indices in objs_a with no match
        unmatched_b: indices in objs_b with no match
    """
    if not objs_a or not objs_b:
        return [], list(range(len(objs_a))), list(range(len(objs_b)))

    n, m = len(objs_a), len(objs_b)
    cost_matrix = [
        [_object_match_cost(objs_a[i], objs_b[j]) for j in range(m)]
        for i in range(n)
    ]

    matched: list[tuple[int, int]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()

    # Greedy: pick minimum cost pairs iteratively
    pairs = sorted(
        [(cost_matrix[i][j], i, j) for i in range(n) for j in range(m)],
        key=lambda x: x[0],
    )
    for cost, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        if cost > cost_threshold:
            break
        matched.append((i, j))
        used_a.add(i)
        used_b.add(j)

    unmatched_a = [i for i in range(n) if i not in used_a]
    unmatched_b = [j for j in range(m) if j not in used_b]
    return matched, unmatched_a, unmatched_b


def compute_D_O(
    objs_a: list[ObjectFeature],
    objs_b: list[ObjectFeature],
) -> float:
    """Compute object-level structural difference D_O in [0, 1].

    Uses Hungarian matching to pair objects, then measures:
    - Deformation cost of matched pairs
    - Penalty for unmatched (added/removed) objects
    """
    matched, unmatched_a, unmatched_b = _hungarian_match(objs_a, objs_b)

    if not objs_a and not objs_b:
        return 0.0

    total_objects = max(len(objs_a), len(objs_b), 1)

    # Deformation cost for matched pairs
    deform_cost = 0.0
    for i, j in matched:
        deform_cost += _object_match_cost(objs_a[i], objs_b[j])

    # Penalty for unmatched objects (normalized)
    unmatched_penalty = (len(unmatched_a) + len(unmatched_b)) * 0.5

    raw = _safe_div(deform_cost + unmatched_penalty, total_objects)
    return _clamp01(raw)


def compute_D_R(
    rels_a: list[RelationEdge],
    rels_b: list[RelationEdge],
) -> float:
    """Compute relation-graph structural difference D_R in [0, 1].

    Measures the symmetric difference of edge keys between two relation sets.
    """
    keys_a = {e.edge_key() for e in rels_a}
    keys_b = {e.edge_key() for e in rels_b}
    union = keys_a | keys_b
    if not union:
        return 0.0
    symmetric_diff = keys_a.symmetric_difference(keys_b)
    return _clamp01(_safe_div(len(symmetric_diff), len(union)))


def compute_D_I(
    inv_a: InvariantFeature,
    inv_b: InvariantFeature,
) -> float:
    """Compute invariant-level structural difference D_I in [0, 1].

    Uses four invariants: connected component count, color mass vector,
    symmetry signature, and border-touch count.
    """
    # Connected component count
    max_cc = max(inv_a.n_connected_components, inv_b.n_connected_components, 1)
    d_cc = _safe_div(abs(inv_a.n_connected_components - inv_b.n_connected_components), max_cc)

    # Color mass vector (L1 distance normalized)
    all_colors = set(inv_a.color_mass) | set(inv_b.color_mass)
    if all_colors:
        total_cells = max(sum(inv_a.color_mass.values()), sum(inv_b.color_mass.values()), 1)
        d_color = sum(
            abs(inv_a.color_mass.get(c, 0) - inv_b.color_mass.get(c, 0))
            for c in all_colors
        )
        d_color = _safe_div(d_color, 2.0 * total_cells)
    else:
        d_color = 0.0

    # Symmetry signature (Hamming distance normalized to [0,1])
    sym_xor = inv_a.symmetry_signature ^ inv_b.symmetry_signature
    d_sym = bin(sym_xor).count("1") / 8.0  # 8-bit signature

    # Border-touch count
    max_bt = max(inv_a.border_touch_count, inv_b.border_touch_count, 1)
    d_bt = _safe_div(abs(inv_a.border_touch_count - inv_b.border_touch_count), max_bt)

    return _clamp01(0.30 * d_cc + 0.35 * d_color + 0.20 * d_sym + 0.15 * d_bt)


def _jsd(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon Divergence between two probability distributions."""
    if len(p) != len(q) or not p:
        return 0.0
    total_p = sum(p)
    total_q = sum(q)
    if total_p <= 0 or total_q <= 0:
        return 0.0
    p_norm = [x / total_p for x in p]
    q_norm = [x / total_q for x in q]
    m = [(a + b) / 2.0 for a, b in zip(p_norm, q_norm)]

    def _kl(a: list[float], b: list[float]) -> float:
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > 1e-12 and bi > 1e-12:
                s += ai * math.log(ai / bi)
        return s

    return _clamp01(0.5 * _kl(p_norm, m) + 0.5 * _kl(q_norm, m))


def compute_D_A(
    aff_a: AffordanceHeatmap,
    aff_b: AffordanceHeatmap,
) -> float:
    """Compute affordance-level structural difference D_A in [0, 1].

    Uses Jensen-Shannon Divergence between the two affordance heatmaps.
    A sharp change in the affordance landscape indicates meaningful structural
    progress (especially for ACTION6 with hidden coordinates).
    """
    flat_a = aff_a.flat()
    flat_b = aff_b.flat()
    if not flat_a or not flat_b or len(flat_a) != len(flat_b):
        return 0.0
    return _jsd(flat_a, flat_b)


def compute_structural_diff(
    model_a: StructuralModel,
    model_b: StructuralModel,
    *,
    w_o: float = 0.35,
    w_r: float = 0.25,
    w_i: float = 0.25,
    w_a: float = 0.15,
) -> dict[str, float]:
    """Compute the composite structural difference d_m between two states.

    d_m = w_O * D_O + w_R * D_R + w_I * D_I + w_A * D_A

    Returns a dict with individual components and the total d_m value.
    """
    d_o = compute_D_O(model_a.objects, model_b.objects)
    d_r = compute_D_R(model_a.relations, model_b.relations)
    d_i = compute_D_I(model_a.invariants, model_b.invariants)
    d_a = compute_D_A(model_a.affordance, model_b.affordance)

    # Normalize weights
    total_w = w_o + w_r + w_i + w_a
    if total_w <= 0:
        total_w = 1.0

    d_m = (w_o * d_o + w_r * d_r + w_i * d_i + w_a * d_a) / total_w

    return {
        "d_m": round(_clamp01(d_m), 6),
        "D_O": round(d_o, 6),
        "D_R": round(d_r, 6),
        "D_I": round(d_i, 6),
        "D_A": round(d_a, 6),
    }


# ---------------------------------------------------------------------------
# Adaptive threshold phi and internal time tau
# ---------------------------------------------------------------------------

def compute_adaptive_epsilon(
    uncertainty: float,
    noise_floor: float = 0.02,
    *,
    phase_scale: float = 0.10,
) -> float:
    """Compute adaptive threshold epsilon for phi.

    epsilon = epsilon_noise + epsilon_phase(u_t)

    - High uncertainty (early exploration): epsilon is high (filter small changes)
    - Low uncertainty (late convergence): epsilon is low (capture fine-grained changes)

    Args:
        uncertainty: current goal hypothesis uncertainty u_t in [0, 1]
        noise_floor: estimated noise floor from no-op / undo actions
        phase_scale: scaling factor for the phase-dependent term
    """
    epsilon_phase = phase_scale * float(uncertainty)
    return _clamp01(noise_floor + epsilon_phase)


def phi_meaningful(
    d_m: float,
    epsilon: float,
    *,
    soft: bool = True,
    hysteresis: float = 0.02,
) -> float:
    """Threshold function phi that filters out non-meaningful structural changes.

    Uses a soft sigmoid threshold with optional hysteresis for stability.

    Args:
        d_m: raw structural difference value in [0, 1]
        epsilon: adaptive threshold
        soft: if True, use soft sigmoid; if False, use hard step
        hysteresis: hysteresis band around epsilon

    Returns:
        Filtered structural difference in [0, 1]. Returns 0 for sub-threshold changes.
    """
    if d_m <= max(0.0, epsilon - hysteresis):
        return 0.0
    if not soft:
        return d_m if d_m > epsilon else 0.0
    # Soft sigmoid: smooth transition around epsilon
    k = 20.0  # steepness
    sigmoid = 1.0 / (1.0 + math.exp(-k * (d_m - epsilon)))
    return _clamp01(d_m * sigmoid)


def update_internal_time(
    tau: float,
    d_m: float,
    uncertainty: float,
    noise_floor: float = 0.02,
) -> tuple[float, float]:
    """Update internal time tau by adding the meaningful structural difference.

    tau += phi(d_m, epsilon(u_t))

    Returns:
        (new_tau, phi_value) tuple
    """
    epsilon = compute_adaptive_epsilon(uncertainty, noise_floor)
    phi_val = phi_meaningful(d_m, epsilon)
    new_tau = tau + phi_val
    return new_tau, phi_val


# ---------------------------------------------------------------------------
# ECA / PSA with structural model
# ---------------------------------------------------------------------------

def compute_structural_eca(
    state: GoalConditionedState,
    *,
    w_mismatch: float = 0.40,
    w_goal_uncertainty: float = 0.40,
    w_bottleneck: float = 0.20,
) -> dict[str, float]:
    """Compute current structural anxiety (ECA) from the goal-conditioned state.

    ECA = w_mismatch * C_obs + w_goal_uncertainty * u_t + w_bottleneck * B

    where:
        C_obs: observation-model mismatch (approximated from uncertainty)
        u_t: goal hypothesis uncertainty
        B: bottleneck intensity (low progress + high uncertainty)

    Returns a dict with the ECA score and its breakdown.
    """
    u_t = _clamp01(state.uncertainty)
    p_t = _clamp01(state.goal_progress)

    # Observation-model mismatch: proxy from uncertainty and low progress
    c_obs = _clamp01(u_t * (1.0 - p_t))

    # Bottleneck: high uncertainty AND low progress
    bottleneck = _clamp01(u_t * (1.0 - p_t) * 2.0)

    eca = (
        w_mismatch * c_obs
        + w_goal_uncertainty * u_t
        + w_bottleneck * bottleneck
    )

    return {
        "eca": round(_clamp01(eca), 6),
        "c_obs": round(c_obs, 6),
        "u_t": round(u_t, 6),
        "bottleneck": round(bottleneck, 6),
    }


def compute_structural_psa(
    state: GoalConditionedState,
    rollout_history: list[dict[str, float]],
    *,
    tau_budget: float = 2.0,
    stagnation_window: int = 3,
    stagnation_threshold: float = 0.02,
) -> dict[str, float]:
    """Compute future structural anxiety (PSA) using tau-budget rollout.

    Replaces the fixed H=3 horizon with a variable horizon determined by
    the tau budget and stagnation detection.

    Args:
        state: current goal-conditioned state
        rollout_history: list of dicts with keys 'd_m', 'phi_val', 'eca'
            from recent steps (used to estimate future trajectory)
        tau_budget: maximum internal time to roll out
        stagnation_window: number of consecutive sub-threshold steps for stagnation
        stagnation_threshold: phi threshold below which a step is "stagnant"

    Returns a dict with PSA score and diagnostics.
    """
    if not rollout_history:
        # No history: PSA defaults to current ECA
        eca_result = compute_structural_eca(state)
        return {
            "psa": eca_result["eca"],
            "horizon_used": 0,
            "tau_consumed": 0.0,
            "stagnation_detected": False,
        }

    # Simulate future trajectory from recent history
    tau_consumed = 0.0
    stagnation_streak = 0
    horizon_used = 0
    future_eca_sum = 0.0

    for step in rollout_history:
        phi_val = float(step.get("phi_val", 0.0))
        step_eca = float(step.get("eca", state.uncertainty))

        tau_consumed += phi_val
        horizon_used += 1
        future_eca_sum += step_eca

        # Check stagnation
        if phi_val < stagnation_threshold:
            stagnation_streak += 1
        else:
            stagnation_streak = 0

        # Stopping conditions
        if tau_consumed >= tau_budget:
            break
        if stagnation_streak >= stagnation_window:
            break

    stagnation_detected = stagnation_streak >= stagnation_window

    # PSA = mean future ECA + stagnation penalty
    mean_future_eca = _safe_div(future_eca_sum, horizon_used) if horizon_used > 0 else state.uncertainty
    stagnation_penalty = 0.3 if stagnation_detected else 0.0

    psa = _clamp01(mean_future_eca + stagnation_penalty)

    return {
        "psa": round(psa, 6),
        "horizon_used": horizon_used,
        "tau_consumed": round(tau_consumed, 6),
        "stagnation_detected": stagnation_detected,
    }


# ---------------------------------------------------------------------------
# Oscillation detection
# ---------------------------------------------------------------------------

class OscillationDetector:
    """Detect action oscillation patterns (e.g., UP/DOWN/UP/DOWN).

    Oscillation occurs when the agent alternates between two opposing actions
    without making meaningful progress. This is a critical failure mode in
    navigation-style games.

    The detector maintains a sliding window of recent actions and checks for
    alternating patterns. When oscillation is detected, phi should be
    suppressed to prevent tau accumulation from meaningless movement.
    """

    # Pairs of opposing actions that commonly oscillate
    OPPOSING_PAIRS: list[tuple[str, str]] = [
        ("ACTION1", "ACTION2"),  # UP / DOWN
        ("ACTION3", "ACTION4"),  # LEFT / RIGHT
    ]

    def __init__(self, window_size: int = 6, threshold: int = 3) -> None:
        """Initialize the oscillation detector.

        Args:
            window_size: number of recent actions to examine
            threshold: minimum number of alternations to declare oscillation
        """
        self.window_size = window_size
        self.threshold = threshold
        self._action_window: list[str] = []
        self._oscillation_count: int = 0
        self._total_oscillations: int = 0
        self._current_pair: tuple[str, str] | None = None

    def record(self, action: str) -> None:
        """Record a new action into the sliding window."""
        self._action_window.append(action)
        if len(self._action_window) > self.window_size:
            self._action_window = self._action_window[-self.window_size:]

    def is_oscillating(self) -> bool:
        """Check if the recent action window shows oscillation."""
        if len(self._action_window) < 4:
            return False

        for a, b in self.OPPOSING_PAIRS:
            alternations = 0
            for i in range(1, len(self._action_window)):
                prev = self._action_window[i - 1]
                curr = self._action_window[i]
                if (prev == a and curr == b) or (prev == b and curr == a):
                    alternations += 1
                else:
                    alternations = 0
            if alternations >= self.threshold:
                self._oscillation_count = alternations
                self._total_oscillations += 1
                self._current_pair = (a, b)
                return True

        self._oscillation_count = 0
        self._current_pair = None
        return False

    def get_escape_actions(self) -> list[str]:
        """Return actions orthogonal to the current oscillation pair.

        If oscillating between UP/DOWN, suggest LEFT/RIGHT, and vice versa.
        """
        if self._current_pair is None:
            return []
        a, b = self._current_pair
        all_nav = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
        return sorted(all_nav - {a, b})

    def get_oscillation_severity(self) -> float:
        """Return oscillation severity in [0, 1].

        0 = no oscillation, 1 = maximum oscillation.
        """
        if not self.is_oscillating():
            return 0.0
        return min(1.0, self._oscillation_count / max(1, self.window_size - 1))

    def reset(self) -> None:
        self._action_window = []
        self._oscillation_count = 0
        self._total_oscillations = 0
        self._current_pair = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_oscillating": self.is_oscillating(),
            "oscillation_count": self._oscillation_count,
            "total_oscillations": self._total_oscillations,
            "current_pair": list(self._current_pair) if self._current_pair else None,
            "severity": round(self.get_oscillation_severity(), 4),
            "window": list(self._action_window),
        }


# ---------------------------------------------------------------------------
# Goal distance tracking for progress-aware phi
# ---------------------------------------------------------------------------

class GoalDistanceTracker:
    """Track distance to goal over time to distinguish progress from oscillation.

    Instead of relying solely on d_m (structural difference) to determine
    whether a change is "meaningful", this tracker monitors whether the
    agent is actually getting closer to its goal.

    This solves the problem where oscillation (UP/DOWN) produces d_m > 0
    but makes no progress toward the goal.
    """

    def __init__(self, window_size: int = 5) -> None:
        self._distances: list[float] = []
        self._window_size = window_size

    def record_distance(self, distance: float) -> None:
        """Record the current distance to goal."""
        self._distances.append(distance)
        if len(self._distances) > self._window_size * 2:
            self._distances = self._distances[-self._window_size * 2:]

    def is_making_progress(self) -> bool:
        """Check if recent distances show a decreasing trend."""
        if len(self._distances) < 3:
            return True  # assume progress when insufficient data
        recent = self._distances[-self._window_size:]
        if len(recent) < 2:
            return True
        # Check if the latest distance is less than the average of the window
        avg = sum(recent[:-1]) / len(recent[:-1])
        return recent[-1] < avg - 0.5  # at least 0.5 closer

    def get_progress_ratio(self) -> float:
        """Return progress ratio in [-1, 1].

        Positive = getting closer, Negative = getting farther, 0 = no change.
        """
        if len(self._distances) < 2:
            return 0.0
        recent = self._distances[-self._window_size:]
        if len(recent) < 2:
            return 0.0
        delta = recent[0] - recent[-1]  # positive if distance decreased
        max_dist = max(abs(d) for d in recent) if recent else 1.0
        if max_dist == 0:
            return 0.0
        return max(-1.0, min(1.0, delta / max_dist))

    def phi_with_progress(self, d_m: float, epsilon: float, is_oscillating: bool) -> float:
        """Compute phi that accounts for goal progress and oscillation.

        If oscillating and not making progress, suppress phi to 0.
        If making progress, use standard phi.
        """
        if is_oscillating and not self.is_making_progress():
            return 0.0  # suppress: oscillation without progress
        # Standard phi
        return phi_meaningful(d_m, epsilon)

    def reset(self) -> None:
        self._distances = []


# ---------------------------------------------------------------------------
# Policy utility function
# ---------------------------------------------------------------------------

def compute_action_utility(
    goal_progress_delta: float,
    phi_val: float,
    psa: float,
    action_cost: float = 1.0,
    *,
    w1: float = 0.40,
    w2: float = 0.30,
    w3: float = 0.20,
    w4: float = 0.10,
) -> float:
    """Compute action utility for policy selection.

    utility = w1 * delta_p + w2 * phi(d_m) - w3 * PSA - w4 * action_cost

    This replaces naive score maximization with a structural-progress-aware
    utility that balances goal progress, meaningful structural change,
    future anxiety reduction, and action cost.

    Args:
        goal_progress_delta: estimated increase in goal progress p_t
        phi_val: meaningful structural difference phi(d_m) for this action
        psa: predicted future structural anxiety after this action
        action_cost: cost of taking this action (default 1.0 for standard actions)

    Returns:
        Utility score (higher is better). Can be negative.
    """
    utility = (
        w1 * _clamp01(goal_progress_delta)
        + w2 * _clamp01(phi_val)
        - w3 * _clamp01(psa)
        - w4 * _clamp01(action_cost)
    )
    return float(utility)


# ---------------------------------------------------------------------------
# Grid-based structural model builder
# ---------------------------------------------------------------------------

def build_structural_model_from_grid(
    grid: list[list[int]],
    step_index: int = 0,
    prev_model: StructuralModel | None = None,
) -> StructuralModel:
    """Build a StructuralModel from a 2-D integer grid.

    This is the primary entry point for converting raw ARC-AGI-3 grid
    observations into the internal structural representation.

    Args:
        grid: 2-D list of integers (color values)
        step_index: current step index
        prev_model: previous structural model (used for frame_stability)

    Returns:
        StructuralModel with objects, relations, invariants, and affordance.
    """
    if not grid or not grid[0]:
        return StructuralModel(step_index=step_index)

    rows = len(grid)
    cols = len(grid[0])

    # Extract objects via connected-component analysis
    objects = _extract_objects(grid, rows, cols)

    # Build relation graph
    relations = _build_relations(objects, rows, cols)

    # Compute global invariants
    invariants = _compute_invariants(grid, objects, rows, cols, prev_model)

    # Build affordance heatmap (object-boundary based heuristic)
    affordance = _build_affordance_heatmap(objects, rows, cols)

    return StructuralModel(
        objects=objects,
        relations=relations,
        invariants=invariants,
        affordance=affordance,
        step_index=step_index,
    )


def _extract_objects(
    grid: list[list[int]],
    rows: int,
    cols: int,
) -> list[ObjectFeature]:
    """Extract objects from grid using flood-fill connected components.

    Background (color 0) is treated as non-object. Each connected component
    of the same color forms one object.
    """
    visited = [[False] * cols for _ in range(rows)]
    objects: list[ObjectFeature] = []
    obj_counter = 0

    for start_r in range(rows):
        for start_c in range(cols):
            color = grid[start_r][start_c]
            if color == 0 or visited[start_r][start_c]:
                continue

            # BFS flood fill
            queue = [(start_r, start_c)]
            visited[start_r][start_c] = True
            cells: list[tuple[int, int]] = []

            while queue:
                r, c = queue.pop(0)
                cells.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))

            if not cells:
                continue

            # Compute features
            rs = [c[0] for c in cells]
            cs_list = [c[1] for c in cells]
            bbox = (min(rs), min(cs_list), max(rs), max(cs_list))
            centroid = (sum(rs) / len(rs), sum(cs_list) / len(cells))
            color_hist = {color: len(cells)}

            # Symmetry check (simple LR/UD for the bounding box)
            sym = _compute_symmetry(cells, bbox)

            obj = ObjectFeature(
                obj_id=f"obj_{obj_counter}",
                frame_id=0,
                mask=cells,
                bbox=bbox,
                centroid=centroid,
                area=len(cells),
                color_hist=color_hist,
                euler=0,  # simplified: no hole detection
                sym=sym,
            )
            objects.append(obj)
            obj_counter += 1

    return objects


def _compute_symmetry(
    cells: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> int:
    """Compute symmetry bitmask for an object.

    Bit 0: left-right symmetry
    Bit 1: up-down symmetry
    """
    cell_set = set(cells)
    r_min, c_min, r_max, c_max = bbox
    sym = 0

    # Left-right symmetry
    lr_sym = all(
        (r, c_min + c_max - c) in cell_set
        for r, c in cells
    )
    if lr_sym:
        sym |= 1

    # Up-down symmetry
    ud_sym = all(
        (r_min + r_max - r, c) in cell_set
        for r, c in cells
    )
    if ud_sym:
        sym |= 2

    return sym


def _build_relations(
    objects: list[ObjectFeature],
    rows: int,
    cols: int,
) -> list[RelationEdge]:
    """Build a relation graph between objects.

    Detects: adjacency, containment, relative direction.
    """
    relations: list[RelationEdge] = []
    obj_cells = {o.obj_id: set(o.mask) for o in objects}

    for i, obj_a in enumerate(objects):
        for j, obj_b in enumerate(objects):
            if i >= j:
                continue

            # Adjacency check (any cell of A is 4-adjacent to any cell of B)
            adjacent = False
            for r, c in obj_a.mask:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (r + dr, c + dc) in obj_cells[obj_b.obj_id]:
                        adjacent = True
                        break
                if adjacent:
                    break

            if adjacent:
                relations.append(RelationEdge(
                    src_id=obj_a.obj_id,
                    dst_id=obj_b.obj_id,
                    relation="adjacent",
                ))

            # Relative direction (centroid-based)
            dr = obj_b.centroid[0] - obj_a.centroid[0]
            dc = obj_b.centroid[1] - obj_a.centroid[1]
            if abs(dr) > abs(dc):
                direction = "below" if dr > 0 else "above"
            else:
                direction = "right_of" if dc > 0 else "left_of"

            relations.append(RelationEdge(
                src_id=obj_a.obj_id,
                dst_id=obj_b.obj_id,
                relation=f"direction_{direction}",
            ))

    return relations


def _compute_invariants(
    grid: list[list[int]],
    objects: list[ObjectFeature],
    rows: int,
    cols: int,
    prev_model: StructuralModel | None,
) -> InvariantFeature:
    """Compute global invariant features for the current grid."""
    # Color mass
    color_mass: dict[int, int] = {}
    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            if color != 0:
                color_mass[color] = color_mass.get(color, 0) + 1

    # Border-touch count
    border_cells = set()
    for r in range(rows):
        border_cells.add((r, 0))
        border_cells.add((r, cols - 1))
    for c in range(cols):
        border_cells.add((0, c))
        border_cells.add((rows - 1, c))

    border_touch = sum(
        1 for obj in objects
        if any(cell in border_cells for cell in obj.mask)
    )

    # Occupancy ratio
    total_cells = rows * cols
    filled_cells = sum(color_mass.values())
    occupancy = filled_cells / total_cells if total_cells > 0 else 0.0

    # Frame stability (compare with previous model's invariants)
    frame_stability = 1.0
    if prev_model is not None:
        prev_inv = prev_model.invariants
        prev_cm = prev_inv.color_mass
        all_colors = set(color_mass) | set(prev_cm)
        if all_colors:
            overlap = sum(min(color_mass.get(c, 0), prev_cm.get(c, 0)) for c in all_colors)
            total = max(sum(color_mass.values()), sum(prev_cm.values()), 1)
            frame_stability = overlap / total

    # Symmetry signature (global grid symmetry)
    sym_sig = _compute_grid_symmetry(grid, rows, cols)

    return InvariantFeature(
        n_objects=len(objects),
        n_connected_components=len(objects),
        color_mass=color_mass,
        symmetry_signature=sym_sig,
        border_touch_count=border_touch,
        occupancy_ratio=occupancy,
        frame_stability=frame_stability,
    )


def _compute_grid_symmetry(
    grid: list[list[int]],
    rows: int,
    cols: int,
) -> int:
    """Compute global grid symmetry bitmask.

    Bit 0: left-right, Bit 1: up-down, Bit 2: 180-degree rotation.
    """
    sym = 0

    # Left-right
    lr = all(grid[r][c] == grid[r][cols - 1 - c] for r in range(rows) for c in range(cols))
    if lr:
        sym |= 1

    # Up-down
    ud = all(grid[r][c] == grid[rows - 1 - r][c] for r in range(rows) for c in range(cols))
    if ud:
        sym |= 2

    # 180-degree rotation
    rot180 = all(
        grid[r][c] == grid[rows - 1 - r][cols - 1 - c]
        for r in range(rows) for c in range(cols)
    )
    if rot180:
        sym |= 4

    return sym


def _build_affordance_heatmap(
    objects: list[ObjectFeature],
    rows: int,
    cols: int,
) -> AffordanceHeatmap:
    """Build an affordance heatmap for hidden-coordinate actions.

    Assigns higher probability to:
    - Object boundary cells
    - Object centroids
    - Cells near symmetry axes

    This heatmap is especially useful for ACTION6 (hidden active region).
    """
    if rows == 0 or cols == 0:
        return AffordanceHeatmap(rows=rows, cols=cols, data=[])

    heatmap = [[0.0] * cols for _ in range(rows)]

    for obj in objects:
        obj_cells = set(obj.mask)
        # Boundary cells: cells adjacent to non-object cells
        for r, c in obj.mask:
            is_boundary = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or (nr, nc) not in obj_cells:
                    is_boundary = True
                    break
            if is_boundary:
                heatmap[r][c] += 1.0

        # Centroid cell
        cr, cc = int(round(obj.centroid[0])), int(round(obj.centroid[1]))
        if 0 <= cr < rows and 0 <= cc < cols:
            heatmap[cr][cc] += 2.0

    # Normalize to [0, 1]
    max_val = max(v for row in heatmap for v in row)
    if max_val > 0:
        heatmap = [[v / max_val for v in row] for row in heatmap]

    return AffordanceHeatmap(rows=rows, cols=cols, data=heatmap)


# ---------------------------------------------------------------------------
# Goal hypothesis generator (DSL-based, online)
# ---------------------------------------------------------------------------

def generate_goal_hypothesis(
    current_model: StructuralModel,
    action_history: list[str],
    prev_hypothesis: GoalHypothesis | None = None,
    *,
    hypothesis_id: str = "h0",
) -> GoalHypothesis:
    """Generate a goal hypothesis online from the current structural model.

    Rather than pre-enumerating game-specific goal labels, this function
    uses the observed object/relation vocabulary to construct a hypothesis.

    The hypothesis is expressed in a small DSL:
        T: termination predicate (what does "done" look like?)
        S: target substructure (what structure are we building toward?)
        I: invariant constraints (what must remain true?)
        P: policy sketch (what general strategy to follow?)
        c: confidence

    This is a heuristic first-version implementation. A full version would
    use MDL scoring and consistency checking.
    """
    inv = current_model.invariants
    n_obj = inv.n_objects
    n_cc = inv.n_connected_components

    # Termination predicate: guess based on object count and symmetry
    if inv.symmetry_signature > 0:
        term_pred = f"grid_symmetric(sig={inv.symmetry_signature})"
    elif n_obj <= 2:
        term_pred = "single_connected_component"
    else:
        term_pred = f"n_objects_reduced(target<{max(1, n_obj - 1)})"

    # Target substructure: describe current dominant structure
    dominant_color = max(inv.color_mass, key=inv.color_mass.get) if inv.color_mass else 0
    target_sub = f"color_{dominant_color}_connected"

    # Invariant constraints: what should stay stable
    constraints = []
    if inv.color_mass:
        constraints.append(f"color_mass_preserved(colors={list(inv.color_mass.keys())})")
    if inv.border_touch_count > 0:
        constraints.append("border_structure_maintained")

    # Policy sketch: based on object count
    if n_obj > 4:
        policy = "reduce_fragmentation"
    elif inv.symmetry_signature == 0:
        policy = "build_symmetry"
    else:
        policy = "refine_structure"

    # Confidence: higher if we have more history
    confidence = min(0.9, 0.3 + 0.1 * len(action_history))
    if prev_hypothesis is not None:
        # Carry over some confidence from previous hypothesis
        confidence = _clamp01(0.7 * confidence + 0.3 * prev_hypothesis.confidence)

    return GoalHypothesis(
        termination_predicate=term_pred,
        target_substructure=target_sub,
        invariant_constraints=constraints,
        policy_sketch=policy,
        confidence=confidence,
        hypothesis_id=hypothesis_id,
    )
