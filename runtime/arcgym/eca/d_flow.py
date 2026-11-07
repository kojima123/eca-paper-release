"""D_flow: Event-driven Structural Flow Estimator (Tier 0 / 1 / 2).

Implements the three-tier D_flow design from the v0.2+ specification:

  Tier 0 (every turn): cheap residual summary — no OT solved.
      - rho_t: unmatched mass ratio
      - delta_n: object count change
      - delta_m: total mass change
      - delta_h: residual histogram JSD

  Tier 1 (conditional): sparse unbalanced Sinkhorn approximation.
      Triggered when residual mass is large, goal uncertainty is high,
      affordance landscape shifts, or object count changes.

  Tier 2 (decision boundary): local refinement.
      Triggered when top-1 vs top-2 action utility margin is small,
      or when a split/merge judgment would change the policy.

Key design principles (from spec):
  - D_flow is NOT a per-turn exact solver; it is an event-driven observer.
  - Accuracy criterion: does NOT destabilize action ranking, not numerical error.
  - q_flow confidence weight: D_O = alpha*D_core + (1-alpha)*q_flow*D_flow
  - D_I acts as a stabilizer when q_flow is low (fallback).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from arcgym.eca.structural_model import (
    AffordanceHeatmap,
    InvariantFeature,
    ObjectFeature,
    StructuralModel,
    _clamp01,
    _safe_div,
    _jsd,
)


# ---------------------------------------------------------------------------
# Tier 0: Residual summary (cheap, every turn)
# ---------------------------------------------------------------------------

@dataclass
class ResidualSummary:
    """Lightweight residual features computed every turn without OT.

    Fields:
        rho: unmatched mass ratio = 1 - matched_mass / total_mass
        delta_n: |O_{t+1}| - |O_t|  (object count change)
        delta_m: |sum(a_i) - sum(b_j)|  (total mass change)
        delta_h: JSD between residual mass histograms
        matched_mass: total mass of matched objects
        total_mass: total mass of all objects
    """
    rho: float = 0.0
    delta_n: int = 0
    delta_m: float = 0.0
    delta_h: float = 0.0
    matched_mass: float = 0.0
    total_mass: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rho": round(self.rho, 6),
            "delta_n": self.delta_n,
            "delta_m": round(self.delta_m, 6),
            "delta_h": round(self.delta_h, 6),
            "matched_mass": round(self.matched_mass, 6),
            "total_mass": round(self.total_mass, 6),
        }


def _coarse_cost(a: ObjectFeature, b: ObjectFeature) -> float:
    """Compute coarse matching cost using only cheap features.

    c_ij = beta1 * d_color + beta2 * d_area + beta3 * d_pos
    (No mask IoU or graph diff — those are expensive.)
    """
    beta1, beta2, beta3 = 0.40, 0.35, 0.25

    # Color distance: 1 - color overlap ratio
    all_colors = set(a.color_hist) | set(b.color_hist)
    if all_colors:
        overlap = sum(min(a.color_hist.get(c, 0), b.color_hist.get(c, 0)) for c in all_colors)
        total = max(sum(a.color_hist.values()), sum(b.color_hist.values()), 1)
        d_color = 1.0 - _safe_div(overlap, total)
    else:
        d_color = 0.0

    # Area distance
    max_area = max(a.area, b.area, 1)
    d_area = abs(a.area - b.area) / max_area

    # Position distance (centroid, normalized by ~30 cells)
    dr = a.centroid[0] - b.centroid[0]
    dc = a.centroid[1] - b.centroid[1]
    d_pos = math.sqrt(dr * dr + dc * dc) / 30.0

    return _clamp01(beta1 * d_color + beta2 * d_area + beta3 * d_pos)


def _greedy_match(
    objs_a: list[ObjectFeature],
    objs_b: list[ObjectFeature],
    cost_threshold: float = 0.65,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Fast greedy matching using coarse cost only (O(n^2), no OT)."""
    if not objs_a or not objs_b:
        return [], list(range(len(objs_a))), list(range(len(objs_b)))

    n, m = len(objs_a), len(objs_b)
    pairs = sorted(
        [(_coarse_cost(objs_a[i], objs_b[j]), i, j) for i in range(n) for j in range(m)],
        key=lambda x: x[0],
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int]] = []

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


def _color_mass_histogram(objs: list[ObjectFeature]) -> dict[int, float]:
    """Aggregate color mass histogram from a list of objects."""
    hist: dict[int, float] = {}
    for obj in objs:
        for color, count in obj.color_hist.items():
            hist[color] = hist.get(color, 0.0) + count
    return hist


def compute_residual_summary(
    model_a: StructuralModel,
    model_b: StructuralModel,
) -> ResidualSummary:
    """Compute Tier 0 residual summary (no OT, cheap, every turn).

    Uses greedy matching to identify matched/unmatched objects and
    computes lightweight residual statistics.
    """
    objs_a = model_a.objects
    objs_b = model_b.objects

    matched, unmatched_a, unmatched_b = _greedy_match(objs_a, objs_b)

    # Total mass = sum of all object areas
    mass_a = sum(o.area for o in objs_a)
    mass_b = sum(o.area for o in objs_b)
    total_mass = max(mass_a, mass_b, 1)

    # Matched mass = sum of areas of matched objects (min side)
    matched_mass = sum(
        min(objs_a[i].area, objs_b[j].area)
        for i, j in matched
    )

    # rho = 1 - matched_mass / total_mass
    rho = _clamp01(1.0 - _safe_div(matched_mass, total_mass))

    # delta_n = object count change
    delta_n = len(objs_b) - len(objs_a)

    # delta_m = |sum(a_i) - sum(b_j)|, normalized
    delta_m = _clamp01(abs(mass_a - mass_b) / total_mass)

    # delta_h = JSD between residual mass histograms
    # Residual = unmatched objects' color mass
    residual_hist_a = _color_mass_histogram([objs_a[i] for i in unmatched_a])
    residual_hist_b = _color_mass_histogram([objs_b[j] for j in unmatched_b])
    all_colors = sorted(set(residual_hist_a) | set(residual_hist_b))
    if all_colors:
        p = [float(residual_hist_a.get(c, 0)) for c in all_colors]
        q = [float(residual_hist_b.get(c, 0)) for c in all_colors]
        delta_h = _jsd(p, q)
    else:
        delta_h = 0.0

    return ResidualSummary(
        rho=rho,
        delta_n=delta_n,
        delta_m=delta_m,
        delta_h=delta_h,
        matched_mass=float(matched_mass),
        total_mass=float(total_mass),
    )


# ---------------------------------------------------------------------------
# Tier 1: Sparse unbalanced Sinkhorn approximation
# ---------------------------------------------------------------------------

@dataclass
class SinkhornResult:
    """Result of sparse unbalanced Sinkhorn computation.

    Fields:
        d_flow: estimated D_flow value in [0, 1]
        transport_cost: raw optimal transport cost
        del_mass: total deleted mass (source unmatched)
        ins_mass: total inserted mass (target unmatched)
        n_iterations: number of Sinkhorn iterations run
        converged: whether the algorithm converged
        split_detected: heuristic split event flag
        merge_detected: heuristic merge event flag
        birth_detected: new object appeared
        death_detected: object disappeared
    """
    d_flow: float = 0.0
    transport_cost: float = 0.0
    del_mass: float = 0.0
    ins_mass: float = 0.0
    n_iterations: int = 0
    converged: bool = False
    split_detected: bool = False
    merge_detected: bool = False
    birth_detected: bool = False
    death_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_flow": round(self.d_flow, 6),
            "transport_cost": round(self.transport_cost, 6),
            "del_mass": round(self.del_mass, 6),
            "ins_mass": round(self.ins_mass, 6),
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "split_detected": self.split_detected,
            "merge_detected": self.merge_detected,
            "birth_detected": self.birth_detected,
            "death_detected": self.death_detected,
        }


def _build_sparse_cost_matrix(
    objs_a: list[ObjectFeature],
    objs_b: list[ObjectFeature],
    top_k: int = 3,
) -> tuple[list[list[float]], list[list[int]]]:
    """Build sparse cost matrix: for each source, keep only top-k candidates.

    Returns:
        costs: n x top_k cost values (padded with 1.0 if fewer candidates)
        indices: n x top_k target indices
    """
    n = len(objs_a)
    m = len(objs_b)
    k = min(top_k, m)

    costs_out: list[list[float]] = []
    indices_out: list[list[int]] = []

    for i in range(n):
        row = sorted(
            [(_coarse_cost(objs_a[i], objs_b[j]), j) for j in range(m)],
            key=lambda x: x[0],
        )[:k]
        costs_out.append([c for c, _ in row])
        indices_out.append([j for _, j in row])

    return costs_out, indices_out


def _sinkhorn_unbalanced(
    a: list[float],
    b: list[float],
    cost_matrix: list[list[float]],
    epsilon: float = 0.1,
    lambda_del: float = 1.0,
    lambda_ins: float = 1.0,
    max_iter: int = 20,
    tol: float = 1e-3,
) -> tuple[list[list[float]], int, bool]:
    """Sparse unbalanced Sinkhorn algorithm.

    Solves the entropic-regularized unbalanced OT problem:
      min_{T>=0} sum_ij T_ij c_ij
        + lambda_del * sum_i (a_i - sum_j T_ij)_+
        + lambda_ins * sum_j (b_j - sum_i T_ij)_+
        + epsilon * sum_ij T_ij log T_ij

    Uses the log-domain Sinkhorn scaling for numerical stability.

    Args:
        a: source marginals (object areas, normalized)
        b: target marginals (object areas, normalized)
        cost_matrix: n x m cost matrix
        epsilon: entropic regularization
        lambda_del: deletion penalty
        lambda_ins: insertion penalty
        max_iter: maximum iterations
        tol: convergence tolerance (dual gap proxy)

    Returns:
        (T, n_iter, converged)
    """
    n = len(a)
    m = len(b)
    if n == 0 or m == 0:
        return [[]], 0, True

    # Initialize log-domain scaling vectors
    log_a = [math.log(max(ai, 1e-12)) for ai in a]
    log_b = [math.log(max(bj, 1e-12)) for bj in b]

    # K_ij = exp(-c_ij / epsilon)
    K = [[math.exp(-cost_matrix[i][j] / max(epsilon, 1e-9)) for j in range(m)] for i in range(n)]

    # Sinkhorn scaling: u, v
    u = [1.0] * n
    v = [1.0] * m

    converged = False
    n_iter = 0

    for it in range(max_iter):
        n_iter = it + 1

        # Update u: proximal step for source marginal
        u_new = []
        for i in range(n):
            kv = sum(K[i][j] * v[j] for j in range(m))
            # Unbalanced: soft marginal constraint via KL proximal
            # u_i = (a_i / (K v)_i)^{lambda_del / (lambda_del + epsilon)}
            ratio = _safe_div(a[i], max(kv, 1e-12))
            exp_pow = lambda_del / max(lambda_del + epsilon, 1e-9)
            u_new.append(max(ratio ** exp_pow, 1e-12))
        u = u_new

        # Update v: proximal step for target marginal
        v_new = []
        for j in range(m):
            ku = sum(K[i][j] * u[i] for i in range(n))
            ratio = _safe_div(b[j], max(ku, 1e-12))
            exp_pow = lambda_ins / max(lambda_ins + epsilon, 1e-9)
            v_new.append(max(ratio ** exp_pow, 1e-12))
        v = v_new

        # Convergence check: marginal violation
        if it > 0 and it % 5 == 0:
            err = 0.0
            for i in range(n):
                row_sum = sum(K[i][j] * u[i] * v[j] for j in range(m))
                err += abs(row_sum - a[i])
            if err < tol * n:
                converged = True
                break

    # Build transport plan T
    T = [[K[i][j] * u[i] * v[j] for j in range(m)] for i in range(n)]
    return T, n_iter, converged


def compute_d_flow_tier1(
    model_a: StructuralModel,
    model_b: StructuralModel,
    residual: ResidualSummary,
    *,
    top_k: int = 3,
    epsilon: float = 0.10,
    lambda_del: float = 1.0,
    lambda_ins: float = 1.0,
    max_iter: int = 20,
) -> SinkhornResult:
    """Compute Tier 1 approximate D_flow using sparse unbalanced Sinkhorn.

    This is only called when Tier 0 triggers indicate significant residual
    activity (see `should_run_tier1()`).

    Args:
        model_a: previous structural model
        model_b: current structural model
        residual: Tier 0 residual summary (used for event detection)
        top_k: number of candidate targets per source object
        epsilon: Sinkhorn entropic regularization
        lambda_del: deletion penalty
        lambda_ins: insertion penalty
        max_iter: maximum Sinkhorn iterations

    Returns:
        SinkhornResult with D_flow estimate and event flags.
    """
    objs_a = model_a.objects
    objs_b = model_b.objects

    if not objs_a and not objs_b:
        return SinkhornResult(d_flow=0.0, converged=True)

    if not objs_a:
        # All objects are new: pure birth
        return SinkhornResult(
            d_flow=1.0,
            ins_mass=float(sum(o.area for o in objs_b)),
            birth_detected=True,
            converged=True,
        )

    if not objs_b:
        # All objects disappeared: pure death
        return SinkhornResult(
            d_flow=1.0,
            del_mass=float(sum(o.area for o in objs_a)),
            death_detected=True,
            converged=True,
        )

    # Normalize marginals
    total_a = max(sum(o.area for o in objs_a), 1)
    total_b = max(sum(o.area for o in objs_b), 1)
    a_marg = [o.area / total_a for o in objs_a]
    b_marg = [o.area / total_b for o in objs_b]

    # Build full cost matrix (n x m)
    n, m = len(objs_a), len(objs_b)
    cost_matrix = [
        [_coarse_cost(objs_a[i], objs_b[j]) for j in range(m)]
        for i in range(n)
    ]

    # Run Sinkhorn
    T, n_iter, converged = _sinkhorn_unbalanced(
        a_marg, b_marg, cost_matrix,
        epsilon=epsilon,
        lambda_del=lambda_del,
        lambda_ins=lambda_ins,
        max_iter=max_iter,
    )

    # Compute transport cost
    transport_cost = 0.0
    for i in range(n):
        for j in range(m):
            transport_cost += T[i][j] * cost_matrix[i][j]

    # Compute deleted/inserted mass
    del_mass = 0.0
    for i in range(n):
        row_sum = sum(T[i][j] for j in range(m))
        del_mass += max(0.0, a_marg[i] - row_sum)

    ins_mass = 0.0
    for j in range(m):
        col_sum = sum(T[i][j] for i in range(n))
        ins_mass += max(0.0, b_marg[j] - col_sum)

    # Normalize D_flow to [0, 1]
    d_flow = _clamp01(transport_cost + 0.5 * del_mass + 0.5 * ins_mass)

    # Event detection heuristics
    birth_detected = residual.delta_n > 0 and ins_mass > 0.15
    death_detected = residual.delta_n < 0 and del_mass > 0.15
    # Split: one source maps to multiple targets with similar cost
    split_detected = False
    merge_detected = False
    if n < m:
        # More targets than sources: possible split
        for i in range(n):
            significant = sum(1 for j in range(m) if T[i][j] > 0.05)
            if significant >= 2:
                split_detected = True
                break
    elif n > m:
        # More sources than targets: possible merge
        for j in range(m):
            significant = sum(1 for i in range(n) if T[i][j] > 0.05)
            if significant >= 2:
                merge_detected = True
                break

    return SinkhornResult(
        d_flow=d_flow,
        transport_cost=round(transport_cost, 6),
        del_mass=round(del_mass, 6),
        ins_mass=round(ins_mass, 6),
        n_iterations=n_iter,
        converged=converged,
        split_detected=split_detected,
        merge_detected=merge_detected,
        birth_detected=birth_detected,
        death_detected=death_detected,
    )


# ---------------------------------------------------------------------------
# Tier 2: Local refinement (decision boundary only)
# ---------------------------------------------------------------------------

def compute_d_flow_tier2(
    model_a: StructuralModel,
    model_b: StructuralModel,
    tier1_result: SinkhornResult,
    *,
    extra_iter: int = 30,
    epsilon: float = 0.05,
) -> SinkhornResult:
    """Tier 2: Local refinement with more Sinkhorn iterations and tighter epsilon.

    Called only at decision boundaries (small utility margin, split/merge judgment).
    Runs more iterations and uses smaller epsilon for higher precision.

    Args:
        model_a: previous structural model
        model_b: current structural model
        tier1_result: Tier 1 result to refine
        extra_iter: additional iterations beyond Tier 1
        epsilon: tighter regularization for refinement

    Returns:
        Refined SinkhornResult.
    """
    # Re-run with tighter parameters
    refined = compute_d_flow_tier1(
        model_a, model_b,
        residual=ResidualSummary(),  # residual already computed
        epsilon=epsilon,
        max_iter=extra_iter,
    )
    # Preserve event flags from Tier 1 (they are more stable)
    refined.split_detected = refined.split_detected or tier1_result.split_detected
    refined.merge_detected = refined.merge_detected or tier1_result.merge_detected
    refined.birth_detected = refined.birth_detected or tier1_result.birth_detected
    refined.death_detected = refined.death_detected or tier1_result.death_detected
    return refined


# ---------------------------------------------------------------------------
# q_flow: D_flow confidence estimator
# ---------------------------------------------------------------------------

def compute_q_flow(
    residual: ResidualSummary,
    sinkhorn: SinkhornResult | None,
    *,
    w_residual: float = 0.30,
    w_sparsity: float = 0.25,
    w_stability: float = 0.25,
    w_mass: float = 0.20,
) -> float:
    """Compute D_flow confidence q_flow in [0, 1].

    q_flow = f(residual_size, sparsity, iteration_stability, mass_conservation)

    Higher q_flow = more reliable D_flow estimate.
    When q_flow is low, D_O falls back toward D_core.

    Args:
        residual: Tier 0 residual summary
        sinkhorn: Tier 1/2 Sinkhorn result (None if only Tier 0 was run)
        w_*: weights for each confidence component

    Returns:
        q_flow in [0, 1]
    """
    # Residual size confidence: lower rho = better matching = higher confidence
    conf_residual = _clamp01(1.0 - residual.rho)

    # Sparsity confidence: fewer unmatched objects = higher confidence
    # Approximated from delta_n (object count change)
    conf_sparsity = _clamp01(1.0 - min(abs(residual.delta_n) / 5.0, 1.0))

    # Iteration stability: did Sinkhorn converge?
    if sinkhorn is not None:
        conf_stability = 1.0 if sinkhorn.converged else 0.5
    else:
        # No Sinkhorn run: moderate confidence from residual alone
        conf_stability = 0.6

    # Mass conservation: low delta_m = better mass conservation = higher confidence
    conf_mass = _clamp01(1.0 - residual.delta_m)

    q_flow = (
        w_residual * conf_residual
        + w_sparsity * conf_sparsity
        + w_stability * conf_stability
        + w_mass * conf_mass
    )
    return _clamp01(q_flow)


# ---------------------------------------------------------------------------
# Tier transition logic
# ---------------------------------------------------------------------------

@dataclass
class TierDecision:
    """Result of the tier selection logic.

    Fields:
        tier: selected tier (0, 1, or 2)
        reasons: list of reasons for the decision
        rho_triggered: Tier 1 triggered by high residual mass
        uncertainty_triggered: Tier 1 triggered by high goal uncertainty
        affordance_triggered: Tier 1 triggered by affordance shift
        count_triggered: Tier 1 triggered by object count change
        margin_triggered: Tier 2 triggered by small utility margin
        event_triggered: Tier 2 triggered by split/merge judgment
    """
    tier: int = 0
    reasons: list[str] = field(default_factory=list)
    rho_triggered: bool = False
    uncertainty_triggered: bool = False
    affordance_triggered: bool = False
    count_triggered: bool = False
    margin_triggered: bool = False
    event_triggered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "reasons": self.reasons,
            "rho_triggered": self.rho_triggered,
            "uncertainty_triggered": self.uncertainty_triggered,
            "affordance_triggered": self.affordance_triggered,
            "count_triggered": self.count_triggered,
            "margin_triggered": self.margin_triggered,
            "event_triggered": self.event_triggered,
        }


def decide_tier(
    residual: ResidualSummary,
    uncertainty: float,
    affordance_entropy_delta: float = 0.0,
    utility_margin: float = float("inf"),
    tier1_result: SinkhornResult | None = None,
    *,
    rho_threshold: float = 0.25,
    uncertainty_threshold: float = 0.60,
    affordance_threshold: float = 0.15,
    margin_threshold: float = 0.05,
) -> TierDecision:
    """Decide which tier to run based on Tier 0 residual and context.

    Tier 1 triggers (any of):
      - rho_t > rho_threshold (high unmatched mass)
      - u_t > uncertainty_threshold (goal hypothesis still uncertain)
      - |H(A_t) - H(A_{t+1})| > affordance_threshold (affordance shift)
      - delta_n != 0 (object count changed)

    Tier 2 triggers (any of, after Tier 1):
      - utility_margin < margin_threshold (top-1 vs top-2 close)
      - split/merge detected in Tier 1 and would change policy

    Args:
        residual: Tier 0 residual summary
        uncertainty: current goal hypothesis uncertainty u_t
        affordance_entropy_delta: |H(A_t) - H(A_{t+1})|
        utility_margin: U(a_1) - U(a_2), the top action utility gap
        tier1_result: Tier 1 result (for Tier 2 decision)
        *_threshold: configurable thresholds

    Returns:
        TierDecision with selected tier and reasons.
    """
    decision = TierDecision(tier=0)

    # --- Tier 1 triggers ---
    if residual.rho > rho_threshold:
        decision.rho_triggered = True
        decision.reasons.append(f"rho={residual.rho:.3f}>{rho_threshold}")

    if uncertainty > uncertainty_threshold:
        decision.uncertainty_triggered = True
        decision.reasons.append(f"u_t={uncertainty:.3f}>{uncertainty_threshold}")

    if affordance_entropy_delta > affordance_threshold:
        decision.affordance_triggered = True
        decision.reasons.append(f"aff_delta={affordance_entropy_delta:.3f}>{affordance_threshold}")

    if residual.delta_n != 0:
        decision.count_triggered = True
        decision.reasons.append(f"delta_n={residual.delta_n}")

    tier1_needed = (
        decision.rho_triggered
        or decision.uncertainty_triggered
        or decision.affordance_triggered
        or decision.count_triggered
    )

    if not tier1_needed:
        return decision  # Tier 0 only

    decision.tier = 1

    # --- Tier 2 triggers (only if Tier 1 was run) ---
    if tier1_result is not None:
        if utility_margin < margin_threshold:
            decision.margin_triggered = True
            decision.reasons.append(f"margin={utility_margin:.4f}<{margin_threshold}")

        if tier1_result.split_detected or tier1_result.merge_detected:
            decision.event_triggered = True
            event = "split" if tier1_result.split_detected else "merge"
            decision.reasons.append(f"{event}_detected")

        if decision.margin_triggered or decision.event_triggered:
            decision.tier = 2

    return decision


# ---------------------------------------------------------------------------
# D_O with q_flow blending
# ---------------------------------------------------------------------------

def compute_D_O_with_flow(
    d_core: float,
    d_flow: float,
    q_flow: float,
    *,
    alpha: float = 0.70,
) -> float:
    """Blend D_core and D_flow using q_flow confidence.

    D_O = alpha * D_core + (1 - alpha) * q_flow * D_flow

    When q_flow is low (unreliable D_flow), D_O falls back toward D_core.
    When q_flow is high (reliable D_flow), D_flow contributes more.

    Args:
        d_core: D_O from the greedy Hungarian matching (structural_model.py)
        d_flow: D_flow from Sinkhorn (Tier 1 or 2)
        q_flow: D_flow confidence in [0, 1]
        alpha: base weight for D_core (default 0.70)

    Returns:
        Blended D_O in [0, 1]
    """
    return _clamp01(alpha * d_core + (1.0 - alpha) * q_flow * d_flow)


# ---------------------------------------------------------------------------
# D_I as stabilizer for low q_flow
# ---------------------------------------------------------------------------

def compute_D_I_stabilized(
    d_i: float,
    d_flow: float,
    q_flow: float,
    *,
    stabilizer_weight: float = 0.30,
) -> float:
    """Apply D_I as a stabilizer when q_flow is low.

    When q_flow is low (D_flow is unreliable), D_I acts as a fallback:
    if D_I is small, the overall structural change is likely small too,
    regardless of what D_flow says.

    D_I_stabilized = D_I + (1 - q_flow) * stabilizer_weight * (D_I - D_flow)

    In practice: if D_I is low and q_flow is low, pull D_flow toward D_I.

    Args:
        d_i: global invariant difference
        d_flow: D_flow estimate
        q_flow: D_flow confidence
        stabilizer_weight: how strongly D_I pulls D_flow when q_flow is low

    Returns:
        Stabilized D_flow estimate in [0, 1]
    """
    correction = (1.0 - q_flow) * stabilizer_weight * (d_i - d_flow)
    return _clamp01(d_flow + correction)


# ---------------------------------------------------------------------------
# Affordance entropy helper
# ---------------------------------------------------------------------------

def _heatmap_entropy(aff: AffordanceHeatmap) -> float:
    """Compute entropy of affordance heatmap distribution."""
    flat = aff.flat()
    if not flat:
        return 0.0
    total = sum(flat)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for v in flat:
        p = v / total
        if p > 1e-12:
            entropy -= p * math.log(p)
    # Normalize by log(n) to get [0, 1]
    n = len(flat)
    max_entropy = math.log(n) if n > 1 else 1.0
    return _clamp01(entropy / max_entropy)


def compute_affordance_entropy_delta(
    aff_a: AffordanceHeatmap,
    aff_b: AffordanceHeatmap,
) -> float:
    """Compute |H(A_t) - H(A_{t+1})| for Tier 1 trigger."""
    h_a = _heatmap_entropy(aff_a)
    h_b = _heatmap_entropy(aff_b)
    return abs(h_a - h_b)


# ---------------------------------------------------------------------------
# Full D_flow pipeline
# ---------------------------------------------------------------------------

@dataclass
class DFlowResult:
    """Full D_flow pipeline result.

    Contains all tier outputs, q_flow, blended D_O, and diagnostics.
    """
    tier_used: int = 0
    tier_decision: TierDecision = field(default_factory=TierDecision)
    residual: ResidualSummary = field(default_factory=ResidualSummary)
    sinkhorn: SinkhornResult | None = None
    q_flow: float = 0.0
    d_flow: float = 0.0
    d_o_blended: float = 0.0
    d_flow_stabilized: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_used": self.tier_used,
            "tier_decision": self.tier_decision.to_dict(),
            "residual": self.residual.to_dict(),
            "sinkhorn": self.sinkhorn.to_dict() if self.sinkhorn else None,
            "q_flow": round(self.q_flow, 6),
            "d_flow": round(self.d_flow, 6),
            "d_o_blended": round(self.d_o_blended, 6),
            "d_flow_stabilized": round(self.d_flow_stabilized, 6),
        }


def run_d_flow_pipeline(
    model_a: StructuralModel,
    model_b: StructuralModel,
    d_core: float,
    d_i: float,
    uncertainty: float,
    utility_margin: float = float("inf"),
    *,
    alpha: float = 0.70,
    rho_threshold: float = 0.25,
    uncertainty_threshold: float = 0.60,
    affordance_threshold: float = 0.15,
    margin_threshold: float = 0.05,
) -> DFlowResult:
    """Run the full D_flow Tier 0/1/2 pipeline.

    This is the main entry point for D_flow computation. It:
    1. Always computes Tier 0 residual summary
    2. Conditionally runs Tier 1 sparse Sinkhorn
    3. Conditionally runs Tier 2 refinement
    4. Computes q_flow confidence
    5. Blends D_core and D_flow into D_O
    6. Applies D_I stabilization when q_flow is low

    Args:
        model_a: previous structural model
        model_b: current structural model
        d_core: D_O from structural_model.compute_D_O() (greedy Hungarian)
        d_i: D_I from structural_model.compute_D_I()
        uncertainty: current goal hypothesis uncertainty u_t
        utility_margin: top-1 vs top-2 action utility gap (for Tier 2)
        alpha: D_core weight in blending
        *_threshold: tier trigger thresholds

    Returns:
        DFlowResult with all outputs.
    """
    # --- Tier 0: always ---
    residual = compute_residual_summary(model_a, model_b)

    # Affordance entropy delta
    aff_delta = compute_affordance_entropy_delta(
        model_a.affordance, model_b.affordance
    )

    # Initial tier decision (without Tier 1 result)
    initial_decision = decide_tier(
        residual, uncertainty, aff_delta, utility_margin,
        tier1_result=None,
        rho_threshold=rho_threshold,
        uncertainty_threshold=uncertainty_threshold,
        affordance_threshold=affordance_threshold,
        margin_threshold=margin_threshold,
    )

    sinkhorn_result: SinkhornResult | None = None
    tier_used = 0

    # --- Tier 1: conditional ---
    if initial_decision.tier >= 1:
        sinkhorn_result = compute_d_flow_tier1(model_a, model_b, residual)
        tier_used = 1

        # Re-evaluate for Tier 2 with Tier 1 result
        final_decision = decide_tier(
            residual, uncertainty, aff_delta, utility_margin,
            tier1_result=sinkhorn_result,
            rho_threshold=rho_threshold,
            uncertainty_threshold=uncertainty_threshold,
            affordance_threshold=affordance_threshold,
            margin_threshold=margin_threshold,
        )

        # --- Tier 2: decision boundary only ---
        if final_decision.tier >= 2:
            sinkhorn_result = compute_d_flow_tier2(model_a, model_b, sinkhorn_result)
            tier_used = 2
    else:
        final_decision = initial_decision

    # --- q_flow confidence ---
    q_flow = compute_q_flow(residual, sinkhorn_result)

    # --- D_flow value ---
    if sinkhorn_result is not None:
        d_flow_val = sinkhorn_result.d_flow
    else:
        # Tier 0 only: use residual as proxy for D_flow
        d_flow_val = _clamp01(0.5 * residual.rho + 0.3 * residual.delta_h + 0.2 * residual.delta_m)

    # --- Blend D_core and D_flow ---
    d_o_blended = compute_D_O_with_flow(d_core, d_flow_val, q_flow, alpha=alpha)

    # --- D_I stabilization ---
    d_flow_stabilized = compute_D_I_stabilized(d_i, d_flow_val, q_flow)

    return DFlowResult(
        tier_used=tier_used,
        tier_decision=final_decision,
        residual=residual,
        sinkhorn=sinkhorn_result,
        q_flow=q_flow,
        d_flow=d_flow_val,
        d_o_blended=d_o_blended,
        d_flow_stabilized=d_flow_stabilized,
    )
