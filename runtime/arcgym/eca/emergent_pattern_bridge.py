"""Production bridge: rod stream -> emergent hypotheses for MechanicHypothesisInducer.

This is the integration point that turns the Gap 6 discovery loop
(rod_pattern_mining + emergent_candidate_injector) into something the
production MechanicHypothesisInducer.build() can accept via its new
external_emergent_hypotheses kwarg.

Design rules (matches docs/eca_psa_internal_llm_role_design_2026-05-16.md):

- Opt-in via ECA_ENABLE_EMERGENT_PATTERN_INDUCTION=1 env var.
- Default OFF -> production behavior unchanged when disabled.
- Promotes ONLY candidates that pass the novelty filter (not in
  PRIMITIVE_SCOPE_REGISTRY).
- Emits MechanicHypothesis with source_kind="emergent_pattern_induced"
  so downstream auditing can separate them from reader_fixed /
  ledger_induced / psa_llm_proposed contributions.
- Pure symbolic; NO LLM call at this layer (the LLM, if used, must
  feed rods upstream via gemma_rod_proposer or similar).

Usage from production runtime:

    from arcgym.eca.emergent_pattern_bridge import build_emergent_hypotheses
    extras = build_emergent_hypotheses(
        rod_population=...,        # list[CausalTraceRod] or list[dict]
        episode_ids=...,
        variants=...,
        game_ids=...,
    )
    hypotheses = MechanicHypothesisInducer().build(
        cycle_states=...,
        records=...,
        external_emergent_hypotheses=extras,
    )

The env-flag check is done by the caller; this module's public API
always honours its inputs. The flag exists so callers can short-circuit
without computing rod clusters at all when disabled.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Mapping

from dataclasses import replace as _dc_replace

from .affordance_hypotheses import AffordanceHypothesis
from .cross_game_evidence_registry import CrossGameRodSignatureRegistry
from .emergent_candidate_injector import (
    SOURCE_KIND_EMERGENT_PATTERN,
    InjectionResult,
    inject_all_affordance_candidates,
    inject_novel_affordance_candidates,
    inject_novel_candidates,
)
from .mechanic_hypotheses import MechanicHypothesis
from .primitive_scope_registry import PRIMITIVE_SCOPE_REGISTRY
from .rod_pattern_mining import (
    cluster_rod_dicts,
    cluster_rods,
    propose_emergent_candidates,
)
from .structural_rods import CausalTraceRod, StructuralRodMemory


_ENV_FLAG = "ECA_ENABLE_EMERGENT_PATTERN_INDUCTION"
_ENV_NOVELTY_MODE = "ECA_EMERGENT_AFFORDANCE_NOVELTY_MODE"

NOVELTY_MODE_PRIMITIVE_ID = "primitive_id"
NOVELTY_MODE_SITE = "site"


def is_enabled() -> bool:
    """True if the env flag is on. Callers may use this to skip work."""
    return str(os.environ.get(_ENV_FLAG, "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def novelty_mode() -> str:
    """Read ECA_EMERGENT_AFFORDANCE_NOVELTY_MODE.

    Returns NOVELTY_MODE_PRIMITIVE_ID (default, strict) or
    NOVELTY_MODE_SITE (loose, bypass primitive_id filter). Unknown values
    fall back to primitive_id mode.
    """
    raw = str(os.environ.get(_ENV_NOVELTY_MODE, "") or "").strip().lower()
    if raw == NOVELTY_MODE_SITE:
        return NOVELTY_MODE_SITE
    return NOVELTY_MODE_PRIMITIVE_ID


def _classify_novelty(sample_rod_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Same logic as tools/rod_pattern_mining_cli._classify_candidate_novelty.

    Inlined here to keep the production module CLI-free.
    """
    registered: set[str] = set()
    unregistered: set[str] = set()
    for s in sample_rod_dicts:
        pid = str((s or {}).get("primitive_id") or "").strip()
        if not pid:
            continue
        if pid in PRIMITIVE_SCOPE_REGISTRY:
            registered.add(pid)
        else:
            unregistered.add(pid)
    return {
        "is_novel": bool(unregistered and not registered),
        "is_mixed": bool(registered and unregistered),
        "registered_primitive_ids": sorted(registered),
        "unregistered_primitive_ids": sorted(unregistered),
    }


def build_emergent_hypotheses(
    *,
    rod_population: Iterable[CausalTraceRod] | Iterable[Mapping[str, Any]] = (),
    episode_ids: Iterable[str] | None = None,
    variants: Iterable[str] | None = None,
    game_ids: Iterable[str] | None = None,
    min_rod_count: int = 3,
    min_net_support: int = 2,
    max_contradiction_rate: float = 0.25,
    family: str = "emergent",
) -> list[MechanicHypothesis]:
    """Produce emergent MechanicHypothesis objects from a rod population.

    Accepts either CausalTraceRod objects (uses cluster_rods) or raw
    rod-shaped dicts (uses cluster_rod_dicts). The shape is auto-detected
    from the first element.

    Returns an empty list when the rod population is empty or when no
    candidate passes the threshold + novelty gates.
    """
    pop = list(rod_population)
    if not pop:
        return []

    is_dict = isinstance(pop[0], Mapping)
    if is_dict:
        clusters = cluster_rod_dicts(
            pop,
            episode_ids=episode_ids,
            variants=variants,
            game_ids=game_ids,
        )
    else:
        clusters = cluster_rods(
            pop,
            episode_ids=episode_ids,
            variants=variants,
            game_ids=game_ids,
        )

    candidates = propose_emergent_candidates(
        clusters,
        min_rod_count=min_rod_count,
        min_net_support=min_net_support,
        max_contradiction_rate=max_contradiction_rate,
    )

    novelty_payloads = [_classify_novelty(c.sample_rod_dicts) for c in candidates]
    results: list[InjectionResult] = inject_novel_candidates(
        candidates,
        novelty_payloads,
        family=family,
    )

    return [r.hypothesis for r in results]


def gather_emergent_extras(
    rod_memory: StructuralRodMemory,
    *,
    episode_id: str = "",
    variant: str = "",
    game_id: str = "",
    min_rod_count: int = 3,
    respect_env_flag: bool = True,
    family: str = "emergent",
) -> list[MechanicHypothesis]:
    """Drop-in helper for production callers.

    Pulls the current rod population from a StructuralRodMemory and
    returns external_emergent_hypotheses extras compatible with
    MechanicHypothesisInducer.build(external_emergent_hypotheses=...).

    Use:
        extras = gather_emergent_extras(self._structural_rod_memory,
                                        episode_id=str(self._epoch_id),
                                        variant="canonical",
                                        game_id=self.game_id)
        hypotheses = self._hypothesis_inducer.build(
            cycle_states=cycle_states,
            records=records,
            external_emergent_hypotheses=extras,
        )

    When respect_env_flag is True (default) and the env flag is OFF,
    returns []. Set respect_env_flag=False to bypass the gate (e.g.
    inside tests).
    """
    if respect_env_flag and not is_enabled():
        return []
    rods = list(rod_memory.rods) if rod_memory is not None else []
    if not rods:
        return []
    n = len(rods)
    return build_emergent_hypotheses(
        rod_population=rods,
        episode_ids=[f"{episode_id or 'ep'}|rod{i}" for i in range(n)],
        variants=[variant or "default"] * n,
        game_ids=[game_id or "unknown"] * n,
        min_rod_count=min_rod_count,
        family=family,
    )


def build_emergent_affordances(
    *,
    rod_population: Iterable[CausalTraceRod] | Iterable[Mapping[str, Any]] = (),
    episode_ids: Iterable[str] | None = None,
    variants: Iterable[str] | None = None,
    game_ids: Iterable[str] | None = None,
    min_rod_count: int = 3,
    min_net_support: int = 2,
    max_contradiction_rate: float = 0.25,
    pre_state_hash: str = "",
    mode: str | None = None,
) -> list[AffordanceHypothesis]:
    """Method A: produce AffordanceHypothesis objects from rod population.

    Mirrors build_emergent_hypotheses but emits the AffordanceHypothesis
    type so the parallel session's AffordanceHypothesisInducer +
    AffordanceProbeCompiler + unknown_affordance_probe pipeline can
    consume them. Each emitted hypothesis carries
    source_kind="emergent_pattern_induced".

    `mode` selects the novelty contract:
    - "primitive_id" (default): strict, skip clusters where any sample's
      primitive_id is in PRIMITIVE_SCOPE_REGISTRY. Returns [] for the
      common strict-generic case where all firing primitives are
      registered.
    - "site": bypass the primitive_id novelty filter and emit every
      passing candidate. Downstream AffordanceHypothesisInducer's
      known_reader_keys mask handles site-level deduplication. Use when
      you want clusters on registered primitives at novel (action, x, y)
      sites to fire.

    If `mode` is None, the env var ECA_EMERGENT_AFFORDANCE_NOVELTY_MODE
    decides (default "primitive_id").
    """
    pop = list(rod_population)
    if not pop:
        return []

    is_dict = isinstance(pop[0], Mapping)
    if is_dict:
        clusters = cluster_rod_dicts(
            pop,
            episode_ids=episode_ids,
            variants=variants,
            game_ids=game_ids,
        )
    else:
        clusters = cluster_rods(
            pop,
            episode_ids=episode_ids,
            variants=variants,
            game_ids=game_ids,
        )

    candidates = propose_emergent_candidates(
        clusters,
        min_rod_count=min_rod_count,
        min_net_support=min_net_support,
        max_contradiction_rate=max_contradiction_rate,
    )

    effective_mode = mode if mode is not None else novelty_mode()
    if effective_mode == NOVELTY_MODE_SITE:
        return inject_all_affordance_candidates(
            candidates,
            pre_state_hash=pre_state_hash,
        )

    novelty_payloads = [_classify_novelty(c.sample_rod_dicts) for c in candidates]
    return inject_novel_affordance_candidates(
        candidates,
        novelty_payloads,
        pre_state_hash=pre_state_hash,
    )


def gather_emergent_affordance_extras(
    rod_memory: StructuralRodMemory,
    *,
    episode_id: str = "",
    variant: str = "",
    game_id: str = "",
    min_rod_count: int = 3,
    respect_env_flag: bool = True,
    pre_state_hash: str = "",
    mode: str | None = None,
    score_event_steps: list[int] | None = None,
    score_correlation_window: int = 0,
    cross_game_registry: CrossGameRodSignatureRegistry | None = None,
) -> list[AffordanceHypothesis]:
    """Method A drop-in for production callers.

    Mirrors gather_emergent_extras but returns AffordanceHypothesis
    objects compatible with AffordanceHypothesisInducer.build(
    external_emergent_affordances=...).

    `mode` controls the novelty contract; see build_emergent_affordances.
    None falls back to env var ECA_EMERGENT_AFFORDANCE_NOVELTY_MODE.

    `score_event_steps` + `score_correlation_window` enable the
    score-bearing rod tagging filter (research branch
    `score-bearing-rod-tagging`, 2026-05-22). When the window is > 0
    and event-steps is non-empty, only rods whose insertion step is
    within `window` of any score event step contribute to clusters.
    Other rods are dropped. This narrows emergent candidates to
    structurally-score-correlated patterns.
    """
    if respect_env_flag and not is_enabled():
        return []
    rods = list(rod_memory.rods) if rod_memory is not None else []
    if not rods:
        return []

    # Score-bearing rod tagging filter
    if (
        score_correlation_window
        and score_correlation_window > 0
        and score_event_steps
        and rod_memory is not None
    ):
        window = int(score_correlation_window)
        events = sorted({int(s) for s in score_event_steps})
        filtered_pairs = [
            (rod, step)
            for (rod, step) in rod_memory.iter_rods_with_steps()
            if isinstance(step, int) and step >= 0
            and any(abs(step - s) <= window for s in events)
        ]
        if not filtered_pairs:
            return []
        rods = [pair[0] for pair in filtered_pairs]

    n = len(rods)
    affordances = build_emergent_affordances(
        rod_population=rods,
        episode_ids=[f"{episode_id or 'ep'}|rod{i}" for i in range(n)],
        variants=[variant or "default"] * n,
        game_ids=[game_id or "unknown"] * n,
        min_rod_count=min_rod_count,
        pre_state_hash=pre_state_hash,
        mode=mode,
    )

    # Cross-game emergent evidence boost: if a candidate's signature
    # is in the registry across multiple games, multiply its confidence
    # by the boost factor (capped at 2.0). Single-game signatures
    # produce no boost.
    if cross_game_registry is not None and affordances:
        boosted: list[AffordanceHypothesis] = []
        for affordance in affordances:
            metadata = affordance.metadata or {}
            sig = str(metadata.get("candidate_signature") or "")
            boost = cross_game_registry.boost_factor_for_signature(sig)
            if boost > 1.0:
                new_conf = min(2.0, float(affordance.confidence) * boost)
                # Record boost provenance into metadata for diagnostics
                new_meta = dict(metadata)
                new_meta["cross_game_boost_factor"] = boost
                new_meta["cross_game_signature_games"] = sorted(
                    cross_game_registry.signature_score_games(sig)
                )
                boosted.append(
                    _dc_replace(
                        affordance,
                        confidence=new_conf,
                        metadata=new_meta,
                    )
                )
            else:
                boosted.append(affordance)
        affordances = boosted

    return affordances


def emergent_pattern_summary(
    *,
    rod_population: Iterable[CausalTraceRod] | Iterable[Mapping[str, Any]] = (),
    episode_ids: Iterable[str] | None = None,
    variants: Iterable[str] | None = None,
    game_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Slim metrics payload for inclusion in benchmark artefacts.

    Reports cluster_count / candidate_count / novel_count / hypothesis_count
    without modifying any execution state.
    """
    pop = list(rod_population)
    if not pop:
        return {
            "enabled": is_enabled(),
            "rod_total": 0,
            "cluster_count": 0,
            "candidate_count": 0,
            "novel_count": 0,
            "hypothesis_count": 0,
        }

    is_dict = isinstance(pop[0], Mapping)
    if is_dict:
        clusters = cluster_rod_dicts(
            pop,
            episode_ids=episode_ids,
            variants=variants,
            game_ids=game_ids,
        )
    else:
        clusters = cluster_rods(
            pop,
            episode_ids=episode_ids,
            variants=variants,
            game_ids=game_ids,
        )
    candidates = propose_emergent_candidates(clusters)
    novelty = [_classify_novelty(c.sample_rod_dicts) for c in candidates]
    novel_count = sum(1 for n in novelty if n["is_novel"])
    injected = inject_novel_candidates(candidates, novelty)

    return {
        "enabled": is_enabled(),
        "source_kind": SOURCE_KIND_EMERGENT_PATTERN,
        "rod_total": len(pop),
        "cluster_count": len(clusters),
        "candidate_count": len(candidates),
        "novel_count": novel_count,
        "hypothesis_count": len(injected),
        "env_flag": _ENV_FLAG,
    }
