"""Symbolic pattern mining over CausalTraceRod collections (Gap 6 trial).

Reads a population of rods, groups them by a coarse structural signature,
and proposes emergent mechanic candidates when a signature accumulates
enough support without crossing the contradiction tolerance.

Design rules:

- No LLM. Purely symbolic clustering over rod fields.
- Signature ignores specific coordinates and numerical values; only
  the cause-effect skeleton + generic_keys + structural shape of
  before_sig/after_sig matter.
- Candidates carry provenance: distinct_episodes / distinct_games /
  distinct_variants so cross-game evidence is visible.
- Promotion thresholds mirror RodPromotionPolicy in structural_rods.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .structural_rods import CausalTraceRod


def rod_signature(rod: CausalTraceRod) -> str:
    """Coarse structural signature for clustering.

    Captures the cause-effect skeleton without coords or step indices.
    Two rods sharing this signature are claimed to be evidence of the
    same underlying mechanic family.
    """
    return _signature_parts(
        effect_type=rod.effect.effect_type,
        action_name=rod.action_name,
        target_kind=rod.target_kind,
        operator=rod.operator,
        generic_keys=rod.generic_keys or (),
        before_sig=rod.before_sig,
        after_sig=rod.after_sig,
    )


def rod_signature_from_dict(rod_dict: Mapping[str, Any]) -> str:
    """Same signature as rod_signature() but takes a serialized rod dict.

    Use this on JSON-loaded trace rows from benchmark artefacts.
    """
    effect = rod_dict.get("effect") or {}
    generic_keys = rod_dict.get("generic_keys") or ()
    return _signature_parts(
        effect_type=effect.get("effect_type") if isinstance(effect, Mapping) else "",
        action_name=rod_dict.get("action_name"),
        target_kind=rod_dict.get("target_kind"),
        operator=rod_dict.get("operator"),
        generic_keys=generic_keys,
        before_sig=rod_dict.get("before_sig") or {},
        after_sig=rod_dict.get("after_sig") or {},
    )


def _signature_parts(
    *,
    effect_type: Any,
    action_name: Any,
    target_kind: Any,
    operator: Any,
    generic_keys: Iterable[Any],
    before_sig: Mapping[str, Any] | None,
    after_sig: Mapping[str, Any] | None,
) -> str:
    parts = [
        str(effect_type or ""),
        str(action_name or ""),
        str(target_kind or ""),
        str(operator or ""),
        ",".join(sorted(str(k) for k in (generic_keys or ()))),
        _sig_shape(before_sig),
        _sig_shape(after_sig),
    ]
    return "|".join(parts)


def _sig_shape(payload: Mapping[str, Any] | None) -> str:
    """Structural fingerprint of a sig payload.

    Records sorted keys with value-type tags. Ignores numeric values to
    abstract away from concrete coordinates.
    """
    if not payload:
        return "{}"
    items: list[str] = []
    for k in sorted(str(key) for key in payload.keys()):
        v = payload[k] if k in payload else payload.get(k)
        items.append(f"{k}:{_type_tag(v)}")
    return "{" + ",".join(items) + "}"


def _type_tag(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "num"
    if isinstance(v, str):
        return "str"
    if isinstance(v, (list, tuple)):
        return "seq"
    if isinstance(v, dict):
        return "dict"
    if v is None:
        return "none"
    return "?"


@dataclass
class RodCluster:
    """One bucket of rods sharing the same signature."""

    signature: str
    rod_count: int = 0
    support_total: int = 0
    contradiction_total: int = 0
    distinct_episodes: set[str] = field(default_factory=set)
    distinct_variants: set[str] = field(default_factory=set)
    distinct_games: set[str] = field(default_factory=set)
    sample_rods: list[Any] = field(default_factory=list)

    def add(
        self,
        rod: CausalTraceRod,
        *,
        episode_id: str = "",
        variant: str = "",
        game_id: str = "",
    ) -> None:
        self.rod_count += 1
        self.support_total += int(rod.support_count or 0)
        self.contradiction_total += int(rod.contradiction_count or 0)
        if episode_id:
            self.distinct_episodes.add(str(episode_id))
        if variant:
            self.distinct_variants.add(str(variant))
        if game_id:
            self.distinct_games.add(str(game_id))
        if len(self.sample_rods) < 5:
            self.sample_rods.append(rod)

    def add_from_dict(
        self,
        rod_dict: Mapping[str, Any],
        *,
        episode_id: str = "",
        variant: str = "",
        game_id: str = "",
    ) -> None:
        """Add a serialized rod (e.g. loaded from JSON trace artefact)."""
        self.rod_count += 1
        self.support_total += int(rod_dict.get("support_count") or 0)
        self.contradiction_total += int(rod_dict.get("contradiction_count") or 0)
        if episode_id:
            self.distinct_episodes.add(str(episode_id))
        if variant:
            self.distinct_variants.add(str(variant))
        if game_id:
            self.distinct_games.add(str(game_id))
        if len(self.sample_rods) < 5:
            self.sample_rods.append(dict(rod_dict))

    @property
    def contradiction_rate(self) -> float:
        total = self.support_total + self.contradiction_total
        return self.contradiction_total / total if total > 0 else 0.0

    @property
    def net_support(self) -> int:
        return self.support_total - self.contradiction_total


@dataclass
class EmergentMechanicCandidate:
    """A signature that has crossed promotion thresholds."""

    signature: str
    rod_count: int
    net_support: int
    contradiction_rate: float
    distinct_episodes: int
    distinct_variants: int
    distinct_games: int
    sample_rod_dicts: list[dict[str, Any]]
    confidence_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "rod_count": int(self.rod_count),
            "net_support": int(self.net_support),
            "contradiction_rate": round(float(self.contradiction_rate), 6),
            "distinct_episodes": int(self.distinct_episodes),
            "distinct_variants": int(self.distinct_variants),
            "distinct_games": int(self.distinct_games),
            "sample_rod_dicts": list(self.sample_rod_dicts),
            "confidence_score": round(float(self.confidence_score), 6),
        }


def cluster_rods(
    rods: Iterable[CausalTraceRod],
    *,
    episode_ids: Iterable[str] | None = None,
    variants: Iterable[str] | None = None,
    game_ids: Iterable[str] | None = None,
) -> dict[str, RodCluster]:
    """Cluster a population of rods by signature.

    The optional iterables (episode_ids / variants / game_ids) are
    zipped 1:1 with rods. When omitted, distinct_* sets stay empty.
    """
    rods_list = list(rods)
    n = len(rods_list)
    ep_list = _pad_strs(episode_ids, n)
    var_list = _pad_strs(variants, n)
    game_list = _pad_strs(game_ids, n)

    clusters: dict[str, RodCluster] = {}
    for rod, ep, var, gid in zip(rods_list, ep_list, var_list, game_list):
        sig = rod_signature(rod)
        if sig not in clusters:
            clusters[sig] = RodCluster(signature=sig)
        clusters[sig].add(rod, episode_id=ep, variant=var, game_id=gid)

    return clusters


def cluster_rod_dicts(
    rod_dicts: Iterable[Mapping[str, Any]],
    *,
    episode_ids: Iterable[str] | None = None,
    variants: Iterable[str] | None = None,
    game_ids: Iterable[str] | None = None,
) -> dict[str, RodCluster]:
    """Cluster serialized rod dicts (e.g. JSON-loaded trace artefacts).

    Produces the same RodCluster shape as cluster_rods() so downstream
    `propose_emergent_candidates` and `summarize_pattern_mining` work
    unchanged.
    """
    dict_list = list(rod_dicts)
    n = len(dict_list)
    ep_list = _pad_strs(episode_ids, n)
    var_list = _pad_strs(variants, n)
    game_list = _pad_strs(game_ids, n)

    clusters: dict[str, RodCluster] = {}
    for rod_d, ep, var, gid in zip(dict_list, ep_list, var_list, game_list):
        sig = rod_signature_from_dict(rod_d)
        if sig not in clusters:
            clusters[sig] = RodCluster(signature=sig)
        clusters[sig].add_from_dict(rod_d, episode_id=ep, variant=var, game_id=gid)

    return clusters


def _pad_strs(seq: Iterable[str] | None, n: int) -> list[str]:
    if seq is None:
        return [""] * n
    result = [str(x) for x in seq]
    if len(result) < n:
        result.extend([""] * (n - len(result)))
    return result[:n]


def propose_emergent_candidates(
    clusters: Mapping[str, RodCluster],
    *,
    min_rod_count: int = 3,
    min_net_support: int = 2,
    max_contradiction_rate: float = 0.25,
    min_distinct_episodes: int = 0,
    min_distinct_variants: int = 0,
    cross_game_bonus: bool = True,
) -> list[EmergentMechanicCandidate]:
    """Pick clusters that pass promotion thresholds.

    Thresholds mirror RodPromotionPolicy defaults so accepted candidates
    are comparable to existing MechanicRod promotions.
    """
    candidates: list[EmergentMechanicCandidate] = []
    for sig, cluster in clusters.items():
        if cluster.rod_count < min_rod_count:
            continue
        if cluster.net_support < min_net_support:
            continue
        if cluster.contradiction_rate > max_contradiction_rate:
            continue
        if len(cluster.distinct_episodes) < min_distinct_episodes:
            continue
        if len(cluster.distinct_variants) < min_distinct_variants:
            continue

        support_density = (
            cluster.net_support / cluster.rod_count if cluster.rod_count > 0 else 0.0
        )
        cross_game_factor = (
            1.0
            if (cross_game_bonus and len(cluster.distinct_games) > 1)
            else 0.5
        )
        confidence = max(0.0, min(1.0, support_density * cross_game_factor))

        sample_dicts: list[dict[str, Any]] = []
        for r in cluster.sample_rods[:3]:
            if isinstance(r, dict):
                sample_dicts.append(dict(r))
            elif hasattr(r, "to_dict"):
                sample_dicts.append(r.to_dict())
        candidates.append(
            EmergentMechanicCandidate(
                signature=sig,
                rod_count=cluster.rod_count,
                net_support=cluster.net_support,
                contradiction_rate=cluster.contradiction_rate,
                distinct_episodes=len(cluster.distinct_episodes),
                distinct_variants=len(cluster.distinct_variants),
                distinct_games=len(cluster.distinct_games),
                sample_rod_dicts=sample_dicts,
                confidence_score=confidence,
            )
        )

    candidates.sort(key=lambda c: (c.confidence_score, c.rod_count), reverse=True)
    return candidates


def summarize_pattern_mining(
    clusters: Mapping[str, RodCluster],
    candidates: list[EmergentMechanicCandidate],
) -> dict[str, Any]:
    """Slim summary suitable for inclusion in step14 artefacts."""
    rod_total = sum(c.rod_count for c in clusters.values())
    return {
        "cluster_count": len(clusters),
        "rod_total": int(rod_total),
        "candidate_count": len(candidates),
        "top_candidate_confidence": (
            round(float(candidates[0].confidence_score), 6) if candidates else 0.0
        ),
        "candidate_signatures": [c.signature for c in candidates[:10]],
        "cluster_size_histogram": _cluster_size_histogram(clusters),
    }


def _cluster_size_histogram(clusters: Mapping[str, RodCluster]) -> dict[str, int]:
    """Coarse histogram of cluster sizes."""
    hist = {"1": 0, "2": 0, "3-5": 0, "6-10": 0, "11+": 0}
    for c in clusters.values():
        n = c.rod_count
        if n <= 1:
            hist["1"] += 1
        elif n == 2:
            hist["2"] += 1
        elif n <= 5:
            hist["3-5"] += 1
        elif n <= 10:
            hist["6-10"] += 1
        else:
            hist["11+"] += 1
    return hist
