"""ScoreEventRuleInducer: abstract a +1 scoring example into a coordinate/id-free
relational rule, and transfer it to the next level by role-signature matching.

The reframing this implements (established with the user, 2026-06-06):

  The system reaches +1 and HAS a positive example (a score-bearing rod), but it
  stores it as a coordinate / comp_N / state-hash INSTANCE, not as a RULE. So it
  cannot transfer the success to the next level (where the target sits at
  different coordinates with reshuffled component ids). Humans induce the
  relational rule from the single success ("cover the marked spot", "rotate to
  match") and re-identify the analogous object next level by its RELATIONAL
  signature, not its coordinates.

This module is the post-score half (much more tractable than pre-score goal
guessing, because it has a positive example):

    score_delta>0 rod + frame objects
      -> induce()  -> RelationalScoreRuleHypothesis  (drop coords/comp_N)
      -> transfer(next-level objects) -> candidate sites  (re-identify by signature)

HONESTY: this is Tier-0 relational abstraction — the signature is color / size /
rarity, not a learned relation model. It can re-identify "the same KIND of object"
across coordinate/id/layout changes (the controls prove this), but whether the
abstracted rule is the CORRECT win rule is still uncertain. NO score claim here;
this module proves coordinate/id-free transfer of a scored example, which is the
necessary substrate. Pure-Python, dependency-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


# ─── coordinate/id-free object signature ────────────────────────────────────

def _size_bucket(size: int) -> str:
    s = int(size or 0)
    if s <= 1:
        return "point"
    if s <= 4:
        return "small"
    if s <= 16:
        return "medium"
    return "large"


def _color_rarity(color: Any, color_counts: dict[Any, int]) -> str:
    n = int(color_counts.get(color, 0))
    if n <= 1:
        return "unique"
    if n <= 2:
        return "rare"
    return "common"


def _frame_color_counts(objects: list[dict[str, Any]]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for o in objects:
        counts[o.get("color")] = counts.get(o.get("color"), 0) + 1
    return counts


@dataclass
class RoleSignature:
    """A coordinate/id-free descriptor of an object's KIND. Deliberately excludes
    absolute coordinate and component_id (the non-invariant features)."""
    color: Any = None
    size_bucket: str = ""
    color_rarity: str = ""

    @classmethod
    def of(cls, obj: dict[str, Any], color_counts: dict[Any, int]) -> "RoleSignature":
        return cls(
            color=obj.get("color"),
            size_bucket=_size_bucket(int(obj.get("size", 0) or 0)),
            color_rarity=_color_rarity(obj.get("color"), color_counts),
        )

    def match_score(self, obj: dict[str, Any], color_counts: dict[Any, int]) -> float:
        """[0,1] similarity to an object by KIND (not position/id). Colour is the
        primary key: a colour mismatch caps the score below the match threshold so
        a same-size/same-rarity but wrong-colour object does NOT transfer."""
        score = 0.0
        if self.size_bucket and _size_bucket(int(obj.get("size", 0) or 0)) == self.size_bucket:
            score += 0.25
        if self.color_rarity and _color_rarity(obj.get("color"), color_counts) == self.color_rarity:
            score += 0.15
        if self.color is not None and obj.get("color") == self.color:
            score += 0.6
        elif self.color is not None:
            # colour mismatch: cap below threshold (size+rarity alone never qualify)
            return round(min(score, 0.4), 4)
        return round(score, 4)

    def to_dict(self) -> dict[str, Any]:
        return {"color": self.color, "size_bucket": self.size_bucket,
                "color_rarity": self.color_rarity}


@dataclass
class RelationalScoreRuleHypothesis:
    source_score_event_id: str = ""
    rule_family: str = ""              # interaction_at_target / reach_target / cover / match / ...
    actor_role: RoleSignature | None = None
    target_role: RoleSignature | None = None
    relation_before: str = ""
    relation_after: str = ""
    action_operator: str = ""          # click / move / rotate / ...
    invariant_features: tuple[str, ...] = ()
    # absolute coordinate + component_id are ALWAYS non-invariant (the whole point)
    non_invariant_features: tuple[str, ...] = ("absolute_coordinate", "component_id", "state_hash")
    confidence: float = 0.0
    status: str = "tentative"          # tentative / transfer_ready / confirmed / falsified

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_score_event_id": self.source_score_event_id,
            "rule_family": self.rule_family,
            "actor_role": self.actor_role.to_dict() if self.actor_role else None,
            "target_role": self.target_role.to_dict() if self.target_role else None,
            "relation_before": self.relation_before,
            "relation_after": self.relation_after,
            "action_operator": self.action_operator,
            "invariant_features": list(self.invariant_features),
            "non_invariant_features": list(self.non_invariant_features),
            "confidence": round(self.confidence, 4),
            "status": self.status,
        }


@dataclass
class TransferCandidate:
    component_id: str
    site: tuple[int, int]
    action_operator: str
    match_score: float

    def to_dict(self) -> dict[str, Any]:
        return {"component_id": self.component_id, "site": list(self.site),
                "action_operator": self.action_operator,
                "match_score": round(self.match_score, 4)}


class ScoreEventRuleInducer:
    """Stage 1-4: induce a relational rule from a scored example and transfer it."""

    def __init__(self, *, min_match: float = 0.5) -> None:
        self.min_match = float(min_match)
        self._rules: list[RelationalScoreRuleHypothesis] = []

    def reset(self) -> None:
        self._rules = []

    @property
    def rules(self) -> list[RelationalScoreRuleHypothesis]:
        return list(self._rules)

    # ── Stage 1-2: induce (abstract; drop coords/comp_N) ────────────────

    def induce(
        self,
        *,
        target_object: dict[str, Any] | None,
        objects: Iterable[dict[str, Any]],
        action_operator: str,
        actor_object: dict[str, Any] | None = None,
        rule_family: str = "",
        source_score_event_id: str = "",
    ) -> RelationalScoreRuleHypothesis | None:
        """Abstract a scored example into a coordinate/id-free rule. Returns None
        when there is no score-bearing target to generalize from (score event
        removed -> no rule)."""
        if target_object is None:
            return None
        obj_list = [o for o in (objects or []) if isinstance(o, dict)]
        counts = _frame_color_counts(obj_list)
        target_role = RoleSignature.of(target_object, counts)
        actor_role = RoleSignature.of(actor_object, counts) if actor_object else None
        family = rule_family or self._infer_family(action_operator, actor_object is not None)
        # confidence: a distinctive (unique-colour) target generalizes more reliably.
        conf = 0.5
        if target_role.color_rarity == "unique":
            conf = 0.8
        elif target_role.color_rarity == "rare":
            conf = 0.65
        rule = RelationalScoreRuleHypothesis(
            source_score_event_id=str(source_score_event_id or ""),
            rule_family=family,
            actor_role=actor_role,
            target_role=target_role,
            relation_before="target_unsatisfied",
            relation_after="target_satisfied",
            action_operator=str(action_operator or ""),
            invariant_features=("color", "size_bucket", "color_rarity"),
            confidence=conf,
            status="transfer_ready",
        )
        self._rules.append(rule)
        return rule

    @staticmethod
    def _infer_family(action_operator: str, has_actor: bool) -> str:
        op = str(action_operator or "").strip().lower()
        if op in ("move", "route", "walk"):
            return "reach_target" if has_actor else "move_target"
        if op in ("click", "action6"):
            return "interaction_at_target"
        if op in ("rotate",):
            return "rotate_to_match"
        return "interaction_at_target"

    # ── Stage 3-4: transfer (re-identify by signature; coordinate-free) ──

    def transfer(
        self,
        rule: RelationalScoreRuleHypothesis | None,
        objects: Iterable[dict[str, Any]],
        *,
        top_k: int = 4,
    ) -> list[TransferCandidate]:
        """Re-identify the analogous target on a NEW frame by role signature and
        emit candidate sites. Uses NO coordinate or component-id from the source
        level — only the KIND signature."""
        if rule is None or rule.target_role is None:
            return []
        obj_list = [o for o in (objects or []) if isinstance(o, dict) and o.get("center") is not None]
        if not obj_list:
            return []
        counts = _frame_color_counts(obj_list)
        scored: list[TransferCandidate] = []
        for o in obj_list:
            m = rule.target_role.match_score(o, counts)
            if m < self.min_match:
                continue
            c = o.get("center")
            scored.append(TransferCandidate(
                component_id=str(o.get("component_id", "") or ""),
                site=(int(c[0]), int(c[1])),
                action_operator=rule.action_operator,
                match_score=m,
            ))
        scored.sort(key=lambda t: t.match_score, reverse=True)
        return scored[:top_k]


# ── scoring ACTION SEQUENCE induction + transfer (operation, not just target) ──
#
# A single target re-identification is necessary-but-not-sufficient: the scoring
# RULE is usually an OPERATION (a short sequence of acts on role-typed objects),
# not "touch the scored colour". This captures the last K acts leading to a score
# as an ordered list of (RoleSignature, operator) — a forward-model-FREE program —
# and transfers it by re-identifying each step's role on the next level. For games
# whose scoring is a fixed click/act sequence on identifiable objects, replaying
# the program on role-matches transfers without coordinates.

@dataclass
class ScoringProgramStep:
    role: RoleSignature
    operator: str

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role.to_dict(), "operator": self.operator}


@dataclass
class ScoringProgram:
    steps: tuple[ScoringProgramStep, ...] = ()
    source_score_event_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps],
                "source_score_event_id": self.source_score_event_id}


def induce_scoring_program(
    rod_window: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    *,
    source_score_event_id: str = "",
    max_steps: int = 4,
) -> ScoringProgram | None:
    """Abstract the last K rods (ending at the scoring rod) into a coordinate-free
    ordered program of (role_signature, operator). Each rod's acted site is mapped
    to the nearest current object and abstracted to its KIND signature."""
    if not rod_window:
        return None
    counts = _frame_color_counts([o for o in objects if isinstance(o, dict)])
    steps: list[ScoringProgramStep] = []
    for rd in rod_window[-max_steps:]:
        t = target_object_from_rod(rd, objects)
        if t is None:
            continue
        steps.append(ScoringProgramStep(
            role=RoleSignature.of(t, counts),
            operator=operator_from_rod(rd),
        ))
    if not steps:
        return None
    return ScoringProgram(steps=tuple(steps), source_score_event_id=str(source_score_event_id or ""))


def transfer_scoring_program(
    program: ScoringProgram | None,
    objects: list[dict[str, Any]],
    *,
    min_match: float = 0.5,
) -> list[tuple[int, int]]:
    """Re-identify each program step's role on a NEW frame -> ordered candidate
    sites (coordinate/id-free). Returns the ordered site list (the plan to replay).
    Empty if any step cannot be re-identified (the program does not transfer)."""
    if program is None or not program.steps:
        return []
    obj_list = [o for o in objects if isinstance(o, dict) and o.get("center") is not None]
    if not obj_list:
        return []
    counts = _frame_color_counts(obj_list)
    sites: list[tuple[int, int]] = []
    for step in program.steps:
        best, best_m = None, -1.0
        for o in obj_list:
            m = step.role.match_score(o, counts)
            if m > best_m:
                best_m, best = m, o
        if best is None or best_m < min_match:
            return []  # a step could not be re-identified -> no transfer
        c = best.get("center")
        sites.append((int(c[0]), int(c[1])))
    return sites


# ── live-path adapter: scored rod -> target object (nearest to the rod site) ──

def target_object_from_rod(
    rod_dict: dict[str, Any], objects: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Map a scored CausalTraceRod.to_dict() to the frame object nearest its acted
    site (target_key = [action, x, y, kind]). Returns None if unmappable."""
    if not isinstance(rod_dict, dict):
        return None
    tk = rod_dict.get("target_key")
    if not (isinstance(tk, (list, tuple)) and len(tk) >= 3):
        return None
    try:
        sx, sy = int(tk[1]), int(tk[2])
    except (TypeError, ValueError):
        return None
    best, best_d = None, None
    for o in objects or []:
        c = o.get("center")
        if c is None or len(c) < 2:
            continue
        d = abs(int(c[0]) - sx) + abs(int(c[1]) - sy)
        if best_d is None or d < best_d:
            best_d, best = d, o
    return best


def operator_from_rod(rod_dict: dict[str, Any]) -> str:
    eff = rod_dict.get("effect") if isinstance(rod_dict.get("effect"), dict) else {}
    action = str(rod_dict.get("action_name", "") or eff.get("action_name", "") or "")
    a = action.strip().upper()
    if a == "ACTION6":
        return "click"
    if a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        return "move"
    return "auxiliary"
