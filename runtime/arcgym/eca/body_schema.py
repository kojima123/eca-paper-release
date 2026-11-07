"""BodySchema: an artificial body schema for ARC closed-loop play.

The premise (user, 2026-06-06): understanding a game's rule and deciding a
goal is NOT linguistic reasoning over a symbol table — it is closer to a
BODY SENSE: grasping *how my own actions affect the world, and which
incomplete relation I can complete by acting*. salience ("what stands out")
is a weak goal signal; the real goal candidate is

    an INCOMPLETE relation that MY action can COMPLETE.

This module organises the agent's own action-effect history into a body
schema and closes the loop over it. Seven layers, one convergent cycle:

    1. action-effect coupling   which action affects which object/colour
    2. controllable-object map  which colours respond to my actions
    3. affordance map           the EFFECT kind (translate/recolor/appear/...)
    4. tension map              which relations are INCOMPLETE (the "deficit")
    5. completion schema        the relational/motor operator that closes it
                                (reach / meet / cover / align / match)
    6. score-event abstraction  a +1 becomes (operator, role_signature) — the
                                OPERATION, not coordinates
    7. transfer                 reapply the learned schema at the next level

It is grounded in the proven lp85 / phase246 primitives
(`robust_movable_detector.detect_movement`,
`relational_objective_inducer.evaluate_relation`,
`score_closure` grounding) but reimplemented PURE-NUMPY (scipy is absent on
this box) and unified under one closed-loop control. The completion step uses
a LEARNED one-step forward model — per (colour, action) mean displacement
read off the affordance history — so it can pick "which action reaches" with
NO game-specific code and NO full simulator.

Reuses `RoleSignature` from `score_rule_inducer` so the score-event
abstraction and the transfer are coordinate/id-free.

HONESTY: this is a Tier-0 body schema. The tension heuristics are generic
(centroid distance / row-col share / overlap), not a learned relation model;
`converged()` means a *stable* chosen relation, not a *correct* one. The
module is OBSERVE-ONLY by construction — it returns a proposed action and a
BodyFrame; the host decides whether to consume them. With the host flag off
nothing here is constructed, so the host loop is byte-identical.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .score_rule_inducer import RoleSignature, _frame_color_counts
from .agent_identifier import ComponentAgentIdentifier

SCHEMA_VERSION = "eca_body_schema.v1"

MOVE_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
CLICK_ACTION = "ACTION6"
EFFECT_KINDS = ("translate", "recolor", "appear", "disappear")
COMPLETION_OPERATORS = ("reach", "meet", "cover", "align", "match")

# Posterior states (faithful to the phase325 / relevance_cycle vocabulary).
CONFIRMED = "confirmed"
PENDING = "pending"
FALSIFIED = "falsified"
REJECTED = "rejected"


# --------------------------------------------------------------------------- #
# Affordance / effect detection (pure-numpy; ported from robust_movable_detector)
# --------------------------------------------------------------------------- #
def _bg_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _dominant(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def detect_movement(prev: np.ndarray, cur: np.ndarray, bg: Optional[int] = None) -> Optional[dict]:
    """Robust changed-region movement summary, or None if nothing changed.

    Identity-free: it summarises the change between two frames as one of
    {translation, recolor, appear, disappear} plus the dominant moving colour
    and (for a translation) the displacement vector. Tolerant of ambiguous /
    shape-changing sprites because it works on changed cells, not on a sprite
    match.
    """
    prev = np.asarray(prev)
    cur = np.asarray(cur)
    if prev.shape != cur.shape:
        return None
    if bg is None:
        bg = _bg_color(prev)
    changed = prev != cur
    if not changed.any():
        return None
    appeared = changed & (cur != bg) & (prev == bg)
    disappeared = changed & (prev != bg) & (cur == bg)
    recolored = changed & (cur != bg) & (prev != bg)
    out: dict = {"kind": None, "color": None, "displacement": None,
                 "changed_count": int(changed.sum())}
    if appeared.any() and disappeared.any():
        ay, ax = np.where(appeared)
        dy, dx = np.where(disappeared)
        out["kind"] = "translate"
        out["color"] = _dominant(cur[appeared])
        out["displacement"] = (round(float(ax.mean()) - float(dx.mean()), 2),
                               round(float(ay.mean()) - float(dy.mean()), 2))
    elif recolored.any():
        out["kind"] = "recolor"
        out["color"] = _dominant(cur[recolored])
    elif appeared.any():
        out["kind"] = "appear"
        out["color"] = _dominant(cur[appeared])
    elif disappeared.any():
        out["kind"] = "disappear"
        out["color"] = _dominant(prev[disappeared])
    return out


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class AffordanceProfile:
    """Per-colour body-schema record: is this colour controllable, what does
    acting do to it, and (the learned one-step forward model) how far does
    each action displace it."""
    color: int
    act_count: int = 0
    change_count: int = 0
    effect_counts: Counter = field(default_factory=Counter)
    action_effect: Counter = field(default_factory=Counter)          # (action, kind) -> n
    action_disp: dict = field(default_factory=dict)                  # action -> list[(dx,dy)]

    @property
    def controllable(self) -> float:
        if self.act_count <= 0:
            return 0.0
        return round(self.change_count / self.act_count, 3)

    def dominant_effect(self) -> Optional[str]:
        if not self.effect_counts:
            return None
        return self.effect_counts.most_common(1)[0][0]

    def is_movable(self) -> bool:
        return self.effect_counts.get("translate", 0) > 0

    def predicted_displacement(self, action: str) -> Optional[tuple]:
        samples = self.action_disp.get(action)
        if not samples:
            return None
        xs = sum(s[0] for s in samples) / len(samples)
        ys = sum(s[1] for s in samples) / len(samples)
        return (round(xs, 2), round(ys, 2))

    def record(self, action: str, kind: str, displacement: Optional[tuple]) -> None:
        self.change_count += 1
        self.effect_counts[kind] += 1
        self.action_effect[(action, kind)] += 1
        if kind == "translate" and displacement is not None:
            buf = self.action_disp.setdefault(action, [])
            buf.append((float(displacement[0]), float(displacement[1])))
            if len(buf) > 8:
                del buf[0]

    def to_dict(self) -> dict:
        return {
            "color": self.color,
            "controllable": self.controllable,
            "dominant_effect": self.dominant_effect(),
            "is_movable": self.is_movable(),
            "effect_counts": dict(self.effect_counts),
            "learned_displacements": {a: self.predicted_displacement(a)
                                      for a in self.action_disp},
        }


@dataclass
class ClickMover:
    """Learned click->move affordance: how does clicking at a position move the
    agent? The strict-generic suite is CLICK-driven (the agent moves via ACTION6
    at a target, not ACTION1-4), so a fixed per-action displacement does not
    apply — the displacement is a function of the click position. This records
    (agent_prev, click, agent_new) samples and classifies the relationship."""
    samples: list = field(default_factory=list)

    def record(self, prev_xy: tuple, click_xy: tuple, new_xy: tuple) -> None:
        self.samples.append((tuple(prev_xy), tuple(click_xy), tuple(new_xy)))
        if len(self.samples) > 24:
            del self.samples[0]

    def kind(self) -> str:
        if len(self.samples) < 2:
            return "unknown"
        teleport = directional = 0
        for prev, click, new in self.samples:
            if abs(new[0] - prev[0]) + abs(new[1] - prev[1]) < 0.5:
                continue  # did not move
            if abs(new[0] - click[0]) + abs(new[1] - click[1]) <= 1.5:
                teleport += 1
            elif (click[0] - prev[0]) * (new[0] - prev[0]) + \
                    (click[1] - prev[1]) * (new[1] - prev[1]) > 0:
                directional += 1
        if teleport >= directional and teleport > 0:
            return "teleport"
        if directional > 0:
            return "directional"
        return "unknown"

    def active(self) -> bool:
        return self.kind() in ("teleport", "directional")


@dataclass(frozen=True)
class Tension:
    """A candidate INCOMPLETE relation: subject (the thing I can move) should
    stand in `operator` relation to object (the goal/marker), but currently
    falls short by `deficit` (>= 0; 0 = satisfied)."""
    operator: str
    subject_color: int
    object_color: Optional[int]
    deficit: float
    satisfied: bool
    subject_center: tuple
    object_center: Optional[tuple]

    def key(self) -> str:
        return f"{self.operator}:s{self.subject_color}:o{self.object_color}"

    def to_dict(self) -> dict:
        return {"operator": self.operator, "subject_color": self.subject_color,
                "object_color": self.object_color, "deficit": round(self.deficit, 2),
                "satisfied": self.satisfied, "key": self.key()}


@dataclass
class CompletionSchema:
    """A +1 abstracted to the OPERATION + the role kinds it acted on —
    coordinate/id-free, so it can transfer to the next level."""
    operator: str
    subject_role: RoleSignature
    object_role: Optional[RoleSignature]
    support: int = 1

    def to_dict(self) -> dict:
        return {"operator": self.operator,
                "subject_role": self.subject_role.to_dict(),
                "object_role": self.object_role.to_dict() if self.object_role else None,
                "support": self.support}


@dataclass
class BodyFrame:
    step: int
    objects: list
    affordances: dict
    tensions: list
    chosen: Optional[dict]
    proposed_action: Optional[dict]
    posterior: str
    converged: bool
    schemas: list
    transferred: Optional[dict]

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "step": self.step,
            "object_count": len(self.objects),
            "affordances": self.affordances,
            "tensions": self.tensions,
            "chosen": self.chosen,
            "proposed_action": self.proposed_action,
            "posterior": self.posterior,
            "converged": self.converged,
            "schemas": self.schemas,
            "transferred": self.transferred,
        }


# --------------------------------------------------------------------------- #
# The closed loop
# --------------------------------------------------------------------------- #
class BodySchema:
    """The convergent body-schema cycle. One `step()` per game step closes the
    loop: perceive -> update affordances from the last action's effect ->
    build tensions -> abstract any score event -> select the completion that
    my body can perform -> propose its action -> revise posterior -> repeat."""

    def __init__(
        self,
        *,
        falsify_window: int = 8,
        converge_window: int = 4,
        max_subject_colors: int = 6,
        max_target_colors: int = 3,
        satisfied_dist: float = 1.0,
    ) -> None:
        self.falsify_window = int(falsify_window)
        self.converge_window = int(converge_window)
        self.max_subject_colors = int(max_subject_colors)
        self.max_target_colors = int(max_target_colors)
        self.satisfied_dist = float(satisfied_dist)

        # Action-contingency agent identifier (2026-06-06): isolates the AGENT
        # (the colour whose per-action displacement DIRECTION tracks the action)
        # from board scroll / step counters / walkers — solving the
        # agent-identification frame problem the single-dominant detect_movement
        # could not. Used to PREFER the true agent in subject selection +
        # supply its isolated forward model.
        self._agent_ident = ComponentAgentIdentifier()
        self.affordances: dict[int, AffordanceProfile] = {}
        self.click_model: dict[int, ClickMover] = {}
        self.schemas: list[CompletionSchema] = []

        self._prev_grid: Optional[np.ndarray] = None
        self._prev_objects: list[dict] = []
        self._prev_chosen: Optional[Tension] = None
        self._prev_deficit_by_key: dict[str, float] = {}
        self._chosen_key: Optional[str] = None
        self._chosen_stable_steps = 0
        self._chosen_no_progress = 0
        self._step = 0
        self._probe_cursor = 0
        self._score_events = 0
        self._last_posterior = REJECTED
        # obstacle model (model-repair): colours that BLOCKED a predicted move.
        # Learned when the subject was expected to translate (it has a learned
        # displacement for the action) but did not — the cell it tried to enter
        # is a wall. This is the phase246 model-repair mechanism, scoped to the
        # body's own forward model.
        self.obstacle_colors: Counter = Counter()
        self._last_subject_color: Optional[int] = None
        self._last_subject_center: Optional[tuple] = None
        self._last_move_action: Optional[str] = None

    # -- public ------------------------------------------------------------- #
    def step(
        self,
        grid: Any,
        objects: Sequence[dict],
        action_name: Optional[str] = None,
        score_delta: int = 0,
        available_actions: Optional[Sequence[str]] = None,
        click_xy: Optional[tuple] = None,
    ) -> BodyFrame:
        """Advance one closed-loop turn. `grid` is the CURRENT frame (2D);
        `objects` are {component_id, center=(x,y), color, size} for it;
        `action_name` is the action that PRODUCED this frame from the previous
        one (None on the first call); `score_delta` its score change; `click_xy`
        is the (x,y) of an ACTION6 click (so the body learns its click->move
        affordance, the modality the strict-generic suite actually uses)."""
        self._step += 1
        arr = np.asarray(grid)
        objs = [o for o in objects if isinstance(o, dict)]

        # Layers 1-3: update affordances from the last action's observed effect.
        if action_name and self._prev_grid is not None:
            self._update_affordances(self._prev_grid, arr, action_name, objs)
            # model-repair: learn a wall if a predicted move was blocked.
            self._learn_obstacle(self._prev_grid, self._prev_objects, objs, action_name)
            # click affordance: learn how clicking at a position moves the agent
            # (teleport vs directional) — the click-driven modality.
            if _norm(action_name) == "ACTION6" and click_xy is not None:
                self._learn_click_model(self._prev_objects, objs, click_xy)

        # Layer 6 (first, so the just-scored relation is captured before re-typing).
        scored_schema = None
        if int(score_delta or 0) > 0 and self._prev_chosen is not None:
            scored_schema = self._abstract_score_event(self._prev_chosen, self._prev_objects)
            self._score_events += 1

        # Layer 4: build tensions (incomplete relations) on the current frame.
        tensions = self._build_tensions(arr, objs)

        # Layer 7: does a learned schema re-identify a relation here?
        transferred = self._transfer(objs, tensions)

        # Layer 5: select the completion my body can perform.
        chosen = self._select_completion(tensions, transferred)

        # Posterior + role revision (did the chosen relation make progress?).
        posterior = self._update_posterior(chosen, scored=int(score_delta or 0) > 0)

        # Layer 5b: propose the completion action (learned 1-step forward model).
        proposed = self._propose_action(chosen, objs, arr, available_actions)

        converged = (self._chosen_stable_steps >= self.converge_window
                     and chosen is not None and not chosen.satisfied) or \
                    (chosen is not None and chosen.satisfied)

        frame = BodyFrame(
            step=self._step,
            objects=objs,
            affordances={c: p.to_dict() for c, p in self.affordances.items()},
            tensions=[t.to_dict() for t in tensions],
            chosen=chosen.to_dict() if chosen else None,
            proposed_action=proposed,
            posterior=posterior,
            converged=bool(converged),
            schemas=[s.to_dict() for s in self.schemas],
            transferred=transferred.to_dict() if transferred else None,
        )

        # carry state
        self._prev_grid = arr
        self._prev_objects = objs
        self._prev_chosen = chosen
        self._prev_deficit_by_key = {t.key(): t.deficit for t in tensions}
        self._last_posterior = posterior
        return frame

    # -- layers 1-3: affordance -------------------------------------------- #
    def _update_affordances(self, prev: np.ndarray, cur: np.ndarray,
                            action: str, objs: list[dict]) -> None:
        # every present colour was "acted under" this step (controllability denom)
        for color in {int(o.get("color")) for o in objs if o.get("color") is not None}:
            self._profile(color).act_count += 1
        # NOTE on attribution (honest, learned 2026-06-06): a per-colour centroid
        # forward model was tried to fix detect_movement's single-dominant
        # mis-attribution on mazes, but it captures BOARD SCROLL (the environment
        # rigidly shifts several colours) rather than the agent's self-motion —
        # the agent-identification frame problem. So affordance attribution stays
        # the identity-free single-dominant change summary; it grounds the GOAL
        # relation correctly (cover/reach/meet) even though the precise per-action
        # agent forward model is not reliably isolated. See docs/body_schema_*.
        # Action-contingency: accumulate per-colour centroid displacements so
        # the agent identifier can isolate the agent (the colour whose
        # displacement varies by action) from scroll/counter/walker motion.
        self._agent_ident.observe_frames(action, prev, cur)
        info = detect_movement(prev, cur)
        if info is None or info.get("color") is None:
            return
        color = int(info["color"])
        self._profile(color).record(action, str(info["kind"]), info.get("displacement"))

    def agent_color(self) -> Optional[int]:
        """The action-contingent agent colour (or None if not yet isolated)."""
        return self._agent_ident.agent_color()

    def agent_forward_model(self) -> dict:
        """The isolated per-action forward model for the agent."""
        return self._agent_ident.forward_model()

    def _profile(self, color: int) -> AffordanceProfile:
        p = self.affordances.get(color)
        if p is None:
            p = AffordanceProfile(color=color)
            self.affordances[color] = p
        return p

    # -- layer 4: tension --------------------------------------------------- #
    def _subject_colors(self, objs: list[dict]) -> list[int]:
        present = [int(o["color"]) for o in objs if o.get("color") is not None]
        present_set = set(present)
        # movable colours first (body knows it can move them), else all present.
        movable = [c for c in present_set if self.affordances.get(c) and self.affordances[c].is_movable()]
        ordered = movable or sorted(present_set, key=lambda c: present.count(c))
        # PREFER the action-contingent agent (isolated from scroll/counter/walker)
        # as the primary subject — it is the body the agent actually controls.
        ag = self._agent_ident.agent_color()
        if ag is not None and ag in present_set:
            ordered = [ag] + [c for c in ordered if c != ag]
        return ordered[: self.max_subject_colors]

    def _target_colors(self, objs: list[dict]) -> list[int]:
        counts = Counter(int(o["color"]) for o in objs if o.get("color") is not None)
        # rarest distinct colours are candidate markers/goals.
        return [c for c, _ in sorted(counts.items(), key=lambda kv: kv[1])[: self.max_target_colors]]

    def _build_tensions(self, arr: np.ndarray, objs: list[dict]) -> list[Tension]:
        if not objs:
            return []
        centers_by_color: dict[int, list[tuple]] = {}
        for o in objs:
            c = o.get("color")
            if c is None:
                continue
            centers_by_color.setdefault(int(c), []).append(tuple(o.get("center", (0, 0))))
        subjects = self._subject_colors(objs)
        targets = self._target_colors(objs)
        tensions: list[Tension] = []
        for mc in subjects:
            s_centers = centers_by_color.get(mc, [])
            if not s_centers:
                continue
            # meet: same-colour partner (mirror pair / collide-to-merge)
            if len(s_centers) >= 2:
                d = min(_manh(a, b) for i, a in enumerate(s_centers)
                        for b in s_centers[i + 1:])
                tensions.append(Tension("meet", mc, mc, float(d), d <= self.satisfied_dist,
                                        s_centers[0], s_centers[1]))
            for tc in targets:
                if tc == mc:
                    continue
                t_centers = centers_by_color.get(tc, [])
                if not t_centers:
                    continue
                # reach: nearest subject->target centroid distance (dense gradient)
                best = min((_manh(a, b), a, b) for a in s_centers for b in t_centers)
                d, sa, ob = best
                tensions.append(Tension("reach", mc, tc, float(d), d <= self.satisfied_dist, sa, ob))
                # cover: target cells with a subject cell within 1 (sparse overlap)
                cov_deficit = _cover_deficit(arr, mc, tc)
                if cov_deficit is not None:
                    tensions.append(Tension("cover", mc, tc, float(cov_deficit),
                                            cov_deficit <= 0, sa, ob))
                # align: target cells sharing a row/col with a subject cell
                al_deficit = _align_deficit(arr, mc, tc)
                if al_deficit is not None:
                    tensions.append(Tension("align", mc, tc, float(al_deficit),
                                            al_deficit <= 0, sa, ob))
        tensions.sort(key=lambda t: (t.satisfied, t.deficit))
        return tensions

    # -- layer 5: completion selection ------------------------------------- #
    def _select_completion(self, tensions: list[Tension],
                           transferred: Optional[CompletionSchema]) -> Optional[Tension]:
        if not tensions:
            self._chosen_key = None
            return None
        unsat = [t for t in tensions if not t.satisfied]
        pool = unsat or tensions
        # (1) transfer: prefer the relation matching a learned scoring operation.
        if transferred is not None:
            for t in pool:
                if t.operator == transferred.operator and \
                        t.subject_color == transferred.subject_role.color:
                    return self._commit(t)
        # (2) grounded progress: the relation whose deficit DECREASED most last step
        # ("this is the relation my actions are completing") — relevance narrowing.
        best_prog = None
        best_drop = 0.0
        for t in pool:
            prev = self._prev_deficit_by_key.get(t.key())
            if prev is not None:
                drop = prev - t.deficit
                if drop > best_drop:
                    best_drop, best_prog = drop, t
        if best_prog is not None:
            return self._commit(best_prog)
        # (3) closest-to-completion (smallest deficit = "most wants to fall into place").
        return self._commit(pool[0])

    def _commit(self, t: Tension) -> Tension:
        key = t.key()
        if key == self._chosen_key:
            self._chosen_stable_steps += 1
        else:
            self._chosen_key = key
            self._chosen_stable_steps = 0
            self._chosen_no_progress = 0
        return t

    # -- posterior + role revision ----------------------------------------- #
    def _update_posterior(self, chosen: Optional[Tension], scored: bool) -> str:
        if chosen is None:
            return REJECTED
        if scored or chosen.satisfied:
            self._chosen_no_progress = 0
            return CONFIRMED
        prev = self._prev_deficit_by_key.get(chosen.key())
        if prev is not None and chosen.deficit < prev - 1e-9:
            self._chosen_no_progress = 0
            return CONFIRMED  # making progress toward completion
        # no progress
        self._chosen_no_progress += 1
        if self._chosen_no_progress >= self.falsify_window:
            # falsify: this relation is not the goal -> drop it so the next step
            # commits elsewhere (the role-revision the loop needs).
            self._prev_deficit_by_key.pop(chosen.key(), None)
            self._chosen_key = None
            self._chosen_stable_steps = 0
            self._chosen_no_progress = 0
            return FALSIFIED
        return PENDING

    # -- layer 5b: completion action (learned 1-step forward model) -------- #
    def _propose_action(self, chosen: Optional[Tension], objs: list[dict],
                        arr: np.ndarray, available: Optional[Sequence[str]]) -> Optional[dict]:
        if chosen is None:
            return None
        avail = set(available) if available else None
        prof = self.affordances.get(chosen.subject_color)
        sx, sy = chosen.subject_center
        ox, oy = chosen.object_center if chosen.object_center is not None else (sx, sy)

        # click-driven subject -> click toward the goal (the suite's modality).
        if chosen.operator in ("reach", "cover", "meet") and self._click_driven(chosen.subject_color):
            if avail is None or CLICK_ACTION in avail:
                site = self.click_completion_site(arr, objs)
                if site is not None:
                    return {"action": CLICK_ACTION, "x": int(site["x"]), "y": int(site["y"]),
                            "reason": site["reason"]}

        # recolor / match family -> click the object (paint / cycle toward it).
        if chosen.operator in ("cover", "align") and prof and prof.dominant_effect() == "recolor":
            if avail is None or CLICK_ACTION in avail:
                return {"action": CLICK_ACTION, "x": int(ox), "y": int(oy),
                        "reason": f"{chosen.operator}:recolor-click"}

        # movement family -> pick the move action whose LEARNED displacement most
        # reduces the centroid distance to the object (1-step forward model).
        if prof and prof.is_movable():
            best_action, best_dist = None, _manh((sx, sy), (ox, oy))
            for a in MOVE_ACTIONS:
                if avail is not None and a not in avail:
                    continue
                disp = prof.predicted_displacement(a)
                if disp is None:
                    continue
                nd = _manh((sx + disp[0], sy + disp[1]), (ox, oy))
                if nd < best_dist - 1e-9:
                    best_dist, best_action = nd, a
            if best_action is not None:
                return {"action": best_action, "reason": f"{chosen.operator}:forward-model",
                        "predicted_dist": round(best_dist, 2)}
            # movable but no useful learned displacement yet -> PROBE a move to learn.
            probe_pool = [a for a in MOVE_ACTIONS if avail is None or a in avail]
            if probe_pool:
                a = probe_pool[self._probe_cursor % len(probe_pool)]
                self._probe_cursor += 1
                return {"action": a, "reason": f"{chosen.operator}:probe-to-learn"}

        # unknown affordance -> click the subject to discover what it does.
        if avail is None or CLICK_ACTION in avail:
            return {"action": CLICK_ACTION, "x": int(sx), "y": int(sy),
                    "reason": f"{chosen.operator}:probe-subject"}
        return None

    # -- model-repair: learn walls from blocked moves ---------------------- #
    def _learn_obstacle(self, prev_grid: np.ndarray, prev_objs: list[dict],
                        cur_objs: list[dict], action: str) -> None:
        """If the body EXPECTED its subject to translate under `action` (it has a
        learned displacement) but the subject did NOT move, the cell it tried to
        enter is a wall — record its colour. This is the phase246 model-repair
        edge: prediction error -> structural hypothesis (the blocking colour)."""
        if action not in MOVE_ACTIONS:
            return
        # the subject = the movable colour with a learned displacement for this action
        subj_color = None
        disp = None
        for c, p in self.affordances.items():
            d = p.predicted_displacement(action)
            if d is not None and (abs(d[0]) + abs(d[1])) >= 1.0:
                subj_color, disp = c, d
                break
        if subj_color is None:
            return
        prev_c = _color_center(prev_objs, subj_color)
        cur_c = _color_center(cur_objs, subj_color)
        if prev_c is None or cur_c is None:
            return
        moved = abs(cur_c[0] - prev_c[0]) + abs(cur_c[1] - prev_c[1])
        if moved >= 1.0:
            return  # it moved -> not blocked
        # blocked: the obstacle is whatever sits at the PREDICTED destination
        # (centroid + full learned displacement), read from the pre-move frame.
        tx = int(round(prev_c[0] + disp[0]))
        ty = int(round(prev_c[1] + disp[1]))
        h, w = prev_grid.shape[:2]
        if 0 <= ty < h and 0 <= tx < w:
            wall = int(prev_grid[ty, tx])  # grid is [row=y, col=x]
            bg = _bg_color(prev_grid)
            if wall != subj_color and wall != bg:
                self.obstacle_colors[wall] += 1

    def _learn_click_model(self, prev_objs: list[dict], cur_objs: list[dict],
                           click_xy: tuple) -> None:
        """Learn the click->move affordance for any movable colour that moved
        this click — the click-driven modality the suite actually uses."""
        for color, prof in self.affordances.items():
            if not prof.is_movable():
                continue
            pc = _color_center(prev_objs, color)
            cc = _color_center(cur_objs, color)
            if pc is None or cc is None:
                continue
            if abs(cc[0] - pc[0]) + abs(cc[1] - pc[1]) < 0.5:
                continue  # this colour did not move on the click
            self.click_model.setdefault(color, ClickMover()).record(
                pc, (float(click_xy[0]), float(click_xy[1])), cc)

    def _click_driven(self, color: int) -> bool:
        m = self.click_model.get(color)
        return bool(m and m.active())

    def click_completion_site(self, grid: Any, objects: Sequence[dict]) -> Optional[dict]:
        """For a CLICK-driven subject, the ACTION6 site that drives the current
        chosen reach/cover/meet relation toward completion: the next cell along
        an obstacle-aware path to the goal (clicking adjacent cells walks the
        agent), else the goal cell itself. Returns {x,y,reason} or None."""
        arr = np.asarray(grid)
        objs = [o for o in objects if isinstance(o, dict)]
        tensions = self._build_tensions(arr, objs)
        chosen = self._select_completion(tensions, self._transfer(objs, tensions))
        if chosen is None or chosen.operator not in ("reach", "cover", "meet") or chosen.satisfied:
            return None
        if not self._click_driven(chosen.subject_color):
            return None
        if chosen.object_center is None:
            return None
        sx, sy = int(round(chosen.subject_center[0])), int(round(chosen.subject_center[1]))
        gx, gy = int(round(chosen.object_center[0])), int(round(chosen.object_center[1]))
        # teleport-click: aim straight at the goal. directional/path: the next
        # cell on an obstacle-aware unit-step path toward the goal.
        kind = self.click_model[chosen.subject_color].kind()
        if kind == "teleport":
            return {"x": gx, "y": gy, "reason": f"{chosen.operator}:click-teleport"}
        first = self._first_path_cell(arr, (sx, sy), (gx, gy))
        if first is not None:
            return {"x": int(first[0]), "y": int(first[1]),
                    "reason": f"{chosen.operator}:click-path"}
        return {"x": gx, "y": gy, "reason": f"{chosen.operator}:click-goal"}

    def _first_path_cell(self, arr: np.ndarray, start: tuple, goal: tuple,
                         *, max_nodes: int = 4000) -> Optional[tuple]:
        """First unit-step cell on an obstacle-aware BFS path start->goal."""
        h, w = arr.shape[:2]
        wall = np.zeros((h, w), dtype=bool)
        for c in self.obstacle_colors:
            wall |= (arr == c)
        sx, sy = start
        gx, gy = goal
        import collections
        q = collections.deque([(sx, sy, None)])
        seen = {(sx, sy)}
        nodes = 0
        while q and nodes < max_nodes:
            x, y, first = q.popleft()
            nodes += 1
            if abs(x - gx) + abs(y - gy) <= self.satisfied_dist:
                return first
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h) or wall[ny, nx] or (nx, ny) in seen:
                    continue
                seen.add((nx, ny))
                q.append((nx, ny, (nx, ny) if first is None else first))
        return None

    # -- generic planner: forward model + obstacle model + BFS ------------- #
    def plan_completion(self, grid: Any, objects: Sequence[dict],
                        available: Optional[Sequence[str]] = None,
                        *, max_nodes: int = 4000) -> Optional[dict]:
        """Best-first plan toward the current chosen reach/meet tension using the
        LEARNED forward model (per-action stride displacement) over a grid whose
        walls are the learned `obstacle_colors`. Returns the FIRST action of the
        plan (with a 'reason') or None when no model/plan exists. This is the
        generic version of a maze solver — no per-game code, driven entirely by
        what the body learned about its own movement + what blocked it."""
        arr = np.asarray(grid)
        objs = [o for o in objects if isinstance(o, dict)]
        tensions = self._build_tensions(arr, objs)
        chosen = self._select_completion(tensions, self._transfer(objs, tensions))
        if chosen is None or chosen.operator not in ("reach", "meet", "cover") or chosen.satisfied:
            return None
        prof = self.affordances.get(chosen.subject_color)
        if prof is None or not prof.is_movable():
            return None
        # learned stride moves available to the body
        avail = {_norm(a) for a in available} if available else None
        moves: dict[str, tuple] = {}
        for a in MOVE_ACTIONS:
            if avail is not None and a not in avail:
                continue
            d = prof.predicted_displacement(a)
            if d is not None and (abs(d[0]) + abs(d[1])) >= 1.0:
                moves[a] = (int(round(d[0])), int(round(d[1])))
        if not moves:
            return None
        h, w = arr.shape[:2]
        wall_mask = np.zeros((h, w), dtype=bool)
        for c in self.obstacle_colors:
            wall_mask |= (arr == c)
        start = chosen.subject_center
        goal = chosen.object_center if chosen.object_center is not None else start
        sx, sy = int(round(start[0])), int(round(start[1]))
        gx, gy = int(round(goal[0])), int(round(goal[1]))

        def blocked(x: int, y: int) -> bool:
            return not (0 <= x < w and 0 <= y < h) or bool(wall_mask[y, x])

        import heapq
        counter = 0
        h0 = abs(sx - gx) + abs(sy - gy)
        frontier = [(h0, 0, counter, sx, sy, None)]
        visited = {(sx, sy)}
        best_first = None
        while frontier and counter < max_nodes:
            _pri, depth, _c, x, y, first = heapq.heappop(frontier)
            if abs(x - gx) + abs(y - gy) <= self.satisfied_dist:
                return {"action": first or next(iter(moves)),
                        "reason": f"{chosen.operator}:planned",
                        "plan_depth": depth, "nodes": counter}
            if depth >= 60:
                continue
            for a, (dx, dy) in moves.items():
                nx, ny = x + dx, y + dy
                if blocked(nx, ny) or (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                counter += 1
                f0 = a if first is None else first
                heapq.heappush(frontier, (abs(nx - gx) + abs(ny - gy), depth + 1,
                                          counter, nx, ny, f0))
        return None

    # -- layer 6: score-event abstraction ---------------------------------- #
    def _abstract_score_event(self, chosen: Tension, prev_objs: list[dict]) -> Optional[CompletionSchema]:
        counts = _frame_color_counts(prev_objs)
        subj = _object_of_color(prev_objs, chosen.subject_color)
        obj = _object_of_color(prev_objs, chosen.object_color) if chosen.object_color is not None else None
        if subj is None:
            return None
        subject_role = RoleSignature.of(subj, counts)
        object_role = RoleSignature.of(obj, counts) if obj is not None else None
        # merge into an existing identical schema (operator + roles) -> support++
        for s in self.schemas:
            if s.operator == chosen.operator and s.subject_role == subject_role and \
                    s.object_role == object_role:
                s.support += 1
                return s
        schema = CompletionSchema(operator=chosen.operator, subject_role=subject_role,
                                  object_role=object_role)
        self.schemas.append(schema)
        return schema

    # -- layer 7: transfer -------------------------------------------------- #
    def _transfer(self, objs: list[dict], tensions: list[Tension]) -> Optional[CompletionSchema]:
        if not self.schemas or not objs:
            return None
        counts = _frame_color_counts(objs)
        # a schema transfers if a subject AND (object, if any) of the right role
        # kind are present on this frame.
        best = None
        best_support = -1
        for s in self.schemas:
            subj_ok = any(s.subject_role.match_score(o, counts) >= 0.5 for o in objs)
            obj_ok = s.object_role is None or any(
                s.object_role.match_score(o, counts) >= 0.5 for o in objs)
            if subj_ok and obj_ok and s.support > best_support:
                best, best_support = s, s.support
        return best

    def current_goal_site(self) -> Optional[tuple]:
        """The (x,y) of the currently grounded reach/cover/meet relation's GOAL
        (object centre) — the click destination a lost click-driven loop should
        aim at. None when the chosen relation is not a goal-directed one."""
        ch = self._prev_chosen
        if ch is None or ch.operator not in ("reach", "cover", "meet"):
            return None
        if ch.object_center is None:
            return None
        return (int(round(ch.object_center[0])), int(round(ch.object_center[1])))

    # -- introspection ------------------------------------------------------ #
    def summary(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "steps": self._step,
            "score_events": self._score_events,
            "affordances": {c: p.to_dict() for c, p in self.affordances.items()},
            "movable_colors": sorted(c for c, p in self.affordances.items() if p.is_movable()),
            "click_driven_colors": sorted(c for c in self.click_model if self._click_driven(c)),
            "obstacle_colors": sorted(self.obstacle_colors),
            "schema_count": len(self.schemas),
            "schemas": [s.to_dict() for s in self.schemas],
            "chosen_key": self._chosen_key,
            "posterior": self._last_posterior,
        }


# --------------------------------------------------------------------------- #
# pure-numpy relation helpers
# --------------------------------------------------------------------------- #
def _manh(a: tuple, b: tuple) -> float:
    return abs(float(a[0]) - float(b[0])) + abs(float(a[1]) - float(b[1]))


def _norm(action: str) -> str:
    return str(action or "").strip().upper()




def _object_of_color(objs: list[dict], color: Optional[int]) -> Optional[dict]:
    if color is None:
        return None
    best = None
    for o in objs:
        if o.get("color") == color:
            if best is None or int(o.get("size", 0)) > int(best.get("size", 0)):
                best = o
    return best


def _color_center(objs: list[dict], color: Optional[int]) -> Optional[tuple]:
    o = _object_of_color(objs, color)
    if o is None:
        return None
    cx, cy = o.get("center", (0, 0))
    return (float(cx), float(cy))


def _cover_deficit(arr: np.ndarray, mc: int, tc: int) -> Optional[float]:
    """Number of target(tc) cells NOT within 1 cell of a subject(mc) cell.
    0 = every target cell is covered/adjacent. None if no target cells."""
    tgt = (arr == tc)
    n = int(tgt.sum())
    if n == 0:
        return None
    sub = (arr == mc)
    if not sub.any():
        return float(n)
    # 4-neighbour dilation of the subject mask (pure numpy, no scipy).
    dil = sub.copy()
    dil[1:, :] |= sub[:-1, :]
    dil[:-1, :] |= sub[1:, :]
    dil[:, 1:] |= sub[:, :-1]
    dil[:, :-1] |= sub[:, 1:]
    covered = int((tgt & dil).sum())
    return float(n - covered)


def _align_deficit(arr: np.ndarray, mc: int, tc: int) -> Optional[float]:
    """Number of target(tc) cells that DON'T share a row or column with any
    subject(mc) cell. 0 = every target is line-of-sight aligned. None if no
    targets."""
    tys, txs = np.where(arr == tc)
    if txs.size == 0:
        return None
    sys_, sxs = np.where(arr == mc)
    if sxs.size == 0:
        return float(txs.size)
    rows = set(int(y) for y in sys_)
    cols = set(int(x) for x in sxs)
    uncovered = sum(1 for x, y in zip(txs.tolist(), tys.tolist())
                    if int(y) not in rows and int(x) not in cols)
    return float(uncovered)


__all__ = [
    "BodySchema", "BodyFrame", "AffordanceProfile", "Tension", "CompletionSchema",
    "detect_movement", "SCHEMA_VERSION",
]
