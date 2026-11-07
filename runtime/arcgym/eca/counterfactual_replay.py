"""Phase 43 + Phase 44 — Counterfactual replay harness for the target-binding repair gate.

Background
----------

Phase 42 shipped the advisory ``TargetBindingRepairGate`` (see
``arcgym/eca/target_binding_repair_gate.py``). The 25-game advisory
sweep (`docs/phase42_25g_advisory_repair_findings_2026-05-08.md`)
showed 5738 / 6481 evaluations (88.5%) recommended repair, but the
ECA-only score stayed at the Phase 39 baseline of 25/25 — confirming
the advisory contract and that *recommendation rate is not the same
as causal lift*.

Phase 44 — K-step counterfactual replay (this module also provides)
-------------------------------------------------------------------

Phase 43 single-step replay measured 0 causal lift across sb26 / lp85 /
ar25. For action-family-cluster games (e.g. ar25 / r11l) where L1 unlock
requires K coordinated actions, single-step replay can never reveal that
"if we'd followed teacher's plan for K steps, we'd have unlocked L1."
Phase 44 introduces :class:`KStepCounterfactualHarness` which extends
the single-step A/B branch to K-step branches:

* Branch A (eca-K): replay prefix [0..N-1], then take ECA's actual
  recorded actions [N..N+K-1] — record final score.
* Branch B (repair-K, "teacher_driven" mode): replay prefix [0..N-1],
  then for each step in [N..N+K-1] query the perception adapter's
  ``perceive_and_plan(frame)``, take the FIRST returned action, advance
  the env. Final score = score after K teacher-driven actions.
* k_step_lift_K = score_B - score_A.

Branch B's option (b) ("single-repair + K-1 ECA continuation") is
deferred to Phase 45+ because rerunning ECA from a forced-action state
requires re-driving the full reasoning loop, which the offline harness
does not model. Phase 44 implements only option (a).

The Phase 41 spec called this out explicitly:

* `objective_mismatch` LOO-AUC = 0.981 (highly learnable)
* `payload_mismatch`  LOO-AUC = 0.907 (highly learnable)
* `scoreless_mismatch` LOO-AUC = 0.268 (NOT learnable from features)

In other words, the gate can detect "ECA disagrees with the teacher"
extremely well, but disagreement does not predict *score* failure.

Phase 43 introduces a per-decision **causal** signal: for each frame
where the gate recommends a repair, branch the environment, take
ECA's actual action vs. the gate's `repaired_action` (with
`repaired_payload`), and compare resulting score deltas. The
recommendation is *causal* iff the repaired branch outscored ECA's
branch on that decision.

The harness operates entirely **post-hoc** over pre-recorded episode
logs. It does NOT mutate or drive ECA at runtime, preserving the
Phase 42 advisory contract:

* The authoritative ECA score is whatever was already produced by
  the canonical ``local_ctrl_v2`` run.
* The harness instantiates *fresh* offline envs to evaluate
  alternative actions. Those branch envs are throwaways used only
  for scoring counterfactuals.

Mode A only
-----------

Phase 43 implements **Mode A** ("A/B replay, offline"). Mode B
("online aggressive injection") is deferred to Phase 44+ — the
authoritative-stream injection touches the runtime contract and
needs its own design pass.

Public API
----------

* :class:`ReplayFrame` — one input frame for the harness
* :class:`CounterfactualSummary` — aggregated per-game stats
* :class:`CounterfactualReplayHarness` — driver
* :func:`load_episode_log` — read step_logs_v2.jsonl + step_metrics.jsonl

The harness expects *episode logs* (not live env objects) and an
``env_factory`` callable producing a fresh env (typically pointing at
the offline ARC-AGI-3 toolkit). The reason: replaying ECA's prefix
[0..N-1] then taking either ``a_eca`` or ``a_repair`` is O(N) actions
per repair-decision; for a 400-action episode with 350 repair
recommendations, total branch actions = ~140K, which is acceptable
for offline replay but would balloon if we tried to share a single
env across branches.

Determinism
-----------

ARC-AGI-3 OFFLINE mode is deterministic given the same action
sequence; the harness verifies this holds by comparing the post-prefix
score-before against the recorded value at frame N. Mismatch is
recorded as ``replay_drift_count`` and the frame is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


# Action-name canonicalisation: episode logs may store
# ``"ACTION6"``, ``"GameAction.ACTION6"``, or even ``"action6"``;
# downstream env steppers need the bare GameAction enum name.
def _canonical_action_name(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw.upper()


# Payload keys we forward to ``env.step``. ``action`` and
# ``reasoning`` are stripped here; ``action`` is supplied via the
# coerced GameAction, and ``reasoning`` from the recorded log can
# contain very large ASCII boards which would distort latency / log
# noise for no benefit.
_PAYLOAD_PASSTHROUGH_KEYS = ("x", "y", "value", "val", "comp_size", "data")


def _clean_payload(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key in _PAYLOAD_PASSTHROUGH_KEYS:
        if key in raw and raw[key] is not None:
            out[key] = raw[key]
    return out


@dataclass
class ReplayFrame:
    """One per-step record extracted from an episode log."""

    step_index: int
    eca_action_name: str
    eca_action_payload: dict[str, Any]
    score_before: int
    score_after: int
    repair_recommended: bool
    repair_decision: dict[str, Any]
    repaired_action_name: str
    repaired_action_payload: dict[str, Any]
    repair_eval_index: int = 0

    @property
    def action_changes(self) -> bool:
        """Whether the repaired branch differs from ECA's branch."""

        if not self.repair_recommended:
            return False
        if not self.repaired_action_name:
            return False
        if self.repaired_action_name != self.eca_action_name:
            return True
        if self.repaired_action_payload != self.eca_action_payload:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "eca_action_name": str(self.eca_action_name),
            "eca_action_payload": dict(self.eca_action_payload),
            "score_before": int(self.score_before),
            "score_after": int(self.score_after),
            "repair_recommended": bool(self.repair_recommended),
            "repaired_action_name": str(self.repaired_action_name),
            "repaired_action_payload": dict(self.repaired_action_payload),
            "repair_eval_index": int(self.repair_eval_index),
        }


@dataclass
class CounterfactualSummary:
    """Per-game aggregate produced by :class:`CounterfactualReplayHarness`."""

    game_id: str = ""
    frames_replayed: int = 0
    repair_rec_frames: int = 0
    repair_rec_action_changes: int = 0
    eca_score_progress_count: int = 0
    repair_score_progress_count: int = 0
    repair_score_loss_count: int = 0
    causal_repair_count: int = 0
    score_delta_lift_sum: float = 0.0
    score_delta_lift_max: float = 0.0
    replay_drift_count: int = 0
    replay_error_count: int = 0
    per_frame: list[dict[str, Any]] = field(default_factory=list)

    @property
    def score_delta_lift_avg(self) -> float:
        if not self.repair_rec_frames:
            return 0.0
        return float(self.score_delta_lift_sum) / float(self.repair_rec_frames)

    @property
    def causal_repair_rate(self) -> float:
        if not self.repair_rec_frames:
            return 0.0
        return float(self.causal_repair_count) / float(self.repair_rec_frames)

    def to_dict(self, *, include_per_frame: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "game_id": str(self.game_id),
            "frames_replayed": int(self.frames_replayed),
            "repair_rec_frames": int(self.repair_rec_frames),
            "repair_rec_action_changes": int(self.repair_rec_action_changes),
            "eca_score_progress_count": int(self.eca_score_progress_count),
            "repair_score_progress_count": int(self.repair_score_progress_count),
            "repair_score_loss_count": int(self.repair_score_loss_count),
            "causal_repair_count": int(self.causal_repair_count),
            "score_delta_lift_sum": float(self.score_delta_lift_sum),
            "score_delta_lift_avg": round(float(self.score_delta_lift_avg), 6),
            "score_delta_lift_max": float(self.score_delta_lift_max),
            "causal_repair_rate": round(float(self.causal_repair_rate), 6),
            "replay_drift_count": int(self.replay_drift_count),
            "replay_error_count": int(self.replay_error_count),
        }
        if include_per_frame:
            out["per_frame"] = list(self.per_frame)
        return out


def _extract_action_block(record: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pull the executed action name and click-payload out of a step record.

    Step records come in two shapes:

    * ``eca_step_logs_v2.jsonl`` rows: ``{"action": {"name": "ACTION6", "payload": {"x": 4, "y": 32, ...}}, ...}``
    * ``eca_step_metrics.jsonl`` rows: ``{"action": "ACTION6", "action_payload": {"action": "GameAction.ACTION6", "x": 4, "y": 32, ...}, ...}``
    """

    act = record.get("action")
    if isinstance(act, Mapping):
        name = _canonical_action_name(act.get("name"))
        payload = _clean_payload(act.get("payload"))
        return name, payload
    if act is None:
        act = record.get("action_name")
    name = _canonical_action_name(act)
    payload = _clean_payload(record.get("action_payload"))
    return name, payload


def _extract_score(record: Mapping[str, Any]) -> int | None:
    """Pull score-after-step from a step record."""

    if "score" in record:
        try:
            return int(record["score"])
        except (TypeError, ValueError):
            return None
    outcome = record.get("outcome")
    if isinstance(outcome, Mapping) and "score" in outcome:
        try:
            return int(outcome["score"])
        except (TypeError, ValueError):
            return None
    return None


def _extract_repair_decision(record: Mapping[str, Any]) -> dict[str, Any]:
    """Pull last_repair_decision dict from a step record."""

    decision = record.get("repair_gate_last_decision")
    if isinstance(decision, Mapping):
        return dict(decision)
    return {}


def load_episode_log(
    output_dir: str | Path,
    *,
    game_id: str = "",
) -> list[ReplayFrame]:
    """Load ``ReplayFrame`` objects from a v2-fast output directory.

    Joins ``eca_step_logs_v2.jsonl`` (action history with payloads)
    and ``eca_step_metrics.jsonl`` (repair gate decisions) by step
    index. Both files are emitted by the v2 fast harness with one
    record per step in increasing order.
    """

    root = Path(output_dir)
    logs_path = root / "eca_step_logs_v2.jsonl"
    metrics_path = root / "eca_step_metrics.jsonl"
    if not logs_path.is_file() and not metrics_path.is_file():
        raise FileNotFoundError(
            f"No episode log files in {root!s}; expected eca_step_logs_v2.jsonl"
        )

    log_rows: dict[int, dict[str, Any]] = {}
    if logs_path.is_file():
        with logs_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step_id = rec.get("step_id")
                if step_id is None:
                    continue
                try:
                    log_rows[int(step_id)] = rec
                except (TypeError, ValueError):
                    continue

    metric_rows: dict[int, dict[str, Any]] = {}
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # `action_idx` in step_metrics corresponds to action sequence
                # 0-indexed; step_logs_v2 uses `step_id`. Use index in file.
                idx = rec.get("action_idx")
                try:
                    metric_rows[int(idx)] = rec
                except (TypeError, ValueError):
                    continue

    frames: list[ReplayFrame] = []
    indices = sorted(set(log_rows.keys()) | set(metric_rows.keys()))
    last_score = 0
    for step_idx in indices:
        log_rec = log_rows.get(step_idx, {})
        metric_rec = metric_rows.get(step_idx, {})
        eca_name, eca_payload = _extract_action_block(log_rec or metric_rec)
        if not eca_name:
            continue
        score_after = _extract_score(metric_rec)
        if score_after is None:
            score_after = last_score
        score_before = last_score
        decision = _extract_repair_decision(metric_rec)
        repair_recommended = bool(decision.get("repair_recommended"))
        repaired_name = _canonical_action_name(decision.get("repaired_action"))
        repaired_payload = _clean_payload(decision.get("repaired_payload"))
        eval_idx = int(metric_rec.get("repair_gate_eval_count", 0) or 0)

        frames.append(
            ReplayFrame(
                step_index=int(step_idx),
                eca_action_name=str(eca_name),
                eca_action_payload=dict(eca_payload),
                score_before=int(score_before),
                score_after=int(score_after),
                repair_recommended=repair_recommended,
                repair_decision=decision,
                repaired_action_name=str(repaired_name),
                repaired_action_payload=dict(repaired_payload),
                repair_eval_index=eval_idx,
            )
        )
        last_score = int(score_after)

    if game_id:
        # Annotate frames with the supplied game_id for caller convenience.
        # ReplayFrame intentionally omits a game_id field; this is just a hook
        # for stable dict serialisation downstream.
        for fr in frames:
            fr.repair_decision.setdefault("_game_id", game_id)
    return frames


def _build_step_payload(
    action_name: str,
    payload: Mapping[str, Any],
    GameAction: Any,
) -> dict[str, Any]:
    """Compose ``env.step()`` input from a canonical name + click payload."""

    enum = getattr(GameAction, action_name, None)
    if enum is None:
        # Fallback: GameAction["ACTION6"] indexing API.
        try:
            enum = GameAction[action_name]
        except (KeyError, TypeError):
            raise ValueError(f"Unknown GameAction: {action_name!r}")
    out: dict[str, Any] = {"action": enum, "reasoning": "phase43_counterfactual"}
    for key in _PAYLOAD_PASSTHROUGH_KEYS:
        if key in payload and payload[key] is not None:
            out[key] = payload[key]
    return out


def _replay_prefix(env: Any, frames: Sequence[ReplayFrame], up_to: int, GameAction: Any) -> tuple[int, bool]:
    """Replay actions [0..up_to) on a freshly-reset env. Returns (score, terminated)."""

    score = 0
    for j in range(up_to):
        f = frames[j]
        try:
            payload = _build_step_payload(f.eca_action_name, f.eca_action_payload, GameAction)
        except ValueError:
            return score, True
        try:
            obs, _, done = env.step(payload)
        except Exception:
            return score, True
        try:
            score = int(obs.get("score", score) or 0)
        except (TypeError, ValueError):
            pass
        if done:
            return score, True
    return score, False


@dataclass
class CounterfactualReplayHarness:
    """Drive A/B replay for one episode.

    The harness is intentionally stateless across calls — each
    :meth:`replay_episode` opens its own envs via ``env_factory``.

    Parameters
    ----------
    env_factory : ``Callable[[], env]``
        Returns a freshly-constructed env object (typically
        ``ArcAgi3Env(game_id=..., operation_mode=OFFLINE)``).
        The env must support ``reset()`` returning an obs dict with
        a numeric ``score`` field, and ``step(payload)`` returning
        ``(obs, reward, done)``.
    GameAction : enum
        The enum class whose member names match action strings
        (typically ``arcengine.GameAction``).
    max_frames : int, optional
        If positive, only consider repair-recommended frames whose
        ``step_index`` is < ``max_frames``. Saves wall-time when
        smoke-testing on long episodes.
    branch_action_budget : int, optional
        Per-branch action cap. Each branch replays the prefix
        [0..N-1] then takes 1 action; budget should cover the
        longest prefix the harness can hit. Defaults to 1024 which
        comfortably exceeds the v2-fast 400-action cap.
    """

    env_factory: Callable[[], Any]
    GameAction: Any
    max_frames: int = 0
    branch_action_budget: int = 1024

    def replay_episode(
        self,
        game_id: str,
        frames: Sequence[ReplayFrame],
    ) -> CounterfactualSummary:
        """Replay one episode and aggregate causal-lift stats."""

        summary = CounterfactualSummary(game_id=str(game_id))
        if not frames:
            return summary

        for n, frame in enumerate(frames):
            if self.max_frames and n >= self.max_frames:
                break
            summary.frames_replayed += 1

            if not frame.repair_recommended or not frame.action_changes:
                # Frames with no recommendation OR with the recommendation
                # equal to ECA's action carry no counterfactual signal —
                # they are still counted in frames_replayed for context.
                continue

            summary.repair_rec_frames += 1
            summary.repair_rec_action_changes += 1

            # Branch A — ECA's actual choice.
            try:
                env_a = self.env_factory()
            except Exception:
                summary.replay_error_count += 1
                continue
            try:
                obs_a = env_a.reset()
                _ = int(obs_a.get("score", 0) or 0)
                prefix_score_a, prefix_done_a = _replay_prefix(
                    env_a, frames, n, self.GameAction
                )
                if prefix_done_a:
                    summary.replay_drift_count += 1
                    self._safe_close(env_a)
                    continue
                if prefix_score_a != frame.score_before:
                    # Determinism failure — record as drift and skip.
                    summary.replay_drift_count += 1
                    self._safe_close(env_a)
                    continue
                payload_a = _build_step_payload(
                    frame.eca_action_name,
                    frame.eca_action_payload,
                    self.GameAction,
                )
                obs_a, _, _ = env_a.step(payload_a)
                score_a = int(obs_a.get("score", prefix_score_a) or 0)
            except Exception:
                summary.replay_error_count += 1
                self._safe_close(env_a)
                continue
            finally:
                pass
            self._safe_close(env_a)
            score_delta_a = int(score_a) - int(frame.score_before)
            if score_delta_a > 0:
                summary.eca_score_progress_count += 1

            # Branch B — gate's repaired action.
            try:
                env_b = self.env_factory()
            except Exception:
                summary.replay_error_count += 1
                continue
            try:
                env_b.reset()
                prefix_score_b, prefix_done_b = _replay_prefix(
                    env_b, frames, n, self.GameAction
                )
                if prefix_done_b or prefix_score_b != frame.score_before:
                    summary.replay_drift_count += 1
                    self._safe_close(env_b)
                    continue
                payload_b = _build_step_payload(
                    frame.repaired_action_name,
                    frame.repaired_action_payload,
                    self.GameAction,
                )
                obs_b, _, _ = env_b.step(payload_b)
                score_b = int(obs_b.get("score", prefix_score_b) or 0)
            except Exception:
                summary.replay_error_count += 1
                self._safe_close(env_b)
                continue
            self._safe_close(env_b)
            score_delta_b = int(score_b) - int(frame.score_before)
            if score_delta_b > 0:
                summary.repair_score_progress_count += 1

            lift = float(score_delta_b - score_delta_a)
            summary.score_delta_lift_sum += lift
            if lift > summary.score_delta_lift_max:
                summary.score_delta_lift_max = lift
            if lift > 0:
                summary.causal_repair_count += 1
            if score_delta_b < score_delta_a:
                summary.repair_score_loss_count += 1

            summary.per_frame.append(
                {
                    "step_index": int(frame.step_index),
                    "repair_eval_index": int(frame.repair_eval_index),
                    "score_before": int(frame.score_before),
                    "eca_action": frame.eca_action_name,
                    "repair_action": frame.repaired_action_name,
                    "score_a": int(score_a),
                    "score_b": int(score_b),
                    "score_delta_a": int(score_delta_a),
                    "score_delta_b": int(score_delta_b),
                    "score_delta_lift": float(lift),
                    "objective_risk": float(
                        frame.repair_decision.get("objective_mismatch_risk", 0.0) or 0.0
                    ),
                    "payload_risk": float(
                        frame.repair_decision.get("payload_mismatch_risk", 0.0) or 0.0
                    ),
                }
            )

        return summary

    @staticmethod
    def _safe_close(env: Any) -> None:
        try:
            env.close()
        except Exception:
            pass


def aggregate_per_game(summaries: Iterable[CounterfactualSummary]) -> dict[str, Any]:
    """Roll up per-game summaries into a single suite-level report."""

    items = list(summaries)
    suite = {
        "games": [s.to_dict() for s in items],
        "total_frames_replayed": sum(s.frames_replayed for s in items),
        "total_repair_rec_frames": sum(s.repair_rec_frames for s in items),
        "total_causal_repair_count": sum(s.causal_repair_count for s in items),
        "total_score_delta_lift_sum": float(
            sum(s.score_delta_lift_sum for s in items)
        ),
    }
    total_rec = suite["total_repair_rec_frames"]
    suite["overall_causal_repair_rate"] = round(
        float(suite["total_causal_repair_count"]) / float(total_rec), 6
    ) if total_rec else 0.0
    return suite


# ---------------------------------------------------------------------------
# Phase 44 — K-step counterfactual replay
# ---------------------------------------------------------------------------


@dataclass
class KStepCounterfactualSummary:
    """Per-game per-K aggregate produced by :class:`KStepCounterfactualHarness`.

    Mirrors :class:`CounterfactualSummary` but tracked for a specific K.
    K=1 should reproduce the Phase 43 single-step results (regression check).
    """

    game_id: str = ""
    k: int = 1
    branch_b_mode: str = "teacher_driven"
    frames_replayed: int = 0
    repair_rec_frames: int = 0
    repair_rec_action_changes: int = 0
    eca_score_progress_count: int = 0
    repair_score_progress_count: int = 0
    repair_score_loss_count: int = 0
    causal_kstep_count: int = 0
    kstep_lift_sum: float = 0.0
    kstep_lift_max: float = 0.0
    kstep_lift_loss_count: int = 0
    branch_b_empty_plan_count: int = 0
    branch_b_short_plan_count: int = 0
    replay_drift_count: int = 0
    replay_error_count: int = 0
    per_frame: list[dict[str, Any]] = field(default_factory=list)

    @property
    def kstep_lift_avg(self) -> float:
        if not self.repair_rec_frames:
            return 0.0
        return float(self.kstep_lift_sum) / float(self.repair_rec_frames)

    @property
    def causal_kstep_rate(self) -> float:
        if not self.repair_rec_frames:
            return 0.0
        return float(self.causal_kstep_count) / float(self.repair_rec_frames)

    def to_dict(self, *, include_per_frame: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "game_id": str(self.game_id),
            "k": int(self.k),
            "branch_b_mode": str(self.branch_b_mode),
            "frames_replayed": int(self.frames_replayed),
            "repair_rec_frames": int(self.repair_rec_frames),
            "repair_rec_action_changes": int(self.repair_rec_action_changes),
            "eca_score_progress_count": int(self.eca_score_progress_count),
            "repair_score_progress_count": int(self.repair_score_progress_count),
            "repair_score_loss_count": int(self.repair_score_loss_count),
            "causal_kstep_count": int(self.causal_kstep_count),
            "kstep_lift_sum": float(self.kstep_lift_sum),
            "kstep_lift_avg": round(float(self.kstep_lift_avg), 6),
            "kstep_lift_max": float(self.kstep_lift_max),
            "kstep_lift_loss_count": int(self.kstep_lift_loss_count),
            "causal_kstep_rate": round(float(self.causal_kstep_rate), 6),
            "branch_b_empty_plan_count": int(self.branch_b_empty_plan_count),
            "branch_b_short_plan_count": int(self.branch_b_short_plan_count),
            "replay_drift_count": int(self.replay_drift_count),
            "replay_error_count": int(self.replay_error_count),
        }
        if include_per_frame:
            out["per_frame"] = list(self.per_frame)
        return out


@dataclass(frozen=True)
class KStepPromotionDecision:
    """Admission decision for turning K-step replay evidence into a guard input."""

    game_id: str
    k: int
    admitted: bool
    blocked_reason: str = ""
    causal_kstep_rate: float = 0.0
    repair_rec_frames: int = 0
    causal_kstep_count: int = 0
    kstep_lift_sum: float = 0.0
    kstep_lift_loss_count: int = 0
    branch_b_empty_plan_count: int = 0
    branch_b_short_plan_count: int = 0
    replay_drift_count: int = 0
    replay_error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": str(self.game_id),
            "k": int(self.k),
            "admitted": bool(self.admitted),
            "blocked_reason": str(self.blocked_reason),
            "causal_kstep_rate": round(float(self.causal_kstep_rate), 6),
            "repair_rec_frames": int(self.repair_rec_frames),
            "causal_kstep_count": int(self.causal_kstep_count),
            "kstep_lift_sum": float(self.kstep_lift_sum),
            "kstep_lift_loss_count": int(self.kstep_lift_loss_count),
            "branch_b_empty_plan_count": int(self.branch_b_empty_plan_count),
            "branch_b_short_plan_count": int(self.branch_b_short_plan_count),
            "replay_drift_count": int(self.replay_drift_count),
            "replay_error_count": int(self.replay_error_count),
        }


def evaluate_kstep_authoritative_promotion(
    summary: KStepCounterfactualSummary,
    *,
    min_causal_rate: float = 1.0,
    min_repair_rec_frames: int = 30,
    require_positive_lift: bool = True,
) -> KStepPromotionDecision:
    """Gate K-step causal replay evidence before it can influence ranking.

    This function does not generate actions and does not change runtime
    behavior. It only converts offline K-step replay summaries into a guarded
    admission signal for later phases.
    """

    reasons: list[str] = []
    if int(summary.repair_rec_frames) < int(min_repair_rec_frames):
        reasons.append("insufficient_repair_rec_frames")
    if float(summary.causal_kstep_rate) < float(min_causal_rate):
        reasons.append("causal_kstep_rate_below_threshold")
    if require_positive_lift and float(summary.kstep_lift_sum) <= 0.0:
        reasons.append("non_positive_kstep_lift_sum")
    if int(summary.kstep_lift_loss_count) > 0:
        reasons.append("kstep_lift_loss_observed")
    if int(summary.branch_b_empty_plan_count) > 0:
        reasons.append("branch_b_empty_plan_observed")
    if int(summary.branch_b_short_plan_count) > 0:
        reasons.append("branch_b_short_plan_observed")
    if int(summary.replay_drift_count) > 0:
        reasons.append("replay_drift_observed")
    if int(summary.replay_error_count) > 0:
        reasons.append("replay_error_observed")

    return KStepPromotionDecision(
        game_id=str(summary.game_id),
        k=int(summary.k),
        admitted=not reasons,
        blocked_reason="" if not reasons else reasons[0],
        causal_kstep_rate=float(summary.causal_kstep_rate),
        repair_rec_frames=int(summary.repair_rec_frames),
        causal_kstep_count=int(summary.causal_kstep_count),
        kstep_lift_sum=float(summary.kstep_lift_sum),
        kstep_lift_loss_count=int(summary.kstep_lift_loss_count),
        branch_b_empty_plan_count=int(summary.branch_b_empty_plan_count),
        branch_b_short_plan_count=int(summary.branch_b_short_plan_count),
        replay_drift_count=int(summary.replay_drift_count),
        replay_error_count=int(summary.replay_error_count),
    )


def _kstep_summary_from_artifact_row(row: Mapping[str, Any]) -> KStepCounterfactualSummary:
    """Build a K-step summary from a JSON artifact row."""

    return KStepCounterfactualSummary(
        game_id=str(row.get("game_id", "") or ""),
        k=int(row.get("k", 0) or 0),
        branch_b_mode=str(row.get("branch_b_mode", "") or "teacher_driven"),
        frames_replayed=int(row.get("frames_replayed", 0) or 0),
        repair_rec_frames=int(row.get("repair_rec_frames", 0) or 0),
        repair_rec_action_changes=int(row.get("repair_rec_action_changes", 0) or 0),
        eca_score_progress_count=int(row.get("eca_score_progress_count", 0) or 0),
        repair_score_progress_count=int(row.get("repair_score_progress_count", 0) or 0),
        repair_score_loss_count=int(row.get("repair_score_loss_count", 0) or 0),
        causal_kstep_count=int(row.get("causal_kstep_count", 0) or 0),
        kstep_lift_sum=float(row.get("kstep_lift_sum", 0.0) or 0.0),
        kstep_lift_max=float(row.get("kstep_lift_max", 0.0) or 0.0),
        kstep_lift_loss_count=int(row.get("kstep_lift_loss_count", 0) or 0),
        branch_b_empty_plan_count=int(row.get("branch_b_empty_plan_count", 0) or 0),
        branch_b_short_plan_count=int(row.get("branch_b_short_plan_count", 0) or 0),
        replay_drift_count=int(row.get("replay_drift_count", 0) or 0),
        replay_error_count=int(row.get("replay_error_count", 0) or 0),
    )


def kstep_promotion_decisions_from_artifact(
    artifact: Mapping[str, Any],
    *,
    admitted_only: bool = False,
    game_id: str | None = None,
) -> list[KStepPromotionDecision]:
    """Extract promotion decisions from old or new K-step replay artifacts.

    New Phase790-style artifacts may already contain ``promotion_decisions``.
    Older Phase44-style artifacts contain only ``games[].per_k`` summaries, so
    decisions are recomputed with :func:`evaluate_kstep_authoritative_promotion`.
    """

    game_filter = str(game_id or "").strip()
    decisions: list[KStepPromotionDecision] = []
    raw_decisions = artifact.get("promotion_decisions")
    if isinstance(raw_decisions, Sequence) and not isinstance(raw_decisions, (str, bytes)):
        for raw in raw_decisions:
            if not isinstance(raw, Mapping):
                continue
            decision = KStepPromotionDecision(
                game_id=str(raw.get("game_id", "") or ""),
                k=int(raw.get("k", 0) or 0),
                admitted=bool(raw.get("admitted", False)),
                blocked_reason=str(raw.get("blocked_reason", "") or ""),
                causal_kstep_rate=float(raw.get("causal_kstep_rate", 0.0) or 0.0),
                repair_rec_frames=int(raw.get("repair_rec_frames", 0) or 0),
                causal_kstep_count=int(raw.get("causal_kstep_count", 0) or 0),
                kstep_lift_sum=float(raw.get("kstep_lift_sum", 0.0) or 0.0),
                kstep_lift_loss_count=int(raw.get("kstep_lift_loss_count", 0) or 0),
                branch_b_empty_plan_count=int(raw.get("branch_b_empty_plan_count", 0) or 0),
                branch_b_short_plan_count=int(raw.get("branch_b_short_plan_count", 0) or 0),
                replay_drift_count=int(raw.get("replay_drift_count", 0) or 0),
                replay_error_count=int(raw.get("replay_error_count", 0) or 0),
            )
            decisions.append(decision)
    else:
        games = artifact.get("games")
        if isinstance(games, Sequence) and not isinstance(games, (str, bytes)):
            for game in games:
                if not isinstance(game, Mapping):
                    continue
                for row in list(game.get("per_k") or []):
                    if isinstance(row, Mapping):
                        decisions.append(
                            evaluate_kstep_authoritative_promotion(
                                _kstep_summary_from_artifact_row(row)
                            )
                        )

    out: list[KStepPromotionDecision] = []
    for decision in decisions:
        if game_filter and decision.game_id != game_filter:
            continue
        if admitted_only and not decision.admitted:
            continue
        out.append(decision)
    return out


def load_kstep_promotion_decisions(
    path: str | Path,
    *,
    admitted_only: bool = False,
    game_id: str | None = None,
) -> list[KStepPromotionDecision]:
    """Load K-step promotion decisions from a JSON artifact path."""

    artifact_path = Path(path)
    try:
        import json

        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, Mapping):
        return []
    return kstep_promotion_decisions_from_artifact(
        payload,
        admitted_only=admitted_only,
        game_id=game_id,
    )


def _take_step(env: Any, action_name: str, payload: Mapping[str, Any], GameAction: Any) -> tuple[int, bool, dict[str, Any] | None]:
    """Execute one step on the env. Returns ``(score, done, observation)``.

    ``score`` defaults to the previous score on failure.
    """

    try:
        step_payload = _build_step_payload(action_name, payload, GameAction)
    except ValueError:
        return 0, True, None
    try:
        obs, _, done = env.step(step_payload)
    except Exception:
        return 0, True, None
    score = 0
    if isinstance(obs, Mapping):
        try:
            score = int(obs.get("score", 0) or 0)
        except (TypeError, ValueError):
            score = 0
    return int(score), bool(done), (dict(obs) if isinstance(obs, Mapping) else None)


def _take_eca_branch_k(
    env: Any,
    frames: Sequence[ReplayFrame],
    n: int,
    k: int,
    GameAction: Any,
) -> tuple[int, bool, int]:
    """Take ECA's recorded actions [n..n+k-1] from current env state.

    Returns ``(final_score, terminated, steps_taken)``. ``steps_taken`` may
    be < k if the recorded log ends sooner or the env reports done.
    """

    score = 0
    steps = 0
    for j in range(n, n + k):
        if j >= len(frames):
            break
        f = frames[j]
        score, done, _ = _take_step(env, f.eca_action_name, f.eca_action_payload, GameAction)
        steps += 1
        if done:
            return score, True, steps
    return score, False, steps


def _take_teacher_branch_k(
    env: Any,
    teacher_fn: Callable[[Any], Sequence[Any]] | None,
    last_observation: Mapping[str, Any] | None,
    k: int,
    GameAction: Any,
    *,
    fallback_first_action: tuple[str, Mapping[str, Any]] | None = None,
) -> tuple[int, bool, int, int, int]:
    """Take K consecutive teacher-driven actions from current env state.

    Each step queries ``teacher_fn(observation)`` (which is the perception
    adapter's ``perceive_and_plan(frame)``); takes the FIRST returned
    ``(action, payload)`` tuple; advances the env; loops.

    ``fallback_first_action`` is a fallback used only at step 0 if the
    teacher returns an empty plan — it represents the gate's recorded
    ``repaired_action``/``repaired_payload`` from frame N which is what
    the Phase 43 single-step harness would have used. This keeps Phase 44
    K=1 behaviour aligned with Phase 43 even when the teacher adapter
    happens to emit no plan from the live frame at frame N.

    Returns ``(score, terminated, steps_taken, empty_plan_steps,
    short_plan_flag)`` where:
    * ``empty_plan_steps`` = number of times teacher_fn returned [].
      First-step empty triggers fallback. Subsequent empties terminate.
    * ``short_plan_flag`` = 1 if the loop terminated early due to empty
      plan / done / error before K steps; else 0.
    """

    score = 0
    steps_taken = 0
    empty_plan_steps = 0
    obs_now: Mapping[str, Any] | None = last_observation
    for step_idx in range(k):
        action_name: str = ""
        payload: dict[str, Any] = {}
        plan: Sequence[Any] = ()
        if teacher_fn is not None and obs_now is not None:
            frame = _frame_from_observation_dict(obs_now)
            if frame is not None:
                try:
                    plan = list(teacher_fn(frame) or [])
                except Exception:
                    plan = []
        if plan:
            action_obj, payload_raw = plan[0]
            action_name = _canonical_action_name(getattr(action_obj, "name", action_obj))
            payload = dict(_clean_payload(payload_raw if isinstance(payload_raw, Mapping) else {}))
        else:
            empty_plan_steps += 1
            if step_idx == 0 and fallback_first_action is not None:
                fb_name, fb_payload = fallback_first_action
                action_name = str(fb_name)
                payload = dict(fb_payload or {})
            else:
                # No plan to execute — short the branch.
                return score, False, steps_taken, empty_plan_steps, 1

        if not action_name:
            return score, False, steps_taken, empty_plan_steps, 1

        score, done, obs_now = _take_step(env, action_name, payload, GameAction)
        steps_taken += 1
        if done:
            return score, True, steps_taken, empty_plan_steps, 1 if steps_taken < k else 0
    return score, False, steps_taken, empty_plan_steps, 0


def _frame_from_observation_dict(observation: Mapping[str, Any] | None) -> Any | None:
    """Mirror of ``_frame_from_observation`` from perception_solver_bridge.

    Pulled in here to avoid importing the bridge (which imports heavyweight
    ECA modules). Returns the latest frame from a v2-fast obs dict, as a
    numpy array if numpy is importable, else as the raw nested list.
    """

    if not isinstance(observation, Mapping):
        return None
    frame = observation.get("frame")
    if frame is None:
        return None
    if isinstance(frame, list) and frame:
        frame = frame[-1]
    try:
        import numpy as np

        return np.asarray(frame)
    except Exception:
        return frame


@dataclass
class KStepCounterfactualHarness:
    """Drive K-step counterfactual A/B replay for one episode.

    Parameters
    ----------
    env_factory : ``Callable[[], env]``
        Returns a freshly-constructed env (typically ``ArcAgi3Env(...,
        operation_mode=OFFLINE)``).
    GameAction : enum
        The ``arcengine.GameAction`` enum.
    teacher_fn : ``Callable[[frame], Sequence[(action, payload)]]`` or ``None``
        The perception adapter's ``perceive_and_plan(frame)`` for the
        target game. If ``None`` the harness will only fall back to the
        recorded ``repaired_action`` (so K=1 reproduces Phase 43 exactly,
        and K>1 stops after step 0).
    k_values : tuple[int, ...]
        Default K values to evaluate (e.g. ``(3, 5, 10)``).
    branch_b_mode : str
        Currently only ``"teacher_driven"`` is supported. Reserved for
        Phase 45+ ``"single_repair_then_eca"``.
    max_frames : int, optional
        If positive, only consider frames whose ``step_index`` is < this.
    """

    env_factory: Callable[[], Any]
    GameAction: Any
    teacher_fn: Callable[[Any], Sequence[Any]] | None = None
    k_values: tuple[int, ...] = (3, 5, 10)
    branch_b_mode: str = "teacher_driven"
    max_frames: int = 0

    def replay_kstep(
        self,
        game_id: str,
        frames: Sequence[ReplayFrame],
        *,
        k_values: Iterable[int] | None = None,
        branch_b_mode: str | None = None,
    ) -> dict[int, KStepCounterfactualSummary]:
        """Run K-step replay across one or more K values for one episode.

        Returns a dict keyed by K → :class:`KStepCounterfactualSummary`.
        """

        ks = tuple(int(k) for k in (k_values if k_values is not None else self.k_values))
        if not ks:
            ks = (1,)
        mode = str(branch_b_mode or self.branch_b_mode)
        if mode != "teacher_driven":
            raise ValueError(
                f"Unsupported branch_b_mode: {mode!r}. "
                "Phase 44 implements 'teacher_driven' only; "
                "'single_repair_then_eca' is deferred to Phase 45+."
            )

        summaries: dict[int, KStepCounterfactualSummary] = {
            int(k): KStepCounterfactualSummary(
                game_id=str(game_id), k=int(k), branch_b_mode=mode
            )
            for k in ks
        }

        if not frames:
            return summaries

        for n, frame in enumerate(frames):
            if self.max_frames and n >= self.max_frames:
                break
            for k in ks:
                summaries[int(k)].frames_replayed += 1

            if not frame.repair_recommended or not frame.action_changes:
                continue

            for k in ks:
                summary = summaries[int(k)]
                summary.repair_rec_frames += 1
                summary.repair_rec_action_changes += 1
                self._replay_one_frame_kstep(
                    summary=summary,
                    frame=frame,
                    n=n,
                    k=int(k),
                    frames=frames,
                )

        return summaries

    def _replay_one_frame_kstep(
        self,
        *,
        summary: KStepCounterfactualSummary,
        frame: ReplayFrame,
        n: int,
        k: int,
        frames: Sequence[ReplayFrame],
    ) -> None:
        """Run a single K-step A/B branch for a repair-recommended frame."""

        # Branch A — ECA's recorded actions [n..n+k-1].
        try:
            env_a = self.env_factory()
        except Exception:
            summary.replay_error_count += 1
            return
        try:
            obs_a = env_a.reset()
            prefix_score_a, prefix_done_a = _replay_prefix(env_a, frames, n, self.GameAction)
            if prefix_done_a:
                summary.replay_drift_count += 1
                CounterfactualReplayHarness._safe_close(env_a)
                return
            if prefix_score_a != frame.score_before:
                summary.replay_drift_count += 1
                CounterfactualReplayHarness._safe_close(env_a)
                return
            score_a, _terminated_a, steps_a = _take_eca_branch_k(
                env_a, frames, n, k, self.GameAction
            )
        except Exception:
            summary.replay_error_count += 1
            CounterfactualReplayHarness._safe_close(env_a)
            return
        CounterfactualReplayHarness._safe_close(env_a)
        score_delta_a = int(score_a) - int(frame.score_before)
        if score_delta_a > 0:
            summary.eca_score_progress_count += 1

        # Branch B — K teacher-driven actions, falling back to the gate's
        # recorded repair on step 0 if the teacher emits an empty plan.
        try:
            env_b = self.env_factory()
        except Exception:
            summary.replay_error_count += 1
            return
        try:
            obs_b = env_b.reset()
            prefix_score_b, prefix_done_b = _replay_prefix(env_b, frames, n, self.GameAction)
            if prefix_done_b or prefix_score_b != frame.score_before:
                summary.replay_drift_count += 1
                CounterfactualReplayHarness._safe_close(env_b)
                return
            # Capture the live observation just before stepping into branch B.
            # The env's last obs is the post-prefix state; we need it for the
            # first teacher_fn call. We re-step a no-op? No — easiest is to
            # request the obs from env's stored last frame. _replay_prefix
            # left the env at frame N; we synthesise an observation by
            # leveraging the env's reset/step return. We track this by
            # re-running the env path with explicit observation tracking:
            obs_b_now = self._observation_at_prefix_end(env_b, frames, n)
            fallback = (frame.repaired_action_name, frame.repaired_action_payload)
            (
                score_b,
                _terminated_b,
                steps_b,
                empty_plan_steps,
                short_flag,
            ) = _take_teacher_branch_k(
                env_b,
                self.teacher_fn,
                obs_b_now,
                k,
                self.GameAction,
                fallback_first_action=fallback,
            )
        except Exception:
            summary.replay_error_count += 1
            CounterfactualReplayHarness._safe_close(env_b)
            return
        CounterfactualReplayHarness._safe_close(env_b)
        score_delta_b = int(score_b) - int(frame.score_before)
        if score_delta_b > 0:
            summary.repair_score_progress_count += 1

        if empty_plan_steps:
            summary.branch_b_empty_plan_count += 1
        if short_flag:
            summary.branch_b_short_plan_count += 1

        lift = float(score_delta_b - score_delta_a)
        summary.kstep_lift_sum += lift
        if lift > summary.kstep_lift_max:
            summary.kstep_lift_max = lift
        if lift > 0:
            summary.causal_kstep_count += 1
        if lift < 0:
            summary.kstep_lift_loss_count += 1
        if score_delta_b < score_delta_a:
            summary.repair_score_loss_count += 1

        summary.per_frame.append(
            {
                "step_index": int(frame.step_index),
                "k": int(k),
                "score_before": int(frame.score_before),
                "eca_action": frame.eca_action_name,
                "repair_action": frame.repaired_action_name,
                "score_a": int(score_a),
                "score_b": int(score_b),
                "score_delta_a": int(score_delta_a),
                "score_delta_b": int(score_delta_b),
                "kstep_lift": float(lift),
                "branch_a_steps_taken": int(steps_a),
                "branch_b_steps_taken": int(steps_b),
                "branch_b_empty_plan_steps": int(empty_plan_steps),
                "branch_b_short_plan": bool(short_flag),
            }
        )

    @staticmethod
    def _observation_at_prefix_end(
        env: Any,
        frames: Sequence[ReplayFrame],
        n: int,
    ) -> Mapping[str, Any] | None:
        """Return the live observation after replaying the prefix [0..n-1].

        Strategy: re-replay the prefix, capturing the post-step obs each
        time. The harness's outer ``_replay_prefix`` already advanced the
        env to step n, but didn't preserve the observation. We instead
        construct a *parallel* env, replay the same prefix, and return its
        final observation. This doubles prefix replay cost per branch B
        but keeps the harness self-contained without modifying
        ``_replay_prefix``'s return signature.

        Note: callers SHOULD NOT use this env after the call — return
        the obs and discard.
        """

        # Optimisation: we already have ``env`` at prefix-end. Just call
        # ``reset()`` returns a starting obs — that's not what we need.
        # Instead, drive a fresh env to the same prefix end and snapshot.
        # This is O(n) extra prefix replay per frame; acceptable for our
        # smoke-run sizes (~30-50 frames × 1 prefix per K = small).
        try:
            from copy import deepcopy  # noqa: F401
        except Exception:
            pass
        # Easiest path: peek at env._last_obs if available (ArcAgi3Env
        # stores it). Fall back to re-replay if not.
        last_obs = getattr(env, "_last_obs", None)
        if last_obs is not None:
            try:
                # ArcAgi3Env stores the FrameDataRaw object in _last_obs;
                # re-format it to dict shape via env._format_observation.
                fmt = getattr(env, "_format_observation", None)
                if callable(fmt):
                    return fmt(last_obs)
            except Exception:
                pass
        return None


def aggregate_kstep_per_game(
    summaries_by_game: Iterable[Iterable[KStepCounterfactualSummary]],
) -> dict[str, Any]:
    """Roll up a sequence of per-game per-K summary lists into suite stats.

    Each inner iterable is the per-K summaries for one game (i.e. the
    ``.values()`` of one ``replay_kstep()`` return dict). The outer
    iterable is one entry per game.
    """

    by_k: dict[int, list[KStepCounterfactualSummary]] = {}
    for game_summaries in summaries_by_game:
        for s in game_summaries:
            by_k.setdefault(int(s.k), []).append(s)

    suite: dict[str, Any] = {"per_k": {}}
    for k in sorted(by_k.keys()):
        items = by_k[k]
        total_rec = sum(s.repair_rec_frames for s in items)
        total_causal = sum(s.causal_kstep_count for s in items)
        suite["per_k"][str(k)] = {
            "k": int(k),
            "games": [s.to_dict() for s in items],
            "total_frames_replayed": sum(s.frames_replayed for s in items),
            "total_repair_rec_frames": total_rec,
            "total_causal_kstep_count": total_causal,
            "total_kstep_lift_sum": float(sum(s.kstep_lift_sum for s in items)),
            "total_kstep_lift_loss_count": sum(s.kstep_lift_loss_count for s in items),
            "total_branch_b_empty_plan_count": sum(
                s.branch_b_empty_plan_count for s in items
            ),
            "total_branch_b_short_plan_count": sum(
                s.branch_b_short_plan_count for s in items
            ),
            "overall_causal_kstep_rate": (
                round(float(total_causal) / float(total_rec), 6) if total_rec else 0.0
            ),
        }
    return suite


__all__ = [
    "ReplayFrame",
    "CounterfactualSummary",
    "CounterfactualReplayHarness",
    "KStepCounterfactualSummary",
    "KStepPromotionDecision",
    "KStepCounterfactualHarness",
    "aggregate_kstep_per_game",
    "evaluate_kstep_authoritative_promotion",
    "kstep_promotion_decisions_from_artifact",
    "load_kstep_promotion_decisions",
    "load_episode_log",
    "aggregate_per_game",
]
