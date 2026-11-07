"""Judge for the modeler arm (docs/modeler_registration_2026-10-31.md).

    python tools/judge_modeler_colab.py <m1/out_canon> <m2/out_canon> <ctl1/out_canon> <ctl2/out_canon> <report.txt>

Reads benchmark.json and the *_requests.jsonl of each run. The harness writes one `event: modeler` row per host
check (verify / search / run) and the model-check rows of every snippet; the openers carry the status line and the
system prompt the regime addendum. A level counts when the openers on it span >= 20 actions.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys

GAMES = ("ar25", "bp35", "cn04", "dc22", "ls20", "re86", "s5i5", "tn36")
CONTROL_LEVELS, CONTROL_SCORE = 69, 685.0
LEVEL_FLOOR, SCORE_FLOOR = CONTROL_LEVELS - 5, 0.85 * CONTROL_SCORE
MIN_LEVEL_ACTIONS = 20
WRITTEN_RATE, EXACT_RATE, PLAN_RATE, TAKEN_LEVELS = 0.5, 0.3, 0.3, 3
SYS_MARK = "Modeler regime:"
STATUS_MARK = "Model status (harness):"
STATE_RE = re.compile(r"Current state: step (\d+), level (\d+)")


def bench(out):
    b = json.load(open(os.path.join(out, "benchmark.json")))
    games, levels, score = set(), 0, 0.0
    for g in b["game_runs"]:
        games.add(g["game_id"][:4])
        levels += int(g["levels_completed"])
        score += float(g.get("final_score") or g.get("score") or 0)
    return {"games": games, "levels": levels, "score": score}


def text_of(c):
    return c if isinstance(c, str) else " ".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))


def scan(out):
    """Per run: level spans, host rows per level, flags, tool-result counts."""
    r = {"sys": 0, "status": 0, "requests": 0, "host_rows": [], "refusals": 0, "mismatch_stops": 0,
         "span": collections.defaultdict(lambda: [10 ** 9, -1]), "step_levels": set(), "exact_levels": set(),
         "goal_levels": set(), "plan_levels": set(), "host_levels_done": 0, "host_actions": 0, "host_runs": 0,
         "search_seconds": 0.0, "searches": 0, "probes_max": collections.defaultdict(int), "relief": 0,
         "mc_verify": 0, "mc_plan": 0, "mc_run": 0}
    for rf in glob.glob(os.path.join(out, "*_requests.jsonl")):
        g4 = os.path.basename(rf)[:4]
        if g4 not in GAMES:
            continue
        seen_tools = set()
        for raw in open(rf, encoding="utf-8"):
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            ev = d.get("event")
            if ev == "modeler":
                h = d.get("host") or {}
                key = (g4, int(h.get("level", -1)))
                r["host_rows"].append((key, h))
                if h.get("step"):
                    r["step_levels"].add(key)
                if h.get("exact"):
                    r["exact_levels"].add(key)
                if h.get("goal"):
                    r["goal_levels"].add(key)
                s = h.get("search") or {}
                if s:
                    r["searches"] += 1
                    r["search_seconds"] += float(s.get("seconds") or 0)
                    if s.get("found"):
                        r["plan_levels"].add(key)
                run = h.get("run") or {}
                if run:
                    r["host_runs"] += 1
                    r["host_actions"] += int(run.get("executed") or 0)
                    if run.get("outcome") == "level_completed":
                        r["host_levels_done"] += 1
                r["probes_max"][key] = max(r["probes_max"][key], int(d.get("probes_used") or 0))
                r["relief"] += bool(d.get("relief"))
                if h.get("at_step") is not None:
                    sp = r["span"][key]
                    sp[0], sp[1] = min(sp[0], int(h["at_step"])), max(sp[1], int(h["at_step"]))
                continue
            if ev == "model_check":
                for e in d.get("entries") or []:
                    fn = e.get("fn")
                    if fn == "verify_model":
                        r["mc_verify"] += 1
                    elif fn == "find_plan":
                        r["mc_plan"] += 1
                    elif fn == "run_plan":
                        r["mc_run"] += 1
                continue
            if ev != "request":
                continue
            r["requests"] += 1
            msgs = d.get("messages") or []
            if any(m.get("role") == "system" and SYS_MARK in text_of(m.get("content")) for m in msgs):
                r["sys"] += 1
            if d.get("request_index_within_turn") == 1 and msgs and msgs[-1].get("role") == "user":
                t = text_of(msgs[-1].get("content"))
                if STATUS_MARK in t:
                    r["status"] += 1
                m = STATE_RE.search(t)
                if m:
                    key = (g4, int(m.group(2)))
                    sp = r["span"][key]
                    sp[0], sp[1] = min(sp[0], int(m.group(1))), max(sp[1], int(m.group(1)))
            for m in msgs:
                if m.get("role") == "tool":
                    t = text_of(m.get("content"))
                    if t in seen_tools:
                        continue
                    seen_tools.add(t)
                    r["refusals"] += t.count('"probe_budget"')
                    r["mismatch_stops"] += t.count('"model_mismatch"')
    r["counted"] = {k for k, (a, b) in r["span"].items() if b - a >= MIN_LEVEL_ACTIONS}
    return r


def main():
    m1, m2, c1, c2, report = sys.argv[1:6]
    runs = {"m1": m1, "m2": m2, "ctl1": c1, "ctl2": c2}
    B = {k: bench(p) for k, p in runs.items()}
    S = {k: scan(p) for k, p in runs.items()}
    out = ["modeler arm verdict (docs/modeler_registration_2026-10-31.md)"]
    out.append("%-5s %5s %7s | %5s %6s %7s %6s %6s %6s %5s %5s %6s %7s %6s %6s | %6s %6s" % (
        "run", "lev", "score", "sys", "status", "hostrow", "cnt", "step", "exact", "goal", "plan", "hruns",
        "hacts", "hlvls", "refus", "mism", "mcver"))
    for k in runs:
        s = S[k]
        out.append("%-5s %5d %7.1f | %5d %6d %7d %6d %6d %6d %5d %5d %6d %7d %6d %6d | %6d %6d" % (
            k, B[k]["levels"], B[k]["score"], s["sys"], s["status"], len(s["host_rows"]), len(s["counted"]),
            len(s["step_levels"] & s["counted"]), len(s["exact_levels"] & s["counted"]),
            len(s["goal_levels"] & s["counted"]), len(s["plan_levels"]), s["host_runs"], s["host_actions"],
            s["host_levels_done"], s["refusals"], s["mismatch_stops"], s["mc_verify"]))
    arm, ctl = ("m1", "m2"), ("ctl1", "ctl2")
    counted = sum(len(S[k]["counted"]) for k in arm)
    step_l = sum(len(S[k]["step_levels"] & S[k]["counted"]) for k in arm)
    step_any = sum(len(S[k]["step_levels"]) for k in arm)
    exact_l = sum(len(S[k]["exact_levels"] & S[k]["step_levels"]) for k in arm)
    goal_exact = sum(len(S[k]["exact_levels"] & S[k]["goal_levels"]) for k in arm)
    plan_l = sum(len(S[k]["plan_levels"] & S[k]["exact_levels"] & S[k]["goal_levels"]) for k in arm)
    host_done = sum(S[k]["host_levels_done"] for k in arm)
    levels = sum(B[k]["levels"] for k in arm)
    score = sum(B[k]["score"] for k in arm)
    out.append("arm pooled: counted levels %d, step retained on %d (%.0f %%; %d levels with step at all), exact on %d of %d "
               "(%.0f %%), plans on %d of %d exact+goal levels, host-completed levels %d, host actions %d, searches %d "
               "(%.1f s mean), refusals %d, mismatch stops %d, relief rows %d" % (
                   counted, step_l, 100 * step_l / max(counted, 1), step_any, exact_l, step_any,
                   100 * exact_l / max(step_any, 1), plan_l, goal_exact, host_done,
                   sum(S[k]["host_actions"] for k in arm), sum(S[k]["searches"] for k in arm),
                   sum(S[k]["search_seconds"] for k in arm) / max(sum(S[k]["searches"] for k in arm), 1),
                   sum(S[k]["refusals"] for k in arm), sum(S[k]["mismatch_stops"] for k in arm),
                   sum(S[k]["relief"] for k in arm)))
    out.append("levels at 60 min: arm %d vs control %d; final_score sum: arm %.1f vs control %.1f" % (
        levels, sum(B[k]["levels"] for k in ctl), score, sum(B[k]["score"] for k in ctl)))
    verdict = []
    if any(B[k]["games"] != set(GAMES) for k in runs):
        verdict.append("WRONG_GAMES")
    elif any(S[k]["sys"] or S[k]["status"] or S[k]["host_rows"] for k in ctl):
        verdict.append("CONTROL_LEAK")
    elif any(S[k]["sys"] == 0 or S[k]["status"] == 0 for k in arm):
        verdict.append("NOT_ON")
    elif step_l < WRITTEN_RATE * counted:
        verdict.append("MODEL_NOT_WRITTEN")
    elif exact_l < EXACT_RATE * step_any:
        verdict.append("MODEL_NOT_EXACT")
    elif plan_l < PLAN_RATE * goal_exact:
        verdict.append("PLAN_NOT_FOUND")
    elif host_done >= TAKEN_LEVELS:
        verdict.append("EXECUTOR_TAKEN")
    else:
        verdict.append("EXECUTOR_NOT_TAKEN")
    if levels <= LEVEL_FLOOR or score < SCORE_FLOOR:
        verdict.append("HARMFUL")
    elif levels >= CONTROL_LEVELS and score >= CONTROL_SCORE:
        verdict.append("PROMISING")
    else:
        verdict.append("NULL")
    out.append("VERDICT " + " ".join(verdict))
    text = "\n".join(out)
    print(text)
    open(report, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
