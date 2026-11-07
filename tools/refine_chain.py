"""Follow each gate stop of the modeler draws forward (docs/refine_chain_registration_2026-11-05.md).

    python3 tools/refine_chain.py <m1/out_canon> <m2/out_canon> <report.json>

A stop is a host-run row (`event: modeler`, `host.run.outcome == "mismatch"`). After it, on the same (game, level),
until the level ends: rewrite (an assistant tool call defining `def step(`), re-certified (a later host row with
`exact` and a found plan), host-completed (a later host run ending `level_completed`), level completed (by anyone).
"""
from __future__ import annotations

import glob
import json
import os
import sys

GAMES = ("ar25", "bp35", "cn04", "dc22", "ls20", "re86", "s5i5", "tn36")


def level_bounds(game_run):
    """[(level, first action, end action)] with 1-based levels; end is None for the level still open."""
    out, start = [], 0
    done = int(game_run["levels_completed"])
    for i, n in enumerate(game_run["actions_per_level"]):
        level = i + 1
        if level <= done:
            out.append((level, start, start + n))
            start += n
        else:
            out.append((level, start, None))
            break
    return out


def calls(rf):
    """Assistant tool-call code with the action counter of the request that carried it, deduplicated by call id."""
    seen, out = set(), []
    for raw in open(rf, encoding="utf-8"):
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        if d.get("event") != "request":
            continue
        at = d.get("action")
        for m in d.get("messages") or []:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                cid = tc.get("id")
                if cid in seen:
                    continue
                seen.add(cid)
                args = (tc.get("function") or {}).get("arguments") or ""
                try:
                    code = json.loads(args).get("code", args)
                except (ValueError, AttributeError):
                    code = args
                out.append((at, code))
    return out


def host_rows(rf):
    rows = []
    for raw in open(rf, encoding="utf-8"):
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        if d.get("event") == "modeler" and d.get("host"):
            rows.append(d["host"])
    return rows


def draw(out):
    bench = {g["game_id"][:4]: g for g in json.load(open(os.path.join(out, "benchmark.json")))["game_runs"]}
    stops, planned_levels = [], {}
    for rf in sorted(glob.glob(os.path.join(out, "*_p0_requests.jsonl"))):
        g4 = os.path.basename(rf)[:4]
        if g4 not in GAMES:
            continue
        rows, code = host_rows(rf), calls(rf)
        bounds = {lv: (a, b) for lv, a, b in level_bounds(bench[g4])}
        for i, h in enumerate(rows):
            run = h.get("run") or {}
            if not run:
                continue
            level = int(h.get("level", -1))
            planned_levels.setdefault((g4, level), []).append(run.get("outcome"))
            if run.get("outcome") != "mismatch":
                continue
            at = int(h.get("at_step") or 0) + int(run.get("executed") or 0)
            first, end = bounds.get(level, (None, None))
            later = [r for r in rows[i + 1:] if int(r.get("level", -1)) == level]
            rewrites = [c for a, c in code if a is not None and a >= at and (end is None or a <= end) and "def step(" in c]
            recert = [r for r in later if r.get("exact") and (r.get("search") or {}).get("found")]
            host_done = [r for r in later if (r.get("run") or {}).get("outcome") == "level_completed"]
            again = [r for r in later if (r.get("run") or {}).get("outcome") == "mismatch"]
            stops.append({
                "game": g4, "level": level, "stop_at": at, "level_actions": [first, end],
                "action": run.get("action"), "differing_cells": run.get("differing_cells"),
                "executed_before_stop": run.get("executed"),
                "rewrites": len(rewrites), "later_host_rows": len(later), "recertified": len(recert),
                "stopped_again": len(again), "host_completed": len(host_done),
                "level_completed": end is not None,
                "actions_after_stop_to_completion": (end - at) if end is not None else None,
            })
    return {"stops": stops, "planned_levels": {"%s L%d" % k: v for k, v in sorted(planned_levels.items())},
            "levels": {g: int(v["levels_completed"]) for g, v in sorted(bench.items())}}


def main():
    m1, m2, report = sys.argv[1:4]
    res = {"m1": draw(m1), "m2": draw(m2)}
    pooled = res["m1"]["stops"] + res["m2"]["stops"]
    stop_levels = {(d, s["game"], s["level"]) for d in ("m1", "m2") for s in res[d]["stops"]}
    clean_levels = {(d, k) for d in ("m1", "m2") for k, v in res[d]["planned_levels"].items() if "mismatch" not in v}

    def completed(d, game, level):
        return res[d]["levels"][game] >= level
    res["pooled"] = {
        "stops": len(pooled),
        "with_rewrite": sum(1 for s in pooled if s["rewrites"]),
        "recertified": sum(1 for s in pooled if s["recertified"]),
        "host_completed_after_stop": sum(1 for s in pooled if s["host_completed"]),
        "levels_with_stop": len(stop_levels),
        "levels_with_stop_completed": sum(1 for d, g, lv in stop_levels if completed(d, g, lv)),
        "levels_planned_without_stop": len(clean_levels),
        "levels_planned_without_stop_completed": sum(
            1 for d, k in clean_levels if completed(d, k.split()[0], int(k.split()[1][1:]))),
    }
    json.dump(res, open(report, "w"), indent=1)
    for d in ("m1", "m2"):
        for s in res[d]["stops"]:
            print(d, s)
    print("pooled", res["pooled"])


if __name__ == "__main__":
    main()
