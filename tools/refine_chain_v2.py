"""Refine-chain v2: tie each gate stop to the model call that caused it and to the calls made after it.

    python3 -I tools/refine_chain_v2.py <m1/out_canon> <m2/out_canon> <report.json>
    python3 -I tools/refine_chain_v2.py --selftest

Why v2 (docs/refine_chain_registration_2026-11-05.md, addendum). v1 dated a tool call by the action counter of the
first request whose history carried it. The call that defined the model and triggered the stopped run first appears
in the request right after the stop, so v1 could count the stopping definition itself as a rewrite after the stop.

v2 works on the order of rows in each `*_p0_requests.jsonl`:
  response k  ->  tool execution rows (`model_check`, `modeler` host rows)  ->  request k+1
A `response` row does not carry the generated message; the call generated in response k first appears in the
history of request k+1. So a call is placed at the response row just before its first appearance, and every host row
between that response and that request was produced by that call (`trigger`).

For each stop (host row with run.outcome == "mismatch", produced by call S) on (game, level):
  - later calls: calls placed after S's request, generated while the level was open (response `action` <= level end);
  - step redefined: a later call whose code defines `def step(` at top level;
  - model version: every host row carries the version of `step` in effect when it ran (a new version whenever a
    call defines a `step` whose source - ast.get_source_segment, whitespace-normalised - differs from the one in
    effect). Helper functions are not compared, so versions are a lower bound on model changes;
  - new version after the stop: a later call put a version in effect other than the one that stopped;
  - re-certified: a later host row on the level, under a new version, with the host's `exact` flag (near-exact:
    <= 10 % of judged transitions mismatched, docs/modeler_registration_2026-10-31.md) and search.found;
    strict: the same with verify.mismatched == 0 and errors == 0;
  - host-completed: a later host run under a new version on the level with run.outcome == "level_completed".
"""
from __future__ import annotations

import ast
import glob
import json
import os
import re
import sys

GAMES = ("ar25", "bp35", "cn04", "dc22", "ls20", "re86", "s5i5", "tn36")


def code_of(tc):
    args = (tc.get("function") or {}).get("arguments") or ""
    try:
        return json.loads(args).get("code", args)
    except (ValueError, AttributeError):
        return args


def step_source(code):
    """Normalised source of a top-level `def step` in code, or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        m = re.search(r"^def step\(.*?(?=^\S|\Z)", code, re.S | re.M)
        return re.sub(r"\s+", " ", m.group(0)).strip() if m else None
    src = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "step":
            src = ast.get_source_segment(code, node)
    return re.sub(r"\s+", " ", src).strip() if src else None


def timeline(rows):
    """Calls in generation order: dict(id, pos, action, code, step, hosts=[host rows it produced])."""
    seen, out, last_resp, pending = set(), [], None, []
    for i, d in enumerate(rows):
        ev = d.get("event")
        if ev == "response":
            last_resp, pending = d, []
        elif ev == "modeler" and d.get("host"):
            pending.append(dict(d["host"], _row=i))
        elif ev == "request":
            new = []
            for m in d.get("messages") or []:
                if m.get("role") != "assistant":
                    continue
                for tc in m.get("tool_calls") or []:
                    if tc.get("id") not in seen:
                        seen.add(tc.get("id"))
                        new.append(tc)
            for k, tc in enumerate(new):
                code = code_of(tc)
                act = (last_resp or {}).get("action")
                out.append({"id": tc.get("id"), "pos": i, "action": int(act) if act not in (None, "") else None,
                            "code": code, "step": step_source(code),
                            "hosts": pending if k == len(new) - 1 else []})
            pending = []
    # host rows after the last request (run ended) are produced by a call never shown again; keep them unattributed
    return out, pending


def level_end(game_run, level):
    done, start = int(game_run["levels_completed"]), 0
    for i, n in enumerate(game_run["actions_per_level"]):
        if i + 1 == level:
            return start + n if level <= done else None
        start += n
    return None


def strict(h):
    v = h.get("verify") or {}
    return bool(h.get("exact")) and not v.get("mismatched") and not v.get("errors")


def analyse(rows, game_run):
    """Every host row carries the model version in effect when it ran (the last `step` source defined up to and
    including the call that produced it). A stop's chain is followed through host rows of later calls on the same
    level whose version differs from the version that stopped."""
    calls, orphan = timeline(rows)
    versions, in_effect = [], None
    for c in calls:
        if c["step"] and c["step"] != in_effect:
            in_effect = c["step"]
            versions.append(in_effect)
        c["version"] = len(versions)          # 0 = no step yet
    stops = []
    for ci, c in enumerate(calls):
        for h in c["hosts"]:
            run = h.get("run") or {}
            if run.get("outcome") != "mismatch":
                continue
            level = int(h.get("level", -1))
            end = level_end(game_run, level) if game_run else None
            later = [x for x in calls[ci + 1:] if end is None or x["action"] is None or x["action"] <= end]
            redef = [x for x in later if x["step"]]
            new_versions = sorted({x["version"] for x in later if x["version"] != c["version"]})
            rows_new = [(x, hh) for x in later for hh in x["hosts"]
                        if int(hh.get("level", -1)) == level and x["version"] != c["version"]]
            cert = [(x, hh) for x, hh in rows_new if hh.get("exact") and (hh.get("search") or {}).get("found")]
            cert_strict = [(x, hh) for x, hh in cert if strict(hh)]
            done = [(x, hh) for x, hh in rows_new if (hh.get("run") or {}).get("outcome") == "level_completed"]
            first = rows_new[0][1] if rows_new else None
            stops.append({
                "level": level, "stop_call": c["id"], "stop_version": c["version"],
                "stop_call_defines_step": bool(c["step"]), "at_step": h.get("at_step"),
                "action": run.get("action"), "differing_cells": run.get("differing_cells"),
                "executed_before_stop": run.get("executed"),
                "later_calls": len(later), "step_redefined": len(redef), "new_versions": len(new_versions),
                "first_new_version_host_row": None if first is None else {
                    "exact": first.get("exact"), "strict": strict(first),
                    "found": (first.get("search") or {}).get("found"), "run": (first.get("run") or {}).get("outcome")},
                "recertified_new_version": len(cert), "recertified_new_version_strict": len(cert_strict),
                "host_completed_new_version": bool(done),
                "completing_version_transitions": None if not done else (done[0][1].get("verify") or {}).get("transitions"),
                "level_completed": end is not None,
            })
    return stops, len(orphan)


def load(rf):
    rows = []
    for raw in open(rf, encoding="utf-8"):
        try:
            rows.append(json.loads(raw))
        except ValueError:
            pass
    return rows


def draw(out):
    bench = {g["game_id"][:4]: g for g in json.load(open(os.path.join(out, "benchmark.json")))["game_runs"]}
    stops, orphans = [], 0
    for rf in sorted(glob.glob(os.path.join(out, "*_p0_requests.jsonl"))):
        g4 = os.path.basename(rf)[:4]
        if g4 not in GAMES:
            continue
        s, o = analyse(load(rf), bench[g4])
        orphans += o
        for x in s:
            stops.append(dict(x, game=g4))
    return {"stops": stops, "unattributed_host_rows": orphans}


def pooled(stops):
    lv = {}
    for s in stops:
        lv.setdefault((s["draw"], s["game"], s["level"]), []).append(s)
    return {
        "stops": len(stops),
        "with_step_redefined": sum(1 for s in stops if s["step_redefined"]),
        "with_new_version": sum(1 for s in stops if s["new_versions"]),
        "recertified_new_version": sum(1 for s in stops if s["recertified_new_version"]),
        "recertified_new_version_strict": sum(1 for s in stops if s["recertified_new_version_strict"]),
        "host_completed_new_version": sum(1 for s in stops if s["host_completed_new_version"]),
        "level_runs_with_stop": len(lv),
        "level_runs_new_version": sum(1 for v in lv.values() if any(s["new_versions"] for s in v)),
        "level_runs_recertified_strict": sum(1 for v in lv.values() if any(s["recertified_new_version_strict"] for s in v)),
        "level_runs_recertified_near_exact_only": sum(1 for v in lv.values() if any(s["recertified_new_version"] for s in v)
                                                      and not any(s["recertified_new_version_strict"] for s in v)),
        "level_runs_host_completed_after_stop": sorted("%s %s L%d" % k for k, v in lv.items()
                                                       if any(s["host_completed_new_version"] for s in v)),
    }


def selftest():
    """The reviewer's case: model A defined, A's plan stops, no new definition afterwards -> 0 rewrites."""
    A = "def step(board, action):\n    return board\n"
    call = lambda cid, code: {"id": cid, "function": {"arguments": json.dumps({"code": code})}}
    sys_ = {"role": "system", "content": "s"}
    rows = [
        {"event": "request", "action": "1", "messages": [sys_]},
        {"event": "response", "action": "1", "messages": [sys_]},
        {"event": "modeler", "host": {"level": 1, "exact": True, "search": {"found": True}, "at_step": 3,
                                      "run": {"outcome": "mismatch", "executed": 2, "action": "UP"}}},
        {"event": "request", "action": "5", "messages": [sys_, {"role": "assistant", "tool_calls": [call("a", A)]}]},
        {"event": "response", "action": "5", "messages": []},
        {"event": "request", "action": "6", "messages": [sys_, {"role": "assistant", "tool_calls": [call("a", A)]},
                                                         {"role": "assistant", "tool_calls": [call("b", "print(1)")]}]},
        {"event": "response", "action": "6", "messages": []},
        {"event": "request", "action": "7", "messages": [sys_, {"role": "assistant", "tool_calls": [call("a", A)]},
                                                         {"role": "assistant", "tool_calls": [call("b", "print(1)")]},
                                                         {"role": "assistant", "tool_calls": [call("c", A)]}]},
    ]
    s, _ = analyse(rows, None)
    assert len(s) == 1 and s[0]["stop_call"] == "a", s
    assert s[0]["step_redefined"] == 1 and s[0]["new_versions"] == 0, s   # c re-sends A unchanged
    rows[-1]["messages"][-1]["tool_calls"][0] = call("c", A.replace("return board", "return board[::-1]"))
    s, _ = analyse(rows, None)
    assert s[0]["new_versions"] == 1, s
    print("selftest ok: the stopping definition is not counted; an unchanged re-send is not a new version")


def main():
    if sys.argv[1:] == ["--selftest"]:
        return selftest()
    m1, m2, report = sys.argv[1:4]
    res = {"m1": draw(m1), "m2": draw(m2)}
    allstops = [dict(s, draw=d) for d in ("m1", "m2") for s in res[d]["stops"]]
    res["pooled"] = pooled(allstops)
    res["per_draw"] = {d: pooled([s for s in allstops if s["draw"] == d]) for d in ("m1", "m2")}
    json.dump(res, open(report, "w"), indent=1)
    for s in allstops:
        print(s["draw"], s["game"], "L%d" % s["level"], "stop", s["at_step"], "+", s["executed_before_stop"],
              "stopcall_step", s["stop_call_defines_step"], "redef", s["step_redefined"],
              "newver", s["new_versions"], "first", s["first_new_version_host_row"],
              "cert", s["recertified_new_version"], "strict", s["recertified_new_version_strict"],
              "done", s["host_completed_new_version"], s["completing_version_transitions"])
    print("pooled", res["pooled"])
    print("per_draw", res["per_draw"])


if __name__ == "__main__":
    main()
