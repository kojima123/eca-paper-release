"""Measures the paper outline quotes, reproducible from the run directories (review 11-03: tie every number to inputs, script, output).

1 control rate: seconds since the previous action for (a) library replay actions (the first n actions of each game-run in the library
  draws l1-l4, n = the library's actions for that game; every one checked to carry 0 generated tokens) and (b) actions that carry
  generated tokens (the first action of an LLM turn; its interval includes the turn's thinking), in l1-l4 + ctl1-ctl2 and in
  ctl1-ctl2 alone. These are per-action decision intervals, not task-level speed.
2 tokens per LLM turn: generated tokens on the actions in (b).
3 context: per request prompt tokens (mean and median) in the control-family draws ctl1, ctl2, f15, f16; drains = a request whose
  prompt is < 0.6 x the previous one; requests between drains, prompt tokens just before a drain, drains per game-hour with the
  game-runs' own durations (final_wallclock_seconds).
4 frontier budget in the library draws: per game-run whose game the library does not complete, the time from the end of the replay
  to the end of the run, and the levels cleared after the replay.
5 s5i5 L7 chain: cleared by the model in l2 (actions on the level), harvested (library v2 length), replayed in l3 / l4 (0 tokens).

usage: python3 tools/paper_measures.py OUT.json
"""
import glob, json, statistics as st, sys

E = "/mnt/e/arc-agi3-online-specialists-runs-e"
F = "/mnt/f/arc-agi3-online-specialists-20261014/research/solver_library"
LIB = {1: "library_v1", 2: "library_v1", 3: "library_v2", 4: "library_v2"}
CTL = [f"{E}/explain2_runs/ctl1/out_canon", f"{E}/explain2_runs/ctl2/out_canon"]
CTX = CTL + [f"{E}/failnote_runs/f15/out_canon", f"{E}/failnote_runs/f16/out_canon"]


def q(x):
    x = sorted(x); n = len(x)
    return dict(n=n, median=round(x[n // 2], 3), p90=round(x[int(n * .9)], 3), mean=round(st.mean(x), 3)) if n else dict(n=0)


def bench(d):
    return json.load(open(d + "/benchmark.json"))


def library(v):
    L = json.load(open(f"{F}/{v}.json"))["games"]
    return {g[:4]: es for g, es in L.items()}


def intervals(h):
    w = [float(x.get("wallclock_seconds") or 0) for x in h]
    return [(j, w[j] - w[j - 1]) for j in range(1, len(h)) if w[j] >= w[j - 1]]


def main():
    out = {}
    # 1, 2
    rep, turn_all, turn_ctl, tok_all = [], [], [], []
    for i in (1, 2, 3, 4):
        lib = library(LIB[i])
        for g in bench(f"{E}/library_runs/l{i}/out_canon")["game_runs"]:
            h = g["history"]; n = sum(len(e["actions"]) for e in lib.get(g["game_id"][:4], []))
            assert all(int(x.get("generated_tokens") or 0) == 0 for x in h[:n]), (i, g["game_id"])
            for j, d in intervals(h):
                if j < n:
                    rep.append(d)
                elif int(h[j].get("generated_tokens") or 0) > 0:
                    turn_all.append(d); tok_all.append(int(h[j]["generated_tokens"]))
    for d in CTL:
        for g in bench(d)["game_runs"]:
            h = g["history"]
            for j, dt in intervals(h):
                if int(h[j].get("generated_tokens") or 0) > 0:
                    turn_all.append(dt); turn_ctl.append(dt); tok_all.append(int(h[j]["generated_tokens"]))
    out["library_replay_interval_s"] = q(rep)
    out["llm_turn_interval_s_l1-4_ctl1-2"] = q(turn_all)
    out["llm_turn_interval_s_ctl1-2"] = q(turn_ctl)
    out["generated_tokens_per_llm_turn"] = q(tok_all)
    # 3
    pts_all, gaps, caps, drains, hours, runs = [], [], [], 0, 0.0, 0
    for d in CTX:
        dur = {g["game_id"]: float(g.get("final_wallclock_seconds") or 0) for g in bench(d)["game_runs"]}
        for f in sorted(glob.glob(d + "/*_p0_requests.jsonl")):
            pts = []
            for line in open(f, encoding="utf-8", errors="replace"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                u = r.get("usage") or (r.get("response") or {}).get("usage") or {}
                if u.get("prompt_tokens"):
                    pts.append(int(u["prompt_tokens"]))
            if not pts:
                continue
            gid = f.split("/")[-1].replace("_p0_requests.jsonl", "")
            runs += 1; hours += dur.get(gid, 0) / 3600; pts_all += pts; last = 0
            for k, (a, b) in enumerate(zip(pts, pts[1:])):
                if b < 0.6 * a:
                    drains += 1; gaps.append(k + 1 - last); last = k + 1; caps.append(a)
    out["context"] = dict(draws=[d.split("/")[-2] for d in CTX], game_runs=runs, game_hours=round(hours, 2),
                          prompt_tokens_per_request=q(pts_all), drains=drains, drains_per_game_hour=round(drains / hours, 2),
                          requests_between_drains=q(gaps), prompt_tokens_before_drain=q(caps))
    # 4
    fr = []
    for i in (1, 2, 3, 4):
        lib = library(LIB[i])
        for g in bench(f"{E}/library_runs/l{i}/out_canon")["game_runs"]:
            es = lib.get(g["game_id"][:4], []); n = sum(len(e["actions"]) for e in es)
            if len(es) >= g["number_of_levels"]:
                continue
            h = g["history"]; t0 = float(h[n - 1]["wallclock_seconds"]) if n else 0.0
            end = float(g.get("final_wallclock_seconds") or h[-1]["wallclock_seconds"])
            fr.append(dict(draw="l%d" % i, game=g["game_id"][:4], library_levels=len(es), levels=g["levels_completed"],
                           new_levels=g["levels_completed"] - len(es), frontier_minutes=round((end - t0) / 60, 1),
                           model_actions=len(h) - n))
    out["frontier"] = dict(game_runs=len(fr), game_hours=round(sum(x["frontier_minutes"] for x in fr) / 60, 2),
                           new_levels=sum(x["new_levels"] for x in fr), runs=fr)
    # 5
    v2 = library("library_v2")["s5i5"]
    l2 = next(g for g in bench(f"{E}/library_runs/l2/out_canon")["game_runs"] if g["game_id"].startswith("s5i5"))
    chain = dict(l2_levels=l2["levels_completed"], l2_actions_per_level=l2.get("actions_per_level"),
                 v2_L7_actions=len(v2[6]["actions"]), v2_L7_best_draw=v2[6]["n_actions_best_draw"])
    n = sum(len(e["actions"]) for e in v2)
    for i in (3, 4):
        g = next(g for g in bench(f"{E}/library_runs/l{i}/out_canon")["game_runs"] if g["game_id"].startswith("s5i5"))
        chain["l%d" % i] = dict(levels=g["levels_completed"], replay_actions=n,
                                replay_tokens=sum(int(x.get("generated_tokens") or 0) for x in g["history"][:n]))
    out["s5i5_L7_chain"] = chain
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
