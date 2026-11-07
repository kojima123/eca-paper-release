"""Automaticity within a first encounter (docs/within_run_repeats_registration_2026-11-03.md): repeated prefixes after RESET / GAME OVER
and known transitions in one run, with the tokens and seconds of the turns spent on them. LLM-free; replays on the engines.

usage (Windows python, cwd C:/arc-agi3):
  python within_run_repeats.py OUT.json public @DRAWS.txt
  python within_run_repeats.py OUT.json nv25 NV25_BENCHMARK.json
"""
import collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from closure_census import fhash, akey, public_runner, nv25_runner  # noqa: E402


def analyse(run, g):
    hist = g["history"]
    steps = list(run(g["game_id"], hist))           # [(None, frame0, ...), (a1, frame1, ...), ...]
    attempts = collections.defaultdict(list)          # level -> list of attempts (lists of (action key, history index))
    cur, lv, seen = [], None, set()
    rows = []                                          # per model action: (index, repeated_prefix?, known_transition?)
    prev_s = None
    for i, (a, frame, levels, state) in enumerate(steps):
        s = fhash(frame)
        if a is None:
            lv, prev_s = levels, s
            continue
        k = akey(a)
        if a["id"] == "RESET":
            if cur:
                attempts[lv].append(cur)
            cur, prev_s, lv = [], s, levels
            continue
        known = (prev_s, k) in seen
        seen.add((prev_s, k))
        cur.append(k)
        rep = any(len(p) >= len(cur) and p[:len(cur)] == cur for p in attempts[lv])
        rows.append((i - 1, rep, known))
        if levels != lv or state.endswith("GAME_OVER"):
            if cur and levels == lv:
                attempts[lv].append(cur)
            cur, lv = [], levels
        prev_s = s
    # turn costs: tokens / seconds attach to the first action of a turn (generated_tokens > 0 opens a model turn)
    tok = [int(h.get("generated_tokens") or 0) for h in hist]
    t = [float(h.get("wallclock_seconds") or 0) for h in hist]
    flag = {i: (r, kn) for i, r, kn in rows}
    turns = []                                         # (start, end) index ranges of model turns
    for i, v in enumerate(tok):
        if v > 0:
            turns.append([i, i])
        elif turns and i == turns[-1][1] + 1:
            turns[-1][1] = i
    rep_tok = rep_sec = all_tok = all_sec = 0.0
    for a0, a1 in turns:
        dur = (t[a0] - t[a0 - 1]) if a0 > 0 else 0.0
        all_tok += tok[a0]; all_sec += max(0.0, dur)
        if all(flag.get(j, (False, False))[0] for j in range(a0, a1 + 1)):
            rep_tok += tok[a0]; rep_sec += max(0.0, dur)
    model_idx = {j for a0, a1 in turns for j in range(a0, a1 + 1)}
    m = [r for r in rows if r[0] in model_idx]
    return dict(actions=len(rows), model_actions=len(m), restarts=sum(1 for h in hist if h["action"]["id"] == "RESET"),
                repeated_prefix_actions=sum(1 for r in m if r[1]), known_transition_actions=sum(1 for r in m if r[2]),
                turns=len(turns), tokens=all_tok, seconds=all_sec, repeated_turn_tokens=rep_tok, repeated_turn_seconds=rep_sec)


def main():
    out, src = sys.argv[1], sys.argv[2]
    if src == "public":
        paths = [l.strip() for l in open(sys.argv[3][1:])] if sys.argv[3].startswith("@") else sys.argv[3:]
        run = public_runner()
    else:
        paths = sys.argv[3:]; run = nv25_runner()
    per = collections.defaultdict(collections.Counter); bad = []
    for i, p in enumerate(x for x in paths if x):
        for g in json.load(open(p))["game_runs"]:
            try:
                per[g["game_id"][:6]].update(analyse(run, g))
            except Exception as e:  # noqa: BLE001
                bad.append((p, g["game_id"], "%s: %s" % (type(e).__name__, e)))
        print("read", i, flush=True)
    pooled = collections.Counter()
    for v in per.values():
        pooled.update(v)

    def shares(c):
        d = dict(c)
        d["repeated_prefix_share"] = round(c["repeated_prefix_actions"] / max(1, c["model_actions"]), 4)
        d["known_transition_share"] = round(c["known_transition_actions"] / max(1, c["model_actions"]), 4)
        d["repeated_token_share"] = round(c["repeated_turn_tokens"] / max(1, c["tokens"]), 4)
        d["repeated_time_share"] = round(c["repeated_turn_seconds"] / max(1, c["seconds"]), 4)
        return d
    res = dict(source=src, draws=len(paths), errors=bad, pooled=shares(pooled), per_game={g: shares(v) for g, v in sorted(per.items())})
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res["pooled"], indent=1)); print("WITHIN_RUN_DONE", src, "errors", len(bad))


if __name__ == "__main__":
    main()
