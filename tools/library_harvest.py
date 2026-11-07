"""Solved-level library, stage 1: harvest every cleared level from past draws, verify it by replay on the local engine, shorten it.

For each draw's benchmark.json (game_runs[].history: action id + data, generated_tokens, wallclock_seconds) the history is replayed on the
local engine (arc_agi OFFLINE, run with cwd C:/arc-agi3 for environment_files) and cut into levels. A level's clean sequence is the
actions after its last RESET (automatic after game over, or the player's) up to the completing action; its cost is everything spent on
the level (all attempts). Per (game, level) the candidates are tried shortest first from the level's canonical start (reached by
replaying the chosen sequences of the earlier levels); the first one that completes the level is kept, then shortened by cutting
revisits of an identical frame and dropping actions that leave the frame unchanged, each cut kept only if the level still completes.
The entry stores the start frame hash and the frame hash after every action (the runtime check), the human baseline, the source draws
and whether the level was played by the model (any generated token) or by the solver.

usage (Windows python, cwd C:/arc-agi3): python library_harvest.py OUT.json BENCHMARK.json [BENCHMARK.json ...] | @LIST.txt
"""
import hashlib, json, sys, collections
import numpy as np
import arc_agi
from arcengine import GameAction


def fhash(obs):
    return hashlib.sha1(np.asarray(obs.frame[-1], dtype=np.int16).tobytes()).hexdigest()[:16]


ARC = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)


def fresh(game):
    env = ARC.make(game.split("-")[0])
    return env, env.reset()


def step(env, obs, act):
    r = env.step(getattr(GameAction, act[0]), data=act[1] or {})
    return r if r is not None else obs


def cut(game_run):
    """Replay one history; return per level (clean sequence, model played?, tokens, seconds) and whether the replay matched."""
    env, obs = fresh(game_run["game_id"])
    out, cur, tok, sec, model = {}, [], 0, 0.0, False
    for h in game_run["history"]:
        a = h["action"]
        act = (a["id"], {k: int(v) for k, v in (a.get("data") or {}).items()})
        tok += int(h.get("generated_tokens") or 0)
        sec += float(h.get("wallclock_seconds") or 0)
        model = model or int(h.get("generated_tokens") or 0) > 0
        lv0 = obs.levels_completed
        obs = step(env, obs, act)
        if act[0] == "RESET":
            cur = []
            continue
        cur.append(act)
        if obs.levels_completed > lv0:
            out[lv0 + 1] = (cur, model, tok, sec)
            cur, tok, sec, model = [], 0, 0.0, False
        elif str(obs.state).endswith("GAME_OVER"):
            cur = []
    return out, obs.levels_completed == game_run["levels_completed"]


def play(game, prefix_levels, seq):
    """From game start play the chosen earlier levels, then seq; return (completed?, start hash, hashes after each action)."""
    env, obs = fresh(game)
    for s in prefix_levels:
        for act in s:
            obs = step(env, obs, act)
    lv0 = obs.levels_completed
    if lv0 != len(prefix_levels):
        return False, None, []
    start, hs = fhash(obs), []
    for i, act in enumerate(seq):
        obs = step(env, obs, act)
        hs.append(fhash(obs))
        if obs.levels_completed > lv0:
            return i == len(seq) - 1, start, hs
        if str(obs.state).endswith("GAME_OVER"):
            return False, start, hs
    return False, start, hs


def shorten(game, prefix, seq):
    ok, start, hs = play(game, prefix, seq)
    changed = True
    while changed:
        changed = False
        seen = {start: -1}
        for i, h in enumerate(hs):
            if h in seen:
                cand = seq[:seen[h] + 1] + seq[i + 1:]
                ok2, _, hs2 = play(game, prefix, cand)
                if ok2:
                    seq, hs, changed = cand, hs2, True
                    break
            seen[h] = i
        if changed:
            continue
        prev = start
        for i, h in enumerate(hs):
            if h == prev:
                cand = seq[:i] + seq[i + 1:]
                ok2, _, hs2 = play(game, prefix, cand)
                if ok2:
                    seq, hs, changed = cand, hs2, True
                    break
            prev = h
    return seq, start, hs


def main(out_path, paths):
    cands = collections.defaultdict(list)   # (game, level) -> [(seq, model, tok, sec, source)]
    base, replay_bad = {}, []
    for p in paths:
        b = json.load(open(p))
        for g in b.get("game_runs", []):
            gid = g["game_id"]
            base[gid] = g["base_actions_per_level"]
            levels, matched = cut(g)
            if not matched:
                replay_bad.append((p, gid))
            for lv, (seq, model, tok, sec) in levels.items():
                cands[(gid, lv)].append((seq, model, tok, sec, p))
        print("read", p, flush=True)
    lib = {"games": {}, "replay_mismatch": replay_bad}
    for gid in sorted({k[0] for k in cands}):
        chosen, entries = [], []
        lv = 1
        while (gid, lv) in cands:
            cs = sorted(cands[(gid, lv)], key=lambda c: len(c[0]))
            pick = None
            for seq, *_ in cs:
                ok, _, _ = play(gid, chosen, seq)
                if ok:
                    pick = seq
                    break
            if pick is None:
                print(gid, "L%d" % lv, "no candidate completes from the canonical start", flush=True)
                break
            short, start, hs = shorten(gid, chosen, list(pick))
            n_model = sum(1 for c in cs if c[1])
            entries.append({
                "level": lv, "start_hash": start, "actions": [[a, d] for a, d in short], "frame_hashes": hs,
                "n_actions": len(short), "n_actions_best_draw": len(pick), "human_actions": base[gid][lv - 1],
                "draws_cleared": len(cs), "draws_cleared_by_model": n_model,
                "model_cost_mean": {"tokens": round(sum(c[2] for c in cs if c[1]) / max(n_model, 1)),
                                    "seconds": round(sum(c[3] for c in cs if c[1]) / max(n_model, 1), 1)},
            })
            print(gid, "L%d" % lv, "draws %d (model %d), best %d -> %d actions, human %d" % (
                len(cs), n_model, len(pick), len(short), base[gid][lv - 1]), flush=True)
            chosen.append(short)
            lv += 1
        lib["games"][gid] = entries
    json.dump(lib, open(out_path, "w"), indent=1)
    print("LIBRARY_WRITTEN", out_path, sum(len(v) for v in lib["games"].values()), "levels; replay mismatches", len(replay_bad))


if __name__ == "__main__":
    args = sys.argv[2:]
    if len(args) == 1 and args[0].startswith("@"):
        args = [l.strip() for l in open(args[0][1:]) if l.strip()]
    main(sys.argv[1], args)
