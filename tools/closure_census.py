"""Closure census and apartness repair on recorded play (docs/closure_census_registration_2026-11-03.md). LLM-free.

Every history is replayed on its engine; each non-RESET action gives a transition (game, state, action) -> next state, state = the
last frame's hash (the library's). A key seen >= 2 times is broken if its observations disagree. Apartness repair: the shortest
suffix k <= 6 of the level's action history (RESET and level change restart it) that makes every branch of a broken key pure.
Held-out: even draws learn, odd draws test.
CENSUS_STATE=level (default frame) adds the level index to the state: state = (levels completed, last frame's hash), so a key can
no longer pool observations from different levels (review 11-03: separate level-rule differences from within-level history).

usage (Windows python, cwd C:/arc-agi3):
  python closure_census.py OUT.json public @DRAWS.txt
  python closure_census.py OUT.json nv25 NV25_BENCHMARK.json
"""
import collections, hashlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np

K = 6
STATE = os.environ.get("CENSUS_STATE", "frame")
NV = Path(r"C:\arc-agi3-artifacts\unseen_ds\nvidia25")


def fhash(frame):
    return hashlib.sha1(np.asarray(frame, dtype=np.int16).tobytes()).hexdigest()[:16]


def akey(a):
    d = a.get("data") or {}
    return a["id"] + ("@%d,%d" % (int(d.get("x", -1)), int(d.get("y", -1))) if a["id"] == "ACTION6" else "")


def public_runner():
    import arc_agi
    from arcengine import GameAction
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)

    def run(game_id, history):
        env = arc.make(game_id.split("-")[0]); obs = env.reset()
        yield None, obs.frame[-1], obs.levels_completed, str(obs.state)
        for h in history:
            a = h["action"]
            r = env.step(getattr(GameAction, a["id"]), data={k: int(v) for k, v in (a.get("data") or {}).items()})
            obs = r if r is not None else obs
            yield a, obs.frame[-1], obs.levels_completed, str(obs.state)
    return run


def nv25_runner():
    sys.path.insert(0, str(NV / "runtime_support"))
    from arcengine import ActionInput, GameAction

    def run(game_id, history):
        name = game_id.split("-")[0]
        f = next((NV / "environment_files" / name).glob("*/%s.py" % name))
        meta = json.load(open(f.parent / "metadata.json"))
        sp = importlib.util.spec_from_file_location(name, f); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
        game = getattr(m, meta["class_name"])()
        act = lambda i, data=None: game.perform_action(ActionInput(id=GameAction.from_id(int(i)), data=data or {}), raw=True)
        raw = act(0)
        yield None, raw.frame[-1], raw.levels_completed, str(raw.state)
        for h in history:
            a = h["action"]
            raw = act(GameAction[a["id"]].value, {k: int(v) for k, v in (a.get("data") or {}).items()} or None)
            yield a, raw.frame[-1], raw.levels_completed, str(raw.state)
    return run


def transitions(run, game_id, history, draw):
    out, prev, hist, lv = [], None, [], None
    for a, frame, levels, state in run(game_id, history):
        s = fhash(frame) if STATE == "frame" else "L%d:%s" % (levels, fhash(frame))
        if a is not None and a["id"] != "RESET" and prev is not None:
            out.append(dict(game=game_id[:6], draw=draw, s=prev, a=akey(a), t=s, suffix=tuple(hist[-K:])))
        if a is None or a["id"] == "RESET" or levels != lv or state.endswith("GAME_OVER"):
            hist = []
        else:
            hist.append(akey(a))
        if a is not None and a["id"] == "RESET":
            hist = []
        prev, lv = s, levels
    return out


def pure_k(obs):
    """Smallest k in 1..K at which every branch (suffix[-k:]) of these observations has one outcome; supported if every branch >= 2."""
    for k in range(1, K + 1):
        br = collections.defaultdict(list)
        for o in obs:
            br[("<s>",) * max(0, k - len(o["suffix"])) + o["suffix"][-k:] if k else ()].append(o["t"])
        if all(len(set(v)) == 1 for v in br.values()):
            return k, all(len(v) >= 2 for v in br.values()), br
    return None, False, None


def census(trs):
    by = collections.defaultdict(list)
    for o in trs:
        by[(o["game"], o["s"], o["a"])].append(o)
    rep = {k: v for k, v in by.items() if len(v) >= 2}
    broken = {k: v for k, v in rep.items() if len({o["t"] for o in v}) > 1}
    ks = collections.Counter(); sup = 0
    for v in broken.values():
        k, s, _ = pure_k(v); ks[k] += 1; sup += s
    return dict(transitions=len(trs), keys=len(by), repeated_keys=len(rep), broken_keys=len(broken),
                broken_share=round(len(broken) / max(1, len(rep)), 4),
                transitions_on_broken=sum(len(v) for v in broken.values()),
                repair_k={str(k): n for k, n in sorted(ks.items(), key=lambda x: (x[0] is None, x[0] or 0))},
                repaired=sum(n for k, n in ks.items() if k is not None), repaired_supported=sup)


def heldout(trs):
    learn = [o for o in trs if o["draw"] % 2 == 0]; test = [o for o in trs if o["draw"] % 2 == 1]
    by = collections.defaultdict(list)
    for o in learn:
        by[(o["game"], o["s"], o["a"])].append(o)
    n = a_ok = b_ans = b_ok = ab_ok = 0
    for o in test:
        v = by.get((o["game"], o["s"], o["a"]))
        if not v or len({x["t"] for x in v}) < 2:
            continue
        n += 1
        maj = collections.Counter(x["t"] for x in v).most_common(1)[0][0]
        a_ok += maj == o["t"]
        k, _, br = pure_k(v)
        if k is None:
            continue
        key = ("<s>",) * max(0, k - len(o["suffix"])) + o["suffix"][-k:]
        if key in br:
            b_ans += 1; b_ok += br[key][0] == o["t"]; ab_ok += maj == o["t"]
    return dict(test_on_broken=n, majority_acc=round(a_ok / max(1, n), 4), suffix_answers=b_ans,
                suffix_coverage=round(b_ans / max(1, n), 4), suffix_acc=round(b_ok / max(1, b_ans), 4),
                majority_acc_same_cases=round(ab_ok / max(1, b_ans), 4))


def main():
    out, src = sys.argv[1], sys.argv[2]
    if src == "public":
        paths = [l.strip() for l in open(sys.argv[3][1:])] if sys.argv[3].startswith("@") else sys.argv[3:]
        run = public_runner()
    else:
        paths = sys.argv[3:]; run = nv25_runner()
    trs, bad = [], []
    for i, p in enumerate(p for p in paths if p):
        b = json.load(open(p))
        for g in b["game_runs"]:
            try:
                trs += transitions(run, g["game_id"], g["history"], i)
            except Exception as e:  # noqa: BLE001
                bad.append((p, g["game_id"], "%s: %s" % (type(e).__name__, e)))
        print("read", i, p, len(trs), flush=True)
    games = sorted({o["game"] for o in trs})
    res = dict(source=src, state=STATE, draws=len(paths), errors=bad, pooled=census(trs), heldout=heldout(trs),
               per_game={g: dict(census([o for o in trs if o["game"] == g]), heldout=heldout([o for o in trs if o["game"] == g])) for g in games})
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(dict(pooled=res["pooled"], heldout=res["heldout"], errors=len(bad)), indent=1))
    print("CLOSURE_CENSUS_DONE", src)


if __name__ == "__main__":
    main()
