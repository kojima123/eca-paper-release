"""Divergence arms of the solved-level library (docs/library_divergence_registration_2026-11-03.md): one level per game, chosen by a
fixed rule, is made to diverge mid-way (arm R) or is removed (arm S); every other entry is the shipped library v2.

Choice (per game): library levels >= 2 outside the ECA solver's own levels (ar25 L1-3, cn04 L1-2, re86 L1, s5i5 L1-2, tn36 L1-5)
that the model played in most of the draws that cleared them outside the library draws (model clears / (clears - library
replays) >= 0.5); among them the one cleared in the fewest draws, ties to the higher level.
Arm R: action m = ceil(n/2) of the chosen level is replaced by the first substitute, in a fixed order (ACTION1..ACTION5 other than
the stored action, then ACTION6 at (0, 0), (63, 63), (32, 32)), among the actions the game offers at that board, and that leaves a board different from the
stored one after action m without completing the level or ending the game; the replay then stops right after action m with a real
board difference, and the level is the model's from there. Arm S: the chosen entry is removed (the model plays the level from its
start). Both libraries are checked here by replaying every game with the harness's rule (stop at the first stored-hash mismatch).

usage (Windows python, cwd C:/arc-agi3): python library_divergence.py LIBRARY_V3.json OUT_R.json OUT_S.json
(v3 holds the same 46 solutions as the shipped v2 and the clear counts of all four library draws, which LIB_REPLAYS assumes)
"""
import copy, json, math, sys
import numpy as np
import hashlib
import arc_agi
from arcengine import GameAction

ARC = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
SOLVER = {"ar25": {1, 2, 3}, "cn04": {1, 2}, "re86": {1}, "s5i5": {1, 2}, "tn36": {1, 2, 3, 4, 5}}
LIB_REPLAYS = 4          # library draws l1-l4 replayed every v1 level
NEW_IN_V2 = {("s5i5", 7): 2}   # s5i5 L7 entered at v2: replayed in l3, l4
SUBS = [("ACTION1", {}), ("ACTION2", {}), ("ACTION3", {}), ("ACTION4", {}), ("ACTION5", {}),
        ("ACTION6", {"x": 0, "y": 0}), ("ACTION6", {"x": 63, "y": 63}), ("ACTION6", {"x": 32, "y": 32})]


def fhash(obs):
    return hashlib.sha1(np.asarray(obs.frame[-1], dtype=np.int16).tobytes()).hexdigest()[:16]


def run(game, acts):
    env = ARC.make(game.split("-")[0]); obs = env.reset()
    for a in acts:
        r = env.step(getattr(GameAction, a[0]), data=a[1] or {})
        obs = r if r is not None else obs
    return obs


def choose(entries, g4):
    cands = []
    for e in entries:
        lv = int(e["level"])
        if lv < 2 or lv in SOLVER.get(g4, set()):
            continue
        base = int(e["draws_cleared"]) - NEW_IN_V2.get((g4, lv), LIB_REPLAYS)
        if base <= 0 or int(e["draws_cleared_by_model"]) / base < 0.5:
            continue
        cands.append((base, -lv, e))
    return min(cands, key=lambda c: (c[0], c[1]))[2] if cands else None


def check(lib, game):
    """Replay the game's entries with the harness rule; return [(level, outcome, done, n)]."""
    env = ARC.make(game.split("-")[0]); obs = env.reset(); out = []
    for e in sorted(lib["games"][game], key=lambda e: e["level"]):
        if obs.levels_completed + 1 != int(e["level"]) or fhash(obs) != e["start_hash"]:
            out.append((int(e["level"]), "not_reached", 0, len(e["actions"]))); break
        lv0, outcome, i = obs.levels_completed, "diverged", -1
        for i, (name, data) in enumerate(e["actions"]):
            r = env.step(getattr(GameAction, name), data=data or {}); obs = r if r is not None else obs
            if obs.levels_completed > lv0:
                outcome = "completed" if i == len(e["actions"]) - 1 else "diverged"; break
            if str(obs.state).endswith("GAME_OVER"):
                outcome = "game_over"; break
            if fhash(obs) != e["frame_hashes"][i]:
                break
        out.append((int(e["level"]), outcome, i + 1, len(e["actions"])))
        if outcome != "completed":
            break
    return out


def main():
    lib = json.load(open(sys.argv[1], encoding="utf-8"))
    R, S = copy.deepcopy(lib), copy.deepcopy(lib)
    for game, entries in sorted(lib["games"].items()):
        g4 = game[:4]
        e = choose(entries, g4)
        lv, n = int(e["level"]), len(e["actions"])
        m = math.ceil(n / 2)
        prefix = [a for x in sorted(entries, key=lambda x: x["level"]) if x["level"] < lv for a in x["actions"]]
        before = run(game, prefix + e["actions"][:m - 1])
        h_before, lv_before = fhash(before), before.levels_completed
        stored = e["actions"][m - 1]
        pick = None
        avail = {int(getattr(a, "value", a)) for a in before.available_actions}
        for name, data in SUBS:
            if [name, data] == [stored[0], stored[1] or {}] or int(getattr(GameAction, name).value) not in avail:
                continue
            try:
                after = run(game, prefix + e["actions"][:m - 1] + [[name, data]])
            except Exception:
                continue
            if after.levels_completed > lv_before or str(after.state).endswith("GAME_OVER"):
                continue
            if fhash(after) == e["frame_hashes"][m - 1]:
                continue
            pick = ([name, data], fhash(after) == h_before)
            break
        assert pick is not None, (game, lv)
        re_ = next(x for x in R["games"][game] if x["level"] == lv)
        re_["actions"][m - 1] = pick[0]
        re_["divergence"] = {"at": m, "stored": stored, "substituted": pick[0], "noop": pick[1]}
        S["games"][game] = [x for x in S["games"][game] if x["level"] != lv]
        print("%s L%d (cleared %d, model %d, %d actions): R replaces action %d %s by %s (%s); S removes it" % (
            game, lv, e["draws_cleared"], e["draws_cleared_by_model"], n, m, stored, pick[0],
            "leaves the board as it was" if pick[1] else "changes the board differently"), flush=True)
        r = check(R, game); s = check(S, game)
        print("   check R:", r); print("   check S:", s, flush=True)
        assert r[-1][0] == lv and r[-1][1] == "diverged" and r[-1][2] == m and all(x[1] == "completed" for x in r[:-1]), r
        s_done = [x for x in s if x[1] != "not_reached"]   # entries above the removed level are never reached without the model
        assert all(x[1] == "completed" for x in s_done) and [x[0] for x in s_done] == [x["level"] for x in S["games"][game] if x["level"] < lv], s
    for p, d in ((sys.argv[2], R), (sys.argv[3], S)):
        json.dump(d, open(p, "w", encoding="utf-8"))
    print("DIVERGENCE_LIBRARIES_OK R %d levels, S %d levels" % (sum(len(v) for v in R["games"].values()),
                                                              sum(len(v) for v in S["games"].values())))


if __name__ == "__main__":
    main()
