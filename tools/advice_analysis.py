"""Authority vs advice (experiment 2) and same-configuration pass@k (experiment 3) - docs/eca_advice_registration_2026-11-03.md.

Per draw: levels per game and in total, mean game score, generated tokens. Solver-owned levels (registered list: ar25 L1-3,
cn04 L1-2, re86 L1, s5i5 L1-2, tn36 L1-5 - 13 levels; the registration says 15, the list sums to 13) completed, and the actions,
generated tokens and game minutes until each game's last owned level was completed (actions_per_level). Advice draws: every "eca advice: <game> L<level> at=<n> family=<f>
plan=<json>" line of colab_run.log is an offer made when the game's history had n actions; it is followed when history[n] (the next
executed action) equals the plan's first action (UP/DOWN/LEFT/RIGHT/SPACE = ACTION1-5, MOUSE(row=r, col=c) = ACTION6 x=c, y=r).
pass@k over the auth draws: per (game, level) the draws that cleared it.

usage: python3 tools/advice_analysis.py OUT.json
"""
import collections, json, re, sys

E = "/mnt/e/arc-agi3-online-specialists-runs-e"
AUTH = {d: f"{E}/advice_runs/{d}/out_canon" for d in ("a1", "a2", "a3", "a4")}
ADV = {d: f"{E}/advice_runs/{d}/out_canon" for d in ("v1", "v2")}
CTL = {d: f"{E}/explain2_runs/{d}/out_canon" for d in ("ctl1", "ctl2")}
OWNED = {"ar25": [1, 2, 3], "cn04": [1, 2], "re86": [1], "s5i5": [1, 2], "tn36": [1, 2, 3, 4, 5]}
NAMES = {"UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3", "RIGHT": "ACTION4", "SPACE": "ACTION5"}
LINE = re.compile(r"eca advice: (\S+) L(\d+) at=(\d+) family=(\S+) plan=(\[.*\])")


def load(d):
    try:
        return json.load(open(d + "/benchmark.json"))
    except FileNotFoundError:
        return None


def draw_summary(b):
    games = {g["game_id"][:4]: g for g in b["game_runs"]}
    owned = sum(1 for k, ls in OWNED.items() for L in ls if games[k]["levels_completed"] >= L)
    cost = {}
    for k, ls in OWNED.items():   # actions, generated tokens and minutes until the last owned level was completed
        g = games[k]
        if g["levels_completed"] >= max(ls):
            n = sum(g["actions_per_level"][:max(ls)]); seg = g["history"][:n]
            cost[k] = [n, sum(int(x.get("generated_tokens") or 0) for x in seg), round(float(seg[-1]["wallclock_seconds"]) / 60, 1)]
    return dict(levels=sum(g["levels_completed"] for g in games.values()),
                per_game={k: g["levels_completed"] for k, g in sorted(games.items())},
                mean_score=round(sum(g["final_score"] for g in games.values()) / len(games), 2),
                tokens_M=round(sum(int(h.get("generated_tokens") or 0) for g in games.values() for h in g["history"]) / 1e6, 3),
                owned_completed=owned, owned_total=sum(len(v) for v in OWNED.values()), owned_cost=cost,
                owned_cost_sum=[sum(v[0] for v in cost.values()), sum(v[1] for v in cost.values()),
                                round(sum(v[2] for v in cost.values()), 1)])


def same(plan0, h):
    a = h["action"]; d = a.get("data") or {}
    m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", plan0)
    if m:
        return a["id"] == "ACTION6" and int(d.get("y", -1)) == int(m.group(1)) and int(d.get("x", -1)) == int(m.group(2))
    return a["id"] == NAMES.get(plan0)


def offers(d, b):
    games = {g["game_id"]: g["history"] for g in b["game_runs"]}
    out = []
    for line in open(d + "/colab_run.log", errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        gid, lv, at, fam, plan = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), json.loads(m.group(5))
        h = games.get(gid, [])
        nxt = h[at] if at < len(h) else None
        out.append(dict(game=gid[:4], level=lv, at=at, family=fam, plan0=plan[0] if plan else None,
                        next=None if nxt is None else nxt["action"], followed=bool(nxt and plan and same(plan[0], nxt))))
    return out


def main():
    res = {}
    for name, arm in (("auth", AUTH), ("advice", ADV), ("control", CTL)):
        for d, path in arm.items():
            b = load(path)
            if b is None:
                res[d] = "missing"; continue
            s = draw_summary(b); s["arm"] = name
            if name == "advice":
                o = offers(path, b)
                s["offers"] = len(o); s["followed"] = sum(x["followed"] for x in o)
                s["follow_rate"] = round(s["followed"] / max(1, len(o)), 3)
                s["offers_by_game"] = {g: [sum(1 for x in o if x["game"] == g), sum(x["followed"] for x in o if x["game"] == g)]
                                       for g in sorted({x["game"] for x in o})}
            res[d] = s
    # pass@k over auth draws present
    auth = [load(p) for p in AUTH.values()]; auth = [b for b in auth if b]
    clears = collections.Counter()
    for b in auth:
        for g in b["game_runs"]:
            for L in range(1, g["levels_completed"] + 1):
                clears[(g["game_id"][:4], L)] += 1
    k = len(auth)
    res["passk_auth"] = dict(draws=k, mean_levels=round(sum(clears.values()) / max(1, k), 2), union=len(clears),
                             all_draws=sum(1 for v in clears.values() if v == k),
                             some_not_all=sum(1 for v in clears.values() if 0 < v < k),
                             per_level={"%s L%d" % key: v for key, v in sorted(clears.items())})
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    for d, s in res.items():
        if isinstance(s, dict) and "levels" in s:
            print(d, s["arm"], "levels", s["levels"], "owned", s["owned_completed"], "/", s["owned_total"], "score", s["mean_score"], "owned cost [actions, tokens, min]", s["owned_cost_sum"],
                  "tokens", s["tokens_M"], ("offers %d followed %d rate %.3f %s" % (s["offers"], s["followed"], s["follow_rate"], s["offers_by_game"])) if "offers" in s else "")
    print("passk_auth", {k: v for k, v in res["passk_auth"].items() if k != "per_level"})


if __name__ == "__main__":
    main()
