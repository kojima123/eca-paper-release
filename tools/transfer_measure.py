"""Transfer measurement: can a host-side model fit on levels 1..k of a run play level k+1, without the LLM's turns?

Input: the boards of the solved-level library (tools/transfer_frames.py). For each game and each pair k -> k+1:

1. Dynamics. An ECA-style effect table fit only on the non-completing transitions of levels 1..k (generic, pixel objects from
   board_digest.components / object_changes, border 4): per action key (the action, or for a click the clicked object's colour
   and size, board_digest.ledger_key's convention) the most common effect template - nothing, objects of a colour and shape moved
   by a displacement (with the colours they were seen to enter and the colour they leave behind), or the clicked / named objects
   turned another colour. Applied to level k+1's boards: exact = the predicted board equals the real one inside the border;
   object-level = the same effect signature (board_digest.effect_signature). Baseline: nothing changes.
2. Goal. From the completions of levels 1..k (before vs the settled board, as completion_facts takes it): the filled-in relation
   when every earlier completion had it, else the effect-signature terms common to all of them. Fires on a transition of level
   k+1 when the relation holds / the terms are all present; precision and recall over that level's transitions (one positive:
   the completing action, judged on its settled board).
3. Plan. Where the goal is exact on level k+1 (precision = recall = 1) and the dynamics are good enough (exact >= MIN_EXACT on the
   level), a BFS on the effect-table model from level k+1's start until the goal fires (keys seen in training, one click per object
   whose click key has a non-trivial effect; caps MAX_STATES / MAX_SECONDS); a plan found is replayed on the real engine after the
   library's levels 1..k.
Also: the coarse-grid ECA pipeline of eca_loop_v1 (grid_perception) is tried on each game's first level for the record.

usage (Windows python, cwd C:/arc-agi3): python transfer_measure.py HARNESS_ROOT ECA_LOOP_DIR LIBRARY.json FRAMES.pkl OUT.json [MIN_EXACT]
"""
import collections, json, pickle, sys, time
import numpy as np

B = 4
MIN_EXACT = float(__import__("os").environ.get("TRANSFER_MIN_EXACT", "0.8"))   # or the 6th argument (WSL drops env vars)
MAX_STATES = int(__import__("os").environ.get("TRANSFER_MAX_STATES", "200000"))
MAX_SECONDS = float(__import__("os").environ.get("TRANSFER_MAX_SECONDS", "180"))
BD = None


def inner(g):
    return g[B:-B, B:-B]


def same(a, b):
    return np.array_equal(inner(a), inner(b))


def comps(g):
    return BD.components(g.tolist(), border=B)


def cells_of(c):
    (r, cc), s = c[1], c[2]
    return [(r + a, cc + b) for a, b in s]


def click_point(c):
    cs = cells_of(c)
    rows = [r for r, _ in cs]; cols = [x for _, x in cs]
    cr, cc = (min(rows) + max(rows)) / 2.0, (min(cols) + max(cols)) / 2.0
    return min(cs, key=lambda p: (abs(p[0] - cr) + abs(p[1] - cc), p))


def akey(action, data, before):
    if action == "ACTION6":
        t = BD.click_target(before.tolist(), int(data.get("y", -1)), int(data.get("x", -1)), border=B)
        if t.get("outside") or t.get("edge"):
            return "CLICK edge"
        if t.get("background"):
            return "CLICK background"
        return "CLICK %s %s" % (t["colour"], t["size"])
    return action


def clicked_comp(g, data):
    r, c = int(data.get("y", -1)), int(data.get("x", -1))
    for o in comps(g):
        if (r, c) in set(cells_of(o)):
            return o
    return None


def bbox(c):
    cs = cells_of(c)
    return min(r for r, _ in cs), min(x for _, x in cs), max(r for r, _ in cs), max(x for _, x in cs)


def inside(z, x):
    a, b = bbox(z), bbox(x)
    return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]


def body(o, cs):
    """An object and the smaller objects lying inside its bounding box (dots, holes' contents): moved as one rigid body."""
    return [o] + [z for z in cs if z is not o and len(z[2]) < len(o[2]) and inside(z, o)]


def template(before, after, action, data):
    """The transition's effect as a position-free template, or ('other',)."""
    if same(before, after):
        return ("none",)
    diff = {(int(r) + B, int(c) + B) for r, c in zip(*np.nonzero(inner(before) != inner(after)))}
    cb, ca = comps(before), comps(after)
    hb = [o for o in cb if set(cells_of(o)) & diff]
    ha = [o for o in ca if set(cells_of(o)) & diff]
    # recolour: the same cells, another colour
    if len(hb) == len(ha) and all(any(set(cells_of(x)) == set(cells_of(y)) and x[0] != y[0] for y in ha) for x in hb):
        rel = []
        for x in hb:
            y = next(y for y in ha if set(cells_of(x)) == set(cells_of(y)))
            rel.append((int(x[0]), x[2], int(y[0])))
        if action == "ACTION6":
            cl = clicked_comp(before, data)
            if cl is not None and len(rel) == 1 and set(cells_of(cl)) == set(cells_of(hb[0])):
                return ("click_recolour", rel[0][2])
        return ("recolour", tuple(sorted(rel, key=repr)))
    # moves: rigid bodies (an object and what lies inside its box) reappearing elsewhere, largest first
    moves, enter, leave, used_b, used_a = [], set(), collections.Counter(), set(), set()
    for i, x in sorted(enumerate(hb), key=lambda t: -len(t[1][2])):
        if i in used_b or len(x[2]) > BD.LARGE:
            continue
        best = None
        for j, y in enumerate(ha):
            if j in used_a or y[0] != x[0] or y[2] != x[2] or y[1] == x[1]:
                continue
            d = abs(y[1][0] - x[1][0]) + abs(y[1][1] - x[1][1])
            if best is None or d < best[0]:
                best = (d, j, y)
        if best is None:
            continue
        y = best[2]; dr, dc = y[1][0] - x[1][0], y[1][1] - x[1][1]
        used_b.add(i); used_a.add(best[1])
        own = set()
        for z in body(x, cb):
            own |= set(cells_of(z))
            for k, z2 in enumerate(hb):
                if z2 is z:
                    used_b.add(k)
            for k, w in enumerate(ha):
                if w[0] == z[0] and w[2] == z[2] and w[1] == (z[1][0] + dr, z[1][1] + dc):
                    used_a.add(k)
        new = {(r + dr, c + dc) for r, c in own}
        for p in new - own:
            if 0 <= p[0] < before.shape[0] and 0 <= p[1] < before.shape[1]:
                enter.add(int(before[p]))
        for p in own - new:
            leave[int(after[p])] += 1
        moves.append((int(x[0]), x[2], dr, dc, x[1]))
    if moves:
        p0 = moves[0]   # the largest mover; the others with its displacement are kept by their offset from it (rigid body)
        moves = [(v, sh, dr, dc, (pos[0] - p0[4][0], pos[1] - p0[4][1]) if (dr, dc) == (p0[2], p0[3]) else None)
                 for v, sh, dr, dc, pos in moves]
        floor = enter | set(leave)
        rest = [o for k, o in enumerate(hb) if k not in used_b] + [o for k, o in enumerate(ha) if k not in used_a]
        if all(int(o[0]) in floor or len(o[2]) > BD.LARGE for o in rest):
            return ("move", tuple(moves), frozenset(enter), leave.most_common(1)[0][0] if leave else None)
    return ("other",)


class EffectTable:
    def __init__(self):
        self.seen = collections.defaultdict(collections.Counter)
        self.enter = collections.defaultdict(set)
        self.leave = collections.defaultdict(collections.Counter)

    def add(self, key, t):
        if key.startswith("CLICK ") and key.count(" ") >= 2:   # also under the clicked colour alone (sizes differ per level)
            self.add(" ".join(key.split(" ")[:2]), t)
        if t[0] == "move":
            k = ("move", t[1])
            self.enter[(key, t[1])] |= set(t[2])
            if t[3] is not None:
                self.leave[(key, t[1])][t[3]] += 1
        else:
            k = t
        self.seen[key][k] += 1

    def resolve(self, key):
        if key in self.seen or not key.startswith("CLICK "):
            return key
        return " ".join(key.split(" ")[:2])

    def rule(self, key):
        key = self.resolve(key)
        c = self.seen.get(key)
        if not c:
            return None
        tot = sum(c.values())
        nontriv = [(k, n) for k, n in c.most_common() if k[0] not in ("none", "other")]
        if nontriv and (nontriv[0][1] >= 2 or nontriv[0][1] >= 0.3 * tot):
            return nontriv[0][0]
        return ("none",)

    def predict(self, key, before, data):
        key = self.resolve(key)
        r = self.rule(key)
        if r is None or r[0] == "none":
            return before.copy()
        out = before.copy()
        if r[0] == "click_recolour":
            cl = clicked_comp(before, data)
            if cl is None:
                return out
            for p in cells_of(cl):
                out[p] = r[1]
            return out
        cs = comps(before)
        if r[0] == "recolour":
            for v, s, v2 in r[1]:
                for o in cs:
                    if o[0] == v and o[2] == s:
                        for p in cells_of(o):
                            out[p] = v2
            return out
        if r[0] == "move":
            enter = self.enter[(key, r[1])]
            leave = self.leave[(key, r[1])].most_common(1)
            h, w = before.shape
            plan, anchor = [], None
            for i, (v, s, dr, dc, rel) in enumerate(r[1]):
                objs = []
                if rel is not None and anchor is not None:
                    objs = [o for o in cs if o[0] == v and o[2] == s and o[1] == (anchor[0] + rel[0], anchor[1] + rel[1])]
                if not objs:
                    objs = [o for o in cs if o[0] == v and o[2] == s]
                if len(objs) != 1:   # another level: the object of that colour, when there is one (shapes differ per level)
                    objs = [o for o in cs if o[0] == v and len(o[2]) <= BD.LARGE]
                if len(objs) != 1:
                    return before.copy()
                if i == 0:
                    anchor = objs[0][1]
                own = set()
                for z in body(objs[0], cs):
                    own |= set(cells_of(z))
                new = {(y + dr, x + dc) for y, x in own}
                if any(not (B <= y < h - B and B <= x < w - B) for y, x in new):
                    continue
                if any(int(before[p]) not in enter for p in new - own):
                    continue                                   # blocked: a colour it was never seen to enter
                plan.append((own, dr, dc))
            allown = set().union(*[o for o, _, _ in plan]) if plan else set()
            for own, dr, dc in plan:
                ring = collections.Counter(int(before[y + a, x + b]) for y, x in own for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))
                                           if 0 <= y + a < h and 0 <= x + b < w and (y + a, x + b) not in allown)
                # the colour levels 1..k left behind when it borders the body here, else the colour around the body
                fill = leave[0][0] if (leave and leave[0][0] in ring) else (ring.most_common(1)[0][0] if ring else (
                    leave[0][0] if leave else int(np.bincount(before.ravel().astype(np.int64) & 15).argmax())))
                for p in own:
                    out[p] = fill
            for own, dr, dc in plan:
                for y, x in own:
                    out[y + dr, x + dc] = before[y, x]
            return out
        return out


def sig_terms(before, after):
    lines, _ = BD.object_changes(before.tolist(), after.tolist(), border=B)
    s = BD.effect_signature(lines)
    return set(t for t in s.split("; ") if t and t != "no change inside the board")


def filled(before, after):
    return bool(BD.completion_facts(before.tolist(), after.tolist(), border=B)[1])


def transitions(level):
    prev = level["start"]
    for st in level["steps"]:
        yield prev, st
        prev = st["after"]


def coarse_check(levels, gp):
    fr = [levels[0]["start"]] + [st["after"] for st in levels[0]["steps"][:4]]
    flat = [f.astype(int).ravel().tolist() for f in fr]
    g = gp.detect_grid_combined(flat)
    pairs = [(a.astype(int).ravel().tolist(), st["after"].astype(int).ravel().tolist())
             for a, st in transitions(levels[0]) if st["action"] in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")][:30]
    m = gp.detect_grid_motion(pairs) if pairs else None
    return {"combined": g, "motion": m}


def plan_bfs(table, start, goal, keys, deadline):
    from collections import deque
    prev = {inner(start).tobytes(): None}
    boards = {inner(start).tobytes(): start}
    q = deque([start]); n = 0
    while q and len(prev) < MAX_STATES and time.monotonic() < deadline:
        s = q.popleft(); sk = inner(s).tobytes()
        acts = [(k, k, {}) for k in keys]
        for o in comps(s):
            if len(o[2]) > BD.LARGE:
                continue
            r, c = click_point(o)
            d = {"x": int(c), "y": int(r)}
            k = akey("ACTION6", d, s)
            rr = table.rule(k)
            if rr is not None and rr[0] != "none":
                acts.append(("ACTION6", k, d))
        for a, k, d in acts:
            ns = table.predict(k, s, d)
            if same(ns, s):
                continue
            nk = inner(ns).tobytes()
            if goal(s, ns):
                out = [(a, d)]; cur = sk
                while prev[cur]:
                    cur, a2 = prev[cur]; out.append(a2)
                return out[::-1], len(prev)
            if nk not in prev:
                prev[nk] = (sk, (a, d)); q.append(ns)
    return None, len(prev)


def replay_plan(gid, lib_entries, k, plan):
    import arc_agi
    from arcengine import GameAction
    env = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE).make(gid.split("-")[0])
    obs = env.reset()
    for e in lib_entries[:k]:
        for a, d in e["actions"]:
            r = env.step(getattr(GameAction, a), data=d or {}); obs = r if r is not None else obs
    lv0 = obs.levels_completed
    for i, (a, d) in enumerate(plan):
        r = env.step(getattr(GameAction, a), data=d or {}); obs = r if r is not None else obs
        if obs.levels_completed > lv0:
            return {"completed": True, "at": i + 1}
    return {"completed": False}


def main(root, eca_dir, lib_path, frames_path, out_path):
    global BD
    sys.path.insert(0, root)
    sys.path.append(eca_dir)
    from inference.utils import board_digest as _bd
    BD = _bd
    import grid_perception as gp
    lib = json.load(open(lib_path))["games"]
    data = pickle.load(open(frames_path, "rb"))
    result = {}
    for gid, levels in sorted(data.items()):
        g = {"coarse_grid": None, "pairs": []}
        try:
            g["coarse_grid"] = coarse_check(levels, gp)
        except Exception as exc:
            g["coarse_grid"] = {"error": repr(exc)}
        for k in range(1, len(levels)):
            table = EffectTable()
            comp_terms, comp_filled = [], []
            for lv in levels[:k]:
                for before, st in transitions(lv):
                    if st["completed"]:
                        settled = st["settled"] if st["settled"] is not None else st["after"]
                        comp_terms.append(sig_terms(before, settled)); comp_filled.append(filled(before, settled))
                        continue
                    table.add(akey(st["action"], st["data"], before), template(before, st["after"], st["action"], st["data"]))
            tgt = levels[k]
            n = ex_m = ex_0 = wk_m = wk_0 = unseen = 0
            why = collections.Counter()
            for before, st in transitions(tgt):
                if st["completed"]:
                    continue
                n += 1
                key = akey(st["action"], st["data"], before)
                if table.rule(key) is None:
                    unseen += 1
                pred = table.predict(key, before, st["data"])
                real = st["after"]
                ex_m += same(pred, real); ex_0 += same(before, real)
                # why: the real effect outside the model class / key unseen / another template than levels 1..k / applied wrong
                if same(pred, real):
                    why["exact"] += 1
                else:
                    t = template(before, real, st["action"], st["data"])
                    r = table.rule(key)
                    tk = ("move", t[1]) if t[0] == "move" else t
                    if t[0] == "other":
                        why["effect outside the model class"] += 1
                    elif r is None:
                        why["action key unseen"] += 1
                    elif r != tk:
                        why["other template than levels 1..k"] += 1
                    else:
                        why["same template, applied wrong"] += 1
                rs = sig_terms(before, real)
                wk_m += sig_terms(before, pred) == rs; wk_0 += (not rs)
            # goal
            if all(comp_filled):
                kind = "filled relation"
                goal = lambda b, a: filled(b, a)
            else:
                common = set.intersection(*comp_terms) if comp_terms else set()
                kind = ("terms: " + "; ".join(sorted(common))) if common else "none"
                goal = (lambda b, a, c=frozenset(common): bool(c) and c <= sig_terms(b, a))
            tp = fp = 0
            for before, st in transitions(tgt):
                after = st["settled"] if (st["completed"] and st["settled"] is not None) else st["after"]
                fire = goal(before, after)
                if st["completed"]:
                    tp += fire
                else:
                    fp += fire
            prec = tp / (tp + fp) if (tp + fp) else None
            rec = float(tp)
            row = {"k": k, "next_level": k + 1, "transitions": n, "unseen_keys": unseen,
                   "exact_model": round(ex_m / n, 3) if n else None, "exact_nothing": round(ex_0 / n, 3) if n else None,
                   "object_model": round(wk_m / n, 3) if n else None, "object_nothing": round(wk_0 / n, 3) if n else None,
                   "why": dict(why), "goal": kind, "goal_precision": None if prec is None else round(prec, 3), "goal_recall": rec,
                   "library_actions": len(tgt["steps"]), "human_actions": lib[gid][k]["human_actions"], "plan": None}
            if rec == 1.0 and prec == 1.0 and n and ex_m / n >= MIN_EXACT:
                keys = sorted({kk for kk in table.seen if not kk.startswith("CLICK")})
                t0 = time.monotonic()
                plan, states = plan_bfs(table, tgt["start"], goal, keys, t0 + MAX_SECONDS)
                row["plan"] = {"found": plan is not None, "states": states, "seconds": round(time.monotonic() - t0, 1)}
                if plan is not None:
                    row["plan"].update(length=len(plan), real=replay_plan(gid, lib[gid], k, plan))
            g["pairs"].append(row)
            print(gid[:4], "L%d->L%d" % (k, k + 1), json.dumps({x: row[x] for x in ("transitions", "unseen_keys", "exact_model",
                  "exact_nothing", "object_model", "object_nothing", "why", "goal", "goal_precision", "goal_recall", "plan")}), flush=True)
        result[gid] = g
    json.dump(result, open(out_path, "w"), indent=1, default=str)
    print("TRANSFER_WRITTEN", out_path)


if __name__ == "__main__":
    if len(sys.argv) > 6:
        MIN_EXACT = float(sys.argv[6])
    main(*sys.argv[1:6])
