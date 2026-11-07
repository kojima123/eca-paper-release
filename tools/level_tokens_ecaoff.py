"""Generated tokens per level from the request logs (<game>_p0_requests.jsonl).

Each analyzer turn's first request carries "Current state: step N, level L" in its last user message; the response rows carry
usage.completion_tokens. Per run and game: the level of the model's first turn, the tokens it generated on each level, and the
total (compare benchmark.json's tokens=N). With a reference arm (REF=...), the levels below the reference's first model level are
the levels the solver played there; for the other arms the tool prints the tokens the model spent on those levels and whether it
got past them.

usage: python3 tools/level_tokens_ecaoff.py REF=C ARM=dir,dir ARM=dir,dir ...
"""
import glob, json, os, re, sys

PAT = re.compile(r"Current state: step (\d+), level (\d+)")


def text(m):
    c = m.get("content")
    if isinstance(c, str):
        return c
    return " ".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))


REQ = {}


def game_levels(path):
    level, first, per = None, None, {}
    stats = REQ.setdefault(os.path.dirname(path), [0, 0, 0])
    for line in open(path, errors="replace"):
        r = json.loads(line)
        if r.get("event") == "request":
            if str(r.get("request_index_within_turn")) == "1":
                s = PAT.findall(text(r["messages"][-1]))
                if s:
                    level = int(s[-1][1])
                    if first is None:
                        first = level
        elif r.get("event") == "response":
            u = r.get("usage") or {}
            stats[0] += 1
            stats[1] += int(u.get("prompt_tokens") or 0)
            stats[2] += int(u.get("completion_tokens") or 0)
            if level is not None:
                per[level] = per.get(level, 0) + int(u.get("completion_tokens") or 0)
    return first, per


def run(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*_p0_requests.jsonl"))):
        out[os.path.basename(f)[:4]] = game_levels(f)
    return out


def main(argv):
    ref = None
    arms = []
    for a in argv:
        k, v = a.split("=", 1)
        if k == "REF":
            ref = v
        else:
            arms.append((k, [run(d) for d in v.split(",")], v.split(",")))
    starts = {}
    for k, runs, _ in arms:
        if k == ref:
            for r in runs:
                for g, (first, _) in r.items():
                    if first:
                        starts[g] = min(starts.get(g, first), first)
    if ref:
        print("levels the solver played in %s (below its first model level): %s" % (
            ref, " ".join("%s L1-%d" % (g, s - 1) for g, s in sorted(starts.items()) if s > 1)))
    pooled = {}
    for k, runs, dirs in arms:
        for r, d in zip(runs, dirs):
            row, tot, sol = [], 0, 0
            for g, (first, per) in sorted(r.items()):
                tot += sum(per.values())
                s = starts.get(g, 1)
                spent = sum(t for lv, t in per.items() if lv < s)
                past = max(per) >= s if per else False
                if s > 1 and k != ref:
                    sol += spent
                    row.append("%s %dk%s" % (g, round(spent / 1000), "" if past else " (not past L%d)" % (s - 1)))
            name = os.path.basename(os.path.dirname(d.rstrip("/")))
            n, pt, ct = REQ.get(d.rstrip("/"), [0, 0, 0])
            print("%-4s %-6s total %.2f M, %d requests, prompt %d / generated %d tokens per request%s" % (
                k, name, tot / 1e6, n, pt // max(n, 1), ct // max(n, 1), (" | on the solver's levels %.2f M: " % (sol / 1e6) + "  ".join(row)) if row else ""))
            pooled.setdefault(k, []).append(sol)
    for k, v in pooled.items():
        if k != ref and starts:
            print("%s: tokens on the solver's levels per draw %.2f M" % (k, sum(v) / len(v) / 1e6))


if __name__ == "__main__":
    main(sys.argv[1:])
