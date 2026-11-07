"""Solver coverage per game from eca_port_integration_local.py runs (docs/nv25_solver_coverage_2026-11-03.md): family
evaluations, families detected / certified, actions the solver executed, levels completed.

    python tools/nv25_solver_coverage.py <run dir> [<run dir> ...]    (run dir = the integration harness's output directory)"""
import collections
import glob
import json
import os
import sys


def run(d: str) -> None:
    s = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
    tot = collections.Counter()
    print("== %s (%d games)" % (os.path.basename(d.rstrip("/\\")), len(s["games"])))
    for pre, g in sorted(s["games"].items()):
        tel = glob.glob(os.path.join(d, g["game"], "artifacts", "*_eca_telemetry.jsonl"))
        rows = [json.loads(l) for f in tel for l in open(f, encoding="utf-8")]
        det = sorted({c for r in rows for c in (r.get("detected_certified") or [])})
        cert = sorted({c for r in rows for c in (r.get("certified") or [])})
        auto = max([r.get("auto_exec_count") or 0 for r in rows] or [0])
        print("  %-16s evals %3d | detected %-30s | certified %-30s | solver actions %3d | levels %d%s" % (
            g["game"], len(rows), ",".join(det) or "-", ",".join(cert) or "-", auto, g["levels"],
            " | ERROR %s" % g["error"] if g.get("error") else ""))
        tot["games"] += 1; tot["evals"] += len(rows); tot["levels"] += g["levels"]; tot["actions"] += auto
        tot["games_certified"] += bool(cert); tot["games_detected"] += bool(det); tot["games_with_levels"] += bool(g["levels"])
        tot["errors"] += bool(g.get("error"))
    print("  total: " + ", ".join("%s %d" % kv for kv in tot.items()))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        run(d)
