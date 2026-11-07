"""Figures for the paper (docs/paper/). Windows python (matplotlib), reads the run directories on E: and the F research files.

fig2_library.png  - levels and generated tokens per draw: control (ctl1, ctl2) vs library (l1-l4)
fig3_advice.png   - the 13 solver-owned levels: who played each (routine / LLM completed / LLM open) and the LLM tokens on them,
                    authority (a1-a4) vs advice (v1, v2); open levels count their tokens to the end of the run
fig4_handback.png - divergence test: levels completed and model actions from the hand-back, R vs S

usage: python paper_figures.py OUTDIR
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = "E:/arc-agi3-online-specialists-runs-e"
F = "F:/arc-agi3-online-specialists-20261014"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def bench(d):
    return json.load(open(f"{E}/{d}/out_canon/benchmark.json"))


def solver_tokens(b):
    return sum(int(str(g["solver_note"]).split("tokens=")[1].split()[0].rstrip(",")) for g in b["game_runs"]
               if g.get("solver_note") and "tokens=" in str(g["solver_note"]))


def bars(ax, labels, values, colors, fmt, ylabel, title):
    x = range(len(labels))
    ax.bar(x, values, color=colors, width=0.62, edgecolor="white", linewidth=2)
    for i, v in zip(x, values):
        ax.text(i, v, fmt % v, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def fig2(out):
    runs = [("ctl1", "explain2_runs/ctl1"), ("ctl2", "explain2_runs/ctl2")] + [("l%d" % i, "library_runs/l%d" % i) for i in (1, 2, 3, 4)]
    bs = [bench(d) for _, d in runs]
    lv = [sum(g["levels_completed"] for g in b["game_runs"]) for b in bs]
    tok = [solver_tokens(b) / 1e6 for b in bs]
    col = [ORANGE] * 2 + [BLUE] * 4
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    bars(axs[0], [r for r, _ in runs], lv, col, "%d", "levels completed (of 59)", "Levels per draw")
    bars(axs[1], [r for r, _ in runs], tok, col, "%.2f", "generated tokens (M)", "LLM tokens per draw")
    axs[0].set_ylim(0, 59)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=ORANGE, label="control (no library)"), Patch(color=BLUE, label="library on")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(out, dpi=220); plt.close(fig)
    return dict(levels=lv, tokens_M=[round(t, 3) for t in tok])


OWNED = {"ar25": [1, 2, 3], "cn04": [1, 2], "re86": [1], "s5i5": [1, 2], "tn36": [1, 2, 3, 4, 5]}


def owned_split(b):
    """Per owned level: played by the routine (no action carried LLM tokens) or by the LLM; tokens; completed or open."""
    games = {g["game_id"][:4]: g for g in b["game_runs"]}
    out = []
    for k, ls in OWNED.items():
        g, start = games[k], 0
        for L in range(1, max(ls) + 1):
            done = g["levels_completed"] >= L
            n = g["actions_per_level"][L - 1]
            seg = g["history"][start:start + n] if done else g["history"][start:]
            tok = [int(x.get("generated_tokens") or 0) for x in seg]
            out.append(dict(level="%s L%d" % (k, L), routine=not any(tok), tokens=sum(tok), done=done))
            if not done:
                break
            start += n
    return out


def fig3(out):
    ds = ["a1", "a2", "a3", "a4", "v1", "v2"]
    sp = {d: owned_split(bench("advice_runs/" + d)) for d in ds}
    rout = [sum(1 for x in sp[d] if x["routine"]) for d in ds]
    llm_done = [sum(1 for x in sp[d] if not x["routine"] and x["done"]) for d in ds]
    llm_open = [sum(1 for x in sp[d] if not x["routine"] and not x["done"]) for d in ds]
    tok_done = [sum(x["tokens"] for x in sp[d] if not x["routine"] and x["done"]) / 1e3 for d in ds]
    tok_open = [sum(x["tokens"] for x in sp[d] if not x["routine"] and not x["done"]) / 1e3 for d in ds]
    tok_rout = [sum(x["tokens"] for x in sp[d] if x["routine"]) for d in ds]
    LIGHT = "#f5b89d"
    x = range(len(ds))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axs[0]
    ax.bar(x, rout, color=BLUE, width=0.62, edgecolor="white", linewidth=1.5)
    ax.bar(x, llm_done, bottom=rout, color=ORANGE, width=0.62, edgecolor="white", linewidth=1.5)
    ax.bar(x, llm_open, bottom=[r + l for r, l in zip(rout, llm_done)], color=LIGHT, hatch="///", width=0.62,
           edgecolor="white", linewidth=1.5)
    for i in x:
        ax.text(i, rout[i] / 2, str(rout[i]) if rout[i] else "", ha="center", va="center", fontsize=8, color="white")
    ax.set_xticks(list(x)); ax.set_xticklabels(ds); ax.set_ylim(0, 13.5); ax.set_yticks([0, 4, 8, 13])
    ax.set_ylabel("solver-assigned levels (13)"); ax.set_title("Who played each assigned level", fontsize=9.5, color=INK, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax = axs[1]
    ax.bar(x, tok_done, color=ORANGE, width=0.62, edgecolor="white", linewidth=1.5)
    ax.bar(x, tok_open, bottom=tok_done, color=LIGHT, hatch="///", width=0.62, edgecolor="white", linewidth=1.5)
    for i in x:
        t = tok_done[i] + tok_open[i]
        ax.text(i, t, "%.0f k" % t, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels(ds)
    ax.set_ylabel("generated tokens (thousands)"); ax.set_title("LLM tokens on those levels", fontsize=9.5, color=INK, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label="routine played it (0 tokens)"),
                        Patch(color=ORANGE, label="LLM played it, completed"),
                        Patch(facecolor=LIGHT, hatch="///", edgecolor="white", label="LLM played it, not completed")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(out, dpi=220); plt.close(fig)
    return dict(routine=rout, llm_done=llm_done, llm_open=llm_open, tokens_k_done=[round(t, 1) for t in tok_done],
                tokens_k_open=[round(t, 1) for t in tok_open], routine_tokens=tok_rout,
                llm_levels={d: [(x["level"], x["tokens"], x["done"]) for x in sp[d] if not x["routine"]] for d in ds[:4]})


def fig4(out):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    bars(axs[0], ["R: replay, then hand-back", "S: from scratch"], [11, 7], [BLUE, ORANGE], "%d / 16", "levels completed (of 16)",
         "Planted divergence: level completed")
    axs[0].set_ylim(0, 16)
    bars(axs[1], ["R", "S"], [41, 117], [BLUE, ORANGE], "%d", "model actions (median)", "Model actions from the hand-back (median)")
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


if __name__ == "__main__":
    o = sys.argv[1]; os.makedirs(o, exist_ok=True)
    print(json.dumps(dict(fig2=fig2(os.path.join(o, "fig2_library.png")), fig3=fig3(os.path.join(o, "fig3_advice.png"))), indent=1))
    fig4(os.path.join(o, "fig4_handback.png")); print("FIGURES_OK")
