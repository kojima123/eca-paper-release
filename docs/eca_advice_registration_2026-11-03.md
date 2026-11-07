# Authority vs advice, and same-configuration pass@k (registered 2026-11-03 ~08:40Z, before any draw)

User 11-03 「やって」 (experiments 2 and 3 of 3) and 「追加した」 (Colab units added after G4 was refused from 06:37Z).
Paper axis "the harness governs the LLM" (outline v2): Tycho's validated routes are advice its frontier actor follows 51-86 %; here
the same comparison with the small competition model.

## Arms (Colab G4, 8 games, 60 min; tools/build_advice_colab.py; harness branch eca-advice a3c0239 = ecaod-changelog 900261c +
ECA_FRZ_ECA_ADVICE, default off; patch 7833624b3149bfb7)
- **auth** (franzen-m2-auth-colab-20261103, ipynb a620e0c2fdc326b5): the control configuration - environment identical to the
  explain2 control kernel of ctl1 / ctl2 (checked: same ECA_* keys and values) on the current tree; certified plans are played by the
  prefix executor (authority). 4 draws a1-a4.
- **advice** (franzen-m2-advice-colab-20261103, 0384499f2ec4e68e): the same + ECA_FRZ_ECA_ADVICE=1: whenever the executor would play,
  the certified plan goes to the model's next opener as one line ("Solver advice: ... From the current board its plan is: ... The
  harness does not play it; whether to follow it is your choice.") and the harness plays nothing. 2 draws v1-v2.
Local checks: advice on - 0 solver actions, the line once per offer; off - unchanged; a scripted follower is re-offered after every
step (tn36 L1 in 7 actions).

## Experiment 2 measures (vs auth a1, a2 - the first two auth draws, run alongside v1, v2)
Offers per draw ("eca advice:" log lines); **follow rate** = offers whose next executed model action equals the plan's first action;
levels the solver owns in the authority arm (ar25 L1-3, cn04 L1-2, re86 L1, s5i5 L1-2, tn36 L1-5: 15 levels per draw) completed and
the actions / tokens spent on them; total levels, mean game score, tokens per draw.
Prediction: follow rate < 0.8; advice completes fewer of the solver's levels (mostly tn36 L1-5) and spends tokens on them; total levels
lower. Reading: auth - advice >= 3 solver levels per draw (pooled 2 v 2) = holding the authority pays with this model; <= 2 = advice
is as good (the model follows); advice > auth by >= 3 = the model improves on the plans.

## Experiment 3 measures (a1-a4, same kernel)
Per level: draws that cleared it (of 4); mean levels per draw (pass@1-like) vs levels cleared in at least one draw (pass@4) and in all
draws (pass^4); the same with ctl1 / ctl2 added (6 draws; same environment, older tree - reported separately).
Prediction: union over 4 >= mean + 4 levels; >= 6 levels cleared in some but not all draws.
Never Kaggle; nothing submitted or adopted without the user's word.

## Result, experiment 2 (a1, a2 vs v1, v2; 11-03 ~14:00Z; tools/advice_analysis.py -> research/eca_advice/advice_interim.json)
| draw | arm | levels | solver-owned levels completed | actions / generated tokens / game-minutes until the owned levels were done (5 games summed) | mean game score | tokens per draw |
|---|---|---|---|---|---|---|
| a1 | authority | 37 | 13 / 13 | 222 / 9,081 / 10.3 | 45.90 | 1.79 M |
| a2 | authority | 37 | 13 / 13 | 223 / 4,593 / 9.7 | 50.18 | 1.72 M |
| v1 | advice | 35 | 13 / 13 | 275 / 235,303 / 90.2 | 42.92 | 1.58 M |
| v2 | advice | 31 | 13 / 13 | 263 / 251,976 / 71.1 | 35.69 | 1.64 M |
| ctl1 / ctl2 (control, older tree) | authority | 31 / 38 | 13 / 13 | 221 / 0 / 4.8; 221 / 0 / 6.5 | 34.89 / 50.74 | 1.78 / 1.73 M |
(The registered list of owned levels - ar25 L1-3, cn04 L1-2, re86 L1, s5i5 L1-2, tn36 L1-5 - sums to 13, not the 15 stated above.)
Offers and follow rate (the next executed action equals the plan's first action): v1 44 offers, 32 followed (0.727); v2 69, 59
(0.855); per game v1 / v2: tn36 15/16, 38/40; ar25 9/11, 4/6; re86 4/9, 8/10; cn04 2/3, 7/8; s5i5 2/2, 1/2; ls20 0/3, 1/3.
Predictions: follow rate < 0.8 - met in v1, not in v2; advice completes fewer owned levels - **not met** (13 / 13 in every draw);
total levels lower - 33 vs 37 mean, inside the draw spread (control 31 vs 38). Registered reading: auth - advice = 0 owned levels
(<= 2) = **advice is as good on the owned levels: this model follows the plans** (unlike the July Qwen3.6-27B ladder, advice 0 / 9).
What differs is the cost of the same levels: with advice the model takes 241 k generated tokens (25-55x) and 80 game-minutes (about
8x) per draw for what the executor does in 10 minutes at 0-9 k tokens - the knowledge is the same, the authority decides whether
it is used without deliberation. Experiment 3 (pass@k over a1-a4) waits for a3 / a4.

## Result, experiment 3 (a1-a4 same kernel; 11-03 ~14:50Z; research/eca_advice/advice_final.json)
Levels per draw a1-a4: 37, 37, 33, 36 (mean 35.75). Cleared in at least one draw (pass@4) 42; in all four (pass^4) 30; in some but
not all 12 (bp35 L2, cn04 L5-L6, dc22 L4, ls20 L4, re86 L6, s5i5 L2-L5, tn36 L6-L7). Predictions: union >= mean + 4 - met (42 vs
39.75); >= 6 levels cleared in some but not all - met (12). With ctl1 / ctl2 (6 draws, older tree): 37, 37, 33, 36, 31, 38, mean
35.3, union 42, all six 26, some but not all 16.
Effect on experiment 2's cost reading: a3 and a4 are the same authority configuration. Their owned levels cost more because the
solver did not certify every owned level in time: per draw the solver played 12, 12, 10, 11 of the 13 owned levels with no LLM turn
(45 of 52); the LLM played the rest (cn04 L1 in a1, a3, a4 at 5-6 turns; ar25 L1 in a2 at 4; tn36 L2 in a3 at 8; s5i5 L2 in a3 and a4
at 20 and 17 turns, and a3 failed it - owned completed 12 / 13). Over a1-a4 the owned levels took 4 / 4 / 34 / 22 LLM turns,
4.6-139.6 k tokens and 9.7-38.8 game-minutes; under advice every owned level-run took turns (76 / 85 per draw), 235-252 k tokens,
71-90 game-minutes. The registered reading of experiment 2 (a1, a2 vs v1, v2) stands; the cost gap holds over all four authority
draws, smaller in a3 / a4 because the LLM played some owned levels there too.
