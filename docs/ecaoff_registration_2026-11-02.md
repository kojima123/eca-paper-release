# ECA off: what the solver and the model-side facts contribute (registered 2026-11-02 ~14:20Z, before any draw)

User 11-02 (for the Paper Track): 「論文ネタで colabで対象実験したいです。ECA offでソルバーが動作しないときも測定したい」, scale A×2 + B×2,
「トークン数も測って下さい」.

## Arms (same 8 games ar25 bp35 cn04 dc22 ls20 re86 s5i5 tn36, Colab G4, 60 min per game, 8 concurrent, same model and server)

| arm | ECA (solver) | model-side facts (DIGEST, WATCH, CHANGELOG, pinned levels, host note) | failure note / end facts | draws |
|---|---|---|---|---|
| A Franzen | off | off | off | a1, a2 (new) |
| B solver off | off | on | off | b1, b2 (new) |
| C control | on | on | off | explain2 ctl1, ctl2 (existing, 10-29/30) |
| D v8 | on | on | on (v8) | failnote f15, f16 (existing, 11-02) |

A and B: tools/build_ecaoff_colab.py on the change-record canon kernel (the base of C) with the patch bdb3c20..ecaod-changelog
342e2df (every arm flag default off; sha 81508fc27c1469a8, 724869 bytes): B sets ECA_FRZ_PORT=0, A also DIGEST / WATCH / CHANGELOG / PIN_LEVELS /
HOST_NOTE = 0; cell 17's checks inverted (port must be off; B requires the facts on, A off; no failure note / end facts).
Kernels franzen-m2-ecaoff-a-colab-20261102 (8abc179979aa234a), franzen-m2-ecaoff-b-colab-20261102 (ccb7081bfb4ff725). Infrastructure keys as in every
draw (keep-alive RESET 600 s, no give-up before ready, warmup game). Queue E:/ecaoff_runs/queue_ecaoff.sh: CPU dry runs of B and A
with their gates, then b1, b2, a1, a2 (B first).

## Measures (per arm, per draw and per game)
levels at 60 min, final_score (the competition per-game score), generated tokens per game (benchmark.json solver_note), tokens per
cleared level; for C and D also the levels the solver completed (host notes) and the model's tokens on them.

## Predictions
1. B < C in levels (the solver plays tn36 L1-5, ar25 L1-3, cn04 L1-2, re86 L1-4, s5i5 L1-2 in C/D; without it the model has to find
   them), and B spends more tokens before its first levels.
2. A vs B: no prediction on levels (the model-side facts were NULL to mildly positive arm by arm); A spends fewer tokens per request.
3. The comparisons are 2 vs 2 draws per arm; the per-draw spread of the control (31 vs 38 levels) means only a difference of about
   7 levels or more is readable; smaller differences are reported as inside the spread.
Never Kaggle; nothing here is submitted.

## Result, arm B (b1, b2 done 11-02 15:25Z; arm A a1/a2 still to run)

tools/summarize_ecaoff.py (levels at 60 min, mean final_score over the 8 games, generated tokens, tokens per cleared level, solver-completed levels):

| arm | draws | levels | mean score | tokens (M) | tokens / level | solver levels |
|---|---|---|---|---|---|---|
| B solver off | b1 29, b2 32 | 30.5 | 38.19 | 1.80 | 58980 | 0 |
| C control | ctl1 31, ctl2 38 | 34.5 | 42.81 | 1.82 | 52612 | 16.0 |
| D v8 | f15 35, f16 33 | 34.0 | 42.97 | 1.79 | 52645 | 15.5 |

Per game (B b1/b2 vs C ctl1/ctl2): ar25 8/8 vs 6/8, bp35 2/1 vs 1/2, cn04 4/6 vs 2/5, dc22 4/4 vs 4/4, ls20 2/4 vs 4/4,
re86 5/6 vs 5/5, s5i5 3/2 vs 3/3, tn36 1/1 vs 6/7.

- Prediction 1 (B < C in levels): by direction yes (30.5 vs 34.5 per draw), but the 4-level gap is inside the registered
  ~7-level spread. The whole gap is tn36 (2 vs 13 levels over two draws); without tn36 B has 59 levels against C's 56.
  The model finds by itself the early levels the solver plays in C: ar25 L1 in 4/6 turns, cn04 L1 in 5/6, re86 L1 in 6/5,
  s5i5 L1 in 7/7 (tools/first_level_ecaoff.py, model turns before the first level-up); tn36 L1 took 23/30 turns
  (218/329 actions) and neither draw got past L2.
- Tokens: the 60-minute cap binds, so every arm spends ~1.8 M tokens; B spends ~12 % more per cleared level.
  "More tokens before the first levels" was first read as turns above; the request log's response rows do carry
  usage.completion_tokens, so the final section measures it in tokens (correction of an earlier wording here).

## Final result, all four arms (arm A a1, a2 done 11-02 18:48Z; a1's first launch failed on a model-file download before any game and relaunched)

| arm | draws | levels / draw | mean score | tokens (M) | tokens / level | solver levels | prompt / generated tokens per request |
|---|---|---|---|---|---|---|---|
| A Franzen | a1 36, a2 29 | 32.5 | 37.07 | 1.84 | 56769 | 0 | 74.7k / 2.13k |
| B solver off | b1 29, b2 32 | 30.5 | 38.19 | 1.80 | 58980 | 0 | 75.2k / 2.08k |
| C control | ctl1 31, ctl2 38 | 34.5 | 42.81 | 1.82 | 52612 | 16.0 | 75.0k / 2.11k |
| D v8 | f15 35, f16 33 | 34.0 | 42.97 | 1.79 | 52645 | 15.5 | 74.8k / 2.08k |

Per game, levels per draw (A | B | C | D): ar25 8,8 | 8,8 | 6,8 | 8,6; bp35 2,1 | 2,1 | 1,2 | 2,2; cn04 4,2 | 4,6 | 2,5 | 2,5;
dc22 3,4 | 4,4 | 4,4 | 4,4; ls20 4,4 | 2,4 | 4,4 | 4,2; re86 6,5 | 5,6 | 5,5 | 6,6; s5i5 5,2 | 3,2 | 3,3 | 2,2; tn36 4,3 | 1,1 | 6,7 | 7,6.

tools/level_tokens_ecaoff.py (generated tokens per level from the request logs; totals agree with benchmark.json): in C the solver
plays ar25 L1-3, cn04 L1-2, re86 L1, s5i5 L1-2, tn36 L1-5. On those levels the model alone spends 0.51 M tokens per draw in A and
0.36 M in B (28 % / 20 % of the 1.8 M budget), of which tn36 takes 216-248 k per draw, i.e. its whole budget, and no A or B draw gets
past tn36 L5; the other four games' solver levels cost 6-253 k (ar25 25-68 k, re86 6-8 k, s5i5 41-61 k, cn04 44-253 k).

- Prediction 1 (B < C): by direction, 30.5 vs 34.5 per draw, inside the registered ~7-level spread. Readable at the game level only:
  tn36 2 levels (B) and 7 (A) over two draws against 13 (C) and 13 (D).
- Prediction 2 (A vs B no level prediction; A fewer tokens per request): A 32.5 vs B 30.5, inside the spread; prompt tokens per
  request are the same in all arms (74.7-75.2 k, dominated by the history and the image), so the model-side facts do not make the
  requests measurably larger. tn36 A 7 vs B 2 levels is the largest per-game difference between them (two draws each).
- Prediction 3 holds: every arm-level difference is inside the control's own spread (31 vs 38).
- Reading for the paper: the solver completes 16 of 34.5 levels per draw, but removing it costs about 2-4 levels per draw, because the
  model finds most of those levels itself (ar25, cn04, re86, s5i5) at 20-28 % of its token budget; the one game the solver decides is
  tn36 (L1-5, a program-execution family), where the model alone uses its whole budget and stays below L6.

### Where the solver pays (follow-up 11-02 ~20:20Z, user 「ソルバーの長所ないの？」)
- Score by game, sum of the 8 per draw: A 335 / 259, B 262 / 350, C 280 / 406, D 357 / 329. Without tn36: A 299 / 241, B 262 / 350,
  C 207 / 306, D 259 / 254 - the arms without the solver are level with or above it; the solver's whole score advantage is tn36
  (C 73 / 100, D 98 / 75 against A 36 / 18, B 0 / 0).
- Actions to get past the solver's levels (tools/solver_level_actions_ecaoff.py, step of the first model turn at the level C's model
  starts on; A | B | C): ar25 L1-3 86,72 | 79,104 | 82,82; cn04 L1-2 52,107 | 65,70 | 44,44; re86 L1 33,24 | 22,22 | 21; s5i5 L1-2
  70,80 | 54,58 | 40,40; tn36 L1-5 never | never | 63,63 (D 95, 54). The solver is the same every draw, as short or shorter
  (cn04, s5i5 clearly), and spends no tokens there.
- The time and tokens it frees (0.36-0.51 M per draw) do not become later levels: after the solver, C's model clears no more of the
  later levels than A or B reach on their own.

### On the levels it was built on (follow-up 11-02 ~20:40Z, user 「逆に、既知のタスクには強い？」)
tools/level_efficiency_ecaoff.py (benchmark.json actions_per_level against base_actions_per_level, per-level s = min(1.15, (h/a)^2)):
- On its own levels (ar25 L1-3, cn04 L1-2, re86 L1, s5i5 L1-2, tn36 L1-5) the solver clears every level in every C/D draw at the cap
  (mean s 1.15 / 1.12; D f15's tn36 L5 was the model's after the solver stopped), with the same short action counts each time
  (tn36 L1 7 vs human 32, ar25 L2 11 vs 50, s5i5 L1 13 vs 20).
- The model alone clears 23 (A) / 18 (B) of the same 26 level-draws, also mostly at the cap (mean s 0.97 / 0.96): the human baselines
  leave room, so the solver's shorter play earns little extra except where the model is slow (s5i5 L1 0.28-0.69, cn04 L2 0.36, re86
  L1 0.66, tn36 L2 0.25-0.68) or stuck (tn36 L1-5 in B).
- Beyond those levels (the same games' later levels) the solver stops (ar25 L4, cn04 L3, re86 L2/L5, s5i5 L3, tn36 L6): strength is
  per known level, not per known game. The local 8 games are the solver's own build set, so C - A/B is an upper bound of what it adds.
