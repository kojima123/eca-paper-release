# Closure census and apartness repair on recorded play (registered 2026-11-03 ~08:20Z, before the tool ran on any data)

User 11-03 「補強する実験はある？」 -> 「やって」, experiment 1 of 3 (CPU only). Supports the paper's theory axis (outline v2): ECA's
Markov closure and apartness have only offline / shadow n = 1 evidence (eca_apartness_* 09-27, sc25). This measures them on all
recorded play, LLM-free.

**Data.** Public: the 87 draws of research/solver_library/draws_v3.txt (8 local games; histories replayed on the local engine,
arc_agi OFFLINE). Unseen: the SIMPLAN-2e NV25 run (F:/arc-agi3-evidence-archive/simplan2/out_unseen_nv25/benchmark.json, 25 games,
replayed on the NV25 sources) - measurement only, nothing enters any solver or prompt.
**Definitions** (tools/closure_census.py). State = hash of the last frame (int16 bytes, sha1, 16 hex; the library's hash).
Transition = (game, state, action incl. click x/y) -> next state; RESET actions are not transitions and clear the history.
A key (game, state, action) seen >= 2 times is *closed* if every observation has the same next state, *broken* otherwise (the frame
does not hold the whole state). *Apartness repair*: extend a broken key with the last k actions of the level's history (k = 1..6,
padded at the level start) and take the smallest k at which every branch is pure; *supported* if every branch also has >= 2
observations. Held-out test: draws split by index parity (even = learn, odd = test); on test transitions whose key is broken in the
learn half, compare (a) the board-only majority outcome with (b) the shortest pure suffix learned on the learn half (abstains when the
test suffix was never seen): accuracy where (b) answers, and coverage.
**Measures.** Per game and pooled: transitions, keys seen >= 2, broken keys (share), transitions on broken keys (share); repair
k distribution, supported share, unresolved at k = 6; held-out accuracy (a) vs (b) and (b)'s coverage. Public and NV25 separately.
**Prediction** (weak priors, stated before any count): broken keys < 10 % of repeated keys on the public games, concentrated in at most
3 games; >= 50 % of broken keys pure at k <= 6, fewer supported; on held-out broken-key transitions (b) beats (a) where it answers.
NV25: no prediction on the rate (one run, fewer repeats). Reading: a low broken share says the frame is (almost) a Markov state on
these games, so certificates keyed on the frame are sound there; a high repair share with held-out gain is evidence that apartness
identifies hidden state; unresolved or unsupported repairs bound what the method can do. Never Kaggle; nothing adopted.

## Result (11-03 ~08:30Z; research/closure_census/census_{public,nv25}.json, tools/closure_census.py)
**Public, 87 draws, 170,714 transitions, 13,300 keys seen >= 2 times:**
| game | repeated keys | broken (share) | pure at k <= 6 | held-out broken-key tests | board-only majority | suffix: answers / acc (majority on the same) |
|---|---|---|---|---|---|---|
| ar25 | 1,974 | 0 (0 %) | - | 0 | - | - |
| bp35 | 981 | 9 (0.9 %) | 9 | 4 | 0.50 | 2 / 1.00 (1.00) |
| cn04 | 1,048 | 353 (33.7 %) | 190 | 99 | 0.60 | 30 / 0.77 (0.80) |
| dc22 | 3,358 | 1,652 (49.2 %) | 1,081 | 1,251 | 0.65 | 422 / 0.80 (0.70) |
| ls20 | 3,370 | 11 (0.3 %) | 3 | 3 | 1.00 | 0 / - |
| re86 | 1,340 | 275 (20.5 %) | 158 | 127 | 0.84 | 85 / 0.95 (0.82) |
| s5i5 | 747 | 213 (28.5 %) | 117 | 92 | 0.92 | 74 / 0.96 (0.93) |
| tn36 | 482 | 122 (25.3 %) | 115 | 40 | 0.65 | 36 / 0.75 (0.64) |
| pooled | 13,300 | **2,635 (19.8 %)** | **1,673 (63 %)**, supported 10 | 1,616 | **0.676** | **649 (40 %) / 0.837 (0.744)** |
Repair k: 1: 560, 2: 301, 3: 308, 4: 210, 5: 156, 6: 138, unresolved 962.

Predictions: broken < 10 % concentrated in <= 3 games - **not met** (19.8 %, five games at 20-49 %; only ar25, bp35, ls20 near 0);
>= 50 % pure at k <= 6 - met (63 %); fewer supported - met (10: with one run per draw most suffix branches hold one observation);
held-out (b) beats (a) where it answers - met (0.837 vs 0.744 on the same 649 cases, 40 % coverage; dc22 +10, re86 +13, tn36 +11,
s5i5 +3, cn04 -3 points).
**NV25 (one SIMPLAN-2e run, 7,663 transitions):** only 148 keys repeat (frames rarely recur within one run), 1 broken - no reading
possible; the census needs repeated play of the same game.

Reading: (1) on five of the eight public games the frame is not a Markov state for a fifth to a half of the revisited
(state, action) pairs - hidden state is the rule, not the exception, so any model keyed on the frame alone (an ECA family's step,
an LLM's step()) is exposed there; the library is not, because it replays a whole sequence from the level start and so reproduces the
hidden state with the frames (182 / 182). (2) ECA's apartness repair - the shortest action suffix that separates the outcomes - splits
63 % of the broken keys within 6 actions and, learned on half the draws, predicts the other half better than the frame alone where it
answers (83.7 % vs 74.4 %, n = 649); it abstains on 60 %, and cn04 is the exception. Large-n support for the closure / apartness part
of the theory, with its limit (coverage) measured.

## Follow-up: level in the key (11-03 ~13:40Z; review of the paper outline v3)
The key (game, frame hash, action) carries no level, so a broken key could pool two levels whose rules differ. Re-run with
`CENSUS_STATE=level` (state = (levels completed, frame hash); tools/closure_census.py, default unchanged) on the same 87 draws ->
research/closure_census/census_public_level.json: every count is identical to the frame-keyed census (170,714 transitions, 13,300
repeated keys, 2,635 broken, 1,673 split at k <= 6, held-out 649 answers, 0.837 vs 0.744; per game the same). No repeated frame occurs
on two levels, so level-rule differences do not explain any broken key: the ambiguity is within a level - state the last frame does
not show. Engine state vs action history is not separated by this (it would need an engine state snapshot per step).
