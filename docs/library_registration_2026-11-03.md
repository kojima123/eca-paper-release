# Solved-level library: what the model solved once is reused (registered 2026-11-02 ~21:00Z = 11-03 06:00 JST, before any draw)

User 11-03: 「解けた場合はソルバー側に反映できるようにしたい。研究としては、能力の上限=LLMの限界でも良いのですが、解けるものは再利用する
とこまで繋げたい」, both tracks in parallel (「両方を並行」): (1) a library of solved levels for known games (this registration), (2) the
within-run transfer of a solved level's model to the next level (offline measurement first; separate note).

## Why
ECA-off comparison (docs/ecaoff_registration_2026-11-02.md): the solver's strength is replaying what it has verified, exactly and
every time; the model alone finds most early levels too but not every draw. Over the 83 local draws of the 8 games the mean is 33.3
levels per draw while the union of the best per game is 45 (ar25 8, bp35 4, cn04 6, dc22 4, ls20 4, re86 6, s5i5 6, tn36 7): most of
what the model can solve, it solves only sometimes.

## Mechanism
- Harvest (tools/library_harvest.py, local engine, no LLM): every draw's history is replayed and cut into levels; per level the
  shortest clean sequence (after the level's last RESET) that completes it from the canonical start is kept and shortened (revisits
  and no-change actions cut when the level still completes); the entry keeps the start-board hash and the board hash after every
  action. Library v1 (research/solver_library/library_v1.json, from research/solver_library/draws_v1.txt = 83 draws): 45 levels, all
  verified, 0 replay mismatches; every stored solution is at or below the human baseline in actions.
- Reuse (harness changelog 3b53c3a, ECA_FRZ_LIBRARY=1, default off): once per level at its first sight, when the board hashes to the
  stored start board, the solver replays the stored actions through the ordinary _execute_action (automatic), checks the board hash
  after each, stops at the first difference; consecutive stored levels chain. The actions go to the host note as family 'library'
  (levels completed, which levels were replayed, where a replay stopped). Tests: LIBRARY_E2E_OK (local cn04: whole game; corrupted hash
  stops L3 after 5 actions with the note saying so; wrong start hash and flag off replay nothing), LIBRARY_ALLGAMES_OK (all 45 levels
  of the shipped library through the harness path on the local engine), earlier tests unchanged.
- Loop: after each round the new draws are harvested into the next library version (same tool, the old draws plus the new); a level
  the model newly solved enters it and is replayed from the next round on.

## Arm and draws
Kernel franzen-m2-library-v1-colab-20261103 (tools/build_library_colab.py; ipynb sha 1695ce63579e5e67, patch bdb3c20..3b53c3a
ce2b878a832c980e): the control configuration (DIGEST + WATCH + CHANGELOG, ECA port on) + ECA_FRZ_LIBRARY=1 + host-note level fix +
warmup RESET fix; failure note / end facts / repair off. Draws gyro-l1, l2 (Colab G4, 8 games, 60 min), compared with the control
ctl1/ctl2 and the union (45).

## Measures
tools/check_library.py: library levels replayed per game (completed / stopped / start differs), levels at 60 min, frontier levels
(beyond the library: bp35 L5-9, dc22 L5-6, ls20 L5-7, re86 L7-8, s5i5 L7-8; ar25, cn04, tn36 are complete), model tokens;
tools/summarize_ecaoff.py for levels, mean score and tokens per draw.

## Predictions
1. LIBRARY_REPLAYED_OK: each draw replays at least 44 of the 45 levels (the start boards on Colab hash as on the local engine). A
   systematic start-board difference would be a defect of the mechanism, not a result.
2. Levels per draw >= 45 (control 34.5), a mechanical consequence of 1.
3. Frontier: 0-2 new levels per draw. None of these 14 levels was solved in 83 draws; the model now spends nearly all its 60 minutes
   on them (the ECA-off comparison found that freed time did not become later levels, so a large gain here is not expected).
4. Tokens per draw stay ~1.8 M (time-capped), all spent on the frontier.
5. Mean game score above the control's 42.8 (library levels at or under the human baseline).
Reading: this is the known-task setting by construction (the 8 games are the library's source); it measures how far reuse lifts a
run to the union of past successes and whether the time it frees moves the frontier. It says nothing about unknown games (the
hidden set has no library entries). Never Kaggle; nothing here is submitted or adopted without the user's word.

## Result (l1 / l2, finished 11-02 22:31Z / 22:54Z; dry run DRY_OK, check LIBRARY_REPLAYED_OK)

| run | levels | mean game score | tokens (M) | tokens per level | library levels replayed | frontier levels |
|---|---|---|---|---|---|---|
| C control ctl1 / ctl2 | 31 / 38 (34.5) | 42.81 | 1.82 | 52612 | - | - |
| D v8 f15 / f16 | 35 / 33 (34.0) | 42.97 | 1.79 | 52645 | - | - |
| L library l1 / l2 | **45 / 46 (45.5)** | **66.49** | 1.38 | 30379 | 45/45, 45/45 | 0 / 1 (s5i5 L7) |

Predictions: 1 met (every stored level replayed in both draws: start boards on Colab hash as on the local engine, no stop);
2 met (45, 46 >= 45; control 34.5); 3 met (0 and 1 new levels; s5i5 L7 in l2, 65 actions, human baseline 86); 4 **not met as
worded**: 1.38 M, not ~1.8 M - ar25, cn04 and tn36 were complete after the library (no model turn at all, 0 tokens), the other five
games spent 257-295 k each (control 217-262 k); 5 met (66.49 vs 42.81).

- Library replay is exact: every level at the stored length in both draws (ar25 253, cn04 178, tn36 94 actions, the same as the
  local test LIBRARY_ALLGAMES_OK), so the replayed levels score the same every draw (ar25 / cn04 / tn36 100, the others capped by
  their frontier).
- Frontier (14 levels, none solved in 83 earlier draws): the model gets the whole 60 minutes per game on one level and takes few
  actions on it - bp35 L5 74 / 93, dc22 L5 70 / 110, ls20 L5 152 / 111, re86 L7 102 / 64, s5i5 L7 19 (l1) / 65 cleared (l2, then
  33 on L8). One new level in two draws: the freed time does not turn into frontier levels at this rate (as the ECA-off comparison
  found for the solver's freed time). This is the LLM limit of the harness on these games, measured with every known level free.
- The summary tool's "solver-completed" column counts delivered host notes (24 per draw); a game whose levels are all replayed has no
  model turn, so its library note is never delivered - the column undercounts the library levels (45 per draw from the check tool).
- Reading: known-task reuse works as built - a run starts at the union of earlier successes (45 vs 34.5 per draw, +11) at no model
  cost for those levels. The gain is only on the 8 known games; the hidden set has no entries.

## Loop, round 1 -> library v2
Harvest of the 85 draws (draws_v1 + l1 + l2; research/solver_library/draws_v2.txt, harvest_v2.log): library_v2.json, **46 levels**
(v1's 45 unchanged + s5i5 L7, 61 actions after shortening, human 86), replay mismatches 0. Round 2 (draws with library v2) is not
launched; it needs the user's word.

## Round 2: library v2 (registered 2026-11-03 ~01:00Z, before the dry run and any draw)
User 11-03 「おねがいします」 (go for round 2). Harness changelog 900261c on 3b53c3a: inference/data/solver_library.json = library v2
(46 levels), code unchanged; LIBRARY_ALLGAMES_OK on all 46 levels (s5i5 7/7, 226 actions). Kernel franzen-m2-library-v2-colab-20261103
(tools/build_library_colab.py ... v2; ipynb sha dbd641d902a1c901, patch bdb3c20..900261c ea6eb2f493e552eb): differs from v1 only in
cell 3 (the patch) and cell 17 (library check 46); the builder still reproduces v1 byte for byte (1695ce63579e5e67). Draws gyro-l3, l4
(Colab G4, 8 games, 60 min), compared with round 1 (l1/l2) and the control (ctl1/ctl2).

Predictions: (1) each draw replays at least 45 of the 46 levels; (2) levels per draw >= 46 (round 1 45.5); (3) frontier 0-2 new levels
per draw among the 13 left (bp35 L5-9, dc22 L5-6, ls20 L5-7, re86 L7-8, s5i5 L8; round 1: 0.5 per draw); (4) tokens ~1.4 M per draw
(ar25, cn04, tn36 complete without a model turn); (5) mean game score >= 66 (round 1 66.49; s5i5 starts at L8). After the draws the
loop harvests l3/l4 into library v3 (draws_v3 = draws_v2 + l3 + l4). Never Kaggle; nothing submitted or adopted without the user's word.

## Round 2 result (l3 / l4, finished 11-03 02:21Z / 02:24Z; dry run DRY_OK, check LIBRARY_REPLAYED_OK)

| round | runs | levels per draw | mean game score | tokens (M) | library replayed | frontier new |
|---|---|---|---|---|---|---|
| control | ctl1 / ctl2 | 31 / 38 (34.5) | 42.81 | 1.82 | - | - |
| 1 (v1, 45) | l1 / l2 | 45 / 46 (45.5) | 66.49 | 1.38 | 45/45, 45/45 | 0 / 1 |
| 2 (v2, 46) | l3 / l4 | **46 / 46 (46.0)** | **67.71** | 1.34 | 46/46, 46/46 | 0 / 0 |

All five predictions met: 46/46 replayed in both draws (s5i5 L7, harvested from l2, replayed at 61 actions both times), 46 levels
per draw, 0 new frontier levels (inside 0-2), 1.34 / 1.35 M tokens, mean game score 67.71 in both draws (identical per game: the
library levels score the same every draw, the frontier games stay at their library level).
- Frontier, where the model spent its 60 minutes: s5i5 L8 65 / 51 actions (human 83), dc22 L5 84 / 97 (324), re86 L7 126 / 171
  (424), bp35 L5 73 / 87 (33), ls20 L5 167 / 163 (96); about 185-292 k generated tokens per game on one level, none completed.
- Loop, round 2 -> library v3: harvest of the 87 draws (draws_v3.txt, harvest_v3.log) gives the same 46 stored solutions as v2 (no
  solution, start hash or board hash changed; only the draws_cleared counts), replay mismatches 0. The loop has reached its fixed
  point on these 8 games: the library holds every level the model has ever solved, and two more rounds added nothing.
- Over both rounds the frontier was tried 54 times (14 + 14 + 13 + 13 level-draws, a full hour each) and gave one level (s5i5 L7).
  Reading for the paper: reuse makes a known game's score the union of past successes, exactly and at no model cost (34.5 -> 46
  levels per draw, mean game score 42.8 -> 67.7); beyond it the run is bounded by what the LLM finds in one hour on one level, about
  one new level per 50 level-hours here. Nothing changes for unknown games (no entries). Not adopted into any submission; no rounds
  further unless the user asks.
