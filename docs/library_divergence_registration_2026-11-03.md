# When a stored skill deviates, back to the model: divergence arms of the solved-level library (registered 2026-11-03 ~05:30Z, before any draw)

User 11-03, paper claim: unknown games need the LLM's reasoning; the LLM solves the same level in some runs and not in others, which
is fatal in real use; turning what was solved into skills reproduces it, and when the skill deviates control must return to thinking
(「問いた技能を技能化することで、動作を再現して外れたときに思考に戻す必要がある」); asked to register and run the experiment for the
second half (「おねがいします」). In l1-l4 the library never stopped (182 level replays, 0 stops), so the hand-back path was only unit-tested.

## Arms (Colab G4, 8 games, 60 min, 2 draws each; the library-run configuration of docs/library_registration_2026-11-03.md)
One level per game, chosen by a fixed rule (tools/library_divergence.py): library levels >= 2 outside the ECA solver's own levels that
the model played in most draws clearing them, the one cleared in the fewest draws (ties to the higher level): **ar25 L8, bp35 L4,
cn04 L6, dc22 L4, ls20 L4, re86 L6, s5i5 L7, tn36 L7** (the last library level of every game).
- **R (diverge)**: action m = ceil(n/2) of the chosen level is replaced by the first offered action (fixed order) that leaves a board
  different from the stored one without completing the level or ending the game; the replay stops right after action m with a real
  board difference and the host note says "The stored solution of level k stopped after m of its n actions (the board differed from
  the stored one); the level is yours from here." Six substitutes change the board differently (ar25, bp35, cn04, ls20, re86,
  tn36), two leave it as it was (dc22, s5i5): research/solver_library/divergence_v1.log.
- **S (scratch)**: the chosen entry is removed; the model plays the level from its start.
Every other entry is library v2. Harness: branches library-divR 28079de / library-divS 466eda5 on 900261c (only the data file
differs); kernels franzen-m2-library-divR-colab-20261103 839b49bce72c6017, -divS- 16b89ff061c1ed86 (tools/build_library_colab.py divR|divS;
the builder still reproduces v1). Test LIBRARY_DIVERGENCE_OK (real _library_due / _run_library / _deliver_host_note on the local
engine: R stops each game at the chosen level after m actions with the note line; S replays up to the level before).

## Measures (tools to be written before the draws are read; nothing else is changed after they start)
Primary: chosen levels completed in 60 minutes, R vs S, out of 16 level-draws per arm. Secondary: per completed chosen level, actions
and wall time from the hand-back (R: after the replay's stop; S: the level's first board) to completion, and model tokens; whether the
model RESET the chosen level in R (discarding the partial progress); later levels and total levels per draw. Reference: l1-l4
(the same levels replayed whole, 8/8 every draw).

## Reading rule
R - S >= 3 of 16: handing back mid-way helps (a skill that deviates still pays); |R - S| <= 2: no readable difference (hand-back is
safe but not a gain); S - R >= 3: the deviated state hurts (the hand-back should rewind to the level start). Prediction: R >= S; the
two hand-backs that left the board as it was are completed at least as often as the six that moved it elsewhere; some R draws RESET.
Historical clears of these levels (non-library draws, the model reaching them late in the hour): ar25 L8 64, dc22 L4 59, ls20 L4 36,
re86 L6 36, tn36 L7 21, cn04 L6 7, bp35 L4 1, s5i5 L7 1 of ~83, so S should clear ar25 / dc22 most draws and bp35 / s5i5 rarely.
Never Kaggle; nothing submitted or adopted without the user's word.

## Interim (r1 / s1 only, 11-03 ~06:35Z; not a verdict - r2 / s2 have been waiting for a G4 since 06:37Z: "Backend rejected accelerator 'G4'. You may not have quota or entitlement")
Every R replay stopped where registered (8 / 8 "diverged after m/n"). Chosen levels completed: **R 5 / 8** (ar25 L8, dc22 L4, ls20 L4,
re86 L6, s5i5 L7) vs **S 3 / 8** (ar25 L8, dc22 L4, ls20 L4); bp35 L4, cn04 L6, tn36 L7 in neither. Actions on the level (R incl. the
m replayed): ar25 49 vs 96, dc22 65 vs 203, ls20 158 vs 67, re86 66 (cleared) vs 163 (not), s5i5 74 (cleared) vs 8 (reached late).
Total levels 43 vs 41. One draw per arm: direction only.

## Result (r1, r2, s1, s2; r2 / s2 ran 08:28-~10:00Z after the Colab top-up; every R replay stopped where registered, 16 / 16)
| draw | chosen levels completed | total levels |
|---|---|---|
| r1 | 5 / 8 (ar25, dc22, ls20, re86, s5i5) | 43 |
| r2 | 6 / 8 (ar25, bp35, dc22, ls20, s5i5, tn36) | 44 |
| s1 | 3 / 8 (ar25, dc22, ls20) | 41 |
| s2 | 4 / 8 (dc22, ls20, re86, tn36) | 42 |
**R 11 / 16 vs S 7 / 16: R - S = 4 >= 3 - by the registered rule, handing the level back mid-way helps** (a skill that deviates still
pays). cn04 L6 was completed by neither arm.
Secondary (completed chosen levels, from the hand-back in R / the level start in S): model actions median **41 vs 117**, generated
tokens median **112 k vs 157 k**, minutes median **28 vs 35**; paired levels completed in both arms: ar25 25 / 41 vs 96, dc22 34 / 36
vs 203 / 168, re86 39 vs 117, tn36 59 vs 151, ls20 131 / 60 vs 67 / 111 (the one level where R was not shorter). RESET after the
hand-back (discarding the partial progress): 3 of 16 R level-draws (bp35 in both, re86 in r2). The two hand-backs that left the board
as it was (dc22, s5i5) were completed 4 / 4, the six that moved it elsewhere 7 / 12. All three predictions met (R >= S; no-change
hand-backs at least as good; some RESETs). In S, s5i5 L7 took 17 / 48 minutes and 100 k / 227 k tokens for 8 / 10 actions: the model
deliberated without acting. Two draws per arm; the counts are small, the direction is consistent across both pairs (5 v 3, 6 v 4).
