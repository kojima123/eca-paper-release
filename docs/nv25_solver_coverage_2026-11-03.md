# Solver coverage on games it was not built from: NV25 (registered 2026-11-03 ~03:30Z, before the run)

User 11-03: after 「当てはまるゲームを当てたられるって簡単なの？」 the fair measurement is how often the solver certifies and plays
on games it was not written from; asked whether NV25 may be used for measurement only (no tuning), the user said 「おねがいします」.

**What is measured.** The current harness tree (changelog_repo ecaod-changelog 900261c; library flag off, so no stored solutions)
in the local integration harness (changelog_check_v1/eca_port_integration_local.py, arm `on`, the RWS arcgym bundle) with a fake
model that only inspects: every action is the ECA solver's own certified-prefix executor. Exactly the setup of
unknown_dev_eca_alone_2026-10-30.md (arcint 8: 0 certified, 0 actions, 0 levels), at 200 requests per game as that note's
public reference. Per game: family evaluations, certified families, actions the solver executed, levels completed, and the
competition score from the NV25 metadata baselines.

**Games.** NV25 (C:/arc-agi3-artifacts/unseen_ds/nvidia25, 25 games, runtime_support on PYTHONPATH). Same-day reference: the 8
public games (C:/arc-agi3/environment_files), same tree and settings. Smoke test before registration: al7306 loads, 30
evaluations, 0 certified, 0 actions (that one game is part of the run).

**Rules.** NV25 is run once; nothing in the solver or the harness is changed on its result; no NV25 file enters any solver,
library or prompt. The output is a count, not a tuning signal.

**Prediction.** The families (shape_marker, program_execution, reflection_orbit, linked_extent, anchor_overlap) are the public
games' mechanics; arcint gave 0 of 8. Expected on NV25: certification on at most 2 of 25 games, at most 3 levels in all; the public
reference reproduces its 10-30 levels (re86 4, tn36 5, ar25 3, s5i5 2-3, cn04 2) within one level per game. A public reference far
below that would make the NV25 count unreadable (harness defect), not a result.

## Addendum (registered ~03:45Z, while the inspect-only run is going, before any acting run)
The inspect-only model never acts, so a family that needs observed transitions (a key that moves something) has nothing to
certify from; in a real run the model's own actions give the solver that evidence. Second condition, the same games, tree and
cap: the harness's default fake model (no script file): it follows the solver's plan whenever exactly one authorized plan is
offered and otherwise cycles the valid actions (clicks at fixed pseudo-random cells), so the solver sees transitions. Run with
arm `on` and arm `off` (no ECA key at all, the same cycling model): levels with `on` minus levels with `off` is what the solver adds
over blind play. Same rules (once, no change on the result). Prediction unchanged: certification on at most 2 of 25 NV25 games,
on - off at most 3 levels in all; public on - off near the 10-30 solver levels.

## Result, inspect-only (11-03 ~04:50Z; tools/nv25_solver_coverage.py)
| set | games | games with a certified family | solver actions | levels |
|---|---|---|---|---|
| public 8 (reference) | 8 | 5 (ar25 reflection_orbit, cn04 anchor_overlap, re86 mirror + shape_marker, s5i5 linked_extent, tn36 program_execution) | 149 | 14 (ar25 3, cn04 2, re86 4, s5i5 2, tn36 3) |
| **NV25** | 25 | **2** (mirror on fl5273 and ml2048) | **2** (1 each) | **0** |

Prediction met (certification on <= 2 of 25, <= 3 levels). Both NV25 certifications acted once and stopped: the mirror family's
prediction failed on the first action - a wrong certification on an unseen game cost one action. Reference: 4 of 5 public games
within one level of 10-30; tn36 3 vs 5 (two lower), total 14 vs 15-16 - not "far below", the NV25 count stands. 754 family evaluations
on NV25 (30 per game, 15 families each), so the solver ran; nothing fits.

## Acting condition: attempt 1 failed, re-run (11-03 ~06:20Z)
Attempt 1 produced no summary: with the cycling model every game runs to the 900 s wall cap (public: 70-195 requests, 109-231
actions), so the public_on run hit its 7200 s outer timeout before writing summary.json, and the three later runs never started
(WSL interop dropped: "Exec format error"). Logs kept in failed_attempt1/. Re-run with the same settings (cap 200 requests, wall
900 s, cycling fake model, arm on / off), games split into 5 shards per arm run in parallel (the fake server binds an ephemeral
port; interop re-registered before each call): run_acting_shards.sh. Nothing in the solver or the method changed.

## Result, acting condition (re-run in shards, finished 11-03 ~08:40Z; all 66 game runs rc 0, no errors)
| set | arm on: levels (games with a certificate, solver actions) | arm off: levels | on - off |
|---|---|---|---|
| public 8 | **10** (5: ar25 reflection_orbit 3 levels / 39 actions, cn04 anchor_overlap 2 / 28, re86 shape_marker 3 / 46, s5i5 linked_extent 1 / 12, tn36 program_execution 1 / 6) | 0 | +10 |
| **NV25 25** | **2** (2: mirror on fl5273 and ml2048, 1 action each, 0 levels; the 2 levels are ll4821 and sf2048, where the solver acted 0 times) | 1 (sf2048) | +1, not the solver's |
Prediction met (certification on <= 2 of 25, on - off <= 3). With a cycling model that gives the families transitions to certify
from, the hand-written families add 10 levels over blind play on the public games they were written from and nothing on NV25: the
same two mirror certifications as inspect-only, each revoked after one action. The extra ll4821 level is blind play (the solver never
acted there; arm timing changes how many cycling actions fit in 900 s). Solver alone on unseen games, both conditions: 0 levels.
