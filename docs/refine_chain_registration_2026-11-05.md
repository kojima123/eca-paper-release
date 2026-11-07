# After a gate stop: counterexample, rewrite, re-certification, completion (registration, 2026-11-05)

**Why.** A review of the paper (user, 11-05) asks whether the execution contract can serve as a scaffold for model
repair: the gate stops a plan at its first prediction mismatch; can the counterexample lead to a rewritten `step`, a
new certificate and a completed level? The user asked for an additional experiment within a realistic range
(deadline 11-08/09).

**What already exists.** The modeler arm (docs/modeler_registration_2026-10-31.md, draws gyro-m1 / gyro-m2,
E:/arc-agi3-online-specialists-runs-e/modeler_runs) already ran this loop in a soft form: after a stop the LLM's
next tool result reported the mismatch (action, differing cells), the LLM was free to rewrite `step`, and the harness
re-verified, searched and executed after every model call that changed `step` or `goal`. The verdict counted host
runs and completions but never followed each stop forward. This analysis does that, on the recorded draws only.
No new draw, no GPU. Before writing this file I had seen four host-run rows of m1 (cn04 L1 stop at action 9, cn04 L3
plan_done of 2, dc22 L1 stops at 4 and 1); nothing else of the chain.

**Definitions** (per draw m1, m2; tool `tools/refine_chain.py`):
- *Stop*: a host-run row (`event: modeler`, `host.run`) with `run.outcome == "mismatch"`. Its key is (game, level),
  its time the row's `at_step` plus the executed actions.
- After a stop, on the same (game, level), until the level ends or the run ends:
  - *rewrite*: an assistant tool call whose code defines `def step(` in a request whose `action` counter is past the stop;
  - *re-certified*: a later host row on the level with `exact == true` and `search.found == true`;
  - *host-completed*: a later host run on the level with outcome `level_completed`;
  - *level completed*: the game's `levels_completed` passed the level (by the host or the LLM).
- Chain counts are reported per stop and pooled; one level may carry several stops, so levels are reported too.

**Predictions** (from the verdict: 20 host runs, 13 stopped on new ground, 5 host-completed levels pooled):
1. Rewrites follow most stops (>= 8 of 13).
2. Re-certification after a stop is rare (<= 4 of 13).
3. Host completions that follow a stop on the same level: 0-2.
4. Levels with a stop are completed by someone at a lower rate than levels with a host plan that never stopped.

**Reading.** If at least one stop leads through rewrite and re-certification to a host completion, the paper reports
it as a case study of the loop, with every other stop listed beside it (no selection). If none does, the paper
reports where the chain breaks (no rewrite, rewrite not exact, no plan, plan stopped again). Either way the result is
descriptive: two draws, eight games.

**Result** (11-05, `python3 -I tools/refine_chain.py m1/out_canon m2/out_canon research/refine_chain/refine_chain_v1.json`):

| | m1 | m2 | pooled |
|---|---|---|---|
| stops (host run ended on a mismatch) | 8 | 5 | 13 |
| followed by a `def step(` rewrite on the level | 8 | 5 | 13 |
| followed by re-certification (exact + plan found) | 7 | 5 | 12 |
| followed by a host completion of the level | 5 | 0 | 5 stops, 2 levels |
| levels with a stop / completed by anyone | 4 / 4 | 1 / 0 | 5 / 4 |

Predictions: (1) rewrites >= 8 of 13 - met (13); (2) re-certification <= 4 of 13 - **not met** (12: the LLM nearly
always produced a model exact on the record again); (3) host completions after a stop 0-2 levels - met (2); (4) levels
with a stop completed less often than levels planned without one - 4/5 vs 3/3, direction met, too few to read.

Where the chain completed (m1 dc22, levels not assigned to the pre-authored families):
- dc22 L1: plan 9 stopped at action 4 (MOUSE, 48 cells) -> rewrite -> 26/26 exact, plan 14 stopped at 1 (UP, 4 cells)
  -> rewrite -> 27/27, plan 13 stopped after 11 (RIGHT, 4 cells) -> rewrite -> 38/38, plan 2 -> level completed.
  Level in 40 actions; LLM-only controls 38 / 37; human baseline 59.
- dc22 L2: plan 23 stopped at 1 (UP, 1 cell) -> 50/50, plan 22 stopped after 10 (RIGHT, 4 cells) -> 60/60, plan 12 ->
  level completed. 72 actions; controls 82 / 72; baseline 102.
Where it did not: m2 ls20 L3 (5 stops, 46- and 6-cell mismatches, re-certified each time, never completed); m1 ls20 L2
(2 stops, completed later by the LLM after 185-207 more actions); m1 cn04 L3 (1 stop, re-certified, a plan of 2 ran
without completing; the LLM completed it 30 actions later).

Reading: the loop the review asks for is observable in the recorded draws: a counterexample from the gate was followed
by a model that agreed with the record again and, on two dc22 levels, by a host completion. It completed levels the
LLM alone also completes in about the same number of actions; it does not show a capability beyond the controls, and
the arm as a whole lost levels (55 vs 69). Re-certification is easy (exact on the record); being right on the next
unobserved transition is not.

## Addendum (11-05, after review): v2 ties each check to the model version in effect

**Defect in v1.** `calls()` dated an assistant tool call by the action counter of the first request whose history
carried it. The call that defined the model and triggered the stopped run first appears in the request right after
the stop, so v1's rule `a >= stop_at and "def step(" in code` could count the stopping definition itself as a rewrite.
v1 also took any later exact+found host row on the level, without tying it to a model defined after the stop, and
the host's `exact` flag is the near-exact rule of the modeler registration (<= 10 % of judged transitions
mismatched, >= 3 board-changing transitions, no errors), not agreement with every transition. A reviewer reproduced
the false positive on a minimal synthetic log (11-05). v1 and refine_chain_v1.json are kept for the record and
superseded.

**v2** (`tools/refine_chain_v2.py`, `--selftest` reproduces the reviewer's case: 0 rewrites; an unchanged re-send is not
a new version). It uses row order: `response` k -> tool rows (`model_check`, `modeler`) -> `request` k+1, where the call
generated in response k first appears; every host row in between was produced by that call. Each host row carries the
version of `step` in effect (a new version when a call defines a `step` whose normalised source differs; helpers are
not compared, so this is a lower bound on model changes). A stop is followed only through host rows of later calls
on the same level whose version differs from the stopping version. "Strict" = exact flag and verify.mismatched == 0
and errors == 0.

`python3 -I tools/refine_chain_v2.py m1/out_canon m2/out_canon research/refine_chain/refine_chain_v2.json`

| | m1 | m2 | pooled |
|---|---|---|---|
| stops | 8 | 5 | 13 |
| followed by a `def step(` call | 8 | 5 | 13 |
| followed by a new version of `step` | 8 | 5 | 13 |
| ... re-certified under a new version (near-exact flag + plan) | 6 | 3 | 9 (v1: 12) |
| ... re-certified under a new version, strict | 6 | 0 | 6 |
| ... host-completed under a new version | 5 | 0 | 5 stops |
| level attempts with a stop | 4 | 1 | 5 |
| level attempts reaching a strict re-certification | 3 | 0 | 3 (cn04 L3, dc22 L1, dc22 L2) |
| level attempts host-completed after a stop | 2 | 0 | 2 (m1 dc22 L1, L2) |

What changed: rewrites stand (13/13 followed by a changed `step`, not only by the stopping definition); re-certification
falls from 12 to 9 near-exact and 6 strict; ls20 L3 (m2) never reached a strict version (its "re-certifications" were
near-exact); ls20 L2 (m1) never reached an accepted version. The dc22 chains are unchanged: each new version agreed with
all 26, 27, 38 (L1) and 50, 60 (L2) recorded transitions, and the completing versions are the 38- and 60-transition
ones. Prediction 2 (<= 4 of 13) remains not met (6 strict). The 13 stops are events in 5 level attempts; the two
completions are in one run.

Separately checked: the 49 of 91 no-path searches quoted in the paper are exactly the searches under strict versions
(the other 22 no-path searches were under near-exact versions), so "under a model consistent with every recorded
transition" stands.
