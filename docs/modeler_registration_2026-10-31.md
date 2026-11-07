# Modeler arm: the model writes the operating model as code, the harness probes, verifies and executes (registration, 2026-10-31)

**Why.** The user (10-31 JST), after the status-sheet verdict: 「LLMに考えるだけさせることできますか？読み取り、行動などは
すべて別の方法でおこなう」, then 「未知ゲームだとどうなりますか？少なくとも操作系はLLMが作る必要がある」, and to the
reply (the LLM writes the operating model as code; code verifies it on every action and uses it to plan; the LLM
never issues individual hand-chosen moves beyond a probe budget and never reads the raw frame by hand): 「それでいこう」.

**What exists already, honestly.**
- FRZ-SIMPLAN (10-05) / SIMPLAN-2 (10-17): the same `step(board, action)` / `goal(board)` idea as *optional* helpers
  (`verify_model`, `find_plan`, `run_plan`) plus a one-line nudge after 40 stuck actions. The model used them in 2-3 %
  of requests, `find_plan` found 3 plans, one level came through `run_plan`; hidden draws 21.9 mean over 5 (other
  arms on the same base 27.2); games stuck at level 1 spent the hour rewriting the simulator (m0r0 4->0, sk48 4->0,
  sc25 6->0). s0, the current base, has SIMPLAN off (`docs/simplan_suppression_note_2026-10-29.md`,
  `docs/eca_harness_repair_feasibility_2026-10-26.md` section 1).
- The ECA research line (`docs/eca_insights_for_llm_20261020.md`, feasibility doc section 7.6) concluded: separate
  perception (ECA) from rules (LLM); verify over every recorded transition automatically ("ECA が自動で呼んで結果を渡す。
  LLM に呼ばせる形にはしない"); plan with code. Offline, with Qwen3.8-27B, the LLM's step reached 70-97 % agreement and
  record agreement did not equal levels (96.8 % agreement, 3/8 levels).
- The harness has the pieces: `model_check.py` (verify / BFS plan / checked execution), persistent functions (scope
  `game` in the canon), the host-takeover pattern of the level-1 explorer, the effect ledger and stall report.

**What this arm changes relative to SIMPLAN** (the differences are the experiment):

| | SIMPLAN-2 | Modeler |
|---|---|---|
| Who calls verification | the model, when it chooses (2-3 % of turns) | the harness, after every model call that acted or changed `step` / `goal` |
| Who executes a plan | the model, via `run_plan` | the harness, between turns, when `step` is near-exact on the level's record (<= 10 % mismatched) and `goal` is false now |
| Unchecked actions | unlimited | every `action()` is checked against the kept `step`: a predicted action stops the batch at the first mismatch; an unpredicted one is a probe, 30 per level, then refused until `step` covers it (relief after 10 action-less turns) |
| Prompt | helpers advertised + nudge | one addendum stating the regime; no nudge |

**Change** (eca-port tree ecaod-changelog 6184a4e = 13a98a4 + this; default off):

| Flag | Effect |
|---|---|
| `ECA_FRZ_MODELER=1` | Implies the SIMPLAN helpers (`board_of`, `verify_model`, `find_plan`, `run_plan`) with the nudge off. (1) Checked actions: in every snippet, `action(actions)` runs one action at a time; if a callable `step` is in the namespace (retained or defined in the call) its prediction for the action is computed first; a predicted action executes and, if the board differs from the prediction outside the 4-cell border, the batch stops with `model_mismatch` (action, differing cells) in the result and a `MODEL MISMATCH` echo line; an action `step` abstains on (None, error, or no `step`) executes as a probe. Probes are counted per level (`ECA_FRZ_MODELER_PROBES`, default 30); past the cap an abstained action is refused (`stop_reason` `probe_budget`, nothing spent) until `step` predicts it; after `ECA_FRZ_MODELER_RELIEF_TURNS` (default 10) consecutive turns on the level without an executed action the cap is lifted for that level. (2) Harness check: after each model snippet that executed an action or retained a changed `step` / `goal` / `click_candidates`, the harness runs a hidden snippet in the same sandbox: `verify_model(step, goal, border=4)` over the level's record; when `step` is near-exact (no errors, no false or missed deaths, >= 3 board-changing transitions judged, mismatches on at most 10 % of the judged transitions; `run_plan` stops at the first wrong prediction anyway), `goal` is retained and false on the current board, it runs `find_plan(step, goal, border=4, seconds=12, max_nodes=60000)` over the keyboard actions plus `click_candidates(board)` when retained, and executes a found plan with `run_plan(actions, step)` (checked, stops at the first mismatch, at most `ECA_FRZ_MODELER_RUN_MAX`=40 actions, sandbox timeout 90 s). A search that found nothing is not repeated for the same `step` / `goal` on the same board. (3) Opener line every turn: `Model status (harness): ...` = whether `step` / `goal` are retained, the last verification counts and first mismatch, the last search and execution outcome, probes used / cap. (4) One system-prompt addendum. Per-level counters reset at the level change; retained functions follow the canon's scope (`game`). |

Facts only, no advice on what rule to write. Nothing changes in a request when the flag is off.

**Prediction.** Forced by the probe cap, `step` is retained on most levels (SIMPLAN: 2-3 % of turns). Exactness is the
binding constraint: the model's step will be contradicted on most levels (prior 70-97 % agreement), so few harness
plans run; levels at 60 minutes are expected at or below the control (the SIMPLAN precedent), and the floors are
expected to fail. What the draw settles: (a) whether forcing plus automatic verification changes the modelling rate
and exactness versus SIMPLAN's opt-in, (b) whether any level is cleared by the harness executor, (c) what the probe
cap costs. Adoption is not expected from this draw; the result decides whether the line continues (ECA perception
functions for the step, object-level rules) or stops.

**Design.** Two draws on Colab G4 (gyro-m1, gyro-m2), same 8 games (ar25 bp35 cn04 dc22 ls20 re86 s5i5 tn36), 8
concurrent, 60 min, kernel from `tools/build_modeler_colab.py` (changelog canon d4045d4d + patch + `ECA_FRZ_MODELER=1`
on DIGEST / WATCH / CHANGELOG; cells 3, 5, 17 only). Controls: explain2 ctl1 / ctl2 (69 levels / 685.0). Never
Kaggle, never submitted.

**Checks before the draws** (done 10-31 01:00-02:40Z, `C:/arc-agi3-online-specialists-runs/modeler_check_v1`).
- Sandbox unit test on a synthetic 40x40 mover game: no step + cap 3 -> 3 probes then a `probe_budget` refusal; relief lifts
  the cap; a correct step -> 3 predicted actions, no mismatch, step retained; a no-change step -> the first changing
  action executes and stops the batch with `model_mismatch`; host code with a correct step + goal on a 4-move record ->
  verify 4/4 exact, plan of 16 actions found, run_plan executed it to `level_completed`; contradicted step -> no search;
  no step -> `{step: false}`; flag off -> no `modeler` key, the batch runs as one call.
- Fake-model integration (real session loop and ToolAgent, tu93 circling script, 60 requests, cap 5, relief 4): cap
  mode (no step) 5 probes, 4 refusals, relief, 44 actions; step mode (no-change step + goal False) 48 host rows, step
  retained, verify each call; abstain mode 44 host rows, 44 abstained, cap and relief; off (same fake model, no flag) 0
  addendum, 0 status lines, 0 host rows, 0 echoes, actions unchanged. On sp80 (board-changing script, step mode): 36
  mismatch stops, 40 host rows all contradicted (first mismatch 160 cells), level 2 reached, 0 tracebacks everywhere.
- Colab CPU dry run: required lines incl. `modeler: True 30 0`, `simplan2 check: True False`, `status sheet: 0`.

**Verdict** (`tools/judge_modeler_colab.py m1 m2 ctl1 ctl2 report`; a level counts when it saw >= 20 actions):

| Verdict | Condition |
|---|---|
| `WRONG_GAMES` / `CONTROL_LEAK` / `NOT_ON` | games differ; a control carries the addendum or a host row; an arm lacks them |
| `MODEL_NOT_WRITTEN` | `step` retained on < 50 % of counted levels (pooled over the two draws) |
| `MODEL_NOT_EXACT` | among levels with `step`, those where a harness verification found it near-exact (the host's `exact` flag: <= 10 % mismatched, >= 3 board-changing transitions, no errors or death mistakes) < 30 % |
| `PLAN_NOT_FOUND` | among exact-model levels with `goal` retained, those where a harness search found a plan < 30 % |
| `EXECUTOR_TAKEN` | >= 3 level completions by harness-run plans, pooled |
| `EXECUTOR_NOT_TAKEN` | otherwise |
| `HARMFUL` / `PROMISING` / `NULL` | levels at 60 min <= control - 5 (69) or score < 0.85 x 685.0; both >= control; otherwise |

Descriptive, no verdict: probe refusals, mismatch stops, host-executed action share, search seconds, per-game levels.
Adoption into the canon only on EXECUTOR_TAKEN and not HARMFUL, and only with the user's word.

**Result** (draws gyro-m1 / gyro-m2, 2026-10-31 02:34-04:01Z, E:/modeler_runs/m1, m2; judge
`docs/modeler_colab_verdict_2026-10-31.txt`, rc 0): **EXECUTOR_TAKEN HARMFUL**. Not adopted; the flag stays default-off.

| run | levels | score | host rows | exact rows | searches | plans found | host runs | host actions | host-completed levels | host seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| m1 | 27 | 228.0 | 182 | 80 | 61 | 12 | 12 | 103 | 3 | 302 |
| m2 | 28 | 270.1 | 167 | 68 | 30 | 8 | 8 | 95 | 2 | 287 |
| ctl1 | 31 | 279.1 | 0 | – | – | – | – | – | – | – |
| ctl2 | 38 | 405.9 | 0 | – | – | – | – | – | – | – |

Chain (pooled): counted levels 30, `step` retained on 27 (90 %; 34 levels with a step at all), near-exact on 20 of 34
(59 %), plans on 8 of 16 exact+goal levels (50 %), 5 levels completed by harness-run plans (dc22 x3, s5i5 x1, ls20 x1),
198 harness-executed actions. Every mechanism gate passed; the floors failed: levels 55 vs 69 (line <= 64), score
498.1 vs 685.0 (0.73, line 0.85). Both draws ran the full 8 x 3600 s, 0 tracebacks (the m1 monitor stream dropped
once at 03:20Z; the run itself continued and exited 0).

Per-game levels (m1 / m2 vs ctl1 / ctl2): ar25 6/7 vs 6/8; bp35 1/1 vs 1/2; cn04 3/2 vs 2/5; dc22 4/4 vs 4/4;
ls20 2/2 vs 4/4; re86 4/5 vs 5/5; s5i5 2/1 vs 3/3; tn36 5/6 vs 6/7. No game gained in both draws; ls20 and s5i5 lost
in both. Time to level (minutes): dc22 L1 at 14 / 22 vs 9 / 8; ls20 L2 at 46 / 31 vs 12 / 30; s5i5 L2 at 38 / never
vs 1 / 2; cn04 L3 at 47 / never vs never / 25.

Where the hour went. Requests per hour were unchanged (arm 1790 / 1806, control 1702 / 1735) and so were python
calls (~885 per draw), but actions fell by a fifth (1378 / 1477 vs 1697 / 1850): the model wrote or rewrote `step`
in 64 / 58 of its python calls (7 %) and the checked actions executed one at a time behind it. Mismatch stops
(recount from the `MODEL MISMATCH` markers, unique: m1 86, m2 67; the judge's text counter, which looked for the
JSON-quoted key, printed 0 and is wrong in that column only) ended batches early in every game; the probe cap of 30
was hit in 3 games (m1) and 5 games (m2). Harness search outcomes over the 91 searches: plan found 20, "no reachable
state satisfies goal" 49, time limit (12 s) 18, no actions passed 4. Of the 20 harness runs: level completed 5,
stopped by a mismatch 13 (the model agreed with the record but not with the unseen part of the level), plan done
without a level-up 1 (goal wrong), refused 1. Host verification itself was cheap (287-302 s per draw, 2.7 s mean
search).

Reading. The LLM-written models were right on what they had seen (59 % near-exact) and wrong on what they had not:
half the searches proved "unreachable" under a model that was consistent with every recorded transition, and two
thirds of executed plans stopped at the first action that reached new ground. That is the SIMPLAN-2 result again
(opt-in then: 2-3 % uptake, 3 plans), now with the uptake forced to 90 %: the executor works when the model is
complete, and on these 8 games the model is complete for about one level in six. The checked-action regime also
charged the games the control solves by exploration (ls20, s5i5, cn04, bp35) with a probe cap and one-action
batches they never needed. Nothing here is a tuning target; the fail-closed verdict stands.
