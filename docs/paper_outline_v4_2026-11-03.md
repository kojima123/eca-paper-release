# Paper outline v4: verified routines and supervisory control for an LLM agent (2026-11-03 ~13:30Z)

Supersedes v3 (docs/paper_outline_v3_2026-11-03.md, 0efcd663), after the user's review of v3 (11-03). The user's framing is kept (the
bicycle; physical-AI middleware as the origin; the 1.86 history; the reason for not knowing can be analysed; "experience is reset, and
memory is read"; no persistent learning and bounded context). What changed is where each number is allowed to point. Every number
quoted here is reproduced by a script in tools/ from the run directories; the context and control-rate numbers by
tools/paper_measures.py -> research/paper_measures/measures_v1.json (new; v3 quoted them from inline computations).

## Changes from v3 (review 11-03; each item checked against the records)
| v3 | v4 | check |
|---|---|---|
| "a model of level k predicts k + 1 exactly 10 %" as evidence that the LLM does not carry experience over | moved to Limits as a result about the generalisation method tested: a host-side effect-template predictor, no LLM | transfer_v1.md: "No LLM, no Colab"; 445 of the 1,112 failures are effects outside its template class |
| "the machine changed" = all 1,112 failed level-transition predictions | "prediction failures at the level change", with the breakdown as candidate causes | same file: 457 new effect for the same key / 445 outside the model class / 117 misapplied / 93 unseen key |
| routines = "the LLM's own past solutions" | the agent's past successful trajectories (LLM and built-in solver; the source of the chosen path is not recorded) | library_harvest.py keeps per level the shortest verified candidate; library_v2: 45 of 46 levels were cleared by both the model and the solver in the source draws, s5i5 L7 by the model only |
| frontier "1 new level in 54 level-hours" | 1 new level in 20 game-runs, 19.1 game-hours after the replay | 54 = 14 + 14 + 13 + 13 levels left per draw, each counted as an hour; measures_v1.json frontier |
| R vs S 5 / 8 vs 3 / 8, stops 8 / 8 (interim) | R 11 / 16 vs S 7 / 16, stops 16 / 16 (final) | library_divergence_registration result |
| "fully observed" environment | deterministic, the board image observed; the rendered frame is not guaranteed to be a sufficient state | the census's broken keys contradict "fully observed" |
| 19.8 % broken = hidden state; suffix length = the context the hidden state needs | 19.8 % = the frame-only key (game, frame, action) does not determine the next frame in these records; k = the shortest history that separated the observed samples. Level-rule differences are now ruled out: with the level in the key every count is identical (no broken key pools two levels), so the ambiguity is within a level - state the last frame does not show; engine state vs action history not yet separated | closure_census.py keyed on (game, frame hash, action); re-run with CENSUS_STATE=level -> census_public_level.json |
| goal mining 0 / 114 = "the goal is unknown" | a specific completion test (two pieces meeting on the completing action) held on 0 of 114 clears; the NV25 planner skipped 20 of 28 levels for missing pieces / goals, planned 8, cleared 0 | eca_harness_repair_feasibility s4, s7.5 |
| ~75 k tokens re-read per turn "(median)"; "remembering costs as much as thinking" | 74.9 k mean / 77.6 k median prompt tokens per request; the cost comparison removed (cached prefix vs fresh input vs generation are not separated) | level_tokens_ecaoff.py divides the sum by the count (mean); measures_v1.json |
| 4.5 drains per game-hour (32 runs counted as 32 hours) | 4.66 per game-hour over the runs' own 30.9 hours | measures_v1.json context |
| "190x faster per action"; "free to execute" | per-action decision interval 0.24 s (replay, checks included) vs 46 s (an action that opens an LLM turn); 0 LLM tokens during a replay | not a task-level speed ratio; the check and the environment step remain |
| "no persistent learning ... every draw starts from zero" | the control agent keeps nothing from one run to the next in an executable form; frozen weights alone do not imply that (Voyager keeps a code-skill library without weight updates) | |
| novelty = "automaticity made exact and verified" | the cognitive mapping is the explanatory frame; the contribution is the execution contract and its measurement | MobiMem motivates its replay by procedural / "muscle" memory (s1); SkillDroid verifies the state at each replay step, hands back with the completed steps, recompiles failing skills; the Schema harness self-checks each step and returns to deliberation |
| "cannot help on a first encounter" | in these runs the stored routines and the existing families gave no level on the 33 unseen games | nv25_solver_coverage; within-run reuse remains possible |
| "a wrong certificate costs one action" (physical-AI frame) | the gate stops routine execution after an observed mismatch and hands back; it does not make the mismatching action harmless | Simplex hands control to a verified safety controller; we hand it to an unverified LLM |
| control vs library as the flag's effect | an observed system-level comparison: the library kernel also carries the host-note level fix and the warmup RESET fix | library_registration: "control configuration + ECA_FRZ_LIBRARY=1 + host-note level fix + warmup RESET fix" |

## Format constraint (docs/paper_submission_requirements_2026-11-03.md)
The Kaggle Writeup is limited to 1,500 words; this outline is the full paper (optional PDF), the Writeup its condensed form. The
linked submission (36.14 ECAOD v1) demonstrates the gate with certified families; the library layer and the R/S, census and advice
experiments are post-competition experiments on the submitted harness. Deadline: Kaggle 2026-11-09 23:59 UTC (arcprize.org: Nov 8).

## Working title
Riding without thinking: verified routines and supervisory control for an LLM agent on ARC-AGI-3

## Central sentence
Successful experience is kept as verifiable execution routines; where a routine holds, deliberation is skipped; at the first
counterexample control returns to deliberation. We measure the effect and the limits of applicability of that contract.

## Thesis
An LLM agent that re-derives every step is costly and run-to-run unreliable: in the same configuration it cleared 31 and 38 levels.
Human skill differs: practised routines run without deliberation and attention is recruited on novelty or when a routine's prediction
fails. We give an LLM agent that structure under one execution contract - a routine acts without the LLM only while every observed
transition matches its prediction, and hands back at the first mismatch, with the position - and measure where it pays (what was met
before: verified, 0 LLM tokens) and where it does not (in these runs, the 33 unseen games).

## Origin and frame: middleware for physical AI (user 11-03)
ECA started as middleware between a foundation model and the actuators of a physical system; ARC-AGI-3 is the testbed. A robot cannot
run LLM inference at every control step (latency, cost, run-to-run variance), and people do not think about each movement either.
ARC-AGI-3 is deterministic, the board image is observed and the score is external, so the middleware's governance can be measured:
what runs without the model, how exactly, at what cost, when control returns, and why. Measured per-action decision interval
(Colab G4, Qwen3.8 Flash-Next; measures_v1.json): library replay 0.24 s median (p90 1.09 s, n = 5,666, the hash check included); an
action that opens an LLM turn 46 s after the previous action (median; p90 227 s, n = 1,232; ctl1-2 alone 40 s), with 3.3 k generated
tokens per turn (median). Not a task-level speed ratio.
What carries over to physical systems, and what is not shown: the contract stops routine execution after an observed prediction
mismatch and hands back; here a wrong certificate was revoked after one action (2 cases), but that action had been executed - its
harmlessness is not shown, nor that the LLM recovers safely. A physical deployment needs, in addition, a pre-action admissibility
check and a safe hold / retreat controller (Simplex hands control to a verified safety controller; our hand-back goes to an unverified
deliberator). Continuous noisy states turn exact hashes into tolerance bands; open-loop replay fails under stochasticity, so the
per-action check is the part that carries over.

## History: the 1.86 submission (V6, July) - authority without words (user 11-03)
(Unchanged from v3: submission 54666986, 2026-07-14, public score 1.86, first on the public leaderboard at the time (the user, 11-03;
no snapshot in our records - attach one if available); family_lease with one certified action at a time, no ECA text; same-era single draws pure 0.90, advice
only 0.75, broad advisory 0.42, V6 rerun 1.10; local ladder advice 0 / 9 cells vs lease-4 6 / 6, advice x11.5 thinking tokens per
action. History, not a controlled result; experiment 2 (authority vs advice) re-tests it on the current harness - draws a1 / v1 done,
a2 / v2 running.)

## Flagship case: from a new success to a routine (s5i5 L7)
One trace shows the whole loop (measures_v1.json s5i5_L7_chain):
1. draw l2: after the replay of L1-L6, the model clears s5i5 L7, never cleared before in 85 draws: 65 actions, 19 LLM turns,
   224,901 generated tokens, 46.5 minutes (human baseline 86 actions);
2. harvest: the history is replayed on the local engine, cut to the level, shortened under replay verification to 61 actions, and
   stored with the start-board hash and the hash after every action (library v2);
3. draws l3 and l4: L1-L7 replayed, 226 actions, 0 generated tokens, every hash matched.
Then the general results: 182 / 182 level replays matched (45 x 2 draws on library v1, 46 x 2 on v2: known levels only); levels per
draw 31 / 38 (control, 34.5) -> 45 / 46 -> 46 / 46; generated tokens per draw 1.82 M -> 1.38 M -> 1.34 M. The library kernel also
carries the host-note level fix and the warmup RESET fix, so these are observed system-level differences; the replay itself is shown
directly (182 / 182 at 0 tokens).

## The execution contract (the system section)
Two routine sources, one contract:
| | instance replay (library) | mechanic families (ECA) |
|---|---|---|
| authority granted when | the start board's hash equals a stored start | the family's model is closed on every observed transition of the level (certificate) |
| prediction per action | the stored hash after the action | the family model's next board |
| revoked when | the observed hash differs | the observed board differs (first counterexample) |
| on revocation | stop, note the action and level to the LLM; the LLM continues from the current board | stop, failure note (model, breaking action) |
| measured | 182 / 182 replays; 16 / 16 registered divergences stopped where planted | 2 false certificates on NV25 revoked after action 1; public +10 levels over blind play |
What a hash match guarantees: in a deterministic game the same start and the same actions give the same frames as long as the hidden
state agrees; a replay either reproduces the stored frames or stops at the first frame that differs. It does not guarantee that a
hidden difference is caught before it shows. The contract unifies when a routine may act (closure / match on the evidence), when it
must stop (first observed mismatch) and what the deliberator receives (where and why).

## Hand-back after a deviation (completed)
R (the library level diverges after half its actions; the replay stops, the model continues) vs S (the level removed; the model
plays it from the start), 8 levels x 2 draws each: stops where registered 16 / 16; levels completed R 11 / 16 vs S 7 / 16 (registered
rule R - S >= 3: met); from the hand-back, model actions median 41 vs 117, tokens 112 k vs 157 k, minutes 28 vs 35; hand-backs that
left the board as it was 4 / 4 completed, the others 7 / 12; RESET after the hand-back 3 / 16. Supports the combination of partial
reuse, stopping and hand-back - not the hand-back alone, and not statistical significance (n = 16).

## The reason for not knowing can be analysed (user 11-03) - observed stop, candidate causes, what would separate them
| observed stop condition | candidate causes | information needed to separate | measured |
|---|---|---|---|
| no stored start, no family certifies | new game / level; perception or family coverage | the routine sources' coverage on the game | unseen games: 0 certified on 31 of 33; frontier: 1 new level in 19.1 game-hours |
| routine contradicted at action k | wrong routine; hidden state differs; dynamics changed | replay of the same prefix from the same start; the changed cells | NV25: 2 false certificates revoked after action 1; R arm: 16 / 16 stops at the registered action |
| same (frame, action), different next frame | state the last frame does not show (engine state; history dependence within a level); level-rule difference ruled out | engine state snapshot per step; held-out history test | 19.8 % of 13,300 repeated keys, unchanged with the level in the key (2,635 broken either way); history up to 6 actions separates 63 % of the broken keys in sample (10 with >= 2 observations per branch); held-out on other draws 0.837 vs 0.744 (n = 649, answers 40 %) |
| prediction fails at the level change | new effect for the same key; effect outside the model class; misapplied template; unseen key | richer effect language; object identity | 1,112 failures: 457 / 445 / 117 / 93 (effect-template predictor, no LLM) |
| transitions predicted, no plan reaches a completion | goal representation; object representation; unobserved transitions; search range | the goal test's coverage; planner skip reasons | completion test held on 0 / 114 clears; NV25 planner: 20 of 28 levels skipped (no pieces / goal), 8 planned, 0 cleared |
| hand-back delivered, next actions ignore it | supervisor attention; note wording | follow rate per note | repair requests taken up 1 / 5; SIMPLAN tools 2-3 % of turns; advice follow rate pending |
The claim is that each stop is recorded with its observation and the candidate causes are narrowed by named measurements - not that
the cause is identified uniquely.

## Discussion: solving is not consolidating; "experience is reset, and memory is read" (user 11-03)
Scope: the control agent in this harness (Qwen3.8 Flash-Next, no cross-run memory), not every LLM. Evidence:
- each request re-reads the record: 74.9 k prompt tokens per request (mean; median 77.6 k);
- what it solved costs again: without the solver the model spends 0.36-0.51 M tokens per draw (20-28 %) on the solver's known
  levels, and clears them 23 / 18 of 26 times; with routines those levels take 0 LLM tokens;
- deliberation without action: arm S, s5i5 L7: 17 and 48 minutes (100 k / 227 k tokens) for 8 and 10 actions;
- within a run it re-takes (frame, action) pairs whose outcome it has observed: 1.4 % of model actions on the public games, 62 % on
  NV25 (older harness).
Fitts & Posner / Anderson: the agent stays in the cognitive stage, interpreting a declarative record at every step; the routines layer
is a compilation path outside the LLM. The 10.3 % transfer result is not evidence here (it measured a host-side predictor).

## Why it does not consolidate: no cross-run retention in the control agent, bounded context (user 11-03)
1. No cross-run retention. The weights are fixed at test time, and the control agent has no path that keeps a run's experience in an
   executable form for the next run: each draw begins without the previous draws' experience (pre-training and the built-in solver
   remain). Frozen weights alone do not imply this - Voyager (Wang et al. 2023) keeps a library of executable code skills without
   weight updates; the routines layer adds such a path, exact and verified, without changing weights. Weight updates at test time
   are the other road (test-time training, Akyurek et al. 2024; SOAR 2025) with its classical problem (catastrophic interference,
   McCloskey & Cohen 1989; Kirkpatrick et al. 2017 - to verify).
2. Bounded context (measures_v1.json; ctl1, ctl2, f15, f16; 32 game-runs, 30.9 game-hours): the prompt reaches 116 k tokens (median
   just before a drain) and is drained 144 times - every 20 requests (median), 4.66 times per game-hour; afterwards experience survives
   as the harness's text summary and pinned sequences. Background, not a diagnosis of our failures: long contexts are used unevenly
   (Liu et al., Lost in the Middle, TACL 2024, multi-document QA and key-value retrieval). Symptoms in our logs: 20 turns searching the
   history for a played board (ar25 L4, failnote v6); re-taking recorded transitions (above).
| memory | persists | cost to use | exact | generalises | in our agent |
|---|---|---|---|---|---|
| weights | yes, fixed at test time | - | no | yes | the LLM's priors |
| context | within a run, drained 4.66x per game-hour | 75 k prompt tokens per request; 46 s per opening action | no (summaries) | in context | the record the LLM re-reads |
| routines (ours) | across runs | 0 LLM tokens; 0.24 s per action with the check | yes, up to visible frames | no (exact) / within a family | library, mechanic families |

## AGI perspective
Chollet (2019): intelligence as skill-acquisition efficiency. The runs separate finding a solution (the LLM sometimes does) from
keeping it as a routine (the control agent does not; the routines layer does, exactly, without generalising). Non-claims as in v3
(not "LLMs do not learn"; not level-wise efficiency - the per-level human / agent ratio does not fall).

## Related work and the contribution
Taken: fast / slow agents; switching to deliberation on prediction failure (Schema harness, Rodionov, AdaPlanner, WebATLAS); LLM-free
replay with per-step state verification, hand-back with the completed steps and recompilation (SkillDroid, arXiv 2604.14872); replay
motivated by procedural / "muscle" memory (MobiMem, arXiv 2512.15784, s1); executable skill libraries without weight updates
(Voyager). The cognitive mapping (Logan, Schmidt, Norman & Shallice, Wolpert, Daw, Fitts & Posner) is the explanatory frame, not the
novelty. Contribution:
1. one execution contract for two routine sources (exact cross-run replay keyed on hashes in a deterministic setting - no embedding,
   regex or LLM confirmation in the match; certified mechanic families), with grant, per-action check, revocation and hand-back
   defined the same way, and what it stopped (0 mismatches in 182 replays; 16 / 16 planted divergences; 2 false certificates);
2. measurements under a small model and competition constraints, with hidden scores (36.14 ECAOD v1, 30.68 failure note): reuse
   reliability, cost, hand-back, the run-to-run variance it removes on known levels, and its applicability limit;
3. the stop record: each stop with its observation, candidate causes and the measurement that would separate them.

## Scope and limits
Recurring tasks only: in these runs the stored routines and the existing families gave no level on the 33 unseen games (NV25 acting:
on 2 vs off 1 level, the extra one blind play). Within-run reuse is possible but measured small (re-played prefixes 0.3-0.4 % of model
actions). Generalisation: the effect-template predictor tested (no LLM) predicted the next level's transitions exactly on 10.3 % of
1,240 (37 level pairs); 445 of 1,112 failures were effects outside its class - the generalisation method is insufficient, which says
nothing about the LLM. Closure: the frame-only key is not sufficient on five of eight games, within levels (the level-keyed census is identical); the suffix repair answers 40 % of held-out
cases. Accuracy is the linked Kaggle submissions'.

## Sections
1 Introduction (problem: once solved is not reliably cheap next time; the bicycle) - 2 Background (automaticity and supervisory
control; closure, certificates) - 3 Related work - 4 Setting (ARC-AGI-3; harness; Kaggle constraints; unseen sets for measurement
only) - 5 The execution contract - 6 From success to routine (s5i5 L7) - 7 Results: replay, cost, hand-back - 8 Limits of
applicability: unseen games, state representation, generalisation - 9 Discussion: automaticity, consolidation and context, physical
AI - 10 Conclusion.
