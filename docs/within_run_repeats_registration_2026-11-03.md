# Automaticity within a first encounter: how much the model re-derives what it already played in the same run (registered 2026-11-03 ~10:05Z, before the tool ran on any data)

User 11-03: the hidden ARC-AGI-3 evaluation is a first encounter, the expected weak domain of automaticity; asked to proceed
(「おねがいします」) with the offline measure proposed in outline v3 (scope section): within one run a level restarts from the same
board after RESET or GAME OVER, and the model may re-play (and re-think) a prefix it has already played. Exact, verified replay of
the run's own prefix would act there without the LLM - on unseen games too.

**Data.** First-encounter play only (no library): the 83 local draws of research/solver_library/draws_v1.txt (8 public games) and the
SIMPLAN-2e NV25 run (25 unseen games; measurement only). Histories replayed on the engines (as tools/closure_census.py).
**Definitions** (tools/within_run_repeats.py). An *attempt* = the actions on a level from its start board until the level ends, a
RESET, or GAME OVER (the next attempt starts again from the level's start board). For each attempt after the first on a level: the
*repeated prefix* = the longest common prefix of its actions with any earlier attempt on that level (same start board, so the same
states - verified by the frame hashes along the prefix). Turn cost: generated tokens and wall time attach to the turn's first action;
a turn is *repeated* when all its actions lie in a repeated prefix. Also: *known transitions* = (state, action) pairs already taken
earlier in the same run (any attempt), whose outcome the run already observed.
**Measures** (per game and pooled, public and NV25 separately): restarts; repeated-prefix actions as a share of all model actions;
repeated turns' tokens and seconds as a share of the run's; known-transition actions share.
**Prediction** (weak prior): repeated-prefix actions 5-20 % of model actions, their turns a smaller share of tokens (prefixes are
often batched); known transitions a larger share (revisits without restarts). Reading: the upper bound of what within-run automaticity
could save on first encounters - an answer to "it cannot help on ARC-AGI-3" in part, or a measured confirmation that it cannot.
Nothing is changed in any harness; never Kaggle.

## Result (11-03 ~10:30Z; research/within_run/within_{public,nv25}.json)
| set | model actions | restarts | repeated-prefix actions | known-transition actions | repeated turns' tokens / time |
|---|---|---|---|---|---|
| public, 83 draws (Oct harness) | 137,604 | 535 | **574 (0.42 %)** (bp35 4.1 %, others < 1 %) | **1,889 (1.37 %)** (bp35 5.7 %) | 0.7 % / 0.7 % |
| NV25, SIMPLAN-2e run (Oct 17 harness) | 7,661 | 17 | **24 (0.31 %)** | **4,749 (62 %)**: rs0427 97 %, al7306 93 %, ma4173 92 %, td4826 90 %, cc2048 88 %, hr2048 86 % ... | n/a (the run's history has no token counts) |
Predictions: repeated prefixes 5-20 % - **not met** (0.3-0.4 %: the model seldom restarts, so replaying its own prefix after a restart
would save little); known transitions a larger share - met, and on the unseen games by far: 62 % of the model's actions re-took a
(state, action) whose outcome the run had already observed - mostly actions that change nothing, pressed again and again (rs0427:
6 of 381 actions changed the board, eca_harness_repair_feasibility s7.3).
Reading: within a first encounter the opportunity is not "replay what worked" but "do not re-ask what is known": the model does not
use its own observed outcomes, it re-reads the record and tries again. On the public games under the current harness (no-op stops,
change log) this is 1.4 %; on the unseen games under the older harness it was the majority of all actions. The two harnesses differ,
so the 62 % is the older stack's figure, not the current one's (the 10-30 arcint census under the current harness: FULLKEY, one game).
