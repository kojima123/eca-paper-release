# SIMPLAN's narrow hidden scores read as suppression (note, 2026-10-29)

**Observation** (user 10-29: 「simplan でバラツキが少ないけどスコアが下がる理由って結果的に抑制が働いてるのでは？」).

Hidden scores of the SIMPLAN-2b/2d submissions, against every other arm's draw on the same base
(FRZ-ECAOD + D' + host note + pinned levels; one failed 0.00 draw excluded on each side):

| | Draws | Mean | SD | Range |
|---|---|---|---|---|
| SIMPLAN-2b / 2d | 22.35, 22.04, 24.13, 18.91, 22.05 | 21.9 | 1.9 | 18.9-24.1 |
| Other arms (ECAOD, +D', +host note, +pin, +guards, s0, s1, s2) | 36.14, 32.60, 23.67, 30.55, 21.87, 26.60, 21.50, 26.89, 28.84, 29.58, 27.97, 20.35 | 27.2 | 4.8 | 20.4-36.1 |

Variance ratio 6.4 (F with 11 and 4 degrees of freedom; about 3 % under equal variances). The public canons do not
show the gap: SIMPLAN-2 115 and 2b 119 levels against the pin canon's 122.

**What SIMPLAN did in the runs.** Requests whose snippet called `verify_model` / `find_plan` / `run_plan`: 76 of
2304 (3 %) in the SIMPLAN-2 canon, 59 of 2467 (2 %) in 2b, 9 of 438 (2 %) in the 2d submission demo. So the
modelling procedure itself took almost no turns; the time budget was not spent in it.

**Reading.** The narrowing is consistent with suppression through two always-on channels, not through the procedure:
1. *Where the task fires.* The task line comes after 40 actions on a level without a level change, which is exactly a
   long exploration in progress. The high hidden draws of the other arms (32-36) are plausibly a few long explorations
   that paid off; interrupting every one of them removes the upper tail and leaves the mean a little lower.
2. *Constant prompt weight.* The tool descriptions and the task text ride on every request, so history is trimmed
   earlier and long coherent lines of play are harder to keep.

In this reading SIMPLAN suppressed the LLM's exploratory variance and replaced it with an LLM-written procedure that
rarely found plans (3 `find_plan` hits, one level through `run_plan`, see docs/simplan2_2026-10-17.md section 8), so
the result was "narrower and lower". It is a reading, not a measurement: five hidden draws cannot separate it from a
tight lower draw of a wide distribution, and the canons do not reproduce the gap.

**Why it matters for the surprise stop** ([surprise_registration_2026-10-29.md](surprise_registration_2026-10-29.md)).
The surprise stop also suppresses, but it suppresses LLM calls during known stretches and leaves the exploration
itself to the model; nothing LLM-dependent is put in the exploration's place. If the reading above is right, the
expected shape is "narrower like SIMPLAN, without the drop". The surprise judge therefore also reports the
per-game level spread against the controls (descriptive, not a verdict); hidden-score spread needs several
hidden draws and is the user's decision.
