# Within-run transfer, offline (Track 2, 2026-11-02): can a host-side model fit on levels 1..k play level k+1?

Data: the solved-level library (library_v1.json, 45 levels, 8 games), replayed on the local engine (tools/transfer_frames.py ->
transfer_frames_v1.pkl). Tool: tools/transfer_measure.py. Results: transfer_v1.json (plan tier: dynamics exact >= 0.8),
transfer_v1_loose.json (plan tier >= 0.5). No LLM, no Colab, generic code (no game names or per-game constants).

## What was reused and what was not
- eca_loop_v1 is not usable as is on these games. `ObjectWorld` needs an LLM-written rule (`init`/`step` code) and its goal is
  hard-coded as two pieces meeting (merge), the m0r0 L5/L6 end. The coarse-grid perception (`grid_perception`) fits these
  boards poorly: the combined detector's off-boundary edge share is 0.13-0.73 on six of the eight games (bp35 0.65, dc22 0.73,
  ls20 0.43, re86 0.43). The motion and combined detectors disagree on the cell size for dc22 (2 vs 6) and re86 (3 vs 4). The
  motion detector finds no grid for the click games s5i5 and tn36.
- Instead, the same ECA-table idea was rebuilt on pixel objects with the harness's `board_digest` (components, object_changes,
  effect_signature, completion_facts). For each action key (a click is keyed by the clicked object's colour and size, falling
  back to colour), a table holds the most common effect template:
  - nothing;
  - rigid bodies (an object plus what lies in its box) moved by a displacement, with the colours they were seen to enter and the
    colour they leave behind;
  - a recolour.
  The model is fit only on levels 1..k.

## Results per game (all k -> k+1 pairs pooled; "exact" = predicted board equals the real one inside the 4-cell border)

| game | pairs | transitions | exact (model) | object-level | exact (nothing changes) | goal recall | goal exact (P=R=1) | plans |
|---|---|---|---|---|---|---|---|---|
| ar25 | 7 | 231 | 0.017 | 0.030 | 0.000 | 0/7 | 0 | - |
| bp35 | 3 | 94 | 0.011 | 0.224 | 0.011 | 0/3 | 0 | - |
| cn04 | 5 | 159 | 0.000 | 0.000 | 0.000 | 4/5 | 1 (L3) | - |
| dc22 | 3 | 153 | 0.170 | 0.170 | 0.013 | 3/3 (P 0.11-0.17) | 0 | - |
| ls20 | 3 | 137 | 0.686 (0.57 / 0.78 / 0.72) | 0.686 | 0.022 | 2/3 | 2 (L3, L4) | 2 searched, 0 found |
| re86 | 5 | 238 | 0.000 | 0.000 | 0.000 | 0/5 | 0 | - |
| s5i5 | 5 | 147 | 0.020 | 0.020 | 0.020 | 2/5 | 1 (L4) | - |
| tn36 | 6 | 81 | 0.000 | 0.000 | 0.000 | 5/6 | 5 (L2-L6, filled relation) | - |
| all | 37 | 1240 | 0.103 | 0.122 | 0.007 | 16/37 | 9 | 0 found |

Why the 1,112 wrong predictions are wrong:
- 457: level k+1 shows another effect than levels 1..k for the same key (other objects, shapes or displacements).
- 445: the effect is outside the model class. Examples: tn36's program run, cn04's rotations and assembly, ar25's later reflections
  with several objects, re86.
- 117: the same template was applied wrong. Examples: bp35's several identical tiles; dc22's floor under the mover.
- 93: the action key had never been seen in levels 1..k.

Plans: no pair reaches the strict tier (dynamics exact >= 0.8 together with an exact goal). At the loose tier (>= 0.5), only ls20
L3 and L4 qualify:
- L3: the search exhausted the model's 240 reachable states without a plan.
- L4: 6,028 states in 180 s, no plan.

The goal there is the completion event itself ("B changed shape; b vanished"), and a dynamics model fit on non-completing
transitions never produces it. The reachable space is also cut to the colours the piece entered in levels 1..k.

## Reading
- **Goals transfer more often than dynamics.** The end-facts goal (the filled relation, or effect terms common to all earlier
  completions) is exact on 9 of 37 pairs: tn36 L2-L6, cn04 L3, ls20 L3-L4 and s5i5 L4. It is high-recall but imprecise on dc22.
- **Dynamics transfer only for ls20**, the one plain keyboard mover (0.57-0.78 exact). Elsewhere they are at or near the
  nothing-changes baseline.
- **The model never finds a plan.** Where dynamics and goal both hold, the goal is stated as a completion event that the model
  cannot produce. A state goal (where the piece must stand, what must be filled) would be needed.
- **Caveat on the data.** The dynamics were fit on the library's shortened successful sequences only (no failed attempts). A live
  run has more transitions per level, but the two largest causes (another template in the next level; effects outside the model
  class) are not data-volume problems.
