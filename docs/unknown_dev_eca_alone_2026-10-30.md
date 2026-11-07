# ECA alone on 8 unknown games (arcint), CPU only (2026-10-30 ~14:00Z)

No GPU is available (Colab compute units 2.80: G4 and A100 refused with 503 since 10:05Z, H100 not entitled; Kaggle
GPU quota also used up, user 10-30). This check needs none: the local integration harness (changelog_check_v1,
arm `on`, the kernel's ECA env, the 2026-09-28 RWS arcgym bundle) with a fake model that only inspects, so every action
taken is ECA's own certified-prefix executor. Games: the census set of
[unknown_dev_census_registration_2026-10-30.md](unknown_dev_census_registration_2026-10-30.md) (ab01 as01 av01 bi01
bl01 bn01 bn02 bn03), 120 requests each.

| | public (same harness, 10-30 local, 200 requests) | unknown arcint 8 |
|---|---|---|
| levels completed by ECA alone | re86 4, tn36 5, ar25 3, s5i5 2-3, cn04 2 | 0 on all 8 |
| actions ECA executed | 39-146 per game | 0 on all 8 |
| certified families | shape_marker, program_execution, reflection_orbit, linked_extent, anchor_overlap | none on any game (19 evaluations each) |

Reading: the certified families are the public games' mechanics; on games they were not written for they certify
nothing, so on unknown games ECA contributes no actions and no levels - the same picture as the NV25 records
(eca_harness_repair_feasibility_2026-10-26.md section 7). Its cost there is the evaluation time only. Raising the
per-family timeout from 750 to 3000 ms changed one public level (s5i5 L3) and cannot change this.
