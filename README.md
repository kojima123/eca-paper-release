# When Can an Agent Act Without Thinking? — records and analysis code

Companion repository for the ARC Prize 2026 Paper Track paper
*When Can an Agent Act Without Thinking? Execution contracts and stop-record diagnostics for an LLM agent on
ARC-AGI-3* (Yuto Kojima). The paper is in `docs/paper/` (English `paper.pdf`, Japanese `paper_ja.pdf`, figures in
`docs/paper/fig/`, the Kaggle Writeup in `writeup.md`).

This repository holds what the paper's Appendix A points to: the pre-registration and result documents, the
analysis tools that produced each reported number, and their outputs; and, in `runtime/`, the ECA package and the
ECA layer on Franzen's milestone-2 notebook as patches (runtime/README.md). It does not contain the games, the
base notebook or the raw runs (see NOTICE.md).

| Experiment or analysis (paper section) | Registration / record | Tool and output |
|---|---|---|
| Library (5.3, 5.4) | docs/library_registration_2026-11-03.md | tools/library_harvest.py; research/solver_library/ |
| Authority, advice, consistency (5.2, 5.1) | docs/eca_advice_registration_2026-11-03.md | tools/advice_analysis.py; research/eca_advice/ |
| Deviation (5.5) | docs/library_divergence_registration_2026-11-03.md | tools/library_divergence.py |
| Action interval, context, unsolved levels (5.1, 5.6) | docs/paper_outline_v4_2026-11-03.md | tools/paper_measures.py; research/paper_measures/ |
| Unseen games (5.6) | docs/nv25_solver_coverage_2026-11-03.md; docs/unknown_dev_eca_alone_2026-10-30.md | tools/nv25_solver_coverage.py |
| Closure census (6) | docs/closure_census_registration_2026-11-03.md | tools/closure_census.py; research/closure_census/ |
| Within-run repetition (5.6, 7.1) | docs/within_run_repeats_registration_2026-11-03.md | tools/within_run_repeats.py |
| Solver disabled (1, 5.1) | docs/ecaoff_registration_2026-11-02.md | tools/level_tokens_ecaoff.py |
| Level-to-level transfer (6) | research/solver_library/transfer_v1.md | tools/transfer_measure.py |
| Repair after a stop (5.7) | docs/refine_chain_registration_2026-11-05.md | tools/refine_chain_v2.py (supersedes refine_chain.py); research/refine_chain/ |
| LLM-written models (5.7) | docs/modeler_registration_2026-10-31.md; docs/simplan_suppression_note_2026-10-29.md | tools/judge_modeler_colab.py; docs/modeler_colab_verdict_2026-10-31.txt |
| Figures | — | tools/paper_figures.py |

Registrations were written before the corresponding analysis or run; results and later corrections are appended
to the same file with their date (for example the v2 addendum in the refine-chain registration).

`research/solver_library/library_v*.json` are the solved-level libraries: per public-game level, the action
sequence, the start-frame hash and the frame hash after every action.

Linked Kaggle submissions: 56836939 (ECAOD v1) and 57877794 (failure note).

Licence: MIT-0 (LICENSE). Citation: CITATION.cff (a request, not a licence condition).
