# ECA runtime

The code that ran with the agent in the paper, in two parts.

**`arcgym/`** — the ECA package (import name `arcgym`) as shipped to the Kaggle kernels (source snapshot of
2026-09-28 used by the submitted and local kernels), reduced to the modules the harness imports and their
dependencies (203 modules): the gate, the 15 mechanic families (`eca/tufa_mechanics/`), certificates, perception
and search. Five small files that the private research repository had inherited from the RGB-Agent framework
(`arcgym/__init__.py`, `core.py`, `environments/__init__.py`, `environments/arcagi3/arcagi3_env.py`,
`utils/grid_utils.py`) were rewritten for this release with the same interface. Equivalence was checked against
the inherited versions: identical outputs on 3,000 random grids for the grid helpers, identical observations,
rewards and end flags over 657 steps on 8 public games for the environment, and the same import result for all
203 modules (one adapter, `adapters/tr87_perception_driven`, needs the ARC-AGI-3 game sources and is inert
without them).

**`harness/`** — the ECA layer on Daniel Franzen's milestone-2 notebook, as diffs against the port of that notebook
(Tufa Labs Duck 5297846 + Franzen's harness changes; base commit `bdb3c20`):

| patch | branch tip | used for |
|---|---|---|
| `eca-on-franzen-m2_master.patch` | c6cb4cc (2026-10-05) | the ECA port as of the ECAOD v1 period |
| `eca-on-franzen-m2_failnote-s0.patch` | 3caf79e (2026-11-02) | submission 57877794 (failure note) |
| `eca-on-franzen-m2_ecaod-changelog.patch` | 3b53c3a (2026-11-03) | the post-competition local runs (library, advice, deviation) |

Every ECA behaviour is behind an `ECA_FRZ_*` environment flag, default off. The base notebook is not included;
the patches apply to it and it remains under its authors' terms. The runtime finds this package through
`ECA_TUFA_ARCGYM_ROOT` (the directory that contains `arcgym/`).
