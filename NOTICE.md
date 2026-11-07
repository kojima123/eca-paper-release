# Notice

Everything in this repository is original work by the author (with AI coding assistance) and is released under
MIT-0 (see LICENSE), except as stated here.

**Not included.** Third-party code and content are not redistributed here:
- the ARC-AGI-3 games and their source files (ARC Prize Foundation; obtained through the public ARC-AGI-3 toolkit);
- Daniel Franzen's milestone-2 Kaggle notebook and the Tufa Labs Duck harness (ARC3-Inference, TAAF), on which the
  agent runs;
- raw run outputs (prompts, model outputs, frames), which are large and contain game frames.

**The runtime (`runtime/`).** `runtime/arcgym` is the author's ECA package; it contains no code from the
RGB-Agent framework (the five files the private repository had inherited were rewritten for this release; see
runtime/README.md). `runtime/harness/*.patch` are diffs against the port of Franzen's milestone-2 notebook (Tufa Labs
Duck harness); the lines they add are the author's and are released under MIT-0, while the context and removed lines
they quote belong to that base, which keeps its own terms and is not redistributed here.

**Dependencies.** Some analysis tools import the public `arc_agi` / `arcengine` packages (to replay games locally)
or modules of the patched harness (`inference.framework.*`), which needs the base notebook. Those tools are
included so the computations behind each reported number can be read; running them needs the runtime and the
recorded runs. The tools that read only recorded run directories (for example `tools/refine_chain_v2.py`) run with
the Python standard library.
