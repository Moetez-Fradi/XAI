# `research/` — Next-step analysis scripts (masked-traversal study)

Option/step-specific analysis code for the plan in [`../../steps.md`](../../steps.md). The
**reusable** suite library stays at the `DB_systems/` root (`bench_common.py`, `bench_backends.py`,
`bench_viz.py`, `bench_suite.py`, ...). This folder holds the **non-reusable, per-step** code, one
subfolder per option/metric, so each experiment is self-contained and reproducible.

`common_research.py` is the only shared piece here: run-folder naming, request-body builders for the
proposed focus knobs, an HTTP client that records unsupported fields instead of crashing, a simulated
model for offline pipeline checks, and plot helpers.

## Layout

```
research/
├── common_research.py                # shared helpers (reusable within research/)
├── step0_baseline_done/              # pointer: which reusable code produced A–E
├── option1_naive_masked/             # Step 1  — no Rust change (focus.masked ships)
├── option2_hybrid_layer_cutoff/      # Step 2  — needs focus.mask_from_layer
├── option3_projected_index/          # Step 3  — needs projected-index API (ceiling measurable now)
├── option4_weighted_blend/           # Step 4  — needs focus.alpha
├── xcut1_gather_vs_repack/           # X1 — Rust criterion bench + plot
├── xcut2_subspace_coherence/         # X2 — offline, no server
├── xcut3_verify_pass/                # X3 — needs focus.verify
└── xcut4_divergence/                 # X4 — needs visited-node logging (proxy works now)
```

## Result folders (self-describing + timestamped)

Every script writes to:

```
DB_systems/experiments/<YYYY-MM-DD_HH-MM-SS>__<step_id>__<metric>/
├── manifest.json     # step, metric, mode, URLs, dims, sweep, requires_xqdrant_change
├── results/*.csv
└── plots/*.png|pdf
```

So a folder name like `2026-07-09_10-15-00__option2_hybrid_layer_cutoff__recall_latency_layer_heatmap`
tells you the step, the metric, and when it ran — no guessing later.

**Which folders to cite for this study:** see
[`../experiments/README.md`](../experiments/README.md) (canonical vs superseded runs,
one-page story, cheat sheet). Design writeups: [`../../xqdrant_docs/README.md`](../../xqdrant_docs/README.md).

## Which step needs a Rust change vs. just a script

| Step | Rust change | Script | Metric |
|------|-------------|--------|--------|
| 1 Option 1 | none (ships) | `option1_naive_masked/run_option1_recall_collapse.py` | recall_vs_ratio |
| 2 Option 2 | `focus.mask_from_layer` | `option2_hybrid_layer_cutoff/run_option2_layer_sweep.py` | recall_latency_layer_heatmap |
| 3 Option 3 | projected-index API | `option3_projected_index/run_option3_projected_index.py` | recall_memory_buildtime |
| 4 Option 4 | `focus.alpha` | `option4_weighted_blend/run_option4_alpha_sweep.py` | recall_speed_alpha |
| X1 | Rust criterion bench | `xcut1_gather_vs_repack/plot_xcut1_kernel_bench.py` | kernel_gather_vs_repack |
| X2 | none (offline) | `xcut2_subspace_coherence/run_xcut2_coherence.py` | subspace_coherence |
| X3 | `focus.verify` | `xcut3_verify_pass/run_xcut3_verify.py` | verify_recall_recovery |
| X4 | visited-node logging | `xcut4_divergence/run_xcut4_divergence.py` | traversal_divergence |

## General usage

Every script supports `--mode simulated` (offline, fabricated data tagged `simulated=1`,
plot titles prefixed `[SIMULATED]`) for validating the pipeline before the Rust change lands, and
`--mode http` for live runs after it lands. Runnable-now steps (X2 coherence, Option 3 ceiling,
X4 proxy, Option 1) need no XQdrant change.

```bash
cd DB_systems
source .venv/bin/activate            # or: python3 with requirements.txt installed
python research/option1_naive_masked/run_option1_recall_collapse.py --mode simulated --queries 50
```

If XQdrant does not yet understand a proposed field, that config is recorded as
`status="unsupported"` (non-zero `unsupported_queries`, `NaN` cells) — the signal that the
corresponding Rust change is still pending. Field names are defined once in
`common_research.focus_body`; keep them in sync with the Rust API.
