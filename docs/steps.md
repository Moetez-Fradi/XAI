# XQdrant Masked-Traversal Research — Execution Plan (`steps.md`)

Master checklist for turning the design options in [`steps_vdbt.md`](./steps_vdbt.md) into
runnable experiments. It records, for each step: **what it is**, **whether it needs a Rust
change to XQdrant or is just a Python analysis script**, **which analysis script runs it**,
**what metric it produces**, and **the keep/skip criterion**.

The Python analysis code lives in [`DB_systems/research/`](./DB_systems/research/). Every run
writes a **timestamped, self-describing result folder** so provenance is never lost:

```
DB_systems/experiments/<YYYY-MM-DD_HH-MM-SS>__<step_id>__<metric>/
├── manifest.json     # step, metric, mode, URLs, dims, sweep params, requires_xqdrant_change
├── results/*.csv
└── plots/*.png|pdf
```

---

## The one-paragraph answer (Rust vs. script)

The current `focus.masked` code (working.md Milestone 3) is **already Option 1** — it masks all
HNSW layers. So of the eight research items below, **~6 need a one-time Rust change** to add a
new knob and thread it through the API (`REST → gRPC proto → QueryEnum/QueryVector → raw_scorer`,
the same path `focus.masked` already took), and **2 need no Rust at all**. Everything else —
sweeps, plots, ablations, tuning — is Python that hits the HTTP API. **You touch Rust once per
mechanism; you re-run the Python scripts as many times as you like.**

| # | Step | Rust change? | Analysis script | Metric |
|---|------|--------------|-----------------|--------|
| 0 | Baseline suite A–E (already done) | done | `research/step0_baseline_done/` (pointer) | latency/recall/QPS/attribution/rescore/masked |
| 1 | Option 1 — Naive full-masked | **No** (already built) | `research/option1_naive_masked/run_option1_recall_collapse.py` | recall_vs_ratio |
| 2 | Option 2 — Hybrid layer cutoff | **Yes** — add `mask_from_layer` | `research/option2_hybrid_layer_cutoff/run_option2_layer_sweep.py` | recall_latency_layer_heatmap |
| 3 | Option 3 — Auxiliary projected index | **Yes** (most invasive) | `research/option3_projected_index/run_option3_projected_index.py` | recall_memory_buildtime |
| 4 | Option 4 — Weighted blend `alpha` | **Yes** — add `alpha` | `research/option4_weighted_blend/run_option4_alpha_sweep.py` | recall_speed_alpha |
| X1 | Cross-cut — SIMD gather vs. repack kernels | **Yes** (Rust criterion bench) | `research/xcut1_gather_vs_repack/plot_xcut1_kernel_bench.py` | kernel_gather_vs_repack |
| X2 | Cross-cut — Subspace coherence diagnostic | **No** (offline NumPy) | `research/xcut2_subspace_coherence/run_xcut2_coherence.py` | subspace_coherence |
| X3 | Cross-cut — Fallback / verify pass | **Yes** — add `verify` | `research/xcut3_verify_pass/run_xcut3_verify.py` | verify_recall_recovery |
| X4 | Cross-cut — Divergence instrumentation | **Yes** (logging) | `research/xcut4_divergence/plot_xcut4_divergence.py` | traversal_divergence |

**Rust modification episodes: ~6** (Options 2, 3, 4; cross-cuts X1, X3, X4).
**Python-only work: everything else**, plus a plot script for each item above.

---

## Suggested order

1. **Cross-cut kernels (X1)** — repack scratch buffer already exists in `MaskedMetricQueryScorer`;
   add the gather-vs-repack criterion bench so all later options ride on the faster kernel.
2. **Option 1 (Step 1)** — no Rust needed; run the recall-collapse sweep to get the motivating baseline.
3. **Option 2 (Step 2)** — the priority result (headline Figure 4 candidate).
4. **X2 coherence** — run offline anytime; explains *why* the recall curves look the way they do.
5. **Option 4 (Step 4)** — cheap add-on to Option 2's best config.
6. **X3 verify pass** — optional safety mode; measure recall recovered vs. cost.
7. **X4 divergence** — instrument during Option 1/2 runs; no separate experiment.
8. **Option 3 (Step 3)** — only if a concrete small focus-set palette exists; else future work.

---

## Step detail

### Step 0 — Baseline suite (DONE)
- **Status:** already run; CSVs/plots exist under `DB_systems/experiments/2026-07-07_http_baseline/`.
- **Code:** the reusable suite `bench_suite.py` → `bench_tests.py` (Tests A–E) → `bench_viz.py`.
- **Reproducibility pointer:** `research/step0_baseline_done/README.md` maps each metric to its
  test function, CSV, and plot.

### Step 1 — Option 1: Naive full-masked traversal
- **Rust:** none — `focus.masked = true` already masks all layers.
- **Script:** `run_option1_recall_collapse.py` sweeps `D_sub/D ∈ {0.1, 0.25, 0.5, 0.75, 1.0}` at
  fixed `ef_search`, measures **Recall@10 vs. full-vector ground truth** and latency speedup.
- **Metric:** `recall_vs_ratio`.
- **Keep/skip:** keep as motivating baseline regardless; note the crossover ratio where recall
  drops below ~0.8.

### Step 2 — Option 2: Hybrid full-coarse / masked-bottom
- **Rust:** add `mask_from_layer: usize` to `DimsFocus` (REST + gRPC), thread to the scorer, branch
  distance selection on `current_layer < mask_from_layer` in the traversal loop.
- **Script:** `run_option2_layer_sweep.py` — 2D sweep `mask_from_layer × D_sub/D`; emits a
  recall heatmap + latency heatmap.
- **Metric:** `recall_latency_layer_heatmap`.
- **Keep/skip:** keep any `mask_from_layer` where latency improves and recall stays within
  ~0.01–0.02 of the full-vector baseline → new Figure 4.

### Step 3 — Option 3: Auxiliary projected index
- **Rust (largest):** new index-build path per registered focus set; query routing to the
  projected index; registration/selection API surface.
- **Script:** `run_option3_projected_index.py` — compares projected-index recall (the ceiling)
  against Option 2's best config at matching `D_sub/D`; records extra memory + build time.
- **Metric:** `recall_memory_buildtime`.
- **Keep/skip:** keep only if there is a small, known palette of focus sets; else future work.

### Step 4 — Option 4: Weighted blend distance
- **Rust:** add `alpha: f32` to `DimsFocus`; scorer computes `α·d_full + (1-α)·d_focus`
  (pairs with Option 2's cutoff for real latency gains).
- **Script:** `run_option4_alpha_sweep.py` — sweep `α ∈ {0,0.25,0.5,0.75,1.0}` × `mask_from_layer`.
- **Metric:** `recall_speed_alpha`.
- **Keep/skip:** keep if blending lets you mask more aggressively while staying accurate; else drop.

### Cross-cut X1 — SIMD gather vs. repack kernels
- **Rust:** add a `criterion` benchmark comparing gather-based vs. repack-scratch kernels at
  `D_sub ∈ {32,128,384}`; have it emit `kernel_bench.csv`.
- **Script:** `plot_xcut1_kernel_bench.py` reads that CSV and plots throughput vs. `D_sub`.
- **Metric:** `kernel_gather_vs_repack`.

### Cross-cut X2 — Subspace coherence diagnostic
- **Rust:** none — pure offline NumPy on the corpus.
- **Script:** `run_xcut2_coherence.py` — correlation between full-space and subspace k-NN sets
  across ratios; predicts which `D_sub/D` are "safe".
- **Metric:** `subspace_coherence`.

### Cross-cut X3 — Fallback / verify pass
- **Rust:** add `verify: bool` to `DimsFocus`; optional final full-distance rescore of top-k.
- **Script:** `run_xcut3_verify.py` — masked with/without verify; recall recovered vs. added latency.
- **Metric:** `verify_recall_recovery`.

### Cross-cut X4 — Candidate-list divergence instrumentation
- **Rust:** log/return how often the masked visited-node set diverges from the full traversal.
- **Script:** `plot_xcut4_divergence.py` — reads the emitted divergence CSV and plots divergence
  onset vs. `D_sub`.
- **Metric:** `traversal_divergence`.

---

## Proposed REST field names (must match the Rust change)

The analysis scripts send these fields inside `query.nearest.focus`. When you implement each Rust
change, use these names (or update the scripts' body builders in `research/common_research.py`):

| Field | Type | Used by | Step |
|-------|------|---------|------|
| `masked` | bool | already implemented | 1 |
| `mask_from_layer` | int | Option 2 | 2 |
| `alpha` | float | Option 4 | 4 |
| `verify` | bool | verify pass | X3 |

Until a field is implemented, its script records `status="unsupported"` per config (it will not
crash) so you can see exactly which Rust change is still pending.
