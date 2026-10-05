# `research/` — per-mechanism runners (paper order)

One subfolder per claim in the paper. Directory names keep the older
`option*` / `xcut*` labels used in experiment IDs; **paper names** (M1, M2, …)
are what you want when matching [`../../deliverable/`](../../deliverable/).

Shared helpers: [`common_research.py`](./common_research.py) (focus request bodies,
HTTP client that records unsupported fields, simulated mode, plot helpers).

Parent entry point: [`../README.md`](../README.md).

## Paper map (read this first)

| Order | Paper | Paper § | Folder | Runner | Verdict |
|------:|-------|---------|--------|--------|---------|
| 0 | Attribution / focus rescore | §5.2 | [`step0_baseline_done/`](./step0_baseline_done/) | root `bench_suite.py` Tests A–D | Keep attribution; rescore not accel. |
| 1 | **M1** + **C1** | §5.3 | [`option1_naive_masked/`](./option1_naive_masked/) (also [`xcut2_subspace_coherence/`](./xcut2_subspace_coherence/)) | `run_option1_recall_collapse.py` | Keep (full-space recall = coherence) |
| 2 | **M2** | §5.4 | [`option2_hybrid_layer_cutoff/`](./option2_hybrid_layer_cutoff/) | `run_option2_layer_sweep.py` | Optional (≤ +0.04 over M1) |
| 3 | **K1** + in-engine latency | §5.5 | [`xcut1_gather_vs_repack/`](./xcut1_gather_vs_repack/), [`inengine_checks/`](./inengine_checks/) | `masked_kernel_bench` (engine fork `6cb79ea`), `run_server_time.py`, `run_ratio1_kernel_cost.py`, `run_kernel_inengine.py` | Skip gather |
| 4 | **V1** | §5.6 | [`xcut3_verify_pass/`](./xcut3_verify_pass/) | `run_xcut3_verify.py` | Keep with overfetch |
| 5 | **M3** | §5.7 | [`option4_weighted_blend/`](./option4_weighted_blend/) | `run_option4_alpha_sweep.py` | **Skip** |

Not reported in the paper (future work):

| Folder | Why it is here |
|--------|----------------|
| [`option3_projected_index/`](./option3_projected_index/) | Future work from §6–7 (projected HNSW per focus set) |
| [`xcut4_divergence/`](./xcut4_divergence/) | Visited-set overlap idea; not reported |

Design docs (same names): [`../../xqdrant_docs/`](../../xqdrant_docs/).

## Layout

```
research/
├── README.md                         ← this file
├── common_research.py
├── step0_baseline_done/              # pointer → root suite (attribution + Test D)
├── option1_naive_masked/             # M1
├── option2_hybrid_layer_cutoff/      # M2
├── xcut1_gather_vs_repack/           # K1
├── xcut2_subspace_coherence/         # C1
├── xcut3_verify_pass/                # V1
├── option4_weighted_blend/           # M3 (skip)
├── inengine_checks/                  # server-side latency, ratio-1.0 kernel cost, D=768 baseline
├── option3_projected_index/          # future work
└── xcut4_divergence/                 # future work (not reported)
```

## Result folders

Every script writes:

```
DB_systems/experiments/<YYYY-MM-DD_HH-MM-SS>__<step_id>__<metric>/
├── manifest.json
├── results/*.csv
└── plots/*.png|pdf
```

`experiments/` is gitignored (regenerable). Canonical folder names for each
claim are listed in the per-mechanism READMEs and mirrored into
[`../deliverable/figures/`](../../deliverable/figures/) for the PDF.

## Usage

```bash
cd DB_systems
source .venv/bin/activate   # or: pip install -r requirements.txt

# offline smoke (fabricated / tagged simulated)
python research/option1_naive_masked/run_option1_recall_collapse.py \
  --mode simulated --queries 50

# live (needs XQdrant with the relevant focus.* fields)
python research/option2_hybrid_layer_cutoff/run_option2_layer_sweep.py \
  --mode http --xqdrant-url http://127.0.0.1:6333 --dataset sift1m \
  --queries 500 --trials 5
```

`--mode simulated` validates the pipeline without a server. `--mode http` is
what the paper reports. Field names for `focus.*` live in
`common_research.focus_body` — keep them aligned with the XQdrant REST/gRPC API.

Unified same-host re-run of the main paper suite:

```bash
../run_unified_paper_bench.sh          # from DB_systems/
```
