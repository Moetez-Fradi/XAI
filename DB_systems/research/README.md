# `research/` — per-mechanism runners

One subfolder per mechanism. Directory names keep the older `option*` / `xcut*`
labels used in experiment IDs. The VLDB 2027 paper's copies of these runners, with its
paper → figure map, live in [`../VLDB 2027/research/`](../VLDB%202027/research/).

Shared helpers: [`common_research.py`](./common_research.py) (focus request bodies,
HTTP client that records unsupported fields, simulated mode, plot helpers).

Parent entry point: [`../README.md`](../README.md).

## Mechanism map

| Mechanism | Folder | Runner |
|-----------|--------|--------|
| Attribution / focus rescore | [`step0_baseline_done/`](./step0_baseline_done/) | root `bench_suite.py` Tests A–D |
| **M1** | [`option1_naive_masked/`](./option1_naive_masked/) | `run_option1_recall_collapse.py` |
| **M2** | [`option2_hybrid_layer_cutoff/`](./option2_hybrid_layer_cutoff/) | `run_option2_layer_sweep.py` |
| **K1** | [`xcut1_gather_vs_repack/`](./xcut1_gather_vs_repack/) | Criterion bench + `plot_xcut1_kernel_bench.py` |
| **C1** | [`xcut2_subspace_coherence/`](./xcut2_subspace_coherence/) | `run_xcut2_coherence.py` |
| **V1** | [`xcut3_verify_pass/`](./xcut3_verify_pass/) | `run_xcut3_verify.py` |
| **M3** | [`option4_weighted_blend/`](./option4_weighted_blend/) | `run_option4_alpha_sweep.py` |
| Projected index | [`option3_projected_index/`](./option3_projected_index/) | `run_option3_projected_index.py` |
| Visited-set divergence | [`xcut4_divergence/`](./xcut4_divergence/) | `run_xcut4_divergence.py` |

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
├── option3_projected_index/          # future work
└── xcut4_divergence/                 # visited-set divergence
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
claim are listed in the per-mechanism READMEs.

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

Unified same-host re-run of the main suite:

```bash
../run_unified_paper_bench.sh          # from DB_systems/
```
