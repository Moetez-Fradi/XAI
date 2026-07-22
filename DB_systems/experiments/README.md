# `experiments/` — local run archives

Timestamped CSV / plot outputs from the harness. **This directory is gitignored**
(regenerable). Canonical figures that ship with the PDF live in
[`../../deliverable/figures/`](../../deliverable/figures/).

## Naming

```
<YYYY-MM-DD_HH-MM-SS>__<step_id>__<metric>/
├── manifest.json
├── results/*.csv
└── plots/*.png|pdf
```

`<step_id>` matches the **folder name** under `research/` (e.g.
`option2_hybrid_layer_cutoff`), not the paper M/K/C/V label. Use
[`../README.md`](../README.md) to translate.

## Where to find the paper claims

| Paper | Typical `step_id` in folder names | See README under |
|-------|-----------------------------------|------------------|
| Attribution / Fig. 2–4 | `paper_attr_unified_t*` or root suite stamp | [`../research/step0_baseline_done/`](../research/step0_baseline_done/) |
| M1 | `option1_naive_masked` | [`../research/option1_naive_masked/`](../research/option1_naive_masked/) |
| M2 | `option2_*` | [`../research/option2_hybrid_layer_cutoff/`](../research/option2_hybrid_layer_cutoff/) |
| K1 | `xcut1_gather_vs_repack` | [`../research/xcut1_gather_vs_repack/`](../research/xcut1_gather_vs_repack/) |
| C1 | `xcut2_subspace_coherence` | [`../research/xcut2_subspace_coherence/`](../research/xcut2_subspace_coherence/) |
| V1 | `xcut3_*` | [`../research/xcut3_verify_pass/`](../research/xcut3_verify_pass/) |
| M3 | `option4_weighted_blend` | [`../research/option4_weighted_blend/`](../research/option4_weighted_blend/) |

Regenerate the short-paper suite:

```bash
cd ..
./run_unified_paper_bench.sh
```
