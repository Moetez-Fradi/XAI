# `research/` — VLDB 2027 per-mechanism runners

See the parent [README](../README.md) for the paper → figure map.

Shared helpers: [`common_research.py`](./common_research.py).

Short-paper mechanisms (copied, paths retargeted to this folder's `experiments/`):

| Paper | Folder | Runner |
|-------|--------|--------|
| M1 | `option1_naive_masked/` | `run_option1_recall_collapse.py` |
| M2 | `option2_hybrid_layer_cutoff/` | `run_option2_layer_sweep.py` |
| K1 | `xcut1_gather_vs_repack/` | `plot_xcut1_kernel_bench.py` |
| C1 | `xcut2_subspace_coherence/` | `run_xcut2_coherence.py` |
| V1 | `xcut3_verify_pass/` | `run_xcut3_verify.py` |
| M3 | `option4_weighted_blend/` | `run_option4_alpha_sweep.py` |
| X4 | `xcut4_divergence/` | `run_xcut4_divergence.py` (M1+M2 Jaccard) |
| Ceiling | `option3_projected_index/` | `run_option3_projected_index.py` (vs M2) |

VLDB-only expansions:

| Paper | Folder | Runner |
|-------|--------|--------|
| Fig 2 | `cost_model/` | `plot_cost_model.py` |
| Fig 6 | `systems_load/` | `run_systems_load.py` |
| Fig 8 / Table 2 | `cross_corpus/` | `run_minilm_gist_m1.py` |
| Fig 12 | `scale_ef/` | `run_scale_ef.py` |
| Fig 13 | `focus_baselines/` | `run_focus_baselines.py` |
| Fig 20 | `filter_m2/` | `run_filter_m2.py` |
| Fig 21 | `quantization_sq/` | `run_sq_char.py` |
| Fig 23 | `case_study/` | `run_marco_case.py` |

Every script accepts `--mode simulated` (except C1, which is offline NumPy, and the cost-model/K1 plotters).
