# VLDB 2027 artifact — XQdrant

Self-contained experiment harness and PVLDB source for:

> Moetez Fradi. *XQdrant: In-Database Attribution and Masked-Distance Subspace Search over HNSW*. PVLDB, 20(1), 2027.

This folder mirrors the EDBT short-paper layout in [`../`](../) and adds the full-paper expansions (MiniLM/GIST, systems load, scale×ef, filters, SQ, cost model, projected ceiling, case study). Parent [`../`](../) remains the short-paper artifact.

**Published figures and numbers in [`paper/`](./paper/) are frozen from the 2026-08-04 submission PDF.** Scripts below can regenerate CSVs/plots; they do not overwrite `paper/figures/` unless you pass `--copy-paper-figures` / run `copy_paper_figures.py`.

Artifact URL (this branch): `https://github.com/Moetez-Fradi/XAI/tree/vldb27/DB_systems/VLDB%202027`

## Quick start

```bash
cd "DB_systems/VLDB 2027"
./fetch_sift.sh
# optional: ./fetch_gist.sh && MINILM_N=20000 ./fetch_minilm.sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# offline pipeline check (no servers)
./run_unified_vldb_bench.sh --smoke --simulated

# live (needs release binaries under ../../qdrant and ../../XQdrant)
./run_unified_vldb_bench.sh --smoke          # tiny HTTP
./run_unified_vldb_bench.sh --expansion      # paper §5.1 expansion protocol N=20k Q=40 T=5
```

Compile the paper:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Paper → code map

| Paper | § | Script | Figure |
|-------|---|--------|--------|
| Algorithm 1 | 3 | typeset in `paper/main.tex` | Fig 1 |
| Cost model | 4.3 | [`research/cost_model/plot_cost_model.py`](research/cost_model/plot_cost_model.py) | Fig 2 |
| Attribution latency–recall | 5.2 | `bench_suite.py` Test A | Fig 3 |
| Attribution depth | 5.2 | `bench_suite.py` Test C | Fig 4 |
| Focus rescore | 5.2 | `bench_suite.py` Test D | Fig 5 |
| Systems load | 5.2 | [`research/systems_load/run_systems_load.py`](research/systems_load/run_systems_load.py) | Fig 6 |
| M1 SIFT | 5.3 | [`research/option1_naive_masked/run_option1_recall_collapse.py`](research/option1_naive_masked/run_option1_recall_collapse.py) | Fig 7 |
| MiniLM / GIST M1 | 5.3 | [`research/cross_corpus/run_minilm_gist_m1.py`](research/cross_corpus/run_minilm_gist_m1.py) | Fig 8, Table 2 |
| M2 SIFT / D=768 | 5.4 | [`research/option2_hybrid_layer_cutoff/run_option2_layer_sweep.py`](research/option2_hybrid_layer_cutoff/run_option2_layer_sweep.py) | Figs 9–11 |
| Scale × ef | 5.5 | [`research/scale_ef/run_scale_ef.py`](research/scale_ef/run_scale_ef.py) | Fig 12 |
| Focus constructions | 5.6 | [`research/focus_baselines/run_focus_baselines.py`](research/focus_baselines/run_focus_baselines.py) | Fig 13 |
| K1 kernels | 5.7 | [`research/xcut1_gather_vs_repack/plot_xcut1_kernel_bench.py`](research/xcut1_gather_vs_repack/plot_xcut1_kernel_bench.py) | Fig 14 |
| C1 coherence | 5.8 | [`research/xcut2_subspace_coherence/run_xcut2_coherence.py`](research/xcut2_subspace_coherence/run_xcut2_coherence.py) | Figs 15–16 |
| X4 Jaccard | 5.8 | [`research/xcut4_divergence/run_xcut4_divergence.py`](research/xcut4_divergence/run_xcut4_divergence.py) | Fig 17 |
| V1 verify | 5.9 | [`research/xcut3_verify_pass/run_xcut3_verify.py`](research/xcut3_verify_pass/run_xcut3_verify.py) | Fig 18 |
| M3 blend | 5.10 | [`research/option4_weighted_blend/run_option4_alpha_sweep.py`](research/option4_weighted_blend/run_option4_alpha_sweep.py) | Fig 19 |
| Filter × M2 | 5.11 | [`research/filter_m2/run_filter_m2.py`](research/filter_m2/run_filter_m2.py) | Fig 20 |
| SQ characterization | 5.12 | [`research/quantization_sq/run_sq_char.py`](research/quantization_sq/run_sq_char.py) | Fig 21 |
| Projected ceiling | 5.13 | [`research/option3_projected_index/run_option3_projected_index.py`](research/option3_projected_index/run_option3_projected_index.py) | Fig 22 |
| Case study | 6 | [`research/case_study/run_marco_case.py`](research/case_study/run_marco_case.py) | Fig 23 |

## Protocols

| Protocol | N | Q | T | Use |
|----------|---|---|---|-----|
| `--smoke` | 512 | 8 | 1 | HTTP/field check |
| `--expansion` (paper §5.1 new panels) | 20k | 40 | 5 | filters, SQ, load, MiniLM/GIST, case study, … |
| `--full` | SIFT 1M / MiniLM 100k | 500 | 5 | optional long job |

Engines: vanilla Qdrant `:6335`, XQdrant `:6333`, `XQDRANT_MASKED_KERNEL=gather`. Build:

```bash
# protobuf compiler (prost-wkt-types). A local copy lives at repo-root tools/protoc/:
export PROTOC="$PWD/../../tools/protoc/bin/protoc"
export PROTOC_INCLUDE="$PWD/../../tools/protoc/include"
cargo build --release --bin qdrant     # repo-root qdrant/
cargo build --release --bin xqdrant    # repo-root XQdrant/
```

## Layout

```
VLDB 2027/
├── README.md                     ← you are here
├── artifacts/README.md           ← public artifact entry
├── paper/                        ← PVLDB source + frozen figures
├── research/                     ← one folder per claim
├── experiments/                  ← timestamped CSV/plots (gitignored)
├── run_unified_vldb_bench.sh     ← download-to-plots orchestrator
├── fetch_sift.sh / fetch_gist.sh / fetch_minilm.sh
└── bench_*.py                    ← shared HTTP harness
```
