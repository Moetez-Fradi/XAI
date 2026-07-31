# Bioinformatics — XQdrant explainable protein retrieval

ESM2 embeddings, Qdrant `dims_explained` attribution, and Experiments A–F for the PSB paper track.

## Navigation

| Path | Contents |
|------|----------|
| [`scripts/`](./scripts/README.md) | Pipeline and experiment Python modules |
| [`plot_scripts/`](./plot_scripts/README.md) | Paper figures + PyMOL session generators |
| [`configs/`](./configs/README.md) | YAML configs per stage/experiment |
| [`data/`](./data/README.md) | Raw downloads, processed corpus, annotations |
| [`embeddings/`](./embeddings/README.md) | ESM2 vectors (paper, smoke, ablation) |
| [`results/`](./results/README.md) | Experiment metrics and checkpoints |
| [`paper/`](./paper/README.md) | Per-experiment notes (methods, headline numbers) |
| [`Deliverables/`](./Deliverables/README.md) | PSB 2027 LaTeX manuscript + `make` |
| [`tools/`](./tools/README.md) | mkdssp, Foldseek, BLAST+, PyMOL (local install) |
| [`steps.md`](./steps.md) | Original paper plan and checklist |

## Quick start

```bash
cd Bioinformatics
./setup_env.sh              # optional: ./setup_env.sh --cuda
./run_downloads.sh          # paper corpus; --pilot for ~200 PDBs
./fetch_tools_linux.sh      # or fetch_tools_macos.sh
source .venv/bin/activate
```

**Smoke (safe, no holdout lock):**

```bash
./run_smoke.sh
./start_xqdrant.sh --daemon
./run_exp_a_smoke.sh
./run_exp_b_v3.sh --smoke
```

**Full paper pipeline:**

```bash
.venv/bin/python scripts/build_corpus.py
.venv/bin/python scripts/embed_esm2.py          # prefer --device cuda
./start_xqdrant.sh --daemon
.venv/bin/python scripts/index_xqdrant.py
.venv/bin/python scripts/exp_a_retrieval.py
./run_exp_a_baselines.sh
./run_exp_b_v3.sh
./run_exp_c.sh
./run_exp_d_downloads.sh && ./run_exp_d_finalize.sh
./run_exp_e.sh
./run_exp_f.sh --complete
./run_exp_ablation.sh --device cuda
./plot_scripts/run_all.sh
make -C Deliverables figures && make -C Deliverables
```

Export bio-tools PATH when needed:

```bash
export PATH="$PWD/tools/mamba/envs/bio-tools/bin:$PATH"
source tools/env_linux.sh   # after fetch_tools_linux.sh
```

## Shell runners (`run_*.sh`)

| Script | Purpose |
|--------|---------|
| `run_downloads.sh` | All paper downloads (`configs/download.yaml`) |
| `run_smoke.sh` | Tiny corpus + smoke embed |
| `run_exp_a_smoke.sh` | Smoke index + Exp A |
| `run_exp_a_baselines.sh` | BLAST / Foldseek / TM / ESM2 cosine |
| `run_exp_b_v3.sh` | **Exp B (paper)** — attribution validity |
| `run_exp_c.sh` | ROC / negative controls |
| `run_exp_d_downloads.sh` | OMA structures + merged corpus |
| `run_exp_d.sh` / `run_exp_d_finalize.sh` | Exp D thermo case study |
| `run_exp_e.sh` | Active-site localization |
| `run_exp_f.sh` | Workflow timing (`--indexed`, `--complete`) |
| `run_exp_ablation.sh` | Layer/pooling ablation |
| `run_exp_b_smoke.sh`, `run_exp_b_v2_smoke.sh` | **historical** — v1/v2 only |

Full Exp A (paper): `index_xqdrant.py` + `exp_a_retrieval.py` (no `run_exp_a.sh` wrapper).

## Other entry points

| Script | Purpose |
|--------|---------|
| `setup_env.sh` | Python venv + `requirements.txt` |
| `start_xqdrant.sh` | Local XQdrant fork (`../XQdrant`) |
| `fetch_tools_linux.sh` / `fetch_tools_macos.sh` | mkdssp, Foldseek, BLAST+, PyMOL conda |

## Repositories (paper)

- Experiments: https://github.com/Moetez-Fradi/XAI/tree/bioinformatics-track/Bioinformatics  
- XQdrant fork: https://github.com/Moetez-Fradi/XQdrant/tree/e91022ab36c724d2110b32f004ab688fee062af0  

## Download size (paper mode)

See [`configs/download.yaml`](./configs/download.yaml). Rough totals: **~8–15 GB** download, **~10–18 GB** on disk before embeddings/indexes. Full corpus embedding and indexes need substantially more free space (~30 GB+ recommended).

## Experiment status (paper)

| Exp | Results | Doc |
|-----|---------|-----|
| A | `results/exp_a/` | [paper/experiment_a.md](./paper/experiment_a.md) |
| B v3 | `results/exp_b_v3/` | [paper/experiment_b.md](./paper/experiment_b.md) |
| C | `results/exp_c/` | [paper/experiment_c.md](./paper/experiment_c.md) |
| D | `results/exp_d/` | [paper/experiment_d.md](./paper/experiment_d.md) |
| E | `results/exp_e/` | [paper/experiment_e.md](./paper/experiment_e.md) |
| F | `results/exp_f/` | [paper/experiment_f.md](./paper/experiment_f.md) |
| Ablations | `results/exp_ablation/` | [paper/experiment_ablation.md](./paper/experiment_ablation.md) |

Manuscript: [`Deliverables/xqdrant_psb2027.pdf`](./Deliverables/xqdrant_psb2027.pdf) — build with `make -C Deliverables`.
