# Paper figure scripts (Experiments A–E)

Generate all figures from completed experiment results. Outputs go to `plot_scripts/outputs/`.

## Quick start

```bash
cd Bioinformatics
chmod +x plot_scripts/run_all.sh
./plot_scripts/run_all.sh
```

Or run individually:

```bash
.venv/bin/python plot_scripts/plot_exp_a_retrieval.py
.venv/bin/python plot_scripts/plot_exp_b_v3_validity.py
.venv/bin/python plot_scripts/plot_exp_c_roc.py
.venv/bin/python plot_scripts/plot_exp_d_thermo.py
.venv/bin/python plot_scripts/generate_pymol_exp_d.py
.venv/bin/python plot_scripts/plot_exp_e_functional.py
.venv/bin/python plot_scripts/generate_pymol_exp_e.py
```

## Outputs

| Script | Figures |
|--------|---------|
| `plot_exp_a_retrieval.py` | `exp_a_retrieval_recall.png/pdf` — R@1/5/10 vs baselines |
| `plot_exp_b_v3_validity.py` | `exp_b_v3_violin_spearman`, `exp_b_v3_label_perm`, `exp_b_v3_ablation_topn`, `exp_b_v3_same_vs_diff_means` |
| `plot_exp_c_roc.py` | `exp_c_roc.png/pdf` — ROC for XQ profile vs controls |
| `plot_exp_d_thermo.py` | `exp_d_scatter_*`, `exp_d_forest_significance`, `exp_d_source_breakdown`, `exp_d_mean_deltas` |
| `generate_pymol_exp_d.py` | `outputs/pymol/exp_d_groel_session.pml` + charged variant |
| `plot_exp_e_functional.py` | `exp_e_site_enrichment.png/pdf` — XQ vs TM site localization |
| `generate_pymol_exp_e.py` | `outputs/pymol/exp_e_session.pml` + attribution/site CSVs |

## PyMOL rendering (manual)

Requires PyMOL (`pymol` or `pymol-open-source`):

```bash
cd Bioinformatics
pymol plot_scripts/outputs/pymol/exp_d_groel_session.pml
# In PyMOL:
#   png plot_scripts/outputs/pymol/exp_d_groel_rsa.png, 2400, 1800

pymol plot_scripts/outputs/pymol/exp_d_groel_charged_session.pml
#   png plot_scripts/outputs/pymol/exp_d_groel_charged.png, 2400, 1800
```

## Prerequisites

- Completed results: `results/exp_a/`, `exp_b_v3/`, `exp_c/`, `exp_d/`, `exp_e/`
- Python deps: `matplotlib` (in `requirements.txt`)
- PyMOL: optional, for structure panels only

## Input paths (defaults)

| Experiment | Primary inputs |
|------------|------------------|
| A | `results/exp_a/baselines/comparison.tsv` |
| B v3 | `results/exp_b_v3/pairs.json`, `summary.tsv` |
| C | `results/exp_c/roc_curves.tsv`, `summary.tsv` |
| D | `results/exp_d/metrics.json`, `significance.tsv` |
| D PyMOL | `results/exp_d/metrics.json`, `data/annotations/dssp/*.json` |
| E | `results/exp_e/localization_comparison.tsv`, `metrics.json` |
| E PyMOL | `data/processed/exp_e/sites/`, Exp A `dims_explained` |
