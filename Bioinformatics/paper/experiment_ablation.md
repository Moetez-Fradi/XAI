# Experiment ablations — ESM2 layer + pooling + top-N dims

## Top-N dimensions (Exp B v3 — already complete)

Configured in `configs/exp_b_v3.yaml` (`ablation_top_dims: [1,3,5,10,32]`).  
Results: `results/exp_b_v3/metrics.json` → `ablations`.  
Plot: `plot_scripts/plot_exp_b_v3_validity.py` → `exp_b_v3_ablation_topn.pdf`.

## Layer + pooling (Exp ablation)

Re-embeds ~2,283 chains (150 Exp A queries + neighbors + 800 train for map) under:

| Config | Layer | Pooling |
|--------|-------|---------|
| layer33_mean | last | mean (extracted from full corpus) |
| layer16_mean | 16 | mean |
| layer8_mean | 8 | mean |
| layer33_cls | last | cls |

Scores attribution fold-gap (same-fold vs diff-fold Spearman) on Exp A pairs.

## Run

```bash
cd Bioinformatics
source .venv/bin/activate

# Full pipeline (~2–4h CPU / ~30–60min GPU for 3 embed configs)
./run_exp_ablation.sh --device cuda

# After embeddings exist, re-score only:
./run_exp_ablation.sh --score-only

# Figures
python plot_scripts/plot_exp_ablation.py
```

Artifacts: `results/exp_ablation/metrics.json`, `ablation_comparison.tsv`.

## Exp A mAP (added)

```bash
python scripts/recompute_exp_a_metrics.py
# → updates results/exp_a/metrics.json with fold_map / superfamily_map / family_map
```

Paper fold mAP ≈ **0.896** (with R@1 0.898).
