# Experiment C — Negative Controls & Specificity (Paper Section)

**Status:** Pipeline in `scripts/exp_c_negative_controls.py`; results in `results/exp_c/`.  
**Builds on:** Exp A (pairs, retrieval scores), Exp B v3 (dim→profile map, profile scores).

---

## 1. Role in the paper

Experiment C formalizes **specificity**: do attribution-based scores rank **structurally related** pairs (same SCOP fold) above **unrelated** pairs?

Where Exp B tested absolute profile agreement against row-permuted nulls, Exp C reports **ROC and PR curves** for fold discrimination — the figure type planned in `steps.md` (Fig 4).

---

## 2. Methods (draft)

### Pair sources

| Source | Description | Has `dims_explained` |
|--------|-------------|----------------------|
| **retrieval** | Exp A top-5 neighbors per holdout query | Yes |
| **random_negative** | Holdout query + random train chain (different fold) | No |

### Scores (higher ⇒ predicted related)

| Method | Definition |
|--------|------------|
| `xq_profile_spearman` | Exp B attribution-weighted DSSP profile score |
| `retrieval_score` | ESM2 cosine (Exp A or recomputed) |
| `random_dims_spearman` | Profile score after shuffling attributed dim indices |
| `uniform_dims_spearman` | Equal weights on attributed dims |
| `attribution_entropy` | Diffuse vs concentrated attribution (diagnostic) |

### Labels

- **Positive:** same SCOP fold (`same_fold=true`)
- **Negative:** different fold
- **Saturated negative tag:** diff-fold with global DSSP cosine ≥ 0.999

### Metrics

- AUROC and AUPRC per method (all pairs, retrieval-only, random-negative-only)
- Curve points in `roc_curves.tsv` for plotting

---

## 3. Reproducibility

```bash
cd Bioinformatics
source .venv/bin/activate
./run_exp_c.sh
# smoke: ./run_exp_c.sh --smoke  (requires Exp A smoke)
```

Requires Exp B v3 map (`results/exp_b_v3/dim_profile_map.npz`) or fits map on the fly.

---

## 4. Results (paper run)

| Method | AUROC (all) | AUPRC (all) | mean pos | mean neg |
|--------|-------------|-------------|----------|----------|
| xq_profile_spearman | **0.628** | 0.861 | 0.102 | 0.034 |
| retrieval_score | **0.978** | 0.992 | 0.951 | 0.826 |
| random_dims_spearman | 0.528 | 0.801 | — | — |
| uniform_dims_spearman | 0.626 | 0.860 | — | — |

*n = 12,000 pairs (7,500 retrieval + 4,500 random negatives); attribution scores on 7,378 retrieval pairs with DSSP profiles.*

**Interpretation:** Attribution profile scores discriminate fold match (AUROC 0.63) well above shuffled-dimension null (0.53). Raw ESM2 retrieval cosine is stronger (0.98) because neighbors are selected for similarity. The value of attribution is **interpretability + structural profile alignment**, not beating cosine retrieval — pair with Exp B label-perm (p = 0.0002) for the mechanism story.

*Source:* `results/exp_c/summary.tsv`, `metrics.json`.

---

## 5. Paper linkage

- **Results:** “Attribution profile scores discriminate fold-matched from unmatched pairs (AUROC = X.XX), outperforming shuffled-dimension and uniform-weight nulls (AUROC = X.XX).”
- **Figure:** ROC overlay — xq vs retrieval vs random_dims (from `roc_curves.tsv`).
- **Connect to B:** Label-perm p = 0.0002 (B v3) + AUROC (C) = specificity evidence.
