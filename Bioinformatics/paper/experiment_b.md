# Experiment B — Dimension-Attribution Validity (Paper Section)

**Status:** Locked results in `results/exp_b/` (v1), `results/exp_b_v2/` (v2), `results/exp_b_v3/` (v3).  
**Primary results for the paper:** **v3** (`results/exp_b_v3/`). v1/v2 are historical baselines only.  
**Companion experiment:** C (ROC/specificity) — see `paper/experiment_c.md` when available.

---

## 1. Role in the paper

Experiment B tests whether XQdrant’s **dimension-level attributions** (`dims_explained`) reflect **independent structural ground truth** from DSSP, not merely embedding geometry.

**Pipeline (per query–neighbor pair from Exp A):**

1. Extract top-32 contributing embedding dimensions and weights from XQdrant.
2. Fit a **train-only** map **M** from embedding dimensions → structural features (never using holdout queries).
3. Form an attribution-weighted prediction: `pred = a · M` where `a` is the normalized contribution vector.
4. Compare `pred` to DSSP-derived ground truth for the pair and test against null models.

Exp A already showed retrieval works (fold R@1 = 0.898) and attribution ranks are stable. Exp B asks: *do the attributed dimensions mean something structurally?*

---

## 2. Methods (for Methods section)

### 2.1 Pairs and splits

- **Queries:** 1,500 holdout chains (SCOP-stratified; locked in `corpus_lock.json`).
- **Neighbors:** top-k retrieved chains from Exp A with non-empty `dims_explained` (k = 5 in v1/v2; k = 10 in v3).
- **Leakage control:** dim→feature map **M** fit only on `split=train` chains; holdout appears only as queries or hits in pairs, never in map training.

### 2.2 Ground truth (three iterations)

| Version | Ground truth | Profile dimension |
|---------|--------------|-----------------|
| **B v1** | Six global DSSP aggregates per chain (`frac_helix`, `frac_sheet`, `frac_coil`, `mean_hydrophobicity`, `mean_rsa`, `mean_b_factor`) | 6 |
| **B v2** | Length-normalized **binned profile**: 32 bins × 5 features (helix/sheet/coil/hydrophobicity/RSA) | 160 |
| **B v3** | Same as v2; larger map (all train chains with profiles); additional null tests | 160 |

For each pair (query **q**, hit **h**), ground-truth **agreement** per feature dimension:

\[
\text{agreement}_f = 1 - \min\left(1,\ \frac{|p_{q,f} - p_{h,f}|}{\text{range}_f}\right)
\]

where `range_f` is the train-set span of feature `f`. Pair score = Spearman(`pred`, `agreement`) (v2/v3 primary); Pearson also computed (v1 primary).

### 2.3 Dim→feature map **M**

- **v1/v2:** Random subsample of 3,000 train chains with DSSP annotations.
- **v3:** All train chains with binned profiles on disk (**n = 7,661**).
- **Map:** Column-wise Pearson correlation between z-scored embedding matrix and z-scored feature matrix: `M[d,f] = corr(E_{\cdot,d}, F_{\cdot,f})`.

### 2.4 Attribution vector **a**

From XQdrant `dims_explained` (top-32 dims, non-negative contributions):

- Take absolute contributions, zero-pad to 1,280 dims, **L1-normalize** to sum to 1.
- Optional ablation: keep only top-N dims before normalization (v3: N ∈ {1,3,5,10,32}).

### 2.5 Statistical tests

| Test | Null hypothesis | Used in |
|------|-----------------|---------|
| **Global row-permutation** | Mean pair score ≤ chance if dim→feature rows of **M** are shuffled | v1, v2, v3 |
| **Label-permutation (fold gap)** | Same-fold minus diff-fold mean score ≤ chance if fold labels shuffled | v3 |
| **Same-fold row-permutation** | Mean score on same-fold pairs ≤ row-shuffled **M** null | v3 |
| **Mann–Whitney U** | Same-fold scores ≤ diff-fold scores | v2, v3 |
| **Negative control** | Saturated diff-fold pairs (global DSSP cosine ≥ 0.999, different fold) score as high as homologs | v2, v3 |

Permutation count: 1,000 (v1/v2) or 5,000 (v3). Right-tail p-values: `(#{null ≥ real} + 1) / (N + 1)`.

### 2.6 Attribution concentration (v3)

Because one dimension often dominates (~90% of mass on top-1 dim), v3 reports mean top-1 mass and entropy of **a** across pairs.

---

## 3. Results (for Results section)

### 3.1 Summary table (primary cohort, all pairs)

| Metric | B v1 | B v2 | B v3 |
|--------|------|------|------|
| Pairs | 7,378 | 7,378 | 10,235 |
| Train map size | 3,000 | 3,000 | 7,661 |
| Mean pair score (Spearman) | — | **0.103** | 0.069 |
| Mean pair score (Pearson, v1) | 0.070 | — | — |
| Global row-perm p | 0.275 | 0.168 | 0.201 |
| Label-perm fold-gap p | — | — | **0.0002** |
| Same-fold row-perm p | — | — | 0.240 |
| Mean score same-fold | 0.080 | **0.122** | 0.083 |
| Mean score diff-fold | 0.028 | **0.028** | 0.037 |
| Mann–Whitney p (same vs diff) | — | **1.8×10⁻⁷⁸** | **5.3×10⁻⁵⁸** |
| AUROC (same-fold label) | — | 0.657 | 0.600 |

*Artifact paths:* `results/exp_b/metrics.json`, `results/exp_b_v2/metrics.json`, `results/exp_b_v3/metrics.json`.

### 3.2 Key findings

**Finding 1 — Specificity (supported).**  
Attribution-derived profile scores **separate SCOP fold-matched pairs from unmatched pairs** (Mann–Whitney p ≪ 0.05; label-permutation gap p = 0.0002 in v3). Same-fold pairs score higher than diff-fold pairs (v2 gap ≈ 0.094 Spearman; v3 gap ≈ 0.046 with noisier top-10 neighbors).

**Finding 2 — Absolute correlation null (not supported).**  
Shuffling rows of **M** does **not** produce a significantly lower mean pair score than the real map (global row-perm p ≈ 0.17–0.28 across v1–v3). The real mean (~0.07–0.10) is only modestly above the row-null (~−0.01 to 0).

**Finding 3 — Negative control (supported).**  
Pairs that are **different fold** but **saturated in global DSSP cosine** (≥ 0.999) score low (v2: 0.027 vs same-fold 0.122; v3: 0.035 vs 0.094), indicating attributions track fold-level structure rather than generic secondary-structure similarity alone.

**Finding 4 — Concentrated attributions (diagnostic).**  
In v3, **~70%** of pairs have top-1 dimension mass ≥ 0.9; ablations show top-1 vs top-32 scores differ but neither passes row-perm. Explanations are effectively **low-dimensional** in the current XQdrant cosine breakdown.

### 3.3 Suggested text (Results draft)

> **Attribution validity against DSSP profiles.** We evaluated whether XQdrant’s top-32 dimension attributions predict independent structural similarity between retrieved pairs (Experiment B). Using a train-only map from ESM2 dimensions to DSSP-derived binned structural profiles (32×5 features), attribution-weighted predictions were compared to profile agreement via Spearman correlation (7,378–10,235 query–neighbor pairs from holdout retrieval). Row-permutation of the dim→profile map did not yield significant enrichment (p = 0.17–0.28 across protocol versions), indicating that absolute profile agreement from attributions is only weakly above a shuffled-map null. In contrast, fold-matched pairs scored significantly higher than fold-mismatched pairs (Mann–Whitney p < 10⁻⁵⁷; label-permutation gap p = 0.0002), and saturated negative controls (different fold, global DSSP cosine ≥ 0.999) remained near chance. Attribution mass was highly concentrated (mean top-1 fraction 0.90). **Interpretation:** dimension attributions encode **fold-specific** structural relatedness more reliably than fine-grained absolute profile reconstruction; specificity is developed further in Experiment C (ROC).

---

## 4. Limitations (for Discussion)

1. **Correlational map **M**** is not causal; row-perm failure may reflect map weakness rather than attribution noise alone.
2. **`dims_explained` are non-negative** and often **single-dimension dominated**, limiting interpretability as a multi-feature explanation.
3. **DSSP coverage:** binned profiles available for ~7.6k train chains; top-10 neighbors increased missing-profile warnings in v3.
4. **Holdout** is used in pairs; only **M** training is strictly train-only — pair scores themselves are computed on holdout-involved pairs by design (retrieval evaluation).

---

## 5. Figures and tables

| ID | Content | Source |
|----|---------|--------|
| **Table B1** | v1/v2/v3 headline metrics (above) | `exp_b_v3/v2_vs_v3_comparison.tsv` + v1 metrics |
| **Fig B1** | Distribution of pair Spearman scores: same-fold vs diff-fold (violin/box) | `exp_b_v2/checkpoints/pairs.jsonl` or v3 |
| **Fig B2** | Row-perm null distribution vs real mean (v2 primary) | Replot from `metrics.json` null fields |
| **Fig B3** | Attribution concentration: top-1 mass histogram (v3) | `exp_b_v3/metrics.json` → `attribution_diagnostics` |
| **Fig B4** | *(Moved to Exp C)* ROC: attribution score vs fold match | `results/exp_c/` |

---

## 6. Reproducibility

```bash
cd Bioinformatics
source .venv/bin/activate
export PATH="$PWD/tools/mamba/envs/bio-tools/bin:$PATH"

# v2 (locked paper run)
.venv/bin/python scripts/annotate_dssp.py --from-exp-a --also-train 3000 --store-profiles
.venv/bin/python scripts/exp_b_v2_attribution.py

# v3 (extended; does not overwrite v2)
./run_exp_b_v3.sh
# optional: ./run_exp_b_v3.sh --backfill-profiles
```

Configs: `configs/exp_b.yaml`, `configs/exp_b_v2.yaml`, `configs/exp_b_v3.yaml`.

---

## 7. Claims checklist (PSB abstract alignment)

| Paper claim | B evidence | Strength |
|-------------|------------|----------|
| Attributions reflect structural signal | Same-fold > diff-fold; label-perm p = 0.0002 | **Strong** |
| Not explained by global DSSP similarity alone | Saturated diff-fold control | **Moderate** |
| Permutation-significant profile correlation | Global row-perm | **Not supported** |
| Mechanistic residue-level mapping | Single-dim dominance; weak absolute corr | **Weak / future (E, PyMOL)** |

**Recommended framing:** Pair Exp B **specificity** with Exp C **ROC**, Exp D **case study**, and Exp A **retrieval** for the full explainability story.
