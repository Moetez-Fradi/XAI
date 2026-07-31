# Experiment D — Thermostability Case Study (Paper Section)

**Status:** Complete — `results/exp_d/` (n = 46 curated meso/thermo pairs).  
**Run:** `./run_exp_d_finalize.sh` (see §5)

---

## 1. Role in the paper

Experiment D applies XQdrant attribution to **thermophile/mesophile ortholog pairs** — the biology-forward case study from `steps.md`. It tests whether attributed dimensions correlate with **known thermostability correlates**:

- Charged-residue surface content (DSSP RSA)
- Loop length (DSSP coil runs)
- Packing density (Biopython Shrake-Rupley SASA → `1 − mean(RSA)`)
- Ion-pair density (oppositely charged residues within 4 Å)

Cross-pair Spearman correlations include **5000-permutation p-values** and **95% bootstrap CIs** (`significance.tsv`).

---

## 2. Methods

### Corpus

- Main paper corpus: 22,118 chains (holdout lock unchanged).
- Exp D supplement: +75 chains from OMA-mapped enzyme PDB downloads.
- Merged evaluation corpus: **22,193 chains** (`corpus_merged/`).

### Pair construction (priority order)

| Source | n | Description |
|--------|---|-------------|
| literature | 33 | GroEL E. coli (`P61112`) × *T. aquaticus* (`P0A6F5`), 3 meso × 11 thermo PDB reps |
| structure_map | 3 | OMA enzyme orthologs with structures: Adk, AtpD, PykF |
| auto_fold | 10 | Same-SCOP-fold meso/thermo fallback |
| oma | 0 | Deduped with structure_map |
| **Total** | **46** | Ceiling with current merged corpus (~49 max) |

### Per pair (`exp_d_thermo.py`)

1. **Retrieval:** meso embedding queries train index (top-20); record thermo rank
2. **Direct attribution:** filtered XQdrant query on thermo `chain_id`
3. **DSSP proxies:** charged surface, loop length
4. **Structure proxies** (`exp_d_structure_features.py`): ion-pair density, SASA packing
5. **Attribution score:** Exp B v3 profile Spearman from attributed dims
6. **Cross-pair stats** (`exp_d_stats.py`): Spearman vs thermo−meso deltas, permutation p, bootstrap CI

---

## 3. Results (final paper run, n = 46)

| Metric | Value |
|--------|-------|
| Pairs | **46** |
| Ortholog recovery @20 | **1/46** (PykF `1cg1_A↔6jrq_A` at rank 3) |
| Mean xq profile Spearman | **0.088** |
| Δ charged exposed | **−0.020** |
| Δ loop length | **+0.30** |
| Δ packing density (SASA) | **−0.023** |
| Δ ion-pair density | **−0.041** |

### Attribution vs thermostability delta (Spearman, n = 46)

| Feature delta | ρ | p (perm) | 95% CI | sig @0.05 |
|---------------|---|----------|--------|-----------|
| Charged exposed | +0.22 | 0.136 | [−0.10, 0.53] | no |
| Loop length | −0.025 | 0.871 | [−0.35, 0.29] | no |
| Packing density | +0.021 | 0.885 | [−0.32, 0.32] | no |
| **Ion-pair density** | **−0.30** | **0.045** | **[−0.57, 0.01]** | **yes** |

**Interpretation:** Attribution scores correlate inversely with ion-pair density deltas across ortholog pairs (ρ ≈ −0.30, p ≈ 0.045). Other thermostability proxies show modest trends without reaching significance at n = 46. GroEL literature pairs dominate; enzyme ortholog pairs provide structural diversity but sparse retrieval signal.

---

## 4. Artifacts

| File | Content |
|------|---------|
| `data/processed/exp_d/pairs.csv` | Final 46 pairs |
| `data/annotations/exp_d/*.json` | Per-chain ion-pair + SASA packing cache |
| `results/exp_d/metrics.json` | Full per-pair + aggregate stats |
| `results/exp_d/summary.tsv` | Headline metrics |
| `results/exp_d/significance.tsv` | Correlation p-values + CIs (paper table) |
| `results/exp_d/pairs_by_source.tsv` | Per-source breakdown |
| `results/exp_d/pymol_pairs.tsv` | PyMOL helper (co-author) |

---

## 5. Reproduce (100% code complete; plots/PyMOL separate)

```bash
cd Bioinformatics

# 0) One-time environment (if not done)
./setup_env.sh
./fetch_tools_linux.sh          # mkdssp, etc.

# 1) Start vector index
./start_xqdrant.sh --daemon

# 2) First-time only: download thermo structures + build supplement corpus
./run_exp_d_downloads.sh

# 3) Final Experiment D (merge → curate → pairs → case study + stats)
./run_exp_d_finalize.sh

# If corpus/index already synced from a prior run:
./run_exp_d_finalize.sh --skip-integrate

# Re-run case study only (pairs unchanged):
.venv/bin/python scripts/exp_d_thermo.py --restart
```

**Prerequisites:** Exp B v3 map at `results/exp_b_v3/dim_profile_map.npz`, merged corpus embedded and indexed.

---

## 6. Remaining paper deliverables (not code)

| Item | Owner | Status |
|------|-------|--------|
| Scatter/bar figures from `metrics.json` | CS | → `plot_scripts/` (planned) |
| PyMOL thermo/meso attribution coloring | Biology co-author | input: `pymol_pairs.tsv` |
| Results section prose | Both | draft from this doc |

---

## 7. steps.md checklist (Experiment D)

| Requirement | Status |
|-------------|--------|
| Thermophile/mesophile ortholog pairs | ✅ 46 pairs |
| Charged surface content | ✅ DSSP |
| Loop length | ✅ DSSP |
| Packing density (FreeSASA or similar) | ✅ Shrake-Rupley SASA |
| Ion-pair density | ✅ 4 Å heavy-atom cutoff |
| Correlation of attribution with correlates | ✅ cross-pair Spearman |
| Significance tests (steps.md §5) | ✅ permutation p + bootstrap CI |
| PyMOL figures | ⏳ co-author |
| Paper plots | ⏳ `plot_scripts/` |
