# Experiment E — Functional Annotation Transfer

**Status:** Complete — `results/exp_e/`

## Question (steps.md §4E)

For SwissProt/GO-labeled enzymes with UniProt active/binding sites mapped onto PDB chains:

1. Do XQdrant **dimension attributions** enrich at annotated functional sites vs the rest of the query protein?
2. Does **TM-align** structural overlap localize to those sites, or only report global fold similarity?

## Headline results (paper run)

| Metric | Value |
|--------|-------|
| Benchmark queries | **200** holdout chains (UniProt sites + EC/GO) |
| Pairs analyzed | **1,590** (1,550 functional_match, 40 fold_only) |
| XQ site enrichment (functional) | **1.104** (ratio >1 = localized) |
| XQ site enrichment (fold_only control) | **1.178** |
| TM-align site enrichment (functional) | **1.006** |
| TM-align site enrichment (fold_only) | **1.021** |
| Mann–Whitney site vs background (per-residue) | **p ≈ 0**, AUC ≈ 0.55 |
| XQ functional vs fold_only (pair-level) | **p ≈ 0.051** |
| TM functional vs fold_only (pair-level) | **p ≈ 2.7×10⁻⁵** |
| XQ site AUC (functional pairs) | **0.557** |
| TM-align pairs with mapping | **1,590 / 1,590** |

**Interpretation:** XQ attributions show significant per-residue enrichment at annotated active/binding sites (MW p≈0). Mean pair enrichment exceeds 1.0 for both XQ and TM-align. TM-align site overlap differs significantly between functional-match and fold-only cohorts; XQ cohort difference is borderline (p≈0.05). TM-align localizes weakly (~1.0) on average because global structural alignment covers much of the chain; XQ attributions are similarly modest but consistently above background at annotated sites.

## Pipeline

```bash
./run_exp_e.sh              # full paper run
./run_exp_e.sh --restart    # wipe results and rerun
./run_exp_e.sh --skip-tmalign
```

Steps:

1. `prep_exp_e_benchmark.py` — holdout curation + DSSP backfill (top 250 site-rich candidates)
2. `exp_e_functional.py` — Exp A pairs, XQ site enrichment, TM-align overlap, stats

## Artifacts

```
data/processed/exp_e/benchmark_queries.tsv   # 200 queries
data/processed/exp_e/sites/<chain_id>.json
results/exp_e/metrics.json
results/exp_e/summary.tsv
results/exp_e/significance.tsv
results/exp_e/localization_comparison.tsv
```

## Figures

```bash
python plot_scripts/plot_exp_e_functional.py   # exp_e_site_enrichment.pdf/png
python plot_scripts/generate_pymol_exp_e.py    # exp_e_session.pml
```

Config: `configs/exp_e.yaml`

## Fixes applied (2026-07-30)

- Exp A query field (`query` not `query_chain_id`)
- TM-align 2024+ stdout parser (replaces deprecated `-a` file)
- DSSP backfill for benchmark holdout chains (5 → 200 queries)
- Stricter functional_match (EC / same UniProt / non-generic GO)
- fold_only control cohort (same SCOP fold, different function)
