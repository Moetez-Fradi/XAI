# Experiment A — retrieval + attribution stability

**Question:** Does XQdrant retrieval preserve fold-level accuracy vs baselines, and does `dims_explained` change rankings?

## Run

```bash
./start_xqdrant.sh --daemon
.venv/bin/python scripts/index_xqdrant.py          # paper index (resume default)
.venv/bin/python scripts/exp_a_retrieval.py        # 1500 holdout queries
./run_exp_a_baselines.sh                           # BLAST / Foldseek / TM / ESM2 cosine
.venv/bin/python scripts/recompute_exp_a_metrics.py  # add mAP if needed
```

Smoke: `./run_exp_a_smoke.sh`

## Key outputs

| Path | Description |
|------|-------------|
| `results/exp_a/metrics.json` | fold R@1, mAP, rank-stable attribution flag |
| `results/exp_a/checkpoints/queries.jsonl` | per-query neighbors + `dims_explained` |
| `results/exp_a/baselines/comparison.tsv` | baseline comparison table |

## Headline results (paper)

- XQdrant fold R@1 **0.898**, mAP **0.896**
- ESM2 exact cosine **0.957** (ANN gap vs HNSW, not attribution)
- **0** rank mismatches plain vs `dims_explained` Qdrant search
- BLASTp **0.965** (train DB; no redundancy filtering — see manuscript)

## Scripts

`index_xqdrant.py`, `exp_a_retrieval.py`, `exp_a_metrics_lib.py`, `exp_a_baseline_*.py`, `exp_a_aggregate_baselines.py`
