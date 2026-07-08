# Option 1 — Naive full-masked traversal (`recall_vs_ratio`)

**XQdrant change needed:** **None.** The shipped `focus.masked = true` already computes the
reduced distance across *all* HNSW layers, which is exactly Option 1. This step is pure analysis.

**What it proves:** where recall collapses as the focus subspace shrinks — the motivating baseline
for the hybrid (Option 2). Report the crossover ratio where Recall@10 drops below ~0.8.

## Run

```bash
# live
python run_option1_recall_collapse.py --mode http --xqdrant-url http://127.0.0.1:6333 --dimensions 768 1536

# offline pipeline check (fabricated, tagged simulated=1)
python run_option1_recall_collapse.py --mode simulated --dimensions 768 --queries 50
```

## Output

`experiments/<ts>__option1_naive_masked__recall_vs_ratio/`
- `results/option1_recall_vs_ratio_d{D}.csv` — recall vs full-vector GT + p50/p95 per ratio
- `plots/option1_recall_vs_ratio_d{D}.png|pdf`
- `manifest.json` — step, metric, mode, URLs, sweep

**Recall here is measured against full-vector ground truth**, not vanilla Qdrant — the graph itself
can misroute under masking, so absolute recall is what matters.
