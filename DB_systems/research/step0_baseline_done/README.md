# Step 0 — Baseline suite (ALREADY DONE) — reproducibility pointer

These metrics are already produced by the **reusable** suite at the `DB_systems/` root — nothing
here needs to be re-created. This folder only records *which reusable code produced which metric*
so provenance is clear when summarizing.

Re-run all baselines:

```bash
cd DB_systems
BENCH_MODE=http QDRANT_URL=http://127.0.0.1:6335 XQDRANT_URL=http://127.0.0.1:6333 ./run_bench.sh
```

Existing results live under `DB_systems/experiments/2026-07-07_http_baseline/`.

| Test | Metric | Runner (`bench_tests.py`) | CSV | Plot (`bench_viz.py`) | Proves |
|------|--------|---------------------------|-----|-----------------------|--------|
| A | latency vs recall parity | `run_test_latency_recall` | `latency_recall_d{D}.csv` | `plot1_latency_recall` | Explainability is free vs vanilla |
| B | throughput (QPS) | `run_test_throughput` | `throughput_d{D}.csv` | `plot2_throughput` | Scales under concurrency |
| C | attribution depth m | `run_test_attribution_depth` | `attribution_depth_d{D}.csv` | `plot3_attribution_depth` | In-DB attribution beats client post-query |
| D | focus rescore speedup | `run_test_subspace_pruning` | `subspace_rescore_d{D}.csv` | `plot4_subspace_rescore` | Rescore path has speedup < 1 (motivates masking) |
| E | masked HNSW speedup + recall | `run_test_masked_subspace` | `masked_subspace_d{D}.csv` | `plot5_masked_subspace` | Option 1 real speedup vs recall cost |

**Note:** Test E already exercises the shipped `focus.masked` (= Option 1). The `research/option1_*`
script extends it to the wider ratio sweep `{0.1 … 1.0}` against full-vector ground truth and lands
results in a self-describing `option1_naive_masked` folder.

Rust unit tests backing these (see `working.md`): calculator (8), focus rescore (5), masked scorer (3),
OpenAPI integration.
