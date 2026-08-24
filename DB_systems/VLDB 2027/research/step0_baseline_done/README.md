# Attribution suite pointer (`step0_baseline_done`)

**Paper:** §3 (design) · §5.2 (attribution + focus-rescore motivation) · Table 2 top rows  
**Code:** reusable root suite — nothing unique lives in this folder.

This directory only records *which* root-suite tests produced the attribution /
motivation figures so provenance stays clear.

Re-run:

```bash
cd ../..   # DB_systems/
BENCH_MODE=http QDRANT_URL=http://127.0.0.1:6335 XQDRANT_URL=http://127.0.0.1:6333 \
  ./run_bench.sh --tests A C D --trials 5

# or the unified paper driver (includes A/C/D + masked suite)
./run_unified_paper_bench.sh
```

| Test | Paper role | Runner (`bench_tests.py`) | CSV | Approx. figure |
|------|------------|---------------------------|-----|----------------|
| A | Attribution recall-neutral vs vanilla | `run_test_latency_recall` | `latency_recall_d{D}.csv` | Fig. 2 |
| B | Throughput (supplementary) | `run_test_throughput` | `throughput_d{D}.csv` | — |
| C | In-DB vs post-query attribution depth | `run_test_attribution_depth` | `attribution_depth_d{D}.csv` | Fig. 3 |
| D | Focus rescoring speedup &lt; 1 | `run_test_subspace_pruning` | `subspace_rescore_d{D}.csv` | Fig. 4 |
| E | Early masked probe | `run_test_masked_subspace` | `masked_subspace_d{D}.csv` | prefer **M1** script |

Test E already exercises `focus.masked` (**M1**). The dedicated
[`../option1_naive_masked/`](../option1_naive_masked/) runner extends the ratio sweep
against **full-space** ground truth for §5.3.

See: [`../README.md`](../README.md) · [`../../README.md`](../../README.md).
