# K1 — Gather vs repack kernels (`xcut1_gather_vs_repack`)

**Paper:** K1 · §4.5 design · §5.5 results · Table 2: *keep (micro)*; does not close e2e gap  
**API / env:** `XQDRANT_MASKED_KERNEL=gather|repack` (orthogonal to M1–M3)  
**Folder name (legacy):** `xcut1_gather_vs_repack`  
**Design doc:** [`../../../xqdrant_docs/X1.md`](../../../xqdrant_docs/X1.md)

**What it proves:** gather is faster than repack-then-score in ns/op microbenchmarks
(~1.1–1.8×), but re-running **M2** with gather still yields no end-to-end cell &gt; 1× —
so the wall-clock bottleneck is traversal overhead, not masked arithmetic.

## Run (bench + plot)

```bash
cd ../../../XQdrant
XQDRANT_KERNEL_BENCH_CSV_ONLY=1 \
XQDRANT_KERNEL_BENCH_CSV=$PWD/target/masked_kernel_bench/kernel_bench.csv \
  cargo bench -p segment --bench masked_kernel_bench

cd ../DB_systems/research/xcut1_gather_vs_repack
python plot_xcut1_kernel_bench.py \
  --kernel-csv ../../../XQdrant/target/masked_kernel_bench/kernel_bench.csv
```

## Canonical results

`experiments/2026-07-11_16-02-59__xcut1_gather_vs_repack__kernel_gather_vs_repack/`

E2e pairing: M2 gather folders in [`../option2_hybrid_layer_cutoff/`](../option2_hybrid_layer_cutoff/).

See also: [`../README.md`](../README.md).
