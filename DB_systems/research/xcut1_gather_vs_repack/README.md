# Cross-cut X1 — SIMD gather vs repack (`kernel_gather_vs_repack`)

**XQdrant:** gather/repack kernels **implemented** (`XQDRANT_MASKED_KERNEL=gather|repack`).
Criterion bench emits `d_sub,kernel,ns_per_op`. See `xqdrant_docs/X1.md`.

## Run (bench + plot)

```bash
cd /mnt/data/xq/XQdrant
XQDRANT_KERNEL_BENCH_CSV_ONLY=1 \
XQDRANT_KERNEL_BENCH_CSV=$PWD/target/masked_kernel_bench/kernel_bench.csv \
  cargo bench -p segment --bench masked_kernel_bench

cd /mnt/data/xq/DB_systems/research/xcut1_gather_vs_repack
python plot_xcut1_kernel_bench.py \
  --kernel-csv /mnt/data/xq/XQdrant/target/masked_kernel_bench/kernel_bench.csv
```

## Canonical results

`experiments/2026-07-11_16-02-59__xcut1_gather_vs_repack__kernel_gather_vs_repack/`

**Measured:** gather faster almost everywhere (~1.1–1.8×). E2e Option 2 + gather still
has **no** wall-clock speedup &gt; 1 vs full search (see Option 2 gather folders).
