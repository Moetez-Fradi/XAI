# Cross-cut X1 — SIMD gather vs. repack kernels (`kernel_gather_vs_repack`)

**XQdrant change needed:** the measurement is a **Rust `criterion` benchmark** comparing two ways
to score a non-contiguous focus-dim subset:
- **gather** kernels (`vgatherdps` on x86; manual gather loop on NEON), vs.
- **repack** — copy focus dims into a small contiguous scratch buffer once per query, then reuse
  the existing contiguous SIMD kernel (the `MaskedMetricQueryScorer` already keeps a `RefCell`
  scratch buffer, so repack is largely in place).

Sweep `D_sub ∈ {32, 128, 384}`. Repack may win at small `D_sub` (gather-instruction overhead).

The Rust bench must emit a CSV the plot script reads:

```
d_sub,kernel,ns_per_op
32,gather,41.2
32,repack,33.8
128,gather,88.0
...
```

## Run (plot side)

```bash
# after the Rust bench writes kernel_bench.csv
python plot_xcut1_kernel_bench.py --kernel-csv /path/to/kernel_bench.csv

# validate the plot pipeline before the Rust bench exists (template data)
python plot_xcut1_kernel_bench.py --make-example
```

## Output

`experiments/<ts>__xcut1_gather_vs_repack__kernel_gather_vs_repack/`
- `results/kernel_bench.csv` (copied/normalized)
- `plots/xcut1_kernel_gather_vs_repack.png|pdf`
- `manifest.json`
