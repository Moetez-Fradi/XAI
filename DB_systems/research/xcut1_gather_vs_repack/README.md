# K1 — Masked kernel cost (`xcut1_gather_vs_repack`)

**Paper:** K1 · §4.5 design · §5.5 results (Figure 4, Table 3) · Table 4: *skip gather*  
**API / env:** `XQDRANT_MASKED_KERNEL=repack|gather` (engine default: `repack`; orthogonal to M1–M3)  
**Folder name (legacy):** `xcut1_gather_vs_repack`  
**Design doc:** [`../../../xqdrant_docs/X1.md`](../../../xqdrant_docs/X1.md)

**What it shows:** per distance, a masked score costs far more than a full contiguous one,
whichever kernel computes it. At SIFT width (Euclid, D=128) on host B (Ryzen 5 220, AVX2):

| Focus dims (ratio) | contiguous | repack | gather |
|--------------------|-----------:|-------:|-------:|
| 32 (0.25)  | 13.6 ns | 36.4 ns | 36.9 ns |
| 96 (0.75)  | 17.1 ns | 65.1 ns | 78.4 ns |
| 128 (1.0)  | **18.3 ns** (= plain full distance) | 81.9 ns | 94.5 ns |

Mean over T=5 seeded trials (`results_2026-10-05_hostB/run_a`; `run_b` reproduces it within
~6%). A masked distance over a quarter of the dimensions already costs 2× the full
contiguous distance, which is why masked search never beats plain search in-engine
(`../inengine_checks/`). AVX2 gather does not help on this CPU: it ties repack at 32 dims,
is 15–20% slower at 96–128, and about 2× slower at D=1536 for 128+ focus dims
(`xcut1_kernel_wide.png`). In-engine the two kernels are within 3%: M1 runs at 0.84× / 0.67× / 0.65× of
plain speed at ratios 0.25 / 0.75 / 1.0 with gather and 0.82× / 0.65× / 0.63× with repack
(`../inengine_checks/results_2026-10-05_hostB/`, interleaved).

### Correction to the earlier K1 numbers

The earlier version of this bench (2026-07-11, run on an Apple M2) reported gather
1.1–1.8× faster than repack. Two problems: (1) it passed the *scalar* reference
`segment::spaces::simple::dot_similarity` to repack, whereas the engine's repack path calls
the SIMD-dispatching `Metric::similarity`; (2) on aarch64 the gather kernel is the scalar
loop, so neither side used SIMD. The current bench uses `Metric::similarity` for repack and
contiguous, exactly as the engine does, and runs on the x86/AVX2 host that produced every
in-engine timing.

## Bench

Rust bench `masked_kernel_bench` in the engine fork, branch `edbt27-k1-bench`, commit
`6cb79ea` (adds only the bench on top of `ed5ca79`; engine code unchanged). Three kernels
per focus width: `repack`, `gather`, and `contiguous` (the same metric over already
contiguous coordinates; at `d_sub == full_dim` it is the plain full-vector distance). Two
configs: `wide` (Dot, D=1536) and `sift` (Euclid, D=128). Each config runs T=5 trials with
fresh fixture seeds, rotated kernel order, and a gather/repack equivalence check.

## Run (bench + plot)

```bash
cd ../../../XQdrant            # engine fork at 6cb79ea
XQDRANT_KERNEL_BENCH_CSV_ONLY=1 \
XQDRANT_KERNEL_BENCH_CSV=$PWD/target/masked_kernel_bench/kernel_bench.csv \
  taskset -c 2 cargo bench -p segment --bench masked_kernel_bench

cd ../DB_systems/research/xcut1_gather_vs_repack
python plot_xcut1_kernel_bench.py \
  --kernel-csv ../../../XQdrant/target/masked_kernel_bench/kernel_bench.csv
```

## Canonical results

[`results_2026-10-05_hostB/`](./results_2026-10-05_hostB/): `run_a/` (paper) and `run_b/`
(repeat), each with `kernel_bench.csv`, `bench.log`, and `host.txt`; plots
`xcut1_kernel_sift.png` (paper Figure 4) and `xcut1_kernel_wide.png`.

See also: [`../README.md`](../README.md).
