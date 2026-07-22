# DB_systems — experiment harness for the XQdrant paper

This folder is the **artifact companion** to the short paper in
[`../deliverable/`](../deliverable/)
(*XQdrant: In-Database Attribution and Masked-Distance Subspace Search over HNSW*).

If you opened the repo after reading the paper, start here. Folder names still
use older `option*` / `xcut*` labels; the paper uses **M1 / M2 / M3 / K1 / C1 / V1**.
Use the map below — do not rename directories (experiment IDs and scripts depend on them).

| Paper | Paper § | Folder under `research/` | Verdict (Table 2) |
|-------|---------|--------------------------|-------------------|
| Attribution | §3, §5.2 | root suite Tests A–C (`bench_suite.py`) | **Keep** |
| Focus rescoring only | §5.2 | root suite Test D | Not an accelerator |
| **M1** naive mask | §4.2, §5.3 | [`option1_naive_masked/`](./research/option1_naive_masked/) | Baseline only |
| **M2** hybrid cutoff | §4.3, §5.4 | [`option2_hybrid_layer_cutoff/`](./research/option2_hybrid_layer_cutoff/) | **Keep** (navigability; e2e speedup &lt; 1) |
| **K1** gather vs repack | §4.5, §5.5 | [`xcut1_gather_vs_repack/`](./research/xcut1_gather_vs_repack/) | **Keep** (micro); does not close e2e gap |
| **C1** coherence | §2.3, §5.6 | [`xcut2_subspace_coherence/`](./research/xcut2_subspace_coherence/) | **Keep** diagnostic |
| **V1** verify ± overfetch | §4.6, §5.7 | [`xcut3_verify_pass/`](./research/xcut3_verify_pass/) | Conditional (needs overfetch + coherent data) |
| **M3** α-blend | §4.4, §5.8 | [`option4_weighted_blend/`](./research/option4_weighted_blend/) | **Skip** (best at α=0 ≡ M2) |

Out of short-paper scope (mentioned as future / full-track only):

| Idea | Folder | Notes |
|------|--------|-------|
| Projected index | [`option3_projected_index/`](./research/option3_projected_index/) | Future work (§6–7); ceiling measurable without Rust |
| Visited-set divergence | [`xcut4_divergence/`](./research/xcut4_divergence/) | Full-paper plan only; not in short draft |
| Full-track expansion | [`full_paper_plan/`](./full_paper_plan/) | Optional 12-page expansions — ignore for the short paper |

Design writeups for each mechanism: [`../xqdrant_docs/`](../xqdrant_docs/).  
Engine fork: [`../XQdrant/`](../XQdrant/).  
Canonical figures: [`../deliverable/figures/`](../deliverable/figures/).  
Local CSV/plot archives: [`./experiments/`](./experiments/) (gitignored; regenerate with the scripts below).

---

## Paper argument → what to open

Follow the same order as §5 of the paper:

1. **Attribution is a win** — run Tests A + C (latency–recall + depth *m*), or open
   [`research/step0_baseline_done/`](./research/step0_baseline_done/) for which root-suite
   outputs map to Figures 2–3.
2. **Focus rescoring ≠ traversal speedup** — Test D (§5.2 / Figure 4).
3. **M1 collapses navigability** — `option1_naive_masked` (§5.3 / Figure 5).
4. **M2 restores recall, not latency** — `option2_hybrid_layer_cutoff` (§5.4 / Figures 6–8).
5. **K1 explains the latency ceiling** — `xcut1_gather_vs_repack` (§5.5 / Figure 9).
6. **C1 explains the high-D recall ceiling** — `xcut2_subspace_coherence` (§5.6 / Figures 10–11).
7. **V1 + overfetch is conditional** — `xcut3_verify_pass` (§5.7 / Figures 12–13).
8. **M3 is dominated by M2** — `option4_weighted_blend` (§5.8 / Figure 14).

Per-mechanism runners, flags, and canonical experiment folder names live in
[`research/README.md`](./research/README.md).

---

## Quick reproduce (paper protocol)

Same-host HTTP eval with *T*≥5 trials (paper §5.1 / §8):

```bash
cd DB_systems
./fetch_sift.sh                          # once; SIFT1M → data/sift/
./run_unified_paper_bench.sh             # starts :6335 Qdrant + :6333 XQdrant
# or: SKIP_SERVERS=1 BENCH_TRIALS=5 ./run_unified_paper_bench.sh
./run_unified_paper_bench.sh --smoke     # short pipeline check
```

Defaults: gather kernel (`XQDRANT_MASKED_KERNEL=gather`), vanilla Qdrant on
`:6335`, XQdrant on `:6333`. Build release binaries first under `../qdrant` and
`../XQdrant`.

Attribution-only / exploratory suite (Tests A–E):

```bash
BENCH_MODE=http \
QDRANT_URL=http://127.0.0.1:6335 XQDRANT_URL=http://127.0.0.1:6333 \
./run_bench.sh
```

Offline pipeline check (no servers): `./run_bench.sh` (simulated backend).

---

## Layout

```
DB_systems/
├── README.md                    ← you are here (paper → code map)
├── run_unified_paper_bench.sh   ← §5 / §8 unified host re-run
├── run_bench.sh / bench_*.py    ← attribution suite (Tests A–E)
├── fetch_sift.sh                ← SIFT1M download
├── research/                    ← one folder per paper mechanism (M1–V1, …)
├── experiments/                 ← timestamped CSV/plots (local; gitignored)
└── full_paper_plan/             ← optional long-track ideas (not short paper)
```

---

## Datasets

| `--dataset` | Vectors | D | Distance | Used in paper |
|-------------|---------|---|----------|---------------|
| `synthetic` | Gaussian, L2-normalized | 768 / 1536 | Dot | Attribution suite; high-D masked sweeps |
| `sift1m` | SIFT1M | 128 | Euclid | Primary masked-distance figures |

```bash
./fetch_sift.sh   # ~161MB → data/sift/
```

---

## Root suite (attribution + motivation)

| Test | Paper role | CSV | Figure-ish |
|------|------------|-----|-----------|
| **A** Latency vs Recall@K | Attribution recall-neutral | `latency_recall_d{D}.csv` | Fig. 2 |
| **B** Throughput (QPS) | Scaling (supplementary) | `throughput_d{D}.csv` | — |
| **C** Attribution depth *m* | In-DB vs post-query | `attribution_depth_d{D}.csv` | Fig. 3 |
| **D** Focus rescore | Motivates masked traversal | `subspace_rescore_d{D}.csv` | Fig. 4 |
| **E** Masked HNSW (legacy) | Early M1 probe | `masked_subspace_d{D}.csv` | prefer `option1_*` |

HTTP endpoints used by the harness are documented in `bench_backends.py`
(`# [HTTP]` markers): `with_dims_explained`, `nearest.focus`, `focus.masked`, etc.

---

## API fields the paper measures

```json
"focus": {
  "dims": [...],
  "masked": true,
  "mask_from_layer": 1,
  "verify": true,
  "alpha": 0.25
}
```

Plus request-root `with_dims_explained`, and env `XQDRANT_MASKED_KERNEL=gather|repack`
for K1. Longer design notes: [`../docs/working.md`](../docs/working.md).
