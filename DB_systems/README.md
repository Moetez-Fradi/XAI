# DB_systems — shared experiment harness for XQdrant

This folder holds the shared XQdrant experiment harness. The **VLDB 2027 paper's
self-contained artifact** (runners, figures, run notes, paper → code map) is in
[`VLDB 2027/`](./VLDB%202027/); start there if you are reproducing that paper.

Folder names use older `option*` / `xcut*` labels; mechanism names are **M1 / M2 / M3 /
K1 / C1 / V1**. Do not rename directories (experiment IDs and scripts depend on them).

| Mechanism | Folder under `research/` |
|-----------|--------------------------|
| Attribution, focus rescoring | root suite (`bench_suite.py`); [`step0_baseline_done/`](./research/step0_baseline_done/) |
| **M1** all-layer mask | [`option1_naive_masked/`](./research/option1_naive_masked/) |
| **M2** hybrid cutoff | [`option2_hybrid_layer_cutoff/`](./research/option2_hybrid_layer_cutoff/) |
| **M3** α-blend | [`option4_weighted_blend/`](./research/option4_weighted_blend/) |
| **K1** gather vs repack | [`xcut1_gather_vs_repack/`](./research/xcut1_gather_vs_repack/) |
| **C1** coherence | [`xcut2_subspace_coherence/`](./research/xcut2_subspace_coherence/) |
| **V1** verify ± overfetch | [`xcut3_verify_pass/`](./research/xcut3_verify_pass/) |
| Projected index | [`option3_projected_index/`](./research/option3_projected_index/) |
| Visited-set divergence | [`xcut4_divergence/`](./research/xcut4_divergence/) |

Design writeups for each mechanism: [`../xqdrant_docs/`](../xqdrant_docs/).
Engine fork: [`../XQdrant/`](../XQdrant/).
Local CSV/plot archives: [`./experiments/`](./experiments/) (gitignored; regenerate with the scripts below).

Per-mechanism runners, flags, and experiment folder names live in
[`research/README.md`](./research/README.md).

---

## Quick reproduce (paper protocol)

Same-host HTTP eval with *T*≥5 trials:

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
└── VLDB 2027/                   ← self-contained VLDB 2027 artifact
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
