# XQdrant Benchmarking Suite

Automated, end-to-end evaluation harness for **XQdrant** — our Qdrant fork that adds per-dimension feature attribution to the vector search API — targeting SIGMOD/VLDB-style systems papers.

## Qdrant vs. XQdrant (summary)


| Aspect          | Vanilla Qdrant (`/qdrant`)                                                   | XQdrant (`/XQdrant`)                                                               |
| --------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Query API       | `POST /collections/{name}/points/query` returns top-k IDs + aggregate scores | Same endpoint, extended with optional fields                                       |
| Attribution     | Not available; client must fetch vectors and compute top-m dimensions        | `with_dims_explained: true | { "top": m }` returns per-hit dimension contributions |
| Subspace search | Full-vector HNSW only                                                        | `query.nearest.focus` rescore **or** `focus.masked` in HNSW hot path               |
| Hot path        | Standard HNSW traversal                                                      | HNSW unchanged; attribution runs as collection-layer post-processing               |
| Top-m selection | Client: full sort O(D log D)                                                 | Server: bounded selection O(D log m) via `select_nth_unstable`                     |


See `[../working.md](../working.md)` for full design notes.

## Baselines evaluated

1. **Vanilla Qdrant** — standard nearest-neighbor query (`limit`, `params.hnsw_ef`).
2. **Post-Query Extraction** — vanilla search, then `POST /points` vector retrieve, then client-side full-sort attribution.
3. **XQdrant (In-Database)** — single query with `with_dims_explained` (and `nearest.focus` for subspace tests).



## Datasets

Two corpora are supported via `--dataset`:

| `--dataset` | Vectors | D | Distance | Ground truth |
|-------------|---------|---|----------|--------------|
| `synthetic` (default) | random Gaussian, L2-normalized | 768 / 1536 (`--dimensions`) | Dot | computed exactly |
| `sift1m` | SIFT1M descriptors | 128 | **Euclid** | bundled top-100 (full runs) |

SIFT1M is the realistic option: real correlated dimensions, so subspace/masked recall is meaningful (random vectors have near-zero subspace coherence). Fetch it once:

```bash
cd DB_systems
./fetch_sift.sh            # downloads into data/sift/ (~161MB)
```

Then run the suite against it (the collection is auto-created with Euclid distance):

```bash
BENCH_MODE=http BENCH_DATASET=sift1m \
QDRANT_URL=http://127.0.0.1:6335 XQDRANT_URL=http://127.0.0.1:6333 \
./run_bench.sh --dataset sift1m --queries 500
```

Quick subsampled iteration (invalidates bundled GT → recomputed):

```bash
python bench_suite.py --mode http --dataset sift1m --num-vectors 100000 --queries 200
```

Research scripts accept the same flags, e.g.:

```bash
python research/option1_naive_masked/run_option1_recall_collapse.py \
    --mode http --xqdrant-url http://127.0.0.1:6333 --dataset sift1m --queries 200 --trials 5
```

### Paper-quality unified re-run (same host + multi-trial)

To close the hardware-skew and point-estimate gaps, run attribution and masked
suites on one Linux host with $T\geq 5$ trials:

```bash
# starts local release binaries on :6335 (Qdrant) and :6333 (XQdrant)
./run_unified_paper_bench.sh

# quick pipeline check
./run_unified_paper_bench.sh --smoke

# reuse already-running servers
SKIP_SERVERS=1 BENCH_TRIALS=5 ./run_unified_paper_bench.sh
```

Each CSV gets a `*_trials.csv` sibling plus a mean±std summary; plots draw error
bars when `*_std` columns are present. Manifests record `host` + `trials`.

Override the data location with `--sift-dir` or `SIFT_DIR=/path/to/sift`.

## Layout

```
DB_systems/
├── run_bench.sh
├── bench_suite.py
├── regenerate_plots.py   # Re-render charts from saved CSVs
├── validate_http.py
├── bench_backends.py
├── bench_tests.py        # Tests A–E
├── bench_common.py
├── bench_viz.py
├── requirements.txt
└── experiments/          # One timestamped folder per run
    └── 2026-07-08_14-30-00/
        ├── manifest.json
        ├── results/*.csv
        └── plots/*.png|pdf
```



## Quick start (simulated — no server)

```bash
cd DB_systems
chmod +x run_bench.sh
./run_bench.sh
```

This runs entirely in-memory using NumPy cost models. Useful for validating the pipeline and generating chart templates before live cluster runs.

## Live evaluation (HTTP)

Qdrant does **not** accept a `--port` CLI flag. Use a config file instead:

```bash
# Vanilla Qdrant on port 6335
cat > /tmp/qdrant-6335.yaml <<'EOF'
service:
  http_port: 6335
  grpc_port: 6336
storage:
  storage_path: /tmp/qdrant-6335-storage
EOF
cd ../qdrant && ./target/release/qdrant --config-path /tmp/qdrant-6335.yaml

# XQdrant on default port 6333
cd ../XQdrant && ./target/release/xqdrant
```

Then run benchmarks:

```bash
cd DB_systems
BENCH_MODE=http \
QDRANT_URL=http://127.0.0.1:6335 \
XQDRANT_URL=http://127.0.0.1:6333 \
./run_bench.sh
```

For vanilla-vs-XQdrant on different binaries:

```bash
BENCH_MODE=http \
QDRANT_URL=http://127.0.0.1:6335 \
XQDRANT_URL=http://127.0.0.1:6333 \
./run_bench.sh
```

**Pre-flight check** (recommended before a long HTTP run):

```bash
python3 validate_http.py \
  --qdrant-url http://127.0.0.1:6335 \
  --xqdrant-url http://127.0.0.1:6333

# Or via the orchestrator:
VALIDATE_HTTP=1 BENCH_MODE=http ./run_bench.sh --validate-http
```



## Benchmark tests


| Test | Module | CSV output | Chart |
|------|--------|------------|-------|
| **A** Latency vs. Recall@K | `run_test_latency_recall` | `latency_recall_d{D}.csv` | Plot 1 |
| **B** Throughput (QPS) | `run_test_throughput` | `throughput_d{D}.csv` | Plot 2 |
| **C** Attribution depth m | `run_test_attribution_depth` | `attribution_depth_d{D}.csv` | Plot 3 |
| **D** Focus rescore | `run_test_subspace_pruning` | `subspace_rescore_d{D}.csv` | Plot 4 |
| **E** Masked HNSW | `run_test_masked_subspace` | `masked_subspace_d{D}.csv` | Plot 5 |

Each run creates `experiments/<timestamp>/` with `manifest.json` recording mode, URLs, dimensions, and tests.




### Parameters (defaults)

- Vectors: N = 10,000 per dimension
- Dimensions: D ∈ {768, 1536}
- Queries: Q = 500
- Seed: 42 (deterministic)
- Warm-up: 1,000 unmeasured queries per test block
- `ef_search` sweep: 16 → 512 (log-spaced)
- Thread counts: 1, 2, 4, 8, 16 (capped at detected cores)
- Attribution depths m: 1, 5, 10, 20, 50
- Subspace ratios: 0.25, 0.5, 0.75



## macOS notes

| Topic | Behavior |
|-------|----------|
| **Page-cache drop** | Linux-only (`/proc/sys/vm/drop_caches`). Restart Qdrant between cold runs on macOS. |
| **CPU pinning** | No native `taskset`. The script runs unpinned by default. Optional: `brew install util-linux` (provides `gtaskset`). |
| **Matplotlib cache** | `run_bench.sh` sets `MPLCONFIGDIR=./.mplconfig` automatically. |
| **Shell** | Tested with macOS default bash 3.2 (`set -u` empty-array safe). |

No extra downloads are required for a basic run — only Python 3 and the pip packages in `requirements.txt` (installed automatically by `run_bench.sh`).

## System isolation (`run_bench.sh`)

- **Page-cache drop**: `sync; echo 3 > /proc/sys/vm/drop_caches` on Linux (prints manual `sudo` instructions if permissions fail).
- **CPU pinning**: `taskset -c …` (fallback: `numactl --physcpubind=…`).
- **Turbo Boost**: best-effort disable via `intel_pstate/no_turbo` when writable.

Override pinned cores:

```bash
BENCH_CPU_CORES="4,5,6,7" ./run_bench.sh
```



## Hook points for real network calls

In `bench_backends.py`, methods on `HttpBackend` are annotated with `# [HTTP]`:


| Operation               | Endpoint                                                   |
| ----------------------- | ---------------------------------------------------------- |
| Collection create       | `PUT /collections/{name}`                                  |
| Bulk upsert             | `PUT /collections/{name}/points?wait=true`                 |
| Vanilla search          | `POST /collections/{name}/points/query`                    |
| XQdrant attribution     | same + `with_dims_explained`                               |
| Post-query vector fetch | `POST /collections/{name}/points` with `with_vector: true` |
| Subspace / focus rescore | same + `query.nearest.focus` (no `masked`) |
| Masked HNSW subspace | same + `focus.masked: true` |


Simulated equivalents live in `SimulatedBackend` in the same file.

## Selective runs

```bash
# Only Tests A and C, single dimension, fewer queries
./run_bench.sh --tests A C --dimensions 768 --queries 100

# CSV only, skip plots
./run_bench.sh --skip-plots

# Regenerate charts from a past experiment (no re-benchmark)
python3 regenerate_plots.py --experiment experiments/2026-07-08_14-30-00
python3 regenerate_plots.py --experiment experiments/2026-07-08_14-30-00 --plots 3 5
```



## Output metrics

Each CSV records distribution statistics where applicable:

- **p50 / p95 / p99** latencies (milliseconds), captured via `time.perf_counter_ns()`
- **Recall@K** vs. brute-force ground truth (Test A)
- **QPS** = total queries / wall-clock time (Test B)
- **Speedup factor** = full-dimension p50 / subspace p50 (Tests D & E)
- **Masked recall@K** = vs exact subspace brute-force ground truth (Test E)



## Citation context

When reporting results in a paper:

- State whether runs used **simulated** or **live HTTP** backends.
- For Test A recall curves, prefer **live HNSW** (HTTP mode) — the simulated backend uses an `ef_search`-sized candidate-pool approximation, not a full graph traversal.
- Disclose hardware, `BENCH_CPU_CORES`, and cache-drop procedure in the experimental setup section.

