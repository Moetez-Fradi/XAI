# Cross-cut X3 — Fallback / verify pass (`verify_recall_recovery`)

**XQdrant change needed:** `focus.verify: bool` (already implemented). When true, after masked /
hybrid traversal do a final full-distance rescore of the returned candidates.

**What it proves:** how much **full-space** recall a verify pass recovers (with optional
overfetch), and what latency it costs — i.e. whether masked search + verify is a good
"fast filter with a safety net."

## Evaluation notes

- **Primary metric:** Recall@K vs **full-space** ground truth.
- **Secondary:** Recall@K vs subspace GT (often drops under verify — expected; do not use as
  keep/skip).
- **Overfetch:** `--overfetch m` queries with `limit = m·k` + `verify=true`, then keeps top-k.
  `m=1` is plain verify; `m>1` is the real fast-filter test.
- **Hybrid:** `--layers -1 1` runs fully-masked (`-1`) and `mask_from_layer=1`.

## Run

```bash
cd /mnt/data/xq/DB_systems
source .venv/bin/activate

# High-D synthetic (full GT + overfetch + hybrid)
cd research/xcut3_verify_pass
python run_xcut3_verify.py \
  --mode http \
  --xqdrant-url http://127.0.0.1:6333 \
  --qdrant-url http://127.0.0.1:6333 \
  --dataset synthetic \
  --dimensions 768 1536 \
  --num-vectors 50000 \
  --queries 500 \
  --ratios 0.1 0.25 0.5 0.75 \
  --overfetch 1 8 \
  --layers -1 1 \
  --ef-search 128 \
  --k 10 \
  --experiment-id xcut3_highD_fullGT_overfetch_hybrid_v1

# SIFT
python run_xcut3_verify.py \
  --mode http \
  --xqdrant-url http://127.0.0.1:6333 \
  --qdrant-url http://127.0.0.1:6333 \
  --dataset sift1m \
  --queries 500 \
  --ratios 0.1 0.25 0.5 0.75 \
  --overfetch 1 8 \
  --layers -1 1 \
  --ef-search 128 \
  --k 10 \
  --experiment-id xcut3_sift_fullGT_overfetch_hybrid_v1

# offline smoke
python run_xcut3_verify.py --mode simulated --dimensions 768 --queries 50
```

## Output

`experiments/<ts>__xcut3_verify_pass__verify_recall_recovery/` (or custom `--experiment-id`)

- `results/xcut3_verify_d{D}.csv` — one row per `(layer, ratio, overfetch)`
- `plots/xcut3_verify_full_recall_d{D}_layer_{...}.png|pdf` — primary
- `plots/xcut3_verify_subspace_recall_d{D}_layer_{...}.png|pdf` — reference only
- `manifest.json`

## Keep / skip (from 2026-07-14 runs)

**Canonical folders:**
`experiments/2026-07-14_09-34-50__xcut3_highD_fullGT_overfetch_hybrid_v1`,
`experiments/2026-07-14_09-46-22__xcut3_sift_fullGT_overfetch_hybrid_v1`
(see [`../../experiments/README.md`](../../experiments/README.md)).

- **`m=1`:** does not change Recall@K (same candidate set) — keep as score/API feature only.
- **SIFT + `m=8`:** strong full-recall recovery at ratio ≥ 0.5 (~0.46→0.91 @0.75) — keep as
  paper subplot / optional safety mode **with overfetch**.
- **High-D:** still ≤ ~0.19 full recall even with `m=8` — demote as a fix (matches X2).
- **Latency:** `m=8` adds ~+45 ms — not a cheap net vs masked baseline.

Ignore 2026-07-13 `xcut3_verify_pass` folders (subspace-GT-only or simulated).
