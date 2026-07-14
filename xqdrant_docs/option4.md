# Option 4 — Weighted blend distance (`focus.alpha`)

**Status:** implemented on the Option 2 / X1 / X3 branch. Does **not** change
`mask_from_layer`, gather/repack kernels, or `verify` semantics — only adds an
optional blend of full and masked **similarity scores** on layers that already
use the focus path.

**Goal:** A cheap recall/routing tuning knob so you can mask more aggressively
(lower `mask_from_layer`) while keeping navigability. Motivating find: hybrid
recovers SIFT navigability but high-D recall still plateaus; X3 verify at
`limit=k` does not move Recall@K. Alpha biases mid-traversal ranking without
changing the HNSW graph or the verify pass.

## Semantics

Blended comparison score (same polarity as existing RawScorers — higher is better):

```
score = α · s_full + (1 − α) · s_focus
```

| `focus` fields | Behaviour |
|----------------|-----------|
| `alpha` omitted / `None` | Identical to Option 1 / hybrid today (no blend) |
| `masked=true`, `alpha=0` | Pure masked score on focus layers |
| `masked=true`, `alpha=1` | Pure full score on focus layers |
| `masked=true`, `0 < alpha < 1` | Linear blend on focus layers |
| `alpha` set without `masked=true` | Fail fast: `"alpha requires focus.masked = true."` |
| `alpha` outside `[0, 1]` | Fail fast: `"alpha must be in [0, 1]."` |

### Interaction with `mask_from_layer` (preferred pairing)

| Layer policy | Scoring |
|--------------|---------|
| Coarse layers (Option 2 already full) | Full-only — **no blend** (preserves navigability; avoids double work) |
| Focus layers (`L < mask_from_layer`) | Blend when `alpha` is set; else pure masked |
| `mask_from_layer` omitted (all-layers masked / Option 1) | Every compare blends when `alpha` is set (both full + masked work — expected) |

### Interaction with `verify` (orthogonal)

`verify` still means: after traversal, re-score returned candidates with **pure
full** distance. Alpha does not change the verify pass. Scores under
`verify=true` are full-distance scores.

### Interaction with X1 kernels

The focus half of the blend uses the same masked kernel path
(`XQDRANT_MASKED_KERNEL=gather|repack`). The full half uses the normal full
scorer.

## Important performance note

**Option 4 alone does NOT reduce full-distance work.** Every blended compare
computes both `s_full` and `s_focus`. Latency wins still come from Option 2’s
layer cutoff (fewer bottom-layer compares that are “mostly focus”). Treat
`alpha` as a recall/routing knob for Options 1/2, not a standalone speedup.

## API field

REST / gRPC field name is exactly **`alpha`** (float, optional), inside `focus`
next to `masked` / `dims` / `mask_from_layer` / `verify`:

```json
{
  "query": {
    "nearest": [0.1, 0.2, "..."],
    "focus": {
      "dims": [0, 1, 2, 3],
      "masked": true,
      "mask_from_layer": 1,
      "alpha": 0.25
    }
  },
  "limit": 10,
  "params": { "hnsw_ef": 64 }
}
```

## Design

Smallest diff on the dual-scorer Option 2 path:

- Carry `alpha` as the fifth field of `QueryEnum::NearestMasked` /
  `QueryVector::NearestMasked` (`Option<OrderedFloat<f32>>` for `Hash`).
- When `alpha` is set (and/or hybrid cutoff > 0), `FilteredScorer` builds both
  full and masked RawScorers.
- On focus layers with `alpha`: `α·s_full + (1-α)·s_focus`.
- On coarse hybrid layers: full only.
- All-masked + `alpha`: cutoff treated as `usize::MAX` so every layer blends.

```
REST `focus.alpha`
  → gRPC proto `DimsFocus.alpha`
  → collection `DimsFocus`
  → QueryEnum::NearestMasked(..., verify, alpha)
  → QueryVector::NearestMasked(..., verify, alpha)
  → FilteredScorer::new (dual scorers when hybrid and/or alpha)
  → set_layer(L) → blend or full/masked
```

## Constraints (unchanged; alpha does not relax them)

Dense only; no quantization / turbo storage; not with `with_dims_explained`;
local shards only. `alpha` requires `masked=true`.

## Key files

| Area | Path |
|------|------|
| REST schema | `lib/api/src/rest/schema.rs` |
| gRPC proto | `lib/api/src/grpc/proto/points.proto` (+ `qdrant.rs`) |
| Collection routing | `lib/collection/.../collection_query.rs` |
| Query carriers | `lib/shard/.../query_enum.rs`, `lib/segment/.../vectors.rs` |
| Blend scoring | `lib/segment/.../point_scorer.rs` (`FilteredScorer`) |
| Analysis script | `DB_systems/research/option4_weighted_blend/run_option4_alpha_sweep.py` |

## Unit tests

```bash
cd /mnt/data/xq/XQdrant
export CARGO_TARGET_DIR=/mnt/data/xq/XQdrant/target   # if sandbox cache is full

# Collection routing: alpha requires masked; range; threaded; hybrid+verify+alpha
cargo test -p collection --lib alpha_

# Segment blend: alpha=0 ≡ masked; alpha=1 ≡ full; hybrid blends only focus layers
cargo test -p segment --lib alpha_

# Broader regressions
cargo test -p segment --lib hybrid_
cargo test -p segment --lib verify_
cargo test -p segment --lib masked_
cargo test -p collection --lib mask_from_layer
cargo test -p collection --lib verify_
```

## Live experiment (`run_option4_alpha_sweep.py`)

Harness scores against **subspace GT** at fixed `ratio` (default 0.25), sweeping
`alpha × mask_from_layer`. `mask_from_layer=0` is plain nearest (no blend) —
recall ≈ X2 coherence vs subspace GT.

```bash
cd /mnt/data/xq/DB_systems && source .venv/bin/activate
cd research/option4_weighted_blend

python run_option4_alpha_sweep.py --mode http \
  --xqdrant-url http://127.0.0.1:6333 --qdrant-url http://127.0.0.1:6333 \
  --dataset synthetic --dimensions 768 1536 --num-vectors 50000 --queries 500 \
  --ratio 0.25 --alphas 0 0.25 0.5 0.75 1.0 --layers 0 1 2 \
  --ef-search 128 --k 10

python run_option4_alpha_sweep.py --mode http \
  --xqdrant-url http://127.0.0.1:6333 --qdrant-url http://127.0.0.1:6333 \
  --dataset sift1m --queries 500 \
  --ratio 0.25 --alphas 0 0.25 0.5 0.75 1.0 --layers 0 1 2 \
  --ef-search 128 --k 10
```

## Measured results (authoritative)

| Run | Folder under `DB_systems/experiments/` |
|-----|----------------------------------------|
| High-D synthetic | `2026-07-14_12-31-59__option4_weighted_blend__recall_speed_alpha` |
| SIFT1M | `2026-07-14_12-37-16__option4_weighted_blend__recall_speed_alpha` |

Settings: ratio `0.25`, `k=10`, `ef=128`, alphas `{0,…,1}`, layers `{0,1,2}`, gather.

| Finding | Detail |
|---------|--------|
| **`layer=0` (no mask)** | Recall flat across alpha (~0.016 high-D, ~0.078 SIFT) ≈ X2 coherence; alpha unused. |
| **Best hybrid cell is `alpha=0`** | Pure Option 2: SIFT layer≥1 ~**0.84**; D=768 ~**0.21**; D=1536 layer=1 ~**0.29**. |
| **Raising alpha hurts** | Monotonic collapse toward the layer=0 floor (e.g. SIFT layer=1: 0.84→0.23→0.13→0.09→0.08). |
| **Does not unlock lower cutoffs** | Blending does *not* let you mask more aggressively while keeping recall — it trades away the Option 2 win. |

## Keep / skip (decision)

| Role | Decision |
|------|----------|
| API completeness | Optional — feature works (`unsupported=0`). |
| Enhancement to Option 2 | **Skip / demote.** Hard `mask_from_layer` + `alpha=0` (omit alpha) is strictly better for subspace recall. Blend does not improve the frontier. |

Paper note: negative result / ablation — weighted blend on focus layers is not worth the API surface unless a different objective (full-space GT) is shown later.

Docs: `docs/working.md` (Milestone 7), `docs/steps.md` Step 4,
`DB_systems/experiments/README.md`.
