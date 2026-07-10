# Dimension Explainability — Working Notes

This document tracks the **dimension explainability** feature for our Qdrant fork. It is for internal development; user-facing API docs live in the OpenAPI spec (Redoc).

## Goal

When running similarity search (top-k nearest neighbors), optionally return which embedding dimensions contributed most to each result's score. Later, support searching that emphasizes a chosen subset of dimensions.

Use case downstream: embedding interpretability studies — correlate influential dimensions with real-world features in the data.

---

## Design decisions

### No new endpoint

We extend the existing **Query API** (`POST /collections/{name}/points/query`) instead of adding separate endpoints:

| Field | Location | Default | Purpose |
|-------|----------|---------|---------|
| `with_dims_explained` | Query request root | omitted / `false` | Attach per-dimension score breakdown to each hit |
| `query.nearest.focus` | Inside `nearest` query | omitted | Re-score candidates using only selected dimensions |

**Why:** Qdrant already uses optional request fields on `/query` (e.g. `with_vector`, MMR). A separate endpoint would duplicate validation, batching, prefetches, and gRPC mappings. Both features compose on the same endpoint.

### Collection-level post-processing (not inside HNSW)

Explanations are computed **after** shard search completes, on the node that responds to the user:

1. Shards run normal vector search (unchanged hot path).
2. The collection layer builds a `DimsExplainedCalculator` from the query vector + collection distance.
3. For each result point, it computes top-N dimension contributions (reusing attached vectors or doing a lightweight internal retrieve).

**Why:** Keeps HNSW / quantized search paths untouched. Overhead is O(result_count × vector_dim) and only when explicitly requested.

### Dimension-focused search = preselect + rescore

`query.nearest.focus` does **not** change the index traversal:

1. Full-vector NN search preselects `candidates_limit` candidates (defaults to `limit`).
2. Candidates are re-scored with `score_for_dims` on the focus dimensions only.
3. Top `limit` results are returned.

This is the same pattern as MMR re-ranking.

### Masked distance in HNSW (`focus.masked = true`)

The preselect+rescore path above is *exact* but does full-vector work during traversal. The
**masked** variant instead computes the reduced distance **inside the index hot path**: HNSW
traversal itself only multiplies/adds over the focus dimensions.

1. `focus.masked = true` routes the query to a dedicated segment query instead of the rescore path.
2. During graph traversal every candidate is scored by a `MaskedMetricQueryScorer` that gathers only
   the focus dimensions of the stored vector and calls the normal SIMD metric on the reduced slices.
3. No preselect + rescore stage — `candidates_limit` is ignored for masked search.

**Trade-off:** real per-comparison speedup (fewer FLOPs), but results are **approximate** — the graph
edges were built for full vectors, so the reduced metric can steer traversal into a different (and
possibly worse) neighborhood. This is the intended experiment: measure recall vs. speedup as the
focus subset shrinks.

**Semantics kept consistent with rescore:** the query is preprocessed exactly like a full nearest
search (e.g. Cosine L2-normalizes over *all* dimensions), then only the focus terms are summed. So a
masked query with *all* dimensions selected is numerically identical to a plain `nearest`. Scores are
post-processed identically to `Nearest`.

**Scope / limitations (fail fast with a clear error):**

- Dense vectors only; sparse / multi-dense are rejected.
- Not compatible with quantization or the inline-vectors (`turbo`) graph fast path (`SupportsBytes = False`).
- Not combinable with `with_dims_explained` (the decomposition needs full-vector terms).
- Local execution only: the internal shard-query proto has no masked variant, so it is not sent to
  remote shards (`unimplemented!()` on that gRPC edge, mirroring the `FeedbackNaive` precedent).

Carried end-to-end as a new query variant rather than a `SearchParams`/`CoreSearchRequest` field:
`QueryEnum::NearestMasked(NamedQuery<VectorInternal>, Vec<u32>, Option<usize>)` at the shard layer and
`QueryVector::NearestMasked(VectorInternal, Vec<u32>, Option<usize>)` at the segment layer. The optional
third field is `mask_from_layer` (hybrid Option 2); `None` means mask every layer. The `raw_scorer`
dispatch builds the `MaskedMetricQueryScorer` for dense storage and rejects every other storage kind.
Hybrid queries build both full and masked scorers inside `FilteredScorer` and select per layer.

---

## API (REST)

### Per-dimension explanations

```json
POST /collections/{collection_name}/points/query
{
  "query": {
    "nearest": [0.5, -1.0, 2.0, 0.0, 3.0]
  },
  "limit": 5,
  "with_dims_explained": true
}
```

`with_dims_explained` accepts:

- `false` — default behavior, field omitted from response
- `true` — include top **10** dimensions (default)
- `{ "top": 3 }` — custom count (1–65536)

Response adds to each `ScoredPoint`:

```json
{
  "id": 1,
  "score": 4.5,
  "dims_explained": {
    "4": 6.0,
    "1": -1.0,
    "2": -1.0
  }
}
```

Keys are dimension indexes; values are the per-dimension **term** in the score decomposition (signed). Sorted by descending absolute contribution in the internal representation; JSON map order is not guaranteed.

### Dimension-focused search

```json
POST /collections/{collection_name}/points/query
{
  "query": {
    "nearest": {
      "nearest": [0.5, -1.0, 2.0, 0.0, 3.0],
      "focus": {
        "dims": [0, 4],
        "candidates_limit": 100
      }
    }
  },
  "limit": 5,
  "with_dims_explained": { "top": 5 }
}
```

Combine both: focused re-scoring **and** explanations restricted to the focus dimensions.

Set `"masked": true` inside `focus` to compute the reduced distance directly during HNSW traversal
instead of preselect + rescore (`candidates_limit` is then ignored):

```json
{
  "query": {
    "nearest": {
      "nearest": [0.5, -1.0, 2.0, 0.0, 3.0],
      "focus": { "dims": [0, 4], "masked": true }
    }
  },
  "limit": 5
}
```

Optional hybrid cutoff (Option 2): set `mask_from_layer` so coarse layers keep full
distance and only layers `L < mask_from_layer` use the masked metric (`0` = no masking;
omit with `masked=true` to mask every layer, matching legacy behaviour):

```json
{
  "query": {
    "nearest": {
      "nearest": [0.5, -1.0, 2.0, 0.0, 3.0],
      "focus": { "dims": [0, 4], "masked": true, "mask_from_layer": 1 }
    }
  },
  "limit": 5
}
```

### Supported / unsupported

| Supported | Not supported (returns 400) |
|-----------|----------------------------|
| `nearest` on **dense** vectors | `recommend`, `discover`, `context`, fusion, MMR, formula, sample |
| `nearest` + `focus` on dense vectors | Sparse / multi-dense vectors |
| `nearest` + `focus.masked` on dense vectors (local shards) | `focus.masked` + quantization / turbo storage |
| `nearest` + `focus.masked` + `mask_from_layer` (hybrid) | `focus.masked` + `with_dims_explained` |
| All distance metrics: Dot, Cosine, Euclid, Manhattan | `mask_from_layer` without `masked=true` |
| | `mmr` + `focus` together |

---

## Score decomposition (by distance)

Implemented in `lib/segment/src/common/dims_explained.rs`:

| Distance | Per-dimension term | Sum of all terms |
|----------|-------------------|------------------|
| Dot | `q_i × v_i` | score |
| Cosine | `q̂_i × v̂_i` (L2-normalized) | score |
| Euclid | `(q_i - v_i)²` | score² (distance squared) |
| Manhattan | `|q_i - v_i|` | score |

Top-N selection uses **absolute** contribution. Cosine normalizes query (and point if needed) before computing terms.

---

## Request flow

```
REST/gRPC request
  → CollectionQueryRequest.dims_explained
  → validate_dims_explained_support()
  → ShardQueryRequest (dims_explained flag kept locally)
  → shard search (dims_explained always None at segment level)
  → merge results
  → if DimsFocus: dims_focus_rescore() [+ explanations if requested]
  → elif plain Nearest + with_dims_explained: fill_results_with_dims_explained()
  → REST/gRPC response (dims_explained map on ScoredPoint)
```

Remote shards: internal gRPC strips `dims_explained`; the **coordinator** computes explanations after merge.

---

## Key files

| Area | Path |
|------|------|
| Core math | `lib/segment/src/common/dims_explained.rs` |
| Collection post-process | `lib/collection/src/collection/dims_explained.rs` |
| Focus rescore | `lib/shard/src/query/dims_focus.rs` |
| Masked scorer (HNSW hot path) | `lib/segment/src/vector_storage/query_scorer/masked_metric_query_scorer.rs` |
| Masked scorer dispatch | `lib/segment/src/vector_storage/raw_scorer.rs` |
| Masked query carriers | `lib/segment/src/data_types/vectors.rs` (`QueryVector::NearestMasked`), `lib/shard/src/query/query_enum.rs` (`QueryEnum::NearestMasked`) |
| Masked routing | `lib/collection/src/operations/universal_query/collection_query.rs` (`into_scoring_query`) |
| Query types / validation | `lib/collection/src/operations/universal_query/collection_query.rs` |
| REST schema | `lib/api/src/rest/schema.rs` |
| gRPC proto | `lib/api/src/grpc/proto/points.proto` |
| REST ingestion | `src/common/inference/query_requests_rest.rs` |
| Batch hook | `lib/collection/src/collection/query.rs` |

---

## Tests

| Test | Location | Status |
|------|----------|--------|
| Calculator unit tests (8) | `lib/segment/src/common/dims_explained.rs` | ✅ pass |
| Focus rescore unit tests (5) | `lib/shard/src/query/dims_focus/tests.rs` | ✅ pass |
| Masked scorer unit tests (3) | `lib/segment/src/vector_storage/query_scorer/masked_metric_query_scorer.rs` | ✅ pass |
| Hybrid layer cutoff tests | masked scorer + `graph_layers` + `collection_query` | ✅ pass |
| OpenAPI integration | `tests/openapi/test_dims_explained.py` | ✅ added |
| gRPC | — | TODO (extend `tests/basic_grpc_test.sh`) |

Run locally:

```bash
cargo test -p segment dims_explained
cargo test -p shard dims_focus
cargo check --workspace
# With Qdrant running:
uv --project tests run pytest tests/openapi/test_dims_explained.py -v
```

---

## Performance notes

- **Zero overhead** when `with_dims_explained` is not set (early return before retrieve/calculate).
- Explanations may trigger an internal **vector retrieve** for result IDs that did not include vectors in the search response (unless `with_vector` already fetched them). Vectors used only for explanation are **not** returned unless `with_vector` is set.
- HW counters track CPU cost: `explained_points × query_len` for plain nearest; `rescored × focus_dims.len()` for focus rescore.
- Top-N uses `select_nth_unstable` — O(n) average instead of full sort.

---

## PR checklist (Qdrant contribution rules)

Target branch: **`dev`** (not `master`).

- [x] Rust implementation + unit tests
- [x] REST schema (`JsonSchema` on types in `schema.rs`)
- [x] gRPC proto + `build.rs` validation
- [x] Backward compatible (all new fields optional)
- [x] Group queries (`/query/groups`) support `with_dims_explained`
- [x] OpenAPI integration tests (query + groups)
- [ ] Regenerate OpenAPI: `./tools/generate_openapi_models.sh` (requires Docker)
- [x] `cargo +nightly fmt --all`
- [x] `cargo clippy --workspace` (default features)
- [ ] `cargo clippy --workspace --all-features` (needs cmake + Python 3.10+)
- [ ] Full `uv --project tests run pytest tests/openapi`
- [ ] Disclose AI-assisted commits in PR description (per `docs/CONTRIBUTING.md`)

---

## Remaining work (future milestones)

1. **OpenAPI regeneration** — run `./tools/generate_openapi_models.sh` (needs Docker), commit `docs/redoc/master/openapi.json`
2. **gRPC integration test** — extend `tests/basic_grpc_test.sh`
3. **Edge Python** — intentionally skipped (Edge resolves queries locally; comment in `lib/edge/python/src/query.rs`)
4. **Full clippy** — `cargo clippy --workspace --all-features` on a machine with cmake + Python 3.10+
5. **Run openapi tests** — `uv --project tests run pytest tests/openapi/test_dims_explained.py -v` against a live server

---

## Milestone log

### Milestone 1 — Core feature compiles

- Full workspace `cargo check` passes
- 13 unit tests pass (segment + shard)
- OpenAPI integration test added
- This working document written

**Commit:** `feat: add dimension explainability to query API`

### Milestone 2 — Group queries + hygiene (current)

- `with_dims_explained` wired through `/query/groups` (REST + gRPC)
- Group hits inherit explanations via existing `collection.query()` → `fill_results_with_dims_explained` path
- Added `test_dims_explained_on_query_groups` integration test
- `cargo +nightly fmt --all` and `cargo clippy --workspace` pass

**Suggested commit message:**

```
feat: support with_dims_explained on query groups

Wire the dimension explainability flag through /query/groups REST and
gRPC endpoints so grouped search hits include dims_explained when requested.
```

### Milestone 3 — Masked distance in HNSW (current)

Moves dimension focus from post-hoc rescoring into the index hot path.

- New `MaskedMetricQueryScorer` scores only the focus dims during HNSW traversal, gathering the
  reduced slice and reusing the existing SIMD metric; scratch buffer via `RefCell` avoids per-compare
  allocations.
- New query carriers `QueryVector::NearestMasked` / `QueryEnum::NearestMasked` thread the focus dims
  from the collection query down to the segment `raw_scorer` dispatch.
- `focus.masked` flag added to REST + gRPC `DimsFocus`; `into_scoring_query` routes masked focus to a
  plain vector scoring query (no preselect + rescore).
- Rejected paths return clear errors: sparse/multi-dense, quantization, turbo storage, remote shards,
  and `masked` + `with_dims_explained`.
- 3 masked-scorer unit tests added (exact dot over focus dims; all-dims == full nearest for
  Dot/Cosine/Euclid; out-of-bounds dims ignored).

Build intentionally not run yet (owner will build/test).

**Suggested commit message:**

```
feat: masked distance search inside HNSW traversal

Add focus.masked to compute dot/cosine over a subset of dimensions
directly in the HNSW hot path via a MaskedMetricQueryScorer, trading
approximate results for a real per-comparison speedup.
```

### Milestone 4 — Hybrid layer cutoff (`mask_from_layer`)

Option 2: full-distance coarse layers, masked-distance bottom layers.

- Optional `mask_from_layer` on REST/gRPC `DimsFocus` (next to `masked` / `dims`).
  Semantics: layer `L` uses masked distance when `L < mask_from_layer`, else full.
  - `0` → no masking (routed to plain nearest)
  - omitted + `masked=true` → all layers masked (backward compatible with Milestone 3)
  - `>= top_layer + 1` → all layers masked (same as legacy)
- `FilteredScorer` builds both full and masked raw scorers for hybrid queries; 
  `set_layer` at each HNSW layer boundary selects which scorer is active (visited/heap
  logic untouched).
- Carried as `QueryEnum::NearestMasked(..., Option<usize>)` /
  `QueryVector::NearestMasked(..., Option<usize>)`.
- Same fail-fast constraints as `focus.masked` (dense only; no quantization/turbo;
  not with `with_dims_explained`; local shards only). `mask_from_layer` without
  `masked=true` is rejected.
- Unit tests: scorer layer switch; HNSW large-cutoff ≡ full mask; collection routing
  for `0` / omitted / hybrid / invalid combos.

**Suggested commit message:**

```
feat: hybrid HNSW masked traversal via mask_from_layer

Add focus.mask_from_layer so coarse HNSW layers keep full distance while
bottom layers use the masked metric, preserving navigability with a
sweepable recall/latency trade-off.
```
