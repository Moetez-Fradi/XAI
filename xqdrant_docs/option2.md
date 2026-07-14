# Option 2 — Hybrid HNSW layer cutoff (`mask_from_layer`)

**Status:** implemented on branch `option2-hybrid-layer-cutoff` (based on
`DB/masked-distance-HNSW`). Target merge base: `dev`.

**Goal:** Keep full-vector distance on coarse HNSW layers (navigability) and use
masked (focus-dims-only) distance on bottom layers (where the candidate list is
largest). Headline Figure 4 candidate.

## Semantics

Layer `L` uses **masked** distance when `L < mask_from_layer`, else **full**
distance. Layer 0 = bottom/base (densest); higher = coarser.

| `mask_from_layer` | Behaviour |
|-------------------|-----------|
| `0` | No masking → identical to plain nearest |
| omitted + `masked=true` | All layers masked → identical to Option 1 / legacy `focus.masked` |
| `>= top_layer + 1` | All layers masked (same as legacy) |
| intermediate (e.g. `1`) | Full distance on coarse layers; masked on layers below the cutoff |

`mask_from_layer` is only meaningful with `focus.masked=true`. Setting it without
`masked` returns a clear 400.

## Design

Current masked search used **one** scorer for the whole traversal. Option 2 needs
distance to depend on the current layer.

**Chosen approach:** build **both** full and masked `RawScorer`s inside
`FilteredScorer` for hybrid queries; call `set_layer(L)` at each HNSW layer
boundary so the active scorer is selected with a single integer compare (no
per-compare mode branch beyond that). Visited-list / heap logic untouched.

- `mask_from_layer = 0` is routed to plain `QueryEnum::Nearest` in
  `into_scoring_query` (identical results, no dual-scorer overhead).
- Omitted cutoff keeps a single masked scorer (zero overhead vs Milestone 3).

## Wire-up path (same as `focus.masked`)

```
REST `focus.mask_from_layer`
  → gRPC proto `DimsFocus.mask_from_layer`
  → collection `DimsFocus`
  → QueryEnum::NearestMasked(..., Option<usize>)
  → QueryVector::NearestMasked(..., Option<usize>)
  → FilteredScorer::new (dual scorers when Some(n) with n > 0)
  → GraphLayers search_entry / search_on_level → set_layer(L)
```

## Constraints (unchanged from Option 1)

Dense only; no quantization / turbo storage; not with `with_dims_explained`;
local shards only. `mask_from_layer` does not relax these.

## Key files

| Area | Path |
|------|------|
| REST schema | `lib/api/src/rest/schema.rs` |
| gRPC proto | `lib/api/src/grpc/proto/points.proto` |
| Collection routing | `lib/collection/.../collection_query.rs` |
| Query carriers | `lib/shard/.../query_enum.rs`, `lib/segment/.../vectors.rs` |
| Dual scorer | `lib/segment/.../point_scorer.rs` (`FilteredScorer`) |
| Layer switch | `lib/segment/.../graph_layers.rs` (`set_layer`) |
| Masked scorer | `lib/segment/.../masked_metric_query_scorer.rs` |

## Tests

- `mask_from_layer=0` ≡ plain nearest (collection routing)
- Large cutoff ≡ legacy full-mask (HNSW graph test)
- Intermediate cutoff returns valid top-k
- Invalid combos (`mask_from_layer` without `masked`, non-dense) error cleanly
- Scorer layer switch: layer 0 masked, layer ≥ cutoff full

## Measured results (authoritative)

| Run | Folder under `DB_systems/experiments/` |
|-----|----------------------------------------|
| High-D + gather (prefer) | `option2_highD_gather_d768_1536_n50k_q500_v1` |
| SIFT + gather (prefer) | `option2_sift1m_gather_q500_v1` |
| High-D + repack (pre-X1 e2e) | `option2_highD_d768_1536_n50k_q500_v1` |
| Early SIFT timestamped | `2026-07-10_14-02-44__option2_hybrid_layer_cutoff__recall_latency_layer_heatmap` |

Headline (gather e2e, subspace/hybrid harness GT as in script):

- **SIFT:** `mask_from_layer ≥ 1` recovers ~0.79–0.98; layer=0 (all-masked path via
  omitted / legacy) collapses much lower. **Speedup vs full still &lt; 1** in the
  canonical gather SIFT run (~0.5× class).
- **High-D:** hybrid lifts vs collapsed layer=0 (e.g. D=768 ratio 0.25 ~0.01→~0.20)
  but plateaus ~0.15–0.22; still **no** cells with speedup &gt; 1 after gather.
- Motivated **X1** (kernel) then showed microbench win without e2e speedup — see
  `X1.md`. Low full↔subspace overlap: **X2.md**.

## Keep / skip

| Role | Decision |
|------|----------|
| Navigability / paper headline | **Keep** — hard layer cutoff is the main positive result. |
| Latency win vs full search | **Not achieved** with current masked kernels + HNSW. |

## Related docs

- `docs/working.md` — Milestone 4
- `DB_systems/experiments/README.md` — folder index
- `xqdrant_docs/X1.md`, `X2.md`, `X3.md`, `option4.md`
