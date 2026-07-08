# Masked-Distance HNSW Traversal — Design & Test Plan

**Goal:** Replace the current preselect-and-rescore `nearest.focus` (which has speedup < 1)
with a traversal-level optimization that computes distance only over focus dimensions
during graph descent — a *true* subspace speedup rather than a rescoring convenience.

**Core risk (applies to all options):** HNSW's navigability relies on graph edges encoding
full-space proximity. Changing the distance function mid-traversal can break greedy descent,
causing recall collapse — especially as `D_sub / D` shrinks. Every option below must be
evaluated on **recall**, not just latency.

---

## Option 1 — Naive Full-Masked Traversal

**What it is:** Use the masked (focus-dims-only) distance for *all* HNSW layers, top to bottom.

**Why try it:** Simplest to implement; gives you the baseline "worst case" data point and
demonstrates *why* naive masking breaks navigability — useful even as a negative result.

**Code changes:**
- Add a `DistanceMode::Masked(Vec<usize>)` variant alongside existing metric enum.
- Thread it through the graph traversal function so every `s(q, v)` call during descent
  uses only the focus dims.
- No new data structures needed — reuse existing SIMD kernels via a repack step (see
  "Cross-cutting code work" below).

**How to test / keep-or-skip criteria:**
- Sweep `D_sub / D ∈ {0.1, 0.25, 0.5, 0.75, 1.0}` at fixed `ef_search`.
- Measure Recall@10 vs. full-vector ground truth (not vs. vanilla Qdrant — you need
  absolute recall here, since the graph itself may misroute).
- **Skip condition:** if Recall@10 drops below ~0.8 at `D_sub/D ≥ 0.5`, this option is not
  viable standalone — but keep the recall-vs-ratio curve as motivating evidence for
  Option 2.
- **Keep condition:** if recall stays acceptable at some ratio range, report the crossover
  point where it degrades — this alone is a publishable characterization.

---

## Option 2 — Hybrid: Full-Distance Coarse Layers, Masked Bottom Layer(s)

**What it is:** Keep full-vector distance for upper HNSW layers (coarse region-finding),
switch to masked distance only for the bottom `N` layers (local refinement).

**Why try it first among the "real" fixes:** Preserves most of the navigability guarantee
(coarse routing still full-space) while cutting distance-computation cost where the
candidate list is largest. Best risk/reward of the four.

**Code changes:**
- Add a `mask_from_layer: usize` config parameter (0 = fully masked, `max_layer` = fully
  unmasked / current behavior).
- In the layer-traversal loop, branch distance-function selection on `current_layer <
  mask_from_layer`.
- Expose this as a sweep-able parameter in the benchmarking harness (not just a fixed
  constant) — you'll want it as an experimental axis.

**How to test / keep-or-skip criteria:**
- 2D sweep: `mask_from_layer ∈ {0, 1, 2, ..., max_layer}` × `D_sub/D ∈ {0.25, 0.5, 0.75}`.
- Plot Recall@10 and p50 latency as a heatmap or small-multiples grid over this sweep.
- **Keep condition:** any `mask_from_layer` value where latency improves *and* recall stays
  within ~0.01–0.02 of full-vector baseline. This becomes your new headline Figure 4.
- **Skip condition:** if no layer cutoff recovers acceptable recall at any useful `D_sub/D`,
  fall back to reporting this as a negative result alongside Option 1's data — still
  useful, just reframe the contribution as a characterization rather than a speedup.

---

## Option 3 — Auxiliary Projected Index for Common Focus Sets

**What it is:** Pre-build a separate, small HNSW graph on vectors projected onto a
*predefined* focus subspace (e.g., dims commonly queried together). Search that graph
directly instead of masking traversal on the main index.

**Why try it:** Sidesteps the navigability problem entirely — the auxiliary graph is built
*and* searched in the same space, so its own small-world guarantees hold. Cost is extra
memory and only covers anticipated focus sets, not arbitrary ones.

**Code changes:**
- New index-build path: given a list of registered focus sets (e.g., via collection
  config), project all vectors onto each set and build a secondary HNSW graph per set.
- Query routing: if `focus.dims` matches (or is a subset of) a registered projected index,
  route there instead of the main index + masking.
- Needs an index-selection/registration API surface — this is the most invasive option
  code-wise.

**How to test / keep-or-skip criteria:**
- Compare against Option 2's best configuration at matching `D_sub/D`.
- Measure: (a) recall — should be ~identical to a from-scratch full HNSW build in that
  subspace (it's a "real" index, so this is your recall ceiling); (b) extra memory per
  registered focus set; (c) index build time overhead.
- **Keep condition:** if your target use case has a small, known palette of focus sets
  (e.g., "explain by embedding-block" or fixed attribute groups) — likely worth keeping as
  a complementary feature, not a replacement for Option 2.
- **Skip condition:** if focus sets are arbitrary/ad-hoc per query, the memory cost doesn't
  scale — mention as future work instead of implementing fully.

---

## Option 4 — Weighted Blend Distance

**What it is:** `d(q,v) = α · d_full(q,v) + (1-α) · d_focus(q,v)`, tunable per query, used
throughout traversal.

**Why try it:** Natural ablation axis producing a smooth recall-vs-speedup curve; cheap to
implement since it's a linear combination of distances you already compute.

**Code changes:**
- Add `alpha: f32` to the focus query param.
- Compute both full and masked distance per comparison (note: this does *not* save
  full-distance computation cost — it only biases routing, so it's not a source of
  additional speedup on its own; likely to be paired with Option 2's layer cutoff for
  actual latency gains).
- Cheapest of the four to add, but least likely to independently produce a large speedup —
  treat it as a tuning knob for Option 1 or 2 rather than a standalone option.

**How to test / keep-or-skip criteria:**
- Sweep `α ∈ {0, 0.25, 0.5, 0.75, 1.0}` combined with Option 2's `mask_from_layer`.
- **Keep condition:** if blending recovers meaningful recall at low `mask_from_layer`
  values (i.e., lets you mask more aggressively while staying accurate) — report as an
  enhancement to Option 2.
- **Skip condition:** if `α` doesn't materially change the recall/speed frontier compared
  to a hard layer cutoff — drop it, not worth the added API surface complexity.

---

## Cross-Cutting Code Work (needed regardless of which option(s) survive)

1. **SIMD gather kernels for non-contiguous dims.**
   Current AVX2 (`simple_avx.rs`) / NEON (`simple_neon.rs`) kernels assume contiguous
   `f32` slices. Focus dims are a sparse subset, so you need either:
   - Gather-based kernels (`vgatherdps` on x86; manual gather loop on NEON, since it lacks
     a native gather instruction), or
   - A repack step: copy focus dims into a small contiguous scratch buffer once per query,
     reuse across the whole traversal.
   **Test:** benchmark both approaches at `D_sub ∈ {32, 128, 384}` — repacking may win at
   small `D_sub` due to gather-instruction overhead. This comparison is itself worth a
   paragraph/table in the paper.

2. **Subspace coherence diagnostic (offline metric).**
   Compute correlation between full-space and subspace k-NN sets on a corpus sample,
   independent of any traversal changes. Gives you a principled *why* for recall
   degradation curves rather than just reporting numbers.
   **Test:** run once per dataset/focus-set combination; correlate with the recall curves
   from Options 1–4 to see if it predicts which `D_sub/D` ratios are safe.

3. **Fallback / verification pass.**
   Optional final full-distance rescore of top-`k` candidates before returning (cheap,
   since `k` is small). Turns masked traversal into a *fast filter* rather than the final
   arbiter of correctness.
   **Test:** measure the latency cost of the rescore pass vs. the recall it recovers —
   report as an optional safety mode (`verify: true`) in the API.

4. **Candidate-list divergence instrumentation.**
   Log how often the masked-traversal visited-node set diverges from the full-traversal
   visited-node set. Useful for debugging and makes a good supplementary figure showing
   *where* divergence starts as `D_sub` shrinks.
   **Test:** instrument during the Option 1/2 sweeps above — no separate experiment needed,
   just extra logging on runs you're already doing.

---

## Suggested Execution Order

1. Implement cross-cutting repack/gather kernel (needed by all options).
2. Run **Option 1** sweep — cheapest to build, gives you the recall-collapse baseline and
   motivates everything else.
3. Run **Option 2** sweep — this is your most likely paper-worthy result.
4. If time allows: **Option 4** as an add-on to Option 2's best config.
5. **Option 3** only if you have a concrete use case with a small, known focus-set palette
   — otherwise mention as future work rather than implementing fully.

## Decision Table for the Paper

| Option | Implement? | If it works | If it doesn't |
|---|---|---|---|
| 1. Naive masked | Yes (cheap, motivating) | Report as upper-bound-speedup / recall-cost baseline | Keep as "why naive masking fails" evidence |
| 2. Hybrid layer cutoff | Yes (priority) | New headline Figure 4 replacement | Report as negative result + explanation |
| 3. Auxiliary projected index | Only if you have a real focus-set use case | Secondary contribution / complementary feature | Cut to future work paragraph |
| 4. Weighted blend | Optional, cheap add-on to Option 2 | Minor tuning-knob result | Drop entirely, not worth API surface cost |