# XQdrant: Path to a 12-Page Full Paper
 
**Current state (short track):** ~7-page ACM sigconf draft in `deliverable/`
with unified-host T=5 mean±std for the primary attribution + SIFT masked
results. Short-track variance/hardware gaps are closed; items below are
*optional full-track expansions*.
 
**Target (full track):** ~12 pages, research-track submission quality for
SIGMOD/VLDB/ICDE, and full-track (not just short-track) EDBT.
 
This plan is additive — nothing in the current 7 pages needs to be removed,
only strengthened and extended. Each section below states what to add, why it
closes a specific reviewer objection, and roughly how many pages it costs.
 
---
 
## Page Budget Overview
 
| Addition | New/expanded section | Est. pages added | Effort |
|---|---|---|---|
| 1. Variance & statistical rigor | Throughout §5 + new subsection | +0.5 | Medium (rerun existing experiments N times) |
| 2. Formal complexity analysis for M2 | New §4.7 | +1.0 | Low (mostly derivation, some validation) |
| 3. Second real dataset (beyond SIFT1M) | Expanded §5.1, new results interleaved | +1.5 | High (new data + full sweep) |
| 4. Filtering interaction | New §5.9 | +1.0 | Medium-High (new code path + experiments) |
| 5. Quantization interaction | New §5.10 or Discussion expansion | +0.75 | Medium (scope: characterize, not fully solve) |
| 6. Case study / worked example | New §6 (before Related Work) | +1.0 | Low-Medium (narrative + 1-2 figures) |
| 7. Visited-set divergence instrumentation (X4) | Expand §5.5 (K1) | +0.5 | Medium (new instrumentation) |
| 8. Expanded related work | Expand §7 | +0.5 | Low |
| 9. Reformatted/tightened intro & background | §1-2 | +0.25 | Low |
| **Total** | | **~7 pages added → ~14 pages raw, trim to ~12** | |
 
Budget in a little slack — plan to draft ~13-14 pages of content and cut ~1-2
pages during editing, which is much easier than trying to pad a thin draft to
exactly 12.
 
---
 
## 1. Variance and Statistical Rigor
 
**Why:** Flagged repeatedly as the single most reviewer-visible gap. Every
recall/latency number in the current draft is a single run.
 
**What to do:**
- Re-run each experiment (M1, M2, K1, C1, V1, M3) with **≥5 repeated trials**
  per configuration (different random seeds for query sampling; SIFT1M ground
  truth stays fixed since it's a real dataset).
- Report **mean ± std** or a **95% CI** for every recall and latency number
  currently stated as a single point estimate.
- Add error bars to every plot (matplotlib: `yerr=` on line plots, or shaded
  bands for the heatmaps' non-applicable — for heatmaps, report a companion
  std-dev heatmap or fold variance into the caption as a summary range).
- Add one new short paragraph explicitly stating your variance methodology
  (e.g., "each cell is the mean of 5 runs with independently sampled query
  sets; we report standard deviation in the appendix / inline").
**Where it lands:** Distributed across all of §5 (each figure gets error
bars), plus 2-3 new sentences per subsection stating the CI. No new section
needed — this is a rigor upgrade to existing content, not new content. Budget
~0.5 pages net (mostly captions and inline text growing slightly).
 
**Concrete tasks:**
- [x] Modify experiment harness to loop N=5 (or more) trials per config
- [x] Aggregate mean/std in the CSV output layer
- [x] Update all plotting scripts to render error bars / std bands
- [x] Re-run unified HTTP eval (`run_unified_paper_bench.sh`) and refresh
      numeric claims / figures in `deliverable/` with mean±std
      (`2026-07-21_10-15-46__paper_attr_unified_t5` + sibling masked runs)
---
 
## 2. Formal Complexity Analysis for M2 (Hybrid Masked Traversal)
 
**Why:** Your first draft had real Big-O treatment for attribution
($T_{XQ} = T_{HNSW} + k\cdot O(D) + O(k\log k)$) but nothing formal for the
masked-traversal side, which is now the paper's second major contribution.
SIGMOD/VLDB reviewers expect theoretical grounding alongside empirical results
for a systems paper, not just benchmarks.
 
**What to do:**
Add a new §4.7 ("Complexity of Hybrid Masked Traversal") deriving:
- Per-comparison cost as a function of cutoff $L$: coarse layers cost
  $O(D)$ per comparison, masked layers cost $O(D_{\mathrm{sub}})$, so expected
  traversal cost is a weighted combination based on the fraction of graph
  layers/comparisons above vs. below $L$.
- An explicit term for **why this doesn't reduce wall-clock time** despite
  reducing arithmetic — formalize the I/O-bound argument from your Discussion
  section as a simple cost model: $T_{\mathrm{traversal}} = T_{\mathrm{arith}} +
  T_{\mathrm{io}}$, and argue/show (via your K1 kernel isolation data, which
  you already have) that $T_{\mathrm{io}} \gg T_{\mathrm{arith}}$ at tested
  scales, so reducing the second term doesn't move the sum much. This turns
  your existing empirical negative result into a *predicted* one, which reads
  as far more rigorous.
- A short proposition/lemma-style claim (doesn't need to be heavyweight) about
  when masked traversal **could** yield a net win: i.e., the crossover
  point where $T_{\mathrm{arith}}$ dominates (very high $D$, or hardware with
  much slower per-comparison arithmetic relative to memory bandwidth, e.g.
  future SIMD-width increases). This gives the paper a forward-looking,
  falsifiable claim reviewers like.
**Where it lands:** New §4.7, ~1 page including a short derivation and maybe
one small illustrative figure (cost model diagram, can reuse Visualizer/plot
style consistent with your other figures).
 
**Concrete tasks:**
- [ ] Write the cost-model derivation (mostly math, low implementation cost)
- [ ] Cross-check the derived crossover condition against your K1 kernel
      throughput numbers to see if it's empirically plausible
- [ ] One supporting figure (cost model sketch, or a plot of measured
      $T_{\mathrm{arith}}/T_{\mathrm{io}}$ ratio across your existing $D$ values)
---
 
## 3. Second Real Dataset
 
**Why:** SIFT1M alone plus synthetic data is the most common "toy benchmark"
objection for a vector-DB systems paper today. A second real, differently-
shaped dataset substantially blunts this.
 
**What to do:** Pick one dataset that's meaningfully different in character
from SIFT1M (which is 128-dim SIFT descriptors, not a modern embedding):
- **Recommended:** a subset of a real sentence/text embedding corpus (e.g.
  embeddings from a public sentence-transformer model over a public text
  corpus — MS MARCO passages or similar, embedded at a realistic width like
  384 or 768). This directly matches your "encoder-scale widths" framing from
  the attribution section and closes the SIFT-is-not-representative gap in
  one move.
- **Alternative (lower effort):** GIST1M (960-dim, same SIFT-family tooling,
  so minimal new harness work) if a text-embedding pipeline is too much extra
  work before the deadline.
**What to run on it:** At minimum, replicate the M1/M2 recall-collapse and
recall-recovery experiments (your headline results) on the new dataset. You
don't need to replicate every ablation (K1, V1, M3) on both datasets — one
full pass (M1, M2, C1) plus a mention that K1/V1/M3 trends held qualitatively
is enough, and keeps this from ballooning the page budget.
 
**Where it lands:** Interleaved into existing §5.3-5.6 as a second series on
existing figures (e.g., add a third heatmap column-set), or as a compact new
§5.2b if the shape of the data differs enough to warrant separate discussion.
Aim for ~1.5 pages: 2-3 new/extended figures plus a short interpretive
paragraph per section on whether the new dataset confirms or complicates the
SIFT1M story.
 
**Concrete tasks:**
- [ ] Pick and download/prepare the second dataset
- [ ] Build ground truth (exact k-NN) for it, same as done for SIFT1M
- [ ] Re-run M1, M2, C1 sweeps
- [ ] Add to existing figures or create parallel ones
- [ ] Write 1-2 paragraphs synthesizing cross-dataset agreement/disagreement
---
 
## 4. Filtering Interaction
 
**Why:** Qdrant's headline feature is payload filtering combined with ANN
search. A masked-distance / attribution paper that never touches filtering
will draw an immediate "what about filtered queries?" from any reviewer
familiar with Qdrant or production vector-DB workloads.
 
**What to do:**
- Add filtering as an orthogonal dimension to at least the M2 hybrid-cutoff
  experiment: run the same recall/latency sweep with a moderate-selectivity
  payload filter active (e.g., filter that keeps ~10-50% of points), and
  report whether/how filtering changes the picture. Two plausible findings,
  both publishable:
  - Filtering + masking compose cleanly (filtered candidate set is smaller,
    so relative overhead of masking changes predictably) — good news story.
  - Filtering interacts badly with the hybrid cutoff (e.g., candidate
    starvation on the masked layers) — another honest negative/nuanced
    result consistent with your paper's voice.
- Keep scope tight: one dataset (whichever is more convenient — SIFT1M or the
  new real dataset), one or two filter selectivities, reusing the M2
  configuration space you've already built rather than a full new ablation
  grid.
**Where it lands:** New §5.9 ("Interaction with Payload Filtering"), ~1 page:
setup paragraph, 1 figure (recall or speedup vs. filter selectivity, ideally
overlaid with the unfiltered baseline), interpretive paragraph.
 
**Concrete tasks:**
- [ ] Confirm/implement that masked traversal composes with existing Qdrant
      filter pushdown in your fork (may already work — verify, don't assume)
- [ ] Design 2-3 filter selectivity levels
- [ ] Re-run M2 sweep at each selectivity level
- [ ] One comparison figure + write-up
---
 
## 5. Quantization Interaction
 
**Why:** You already flag this as a limitation/future work in the current
draft ("Quantization... A principled extension would define decompositions in
reconstructed space"). Reviewers will read that line and immediately want to
know more — even a partial empirical characterization (not a full solution)
converts a hand-wave into a contribution.
 
**What to do — keep this scoped, not a full redesign:**
- Run your existing attribution and/or M2 experiments against a
  **scalar-quantized (SQ)** index (Qdrant supports this natively, so this
  should be a configuration change, not new engine code) and report:
  - Whether attribution accuracy degrades (does the decomposition still sum
    exactly to the reported score in quantized space, or does it only
    approximate it? This is a concrete, checkable claim).
  - Whether M2's recall/latency picture changes materially under
    quantization.
- Explicitly scope this as **characterization, not a fix** — you are not
  obligated to solve the reconstructed-space decomposition problem for this
  paper, just to show what breaks and by how much, which is legitimate
  systems-paper content on its own.
**Where it lands:** Either a new short §5.10, or an expanded Discussion
paragraph with 1 supporting figure/table — whichever fits the page budget
better once the rest is drafted. Budget ~0.75 pages.
 
**Concrete tasks:**
- [ ] Build a quantized (SQ) version of at least one existing index
- [ ] Re-run attribution accuracy check (does decomposition sum match score?)
- [ ] Re-run a subset of the M2 sweep against the quantized index
- [ ] Write up findings, explicitly scoped as characterization
---
 
## 6. Case Study / Worked Example
 
**Why:** The paper is currently entirely abstract — no example ever shows a
human using attribution or masked search to actually understand or debug a
real retrieval. A half-page concrete example makes the whole paper easier to
grade for a reviewer skimming for "why does this matter."
 
**What to do:**
- Pick one query from one of your real datasets (SIFT1M or the new text
  corpus) where:
  - `with_dims_explained` reveals something interpretable (e.g., "this match
    is driven almost entirely by dimensions X, Y, Z" — ideally something
    that maps to an intuitive concept if using text embeddings, e.g. a
    specific semantic cluster).
  - Optionally, pair it with a `nearest.focus` example showing how
    restricting to a coordinate subset changes which results surface, tying
    back to your coherence diagnostic (C1) — e.g., "this focus set has low
    coherence, and indeed the top result changes entirely under masked
    search."
- Keep it concrete and visual: one small figure or table showing the query,
  its top attributed dimensions, and (if using the focus tie-in) the
  before/after result list.
**Where it lands:** New §6, placed right after Experimental Evaluation and
before Related Work (so it reads as "here's what this all means in practice"
before the paper moves to positioning against prior work). Budget ~1 page.
 
**Concrete tasks:**
- [ ] Pick a compelling, explainable example query
- [ ] Run it through `with_dims_explained` (and optionally `nearest.focus`)
- [ ] Build 1 figure/table
- [ ] Write ~0.5-0.75 pages of narrative around it
---
 
## 7. Visited-Set Divergence Instrumentation (X4)
 
**Why:** Already flagged as a limitation in your current Discussion
("We did not instrument visited-set divergence between masked and full
traversal directly, which would strengthen the mechanistic story"). This is
low-effort relative to payoff — you already have the traversal code, this is
mostly logging.
 
**What to do:**
- Instrument the M1/M2 traversal to log which nodes are visited during masked
  vs. full-distance search for the same queries.
- Report the divergence (e.g., Jaccard overlap of visited-node sets) as a
  function of cutoff $L$ and subspace ratio — this should directly
  corroborate/strengthen the coherence (C1) story: low-coherence configs
  should show high visited-set divergence, explaining *mechanistically* why
  recall collapses, not just *statistically* (via the k-NN overlap proxy you
  already have).
**Where it lands:** Fold into existing §5.5 (K1 kernel section) or §5.6 (C1
coherence section) as a strengthening sub-result rather than a whole new
section — it's evidence for a claim you already make, not a new claim.
Budget ~0.5 pages (1 figure + a paragraph).
 
**Concrete tasks:**
- [ ] Add visited-node-set logging to the traversal implementation
- [ ] Compute divergence metric (Jaccard or similar) vs. full-distance
      traversal on matched queries
- [ ] One figure correlating divergence with recall degradation
- [ ] 1-2 paragraphs tying it back to the C1 coherence narrative
---
 
## 8. Expanded Related Work
 
**Why:** Current Related Work is four short paragraphs (~0.4 pages). A
12-page full paper typically has a more thorough positioning section,
especially given the new content areas (filtering, quantization) pull in
additional related literature.
 
**What to add:**
- A short paragraph on **filtered ANN search** literature (this is an active
  research area — filtered-HNSW variants, pre/post-filtering tradeoffs) to
  support the new §5.9.
- A short paragraph on **quantization-aware search / reconstructed-space
  methods** to support the new quantization section.
- Slightly expand the existing paragraphs with 1-2 more specific citations
  each if you find natural ones while researching the above.
**Where it lands:** Expand §7 (or renumbered based on final structure) from
~0.4 to ~0.9 pages.
 
**Concrete tasks:**
- [ ] Find 2-4 citations on filtered ANN search
- [ ] Find 2-4 citations on quantization-aware / reconstructed-space search
- [ ] Write ~2 new short paragraphs
- [ ] Add corresponding BibTeX entries
---
 
## 9. Tightening Intro/Background (Editing Pass, Not New Content)
 
**Why:** With ~7 new pages of content, the paper's front matter should be
re-read once everything else is drafted to make sure the contributions list
still accurately previews the (now larger) paper, and that nothing in the
Introduction promises something the expanded paper doesn't deliver (or vice
versa — new contributions should be previewed).
 
**What to do:**
- Update the contributions bullet list to include: the complexity analysis,
  the second dataset, filtering/quantization characterization, and the case
  study, briefly.
- Update the abstract's final 1-2 sentences to gesture at the broader
  evaluation scope (second dataset, filtering/quantization) without bloating
  it — abstracts should stay roughly the same length even as the paper grows.
**Where it lands:** No page growth, or even slight tightening/consolidation
elsewhere to make room. Do this pass **last**, after all other sections are
drafted.
 
---
 
## Suggested Drafting Order
 
Given dependencies (some items need new experiments run before they can be
written, others are pure writing/derivation):
 
1. **Variance/CI reruns** (§1) — do this first; every other section's numbers
   should be reported with the same rigor once you're editing them anyway.
2. **Complexity analysis** (§2) — can be drafted in parallel with experiments
   since it's mostly derivation; use existing K1 data to validate.
3. **Second dataset** (§3) — highest effort, start early, run in parallel
   with other work while waiting on compute.
4. **Filtering interaction** (§4) — depends on confirming filter+mask
   composition works in your fork; verify this early so you're not blocked
   near the deadline.
5. **Quantization characterization** (§5) — similar to filtering, verify SQ
   index + attribution/masking composition works early.
6. **Visited-set divergence** (§7) — cheap instrumentation, can slot in
   whenever convenient once the traversal code is being touched anyway for
   items 3-5.
7. **Case study** (§6) — do this once the second dataset (§3) is in hand, so
   you can pick the most compelling example from real data.
8. **Related work expansion** (§8) — do alongside or after drafting §4/§5,
   since the new citations are naturally discovered while researching those.
9. **Front-matter tightening** (§9) — last step before final proofread.
## Target Timeline (assuming ~10-11 weeks before an October deadline)
 
| Weeks | Focus |
|---|---|
| 1-2 | Variance rerun infrastructure + second dataset acquisition/prep |
| 3-4 | Complexity analysis draft; second dataset experiments running |
| 5-6 | Filtering + quantization experiments and write-up |
| 7 | Visited-set divergence + case study |
| 8 | Related work expansion + all figures finalized with error bars |
| 9 | Full draft assembly, front-matter tightening, internal consistency pass |
| 10 | Buffer / address anything that slipped; proofread |
| 11 | Final formatting pass, submit |