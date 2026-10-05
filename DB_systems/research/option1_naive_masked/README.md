# M1 — Naive all-layer masking (`option1_naive_masked`)

**Paper:** M1 · §4.2 design · §5.3 results · Table 2: *baseline only*  
**API:** `focus.masked=true` with no `mask_from_layer` (masks every HNSW layer)  
**Folder name (legacy):** `option1_naive_masked`

**What it proves:** full-space Recall@10 falls as `D_sub/D` shrinks, but it tracks exact
subspace/full coherence (**C1**) almost exactly, i.e. masked traversal returns the
*subspace* neighbours and those differ from the full-space ones. The script records
recall against both ground truths plus coherence, so the run separates a ground-truth
mismatch (full-GT recall ≈ coherence, subspace-GT recall high) from a genuine
navigability failure (subspace-GT recall low).

No extra XQdrant change beyond shipped `focus.masked`.

## Run

```bash
python run_option1_recall_collapse.py --mode http \
  --xqdrant-url http://127.0.0.1:6333 --dataset sift1m --queries 500 --trials 5
```

## Canonical results

| Corpus | Folder under `DB_systems/experiments/` |
|--------|----------------------------------------|
| SIFT | `2026-07-10_10-19-42__option1_naive_masked__recall_vs_ratio` |

Full-GT recall@10 (SIFT, indicative): ~0.01 @0.1 → ~0.08 @0.25 → ~0.25 @0.5 → ~0.49 @0.75 → ~1.0 @1.0
(same values as C1 coherence). The 2026-07-10 run predates the subspace-GT column.

| SIFT, both GTs + coherence, T=5 (paper Fig. 3) | `2026-10-02_17-01-35__option1_naive_masked__recall_vs_ratio` |
|---|---|

Subspace-GT recall@10 (SIFT): 0.38 @0.1 → 0.77 @0.25 → 0.93 @0.5 → 0.98 @0.75 → 0.99 @1.0.
Without `--no-provision` the script now waits for HNSW indexing before querying.

High-D Option 1 was not archived as a separate HTTP run; use M2 low-`L` / all-masked
cells plus **C1** coherence for that narrative (paper Limitations).

See also: [`../README.md`](../README.md) · [`../../../xqdrant_docs/`](../../../xqdrant_docs/).
