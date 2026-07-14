# Option 1 — Naive full-masked traversal (`recall_vs_ratio`)

**XQdrant change:** **none** — shipped `focus.masked=true` masks all layers.

**What it proves:** recall collapse vs **full-vector GT** as `D_sub/D` shrinks.

## Run

```bash
python run_option1_recall_collapse.py --mode http \
  --xqdrant-url http://127.0.0.1:6333 --dataset sift1m --queries 500
```

## Canonical results

| Corpus | Folder |
|--------|--------|
| SIFT | `experiments/2026-07-10_10-19-42__option1_naive_masked__recall_vs_ratio` |

Full-GT recall@10 (SIFT): ~0.006 @0.1 → ~0.08 @0.25 → ~0.26 @0.5 → ~0.46 @0.75 → ~0.99 @1.0.

**Note:** a separate high-D Option 1 HTTP run was not archived; use Option 2
`mask_from_layer` all-masked / layer=0 cells + X2 coherence for high-D baseline
narrative. Index: [`../../experiments/README.md`](../../experiments/README.md).
