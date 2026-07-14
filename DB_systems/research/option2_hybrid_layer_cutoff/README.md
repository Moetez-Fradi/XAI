# Option 2 — Hybrid layer cutoff (`recall_latency_layer_heatmap`)

**XQdrant:** `focus.mask_from_layer` **implemented**. Layer `L` is masked iff
`L < mask_from_layer`. **`0` = no masking** (plain nearest); omit field + `masked=true`
= all layers masked (Option 1). See `xqdrant_docs/option2.md`.

**What it proves:** whether full-distance coarse layers + masked bottom layers recover
recall while still cutting latency (headline Figure 4 candidate).

## Run

```bash
python run_option2_layer_sweep.py --mode http \
  --xqdrant-url http://127.0.0.1:6333 --qdrant-url http://127.0.0.1:6333 \
  --dataset synthetic --dimensions 768 1536 --num-vectors 50000 --queries 500 \
  --ratios 0.25 0.5 0.75 --layers 0 1 2 3 --ef-search 128 --k 10
```

Prefer server with `XQDRANT_MASKED_KERNEL=gather` for e2e after X1.

## Canonical results

| Corpus | Folder |
|--------|--------|
| High-D + gather | `experiments/option2_highD_gather_d768_1536_n50k_q500_v1` |
| SIFT + gather | `experiments/option2_sift1m_gather_q500_v1` |
| High-D + repack | `experiments/option2_highD_d768_1536_n50k_q500_v1` |

**Keep:** hybrid recovers navigability (esp. SIFT). **Latency:** speedup vs full still
&lt; 1 even with gather. Index: [`../../experiments/README.md`](../../experiments/README.md).
