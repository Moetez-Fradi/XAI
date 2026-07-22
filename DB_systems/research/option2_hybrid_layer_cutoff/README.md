# M2 — Hybrid layer cutoff (`option2_hybrid_layer_cutoff`)

**Paper:** M2 · §4.3 design · §5.4 results · Table 2: *keep* (navigability; e2e speedup &lt; 1)  
**API:** `focus.masked=true` + `mask_from_layer=L`  
**Folder name (legacy):** `option2_hybrid_layer_cutoff`  
**Design doc:** [`../../../xqdrant_docs/option2.md`](../../../xqdrant_docs/option2.md)

**Semantics:** layer `ℓ` uses masked distance iff `ℓ < L`; otherwise full distance.
`L=0` ≡ plain nearest; omit cutoff + `masked=true` ≡ **M1** (all layers masked).

**What it proves:** full-distance coarse layers restore navigability (esp. SIFT);
end-to-end speedup vs full search still stays &lt; 1 (motivates **K1** / bottleneck claim).

## Run

```bash
python run_option2_layer_sweep.py --mode http \
  --xqdrant-url http://127.0.0.1:6333 --qdrant-url http://127.0.0.1:6333 \
  --dataset sift1m --queries 500 --trials 5 \
  --ratios 0.25 0.5 0.75 --layers 0 1 2 3 --ef-search 128 --k 10
```

Prefer server with `XQDRANT_MASKED_KERNEL=gather` (paper default / **K1**).

## Canonical results

| Corpus | Folder under `DB_systems/experiments/` |
|--------|----------------------------------------|
| High-D + gather | `option2_highD_gather_d768_1536_n50k_q500_v1` |
| SIFT + gather | `option2_sift1m_gather_q500_v1` |
| High-D + repack | `option2_highD_d768_1536_n50k_q500_v1` |

See also: [`../README.md`](../README.md).
