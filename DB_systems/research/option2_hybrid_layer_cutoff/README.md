# M2 — Hybrid layer cutoff (`option2_hybrid_layer_cutoff`)

**Paper:** M2 · §4.3 design · §5.4 results · Table 2: *keep* (navigability; e2e speedup &lt; 1)  
**API:** `focus.masked=true` + `mask_from_layer=L`  
**Folder name (legacy):** `option2_hybrid_layer_cutoff`  
**Design doc:** [`../../../xqdrant_docs/option2.md`](../../../xqdrant_docs/option2.md)

**Semantics:** layer `ℓ` uses masked distance iff `ℓ < L`; otherwise full distance.
`L=0` ≡ plain nearest; omit cutoff + `masked=true` ≡ **M1** (all layers masked).

**What it proves:** on one SIFT1M index, the cutoff barely changes subspace recall
relative to M1 (at most +0.04 at ratio 0.1, ≤0.01 at ≥0.25; `L=6` reproduces M1), so
full-distance coarse layers are not what limits small focus sets. On the i.i.d. synthetic
D=768 corpus, plain search itself reaches only 0.19 Recall@10 at ef=128, so that corpus
cannot inform navigability. Speedup vs full search stays < 1 (see **K1** / in-engine timing).

Recall is measured against **subspace** GT; compare with M1 only on the same index
(absolute recall varies across index builds).

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
| SIFT, same index as M1/M3/V1, L=1..6, ratios 0.1–0.5, T=5 (paper Table 2) | `2026-10-03_15-41-13__option2_hybrid_layer_cutoff__recall_latency_layer_heatmap` |
| Synthetic D=768, T=5 | `2026-10-03_16-07-19__option2_hybrid_layer_cutoff__recall_latency_layer_heatmap` |

The 2026-10-02/03 runs were made on a second host (Ryzen 5 220, XQdrant `ed5ca79` in WSL2)
with `RESEARCH_HTTP_CLOSE=1`; use their recall, not their client-side latency.

See also: [`../README.md`](../README.md).
