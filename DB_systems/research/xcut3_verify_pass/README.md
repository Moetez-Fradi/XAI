# V1 — Verify / overfetch (`xcut3_verify_pass`)

**Paper:** V1 · §4.6 design · §5.7 results · Table 2: *conditional*  
**API:** `focus.verify=true` (requires `masked=true`); overfetch via `limit = m·k`  
**Folder name (legacy):** `xcut3_verify_pass`  
**Design doc:** [`../../../xqdrant_docs/X3.md`](../../../xqdrant_docs/X3.md)

**What it proves:**

- At overfetch **m=1**, Recall@k is unchanged (same candidate set; only scores/order).
- At **m&gt;1** on coherent data (SIFT, hybrid *L*=1), full-space recall recovers
  (0.48 → 0.91 at ratio 0.75, m=8) for +0.16–0.26 ms of server-side time.
- The ~+45 ms p50 reported by earlier runs is a client-side TCP delayed-ACK stall on the
  larger (limit=80) response over a keep-alive connection, not engine work. Measure with
  `RESEARCH_HTTP_CLOSE=1` or Qdrant's server-side `time` field.
- On the synthetic high-D corpus overfetch cannot rescue recall: plain search there only
  reaches 0.19 Recall@10.

## Run

```bash
# SIFT (paper primary)
python run_xcut3_verify.py --mode http \
  --xqdrant-url http://127.0.0.1:6333 --qdrant-url http://127.0.0.1:6333 \
  --dataset sift1m --queries 500 --trials 5 \
  --ratios 0.1 0.25 0.5 0.75 --overfetch 1 8 --layers -1 1 \
  --ef-search 128 --k 10

# High-D synthetic
python run_xcut3_verify.py --mode http \
  --xqdrant-url http://127.0.0.1:6333 \
  --dataset synthetic --dimensions 768 1536 --num-vectors 50000 \
  --queries 500 --trials 5 \
  --ratios 0.1 0.25 0.5 0.75 --overfetch 1 8 --layers -1 1
```

`--layers -1` = fully masked (M1); `1` = hybrid `mask_from_layer=1` (M2).

## Canonical results

| Corpus | Folder under `DB_systems/experiments/` |
|--------|----------------------------------------|
| High-D | `2026-07-14_09-34-50__xcut3_highD_fullGT_overfetch_hybrid_v1` |
| SIFT | `2026-07-14_09-46-22__xcut3_sift_fullGT_overfetch_hybrid_v1` |
| SIFT, T=5, same index as M1/M2/M3 (paper Fig. 5) | `2026-10-02_17-56-19__xcut3_verify_pass__verify_recall_recovery` |

Ignore earlier `xcut3_verify_pass` folders that used subspace-GT-only or simulated mode.

See also: [`../README.md`](../README.md).
