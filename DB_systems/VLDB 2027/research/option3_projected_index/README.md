# Future work — projected index (`option3_projected_index`)

**Paper:** mentioned in §6–7 as the natural next step; **not** part of the short-paper
eval (no M-name). Folder kept for the ceiling measurement harness.

**Idea:** build a dedicated HNSW over projected (focus-dim-only) vectors per registered
focus set — sidesteps M1 navigability failure, at memory/build cost per palette entry.

**Status:** not implemented as an XQdrant feature. This script measures the *ceiling*
today by creating a stock collection over projected vectors (recall / build time /
memory proxy) with no Rust change.

## Run

```bash
python run_option3_projected_index.py --mode http --xqdrant-url http://127.0.0.1:6333 \
  --dimensions 768 --ratios 0.25 0.5 0.75

python run_option3_projected_index.py --mode simulated --dimensions 768 --queries 50
```

Output: `experiments/<ts>__option3_projected_index__recall_memory_buildtime/`.

Compare against best **M2** cells at matching `D_sub/D`. Memory is a proxy
(`points × D_sub × 4`), not RSS.

Short-paper map: [`../README.md`](../README.md).
