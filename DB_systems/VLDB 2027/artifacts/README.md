# XQdrant VLDB 2027 reproducibility package

Public entry for the PVLDB artifact block.

- **Paper source:** [`../paper/`](../paper/)
- **Experiment index (every figure → script):** [`../README.md`](../README.md)
- **Engine fork:** https://github.com/Moetez-Fradi/XQdrant (branch `DB/masked-distance-HNSW`)
- **Vanilla baseline:** https://github.com/qdrant/qdrant
- **This repo / branch:** https://github.com/Moetez-Fradi/XAI/tree/vldb27/DB_systems/VLDB%202027

No access monitoring is enabled. Canonical figures shipped with the PDF live in [`../paper/figures/`](../paper/figures/). Regenerable CSVs go under [`../experiments/`](../experiments/).

```bash
cd ..
./run_unified_vldb_bench.sh --smoke --simulated
```
