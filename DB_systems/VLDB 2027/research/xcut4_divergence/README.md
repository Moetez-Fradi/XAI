# Traversal divergence (`xcut4_divergence`)

**Paper:** §5.8, Figure 17 (X4 Jaccard proxy). No M/K/C/V name.

**Ideal metric:** overlap of masked vs full *visited-node* sets (needs engine logging).  
**Runnable proxy today:** overlap of returned top-*k* sets between masked and full search.

## Run

```bash
python run_xcut4_divergence.py --mode http --xqdrant-url http://127.0.0.1:6333 \
  --dimensions 768 --queries 200

python run_xcut4_divergence.py --mode simulated --dimensions 768 --queries 50
```

Runner map: [`../README.md`](../README.md).
