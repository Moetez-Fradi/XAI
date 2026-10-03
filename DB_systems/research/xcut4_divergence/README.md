# Future work — traversal divergence (`xcut4_divergence`)

**Paper:** *not* reported. Optional visited-set instrumentation. No M/K/C/V name.

**Ideal metric:** overlap of masked vs full *visited-node* sets (needs engine logging).  
**Runnable proxy today:** overlap of returned top-*k* sets between masked and full search.

Not required to reproduce the paper.

## Run

```bash
python run_xcut4_divergence.py --mode http --xqdrant-url http://127.0.0.1:6333 \
  --dimensions 768 --queries 200

python run_xcut4_divergence.py --mode simulated --dimensions 768 --queries 50
```

Paper map: [`../README.md`](../README.md).
