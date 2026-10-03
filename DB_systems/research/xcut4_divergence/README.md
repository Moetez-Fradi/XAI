# Traversal divergence (`xcut4_divergence`)

Visited-set divergence harness (no M/K/C/V name). The VLDB 2027 paper's copy is in
[`../../VLDB 2027/research/xcut4_divergence/`](../../VLDB%202027/research/xcut4_divergence/).

**Ideal metric:** overlap of masked vs full *visited-node* sets (needs engine logging).  
**Runnable proxy today:** overlap of returned top-*k* sets between masked and full search.

## Run

```bash
python run_xcut4_divergence.py --mode http --xqdrant-url http://127.0.0.1:6333 \
  --dimensions 768 --queries 200

python run_xcut4_divergence.py --mode simulated --dimensions 768 --queries 50
```

Runner map: [`../README.md`](../README.md).
