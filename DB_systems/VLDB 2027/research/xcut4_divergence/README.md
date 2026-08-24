# Full-track only — traversal divergence (`xcut4_divergence`)

**Paper:** *not* in the short draft. Listed under Limitations / full-paper plan as
optional visited-set instrumentation. No M/K/C/V name.

**Ideal metric:** overlap of masked vs full *visited-node* sets (needs engine logging).  
**Runnable proxy today:** overlap of returned top-*k* sets between masked and full search.

Not required to reproduce the short paper. See [`../../full_paper_plan/steps.md`](../../full_paper_plan/steps.md).

## Run

```bash
python run_xcut4_divergence.py --mode http --xqdrant-url http://127.0.0.1:6333 \
  --dimensions 768 --queries 200

python run_xcut4_divergence.py --mode simulated --dimensions 768 --queries 50
```

Short-paper map: [`../README.md`](../README.md).
