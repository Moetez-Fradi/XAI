# Fig 20 — payload filter × M2

Upserts `payload.bucket = id % 10` and filters `{0.1, 0.5, 1.0}` selectivities.

```bash
python run_filter_m2.py --mode simulated --queries 8 --trials 1
```
