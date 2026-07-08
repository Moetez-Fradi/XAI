# Cross-cut X3 — Fallback / verify pass (`verify_recall_recovery`)

**XQdrant change needed:** add `focus.verify: bool`. When true, after masked traversal do a final
full-distance rescore of the top-k before returning (cheap, since k is small).

**What it proves:** how much recall a cheap verification pass recovers, and what latency it costs —
i.e. whether masked search + verify is a good "fast filter with a safety net."

## Run

```bash
# after implementing focus.verify
python run_xcut3_verify.py --mode http --xqdrant-url http://127.0.0.1:6333 \
    --dimensions 768 --ratios 0.1 0.25 0.5 0.75

# offline
python run_xcut3_verify.py --mode simulated --dimensions 768 --queries 50
```

## Output

`experiments/<ts>__xcut3_verify_pass__verify_recall_recovery/`
- `results/xcut3_verify_d{D}.csv` — masked vs masked+verify recall + latency overhead per ratio
- `plots/xcut3_verify_recall_d{D}.png|pdf`
- `manifest.json`
