# Option 2 — Hybrid layer cutoff (`recall_latency_layer_heatmap`)

**XQdrant change needed:** add `focus.mask_from_layer: usize` (REST + gRPC), thread it to the
scorer, and branch the distance function on `current_layer < mask_from_layer` in the traversal
loop. `0` = fully masked (current Option 1); `max_layer` = fully unmasked.

**What it proves:** whether keeping full-distance routing on the coarse top layers while masking
only the bottom layers **recovers recall while still cutting latency**. The priority result — the
new Figure 4 candidate.

## Run

```bash
# after implementing focus.mask_from_layer
python run_option2_layer_sweep.py --mode http --xqdrant-url http://127.0.0.1:6333 \
    --dimensions 768 --ratios 0.25 0.5 0.75 --layers 0 1 2 3

# offline pipeline check
python run_option2_layer_sweep.py --mode simulated --dimensions 768 --queries 50
```

## Output

`experiments/<ts>__option2_hybrid_layer_cutoff__recall_latency_layer_heatmap/`
- `results/option2_layer_sweep_d{D}.csv` — one row per `(mask_from_layer, ratio)`
- `plots/option2_recall_heatmap_d{D}.png|pdf` and `option2_speedup_heatmap_d{D}.png|pdf`
- `manifest.json`

**Keep condition:** any `mask_from_layer` where speedup > 1 and recall stays within ~0.01–0.02 of
the full-vector baseline. If XQdrant does not yet know `mask_from_layer`, cells show as `NaN` and
`unsupported_queries` is non-zero — that is the signal the Rust change is still pending.
