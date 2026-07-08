# Option 4 — Weighted blend distance (`recall_speed_alpha`)

**XQdrant change needed:** add `focus.alpha: f32`. The scorer computes
`d = alpha*d_full + (1-alpha)*d_focus`. On its own this does **not** save full-distance work
(both distances are computed) — it only biases routing. Pair it with Option 2's `mask_from_layer`
for real latency gains. Treat as a tuning knob, not a standalone speedup.

**What it proves:** whether blending lets you mask more aggressively (lower `mask_from_layer`)
while keeping recall acceptable.

## Run

```bash
# after implementing focus.alpha (and ideally focus.mask_from_layer)
python run_option4_alpha_sweep.py --mode http --xqdrant-url http://127.0.0.1:6333 \
    --dimensions 768 --ratio 0.25 --alphas 0 0.25 0.5 0.75 1.0 --layers 0 1

# offline
python run_option4_alpha_sweep.py --mode simulated --dimensions 768 --queries 50
```

## Output

`experiments/<ts>__option4_weighted_blend__recall_speed_alpha/`
- `results/option4_alpha_sweep_d{D}.csv` — one row per `(mask_from_layer, alpha)`
- `plots/option4_alpha_recall_d{D}.png|pdf`
- `manifest.json`

**Keep condition:** blending recovers meaningful recall at low `mask_from_layer`. **Skip:** if
`alpha` barely moves the recall/speed frontier vs. a hard layer cutoff, drop it.
