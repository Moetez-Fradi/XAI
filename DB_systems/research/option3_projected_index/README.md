# Option 3 — Auxiliary projected index (`recall_memory_buildtime`)

> **Campaign status:** not implemented / not run. Plan says **future work** unless a small
> fixed focus-set palette exists. See `experiments/README.md`.

**XQdrant change needed (largest):** a new index-build path per registered focus set, plus query
routing to the projected index and a registration/selection API. This is the most invasive option.

**But the ceiling is measurable now:** this script builds a *stock* Qdrant collection over the
projected (focus-dim-only) vectors — which is exactly "a from-scratch full HNSW build in that
subspace." That gives you the **recall ceiling**, **build time**, and a **memory proxy** without
any XQdrant change. The Rust work is only to make it a routed, first-class feature.

**What it proves:** the recall you *could* achieve (Option 2's target), and the memory/build-time
price you'd pay per registered focus set.

## Run

```bash
python run_option3_projected_index.py --mode http --xqdrant-url http://127.0.0.1:6333 \
    --dimensions 768 --ratios 0.25 0.5 0.75

# offline
python run_option3_projected_index.py --mode simulated --dimensions 768 --queries 50
```

## Output

`experiments/<ts>__option3_projected_index__recall_memory_buildtime/`
- `results/option3_projected_index_d{D}.csv` — recall ceiling, p50, build time, memory proxy
- `plots/option3_recall_ceiling_d{D}.png|pdf`, `option3_cost_d{D}.png|pdf`
- `manifest.json`

To compare against Option 2's best config, read that run's CSV and overlay at matching `D_sub/D`.
Memory is a proxy (`points × D_sub × 4 bytes`), not RSS — disclose that in the paper.
