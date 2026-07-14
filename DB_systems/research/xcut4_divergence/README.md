# Cross-cut X4 — Candidate-list divergence (`traversal_divergence`)

> **Campaign status:** not run in the completed Options/X1–X3/Option4 sweep.
> Optional supplementary figure only — see `experiments/README.md`.

**XQdrant change needed (for the ideal metric):** log/return how often the masked-traversal
*visited-node* set diverges from the full-traversal visited-node set. Instrument this during the
Option 1/2 runs — no separate experiment. Feed the emitted CSV via `--divergence-csv`
(schema: `subspace_ratio,visited_divergence`).

**Runnable proxy now (no change):** divergence of the *returned top-k* sets between masked and full
search across ratios. Shows where masking starts steering results away from the full-space answer.

**What it proves:** *where* divergence begins as `D_sub` shrinks — a debugging aid and a
supplementary figure that complements the recall-collapse curve (Option 1).

## Run

```bash
# proxy only (works today)
python run_xcut4_divergence.py --mode http --xqdrant-url http://127.0.0.1:6333 --dimensions 768

# with instrumented visited-node divergence overlaid
python run_xcut4_divergence.py --mode http --xqdrant-url http://127.0.0.1:6333 \
    --dimensions 768 --divergence-csv /path/to/visited_divergence.csv

# offline
python run_xcut4_divergence.py --mode simulated --dimensions 768 --queries 50
```

## Output

`experiments/<ts>__xcut4_divergence__traversal_divergence/`
- `results/xcut4_divergence_d{D}.csv`
- `plots/xcut4_divergence_d{D}.png|pdf`
- `manifest.json`
