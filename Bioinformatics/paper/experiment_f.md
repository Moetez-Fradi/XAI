# Experiment F — Runtime / Workflow Comparison

**Status:** Complete — `results/exp_f/`

## Question (steps.md §4F)

For a single holdout protein query, how does an **XQdrant attributed search** workflow compare to traditional **BLAST/Foldseek → TM-align → manual PyMOL** in wall-clock time and analyst steps?

## Headline results (paper run, n=50)

| Workflow | Auto steps | Manual | Median auto (s) | Median total (s) |
|----------|------------|--------|-----------------|------------------|
| **XQdrant (cold: embed + search)** | 2 | 0 | **2.55** | **2.55** |
| **XQdrant (indexed: search only)** | 1 | 0 | **0.074** | **0.074** |
| **Foldseek → TM-align → PyMOL** | 2 | 1 | **4.17** | **604** |
| **BLAST → TM-align → PyMOL** | 2 | 1 | **0.61** | **601** |

- Speedup XQ (cold) vs Foldseek+TM automated: **1.6×**
- Speedup XQ (indexed) vs Foldseek+TM automated: **56×**
- BLAST+TM is fastest on automated steps alone (~0.6 s) but still requires manual PyMOL; XQ delivers **automatic `dims_explained` rationale** with zero manual steps.
- PyMOL time is a fixed **10 min estimate** (not measured).

## Run

```bash
./start_xqdrant.sh --daemon
./run_exp_f.sh --complete    # full cold path + BLAST + indexed pass + plots
# or stepwise:
./run_exp_f.sh --restart
./run_exp_f.sh --indexed
cd plot_scripts && python plot_exp_f_workflow.py
```

**Prerequisites:** Exp A index + `./run_exp_a_baselines.sh`. BLAST+ installed at `tools/blast/` (via `./fetch_tools_linux.sh`).

## Artifacts

```
results/exp_f/metrics.json              # cold path (embed + search + baselines)
results/exp_f/metrics_indexed.json      # production search-only latency
results/exp_f/workflow_comparison.tsv
results/exp_f/workflow_comparison_indexed.tsv
results/exp_f/summary.tsv
results/exp_f/per_query_timing.jsonl
results/exp_f/per_query_timing_indexed.jsonl
plot_scripts/outputs/exp_f_workflow_time.png
plot_scripts/outputs/exp_f_workflow_table.png
```

## Reporting notes

- Per-query times: single-thread Foldseek/BLAST/TM-align; hardware-specific.
- Two XQ latency columns: **cold** (includes ESM2 forward pass) vs **indexed** (precomputed embedding / production deployment).
- One-time DB/index build in `setup_costs.json` (amortized, excluded from per-query medians).
