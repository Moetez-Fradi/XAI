# Results layout

Experiment outputs are under `results/<name>/`. Checkpoints allow resume unless `--restart` is passed.

| Directory | Experiment | Key files |
|-----------|------------|-----------|
| `exp_a/` | Retrieval + attribution stability | `metrics.json`, `checkpoints/queries.jsonl`, `baselines/` |
| `exp_a_smoke/` | Smoke retrieval | same layout, smaller |
| `exp_b/` | **historical v1** | locked — do not overwrite |
| `exp_b_v2/` | **historical v2** | locked — do not overwrite |
| `exp_b_v3/` | Attribution validity (paper) | `metrics.json`, `pairs.json`, `summary.tsv` |
| `exp_c/` | ROC / negative controls | `summary.tsv`, `roc_curves.tsv` |
| `exp_d/` | Thermostability case study | `metrics.json`, `significance.tsv`, `pairs_by_source.tsv` |
| `exp_e/` | Active-site localization | `metrics.json`, `localization_comparison.tsv` |
| `exp_f/` | Workflow timing | `metrics.json`, `metrics_indexed.json`, `workflow_comparison.tsv` |
| `exp_ablation/` | Layer/pooling ablation | `metrics.json`, `ablation_comparison.tsv` |

Figures are generated separately into `plot_scripts/outputs/` and copied to `Deliverables/figures/` via `make figures`.

To regenerate plots after editing results:

```bash
./plot_scripts/run_all.sh
make -C Deliverables figures
```
