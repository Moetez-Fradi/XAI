# Configuration files

All paths in YAML are relative to `Bioinformatics/` unless absolute. Loaded by `scripts/common_bio.load_yaml()`.

| File | Used by |
|------|---------|
| `download.yaml` | `run_downloads.sh`, all `download_*.py` |
| `corpus.yaml` | `build_corpus.py`, embed/index, experiments |
| `esm2.yaml` | `embed_esm2.py`, `extract_ablation_embeddings.py`, Exp F |
| `paths.yaml` | shared path overrides (if referenced) |
| `xqdrant.yaml` | `index_xqdrant.py`, retrieval, Exp C/D/E/F |
| `exp_a_baselines.yaml` | `run_exp_a_baselines.sh`, baseline scripts |
| `exp_b.yaml` | **historical v1** |
| `exp_b_v2.yaml` | **historical v2** binned profiles |
| `exp_b_v3.yaml` | **current Exp B** — map, permutations, ablation top-N |
| `exp_c.yaml` | `run_exp_c.sh`, ROC / negative controls |
| `exp_d.yaml` | thermo pairs, ion-pair cutoff, corpus supplement |
| `exp_e.yaml` | active-site benchmark, annotations |
| `exp_f.yaml` | workflow timing, Foldseek/BLAST settings |
| `exp_ablation.yaml` | layer/pooling ablation chain set |

Edit a config, then re-run the matching `run_exp_*.sh` or script with `--restart` if outputs must be rebuilt.
