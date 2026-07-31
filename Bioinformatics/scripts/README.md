# Python scripts

Shared utilities: `common_bio.py`, `xqdrant_rest.py`, `exp_a_metrics_lib.py`, `exp_e_lib.py`, `exp_f_lib.py`.

## Downloads

| Script | Purpose |
|--------|---------|
| `download_pdb.py` | PDB mmCIF by SCOP stratification |
| `download_scop.py` | SCOPe tables + ASTRAL |
| `download_cath.py` | CATH domain lists |
| `download_uniprot.py` | SIFTS + Swiss-Prot + mapped JSON |
| `download_pfam.py` | Pfam regions |
| `download_oma_thermo.py` | OMA thermo/meso CSV |
| `download_esm2.py` | ESM2 650M weights |
| `download_exp_d_structures.py` | Extra PDBs for Exp D supplement |

## Corpus and embeddings

| Script | Purpose |
|--------|---------|
| `build_corpus.py` | SCOP-labeled chain table + holdout lock |
| `embed_esm2.py` | ESM2 vectors (`--layer`, `--pooling`, `--chains-file`) |
| `index_xqdrant.py` | Upsert embeddings into Qdrant |
| `annotate_dssp.py` | mkdssp per chain; optional binned profiles |
| `integrate_exp_d_supplement.py` | Merge Exp D structures into corpus |
| `build_exp_d_supplement.py` | Build supplement chain table |

## Experiment A

| Script | Purpose |
|--------|---------|
| `exp_a_retrieval.py` | Holdout retrieval + dims_explained |
| `recompute_exp_a_metrics.py` | Patch mAP into existing metrics |
| `exp_a_baseline_esm2.py` | Exact cosine baseline |
| `exp_a_baseline_blast.py` | BLASTp baseline |
| `exp_a_baseline_foldseek.py` | Foldseek baseline |
| `exp_a_baseline_tmalign.py` | TM-align rerank |
| `exp_a_aggregate_baselines.py` | Merge baseline metrics |
| `prep_exp_a_baselines.py` | FASTA/DB prep |

## Experiment B

| Script | Purpose |
|--------|---------|
| `exp_b_v3_attribution.py` | **paper** — validity + permutations |
| `exp_b_v2_attribution.py` | historical v2 |
| `exp_b_attribution.py` | historical v1 |
| `_exp_b_v3_probe.py` | debug / probe helper |

## Experiments C–F

| Script | Purpose |
|--------|---------|
| `exp_c_negative_controls.py` | ROC curves |
| `prep_exp_d_pairs.py` | Curate meso/thermo pairs |
| `curate_exp_d_literature.py` | Literature pair CSV |
| `exp_d_structure_features.py` | Ion pairs + SASA packing |
| `exp_d_thermo.py` | Case study scoring |
| `exp_d_stats.py` | Stats helpers |
| `prep_exp_e_benchmark.py` | Exp E holdout + sites |
| `exp_e_functional.py` | Site enrichment vs TM-align |
| `exp_f_runtime.py` | Workflow timing |

## Ablations

| Script | Purpose |
|--------|---------|
| `prep_exp_ablation.py` | Chain set for ablation |
| `extract_ablation_embeddings.py` | Alternate layer/pooling embeds |
| `exp_ablation_attribution.py` | Fold-gap scoring |

Run via root `run_*.sh` wrappers when available; see [../README.md](../README.md).
