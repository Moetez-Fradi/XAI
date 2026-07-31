# Data layout

Large binaries and downloads live here. Not all subdirs exist until you run downloads and pipeline steps.

```
data/
├── raw/                 # downloaded sources (PDB, SCOP, CATH, UniProt, Pfam, thermo)
├── processed/           # derived tables and experiment inputs
│   ├── corpus/          # paper chain split (holdout lock)
│   ├── corpus_smoke/    # tiny smoke corpus
│   ├── corpus_merged/   # paper + Exp D supplement
│   ├── xqdrant/         # index manifests, id maps, checkpoints
│   ├── exp_d/           # thermo pair lists
│   ├── exp_e/           # site annotations per chain
│   └── exp_ablation/    # ablation chain list
├── annotations/         # DSSP JSON per chain, exp_d structure features
├── baselines/           # BLAST/Foldseek DBs and prep for Exp A
└── xqdrant_storage/     # local Qdrant fork persistence
```

Embeddings (separate from `data/`): `embeddings/esm2_t33_650M/`, `embeddings/smoke/`, `embeddings/ablation/`.

Models: `models/esm2_t33_650M_UR50D/`.

Do not commit multi-GB artifacts; use `.gitignore` patterns already in repo.
