# Bioinformatics — XQdrant explainable protein retrieval

Paper track: ESM2 embeddings + XQdrant dimension attribution + structural validation.
Scientific outline: [`steps.md`](./steps.md).

## Quick start

```bash
cd Bioinformatics

# 1) Python env (uv)
./setup_env.sh
# Linux + NVIDIA GPU (recommended for full-corpus embedding):
# ./setup_env.sh --cuda

# 2) Paper-scale downloads (default — PSB corpus)
./run_downloads.sh

# Smoke test only (~200 PDBs):
# ./run_downloads.sh --pilot

# 3) CLI tools (pick your OS)
./fetch_tools_macos.sh
# or
./fetch_tools_linux.sh
```

Activate later sessions with:

```bash
source .venv/bin/activate
```

## What paper mode downloads

| # | Script | Contents |
|---|--------|----------|
| 1 | `download_scop.py` | SCOPe class/desc/hierarchy + ASTRAL 40%/95% domain FASTA |
| 2 | `download_cath.py` | CATH-B newest + pinned `cath-domain-list.txt` |
| 3 | `download_pdb.py` | **10 000** unique PDBs, stratified by SCOPe fold (`.cif.gz`) |
| 4 | `download_uniprot.py` | SIFTS (UniProt/GO/EC), Swiss-Prot `.dat`+`.fasta`, JSON for mapped accessions |
| 5 | `download_pfam.py` | clans, pdbmap, Pfam-A regions |
| 6 | `download_oma_thermo.py` | OMA 1:1 thermo/meso pairs (up to 200) |
| 7 | `download_esm2.py` | ESM2 650M **PyTorch/safetensors only** (~2.6 GB; skips TF) |

Config: [`configs/download.yaml`](./configs/download.yaml).

## Size estimate (paper / PSB defaults)

| Component | Network ≈ | Disk ≈ |
|-----------|-----------|--------|
| PDB 10k × `.cif.gz` | **4–8 GB** | **4–8 GB** (kept compressed) |
| ESM2 650M (no TF) | **~2.6 GB** | **~2.6 GB** |
| Swiss-Prot `.dat.gz` + `.fasta.gz` | **~0.6–1.0 GB** | **~0.6–1.0 GB** (gzipped) |
| UniProt JSON (PDB-mapped, ~5–15k) | **0.5–2 GB** | **0.5–2 GB** |
| SIFTS GO/EC/UniProt CSVs | **~0.1–0.3 GB** | **~0.3–0.8 GB** (with decompress) |
| SCOPe + ASTRAL + CATH + Pfam | **≲0.5 GB** | **≲1 GB** |
| OMA thermo CSV | negligible | negligible |
| **Total `./run_downloads.sh`** | **~8–15 GB from the internet** | **~10–18 GB on disk** |

`fetch_tools_linux.sh` adds CLI packages (DSSP, Foldseek, BLAST+, …) — usually **well under 1–2 GB**, not part of the 8–15 GB data pull.

**~30 GB is not the download size** — that is recommended free disk later once you build Foldseek indexes, embeddings, and experiment archives.

Progress: each stage prints **percent complete + ETA** (PDB/UniProt batch lines; large files show byte %; ESM2 uses the Hugging Face progress bar). Re-runs skip existing files.

Runtime: paper PDB+UniProt can take **several hours** (rate limits + 10k files).

## Mac vs Linux

**Mac is fine for downloading.** For embedding the full 10k set with ESM2-650M, prefer **Linux + CUDA** (`./setup_env.sh --cuda`).

## Next (after downloads + tools)

```bash
cd Bioinformatics
source .venv/bin/activate

# 4) Smoke test — tiny corpus + ESM2 embed (safe; does not touch paper lock)
./run_smoke.sh
# optional: ./run_smoke.sh --limit-pdbs 20 --embed-limit 16 --device cuda

# 5) Full paper corpus (locks held-out split — run once)
.venv/bin/python scripts/build_corpus.py

# 6) Full embed (slow on CPU; prefer CUDA)
.venv/bin/python scripts/embed_esm2.py
# .venv/bin/python scripts/embed_esm2.py --device cuda --batch-size 8

# 7) Start XQdrant fork (attribution-capable)
./start_xqdrant.sh --daemon

# 8) Smoke index + Experiment A (retrieval + dims_explained)
./run_exp_a_smoke.sh

# 9) Paper index + Experiment A (resumes checkpoints by default)
.venv/bin/python scripts/index_xqdrant.py          # skip if already 22118; --restart to rebuild
.venv/bin/python scripts/exp_a_retrieval.py         # --restart to wipe results/exp_a
```

### Experiment A artifacts

| Path | Contents |
|------|----------|
| `data/processed/xqdrant/<coll>_manifest.json` | collection provenance + embedding fingerprint |
| `data/processed/xqdrant/<coll>_id_map.tsv` | `point_id → chain_id` |
| `data/processed/xqdrant/<coll>_checkpoint.json` | upsert resume cursor |
| `results/exp_a/run_config.json` | exact run settings + corpus/embed refs |
| `results/exp_a/query_list.txt` | holdout queries |
| `results/exp_a/checkpoints/queries.jsonl` | **per-query checkpoint** (neighbors + `dims_explained`) |
| `results/exp_a/metrics.json` | Recall/Precision + bootstrap CIs |
| `results/exp_a/per_query.json` | full final dump |

Resume is default; pass `--restart` (index: also `--recreate`) to start over.

### Experiment B (attribution validity)

**v1 (null baseline, locked):** `results/exp_b/` — global 6-feature test failed (p≈0.28). Do not overwrite.

**v2 (current):** per-residue binned profile validation — see `results/exp_b_v2/DESIGN.md`.

```bash
export PATH="$PWD/tools/mamba/envs/bio-tools/bin:$PATH"

# Smoke v2
./run_exp_b_v2_smoke.sh

# Paper v2 (DSSP resumes skipped JSONs; --store-profiles backfills existing chains)
.venv/bin/python scripts/annotate_dssp.py --from-exp-a --also-train 3000 --store-profiles
.venv/bin/python scripts/exp_b_v2_attribution.py
# optional later for PyMOL/Exp E full residue tables:
# .venv/bin/python scripts/annotate_dssp.py --from-exp-a --store-residues

# v1 commands (historical only — writes to results/exp_b/)
# .venv/bin/python scripts/annotate_dssp.py --from-exp-a --also-train 3000
# .venv/bin/python scripts/exp_b_attribution.py
```

### Exp A baselines (parallel-safe with Exp B)

```bash
source tools/env_linux.sh

# While B runs — start with FASTA baselines (no structure extract):
./run_exp_a_baselines.sh --fast-only

# Full baselines (chain PDBs + Foldseek + TM-align re-rank):
./run_exp_a_baselines.sh
# comparison → results/exp_a/baselines/comparison.tsv
```

Still to add later: Experiments C–F.
