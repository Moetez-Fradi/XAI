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

## Next

- `build_corpus.py` — filter chains, lock held-out split  
- `embed_esm2.py` / `index_xqdrant.py`  
- Experiments A–F
