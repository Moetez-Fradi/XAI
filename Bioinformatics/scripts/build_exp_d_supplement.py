#!/usr/bin/env python3
"""Build a supplemental chain corpus from Experiment D PDB downloads.

Processes PDB IDs listed in data/raw/thermo/exp_d_pdb_ids.txt (from
download_exp_d_structures.py). Only chains whose UniProt matches the Exp D
target set are kept. All supplement chains are assigned split=train.

Writes data/processed/corpus_supplement/:
  chains.csv, sequences.fasta, build_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus import (  # noqa: E402
    _find_cla,
    extract_polymer_chains,
    find_structure,
    load_scop_chain_map,
    load_sifts_uniprot,
    pass_filters,
)
from common_bio import (  # noqa: E402
    BIO_ROOT,
    BatchProgress,
    ensure_dir,
    load_yaml,
    raw_dir,
    resolve_path,
    write_json,
)


def load_target_uniprots(thermo_dir: Path) -> set[str]:
    tsv = thermo_dir / "exp_d_uniprot_pdbs.tsv"
    out: set[str] = set()
    if not tsv.exists():
        return out
    with tsv.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            u = (row.get("uniprot") or "").strip().upper()
            if u:
                out.add(u)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids-file", type=Path, default=None)
    ap.add_argument("--all-chains", action="store_true", help="Keep all valid chains from Exp D PDBs")
    args = ap.parse_args()

    dcfg = load_yaml("exp_d.yaml")
    cfg = load_yaml("corpus.yaml")
    outdir = resolve_path(dcfg.get("corpus_supplement_dir", "data/processed/corpus_supplement"))
    ensure_dir(outdir)

    thermo_dir = raw_dir("thermo")
    ids_file = args.ids_file or thermo_dir / "exp_d_pdb_ids.txt"
    if not ids_file.exists():
        print(f"ERROR: missing {ids_file}. Run download_exp_d_structures.py first.", file=sys.stderr)
        return 1

    pdb_ids = [
        ln.strip().lower()
        for ln in ids_file.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if not pdb_ids:
        print("ERROR: empty PDB ID list.", file=sys.stderr)
        return 1

    target_u = load_target_uniprots(thermo_dir)
    filter_u = not args.all_chains

    pdb_dir = raw_dir("pdb")
    scop_dir = raw_dir("scop")
    sifts_dir = raw_dir("sifts")
    cla = _find_cla(scop_dir)
    scop_map = load_scop_chain_map(cla)
    sifts_path = sifts_dir / "pdb_chain_uniprot.csv"
    if not sifts_path.exists():
        sifts_path = sifts_dir / "pdb_chain_uniprot.csv.gz"
    uniprot_map = load_sifts_uniprot(sifts_path) if sifts_path.exists() else {}

    rows: list[dict] = []
    skipped = {"no_structure": 0, "filter": 0, "no_scop": 0, "no_target_uniprot": 0, "parse_error": 0}
    progress = BatchProgress(len(pdb_ids), label="supplement-PDB")
    t0 = time.time()

    for i, pdb_id in enumerate(pdb_ids, 1):
        path = find_structure(pdb_dir, pdb_id)
        if path is None:
            skipped["no_structure"] += 1
            progress.tick(ok=False, force_print=(i == len(pdb_ids)))
            continue
        try:
            polymers = extract_polymer_chains(path)
        except Exception as e:  # noqa: BLE001
            skipped["parse_error"] += 1
            if skipped["parse_error"] <= 5:
                print(f"  parse fail {pdb_id}: {e}", file=sys.stderr)
            progress.tick(ok=False, force_print=(i == len(pdb_ids)))
            continue

        kept = 0
        for chain, seq in polymers:
            if not pass_filters(seq, cfg):
                skipped["filter"] += 1
                continue
            scop = scop_map.get((pdb_id, chain))
            if cfg.get("require_scop_fold", True) and not scop:
                skipped["no_scop"] += 1
                continue
            u = uniprot_map.get((pdb_id, chain), "").upper()
            if filter_u and u not in target_u:
                skipped["no_target_uniprot"] += 1
                continue
            rows.append(
                {
                    "chain_id": f"{pdb_id}_{chain}",
                    "pdb_id": pdb_id,
                    "chain": chain,
                    "length": len(seq),
                    "split": "train",
                    "scop_domain": (scop or {}).get("scop_domain", ""),
                    "scop_sccs": (scop or {}).get("scop_sccs", ""),
                    "scop_fold": (scop or {}).get("scop_fold", ""),
                    "scop_class": (scop or {}).get("scop_class", ""),
                    "uniprot": u,
                    "structure_file": str(path.relative_to(BIO_ROOT)),
                    "sequence": seq,
                }
            )
            kept += 1
        progress.tick(ok=kept > 0, force_print=(i == len(pdb_ids)))

    dedup: dict[str, dict] = {}
    for r in rows:
        dedup.setdefault(r["chain_id"], r)
    rows = sorted(dedup.values(), key=lambda r: r["chain_id"])

    if not rows:
        print("ERROR: no supplement chains kept.", file=sys.stderr)
        print(f"  skipped={skipped}", file=sys.stderr)
        return 1

    fieldnames = [
        "chain_id",
        "pdb_id",
        "chain",
        "length",
        "split",
        "scop_domain",
        "scop_sccs",
        "scop_fold",
        "scop_class",
        "uniprot",
        "structure_file",
        "sequence",
    ]
    csv_path = outdir / "chains.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    fasta = outdir / "sequences.fasta"
    with fasta.open("w") as f:
        for r in rows:
            f.write(f">{r['chain_id']}\n")
            seq = r["sequence"]
            for j in range(0, len(seq), 80):
                f.write(seq[j : j + 80] + "\n")

    write_json(
        outdir / "build_manifest.json",
        {
            "source": "exp_d_supplement",
            "n_chains": len(rows),
            "n_pdbs_requested": len(pdb_ids),
            "n_target_uniprots": len(target_u),
            "skipped": skipped,
            "elapsed_s": round(time.time() - t0, 1),
            "note": "All supplement chains are train split; paper holdout lock unchanged.",
        },
    )
    print(f"Supplement corpus: {len(rows)} chains → {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
