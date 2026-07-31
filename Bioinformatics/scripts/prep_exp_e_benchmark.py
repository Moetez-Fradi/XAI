#!/usr/bin/env python3
"""Build Experiment E benchmark: holdout chains with UniProt functional sites.

Reads:
  corpus/chains.csv (holdout queries)
  SIFTS pdb_chain_uniprot, pdb_chain_enzyme, pdb_chain_go
  UniProt JSON per accession
  DSSP per-residue annotations (auto-backfilled for top candidates)

Writes:
  data/processed/exp_e/benchmark_queries.tsv
  data/processed/exp_e/sites/<chain_id>.json
  data/processed/exp_e/benchmark_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import ensure_dir, load_yaml, resolve_path, write_json  # noqa: E402
from exp_e_lib import (  # noqa: E402
    _SITE_TYPES_DEFAULT,
    ensure_benchmark_dssp,
    load_dssp_residues,
    load_sifts_chain_table,
    load_sifts_chain_uniprot,
    load_uniprot_entry,
    map_sites_for_chain,
    normalize_feature_type,
)


def gunzip_if_needed(path: Path) -> Path:
    if path.exists():
        return path
    gz = path.parent / f"{path.name}.gz"
    if gz.exists():
        import gzip

        with gzip.open(gz, "rt") as fin, path.open("w") as fout:
            fout.write(fin.read())
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-dssp", action="store_true", help="Do not backfill missing DSSP")
    args = ap.parse_args()

    ecfg = load_yaml("exp_e.yaml")
    smoke = bool(args.smoke)
    corpus_dir = resolve_path(
        ecfg.get("corpus_smoke_dir" if smoke else "corpus_dir", "data/processed/corpus")
    )
    chains_csv = corpus_dir / "chains.csv"
    if not chains_csv.exists():
        print(f"ERROR: missing {chains_csv}", file=sys.stderr)
        return 1

    proc_dir = ensure_dir(resolve_path(ecfg["processed_dir"]))
    sites_dir = ensure_dir(proc_dir / "sites")
    ann_dir = resolve_path(ecfg["annotations_dir"])
    uni_json = resolve_path(ecfg["uniprot_json_dir"])
    sifts_dir = resolve_path(ecfg.get("sifts_dir", "data/raw/sifts"))

    uniprot_csv = gunzip_if_needed(sifts_dir / "pdb_chain_uniprot.csv")
    enzyme_csv = gunzip_if_needed(sifts_dir / "pdb_chain_enzyme.csv")
    go_csv = gunzip_if_needed(sifts_dir / "pdb_chain_go.csv")

    if not uniprot_csv.exists():
        print("ERROR: run download_uniprot.py first (SIFTS)", file=sys.stderr)
        return 1

    sifts_map = load_sifts_chain_uniprot(uniprot_csv)
    ec_map = load_sifts_chain_table(enzyme_csv, ("PDB", "CHAIN"), "EC") if enzyme_csv.exists() else {}
    go_map = load_sifts_chain_table(go_csv, ("PDB", "CHAIN"), "GO_ID") if go_csv.exists() else {}

    site_types = {normalize_feature_type(t) for t in (ecfg.get("site_feature_types") or [])}
    site_types |= _SITE_TYPES_DEFAULT
    catalytic_kw = list(ecfg.get("catalytic_keywords") or [])
    buffer = int(ecfg.get("site_residue_buffer", 2))
    min_sites = int(ecfg.get("min_site_residues", 1))
    min_len = int(ecfg.get("min_chain_length", 100))
    require_ec_go = bool(ecfg.get("require_ec_or_go", True))
    max_q = int(
        ecfg.get("max_benchmark_queries_smoke" if smoke else "max_benchmark_queries", 200)
    )
    max_dssp = int(ecfg.get("max_dssp_annotate", 250))

    # Pass 1 — score holdout candidates from UniProt sites (no DSSP required yet)
    scored: list[dict] = []
    with chains_csv.open() as f:
        for rec in csv.DictReader(f):
            if rec.get("split") != "holdout":
                continue
            cid = rec["chain_id"]
            try:
                length = int(rec.get("length") or 0)
            except ValueError:
                length = 0
            if length < min_len:
                continue
            pdb_id, chain = cid.split("_", 1)
            key = (pdb_id.lower(), chain)
            segs = sifts_map.get(key) or []
            if not segs:
                continue
            acc = (segs[0].get("SP_PRIMARY") or segs[0].get("ACCESSION") or "").strip()
            if not acc:
                continue
            entry = load_uniprot_entry(uni_json / f"{acc}.json")
            if not entry:
                continue
            ecs = ec_map.get(key, set())
            gos = go_map.get(key, set())
            if require_ec_go and not ecs and not gos:
                continue
            site_doc = map_sites_for_chain(
                cid,
                acc,
                entry,
                segs,
                [],  # site count from UniProt mapping only
                site_types=site_types,
                catalytic_keywords=catalytic_kw,
                buffer=0,
            )
            if site_doc["n_site_residues"] < min_sites:
                continue
            scored.append(
                {
                    "chain_id": cid,
                    "uniprot_accession": acc,
                    "scop_fold": rec.get("scop_fold", ""),
                    "length": length,
                    "n_site_residues_est": site_doc["n_site_residues"],
                    "ec_numbers": ecs,
                    "go_ids": gos,
                    "segs": segs,
                    "entry": entry,
                }
            )

    scored.sort(key=lambda r: (-r["n_site_residues_est"], -r["length"], r["chain_id"]))
    dssp_targets = {r["chain_id"] for r in scored[: max_dssp if max_dssp > 0 else len(scored)]}

    if not args.skip_dssp and dssp_targets:
        missing_dssp = {
            cid
            for cid in dssp_targets
            if len(load_dssp_residues(ann_dir, cid)) < min_len
        }
        if missing_dssp:
            print(f"  Backfilling DSSP for {len(missing_dssp)} holdout benchmark candidates...")
            ensure_benchmark_dssp(
                missing_dssp,
                ann_dir,
                str(ecfg.get("mkdssp_bin", "auto")),
                chains_csv,
            )

    holdout: list[dict] = []
    for row in scored:
        cid = row["chain_id"]
        residues = load_dssp_residues(ann_dir, cid)
        if len(residues) < min_len:
            continue
        site_doc = map_sites_for_chain(
            cid,
            row["uniprot_accession"],
            row["entry"],
            row["segs"],
            residues,
            site_types=site_types,
            catalytic_keywords=catalytic_kw,
            buffer=buffer,
        )
        if site_doc["n_site_residues"] < min_sites:
            continue
        site_path = sites_dir / f"{cid}.json"
        if args.force or not site_path.exists():
            write_json(site_path, site_doc)
        holdout.append(
            {
                "chain_id": cid,
                "uniprot_accession": row["uniprot_accession"],
                "scop_fold": row["scop_fold"],
                "length": row["length"],
                "n_site_residues": site_doc["n_site_residues"],
                "ec_numbers": ";".join(sorted(row["ec_numbers"])),
                "go_ids": ";".join(sorted(row["go_ids"])),
                "site_json": str(site_path.relative_to(proc_dir)),
            }
        )

    holdout.sort(key=lambda r: (-r["n_site_residues"], -r["length"], r["chain_id"]))
    if max_q > 0:
        holdout = holdout[:max_q]

    bench_tsv = proc_dir / "benchmark_queries.tsv"
    with bench_tsv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "chain_id",
                "uniprot_accession",
                "scop_fold",
                "length",
                "n_site_residues",
                "ec_numbers",
                "go_ids",
                "site_json",
            ],
        )
        w.writeheader()
        w.writerows(holdout)

    manifest = {
        "mode": "smoke" if smoke else "paper",
        "n_benchmark_queries": len(holdout),
        "n_scored_candidates": len(scored),
        "n_dssp_backfill_targets": len(dssp_targets),
        "require_ec_or_go": require_ec_go,
        "min_site_residues": min_sites,
        "benchmark_tsv": str(bench_tsv),
        "sites_dir": str(sites_dir),
    }
    write_json(proc_dir / "benchmark_manifest.json", manifest)
    print(f"Exp E benchmark: {len(holdout)} queries (from {len(scored)} scored) → {bench_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
