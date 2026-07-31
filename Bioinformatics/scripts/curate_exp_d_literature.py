#!/usr/bin/env python3
"""Generate literature-curated thermo/meso pairs for Experiment D.

Reads ortholog group definitions from configs/exp_d.yaml and writes
data/raw/thermo/literature_pairs.csv with chain IDs present in the corpus.

Each group specifies mesophile / thermophile UniProt accessions (with optional
aliases) and generates one pair per (meso chain, thermo PDB representative).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import ensure_dir, load_yaml, resolve_path, write_json  # noqa: E402


def load_corpus(corpus_csv: Path) -> list[dict]:
    with corpus_csv.open() as f:
        return list(csv.DictReader(f))


def pick_meso_chains(
    rows: list[dict],
    uniprots: set[str],
    *,
    explicit: list[str],
    max_meso: int,
    min_len: int,
    prefer_holdout: bool,
) -> list[dict]:
    if explicit:
        by_id = {r["chain_id"]: r for r in rows}
        out = [by_id[cid] for cid in explicit if cid in by_id]
        return out[:max_meso]

    cands = [
        r
        for r in rows
        if r.get("uniprot") in uniprots and int(r.get("length") or 0) >= min_len
    ]
    # Prefer chain A/B from same structure; holdout first
    cands.sort(
        key=lambda r: (
            r["split"] != "holdout" if prefer_holdout else False,
            r["chain"] not in {"A", "B"},
            r["chain_id"],
        )
    )
    # Deduplicate by pdb_id — keep best chain per structure
    seen_pdb: set[str] = set()
    out: list[dict] = []
    for r in cands:
        pid = r["pdb_id"].lower()
        if pid in seen_pdb:
            continue
        seen_pdb.add(pid)
        out.append(r)
        if len(out) >= max_meso:
            break
    return out


def pick_thermo_reps(
    rows: list[dict],
    uniprots: set[str],
    thermo_pdb_prefixes: set[str],
    *,
    min_len: int,
    one_per_pdb: bool,
) -> list[dict]:
    def is_thermo(r: dict) -> bool:
        u = r.get("uniprot", "")
        pid = r.get("pdb_id", "").upper()
        if u in uniprots:
            return True
        return any(pid.startswith(p) for p in thermo_pdb_prefixes)

    cands = [
        r
        for r in rows
        if is_thermo(r)
        and r.get("chain") == "A"
        and int(r.get("length") or 0) >= min_len
    ]
    if not cands:
        cands = [r for r in rows if is_thermo(r) and int(r.get("length") or 0) >= min_len]

    cands.sort(key=lambda r: (r["chain"] != "A", r["chain_id"]))
    if not one_per_pdb:
        return cands

    reps: dict[str, dict] = {}
    for r in cands:
        reps.setdefault(r["pdb_id"].lower(), r)
    return sorted(reps.values(), key=lambda r: r["chain_id"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dcfg = load_yaml("exp_d.yaml")
    corpus_cfg = load_yaml("corpus.yaml")
    smoke = bool(args.smoke)

    if not smoke and dcfg.get("use_merged_corpus", True):
        merged = resolve_path(dcfg.get("corpus_merged_dir", "data/processed/corpus_merged"))
        corpus_dir = merged if (merged / "chains.csv").exists() else resolve_path(corpus_cfg["paper_outdir"])
    else:
        corpus_dir = resolve_path(
            corpus_cfg["smoke_outdir"] if smoke else corpus_cfg["paper_outdir"]
        )
    rows = load_corpus(corpus_dir / "chains.csv")

    thermo_u = set(dcfg.get("thermo_uniprot") or [])
    thermo_pdb = {p.upper() for p in (dcfg.get("thermo_pdb_prefixes") or [])}

    out_path = resolve_path(dcfg.get("literature_pairs_csv", "data/raw/thermo/literature_pairs.csv"))
    ensure_dir(out_path.parent)

    lit_rows: list[dict] = []
    group_stats: list[dict] = []

    for grp in dcfg.get("ortholog_groups") or []:
        meso_u = set(grp.get("meso_uniprot") or [])
        for alias in grp.get("meso_uniprot_aliases") or []:
            meso_u.add(str(alias))
        # Group-specific thermo IDs only (do not union global thermo_uniprot — that
        # would cross-pair GroEL meso with unrelated OMA enzyme thermo structures).
        thermo_g = set(grp.get("thermo_uniprot") or [])
        thermo_g |= set(grp.get("thermo_uniprot_extra") or [])

        min_len = int(grp.get("min_length", dcfg.get("min_chain_length", 150)))
        meso_chains = pick_meso_chains(
            rows,
            meso_u,
            explicit=list(grp.get("meso_chain_ids") or []),
            max_meso=int(grp.get("max_meso_chains", 3)),
            min_len=min_len,
            prefer_holdout=bool(grp.get("prefer_holdout", True)),
        )
        thermo_chains = pick_thermo_reps(
            rows,
            thermo_g,
            thermo_pdb,
            min_len=min_len,
            one_per_pdb=bool(grp.get("one_thermo_per_pdb", True)),
        )
        n = 0
        for m in meso_chains:
            for t in thermo_chains:
                if m["chain_id"] == t["chain_id"]:
                    continue
                lit_rows.append(
                    {
                        "uniprot_thermo": t.get("uniprot", ""),
                        "uniprot_meso": m.get("uniprot", ""),
                        "pdb_thermo": t["chain_id"],
                        "pdb_meso": m["chain_id"],
                        "note": grp.get("name", ""),
                        "citation": grp.get("citation", ""),
                    }
                )
                n += 1
        group_stats.append(
            {
                "name": grp.get("name"),
                "meso_chains": len(meso_chains),
                "thermo_reps": len(thermo_chains),
                "pairs": n,
            }
        )

    # Append manual rows from stub (explicit chain overrides)
    stub = resolve_path(dcfg.get("literature_pairs_stub", "data/raw/thermo/literature_pairs.stub.csv"))
    if stub.exists():
        with stub.open() as f:
            for row in csv.DictReader(f):
                if (row.get("pdb_meso") or row.get("meso_chain_id") or "").startswith("#"):
                    continue
                meso = row.get("pdb_meso") or row.get("meso_chain_id") or ""
                thermo = row.get("pdb_thermo") or row.get("thermo_chain_id") or ""
                if meso and thermo:
                    lit_rows.append(row)

    fields = [
        "uniprot_thermo",
        "uniprot_meso",
        "pdb_thermo",
        "pdb_meso",
        "note",
        "citation",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(lit_rows)

    write_json(
        out_path.parent / "literature_manifest.json",
        {"n_pairs": len(lit_rows), "groups": group_stats, "csv": str(out_path)},
    )
    print(f"Literature pairs: {len(lit_rows)} → {out_path}")
    for g in group_stats:
        print(f"  {g['name']}: {g['pairs']} pairs ({g['meso_chains']} meso × {g['thermo_reps']} thermo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
