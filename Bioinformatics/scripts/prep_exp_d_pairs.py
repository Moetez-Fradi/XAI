#!/usr/bin/env python3
"""Build thermo/meso chain pairs for Experiment D.

Sources (priority):
  1) literature_pairs.stub.csv (manual pdb/chain or uniprot pairs)
  2) OMA ortholog CSV (THE* species) mapped to corpus
  3) Auto: same SCOP fold, thermo P0A6F5/P0A6F9 vs mesophilic holdout

Writes data/processed/exp_d/pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import ensure_dir, load_yaml, resolve_path, write_json  # noqa: E402


def is_thermo_row(row: dict, thermo_u: set[str], thermo_pdb: set[str]) -> bool:
    u = row.get("uniprot", "")
    pid = row.get("pdb_id", "").upper()
    if u in thermo_u:
        return True
    return any(pid.startswith(p) for p in thermo_pdb)


def superfamily(sccs: str) -> str:
    bits = (sccs or "").split(".")
    return ".".join(bits[:3]) if len(bits) >= 3 else sccs


def resolve_corpus_dir(dcfg: dict, corpus_cfg: dict, smoke: bool) -> Path:
    if not smoke and dcfg.get("use_merged_corpus", True):
        merged = resolve_path(dcfg.get("corpus_merged_dir", "data/processed/corpus_merged"))
        if (merged / "chains.csv").exists():
            return merged
    return resolve_path(
        corpus_cfg["smoke_outdir"] if smoke else corpus_cfg["paper_outdir"]
    )


def load_corpus_chains(corpus_csv: Path) -> list[dict]:
    with corpus_csv.open() as f:
        return list(csv.DictReader(f))


def _best_chain(cands: list[dict], prefer_holdout: bool = True) -> dict | None:
    if not cands:
        return None
    return sorted(
        cands,
        key=lambda r: (
            r["split"] != "holdout" if prefer_holdout else False,
            r["chain"] not in {"A", "B"},
            r["chain_id"],
        ),
    )[0]


def pairs_from_literature(
    path: Path,
    by_uniprot: dict[str, list[dict]],
    by_chain: dict,
) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if (row.get("uniprot_meso") or row.get("pdb_meso") or "").startswith("#"):
                continue
            meso = row.get("pdb_meso") or row.get("meso_chain_id") or ""
            thermo = row.get("pdb_thermo") or row.get("thermo_chain_id") or ""
            if not meso and row.get("uniprot_meso"):
                meso = (_best_chain(by_uniprot.get(row["uniprot_meso"], [])) or {}).get(
                    "chain_id", ""
                )
            if not thermo and row.get("uniprot_thermo"):
                thermo = (_best_chain(by_uniprot.get(row["uniprot_thermo"], [])) or {}).get(
                    "chain_id", ""
                )
            if meso in by_chain and thermo in by_chain:
                out.append(
                    {
                        "meso_chain_id": meso,
                        "thermo_chain_id": thermo,
                        "source": "literature",
                        "note": row.get("note", ""),
                    }
                )
    return out


def _is_thermo_species(species_code: str, species_name: str = "") -> bool:
    code = (species_code or "").upper()
    name = (species_name or "").upper()
    if code.startswith("THE") or code.startswith("THER"):
        return True
    return "THERM" in code or "THERM" in name or "THERMUS" in name


def pairs_from_uniprot_map(
    map_tsv: Path,
    rows: list[dict],
    meso_aliases: dict[str, str],
    thermo_u: set[str],
) -> list[dict]:
    """Pair meso/thermo chains using download_exp_d_structures seed→UniProt map."""
    if not map_tsv.exists():
        return []
    by_uniprot: dict[str, list[dict]] = {}
    for r in rows:
        u = r.get("uniprot", "")
        if u:
            by_uniprot.setdefault(u, []).append(r)

    # unique (seed, thermo_uniprot) with structure support
    combos: dict[tuple[str, str], str] = {}
    with map_tsv.open() as f:
        for rec in csv.DictReader(f, delimiter="\t"):
            roles = rec.get("roles", "")
            if "thermo_ortholog" not in roles:
                continue
            seed = rec.get("seed_uniprot", "")
            tu = rec.get("uniprot", "")
            if not seed or not tu:
                continue
            combos[(seed, tu)] = rec.get("ortholog_species", "")

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for (seed, tu), sp in sorted(combos.items()):
        mu = meso_aliases.get(seed, seed)
        tu_canon = meso_aliases.get(tu, tu)
        if mu == tu or mu == tu_canon or tu in meso_aliases:
            continue
        meso_cands = by_uniprot.get(mu, []) or by_uniprot.get(seed, [])
        thermo_cands = by_uniprot.get(tu, [])
        if not meso_cands or not thermo_cands:
            continue
        meso = sorted(
            meso_cands,
            key=lambda r: (r["split"] != "holdout", r["chain"] not in {"A", "B"}, r["chain_id"]),
        )[0]
        thermo = sorted(
            [t for t in thermo_cands if t.get("uniprot") in thermo_u] or thermo_cands,
            key=lambda r: (r["chain"] != "A", r["chain_id"]),
        )[0]
        key = (meso["chain_id"], thermo["chain_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "meso_chain_id": meso["chain_id"],
                "thermo_chain_id": thermo["chain_id"],
                "source": "structure_map",
                "note": f"seed={seed} thermo={tu} species={sp}",
            }
        )
    return out


def pairs_from_oma(
    oma_csv: Path,
    rows: list[dict],
    thermo_u: set[str],
    meso_aliases: dict[str, str],
) -> list[dict]:
    if not oma_csv.exists():
        return []
    by_uniprot: dict[str, list[dict]] = {}
    for r in rows:
        u = r.get("uniprot", "")
        if u:
            by_uniprot.setdefault(u, []).append(r)

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    with oma_csv.open() as f:
        for rec in csv.DictReader(f):
            sp = (rec.get("ortholog_species") or "").upper()
            sp_name = rec.get("ortholog_species_name") or ""
            if not _is_thermo_species(sp, sp_name):
                continue
            seed = rec.get("seed_uniprot", "")
            corpus_seed = meso_aliases.get(seed, seed)
            oid = rec.get("ortholog_id", "")
            # Resolve OMA locus IDs to UniProt when needed
            try:
                from download_exp_d_structures import normalize_uniprot, resolve_uniprot_id
            except ImportError:
                normalize_uniprot = lambda x: (x or "").strip().split(".")[0].upper()  # noqa: E731
                resolve_uniprot_id = lambda x: None  # noqa: E731
            ortho_u = normalize_uniprot(oid)
            if not ortho_u or len(ortho_u) < 6:
                resolved = resolve_uniprot_id(oid)
                if resolved:
                    ortho_u = resolved
            # ortholog_id may be uniprot-like
            meso_cands = by_uniprot.get(corpus_seed, []) or by_uniprot.get(seed, [])
            thermo_cands = []
            for u in (ortho_u, oid, oid.split(".")[0]):
                if u:
                    thermo_cands.extend(by_uniprot.get(u, []))
            if not meso_cands:
                continue
            # prefer holdout meso, chain A thermo
            meso = sorted(meso_cands, key=lambda r: (r["split"] != "holdout", r["chain_id"]))[0]
            thermo = sorted(
                [t for t in thermo_cands if t.get("uniprot") in thermo_u] or thermo_cands,
                key=lambda r: (r["chain"] != "A", r["chain_id"]),
            )
            if not thermo:
                continue
            thermo = thermo[0]
            key = (meso["chain_id"], thermo["chain_id"])
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "meso_chain_id": meso["chain_id"],
                    "thermo_chain_id": thermo["chain_id"],
                    "source": "oma",
                    "note": f"seed={seed} ortholog_species={sp}",
                }
            )
    return out


def pairs_auto_fold(
    rows: list[dict],
    thermo_u: set[str],
    thermo_pdb: set[str],
    min_len: int,
    max_pairs: int,
) -> list[dict]:
    thermo = [
        r
        for r in rows
        if is_thermo_row(r, thermo_u, thermo_pdb) and r["chain"] == "A" and int(r["length"]) >= min_len
    ]
    meso = [
        r
        for r in rows
        if not is_thermo_row(r, thermo_u, thermo_pdb) and int(r["length"]) >= min_len
    ]
    by_fold_m: dict[str, list[dict]] = {}
    for r in meso:
        by_fold_m.setdefault(r.get("scop_fold", ""), []).append(r)

    # one thermo rep per pdb
    thermo_reps: dict[str, dict] = {}
    for t in thermo:
        thermo_reps.setdefault(t["pdb_id"].lower(), t)

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for t in sorted(thermo_reps.values(), key=lambda r: r["chain_id"]):
        fold = t.get("scop_fold", "")
        mesos = by_fold_m.get(fold, [])
        if not mesos:
            continue
        mesos = sorted(
            mesos,
            key=lambda r: (r["split"] != "holdout", abs(int(r["length"]) - int(t["length"]))),
        )
        m = mesos[0]
        key = (m["chain_id"], t["chain_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "meso_chain_id": m["chain_id"],
                "thermo_chain_id": t["chain_id"],
                "source": "auto_fold",
                "note": f"scop_fold={fold}",
            }
        )
        if len(out) >= max_pairs:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dcfg = load_yaml("exp_d.yaml")
    corpus_cfg = load_yaml("corpus.yaml")
    smoke = bool(args.smoke)

    corpus_dir = resolve_corpus_dir(dcfg, corpus_cfg, smoke)
    rows = load_corpus_chains(corpus_dir / "chains.csv")
    by_chain = {r["chain_id"]: r for r in rows}

    thermo_u = set(dcfg.get("thermo_uniprot") or [])
    thermo_pdb = {p.upper() for p in (dcfg.get("thermo_pdb_prefixes") or [])}
    max_pairs = int(dcfg.get("max_pairs_smoke" if smoke else "max_pairs_paper", 30))
    meso_aliases = {str(k): str(v) for k, v in (dcfg.get("meso_uniprot_aliases") or {}).items()}
    by_uniprot: dict[str, list[dict]] = {}
    for r in rows:
        u = r.get("uniprot", "")
        if u:
            by_uniprot.setdefault(u, []).append(r)

    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add_batch(batch: list[dict]) -> None:
        for p in batch:
            key = (p["meso_chain_id"], p["thermo_chain_id"])
            if key in seen:
                continue
            if key[0] not in by_chain or key[1] not in by_chain:
                continue
            seen.add(key)
            m, t = by_chain[key[0]], by_chain[key[1]]
            p["meso_split"] = m.get("split", "")
            p["thermo_split"] = t.get("split", "")
            p["meso_uniprot"] = m.get("uniprot", "")
            p["thermo_uniprot"] = t.get("uniprot", "")
            p["scop_fold_meso"] = m.get("scop_fold", "")
            p["scop_fold_thermo"] = t.get("scop_fold", "")
            pairs.append(p)

    add_batch(
        pairs_from_literature(
            resolve_path(dcfg["literature_pairs_csv"]),
            by_uniprot,
            by_chain,
        )
    )
    thermo_dir = resolve_path("data/raw/thermo")
    add_batch(
        pairs_from_uniprot_map(
            thermo_dir / "exp_d_uniprot_pdbs.tsv",
            rows,
            meso_aliases,
            thermo_u,
        )
    )
    add_batch(
        pairs_from_oma(
            resolve_path(dcfg["oma_pairs_csv"]),
            rows,
            thermo_u,
            meso_aliases,
        )
    )
    remaining = max(0, max_pairs - len(pairs))
    if remaining:
        add_batch(
            pairs_auto_fold(
                rows,
                thermo_u,
                thermo_pdb,
                int(dcfg.get("min_chain_length", 300)),
                remaining,
            )
        )

    pairs = pairs[:max_pairs]
    out_dir = ensure_dir(resolve_path(dcfg["pairs_out"]).parent)
    out_csv = resolve_path(dcfg["pairs_out"])
    fields = [
        "meso_chain_id",
        "thermo_chain_id",
        "source",
        "note",
        "meso_split",
        "thermo_split",
        "meso_uniprot",
        "thermo_uniprot",
        "scop_fold_meso",
        "scop_fold_thermo",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(pairs)

    write_json(
        out_dir / "pairs_manifest.json",
        {"n_pairs": len(pairs), "sources": {s: sum(1 for p in pairs if p["source"] == s) for s in ("literature", "structure_map", "oma", "auto_fold")}},
    )
    print(f"Wrote {len(pairs)} pairs → {out_csv}")
    return 0 if pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
