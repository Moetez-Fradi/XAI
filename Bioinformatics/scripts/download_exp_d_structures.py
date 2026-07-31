#!/usr/bin/env python3
"""Download PDB structures for Experiment D thermo/meso ortholog pairs.

Resolves UniProt accessions (OMA seeds + thermophile orthologs + meso aliases)
to PDB entries via SIFTS and UniProt REST, then downloads mmCIF files into
data/raw/pdb/ (shared with the main corpus).

Writes:
  data/raw/thermo/exp_d_uniprot_pdbs.tsv   # uniprot → pdb, chain, role
  data/raw/thermo/exp_d_pdb_ids.txt        # unique PDB IDs to fetch
  data/raw/thermo/exp_d_structure_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import (  # noqa: E402
    BIO_ROOT,
    BatchProgress,
    download_cfg,
    download_file,
    ensure_dir,
    gunzip_to,
    load_yaml,
    raw_dir,
    resolve_path,
    write_json,
)

UNIPROT_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]|"
    r"A0A[A-Z0-9]{2}[0-9]{6}(?:-\d+)?|[A-Z0-9]{6,10})(?:\.\d+)?$",
    re.I,
)


def _http_json(url: str, *, timeout: int = 60) -> dict | list:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "XQdrant-Bio/0.1"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def looks_like_uniprot(acc: str) -> bool:
    acc = (acc or "").strip().split(".")[0]
    return bool(acc and UNIPROT_RE.match(acc))


def normalize_uniprot(raw: str) -> str:
    return (raw or "").strip().split(".")[0].upper()


def ensure_sifts(sifts_dir: Path, *, force: bool = False) -> Path:
    """Ensure pdb_chain_uniprot.csv is available (download + decompress if needed)."""
    ensure_dir(sifts_dir)
    csv_path = sifts_dir / "pdb_chain_uniprot.csv"
    gz_path = sifts_dir / "pdb_chain_uniprot.csv.gz"
    if csv_path.exists() and csv_path.stat().st_size > 0 and not force:
        return csv_path
    url = (
        download_cfg()
        .get("uniprot", {})
        .get("sifts", [{}])[0]
        .get("url", "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz")
    )
    print(f"Downloading SIFTS → {gz_path.name}")
    download_file(url, gz_path, force=force, timeout=300, max_retries=4)
    gunzip_to(gz_path, csv_path)
    return csv_path


def load_sifts_reverse(sifts_csv: Path) -> dict[str, list[dict]]:
    """Map UniProt accession → list of {pdb_id, chain}."""
    rev: dict[str, list[dict]] = defaultdict(list)
    opener = gzip.open if sifts_csv.suffix == ".gz" else open
    with opener(sifts_csv, "rt", newline="") as f:  # type: ignore[arg-type]
        rows = csv.reader(f)
        header = None
        for row in rows:
            if not row or row[0].startswith("#"):
                continue
            if header is None:
                header = [c.strip() for c in row]
                continue
            rec = dict(zip(header, row))
            pdb = (rec.get("PDB") or "").lower()
            chain = (rec.get("CHAIN") or "").strip().upper()
            acc = normalize_uniprot(rec.get("SP_PRIMARY") or "")
            if pdb and chain and acc:
                rev[acc].append({"pdb_id": pdb, "chain": chain, "source": "sifts"})
    return rev


def uniprot_rest_pdbs(acc: str, *, sleep: float = 0.15) -> list[dict]:
    """PDB cross-refs from UniProt REST (fallback when SIFTS misses)."""
    acc = normalize_uniprot(acc)
    if not acc:
        return []
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
    try:
        data = _http_json(url)
    except HTTPError as e:
        if e.code == 404:
            return []
        raise
    finally:
        if sleep:
            time.sleep(sleep)
    out: list[dict] = []
    for ref in data.get("uniProtKBCrossReferences") or data.get("dbReferences") or []:
        if (ref.get("database") or ref.get("type")) != "PDB":
            continue
        pdb_id = (ref.get("id") or "").lower()
        if not pdb_id:
            continue
        chains: list[str] = []
        for prop in ref.get("properties") or []:
            key = prop.get("key") or prop.get("type") or ""
            val = prop.get("value") or ""
            if key.lower() in {"chains", "chain"} and val:
                for part in str(val).replace(" ", "").split("/"):
                    if part and part != "-":
                        chains.append(part.split("-")[0].upper())
        if not chains:
            chains = ["A"]
        for ch in chains:
            out.append({"pdb_id": pdb_id, "chain": ch, "source": "uniprot_rest"})
    return out


def resolve_uniprot_id(raw: str, *, sleep: float = 0.15) -> str | None:
    """Map OMA / locus-like IDs to a UniProt accession when possible."""
    raw = (raw or "").strip()
    if not raw:
        return None
    if looks_like_uniprot(raw):
        return normalize_uniprot(raw)
    # UniProt search: try accession / gene / raw id
    queries = [
        f"accession:{raw}",
        f"gene:{raw}",
        raw,
    ]
    for q in queries:
        url = f"https://rest.uniprot.org/uniprotkb/search?query={quote(q)}&format=json&size=1"
        try:
            data = _http_json(url)
        except HTTPError as e:
            if e.code in {400, 404}:
                continue
            return None
        except URLError:
            return None
        finally:
            if sleep:
                time.sleep(sleep)
        results = data.get("results") if isinstance(data, dict) else None
        if results:
            primary = results[0].get("primaryAccession") or results[0].get("uniProtkbId")
            if primary:
                return normalize_uniprot(primary)
    return None


def load_corpus_uniprots(corpus_csv: Path) -> set[str]:
    if not corpus_csv.exists():
        return set()
    out: set[str] = set()
    with corpus_csv.open() as f:
        for row in csv.DictReader(f):
            u = normalize_uniprot(row.get("uniprot", ""))
            if u:
                out.add(u)
    return out


def load_corpus_pdbs(corpus_csv: Path) -> set[str]:
    if not corpus_csv.exists():
        return set()
    return {row["pdb_id"].lower() for row in csv.DictReader(corpus_csv.open())}


def collect_target_accessions(
    oma_csv: Path,
    seeds: list[str],
    meso_aliases: dict[str, str],
    *,
    include_all_thermo: bool,
) -> dict[str, dict]:
    """Return acc → {role, seed, species, oma_id}."""
    targets: dict[str, dict] = {}

    def add(acc_raw: str, meta: dict) -> None:
        acc = normalize_uniprot(acc_raw)
        if not acc:
            acc = (acc_raw or "").strip().upper()
        if not acc:
            return
        if acc not in targets:
            roles = set(meta.get("roles") or [])
            targets[acc] = {**meta, "roles": roles, "raw_id": acc_raw}
        else:
            roles = targets[acc].get("roles") or set()
            if not isinstance(roles, set):
                roles = set(roles)
            roles.update(meta.get("roles") or [])
            targets[acc]["roles"] = roles
            targets[acc].setdefault("raw_id", acc_raw)

    for seed in seeds:
        corpus_seed = meso_aliases.get(seed, seed)
        add(corpus_seed, {"roles": {"meso_seed"}, "seed_uniprot": seed, "species": "ECOLI"})
        add(seed, {"roles": {"meso_seed"}, "seed_uniprot": seed, "species": "ECOLI"})

    if oma_csv.exists():
        with oma_csv.open() as f:
            for row in csv.DictReader(f):
                seed = row.get("seed_uniprot", "")
                oid = row.get("ortholog_id", "")
                sp = row.get("ortholog_species", "")
                sp_name = row.get("ortholog_species_name", "")
                add(
                    oid,
                    {
                        "roles": {"thermo_ortholog"},
                        "seed_uniprot": seed,
                        "ortholog_species": sp,
                        "ortholog_species_name": sp_name,
                        "oma_id": oid,
                    },
                )
                if include_all_thermo:
                    corpus_seed = meso_aliases.get(seed, seed)
                    add(corpus_seed, {"roles": {"meso_seed"}, "seed_uniprot": seed})

    return targets


def rank_pdb_entries(
    entries: list[dict],
    *,
    max_n: int,
    corpus_pdbs: set[str],
    existing_files: set[str],
) -> list[dict]:
    """Prefer PDBs we don't already have on disk; dedupe by pdb_id."""

    def score(e: dict) -> tuple:
        pid = e["pdb_id"]
        have_file = pid in existing_files
        in_corpus = pid in corpus_pdbs
        return (have_file, in_corpus, pid)

    seen: set[str] = set()
    ranked: list[dict] = []
    for e in sorted(entries, key=score):
        pid = e["pdb_id"]
        if pid in seen:
            continue
        seen.add(pid)
        ranked.append(e)
        if len(ranked) >= max_n:
            break
    return ranked


def pdb_file_exists(pdb_dir: Path, pdb_id: str) -> bool:
    pid = pdb_id.upper()
    for name in (f"{pid}.cif.gz", f"{pid}.cif", f"{pid}.pdb"):
        p = pdb_dir / name
        if p.exists() and p.stat().st_size > 0:
            return True
    return False


def download_pdbs(
    pdb_ids: list[str],
    pdb_dir: Path,
    *,
    force: bool,
    sleep_s: float,
    retries: int,
) -> tuple[list[str], list[dict]]:
    cfg = download_cfg()["pdb"]
    fmt = cfg.get("format", "cif.gz")
    base = cfg["base_url"].rstrip("/")
    ok: list[str] = []
    fail: list[dict] = []
    progress = BatchProgress(len(pdb_ids), label="expD-PDB")

    for i, pdb_id in enumerate(pdb_ids, 1):
        pid = pdb_id.upper()
        if fmt == "cif.gz":
            fname, url = f"{pid}.cif.gz", f"{base}/{pid}.cif.gz"
        elif fmt == "cif":
            fname, url = f"{pid}.cif", f"{base}/{pid}.cif"
        else:
            fname, url = f"{pid}.pdb", f"{base}/{pid}.pdb"
        dest = pdb_dir / fname
        existed = dest.exists() and dest.stat().st_size > 0 and not force
        try:
            before = dest.stat().st_size if dest.exists() else 0
            download_file(url, dest, force=force, max_retries=retries, timeout=180, quiet=True)
            nbytes = dest.stat().st_size if dest.exists() else 0
            if existed:
                progress.tick(skipped=True, force_print=(i == len(pdb_ids)))
            else:
                progress.tick(ok=True, nbytes=max(nbytes - before, nbytes), force_print=(i == len(pdb_ids)))
            ok.append(pdb_id.lower())
        except Exception as e:  # noqa: BLE001
            if fmt == "cif.gz":
                try:
                    dest2 = pdb_dir / f"{pid}.cif"
                    download_file(
                        f"{base}/{pid}.cif",
                        dest2,
                        force=force,
                        max_retries=retries,
                        timeout=180,
                        quiet=True,
                    )
                    ok.append(pdb_id.lower())
                    progress.tick(ok=True, force_print=(i == len(pdb_ids)))
                    if sleep_s and i < len(pdb_ids):
                        time.sleep(sleep_s)
                    continue
                except Exception as e2:  # noqa: BLE001
                    e = e2
            fail.append({"pdb_id": pdb_id, "error": str(e)})
            progress.tick(ok=False, force_print=(i == len(pdb_ids)))
        if sleep_s and i < len(pdb_ids):
            time.sleep(sleep_s)
    return ok, fail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="Re-download SIFTS / PDB files")
    ap.add_argument("--skip-download", action="store_true", help="Resolve IDs only (no PDB fetch)")
    ap.add_argument("--max-pdbs-per-uniprot", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    dcfg = load_yaml("exp_d.yaml")
    scfg = dcfg.get("structure_download") or {}
    corpus_cfg = load_yaml("corpus.yaml")
    thermo_cfg = download_cfg()["thermo"]

    max_per_u = int(
        args.max_pdbs_per_uniprot
        if args.max_pdbs_per_uniprot is not None
        else scfg.get("max_pdbs_per_uniprot", 3)
    )

    sifts_dir = raw_dir("sifts")
    sifts_csv = ensure_sifts(sifts_dir, force=args.force)
    print(f"SIFTS: {sifts_csv}")
    sifts_rev = load_sifts_reverse(sifts_csv)
    print(f"  reverse index: {len(sifts_rev)} UniProt accessions")

    corpus_csv = resolve_path(corpus_cfg["paper_outdir"]) / "chains.csv"
    corpus_uniprots = load_corpus_uniprots(corpus_csv)
    corpus_pdbs = load_corpus_pdbs(corpus_csv)
    print(f"Corpus: {len(corpus_uniprots)} uniprots, {len(corpus_pdbs)} PDB IDs")

    oma_csv = resolve_path(dcfg["oma_pairs_csv"])
    seeds = list(thermo_cfg.get("seed_uniprot") or [])
    aliases = {str(k): str(v) for k, v in (dcfg.get("meso_uniprot_aliases") or {}).items()}

    targets = collect_target_accessions(
        oma_csv,
        seeds,
        aliases,
        include_all_thermo=bool(scfg.get("include_all_oma_thermophiles", True)),
    )
    print(f"Target accessions: {len(targets)}")

    pdb_dir = raw_dir("pdb")
    ensure_dir(pdb_dir)
    existing_files = {p.stem.split(".")[0].lower() for p in pdb_dir.glob("*") if p.is_file()}

    mapping_rows: list[dict] = []
    pdb_ids_needed: set[str] = set()
    stats = {"with_sifts": 0, "with_rest": 0, "no_structure": 0, "already_in_corpus": 0}

    for acc, meta in sorted(targets.items()):
        raw_id = meta.get("raw_id", "")
        lookup_acc = acc
        entries: list[dict] = []
        for key in dict.fromkeys([acc, str(raw_id).upper() if raw_id else ""]):
            if key:
                entries.extend(sifts_rev.get(key, []))
        if entries:
            stats["with_sifts"] += 1
        else:
            if not looks_like_uniprot(acc):
                resolved = resolve_uniprot_id(str(raw_id or acc), sleep=args.sleep)
                if resolved:
                    lookup_acc = resolved
                    entries = list(sifts_rev.get(resolved, []))
                    if entries:
                        stats["with_sifts"] += 1
            if not entries and looks_like_uniprot(lookup_acc):
                entries = uniprot_rest_pdbs(lookup_acc, sleep=args.sleep)
                if entries:
                    stats["with_rest"] += 1
        acc = lookup_acc
        if not entries:
            stats["no_structure"] += 1
            mapping_rows.append(
                {
                    "uniprot": acc,
                    "pdb_id": "",
                    "chain": "",
                    "source": "",
                    "roles": ",".join(sorted(meta.get("roles", []))),
                    "seed_uniprot": meta.get("seed_uniprot", ""),
                    "ortholog_species": meta.get("ortholog_species", ""),
                    "in_corpus_uniprot": acc in corpus_uniprots,
                }
            )
            continue

        picked = rank_pdb_entries(
            entries,
            max_n=max_per_u,
            corpus_pdbs=corpus_pdbs,
            existing_files=existing_files,
        )
        if acc in corpus_uniprots:
            stats["already_in_corpus"] += 1

        for e in picked:
            pdb_ids_needed.add(e["pdb_id"])
            mapping_rows.append(
                {
                    "uniprot": acc,
                    "pdb_id": e["pdb_id"],
                    "chain": e["chain"],
                    "source": e["source"],
                    "roles": ",".join(sorted(meta.get("roles", []))),
                    "seed_uniprot": meta.get("seed_uniprot", ""),
                    "ortholog_species": meta.get("ortholog_species", ""),
                    "in_corpus_uniprot": acc in corpus_uniprots,
                }
            )

    thermo_dir = raw_dir("thermo")
    map_tsv = thermo_dir / "exp_d_uniprot_pdbs.tsv"
    ids_txt = thermo_dir / "exp_d_pdb_ids.txt"
    pdb_list = sorted(pdb_ids_needed)

    with map_tsv.open("w", newline="") as f:
        fields = [
            "uniprot",
            "pdb_id",
            "chain",
            "source",
            "roles",
            "seed_uniprot",
            "ortholog_species",
            "in_corpus_uniprot",
        ]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(mapping_rows)

    ids_txt.write_text("\n".join(pdb_list) + ("\n" if pdb_list else ""))
    print(f"Resolved {len(pdb_list)} unique PDB IDs → {ids_txt}")
    print(
        f"  sifts={stats['with_sifts']} rest={stats['with_rest']} "
        f"no_structure={stats['no_structure']} acc_in_corpus={stats['already_in_corpus']}"
    )

    ok, fail = [], []
    if not args.skip_download and pdb_list:
        print(f"Downloading {len(pdb_list)} structures → {pdb_dir}")
        ok, fail = download_pdbs(
            pdb_list,
            pdb_dir,
            force=args.force,
            sleep_s=float(scfg.get("sleep_s", 0.03)),
            retries=int(download_cfg()["pdb"].get("max_retries", 4)),
        )

    manifest = {
        "n_target_uniprots": len(targets),
        "n_mapping_rows": len(mapping_rows),
        "n_pdb_ids": len(pdb_list),
        "n_downloaded_ok": len(ok),
        "n_download_failed": len(fail),
        "stats": stats,
        "max_pdbs_per_uniprot": max_per_u,
        "mapping_tsv": str(map_tsv.relative_to(BIO_ROOT)),
        "pdb_ids_file": str(ids_txt.relative_to(BIO_ROOT)),
        "failures": fail[:30],
    }
    write_json(thermo_dir / "exp_d_structure_manifest.json", manifest)
    print(f"Manifest → {thermo_dir / 'exp_d_structure_manifest.json'}")
    if fail:
        print(f"WARN: {len(fail)} PDB downloads failed", file=sys.stderr)
    return 0 if not fail or args.skip_download else 0


if __name__ == "__main__":
    raise SystemExit(main())
