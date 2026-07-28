#!/usr/bin/env python3
"""Prepare FASTA splits + single-chain PDBs for Exp A baselines.

Writes under data/baselines/exp_a/:
  fasta/train.fasta, holdout.fasta
  structures/<chain_id>.pdb   (CA-only; resume skips existing)
  prep_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import (  # noqa: E402
    BIO_ROOT,
    BatchProgress,
    ensure_dir,
    load_yaml,
    raw_dir,
    resolve_path,
    write_json,
)


def find_structure(pdb_dir: Path, pdb_id: str) -> Path | None:
    for name in (
        f"{pdb_id.upper()}.cif.gz",
        f"{pdb_id.upper()}.cif",
        f"{pdb_id}.cif.gz",
        f"{pdb_id}.cif",
        f"{pdb_id.upper()}.pdb",
    ):
        p = pdb_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def write_ca_pdb(src: Path, chain: str, dest: Path) -> int:
    """Write CA-only PDB for one auth chain. Returns n atoms."""
    from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select

    class CASelect(Select):
        def accept_chain(self, ch):
            return ch.id.upper() == chain.upper()

        def accept_atom(self, atom):
            return atom.get_name() == "CA"

    if dest.exists() and dest.stat().st_size > 0:
        return -1  # skipped

    import tempfile

    tmp_path: Path | None = None
    try:
        if src.name.endswith(".cif.gz") or src.name.endswith(".gz"):
            with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            with gzip.open(src, "rb") as fin, tmp_path.open("wb") as fout:
                shutil_copy = __import__("shutil").copyfileobj
                shutil_copy(fin, fout)
            parser = MMCIFParser(QUIET=True)
            struct = parser.get_structure("x", str(tmp_path))
        elif src.suffix == ".cif" or src.name.endswith(".cif"):
            parser = MMCIFParser(QUIET=True)
            struct = parser.get_structure("x", str(src))
        else:
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure("x", str(src))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    io = PDBIO()
    io.set_structure(struct)
    ensure_dir(dest.parent)
    io.save(str(dest), CASelect())
    # count CA lines
    n = sum(1 for ln in dest.read_text().splitlines() if ln.startswith("ATOM"))
    if n == 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"no CA atoms for chain {chain} in {src.name}")
    return n


def write_split_fastas(sequences: Path, chains_csv: Path, out_dir: Path) -> dict:
    meta = {}
    with chains_csv.open() as f:
        for rec in csv.DictReader(f):
            meta[rec["chain_id"]] = rec["split"]

    seqs: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    with sequences.open() as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None:
                    seqs[cur] = "".join(buf)
                cur = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if cur is not None:
            seqs[cur] = "".join(buf)

    train_fa = out_dir / "train.fasta"
    hold_fa = out_dir / "holdout.fasta"
    n_train = n_hold = 0
    with train_fa.open("w") as ft, hold_fa.open("w") as fh:
        for cid, seq in seqs.items():
            sp = meta.get(cid)
            if sp == "train":
                ft.write(f">{cid}\n{seq}\n")
                n_train += 1
            elif sp == "holdout":
                fh.write(f">{cid}\n{seq}\n")
                n_hold += 1
    return {"n_train_fasta": n_train, "n_holdout_fasta": n_hold}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--skip-structures", action="store_true", help="FASTA only (BLAST/ESM2)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    bcfg = load_yaml("exp_a_baselines.yaml")
    corpus_cfg = load_yaml("corpus.yaml")
    smoke = bool(args.smoke)
    base = resolve_path(
        "data/baselines/exp_a_smoke" if smoke else bcfg["baselines_dir"]
    )
    if args.restart and base.exists():
        import shutil

        shutil.rmtree(base)
    fasta_dir = ensure_dir(base / "fasta")
    struct_dir = ensure_dir(base / "structures")
    corpus_dir = resolve_path(
        corpus_cfg["smoke_outdir"] if smoke else corpus_cfg["paper_outdir"]
    )

    stats = write_split_fastas(
        corpus_dir / "sequences.fasta", corpus_dir / "chains.csv", fasta_dir
    )
    print(f"FASTA: train={stats['n_train_fasta']} holdout={stats['n_holdout_fasta']}")

    if args.skip_structures:
        write_json(base / "prep_manifest.json", {**stats, "structures": False})
        print("Skipped structures.")
        return 0

    # structures for train+holdout
    rows = []
    with (corpus_dir / "chains.csv").open() as f:
        for rec in csv.DictReader(f):
            rows.append(rec)
    if smoke:
        # keep holdout + up to 200 train
        hold = [r for r in rows if r["split"] == "holdout"]
        train = [r for r in rows if r["split"] == "train"][:200]
        rows = hold + train
    if args.limit:
        rows = rows[: args.limit]

    pdb_dir = raw_dir("pdb")
    pending = [
        r
        for r in rows
        if args.restart or not (struct_dir / f"{r['chain_id']}.pdb").exists()
    ]
    print(f"Structures: {len(rows)} targets, {len(pending)} pending → {struct_dir}")
    progress = BatchProgress(max(len(pending), 1), label="chain-PDB")
    t0 = time.time()
    ok = fail = skip = 0
    if not pending:
        progress.tick(skipped=True, force_print=True)

    for i, rec in enumerate(pending):
        cid = rec["chain_id"]
        dest = struct_dir / f"{cid}.pdb"
        src = find_structure(pdb_dir, rec["pdb_id"])
        if src is None:
            fail += 1
            progress.tick(ok=False, force_print=(i + 1 >= len(pending)))
            continue
        try:
            n = write_ca_pdb(src, rec["chain"], dest)
            if n < 0:
                skip += 1
                progress.tick(skipped=True, force_print=(i + 1 >= len(pending)))
            else:
                ok += 1
                progress.tick(ok=True, force_print=(i + 1 >= len(pending)))
        except Exception as e:  # noqa: BLE001
            fail += 1
            if fail <= 20:
                print(f"  FAIL {cid}: {e}", file=sys.stderr)
            progress.tick(ok=False, force_print=(i + 1 >= len(pending)))

    n_struct = len(list(struct_dir.glob("*.pdb")))
    manifest = {
        **stats,
        "structures": True,
        "n_structures": n_struct,
        "ok": ok,
        "fail": fail,
        "skip": skip,
        "elapsed_s": round(time.time() - t0, 1),
        "outdir": str(base.relative_to(BIO_ROOT)),
    }
    write_json(base / "prep_manifest.json", manifest)
    print(f"Prep done: {n_struct} PDBs ({manifest['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
