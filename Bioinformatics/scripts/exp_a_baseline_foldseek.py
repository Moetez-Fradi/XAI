#!/usr/bin/env python3
"""Exp A baseline: Foldseek structure search (holdout → train)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import ensure_dir, load_yaml, resolve_path, write_json  # noqa: E402
from exp_a_metrics_lib import evaluate_rankings, load_chain_meta  # noqa: E402


def which_or_die(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise FileNotFoundError(f"{name} not on PATH")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    bcfg = load_yaml("exp_a_baselines.yaml")
    fcfg = bcfg.get("foldseek") or {}
    corpus_cfg = load_yaml("corpus.yaml")
    smoke = bool(args.smoke)
    foldseek = which_or_die("foldseek")

    base = resolve_path(
        "data/baselines/exp_a_smoke" if smoke else bcfg["baselines_dir"]
    )
    struct_dir = base / "structures"
    if not struct_dir.exists() or not any(struct_dir.glob("*.pdb")):
        print(
            "ERROR: missing structures. Run prep_exp_a_baselines.py (without --skip-structures).",
            file=sys.stderr,
        )
        return 1

    out_dir = ensure_dir(
        resolve_path(bcfg["smoke_results_dir"] if smoke else bcfg["results_dir"])
        / "foldseek"
    )
    corpus_dir = resolve_path(
        corpus_cfg["smoke_outdir"] if smoke else corpus_cfg["paper_outdir"]
    )
    meta = load_chain_meta(corpus_dir / "chains.csv")

    # Link lists: train vs holdout structure dirs
    train_dir = ensure_dir(base / "structures_train")
    hold_dir = ensure_dir(base / "structures_holdout")
    if args.restart:
        for d in (train_dir, hold_dir):
            for p in d.glob("*.pdb"):
                p.unlink()

    for pdb in struct_dir.glob("*.pdb"):
        cid = pdb.stem
        sp = meta.get(cid, {}).get("split")
        if sp == "train":
            link = train_dir / pdb.name
            if not link.exists():
                link.symlink_to(pdb.resolve())
        elif sp == "holdout":
            link = hold_dir / pdb.name
            if not link.exists():
                link.symlink_to(pdb.resolve())

    n_train = len(list(train_dir.glob("*.pdb")))
    n_hold = len(list(hold_dir.glob("*.pdb")))
    print(f"Foldseek: queries={n_hold} targets={n_train}")
    if n_train == 0 or n_hold == 0:
        print("ERROR: empty train/holdout structure dirs", file=sys.stderr)
        return 1

    db_dir = ensure_dir(base / "foldseek_db")
    target_db = db_dir / "train"
    tmp_dir = ensure_dir(base / "foldseek_tmp")
    hits_path = out_dir / "foldseek_hits.m8"
    rankings_path = out_dir / "rankings.json"
    t0 = time.time()

    db_ready = (db_dir / "train.dbtype").exists() or (db_dir / "train_ss.dbtype").exists()
    if args.restart or not db_ready:
        for p in db_dir.glob("train*"):
            if p.is_file():
                p.unlink()
        print("Creating Foldseek DB...")
        subprocess.run(
            [foldseek, "createdb", str(train_dir), str(target_db)],
            check=True,
        )

    if args.restart or not hits_path.exists():
        print("Running foldseek easy-search...")
        # query can be a directory of PDBs
        cmd = [
            foldseek,
            "easy-search",
            str(hold_dir),
            str(target_db),
            str(hits_path),
            str(tmp_dir),
            "-s",
            str(fcfg.get("sensitivity", 9.5)),
            "--max-seqs",
            str(fcfg.get("max_seqs", 50)),
            "--threads",
            str(fcfg.get("threads", 4)),
            "--format-output",
            "query,target,evalue,bits,alntmscore",
        ]
        subprocess.run(cmd, check=True)
    else:
        print(f"Resume: using {hits_path}")

    by_q: dict[str, list[tuple[float, str]]] = defaultdict(list)
    with hits_path.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            q, s = parts[0], parts[1]
            # foldseek may append chain suffixes; normalize to stem
            q = Path(q).stem.split(".")[0]
            s = Path(s).stem.split(".")[0]
            # score: prefer alntmscore if present else bits
            try:
                score = float(parts[4]) if len(parts) > 4 else float(parts[3])
            except ValueError:
                score = float(parts[3]) if len(parts) > 3 else 0.0
            if q == s:
                continue
            # only train targets
            if meta.get(s, {}).get("split") != "train":
                continue
            by_q[q].append((score, s))

    hold_ids = sorted(
        p.stem for p in hold_dir.glob("*.pdb") if meta.get(p.stem, {}).get("split") == "holdout"
    )
    topk = args.topk
    rankings: dict[str, list[str]] = {}
    for q in hold_ids:
        pairs = sorted(by_q.get(q, []), key=lambda x: -x[0])
        seen = set()
        hits = []
        for _sc, s in pairs:
            if s in seen:
                continue
            seen.add(s)
            hits.append(s)
            if len(hits) >= topk:
                break
        rankings[q] = hits

    write_json(rankings_path, rankings)
    ks = list(bcfg.get("limits") or [1, 5, 10])
    metrics = evaluate_rankings(
        hold_ids,
        rankings,
        meta,
        ks,
        n_boot=int(bcfg.get("bootstrap", 1000)),
        seed=int(bcfg.get("bootstrap_seed", 42)),
    )
    metrics["method"] = "foldseek"
    metrics["mode"] = "smoke" if smoke else "paper"
    metrics["elapsed_s"] = round(time.time() - t0, 1)
    write_json(out_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
