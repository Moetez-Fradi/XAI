#!/usr/bin/env python3
"""Curate chain ID set for layer/pooling ablation (Exp A query subset + neighbors + train map)."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import ensure_dir, load_yaml, resolve_path, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    acfg = load_yaml("exp_ablation.yaml")
    proc = ensure_dir(resolve_path(acfg["processed_dir"]))
    chains_path = proc / "chain_ids.txt"
    manifest_path = proc / "manifest.json"
    if chains_path.exists() and manifest_path.exists() and not args.force:
        print(f"Exists: {chains_path} ({sum(1 for _ in chains_path.open())} ids)")
        return 0

    exp_a = resolve_path(acfg["exp_a_jsonl"])
    if not exp_a.exists():
        print(f"ERROR: need {exp_a}", file=sys.stderr)
        return 1

    stride = int(acfg.get("query_stride", 10))
    max_n = int(acfg.get("max_neighbors", 10))
    extra_train = int(acfg.get("extra_train_for_map", 800))

    queries: list[str] = []
    neighbors: set[str] = set()
    with exp_a.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i % stride != 0:
                continue
            rec = json.loads(line)
            q = rec.get("query") or rec.get("query_chain_id")
            if not q:
                continue
            queries.append(q)
            for nb in (rec.get("neighbors") or [])[:max_n]:
                cid = nb.get("chain_id")
                if cid:
                    neighbors.add(cid)

    corpus_csv = resolve_path(acfg["corpus_dir"]) / "chains.csv"
    train_pool: list[str] = []
    splits: dict[str, str] = {}
    with corpus_csv.open() as f:
        for rec in csv.DictReader(f):
            splits[rec["chain_id"]] = rec.get("split", "")
            if rec.get("split") == "train":
                train_pool.append(rec["chain_id"])

    ann_dir = resolve_path(acfg["annotations_dir"])
    train_with_prof = [c for c in train_pool if (ann_dir / f"{c}.json").exists()]
    rng = random.Random(int(acfg.get("permutation_seed", 42)))
    rng.shuffle(train_with_prof)
    train_sample = set(train_with_prof[:extra_train])

    all_ids = sorted(set(queries) | neighbors | train_sample)
    chains_path.write_text("\n".join(all_ids) + "\n")
    manifest = {
        "n_queries": len(queries),
        "n_neighbors": len(neighbors),
        "n_train_map": len(train_sample),
        "n_total": len(all_ids),
        "query_stride": stride,
        "queries": queries,
    }
    write_json(manifest_path, manifest)
    print(f"Wrote {chains_path} ({len(all_ids)} chains, {len(queries)} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
