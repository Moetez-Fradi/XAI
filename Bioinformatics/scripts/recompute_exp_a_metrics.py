#!/usr/bin/env python3
"""Recompute Exp A metrics (incl. mAP) from checkpoints without re-querying XQdrant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import load_yaml, resolve_path, write_json  # noqa: E402
from exp_a_retrieval import (  # noqa: E402
    finalize_metrics,
    load_jsonl_records,
    write_summary_tsv,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    xcfg = load_yaml("xqdrant.yaml")
    exp_cfg = xcfg.get("exp_a") or {}
    smoke = bool(args.smoke)
    results_dir = resolve_path(
        exp_cfg["smoke_results_dir"] if smoke else exp_cfg["results_dir"]
    )
    jsonl = results_dir / "checkpoints" / "queries.jsonl"
    if not jsonl.exists():
        print(f"ERROR: missing {jsonl}", file=sys.stderr)
        return 1

    run_cfg_path = results_dir / "run_config.json"
    elapsed = 0.0
    dims_top = int(xcfg.get("dims_explained_top", 32))
    coll = xcfg["smoke_collection"] if smoke else xcfg["collection"]
    if run_cfg_path.exists():
        rc = json.loads(run_cfg_path.read_text())
        elapsed = float(rc.get("elapsed_s") or 0.0)
    old_metrics = results_dir / "metrics.json"
    if elapsed == 0.0 and old_metrics.exists():
        elapsed = float(json.loads(old_metrics.read_text()).get("elapsed_s") or 0.0)

    records = load_jsonl_records(jsonl)
    ks = list(exp_cfg.get("limits") or [1, 5, 10])
    metrics = finalize_metrics(
        records,
        ks=ks,
        coll=coll,
        smoke=smoke,
        dims_top=dims_top,
        elapsed_s=elapsed,
        n_boot=args.bootstrap,
        boot_seed=int(exp_cfg.get("bootstrap_seed", 42)),
    )
    metrics["partial"] = False
    write_json(results_dir / "metrics.json", metrics)
    write_summary_tsv(results_dir / "summary.tsv", metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Updated {results_dir}/metrics.json (fold mAP={metrics['fold_map']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
