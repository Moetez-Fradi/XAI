#!/usr/bin/env python3
"""Layer + pooling ablation — attribution fold-gap on Exp A subset with alternate ESM2 embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import ensure_dir, load_yaml, resolve_path, write_json  # noqa: E402
from exp_b_attribution import load_embeddings  # noqa: E402
from exp_b_v2_attribution import (  # noqa: E402
    fit_dim_feature_map,
    load_profiles,
    load_split,
    pair_scores,
)
from exp_b_v3_attribution import load_train_profiles_for_map  # noqa: E402


def cosine_dims_explained(q: np.ndarray, h: np.ndarray, top: int = 32) -> dict[str, float]:
    qn = q / (np.linalg.norm(q) + 1e-12)
    hn = h / (np.linalg.norm(h) + 1e-12)
    prod = np.maximum(qn * hn, 0.0)
    s = prod.sum()
    if s < 1e-12:
        return {}
    idx = np.argsort(-prod)[:top]
    return {str(int(i)): float(prod[i] / s) for i in idx if prod[i] > 0}


def load_exp_a_pairs(jsonl: Path, queries: set[str], max_n: int) -> list[tuple[str, str, bool]]:
    pairs: list[tuple[str, str, bool]] = []
    with jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec.get("query") or rec.get("query_chain_id")
            if q not in queries:
                continue
            for nb in (rec.get("neighbors") or [])[:max_n]:
                h = nb.get("chain_id")
                if h:
                    pairs.append((q, h, bool(nb.get("same_fold"))))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="Single embed config name from yaml")
    args = ap.parse_args()

    acfg = load_yaml("exp_ablation.yaml")
    proc = resolve_path(acfg["processed_dir"])
    manifest = json.loads((proc / "manifest.json").read_text())
    query_set = set(manifest["queries"])
    feat_names = list(acfg.get("profile_features") or [])
    n_bins = int(acfg.get("n_bins", 32))
    n_pf = len(feat_names)
    top_dims = int(acfg.get("top_dims", 32))

    exp_a = resolve_path(acfg["exp_a_jsonl"])
    pairs = load_exp_a_pairs(exp_a, query_set, int(acfg.get("max_neighbors", 10)))
    ann_dir = resolve_path(acfg["annotations_dir"])
    corpus_dir = resolve_path(acfg["corpus_dir"])
    splits = load_split(corpus_dir / "chains.csv")
    results_dir = ensure_dir(resolve_path(acfg["results_dir"]))

    need_ids = {q for q, h, _ in pairs} | {h for q, h, _ in pairs}
    profiles = load_profiles(ann_dir, need_ids, n_bins, n_pf)

    configs = acfg.get("embed_configs") or []
    if args.config:
        configs = [c for c in configs if c["name"] == args.config]

    metrics_path = results_dir / "metrics.json"
    all_metrics: dict = {"pairs_planned": len(pairs), "configs": {}}
    if metrics_path.exists():
        all_metrics = json.loads(metrics_path.read_text())

    for cfg in configs:
        name = cfg["name"]
        emb_dir = resolve_path(cfg["outdir"])
        if not (emb_dir / "vectors.npy").exists():
            print(f"SKIP {name}: missing {emb_dir}", file=sys.stderr)
            all_metrics["configs"][name] = {"skipped": True}
            continue

        ids, emb, id_to_row = load_embeddings(emb_dir)
        train_profiles = load_train_profiles_for_map(
            ann_dir, splits, id_to_row, n_bins, n_pf
        )
        if len(train_profiles) < 100:
            print(f"SKIP {name}: train profiles={len(train_profiles)}", file=sys.stderr)
            continue

        train_ids = sorted(train_profiles.keys())
        train_mat = np.stack([emb[id_to_row[c]] for c in train_ids], axis=0)
        prof_mat = np.stack([train_profiles[c] for c in train_ids], axis=0)
        M = fit_dim_feature_map(train_mat, prof_mat)
        ranges = prof_mat.max(axis=0) - prof_mat.min(axis=0)

        scores: list[float] = []
        labels: list[bool] = []
        for q, h, same_fold in pairs:
            if q not in id_to_row or h not in id_to_row:
                continue
            pq, ph = profiles.get(q), profiles.get(h)
            if pq is None or ph is None:
                continue
            contrib = cosine_dims_explained(emb[id_to_row[q]], emb[id_to_row[h]], top_dims)
            if not contrib:
                continue
            _pearson, sp = pair_scores(contrib, M, pq, ph, ranges, top_dims)
            scores.append(sp)
            labels.append(same_fold)

        if len(scores) < 50:
            print(f"SKIP {name}: n_pairs={len(scores)}", file=sys.stderr)
            continue

        arr = np.array(scores, dtype=np.float64)
        lab = np.array(labels, dtype=bool)
        same = arr[lab]
        diff = arr[~lab]
        try:
            _, mw_p = mannwhitneyu(same, diff, alternative="greater")
        except ValueError:
            mw_p = float("nan")

        rec = {
            "n_pairs": int(len(scores)),
            "mean_spearman": float(arr.mean()),
            "mean_spearman_same_fold": float(same.mean()) if len(same) else None,
            "mean_spearman_diff_fold": float(diff.mean()) if len(diff) else None,
            "fold_gap": float(same.mean() - diff.mean()) if len(same) and len(diff) else None,
            "mann_whitney_p": float(mw_p),
            "layer": cfg.get("layer"),
            "pooling": cfg.get("pooling"),
        }
        all_metrics["configs"][name] = rec
        print(f"{name}: gap={rec['fold_gap']:.4f} MW p={rec['mann_whitney_p']:.4g} n={rec['n_pairs']}")

    write_json(metrics_path, all_metrics)
    with (results_dir / "ablation_comparison.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            ["config", "layer", "pooling", "n_pairs", "fold_gap", "mann_whitney_p", "mean_spearman"]
        )
        for name, rec in sorted(all_metrics.get("configs", {}).items()):
            if rec.get("skipped"):
                continue
            w.writerow(
                [
                    name,
                    rec.get("layer"),
                    rec.get("pooling"),
                    rec.get("n_pairs"),
                    f"{rec.get('fold_gap', 0):.4f}",
                    f"{rec.get('mann_whitney_p', 0):.4g}",
                    f"{rec.get('mean_spearman', 0):.4f}",
                ]
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
