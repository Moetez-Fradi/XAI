#!/usr/bin/env python3
"""Experiment B — do attributed dims track real structural features?

Pipeline:
  1) Fit dim→feature affinity matrix M on TRAIN chains only:
        M[d,f] = corr(embedding[:,d], feature[:,f]) across proteins
  2) For each Exp A (query, neighbor) pair with dims_explained:
        a[d] = |contribution_d| (top attributed dims)
        predicted feature profile p = a @ M
        actual feature agreement g[f] = 1 - |Fq[f]-Fh[f]| / range[f]
        pair_score = pearson(p, g)
  3) Report mean pair_score; null by permuting rows of M (shuffle
     dimension-to-feature mapping); one-sided p-value.

Artifacts under results/exp_b[/ _smoke]/
  run_config.json
  dim_feature_map.npz / dim_feature_map.json
  checkpoints/pairs.jsonl
  metrics.json  (real score, null, p-value, per-feature stats)
  summary.tsv

Resume unless --restart.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import (  # noqa: E402
    BIO_ROOT,
    BatchProgress,
    ensure_dir,
    load_yaml,
    resolve_path,
    write_json,
)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def load_features_table(path: Path, feature_names: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with path.open() as f:
        for rec in csv.DictReader(f):
            try:
                vec = np.array([float(rec[n]) for n in feature_names], dtype=np.float64)
            except (KeyError, ValueError):
                continue
            if not np.isfinite(vec).all():
                continue
            out[rec["chain_id"]] = vec
    return out


def load_embeddings(emb_dir: Path) -> tuple[list[str], np.ndarray, dict[str, int]]:
    ids = [ln.strip() for ln in (emb_dir / "ids.txt").read_text().splitlines() if ln.strip()]
    vec = np.load(emb_dir / "vectors.npy").astype(np.float64)
    return ids, vec, {c: i for i, c in enumerate(ids)}


def load_split(corpus_csv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with corpus_csv.open() as f:
        for rec in csv.DictReader(f):
            out[rec["chain_id"]] = rec.get("split", "")
    return out


def fit_dim_feature_map(
    emb: np.ndarray,
    feats: np.ndarray,
) -> np.ndarray:
    """Return M[dim, feature] Pearson correlations across proteins."""
    n, d = emb.shape
    k = feats.shape[1]
    # z-score columns for stable corr
    def z(x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=0)
        sd = x.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return (x - mu) / sd

    ez = z(emb)
    fz = z(feats)
    # M = (E^T F) / (n-1) for z-scored → correlation
    m = (ez.T @ fz) / max(n - 1, 1)
    return m.astype(np.float64)


def feature_agreement(fq: np.ndarray, fh: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    denom = np.where(ranges < 1e-12, 1.0, ranges)
    return 1.0 - np.clip(np.abs(fq - fh) / denom, 0.0, 1.0)


def pair_score(
    contrib: dict[str, float],
    M: np.ndarray,
    fq: np.ndarray,
    fh: np.ndarray,
    ranges: np.ndarray,
) -> float:
    a = np.zeros(M.shape[0], dtype=np.float64)
    for dim_s, val in contrib.items():
        try:
            d = int(dim_s)
        except ValueError:
            continue
        if 0 <= d < a.size:
            a[d] = abs(float(val))
    if a.sum() < 1e-12:
        return 0.0
    a = a / a.sum()
    pred = a @ M
    actual = feature_agreement(fq, fh, ranges)
    return pearson(pred, actual)


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")
        f.flush()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_exp_a(jsonl: Path, legacy: Path):
    """Yield Exp A query records from jsonl or legacy per_query.json."""
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    data = json.loads(legacy.read_text())
    for rec in data.get("queries") or []:
        yield rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--n-perm", type=int, default=None)
    args = ap.parse_args()

    bcfg = load_yaml("exp_b.yaml")
    corpus_cfg = load_yaml("corpus.yaml")
    smoke = bool(args.smoke)

    results_dir = resolve_path(
        bcfg["smoke_results_dir"] if smoke else bcfg["results_dir"]
    )
    if args.restart and results_dir.exists():
        print(f"--restart: clearing {results_dir}")
        shutil.rmtree(results_dir)
    ensure_dir(results_dir)
    ckpt_dir = ensure_dir(results_dir / "checkpoints")
    pairs_jsonl = ckpt_dir / "pairs.jsonl"

    feature_names = list(bcfg["features"])
    feat_table = resolve_path(bcfg["features_table"])
    if not feat_table.exists():
        print(
            f"ERROR: missing {feat_table}. Run annotate_dssp.py --from-exp-a first.",
            file=sys.stderr,
        )
        return 1

    emb_dir = resolve_path("embeddings/smoke" if smoke else "embeddings/esm2_t33_650M")
    corpus_dir = resolve_path(
        corpus_cfg["smoke_outdir"] if smoke else corpus_cfg["paper_outdir"]
    )
    exp_a_jsonl = resolve_path(
        bcfg["exp_a_smoke_jsonl"] if smoke else bcfg["exp_a_jsonl"]
    )
    exp_a_legacy = exp_a_jsonl.parent.parent / "per_query.json"
    if not exp_a_jsonl.exists() and not exp_a_legacy.exists():
        print(
            f"ERROR: missing Exp A results ({exp_a_jsonl} or {exp_a_legacy})",
            file=sys.stderr,
        )
        return 1

    ids, emb, id_to_row = load_embeddings(emb_dir)
    feats_map = load_features_table(feat_table, feature_names)
    splits = load_split(corpus_dir / "chains.csv")

    # Train set for M
    train_ids = [
        cid
        for cid, sp in splits.items()
        if sp == "train" and cid in feats_map and cid in id_to_row
    ]
    train_ids.sort()
    n_map = int(bcfg.get("n_train_for_map", 3000))
    if smoke:
        n_map = min(n_map, len(train_ids), 80)
    rng = np.random.default_rng(int(bcfg.get("map_random_seed", 42)))
    if len(train_ids) > n_map:
        pick = rng.choice(len(train_ids), size=n_map, replace=False)
        train_ids = [train_ids[i] for i in sorted(pick)]

    if len(train_ids) < 30:
        print(
            f"ERROR: only {len(train_ids)} train chains with features+embeddings; "
            "annotate more with annotate_dssp.py --from-exp-a --also-train N",
            file=sys.stderr,
        )
        return 1

    E = np.stack([emb[id_to_row[c]] for c in train_ids], axis=0)
    F = np.stack([feats_map[c] for c in train_ids], axis=0)
    M = fit_dim_feature_map(E, F)
    ranges = F.max(axis=0) - F.min(axis=0)

    map_path = results_dir / "dim_feature_map.npz"
    np.savez_compressed(
        map_path,
        M=M,
        feature_names=np.array(feature_names),
        train_ids=np.array(train_ids),
        feature_ranges=ranges,
    )
    # top dims per feature for inspection
    top_dims = {}
    for fi, name in enumerate(feature_names):
        order = np.argsort(-np.abs(M[:, fi]))[:20]
        top_dims[name] = [
            {"dim": int(d), "corr": float(M[d, fi])} for d in order
        ]
    write_json(
        results_dir / "dim_feature_map.json",
        {
            "n_train": len(train_ids),
            "dim": int(M.shape[0]),
            "features": feature_names,
            "top_dims_per_feature": top_dims,
        },
    )

    run_config = {
        "mode": "smoke" if smoke else "paper",
        "n_train_map": len(train_ids),
        "features": feature_names,
        "exp_a_jsonl": str(exp_a_jsonl.relative_to(BIO_ROOT)),
        "features_table": str(feat_table.relative_to(BIO_ROOT)),
        "embeddings": str(emb_dir.relative_to(BIO_ROOT)),
        "max_neighbors_per_query": int(bcfg.get("max_neighbors_per_query", 5)),
        "n_permutations": int(args.n_perm or bcfg.get("n_permutations", 1000)),
        "permutation_seed": int(bcfg.get("permutation_seed", 42)),
        "started_unix": int(time.time()),
    }
    write_json(results_dir / "run_config.json", run_config)

    max_neigh = int(bcfg.get("max_neighbors_per_query", 5))
    min_mass = float(bcfg.get("min_attribution_mass", 1e-6))

    # Enumerate planned pairs
    planned: list[tuple[str, str, dict]] = []
    for rec in iter_exp_a(exp_a_jsonl, exp_a_legacy):
        q = rec["query"]
        if q not in feats_map or q not in id_to_row:
            continue
        for n in (rec.get("neighbors") or [])[:max_neigh]:
            hid = n["chain_id"]
            dims = n.get("dims_explained") or {}
            if hid not in feats_map or hid not in id_to_row:
                continue
            if not dims:
                continue
            mass = sum(abs(float(v)) for v in dims.values())
            if mass < min_mass:
                continue
            planned.append((q, hid, dims))
    if args.max_pairs is not None:
        planned = planned[: args.max_pairs]

    done = {(r["query"], r["hit"]) for r in load_jsonl(pairs_jsonl)}
    pending = [(q, h, d) for q, h, d in planned if (q, h) not in done]
    print(
        f"Exp B | train_map={len(train_ids)} planned_pairs={len(planned)} "
        f"pending={len(pending)}"
    )

    t0 = time.time()
    progress = BatchProgress(max(len(pending), 1), label="expB-pairs")
    if not pending:
        progress.tick(skipped=True, force_print=True)

    for i, (q, h, dims) in enumerate(pending):
        fq, fh = feats_map[q], feats_map[h]
        score = pair_score(dims, M, fq, fh, ranges)
        # also raw feature cosine similarity (descriptive)
        cos = float(
            np.dot(fq, fh) / (np.linalg.norm(fq) * np.linalg.norm(fh) + 1e-12)
        )
        append_jsonl(
            pairs_jsonl,
            {
                "query": q,
                "hit": h,
                "pair_score": score,
                "feature_cosine": cos,
                "same_fold": False,  # filled below if available
                "n_dims": len(dims),
                "attribution_mass": sum(abs(float(v)) for v in dims.values()),
                "unix": int(time.time()),
            },
        )
        progress.tick(ok=True, force_print=(i + 1 >= len(pending)))

    # Enrich same_fold from features table / corpus — reload jsonl and rewrite metrics
    # Pull fold from annotation json if present
    ann_dir = resolve_path(bcfg["annotations_dir"])

    def fold_of(cid: str) -> str:
        p = ann_dir / f"{cid}.json"
        if p.exists():
            return json.loads(p.read_text()).get("scop_fold") or ""
        return ""

    records = load_jsonl(pairs_jsonl)
    # Keep only planned pairs in stable order
    wanted = {(q, h) for q, h, _ in planned}
    records = [r for r in records if (r["query"], r["hit"]) in wanted]
    for r in records:
        fq = fold_of(r["query"])
        fh = fold_of(r["hit"])
        r["same_fold"] = bool(fq and fq == fh)
        r["scop_fold_query"] = fq
        r["scop_fold_hit"] = fh

    scores = np.array([float(r["pair_score"]) for r in records], dtype=np.float64)
    real_mean = float(scores.mean()) if scores.size else 0.0
    real_median = float(np.median(scores)) if scores.size else 0.0

    # Permutation null: shuffle rows of M
    n_perm = int(args.n_perm or bcfg.get("n_permutations", 1000))
    seed = int(bcfg.get("permutation_seed", 42))
    prng = np.random.default_rng(seed)
    null_means = np.empty(n_perm, dtype=np.float64)
    # rebuild contrib list once
    pair_payloads = []
    # need dims again from exp_a
    dims_lookup: dict[tuple[str, str], dict] = {}
    for rec in iter_exp_a(exp_a_jsonl, exp_a_legacy):
        q = rec["query"]
        for n in (rec.get("neighbors") or [])[:max_neigh]:
            dims_lookup[(q, n["chain_id"])] = n.get("dims_explained") or {}

    for r in records:
        dims = dims_lookup.get((r["query"], r["hit"]), {})
        pair_payloads.append((dims, feats_map[r["query"]], feats_map[r["hit"]]))

    print(f"Running {n_perm} permutations over {len(pair_payloads)} pairs...")
    for b in range(n_perm):
        Mp = M[prng.permutation(M.shape[0])]
        vals = [
            pair_score(dims, Mp, fq, fh, ranges) for dims, fq, fh in pair_payloads
        ]
        null_means[b] = float(np.mean(vals)) if vals else 0.0
        if (b + 1) % 100 == 0 or b + 1 == n_perm:
            print(f"  [perm] {b+1}/{n_perm}", flush=True)

    p_value = float((np.sum(null_means >= real_mean) + 1) / (n_perm + 1))
    np.save(results_dir / "null_means.npy", null_means)

    # Per-feature: corr between |Δfeature| and attribution-weighted |M[:,f]| mass
    per_feature = {}
    for fi, name in enumerate(feature_names):
        xs, ys = [], []
        for r in records:
            dims = dims_lookup.get((r["query"], r["hit"]), {})
            a = np.zeros(M.shape[0])
            for ds, v in dims.items():
                try:
                    a[int(ds)] = abs(float(v))
                except ValueError:
                    pass
            if a.sum() <= 0:
                continue
            a /= a.sum()
            xs.append(float(a @ np.abs(M[:, fi])))
            ys.append(
                float(
                    1.0
                    - abs(feats_map[r["query"]][fi] - feats_map[r["hit"]][fi])
                    / (ranges[fi] if ranges[fi] > 1e-12 else 1.0)
                )
            )
        per_feature[name] = {
            "pearson_attr_vs_agreement": pearson(np.array(xs), np.array(ys)),
            "n": len(xs),
        }

    same = [r["pair_score"] for r in records if r.get("same_fold")]
    diff = [r["pair_score"] for r in records if not r.get("same_fold")]

    metrics = {
        "n_pairs": len(records),
        "n_train_map": len(train_ids),
        "mean_pair_score": real_mean,
        "median_pair_score": real_median,
        "null_mean": float(null_means.mean()),
        "null_std": float(null_means.std()),
        "null_ci_95": [
            float(np.quantile(null_means, 0.025)),
            float(np.quantile(null_means, 0.975)),
        ],
        "p_value_right_tail": p_value,
        "significant_0_05": p_value < 0.05,
        "mean_score_same_fold": float(np.mean(same)) if same else None,
        "mean_score_diff_fold": float(np.mean(diff)) if diff else None,
        "n_same_fold": len(same),
        "n_diff_fold": len(diff),
        "per_feature": per_feature,
        "n_permutations": n_perm,
        "mode": "smoke" if smoke else "paper",
        "elapsed_s": round(time.time() - t0, 1),
        "completed_unix": int(time.time()),
    }
    write_json(results_dir / "metrics.json", metrics)
    write_json(results_dir / "pairs.json", {"pairs": records})

    with (results_dir / "summary.tsv").open("w") as f:
        f.write("metric\tvalue\n")
        for k in (
            "n_pairs",
            "mean_pair_score",
            "median_pair_score",
            "null_mean",
            "p_value_right_tail",
            "significant_0_05",
            "mean_score_same_fold",
            "mean_score_diff_fold",
        ):
            f.write(f"{k}\t{metrics[k]}\n")
        for name, st in per_feature.items():
            f.write(
                f"per_feature.{name}.pearson\t{st['pearson_attr_vs_agreement']}\n"
            )

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {results_dir.relative_to(BIO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
