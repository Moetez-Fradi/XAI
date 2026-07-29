#!/usr/bin/env python3
"""Experiment B v2 — per-residue binned profile attribution validity.

Differs from v1 (exp_b_attribution.py):
  - Ground truth: length-normalized binned DSSP profiles (n_bins × 5 features)
  - Metrics: Pearson + Spearman profile scores; Mann–Whitney / AUROC for same-fold
  - Cohorts: filtered by global feature_cosine (v1-style 6 aggregates)
  - Ablations: top-N attributed dimensions
  - Comparison table vs v1 metrics (read-only from results/exp_b/)

Requires annotate_dssp.py --store-profiles (or existing JSON with binned_profile).

Artifacts under results/exp_b_v2[/ _smoke]/
  run_config.json
  dim_profile_map.npz / dim_profile_map.json
  checkpoints/pairs.jsonl
  metrics.json
  summary.tsv
  v1_vs_v2_comparison.tsv

Resume unless --restart. Never writes to results/exp_a/ or results/exp_b/.
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


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if a.size < 2:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r, _ = spearmanr(a, b)
    if r != r:  # NaN
        return 0.0
    return float(r)


def zscore_cols(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd


def fit_dim_feature_map(emb: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """M[dim, feature] Pearson correlations across proteins."""
    n = emb.shape[0]
    return (zscore_cols(emb).T @ zscore_cols(feats)) / max(n - 1, 1)


def profile_agreement(
    pq: np.ndarray, ph: np.ndarray, ranges: np.ndarray
) -> np.ndarray:
    denom = np.where(ranges < 1e-12, 1.0, ranges)
    return 1.0 - np.clip(np.abs(pq - ph) / denom, 0.0, 1.0)


def truncate_contrib(contrib: dict[str, float], top_n: int | None) -> dict[str, float]:
    if top_n is None or top_n <= 0 or len(contrib) <= top_n:
        return contrib
    items = sorted(contrib.items(), key=lambda kv: -abs(float(kv[1])))
    return dict(items[:top_n])


def contrib_vector(
    contrib: dict[str, float], dim: int, top_n: int | None = None
) -> np.ndarray:
    c = truncate_contrib(contrib, top_n)
    a = np.zeros(dim, dtype=np.float64)
    for dim_s, val in c.items():
        try:
            d = int(dim_s)
        except ValueError:
            continue
        if 0 <= d < dim:
            a[d] = abs(float(val))
    s = a.sum()
    if s > 1e-12:
        a /= s
    return a


def pair_scores(
    contrib: dict[str, float],
    M: np.ndarray,
    pq: np.ndarray,
    ph: np.ndarray,
    ranges: np.ndarray,
    top_n: int | None = None,
) -> tuple[float, float]:
    a = contrib_vector(contrib, M.shape[0], top_n)
    if a.sum() < 1e-12:
        return 0.0, 0.0
    pred = a @ M
    actual = profile_agreement(pq, ph, ranges)
    return pearson(pred, actual), spearman(pred, actual)


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


def load_profiles(
    ann_dir: Path,
    chain_ids: set[str],
    n_bins: int,
    n_profile_feat: int,
) -> dict[str, np.ndarray]:
    """Load binned_profile from DSSP JSON; rebuild from residues if present."""
    from annotate_dssp import binned_profile  # noqa: E402

    expected_len = n_bins * n_profile_feat
    out: dict[str, np.ndarray] = {}
    missing: list[str] = []

    for cid in chain_ids:
        p = ann_dir / f"{cid}.json"
        if not p.exists():
            missing.append(cid)
            continue
        data = json.loads(p.read_text())
        bp = data.get("binned_profile")
        if bp and bp.get("values"):
            vec = np.array(bp["values"], dtype=np.float64)
            if vec.size == expected_len:
                out[cid] = vec
                continue
        residues = data.get("residues")
        if residues:
            vec = np.array(binned_profile(residues, {}, n_bins), dtype=np.float64)
            if vec.size == expected_len:
                out[cid] = vec
                continue
        missing.append(cid)

    if missing and len(missing) <= 5:
        for cid in missing:
            print(f"  WARN: no profile for {cid}", file=sys.stderr)
    elif missing:
        print(
            f"  WARN: {len(missing)} chains missing binned profiles "
            f"(run annotate_dssp.py --store-profiles)",
            file=sys.stderr,
        )
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
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    data = json.loads(legacy.read_text())
    for rec in data.get("queries") or []:
        yield rec


def global_feature_cosine(fq: np.ndarray, fh: np.ndarray) -> float:
    return float(np.dot(fq, fh) / (np.linalg.norm(fq) * np.linalg.norm(fh) + 1e-12))


def score_pairs(
    pair_payloads: list[tuple[dict, np.ndarray, np.ndarray]],
    M: np.ndarray,
    ranges: np.ndarray,
    top_n: int | None,
    metric: str,
) -> list[float]:
    out: list[float] = []
    for d, pq, ph in pair_payloads:
        ps, ss = pair_scores(d, M, pq, ph, ranges, top_n)
        out.append(ss if metric == "spearman" else ps)
    return out


def permutation_test(
    pair_payloads: list[tuple[dict, np.ndarray, np.ndarray]],
    M: np.ndarray,
    ranges: np.ndarray,
    n_perm: int,
    seed: int,
    top_n: int | None,
    metric: str,
) -> tuple[float, float, np.ndarray]:
    """Return (real_mean, p_value_right_tail, null_means)."""
    real_vals = score_pairs(pair_payloads, M, ranges, top_n, metric)
    real_mean = float(np.mean(real_vals)) if real_vals else 0.0

    prng = np.random.default_rng(seed)
    null_means = np.empty(n_perm, dtype=np.float64)
    print(f"Running {n_perm} permutations ({metric}) over {len(pair_payloads)} pairs...")
    for b in range(n_perm):
        Mp = M[prng.permutation(M.shape[0])]
        vals = score_pairs(pair_payloads, Mp, ranges, top_n, metric)
        null_means[b] = float(np.mean(vals)) if vals else 0.0
        if (b + 1) % 100 == 0 or b + 1 == n_perm:
            print(f"  [perm/{metric}] {b + 1}/{n_perm}", flush=True)
    p_value = float((np.sum(null_means >= real_mean) + 1) / (n_perm + 1))
    return real_mean, p_value, null_means


def mann_whitney_auroc(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float | None, float | None]:
    from scipy.stats import mannwhitneyu

    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) < 5 or len(neg) < 5:
        return None, None
    try:
        u_stat, p = mannwhitneyu(pos, neg, alternative="greater")
    except ValueError:
        return None, None
    # AUROC from U: AUC = U / (n1 * n2)
    auc = float(u_stat / (len(pos) * len(neg)))
    return auc, float(p)


def cohort_label(threshold: float) -> str:
    if threshold >= 1.0:
        return "all"
    return f"cos_le_{threshold}"


def load_v1_metrics(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def write_comparison_tsv(
    path: Path,
    v1: dict,
    v2_primary: dict,
) -> None:
    rows = [
        ("metric", "v1", "v2"),
        ("n_pairs", v1.get("n_pairs"), v2_primary.get("n_pairs")),
        (
            "mean_pair_score_pearson",
            v1.get("mean_pair_score"),
            v2_primary.get("mean_pearson"),
        ),
        (
            "mean_pair_score_spearman",
            None,
            v2_primary.get("mean_spearman"),
        ),
        (
            "permutation_p_pearson",
            v1.get("p_value_right_tail"),
            v2_primary.get("p_value_pearson"),
        ),
        (
            "permutation_p_spearman",
            None,
            v2_primary.get("p_value_spearman"),
        ),
        (
            "significant_0_05",
            v1.get("significant_0_05"),
            v2_primary.get("significant_0_05_spearman"),
        ),
        (
            "mean_score_same_fold",
            v1.get("mean_score_same_fold"),
            v2_primary.get("mean_spearman_same_fold"),
        ),
        (
            "mean_score_diff_fold",
            v1.get("mean_score_diff_fold"),
            v2_primary.get("mean_spearman_diff_fold"),
        ),
        ("auroc_same_fold", None, v2_primary.get("auroc_same_fold")),
        ("mann_whitney_p", None, v2_primary.get("mann_whitney_p")),
    ]
    with path.open("w") as f:
        for a, b, c in rows:
            f.write(f"{a}\t{b}\t{c}\n")


def analyze_cohort(
    records: list[dict],
    pair_payloads: list[tuple[dict, np.ndarray, np.ndarray]],
    M: np.ndarray,
    ranges: np.ndarray,
    cos_threshold: float,
    n_perm: int,
    seed: int,
    top_n: int | None,
) -> dict:
    idx = [
        i
        for i, r in enumerate(records)
        if float(r["feature_cosine"]) <= cos_threshold + 1e-12
    ]
    sub_recs = [records[i] for i in idx]
    sub_payloads = [pair_payloads[i] for i in idx]
    if not sub_recs:
        return {"n_pairs": 0, "cohort": cohort_label(cos_threshold)}

    pearsons = np.array([float(r["pearson"]) for r in sub_recs])
    spearmans = np.array([float(r["spearman"]) for r in sub_recs])
    labels = np.array([bool(r.get("same_fold")) for r in sub_recs])

    _, p_pearson, null_p = permutation_test(
        sub_payloads, M, ranges, n_perm, seed, top_n, "pearson"
    )
    _, p_spearman, null_s = permutation_test(
        sub_payloads, M, ranges, n_perm, seed + 1, top_n, "spearman"
    )
    auc, mw_p = mann_whitney_auroc(spearmans, labels)

    same = spearmans[labels]
    diff = spearmans[~labels]

    return {
        "cohort": cohort_label(cos_threshold),
        "feature_cosine_max": cos_threshold,
        "n_pairs": len(sub_recs),
        "mean_pearson": float(pearsons.mean()),
        "median_pearson": float(np.median(pearsons)),
        "mean_spearman": float(spearmans.mean()),
        "median_spearman": float(np.median(spearmans)),
        "p_value_pearson": p_pearson,
        "p_value_spearman": p_spearman,
        "null_mean_pearson": float(null_p.mean()),
        "null_mean_spearman": float(null_s.mean()),
        "significant_0_05_pearson": p_pearson < 0.05,
        "significant_0_05_spearman": p_spearman < 0.05,
        "mean_spearman_same_fold": float(same.mean()) if len(same) else None,
        "mean_spearman_diff_fold": float(diff.mean()) if len(diff) else None,
        "n_same_fold": int(labels.sum()),
        "n_diff_fold": int((~labels).sum()),
        "auroc_same_fold": auc,
        "mann_whitney_p": mw_p,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument(
        "--top-dims",
        type=int,
        default=None,
        help="Ablation: keep top-N attributed dims (default: full from Exp A)",
    )
    args = ap.parse_args()

    bcfg = load_yaml("exp_b_v2.yaml")
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

    global_names = list(bcfg["global_features"])
    feat_table = resolve_path(bcfg["features_table"])
    ann_dir = resolve_path(bcfg["annotations_dir"])
    n_bins = int(bcfg.get("n_bins", 32))
    n_profile_feat = len(bcfg.get("profile_features") or [])
    profile_dim = n_bins * n_profile_feat

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
        print(f"ERROR: missing Exp A results ({exp_a_jsonl})", file=sys.stderr)
        return 1

    ids, emb, id_to_row = load_embeddings(emb_dir)
    global_map = load_features_table(feat_table, global_names)
    splits = load_split(corpus_dir / "chains.csv")

    # Chains needed for profiles
    needed: set[str] = set()
    for rec in iter_exp_a(exp_a_jsonl, exp_a_legacy):
        needed.add(rec["query"])
        for n in (rec.get("neighbors") or [])[: int(bcfg.get("max_neighbors_per_query", 5))]:
            needed.add(n["chain_id"])

    profiles_map = load_profiles(ann_dir, needed, n_bins, n_profile_feat)
    if len(profiles_map) < 30:
        print(
            f"ERROR: only {len(profiles_map)} chains with binned profiles. "
            "Run: annotate_dssp.py --from-exp-a --also-train N --store-profiles",
            file=sys.stderr,
        )
        return 1

    train_ids = [
        cid
        for cid, sp in splits.items()
        if sp == "train" and cid in profiles_map and cid in id_to_row
    ]
    train_ids.sort()
    n_map = int(bcfg.get("n_train_for_map", 3000))
    if smoke:
        n_map = min(n_map, len(train_ids), 80)
    rng = np.random.default_rng(int(bcfg.get("map_random_seed", 42)))
    if len(train_ids) > n_map:
        pick = rng.choice(len(train_ids), size=n_map, replace=False)
        train_ids = [train_ids[i] for i in sorted(pick)]

    E = np.stack([emb[id_to_row[c]] for c in train_ids], axis=0)
    F = np.stack([profiles_map[c] for c in train_ids], axis=0)
    M = fit_dim_feature_map(E, F)
    ranges = F.max(axis=0) - F.min(axis=0)

    map_path = results_dir / "dim_profile_map.npz"
    np.savez_compressed(
        map_path,
        M=M,
        n_bins=n_bins,
        profile_features=np.array(bcfg.get("profile_features") or []),
        train_ids=np.array(train_ids),
        feature_ranges=ranges,
    )
    top_dims = {}
    feat_labels = []
    for bi in range(n_bins):
        for fn in bcfg.get("profile_features") or []:
            feat_labels.append(f"bin{bi}.{fn}")
    for fi, name in enumerate(feat_labels[: min(len(feat_labels), M.shape[1])]):
        order = np.argsort(-np.abs(M[:, fi]))[:10]
        top_dims[name] = [{"dim": int(d), "corr": float(M[d, fi])} for d in order]
    write_json(
        results_dir / "dim_profile_map.json",
        {
            "n_train": len(train_ids),
            "dim": int(M.shape[0]),
            "profile_dim": profile_dim,
            "n_bins": n_bins,
            "top_dims_per_profile_feature_sample": dict(list(top_dims.items())[:15]),
            "max_abs_corr": float(np.abs(M).max()),
            "mean_abs_corr": float(np.abs(M).mean()),
        },
    )

    top_n = args.top_dims
    max_neigh = int(bcfg.get("max_neighbors_per_query", 5))
    min_mass = float(bcfg.get("min_attribution_mass", 1e-6))

    planned: list[tuple[str, str, dict]] = []
    for rec in iter_exp_a(exp_a_jsonl, exp_a_legacy):
        q = rec["query"]
        if q not in profiles_map or q not in id_to_row or q not in global_map:
            continue
        for n in (rec.get("neighbors") or [])[:max_neigh]:
            hid = n["chain_id"]
            dims = n.get("dims_explained") or {}
            if hid not in profiles_map or hid not in id_to_row or hid not in global_map:
                continue
            if not dims:
                continue
            if sum(abs(float(v)) for v in dims.values()) < min_mass:
                continue
            planned.append((q, hid, dims))
    if args.max_pairs is not None:
        planned = planned[: args.max_pairs]

    done = {(r["query"], r["hit"]) for r in load_jsonl(pairs_jsonl)}
    pending = [(q, h, d) for q, h, d in planned if (q, h) not in done]
    print(
        f"Exp B v2 | train_map={len(train_ids)} profile_dim={profile_dim} "
        f"planned={len(planned)} pending={len(pending)}"
    )

    t0 = time.time()
    progress = BatchProgress(max(len(pending), 1), label="expB-v2-pairs")
    if not pending:
        progress.tick(skipped=True, force_print=True)

    for i, (q, h, dims) in enumerate(pending):
        pq, ph = profiles_map[q], profiles_map[h]
        gq, gh = global_map[q], global_map[h]
        ps, ss = pair_scores(dims, M, pq, ph, ranges, top_n)
        append_jsonl(
            pairs_jsonl,
            {
                "query": q,
                "hit": h,
                "pearson": ps,
                "spearman": ss,
                "feature_cosine": global_feature_cosine(gq, gh),
                "n_dims": len(truncate_contrib(dims, top_n)),
                "attribution_mass": sum(
                    abs(float(v)) for v in truncate_contrib(dims, top_n).values()
                ),
                "unix": int(time.time()),
            },
        )
        progress.tick(ok=True, force_print=(i + 1 >= len(pending)))

    def fold_of(cid: str) -> str:
        p = ann_dir / f"{cid}.json"
        if p.exists():
            return json.loads(p.read_text()).get("scop_fold") or ""
        return ""

    records = load_jsonl(pairs_jsonl)
    wanted = {(q, h) for q, h, _ in planned}
    records = [r for r in records if (r["query"], r["hit"]) in wanted]
    for r in records:
        fq, fh = fold_of(r["query"]), fold_of(r["hit"])
        r["same_fold"] = bool(fq and fq == fh)
        r["scop_fold_query"] = fq
        r["scop_fold_hit"] = fh

    dims_lookup: dict[tuple[str, str], dict] = {}
    for rec in iter_exp_a(exp_a_jsonl, exp_a_legacy):
        q = rec["query"]
        for n in (rec.get("neighbors") or [])[:max_neigh]:
            dims_lookup[(q, n["chain_id"])] = n.get("dims_explained") or {}

    pair_payloads = [
        (
            dims_lookup.get((r["query"], r["hit"]), {}),
            profiles_map[r["query"]],
            profiles_map[r["hit"]],
        )
        for r in records
    ]

    n_perm = int(args.n_perm or bcfg.get("n_permutations", 1000))
    seed = int(bcfg.get("permutation_seed", 42))
    cohort_thresholds = [float(x) for x in bcfg.get("feature_cosine_cohorts") or [1.0]]

    cohorts = [
        analyze_cohort(
            records, pair_payloads, M, ranges, t, n_perm, seed, top_n
        )
        for t in cohort_thresholds
    ]
    primary = next((c for c in cohorts if c.get("cohort") == "all"), cohorts[0])

    # Negative control: saturated diff-fold
    sat_min = float(bcfg.get("saturated_diff_fold_cosine_min", 0.999))
    sat_recs = [
        r
        for r in records
        if not r.get("same_fold") and float(r["feature_cosine"]) >= sat_min
    ]
    sat_scores = [float(r["spearman"]) for r in sat_recs]
    same_scores = [float(r["spearman"]) for r in records if r.get("same_fold")]

    # Ablations on primary (all) cohort
    ablations = {}
    for n_top in bcfg.get("ablation_top_dims") or []:
        if top_n is not None and n_top != top_n:
            continue
        ab = analyze_cohort(
            records, pair_payloads, M, ranges, 1.0, min(n_perm, 200), seed, int(n_top)
        )
        ablations[f"top_{n_top}"] = {
            "mean_spearman": ab.get("mean_spearman"),
            "p_value_spearman": ab.get("p_value_spearman"),
            "significant_0_05_spearman": ab.get("significant_0_05_spearman"),
        }

    v1_path = resolve_path(bcfg.get("exp_b_v1_metrics", "results/exp_b/metrics.json"))
    v1 = load_v1_metrics(v1_path)

    metrics = {
        "version": "exp_b_v2",
        "mode": "smoke" if smoke else "paper",
        "n_pairs": len(records),
        "n_train_map": len(train_ids),
        "profile_dim": profile_dim,
        "n_bins": n_bins,
        "top_dims_ablation": top_n,
        "primary_cohort": primary,
        "cohorts": cohorts,
        "ablations": ablations,
        "negative_control": {
            "saturated_diff_fold_cosine_min": sat_min,
            "n_saturated_diff_fold": len(sat_recs),
            "mean_spearman_saturated_diff_fold": (
                float(np.mean(sat_scores)) if sat_scores else None
            ),
            "mean_spearman_same_fold": (
                float(np.mean(same_scores)) if same_scores else None
            ),
        },
        "v1_reference": {
            "path": str(v1_path.relative_to(BIO_ROOT)) if v1_path.exists() else None,
            "mean_pair_score": v1.get("mean_pair_score"),
            "p_value_right_tail": v1.get("p_value_right_tail"),
            "significant_0_05": v1.get("significant_0_05"),
        },
        "n_permutations": n_perm,
        "elapsed_s": round(time.time() - t0, 1),
        "completed_unix": int(time.time()),
    }
    write_json(results_dir / "metrics.json", metrics)
    write_json(results_dir / "pairs.json", {"pairs": records})

    with (results_dir / "summary.tsv").open("w") as f:
        f.write("section\tmetric\tvalue\n")
        for k, v in primary.items():
            f.write(f"primary\t{k}\t{v}\n")
        for c in cohorts:
            label = c.get("cohort", "?")
            for k, v in c.items():
                if k != "cohort":
                    f.write(f"cohort.{label}\t{k}\t{v}\n")
        for name, st in ablations.items():
            for k, v in st.items():
                f.write(f"ablation.{name}\t{k}\t{v}\n")
        nc = metrics["negative_control"]
        for k, v in nc.items():
            f.write(f"negative_control\t{k}\t{v}\n")

    write_comparison_tsv(
        results_dir / "v1_vs_v2_comparison.tsv",
        v1,
        primary,
    )

    print(json.dumps({"primary_cohort": primary, "v1": metrics["v1_reference"]}, indent=2))
    print(f"Wrote {results_dir.relative_to(BIO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
