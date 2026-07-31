#!/usr/bin/env python3
"""Experiment C — negative controls and specificity (ROC / PR curves).

Tests whether attribution-based scores discriminate structurally related pairs
(same SCOP fold) from unrelated pairs, compared to baselines:

  - xq_profile_spearman: attribution-weighted DSSP profile score (from Exp B)
  - retrieval_score: ESM2 cosine from Exp A
  - random_dims: shuffle attributed dimension indices (null)
  - uniform_dims: equal weight on attributed dims

Pair sources:
  - retrieval: Exp A query–neighbor pairs
  - random_negative: holdout query + random train chain (different fold)

Also tags saturated_diff_fold negatives (high global DSSP cosine, diff fold).

Artifacts under results/exp_c[/ _smoke]/:
  checkpoints/pairs.jsonl
  metrics.json
  summary.tsv
  roc_curves.tsv
  pr_curves.tsv

Resume unless --restart. Does not overwrite Exp A/B results.
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
from exp_b_v2_attribution import (  # noqa: E402
    append_jsonl,
    contrib_vector,
    fit_dim_feature_map,
    global_feature_cosine,
    iter_exp_a,
    load_embeddings,
    load_features_table,
    load_jsonl,
    load_profiles,
    load_split,
    pair_scores,
    truncate_contrib,
)
from exp_b_v3_attribution import load_train_profiles_for_map  # noqa: E402


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Higher score ⇒ positive label."""
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = labels[order]
    tpr = np.cumsum(y) / n_pos
    fpr = np.cumsum(~y) / n_neg
    return _trapz(tpr, fpr)


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = labels[order]
    prec = np.cumsum(y) / (np.arange(len(y)) + 1)
    rec = np.cumsum(y) / n_pos
    return _trapz(prec, rec)


def roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(bool)
    n_pos = max(int(labels.sum()), 1)
    n_neg = max(int((~labels).sum()), 1)
    order = np.argsort(-scores)
    y = labels[order]
    tpr = np.cumsum(y) / n_pos
    fpr = np.cumsum(~y) / n_neg
    return fpr, tpr


def pr_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(bool)
    n_pos = max(int(labels.sum()), 1)
    order = np.argsort(-scores)
    y = labels[order]
    prec = np.cumsum(y) / (np.arange(len(y)) + 1)
    rec = np.cumsum(y) / n_pos
    return rec, prec


def attribution_entropy(contrib: dict[str, float], dim: int, top_n: int | None = 32) -> float:
    a = contrib_vector(contrib, dim, top_n)
    pos = a[a > 1e-12]
    if pos.size == 0:
        return 0.0
    return float(-(pos * np.log(pos)).sum())


def attribution_top1_mass(contrib: dict[str, float], dim: int, top_n: int | None = 32) -> float:
    a = contrib_vector(contrib, dim, top_n)
    return float(a.max())


def shuffle_dims_contrib(contrib: dict[str, float], prng: np.random.Generator) -> dict[str, float]:
    items = list(contrib.items())
    if len(items) < 2:
        return dict(contrib)
    vals = [v for _, v in items]
    prng.shuffle(vals)
    return {k: vals[i] for i, (k, _) in enumerate(items)}


def uniform_dims_contrib(contrib: dict[str, float], top_n: int | None = 32) -> dict[str, float]:
    c = truncate_contrib(contrib, top_n)
    if not c:
        return {}
    v = 1.0 / len(c)
    return {k: v for k in c}


def load_or_fit_map(
    smoke: bool,
    bcfg: dict,
    ccfg: dict,
    emb_dir: Path,
    ann_dir: Path,
    splits: dict[str, str],
    id_to_row: dict[str, int],
    n_bins: int,
    n_profile_feat: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    for key in ("exp_b_v3_map", "exp_b_v2_map"):
        rel = ccfg.get(key)
        if not rel:
            continue
        p = resolve_path(rel)
        if p.exists():
            z = np.load(p)
            train_ids = [str(x) for x in z.get("train_ids", [])]
            return z["M"], z["feature_ranges"], train_ids

    map_profiles = load_train_profiles_for_map(
        ann_dir, splits, id_to_row, n_bins, n_profile_feat
    )
    train_ids = sorted(map_profiles.keys())
    n_map_cfg = int(ccfg.get("n_train_for_map", 0))
    if smoke:
        n_map_cfg = min(n_map_cfg or 80, len(train_ids), 80)
    if n_map_cfg > 0 and len(train_ids) > n_map_cfg:
        rng = np.random.default_rng(int(ccfg.get("map_random_seed", 42)))
        pick = rng.choice(len(train_ids), size=n_map_cfg, replace=False)
        train_ids = [train_ids[i] for i in sorted(pick)]

    ids, emb, _ = load_embeddings(emb_dir)
    e = np.stack([emb[id_to_row[c]] for c in train_ids])
    f = np.stack([map_profiles[c] for c in train_ids])
    m = fit_dim_feature_map(e, f)
    ranges = f.max(axis=0) - f.min(axis=0)
    return m, ranges, train_ids


def load_chain_meta(corpus_csv: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with corpus_csv.open() as f:
        for rec in csv.DictReader(f):
            out[rec["chain_id"]] = rec
    return out


def sample_random_negatives(
    queries: list[str],
    query_folds: dict[str, str],
    train_by_fold: dict[str, list[str]],
    all_train: list[str],
    n_per_query: int,
    prng: np.random.Generator,
) -> list[tuple[str, str]]:
    """Return (query, hit) with different SCOP fold."""
    pairs: list[tuple[str, str]] = []
    folds = list(train_by_fold.keys())
    for q in queries:
        fq = query_folds.get(q, "")
        if not fq:
            continue
        for _ in range(n_per_query):
            for _attempt in range(50):
                cand_fold = folds[int(prng.integers(0, len(folds)))]
                if cand_fold == fq:
                    continue
                pool = train_by_fold.get(cand_fold) or []
                if not pool:
                    continue
                h = pool[int(prng.integers(0, len(pool)))]
                pairs.append((q, h))
                break
            else:
                # fallback: any train chain different fold
                for _ in range(50):
                    h = all_train[int(prng.integers(0, len(all_train)))]
                    if query_folds.get(h, "") != fq:
                        pairs.append((q, h))
                        break
    return pairs


def score_pair_record(
    contrib: dict[str, float] | None,
    retrieval_score: float | None,
    q: str,
    h: str,
    profiles: dict[str, np.ndarray],
    global_map: dict[str, np.ndarray],
    m: np.ndarray,
    ranges: np.ndarray,
    dim: int,
    prng: np.random.Generator,
) -> dict:
    out: dict = {"retrieval_score": retrieval_score}
    if not contrib or q not in profiles or h not in profiles:
        out["xq_profile_spearman"] = None
        out["random_dims_spearman"] = None
        out["uniform_dims_spearman"] = None
        out["attribution_entropy"] = None
        out["attribution_top1_mass"] = None
        return out

    pq, ph = profiles[q], profiles[h]
    _, ss = pair_scores(contrib, m, pq, ph, ranges)
    out["xq_profile_spearman"] = ss

    rc = shuffle_dims_contrib(contrib, prng)
    _, ss_r = pair_scores(rc, m, pq, ph, ranges)
    out["random_dims_spearman"] = ss_r

    uc = uniform_dims_contrib(contrib)
    _, ss_u = pair_scores(uc, m, pq, ph, ranges)
    out["uniform_dims_spearman"] = ss_u

    out["attribution_entropy"] = attribution_entropy(contrib, dim)
    out["attribution_top1_mass"] = attribution_top1_mass(contrib, dim)
    if q in global_map and h in global_map:
        out["global_feature_cosine"] = global_feature_cosine(global_map[q], global_map[h])
    return out


def evaluate_method(
    records: list[dict],
    score_key: str,
    label_key: str = "same_fold",
    subset: str | None = None,
) -> dict:
    rows = records
    if subset:
        rows = [r for r in rows if r.get("pair_source") == subset]
    scores = []
    labels = []
    for r in rows:
        v = r.get(score_key)
        if v is None or v != v:
            continue
        scores.append(float(v))
        labels.append(bool(r.get(label_key)))
    if len(scores) < 10:
        return {"n": len(scores), "auroc": None, "auprc": None}
    s = np.array(scores)
    y = np.array(labels)
    return {
        "n": len(scores),
        "n_positive": int(y.sum()),
        "n_negative": int((~y).sum()),
        "auroc": auroc(s, y),
        "auprc": auprc(s, y),
        "mean_score_positive": float(s[y].mean()) if y.any() else None,
        "mean_score_negative": float(s[~y].mean()) if (~y).any() else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--max-queries", type=int, default=None)
    args = ap.parse_args()

    ccfg = load_yaml("exp_c.yaml")
    corpus_cfg = load_yaml("corpus.yaml")
    smoke = bool(args.smoke)

    results_dir = resolve_path(
        ccfg["smoke_results_dir"] if smoke else ccfg["results_dir"]
    )
    if args.restart and results_dir.exists():
        print(f"--restart: clearing {results_dir}")
        shutil.rmtree(results_dir)
    ensure_dir(results_dir)
    ckpt_dir = ensure_dir(results_dir / "checkpoints")
    pairs_jsonl = ckpt_dir / "pairs.jsonl"

    ann_dir = resolve_path(ccfg["annotations_dir"])
    feat_table = resolve_path(ccfg["features_table"])
    n_bins = int(ccfg.get("n_bins", 32))
    n_profile_feat = len(ccfg.get("profile_features") or [])
    global_names = list(ccfg["global_features"])
    max_neigh = int(ccfg.get("max_neighbors_per_query", 5))
    n_rand = int(ccfg.get("n_random_negatives_per_query", 3))
    sat_min = float(ccfg.get("saturated_diff_fold_cosine_min", 0.999))

    emb_dir = resolve_path("embeddings/smoke" if smoke else "embeddings/esm2_t33_650M")
    corpus_dir = resolve_path(
        corpus_cfg["smoke_outdir"] if smoke else corpus_cfg["paper_outdir"]
    )
    exp_a_jsonl = resolve_path(
        ccfg["exp_a_smoke_jsonl"] if smoke else ccfg["exp_a_jsonl"]
    )
    exp_a_legacy = exp_a_jsonl.parent.parent / "per_query.json"
    if not exp_a_jsonl.exists() and not exp_a_legacy.exists():
        print(f"ERROR: missing Exp A results ({exp_a_jsonl})", file=sys.stderr)
        return 1

    ids, emb, id_to_row = load_embeddings(emb_dir)
    splits = load_split(corpus_dir / "chains.csv")
    chain_meta = load_chain_meta(corpus_dir / "chains.csv")
    global_map = load_features_table(feat_table, global_names)

    m, ranges, train_ids = load_or_fit_map(
        smoke, {}, ccfg, emb_dir, ann_dir, splits, id_to_row, n_bins, n_profile_feat
    )
    dim = int(m.shape[0])
    prng = np.random.default_rng(int(ccfg.get("random_negative_seed", 42)))

    # Collect retrieval pairs + dims
    planned: list[dict] = []
    queries: list[str] = []
    for rec in iter_exp_a(exp_a_jsonl, exp_a_legacy):
        q = rec["query"]
        queries.append(q)
        q_fold = (chain_meta.get(q) or {}).get("scop_fold", "")
        for nb in (rec.get("neighbors") or [])[:max_neigh]:
            h = nb["chain_id"]
            h_fold = (chain_meta.get(h) or {}).get("scop_fold", "")
            planned.append(
                {
                    "query": q,
                    "hit": h,
                    "pair_source": "retrieval",
                    "same_fold": bool(q_fold and q_fold == h_fold),
                    "contrib": nb.get("dims_explained") or {},
                    "retrieval_score": float(nb.get("score") or 0.0),
                }
            )
    if args.max_queries is not None:
        keep_q = set(queries[: args.max_queries])
        planned = [p for p in planned if p["query"] in keep_q]
        queries = queries[: args.max_queries]

    # Random negatives
    train_chains = [c for c, sp in splits.items() if sp == "train" and c in id_to_row]
    train_by_fold: dict[str, list[str]] = {}
    query_folds: dict[str, str] = {}
    for c in train_chains + queries:
        meta = chain_meta.get(c) or {}
        fold = meta.get("scop_fold", "")
        query_folds[c] = fold
        if splits.get(c) == "train" and fold:
            train_by_fold.setdefault(fold, []).append(c)

    holdout_queries = [q for q in queries if splits.get(q) == "holdout"]
    for q, h in sample_random_negatives(
        holdout_queries, query_folds, train_by_fold, train_chains, n_rand, prng
    ):
        q_fold = query_folds.get(q, "")
        h_fold = query_folds.get(h, "")
        planned.append(
            {
                "query": q,
                "hit": h,
                "pair_source": "random_negative",
                "same_fold": bool(q_fold and q_fold == h_fold),
                "contrib": {},
                "retrieval_score": None,
            }
        )

    needed = {p["query"] for p in planned} | {p["hit"] for p in planned}
    profiles = load_profiles(ann_dir, needed, n_bins, n_profile_feat)

    done = {(r["query"], r["hit"]) for r in load_jsonl(pairs_jsonl)}
    pending = [p for p in planned if (p["query"], p["hit"]) not in done]
    print(
        f"Exp C | map_train={len(train_ids)} planned={len(planned)} "
        f"pending={len(pending)} (retrieval + random negatives)"
    )

    t0 = time.time()
    progress = BatchProgress(max(len(pending), 1), label="expC-pairs")
    if not pending:
        progress.tick(skipped=True, force_print=True)

    for i, p in enumerate(pending):
        rs = p.get("retrieval_score")
        if rs is None and p["query"] in id_to_row and p["hit"] in id_to_row:
            eq = emb[id_to_row[p["query"]]]
            eh = emb[id_to_row[p["hit"]]]
            rs = float(np.dot(eq, eh) / (np.linalg.norm(eq) * np.linalg.norm(eh) + 1e-12))
        scores = score_pair_record(
            p.get("contrib") or None,
            rs,
            p["query"],
            p["hit"],
            profiles,
            global_map,
            m,
            ranges,
            dim,
            prng,
        )
        gcos = scores.get("global_feature_cosine")
        saturated = (
            not p.get("same_fold")
            and gcos is not None
            and float(gcos) >= sat_min
        )
        row = {
            "query": p["query"],
            "hit": p["hit"],
            "pair_source": p["pair_source"],
            "same_fold": bool(p.get("same_fold")),
            "saturated_diff_fold": saturated,
            **scores,
            "unix": int(time.time()),
        }
        append_jsonl(pairs_jsonl, row)
        progress.tick(ok=True, force_print=(i + 1 >= len(pending)))

    records = load_jsonl(pairs_jsonl)
    wanted = {(p["query"], p["hit"]) for p in planned}
    records = [r for r in records if (r["query"], r["hit"]) in wanted]

    score_methods = [
        "xq_profile_spearman",
        "retrieval_score",
        "random_dims_spearman",
        "uniform_dims_spearman",
        "attribution_entropy",
        "attribution_top1_mass",
    ]
    subsets = ["all", "retrieval", "random_negative"]

    evaluations: dict[str, dict] = {}
    for method in score_methods:
        evaluations[method] = {}
        for sub in subsets:
            subset_key = None if sub == "all" else sub
            ev = evaluate_method(records, method, subset=subset_key)
            # entropy/top1: higher concentration may indicate relatedness — keep as-is
            evaluations[method][sub] = ev

    # Saturated negative cohort (retrieval diff-fold high cosine)
    sat_records = [r for r in records if r.get("saturated_diff_fold")]
    sat_eval = {}
    for method in ("xq_profile_spearman", "retrieval_score"):
        if not sat_records:
            sat_eval[method] = {"n": 0}
            continue
        scores = [float(r[method]) for r in sat_records if r.get(method) is not None]
        sat_eval[method] = {
            "n": len(scores),
            "mean": float(np.mean(scores)) if scores else None,
        }

    # ROC/PR curve points for primary method
    curve_rows: list[tuple[str, float, float, str]] = []
    primary = "xq_profile_spearman"
    labels = np.array([bool(r.get("same_fold")) for r in records])
    sc = np.array(
        [
            float(r[primary])
            for r in records
            if r.get(primary) is not None and r[primary] == r[primary]
        ]
    )
    lab = np.array(
        [
            bool(r.get("same_fold"))
            for r in records
            if r.get(primary) is not None and r[primary] == r[primary]
        ]
    )
    if sc.size > 10:
        fpr, tpr = roc_points(sc, lab)
        for fp, tp in zip(fpr, tpr):
            curve_rows.append((primary, float(fp), float(tp), "roc"))
        rec, prec = pr_points(sc, lab)
        for rv, pv in zip(rec, prec):
            curve_rows.append((primary, float(rv), float(pv), "pr"))

    for baseline in ("retrieval_score", "random_dims_spearman"):
        sc2 = np.array(
            [
                float(r[baseline])
                for r in records
                if r.get(baseline) is not None and r[baseline] == r[baseline]
            ]
        )
        lab2 = np.array(
            [
                bool(r.get("same_fold"))
                for r in records
                if r.get(baseline) is not None and r[baseline] == r[baseline]
            ]
        )
        if sc2.size > 10:
            fpr, tpr = roc_points(sc2, lab2)
            for fp, tp in zip(fpr, tpr):
                curve_rows.append((baseline, float(fp), float(tp), "roc"))

    b3_path = resolve_path(ccfg.get("exp_b_v3_metrics", "results/exp_b_v3/metrics.json"))
    b3_ref = {}
    if b3_path.exists():
        b3_ref = json.loads(b3_path.read_text()).get("primary_cohort") or {}

    metrics = {
        "version": "exp_c",
        "mode": "smoke" if smoke else "paper",
        "n_pairs": len(records),
        "n_retrieval": sum(1 for r in records if r.get("pair_source") == "retrieval"),
        "n_random_negative": sum(
            1 for r in records if r.get("pair_source") == "random_negative"
        ),
        "n_same_fold": sum(1 for r in records if r.get("same_fold")),
        "n_train_map": len(train_ids),
        "evaluations": evaluations,
        "saturated_diff_fold": sat_eval,
        "exp_b_v3_reference": {
            "mann_whitney_p": b3_ref.get("mann_whitney_p"),
            "auroc_same_fold": b3_ref.get("auroc_same_fold"),
            "label_perm_p": b3_ref.get("p_value_label_perm_fold_gap"),
        },
        "elapsed_s": round(time.time() - t0, 1),
        "completed_unix": int(time.time()),
    }
    write_json(results_dir / "metrics.json", metrics)

    with (results_dir / "summary.tsv").open("w") as f:
        f.write("method\tsubset\tn\tauroc\tauprc\tmean_pos\tmean_neg\n")
        for method, subs in evaluations.items():
            for sub, ev in subs.items():
                f.write(
                    f"{method}\t{sub}\t{ev.get('n')}\t{ev.get('auroc')}\t"
                    f"{ev.get('auprc')}\t{ev.get('mean_score_positive')}\t"
                    f"{ev.get('mean_score_negative')}\n"
                )

    with (results_dir / "roc_curves.tsv").open("w") as f:
        f.write("method\tx\ty\tcurve\n")
        for method, x, y, curve in curve_rows:
            f.write(f"{method}\t{x}\t{y}\t{curve}\n")

    primary_auc = evaluations.get("xq_profile_spearman", {}).get("all", {})
    print(
        json.dumps(
            {
                "xq_profile_spearman_auroc": primary_auc.get("auroc"),
                "retrieval_score_auroc": evaluations.get("retrieval_score", {})
                .get("all", {})
                .get("auroc"),
                "random_dims_auroc": evaluations.get("random_dims_spearman", {})
                .get("all", {})
                .get("auroc"),
            },
            indent=2,
        )
    )
    print(f"Wrote {results_dir.relative_to(BIO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
