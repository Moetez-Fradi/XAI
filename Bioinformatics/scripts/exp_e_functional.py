#!/usr/bin/env python3
"""Experiment E — functional annotation transfer / active-site localization.

For holdout queries with UniProt-mapped functional sites, evaluate whether
XQdrant dimension attributions enrich at active/binding sites vs background,
and compare to TM-align structural overlap at those sites.

Pair labels (from Exp A neighbors):
  - functional_match: shared EC (level N) or GO term
  - fold_only: same SCOP fold, no shared EC/GO
  - unrelated: different fold

Artifacts under results/exp_e[/ _smoke]/:
  run_config.json
  checkpoints/pairs.jsonl
  metrics.json
  summary.tsv
  significance.tsv
  localization_comparison.tsv

Resume unless --restart.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import (  # noqa: E402
    BatchProgress,
    ensure_dir,
    load_yaml,
    resolve_path,
    write_json,
)
from exp_b_v2_attribution import (  # noqa: E402
    append_jsonl,
    iter_exp_a,
    load_embeddings,
    load_jsonl,
    load_split,
    mann_whitney_auroc,
)
from exp_e_lib import (  # noqa: E402
    load_dim_map,
    load_dssp_residues,
    neighbor_pair_label,
    permutation_enrichment_null,
    resseq_site_mask,
    residue_attribution_scores,
    site_auc,
    site_enrichment,
    site_seq_indices_from_resseq,
    tmalign_mapping,
    tmalign_site_overlap,
    which_tmalign,
)


def query_id_from_exp_a(rec: dict) -> str | None:
    """Resolve holdout chain id from Exp A checkpoint record."""
    q = rec.get("query") or rec.get("query_chain_id") or rec.get("chain_id")
    if q:
        return str(q)
    pdb_id = rec.get("pdb_id")
    chain = rec.get("chain")
    if pdb_id and chain:
        return f"{str(pdb_id).lower()}_{chain}"
    return None


def fold_key(sccs: str) -> str:
    bits = (sccs or "").split(".")
    return ".".join(bits[:2]) if len(bits) >= 2 else sccs


def parse_ec_go(rec: dict) -> tuple[set[str], set[str]]:
    ecs = {x.strip() for x in (rec.get("ec_numbers") or "").split(";") if x.strip()}
    gos = {x.strip() for x in (rec.get("go_ids") or "").split(";") if x.strip()}
    return ecs, gos


def chain_uniprot(meta: dict[str, dict], neighbor_ec_go: dict, cid: str, nb: dict | None = None) -> str:
    if nb and nb.get("uniprot"):
        return str(nb["uniprot"]).strip()
    acc = meta.get(cid, {}).get("uniprot_accession") or meta.get(cid, {}).get("uniprot")
    if acc:
        return str(acc).strip()
    # fallback: first EC/GO map doesn't have acc — use benchmark lookup
    return ""


def load_benchmark(proc_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    bench_tsv = proc_dir / "benchmark_queries.tsv"
    if not bench_tsv.exists():
        raise FileNotFoundError(f"Run prep_exp_e_benchmark.py first ({bench_tsv})")
    queries: list[dict] = []
    sites: dict[str, dict] = {}
    with bench_tsv.open() as f:
        for rec in csv.DictReader(f):
            queries.append(rec)
            sp = proc_dir / rec["site_json"]
            sites[rec["chain_id"]] = json.loads(sp.read_text())
    return queries, sites


def load_meta(chains_csv: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with chains_csv.open() as f:
        for rec in csv.DictReader(f):
            out[rec["chain_id"]] = rec
    return out


def gunzip_csv(path: Path) -> Path:
    if path.exists():
        return path
    gz = path.parent / f"{path.name}.gz"
    if gz.exists():
        import gzip

        with gzip.open(gz, "rt") as fin, path.open("w") as fout:
            fout.write(fin.read())
    return path


def build_neighbor_ec_go(meta: dict[str, dict], sifts_enzyme: Path, sifts_go: Path) -> dict[str, tuple[set[str], set[str]]]:
    from exp_e_lib import load_sifts_chain_table

    enzyme_csv = gunzip_csv(sifts_enzyme)
    go_csv = gunzip_csv(sifts_go)
    ec_map = load_sifts_chain_table(enzyme_csv, ("PDB", "CHAIN"), "EC") if enzyme_csv.exists() else {}
    go_map = load_sifts_chain_table(go_csv, ("PDB", "CHAIN"), "GO_ID") if go_csv.exists() else {}
    out: dict[str, tuple[set[str], set[str]]] = {}
    for cid in meta:
        pdb_id, chain = cid.split("_", 1)
        key = (pdb_id.lower(), chain)
        out[cid] = (ec_map.get(key, set()), go_map.get(key, set()))
    return out


def run_tmalign_batch(
    tasks: list[tuple[str, str, str, str]],
    struct_dir: Path,
    tm_bin: str,
    threads: int,
    ckpt: Path,
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    done: dict[tuple[str, str], list[tuple[int, int]]] = {}
    if ckpt.exists():
        for rec in load_jsonl(ckpt):
            mapping = [(int(a), int(b)) for a, b in rec.get("mapping") or []]
            key = (rec["query_chain_id"], rec["neighbor_chain_id"])
            if mapping:
                done[key] = mapping
    pending = [t for t in tasks if (t[0], t[1]) not in done]
    if not pending:
        return done

    def _one(q: str, n: str) -> tuple[tuple[str, str], list[tuple[int, int]]]:
        qp = struct_dir / f"{q}.pdb"
        np_ = struct_dir / f"{n}.pdb"
        if not qp.exists() or not np_.exists():
            return (q, n), []
        return (q, n), tmalign_mapping(tm_bin, qp, np_)

    progress = BatchProgress(len(pending), label="expE-tmalign")
    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        futs = {ex.submit(_one, q, n): (q, n) for q, n, _, _ in pending}
        for fut in as_completed(futs):
            key, mapping = fut.result()
            done[key] = mapping
            append_jsonl(ckpt, {"query_chain_id": key[0], "neighbor_chain_id": key[1], "mapping": mapping})
            progress.tick(ok=bool(mapping))
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--skip-tmalign", action="store_true")
    args = ap.parse_args()

    ecfg = load_yaml("exp_e.yaml")
    smoke = bool(args.smoke)
    corpus_dir = resolve_path(
        ecfg.get("corpus_smoke_dir" if smoke else "corpus_dir", "data/processed/corpus")
    )
    proc_dir = resolve_path(ecfg["processed_dir"])
    results_dir = resolve_path(
        ecfg["smoke_results_dir"] if smoke else ecfg["results_dir"]
    )
    if args.restart and results_dir.exists():
        shutil.rmtree(results_dir)
    ensure_dir(results_dir)
    pairs_ckpt = ensure_dir(results_dir / "checkpoints") / "pairs.jsonl"
    tm_ckpt = results_dir / "checkpoints" / "tmalign.jsonl"

    try:
        benchmark, sites = load_benchmark(proc_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if not benchmark:
        print("ERROR: empty benchmark — run prep_exp_e_benchmark.py", file=sys.stderr)
        return 1

    bench_ids = {q["chain_id"] for q in benchmark}
    bench_lookup = {q["chain_id"]: q for q in benchmark}

    exp_a_jsonl = resolve_path(
        ecfg["exp_a_smoke_jsonl"] if smoke else ecfg["exp_a_jsonl"]
    )
    if not exp_a_jsonl.exists():
        print(f"ERROR: missing {exp_a_jsonl}", file=sys.stderr)
        return 1

    meta = load_meta(corpus_dir / "chains.csv")
    sifts_dir = resolve_path(ecfg.get("sifts_dir", "data/raw/sifts"))
    neighbor_ec_go = build_neighbor_ec_go(
        meta,
        sifts_dir / "pdb_chain_enzyme.csv",
        sifts_dir / "pdb_chain_go.csv",
    )

    ann_dir = resolve_path(ecfg["annotations_dir"])
    emb_dir = resolve_path(ecfg.get("embeddings_dir", "embeddings/esm2_t33_650M"))
    ids, emb, id_to_row = load_embeddings(emb_dir)
    splits = load_split(corpus_dir / "chains.csv")
    M, _ = load_dim_map(ecfg, ann_dir, emb_dir, splits, id_to_row)
    n_bins = int(ecfg.get("n_bins", 32))

    max_nb = int(ecfg.get("max_neighbors_per_query", 5))
    ec_level = int(ecfg.get("functional_ec_level", 3))
    generic_go = set(ecfg.get("generic_go_ids") or [])
    include_fold_only = bool(ecfg.get("include_same_fold_nonfunctional", True))
    struct_dir = resolve_path(ecfg.get("baselines_structures_dir", "data/baselines/exp_a/structures"))

    # Collect candidate pairs from Exp A
    candidates: list[dict] = []
    for rec in iter_exp_a(exp_a_jsonl, exp_a_jsonl.parent.parent / "per_query.json"):
        q = query_id_from_exp_a(rec)
        if not q or q not in bench_ids:
            continue
        q_info = bench_lookup[q]
        q_ec, q_go = parse_ec_go(q_info)
        q_fold = q_info.get("scop_fold") or meta.get(q, {}).get("scop_fold", "")
        q_uni = q_info.get("uniprot_accession") or chain_uniprot(meta, neighbor_ec_go, q)
        for nb in (rec.get("neighbors") or [])[:max_nb]:
            n = nb.get("chain_id")
            if not n or n == q:
                continue
            n_ec, n_go = neighbor_ec_go.get(n, (set(), set()))
            n_fold = meta.get(n, {}).get("scop_fold", "")
            n_uni = chain_uniprot(meta, neighbor_ec_go, n, nb)
            label = neighbor_pair_label(
                q_ec,
                q_go,
                q_fold,
                q_uni,
                n_ec,
                n_go,
                n_fold,
                n_uni,
                ec_level=ec_level,
                generic_go=generic_go,
            )
            if label == "unrelated":
                continue
            if label == "fold_only" and not include_fold_only:
                continue
            candidates.append(
                {
                    "query_chain_id": q,
                    "neighbor_chain_id": n,
                    "pair_label": label,
                    "contrib": nb.get("dims_explained") or {},
                    "retrieval_score": float(nb.get("score") or 0),
                    "same_fold": fold_key(q_fold) == fold_key(n_fold),
                }
            )

    done_keys = {(r["query_chain_id"], r["neighbor_chain_id"]) for r in load_jsonl(pairs_ckpt)}
    pending = [c for c in candidates if (c["query_chain_id"], c["neighbor_chain_id"]) not in done_keys]
    print(
        f"Exp E | benchmark={len(benchmark)} candidate_pairs={len(candidates)} "
        f"pending={len(pending)}"
    )
    if not candidates:
        print(
            "WARN: no candidate pairs — check Exp A `query` ids vs benchmark",
            file=sys.stderr,
        )

    progress = BatchProgress(max(len(pending), 1), label="expE-pairs")
    if not pending:
        progress.tick(skipped=True, force_print=True)

    for i, c in enumerate(pending):
        q, n = c["query_chain_id"], c["neighbor_chain_id"]
        site_doc = sites[q]
        residues = load_dssp_residues(ann_dir, q)
        site_resseq = set(site_doc.get("site_resseq") or [])
        rec: dict = {
            "query_chain_id": q,
            "neighbor_chain_id": n,
            "pair_label": c["pair_label"],
            "retrieval_score": c["retrieval_score"],
            "same_fold": c["same_fold"],
            "n_site_residues": len(site_resseq),
        }
        contrib = c["contrib"]
        if not contrib or not residues or not site_resseq:
            rec["error"] = "missing_contrib_or_sites"
            append_jsonl(pairs_ckpt, rec)
            progress.tick(ok=False)
            continue

        scores = residue_attribution_scores(residues, contrib, M, n_bins)
        smask = resseq_site_mask(residues, site_resseq)
        enrich = site_enrichment(scores, smask)
        rec["xq_site_mean"] = enrich["site_mean"]
        rec["xq_background_mean"] = enrich["background_mean"]
        rec["xq_enrichment_ratio"] = enrich["enrichment_ratio"]
        rec["xq_site_auc"] = site_auc(scores, smask)
        rec["n_dims"] = len(contrib)
        rec["unix"] = int(time.time())
        append_jsonl(pairs_ckpt, rec)
        progress.tick(ok=True, force_print=(i + 1 >= len(pending)))

    records = load_jsonl(pairs_ckpt)

    # TM-align localization (subset)
    tm_results: dict[tuple[str, str], dict] = {}
    if not args.skip_tmalign and struct_dir.exists():
        try:
            tm_bin = which_tmalign()
        except FileNotFoundError as e:
            print(f"WARN: {e} — skipping TM-align", file=sys.stderr)
            tm_bin = ""
        if tm_bin:
            max_tm = int(
                ecfg.get("tmalign_max_pairs_smoke" if smoke else "tmalign_max_pairs", 150)
            )
            if max_tm <= 0:
                max_tm = len(records)
            tm_tasks = []
            seen: set[tuple[str, str]] = set()
            for r in records:
                if r.get("error"):
                    continue
                key = (r["query_chain_id"], r["neighbor_chain_id"])
                if key in seen:
                    continue
                seen.add(key)
                tm_tasks.append((key[0], key[1], "", ""))
                if len(tm_tasks) >= max_tm:
                    break
            mappings = run_tmalign_batch(
                tm_tasks,
                struct_dir,
                tm_bin,
                int(ecfg.get("tmalign_threads", 4)),
                tm_ckpt,
            )
            for (q, n), mapping in mappings.items():
                site_doc = sites[q]
                residues = load_dssp_residues(ann_dir, q)
                site_idx = site_seq_indices_from_resseq(residues, site_doc.get("site_resseq") or [])
                aln_q = {i - 1 for i, _ in mapping}  # TMalign 1-based → 0-based
                tm_ov = tmalign_site_overlap(aln_q, site_idx, len(residues))
                tm_results[(q, n)] = tm_ov
                # Patch record
                for r in records:
                    if r["query_chain_id"] == q and r["neighbor_chain_id"] == n:
                        r["tm_aligned_in_site_fraction"] = tm_ov["aligned_in_site_fraction"]
                        r["tm_site_enrichment_ratio"] = tm_ov["enrichment_ratio"]
                        r["tm_n_aligned"] = tm_ov.get("n_aligned")

    # Aggregate metrics
    n_perm = int(ecfg.get("n_permutations", 5000))
    seed = int(ecfg.get("permutation_seed", 42))

    def _filter(label: str | None = None) -> list[dict]:
        rows = [r for r in records if not r.get("error") and r.get("xq_enrichment_ratio") is not None]
        if label:
            rows = [r for r in rows if r.get("pair_label") == label]
        return rows

    func_rows = _filter("functional_match")
    fold_rows = _filter("fold_only")
    all_rows = _filter()

    def _mean(rows: list[dict], key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else None

    # Pooled site vs background Mann-Whitney (functional-match pairs)
    dims_lookup: dict[tuple[str, str], dict] = {}
    for rec in iter_exp_a(exp_a_jsonl, exp_a_jsonl.parent.parent / "per_query.json"):
        q = query_id_from_exp_a(rec)
        if not q or q not in bench_ids:
            continue
        for nb in rec.get("neighbors") or []:
            n = nb.get("chain_id")
            if n:
                dims_lookup[(q, n)] = nb.get("dims_explained") or {}

    pooled_site_scores: list[float] = []
    pooled_bg_scores: list[float] = []
    for r in func_rows:
        q, n = r["query_chain_id"], r["neighbor_chain_id"]
        contrib = dims_lookup.get((q, n))
        if not contrib:
            continue
        site_doc = sites[q]
        residues = load_dssp_residues(ann_dir, q)
        sc = residue_attribution_scores(residues, contrib, M, n_bins)
        sm = resseq_site_mask(residues, site_doc.get("site_resseq") or [])
        if sm.sum() == 0 or (~sm).sum() == 0:
            continue
        pooled_site_scores.extend(sc[sm].tolist())
        pooled_bg_scores.extend(sc[~sm].tolist())

    mw_auc, mw_p = None, None
    if pooled_site_scores and pooled_bg_scores:
        site_labels = np.array(
            [True] * len(pooled_site_scores) + [False] * len(pooled_bg_scores),
            dtype=bool,
        )
        site_scores = np.array(pooled_site_scores + pooled_bg_scores, dtype=np.float64)
        mw_auc, mw_p = mann_whitney_auroc(site_scores, site_labels)

    # Functional vs fold-only cohort comparison (pair-level enrichment)
    from scipy.stats import mannwhitneyu

    def _cohort_mw(rows_a: list[dict], rows_b: list[dict], key: str) -> tuple[float | None, float | None]:
        a = np.array(
            [float(r[key]) for r in rows_a if r.get(key) is not None and np.isfinite(float(r[key]))],
            dtype=np.float64,
        )
        b = np.array(
            [float(r[key]) for r in rows_b if r.get(key) is not None and np.isfinite(float(r[key]))],
            dtype=np.float64,
        )
        if len(a) < 5 or len(b) < 5:
            return None, None
        try:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            return None, None
        auc = float(u / (len(a) * len(b)))
        return auc, float(p)

    xq_cohort_auc, xq_cohort_p = _cohort_mw(func_rows, fold_rows, "xq_enrichment_ratio")
    tm_cohort_auc, tm_cohort_p = _cohort_mw(func_rows, fold_rows, "tm_site_enrichment_ratio")

    # Null: is mean XQ enrichment > 1 for functional pairs? (permutation on pair means)
    enrich_vals = np.array(
        [float(r["xq_enrichment_ratio"]) for r in func_rows if np.isfinite(float(r["xq_enrichment_ratio"]))],
        dtype=np.float64,
    )
    perm_p_gt_one = float("nan")
    perm_p_vs_one = float("nan")
    if len(enrich_vals) >= 3:
        from scipy.stats import binomtest, ttest_1samp

        # Fraction of pairs with enrichment > 1 (binomial vs 0.5 null)
        n_gt = int(np.sum(enrich_vals > 1.0))
        perm_p_vs_one = float(
            binomtest(n_gt, len(enrich_vals), p=0.5, alternative="greater").pvalue
        )
        # One-sample t-test: mean enrichment > 1
        tstat, t_p = ttest_1samp(enrich_vals, popmean=1.0, alternative="greater")
        perm_p_gt_one = float(t_p)

    # Per-pair residue shuffle null (first functional pair with sites)
    residue_perm_p = float("nan")
    if func_rows:
        r0 = func_rows[0]
        q, n = r0["query_chain_id"], r0["neighbor_chain_id"]
        residues = load_dssp_residues(ann_dir, q)
        site_doc = sites[q]
        smask = resseq_site_mask(residues, site_doc.get("site_resseq") or [])
        contrib = dims_lookup.get((q, n))
        if contrib is not None and smask.sum() > 0 and (~smask).sum() > 0:
            sc = residue_attribution_scores(residues, contrib, M, n_bins)
            _, residue_perm_p = permutation_enrichment_null(sc, smask, min(n_perm, 2000), seed + 7)

    tm_func = _mean(func_rows, "tm_site_enrichment_ratio")
    tm_fold = _mean(fold_rows, "tm_site_enrichment_ratio")
    xq_func = _mean(func_rows, "xq_enrichment_ratio")
    xq_fold = _mean(fold_rows, "xq_enrichment_ratio")

    metrics = {
        "version": "exp_e",
        "mode": "smoke" if smoke else "paper",
        "n_benchmark_queries": len(benchmark),
        "n_pairs_total": len(records),
        "n_pairs_functional_match": len(func_rows),
        "n_pairs_fold_only": len(fold_rows),
        "xq_site_enrichment": {
            "functional_match_mean": xq_func,
            "fold_only_mean": xq_fold,
            "functional_vs_fold_delta": (xq_func - xq_fold)
            if xq_func is not None and xq_fold is not None
            else None,
            "mean_site_auc_functional": _mean(func_rows, "xq_site_auc"),
        },
        "tmalign_site_enrichment": {
            "functional_match_mean": tm_func,
            "fold_only_mean": tm_fold,
            "n_pairs_with_tmalign": sum(
                1
                for r in records
                if r.get("tm_site_enrichment_ratio") is not None
                and np.isfinite(float(r["tm_site_enrichment_ratio"]))
            ),
        },
        "significance": {
            "mann_whitney_site_vs_background_auc": mw_auc,
            "mann_whitney_site_vs_background_p": mw_p,
            "mann_whitney_xq_functional_vs_fold_only_p": xq_cohort_p,
            "mann_whitney_tm_functional_vs_fold_only_p": tm_cohort_p,
            "perm_mean_enrichment_vs_shuffle_p": perm_p_gt_one,
            "perm_mean_enrichment_gt_one_p": perm_p_vs_one,
            "residue_shuffle_enrichment_p_example_pair": residue_perm_p,
        },
    }
    write_json(results_dir / "metrics.json", metrics)

    summary_rows = [
        ["metric", "value"],
        ["n_benchmark_queries", len(benchmark)],
        ["n_pairs_functional", len(func_rows)],
        ["n_pairs_fold_only", len(fold_rows)],
        ["xq_enrichment_functional_mean", xq_func],
        ["xq_enrichment_fold_only_mean", xq_fold],
        ["xq_site_auc_functional_mean", _mean(func_rows, "xq_site_auc")],
        ["tm_enrichment_functional_mean", tm_func],
        ["tm_enrichment_fold_only_mean", tm_fold],
        ["mann_whitney_site_vs_bg_auc", mw_auc],
        ["mann_whitney_site_vs_bg_p", mw_p],
        ["xq_functional_vs_fold_only_p", xq_cohort_p],
        ["tm_functional_vs_fold_only_p", tm_cohort_p],
        ["perm_mean_enrichment_gt_one_p", perm_p_vs_one],
    ]
    with (results_dir / "summary.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows(summary_rows)

    sig_rows = [
        ["test", "statistic", "p_value", "notes"],
        ["mann_whitney_site_vs_background", mw_auc, mw_p, "pooled functional-match pairs, per-residue attribution"],
        ["mann_whitney_xq_func_vs_fold", xq_cohort_auc, xq_cohort_p, "pair-level xq_enrichment_ratio"],
        ["mann_whitney_tm_func_vs_fold", tm_cohort_auc, tm_cohort_p, "pair-level tm_site_enrichment_ratio"],
        ["perm_mean_xq_enrichment_gt_one", _mean(func_rows, "xq_enrichment_ratio"), perm_p_vs_one, "null: enrichment<=1"],
        ["residue_shuffle_enrichment", "ratio", residue_perm_p, "example functional pair"],
    ]
    with (results_dir / "significance.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows(sig_rows)

    loc_rows = [
        [
            "query_chain_id",
            "neighbor_chain_id",
            "pair_label",
            "xq_enrichment_ratio",
            "xq_site_auc",
            "tm_site_enrichment_ratio",
            "tm_aligned_in_site_fraction",
        ]
    ]
    for r in records:
        if r.get("error"):
            continue
        loc_rows.append(
            [
                r["query_chain_id"],
                r["neighbor_chain_id"],
                r.get("pair_label"),
                r.get("xq_enrichment_ratio"),
                r.get("xq_site_auc"),
                r.get("tm_site_enrichment_ratio"),
                r.get("tm_aligned_in_site_fraction"),
            ]
        )
    with (results_dir / "localization_comparison.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows(loc_rows)

    write_json(
        results_dir / "run_config.json",
        {
            "ecfg": ecfg,
            "n_benchmark": len(benchmark),
            "n_candidates": len(candidates),
            "exp_a_jsonl": str(exp_a_jsonl),
        },
    )

    print(f"Exp E done → {results_dir}/")
    print(f"  functional pairs={len(func_rows)} xq_enrichment={xq_func} tm_enrichment={tm_func}")
    print(f"  MW site vs bg p={mw_p} xq func vs fold p={xq_cohort_p} tm func vs fold p={tm_cohort_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
