#!/usr/bin/env python3
"""Exp A baseline: TM-align re-rank of Foldseek candidates.

Full holdout×train TM-align is intractable (~30M pairs). We take each
query's Foldseek top-N candidates and re-rank by TM-score — the usual
structure-refinement baseline.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import (  # noqa: E402
    BatchProgress,
    ensure_dir,
    load_yaml,
    resolve_path,
    write_json,
)
from exp_a_metrics_lib import evaluate_rankings, load_chain_meta  # noqa: E402

_TM_RE = re.compile(r"TM-score\s*=\s*([0-9.]+)")


def which_tmalign() -> str:
    for name in ("TMalign", "tmalign"):
        p = shutil.which(name)
        if p:
            return p
    raise FileNotFoundError("TMalign not on PATH")


def tm_score(bin_path: str, q_pdb: Path, t_pdb: Path) -> float:
    proc = subprocess.run(
        [bin_path, str(q_pdb), str(t_pdb)],
        capture_output=True,
        text=True,
        check=False,
    )
    scores = [float(m) for m in _TM_RE.findall(proc.stdout)]
    return max(scores) if scores else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    bcfg = load_yaml("exp_a_baselines.yaml")
    tcfg = bcfg.get("tmalign") or {}
    corpus_cfg = load_yaml("corpus.yaml")
    smoke = bool(args.smoke)
    tmalign = which_tmalign()

    base = resolve_path(
        "data/baselines/exp_a_smoke" if smoke else bcfg["baselines_dir"]
    )
    struct_dir = base / "structures"
    results_root = resolve_path(
        bcfg["smoke_results_dir"] if smoke else bcfg["results_dir"]
    )
    foldseek_rankings = results_root / "foldseek" / "rankings.json"
    if not foldseek_rankings.exists():
        print(
            "ERROR: need Foldseek rankings first (exp_a_baseline_foldseek.py).",
            file=sys.stderr,
        )
        return 1

    out_dir = ensure_dir(results_root / "tmalign")
    ckpt_jsonl = out_dir / "checkpoints" / "queries.jsonl"
    ensure_dir(ckpt_jsonl.parent)
    rankings_path = out_dir / "rankings.json"

    if args.restart:
        if ckpt_jsonl.exists():
            ckpt_jsonl.unlink()
        rankings_path.unlink(missing_ok=True)

    fs_ranks = json.loads(foldseek_rankings.read_text())
    max_cand = int(tcfg.get("max_candidates", 20))
    topk = args.topk
    threads = int(tcfg.get("threads", 4))

    done: set[str] = set()
    if ckpt_jsonl.exists():
        with ckpt_jsonl.open() as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["query"])

    pending = [q for q in sorted(fs_ranks.keys()) if q not in done]
    print(
        f"TM-align re-rank: {len(pending)} pending / {len(fs_ranks)} queries "
        f"(candidates≤{max_cand}, threads={threads})"
    )

    def process_query(qid: str) -> dict:
        q_pdb = struct_dir / f"{qid}.pdb"
        if not q_pdb.exists():
            return {"query": qid, "hits": [], "scores": []}
        cands = fs_ranks.get(qid, [])[:max_cand]
        scored: list[tuple[float, str]] = []
        for hid in cands:
            t_pdb = struct_dir / f"{hid}.pdb"
            if not t_pdb.exists():
                continue
            sc = tm_score(tmalign, q_pdb, t_pdb)
            scored.append((sc, hid))
        scored.sort(key=lambda x: -x[0])
        hits = [h for _s, h in scored[:topk]]
        scores = [s for s, _h in scored[:topk]]
        return {"query": qid, "hits": hits, "scores": scores}

    progress = BatchProgress(max(len(pending), 1), label="tmalign")
    t0 = time.time()
    if not pending:
        progress.tick(skipped=True, force_print=True)

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = {ex.submit(process_query, q): q for q in pending}
        for fut in as_completed(futs):
            rec = fut.result()
            with ckpt_jsonl.open("a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            progress.tick(ok=True)

    rankings: dict[str, list[str]] = {}
    with ckpt_jsonl.open() as f:
        for line in f:
            rec = json.loads(line)
            rankings[rec["query"]] = rec.get("hits") or []
    write_json(rankings_path, rankings)

    corpus_dir = resolve_path(
        corpus_cfg["smoke_outdir"] if smoke else corpus_cfg["paper_outdir"]
    )
    meta = load_chain_meta(corpus_dir / "chains.csv")
    ks = list(bcfg.get("limits") or [1, 5, 10])
    qids = sorted(rankings.keys())
    metrics = evaluate_rankings(
        qids,
        rankings,
        meta,
        ks,
        n_boot=int(bcfg.get("bootstrap", 1000)),
        seed=int(bcfg.get("bootstrap_seed", 42)),
    )
    metrics["method"] = "tmalign_rerank_foldseek"
    metrics["mode"] = "smoke" if smoke else "paper"
    metrics["note"] = (
        "TM-align re-ranks Foldseek candidates (not exhaustive gallery search)."
    )
    metrics["elapsed_s"] = round(time.time() - t0, 1)
    write_json(out_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
