"""Shared Exp A retrieval metrics (fold / superfamily / family)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_chain_meta(chains_csv: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with chains_csv.open() as f:
        for rec in csv.DictReader(f):
            out[rec["chain_id"]] = rec
    return out


def family_key(sccs: str) -> str:
    return sccs or ""


def superfamily_key(sccs: str) -> str:
    bits = (sccs or "").split(".")
    return ".".join(bits[:3]) if len(bits) >= 3 else sccs


def hit_rates(flags: list[list[bool]], ks: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    n = len(flags)
    if n == 0:
        for k in ks:
            out[f"recall@{k}"] = 0.0
            out[f"precision@{k}"] = 0.0
        return out
    for k in ks:
        out[f"recall@{k}"] = sum(1 for row in flags if any(row[:k])) / n
        prec = [(sum(row[:k]) / len(row[:k])) if row[:k] else 0.0 for row in flags]
        out[f"precision@{k}"] = float(np.mean(prec))
    return out


def bootstrap_recall(
    flags: list[list[bool]],
    ks: list[int],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(flags)
    out: dict[str, dict[str, float]] = {}
    if n == 0:
        return out
    idx = np.arange(n)
    for k in ks:
        scores = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            sample = rng.choice(idx, size=n, replace=True)
            scores[b] = sum(1 for i in sample if any(flags[i][:k])) / n
        lo, hi = np.quantile(scores, [0.025, 0.975])
        out[f"recall@{k}"] = {
            "mean": float(scores.mean()),
            "ci_low": float(lo),
            "ci_high": float(hi),
        }
    return out


def evaluate_rankings(
    query_ids: list[str],
    rankings: dict[str, list[str]],
    meta: dict[str, dict],
    ks: list[int],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    fold_flags: list[list[bool]] = []
    sf_flags: list[list[bool]] = []
    fam_flags: list[list[bool]] = []
    for qid in query_ids:
        q = meta[qid]
        qfold = q.get("scop_fold", "")
        qsccs = q.get("scop_sccs", "")
        qsf = superfamily_key(qsccs)
        qfam = family_key(qsccs)
        hits = rankings.get(qid) or []
        f_row, sf_row, fam_row = [], [], []
        for hid in hits:
            h = meta.get(hid) or {}
            f_row.append(bool(qfold) and h.get("scop_fold") == qfold)
            sf_row.append(bool(qsf) and superfamily_key(h.get("scop_sccs", "")) == qsf)
            fam_row.append(bool(qfam) and family_key(h.get("scop_sccs", "")) == qfam)
        fold_flags.append(f_row)
        sf_flags.append(sf_row)
        fam_flags.append(fam_row)
    return {
        "n_queries": len(query_ids),
        "fold": hit_rates(fold_flags, ks),
        "superfamily": hit_rates(sf_flags, ks),
        "family": hit_rates(fam_flags, ks),
        "fold_recall_bootstrap": bootstrap_recall(
            fold_flags, ks, n_boot=n_boot, seed=seed
        ),
    }
