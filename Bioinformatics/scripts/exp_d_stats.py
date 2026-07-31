"""Significance tests for Experiment D cross-pair correlations."""
from __future__ import annotations
import numpy as np

def perm_p_value_spearman(xs: np.ndarray, ys: np.ndarray, n_perm: int, seed: int) -> tuple[float, float, np.ndarray]:
    from exp_b_v2_attribution import spearman
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    real = float(spearman(xs, ys))
    prng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        null[i] = float(spearman(xs, ys[prng.permutation(len(ys))]))
    p = float((np.sum(np.abs(null) >= abs(real)) + 1) / (n_perm + 1))
    return (real, p, null)

def bootstrap_spearman_ci(xs: np.ndarray, ys: np.ndarray, n_boot: int, seed: int, alpha: float=0.05) -> tuple[float, float, float]:
    from exp_b_v2_attribution import spearman
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = len(xs)
    prng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = prng.integers(0, n, n)
        boots[i] = float(spearman(xs[idx], ys[idx]))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (float(spearman(xs, ys)), lo, hi)
