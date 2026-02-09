from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    indices = []
    while len(indices) < n:
        start = rng.integers(0, max(n - block_size + 1, 1))
        block = list(range(start, min(start + block_size, n)))
        indices.extend(block)
    return np.array(indices[:n])


def permutation_test_stat(
    x: pd.Series,
    y: pd.Series,
    stat_fn,
    runs: int,
    block_size: int | None = None,
    random_seed: int = 7,
) -> dict:
    rng = np.random.default_rng(random_seed)
    xy = pd.concat([x, y], axis=1).dropna()
    if xy.empty:
        return {"pvalue": float("nan"), "null_mean": float("nan"), "null_std": float("nan")}
    x_vals = xy.iloc[:, 0].to_numpy()
    y_vals = xy.iloc[:, 1].to_numpy()
    index = xy.index
    null_stats = []
    for _ in range(runs):
        if block_size:
            indices = block_bootstrap_indices(len(x_vals), block_size, rng)
            x_perm = x_vals[indices]
        else:
            x_perm = rng.permutation(x_vals)
        null_stats.append(stat_fn(pd.Series(x_perm, index=index), pd.Series(y_vals, index=index)))
    null_stats = np.array(null_stats)
    observed = stat_fn(pd.Series(x_vals, index=index), pd.Series(y_vals, index=index))
    pvalue = float((np.sum(null_stats >= observed) + 1) / (runs + 1))
    return {
        "pvalue": pvalue,
        "null_mean": float(np.mean(null_stats)),
        "null_std": float(np.std(null_stats, ddof=1)),
    }


def bh_fdr(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    if not np.any(passed):
        return np.zeros(n, dtype=bool)
    max_idx = np.where(passed)[0].max()
    mask = np.zeros(n, dtype=bool)
    mask[order[: max_idx + 1]] = True
    return mask
