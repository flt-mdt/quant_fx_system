from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NormalizeMethod = str


@dataclass(frozen=True)
class MutualInformationConfig:
    n_bins: int = 10
    mi_normalize: bool = False
    normalize_method: NormalizeMethod = "min"
    n_permutations: int = 0
    block_size: int | None = None
    random_state: int | None = None


def _bin_series(series: pd.Series, n_bins: int) -> pd.Categorical:
    return pd.qcut(series, q=n_bins, duplicates="drop")


def _entropy_from_counts(counts: np.ndarray) -> float:
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def mutual_information(
    x: pd.Series,
    y: pd.Series,
    *,
    n_bins: int = 10,
) -> float:
    aligned = pd.concat([x, y], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    bx = _bin_series(aligned.iloc[:, 0], n_bins=n_bins)
    by = _bin_series(aligned.iloc[:, 1], n_bins=n_bins)
    joint = pd.crosstab(bx, by)
    hx = _entropy_from_counts(joint.sum(axis=1).to_numpy())
    hy = _entropy_from_counts(joint.sum(axis=0).to_numpy())
    hxy = _entropy_from_counts(joint.to_numpy().ravel())
    return float(hx + hy - hxy)


def normalized_mutual_information(
    x: pd.Series,
    y: pd.Series,
    *,
    n_bins: int = 10,
    method: NormalizeMethod = "min",
) -> float:
    aligned = pd.concat([x, y], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    bx = _bin_series(aligned.iloc[:, 0], n_bins=n_bins)
    by = _bin_series(aligned.iloc[:, 1], n_bins=n_bins)
    joint = pd.crosstab(bx, by)
    hx = _entropy_from_counts(joint.sum(axis=1).to_numpy())
    hy = _entropy_from_counts(joint.sum(axis=0).to_numpy())
    mi = float(hx + hy - _entropy_from_counts(joint.to_numpy().ravel()))
    if method == "sqrt":
        denom = np.sqrt(hx * hy)
    else:
        denom = min(hx, hy)
    if denom == 0:
        return float("nan")
    return float(mi / denom)


def mutual_information_report(
    x: pd.Series,
    y: pd.Series,
    *,
    cfg: MutualInformationConfig | None = None,
) -> dict[str, float]:
    if cfg is None:
        cfg = MutualInformationConfig()
    mi = mutual_information(x, y, n_bins=cfg.n_bins)
    report = {"mi": mi}
    if cfg.mi_normalize:
        report["nmi"] = normalized_mutual_information(
            x, y, n_bins=cfg.n_bins, method=cfg.normalize_method
        )
    if cfg.n_permutations <= 0:
        report["pvalue"] = float("nan")
        return report

    rng = np.random.default_rng(cfg.random_state)
    aligned = pd.concat([x, y], axis=1).dropna()
    if aligned.empty:
        report["pvalue"] = float("nan")
        return report
    observed = mi
    null_stats = []
    block_size = cfg.block_size
    if block_size is not None and block_size <= 0:
        block_size = None
    for _ in range(cfg.n_permutations):
        if block_size is None:
            shuffled = aligned.iloc[:, 1].sample(
                frac=1.0, replace=False, random_state=rng.integers(0, 2**32 - 1)
            )
            permuted = pd.Series(shuffled.to_numpy(), index=aligned.index)
        else:
            values = aligned.iloc[:, 1].to_numpy()
            n = len(values)
            blocks = [values[i : i + block_size] for i in range(0, n, block_size)]
            rng.shuffle(blocks)
            permuted = pd.Series(np.concatenate(blocks)[:n], index=aligned.index)
        null_stats.append(mutual_information(aligned.iloc[:, 0], permuted, n_bins=cfg.n_bins))
    null_array = np.array(null_stats)
    report["pvalue"] = float((null_array >= observed).mean())
    return report
