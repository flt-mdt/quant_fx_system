from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .drift import js_divergence, kl_divergence, psi
from .ic import CorrelationMethod, ic_stats
from .mutual_information import MutualInformationConfig, mutual_information_report
from .transfer_entropy import transfer_entropy


@dataclass(frozen=True)
class InformationConfig:
    drift_bins: int = 10
    drift_epsilon: float = 1e-12
    ic_window: int | None = None
    ic_method: CorrelationMethod = "spearman"
    ic_nw_lags: int | None = None
    mi_config: MutualInformationConfig | None = None
    te_bins: int = 8
    te_permutations: int = 0
    te_block_size: int | None = None
    te_random_state: int | None = None


@dataclass
class InformationReport:
    drift: pd.DataFrame
    ic: dict[str, Any]
    mi: dict[str, float]
    te: dict[str, float]


def _block_permutation(values: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    blocks = [values[i : i + block_size] for i in range(0, n, block_size)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)[:n]


def _transfer_entropy_pvalue(
    x: pd.Series,
    y: pd.Series,
    *,
    n_bins: int,
    n_permutations: int,
    block_size: int | None,
    random_state: int | None,
) -> float:
    if n_permutations <= 0:
        return float("nan")
    aligned = pd.concat([x, y], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    observed = transfer_entropy(aligned.iloc[:, 0], aligned.iloc[:, 1], n_bins=n_bins)
    rng = np.random.default_rng(random_state)
    values = aligned.iloc[:, 1].to_numpy()
    stats = []
    for _ in range(n_permutations):
        if block_size is None or block_size <= 0:
            permuted = rng.permutation(values)
        else:
            permuted = _block_permutation(values, block_size, rng)
        stats.append(transfer_entropy(aligned.iloc[:, 0], pd.Series(permuted), n_bins=n_bins))
    stats = np.array(stats)
    return float((stats >= observed).mean())


def build_information_report(
    *,
    train_features: pd.DataFrame,
    live_features: pd.DataFrame | None,
    signal: pd.Series,
    target: pd.Series,
    cfg: InformationConfig | None = None,
) -> InformationReport:
    if cfg is None:
        cfg = InformationConfig()
    if cfg.mi_config is None:
        mi_config = MutualInformationConfig()
    else:
        mi_config = cfg.mi_config

    drift_rows = []
    if live_features is not None:
        for feature in train_features.columns:
            drift_rows.append(
                {
                    "feature": feature,
                    "psi": psi(
                        train_features[feature],
                        live_features[feature],
                        n_bins=cfg.drift_bins,
                        epsilon=cfg.drift_epsilon,
                    ),
                    "js": js_divergence(
                        train_features[feature],
                        live_features[feature],
                        n_bins=cfg.drift_bins,
                        epsilon=cfg.drift_epsilon,
                    ),
                    "kl": kl_divergence(
                        train_features[feature],
                        live_features[feature],
                        n_bins=cfg.drift_bins,
                        epsilon=cfg.drift_epsilon,
                    ),
                }
            )
    drift = pd.DataFrame(drift_rows)

    ic = ic_stats(
        signal,
        target,
        window=cfg.ic_window,
        method=cfg.ic_method,
        nw_lags=cfg.ic_nw_lags,
    )
    mi = mutual_information_report(signal, target, cfg=mi_config)
    te_value = transfer_entropy(signal, target, n_bins=cfg.te_bins)
    te_pvalue = _transfer_entropy_pvalue(
        signal,
        target,
        n_bins=cfg.te_bins,
        n_permutations=cfg.te_permutations,
        block_size=cfg.te_block_size,
        random_state=cfg.te_random_state,
    )
    te = {"te": te_value, "pvalue": te_pvalue}

    return InformationReport(drift=drift, ic=ic, mi=mi, te=te)
