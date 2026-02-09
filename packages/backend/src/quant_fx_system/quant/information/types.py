from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class InformationConfig:
    horizon: int
    feature_shift: int = 1
    output_shift: int = 0
    target: Literal["forward_return", "binary_up", "meta_label"] = "forward_return"
    overlap: bool = True

    ic_method: Literal["pearson", "spearman"] = "spearman"
    nw_lags: int | None = None
    ic_window: int | None = None
    regime_conditional: bool = True

    mi_estimator: Literal["hist", "gaussian_copula"] = "gaussian_copula"
    bins: int = 10
    binning: Literal["quantile", "equal_width"] = "quantile"
    mi_normalize: bool = True

    te_lags: int = 1
    te_source_lags: int = 1
    te_bins: int = 8
    te_binning: Literal["quantile", "equal_width"] = "quantile"
    te_significance: bool = True
    te_perm_runs: int = 200
    te_block_size: int | None = None
    te_top_k: int | None = 25
    te_features: tuple[str, ...] | None = None

    drift_bins: int = 10
    drift_binning: Literal["quantile", "equal_width"] = "quantile"

    multiple_testing: Literal["none", "bh_fdr"] = "bh_fdr"
    alpha: float = 0.05
    random_seed: int = 7

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InformationReport:
    ic: pd.DataFrame
    mi: pd.DataFrame
    te: pd.DataFrame
    drift: pd.DataFrame
    redundancy: pd.DataFrame | None
    metadata: dict
