from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

import pandas as pd

RegimeMethod = Literal["quantile_vol", "trend_range", "hmm_gaussian", "cusum"]


@dataclass(frozen=True)
class FeatureConfig:
    windows: tuple[int, ...] = (20,)
    feature_shift: int = 1
    annualization: Literal["auto", "none"] = "auto"
    periods_per_year_override: int | None = None


@dataclass(frozen=True)
class HMMConfig:
    n_states: int = 2
    max_iter: int = 100
    tol: float = 1e-4
    min_var: float = 1e-6
    min_state_weight: float = 1e-6
    transition_epsilon: float = 1e-12
    random_seed: int = 42
    init_method: Literal["quantile", "kmeans_simple"] = "quantile"
    mode: Literal["filter", "smooth", "viterbi"] = "filter"
    warm_start: bool = False
    standardize_features: bool = False


@dataclass(frozen=True)
class CUSUMConfig:
    threshold: float = 0.02
    drift: float = 0.0
    feature: Literal["returns", "vol"] = "returns"


@dataclass(frozen=True)
class RegimeConfig:
    method: RegimeMethod = "quantile_vol"
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    cusum: CUSUMConfig = field(default_factory=CUSUMConfig)
    quantile_thresholds: tuple[float, ...] = (0.3, 0.7)
    trend_slope_quantile: float = 0.7
    trend_r2_threshold: float = 0.5
    hmm_feature_cols: tuple[str, ...] | None = None
    output_shift: int = 0
    calibration_start: pd.Timestamp | None = None
    calibration_end: pd.Timestamp | None = None


@dataclass(frozen=True)
class RegimeResult:
    regime: pd.Series
    proba: pd.DataFrame | None
    features: pd.DataFrame
    diagnostics: dict
    metadata: dict


def serialize_config(cfg: RegimeConfig) -> dict:
    data = asdict(cfg)
    return data


__all__ = [
    "CUSUMConfig",
    "FeatureConfig",
    "HMMConfig",
    "RegimeConfig",
    "RegimeMethod",
    "RegimeResult",
    "serialize_config",
]
