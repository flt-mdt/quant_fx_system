from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .types import MetaModelConfig


def ensure_utc_index(series: pd.Series | pd.DataFrame, name: str) -> None:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a DatetimeIndex")
    if series.index.tz is None or str(series.index.tz) != "UTC":
        raise ValueError(f"{name} must be tz-aware UTC")


def align_inputs(
    *,
    returns: pd.Series,
    base_signal: pd.Series,
    features: pd.DataFrame,
    regimes: pd.Series | None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series | None]:
    ensure_utc_index(returns, "returns")
    ensure_utc_index(base_signal, "base_signal")
    ensure_utc_index(features, "features")
    if regimes is not None:
        ensure_utc_index(regimes, "regimes")

    common = returns.index.intersection(base_signal.index).intersection(features.index)
    if regimes is not None:
        common = common.intersection(regimes.index)

    returns = returns.loc[common]
    base_signal = base_signal.loc[common]
    features = features.loc[common]
    regimes = regimes.loc[common] if regimes is not None else None
    return returns, base_signal, features, regimes


def align_predict_inputs(
    *,
    base_signal: pd.Series,
    features: pd.DataFrame,
    regimes: pd.Series | None,
) -> tuple[pd.Series, pd.DataFrame, pd.Series | None]:
    ensure_utc_index(base_signal, "base_signal")
    ensure_utc_index(features, "features")
    if regimes is not None:
        ensure_utc_index(regimes, "regimes")

    common = base_signal.index.intersection(features.index)
    if regimes is not None:
        common = common.intersection(regimes.index)

    base_signal = base_signal.loc[common]
    features = features.loc[common]
    regimes = regimes.loc[common] if regimes is not None else None
    return base_signal, features, regimes


def shift_features(features: pd.DataFrame, cfg: MetaModelConfig) -> pd.DataFrame:
    return features.shift(cfg.feature_shift)


def feature_schema(features: pd.DataFrame, cfg: MetaModelConfig) -> dict[str, str]:
    columns = sorted(features.columns)
    return {
        "columns": columns,
        "dtypes": {col: str(features[col].dtype) for col in columns},
        "config": asdict(cfg),
    }
