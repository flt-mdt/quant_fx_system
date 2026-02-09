from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd

from .types import InformationConfig


def ensure_utc_index(obj: pd.Series | pd.DataFrame, name: str) -> None:
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a DatetimeIndex")
    if obj.index.tz is None:
        raise ValueError(f"{name} index must be tz-aware UTC")
    if str(obj.index.tz) != "UTC":
        raise ValueError(f"{name} index must be UTC")
    if not obj.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonic increasing")
    if obj.index.has_duplicates:
        raise ValueError(f"{name} index must not contain duplicates")


def align_inputs(
    returns: pd.Series,
    signal: pd.Series | None,
    features: pd.DataFrame,
    regimes: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series | None, pd.DataFrame, pd.Series | None]:
    ensure_utc_index(returns, "returns")
    ensure_utc_index(features, "features")
    if signal is not None:
        ensure_utc_index(signal, "signal")
    if regimes is not None:
        ensure_utc_index(regimes, "regimes")

    indices: list[Iterable[pd.Timestamp]] = [returns.index, features.index]
    if signal is not None:
        indices.append(signal.index)
    if regimes is not None:
        indices.append(regimes.index)

    common_index = indices[0]
    for idx in indices[1:]:
        common_index = common_index.intersection(idx)

    returns_aligned = returns.loc[common_index]
    signal_aligned = signal.loc[common_index] if signal is not None else None
    features_aligned = features.loc[common_index]
    regimes_aligned = regimes.loc[common_index] if regimes is not None else None

    return returns_aligned, signal_aligned, features_aligned, regimes_aligned


def shift_features(features: pd.DataFrame, cfg: InformationConfig) -> pd.DataFrame:
    if cfg.feature_shift == 0:
        return features
    return features.shift(cfg.feature_shift)


def check_no_future_usage(
    returns: pd.Series,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict:
    warnings: list[str] = []
    if target.index.max() > returns.index.max():
        warnings.append("target_uses_future_index")
    if features.index.max() > returns.index.max():
        warnings.append("features_extend_beyond_returns")
    return {"warnings": warnings}


def schema(features: pd.DataFrame, cfg: InformationConfig) -> dict:
    columns = tuple(features.columns)
    dtypes = tuple(str(dt) for dt in features.dtypes)
    payload = "|".join([*columns, *dtypes, str(cfg.to_dict())])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return {
        "columns": list(columns),
        "dtypes": list(dtypes),
        "config_hash": digest,
    }
