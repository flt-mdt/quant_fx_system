from __future__ import annotations

import numpy as np
import pandas as pd

from .types import BacktestConfig


def validate_config(cfg: BacktestConfig) -> None:
    if cfg.initial_equity <= 0:
        raise ValueError("initial_equity must be positive")
    if cfg.transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    if cfg.slippage_bps < 0:
        raise ValueError("slippage_bps must be non-negative")
    if cfg.max_leverage <= 0:
        raise ValueError("max_leverage must be positive")


def _validate_datetime_index(index: pd.Index, label: str) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{label} must have a DatetimeIndex")
    if index.tz is None or str(index.tz) != "UTC":
        raise ValueError(f"{label} index must be timezone-aware UTC")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{label} index must be monotone increasing")
    if not index.is_unique:
        raise ValueError(f"{label} index must be unique")


def validate_price_series(price: pd.Series) -> None:
    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pd.Series")
    _validate_datetime_index(price.index, "price")
    if not price.notna().all():
        raise ValueError("price contains NaN values")
    if not (price > 0).all():
        raise ValueError("price values must be positive")


def validate_position_series(position: pd.Series, max_leverage: float) -> None:
    if not isinstance(position, pd.Series):
        raise TypeError("position must be a pd.Series")
    _validate_datetime_index(position.index, "position")
    if not position.notna().all():
        raise ValueError("position contains NaN values")
    numeric_position = pd.to_numeric(position, errors="coerce")
    if numeric_position.isna().any():
        raise ValueError("position must be numeric")
    if not np.isfinite(numeric_position.to_numpy()).all():
        raise ValueError("position contains non-finite values")
    if not (numeric_position.abs() <= max_leverage).all():
        raise ValueError("position exceeds max_leverage")


def align_price_position(
    price: pd.Series,
    position: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    common_index = price.index.intersection(position.index)
    if common_index.empty:
        raise ValueError("price and position indices have no overlap")
    price_aligned = price.reindex(common_index).sort_index()
    position_aligned = position.reindex(common_index).sort_index()
    return price_aligned, position_aligned


def validate_no_lookahead_alignment(
    position: pd.Series,
    position_applied: pd.Series,
    returns_index: pd.Index,
) -> None:
    expected = position.shift(1).reindex(returns_index)
    if not expected.equals(position_applied):
        raise ValueError("position_applied must be position shifted by one period")
