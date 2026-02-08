"""Core data cleaning and alignment utilities for FX time series."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanAlignConfig:
    """Configuration for cleaning/alignment of FX price series.

    Attributes:
        tz_assume: Timezone to assume when timestamps are naive.
        resample_freq: Frequency for resampling (e.g., "1D").
        price_cols_priority: Ordered list of preferred price column names.
        outlier_max_abs_log_return: Threshold for filtering outlier log returns.
        enable_outlier_filter: Toggle to enable/disable outlier filtering.
    """

    tz_assume: str = "UTC"
    resample_freq: str = "1D"
    price_cols_priority: list[str] = field(
        default_factory=lambda: ["mid", "px_last", "price", "close"]
    )
    outlier_max_abs_log_return: float = 0.10
    enable_outlier_filter: bool = True


def _ensure_datetime_index(index: pd.Index, *, tz_assume: str) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        dt_index = index
    else:
        dt_index = pd.to_datetime(index, errors="raise")

    if dt_index.tz is None:
        dt_index = dt_index.tz_localize(tz_assume, ambiguous="raise", nonexistent="raise")

    dt_index = dt_index.tz_convert("UTC")

    if dt_index.isna().any():
        raise ValueError("Datetime index contains NaT values after parsing.")

    return pd.DatetimeIndex(dt_index, name=dt_index.name)


def ensure_datetime_index_utc(
    df: pd.DataFrame, *, timestamp_col: str | None = None, tz_assume: str = "UTC"
) -> pd.DataFrame:
    """Ensure a UTC-aware DatetimeIndex.

    If the input has a DatetimeIndex, it is localized or converted to UTC.
    Otherwise, `timestamp_col` is required to create the index.
    """

    data = df.copy()
    if isinstance(data.index, pd.DatetimeIndex):
        dt_index = _ensure_datetime_index(data.index, tz_assume=tz_assume)
    else:
        if timestamp_col is None:
            raise ValueError("timestamp_col is required when no DatetimeIndex is present.")
        if timestamp_col not in data.columns:
            raise ValueError(f"timestamp_col '{timestamp_col}' not found in columns.")
        dt_index = _ensure_datetime_index(data[timestamp_col], tz_assume=tz_assume)
        data = data.drop(columns=[timestamp_col])

    data.index = dt_index.tz_convert("UTC")
    data = data.sort_index()
    return data


def _available_columns(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def choose_price_series(
    df: pd.DataFrame,
    *,
    price_cols_priority: list[str],
    bid_col: str = "bid",
    ask_col: str = "ask",
) -> pd.Series:
    """Select or compute a mid price series from input data."""

    if bid_col in df.columns and ask_col in df.columns:
        price = (df[bid_col] + df[ask_col]) / 2.0
        return price.astype(float).rename("price")

    available = _available_columns(df, price_cols_priority)
    if not available:
        raise ValueError(
            "No valid price columns found. Provide bid/ask or one of: "
            f"{price_cols_priority}."
        )

    price = df[available[0]].astype(float)
    return price.rename("price")


def deduplicate_and_validate_index(series: pd.Series) -> pd.Series:
    """Drop duplicate timestamps and validate index ordering."""

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex.")

    if series.index.tz is None:
        raise ValueError("Series index must be timezone-aware.")

    cleaned = series[~series.index.duplicated(keep="last")]
    cleaned = cleaned.sort_index()

    cleaned.index = cleaned.index.tz_convert("UTC")

    if cleaned.empty:
        raise ValueError("Series is empty after deduplication.")

    if not cleaned.index.is_monotonic_increasing or not cleaned.index.is_unique:
        raise ValueError("Series index must be strictly increasing without duplicates.")

    return cleaned


def resample_price(price: pd.Series, *, freq: str = "1D", method: str = "last") -> pd.Series:
    """Resample price series to a target frequency using the given method."""

    if method == "last":
        resampled = price.resample(freq).last()
    elif method == "first":
        resampled = price.resample(freq).first()
    elif method == "mean":
        resampled = price.resample(freq).mean()
    else:
        raise ValueError("Unsupported resample method. Use 'last', 'first', or 'mean'.")

    resampled = resampled.dropna()
    return resampled.rename("price")


def filter_outliers_by_returns(
    price: pd.Series, *, max_abs_log_return: float = 0.10
) -> pd.Series:
    """Filter out price points with extreme log returns."""

    if max_abs_log_return <= 0:
        raise ValueError("max_abs_log_return must be positive.")

    cleaned_price = price.replace([0, -0], np.nan)
    cleaned_price = cleaned_price.where(cleaned_price > 0)
    cleaned_price = cleaned_price.dropna()
    if cleaned_price.empty:
        raise ValueError("Price series is empty after removing non-positive values.")

    log_returns = np.log(cleaned_price).diff()
    mask = log_returns.abs() <= max_abs_log_return
    mask = mask | log_returns.isna()
    cleaned = cleaned_price[mask]
    return cleaned.rename("price")


def clean_price_pipeline(
    raw: pd.DataFrame | pd.Series,
    *,
    cfg: CleanAlignConfig,
    timestamp_col: str | None = None,
) -> pd.Series:
    """Run the full cleaning pipeline on raw price inputs."""

    if isinstance(raw, pd.Series):
        df = raw.to_frame(name=raw.name or "price")
    elif isinstance(raw, pd.DataFrame):
        df = raw.copy()
    else:
        raise ValueError("raw must be a pandas Series or DataFrame.")

    df = ensure_datetime_index_utc(df, timestamp_col=timestamp_col, tz_assume=cfg.tz_assume)

    price = choose_price_series(df, price_cols_priority=cfg.price_cols_priority)
    price = deduplicate_and_validate_index(price)

    if cfg.enable_outlier_filter:
        price = filter_outliers_by_returns(
            price, max_abs_log_return=cfg.outlier_max_abs_log_return
        )

    price = resample_price(price, freq=cfg.resample_freq, method="last")
    return price.rename("price_clean")
