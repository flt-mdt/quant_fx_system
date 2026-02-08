import numpy as np
import pandas as pd

from quant_fx_system.quant.data_pipeline.clean_align import (
    deduplicate_and_validate_index,
    ensure_datetime_index_utc,
    resample_price,
)
from quant_fx_system.quant.data_pipeline.feature_engineering import (
    FeatureConfig,
    build_features,
    compute_log_returns,
)
from quant_fx_system.quant.data_pipeline.stationarity import stationarity_report


def test_ensure_datetime_index_utc_localizes_naive_to_utc() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"price": [1.0, 1.1, 1.2]}, index=idx)

    result = ensure_datetime_index_utc(df)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is not None
    assert str(result.index.tz) == "UTC"


def test_deduplicate_keeps_last() -> None:
    idx = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]).tz_localize("UTC")
    series = pd.Series([1.0, 2.0, 3.0], index=idx, name="price")

    cleaned = deduplicate_and_validate_index(series)

    assert cleaned.loc[pd.Timestamp("2024-01-01", tz="UTC")] == 2.0
    assert cleaned.shape[0] == 2


def test_resample_daily_last() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="6H", tz="UTC")
    series = pd.Series(np.arange(6, dtype=float), index=idx, name="price")

    resampled = resample_price(series, freq="1D", method="last")

    assert resampled.iloc[0] == 3.0
    assert resampled.iloc[1] == 5.0


def test_build_features_shifts_no_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    price = pd.Series(np.linspace(1.0, 1.9, 10), index=idx, name="price")

    cfg = FeatureConfig(momentum_windows=[2], vol_windows=[2], zscore_windows=[2], decision_shift=1)
    features = build_features(price, cfg=cfg)

    expected_ret = compute_log_returns(price).shift(1).loc[features.index]
    assert np.allclose(features["ret_1"].values, expected_ret.values, equal_nan=False)


def test_stationarity_report_runs() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    series = pd.Series(np.random.normal(size=50), index=idx, name="series")

    report = stationarity_report({"series": series})

    expected_cols = {
        "series",
        "adf_stat",
        "adf_pvalue",
        "adf_usedlag",
        "adf_nobs",
        "adf_crit_1pct",
        "adf_crit_5pct",
        "adf_crit_10pct",
        "kpss_stat",
        "kpss_pvalue",
        "kpss_lags",
        "kpss_crit_10pct",
        "kpss_crit_5pct",
        "kpss_crit_2_5pct",
        "kpss_crit_1pct",
    }
    assert expected_cols.issubset(set(report.columns))
    assert report.shape[0] == 1
