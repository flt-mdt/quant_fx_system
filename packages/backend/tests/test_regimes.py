import numpy as np
import pandas as pd
import pytest

from quant_fx_system.quant.regimes import (
    FeatureConfig,
    HMMConfig,
    RegimeConfig,
    cusum_flags,
    infer_regimes,
    regime_durations,
    transition_matrix,
    validate_utc_series,
)


def _make_series(values, tz="UTC"):
    index = pd.date_range("2020-01-01", periods=len(values), tz=tz)
    return pd.Series(values, index=index)


def test_validate_utc_series_invariants():
    series = _make_series([1.0, 2.0, 3.0], tz=None)
    with pytest.raises(ValueError):
        validate_utc_series(series, "price")

    series = _make_series([1.0, 2.0, 3.0])
    series = series.iloc[::-1]
    with pytest.raises(ValueError):
        validate_utc_series(series, "price")

    series = _make_series([1.0, 2.0, 3.0])
    series.index = series.index.where(series.index != series.index[1], series.index[0])
    with pytest.raises(ValueError):
        validate_utc_series(series, "price")


def test_quantile_vol_constant_series():
    returns = _make_series(np.zeros(120))
    cfg = RegimeConfig(method="quantile_vol", feature=FeatureConfig(windows=(10,)))
    result = infer_regimes(returns=returns, cfg=cfg)
    unique = result.regime.dropna().unique()
    assert len(unique) == 1


def test_trend_range_trending_series():
    price = _make_series(np.linspace(1.0, 2.0, 200))
    cfg = RegimeConfig(
        method="trend_range",
        feature=FeatureConfig(windows=(20,)),
        trend_slope_quantile=0.6,
        trend_r2_threshold=0.3,
    )
    result = infer_regimes(price=price, cfg=cfg)
    trend_share = (result.regime == 1).mean()
    assert trend_share > 0.6


def test_hmm_probabilities_sum_to_one_and_deterministic():
    rng = np.random.default_rng(123)
    returns = _make_series(rng.normal(0, 0.01, 300))
    cfg = RegimeConfig(
        method="hmm_gaussian",
        feature=FeatureConfig(windows=(10,)),
        hmm=HMMConfig(n_states=2, random_seed=7, max_iter=50, tol=1e-3),
    )
    result_a = infer_regimes(returns=returns, cfg=cfg)
    result_b = infer_regimes(returns=returns, cfg=cfg)
    assert np.allclose(result_a.proba.sum(axis=1).dropna().to_numpy(), 1.0, atol=1e-6)
    pd.testing.assert_series_equal(result_a.regime, result_b.regime)


def test_hmm_handles_nan_rows():
    returns = _make_series(np.random.default_rng(1).normal(0, 0.01, 80))
    returns.iloc[10:12] = np.nan
    cfg = RegimeConfig(
        method="hmm_gaussian",
        feature=FeatureConfig(windows=(5,)),
        hmm=HMMConfig(n_states=2, random_seed=5, max_iter=20, tol=1e-3),
    )
    result = infer_regimes(returns=returns, cfg=cfg)
    assert "dropped_nan_rows" in result.metadata["warnings"]
    assert result.regime.isna().sum() > 0


def test_hmm_filter_no_lookahead():
    base = np.zeros(120)
    returns_a = _make_series(base.copy())
    returns_b = _make_series(base.copy())
    returns_b.iloc[-1] = 0.2

    cfg = RegimeConfig(
        method="hmm_gaussian",
        feature=FeatureConfig(windows=(5,)),
        hmm=HMMConfig(n_states=2, random_seed=42, max_iter=30, tol=1e-3, mode="filter"),
    )
    result_a = infer_regimes(returns=returns_a, cfg=cfg)
    result_b = infer_regimes(returns=returns_b, cfg=cfg)

    common = result_a.proba.iloc[:-1].dropna()
    compare = result_b.proba.iloc[:-1].dropna()
    np.testing.assert_allclose(common.to_numpy(), compare.to_numpy(), atol=1e-6)


def test_transitions_and_durations():
    regime = pd.Series([0, 0, 1, 1, 1, 0], index=pd.date_range("2020-01-01", periods=6, tz="UTC"))
    matrix = transition_matrix(regime, n_states=2)
    durations = regime_durations(regime)
    assert matrix.shape == (2, 2)
    assert (durations > 0).all()


def test_cusum_flags():
    series = _make_series([0.0, 0.01, 0.03, -0.04, 0.0])
    flags = cusum_flags(series, threshold=0.02, drift=0.0)
    assert flags.sum() >= 1
