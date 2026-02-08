"""Tests for signal module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fx_system.quant.signals.ensemble import EnsembleConfig, EnsembleSignal
from quant_fx_system.quant.signals.mean_reversion import (
    MeanReversionZScoreConfig,
    MeanReversionZScoreSignal,
)
from quant_fx_system.quant.signals.momentum import (
    MomentumZScoreConfig,
    MomentumZScoreSignal,
)
from quant_fx_system.quant.signals.validation import validate_features_for_signals


def _make_features(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=rows, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    z_mom = pd.Series(np.linspace(-2.5, 2.5, rows), index=index)
    ret_1 = pd.Series(rng.normal(scale=0.01, size=rows), index=index)
    features = pd.DataFrame({"z_mom_20": z_mom, "ret_1": ret_1}, index=index)
    features.attrs["decision_shift"] = 1
    return features


def test_validate_features_requires_decision_shift() -> None:
    features = _make_features()
    features.attrs.pop("decision_shift")
    try:
        validate_features_for_signals(features)
    except ValueError as exc:
        assert "decision_shift" in str(exc)
    else:
        raise AssertionError("Expected validation to fail without decision_shift.")


def test_validate_features_casts_decision_shift_string() -> None:
    features = _make_features()
    features.attrs["decision_shift"] = "1"
    validate_features_for_signals(features)
    assert features.attrs["decision_shift"] == 1


def test_momentum_signal_output_bounds() -> None:
    features = _make_features()
    config = MomentumZScoreConfig(window=20, max_leverage=1.5, method="tanh", k=1.2)
    signal = MomentumZScoreSignal(config)
    result = signal.run(features)
    assert result.position.max() <= config.max_leverage
    assert result.position.min() >= -config.max_leverage


def test_mean_reversion_is_negative_of_momentum() -> None:
    features = _make_features()
    mom = MomentumZScoreSignal(MomentumZScoreConfig(window=20))
    mean_rev = MeanReversionZScoreSignal(MeanReversionZScoreConfig(window=20))
    mom_alpha = mom.compute_alpha(features).dropna()
    mean_alpha = mean_rev.compute_alpha(features).dropna()
    aligned = mom_alpha.align(mean_alpha, join="inner")
    assert np.allclose(aligned[0].values, -aligned[1].values, equal_nan=True)


def test_ensemble_combines_alphas_and_bounds_positions() -> None:
    features = _make_features()
    mom = MomentumZScoreSignal(MomentumZScoreConfig(window=20))
    mean_rev = MeanReversionZScoreSignal(MeanReversionZScoreConfig(window=20))
    config = EnsembleConfig(
        weights={mom.name: 0.6, mean_rev.name: 0.4},
        max_leverage=1.0,
        normalize_weights=True,
        combine="sum",
        post_transform="tanh",
    )
    ensemble = EnsembleSignal([mom, mean_rev], config)
    result = ensemble.run(features)
    assert result.position.isna().sum() == 0
    assert result.position.max() <= config.max_leverage
    assert result.position.min() >= -config.max_leverage


def test_outputs_have_utc_monotonic_unique_index() -> None:
    features = _make_features()
    signal = MomentumZScoreSignal(MomentumZScoreConfig(window=20))
    result = signal.run(features)
    assert str(result.alpha.index.tz) == "UTC"
    assert result.alpha.index.is_monotonic_increasing
    assert result.alpha.index.is_unique
    assert str(result.position.index.tz) == "UTC"
    assert result.position.index.is_monotonic_increasing
    assert result.position.index.is_unique


def test_run_raises_when_alpha_empty() -> None:
    features = _make_features(rows=10)
    signal = MomentumZScoreSignal(MomentumZScoreConfig(window=20))
    try:
        signal.run(features)
    except ValueError as exc:
        assert "alpha is empty" in str(exc)
    else:
        raise AssertionError("Expected run to fail when alpha is empty.")
