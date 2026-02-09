import math

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from quant_fx_system.quant.decay import DecayConfig, apply_decay, decay_position_target, generate_weights
from quant_fx_system.quant.decay.ewma import half_life_to_alpha


def _make_series(values, index, name=None):
    return pd.Series(values, index=index, name=name)


def test_index_invariants():
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    series = _make_series([0.0, 1.0, 2.0], index)
    result = apply_decay(series, DecayConfig(kind="none"))
    assert result.output.index.equals(index)

    naive_index = pd.date_range("2024-01-01", periods=3, freq="D")
    series_naive = _make_series([0.0, 1.0, 2.0], naive_index)
    with pytest.raises(ValueError):
        apply_decay(series_naive, DecayConfig(kind="none"))

    non_monotonic = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-01", "2024-01-03"], tz="UTC"
    )
    series_non_monotonic = _make_series([0.0, 1.0, 2.0], non_monotonic)
    with pytest.raises(ValueError):
        apply_decay(series_non_monotonic, DecayConfig(kind="none"))


def test_half_life_to_alpha():
    alpha = half_life_to_alpha(1.0)
    assert alpha == pytest.approx(1.0 - math.exp(-math.log(2.0)), rel=1e-6)

    alpha_short = half_life_to_alpha(2.0)
    alpha_long = half_life_to_alpha(10.0)
    assert alpha_long < alpha_short


def test_ewma_causal_and_shift():
    index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    series = _make_series([0.0, 0.0, 0.0, 10.0, 0.0], index)
    cfg = DecayConfig(kind="ewma", half_life_bars=2.0, min_periods=1)
    result = apply_decay(series, cfg)

    alpha = half_life_to_alpha(2.0)
    expected_pre_spike = series.iloc[:3].ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    assert result.output.loc[index[2]] == pytest.approx(expected_pre_spike.iloc[-1])

    shifted = apply_decay(series, DecayConfig(kind="ewma", half_life_bars=2.0, shift=1))
    assert shifted.output.loc[index[3]] == pytest.approx(result.output.loc[index[2]])


def test_kernel_weights_and_rolling_mean():
    weights = generate_weights("linear", window=5, normalize=True)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[-1] > weights[0]

    index = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    series = _make_series([1.0] * 8, index)
    cfg = DecayConfig(kind="linear", window=5, min_periods=5)
    result = apply_decay(series, cfg)
    assert np.allclose(result.output.dropna().to_numpy(), 1.0)


def test_time_aware_ewma():
    index_short = pd.DatetimeIndex(
        ["2024-01-01", "2024-01-02", "2024-01-03"], tz="UTC"
    )
    index_long = pd.DatetimeIndex(
        ["2024-01-01", "2024-01-11", "2024-01-21"], tz="UTC"
    )
    values = [0.0, 1.0, 1.0]
    series_short = _make_series(values, index_short)
    series_long = _make_series(values, index_long)

    cfg = DecayConfig(kind="ewma", half_life_time=pd.Timedelta(days=5))
    out_short = apply_decay(series_short, cfg).output
    out_long = apply_decay(series_long, cfg).output

    assert out_long.loc[index_long[1]] > out_short.loc[index_short[1]]

    out_repeat = apply_decay(series_short, cfg).output
    pdt.assert_series_equal(out_short, out_repeat)


def test_decay_position_target():
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    target = _make_series([0.0, 1.0, 1.0], index)
    cfg = DecayConfig(kind="ewma", half_life_bars=1.0)
    result = decay_position_target(target, cfg)

    alpha = half_life_to_alpha(1.0)
    k = 1.0 - alpha
    expected = [0.0, (1.0 - k) * 1.0 + k * 0.0, (1.0 - k) * 1.0 + k * ((1.0 - k))]
    expected_series = _make_series(expected, index)
    pdt.assert_series_equal(result, expected_series)
