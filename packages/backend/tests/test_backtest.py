import pandas as pd
import pandas.testing as pdt
import pytest

from quant_fx_system.quant.backtest import BacktestConfig, run_backtest


def _make_price(values, index):
    return pd.Series(values, index=index)


def _make_position(values, index):
    return pd.Series(values, index=index)


def test_no_lookahead_execution():
    index = pd.date_range("2023-01-01", periods=4, freq="D", tz="UTC")
    price = _make_price([1.0, 1.0, 2.0, 2.0], index)
    position = _make_position([0.0, 0.0, 1.0, 0.0], index)

    result = run_backtest(price, position, BacktestConfig())

    spike_time = index[2]
    assert result.pnl.loc[spike_time] == 0.0
    expected_pnl = result.position * result.returns
    pdt.assert_series_equal(result.pnl, expected_pnl)


def test_alignment_on_intersection():
    price_index = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    position_index = pd.date_range("2023-01-02", periods=4, freq="D", tz="UTC")
    price = _make_price([1.0, 1.1, 1.2, 1.3, 1.4], price_index)
    position = _make_position([0.0, 0.5, -0.5, 0.0], position_index)

    result = run_backtest(price, position, BacktestConfig())

    common = price_index.intersection(position_index)
    expected_index = common[1:]
    assert result.returns.index.equals(expected_index)
    assert result.returns.index.tz is not None
    assert str(result.returns.index.tz) == "UTC"


def test_costs_and_turnover():
    index = pd.date_range("2023-01-01", periods=4, freq="D", tz="UTC")
    price = _make_price([1.0, 1.0, 1.0, 1.0], index)
    position = _make_position([0.0, 1.0, 1.0, -1.0], index)

    cfg = BacktestConfig(transaction_cost_bps=10.0)
    result = run_backtest(price, position, cfg)

    expected_turnover = pd.Series([1.0, 0.0, 2.0], index=index[1:])
    expected_costs = expected_turnover * (10.0 / 1e4)

    pdt.assert_series_equal(result.turnover, expected_turnover)
    pdt.assert_series_equal(result.costs, expected_costs)


def test_position_bounds_enforced():
    index = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    price = _make_price([1.0, 1.0, 1.0], index)
    position = _make_position([0.0, 2.0, 0.0], index)

    with pytest.raises(ValueError):
        run_backtest(price, position, BacktestConfig(max_leverage=1.0))


def test_determinism():
    index = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    price = _make_price([1.0, 1.1, 1.05, 1.2, 1.25], index)
    position = _make_position([0.0, 0.5, -0.5, 0.25, 0.0], index)
    cfg = BacktestConfig(transaction_cost_bps=5.0)

    result_a = run_backtest(price, position, cfg)
    result_b = run_backtest(price, position, cfg)

    pdt.assert_series_equal(result_a.returns, result_b.returns)
    pdt.assert_series_equal(result_a.position, result_b.position)
    pdt.assert_series_equal(result_a.pnl, result_b.pnl)
    pdt.assert_series_equal(result_a.equity, result_b.equity)
    pdt.assert_series_equal(result_a.turnover, result_b.turnover)
    pdt.assert_series_equal(result_a.costs, result_b.costs)
