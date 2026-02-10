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
    expected_pnl.name = "pnl"
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

    expected_turnover = pd.Series([0.0, 1.0, 0.0], index=index[1:])
    expected_turnover.name = "turnover"
    expected_costs = expected_turnover * (10.0 / 1e4)
    expected_costs.name = "costs"

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


def test_costs_aligned_with_applied_position():
    index = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    price = _make_price([1.0, 1.0, 1.0], index)
    position = _make_position([0.0, 1.0, 1.0], index)

    cfg = BacktestConfig(transaction_cost_bps=10.0)
    result = run_backtest(price, position, cfg)

    expected_costs = pd.Series([0.0, 10.0 / 1e4], index=index[1:])
    expected_costs.name = "costs"
    expected_pnl = expected_costs * -1.0
    expected_pnl.name = "pnl"

    pdt.assert_series_equal(result.costs, expected_costs)
    pdt.assert_series_equal(result.pnl, expected_pnl)


def test_invariant_flip_produces_turnover_and_costs_when_bps_positive():
    index = pd.date_range("2023-01-01", periods=6, freq="D", tz="UTC")
    price = _make_price([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index)
    position = _make_position([0.0, 1.0, -1.0, 1.0, -1.0, -1.0], index)

    result = run_backtest(price, position, BacktestConfig(transaction_cost_bps=5.0))

    assert (result.turnover > 0).any()
    assert (result.costs > 0).any()


def test_invariant_constant_position_zero_turnover_and_zero_costs():
    index = pd.date_range("2023-01-01", periods=8, freq="D", tz="UTC")
    price = _make_price([1.0, 1.1, 1.2, 1.1, 1.0, 1.05, 1.1, 1.2], index)
    position = _make_position([0.5] * len(index), index)

    result = run_backtest(price, position, BacktestConfig(transaction_cost_bps=10.0, slippage_bps=2.0))

    assert (result.turnover == 0.0).all()
    assert (result.costs == 0.0).all()


def test_invariant_zero_position_means_zero_pnl_without_financing():
    index = pd.date_range("2023-01-01", periods=7, freq="D", tz="UTC")
    price = _make_price([1.0, 1.1, 1.2, 1.1, 1.3, 1.25, 1.24], index)
    position = _make_position([0.0] * len(index), index)

    result = run_backtest(price, position, BacktestConfig(transaction_cost_bps=25.0, slippage_bps=10.0))

    assert (result.pnl == 0.0).all()
    assert (result.turnover == 0.0).all()
    assert (result.costs == 0.0).all()


def test_invariant_no_overlap_raises_error():
    price_index = pd.date_range("2023-01-01", periods=4, freq="D", tz="UTC")
    position_index = pd.date_range("2023-02-01", periods=4, freq="D", tz="UTC")
    price = _make_price([1.0, 1.01, 1.02, 1.03], price_index)
    position = _make_position([0.0, 0.0, 1.0, 1.0], position_index)

    with pytest.raises(ValueError, match="no overlap"):
        run_backtest(price, position, BacktestConfig())


def test_invariant_next_bar_micro_example():
    index = pd.date_range("2023-01-01", periods=4, freq="D", tz="UTC")
    price = _make_price([100.0, 110.0, 121.0, 121.0], index)
    position = _make_position([0.0, 1.0, 1.0, 0.0], index)

    result = run_backtest(price, position, BacktestConfig())

    expected_returns = pd.Series([0.10, 0.10, 0.0], index=index[1:], name="returns")
    expected_position = pd.Series([0.0, 1.0, 1.0], index=index[1:], name="position")
    expected_pnl = pd.Series([0.0, 0.10, 0.0], index=index[1:], name="pnl")
    pdt.assert_series_equal(result.returns, expected_returns)
    pdt.assert_series_equal(result.position, expected_position)
    pdt.assert_series_equal(result.pnl, expected_pnl)


def test_invariant_position_scaling_linearity_without_costs():
    index = pd.date_range("2023-01-01", periods=8, freq="D", tz="UTC")
    price = _make_price([1.0, 1.02, 1.03, 1.01, 1.0, 1.01, 1.05, 1.02], index)
    base_position = _make_position([0.0, 0.25, -0.25, 0.1, -0.1, 0.3, -0.2, 0.0], index)
    k = 2.0

    base = run_backtest(price, base_position, BacktestConfig(max_leverage=2.0))
    scaled = run_backtest(price, base_position * k, BacktestConfig(max_leverage=2.0))

    pdt.assert_series_equal(scaled.pnl, base.pnl * k)


def test_invariant_cost_units_match_turnover_times_bps():
    index = pd.date_range("2023-01-01", periods=6, freq="D", tz="UTC")
    price = _make_price([1.0] * len(index), index)
    position = _make_position([0.0, 1.0, -1.0, 0.0, 0.5, 0.5], index)
    cfg = BacktestConfig(transaction_cost_bps=8.0, slippage_bps=2.0)

    result = run_backtest(price, position, cfg)
    expected_costs = result.turnover * ((cfg.transaction_cost_bps + cfg.slippage_bps) / 1e4)
    pdt.assert_series_equal(result.costs, expected_costs.rename("costs"))


def test_invariant_equity_compounds_from_return_pnl():
    index = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    price = _make_price([100.0, 110.0, 121.0, 108.9, 108.9], index)
    position = _make_position([0.0, 1.0, 1.0, 1.0, 1.0], index)
    initial_equity = 1000.0

    result = run_backtest(price, position, BacktestConfig(initial_equity=initial_equity))
    expected_equity = initial_equity * (1.0 + result.pnl).cumprod()
    pdt.assert_series_equal(result.equity, expected_equity.rename("equity"))


def test_invariant_one_bar_shift_changes_performance_materially():
    index = pd.date_range("2023-01-01", periods=200, freq="D", tz="UTC")
    price = pd.Series(100.0 + pd.Series(range(len(index)), dtype=float).values, index=index)
    position = _make_position([0.0] + [1.0] * (len(index) - 1), index)

    baseline = run_backtest(price, position, BacktestConfig())
    delayed = run_backtest(price, position.shift(1).fillna(0.0), BacktestConfig())

    assert baseline.pnl.mean() > delayed.pnl.mean()
