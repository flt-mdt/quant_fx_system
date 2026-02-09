import pandas as pd
import pytest

from quant_fx_system.quant.evaluation import EvaluationConfig, evaluate_strategy


def _make_series(values, index, name=None):
    return pd.Series(values, index=index, name=name)


def test_index_invariants():
    index = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    returns = _make_series([0.0, 0.01, -0.02, 0.01, 0.0], index)

    result = evaluate_strategy(returns=returns, cfg=EvaluationConfig(annualization="none"))

    assert result.series["equity"].index.equals(index)
    assert result.series["equity"].index.is_monotonic_increasing
    assert result.series["equity"].index.is_unique
    assert str(result.series["equity"].index.tz) == "UTC"

    naive_index = pd.date_range("2023-01-01", periods=5, freq="D")
    returns_naive = _make_series([0.0] * 5, naive_index)

    with pytest.raises(ValueError):
        evaluate_strategy(returns=returns_naive)


def test_sharpe_and_vol_constant_returns():
    index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    returns = _make_series([0.01] * 5, index)

    result = evaluate_strategy(returns=returns, cfg=EvaluationConfig(annualization="none"))

    assert result.summary["vol_annual"] == 0.0
    assert result.summary["sharpe"] == 0.0


def test_max_drawdown():
    index = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    returns = _make_series([0.1, -0.2, 0.05], index)

    result = evaluate_strategy(returns=returns, cfg=EvaluationConfig(annualization="none"))

    assert pytest.approx(result.summary["max_drawdown"], rel=1e-6) == 0.2


def test_drawdown_rejects_non_positive_equity():
    index = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    returns = _make_series([0.1, -1.0, 0.05], index)

    with pytest.raises(ValueError):
        evaluate_strategy(returns=returns, cfg=EvaluationConfig(annualization="none"))


def test_rolling_metrics_window_two():
    index = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    returns = _make_series([0.0, 0.1, -0.1], index)

    cfg = EvaluationConfig(annualization="none", rolling_windows=(2,))
    result = evaluate_strategy(returns=returns, cfg=cfg)

    rolling = result.tables["rolling_metrics"]
    assert pytest.approx(rolling.loc[index[2], "mean_roll_2"], rel=1e-6) == 0.0
    assert pytest.approx(rolling.loc[index[2], "vol_roll_2"], rel=1e-6) == 0.1
    assert pytest.approx(rolling.loc[index[2], "sr_roll_2"], rel=1e-6) == 0.0


def test_benchmark_regression():
    index = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    benchmark = _make_series([0.01, -0.02, 0.03, -0.01, 0.02], index)
    returns = benchmark * 2.0

    cfg = EvaluationConfig(annualization="none")
    result = evaluate_strategy(returns=returns, benchmark_returns=benchmark, cfg=cfg)

    assert pytest.approx(result.summary["beta"], rel=1e-6) == 2.0
    assert pytest.approx(result.summary["alpha"], abs=1e-6) == 0.0


def test_psr_dsr_sanity():
    index = pd.date_range("2023-01-01", periods=20, freq="D", tz="UTC")
    noisy = _make_series([0.01, -0.01] * 10, index)
    strong = _make_series([0.02, 0.01] * 10, index)

    cfg = EvaluationConfig(annualization="none", n_trials=10)
    result_noisy = evaluate_strategy(returns=noisy, cfg=cfg)
    result_strong = evaluate_strategy(returns=strong, cfg=cfg)

    assert 0.3 <= result_noisy.summary["psr_0"] <= 0.7
    assert result_strong.summary["psr_0"] > 0.9
    assert result_strong.summary["dsr_0"] > 0.5
