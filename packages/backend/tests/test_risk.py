import pandas as pd
import pandas.testing as pdt
import pytest

from quant_fx_system.quant.risk import RiskConfig, apply_risk_overlay
from quant_fx_system.quant.risk.drawdown import compute_equity_from_returns
from quant_fx_system.quant.risk.turnover import cap_position_delta
from quant_fx_system.quant.risk.types import DrawdownConfig, TurnoverConfig, VolTargetConfig


def _make_series(values, index, name=None):
    return pd.Series(values, index=index, name=name)


def test_index_invariants():
    index = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    position = _make_series([0.0, 0.5, -0.5, 0.0, 0.25], index)
    returns = _make_series([0.0, 0.01, -0.02, 0.01, 0.0], index, name="returns")

    result = apply_risk_overlay(position_raw=position, returns=returns, cfg=RiskConfig())

    assert result.position.index.equals(index)
    assert result.position.index.is_monotonic_increasing
    assert result.position.index.is_unique
    assert str(result.position.index.tz) == "UTC"

    naive_index = pd.date_range("2023-01-01", periods=5, freq="D")
    position_naive = _make_series([0.0] * 5, naive_index)
    returns_naive = _make_series([0.0] * 5, naive_index)

    with pytest.raises(ValueError):
        apply_risk_overlay(position_raw=position_naive, returns=returns_naive, cfg=RiskConfig())


def test_vol_targeting_scales_with_regime():
    index = pd.date_range("2023-01-01", periods=6, freq="D", tz="UTC")
    returns = _make_series([0.0, 0.10, -0.10, 0.01, -0.01, 0.0], index)
    position = _make_series([1.0] * 6, index)

    cfg = RiskConfig(
        max_leverage=1.0,
        vol_target=VolTargetConfig(window=2, target_vol=0.05, annualization="none"),
    )

    result = apply_risk_overlay(position_raw=position, returns=returns, cfg=cfg)

    scale = result.metrics["scale_vol"]
    assert scale.min() >= cfg.vol_target.min_scale
    assert scale.max() <= cfg.vol_target.max_scale

    high_vol_scale = scale.loc[index[3]]
    low_vol_scale = scale.loc[index[5]]
    assert high_vol_scale < low_vol_scale
    assert (result.position.abs() <= cfg.max_leverage).all()


def test_no_lookahead_vol_targeting():
    index = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    returns = _make_series([0.01, 0.01, 2.0, 0.01, 0.01], index)
    position = _make_series([1.0] * 5, index)

    cfg = RiskConfig(
        max_leverage=1.0,
        vol_target=VolTargetConfig(window=2, target_vol=0.1, annualization="none", shift=1),
    )

    result = apply_risk_overlay(position_raw=position, returns=returns, cfg=cfg)

    vol = returns.rolling(window=2).std(ddof=0)
    expected_scale_raw = cfg.vol_target.target_vol / vol
    expected_scale_raw = expected_scale_raw.clip(
        lower=cfg.vol_target.min_scale, upper=cfg.vol_target.max_scale
    )
    expected_scale = expected_scale_raw.shift(1).fillna(1.0)

    pdt.assert_series_equal(result.metrics["scale_vol"], expected_scale.rename("scale_vol"))
    assert result.metrics["scale_vol"].loc[index[2]] == expected_scale.loc[index[2]]
    assert result.metrics["scale_vol"].loc[index[3]] == expected_scale.loc[index[3]]


def test_drawdown_guard_flatten():
    index = pd.date_range("2023-01-01", periods=6, freq="D", tz="UTC")
    returns = _make_series([-0.1] * 6, index)
    position = _make_series([1.0] * 6, index)

    cfg = RiskConfig(
        max_leverage=1.0,
        drawdown=DrawdownConfig(enabled=True, max_drawdown=0.2, mode="flatten"),
        vol_target=VolTargetConfig(enabled=False),
        turnover=TurnoverConfig(enabled=False),
    )

    result = apply_risk_overlay(position_raw=position, returns=returns, cfg=cfg)

    dd_flag = result.metrics["dd_flag"] > 0
    if dd_flag.any():
        assert (result.position[dd_flag] == 0.0).all()


def test_turnover_cap():
    index = pd.date_range("2023-01-01", periods=4, freq="D", tz="UTC")
    position = _make_series([0.0, 2.0, -2.0, 2.0], index)
    capped = cap_position_delta(position, max_turnover_per_bar=0.5)
    turnover = capped.diff().abs().fillna(0.0)
    assert (turnover <= 0.5 + 1e-12).all()


def test_pipeline_metrics():
    index = pd.date_range("2023-01-01", periods=6, freq="D", tz="UTC")
    price = _make_series([1.0, 1.05, 1.02, 1.03, 1.04, 1.01], index)
    position = _make_series([0.0, 1.0, -1.0, 0.5, 0.0, 0.25], index)

    cfg = RiskConfig(max_leverage=1.0)
    result = apply_risk_overlay(position_raw=position, price=price, cfg=cfg)

    assert not result.position.isna().any()
    assert (result.position.abs() <= cfg.max_leverage).all()

    required_cols = {
        "returns",
        "vol_rolling",
        "scale_vol",
        "position_raw",
        "position_after_vol_target",
        "turnover_raw",
        "turnover_after",
        "equity_proxy",
        "drawdown",
        "dd_flag",
        "dd_guard_active",
        "position_final",
    }
    assert required_cols.issubset(set(result.metrics.columns))


def test_inf_rejected():
    index = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    position = _make_series([0.0, 1.0, 0.0], index)
    returns = _make_series([0.0, float("inf"), 0.0], index)

    with pytest.raises(ValueError):
        apply_risk_overlay(position_raw=position, returns=returns, cfg=RiskConfig())


def test_missing_returns_rejected():
    index = pd.date_range("2023-01-01", periods=4, freq="D", tz="UTC")
    position = _make_series([0.0, 1.0, 0.0, 1.0], index)
    returns = _make_series([0.0, float("nan"), 0.01, 0.0], index)

    with pytest.raises(ValueError):
        apply_risk_overlay(position_raw=position, returns=returns, cfg=RiskConfig())


def test_equity_clamped_at_zero():
    index = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    position = _make_series([2.0, 2.0, 2.0], index)
    returns = _make_series([0.0, -1.5, 0.0], index)

    equity = compute_equity_from_returns(returns, position, initial_equity=1.0)

    assert (equity >= 0.0).all()
    assert equity.iloc[1] == 0.0
    assert equity.iloc[2] == 0.0
