from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fx_system.quant.regimes import FeatureConfig, RegimeConfig, infer_regimes
from quant_fx_system.quant.regimes.validation import infer_periods


def test_infer_periods_accepts_feature_config_directly() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    cfg = FeatureConfig(windows=(5,), annualization="auto")
    value = infer_periods(cfg, idx)
    assert value is None or value > 0


def test_regimes_quantile_vol_smoke_stable() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
    returns = pd.Series(np.random.default_rng(11).normal(0, 0.01, len(idx)), index=idx)
    cfg = RegimeConfig(method="quantile_vol", feature=FeatureConfig(windows=(10,)))
    result = infer_regimes(returns=returns, cfg=cfg)
    assert len(result.features) == len(returns)
    assert result.metadata["periods_per_year"] is None or result.metadata["periods_per_year"] > 0
