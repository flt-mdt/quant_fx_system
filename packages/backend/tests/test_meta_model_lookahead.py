import numpy as np
import pandas as pd
import pandas.testing as pdt

from quant_fx_system.quant.meta_model import MetaModelConfig, fit_meta_model


def _make_data():
    index = pd.date_range("2023-01-01", periods=30, freq="D", tz="UTC")
    returns = pd.Series(np.linspace(-0.01, 0.02, len(index)), index=index)
    base_signal = pd.Series(np.sign(np.sin(np.linspace(0.0, 3.0, len(index)))), index=index)
    base_signal = base_signal.replace(0.0, 1.0)
    features = pd.DataFrame(
        {
            "feat1": returns.rolling(3).mean(),
            "feat2": returns.shift(1),
        },
        index=index,
    )
    return returns, base_signal, features


def test_anti_lookahead_predictions_stable():
    returns, base_signal, features = _make_data()
    cfg = MetaModelConfig(
        version="1.0",
        horizon=2,
        cv_scheme="walk_forward",
        train_window=10,
        test_window=5,
        step_size=5,
        min_train_size=10,
        calibration="none",
        decision_policy="threshold",
    )

    fit_a = fit_meta_model(
        prices=None,
        returns=returns,
        base_signal=base_signal,
        features=features,
        regimes=None,
        cfg=cfg,
    )

    modified_returns = returns.copy()
    modified_returns.iloc[-1] = modified_returns.iloc[-1] + 0.5

    fit_b = fit_meta_model(
        prices=None,
        returns=modified_returns,
        base_signal=base_signal,
        features=features,
        regimes=None,
        cfg=cfg,
    )

    cutoff = returns.index[-cfg.test_window]
    pdt.assert_series_equal(
        fit_a.oos_predictions.loc[:cutoff, "p_follow"],
        fit_b.oos_predictions.loc[:cutoff, "p_follow"],
    )
