import numpy as np
import pandas as pd
import pandas.testing as pdt

from quant_fx_system.quant.meta_model import MetaModelConfig, fit_meta_model


def _make_data():
    index = pd.date_range("2023-01-01", periods=25, freq="D", tz="UTC")
    returns = pd.Series(np.sin(np.linspace(0.0, 3.0, len(index))) * 0.01, index=index)
    base_signal = pd.Series(np.sign(np.cos(np.linspace(0.0, 4.0, len(index)))), index=index)
    base_signal = base_signal.replace(0.0, 1.0)
    features = pd.DataFrame(
        {
            "feat1": returns.rolling(2).mean(),
            "feat2": returns.shift(1),
        },
        index=index,
    )
    return returns, base_signal, features


def test_determinism_fit():
    returns, base_signal, features = _make_data()
    cfg = MetaModelConfig(
        version="1.0",
        horizon=1,
        calibration="platt",
        decision_policy="threshold",
        cv_scheme="purged_kfold",
        n_splits=3,
        min_train_size=10,
        random_seed=11,
    )

    fit_a = fit_meta_model(
        prices=None,
        returns=returns,
        base_signal=base_signal,
        features=features,
        regimes=None,
        cfg=cfg,
    )
    fit_b = fit_meta_model(
        prices=None,
        returns=returns,
        base_signal=base_signal,
        features=features,
        regimes=None,
        cfg=cfg,
    )

    pdt.assert_frame_equal(fit_a.oos_predictions, fit_b.oos_predictions)
    pdt.assert_frame_equal(fit_a.metrics, fit_b.metrics)
    assert fit_a.metadata["config_hash"] == fit_b.metadata["config_hash"]
