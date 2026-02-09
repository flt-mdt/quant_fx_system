import numpy as np
import pandas as pd

from quant_fx_system.quant.information.drift import psi
from quant_fx_system.quant.information.entropy import joint_entropy
from quant_fx_system.quant.information.transfer_entropy import transfer_entropy


def _psi_wrong(train: pd.Series, live: pd.Series, n_bins: int = 10) -> float:
    train_bins = pd.qcut(train.dropna(), q=n_bins, duplicates="drop")
    live_bins = pd.qcut(live.dropna(), q=n_bins, duplicates="drop")
    p = train_bins.value_counts(normalize=True, sort=False).to_numpy()
    q = live_bins.value_counts(normalize=True, sort=False).to_numpy()
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum((p - q) * np.log(p / q)))


def test_drift_bins_detect_shift() -> None:
    rng = np.random.default_rng(42)
    train = pd.Series(rng.normal(0.0, 1.0, 2000))
    live = pd.Series(rng.normal(1.0, 1.0, 2000))
    psi_correct = psi(train, live, n_bins=10)
    psi_wrong = _psi_wrong(train, live, n_bins=10)
    assert psi_correct > psi_wrong
    assert psi_correct > 0.1


def test_transfer_entropy_direction() -> None:
    rng = np.random.default_rng(7)
    x = pd.Series(rng.normal(0.0, 1.0, 1000))
    noise = rng.normal(0.0, 0.1, 1000)
    y = x.shift(1).fillna(0.0) + noise
    te_x_to_y = transfer_entropy(x, y, n_bins=8)
    te_y_to_x = transfer_entropy(y, x, n_bins=8)
    assert te_x_to_y > te_y_to_x
    assert te_x_to_y >= 0.0


def test_joint_entropy_alignment() -> None:
    x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    y = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
    aligned = pd.concat([x, y], axis=1).dropna()
    expected = joint_entropy(aligned.iloc[:, 0], aligned.iloc[:, 1], n_bins=3)
    actual = joint_entropy(x, y, n_bins=3)
    assert np.isfinite(actual)
    assert np.isclose(actual, expected)
