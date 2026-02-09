import numpy as np
import pandas as pd
import pytest

from quant_fx_system.quant.information import InformationConfig
from quant_fx_system.quant.information.core import build_information_report
from quant_fx_system.quant.information.drift import js_divergence, psi
from quant_fx_system.quant.information.entropy import js_divergence as js_divergence_probs
from quant_fx_system.quant.information.entropy import joint_entropy
from quant_fx_system.quant.information.ic import compute_ic
from quant_fx_system.quant.information.mi import mutual_information
from quant_fx_system.quant.information.targets import forward_returns
from quant_fx_system.quant.information.validation import align_inputs, ensure_utc_index, shift_features


def _make_series(values, tz="UTC"):
    index = pd.date_range("2020-01-01", periods=len(values), tz=tz)
    return pd.Series(values, index=index)


def test_validate_utc_index():
    series = _make_series([1.0, 2.0, 3.0], tz=None)
    with pytest.raises(ValueError):
        ensure_utc_index(series, "series")


def test_alignment_and_feature_shift():
    returns = _make_series([0.0, 0.01, -0.02, 0.03])
    signal = _make_series([0.1, 0.2, 0.3, 0.4])
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]}, index=returns.index)

    aligned = align_inputs(returns, signal, features)
    cfg = InformationConfig(horizon=1, feature_shift=1)
    shifted = shift_features(aligned[2], cfg)
    assert np.isnan(shifted.iloc[0, 0])
    assert shifted.iloc[1, 0] == features.iloc[0, 0]


def test_no_lookahead_in_forward_returns():
    rng = np.random.default_rng(42)
    returns_a = _make_series(rng.normal(0, 0.01, 60))
    returns_b = returns_a.copy()
    returns_b.iloc[-1] = 0.5
    features = pd.DataFrame({"f1": rng.normal(0, 1, 60)}, index=returns_a.index)

    horizon = 3
    target_a = forward_returns(returns_a, horizon)
    target_b = forward_returns(returns_b, horizon)
    prefix = returns_a.index[: -(horizon + 1)]
    ic_a = compute_ic(features.loc[prefix, "f1"], target_a.loc[prefix])
    ic_b = compute_ic(features.loc[prefix, "f1"], target_b.loc[prefix])
    assert np.isclose(ic_a, ic_b, atol=1e-12)


def test_ic_positive_and_random():
    rng = np.random.default_rng(0)
    target = _make_series(rng.normal(0, 1, 200))
    signal = target + rng.normal(0, 0.5, 200)
    ic_pos = compute_ic(signal, target, method="pearson")
    assert ic_pos > 0.4

    signal_rand = _make_series(rng.normal(0, 1, 200))
    ic_rand = compute_ic(signal_rand, target, method="pearson")
    assert abs(ic_rand) < 0.15


def test_mi_gaussian_copula():
    rng = np.random.default_rng(1)
    x = _make_series(rng.normal(0, 1, 300))
    y = x + rng.normal(0, 0.3, 300)
    cfg = InformationConfig(horizon=1, mi_estimator="gaussian_copula")
    mi_val = mutual_information(x, y, cfg)
    assert mi_val > 0.2

    y_indep = _make_series(rng.normal(0, 1, 300))
    mi_indep = mutual_information(x, y_indep, cfg)
    assert mi_indep < 0.05


def test_drift_measures():
    rng = np.random.default_rng(2)
    train = _make_series(rng.normal(0, 1, 400))
    live_same = _make_series(rng.normal(0, 1, 400))
    live_shift = _make_series(rng.normal(0.5, 1.2, 400))

    js_same = js_divergence(train, live_same)
    psi_same = psi(train, live_same)
    js_shift = js_divergence(train, live_shift)
    psi_shift = psi(train, live_shift)

    assert js_same < js_shift
    assert psi_same < psi_shift


def test_js_divergence_properties():
    rng = np.random.default_rng(5)
    p = rng.random(10)
    p = p / p.sum()
    js_same = js_divergence_probs(p, p)
    assert np.isclose(js_same, 0.0, atol=1e-12)

    q = rng.random(10)
    q = q / q.sum()
    js_val = js_divergence_probs(p, q)
    assert 0.0 <= js_val <= np.log(2.0) + 1e-12


def test_joint_entropy_alignment():
    x = _make_series([0.0, np.nan, 1.0, 2.0])
    y = _make_series([0.5, 1.5, np.nan, 2.5])
    aligned = pd.concat([x, y], axis=1).dropna()
    je_aligned = joint_entropy(aligned.iloc[:, 0], aligned.iloc[:, 1], bins=3)
    je_misaligned = joint_entropy(x, y, bins=3)
    assert np.isclose(je_aligned, je_misaligned, atol=1e-12)


def test_information_report_deterministic():
    rng = np.random.default_rng(3)
    returns = _make_series(rng.normal(0, 0.01, 120))
    base_signal = _make_series(rng.normal(0, 1, 120))
    features = pd.DataFrame({"f1": rng.normal(0, 1, 120), "f2": rng.normal(0, 1, 120)}, index=returns.index)

    cfg = InformationConfig(horizon=2, te_significance=False, random_seed=7)
    report_a = build_information_report(
        returns=returns,
        base_signal=base_signal,
        features=features,
        regimes=None,
        live_features=None,
        cfg=cfg,
    )
    report_b = build_information_report(
        returns=returns,
        base_signal=base_signal,
        features=features,
        regimes=None,
        live_features=None,
        cfg=cfg,
    )
    pd.testing.assert_frame_equal(report_a.ic, report_b.ic)
    pd.testing.assert_frame_equal(report_a.mi, report_b.mi)
    assert report_a.metadata["config_hash"] == report_b.metadata["config_hash"]
