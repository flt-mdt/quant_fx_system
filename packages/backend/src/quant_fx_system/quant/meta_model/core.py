from __future__ import annotations

import hashlib
from dataclasses import asdict

import numpy as np
import pandas as pd

from .calibration import make_calibrator
from .cv import build_cv_splits
from .decision import apply_decision_policy, select_threshold
from .diagnostics import summarize_oos
from .labeling import build_labels, forward_returns
from .models import fit_model
from .types import MetaModelConfig, MetaModelFit, MetaModelOutput
from .validation import align_inputs, align_predict_inputs, feature_schema, shift_features


def _hash_config(cfg: MetaModelConfig) -> str:
    payload = str(sorted(asdict(cfg).items())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_features(columns: list[str]) -> str:
    payload = ",".join(columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prepare_features(
    features: pd.DataFrame,
    cfg: MetaModelConfig,
) -> pd.DataFrame:
    features = shift_features(features, cfg)
    ordered_cols = sorted(features.columns)
    return features[ordered_cols]


def fit_meta_model(
    *,
    prices: pd.Series | None,
    returns: pd.Series,
    base_signal: pd.Series,
    features: pd.DataFrame,
    regimes: pd.Series | None,
    cfg: MetaModelConfig,
) -> MetaModelFit:
    returns, base_signal, features, regimes = align_inputs(
        returns=returns, base_signal=base_signal, features=features, regimes=regimes
    )
    features = _prepare_features(features, cfg)
    labels = build_labels(prices=prices, returns=returns, base_signal=base_signal, cfg=cfg)

    valid_mask = features.notna().all(axis=1) & labels.notna()
    dropped_nan_rows_train = int((~valid_mask).sum())
    X = features.loc[valid_mask].to_numpy()
    y = labels.loc[valid_mask].astype(float).to_numpy()
    index = features.loc[valid_mask].index

    if len(index) < cfg.min_train_size:
        raise ValueError("Not enough samples to train meta-model")

    splits = build_cv_splits(len(index), cfg)
    oos_rows = []
    for split in splits:
        train_idx = split.train_idx
        test_idx = split.test_idx
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = fit_model(X_train, y_train, cfg)
        raw_scores = model.predict_proba(X_train)
        calibrator = make_calibrator(cfg.calibration)
        calibrator.fit(raw_scores, y_train)

        test_scores = model.predict_proba(X_test)
        p_follow = calibrator.predict(test_scores)
        test_index = index[test_idx]
        oos_rows.append(
            pd.DataFrame(
                {
                    "p_follow": p_follow,
                    "y_true": y_test,
                },
                index=test_index,
            )
        )

    oos_predictions = (
        pd.concat(oos_rows).sort_index() if oos_rows and cfg.store_oos_predictions else pd.DataFrame()
    )
    metrics = summarize_oos(oos_predictions)

    final_model = fit_model(X, y, cfg)
    final_calibrator = make_calibrator(cfg.calibration)
    final_calibrator.fit(final_model.predict_proba(X), y)

    threshold = None
    if cfg.decision_policy == "threshold" and not oos_predictions.empty:
        shifted_p_follow = oos_predictions["p_follow"].shift(cfg.output_shift)
        aligned_index = shifted_p_follow.dropna().index
        aligned_returns = forward_returns(returns=returns, horizon=cfg.horizon).loc[aligned_index]
        direction = np.sign(base_signal.loc[aligned_index])
        pnl = direction * aligned_returns - (cfg.transaction_cost_bps + cfg.slippage_bps) / 1e4
        threshold = select_threshold(shifted_p_follow.loc[aligned_index], pnl)

    schema = feature_schema(features, cfg)
    metadata = {
        "config_hash": _hash_config(cfg),
        "feature_hash": _hash_features(schema["columns"]),
        "dropped_nan_rows_train": dropped_nan_rows_train,
        "threshold": threshold,
        "cv_scheme": cfg.cv_scheme,
    }

    return MetaModelFit(
        model=final_model,
        calibrator=final_calibrator,
        feature_schema=schema,
        oos_predictions=oos_predictions,
        metrics=metrics,
        metadata=metadata,
    )


def predict_meta_model(
    *,
    fit: MetaModelFit,
    features: pd.DataFrame,
    base_signal: pd.Series,
    regimes: pd.Series | None,
    cfg: MetaModelConfig,
) -> MetaModelOutput:
    base_signal, features, regimes = align_predict_inputs(
        base_signal=base_signal, features=features, regimes=regimes
    )
    features = _prepare_features(features, cfg)
    expected_cols = fit.feature_schema["columns"]
    features = features.reindex(columns=expected_cols)

    valid_mask = features.notna().all(axis=1)
    X = features.loc[valid_mask].to_numpy()
    scores = fit.model.predict_proba(X)
    p_follow_values = fit.calibrator.predict(scores)

    p_follow = pd.Series(index=features.index, dtype=float)
    p_follow.loc[valid_mask] = p_follow_values
    p_follow.loc[~valid_mask] = np.nan

    threshold = fit.metadata.get("threshold", 0.5)
    action, size, expected_edge = apply_decision_policy(
        p_follow=p_follow,
        base_signal=base_signal.reindex(features.index),
        threshold=threshold,
        cfg=cfg,
    )
    if cfg.output_shift != 0:
        p_follow = p_follow.shift(cfg.output_shift)
        action = action.shift(cfg.output_shift)
        size = size.shift(cfg.output_shift)
        if expected_edge is not None:
            expected_edge = expected_edge.shift(cfg.output_shift)

    metadata = {
        "dropped_nan_rows_pred": int((~valid_mask).sum()),
    }

    return MetaModelOutput(
        p_follow=p_follow,
        action=action,
        size=size,
        decision_threshold=threshold,
        expected_edge=expected_edge,
        metadata=metadata,
    )
