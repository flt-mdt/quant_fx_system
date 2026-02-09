from __future__ import annotations

import hashlib
import json

import pandas as pd

from .drift import drift_report
from .ic import ic_by_feature, ic_conditional
from .mi import mi_by_feature, pairwise_mi_matrix
from .statistics import bh_fdr
from .targets import binary_up_target, forward_returns, meta_label_target
from .transfer_entropy import te_matrix, te_significance
from .types import InformationConfig, InformationReport
from .validation import align_inputs, check_no_future_usage, schema, shift_features


def _config_hash(cfg: InformationConfig) -> str:
    payload = json.dumps(cfg.to_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_ic_report(
    features: pd.DataFrame,
    target: pd.Series,
    cfg: InformationConfig,
    regimes: pd.Series | None = None,
) -> pd.DataFrame:
    nw_lags = cfg.nw_lags
    if nw_lags is None and cfg.overlap:
        nw_lags = cfg.horizon
    ic_df = ic_by_feature(
        features,
        target,
        method=cfg.ic_method,
        window=cfg.ic_window,
        nw_lags=nw_lags,
    )
    if regimes is not None and cfg.regime_conditional:
        ic_df["ic_regime"] = [
            ic_conditional(features[col], target, regimes, method=cfg.ic_method)["ic"].mean()
            for col in features.columns
        ]
    return ic_df


def compute_mi_report(
    features: pd.DataFrame,
    target: pd.Series,
    cfg: InformationConfig,
    regimes: pd.Series | None = None,
) -> pd.DataFrame:
    return mi_by_feature(features, target, cfg, regimes=regimes)


def compute_drift_report(
    train_df: pd.DataFrame,
    live_df: pd.DataFrame,
    cfg: InformationConfig,
) -> pd.DataFrame:
    return drift_report(train_df, live_df, cfg)


def _apply_multiple_testing(df: pd.DataFrame, cfg: InformationConfig) -> pd.DataFrame:
    if cfg.multiple_testing == "none" or "pvalue" not in df.columns:
        return df
    mask = bh_fdr(df["pvalue"].to_numpy(), cfg.alpha)
    df = df.copy()
    df["fdr_pass"] = mask
    return df


def _select_te_features(
    features: pd.DataFrame,
    ic_report: pd.DataFrame,
    mi_report: pd.DataFrame,
    cfg: InformationConfig,
) -> list[str]:
    cols = list(features.columns)
    if cfg.te_features is not None:
        return [col for col in cfg.te_features if col in features.columns]
    if cfg.te_top_k is None or cfg.te_top_k >= len(cols):
        return cols

    score = pd.Series(0.0, index=cols)
    if "ic_mean" in ic_report.columns:
        score = ic_report["ic_mean"].abs()
    if "mi" in mi_report.columns:
        score = score.where(~score.isna(), mi_report["mi"])
        score = score.fillna(mi_report["mi"])
    score = score.fillna(0.0)
    return score.sort_values(ascending=False).head(cfg.te_top_k).index.tolist()


def build_information_report(
    *,
    returns: pd.Series,
    base_signal: pd.Series | None,
    features: pd.DataFrame,
    regimes: pd.Series | None,
    live_features: pd.DataFrame | None = None,
    cfg: InformationConfig,
) -> InformationReport:
    returns, base_signal, features, regimes = align_inputs(
        returns, base_signal, features, regimes
    )
    shifted_features = shift_features(features, cfg)

    if cfg.target == "forward_return":
        target = forward_returns(returns, cfg.horizon)
    elif cfg.target == "binary_up":
        target = binary_up_target(forward_returns(returns, cfg.horizon))
    else:
        if base_signal is None:
            raise ValueError("base_signal required for meta_label target")
        target = meta_label_target(base_signal, forward_returns(returns, cfg.horizon))

    if cfg.output_shift:
        target = target.shift(cfg.output_shift)

    ic_report = compute_ic_report(shifted_features, target, cfg, regimes=regimes)
    mi_report = compute_mi_report(shifted_features, target, cfg, regimes=regimes)

    te_report = pd.DataFrame()
    if base_signal is not None:
        te_features = _select_te_features(shifted_features, ic_report, mi_report, cfg)
        if len(te_features) > 0:
            te_report = te_matrix(shifted_features[te_features], target, cfg)
        if cfg.te_significance:
            sig_rows = []
            for col in te_features:
                stats = te_significance(shifted_features[col], target, cfg)
                stats["feature"] = col
                sig_rows.append(stats)
            if sig_rows:
                te_report = te_report.join(
                    pd.DataFrame(sig_rows).set_index("feature"), how="left"
                )

    drift_df = pd.DataFrame()
    if live_features is not None:
        common_cols = shifted_features.columns.intersection(live_features.columns)
        if len(common_cols) > 0:
            drift_df = compute_drift_report(
                shifted_features[common_cols], live_features[common_cols], cfg
            )

    redundancy = None
    if shifted_features.shape[1] <= 200:
        redundancy = pairwise_mi_matrix(shifted_features, cfg)

    ic_report = _apply_multiple_testing(ic_report, cfg)
    mi_report = _apply_multiple_testing(mi_report, cfg)

    metadata = {
        "config_hash": _config_hash(cfg),
        "schema": schema(shifted_features, cfg),
        "n_obs": int(len(shifted_features)),
        "warnings": check_no_future_usage(returns, shifted_features, target)["warnings"],
    }

    return InformationReport(
        ic=ic_report,
        mi=mi_report,
        te=te_report,
        drift=drift_df,
        redundancy=redundancy,
        metadata=metadata,
    )
