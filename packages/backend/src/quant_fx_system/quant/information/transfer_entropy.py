from __future__ import annotations

import numpy as np
import pandas as pd

from .statistics import permutation_test_stat
from .types import InformationConfig

_DEF_EPS = 1e-12


def _bin_series(series: pd.Series, bins: int, binning: str) -> pd.Series:
    if binning == "quantile":
        return pd.qcut(series, q=bins, labels=False, duplicates="drop")
    if binning == "equal_width":
        return pd.cut(series, bins=bins, labels=False)
    raise ValueError("binning must be 'quantile' or 'equal_width'")


def _lagged_frame(x: pd.Series, y: pd.Series, lags_x: int, lags_y: int) -> pd.DataFrame:
    data = {"y_t": y}
    for lag in range(1, lags_y + 1):
        data[f"y_lag_{lag}"] = y.shift(lag)
    for lag in range(1, lags_x + 1):
        data[f"x_lag_{lag}"] = x.shift(lag)
    return pd.DataFrame(data).dropna()


def transfer_entropy(
    x: pd.Series,
    y: pd.Series,
    lags_x: int = 1,
    lags_y: int = 1,
    bins: int = 8,
    binning: str = "quantile",
) -> float:
    x_disc = _bin_series(x, bins, binning)
    y_disc = _bin_series(y, bins, binning)
    frame = _lagged_frame(x_disc, y_disc, lags_x, lags_y)
    if frame.empty:
        return float("nan")

    y_t = frame["y_t"].astype(int)
    y_past = frame[[c for c in frame.columns if c.startswith("y_lag")]].astype(int)
    x_past = frame[[c for c in frame.columns if c.startswith("x_lag")]].astype(int)

    def _tuple_rows(df: pd.DataFrame) -> list[tuple[int, ...]]:
        return list(map(tuple, df.to_numpy()))

    y_past_keys = _tuple_rows(y_past)
    x_past_keys = _tuple_rows(x_past)
    joint_keys = list(zip(y_past_keys, x_past_keys))

    counts_y = {}
    counts_ypast = {}
    counts_ypast_xpast = {}
    counts_joint = {}

    for yt, yp, xp, jkey in zip(y_t, y_past_keys, x_past_keys, joint_keys, strict=False):
        counts_y[(yt, yp)] = counts_y.get((yt, yp), 0) + 1
        counts_ypast[yp] = counts_ypast.get(yp, 0) + 1
        counts_ypast_xpast[jkey] = counts_ypast_xpast.get(jkey, 0) + 1
        counts_joint[(yt, jkey)] = counts_joint.get((yt, jkey), 0) + 1

    total = len(frame)
    te = 0.0
    for (yt, jkey), count_joint in counts_joint.items():
        yp, xp = jkey
        p_triple = count_joint / total

        denom_joint = counts_ypast_xpast.get(jkey, 0) + _DEF_EPS
        denom_yp = counts_ypast.get(yp, 0) + _DEF_EPS

        p_y_given_joint = count_joint / denom_joint
        p_y_given_yp = counts_y.get((yt, yp), 0) / denom_yp

        te += p_triple * np.log((p_y_given_joint + _DEF_EPS) / (p_y_given_yp + _DEF_EPS))
    return float(te)


def te_significance(
    x: pd.Series,
    y: pd.Series,
    cfg: InformationConfig,
) -> dict:
    def _stat_fn(a: pd.Series, b: pd.Series) -> float:
        return transfer_entropy(
            a,
            b,
            lags_x=cfg.te_source_lags,
            lags_y=cfg.te_lags,
            bins=cfg.te_bins,
            binning=cfg.te_binning,
        )

    te_value = _stat_fn(x, y)
    result = permutation_test_stat(
        x,
        y,
        stat_fn=_stat_fn,
        runs=cfg.te_perm_runs,
        block_size=cfg.te_block_size,
        random_seed=cfg.random_seed,
    )
    result.update({"te": te_value})
    return result


def te_matrix(features: pd.DataFrame, target: pd.Series, cfg: InformationConfig) -> pd.DataFrame:
    rows = []
    for col in features.columns:
        te_val = transfer_entropy(
            features[col],
            target,
            lags_x=cfg.te_source_lags,
            lags_y=cfg.te_lags,
            bins=cfg.te_bins,
            binning=cfg.te_binning,
        )
        rows.append({"feature": col, "te": te_val})
    return pd.DataFrame(rows).set_index("feature")
