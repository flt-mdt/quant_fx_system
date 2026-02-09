from __future__ import annotations

import numpy as np
import pandas as pd


def _bin_series(series: pd.Series, n_bins: int = 10) -> pd.Categorical:
    return pd.qcut(series, q=n_bins, duplicates="drop")


def transfer_entropy(
    x: pd.Series,
    y: pd.Series,
    n_bins: int = 10,
    epsilon: float = 1e-12,
) -> float:
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    bx = _bin_series(aligned.iloc[:, 0], n_bins=n_bins)
    by = _bin_series(aligned.iloc[:, 1], n_bins=n_bins)

    y_t = by[1:]
    y_past = by[:-1]
    x_past = bx[:-1]

    triples = pd.DataFrame({"y_t": y_t, "y_past": y_past, "x_past": x_past})
    total = len(triples)
    if total == 0:
        return float("nan")

    triple_counts = triples.value_counts().to_dict()
    yp_xp_counts = triples.groupby(["y_past", "x_past"]).size().to_dict()
    yt_yp_counts = triples.groupby(["y_t", "y_past"]).size().to_dict()
    yp_counts = triples.groupby(["y_past"]).size().to_dict()

    te = 0.0
    for (y_t_val, y_past_val, x_past_val), count in triple_counts.items():
        p_triple = count / total
        p_y_given_yp_xp = count / yp_xp_counts[(y_past_val, x_past_val)]
        p_y_given_yp = yt_yp_counts[(y_t_val, y_past_val)] / yp_counts[y_past_val]
        ratio = (p_y_given_yp_xp + epsilon) / (p_y_given_yp + epsilon)
        te += p_triple * np.log(ratio)
    return float(te)
