from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(returns: pd.Series, horizon: int) -> pd.Series:
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    compounded = (1.0 + returns).rolling(window=horizon).apply(np.prod, raw=True) - 1.0
    return compounded.shift(-horizon + 1)


def binary_up_target(ret_fwd: pd.Series) -> pd.Series:
    target = pd.Series(index=ret_fwd.index, dtype="float64")
    target[ret_fwd > 0] = 1.0
    target[ret_fwd <= 0] = 0.0
    return target


def meta_label_target(
    signal: pd.Series,
    ret_fwd: pd.Series,
    costs_bps: float = 0.0,
) -> pd.Series:
    costs = costs_bps / 10000.0
    aligned = pd.concat([signal, ret_fwd], axis=1, keys=["signal", "ret_fwd"]).dropna()
    edge = aligned["signal"].to_numpy() * aligned["ret_fwd"].to_numpy()
    label = (edge > costs).astype(float)
    return pd.Series(label, index=aligned.index)
