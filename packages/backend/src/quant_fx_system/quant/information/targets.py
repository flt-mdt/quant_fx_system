from __future__ import annotations

import pandas as pd


def forward_returns(returns: pd.Series, horizon: int) -> pd.Series:
    """Compute forward log returns as sum over (t+1..t+horizon)."""
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    cumsum = returns.cumsum()
    forward_sum = cumsum.shift(-horizon) - cumsum
    return forward_sum


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
