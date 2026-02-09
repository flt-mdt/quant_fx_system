from __future__ import annotations

import numpy as np
import pandas as pd

from .types import MetaModelConfig


def _forward_returns(returns: pd.Series, horizon: int) -> pd.Series:
    """Compute forward returns using additive (log) returns."""
    cumsum = returns.cumsum()
    forward_sum = cumsum.shift(-horizon) - cumsum
    return forward_sum


def forward_returns(*, returns: pd.Series, horizon: int) -> pd.Series:
    return _forward_returns(returns, horizon)


def meta_label(
    *,
    returns: pd.Series,
    base_signal: pd.Series,
    cfg: MetaModelConfig,
) -> pd.Series:
    """Meta label assumes base_signal is a signed exposure."""
    horizon = cfg.horizon
    r_fwd = _forward_returns(returns, horizon)
    direction = np.sign(base_signal)
    costs = (cfg.transaction_cost_bps + cfg.slippage_bps) / 1e4
    pnl = direction * r_fwd - costs
    label = (pnl > 0).astype(float)
    label[direction == 0] = np.nan
    return label


def return_sign_label(*, returns: pd.Series, cfg: MetaModelConfig) -> pd.Series:
    horizon = cfg.horizon
    r_fwd = _forward_returns(returns, horizon)
    label = (r_fwd > 0).astype(float)
    label = label.where(r_fwd != 0, np.nan)
    return label


def quantile_label(*, returns: pd.Series, cfg: MetaModelConfig) -> pd.Series:
    horizon = cfg.horizon
    r_fwd = _forward_returns(returns, horizon)
    threshold = r_fwd.quantile(0.8)
    return (r_fwd >= threshold).astype(float)


def triple_barrier_label(
    *,
    prices: pd.Series,
    returns: pd.Series,
    base_signal: pd.Series,
    cfg: MetaModelConfig,
) -> pd.Series:
    horizon = cfg.horizon
    pt = cfg.triple_barrier_pt
    sl = cfg.triple_barrier_sl
    direction = np.sign(base_signal)
    label = pd.Series(index=returns.index, dtype=float)

    for idx, current_price in prices.items():
        if idx not in returns.index:
            continue
        if direction.loc[idx] == 0 or pd.isna(direction.loc[idx]):
            label.loc[idx] = np.nan
            continue
        try:
            loc = prices.index.get_loc(idx)
        except KeyError:
            label.loc[idx] = np.nan
            continue
        end_loc = min(loc + horizon, len(prices) - 1)
        path = prices.iloc[loc : end_loc + 1]
        entry = current_price
        up = entry * (1 + pt)
        down = entry * (1 - sl)
        hit = None
        for t, price in path.iloc[1:].items():
            if price >= up:
                hit = 1
                break
            if price <= down:
                hit = -1
                break
        if hit is None:
            r_fwd = (path.iloc[-1] / entry) - 1
            hit = np.sign(r_fwd) if r_fwd != 0 else 0
        label.loc[idx] = 1.0 if hit * direction.loc[idx] > 0 else 0.0

    return label


def build_labels(
    *,
    prices: pd.Series | None,
    returns: pd.Series,
    base_signal: pd.Series,
    cfg: MetaModelConfig,
) -> pd.Series:
    if cfg.labeling == "meta_label":
        return meta_label(returns=returns, base_signal=base_signal, cfg=cfg)
    if cfg.labeling == "triple_barrier":
        if prices is None:
            raise ValueError("prices required for triple_barrier labeling")
        return triple_barrier_label(
            prices=prices, returns=returns, base_signal=base_signal, cfg=cfg
        )
    if cfg.labeling == "return_sign":
        return return_sign_label(returns=returns, cfg=cfg)
    if cfg.labeling == "quantile":
        return quantile_label(returns=returns, cfg=cfg)
    raise ValueError(f"Unsupported labeling: {cfg.labeling}")
