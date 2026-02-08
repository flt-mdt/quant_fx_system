from __future__ import annotations

import numpy as np
import pandas as pd

from .types import DrawdownConfig


def compute_equity_from_returns(
    returns: pd.Series,
    position_applied: pd.Series,
    initial_equity: float = 1.0,
) -> pd.Series:
    pnl = position_applied.shift(1).fillna(0.0) * returns.fillna(0.0)
    equity = (1.0 + pnl).cumprod() * initial_equity
    equity.name = "equity_proxy"
    return equity


def compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    drawdown = 1.0 - equity / peak
    drawdown = drawdown.fillna(0.0)
    drawdown.name = "drawdown"
    return drawdown


def _apply_cooldown_flags(dd_flag: pd.Series, cooldown_bars: int) -> pd.Series:
    if cooldown_bars <= 0:
        return dd_flag
    active = np.zeros(len(dd_flag), dtype=bool)
    cooldown = 0
    for i, triggered in enumerate(dd_flag.to_numpy()):
        if triggered:
            cooldown = cooldown_bars
            active[i] = True
        elif cooldown > 0:
            active[i] = True
            cooldown -= 1
        else:
            active[i] = False
    return pd.Series(active, index=dd_flag.index, name="dd_guard_active")


def apply_drawdown_guard(
    position: pd.Series,
    returns: pd.Series,
    cfg: DrawdownConfig,
    initial_equity: float = 1.0,
) -> tuple[pd.Series, dict[str, pd.Series]]:
    equity = compute_equity_from_returns(returns, position, initial_equity)
    drawdown = compute_drawdown(equity)
    dd_flag = (drawdown >= cfg.max_drawdown).astype(float)
    dd_flag.name = "dd_flag"
    guard_active = _apply_cooldown_flags(drawdown >= cfg.max_drawdown, cfg.cooldown_bars)

    if cfg.mode == "flatten":
        position_guarded = position.where(~guard_active, 0.0)
    else:
        scale = 1.0 - (drawdown / cfg.max_drawdown)
        scale = scale.clip(lower=cfg.floor_leverage, upper=1.0)
        scale = scale.where(~guard_active, cfg.floor_leverage)
        position_guarded = position * scale

    position_guarded.name = "position_after_drawdown"
    metrics = {
        "equity_proxy": equity,
        "drawdown": drawdown,
        "dd_flag": dd_flag,
    }
    return position_guarded, metrics
