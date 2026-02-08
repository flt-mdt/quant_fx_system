from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .drawdown import apply_drawdown_guard
from .metrics import compute_returns_from_price
from .sizing import vol_target_position
from .turnover import cap_position_delta, compute_turnover, ewma_smooth_position
from .types import RiskConfig, RiskResult
from .validation import (
    align_series,
    validate_alpha,
    validate_position_bounds,
    validate_positive,
    validate_utc_series,
    validate_window,
)


def apply_risk_overlay(
    *,
    position_raw: pd.Series,
    price: pd.Series | None = None,
    returns: pd.Series | None = None,
    cfg: RiskConfig,
    initial_equity: float = 1.0,
) -> RiskResult:
    """
    Apply risk sizing + limits + guards to a raw position series.

    Requirements:
      - Inputs must have UTC, monotonic, unique indices.
      - No lookahead: any metric derived from returns must be shifted by cfg.*.shift.
      - Positions are applied with a one-bar lag: position[t] impacts return[t+1].
    """

    if returns is None and price is None:
        raise ValueError("Either price or returns must be provided")

    validate_utc_series(position_raw, "position_raw", allow_nans=False)
    if price is not None:
        validate_utc_series(price, "price", allow_nans=False)
        validate_positive(price, "price")
    if returns is not None:
        validate_utc_series(returns, "returns", allow_nans=True)

    if cfg.max_leverage <= 0:
        raise ValueError("max_leverage must be positive")

    if cfg.turnover.enabled:
        validate_alpha(cfg.turnover.ewma_alpha, "turnover.ewma_alpha")

    if cfg.vol_target.enabled:
        validate_window(cfg.vol_target.window, "vol_target.window")

    if cfg.var_es.enabled:
        raise NotImplementedError("var/es overlay is not implemented yet")

    position_aligned, returns_aligned, price_aligned = align_series(
        position_raw, returns, price
    )

    if returns_aligned is None and price_aligned is not None:
        returns_aligned = compute_returns_from_price(price_aligned)
        returns_aligned = returns_aligned.loc[position_aligned.index]

    if returns_aligned is None:
        raise ValueError("Returns could not be derived from price")

    returns_aligned = returns_aligned.rename("returns")
    missing_returns = returns_aligned.isna()
    if missing_returns.any():
        if not (missing_returns.iloc[0] and not missing_returns.iloc[1:].any()):
            raise ValueError(
                "returns contains NaN values beyond the initial pct_change bar"
            )

    metrics = pd.DataFrame(index=position_aligned.index)
    metrics["returns"] = returns_aligned
    metrics["position_raw"] = position_aligned

    position_current = position_aligned.copy()

    if cfg.vol_target.enabled:
        position_current, vol_metrics = vol_target_position(
            position_current, returns_aligned, cfg.vol_target, cfg.max_leverage
        )
        for key, value in vol_metrics.items():
            metrics[key] = value
    else:
        metrics["position_after_vol_target"] = position_current

    if cfg.turnover.enabled:
        metrics["turnover_raw"] = compute_turnover(position_current)
        if cfg.turnover.mode == "cap_delta":
            position_current = cap_position_delta(
                position_current, cfg.turnover.max_turnover_per_bar
            )
        else:
            position_current = ewma_smooth_position(
                position_current, cfg.turnover.ewma_alpha
            )
        metrics["turnover_after"] = compute_turnover(position_current)
    else:
        metrics["turnover_raw"] = compute_turnover(position_current)
        metrics["turnover_after"] = metrics["turnover_raw"]

    if cfg.drawdown.enabled:
        position_current, dd_metrics = apply_drawdown_guard(
            position_current, returns_aligned, cfg.drawdown, initial_equity
        )
        for key, value in dd_metrics.items():
            metrics[key] = value
    else:
        metrics["equity_proxy"] = (1.0 + returns_aligned.fillna(0.0)).cumprod()
        metrics["drawdown"] = 0.0
        metrics["dd_flag"] = 0.0
        metrics["dd_guard_active"] = 0.0

    position_current = position_current.clip(
        lower=-cfg.max_leverage, upper=cfg.max_leverage
    )
    position_current = position_current.fillna(0.0)
    position_current.name = "position_final"

    metrics["position_final"] = position_current

    validate_position_bounds(position_current, cfg.max_leverage)

    metadata = {
        "config": asdict(cfg),
        "start": position_current.index[0].isoformat(),
        "end": position_current.index[-1].isoformat(),
        "execution_convention": "position[t] applied to return[t+1]",
        "windows": {
            "vol_target": cfg.vol_target.window if cfg.vol_target.enabled else None,
            "var_es": cfg.var_es.window if cfg.var_es.enabled else None,
            "drawdown": None,
        },
        "shifts": {
            "vol_target": cfg.vol_target.shift if cfg.vol_target.enabled else None,
            "turnover": cfg.turnover.shift if cfg.turnover.enabled else None,
            "drawdown": cfg.drawdown.shift if cfg.drawdown.enabled else None,
            "var_es": cfg.var_es.shift if cfg.var_es.enabled else None,
        },
    }

    return RiskResult(position=position_current, metrics=metrics, metadata=metadata)
