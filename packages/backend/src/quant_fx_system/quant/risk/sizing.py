from __future__ import annotations

import pandas as pd

from .metrics import annualize_vol, infer_periods_per_year, rolling_vol
from .types import VolTargetConfig
from .validation import validate_window


def scale_position(position: pd.Series, scale: pd.Series) -> pd.Series:
    scaled = position * scale
    scaled.name = position.name
    return scaled


def vol_target_position(
    position: pd.Series,
    returns: pd.Series,
    cfg: VolTargetConfig,
    max_leverage: float,
) -> tuple[pd.Series, dict[str, pd.Series]]:
    validate_window(cfg.window, "vol_target.window")
    vol = rolling_vol(returns, window=cfg.window, epsilon=cfg.epsilon)
    periods = None
    if cfg.annualization == "auto":
        periods = infer_periods_per_year(returns.index)
    vol_annualized = annualize_vol(vol, periods)

    scale_raw = cfg.target_vol / vol_annualized
    scale_raw = scale_raw.clip(lower=cfg.min_scale, upper=cfg.max_scale)
    scale_applied = scale_raw.shift(cfg.shift)
    scale_applied = scale_applied.fillna(1.0)
    scale_applied.name = "scale_vol"

    position_scaled = scale_position(position, scale_applied)
    position_scaled = position_scaled.clip(lower=-max_leverage, upper=max_leverage)
    position_scaled.name = "position_after_vol_target"

    metrics = {
        "vol_rolling": vol,
        "scale_vol": scale_applied,
        "position_after_vol_target": position_scaled,
    }
    return position_scaled, metrics
