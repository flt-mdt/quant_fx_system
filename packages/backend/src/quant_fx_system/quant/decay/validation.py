from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fx_system.quant.decay.types import DecayConfig


def validate_utc_series(series: pd.Series, name: str, allow_nans: bool = False) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{name} index must be a pandas DatetimeIndex")
    if series.index.tz is None or str(series.index.tz) != "UTC":
        raise ValueError(f"{name} index must be UTC timezone-aware")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonic increasing")
    if not series.index.is_unique:
        raise ValueError(f"{name} index must be unique")

    numeric = pd.to_numeric(series, errors="raise").astype(float)
    values = numeric.to_numpy(dtype=float)
    if not allow_nans and np.isnan(values).any():
        raise ValueError(f"{name} contains NaN values")
    if allow_nans:
        finite_mask = ~np.isnan(values)
        if not np.isfinite(values[finite_mask]).all():
            raise ValueError(f"{name} contains non-finite values")
    else:
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
    return numeric


def validate_config(cfg: DecayConfig) -> None:
    if cfg.shift < 0:
        raise ValueError("shift must be >= 0")
    if cfg.min_periods < 1:
        raise ValueError("min_periods must be >= 1")

    if cfg.alpha is not None and (cfg.alpha <= 0 or cfg.alpha > 1):
        raise ValueError("alpha must be in (0, 1]")
    if cfg.half_life_bars is not None and cfg.half_life_bars <= 0:
        raise ValueError("half_life_bars must be > 0")
    if cfg.half_life_time is not None and cfg.half_life_time <= pd.Timedelta(0):
        raise ValueError("half_life_time must be > 0")

    if cfg.kind == "ewma":
        decay_params = [cfg.alpha, cfg.half_life_bars, cfg.half_life_time]
        provided = sum(param is not None for param in decay_params)
        if provided != 1:
            raise ValueError(
                "ewma requires exactly one of alpha, half_life_bars, or half_life_time"
            )

    if cfg.kind in {"linear", "step", "power"}:
        if cfg.window < 2:
            raise ValueError("window must be >= 2 for kernel decays")
    if cfg.kind == "power" and cfg.power_exponent <= 0:
        raise ValueError("power_exponent must be > 0 for power decay")
