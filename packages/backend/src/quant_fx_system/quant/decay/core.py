from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quant_fx_system.quant.decay.ewma import ewma_time_aware, half_life_to_alpha
from quant_fx_system.quant.decay.types import DecayConfig, DecayResult
from quant_fx_system.quant.decay.validation import validate_config, validate_utc_series
from quant_fx_system.quant.decay.weights import generate_weights


def apply_decay(series: pd.Series, cfg: DecayConfig) -> DecayResult:
    validate_config(cfg)
    validate_utc_series(series, "series", allow_nans=True)

    series_work = series
    if cfg.fillna_value is not None:
        series_work = series_work.fillna(cfg.fillna_value)

    weights_series: pd.Series | None = None
    metadata: dict = {"kind": cfg.kind}

    if cfg.kind == "none":
        output = series_work.copy()
    elif cfg.kind == "ewma":
        if cfg.half_life_time is not None:
            output = ewma_time_aware(
                series_work,
                cfg.half_life_time,
                min_periods=cfg.min_periods,
                fillna_value=None,
                carry_forward_nan=True,
            )
            metadata["half_life_time"] = cfg.half_life_time
        else:
            alpha = cfg.alpha
            if alpha is None:
                alpha = half_life_to_alpha(cfg.half_life_bars or 1.0)
            output = series_work.ewm(alpha=alpha, adjust=False, min_periods=cfg.min_periods).mean()
            metadata["alpha"] = alpha
    elif cfg.kind in {"linear", "step", "power"}:
        weights = generate_weights(
            cfg.kind,
            cfg.window,
            power_exponent=cfg.power_exponent,
            normalize=cfg.normalize_weights,
        )

        def _weighted_mean(x: np.ndarray) -> float:
            w = weights[-len(x) :]
            return float(np.dot(w, x))

        output = series_work.rolling(window=cfg.window, min_periods=cfg.min_periods).apply(
            _weighted_mean,
            raw=True,
        )
        weights_series = pd.Series(weights, index=range(1, len(weights) + 1), name="weights")
        metadata["window"] = cfg.window
    else:
        raise ValueError(f"Unsupported decay kind: {cfg.kind}")

    if cfg.shift > 0:
        output = output.shift(cfg.shift)

    if cfg.fillna_value is not None:
        output = output.fillna(cfg.fillna_value)

    return DecayResult(output=output, weights=weights_series, metadata=metadata)


def decay_position_target(target: pd.Series, cfg: DecayConfig) -> pd.Series:
    validate_config(cfg)
    validate_utc_series(target, "target", allow_nans=True)

    target_work = target
    if cfg.fillna_value is not None:
        target_work = target_work.fillna(cfg.fillna_value)

    if cfg.half_life_time is not None:
        hl_seconds = cfg.half_life_time.total_seconds()
        if hl_seconds <= 0:
            raise ValueError("half_life_time must be > 0")
        index = target_work.index
        values = target_work.to_numpy(dtype=float)
        output = np.empty_like(values)
        prev = np.nan
        for i in range(len(values)):
            x_i = values[i]
            if i == 0:
                prev = x_i
            else:
                dt = (index[i] - index[i - 1]).total_seconds()
                lambda_i = math.exp(-math.log(2.0) * dt / hl_seconds)
                if np.isnan(x_i):
                    prev = prev
                else:
                    if np.isnan(prev):
                        prev = x_i
                    else:
                        prev = (1.0 - lambda_i) * x_i + lambda_i * prev
            output[i] = prev
        result = pd.Series(output, index=target_work.index, name=target_work.name)
    else:
        alpha = cfg.alpha
        if alpha is None:
            alpha = half_life_to_alpha(cfg.half_life_bars or 1.0)
        k = 1.0 - alpha
        values = target_work.to_numpy(dtype=float)
        output = np.empty_like(values)
        prev = np.nan
        for i in range(len(values)):
            x_i = values[i]
            if i == 0:
                prev = x_i
            else:
                if np.isnan(x_i):
                    prev = prev
                else:
                    if np.isnan(prev):
                        prev = x_i
                    else:
                        prev = (1.0 - k) * x_i + k * prev
            output[i] = prev
        result = pd.Series(output, index=target_work.index, name=target_work.name)

    if cfg.shift > 0:
        result = result.shift(cfg.shift)
    if cfg.fillna_value is not None:
        result = result.fillna(cfg.fillna_value)
    return result
