from __future__ import annotations

from datetime import timezone

import numpy as np
import pandas as pd

from quant_fx_system.quant.evaluation.periods import infer_periods_per_year
from .types import FeatureConfig, RegimeConfig


def validate_utc_series(
    series: pd.Series,
    name: str,
    *,
    allow_nans: bool = False,
) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise ValueError(f"{name} must be a pandas Series.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a DatetimeIndex.")
    if series.index.tz is None:
        raise ValueError(f"{name} index must be timezone-aware UTC.")
    if str(series.index.tz) != "UTC" and series.index.tz != timezone.utc:
        raise ValueError(f"{name} index must be UTC.")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonic increasing.")
    if not series.index.is_unique:
        raise ValueError(f"{name} index must be unique.")

    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy()
    if allow_nans:
        mask = ~np.isnan(values)
        if not np.isfinite(values[mask]).all():
            raise ValueError(f"{name} must contain finite values.")
    else:
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite values.")
    return numeric


def align_series(*series: pd.Series | None) -> tuple[pd.Series | None, ...]:
    non_null = [s for s in series if s is not None]
    if not non_null:
        return tuple(series)
    common_index = non_null[0].index
    for s in non_null[1:]:
        common_index = common_index.intersection(s.index)
    aligned = []
    for s in series:
        if s is None:
            aligned.append(None)
        else:
            aligned.append(s.reindex(common_index))
    return tuple(aligned)


def validate_config(cfg: RegimeConfig) -> None:
    if cfg.hmm.n_states < 2 or cfg.hmm.n_states > 4:
        raise ValueError("HMM n_states must be between 2 and 4.")
    if any(q <= 0 or q >= 1 for q in cfg.quantile_thresholds):
        raise ValueError("Quantile thresholds must be between 0 and 1.")
    if cfg.feature.feature_shift < 0:
        raise ValueError("feature_shift must be >= 0.")
    if cfg.output_shift < 0:
        raise ValueError("output_shift must be >= 0.")
    if cfg.calibration_start and cfg.calibration_end:
        if cfg.calibration_start > cfg.calibration_end:
            raise ValueError("calibration_start must be <= calibration_end.")


def infer_periods(cfg: RegimeConfig | FeatureConfig, index: pd.DatetimeIndex) -> int | None:
    feature_cfg = cfg.feature if isinstance(cfg, RegimeConfig) else cfg
    if feature_cfg.annualization == "none":
        return None
    if feature_cfg.periods_per_year_override is not None:
        return feature_cfg.periods_per_year_override
    return infer_periods_per_year(index)


__all__ = ["align_series", "infer_periods", "validate_config", "validate_utc_series"]
