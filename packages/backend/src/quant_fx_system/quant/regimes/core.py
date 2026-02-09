from __future__ import annotations

import numpy as np
import pandas as pd

from .changepoints import cusum_flags
from .features import compute_features
from .hmm import fit_hmm_gaussian, hmm_filter, hmm_smooth, hmm_viterbi
from .labeling import label_from_quantiles, label_trend_range
from .transitions import regime_durations, transition_matrix
from .types import RegimeConfig, RegimeResult, serialize_config
from .validation import align_series, infer_periods, validate_config, validate_utc_series


def _diagnostics(regime: pd.Series, n_states: int) -> dict:
    durations = regime_durations(regime)
    return {
        "transition_matrix": transition_matrix(regime, n_states),
        "state_durations": {
            "avg": durations.mean() if len(durations) else None,
            "median": durations.median() if len(durations) else None,
            "p90": durations.quantile(0.9) if len(durations) else None,
        },
        "state_counts": regime.value_counts(dropna=True).to_dict(),
    }


def _apply_output_shift(regime: pd.Series, proba: pd.DataFrame | None, shift: int) -> tuple[pd.Series, pd.DataFrame | None]:
    if shift <= 0:
        return regime, proba
    regime_shifted = regime.shift(shift)
    proba_shifted = proba.shift(shift) if proba is not None else None
    return regime_shifted, proba_shifted


def _calibration_mask(
    index: pd.DatetimeIndex,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.Series:
    mask = pd.Series(True, index=index)
    if start is not None:
        start_ts = pd.Timestamp(start)
        mask &= index >= start_ts
    if end is not None:
        end_ts = pd.Timestamp(end)
        mask &= index <= end_ts
    return mask


def _warn_if_full_calibration(
    metadata: dict,
    calibration_mask: pd.Series,
    valid_mask: pd.Series,
) -> None:
    if valid_mask.sum() > 0 and (calibration_mask & valid_mask).sum() == valid_mask.sum():
        metadata["warnings"].append("calibration_full_sample")


def infer_regimes(
    *,
    price: pd.Series | None = None,
    returns: pd.Series | None = None,
    cfg: RegimeConfig,
) -> RegimeResult:
    validate_config(cfg)
    if price is None and returns is None:
        raise ValueError("Either price or returns must be provided.")
    if price is not None:
        price = validate_utc_series(price, "price", allow_nans=False)
    if returns is not None:
        returns = validate_utc_series(returns, "returns", allow_nans=True)
    price, returns = align_series(price, returns)

    price_filled = False
    if price is None and returns is not None and returns.isna().any():
        price = (1 + returns.fillna(0.0)).cumprod()
        price_filled = True

    features = compute_features(price, returns, cfg.feature)
    periods_per_year = infer_periods(cfg, features.index)
    metadata = {
        "config": serialize_config(cfg),
        "periods_per_year": periods_per_year,
        "warnings": [],
    }
    if price_filled:
        metadata["warnings"].append("price_from_returns_filled")
    if cfg.feature.annualization == "none":
        metadata["warnings"].append("annualization_disabled")
    if cfg.calibration_start is not None:
        metadata["calibration_start"] = cfg.calibration_start
    if cfg.calibration_end is not None:
        metadata["calibration_end"] = cfg.calibration_end

    regime = None
    proba = None
    diagnostics: dict = {}

    if cfg.method == "quantile_vol":
        vol_cols = [c for c in features.columns if c.startswith("vol_roll_")]
        if not vol_cols:
            raise ValueError("Volatility regime requires vol_roll feature.")
        vol = features[vol_cols[0]]
        calibration_mask = _calibration_mask(vol.index, cfg.calibration_start, cfg.calibration_end)
        _warn_if_full_calibration(metadata, calibration_mask, vol.notna())
        regime, proba = label_from_quantiles(
            vol,
            cfg.quantile_thresholds,
            calibration_vol=vol.loc[calibration_mask],
        )
        diagnostics = _diagnostics(regime, proba.shape[1])
        metadata["warnings"].extend(regime.attrs.get("warnings", []))
        metadata["n_states_effective"] = regime.attrs.get("n_states_effective", proba.shape[1])

    elif cfg.method == "trend_range":
        calibration_mask = _calibration_mask(features.index, cfg.calibration_start, cfg.calibration_end)
        slope_cols = [c for c in features.columns if c.startswith("slope_")]
        valid_mask = ~features[slope_cols[0]].isna() if slope_cols else pd.Series(False, index=features.index)
        _warn_if_full_calibration(metadata, calibration_mask, valid_mask)
        regime, proba = label_trend_range(features, cfg, calibration_mask=calibration_mask)
        diagnostics = _diagnostics(regime, 2)

    elif cfg.method == "hmm_gaussian":
        if cfg.hmm.mode in {"smooth", "viterbi"}:
            metadata["warnings"].append("offline_only")
        hmm_features = features
        if cfg.hmm_feature_cols:
            missing = [col for col in cfg.hmm_feature_cols if col not in features.columns]
            if missing:
                raise ValueError(f"Missing HMM feature columns: {missing}")
            hmm_features = features[list(cfg.hmm_feature_cols)]

        valid_mask = ~hmm_features.isna().any(axis=1)
        dropped = (~valid_mask).sum()
        if dropped > 0:
            metadata["warnings"].append("dropped_nan_rows")
        calibration_mask = _calibration_mask(hmm_features.index, cfg.calibration_start, cfg.calibration_end)
        _warn_if_full_calibration(metadata, calibration_mask, valid_mask)
        fit_mask = valid_mask & calibration_mask
        hmm_features_fit = hmm_features.loc[fit_mask]
        if hmm_features_fit.empty:
            raise ValueError("HMM calibration window contains no valid rows after NaN filtering.")
        hmm_features_valid = hmm_features.loc[valid_mask]
        if hmm_features_valid.empty:
            raise ValueError("HMM features contain no valid rows after NaN filtering.")
        params = fit_hmm_gaussian(hmm_features_fit, cfg.hmm)
        diagnostics.update(
            {
                "loglik_final": params["loglik_final"],
                "loglik_path": params["loglik_path"],
                "converged": params["converged"],
                "n_iter": params["n_iter"],
                "means": params["means"],
                "vars": params["vars"],
                "pi": params["pi"],
                "A": params["A"],
            }
        )
        if cfg.hmm.mode == "filter":
            proba_valid = hmm_filter(hmm_features_valid, params)
        elif cfg.hmm.mode == "smooth":
            proba_valid = hmm_smooth(hmm_features_valid, params)
        else:
            proba_valid = None
        if cfg.hmm.mode == "viterbi":
            regime_valid = hmm_viterbi(hmm_features_valid, params)
        else:
            regime_valid = proba_valid.idxmax(axis=1).str.replace("p_state_", "").astype(int)
        regime = pd.Series(pd.NA, index=hmm_features.index, name="regime")
        regime.loc[valid_mask] = regime_valid.astype(int)
        if proba_valid is not None:
            proba = pd.DataFrame(index=hmm_features.index, columns=proba_valid.columns, dtype=float)
            proba.loc[valid_mask] = proba_valid
        else:
            proba = None
        diagnostics.update(_diagnostics(regime, cfg.hmm.n_states))

    elif cfg.method == "cusum":
        if cfg.cusum.feature == "returns":
            series = features["ret_1"]
        else:
            vol_cols = [c for c in features.columns if c.startswith("vol_roll_")]
            if not vol_cols:
                raise ValueError("CUSUM vol requires vol_roll feature.")
            series = features[vol_cols[0]]
        flags = cusum_flags(series, cfg.cusum.threshold, cfg.cusum.drift)
        regime = flags.astype("Int64")
        proba = None
        diagnostics = {
            "flags": flags.sum(),
        }

    else:
        raise ValueError(f"Unsupported regime method: {cfg.method}")

    regime, proba = _apply_output_shift(regime, proba, cfg.output_shift)
    return RegimeResult(
        regime=regime,
        proba=proba,
        features=features,
        diagnostics=diagnostics,
        metadata=metadata,
    )


__all__ = ["infer_regimes"]
