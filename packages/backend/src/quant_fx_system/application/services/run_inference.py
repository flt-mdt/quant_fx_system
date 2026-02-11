"""Inference service for signals -> risk-adjusted position target."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from quant_fx_system.quant.risk import RiskConfig, apply_risk_overlay
from quant_fx_system.quant.signals.ensemble import EnsembleConfig, EnsembleSignal
from quant_fx_system.quant.signals.mean_reversion import MeanReversionZScoreConfig, MeanReversionZScoreSignal
from quant_fx_system.quant.signals.momentum import MomentumZScoreConfig, MomentumZScoreSignal


def run_signal_pipeline(
    *,
    features: pd.DataFrame,
    price: pd.Series,
    momentum: MomentumZScoreSignal | None = None,
    mean_reversion: MeanReversionZScoreSignal | None = None,
    ensemble_cfg: EnsembleConfig | None = None,
    risk_cfg: RiskConfig | None = None,
) -> dict[str, object]:
    momentum = momentum or MomentumZScoreSignal(MomentumZScoreConfig(window=5))
    mean_reversion = mean_reversion or MeanReversionZScoreSignal(MeanReversionZScoreConfig(window=5))

    momentum_result = momentum.run(features)
    mean_reversion_result = mean_reversion.run(features)

    weights = {
        momentum.name: 0.5,
        mean_reversion.name: 0.5,
    }
    if ensemble_cfg is None:
        ensemble_cfg = EnsembleConfig(weights=weights)

    ensemble = EnsembleSignal([momentum, mean_reversion], ensemble_cfg)
    ensemble_result = ensemble.run(features)

    risk_cfg = risk_cfg or RiskConfig()
    risk_result = apply_risk_overlay(position_raw=ensemble_result.position, price=price, cfg=risk_cfg)

    return {
        "signals": {
            momentum.name: momentum_result,
            mean_reversion.name: mean_reversion_result,
            "ensemble": ensemble_result,
        },
        "position_target": risk_result.position,
        "risk_metrics": risk_result.metrics,
        "metadata": {
            "momentum": momentum_result.metadata,
            "mean_reversion": mean_reversion_result.metadata,
            "ensemble": ensemble_result.metadata,
            "ensemble_config": asdict(ensemble_cfg),
            "risk": risk_result.metadata,
        },
    }
