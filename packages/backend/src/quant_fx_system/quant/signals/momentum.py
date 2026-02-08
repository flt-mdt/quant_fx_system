"""Momentum z-score signal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from quant_fx_system.quant.signals.base import BaseSignal
from quant_fx_system.quant.signals.transforms import to_position_from_score, zscore_rolling


@dataclass(frozen=True)
class MomentumZScoreConfig:
    """Configuration for momentum z-score signal."""

    window: int = 20
    max_leverage: float = 1.0
    method: Literal["tanh", "clip"] = "tanh"
    k: float = 1.0


class MomentumZScoreSignal(BaseSignal):
    """Momentum signal using z-scored momentum features."""

    def __init__(self, config: MomentumZScoreConfig | None = None) -> None:
        self.config = config or MomentumZScoreConfig()
        self.name = f"momentum_zscore_{self.config.window}"

    def compute_alpha(self, features: pd.DataFrame) -> pd.Series:
        column = f"z_mom_{self.config.window}"
        if column in features.columns:
            alpha = features[column]
        else:
            if "ret_1" not in features.columns:
                raise ValueError("features must contain ret_1 to construct momentum.")
            momentum = features["ret_1"].rolling(
                window=self.config.window, min_periods=self.config.window
            ).sum()
            alpha = zscore_rolling(momentum, window=self.config.window, epsilon=1e-12)
        return alpha.astype(float)

    def compute_position(self, alpha: pd.Series, features: pd.DataFrame) -> pd.Series:
        _ = features
        return to_position_from_score(
            alpha,
            method=self.config.method,
            max_leverage=self.config.max_leverage,
            k=self.config.k,
        )
