"""Signal ensemble support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from quant_fx_system.quant.signals.base import BaseSignal
from quant_fx_system.quant.signals.transforms import to_position_from_score


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for ensemble signals."""

    weights: dict[str, float]
    max_leverage: float = 1.0
    normalize_weights: bool = True
    combine: Literal["sum", "mean"] = "sum"
    post_transform: Literal["tanh", "clip", None] = "tanh"
    k: float = 1.0


class EnsembleSignal(BaseSignal):
    """Combine multiple signals into one ensemble."""

    def __init__(self, signals: list[BaseSignal], config: EnsembleConfig) -> None:
        if not signals:
            raise ValueError("EnsembleSignal requires at least one signal.")
        self.signals = signals
        self.config = config
        self.name = "ensemble"

    def compute_alpha(self, features: pd.DataFrame) -> pd.Series:
        alpha_series: list[pd.Series] = []
        for signal in self.signals:
            alpha = signal.compute_alpha(features)
            alpha_series.append(alpha.rename(signal.name))
        combined = pd.concat(alpha_series, axis=1, join="inner")
        if combined.empty:
            raise ValueError("No overlapping data to combine in ensemble.")
        weights = self._resolve_weights(combined.columns.tolist())
        weighted = combined.mul(pd.Series(weights), axis=1)
        if self.config.combine == "mean":
            alpha = weighted.mean(axis=1)
        else:
            alpha = weighted.sum(axis=1)
        return alpha.astype(float)

    def compute_position(self, alpha: pd.Series, features: pd.DataFrame) -> pd.Series:
        _ = features
        if self.config.post_transform is None:
            return alpha.clip(
                lower=-self.config.max_leverage, upper=self.config.max_leverage
            )
        return to_position_from_score(
            alpha,
            method=self.config.post_transform,
            max_leverage=self.config.max_leverage,
            k=self.config.k,
        )

    def _resolve_weights(self, names: list[str]) -> dict[str, float]:
        missing = [name for name in names if name not in self.config.weights]
        if missing:
            raise ValueError(f"Missing weights for signals: {missing}")
        weights = {name: float(self.config.weights[name]) for name in names}
        if not self.config.normalize_weights:
            return weights
        total = sum(abs(value) for value in weights.values())
        if total == 0:
            raise ValueError("Sum of absolute weights must be non-zero.")
        return {name: value / total for name, value in weights.items()}
