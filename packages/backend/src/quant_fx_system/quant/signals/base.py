"""Base signal interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from quant_fx_system.quant.signals.types import SignalResult
from quant_fx_system.quant.signals.validation import (
    validate_features_for_signals,
    validate_series_index,
)


class BaseSignal(ABC):
    """Base interface for signals."""

    name: str

    @abstractmethod
    def compute_alpha(self, features: pd.DataFrame) -> pd.Series:
        """Compute raw alpha values from features."""

    @abstractmethod
    def compute_position(self, alpha: pd.Series, features: pd.DataFrame) -> pd.Series:
        """Compute target positions from alpha."""

    def run(self, features: pd.DataFrame) -> SignalResult:
        """Run signal end-to-end with validation."""

        validate_features_for_signals(features, allow_nans=True)
        alpha = self.compute_alpha(features)
        alpha = alpha.dropna()
        validate_series_index(alpha, name=f"{self.name} alpha")
        position = self.compute_position(alpha, features)
        if position.isna().any():
            raise ValueError(f"{self.name} position contains NaNs after computation.")
        if not position.index.equals(alpha.index):
            raise ValueError(f"{self.name} position index must match alpha index.")
        validate_series_index(position, name=f"{self.name} position")
        metadata: dict[str, Any] = {
            "name": self.name,
            "decision_shift": features.attrs.get("decision_shift"),
        }
        return SignalResult(
            alpha=alpha.astype(float),
            position=position.astype(float),
            metadata=metadata,
        )
