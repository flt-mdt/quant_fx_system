from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd

DecayKind = Literal["none", "ewma", "linear", "step", "power"]


@dataclass(frozen=True)
class DecayConfig:
    kind: DecayKind = "ewma"

    # Causal alignment
    shift: int = 0

    # EWMA params
    alpha: Optional[float] = None
    half_life_bars: Optional[float] = None
    half_life_time: Optional[pd.Timedelta] = None

    # Kernel params
    window: int = 20
    power_exponent: float = 1.0

    # NaN handling
    min_periods: int = 1
    fillna_value: Optional[float] = None

    # Weight normalization
    normalize_weights: bool = True


@dataclass(frozen=True)
class DecayResult:
    output: pd.Series
    weights: Optional[pd.Series] = None
    metadata: dict = field(default_factory=dict)
