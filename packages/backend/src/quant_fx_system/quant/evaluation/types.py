from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class EvaluationConfig:
    risk_free_rate: float = 0.0
    annualization: Literal["auto", "none"] = "auto"
    periods_per_year_override: int | None = None
    rolling_windows: tuple[int, ...] = (20, 60, 252)
    drawdown_enabled: bool = True
    distribution_enabled: bool = True
    regression_enabled: bool = True
    robustness_enabled: bool = True
    psr_enabled: bool = True
    dsr_enabled: bool = True
    n_trials: int = 1


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict[str, float]
    tables: dict[str, pd.DataFrame]
    series: dict[str, pd.Series]
    metadata: dict[str, object]
