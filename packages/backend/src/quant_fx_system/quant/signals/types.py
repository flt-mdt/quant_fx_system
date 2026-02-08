"""Types for signal outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

SeriesF = pd.Series
DF = pd.DataFrame


@dataclass(frozen=True)
class SignalResult:
    """Container for signal outputs."""

    alpha: pd.Series
    position: pd.Series
    metadata: dict[str, Any]
