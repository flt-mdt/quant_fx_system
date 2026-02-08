from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class VolTargetConfig:
    enabled: bool = True
    window: int = 20
    target_vol: float = 0.10
    annualization: Literal["auto", "none"] = "auto"
    epsilon: float = 1e-12
    min_scale: float = 0.0
    max_scale: float = 5.0
    shift: int = 1


@dataclass(frozen=True)
class DrawdownConfig:
    enabled: bool = True
    max_drawdown: float = 0.20
    mode: Literal["flatten", "linear_delever"] = "flatten"
    floor_leverage: float = 0.0
    cooldown_bars: int = 0
    shift: int = 0


@dataclass(frozen=True)
class TurnoverConfig:
    enabled: bool = True
    mode: Literal["cap_delta", "ewma_smooth"] = "cap_delta"
    max_turnover_per_bar: float = 1.0
    ewma_alpha: float = 0.20
    shift: int = 0


@dataclass(frozen=True)
class VarEsConfig:
    enabled: bool = False
    window: int = 250
    alpha: float = 0.01
    shift: int = 1


@dataclass(frozen=True)
class RiskConfig:
    max_leverage: float = 1.0
    vol_target: VolTargetConfig = VolTargetConfig()
    drawdown: DrawdownConfig = DrawdownConfig()
    turnover: TurnoverConfig = TurnoverConfig()
    var_es: VarEsConfig = VarEsConfig()


@dataclass(frozen=True)
class RiskResult:
    position: pd.Series
    metrics: pd.DataFrame
    metadata: dict = field(default_factory=dict)
