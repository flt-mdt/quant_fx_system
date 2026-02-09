"""Schemas for quant state payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    quantity: float
    notional: Optional[float] = None


class Signal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    confidence: Optional[float] = None


class QuantStatePayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    as_of: datetime
    base_currency: str = "USD"
    positions: List[Position] = Field(default_factory=list)
    signals: List[Signal] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuantStateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    created_at: datetime
    state: QuantStatePayload
