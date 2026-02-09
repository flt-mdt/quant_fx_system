"""Backtest endpoints."""

from __future__ import annotations

from datetime import datetime
from math import sqrt
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from quant_fx_system.api.deps import SQLiteStorage, get_storage
from quant_fx_system.api.schemas.common import ErrorResponse
from quant_fx_system.quant.backtest.engine import run_backtest
from quant_fx_system.quant.backtest.types import BacktestConfig

router = APIRouter()


class TimeSeriesPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    value: float


class BacktestConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_equity: float = 1.0
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    max_leverage: float = 1.0


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    price: List[TimeSeriesPoint]
    position: List[TimeSeriesPoint]
    config: Optional[BacktestConfigRequest] = None


class BacktestMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_return: float
    annualized_return: Optional[float]
    annualized_volatility: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]


class BacktestSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: List[TimeSeriesPoint]
    pnl: List[TimeSeriesPoint]
    equity: List[TimeSeriesPoint]
    turnover: List[TimeSeriesPoint]
    costs: List[TimeSeriesPoint]


class BacktestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    created_at: datetime
    metrics: BacktestMetrics
    series: BacktestSeries
    metadata: Dict[str, str]


def _series_from_points(points: List[TimeSeriesPoint]) -> pd.Series:
    if not points:
        raise ValueError("Time series must contain at least one point")
    index = pd.to_datetime([point.timestamp for point in points], utc=True)
    values = [point.value for point in points]
    series = pd.Series(values, index=index).sort_index()
    if series.index.has_duplicates:
        raise ValueError("Time series has duplicate timestamps")
    return series


def _points_from_series(series: pd.Series) -> List[TimeSeriesPoint]:
    return [TimeSeriesPoint(timestamp=idx.to_pydatetime(), value=float(val)) for idx, val in series.items()]


def _compute_metrics(pnl: pd.Series, equity: pd.Series) -> BacktestMetrics:
    if pnl.empty or equity.empty:
        raise ValueError("PNL and equity series must be non-empty")

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if equity.iloc[0] != 0 else 0.0
    pnl_mean = pnl.mean()
    pnl_std = pnl.std(ddof=0)
    annualized_return = float(pnl_mean * 252)
    annualized_volatility = float(pnl_std * sqrt(252)) if pnl_std != 0 else None
    sharpe_ratio = float(annualized_return / annualized_volatility) if annualized_volatility else None

    cumulative = (1 + pnl.fillna(0.0)).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else None

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
    )


@router.post(
    "/backtests",
    response_model=BacktestResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}},
)
def run_backtest_endpoint(
    request: BacktestRequest,
    storage: SQLiteStorage = Depends(get_storage),
) -> BacktestResponse:
    try:
        price_series = _series_from_points(request.price)
        position_series = _series_from_points(request.position)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    config_payload = request.config.model_dump() if request.config else {}
    config = BacktestConfig(**config_payload)

    try:
        result = run_backtest(price_series, position_series, config)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    metrics = _compute_metrics(result.pnl, result.equity)
    series = BacktestSeries(
        returns=_points_from_series(result.returns),
        pnl=_points_from_series(result.pnl),
        equity=_points_from_series(result.equity),
        turnover=_points_from_series(result.turnover),
        costs=_points_from_series(result.costs),
    )

    payload = {
        "metrics": metrics.model_dump(),
        "series": series.model_dump(),
        "metadata": {k: str(v) for k, v in result.metadata.items()},
    }
    record = storage.save_backtest(request.model_dump(), payload)

    return BacktestResponse(
        id=record["id"],
        created_at=datetime.fromisoformat(record["created_at"]),
        metrics=metrics,
        series=series,
        metadata={k: str(v) for k, v in result.metadata.items()},
    )


@router.get("/backtests/{backtest_id}", response_model=BacktestResponse)
def get_backtest(backtest_id: str, storage: SQLiteStorage = Depends(get_storage)) -> BacktestResponse:
    record = storage.get_backtest(backtest_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backtest not found")

    return BacktestResponse(
        id=record["id"],
        created_at=datetime.fromisoformat(record["created_at"]),
        metrics=BacktestMetrics.model_validate(record["result"]["metrics"]),
        series=BacktestSeries.model_validate(record["result"]["series"]),
        metadata=record["result"]["metadata"],
    )
