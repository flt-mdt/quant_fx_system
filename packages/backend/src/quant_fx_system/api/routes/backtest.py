"""Backtest endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from uuid import uuid4
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from quant_fx_system.api.deps import SQLiteStorage, get_storage
from quant_fx_system.api.schemas.common import ErrorResponse
from quant_fx_system.quant.backtest.engine import run_backtest
from quant_fx_system.quant.backtest.types import BacktestConfig

router = APIRouter()


class BacktestDatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamps_utc: list[datetime]
    prices: list[float]
    signals: list[float]
    input_type: Literal["position_target"] = "position_target"


class BacktestConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_cash: float = 100_000.0
    execution: Literal["next_bar"] = "next_bar"
    return_type: Literal["simple"] = "simple"
    pnl_convention: str = "price_times_position"
    costs_alignment: str = "trade_timestamp"
    annualization_factor: int = 252
    fee_bps: float = 0.0
    slippage_bps: float = 0.0


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    dataset: BacktestDatasetRequest
    config: BacktestConfigRequest = BacktestConfigRequest()


class BacktestSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bars: int
    initial_cash: float
    final_equity: float
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    turnover: float
    total_costs: float


class BacktestSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamps_utc: list[str]
    price: list[float]
    signal: list[float]
    position: list[float]
    trade_qty: list[float]
    gross_return: list[float]
    cost_return: list[float]
    net_return: list[float]
    equity: list[float]
    drawdown: list[float]
    cum_costs: list[float]


class DebugCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: Literal["ok", "warn", "error"]
    value: float
    threshold: float


class DebugEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    timestamp_utc: str
    type: Literal["signal_flip", "cost_spike", "drawdown_new_low"]
    signal_prev: float
    signal_new: float
    position_prev: float
    position_new: float
    trade_qty: float
    cost_applied: float
    note: str


class DebugMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    checks: list[DebugCheck]
    events: list[DebugEvent]


class ConfigUsedMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution: str
    return_type: str
    pnl_convention: str
    costs_alignment: str
    annualization_factor: int


class DataQualityMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_count: int
    duplicate_timestamps: int
    non_monotonic_timestamps: int
    nan_prices: int
    nan_signals: int


class BacktestMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_used: ConfigUsedMetadata
    data_quality: DataQualityMetadata
    debug: DebugMetadata | None = None


class BacktestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    created_at_utc: str
    status: str
    summary: BacktestSummary
    series: BacktestSeries
    metadata: BacktestMetadata


def _safe_number(value: float | int | Any) -> float:
    number = float(value)
    if not isfinite(number):
        return 0.0
    return number


def _count_non_monotonic(timestamps: pd.DatetimeIndex) -> int:
    if len(timestamps) <= 1:
        return 0
    diffs = pd.Series(timestamps).diff().dropna()
    return int((diffs <= pd.Timedelta(0)).sum())


def _dataset_to_series(dataset: BacktestDatasetRequest) -> tuple[pd.Series, pd.Series, DataQualityMetadata]:
    timestamps = pd.to_datetime(dataset.timestamps_utc, utc=True)
    if len(timestamps) == 0:
        raise ValueError("dataset.timestamps_utc must not be empty")
    if len(dataset.prices) != len(timestamps) or len(dataset.signals) != len(timestamps):
        raise ValueError("dataset.timestamps_utc, dataset.prices and dataset.signals must have identical lengths")
    if dataset.input_type != "position_target":
        raise ValueError("dataset.input_type must be 'position_target'")

    duplicate_timestamps = int(pd.Index(timestamps).duplicated().sum())
    non_monotonic_timestamps = _count_non_monotonic(timestamps)
    nan_prices = int(pd.Series(dataset.prices, dtype="float64").isna().sum())
    nan_signals = int(pd.Series(dataset.signals, dtype="float64").isna().sum())

    if duplicate_timestamps > 0:
        raise ValueError("dataset.timestamps_utc contains duplicate timestamps")
    if non_monotonic_timestamps > 0:
        raise ValueError("dataset.timestamps_utc must be strictly increasing")

    price = pd.Series(dataset.prices, index=timestamps, dtype="float64")
    signal = pd.Series(dataset.signals, index=timestamps, dtype="float64")

    if not pd.Series(pd.Series(price).apply(isfinite)).all():
        raise ValueError("dataset.prices contains non-finite values")
    if not pd.Series(pd.Series(signal).apply(isfinite)).all():
        raise ValueError("dataset.signals contains non-finite values")

    quality = DataQualityMetadata(
        timestamp_count=int(len(timestamps)),
        duplicate_timestamps=duplicate_timestamps,
        non_monotonic_timestamps=non_monotonic_timestamps,
        nan_prices=nan_prices,
        nan_signals=nan_signals,
    )
    return price, signal, quality


def _compute_cagr(initial_cash: float, final_equity: float, bars: int, annualization_factor: int) -> float:
    if bars <= 0 or annualization_factor <= 0 or initial_cash <= 0:
        return 0.0
    years = bars / annualization_factor
    if years <= 0:
        return 0.0
    ratio = final_equity / initial_cash if initial_cash else 0.0
    if ratio <= 0:
        return -1.0
    return _safe_number(ratio ** (1.0 / years) - 1.0)


def _build_debug(
    timestamps: pd.DatetimeIndex,
    signal: pd.Series,
    position: pd.Series,
    turnover: pd.Series,
    costs: pd.Series,
    drawdown: pd.Series,
    summary: BacktestSummary,
) -> DebugMetadata:
    position_change = position.diff().fillna(0.0)
    turnover_mismatch = (turnover > 0) & (position_change.abs() <= 1e-12)
    costs_without_trade = (costs > 0) & (turnover <= 1e-12)
    equity_nan_count = 0.0

    checks = [
        DebugCheck(
            name="turnover_nonzero_with_position_change",
            status="warn" if bool(turnover_mismatch.any()) else "ok",
            value=float(turnover_mismatch.sum()),
            threshold=0.0,
        ),
        DebugCheck(
            name="costs_applied_on_trade",
            status="warn" if bool(costs_without_trade.any()) else "ok",
            value=float(costs_without_trade.sum()),
            threshold=0.0,
        ),
        DebugCheck(
            name="equity_nan_count",
            status="ok",
            value=equity_nan_count,
            threshold=0.0,
        ),
    ]

    events: list[DebugEvent] = []
    max_cost = float(costs.max()) if not costs.empty else 0.0
    cost_spike_threshold = max(max_cost * 0.8, 1e-12)
    min_drawdown_so_far = 0.0

    for idx, timestamp in enumerate(timestamps):
        prev_signal = float(signal.iloc[idx - 1]) if idx > 0 else float(signal.iloc[idx])
        current_signal = float(signal.iloc[idx])
        prev_position = float(position.iloc[idx - 1]) if idx > 0 else float(position.iloc[idx])
        current_position = float(position.iloc[idx])
        trade_qty = float(turnover.iloc[idx])
        cost_applied = float(costs.iloc[idx])
        dd = float(drawdown.iloc[idx])

        if idx > 0 and current_signal != prev_signal:
            events.append(
                DebugEvent(
                    index=idx,
                    timestamp_utc=timestamp.isoformat().replace("+00:00", "Z"),
                    type="signal_flip",
                    signal_prev=prev_signal,
                    signal_new=current_signal,
                    position_prev=prev_position,
                    position_new=current_position,
                    trade_qty=trade_qty,
                    cost_applied=cost_applied,
                    note="Signal changed between consecutive bars.",
                )
            )

        if cost_applied >= cost_spike_threshold and cost_applied > 0:
            events.append(
                DebugEvent(
                    index=idx,
                    timestamp_utc=timestamp.isoformat().replace("+00:00", "Z"),
                    type="cost_spike",
                    signal_prev=prev_signal,
                    signal_new=current_signal,
                    position_prev=prev_position,
                    position_new=current_position,
                    trade_qty=trade_qty,
                    cost_applied=cost_applied,
                    note="High execution costs detected on this bar.",
                )
            )

        if dd < min_drawdown_so_far:
            min_drawdown_so_far = dd
            events.append(
                DebugEvent(
                    index=idx,
                    timestamp_utc=timestamp.isoformat().replace("+00:00", "Z"),
                    type="drawdown_new_low",
                    signal_prev=prev_signal,
                    signal_new=current_signal,
                    position_prev=prev_position,
                    position_new=current_position,
                    trade_qty=trade_qty,
                    cost_applied=cost_applied,
                    note=f"New drawdown low reached ({summary.max_drawdown:.6f}).",
                )
            )

    return DebugMetadata(enabled=True, checks=checks, events=events)


def _build_response_payload(
    *,
    record_id: str,
    created_at: str,
    request: BacktestRequest,
    result: Any,
    quality: DataQualityMetadata,
) -> BacktestResponse:
    gross_return = (result.position * result.returns).fillna(0.0)
    drawdown = (result.equity / result.equity.cummax() - 1.0).fillna(0.0)
    cum_costs = result.costs.cumsum().fillna(0.0)

    index = result.returns.index
    annualization_factor = request.config.annualization_factor

    bars = int(len(index))
    final_equity = _safe_number(result.equity.iloc[-1]) if bars else _safe_number(request.config.initial_cash)
    total_return = _safe_number(final_equity / request.config.initial_cash - 1.0) if request.config.initial_cash else 0.0
    volatility = _safe_number(result.pnl.std(ddof=0) * (annualization_factor**0.5)) if bars else 0.0
    sharpe = _safe_number((result.pnl.mean() * annualization_factor) / volatility) if volatility != 0 else 0.0

    summary = BacktestSummary(
        bars=bars,
        initial_cash=_safe_number(request.config.initial_cash),
        final_equity=final_equity,
        total_return=total_return,
        cagr=_compute_cagr(request.config.initial_cash, final_equity, bars, annualization_factor),
        volatility=volatility,
        sharpe=sharpe,
        max_drawdown=_safe_number(drawdown.min()) if bars else 0.0,
        turnover=_safe_number(result.turnover.sum()) if bars else 0.0,
        total_costs=_safe_number(result.costs.sum()) if bars else 0.0,
    )

    signal = request.dataset.signals
    price = request.dataset.prices
    aligned_signal = pd.Series(signal, index=pd.to_datetime(request.dataset.timestamps_utc, utc=True)).reindex(index)
    aligned_price = pd.Series(price, index=pd.to_datetime(request.dataset.timestamps_utc, utc=True)).reindex(index)

    def to_list(series: pd.Series) -> list[float]:
        return [_safe_number(v) for v in series.fillna(0.0).tolist()]

    series_payload = BacktestSeries(
        timestamps_utc=[ts.isoformat().replace("+00:00", "Z") for ts in index],
        price=to_list(aligned_price),
        signal=to_list(aligned_signal),
        position=to_list(result.position),
        trade_qty=to_list(result.turnover),
        gross_return=to_list(gross_return),
        cost_return=to_list(result.costs),
        net_return=to_list(result.pnl),
        equity=to_list(result.equity),
        drawdown=to_list(drawdown),
        cum_costs=to_list(cum_costs),
    )

    metadata = BacktestMetadata(
        config_used=ConfigUsedMetadata(
            execution=str(result.metadata.get("execution", request.config.execution)),
            return_type=str(result.metadata.get("return_type", request.config.return_type)),
            pnl_convention=str(result.metadata.get("pnl_convention", request.config.pnl_convention)),
            costs_alignment=str(result.metadata.get("costs_alignment", request.config.costs_alignment)),
            annualization_factor=request.config.annualization_factor,
        ),
        data_quality=quality,
        debug=_build_debug(
            timestamps=index,
            signal=pd.Series(series_payload.signal, index=index),
            position=pd.Series(series_payload.position, index=index),
            turnover=pd.Series(series_payload.trade_qty, index=index),
            costs=pd.Series(series_payload.cost_return, index=index),
            drawdown=pd.Series(series_payload.drawdown, index=index),
            summary=summary,
        ),
    )

    return BacktestResponse(
        id=record_id,
        name=request.name,
        created_at_utc=created_at,
        status="completed",
        summary=summary,
        series=series_payload,
        metadata=metadata,
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
        price_series, signal_series, quality = _dataset_to_series(request.dataset)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    config = BacktestConfig(
        initial_equity=request.config.initial_cash,
        transaction_cost_bps=request.config.fee_bps,
        slippage_bps=request.config.slippage_bps,
        execution=request.config.execution,
        return_type=request.config.return_type,
    )

    try:
        result = run_backtest(price_series, signal_series, config)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    record_id = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    response = _build_response_payload(
        record_id=record_id,
        created_at=created_at,
        request=request,
        result=result,
        quality=quality,
    )
    storage.save_backtest(
        request.model_dump(mode="json"),
        response.model_dump(mode="json"),
        record_id=record_id,
        created_at=created_at,
    )
    return response


@router.get("/backtests/{backtest_id}", response_model=BacktestResponse)
def get_backtest(backtest_id: str, storage: SQLiteStorage = Depends(get_storage)) -> BacktestResponse:
    record = storage.get_backtest(backtest_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backtest not found")

    return BacktestResponse.model_validate(record["result"])
