"""History endpoints for saved backtests and state snapshots."""

from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, Depends

from quant_fx_system.api.deps import SQLiteStorage, get_storage
from quant_fx_system.api.schemas.common import ListResponse, Metadata
from quant_fx_system.api.schemas.quant_state import QuantStateRecord
from quant_fx_system.api.routes.backtest import BacktestResponse

router = APIRouter()


@router.get("/history/backtests", response_model=ListResponse)
def list_backtests(
    limit: int = 25,
    storage: SQLiteStorage = Depends(get_storage),
) -> ListResponse:
    limit = max(1, min(limit, 200))
    records = []
    for item in storage.list_backtests(limit):
        records.append(
            BacktestResponse(
                id=item["id"],
                created_at=datetime.fromisoformat(item["created_at"]),
                metrics=item["result"]["metrics"],
                series=item["result"]["series"],
                metadata=item["result"]["metadata"],
            ).model_dump()
        )
    return ListResponse(data=records, metadata=Metadata(items=len(records), limit=limit))


@router.get("/history/states", response_model=ListResponse)
def list_state_history(
    limit: int = 25,
    storage: SQLiteStorage = Depends(get_storage),
) -> ListResponse:
    limit = max(1, min(limit, 200))
    records = [QuantStateRecord.model_validate(item).model_dump() for item in storage.list_states(limit)]
    return ListResponse(data=records, metadata=Metadata(items=len(records), limit=limit))
