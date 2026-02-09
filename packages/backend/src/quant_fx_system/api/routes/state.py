"""State endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from quant_fx_system.api.deps import SQLiteStorage, get_storage
from quant_fx_system.api.schemas.common import ListResponse, Metadata
from quant_fx_system.api.schemas.quant_state import QuantStatePayload, QuantStateRecord

router = APIRouter()


@router.get("/state", response_model=QuantStateRecord)
def get_latest_state(storage: SQLiteStorage = Depends(get_storage)) -> QuantStateRecord:
    record = storage.latest_state()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No state found")
    return QuantStateRecord.model_validate(record)


@router.put("/state", response_model=QuantStateRecord, status_code=status.HTTP_201_CREATED)
def upsert_state(
    payload: QuantStatePayload,
    storage: SQLiteStorage = Depends(get_storage),
) -> QuantStateRecord:
    record = storage.save_state(payload.model_dump())
    return QuantStateRecord.model_validate(record)


@router.get("/states", response_model=ListResponse)
def list_states(
    limit: int = 25,
    storage: SQLiteStorage = Depends(get_storage),
) -> ListResponse:
    limit = max(1, min(limit, 200))
    records = [QuantStateRecord.model_validate(item).model_dump() for item in storage.list_states(limit)]
    return ListResponse(data=records, metadata=Metadata(items=len(records), limit=limit))
