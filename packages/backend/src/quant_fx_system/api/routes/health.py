"""Health endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from quant_fx_system.api.schemas.common import StatusResponse
from quant_fx_system.settings import get_settings

router = APIRouter()


@router.get("/health", response_model=StatusResponse)
def health_check() -> StatusResponse:
    settings = get_settings()
    return StatusResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc),
        version=settings.api_version,
        environment=settings.environment,
    )
