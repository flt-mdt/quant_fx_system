"""Common API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class APIBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class StatusResponse(APIBaseModel):
    status: str = Field(..., description="Status indicator")
    timestamp: datetime = Field(..., description="Server time in UTC")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Deployment environment")


class Metadata(APIBaseModel):
    items: int = Field(..., ge=0, description="Number of items returned")
    limit: int = Field(..., ge=1, description="Limit applied to the query")


class ListResponse(APIBaseModel):
    data: List[Dict[str, Any]]
    metadata: Metadata


class ErrorResponse(APIBaseModel):
    detail: str
    error_code: Optional[str] = None
