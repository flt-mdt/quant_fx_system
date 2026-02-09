"""Error schemas and helpers."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class APIError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str
    error_code: Optional[str] = None
