"""Application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field("Quant FX System API", description="Public name of the API")
    api_version: str = Field("v1", description="API version prefix")
    log_level: str = Field("INFO", description="Logging level")
    environment: str = Field("production", description="Deployment environment")
    data_dir: Path = Field(Path("/data"), description="Storage directory for state")
    cors_origins: List[str] = Field(
        default_factory=lambda: ["https://quant-fx-backtest.lovable.app"],
        description="CORS origins",
    )

    model_config = SettingsConfigDict(
        env_prefix="QFX_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def database_path(self) -> Path:
        return self.data_dir / "quant_fx.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
