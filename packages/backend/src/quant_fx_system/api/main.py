"""FastAPI application factory for the Quant FX System backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quant_fx_system.api.routes import backtest, health, history, state
from quant_fx_system.logging_config import configure_logging
from quant_fx_system.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.api_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    api_prefix = f"/api/{settings.api_version}"
    app.include_router(health.router, prefix=api_prefix, tags=["health"])
    app.include_router(state.router, prefix=api_prefix, tags=["state"])
    app.include_router(backtest.router, prefix=api_prefix, tags=["backtest"])
    app.include_router(history.router, prefix=api_prefix, tags=["history"])

    return app


app = create_app()
