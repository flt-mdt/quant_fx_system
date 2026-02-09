#!/usr/bin/env bash
set -euo pipefail

APP_MODULE="quant_fx_system.api.main:app"
HOST="0.0.0.0"
PORT="${PORT:-8000}"
LOG_LEVEL="${QFX_LOG_LEVEL:-info}"

exec uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
