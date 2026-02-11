# Quant FX System

A monorepo for a quantitative EUR/USD trading system with a FastAPI backend and React frontend.

## Architecture (High-level)
- Core quant logic in `packages/backend/src/quant_fx_system/quant/`.
- Backend layered structure:
  - `domain/` (entities + ports, partly placeholder)
  - `application/` (orchestration services)
  - `infrastructure/` (adapters)
  - `api/` (FastAPI routes + schemas)

## API Contract Overview (current)
- `GET /api/v1/health`
- `GET /api/v1/state`
- `PUT /api/v1/state`
- `GET /api/v1/states`
- `POST /api/v1/backtests`
- `GET /api/v1/backtests/{backtest_id}`
- `GET /api/v1/history/backtests`
- `GET /api/v1/history/states`
- `POST /api/v1/datasets/excel:ingest`
- `POST /api/v1/strategy-runs`
- `GET /api/v1/strategy-runs/{run_id}`
- `GET /api/v1/strategy-runs/{run_id}/diagnostics`
- `GET /api/v1/strategy-runs/{run_id}/trace?timestamp=...`
- `GET /api/v1/strategy-runs/capabilities`

## Excel-first strategy run flow
`prices -> clean/align -> features -> signals_pack -> optional regimes/information/meta -> risk overlay -> position_target -> backtest accounting -> evaluation`

### Frontend handoff notes
Backend returns stable high-level fields for frontend integration:
- `signals_pack`
- `position_target`
- `backtest.summary`, `backtest.series`, `backtest.engine_metadata`
- `evaluation.summary`
- `diagnostics`
- `metadata.research_hygiene`

Series payload controls are available on run retrieval (`summary_only`, `stride`, `start`, `end`).

## Known limitations
- Var/ES risk overlay is not implemented yet (explicit capability flag).
- Some domain/infrastructure modules remain placeholders.
- TP/SL intrabar simulation is not supported for close-only daily inputs.

## Frontend integration package
- Strategy-run handoff guide: `docs/frontend-handoff-strategy-runs.md`
- API contract details: `docs/api-contract.md`
