# Quant FX System

A monorepo skeleton for a quantitative EUR/USD trading system. This repository is intentionally **structure-only**: it provides a clean, DDD-ish architecture for both backend and frontend, with placeholders for future implementation.

## Goals
- Separate **core quant** logic from delivery mechanisms (API/UI/CLI).
- Provide a deployable FastAPI backend (Render-friendly) and a React + TypeScript dashboard.
- Establish strong conventions early (linting, typing, testing, docs).

## Architecture (High-level)
- **Core quant** lives in `packages/backend/src/quant_fx_system/quant/`.
- **DDD-ish layers** for the backend:
  - `domain/` (entities + ports)
  - `application/` (use-cases orchestration)
  - `infrastructure/` (adapters to real services)
  - `api/` (FastAPI delivery)
- **Frontend UI** follows DDD-ish boundaries:
  - `app/` (composition)
  - `domains/` (bounded contexts)
  - `shared/` (cross-cutting UI + API utilities)
  - `pages/` (route-level screens)

## Monorepo Layout
```
quant-fx-system/
├── packages/
│   ├── backend/     # FastAPI + core quant (Python)
│   └── frontend/    # React + TS dashboard (Vite)
├── docs/            # architecture, API contracts, ADRs
├── config/          # YAML config placeholders
├── infra/           # deployment notes (Render) + optional nginx
├── data/            # raw/processed/features (gitkept)
└── scripts/         # repo bootstrap utilities
```

## Backend (FastAPI, Render)
- **Local run (placeholder):**
  - `make backend-dev`
- **Render build/start (placeholder):**
  - See `packages/backend/render-start.sh` and `infra/render/README.md`.

## Frontend (React + Vite)
- **Local run (placeholder):**
  - `pnpm dev`

## API Contract Overview
Planned endpoints:
- `GET /health`
- `GET /state/latest`
- `GET /state/history?from=...&to=...`
- `POST /backtest/run`
- `GET /backtest/{id}`

See `docs/api-contract.md` for the initial `QuantState` JSON schema.

## Environment Variables
- Root `.env.example` documents shared defaults.
- Backend and frontend each provide their own `.env.example` with scoped settings.

## Coding Standards
- **Backend:** ruff, mypy, pytest (Python 3.11+, src layout).
- **Frontend:** strict TypeScript, linting/formatting via tooling placeholders.
- **CI:** `.github/workflows/ci.yml` is a placeholder for lint/test pipelines.

## Live “State” Flow (planned)
A background job periodically computes:
`features → signals → regimes → decay → meta-proba → decision`

It writes a **QuantState** JSON snapshot to a state store (filesystem/SQLite).
The UI polls `/state/latest` and `/state/history` every 30–60s.

---

This repository is a **skeleton only**. All files are placeholders with minimal headers.
