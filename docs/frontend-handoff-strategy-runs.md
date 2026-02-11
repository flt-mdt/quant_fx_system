# Frontend Handoff — Strategy Runs API

Integration contract for frontend teams consuming Excel-first strategy orchestration.

## API versioning policy
- Current base path: `/api/v1`.
- Backward compatibility target:
  - additive fields are non-breaking,
  - field removals/renames require version bump (`/api/v2`).
- Clients should ignore unknown fields.

## Endpoints
- `POST /api/v1/datasets/excel:ingest`
- `POST /api/v1/strategy-runs`
- `GET /api/v1/strategy-runs/{run_id}`
- `GET /api/v1/strategy-runs/{run_id}/diagnostics`
- `GET /api/v1/strategy-runs/{run_id}/trace?timestamp=...`
- `GET /api/v1/strategy-runs/capabilities`

## Field-level contract (diagnostics)
- `turnover_issues.turnover_raw`: turnover before risk turnover controls.
- `turnover_issues.turnover_after`: turnover after controls.
- `turnover_issues.turnover_spikes`: count above p95 threshold.
- `overlap.stage_counts`: row counts per stage (`input_rows`, `feature_rows`, ...).
- `overlap.drop_reasons`: explicit drop attribution by stage.
- `units.position_unit`: leverage units for `position_target` and `position_applied`.
- `leakage_guards.*`: shift/lag metadata and warning flags.
- `irregular_sampling.gap_histogram`: observed spacing distribution.

## Meta-model and risk capability probing
Use `GET /api/v1/strategy-runs/capabilities` at frontend init:
- enable/disable meta-model panels from `meta_model.supported`
- show risk feature availability from `risk.var_es_supported`

## Series payload controls
`GET /api/v1/strategy-runs/{run_id}` query params:
- `summary_only=true` returns compact dashboard payload
- `stride=N` down-samples time series for charts
- `start`, `end` bound series windows

## Payload size guidance
- Full run payload includes full arrays and can be large.
- Default dashboard calls:
  1) `/strategy-runs/{id}?summary_only=true`
  2) `/strategy-runs/{id}/diagnostics`
  3) `/strategy-runs/{id}/trace?timestamp=...` for drilldown

## Migration notes
- Legacy clients expecting old diagnostics keys should migrate to:
  - `overlap.*` (replacing previous timestamp overlap block shape)
  - `risk_capabilities.var_es_supported` explicit flag
  - `metadata.research_hygiene` machine-readable checklist
