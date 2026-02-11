# API Contract

## Strategy endpoints (Excel-first)

### `POST /api/v1/datasets/excel:ingest`
Request fields:
- `rows[]`: `{ timestamp_utc, price }`
- `duplicate_policy`: `keep_last | fail`
- `price_type`: `close | mid | bid_ask_mid`
- `frequency`: e.g. `1D`
- `ohlc_available`: boolean
- `missing_data_policy`: `keep_sparse | resample`

Response fields:
- `dataset_id`, `created_at_utc`
- `price_type_used`, `frequency_inferred`
- `ohlc_available`, `missing_data_policy`
- `index_range { start, end, bars }`
- `quality { rows_in, rows_clean, dropped_rows, duplicate_rows_detected, invalid_price_rows, invalid_price_details }`
- `canonical_preview { timestamps_utc, price }`
- `metadata`

### `POST /api/v1/strategy-runs`
Request additions:
- `meta_model { enabled, model_type, feature_selection, horizon }`
- `regimes { enabled }`
- `information { enabled }`

Response high-level fields:
- `run_id`, `name`, `status`, `created_at_utc`
- `artifacts_ref { dataset_hash, features_hash, run_hash }`
- `signals_pack`
- `meta_model` (enabled path returns `p_follow` summary + diagnostics)
- `regimes`
- `information`
- `risk_overlay`
- `position_target`
- `backtest { summary, series, engine_metadata }`
- `evaluation { summary, metadata }`
- `diagnostics`
- `metadata { determinism, research_hygiene }`

### `GET /api/v1/strategy-runs/{run_id}`
Returns persisted run artifact.

Query params for series control:
- `summary_only=true|false`
- `stride=<int>=1`
- `start=<UTC timestamp>`
- `end=<UTC timestamp>`

### `GET /api/v1/strategy-runs/{run_id}/diagnostics`
Returns diagnostics block only.

### `GET /api/v1/strategy-runs/{run_id}/trace?timestamp=...`
Returns per-timestamp decision trace:
- `feature_snapshot`
- `signal_contributions`
- `meta_decision`
- `risk_adjustments`
- `position_target`
- `position_applied`

### `GET /api/v1/strategy-runs/capabilities`
Returns machine-readable support flags for meta-model/risk/regimes/information/trace.

## Payload guidance
- Full run payloads can be large because they include arrays.
- Prefer summary mode + diagnostics + trace for default dashboards.

## Compatibility
- Base API version: `/api/v1`.
- Additive field changes are backward compatible.
- Breaking changes require `/api/v2`.
