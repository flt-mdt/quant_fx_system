from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from quant_fx_system.api.deps import SQLiteStorage, get_storage
from quant_fx_system.api.main import create_app


def _rows() -> list[dict[str, object]]:
    rows = []
    for i in range(80):
        day = i + 1
        month = 1 + (day - 1) // 28
        day_in_month = ((day - 1) % 28) + 1
        rows.append(
            {
                "timestamp_utc": f"2024-{month:02d}-{day_in_month:02d}T00:00:00Z",
                "price": 1.10 + i * 0.002 + ((-1) ** i) * 0.001,
            }
        )
    return rows


def _client(tmp_path: Path) -> TestClient:
    app = create_app()
    storage = SQLiteStorage(tmp_path / "strategy.db")
    app.dependency_overrides[get_storage] = lambda: storage
    return TestClient(app)


def _dataset_payload(rows: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "rows": rows or _rows(),
        "duplicate_policy": "keep_last",
        "price_type": "close",
        "frequency": "1D",
        "ohlc_available": False,
        "missing_data_policy": "resample",
    }


def test_strategy_run_core_endpoints_and_diagnostics(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        ingest = client.post("/api/v1/datasets/excel:ingest", json=_dataset_payload())
        assert ingest.status_code == 201

        run = client.post("/api/v1/strategy-runs", json={"name": "run", "dataset": _dataset_payload()})
        assert run.status_code == 201
        body = run.json()
        run_id = body["run_id"]

        diag = client.get(f"/api/v1/strategy-runs/{run_id}/diagnostics")
        assert diag.status_code == 200
        d = diag.json()
        assert "turnover_spikes" in d["turnover_issues"]
        assert "stage_counts" in d["overlap"]
        assert "drop_reasons" in d["overlap"]
        assert d["risk_capabilities"]["var_es_supported"] is False
        assert "warnings" in d["leakage_guards"]

        ts = body["position_target"]["timestamps_utc"][-1]
        trace = client.get(f"/api/v1/strategy-runs/{run_id}/trace", params={"timestamp": ts})
        assert trace.status_code == 200
        assert trace.json()["timestamp_utc"] == ts


def test_duplicate_policy_fail_and_invalid_price(tmp_path: Path) -> None:
    rows = _rows()
    rows.append(rows[-1])
    payload = _dataset_payload(rows)
    payload["duplicate_policy"] = "fail"

    with _client(tmp_path) as client:
        dup = client.post("/api/v1/datasets/excel:ingest", json=payload)
        assert dup.status_code == 400

        bad_rows = _rows()
        bad_rows[3]["price"] = -1.0
        bad = client.post("/api/v1/datasets/excel:ingest", json=_dataset_payload(bad_rows))
        assert bad.status_code == 400
        assert "invalid_price_rows" in bad.json()["detail"]


def test_meta_model_enabled_happy_path_and_unsupported_path(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        ok = client.post(
            "/api/v1/strategy-runs",
            json={
                "name": "meta-ok",
                "dataset": _dataset_payload(),
                "meta_model": {
                    "enabled": True,
                    "model_type": "logistic",
                    "feature_selection": "none",
                    "horizon": 2,
                },
            },
        )
        assert ok.status_code == 201
        body = ok.json()
        assert body["meta_model"]["enabled"] is True
        assert "p_follow_mean" in body["meta_model"]
        assert "diagnostics" in body["meta_model"]

        bad = client.post(
            "/api/v1/strategy-runs",
            json={
                "name": "meta-bad",
                "dataset": _dataset_payload(),
                "meta_model": {
                    "enabled": True,
                    "model_type": "xgb",
                    "feature_selection": "none",
                    "horizon": 2,
                },
            },
        )
        assert bad.status_code == 400
        assert "unsupported" in bad.json()["detail"]


def test_payload_controls_and_research_hygiene_determinism(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        r1 = client.post("/api/v1/strategy-runs", json={"name": "det", "dataset": _dataset_payload()})
        assert r1.status_code == 201
        b1 = r1.json()
        run_id = b1["run_id"]

        summary = client.get(f"/api/v1/strategy-runs/{run_id}", params={"summary_only": True})
        assert summary.status_code == 200
        s = summary.json()
        assert s["position_target"]["timestamps_utc"] == []

        sliced = client.get(
            f"/api/v1/strategy-runs/{run_id}",
            params={"stride": 3, "start": b1["position_target"]["timestamps_utc"][5], "end": b1["position_target"]["timestamps_utc"][-1]},
        )
        assert sliced.status_code == 200
        sl = sliced.json()
        assert len(sl["position_target"]["timestamps_utc"]) < len(b1["position_target"]["timestamps_utc"])

        r2 = client.post("/api/v1/strategy-runs", json={"name": "det", "dataset": _dataset_payload()})
        assert r2.status_code == 201
        b2 = r2.json()

        h1 = b1["metadata"]["research_hygiene"]
        assert "walk_forward_present" in h1
        assert "delay_test_present" in h1
        assert h1["walk_forward_present"]["reason"]
        assert b1["metadata"]["determinism"]["reproducibility_hash"] == b2["metadata"]["determinism"]["reproducibility_hash"]


def test_capabilities_endpoint(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        cap = client.get("/api/v1/strategy-runs/capabilities")
        assert cap.status_code == 200
        body = cap.json()
        assert body["meta_model"]["supported"] is True
        assert body["risk"]["var_es_supported"] is False


def test_to_json_safe_normalizes_numpy_keys_and_values() -> None:
    from quant_fx_system.api.routes.strategy_runs import _to_json_safe

    payload = {np.int64(1): {"nested": np.int64(2)}, "ok": [np.float64(1.5)]}
    sanitized = _to_json_safe(payload)

    assert 1 in sanitized
    assert sanitized[1]["nested"] == 2
    assert isinstance(sanitized[1]["nested"], int)
    assert sanitized["ok"] == [1.5]


def test_strategy_run_response_serializes_numpy_scalars_in_any_payload(tmp_path: Path, monkeypatch) -> None:
    from quant_fx_system.api.routes import strategy_runs as strategy_runs_route

    original = strategy_runs_route.run_signal_pipeline

    def _run_signal_pipeline_with_numpy_metadata(*args, **kwargs):
        payload = original(*args, **kwargs)
        payload["metadata"]["numpy_scalar"] = np.int64(7)
        return payload

    monkeypatch.setattr(strategy_runs_route, "run_signal_pipeline", _run_signal_pipeline_with_numpy_metadata)

    with _client(tmp_path) as client:
        run = client.post(
            "/api/v1/strategy-runs",
            json={"name": "numpy-safe", "dataset": _dataset_payload()},
        )
        assert run.status_code == 201
        body = run.json()
        assert body["metadata"]["inference"]["numpy_scalar"] == 7
        assert isinstance(body["metadata"]["inference"]["numpy_scalar"], int)
