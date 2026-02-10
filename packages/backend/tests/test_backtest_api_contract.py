from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from quant_fx_system.api.deps import SQLiteStorage, get_storage
from quant_fx_system.api.main import create_app


def _payload() -> dict:
    return {
        "name": "Backtest 2026-02-10",
        "dataset": {
            "timestamps_utc": [
                "2015-03-30T17:04:35Z",
                "2015-03-31T17:04:35Z",
                "2015-04-01T17:04:35Z",
                "2015-04-04T17:04:35Z",
            ],
            "prices": [1.0763, 1.0880, 1.0969, 1.0922],
            "signals": [1, -1, -1, 1],
        },
        "config": {
            "initial_cash": 100000,
            "execution": "next_bar",
            "return_type": "simple",
            "pnl_convention": "price_times_position",
            "costs_alignment": "trade_timestamp",
            "annualization_factor": 252,
            "fee_bps": 2,
            "slippage_bps": 1,
        },
    }


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def test_backtest_contract_matches_front_requirements(tmp_path: Path):
    app = create_app()
    storage = SQLiteStorage(tmp_path / "api_contract.db")
    app.dependency_overrides[get_storage] = lambda: storage

    with TestClient(app) as client:
        response = client.post("/api/v1/backtests", json=_payload())
        assert response.status_code == 201

        body = response.json()
        assert body["name"] == "Backtest 2026-02-10"
        assert body["status"] == "completed"
        assert isinstance(body["created_at_utc"], str)

        summary = body["summary"]
        expected_summary_fields = {
            "bars",
            "initial_cash",
            "final_equity",
            "total_return",
            "cagr",
            "volatility",
            "sharpe",
            "max_drawdown",
            "turnover",
            "total_costs",
        }
        assert set(summary.keys()) == expected_summary_fields
        for key in expected_summary_fields - {"bars"}:
            assert _is_number(summary[key])

        series = body["series"]
        expected_series_fields = {
            "timestamps_utc",
            "price",
            "signal",
            "position",
            "trade_qty",
            "gross_return",
            "cost_return",
            "net_return",
            "equity",
            "drawdown",
            "cum_costs",
        }
        assert set(series.keys()) == expected_series_fields

        bar_count = len(series["timestamps_utc"])
        assert summary["bars"] == bar_count
        for key in expected_series_fields - {"timestamps_utc"}:
            assert len(series[key]) == bar_count
            assert all(_is_number(x) for x in series[key])

        debug = body["metadata"]["debug"]
        for check in debug["checks"]:
            assert check["status"] in {"ok", "warn", "error"}
        for event in debug["events"]:
            assert event["type"] in {"signal_flip", "cost_spike", "drawdown_new_low"}

        backtest_id = body["id"]
        get_response = client.get(f"/api/v1/backtests/{backtest_id}")
        assert get_response.status_code == 200
        assert get_response.json() == body


def test_backtest_rejects_non_monotonic_timestamps(tmp_path: Path):
    app = create_app()
    storage = SQLiteStorage(tmp_path / "api_contract_invalid.db")
    app.dependency_overrides[get_storage] = lambda: storage

    payload = _payload()
    payload["dataset"]["timestamps_utc"] = [
        "2015-03-30T17:04:35Z",
        "2015-03-29T17:04:35Z",
        "2015-04-01T17:04:35Z",
        "2015-04-04T17:04:35Z",
    ]

    with TestClient(app) as client:
        response = client.post("/api/v1/backtests", json=payload)
        assert response.status_code == 400
        assert "strictly increasing" in response.json()["detail"]
