"""Excel-first strategy orchestration endpoints."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from statistics import median
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from quant_fx_system.api.deps import SQLiteStorage, get_storage
from quant_fx_system.application.services.build_features import build_features_from_prices
from quant_fx_system.application.services.run_backtest import run_accounting_and_evaluation
from quant_fx_system.application.services.run_inference import run_signal_pipeline
from quant_fx_system.quant.backtest import BacktestConfig
from quant_fx_system.quant.data_pipeline.clean_align import CleanAlignConfig
from quant_fx_system.quant.data_pipeline.feature_engineering import FeatureConfig
from quant_fx_system.quant.evaluation import EvaluationConfig
from quant_fx_system.quant.information import InformationConfig, build_information_report
from quant_fx_system.quant.meta_model import MetaModelConfig, fit_meta_model, predict_meta_model
from quant_fx_system.quant.meta_model.types import FeatureSelection, ModelType
from quant_fx_system.quant.regimes import RegimeConfig, infer_regimes
from quant_fx_system.quant.risk import RiskConfig

router = APIRouter()

SUPPORTED_META_MODEL_TYPES: set[str] = {"logistic", "ridge"}
SUPPORTED_META_FEATURE_SELECTION: set[str] = {"none"}


class PriceRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_utc: datetime
    price: float


class MetaModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    enabled: bool = False
    model_type: ModelType = "logistic"
    feature_selection: FeatureSelection = "none"
    horizon: int = 2


class RegimeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class InformationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class ExcelIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[PriceRow] = Field(min_length=2)
    duplicate_policy: Literal["keep_last", "fail"] = "keep_last"
    price_type: Literal["close", "mid", "bid_ask_mid"] = "close"
    frequency: str = "1D"
    ohlc_available: bool = False
    missing_data_policy: Literal["keep_sparse", "resample"] = "resample"


class StrategyRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    dataset: ExcelIngestRequest
    meta_model: MetaModelRequest = MetaModelRequest()
    regimes: RegimeRequest = RegimeRequest()
    information: InformationRequest = InformationRequest()


class IndexRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str
    end: str
    bars: int


class IngestQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows_in: int
    rows_clean: int
    dropped_rows: int
    duplicate_policy: str
    duplicate_rows_detected: int
    invalid_price_rows: int
    invalid_price_details: list[dict[str, Any]]


class CanonicalPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamps_utc: list[str]
    price: list[float]


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    created_at_utc: str
    price_type_used: str
    frequency_inferred: str
    ohlc_available: bool
    missing_data_policy: str
    index_range: IndexRange
    quality: IngestQuality
    canonical_preview: CanonicalPreview
    metadata: dict[str, Any]


class TurnoverDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turnover_raw: float
    turnover_after: float
    turnover_spikes: int
    costs_without_trade_count: int


class OverlapDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_counts: dict[str, int]
    overlap_ratio_position_vs_backtest: float
    dropped_index_intervals: list[dict[str, str]]
    drop_reasons: dict[str, int]


class UnitDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_unit: str
    position_unit: str
    return_unit: str
    cost_unit: str
    leverage_breach_count: int


class LeakageDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_shift: int
    feature_shift: int
    output_shift: int
    effective_lag: int
    warnings: list[str]


class IrregularSamplingDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frequency: str
    annualization_method: int | None
    gap_histogram: dict[str, int]
    irregularity_score: float


class SensitivityDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cost_delay_test_available: bool
    notes: str


class RiskCapabilitiesDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    var_es_supported: bool
    var_es_reason: str


class StrategyRunDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turnover_issues: TurnoverDiagnostics
    overlap: OverlapDiagnostics
    units: UnitDiagnostics
    leakage_guards: LeakageDiagnostics
    irregular_sampling: IrregularSamplingDiagnostics
    sensitivity: SensitivityDiagnostics
    risk_capabilities: RiskCapabilitiesDiagnostics


class StrategyRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    name: str
    status: str
    created_at_utc: str
    artifacts_ref: dict[str, str]
    signals_pack: dict[str, Any]
    meta_model: dict[str, Any]
    regimes: dict[str, Any]
    information: dict[str, Any]
    risk_overlay: dict[str, Any]
    position_target: dict[str, Any]
    backtest: dict[str, Any]
    evaluation: dict[str, Any]
    diagnostics: StrategyRunDiagnostics
    metadata: dict[str, Any]


class DecisionTraceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    timestamp_utc: str
    feature_snapshot: dict[str, Any]
    signal_contributions: dict[str, float]
    meta_decision: dict[str, Any]
    risk_adjustments: dict[str, float]
    position_target: float
    position_applied: float


def _hash_obj(payload: Any) -> str:
    return sha256(str(payload).encode("utf-8")).hexdigest()[:16]


def _to_json_safe(payload: Any) -> Any:
    """Recursively coerce numpy scalar values to native Python types."""

    if isinstance(payload, dict):
        return {key: _to_json_safe(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_to_json_safe(value) for value in payload]
    if isinstance(payload, tuple):
        return [_to_json_safe(value) for value in payload]
    if isinstance(payload, set):
        return [_to_json_safe(value) for value in payload]

    if payload.__class__.__module__.startswith("numpy") and hasattr(payload, "item"):
        return payload.item()

    return payload


def _rows_to_frame(rows: list[PriceRow]) -> pd.DataFrame:
    return pd.DataFrame({"timestamp_utc": [r.timestamp_utc for r in rows], "price": [r.price for r in rows]})


def _to_float_list(series: pd.Series) -> list[float]:
    return [float(v) for v in series.fillna(0.0).tolist()]


def _to_str_index(index: pd.Index) -> list[str]:
    return [ts.isoformat().replace("+00:00", "Z") for ts in pd.DatetimeIndex(index)]


def _duplicate_count(df: pd.DataFrame) -> int:
    return int(pd.to_datetime(df["timestamp_utc"], utc=True).duplicated().sum())


def _invalid_price_details(df: pd.DataFrame) -> tuple[int, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        p = float(row["price"])
        if p <= 0:
            details.append(
                {
                    "row": int(idx),
                    "timestamp_utc": pd.Timestamp(row["timestamp_utc"]).isoformat().replace("+00:00", "Z"),
                    "price": p,
                    "reason": "non_positive_price",
                }
            )
    return len(details), details


def _compute_gap_histogram(index: pd.DatetimeIndex) -> tuple[dict[str, int], float, list[dict[str, str]]]:
    if len(index) < 2:
        return {}, 0.0, []
    deltas = pd.Series(index).diff().dropna()
    counts = deltas.value_counts().sort_index()
    histogram = {str(k): int(v) for k, v in counts.items()}
    med = median([d.total_seconds() for d in deltas])
    irregularity = float((deltas.dt.total_seconds() != med).sum() / max(len(deltas), 1))
    dropped = []
    for i in range(len(index) - 1):
        if (index[i + 1] - index[i]).total_seconds() > med:
            dropped.append({"start": index[i].isoformat().replace("+00:00", "Z"), "end": index[i + 1].isoformat().replace("+00:00", "Z")})
    return histogram, irregularity, dropped


def _meta_model_capabilities() -> dict[str, Any]:
    return {
        "supported": True,
        "supported_model_types": sorted(SUPPORTED_META_MODEL_TYPES),
        "supported_feature_selection": sorted(SUPPORTED_META_FEATURE_SELECTION),
        "default_seed": 7,
    }


def _validate_meta_model_request(meta_model: MetaModelRequest) -> None:
    if meta_model.model_type not in SUPPORTED_META_MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"meta_model.model_type={meta_model.model_type} is unsupported")
    if meta_model.feature_selection not in SUPPORTED_META_FEATURE_SELECTION:
        raise HTTPException(status_code=400, detail=f"meta_model.feature_selection={meta_model.feature_selection} is unsupported")


def _build_meta_model(
    *,
    enabled: bool,
    request_cfg: MetaModelRequest,
    price_clean: pd.Series,
    features: pd.DataFrame,
    base_signal: pd.Series,
    regimes: pd.Series | None,
) -> tuple[dict[str, Any], pd.Series | None, dict[str, Any]]:
    if not enabled:
        return ({"enabled": False, **_meta_model_capabilities()}, None, {"applied": False})

    returns = price_clean.pct_change().fillna(0.0)
    cfg = MetaModelConfig(
        version="1.0",
        horizon=max(1, int(request_cfg.horizon)),
        min_train_size=5,
        n_splits=2,
        model_type=request_cfg.model_type,
        feature_selection=request_cfg.feature_selection,
        calibration="none",
        random_seed=7,
    )

    fit = fit_meta_model(
        prices=price_clean,
        returns=returns,
        base_signal=base_signal,
        features=features,
        regimes=regimes,
        cfg=cfg,
    )
    pred = predict_meta_model(
        fit=fit,
        features=features,
        base_signal=base_signal,
        regimes=regimes,
        cfg=cfg,
    )
    size = pred.size.reindex(features.index).fillna(0.0).clip(lower=0.0, upper=1.0)

    summary = {
        "enabled": True,
        "model_type": request_cfg.model_type,
        "feature_selection": request_cfg.feature_selection,
        "seed": cfg.random_seed,
        "config_hash": fit.metadata.get("config_hash"),
        "feature_hash": fit.metadata.get("feature_hash"),
        "decision_threshold": float(pred.decision_threshold or 0.5),
        "p_follow_mean": float(pred.p_follow.dropna().mean()) if pred.p_follow.dropna().size else 0.0,
        "p_follow_std": float(pred.p_follow.dropna().std(ddof=0)) if pred.p_follow.dropna().size else 0.0,
        "calibration": "none",
        "oos_rows": int(len(fit.oos_predictions)),
    }
    diagnostics = {
        "action_counts": pred.action.value_counts(dropna=False).astype(int).to_dict(),
        "size_nonzero": int((size > 0).sum()),
    }
    return summary, size, diagnostics


def _trim_series_payload(payload: dict[str, Any], *, summary_only: bool, stride: int, start: datetime | None, end: datetime | None) -> dict[str, Any]:
    out = deepcopy(payload)
    if summary_only:
        if "signals_pack" in out and "series" in out["signals_pack"]:
            out["signals_pack"].pop("series", None)
        if "risk_overlay" in out:
            out["risk_overlay"].pop("position_raw", None)
            out["risk_overlay"].pop("position_target", None)
            out["risk_overlay"].pop("turnover_raw", None)
            out["risk_overlay"].pop("turnover_after", None)
        if "position_target" in out:
            out["position_target"] = {"timestamps_utc": [], "values": []}
        if "backtest" in out and "series" in out["backtest"]:
            out["backtest"]["series"] = {"timestamps_utc": []}
        return out

    start_s = start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if start else None
    end_s = end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if end else None

    def _filter_block(timestamps: list[str], values: list[Any]) -> tuple[list[str], list[Any]]:
        idx = [i for i, t in enumerate(timestamps) if (start_s is None or t >= start_s) and (end_s is None or t <= end_s)]
        idx = idx[:: max(1, stride)]
        return [timestamps[i] for i in idx], [values[i] for i in idx]

    if "position_target" in out:
        ts = out["position_target"].get("timestamps_utc", [])
        vals = out["position_target"].get("values", [])
        nts, nvals = _filter_block(ts, vals)
        out["position_target"] = {"timestamps_utc": nts, "values": nvals}

    if "signals_pack" in out and "series" in out["signals_pack"]:
        s = out["signals_pack"]["series"]
        ts = s.get("timestamps_utc", [])
        idx = [i for i, t in enumerate(ts) if (start_s is None or t >= start_s) and (end_s is None or t <= end_s)][:: max(1, stride)]
        out["signals_pack"]["series"]["timestamps_utc"] = [ts[i] for i in idx]
        for k in ["momentum", "mean_reversion", "ensemble"]:
            if k in s:
                out["signals_pack"]["series"][k] = [s[k][i] for i in idx]

    if "backtest" in out and "series" in out["backtest"]:
        bs = out["backtest"]["series"]
        ts = bs.get("timestamps_utc", [])
        idx = [i for i, t in enumerate(ts) if (start_s is None or t >= start_s) and (end_s is None or t <= end_s)][:: max(1, stride)]
        out["backtest"]["series"]["timestamps_utc"] = [ts[i] for i in idx]
        for k in ["position_applied", "net_return", "equity"]:
            if k in bs:
                out["backtest"]["series"][k] = [bs[k][i] for i in idx]

    return out


def _build_diagnostics(
    *,
    request: StrategyRunRequest,
    bt: Any,
    ev: Any,
    features: pd.DataFrame,
    price_clean: pd.Series,
    position_target: pd.Series,
    risk_metrics: pd.DataFrame,
) -> StrategyRunDiagnostics:
    turnover_spikes = int((bt.turnover > bt.turnover.quantile(0.95)).sum()) if len(bt.turnover) else 0
    gap_hist, irregularity, dropped = _compute_gap_histogram(price_clean.index)

    warnings: list[str] = []
    decision_shift = int(features.attrs.get("decision_shift", 0))
    if decision_shift < 1:
        warnings.append("decision_shift_lt_1")
    if bt.metadata.get("execution") != "next_bar":
        warnings.append("execution_not_next_bar")

    stage_counts = {
        "input_rows": int(len(request.dataset.rows)),
        "clean_price_rows": int(len(price_clean)),
        "feature_rows": int(len(features)),
        "position_target_rows": int(len(position_target)),
        "backtest_rows": int(len(bt.returns)),
    }
    drop_reasons = {
        "feature_warmup_drop": max(stage_counts["clean_price_rows"] - stage_counts["feature_rows"], 0),
        "returns_first_bar_drop": max(stage_counts["position_target_rows"] - stage_counts["backtest_rows"], 0),
    }

    return StrategyRunDiagnostics(
        turnover_issues=TurnoverDiagnostics(
            turnover_raw=float(risk_metrics["turnover_raw"].sum()),
            turnover_after=float(risk_metrics["turnover_after"].sum()),
            turnover_spikes=turnover_spikes,
            costs_without_trade_count=int(((bt.costs > 0) & (bt.turnover <= 1e-12)).sum()),
        ),
        overlap=OverlapDiagnostics(
            stage_counts=stage_counts,
            overlap_ratio_position_vs_backtest=float(stage_counts["backtest_rows"] / max(stage_counts["position_target_rows"], 1)),
            dropped_index_intervals=dropped,
            drop_reasons=drop_reasons,
        ),
        units=UnitDiagnostics(
            signal_unit="score_or_leverage_bounded",
            position_unit="leverage",
            return_unit="simple_return",
            cost_unit="return_fraction",
            leverage_breach_count=int((position_target.abs() > 1.0).sum()),
        ),
        leakage_guards=LeakageDiagnostics(
            decision_shift=decision_shift,
            feature_shift=1,
            output_shift=1,
            effective_lag=1,
            warnings=warnings,
        ),
        irregular_sampling=IrregularSamplingDiagnostics(
            frequency=request.dataset.frequency,
            annualization_method=ev.metadata.get("periods_per_year"),
            gap_histogram=gap_hist,
            irregularity_score=irregularity,
        ),
        sensitivity=SensitivityDiagnostics(
            cost_delay_test_available=False,
            notes="cost and delay sensitivity sweeps are not implemented yet",
        ),
        risk_capabilities=RiskCapabilitiesDiagnostics(
            var_es_supported=False,
            var_es_reason="var/es overlay is intentionally disabled until implemented",
        ),
    )


@router.get("/strategy-runs/capabilities")
def strategy_run_capabilities() -> dict[str, Any]:
    return {
        "meta_model": _meta_model_capabilities(),
        "risk": {"var_es_supported": False, "var_es_reason": "not implemented"},
        "regimes": {"supported": True, "optional": True},
        "information": {"supported": True, "optional": True},
        "trace": {"supported": True, "single_timestamp_only": True},
    }


@router.post("/datasets/excel:ingest", status_code=status.HTTP_201_CREATED, response_model=IngestResponse)
def ingest_excel_dataset(request: ExcelIngestRequest, storage: SQLiteStorage = Depends(get_storage)) -> IngestResponse:
    df = _rows_to_frame(request.rows)
    dup = _duplicate_count(df)
    if request.duplicate_policy == "fail" and dup > 0:
        raise HTTPException(status_code=400, detail="Duplicate timestamps detected and duplicate_policy=fail")

    invalid_count, invalid_details = _invalid_price_details(df)
    if invalid_count > 0:
        raise HTTPException(status_code=400, detail={"invalid_price_rows": invalid_details})

    clean_cfg = CleanAlignConfig(resample_freq=request.frequency)
    try:
        cleaned, _, metadata = build_features_from_prices(
            df,
            timestamp_col="timestamp_utc",
            clean_cfg=clean_cfg,
            feature_cfg=FeatureConfig(momentum_windows=[3, 5, 10], vol_windows=[5], zscore_windows=[5], decision_shift=1),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    dataset_id = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    payload = IngestResponse(
        dataset_id=dataset_id,
        created_at_utc=created_at,
        price_type_used=request.price_type,
        frequency_inferred=request.frequency,
        ohlc_available=request.ohlc_available,
        missing_data_policy=request.missing_data_policy,
        index_range=IndexRange(start=cleaned.index.min().isoformat(), end=cleaned.index.max().isoformat(), bars=int(len(cleaned))),
        quality=IngestQuality(
            rows_in=len(request.rows),
            rows_clean=int(len(cleaned)),
            dropped_rows=int(len(request.rows) - len(cleaned)),
            duplicate_policy=request.duplicate_policy,
            duplicate_rows_detected=dup,
            invalid_price_rows=invalid_count,
            invalid_price_details=invalid_details,
        ),
        canonical_preview=CanonicalPreview(timestamps_utc=_to_str_index(cleaned.index[-10:]), price=_to_float_list(cleaned.tail(10))),
        metadata=metadata,
    )
    storage.save_dataset(request.model_dump(mode="json"), payload.model_dump(mode="json"), record_id=dataset_id, created_at=created_at)
    return payload


@router.post("/strategy-runs", status_code=status.HTTP_201_CREATED, response_model=StrategyRunResponse)
def run_strategy(request: StrategyRunRequest, storage: SQLiteStorage = Depends(get_storage)) -> StrategyRunResponse:
    _validate_meta_model_request(request.meta_model)

    df = _rows_to_frame(request.dataset.rows)
    dup = _duplicate_count(df)
    if request.dataset.duplicate_policy == "fail" and dup > 0:
        raise HTTPException(status_code=400, detail="Duplicate timestamps detected and duplicate_policy=fail")

    invalid_count, invalid_details = _invalid_price_details(df)
    if invalid_count > 0:
        raise HTTPException(status_code=400, detail={"invalid_price_rows": invalid_details})

    try:
        price_clean, features, ingest_metadata = build_features_from_prices(
            df,
            timestamp_col="timestamp_utc",
            clean_cfg=CleanAlignConfig(resample_freq=request.dataset.frequency),
            feature_cfg=FeatureConfig(momentum_windows=[3, 5, 10], vol_windows=[5], zscore_windows=[5], decision_shift=1),
        )
        inference = run_signal_pipeline(features=features, price=price_clean, risk_cfg=RiskConfig())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    signals = inference["signals"]
    momentum_key = next(k for k in signals if k.startswith("momentum"))
    base_signal = signals[momentum_key].position.reindex(features.index).fillna(0.0)

    regimes_payload: dict[str, Any] = {"enabled": False}
    regimes_series: pd.Series | None = None
    if request.regimes.enabled:
        try:
            regime_result = infer_regimes(returns=price_clean.pct_change().fillna(0.0), cfg=RegimeConfig())
            regimes_series = regime_result.regime
            regimes_payload = {
                "enabled": True,
                "warnings": regime_result.metadata.get("warnings", []),
                "state_counts": regime_result.diagnostics.get("state_counts", {}),
            }
        except Exception as exc:  # noqa: BLE001
            regimes_payload = {"enabled": False, "error": str(exc), "failure_safe": True}

    meta_summary, meta_size, meta_diag = _build_meta_model(
        enabled=request.meta_model.enabled,
        request_cfg=request.meta_model,
        price_clean=price_clean,
        features=features,
        base_signal=base_signal,
        regimes=regimes_series,
    )

    position_target = inference["position_target"].copy()
    if meta_size is not None:
        position_target = position_target.reindex(meta_size.index).fillna(0.0) * meta_size.fillna(0.0)

    accounting = run_accounting_and_evaluation(
        price=price_clean,
        position_target=position_target,
        backtest_cfg=BacktestConfig(initial_equity=100_000.0),
        evaluation_cfg=EvaluationConfig(),
    )

    bt = accounting["backtest"]
    ev = accounting["evaluation"]
    risk_metrics = inference["risk_metrics"]

    info_payload: dict[str, Any] = {"enabled": False}
    if request.information.enabled:
        try:
            info_cfg = InformationConfig(horizon=2, te_significance=False)
            info_report = build_information_report(
                returns=price_clean.pct_change().fillna(0.0),
                base_signal=base_signal,
                features=features,
                regimes=regimes_series,
                live_features=None,
                cfg=info_cfg,
            )
            info_payload = {
                "enabled": True,
                "ic_rows": int(len(info_report.ic)),
                "mi_rows": int(len(info_report.mi)),
                "warnings": info_report.metadata.get("warnings", []),
                "config_hash": info_report.metadata.get("config_hash"),
            }
        except Exception as exc:  # noqa: BLE001
            info_payload = {"enabled": False, "error": str(exc), "failure_safe": True}

    run_id = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    diagnostics = _build_diagnostics(
        request=request,
        bt=bt,
        ev=ev,
        features=features,
        price_clean=price_clean,
        position_target=position_target,
        risk_metrics=risk_metrics,
    )

    hygiene = {
        "walk_forward_present": {
            "value": bool(request.meta_model.enabled and meta_summary.get("enabled", False) and meta_summary.get("model_type") in SUPPORTED_META_MODEL_TYPES),
            "reason": "meta-model execution path used" if request.meta_model.enabled else "meta-model disabled",
        },
        "purged_cv_present": {
            "value": bool(request.meta_model.enabled),
            "reason": "meta-model uses purged_kfold by default" if request.meta_model.enabled else "meta-model disabled",
        },
        "embargo_present": {
            "value": False,
            "reason": "embargo currently 0 in strategy-run meta defaults",
        },
        "drift_test_present": {
            "value": bool(request.information.enabled and info_payload.get("enabled", False)),
            "reason": "information module run" if request.information.enabled else "information disabled",
        },
        "cost_sensitivity_present": {
            "value": False,
            "reason": "sensitivity suite pending",
        },
        "delay_test_present": {
            "value": False,
            "reason": "delay suite pending",
        },
    }

    result = StrategyRunResponse(
        run_id=run_id,
        name=request.name,
        status="completed",
        created_at_utc=created_at,
        artifacts_ref={
            "dataset_hash": _hash_obj(request.dataset.model_dump(mode="json")),
            "features_hash": _hash_obj(list(features.columns)),
            "run_hash": _hash_obj({"name": request.name, "bars": len(price_clean), "meta": request.meta_model.model_dump()}),
        },
        signals_pack={
            "signal_names": list(signals.keys()),
            "series": {
                "timestamps_utc": _to_str_index(signals["ensemble"].position.index),
                "momentum": _to_float_list(signals[next(k for k in signals if k.startswith("momentum"))].position),
                "mean_reversion": _to_float_list(signals[next(k for k in signals if k.startswith("mean_reversion"))].position),
                "ensemble": _to_float_list(signals["ensemble"].position),
            },
        },
        meta_model={**meta_summary, "diagnostics": meta_diag},
        regimes=regimes_payload,
        information=info_payload,
        risk_overlay={
            "position_raw": _to_float_list(signals["ensemble"].position),
            "position_target": _to_float_list(position_target),
            "turnover_raw": _to_float_list(risk_metrics["turnover_raw"]),
            "turnover_after": _to_float_list(risk_metrics["turnover_after"]),
            "var_es_supported": False,
            "var_es_reason": "not implemented",
        },
        position_target={"timestamps_utc": _to_str_index(position_target.index), "values": _to_float_list(position_target)},
        backtest={
            "summary": {
                "bars": int(len(bt.returns)),
                "total_return": float(bt.equity.iloc[-1] / 100_000.0 - 1.0),
                "total_costs": float(bt.costs.sum()),
                "turnover": float(bt.turnover.sum()),
            },
            "series": {
                "timestamps_utc": _to_str_index(bt.returns.index),
                "position_applied": _to_float_list(bt.position),
                "net_return": _to_float_list(bt.pnl),
                "equity": _to_float_list(bt.equity),
            },
            "engine_metadata": bt.metadata,
        },
        evaluation={"summary": ev.summary, "metadata": ev.metadata},
        diagnostics=diagnostics,
        metadata={
            "ingest": {
                **ingest_metadata,
                "duplicate_rows_detected": dup,
                "invalid_price_rows": invalid_count,
                "invalid_price_details": invalid_details,
                "missing_data_policy": request.dataset.missing_data_policy,
                "ohlc_available": request.dataset.ohlc_available,
            },
            "inference": inference["metadata"],
            "determinism": {
                "seed": 7,
                "dataset_hash": _hash_obj(request.dataset.model_dump(mode="json")),
                "meta_config_hash": _hash_obj(request.meta_model.model_dump()),
                "reproducibility_hash": _hash_obj({"dataset": request.dataset.model_dump(), "meta": request.meta_model.model_dump(), "bars": len(price_clean)}),
            },
            "research_hygiene": hygiene,
        },
    )

    storage.save_strategy_run(
        request.model_dump(mode="json"),
        _to_json_safe(result.model_dump(mode="python")),
        record_id=run_id,
        created_at=created_at,
    )
    return result


@router.get("/strategy-runs/{run_id}", response_model=StrategyRunResponse)
def get_strategy_run(
    run_id: str,
    summary_only: bool = Query(False),
    stride: int = Query(1, ge=1),
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    storage: SQLiteStorage = Depends(get_storage),
) -> StrategyRunResponse:
    record = storage.get_strategy_run(run_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy run not found")
    payload = _trim_series_payload(record["result"], summary_only=summary_only, stride=stride, start=start, end=end)
    return StrategyRunResponse.model_validate(payload)


@router.get("/strategy-runs/{run_id}/diagnostics", response_model=StrategyRunDiagnostics)
def get_strategy_run_diagnostics(run_id: str, storage: SQLiteStorage = Depends(get_storage)) -> StrategyRunDiagnostics:
    record = storage.get_strategy_run(run_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy run not found")
    return StrategyRunDiagnostics.model_validate(record["result"]["diagnostics"])


@router.get("/strategy-runs/{run_id}/trace", response_model=DecisionTraceResponse)
def get_strategy_run_trace(
    run_id: str,
    timestamp: datetime = Query(...),
    storage: SQLiteStorage = Depends(get_storage),
) -> DecisionTraceResponse:
    record = storage.get_strategy_run(run_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy run not found")

    result = record["result"]
    ts = timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    ts_list = result["position_target"]["timestamps_utc"]
    if ts not in ts_list:
        raise HTTPException(status_code=404, detail="Timestamp not found in run")

    idx = ts_list.index(ts)
    momentum = float(result["signals_pack"]["series"]["momentum"][idx])
    mean_reversion = float(result["signals_pack"]["series"]["mean_reversion"][idx])
    ensemble = float(result["signals_pack"]["series"]["ensemble"][idx])
    pt = float(result["position_target"]["values"][idx])

    applied_ts = result["backtest"]["series"]["timestamps_utc"]
    pa = float(result["backtest"]["series"]["position_applied"][applied_ts.index(ts)]) if ts in applied_ts else 0.0

    return DecisionTraceResponse(
        run_id=run_id,
        timestamp_utc=ts,
        feature_snapshot={
            "decision_shift": result["diagnostics"]["leakage_guards"]["decision_shift"],
            "feature_shift": result["diagnostics"]["leakage_guards"]["feature_shift"],
            "output_shift": result["diagnostics"]["leakage_guards"]["output_shift"],
            "warnings": result["diagnostics"]["leakage_guards"]["warnings"],
        },
        signal_contributions={"momentum": momentum, "mean_reversion": mean_reversion, "ensemble": ensemble},
        meta_decision=result.get("meta_model", {"enabled": False}),
        risk_adjustments={"position_target": pt, "position_raw_ensemble": ensemble},
        position_target=pt,
        position_applied=pa,
    )
