from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from typing import Any

import pandas as pd

from .types import MetaModelFit, MetaModelOutput


def serialize_fit(fit: MetaModelFit) -> bytes:
    payload = {
        "model": pickle.dumps(fit.model),
        "calibrator": pickle.dumps(fit.calibrator),
        "feature_schema": fit.feature_schema,
        "oos_predictions": fit.oos_predictions.to_json(orient="split"),
        "metrics": fit.metrics.to_json(orient="split"),
        "metadata": fit.metadata,
    }
    return pickle.dumps(payload)


def load_fit(blob: bytes) -> MetaModelFit:
    payload: dict[str, Any] = pickle.loads(blob)
    return MetaModelFit(
        model=pickle.loads(payload["model"]),
        calibrator=pickle.loads(payload["calibrator"]),
        feature_schema=payload["feature_schema"],
        oos_predictions=pd.read_json(payload["oos_predictions"], orient="split"),
        metrics=pd.read_json(payload["metrics"], orient="split"),
        metadata=payload["metadata"],
    )


def serialize_output(output: MetaModelOutput) -> str:
    payload = {
        "p_follow": output.p_follow.to_json(orient="split"),
        "action": output.action.to_json(orient="split"),
        "size": output.size.to_json(orient="split"),
        "decision_threshold": output.decision_threshold,
        "expected_edge": output.expected_edge.to_json(orient="split")
        if output.expected_edge is not None
        else None,
        "metadata": output.metadata,
    }
    return json.dumps(payload)


def load_output(blob: str) -> MetaModelOutput:
    payload = json.loads(blob)
    expected_edge = (
        pd.read_json(payload["expected_edge"], orient="split")
        if payload["expected_edge"] is not None
        else None
    )
    return MetaModelOutput(
        p_follow=pd.read_json(payload["p_follow"], orient="split"),
        action=pd.read_json(payload["action"], orient="split"),
        size=pd.read_json(payload["size"], orient="split"),
        decision_threshold=payload["decision_threshold"],
        expected_edge=expected_edge,
        metadata=payload["metadata"],
    )
