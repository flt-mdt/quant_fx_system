from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_true - y_prob) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    clipped = np.clip(y_prob, 0.0, 1.0)
    indices = np.digitize(clipped, bins) - 1
    indices = np.clip(indices, 0, n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        mask = indices == i
        if not np.any(mask):
            continue
        avg_conf = float(np.mean(clipped[mask]))
        avg_acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / len(y_true)) * abs(avg_conf - avg_acc)
    return float(ece)


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "brier": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def summarize_oos(oos: pd.DataFrame) -> pd.DataFrame:
    if oos.empty:
        return pd.DataFrame()
    metrics = calibration_metrics(oos["y_true"].values, oos["p_follow"].values)
    return pd.DataFrame([metrics])
