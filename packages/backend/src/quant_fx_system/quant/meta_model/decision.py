from __future__ import annotations

import numpy as np
import pandas as pd

from .types import MetaModelConfig


def select_threshold(p_follow: pd.Series, pnl: pd.Series) -> float:
    candidates = np.linspace(0.1, 0.9, 17)
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        mask = p_follow >= threshold
        score = pnl[mask].mean() if mask.any() else -np.inf
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def apply_decision_policy(
    *,
    p_follow: pd.Series,
    base_signal: pd.Series,
    threshold: float,
    cfg: MetaModelConfig,
) -> tuple[pd.Series, pd.Series, pd.Series | None]:
    direction = np.sign(base_signal)
    if cfg.decision_policy == "threshold":
        follow = p_follow >= threshold
        size = ((p_follow - threshold) / (1 - threshold)).clip(lower=0.0, upper=1.0)
        action = direction.where(follow, 0.0)
        return action, size, None

    if cfg.decision_policy == "kelly_fraction":
        p = p_follow.clip(1e-6, 1 - 1e-6)
        b = cfg.payoff_ratio
        f = (p * b - (1 - p)) / max(b, 1e-6)
        size = f.clip(lower=0.0, upper=cfg.max_leverage)
        action = direction.where(size > 0, 0.0)
        return action, size, None

    if cfg.decision_policy in {"bayes_risk", "utility"}:
        expected_edge = p_follow - (1 - p_follow)
        action = direction.where(expected_edge > 0, 0.0)
        size = expected_edge.clip(lower=0.0, upper=cfg.max_leverage)
        return action, size, expected_edge

    raise ValueError(f"Unsupported decision policy: {cfg.decision_policy}")
