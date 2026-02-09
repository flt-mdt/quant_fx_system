from __future__ import annotations

import pandas as pd

from .ic import compute_ic
from .mi import mutual_information
from .types import InformationConfig


def mrmr_select(features: pd.DataFrame, target: pd.Series, k: int, cfg: InformationConfig) -> list[str]:
    if k <= 0:
        return []
    candidates = list(features.columns)
    selected: list[str] = []
    relevance = {col: mutual_information(features[col], target, cfg) for col in candidates}

    while candidates and len(selected) < k:
        scores = {}
        for col in candidates:
            if not selected:
                scores[col] = relevance[col]
                continue
            redundancy = []
            for chosen in selected:
                redundancy.append(mutual_information(features[col], features[chosen], cfg))
            scores[col] = relevance[col] - float(sum(redundancy)) / len(redundancy)
        best = max(scores, key=scores.get)
        selected.append(best)
        candidates.remove(best)
    return selected


def rank_features_by_ic_mi(
    features: pd.DataFrame, target: pd.Series, cfg: InformationConfig
) -> pd.DataFrame:
    rows = []
    for col in features.columns:
        rows.append(
            {
                "feature": col,
                "ic": compute_ic(features[col], target, method=cfg.ic_method),
                "mi": mutual_information(features[col], target, cfg),
            }
        )
    return pd.DataFrame(rows).set_index("feature")
