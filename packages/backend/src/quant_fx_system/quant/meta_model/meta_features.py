from __future__ import annotations

import pandas as pd


def build_interaction_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create simple interaction features for signal-quality extensions."""
    if features.empty:
        return features
    cols = list(features.columns)
    interactions = {}
    for i, col in enumerate(cols):
        for other in cols[i + 1 :]:
            interactions[f"{col}_x_{other}"] = features[col] * features[other]
    return pd.concat([features, pd.DataFrame(interactions, index=features.index)], axis=1)
