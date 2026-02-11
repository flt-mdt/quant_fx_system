"""Feature building service."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from quant_fx_system.quant.data_pipeline.clean_align import CleanAlignConfig, clean_price_pipeline
from quant_fx_system.quant.data_pipeline.feature_engineering import FeatureConfig, build_features


def build_features_from_prices(
    raw_prices: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp_utc",
    clean_cfg: CleanAlignConfig | None = None,
    feature_cfg: FeatureConfig | None = None,
) -> tuple[pd.Series, pd.DataFrame, dict[str, object]]:
    clean_cfg = clean_cfg or CleanAlignConfig()
    feature_cfg = feature_cfg or FeatureConfig()

    cleaned_price = clean_price_pipeline(raw_prices, cfg=clean_cfg, timestamp_col=timestamp_col)
    features = build_features(cleaned_price, cfg=feature_cfg)

    metadata: dict[str, object] = {
        "clean_config": asdict(clean_cfg),
        "feature_config": asdict(feature_cfg),
        "price_rows_in": int(len(raw_prices)),
        "price_rows_clean": int(len(cleaned_price)),
        "feature_rows": int(len(features)),
        "decision_shift": int(features.attrs.get("decision_shift", 0)),
        "index_start": cleaned_price.index.min().isoformat(),
        "index_end": cleaned_price.index.max().isoformat(),
    }
    return cleaned_price, features, metadata
