from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


CalibrationType = Literal["none", "isotonic", "platt", "beta"]
CvScheme = Literal["purged_kfold", "combinatorial_purged_cv", "walk_forward"]
DecisionPolicy = Literal["threshold", "bayes_risk", "kelly_fraction", "utility"]
FeatureSelection = Literal["none", "lasso", "mutual_info", "shap"]
LabelingType = Literal["meta_label", "triple_barrier", "return_sign", "quantile"]
ModelType = Literal["logistic", "xgb", "rf", "linear_svm", "ridge"]


@dataclass(frozen=True)
class MetaModelConfig:
    version: str
    horizon: int
    feature_shift: int = 1
    output_shift: int = 1
    labeling: LabelingType = "meta_label"
    decision_policy: DecisionPolicy = "threshold"
    calibration: CalibrationType = "none"

    cv_scheme: CvScheme = "purged_kfold"
    n_splits: int = 5
    purge: int = 0
    embargo: int = 0
    calibration_window: tuple[pd.Timestamp | None, pd.Timestamp | None] = (None, None)
    train_window: int | None = None
    test_window: int | None = None
    step_size: int | None = None
    min_train_size: int = 50

    model_type: ModelType = "logistic"
    class_weight: Literal["balanced", "none"] = "none"
    regularization: dict[str, Any] = field(default_factory=dict)
    random_seed: int = 7
    standardize_features: bool = True
    feature_selection: FeatureSelection = "none"

    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    min_edge_bps: float = 0.0
    max_turnover: float = 1.0
    max_leverage: float = 1.0

    return_artifacts: bool = True
    store_oos_predictions: bool = True

    triple_barrier_pt: float = 0.01
    triple_barrier_sl: float = 0.01
    payoff_ratio: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "horizon": self.horizon,
            "feature_shift": self.feature_shift,
            "output_shift": self.output_shift,
            "labeling": self.labeling,
            "decision_policy": self.decision_policy,
            "calibration": self.calibration,
            "cv_scheme": self.cv_scheme,
            "n_splits": self.n_splits,
            "purge": self.purge,
            "embargo": self.embargo,
            "calibration_window": self.calibration_window,
            "train_window": self.train_window,
            "test_window": self.test_window,
            "step_size": self.step_size,
            "min_train_size": self.min_train_size,
            "model_type": self.model_type,
            "class_weight": self.class_weight,
            "regularization": dict(self.regularization),
            "random_seed": self.random_seed,
            "standardize_features": self.standardize_features,
            "feature_selection": self.feature_selection,
            "transaction_cost_bps": self.transaction_cost_bps,
            "slippage_bps": self.slippage_bps,
            "min_edge_bps": self.min_edge_bps,
            "max_turnover": self.max_turnover,
            "max_leverage": self.max_leverage,
            "return_artifacts": self.return_artifacts,
            "store_oos_predictions": self.store_oos_predictions,
            "triple_barrier_pt": self.triple_barrier_pt,
            "triple_barrier_sl": self.triple_barrier_sl,
            "payoff_ratio": self.payoff_ratio,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetaModelConfig":
        return cls(**payload)


@dataclass
class MetaModelFit:
    model: Any
    calibrator: Any
    feature_schema: dict[str, Any]
    oos_predictions: pd.DataFrame
    metrics: pd.DataFrame
    metadata: dict[str, Any]


@dataclass
class MetaModelOutput:
    p_follow: pd.Series
    action: pd.Series
    size: pd.Series
    decision_threshold: float | None
    expected_edge: pd.Series | None
    metadata: dict[str, Any]
