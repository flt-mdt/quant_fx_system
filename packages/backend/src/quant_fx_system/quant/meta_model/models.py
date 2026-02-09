from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import MetaModelConfig


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


@dataclass
class LogisticModel:
    coef_: np.ndarray
    intercept_: float
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is not None and self.std_ is not None:
            X = _standardize_apply(X, self.mean_, self.std_)
        scores = X @ self.coef_ + self.intercept_
        return _sigmoid(scores)


@dataclass
class RidgeModel:
    coef_: np.ndarray
    intercept_: float
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is not None and self.std_ is not None:
            X = _standardize_apply(X, self.mean_, self.std_)
        scores = X @ self.coef_ + self.intercept_
        return _sigmoid(scores)


def fit_logistic(X: np.ndarray, y: np.ndarray, cfg: MetaModelConfig) -> LogisticModel:
    rng = np.random.default_rng(cfg.random_seed)
    X_fit = X
    mean = std = None
    if cfg.standardize_features:
        X_fit, mean, std = _standardize_fit(X_fit)

    n_samples, n_features = X_fit.shape
    coef = rng.normal(scale=0.01, size=n_features)
    intercept = 0.0
    lr = cfg.regularization.get("lr", 0.1)
    n_iter = cfg.regularization.get("max_iter", 200)
    c_val = cfg.regularization.get("C", 1.0)
    l2 = 1.0 / max(c_val, 1e-6)

    for _ in range(n_iter):
        scores = X_fit @ coef + intercept
        probs = _sigmoid(scores)
        grad = X_fit.T @ (probs - y) / n_samples + l2 * coef
        intercept_grad = np.mean(probs - y)
        coef -= lr * grad
        intercept -= lr * intercept_grad

    return LogisticModel(coef_=coef, intercept_=intercept, mean_=mean, std_=std)


def fit_ridge(X: np.ndarray, y: np.ndarray, cfg: MetaModelConfig) -> RidgeModel:
    X_fit = X
    mean = std = None
    if cfg.standardize_features:
        X_fit, mean, std = _standardize_fit(X_fit)
    alpha = cfg.regularization.get("alpha", 1.0)
    n_features = X_fit.shape[1]
    xtx = X_fit.T @ X_fit + alpha * np.eye(n_features)
    xty = X_fit.T @ y
    coef = np.linalg.solve(xtx, xty)
    intercept = 0.0
    return RidgeModel(coef_=coef, intercept_=intercept, mean_=mean, std_=std)


def fit_model(X: np.ndarray, y: np.ndarray, cfg: MetaModelConfig):
    if cfg.feature_selection != "none":
        raise NotImplementedError("feature_selection options are not implemented")
    if cfg.model_type == "logistic":
        return fit_logistic(X, y, cfg)
    if cfg.model_type == "ridge":
        return fit_ridge(X, y, cfg)
    raise NotImplementedError(f"model_type {cfg.model_type} is not implemented")
