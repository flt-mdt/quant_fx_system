from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _clip_proba(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1 - 1e-6)


@dataclass
class NoCalibration:
    def fit(self, scores: np.ndarray, y: np.ndarray) -> "NoCalibration":
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return _clip_proba(scores)


@dataclass
class PlattCalibration:
    a_: float = 0.0
    b_: float = 0.0

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibration":
        x = scores.reshape(-1)
        y = y.reshape(-1)
        a, b = 0.0, 0.0
        lr = 0.1
        for _ in range(200):
            logits = a * x + b
            p = _sigmoid(logits)
            grad_a = np.mean((p - y) * x)
            grad_b = np.mean(p - y)
            a -= lr * grad_a
            b -= lr * grad_b
        self.a_ = a
        self.b_ = b
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        logits = self.a_ * scores + self.b_
        return _sigmoid(logits)


@dataclass
class BetaCalibration:
    a_: float = 1.0
    b_: float = 1.0
    c_: float = 0.0

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "BetaCalibration":
        p = _clip_proba(scores)
        x1 = np.log(p)
        x2 = np.log(1 - p)
        X = np.column_stack([x1, x2, np.ones_like(x1)])
        y = y.reshape(-1)
        coef = np.zeros(3)
        lr = 0.1
        for _ in range(300):
            logits = X @ coef
            preds = _sigmoid(logits)
            grad = X.T @ (preds - y) / len(y)
            coef -= lr * grad
        self.a_, self.b_, self.c_ = coef
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        p = _clip_proba(scores)
        x1 = np.log(p)
        x2 = np.log(1 - p)
        logits = self.a_ * x1 + self.b_ * x2 + self.c_
        return _sigmoid(logits)


@dataclass
class IsotonicCalibration:
    x_: np.ndarray | None = None
    y_: np.ndarray | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicCalibration":
        order = np.argsort(scores)
        x = scores[order]
        y_sorted = y[order].astype(float)
        y_iso = y_sorted.copy()
        n = len(y_iso)
        weights = np.ones(n)
        i = 0
        while i < n - 1:
            if y_iso[i] > y_iso[i + 1]:
                total_weight = weights[i] + weights[i + 1]
                avg = (y_iso[i] * weights[i] + y_iso[i + 1] * weights[i + 1]) / total_weight
                y_iso[i] = avg
                y_iso[i + 1] = avg
                weights[i] = total_weight
                weights[i + 1] = total_weight
                j = i
                while j > 0 and y_iso[j - 1] > y_iso[j]:
                    total_weight = weights[j - 1] + weights[j]
                    avg = (y_iso[j - 1] * weights[j - 1] + y_iso[j] * weights[j]) / total_weight
                    y_iso[j - 1] = avg
                    y_iso[j] = avg
                    weights[j - 1] = total_weight
                    weights[j] = total_weight
                    j -= 1
            i += 1
        self.x_ = x
        self.y_ = y_iso
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.x_ is None or self.y_ is None:
            raise ValueError("IsotonicCalibration not fitted")
        return np.interp(scores, self.x_, self.y_, left=self.y_[0], right=self.y_[-1])


def make_calibrator(name: str):
    if name == "none":
        return NoCalibration()
    if name == "platt":
        return PlattCalibration()
    if name == "beta":
        return BetaCalibration()
    if name == "isotonic":
        return IsotonicCalibration()
    raise ValueError(f"Unsupported calibration: {name}")
