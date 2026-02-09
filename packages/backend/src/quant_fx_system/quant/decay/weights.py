from __future__ import annotations

import numpy as np

from quant_fx_system.quant.decay.types import DecayKind


def generate_weights(
    kind: DecayKind,
    window: int,
    *,
    power_exponent: float = 1.0,
    alpha: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be >= 1")

    if kind == "linear":
        weights = np.arange(1.0, window + 1.0)
    elif kind == "step":
        weights = np.ones(window, dtype=float)
    elif kind == "power":
        if power_exponent <= 0:
            raise ValueError("power_exponent must be > 0")
        weights = np.arange(1.0, window + 1.0) ** power_exponent
    elif kind == "ewma":
        if alpha is None:
            raise ValueError("alpha must be provided for ewma weights")
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be in (0, 1]")
        weights = (1 - alpha) ** (np.arange(window - 1, -1, -1)) * alpha
    else:
        raise ValueError(f"Unsupported weight kind: {kind}")

    if normalize:
        total = weights.sum()
        if total <= 0:
            raise ValueError("weights sum must be positive")
        weights = weights / total
    return weights
