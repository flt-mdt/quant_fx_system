from __future__ import annotations

import numpy as np
import pandas as pd


def transition_matrix(regime: pd.Series, n_states: int) -> pd.DataFrame:
    counts = np.zeros((n_states, n_states), dtype=float)
    values = regime.dropna().astype(int).to_numpy()
    for prev, curr in zip(values[:-1], values[1:]):
        if 0 <= prev < n_states and 0 <= curr < n_states:
            counts[prev, curr] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.zeros_like(counts)
        np.divide(counts, row_sums, out=probs, where=row_sums != 0)
    matrix = pd.DataFrame(probs, columns=range(n_states), index=range(n_states))
    return matrix


def regime_durations(regime: pd.Series) -> pd.Series:
    values = regime.dropna().astype(int).to_numpy()
    if len(values) == 0:
        return pd.Series(dtype=int)
    durations = []
    current = values[0]
    length = 1
    for value in values[1:]:
        if value == current:
            length += 1
        else:
            durations.append(length)
            current = value
            length = 1
    durations.append(length)
    return pd.Series(durations)


__all__ = ["regime_durations", "transition_matrix"]
