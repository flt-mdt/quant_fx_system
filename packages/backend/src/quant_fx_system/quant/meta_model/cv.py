from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import MetaModelConfig


@dataclass(frozen=True)
class CvSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray


def purged_kfold(n_samples: int, cfg: MetaModelConfig) -> list[CvSplit]:
    if cfg.n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    indices = np.arange(n_samples)
    fold_sizes = np.full(cfg.n_splits, n_samples // cfg.n_splits, dtype=int)
    fold_sizes[: n_samples % cfg.n_splits] += 1
    splits: list[CvSplit] = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        purge_start = max(start - cfg.purge, 0)
        purge_stop = min(stop + cfg.embargo, n_samples)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[purge_start:purge_stop] = False
        train_idx = indices[train_mask]
        splits.append(CvSplit(train_idx=train_idx, test_idx=test_idx))
        current = stop
    return splits


def walk_forward(n_samples: int, cfg: MetaModelConfig) -> list[CvSplit]:
    if cfg.train_window is None or cfg.test_window is None or cfg.step_size is None:
        raise ValueError("train_window, test_window, step_size required for walk_forward")
    splits: list[CvSplit] = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + cfg.train_window
        test_end = train_end + cfg.test_window
        if test_end > n_samples:
            break
        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(train_end, test_end)
        splits.append(CvSplit(train_idx=train_idx, test_idx=test_idx))
        start += cfg.step_size
    if not splits:
        raise ValueError("walk_forward produced no splits")
    return splits


def build_cv_splits(n_samples: int, cfg: MetaModelConfig) -> list[CvSplit]:
    if cfg.cv_scheme == "purged_kfold":
        return purged_kfold(n_samples, cfg)
    if cfg.cv_scheme == "combinatorial_purged_cv":
        return purged_kfold(n_samples, cfg)
    if cfg.cv_scheme == "walk_forward":
        return walk_forward(n_samples, cfg)
    raise ValueError(f"Unsupported cv_scheme: {cfg.cv_scheme}")
