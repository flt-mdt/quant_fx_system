import numpy as np

from quant_fx_system.quant.meta_model.cv import build_cv_splits
from quant_fx_system.quant.meta_model.types import MetaModelConfig


def test_purged_kfold_respects_purge_embargo():
    cfg = MetaModelConfig(
        version="1.0",
        horizon=5,
        cv_scheme="purged_kfold",
        n_splits=4,
        purge=2,
        embargo=1,
    )
    splits = build_cv_splits(20, cfg)
    effective_purge = max(cfg.purge, cfg.horizon)
    effective_embargo = max(cfg.embargo, cfg.horizon)

    for split in splits:
        test_start = int(split.test_idx[0])
        test_end = int(split.test_idx[-1])
        forbidden_start = max(test_start - effective_purge, 0)
        forbidden_end = min(test_end + effective_embargo, 19)
        forbidden = set(range(forbidden_start, forbidden_end + 1))
        assert not forbidden.intersection(split.train_idx.tolist())
        assert np.intersect1d(split.train_idx, split.test_idx).size == 0
