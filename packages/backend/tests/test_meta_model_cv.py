import numpy as np

from quant_fx_system.quant.meta_model.cv import build_cv_splits
from quant_fx_system.quant.meta_model.types import MetaModelConfig


def test_purged_kfold_respects_purge_embargo():
    cfg = MetaModelConfig(
        version="1.0",
        horizon=1,
        cv_scheme="purged_kfold",
        n_splits=4,
        purge=2,
        embargo=2,
    )
    splits = build_cv_splits(20, cfg)

    for split in splits:
        test_start = int(split.test_idx[0])
        test_end = int(split.test_idx[-1])
        forbidden_start = max(test_start - cfg.purge, 0)
        forbidden_end = min(test_end + cfg.embargo, 19)
        forbidden = set(range(forbidden_start, forbidden_end + 1))
        assert not forbidden.intersection(split.train_idx.tolist())
        assert np.intersect1d(split.train_idx, split.test_idx).size == 0
