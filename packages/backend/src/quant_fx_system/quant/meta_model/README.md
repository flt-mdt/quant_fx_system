# quant.meta_model

`quant.meta_model` converts raw alpha signals into tradable decisions (follow/ignore/size) with
strict anti-lookahead controls, calibrated probabilities, and auditability. The implementation
follows research-grade practices inspired by Lopez de Prado and practitioner workflows.

## Core concepts

- **Meta-labeling**: learn when a base signal is worth following by predicting profitability
  conditional on signal quality and market state.
- **Triple-barrier labeling**: optional event-based labels with profit-taking/stop-loss and
  vertical barriers.
- **Purged CV + embargo**: time-series validation that removes leakage around train/test
  boundaries.
- **Calibration**: isotonic/platt/beta calibration with Brier/ECE diagnostics.
- **Decision under costs**: decision policies account for costs, utilities, and sizing
  (threshold, Bayes risk, Kelly fraction).
- **Determinism + audit**: config hashing, feature schema, and serialization helpers.
- **Drift monitoring**: PSI/KS hooks live in diagnostics for production extensions.

## API

```python
from quant_fx_system.quant.meta_model import MetaModelConfig, fit_meta_model, predict_meta_model

cfg = MetaModelConfig(
    version="1.0",
    horizon=5,
    labeling="meta_label",
    decision_policy="threshold",
    calibration="isotonic",
    cv_scheme="purged_kfold",
    n_splits=5,
    purge=2,
    embargo=2,
)

fit = fit_meta_model(
    prices=prices,
    returns=returns,
    base_signal=signal,
    features=features,
    regimes=regimes,
    cfg=cfg,
)

output = predict_meta_model(
    fit=fit,
    features=features,
    base_signal=signal,
    regimes=regimes,
    cfg=cfg,
)
```

## Notes

- Features are shifted by `feature_shift` (default 1) to avoid leakage.
- Cross-validation supports purged k-fold, walk-forward, and a placeholder CPCV interface.
- Output includes `p_follow`, `action`, `size`, and decision metadata for auditing.
