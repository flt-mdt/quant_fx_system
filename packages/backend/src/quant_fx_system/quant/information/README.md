# quant.information

Research-grade + prod-grade information metrics for FX alphas and feature sets. This module
measures predictive power (IC/MI/TE), information content (entropy), and production drift
(PSI/JS/KL) while enforcing UTC, alignment, and anti-lookahead behavior.

## Scope
- Evaluate alpha signals and feature sets against targets.
- Feature selection before meta-models.
- Monitor drift and loss of information in production.

## Usage
```python
from quant_fx_system.quant.information import InformationConfig, build_information_report

cfg = InformationConfig(
    horizon=5,
    feature_shift=1,
    target="forward_return",
    ic_method="spearman",
    mi_estimator="gaussian_copula",
    regime_conditional=True,
    te_significance=True,
    te_perm_runs=200,
)

report = build_information_report(
    returns=eurusd_returns,
    base_signal=alpha_signal,
    features=feature_df,
    regimes=regime_series,
    live_features=live_feature_df,
    cfg=cfg,
)

report.ic.sort_values("ic_mean", ascending=False).head(20)
```

## Notes
- IC + MI + drift is a strong filter: stable IC, non-zero MI, low drift.
- Overlapping horizons require NW/HAC stats for credibility.
- TE is expensive; reserve for top-K features.
