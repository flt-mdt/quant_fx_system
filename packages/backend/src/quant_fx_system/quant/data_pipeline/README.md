# `data_pipeline` — Clean & Features “no-lookahead” pour séries FX

Pipeline **pandas-first** pour transformer des séries de prix FX brutes en **prix nettoyés**, **features décisionnelles sans look-ahead**, et **diagnostics de stationnarité** (ADF/KPSS) — avec garde-fous stricts (timezone, duplicates, outliers, shift de décision).

> Contexte repo : module Python situé sous `packages/backend/src/quant_fx_system/quant/data_pipeline/`.

---

## Badges

À compléter (repo/workflows non détectables depuis ce dossier seul) : build · tests · coverage · license · last commit · docker image · code style.

---

## Table des matières

- [Aperçu](#aperçu)
- [Key features](#key-features)
- [Architecture (haut niveau)](#architecture-haut-niveau)
- [Stack & architecture](#stack--architecture)
- [Arborescence](#arborescence)
- [API du module](#api-du-module)
- [Quickstart](#quickstart)
- [Qualité](#qualité)
- [Contribution](#contribution)
- [Licence](#licence)
- [Crédits](#crédits)

---

## Aperçu

Ce dossier contient **3 briques** cohérentes et testées :

1. **Cleaning & alignment** : index datetime tz-aware, déduplication, filtrage d’outliers via log-returns, resampling.
2. **Feature engineering** : retours log, momentum, vol, z-score — avec **shift “decision-grade”** pour garantir l’absence de look-ahead.
3. **Stationarity diagnostics** : wrappers ADF/KPSS + report tabulaire.

---

## Key features

- **UTC strict** : tout est normalisé en `DatetimeIndex` timezone-aware en UTC (localisation si timestamps naïfs).
- **Robustesse de série** : suppression des timestamps dupliqués (keep=`last`), tri, validation monotonic/unique.
- **Outlier filter simple & explicite** : suppression des points dont `|Δlog(price)|` dépasse un seuil.
- **No-lookahead by construction** : features **shiftées** de `decision_shift` (par défaut 1) et validations associées.
- **Diagnostics stationnarité** : `ADF` + `KPSS` avec sortie structurée / DataFrame “tidy”.

---

## Architecture (haut niveau)

```mermaid
flowchart LR
  Raw[Raw FX prices (Series/DataFrame)] --> A[clean_align.clean_price_pipeline]
  A --> P[price_clean (Series)]
  P --> F[feature_engineering.build_features (decision_shift)]
  F --> X[features (DataFrame)]
  P --> S[stationarity.stationarity_report (ADF + KPSS)]
  S --> R[report (DataFrame)]
  T[tests/test_data_pipeline.py] -. verifies .- A
  T -. verifies .- F
  T -. verifies .- S
```

---

## Stack & architecture

| Couche | Outils | Détails |
| --- | --- | --- |
| Langage | Python | À compléter (version exacte non détectée ici) |
| Data | pandas, numpy | Séries temporelles, rolling ops |
| Stats | statsmodels | adfuller, kpss |
| Tests | pytest | Tests unitaires sur cleaning/features/stationarity |

Notes : aucun framework web / DB / queue détecté dans ce dossier.

---

## Arborescence

```
packages/backend/src/quant_fx_system/quant/data_pipeline/
├── clean_align.py           # normalisation datetime, choix price, dédup, outliers, resampling
├── feature_engineering.py   # features + shift no-lookahead
└── stationarity.py          # ADF/KPSS + report tabulaire

packages/backend/tests/
└── test_data_pipeline.py    # tests unitaires du pipeline
```

---

## API du module

### `clean_align.py`

**`CleanAlignConfig`**

- `tz_assume: str = "UTC"` : timezone à assumer si timestamps naïfs
- `resample_freq: str = "1D"`
- `price_cols_priority: list[str] = ["mid","px_last","price","close"]`
- `outlier_max_abs_log_return: float = 0.10`
- `enable_outlier_filter: bool = True`

**`ensure_datetime_index_utc(df, timestamp_col=None, tz_assume="UTC") -> DataFrame`**
Normalise l’index en DatetimeIndex tz-aware en UTC (ou utilise `timestamp_col` si index non datetime).

**`choose_price_series(df, price_cols_priority, bid_col="bid", ask_col="ask") -> Series`**
Renvoie une série price à partir de bid/ask (mid) ou d’une colonne existante.

**`deduplicate_and_validate_index(series) -> Series`**
Supprime doublons, trie, force UTC, valide monotonie/unicité.

**`filter_outliers_by_returns(price, max_abs_log_return=0.10) -> Series`**
Supprime points “extrêmes” via `abs(diff(log(price)))`.

**`resample_price(price, freq="1D", method="last|first|mean") -> Series`**

**`clean_price_pipeline(raw, cfg, timestamp_col=None) -> Series`**
Pipeline end-to-end → `price_clean`.

### `feature_engineering.py`

**`FeatureConfig`**

- `momentum_windows: [5,20,60]`
- `vol_windows: [20]`
- `zscore_windows: [20]`
- `epsilon: 1e-12`
- `decision_shift: 1` (clé : garantit l’absence de look-ahead)

**`compute_log_returns(price) -> Series`** → `ret_1`

**`momentum(returns, window) -> Series`**

**`rolling_vol(returns, window) -> Series`** (std, `ddof=0`)

**`zscore(series, window, epsilon=...) -> Series`**

**`build_features(price, cfg) -> DataFrame`**
Construit features puis applique `shift(cfg.decision_shift)` et `dropna`.

**`validate_no_lookahead(features) -> None`**
Garde-fous : index tz-aware, monotone/unique, pas de NaN, `decision_shift >= 1`.

### `stationarity.py`

**`run_adf(series, maxlag=None, regression="c") -> dict`**

**`run_kpss(series, regression="c", nlags="auto") -> dict`**

**`stationarity_report({name: series, ...}) -> DataFrame`**
Retourne un report avec stat/pvalues/lags + critical values.

---

## Quickstart

### Prérequis

Python : À compléter (version exacte non détectée ici)

Dépendances minimales (déduites des imports) :

- pandas
- numpy
- statsmodels
- pytest (pour tests)

À compléter : le mode d’installation standard du monorepo (poetry/uv/pip, etc.).
Si vous n’avez aucune convention repo, vous pouvez au minimum installer les dépendances Python ci-dessus.

### Exemple d’usage (pipeline complet)

```python
import numpy as np
import pandas as pd

from quant_fx_system.quant.data_pipeline.clean_align import CleanAlignConfig, clean_price_pipeline
from quant_fx_system.quant.data_pipeline.feature_engineering import FeatureConfig, build_features, validate_no_lookahead
from quant_fx_system.quant.data_pipeline.stationarity import stationarity_report

# Exemple : série daily (index tz-aware recommandé)
idx = pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC")
price_raw = pd.Series(1.0 + np.cumsum(np.random.normal(0, 0.001, size=len(idx))), index=idx, name="price")

# 1) Clean
clean_cfg = CleanAlignConfig(resample_freq="1D", enable_outlier_filter=True, outlier_max_abs_log_return=0.10)
price_clean = clean_price_pipeline(price_raw, cfg=clean_cfg)

# 2) Features no-lookahead
feat_cfg = FeatureConfig(momentum_windows=[5, 20], vol_windows=[20], zscore_windows=[20], decision_shift=1)
features = build_features(price_clean, cfg=feat_cfg)
validate_no_lookahead(features)

# 3) Stationarity report (sur retours, ou sur une feature)
report = stationarity_report({"ret_1": features["ret_1"]})
print(report)
```

### Lancer les tests

À compléter selon l’outillage repo, mais ce dossier expose déjà des tests pytest :

```bash
pytest packages/backend/tests/test_data_pipeline.py
```

---

## Qualité

- Tests : `packages/backend/tests/test_data_pipeline.py`
- timezone UTC
- déduplication keep-last
- resampling daily last
- vérification du `decision_shift` (no-lookahead)
- smoke test stationarity report

Sécurité / secrets : aucune variable d’environnement ni secret détecté dans ce dossier.

---

## Contribution

PR petites et ciblées (une responsabilité par commit).

Conventional Commits (recommandé) :

- `feat(data_pipeline): ...`
- `fix(data_pipeline): ...`
- `test(data_pipeline): ...`
- `refactor(data_pipeline): ...`

---

## Licence
Business Source License 1.1 (BSL 1.1).

---

## Crédits

Florian Mauduit.
