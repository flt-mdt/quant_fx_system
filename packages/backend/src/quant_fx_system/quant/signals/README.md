# `quant.signals` — Génération de signaux (alpha → position) pour FX

Ce dossier fournit une couche **signal engineering** “production-grade” : à partir d’un **feature frame** (time-series, UTC, no-lookahead), on produit un **alpha** puis une **position** bornée (leverage), prête à être consommée par un moteur de backtest/portfolio.

## Table des matières

- [Aperçu](#aperçu)
- [Key features](#key-features)
- [Architecture (high-level)](#architecture-high-level)
- [Stack & Architecture](#stack--architecture)
  - [Tech Stack](#tech-stack)
  - [Arborescence](#arborescence)
  - [Patterns & principes](#patterns--principes)
- [Démarrage rapide](#démarrage-rapide)
  - [Prérequis](#prérequis)
  - [Exemple minimal](#exemple-minimal)
  - [Lancer les tests](#lancer-les-tests)
- [API du module](#api-du-module)
  - [`BaseSignal`](#basesignal)
  - [Signaux fournis](#signaux-fournis)
  - [Transforms](#transforms)
  - [Validation](#validation)
- [Configuration & variables d’environnement](#configuration--variables-denvironnement)
- [Qualité & sécurité](#qualité--sécurité)
- [Roadmap & contribution](#roadmap--contribution)
- [Licence](#licence)
- [Crédits](#crédits)

## Aperçu

Le **contrat** est simple :

1. Entrée : `features: pd.DataFrame` indexé par `DatetimeIndex` **UTC**, strictement croissant, unique, avec `features.attrs["decision_shift"] >= 1`.
2. Sortie : `SignalResult(alpha, position, metadata)`
   - `alpha` : score non borné (ou faiblement borné)
   - `position` : exposition finale bornée dans `[-max_leverage, +max_leverage]`

> Objectif implicite : **zéro look-ahead**. Les features doivent déjà être “decision-grade” (shift) côté `data_pipeline`.

## Key features

- **Interface stable** via `BaseSignal` : `compute_alpha()` + `compute_position()` + `run()`.
- **Validation stricte** des index time-series (UTC, monotone, unique) et du contrat `decision_shift`.
- **Signaux prêts à l’emploi** : momentum z-score, mean reversion (négatif du momentum), plus un **Ensemble** pondéré.
- **Transforms réutilisables** pour passer d’un score à une position : `tanh`/`clip`, z-score rolling, winsorization, scaling.
- **Output bundle typé** (`SignalResult`) + `metadata` utile pour le debug/audit.

## Architecture (high-level)

```mermaid
flowchart LR
  DP[quant/data_pipeline<br/>clean_align + feature_engineering] -->|features (UTC, decision_shift)| SIG[quant/signals<br/>alpha + position]
  SIG -->|position series| BT[quant/backtest]
  BT --> EV[quant/evaluation]
```

## Stack & Architecture

### Tech Stack

| Couche | Outils | Détails |
| --- | --- | --- |
| Core | Python | `quant_fx_system.quant.signals` |
| Time-series | pandas | DataFrame, Series, DatetimeIndex (UTC) |
| Numeric | numpy | tanh, stats rolling, RNG en tests |
| Tests | pytest | tests unitaires sur signaux & invariants |

### Arborescence

```
quant/signals/
  __init__.py
  base.py
  ensemble.py
  mean_reversion.py
  microstructure.py
  momentum.py
  signal_registry.py
  transforms.py
  types.py
  validation.py
```

### Patterns & principes

- Strategy pattern : chaque signal = une stratégie (`BaseSignal`).
- Separation of concerns :
  - `validation.py` : invariants time-series / features
  - `transforms.py` : fonctions pures, réutilisables
  - `types.py` : contrat de sortie stable (`SignalResult`)
- Défense contre erreurs quant classiques : index non-UTC, duplicates, séries non alignées, NaNs en sortie, etc.

## Démarrage rapide

### Prérequis

- Python ≥ 3.10 (déduit du code)
- Dépendances Python : pandas, numpy
- Tests : pytest (si tu exécutes les tests)
- À compléter (si applicable au repo) : méthode d’installation (Poetry/pip/uv), versions exactes, make targets.

### Exemple minimal

```python
import numpy as np
import pandas as pd

from quant_fx_system.quant.signals.ensemble import EnsembleConfig, EnsembleSignal
from quant_fx_system.quant.signals.mean_reversion import (
    MeanReversionZScoreConfig,
    MeanReversionZScoreSignal,
)
from quant_fx_system.quant.signals.momentum import (
    MomentumZScoreConfig,
    MomentumZScoreSignal,
)

# 1) Features: index UTC, monotone, unique
idx = pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC")
rng = np.random.default_rng(42)

features = pd.DataFrame(
    {
        # soit un z-momentum déjà calculé
        "z_mom_20": rng.normal(size=len(idx)),
        # soit au minimum un ret_1 (log-return 1 période)
        "ret_1": rng.normal(scale=0.01, size=len(idx)),
    },
    index=idx,
)

# 2) IMPORTANT : le contrat "decision-grade"
features.attrs["decision_shift"] = 1  # >= 1

# 3) Signaux
mom = MomentumZScoreSignal(
    MomentumZScoreConfig(window=20, max_leverage=1.0, method="tanh", k=1.0)
)
mr = MeanReversionZScoreSignal(
    MeanReversionZScoreConfig(window=20, max_leverage=1.0, method="tanh", k=1.0)
)

# 4) Exécution
mom_res = mom.run(features)
mr_res = mr.run(features)

# 5) Ensemble (pondéré)
ens = EnsembleSignal(
    [mom, mr],
    EnsembleConfig(
        weights={mom.name: 0.6, mr.name: 0.4},
        max_leverage=1.0,
        normalize_weights=True,
        combine="sum",
        post_transform="tanh",
    ),
)
ens_res = ens.run(features)

print(mom_res.position.head())
print(ens_res.metadata)
```

### Lancer les tests

```bash
pytest packages/backend/tests/test_signals.py -q
```

À compléter : commande “tous tests” / tooling (si présent dans le repo : `make test`, `tox`, `poetry run pytest`, etc.).

## API du module

### `BaseSignal`

- `compute_alpha(features: pd.DataFrame) -> pd.Series`
  - Produit le score brut (ex : z-score momentum).
- `compute_position(alpha: pd.Series, features: pd.DataFrame) -> pd.Series`
  - Transforme le score en exposition bornée.
- `run(features: pd.DataFrame) -> SignalResult`
  - Orchestration + validations (features, alpha, position) + metadata.

### Signaux fournis

- `MomentumZScoreSignal`
  - Alpha : `z_mom_{window}` si présent, sinon construit via `ret_1` (rolling sum + z-score).
  - Position : via `to_position_from_score(method="tanh"|"clip")`.
- `MeanReversionZScoreSignal`
  - Alpha : négatif du momentum (mean reversion).
  - Position : idem.
- `EnsembleSignal`
  - Combine plusieurs signaux avec `weights` (normalisation optionnelle).
  - Post-transform optionnel pour borner la position (`tanh` / `clip` / `None`).

### Transforms

Dans `transforms.py` :

- `to_position_from_score(score, method, max_leverage, k=1.0)`
  - `tanh` : `max_leverage * tanh(k * score)`
  - `clip` : normalise puis clip dans `[-max_leverage, +max_leverage]` (logique de saturation)
- `zscore_rolling(series, window, epsilon=1e-12)`
- `winsorize_by_quantile(series, q=0.01)`
- `scale_to_target_std(series, target=1.0, window=60, epsilon=1e-12)`

### Validation

Dans `validation.py` :

- `validate_features_for_signals(features, allow_nans=False)`
  - DatetimeIndex UTC, monotone, unique
  - `features.attrs["decision_shift"]` requis et >= 1
  - NaNs interdits si `allow_nans=False`
- `validate_series_index(series, name=...)`
  - mêmes invariants sur alpha/position

## Configuration & variables d’environnement

Aucune variable d’environnement spécifique à ce module.

## Qualité & sécurité

- Tests unitaires : invariants d’index, bornage des positions, cohérence mean-reversion vs momentum, ensemble.
- Fail fast : les erreurs “classiques” (NaNs, index non UTC, mauvaise alignement) lèvent explicitement.
- Aucun secret / aucune IO : code purement déterministe (hors RNG de tests), pas d’accès réseau/DB.

## Roadmap & contribution

Ajouts naturels :

- nouveaux signaux (carry, value, vol-regime, trend filters)
- standardisation cross-asset (normalisation, risk parity inputs)
- “signal diagnostics” (IC, turnover, exposure stats) côté evaluation/

## Licence

À préciser au niveau du repo (voir `LICENSE` si applicable).

## Crédits

Équipe Quant FX / Contributors.
