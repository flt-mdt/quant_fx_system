# Decay (smoothing & half-life)

## TL;DR
Le module `quant.decay` fournit des primitives **causales** (anti-lookahead) pour appliquer une décroissance temporelle sur des séries : smoothing (EWMA/EMA), poids de kernel (linear/step/power), et inertie de position. Il est conçu pour être **déterministe** et **auditable**.

```mermaid
flowchart LR
    Signals --> Decay --> Risk --> Backtest --> Evaluation
```

## Concepts quant clés
- **Forgetting factor / EWMA** : mise à jour récursive causale.
- **Half-life** : nombre de barres au bout duquel le poids d’une observation est divisé par 2.
- **Causalité** : aucune opération ne dépend du futur (`center=False` implicite, pas de convolution future).

### Half-life → alpha (bars régulières)
`alpha = 1 - exp(-ln(2) / half_life_bars)`

### Half-life time-aware (timestamps irréguliers)
`lambda_t = exp(-ln(2) * dt / half_life_time)`
`y_t = (1 - lambda_t) * x_t + lambda_t * y_{t-1}`

## Usage rapide
```python
import pandas as pd
from quant_fx_system.quant.decay import DecayConfig, apply_decay, decay_position_target

index = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
signal = pd.Series([0.0, 0.5, 1.0, 0.0, -0.5, 0.25], index=index)

result = apply_decay(
    signal,
    DecayConfig(kind="ewma", half_life_bars=5, min_periods=1),
)
print(result.output.tail())

target = pd.Series([0.0, 1.0, 1.0, -1.0, 0.0, 0.5], index=index)
position = decay_position_target(
    target,
    DecayConfig(kind="ewma", half_life_bars=3),
)
```

## Signal decay vs position decay
- **Signal decay** (`apply_decay`) : smoothing/agrégation causale, réduit la variance.
- **Position decay** (`decay_position_target`) : inertie de position (partial adjustment).

## Kernels disponibles
- **Exponential (EWMA)** : pondération exponentielle.
- **Linear / Step / Power** : poids causaux normalisés (somme = 1).

## Convention anti-lookahead
Le module **n’applique jamais de lookahead**. Un `shift` explicite est disponible si l’utilisateur souhaite appliquer un décalage (ex. signal connu à *t* appliqué à *t+1*). Le backtest peut déjà appliquer son propre `shift`, donc utilisez cette option avec parcimonie.

## Notes sur les NaN
- Si `fillna_value` est fourni, les NaN d’entrée sont remplis avant calcul.
- Sinon, les NaN sont conservés (et peuvent être propagés selon l’opérateur).
