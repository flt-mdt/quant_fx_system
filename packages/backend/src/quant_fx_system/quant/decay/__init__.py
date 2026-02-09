"""Causal decay primitives for time-series signals and position targets."""

from quant_fx_system.quant.decay.core import (
    apply_decay,
    decay_position_target,
    half_life_to_alpha,
)
from quant_fx_system.quant.decay.types import DecayConfig, DecayResult
from quant_fx_system.quant.decay.weights import generate_weights

__all__ = [
    "DecayConfig",
    "DecayResult",
    "apply_decay",
    "decay_position_target",
    "generate_weights",
    "half_life_to_alpha",
]
