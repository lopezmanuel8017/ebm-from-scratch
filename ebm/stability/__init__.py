"""
Stability techniques for Energy-Based Model training.

This module provides stability utilities that prevent training from diverging:

- **Spectral Normalization**: Constrains the Lipschitz constant of each layer
  by normalizing weights by their spectral norm (largest singular value).
  This prevents the energy landscape from becoming too "spiky".

- **Energy Clamping**: Prevents energy values from causing numerical overflow
  by clamping them to a safe range.

- **Gradient Clipping**: (In langevin.py) Clips per-sample gradients to prevent
  Langevin chains from diverging.

- **Learning Rate Warmup**: (In optimizer.py) Gradually increases learning rate
  to prevent early training instability.

Usage:
    >>> from ebm.stability import spectral_norm, apply_spectral_norm
    >>> from ebm.stability import clamp_energy, EnergyClipper
    >>> from ebm.stability import StabilityConfig
    >>>
    >>> # Apply spectral normalization to a weight matrix
    >>> W_normalized, u = spectral_norm(W)
    >>>
    >>> # Clamp energy values
    >>> energy = clamp_energy(energy, min_val=-100, max_val=100)

Common Failure Modes and Fixes:
    - Energy -> infinity: Langevin diverging. Reduce step size, increase grad clip.
    - E_real = E_fake: Model not learning. Increase Langevin steps, check gradients.
    - Samples collapse to point: Mode collapse. Add entropy regularization.
    - NaN in training: Numerical overflow. Add energy clamping, check softplus/log.
    - Slow convergence: Step size too small. Increase learning rate, reduce reg.
"""

from ebm.stability.spectral_norm import (
    spectral_norm,
    spectral_norm_power_iteration,
    SpectralNormState,
    apply_spectral_norm_to_layer,
    apply_spectral_norm_to_model,
    remove_spectral_norm_from_layer,
    SpectralNormWrapper,
)

from ebm.stability.energy_clamp import (
    clamp_energy,
    clamp_energy_tensor,
    EnergyClipper,
    SoftClamp,
    soft_clamp,
)

from ebm.stability.config import (
    StabilityConfig,
    get_stability_diagnostics,
    check_training_stability,
    StabilityDiagnostics,
)

__all__ = [
    "spectral_norm",
    "spectral_norm_power_iteration",
    "SpectralNormState",
    "apply_spectral_norm_to_layer",
    "apply_spectral_norm_to_model",
    "remove_spectral_norm_from_layer",
    "SpectralNormWrapper",
    "clamp_energy",
    "clamp_energy_tensor",
    "EnergyClipper",
    "SoftClamp",
    "soft_clamp",
    "StabilityConfig",
    "get_stability_diagnostics",
    "check_training_stability",
    "StabilityDiagnostics",
]
