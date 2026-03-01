"""
Stability configuration and diagnostics for Energy-Based Model training.

This module provides:
- StabilityConfig: Configuration for all stability techniques
- StabilityDiagnostics: Diagnostics to monitor training stability
- Utility functions for checking and monitoring stability

Common failure modes and their indicators:
    - Energy -> infinity: Check max_energy in diagnostics
    - E_real = E_fake: Check energy_gap in diagnostics
    - Mode collapse: Check sample variance and entropy
    - NaN in training: Check has_nan in diagnostics
    - Gradient explosion: Check max_grad_norm in diagnostics

Usage:
    >>> from ebm.stability import StabilityConfig, check_training_stability
    >>>
    >>> config = StabilityConfig(
    ...     grad_clip=0.03,
    ...     energy_clamp_min=-100,
    ...     energy_clamp_max=100,
    ...     use_spectral_norm=True,
    ... )
    >>>
    >>> # During training
    >>> diagnostics = check_training_stability(
    ...     E_real=e_real, E_fake=e_fake,
    ...     grad_norms=grad_norms, samples=samples
    ... )
    >>> if not diagnostics.is_stable:
    ...     print("Warning:", diagnostics.warnings)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union


@dataclass
class StabilityConfig:
    """
    Configuration for stability techniques in EBM training.

    This consolidates all stability-related hyperparameters in one place.

    Attributes:
        # Gradient clipping (Langevin dynamics)
        grad_clip: Maximum gradient norm for Langevin (default: 0.03)

        # Energy clamping
        energy_clamp_enabled: Whether to clamp energy values (default: False)
        energy_clamp_min: Minimum energy value (default: -100)
        energy_clamp_max: Maximum energy value (default: 100)
        use_soft_clamp: Use soft clamping instead of hard (default: False)
        soft_clamp_steepness: Steepness for soft clamping (default: 0.1)

        # Spectral normalization
        use_spectral_norm: Whether to use spectral normalization (default: False)
        spectral_norm_iterations: Power iterations per forward pass (default: 1)

        # Learning rate warmup
        use_lr_warmup: Whether to use LR warmup (default: True)
        warmup_steps: Number of warmup steps (default: 1000)

        # Monitoring thresholds
        max_energy_threshold: Warn if |energy| exceeds this (default: 50)
        min_energy_gap_threshold: Warn if |E_real - E_fake| < this (default: 0.01)
        max_grad_norm_threshold: Warn if grad norm exceeds this (default: 10)
        sample_divergence_threshold: Warn if sample norm exceeds this (default: 100)
    """
    grad_clip: float = 0.03

    energy_clamp_enabled: bool = False
    energy_clamp_min: float = -100.0
    energy_clamp_max: float = 100.0
    use_soft_clamp: bool = False
    soft_clamp_steepness: float = 0.1

    use_spectral_norm: bool = False
    spectral_norm_iterations: int = 1

    use_lr_warmup: bool = True
    warmup_steps: int = 1000

    max_energy_threshold: float = 50.0
    min_energy_gap_threshold: float = 0.01
    max_grad_norm_threshold: float = 10.0
    sample_divergence_threshold: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'grad_clip': self.grad_clip,
            'energy_clamp_enabled': self.energy_clamp_enabled,
            'energy_clamp_min': self.energy_clamp_min,
            'energy_clamp_max': self.energy_clamp_max,
            'use_soft_clamp': self.use_soft_clamp,
            'soft_clamp_steepness': self.soft_clamp_steepness,
            'use_spectral_norm': self.use_spectral_norm,
            'spectral_norm_iterations': self.spectral_norm_iterations,
            'use_lr_warmup': self.use_lr_warmup,
            'warmup_steps': self.warmup_steps,
            'max_energy_threshold': self.max_energy_threshold,
            'min_energy_gap_threshold': self.min_energy_gap_threshold,
            'max_grad_norm_threshold': self.max_grad_norm_threshold,
            'sample_divergence_threshold': self.sample_divergence_threshold,
        }


@dataclass
class StabilityDiagnostics:
    """
    Diagnostics for monitoring training stability.

    These diagnostics help identify common training issues:
    - Energy divergence: max_energy or min_energy extreme
    - Mode collapse: very small energy_gap or low sample diversity
    - Gradient issues: extreme grad_norms or has_nan
    - Sample divergence: samples moving to extreme values

    Attributes:
        is_stable: Overall stability assessment
        warnings: List of warning messages
        metrics: Dictionary of diagnostic metrics
    """
    is_stable: bool = True
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    has_nan: bool = False
    has_inf: bool = False
    energy_diverging: bool = False
    gradient_exploding: bool = False
    samples_diverging: bool = False
    mode_collapse_risk: bool = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        self.is_stable = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostics to dictionary."""
        return {
            'is_stable': self.is_stable,
            'warnings': self.warnings,
            'metrics': self.metrics,
            'has_nan': self.has_nan,
            'has_inf': self.has_inf,
            'energy_diverging': self.energy_diverging,
            'gradient_exploding': self.gradient_exploding,
            'samples_diverging': self.samples_diverging,
            'mode_collapse_risk': self.mode_collapse_risk,
        }

    def __repr__(self) -> str:
        status = "STABLE" if self.is_stable else "UNSTABLE"
        if self.warnings:
            return f"StabilityDiagnostics({status}, warnings={self.warnings})"
        return f"StabilityDiagnostics({status})"


def check_training_stability(
    E_real: Optional[Union[np.ndarray, float]] = None,
    E_fake: Optional[Union[np.ndarray, float]] = None,
    grad_norms: Optional[Union[np.ndarray, List[float]]] = None,
    samples: Optional[np.ndarray] = None,
    loss: Optional[float] = None,
    config: Optional[StabilityConfig] = None,
) -> StabilityDiagnostics:
    """
    Check training stability and return diagnostics.

    Analyzes various training metrics to detect common failure modes:
    - NaN/Inf values
    - Energy divergence
    - Gradient explosion
    - Sample divergence
    - Mode collapse risk

    Args:
        E_real: Energy values for real data
        E_fake: Energy values for fake/generated data
        grad_norms: Gradient norms from training
        samples: Generated samples
        loss: Current loss value
        config: StabilityConfig with thresholds (uses defaults if None)

    Returns:
        StabilityDiagnostics with analysis results

    Example:
        >>> diagnostics = check_training_stability(
        ...     E_real=np.array([-5.0, -3.0]),
        ...     E_fake=np.array([2.0, 4.0]),
        ...     grad_norms=[0.1, 0.2, 0.15],
        ...     samples=np.random.randn(100, 2)
        ... )
        >>> if not diagnostics.is_stable:
        ...     print("Issues detected:", diagnostics.warnings)
    """
    if config is None:
        config = StabilityConfig()

    diagnostics = StabilityDiagnostics()
    metrics = {}

    def check_nan(x, name):
        """Check a value for NaN and Inf, add warnings if found."""
        if x is not None:
            x = np.asarray(x)
            if np.isnan(x).any():
                diagnostics.has_nan = True
                diagnostics.add_warning(f"NaN detected in {name}")
            if np.isinf(x).any():
                diagnostics.has_inf = True
                diagnostics.add_warning(f"Inf detected in {name}")

    check_nan(E_real, "E_real")
    check_nan(E_fake, "E_fake")
    check_nan(grad_norms, "grad_norms")
    check_nan(samples, "samples")
    check_nan(loss, "loss")

    if E_real is not None:
        E_real = np.asarray(E_real)
        metrics['E_real_mean'] = float(np.mean(E_real))
        metrics['E_real_max'] = float(np.max(E_real))
        metrics['E_real_min'] = float(np.min(E_real))

        if np.abs(E_real).max() > config.max_energy_threshold:
            diagnostics.energy_diverging = True
            diagnostics.add_warning(
                f"E_real extreme values: [{E_real.min():.2f}, {E_real.max():.2f}]"
            )

    if E_fake is not None:
        E_fake = np.asarray(E_fake)
        metrics['E_fake_mean'] = float(np.mean(E_fake))
        metrics['E_fake_max'] = float(np.max(E_fake))
        metrics['E_fake_min'] = float(np.min(E_fake))

        if np.abs(E_fake).max() > config.max_energy_threshold:
            diagnostics.energy_diverging = True
            diagnostics.add_warning(
                f"E_fake extreme values: [{E_fake.min():.2f}, {E_fake.max():.2f}]"
            )

    if E_real is not None and E_fake is not None:
        energy_gap = float(np.mean(E_real) - np.mean(E_fake))
        metrics['energy_gap'] = energy_gap

        if abs(energy_gap) < config.min_energy_gap_threshold:
            diagnostics.mode_collapse_risk = True
            diagnostics.add_warning(
                f"Energy gap too small ({energy_gap:.4f}): model may not be learning"
            )

    if grad_norms is not None:
        grad_norms = np.asarray(grad_norms)
        metrics['grad_norm_mean'] = float(np.mean(grad_norms))
        metrics['grad_norm_max'] = float(np.max(grad_norms))

        if grad_norms.max() > config.max_grad_norm_threshold:
            diagnostics.gradient_exploding = True
            diagnostics.add_warning(
                f"Gradient explosion: max norm = {grad_norms.max():.2f}"
            )

    if samples is not None:
        sample_norms = np.linalg.norm(samples, axis=-1) if samples.ndim > 1 else np.abs(samples)
        metrics['sample_norm_mean'] = float(np.mean(sample_norms))
        metrics['sample_norm_max'] = float(np.max(sample_norms))
        metrics['sample_var'] = float(np.var(samples))

        if sample_norms.max() > config.sample_divergence_threshold:
            diagnostics.samples_diverging = True
            diagnostics.add_warning(
                f"Samples diverging: max norm = {sample_norms.max():.2f}"
            )

        if np.var(samples) < 0.01:
            diagnostics.mode_collapse_risk = True
            diagnostics.add_warning(
                f"Low sample variance ({np.var(samples):.4f}): possible mode collapse"
            )

    if loss is not None:
        metrics['loss'] = float(loss)
        if np.isnan(loss) or np.isinf(loss):
            diagnostics.add_warning("Loss is NaN or Inf")

    diagnostics.metrics = metrics
    return diagnostics


def get_stability_diagnostics(
    history: List[Dict[str, float]],
    window: int = 10,
) -> StabilityDiagnostics:
    """
    Analyze training history for stability issues.

    Examines recent training statistics to detect trends and issues.

    Args:
        history: List of training statistics dictionaries
        window: Number of recent steps to analyze (default: 10)

    Returns:
        StabilityDiagnostics based on history analysis
    """
    if not history:
        return StabilityDiagnostics()

    recent = history[-window:] if len(history) >= window else history

    losses = [h.get('loss', 0) for h in recent if 'loss' in h]
    e_reals = [h.get('E_real', 0) for h in recent if 'E_real' in h]
    e_fakes = [h.get('E_fake', 0) for h in recent if 'E_fake' in h]
    grad_norms = [h.get('grad_norm', 0) for h in recent if 'grad_norm' in h]

    return check_training_stability(
        E_real=np.array(e_reals) if e_reals else None,
        E_fake=np.array(e_fakes) if e_fakes else None,
        grad_norms=np.array(grad_norms) if grad_norms else None,
        loss=losses[-1] if losses else None,
    )


def suggest_fixes(diagnostics: StabilityDiagnostics) -> List[str]:
    """
    Suggest fixes based on stability diagnostics.

    Returns a list of actionable suggestions to improve training stability.

    Args:
        diagnostics: StabilityDiagnostics from check_training_stability

    Returns:
        List of suggested fixes
    """
    suggestions = []

    if diagnostics.has_nan or diagnostics.has_inf:
        suggestions.extend([
            "Enable energy clamping to prevent numerical overflow",
            "Reduce learning rate",
            "Check for log(0) or division by zero in loss computation",
        ])

    if diagnostics.energy_diverging:
        suggestions.extend([
            "Enable energy clamping: clamp_energy(energy, -100, 100)",
            "Increase energy regularization (alpha parameter)",
            "Reduce Langevin step size",
        ])

    if diagnostics.gradient_exploding:
        suggestions.extend([
            "Reduce gradient clip threshold (grad_clip parameter)",
            "Enable spectral normalization on layers",
            "Reduce learning rate",
        ])

    if diagnostics.samples_diverging:
        suggestions.extend([
            "Reduce Langevin step size",
            "Increase gradient clipping strength",
            "Check energy function for issues",
        ])

    if diagnostics.mode_collapse_risk:
        suggestions.extend([
            "Increase entropy regularization (lambda_ent parameter)",
            "Increase replay buffer reinitialization probability",
            "Increase Langevin steps to improve sample quality",
            "Check that E_real < E_fake on average (correct training direction)",
        ])

    return suggestions


STABILITY_CONFIG_CONSERVATIVE = StabilityConfig(
    grad_clip=0.01,
    energy_clamp_enabled=True,
    energy_clamp_min=-50.0,
    energy_clamp_max=50.0,
    use_spectral_norm=True,
    spectral_norm_iterations=1,
    use_lr_warmup=True,
    warmup_steps=2000,
)

STABILITY_CONFIG_DEFAULT = StabilityConfig(
    grad_clip=0.03,
    energy_clamp_enabled=False,
    use_spectral_norm=False,
    use_lr_warmup=True,
    warmup_steps=1000,
)

STABILITY_CONFIG_AGGRESSIVE = StabilityConfig(
    grad_clip=0.1,
    energy_clamp_enabled=False,
    use_spectral_norm=False,
    use_lr_warmup=False,
    warmup_steps=0,
)
