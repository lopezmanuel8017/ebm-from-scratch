"""
Energy Clamping for numerical stability in Energy-Based Models.

Energy clamping prevents energy values from causing numerical overflow during
training. Without clamping, energy values can grow to extreme values (+/- infinity)
which causes NaN gradients and training failure.

Two approaches are provided:
1. Hard clamping: Clips energy to [min_val, max_val] (discontinuous gradient)
2. Soft clamping: Smoothly constrains energy using tanh (continuous gradient)

Usage:
    >>> from ebm.stability import clamp_energy, soft_clamp, EnergyClipper
    >>>
    >>> # Hard clamp energy values
    >>> energy_clamped = clamp_energy(energy, min_val=-100, max_val=100)
    >>>
    >>> # Soft clamp for smoother gradients
    >>> energy_soft = soft_clamp(energy, limit=100, steepness=0.1)
    >>>
    >>> # Use EnergyClipper as a training callback
    >>> clipper = EnergyClipper(min_energy=-100, max_energy=100)
    >>> clipper.clip(energy)
"""

import numpy as np
from dataclasses import dataclass
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ebm.core.autodiff import Tensor


def clamp_energy(
    energy: Union[np.ndarray, float],
    min_val: float = -100.0,
    max_val: float = 100.0,
) -> np.ndarray:
    """
    Clamp energy values to a safe range.

    Hard clamping clips values outside [min_val, max_val]. This is a simple
    approach but has discontinuous gradients at the clamp boundaries.

    Args:
        energy: Energy values (numpy array or scalar)
        min_val: Minimum allowed energy value (default: -100)
        max_val: Maximum allowed energy value (default: 100)

    Returns:
        Clamped energy values with same shape as input

    Example:
        >>> energy = np.array([-500, -50, 0, 50, 500])
        >>> clamped = clamp_energy(energy, min_val=-100, max_val=100)
        >>> clamped  # array([-100, -50, 0, 50, 100])
    """
    return np.clip(energy, min_val, max_val)


def clamp_energy_tensor(
    energy: 'Tensor',
    min_val: float = -100.0,
    max_val: float = 100.0,
) -> 'Tensor':
    """
    Clamp energy Tensor values to a safe range.

    Note: This breaks the computational graph - gradients will not flow
    through the clamp operation. Use soft_clamp for differentiable clamping.

    Args:
        energy: Energy Tensor
        min_val: Minimum allowed energy value (default: -100)
        max_val: Maximum allowed energy value (default: 100)

    Returns:
        New Tensor with clamped values (detached from computational graph)

    Example:
        >>> from ebm.core.autodiff import Tensor
        >>> energy = Tensor(np.array([[-150], [50], [200]]))
        >>> clamped = clamp_energy_tensor(energy)
        >>> clamped.data  # array([[-100], [50], [100]])
    """
    from ebm.core.autodiff import Tensor

    clamped_data = np.clip(energy.data, min_val, max_val)
    return Tensor(clamped_data, requires_grad=False)


def soft_clamp(
    x: Union[np.ndarray, float],
    limit: float = 100.0,
    steepness: float = 0.1,
) -> np.ndarray:
    """
    Soft clamping using scaled tanh.

    Maps values to approximately [-limit, limit] with smooth transitions.
    Unlike hard clamping, gradients are always defined and continuous.

    The function: f(x) = limit * tanh(steepness * x / limit)

    Properties:
    - f(0) = 0
    - f(x) -> limit as x -> infinity
    - f(x) -> -limit as x -> -infinity
    - f'(0) = steepness (gradient at origin)

    Args:
        x: Input values (numpy array or scalar)
        limit: Soft limit for output values (default: 100)
        steepness: Controls transition sharpness (default: 0.1)
                   Higher values = sharper transition to limits

    Returns:
        Soft-clamped values in approximately [-limit, limit]

    Example:
        >>> x = np.array([-1000, -100, 0, 100, 1000])
        >>> soft = soft_clamp(x, limit=100, steepness=0.1)
        >>> # Values are smoothly mapped to [-100, 100]
    """
    x = np.asarray(x)
    return limit * np.tanh(steepness * x / limit)


def soft_clamp_gradient(
    x: Union[np.ndarray, float],
    limit: float = 100.0,
    steepness: float = 0.1,
) -> np.ndarray:
    """
    Gradient of soft_clamp function.

    Useful for implementing differentiable soft clamping in the autodiff framework.

    Args:
        x: Input values where gradient is evaluated
        limit: Soft limit (same as in soft_clamp)
        steepness: Steepness parameter (same as in soft_clamp)

    Returns:
        Gradient values d(soft_clamp)/dx
    """
    x = np.asarray(x)
    tanh_val = np.tanh(steepness * x / limit)
    return steepness * (1 - tanh_val ** 2)


class SoftClamp:
    """
    Soft clamping layer for energy values.

    Provides a callable interface for soft clamping with fixed parameters.

    Attributes:
        limit: Soft limit for output values
        steepness: Transition sharpness parameter

    Example:
        >>> clamp = SoftClamp(limit=100, steepness=0.1)
        >>> energy = np.array([-500, 0, 500])
        >>> clamped = clamp(energy)
    """

    def __init__(self, limit: float = 100.0, steepness: float = 0.1):
        """
        Initialize soft clamp.

        Args:
            limit: Soft limit for output values (default: 100)
            steepness: Transition sharpness (default: 0.1)
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        if steepness <= 0:
            raise ValueError(f"steepness must be positive, got {steepness}")

        self.limit = limit
        self.steepness = steepness

    def __call__(self, x: Union[np.ndarray, float]) -> np.ndarray:
        """Apply soft clamping."""
        return soft_clamp(x, self.limit, self.steepness)

    def gradient(self, x: Union[np.ndarray, float]) -> np.ndarray:
        """Compute gradient of soft clamp."""
        return soft_clamp_gradient(x, self.limit, self.steepness)

    def __repr__(self) -> str:
        return f"SoftClamp(limit={self.limit}, steepness={self.steepness})"


@dataclass
class EnergyClipperConfig:
    """
    Configuration for energy clipping.

    Attributes:
        min_energy: Minimum allowed energy value
        max_energy: Maximum allowed energy value
        use_soft_clamp: Whether to use soft clamping (True) or hard clamp (False)
        soft_steepness: Steepness for soft clamping
        warn_on_clip: Whether to issue a warning when clipping occurs
    """
    min_energy: float = -100.0
    max_energy: float = 100.0
    use_soft_clamp: bool = False
    soft_steepness: float = 0.1
    warn_on_clip: bool = False


class EnergyClipper:
    """
    Energy clipper that can be used during training.

    Tracks statistics about clipping events to help diagnose training issues.

    Attributes:
        config: EnergyClipperConfig with clipping parameters
        n_clips: Number of values that have been clipped
        n_total: Total number of values processed
        max_observed: Maximum energy value observed
        min_observed: Minimum energy value observed

    Example:
        >>> clipper = EnergyClipper(min_energy=-100, max_energy=100)
        >>> energy = np.array([-500, 0, 500])
        >>> clamped = clipper.clip(energy)
        >>> print(clipper.clip_ratio)  # Fraction of values clipped
    """

    def __init__(
        self,
        min_energy: float = -100.0,
        max_energy: float = 100.0,
        use_soft_clamp: bool = False,
        soft_steepness: float = 0.1,
        warn_on_clip: bool = False,
    ):
        """
        Initialize energy clipper.

        Args:
            min_energy: Minimum allowed energy (default: -100)
            max_energy: Maximum allowed energy (default: 100)
            use_soft_clamp: Use soft clamping instead of hard (default: False)
            soft_steepness: Steepness for soft clamping (default: 0.1)
            warn_on_clip: Print warning when clipping occurs (default: False)
        """
        if min_energy >= max_energy:
            raise ValueError(
                f"min_energy ({min_energy}) must be less than max_energy ({max_energy})"
            )

        self.config = EnergyClipperConfig(
            min_energy=min_energy,
            max_energy=max_energy,
            use_soft_clamp=use_soft_clamp,
            soft_steepness=soft_steepness,
            warn_on_clip=warn_on_clip,
        )

        self.n_clips = 0
        self.n_total = 0
        self.max_observed = float('-inf')
        self.min_observed = float('inf')
        self._clip_history = []

    def clip(self, energy: Union[np.ndarray, float]) -> np.ndarray:
        """
        Clip energy values and update statistics.

        Args:
            energy: Energy values to clip

        Returns:
            Clipped energy values
        """
        energy = np.asarray(energy)
        n_values = energy.size

        self.n_total += n_values
        self.max_observed = max(self.max_observed, float(energy.max()))
        self.min_observed = min(self.min_observed, float(energy.min()))

        n_clipped = np.sum(
            (energy < self.config.min_energy) | (energy > self.config.max_energy)
        )
        self.n_clips += n_clipped

        if n_clipped > 0:
            self._clip_history.append({
                'n_clipped': int(n_clipped),
                'n_total': n_values,
                'max': float(energy.max()),
                'min': float(energy.min()),
            })

            if self.config.warn_on_clip:
                import warnings
                warnings.warn(
                    f"Energy clipping: {n_clipped}/{n_values} values clipped. "
                    f"Range: [{energy.min():.2f}, {energy.max():.2f}]"
                )

        if self.config.use_soft_clamp:
            center = (self.config.max_energy + self.config.min_energy) / 2
            half_range = (self.config.max_energy - self.config.min_energy) / 2
            return center + soft_clamp(
                energy - center, half_range, self.config.soft_steepness
            )
        else:
            return clamp_energy(
                energy, self.config.min_energy, self.config.max_energy
            )

    def clip_tensor(self, energy: 'Tensor') -> 'Tensor':
        """
        Clip Tensor energy values.

        Note: This breaks the computational graph.

        Args:
            energy: Energy Tensor to clip

        Returns:
            Clipped Tensor (detached)
        """
        from ebm.core.autodiff import Tensor

        clipped_data = self.clip(energy.data)
        return Tensor(clipped_data, requires_grad=False)

    @property
    def clip_ratio(self) -> float:
        """Fraction of values that have been clipped."""
        if self.n_total == 0:
            return 0.0
        return self.n_clips / self.n_total

    @property
    def observed_range(self) -> tuple:
        """Range of observed energy values."""
        return (self.min_observed, self.max_observed)

    def reset_stats(self) -> None:
        """Reset clipping statistics."""
        self.n_clips = 0
        self.n_total = 0
        self.max_observed = float('-inf')
        self.min_observed = float('inf')
        self._clip_history = []

    def get_stats(self) -> dict:
        """Get clipping statistics."""
        return {
            'n_clips': self.n_clips,
            'n_total': self.n_total,
            'clip_ratio': self.clip_ratio,
            'max_observed': self.max_observed,
            'min_observed': self.min_observed,
            'observed_range': self.observed_range,
        }

    def needs_attention(self, threshold: float = 0.1) -> bool:
        """
        Check if clipping ratio exceeds threshold.

        High clipping ratios indicate training instability.

        Args:
            threshold: Clip ratio threshold (default: 0.1 = 10%)

        Returns:
            True if clip_ratio > threshold
        """
        return self.clip_ratio > threshold

    def __repr__(self) -> str:
        return (
            f"EnergyClipper(min={self.config.min_energy}, max={self.config.max_energy}, "
            f"clip_ratio={self.clip_ratio:.2%})"
        )


def check_energy_stability(
    energy: Union[np.ndarray, float],
    threshold: float = 100.0,
) -> dict:
    """
    Check energy values for stability issues.

    Args:
        energy: Energy values to check
        threshold: Values beyond +/- threshold are considered problematic

    Returns:
        Dictionary with stability diagnostics:
            - is_stable: True if all values are within threshold
            - has_nan: True if any NaN values
            - has_inf: True if any infinite values
            - max_abs: Maximum absolute value
            - n_problematic: Number of problematic values
    """
    energy = np.asarray(energy)

    has_nan = bool(np.isnan(energy).any())
    has_inf = bool(np.isinf(energy).any())

    energy_clean = np.where(np.isfinite(energy), energy, 0)
    max_abs = float(np.abs(energy_clean).max()) if energy_clean.size > 0 else 0.0

    n_problematic = np.sum(
        ~np.isfinite(energy) | (np.abs(energy) > threshold)
    )

    is_stable = bool(not has_nan and not has_inf and (max_abs <= threshold))

    return {
        'is_stable': is_stable,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'max_abs': max_abs,
        'n_problematic': int(n_problematic),
        'n_total': energy.size,
    }
