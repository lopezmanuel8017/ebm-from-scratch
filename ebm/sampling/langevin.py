"""
Langevin dynamics sampler for Energy-Based Models.

Langevin dynamics is a physics-inspired MCMC method for sampling from
p(x) = exp(-E(x))/Z without computing the intractable partition function Z.

The discretized overdamped Langevin update rule (Euler-Maruyama):
    x_{t+1} = x_t - epsilon * grad_E(x_t) + sqrt(2 * epsilon) * noise_scale * eta

where:
    - epsilon is the step size
    - eta ~ N(0, I) is standard Gaussian noise
    - noise_scale = 1.0 for standard T=1 sampling
    - The factor sqrt(2 * epsilon) ensures correct stationary distribution

This module provides:
    - clip_grad: Per-sample gradient clipping for stability
    - langevin_sample: Main sampling function
    - langevin_sample_with_diagnostics: Sampling with convergence monitoring
    - Step size annealing schedules (linear, geometric)
    - Initialization strategies for Langevin chains
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

from ebm.core.autodiff import Tensor
from ebm.core.ops import tensor_sum



def clip_grad(grad: np.ndarray, max_norm: float) -> np.ndarray:
    """
    Clip per-sample gradient to maximum norm.

    Early in training, the energy function is random and gradients can be huge,
    sending samples to infinity. Clipping prevents chain divergence.

    Args:
        grad: Gradient array of shape (batch_size, dim)
        max_norm: Maximum gradient norm per sample

    Returns:
        Clipped gradient with same shape as input

    Example:
        >>> grad = np.array([[3.0, 4.0], [0.1, 0.2]])  # Norms: 5.0, 0.224
        >>> clipped = clip_grad(grad, max_norm=1.0)
        >>> np.linalg.norm(clipped[0])  # Will be 1.0
        >>> np.linalg.norm(clipped[1])  # Will be 0.224 (unchanged)
    """
    if grad.ndim == 1:
        grad_norm = np.sqrt((grad ** 2).sum() + 1e-8)
        scale = min(1.0, max_norm / grad_norm)
        return grad * scale

    grad_norm = np.sqrt((grad ** 2).sum(axis=-1, keepdims=True) + 1e-8)
    scale = np.minimum(1.0, max_norm / grad_norm)

    return grad * scale



def linear_annealing(
    step_size_start: float,
    step_size_end: float,
    n_steps: int,
) -> np.ndarray:
    """
    Linear step size annealing schedule.

    Linearly interpolates from step_size_start to step_size_end over n_steps.
    Useful for gradually reducing step size to get more accurate final samples.

    Args:
        step_size_start: Initial step size
        step_size_end: Final step size
        n_steps: Number of steps

    Returns:
        Array of step sizes of length n_steps

    Example:
        >>> schedule = linear_annealing(0.1, 0.01, 10)
        >>> schedule[0]  # 0.1
        >>> schedule[-1]  # 0.01
    """
    return np.linspace(step_size_start, step_size_end, n_steps)


def geometric_annealing(
    step_size_start: float,
    step_size_end: float,
    n_steps: int,
) -> np.ndarray:
    """
    Geometric (exponential) step size annealing schedule.

    Exponentially decays from step_size_start to step_size_end over n_steps.
    Provides faster initial decay than linear annealing.

    Args:
        step_size_start: Initial step size
        step_size_end: Final step size
        n_steps: Number of steps

    Returns:
        Array of step sizes of length n_steps

    Example:
        >>> schedule = geometric_annealing(0.1, 0.01, 10)
        >>> schedule[0]  # 0.1
        >>> schedule[-1]  # 0.01 (approximately)
    """
    if n_steps <= 1:
        return np.array([step_size_start])

    ratio = step_size_end / step_size_start
    t = np.arange(n_steps) / (n_steps - 1)
    return step_size_start * (ratio ** t)


def init_from_noise(
    n_samples: int,
    dim: int,
    noise_type: str = "uniform",
    low: float = -1.0,
    high: float = 1.0,
    std: float = 1.0,
) -> np.ndarray:
    """
    Initialize samples from noise distribution.

    Args:
        n_samples: Number of samples to generate
        dim: Dimensionality of each sample
        noise_type: Type of noise - 'uniform' or 'gaussian'
        low: Lower bound for uniform noise (default: -1.0)
        high: Upper bound for uniform noise (default: 1.0)
        std: Standard deviation for Gaussian noise (default: 1.0)

    Returns:
        Array of shape (n_samples, dim) with random samples

    Raises:
        ValueError: If noise_type is not 'uniform' or 'gaussian'

    Example:
        >>> x_init = init_from_noise(100, 2, noise_type='uniform')
        >>> x_init.shape  # (100, 2)
        >>> x_init.min() >= -1.0  # True
        >>> x_init.max() <= 1.0   # True

        >>> x_init = init_from_noise(100, 2, noise_type='gaussian', std=0.5)
        >>> x_init.shape  # (100, 2)
    """
    if noise_type == "uniform":
        return np.random.uniform(low, high, (n_samples, dim))
    elif noise_type == "gaussian":
        return np.random.randn(n_samples, dim) * std
    else:
        raise ValueError(
            f"Unknown noise_type: {noise_type}. Use 'uniform' or 'gaussian'."
        )


def init_from_data(
    data: np.ndarray,
    n_samples: int,
    noise_std: float = 0.0,
) -> np.ndarray:
    """
    Initialize samples near training data points.

    Randomly selects points from the data and optionally adds Gaussian noise.
    This can help Langevin chains start closer to data modes.

    Args:
        data: Training data array of shape (n_data, dim)
        n_samples: Number of samples to generate
        noise_std: Standard deviation of Gaussian noise to add (default: 0.0)

    Returns:
        Array of shape (n_samples, dim) initialized near data points

    Example:
        >>> data = np.random.randn(1000, 2)  # Training data
        >>> x_init = init_from_data(data, 100, noise_std=0.1)
        >>> x_init.shape  # (100, 2)
    """
    n_data = data.shape[0]
    indices = np.random.randint(0, n_data, n_samples)
    samples = data[indices].copy()

    if noise_std > 0:
        samples += np.random.randn(*samples.shape) * noise_std

    return samples


def init_mixed(
    buffer_samples: np.ndarray,
    n_samples: int,
    reinit_prob: float = 0.05,
    noise_type: str = "uniform",
    low: float = -1.0,
    high: float = 1.0,
    std: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mixed initialization: mostly from buffer, some from fresh noise.

    This is the recommended initialization strategy. Starting from buffer
    samples allows chains to begin near modes (faster convergence), while
    occasional fresh noise ensures continued exploration and prevents the
    buffer from getting stuck in local modes.

    Args:
        buffer_samples: Array of shape (n_buffer, dim) with buffered samples
        n_samples: Number of samples to generate
        reinit_prob: Probability of reinitializing from noise (default: 0.05)
        noise_type: Type of noise for reinitialization - 'uniform' or 'gaussian'
        low: Lower bound for uniform noise (default: -1.0)
        high: Upper bound for uniform noise (default: 1.0)
        std: Standard deviation for Gaussian noise (default: 1.0)

    Returns:
        Tuple of (samples, indices) where:
            - samples: Array of shape (n_samples, dim)
            - indices: Array of buffer indices that were sampled (for updating)

    Raises:
        ValueError: If noise_type is not 'uniform' or 'gaussian'
        ValueError: If reinit_prob is not in [0, 1]

    Example:
        >>> buffer = np.random.randn(1000, 2)  # Previous samples
        >>> x_init, indices = init_mixed(buffer, 64, reinit_prob=0.05)
        >>> x_init.shape  # (64, 2)
        >>> len(indices)  # 64

    Note:
        The indices returned can be used to update the buffer after Langevin
        refinement. Samples that were reinitialized from noise will still
        have valid indices (pointing to random buffer locations), but these
        can be identified by the reinit mask if needed.
    """
    if not 0.0 <= reinit_prob <= 1.0:
        raise ValueError(
            f"reinit_prob must be in [0, 1], got {reinit_prob}"
        )

    n_buffer = buffer_samples.shape[0]
    dim = buffer_samples.shape[1]

    indices = np.random.randint(0, n_buffer, n_samples)
    samples = buffer_samples[indices].copy()

    reinit_mask = np.random.rand(n_samples) < reinit_prob
    n_reinit = reinit_mask.sum()

    if n_reinit > 0:
        if noise_type == "uniform":
            samples[reinit_mask] = np.random.uniform(low, high, (n_reinit, dim))
        elif noise_type == "gaussian":
            samples[reinit_mask] = np.random.randn(n_reinit, dim) * std
        else:
            raise ValueError(
                f"Unknown noise_type: {noise_type}. Use 'uniform' or 'gaussian'."
            )

    return samples, indices


def init_persistent_chains(
    energy_fn: Callable,
    n_chains: int,
    dim: int,
    n_warmup_steps: int = 100,
    step_size: float = 0.01,
    noise_scale: float = 1.0,
    grad_clip: float = 0.03,
    noise_type: str = "uniform",
    low: float = -1.0,
    high: float = 1.0,
) -> np.ndarray:
    """
    Initialize persistent chains by running warmup Langevin steps.

    Creates chains starting from noise and runs them for warmup steps to
    get them closer to the model's current distribution. Useful for
    initializing a replay buffer at the start of training.

    Args:
        energy_fn: Energy function E(x)
        n_chains: Number of chains to initialize
        dim: Dimensionality of samples
        n_warmup_steps: Number of Langevin warmup steps (default: 100)
        step_size: Langevin step size (default: 0.01)
        noise_scale: Langevin noise scale (default: 1.0)
        grad_clip: Gradient clipping threshold (default: 0.03)
        noise_type: Initial noise type - 'uniform' or 'gaussian'
        low: Lower bound for uniform initialization (default: -1.0)
        high: Upper bound for uniform initialization (default: 1.0)

    Returns:
        Array of shape (n_chains, dim) with warmed-up chain states

    Example:
        >>> from ebm.core.energy import EnergyMLP
        >>> energy_fn = EnergyMLP(input_dim=2, hidden_dims=[64, 64])
        >>> chains = init_persistent_chains(energy_fn, n_chains=1000, dim=2)
        >>> chains.shape  # (1000, 2)
    """
    from ebm.sampling.langevin import langevin_sample

    x_init = init_from_noise(n_chains, dim, noise_type=noise_type, low=low, high=high)

    chains = langevin_sample(
        energy_fn=energy_fn,
        x_init=x_init,
        n_steps=n_warmup_steps,
        step_size=step_size,
        noise_scale=noise_scale,
        grad_clip=grad_clip,
    )

    return chains


@dataclass
class LangevinConfig:
    """
    Configuration for Langevin dynamics sampling.

    Attributes:
        n_steps: Number of Langevin iterations (20-100 for training, more for final samples)
        step_size: Epsilon in the update rule (start with 0.01, tune carefully)
        noise_scale: Multiplier on noise (1.0 for standard T=1 sampling)
        grad_clip: Maximum gradient norm per sample (0.03 is a reasonable default)
        anneal_step_size: Whether to anneal step size during sampling
        step_size_end: Final step size if annealing (defaults to step_size / 10)
        anneal_type: Annealing schedule type ('linear' or 'geometric')
        return_trajectory: Whether to return all intermediate samples
    """
    n_steps: int = 40
    step_size: float = 0.01
    noise_scale: float = 1.0
    grad_clip: float = 0.03
    anneal_step_size: bool = False
    step_size_end: Optional[float] = None
    anneal_type: str = "linear"
    return_trajectory: bool = False


@dataclass
class LangevinDiagnostics:
    """
    Diagnostics from Langevin sampling for monitoring convergence.

    Attributes:
        energies: Mean energy at each step
        grad_norms: Mean gradient norm at each step
        step_sizes: Step size at each step (if annealing)
        acceptance_rate: Not used in basic Langevin (always 1.0)

    What to look for:
        - Energy should decrease initially, then fluctuate around equilibrium
        - Gradient norms should stabilize (not explode or vanish)
        - If energy keeps decreasing: not enough steps or step size too large
        - If energy explodes: step size too large or clipping threshold too high
    """
    energies: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    step_sizes: List[float] = field(default_factory=list)
    sample_norms: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert diagnostics to dictionary."""
        return {
            "energies": self.energies,
            "grad_norms": self.grad_norms,
            "step_sizes": self.step_sizes,
            "sample_norms": self.sample_norms,
        }


def langevin_sample(
    energy_fn: Callable[[Tensor], Tensor],
    x_init: np.ndarray,
    n_steps: int,
    step_size: float,
    noise_scale: float = 1.0,
    grad_clip: float = 0.03,
    return_trajectory: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Sample from p(x) proportional to exp(-E(x)) using Langevin dynamics.

    The Langevin update rule:
        x_{t+1} = x_t - epsilon * grad_E(x_t) + sqrt(2 * epsilon) * noise_scale * eta

    where eta ~ N(0, I). The factor sqrt(2 * epsilon) is mathematically required
    for the stationary distribution to be exactly p(x) = exp(-E(x))/Z at T=1.

    Args:
        energy_fn: Energy function E(x) that takes a Tensor and returns energy values.
                  Should support computing gradients w.r.t. input.
        x_init: Initial samples, shape (batch_size, dim)
        n_steps: Number of Langevin iterations
        step_size: Epsilon in the update rule
        noise_scale: Multiplier on noise (1.0 = standard T=1 sampling).
                    For temperature T, use noise_scale = sqrt(T).
        grad_clip: Maximum gradient norm per sample for stability
        return_trajectory: If True, also return all intermediate samples

    Returns:
        If return_trajectory is False:
            Final samples after n_steps iterations, shape (batch_size, dim)
        If return_trajectory is True:
            Tuple of (final_samples, trajectory) where trajectory has shape
            (n_steps + 1, batch_size, dim)

    Example:
        >>> # Quadratic energy E(x) = 0.5 * ||x||^2 corresponds to N(0, I)
        >>> def quadratic_energy(x):
        ...     return (x * x).sum(axis=-1, keepdims=True) * 0.5
        >>> x_init = np.random.randn(100, 2) * 3
        >>> samples = langevin_sample(quadratic_energy, x_init, n_steps=100, step_size=0.1)
        >>> # samples should approximate N(0, I)

    Note:
        The noise scale sqrt(2 * epsilon) (not sqrt(epsilon)) is derived from the
        fluctuation-dissipation theorem. Using sqrt(epsilon) instead effectively
        samples at a different temperature.
    """
    x = x_init.copy()

    trajectory = None
    if return_trajectory:
        trajectory = np.zeros((n_steps + 1,) + x.shape)
        trajectory[0] = x.copy()

    noise_coeff = np.sqrt(2 * step_size) * noise_scale

    for t in range(n_steps):
        x_tensor = Tensor(x, requires_grad=True)
        energy = energy_fn(x_tensor)

        energy_sum = tensor_sum(energy)
        energy_sum.backward()

        grad = x_tensor.grad
        if grad is None:
            grad = np.zeros_like(x)

        grad = clip_grad(grad, max_norm=grad_clip)

        noise = np.random.randn(*x.shape)
        x = x - step_size * grad + noise_coeff * noise

        if return_trajectory:
            trajectory[t + 1] = x.copy()

    if return_trajectory:
        return x, trajectory
    return x


def langevin_sample_with_diagnostics(
    energy_fn: Callable[[Tensor], Tensor],
    x_init: np.ndarray,
    n_steps: int,
    step_size: float,
    noise_scale: float = 1.0,
    grad_clip: float = 0.03,
    anneal_step_size: bool = False,
    step_size_end: Optional[float] = None,
    anneal_type: str = "linear",
) -> Tuple[np.ndarray, LangevinDiagnostics]:
    """
    Sample from p(x) with convergence diagnostics.

    Same as langevin_sample but also tracks diagnostic statistics for
    monitoring convergence and debugging.

    Args:
        energy_fn: Energy function E(x)
        x_init: Initial samples, shape (batch_size, dim)
        n_steps: Number of Langevin iterations
        step_size: Epsilon in the update rule (or starting epsilon if annealing)
        noise_scale: Multiplier on noise (1.0 = standard T=1 sampling)
        grad_clip: Maximum gradient norm per sample
        anneal_step_size: Whether to anneal step size during sampling
        step_size_end: Final step size if annealing (defaults to step_size / 10)
        anneal_type: Annealing schedule type ('linear' or 'geometric')

    Returns:
        Tuple of (samples, diagnostics) where:
            - samples: Final samples after n_steps iterations
            - diagnostics: LangevinDiagnostics with tracked statistics

    Diagnostics interpretation:
        - Energy should decrease initially, then fluctuate around equilibrium
        - Gradient norms should stabilize (not explode or vanish)
        - If energy keeps decreasing: not enough steps or step size too large
        - If energy explodes: step size too large or clipping threshold too high
    """
    x = x_init.copy()

    diagnostics = LangevinDiagnostics()

    if anneal_step_size:
        if step_size_end is None:
            step_size_end = step_size / 10.0
        if anneal_type == "linear":
            step_sizes = linear_annealing(step_size, step_size_end, n_steps)
        elif anneal_type == "geometric":
            step_sizes = geometric_annealing(step_size, step_size_end, n_steps)
        else:
            raise ValueError(f"Unknown anneal_type: {anneal_type}. Use 'linear' or 'geometric'.")
    else:
        step_sizes = np.full(n_steps, step_size)

    for t in range(n_steps):
        current_step_size = step_sizes[t]

        x_tensor = Tensor(x, requires_grad=True)
        energy = energy_fn(x_tensor)

        diagnostics.energies.append(float(energy.data.mean()))

        energy_sum = tensor_sum(energy)
        energy_sum.backward()

        grad = x_tensor.grad
        if grad is None:
            grad = np.zeros_like(x)

        grad_norm_before_clip = np.linalg.norm(grad, axis=-1).mean()
        diagnostics.grad_norms.append(float(grad_norm_before_clip))

        diagnostics.step_sizes.append(float(current_step_size))

        diagnostics.sample_norms.append(float(np.linalg.norm(x, axis=-1).mean()))

        grad = clip_grad(grad, max_norm=grad_clip)

        noise_coeff = np.sqrt(2 * current_step_size) * noise_scale
        noise = np.random.randn(*x.shape)
        x = x - current_step_size * grad + noise_coeff * noise

    return x, diagnostics


def langevin_sample_with_config(
    energy_fn: Callable[[Tensor], Tensor],
    x_init: np.ndarray,
    config: LangevinConfig,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, LangevinDiagnostics]]:
    """
    Sample using a LangevinConfig object.

    Convenience wrapper that uses config parameters.

    Args:
        energy_fn: Energy function E(x)
        x_init: Initial samples
        config: LangevinConfig with sampling parameters

    Returns:
        Samples, optionally with trajectory or diagnostics based on config
    """
    if config.return_trajectory:
        return langevin_sample(
            energy_fn=energy_fn,
            x_init=x_init,
            n_steps=config.n_steps,
            step_size=config.step_size,
            noise_scale=config.noise_scale,
            grad_clip=config.grad_clip,
            return_trajectory=True,
        )

    if config.anneal_step_size:
        return langevin_sample_with_diagnostics(
            energy_fn=energy_fn,
            x_init=x_init,
            n_steps=config.n_steps,
            step_size=config.step_size,
            noise_scale=config.noise_scale,
            grad_clip=config.grad_clip,
            anneal_step_size=True,
            step_size_end=config.step_size_end,
            anneal_type=config.anneal_type,
        )

    return langevin_sample(
        energy_fn=energy_fn,
        x_init=x_init,
        n_steps=config.n_steps,
        step_size=config.step_size,
        noise_scale=config.noise_scale,
        grad_clip=config.grad_clip,
        return_trajectory=False,
    )
