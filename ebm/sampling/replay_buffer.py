"""
Replay Buffer for Energy-Based Model training.

The replay buffer stores samples from previous iterations so new Langevin chains
can start near modes, dramatically improving sampling efficiency. Without a buffer,
chains must traverse from pure noise to data modes every iteration, which is slow
and often fails to reach all modes.

Key benefits:
- Faster convergence: Chains start near modes instead of from noise
- Better mode coverage: Buffer maintains diversity across training
- Exploration-exploitation balance: reinit_prob controls fresh exploration

Usage:
    buffer = ReplayBuffer(capacity=10000, sample_dim=2)

    # Training loop
    for batch in data_loader:
        # Sample from buffer (95% buffer, 5% fresh noise)
        x_init, indices = buffer.sample(batch_size, reinit_prob=0.05)

        # Run Langevin to refine samples
        x_fake = langevin_sample(energy_fn, x_init, n_steps=40, ...)

        # Update buffer with refined samples
        buffer.update(indices, x_fake)

        # ... compute loss and update energy function ...
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union


@dataclass
class ReplayBufferConfig:
    """
    Configuration for ReplayBuffer.

    Attributes:
        capacity: Maximum number of samples to store in the buffer
        sample_dim: Dimensionality of each sample
        reinit_prob: Default probability of reinitializing from noise (0.05 = 5%)
        init_type: Type of initialization for buffer - 'uniform' or 'gaussian'
        init_low: Lower bound for uniform initialization (default: -1.0)
        init_high: Upper bound for uniform initialization (default: 1.0)
        init_std: Standard deviation for Gaussian initialization (default: 1.0)
    """
    capacity: int = 10000
    sample_dim: int = 2
    reinit_prob: float = 0.05
    init_type: str = "uniform"
    init_low: float = -1.0
    init_high: float = 1.0
    init_std: float = 1.0


@dataclass
class ReplayBufferStats:
    """
    Statistics from replay buffer operations.

    Useful for monitoring buffer behavior during training.

    Attributes:
        n_samples: Number of samples drawn
        n_reinit: Number of samples reinitialized from noise
        reinit_ratio: Actual ratio of reinitialized samples
        sample_mean_norm: Mean norm of sampled data
        buffer_mean_norm: Mean norm of all buffer data
    """
    n_samples: int = 0
    n_reinit: int = 0
    reinit_ratio: float = 0.0
    sample_mean_norm: float = 0.0
    buffer_mean_norm: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert stats to dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_reinit": self.n_reinit,
            "reinit_ratio": self.reinit_ratio,
            "sample_mean_norm": self.sample_mean_norm,
            "buffer_mean_norm": self.buffer_mean_norm,
        }


class ReplayBuffer:
    """
    Replay buffer for storing and retrieving samples during EBM training.

    The replay buffer is essential for efficient EBM training. Langevin chains
    take many steps to reach modes from pure noise. By storing samples from
    previous iterations, new chains can start near modes, dramatically
    improving sampling efficiency.

    Attributes:
        capacity: Maximum number of samples to store
        sample_dim: Dimensionality of each sample
        buffer: NumPy array of shape (capacity, sample_dim) storing samples
        position: Current position for FIFO updates (optional, for sequential updates)

    Example:
        >>> buffer = ReplayBuffer(capacity=10000, sample_dim=2)
        >>>
        >>> # Sample from buffer with 5% reinitialization
        >>> samples, indices = buffer.sample(64, reinit_prob=0.05)
        >>>
        >>> # Run Langevin dynamics on samples
        >>> refined_samples = langevin_sample(energy_fn, samples, n_steps=40, ...)
        >>>
        >>> # Update buffer with refined samples
        >>> buffer.update(indices, refined_samples)

    Note:
        - Buffer is initialized with random samples (uniform or Gaussian)
        - reinit_prob should be small (0.05) for most cases to balance
          exploitation (using good samples) with exploration (fresh noise)
        - The update method replaces samples at the given indices with new samples
    """

    def __init__(
        self,
        capacity: int,
        sample_dim: int,
        init_type: str = "uniform",
        init_low: float = -1.0,
        init_high: float = 1.0,
        init_std: float = 1.0,
    ):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of samples to store
            sample_dim: Dimensionality of each sample
            init_type: Type of initialization - 'uniform' or 'gaussian'
            init_low: Lower bound for uniform initialization (default: -1.0)
            init_high: Upper bound for uniform initialization (default: 1.0)
            init_std: Standard deviation for Gaussian initialization (default: 1.0)

        Raises:
            ValueError: If capacity <= 0 or sample_dim <= 0
            ValueError: If init_type is not 'uniform' or 'gaussian'
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if sample_dim <= 0:
            raise ValueError(f"sample_dim must be positive, got {sample_dim}")
        if init_type not in ("uniform", "gaussian"):
            raise ValueError(
                f"init_type must be 'uniform' or 'gaussian', got '{init_type}'"
            )

        self.capacity = capacity
        self.sample_dim = sample_dim
        self.init_type = init_type
        self.init_low = init_low
        self.init_high = init_high
        self.init_std = init_std

        if init_type == "uniform":
            self.buffer = np.random.uniform(
                init_low, init_high, (capacity, sample_dim)
            )
        else:
            self.buffer = np.random.randn(capacity, sample_dim) * init_std

        self.position = 0

        self._total_samples_drawn = 0
        self._total_reinit = 0

    def sample(
        self,
        n: int,
        reinit_prob: float = 0.05,
        return_stats: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray, ReplayBufferStats]]:
        """
        Sample from buffer with occasional reinitialization.

        Randomly selects n samples from the buffer. With probability reinit_prob,
        each sample is replaced with fresh noise instead of the buffer value.
        This ensures continued exploration while mostly exploiting good samples.

        Args:
            n: Number of samples to draw
            reinit_prob: Probability of drawing fresh noise instead of buffer value
                        (default: 0.05 = 5% fresh noise, 95% from buffer)
            return_stats: If True, also return ReplayBufferStats

        Returns:
            If return_stats is False:
                Tuple of (samples, indices) where:
                    - samples: Array of shape (n, sample_dim)
                    - indices: Array of buffer indices that were sampled (for updating)
            If return_stats is True:
                Tuple of (samples, indices, stats)

        Raises:
            ValueError: If n <= 0
            ValueError: If reinit_prob is not in [0, 1]

        Example:
            >>> buffer = ReplayBuffer(capacity=1000, sample_dim=2)
            >>>
            >>> # Draw 64 samples, ~5% will be fresh noise
            >>> samples, indices = buffer.sample(64, reinit_prob=0.05)
            >>>
            >>> # After Langevin refinement, update buffer
            >>> buffer.update(indices, refined_samples)

        Note:
            The returned indices can be used to update the buffer after Langevin
            refinement. Samples that were reinitialized from noise will still
            have valid indices pointing to random buffer locations - these will
            be overwritten during update, which is desired behavior.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if not 0.0 <= reinit_prob <= 1.0:
            raise ValueError(f"reinit_prob must be in [0, 1], got {reinit_prob}")

        indices = np.random.randint(0, self.capacity, n)
        samples = self.buffer[indices].copy()

        reinit_mask = np.random.rand(n) < reinit_prob
        n_reinit = reinit_mask.sum()

        if n_reinit > 0:
            if self.init_type == "uniform":
                samples[reinit_mask] = np.random.uniform(
                    self.init_low, self.init_high, (n_reinit, self.sample_dim)
                )
            else:
                samples[reinit_mask] = (
                    np.random.randn(n_reinit, self.sample_dim) * self.init_std
                )

        self._total_samples_drawn += n
        self._total_reinit += n_reinit

        if return_stats:
            stats = ReplayBufferStats(
                n_samples=n,
                n_reinit=n_reinit,
                reinit_ratio=n_reinit / n if n > 0 else 0.0,
                sample_mean_norm=np.linalg.norm(samples, axis=-1).mean(),
                buffer_mean_norm=np.linalg.norm(self.buffer, axis=-1).mean(),
            )
            return samples, indices, stats

        return samples, indices

    def update(
        self,
        indices: np.ndarray,
        new_samples: np.ndarray,
    ) -> None:
        """
        Update buffer with new samples after Langevin refinement.

        Replaces samples at the given indices with new samples. This is typically
        called after running Langevin dynamics on samples drawn from the buffer.

        Args:
            indices: Array of buffer indices to update
            new_samples: Array of new samples, shape (len(indices), sample_dim)

        Raises:
            ValueError: If indices and new_samples have incompatible shapes
            ValueError: If new_samples has wrong dimensionality
            IndexError: If any index is out of bounds

        Example:
            >>> samples, indices = buffer.sample(64)
            >>> refined_samples = langevin_sample(energy_fn, samples, ...)
            >>> buffer.update(indices, refined_samples)
        """
        indices = np.asarray(indices)
        new_samples = np.asarray(new_samples)

        if len(indices) != len(new_samples):
            raise ValueError(
                f"indices and new_samples must have same length, "
                f"got {len(indices)} and {len(new_samples)}"
            )

        if new_samples.ndim != 2:
            raise ValueError(
                f"new_samples must be 2D, got {new_samples.ndim}D"
            )

        if new_samples.shape[1] != self.sample_dim:
            raise ValueError(
                f"new_samples must have {self.sample_dim} features, "
                f"got {new_samples.shape[1]}"
            )

        if len(indices) > 0:
            if indices.min() < 0 or indices.max() >= self.capacity:
                raise IndexError(
                    f"Index out of bounds: indices must be in [0, {self.capacity}), "
                    f"got min={indices.min()}, max={indices.max()}"
                )

        self.buffer[indices] = new_samples

    def push(self, samples: np.ndarray) -> np.ndarray:
        """
        Push new samples to buffer using FIFO policy.

        Adds samples starting at the current position, wrapping around if needed.
        This is an alternative to update() when you want to add samples without
        tracking specific indices.

        Args:
            samples: Array of shape (n_samples, sample_dim) to add

        Returns:
            Array of indices where samples were placed

        Raises:
            ValueError: If samples has wrong dimensionality

        Example:
            >>> new_samples = np.random.randn(100, 2)
            >>> indices = buffer.push(new_samples)
        """
        samples = np.asarray(samples)

        if samples.ndim != 2:
            raise ValueError(f"samples must be 2D, got {samples.ndim}D")

        if samples.shape[1] != self.sample_dim:
            raise ValueError(
                f"samples must have {self.sample_dim} features, "
                f"got {samples.shape[1]}"
            )

        n = len(samples)
        indices = np.zeros(n, dtype=np.int64)

        for i, sample in enumerate(samples):
            indices[i] = self.position
            self.buffer[self.position] = sample
            self.position = (self.position + 1) % self.capacity

        return indices

    def get_all(self) -> np.ndarray:
        """
        Get all samples in the buffer.

        Returns:
            Copy of the entire buffer, shape (capacity, sample_dim)
        """
        return self.buffer.copy()

    def set_all(self, data: np.ndarray) -> None:
        """
        Replace entire buffer with new data.

        Args:
            data: Array of shape (capacity, sample_dim)

        Raises:
            ValueError: If data shape doesn't match buffer shape
        """
        data = np.asarray(data)

        if data.shape != (self.capacity, self.sample_dim):
            raise ValueError(
                f"data must have shape ({self.capacity}, {self.sample_dim}), "
                f"got {data.shape}"
            )

        self.buffer = data.copy()
        self.position = 0

    def reset(self) -> None:
        """
        Reset buffer to fresh random samples.

        Reinitializes all buffer entries with random noise according to
        init_type (uniform or gaussian).
        """
        if self.init_type == "uniform":
            self.buffer = np.random.uniform(
                self.init_low, self.init_high, (self.capacity, self.sample_dim)
            )
        else:
            self.buffer = np.random.randn(self.capacity, self.sample_dim) * self.init_std

        self.position = 0
        self._total_samples_drawn = 0
        self._total_reinit = 0

    def get_statistics(self) -> Dict[str, float]:
        """
        Get buffer statistics.

        Returns:
            Dictionary with buffer statistics:
                - capacity: Buffer capacity
                - sample_dim: Sample dimensionality
                - mean_norm: Mean sample norm in buffer
                - std_norm: Std of sample norms in buffer
                - min_norm: Minimum sample norm
                - max_norm: Maximum sample norm
                - total_samples_drawn: Total samples drawn since creation/reset
                - total_reinit: Total samples reinitialized since creation/reset
                - overall_reinit_ratio: Overall reinitialization ratio
        """
        norms = np.linalg.norm(self.buffer, axis=-1)

        overall_reinit_ratio = (
            self._total_reinit / self._total_samples_drawn
            if self._total_samples_drawn > 0 else 0.0
        )

        return {
            "capacity": self.capacity,
            "sample_dim": self.sample_dim,
            "mean_norm": float(norms.mean()),
            "std_norm": float(norms.std()),
            "min_norm": float(norms.min()),
            "max_norm": float(norms.max()),
            "mean": float(self.buffer.mean()),
            "std": float(self.buffer.std()),
            "total_samples_drawn": self._total_samples_drawn,
            "total_reinit": self._total_reinit,
            "overall_reinit_ratio": overall_reinit_ratio,
        }

    def __len__(self) -> int:
        """Return buffer capacity."""
        return self.capacity

    def __repr__(self) -> str:
        """String representation of buffer."""
        return (
            f"ReplayBuffer(capacity={self.capacity}, sample_dim={self.sample_dim}, "
            f"init_type='{self.init_type}')"
        )

    @classmethod
    def from_config(cls, config: ReplayBufferConfig) -> "ReplayBuffer":
        """
        Create a ReplayBuffer from a configuration object.

        Args:
            config: ReplayBufferConfig with buffer parameters

        Returns:
            New ReplayBuffer instance

        Example:
            >>> config = ReplayBufferConfig(capacity=10000, sample_dim=29)
            >>> buffer = ReplayBuffer.from_config(config)
        """
        return cls(
            capacity=config.capacity,
            sample_dim=config.sample_dim,
            init_type=config.init_type,
            init_low=config.init_low,
            init_high=config.init_high,
            init_std=config.init_std,
        )

    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        capacity: Optional[int] = None,
        noise_std: float = 0.0,
    ) -> "ReplayBuffer":
        """
        Create a ReplayBuffer initialized from data samples.

        Useful for initializing the buffer near training data distribution
        instead of from pure noise.

        Args:
            data: Array of shape (n_samples, sample_dim)
            capacity: Buffer capacity (defaults to len(data))
            noise_std: Optional noise to add to data samples

        Returns:
            New ReplayBuffer instance initialized with data

        Example:
            >>> train_data = np.random.randn(5000, 2)
            >>> buffer = ReplayBuffer.from_data(train_data, capacity=10000)
        """
        data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError(f"data must be 2D, got {data.ndim}D")

        n_data, sample_dim = data.shape

        if capacity is None:
            capacity = n_data

        buffer = cls(capacity=capacity, sample_dim=sample_dim, init_type="uniform")

        if n_data >= capacity:
            indices = np.random.choice(n_data, capacity, replace=False)
            buffer.buffer = data[indices].copy()
        else:
            indices = np.random.choice(n_data, capacity, replace=True)
            buffer.buffer = data[indices].copy()

        if noise_std > 0:
            buffer.buffer += np.random.randn(capacity, sample_dim) * noise_std

        return buffer


def create_replay_buffer(
    capacity: int = 10000,
    sample_dim: int = 2,
    init_type: str = "uniform",
    init_low: float = -1.0,
    init_high: float = 1.0,
    init_std: float = 1.0,
) -> ReplayBuffer:
    """
    Create a ReplayBuffer with specified parameters.

    Convenience function for creating replay buffers.

    Args:
        capacity: Maximum number of samples (default: 10000)
        sample_dim: Dimensionality of samples (default: 2)
        init_type: Initialization type - 'uniform' or 'gaussian'
        init_low: Lower bound for uniform init (default: -1.0)
        init_high: Upper bound for uniform init (default: 1.0)
        init_std: Standard deviation for Gaussian init (default: 1.0)

    Returns:
        New ReplayBuffer instance

    Example:
        >>> buffer = create_replay_buffer(capacity=10000, sample_dim=29)
    """
    return ReplayBuffer(
        capacity=capacity,
        sample_dim=sample_dim,
        init_type=init_type,
        init_low=init_low,
        init_high=init_high,
        init_std=init_std,
    )
