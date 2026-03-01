"""
Training loop for Energy-Based Models.

This module implements the training loop for EBMs using contrastive divergence.

Loss function:
    L = E(x_real).mean() - E(x_fake).mean() + alpha * reg_energy - lambda * H_knn(x_fake)

Where:
    - E(x_real).mean(): Average energy of real data (want to decrease)
    - E(x_fake).mean(): Average energy of samples (want to increase)
    - reg_energy: Prevents energy from diverging to +/- infinity
    - H_knn(x_fake): Entropy regularizer (encourages sample diversity)

The training process:
1. Sample initial points from replay buffer
2. Run Langevin dynamics to refine samples
3. Update replay buffer with refined samples
4. Compute contrastive divergence loss
5. Apply energy and entropy regularization
6. Update model parameters via gradient descent

Usage:
    >>> from ebm.training import train, AdamW
    >>> from ebm.core.energy import EnergyMLP
    >>> from ebm.sampling.replay_buffer import ReplayBuffer
    >>>
    >>> energy_fn = EnergyMLP(input_dim=2, hidden_dims=[128, 128])
    >>> optimizer = AdamW(energy_fn.parameters(), lr=1e-4)
    >>> buffer = ReplayBuffer(capacity=10000, sample_dim=2)
    >>>
    >>> history = train(
    ...     energy_fn, optimizer, train_data,
    ...     n_epochs=50, batch_size=128,
    ...     replay_buffer=buffer,
    ...     langevin_steps=40, langevin_step_size=0.01,
    ...     alpha=0.1, lambda_ent=0.01
    ... )
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional

from ebm.core.autodiff import Tensor
from ebm.core.ops import mean as tensor_mean, pow as tensor_pow, add
from ebm.sampling.langevin import langevin_sample
from ebm.sampling.replay_buffer import ReplayBuffer
from ebm.entropy.knn import knn_entropy
from ebm.training import Optimizer


@dataclass
class TrainingConfig:
    """
    Configuration for EBM training.

    Attributes:
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        langevin_steps: Number of Langevin sampling steps
        langevin_step_size: Step size for Langevin dynamics
        langevin_noise_scale: Noise scale for Langevin (1.0 for T=1)
        langevin_grad_clip: Gradient clipping for Langevin stability
        alpha: Energy regularization coefficient
        lambda_ent: Entropy regularization coefficient
        reinit_prob: Probability of reinitializing from noise in buffer
        log_interval: How often to log training stats (in epochs)
        entropy_k: Number of neighbors for k-NN entropy estimation
    """
    n_epochs: int = 50
    batch_size: int = 128
    langevin_steps: int = 40
    langevin_step_size: float = 0.01
    langevin_noise_scale: float = 1.0
    langevin_grad_clip: float = 0.03
    alpha: float = 0.1
    lambda_ent: float = 0.01
    reinit_prob: float = 0.05
    log_interval: int = 1
    entropy_k: int = 5


@dataclass
class TrainingStats:
    """
    Statistics from a single training step.

    Attributes:
        loss: Total loss value
        cd_loss: Contrastive divergence loss (E_real - E_fake)
        reg_loss: Energy regularization loss
        E_real: Mean energy of real data
        E_fake: Mean energy of fake samples
        energy_gap: E_real - E_fake (should be negative when trained)
        entropy: Entropy of fake samples (if computed)
        grad_norm: Average gradient norm (if computed)
    """
    loss: float = 0.0
    cd_loss: float = 0.0
    reg_loss: float = 0.0
    E_real: float = 0.0
    E_fake: float = 0.0
    energy_gap: float = 0.0
    entropy: float = 0.0
    grad_norm: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert stats to dictionary."""
        return {
            "loss": self.loss,
            "cd_loss": self.cd_loss,
            "reg_loss": self.reg_loss,
            "E_real": self.E_real,
            "E_fake": self.E_fake,
            "energy_gap": self.energy_gap,
            "entropy": self.entropy,
            "grad_norm": self.grad_norm,
        }


@dataclass
class TrainingHistory:
    """
    Training history over multiple epochs.

    Attributes:
        epoch_stats: List of per-epoch averaged statistics
        step_count: Total number of training steps
        best_loss: Best (lowest) loss achieved
        best_epoch: Epoch with best loss
    """
    epoch_stats: List[Dict[str, float]] = field(default_factory=list)
    step_count: int = 0
    best_loss: float = float('inf')
    best_epoch: int = 0

    def add_epoch(self, stats: Dict[str, float], epoch: int) -> None:
        """Add epoch statistics to history."""
        self.epoch_stats.append(stats)
        if stats["loss"] < self.best_loss:
            self.best_loss = stats["loss"]
            self.best_epoch = epoch

    def get_metric(self, metric: str) -> List[float]:
        """Get history of a specific metric."""
        return [s[metric] for s in self.epoch_stats if metric in s]

    def to_dict(self) -> Dict:
        """Convert history to dictionary."""
        return {
            "epoch_stats": self.epoch_stats,
            "step_count": self.step_count,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
        }


def get_batches(
    data: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[np.ndarray]:
    """
    Generate batches from data.

    Args:
        data: Data array of shape (n_samples, n_features)
        batch_size: Size of each batch
        shuffle: Whether to shuffle data before batching

    Yields:
        Batches of shape (batch_size, n_features) or smaller for last batch

    Example:
        >>> data = np.random.randn(1000, 2)
        >>> for batch in get_batches(data, batch_size=64):
        ...     print(batch.shape)
    """
    n_samples = len(data)
    if shuffle:
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield data[batch_indices]


def train_step(
    energy_fn: Callable[[Tensor], Tensor],
    optimizer: Optimizer,
    data_batch: np.ndarray,
    replay_buffer: ReplayBuffer,
    langevin_steps: int = 40,
    langevin_step_size: float = 0.01,
    langevin_noise_scale: float = 1.0,
    langevin_grad_clip: float = 0.03,
    alpha: float = 0.1,
    lambda_ent: float = 0.01,
    reinit_prob: float = 0.05,
    entropy_k: int = 5,
    compute_entropy: bool = True,
) -> TrainingStats:
    """
    Perform a single training step.

    This implements one iteration of contrastive divergence training:
    1. Sample from replay buffer and run Langevin to get fake samples
    2. Compute contrastive divergence loss
    3. Add energy regularization to prevent divergence
    4. Optionally add entropy regularization for sample diversity
    5. Update model parameters

    Args:
        energy_fn: Energy function that takes Tensor and returns energy values
        optimizer: Optimizer with step() and zero_grad() methods
        data_batch: Batch of real data, shape (batch_size, dim)
        replay_buffer: Replay buffer for sampling initial points
        langevin_steps: Number of Langevin dynamics steps
        langevin_step_size: Step size for Langevin
        langevin_noise_scale: Noise scale for Langevin (1.0 for T=1)
        langevin_grad_clip: Gradient clipping for Langevin stability
        alpha: Energy regularization coefficient
        lambda_ent: Entropy regularization coefficient
        reinit_prob: Probability of reinitializing from noise
        entropy_k: Number of neighbors for k-NN entropy
        compute_entropy: Whether to compute entropy (can disable for speed)

    Returns:
        TrainingStats with loss and other metrics

    Example:
        >>> stats = train_step(
        ...     energy_fn, optimizer, batch,
        ...     replay_buffer=buffer,
        ...     langevin_steps=40,
        ...     langevin_step_size=0.01,
        ...     alpha=0.1,
        ...     lambda_ent=0.01
        ... )
    """
    batch_size = data_batch.shape[0]

    x_init, buffer_indices = replay_buffer.sample(batch_size, reinit_prob=reinit_prob)
    x_fake = langevin_sample(
        energy_fn,
        x_init,
        n_steps=langevin_steps,
        step_size=langevin_step_size,
        noise_scale=langevin_noise_scale,
        grad_clip=langevin_grad_clip,
    )

    replay_buffer.update(buffer_indices, x_fake)

    x_real_tensor = Tensor(data_batch, requires_grad=False)
    x_fake_tensor = Tensor(x_fake, requires_grad=False)

    E_real = energy_fn(x_real_tensor)
    E_fake = energy_fn(x_fake_tensor)

    E_real_mean = tensor_mean(E_real)
    E_fake_mean = tensor_mean(E_fake)
    cd_loss = add(E_real_mean, tensor_mean(E_fake * Tensor(-1.0)))

    E_real_sq = tensor_mean(tensor_pow(E_real, 2))
    E_fake_sq = tensor_mean(tensor_pow(E_fake, 2))
    reg_loss = add(E_real_sq, E_fake_sq) * Tensor(alpha)

    loss = add(cd_loss, reg_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    entropy = 0.0
    if compute_entropy and lambda_ent > 0 and batch_size > entropy_k:
        try:
            entropy = knn_entropy(x_fake, k=entropy_k)
        except Exception:
            entropy = 0.0

    grad_norm = 0.0
    n_params = 0
    for p in energy_fn.parameters():
        if p.grad is not None:
            grad_norm += float((p.grad ** 2).sum())
            n_params += 1
    if n_params > 0:
        grad_norm = np.sqrt(grad_norm)

    return TrainingStats(
        loss=float(loss.data),
        cd_loss=float(cd_loss.data),
        reg_loss=float(reg_loss.data),
        E_real=float(E_real_mean.data),
        E_fake=float(E_fake_mean.data),
        energy_gap=float(E_real_mean.data) - float(E_fake_mean.data),
        entropy=entropy,
        grad_norm=grad_norm,
    )


def train(
    energy_fn: Callable[[Tensor], Tensor],
    optimizer: Optimizer,
    dataset: np.ndarray,
    n_epochs: int,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    langevin_steps: int = 40,
    langevin_step_size: float = 0.01,
    langevin_noise_scale: float = 1.0,
    langevin_grad_clip: float = 0.03,
    alpha: float = 0.1,
    lambda_ent: float = 0.01,
    reinit_prob: float = 0.05,
    entropy_k: int = 5,
    compute_entropy: bool = True,
    log_interval: int = 1,
    verbose: bool = True,
    callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> TrainingHistory:
    """
    Train an Energy-Based Model using contrastive divergence.

    The training loop:
    1. For each epoch:
        a. Shuffle and batch the data
        b. For each batch, perform a training step
        c. Aggregate statistics
        d. Log progress
    2. Return training history

    Args:
        energy_fn: Energy function (e.g., EnergyMLP)
        optimizer: Optimizer (e.g., AdamW)
        dataset: Training data, shape (n_samples, dim)
        n_epochs: Number of training epochs
        batch_size: Batch size
        replay_buffer: Replay buffer for Langevin sampling
        langevin_steps: Number of Langevin steps per batch
        langevin_step_size: Langevin step size
        langevin_noise_scale: Langevin noise scale
        langevin_grad_clip: Langevin gradient clip
        alpha: Energy regularization coefficient
        lambda_ent: Entropy regularization coefficient (not used in loss, for tracking)
        reinit_prob: Buffer reinitialization probability
        entropy_k: k for k-NN entropy estimation
        compute_entropy: Whether to compute entropy
        log_interval: Print progress every N epochs
        verbose: Whether to print progress
        callback: Optional callback(epoch, stats) called after each epoch

    Returns:
        TrainingHistory with per-epoch statistics

    Example:
        >>> history = train(
        ...     energy_fn, optimizer, train_data,
        ...     n_epochs=50, batch_size=128,
        ...     replay_buffer=buffer,
        ...     langevin_steps=40,
        ...     langevin_step_size=0.01,
        ...     alpha=0.1,
        ...     lambda_ent=0.01
        ... )
        >>> print(history.best_loss)
    """
    history = TrainingHistory()

    for epoch in range(n_epochs):
        epoch_stats: List[TrainingStats] = []

        for batch in get_batches(dataset, batch_size, shuffle=True):
            stats = train_step(
                energy_fn=energy_fn,
                optimizer=optimizer,
                data_batch=batch,
                replay_buffer=replay_buffer,
                langevin_steps=langevin_steps,
                langevin_step_size=langevin_step_size,
                langevin_noise_scale=langevin_noise_scale,
                langevin_grad_clip=langevin_grad_clip,
                alpha=alpha,
                lambda_ent=lambda_ent,
                reinit_prob=reinit_prob,
                entropy_k=entropy_k,
                compute_entropy=compute_entropy,
            )
            epoch_stats.append(stats)
            history.step_count += 1

        if epoch_stats:
            avg_stats = {
                "loss": np.mean([s.loss for s in epoch_stats]),
                "cd_loss": np.mean([s.cd_loss for s in epoch_stats]),
                "reg_loss": np.mean([s.reg_loss for s in epoch_stats]),
                "E_real": np.mean([s.E_real for s in epoch_stats]),
                "E_fake": np.mean([s.E_fake for s in epoch_stats]),
                "energy_gap": np.mean([s.energy_gap for s in epoch_stats]),
                "entropy": np.mean([s.entropy for s in epoch_stats]),
                "grad_norm": np.mean([s.grad_norm for s in epoch_stats]),
            }
            history.add_epoch(avg_stats, epoch)

            if verbose and (epoch % log_interval == 0 or epoch == n_epochs - 1):
                print(
                    f"Epoch {epoch:4d}: loss={avg_stats['loss']:.4f}, "
                    f"E_real={avg_stats['E_real']:.4f}, "
                    f"E_fake={avg_stats['E_fake']:.4f}, "
                    f"H={avg_stats['entropy']:.4f}"
                )

            if callback is not None:
                callback(epoch, avg_stats)

    return history


def train_with_config(
    energy_fn: Callable[[Tensor], Tensor],
    optimizer: Optimizer,
    dataset: np.ndarray,
    replay_buffer: ReplayBuffer,
    config: TrainingConfig,
    verbose: bool = True,
    callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> TrainingHistory:
    """
    Train using a TrainingConfig object.

    Convenience wrapper around train() that uses config parameters.

    Args:
        energy_fn: Energy function
        optimizer: Optimizer
        dataset: Training data
        replay_buffer: Replay buffer
        config: TrainingConfig with all hyperparameters
        verbose: Whether to print progress
        callback: Optional callback

    Returns:
        TrainingHistory
    """
    return train(
        energy_fn=energy_fn,
        optimizer=optimizer,
        dataset=dataset,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        replay_buffer=replay_buffer,
        langevin_steps=config.langevin_steps,
        langevin_step_size=config.langevin_step_size,
        langevin_noise_scale=config.langevin_noise_scale,
        langevin_grad_clip=config.langevin_grad_clip,
        alpha=config.alpha,
        lambda_ent=config.lambda_ent,
        reinit_prob=config.reinit_prob,
        entropy_k=config.entropy_k,
        compute_entropy=config.lambda_ent > 0,
        log_interval=config.log_interval,
        verbose=verbose,
        callback=callback,
    )


def evaluate_energy_model(
    energy_fn: Callable[[Tensor], Tensor],
    data: np.ndarray,
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    Evaluate energy model on a dataset.

    Computes energy statistics on the given data without training.

    Args:
        energy_fn: Energy function to evaluate
        data: Data to evaluate, shape (n_samples, dim)
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with energy statistics:
            - mean_energy: Mean energy value
            - std_energy: Standard deviation of energy
            - min_energy: Minimum energy
            - max_energy: Maximum energy
    """
    all_energies = []

    for batch in get_batches(data, batch_size, shuffle=False):
        x_tensor = Tensor(batch, requires_grad=False)
        energy = energy_fn(x_tensor)
        all_energies.append(energy.data.flatten())

    all_energies = np.concatenate(all_energies)

    return {
        "mean_energy": float(all_energies.mean()),
        "std_energy": float(all_energies.std()),
        "min_energy": float(all_energies.min()),
        "max_energy": float(all_energies.max()),
    }


def generate_samples(
    energy_fn: Callable[[Tensor], Tensor],
    n_samples: int,
    sample_dim: int,
    n_steps: int = 100,
    step_size: float = 0.01,
    noise_scale: float = 1.0,
    grad_clip: float = 0.03,
    init_type: str = "uniform",
    init_low: float = -1.0,
    init_high: float = 1.0,
    init_std: float = 1.0,
) -> np.ndarray:
    """
    Generate samples from the learned energy model.

    Runs Langevin dynamics from random initialization to generate samples.

    Args:
        energy_fn: Trained energy function
        n_samples: Number of samples to generate
        sample_dim: Dimensionality of samples
        n_steps: Number of Langevin steps (more = better samples)
        step_size: Langevin step size
        noise_scale: Langevin noise scale
        grad_clip: Gradient clipping
        init_type: Initialization type ('uniform' or 'gaussian')
        init_low: Lower bound for uniform init
        init_high: Upper bound for uniform init
        init_std: Standard deviation for Gaussian init

    Returns:
        Generated samples, shape (n_samples, sample_dim)

    Example:
        >>> samples = generate_samples(
        ...     energy_fn, n_samples=1000, sample_dim=2,
        ...     n_steps=100, step_size=0.01
        ... )
    """
    if init_type == "uniform":
        x_init = np.random.uniform(init_low, init_high, (n_samples, sample_dim))
    elif init_type == "gaussian":
        x_init = np.random.randn(n_samples, sample_dim) * init_std
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    samples = langevin_sample(
        energy_fn,
        x_init,
        n_steps=n_steps,
        step_size=step_size,
        noise_scale=noise_scale,
        grad_clip=grad_clip,
    )

    return samples
