"""
Optimizers for Energy-Based Model training.

This module implements:
- SGD: Stochastic Gradient Descent with momentum
- AdamW: Adam with decoupled weight decay (preferred for EBMs)

AdamW is recommended over vanilla Adam because it decouples weight decay from
the adaptive learning rate, giving more consistent regularization and typically
better generalization.

Usage:
    >>> from ebm.training.optimizer import AdamW
    >>> from ebm.core.energy import EnergyMLP
    >>>
    >>> energy_fn = EnergyMLP(input_dim=2, hidden_dims=[128, 128])
    >>> optimizer = AdamW(energy_fn.parameters(), lr=1e-4, weight_decay=0.01)
    >>>
    >>> # Training step
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterator, List, Union

from ebm.core.autodiff import Tensor


@dataclass
class OptimizerConfig:
    """
    Configuration for optimizers.

    Attributes:
        optimizer_type: Type of optimizer ('sgd' or 'adamw')
        lr: Learning rate
        momentum: Momentum coefficient for SGD (default: 0.9)
        beta1: First moment decay for Adam (default: 0.9)
        beta2: Second moment decay for Adam (default: 0.999)
        eps: Small constant for numerical stability in Adam (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
    """
    optimizer_type: str = "adamw"
    lr: float = 1e-4
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01


class Optimizer(ABC):
    """
    Abstract base class for optimizers.

    All optimizers must implement:
    - step(): Update parameters based on gradients
    - zero_grad(): Reset gradients to None
    """

    def __init__(self, parameters: Union[List[Tensor], Iterator[Tensor]]):
        """
        Initialize optimizer with parameters to optimize.

        Args:
            parameters: List or iterator of Tensors to optimize
        """
        self.parameters = list(parameters)

    @abstractmethod
    def step(self) -> None:
        """Update parameters based on current gradients."""
        pass

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to None."""
        for p in self.parameters:
            p.grad = None

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.lr

    def set_lr(self, lr: float) -> None:
        """Set learning rate."""
        self.lr = lr


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum.

    Update rule:
        v_t = momentum * v_{t-1} - lr * grad
        param = param + v_t

    This is the "heavy ball" momentum formulation.

    Attributes:
        parameters: List of parameters to optimize
        lr: Learning rate
        momentum: Momentum coefficient (default: 0.9)
        velocities: Velocity buffers for each parameter

    Example:
        >>> params = [Tensor(np.random.randn(10, 5), requires_grad=True)]
        >>> optimizer = SGD(params, lr=0.01, momentum=0.9)
        >>> # After computing loss and calling backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        parameters: Union[List[Tensor], Iterator[Tensor]],
        lr: float,
        momentum: float = 0.9,
    ):
        """
        Initialize SGD optimizer.

        Args:
            parameters: Parameters to optimize
            lr: Learning rate
            momentum: Momentum coefficient (default: 0.9, set to 0 for vanilla SGD)

        Raises:
            ValueError: If lr <= 0 or momentum < 0 or momentum >= 1
        """
        super().__init__(parameters)

        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")

        self.lr = lr
        self.momentum = momentum

        self.velocities: List[np.ndarray] = [
            np.zeros_like(p.data) for p in self.parameters
        ]

    def step(self) -> None:
        """
        Perform a single optimization step.

        Updates parameters using gradient descent with momentum:
            v = momentum * v - lr * grad
            param = param + v
        """
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            self.velocities[i] = (
                self.momentum * self.velocities[i] - self.lr * p.grad
            )

            p.data = p.data + self.velocities[i]

    def reset_state(self) -> None:
        """Reset optimizer state (velocities)."""
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]

    def get_state(self) -> Dict:
        """Get optimizer state for checkpointing."""
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "velocities": [v.copy() for v in self.velocities],
        }

    def load_state(self, state: Dict) -> None:
        """Load optimizer state from checkpoint."""
        self.lr = state["lr"]
        self.momentum = state["momentum"]
        self.velocities = [v.copy() for v in state["velocities"]]

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.momentum}, n_params={len(self.parameters)})"


class AdamW(Optimizer):
    """
    AdamW optimizer - Adam with decoupled weight decay.

    Unlike vanilla Adam where weight decay is coupled with the adaptive learning
    rate, AdamW applies weight decay directly to the weights. This provides more
    consistent regularization across parameters with different gradient magnitudes.

    Update rule:
        # Weight decay (decoupled, applied first)
        param = param - lr * weight_decay * param

        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)

    Attributes:
        parameters: List of parameters to optimize
        lr: Learning rate
        beta1: First moment decay (default: 0.9)
        beta2: Second moment decay (default: 0.999)
        eps: Numerical stability constant (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        t: Current timestep (for bias correction)

    Example:
        >>> params = energy_fn.parameters()
        >>> optimizer = AdamW(params, lr=1e-4, weight_decay=0.01)
        >>> # After computing loss and calling backward()
        >>> optimizer.step()

    Reference:
        Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017)
    """

    def __init__(
        self,
        parameters: Union[List[Tensor], Iterator[Tensor]],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        Initialize AdamW optimizer.

        Args:
            parameters: Parameters to optimize
            lr: Learning rate (default: 1e-3)
            beta1: First moment exponential decay rate (default: 0.9)
            beta2: Second moment exponential decay rate (default: 0.999)
            eps: Small constant for numerical stability (default: 1e-8)
            weight_decay: Weight decay coefficient (default: 0.01)

        Raises:
            ValueError: If any hyperparameter is out of valid range
        """
        super().__init__(parameters)

        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not 0 <= beta1 < 1:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.t = 0
        self.m: List[np.ndarray] = [np.zeros_like(p.data) for p in self.parameters]
        self.v: List[np.ndarray] = [np.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        """
        Perform a single optimization step.

        Applies decoupled weight decay and then Adam update with bias correction.
        """
        self.t += 1

        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            if self.weight_decay > 0:
                p.data = p.data - self.lr * self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad

            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset_state(self) -> None:
        """Reset optimizer state (moments and timestep)."""
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0

    def get_state(self) -> Dict:
        """Get optimizer state for checkpointing."""
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "m": [m.copy() for m in self.m],
            "v": [v.copy() for v in self.v],
            "t": self.t,
        }

    def load_state(self, state: Dict) -> None:
        """Load optimizer state from checkpoint."""
        self.lr = state["lr"]
        self.beta1 = state["beta1"]
        self.beta2 = state["beta2"]
        self.eps = state["eps"]
        self.weight_decay = state["weight_decay"]
        self.m = [m.copy() for m in state["m"]]
        self.v = [v.copy() for v in state["v"]]
        self.t = state["t"]

    def __repr__(self) -> str:
        return (
            f"AdamW(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, "
            f"eps={self.eps}, weight_decay={self.weight_decay}, n_params={len(self.parameters)})"
        )


def create_optimizer(
    parameters: Union[List[Tensor], Iterator[Tensor]],
    config: OptimizerConfig,
) -> Optimizer:
    """
    Create an optimizer from configuration.

    Args:
        parameters: Parameters to optimize
        config: OptimizerConfig with optimizer settings

    Returns:
        Optimizer instance (SGD or AdamW)

    Raises:
        ValueError: If optimizer_type is not recognized
    """
    if config.optimizer_type.lower() == "sgd":
        return SGD(
            parameters,
            lr=config.lr,
            momentum=config.momentum,
        )
    elif config.optimizer_type.lower() == "adamw":
        return AdamW(
            parameters,
            lr=config.lr,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer type: {config.optimizer_type}. "
            f"Choose from 'sgd' or 'adamw'."
        )


def get_lr_with_warmup(
    base_lr: float,
    step: int,
    warmup_steps: int = 1000,
) -> float:
    """
    Get learning rate with linear warmup.

    Args:
        base_lr: Base learning rate after warmup
        step: Current training step
        warmup_steps: Number of warmup steps

    Returns:
        Current learning rate

    Example:
        >>> for step in range(10000):
        ...     lr = get_lr_with_warmup(1e-4, step, warmup_steps=1000)
        ...     optimizer.set_lr(lr)
        ...     # training step...
    """
    if warmup_steps <= 0:
        return base_lr

    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    return base_lr


def get_lr_with_cosine_decay(
    base_lr: float,
    step: int,
    total_steps: int,
    min_lr: float = 0.0,
    warmup_steps: int = 0,
) -> float:
    """
    Get learning rate with optional warmup and cosine decay.

    Args:
        base_lr: Peak learning rate
        step: Current training step
        total_steps: Total number of training steps
        min_lr: Minimum learning rate at end of training
        warmup_steps: Number of warmup steps

    Returns:
        Current learning rate

    Example:
        >>> for step in range(10000):
        ...     lr = get_lr_with_cosine_decay(1e-4, step, 10000, warmup_steps=1000)
        ...     optimizer.set_lr(lr)
    """
    if total_steps <= 0:
        return base_lr

    if step < warmup_steps:
        return base_lr * (step / warmup_steps) if warmup_steps > 0 else base_lr

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)

    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
