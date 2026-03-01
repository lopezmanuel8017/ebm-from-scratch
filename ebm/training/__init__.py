"""
Training components for Energy-Based Models.

This module provides:
- Optimizers: SGD (with momentum) and AdamW
- Training loop: train_step and train functions
- Configuration dataclasses for training hyperparameters
"""

from ebm.training.optimizer import (
    SGD,
    AdamW,
    Optimizer,
    OptimizerConfig,
)

from ebm.training.trainer import (
    train_step,
    train,
    get_batches,
    TrainingConfig,
    TrainingStats,
    TrainingHistory,
)

__all__ = [
    "SGD",
    "AdamW",
    "Optimizer",
    "OptimizerConfig",
    "train_step",
    "train",
    "get_batches",
    "TrainingConfig",
    "TrainingStats",
    "TrainingHistory",
]
