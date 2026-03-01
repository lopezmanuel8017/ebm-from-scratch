"""
Sampling methods for Energy-Based Models.

This module provides:
- Langevin dynamics sampler for sampling from p(x) = exp(-E(x))/Z
- Gradient clipping utilities for stable sampling
- Step size annealing schedules
- Initialization strategies for Langevin chains
- Replay buffer for efficient sample storage and retrieval
"""

from ebm.sampling.langevin import (
    clip_grad,
    langevin_sample,
    langevin_sample_with_diagnostics,
    langevin_sample_with_config,
    linear_annealing,
    geometric_annealing,
    LangevinConfig,
    LangevinDiagnostics,
    init_from_noise,
    init_from_data,
    init_mixed,
    init_persistent_chains,
)

from ebm.sampling.replay_buffer import (
    ReplayBuffer,
    ReplayBufferConfig,
    ReplayBufferStats,
    create_replay_buffer,
)

__all__ = [
    "clip_grad",
    "langevin_sample",
    "langevin_sample_with_diagnostics",
    "langevin_sample_with_config",
    "linear_annealing",
    "geometric_annealing",
    "LangevinConfig",
    "LangevinDiagnostics",
    "init_from_noise",
    "init_from_data",
    "init_mixed",
    "init_persistent_chains",
    "ReplayBuffer",
    "ReplayBufferConfig",
    "ReplayBufferStats",
    "create_replay_buffer",
]
