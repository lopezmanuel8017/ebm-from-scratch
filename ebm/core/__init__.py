"""
Core autodiff and neural network components.

This module serves as the public API for the EBM core functionality.
It re-exports all the essential classes and functions from submodules,
allowing users to import directly from ebm.core instead of navigating
to individual submodules.

The core package contains:
- Tensor: The fundamental data structure with automatic differentiation
- Operations: Differentiable mathematical operations (add, mul, matmul, etc.)
- Neural Network Layers: Linear, ReLU, Swish, Softplus, Sequential
- Energy Functions: EnergyMLP for defining energy-based models

Example usage:
    from ebm.core import Tensor, EnergyMLP, Linear, Swish

    # Create a tensor with gradient tracking
    x = Tensor(data, requires_grad=True)

    # Build an energy function
    energy_fn = EnergyMLP(input_dim=2, hidden_dims=[128, 128])
"""

from ebm.core.autodiff import Tensor

from ebm.core.ops import (
    add,
    sub,
    mul,
    div,
    neg,
    matmul,
    transpose,
    tensor_sum,
    mean,
    relu,
    sigmoid,
    softplus,
    swish,
    exp,
    log,
    pow,
    sqrt,
    unbroadcast,
)

from ebm.core.nn import (
    Module,
    Linear,
    ReLU,
    Swish,
    Softplus,
    Sigmoid,
    Sequential,
)

from ebm.core.energy import (
    EnergyMLP,
    create_energy_network_2d,
    create_energy_network_tabular,
)

__all__ = [
    "Tensor",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "matmul",
    "transpose",
    "tensor_sum",
    "mean",
    "relu",
    "sigmoid",
    "softplus",
    "swish",
    "exp",
    "log",
    "pow",
    "sqrt",
    "unbroadcast",
    "Module",
    "Linear",
    "ReLU",
    "Swish",
    "Softplus",
    "Sigmoid",
    "Sequential",
    "EnergyMLP",
    "create_energy_network_2d",
    "create_energy_network_tabular",
]
