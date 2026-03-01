"""
Energy function wrapper for Energy-Based Models.

This module implements:
- EnergyMLP: A wrapper class that ensures proper energy function behavior
  - Scalar output (batch_size, 1)
  - Score function computation (gradient w.r.t. input)
  - Flexible architecture specification

The energy function E(x) defines a probability distribution:
    p(x) = exp(-E(x)) / Z

where Z is the intractable partition function.
"""

import numpy as np
from typing import List, Optional, Union

from ebm.core.autodiff import Tensor
from ebm.core.nn import Module, Sequential, Linear, ReLU, Swish, Softplus, Sigmoid
from ebm.core.ops import tensor_sum


class EnergyMLP(Module):
    """
    Multi-Layer Perceptron for computing energy values.

    Wraps a Sequential network and provides energy-specific methods.
    The output is always a scalar per sample (shape: batch_size, 1).

    Attributes:
        network: The underlying Sequential network
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function type used

    Methods:
        forward(x): Compute energy values
        energy(x): Alias for forward
        score(x): Compute negative gradient of energy w.r.t. input
        parameters(): Return all trainable weights

    The score function computes:
        score(x) = -nabla_x E(x)

    This is used in:
    - Langevin dynamics sampling (gradient descent on energy)
    - Score matching training objectives

    Example:
        >>> energy_fn = EnergyMLP(input_dim=2, hidden_dims=[128, 128])
        >>> x = Tensor(np.random.randn(32, 2), requires_grad=True)
        >>> E = energy_fn(x)  # Shape: (32, 1)
        >>> scores = energy_fn.score(x)  # Shape: (32, 2)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = "swish",
        network: Optional[Sequential] = None
    ):
        """
        Initialize an EnergyMLP.

        Args:
            input_dim: Input feature dimension (e.g., 2 for 2D data, 29 for credit card)
            hidden_dims: List of hidden layer dimensions (e.g., [128, 128] or [256, 256])
            activation: Activation function to use. Options: "relu", "swish", "softplus", "sigmoid"
                       Default is "swish" which is recommended for EBMs due to smooth gradients.
            network: Optional pre-built Sequential network. If provided, input_dim,
                    hidden_dims, and activation are ignored.

        Raises:
            ValueError: If activation is not one of the supported types
            ValueError: If hidden_dims is empty and network is not provided

        Architecture:
            For 2D data (recommended):
                Linear(2, 128) -> Swish -> Linear(128, 128) -> Swish -> Linear(128, 1)

            For tabular data (recommended):
                Linear(d, 256) -> Swish -> Linear(256, 256) -> Swish -> Linear(256, 1)
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation_name = activation

        if network is not None:
            self.network = network
        else:
            if not hidden_dims:
                raise ValueError("hidden_dims must not be empty when network is not provided")

            self.network = self._build_network(input_dim, hidden_dims, activation)

    def _build_network(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str
    ) -> Sequential:
        """
        Build the MLP network.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name

        Returns:
            Sequential network ending with output dimension 1
        """
        activation_map = {
            "relu": ReLU,
            "swish": Swish,
            "softplus": Softplus,
            "sigmoid": Sigmoid,
        }

        if activation not in activation_map:
            raise ValueError(
                f"Unknown activation: {activation}. "
                f"Choose from {list(activation_map.keys())}"
            )

        act_class = activation_map[activation]
        layers: List[Module] = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(Linear(prev_dim, hidden_dim))
            layers.append(act_class())
            prev_dim = hidden_dim

        layers.append(Linear(prev_dim, 1))

        return Sequential(layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute energy values for input samples.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)

        Returns:
            Energy values of shape (batch_size, 1) or (1,) for single sample

        Note:
            Lower energy corresponds to higher probability under the model:
            p(x) propto exp(-E(x))
        """
        _squeeze_output = False
        if x.data.ndim == 1:
            x = Tensor(x.data.reshape(1, -1), requires_grad=x.requires_grad)
            _squeeze_output = True

        output = self.network(x)

        if output.data.ndim == 1:
            output = Tensor(
                output.data.reshape(-1, 1),
                requires_grad=output.requires_grad,
                _prev=output._prev,
                _backward=output._backward
            )

        return output

    def energy(self, x: Tensor) -> Tensor:
        """
        Alias for forward(). Compute energy values.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Energy values of shape (batch_size, 1)
        """
        return self.forward(x)

    def score(self, x: Union[Tensor, np.ndarray]) -> np.ndarray:
        """
        Compute the score function: negative gradient of energy w.r.t. input.

        score(x) = -nabla_x E(x)

        The score function points in the direction of decreasing energy
        (increasing probability). It is used in:
        - Langevin dynamics: x_{t+1} = x_t + eps * score(x_t) + noise
        - Score matching: minimize ||s_theta(x) - nabla_x log p(x)||^2

        Args:
            x: Input tensor of shape (batch_size, input_dim) or numpy array

        Returns:
            Score values as numpy array of shape (batch_size, input_dim)

        Note:
            This method always returns a numpy array (not a Tensor) because
            the score is typically used for sampling, not further autodiff.
        """
        if isinstance(x, np.ndarray):
            x_tensor = Tensor(x, requires_grad=True)
        else:
            x_tensor = Tensor(x.data.copy(), requires_grad=True)

        energy = self.forward(x_tensor)

        energy_sum = tensor_sum(energy)

        energy_sum.backward()

        if x_tensor.grad is None:
            return np.zeros_like(x_tensor.data)

        return -x_tensor.grad.copy()

    def energy_and_score(self, x: Union[Tensor, np.ndarray]) -> tuple:
        """
        Compute both energy and score in a single forward-backward pass.

        This is more efficient than calling energy() and score() separately
        when both values are needed (e.g., during Langevin sampling diagnostics).

        Args:
            x: Input tensor of shape (batch_size, input_dim) or numpy array

        Returns:
            Tuple of (energy_values, score_values) where:
            - energy_values: numpy array of shape (batch_size, 1)
            - score_values: numpy array of shape (batch_size, input_dim)
        """
        if isinstance(x, np.ndarray):
            x_tensor = Tensor(x, requires_grad=True)
        else:
            x_tensor = Tensor(x.data.copy(), requires_grad=True)

        energy = self.forward(x_tensor)
        energy_values = energy.data.copy()

        energy_sum = tensor_sum(energy)

        energy_sum.backward()

        if x_tensor.grad is None:
            score_values = np.zeros_like(x_tensor.data)
        else:
            score_values = -x_tensor.grad.copy()

        return energy_values, score_values

    def parameters(self) -> List[Tensor]:
        """
        Return all trainable parameters.

        Returns:
            List of all weight and bias Tensors from the network
        """
        return self.network.parameters()

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to None."""
        self.network.zero_grad()

    def __repr__(self) -> str:
        """Return string representation of the energy function."""
        return (
            f"EnergyMLP(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dims={self.hidden_dims},\n"
            f"  activation={self.activation_name}\n"
            f")"
        )

    def __call__(self, x: Tensor) -> Tensor:
        """Allow EnergyMLP to be called like a function."""
        return self.forward(x)


def create_energy_network_2d(hidden_dims: Optional[List[int]] = None) -> EnergyMLP:
    """
    Create an energy network for 2D data.

    Uses the recommended architecture from the EBM paper:
    Linear(2, 128) -> Swish -> Linear(128, 128) -> Swish -> Linear(128, 1)

    Args:
        hidden_dims: Optional custom hidden dimensions. Defaults to [128, 128].

    Returns:
        EnergyMLP configured for 2D input
    """
    if hidden_dims is None:
        hidden_dims = [128, 128]
    return EnergyMLP(input_dim=2, hidden_dims=hidden_dims, activation="swish")


def create_energy_network_tabular(
    input_dim: int,
    hidden_dims: Optional[List[int]] = None
) -> EnergyMLP:
    """
    Create an energy network for tabular data.

    Uses the recommended architecture from the EBM paper:
    Linear(d, 256) -> Swish -> Linear(256, 256) -> Swish -> Linear(256, 1)

    Args:
        input_dim: Number of input features
        hidden_dims: Optional custom hidden dimensions. Defaults to [256, 256].

    Returns:
        EnergyMLP configured for tabular data
    """
    if hidden_dims is None:
        hidden_dims = [256, 256]
    return EnergyMLP(input_dim=input_dim, hidden_dims=hidden_dims, activation="swish")
