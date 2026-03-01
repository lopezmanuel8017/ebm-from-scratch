"""
Neural network layers for Energy-Based Models.

This module implements:
- Linear (fully connected) layer with He initialization
- Activation layers (ReLU, Swish, Softplus) as thin wrappers around ops
- Sequential container for composing layers

All layers follow a consistent interface:
- __call__(x) or forward(x): compute output
- parameters(): return list of trainable Tensors
"""

import numpy as np
from typing import List, Iterator, Optional
from abc import ABC, abstractmethod

from ebm.core.autodiff import Tensor
from ebm.core.ops import relu, swish, softplus, sigmoid, matmul, add


class Module(ABC):
    """
    Base class for all neural network modules.

    Provides a consistent interface for layers and models.
    All modules should implement forward() and parameters().
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    def __call__(self, x: Tensor) -> Tensor:
        """Allow module to be called like a function."""
        return self.forward(x)

    @abstractmethod
    def parameters(self) -> List[Tensor]:
        """
        Return all trainable parameters.

        Returns:
            List of Tensors that should be updated during training
        """
        pass

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to None."""
        for param in self.parameters():
            param.zero_grad()


class Linear(Module):
    """
    Fully connected (dense) layer.

    Computes: output = input @ W + b

    Attributes:
        W: Weight matrix of shape (in_features, out_features)
        b: Bias vector of shape (out_features,)

    Weight Initialization:
        - W: He initialization (sqrt(2 / fan_in) scale)
        - b: Zeros

    He initialization is recommended for ReLU-family activations as it
    accounts for the variance reduction caused by zeroing negative values.

    Example:
        >>> layer = Linear(10, 5)
        >>> x = Tensor(np.random.randn(32, 10), requires_grad=True)
        >>> y = layer(x)  # Shape: (32, 5)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize a Linear layer.

        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If True, adds a learnable bias. Default: True
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        scale = np.sqrt(2.0 / in_features)
        self.W = Tensor(
            np.random.randn(in_features, out_features) * scale,
            requires_grad=True
        )

        if bias:
            self.b = Tensor(
                np.zeros(out_features),
                requires_grad=True
            )
        else:
            self.b = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute forward pass: output = x @ W + b.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        out = matmul(x, self.W)
        if self.b is not None:
            out = add(out, self.b)
        return out

    def parameters(self) -> List[Tensor]:
        """Return list of trainable parameters [W, b] or [W] if no bias."""
        if self.b is not None:
            return [self.W, self.b]
        return [self.W]

    def __repr__(self) -> str:
        """Return string representation for debugging and printing."""
        return f"Linear({self.in_features}, {self.out_features}, bias={self.use_bias})"


class ReLU(Module):
    """
    Rectified Linear Unit activation layer.

    Computes: output = max(0, input)

    ReLU is computationally efficient but has a "dying ReLU" problem
    where neurons can become permanently inactive. Consider using
    Swish or Softplus for smoother gradients in EBM training.

    Example:
        >>> activation = ReLU()
        >>> x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        >>> y = activation(x)  # [0.0, 0.0, 1.0]
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU activation."""
        return relu(x)

    def parameters(self) -> List[Tensor]:
        """ReLU has no trainable parameters."""
        return []

    def __repr__(self) -> str:
        """Return string representation."""
        return "ReLU()"


class Swish(Module):
    """
    Swish activation layer (also known as SiLU).

    Computes: output = x * sigmoid(x)

    Properties:
    - Smooth and non-monotonic
    - Allows negative values (unlike ReLU)
    - Has continuous first and second derivatives
    - Recommended for EBMs because smooth gradients improve Langevin dynamics

    Reference:
        Ramachandran et al., "Searching for Activation Functions" (2017)

    Example:
        >>> activation = Swish()
        >>> x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        >>> y = activation(x)  # Approximately [-0.269, 0.0, 0.731]
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply Swish activation."""
        return swish(x)

    def parameters(self) -> List[Tensor]:
        """Swish has no trainable parameters."""
        return []

    def __repr__(self) -> str:
        """Return string representation."""
        return "Swish()"


class Softplus(Module):
    """
    Softplus activation layer.

    Computes: output = log(1 + exp(input))

    Properties:
    - Smooth approximation to ReLU
    - Always positive
    - Has continuous first and second derivatives
    - Recommended for EBMs because smooth gradients improve Langevin dynamics

    Note:
        The implementation uses numerical stability tricks for large inputs
        to avoid overflow.

    Example:
        >>> activation = Softplus()
        >>> x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        >>> y = activation(x)  # Approximately [0.313, 0.693, 1.313]
    """

    def __init__(self, threshold: float = 20.0):
        """
        Initialize Softplus.

        Args:
            threshold: Above this value, softplus(x) ≈ x for numerical stability
        """
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        """Apply Softplus activation."""
        return softplus(x, threshold=self.threshold)

    def parameters(self) -> List[Tensor]:
        """Softplus has no trainable parameters."""
        return []

    def __repr__(self) -> str:
        """Return string representation with threshold parameter."""
        return f"Softplus(threshold={self.threshold})"


class Sigmoid(Module):
    """
    Sigmoid activation layer.

    Computes: output = 1 / (1 + exp(-input))

    Properties:
    - Output bounded to (0, 1)
    - Smooth with continuous derivatives
    - Can suffer from vanishing gradients for very large/small inputs

    Example:
        >>> activation = Sigmoid()
        >>> x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        >>> y = activation(x)  # Approximately [0.269, 0.5, 0.731]
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply Sigmoid activation."""
        return sigmoid(x)

    def parameters(self) -> List[Tensor]:
        """Sigmoid has no trainable parameters."""
        return []

    def __repr__(self) -> str:
        """Return string representation."""
        return "Sigmoid()"


class Sequential(Module):
    """
    A sequential container for stacking layers.

    Layers are added in order and called sequentially during forward pass.
    The output of each layer is passed as input to the next.

    Example:
        >>> model = Sequential([
        ...     Linear(10, 64),
        ...     Swish(),
        ...     Linear(64, 64),
        ...     Swish(),
        ...     Linear(64, 1)
        ... ])
        >>> x = Tensor(np.random.randn(32, 10), requires_grad=True)
        >>> y = model(x)  # Shape: (32, 1)
    """

    def __init__(self, layers: Optional[List[Module]] = None):
        """
        Initialize Sequential container.

        Args:
            layers: Optional list of modules to add initially
        """
        self.layers: List[Module] = layers if layers is not None else []

    def add(self, layer: Module) -> 'Sequential':
        """
        Add a layer to the end of the sequence.

        Args:
            layer: Module to add

        Returns:
            self (for method chaining)
        """
        self.layers.append(layer)
        return self

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply all layers in sequence.

        Args:
            x: Input tensor

        Returns:
            Output tensor after passing through all layers
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        """
        Return all trainable parameters from all layers.

        Returns:
            Flat list of all Tensors from all layers
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __getitem__(self, idx: int) -> Module:
        """Get layer by index."""
        return self.layers[idx]

    def __len__(self) -> int:
        """Return number of layers."""
        return len(self.layers)

    def __iter__(self) -> Iterator[Module]:
        """Iterate over layers."""
        return iter(self.layers)

    def __repr__(self) -> str:
        """Return formatted string representation of all layers."""
        lines = ["Sequential("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {layer}")
        lines.append(")")
        return "\n".join(lines)


def _create_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "swish",
    output_activation: bool = False
) -> Sequential:
    """
    Helper function to create a Multi-Layer Perceptron.

    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        activation: Activation function ("relu", "swish", "softplus")
        output_activation: Whether to apply activation after final layer

    Returns:
        Sequential model
    """
    activation_map = {
        "relu": ReLU,
        "swish": Swish,
        "softplus": Softplus,
        "sigmoid": Sigmoid,
    }

    if activation not in activation_map:
        raise ValueError(f"Unknown activation: {activation}. Choose from {list(activation_map.keys())}")

    act_class = activation_map[activation]
    layers: List[Module] = []

    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(Linear(prev_dim, hidden_dim))
        layers.append(act_class())
        prev_dim = hidden_dim

    layers.append(Linear(prev_dim, output_dim))
    if output_activation:
        layers.append(act_class())

    return Sequential(layers)
