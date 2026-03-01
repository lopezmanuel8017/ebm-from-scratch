"""
Spectral Normalization for Energy-Based Models.

Spectral normalization constrains the Lipschitz constant of each layer by
normalizing weights by their spectral norm (largest singular value). This
prevents the energy landscape from becoming too "spiky" and improves training
stability.

The spectral norm of a matrix W is its largest singular value:
    sigma(W) = max_{x != 0} ||Wx|| / ||x||

Power iteration is used to efficiently approximate the spectral norm without
computing the full SVD.

Reference:
    Miyato et al., "Spectral Normalization for Generative Adversarial Networks" (2018)

Usage:
    >>> from ebm.stability import spectral_norm, SpectralNormWrapper
    >>>
    >>> # Normalize a weight matrix
    >>> W_normalized, u = spectral_norm(W)
    >>>
    >>> # Wrap a Linear layer with spectral normalization
    >>> layer = SpectralNormWrapper(Linear(10, 5))
    >>> output = layer(input)  # Weights are normalized before each forward pass
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

from ebm.core import Tensor, Module, Linear

@dataclass
class SpectralNormState:
    """
    State for spectral normalization power iteration.

    Storing u between iterations improves efficiency by starting power iteration
    from the previous estimate rather than random initialization.

    Attributes:
        u: Right singular vector estimate, shape (n_cols,) or (fan_out,)
        v: Left singular vector estimate, shape (n_rows,) or (fan_in,)
        sigma: Estimated spectral norm (largest singular value)
        n_iterations: Number of power iterations performed
    """
    u: np.ndarray
    v: Optional[np.ndarray] = None
    sigma: float = 1.0
    n_iterations: int = 0


def spectral_norm_power_iteration(
    W: np.ndarray,
    u: Optional[np.ndarray] = None,
    n_iterations: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute spectral norm using power iteration.

    Power iteration estimates the largest singular value (spectral norm) of W
    without computing the full SVD. Each iteration refines the estimate.

    Algorithm:
        1. v = W @ u / ||W @ u||
        2. u = W.T @ v / ||W.T @ v||
        3. sigma = ||W @ u||

    Args:
        W: Weight matrix of shape (n_rows, n_cols) or (fan_in, fan_out)
        u: Initial estimate of right singular vector, shape (n_cols,).
           If None, initialized randomly.
        n_iterations: Number of power iterations (default: 1)

    Returns:
        Tuple of (u, v, sigma) where:
            - u: Right singular vector estimate, shape (n_cols,)
            - v: Left singular vector estimate, shape (n_rows,)
            - sigma: Estimated spectral norm

    Example:
        >>> W = np.random.randn(10, 5)
        >>> u, v, sigma = spectral_norm_power_iteration(W, n_iterations=10)
        >>> # sigma is approximately the largest singular value of W
    """
    if W.ndim == 1:
        sigma = np.linalg.norm(W)
        return W / (sigma + 1e-8), W / (sigma + 1e-8), sigma

    _n_rows, n_cols = W.shape

    if u is None:
        u = np.random.randn(n_cols)
    u = u / (np.linalg.norm(u) + 1e-8)

    for _ in range(n_iterations):
        v = W @ u
        v = v / (np.linalg.norm(v) + 1e-8)

        u = W.T @ v
        u = u / (np.linalg.norm(u) + 1e-8)

    sigma = np.linalg.norm(W @ u)

    return u, v, sigma


def spectral_norm(
    W: np.ndarray,
    u: Optional[np.ndarray] = None,
    n_iterations: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize weight matrix by its spectral norm.

    Returns W / sigma(W) where sigma(W) is the spectral norm (largest singular value).
    The normalized matrix has spectral norm 1, bounding the Lipschitz constant.

    Args:
        W: Weight matrix of shape (n_rows, n_cols)
        u: Previous estimate of right singular vector for warm starting
        n_iterations: Number of power iterations (default: 1)

    Returns:
        Tuple of (W_normalized, u) where:
            - W_normalized: Weight matrix normalized by spectral norm
            - u: Updated right singular vector (save for next call)

    Example:
        >>> W = np.random.randn(10, 5) * 3  # Large weights
        >>> W_norm, u = spectral_norm(W)
        >>> np.linalg.norm(W_norm, ord=2)  # Approximately 1.0
    """
    u, _v, sigma = spectral_norm_power_iteration(W, u, n_iterations)

    W_normalized = W / (sigma + 1e-8)

    return W_normalized, u


def compute_spectral_norm_exact(W: np.ndarray) -> float:
    """
    Compute exact spectral norm using SVD.

    This is slower than power iteration but provides the exact value.
    Useful for testing and validation.

    Args:
        W: Weight matrix

    Returns:
        Exact spectral norm (largest singular value)
    """
    if W.ndim == 1:
        return np.linalg.norm(W)

    s = np.linalg.svd(W, compute_uv=False)
    return s[0]


class SpectralNormWrapper:
    """
    Wrapper that applies spectral normalization to a Linear layer.

    Normalizes the weight matrix before each forward pass, ensuring the
    layer has Lipschitz constant <= 1.

    Attributes:
        layer: The wrapped Linear layer
        n_iterations: Power iterations per forward pass
        enabled: Whether normalization is active (can be disabled for inference)

    Example:
        >>> from ebm.core.nn import Linear
        >>> layer = Linear(10, 5)
        >>> wrapped = SpectralNormWrapper(layer)
        >>> x = Tensor(np.random.randn(32, 10), requires_grad=True)
        >>> y = wrapped(x)  # Weights normalized before forward
    """

    def __init__(
        self,
        layer: Linear,
        n_iterations: int = 1,
        name: str = "W",
    ):
        """
        Initialize spectral norm wrapper.

        Args:
            layer: Linear layer to wrap
            n_iterations: Power iterations per forward pass (default: 1)
            name: Name of weight attribute (default: "W")
        """
        self.layer = layer
        self.n_iterations = n_iterations
        self.name = name
        self.enabled = True

        W = getattr(layer, name)
        if W.data.ndim == 1:
            self._u = W.data.copy()
        else:
            n_cols = W.data.shape[1]
            self._u = np.random.randn(n_cols)
            self._u = self._u / (np.linalg.norm(self._u) + 1e-8)

        self._original_W = W.data.copy()

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass with spectral normalization."""
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply forward pass with spectral normalized weights.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        if self.enabled:
            self._apply_spectral_norm()
        return self.layer(x)

    def _apply_spectral_norm(self) -> None:
        """Apply spectral normalization to weights."""
        W = getattr(self.layer, self.name)
        W_normalized, self._u = spectral_norm(
            W.data, self._u, self.n_iterations
        )
        W.data = W_normalized

    def get_spectral_norm(self) -> float:
        """Get current spectral norm of the weight matrix."""
        W = getattr(self.layer, self.name)
        _, _, sigma = spectral_norm_power_iteration(W.data, self._u, 10)
        return sigma

    def parameters(self) -> List[Tensor]:
        """Return parameters from the wrapped layer."""
        return self.layer.parameters()

    def enable(self) -> None:
        """Enable spectral normalization."""
        self.enabled = True

    def disable(self) -> None:
        """Disable spectral normalization."""
        self.enabled = False

    def restore_original(self) -> None:
        """Restore original (non-normalized) weights."""
        W = getattr(self.layer, self.name)
        W.data = self._original_W.copy()

    def __repr__(self) -> str:
        return f"SpectralNormWrapper({self.layer}, n_iter={self.n_iterations})"


def apply_spectral_norm_to_layer(
    layer: Linear,
    n_iterations: int = 1,
) -> SpectralNormWrapper:
    """
    Apply spectral normalization to a Linear layer.

    Args:
        layer: Linear layer to normalize
        n_iterations: Power iterations per forward pass

    Returns:
        SpectralNormWrapper wrapping the layer
    """
    return SpectralNormWrapper(layer, n_iterations=n_iterations)


def apply_spectral_norm_to_model(
    model: Module,
    n_iterations: int = 1,
) -> List[SpectralNormWrapper]:
    """
    Apply spectral normalization to all Linear layers in a model.

    Note: This modifies the model in place by wrapping Linear layers.
    For Sequential models, returns a list of wrappers that replace Linear layers.

    Args:
        model: Model (Sequential) containing Linear layers
        n_iterations: Power iterations per forward pass

    Returns:
        List of SpectralNormWrapper objects

    Example:
        >>> from ebm.core.nn import Sequential, Linear, Swish
        >>> model = Sequential([Linear(10, 32), Swish(), Linear(32, 1)])
        >>> wrappers = apply_spectral_norm_to_model(model)
        >>> len(wrappers)  # 2 (one for each Linear layer)
    """

    wrappers = []

    if hasattr(model, 'layers'):
        new_layers = []
        for layer in model.layers:
            if isinstance(layer, Linear):
                wrapper = SpectralNormWrapper(layer, n_iterations=n_iterations)
                new_layers.append(wrapper)
                wrappers.append(wrapper)
            else:
                new_layers.append(layer)
        model.layers = new_layers
    elif isinstance(model, Linear):
        wrapper = SpectralNormWrapper(model, n_iterations=n_iterations)
        wrappers.append(wrapper)

    return wrappers


def remove_spectral_norm_from_layer(wrapper: SpectralNormWrapper) -> Linear:
    """
    Remove spectral normalization from a wrapped layer.

    Args:
        wrapper: SpectralNormWrapper to unwrap

    Returns:
        The original Linear layer with normalized weights
    """
    wrapper.disable()
    return wrapper.layer


def get_layer_spectral_norms(model: Module) -> List[float]:
    """
    Compute spectral norms for all Linear layers in a model.

    Useful for monitoring and debugging - large spectral norms indicate
    layers that may cause instability.

    Args:
        model: Model containing Linear layers

    Returns:
        List of spectral norms for each Linear layer
    """

    norms = []

    if hasattr(model, 'layers'):
        for layer in model.layers:
            if isinstance(layer, Linear):
                sigma = compute_spectral_norm_exact(layer.W.data)
                norms.append(sigma)
            elif isinstance(layer, SpectralNormWrapper):
                sigma = layer.get_spectral_norm()
                norms.append(sigma)
    elif isinstance(model, Linear):
        norms.append(compute_spectral_norm_exact(model.W.data))

    return norms
