"""
Differentiable operations for automatic differentiation.

Each operation:
1. Computes the forward pass
2. Defines a _backward function that computes gradients
3. Returns a new Tensor with _prev set to inputs

Broadcasting is handled by summing gradients along broadcast dimensions.
Numerical stability techniques are used for activations.
"""

import numpy as np
from typing import Optional, Union, Tuple

from ebm.core.autodiff import Tensor


def unbroadcast(grad: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Sum gradient along broadcast dimensions to match original shape.

    When operations broadcast inputs to a common shape, the gradient
    must be summed back along the broadcast dimensions.

    Args:
        grad: Gradient with potentially larger shape due to broadcasting
        original_shape: The shape the gradient should have

    Returns:
        Gradient summed along broadcast dimensions to match original_shape
    """
    ndims_added = grad.ndim - len(original_shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(original_shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


def add(a: Tensor, b: Tensor) -> Tensor:
    """
    Element-wise addition with broadcasting support.

    Forward: c = a + b
    Backward: da = dc, db = dc (with unbroadcasting)

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        New tensor containing a + b
    """
    result_data = a.data + b.data
    requires_grad = a.requires_grad or b.requires_grad
    result = Tensor(result_data, requires_grad=requires_grad, _prev={a, b})

    if requires_grad:
        def _backward():
            if a.requires_grad:
                grad_a = unbroadcast(result.grad, a.shape)
                if a.grad is None:
                    a.grad = np.zeros_like(a.data)
                a.grad = a.grad + grad_a

            if b.requires_grad:
                grad_b = unbroadcast(result.grad, b.shape)
                if b.grad is None:
                    b.grad = np.zeros_like(b.data)
                b.grad = b.grad + grad_b

        result._backward = _backward

    return result


def sub(a: Tensor, b: Tensor) -> Tensor:
    """
    Element-wise subtraction with broadcasting support.

    Forward: c = a - b
    Backward: da = dc, db = -dc (with unbroadcasting)

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        New tensor containing a - b
    """
    result_data = a.data - b.data
    requires_grad = a.requires_grad or b.requires_grad
    result = Tensor(result_data, requires_grad=requires_grad, _prev={a, b})

    if requires_grad:
        def _backward():
            if a.requires_grad:
                grad_a = unbroadcast(result.grad, a.shape)
                if a.grad is None:
                    a.grad = np.zeros_like(a.data)
                a.grad = a.grad + grad_a

            if b.requires_grad:
                grad_b = unbroadcast(-result.grad, b.shape)
                if b.grad is None:
                    b.grad = np.zeros_like(b.data)
                b.grad = b.grad + grad_b

        result._backward = _backward

    return result


def mul(a: Tensor, b: Tensor) -> Tensor:
    """
    Element-wise multiplication with broadcasting support.

    Forward: c = a * b
    Backward: da = dc * b, db = dc * a (with unbroadcasting)

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        New tensor containing a * b
    """
    result_data = a.data * b.data
    requires_grad = a.requires_grad or b.requires_grad
    result = Tensor(result_data, requires_grad=requires_grad, _prev={a, b})

    if requires_grad:
        def _backward():
            if a.requires_grad:
                grad_a = unbroadcast(result.grad * b.data, a.shape)
                if a.grad is None:
                    a.grad = np.zeros_like(a.data)
                a.grad = a.grad + grad_a

            if b.requires_grad:
                grad_b = unbroadcast(result.grad * a.data, b.shape)
                if b.grad is None:
                    b.grad = np.zeros_like(b.data)
                b.grad = b.grad + grad_b

        result._backward = _backward

    return result


def div(a: Tensor, b: Tensor) -> Tensor:
    """
    Element-wise division with broadcasting support.

    Forward: c = a / b
    Backward: da = dc / b, db = -dc * a / b^2 (with unbroadcasting)

    Args:
        a: Numerator tensor
        b: Denominator tensor

    Returns:
        New tensor containing a / b
    """
    result_data = a.data / b.data
    requires_grad = a.requires_grad or b.requires_grad
    result = Tensor(result_data, requires_grad=requires_grad, _prev={a, b})

    if requires_grad:
        def _backward():
            if a.requires_grad:
                grad_a = unbroadcast(result.grad / b.data, a.shape)
                if a.grad is None:
                    a.grad = np.zeros_like(a.data)
                a.grad = a.grad + grad_a

            if b.requires_grad:
                grad_b = unbroadcast(-result.grad * a.data / (b.data ** 2), b.shape)
                if b.grad is None:
                    b.grad = np.zeros_like(b.data)
                b.grad = b.grad + grad_b

        result._backward = _backward

    return result


def neg(a: Tensor) -> Tensor:
    """
    Element-wise negation.

    Forward: c = -a
    Backward: da = -dc

    Args:
        a: Input tensor

    Returns:
        New tensor containing -a
    """
    result_data = -a.data
    result = Tensor(result_data, requires_grad=a.requires_grad, _prev={a})

    if a.requires_grad:
        def _backward():
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad = a.grad - result.grad

        result._backward = _backward

    return result


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Matrix multiplication.

    Forward: c = a @ b
    Backward: da = dc @ b.T, db = a.T @ dc

    For batched matmul, the batch dimensions are handled automatically.

    Args:
        a: First tensor of shape (..., m, k)
        b: Second tensor of shape (..., k, n)

    Returns:
        New tensor of shape (..., m, n)
    """
    result_data = a.data @ b.data
    requires_grad = a.requires_grad or b.requires_grad
    result = Tensor(result_data, requires_grad=requires_grad, _prev={a, b})

    if requires_grad:
        def _backward():
            if a.requires_grad:
                if b.data.ndim == 1:
                    grad_a = np.outer(result.grad, b.data) if result.grad.ndim == 1 else result.grad[..., np.newaxis] @ b.data[np.newaxis, :]
                else:
                    grad_a = result.grad @ np.swapaxes(b.data, -2, -1)

                if a.grad is None:
                    a.grad = np.zeros_like(a.data)
                a.grad = a.grad + grad_a

            if b.requires_grad:
                if a.data.ndim == 1:
                    grad_b = np.outer(a.data, result.grad) if result.grad.ndim == 1 else a.data[..., np.newaxis] @ result.grad[np.newaxis, :]
                else:
                    grad_b = np.swapaxes(a.data, -2, -1) @ result.grad

                if b.grad is None:
                    b.grad = np.zeros_like(b.data)
                b.grad = b.grad + grad_b

        result._backward = _backward

    return result


def transpose(a: Tensor, axes: Optional[Tuple[int, ...]] = None) -> Tensor:
    """
    Transpose tensor axes.

    Forward: c = a.transpose(axes)
    Backward: da = dc.transpose(inverse_axes)

    Args:
        a: Input tensor
        axes: Permutation of axes. If None, reverses all axes.

    Returns:
        Transposed tensor
    """
    result_data = a.data.transpose(axes)
    result = Tensor(result_data, requires_grad=a.requires_grad, _prev={a})

    if a.requires_grad:
        def _backward():
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            if axes is None:
                inv_axes = None
            else:
                inv_axes = tuple(np.argsort(axes))
            a.grad = a.grad + result.grad.transpose(inv_axes)

        result._backward = _backward

    return result


# =============================================================================
# Reduction Operations
# =============================================================================

def tensor_sum(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None,
               keepdims: bool = False) -> Tensor:
    """
    Sum of tensor elements over given axis.

    Forward: c = sum(a, axis)
    Backward: da = broadcast(dc) to original shape

    Args:
        a: Input tensor
        axis: Axis or axes to sum over. None sums all elements.
        keepdims: If True, reduced axes are kept as size-1 dimensions.

    Returns:
        Reduced tensor
    """
    result_data = a.data.sum(axis=axis, keepdims=keepdims)
    result = Tensor(result_data, requires_grad=a.requires_grad, _prev={a})

    if a.requires_grad:
        def _backward():
            if a.grad is None:
                a.grad = np.zeros_like(a.data)

            grad = result.grad

            if not keepdims and axis is not None:
                if isinstance(axis, int):
                    grad = np.expand_dims(grad, axis=axis)
                else:
                    for ax in sorted(axis):
                        grad = np.expand_dims(grad, axis=ax)

            a.grad = a.grad + np.broadcast_to(grad, a.shape)

        result._backward = _backward

    return result


def mean(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None,
         keepdims: bool = False) -> Tensor:
    """
    Mean of tensor elements over given axis.

    Forward: c = mean(a, axis)
    Backward: da = broadcast(dc) / n to original shape

    Args:
        a: Input tensor
        axis: Axis or axes to average over. None averages all elements.
        keepdims: If True, reduced axes are kept as size-1 dimensions.

    Returns:
        Reduced tensor
    """
    result_data = a.data.mean(axis=axis, keepdims=keepdims)
    result = Tensor(result_data, requires_grad=a.requires_grad, _prev={a})

    if a.requires_grad:
        def _backward():
            if a.grad is None:
                a.grad = np.zeros_like(a.data)

            if axis is None:
                n = a.data.size
            elif isinstance(axis, int):
                n = a.data.shape[axis]
            else:
                n = np.prod([a.data.shape[ax] for ax in axis])

            grad = result.grad

            if not keepdims and axis is not None:
                if isinstance(axis, int):
                    grad = np.expand_dims(grad, axis=axis)
                else:
                    for ax in sorted(axis):
                        grad = np.expand_dims(grad, axis=ax)

            a.grad = a.grad + np.broadcast_to(grad, a.shape) / n

        result._backward = _backward

    return result


def relu(a: Tensor) -> Tensor:
    """
    Rectified Linear Unit activation.

    Forward: c = max(0, a)
    Backward: da = dc * (a > 0)

    Args:
        a: Input tensor

    Returns:
        ReLU activated tensor
    """
    result_data = np.maximum(0, a.data)
    result = Tensor(result_data, requires_grad=a.requires_grad, _prev={a})

    if a.requires_grad:
        def _backward():
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad = a.grad + result.grad * (a.data > 0).astype(np.float64)

        result._backward = _backward

    return result


def sigmoid(a: Tensor) -> Tensor:
    """
    Sigmoid activation with numerical stability.

    Forward: c = 1 / (1 + exp(-a))
    Backward: da = dc * c * (1 - c)

    Numerical stability:
    - For a >= 0: use 1 / (1 + exp(-a))
    - For a < 0: use exp(a) / (1 + exp(a))

    Args:
        a: Input tensor

    Returns:
        Sigmoid activated tensor
    """
    result_data = np.where(
        a.data >= 0,
        1 / (1 + np.exp(-a.data)),
        np.exp(a.data) / (1 + np.exp(a.data))
    )
    result = Tensor(result_data, requires_grad=a.requires_grad, _prev={a})

    if a.requires_grad:
        def _backward():
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad = a.grad + result.grad * result.data * (1 - result.data)

        result._backward = _backward

    return result


def softplus(a: Tensor, threshold: float = 20.0) -> Tensor:
    """
    Softplus activation with numerical stability.

    Forward: c = log(1 + exp(a))
    Backward: da = dc * sigmoid(a)

    Numerical stability:
    - For a < threshold: use log(1 + exp(a))
    - For a >= threshold: return a directly (avoids overflow)

    Args:
        a: Input tensor
        threshold: Above this value, softplus(a) ≈ a

    Returns:
        Softplus activated tensor
    """
    result_data = np.where(
        a.data < threshold,
        np.log1p(np.exp(a.data)),
        a.data
    )
    result = Tensor(result_data, requires_grad=a.requires_grad, _prev={a})

    if a.requires_grad:
        def _backward():
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            sig = np.where(
                a.data >= 0,
                1 / (1 + np.exp(-a.data)),
                np.exp(a.data) / (1 + np.exp(a.data))
            )
            grad_local = np.where(a.data < threshold, sig, 1.0)
            a.grad = a.grad + result.grad * grad_local

        result._backward = _backward

    return result


def swish(a: Tensor) -> Tensor:
    """
    Swish activation (also known as SiLU).

    Forward: c = a * sigmoid(a)
    Backward: da = dc * (c + sigmoid(a) * (1 - c))
             = dc * (sigmoid(a) + a * sigmoid(a) * (1 - sigmoid(a)))
             = dc * sigmoid(a) * (1 + a * (1 - sigmoid(a)))

    Args:
        a: Input tensor

    Returns:
        Swish activated tensor
    """
    sig = np.where(
        a.data >= 0,
        1 / (1 + np.exp(-a.data)),
        np.exp(a.data) / (1 + np.exp(a.data))
    )
    result_data = a.data * sig
    result = Tensor(result_data, requires_grad=a.requires_grad, _prev={a})

    if a.requires_grad:
        def _backward():
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            grad_local = result.data + sig * (1 - result.data)
            a.grad = a.grad + result.grad * grad_local

        result._backward = _backward

    return result


def exp(a: Tensor) -> Tensor:
    """
    Element-wise exponential.

    Forward: c = exp(a)
    Backward: da = dc * exp(a)

    Args:
        a: Input tensor

    Returns:
        Exponentiated tensor
    """
    result_data = np.exp(a.data)
    result = Tensor(result_data, requires_grad=a.requires_grad, _prev={a})

    if a.requires_grad:
        def _backward():
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad = a.grad + result.grad * result.data

        result._backward = _backward

    return result


def log(a: Tensor, eps: float = 1e-10) -> Tensor:
    """
    Element-wise natural logarithm with numerical stability.

    Forward: c = log(a + eps)
    Backward: da = dc / (a + eps)

    Args:
        a: Input tensor (should be positive)
        eps: Small constant to prevent log(0)

    Returns:
        Log of tensor
    """
    safe_data = a.data + eps
    result_data = np.log(safe_data)
    result = Tensor(result_data, requires_grad=a.requires_grad, _prev={a})

    if a.requires_grad:
        def _backward():
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad = a.grad + result.grad / safe_data

        result._backward = _backward

    return result


def pow(a: Tensor, n: Union[int, float]) -> Tensor:
    """
    Element-wise power.

    Forward: c = a^n
    Backward: da = dc * n * a^(n-1)

    Args:
        a: Base tensor
        n: Exponent (scalar)

    Returns:
        a raised to the power n
    """
    result_data = a.data ** n
    result = Tensor(result_data, requires_grad=a.requires_grad, _prev={a})

    if a.requires_grad:
        def _backward():
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            if n == 0:
                grad_local = np.zeros_like(a.data)
            elif n == 1:
                grad_local = np.ones_like(a.data)
            else:
                grad_local = n * (a.data ** (n - 1))
            a.grad = a.grad + result.grad * grad_local

        result._backward = _backward

    return result


def sqrt(a: Tensor, eps: float = 1e-10) -> Tensor:
    """
    Element-wise square root.

    Forward: c = sqrt(a)
    Backward: da = dc * 0.5 / sqrt(a)

    Args:
        a: Input tensor (should be non-negative)
        eps: Small constant to prevent division by zero in gradient

    Returns:
        Square root of tensor
    """
    result_data = np.sqrt(a.data)
    result = Tensor(result_data, requires_grad=a.requires_grad, _prev={a})

    if a.requires_grad:
        def _backward():
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad = a.grad + result.grad * 0.5 / (result.data + eps)

        result._backward = _backward

    return result


def _tensor_add(self: Tensor, other: Union[Tensor, np.ndarray, float, int]) -> Tensor:
    """Add method for Tensor class."""
    if not isinstance(other, Tensor):
        other = Tensor(other, requires_grad=False)
    return add(self, other)


def _tensor_radd(self: Tensor, other: Union[np.ndarray, float, int]) -> Tensor:
    """Reverse add for Tensor class."""
    other = Tensor(other, requires_grad=False)
    return add(other, self)


def _tensor_sub(self: Tensor, other: Union[Tensor, np.ndarray, float, int]) -> Tensor:
    """Subtract method for Tensor class."""
    if not isinstance(other, Tensor):
        other = Tensor(other, requires_grad=False)
    return sub(self, other)


def _tensor_rsub(self: Tensor, other: Union[np.ndarray, float, int]) -> Tensor:
    """Reverse subtract for Tensor class."""
    other = Tensor(other, requires_grad=False)
    return sub(other, self)


def _tensor_mul(self: Tensor, other: Union[Tensor, np.ndarray, float, int]) -> Tensor:
    """Multiply method for Tensor class."""
    if not isinstance(other, Tensor):
        other = Tensor(other, requires_grad=False)
    return mul(self, other)


def _tensor_rmul(self: Tensor, other: Union[np.ndarray, float, int]) -> Tensor:
    """Reverse multiply for Tensor class."""
    other = Tensor(other, requires_grad=False)
    return mul(other, self)


def _tensor_truediv(self: Tensor, other: Union[Tensor, np.ndarray, float, int]) -> Tensor:
    """Division method for Tensor class."""
    if not isinstance(other, Tensor):
        other = Tensor(other, requires_grad=False)
    return div(self, other)


def _tensor_rtruediv(self: Tensor, other: Union[np.ndarray, float, int]) -> Tensor:
    """Reverse division for Tensor class."""
    other = Tensor(other, requires_grad=False)
    return div(other, self)


def _tensor_neg(self: Tensor) -> Tensor:
    """Negation method for Tensor class."""
    return neg(self)


def _tensor_matmul(self: Tensor, other: Union[Tensor, np.ndarray]) -> Tensor:
    """Matrix multiplication method for Tensor class."""
    if not isinstance(other, Tensor):
        other = Tensor(other, requires_grad=False)
    return matmul(self, other)


def _tensor_pow(self: Tensor, n: Union[int, float]) -> Tensor:
    """Power method for Tensor class."""
    return pow(self, n)


def _tensor_sum(self: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None,
                keepdims: bool = False) -> Tensor:
    """Sum method for Tensor class."""
    return tensor_sum(self, axis=axis, keepdims=keepdims)


def _tensor_mean(self: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None,
                 keepdims: bool = False) -> Tensor:
    """Mean method for Tensor class."""
    return mean(self, axis=axis, keepdims=keepdims)


Tensor.__add__ = _tensor_add
Tensor.__radd__ = _tensor_radd
Tensor.__sub__ = _tensor_sub
Tensor.__rsub__ = _tensor_rsub
Tensor.__mul__ = _tensor_mul
Tensor.__rmul__ = _tensor_rmul
Tensor.__truediv__ = _tensor_truediv
Tensor.__rtruediv__ = _tensor_rtruediv
Tensor.__neg__ = _tensor_neg
Tensor.__matmul__ = _tensor_matmul
Tensor.__pow__ = _tensor_pow

Tensor.sum = _tensor_sum
Tensor.mean = _tensor_mean
