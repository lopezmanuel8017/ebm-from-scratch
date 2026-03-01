"""
Tensor class for automatic differentiation.

This module implements a Tensor class that supports reverse-mode automatic
differentiation (backpropagation). Each Tensor maintains:
- data: the actual numerical values (NumPy array)
- grad: accumulated gradients (same shape as data)
- requires_grad: whether to compute gradients for this tensor
- _backward: function to propagate gradients to parent tensors
- _prev: set of parent tensors in the computational graph
"""

import numpy as np
from typing import Callable, Optional, Set, Tuple, Union


class Tensor:
    """
    A tensor that supports automatic differentiation.

    The Tensor class forms the foundation of the autodiff engine. It wraps
    NumPy arrays and tracks operations to build a computational graph.
    When backward() is called, gradients are propagated through the graph
    using reverse-mode autodiff.

    Attributes:
        data: NumPy array holding the value.
        grad: NumPy array holding the accumulated gradient (same shape as data).
        requires_grad: Whether to compute gradients for this tensor.
        _backward: Function to compute gradients for this node.
        _prev: Set of parent tensors in the computational graph.

    Example:
        >>> a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        >>> c = a + b  # Creates a new tensor with _prev = {a, b}
        >>> loss = c.sum()
        >>> loss.backward()
        >>> print(a.grad)  # [1., 1., 1.]
    """

    def __init__(
        self,
        data: Union[np.ndarray, list, float, int],
        requires_grad: bool = False,
        _prev: Optional[Set['Tensor']] = None,
        _backward: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize a Tensor.

        Args:
            data: The numerical data. Will be converted to a NumPy array.
            requires_grad: If True, gradients will be computed for this tensor.
            _prev: Set of parent tensors (used internally during graph construction).
            _backward: Function to propagate gradients (used internally).
        """
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        else:
            self.data = np.array(data, dtype=np.float64)

        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None

        self._prev: Set['Tensor'] = _prev if _prev is not None else set()
        self._backward: Callable[[], None] = _backward if _backward is not None else lambda: None

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor data."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Return the total number of elements."""
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        """Return the data type of the tensor."""
        return self.data.dtype

    def backward(self, gradient: Optional[np.ndarray] = None) -> None:
        """
        Perform reverse-mode automatic differentiation from this tensor.

        Computes gradients for all tensors in the computational graph that:
        1. Have requires_grad=True
        2. Are ancestors of this tensor in the graph

        The algorithm:
        1. Build a topological ordering of the graph (from leaves to this tensor)
        2. Initialize this tensor's gradient to 1 (or the provided gradient)
        3. Traverse in reverse topological order, calling _backward on each node

        Args:
            gradient: External gradient to seed the backward pass. If None,
                     assumes this tensor is a scalar and uses gradient of 1.

        Raises:
            RuntimeError: If backward is called on a non-scalar tensor without
                         providing an explicit gradient.
        """
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() can only be called on scalar tensors or with "
                    f"an explicit gradient. Got tensor with shape {self.shape}"
                )
            gradient = np.ones_like(self.data)

        if self.grad is None:
            self.grad = gradient
        else:
            self.grad = self.grad + gradient

        topo_order = []
        visited = set()

        def build_topo(tensor: 'Tensor') -> None:
            """Build topological ordering via DFS."""
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    build_topo(parent)
                topo_order.append(tensor)

        build_topo(self)

        for tensor in reversed(topo_order):
            tensor._backward()

    def zero_grad(self) -> None:
        """
        Reset the gradient to None.

        This should be called before each backward pass to avoid
        accumulating gradients from previous iterations.
        """
        self.grad = None

    def zero_grad_recursive(self) -> None:
        """
        Recursively zero gradients for this tensor and all parents.

        Useful for clearing the entire computational graph's gradients.
        """
        visited = set()

        def _zero(tensor: 'Tensor') -> None:
            """Inner recursive function to zero gradients."""
            if tensor not in visited:
                visited.add(tensor)
                tensor.grad = None
                for parent in tensor._prev:
                    _zero(parent)

        _zero(self)

    def detach(self) -> 'Tensor':
        """
        Return a new tensor detached from the computational graph.

        The detached tensor shares the same data but has no gradient
        tracking. Useful for stopping gradient flow.

        Returns:
            A new Tensor with the same data but requires_grad=False.
        """
        return Tensor(self.data.copy(), requires_grad=False)

    def numpy(self) -> np.ndarray:
        """
        Return the underlying NumPy array.

        Returns:
            A copy of the tensor's data as a NumPy array.
        """
        return self.data.copy()

    def item(self) -> float:
        """
        Return the scalar value if the tensor has a single element.

        Returns:
            The scalar value.

        Raises:
            ValueError: If the tensor has more than one element.
        """
        if self.data.size != 1:
            raise ValueError(
                f"item() only works on tensors with one element, got {self.size}"
            )
        return float(self.data.flat[0])

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        grad_str = ""
        if self.requires_grad:
            if self.grad is not None:
                grad_str = f", grad={self.grad}"
            else:
                grad_str = ", grad=None"
        return f"Tensor({self.data}, requires_grad={self.requires_grad}{grad_str})"

    def __str__(self) -> str:
        """Return a concise string representation."""
        return f"Tensor({self.data})"

    def __hash__(self) -> int:
        """
        Return hash based on object identity.

        This is needed because we override __eq__ for element-wise comparison,
        which would normally make the class unhashable. We use id() to allow
        Tensors to be used in sets (for the computational graph).
        """
        return id(self)

    def reshape(self, *shape: int) -> 'Tensor':
        """
        Reshape the tensor to the given shape.

        Args:
            *shape: The new shape dimensions.

        Returns:
            A new tensor with the reshaped data.
        """
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        result_data = self.data.reshape(new_shape)
        result = Tensor(result_data, requires_grad=self.requires_grad, _prev={self})

        if self.requires_grad:
            def _backward():
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + result.grad.reshape(self.data.shape)
            result._backward = _backward

        return result

    def flatten(self) -> 'Tensor':
        """
        Flatten the tensor to 1D.

        Returns:
            A new 1D tensor.
        """
        return self.reshape(-1)

    def transpose(self, *axes: int) -> 'Tensor':
        """
        Transpose the tensor.

        Args:
            *axes: The new axis order. If not provided, reverses all axes.

        Returns:
            A new transposed tensor.
        """
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])

        result_data = self.data.transpose(axes)
        result = Tensor(result_data, requires_grad=self.requires_grad, _prev={self})

        if self.requires_grad:
            def _backward():
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                if axes is None:
                    inv_axes = None
                else:
                    inv_axes = tuple(np.argsort(axes))
                self.grad = self.grad + result.grad.transpose(inv_axes)
            result._backward = _backward

        return result

    @property
    def T(self) -> 'Tensor':
        """Return the transposed tensor (reverses all axes)."""
        return self.transpose()

    def __eq__(self, other: Union['Tensor', np.ndarray, float, int]) -> np.ndarray:
        """Element-wise equality comparison."""
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data == other_data

    def __ne__(self, other: Union['Tensor', np.ndarray, float, int]) -> np.ndarray:
        """Element-wise inequality comparison."""
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data != other_data

    def __lt__(self, other: Union['Tensor', np.ndarray, float, int]) -> np.ndarray:
        """Element-wise less-than comparison."""
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data < other_data

    def __le__(self, other: Union['Tensor', np.ndarray, float, int]) -> np.ndarray:
        """Element-wise less-than-or-equal comparison."""
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data <= other_data

    def __gt__(self, other: Union['Tensor', np.ndarray, float, int]) -> np.ndarray:
        """Element-wise greater-than comparison."""
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data > other_data

    def __ge__(self, other: Union['Tensor', np.ndarray, float, int]) -> np.ndarray:
        """Element-wise greater-than-or-equal comparison."""
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data >= other_data

    def copy(self) -> 'Tensor':
        """
        Create a copy of the tensor.

        Returns:
            A new Tensor with copied data but same requires_grad setting.
        """
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)
