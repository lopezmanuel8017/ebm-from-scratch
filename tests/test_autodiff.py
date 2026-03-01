"""
Comprehensive tests for the Tensor class and automatic differentiation.

These tests verify:
1. Tensor initialization and basic properties
2. Backward propagation and gradient computation
3. Topological sorting of computational graph
4. Gradient accumulation when tensors are reused
5. Shape operations (reshape, transpose, flatten)
6. Edge cases and error handling
"""

import pytest
import numpy as np

from ebm.core.autodiff import Tensor


class TestTensorInitialization:
    """Tests for Tensor creation and basic attributes."""

    def test_init_from_list(self):
        """Tensor can be created from a Python list."""
        t = Tensor([1, 2, 3])
        assert t.shape == (3,)
        assert np.allclose(t.data, [1.0, 2.0, 3.0])
        assert t.dtype == np.float64

    def test_init_from_nested_list(self):
        """Tensor can be created from nested lists (2D)."""
        t = Tensor([[1, 2], [3, 4]])
        assert t.shape == (2, 2)
        assert np.allclose(t.data, [[1.0, 2.0], [3.0, 4.0]])

    def test_init_from_numpy(self):
        """Tensor can be created from NumPy array."""
        arr = np.array([1.0, 2.0, 3.0])
        t = Tensor(arr)
        assert t.shape == (3,)
        assert np.allclose(t.data, arr)

    def test_init_from_scalar(self):
        """Tensor can be created from a scalar value."""
        t = Tensor(5.0)
        assert t.shape == ()
        assert t.data == 5.0

    def test_init_from_int(self):
        """Tensor converts integers to float64."""
        t = Tensor(5)
        assert t.dtype == np.float64
        assert t.data == 5.0

    def test_requires_grad_default_false(self):
        """requires_grad defaults to False."""
        t = Tensor([1, 2, 3])
        assert t.requires_grad is False

    def test_requires_grad_true(self):
        """requires_grad can be set to True."""
        t = Tensor([1, 2, 3], requires_grad=True)
        assert t.requires_grad is True

    def test_grad_initially_none(self):
        """Gradient is None before backward pass."""
        t = Tensor([1, 2, 3], requires_grad=True)
        assert t.grad is None

    def test_prev_initially_empty(self):
        """_prev is empty set for leaf tensors."""
        t = Tensor([1, 2, 3], requires_grad=True)
        assert len(t._prev) == 0


class TestTensorProperties:
    """Tests for Tensor properties."""

    def test_shape_1d(self, simple_tensor):
        """Shape property works for 1D tensor."""
        assert simple_tensor.shape == (3,)

    def test_shape_2d(self, matrix_tensor):
        """Shape property works for 2D tensor."""
        assert matrix_tensor.shape == (3, 2)

    def test_ndim(self, matrix_tensor):
        """ndim returns number of dimensions."""
        assert matrix_tensor.ndim == 2

    def test_size(self, matrix_tensor):
        """size returns total number of elements."""
        assert matrix_tensor.size == 6

    def test_dtype(self, simple_tensor):
        """dtype returns float64."""
        assert simple_tensor.dtype == np.float64


class TestTensorStringRepresentation:
    """Tests for __repr__ and __str__."""

    def test_repr_without_grad(self):
        """__repr__ shows data and requires_grad."""
        t = Tensor([1.0, 2.0])
        assert "Tensor" in repr(t)
        assert "requires_grad=False" in repr(t)

    def test_repr_with_requires_grad(self):
        """__repr__ shows requires_grad=True."""
        t = Tensor([1.0, 2.0], requires_grad=True)
        assert "requires_grad=True" in repr(t)
        assert "grad=None" in repr(t)

    def test_repr_with_grad_computed(self):
        """__repr__ shows gradient when it exists."""
        t = Tensor([1.0, 2.0], requires_grad=True)
        t.backward(np.array([1.0, 1.0]))
        repr_str = repr(t)
        assert "requires_grad=True" in repr_str
        assert "grad=" in repr_str
        assert "grad=None" not in repr_str

    def test_str(self):
        """__str__ is concise."""
        t = Tensor([1.0, 2.0])
        assert "Tensor" in str(t)
        assert "requires_grad" not in str(t)


class TestTensorItem:
    """Tests for item() method."""

    def test_item_scalar(self, scalar_tensor):
        """item() works on scalar tensor."""
        assert scalar_tensor.item() == 5.0

    def test_item_single_element_array(self):
        """item() works on tensor with single element."""
        t = Tensor([42.0])
        assert t.item() == 42.0

    def test_item_fails_on_multi_element(self, simple_tensor):
        """item() raises ValueError for multi-element tensors."""
        with pytest.raises(ValueError):
            simple_tensor.item()


class TestTensorNumpy:
    """Tests for numpy() method."""

    def test_numpy_returns_copy(self, simple_tensor):
        """numpy() returns a copy, not a view."""
        arr = simple_tensor.numpy()
        arr[0] = 999.0
        assert simple_tensor.data[0] == 1.0


class TestTensorDetach:
    """Tests for detach() method."""

    def test_detach_returns_new_tensor(self, simple_tensor):
        """detach() returns a new Tensor."""
        detached = simple_tensor.detach()
        assert detached is not simple_tensor

    def test_detach_copies_data(self, simple_tensor):
        """detach() copies the data."""
        detached = simple_tensor.detach()
        assert np.allclose(detached.data, simple_tensor.data)

    def test_detach_no_requires_grad(self, simple_tensor):
        """detached tensor has requires_grad=False."""
        detached = simple_tensor.detach()
        assert detached.requires_grad is False

    def test_detach_independent_data(self, simple_tensor):
        """Modifying detached tensor doesn't affect original."""
        detached = simple_tensor.detach()
        detached.data[0] = 999.0
        assert simple_tensor.data[0] == 1.0


class TestTensorCopy:
    """Tests for copy() method."""

    def test_copy_returns_new_tensor(self, simple_tensor):
        """copy() returns a new Tensor."""
        copied = simple_tensor.copy()
        assert copied is not simple_tensor

    def test_copy_preserves_requires_grad(self, simple_tensor):
        """copy() preserves requires_grad setting."""
        copied = simple_tensor.copy()
        assert copied.requires_grad == simple_tensor.requires_grad

    def test_copy_independent_data(self, simple_tensor):
        """Modifying copied tensor doesn't affect original."""
        copied = simple_tensor.copy()
        copied.data[0] = 999.0
        assert simple_tensor.data[0] == 1.0


class TestTensorZeroGrad:
    """Tests for zero_grad() and zero_grad_recursive()."""

    def test_zero_grad_clears_gradient(self):
        """zero_grad() sets gradient to None."""
        t = Tensor([1.0, 2.0], requires_grad=True)
        t.grad = np.array([0.5, 0.5])
        t.zero_grad()
        assert t.grad is None

    def test_zero_grad_on_none_gradient(self):
        """zero_grad() works when gradient is already None."""
        t = Tensor([1.0, 2.0], requires_grad=True)
        t.zero_grad()
        assert t.grad is None


class TestBackwardBasic:
    """Basic tests for backward propagation."""

    def test_backward_on_scalar_no_gradient_arg(self):
        """backward() on scalar tensor uses gradient of 1."""
        t = Tensor(5.0, requires_grad=True)
        t.backward()
        assert t.grad is not None
        assert np.allclose(t.grad, 1.0)

    def test_backward_on_scalar_with_gradient_arg(self):
        """backward() can accept external gradient."""
        t = Tensor(5.0, requires_grad=True)
        t.backward(np.array(2.0))
        assert np.allclose(t.grad, 2.0)

    def test_backward_requires_scalar_or_gradient(self):
        """backward() on non-scalar without gradient raises error."""
        t = Tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(RuntimeError):
            t.backward()

    def test_backward_with_explicit_gradient(self):
        """backward() works on non-scalar with explicit gradient."""
        t = Tensor([1.0, 2.0], requires_grad=True)
        t.backward(np.array([1.0, 1.0]))
        assert np.allclose(t.grad, [1.0, 1.0])

    def test_gradient_accumulates(self):
        """Gradients accumulate when backward is called multiple times."""
        t = Tensor(5.0, requires_grad=True)
        t.backward()
        t.backward()
        assert np.allclose(t.grad, 2.0)


class TestReshape:
    """Tests for reshape operation."""

    def test_reshape_basic(self, matrix_tensor):
        """reshape changes the shape."""
        reshaped = matrix_tensor.reshape(6)
        assert reshaped.shape == (6,)
        assert np.allclose(reshaped.data, [1, 2, 3, 4, 5, 6])

    def test_reshape_2d_to_3d(self):
        """reshape can change number of dimensions."""
        t = Tensor(np.arange(12).reshape(3, 4), requires_grad=True)
        reshaped = t.reshape(2, 2, 3)
        assert reshaped.shape == (2, 2, 3)

    def test_reshape_preserves_elements(self, matrix_tensor):
        """reshape preserves all elements."""
        reshaped = matrix_tensor.reshape(2, 3)
        assert reshaped.size == matrix_tensor.size

    def test_reshape_backward(self, matrix_tensor):
        """reshape propagates gradients correctly."""
        reshaped = matrix_tensor.reshape(6)
        reshaped.backward(np.ones(6))
        assert matrix_tensor.grad is not None
        assert matrix_tensor.grad.shape == (3, 2)
        assert np.allclose(matrix_tensor.grad, 1.0)

    def test_reshape_with_negative_one(self):
        """reshape supports -1 for inferred dimension."""
        t = Tensor(np.arange(12), requires_grad=True)
        reshaped = t.reshape(3, -1)
        assert reshaped.shape == (3, 4)

    def test_reshape_tuple_arg(self):
        """reshape accepts tuple as single argument."""
        t = Tensor(np.arange(12), requires_grad=True)
        reshaped = t.reshape((3, 4))
        assert reshaped.shape == (3, 4)


class TestFlatten:
    """Tests for flatten operation."""

    def test_flatten_2d(self, matrix_tensor):
        """flatten converts 2D to 1D."""
        flat = matrix_tensor.flatten()
        assert flat.shape == (6,)

    def test_flatten_3d(self, random_tensor_3d):
        """flatten converts 3D to 1D."""
        flat = random_tensor_3d.flatten()
        assert flat.shape == (24,)

    def test_flatten_backward(self, matrix_tensor):
        """flatten propagates gradients correctly."""
        flat = matrix_tensor.flatten()
        flat.backward(np.ones(6))
        assert matrix_tensor.grad is not None
        assert matrix_tensor.grad.shape == (3, 2)


class TestTranspose:
    """Tests for transpose operation."""

    def test_transpose_2d(self, matrix_tensor):
        """transpose swaps dimensions for 2D."""
        transposed = matrix_tensor.transpose()
        assert transposed.shape == (2, 3)

    def test_transpose_T_property(self, matrix_tensor):
        """T property is equivalent to transpose()."""
        assert np.allclose(matrix_tensor.T.data, matrix_tensor.transpose().data)

    def test_transpose_specific_axes(self, random_tensor_3d):
        """transpose can reorder axes specifically."""
        transposed = random_tensor_3d.transpose(2, 0, 1)
        assert transposed.shape == (4, 2, 3)

    def test_transpose_with_tuple_arg(self, random_tensor_3d):
        """transpose accepts axes as a tuple."""
        transposed = random_tensor_3d.transpose((2, 0, 1))
        assert transposed.shape == (4, 2, 3)

    def test_transpose_with_list_arg(self, random_tensor_3d):
        """transpose accepts axes as a list."""
        transposed = random_tensor_3d.transpose([2, 0, 1])
        assert transposed.shape == (4, 2, 3)

    def test_transpose_backward(self, matrix_tensor):
        """transpose propagates gradients correctly."""
        transposed = matrix_tensor.T
        transposed.backward(np.ones((2, 3)))
        assert matrix_tensor.grad is not None
        assert matrix_tensor.grad.shape == (3, 2)
        assert np.allclose(matrix_tensor.grad, 1.0)

    def test_transpose_backward_with_axes(self, random_tensor_3d):
        """transpose with specific axes propagates gradients correctly."""
        transposed = random_tensor_3d.transpose(2, 0, 1)
        transposed.backward(np.ones((4, 2, 3)))
        assert random_tensor_3d.grad is not None
        assert random_tensor_3d.grad.shape == (2, 3, 4)


class TestComparisons:
    """Tests for comparison operations."""

    def test_eq(self, simple_tensor):
        """Element-wise equality comparison."""
        result = simple_tensor == 2.0
        assert result.sum() == 1

    def test_ne(self, simple_tensor):
        """Element-wise inequality comparison."""
        result = simple_tensor != 2.0
        assert result.sum() == 2

    def test_lt(self, simple_tensor):
        """Element-wise less-than comparison."""
        result = simple_tensor < 2.0
        assert result.sum() == 1

    def test_le(self, simple_tensor):
        """Element-wise less-than-or-equal comparison."""
        result = simple_tensor <= 2.0
        assert result.sum() == 2

    def test_gt(self, simple_tensor):
        """Element-wise greater-than comparison."""
        result = simple_tensor > 2.0
        assert result.sum() == 1

    def test_ge(self, simple_tensor):
        """Element-wise greater-than-or-equal comparison."""
        result = simple_tensor >= 2.0
        assert result.sum() == 2

    def test_comparison_with_tensor(self, simple_tensor):
        """Comparisons work between tensors."""
        other = Tensor([1.0, 2.0, 4.0])
        result = simple_tensor == other
        assert np.allclose(result, [True, True, False])


class TestComputationalGraph:
    """Tests for computational graph structure."""

    def test_leaf_tensor_empty_prev(self, simple_tensor):
        """Leaf tensors have empty _prev set."""
        assert len(simple_tensor._prev) == 0

    def test_operation_creates_prev_link(self, simple_tensor):
        """Operations create links in _prev."""
        reshaped = simple_tensor.reshape(1, 3)
        assert simple_tensor in reshaped._prev


class TestGradientFlow:
    """Tests for gradient flow through computational graph."""

    def test_no_grad_when_requires_grad_false(self):
        """Gradients are not computed when requires_grad=False."""
        t = Tensor([1.0, 2.0, 3.0], requires_grad=False)
        reshaped = t.reshape(1, 3)
        reshaped.backward(np.ones((1, 3)))
        assert t.grad is None

    def test_chain_of_operations(self):
        """Gradients flow through chain of operations."""
        t = Tensor(np.arange(6), requires_grad=True)
        a = t.reshape(2, 3)
        b = a.T
        b.backward(np.ones((3, 2)))
        assert t.grad is not None
        assert t.grad.shape == (6,)


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_empty_array(self):
        """Tensor can be created from empty array."""
        t = Tensor([])
        assert t.shape == (0,)

    def test_very_large_values(self):
        """Tensor handles large values."""
        t = Tensor([1e100, 2e100, 3e100], requires_grad=True)
        assert np.isfinite(t.data).all()

    def test_very_small_values(self):
        """Tensor handles small values."""
        t = Tensor([1e-100, 2e-100, 3e-100], requires_grad=True)
        assert np.isfinite(t.data).all()

    def test_mixed_positive_negative(self):
        """Tensor handles mixed positive and negative values."""
        t = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        reshaped = t.reshape(1, 3)
        reshaped.backward(np.ones((1, 3)))
        assert np.allclose(t.grad, 1.0)


class TestMultipleBackwardCalls:
    """Tests for calling backward multiple times."""

    def test_gradients_accumulate_on_leaf(self):
        """Calling backward multiple times accumulates gradients on leaf."""
        t = Tensor([1.0, 2.0], requires_grad=True)

        reshaped1 = t.reshape(1, 2)
        reshaped1.backward(np.ones((1, 2)))
        first_grad = t.grad.copy()

        reshaped2 = t.reshape(1, 2)
        reshaped2.backward(np.ones((1, 2)))

        assert np.allclose(t.grad, first_grad * 2)

    def test_zero_grad_then_backward(self):
        """zero_grad followed by backward gives fresh gradients."""
        t = Tensor([1.0, 2.0], requires_grad=True)

        reshaped1 = t.reshape(1, 2)
        reshaped1.backward(np.ones((1, 2)))

        t.zero_grad()

        reshaped2 = t.reshape(1, 2)
        reshaped2.backward(np.ones((1, 2)))

        assert np.allclose(t.grad, 1.0)

    def test_zero_grad_recursive_clears_all(self):
        """zero_grad_recursive clears gradients on entire graph."""
        t = Tensor([1.0, 2.0], requires_grad=True)
        reshaped = t.reshape(1, 2)

        reshaped.backward(np.ones((1, 2)))

        assert t.grad is not None
        assert reshaped.grad is not None

        reshaped.zero_grad_recursive()

        assert t.grad is None
        assert reshaped.grad is None


class TestComplexGraphs:
    """Tests for more complex computational graphs."""

    def test_sequential_operations(self):
        """Gradients flow through sequential operations."""
        t = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)

        a = t.transpose(1, 2, 0)
        b = a.reshape(6, 4)
        c = b.flatten()
        d = c.reshape(4, 6)
        e = d.T

        e.backward(np.ones((6, 4)))

        assert t.grad is not None
        assert t.grad.shape == (2, 3, 4)
        assert np.allclose(t.grad, 1.0)


class TestNumericalGradientChecks:
    """
    Tests that verify gradients against finite differences.

    These tests are crucial for validating that the backward pass
    computes correct gradients by comparing against numerical approximation.
    """

    def numerical_gradient(self, func, x, eps=1e-5):
        """
        Compute numerical gradient using central differences.

        Args:
            func: Function that takes numpy array and returns scalar
            x: Point at which to compute gradient
            eps: Finite difference step size

        Returns:
            Numerical gradient approximation
        """
        grad = np.zeros_like(x)
        for i in range(x.size):
            x_plus = x.copy()
            x_plus.flat[i] += eps
            x_minus = x.copy()
            x_minus.flat[i] -= eps
            grad.flat[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
        return grad

    def test_reshape_gradient_numerical_check(self):
        """Verify reshape gradient against finite differences."""
        x_data = np.random.randn(3, 4)

        def func(x):
            t = Tensor(x, requires_grad=True)
            y = t.reshape(2, 6)
            return (y.data ** 2).sum()

        numerical_grad = self.numerical_gradient(func, x_data)

        t = Tensor(x_data, requires_grad=True)
        y = t.reshape(2, 6)
        y.backward(2 * y.data)

        assert np.allclose(t.grad, numerical_grad, rtol=1e-4, atol=1e-4)

    def test_transpose_gradient_numerical_check(self):
        """Verify transpose gradient against finite differences."""
        x_data = np.random.randn(3, 4)

        def func(x):
            t = Tensor(x, requires_grad=True)
            y = t.T
            return (y.data ** 2).sum()

        numerical_grad = self.numerical_gradient(func, x_data)

        t = Tensor(x_data, requires_grad=True)
        y = t.T
        y.backward(2 * y.data)

        assert np.allclose(t.grad, numerical_grad, rtol=1e-4, atol=1e-4)

    def test_flatten_gradient_numerical_check(self):
        """Verify flatten gradient against finite differences."""
        x_data = np.random.randn(2, 3, 4)

        def func(x):
            t = Tensor(x, requires_grad=True)
            y = t.flatten()
            return (y.data ** 2).sum()

        numerical_grad = self.numerical_gradient(func, x_data)

        t = Tensor(x_data, requires_grad=True)
        y = t.flatten()
        y.backward(2 * y.data)

        assert np.allclose(t.grad, numerical_grad, rtol=1e-4, atol=1e-4)

    def test_chained_operations_gradient_numerical_check(self):
        """Verify gradient through chain of operations against finite differences."""
        x_data = np.random.randn(2, 3, 4)

        def func(x):
            t = Tensor(x, requires_grad=True)
            y = t.transpose(2, 0, 1)
            z = y.reshape(4, 6)
            w = z.flatten()
            return (w.data ** 2).sum()

        numerical_grad = self.numerical_gradient(func, x_data)

        t = Tensor(x_data, requires_grad=True)
        y = t.transpose(2, 0, 1)
        z = y.reshape(4, 6)
        w = z.flatten()
        w.backward(2 * w.data)

        assert np.allclose(t.grad, numerical_grad, rtol=1e-4, atol=1e-4)


class TestSharedNodes:
    """
    Tests for when a tensor is used multiple times in the graph.

    This is important because gradients must accumulate correctly when
    a tensor contributes to the output through multiple paths.
    """

    def test_tensor_used_twice_in_reshape(self):
        """Test gradient when tensor is used in two different reshapes."""
        t = Tensor(np.random.randn(6), requires_grad=True)

        a = t.reshape(2, 3)
        b = t.reshape(3, 2)

        _loss = a.data.sum() + b.data.sum()

        a.backward(np.ones((2, 3)))
        b.backward(np.ones((3, 2)))

        assert np.allclose(t.grad, 2.0)

    def test_diamond_pattern(self):
        """Test gradient flow in diamond-shaped graph."""
        t = Tensor(np.random.randn(4), requires_grad=True)

        a = t.reshape(2, 2)
        b = t.reshape(1, 4)

        a.backward(np.ones((2, 2)))
        b.backward(np.ones((1, 4)))

        assert np.allclose(t.grad, 2.0)


class TestGradientShapes:
    """Tests ensuring gradient shapes match data shapes."""

    @pytest.mark.parametrize("shape", [
        (5,),
        (3, 4),
        (2, 3, 4),
        (2, 2, 2, 2),
    ])
    def test_gradient_shape_matches_data(self, shape):
        """Gradient shape should always match data shape."""
        t = Tensor(np.random.randn(*shape), requires_grad=True)
        flat = t.flatten()
        flat.backward(np.ones(flat.shape))

        assert t.grad.shape == t.data.shape

    @pytest.mark.parametrize("orig_shape,new_shape", [
        ((12,), (3, 4)),
        ((3, 4), (12,)),
        ((2, 3, 4), (6, 4)),
        ((6, 4), (2, 3, 4)),
    ])
    def test_gradient_shape_after_reshape(self, orig_shape, new_shape):
        """Gradient should have original shape after reshape backward."""
        t = Tensor(np.random.randn(*orig_shape), requires_grad=True)
        reshaped = t.reshape(*new_shape)
        reshaped.backward(np.ones(new_shape))

        assert t.grad.shape == orig_shape


class TestSpecialCases:
    """Tests for special and boundary cases."""

    def test_scalar_to_1d_reshape(self):
        """Reshape scalar to 1D array."""
        t = Tensor(5.0, requires_grad=True)
        reshaped = t.reshape(1)
        reshaped.backward(np.array([2.0]))

        assert t.grad.shape == ()
        assert np.allclose(t.grad, 2.0)

    def test_1d_to_scalar_reshape(self):
        """Reshape 1D array to scalar-like shape."""
        t = Tensor([5.0], requires_grad=True)
        reshaped = t.reshape(())
        reshaped.backward(np.array(2.0))

        assert t.grad.shape == (1,)
        assert np.allclose(t.grad, 2.0)

    def test_high_dimensional_transpose(self):
        """Test transpose with many dimensions."""
        t = Tensor(np.random.randn(2, 3, 4, 5), requires_grad=True)
        transposed = t.transpose(3, 1, 2, 0)

        assert transposed.shape == (5, 3, 4, 2)

        transposed.backward(np.ones((5, 3, 4, 2)))
        assert t.grad.shape == (2, 3, 4, 5)
        assert np.allclose(t.grad, 1.0)

    def test_identity_reshape(self):
        """Reshaping to same shape should work correctly."""
        t = Tensor(np.random.randn(3, 4), requires_grad=True)
        reshaped = t.reshape(3, 4)

        assert reshaped.shape == t.shape
        assert np.allclose(reshaped.data, t.data)

        reshaped.backward(np.ones((3, 4)) * 2.0)
        assert np.allclose(t.grad, 2.0)

    def test_identity_transpose(self):
        """Transposing with identity permutation should work correctly."""
        t = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        transposed = t.transpose(0, 1, 2)

        assert transposed.shape == t.shape
        assert np.allclose(transposed.data, t.data)

        transposed.backward(np.ones((2, 3, 4)) * 3.0)
        assert np.allclose(t.grad, 3.0)
