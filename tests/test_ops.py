"""
Comprehensive tests for differentiable operations.

These tests verify:
1. Forward pass correctness
2. Backward pass gradient computation
3. Numerical gradient verification via finite differences
4. Broadcasting gradient handling
5. Numerical stability
6. Edge cases
"""

import pytest
import numpy as np

from ebm.core.autodiff import Tensor
from ebm.core.ops import (
    add, sub, mul, div, neg,
    matmul, transpose,
    tensor_sum, mean,
    relu, sigmoid, softplus, swish,
    exp, log, pow, sqrt,
    unbroadcast,
)


def numerical_gradient(func, x, eps=1e-5):
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


def check_gradient(op_func, inputs, eps=1e-5, rtol=1e-4, atol=1e-4):
    """
    Verify gradient against finite differences.

    Args:
        op_func: Function that takes tensors and returns output tensor
        inputs: List of input numpy arrays
        eps: Finite difference step size
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if gradients match
    """
    tensors = [Tensor(x, requires_grad=True) for x in inputs]
    output = op_func(*tensors)
    _loss = output.data.sum()

    output.backward(np.ones_like(output.data))

    for i, (x, t) in enumerate(zip(inputs, tensors)):
        def func(x_val):
            test_inputs = [Tensor(inp.copy(), requires_grad=False) for inp in inputs]
            test_inputs[i] = Tensor(x_val, requires_grad=False)
            return op_func(*test_inputs).data.sum()

        num_grad = numerical_gradient(func, x, eps)
        if not np.allclose(t.grad, num_grad, rtol=rtol, atol=atol):
            return False

    return True


class TestUnbroadcast:
    """Tests for the unbroadcast utility function."""

    def test_no_broadcast(self):
        """No broadcasting needed when shapes match."""
        grad = np.random.randn(3, 4)
        result = unbroadcast(grad, (3, 4))
        assert result.shape == (3, 4)
        assert np.allclose(result, grad)

    def test_broadcast_leading_dims(self):
        """Handle added leading dimensions."""
        grad = np.random.randn(2, 3, 4)
        result = unbroadcast(grad, (4,))
        assert result.shape == (4,)

    def test_broadcast_size_one_dims(self):
        """Handle size-1 dimensions that got broadcast."""
        grad = np.random.randn(3, 4)
        result = unbroadcast(grad, (1, 4))
        assert result.shape == (1, 4)

    def test_broadcast_combined(self):
        """Handle both leading dims and size-1 dims."""
        grad = np.random.randn(2, 3, 4)
        result = unbroadcast(grad, (1, 4))
        assert result.shape == (1, 4)

    def test_scalar_broadcast(self):
        """Handle scalar original shape."""
        grad = np.random.randn(3, 4)
        result = unbroadcast(grad, ())
        assert result.shape == ()


class TestAdd:
    """Tests for add operation."""

    def test_forward_same_shape(self):
        """Add tensors with same shape."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = add(a, b)
        assert np.allclose(c.data, [5.0, 7.0, 9.0])

    def test_forward_broadcast(self):
        """Add with broadcasting."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([10.0, 20.0])
        c = add(a, b)
        assert np.allclose(c.data, [[11.0, 22.0], [13.0, 24.0]])

    def test_backward_same_shape(self):
        """Gradients for same shape add."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        c = add(a, b)
        c.backward(np.array([1.0, 1.0, 1.0]))

        assert np.allclose(a.grad, [1.0, 1.0, 1.0])
        assert np.allclose(b.grad, [1.0, 1.0, 1.0])

    def test_backward_broadcast(self):
        """Gradients with broadcasting."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([10.0, 20.0], requires_grad=True)
        c = add(a, b)
        c.backward(np.ones((2, 2)))

        assert a.grad.shape == (2, 2)
        assert b.grad.shape == (2,)
        assert np.allclose(a.grad, 1.0)
        assert np.allclose(b.grad, 2.0)

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(add, [np.random.randn(3, 4), np.random.randn(3, 4)])

    def test_numerical_gradient_broadcast(self):
        """Verify gradient with broadcasting."""
        assert check_gradient(add, [np.random.randn(3, 4), np.random.randn(4,)])

    def test_operator_overload(self):
        """Test + operator."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        c = a + b
        c.backward(np.ones(2))
        assert np.allclose(c.data, [4.0, 6.0])
        assert np.allclose(a.grad, 1.0)

    def test_add_scalar(self):
        """Test adding scalar."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = a + 5.0
        c.backward(np.ones(3))
        assert np.allclose(c.data, [6.0, 7.0, 8.0])
        assert np.allclose(a.grad, 1.0)

    def test_radd_scalar(self):
        """Test reverse add with scalar."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = 5.0 + a
        c.backward(np.ones(3))
        assert np.allclose(c.data, [6.0, 7.0, 8.0])
        assert np.allclose(a.grad, 1.0)


class TestSub:
    """Tests for subtraction operation."""

    def test_forward(self):
        """Basic subtraction."""
        a = Tensor([5.0, 6.0, 7.0])
        b = Tensor([1.0, 2.0, 3.0])
        c = sub(a, b)
        assert np.allclose(c.data, [4.0, 4.0, 4.0])

    def test_backward(self):
        """Gradients for subtraction."""
        a = Tensor([5.0, 6.0, 7.0], requires_grad=True)
        b = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = sub(a, b)
        c.backward(np.ones(3))

        assert np.allclose(a.grad, 1.0)
        assert np.allclose(b.grad, -1.0)

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(sub, [np.random.randn(3, 4), np.random.randn(3, 4)])

    def test_operator_overload(self):
        """Test - operator."""
        a = Tensor([5.0, 6.0], requires_grad=True)
        b = Tensor([1.0, 2.0], requires_grad=True)
        c = a - b
        assert np.allclose(c.data, [4.0, 4.0])

    def test_rsub_scalar(self):
        """Test reverse sub with scalar."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = 10.0 - a
        c.backward(np.ones(3))
        assert np.allclose(c.data, [9.0, 8.0, 7.0])
        assert np.allclose(a.grad, -1.0)


class TestMul:
    """Tests for multiplication operation."""

    def test_forward(self):
        """Basic multiplication."""
        a = Tensor([2.0, 3.0, 4.0])
        b = Tensor([5.0, 6.0, 7.0])
        c = mul(a, b)
        assert np.allclose(c.data, [10.0, 18.0, 28.0])

    def test_backward(self):
        """Gradients for multiplication."""
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([5.0, 6.0], requires_grad=True)
        c = mul(a, b)
        c.backward(np.ones(2))

        assert np.allclose(a.grad, [5.0, 6.0])
        assert np.allclose(b.grad, [2.0, 3.0])

    def test_backward_broadcast(self):
        """Gradients with broadcasting."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([2.0, 3.0], requires_grad=True)
        c = mul(a, b)
        c.backward(np.ones((2, 2)))

        assert a.grad.shape == (2, 2)
        assert b.grad.shape == (2,)
        assert np.allclose(a.grad, [[2.0, 3.0], [2.0, 3.0]])
        assert np.allclose(b.grad, [4.0, 6.0])

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(mul, [np.random.randn(3, 4), np.random.randn(3, 4)])

    def test_operator_overload(self):
        """Test * operator."""
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([5.0, 6.0], requires_grad=True)
        c = a * b
        assert np.allclose(c.data, [10.0, 18.0])

    def test_mul_scalar(self):
        """Test multiplying by scalar."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = a * 2.0
        c.backward(np.ones(3))
        assert np.allclose(c.data, [2.0, 4.0, 6.0])
        assert np.allclose(a.grad, 2.0)


class TestDiv:
    """Tests for division operation."""

    def test_forward(self):
        """Basic division."""
        a = Tensor([10.0, 20.0, 30.0])
        b = Tensor([2.0, 4.0, 5.0])
        c = div(a, b)
        assert np.allclose(c.data, [5.0, 5.0, 6.0])

    def test_backward(self):
        """Gradients for division."""
        a = Tensor([10.0, 20.0], requires_grad=True)
        b = Tensor([2.0, 4.0], requires_grad=True)
        c = div(a, b)
        c.backward(np.ones(2))

        assert np.allclose(a.grad, [0.5, 0.25])
        assert np.allclose(b.grad, [-2.5, -1.25])

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(div, [np.random.randn(3, 4), np.abs(np.random.randn(3, 4)) + 0.5])

    def test_operator_overload(self):
        """Test / operator."""
        a = Tensor([10.0, 20.0], requires_grad=True)
        b = Tensor([2.0, 4.0], requires_grad=True)
        c = a / b
        assert np.allclose(c.data, [5.0, 5.0])

    def test_div_scalar(self):
        """Test dividing by scalar."""
        a = Tensor([10.0, 20.0, 30.0], requires_grad=True)
        c = a / 2.0
        c.backward(np.ones(3))
        assert np.allclose(c.data, [5.0, 10.0, 15.0])
        assert np.allclose(a.grad, 0.5)


class TestNeg:
    """Tests for negation operation."""

    def test_forward(self):
        """Basic negation."""
        a = Tensor([1.0, -2.0, 3.0])
        c = neg(a)
        assert np.allclose(c.data, [-1.0, 2.0, -3.0])

    def test_backward(self):
        """Gradients for negation."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = neg(a)
        c.backward(np.ones(3))
        assert np.allclose(a.grad, -1.0)

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(neg, [np.random.randn(3, 4)])

    def test_operator_overload(self):
        """Test unary - operator."""
        a = Tensor([1.0, -2.0, 3.0], requires_grad=True)
        c = -a
        assert np.allclose(c.data, [-1.0, 2.0, -3.0])


class TestMatmul:
    """Tests for matrix multiplication."""

    def test_forward_2d(self):
        """Basic 2D matmul."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = matmul(a, b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(c.data, expected)

    def test_forward_shapes(self):
        """Test matmul with various shapes."""
        a = Tensor(np.random.randn(3, 4))
        b = Tensor(np.random.randn(4, 5))
        c = matmul(a, b)
        assert c.shape == (3, 5)

    def test_backward_2d(self):
        """Gradients for 2D matmul."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4, 5), requires_grad=True)
        c = matmul(a, b)
        c.backward(np.ones((3, 5)))

        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (4, 5)

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(matmul, [np.random.randn(3, 4), np.random.randn(4, 5)])

    def test_operator_overload(self):
        """Test @ operator."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4, 5), requires_grad=True)
        c = a @ b
        assert c.shape == (3, 5)


class TestTranspose:
    """Tests for transpose operation."""

    def test_forward_2d(self):
        """Basic 2D transpose."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        c = transpose(a)
        assert c.shape == (3, 2)
        assert np.allclose(c.data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_forward_with_axes(self):
        """Transpose with specific axes."""
        a = Tensor(np.random.randn(2, 3, 4))
        c = transpose(a, (2, 0, 1))
        assert c.shape == (4, 2, 3)

    def test_backward(self):
        """Gradients for transpose."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        c = transpose(a)
        c.backward(np.ones((4, 3)))
        assert a.grad.shape == (3, 4)
        assert np.allclose(a.grad, 1.0)

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(transpose, [np.random.randn(3, 4)])


class TestSum:
    """Tests for sum operation."""

    def test_forward_all(self):
        """Sum all elements."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        c = tensor_sum(a)
        assert np.allclose(c.data, 10.0)

    def test_forward_axis(self):
        """Sum along axis."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        c = tensor_sum(a, axis=0)
        assert np.allclose(c.data, [4.0, 6.0])
        c = tensor_sum(a, axis=1)
        assert np.allclose(c.data, [3.0, 7.0])

    def test_forward_keepdims(self):
        """Sum with keepdims."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        c = tensor_sum(a, axis=0, keepdims=True)
        assert c.shape == (1, 2)
        assert np.allclose(c.data, [[4.0, 6.0]])

    def test_backward_all(self):
        """Gradients for sum all."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        c = tensor_sum(a)
        c.backward(np.array(1.0))
        assert a.grad.shape == (2, 2)
        assert np.allclose(a.grad, 1.0)

    def test_backward_axis(self):
        """Gradients for sum along axis."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        c = tensor_sum(a, axis=0)
        c.backward(np.array([1.0, 2.0]))
        assert np.allclose(a.grad, [[1.0, 2.0], [1.0, 2.0]])

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(tensor_sum, [np.random.randn(3, 4)])
        assert check_gradient(lambda x: tensor_sum(x, axis=0), [np.random.randn(3, 4)])

    def test_tensor_method(self):
        """Test .sum() method on Tensor."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        c = a.sum()
        c.backward()
        assert np.allclose(c.data, 10.0)
        assert np.allclose(a.grad, 1.0)


class TestMean:
    """Tests for mean operation."""

    def test_forward_all(self):
        """Mean of all elements."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        c = mean(a)
        assert np.allclose(c.data, 2.5)

    def test_forward_axis(self):
        """Mean along axis."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        c = mean(a, axis=0)
        assert np.allclose(c.data, [2.0, 3.0])
        c = mean(a, axis=1)
        assert np.allclose(c.data, [1.5, 3.5])

    def test_backward_all(self):
        """Gradients for mean all."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        c = mean(a)
        c.backward(np.array(1.0))
        assert a.grad.shape == (2, 2)
        assert np.allclose(a.grad, 0.25)

    def test_backward_axis(self):
        """Gradients for mean along axis."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        c = mean(a, axis=0)
        c.backward(np.array([1.0, 1.0]))
        assert np.allclose(a.grad, 0.5)

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(mean, [np.random.randn(3, 4)])
        assert check_gradient(lambda x: mean(x, axis=0), [np.random.randn(3, 4)])

    def test_tensor_method(self):
        """Test .mean() method on Tensor."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        c = a.mean()
        c.backward()
        assert np.allclose(c.data, 2.5)
        assert np.allclose(a.grad, 0.25)


class TestRelu:
    """Tests for ReLU activation."""

    def test_forward(self):
        """Basic ReLU."""
        a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        c = relu(a)
        assert np.allclose(c.data, [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_backward(self):
        """Gradients for ReLU."""
        a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        c = relu(a)
        c.backward(np.ones(5))
        assert np.allclose(a.grad, [0.0, 0.0, 0.0, 1.0, 1.0])

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        x = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        assert check_gradient(relu, [x])

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_shapes(self, shape):
        """Test ReLU with various shapes."""
        a = Tensor(np.random.randn(*shape), requires_grad=True)
        c = relu(a)
        c.backward(np.ones(shape))
        assert c.shape == shape
        assert a.grad.shape == shape


class TestSigmoid:
    """Tests for sigmoid activation."""

    def test_forward(self):
        """Basic sigmoid."""
        a = Tensor([0.0])
        c = sigmoid(a)
        assert np.allclose(c.data, 0.5)

    def test_forward_extreme_values(self):
        """Sigmoid with extreme values (numerical stability)."""
        a = Tensor([-100.0, -50.0, 0.0, 50.0, 100.0])
        c = sigmoid(a)
        assert np.isfinite(c.data).all()
        assert c.data[0] < 1e-10
        assert c.data[-1] > 1 - 1e-10

    def test_backward(self):
        """Gradients for sigmoid."""
        a = Tensor([0.0], requires_grad=True)
        c = sigmoid(a)
        c.backward(np.ones(1))
        assert np.allclose(a.grad, 0.25)

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(sigmoid, [np.random.randn(3, 4) * 2])

    def test_numerical_stability(self):
        """Test numerical stability for large inputs."""
        a = Tensor(np.array([-500.0, 500.0]), requires_grad=True)
        c = sigmoid(a)
        c.backward(np.ones(2))
        assert np.isfinite(c.data).all()
        assert np.isfinite(a.grad).all()


class TestSoftplus:
    """Tests for softplus activation."""

    def test_forward(self):
        """Basic softplus."""
        a = Tensor([0.0])
        c = softplus(a)
        assert np.allclose(c.data, np.log(2.0))

    def test_forward_extreme_values(self):
        """Softplus with extreme values (numerical stability)."""
        a = Tensor([-100.0, 0.0, 100.0])
        c = softplus(a)
        assert np.isfinite(c.data).all()
        assert c.data[0] < 1e-10
        assert np.allclose(c.data[2], 100.0)

    def test_backward(self):
        """Gradients for softplus."""
        a = Tensor([0.0], requires_grad=True)
        c = softplus(a)
        c.backward(np.ones(1))
        assert np.allclose(a.grad, 0.5)

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(softplus, [np.random.randn(3, 4) * 5])

    def test_numerical_stability(self):
        """Test numerical stability for large inputs."""
        a = Tensor(np.array([-500.0, 500.0]), requires_grad=True)
        c = softplus(a)
        c.backward(np.ones(2))
        assert np.isfinite(c.data).all()
        assert np.isfinite(a.grad).all()


class TestSwish:
    """Tests for swish activation."""

    def test_forward(self):
        """Basic swish."""
        a = Tensor([0.0])
        c = swish(a)
        assert np.allclose(c.data, 0.0)

    def test_forward_values(self):
        """Test swish for various values."""
        a = Tensor([1.0])
        c = swish(a)
        expected = 1.0 / (1 + np.exp(-1.0))
        assert np.allclose(c.data, expected)

    def test_backward(self):
        """Gradients for swish."""
        a = Tensor([0.0], requires_grad=True)
        c = swish(a)
        c.backward(np.ones(1))
        assert np.allclose(a.grad, 0.5)

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(swish, [np.random.randn(3, 4) * 2])

    def test_numerical_stability(self):
        """Test numerical stability for large inputs."""
        a = Tensor(np.array([-500.0, 500.0]), requires_grad=True)
        c = swish(a)
        c.backward(np.ones(2))
        assert np.isfinite(c.data).all()
        assert np.isfinite(a.grad).all()


class TestExp:
    """Tests for exp operation."""

    def test_forward(self):
        """Basic exp."""
        a = Tensor([0.0, 1.0, 2.0])
        c = exp(a)
        assert np.allclose(c.data, [1.0, np.e, np.e ** 2])

    def test_backward(self):
        """Gradients for exp."""
        a = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        c = exp(a)
        c.backward(np.ones(3))
        assert np.allclose(a.grad, [1.0, np.e, np.e ** 2])

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(exp, [np.random.randn(3, 4)])


class TestLog:
    """Tests for log operation."""

    def test_forward(self):
        """Basic log."""
        a = Tensor([1.0, np.e, np.e ** 2])
        c = log(a)
        assert np.allclose(c.data, [0.0, 1.0, 2.0], atol=1e-9)

    def test_backward(self):
        """Gradients for log."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = log(a)
        c.backward(np.ones(3))
        assert np.allclose(a.grad, [1.0, 0.5, 1/3])

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(log, [np.abs(np.random.randn(3, 4)) + 0.1])

    def test_numerical_stability(self):
        """Test numerical stability near zero."""
        a = Tensor([1e-15, 1e-10, 1e-5], requires_grad=True)
        c = log(a)
        c.backward(np.ones(3))
        assert np.isfinite(c.data).all()
        assert np.isfinite(a.grad).all()


class TestPow:
    """Tests for pow operation."""

    def test_forward_square(self):
        """Square operation."""
        a = Tensor([1.0, 2.0, 3.0])
        c = pow(a, 2)
        assert np.allclose(c.data, [1.0, 4.0, 9.0])

    def test_forward_sqrt(self):
        """Square root via pow."""
        a = Tensor([1.0, 4.0, 9.0])
        c = pow(a, 0.5)
        assert np.allclose(c.data, [1.0, 2.0, 3.0])

    def test_backward(self):
        """Gradients for pow."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = pow(a, 2)
        c.backward(np.ones(3))
        assert np.allclose(a.grad, [2.0, 4.0, 6.0])

    def test_backward_fractional(self):
        """Gradients for fractional power."""
        a = Tensor([1.0, 4.0, 9.0], requires_grad=True)
        c = pow(a, 0.5)
        c.backward(np.ones(3))
        expected = 0.5 / np.sqrt([1.0, 4.0, 9.0])
        assert np.allclose(a.grad, expected)

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(lambda x: pow(x, 2), [np.random.randn(3, 4)])
        assert check_gradient(lambda x: pow(x, 0.5), [np.abs(np.random.randn(3, 4)) + 0.1])

    def test_operator_overload(self):
        """Test ** operator."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = a ** 2
        c.backward(np.ones(3))
        assert np.allclose(c.data, [1.0, 4.0, 9.0])
        assert np.allclose(a.grad, [2.0, 4.0, 6.0])


class TestSqrt:
    """Tests for sqrt operation."""

    def test_forward(self):
        """Basic sqrt."""
        a = Tensor([1.0, 4.0, 9.0])
        c = sqrt(a)
        assert np.allclose(c.data, [1.0, 2.0, 3.0])

    def test_backward(self):
        """Gradients for sqrt."""
        a = Tensor([1.0, 4.0, 9.0], requires_grad=True)
        c = sqrt(a)
        c.backward(np.ones(3))
        expected = 0.5 / np.sqrt([1.0, 4.0, 9.0])
        assert np.allclose(a.grad, expected)

    def test_numerical_gradient(self):
        """Verify gradient against finite differences."""
        assert check_gradient(sqrt, [np.abs(np.random.randn(3, 4)) + 0.1])

    def test_numerical_stability(self):
        """Test numerical stability near zero."""
        a = Tensor([1e-10, 1e-5, 1.0], requires_grad=True)
        c = sqrt(a)
        c.backward(np.ones(3))
        assert np.isfinite(c.data).all()
        assert np.isfinite(a.grad).all()


class TestComplexExpressions:
    """Tests for complex combinations of operations."""

    def test_quadratic(self):
        """Test quadratic expression: x^2 + 2x + 1 = (x+1)^2."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x ** 2 + x * 2.0 + 1.0
        loss = y.sum()
        loss.backward()

        expected = 2 * np.array([1.0, 2.0, 3.0]) + 2
        assert np.allclose(x.grad, expected)

    def test_sigmoid_cross_entropy_like(self):
        """Test expression similar to binary cross entropy."""
        x = Tensor([0.5, 1.0, -0.5], requires_grad=True)
        sig = sigmoid(x)
        loss = -log(sig).sum()
        loss.backward()

        assert np.isfinite(x.grad).all()

    def test_mlp_like_forward(self):
        """Test MLP-like computation."""
        x = Tensor(np.random.randn(32, 10), requires_grad=True)
        W1 = Tensor(np.random.randn(10, 64), requires_grad=True)
        b1 = Tensor(np.zeros(64), requires_grad=True)
        W2 = Tensor(np.random.randn(64, 1), requires_grad=True)
        b2 = Tensor(np.zeros(1), requires_grad=True)

        h1 = relu(matmul(x, W1) + b1)
        out = matmul(h1, W2) + b2
        loss = out.sum()
        loss.backward()

        assert x.grad is not None and x.grad.shape == (32, 10)
        assert W1.grad is not None and W1.grad.shape == (10, 64)
        assert b1.grad is not None and b1.grad.shape == (64,)
        assert W2.grad is not None and W2.grad.shape == (64, 1)
        assert b2.grad is not None and b2.grad.shape == (1,)

    def test_energy_function_like(self):
        """Test computation similar to energy function in EBM."""
        x = Tensor(np.random.randn(32, 2), requires_grad=True)
        W = Tensor(np.random.randn(2, 64), requires_grad=True)
        b = Tensor(np.zeros(64), requires_grad=True)

        h = swish(matmul(x, W) + b)
        energy = h.sum()
        energy.backward()

        assert x.grad is not None and x.grad.shape == (32, 2)
        assert W.grad is not None and W.grad.shape == (2, 64)
        assert b.grad is not None and b.grad.shape == (64,)


class TestGradientAccumulation:
    """Tests for gradient accumulation when tensors are reused."""

    def test_tensor_used_twice_add(self):
        """Test gradient when tensor is added to itself."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = a + a
        c.backward(np.ones(3))
        assert np.allclose(a.grad, 2.0)

    def test_tensor_used_in_mul(self):
        """Test gradient when tensor is multiplied by itself."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = a * a
        c.backward(np.ones(3))
        assert np.allclose(a.grad, [2.0, 4.0, 6.0])

    def test_diamond_pattern(self):
        """Test gradient flow in diamond graph: a -> b, c -> d."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = a * 2.0
        c = a + 1.0
        d = b + c
        d.backward(np.ones(3))

        assert np.allclose(a.grad, 3.0)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_scalar_operations(self):
        """Test operations on scalars."""
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(3.0, requires_grad=True)
        c = a * b + a ** 2
        c.backward()

        assert np.allclose(a.grad, 7.0)
        assert np.allclose(b.grad, 2.0)

    def test_high_dimensional(self):
        """Test operations on high-dimensional tensors."""
        a = Tensor(np.random.randn(2, 3, 4, 5), requires_grad=True)
        c = relu(a)
        d = c.sum()
        d.backward()

        assert a.grad.shape == (2, 3, 4, 5)

    def test_zero_gradient_flow(self):
        """Test that zero gradients propagate correctly."""
        a = Tensor([-1.0, -2.0, -3.0], requires_grad=True)
        c = relu(a)
        d = c.sum()
        d.backward()

        assert np.allclose(a.grad, 0.0)

    def test_mixed_requires_grad(self):
        """Test operations with mixed requires_grad."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=False)
        c = a + b
        c.backward(np.ones(2))

        assert a.grad is not None
        assert b.grad is None


class TestEdgeCaseCoverage:
    """Tests to cover remaining edge cases for 100% coverage."""

    def test_matmul_1d_vector(self):
        """Test matmul with 1D vectors."""
        a = Tensor(np.random.randn(4), requires_grad=True)
        b = Tensor(np.random.randn(4, 3), requires_grad=True)
        c = matmul(a, b)
        assert c.shape == (3,)
        c.backward(np.ones(3))
        assert a.grad.shape == (4,)
        assert b.grad.shape == (4, 3)

    def test_matmul_matrix_1d_vector(self):
        """Test matmul with matrix @ 1D vector."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4), requires_grad=True)
        c = matmul(a, b)
        assert c.shape == (3,)
        c.backward(np.ones(3))
        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (4,)

    def test_sum_multi_axis(self):
        """Test sum with tuple axis."""
        a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        c = tensor_sum(a, axis=(0, 2))
        assert c.shape == (3,)
        c.backward(np.ones(3))
        assert a.grad.shape == (2, 3, 4)
        assert np.allclose(a.grad, 1.0)

    def test_mean_multi_axis(self):
        """Test mean with tuple axis."""
        a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        c = mean(a, axis=(0, 2))
        assert c.shape == (3,)
        c.backward(np.ones(3))
        assert a.grad.shape == (2, 3, 4)
        assert np.allclose(a.grad, 1.0 / 8)

    def test_pow_zero_exponent(self):
        """Test pow with n=0."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = pow(a, 0)
        assert np.allclose(c.data, 1.0)
        c.backward(np.ones(3))
        assert np.allclose(a.grad, 0.0)

    def test_pow_one_exponent(self):
        """Test pow with n=1."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = pow(a, 1)
        assert np.allclose(c.data, [1.0, 2.0, 3.0])
        c.backward(np.ones(3))
        assert np.allclose(a.grad, 1.0)

    def test_operator_tensor_to_tensor(self):
        """Test operators with two Tensors (not scalar)."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)

        c = a + b
        c.backward(np.ones(2))
        assert np.allclose(a.grad, 1.0)
        assert np.allclose(b.grad, 1.0)

    def test_operator_sub_tensor_to_tensor(self):
        """Test subtraction operator with two Tensors."""
        a = Tensor([5.0, 6.0], requires_grad=True)
        b = Tensor([1.0, 2.0], requires_grad=True)
        c = a - b
        c.backward(np.ones(2))
        assert np.allclose(a.grad, 1.0)
        assert np.allclose(b.grad, -1.0)

    def test_operator_mul_tensor_to_tensor(self):
        """Test multiplication operator with two Tensors."""
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        c = a * b
        c.backward(np.ones(2))
        assert np.allclose(a.grad, [4.0, 5.0])
        assert np.allclose(b.grad, [2.0, 3.0])

    def test_operator_div_tensor_to_tensor(self):
        """Test division operator with two Tensors."""
        a = Tensor([10.0, 20.0], requires_grad=True)
        b = Tensor([2.0, 4.0], requires_grad=True)
        c = a / b
        c.backward(np.ones(2))
        assert np.allclose(a.grad, [0.5, 0.25])

    def test_operator_matmul_tensor_to_tensor(self):
        """Test matmul operator with two Tensors."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4, 5), requires_grad=True)
        c = a @ b
        c.backward(np.ones((3, 5)))
        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (4, 5)

    def test_rmul_scalar(self):
        """Test reverse multiplication with scalar."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = 2.0 * a
        c.backward(np.ones(3))
        assert np.allclose(c.data, [2.0, 4.0, 6.0])
        assert np.allclose(a.grad, 2.0)

    def test_rtruediv_scalar(self):
        """Test reverse division with scalar."""
        a = Tensor([1.0, 2.0, 4.0], requires_grad=True)
        c = 8.0 / a
        c.backward(np.ones(3))
        assert np.allclose(c.data, [8.0, 4.0, 2.0])
        assert np.allclose(a.grad, [-8.0, -2.0, -0.5])

    def test_transpose_with_axes_backward(self):
        """Test transpose backward with specific axes."""
        a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        c = transpose(a, (2, 0, 1))
        c.backward(np.ones((4, 2, 3)))
        assert a.grad.shape == (2, 3, 4)
        assert np.allclose(a.grad, 1.0)

    def test_sub_tensor_to_tensor_via_function(self):
        """Test sub with two Tensors via function."""
        a = Tensor([5.0, 6.0], requires_grad=True)
        b = Tensor([1.0, 2.0], requires_grad=True)
        c = sub(a, b)
        c.backward(np.ones(2))
        assert np.allclose(a.grad, 1.0)

    def test_mul_tensor_to_tensor_via_function(self):
        """Test mul with two Tensors via function."""
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        c = mul(a, b)
        c.backward(np.ones(2))
        assert np.allclose(a.grad, [4.0, 5.0])

    def test_div_tensor_to_tensor_via_function(self):
        """Test div with two Tensors via function."""
        a = Tensor([10.0, 20.0], requires_grad=True)
        b = Tensor([2.0, 4.0], requires_grad=True)
        c = div(a, b)
        c.backward(np.ones(2))
        assert np.allclose(a.grad, [0.5, 0.25])

    def test_matmul_tensor_to_tensor_via_function(self):
        """Test matmul with two Tensors via function."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4, 5), requires_grad=True)
        c = matmul(a, b)
        c.backward(np.ones((3, 5)))
        assert a.grad.shape == (3, 4)

    def test_sub_scalar_via_operator(self):
        """Test tensor - scalar via operator (covers _tensor_sub conversion)."""
        a = Tensor([5.0, 6.0, 7.0], requires_grad=True)
        c = a - 1.0
        c.backward(np.ones(3))
        assert np.allclose(c.data, [4.0, 5.0, 6.0])
        assert np.allclose(a.grad, 1.0)

    def test_matmul_tensor_numpy_via_operator(self):
        """Test tensor @ numpy via operator (covers _tensor_matmul conversion)."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = np.random.randn(4, 5)
        c = a @ b
        c.backward(np.ones((3, 5)))
        assert c.shape == (3, 5)
        assert a.grad.shape == (3, 4)


class TestBroadcastingGradients:
    """
    Comprehensive tests for broadcasting gradient handling.

    Section 1.3 of the README specifies that when inputs have different shapes,
    gradients must be summed along broadcast dimensions. These tests verify
    this behavior across all operations that support broadcasting.
    """

    def test_unbroadcast_multiple_size_one_dims(self):
        """Test unbroadcast with multiple size-1 dimensions."""
        grad = np.random.randn(3, 4, 5)
        result = unbroadcast(grad, (1, 1, 5))
        assert result.shape == (1, 1, 5)
        expected = grad.sum(axis=(0, 1), keepdims=True)
        assert np.allclose(result, expected)

    def test_unbroadcast_all_size_one(self):
        """Test unbroadcast when all original dims are size 1."""
        grad = np.random.randn(3, 4, 5)
        result = unbroadcast(grad, (1, 1, 1))
        assert result.shape == (1, 1, 1)
        expected = grad.sum(keepdims=True)
        assert np.allclose(result, expected)

    def test_unbroadcast_mixed_leading_and_size_one(self):
        """Test unbroadcast with both leading dims and size-1 dims."""
        grad = np.random.randn(2, 3, 4, 5)
        result = unbroadcast(grad, (1, 5))
        assert result.shape == (1, 5)

    def test_unbroadcast_high_dimensional(self):
        """Test unbroadcast with high-dimensional tensors."""
        grad = np.random.randn(2, 3, 4, 5, 6)
        result = unbroadcast(grad, (4, 1, 6))
        assert result.shape == (4, 1, 6)

    def test_add_broadcast_2d_1d(self):
        """Test add with (3,4) + (4,) - the README example."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4,), requires_grad=True)
        c = add(a, b)
        c.backward(np.ones((3, 4)))

        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (4,)
        assert np.allclose(a.grad, 1.0)
        assert np.allclose(b.grad, 3.0)

    def test_add_broadcast_3d_1d(self):
        """Test add with (2,3,4) + (4,)."""
        a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4,), requires_grad=True)
        c = add(a, b)
        c.backward(np.ones((2, 3, 4)))

        assert a.grad.shape == (2, 3, 4)
        assert b.grad.shape == (4,)
        assert np.allclose(a.grad, 1.0)
        assert np.allclose(b.grad, 6.0)

    def test_add_broadcast_size_one_middle_dim(self):
        """Test add with (3,1,4) + (3,5,4)."""
        a = Tensor(np.random.randn(3, 1, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 5, 4), requires_grad=True)
        c = add(a, b)
        c.backward(np.ones((3, 5, 4)))

        assert a.grad.shape == (3, 1, 4)
        assert b.grad.shape == (3, 5, 4)
        assert np.allclose(b.grad, 1.0)
        assert np.allclose(a.grad, 5.0)

    def test_add_broadcast_scalar_tensor(self):
        """Test add with scalar + (3,4)."""
        a = Tensor(np.array(2.0), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        c = add(a, b)
        c.backward(np.ones((3, 4)))

        assert a.grad.shape == ()
        assert b.grad.shape == (3, 4)
        assert np.allclose(a.grad, 12.0)
        assert np.allclose(b.grad, 1.0)

    def test_add_broadcast_numerical_gradient_3d_2d(self):
        """Verify add gradient with (2,3,4) + (3,4) via finite differences."""
        assert check_gradient(add, [np.random.randn(2, 3, 4), np.random.randn(3, 4)])

    def test_add_broadcast_numerical_gradient_size_one(self):
        """Verify add gradient with size-1 dims via finite differences."""
        assert check_gradient(add, [np.random.randn(3, 1, 4), np.random.randn(3, 5, 4)])

    def test_sub_broadcast_2d_1d(self):
        """Test sub with (3,4) - (4,)."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4,), requires_grad=True)
        c = sub(a, b)
        c.backward(np.ones((3, 4)))

        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (4,)
        assert np.allclose(a.grad, 1.0)
        assert np.allclose(b.grad, -3.0)

    def test_sub_broadcast_3d_2d(self):
        """Test sub with (2,3,4) - (3,4)."""
        a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        c = sub(a, b)
        c.backward(np.ones((2, 3, 4)))

        assert a.grad.shape == (2, 3, 4)
        assert b.grad.shape == (3, 4)
        assert np.allclose(a.grad, 1.0)
        assert np.allclose(b.grad, -2.0)

    def test_sub_broadcast_size_one_dim(self):
        """Test sub with (3,1) - (3,4)."""
        a = Tensor(np.random.randn(3, 1), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        c = sub(a, b)
        c.backward(np.ones((3, 4)))

        assert a.grad.shape == (3, 1)
        assert b.grad.shape == (3, 4)
        assert np.allclose(a.grad, 4.0)
        assert np.allclose(b.grad, -1.0)

    def test_sub_broadcast_numerical_gradient(self):
        """Verify sub gradient with broadcasting via finite differences."""
        assert check_gradient(sub, [np.random.randn(3, 4), np.random.randn(4,)])
        assert check_gradient(sub, [np.random.randn(2, 3, 4), np.random.randn(3, 4)])

    def test_mul_broadcast_2d_1d(self):
        """Test mul with (3,4) * (4,)."""
        a_data = np.random.randn(3, 4)
        b_data = np.random.randn(4,)
        a = Tensor(a_data, requires_grad=True)
        b = Tensor(b_data, requires_grad=True)
        c = mul(a, b)
        c.backward(np.ones((3, 4)))

        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (4,)
        expected_grad_a = np.broadcast_to(b_data, (3, 4))
        assert np.allclose(a.grad, expected_grad_a)
        expected_grad_b = a_data.sum(axis=0)
        assert np.allclose(b.grad, expected_grad_b)

    def test_mul_broadcast_3d_1d(self):
        """Test mul with (2,3,4) * (4,)."""
        a_data = np.random.randn(2, 3, 4)
        b_data = np.random.randn(4,)
        a = Tensor(a_data, requires_grad=True)
        b = Tensor(b_data, requires_grad=True)
        c = mul(a, b)
        c.backward(np.ones((2, 3, 4)))

        assert a.grad.shape == (2, 3, 4)
        assert b.grad.shape == (4,)

    def test_mul_broadcast_size_one_dim(self):
        """Test mul with (3,1,4) * (3,5,4)."""
        a_data = np.random.randn(3, 1, 4)
        b_data = np.random.randn(3, 5, 4)
        a = Tensor(a_data, requires_grad=True)
        b = Tensor(b_data, requires_grad=True)
        c = mul(a, b)
        c.backward(np.ones((3, 5, 4)))

        assert a.grad.shape == (3, 1, 4)
        assert b.grad.shape == (3, 5, 4)

    def test_mul_broadcast_numerical_gradient_various(self):
        """Verify mul gradient with various broadcast patterns."""
        assert check_gradient(mul, [np.random.randn(3, 4), np.random.randn(4,)])
        assert check_gradient(mul, [np.random.randn(2, 3, 4), np.random.randn(4,)])
        assert check_gradient(mul, [np.random.randn(3, 1), np.random.randn(3, 4)])

    def test_div_broadcast_2d_1d(self):
        """Test div with (3,4) / (4,)."""
        a_data = np.random.randn(3, 4)
        b_data = np.abs(np.random.randn(4,)) + 0.5
        a = Tensor(a_data, requires_grad=True)
        b = Tensor(b_data, requires_grad=True)
        c = div(a, b)
        c.backward(np.ones((3, 4)))

        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (4,)
        expected_grad_a = 1.0 / np.broadcast_to(b_data, (3, 4))
        assert np.allclose(a.grad, expected_grad_a)

    def test_div_broadcast_3d_2d(self):
        """Test div with (2,3,4) / (3,4)."""
        a_data = np.random.randn(2, 3, 4)
        b_data = np.abs(np.random.randn(3, 4)) + 0.5
        a = Tensor(a_data, requires_grad=True)
        b = Tensor(b_data, requires_grad=True)
        c = div(a, b)
        c.backward(np.ones((2, 3, 4)))

        assert a.grad.shape == (2, 3, 4)
        assert b.grad.shape == (3, 4)

    def test_div_broadcast_size_one_dim(self):
        """Test div with (3,4) / (1,4)."""
        a_data = np.random.randn(3, 4)
        b_data = np.abs(np.random.randn(1, 4)) + 0.5
        a = Tensor(a_data, requires_grad=True)
        b = Tensor(b_data, requires_grad=True)
        c = div(a, b)
        c.backward(np.ones((3, 4)))

        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (1, 4)

    def test_div_broadcast_numerical_gradient_various(self):
        """Verify div gradient with various broadcast patterns."""
        assert check_gradient(div, [np.random.randn(3, 4), np.abs(np.random.randn(4,)) + 0.5])
        assert check_gradient(div, [np.random.randn(2, 3, 4), np.abs(np.random.randn(3, 4)) + 0.5])
        assert check_gradient(div, [np.random.randn(3, 4), np.abs(np.random.randn(1, 4)) + 0.5])

    def test_chained_broadcast_ops(self):
        """Test gradient flow through chained operations with broadcasting."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        w = Tensor(np.random.randn(4,), requires_grad=True)
        b = Tensor(np.random.randn(3, 1), requires_grad=True)

        y = add(mul(x, w), b)
        y.backward(np.ones((3, 4)))

        assert x.grad.shape == (3, 4)
        assert w.grad.shape == (4,)
        assert b.grad.shape == (3, 1)
        assert np.allclose(b.grad, 4.0)

    def test_broadcast_with_shared_tensor(self):
        """Test when same tensor is broadcast in multiple operations."""
        x = Tensor(np.random.randn(4,), requires_grad=True)
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(2, 4), requires_grad=True)

        y1 = add(a, x)
        y2 = mul(b, x)

        y1.backward(np.ones((3, 4)))
        y2.backward(np.ones((2, 4)))

        assert x.grad.shape == (4,)
        expected_contrib_from_y1 = 3.0
        expected_contrib_from_y2 = b.data.sum(axis=0)
        assert np.allclose(x.grad, expected_contrib_from_y1 + expected_contrib_from_y2)

    def test_broadcast_energy_like_computation(self):
        """Test broadcasting in computation similar to EBM energy function."""
        batch_size, input_dim, hidden_dim = 32, 10, 64

        x = Tensor(np.random.randn(batch_size, input_dim), requires_grad=True)
        W = Tensor(np.random.randn(input_dim, hidden_dim), requires_grad=True)
        bias = Tensor(np.random.randn(hidden_dim), requires_grad=True)

        h = add(matmul(x, W), bias)
        energy = h.sum()
        energy.backward()

        assert x.grad.shape == (batch_size, input_dim)
        assert W.grad.shape == (input_dim, hidden_dim)
        assert bias.grad.shape == (hidden_dim,)
        assert np.allclose(bias.grad, batch_size)


class TestNumericalStability:
    """Tests for numerical stability in edge cases."""

    def test_softplus_large_positive(self):
        """Softplus should not overflow for large positive inputs."""
        a = Tensor([100.0, 500.0, 1000.0], requires_grad=True)
        c = softplus(a)
        c.backward(np.ones(3))

        assert np.isfinite(c.data).all()
        assert np.isfinite(a.grad).all()
        assert np.allclose(c.data, a.data, rtol=0.01)

    def test_softplus_large_negative(self):
        """Softplus should not underflow for large negative inputs."""
        a = Tensor([-100.0, -500.0, -1000.0], requires_grad=True)
        c = softplus(a)
        c.backward(np.ones(3))

        assert np.isfinite(c.data).all()
        assert np.isfinite(a.grad).all()
        assert c.data[0] < 1e-40

    def test_sigmoid_extreme(self):
        """Sigmoid should handle extreme values."""
        a = Tensor([-1000.0, 0.0, 1000.0], requires_grad=True)
        c = sigmoid(a)
        c.backward(np.ones(3))

        assert np.isfinite(c.data).all()
        assert np.isfinite(a.grad).all()
        assert c.data[0] < 1e-10
        assert c.data[1] == 0.5
        assert c.data[2] > 1 - 1e-10

    def test_log_small_values(self):
        """Log should handle very small positive values."""
        a = Tensor([1e-100, 1e-50, 1e-10], requires_grad=True)
        c = log(a)
        c.backward(np.ones(3))

        assert np.isfinite(c.data).all()
        assert np.isfinite(a.grad).all()
