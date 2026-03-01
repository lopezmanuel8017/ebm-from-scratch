"""
Comprehensive tests for neural network layers (Section 2.1).

These tests verify:
1. Linear layer initialization (He init, shapes)
2. Linear layer forward pass and gradient computation
3. Activation layers (ReLU, Swish, Softplus, Sigmoid)
4. Sequential container functionality
5. Gradient flow through complete networks
6. Numerical gradient verification
7. Edge cases and numerical stability
"""

import pytest
import numpy as np

from ebm.core.autodiff import Tensor
from ebm.core.nn import (
    Linear, ReLU, Swish, Softplus, Sigmoid, Sequential, _create_mlp
)
from ebm.core.ops import relu, swish, softplus, sigmoid


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


def check_layer_gradient(layer, input_shape, eps=1e-5, rtol=1e-4, atol=1e-4):
    """
    Verify layer gradients against finite differences.

    Args:
        layer: Module to test
        input_shape: Shape of input tensor
        eps: Finite difference step size
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if gradients match
    """
    x_data = np.random.randn(*input_shape)
    x = Tensor(x_data, requires_grad=True)

    y = layer(x)
    _loss = y.data.sum()
    y.backward(np.ones_like(y.data))

    def func(x_val):
        t = Tensor(x_val, requires_grad=False)
        return layer(t).data.sum()

    num_grad = numerical_gradient(func, x_data, eps)
    if not np.allclose(x.grad, num_grad, rtol=rtol, atol=atol):
        return False

    return True


class TestLinearInitialization:
    """Tests for Linear layer initialization."""

    def test_weight_shape(self):
        """Weight matrix has correct shape (in_features, out_features)."""
        layer = Linear(10, 5)
        assert layer.W.shape == (10, 5)

    def test_bias_shape(self):
        """Bias vector has correct shape (out_features,)."""
        layer = Linear(10, 5)
        assert layer.b.shape == (5,)

    def test_weight_requires_grad(self):
        """Weight has requires_grad=True."""
        layer = Linear(10, 5)
        assert layer.W.requires_grad is True

    def test_bias_requires_grad(self):
        """Bias has requires_grad=True."""
        layer = Linear(10, 5)
        assert layer.b.requires_grad is True

    def test_bias_initialized_to_zeros(self):
        """Bias is initialized to zeros."""
        layer = Linear(10, 5)
        assert np.allclose(layer.b.data, 0.0)

    def test_he_initialization_scale(self):
        """Weights follow He initialization scale."""
        np.random.seed(42)
        in_features = 100
        out_features = 50

        variances = []
        for _ in range(100):
            layer = Linear(in_features, out_features)
            variances.append(layer.W.data.var())

        mean_var = np.mean(variances)
        expected_var = 2.0 / in_features

        assert np.abs(mean_var - expected_var) / expected_var < 0.2

    def test_no_bias_option(self):
        """Linear layer can be created without bias."""
        layer = Linear(10, 5, bias=False)
        assert layer.b is None

    def test_parameters_with_bias(self):
        """parameters() returns [W, b] when bias=True."""
        layer = Linear(10, 5)
        params = layer.parameters()
        assert len(params) == 2
        assert any(p is layer.W for p in params)
        assert any(p is layer.b for p in params)

    def test_parameters_without_bias(self):
        """parameters() returns [W] when bias=False."""
        layer = Linear(10, 5, bias=False)
        params = layer.parameters()
        assert len(params) == 1
        assert any(p is layer.W for p in params)

    def test_repr(self):
        """__repr__ shows layer configuration."""
        layer = Linear(10, 5)
        repr_str = repr(layer)
        assert "Linear" in repr_str
        assert "10" in repr_str
        assert "5" in repr_str


class TestLinearForward:
    """Tests for Linear layer forward pass."""

    def test_forward_output_shape(self):
        """Forward pass produces correct output shape."""
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(32, 10))
        y = layer(x)
        assert y.shape == (32, 5)

    def test_forward_single_sample(self):
        """Forward pass works with single sample."""
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(1, 10))
        y = layer(x)
        assert y.shape == (1, 5)

    def test_forward_correctness(self):
        """Forward pass computes x @ W + b correctly."""
        layer = Linear(3, 2)
        x = Tensor([[1.0, 2.0, 3.0]])

        expected = x.data @ layer.W.data + layer.b.data
        y = layer(x)

        assert np.allclose(y.data, expected)

    def test_forward_no_bias_correctness(self):
        """Forward pass without bias computes x @ W correctly."""
        layer = Linear(3, 2, bias=False)
        x = Tensor([[1.0, 2.0, 3.0]])

        expected = x.data @ layer.W.data
        y = layer(x)

        assert np.allclose(y.data, expected)

    def test_forward_batch(self):
        """Forward pass handles batch correctly."""
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(64, 10))
        y = layer(x)

        for i in range(64):
            expected = x.data[i:i+1] @ layer.W.data + layer.b.data
            assert np.allclose(y.data[i:i+1], expected)


class TestLinearBackward:
    """Tests for Linear layer backward pass."""

    def test_backward_weight_gradient_shape(self):
        """Weight gradient has correct shape."""
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(32, 10), requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.W.grad.shape == (10, 5)

    def test_backward_bias_gradient_shape(self):
        """Bias gradient has correct shape."""
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(32, 10), requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.b.grad.shape == (5,)

    def test_backward_input_gradient_shape(self):
        """Input gradient has correct shape."""
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(32, 10), requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad.shape == (32, 10)

    def test_backward_numerical_gradient_weight(self):
        """Weight gradient matches finite differences."""
        layer = Linear(4, 3)
        x_data = np.random.randn(8, 4)

        x = Tensor(x_data, requires_grad=True)
        y = layer(x)
        y.backward(np.ones_like(y.data))

        def func_w(w_val):
            old_w = layer.W.data.copy()
            layer.W.data = w_val.reshape(4, 3)
            t = Tensor(x_data, requires_grad=False)
            out = layer(t).data.sum()
            layer.W.data = old_w
            return out

        num_grad_w = numerical_gradient(func_w, layer.W.data.flatten())
        assert np.allclose(layer.W.grad.flatten(), num_grad_w, rtol=1e-4, atol=1e-4)

    def test_backward_numerical_gradient_bias(self):
        """Bias gradient matches finite differences."""
        layer = Linear(4, 3)
        x_data = np.random.randn(8, 4)

        x = Tensor(x_data, requires_grad=True)
        y = layer(x)
        y.backward(np.ones_like(y.data))

        def func_b(b_val):
            old_b = layer.b.data.copy()
            layer.b.data = b_val
            t = Tensor(x_data, requires_grad=False)
            out = layer(t).data.sum()
            layer.b.data = old_b
            return out

        num_grad_b = numerical_gradient(func_b, layer.b.data)
        assert np.allclose(layer.b.grad, num_grad_b, rtol=1e-4, atol=1e-4)

    def test_backward_numerical_gradient_input(self):
        """Input gradient matches finite differences."""
        layer = Linear(4, 3)
        x_data = np.random.randn(8, 4)

        x = Tensor(x_data, requires_grad=True)
        y = layer(x)
        y.backward(np.ones_like(y.data))

        def func_x(x_val):
            t = Tensor(x_val.reshape(8, 4), requires_grad=False)
            return layer(t).data.sum()

        num_grad_x = numerical_gradient(func_x, x_data.flatten())
        assert np.allclose(x.grad.flatten(), num_grad_x, rtol=1e-4, atol=1e-4)

    def test_backward_no_bias(self):
        """Backward pass works correctly without bias."""
        layer = Linear(4, 3, bias=False)
        x = Tensor(np.random.randn(8, 4), requires_grad=True)
        y = layer(x)
        y.backward(np.ones_like(y.data))

        assert layer.W.grad is not None
        assert x.grad is not None


class TestLinearZeroGrad:
    """Tests for gradient zeroing."""

    def test_zero_grad(self):
        """zero_grad() clears gradients."""
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(32, 10), requires_grad=True)
        y = layer(x)
        y.sum().backward()

        assert layer.W.grad is not None
        assert layer.b.grad is not None

        layer.zero_grad()

        assert layer.W.grad is None
        assert layer.b.grad is None


class TestReLUModule:
    """Tests for ReLU activation module."""

    def test_forward(self):
        """ReLU module produces same result as relu op."""
        act = ReLU()
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

        y_module = act(x)
        y_op = relu(Tensor(x.data, requires_grad=True))

        assert np.allclose(y_module.data, y_op.data)

    def test_backward(self):
        """ReLU module backward matches relu op."""
        act = ReLU()
        x1 = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        x2 = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

        y1 = act(x1)
        y1.backward(np.ones(5))

        y2 = relu(x2)
        y2.backward(np.ones(5))

        assert np.allclose(x1.grad, x2.grad)

    def test_no_parameters(self):
        """ReLU has no trainable parameters."""
        act = ReLU()
        assert len(act.parameters()) == 0

    def test_repr(self):
        """ReLU __repr__ is correct."""
        act = ReLU()
        assert repr(act) == "ReLU()"

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_shapes(self, shape):
        """ReLU works with various shapes."""
        act = ReLU()
        x = Tensor(np.random.randn(*shape), requires_grad=True)
        y = act(x)
        y.backward(np.ones(shape))

        assert y.shape == shape
        assert x.grad.shape == shape


class TestSwishModule:
    """Tests for Swish activation module."""

    def test_forward(self):
        """Swish module produces same result as swish op."""
        act = Swish()
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

        y_module = act(x)
        y_op = swish(Tensor(x.data, requires_grad=True))

        assert np.allclose(y_module.data, y_op.data)

    def test_backward(self):
        """Swish module backward matches swish op."""
        act = Swish()
        x1 = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        x2 = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

        y1 = act(x1)
        y1.backward(np.ones(5))

        y2 = swish(x2)
        y2.backward(np.ones(5))

        assert np.allclose(x1.grad, x2.grad)

    def test_no_parameters(self):
        """Swish has no trainable parameters."""
        act = Swish()
        assert len(act.parameters()) == 0

    def test_repr(self):
        """Swish __repr__ is correct."""
        act = Swish()
        assert repr(act) == "Swish()"

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_shapes(self, shape):
        """Swish works with various shapes."""
        act = Swish()
        x = Tensor(np.random.randn(*shape), requires_grad=True)
        y = act(x)
        y.backward(np.ones(shape))

        assert y.shape == shape
        assert x.grad.shape == shape

    def test_numerical_stability(self):
        """Swish handles extreme values."""
        act = Swish()
        x = Tensor([-500.0, 500.0], requires_grad=True)
        y = act(x)
        y.backward(np.ones(2))

        assert np.isfinite(y.data).all()
        assert np.isfinite(x.grad).all()


class TestSoftplusModule:
    """Tests for Softplus activation module."""

    def test_forward(self):
        """Softplus module produces same result as softplus op."""
        act = Softplus()
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

        y_module = act(x)
        y_op = softplus(Tensor(x.data, requires_grad=True))

        assert np.allclose(y_module.data, y_op.data)

    def test_backward(self):
        """Softplus module backward matches softplus op."""
        act = Softplus()
        x1 = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        x2 = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

        y1 = act(x1)
        y1.backward(np.ones(5))

        y2 = softplus(x2)
        y2.backward(np.ones(5))

        assert np.allclose(x1.grad, x2.grad)

    def test_no_parameters(self):
        """Softplus has no trainable parameters."""
        act = Softplus()
        assert len(act.parameters()) == 0

    def test_threshold_parameter(self):
        """Softplus threshold can be customized."""
        act = Softplus(threshold=10.0)
        assert act.threshold == 10.0

    def test_repr(self):
        """Softplus __repr__ shows threshold."""
        act = Softplus(threshold=15.0)
        assert "15.0" in repr(act)

    def test_numerical_stability(self):
        """Softplus handles extreme values."""
        act = Softplus()
        x = Tensor([-500.0, 500.0], requires_grad=True)
        y = act(x)
        y.backward(np.ones(2))

        assert np.isfinite(y.data).all()
        assert np.isfinite(x.grad).all()


class TestSigmoidModule:
    """Tests for Sigmoid activation module."""

    def test_forward(self):
        """Sigmoid module produces same result as sigmoid op."""
        act = Sigmoid()
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

        y_module = act(x)
        y_op = sigmoid(Tensor(x.data, requires_grad=True))

        assert np.allclose(y_module.data, y_op.data)

    def test_backward(self):
        """Sigmoid module backward matches sigmoid op."""
        act = Sigmoid()
        x1 = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        x2 = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

        y1 = act(x1)
        y1.backward(np.ones(5))

        y2 = sigmoid(x2)
        y2.backward(np.ones(5))

        assert np.allclose(x1.grad, x2.grad)

    def test_no_parameters(self):
        """Sigmoid has no trainable parameters."""
        act = Sigmoid()
        assert len(act.parameters()) == 0

    def test_repr(self):
        """Sigmoid __repr__ is correct."""
        act = Sigmoid()
        assert repr(act) == "Sigmoid()"


class TestSequentialBasic:
    """Basic tests for Sequential container."""

    def test_empty_sequential(self):
        """Empty Sequential can be created."""
        seq = Sequential()
        assert len(seq) == 0

    def test_sequential_from_list(self):
        """Sequential can be initialized with list of layers."""
        layers = [Linear(10, 5), ReLU(), Linear(5, 2)]
        seq = Sequential(layers)
        assert len(seq) == 3

    def test_add_layer(self):
        """Layers can be added with add()."""
        seq = Sequential()
        seq.add(Linear(10, 5))
        seq.add(ReLU())
        assert len(seq) == 2

    def test_add_returns_self(self):
        """add() returns self for chaining."""
        seq = Sequential()
        result = seq.add(Linear(10, 5))
        assert result is seq

    def test_method_chaining(self):
        """Methods can be chained."""
        seq = Sequential().add(Linear(10, 5)).add(ReLU()).add(Linear(5, 2))
        assert len(seq) == 3

    def test_getitem(self):
        """Layers can be accessed by index."""
        linear = Linear(10, 5)
        seq = Sequential([linear, ReLU()])
        assert id(seq[0]) == id(linear)

    def test_iteration(self):
        """Sequential can be iterated."""
        layers = [Linear(10, 5), ReLU(), Linear(5, 2)]
        seq = Sequential(layers)

        iterated = list(seq)
        assert iterated == layers


class TestSequentialForward:
    """Tests for Sequential forward pass."""

    def test_forward_basic(self):
        """Forward pass applies layers in order."""
        seq = Sequential([
            Linear(10, 5),
            ReLU(),
            Linear(5, 2)
        ])
        x = Tensor(np.random.randn(32, 10))
        y = seq(x)
        assert y.shape == (32, 2)

    def test_forward_single_layer(self):
        """Sequential with single layer works."""
        seq = Sequential([Linear(10, 5)])
        x = Tensor(np.random.randn(32, 10))
        y = seq(x)
        assert y.shape == (32, 5)

    def test_forward_only_activation(self):
        """Sequential with only activation works."""
        seq = Sequential([ReLU()])
        x = Tensor(np.random.randn(32, 10))
        y = seq(x)
        assert y.shape == (32, 10)

    def test_forward_correctness(self):
        """Forward pass produces correct output."""
        linear1 = Linear(3, 4)
        act = ReLU()
        linear2 = Linear(4, 2)

        seq = Sequential([linear1, act, linear2])
        x = Tensor(np.random.randn(1, 3))

        h1 = np.maximum(0, x.data @ linear1.W.data + linear1.b.data)
        expected = h1 @ linear2.W.data + linear2.b.data

        y = seq(x)
        assert np.allclose(y.data, expected)


class TestSequentialParameters:
    """Tests for Sequential parameters() method."""

    def test_parameters_collects_all(self):
        """parameters() returns all layer parameters."""
        seq = Sequential([
            Linear(10, 5),
            ReLU(),
            Linear(5, 2)
        ])
        params = seq.parameters()
        assert len(params) == 4

    def test_parameters_order(self):
        """Parameters are in layer order."""
        linear1 = Linear(10, 5)
        linear2 = Linear(5, 2)
        seq = Sequential([linear1, ReLU(), linear2])

        params = seq.parameters()
        assert id(params[0]) == id(linear1.W)
        assert id(params[1]) == id(linear1.b)
        assert id(params[2]) == id(linear2.W)
        assert id(params[3]) == id(linear2.b)

    def test_parameters_empty_sequential(self):
        """Empty Sequential returns empty parameters list."""
        seq = Sequential()
        assert seq.parameters() == []

    def test_parameters_no_bias(self):
        """Parameters work correctly without bias."""
        seq = Sequential([
            Linear(10, 5, bias=False),
            ReLU(),
            Linear(5, 2, bias=False)
        ])
        params = seq.parameters()
        assert len(params) == 2


class TestSequentialBackward:
    """Tests for Sequential backward pass."""

    def test_backward_gradient_shapes(self):
        """All gradients have correct shapes."""
        seq = Sequential([
            Linear(10, 5),
            ReLU(),
            Linear(5, 2)
        ])
        x = Tensor(np.random.randn(32, 10), requires_grad=True)
        y = seq(x)
        loss = y.sum()
        loss.backward()

        assert x.grad.shape == (32, 10)

        assert seq[0].W.grad.shape == (10, 5)
        assert seq[0].b.grad.shape == (5,)

        assert seq[2].W.grad.shape == (5, 2)
        assert seq[2].b.grad.shape == (2,)

    def test_backward_numerical_gradient(self):
        """Gradients match finite differences."""
        seq = Sequential([
            Linear(4, 3),
            Swish(),
            Linear(3, 2)
        ])
        x_data = np.random.randn(8, 4)

        x = Tensor(x_data, requires_grad=True)
        y = seq(x)
        y.backward(np.ones_like(y.data))

        def func_x(x_val):
            t = Tensor(x_val.reshape(8, 4), requires_grad=False)
            return seq(t).data.sum()

        num_grad = numerical_gradient(func_x, x_data.flatten())
        assert np.allclose(x.grad.flatten(), num_grad, rtol=1e-4, atol=1e-4)


class TestSequentialZeroGrad:
    """Tests for Sequential zero_grad()."""

    def test_zero_grad_all_layers(self):
        """zero_grad() clears all layer gradients."""
        seq = Sequential([
            Linear(10, 5),
            ReLU(),
            Linear(5, 2)
        ])
        x = Tensor(np.random.randn(32, 10), requires_grad=True)
        y = seq(x)
        y.sum().backward()

        assert seq[0].W.grad is not None
        assert seq[2].W.grad is not None

        seq.zero_grad()

        assert seq[0].W.grad is None
        assert seq[0].b.grad is None
        assert seq[2].W.grad is None
        assert seq[2].b.grad is None


class TestSequentialRepr:
    """Tests for Sequential __repr__."""

    def test_repr_format(self):
        """__repr__ shows layer structure."""
        seq = Sequential([
            Linear(10, 5),
            ReLU(),
            Linear(5, 2)
        ])
        repr_str = repr(seq)

        assert "Sequential" in repr_str
        assert "Linear" in repr_str
        assert "ReLU" in repr_str
        assert "(0)" in repr_str
        assert "(1)" in repr_str
        assert "(2)" in repr_str


class TestMLPForward:
    """Integration tests for MLP-like architectures."""

    def test_mlp_2d_data(self):
        """MLP works with 2D data (like energy function input)."""
        model = Sequential([
            Linear(2, 128),
            Swish(),
            Linear(128, 128),
            Swish(),
            Linear(128, 1)
        ])

        x = Tensor(np.random.randn(64, 2), requires_grad=True)
        y = model(x)

        assert y.shape == (64, 1)

    def test_mlp_tabular_data(self):
        """MLP works with tabular data."""
        model = Sequential([
            Linear(29, 256),
            Swish(),
            Linear(256, 256),
            Swish(),
            Linear(256, 1)
        ])

        x = Tensor(np.random.randn(128, 29), requires_grad=True)
        y = model(x)

        assert y.shape == (128, 1)

    def test_mlp_different_activations(self):
        """MLP works with different activation functions."""
        for act_class in [ReLU, Swish, Softplus]:
            model = Sequential([
                Linear(10, 32),
                act_class(),
                Linear(32, 1)
            ])

            x = Tensor(np.random.randn(16, 10), requires_grad=True)
            y = model(x)
            y.sum().backward()

            assert y.shape == (16, 1)
            assert x.grad is not None


class TestGradientFlowThroughNetwork:
    """Tests for gradient flow through complete networks."""

    def test_gradient_reaches_all_layers(self):
        """Gradients reach all layers in deep network."""
        model = Sequential([
            Linear(10, 32),
            Swish(),
            Linear(32, 32),
            Swish(),
            Linear(32, 32),
            Swish(),
            Linear(32, 1)
        ])

        x = Tensor(np.random.randn(16, 10), requires_grad=True)
        y = model(x)
        y.sum().backward()

        for layer in model:
            if isinstance(layer, Linear):
                assert layer.W.grad is not None
                assert layer.b.grad is not None
                assert np.isfinite(layer.W.grad).all()
                assert np.isfinite(layer.b.grad).all()

    def test_gradient_nonzero(self):
        """Gradients are non-zero for reasonable inputs."""
        model = Sequential([
            Linear(10, 32),
            Swish(),
            Linear(32, 1)
        ])

        x = Tensor(np.random.randn(16, 10), requires_grad=True)
        y = model(x)
        y.sum().backward()

        assert np.abs(model[0].W.grad).max() > 0
        assert np.abs(model[2].W.grad).max() > 0

    def test_input_gradient_for_langevin(self):
        """Verify we can get gradients w.r.t. input (needed for Langevin)."""
        model = Sequential([
            Linear(2, 64),
            Swish(),
            Linear(64, 64),
            Swish(),
            Linear(64, 1)
        ])

        x = Tensor(np.random.randn(32, 2), requires_grad=True)
        energy = model(x)
        energy.sum().backward()

        assert x.grad is not None
        assert x.grad.shape == (32, 2)
        assert np.isfinite(x.grad).all()


class TestEnergyFunctionPattern:
    """Tests for energy function patterns from the README."""

    def test_recommended_2d_architecture(self):
        """Test recommended architecture for 2D data."""
        model = Sequential([
            Linear(2, 128),
            Swish(),
            Linear(128, 128),
            Swish(),
            Linear(128, 1)
        ])

        x = Tensor(np.random.randn(64, 2), requires_grad=True)
        energy = model(x)

        assert energy.shape == (64, 1)

        energy.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (64, 2)

    def test_recommended_tabular_architecture(self):
        """Test recommended architecture for tabular data."""
        d = 29
        model = Sequential([
            Linear(d, 256),
            Swish(),
            Linear(256, 256),
            Swish(),
            Linear(256, 1)
        ])

        x = Tensor(np.random.randn(128, d), requires_grad=True)
        energy = model(x)

        assert energy.shape == (128, 1)

        energy.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (128, d)

    def test_energy_scalar_output(self):
        """Energy function produces scalar per sample."""
        model = Sequential([
            Linear(10, 64),
            Swish(),
            Linear(64, 1)
        ])

        x = Tensor(np.random.randn(32, 10))
        energy = model(x)

        assert energy.shape == (32, 1)


class TestNumericalGradients:
    """Comprehensive numerical gradient tests."""

    def test_linear_numerical_gradient(self):
        """Linear layer gradients match finite differences."""
        layer = Linear(8, 4)
        assert check_layer_gradient(layer, (16, 8))

    def test_relu_numerical_gradient(self):
        """ReLU gradients match finite differences (away from zero)."""
        layer = ReLU()
        x_data = np.random.randn(16, 8)
        x_data = np.where(np.abs(x_data) < 0.1, 0.5 * np.sign(x_data), x_data)

        x = Tensor(x_data, requires_grad=True)
        y = layer(x)
        y.backward(np.ones_like(y.data))

        def func(x_val):
            t = Tensor(x_val.reshape(16, 8), requires_grad=False)
            return layer(t).data.sum()

        num_grad = numerical_gradient(func, x_data.flatten())
        assert np.allclose(x.grad.flatten(), num_grad, rtol=1e-4, atol=1e-4)

    def test_swish_numerical_gradient(self):
        """Swish gradients match finite differences."""
        layer = Swish()
        assert check_layer_gradient(layer, (16, 8))

    def test_softplus_numerical_gradient(self):
        """Softplus gradients match finite differences."""
        layer = Softplus()
        assert check_layer_gradient(layer, (16, 8))

    def test_sequential_numerical_gradient(self):
        """Sequential network gradients match finite differences."""
        seq = Sequential([
            Linear(4, 8),
            Swish(),
            Linear(8, 2)
        ])

        x_data = np.random.randn(8, 4)
        x = Tensor(x_data, requires_grad=True)
        y = seq(x)
        y.backward(np.ones_like(y.data))

        def func(x_val):
            t = Tensor(x_val.reshape(8, 4), requires_grad=False)
            return seq(t).data.sum()

        num_grad = numerical_gradient(func, x_data.flatten())
        assert np.allclose(x.grad.flatten(), num_grad, rtol=1e-4, atol=1e-4)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_sample(self):
        """Network works with single sample."""
        model = Sequential([Linear(10, 5), ReLU(), Linear(5, 1)])
        x = Tensor(np.random.randn(1, 10), requires_grad=True)
        y = model(x)
        y.sum().backward()

        assert y.shape == (1, 1)
        assert x.grad.shape == (1, 10)

    def test_large_batch(self):
        """Network works with large batch."""
        model = Sequential([Linear(10, 5), ReLU(), Linear(5, 1)])
        x = Tensor(np.random.randn(1024, 10), requires_grad=True)
        y = model(x)
        y.sum().backward()

        assert y.shape == (1024, 1)
        assert x.grad.shape == (1024, 10)

    def test_wide_network(self):
        """Wide network (many neurons) works correctly."""
        model = Sequential([
            Linear(10, 1024),
            Swish(),
            Linear(1024, 1)
        ])

        x = Tensor(np.random.randn(32, 10), requires_grad=True)
        y = model(x)
        y.sum().backward()

        assert y.shape == (32, 1)
        assert np.isfinite(x.grad).all()

    def test_deep_network(self):
        """Deep network works correctly."""
        layers = []
        for _ in range(10):
            layers.append(Linear(32, 32))
            layers.append(Swish())
        layers.append(Linear(32, 1))

        model = Sequential(layers)

        x = Tensor(np.random.randn(16, 32), requires_grad=True)
        y = model(x)
        y.sum().backward()

        assert y.shape == (16, 1)
        assert np.isfinite(x.grad).all()

    def test_zero_input(self):
        """Network handles zero input."""
        model = Sequential([Linear(10, 5), Swish(), Linear(5, 1)])
        x = Tensor(np.zeros((16, 10)), requires_grad=True)
        y = model(x)
        y.sum().backward()

        assert np.isfinite(y.data).all()
        assert np.isfinite(x.grad).all()

    def test_large_input_values(self):
        """Network handles large input values."""
        model = Sequential([Linear(10, 5), Swish(), Linear(5, 1)])
        x = Tensor(np.random.randn(16, 10) * 100, requires_grad=True)
        y = model(x)
        y.sum().backward()

        assert np.isfinite(y.data).all()
        assert np.isfinite(x.grad).all()


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_swish_large_values(self):
        """Swish handles large values without overflow."""
        model = Sequential([Linear(2, 4), Swish(), Linear(4, 1)])

        x = Tensor(np.array([[100.0, 100.0], [-100.0, -100.0]]), requires_grad=True)
        y = model(x)
        y.sum().backward()

        assert np.isfinite(y.data).all()
        assert np.isfinite(x.grad).all()

    def test_softplus_large_values(self):
        """Softplus handles large values without overflow."""
        model = Sequential([Linear(2, 4), Softplus(), Linear(4, 1)])

        x = Tensor(np.array([[100.0, 100.0], [-100.0, -100.0]]), requires_grad=True)
        y = model(x)
        y.sum().backward()

        assert np.isfinite(y.data).all()
        assert np.isfinite(x.grad).all()

    def test_repeated_forward_backward(self):
        """Network remains stable over many forward-backward passes."""
        model = Sequential([Linear(10, 32), Swish(), Linear(32, 1)])

        for _ in range(100):
            x = Tensor(np.random.randn(16, 10), requires_grad=True)
            y = model(x)
            y.sum().backward()
            model.zero_grad()

            assert np.isfinite(y.data).all()


class TestCreateMLP:
    """Tests for _create_mlp helper function."""

    def test_basic_mlp(self):
        """Create basic MLP."""
        model = _create_mlp(
            input_dim=10,
            hidden_dims=[64, 32],
            output_dim=1
        )

        x = Tensor(np.random.randn(16, 10))
        y = model(x)
        assert y.shape == (16, 1)

    def test_mlp_with_relu(self):
        """Create MLP with ReLU activation."""
        model = _create_mlp(
            input_dim=10,
            hidden_dims=[32],
            output_dim=5,
            activation="relu"
        )

        x = Tensor(np.random.randn(16, 10))
        y = model(x)
        assert y.shape == (16, 5)

    def test_mlp_with_output_activation(self):
        """Create MLP with output activation."""
        model = _create_mlp(
            input_dim=10,
            hidden_dims=[32],
            output_dim=1,
            activation="sigmoid",
            output_activation=True
        )

        x = Tensor(np.random.randn(16, 10))
        y = model(x)

        assert (y.data > 0).all()
        assert (y.data < 1).all()

    def test_mlp_invalid_activation(self):
        """Invalid activation raises ValueError."""
        with pytest.raises(ValueError):
            _create_mlp(
                input_dim=10,
                hidden_dims=[32],
                output_dim=1,
                activation="invalid"
            )

    def test_mlp_layer_count(self):
        """MLP has correct number of layers."""
        model = _create_mlp(
            input_dim=10,
            hidden_dims=[64, 32],
            output_dim=1,
            activation="swish"
        )

        assert len(model) == 5


class TestParameterCounting:
    """Tests for parameter counting."""

    def test_linear_parameter_count(self):
        """Linear layer has correct parameter count."""
        layer = Linear(10, 5)
        params = layer.parameters()

        total_params = sum(p.data.size for p in params)
        expected = 10 * 5 + 5
        assert total_params == expected

    def test_linear_no_bias_parameter_count(self):
        """Linear without bias has correct parameter count."""
        layer = Linear(10, 5, bias=False)
        params = layer.parameters()

        total_params = sum(p.data.size for p in params)
        expected = 10 * 5
        assert total_params == expected

    def test_sequential_parameter_count(self):
        """Sequential has correct total parameter count."""
        model = Sequential([
            Linear(10, 64),
            Swish(),
            Linear(64, 32),
            Swish(),
            Linear(32, 1)
        ])

        params = model.parameters()
        total_params = sum(p.data.size for p in params)
        expected = 10 * 64 + 64 + 64 * 32 + 32 + 32 * 1 + 1
        assert total_params == expected
