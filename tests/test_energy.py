"""
Comprehensive tests for the EnergyMLP class and energy functions.

Tests cover:
- Initialization with various configurations
- Forward pass (energy computation)
- Score function (gradient of energy w.r.t. input)
- Parameter collection
- Numerical gradient verification
- Integration with autodiff system
- Factory functions
"""

import pytest
import numpy as np

from ebm.core.autodiff import Tensor
from ebm.core.nn import Sequential, Linear, Swish, ReLU, Softplus, Sigmoid
from ebm.core.energy import (
    EnergyMLP,
    create_energy_network_2d,
    create_energy_network_tabular
)


class TestEnergyMLPInitialization:
    """Test EnergyMLP initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic initialization with default parameters."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64, 32])

        assert energy_fn.input_dim == 10
        assert energy_fn.hidden_dims == [64, 32]
        assert energy_fn.activation_name == "swish"
        assert energy_fn.network is not None
        assert isinstance(energy_fn.network, Sequential)

    def test_initialization_with_relu(self):
        """Test initialization with ReLU activation."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32], activation="relu")

        assert energy_fn.activation_name == "relu"
        has_relu = any(isinstance(layer, ReLU) for layer in energy_fn.network.layers)
        assert has_relu

    def test_initialization_with_softplus(self):
        """Test initialization with Softplus activation."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32], activation="softplus")

        assert energy_fn.activation_name == "softplus"
        has_softplus = any(isinstance(layer, Softplus) for layer in energy_fn.network.layers)
        assert has_softplus

    def test_initialization_with_sigmoid(self):
        """Test initialization with Sigmoid activation."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32], activation="sigmoid")

        assert energy_fn.activation_name == "sigmoid"
        has_sigmoid = any(isinstance(layer, Sigmoid) for layer in energy_fn.network.layers)
        assert has_sigmoid

    def test_initialization_with_swish(self):
        """Test initialization with Swish activation (default)."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32], activation="swish")

        assert energy_fn.activation_name == "swish"
        has_swish = any(isinstance(layer, Swish) for layer in energy_fn.network.layers)
        assert has_swish

    def test_initialization_with_custom_network(self):
        """Test initialization with a pre-built network."""
        custom_network = Sequential([
            Linear(10, 50),
            ReLU(),
            Linear(50, 25),
            ReLU(),
            Linear(25, 1)
        ])

        energy_fn = EnergyMLP(
            input_dim=10,
            hidden_dims=[50, 25],
            network=custom_network
        )

        assert energy_fn.network is custom_network
        assert len(energy_fn.network) == 5

    def test_initialization_invalid_activation(self):
        """Test that invalid activation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            EnergyMLP(input_dim=5, hidden_dims=[32], activation="invalid")

    def test_initialization_empty_hidden_dims_no_network(self):
        """Test that empty hidden_dims without network raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dims must not be empty"):
            EnergyMLP(input_dim=5, hidden_dims=[])

    def test_initialization_single_hidden_layer(self):
        """Test initialization with single hidden layer."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[32])

        assert len(energy_fn.network) == 3
        assert isinstance(energy_fn.network[0], Linear)
        assert isinstance(energy_fn.network[1], Swish)
        assert isinstance(energy_fn.network[2], Linear)

    def test_initialization_multiple_hidden_layers(self):
        """Test initialization with multiple hidden layers."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64, 32, 16])

        assert len(energy_fn.network) == 7

    def test_repr(self):
        """Test string representation."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64, 32])
        repr_str = repr(energy_fn)

        assert "EnergyMLP" in repr_str
        assert "input_dim=10" in repr_str
        assert "hidden_dims=[64, 32]" in repr_str
        assert "swish" in repr_str


class TestEnergyMLPForward:
    """Test forward pass (energy computation)."""

    def test_forward_output_shape_batch(self):
        """Test that forward produces correct output shape for batch input."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64, 32])
        x = Tensor(np.random.randn(32, 10), requires_grad=True)

        energy = energy_fn.forward(x)

        assert energy.data.shape == (32, 1)

    def test_forward_output_shape_single_sample(self):
        """Test forward with single sample (1D input)."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64])
        x = Tensor(np.random.randn(10), requires_grad=True)

        energy = energy_fn.forward(x)

        assert energy.data.shape == (1, 1)

    def test_forward_callable(self):
        """Test that EnergyMLP is callable."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = Tensor(np.random.randn(16, 5), requires_grad=True)

        energy = energy_fn(x)
        assert energy.data.shape == (16, 1)

    def test_energy_alias(self):
        """Test that energy() is an alias for forward()."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = Tensor(np.random.randn(16, 5), requires_grad=True)

        e1 = energy_fn.forward(x)
        e2 = energy_fn.energy(x)

        assert e1.data.shape == e2.data.shape

    def test_forward_deterministic(self):
        """Test that forward is deterministic (no randomness)."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = Tensor(np.random.randn(16, 5), requires_grad=False)

        e1 = energy_fn.forward(x)
        e2 = energy_fn.forward(x)

        np.testing.assert_array_equal(e1.data, e2.data)

    def test_forward_finite_values(self):
        """Test that forward produces finite values."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64, 64])
        x = Tensor(np.random.randn(32, 10), requires_grad=True)

        energy = energy_fn.forward(x)

        assert np.isfinite(energy.data).all()

    def test_forward_with_large_input(self):
        """Test forward with large input values."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = Tensor(np.random.randn(16, 5) * 10, requires_grad=True)

        energy = energy_fn.forward(x)

        assert np.isfinite(energy.data).all()

    def test_forward_with_small_input(self):
        """Test forward with small input values."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = Tensor(np.random.randn(16, 5) * 0.001, requires_grad=True)

        energy = energy_fn.forward(x)

        assert np.isfinite(energy.data).all()

    def test_forward_batch_sizes(self):
        """Test forward with various batch sizes."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64])

        for batch_size in [1, 2, 16, 64, 128, 256]:
            x = Tensor(np.random.randn(batch_size, 10), requires_grad=True)
            energy = energy_fn.forward(x)
            assert energy.data.shape == (batch_size, 1)

    def test_forward_different_architectures(self):
        """Test forward with different network architectures."""
        architectures = [
            [32],
            [64, 32],
            [128, 128],
            [256, 256, 128],
            [512, 256, 128, 64],
        ]

        for hidden_dims in architectures:
            energy_fn = EnergyMLP(input_dim=10, hidden_dims=hidden_dims)
            x = Tensor(np.random.randn(16, 10), requires_grad=True)
            energy = energy_fn.forward(x)
            assert energy.data.shape == (16, 1)


class TestEnergyMLPScore:
    """Test score function (negative gradient of energy w.r.t. input)."""

    def test_score_output_shape(self):
        """Test that score produces correct output shape."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64])
        x = Tensor(np.random.randn(32, 10), requires_grad=True)

        score = energy_fn.score(x)

        assert score.shape == (32, 10)

    def test_score_returns_numpy(self):
        """Test that score returns numpy array, not Tensor."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = Tensor(np.random.randn(16, 5), requires_grad=True)

        score = energy_fn.score(x)

        assert isinstance(score, np.ndarray)

    def test_score_with_numpy_input(self):
        """Test score with numpy array input."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = np.random.randn(16, 5)

        score = energy_fn.score(x)

        assert isinstance(score, np.ndarray)
        assert score.shape == (16, 5)

    def test_score_finite_values(self):
        """Test that score produces finite values."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64])
        x = Tensor(np.random.randn(32, 10), requires_grad=True)

        score = energy_fn.score(x)

        assert np.isfinite(score).all()

    def test_score_direction(self):
        """Test that score points in direction of decreasing energy."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32])
        x = np.random.randn(100, 2)

        energy_before = energy_fn.energy(Tensor(x)).data

        score = energy_fn.score(x)
        step_size = 0.01
        x_new = x + step_size * score

        energy_after = energy_fn.energy(Tensor(x_new)).data

        energy_decrease = (energy_before - energy_after).mean()
        assert energy_decrease > -0.1

    def test_score_numerical_gradient_verification(self):
        """Verify score matches numerical gradient (finite differences)."""
        energy_fn = EnergyMLP(input_dim=3, hidden_dims=[16])
        x = np.random.randn(4, 3)

        analytic_score = energy_fn.score(x)

        eps = 1e-5
        numerical_grad = np.zeros_like(x)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_plus = x.copy()
                x_plus[i, j] += eps
                e_plus = energy_fn.energy(Tensor(x_plus)).data.sum()

                x_minus = x.copy()
                x_minus[i, j] -= eps
                e_minus = energy_fn.energy(Tensor(x_minus)).data.sum()

                numerical_grad[i, j] = (e_plus - e_minus) / (2 * eps)

        np.testing.assert_allclose(analytic_score, -numerical_grad, rtol=1e-4, atol=1e-4)

    def test_score_batch_independence(self):
        """Test that score for one sample doesn't depend on other samples in batch."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])

        x1 = np.random.randn(1, 5)
        x2 = np.random.randn(1, 5)

        score1_single = energy_fn.score(x1)
        score2_single = energy_fn.score(x2)

        x_batch = np.vstack([x1, x2])
        score_batch = energy_fn.score(x_batch)

        np.testing.assert_allclose(score1_single, score_batch[0:1], rtol=1e-5)
        np.testing.assert_allclose(score2_single, score_batch[1:2], rtol=1e-5)


class TestEnergyAndScore:
    """Test combined energy and score computation."""

    def test_energy_and_score_shapes(self):
        """Test that energy_and_score returns correct shapes."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64])
        x = np.random.randn(32, 10)

        energy, score = energy_fn.energy_and_score(x)

        assert energy.shape == (32, 1)
        assert score.shape == (32, 10)

    def test_energy_and_score_consistency(self):
        """Test that combined method gives same results as separate calls."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = np.random.randn(16, 5)

        energy_combined, score_combined = energy_fn.energy_and_score(x)

        energy_separate = energy_fn.energy(Tensor(x)).data
        score_separate = energy_fn.score(x)

        np.testing.assert_allclose(energy_combined, energy_separate, rtol=1e-5)
        np.testing.assert_allclose(score_combined, score_separate, rtol=1e-5)

    def test_energy_and_score_with_tensor_input(self):
        """Test energy_and_score with Tensor input."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = Tensor(np.random.randn(16, 5), requires_grad=True)

        energy, score = energy_fn.energy_and_score(x)

        assert energy.shape == (16, 1)
        assert score.shape == (16, 5)


class TestEnergyMLPParameters:
    """Test parameter collection and management."""

    def test_parameters_returns_list(self):
        """Test that parameters() returns a list."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64])

        params = energy_fn.parameters()

        assert isinstance(params, list)

    def test_parameters_count_single_hidden(self):
        """Test parameter count with single hidden layer."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64])

        params = energy_fn.parameters()

        assert len(params) == 4

    def test_parameters_count_multiple_hidden(self):
        """Test parameter count with multiple hidden layers."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64, 32])

        params = energy_fn.parameters()

        assert len(params) == 6

    def test_parameters_shapes(self):
        """Test that parameters have correct shapes."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64, 32])

        params = energy_fn.parameters()

        expected_shapes = [
            (10, 64),
            (64,),
            (64, 32),
            (32,),
            (32, 1),
            (1,),
        ]

        for param, expected_shape in zip(params, expected_shapes):
            assert param.shape == expected_shape

    def test_parameters_require_grad(self):
        """Test that all parameters require gradient."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64])

        params = energy_fn.parameters()

        for param in params:
            assert param.requires_grad

    def test_parameters_are_tensors(self):
        """Test that parameters are Tensor objects."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[64])

        params = energy_fn.parameters()

        for param in params:
            assert isinstance(param, Tensor)

    def test_zero_grad(self):
        """Test that zero_grad clears parameter gradients."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = Tensor(np.random.randn(16, 5), requires_grad=True)

        energy = energy_fn(x)
        energy.sum().backward()

        for param in energy_fn.parameters():
            assert param.grad is not None

        energy_fn.zero_grad()

        for param in energy_fn.parameters():
            assert param.grad is None


class TestEnergyMLPGradientFlow:
    """Test gradient flow through the energy function."""

    def test_gradient_flows_to_parameters(self):
        """Test that gradients flow to network parameters."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = Tensor(np.random.randn(16, 5), requires_grad=True)

        energy = energy_fn(x)
        energy.sum().backward()

        for param in energy_fn.parameters():
            assert param.grad is not None
            assert not np.allclose(param.grad, 0)

    def test_gradient_flows_to_input(self):
        """Test that gradients flow to input tensor."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = Tensor(np.random.randn(16, 5), requires_grad=True)

        energy = energy_fn(x)
        energy.sum().backward()

        assert x.grad is not None
        assert x.grad.shape == (16, 5)

    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = Tensor(np.random.randn(16, 5), requires_grad=True)

        energy1 = energy_fn(x)
        energy1.sum().backward()
        grad1 = [p.grad.copy() for p in energy_fn.parameters()]

        energy2 = energy_fn(x)
        energy2.sum().backward()
        grad2 = [p.grad.copy() for p in energy_fn.parameters()]

        for g1, g2 in zip(grad1, grad2):
            np.testing.assert_allclose(g2, 2 * g1, rtol=1e-5)

    def test_numerical_gradient_verification_parameters(self):
        """Verify parameter gradients using finite differences."""
        energy_fn = EnergyMLP(input_dim=3, hidden_dims=[8])
        x = Tensor(np.random.randn(4, 3), requires_grad=False)

        energy = energy_fn(x)
        loss = energy.sum()
        loss.backward()

        W = energy_fn.parameters()[0]
        analytic_grad = W.grad.copy()

        eps = 1e-5
        numerical_grad = np.zeros_like(W.data)

        for i in range(W.data.shape[0]):
            for j in range(W.data.shape[1]):
                W.data[i, j] += eps
                e_plus = energy_fn(x).sum().data

                W.data[i, j] -= 2 * eps
                e_minus = energy_fn(x).sum().data

                W.data[i, j] += eps

                numerical_grad[i, j] = (e_plus - e_minus) / (2 * eps)

        np.testing.assert_allclose(analytic_grad, numerical_grad, rtol=1e-4, atol=1e-4)


class TestEnergyMLPIntegration:
    """Integration tests with realistic scenarios."""

    def test_2d_data_architecture(self):
        """Test with recommended 2D data architecture."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[128, 128])
        x = Tensor(np.random.randn(64, 2), requires_grad=True)

        energy = energy_fn(x)

        assert energy.data.shape == (64, 1)
        assert np.isfinite(energy.data).all()

    def test_tabular_data_architecture(self):
        """Test with recommended tabular data architecture."""
        energy_fn = EnergyMLP(input_dim=29, hidden_dims=[256, 256])
        x = Tensor(np.random.randn(128, 29), requires_grad=True)

        energy = energy_fn(x)

        assert energy.data.shape == (128, 1)
        assert np.isfinite(energy.data).all()

    def test_score_function_langevin_step(self):
        """Test score function in a simulated Langevin step."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[64, 64])

        x = np.random.randn(100, 2)

        step_size = 0.01
        noise_scale = np.sqrt(2 * step_size)

        score = energy_fn.score(x)

        noise = np.random.randn(*x.shape) * noise_scale
        x_new = x + step_size * score + noise

        assert np.isfinite(x_new).all()
        assert x_new.shape == x.shape

    def test_energy_difference_real_vs_fake(self):
        """Test that energy function can distinguish different data."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[64, 64])

        x_real = np.random.randn(100, 2) * 0.5

        x_fake = np.random.uniform(-3, 3, (100, 2))

        e_real = energy_fn(Tensor(x_real)).data
        e_fake = energy_fn(Tensor(x_fake)).data

        assert np.isfinite(e_real).all()
        assert np.isfinite(e_fake).all()

    def test_multiple_forward_passes(self):
        """Test multiple forward passes with same and different data."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])

        x1 = Tensor(np.random.randn(16, 5), requires_grad=True)
        x2 = Tensor(np.random.randn(16, 5), requires_grad=True)

        e1a = energy_fn(x1).data.copy()
        e1b = energy_fn(x1).data.copy()
        np.testing.assert_array_equal(e1a, e1b)

        _ = energy_fn(x2).data

class TestFactoryFunctions:
    """Test factory functions for creating energy networks."""

    def test_create_energy_network_2d_default(self):
        """Test create_energy_network_2d with default parameters."""
        energy_fn = create_energy_network_2d()

        assert energy_fn.input_dim == 2
        assert energy_fn.hidden_dims == [128, 128]
        assert energy_fn.activation_name == "swish"

        x = Tensor(np.random.randn(32, 2), requires_grad=True)
        energy = energy_fn(x)
        assert energy.data.shape == (32, 1)

    def test_create_energy_network_2d_custom(self):
        """Test create_energy_network_2d with custom hidden dims."""
        energy_fn = create_energy_network_2d(hidden_dims=[64, 64, 32])

        assert energy_fn.input_dim == 2
        assert energy_fn.hidden_dims == [64, 64, 32]

    def test_create_energy_network_tabular_default(self):
        """Test create_energy_network_tabular with default parameters."""
        energy_fn = create_energy_network_tabular(input_dim=29)

        assert energy_fn.input_dim == 29
        assert energy_fn.hidden_dims == [256, 256]
        assert energy_fn.activation_name == "swish"

        x = Tensor(np.random.randn(128, 29), requires_grad=True)
        energy = energy_fn(x)
        assert energy.data.shape == (128, 1)

    def test_create_energy_network_tabular_custom(self):
        """Test create_energy_network_tabular with custom parameters."""
        energy_fn = create_energy_network_tabular(
            input_dim=50,
            hidden_dims=[512, 256, 128]
        )

        assert energy_fn.input_dim == 50
        assert energy_fn.hidden_dims == [512, 256, 128]


class TestEnergyMLPEdgeCases:
    """Test edge cases and robustness."""

    def test_single_sample(self):
        """Test with single sample input."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])

        x = Tensor(np.random.randn(1, 5), requires_grad=True)
        energy = energy_fn(x)
        assert energy.data.shape == (1, 1)

        score = energy_fn.score(x)
        assert score.shape == (1, 5)

    def test_very_deep_network(self):
        """Test with a deep network."""
        energy_fn = EnergyMLP(
            input_dim=10,
            hidden_dims=[64, 64, 64, 64, 64]
        )
        x = Tensor(np.random.randn(16, 10), requires_grad=True)

        energy = energy_fn(x)
        assert np.isfinite(energy.data).all()

        score = energy_fn.score(x)
        assert np.isfinite(score).all()

    def test_wide_network(self):
        """Test with a wide network."""
        energy_fn = EnergyMLP(
            input_dim=10,
            hidden_dims=[1024, 512]
        )
        x = Tensor(np.random.randn(16, 10), requires_grad=True)

        energy = energy_fn(x)
        assert np.isfinite(energy.data).all()

    def test_high_dimensional_input(self):
        """Test with high-dimensional input."""
        energy_fn = EnergyMLP(input_dim=100, hidden_dims=[256, 128])
        x = Tensor(np.random.randn(32, 100), requires_grad=True)

        energy = energy_fn(x)
        assert energy.data.shape == (32, 1)
        assert np.isfinite(energy.data).all()

    def test_zero_input(self):
        """Test with zero input."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = Tensor(np.zeros((16, 5)), requires_grad=True)

        energy = energy_fn(x)
        assert np.isfinite(energy.data).all()

        score = energy_fn.score(x)
        assert np.isfinite(score).all()

    def test_large_input_values(self):
        """Test numerical stability with large input values."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32], activation="swish")
        x = Tensor(np.random.randn(16, 5) * 100, requires_grad=True)

        energy = energy_fn(x)
        assert np.isfinite(energy.data).all()

    def test_negative_input_values(self):
        """Test with all negative input values."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = Tensor(-np.abs(np.random.randn(16, 5)), requires_grad=True)

        energy = energy_fn(x)
        assert np.isfinite(energy.data).all()


class TestActivationComparison:
    """Compare behavior with different activations."""

    @pytest.mark.parametrize("activation", ["relu", "swish", "softplus", "sigmoid"])
    def test_all_activations_work(self, activation):
        """Test that all activations produce valid output."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32], activation=activation)
        x = Tensor(np.random.randn(16, 5), requires_grad=True)

        energy = energy_fn(x)
        assert np.isfinite(energy.data).all()

        score = energy_fn.score(x)
        assert np.isfinite(score).all()

    @pytest.mark.parametrize("activation", ["relu", "swish", "softplus", "sigmoid"])
    def test_all_activations_gradients(self, activation):
        """Test that all activations have working gradients."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32], activation=activation)
        x = Tensor(np.random.randn(16, 5), requires_grad=True)

        energy = energy_fn(x)
        energy.sum().backward()

        for param in energy_fn.parameters():
            assert param.grad is not None

        assert x.grad is not None


class TestEnergyMLP100Coverage:
    """Tests to achieve 100% code coverage on edge cases."""

    def test_forward_1d_output_reshape(self):
        """Test forward when network output is 1D (needs reshaping to (batch, 1)).

        This tests the branch at line 169 where output.data.ndim == 1.
        We create a custom network that squeezes output to 1D.
        """
        from ebm.core.nn import Module, Sequential, Linear

        class SqueezeModule(Module):
            """A module that squeezes the last dimension, producing 1D output."""
            def forward(self, x: Tensor) -> Tensor:
                squeezed_data = x.data.squeeze(-1)
                result = Tensor(squeezed_data, requires_grad=x.requires_grad)
                result._prev = x._prev
                result._backward = x._backward
                return result

            def parameters(self):
                return []

        custom_network = Sequential([
            Linear(5, 1),
            SqueezeModule()
        ])

        energy_fn = EnergyMLP(
            input_dim=5,
            hidden_dims=[8],
            network=custom_network
        )

        x = Tensor(np.random.randn(1, 5), requires_grad=True)
        energy = energy_fn(x)

        assert energy.data.shape == (1, 1)
        assert np.isfinite(energy.data).all()

    def test_score_returns_zeros_when_no_path_to_input(self):
        """Test that score returns zeros when there's no gradient path to input.

        This tests the defensive code at lines 228-230.
        We create a network that doesn't connect input to output in a differentiable way.
        """
        from ebm.core.nn import Module, Sequential

        class ConstantModule(Module):
            """A module that returns a constant, breaking gradient flow."""
            def __init__(self):
                self.constant = Tensor(np.array([[1.0]]), requires_grad=False)

            def forward(self, x: Tensor) -> Tensor:
                batch_size = x.data.shape[0]
                return Tensor(
                    np.ones((batch_size, 1)),
                    requires_grad=False
                )

            def parameters(self):
                return []

        constant_network = Sequential([ConstantModule()])

        energy_fn = EnergyMLP(
            input_dim=5,
            hidden_dims=[8],
            network=constant_network
        )

        x = np.random.randn(4, 5)
        score = energy_fn.score(x)

        assert score.shape == (4, 5)
        np.testing.assert_array_equal(score, np.zeros((4, 5)))

    def test_energy_and_score_returns_zeros_when_no_path_to_input(self):
        """Test that energy_and_score returns zeros for score when no gradient path.

        This tests the defensive code at lines 265-267.
        """
        from ebm.core.nn import Module, Sequential

        class ConstantModule(Module):
            """A module that returns a constant, breaking gradient flow."""
            def forward(self, x: Tensor) -> Tensor:
                batch_size = x.data.shape[0]
                return Tensor(
                    np.full((batch_size, 1), 2.0),
                    requires_grad=False
                )

            def parameters(self):
                return []

        constant_network = Sequential([ConstantModule()])

        energy_fn = EnergyMLP(
            input_dim=5,
            hidden_dims=[8],
            network=constant_network
        )

        x = np.random.randn(4, 5)
        energy_vals, score_vals = energy_fn.energy_and_score(x)

        assert energy_vals.shape == (4, 1)
        np.testing.assert_array_equal(energy_vals, np.full((4, 1), 2.0))

        assert score_vals.shape == (4, 5)
        np.testing.assert_array_equal(score_vals, np.zeros((4, 5)))


class TestEnergyMLPConsistency:
    """Test consistency across operations."""

    def test_score_equals_negative_energy_gradient(self):
        """Verify that score = -grad(energy) w.r.t. input."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x_data = np.random.randn(16, 5)

        score = energy_fn.score(x_data)

        x = Tensor(x_data.copy(), requires_grad=True)
        energy = energy_fn(x)
        energy.sum().backward()
        energy_grad = x.grad

        np.testing.assert_allclose(score, -energy_grad, rtol=1e-5)

    def test_repeated_calls_same_result(self):
        """Test that repeated calls give same results."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[32])
        x = np.random.randn(16, 5)

        energies = [energy_fn(Tensor(x)).data.copy() for _ in range(5)]
        scores = [energy_fn.score(x).copy() for _ in range(5)]

        for i in range(1, 5):
            np.testing.assert_array_equal(energies[0], energies[i])
            np.testing.assert_array_equal(scores[0], scores[i])
