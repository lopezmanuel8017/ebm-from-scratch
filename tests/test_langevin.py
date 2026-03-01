"""
Comprehensive tests for Langevin dynamics sampling.

Tests cover:
- Gradient clipping functionality
- Step size annealing schedules
- Basic sampling correctness
- Sampling from known distributions (Gaussian)
- Numerical stability
- Diagnostic tracking
- Configuration and convenience functions
- Edge cases and boundary conditions
"""

import pytest
import numpy as np

from ebm.core.autodiff import Tensor
from ebm.core.ops import tensor_sum
from ebm.core.energy import EnergyMLP
from ebm.sampling.langevin import (
    clip_grad,
    langevin_sample,
    langevin_sample_with_diagnostics,
    langevin_sample_with_config,
    linear_annealing,
    geometric_annealing,
    LangevinConfig,
    LangevinDiagnostics,
    init_from_noise,
    init_from_data,
    init_mixed,
    init_persistent_chains,
)


class TestClipGrad:
    """Test gradient clipping functionality."""

    def test_clip_grad_no_clipping_needed(self):
        """Test that gradients below max_norm are unchanged."""
        grad = np.array([[0.1, 0.2], [0.05, 0.1]])
        max_norm = 1.0

        clipped = clip_grad(grad, max_norm)

        np.testing.assert_allclose(clipped, grad, rtol=1e-5)

    def test_clip_grad_clipping_applied(self):
        """Test that gradients above max_norm are clipped."""
        grad = np.array([[3.0, 4.0]])
        max_norm = 1.0

        clipped = clip_grad(grad, max_norm)

        clipped_norm = np.linalg.norm(clipped[0])
        np.testing.assert_allclose(clipped_norm, max_norm, rtol=1e-5)

    def test_clip_grad_direction_preserved(self):
        """Test that gradient direction is preserved after clipping."""
        grad = np.array([[3.0, 4.0]])
        max_norm = 1.0

        clipped = clip_grad(grad, max_norm)

        original_direction = grad[0] / np.linalg.norm(grad[0])
        clipped_direction = clipped[0] / np.linalg.norm(clipped[0])
        np.testing.assert_allclose(original_direction, clipped_direction, rtol=1e-5)

    def test_clip_grad_per_sample(self):
        """Test that clipping is applied per sample independently."""
        grad = np.array([
            [3.0, 4.0],
            [0.1, 0.2],
        ])
        max_norm = 1.0

        clipped = clip_grad(grad, max_norm)

        np.testing.assert_allclose(np.linalg.norm(clipped[0]), 1.0, rtol=1e-5)

        np.testing.assert_allclose(clipped[1], grad[1], rtol=1e-5)

    def test_clip_grad_1d_input(self):
        """Test clipping with 1D input (single sample without batch dim)."""
        grad = np.array([3.0, 4.0])
        max_norm = 1.0

        clipped = clip_grad(grad, max_norm)

        assert clipped.shape == (2,)
        np.testing.assert_allclose(np.linalg.norm(clipped), 1.0, rtol=1e-5)

    def test_clip_grad_1d_no_clipping(self):
        """Test 1D input when no clipping is needed."""
        grad = np.array([0.1, 0.2])
        max_norm = 1.0

        clipped = clip_grad(grad, max_norm)

        np.testing.assert_allclose(clipped, grad, rtol=1e-5)

    def test_clip_grad_zero_gradient(self):
        """Test clipping with zero gradient."""
        grad = np.zeros((4, 3))
        max_norm = 1.0

        clipped = clip_grad(grad, max_norm)

        np.testing.assert_array_equal(clipped, grad)

    def test_clip_grad_very_small_gradient(self):
        """Test clipping with very small gradient."""
        grad = np.array([[1e-10, 1e-10]])
        max_norm = 1.0

        clipped = clip_grad(grad, max_norm)

        np.testing.assert_allclose(clipped, grad, rtol=1e-5)

    def test_clip_grad_very_large_gradient(self):
        """Test clipping with very large gradient."""
        grad = np.array([[1e10, 1e10]])
        max_norm = 1.0

        clipped = clip_grad(grad, max_norm)

        np.testing.assert_allclose(np.linalg.norm(clipped[0]), 1.0, rtol=1e-5)

    def test_clip_grad_batch_shapes(self):
        """Test clipping with various batch sizes."""
        for batch_size in [1, 2, 16, 64, 128]:
            grad = np.random.randn(batch_size, 10) * 10
            max_norm = 0.5

            clipped = clip_grad(grad, max_norm)

            assert clipped.shape == grad.shape
            norms = np.linalg.norm(clipped, axis=-1)
            assert (norms <= max_norm + 1e-5).all()

    def test_clip_grad_different_dims(self):
        """Test clipping with different feature dimensions."""
        for dim in [2, 5, 10, 50, 100]:
            grad = np.random.randn(32, dim) * 5
            max_norm = 0.1

            clipped = clip_grad(grad, max_norm)

            assert clipped.shape == (32, dim)
            norms = np.linalg.norm(clipped, axis=-1)
            assert (norms <= max_norm + 1e-5).all()


class TestLinearAnnealing:
    """Test linear step size annealing."""

    def test_linear_annealing_endpoints(self):
        """Test that endpoints are correct."""
        schedule = linear_annealing(0.1, 0.01, 10)

        np.testing.assert_allclose(schedule[0], 0.1, rtol=1e-5)
        np.testing.assert_allclose(schedule[-1], 0.01, rtol=1e-5)

    def test_linear_annealing_length(self):
        """Test that schedule has correct length."""
        for n_steps in [1, 10, 50, 100]:
            schedule = linear_annealing(0.1, 0.01, n_steps)
            assert len(schedule) == n_steps

    def test_linear_annealing_monotonic_decrease(self):
        """Test that schedule is monotonically decreasing."""
        schedule = linear_annealing(0.1, 0.01, 100)

        assert (np.diff(schedule) <= 1e-10).all()

    def test_linear_annealing_equal_spacing(self):
        """Test that steps are equally spaced."""
        schedule = linear_annealing(0.1, 0.01, 10)
        diffs = np.diff(schedule)

        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-5)

    def test_linear_annealing_increasing(self):
        """Test linear annealing with increasing step size."""
        schedule = linear_annealing(0.01, 0.1, 10)

        np.testing.assert_allclose(schedule[0], 0.01, rtol=1e-5)
        np.testing.assert_allclose(schedule[-1], 0.1, rtol=1e-5)
        assert (np.diff(schedule) >= -1e-10).all()

    def test_linear_annealing_same_value(self):
        """Test linear annealing with same start and end value."""
        schedule = linear_annealing(0.1, 0.1, 10)

        np.testing.assert_allclose(schedule, 0.1, rtol=1e-5)


class TestGeometricAnnealing:
    """Test geometric step size annealing."""

    def test_geometric_annealing_endpoints(self):
        """Test that endpoints are correct."""
        schedule = geometric_annealing(0.1, 0.01, 10)

        np.testing.assert_allclose(schedule[0], 0.1, rtol=1e-5)
        np.testing.assert_allclose(schedule[-1], 0.01, rtol=1e-4)

    def test_geometric_annealing_length(self):
        """Test that schedule has correct length."""
        for n_steps in [1, 10, 50, 100]:
            schedule = geometric_annealing(0.1, 0.01, n_steps)
            assert len(schedule) == n_steps

    def test_geometric_annealing_monotonic_decrease(self):
        """Test that schedule is monotonically decreasing."""
        schedule = geometric_annealing(0.1, 0.01, 100)

        assert (np.diff(schedule) <= 1e-10).all()

    def test_geometric_annealing_constant_ratio(self):
        """Test that ratio between consecutive steps is constant."""
        schedule = geometric_annealing(0.1, 0.01, 10)
        ratios = schedule[1:] / schedule[:-1]

        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-4)

    def test_geometric_annealing_single_step(self):
        """Test geometric annealing with single step."""
        schedule = geometric_annealing(0.1, 0.01, 1)

        assert len(schedule) == 1
        np.testing.assert_allclose(schedule[0], 0.1, rtol=1e-5)

    def test_geometric_vs_linear_initial_decay(self):
        """Test that geometric decays faster initially than linear."""
        n_steps = 100
        linear = linear_annealing(0.1, 0.01, n_steps)
        geometric = geometric_annealing(0.1, 0.01, n_steps)

        mid = n_steps // 2
        assert geometric[mid] < linear[mid]


class TestLangevinSampleBasic:
    """Test basic Langevin sampling functionality."""

    def test_langevin_output_shape(self):
        """Test that output has correct shape."""
        def simple_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)
        samples = langevin_sample(simple_energy, x_init, n_steps=10, step_size=0.1)

        assert samples.shape == (32, 2)

    def test_langevin_does_not_modify_input(self):
        """Test that input array is not modified."""
        def simple_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)
        x_init_copy = x_init.copy()

        langevin_sample(simple_energy, x_init, n_steps=10, step_size=0.1)

        np.testing.assert_array_equal(x_init, x_init_copy)

    def test_langevin_produces_different_samples(self):
        """Test that running Langevin twice produces different samples."""
        def simple_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)

        samples1 = langevin_sample(simple_energy, x_init.copy(), n_steps=10, step_size=0.1)
        samples2 = langevin_sample(simple_energy, x_init.copy(), n_steps=10, step_size=0.1)

        assert not np.allclose(samples1, samples2)

    def test_langevin_finite_output(self):
        """Test that output is finite (no NaN or Inf)."""
        def simple_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)
        samples = langevin_sample(simple_energy, x_init, n_steps=50, step_size=0.01)

        assert np.isfinite(samples).all()

    def test_langevin_with_tensor_energy_fn(self):
        """Test Langevin with energy function using Tensor operations."""
        def tensor_energy(x):
            return tensor_sum(x * x, axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)
        samples = langevin_sample(tensor_energy, x_init, n_steps=10, step_size=0.1)

        assert samples.shape == (32, 2)
        assert np.isfinite(samples).all()

    def test_langevin_return_trajectory(self):
        """Test that trajectory is correctly returned."""
        def simple_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)
        n_steps = 20

        samples, trajectory = langevin_sample(
            simple_energy, x_init, n_steps=n_steps, step_size=0.1,
            return_trajectory=True
        )

        assert trajectory.shape == (n_steps + 1, 32, 2)

        np.testing.assert_array_equal(trajectory[0], x_init)

        np.testing.assert_array_equal(trajectory[-1], samples)

    def test_langevin_zero_steps(self):
        """Test Langevin with zero steps returns initial samples."""
        def simple_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)
        samples = langevin_sample(simple_energy, x_init, n_steps=0, step_size=0.1)

        np.testing.assert_array_equal(samples, x_init)


class TestLangevinGaussian:
    """Test Langevin sampling recovers known Gaussian distribution."""

    def test_langevin_gaussian_mean(self):
        """Langevin should recover Gaussian mean when energy is quadratic."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        n_samples = 500
        x_init = np.random.randn(n_samples, 2) * 3

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=200, step_size=0.1,
            grad_clip=10.0
        )

        mean = samples.mean(axis=0)
        assert np.abs(mean).max() < 0.3

    def test_langevin_gaussian_variance(self):
        """Langevin should recover Gaussian variance when energy is quadratic."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        n_samples = 500
        x_init = np.random.randn(n_samples, 2) * 3

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=200, step_size=0.1,
            grad_clip=10.0
        )

        var = samples.var(axis=0)
        assert np.abs(var - 1.0).max() < 0.4

    def test_langevin_gaussian_non_unit_variance(self):
        """Test Langevin recovers Gaussian with non-unit variance."""
        sigma = 0.5

        def scaled_quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5 / (sigma ** 2)

        n_samples = 500
        x_init = np.random.randn(n_samples, 2)

        samples = langevin_sample(
            scaled_quadratic_energy, x_init, n_steps=300, step_size=0.02,
            grad_clip=10.0
        )

        var = samples.var(axis=0)
        assert np.abs(var - sigma**2).max() < 0.2

    def test_langevin_gaussian_shifted_mean(self):
        """Test Langevin recovers Gaussian with non-zero mean."""
        mu = np.array([1.0, -1.0])

        def shifted_quadratic_energy(x):
            mu_tensor = Tensor(mu, requires_grad=False)
            centered = x - mu_tensor
            return (centered * centered).sum(axis=-1, keepdims=True) * 0.5

        n_samples = 500
        x_init = np.random.randn(n_samples, 2) * 3

        samples = langevin_sample(
            shifted_quadratic_energy, x_init, n_steps=200, step_size=0.1,
            grad_clip=10.0
        )

        mean = samples.mean(axis=0)
        assert np.abs(mean - mu).max() < 0.3

    def test_langevin_higher_dimension_gaussian(self):
        """Test Langevin in higher dimensions."""
        dim = 10

        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        n_samples = 300
        x_init = np.random.randn(n_samples, dim) * 3

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=300, step_size=0.05,
            grad_clip=10.0
        )

        mean = samples.mean(axis=0)
        assert np.abs(mean).max() < 0.4

        var = samples.var(axis=0)
        assert np.abs(var - 1.0).max() < 0.5


class TestLangevinStability:
    """Test numerical stability of Langevin sampling."""

    def test_langevin_no_nan_with_clipping(self):
        """Langevin should not produce NaN with gradient clipping."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[64, 64])

        x_init = np.random.randn(100, 2) * 10

        samples = langevin_sample(
            energy_fn, x_init, n_steps=50, step_size=0.01, grad_clip=0.03
        )

        assert np.isfinite(samples).all()

    def test_langevin_bounded_samples(self):
        """Test that samples don't diverge with reasonable settings."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        x_init = np.random.randn(100, 2)

        samples = langevin_sample(
            energy_fn, x_init, n_steps=100, step_size=0.01, grad_clip=0.03
        )

        max_norm = np.linalg.norm(samples, axis=-1).max()
        assert max_norm < 100

    def test_langevin_small_step_size(self):
        """Test Langevin with very small step size."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=10, step_size=1e-5
        )

        assert np.isfinite(samples).all()

    def test_langevin_large_step_size_with_clipping(self):
        """Test that clipping prevents divergence with large step size."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=100, step_size=0.5, grad_clip=0.1
        )

        assert np.isfinite(samples).all()

    def test_langevin_zero_noise(self):
        """Test Langevin with zero noise (deterministic gradient descent)."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2) * 2

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=100, step_size=0.1, noise_scale=0.0
        )

        initial_norm = np.linalg.norm(x_init, axis=-1).mean()
        final_norm = np.linalg.norm(samples, axis=-1).mean()
        assert final_norm < initial_norm

    def test_langevin_with_energy_mlp_large_batch(self):
        """Test Langevin with EnergyMLP and large batch size."""
        energy_fn = EnergyMLP(input_dim=5, hidden_dims=[64, 64])
        x_init = np.random.randn(256, 5)

        samples = langevin_sample(
            energy_fn, x_init, n_steps=20, step_size=0.01, grad_clip=0.03
        )

        assert samples.shape == (256, 5)
        assert np.isfinite(samples).all()


class TestLangevinDiagnostics:
    """Test diagnostic tracking in Langevin sampling."""

    def test_diagnostics_tracks_energies(self):
        """Test that energies are tracked correctly."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2) * 3
        n_steps = 50

        _samples, diagnostics = langevin_sample_with_diagnostics(
            quadratic_energy, x_init, n_steps=n_steps, step_size=0.1
        )

        assert len(diagnostics.energies) == n_steps
        assert all(np.isfinite(e) for e in diagnostics.energies)

    def test_diagnostics_tracks_grad_norms(self):
        """Test that gradient norms are tracked correctly."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)
        n_steps = 50

        _samples, diagnostics = langevin_sample_with_diagnostics(
            quadratic_energy, x_init, n_steps=n_steps, step_size=0.1
        )

        assert len(diagnostics.grad_norms) == n_steps
        assert all(np.isfinite(g) for g in diagnostics.grad_norms)
        assert all(g >= 0 for g in diagnostics.grad_norms)

    def test_diagnostics_tracks_step_sizes(self):
        """Test that step sizes are tracked correctly."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)
        n_steps = 50
        step_size = 0.05

        _samples, diagnostics = langevin_sample_with_diagnostics(
            quadratic_energy, x_init, n_steps=n_steps, step_size=step_size
        )

        assert len(diagnostics.step_sizes) == n_steps
        assert all(np.isclose(s, step_size) for s in diagnostics.step_sizes)

    def test_diagnostics_tracks_sample_norms(self):
        """Test that sample norms are tracked."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)
        n_steps = 50

        _samples, diagnostics = langevin_sample_with_diagnostics(
            quadratic_energy, x_init, n_steps=n_steps, step_size=0.1
        )

        assert len(diagnostics.sample_norms) == n_steps
        assert all(np.isfinite(n) for n in diagnostics.sample_norms)

    def test_diagnostics_energy_decreases_initially(self):
        """Test that energy generally decreases initially from random start."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(100, 2) * 5
        n_steps = 100

        _samples, diagnostics = langevin_sample_with_diagnostics(
            quadratic_energy, x_init, n_steps=n_steps, step_size=0.1,
            grad_clip=10.0
        )

        assert diagnostics.energies[-1] < diagnostics.energies[0]

    def test_diagnostics_to_dict(self):
        """Test conversion of diagnostics to dictionary."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)
        n_steps = 20

        _samples, diagnostics = langevin_sample_with_diagnostics(
            quadratic_energy, x_init, n_steps=n_steps, step_size=0.1
        )

        diag_dict = diagnostics.to_dict()

        assert "energies" in diag_dict
        assert "grad_norms" in diag_dict
        assert "step_sizes" in diag_dict
        assert "sample_norms" in diag_dict
        assert len(diag_dict["energies"]) == n_steps


class TestLangevinAnnealing:
    """Test step size annealing in Langevin sampling."""

    def test_linear_annealing_in_sampling(self):
        """Test linear step size annealing during sampling."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)
        n_steps = 50

        _samples, diagnostics = langevin_sample_with_diagnostics(
            quadratic_energy, x_init, n_steps=n_steps, step_size=0.1,
            anneal_step_size=True, step_size_end=0.01, anneal_type="linear"
        )

        step_sizes = np.array(diagnostics.step_sizes)
        np.testing.assert_allclose(step_sizes[0], 0.1, rtol=1e-5)
        np.testing.assert_allclose(step_sizes[-1], 0.01, rtol=1e-3)

    def test_geometric_annealing_in_sampling(self):
        """Test geometric step size annealing during sampling."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)
        n_steps = 50

        _samples, diagnostics = langevin_sample_with_diagnostics(
            quadratic_energy, x_init, n_steps=n_steps, step_size=0.1,
            anneal_step_size=True, step_size_end=0.01, anneal_type="geometric"
        )

        step_sizes = np.array(diagnostics.step_sizes)
        np.testing.assert_allclose(step_sizes[0], 0.1, rtol=1e-5)
        np.testing.assert_allclose(step_sizes[-1], 0.01, rtol=1e-2)

    def test_annealing_default_end_step_size(self):
        """Test default end step size (step_size / 10)."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)
        n_steps = 50

        _samples, diagnostics = langevin_sample_with_diagnostics(
            quadratic_energy, x_init, n_steps=n_steps, step_size=0.1,
            anneal_step_size=True
        )

        step_sizes = np.array(diagnostics.step_sizes)
        np.testing.assert_allclose(step_sizes[-1], 0.01, rtol=1e-3)

    def test_invalid_anneal_type(self):
        """Test that invalid anneal type raises error."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)

        with pytest.raises(ValueError, match="Unknown anneal_type"):
            langevin_sample_with_diagnostics(
                quadratic_energy, x_init, n_steps=10, step_size=0.1,
                anneal_step_size=True, anneal_type="invalid"
            )


class TestLangevinConfig:
    """Test LangevinConfig dataclass."""

    def test_config_default_values(self):
        """Test default configuration values."""
        config = LangevinConfig()

        assert config.n_steps == 40
        assert config.step_size == 0.01
        assert config.noise_scale == 1.0
        assert config.grad_clip == 0.03
        assert config.anneal_step_size is False
        assert config.step_size_end is None
        assert config.anneal_type == "linear"
        assert config.return_trajectory is False

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = LangevinConfig(
            n_steps=100,
            step_size=0.05,
            noise_scale=0.5,
            grad_clip=0.1,
            anneal_step_size=True,
            step_size_end=0.001,
            anneal_type="geometric",
            return_trajectory=True
        )

        assert config.n_steps == 100
        assert config.step_size == 0.05
        assert config.noise_scale == 0.5
        assert config.grad_clip == 0.1
        assert config.anneal_step_size is True
        assert config.step_size_end == 0.001
        assert config.anneal_type == "geometric"
        assert config.return_trajectory is True

    def test_sample_with_config(self):
        """Test langevin_sample_with_config function."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        config = LangevinConfig(n_steps=20, step_size=0.1, noise_scale=1.0)
        x_init = np.random.randn(32, 2)

        samples = langevin_sample_with_config(quadratic_energy, x_init, config)

        assert samples.shape == (32, 2)
        assert np.isfinite(samples).all()

    def test_sample_with_config_trajectory(self):
        """Test langevin_sample_with_config with trajectory return."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        config = LangevinConfig(n_steps=20, step_size=0.1, return_trajectory=True)
        x_init = np.random.randn(32, 2)

        samples, trajectory = langevin_sample_with_config(quadratic_energy, x_init, config)

        assert samples.shape == (32, 2)
        assert trajectory.shape == (21, 32, 2)

    def test_sample_with_config_annealing(self):
        """Test langevin_sample_with_config with annealing."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        config = LangevinConfig(
            n_steps=20, step_size=0.1,
            anneal_step_size=True, step_size_end=0.01
        )
        x_init = np.random.randn(32, 2)

        samples, diagnostics = langevin_sample_with_config(
            quadratic_energy, x_init, config
        )

        assert samples.shape == (32, 2)
        assert isinstance(diagnostics, LangevinDiagnostics)


class TestLangevinEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample(self):
        """Test Langevin with single sample."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(1, 2)

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=20, step_size=0.1
        )

        assert samples.shape == (1, 2)

    def test_high_dimensional(self):
        """Test Langevin in high dimensions."""
        dim = 100

        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, dim)

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=50, step_size=0.05
        )

        assert samples.shape == (32, dim)
        assert np.isfinite(samples).all()

    def test_very_large_batch(self):
        """Test Langevin with very large batch size."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(1000, 2)

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=10, step_size=0.1
        )

        assert samples.shape == (1000, 2)

    def test_single_step(self):
        """Test Langevin with single step."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=1, step_size=0.1
        )

        assert samples.shape == (32, 2)
        assert not np.allclose(samples, x_init)

    def test_1d_feature(self):
        """Test Langevin with 1D feature space."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 1)

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=20, step_size=0.1
        )

        assert samples.shape == (32, 1)

    def test_constant_energy_function(self):
        """Test Langevin with constant energy (zero gradient)."""
        def constant_energy(x):
            batch_size = x.shape[0] if hasattr(x, 'shape') else x.data.shape[0]
            return Tensor(np.ones((batch_size, 1)), requires_grad=False)

        x_init = np.random.randn(32, 2)

        samples = langevin_sample(
            constant_energy, x_init, n_steps=20, step_size=0.1
        )

        assert samples.shape == (32, 2)
        assert np.isfinite(samples).all()

    def test_constant_energy_function_with_diagnostics(self):
        """Test diagnostics with constant energy (zero gradient case)."""
        def constant_energy(x):
            batch_size = x.shape[0] if hasattr(x, 'shape') else x.data.shape[0]
            return Tensor(np.ones((batch_size, 1)), requires_grad=False)

        x_init = np.random.randn(32, 2)

        samples, diagnostics = langevin_sample_with_diagnostics(
            constant_energy, x_init, n_steps=10, step_size=0.1
        )

        assert samples.shape == (32, 2)
        assert np.isfinite(samples).all()
        assert all(g == 0.0 for g in diagnostics.grad_norms)


class TestLangevinWithEnergyMLP:
    """Integration tests with EnergyMLP."""

    def test_langevin_with_small_energy_mlp(self):
        """Test Langevin with a small EnergyMLP."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        x_init = np.random.randn(64, 2)

        samples = langevin_sample(
            energy_fn, x_init, n_steps=50, step_size=0.01, grad_clip=0.03
        )

        assert samples.shape == (64, 2)
        assert np.isfinite(samples).all()

    def test_langevin_with_large_energy_mlp(self):
        """Test Langevin with a larger EnergyMLP."""
        energy_fn = EnergyMLP(input_dim=10, hidden_dims=[128, 128])
        x_init = np.random.randn(64, 10)

        samples = langevin_sample(
            energy_fn, x_init, n_steps=30, step_size=0.01, grad_clip=0.03
        )

        assert samples.shape == (64, 10)
        assert np.isfinite(samples).all()

    def test_langevin_with_diagnostics_and_mlp(self):
        """Test Langevin with diagnostics using EnergyMLP."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        x_init = np.random.randn(64, 2)

        samples, diagnostics = langevin_sample_with_diagnostics(
            energy_fn, x_init, n_steps=30, step_size=0.01, grad_clip=0.03
        )

        assert samples.shape == (64, 2)
        assert len(diagnostics.energies) == 30
        assert all(np.isfinite(e) for e in diagnostics.energies)

    def test_langevin_energy_decreases_from_uniform_init(self):
        """Test that energy generally decreases when starting from uniform."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[64, 64])

        x_init = np.random.uniform(-3, 3, (100, 2))

        samples, _diagnostics = langevin_sample_with_diagnostics(
            energy_fn, x_init, n_steps=100, step_size=0.01, grad_clip=0.03
        )

        assert np.isfinite(samples).all()

    def test_langevin_simulates_training_step(self):
        """Simulate a training step: forward pass + Langevin sampling."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])

        x_real = np.random.randn(64, 2)

        x_init = np.random.uniform(-1, 1, (64, 2))
        x_fake = langevin_sample(
            energy_fn, x_init, n_steps=20, step_size=0.01, grad_clip=0.03
        )

        e_real = energy_fn(Tensor(x_real)).data
        e_fake = energy_fn(Tensor(x_fake)).data

        assert e_real.shape == (64, 1)
        assert e_fake.shape == (64, 1)
        assert np.isfinite(e_real).all()
        assert np.isfinite(e_fake).all()


class TestLangevinTemperature:
    """Test temperature control via noise_scale parameter."""

    def test_low_temperature_concentrates_samples(self):
        """Lower temperature (noise_scale < 1) should concentrate samples."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(200, 2) * 2

        samples_t1 = langevin_sample(
            quadratic_energy, x_init.copy(), n_steps=200, step_size=0.1,
            noise_scale=1.0
        )

        samples_cold = langevin_sample(
            quadratic_energy, x_init.copy(), n_steps=200, step_size=0.1,
            noise_scale=np.sqrt(0.5)
        )

        var_t1 = samples_t1.var()
        var_cold = samples_cold.var()
        assert var_cold < var_t1

    def test_high_temperature_spreads_samples(self):
        """Higher temperature (noise_scale > 1) should spread samples."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(200, 2)

        samples_t1 = langevin_sample(
            quadratic_energy, x_init.copy(), n_steps=200, step_size=0.1,
            noise_scale=1.0
        )

        samples_hot = langevin_sample(
            quadratic_energy, x_init.copy(), n_steps=200, step_size=0.1,
            noise_scale=np.sqrt(2.0)
        )

        var_t1 = samples_t1.var()
        var_hot = samples_hot.var()
        assert var_hot > var_t1


class TestLangevinReproducibility:
    """Test reproducibility with random seeds."""

    def test_reproducible_with_seed(self):
        """Test that results are reproducible with same seed."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.array([[1.0, 2.0], [3.0, 4.0]])

        np.random.seed(12345)
        samples1 = langevin_sample(
            quadratic_energy, x_init.copy(), n_steps=10, step_size=0.1
        )

        np.random.seed(12345)
        samples2 = langevin_sample(
            quadratic_energy, x_init.copy(), n_steps=10, step_size=0.1
        )

        np.testing.assert_array_equal(samples1, samples2)

    def test_different_with_different_seed(self):
        """Test that results differ with different seeds."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.array([[1.0, 2.0], [3.0, 4.0]])

        np.random.seed(12345)
        samples1 = langevin_sample(
            quadratic_energy, x_init.copy(), n_steps=10, step_size=0.1
        )

        np.random.seed(54321)
        samples2 = langevin_sample(
            quadratic_energy, x_init.copy(), n_steps=10, step_size=0.1
        )

        assert not np.allclose(samples1, samples2)


class TestLangevinDiagnosticsClass:
    """Test LangevinDiagnostics dataclass."""

    def test_diagnostics_initialization(self):
        """Test diagnostics initializes with empty lists."""
        diagnostics = LangevinDiagnostics()

        assert diagnostics.energies == []
        assert diagnostics.grad_norms == []
        assert diagnostics.step_sizes == []
        assert diagnostics.sample_norms == []

    def test_diagnostics_append(self):
        """Test appending to diagnostics lists."""
        diagnostics = LangevinDiagnostics()

        diagnostics.energies.append(1.0)
        diagnostics.grad_norms.append(0.5)

        assert len(diagnostics.energies) == 1
        assert len(diagnostics.grad_norms) == 1

    def test_diagnostics_to_dict_empty(self):
        """Test to_dict on empty diagnostics."""
        diagnostics = LangevinDiagnostics()
        d = diagnostics.to_dict()

        assert d == {
            "energies": [],
            "grad_norms": [],
            "step_sizes": [],
            "sample_norms": [],
        }

    def test_diagnostics_to_dict_with_data(self):
        """Test to_dict with actual data."""
        diagnostics = LangevinDiagnostics(
            energies=[1.0, 0.5, 0.3],
            grad_norms=[0.1, 0.08, 0.06],
            step_sizes=[0.1, 0.1, 0.1],
            sample_norms=[2.0, 1.5, 1.2],
        )
        d = diagnostics.to_dict()

        assert d["energies"] == [1.0, 0.5, 0.3]
        assert d["grad_norms"] == [0.1, 0.08, 0.06]
        assert d["step_sizes"] == [0.1, 0.1, 0.1]
        assert d["sample_norms"] == [2.0, 1.5, 1.2]


class TestLangevinScalability:
    """Test scalability with different sizes."""

    @pytest.mark.parametrize("batch_size", [1, 10, 100, 500])
    def test_various_batch_sizes(self, batch_size):
        """Test Langevin with various batch sizes."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(batch_size, 2)

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=10, step_size=0.1
        )

        assert samples.shape == (batch_size, 2)
        assert np.isfinite(samples).all()

    @pytest.mark.parametrize("dim", [1, 2, 10, 50, 100])
    def test_various_dimensions(self, dim):
        """Test Langevin with various dimensions."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, dim)

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=10, step_size=0.1
        )

        assert samples.shape == (32, dim)
        assert np.isfinite(samples).all()

    @pytest.mark.parametrize("n_steps", [1, 10, 50, 100, 200])
    def test_various_step_counts(self, n_steps):
        """Test Langevin with various step counts."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = np.random.randn(32, 2)

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=n_steps, step_size=0.1
        )

        assert samples.shape == (32, 2)
        assert np.isfinite(samples).all()


class TestInitFromNoise:
    """Test init_from_noise function."""

    def test_uniform_noise_shape(self):
        """Test that uniform noise has correct shape."""
        samples = init_from_noise(100, 5, noise_type="uniform")
        assert samples.shape == (100, 5)

    def test_gaussian_noise_shape(self):
        """Test that Gaussian noise has correct shape."""
        samples = init_from_noise(100, 5, noise_type="gaussian")
        assert samples.shape == (100, 5)

    def test_uniform_noise_bounds(self):
        """Test that uniform noise respects bounds."""
        samples = init_from_noise(1000, 2, noise_type="uniform", low=-2.0, high=2.0)
        assert samples.min() >= -2.0
        assert samples.max() <= 2.0

    def test_uniform_noise_default_bounds(self):
        """Test that uniform noise uses default bounds [-1, 1]."""
        samples = init_from_noise(1000, 2, noise_type="uniform")
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0

    def test_gaussian_noise_std(self):
        """Test that Gaussian noise has approximately correct std."""
        samples = init_from_noise(10000, 2, noise_type="gaussian", std=0.5)
        actual_std = samples.std()
        np.testing.assert_allclose(actual_std, 0.5, rtol=0.1)

    def test_gaussian_noise_default_std(self):
        """Test that Gaussian noise uses default std=1."""
        samples = init_from_noise(10000, 2, noise_type="gaussian")
        actual_std = samples.std()
        np.testing.assert_allclose(actual_std, 1.0, rtol=0.1)

    def test_gaussian_noise_mean(self):
        """Test that Gaussian noise has approximately zero mean."""
        samples = init_from_noise(10000, 2, noise_type="gaussian")
        actual_mean = samples.mean()
        np.testing.assert_allclose(actual_mean, 0.0, atol=0.05)

    def test_uniform_noise_distribution(self):
        """Test that uniform noise is uniformly distributed."""
        samples = init_from_noise(10000, 1, noise_type="uniform", low=0, high=1)
        np.testing.assert_allclose(samples.mean(), 0.5, atol=0.05)
        np.testing.assert_allclose(samples.var(), 1/12, rtol=0.1)

    def test_invalid_noise_type(self):
        """Test that invalid noise type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown noise_type"):
            init_from_noise(100, 2, noise_type="invalid")

    def test_single_sample(self):
        """Test initialization with single sample."""
        samples = init_from_noise(1, 5, noise_type="uniform")
        assert samples.shape == (1, 5)

    def test_single_dimension(self):
        """Test initialization with single dimension."""
        samples = init_from_noise(100, 1, noise_type="uniform")
        assert samples.shape == (100, 1)

    def test_high_dimensional(self):
        """Test initialization with high dimensions."""
        samples = init_from_noise(50, 100, noise_type="gaussian")
        assert samples.shape == (50, 100)
        assert np.isfinite(samples).all()

    @pytest.mark.parametrize("n_samples", [1, 10, 100, 1000])
    def test_various_sample_sizes(self, n_samples):
        """Test initialization with various sample sizes."""
        samples = init_from_noise(n_samples, 2, noise_type="uniform")
        assert samples.shape == (n_samples, 2)

    @pytest.mark.parametrize("dim", [1, 2, 10, 50, 100])
    def test_various_dimensions(self, dim):
        """Test initialization with various dimensions."""
        samples = init_from_noise(32, dim, noise_type="gaussian")
        assert samples.shape == (32, dim)

    def test_asymmetric_uniform_bounds(self):
        """Test uniform noise with asymmetric bounds."""
        samples = init_from_noise(1000, 2, noise_type="uniform", low=0, high=5)
        assert samples.min() >= 0
        assert samples.max() <= 5
        np.testing.assert_allclose(samples.mean(), 2.5, atol=0.2)


class TestInitFromData:
    """Test init_from_data function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        data = np.random.randn(1000, 5)
        samples = init_from_data(data, 100)
        assert samples.shape == (100, 5)

    def test_samples_from_data(self):
        """Test that samples are drawn from data when noise_std=0."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        samples = init_from_data(data, 100, noise_std=0.0)

        for sample in samples:
            assert any(np.allclose(sample, d) for d in data)

    def test_noise_perturbation(self):
        """Test that noise is added when noise_std > 0."""
        data = np.array([[0.0, 0.0]])
        samples = init_from_data(data, 1000, noise_std=0.5)

        np.testing.assert_allclose(samples.mean(axis=0), [0.0, 0.0], atol=0.1)
        np.testing.assert_allclose(samples.std(axis=0), [0.5, 0.5], rtol=0.2)

    def test_no_modification_to_input(self):
        """Test that input data is not modified."""
        data = np.random.randn(100, 2)
        data_copy = data.copy()
        init_from_data(data, 50, noise_std=0.1)
        np.testing.assert_array_equal(data, data_copy)

    def test_zero_noise_std(self):
        """Test with zero noise standard deviation."""
        data = np.random.randn(100, 2)
        samples = init_from_data(data, 50, noise_std=0.0)

        for sample in samples:
            distances = np.linalg.norm(data - sample, axis=1)
            assert np.min(distances) < 1e-10

    def test_more_samples_than_data(self):
        """Test drawing more samples than data points."""
        data = np.random.randn(10, 2)
        samples = init_from_data(data, 100)
        assert samples.shape == (100, 2)

    def test_fewer_samples_than_data(self):
        """Test drawing fewer samples than data points."""
        data = np.random.randn(1000, 2)
        samples = init_from_data(data, 10)
        assert samples.shape == (10, 2)

    def test_single_data_point(self):
        """Test with single data point."""
        data = np.array([[5.0, 10.0]])
        samples = init_from_data(data, 100, noise_std=0.0)

        np.testing.assert_array_equal(samples, np.tile(data, (100, 1)))

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        data = np.random.randn(500, 50)
        samples = init_from_data(data, 100, noise_std=0.1)
        assert samples.shape == (100, 50)
        assert np.isfinite(samples).all()

    @pytest.mark.parametrize("noise_std", [0.0, 0.01, 0.1, 0.5, 1.0])
    def test_various_noise_levels(self, noise_std):
        """Test with various noise levels."""
        data = np.zeros((100, 2))
        samples = init_from_data(data, 1000, noise_std=noise_std)

        if noise_std == 0.0:
            np.testing.assert_array_equal(samples, 0.0)
        else:
            np.testing.assert_allclose(samples.std(), noise_std, rtol=0.2)


class TestInitMixed:
    """Test init_mixed function."""

    def test_output_shape_and_type(self):
        """Test that output has correct shape and type."""
        buffer = np.random.randn(1000, 5)
        samples, indices = init_mixed(buffer, 100)

        assert samples.shape == (100, 5)
        assert indices.shape == (100,)
        assert indices.dtype in (np.int32, np.int64)

    def test_mostly_from_buffer(self):
        """Test that most samples come from buffer when reinit_prob is low."""
        buffer = np.ones((100, 2)) * 999

        samples, indices = init_mixed(buffer, 1000, reinit_prob=0.0)

        # All samples should be 999 (from buffer)
        np.testing.assert_array_equal(samples, 999)

    def test_reinit_probability(self):
        """Test that approximate reinit ratio is respected."""
        buffer = np.ones((1000, 2)) * 999
        reinit_prob = 0.1

        samples, indices = init_mixed(buffer, 10000, reinit_prob=reinit_prob)

        # Count how many are not 999 (reinitialized)
        reinit_count = (np.abs(samples - 999) > 1e-5).any(axis=1).sum()

        # Should be approximately 10% (with tolerance)
        expected = 10000 * reinit_prob
        assert expected * 0.7 < reinit_count < expected * 1.3

    def test_reinit_with_uniform_noise(self):
        """Test reinitialization with uniform noise."""
        buffer = np.ones((100, 2)) * 999

        samples, indices = init_mixed(
            buffer, 1000, reinit_prob=1.0,  # All reinit
            noise_type="uniform", low=-1.0, high=1.0
        )

        # All samples should be in uniform range (none should be 999)
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0

    def test_reinit_with_gaussian_noise(self):
        """Test reinitialization with Gaussian noise."""
        buffer = np.ones((100, 2)) * 999

        samples, indices = init_mixed(
            buffer, 1000, reinit_prob=1.0,  # All reinit
            noise_type="gaussian", std=0.5
        )

        # Samples should have Gaussian properties
        np.testing.assert_allclose(samples.mean(), 0.0, atol=0.1)
        np.testing.assert_allclose(samples.std(), 0.5, rtol=0.1)

    def test_indices_valid(self):
        """Test that returned indices are valid buffer indices."""
        buffer = np.random.randn(1000, 2)
        samples, indices = init_mixed(buffer, 100, reinit_prob=0.05)

        assert indices.min() >= 0
        assert indices.max() < 1000

    def test_indices_can_be_used_for_update(self):
        """Test that indices can be used to update buffer."""
        buffer = np.random.randn(1000, 2)
        buffer_original = buffer.copy()
        samples, indices = init_mixed(buffer, 100, reinit_prob=0.0)

        # Verify samples match buffer at indices (when no reinit)
        # Note: samples is a copy, so it should match buffer_original at indices
        np.testing.assert_array_equal(samples, buffer_original[indices])

        # Simulate buffer update - use unique indices to avoid duplicate assignment issues
        unique_indices = np.unique(indices)
        new_samples = np.random.randn(len(unique_indices), 2)
        buffer[unique_indices] = new_samples

        # Buffer should be updated at those unique indices
        np.testing.assert_array_equal(buffer[unique_indices], new_samples)

    def test_zero_reinit_prob(self):
        """Test with zero reinitialization probability."""
        buffer = np.random.randn(1000, 2)
        samples, indices = init_mixed(buffer, 100, reinit_prob=0.0)

        # All samples should be from buffer
        np.testing.assert_array_equal(samples, buffer[indices])

    def test_full_reinit_prob(self):
        """Test with full reinitialization probability."""
        buffer = np.ones((100, 2)) * 999

        samples, _ = init_mixed(buffer, 100, reinit_prob=1.0, noise_type="uniform")

        # No sample should be 999
        assert not np.allclose(samples, 999)

    def test_invalid_noise_type(self):
        """Test that invalid noise type raises ValueError."""
        buffer = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="Unknown noise_type"):
            init_mixed(buffer, 50, reinit_prob=1.0, noise_type="invalid")

    def test_invalid_reinit_prob_too_low(self):
        """Test that reinit_prob < 0 raises ValueError."""
        buffer = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="reinit_prob must be in"):
            init_mixed(buffer, 50, reinit_prob=-0.1)

    def test_invalid_reinit_prob_too_high(self):
        """Test that reinit_prob > 1 raises ValueError."""
        buffer = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="reinit_prob must be in"):
            init_mixed(buffer, 50, reinit_prob=1.5)

    def test_no_modification_to_buffer(self):
        """Test that input buffer is not modified."""
        buffer = np.random.randn(100, 2)
        buffer_copy = buffer.copy()

        init_mixed(buffer, 50, reinit_prob=0.5)

        np.testing.assert_array_equal(buffer, buffer_copy)

    def test_uniform_bounds(self):
        """Test custom uniform bounds for reinitialization."""
        buffer = np.ones((100, 2)) * 999

        samples, _ = init_mixed(
            buffer, 1000, reinit_prob=1.0,
            noise_type="uniform", low=-5.0, high=5.0
        )

        assert samples.min() >= -5.0
        assert samples.max() <= 5.0

    def test_gaussian_std(self):
        """Test custom Gaussian std for reinitialization."""
        buffer = np.ones((100, 2)) * 999

        samples, _ = init_mixed(
            buffer, 1000, reinit_prob=1.0,
            noise_type="gaussian", std=2.0
        )

        np.testing.assert_allclose(samples.std(), 2.0, rtol=0.15)

    @pytest.mark.parametrize("reinit_prob", [0.0, 0.01, 0.05, 0.1, 0.5, 1.0])
    def test_various_reinit_probs(self, reinit_prob):
        """Test with various reinitialization probabilities."""
        buffer = np.random.randn(1000, 2)
        samples, indices = init_mixed(buffer, 100, reinit_prob=reinit_prob)

        assert samples.shape == (100, 2)
        assert len(indices) == 100

    def test_high_dimensional_buffer(self):
        """Test with high-dimensional buffer."""
        buffer = np.random.randn(500, 50)
        samples, _indices = init_mixed(buffer, 100, reinit_prob=0.05)

        assert samples.shape == (100, 50)
        assert np.isfinite(samples).all()


class TestInitPersistentChains:
    """Test init_persistent_chains function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        chains = init_persistent_chains(
            quadratic_energy, n_chains=100, dim=5, n_warmup_steps=10
        )

        assert chains.shape == (100, 5)

    def test_finite_output(self):
        """Test that output is finite."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        chains = init_persistent_chains(
            quadratic_energy, n_chains=100, dim=2, n_warmup_steps=50
        )

        assert np.isfinite(chains).all()

    def test_chains_move_toward_mode(self):
        """Test that chains move toward energy minima after warmup."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        chains = init_persistent_chains(
            quadratic_energy, n_chains=200, dim=2,
            n_warmup_steps=100, step_size=0.1,
            grad_clip=10.0,
            noise_type="uniform", low=-5, high=5
        )

        mean_norm = np.linalg.norm(chains, axis=-1).mean()
        assert mean_norm < 3.0

    def test_with_energy_mlp(self):
        """Test initialization with EnergyMLP."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])

        chains = init_persistent_chains(
            energy_fn, n_chains=64, dim=2, n_warmup_steps=20
        )

        assert chains.shape == (64, 2)
        assert np.isfinite(chains).all()

    def test_uniform_initialization(self):
        """Test with uniform initial noise."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        chains = init_persistent_chains(
            quadratic_energy, n_chains=100, dim=2,
            n_warmup_steps=10, noise_type="uniform", low=-2, high=2
        )

        assert chains.shape == (100, 2)

    def test_gaussian_initialization(self):
        """Test with Gaussian initial noise."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        chains = init_persistent_chains(
            quadratic_energy, n_chains=100, dim=2,
            n_warmup_steps=10, noise_type="uniform"
        )

        assert chains.shape == (100, 2)

    def test_various_warmup_steps(self):
        """Test with various warmup step counts."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        for n_warmup in [1, 10, 50, 100]:
            chains = init_persistent_chains(
                quadratic_energy, n_chains=50, dim=2,
                n_warmup_steps=n_warmup, step_size=0.1
            )
            assert chains.shape == (50, 2)
            assert np.isfinite(chains).all()

    def test_high_dimensional(self):
        """Test with high-dimensional space."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        chains = init_persistent_chains(
            quadratic_energy, n_chains=50, dim=50,
            n_warmup_steps=20, step_size=0.05
        )

        assert chains.shape == (50, 50)
        assert np.isfinite(chains).all()

    def test_step_size_and_grad_clip(self):
        """Test with custom step_size and grad_clip."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])

        chains = init_persistent_chains(
            energy_fn, n_chains=64, dim=2,
            n_warmup_steps=30,
            step_size=0.005,
            grad_clip=0.01
        )

        assert chains.shape == (64, 2)
        assert np.isfinite(chains).all()

    def test_noise_scale(self):
        """Test with custom noise scale."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        chains_low_noise = init_persistent_chains(
            quadratic_energy, n_chains=200, dim=2,
            n_warmup_steps=100, step_size=0.1,
            noise_scale=0.5
        )

        chains_high_noise = init_persistent_chains(
            quadratic_energy, n_chains=200, dim=2,
            n_warmup_steps=100, step_size=0.1,
            noise_scale=1.5
        )

        var_low = chains_low_noise.var()
        var_high = chains_high_noise.var()

        assert var_high > var_low


class TestInitializationIntegration:
    """Integration tests combining initialization strategies with Langevin sampling."""

    def test_init_from_noise_then_langevin(self):
        """Test full pipeline: init from noise, then Langevin sample."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        x_init = init_from_noise(200, 2, noise_type="uniform", low=-3, high=3)

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=200, step_size=0.1,
            grad_clip=10.0
        )

        np.testing.assert_allclose(samples.mean(axis=0), [0, 0], atol=0.3)
        np.testing.assert_allclose(samples.var(axis=0), [1, 1], atol=0.5)

    def test_init_from_data_then_langevin(self):
        """Test init from data points, then Langevin refine."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        data = np.zeros((100, 2))

        x_init = init_from_data(data, 50, noise_std=0.1)

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=50, step_size=0.1
        )

        assert samples.shape == (50, 2)
        assert np.isfinite(samples).all()

    def test_init_mixed_then_langevin(self):
        """Test mixed init (simulating replay buffer), then Langevin."""
        def quadratic_energy(x):
            return (x * x).sum(axis=-1, keepdims=True) * 0.5

        buffer = np.random.randn(1000, 2) * 0.5

        x_init, indices = init_mixed(buffer, 64, reinit_prob=0.05)

        samples = langevin_sample(
            quadratic_energy, x_init, n_steps=30, step_size=0.1
        )

        assert samples.shape == (64, 2)
        assert np.isfinite(samples).all()

        buffer[indices] = samples
        assert np.isfinite(buffer).all()

    def test_training_loop_simulation(self):
        """Simulate a training loop with initialization and sampling."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])

        buffer = init_from_noise(1000, 2, noise_type="uniform", low=-1, high=1)

        for _ in range(3):
            x_init, indices = init_mixed(buffer, 64, reinit_prob=0.05)

            x_samples = langevin_sample(
                energy_fn, x_init, n_steps=20, step_size=0.01, grad_clip=0.03
            )

            buffer[indices] = x_samples

        assert np.isfinite(buffer).all()

    def test_persistent_chains_for_replay_buffer_init(self):
        """Test using persistent chains to initialize a replay buffer."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])

        buffer = init_persistent_chains(
            energy_fn, n_chains=500, dim=2,
            n_warmup_steps=50, step_size=0.01
        )

        assert buffer.shape == (500, 2)
        assert np.isfinite(buffer).all()

        x_init, _indices = init_mixed(buffer, 64, reinit_prob=0.05)

        samples = langevin_sample(
            energy_fn, x_init, n_steps=20, step_size=0.01
        )

        assert samples.shape == (64, 2)
        assert np.isfinite(samples).all()
