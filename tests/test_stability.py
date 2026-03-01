"""
Comprehensive tests for Stability module (Section 8).

Tests cover:
- Spectral normalization
  - Power iteration algorithm
  - Spectral norm computation
  - SpectralNormWrapper for layers
  - Integration with models
- Energy clamping
  - Hard clamping
  - Soft clamping
  - EnergyClipper class
  - Statistics tracking
- Stability configuration and diagnostics
  - StabilityConfig
  - StabilityDiagnostics
  - Training stability checking
  - Suggested fixes
- Edge cases and numerical stability
- Integration tests
"""

import pytest
import numpy as np

from ebm.core.autodiff import Tensor
from ebm.core.nn import Linear, Sequential, Swish
from ebm.core.energy import EnergyMLP
from ebm.stability.spectral_norm import (
    spectral_norm,
    spectral_norm_power_iteration,
    compute_spectral_norm_exact,
    SpectralNormWrapper,
    apply_spectral_norm_to_layer,
    apply_spectral_norm_to_model,
    remove_spectral_norm_from_layer,
    get_layer_spectral_norms,
)
from ebm.stability.energy_clamp import (
    clamp_energy,
    clamp_energy_tensor,
    soft_clamp,
    soft_clamp_gradient,
    SoftClamp,
    EnergyClipper,
    check_energy_stability,
)
from ebm.stability.config import (
    StabilityConfig,
    StabilityDiagnostics,
    check_training_stability,
    get_stability_diagnostics,
    suggest_fixes,
    STABILITY_CONFIG_CONSERVATIVE,
    STABILITY_CONFIG_DEFAULT,
    STABILITY_CONFIG_AGGRESSIVE,
)


class TestSpectralNormPowerIteration:
    """Test power iteration algorithm for spectral norm computation."""

    def test_power_iteration_converges_to_spectral_norm(self):
        """Test that power iteration converges to the true spectral norm."""
        np.random.seed(42)
        W = np.random.randn(10, 5)

        _, _, sigma_approx = spectral_norm_power_iteration(W, n_iterations=100)

        sigma_exact = compute_spectral_norm_exact(W)

        np.testing.assert_allclose(sigma_approx, sigma_exact, rtol=1e-4)

    def test_power_iteration_1d_input(self):
        """Test power iteration with 1D weight (bias vector)."""
        W = np.array([1.0, 2.0, 3.0])

        _, _, sigma = spectral_norm_power_iteration(W)

        expected_sigma = np.linalg.norm(W)
        np.testing.assert_allclose(sigma, expected_sigma, rtol=1e-5)

    def test_power_iteration_with_u_init(self):
        """Test power iteration with provided initial u."""
        np.random.seed(42)
        W = np.random.randn(10, 5)
        u_init = np.random.randn(5)
        u_init = u_init / np.linalg.norm(u_init)

        _, _, sigma = spectral_norm_power_iteration(W, u=u_init, n_iterations=50)

        sigma_exact = compute_spectral_norm_exact(W)
        np.testing.assert_allclose(sigma, sigma_exact, rtol=1e-3)

    def test_power_iteration_single_iteration(self):
        """Test single power iteration step."""
        np.random.seed(42)
        W = np.random.randn(5, 3)

        u, v, sigma = spectral_norm_power_iteration(W, n_iterations=1)

        assert u.shape == (3,)
        assert v.shape == (5,)
        assert sigma > 0

    def test_power_iteration_returns_unit_vectors(self):
        """Test that returned u and v are unit vectors."""
        np.random.seed(42)
        W = np.random.randn(10, 5)

        u, v, _ = spectral_norm_power_iteration(W, n_iterations=10)

        np.testing.assert_allclose(np.linalg.norm(u), 1.0, rtol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, rtol=1e-5)

    def test_power_iteration_identity_matrix(self):
        """Test power iteration on identity matrix (spectral norm = 1)."""
        W = np.eye(5)

        _, _, sigma = spectral_norm_power_iteration(W, n_iterations=10)

        np.testing.assert_allclose(sigma, 1.0, rtol=1e-5)

    def test_power_iteration_scaled_matrix(self):
        """Test power iteration on scaled identity matrix."""
        scale = 3.5
        W = np.eye(5) * scale

        _, _, sigma = spectral_norm_power_iteration(W, n_iterations=10)

        np.testing.assert_allclose(sigma, scale, rtol=1e-5)

    def test_power_iteration_rank_1_matrix(self):
        """Test power iteration on rank-1 matrix."""
        a = np.array([1, 2, 3], dtype=float)
        b = np.array([4, 5], dtype=float)
        W = np.outer(a, b)

        _, _, sigma = spectral_norm_power_iteration(W, n_iterations=20)

        expected = np.linalg.norm(a) * np.linalg.norm(b)
        np.testing.assert_allclose(sigma, expected, rtol=1e-4)


class TestSpectralNorm:
    """Test spectral_norm function."""

    def test_spectral_norm_normalizes_matrix(self):
        """Test that spectral norm normalizes the matrix."""
        np.random.seed(42)
        W = np.random.randn(10, 5) * 5

        W_normalized, u = spectral_norm(W, n_iterations=50)

        sigma = compute_spectral_norm_exact(W_normalized)
        np.testing.assert_allclose(sigma, 1.0, rtol=1e-3)

    def test_spectral_norm_preserves_shape(self):
        """Test that spectral norm preserves matrix shape."""
        W = np.random.randn(8, 4)

        W_normalized, u = spectral_norm(W)

        assert W_normalized.shape == W.shape
        assert u.shape == (4,)

    def test_spectral_norm_preserves_direction(self):
        """Test that normalization preserves the direction of W."""
        np.random.seed(42)
        W = np.random.randn(5, 3)

        W_normalized, u = spectral_norm(W)

        ratio = W / (W_normalized + 1e-10)
        np.testing.assert_allclose(ratio, ratio[0, 0], rtol=1e-3)

    def test_spectral_norm_already_normalized(self):
        """Test spectral norm on already normalized matrix."""
        W = np.eye(5)

        W_normalized, u = spectral_norm(W, n_iterations=10)

        np.testing.assert_allclose(W_normalized, W, rtol=1e-3)

    def test_spectral_norm_u_warmstart(self):
        """Test that u can be used for warm starting."""
        np.random.seed(42)
        W = np.random.randn(10, 5)

        _W_norm1, u1 = spectral_norm(W, n_iterations=1)

        _W_norm2, u2 = spectral_norm(W, u=u1, n_iterations=1)

        assert u1.shape == u2.shape


class TestComputeSpectralNormExact:
    """Test exact spectral norm computation via SVD."""

    def test_exact_spectral_norm_identity(self):
        """Test exact spectral norm of identity matrix."""
        W = np.eye(5)

        sigma = compute_spectral_norm_exact(W)

        np.testing.assert_allclose(sigma, 1.0)

    def test_exact_spectral_norm_scaled_identity(self):
        """Test exact spectral norm of scaled identity."""
        W = np.eye(5) * 3.0

        sigma = compute_spectral_norm_exact(W)

        np.testing.assert_allclose(sigma, 3.0)

    def test_exact_spectral_norm_1d(self):
        """Test exact spectral norm of 1D vector."""
        W = np.array([3.0, 4.0])

        sigma = compute_spectral_norm_exact(W)

        np.testing.assert_allclose(sigma, 5.0)

    def test_exact_spectral_norm_rectangular(self):
        """Test exact spectral norm of rectangular matrix."""
        W = np.array([[1, 2], [3, 4], [5, 6]])

        sigma = compute_spectral_norm_exact(W)

        s = np.linalg.svd(W, compute_uv=False)
        np.testing.assert_allclose(sigma, s[0])


class TestSpectralNormWrapper:
    """Test SpectralNormWrapper class."""

    def test_wrapper_creation(self):
        """Test creating a SpectralNormWrapper."""
        layer = Linear(10, 5)
        wrapper = SpectralNormWrapper(layer)

        assert wrapper.layer is layer
        assert wrapper.enabled is True
        assert wrapper.n_iterations == 1

    def test_wrapper_forward_pass(self):
        """Test forward pass through wrapper."""
        layer = Linear(10, 5)
        wrapper = SpectralNormWrapper(layer)

        x = Tensor(np.random.randn(32, 10), requires_grad=True)
        y = wrapper(x)

        assert y.data.shape == (32, 5)
        assert np.isfinite(y.data).all()

    def test_wrapper_normalizes_weights(self):
        """Test that wrapper normalizes weights on forward pass."""
        layer = Linear(10, 5)
        layer.W.data = np.random.randn(10, 5) * 10

        wrapper = SpectralNormWrapper(layer, n_iterations=10)

        x = Tensor(np.random.randn(1, 10), requires_grad=True)
        wrapper(x)

        sigma = compute_spectral_norm_exact(layer.W.data)
        np.testing.assert_allclose(sigma, 1.0, rtol=1e-2)

    def test_wrapper_enable_disable(self):
        """Test enable/disable functionality."""
        layer = Linear(10, 5)
        layer.W.data = np.random.randn(10, 5) * 5
        _original_W = layer.W.data.copy()

        wrapper = SpectralNormWrapper(layer)

        wrapper.disable()
        x = Tensor(np.random.randn(1, 10))
        wrapper(x)

        assert wrapper.enabled is False

        wrapper.enable()
        assert wrapper.enabled is True

    def test_wrapper_get_spectral_norm(self):
        """Test get_spectral_norm method."""
        layer = Linear(10, 5)
        wrapper = SpectralNormWrapper(layer)

        sigma = wrapper.get_spectral_norm()

        assert sigma > 0
        assert np.isfinite(sigma)

    def test_wrapper_parameters(self):
        """Test that wrapper returns layer parameters."""
        layer = Linear(10, 5)
        wrapper = SpectralNormWrapper(layer)

        params = wrapper.parameters()

        assert len(params) == 2
        assert params[0] is layer.W
        assert params[1] is layer.b

    def test_wrapper_repr(self):
        """Test string representation."""
        layer = Linear(10, 5)
        wrapper = SpectralNormWrapper(layer)

        repr_str = repr(wrapper)

        assert "SpectralNormWrapper" in repr_str

    def test_wrapper_custom_iterations(self):
        """Test wrapper with custom number of iterations."""
        layer = Linear(10, 5)
        wrapper = SpectralNormWrapper(layer, n_iterations=5)

        assert wrapper.n_iterations == 5


class TestApplySpectralNorm:
    """Test spectral norm application functions."""

    def test_apply_to_layer(self):
        """Test apply_spectral_norm_to_layer."""
        layer = Linear(10, 5)
        wrapper = apply_spectral_norm_to_layer(layer)

        assert isinstance(wrapper, SpectralNormWrapper)
        assert wrapper.layer is layer

    def test_apply_to_model(self):
        """Test apply_spectral_norm_to_model."""
        model = Sequential([
            Linear(10, 32),
            Swish(),
            Linear(32, 16),
            Swish(),
            Linear(16, 1),
        ])

        wrappers = apply_spectral_norm_to_model(model)

        assert len(wrappers) == 3

    def test_apply_to_single_linear(self):
        """Test applying to single Linear layer."""
        layer = Linear(10, 5)
        wrappers = apply_spectral_norm_to_model(layer)

        assert len(wrappers) == 1

    def test_remove_spectral_norm(self):
        """Test remove_spectral_norm_from_layer."""
        layer = Linear(10, 5)
        wrapper = SpectralNormWrapper(layer)

        removed_layer = remove_spectral_norm_from_layer(wrapper)

        assert removed_layer is layer
        assert wrapper.enabled is False

    def test_get_layer_spectral_norms(self):
        """Test get_layer_spectral_norms."""
        model = Sequential([
            Linear(10, 32),
            Swish(),
            Linear(32, 1),
        ])

        norms = get_layer_spectral_norms(model)

        assert len(norms) == 2
        assert all(n > 0 for n in norms)


class TestClampEnergy:
    """Test clamp_energy function."""

    def test_clamp_energy_within_range(self):
        """Test that values within range are unchanged."""
        energy = np.array([-50, -25, 0, 25, 50])

        clamped = clamp_energy(energy, min_val=-100, max_val=100)

        np.testing.assert_array_equal(clamped, energy)

    def test_clamp_energy_clips_high(self):
        """Test that high values are clipped."""
        energy = np.array([50, 100, 200, 500])

        clamped = clamp_energy(energy, min_val=-100, max_val=100)

        np.testing.assert_array_equal(clamped, [50, 100, 100, 100])

    def test_clamp_energy_clips_low(self):
        """Test that low values are clipped."""
        energy = np.array([-500, -200, -100, -50])

        clamped = clamp_energy(energy, min_val=-100, max_val=100)

        np.testing.assert_array_equal(clamped, [-100, -100, -100, -50])

    def test_clamp_energy_scalar(self):
        """Test clamping scalar energy."""
        assert clamp_energy(500, max_val=100) == 100
        assert clamp_energy(-500, min_val=-100) == -100
        assert clamp_energy(50, min_val=-100, max_val=100) == 50

    def test_clamp_energy_2d_array(self):
        """Test clamping 2D energy array."""
        energy = np.array([[-200, 0], [50, 300]])

        clamped = clamp_energy(energy, min_val=-100, max_val=100)

        expected = np.array([[-100, 0], [50, 100]])
        np.testing.assert_array_equal(clamped, expected)

    def test_clamp_energy_default_values(self):
        """Test clamping with default values."""
        energy = np.array([-500, 0, 500])

        clamped = clamp_energy(energy)

        np.testing.assert_array_equal(clamped, [-100, 0, 100])


class TestClampEnergyTensor:
    """Test clamp_energy_tensor function."""

    def test_clamp_energy_tensor_basic(self):
        """Test basic tensor clamping."""
        energy = Tensor(np.array([[-200], [50], [300]]), requires_grad=True)

        clamped = clamp_energy_tensor(energy, min_val=-100, max_val=100)

        expected = np.array([[-100], [50], [100]])
        np.testing.assert_array_equal(clamped.data, expected)
        assert clamped.requires_grad is False

    def test_clamp_energy_tensor_preserves_shape(self):
        """Test that tensor clamping preserves shape."""
        energy = Tensor(np.random.randn(10, 5) * 200, requires_grad=True)

        clamped = clamp_energy_tensor(energy, min_val=-100, max_val=100)

        assert clamped.data.shape == (10, 5)


class TestSoftClamp:
    """Test soft clamping functions."""

    def test_soft_clamp_zero_input(self):
        """Test soft clamp at zero."""
        result = soft_clamp(0.0, limit=100)
        np.testing.assert_allclose(result, 0.0)

    def test_soft_clamp_small_input(self):
        """Test soft clamp for small inputs (approximately linear)."""
        x = 10.0
        result = soft_clamp(x, limit=100, steepness=1.0)

        assert abs(result - x) < 1.0

    def test_soft_clamp_large_input(self):
        """Test soft clamp approaches limit for large inputs."""
        x = 1000.0
        result = soft_clamp(x, limit=100, steepness=1.0)

        np.testing.assert_allclose(result, 100.0, rtol=0.01)

    def test_soft_clamp_negative_input(self):
        """Test soft clamp for negative inputs."""
        x = -1000.0
        result = soft_clamp(x, limit=100, steepness=1.0)

        np.testing.assert_allclose(result, -100.0, rtol=0.01)

    def test_soft_clamp_array(self):
        """Test soft clamp on array."""
        x = np.array([-1000, -100, 0, 100, 1000])
        result = soft_clamp(x, limit=100, steepness=0.1)

        assert result.shape == x.shape
        assert result[2] == 0
        assert result[0] < 0 and result[0] > -100
        assert result[4] > 0 and result[4] < 100

    def test_soft_clamp_symmetry(self):
        """Test that soft clamp is symmetric (odd function)."""
        x = np.array([10, 50, 100, 500])
        pos = soft_clamp(x, limit=100)
        neg = soft_clamp(-x, limit=100)

        np.testing.assert_allclose(pos, -neg)

    def test_soft_clamp_gradient_at_zero(self):
        """Test soft clamp gradient at zero."""
        grad = soft_clamp_gradient(0.0, limit=100, steepness=0.1)

        np.testing.assert_allclose(grad, 0.1)

    def test_soft_clamp_gradient_array(self):
        """Test soft clamp gradient on array."""
        x = np.array([-100, 0, 100])
        grad = soft_clamp_gradient(x, limit=100, steepness=0.1)

        assert grad.shape == x.shape
        assert grad[1] > grad[0]
        assert grad[1] > grad[2]


class TestSoftClampClass:
    """Test SoftClamp class."""

    def test_soft_clamp_class_basic(self):
        """Test SoftClamp class basic usage."""
        clamp = SoftClamp(limit=100, steepness=0.1)

        result = clamp(50.0)

        assert isinstance(result, (float, np.ndarray))

    def test_soft_clamp_class_call(self):
        """Test SoftClamp callable interface."""
        clamp = SoftClamp(limit=100)
        x = np.array([-500, 0, 500])

        result = clamp(x)

        assert result.shape == x.shape

    def test_soft_clamp_class_gradient(self):
        """Test SoftClamp gradient method."""
        clamp = SoftClamp(limit=100, steepness=0.1)

        grad = clamp.gradient(0.0)

        np.testing.assert_allclose(grad, 0.1)

    def test_soft_clamp_class_invalid_params(self):
        """Test SoftClamp with invalid parameters."""
        with pytest.raises(ValueError, match="limit must be positive"):
            SoftClamp(limit=-100)

        with pytest.raises(ValueError, match="steepness must be positive"):
            SoftClamp(limit=100, steepness=-0.1)

    def test_soft_clamp_class_repr(self):
        """Test SoftClamp string representation."""
        clamp = SoftClamp(limit=100, steepness=0.1)
        repr_str = repr(clamp)

        assert "SoftClamp" in repr_str
        assert "100" in repr_str


class TestEnergyClipper:
    """Test EnergyClipper class."""

    def test_clipper_creation(self):
        """Test creating EnergyClipper."""
        clipper = EnergyClipper(min_energy=-100, max_energy=100)

        assert clipper.config.min_energy == -100
        assert clipper.config.max_energy == 100
        assert clipper.n_clips == 0
        assert clipper.n_total == 0

    def test_clipper_clip(self):
        """Test basic clipping functionality."""
        clipper = EnergyClipper(min_energy=-100, max_energy=100)
        energy = np.array([-200, -50, 0, 50, 200])

        clamped = clipper.clip(energy)

        np.testing.assert_array_equal(clamped, [-100, -50, 0, 50, 100])

    def test_clipper_tracks_statistics(self):
        """Test that clipper tracks statistics."""
        clipper = EnergyClipper(min_energy=-100, max_energy=100)
        energy = np.array([-200, 50, 200])

        clipper.clip(energy)

        assert clipper.n_total == 3
        assert clipper.n_clips == 2
        assert clipper.max_observed == 200
        assert clipper.min_observed == -200

    def test_clipper_clip_ratio(self):
        """Test clip_ratio property."""
        clipper = EnergyClipper(min_energy=-100, max_energy=100)
        energy = np.array([-200, -50, 50, 200])

        clipper.clip(energy)

        assert clipper.clip_ratio == 0.5

    def test_clipper_clip_ratio_empty(self):
        """Test clip_ratio with no data."""
        clipper = EnergyClipper()

        assert clipper.clip_ratio == 0.0

    def test_clipper_observed_range(self):
        """Test observed_range property."""
        clipper = EnergyClipper()
        clipper.clip(np.array([-50, 0, 100]))
        clipper.clip(np.array([-200, 50]))

        assert clipper.observed_range == (-200, 100)

    def test_clipper_reset_stats(self):
        """Test reset_stats method."""
        clipper = EnergyClipper()
        clipper.clip(np.array([-200, 200]))

        clipper.reset_stats()

        assert clipper.n_clips == 0
        assert clipper.n_total == 0
        assert clipper.max_observed == float('-inf')
        assert clipper.min_observed == float('inf')

    def test_clipper_get_stats(self):
        """Test get_stats method."""
        clipper = EnergyClipper(min_energy=-100, max_energy=100)
        clipper.clip(np.array([-200, 50, 200]))

        stats = clipper.get_stats()

        assert 'n_clips' in stats
        assert 'n_total' in stats
        assert 'clip_ratio' in stats
        assert stats['n_clips'] == 2
        assert stats['n_total'] == 3

    def test_clipper_needs_attention(self):
        """Test needs_attention method."""
        clipper = EnergyClipper(min_energy=-100, max_energy=100)

        clipper.clip(np.array([-50, 0, 50]))
        assert not clipper.needs_attention(threshold=0.1)

        clipper.reset_stats()

        clipper.clip(np.array([-500, -400, -300, 0, 300, 400, 500]))
        assert clipper.needs_attention(threshold=0.1)

    def test_clipper_soft_clamp_mode(self):
        """Test clipper in soft clamp mode."""
        clipper = EnergyClipper(
            min_energy=-100, max_energy=100,
            use_soft_clamp=True, soft_steepness=0.1
        )
        energy = np.array([0.0, 500.0, -500.0])

        clamped = clipper.clip(energy)

        assert clamped[0] == 0.0
        assert 0 < clamped[1] < 100
        assert -100 < clamped[2] < 0

    def test_clipper_invalid_params(self):
        """Test clipper with invalid parameters."""
        with pytest.raises(ValueError, match="must be less than"):
            EnergyClipper(min_energy=100, max_energy=-100)

    def test_clipper_repr(self):
        """Test clipper string representation."""
        clipper = EnergyClipper(min_energy=-50, max_energy=50)
        repr_str = repr(clipper)

        assert "EnergyClipper" in repr_str
        assert "-50" in repr_str
        assert "50" in repr_str

    def test_clipper_clip_tensor(self):
        """Test clip_tensor method."""
        clipper = EnergyClipper(min_energy=-100, max_energy=100)
        energy = Tensor(np.array([[-200], [50], [200]]))

        clamped = clipper.clip_tensor(energy)

        expected = np.array([[-100], [50], [100]])
        np.testing.assert_array_equal(clamped.data, expected)


class TestCheckEnergyStability:
    """Test check_energy_stability function."""

    def test_stable_energy(self):
        """Test stability check with stable energy."""
        energy = np.array([-10, 0, 10])

        result = check_energy_stability(energy, threshold=100)

        assert result['is_stable'] is True
        assert result['has_nan'] is False
        assert result['has_inf'] is False

    def test_unstable_energy_nan(self):
        """Test stability check with NaN."""
        energy = np.array([1, np.nan, 3])

        result = check_energy_stability(energy)

        assert result['is_stable'] is False
        assert result['has_nan'] is True

    def test_unstable_energy_inf(self):
        """Test stability check with Inf."""
        energy = np.array([1, np.inf, 3])

        result = check_energy_stability(energy)

        assert result['is_stable'] is False
        assert result['has_inf'] is True

    def test_unstable_energy_extreme(self):
        """Test stability check with extreme values."""
        energy = np.array([500])

        result = check_energy_stability(energy, threshold=100)

        assert result['is_stable'] is False
        assert result['n_problematic'] == 1


class TestStabilityConfig:
    """Test StabilityConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StabilityConfig()

        assert config.grad_clip == 0.03
        assert config.energy_clamp_enabled is False
        assert config.use_spectral_norm is False
        assert config.use_lr_warmup is True
        assert config.warmup_steps == 1000

    def test_custom_config(self):
        """Test custom configuration."""
        config = StabilityConfig(
            grad_clip=0.01,
            energy_clamp_enabled=True,
            use_spectral_norm=True,
            warmup_steps=2000,
        )

        assert config.grad_clip == 0.01
        assert config.energy_clamp_enabled is True
        assert config.use_spectral_norm is True
        assert config.warmup_steps == 2000

    def test_config_to_dict(self):
        """Test to_dict method."""
        config = StabilityConfig(grad_clip=0.05)
        d = config.to_dict()

        assert 'grad_clip' in d
        assert d['grad_clip'] == 0.05

    def test_predefined_configs(self):
        """Test predefined configurations."""
        assert STABILITY_CONFIG_CONSERVATIVE.energy_clamp_enabled is True
        assert STABILITY_CONFIG_CONSERVATIVE.use_spectral_norm is True

        assert STABILITY_CONFIG_DEFAULT.grad_clip == 0.03

        assert STABILITY_CONFIG_AGGRESSIVE.use_lr_warmup is False


class TestStabilityDiagnostics:
    """Test StabilityDiagnostics dataclass."""

    def test_default_diagnostics(self):
        """Test default diagnostics are stable."""
        diag = StabilityDiagnostics()

        assert diag.is_stable is True
        assert diag.warnings == []
        assert diag.has_nan is False

    def test_add_warning(self):
        """Test add_warning method."""
        diag = StabilityDiagnostics()

        diag.add_warning("Test warning")

        assert diag.is_stable is False
        assert "Test warning" in diag.warnings

    def test_to_dict(self):
        """Test to_dict method."""
        diag = StabilityDiagnostics()
        diag.add_warning("Warning 1")

        d = diag.to_dict()

        assert 'is_stable' in d
        assert 'warnings' in d
        assert d['is_stable'] is False

    def test_repr(self):
        """Test string representation."""
        diag = StabilityDiagnostics()
        repr_str = repr(diag)

        assert "STABLE" in repr_str

        diag.add_warning("Problem")
        repr_str = repr(diag)
        assert "UNSTABLE" in repr_str


class TestCheckTrainingStability:
    """Test check_training_stability function."""

    def test_stable_training(self):
        """Test stability check with normal training values."""
        diag = check_training_stability(
            E_real=np.array([-5, -3, -4]),
            E_fake=np.array([2, 3, 2.5]),
            grad_norms=np.array([0.1, 0.2, 0.15]),
            samples=np.random.randn(100, 2),
        )

        assert diag.is_stable is True

    def test_unstable_nan_energy(self):
        """Test stability check with NaN energy."""
        diag = check_training_stability(
            E_real=np.array([np.nan, 1, 2]),
        )

        assert diag.is_stable is False
        assert diag.has_nan is True

    def test_unstable_extreme_energy(self):
        """Test stability check with extreme energy."""
        diag = check_training_stability(
            E_real=np.array([-500, -300, -400]),
        )

        assert diag.is_stable is False
        assert diag.energy_diverging is True

    def test_unstable_gradient_explosion(self):
        """Test stability check with exploding gradients."""
        diag = check_training_stability(
            grad_norms=np.array([0.1, 0.5, 100]),
        )

        assert diag.is_stable is False
        assert diag.gradient_exploding is True

    def test_unstable_sample_divergence(self):
        """Test stability check with diverging samples."""
        diag = check_training_stability(
            samples=np.ones((100, 2)) * 500,
        )

        assert diag.is_stable is False
        assert diag.samples_diverging is True

    def test_mode_collapse_detection(self):
        """Test mode collapse detection."""
        diag = check_training_stability(
            E_real=np.array([5.0, 5.0, 5.0]),
            E_fake=np.array([5.0, 5.0, 5.0]),
        )

        assert diag.mode_collapse_risk is True

    def test_low_variance_detection(self):
        """Test low variance detection (mode collapse indicator)."""
        diag = check_training_stability(
            samples=np.zeros((100, 2)) + 0.001 * np.random.randn(100, 2),
        )

        assert diag.mode_collapse_risk is True

    def test_custom_config(self):
        """Test stability check with custom config."""
        config = StabilityConfig(
            max_energy_threshold=10,
        )

        diag = check_training_stability(
            E_real=np.array([-20]),
            config=config,
        )

        assert diag.is_stable is False

    def test_metrics_populated(self):
        """Test that metrics are populated."""
        diag = check_training_stability(
            E_real=np.array([-5, -3]),
            E_fake=np.array([2, 3]),
            grad_norms=np.array([0.1, 0.2]),
            samples=np.random.randn(100, 2),
        )

        assert 'E_real_mean' in diag.metrics
        assert 'E_fake_mean' in diag.metrics
        assert 'energy_gap' in diag.metrics
        assert 'grad_norm_mean' in diag.metrics


class TestGetStabilityDiagnostics:
    """Test get_stability_diagnostics function."""

    def test_empty_history(self):
        """Test with empty history."""
        diag = get_stability_diagnostics([])

        assert diag.is_stable is True

    def test_normal_history(self):
        """Test with normal training history."""
        history = [
            {'loss': 2.0, 'E_real': -5.0, 'E_fake': 2.0, 'grad_norm': 0.1},
            {'loss': 1.5, 'E_real': -4.0, 'E_fake': 2.5, 'grad_norm': 0.15},
            {'loss': 1.0, 'E_real': -3.0, 'E_fake': 3.0, 'grad_norm': 0.12},
        ]

        diag = get_stability_diagnostics(history)

        assert diag.is_stable is True

    def test_problematic_history(self):
        """Test with problematic training history."""
        history = [
            {'loss': 2.0, 'E_real': -100.0, 'E_fake': 100.0, 'grad_norm': 50},
        ]

        diag = get_stability_diagnostics(history)

        assert diag.is_stable is False


class TestSuggestFixes:
    """Test suggest_fixes function."""

    def test_suggest_fixes_nan(self):
        """Test suggestions for NaN issue."""
        diag = StabilityDiagnostics()
        diag.has_nan = True
        diag.is_stable = False

        suggestions = suggest_fixes(diag)

        assert len(suggestions) > 0
        assert any("clamp" in s.lower() for s in suggestions)

    def test_suggest_fixes_energy_diverging(self):
        """Test suggestions for energy divergence."""
        diag = StabilityDiagnostics()
        diag.energy_diverging = True
        diag.is_stable = False

        suggestions = suggest_fixes(diag)

        assert len(suggestions) > 0

    def test_suggest_fixes_gradient_explosion(self):
        """Test suggestions for gradient explosion."""
        diag = StabilityDiagnostics()
        diag.gradient_exploding = True
        diag.is_stable = False

        suggestions = suggest_fixes(diag)

        assert len(suggestions) > 0
        assert any("gradient" in s.lower() or "clip" in s.lower() for s in suggestions)

    def test_suggest_fixes_mode_collapse(self):
        """Test suggestions for mode collapse."""
        diag = StabilityDiagnostics()
        diag.mode_collapse_risk = True
        diag.is_stable = False

        suggestions = suggest_fixes(diag)

        assert len(suggestions) > 0
        assert any("entropy" in s.lower() for s in suggestions)

    def test_suggest_fixes_stable(self):
        """Test suggestions when stable (should be empty)."""
        diag = StabilityDiagnostics()

        suggestions = suggest_fixes(diag)

        assert suggestions == []


class TestStabilityIntegration:
    """Integration tests for stability module."""

    def test_spectral_norm_with_energy_mlp(self):
        """Test spectral normalization with EnergyMLP."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])

        wrappers = apply_spectral_norm_to_model(energy_fn.network)

        assert len(wrappers) == 3

        x = Tensor(np.random.randn(64, 2), requires_grad=True)
        energy = energy_fn(x)

        assert energy.data.shape == (64, 1)
        assert np.isfinite(energy.data).all()

    def test_energy_clamping_in_training_simulation(self):
        """Test energy clamping in simulated training."""
        clipper = EnergyClipper(min_energy=-100, max_energy=100)

        for _ in range(10):
            energy = np.random.randn(64) * 50 + np.random.choice([-1, 1]) * 100
            clamped = clipper.clip(energy)

            assert np.all(clamped >= -100)
            assert np.all(clamped <= 100)

        stats = clipper.get_stats()
        assert stats['n_total'] == 640

    def test_stability_check_during_training(self):
        """Test stability checking during simulated training."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])

        for _ in range(5):
            x_real = Tensor(np.random.randn(64, 2), requires_grad=False)
            x_fake = Tensor(np.random.randn(64, 2), requires_grad=False)

            E_real = energy_fn(x_real).data
            E_fake = energy_fn(x_fake).data

            diag = check_training_stability(
                E_real=E_real,
                E_fake=E_fake,
                samples=x_fake.data,
            )

            assert 'E_real_mean' in diag.metrics
            assert 'E_fake_mean' in diag.metrics

    def test_full_stability_pipeline(self):
        """Test full stability pipeline."""
        config = StabilityConfig(
            grad_clip=0.03,
            energy_clamp_enabled=True,
            energy_clamp_min=-50,
            energy_clamp_max=50,
            use_spectral_norm=True,
        )

        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        _wrappers = apply_spectral_norm_to_model(energy_fn.network)

        clipper = EnergyClipper(
            min_energy=config.energy_clamp_min,
            max_energy=config.energy_clamp_max,
        )

        x = Tensor(np.random.randn(64, 2), requires_grad=True)
        energy = energy_fn(x)

        if config.energy_clamp_enabled:
            energy_clamped = clipper.clip(energy.data)
        else:
            energy_clamped = energy.data

        _diag = check_training_stability(
            E_real=energy_clamped,
            samples=x.data,
            config=config,
        )

        assert np.isfinite(energy_clamped).all()


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_spectral_norm_zero_matrix(self):
        """Test spectral norm on zero matrix."""
        W = np.zeros((5, 3))

        W_norm, u = spectral_norm(W)

        assert np.allclose(W_norm, 0)

    def test_spectral_norm_very_small_matrix(self):
        """Test spectral norm on very small matrix."""
        W = np.array([[1e-5]])

        _, _, sigma = spectral_norm_power_iteration(W, n_iterations=10)

        np.testing.assert_allclose(sigma, 1e-5, rtol=1e-2)

    def test_spectral_norm_very_large_matrix(self):
        """Test spectral norm on very large matrix."""
        W = np.array([[1e10]])

        W_norm, u = spectral_norm(W)

        np.testing.assert_allclose(W_norm[0, 0], 1.0, rtol=1e-5)

    def test_clamp_energy_empty_array(self):
        """Test clamp_energy with empty array."""
        energy = np.array([])

        clamped = clamp_energy(energy)

        assert clamped.shape == (0,)

    def test_soft_clamp_very_large_input(self):
        """Test soft clamp with very large inputs."""
        x = np.array([1e10, -1e10])

        result = soft_clamp(x, limit=100)

        np.testing.assert_allclose(result, [100, -100], rtol=0.01)

    def test_stability_check_all_none(self):
        """Test stability check with all None inputs."""
        diag = check_training_stability()

        assert diag.is_stable is True
        assert diag.metrics == {}

    def test_clipper_multiple_clips(self):
        """Test clipper with multiple clip calls."""
        clipper = EnergyClipper(min_energy=-10, max_energy=10)

        for i in range(100):
            energy = np.random.randn(32) * (i + 1)
            clipper.clip(energy)

        assert clipper.n_total == 3200
        assert clipper.clip_ratio > 0

    @pytest.mark.parametrize("shape", [(1,), (10,), (10, 5), (2, 3, 4)])
    def test_clamp_energy_various_shapes(self, shape):
        """Test clamp_energy with various array shapes."""
        energy = np.random.randn(*shape) * 200

        clamped = clamp_energy(energy, min_val=-100, max_val=100)

        assert clamped.shape == shape
        assert np.all(clamped >= -100)
        assert np.all(clamped <= 100)

    @pytest.mark.parametrize("n_rows,n_cols", [(1, 1), (5, 5), (10, 5), (5, 10), (100, 50)])
    def test_spectral_norm_various_sizes(self, n_rows, n_cols):
        """Test spectral norm with various matrix sizes."""
        np.random.seed(42)
        W = np.random.randn(n_rows, n_cols) * 5

        W_norm, _ = spectral_norm(W, n_iterations=20)

        sigma = compute_spectral_norm_exact(W_norm)
        np.testing.assert_allclose(sigma, 1.0, rtol=0.1)


class TestReproducibility:
    """Test reproducibility of stability operations."""

    def test_spectral_norm_deterministic_with_u(self):
        """Test that spectral norm is deterministic with same u."""
        np.random.seed(42)
        W = np.random.randn(10, 5)
        u = np.random.randn(5)
        u = u / np.linalg.norm(u)

        W_norm1, u1 = spectral_norm(W, u=u.copy(), n_iterations=1)
        W_norm2, u2 = spectral_norm(W, u=u.copy(), n_iterations=1)

        np.testing.assert_array_equal(W_norm1, W_norm2)
        np.testing.assert_array_equal(u1, u2)

    def test_clamp_energy_deterministic(self):
        """Test that clamp_energy is deterministic."""
        energy = np.array([-500, 0, 500])

        clamped1 = clamp_energy(energy)
        clamped2 = clamp_energy(energy)

        np.testing.assert_array_equal(clamped1, clamped2)

    def test_soft_clamp_deterministic(self):
        """Test that soft_clamp is deterministic."""
        x = np.array([-100, 0, 100])

        result1 = soft_clamp(x)
        result2 = soft_clamp(x)

        np.testing.assert_array_equal(result1, result2)

class TestAdditionalCoverage:
    """Additional tests to achieve 100% coverage."""

    def test_check_training_stability_inf_detection(self):
        """Test Inf detection in check_training_stability."""
        diag = check_training_stability(
            E_real=np.array([np.inf, 1, 2]),
        )

        assert diag.is_stable is False
        assert diag.has_inf is True
        assert any("Inf detected" in w for w in diag.warnings)

    def test_check_training_stability_loss_nan(self):
        """Test loss NaN detection."""
        diag = check_training_stability(loss=np.nan)

        assert diag.is_stable is False
        assert any("Loss is NaN or Inf" in w for w in diag.warnings)

    def test_check_training_stability_loss_inf(self):
        """Test loss Inf detection."""
        diag = check_training_stability(loss=np.inf)

        assert diag.is_stable is False
        assert any("Loss is NaN or Inf" in w for w in diag.warnings)

    def test_suggest_fixes_samples_diverging(self):
        """Test suggestions for samples diverging."""
        diag = StabilityDiagnostics()
        diag.samples_diverging = True
        diag.is_stable = False

        suggestions = suggest_fixes(diag)

        assert len(suggestions) > 0
        assert any("langevin" in s.lower() or "step" in s.lower() for s in suggestions)

    def test_clipper_warn_on_clip(self):
        """Test EnergyClipper warn_on_clip functionality."""
        import warnings

        clipper = EnergyClipper(
            min_energy=-100, max_energy=100,
            warn_on_clip=True
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clipper.clip(np.array([-200, 200]))

            assert len(w) == 1
            assert "clipping" in str(w[0].message).lower()

    def test_spectral_norm_wrapper_restore_original(self):
        """Test SpectralNormWrapper restore_original method."""
        layer = Linear(10, 5)
        original_W = layer.W.data.copy()

        wrapper = SpectralNormWrapper(layer, n_iterations=10)

        x = Tensor(np.random.randn(1, 10))
        wrapper(x)

        assert not np.allclose(layer.W.data, original_W)

        wrapper.restore_original()

        np.testing.assert_array_equal(layer.W.data, original_W)

    def test_get_layer_spectral_norms_with_wrapped_layers(self):
        """Test get_layer_spectral_norms with SpectralNormWrapper layers."""
        model = Sequential([
            Linear(10, 32),
            Swish(),
            Linear(32, 1),
        ])

        apply_spectral_norm_to_model(model)

        norms = get_layer_spectral_norms(model)

        assert len(norms) == 2
        assert all(n > 0 for n in norms)

    def test_get_layer_spectral_norms_single_linear(self):
        """Test get_layer_spectral_norms with single Linear layer."""
        layer = Linear(10, 5)

        norms = get_layer_spectral_norms(layer)

        assert len(norms) == 1
        assert norms[0] > 0

    def test_spectral_norm_wrapper_1d_weight(self):
        """Test SpectralNormWrapper with 1D weight attribute."""
        class Layer1D:
            def __init__(self):
                self.W = Tensor(np.array([1.0, 2.0, 3.0]))

            def __call__(self, x):
                return x

            def parameters(self):
                return [self.W]

        layer = Layer1D()
        wrapper = SpectralNormWrapper(layer)

        assert wrapper._u.shape == (3,)
