"""
Comprehensive tests for Optimizer module.

Tests cover:
- SGD optimizer with and without momentum
- AdamW optimizer with weight decay
- Learning rate management
- State saving and loading
- Edge cases and boundary conditions
- Numerical correctness
"""

import pytest
import numpy as np

from ebm.core.autodiff import Tensor
from ebm.core.nn import Linear, Sequential, Swish
from ebm.training.optimizer import (
    SGD,
    AdamW,
    OptimizerConfig,
    create_optimizer,
    get_lr_with_warmup,
    get_lr_with_cosine_decay,
)


class TestSGDInit:
    """Test SGD initialization."""

    def test_init_basic(self):
        """Test basic SGD initialization."""
        params = [Tensor(np.random.randn(10, 5), requires_grad=True)]
        optimizer = SGD(params, lr=0.01, momentum=0.9)

        assert optimizer.lr == 0.01
        assert optimizer.momentum == 0.9
        assert len(optimizer.parameters) == 1
        assert len(optimizer.velocities) == 1

    def test_init_no_momentum(self):
        """Test SGD without momentum (vanilla SGD)."""
        params = [Tensor(np.random.randn(5, 5), requires_grad=True)]
        optimizer = SGD(params, lr=0.1, momentum=0.0)

        assert optimizer.momentum == 0.0

    def test_init_multiple_params(self):
        """Test SGD with multiple parameters."""
        params = [
            Tensor(np.random.randn(10, 5), requires_grad=True),
            Tensor(np.random.randn(5,), requires_grad=True),
            Tensor(np.random.randn(5, 3), requires_grad=True),
        ]
        optimizer = SGD(params, lr=0.01)

        assert len(optimizer.parameters) == 3
        assert len(optimizer.velocities) == 3
        assert optimizer.velocities[0].shape == (10, 5)
        assert optimizer.velocities[1].shape == (5,)
        assert optimizer.velocities[2].shape == (5, 3)

    def test_init_from_iterator(self):
        """Test SGD initialization from iterator."""
        params_list = [
            Tensor(np.random.randn(5, 5), requires_grad=True),
            Tensor(np.random.randn(5,), requires_grad=True),
        ]
        optimizer = SGD(iter(params_list), lr=0.01)

        assert len(optimizer.parameters) == 2

    def test_init_invalid_lr(self):
        """Test that invalid learning rate raises error."""
        params = [Tensor(np.random.randn(5, 5), requires_grad=True)]

        with pytest.raises(ValueError, match="positive"):
            SGD(params, lr=0)

        with pytest.raises(ValueError, match="positive"):
            SGD(params, lr=-0.01)

    def test_init_invalid_momentum(self):
        """Test that invalid momentum raises error."""
        params = [Tensor(np.random.randn(5, 5), requires_grad=True)]

        with pytest.raises(ValueError, match="\\[0, 1\\)"):
            SGD(params, lr=0.01, momentum=-0.1)

        with pytest.raises(ValueError, match="\\[0, 1\\)"):
            SGD(params, lr=0.01, momentum=1.0)


class TestSGDStep:
    """Test SGD step method."""

    def test_step_basic(self):
        """Test basic SGD step updates parameters."""
        param = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        optimizer = SGD([param], lr=0.1, momentum=0.0)

        param.grad = np.array([0.5, 1.0, 1.5])

        optimizer.step()

        expected = np.array([1.0 - 0.1 * 0.5, 2.0 - 0.1 * 1.0, 3.0 - 0.1 * 1.5])
        np.testing.assert_allclose(param.data, expected)

    def test_step_with_momentum(self):
        """Test SGD with momentum accumulates velocity."""
        param = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        optimizer = SGD([param], lr=0.1, momentum=0.9)

        param.grad = np.array([1.0, 1.0])
        optimizer.step()

        np.testing.assert_allclose(param.data, [0.9, 1.9])
        np.testing.assert_allclose(optimizer.velocities[0], [-0.1, -0.1])

        param.grad = np.array([1.0, 1.0])
        optimizer.step()

        np.testing.assert_allclose(optimizer.velocities[0], [-0.19, -0.19])
        np.testing.assert_allclose(param.data, [0.71, 1.71])

    def test_step_skips_none_grad(self):
        """Test that step skips parameters with None gradient."""
        param1 = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        param2 = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        optimizer = SGD([param1, param2], lr=0.1, momentum=0.0)

        param1.grad = np.array([1.0, 1.0])
        param2.grad = None

        original_param2 = param2.data.copy()
        optimizer.step()

        np.testing.assert_allclose(param1.data, [0.9, 1.9])
        np.testing.assert_array_equal(param2.data, original_param2)

    def test_step_multiple_params(self):
        """Test step with multiple parameters."""
        params = [
            Tensor(np.random.randn(5, 3), requires_grad=True),
            Tensor(np.random.randn(3,), requires_grad=True),
        ]
        optimizer = SGD(params, lr=0.01, momentum=0.9)

        for p in params:
            p.grad = np.random.randn(*p.shape)

        original_data = [p.data.copy() for p in params]
        optimizer.step()

        for i, p in enumerate(params):
            assert not np.allclose(p.data, original_data[i])


class TestSGDMethods:
    """Test SGD utility methods."""

    def test_zero_grad(self):
        """Test zero_grad resets gradients."""
        params = [
            Tensor(np.random.randn(5, 3), requires_grad=True),
            Tensor(np.random.randn(3,), requires_grad=True),
        ]
        optimizer = SGD(params, lr=0.01)

        for p in params:
            p.grad = np.random.randn(*p.shape)

        optimizer.zero_grad()

        for p in params:
            assert p.grad is None

    def test_get_set_lr(self):
        """Test learning rate getter and setter."""
        params = [Tensor(np.random.randn(5, 5), requires_grad=True)]
        optimizer = SGD(params, lr=0.01)

        assert optimizer.get_lr() == 0.01

        optimizer.set_lr(0.001)
        assert optimizer.get_lr() == 0.001

    def test_reset_state(self):
        """Test reset_state clears velocities."""
        param = Tensor(np.random.randn(5, 5), requires_grad=True)
        optimizer = SGD([param], lr=0.01, momentum=0.9)

        param.grad = np.random.randn(5, 5)
        optimizer.step()

        assert not np.allclose(optimizer.velocities[0], 0)

        optimizer.reset_state()

        np.testing.assert_array_equal(optimizer.velocities[0], 0)

    def test_get_load_state(self):
        """Test state save and load."""
        param = Tensor(np.random.randn(5, 5), requires_grad=True)
        optimizer = SGD([param], lr=0.01, momentum=0.9)

        param.grad = np.random.randn(5, 5)
        optimizer.step()

        state = optimizer.get_state()

        optimizer2 = SGD([param], lr=0.1, momentum=0.5)
        optimizer2.load_state(state)

        assert optimizer2.lr == 0.01
        assert optimizer2.momentum == 0.9
        np.testing.assert_array_equal(
            optimizer2.velocities[0], optimizer.velocities[0]
        )

    def test_repr(self):
        """Test string representation."""
        params = [Tensor(np.random.randn(5, 5), requires_grad=True)]
        optimizer = SGD(params, lr=0.01, momentum=0.9)

        repr_str = repr(optimizer)
        assert "SGD" in repr_str
        assert "0.01" in repr_str
        assert "0.9" in repr_str


class TestAdamWInit:
    """Test AdamW initialization."""

    def test_init_default(self):
        """Test AdamW with default parameters."""
        params = [Tensor(np.random.randn(10, 5), requires_grad=True)]
        optimizer = AdamW(params)

        assert optimizer.lr == 1e-3
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        assert optimizer.eps == 1e-8
        assert optimizer.weight_decay == 0.01
        assert optimizer.t == 0

    def test_init_custom(self):
        """Test AdamW with custom parameters."""
        params = [Tensor(np.random.randn(10, 5), requires_grad=True)]
        optimizer = AdamW(
            params, lr=1e-4, beta1=0.8, beta2=0.99,
            eps=1e-6, weight_decay=0.1
        )

        assert optimizer.lr == 1e-4
        assert optimizer.beta1 == 0.8
        assert optimizer.beta2 == 0.99
        assert optimizer.eps == 1e-6
        assert optimizer.weight_decay == 0.1

    def test_init_moment_buffers(self):
        """Test that moment buffers are initialized correctly."""
        params = [
            Tensor(np.random.randn(10, 5), requires_grad=True),
            Tensor(np.random.randn(5,), requires_grad=True),
        ]
        optimizer = AdamW(params)

        assert len(optimizer.m) == 2
        assert len(optimizer.v) == 2
        assert optimizer.m[0].shape == (10, 5)
        assert optimizer.m[1].shape == (5,)
        np.testing.assert_array_equal(optimizer.m[0], 0)
        np.testing.assert_array_equal(optimizer.v[0], 0)

    def test_init_invalid_lr(self):
        """Test invalid learning rate."""
        params = [Tensor(np.random.randn(5, 5), requires_grad=True)]

        with pytest.raises(ValueError, match="positive"):
            AdamW(params, lr=0)

        with pytest.raises(ValueError, match="positive"):
            AdamW(params, lr=-1e-4)

    def test_init_invalid_betas(self):
        """Test invalid beta values."""
        params = [Tensor(np.random.randn(5, 5), requires_grad=True)]

        with pytest.raises(ValueError, match="beta1"):
            AdamW(params, beta1=-0.1)

        with pytest.raises(ValueError, match="beta1"):
            AdamW(params, beta1=1.0)

        with pytest.raises(ValueError, match="beta2"):
            AdamW(params, beta2=-0.1)

        with pytest.raises(ValueError, match="beta2"):
            AdamW(params, beta2=1.0)

    def test_init_invalid_eps(self):
        """Test invalid epsilon."""
        params = [Tensor(np.random.randn(5, 5), requires_grad=True)]

        with pytest.raises(ValueError, match="eps"):
            AdamW(params, eps=0)

        with pytest.raises(ValueError, match="eps"):
            AdamW(params, eps=-1e-8)

    def test_init_invalid_weight_decay(self):
        """Test invalid weight decay."""
        params = [Tensor(np.random.randn(5, 5), requires_grad=True)]

        with pytest.raises(ValueError, match="weight_decay"):
            AdamW(params, weight_decay=-0.01)


class TestAdamWStep:
    """Test AdamW step method."""

    def test_step_basic(self):
        """Test basic AdamW step updates parameters."""
        param = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        optimizer = AdamW([param], lr=0.1, weight_decay=0.0)

        param.grad = np.array([1.0, 1.0, 1.0])
        original_data = param.data.copy()

        optimizer.step()

        assert optimizer.t == 1
        assert not np.allclose(param.data, original_data)

    def test_step_weight_decay(self):
        """Test that weight decay is applied."""
        param = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        optimizer = AdamW([param], lr=0.1, weight_decay=0.1)

        param.grad = np.zeros(3)

        optimizer.step()

        assert param.data[0] < 1.0
        assert param.data[1] < 2.0
        assert param.data[2] < 3.0

    def test_step_no_weight_decay(self):
        """Test AdamW without weight decay."""
        param = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        optimizer = AdamW([param], lr=0.1, weight_decay=0.0)

        param.grad = np.zeros(3)
        original_data = param.data.copy()

        optimizer.step()

        np.testing.assert_allclose(param.data, original_data, rtol=1e-5)

    def test_step_bias_correction(self):
        """Test that bias correction is applied correctly."""
        param = Tensor(np.ones(3), requires_grad=True)
        optimizer = AdamW([param], lr=0.01, weight_decay=0.0)

        param.grad = np.ones(3)

        optimizer.step()
        step1_data = param.data.copy()

        param.data = np.ones(3)
        optimizer.reset_state()

        for _ in range(10):
            param.grad = np.ones(3)
            optimizer.step()

        step10_data = param.data.copy()

        assert not np.allclose(step1_data, np.ones(3))
        assert not np.allclose(step10_data, np.ones(3))

    def test_step_skips_none_grad(self):
        """Test that step skips parameters with None gradient."""
        param1 = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        param2 = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        optimizer = AdamW([param1, param2], lr=0.1)

        param1.grad = np.array([1.0, 1.0])
        param2.grad = None

        original_param2 = param2.data.copy()
        optimizer.step()

        param2.data = original_param2.copy()
        optimizer2 = AdamW([param1, param2], lr=0.1, weight_decay=0.0)
        optimizer2.step()

        np.testing.assert_array_equal(param2.data, original_param2)

    def test_step_accumulates_moments(self):
        """Test that moments accumulate across steps."""
        param = Tensor(np.ones(5), requires_grad=True)
        optimizer = AdamW([param], lr=0.01)

        for _ in range(5):
            param.grad = np.ones(5)
            optimizer.step()

        assert not np.allclose(optimizer.m[0], 0)
        assert not np.allclose(optimizer.v[0], 0)
        assert optimizer.t == 5


class TestAdamWMethods:
    """Test AdamW utility methods."""

    def test_zero_grad(self):
        """Test zero_grad resets gradients."""
        params = [
            Tensor(np.random.randn(5, 3), requires_grad=True),
            Tensor(np.random.randn(3,), requires_grad=True),
        ]
        optimizer = AdamW(params)

        for p in params:
            p.grad = np.random.randn(*p.shape)

        optimizer.zero_grad()

        for p in params:
            assert p.grad is None

    def test_reset_state(self):
        """Test reset_state clears moments and timestep."""
        param = Tensor(np.random.randn(5, 5), requires_grad=True)
        optimizer = AdamW([param])

        param.grad = np.random.randn(5, 5)
        optimizer.step()
        optimizer.step()

        assert optimizer.t == 2
        assert not np.allclose(optimizer.m[0], 0)
        assert not np.allclose(optimizer.v[0], 0)

        optimizer.reset_state()

        assert optimizer.t == 0
        np.testing.assert_array_equal(optimizer.m[0], 0)
        np.testing.assert_array_equal(optimizer.v[0], 0)

    def test_get_load_state(self):
        """Test state save and load."""
        param = Tensor(np.random.randn(5, 5), requires_grad=True)
        optimizer = AdamW([param], lr=1e-4, weight_decay=0.05)

        param.grad = np.random.randn(5, 5)
        optimizer.step()
        optimizer.step()

        state = optimizer.get_state()

        optimizer2 = AdamW([param], lr=0.1)
        optimizer2.load_state(state)

        assert optimizer2.lr == 1e-4
        assert optimizer2.weight_decay == 0.05
        assert optimizer2.t == 2
        np.testing.assert_array_equal(optimizer2.m[0], optimizer.m[0])
        np.testing.assert_array_equal(optimizer2.v[0], optimizer.v[0])

    def test_repr(self):
        """Test string representation."""
        params = [Tensor(np.random.randn(5, 5), requires_grad=True)]
        optimizer = AdamW(params, lr=1e-4, weight_decay=0.01)

        repr_str = repr(optimizer)
        assert "AdamW" in repr_str
        assert "1e-04" in repr_str or "0.0001" in repr_str


class TestOptimizerConfig:
    """Test OptimizerConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = OptimizerConfig()

        assert config.optimizer_type == "adamw"
        assert config.lr == 1e-4
        assert config.momentum == 0.9
        assert config.beta1 == 0.9
        assert config.beta2 == 0.999
        assert config.eps == 1e-8
        assert config.weight_decay == 0.01

    def test_config_custom(self):
        """Test custom configuration values."""
        config = OptimizerConfig(
            optimizer_type="sgd",
            lr=0.01,
            momentum=0.95,
        )

        assert config.optimizer_type == "sgd"
        assert config.lr == 0.01
        assert config.momentum == 0.95


class TestCreateOptimizer:
    """Test create_optimizer factory function."""

    def test_create_sgd(self):
        """Test creating SGD optimizer."""
        params = [Tensor(np.random.randn(5, 5), requires_grad=True)]
        config = OptimizerConfig(optimizer_type="sgd", lr=0.01, momentum=0.9)

        optimizer = create_optimizer(params, config)

        assert isinstance(optimizer, SGD)
        assert optimizer.lr == 0.01
        assert optimizer.momentum == 0.9

    def test_create_adamw(self):
        """Test creating AdamW optimizer."""
        params = [Tensor(np.random.randn(5, 5), requires_grad=True)]
        config = OptimizerConfig(
            optimizer_type="adamw",
            lr=1e-4,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.01,
        )

        optimizer = create_optimizer(params, config)

        assert isinstance(optimizer, AdamW)
        assert optimizer.lr == 1e-4
        assert optimizer.beta1 == 0.9
        assert optimizer.weight_decay == 0.01

    def test_create_invalid_type(self):
        """Test that invalid optimizer type raises error."""
        params = [Tensor(np.random.randn(5, 5), requires_grad=True)]
        config = OptimizerConfig(optimizer_type="invalid")

        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(params, config)


class TestLRSchedules:
    """Test learning rate scheduling functions."""

    def test_warmup_zero_steps(self):
        """Test warmup with zero warmup steps."""
        lr = get_lr_with_warmup(1e-4, step=100, warmup_steps=0)
        assert lr == 1e-4

    def test_warmup_during(self):
        """Test learning rate during warmup."""
        warmup_steps = 1000
        base_lr = 1e-4

        lr = get_lr_with_warmup(base_lr, step=0, warmup_steps=warmup_steps)
        assert lr == 0.0

        lr = get_lr_with_warmup(base_lr, step=500, warmup_steps=warmup_steps)
        assert lr == base_lr * 0.5

        lr = get_lr_with_warmup(base_lr, step=1000, warmup_steps=warmup_steps)
        assert lr == base_lr

    def test_warmup_after(self):
        """Test learning rate after warmup."""
        warmup_steps = 1000
        base_lr = 1e-4

        lr = get_lr_with_warmup(base_lr, step=2000, warmup_steps=warmup_steps)
        assert lr == base_lr

    def test_cosine_decay_no_warmup(self):
        """Test cosine decay without warmup."""
        total_steps = 10000
        base_lr = 1e-4
        min_lr = 1e-6

        lr = get_lr_with_cosine_decay(base_lr, 0, total_steps, min_lr=min_lr)
        assert lr == base_lr

        lr = get_lr_with_cosine_decay(base_lr, 5000, total_steps, min_lr=min_lr)
        expected = min_lr + 0.5 * (base_lr - min_lr)
        np.testing.assert_allclose(lr, expected, rtol=1e-5)

        lr = get_lr_with_cosine_decay(base_lr, total_steps, total_steps, min_lr=min_lr)
        np.testing.assert_allclose(lr, min_lr, rtol=1e-5)

    def test_cosine_decay_with_warmup(self):
        """Test cosine decay with warmup."""
        total_steps = 10000
        warmup_steps = 1000
        base_lr = 1e-4
        min_lr = 1e-6

        lr = get_lr_with_cosine_decay(
            base_lr, 500, total_steps, min_lr=min_lr, warmup_steps=warmup_steps
        )
        np.testing.assert_allclose(lr, base_lr * 0.5)

        lr = get_lr_with_cosine_decay(
            base_lr, warmup_steps, total_steps, min_lr=min_lr, warmup_steps=warmup_steps
        )
        np.testing.assert_allclose(lr, base_lr)

    def test_cosine_decay_zero_total_steps(self):
        """Test cosine decay with zero total steps."""
        lr = get_lr_with_cosine_decay(1e-4, step=100, total_steps=0)
        assert lr == 1e-4


class TestOptimizerIntegration:
    """Integration tests for optimizers with neural networks."""

    def test_sgd_linear_regression(self):
        """Test SGD on simple linear regression."""
        X = np.random.randn(100, 1)
        y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

        layer = Linear(1, 1)
        optimizer = SGD(layer.parameters(), lr=0.1, momentum=0.9)

        for _ in range(100):
            x_tensor = Tensor(X, requires_grad=False)
            y_tensor = Tensor(y, requires_grad=False)

            pred = layer(x_tensor)
            loss = ((pred - y_tensor) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        np.testing.assert_allclose(layer.W.data[0, 0], 2.0, atol=0.2)
        np.testing.assert_allclose(layer.b.data[0], 1.0, atol=0.2)

    def test_adamw_linear_regression(self):
        """Test AdamW on simple linear regression."""
        X = np.random.randn(100, 1)
        y = 3 * X - 2 + np.random.randn(100, 1) * 0.1

        layer = Linear(1, 1)
        optimizer = AdamW(layer.parameters(), lr=0.1, weight_decay=0.0)

        for _ in range(200):
            x_tensor = Tensor(X, requires_grad=False)
            y_tensor = Tensor(y, requires_grad=False)

            pred = layer(x_tensor)
            loss = ((pred - y_tensor) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        np.testing.assert_allclose(layer.W.data[0, 0], 3.0, atol=0.3)
        np.testing.assert_allclose(layer.b.data[0], -2.0, atol=0.3)

    def test_optimizer_with_mlp(self):
        """Test optimizer with MLP."""
        mlp = Sequential([
            Linear(5, 10),
            Swish(),
            Linear(10, 1)
        ])

        optimizer = AdamW(mlp.parameters(), lr=0.01)

        x = Tensor(np.random.randn(32, 5), requires_grad=False)
        y = mlp(x)
        loss = y.mean()

        optimizer.zero_grad()
        loss.backward()

        assert mlp.parameters()[0].grad is not None

        optimizer.step()

        optimizer.zero_grad()

        assert mlp.parameters()[0].grad is None

    def test_loss_decreases(self):
        """Test that loss decreases during training."""
        X = np.random.randn(100, 2)
        y = np.sin(X[:, 0:1]) + X[:, 1:2]

        mlp = Sequential([
            Linear(2, 20),
            Swish(),
            Linear(20, 1)
        ])

        optimizer = AdamW(mlp.parameters(), lr=0.01)

        losses = []
        for _ in range(50):
            x_tensor = Tensor(X, requires_grad=False)
            y_tensor = Tensor(y, requires_grad=False)

            pred = mlp(x_tensor)
            loss = ((pred - y_tensor) ** 2).mean()
            losses.append(float(loss.data))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert losses[-1] < losses[0]


class TestOptimizerEdgeCases:
    """Test edge cases and numerical stability."""

    def test_empty_parameters(self):
        """Test optimizer with empty parameter list."""
        optimizer = SGD([], lr=0.01)
        optimizer.step()
        optimizer.zero_grad()

    def test_single_element_parameter(self):
        """Test optimizer with scalar parameter."""
        param = Tensor(np.array(1.0), requires_grad=True)
        optimizer = AdamW([param], lr=0.1)

        param.grad = np.array(0.5)
        optimizer.step()

        assert param.data != 1.0

    def test_very_large_gradients(self):
        """Test handling of large gradients."""
        param = Tensor(np.ones(10), requires_grad=True)
        optimizer = AdamW([param], lr=0.001)

        param.grad = np.ones(10) * 1e6
        optimizer.step()

        assert np.isfinite(param.data).all()

    def test_very_small_gradients(self):
        """Test handling of small gradients."""
        param = Tensor(np.ones(10), requires_grad=True)
        optimizer = AdamW([param], lr=0.001)

        param.grad = np.ones(10) * 1e-10
        optimizer.step()

        assert np.isfinite(param.data).all()

    def test_mixed_gradients(self):
        """Test handling of mixed positive/negative gradients."""
        param = Tensor(np.ones(10), requires_grad=True)
        optimizer = AdamW([param], lr=0.1)

        param.grad = np.array([1, -1, 2, -2, 3, -3, 4, -4, 5, -5], dtype=float)
        optimizer.step()

        assert np.isfinite(param.data).all()

    def test_consistent_state_across_steps(self):
        """Test that optimizer state is consistent across many steps."""
        param = Tensor(np.ones(100), requires_grad=True)
        optimizer = AdamW([param], lr=0.001)

        for _ in range(1000):
            param.grad = np.random.randn(100) * 0.1
            optimizer.step()

        assert np.isfinite(param.data).all()
        assert np.isfinite(optimizer.m[0]).all()
        assert np.isfinite(optimizer.v[0]).all()


class TestParametrizedOptimizer:
    """Parametrized tests for optimizers."""

    @pytest.mark.parametrize("lr", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    def test_sgd_various_lr(self, lr):
        """Test SGD with various learning rates."""
        param = Tensor(np.ones(10), requires_grad=True)
        optimizer = SGD([param], lr=lr)

        param.grad = np.ones(10)
        optimizer.step()

        expected = 1.0 - lr
        np.testing.assert_allclose(param.data, expected)

    @pytest.mark.parametrize("momentum", [0.0, 0.5, 0.9, 0.99])
    def test_sgd_various_momentum(self, momentum):
        """Test SGD with various momentum values."""
        param = Tensor(np.ones(10), requires_grad=True)
        optimizer = SGD([param], lr=0.1, momentum=momentum)

        param.grad = np.ones(10)
        optimizer.step()
        optimizer.step()

        assert np.isfinite(param.data).all()

    @pytest.mark.parametrize("weight_decay", [0.0, 0.001, 0.01, 0.1])
    def test_adamw_various_wd(self, weight_decay):
        """Test AdamW with various weight decay values."""
        param = Tensor(np.ones(10), requires_grad=True)
        optimizer = AdamW([param], lr=0.01, weight_decay=weight_decay)

        param.grad = np.ones(10)
        optimizer.step()

        assert np.isfinite(param.data).all()

    @pytest.mark.parametrize("shape", [(1,), (10,), (10, 5), (2, 3, 4)])
    def test_various_shapes(self, shape):
        """Test optimizers with various tensor shapes."""
        param = Tensor(np.random.randn(*shape), requires_grad=True)

        for OptClass in [SGD, AdamW]:
            if OptClass == SGD:
                optimizer = OptClass([param], lr=0.01)
            else:
                optimizer = OptClass([param])

            param.grad = np.random.randn(*shape)
            optimizer.step()

            assert param.data.shape == shape
            assert np.isfinite(param.data).all()
