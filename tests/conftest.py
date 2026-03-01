"""
Pytest configuration and shared fixtures for EBM tests.
"""

import pytest
import numpy as np

from ebm.core.autodiff import Tensor
from ebm.core.nn import Linear, Swish, Sequential
from ebm.core.energy import EnergyMLP


@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility in all tests."""
    np.random.seed(42)
    yield


@pytest.fixture
def simple_tensor():
    """Create a simple 1D tensor with gradient tracking."""
    return Tensor([1.0, 2.0, 3.0], requires_grad=True)


@pytest.fixture
def matrix_tensor():
    """Create a 2D tensor with gradient tracking."""
    return Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)


@pytest.fixture
def scalar_tensor():
    """Create a scalar tensor with gradient tracking."""
    return Tensor(5.0, requires_grad=True)


@pytest.fixture
def random_tensor_1d():
    """Create a random 1D tensor."""
    return Tensor(np.random.randn(10), requires_grad=True)


@pytest.fixture
def random_tensor_2d():
    """Create a random 2D tensor."""
    return Tensor(np.random.randn(5, 4), requires_grad=True)


@pytest.fixture
def random_tensor_3d():
    """Create a random 3D tensor."""
    return Tensor(np.random.randn(2, 3, 4), requires_grad=True)


@pytest.fixture
def small_linear():
    """Create a small Linear layer for testing."""
    return Linear(10, 5)


@pytest.fixture
def small_mlp():
    """Create a small MLP for testing."""
    return Sequential([
        Linear(10, 32),
        Swish(),
        Linear(32, 1)
    ])


@pytest.fixture
def energy_network_2d():
    """
    Create an energy network for 2D data.

    Recommended architecture from README:
    Linear(2, 128) -> Swish -> Linear(128, 128) -> Swish -> Linear(128, 1)
    """
    return Sequential([
        Linear(2, 128),
        Swish(),
        Linear(128, 128),
        Swish(),
        Linear(128, 1)
    ])


@pytest.fixture
def energy_network_tabular():
    """
    Create an energy network for tabular data (29 features, like credit card).

    Recommended architecture from README:
    Linear(d, 256) -> Swish -> Linear(256, 256) -> Swish -> Linear(256, 1)
    """
    return Sequential([
        Linear(29, 256),
        Swish(),
        Linear(256, 256),
        Swish(),
        Linear(256, 1)
    ])


@pytest.fixture
def toy_data_gaussian():
    """Create 2D Gaussian data for testing."""
    return np.random.randn(1000, 2)


@pytest.fixture
def toy_data_moons():
    """
    Create two-moons dataset.

    This is a classic non-convex distribution useful for testing
    generative models.
    """
    n = 500
    t = np.linspace(0, np.pi, n)
    x1 = np.stack([np.cos(t), np.sin(t)], axis=1)
    x2 = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    data = np.vstack([x1, x2])
    data += np.random.randn(*data.shape) * 0.1
    return data


@pytest.fixture
def batch_2d():
    """Create a batch of 2D data."""
    return Tensor(np.random.randn(64, 2), requires_grad=True)


@pytest.fixture
def batch_tabular():
    """Create a batch of tabular data (29 features)."""
    return Tensor(np.random.randn(128, 29), requires_grad=True)


@pytest.fixture
def energy_fn_2d():
    """
    Create an EnergyMLP for 2D data with recommended architecture.

    Architecture: Linear(2, 128) -> Swish -> Linear(128, 128) -> Swish -> Linear(128, 1)
    """
    return EnergyMLP(input_dim=2, hidden_dims=[128, 128], activation="swish")


@pytest.fixture
def energy_fn_tabular():
    """
    Create an EnergyMLP for tabular data (29 features).

    Architecture: Linear(29, 256) -> Swish -> Linear(256, 256) -> Swish -> Linear(256, 1)
    """
    return EnergyMLP(input_dim=29, hidden_dims=[256, 256], activation="swish")


@pytest.fixture
def small_energy_fn():
    """Create a small EnergyMLP for fast testing."""
    return EnergyMLP(input_dim=2, hidden_dims=[32, 32], activation="swish")


@pytest.fixture
def quadratic_energy_fn():
    """
    Create a simple quadratic energy function.

    E(x) = 0.5 * ||x||^2 corresponds to N(0, I).
    Useful for testing Langevin sampling recovers known distributions.
    """
    def energy(x):
        return (x * x).sum(axis=-1, keepdims=True) * 0.5
    return energy


@pytest.fixture
def langevin_init_2d():
    """Create initial samples for 2D Langevin testing."""
    return np.random.randn(100, 2) * 2


@pytest.fixture
def langevin_init_high_dim():
    """Create initial samples for high-dimensional Langevin testing."""
    return np.random.randn(64, 10)


@pytest.fixture
def replay_buffer_2d():
    """
    Create a replay buffer for 2D data.

    Default capacity of 1000 samples, initialized with uniform noise.
    """
    from ebm.sampling.replay_buffer import ReplayBuffer
    return ReplayBuffer(capacity=1000, sample_dim=2)


@pytest.fixture
def replay_buffer_tabular():
    """
    Create a replay buffer for tabular data (29 features).

    Default capacity of 10000 samples, initialized with uniform noise.
    """
    from ebm.sampling.replay_buffer import ReplayBuffer
    return ReplayBuffer(capacity=10000, sample_dim=29)


@pytest.fixture
def small_replay_buffer():
    """Create a small replay buffer for fast testing."""
    from ebm.sampling.replay_buffer import ReplayBuffer
    return ReplayBuffer(capacity=100, sample_dim=2)


@pytest.fixture
def replay_buffer_from_gaussian():
    """Create a replay buffer initialized from Gaussian data."""
    from ebm.sampling.replay_buffer import ReplayBuffer
    return ReplayBuffer(capacity=1000, sample_dim=2, init_type="gaussian", init_std=1.0)


@pytest.fixture
def entropy_estimator():
    """Create a default k-NN entropy estimator."""
    from ebm.entropy.knn import KNNEntropyEstimator
    return KNNEntropyEstimator(k=5)


@pytest.fixture
def entropy_estimator_k3():
    """Create a k-NN entropy estimator with k=3."""
    from ebm.entropy.knn import KNNEntropyEstimator
    return KNNEntropyEstimator(k=3)


@pytest.fixture
def gaussian_samples_2d():
    """Create 2D standard Gaussian samples for entropy testing."""
    return np.random.randn(1000, 2)


@pytest.fixture
def gaussian_samples_5d():
    """Create 5D standard Gaussian samples for entropy testing."""
    return np.random.randn(1000, 5)


@pytest.fixture
def uniform_samples_2d():
    """Create 2D uniform samples in [0, 1]^2."""
    return np.random.rand(1000, 2)


@pytest.fixture
def clustered_samples_2d():
    """Create 2D clustered samples around origin."""
    return np.random.randn(1000, 2) * 0.1


@pytest.fixture
def entropy_config_default():
    """Create default entropy configuration."""
    from ebm.entropy.knn import KNNEntropyConfig
    return KNNEntropyConfig()


@pytest.fixture
def entropy_config_custom():
    """Create custom entropy configuration."""
    from ebm.entropy.knn import KNNEntropyConfig
    return KNNEntropyConfig(k=3, epsilon=1e-8, metric="euclidean")


@pytest.fixture
def sgd_optimizer():
    """Create an SGD optimizer with default parameters."""
    from ebm.training.optimizer import SGD
    params = [Tensor(np.random.randn(10, 5), requires_grad=True)]
    return SGD(params, lr=0.01, momentum=0.9)


@pytest.fixture
def adamw_optimizer():
    """Create an AdamW optimizer with default parameters."""
    from ebm.training.optimizer import AdamW
    params = [Tensor(np.random.randn(10, 5), requires_grad=True)]
    return AdamW(params, lr=1e-4, weight_decay=0.01)


@pytest.fixture
def optimizer_config_sgd():
    """Create SGD optimizer configuration."""
    from ebm.training.optimizer import OptimizerConfig
    return OptimizerConfig(optimizer_type="sgd", lr=0.01, momentum=0.9)


@pytest.fixture
def optimizer_config_adamw():
    """Create AdamW optimizer configuration."""
    from ebm.training.optimizer import OptimizerConfig
    return OptimizerConfig(
        optimizer_type="adamw",
        lr=1e-4,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.01,
    )


@pytest.fixture
def training_config_default():
    """Create default training configuration."""
    from ebm.training.trainer import TrainingConfig
    return TrainingConfig()


@pytest.fixture
def training_config_fast():
    """Create fast training configuration for testing."""
    from ebm.training.trainer import TrainingConfig
    return TrainingConfig(
        n_epochs=3,
        batch_size=32,
        langevin_steps=5,
        langevin_step_size=0.01,
        alpha=0.1,
        lambda_ent=0.0,
    )


@pytest.fixture
def training_setup_2d():
    """
    Create a complete training setup for 2D data.

    Returns a dict with:
        - energy_fn: EnergyMLP for 2D
        - optimizer: AdamW optimizer
        - buffer: ReplayBuffer
        - data: Training data
    """
    from ebm.training.optimizer import AdamW
    from ebm.sampling.replay_buffer import ReplayBuffer

    energy_fn = EnergyMLP(input_dim=2, hidden_dims=[64, 64])
    optimizer = AdamW(energy_fn.parameters(), lr=1e-3, weight_decay=0.01)
    buffer = ReplayBuffer(capacity=1000, sample_dim=2)
    data = np.random.randn(500, 2)

    return {
        "energy_fn": energy_fn,
        "optimizer": optimizer,
        "buffer": buffer,
        "data": data,
    }


@pytest.fixture
def training_setup_tabular():
    """
    Create a complete training setup for tabular data (29D like credit card).

    Returns a dict with:
        - energy_fn: EnergyMLP for 29D
        - optimizer: AdamW optimizer
        - buffer: ReplayBuffer
        - data: Training data
    """
    from ebm.training.optimizer import AdamW
    from ebm.sampling.replay_buffer import ReplayBuffer

    energy_fn = EnergyMLP(input_dim=29, hidden_dims=[256, 256])
    optimizer = AdamW(energy_fn.parameters(), lr=1e-4, weight_decay=0.01)
    buffer = ReplayBuffer(capacity=5000, sample_dim=29)
    data = np.random.randn(1000, 29)

    return {
        "energy_fn": energy_fn,
        "optimizer": optimizer,
        "buffer": buffer,
        "data": data,
    }


@pytest.fixture
def synthetic_anomaly_data():
    """
    Create synthetic anomaly detection data.

    Returns a tuple of (X_train, X_test, y_train, y_test) where:
    - X_train contains only normal samples
    - X_test contains both normal and anomaly samples
    - y_train is all zeros (normal)
    - y_test has 0 for normal, 1 for anomaly
    """
    from ebm.anomaly.data import create_synthetic_anomaly_data
    return create_synthetic_anomaly_data(
        n_normal=500,
        n_anomaly=50,
        n_features=10,
        normal_mean=0.0,
        normal_std=1.0,
        anomaly_shift=3.0,
        random_state=42,
    )


@pytest.fixture
def anomaly_detector_10d():
    """Create an anomaly detector for 10D data."""
    from ebm.anomaly.detector import EBMAnomalyDetector
    energy_fn = EnergyMLP(input_dim=10, hidden_dims=[32, 32])
    return EBMAnomalyDetector(energy_fn)


@pytest.fixture
def anomaly_detector_tabular():
    """Create an anomaly detector for tabular data (29D)."""
    from ebm.anomaly.detector import EBMAnomalyDetector
    energy_fn = EnergyMLP(input_dim=29, hidden_dims=[64, 64])
    return EBMAnomalyDetector(energy_fn)


@pytest.fixture
def anomaly_detector_with_threshold():
    """Create an anomaly detector with a fitted threshold."""
    from ebm.anomaly.detector import EBMAnomalyDetector
    energy_fn = EnergyMLP(input_dim=10, hidden_dims=[32, 32])
    detector = EBMAnomalyDetector(energy_fn)

    X_val = np.random.randn(100, 10)
    y_val = np.array([0] * 90 + [1] * 10)
    detector.fit_threshold(X_val, y_val, percentile=95)

    return detector
