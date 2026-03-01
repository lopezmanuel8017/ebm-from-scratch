"""
Comprehensive tests for Replay Buffer.

Tests cover:
- Initialization with different parameters
- Sample method functionality
- Update method functionality
- Push method (FIFO updates)
- Buffer statistics
- Configuration and factory methods
- Edge cases and boundary conditions
- Integration with Langevin sampling
"""

import pytest
import numpy as np

from ebm.sampling.replay_buffer import (
    ReplayBuffer,
    ReplayBufferConfig,
    ReplayBufferStats,
    create_replay_buffer,
)
from ebm.sampling.langevin import langevin_sample
from ebm.core.energy import EnergyMLP


class TestReplayBufferInit:
    """Test ReplayBuffer initialization."""

    def test_init_default_uniform(self):
        """Test default initialization with uniform distribution."""
        buffer = ReplayBuffer(capacity=100, sample_dim=5)

        assert buffer.capacity == 100
        assert buffer.sample_dim == 5
        assert buffer.buffer.shape == (100, 5)
        assert buffer.init_type == "uniform"
        assert buffer.position == 0

    def test_init_uniform_bounds(self):
        """Test uniform initialization respects bounds."""
        buffer = ReplayBuffer(
            capacity=1000, sample_dim=2,
            init_type="uniform", init_low=-2.0, init_high=2.0
        )

        assert buffer.buffer.min() >= -2.0
        assert buffer.buffer.max() <= 2.0

    def test_init_uniform_default_bounds(self):
        """Test uniform initialization uses default [-1, 1] bounds."""
        buffer = ReplayBuffer(capacity=1000, sample_dim=2, init_type="uniform")

        assert buffer.buffer.min() >= -1.0
        assert buffer.buffer.max() <= 1.0

    def test_init_gaussian(self):
        """Test Gaussian initialization."""
        buffer = ReplayBuffer(
            capacity=10000, sample_dim=2,
            init_type="gaussian", init_std=0.5
        )

        assert buffer.buffer.shape == (10000, 2)
        np.testing.assert_allclose(buffer.buffer.mean(), 0.0, atol=0.05)
        np.testing.assert_allclose(buffer.buffer.std(), 0.5, rtol=0.1)

    def test_init_gaussian_default_std(self):
        """Test Gaussian initialization with default std=1.0."""
        buffer = ReplayBuffer(capacity=10000, sample_dim=2, init_type="gaussian")

        np.testing.assert_allclose(buffer.buffer.std(), 1.0, rtol=0.1)

    def test_init_invalid_capacity(self):
        """Test that invalid capacity raises ValueError."""
        with pytest.raises(ValueError, match="capacity must be positive"):
            ReplayBuffer(capacity=0, sample_dim=2)

        with pytest.raises(ValueError, match="capacity must be positive"):
            ReplayBuffer(capacity=-10, sample_dim=2)

    def test_init_invalid_sample_dim(self):
        """Test that invalid sample_dim raises ValueError."""
        with pytest.raises(ValueError, match="sample_dim must be positive"):
            ReplayBuffer(capacity=100, sample_dim=0)

        with pytest.raises(ValueError, match="sample_dim must be positive"):
            ReplayBuffer(capacity=100, sample_dim=-5)

    def test_init_invalid_type(self):
        """Test that invalid init_type raises ValueError."""
        with pytest.raises(ValueError, match="init_type must be"):
            ReplayBuffer(capacity=100, sample_dim=2, init_type="invalid")

    def test_init_various_capacities(self):
        """Test initialization with various capacities."""
        for capacity in [1, 10, 100, 1000, 10000]:
            buffer = ReplayBuffer(capacity=capacity, sample_dim=2)
            assert buffer.capacity == capacity
            assert len(buffer) == capacity

    def test_init_various_dimensions(self):
        """Test initialization with various dimensions."""
        for dim in [1, 2, 10, 50, 100]:
            buffer = ReplayBuffer(capacity=100, sample_dim=dim)
            assert buffer.sample_dim == dim
            assert buffer.buffer.shape == (100, dim)

    def test_init_asymmetric_bounds(self):
        """Test uniform initialization with asymmetric bounds."""
        buffer = ReplayBuffer(
            capacity=1000, sample_dim=2,
            init_type="uniform", init_low=0, init_high=5
        )

        assert buffer.buffer.min() >= 0
        assert buffer.buffer.max() <= 5
        np.testing.assert_allclose(buffer.buffer.mean(), 2.5, atol=0.2)


class TestReplayBufferSample:
    """Test ReplayBuffer sample method."""

    def test_sample_shape(self):
        """Test that sampled data has correct shape."""
        buffer = ReplayBuffer(capacity=100, sample_dim=5)
        samples, indices = buffer.sample(32)

        assert samples.shape == (32, 5)
        assert len(indices) == 32

    def test_sample_indices_valid(self):
        """Test that returned indices are valid buffer indices."""
        buffer = ReplayBuffer(capacity=1000, sample_dim=2)
        _samples, indices = buffer.sample(100)

        assert indices.min() >= 0
        assert indices.max() < 1000

    def test_sample_indices_dtype(self):
        """Test that indices have integer dtype."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        _, indices = buffer.sample(32)

        assert indices.dtype in (np.int32, np.int64)

    def test_sample_returns_copy(self):
        """Test that samples are copies, not views of buffer."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        samples, indices = buffer.sample(32)

        samples *= 0

        assert not np.allclose(buffer.buffer[indices], 0)

    def test_sample_no_reinit(self):
        """Test sampling with zero reinitialization."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        buffer.buffer[:] = 999

        samples, _indices = buffer.sample(100, reinit_prob=0.0)

        np.testing.assert_array_equal(samples, 999)

    def test_sample_full_reinit(self):
        """Test sampling with full reinitialization."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2, init_type="uniform")
        buffer.buffer[:] = 999

        samples, _ = buffer.sample(100, reinit_prob=1.0)

        assert not np.any(samples == 999)
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0

    def test_sample_reinit_ratio(self):
        """Test that approximate reinit ratio is respected."""
        buffer = ReplayBuffer(capacity=1000, sample_dim=2)
        buffer.buffer[:] = 999
        reinit_prob = 0.1

        samples, _ = buffer.sample(10000, reinit_prob=reinit_prob)

        reinit_count = (np.abs(samples - 999) > 1e-5).any(axis=1).sum()

        expected = 10000 * reinit_prob
        assert expected * 0.7 < reinit_count < expected * 1.3

    def test_sample_with_stats(self):
        """Test sampling with statistics returned."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        _samples, _indices, stats = buffer.sample(32, return_stats=True)

        assert isinstance(stats, ReplayBufferStats)
        assert stats.n_samples == 32
        assert 0 <= stats.reinit_ratio <= 1.0
        assert stats.sample_mean_norm >= 0
        assert stats.buffer_mean_norm >= 0

    def test_sample_stats_reinit_tracking(self):
        """Test that stats correctly track reinitialization."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        buffer.buffer[:] = 999

        _, _, stats = buffer.sample(1000, reinit_prob=0.0, return_stats=True)
        assert stats.n_reinit == 0
        assert stats.reinit_ratio == 0.0

        _, _, stats = buffer.sample(100, reinit_prob=1.0, return_stats=True)
        assert stats.n_reinit == 100
        assert stats.reinit_ratio == 1.0

    def test_sample_invalid_n(self):
        """Test that invalid n raises ValueError."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        with pytest.raises(ValueError, match="n must be positive"):
            buffer.sample(0)

        with pytest.raises(ValueError, match="n must be positive"):
            buffer.sample(-10)

    def test_sample_invalid_reinit_prob(self):
        """Test that invalid reinit_prob raises ValueError."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        with pytest.raises(ValueError, match="reinit_prob must be in"):
            buffer.sample(10, reinit_prob=-0.1)

        with pytest.raises(ValueError, match="reinit_prob must be in"):
            buffer.sample(10, reinit_prob=1.5)

    def test_sample_various_sizes(self):
        """Test sampling with various batch sizes."""
        buffer = ReplayBuffer(capacity=1000, sample_dim=5)

        for n in [1, 10, 64, 128, 500, 1000, 2000]:
            samples, indices = buffer.sample(n)
            assert samples.shape == (n, 5)
            assert len(indices) == n

    def test_sample_more_than_capacity(self):
        """Test sampling more samples than buffer capacity."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        samples, indices = buffer.sample(1000)

        assert samples.shape == (1000, 2)
        assert len(indices) == 1000
        assert indices.max() < 100

    def test_sample_gaussian_reinit(self):
        """Test reinitialization with Gaussian distribution."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2, init_type="gaussian", init_std=0.5)
        buffer.buffer[:] = 999

        samples, _ = buffer.sample(10000, reinit_prob=1.0)

        np.testing.assert_allclose(samples.mean(), 0.0, atol=0.05)
        np.testing.assert_allclose(samples.std(), 0.5, rtol=0.1)

    def test_sample_single(self):
        """Test sampling a single sample."""
        buffer = ReplayBuffer(capacity=100, sample_dim=5)
        samples, indices = buffer.sample(1)

        assert samples.shape == (1, 5)
        assert len(indices) == 1


class TestReplayBufferUpdate:
    """Test ReplayBuffer update method."""

    def test_update_basic(self):
        """Test basic update functionality."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        buffer.buffer[:] = 0

        indices = np.array([0, 1, 2])
        new_samples = np.array([[1, 1], [2, 2], [3, 3]])

        buffer.update(indices, new_samples)

        np.testing.assert_array_equal(buffer.buffer[0], [1, 1])
        np.testing.assert_array_equal(buffer.buffer[1], [2, 2])
        np.testing.assert_array_equal(buffer.buffer[2], [3, 3])

    def test_update_from_sample(self):
        """Test update workflow: sample -> modify -> update."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        samples, indices = buffer.sample(32, reinit_prob=0.0)

        new_samples = samples + 1.0

        buffer.update(indices, new_samples)

        unique_indices, unique_positions = np.unique(indices, return_index=True)
        expected = new_samples[unique_positions]
        np.testing.assert_array_equal(buffer.buffer[unique_indices], expected)

    def test_update_does_not_modify_others(self):
        """Test that update only modifies specified indices."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        original = buffer.buffer.copy()

        indices = np.arange(10)
        new_samples = np.random.randn(10, 2)

        buffer.update(indices, new_samples)

        np.testing.assert_array_equal(buffer.buffer[10:], original[10:])

    def test_update_duplicate_indices(self):
        """Test update with duplicate indices (last value wins)."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        indices = np.array([5, 5])
        new_samples = np.array([[1, 1], [2, 2]])

        buffer.update(indices, new_samples)

        np.testing.assert_array_equal(buffer.buffer[5], [2, 2])

    def test_update_empty(self):
        """Test update with empty arrays."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        original = buffer.buffer.copy()

        indices = np.array([], dtype=np.int64)
        new_samples = np.empty((0, 2))

        buffer.update(indices, new_samples)

        np.testing.assert_array_equal(buffer.buffer, original)

    def test_update_mismatched_lengths(self):
        """Test that mismatched lengths raise ValueError."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        indices = np.array([0, 1, 2])
        new_samples = np.array([[1, 1], [2, 2]])

        with pytest.raises(ValueError, match="same length"):
            buffer.update(indices, new_samples)

    def test_update_wrong_dim(self):
        """Test that wrong sample dimensionality raises ValueError."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        indices = np.array([0, 1, 2])
        new_samples = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

        with pytest.raises(ValueError, match="features"):
            buffer.update(indices, new_samples)

    def test_update_1d_samples(self):
        """Test that 1D samples array raises ValueError."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        indices = np.array([0, 1])
        new_samples = np.array([1, 2])

        with pytest.raises(ValueError, match="must be 2D"):
            buffer.update(indices, new_samples)

    def test_update_out_of_bounds(self):
        """Test that out of bounds indices raise IndexError."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        indices = np.array([0, 100])
        new_samples = np.array([[1, 1], [2, 2]])

        with pytest.raises(IndexError, match="out of bounds"):
            buffer.update(indices, new_samples)

    def test_update_negative_index(self):
        """Test that negative indices raise IndexError."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        indices = np.array([-1, 0])
        new_samples = np.array([[1, 1], [2, 2]])

        with pytest.raises(IndexError, match="out of bounds"):
            buffer.update(indices, new_samples)

    def test_update_all_indices(self):
        """Test updating all buffer positions."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        indices = np.arange(100)
        new_samples = np.random.randn(100, 2)

        buffer.update(indices, new_samples)

        np.testing.assert_array_equal(buffer.buffer, new_samples)


class TestReplayBufferPush:
    """Test ReplayBuffer push method (FIFO updates)."""

    def test_push_basic(self):
        """Test basic push functionality."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        buffer.buffer[:] = 0

        new_samples = np.array([[1, 1], [2, 2], [3, 3]])
        indices = buffer.push(new_samples)

        assert len(indices) == 3
        np.testing.assert_array_equal(indices, [0, 1, 2])
        np.testing.assert_array_equal(buffer.buffer[0], [1, 1])
        np.testing.assert_array_equal(buffer.buffer[1], [2, 2])
        np.testing.assert_array_equal(buffer.buffer[2], [3, 3])

    def test_push_position_advances(self):
        """Test that position advances after push."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        assert buffer.position == 0

        buffer.push(np.random.randn(10, 2))
        assert buffer.position == 10

        buffer.push(np.random.randn(5, 2))
        assert buffer.position == 15

    def test_push_wraps_around(self):
        """Test that push wraps around at capacity."""
        buffer = ReplayBuffer(capacity=10, sample_dim=2)

        samples = np.arange(30).reshape(15, 2).astype(float)
        indices = buffer.push(samples)

        np.testing.assert_array_equal(indices[:10], np.arange(10))
        np.testing.assert_array_equal(indices[10:], np.arange(5))

        np.testing.assert_array_equal(buffer.buffer[0], [20, 21])
        np.testing.assert_array_equal(buffer.buffer[5], [10, 11])

    def test_push_empty(self):
        """Test push with empty array."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        original_position = buffer.position

        indices = buffer.push(np.empty((0, 2)))

        assert len(indices) == 0
        assert buffer.position == original_position

    def test_push_wrong_dim(self):
        """Test that wrong dimensionality raises ValueError."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        with pytest.raises(ValueError, match="features"):
            buffer.push(np.random.randn(10, 3))

    def test_push_1d_array(self):
        """Test that 1D array raises ValueError."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        with pytest.raises(ValueError, match="must be 2D"):
            buffer.push(np.random.randn(10))

    def test_push_returns_indices(self):
        """Test that push returns correct indices."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        indices1 = buffer.push(np.random.randn(20, 2))
        np.testing.assert_array_equal(indices1, np.arange(20))

        indices2 = buffer.push(np.random.randn(30, 2))
        np.testing.assert_array_equal(indices2, np.arange(20, 50))


class TestReplayBufferManipulation:
    """Test buffer manipulation methods."""

    def test_get_all(self):
        """Test get_all returns copy of buffer."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        all_data = buffer.get_all()

        assert all_data.shape == (100, 2)
        np.testing.assert_array_equal(all_data, buffer.buffer)

        all_data *= 0
        assert not np.allclose(buffer.buffer, 0)

    def test_set_all(self):
        """Test set_all replaces entire buffer."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        new_data = np.random.randn(100, 2)
        buffer.set_all(new_data)

        np.testing.assert_array_equal(buffer.buffer, new_data)
        assert buffer.position == 0

    def test_set_all_copies_data(self):
        """Test that set_all copies the data."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        new_data = np.random.randn(100, 2)
        buffer.set_all(new_data)

        new_data *= 0

        assert not np.allclose(buffer.buffer, 0)

    def test_set_all_wrong_shape(self):
        """Test that wrong shape raises ValueError."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        with pytest.raises(ValueError, match="shape"):
            buffer.set_all(np.random.randn(50, 2))

        with pytest.raises(ValueError, match="shape"):
            buffer.set_all(np.random.randn(100, 3))

    def test_reset(self):
        """Test reset reinitializes buffer."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2, init_type="uniform")
        buffer.buffer[:] = 999
        buffer.position = 50
        buffer._total_samples_drawn = 1000
        buffer._total_reinit = 100

        buffer.reset()

        assert buffer.position == 0
        assert buffer._total_samples_drawn == 0
        assert buffer._total_reinit == 0
        assert buffer.buffer.min() >= -1.0
        assert buffer.buffer.max() <= 1.0

    def test_reset_gaussian(self):
        """Test reset with Gaussian initialization."""
        buffer = ReplayBuffer(capacity=10000, sample_dim=2, init_type="gaussian", init_std=0.5)
        buffer.buffer[:] = 999

        buffer.reset()

        np.testing.assert_allclose(buffer.buffer.std(), 0.5, rtol=0.1)


class TestReplayBufferStatistics:
    """Test buffer statistics methods."""

    def test_get_statistics_basic(self):
        """Test basic statistics retrieval."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        stats = buffer.get_statistics()

        assert stats["capacity"] == 100
        assert stats["sample_dim"] == 2
        assert "mean_norm" in stats
        assert "std_norm" in stats
        assert "min_norm" in stats
        assert "max_norm" in stats
        assert "mean" in stats
        assert "std" in stats

    def test_get_statistics_tracking(self):
        """Test that statistics track sampling history."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        buffer.sample(50, reinit_prob=0.1)
        buffer.sample(30, reinit_prob=0.2)

        stats = buffer.get_statistics()

        assert stats["total_samples_drawn"] == 80
        assert stats["total_reinit"] > 0
        assert 0 < stats["overall_reinit_ratio"] < 1

    def test_get_statistics_after_reset(self):
        """Test that reset clears statistics."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        buffer.sample(100, reinit_prob=0.5)
        buffer.reset()

        stats = buffer.get_statistics()
        assert stats["total_samples_drawn"] == 0
        assert stats["total_reinit"] == 0
        assert stats["overall_reinit_ratio"] == 0.0

    def test_replay_buffer_stats_to_dict(self):
        """Test ReplayBufferStats to_dict method."""
        stats = ReplayBufferStats(
            n_samples=100,
            n_reinit=10,
            reinit_ratio=0.1,
            sample_mean_norm=1.5,
            buffer_mean_norm=1.2,
        )

        d = stats.to_dict()

        assert d["n_samples"] == 100
        assert d["n_reinit"] == 10
        assert d["reinit_ratio"] == 0.1
        assert d["sample_mean_norm"] == 1.5
        assert d["buffer_mean_norm"] == 1.2


class TestReplayBufferDunder:
    """Test dunder methods."""

    def test_len(self):
        """Test __len__ returns capacity."""
        buffer = ReplayBuffer(capacity=123, sample_dim=5)
        assert len(buffer) == 123

    def test_repr(self):
        """Test __repr__ produces valid string."""
        buffer = ReplayBuffer(capacity=100, sample_dim=5, init_type="gaussian")
        repr_str = repr(buffer)

        assert "ReplayBuffer" in repr_str
        assert "100" in repr_str
        assert "5" in repr_str
        assert "gaussian" in repr_str


class TestReplayBufferConfig:
    """Test ReplayBufferConfig and factory methods."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ReplayBufferConfig()

        assert config.capacity == 10000
        assert config.sample_dim == 2
        assert config.reinit_prob == 0.05
        assert config.init_type == "uniform"
        assert config.init_low == -1.0
        assert config.init_high == 1.0
        assert config.init_std == 1.0

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = ReplayBufferConfig(
            capacity=5000,
            sample_dim=29,
            reinit_prob=0.1,
            init_type="gaussian",
            init_std=0.5,
        )

        assert config.capacity == 5000
        assert config.sample_dim == 29
        assert config.reinit_prob == 0.1
        assert config.init_type == "gaussian"
        assert config.init_std == 0.5

    def test_from_config(self):
        """Test creating buffer from config."""
        config = ReplayBufferConfig(
            capacity=500,
            sample_dim=10,
            init_type="gaussian",
            init_std=2.0,
        )

        buffer = ReplayBuffer.from_config(config)

        assert buffer.capacity == 500
        assert buffer.sample_dim == 10
        assert buffer.init_type == "gaussian"
        assert buffer.init_std == 2.0

    def test_create_replay_buffer_factory(self):
        """Test create_replay_buffer factory function."""
        buffer = create_replay_buffer(
            capacity=1000,
            sample_dim=5,
            init_type="gaussian",
            init_std=0.5,
        )

        assert buffer.capacity == 1000
        assert buffer.sample_dim == 5
        assert buffer.init_type == "gaussian"

    def test_from_data(self):
        """Test creating buffer from data."""
        data = np.random.randn(500, 3)
        buffer = ReplayBuffer.from_data(data)

        assert buffer.capacity == 500
        assert buffer.sample_dim == 3

    def test_from_data_with_capacity(self):
        """Test from_data with specified capacity."""
        data = np.random.randn(100, 2)
        buffer = ReplayBuffer.from_data(data, capacity=1000)

        assert buffer.capacity == 1000
        assert buffer.sample_dim == 2

    def test_from_data_with_noise(self):
        """Test from_data with added noise."""
        data = np.zeros((100, 2))
        buffer = ReplayBuffer.from_data(data, noise_std=0.5)

        np.testing.assert_allclose(buffer.buffer.mean(), 0.0, atol=0.1)
        np.testing.assert_allclose(buffer.buffer.std(), 0.5, rtol=0.2)

    def test_from_data_samples_from_data(self):
        """Test that from_data samples from input data."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        buffer = ReplayBuffer.from_data(data, capacity=100)

        for row in buffer.buffer:
            assert any(np.allclose(row, d) for d in data)

    def test_from_data_invalid_shape(self):
        """Test that invalid data shape raises ValueError."""
        data_1d = np.random.randn(100)

        with pytest.raises(ValueError, match="must be 2D"):
            ReplayBuffer.from_data(data_1d)


class TestReplayBufferEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_capacity(self):
        """Test buffer with capacity of 1."""
        buffer = ReplayBuffer(capacity=1, sample_dim=2)

        samples, indices = buffer.sample(10)
        assert samples.shape == (10, 2)
        assert (indices == 0).all()

        buffer.update(np.array([0]), np.array([[5, 5]]))
        assert np.allclose(buffer.buffer[0], [5, 5])

    def test_single_dimension(self):
        """Test buffer with 1D samples."""
        buffer = ReplayBuffer(capacity=100, sample_dim=1)

        samples, indices = buffer.sample(32)
        assert samples.shape == (32, 1)

        buffer.update(indices, np.random.randn(32, 1))

    def test_high_dimensional(self):
        """Test buffer with high-dimensional samples."""
        buffer = ReplayBuffer(capacity=500, sample_dim=100)

        samples, _indices = buffer.sample(64)
        assert samples.shape == (64, 100)
        assert np.isfinite(samples).all()

    def test_large_capacity(self):
        """Test buffer with large capacity."""
        buffer = ReplayBuffer(capacity=100000, sample_dim=2)

        samples, _indices = buffer.sample(1000)
        assert samples.shape == (1000, 2)

    def test_repeated_sample_update_cycle(self):
        """Test repeated sample-update cycles."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        for _ in range(10):
            samples, indices = buffer.sample(32, reinit_prob=0.05)
            refined = samples + np.random.randn(*samples.shape) * 0.01
            buffer.update(indices, refined)

        assert np.isfinite(buffer.buffer).all()

    def test_extreme_reinit_probs(self):
        """Test with extreme reinit probabilities."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        samples, _ = buffer.sample(100, reinit_prob=0.0)
        assert samples.shape == (100, 2)

        samples, _ = buffer.sample(100, reinit_prob=1.0)
        assert samples.shape == (100, 2)

    def test_buffer_preserves_dtype(self):
        """Test that buffer preserves float dtype."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        assert buffer.buffer.dtype == np.float64

        samples, _ = buffer.sample(32)
        assert samples.dtype == np.float64


class TestReplayBufferLangevinIntegration:
    """Integration tests combining ReplayBuffer with Langevin sampling."""

    def test_basic_training_loop(self):
        """Test basic training loop workflow."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        buffer = ReplayBuffer(capacity=500, sample_dim=2)

        for _ in range(3):
            x_init, indices = buffer.sample(64, reinit_prob=0.05)

            x_samples = langevin_sample(
                energy_fn, x_init, n_steps=10, step_size=0.01, grad_clip=0.03
            )

            buffer.update(indices, x_samples)

        assert np.isfinite(buffer.buffer).all()

    def test_full_training_simulation(self):
        """Simulate multiple training epochs."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        buffer = ReplayBuffer(capacity=1000, sample_dim=2)

        n_epochs = 5
        batch_size = 64

        for _epoch in range(n_epochs):
            for _ in range(3):
                x_init, indices = buffer.sample(batch_size, reinit_prob=0.05)

                x_samples = langevin_sample(
                    energy_fn, x_init, n_steps=20, step_size=0.01, grad_clip=0.03
                )

                buffer.update(indices, x_samples)

        assert np.isfinite(buffer.buffer).all()

        stats = buffer.get_statistics()
        assert stats["total_samples_drawn"] == n_epochs * 3 * batch_size

    def test_buffer_from_data_then_langevin(self):
        """Test initializing buffer from data, then running Langevin."""
        train_data = np.random.randn(500, 2)

        buffer = ReplayBuffer.from_data(train_data, capacity=1000)

        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])

        x_init, indices = buffer.sample(64, reinit_prob=0.05)
        x_samples = langevin_sample(
            energy_fn, x_init, n_steps=20, step_size=0.01, grad_clip=0.03
        )
        buffer.update(indices, x_samples)

        assert np.isfinite(buffer.buffer).all()

    def test_buffer_statistics_during_training(self):
        """Test tracking buffer statistics during training."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        buffer = ReplayBuffer(capacity=500, sample_dim=2)

        norm_history = []

        for _ in range(5):
            x_init, indices, stats = buffer.sample(64, reinit_prob=0.05, return_stats=True)
            norm_history.append(stats.buffer_mean_norm)

            x_samples = langevin_sample(
                energy_fn, x_init, n_steps=20, step_size=0.01, grad_clip=0.03
            )
            buffer.update(indices, x_samples)

        assert all(np.isfinite(n) for n in norm_history)

    def test_high_dimensional_integration(self):
        """Test with high-dimensional data (like tabular)."""
        dim = 29
        energy_fn = EnergyMLP(input_dim=dim, hidden_dims=[64, 64])
        buffer = ReplayBuffer(capacity=1000, sample_dim=dim)

        x_init, indices = buffer.sample(64, reinit_prob=0.05)
        x_samples = langevin_sample(
            energy_fn, x_init, n_steps=20, step_size=0.01, grad_clip=0.03
        )
        buffer.update(indices, x_samples)

        assert x_samples.shape == (64, dim)
        assert np.isfinite(buffer.buffer).all()


class TestReplayBufferReproducibility:
    """Test reproducibility with random seeds."""

    def test_init_reproducible(self):
        """Test that initialization is reproducible with seed."""
        np.random.seed(12345)
        buffer1 = ReplayBuffer(capacity=100, sample_dim=2)

        np.random.seed(12345)
        buffer2 = ReplayBuffer(capacity=100, sample_dim=2)

        np.testing.assert_array_equal(buffer1.buffer, buffer2.buffer)

    def test_sample_reproducible(self):
        """Test that sampling is reproducible with seed."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        np.random.seed(12345)
        samples1, indices1 = buffer.sample(32, reinit_prob=0.05)

        np.random.seed(12345)
        samples2, indices2 = buffer.sample(32, reinit_prob=0.05)

        np.testing.assert_array_equal(samples1, samples2)
        np.testing.assert_array_equal(indices1, indices2)

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        np.random.seed(12345)
        buffer1 = ReplayBuffer(capacity=100, sample_dim=2)

        np.random.seed(54321)
        buffer2 = ReplayBuffer(capacity=100, sample_dim=2)

        assert not np.allclose(buffer1.buffer, buffer2.buffer)


class TestReplayBufferScalability:
    """Test scalability with different sizes."""

    @pytest.mark.parametrize("capacity", [10, 100, 1000, 10000])
    def test_various_capacities(self, capacity):
        """Test with various capacities."""
        buffer = ReplayBuffer(capacity=capacity, sample_dim=2)

        samples, indices = buffer.sample(min(64, capacity))
        assert samples.shape == (min(64, capacity), 2)

        buffer.update(indices, samples + 0.1)
        assert np.isfinite(buffer.buffer).all()

    @pytest.mark.parametrize("sample_dim", [1, 2, 10, 50, 100])
    def test_various_dimensions(self, sample_dim):
        """Test with various dimensions."""
        buffer = ReplayBuffer(capacity=100, sample_dim=sample_dim)

        samples, _indices = buffer.sample(32)
        assert samples.shape == (32, sample_dim)

    @pytest.mark.parametrize("batch_size", [1, 32, 64, 128, 256])
    def test_various_batch_sizes(self, batch_size):
        """Test with various batch sizes."""
        buffer = ReplayBuffer(capacity=1000, sample_dim=5)

        samples, _indices = buffer.sample(batch_size)
        assert samples.shape == (batch_size, 5)

    @pytest.mark.parametrize("reinit_prob", [0.0, 0.01, 0.05, 0.1, 0.5, 1.0])
    def test_various_reinit_probs(self, reinit_prob):
        """Test with various reinit probabilities."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)

        samples, _indices = buffer.sample(100, reinit_prob=reinit_prob)
        assert samples.shape == (100, 2)
        assert np.isfinite(samples).all()


class TestReplayBufferMemorySafety:
    """Test memory safety and data isolation."""

    def test_sample_does_not_share_memory(self):
        """Test that sampled data doesn't share memory with buffer."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        samples, _indices = buffer.sample(32, reinit_prob=0.0)

        original_buffer = buffer.buffer.copy()
        samples[:] = 999

        np.testing.assert_array_equal(buffer.buffer, original_buffer)

    def test_update_does_not_share_memory(self):
        """Test that update copies data, doesn't share memory."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        new_data = np.array([[1, 2], [3, 4]])
        indices = np.array([0, 1])

        buffer.update(indices, new_data)

        new_data[:] = 999

        np.testing.assert_array_equal(buffer.buffer[0], [1, 2])
        np.testing.assert_array_equal(buffer.buffer[1], [3, 4])

    def test_get_all_does_not_share_memory(self):
        """Test that get_all returns a copy."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        all_data = buffer.get_all()

        original_buffer = buffer.buffer.copy()
        all_data[:] = 999

        np.testing.assert_array_equal(buffer.buffer, original_buffer)

    def test_set_all_does_not_share_memory(self):
        """Test that set_all copies data."""
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        new_data = np.random.randn(100, 2)

        buffer.set_all(new_data)
        buffer_copy = buffer.buffer.copy()

        new_data[:] = 999

        np.testing.assert_array_equal(buffer.buffer, buffer_copy)
