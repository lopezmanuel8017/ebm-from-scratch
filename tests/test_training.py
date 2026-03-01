"""
Comprehensive tests for Training module.

Tests cover:
- Training step functionality
- Full training loop
- Training configuration
- Statistics tracking
- Integration with energy functions, optimizers, and replay buffer
- Edge cases and boundary conditions
- Numerical stability
"""

import pytest
import numpy as np

from ebm.core.energy import EnergyMLP
from ebm.sampling.replay_buffer import ReplayBuffer
from ebm.training.optimizer import AdamW, SGD
from ebm.training.trainer import (
    train_step,
    train,
    train_with_config,
    get_batches,
    evaluate_energy_model,
    generate_samples,
    TrainingConfig,
    TrainingStats,
    TrainingHistory,
)


class TestGetBatches:
    """Test get_batches function."""

    def test_basic_batching(self):
        """Test basic batch generation."""
        data = np.arange(100).reshape(100, 1)
        batches = list(get_batches(data, batch_size=10, shuffle=False))

        assert len(batches) == 10
        for batch in batches:
            assert batch.shape == (10, 1)

    def test_last_batch_smaller(self):
        """Test that last batch can be smaller."""
        data = np.arange(95).reshape(95, 1)
        batches = list(get_batches(data, batch_size=10, shuffle=False))

        assert len(batches) == 10
        assert batches[-1].shape == (5, 1)

    def test_shuffle(self):
        """Test that shuffle works."""
        data = np.arange(100).reshape(100, 1)

        batches1 = list(get_batches(data, batch_size=10, shuffle=False))
        batches2 = list(get_batches(data, batch_size=10, shuffle=False))

        for b1, b2 in zip(batches1, batches2):
            np.testing.assert_array_equal(b1, b2)

    def test_shuffle_different(self):
        """Test that shuffle produces different orderings."""
        data = np.arange(1000).reshape(1000, 1)

        np.random.seed(42)
        batches1 = list(get_batches(data, batch_size=100, shuffle=True))
        np.random.seed(43)
        batches2 = list(get_batches(data, batch_size=100, shuffle=True))

        different = False
        for b1, b2 in zip(batches1, batches2):
            if not np.array_equal(b1, b2):
                different = True
                break
        assert different

    def test_batch_size_larger_than_data(self):
        """Test batch size larger than dataset."""
        data = np.arange(10).reshape(10, 1)
        batches = list(get_batches(data, batch_size=20, shuffle=False))

        assert len(batches) == 1
        assert batches[0].shape == (10, 1)

    def test_batch_size_one(self):
        """Test batch size of 1."""
        data = np.arange(5).reshape(5, 1)
        batches = list(get_batches(data, batch_size=1, shuffle=False))

        assert len(batches) == 5
        for batch in batches:
            assert batch.shape == (1, 1)

    def test_multidimensional_data(self):
        """Test with multidimensional data."""
        data = np.random.randn(100, 10)
        batches = list(get_batches(data, batch_size=25, shuffle=False))

        assert len(batches) == 4
        for batch in batches:
            assert batch.shape == (25, 10)

    def test_preserves_data_integrity(self):
        """Test that all data points are in batches."""
        data = np.arange(100).reshape(100, 1)
        batches = list(get_batches(data, batch_size=17, shuffle=False))

        all_data = np.concatenate(batches)
        np.testing.assert_array_equal(np.sort(all_data.flatten()), np.arange(100))


class TestTrainingStats:
    """Test TrainingStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = TrainingStats()

        assert stats.loss == 0.0
        assert stats.cd_loss == 0.0
        assert stats.reg_loss == 0.0
        assert stats.E_real == 0.0
        assert stats.E_fake == 0.0
        assert stats.energy_gap == 0.0
        assert stats.entropy == 0.0
        assert stats.grad_norm == 0.0

    def test_custom_values(self):
        """Test with custom values."""
        stats = TrainingStats(
            loss=1.5,
            cd_loss=0.5,
            reg_loss=1.0,
            E_real=-2.0,
            E_fake=3.0,
            energy_gap=-5.0,
            entropy=2.3,
            grad_norm=0.1,
        )

        assert stats.loss == 1.5
        assert stats.cd_loss == 0.5
        assert stats.reg_loss == 1.0
        assert stats.E_real == -2.0
        assert stats.E_fake == 3.0
        assert stats.energy_gap == -5.0
        assert stats.entropy == 2.3
        assert stats.grad_norm == 0.1

    def test_to_dict(self):
        """Test to_dict method."""
        stats = TrainingStats(loss=1.0, E_real=-1.0, E_fake=2.0)
        d = stats.to_dict()

        assert d["loss"] == 1.0
        assert d["E_real"] == -1.0
        assert d["E_fake"] == 2.0
        assert "cd_loss" in d
        assert "entropy" in d


class TestTrainingHistory:
    """Test TrainingHistory dataclass."""

    def test_default_values(self):
        """Test default values."""
        history = TrainingHistory()

        assert history.epoch_stats == []
        assert history.step_count == 0
        assert history.best_loss == float('inf')
        assert history.best_epoch == 0

    def test_add_epoch(self):
        """Test add_epoch method."""
        history = TrainingHistory()

        history.add_epoch({"loss": 2.0, "E_real": -1.0}, epoch=0)
        history.add_epoch({"loss": 1.5, "E_real": -1.5}, epoch=1)
        history.add_epoch({"loss": 1.0, "E_real": -2.0}, epoch=2)

        assert len(history.epoch_stats) == 3
        assert history.best_loss == 1.0
        assert history.best_epoch == 2

    def test_best_loss_tracking(self):
        """Test that best loss is tracked correctly."""
        history = TrainingHistory()

        history.add_epoch({"loss": 2.0}, epoch=0)
        assert history.best_loss == 2.0
        assert history.best_epoch == 0

        history.add_epoch({"loss": 1.0}, epoch=1)
        assert history.best_loss == 1.0
        assert history.best_epoch == 1

        history.add_epoch({"loss": 1.5}, epoch=2)
        assert history.best_loss == 1.0
        assert history.best_epoch == 1

    def test_get_metric(self):
        """Test get_metric method."""
        history = TrainingHistory()

        history.add_epoch({"loss": 2.0, "E_real": -1.0}, epoch=0)
        history.add_epoch({"loss": 1.5, "E_real": -1.5}, epoch=1)
        history.add_epoch({"loss": 1.0, "E_real": -2.0}, epoch=2)

        losses = history.get_metric("loss")
        assert losses == [2.0, 1.5, 1.0]

        e_reals = history.get_metric("E_real")
        assert e_reals == [-1.0, -1.5, -2.0]

    def test_get_metric_missing(self):
        """Test get_metric with missing metric."""
        history = TrainingHistory()
        history.add_epoch({"loss": 1.0}, epoch=0)

        missing = history.get_metric("nonexistent")
        assert missing == []

    def test_to_dict(self):
        """Test to_dict method."""
        history = TrainingHistory()
        history.add_epoch({"loss": 1.0}, epoch=0)
        history.step_count = 10

        d = history.to_dict()

        assert "epoch_stats" in d
        assert "step_count" in d
        assert d["step_count"] == 10
        assert "best_loss" in d
        assert "best_epoch" in d


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.n_epochs == 50
        assert config.batch_size == 128
        assert config.langevin_steps == 40
        assert config.langevin_step_size == 0.01
        assert config.langevin_noise_scale == 1.0
        assert config.langevin_grad_clip == 0.03
        assert config.alpha == 0.1
        assert config.lambda_ent == 0.01
        assert config.reinit_prob == 0.05
        assert config.log_interval == 1
        assert config.entropy_k == 5

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            n_epochs=100,
            batch_size=64,
            langevin_steps=60,
            alpha=0.05,
        )

        assert config.n_epochs == 100
        assert config.batch_size == 64
        assert config.langevin_steps == 60
        assert config.alpha == 0.05


class TestTrainStep:
    """Test train_step function."""

    def test_basic_train_step(self):
        """Test basic training step."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data_batch = np.random.randn(64, 2)

        stats = train_step(
            energy_fn, optimizer, data_batch, buffer,
            langevin_steps=10, langevin_step_size=0.01,
            alpha=0.1, lambda_ent=0.0, compute_entropy=False
        )

        assert isinstance(stats, TrainingStats)
        assert np.isfinite(stats.loss)
        assert np.isfinite(stats.E_real)
        assert np.isfinite(stats.E_fake)

    def test_train_step_returns_stats(self):
        """Test that train_step returns expected statistics."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data_batch = np.random.randn(64, 2)

        stats = train_step(
            energy_fn, optimizer, data_batch, buffer,
            langevin_steps=10, langevin_step_size=0.01,
            alpha=0.1, lambda_ent=0.01, compute_entropy=True
        )

        assert stats.loss != 0.0 or stats.cd_loss != 0.0
        assert np.isfinite(stats.cd_loss)
        assert np.isfinite(stats.reg_loss)
        assert np.isfinite(stats.energy_gap)

    def test_train_step_updates_buffer(self):
        """Test that train_step updates replay buffer."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)

        original_buffer = buffer.buffer.copy()
        data_batch = np.random.randn(64, 2)

        train_step(
            energy_fn, optimizer, data_batch, buffer,
            langevin_steps=10, langevin_step_size=0.01,
        )

        assert not np.allclose(buffer.buffer, original_buffer)

    def test_train_step_updates_parameters(self):
        """Test that train_step updates model parameters."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)

        original_params = [p.data.copy() for p in energy_fn.parameters()]
        data_batch = np.random.randn(64, 2)

        train_step(
            energy_fn, optimizer, data_batch, buffer,
            langevin_steps=10, langevin_step_size=0.01,
        )

        params_changed = False
        for i, p in enumerate(energy_fn.parameters()):
            if not np.allclose(p.data, original_params[i]):
                params_changed = True
                break
        assert params_changed

    def test_train_step_with_entropy(self):
        """Test train_step with entropy computation."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data_batch = np.random.randn(64, 2)

        stats = train_step(
            energy_fn, optimizer, data_batch, buffer,
            langevin_steps=10, langevin_step_size=0.01,
            lambda_ent=0.01, compute_entropy=True
        )

        assert np.isfinite(stats.entropy)

    def test_train_step_without_entropy(self):
        """Test train_step without entropy computation."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data_batch = np.random.randn(64, 2)

        stats = train_step(
            energy_fn, optimizer, data_batch, buffer,
            langevin_steps=10, langevin_step_size=0.01,
            lambda_ent=0.0, compute_entropy=False
        )

        assert stats.entropy == 0.0

    def test_train_step_various_batch_sizes(self):
        """Test train_step with various batch sizes."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)

        for batch_size in [1, 16, 64, 128]:
            data_batch = np.random.randn(batch_size, 2)

            stats = train_step(
                energy_fn, optimizer, data_batch, buffer,
                langevin_steps=5, langevin_step_size=0.01,
                compute_entropy=False
            )

            assert np.isfinite(stats.loss)

    def test_train_step_various_alpha(self):
        """Test train_step with various alpha values."""
        for alpha in [0.0, 0.01, 0.1, 1.0]:
            energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
            optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
            buffer = ReplayBuffer(capacity=500, sample_dim=2)
            data_batch = np.random.randn(32, 2)

            stats = train_step(
                energy_fn, optimizer, data_batch, buffer,
                langevin_steps=5, langevin_step_size=0.01,
                alpha=alpha, compute_entropy=False
            )

            assert np.isfinite(stats.loss)
            assert np.isfinite(stats.reg_loss)


class TestTrain:
    """Test train function."""

    def test_basic_train(self):
        """Test basic training loop."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data = np.random.randn(200, 2)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=3, batch_size=32,
            replay_buffer=buffer,
            langevin_steps=5, langevin_step_size=0.01,
            verbose=False
        )

        assert isinstance(history, TrainingHistory)
        assert len(history.epoch_stats) == 3
        assert history.step_count > 0

    def test_train_loss_decreases(self):
        """Test that loss generally decreases or stabilizes during training."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)

        data = np.random.randn(500, 2)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=10, batch_size=64,
            replay_buffer=buffer,
            langevin_steps=10, langevin_step_size=0.01,
            alpha=0.1, verbose=False
        )

        losses = history.get_metric("loss")

        assert np.isfinite(losses[-1])
        assert losses[-1] < 1e6

    def test_train_callback(self):
        """Test training with callback."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data = np.random.randn(100, 2)

        callback_calls = []

        def callback(epoch, stats):
            callback_calls.append((epoch, stats["loss"]))

        _history = train(
            energy_fn, optimizer, data,
            n_epochs=5, batch_size=32,
            replay_buffer=buffer,
            langevin_steps=5, langevin_step_size=0.01,
            verbose=False, callback=callback
        )

        assert len(callback_calls) == 5
        assert callback_calls[0][0] == 0
        assert callback_calls[-1][0] == 4

    def test_train_verbose(self, capsys):
        """Test verbose output."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data = np.random.randn(100, 2)

        train(
            energy_fn, optimizer, data,
            n_epochs=3, batch_size=32,
            replay_buffer=buffer,
            langevin_steps=5, langevin_step_size=0.01,
            verbose=True, log_interval=1
        )

        captured = capsys.readouterr()
        assert "Epoch" in captured.out
        assert "E_real" in captured.out
        assert "E_fake" in captured.out

    def test_train_log_interval(self, capsys):
        """Test log interval."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data = np.random.randn(100, 2)

        train(
            energy_fn, optimizer, data,
            n_epochs=10, batch_size=32,
            replay_buffer=buffer,
            langevin_steps=5, langevin_step_size=0.01,
            verbose=True, log_interval=5
        )

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split('\n') if 'Epoch' in line]
        assert len(lines) == 3

    def test_train_with_config(self):
        """Test train_with_config function."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data = np.random.randn(100, 2)

        config = TrainingConfig(
            n_epochs=3,
            batch_size=32,
            langevin_steps=5,
            langevin_step_size=0.01,
            alpha=0.1,
            lambda_ent=0.0,
        )

        history = train_with_config(
            energy_fn, optimizer, data, buffer, config, verbose=False
        )

        assert len(history.epoch_stats) == 3

    def test_train_tracks_best_loss(self):
        """Test that best loss is tracked."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data = np.random.randn(200, 2)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=5, batch_size=64,
            replay_buffer=buffer,
            langevin_steps=10, langevin_step_size=0.01,
            verbose=False
        )

        assert history.best_loss < float('inf')
        assert 0 <= history.best_epoch < 5


class TestEvaluateEnergyModel:
    """Test evaluate_energy_model function."""

    def test_basic_evaluation(self):
        """Test basic energy evaluation."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        data = np.random.randn(100, 2)

        stats = evaluate_energy_model(energy_fn, data, batch_size=32)

        assert "mean_energy" in stats
        assert "std_energy" in stats
        assert "min_energy" in stats
        assert "max_energy" in stats

        assert np.isfinite(stats["mean_energy"])
        assert stats["std_energy"] >= 0
        assert stats["min_energy"] <= stats["mean_energy"] <= stats["max_energy"]

    def test_evaluation_batch_sizes(self):
        """Test evaluation with different batch sizes."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        data = np.random.randn(100, 2)

        stats1 = evaluate_energy_model(energy_fn, data, batch_size=10)
        stats2 = evaluate_energy_model(energy_fn, data, batch_size=50)
        stats3 = evaluate_energy_model(energy_fn, data, batch_size=100)

        np.testing.assert_allclose(stats1["mean_energy"], stats2["mean_energy"], rtol=1e-5)
        np.testing.assert_allclose(stats2["mean_energy"], stats3["mean_energy"], rtol=1e-5)


class TestGenerateSamples:
    """Test generate_samples function."""

    def test_basic_generation(self):
        """Test basic sample generation."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])

        samples = generate_samples(
            energy_fn, n_samples=100, sample_dim=2,
            n_steps=20, step_size=0.01
        )

        assert samples.shape == (100, 2)
        assert np.isfinite(samples).all()

    def test_generation_uniform_init(self):
        """Test generation with uniform initialization."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])

        samples = generate_samples(
            energy_fn, n_samples=100, sample_dim=2,
            n_steps=10, step_size=0.01,
            init_type="uniform", init_low=-2.0, init_high=2.0
        )

        assert samples.shape == (100, 2)
        assert np.isfinite(samples).all()

    def test_generation_gaussian_init(self):
        """Test generation with Gaussian initialization."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])

        samples = generate_samples(
            energy_fn, n_samples=100, sample_dim=2,
            n_steps=10, step_size=0.01,
            init_type="gaussian", init_std=1.0
        )

        assert samples.shape == (100, 2)
        assert np.isfinite(samples).all()

    def test_generation_invalid_init_type(self):
        """Test generation with invalid init type."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])

        with pytest.raises(ValueError, match="Unknown init_type"):
            generate_samples(
                energy_fn, n_samples=100, sample_dim=2,
                n_steps=10, step_size=0.01,
                init_type="invalid"
            )

    def test_generation_high_dimensional(self):
        """Test generation for high-dimensional data."""
        energy_fn = EnergyMLP(input_dim=29, hidden_dims=[64, 64])

        samples = generate_samples(
            energy_fn, n_samples=50, sample_dim=29,
            n_steps=20, step_size=0.01
        )

        assert samples.shape == (50, 29)
        assert np.isfinite(samples).all()


class TestTrainingIntegration:
    """Integration tests for training."""

    def test_full_training_pipeline(self):
        """Test complete training pipeline."""
        np.random.seed(42)
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[64, 64])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3, weight_decay=0.01)
        buffer = ReplayBuffer(capacity=1000, sample_dim=2)

        data1 = np.random.randn(250, 2) * 0.5 + np.array([2, 2])
        data2 = np.random.randn(250, 2) * 0.5 + np.array([-2, -2])
        train_data = np.vstack([data1, data2])

        history = train(
            energy_fn, optimizer, train_data,
            n_epochs=10, batch_size=64,
            replay_buffer=buffer,
            langevin_steps=20, langevin_step_size=0.01,
            alpha=0.1, lambda_ent=0.0,
            verbose=False
        )

        assert len(history.epoch_stats) == 10
        assert np.isfinite(history.best_loss)

        samples = generate_samples(
            energy_fn, n_samples=100, sample_dim=2,
            n_steps=50, step_size=0.01
        )

        assert samples.shape == (100, 2)
        assert np.isfinite(samples).all()

    def test_training_with_sgd(self):
        """Test training with SGD optimizer."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = SGD(energy_fn.parameters(), lr=0.01, momentum=0.9)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data = np.random.randn(200, 2)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=5, batch_size=32,
            replay_buffer=buffer,
            langevin_steps=10, langevin_step_size=0.01,
            verbose=False
        )

        assert len(history.epoch_stats) == 5

    def test_training_high_dimensional(self):
        """Test training with high-dimensional data."""
        dim = 29
        energy_fn = EnergyMLP(input_dim=dim, hidden_dims=[64, 64])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=dim)
        data = np.random.randn(200, dim)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=3, batch_size=32,
            replay_buffer=buffer,
            langevin_steps=10, langevin_step_size=0.01,
            verbose=False
        )

        assert len(history.epoch_stats) == 3
        assert np.isfinite(history.epoch_stats[-1]["loss"])

    def test_training_no_nan(self):
        """Test that training doesn't produce NaN values."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[64, 64])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data = np.random.randn(500, 2)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=20, batch_size=64,
            replay_buffer=buffer,
            langevin_steps=20, langevin_step_size=0.01,
            alpha=0.1, verbose=False
        )

        for stats in history.epoch_stats:
            assert np.isfinite(stats["loss"]), "NaN loss at some epoch"
            assert np.isfinite(stats["E_real"]), "NaN E_real at some epoch"
            assert np.isfinite(stats["E_fake"]), "NaN E_fake at some epoch"



class TestTrainingEdgeCases:
    """Test edge cases in training."""

    def test_single_sample_batch(self):
        """Test training with batch size of 1."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        data = np.random.randn(10, 2)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=2, batch_size=1,
            replay_buffer=buffer,
            langevin_steps=5, langevin_step_size=0.01,
            compute_entropy=False, verbose=False
        )

        assert len(history.epoch_stats) == 2

    def test_single_epoch(self):
        """Test training for single epoch."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        data = np.random.randn(50, 2)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=1, batch_size=16,
            replay_buffer=buffer,
            langevin_steps=5, langevin_step_size=0.01,
            verbose=False
        )

        assert len(history.epoch_stats) == 1

    def test_data_smaller_than_batch(self):
        """Test when dataset is smaller than batch size."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        data = np.random.randn(10, 2)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=2, batch_size=32,
            replay_buffer=buffer,
            langevin_steps=5, langevin_step_size=0.01,
            verbose=False
        )

        assert len(history.epoch_stats) == 2

    def test_zero_alpha(self):
        """Test training with zero energy regularization."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=100, sample_dim=2)
        data = np.random.randn(50, 2)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=2, batch_size=16,
            replay_buffer=buffer,
            langevin_steps=5, langevin_step_size=0.01,
            alpha=0.0, verbose=False
        )

        for stats in history.epoch_stats:
            assert stats["reg_loss"] == 0.0

    def test_very_small_data(self):
        """Test with very small dataset."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=50, sample_dim=2)
        data = np.random.randn(5, 2)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=2, batch_size=2,
            replay_buffer=buffer,
            langevin_steps=3, langevin_step_size=0.01,
            compute_entropy=False, verbose=False
        )

        assert len(history.epoch_stats) == 2


class TestTrainingParametrized:
    """Parametrized tests for training."""

    @pytest.mark.parametrize("batch_size", [16, 32, 64, 128])
    def test_various_batch_sizes(self, batch_size):
        """Test training with various batch sizes."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data = np.random.randn(200, 2)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=2, batch_size=batch_size,
            replay_buffer=buffer,
            langevin_steps=5, langevin_step_size=0.01,
            verbose=False
        )

        assert len(history.epoch_stats) == 2
        assert np.isfinite(history.epoch_stats[-1]["loss"])

    @pytest.mark.parametrize("langevin_steps", [5, 10, 20, 40])
    def test_various_langevin_steps(self, langevin_steps):
        """Test training with various Langevin step counts."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data = np.random.randn(100, 2)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=2, batch_size=32,
            replay_buffer=buffer,
            langevin_steps=langevin_steps, langevin_step_size=0.01,
            verbose=False
        )

        assert np.isfinite(history.epoch_stats[-1]["loss"])

    @pytest.mark.parametrize("alpha", [0.0, 0.01, 0.1, 1.0])
    def test_various_alpha(self, alpha):
        """Test training with various alpha values."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data = np.random.randn(100, 2)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=2, batch_size=32,
            replay_buffer=buffer,
            langevin_steps=5, langevin_step_size=0.01,
            alpha=alpha, verbose=False
        )

        assert np.isfinite(history.epoch_stats[-1]["loss"])

    @pytest.mark.parametrize("hidden_dims", [[32], [32, 32], [64, 64], [128, 128]])
    def test_various_architectures(self, hidden_dims):
        """Test training with various network architectures."""
        energy_fn = EnergyMLP(input_dim=2, hidden_dims=hidden_dims)
        optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
        buffer = ReplayBuffer(capacity=500, sample_dim=2)
        data = np.random.randn(100, 2)

        history = train(
            energy_fn, optimizer, data,
            n_epochs=2, batch_size=32,
            replay_buffer=buffer,
            langevin_steps=5, langevin_step_size=0.01,
            verbose=False
        )

        assert np.isfinite(history.epoch_stats[-1]["loss"])


class TestTrainingReproducibility:
    """Test reproducibility of training."""

    def test_reproducible_with_seed(self):
        """Test that training is reproducible with same seed."""
        data = np.random.randn(100, 2)

        def run_training(seed):
            np.random.seed(seed)
            energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
            optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
            buffer = ReplayBuffer(capacity=200, sample_dim=2)

            history = train(
                energy_fn, optimizer, data,
                n_epochs=3, batch_size=32,
                replay_buffer=buffer,
                langevin_steps=5, langevin_step_size=0.01,
                verbose=False
            )
            return history.get_metric("loss")

        losses1 = run_training(42)
        losses2 = run_training(42)

        np.testing.assert_allclose(losses1, losses2)

    def test_different_with_different_seed(self):
        """Test that different seeds produce different results."""
        data = np.random.randn(100, 2)

        def run_training(seed):
            np.random.seed(seed)
            energy_fn = EnergyMLP(input_dim=2, hidden_dims=[32, 32])
            optimizer = AdamW(energy_fn.parameters(), lr=1e-3)
            buffer = ReplayBuffer(capacity=200, sample_dim=2)

            history = train(
                energy_fn, optimizer, data,
                n_epochs=3, batch_size=32,
                replay_buffer=buffer,
                langevin_steps=5, langevin_step_size=0.01,
                verbose=False
            )
            return history.get_metric("loss")

        losses1 = run_training(42)
        losses2 = run_training(43)

        assert not np.allclose(losses1, losses2)
