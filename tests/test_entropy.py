"""
Comprehensive tests for k-NN Entropy Estimator.

Tests cover:
- Mathematical utility functions (digamma, log_gamma, gamma, unit_ball_volume)
- Distance computation functions (pairwise_distances, kth_nearest_distances)
- Core entropy estimator (knn_entropy, knn_entropy_batch)
- Class-based estimator (KNNEntropyEstimator)
- Configuration and statistics classes
- Known distribution validation (Gaussian, Uniform)
- Edge cases and boundary conditions
- Numerical stability
- Reproducibility
"""

import pytest
import numpy as np

from ebm.entropy.knn import (
    digamma,
    log_gamma,
    gamma,
    unit_ball_volume,
    pairwise_distances,
    kth_nearest_distances,
    knn_entropy,
    knn_entropy_batch,
    KNNEntropyEstimator,
    KNNEntropyConfig,
    KNNEntropyStats,
    entropy_gaussian,
    entropy_uniform,
    mutual_information_knn,
)


class TestDigamma:
    """Test digamma function implementation."""

    def test_digamma_positive_integers(self):
        """Test digamma at positive integers."""
        euler_gamma = 0.5772156649015329
        np.testing.assert_allclose(digamma(1), -euler_gamma, rtol=1e-6)

        np.testing.assert_allclose(digamma(2), 1 - euler_gamma, rtol=1e-6)

        np.testing.assert_allclose(digamma(3), 1.5 - euler_gamma, rtol=1e-6)

    def test_digamma_large_values(self):
        """Test digamma for large values (asymptotic regime)."""
        for x in [10.0, 50.0, 100.0, 1000.0]:
            asymptotic = np.log(x) - 0.5 / x
            result = digamma(x)
            np.testing.assert_allclose(result, asymptotic, rtol=0.01)

    def test_digamma_small_values(self):
        """Test digamma for small values (uses recurrence)."""
        np.testing.assert_allclose(digamma(0.5), -1.9635100260214235, rtol=1e-6)
        np.testing.assert_allclose(digamma(0.1), -10.423754940411075, rtol=1e-5)

    def test_digamma_array_input(self):
        """Test digamma with array input."""
        x = np.array([1.0, 2.0, 5.0, 10.0])
        result = digamma(x)

        assert result.shape == (4,)
        for i, xi in enumerate(x):
            np.testing.assert_allclose(result[i], digamma(xi), rtol=1e-10)

    def test_digamma_invalid_input(self):
        """Test digamma raises error for non-positive input."""
        with pytest.raises(ValueError, match="positive"):
            digamma(0.0)

        with pytest.raises(ValueError, match="positive"):
            digamma(-1.0)

        with pytest.raises(ValueError, match="positive"):
            digamma(np.array([1.0, -1.0]))

    def test_digamma_scalar_return(self):
        """Test that scalar input returns scalar output."""
        result = digamma(5.0)
        assert isinstance(result, float)

    def test_digamma_monotonic(self):
        """Test that digamma is monotonically increasing for x > 0."""
        x = np.linspace(0.1, 10, 100)
        psi = digamma(x)
        assert np.all(np.diff(psi) > 0)


class TestLogGamma:
    """Test log-gamma function implementation."""

    def test_log_gamma_integers(self):
        """Test log_gamma at positive integers (log of factorials)."""
        np.testing.assert_allclose(log_gamma(1), 0.0, atol=1e-10)
        np.testing.assert_allclose(log_gamma(2), 0.0, atol=1e-10)
        np.testing.assert_allclose(log_gamma(3), np.log(2), rtol=1e-6)
        np.testing.assert_allclose(log_gamma(4), np.log(6), rtol=1e-6)
        np.testing.assert_allclose(log_gamma(5), np.log(24), rtol=1e-6)
        np.testing.assert_allclose(log_gamma(6), np.log(120), rtol=1e-6)

    def test_log_gamma_half_integers(self):
        """Test log_gamma at half-integers."""
        np.testing.assert_allclose(log_gamma(0.5), 0.5 * np.log(np.pi), rtol=1e-6)
        np.testing.assert_allclose(log_gamma(1.5), np.log(np.sqrt(np.pi) / 2), rtol=1e-6)
        np.testing.assert_allclose(log_gamma(2.5), np.log(3 * np.sqrt(np.pi) / 4), rtol=1e-6)

    def test_log_gamma_large_values(self):
        """Test log_gamma for large values (Stirling's regime)."""
        for x in [10.0, 50.0, 100.0]:
            stirling = (x - 0.5) * np.log(x) - x + 0.5 * np.log(2 * np.pi)
            result = log_gamma(x)
            np.testing.assert_allclose(result, stirling, rtol=0.01)

    def test_log_gamma_array_input(self):
        """Test log_gamma with array input."""
        x = np.array([1.0, 2.0, 5.0, 10.0])
        result = log_gamma(x)

        assert result.shape == (4,)
        for i, xi in enumerate(x):
            np.testing.assert_allclose(result[i], log_gamma(xi), rtol=1e-10)

    def test_log_gamma_invalid_input(self):
        """Test log_gamma raises error for non-positive input."""
        with pytest.raises(ValueError, match="positive"):
            log_gamma(0.0)

        with pytest.raises(ValueError, match="positive"):
            log_gamma(-1.0)


class TestGamma:
    """Test gamma function implementation."""

    def test_gamma_integers(self):
        """Test gamma at positive integers (factorials)."""
        np.testing.assert_allclose(gamma(1), 1.0, rtol=1e-6)
        np.testing.assert_allclose(gamma(2), 1.0, rtol=1e-6)
        np.testing.assert_allclose(gamma(3), 2.0, rtol=1e-6)
        np.testing.assert_allclose(gamma(4), 6.0, rtol=1e-6)
        np.testing.assert_allclose(gamma(5), 24.0, rtol=1e-6)

    def test_gamma_half_integers(self):
        """Test gamma at half-integers."""
        np.testing.assert_allclose(gamma(0.5), np.sqrt(np.pi), rtol=1e-6)
        np.testing.assert_allclose(gamma(1.5), np.sqrt(np.pi) / 2, rtol=1e-6)

    def test_gamma_consistency_with_log_gamma(self):
        """Test that gamma(x) = exp(log_gamma(x))."""
        for x in [0.5, 1.0, 2.5, 5.0, 10.0]:
            np.testing.assert_allclose(gamma(x), np.exp(log_gamma(x)), rtol=1e-10)


class TestUnitBallVolume:
    """Test unit ball volume computation."""

    def test_unit_ball_1d(self):
        """Test 1D unit ball volume (length of interval [-1, 1])."""
        np.testing.assert_allclose(unit_ball_volume(1), 2.0, rtol=1e-6)

    def test_unit_ball_2d(self):
        """Test 2D unit ball volume (area of unit circle: pi)."""
        np.testing.assert_allclose(unit_ball_volume(2), np.pi, rtol=1e-6)

    def test_unit_ball_3d(self):
        """Test 3D unit ball volume (volume of unit sphere: 4/3 * pi)."""
        np.testing.assert_allclose(unit_ball_volume(3), 4.0 / 3.0 * np.pi, rtol=1e-6)

    def test_unit_ball_4d(self):
        """Test 4D unit ball volume (pi^2 / 2)."""
        np.testing.assert_allclose(unit_ball_volume(4), np.pi ** 2 / 2, rtol=1e-6)

    def test_unit_ball_high_dimensions(self):
        """Test unit ball volume decreases for high dimensions."""
        volumes = [unit_ball_volume(d) for d in range(5, 20)]
        for i in range(1, len(volumes)):
            assert volumes[i] < volumes[i - 1]

    def test_unit_ball_invalid_dimension(self):
        """Test unit_ball_volume raises error for non-positive dimension."""
        with pytest.raises(ValueError, match="positive"):
            unit_ball_volume(0)

        with pytest.raises(ValueError, match="positive"):
            unit_ball_volume(-1)

    def test_unit_ball_formula_consistency(self):
        """Test formula V_d = pi^(d/2) / Gamma(d/2 + 1)."""
        for d in [1, 2, 3, 5, 10]:
            expected = (np.pi ** (d / 2)) / gamma(d / 2 + 1)
            np.testing.assert_allclose(unit_ball_volume(d), expected, rtol=1e-6)


class TestPairwiseDistances:
    """Test pairwise distance computation."""

    def test_pairwise_distances_basic(self):
        """Test basic pairwise distance computation."""
        samples = np.array([[0, 0], [1, 0], [0, 1]])
        distances = pairwise_distances(samples)

        assert distances.shape == (3, 3)

        np.testing.assert_allclose(distances[0, 1], 1.0, rtol=1e-6)

        np.testing.assert_allclose(distances[0, 2], 1.0, rtol=1e-6)

        np.testing.assert_allclose(distances[1, 2], np.sqrt(2), rtol=1e-6)

    def test_pairwise_distances_symmetric(self):
        """Test that distance matrix is symmetric."""
        samples = np.random.randn(50, 5)
        distances = pairwise_distances(samples)

        np.testing.assert_allclose(distances, distances.T, rtol=1e-10)

    def test_pairwise_distances_diagonal_small(self):
        """Test that diagonal (self-distances) are near zero."""
        samples = np.random.randn(20, 3)
        distances = pairwise_distances(samples)

        np.testing.assert_allclose(np.diag(distances), 0.0, atol=1e-4)

    def test_pairwise_distances_triangle_inequality(self):
        """Test that distances satisfy triangle inequality."""
        samples = np.random.randn(10, 3)
        distances = pairwise_distances(samples)

        for i in range(10):
            for j in range(10):
                for k in range(10):
                    assert distances[i, j] <= distances[i, k] + distances[k, j] + 1e-8

    def test_pairwise_distances_euclidean(self):
        """Test Euclidean distance computation."""
        samples = np.array([[0, 0], [3, 4]])
        distances = pairwise_distances(samples, metric="euclidean")

        np.testing.assert_allclose(distances[0, 1], 5.0, rtol=1e-6)

    def test_pairwise_distances_chebyshev(self):
        """Test Chebyshev (max-norm) distance computation."""
        samples = np.array([[0, 0], [3, 4]])
        distances = pairwise_distances(samples, metric="chebyshev")

        np.testing.assert_allclose(distances[0, 1], 4.0, rtol=1e-6)

    def test_pairwise_distances_invalid_metric(self):
        """Test that invalid metric raises error."""
        samples = np.random.randn(10, 2)
        with pytest.raises(ValueError, match="metric"):
            pairwise_distances(samples, metric="invalid")

    def test_pairwise_distances_invalid_shape(self):
        """Test that non-2D input raises error."""
        with pytest.raises(ValueError, match="2D"):
            pairwise_distances(np.random.randn(10))

        with pytest.raises(ValueError, match="2D"):
            pairwise_distances(np.random.randn(2, 3, 4))

    def test_pairwise_distances_various_shapes(self):
        """Test with various input shapes."""
        for n, d in [(2, 1), (10, 2), (50, 5), (100, 10)]:
            samples = np.random.randn(n, d)
            distances = pairwise_distances(samples)
            assert distances.shape == (n, n)


class TestKthNearestDistances:
    """Test k-th nearest neighbor distance computation."""

    def test_kth_nearest_basic(self):
        """Test basic k-th nearest distance computation."""
        distances = np.array([
            [0, 1, 2, 3],
            [1, 0, 1.5, 2.5],
            [2, 1.5, 0, 1],
            [3, 2.5, 1, 0]
        ])

        kth = kth_nearest_distances(distances, k=1, exclude_self=True)
        np.testing.assert_allclose(kth, [1, 1, 1, 1], rtol=1e-6)

        kth = kth_nearest_distances(distances, k=2, exclude_self=True)
        np.testing.assert_allclose(kth, [2, 1.5, 1.5, 2.5], rtol=1e-6)

    def test_kth_nearest_sorted(self):
        """Test that k-th distances are correctly ordered."""
        n = 50
        samples = np.random.randn(n, 3)
        distances = pairwise_distances(samples)

        for k in [1, 2, 3, 5]:
            kth = kth_nearest_distances(distances, k=k, exclude_self=True)

            for i in range(n):
                row = distances[i].copy()
                row[i] = np.inf
                sorted_row = np.sort(row)
                np.testing.assert_allclose(kth[i], sorted_row[k - 1], rtol=1e-10)

    def test_kth_nearest_invalid_k(self):
        """Test that invalid k raises error."""
        distances = np.eye(10) * np.inf
        np.fill_diagonal(distances, 0)

        with pytest.raises(ValueError, match="k must be"):
            kth_nearest_distances(distances, k=0, exclude_self=True)

        with pytest.raises(ValueError, match="k must be"):
            kth_nearest_distances(distances, k=10, exclude_self=True)

    def test_kth_nearest_invalid_shape(self):
        """Test that non-square matrix raises error."""
        with pytest.raises(ValueError, match="square"):
            kth_nearest_distances(np.random.randn(10, 5), k=1)

    def test_kth_nearest_no_exclude_self_invalid_k(self):
        """Test that invalid k with exclude_self=False raises error."""
        distances = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])

        with pytest.raises(ValueError, match="k must be"):
            kth_nearest_distances(distances, k=0, exclude_self=False)

        with pytest.raises(ValueError, match="k must be"):
            kth_nearest_distances(distances, k=4, exclude_self=False)


class TestKNNEntropy:
    """Test k-NN entropy estimator."""

    def test_entropy_gaussian_standard(self):
        """Test entropy estimation for standard normal N(0, I)."""
        d = 2
        n = 2000
        samples = np.random.randn(n, d)

        H_estimated = knn_entropy(samples, k=5)
        H_true = 0.5 * d * np.log(2 * np.pi * np.e)

        np.testing.assert_allclose(H_estimated, H_true, rtol=0.1)

    def test_entropy_gaussian_various_dimensions(self):
        """Test entropy estimation for various dimensions."""
        estimates = []
        true_values = []

        for d in [2, 3, 5]:
            n = 1500
            samples = np.random.randn(n, d)
            H_estimated = knn_entropy(samples, k=5)
            H_true = 0.5 * d * np.log(2 * np.pi * np.e)
            estimates.append(H_estimated)
            true_values.append(H_true)

            np.testing.assert_allclose(H_estimated, H_true, rtol=0.15)

        for i in range(1, len(estimates)):
            assert estimates[i] > estimates[i - 1]

    def test_entropy_uniform_vs_clustered(self):
        """Test that uniform samples have higher entropy than clustered."""
        n = 500

        uniform_samples = np.random.rand(n, 2)
        clustered_samples = np.random.randn(n, 2) * 0.1

        H_uniform = knn_entropy(uniform_samples, k=5)
        H_clustered = knn_entropy(clustered_samples, k=5)

        assert H_uniform > H_clustered

    def test_entropy_scale_dependence(self):
        """Test that entropy scales correctly with data scaling."""
        n = 1000
        d = 3
        samples = np.random.randn(n, d)

        H_original = knn_entropy(samples, k=5)

        s = 2.0
        H_scaled = knn_entropy(samples * s, k=5)

        expected_diff = d * np.log(s)
        actual_diff = H_scaled - H_original

        np.testing.assert_allclose(actual_diff, expected_diff, rtol=0.1)

    def test_entropy_different_k(self):
        """Test entropy estimation with different k values."""
        samples = np.random.randn(500, 3)

        entropies = [knn_entropy(samples, k=k) for k in [1, 3, 5, 10]]

        assert np.std(entropies) < 0.5

    def test_entropy_reproducible(self):
        """Test that same samples give same entropy."""
        samples = np.random.randn(500, 3)

        H1 = knn_entropy(samples, k=5)
        H2 = knn_entropy(samples, k=5)

        np.testing.assert_equal(H1, H2)

    def test_entropy_invalid_samples(self):
        """Test that invalid samples raise errors."""
        with pytest.raises(ValueError, match="2D"):
            knn_entropy(np.random.randn(100))

        with pytest.raises(ValueError, match="at least 2"):
            knn_entropy(np.random.randn(1, 5), k=1)

    def test_entropy_invalid_k(self):
        """Test that invalid k raises error."""
        samples = np.random.randn(100, 5)

        with pytest.raises(ValueError, match="k must be"):
            knn_entropy(samples, k=0)

        with pytest.raises(ValueError, match="k must be"):
            knn_entropy(samples, k=100)

    def test_entropy_sample_size_effect(self):
        """Test that more samples give more stable estimates."""
        d = 3
        estimates_small = []
        estimates_large = []

        for _ in range(10):
            small = np.random.randn(100, d)
            large = np.random.randn(500, d)
            estimates_small.append(knn_entropy(small, k=5))
            estimates_large.append(knn_entropy(large, k=5))

        assert np.std(estimates_large) < np.std(estimates_small)


class TestKNNEntropyBatch:
    """Test batch entropy estimation."""

    def test_entropy_batch_basic(self):
        """Test basic batch entropy estimation."""
        batch = np.random.randn(10, 100, 5)
        entropies = knn_entropy_batch(batch, k=5)

        assert entropies.shape == (10,)
        assert np.all(np.isfinite(entropies))

    def test_entropy_batch_consistency(self):
        """Test that batch results match individual results."""
        batch = np.random.randn(5, 200, 3)

        batch_entropies = knn_entropy_batch(batch, k=5)
        individual_entropies = [knn_entropy(batch[i], k=5) for i in range(5)]

        np.testing.assert_allclose(batch_entropies, individual_entropies, rtol=1e-10)

    def test_entropy_batch_invalid_shape(self):
        """Test that invalid batch shape raises error."""
        with pytest.raises(ValueError, match="3D"):
            knn_entropy_batch(np.random.randn(100, 5))


class TestKNNEntropyEstimator:
    """Test class-based k-NN entropy estimator."""

    def test_estimator_basic(self):
        """Test basic estimator functionality."""
        estimator = KNNEntropyEstimator(k=5)
        samples = np.random.randn(500, 3)

        H = estimator.estimate(samples)

        assert isinstance(H, float)
        assert np.isfinite(H)

    def test_estimator_with_stats(self):
        """Test estimator with statistics."""
        estimator = KNNEntropyEstimator(k=5)
        samples = np.random.randn(500, 3)

        H, stats = estimator.estimate(samples, return_stats=True)

        assert isinstance(stats, KNNEntropyStats)
        assert stats.entropy == H
        assert stats.n_samples == 500
        assert stats.dimensionality == 3
        assert stats.k == 5
        assert stats.mean_kth_distance > 0
        assert stats.std_kth_distance >= 0
        assert stats.min_kth_distance > 0
        assert stats.max_kth_distance > 0

    def test_estimator_callable(self):
        """Test that estimator can be called as function."""
        estimator = KNNEntropyEstimator(k=5)
        samples = np.random.randn(500, 3)

        H1 = estimator(samples)
        H2 = estimator.estimate(samples)

        np.testing.assert_equal(H1, H2)

    def test_estimator_history(self):
        """Test that estimator tracks history."""
        estimator = KNNEntropyEstimator(k=5)

        for _ in range(5):
            samples = np.random.randn(200, 3)
            estimator.estimate(samples)

        history = estimator.get_history()
        assert len(history) == 5

    def test_estimator_reset(self):
        """Test history reset."""
        estimator = KNNEntropyEstimator(k=5)

        estimator.estimate(np.random.randn(100, 3))
        estimator.estimate(np.random.randn(100, 3))

        assert len(estimator.get_history()) == 2

        estimator.reset_history()

        assert len(estimator.get_history()) == 0
        assert estimator._n_estimations == 0
        assert estimator._total_samples == 0

    def test_estimator_from_config(self):
        """Test creating estimator from config."""
        config = KNNEntropyConfig(k=3, epsilon=1e-8, metric="euclidean")
        estimator = KNNEntropyEstimator.from_config(config)

        assert estimator.k == 3
        assert estimator.epsilon == 1e-8
        assert estimator.metric == "euclidean"

    def test_estimator_invalid_k(self):
        """Test that invalid k raises error."""
        with pytest.raises(ValueError, match="k must be"):
            KNNEntropyEstimator(k=0)

    def test_estimator_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError, match="metric"):
            KNNEntropyEstimator(k=5, metric="invalid")

    def test_estimator_chebyshev_metric(self):
        """Test estimator with Chebyshev metric."""
        estimator = KNNEntropyEstimator(k=5, metric="chebyshev")
        samples = np.random.randn(500, 3)

        H = estimator.estimate(samples)
        assert np.isfinite(H)

    def test_estimator_repr(self):
        """Test string representation."""
        estimator = KNNEntropyEstimator(k=5, epsilon=1e-10, metric="euclidean")
        repr_str = repr(estimator)

        assert "KNNEntropyEstimator" in repr_str
        assert "k=5" in repr_str
        assert "euclidean" in repr_str

    def test_estimator_invalid_samples_1d(self):
        """Test that 1D samples raise error."""
        estimator = KNNEntropyEstimator(k=5)
        with pytest.raises(ValueError, match="2D"):
            estimator.estimate(np.random.randn(100))

    def test_estimator_too_few_samples(self):
        """Test that too few samples raise error."""
        estimator = KNNEntropyEstimator(k=5)
        with pytest.raises(ValueError, match="at least 2"):
            estimator.estimate(np.random.randn(1, 3))

    def test_estimator_k_too_large(self):
        """Test that k >= n raises error."""
        estimator = KNNEntropyEstimator(k=50)
        with pytest.raises(ValueError, match="k must be"):
            estimator.estimate(np.random.randn(30, 3))


class TestKNNEntropyStats:
    """Test entropy statistics class."""

    def test_stats_to_dict(self):
        """Test converting stats to dictionary."""
        stats = KNNEntropyStats(
            entropy=5.0,
            n_samples=100,
            dimensionality=3,
            k=5,
            mean_kth_distance=0.5,
            std_kth_distance=0.1,
            min_kth_distance=0.1,
            max_kth_distance=1.0,
        )

        d = stats.to_dict()

        assert d["entropy"] == 5.0
        assert d["n_samples"] == 100
        assert d["dimensionality"] == 3
        assert d["k"] == 5
        assert d["mean_kth_distance"] == 0.5
        assert d["std_kth_distance"] == 0.1
        assert d["min_kth_distance"] == 0.1
        assert d["max_kth_distance"] == 1.0


class TestKNNEntropyConfig:
    """Test entropy configuration class."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = KNNEntropyConfig()

        assert config.k == 5
        assert config.epsilon == 1e-10
        assert config.metric == "euclidean"
        assert config.normalize_by_dim is False

    def test_config_custom(self):
        """Test custom configuration."""
        config = KNNEntropyConfig(k=3, epsilon=1e-8, metric="chebyshev", normalize_by_dim=True)

        assert config.k == 3
        assert config.epsilon == 1e-8
        assert config.metric == "chebyshev"
        assert config.normalize_by_dim is True


class TestEntropyGaussian:
    """Test analytical Gaussian entropy computation."""

    def test_entropy_gaussian_standard(self):
        """Test entropy of standard normal."""
        for d in [1, 2, 5, 10]:
            H = entropy_gaussian(d)
            expected = 0.5 * d * np.log(2 * np.pi * np.e)
            np.testing.assert_allclose(H, expected, rtol=1e-10)

    def test_entropy_gaussian_scaled(self):
        """Test entropy with scaled covariance."""
        d = 3
        sigma = 2.0
        cov = sigma ** 2 * np.eye(d)

        H = entropy_gaussian(d, cov=cov)
        expected = 0.5 * d * np.log(2 * np.pi * np.e * sigma ** 2)

        np.testing.assert_allclose(H, expected, rtol=1e-6)

    def test_entropy_gaussian_invalid_cov(self):
        """Test that invalid covariance raises error."""
        with pytest.raises(ValueError, match="cov must be"):
            entropy_gaussian(3, cov=np.eye(2))

    def test_entropy_gaussian_non_positive_definite(self):
        """Test that non-positive-definite covariance raises error."""
        singular_cov = np.array([[1, 0], [0, 0]])
        with pytest.raises(ValueError, match="positive definite"):
            entropy_gaussian(2, cov=singular_cov)


class TestEntropyUniform:
    """Test analytical uniform entropy computation."""

    def test_entropy_uniform_unit(self):
        """Test entropy of uniform on [0, 1]."""
        for d in [1, 2, 5]:
            H = entropy_uniform(d, low=0.0, high=1.0)
            expected = d * np.log(1.0)
            np.testing.assert_allclose(H, expected, atol=1e-10)

    def test_entropy_uniform_symmetric(self):
        """Test entropy of uniform on [-1, 1]."""
        for d in [1, 2, 5]:
            H = entropy_uniform(d, low=-1.0, high=1.0)
            expected = d * np.log(2.0)
            np.testing.assert_allclose(H, expected, rtol=1e-10)

    def test_entropy_uniform_invalid_bounds(self):
        """Test that invalid bounds raise error."""
        with pytest.raises(ValueError, match="high must be"):
            entropy_uniform(2, low=1.0, high=0.0)


class TestMutualInformationKNN:
    """Test mutual information estimation."""

    def test_mi_independent(self):
        """Test that MI is near zero for independent variables."""
        n = 500
        x = np.random.randn(n, 2)
        y = np.random.randn(n, 2)

        mi = mutual_information_knn(x, y, k=5)

        np.testing.assert_allclose(mi, 0.0, atol=0.5)

    def test_mi_dependent(self):
        """Test that MI is positive for dependent variables."""
        n = 500
        x = np.random.randn(n, 2)
        y = x + np.random.randn(n, 2) * 0.5

        mi = mutual_information_knn(x, y, k=5)

        assert mi > 0.1

    def test_mi_stronger_dependence(self):
        """Test that stronger dependence gives higher MI."""
        n = 500
        x = np.random.randn(n, 2)
        y_weak = x + np.random.randn(n, 2) * 1.0
        y_strong = x + np.random.randn(n, 2) * 0.1

        mi_weak = mutual_information_knn(x, y_weak, k=5)
        mi_strong = mutual_information_knn(x, y_strong, k=5)

        assert mi_strong > mi_weak

    def test_mi_mismatched_samples(self):
        """Test that mismatched sample sizes raise error."""
        x = np.random.randn(100, 2)
        y = np.random.randn(50, 2)

        with pytest.raises(ValueError, match="same number"):
            mutual_information_knn(x, y)


class TestNumericalStability:
    """Test numerical stability of entropy estimation."""

    def test_stability_small_samples(self):
        """Test stability with small number of samples."""
        for n in [5, 10, 20, 50]:
            samples = np.random.randn(n, 3)
            H = knn_entropy(samples, k=min(3, n - 1))
            assert np.isfinite(H)

    def test_stability_high_dimensions(self):
        """Test stability with high-dimensional data."""
        for d in [10, 20, 50]:
            samples = np.random.randn(500, d)
            H = knn_entropy(samples, k=5)
            assert np.isfinite(H)

    def test_stability_very_clustered(self):
        """Test stability with very clustered data."""
        samples = np.random.randn(100, 3) * 1e-5
        H = knn_entropy(samples, k=5)
        assert np.isfinite(H)

    def test_stability_very_spread(self):
        """Test stability with very spread out data."""
        samples = np.random.randn(100, 3) * 1e5
        H = knn_entropy(samples, k=5)
        assert np.isfinite(H)

    def test_stability_mixed_scales(self):
        """Test stability with features at different scales."""
        samples = np.random.randn(500, 3)
        samples[:, 0] *= 1e3
        samples[:, 2] *= 1e-3
        H = knn_entropy(samples, k=5)
        assert np.isfinite(H)

    def test_no_nan_in_distances(self):
        """Test that distance computation doesn't produce NaN."""
        for _ in range(10):
            samples = np.random.randn(100, 5)
            distances = pairwise_distances(samples)
            assert np.all(np.isfinite(distances))

    def test_no_zero_kth_distances(self):
        """Test that k-th distances are always positive."""
        for _ in range(10):
            samples = np.random.randn(100, 5)
            distances = pairwise_distances(samples)
            np.fill_diagonal(distances, np.inf)
            kth = kth_nearest_distances(distances, k=5, exclude_self=True)
            assert np.all(kth > 0)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_samples(self):
        """Test with minimum number of samples (2)."""
        samples = np.random.randn(2, 3)
        H = knn_entropy(samples, k=1)
        assert np.isfinite(H)

    def test_single_dimension(self):
        """Test with 1D samples."""
        samples = np.random.randn(500, 1)
        H = knn_entropy(samples, k=5)
        assert np.isfinite(H)

    def test_k_equals_1(self):
        """Test with k=1 (nearest neighbor only)."""
        samples = np.random.randn(100, 3)
        H = knn_entropy(samples, k=1)
        assert np.isfinite(H)

    def test_k_equals_n_minus_1(self):
        """Test with k = n-1 (all neighbors)."""
        n = 20
        samples = np.random.randn(n, 3)
        H = knn_entropy(samples, k=n - 1)
        assert np.isfinite(H)

    def test_identical_samples_different_batches(self):
        """Test that identical samples in different batches give same entropy."""
        samples = np.random.randn(200, 3)

        H1 = knn_entropy(samples, k=5)
        H2 = knn_entropy(samples.copy(), k=5)

        np.testing.assert_equal(H1, H2)

    def test_permuted_samples(self):
        """Test that permutation doesn't change entropy."""
        samples = np.random.randn(200, 3)

        H1 = knn_entropy(samples, k=5)

        perm = np.random.permutation(200)
        H2 = knn_entropy(samples[perm], k=5)

        np.testing.assert_allclose(H1, H2, rtol=1e-10)

    def test_translated_samples(self):
        """Test that translation doesn't change entropy."""
        samples = np.random.randn(200, 3)

        H1 = knn_entropy(samples, k=5)
        H2 = knn_entropy(samples + 1000, k=5)

        np.testing.assert_allclose(H1, H2, rtol=1e-10)


class TestReproducibility:
    """Test reproducibility with random seeds."""

    def test_reproducible_with_seed(self):
        """Test that results are reproducible with fixed seed."""
        np.random.seed(12345)
        samples1 = np.random.randn(500, 3)
        H1 = knn_entropy(samples1, k=5)

        np.random.seed(12345)
        samples2 = np.random.randn(500, 3)
        H2 = knn_entropy(samples2, k=5)

        np.testing.assert_equal(H1, H2)

    def test_different_seeds_different_samples(self):
        """Test that different seeds produce different samples."""
        np.random.seed(12345)
        H1 = knn_entropy(np.random.randn(500, 3), k=5)

        np.random.seed(54321)
        H2 = knn_entropy(np.random.randn(500, 3), k=5)

        assert H1 != H2


class TestScalability:
    """Test scalability with different sizes."""

    @pytest.mark.parametrize("n", [10, 50, 100, 500, 1000])
    def test_various_sample_sizes(self, n):
        """Test with various sample sizes."""
        samples = np.random.randn(n, 3)
        k = min(5, n - 1)
        H = knn_entropy(samples, k=k)
        assert np.isfinite(H)

    @pytest.mark.parametrize("d", [1, 2, 5, 10, 20])
    def test_various_dimensions(self, d):
        """Test with various dimensions."""
        samples = np.random.randn(500, d)
        H = knn_entropy(samples, k=5)
        assert np.isfinite(H)

    @pytest.mark.parametrize("k", [1, 2, 3, 5, 10, 20])
    def test_various_k(self, k):
        """Test with various k values."""
        samples = np.random.randn(500, 3)
        if k < 500:
            H = knn_entropy(samples, k=k)
            assert np.isfinite(H)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self):
        """Test complete workflow: config -> estimator -> estimate -> stats."""
        config = KNNEntropyConfig(k=5, epsilon=1e-10, metric="euclidean")
        estimator = KNNEntropyEstimator.from_config(config)

        d = 2
        n = 1000
        samples = np.random.randn(n, d)
        H, stats = estimator.estimate(samples, return_stats=True)

        assert np.isfinite(H)
        assert stats.entropy == H
        assert stats.n_samples == n
        assert stats.dimensionality == d
        assert stats.k == 5

        H_true = 0.5 * d * np.log(2 * np.pi * np.e)
        np.testing.assert_allclose(H, H_true, rtol=0.1)

    def test_multiple_estimations(self):
        """Test multiple estimations with history tracking."""
        estimator = KNNEntropyEstimator(k=5)
        d = 2
        H_true = 0.5 * d * np.log(2 * np.pi * np.e)

        entropies = []
        for _ in range(10):
            samples = np.random.randn(500, d)
            H = estimator.estimate(samples)
            entropies.append(H)

        history = estimator.get_history()
        np.testing.assert_allclose(history, entropies, rtol=1e-10)

        np.testing.assert_allclose(np.mean(entropies), H_true, rtol=0.1)

    def test_compare_distributions(self):
        """Test comparing entropy of different distributions."""
        n = 1000

        H_normal = knn_entropy(np.random.randn(n, 2), k=5)

        H_narrow = knn_entropy(np.random.randn(n, 2) * 0.5, k=5)

        H_wide = knn_entropy(np.random.randn(n, 2) * 2.0, k=5)

        assert H_narrow < H_normal < H_wide

    def test_entropy_regularization_simulation(self):
        """Simulate using entropy as regularizer in training."""
        entropies = []

        for i in range(5):
            spread = 0.5 + 0.1 * i
            samples = np.random.randn(200, 2) * spread
            H = knn_entropy(samples, k=5)
            entropies.append(H)

        for i in range(1, len(entropies)):
            assert entropies[i] > entropies[i - 1]
