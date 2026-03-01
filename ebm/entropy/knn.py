"""
k-NN Entropy Estimator (Kozachenko-Leonenko Estimator).

Estimates differential entropy from samples using k-nearest neighbor distances.
This is a non-parametric estimator that doesn't assume any specific distribution.

The Kozachenko-Leonenko estimator formula is:

    H = psi(n) - psi(k) + log(V_d) + (d/n) * sum_{i=1}^{n} log(epsilon_k(i))

Where:
    - n = number of samples
    - d = dimensionality
    - k = number of neighbors
    - psi = digamma function
    - epsilon_k(i) = distance from sample i to its k-th nearest neighbor
    - V_d = pi^(d/2) / Gamma(d/2 + 1) = volume of d-dimensional unit ball

References:
    - Kozachenko & Leonenko (1987) - Original k-NN entropy estimator
    - Kraskov et al. (2004) - Improved estimator with bias correction
    - Singh et al. (2003) - Analysis and improvements

Usage:
    >>> samples = np.random.randn(1000, 5)  # 1000 samples in 5D
    >>> H = knn_entropy(samples, k=5)
    >>> print(f"Entropy estimate: {H:.4f}")

Note:
    For EBM training, entropy is used as a regularizer to encourage sample
    diversity. Start with stopped gradients (don't backprop through entropy).
    The entropy term is a regularizer; exact gradients aren't critical.
"""
import numpy as np
from dataclasses import dataclass

from typing import Dict, Optional, Tuple, Union


def digamma(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the digamma function psi(x) = d/dx ln(Gamma(x)).

    Uses a recursive formula for small x and asymptotic expansion for large x.
    This avoids requiring scipy.special.digamma.

    Args:
        x: Input value(s). Must be positive.

    Returns:
        Digamma function value(s)

    Raises:
        ValueError: If x <= 0

    Example:
        >>> digamma(1.0)  # Returns -gamma (Euler-Mascheroni constant)
        -0.5772...
        >>> digamma(5.0)
        1.5061...

    Note:
        The asymptotic expansion is accurate to ~1e-10 for x >= 6.
        For smaller x, we use the recurrence psi(x) = psi(x+1) - 1/x.
    """
    x = np.asarray(x, dtype=np.float64)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    if np.any(x <= 0):
        raise ValueError(f"digamma requires positive input, got min={x.min()}")

    result = np.zeros_like(x, dtype=np.float64)

    for idx in np.ndindex(x.shape):
        xi = x[idx]

        shift = 0.0

        while xi < 6.0:
            shift -= 1.0 / xi
            xi += 1.0

        inv_x = 1.0 / xi
        inv_x2 = inv_x * inv_x

        result[idx] = (
            np.log(xi)
            - 0.5 * inv_x
            - inv_x2 / 12.0
            + (inv_x2 * inv_x2) / 120.0
            - (inv_x2 * inv_x2 * inv_x2) / 252.0
            + shift
        )

    return float(result[0]) if scalar_input else result


def log_gamma(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the natural logarithm of the gamma function: log(Gamma(x)).

    Uses Stirling's approximation with correction terms for accuracy.

    Args:
        x: Input value(s). Must be positive.

    Returns:
        Log-gamma function value(s)

    Raises:
        ValueError: If x <= 0

    Example:
        >>> np.exp(log_gamma(5))  # Gamma(5) = 4! = 24
        24.0
        >>> log_gamma(0.5)  # log(sqrt(pi))
        0.5723...

    Note:
        For integer n, Gamma(n) = (n-1)!
        For half-integers: Gamma(n+0.5) = sqrt(pi) * (2n)! / (4^n * n!)
    """
    x = np.asarray(x, dtype=np.float64)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)
    if np.any(x <= 0):
        raise ValueError(f"log_gamma requires positive input, got min={x.min()}")

    result = np.zeros_like(x, dtype=np.float64)

    for idx in np.ndindex(x.shape):
        xi = x[idx]

        shift = 0.0
        while xi < 7.0:
            shift -= np.log(xi)
            xi += 1.0

        inv_x = 1.0 / xi
        inv_x2 = inv_x * inv_x

        result[idx] = (
            (xi - 0.5) * np.log(xi)
            - xi
            + 0.5 * np.log(2.0 * np.pi)
            + inv_x / 12.0
            - (inv_x * inv_x2) / 360.0
            + (inv_x * inv_x2 * inv_x2) / 1260.0
            - (inv_x * inv_x2 * inv_x2 * inv_x2) / 1680.0
            + shift
        )

    return float(result[0]) if scalar_input else result


def gamma(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the gamma function Gamma(x).

    Args:
        x: Input value(s). Must be positive.

    Returns:
        Gamma function value(s)

    Example:
        >>> gamma(5)  # 4! = 24
        24.0
        >>> gamma(0.5)  # sqrt(pi)
        1.7724...
    """
    return np.exp(log_gamma(x))


def unit_ball_volume(d: int) -> float:
    """
    Compute the volume of the d-dimensional unit ball.

    The volume is V_d = pi^(d/2) / Gamma(d/2 + 1).

    Args:
        d: Dimensionality (must be positive integer)

    Returns:
        Volume of the d-dimensional unit ball

    Raises:
        ValueError: If d <= 0

    Example:
        >>> unit_ball_volume(1)  # Length of interval [-1, 1]
        2.0
        >>> unit_ball_volume(2)  # Area of unit circle: pi
        3.1415...
        >>> unit_ball_volume(3)  # Volume of unit sphere: 4/3 * pi
        4.1887...

    Note:
        This formula uses the gamma function which is more numerically
        stable than computing factorials directly for large d.
    """
    if d <= 0:
        raise ValueError(f"Dimensionality must be positive, got {d}")

    half_d = d / 2.0

    log_volume = half_d * np.log(np.pi) - log_gamma(half_d + 1.0)

    return float(np.exp(log_volume))


def pairwise_distances(
    samples: np.ndarray,
    metric: str = "euclidean",
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Compute pairwise distances between all samples.

    Args:
        samples: Array of shape (n, d) containing n d-dimensional samples
        metric: Distance metric - 'euclidean' or 'chebyshev' (max-norm)
        epsilon: Small constant for numerical stability in sqrt

    Returns:
        Distance matrix of shape (n, n) with distances[i, j] = ||samples[i] - samples[j]||

    Raises:
        ValueError: If samples is not 2D or metric is invalid

    Example:
        >>> samples = np.array([[0, 0], [1, 0], [0, 1]])
        >>> distances = pairwise_distances(samples)
        >>> distances[0, 1]  # Distance from [0,0] to [1,0]
        1.0
    """
    samples = np.asarray(samples, dtype=np.float64)

    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D, got {samples.ndim}D")

    if metric not in ("euclidean", "chebyshev"):
        raise ValueError(f"metric must be 'euclidean' or 'chebyshev', got '{metric}'")

    diff = samples[:, np.newaxis, :] - samples[np.newaxis, :, :]

    if metric == "euclidean":
        sq_distances = (diff ** 2).sum(axis=-1)
        distances = np.sqrt(sq_distances + epsilon)
    else:
        distances = np.abs(diff).max(axis=-1)
    return distances


def kth_nearest_distances(
    distances: np.ndarray,
    k: int,
    exclude_self: bool = True,
) -> np.ndarray:
    """
    Find the k-th nearest neighbor distance for each sample.

    Args:
        distances: Pairwise distance matrix of shape (n, n)
        k: Number of nearest neighbors (k-th neighbor distance returned)
        exclude_self: If True, exclude self-distances (diagonal)

    Returns:
        Array of shape (n,) with k-th nearest distances for each sample

    Raises:
        ValueError: If k is invalid for the given distance matrix

    Example:
        >>> distances = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
        >>> kth_nearest_distances(distances, k=1, exclude_self=True)
        array([1., 1., 1.5])
    """
    distances = np.asarray(distances, dtype=np.float64)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError(f"distances must be square 2D array, got shape {distances.shape}")

    n = distances.shape[0]

    if exclude_self:
        if k < 1 or k >= n:
            raise ValueError(
                f"k must be in [1, n-1] when excluding self, got k={k}, n={n}"
            )
    else:
        if k < 1 or k > n:
            raise ValueError(f"k must be in [1, n] got k={k}, n={n}")

    dist_copy = distances.copy()

    if exclude_self:
        np.fill_diagonal(dist_copy, np.inf)

    kth_idx = k - 1 if not exclude_self else k - 1

    kth_distances = np.partition(dist_copy, kth_idx, axis=1)[:, kth_idx]

    return kth_distances


def knn_entropy(
    samples: np.ndarray,
    k: int = 5,
    epsilon: float = 1e-10,
) -> float:
    """
    Estimate differential entropy using k-NN (Kozachenko-Leonenko estimator).

    The estimator formula is:
        H = psi(n) - psi(k) + log(V_d) + (d/n) * sum_{i=1}^{n} log(epsilon_k(i))

    Where:
        - n = number of samples
        - d = dimensionality
        - k = number of neighbors
        - psi = digamma function
        - epsilon_k(i) = distance from sample i to its k-th nearest neighbor
        - V_d = volume of d-dimensional unit ball

    Args:
        samples: Array of shape (n, d) containing n d-dimensional samples
        k: Number of nearest neighbors (default: 5)
        epsilon: Small constant for numerical stability in log

    Returns:
        Estimated differential entropy (scalar)

    Raises:
        ValueError: If samples has wrong shape or k is invalid

    Example:
        >>> # Entropy of N(0, I) in d dimensions is 0.5 * d * log(2*pi*e)
        >>> d = 5
        >>> samples = np.random.randn(5000, d)
        >>> H_estimated = knn_entropy(samples, k=5)
        >>> H_true = 0.5 * d * np.log(2 * np.pi * np.e)
        >>> print(f"Estimated: {H_estimated:.4f}, True: {H_true:.4f}")

    Note:
        - Higher k = lower variance but higher bias
        - k=5 is a good default for most cases
        - For small samples, use smaller k (k=3)
        - Entropy can be negative for continuous distributions
    """
    samples = np.asarray(samples, dtype=np.float64)

    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D, got {samples.ndim}D")

    n, d = samples.shape

    if n < 2:
        raise ValueError(f"Need at least 2 samples, got {n}")

    if k < 1 or k >= n:
        raise ValueError(f"k must be in [1, n-1], got k={k}, n={n}")

    distances = pairwise_distances(samples, metric="euclidean", epsilon=epsilon)

    np.fill_diagonal(distances, np.inf)

    kth_distances = np.partition(distances, k - 1, axis=1)[:, k - 1]

    V_d = unit_ball_volume(d)

    H = (
        digamma(n)
        - digamma(k)
        + np.log(V_d)
        + (d / n) * np.sum(np.log(kth_distances + epsilon))
    )

    return float(H)


def knn_entropy_batch(
    samples_batch: np.ndarray,
    k: int = 5,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Estimate entropy for a batch of sample sets.

    Useful when computing entropy for multiple groups of samples independently.

    Args:
        samples_batch: Array of shape (batch_size, n, d) containing
                       batch_size sets of n d-dimensional samples
        k: Number of nearest neighbors (default: 5)
        epsilon: Small constant for numerical stability

    Returns:
        Array of shape (batch_size,) with entropy estimates

    Example:
        >>> batch = np.random.randn(10, 100, 5)  # 10 sets of 100 5D samples
        >>> entropies = knn_entropy_batch(batch, k=5)
        >>> print(entropies.shape)
        (10,)
    """
    samples_batch = np.asarray(samples_batch, dtype=np.float64)

    if samples_batch.ndim != 3:
        raise ValueError(f"samples_batch must be 3D, got {samples_batch.ndim}D")

    batch_size = samples_batch.shape[0]

    entropies = np.zeros(batch_size)

    for i in range(batch_size):
        entropies[i] = knn_entropy(samples_batch[i], k=k, epsilon=epsilon)

    return entropies


@dataclass
class KNNEntropyConfig:
    """
    Configuration for k-NN entropy estimator.

    Attributes:
        k: Number of nearest neighbors (default: 5)
        epsilon: Small constant for numerical stability (default: 1e-10)
        metric: Distance metric - 'euclidean' or 'chebyshev' (default: 'euclidean')
        normalize_by_dim: If True, return entropy per dimension (default: False)
    """
    k: int = 5
    epsilon: float = 1e-10

    metric: str = "euclidean"
    normalize_by_dim: bool = False


@dataclass
class KNNEntropyStats:
    """
    Statistics from entropy estimation.

    Attributes:
        entropy: Estimated entropy
        n_samples: Number of samples used
        dimensionality: Sample dimensionality
        k: Number of neighbors used
        mean_kth_distance: Mean of k-th nearest neighbor distances
        std_kth_distance: Std of k-th nearest neighbor distances
        min_kth_distance: Minimum k-th nearest neighbor distance
        max_kth_distance: Maximum k-th nearest neighbor distance
    """
    entropy: float = 0.0
    n_samples: int = 0
    dimensionality: int = 0

    k: int = 0
    mean_kth_distance: float = 0.0
    std_kth_distance: float = 0.0
    min_kth_distance: float = 0.0
    max_kth_distance: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert stats to dictionary."""
        return {
            "entropy": self.entropy,
            "n_samples": self.n_samples,
            "dimensionality": self.dimensionality,
            "k": self.k,
            "mean_kth_distance": self.mean_kth_distance,
            "std_kth_distance": self.std_kth_distance,
            "min_kth_distance": self.min_kth_distance,
            "max_kth_distance": self.max_kth_distance,
        }


class KNNEntropyEstimator:
    """
    Class-based k-NN entropy estimator with configuration and statistics.

    Provides a stateful interface for entropy estimation with configurable
    parameters and detailed statistics tracking.

    Example:
        >>> estimator = KNNEntropyEstimator(k=5)
        >>> samples = np.random.randn(1000, 5)
        >>> entropy, stats = estimator.estimate(samples, return_stats=True)
        >>> print(f"Entropy: {entropy:.4f}")
        >>> print(f"Mean k-th distance: {stats.mean_kth_distance:.4f}")

    Note:
        For most use cases, the functional interface knn_entropy() is simpler.
        Use this class when you need statistics or custom configuration.
    """

    def __init__(
        self,
        k: int = 5,
        epsilon: float = 1e-10,
        metric: str = "euclidean",
    ):
        """
        Initialize the entropy estimator.

        Args:
            k: Number of nearest neighbors (default: 5)
            epsilon: Small constant for numerical stability (default: 1e-10)
            metric: Distance metric - 'euclidean' or 'chebyshev' (default: 'euclidean')

        Raises:
            ValueError: If k < 1 or metric is invalid
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        if metric not in ("euclidean", "chebyshev"):
            raise ValueError(f"metric must be 'euclidean' or 'chebyshev', got '{metric}'")

        self.k = k
        self.epsilon = epsilon
        self.metric = metric

        self._n_estimations = 0
        self._total_samples = 0
        self._entropy_history: list = []

    @classmethod
    def from_config(cls, config: KNNEntropyConfig) -> "KNNEntropyEstimator":
        """
        Create estimator from configuration.

        Args:
            config: KNNEntropyConfig with estimator parameters

        Returns:
            New KNNEntropyEstimator instance
        """
        return cls(
            k=config.k,
            epsilon=config.epsilon,
            metric=config.metric,
        )

    def estimate(
        self,
        samples: np.ndarray,
        return_stats: bool = False,
    ) -> Union[float, Tuple[float, KNNEntropyStats]]:
        """
        Estimate entropy of samples.

        Args:
            samples: Array of shape (n, d) containing samples
            return_stats: If True, also return detailed statistics

        Returns:
            If return_stats is False: entropy value (float)
            If return_stats is True: tuple of (entropy, KNNEntropyStats)

        Raises:
            ValueError: If samples has wrong shape or k >= n
        """
        samples = np.asarray(samples, dtype=np.float64)

        if samples.ndim != 2:
            raise ValueError(f"samples must be 2D, got {samples.ndim}D")

        n, d = samples.shape

        if n < 2:
            raise ValueError(f"Need at least 2 samples, got {n}")

        if self.k >= n:
            raise ValueError(f"k must be < n, got k={self.k}, n={n}")

        distances = pairwise_distances(samples, metric=self.metric, epsilon=self.epsilon)

        np.fill_diagonal(distances, np.inf)

        kth_distances = np.partition(distances, self.k - 1, axis=1)[:, self.k - 1]

        V_d = unit_ball_volume(d)

        entropy = (
            digamma(n)
            - digamma(self.k)
            + np.log(V_d)
            + (d / n) * np.sum(np.log(kth_distances + self.epsilon))
        )

        self._n_estimations += 1
        self._total_samples += n
        self._entropy_history.append(float(entropy))

        if return_stats:
            stats = KNNEntropyStats(
                entropy=float(entropy),
                n_samples=n,
                dimensionality=d,
                k=self.k,
                mean_kth_distance=float(kth_distances.mean()),
                std_kth_distance=float(kth_distances.std()),
                min_kth_distance=float(kth_distances.min()),
                max_kth_distance=float(kth_distances.max()),
            )
            return float(entropy), stats

        return float(entropy)

    def get_history(self) -> np.ndarray:
        """Get history of entropy estimates."""
        return np.array(self._entropy_history)

    def reset_history(self) -> None:
        """Reset estimation history and statistics."""
        self._n_estimations = 0
        self._total_samples = 0
        self._entropy_history = []

    def __call__(
        self,
        samples: np.ndarray,
        return_stats: bool = False,
    ) -> Union[float, Tuple[float, KNNEntropyStats]]:
        """Alias for estimate() - allows using estimator as a function."""
        return self.estimate(samples, return_stats=return_stats)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"KNNEntropyEstimator(k={self.k}, epsilon={self.epsilon}, "
            f"metric='{self.metric}')"
        )


def entropy_gaussian(d: int, cov: Optional[np.ndarray] = None) -> float:
    """
    Compute the true differential entropy of a Gaussian distribution.

    For N(mu, Sigma), entropy is:
        H = 0.5 * d * log(2*pi*e) + 0.5 * log(det(Sigma))

    For standard normal N(0, I):
        H = 0.5 * d * log(2*pi*e)

    Args:
        d: Dimensionality
        cov: Covariance matrix (default: identity)

    Returns:
        True entropy value

    Example:
        >>> entropy_gaussian(5)  # Entropy of 5D standard normal
        7.267...
    """
    if cov is None:
        log_det = 0.0
    else:
        cov = np.asarray(cov)

        if cov.shape != (d, d):
            raise ValueError(f"cov must be (d, d), got {cov.shape}")

        sign, log_det = np.linalg.slogdet(cov)

        if sign <= 0:
            raise ValueError("Covariance matrix must be positive definite")

    return 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * log_det


def entropy_uniform(d: int, low: float = 0.0, high: float = 1.0) -> float:
    """
    Compute the true differential entropy of a uniform distribution.

    For Uniform(low, high) in d dimensions:
        H = d * log(high - low)

    Args:
        d: Dimensionality
        low: Lower bound (same for all dimensions)
        high: Upper bound (same for all dimensions)

    Returns:
        True entropy value

    Example:
        >>> entropy_uniform(2, 0, 1)  # Entropy of 2D uniform on [0,1]^2
        0.0  # = 2 * log(1)
        >>> entropy_uniform(2, -1, 1)  # 2D uniform on [-1,1]^2
        1.386...  # = 2 * log(2)
    """
    if high <= low:
        raise ValueError(f"high must be > low, got low={low}, high={high}")

    return d * np.log(high - low)


def mutual_information_knn(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5,
) -> float:
    """
    Estimate mutual information I(X; Y) using k-NN entropy estimates.

    Uses the identity: I(X; Y) = H(X) + H(Y) - H(X, Y)

    Args:
        x: Array of shape (n, d_x)
        y: Array of shape (n, d_y)
        k: Number of nearest neighbors

    Returns:
        Estimated mutual information

    Example:
        >>> x = np.random.randn(1000, 2)
        >>> y = x + np.random.randn(1000, 2) * 0.1  # y depends on x
        >>> mi = mutual_information_knn(x, y, k=5)
        >>> print(f"MI: {mi:.4f}")  # Should be positive

    Note:
        This is a simple estimate. For more accurate MI estimation,
        consider the KSG estimator (Kraskov-Stogbauer-Grassberger).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must have same number of samples, "
                         f"got {x.shape[0]} and {y.shape[0]}")

    xy = np.concatenate([x, y], axis=1)

    H_x = knn_entropy(x, k=k)

    H_y = knn_entropy(y, k=k)

    H_xy = knn_entropy(xy, k=k)

    return H_x + H_y - H_xy
