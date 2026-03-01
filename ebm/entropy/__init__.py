"""
k-NN Entropy Estimation module.

Provides the Kozachenko-Leonenko estimator for differential entropy
estimation from samples. This is used as a regularizer in EBM training
to encourage sample diversity.
"""
from ebm.entropy.knn import (
    knn_entropy,
    knn_entropy_batch,
    digamma,
    log_gamma,
    unit_ball_volume,
    pairwise_distances,
    kth_nearest_distances,
    KNNEntropyEstimator,
    KNNEntropyConfig,
    KNNEntropyStats,
)

__all__ = [
    "knn_entropy",
    "knn_entropy_batch",
    "digamma",
    "log_gamma",
    "unit_ball_volume",
    "pairwise_distances",
    "kth_nearest_distances",
    "KNNEntropyEstimator",
    "KNNEntropyConfig",
    "KNNEntropyStats",
]
