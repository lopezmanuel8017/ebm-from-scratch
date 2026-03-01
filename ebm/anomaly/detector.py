"""
Anomaly detector using Energy-Based Models.

This module provides the EBMAnomalyDetector class that uses trained
energy functions to detect anomalies. The key insight is that:

- Normal samples should have LOW energy (high probability)
- Anomaly samples should have HIGH energy (low probability)

The detector provides methods for:
- Scoring samples (computing energy values)
- Fitting a threshold based on validation data
- Predicting anomaly labels
"""

import numpy as np
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass

from ebm.core.autodiff import Tensor
from ebm.core.energy import EnergyMLP


@dataclass
class DetectionResult:
    """
    Container for anomaly detection results.

    Attributes:
        scores: Anomaly scores (energy values) for each sample
        predictions: Binary predictions (1 = anomaly, 0 = normal)
        threshold: Threshold used for predictions
    """
    scores: np.ndarray
    predictions: np.ndarray
    threshold: float

    def __repr__(self) -> str:
        n_anomalies = self.predictions.sum()
        return (
            f"DetectionResult(\n"
            f"  n_samples: {len(self.scores)},\n"
            f"  n_anomalies_detected: {n_anomalies},\n"
            f"  anomaly_rate: {n_anomalies / len(self.scores):.4f},\n"
            f"  threshold: {self.threshold:.4f}\n"
            f")"
        )


@dataclass
class ThresholdInfo:
    """
    Information about the fitted threshold.

    Attributes:
        threshold: The fitted threshold value
        percentile: The percentile used for fitting
        n_samples: Number of samples used for fitting
        n_normal: Number of normal samples used
        normal_score_mean: Mean score of normal samples
        normal_score_std: Standard deviation of normal scores
        normal_score_min: Minimum normal score
        normal_score_max: Maximum normal score
    """
    threshold: float
    percentile: float
    n_samples: int
    n_normal: int
    normal_score_mean: float
    normal_score_std: float
    normal_score_min: float
    normal_score_max: float

    def __repr__(self) -> str:
        return (
            f"ThresholdInfo(\n"
            f"  threshold: {self.threshold:.4f},\n"
            f"  percentile: {self.percentile},\n"
            f"  normal_score_range: [{self.normal_score_min:.4f}, {self.normal_score_max:.4f}],\n"
            f"  normal_score_mean: {self.normal_score_mean:.4f} +/- {self.normal_score_std:.4f}\n"
            f")"
        )


# Define the main anomaly detector class
class EBMAnomalyDetector:
    """
    Anomaly detector using Energy-Based Models.

    Uses an energy function trained on normal data to detect anomalies.
    Samples with energy above a threshold are classified as anomalies.

    The intuition:
    - Energy function is trained to assign low energy to normal data
    - Anomalies (not seen during training) will have higher energy
    - Threshold separates normal from anomalous

    Attributes:
        energy_fn: Trained energy function (EnergyMLP)
        threshold: Decision threshold (None until fit_threshold is called)
        threshold_info: Detailed information about the fitted threshold

    Example:
        >>> energy_fn = EnergyMLP(input_dim=29, hidden_dims=[256, 256])
        >>> # ... train energy_fn ...
        >>> detector = EBMAnomalyDetector(energy_fn)
        >>> detector.fit_threshold(X_val, y_val, percentile=95)
        >>> predictions = detector.predict(X_test)
        >>> scores = detector.score(X_test)
    """

    def __init__(
        self,
        energy_fn: EnergyMLP,
        threshold: Optional[float] = None
    ):
        """
        Initialize the anomaly detector.

        Args:
            energy_fn: Trained energy function
            threshold: Optional pre-defined threshold. If not provided,
                      must call fit_threshold before predict.
        """
        self.energy_fn = energy_fn
        self.threshold = threshold
        self.threshold_info: Optional[ThresholdInfo] = None

    def score(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Compute anomaly scores (energy values) for input samples.

        Higher scores indicate more anomalous samples.

        Args:
            X: Input samples of shape (n_samples, n_features) or
               (n_features,) for a single sample

        Returns:
            Anomaly scores of shape (n_samples,)

        Note:
            The score is simply the energy value. For EBMs trained on
            normal data, anomalies should have higher energy.
        """
        if isinstance(X, np.ndarray):
            x_tensor = Tensor(X, requires_grad=False)
        else:
            x_tensor = X

        single_sample = (x_tensor.data.ndim == 1)
        if single_sample:
            x_tensor = Tensor(x_tensor.data.reshape(1, -1), requires_grad=False)

        energy = self.energy_fn(x_tensor)

        scores = energy.data.flatten()

        if single_sample:
            return scores[0]

        return scores

    def fit_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        percentile: float = 95.0
    ) -> float:
        """
        Fit the decision threshold based on validation data.

        Sets the threshold such that `percentile`% of normal samples
        have scores below the threshold.

        Args:
            X_val: Validation features of shape (n_samples, n_features)
            y_val: Validation labels (0 = normal, 1 = anomaly)
            percentile: Percentile of normal scores to use as threshold
                       (default 95 means 5% false positive rate on normal)

        Returns:
            The fitted threshold value

        Raises:
            ValueError: If percentile is not in (0, 100]
            ValueError: If y_val contains no normal samples (all 1s)

        Example:
            >>> detector.fit_threshold(X_val, y_val, percentile=95)
            >>> # Now 5% of normal samples would be flagged as anomalies
        """
        if percentile <= 0 or percentile > 100:
            raise ValueError(f"percentile must be in (0, 100], got {percentile}")

        normal_mask = (y_val == 0)
        if not normal_mask.any():
            raise ValueError("y_val contains no normal samples (all 1s)")

        X_normal = X_val[normal_mask]

        normal_scores = self.score(X_normal)

        self.threshold = float(np.percentile(normal_scores, percentile))

        self.threshold_info = ThresholdInfo(
            threshold=self.threshold,
            percentile=percentile,
            n_samples=len(y_val),
            n_normal=len(X_normal),
            normal_score_mean=float(normal_scores.mean()),
            normal_score_std=float(normal_scores.std()),
            normal_score_min=float(normal_scores.min()),
            normal_score_max=float(normal_scores.max()),
        )

        return self.threshold

    def fit_threshold_unsupervised(
        self,
        X: np.ndarray,
        percentile: float = 95.0
    ) -> float:
        """
        Fit threshold without labels (assumes all samples are normal).

        Useful when you only have unlabeled data assumed to be mostly normal.

        Args:
            X: Input samples of shape (n_samples, n_features)
            percentile: Percentile to use as threshold (default 95)

        Returns:
            The fitted threshold value
        """
        if percentile <= 0 or percentile > 100:
            raise ValueError(f"percentile must be in (0, 100], got {percentile}")

        scores = self.score(X)
        self.threshold = float(np.percentile(scores, percentile))

        self.threshold_info = ThresholdInfo(
            threshold=self.threshold,
            percentile=percentile,
            n_samples=len(X),
            n_normal=len(X),
            normal_score_mean=float(scores.mean()),
            normal_score_std=float(scores.std()),
            normal_score_min=float(scores.min()),
            normal_score_max=float(scores.max()),
        )

        return self.threshold

    def predict(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Predict anomaly labels for input samples.

        Args:
            X: Input samples of shape (n_samples, n_features) or
               (n_features,) for a single sample

        Returns:
            Binary predictions of shape (n_samples,)
            1 = anomaly, 0 = normal

        Raises:
            ValueError: If threshold has not been set
        """
        if self.threshold is None:
            raise ValueError(
                "Threshold not set. Call fit_threshold first or "
                "provide threshold in constructor."
            )

        scores = self.score(X)

        if np.isscalar(scores):
            return int(scores > self.threshold)

        return (scores > self.threshold).astype(int)

    def detect(self, X: Union[np.ndarray, Tensor]) -> DetectionResult:
        """
        Perform anomaly detection and return detailed results.

        Args:
            X: Input samples

        Returns:
            DetectionResult containing scores, predictions, and threshold

        Raises:
            ValueError: If threshold has not been set
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call fit_threshold first.")

        scores = self.score(X)

        if np.isscalar(scores):
            scores = np.array([scores])

        predictions = (scores > self.threshold).astype(int)

        return DetectionResult(
            scores=scores,
            predictions=predictions,
            threshold=self.threshold,
        )

    def predict_proba(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Compute pseudo-probabilities of being anomaly.

        Uses a sigmoid transform of the energy relative to threshold
        to produce values in [0, 1].

        Args:
            X: Input samples

        Returns:
            Pseudo-probabilities of shape (n_samples,)
            Higher values indicate more likely anomaly

        Note:
            These are not true probabilities but a monotonic transform
            of the energy scores that may be more interpretable.
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call fit_threshold first.")

        scores = self.score(X)

        scale = self.threshold_info.normal_score_std if self.threshold_info else 1.0
        if scale < 1e-8:
            scale = 1.0

        proba = 1.0 / (1.0 + np.exp(-(scores - self.threshold) / scale))

        return proba

    def get_score_statistics(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute statistics of scores for analysis.

        Args:
            X: Input samples
            y: Optional labels for separate normal/anomaly stats

        Returns:
            Dictionary of score statistics
        """
        scores = self.score(X)

        stats = {
            'all': {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'median': float(np.median(scores)),
            }
        }

        if y is not None:
            normal_mask = (y == 0)
            anomaly_mask = (y == 1)

            if normal_mask.any():
                normal_scores = scores[normal_mask]
                stats['normal'] = {
                    'mean': float(normal_scores.mean()),
                    'std': float(normal_scores.std()),
                    'min': float(normal_scores.min()),
                    'max': float(normal_scores.max()),
                    'count': int(normal_mask.sum()),
                }

            if anomaly_mask.any():
                anomaly_scores = scores[anomaly_mask]
                stats['anomaly'] = {
                    'mean': float(anomaly_scores.mean()),
                    'std': float(anomaly_scores.std()),
                    'min': float(anomaly_scores.min()),
                    'max': float(anomaly_scores.max()),
                    'count': int(anomaly_mask.sum()),
                }

        return stats

    def __repr__(self) -> str:
        threshold_str = f"{self.threshold:.4f}" if self.threshold is not None else "None"
        return (
            f"EBMAnomalyDetector(\n"
            f"  energy_fn: {self.energy_fn.__class__.__name__},\n"
            f"  threshold: {threshold_str}\n"
            f")"
        )


def create_detector(
    energy_fn: EnergyMLP,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    percentile: float = 95.0,
    threshold: Optional[float] = None
) -> EBMAnomalyDetector:
    """
    Factory function to create and optionally configure a detector.

    Args:
        energy_fn: Trained energy function
        X_val: Optional validation features for threshold fitting
        y_val: Optional validation labels
        percentile: Percentile for threshold (default 95)
        threshold: Optional explicit threshold (overrides fitting)

    Returns:
        Configured EBMAnomalyDetector

    Example:
        >>> detector = create_detector(
        ...     energy_fn,
        ...     X_val=X_val,
        ...     y_val=y_val,
        ...     percentile=95
        ... )
    """
    detector = EBMAnomalyDetector(energy_fn, threshold=threshold)

    if threshold is None and X_val is not None:
        if y_val is not None:
            detector.fit_threshold(X_val, y_val, percentile=percentile)
        else:
            detector.fit_threshold_unsupervised(X_val, percentile=percentile)

    return detector
