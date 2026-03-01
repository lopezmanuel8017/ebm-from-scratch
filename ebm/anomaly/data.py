"""
Data loading and preprocessing for anomaly detection.

This module provides utilities for loading and preprocessing datasets
for anomaly detection with Energy-Based Models, particularly focused
on the Credit Card Fraud Detection dataset.

The key function is load_credit_card_data which:
- Loads the CSV file
- Normalizes the Amount feature
- Splits data for semi-supervised learning (train on normal only)
- Returns train/val/test splits with proper labels
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class AnomalyDataset:
    """
    Container for anomaly detection dataset splits.

    Attributes:
        X_train: Training features (normal samples only)
        X_val: Validation features (mix of normal and anomaly)
        X_test: Test features (mix of normal and anomaly)
        y_val: Validation labels (0 = normal, 1 = anomaly)
        y_test: Test labels (0 = normal, 1 = anomaly)
        feature_names: List of feature names
        stats: Dictionary of dataset statistics
    """
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: Optional[list] = None
    stats: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return (
            f"AnomalyDataset(\n"
            f"  X_train: {self.X_train.shape},\n"
            f"  X_val: {self.X_val.shape} (anomaly rate: {self.y_val.mean():.3f}),\n"
            f"  X_test: {self.X_test.shape} (anomaly rate: {self.y_test.mean():.3f})\n"
            f")"
        )


def load_credit_card_data(
    path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    normalize: bool = True,
    drop_time: bool = True,
    random_state: Optional[int] = 42
) -> AnomalyDataset:
    """
    Load and preprocess credit card fraud dataset.

    The dataset is preprocessed for semi-supervised anomaly detection:
    - Training set contains only normal transactions
    - Validation and test sets contain both normal and fraud transactions

    Args:
        path: Path to the creditcard.csv file
        train_ratio: Fraction of normal data to use for training (default 0.8)
        val_ratio: Fraction of normal data for validation (default 0.1)
                   The remaining (1 - train_ratio - val_ratio) goes to test
        normalize: Whether to normalize features (default True)
        drop_time: Whether to drop the Time column (default True)
        random_state: Random seed for reproducible splits (default 42)

    Returns:
        AnomalyDataset containing train/val/test splits

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If ratios don't sum to <= 1.0

    Example:
        >>> dataset = load_credit_card_data("creditcard.csv")
        >>> print(dataset)
        AnomalyDataset(
          X_train: (227451, 29),
          X_val: (28431, 29) (anomaly rate: 0.008),
          X_test: (28925, 29) (anomaly rate: 0.008)
        )
    """

    if train_ratio + val_ratio > 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be <= 1.0"
        )

    data = _load_csv(path)

    header = data[0]
    values = data[1:]

    X_full = np.array(values, dtype=np.float64)

    class_idx = header.index('Class')
    time_idx = header.index('Time') if 'Time' in header else None

    y = X_full[:, class_idx].astype(int)

    drop_cols = [class_idx]
    if drop_time and time_idx is not None:
        drop_cols.append(time_idx)

    keep_cols = [i for i in range(X_full.shape[1]) if i not in drop_cols]
    X = X_full[:, keep_cols]

    feature_names = [header[i] for i in keep_cols]

    if normalize:
        X, normalization_stats = _normalize_features(X, feature_names)
    else:
        normalization_stats = None

    if random_state is not None:
        np.random.seed(random_state)

    normal_mask = (y == 0)
    anomaly_mask = (y == 1)

    X_normal = X[normal_mask]
    X_anomaly = X[anomaly_mask]

    n_normal = len(X_normal)
    n_anomaly = len(X_anomaly)

    normal_indices = np.random.permutation(n_normal)
    X_normal = X_normal[normal_indices]

    n_train = int(train_ratio * n_normal)
    n_val = int(val_ratio * n_normal)

    X_train = X_normal[:n_train]
    X_val_normal = X_normal[n_train:n_train + n_val]
    X_test_normal = X_normal[n_train + n_val:]

    anomaly_indices = np.random.permutation(n_anomaly)
    X_anomaly = X_anomaly[anomaly_indices]

    n_anomaly_val = n_anomaly // 2
    X_val_anomaly = X_anomaly[:n_anomaly_val]
    X_test_anomaly = X_anomaly[n_anomaly_val:]

    X_val = np.vstack([X_val_normal, X_val_anomaly])
    y_val = np.hstack([
        np.zeros(len(X_val_normal), dtype=int),
        np.ones(len(X_val_anomaly), dtype=int)
    ])

    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.hstack([
        np.zeros(len(X_test_normal), dtype=int),
        np.ones(len(X_test_anomaly), dtype=int)
    ])

    val_perm = np.random.permutation(len(y_val))
    X_val = X_val[val_perm]
    y_val = y_val[val_perm]

    test_perm = np.random.permutation(len(y_test))
    X_test = X_test[test_perm]
    y_test = y_test[test_perm]

    stats = {
        'n_total': len(y),
        'n_normal': n_normal,
        'n_anomaly': n_anomaly,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'n_features': X.shape[1],
        'anomaly_rate_original': n_anomaly / len(y),
        'anomaly_rate_val': y_val.mean(),
        'anomaly_rate_test': y_test.mean(),
        'normalization': normalization_stats,
    }

    return AnomalyDataset(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        stats=stats,
    )


def _load_csv(path: str) -> list:
    """
    Load CSV file without pandas dependency.

    Args:
        path: Path to CSV file

    Returns:
        List of lists (rows)

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    import csv

    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    return data


def _normalize_features(
    X: np.ndarray,
    feature_names: list
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize features using z-score normalization.

    For the credit card dataset:
    - V1-V28 are already PCA-transformed and relatively normalized
    - Amount needs explicit normalization

    Args:
        X: Feature array of shape (n_samples, n_features)
        feature_names: List of feature names

    Returns:
        Tuple of (normalized X, normalization statistics dict)
    """
    X_normalized = X.copy()
    stats = {}

    amount_idx = None
    for i, name in enumerate(feature_names):
        if name.lower() == 'amount':
            amount_idx = i
            break

    if amount_idx is not None:
        amount_mean = X[:, amount_idx].mean()
        amount_std = X[:, amount_idx].std()

        if amount_std < 1e-8:
            amount_std = 1.0

        X_normalized[:, amount_idx] = (X[:, amount_idx] - amount_mean) / amount_std

        stats['amount'] = {
            'mean': amount_mean,
            'std': amount_std,
            'index': amount_idx,
        }

    stats['global'] = {
        'means': X.mean(axis=0),
        'stds': X.std(axis=0),
    }

    return X_normalized, stats


def normalize_like_training(
    X: np.ndarray,
    normalization_stats: Dict[str, Any],
    feature_names: Optional[list] = None,
) -> np.ndarray:
    """
    Normalize new data using statistics from training data.

    This is useful for normalizing test data with the same parameters
    used during training.

    Args:
        X: Feature array to normalize
        normalization_stats: Statistics from load_credit_card_data
        feature_names: List of feature names

    Returns:
        Normalized feature array
    """
    X_normalized = X.copy()

    if 'amount' in normalization_stats:
        amount_info = normalization_stats['amount']
        amount_idx = amount_info['index']
        X_normalized[:, amount_idx] = (
            (X[:, amount_idx] - amount_info['mean']) / amount_info['std']
        )

    return X_normalized


def create_synthetic_anomaly_data(
    n_normal: int = 1000,
    n_anomaly: int = 50,
    n_features: int = 10,
    normal_mean: float = 0.0,
    normal_std: float = 1.0,
    anomaly_shift: float = 3.0,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic data for testing anomaly detection.

    Normal samples are drawn from N(normal_mean, normal_std^2 * I).
    Anomaly samples are shifted by anomaly_shift in a random direction.

    Args:
        n_normal: Number of normal samples
        n_anomaly: Number of anomaly samples
        n_features: Number of features
        normal_mean: Mean of normal distribution
        normal_std: Standard deviation of normal distribution
        anomaly_shift: Shift magnitude for anomalies
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        where train contains only normal samples
    """

    if random_state is not None:
        np.random.seed(random_state)

    X_normal = np.random.randn(n_normal, n_features) * normal_std + normal_mean

    X_anomaly = np.random.randn(n_anomaly, n_features) * normal_std + normal_mean
    shift_direction = np.random.randn(n_features)
    shift_direction = shift_direction / np.linalg.norm(shift_direction)
    X_anomaly = X_anomaly + anomaly_shift * shift_direction

    n_train = int(0.8 * n_normal)
    X_train = X_normal[:n_train]
    X_test_normal = X_normal[n_train:]

    X_test = np.vstack([X_test_normal, X_anomaly])
    y_test = np.hstack([
        np.zeros(len(X_test_normal), dtype=int),
        np.ones(n_anomaly, dtype=int)
    ])

    perm = np.random.permutation(len(y_test))
    X_test = X_test[perm]
    y_test = y_test[perm]

    y_train = np.zeros(len(X_train), dtype=int)

    return X_train, X_test, y_train, y_test


def get_dataset_summary(dataset: AnomalyDataset) -> str:
    """
    Generate a human-readable summary of the dataset.

    Args:
        dataset: AnomalyDataset object

    Returns:
        Formatted summary string
    """
    stats = dataset.stats or {}

    lines = [
        "=" * 50,
        "Anomaly Detection Dataset Summary",
        "=" * 50,
        f"Training samples (normal only): {len(dataset.X_train):,}",
        f"Validation samples: {len(dataset.X_val):,}",
        f"  - Normal: {int((1 - dataset.y_val.mean()) * len(dataset.y_val)):,}",
        f"  - Anomaly: {int(dataset.y_val.sum()):,}",
        f"  - Anomaly rate: {dataset.y_val.mean():.4f}",
        f"Test samples: {len(dataset.X_test):,}",
        f"  - Normal: {int((1 - dataset.y_test.mean()) * len(dataset.y_test)):,}",
        f"  - Anomaly: {int(dataset.y_test.sum()):,}",
        f"  - Anomaly rate: {dataset.y_test.mean():.4f}",
        f"Number of features: {dataset.X_train.shape[1]}",
    ]

    if 'anomaly_rate_original' in stats:
        lines.append(f"Original anomaly rate: {stats['anomaly_rate_original']:.4f}")

    lines.append("=" * 50)

    return "\n".join(lines)
