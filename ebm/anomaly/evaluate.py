"""
Evaluation metrics for anomaly detection.

This module provides comprehensive evaluation functions for anomaly
detection systems, including:

- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- F1 Score
- Precision and Recall at various thresholds
- Confusion matrix statistics

Note: Some functions use sklearn for computation but we also provide
pure NumPy implementations for the core metrics.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from ebm.anomaly.detector import EBMAnomalyDetector


def _trapezoid(y, x=None):
    """Compute integral using trapezoidal rule (numpy version agnostic)."""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    else:
        return np.trapz(y, x)


@dataclass
class EvaluationResult:
    """
    Container for evaluation metrics.

    Attributes:
        auroc: Area Under ROC Curve
        auprc: Area Under Precision-Recall Curve
        f1: F1 score at the detector's threshold
        precision: Precision at the detector's threshold
        recall: Recall at the detector's threshold
        accuracy: Overall accuracy
        true_positive_rate: Same as recall
        false_positive_rate: Rate of normal samples incorrectly flagged
        confusion_matrix: Dict with TP, TN, FP, FN counts
    """
    auroc: float
    auprc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    true_positive_rate: float
    false_positive_rate: float
    confusion_matrix: Dict[str, int]

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(\n"
            f"  AUROC: {self.auroc:.4f},\n"
            f"  AUPRC: {self.auprc:.4f},\n"
            f"  F1: {self.f1:.4f},\n"
            f"  Precision: {self.precision:.4f},\n"
            f"  Recall: {self.recall:.4f},\n"
            f"  Accuracy: {self.accuracy:.4f},\n"
            f"  FPR: {self.false_positive_rate:.4f},\n"
            f"  TP: {self.confusion_matrix['TP']}, TN: {self.confusion_matrix['TN']}, "
            f"FP: {self.confusion_matrix['FP']}, FN: {self.confusion_matrix['FN']}\n"
            f")"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'auroc': self.auroc,
            'auprc': self.auprc,
            'f1': self.f1,
            'precision': self.precision,
            'recall': self.recall,
            'accuracy': self.accuracy,
            'true_positive_rate': self.true_positive_rate,
            'false_positive_rate': self.false_positive_rate,
            **self.confusion_matrix,
        }


@dataclass
class CurveData:
    """
    Container for curve data (ROC or PR).

    Attributes:
        x: X-axis values (e.g., FPR for ROC, Recall for PR)
        y: Y-axis values (e.g., TPR for ROC, Precision for PR)
        thresholds: Threshold values corresponding to each point
        auc: Area under the curve
    """
    x: np.ndarray
    y: np.ndarray
    thresholds: np.ndarray
    auc: float

    def __repr__(self) -> str:
        return f"CurveData(n_points={len(self.x)}, auc={self.auc:.4f})"


def evaluate_detector(
    detector: EBMAnomalyDetector,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> EvaluationResult:
    """
    Compute comprehensive evaluation metrics for an anomaly detector.

    Args:
        detector: Trained EBMAnomalyDetector with threshold set
        X_test: Test features of shape (n_samples, n_features)
        y_test: True labels (0 = normal, 1 = anomaly)

    Returns:
        EvaluationResult with all metrics

    Raises:
        ValueError: If detector threshold is not set
    """
    if detector.threshold is None:
        raise ValueError("Detector threshold must be set before evaluation")

    scores = detector.score(X_test)
    predictions = detector.predict(X_test)

    auroc = compute_auroc(y_test, scores)
    auprc = compute_auprc(y_test, scores)

    cm = confusion_matrix(y_test, predictions)
    precision = compute_precision(cm)
    recall = compute_recall(cm)
    f1 = compute_f1(cm)
    accuracy = compute_accuracy(cm)
    fpr = compute_false_positive_rate(cm)
    tpr = recall

    return EvaluationResult(
        auroc=auroc,
        auprc=auprc,
        f1=f1,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        true_positive_rate=tpr,
        false_positive_rate=fpr,
        confusion_matrix=cm,
    )


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve.

    Uses the trapezoidal rule for integration.

    Args:
        y_true: True binary labels (0 or 1)
        scores: Anomaly scores (higher = more anomalous)

    Returns:
        AUROC value in [0, 1]
    """
    y_true = np.asarray(y_true).flatten()
    scores = np.asarray(scores).flatten()

    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, _ = roc_curve(y_true, scores)

    auc = _trapezoid(tpr, fpr)

    return float(auc)


def compute_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Area Under the Precision-Recall Curve.

    Args:
        y_true: True binary labels (0 or 1)
        scores: Anomaly scores (higher = more anomalous)

    Returns:
        AUPRC value in [0, 1]
    """
    y_true = np.asarray(y_true).flatten()
    scores = np.asarray(scores).flatten()

    if len(np.unique(y_true)) < 2:
        return y_true.mean()

    precision, recall, _ = precision_recall_curve(y_true, scores)

    auc = _trapezoid(precision, recall)

    return float(abs(auc))


def roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.

    Args:
        y_true: True binary labels
        scores: Anomaly scores

    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    y_true = np.asarray(y_true).flatten()
    scores = np.asarray(scores).flatten()

    desc_score_indices = np.argsort(scores)[::-1]
    scores_sorted = scores[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.concatenate(([0], distinct_value_indices + 1))

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    total_positive = y_true.sum()
    total_negative = len(y_true) - total_positive

    tpr = np.zeros(len(threshold_idxs) + 1)
    fpr = np.zeros(len(threshold_idxs) + 1)
    thresholds = np.zeros(len(threshold_idxs) + 1)

    tpr[0] = 0.0
    fpr[0] = 0.0
    thresholds[0] = scores_sorted[0] + 1

    for i, idx in enumerate(threshold_idxs, 1):
        tpr[i] = tps[idx] / total_positive if total_positive > 0 else 0
        fpr[i] = fps[idx] / total_negative if total_negative > 0 else 0
        thresholds[i] = scores_sorted[idx]

    return fpr, tpr, thresholds


def precision_recall_curve(
    y_true: np.ndarray,
    scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall curve.

    Args:
        y_true: True binary labels
        scores: Anomaly scores

    Returns:
        Tuple of (precision, recall, thresholds)
        - precision and recall are arrays of the same length
        - thresholds has length = len(precision) - 1
    """
    y_true = np.asarray(y_true).flatten()
    scores = np.asarray(scores).flatten()

    desc_score_indices = np.argsort(scores)[::-1]
    scores_sorted = scores[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    tps = np.cumsum(y_true_sorted)

    n_predicted = np.arange(1, len(y_true) + 1)

    total_positive = y_true.sum()

    if total_positive == 0:
        return np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([scores_sorted[0]])

    precision_arr = tps / n_predicted
    recall_arr = tps / total_positive

    distinct_mask = np.concatenate([[True], np.diff(scores_sorted) != 0])

    distinct_mask[-1] = True

    precision = precision_arr[distinct_mask]
    recall = recall_arr[distinct_mask]
    thresholds = scores_sorted[distinct_mask]

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])

    return precision, recall, thresholds


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, int]:
    """
    Compute confusion matrix elements.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels

    Returns:
        Dictionary with TP, TN, FP, FN counts
    """
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def compute_precision(cm: Dict[str, int]) -> float:
    """
    Compute precision from confusion matrix.

    Precision = TP / (TP + FP)

    Args:
        cm: Confusion matrix dict with TP, TN, FP, FN

    Returns:
        Precision value in [0, 1]
    """
    tp = cm['TP']
    fp = cm['FP']
    denominator = tp + fp
    return tp / denominator if denominator > 0 else 0.0


def compute_recall(cm: Dict[str, int]) -> float:
    """
    Compute recall (sensitivity, true positive rate) from confusion matrix.

    Recall = TP / (TP + FN)

    Args:
        cm: Confusion matrix dict

    Returns:
        Recall value in [0, 1]
    """
    tp = cm['TP']
    fn = cm['FN']
    denominator = tp + fn
    return tp / denominator if denominator > 0 else 0.0


def compute_f1(cm: Dict[str, int]) -> float:
    """
    Compute F1 score from confusion matrix.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        cm: Confusion matrix dict

    Returns:
        F1 score in [0, 1]
    """
    precision = compute_precision(cm)
    recall = compute_recall(cm)
    denominator = precision + recall
    return 2 * precision * recall / denominator if denominator > 0 else 0.0


def compute_accuracy(cm: Dict[str, int]) -> float:
    """
    Compute accuracy from confusion matrix.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Args:
        cm: Confusion matrix dict

    Returns:
        Accuracy value in [0, 1]
    """
    tp, tn, fp, fn = cm['TP'], cm['TN'], cm['FP'], cm['FN']
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def compute_false_positive_rate(cm: Dict[str, int]) -> float:
    """
    Compute false positive rate from confusion matrix.

    FPR = FP / (FP + TN)

    Args:
        cm: Confusion matrix dict

    Returns:
        FPR value in [0, 1]
    """
    fp = cm['FP']
    tn = cm['TN']
    denominator = fp + tn
    return fp / denominator if denominator > 0 else 0.0


def compute_specificity(cm: Dict[str, int]) -> float:
    """
    Compute specificity (true negative rate) from confusion matrix.

    Specificity = TN / (TN + FP) = 1 - FPR

    Args:
        cm: Confusion matrix dict

    Returns:
        Specificity value in [0, 1]
    """
    return 1.0 - compute_false_positive_rate(cm)


def get_roc_curve_data(
    detector: EBMAnomalyDetector,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> CurveData:
    """
    Get ROC curve data for plotting.

    Args:
        detector: Anomaly detector
        X_test: Test features
        y_test: True labels

    Returns:
        CurveData with FPR, TPR, thresholds, and AUC
    """
    scores = detector.score(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    auc = _trapezoid(tpr, fpr)

    return CurveData(
        x=fpr,
        y=tpr,
        thresholds=thresholds,
        auc=float(auc),
    )


def get_pr_curve_data(
    detector: EBMAnomalyDetector,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> CurveData:
    """
    Get Precision-Recall curve data for plotting.

    Args:
        detector: Anomaly detector
        X_test: Test features
        y_test: True labels

    Returns:
        CurveData with Recall, Precision, thresholds, and AUC
    """
    scores = detector.score(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, scores)
    auc = _trapezoid(precision, recall)

    return CurveData(
        x=recall,
        y=precision,
        thresholds=thresholds,
        auc=float(abs(auc)),
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find the threshold that optimizes a given metric.

    Args:
        y_true: True binary labels
        scores: Anomaly scores
        metric: Metric to optimize ('f1', 'youden', 'precision', 'recall')

    Returns:
        Tuple of (optimal_threshold, optimal_metric_value)

    Raises:
        ValueError: If metric is not recognized
    """
    valid_metrics = ['f1', 'youden', 'precision', 'recall']
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}, got {metric}")

    y_true = np.asarray(y_true).flatten()
    scores = np.asarray(scores).flatten()

    thresholds = np.unique(scores)

    best_threshold = thresholds[0]
    best_value = -np.inf

    for thresh in thresholds:
        predictions = (scores > thresh).astype(int)
        cm = confusion_matrix(y_true, predictions)

        if metric == 'f1':
            value = compute_f1(cm)
        elif metric == 'youden':
            value = compute_recall(cm) - compute_false_positive_rate(cm)
        elif metric == 'precision':
            value = compute_precision(cm)
        elif metric == 'recall':
            value = compute_recall(cm)

        if value > best_value:
            best_value = value
            best_threshold = thresh

    return float(best_threshold), float(best_value)


def evaluate_at_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Evaluate metrics at multiple thresholds.

    Args:
        y_true: True binary labels
        scores: Anomaly scores
        thresholds: Optional array of thresholds. If None, uses unique scores.

    Returns:
        Dictionary with arrays of metric values at each threshold
    """
    y_true = np.asarray(y_true).flatten()
    scores = np.asarray(scores).flatten()

    if thresholds is None:
        thresholds = np.unique(scores)

    results = {
        'thresholds': thresholds,
        'precision': np.zeros(len(thresholds)),
        'recall': np.zeros(len(thresholds)),
        'f1': np.zeros(len(thresholds)),
        'fpr': np.zeros(len(thresholds)),
        'accuracy': np.zeros(len(thresholds)),
    }

    for i, thresh in enumerate(thresholds):
        predictions = (scores > thresh).astype(int)
        cm = confusion_matrix(y_true, predictions)

        results['precision'][i] = compute_precision(cm)
        results['recall'][i] = compute_recall(cm)
        results['f1'][i] = compute_f1(cm)
        results['fpr'][i] = compute_false_positive_rate(cm)
        results['accuracy'][i] = compute_accuracy(cm)

    return results


def format_evaluation_report(result: EvaluationResult) -> str:
    """
    Format evaluation results as a human-readable report.

    Args:
        result: EvaluationResult object

    Returns:
        Formatted string report
    """
    cm = result.confusion_matrix
    lines = [
        "=" * 50,
        "Anomaly Detection Evaluation Report",
        "=" * 50,
        "",
        "Classification Metrics:",
        f"  AUROC:     {result.auroc:.4f}",
        f"  AUPRC:     {result.auprc:.4f}",
        f"  F1 Score:  {result.f1:.4f}",
        f"  Precision: {result.precision:.4f}",
        f"  Recall:    {result.recall:.4f}",
        f"  Accuracy:  {result.accuracy:.4f}",
        "",
        "Confusion Matrix:",
        f"  True Positives:  {cm['TP']:>6}",
        f"  True Negatives:  {cm['TN']:>6}",
        f"  False Positives: {cm['FP']:>6}",
        f"  False Negatives: {cm['FN']:>6}",
        "",
        "",
        "Rates:",
        f"  True Positive Rate (Recall):  {result.true_positive_rate:.4f}",
        f"  False Positive Rate:          {result.false_positive_rate:.4f}",
        "=" * 50,
    ]

    return "\n".join(lines)
