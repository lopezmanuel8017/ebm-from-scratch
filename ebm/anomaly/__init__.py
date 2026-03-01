"""
Anomaly detection module for Energy-Based Models.

This module provides tools for using EBMs for anomaly detection:
- Data loading and preprocessing (credit card fraud dataset)
- EBM-based anomaly detector
- Evaluation metrics and curves
"""

from ebm.anomaly.data import (
    AnomalyDataset,
    load_credit_card_data,
    normalize_like_training,
    create_synthetic_anomaly_data,
    get_dataset_summary,
)

from ebm.anomaly.detector import (
    EBMAnomalyDetector,
    DetectionResult,
    ThresholdInfo,
    create_detector,
)

from ebm.anomaly.evaluate import (
    EvaluationResult,
    CurveData,
    evaluate_detector,
    compute_auroc,
    compute_auprc,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_accuracy,
    compute_false_positive_rate,
    compute_specificity,
    get_roc_curve_data,
    get_pr_curve_data,
    find_optimal_threshold,
    evaluate_at_thresholds,
    format_evaluation_report,
)

__all__ = [
    "AnomalyDataset",
    "load_credit_card_data",
    "normalize_like_training",
    "create_synthetic_anomaly_data",
    "get_dataset_summary",
    "EBMAnomalyDetector",
    "DetectionResult",
    "ThresholdInfo",
    "create_detector",
    "EvaluationResult",
    "CurveData",
    "evaluate_detector",
    "compute_auroc",
    "compute_auprc",
    "roc_curve",
    "precision_recall_curve",
    "confusion_matrix",
    "compute_precision",
    "compute_recall",
    "compute_f1",
    "compute_accuracy",
    "compute_false_positive_rate",
    "compute_specificity",
    "get_roc_curve_data",
    "get_pr_curve_data",
    "find_optimal_threshold",
    "evaluate_at_thresholds",
    "format_evaluation_report",
]
