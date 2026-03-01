"""
Comprehensive tests for Anomaly Detection module (Section 10).

Tests cover:
- Data loading and preprocessing (load_credit_card_data, synthetic data)
- AnomalyDataset structure and properties
- EBMAnomalyDetector (scoring, threshold fitting, prediction)
- Evaluation metrics (AUROC, AUPRC, F1, precision, recall)
- Curve computations (ROC, PR)
- Edge cases and boundary conditions
- Integration tests
"""

import pytest
import numpy as np
import tempfile
import os

from ebm.core.autodiff import Tensor
from ebm.core.energy import EnergyMLP
from ebm.anomaly.data import (
    AnomalyDataset,
    load_credit_card_data,
    normalize_like_training,
    create_synthetic_anomaly_data,
    get_dataset_summary,
    _load_csv,
    _normalize_features,
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


@pytest.fixture
def synthetic_data():
    """Create synthetic anomaly detection data."""
    np.random.seed(42)
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
def small_energy_fn():
    """Create a small energy function for testing."""
    np.random.seed(42)
    return EnergyMLP(input_dim=10, hidden_dims=[32, 32], activation="swish")


@pytest.fixture
def trained_energy_fn():
    """Create a 'trained' energy function that separates normal/anomaly."""
    np.random.seed(42)

    class SimpleQuadraticEnergy(EnergyMLP):
        """Simple energy that gives low energy near origin."""

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x_tensor = Tensor(x, requires_grad=False)
            else:
                x_tensor = x
            from ebm.core.ops import tensor_sum, mul
            x_sq = mul(x_tensor, x_tensor)
            energy = tensor_sum(x_sq, axis=-1, keepdims=True)
            return mul(energy, Tensor(0.5))

    return SimpleQuadraticEnergy(input_dim=10, hidden_dims=[32, 32])


@pytest.fixture
def detector(small_energy_fn):
    """Create an anomaly detector."""
    return EBMAnomalyDetector(small_energy_fn)


@pytest.fixture
def detector_with_threshold(trained_energy_fn):
    """Create a detector with a fitted threshold."""
    detector = EBMAnomalyDetector(trained_energy_fn)
    X_val = np.random.randn(100, 10)
    y_val = np.zeros(100, dtype=int)
    y_val[-10:] = 1  # Last 10 are anomalies
    detector.fit_threshold(X_val, y_val, percentile=95)
    return detector


@pytest.fixture
def mock_csv_file():
    """Create a mock credit card CSV file for testing."""
    header = ['Time', 'V1', 'V2', 'V3', 'Amount', 'Class']
    rows = []

    for i in range(100):
        row = [
            str(float(i)),
            str(np.random.randn()),
            str(np.random.randn()),
            str(np.random.randn()),
            str(abs(np.random.randn()) * 100),
            '0',
        ]
        rows.append(row)

    for i in range(10):
        row = [
            str(float(100 + i)),
            str(np.random.randn() + 3),
            str(np.random.randn() + 3),
            str(np.random.randn() + 3),
            str(abs(np.random.randn()) * 100),
            '1',
        ]
        rows.append(row)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(row) + '\n')
        temp_path = f.name

    yield temp_path

    os.unlink(temp_path)


class TestAnomalyDataset:
    """Test AnomalyDataset dataclass."""

    def test_dataset_creation(self):
        """Test creating an AnomalyDataset."""
        dataset = AnomalyDataset(
            X_train=np.random.randn(100, 5),
            X_val=np.random.randn(20, 5),
            X_test=np.random.randn(30, 5),
            y_val=np.array([0] * 15 + [1] * 5),
            y_test=np.array([0] * 25 + [1] * 5),
        )

        assert dataset.X_train.shape == (100, 5)
        assert dataset.X_val.shape == (20, 5)
        assert dataset.X_test.shape == (30, 5)
        assert len(dataset.y_val) == 20
        assert len(dataset.y_test) == 30

    def test_dataset_repr(self):
        """Test AnomalyDataset string representation."""
        dataset = AnomalyDataset(
            X_train=np.random.randn(100, 5),
            X_val=np.random.randn(20, 5),
            X_test=np.random.randn(30, 5),
            y_val=np.array([0] * 18 + [1] * 2),
            y_test=np.array([0] * 27 + [1] * 3),
        )

        repr_str = repr(dataset)
        assert 'AnomalyDataset' in repr_str
        assert 'X_train' in repr_str
        assert 'anomaly rate' in repr_str


class TestLoadCreditCardData:
    """Test credit card data loading functions."""

    def test_load_csv_basic(self, mock_csv_file):
        """Test basic CSV loading."""
        data = _load_csv(mock_csv_file)

        assert len(data) == 111
        assert data[0] == ['Time', 'V1', 'V2', 'V3', 'Amount', 'Class']

    def test_load_csv_file_not_found(self):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            _load_csv('/nonexistent/path/to/file.csv')

    def test_load_credit_card_data(self, mock_csv_file):
        """Test full data loading pipeline."""
        dataset = load_credit_card_data(
            mock_csv_file,
            train_ratio=0.8,
            val_ratio=0.1,
            normalize=True,
            drop_time=True,
            random_state=42,
        )

        assert isinstance(dataset, AnomalyDataset)
        assert dataset.X_train.ndim == 2
        assert dataset.X_train.shape[1] == 4

    def test_load_credit_card_data_no_normalize(self, mock_csv_file):
        """Test loading without normalization."""
        dataset = load_credit_card_data(
            mock_csv_file,
            normalize=False,
            random_state=42,
        )

        assert dataset.stats['normalization'] is None

    def test_load_credit_card_data_keep_time(self, mock_csv_file):
        """Test loading with Time column kept."""
        dataset = load_credit_card_data(
            mock_csv_file,
            drop_time=False,
            random_state=42,
        )

        assert 'Time' in dataset.feature_names

    def test_load_credit_card_data_invalid_ratios(self, mock_csv_file):
        """Test error on invalid train/val ratios."""
        with pytest.raises(ValueError, match="must be <= 1.0"):
            load_credit_card_data(
                mock_csv_file,
                train_ratio=0.8,
                val_ratio=0.3,
            )

    def test_load_credit_card_data_stats(self, mock_csv_file):
        """Test that stats are computed correctly."""
        dataset = load_credit_card_data(mock_csv_file, random_state=42)

        stats = dataset.stats
        assert 'n_total' in stats
        assert 'n_normal' in stats
        assert 'n_anomaly' in stats
        assert 'anomaly_rate_original' in stats
        assert stats['n_total'] == 110
        assert stats['n_normal'] == 100
        assert stats['n_anomaly'] == 10


class TestNormalizeFeatures:
    """Test feature normalization."""

    def test_normalize_features_basic(self):
        """Test basic normalization."""
        X = np.random.randn(100, 5)
        X[:, -1] = np.random.exponential(100, 100)
        feature_names = ['V1', 'V2', 'V3', 'V4', 'Amount']

        X_norm, stats = _normalize_features(X, feature_names)

        assert X_norm.shape == X.shape
        assert 'amount' in stats

        assert abs(X_norm[:, -1].mean()) < 0.5

    def test_normalize_features_no_amount(self):
        """Test normalization without Amount column."""
        X = np.random.randn(100, 3)
        feature_names = ['V1', 'V2', 'V3']

        X_norm, stats = _normalize_features(X, feature_names)

        assert X_norm.shape == X.shape
        assert 'amount' not in stats


class TestNormalizeLikeTraining:
    """Test normalize_like_training function."""

    def test_normalize_like_training(self, mock_csv_file):
        """Test normalizing new data like training data."""
        dataset = load_credit_card_data(mock_csv_file, random_state=42)

        X_new = np.random.randn(10, 4)
        X_new[:, -1] = abs(np.random.randn(10)) * 100

        X_normalized = normalize_like_training(
            X_new,
            dataset.stats['normalization'],
            dataset.feature_names,
        )

        assert X_normalized.shape == X_new.shape


class TestCreateSyntheticData:
    """Test synthetic data creation."""

    def test_create_synthetic_data_basic(self):
        """Test basic synthetic data creation."""
        X_train, _X_test, y_train, y_test = create_synthetic_anomaly_data(
            n_normal=100,
            n_anomaly=10,
            n_features=5,
            random_state=42,
        )

        assert X_train.shape == (80, 5)
        assert y_train.sum() == 0
        assert y_test.sum() == 10

    def test_create_synthetic_data_shapes(self):
        """Test synthetic data shapes."""
        X_train, X_test, y_train, y_test = create_synthetic_anomaly_data(
            n_normal=500,
            n_anomaly=50,
            n_features=20,
        )

        assert X_train.ndim == 2
        assert X_test.ndim == 2
        assert X_train.shape[1] == 20
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_create_synthetic_data_separability(self):
        """Test that anomalies are shifted from normal."""
        _X_train, X_test, _y_train, y_test = create_synthetic_anomaly_data(
            n_normal=500,
            n_anomaly=50,
            n_features=10,
            anomaly_shift=5.0,
            random_state=42,
        )

        normal_test = X_test[y_test == 0]
        anomaly_test = X_test[y_test == 1]

        normal_norm = np.linalg.norm(normal_test, axis=1).mean()
        anomaly_norm = np.linalg.norm(anomaly_test, axis=1).mean()

        assert anomaly_norm > normal_norm


class TestGetDatasetSummary:
    """Test dataset summary function."""

    def test_get_dataset_summary(self, mock_csv_file):
        """Test summary generation."""
        dataset = load_credit_card_data(mock_csv_file, random_state=42)
        summary = get_dataset_summary(dataset)

        assert 'Dataset Summary' in summary
        assert 'Training samples' in summary
        assert 'Validation samples' in summary
        assert 'Test samples' in summary


class TestEBMAnomalyDetector:
    """Test EBMAnomalyDetector class."""

    def test_detector_creation(self, small_energy_fn):
        """Test creating a detector."""
        detector = EBMAnomalyDetector(small_energy_fn)

        assert detector.energy_fn is small_energy_fn
        assert detector.threshold is None

    def test_detector_creation_with_threshold(self, small_energy_fn):
        """Test creating a detector with preset threshold."""
        detector = EBMAnomalyDetector(small_energy_fn, threshold=5.0)

        assert detector.threshold == 5.0

    def test_score_single_sample(self, detector):
        """Test scoring a single sample."""
        X = np.random.randn(10)
        score = detector.score(X)

        assert np.isscalar(score)
        assert np.isfinite(score)

    def test_score_batch(self, detector):
        """Test scoring a batch of samples."""
        X = np.random.randn(32, 10)
        scores = detector.score(X)

        assert scores.shape == (32,)
        assert np.all(np.isfinite(scores))

    def test_score_tensor_input(self, detector):
        """Test scoring with Tensor input."""
        X = Tensor(np.random.randn(16, 10), requires_grad=False)
        scores = detector.score(X)

        assert scores.shape == (16,)

    def test_fit_threshold(self, detector):
        """Test fitting threshold."""
        X_val = np.random.randn(100, 10)
        y_val = np.zeros(100, dtype=int)
        y_val[-10:] = 1

        threshold = detector.fit_threshold(X_val, y_val, percentile=95)

        assert detector.threshold is not None
        assert np.isfinite(threshold)
        assert detector.threshold_info is not None
        assert detector.threshold_info.percentile == 95

    def test_fit_threshold_invalid_percentile(self, detector):
        """Test error on invalid percentile."""
        X_val = np.random.randn(100, 10)
        y_val = np.zeros(100, dtype=int)

        with pytest.raises(ValueError, match="percentile"):
            detector.fit_threshold(X_val, y_val, percentile=0)

        with pytest.raises(ValueError, match="percentile"):
            detector.fit_threshold(X_val, y_val, percentile=101)

    def test_fit_threshold_no_normal_samples(self, detector):
        """Test error when no normal samples."""
        X_val = np.random.randn(10, 10)
        y_val = np.ones(10, dtype=int)

        with pytest.raises(ValueError, match="no normal samples"):
            detector.fit_threshold(X_val, y_val)

    def test_fit_threshold_unsupervised(self, detector):
        """Test unsupervised threshold fitting."""
        X = np.random.randn(100, 10)

        threshold = detector.fit_threshold_unsupervised(X, percentile=95)

        assert detector.threshold is not None
        assert np.isfinite(threshold)

    def test_predict_without_threshold(self, detector):
        """Test error when predicting without threshold."""
        X = np.random.randn(10, 10)

        with pytest.raises(ValueError, match="Threshold not set"):
            detector.predict(X)

    def test_predict_batch(self, detector_with_threshold):
        """Test batch prediction."""
        X = np.random.randn(50, 10)
        predictions = detector_with_threshold.predict(X)

        assert predictions.shape == (50,)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_predict_single_sample(self, detector_with_threshold):
        """Test single sample prediction."""
        X = np.random.randn(10)
        prediction = detector_with_threshold.predict(X)

        assert prediction in {0, 1}

    def test_detect(self, detector_with_threshold):
        """Test detect method returns DetectionResult."""
        X = np.random.randn(50, 10)
        result = detector_with_threshold.detect(X)

        assert isinstance(result, DetectionResult)
        assert len(result.scores) == 50
        assert len(result.predictions) == 50
        assert result.threshold is not None

    def test_detect_without_threshold(self, detector):
        """Test error on detect without threshold."""
        X = np.random.randn(10, 10)

        with pytest.raises(ValueError, match="Threshold not set"):
            detector.detect(X)

    def test_predict_proba(self, detector_with_threshold):
        """Test probability prediction."""
        X = np.random.randn(50, 10)
        proba = detector_with_threshold.predict_proba(X)

        assert proba.shape == (50,)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_get_score_statistics(self, detector):
        """Test score statistics computation."""
        X = np.random.randn(100, 10)
        y = np.array([0] * 80 + [1] * 20)

        stats = detector.get_score_statistics(X, y)

        assert 'all' in stats
        assert 'normal' in stats
        assert 'anomaly' in stats
        assert 'mean' in stats['all']
        assert stats['normal']['count'] == 80
        assert stats['anomaly']['count'] == 20

    def test_get_score_statistics_no_labels(self, detector):
        """Test score statistics without labels."""
        X = np.random.randn(100, 10)
        stats = detector.get_score_statistics(X)

        assert 'all' in stats
        assert 'normal' not in stats
        assert 'anomaly' not in stats

    def test_detector_repr(self, detector):
        """Test detector string representation."""
        repr_str = repr(detector)
        assert 'EBMAnomalyDetector' in repr_str

    def test_detector_repr_with_threshold(self, detector_with_threshold):
        """Test detector repr with threshold."""
        repr_str = repr(detector_with_threshold)
        assert 'EBMAnomalyDetector' in repr_str
        assert 'threshold:' in repr_str


class TestDetectionResult:
    """Test DetectionResult dataclass."""

    def test_detection_result_repr(self):
        """Test DetectionResult representation."""
        result = DetectionResult(
            scores=np.array([1.0, 2.0, 3.0]),
            predictions=np.array([0, 0, 1]),
            threshold=2.5,
        )

        repr_str = repr(result)
        assert 'DetectionResult' in repr_str
        assert 'n_samples' in repr_str
        assert 'n_anomalies_detected' in repr_str


class TestThresholdInfo:
    """Test ThresholdInfo dataclass."""

    def test_threshold_info_repr(self):
        """Test ThresholdInfo representation."""
        info = ThresholdInfo(
            threshold=5.0,
            percentile=95.0,
            n_samples=100,
            n_normal=90,
            normal_score_mean=2.0,
            normal_score_std=1.0,
            normal_score_min=0.5,
            normal_score_max=4.5,
        )

        repr_str = repr(info)
        assert 'ThresholdInfo' in repr_str
        assert 'percentile' in repr_str


class TestCreateDetector:
    """Test create_detector factory function."""

    def test_create_detector_basic(self, small_energy_fn):
        """Test basic detector creation."""
        detector = create_detector(small_energy_fn)

        assert isinstance(detector, EBMAnomalyDetector)
        assert detector.threshold is None

    def test_create_detector_with_threshold(self, small_energy_fn):
        """Test detector creation with explicit threshold."""
        detector = create_detector(small_energy_fn, threshold=3.0)

        assert detector.threshold == 3.0

    def test_create_detector_with_fitting(self, small_energy_fn):
        """Test detector creation with automatic threshold fitting."""
        X_val = np.random.randn(100, 10)
        y_val = np.array([0] * 90 + [1] * 10)

        detector = create_detector(
            small_energy_fn,
            X_val=X_val,
            y_val=y_val,
            percentile=95,
        )

        assert detector.threshold is not None

    def test_create_detector_unsupervised_fitting(self, small_energy_fn):
        """Test detector creation with unsupervised fitting."""
        X_val = np.random.randn(100, 10)

        detector = create_detector(
            small_energy_fn,
            X_val=X_val,
            percentile=95,
        )

        assert detector.threshold is not None


class TestConfusionMatrix:
    """Test confusion matrix computation."""

    def test_confusion_matrix_basic(self):
        """Test basic confusion matrix."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])

        cm = confusion_matrix(y_true, y_pred)

        assert cm['TP'] == 2
        assert cm['TN'] == 1
        assert cm['FP'] == 1
        assert cm['FN'] == 1

    def test_confusion_matrix_all_correct(self):
        """Test confusion matrix with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        cm = confusion_matrix(y_true, y_pred)

        assert cm['TP'] == 2
        assert cm['TN'] == 2
        assert cm['FP'] == 0
        assert cm['FN'] == 0

    def test_confusion_matrix_all_wrong(self):
        """Test confusion matrix with all wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])

        cm = confusion_matrix(y_true, y_pred)

        assert cm['TP'] == 0
        assert cm['TN'] == 0
        assert cm['FP'] == 2
        assert cm['FN'] == 2


class TestPrecisionRecallF1:
    """Test precision, recall, F1 computation."""

    def test_compute_precision(self):
        """Test precision computation."""
        cm = {'TP': 10, 'TN': 80, 'FP': 5, 'FN': 5}
        precision = compute_precision(cm)

        assert precision == 10 / 15

    def test_compute_precision_no_positives(self):
        """Test precision when no positive predictions."""
        cm = {'TP': 0, 'TN': 100, 'FP': 0, 'FN': 10}
        precision = compute_precision(cm)

        assert precision == 0.0

    def test_compute_recall(self):
        """Test recall computation."""
        cm = {'TP': 10, 'TN': 80, 'FP': 5, 'FN': 5}
        recall = compute_recall(cm)

        assert recall == 10 / 15

    def test_compute_recall_no_actual_positives(self):
        """Test recall when no actual positives."""
        cm = {'TP': 0, 'TN': 100, 'FP': 10, 'FN': 0}
        recall = compute_recall(cm)

        assert recall == 0.0

    def test_compute_f1(self):
        """Test F1 computation."""
        cm = {'TP': 10, 'TN': 80, 'FP': 5, 'FN': 5}
        f1 = compute_f1(cm)

        precision = 10 / 15
        recall = 10 / 15
        expected_f1 = 2 * precision * recall / (precision + recall)

        np.testing.assert_allclose(f1, expected_f1)

    def test_compute_f1_zero(self):
        """Test F1 when precision and recall are zero."""
        cm = {'TP': 0, 'TN': 100, 'FP': 0, 'FN': 0}
        f1 = compute_f1(cm)

        assert f1 == 0.0

    def test_compute_accuracy(self):
        """Test accuracy computation."""
        cm = {'TP': 10, 'TN': 80, 'FP': 5, 'FN': 5}
        accuracy = compute_accuracy(cm)

        assert accuracy == 90 / 100

    def test_compute_false_positive_rate(self):
        """Test FPR computation."""
        cm = {'TP': 10, 'TN': 80, 'FP': 20, 'FN': 5}
        fpr = compute_false_positive_rate(cm)

        assert fpr == 20 / 100

    def test_compute_specificity(self):
        """Test specificity computation."""
        cm = {'TP': 10, 'TN': 80, 'FP': 20, 'FN': 5}
        specificity = compute_specificity(cm)

        assert specificity == 80 / 100


class TestROCCurve:
    """Test ROC curve computation."""

    def test_roc_curve_perfect(self):
        """Test ROC curve with perfect separation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        fpr, tpr, _thresholds = roc_curve(y_true, scores)

        assert len(fpr) == len(tpr)
        assert fpr[0] == 0
        assert tpr[0] == 0
        assert fpr[-1] == 1 or tpr[-1] == 1

    def test_roc_curve_random(self):
        """Test ROC curve with random scores."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        scores = np.random.rand(100)

        fpr, tpr, _thresholds = roc_curve(y_true, scores)

        assert len(fpr) == len(tpr)
        assert np.all(np.diff(fpr) >= -1e-10)


class TestPRCurve:
    """Test Precision-Recall curve computation."""

    def test_pr_curve_perfect(self):
        """Test PR curve with perfect separation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        precision, recall, _thresholds = precision_recall_curve(y_true, scores)

        assert len(precision) == len(recall)
        assert recall[0] == 0
        assert recall[-1] == 1
        assert np.all(np.diff(recall) >= 0)

    def test_pr_curve_random(self):
        """Test PR curve with random scores."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        scores = np.random.rand(100)

        precision, recall, _thresholds = precision_recall_curve(y_true, scores)

        assert len(precision) == len(recall)


class TestAUROC:
    """Test AUROC computation."""

    def test_auroc_perfect(self):
        """Test AUROC with perfect separation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        auroc = compute_auroc(y_true, scores)

        assert auroc == 1.0

    def test_auroc_random(self):
        """Test AUROC with random scores (should be ~0.5)."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        scores = np.random.rand(1000)

        auroc = compute_auroc(y_true, scores)

        assert 0.4 < auroc < 0.6

    def test_auroc_single_class(self):
        """Test AUROC with single class (returns 0.5)."""
        y_true = np.zeros(100)
        scores = np.random.rand(100)

        auroc = compute_auroc(y_true, scores)

        assert auroc == 0.5


class TestAUPRC:
    """Test AUPRC computation."""

    def test_auprc_perfect(self):
        """Test AUPRC with perfect separation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        auprc = compute_auprc(y_true, scores)

        assert auprc > 0.9

    def test_auprc_random(self):
        """Test AUPRC with random scores."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        scores = np.random.rand(1000)

        auprc = compute_auprc(y_true, scores)

        assert 0.3 < auprc < 0.7


class TestEvaluateDetector:
    """Test evaluate_detector function."""

    def test_evaluate_detector(self, detector_with_threshold):
        """Test comprehensive detector evaluation."""
        X_test = np.random.randn(100, 10)
        y_test = np.array([0] * 90 + [1] * 10)

        result = evaluate_detector(detector_with_threshold, X_test, y_test)

        assert isinstance(result, EvaluationResult)
        assert 0 <= result.auroc <= 1
        assert 0 <= result.auprc <= 1
        assert 0 <= result.f1 <= 1
        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1
        assert 0 <= result.accuracy <= 1

    def test_evaluate_detector_no_threshold(self, detector):
        """Test error when evaluating without threshold."""
        X_test = np.random.randn(100, 10)
        y_test = np.zeros(100)

        with pytest.raises(ValueError, match="threshold must be set"):
            evaluate_detector(detector, X_test, y_test)


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_evaluation_result_repr(self):
        """Test EvaluationResult representation."""
        result = EvaluationResult(
            auroc=0.95,
            auprc=0.90,
            f1=0.85,
            precision=0.88,
            recall=0.82,
            accuracy=0.92,
            true_positive_rate=0.82,
            false_positive_rate=0.05,
            confusion_matrix={'TP': 82, 'TN': 855, 'FP': 45, 'FN': 18},
        )

        repr_str = repr(result)
        assert 'EvaluationResult' in repr_str
        assert 'AUROC' in repr_str

    def test_evaluation_result_to_dict(self):
        """Test EvaluationResult to_dict method."""
        result = EvaluationResult(
            auroc=0.95,
            auprc=0.90,
            f1=0.85,
            precision=0.88,
            recall=0.82,
            accuracy=0.92,
            true_positive_rate=0.82,
            false_positive_rate=0.05,
            confusion_matrix={'TP': 82, 'TN': 855, 'FP': 45, 'FN': 18},
        )

        d = result.to_dict()

        assert d['auroc'] == 0.95
        assert d['TP'] == 82


class TestCurveData:
    """Test CurveData class."""

    def test_curve_data_repr(self):
        """Test CurveData representation."""
        curve = CurveData(
            x=np.linspace(0, 1, 10),
            y=np.linspace(0, 1, 10),
            thresholds=np.linspace(1, 0, 10),
            auc=0.85,
        )

        repr_str = repr(curve)
        assert 'CurveData' in repr_str
        assert 'auc' in repr_str


class TestGetCurveData:
    """Test curve data extraction functions."""

    def test_get_roc_curve_data(self, detector_with_threshold):
        """Test ROC curve data extraction."""
        X_test = np.random.randn(100, 10)
        y_test = np.array([0] * 90 + [1] * 10)

        curve = get_roc_curve_data(detector_with_threshold, X_test, y_test)

        assert isinstance(curve, CurveData)
        assert len(curve.x) == len(curve.y)
        assert 0 <= curve.auc <= 1

    def test_get_pr_curve_data(self, detector_with_threshold):
        """Test PR curve data extraction."""
        X_test = np.random.randn(100, 10)
        y_test = np.array([0] * 90 + [1] * 10)

        curve = get_pr_curve_data(detector_with_threshold, X_test, y_test)

        assert isinstance(curve, CurveData)
        assert len(curve.x) == len(curve.y)


class TestFindOptimalThreshold:
    """Test find_optimal_threshold function."""

    def test_find_optimal_threshold_f1(self):
        """Test finding optimal threshold for F1."""
        y_true = np.array([0] * 80 + [1] * 20)
        scores = np.concatenate([
            np.random.randn(80) * 0.5,
            np.random.randn(20) * 0.5 + 2,
        ])

        threshold, value = find_optimal_threshold(y_true, scores, metric='f1')

        assert np.isfinite(threshold)
        assert 0 <= value <= 1

    def test_find_optimal_threshold_youden(self):
        """Test finding optimal threshold for Youden's J."""
        y_true = np.array([0] * 80 + [1] * 20)
        scores = np.concatenate([
            np.random.randn(80) * 0.5,
            np.random.randn(20) * 0.5 + 2,
        ])

        threshold, _value = find_optimal_threshold(y_true, scores, metric='youden')

        assert np.isfinite(threshold)

    def test_find_optimal_threshold_invalid_metric(self):
        """Test error on invalid metric."""
        y_true = np.zeros(100)
        scores = np.random.rand(100)

        with pytest.raises(ValueError, match="metric must be"):
            find_optimal_threshold(y_true, scores, metric='invalid')


class TestEvaluateAtThresholds:
    """Test evaluate_at_thresholds function."""

    def test_evaluate_at_thresholds(self):
        """Test evaluating at multiple thresholds."""
        y_true = np.array([0] * 80 + [1] * 20)
        scores = np.random.rand(100)

        results = evaluate_at_thresholds(y_true, scores)

        assert 'thresholds' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert len(results['precision']) == len(results['thresholds'])

    def test_evaluate_at_specific_thresholds(self):
        """Test evaluating at specific thresholds."""
        y_true = np.array([0] * 80 + [1] * 20)
        scores = np.random.rand(100)
        thresholds = np.array([0.2, 0.4, 0.6, 0.8])

        results = evaluate_at_thresholds(y_true, scores, thresholds=thresholds)

        assert len(results['thresholds']) == 4


class TestFormatEvaluationReport:
    """Test format_evaluation_report function."""

    def test_format_evaluation_report(self):
        """Test report formatting."""
        result = EvaluationResult(
            auroc=0.95,
            auprc=0.90,
            f1=0.85,
            precision=0.88,
            recall=0.82,
            accuracy=0.92,
            true_positive_rate=0.82,
            false_positive_rate=0.05,
            confusion_matrix={'TP': 82, 'TN': 855, 'FP': 45, 'FN': 18},
        )

        report = format_evaluation_report(result)

        assert 'Evaluation Report' in report
        assert 'AUROC' in report
        assert 'Confusion Matrix' in report
        assert 'True Positives' in report


class TestIntegration:
    """Integration tests for the complete anomaly detection pipeline."""

    def test_full_pipeline_synthetic(self, synthetic_data, small_energy_fn):
        """Test complete pipeline with synthetic data."""
        _X_train, X_test, _y_train, y_test = synthetic_data

        detector = EBMAnomalyDetector(small_energy_fn)

        n_val = len(X_test) // 2
        X_val, X_test_final = X_test[:n_val], X_test[n_val:]
        y_val, y_test_final = y_test[:n_val], y_test[n_val:]

        detector.fit_threshold(X_val, y_val, percentile=95)

        result = evaluate_detector(detector, X_test_final, y_test_final)

        assert 0 <= result.auroc <= 1
        assert 0 <= result.f1 <= 1

    def test_full_pipeline_with_curves(self, synthetic_data, small_energy_fn):
        """Test pipeline with curve computation."""
        X_train, X_test, _y_train, y_test = synthetic_data

        detector = EBMAnomalyDetector(small_energy_fn)
        detector.fit_threshold_unsupervised(X_train, percentile=95)

        roc_data = get_roc_curve_data(detector, X_test, y_test)
        pr_data = get_pr_curve_data(detector, X_test, y_test)

        assert roc_data.auc > 0
        assert pr_data.auc > 0

    def test_factory_function_pipeline(self, synthetic_data, small_energy_fn):
        """Test using factory function in pipeline."""
        X_train, X_test, _y_train, y_test = synthetic_data

        detector = create_detector(
            small_energy_fn,
            X_val=X_train,
            percentile=95,
        )

        assert detector.threshold is not None

        predictions = detector.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_csv_to_evaluation(self, mock_csv_file):
        """Test from CSV loading to evaluation."""
        dataset = load_credit_card_data(mock_csv_file, random_state=42)

        n_features = dataset.X_train.shape[1]
        energy_fn = EnergyMLP(input_dim=n_features, hidden_dims=[32, 32])

        detector = EBMAnomalyDetector(energy_fn)
        detector.fit_threshold(dataset.X_val, dataset.y_val, percentile=95)

        result = evaluate_detector(detector, dataset.X_test, dataset.y_test)

        assert isinstance(result, EvaluationResult)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample_scoring(self, detector):
        """Test scoring a single sample."""
        X = np.random.randn(10)
        score = detector.score(X)

        assert np.isscalar(score)

    def test_empty_predictions(self, detector_with_threshold):
        """Test that very high threshold gives no anomaly predictions."""
        detector_with_threshold.threshold = 1e10

        X = np.random.randn(100, 10)
        predictions = detector_with_threshold.predict(X)

        assert predictions.sum() == 0

    def test_all_anomaly_predictions(self, detector_with_threshold):
        """Test that very low threshold gives all anomaly predictions."""
        detector_with_threshold.threshold = -1e10

        X = np.random.randn(100, 10)
        predictions = detector_with_threshold.predict(X)

        assert predictions.sum() == 100

    def test_threshold_at_percentile_boundary(self, detector):
        """Test threshold at 100th percentile."""
        X_val = np.random.randn(100, 10)
        y_val = np.zeros(100, dtype=int)

        threshold = detector.fit_threshold(X_val, y_val, percentile=100)

        scores = detector.score(X_val)
        np.testing.assert_allclose(threshold, scores.max(), rtol=1e-5)

    def test_small_dataset(self, small_energy_fn):
        """Test with very small dataset."""
        detector = EBMAnomalyDetector(small_energy_fn)

        X_val = np.random.randn(5, 10)
        y_val = np.array([0, 0, 0, 0, 1])

        threshold = detector.fit_threshold(X_val, y_val, percentile=75)

        assert np.isfinite(threshold)

    def test_balanced_dataset(self, small_energy_fn):
        """Test with balanced classes."""
        detector = EBMAnomalyDetector(small_energy_fn)

        X_val = np.random.randn(100, 10)
        y_val = np.array([0] * 50 + [1] * 50)

        detector.fit_threshold(X_val, y_val, percentile=95)

        X_test = np.random.randn(100, 10)
        y_test = np.array([0] * 50 + [1] * 50)

        result = evaluate_detector(detector, X_test, y_test)

        assert 0 <= result.auroc <= 1

    def test_auroc_inverse_scores(self):
        """Test that inverse scores give complement AUROC."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        auroc1 = compute_auroc(y_true, scores)
        auroc2 = compute_auroc(y_true, -scores)

        np.testing.assert_allclose(auroc1 + auroc2, 1.0, rtol=1e-5)

class TestNumericalStability:
    """Test numerical stability of computations."""

    def test_scores_with_extreme_values(self, detector):
        """Test scoring with extreme input values."""
        X = np.random.randn(50, 10) * 100
        scores = detector.score(X)

        assert np.all(np.isfinite(scores))

    def test_auroc_with_identical_scores(self):
        """Test AUROC when all scores are identical."""
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])

        auroc = compute_auroc(y_true, scores)

        assert np.isfinite(auroc)

    def test_precision_with_no_predictions(self):
        """Test precision when no positive predictions."""
        cm = {'TP': 0, 'TN': 100, 'FP': 0, 'FN': 10}
        precision = compute_precision(cm)

        assert precision == 0.0

    def test_recall_with_no_positives(self):
        """Test recall when no actual positives."""
        cm = {'TP': 0, 'TN': 100, 'FP': 10, 'FN': 0}
        recall = compute_recall(cm)

        assert recall == 0.0


class TestReproducibility:
    """Test reproducibility with random seeds."""

    def test_synthetic_data_reproducibility(self):
        """Test that synthetic data is reproducible."""
        data1 = create_synthetic_anomaly_data(random_state=42)
        data2 = create_synthetic_anomaly_data(random_state=42)

        np.testing.assert_array_equal(data1[0], data2[0])
        np.testing.assert_array_equal(data1[1], data2[1])

    def test_data_loading_reproducibility(self, mock_csv_file):
        """Test that data loading is reproducible."""
        dataset1 = load_credit_card_data(mock_csv_file, random_state=42)
        dataset2 = load_credit_card_data(mock_csv_file, random_state=42)

        np.testing.assert_array_equal(dataset1.X_train, dataset2.X_train)
        np.testing.assert_array_equal(dataset1.y_val, dataset2.y_val)

    def test_different_seeds_different_splits(self, mock_csv_file):
        """Test that different seeds give different splits."""
        dataset1 = load_credit_card_data(mock_csv_file, random_state=42)
        dataset2 = load_credit_card_data(mock_csv_file, random_state=123)

        assert dataset1.X_train.shape == dataset2.X_train.shape
        assert not np.allclose(dataset1.X_train, dataset2.X_train)


class TestAdditionalCoverage:
    """Additional tests for edge cases to improve coverage."""

    def test_normalize_features_zero_std(self):
        """Test normalization when Amount has zero std."""
        X = np.ones((100, 3))
        feature_names = ['V1', 'V2', 'Amount']

        X_norm, _stats = _normalize_features(X, feature_names)

        assert np.all(np.isfinite(X_norm))

    def test_normalize_like_training_no_amount(self):
        """Test normalize_like_training when no amount stats."""
        X = np.random.randn(10, 3)
        normalization_stats = {'global': {'means': np.zeros(3), 'stds': np.ones(3)}}
        feature_names = ['V1', 'V2', 'V3']

        X_normalized = normalize_like_training(X, normalization_stats, feature_names)

        np.testing.assert_array_equal(X_normalized, X)

    def test_detector_predict_proba_zero_std(self, small_energy_fn):
        """Test predict_proba when threshold_info has zero std."""
        detector = EBMAnomalyDetector(small_energy_fn)

        detector.threshold = 1.0
        detector.threshold_info = ThresholdInfo(
            threshold=1.0,
            percentile=95.0,
            n_samples=100,
            n_normal=90,
            normal_score_mean=0.5,
            normal_score_std=0.0,
            normal_score_min=0.5,
            normal_score_max=0.5,
        )

        X = np.random.randn(10, 10)
        proba = detector.predict_proba(X)

        assert np.all(np.isfinite(proba))
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_pr_curve_no_positives(self):
        """Test PR curve when no positive samples."""
        y_true = np.zeros(10, dtype=int)
        scores = np.random.rand(10)

        precision, recall, _thresholds = precision_recall_curve(y_true, scores)

        assert len(precision) == 2
        assert len(recall) == 2

    def test_evaluation_result_repr(self):
        """Test EvaluationResult repr includes all fields."""
        result = EvaluationResult(
            auroc=0.95,
            auprc=0.90,
            f1=0.85,
            precision=0.88,
            recall=0.82,
            accuracy=0.92,
            true_positive_rate=0.82,
            false_positive_rate=0.05,
            confusion_matrix={'TP': 82, 'TN': 855, 'FP': 45, 'FN': 18},
        )

        repr_str = repr(result)

        assert 'AUROC' in repr_str
        assert 'AUPRC' in repr_str
        assert 'F1' in repr_str
        assert 'Precision' in repr_str
        assert 'FPR' in repr_str

    def test_find_optimal_threshold_precision(self):
        """Test finding optimal threshold for precision."""
        y_true = np.array([0] * 80 + [1] * 20)
        scores = np.concatenate([
            np.random.randn(80) * 0.5,
            np.random.randn(20) * 0.5 + 2,
        ])

        threshold, value = find_optimal_threshold(y_true, scores, metric='precision')

        assert np.isfinite(threshold)
        assert 0 <= value <= 1

    def test_find_optimal_threshold_recall(self):
        """Test finding optimal threshold for recall."""
        y_true = np.array([0] * 80 + [1] * 20)
        scores = np.concatenate([
            np.random.randn(80) * 0.5,
            np.random.randn(20) * 0.5 + 2,
        ])

        threshold, value = find_optimal_threshold(y_true, scores, metric='recall')

        assert np.isfinite(threshold)
        assert 0 <= value <= 1

    def test_get_score_statistics_empty_classes(self, detector):
        """Test score statistics with single class only."""
        X = np.random.randn(50, 10)
        y = np.zeros(50, dtype=int)

        stats = detector.get_score_statistics(X, y)

        assert 'all' in stats
        assert 'normal' in stats
        assert 'anomaly' not in stats

    def test_detection_result_single_sample(self, detector_with_threshold):
        """Test detection with single sample returns proper array."""
        X = np.random.randn(1, 10)
        result = detector_with_threshold.detect(X)

        assert len(result.scores) == 1
        assert len(result.predictions) == 1

    def test_detect_single_sample_1d(self, detector_with_threshold):
        """Test detect with 1D input (single sample without batch dim)."""
        X = np.random.randn(10)
        result = detector_with_threshold.detect(X)

        assert len(result.scores) == 1
        assert len(result.predictions) == 1

    def test_predict_proba_without_threshold(self, detector):
        """Test predict_proba raises error without threshold."""
        X = np.random.randn(10, 10)

        with pytest.raises(ValueError, match="Threshold not set"):
            detector.predict_proba(X)

    def test_fit_threshold_unsupervised_invalid_percentile(self, detector):
        """Test unsupervised threshold fitting with invalid percentile."""
        X = np.random.randn(100, 10)

        with pytest.raises(ValueError, match="percentile"):
            detector.fit_threshold_unsupervised(X, percentile=0)

        with pytest.raises(ValueError, match="percentile"):
            detector.fit_threshold_unsupervised(X, percentile=101)

    def test_auprc_single_class(self):
        """Test AUPRC with single class (all positive or all negative)."""
        y_true = np.zeros(100, dtype=int)
        scores = np.random.rand(100)

        auprc = compute_auprc(y_true, scores)
        assert auprc == 0.0

        y_true = np.ones(100, dtype=int)
        scores = np.random.rand(100)

        auprc = compute_auprc(y_true, scores)
        assert auprc == 1.0
