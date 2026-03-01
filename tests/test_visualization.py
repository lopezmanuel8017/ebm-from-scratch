"""
Tests for visualization utilities.

Note: Most visualization functions return matplotlib figures. We test that:
- Functions don't crash
- They return the expected types (Figure or None if matplotlib not installed)
- They handle edge cases gracefully
"""

import pytest
import numpy as np

from ebm.utils.visualization import (
    plot_energy_histogram,
    plot_2d_energy_landscape,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_training_history,
    plot_sample_comparison,
    plot_score_distribution,
    create_evaluation_figure,
)

from ebm.core.energy import EnergyMLP
from ebm.anomaly.detector import EBMAnomalyDetector


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@pytest.fixture
def small_energy_fn():
    """Create a small energy function for testing."""
    np.random.seed(42)
    return EnergyMLP(input_dim=10, hidden_dims=[32, 32], activation="swish")


@pytest.fixture
def detector(small_energy_fn):
    """Create an anomaly detector with threshold."""
    detector = EBMAnomalyDetector(small_energy_fn)
    X_val = np.random.randn(100, 10)
    y_val = np.array([0] * 90 + [1] * 10)
    detector.fit_threshold(X_val, y_val, percentile=95)
    return detector


@pytest.fixture
def sample_data_10d():
    """Create sample 10D data."""
    return np.random.randn(100, 10)


@pytest.fixture
def sample_data_2d():
    """Create sample 2D data."""
    return np.random.randn(100, 2)


@pytest.fixture
def energy_fn_2d():
    """Create a 2D energy function for landscape plots."""
    np.random.seed(42)
    return EnergyMLP(input_dim=2, hidden_dims=[32, 32], activation="swish")


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotEnergyHistogram:
    """Test energy histogram plotting."""

    def test_basic_histogram(self, detector, sample_data_10d):
        """Test basic histogram plotting."""
        X_normal = sample_data_10d
        X_anomaly = sample_data_10d + 5

        fig = plot_energy_histogram(
            detector, X_normal, X_anomaly, show=False
        )

        assert fig is not None
        plt.close(fig)

    def test_histogram_with_options(self, detector, sample_data_10d):
        """Test histogram with custom options."""
        X_normal = sample_data_10d[:50]
        X_anomaly = sample_data_10d[50:]

        fig = plot_energy_histogram(
            detector, X_normal, X_anomaly,
            bins=30,
            alpha=0.7,
            density=False,
            figsize=(8, 5),
            title="Custom Title",
            show=False
        )

        assert fig is not None
        plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlot2DEnergyLandscape:
    """Test 2D energy landscape plotting."""

    def test_basic_landscape(self, energy_fn_2d):
        """Test basic landscape plotting."""
        fig = plot_2d_energy_landscape(
            energy_fn_2d,
            xlim=(-2, 2),
            ylim=(-2, 2),
            resolution=20,
            show=False
        )

        assert fig is not None
        plt.close(fig)

    def test_landscape_with_data(self, energy_fn_2d, sample_data_2d):
        """Test landscape with data overlay."""
        fig = plot_2d_energy_landscape(
            energy_fn_2d,
            xlim=(-3, 3),
            ylim=(-3, 3),
            resolution=20,
            data=sample_data_2d,
            data_color='white',
            data_size=5,
            show=False
        )

        assert fig is not None
        plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotROCCurve:
    """Test ROC curve plotting."""

    def test_basic_roc(self):
        """Test basic ROC curve plotting."""
        fpr = np.array([0, 0.1, 0.3, 0.5, 1.0])
        tpr = np.array([0, 0.5, 0.7, 0.9, 1.0])

        fig = plot_roc_curve(fpr, tpr, auc=0.85, show=False)

        assert fig is not None
        plt.close(fig)

    def test_roc_without_auc(self):
        """Test ROC curve without AUC in legend."""
        fpr = np.array([0, 0.2, 0.5, 1.0])
        tpr = np.array([0, 0.6, 0.8, 1.0])

        fig = plot_roc_curve(fpr, tpr, show=False)

        assert fig is not None
        plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotPRCurve:
    """Test Precision-Recall curve plotting."""

    def test_basic_pr(self):
        """Test basic PR curve plotting."""
        recall = np.array([0, 0.3, 0.6, 0.8, 1.0])
        precision = np.array([1.0, 0.9, 0.8, 0.6, 0.5])

        fig = plot_precision_recall_curve(recall, precision, auc=0.75, show=False)

        assert fig is not None
        plt.close(fig)

    def test_pr_with_baseline(self):
        """Test PR curve with baseline."""
        recall = np.array([0, 0.5, 1.0])
        precision = np.array([1.0, 0.7, 0.5])

        fig = plot_precision_recall_curve(
            recall, precision,
            auc=0.7,
            baseline=0.1,
            show=False
        )

        assert fig is not None
        plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotTrainingHistory:
    """Test training history plotting."""

    def test_basic_history(self):
        """Test basic history plotting."""
        history = [
            {'loss': 1.0, 'E_real': -1.0, 'E_fake': 0.5},
            {'loss': 0.8, 'E_real': -0.8, 'E_fake': 0.3},
            {'loss': 0.6, 'E_real': -0.6, 'E_fake': 0.1},
        ]

        fig = plot_training_history(history, show=False)

        assert fig is not None
        plt.close(fig)

    def test_history_specific_metrics(self):
        """Test history with specific metrics."""
        history = [
            {'loss': 1.0, 'E_real': -1.0, 'entropy': 2.0},
            {'loss': 0.5, 'E_real': -0.5, 'entropy': 2.5},
        ]

        fig = plot_training_history(
            history,
            metrics=['loss', 'entropy'],
            show=False
        )

        assert fig is not None
        plt.close(fig)

    def test_empty_history(self):
        """Test with empty history."""
        fig = plot_training_history([], show=False)
        assert fig is None


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotSampleComparison:
    """Test sample comparison plotting."""

    def test_basic_comparison(self, sample_data_2d):
        """Test basic sample comparison."""
        real = sample_data_2d
        generated = sample_data_2d + np.random.randn(*sample_data_2d.shape) * 0.1

        fig = plot_sample_comparison(real, generated, show=False)

        assert fig is not None
        plt.close(fig)

    def test_comparison_different_dims(self):
        """Test comparison with different dimension selection."""
        real = np.random.randn(100, 5)
        generated = np.random.randn(100, 5)

        fig = plot_sample_comparison(
            real, generated,
            dims=(2, 4),
            show=False
        )

        assert fig is not None
        plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotScoreDistribution:
    """Test score distribution plotting."""

    def test_basic_distribution(self):
        """Test basic score distribution."""
        scores = np.random.randn(100)

        fig = plot_score_distribution(scores, show=False)

        assert fig is not None
        plt.close(fig)

    def test_distribution_with_labels(self):
        """Test distribution with labels."""
        scores = np.concatenate([
            np.random.randn(80),
            np.random.randn(20) + 2
        ])
        labels = np.array([0] * 80 + [1] * 20)

        fig = plot_score_distribution(
            scores, labels=labels,
            threshold=1.0,
            show=False
        )

        assert fig is not None
        plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestCreateEvaluationFigure:
    """Test comprehensive evaluation figure creation."""

    def test_basic_figure(self):
        """Test basic evaluation figure."""
        roc_data = {
            'fpr': np.array([0, 0.1, 0.3, 1.0]),
            'tpr': np.array([0, 0.5, 0.8, 1.0]),
            'auc': 0.85,
        }
        pr_data = {
            'recall': np.array([0, 0.5, 1.0]),
            'precision': np.array([1.0, 0.8, 0.5]),
            'auc': 0.7,
        }

        fig = create_evaluation_figure(roc_data, pr_data, show=False)

        assert fig is not None
        plt.close(fig)

    def test_figure_with_scores(self):
        """Test evaluation figure with score distribution."""
        roc_data = {
            'fpr': np.array([0, 0.2, 1.0]),
            'tpr': np.array([0, 0.7, 1.0]),
            'auc': 0.8,
        }
        pr_data = {
            'recall': np.array([0, 0.5, 1.0]),
            'precision': np.array([1.0, 0.7, 0.5]),
            'auc': 0.65,
        }
        score_data = {
            'scores': np.concatenate([np.random.randn(80), np.random.randn(20) + 2]),
            'labels': np.array([0] * 80 + [1] * 20),
            'threshold': 1.0,
        }

        fig = create_evaluation_figure(roc_data, pr_data, score_data, show=False)

        assert fig is not None
        plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestEdgeCases:
    """Test edge cases for visualization functions."""

    def test_single_sample_histogram(self, detector):
        """Test histogram with single sample."""
        X_normal = np.random.randn(1, 10)
        X_anomaly = np.random.randn(1, 10)

        fig = plot_energy_histogram(detector, X_normal, X_anomaly, show=False)
        assert fig is not None
        plt.close(fig)

    def test_many_samples_histogram(self, detector):
        """Test histogram with many samples."""
        X_normal = np.random.randn(1000, 10)
        X_anomaly = np.random.randn(100, 10)

        fig = plot_energy_histogram(detector, X_normal, X_anomaly, show=False)
        assert fig is not None
        plt.close(fig)

    def test_score_distribution_no_labels(self):
        """Test score distribution without labels."""
        scores = np.random.randn(100)

        fig = plot_score_distribution(scores, show=False)
        assert fig is not None
        plt.close(fig)

    def test_history_single_epoch(self):
        """Test history with single epoch."""
        history = [{'loss': 1.0}]

        fig = plot_training_history(history, show=False)
        assert fig is not None
        plt.close(fig)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestSaveFunctionality:
    """Test that plots can be saved to files."""

    def test_save_histogram(self, detector, sample_data_10d, tmp_path):
        """Test saving histogram to file."""
        save_path = tmp_path / "histogram.png"
        X_normal = sample_data_10d
        X_anomaly = sample_data_10d + 3

        fig = plot_energy_histogram(
            detector, X_normal, X_anomaly,
            save_path=str(save_path),
            show=False
        )

        assert save_path.exists()
        assert fig is not None
        plt.close(fig)

    def test_save_landscape(self, energy_fn_2d, tmp_path):
        """Test saving energy landscape to file."""
        save_path = tmp_path / "landscape.png"

        fig = plot_2d_energy_landscape(
            energy_fn_2d,
            xlim=(-2, 2),
            ylim=(-2, 2),
            resolution=10,
            save_path=str(save_path),
            show=False
        )

        assert save_path.exists()
        assert fig is not None
        plt.close(fig)

    def test_save_roc_curve(self, tmp_path):
        """Test saving ROC curve to file."""
        save_path = tmp_path / "roc.png"
        fpr = np.array([0, 0.1, 0.5, 1.0])
        tpr = np.array([0, 0.6, 0.9, 1.0])

        fig = plot_roc_curve(
            fpr, tpr, auc=0.85,
            save_path=str(save_path),
            show=False
        )

        assert save_path.exists()
        assert fig is not None
        plt.close(fig)

    def test_save_pr_curve(self, tmp_path):
        """Test saving PR curve to file."""
        save_path = tmp_path / "pr.png"
        recall = np.array([0, 0.5, 1.0])
        precision = np.array([1.0, 0.8, 0.5])

        fig = plot_precision_recall_curve(
            recall, precision, auc=0.7,
            save_path=str(save_path),
            show=False
        )

        assert save_path.exists()
        assert fig is not None
        plt.close(fig)

    def test_save_training_history(self, tmp_path):
        """Test saving training history to file."""
        save_path = tmp_path / "history.png"
        history = [
            {'loss': 1.0, 'E_real': -1.0},
            {'loss': 0.5, 'E_real': -0.5},
        ]

        fig = plot_training_history(
            history,
            save_path=str(save_path),
            show=False
        )

        assert save_path.exists()
        assert fig is not None
        plt.close(fig)

    def test_save_sample_comparison(self, sample_data_2d, tmp_path):
        """Test saving sample comparison to file."""
        save_path = tmp_path / "comparison.png"
        real = sample_data_2d
        generated = sample_data_2d + np.random.randn(*sample_data_2d.shape) * 0.1

        fig = plot_sample_comparison(
            real, generated,
            save_path=str(save_path),
            show=False
        )

        assert save_path.exists()
        assert fig is not None
        plt.close(fig)

    def test_save_score_distribution(self, tmp_path):
        """Test saving score distribution to file."""
        save_path = tmp_path / "scores.png"
        scores = np.random.randn(100)
        labels = np.array([0] * 80 + [1] * 20)

        fig = plot_score_distribution(
            scores, labels=labels,
            save_path=str(save_path),
            show=False
        )

        assert save_path.exists()
        assert fig is not None
        plt.close(fig)

    def test_save_evaluation_figure(self, tmp_path):
        """Test saving evaluation figure to file."""
        save_path = tmp_path / "eval.png"
        roc_data = {
            'fpr': np.array([0, 0.2, 1.0]),
            'tpr': np.array([0, 0.8, 1.0]),
            'auc': 0.85,
        }
        pr_data = {
            'recall': np.array([0, 0.5, 1.0]),
            'precision': np.array([1.0, 0.7, 0.5]),
            'auc': 0.7,
        }

        fig = create_evaluation_figure(
            roc_data, pr_data,
            save_path=str(save_path),
            show=False
        )

        assert save_path.exists()
        assert fig is not None
        plt.close(fig)
