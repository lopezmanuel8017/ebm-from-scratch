"""
Visualization utilities for Energy-Based Models.

This module provides plotting functions for:
- Energy histograms (normal vs anomaly)
- Energy landscapes (2D contour plots)
- ROC and PR curves
- Training history
- Sample distributions
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from ebm.core.autodiff import Tensor


def plot_energy_histogram(
    detector,
    X_normal: np.ndarray,
    X_anomaly: np.ndarray,
    bins: int = 50,
    alpha: float = 0.5,
    density: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Energy Distribution: Normal vs. Anomaly",
    xlabel: str = "Energy (Anomaly Score)",
    ylabel: str = "Density",
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[Any]:
    """
    Plot overlapping histograms of energy for normal vs anomaly samples.

    Args:
        detector: EBMAnomalyDetector with trained energy function
        X_normal: Normal samples of shape (n_normal, n_features)
        X_anomaly: Anomaly samples of shape (n_anomaly, n_features)
        bins: Number of histogram bins
        alpha: Transparency of histograms
        density: If True, normalize to probability density
        figsize: Figure size (width, height)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save the figure
        show: If True, display the plot

    Returns:
        Figure object if matplotlib is available, None otherwise
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return None

    scores_normal = detector.score(X_normal)
    scores_anomaly = detector.score(X_anomaly)

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        scores_normal,
        bins=bins,
        alpha=alpha,
        label=f'Normal (n={len(X_normal)})',
        density=density,
        color='blue'
    )
    ax.hist(
        scores_anomaly,
        bins=bins,
        alpha=alpha,
        label=f'Anomaly (n={len(X_anomaly)})',
        density=density,
        color='red'
    )

    if detector.threshold is not None:
        ax.axvline(
            detector.threshold,
            color='green',
            linestyle='--',
            linewidth=2,
            label=f'Threshold ({detector.threshold:.2f})'
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches='tight'
        )

    if show:
        plt.show()

    return fig


def plot_2d_energy_landscape(
    energy_fn,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    resolution: int = 100,
    data: Optional[np.ndarray] = None,
    data_color: str = 'red',
    data_size: int = 1,
    data_alpha: float = 0.5,
    cmap: str = 'viridis',
    n_levels: int = 50,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Energy Landscape",
    colorbar_label: str = "Energy",
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[Any]:
    """
    Plot 2D energy landscape as a heatmap with optional data overlay.

    Only works for 2D input energy functions.

    Args:
        energy_fn: Energy function (EnergyMLP or callable)
        xlim: X-axis limits (min, max)
        ylim: Y-axis limits (min, max)
        resolution: Number of points along each axis
        data: Optional 2D data points to overlay (n_samples, 2)
        data_color: Color for data points
        data_size: Size of data points
        data_alpha: Transparency of data points
        cmap: Colormap for energy contours
        n_levels: Number of contour levels
        figsize: Figure size
        title: Plot title
        colorbar_label: Label for colorbar
        save_path: Optional path to save figure
        show: If True, display the plot

    Returns:
        Figure object if matplotlib is available, None otherwise
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return None

    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx.flatten(), yy.flatten()], axis=1)

    if hasattr(energy_fn, '__call__'):
        energy = energy_fn(Tensor(grid, requires_grad=False))
        if hasattr(energy, 'data'):
            energy = energy.data
    else:
        energy = energy_fn(grid)

    energy = energy.reshape(resolution, resolution)

    fig, ax = plt.subplots(figsize=figsize)

    contour = ax.contourf(
        xx,
        yy,
        energy,
        levels=n_levels,
        cmap=cmap
    )
    plt.colorbar(
        contour,
        ax=ax,
        label=colorbar_label
    )

    if data is not None:
        ax.scatter(
            data[:, 0],
            data[:, 1],
            c=data_color,
            s=data_size,
            alpha=data_alpha,
            label='Data'
        )
        ax.legend()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches='tight'
        )

    if show:
        plt.show()

    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 8),
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[Any]:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Optional AUC value to display in legend
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
        show: If True, display the plot

    Returns:
        Figure object if matplotlib is available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    label = 'ROC Curve'
    if auc is not None:
        label += f' (AUC = {auc:.4f})'

    ax.plot(
        fpr,
        tpr,
        'b-',
        linewidth=2,
        label=label
    )
    ax.plot(
        [0, 1],
        [0, 1],
        'k--',
        linewidth=1,
        label='Random (AUC = 0.5)'
    )

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches='tight'
        )

    if show:
        plt.show()

    return fig


def plot_precision_recall_curve(
    recall: np.ndarray,
    precision: np.ndarray,
    auc: Optional[float] = None,
    baseline: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 8),
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[Any]:
    """
    Plot Precision-Recall curve.

    Args:
        recall: Recall values
        precision: Precision values
        auc: Optional AUPRC value to display in legend
        baseline: Optional baseline precision (e.g., positive class ratio)
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
        show: If True, display the plot

    Returns:
        Figure object if matplotlib is available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    label = 'PR Curve'
    if auc is not None:
        label += f' (AUPRC = {auc:.4f})'

    ax.plot(
        recall,
        precision,
        'b-',
        linewidth=2,
        label=label
    )

    if baseline is not None:
        ax.axhline(
            baseline,
            color='k',
            linestyle='--',
            linewidth=1,
            label=f'Baseline ({baseline:.4f})'
        )

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches='tight'
        )

    if show:
        plt.show()

    return fig


def plot_training_history(
    history: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Training History",
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[Any]:
    """
    Plot training history metrics over epochs.

    Args:
        history: List of dictionaries with metric values per epoch
        metrics: List of metric names to plot. If None, plots all available.
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
        show: If True, display the plot

    Returns:
        Figure object if matplotlib is available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return None

    if not history:
        print("Empty history, nothing to plot.")
        return None

    available_metrics = list(history[0].keys())
    if metrics is None:
        metrics = available_metrics
    else:
        metrics = [m for m in metrics if m in available_metrics]

    if not metrics:
        print("No matching metrics found.")
        return None

    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    epochs = range(1, len(history) + 1)

    for i, metric in enumerate(metrics):
        values = [h.get(metric, np.nan) for h in history]
        axes[i].plot(
            epochs,
            values,
            'b-',
            linewidth=2
        )
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric)
        axes[i].set_title(metric)
        axes[i].grid(True, alpha=0.3)

    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches='tight'
        )

    if show:
        plt.show()

    return fig


def plot_sample_comparison(
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    dims: Tuple[int, int] = (0, 1),
    figsize: Tuple[int, int] = (12, 5),
    title: str = "Real vs Generated Samples",
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[Any]:
    """
    Plot comparison of real and generated samples (2D projection).

    Args:
        real_samples: Real data samples
        generated_samples: Generated/sampled data
        dims: Tuple of dimension indices to plot
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
        show: If True, display the plot

    Returns:
        Figure object if matplotlib is available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return None

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize
    )

    d1, d2 = dims

    axes[0].scatter(
        real_samples[:, d1],
        real_samples[:, d2],
        c='blue',
        s=5,
        alpha=0.5
    )
    axes[0].set_xlabel(f'Dimension {d1}')
    axes[0].set_ylabel(f'Dimension {d2}')
    axes[0].set_title('Real Samples')

    axes[1].scatter(
        generated_samples[:, d1],
        generated_samples[:, d2],
        c='red',
        s=5,
        alpha=0.5
    )
    axes[1].set_xlabel(f'Dimension {d1}')
    axes[1].set_ylabel(f'Dimension {d2}')
    axes[1].set_title('Generated Samples')

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches='tight'
        )

    if show:
        plt.show()

    return fig


def plot_score_distribution(
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    bins: int = 50,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Score Distribution",
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[Any]:
    """
    Plot distribution of anomaly scores.

    Args:
        scores: Anomaly scores
        labels: Optional true labels (0=normal, 1=anomaly)
        threshold: Optional decision threshold to display
        bins: Number of histogram bins
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
        show: If True, display the plot

    Returns:
        Figure object if matplotlib is available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        normal_mask = (labels == 0)
        anomaly_mask = (labels == 1)

        if normal_mask.any():
            ax.hist(
                scores[normal_mask],
                bins=bins,
                alpha=0.5,
                density=True,
                label='Normal',
                color='blue'
            )
        if anomaly_mask.any():
            ax.hist(
                scores[anomaly_mask],
                bins=bins,
                alpha=0.5,
                density=True,
                label='Anomaly',
                color='red'
            )
    else:
        ax.hist(
            scores,
            bins=bins,
            alpha=0.7,
            density=True,
            color='blue'
        )

    if threshold is not None:
        ax.axvline(
            threshold,
            color='green',
            linestyle='--',
            linewidth=2,
            label=f'Threshold ({threshold:.2f})'
        )

    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches='tight'
        )

    if show:
        plt.show()

    return fig


def create_evaluation_figure(
    roc_data: Dict[str, np.ndarray],
    pr_data: Dict[str, np.ndarray],
    score_data: Optional[Dict[str, Any]] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[Any]:
    """
    Create a comprehensive evaluation figure with multiple subplots.

    Args:
        roc_data: Dict with 'fpr', 'tpr', 'auc' keys
        pr_data: Dict with 'recall', 'precision', 'auc' keys
        score_data: Optional dict with 'scores', 'labels', 'threshold' keys
        figsize: Figure size
        save_path: Optional path to save figure
        show: If True, display the plot

    Returns:
        Figure object if matplotlib is available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return None

    n_plots = 3 if score_data is not None else 2
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=figsize
    )

    axes[0].plot(
        roc_data['fpr'],
        roc_data['tpr'],
        'b-',
        linewidth=2,
        label=f"AUC = {roc_data['auc']:.4f}"
    )
    axes[0].plot(
        [0, 1],
        [0, 1],
        'k--',
        linewidth=1
    )
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        pr_data['recall'],
        pr_data['precision'],
        'b-',
        linewidth=2,
        label=f"AUPRC = {pr_data['auc']:.4f}"
    )
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    if score_data is not None:
        scores = score_data['scores']
        labels = score_data.get('labels')
        threshold = score_data.get('threshold')

        if labels is not None:
            normal_mask = (labels == 0)
            anomaly_mask = (labels == 1)

            if normal_mask.any():
                axes[2].hist(
                    scores[normal_mask],
                    bins=50,
                    alpha=0.5,
                    density=True,
                    label='Normal',
                    color='blue'
                )
            if anomaly_mask.any():
                axes[2].hist(
                    scores[anomaly_mask],
                    bins=50,
                    alpha=0.5,
                    density=True,
                    label='Anomaly',
                    color='red'
                )
        else:
            axes[2].hist(
                scores,
                bins=50,
                alpha=0.7,
                density=True
            )

        if threshold is not None:
            axes[2].axvline(
                threshold,
                color='green',
                linestyle='--',
                linewidth=2,
                label='Threshold'
            )

        axes[2].set_xlabel('Anomaly Score')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Score Distribution')
        axes[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches='tight'
        )

    if show:
        plt.show()

    return fig
