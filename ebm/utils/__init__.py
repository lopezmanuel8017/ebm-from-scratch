"""
Utility functions for Energy-Based Models.

This module provides:
- Visualization utilities (energy histograms, landscapes, curves)
"""

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

__all__ = [
    "plot_energy_histogram",
    "plot_2d_energy_landscape",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_training_history",
    "plot_sample_comparison",
    "plot_score_distribution",
    "create_evaluation_figure",
]
