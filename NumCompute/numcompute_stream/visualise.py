"""matplotlib helpers for streaming demo and benchmark plots"""

import matplotlib.pyplot as plt
import numpy as np

from numcompute.utils import validate_metrics_array


def plot_metric_over_time(metric_values, title, ylabel, save_path=None):
    """Plot one metric across chunks (e.g. accuracy per chunk)

    metric_values - one value per chunk, list or 1d array
    """
    values = np.asarray(metric_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("metric_values must be a 1d sequence")

    chunks = np.arange(values.size)

    fig, ax = plt.subplots()
    ax.plot(chunks, values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    _show_or_save(fig, save_path)


def compare_models(metric1, metric2, labels, save_path=None):
    """Plot two models' per-chunk metrics on the same axes

    metric1, metric2 - one value per chunk each, same length
    labels - legend names, e.g. ['Tree', 'Bagging']

    Raises ValueError if the two metric lists are different lengths
    """
    values1 = np.asarray(metric1, dtype=float)
    values2 = np.asarray(metric2, dtype=float)

    if values1.ndim != 1 or values2.ndim != 1:
        raise ValueError("metric1 and metric2 must be 1d sequences")
    if values1.size != values2.size:
        raise ValueError("metric1 and metric2 must have the same length")
    if len(labels) != 2:
        raise ValueError("labels must have exactly two entries")

    chunks = np.arange(values1.size)

    fig, ax = plt.subplots()
    ax.plot(chunks, values1, marker="o", label=labels[0])
    ax.plot(chunks, values2, marker="o", label=labels[1])
    ax.set_title("Model comparison")
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _show_or_save(fig, save_path)


def plot_predictions_vs_ground_truth(y_true, y_pred, save_path=None):
    """Scatter plot of predictions against ground truth for one chunk

    Raises ValueError if arrays are empty, wrong shape, or incompatible
    """
    y_true, y_pred = validate_metrics_array(y_true, y_pred)

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.7)
    ax.set_title("Predictions vs ground truth")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predictions")
    ax.grid(True, alpha=0.3)

    # step 1: draw a diagonal so perfect predictions line up
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)

    _show_or_save(fig, save_path)


def _show_or_save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)
