"""Matplotlib visualizations used by the GUI and experiment runner."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.encoding import CLASS_NAMES


def _save(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")


def plot_training_loss(
    loss_history: Sequence[float],
    title: str = "Training loss (MSE)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot mean squared error after each epoch."""
    losses = np.asarray(loss_history, dtype=float)
    if losses.size == 0:
        raise ValueError("Loss history is empty.")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(np.arange(1, losses.size + 1), losses, color="tab:blue", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared error")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.annotate(
        f"Final MSE: {losses[-1]:.6f}",
        xy=(losses.size, losses[-1]),
        xytext=(-8, 12),
        textcoords="offset points",
        ha="right",
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
    title: str = "Confusion matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Draw a readable 3x3 confusion-matrix heatmap."""
    values = np.asarray(matrix, dtype=int)
    if values.shape != (len(class_names), len(class_names)):
        raise ValueError("Confusion matrix shape does not match the class count.")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(values, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    tick_positions = np.arange(len(class_names))
    ax.set_xticks(tick_positions, labels=class_names)
    ax.set_yticks(tick_positions, labels=class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)

    threshold = values.max() / 2.0 if values.size else 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] > threshold else "black"
            ax.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color=color,
                fontsize=14,
                fontweight="bold",
            )
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    class_names: Sequence[str] = CLASS_NAMES,
    title: str = "Actual vs predicted counts",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing actual and predicted counts per class."""
    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    labels = np.arange(len(class_names))
    actual_counts = [int(np.sum(true == name)) for name in class_names]
    predicted_counts = [int(np.sum(predicted == name)) for name in class_names]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    width = 0.38
    ax.bar(labels - width / 2, actual_counts, width, label="Actual", color="tab:blue")
    ax.bar(
        labels + width / 2,
        predicted_counts,
        width,
        label="Predicted",
        color="tab:orange",
    )
    ax.set_xticks(labels, labels=class_names)
    ax.set_ylabel("Number of samples")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, save_path)
    return fig
