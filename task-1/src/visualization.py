"""Plot binary samples and a learned decision boundary."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(
    X_train,
    y_train,
    X_test,
    y_test,
    model,
    feature_names,
    class_names,
    model_name,
    save_path=None,
):
    """Plot standardized two-feature samples and the model's zero boundary."""
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, color, name in [
        (-1, "tab:blue", class_names[0]),
        (1, "tab:orange", class_names[1]),
    ]:
        train_mask = y_train == label
        test_mask = y_test == label
        ax.scatter(
            X_train[train_mask, 0],
            X_train[train_mask, 1],
            c=color,
            marker="o",
            alpha=0.72,
            label=f"{name} — train",
        )
        ax.scatter(
            X_test[test_mask, 0],
            X_test[test_mask, 1],
            c=color,
            marker="x",
            s=65,
            linewidths=1.8,
            label=f"{name} — test",
        )

    weights = model.weights_
    bias = model.bias_
    x_min = min(X_train[:, 0].min(), X_test[:, 0].min()) - 0.5
    x_max = max(X_train[:, 0].max(), X_test[:, 0].max()) + 0.5
    x_values = np.linspace(x_min, x_max, 200)
    if weights is not None and abs(weights[1]) > 1e-12:
        y_values = -(weights[0] * x_values + bias) / weights[1]
        ax.plot(x_values, y_values, "k--", linewidth=2, label="Decision boundary")
    elif weights is not None and abs(weights[0]) > 1e-12:
        ax.axvline(
            -bias / weights[0],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Decision boundary",
        )

    ax.set_xlabel(f"{feature_names[0]} (standardized)")
    ax.set_ylabel(f"{feature_names[1]} (standardized)")
    ax.set_title(f"{model_name}: {class_names[0]} vs {class_names[1]}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig
