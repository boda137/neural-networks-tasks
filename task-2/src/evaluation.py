"""Manual confusion matrix and accuracy calculations (no sklearn metrics)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.encoding import CLASS_NAMES


def _validate_predictions(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if true.ndim != 1 or predicted.ndim != 1 or len(true) != len(predicted):
        raise ValueError("y_true and y_pred must be 1D arrays of equal length.")
    if len(true) == 0:
        raise ValueError("Evaluation requires at least one sample.")
    return true, predicted


def confusion_matrix_manual(
    y_true,
    y_pred,
    labels: Sequence[str] = CLASS_NAMES,
) -> np.ndarray:
    """Return a confusion matrix with rows=true and columns=predicted.

    ``labels`` fixes the class order. Every label in ``labels`` gets a row and
    column even if the model never predicts that class.
    """
    true, predicted = _validate_predictions(y_true, y_pred)
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, guessed in zip(true, predicted):
        if actual not in label_to_index:
            raise ValueError(f"Encountered unknown true label: {actual!r}")
        if guessed not in label_to_index:
            raise ValueError(f"Encountered unknown predicted label: {guessed!r}")
        matrix[label_to_index[actual], label_to_index[guessed]] += 1
    return matrix


def accuracy_manual(y_true, y_pred) -> float:
    """Return accuracy as a fraction: correct predictions / total samples."""
    true, predicted = _validate_predictions(y_true, y_pred)
    return float(np.sum(true == predicted) / len(true))


def evaluate_predictions(
    y_true,
    y_pred,
    labels: Sequence[str] = CLASS_NAMES,
) -> dict[str, object]:
    """Return the manual confusion matrix, accuracy, and counts."""
    matrix = confusion_matrix_manual(y_true, y_pred, labels=labels)
    accuracy = accuracy_manual(y_true, y_pred)
    correct = int(np.trace(matrix))
    return {
        "confusion_matrix": matrix,
        "accuracy": accuracy,
        "correct": correct,
        "total": int(len(np.asarray(y_true))),
    }
