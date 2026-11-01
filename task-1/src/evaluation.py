"""Evaluation helpers implemented manually without sklearn metrics."""

import numpy as np


def confusion_matrix_manual(y_true, y_pred, labels=(-1, 1)) -> np.ndarray:
    """Return a confusion matrix with rows=true labels and columns=predictions."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1 or len(true) != len(pred):
        raise ValueError("y_true and y_pred must be 1D arrays of equal length.")
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_index = {label: index for index, label in enumerate(labels)}
    for actual, guessed in zip(true, pred):
        if actual not in label_to_index or guessed not in label_to_index:
            raise ValueError("Found a label that is not included in labels.")
        matrix[label_to_index[actual], label_to_index[guessed]] += 1
    return matrix


def accuracy_manual(y_true, y_pred) -> float:
    """Calculate accuracy as correct predictions / all predictions."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1 or len(true) != len(pred):
        raise ValueError("y_true and y_pred must be 1D arrays of equal length.")
    if len(true) == 0:
        raise ValueError("Accuracy is undefined for an empty set.")
    return float(np.sum(true == pred) / len(true))
