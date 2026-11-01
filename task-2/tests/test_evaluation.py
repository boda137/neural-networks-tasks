import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.encoding import CLASS_NAMES
from src.evaluation import (
    accuracy_manual,
    confusion_matrix_manual,
    evaluate_predictions,
)


def test_confusion_matrix_shape_and_order():
    y_true = ["Adelie", "Chinstrap", "Gentoo", "Adelie", "Chinstrap", "Gentoo"]
    y_pred = ["Adelie", "Gentoo", "Gentoo", "Adelie", "Chinstrap", "Chinstrap"]
    matrix = confusion_matrix_manual(y_true, y_pred)
    expected = np.array(
        [
            [2, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ]
    )
    assert matrix.shape == (3, 3)
    np.testing.assert_array_equal(matrix, expected)


def test_accuracy_is_correct_predictions_over_total():
    y_true = ["Adelie", "Chinstrap", "Gentoo", "Adelie"]
    y_pred = ["Adelie", "Chinstrap", "Adelie", "Adelie"]
    assert accuracy_manual(y_true, y_pred) == pytest.approx(0.75)


def test_missing_predicted_class_still_has_a_column():
    y_true = ["Adelie", "Chinstrap", "Gentoo"]
    y_pred = ["Adelie", "Adelie", "Adelie"]
    matrix = confusion_matrix_manual(y_true, y_pred)
    np.testing.assert_array_equal(
        matrix,
        np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
            ]
        ),
    )
    assert matrix.shape == (3, 3)


def test_evaluate_predictions_returns_counts_and_matrix():
    y_true = ["Adelie", "Chinstrap", "Gentoo", "Adelie"]
    y_pred = ["Adelie", "Chinstrap", "Adelie", "Adelie"]
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["correct"] == 3
    assert metrics["total"] == 4
    assert metrics["accuracy"] == pytest.approx(0.75)
    np.testing.assert_array_equal(
        metrics["confusion_matrix"],
        np.array(
            [
                [2, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
            ]
        ),
    )


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        confusion_matrix_manual(["Adelie"], ["Adelie", "Gentoo"])
    with pytest.raises(ValueError):
        confusion_matrix_manual(["Unknown"], ["Adelie"])
    with pytest.raises(ValueError):
        accuracy_manual([], [])
    assert len(CLASS_NAMES) == 3
