import numpy as np
import pytest
from src.perceptron import Perceptron


def test_perceptron_learns_simple_linearly_separable_data():
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ]
    )
    y = np.array([-1, -1, -1, 1, 1, 1])

    model = Perceptron(learning_rate=0.1, epochs=100, random_state=7)
    model.fit(X, y)

    assert np.array_equal(model.predict(X), y)
    assert len(model.errors_) >= 1
    assert model.errors_[-1] == 0


def test_perceptron_is_reproducible_with_same_random_state():
    X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([-1, -1, 1, 1])

    first = Perceptron(random_state=123).fit(X, y)
    second = Perceptron(random_state=123).fit(X, y)

    assert np.allclose(first.weights_, second.weights_)
    assert first.bias_ == pytest.approx(second.bias_)
    assert first.errors_ == second.errors_


def test_bias_can_be_disabled():
    X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([-1, -1, 1, 1])
    model = Perceptron(use_bias=False, random_state=1).fit(X, y)
    assert model.bias_ == 0.0


def test_rejects_labels_other_than_minus_one_and_plus_one():
    model = Perceptron()
    with pytest.raises(ValueError, match=r"-1 and \+1"):
        model.fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0, 1]))


def test_rejects_input_with_wrong_number_of_features():
    model = Perceptron()
    with pytest.raises(ValueError, match="exactly two features"):
        model.fit(np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]]), np.array([-1, 1]))


def test_adaline_trains_and_predicts_binary_labels():
    from src.adaline import Adaline

    X = np.array(
        [[-2.0, -1.0], [-1.0, -2.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 2.0], [2.0, 1.0]]
    )
    y = np.array([-1, -1, -1, 1, 1, 1])
    model = Adaline(
        learning_rate=0.05, epochs=100, mse_threshold=0.001, random_state=3
    ).fit(X, y)
    assert set(model.predict(X)).issubset({-1, 1})
    assert len(model.mse_history_) >= 1
    assert np.isfinite(model.final_mse_)


def test_adaline_bias_can_be_disabled():
    from src.adaline import Adaline

    X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([-1, -1, 1, 1])
    model = Adaline(use_bias=False, random_state=2).fit(X, y)
    assert model.bias_ == 0.0


def test_manual_evaluation_functions():
    from src.evaluation import accuracy_manual, confusion_matrix_manual

    y_true = np.array([-1, -1, 1, 1, 1])
    y_pred = np.array([-1, 1, 1, -1, 1])
    assert accuracy_manual(y_true, y_pred) == pytest.approx(0.6)
    assert np.array_equal(
        confusion_matrix_manual(y_true, y_pred), np.array([[1, 1], [1, 2]])
    )
