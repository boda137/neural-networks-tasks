import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations import (
    get_activation,
    normalize_activation_name,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
)


def test_sigmoid_is_inside_open_unit_interval():
    values = sigmoid(np.array([-50.0, -1.0, 0.0, 1.0, 50.0]))
    assert np.isfinite(values).all()
    assert np.all(values > 0.0)
    assert np.all(values < 1.0)


def test_tanh_is_bounded():
    values = tanh(np.array([-50.0, -1.0, 0.0, 1.0, 50.0]))
    assert np.isfinite(values).all()
    assert np.all(values >= -1.0)
    assert np.all(values <= 1.0)
    assert values[2] == pytest.approx(0.0)
    assert np.all(values[[1, 3]] > -1.0)
    assert np.all(values[[1, 3]] < 1.0)


def test_sigmoid_derivative_matches_numerical_derivative():
    x = np.linspace(-3.0, 3.0, 13)
    epsilon = 1e-6
    numerical = (sigmoid(x + epsilon) - sigmoid(x - epsilon)) / (2 * epsilon)
    np.testing.assert_allclose(
        sigmoid_derivative(x), numerical, rtol=1e-5, atol=1e-8
    )


def test_tanh_derivative_matches_numerical_derivative():
    x = np.linspace(-3.0, 3.0, 13)
    epsilon = 1e-6
    numerical = (tanh(x + epsilon) - tanh(x - epsilon)) / (2 * epsilon)
    np.testing.assert_allclose(
        tanh_derivative(x), numerical, rtol=1e-5, atol=1e-8
    )


def test_extreme_inputs_do_not_overflow():
    extreme = np.array([-1e6, -1000.0, 0.0, 1000.0, 1e6])
    assert np.isfinite(sigmoid(extreme)).all()
    assert np.isfinite(sigmoid_derivative(extreme)).all()
    assert np.isfinite(tanh(extreme)).all()
    assert np.isfinite(tanh_derivative(extreme)).all()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Sigmoid", "sigmoid"),
        ("sigmoid", "sigmoid"),
        ("Tanh", "tanh"),
        ("hyperbolic tangent", "tanh"),
    ],
)
def test_activation_name_normalization(name, expected):
    assert normalize_activation_name(name) == expected
    canonical, function, derivative = get_activation(name)
    assert canonical == expected
    assert callable(function)
    assert callable(derivative)


def test_unknown_activation_is_rejected():
    with pytest.raises(ValueError):
        get_activation("relu")
