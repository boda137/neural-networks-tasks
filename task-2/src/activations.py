"""Numerically stable activation functions and their derivatives."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ActivationFunction = Callable[[np.ndarray | float], np.ndarray]
DerivativeFunction = Callable[[np.ndarray | float], np.ndarray]


def sigmoid(net: np.ndarray | float) -> np.ndarray:
    """Stable logistic sigmoid: ``1 / (1 + exp(-net))``.

    The branch form avoids overflow because ``exp`` only ever receives a
    non-positive argument.
    """
    values = np.asarray(net, dtype=float)
    magnitude = np.exp(-np.abs(values))
    result = np.where(
        values >= 0.0,
        1.0 / (1.0 + magnitude),
        magnitude / (1.0 + magnitude),
    )
    # Floating-point rounding can reach exactly 0.0 or 1.0 for huge inputs.
    # Keep the documented open interval (0, 1) with the nearest representable
    # values; this changes nothing at ordinary net inputs.
    return np.clip(
        result,
        np.nextafter(0.0, 1.0),
        np.nextafter(1.0, 0.0),
    )


def sigmoid_derivative(net: np.ndarray | float) -> np.ndarray:
    """Derivative ``f(net) * (1 - f(net))`` evaluated from the net input."""
    output = sigmoid(net)
    return output * (1.0 - output)


def tanh(net: np.ndarray | float) -> np.ndarray:
    """Hyperbolic tangent, using NumPy's stable implementation."""
    return np.tanh(np.asarray(net, dtype=float))


def tanh_derivative(net: np.ndarray | float) -> np.ndarray:
    """Derivative ``1 - tanh(net)^2`` evaluated from the net input."""
    output = np.tanh(np.asarray(net, dtype=float))
    return 1.0 - output * output


def normalize_activation_name(name: str) -> str:
    """Return ``"sigmoid"`` or ``"tanh"`` for a user-facing activation name."""
    normalized = name.strip().lower()
    aliases = {
        "sigmoid": "sigmoid",
        "logistic": "sigmoid",
        "tanh": "tanh",
        "hyperbolic tangent": "tanh",
        "hyperbolic-tangent": "tanh",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown activation {name!r}. Choose 'Sigmoid' or 'Tanh'."
        )
    return aliases[normalized]


def get_activation(name: str) -> tuple[str, ActivationFunction, DerivativeFunction]:
    """Return ``(canonical_name, function, derivative)`` for an activation."""
    canonical = normalize_activation_name(name)
    if canonical == "sigmoid":
        return canonical, sigmoid, sigmoid_derivative
    return canonical, tanh, tanh_derivative
