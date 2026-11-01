"""Class ordering, label encoding, and one-hot encoding helpers.

The class order is a project-wide contract. Every module that converts between
class names and numeric indices imports it from here so training, prediction,
and evaluation cannot disagree about what an output neuron means.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

CLASS_NAMES: tuple[str, str, str] = ("Adelie", "Chinstrap", "Gentoo")
CLASS_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(CLASS_NAMES)
}

# The assignment fixes both the five inputs and the three outputs.
FEATURE_NAMES: tuple[str, str, str, str, str] = (
    "CulmenLength",
    "CulmenDepth",
    "FlipperLength",
    "OriginLocation",
    "BodyMass",
)
NUMERIC_FEATURES: tuple[str, str, str, str] = (
    "CulmenLength",
    "CulmenDepth",
    "FlipperLength",
    "BodyMass",
)
CATEGORICAL_FEATURE = "OriginLocation"


def encode_labels(labels: Iterable[str]) -> np.ndarray:
    """Convert class names to integer indices using ``CLASS_NAMES``."""
    encoded = []
    for label in labels:
        if label not in CLASS_TO_INDEX:
            raise ValueError(
                f"Unknown class label {label!r}. Expected one of {CLASS_NAMES}."
            )
        encoded.append(CLASS_TO_INDEX[label])
    return np.asarray(encoded, dtype=int)


def decode_labels(indices: Iterable[int]) -> list[str]:
    """Convert integer indices back to class names using ``CLASS_NAMES``."""
    decoded = []
    for index in indices:
        integer = int(index)
        if integer < 0 or integer >= len(CLASS_NAMES):
            raise ValueError(
                f"Class index {integer} is outside the valid range "
                f"0..{len(CLASS_NAMES) - 1}."
            )
        decoded.append(CLASS_NAMES[integer])
    return decoded


def one_hot_encode(
    indices: Sequence[int] | np.ndarray,
    num_classes: int = len(CLASS_NAMES),
) -> np.ndarray:
    """Return a ``(n_samples, num_classes)`` one-hot matrix."""
    values = np.asarray(indices, dtype=int).reshape(-1)
    if values.size and (values.min() < 0 or values.max() >= num_classes):
        raise ValueError(f"Class indices must lie in 0..{num_classes - 1}.")
    encoded = np.zeros((values.size, num_classes), dtype=float)
    encoded[np.arange(values.size), values] = 1.0
    return encoded
