"""Create the professor's fixed per-class training/testing split."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.encoding import CLASS_NAMES


def split_train_test_per_class(
    data: pd.DataFrame,
    class_names: Sequence[str] = CLASS_NAMES,
    train_per_class: int = 30,
    test_per_class: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the first 30 rows per class for training and the next 20 for testing.

    Rows are selected in their original dataset order. No shuffling happens
    before the split, and no rows are discarded. The original DataFrame index
    is retained so leakage can be checked directly.
    """
    required = train_per_class + test_per_class
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for label in class_names:
        class_data = data[data["Species"] == label]
        if len(class_data) != required:
            raise ValueError(
                f"Class {label!r} has {len(class_data)} rows; exactly {required} "
                f"are required ({train_per_class} train + {test_per_class} test)."
            )
        train_parts.append(class_data.iloc[:train_per_class])
        test_parts.append(class_data.iloc[train_per_class:required])

    train = pd.concat(train_parts)
    test = pd.concat(test_parts)

    shared_indices = set(train.index).intersection(set(test.index))
    if shared_indices:
        raise RuntimeError("Data leakage detected: a row appears in both splits.")
    if len(train) != train_per_class * len(class_names):
        raise RuntimeError("Unexpected training-set size.")
    if len(test) != test_per_class * len(class_names):
        raise RuntimeError("Unexpected testing-set size.")

    return train, test
