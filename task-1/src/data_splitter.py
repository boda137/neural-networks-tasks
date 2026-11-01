"""Reproducible, per-class train/test split for binary classification."""

import pandas as pd


def split_two_classes(
    data: pd.DataFrame,
    class_1: str,
    class_2: str,
    train_per_class: int = 30,
    test_per_class: int = 20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return disjoint train/test sets with the requested count per class."""
    if class_1 == class_2:
        raise ValueError("Choose two different classes.")

    selected = data[data["Species"].isin([class_1, class_2])].copy()
    if set(selected["Species"].unique()) != {class_1, class_2}:
        raise ValueError("One or both selected classes are not present in the dataset.")

    train_parts, test_parts = [], []
    for label in (class_1, class_2):
        class_data = selected[selected["Species"] == label]
        required = train_per_class + test_per_class
        if len(class_data) < required:
            raise ValueError(
                f"Class '{label}' has {len(class_data)} samples; {required} are required."
            )

        shuffled = class_data.sample(frac=1, random_state=random_state)
        train_parts.append(shuffled.iloc[:train_per_class])
        test_parts.append(shuffled.iloc[train_per_class:required])

    train = pd.concat(train_parts).sample(frac=1, random_state=random_state)
    test = pd.concat(test_parts).sample(frac=1, random_state=random_state)

    # Original row indices are preserved until this check is complete.
    if not set(train.index).isdisjoint(set(test.index)):
        raise RuntimeError("Data leakage detected: a row appears in both splits.")

    return train.reset_index(drop=True), test.reset_index(drop=True)
