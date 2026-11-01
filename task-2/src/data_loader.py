"""Load and validate the Penguins dataset without dropping rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.encoding import (
    CATEGORICAL_FEATURE,
    CLASS_NAMES,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
)

EXPECTED_COLUMNS = ["Species", *FEATURE_NAMES]


def load_penguins(csv_path: str | Path) -> pd.DataFrame:
    """Load the CSV file, validate its schema, and preserve all 150 rows.

    Missing numeric values remain ``NaN`` here. They are imputed later using
    statistics learned from the training split only.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path, na_values=["NA", "N/A", ""])
    missing_columns = set(EXPECTED_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    data = data[EXPECTED_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data[CATEGORICAL_FEATURE] = data[CATEGORICAL_FEATURE].astype("string").str.strip()
    data.loc[data[CATEGORICAL_FEATURE] == "", CATEGORICAL_FEATURE] = pd.NA

    if data["Species"].isna().any():
        raise ValueError("Species labels must not be missing.")
    unknown_classes = set(data["Species"].unique()) - set(CLASS_NAMES)
    if unknown_classes:
        raise ValueError(f"Unexpected class labels: {sorted(unknown_classes)}")

    return data
