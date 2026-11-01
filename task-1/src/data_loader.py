"""Load and validate the Penguins dataset."""

from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "Species",
    "CulmenLength",
    "CulmenDepth",
    "FlipperLength",
    "OriginLocation",
    "BodyMass",
]
NUMERIC_FEATURES = [
    "CulmenLength",
    "CulmenDepth",
    "FlipperLength",
    "BodyMass",
]


def load_penguins(csv_path: str | Path) -> pd.DataFrame:
    """Load the dataset and validate its schema without dropping any rows."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path, na_values=["NA", "N/A", ""])
    missing_columns = set(EXPECTED_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    # Keep only the columns specified by the assignment, preserving every row.
    data = data[EXPECTED_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data["Species"].isna().any():
        raise ValueError("Species labels must not be missing.")

    return data
