"""Preprocessing fitted on training data only, then reused for all other data.

Design notes
------------
* Numeric missing values are replaced by the training-set mean, then the
  numeric columns are standardized with training-set mean/std statistics.
* ``OriginLocation`` is missing in no supplied row, but the code still imputes
  a missing category with the training-set mode instead of dropping the row.
* ``OriginLocation`` is nominal. The input layer is fixed at five neurons by
  the assignment, so it is represented by an ordinal code (one column) rather
  than one-hot encoded (three columns). This is a documented limitation:
  ordinal codes introduce an artificial ordering (Biscoe < Dream < Torgersen)
  that the network may interpret as a magnitude. An unseen category is mapped
  to the training mode code to keep inference defined.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.encoding import (
    CATEGORICAL_FEATURE,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
)


class PenguinPreprocessor:
    """Impute and standardize the five assignment features."""

    def __init__(
        self,
        numerical_features: Sequence[str] = NUMERIC_FEATURES,
        categorical_feature: str = CATEGORICAL_FEATURE,
    ):
        self.numerical_features = tuple(numerical_features)
        self.categorical_feature = categorical_feature
        if (
            self.numerical_features == NUMERIC_FEATURES
            and self.categorical_feature == CATEGORICAL_FEATURE
        ):
            # Keep the assignment's advertised input order:
            # CulmenLength, CulmenDepth, FlipperLength, OriginLocation, BodyMass.
            self.feature_names: tuple[str, ...] = FEATURE_NAMES
        else:
            self.feature_names = (
                *self.numerical_features,
                self.categorical_feature,
            )
        self.means_: dict[str, float] = {}
        self.scales_: dict[str, float] = {}
        self.category_mapping_: dict[str, float] = {}
        self.categories_: list[str] = []
        self.mode_: str | None = None
        self.fitted_ = False

    def fit(self, train_data: pd.DataFrame) -> "PenguinPreprocessor":
        """Learn imputation, scaling, and category codes from training data."""
        self._require_columns(train_data)

        for feature in self.numerical_features:
            values = pd.to_numeric(train_data[feature], errors="coerce").astype(float)
            mean = float(values.mean())
            if not np.isfinite(mean):
                raise ValueError(
                    f"Feature {feature!r} has no usable training values for imputation."
                )
            scale = float(values.std(ddof=0))
            self.means_[feature] = mean
            self.scales_[feature] = scale if scale > 1e-12 else 1.0

        raw_categories = self._clean_categories(train_data[self.categorical_feature])
        non_missing = raw_categories.dropna()
        if non_missing.empty:
            raise ValueError("OriginLocation has no usable training values.")
        self.categories_ = sorted(str(value) for value in non_missing.unique())
        self.category_mapping_ = {
            category: float(code) for code, category in enumerate(self.categories_)
        }
        self.mode_ = str(non_missing.mode().iloc[0])
        self.fitted_ = True
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Apply the fitted imputation, scaling, and encoding to ``data``."""
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")
        self._require_columns(data)

        columns_by_feature: dict[str, np.ndarray] = {}
        for feature in self.numerical_features:
            values = pd.to_numeric(data[feature], errors="coerce").astype(float)
            values = values.fillna(self.means_[feature])
            standardized = (values - self.means_[feature]) / self.scales_[feature]
            columns_by_feature[feature] = standardized.to_numpy(dtype=float)

        raw_categories = self._clean_categories(data[self.categorical_feature])
        assert self.mode_ is not None
        raw_categories = raw_categories.fillna(self.mode_)
        fallback_code = self.category_mapping_[self.mode_]
        codes = raw_categories.map(self.category_mapping_).fillna(fallback_code)
        columns_by_feature[self.categorical_feature] = codes.to_numpy(dtype=float)

        transformed = np.column_stack(
            [columns_by_feature[feature] for feature in self.feature_names]
        )
        if not np.isfinite(transformed).all():
            raise RuntimeError("Preprocessing produced non-finite feature values.")
        return transformed

    def fit_transform(self, train_data: pd.DataFrame) -> np.ndarray:
        """Fit on ``train_data`` and return its transformed matrix."""
        return self.fit(train_data).transform(train_data)

    @staticmethod
    def _clean_categories(series: pd.Series) -> pd.Series:
        """Normalize category strings while preserving missing values."""
        cleaned = series.astype("string").str.strip()
        return cleaned.replace({"": pd.NA})

    def _require_columns(self, data: pd.DataFrame) -> None:
        missing = set(self.feature_names) - set(data.columns)
        if missing:
            raise ValueError(f"Data is missing feature columns: {sorted(missing)}")
