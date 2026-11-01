"""Fit preprocessing on training data and reuse it for test data."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeaturePreprocessor:
    """Impute missing values and standardize two selected features.

    OriginLocation is encoded using a stable, documented mapping. Numeric
    features are standardized because ADALINE is sensitive to feature scales.
    Statistics are learned from training data only.
    """

    selected_features: tuple[str, str]
    means_: dict[str, float] = field(default_factory=dict)
    scales_: dict[str, float] = field(default_factory=dict)
    origin_mapping_: dict[str, float] = field(
        default_factory=lambda: {"Torgersen": 0.0, "Biscoe": 1.0, "Dream": 2.0}
    )
    fitted_: bool = False

    def fit(self, train_data: pd.DataFrame) -> "FeaturePreprocessor":
        for feature in self.selected_features:
            if feature == "OriginLocation":
                values = train_data[feature].map(self.origin_mapping_).dropna()
                if values.empty:
                    raise ValueError("No valid OriginLocation values in training data.")
                self.means_[feature] = float(values.mean())
                self.scales_[feature] = float(values.std(ddof=0)) or 1.0
            else:
                values = pd.to_numeric(train_data[feature], errors="coerce")
                mean = float(values.mean())
                if pd.isna(mean):
                    raise ValueError(
                        f"Feature '{feature}' has no usable training values."
                    )
                scale = float(values.std(ddof=0))
                self.means_[feature] = mean
                self.scales_[feature] = scale if scale > 0 else 1.0

        self.fitted_ = True
        return self

    def transform(self, data: pd.DataFrame):
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")

        columns = []
        for feature in self.selected_features:
            if feature == "OriginLocation":
                values = data[feature].map(self.origin_mapping_).astype(float)
            else:
                values = pd.to_numeric(data[feature], errors="coerce")

            values = values.fillna(self.means_[feature])
            standardized = (values - self.means_[feature]) / self.scales_[feature]
            columns.append(standardized.to_numpy(dtype=float))

        # Shape: (number of samples, 2)
        return np.column_stack(columns)
