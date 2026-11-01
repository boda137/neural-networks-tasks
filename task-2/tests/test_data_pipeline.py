import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_penguins
from src.data_splitter import split_train_test_per_class
from src.encoding import CLASS_NAMES, decode_labels, encode_labels, one_hot_encode
from src.preprocessing import PenguinPreprocessor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "penguins.csv"


@pytest.fixture(scope="module")
def data():
    return load_penguins(DATA_PATH)


@pytest.fixture(scope="module")
def split(data):
    return split_train_test_per_class(data)


def test_loader_preserves_all_rows_and_classes(data):
    assert len(data) == 150
    assert data["Species"].value_counts().to_dict() == {
        "Adelie": 50,
        "Chinstrap": 50,
        "Gentoo": 50,
    }
    assert set(data["Species"]) == set(CLASS_NAMES)


def test_split_has_exact_counts(split):
    train, test = split
    assert len(train) == 90
    assert len(test) == 60
    assert train["Species"].value_counts().to_dict() == {
        "Adelie": 30,
        "Chinstrap": 30,
        "Gentoo": 30,
    }
    assert test["Species"].value_counts().to_dict() == {
        "Adelie": 20,
        "Chinstrap": 20,
        "Gentoo": 20,
    }


def test_split_has_no_overlap(data, split):
    train, test = split
    assert set(train.index).isdisjoint(set(test.index))
    assert list(train.index.intersection(test.index)) == []
    assert sorted([*train.index, *test.index]) == sorted(data.index)


def test_split_preserves_original_order_within_each_class(data, split):
    train, test = split
    for label in CLASS_NAMES:
        original = data[data["Species"] == label]
        train_class = train[train["Species"] == label]
        test_class = test[test["Species"] == label]
        pd.testing.assert_frame_equal(
            train_class, original.iloc[:30], check_dtype=False
        )
        pd.testing.assert_frame_equal(
            test_class, original.iloc[30:50], check_dtype=False
        )


def test_preprocessing_returns_finite_five_features(split):
    train, test = split
    preprocessor = PenguinPreprocessor().fit(train)
    X_train = preprocessor.transform(train)
    X_test = preprocessor.transform(test)
    assert X_train.shape == (90, 5)
    assert X_test.shape == (60, 5)
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()
    assert preprocessor.feature_names == (
        "CulmenLength",
        "CulmenDepth",
        "FlipperLength",
        "OriginLocation",
        "BodyMass",
    )


def test_preprocessing_is_fitted_on_training_data_only(data, split):
    train, _test = split
    preprocessor = PenguinPreprocessor().fit(train)
    for feature in (
        "CulmenLength",
        "CulmenDepth",
        "FlipperLength",
        "BodyMass",
    ):
        assert preprocessor.means_[feature] == pytest.approx(
            float(train[feature].mean())
        )
        expected_scale = float(train[feature].std(ddof=0))
        assert preprocessor.scales_[feature] == pytest.approx(expected_scale)
    assert preprocessor.categories_ == sorted(
        str(value) for value in train["OriginLocation"].dropna().unique()
    )


def test_numeric_missing_values_are_imputed_with_training_mean(split):
    train, test = split
    preprocessor = PenguinPreprocessor().fit(train)
    modified = test.iloc[[0]].copy()
    modified.loc[modified.index[0], "CulmenLength"] = np.nan
    transformed = preprocessor.transform(modified)
    # A missing numeric value becomes the training mean, hence z-score 0.
    assert transformed[0, 0] == pytest.approx(0.0)


def test_missing_categorical_values_are_handled_without_dropping_rows(split):
    train, _test = split
    modified = train.copy()
    modified.loc[modified.index[:3], "OriginLocation"] = pd.NA
    preprocessor = PenguinPreprocessor().fit(modified)
    transformed = preprocessor.transform(modified)
    assert transformed.shape == (90, 5)
    assert np.isfinite(transformed).all()


def test_one_hot_encoding_uses_fixed_class_order():
    labels = encode_labels(["Adelie", "Chinstrap", "Gentoo"])
    assert labels.tolist() == [0, 1, 2]
    encoded = one_hot_encode(labels)
    np.testing.assert_array_equal(encoded, np.eye(3))
    assert decode_labels(labels) == ["Adelie", "Chinstrap", "Gentoo"]


def test_invalid_labels_are_rejected():
    with pytest.raises(ValueError):
        encode_labels(["Adelie", "Unknown"])
    with pytest.raises(ValueError):
        decode_labels([3])
    with pytest.raises(ValueError):
        one_hot_encode([0, 3])
