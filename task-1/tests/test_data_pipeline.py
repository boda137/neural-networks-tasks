import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_penguins
from src.data_splitter import split_two_classes
from src.preprocessing import FeaturePreprocessor

ROOT = Path(__file__).resolve().parents[1]


def test_loader_keeps_all_rows():
    data = load_penguins(ROOT / "data" / "penguins.csv")
    assert len(data) == 150
    assert set(data["Species"]) == {"Adelie", "Chinstrap", "Gentoo"}


def test_split_has_expected_counts_and_no_shared_rows():
    data = load_penguins(ROOT / "data" / "penguins.csv")
    train, test = split_two_classes(data, "Adelie", "Gentoo", random_state=42)
    assert len(train) == 60
    assert len(test) == 40
    assert train["Species"].value_counts().to_dict() == {"Adelie": 30, "Gentoo": 30}
    assert test["Species"].value_counts().to_dict() == {"Adelie": 20, "Gentoo": 20}


def test_preprocessor_returns_two_finite_features():
    data = load_penguins(ROOT / "data" / "penguins.csv")
    train, test = split_two_classes(data, "Adelie", "Gentoo", random_state=42)
    prep = FeaturePreprocessor(("CulmenLength", "BodyMass")).fit(train)
    x_train = prep.transform(train)
    x_test = prep.transform(test)
    assert x_train.shape == (60, 2)
    assert x_test.shape == (40, 2)
    assert pd.notna(x_train).all()
    assert pd.notna(x_test).all()


def test_origin_location_is_preprocessed():
    data = load_penguins(ROOT / "data" / "penguins.csv")
    train, test = split_two_classes(data, "Adelie", "Gentoo", random_state=42)
    prep = FeaturePreprocessor(("OriginLocation", "FlipperLength")).fit(train)
    assert prep.transform(train).shape == (60, 2)
    assert prep.transform(test).shape == (40, 2)
