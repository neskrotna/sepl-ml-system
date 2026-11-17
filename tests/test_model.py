import numpy as np
import pandas as pd
import pytest

from ml_model.model import (
    FEATURE_NAMES,
    load_model,
    predict_wine_quality,
    preprocess_input,
)

# Test data

@pytest.fixture
def valid_wine():
    """A valid white wine sample from the dataset."""
    return {
        "fixed acidity": 7.0,
        "volatile acidity": 0.27,
        "citric acid": 0.36,
        "residual sugar": 20.7,
        "chlorides": 0.045,
        "free sulfur dioxide": 45.0,
        "total sulfur dioxide": 170.0,
        "density": 1.001,
        "pH": 3.0,
        "sulphates": 0.45,
        "alcohol": 8.8,
    }


# Test 1: Model loading

def test_model_loads_successfully():
    model = load_model()

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


# Test 2: Valid preprocessing

def test_preprocess_valid_input(valid_wine):
    result = preprocess_input(valid_wine)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 11)
    assert list(result.columns) == FEATURE_NAMES


# Test 3: Missing feature

def test_preprocess_missing_feature(valid_wine):
    invalid_wine = valid_wine.copy()
    del invalid_wine["alcohol"]

    with pytest.raises(
        ValueError,
        match="Missing required features",
    ):
        preprocess_input(invalid_wine)


# Test 4: Unexpected feature

def test_preprocess_unexpected_feature(valid_wine):
    invalid_wine = valid_wine.copy()
    invalid_wine["wine_color"] = "white"

    with pytest.raises(
        ValueError,
        match="Unexpected features",
    ):
        preprocess_input(invalid_wine)

# Test 5: Nonnumeric feature

def test_preprocess_nonnumeric_feature(valid_wine):
    invalid_wine = valid_wine.copy()
    invalid_wine["alcohol"] = "not_a_number"

    with pytest.raises(
        ValueError,
        match="All feature values must be numeric",
    ):
        preprocess_input(invalid_wine)


# Test 6: NaN feature

def test_preprocess_nan_feature(valid_wine):
    invalid_wine = valid_wine.copy()
    invalid_wine["alcohol"] = np.nan

    with pytest.raises(
        ValueError,
        match="NaN or missing",
    ):
        preprocess_input(invalid_wine)


# Test 7: Infinite feature

def test_preprocess_infinite_feature(valid_wine):
    invalid_wine = valid_wine.copy()
    invalid_wine["alcohol"] = np.inf

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        preprocess_input(invalid_wine)


# Test 8: Prediction structure

def test_prediction_structure(valid_wine):
    result = predict_wine_quality(valid_wine)

    assert isinstance(result, dict)

    assert "prediction" in result
    assert "class_id" in result
    assert "confidence" in result

    assert result["prediction"] in [
        "good_quality",
        "lower_quality",
    ]

    assert result["class_id"] in [0, 1]


# Test 9: Confidence range

def test_prediction_confidence(valid_wine):
    result = predict_wine_quality(valid_wine)

    assert 0.0 <= result["confidence"] <= 1.0


# Test 10: Known sample prediction

def test_known_wine_prediction(valid_wine):
    result = predict_wine_quality(valid_wine)

    assert result["prediction"] == "good_quality"
    assert result["class_id"] == 1
    assert result["confidence"] == pytest.approx(
        0.75,
        abs=0.0001,
    )