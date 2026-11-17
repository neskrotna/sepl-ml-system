
from pathlib import Path

import joblib
import pandas as pd

# Path to the trained model
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


# Feature names must match the training dataset
FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def load_model():
    """Load the previously trained Random Forest model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def preprocess_input(data: dict) -> pd.DataFrame:
    """Validate and prepare input features for prediction."""

    missing = [
        feature
        for feature in FEATURE_NAMES
        if feature not in data
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    unexpected = [
        feature
        for feature in data
        if feature not in FEATURE_NAMES
    ]

    if unexpected:
        raise ValueError(
            f"Unexpected features: {unexpected}"
        )

    try:
        processed = {
            feature: float(data[feature])
            for feature in FEATURE_NAMES
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "All feature values must be numeric."
        ) from exc

    df = pd.DataFrame([processed])

    if not pd.notna(df.to_numpy()).all():
        raise ValueError(
            "Feature values must not be NaN or missing."
        )

    import numpy as np

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(
            "Feature values must be finite."
        )

    return df


def predict_wine_quality(data: dict) -> dict:
    """Predict wine quality using the saved model."""

    model = load_model()

    features = preprocess_input(data)

    prediction = int(model.predict(features)[0])

    probabilities = model.predict_proba(features)[0]

    class_index = list(model.classes_).index(prediction)
    confidence = float(probabilities[class_index])

    quality_label = (
        "good_quality"
        if prediction == 1
        else "lower_quality"
    )

    return {
        "prediction": quality_label,
        "class_id": prediction,
        "confidence": round(confidence, 4),
    }