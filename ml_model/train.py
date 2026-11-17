
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# 1. Define project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "winequality-white.csv"
MODEL_PATH = PROJECT_ROOT / "ml_model" / "model.pkl"


# 2. Load and prepare dataset

def load_data():
    print("\n=== LOADING DATASET ===")

    df = pd.read_csv(DATA_PATH, sep=";")

    print(f"Original samples: {len(df)}")

    # Remove exact duplicate observations.
    df = df.drop_duplicates().copy()

    print(f"Samples after duplicate removal: {len(df)}")

    # Binary target:
    # 1 = good quality (score >= 6)
    # 0 = lower quality (score < 6)
    df["quality_label"] = (df["quality"] >= 6).astype(int)

    # Remove the target and original quality score
    # from the input features to prevent leakage.
    X = df.drop(columns=["quality", "quality_label"])
    y = df["quality_label"]

    print(f"Number of features: {X.shape[1]}")

    conflicting = (
        df.groupby(list(X.columns))["quality_label"]
        .nunique()
        .gt(1)
        .sum()
    )
    print(f"Feature groups with conflicting labels: {conflicting}")

    return X, y


# 3. Train the model

def train_model(X, y):
    print("\n=== TRAINING MODEL ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Training completed successfully!")

    return model, X_test, y_test


# 4. Evaluate the model

def evaluate_model(model, X_test, y_test):
    print("\n=== MODEL EVALUATION ===")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Lower quality", "Good quality"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# 5. Save the trained model

def save_model(model):
    print("\n=== SAVING MODEL ===")

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")


# 6. Main

def main():
    X, y = load_data()

    model, X_test, y_test = train_model(X, y)

    evaluate_model(model, X_test, y_test)

    save_model(model)

    print("\n=== ALL STEPS COMPLETED ===")


if __name__ == "__main__":
    main()