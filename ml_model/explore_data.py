
from pathlib import Path

import pandas as pd

# Locate the dataset relative to this script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "winequality-white.csv"


def main():
    # Load the white wine dataset.
    df = pd.read_csv(DATA_PATH, sep=";")

    print("\n=== WHITE WINE DATASET ===")
    print(f"Number of samples: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")

    # Display the first five observations.
    print("\n=== FIRST FIVE ROWS ===")
    print(df.head())

    # Check for missing values.
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    # Check for duplicate observations.
    print("\n=== DUPLICATES ===")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # Inspect the original quality scores.
    print("\n=== ORIGINAL QUALITY DISTRIBUTION ===")
    print(df["quality"].value_counts().sort_index())

    # Create binary classification labels.
    # 1 = good quality (score >= 6)
    # 0 = lower quality (score < 6)
    df["quality_label"] = (df["quality"] >= 6).astype(int)

    print("\n=== BINARY CLASS DISTRIBUTION ===")
    print(df["quality_label"].value_counts().sort_index())

    # Display summary statistics.
    print("\n=== FEATURE STATISTICS ===")
    print(df.describe().round(2))


if __name__ == "__main__":
    main()