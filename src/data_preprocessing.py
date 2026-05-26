"""
============================================================
STEP 1 — DATA LOADING & PREPROCESSING
============================================================
Module: data_preprocessing.py

Responsibilities
----------------
1a. Load Fake.csv and True.csv with pandas.
1b. Add a binary label column  (Fake = 1, True = 0).
1c. Merge both DataFrames into a single DataFrame `df`.
1d. Shuffle the merged DataFrame with a fixed random_state = 42.
1e. Handle missing values.
1f. Parse the `date` column into datetime (errors='coerce').
1g. Create a `combined_text` column = title + " [SEP] " + text.
1h. Print dataset statistics.
1i. Save the cleaned DataFrame to data/cleaned_data.csv.

The module is import-safe (everything lives in functions) and can also
be run directly:  `python src/data_preprocessing.py`
"""

from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Reproducibility (Optimization O1)
# ------------------------------------------------------------------ #
RANDOM_STATE: int = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ------------------------------------------------------------------ #
# Path configuration — relative to the project root
# ------------------------------------------------------------------ #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")
CLEANED_CSV = os.path.join(DATA_DIR, "cleaned_data.csv")


def _safe_read_csv(path: str) -> pd.DataFrame:
    """Read a CSV file with robust error handling (Optimization O4).

    Args:
        path: Absolute path to the CSV file.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: For any other read failure.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected dataset file not found: {path}\n"
            f"Place Fake.csv and True.csv inside the data/ directory."
        )
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - we want to surface any read error
        raise RuntimeError(f"Failed to read CSV '{path}': {exc}") from exc


def load_raw_data(fake_path: str = FAKE_CSV,
                  true_path: str = TRUE_CSV) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Fake and True news CSV files (Step 1a).

    Args:
        fake_path: Path to Fake.csv.
        true_path: Path to True.csv.

    Returns:
        A tuple ``(fake_df, true_df)`` of raw DataFrames.
    """
    fake_df = _safe_read_csv(fake_path)
    true_df = _safe_read_csv(true_path)
    return fake_df, true_df


def add_labels(fake_df: pd.DataFrame,
               true_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add the binary `label` column (Step 1b): Fake = 1, True = 0.

    Args:
        fake_df: DataFrame of fake articles.
        true_df: DataFrame of true articles.

    Returns:
        The two DataFrames, each with a new ``label`` column.
    """
    fake_df = fake_df.copy()
    true_df = true_df.copy()
    fake_df["label"] = 1
    true_df["label"] = 0
    return fake_df, true_df


def merge_and_shuffle(fake_df: pd.DataFrame,
                      true_df: pd.DataFrame) -> pd.DataFrame:
    """Merge then shuffle the two DataFrames (Steps 1c & 1d).

    Args:
        fake_df: Labelled fake-news DataFrame.
        true_df: Labelled true-news DataFrame.

    Returns:
        A single shuffled DataFrame with a reset index.
    """
    df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle NaN values in `title` / `text` (Step 1e).

    - Drop rows where BOTH title AND text are NaN.
    - Fill remaining NaN titles with "".
    - Fill remaining NaN texts with "".

    Args:
        df: The merged DataFrame.

    Returns:
        The DataFrame with missing values handled.
    """
    df = df.copy()
    # Ensure the columns exist even if a malformed CSV is supplied.
    for col in ("title", "text"):
        if col not in df.columns:
            df[col] = ""

    both_nan = df["title"].isna() & df["text"].isna()
    df = df[~both_nan].reset_index(drop=True)

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the `date` column to datetime (Step 1f).

    Multiple/odd date formats are coerced to NaT rather than raising.

    Args:
        df: The DataFrame containing a ``date`` column.

    Returns:
        The DataFrame with `date` parsed to ``datetime64[ns]``.
    """
    df = df.copy()
    if "date" not in df.columns:
        df["date"] = pd.NaT
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create the BERT-friendly `combined_text` column (Step 1g).

    combined_text = title + " [SEP] " + text

    Args:
        df: DataFrame with `title` and `text` columns.

    Returns:
        The DataFrame with an added ``combined_text`` column.
    """
    df = df.copy()
    df["combined_text"] = (
        df["title"].astype(str) + " [SEP] " + df["text"].astype(str)
    )
    return df


def print_statistics(df: pd.DataFrame) -> None:
    """Print dataset statistics (Step 1h).

    Args:
        df: The cleaned DataFrame.
    """
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples           : {len(df):,}")
    print("\nClass distribution (1 = Fake, 0 = True):")
    print(df["label"].value_counts().to_string())
    print("\nClass distribution (%):")
    print((df["label"].value_counts(normalize=True) * 100).round(2).to_string())
    print("\nNull counts per column:")
    print(df.isna().sum().to_string())
    if "subject" in df.columns:
        print("\nSubject distribution:")
        print(df["subject"].value_counts().to_string())
    print("=" * 60)


def save_cleaned(df: pd.DataFrame, path: str = CLEANED_CSV) -> None:
    """Save the cleaned DataFrame to disk (Step 1i).

    Args:
        df: The cleaned DataFrame.
        path: Output CSV path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n[OK] Cleaned data saved to: {path}")


def run_preprocessing(fake_path: str = FAKE_CSV,
                      true_path: str = TRUE_CSV,
                      save: bool = True) -> pd.DataFrame:
    """Execute the full Step-1 pipeline end to end.

    Args:
        fake_path: Path to Fake.csv.
        true_path: Path to True.csv.
        save: Whether to write data/cleaned_data.csv.

    Returns:
        The fully cleaned DataFrame.
    """
    fake_df, true_df = load_raw_data(fake_path, true_path)
    fake_df, true_df = add_labels(fake_df, true_df)
    df = merge_and_shuffle(fake_df, true_df)
    df = handle_missing_values(df)
    df = parse_dates(df)
    df = build_combined_text(df)
    print_statistics(df)
    if save:
        save_cleaned(df)
    return df


if __name__ == "__main__":
    run_preprocessing()
