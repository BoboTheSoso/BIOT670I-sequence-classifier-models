"""
03_balance_and_split.py

Purpose:
- Load labeled windows from 02_windows_labels.py
- Balance classes (coding vs noncoding) using undersampling
- Split into train/validation/test sets (coordinate-based split)
- Save balanced dataset + splits

Input:
- Data/processed/chr22_windows_250_labeled_all.csv

Outputs:
- Data/processed/chr22_windows_250_labeled_balanced.csv
- Data/processed/train.csv
- Data/processed/val.csv
- Data/processed/test.csv
"""

from __future__ import annotations
import os
import pandas as pd

IN_ALL = "Data/processed/chr22_windows_250_labeled_all.csv"

OUT_DIR = "Data/processed"
OUT_BAL = os.path.join(OUT_DIR, "chr22_windows_250_labeled_balanced.csv")
OUT_TRAIN = os.path.join(OUT_DIR, "train.csv")
OUT_VAL = os.path.join(OUT_DIR, "val.csv")
OUT_TEST = os.path.join(OUT_DIR, "test.csv")

# Coordinate-based splits reduce leakage when windows are nearby
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

RANDOM_STATE = 42

REQUIRED_COLS = {"window_id", "start0", "end0", "label", "sequence"}


def assert_expected_columns(df: pd.DataFrame) -> None:
    """Ensure the input CSV has the columns we expect."""
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Found: {list(df.columns)}")


def balance_by_undersampling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance classes by undersampling the majority class.

    Why?
      There are usually many more noncoding windows than coding windows.
      If unbalanced, the model can become biased to always predict noncoding.

    Strategy:
      Keep all minority class rows.
      Randomly sample the same number from the majority class.
    """
    coding = df[df["label"] == 1]
    noncoding = df[df["label"] == 0]

    if len(coding) == 0 or len(noncoding) == 0:
        raise ValueError(f"Cannot balance. coding={len(coding)} noncoding={len(noncoding)}")

    # Determine which is majority
    if len(noncoding) > len(coding):
        noncoding_sample = noncoding.sample(n=len(coding), random_state=RANDOM_STATE)
        balanced = pd.concat([coding, noncoding_sample], ignore_index=True)
    else:
        coding_sample = coding.sample(n=len(noncoding), random_state=RANDOM_STATE)
        balanced = pd.concat([coding_sample, noncoding], ignore_index=True)

    # Shuffle rows
    balanced = balanced.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return balanced


def split_by_coordinate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train/val/test using genomic coordinate order.

    Why coordinate-based split?
      If windows are near each other, they can be very similar.
      Random splitting could put highly similar windows in train and test,
      inflating performance (data leakage).

    Approach:
      Sort by start coordinate and slice into fractions.
    """
    df_sorted = df.sort_values("start0").reset_index(drop=True)
    n = len(df_sorted)

    n_train = int(TRAIN_FRAC * n)
    n_val_end = int((TRAIN_FRAC + VAL_FRAC) * n)

    train = df_sorted.iloc[:n_train].reset_index(drop=True)
    val = df_sorted.iloc[n_train:n_val_end].reset_index(drop=True)
    test = df_sorted.iloc[n_val_end:].reset_index(drop=True)

    return train, val, test


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(IN_ALL)
    assert_expected_columns(df)

    # Ensure correct data types
    df["label"] = df["label"].astype(int)
    df["start0"] = df["start0"].astype(int)
    df["end0"] = df["end0"].astype(int)

    print("Loaded:", IN_ALL)
    print("Total windows:", len(df))
    print("Class counts (before):", df["label"].value_counts().to_dict())

    # Balance
    balanced = balance_by_undersampling(df)
    print("Class counts (balanced):", balanced["label"].value_counts().to_dict())

    # Save balanced dataset
    balanced.to_csv(OUT_BAL, index=False)
    print("Saved:", OUT_BAL)

    # Split
    train, val, test = split_by_coordinate(balanced)

    # Save splits
    train.to_csv(OUT_TRAIN, index=False)
    val.to_csv(OUT_VAL, index=False)
    test.to_csv(OUT_TEST, index=False)

    print("Saved splits:")
    print("  Train:", OUT_TRAIN, "rows:", len(train))
    print("  Val:  ", OUT_VAL, "rows:", len(val))
    print("  Test: ", OUT_TEST, "rows:", len(test))

    print("Train label counts:", train["label"].value_counts().to_dict())
    print("Val label counts:", val["label"].value_counts().to_dict())
    print("Test label counts:", test["label"].value_counts().to_dict())

    print("✅ Done.")


if __name__ == "__main__":
    main()