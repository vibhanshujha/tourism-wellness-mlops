"""
Data Preparation
----------------
1. Loads the dataset directly from the repository data folder.
2. Cleans it: drops identifier/index columns, fixes category typos,
   removes duplicates and imputes any missing values.
3. Splits into stratified train/test sets and saves them locally as CSV.

The GitHub Actions workflow uploads the four CSV files as an artifact so the
training job can download them.
"""

# ============================== OBSERVATIONS ===============================
# - "Unnamed: 0" (row index) and "CustomerID" (identifier) are removed: they
#   carry no predictive signal and would let the model memorise rows.
# - "Fe Male" is merged into "Female", leaving only Female / Male.
# - 117 exact duplicate rows are dropped so the same record cannot appear in
#   both train and test (which would inflate the test score).
# - 80/20 stratified split: 3,208 train rows and 803 test rows, both keeping
#   the same ~19.3% buyer rate.
# - The four CSVs are uploaded by the workflow as the "data-splits" artifact
#   and downloaded by the training job.
# ===========================================================================

import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")
TARGET = "ProdTaken"

# Columns that carry no predictive signal:
#  - "Unnamed: 0" is a leftover row index from an earlier export
#  - "CustomerID" is a unique identifier
DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]

# Output files (written to the repository root, where the workflow expects them)
OUTPUT_FILES = {
    "Xtrain": "Xtrain.csv",
    "Xtest": "Xtest.csv",
    "ytrain": "ytrain.csv",
    "ytest": "ytest.csv",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop unnecessary columns (only the ones that are actually present)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # Trim stray whitespace in text columns
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].str.strip()

    # Fix data-entry typo: "Fe Male" should be "Female"
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    # Impute missing values (robust for future data refreshes):
    # numeric -> median, categorical -> most frequent value
    for col in df.columns:
        if df[col].isna().any():
            if col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    # Rows without a target cannot be used for training
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    return df


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded raw data from {DATA_PATH}: {df.shape}")

    df = clean(df)
    print(f"Cleaned data shape: {df.shape}")
    print(f"Gender categories after cleaning: {sorted(df['Gender'].unique())}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratify so both splits keep the same ~19% buyer rate
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Xtrain.to_csv(OUTPUT_FILES["Xtrain"], index=False)
    Xtest.to_csv(OUTPUT_FILES["Xtest"], index=False)
    ytrain.to_csv(OUTPUT_FILES["ytrain"], index=False)
    ytest.to_csv(OUTPUT_FILES["ytest"], index=False)

    print(f"Train: {Xtrain.shape}, buyer rate {ytrain.mean():.3f}")
    print(f"Test:  {Xtest.shape}, buyer rate {ytest.mean():.3f}")
    print("Saved:", ", ".join(OUTPUT_FILES.values()))


if __name__ == "__main__":
    main()
