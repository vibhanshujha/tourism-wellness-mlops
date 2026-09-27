"""
Data Registration
-----------------
Registers the raw tourism dataset that lives inside the GitHub repository
(tourism_project/data/tourism.csv).

The script validates that the file exists, that every expected column is
present, and that the target is binary. It then prints a short summary.
If any check fails, the script exits with a non-zero code, which stops the
GitHub Actions pipeline before any bad data reaches training.
"""

# ============================== OBSERVATIONS ===============================
# - The dataset has 4,128 rows and 21 columns. All 20 columns from the data
#   dictionary are present, plus an extra "Unnamed: 0" column (a leftover row
#   index), which is dropped in prep.py.
# - There are no missing values and no duplicate CustomerIDs.
# - The target is imbalanced: only ~19% of customers bought a package, so the
#   model is judged on F1 / recall / precision for buyers, not just accuracy.
# - "Gender" contains a data-entry typo ("Fe Male", 155 rows), fixed in prep.py.
# - Quick look at purchase rates: passport holders ~36% vs ~12% for non-holders;
#   Basic package pitch ~30% vs 8-16% for premium packages; single customers
#   ~36%; customers aged 25 or under ~43%; 6 follow-ups ~40% vs 1-2 ~10-13%.
# - If a required column is ever missing, this script exits with an error and
#   the GitHub Actions pipeline stops before training.
# ===========================================================================

import os
import sys

import pandas as pd

# Path of the dataset relative to the repository root
DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")

TARGET = "ProdTaken"

# Every column listed in the data dictionary
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def main() -> None:
    # 1. The file must exist in the repo's data folder
    if not os.path.exists(DATA_PATH):
        sys.exit(f"[FAIL] Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[OK] Loaded {DATA_PATH}")

    # 2. All expected columns must be present
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"[FAIL] Missing expected columns: {missing}")
    print(f"[OK] All {len(EXPECTED_COLUMNS)} expected columns are present")

    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        print(f"[INFO] Extra columns (will be dropped during prep): {extra}")

    # 3. Target must be binary 0/1
    target_values = set(df[TARGET].dropna().unique())
    if not target_values.issubset({0, 1}):
        sys.exit(f"[FAIL] Target {TARGET} is not binary: {target_values}")
    print(f"[OK] Target '{TARGET}' is binary")

    # 4. Summary
    print("\n===== DATASET SUMMARY =====")
    print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
    print(f"Duplicate CustomerIDs: {df['CustomerID'].duplicated().sum()}")
    print(f"Total missing values: {int(df.isna().sum().sum())}")
    print("\nTarget distribution (share of customers):")
    print(df[TARGET].value_counts(normalize=True).round(3).to_string())
    print("\nColumn types:")
    print(df.dtypes.to_string())
    print("\nDataset registered successfully.")


if __name__ == "__main__":
    main()
