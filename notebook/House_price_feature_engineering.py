# ============================================================
# FEATURE ENGINEERING — AMES HOUSING DATASET
# ============================================================

import pandas as pd
import numpy as np

# ── 1. LOAD PREPROCESSED DATA ─────────────────────────────────
# We load the original train.csv again here because preprocessing
# already saved X_processed.csv with scaled + encoded data.
# Feature engineering works better on the raw numerical values
# before scaling — easier to interpret and combine meaningfully.
df = pd.read_csv("data/Raw/train.csv") #why are we taking raw instead of processed train data

print("Shape before feature engineering:", df.shape)

# ── 2. TOTAL SQUARE FEET ──────────────────────────────────────
# Buyers care about total livable space, not which floor it is on.
# Combining all floor areas into one feature captures this better
# than three separate features pulling in different directions.

df["TotalSF"] = (
    df["TotalBsmtSF"] +   # basement square feet
    df["1stFlrSF"] +      # first floor square feet
    df["2ndFlrSF"]        # second floor square feet
)

print(f"\nTotalSF sample values:\n{df['TotalSF'].describe().round(2)}") #is it used for mentioning till 2 decimals

# ── 3. TOTAL BATHROOMS ────────────────────────────────────────
# A buyer counts total bathrooms across the house.
# Half bathrooms (no shower/tub) are counted as 0.5
# because they add value but not as much as a full bathroom.

df["TotalBathrooms"] = (
    df["FullBath"] +               # full bathrooms above ground
    df["HalfBath"] * 0.5 +        # half bathrooms above ground
    df["BsmtFullBath"] +           # full bathrooms in basement
    df["BsmtHalfBath"] * 0.5      # half bathrooms in basement
)

print(f"\nTotalBathrooms value counts:\n{df['TotalBathrooms'].value_counts().sort_index()}")

# ── 5. HOUSE AGE AT SALE ──────────────────────────────────────
# A house built in 1950 and sold in 2010 is 60 years old.
# A house built in 2005 and sold in 2010 is only 5 years old.
# Newer houses generally command higher prices.

df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

# Sanity check — age should never be negative
# (a house can't be sold before it was built)
negative_age = (df["HouseAge"] < 0).sum()
print(f"\nNegative HouseAge count (should be 0): {negative_age}")

# ── 6. YEARS SINCE REMODEL ────────────────────────────────────
# Buyers care about how recently the house was renovated.
# A recently remodeled old house can sell for much more
# than an unremodeled house of the same age.
# Note: When no remodel happened, Kaggle sets YearRemodAdd = YearBuilt
# so this naturally equals HouseAge for unremodeled houses.

df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"] # don't we need to do a sanity check whether it is negative because sometimes there might errors in the dataset like the previous new feature

negative_remodel = (df["YearsSinceRemodel"] < 0).sum()
print(f"Negative YearsSinceRemodel count (should be 0): {negative_remodel}")

print(f"\nYearsSinceRemodel sample:\n{df['YearsSinceRemodel'].describe().round(2)}")

# ── 7. OVERALL SCORE ──────────────────────────────────────────
# OverallQual = quality of materials (1-10)
# OverallCond = condition of the house (1-10)
# Averaging gives a balanced single quality score.
# A house with great materials but poor condition
# will score in the middle — which is fair.

df["OverallScore"] = (df["OverallQual"] + df["OverallCond"]) / 2

print(f"\nOverallScore sample:\n{df['OverallScore'].describe().round(2)}")

# ── 8. TOTAL OUTDOOR AREA ─────────────────────────────────────
# Combines meaningful outdoor/exterior areas into one feature.
# We exclude PoolArea, EnclosedPorch, 3SsnPorch, ScreenPorch
# because most houses have 0 for these — including them would
# add noise rather than signal for the majority of houses.

df["TotalOutdoorSF"] = (
    df["MasVnrArea"] +    # masonry veneer area (brick/stone facing)
    df["GarageArea"] +    # garage square feet
    df["WoodDeckSF"] +    # wood deck area
    df["OpenPorchSF"]     # open porch area
)

print(f"\nTotalOutdoorSF sample:\n{df['TotalOutdoorSF'].describe().round(2)}")

# ── 9. VERIFY NEW FEATURES ────────────────────────────────────
new_features = [
    "TotalSF", "TotalBathrooms", 
    "HouseAge", "YearsSinceRemodel", "OverallScore", "TotalOutdoorSF"
]

print("\n" + "=" * 50)
print("NEW FEATURES CORRELATION WITH SALEPRICE:")
correlations = df[new_features + ["SalePrice"]].corr()["SalePrice"].drop("SalePrice")
print(correlations.sort_values(ascending=False).round(3))

# ── 10. DROP ORIGINAL COLUMNS USED IN FEATURE CREATION ────────
# Now that we have combined features, the original columns
# carry redundant information. Keeping them alongside the new
# features would cause multicollinearity — where two features
# tell the model the same thing, confusing it.

cols_to_drop = [
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",          # replaced by TotalSF
    "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",  # replaced by TotalBathrooms
    "YearBuilt", "YearRemodAdd",                      # replaced by HouseAge, YearsSinceRemodel
    "OverallQual", "OverallCond",                     # replaced by OverallScore
    "MasVnrArea", "GarageArea", "WoodDeckSF", "OpenPorchSF"  # replaced by TotalOutdoorSF
]

df.drop(columns=cols_to_drop, inplace=True)

print(f"\nShape after dropping redundant columns: {df.shape}")

# ── 11. SAVE ENGINEERED DATA ──────────────────────────────────
df.to_csv("data/Raw/train_engineered.csv", index=False)
print("\nEngineered data saved to train_engineered.csv")

# ── 12. SUMMARY ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("FEATURE ENGINEERING SUMMARY")
print(f"  New features created : {len(new_features)}")
print(f"  Columns dropped      : {len(cols_to_drop)}")
print(f"  Final shape          : {df.shape}")
print("\n  New features and their correlation with SalePrice:")
for feat in new_features:
    corr = df[feat].corr(df["SalePrice"]) if "SalePrice" in df.columns else correlations[feat]
    print(f"    {feat:<25} → {correlations[feat]:.3f}")
print("=" * 50)
