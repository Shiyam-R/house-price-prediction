# ============================================================
# DATA PREPROCESSING — AMES HOUSING DATASET
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ── 1. LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv("data/Raw/train_engineered.csv")

# Always keep a copy of the original — never modify raw data directly
# If something goes wrong, you can always restart from df_original
df_original = df.copy()

print("Shape before preprocessing:", df.shape)

# ── 2. DROP UNNECESSARY COLUMNS ───────────────────────────────
# 'Id' is just a row number — it has no predictive value
# Keeping it could confuse the model into thinking row number matters
df.drop(columns=["Id"], inplace=True)

print("Shape after dropping Id:", df.shape)

# ── 3. HANDLE MISSING VALUES ──────────────────────────────────
# From EDA we know missing values fall into 2 types:
#   Type A — "None" missing: e.g. PoolQC is NaN because house has NO pool
#   Type B — truly missing: e.g. LotFrontage we don't know the value

# -- Type A: Fill with "None" (categorical) or 0 (numerical) --
# These columns are NaN because the feature simply doesn't exist
# for that house — not because data was lost

none_cols = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType"
]
# fill_value="None" means: the string "None", not Python's None/NaN
df[none_cols] = df[none_cols].fillna("None")

# Numerical counterparts — no garage/basement means 0 area, 0 cars, etc.
zero_cols = [
    "GarageYrBlt", "TotalOutdoorSF", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalSF", "TotalBathrooms",
]
df[zero_cols] = df[zero_cols].fillna(0)

# -- Type B: Fill with statistical values --
# LotFrontage: street length connected to property
# We fill with the median of the same neighborhood
# because lot sizes tend to be similar within a neighborhood
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"] \
                      .transform(lambda x: x.fillna(x.median()))

# Electrical: only 1 missing — just use the most common value (mode)
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
# .mode() returns a Series, [0] gets the first (most common) value

# -- Confirm no missing values remain --
remaining = df.isnull().sum().sum()   # total nulls across all columns
print(f"\nTotal missing values remaining: {remaining}")  # should be 0

# ── 4. LOG-TRANSFORM SKEWED NUMERICAL FEATURES ────────────────
# From EDA we saw SalePrice is right-skewed
# Many other numerical features are also skewed
# Log-transforming them helps models learn better patterns

# Select only numerical columns (excluding the target SalePrice)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove("SalePrice")

# Calculate skewness for each numerical column
skewness = df[num_cols].skew()

# Keep only columns where skewness is above threshold (1.0 is common rule)
skewed_cols = skewness[skewness.abs() > 1.0].index.tolist()
print(f"\nNumber of skewed features to transform: {len(skewed_cols)}")

# Save skewed columns for API log transform
joblib.dump(skewed_cols, "artifacts/skewed_columns.pkl")
print("Skewed columns saved to artifacts/skewed_columns.pkl")

# Apply log1p to all skewed columns
# log1p = log(1 + x) — safe for 0 values (log(0) is undefined)
df[skewed_cols] = np.log1p(df[skewed_cols])

# Also log-transform the target variable SalePrice (as we saw in EDA)
df["SalePrice"] = np.log1p(df["SalePrice"])
print("SalePrice log-transformed.")

# ── 5. ENCODE CATEGORICAL COLUMNS ─────────────────────────────
# ML models only understand numbers — we must convert text columns

# Identify all remaining categorical (object) columns
cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
print(f"\nCategorical columns to encode: {len(cat_cols)}")

# One-Hot Encoding — creates a new binary column for each category
# Example: "Neighborhood" with values [A, B, C] becomes:
#   Neighborhood_A  Neighborhood_B  Neighborhood_C
#          1               0               0        ← was A
#          0               1               0        ← was B
#
# drop_first=True removes one column per feature to avoid redundancy
# (if not A and not B, it must be C — so C column is unnecessary)
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("Shape after encoding:", df.shape)

# ── 6. FEATURE SCALING ────────────────────────────────────────
# Many ML models (Linear Regression, SVM, KNN) are sensitive to scale
# A feature with values 0–500000 will dominate one with values 1–10
# Scaling brings all features to a similar range

# Separate features (X) and target (y)
X = df.drop(columns=["SalePrice"])   # everything except target
y = df["SalePrice"]                  # only the target

# Save training means 
train_means = pd.Series(X.mean(), index=X.columns)
joblib.dump(train_means, "artifacts/train_means.pkl")
print("Training means saved to artifacts/train_means.pkl")

# Save encoded column structure
encoded_columns = X.columns.tolist()
joblib.dump(encoded_columns, "artifacts/encoded_columns.pkl")
print("Encoded columns saved to artifacts/encoded_columns.pkl")

# StandardScaler transforms each feature to have:
#   mean = 0 and standard deviation = 1
# Formula: z = (x - mean) / std
scaler = StandardScaler()

# fit_transform: learns the mean & std from X, then scales it
# Result is a numpy array, so we convert back to DataFrame
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns     # keep original column names
)

joblib.dump(scaler, "artifacts/scaler.pkl")
print("Scaler saved to artifacts/scaler.pkl")

print("\nFeature scaling complete.")
print(f"X shape: {X_scaled.shape}")
print(f"y shape: {y.shape}")

# Quick check — mean and std of first 3 columns after scaling
print("\nSample mean after scaling (should be ≈ 0):")
print(X_scaled.iloc[:, :3].mean().round(4))

print("\nSample std after scaling (should be ≈ 1):")
print(X_scaled.iloc[:, :3].std().round(4))

# ── 7. SAVE PROCESSED DATA ────────────────────────────────────
# Save for use in the next step (Feature Engineering / Model Building)
X_scaled.to_csv("data/Processed/X_processed.csv", index=False)
y.to_csv("data/Processed/y_processed.csv", index=False)

print("\nProcessed data saved to X_processed.csv and y_processed.csv")

# ── 8. SUMMARY ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("PREPROCESSING SUMMARY")
print(f"  Original shape       : {df_original.shape}")
print(f"  After preprocessing  : {X_scaled.shape}")
print(f"  Missing values left  : {remaining}")
print(f"  Skewed cols fixed    : {len(skewed_cols)}")
print(f"  Categorical cols enc : {len(cat_cols)}")
print("=" * 50)
