# ============================================================
# EXPLORATORY DATA ANALYSIS — AMES HOUSING DATASET
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── 0. SETUP ─────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")   # clean plot style
pd.set_option("display.max_columns", 100) # show all columns in terminal

# ── 1. LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv("data/Raw/train.csv")  # download from Kaggle

# ── 2. FIRST LOOK ─────────────────────────────────────────────
print("=" * 50)
print("SHAPE:", df.shape)           # rows × columns
print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATA TYPES:")
print(df.dtypes.value_counts())     # how many numerical vs object columns

print("\nBASIC STATS (numerical columns):")
print(df.describe())                # count, mean, std, min, max, quartiles

# ── 3. MISSING VALUES ─────────────────────────────────────────
print("\n" + "=" * 50)
print("MISSING VALUES (top 20):")

missing = df.isnull().sum()                        # count nulls per column
missing_pct = (missing / len(df)) * 100            # as percentage
missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
}).sort_values("Missing %", ascending=False)

print(missing_df[missing_df["Missing Count"] > 0].head(20))

# PLOT — missing values bar chart
plt.figure(figsize=(12, 5))
cols_with_missing = missing_df[missing_df["Missing Count"] > 0].head(20)
plt.bar(cols_with_missing.index, cols_with_missing["Missing %"], color="steelblue")
plt.xticks(rotation=45, ha="right")
plt.title("Top 20 Columns with Missing Values (%)")
plt.ylabel("Missing %")
plt.tight_layout()
plt.savefig("missing_values.png", dpi=150)
plt.show()

# ── 4. TARGET VARIABLE — SalePrice ────────────────────────────
print("\n" + "=" * 50)
print("TARGET VARIABLE: SalePrice")
print(df["SalePrice"].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original distribution
axes[0].hist(df["SalePrice"], bins=50, color="steelblue", edgecolor="white")
axes[0].set_title("SalePrice Distribution (Original)")
axes[0].set_xlabel("Sale Price")

# Log-transformed distribution (more "normal" — better for modelling)
axes[1].hist(np.log1p(df["SalePrice"]), bins=50, color="coral", edgecolor="white")
axes[1].set_title("SalePrice Distribution (Log-Transformed)")
axes[1].set_xlabel("log(Sale Price)")

plt.tight_layout()
plt.savefig("saleprice_distribution.png", dpi=150)
plt.show()

# Skewness tells us how lopsided the distribution is
print(f"\nSkewness (original): {df['SalePrice'].skew():.2f}")
print(f"Skewness (log):      {np.log1p(df['SalePrice']).skew():.2f}")

# ── 5. NUMERICAL FEATURES ─────────────────────────────────────
# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("\n" + "=" * 50)
print(f"Numerical columns : {len(num_cols)}")
print(f"Categorical columns: {len(cat_cols)}")

# CORRELATION with SalePrice
corr = df[num_cols].corr()["SalePrice"].drop("SalePrice")
top_corr = corr.abs().sort_values(ascending=False).head(10)

print("\nTop 10 numerical features correlated with SalePrice:")
print(top_corr)

# PLOT — heatmap of top correlated features
top_features = top_corr.index.tolist() + ["SalePrice"]
plt.figure(figsize=(10, 8))
sns.heatmap(
    df[top_features].corr(),
    annot=True, fmt=".2f",
    cmap="coolwarm", center=0,
    square=True, linewidths=0.5
)
plt.title("Correlation Heatmap — Top Features vs SalePrice")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

# PLOT — scatter: OverallQual and GrLivArea vs SalePrice
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df["OverallQual"], df["SalePrice"], alpha=0.4, color="steelblue")
axes[0].set_title("Overall Quality vs Sale Price")
axes[0].set_xlabel("Overall Quality (1–10)")
axes[0].set_ylabel("Sale Price")

axes[1].scatter(df["GrLivArea"], df["SalePrice"], alpha=0.4, color="coral")
axes[1].set_title("Above-Ground Living Area vs Sale Price")
axes[1].set_xlabel("GrLivArea (sq ft)")
axes[1].set_ylabel("Sale Price")

plt.tight_layout()
plt.savefig("scatter_plots.png", dpi=150)
plt.show()

# ── 6. CATEGORICAL FEATURES ───────────────────────────────────
print("\n" + "=" * 50)
print("Top categorical features (unique value count):")
print(df[cat_cols].nunique().sort_values(ascending=False).head(10))

# PLOT — SalePrice by Neighborhood (box plot)
plt.figure(figsize=(16, 6))
order = df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="Neighborhood", y="SalePrice", order=order, palette="coolwarm")
plt.xticks(rotation=45, ha="right")
plt.title("Sale Price by Neighborhood")
plt.tight_layout()
plt.savefig("neighborhood_boxplot.png", dpi=150)
plt.show()

# ── 7. OUTLIERS ───────────────────────────────────────────────
# Rule of thumb: GrLivArea > 4000 with low SalePrice = likely outliers
print("\n" + "=" * 50)
print("Potential outliers (GrLivArea > 4000):")
outliers = df[df["GrLivArea"] > 4000][["GrLivArea", "SalePrice"]]
print(outliers)

plt.figure(figsize=(8, 5))
plt.scatter(df["GrLivArea"], df["SalePrice"], alpha=0.4, color="steelblue", label="Normal")
plt.scatter(outliers["GrLivArea"], outliers["SalePrice"], color="red", s=80, label="Outlier?")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.title("Outlier Detection — GrLivArea vs SalePrice")
plt.legend()
plt.tight_layout()
plt.savefig("outliers.png", dpi=150)
plt.show()

# ── 8. SUMMARY ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("EDA SUMMARY")
print(f"  Dataset shape    : {df.shape}")
print(f"  Numerical cols   : {len(num_cols)}")
print(f"  Categorical cols : {len(cat_cols)}")
print(f"  Columns with missing values: {(missing > 0).sum()}")
print(f"  SalePrice range  : ${df['SalePrice'].min():,.0f} – ${df['SalePrice'].max():,.0f}")
print(f"  Top correlated feature: {top_corr.index[0]} ({top_corr.iloc[0]:.2f})")
print("=" * 50)
