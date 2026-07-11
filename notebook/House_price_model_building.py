# ============================================================
# MODEL BUILDING — AMES HOUSING DATASET
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")  # suppress minor version warnings

# ── 1. LOAD PROCESSED DATA ────────────────────────────────────
# We load the final processed file that went through:
# Feature Engineering → Preprocessing
X = pd.read_csv("data/Processed/X_processed.csv")
y = pd.read_csv("data/Processed/y_processed.csv").squeeze()
# .squeeze() converts single column DataFrame to a Series
# sklearn expects y as a 1D Series, not a DataFrame

print("X shape:", X.shape)
print("y shape:", y.shape)

# ── 2. TRAIN TEST SPLIT ───────────────────────────────────────
# We split data into two parts:
#   Training set  → model learns patterns from this (80%)
#   Test set      → we evaluate the model on unseen data (20%)
#
# test_size=0.2   → 20% of data goes to test set
# random_state=42 → fixes the random split so results are
#                   reproducible every time you run the code # what is random state first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size : {X_train.shape[0]} rows")
print(f"Test set size     : {X_test.shape[0]} rows")

# ── 3. DEFINE A HELPER FUNCTION FOR EVALUATION ────────────────
# Instead of writing the same evaluation code 4 times (once per model)
# we write it once as a function and call it for each model.
# This is the DRY principle — Don't Repeat Yourself.

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Trains a model and prints evaluation metrics.
    Parameters:
        name    : string label for the model
        model   : sklearn model object
        X_train, X_test, y_train, y_test : split data
    """
    # Train the model on training data
    model.fit(X_train, y_train)

    # Predict on test data — model has never seen this
    y_pred = model.predict(X_test)

    # ── Metrics ──────────────────────────────────────────────
    # RMSE: Root Mean Squared Error
    # Average distance between predicted and actual values
    # Lower is better. In our case SalePrice is log-transformed
    # so RMSE is in log units — we will interpret relatively.
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # MAE: Mean Absolute Error
    # Average of absolute differences between predicted and actual
    # Less sensitive to outliers than RMSE
    mae = mean_absolute_error(y_test, y_pred)

    # R²: R-squared (coefficient of determination)
    # How much of the variance in SalePrice the model explains
    # 1.0 = perfect predictions, 0.0 = no better than predicting mean
    r2 = r2_score(y_test, y_pred)

    # Cross Validation Score
    # Instead of evaluating on just one test split,
    # we split data into 5 parts (folds) and evaluate 5 times.
    # Each fold takes a turn being the test set.
    # This gives a more reliable performance estimate.
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,                    # 5 folds
        scoring="neg_root_mean_squared_error"  # sklearn uses negative RMSE
    )
    # Convert negative RMSE back to positive for readability
    cv_rmse = -cv_scores.mean()

    print(f"\n{'='*40}")
    print(f"Model     : {name}")
    print(f"RMSE      : {rmse:.4f}")
    print(f"MAE       : {mae:.4f}")
    print(f"R²        : {r2:.4f}")
    print(f"CV RMSE   : {cv_rmse:.4f} (±{-cv_scores.std():.4f})")

    return {
        "Model": name, "RMSE": rmse,
        "MAE": mae, "R2": r2, "CV_RMSE": cv_rmse
    }

# ── 4. MODEL 1 — LINEAR REGRESSION (BASELINE) ─────────────────
# Always start with the simplest model.
# Linear Regression assumes a straight line relationship between
# features and target. It is fast, interpretable, and gives us
# a baseline score to beat with more complex models.

lr = LinearRegression()
lr_results = evaluate_model(
    "Linear Regression", lr,
    X_train, X_test, y_train, y_test
)

# ── 5. MODEL 2 — SVR (SUPPORT VECTOR REGRESSOR) ───────────────
# SVR finds a boundary (hyperplane) that fits the data within
# a margin of tolerance (epsilon). Points inside the margin
# are not penalized — only points outside are.
# kernel="rbf" uses a radial basis function — handles non-linear
# relationships that Linear Regression cannot capture.

svr = SVR(kernel="rbf")
svr_results = evaluate_model(
    "SVR", svr,
    X_train, X_test, y_train, y_test
)

# ── 6. MODEL 3 — RANDOM FOREST ────────────────────────────────
# Random Forest builds many decision trees (n_estimators=100)
# and averages their predictions.
# Each tree is trained on a random subset of data and features
# — this reduces overfitting compared to a single decision tree.
# n_estimators=100 → build 100 trees and average predictions
# random_state=42  → reproducible results

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf_results = evaluate_model(
    "Random Forest", rf,
    X_train, X_test, y_train, y_test
)

# ── 7. MODEL 4 — XGBOOST ──────────────────────────────────────
# XGBoost builds trees sequentially — each new tree corrects
# the errors of the previous tree. This is called boosting.
# It is one of the most powerful algorithms for tabular data
# and frequently wins Kaggle competitions.
# n_estimators=100    → build 100 sequential trees
# learning_rate=0.1   → how much each tree corrects the error
#                        lower = slower but more precise learning
# random_state=42     → reproducible results

xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    verbosity=0        # suppress XGBoost training logs
)
xgb_results = evaluate_model(
    "XGBoost", xgb,
    X_train, X_test, y_train, y_test
)

# ── 8. COMPARE ALL MODELS ─────────────────────────────────────
# Collect all results into a DataFrame for easy comparison
results_df = pd.DataFrame([lr_results, svr_results, rf_results, xgb_results])
results_df = results_df.sort_values("RMSE")  # sort by RMSE ascending (lower is better)

print("\n" + "=" * 50)
print("MODEL COMPARISON (sorted by RMSE):")
print(results_df.to_string(index=False))

# Identify the best model
best_model_name = results_df.iloc[0]["Model"]
best_rmse = results_df.iloc[0]["RMSE"]
best_r2 = results_df.iloc[0]["R2"]

print(f"\nBest Model : {best_model_name}")
print(f"Best RMSE  : {best_rmse:.4f}")
print(f"Best R²    : {best_r2:.4f}")

# ── 9. SUMMARY ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL BUILDING SUMMARY")
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Test samples     : {X_test.shape[0]}")
print(f"  Models trained   : 4")
print(f"  Best model       : {best_model_name}")
print(f"  Best RMSE        : {best_rmse:.4f}")
print(f"  Best R²          : {best_r2:.4f}")
print("=" * 50)
