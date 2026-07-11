# ============================================================
# HYPERPARAMETER TUNING — AMES HOUSING DATASET
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import joblib
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")

# ── 1. LOAD DATA ─────────────────────────────────────────────
X = pd.read_csv("data/Processed/X_processed.csv")
y = pd.read_csv("data/Processed/y_processed.csv").squeeze()

# Same random_state=42 — ensures identical split as model building
# so results are directly comparable to our earlier evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data loaded.")
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ── 2. BASELINE — UNTUNED XGBOOST ────────────────────────────
# Record the untuned performance first so we can compare
# after tuning and measure how much improvement we achieved

baseline = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)
baseline.fit(X_train, y_train)
baseline_pred  = baseline.predict(X_test)
baseline_rmse  = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2    = r2_score(y_test, baseline_pred)

# CV RMSE for baseline
baseline_cv = -cross_val_score(
    baseline, X_train, y_train,
    cv=5, scoring="neg_root_mean_squared_error"
).mean()

print(f"\nBaseline XGBoost:")
print(f"  Test RMSE : {baseline_rmse:.4f}")
print(f"  R²        : {baseline_r2:.4f}")
print(f"  CV RMSE   : {baseline_cv:.4f}")

# ── 3. DEFINE PARAMETER GRID ──────────────────────────────────
# This is the search space — the range of values to sample from
# RandomizedSearchCV picks random combinations from these lists
#
# max_depth        → controls how deep each tree grows
#                    lower = simpler trees = less overfitting
# learning_rate    → how much each tree corrects previous errors
#                    lower = slower but more precise learning
# n_estimators     → number of sequential trees
#                    more trees + lower learning rate = better
# subsample        → fraction of training data used per tree
#                    <1.0 introduces randomness = less overfitting
# min_child_weight → minimum data points needed to create a new split
#                    higher = more conservative splits = less overfitting
# colsample_bytree → fraction of features used per tree
#                    similar to subsample but for features not rows

param_grid = {
    "max_depth"        : [3, 4, 5, 6],
    "learning_rate"    : [0.01, 0.05, 0.1],
    "n_estimators"     : [100, 200, 300],
    "subsample"        : [0.7, 0.8, 0.9],
    "min_child_weight" : [1, 3, 5],
    "colsample_bytree" : [0.7, 0.8, 0.9]
}

# Total possible combinations if we used GridSearchCV:
total = 1
for v in param_grid.values():
    total *= len(v)
print(f"\nTotal possible combinations (GridSearch): {total}")
print(f"RandomizedSearchCV will only try: 50")
print(f"Saving {total - 50} unnecessary model fits!")

# ── 4. RANDOMIZED SEARCH ──────────────────────────────────────
# n_iter=50        → try 50 random combinations from param_grid
# cv=5             → evaluate each combination with 5-fold CV
# scoring          → use negative RMSE (sklearn convention)
# n_jobs=-1        → use all CPU cores to speed up search
#                    -1 means use everything available
# random_state=42  → reproducible random sampling
# verbose=2        → print progress so you know it is working

xgb = XGBRegressor(random_state=42, verbosity=0)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=50,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=2
)

print("\nStarting RandomizedSearchCV — this may take a few minutes...")
print("(50 combinations × 5 folds = 250 model fits)")

# fit() runs the entire search — trains 250 models internally
# and tracks which combination gives the best CV RMSE
random_search.fit(X_train, y_train)

print("\nSearch complete!")

joblib.dump(random_search.best_estimator_, "artifacts/model.pkl")
print("Model saved to artifacts/model.pkl")

# ── 5. BEST PARAMETERS ────────────────────────────────────────
# best_params_ → the combination that gave lowest CV RMSE
# best_score_  → the best CV RMSE (negative — flip it)

best_params = random_search.best_params_
best_cv_rmse = -random_search.best_score_

print("\n" + "=" * 50)
print("BEST PARAMETERS FOUND:")
for param, value in best_params.items():
    print(f"  {param:<20} : {value}")
print(f"\nBest CV RMSE : {best_cv_rmse:.4f}")
print(f"Baseline CV RMSE : {baseline_cv:.4f}")
print(f"Improvement  : {baseline_cv - best_cv_rmse:.4f}")

# ── 6. EVALUATE TUNED MODEL ───────────────────────────────────
# best_estimator_ → the actual trained model with best parameters
# already fitted on full training data — ready to predict

tuned_model = random_search.best_estimator_ #where did we get this estimator
tuned_pred  = tuned_model.predict(X_test)

tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_pred))
tuned_mae  = mean_absolute_error(y_test, tuned_pred)
tuned_r2   = r2_score(y_test, tuned_pred)

# Overfitting check for tuned model
train_pred_tuned = tuned_model.predict(X_train)
train_rmse_tuned = np.sqrt(mean_squared_error(y_train, train_pred_tuned))
gap_tuned        = tuned_rmse - train_rmse_tuned

print("\n" + "=" * 50)
print("TUNED MODEL PERFORMANCE:")
print(f"  Test RMSE  : {tuned_rmse:.4f}  (baseline: {baseline_rmse:.4f})")
print(f"  MAE        : {tuned_mae:.4f}")
print(f"  R²         : {tuned_r2:.4f}  (baseline: {baseline_r2:.4f})")
print(f"  Train RMSE : {train_rmse_tuned:.4f}")
print(f"  Gap        : {gap_tuned:.4f}  {'⚠ still overfitting' if gap_tuned > 0.05 else '✓ improved'}")

# ── 7. DOLLAR VALUE COMPARISON ────────────────────────────────
# Convert RMSE back to actual dollars for real world interpretation
y_test_inv        = np.expm1(y_test)
baseline_pred_inv = np.expm1(baseline_pred)
tuned_pred_inv    = np.expm1(tuned_pred)

baseline_dollar = np.sqrt(mean_squared_error(y_test_inv, baseline_pred_inv))
tuned_dollar    = np.sqrt(mean_squared_error(y_test_inv, tuned_pred_inv))

print("\n" + "=" * 50)
print("DOLLAR VALUE COMPARISON:")
print(f"  Baseline RMSE : ${baseline_dollar:,.0f}")
print(f"  Tuned RMSE    : ${tuned_dollar:,.0f}")
print(f"  Improvement   : ${baseline_dollar - tuned_dollar:,.0f}")

# ── 8. FEATURE IMPORTANCE ─────────────────────────────────────
# XGBoost tracks how much each feature contributed to
# reducing error across all trees — called feature importance
# Higher importance = feature was used more often to split trees
# and contributed more to reducing prediction error

importance = pd.Series(
    tuned_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False).head(15) #what is this function, is this the function where we found the best features

plt.figure(figsize=(10, 6))
plt.barh(importance.index[::-1], importance.values[::-1], color="steelblue")
plt.title("Top 15 Feature Importances — Tuned XGBoost")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("\nFeature importance plot saved.")

# ── 9. BEFORE VS AFTER COMPARISON PLOT ───────────────────────
# Visual summary of improvement from tuning

labels   = ["Baseline\nXGBoost", "Tuned\nXGBoost"]
rmse_vals = [baseline_rmse, tuned_rmse]
r2_vals   = [baseline_r2,   tuned_r2]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# RMSE comparison — lower is better
axes[0].bar(labels, rmse_vals, color=["steelblue", "coral"], width=0.4)
axes[0].set_title("RMSE Comparison (lower is better)")
axes[0].set_ylabel("RMSE (log scale)")
for i, v in enumerate(rmse_vals):
    axes[0].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=11) # what is this and why it is used

# R² comparison — higher is better
axes[1].bar(labels, r2_vals, color=["steelblue", "coral"], width=0.4)
axes[1].set_title("R² Comparison (higher is better)")
axes[1].set_ylabel("R² Score")
axes[1].set_ylim(0.85, 0.95)
for i, v in enumerate(r2_vals):
    axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=11)

plt.tight_layout()
plt.savefig("tuning_comparison.png", dpi=150)
plt.show()
print("Comparison plot saved.")

# ── 10. SUMMARY ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("HYPERPARAMETER TUNING SUMMARY")
print(f"  Strategy          : RandomizedSearchCV")
print(f"  Combinations tried: 50 out of {total} possible")
print(f"  CV folds          : 5")
print(f"  Total model fits  : 250")
print(f"\n  Baseline CV RMSE  : {baseline_cv:.4f}")
print(f"  Tuned CV RMSE     : {best_cv_rmse:.4f}")
print(f"  Improvement       : {baseline_cv - best_cv_rmse:.4f}")
print(f"\n  Baseline dollar RMSE : ${baseline_dollar:,.0f}")
print(f"  Tuned dollar RMSE    : ${tuned_dollar:,.0f}")
print(f"  Dollar improvement   : ${baseline_dollar - tuned_dollar:,.0f}")
print("=" * 50)
