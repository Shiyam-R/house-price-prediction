# ============================================================
# MODEL EVALUATION — AMES HOUSING DATASET
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")

# ── 1. LOAD DATA ─────────────────────────────────────────────
X = pd.read_csv("data/Processed/X_processed.csv")
y = pd.read_csv("data/Processed/y_processed.csv").squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data loaded and split.")
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ── 2. TRAIN TOP 3 MODELS ─────────────────────────────────────
# From model building we know SVR performed worst
# So we evaluate the top 3 models in detail here
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1,
                            random_state=42, verbosity=0)
}

# Train all models
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} trained.")

# ── 3. OVERFITTING CHECK — TRAIN VS TEST RMSE ─────────────────
# A good model should perform similarly on train and test data.
# Large gap = model memorized training data = overfitting
# Small gap = model learned generalizable patterns = good

print("\n" + "=" * 50)
print("OVERFITTING CHECK — Train vs Test RMSE")
print("=" * 50)

for name, model in models.items():
    # Predict on BOTH train and test
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    # Calculate RMSE for both
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse  = np.sqrt(mean_squared_error(y_test,  y_test_pred))

    # Gap between train and test RMSE is the overfitting signal
    # Small gap = generalizes well, Large gap = overfitting
    gap = test_rmse - train_rmse

    print(f"\n{name}")
    print(f"  Train RMSE : {train_rmse:.4f}")
    print(f"  Test RMSE  : {test_rmse:.4f}")
    print(f"  Gap        : {gap:.4f}  {'⚠ possible overfit' if gap > 0.02 else '✓ good'}")

# ── 4. PREDICTED VS ACTUAL PLOT ───────────────────────────────
# For a perfect model, all points lie on the diagonal y=x line
# Points above the line = model underpredicted
# Points below the line = model overpredicted
# Scatter around the line = random error (acceptable)
# Curved pattern = systematic error (bad)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)

    ax.scatter(y_test, y_pred, alpha=0.4, color="steelblue", s=20)

    # Draw the perfect prediction line (y = x)
    # min and max of actual values define the line range
    line_min = min(y_test.min(), y_pred.min())
    line_max = max(y_test.max(), y_pred.max())
    ax.plot([line_min, line_max], [line_min, line_max],
            color="red", linewidth=1.5, linestyle="--", label="Perfect prediction")

    ax.set_title(f"{name}\nPredicted vs Actual")
    ax.set_xlabel("Actual SalePrice (log)")
    ax.set_ylabel("Predicted SalePrice (log)")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plots/predicted_vs_actual.png", dpi=150)
plt.show()
print("\nPredicted vs Actual plot saved.")

# ── 5. RESIDUAL PLOT ──────────────────────────────────────────
# Residual = Actual - Predicted
# Good model: residuals randomly scattered around 0
# Bad model : residuals show a pattern (curve, trend, fan shape)
#
# Fan shape = heteroscedasticity (error grows with prediction value)
# Curve     = the model is missing a non-linear relationship
# Trend     = systematic bias in predictions

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, model) in zip(axes, models.items()):
    y_pred    = model.predict(X_test)
    residuals = y_test - y_pred    # actual minus predicted

    # Scatter: x=predicted value, y=residual
    # We use predicted on x-axis (not actual) — standard practice
    ax.scatter(y_pred, residuals, alpha=0.4, color="coral", s=20)

    # Horizontal line at 0 — residuals should be centered around this
    ax.axhline(y=0, color="black", linewidth=1.2, linestyle="--")

    ax.set_title(f"{name}\nResidual Plot")
    ax.set_xlabel("Predicted SalePrice (log)")
    ax.set_ylabel("Residuals (Actual - Predicted)")

plt.tight_layout()
plt.savefig("plots/residual_plots.png", dpi=150)
plt.show()
print("Residual plots saved.")

# ── 6. RESIDUAL DISTRIBUTION ──────────────────────────────────
# Residuals should follow a normal distribution centered at 0
# A bell curve shape confirms the model errors are random
# Skewed distribution = model is systematically wrong in one direction

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, model) in zip(axes, models.items()):
    y_pred    = model.predict(X_test)
    residuals = y_test - y_pred

    ax.hist(residuals, bins=30, color="steelblue",
            edgecolor="white", alpha=0.8)

    # Vertical line at 0 — distribution should be centered here
    ax.axvline(x=0, color="red", linewidth=1.5,
               linestyle="--", label="Zero error")

    # Print skewness on the plot
    skew = residuals.skew()
    ax.set_title(f"{name}\nResidual Distribution (skew={skew:.2f})")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plots/residual_distribution.png", dpi=150)
plt.show()
print("Residual distribution plots saved.")

# ── 7. INVERSE TRANSFORM RMSE TO DOLLAR VALUE ─────────────────
# As you correctly identified — RMSE is currently in log scale
# because we applied log1p to SalePrice during preprocessing.
# To interpret RMSE in actual dollars, we inverse transform
# using np.expm1() which is the reverse of np.log1p()
#
# log1p(x)  = log(1 + x)   → forward transform
# expm1(x)  = exp(x) - 1   → reverse transform

print("\n" + "=" * 50)
print("RMSE IN ACTUAL DOLLAR VALUE (after inverse transform)")
print("=" * 50)

for name, model in models.items():
    y_pred     = model.predict(X_test)

    # Inverse transform both actual and predicted back to dollar scale
    y_test_inv = np.expm1(y_test)
    y_pred_inv = np.expm1(y_pred)

    # RMSE in dollars — now interpretable as average prediction error
    rmse_dollars = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae_dollars  = mean_absolute_error(y_test_inv, y_pred_inv)

    print(f"\n{name}")
    print(f"  RMSE : ${rmse_dollars:,.0f}")
    print(f"  MAE  : ${mae_dollars:,.0f}")

# ── 8. CROSS VALIDATION SUMMARY ───────────────────────────────
# Final reliable comparison using CV RMSE
# As discussed — CV RMSE is more trustworthy than single test RMSE
# because it averages across 5 different splits

print("\n" + "=" * 50)
print("CROSS VALIDATION SUMMARY (5-fold CV RMSE)")
print("=" * 50)

cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )
    cv_rmse = -cv_scores.mean()
    cv_std  = -cv_scores.std()
    cv_results[name] = cv_rmse
    print(f"{name:<25} CV RMSE: {cv_rmse:.4f} (±{cv_std:.4f})")

# Best model by CV RMSE
best = min(cv_results, key=cv_results.get)
print(f"\nBest model by CV RMSE: {best}")

# ── 9. SUMMARY ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL EVALUATION SUMMARY")
print("  3 evaluation checks performed:")
print("  1. Overfitting check   → Train vs Test RMSE gap")
print("  2. Predicted vs Actual → Points close to diagonal?")
print("  3. Residual analysis   → Random scatter around 0?")
print("  4. Dollar RMSE         → Real world interpretation")
print("  5. CV RMSE             → Most reliable model ranking")
print(f"\n  Best model (CV RMSE)  : {best}")
print(f"  CV RMSE               : {cv_results[best]:.4f}")
print("=" * 50)
