# Model Card — House Price Prediction v1.0.0

## Overview

| | |
|---|---|
| **Model version** | v1.0.0 |
| **Model type** | XGBoost Regressor (hyperparameter-tuned) |
| **Trained** | 2026-05-03 *(inferred from artifact file timestamps — update if you have the actual training log date)* |
| **Task** | Regression — predicts house sale price in USD |
| **Training dataset** | Kaggle Ames Housing dataset (`train.csv`) |

## Performance

| Metric | Value | Meaning |
|---|---|---|
| R² Score | 0.9005 | The model explains ~90% of the variance in sale price |
| RMSE | $28,050 | Typical prediction error magnitude — used directly to compute the `price_range` returned by the API |

These metrics come from evaluation on a held-out test split during training (see `notebook/House_price_model_evaluation.py`), not from live production traffic. No production monitoring currently re-validates these numbers against real-world outcomes — see **Limitations** below.

## Features Used

The model was trained on 250 encoded features, derived from the following raw and engineered inputs (see `app/schemas.py` for the exact API input contract, and `app/pipeline.py` for the engineering logic):

**Engineered features:**
- `TotalSF` — combined basement + first floor + second floor square footage
- `HouseAge` — years between construction and sale
- `OverallScore` — average of overall quality and condition ratings
- `TotalBathrooms` — weighted count of full/half bathrooms above and below grade

**Key raw features** (top contributors by importance): `OverallQual`, `ExterQual`, `KitchenQual`, `TotalBsmtSF`, `GarageCars`, `Foundation`, `BsmtFinType1`, `MSZoning`, `CentralAir`, `GarageCond`.

Categorical features are one-hot encoded; skewed numerical features are log-transformed (`log1p`) prior to scaling. Full preprocessing logic lives in `app/pipeline.py` and is intentionally re-implemented there — the trained artifacts alone do not capture preprocessing steps.

## Intended Use

This model is intended to provide an **estimated price range** for single-family residential properties similar in profile to those in the Ames, Iowa training data (early-to-mid 2000s sales). It is a portfolio/demonstration project and is **not** intended for real financial, lending, insurance, or investment decisions.

## Limitations

- **Geographic and temporal scope**: trained exclusively on Ames, Iowa sales data. Predictions for other markets, or for the current real-world housing market, are extrapolations outside the model's training distribution and should not be trusted at face value.
- **No drift detection**: nothing in this system currently monitors whether live prediction inputs are drifting away from the training data's distribution over time.
- **No fairness/bias audit** has been performed on this model with respect to protected characteristics or proxy variables (e.g. zoning, neighborhood-correlated features).
- **Static artifacts**: this model does not retrain or update automatically. A new version (e.g. `v1.1.0`) would need to be trained and registered separately — see **Versioning** below.

## Versioning

Each model version lives in its own subfolder under `artifacts/`, e.g. `artifacts/v1.0.0/`, containing:
- The trained artifacts (`model.pkl`, `scaler.pkl`, `train_means.pkl`, `encoded_columns.pkl`, `skewed_columns.pkl`)
- `metadata.json` — machine-readable version, metrics, and framework versions, read directly by the API at startup and exposed via `/version` and `/health`
- `MODEL_CARD.md` — this file, documenting that specific version in human-readable form

The API selects which version to load via the `MODEL_VERSION` environment variable (defaults to `v1.0.0` if unset — see `app/main.py`). To deploy a new model version without touching code:
1. Train the new model and save its artifacts to `artifacts/v1.1.0/` (or whatever the next version is)
2. Write that version's `metadata.json` and `MODEL_CARD.md`
3. Set `MODEL_VERSION=v1.1.0` in the deployment environment
4. Restart the API — no code changes required

This is a lightweight, file-based registry — sufficient for a single-maintainer project. A team environment would typically use a dedicated tool (MLflow Model Registry, DVC, etc.) instead, which additionally handles experiment tracking, model lineage, and automated promotion between environments.
