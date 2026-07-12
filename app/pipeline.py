# ============================================================
# PIPELINE.PY — FEATURE ENGINEERING + PREPROCESSING LOGIC
# ============================================================
# This class handles all transformations needed to convert
# raw user input into a format the model can predict on.
# Keeping this separate from main.py follows the principle
# of separation of concerns — each file has one clear job.

import pandas as pd
import numpy as np
import joblib
import logging

# Module-level logger — inherits configuration (level, format,
# handlers) from whatever is set up in main.py at startup, so
# this file doesn't need to configure logging itself.
logger = logging.getLogger(__name__)

class PredictionPipeline:

    def __init__(self, artifacts_path: str = "artifacts"):
        """
        Loads all saved artifacts when the pipeline is created.
        This happens ONCE when the API starts — not on every request.
        Loading from disk on every request would make the API slow.
        """
        self.model = joblib.load(f"{artifacts_path}/model.pkl")
        self.scaler = joblib.load(f"{artifacts_path}/scaler.pkl")
        self.train_means = joblib.load(f"{artifacts_path}/train_means.pkl")
        self.encoded_columns = joblib.load(f"{artifacts_path}/encoded_columns.pkl")
        self.skewed_columns = joblib.load(f"{artifacts_path}/skewed_columns.pkl")

        # Model RMSE in dollars — used to calculate price range
        self.model_rmse_dollars = 28050

        logger.info("Pipeline loaded successfully — model, scaler, and encoding artifacts ready.")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the same feature engineering we did during training.
        Must match EXACTLY — any difference causes wrong predictions.
        """
        # TotalSF — total living area across all floors
        df["TotalSF"] = (
            df["TotalBsmtSF"] +
            df["FirstFlrSF"] +
            df["SecondFlrSF"]
        )

        # HouseAge — age of house at time of sale
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

        # OverallScore — average of quality and condition
        df["OverallScore"] = (df["OverallQual"] + df["OverallCond"]) / 2

        # TotalBathrooms — weighted sum of all bathrooms
        df["TotalBathrooms"] = (
            df["FullBath"] +
            df["HalfBath"] * 0.5 +
            df["BsmtFullBath"] +
            df["BsmtHalfBath"] * 0.5
        )

        return df

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies one hot encoding to categorical columns.
        Then reindexes to match the exact column structure
        the model was trained on — missing columns filled with 0.
        """
        # Apply get_dummies — same as training
        # This creates binary columns from categorical values
        df = pd.get_dummies(df)

        # Reindex to match training columns exactly
        # columns that exist in training but not in user input → 0
        # columns in user input but not in training → dropped
        df = df.reindex(columns=self.encoded_columns, fill_value=0)

        return df

    def apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies log1p to the same skewed columns that were
        transformed during training. Must match exactly.
        Only transforms columns that exist in user input.
        """
        # Filter to only skewed columns that exist in current df
        cols_to_transform = [
            col for col in self.skewed_columns
            if col in df.columns
        ]
        df[cols_to_transform] = np.log1p(df[cols_to_transform])

        return df

    def fill_missing_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills only missing NUMERICAL columns with training means.
        Encoded categorical columns handled separately by reindex.
        """

        for col in self.encoded_columns:
            if col not in df.columns and "_" not in col:
                df[col] = self.train_means.get(col, 0)

        df = df[self.encoded_columns]

        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scales features using the saved StandardScaler.
        Uses transform() NOT fit_transform() — we never refit
        the scaler on new data. We only apply the training statistics.
        """
        scaled = self.scaler.transform(df)
        return pd.DataFrame(scaled, columns=df.columns)

    def predict(self, input_data: dict) -> dict:
        """
        Full prediction pipeline — takes raw user input dict,
        runs all transformations, returns prediction with price range.

        Flow:
        raw input → feature engineering → encoding →
        log transform → fill missing → scale → predict → inverse transform
        """
        # Convert input dict to DataFrame — pipeline expects DataFrame
        df = pd.DataFrame([input_data])

        # Step 1 — Feature engineering
        df = self.engineer_features(df)

        # Step 2 — One hot encoding
        df = self.encode_features(df)

        # Step 3 — Log transform skewed features
        df = self.apply_log_transform(df)

        # Step 4 — Fill missing columns with training means
        df = self.fill_missing_numerical(df)

        # Step 5 — Scale features
        df = self.scale_features(df)

        # Step 6 — Predict (returns log transformed price)
        log_prediction = self.model.predict(df)[0] # for make it an interger or string instead of a array

        # Step 7 — Inverse transform log prediction to dollar value
        # expm1 is the reverse of log1p
        predicted_price = np.expm1(log_prediction)

        # Step 8 — Calculate price range using model RMSE
        low  = max(0, predicted_price - self.model_rmse_dollars)
        high = predicted_price + self.model_rmse_dollars

        # Step 9 — Determine confidence based on price range width
        # R² of 0.9005 means model explains 90% of variance
        confidence = "high" if predicted_price > 100000 else "medium"

        return {
            "predicted_price": f"${predicted_price:,.0f}",
            "price_range": {
                "low" : f"${low:,.0f}",
                "high": f"${high:,.0f}"
            },
            "confidence": confidence,
            "model_used": "XGBoost (tuned)"
        }