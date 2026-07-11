# ============================================================
# SCHEMAS.PY — INPUT/OUTPUT STRUCTURE AND VALIDATION
# ============================================================
# Pydantic models define exactly what the API expects to receive
# and what it will send back. FastAPI uses these automatically
# for validation — wrong types or missing fields return a
# clear 422 error before any processing happens.

from pydantic import BaseModel, Field, field_validator
from typing import Literal

# ── INPUT SCHEMA ─────────────────────────────────────────────
# These are the raw features the user sends.
# Based on top 15 feature importances from tuned XGBoost.
# Categorical fields use Literal to restrict allowed values —
# only values the model was trained on are accepted.

class HouseFeatures(BaseModel):

    # ── Categorical Features ──────────────────────────────────
    # Literal restricts input to only valid categories
    # If user sends "Good" instead of "Gd" → 422 error immediately

    ExterQual: Literal["Ex", "Gd", "TA", "Fa", "Po"] = Field(
        description="Exterior material quality: Ex=Excellent, Gd=Good, TA=Average, Fa=Fair, Po=Poor"
    )

    KitchenQual: Literal["Ex", "Gd", "TA", "Fa", "Po"] = Field(
        description="Kitchen quality: Ex=Excellent, Gd=Good, TA=Average, Fa=Fair, Po=Poor"
    )

    CentralAir: Literal["Y", "N"] = Field(
        description="Central air conditioning: Y=Yes, N=No"
    )

    GarageCond: Literal["Ex", "Gd", "TA", "Fa", "Po", "None"] = Field(
        description="Garage condition: Ex=Excellent, Gd=Good, TA=Average, Fa=Fair, Po=Poor, None=No Garage"
    )

    Foundation: Literal["BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"] = Field(
        description="Foundation type"
    )

    BsmtFinType1: Literal["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "None"] = Field(
        description="Basement finished area quality. None=No Basement"
    )

    MSZoning: Literal["A", "C", "FV", "I", "RH", "RL", "RP", "RM"] = Field(
        description="General zoning classification"
    )

    # ── Numerical Features (raw — used for feature engineering) ──
    # ge = greater than or equal to (minimum value validation)
    # le = less than or equal to (maximum value validation)

    TotalBsmtSF: float = Field(
        ge=0,
        description="Total square feet of basement area"
    )

    FirstFlrSF: float = Field(
        ge=0,
        description="First floor square feet"
    )

    SecondFlrSF: float = Field(
        ge=0,
        description="Second floor square feet"
    )

    YearBuilt: int = Field(
        ge=1800, le=2026,
        description="Original construction year"
    )

    YrSold: int = Field(
        ge=2006, le=2026,
        description="Year the house was sold"
    )

    OverallQual: int = Field(
        ge=1, le=10,
        description="Overall material and finish quality (1=Very Poor, 10=Very Excellent)"
    )

    OverallCond: int = Field(
        ge=1, le=10,
        description="Overall condition rating (1=Very Poor, 10=Very Excellent)"
    )

    GarageCars: int = Field(
        ge=0, le=5,
        description="Size of garage in car capacity"
    )

    Fireplaces: int = Field(
        ge=0, le=5,
        description="Number of fireplaces"
    )

    KitchenAbvGr: int = Field(
        ge=0, le=5,
        description="Number of kitchens above grade"
    )

    FullBath: int = Field(
        ge=0, le=5,
        description="Full bathrooms above grade"
    )

    HalfBath: int = Field(
        ge=0, le=3,
        description="Half bathrooms above grade"
    )

    BsmtFullBath: int = Field(
        ge=0, le=3,
        description="Basement full bathrooms"
    )

    BsmtHalfBath: int = Field(
        ge=0, le=2,
        description="Basement half bathrooms"
    )

    # ── Custom Validator ──────────────────────────────────────
    # YearBuilt must be before YrSold — a house cannot be sold
    # before it was built. This cross-field validation catches
    # logical errors that type checking alone cannot catch.

    @field_validator("YrSold")
    @classmethod
    def yr_sold_after_built(cls, v, info):
        if "YearBuilt" in info.data and v < info.data["YearBuilt"]:
            raise ValueError("YrSold must be greater than or equal to YearBuilt")
        return v

    # Example valid input for testing:
    model_config = {
        "json_schema_extra": {
            "example": {
                "ExterQual": "Gd",
                "KitchenQual": "TA",
                "CentralAir": "Y",
                "GarageCond": "TA",
                "Foundation": "PConc",
                "BsmtFinType1": "GLQ",
                "MSZoning": "RL",
                "TotalBsmtSF": 856.0,
                "FirstFlrSF": 856.0,
                "SecondFlrSF": 854.0,
                "YearBuilt": 2003,
                "YrSold": 2008,
                "OverallQual": 7,
                "OverallCond": 5,
                "GarageCars": 2,
                "Fireplaces": 0,
                "KitchenAbvGr": 1,
                "FullBath": 2,
                "HalfBath": 1,
                "BsmtFullBath": 1,
                "BsmtHalfBath": 0
            }
        }
    }


# ── OUTPUT SCHEMA ─────────────────────────────────────────────
# Defines exactly what the API sends back to the user.
# Price range accounts for model RMSE of $28,050
# Low  = predicted price - RMSE
# High = predicted price + RMSE

class PredictionResponse(BaseModel):

    predicted_price: str = Field(
        description="Predicted house price in dollars"
    )

    price_range: dict = Field(
        description="Confidence range based on model RMSE of $28,050"
    )

    confidence: str = Field(
        description="Model confidence level based on R² score of 0.9005"
    )

    model_used: str = Field(
        description="Model used for prediction"
    )
