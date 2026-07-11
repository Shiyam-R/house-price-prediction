# ============================================================
# MAIN.PY — FASTAPI APPLICATION
# ============================================================
# This is the entry point of the API.
# It defines the endpoints, loads the pipeline once at startup,
# and connects user requests to the prediction pipeline.

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from .schemas import HouseFeatures, PredictionResponse
from .pipeline import PredictionPipeline

# ── LIFESPAN — Load pipeline once at startup ──────────────────
# @asynccontextmanager runs code before and after the API starts
# We load all pkl files ONCE when API starts — not on every request
# This makes predictions fast — no disk reads per request

pipeline = None  # global variable to hold the pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code here runs BEFORE the API starts accepting requests
    global pipeline
    pipeline = PredictionPipeline(artifacts_path="artifacts")
    print("API started — pipeline ready.")
    yield
    # Code here runs AFTER the API shuts down (cleanup)
    print("API shutting down.")

# ── CREATE FASTAPI APP ────────────────────────────────────────
# title, description, version appear in the auto-generated docs
# at http://127.0.0.1:8000/docs

app = FastAPI(
    title="House Price Prediction API",
    description="""
    Predicts house sale prices using a tuned XGBoost model
    trained on the Ames Housing dataset.

    ## How to use
    Send a POST request to **/predict** with house features.
    The API handles all preprocessing and returns a predicted price.

    ## Model Performance
    - R² Score : 0.9005
    - RMSE     : $28,050
    """,
    version="1.0.0",
    lifespan=lifespan
)

# ── ENDPOINT 1 — HEALTH CHECK ─────────────────────────────────
# GET /health → quick check that API is running
# Used by Docker and CI/CD to verify the service is alive
# Returns 200 OK if everything is fine

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Returns status and model information.
    Used by Docker healthcheck and CI/CD pipeline.
    """
    return {
        "status"     : "healthy",
        "model"      : "XGBoost (tuned)",
        "r2_score"   : 0.9005,
        "rmse_dollars": "$28,050"
    }

# ── ENDPOINT 2 — PREDICTION ───────────────────────────────────
# POST /predict → main endpoint
# Receives HouseFeatures (validated by Pydantic automatically)
# Returns PredictionResponse

@app.post("/predict", response_model=PredictionResponse)
def predict_price(features: HouseFeatures):
    """
    Predicts house sale price from raw house features.

    **Input:** Raw house features (see schema for valid values)
    **Output:** Predicted price with confidence range

    The API handles internally:
    - Feature engineering (TotalSF, HouseAge, OverallScore, TotalBathrooms)
    - One hot encoding of categorical features
    - Log transformation of skewed features
    - Standard scaling
    - Prediction and inverse transformation to dollar value
    """
    try:
        # Convert Pydantic model to dict
        # .model_dump() extracts all fields as a Python dictionary
        input_data = features.model_dump()

        # Run full prediction pipeline
        # pipeline was loaded once at startup — reused here
        result = pipeline.predict(input_data)

        return result

    except Exception as e:
        # If anything goes wrong in the pipeline
        # return a 500 error with the error message
        # HTTPException is FastAPI's way of returning HTTP errors
        # 500 = Internal Server Error
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# ── ENDPOINT 3 — ROOT ─────────────────────────────────────────
# GET / → welcome message
# Good practice to have a root endpoint that tells users
# what the API does and where to find the docs

@app.get("/")
def root():
    """
    Root endpoint — API information and links.
    """
    return {
        "message"  : "House Price Prediction API",
        "version"  : "1.0.0",
        "docs"     : "http://127.0.0.1:8000/docs",
        "health"   : "http://127.0.0.1:8000/health",
        "predict"  : "POST http://127.0.0.1:8000/predict"
    }

# ── RUN THE API (local dev fallback only) ──────────────────────
# This block only runs when you execute this file directly.
# NOTE: with app/ now a package, direct execution requires
# running it as a module from the project root:
#     python -m app.main
# For actual local development and for Docker, the standard way
# to run this app is from the project root:
#     uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# uvicorn is the ASGI server that runs FastAPI applications.
# host="0.0.0.0"  → accessible from any network interface
#                   needed for Docker to expose the port
# host="127.0.0.1"→ localhost only (development)
# port=8000       → the port the API listens on
# reload=True     → auto-restart when code changes (development only)
#                   never use reload=True in production/Docker

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )