# ============================================================
# MAIN.PY — FASTAPI APPLICATION
# ============================================================
# This is the entry point of the API.
# It defines the endpoints, loads the pipeline once at startup,
# and connects user requests to the prediction pipeline.

import logging
import platform
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from .schemas import HouseFeatures, PredictionResponse
from .pipeline import PredictionPipeline
from .config import settings
from .drift_monitor import DriftMonitor

# ── LOGGING CONFIGURATION ───────────────────────────────────
# Configured once, at import time, so every module's logger
# (including pipeline.py's) inherits this format and level.
# Logs go to stdout by default — combined with the Dockerfile's
# PYTHONUNBUFFERED=1, they appear immediately in `docker logs`
# and in GitHub Actions output, with no extra setup needed.
#
# Format includes: timestamp, logger name, level, message —
# enough to trace what happened, when, and where, without
# needing a full observability stack for a project this size.
# Level is now driven by settings.log_level (overridable via
# LOG_LEVEL env var or .env), not hardcoded.
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"

# ── LIFESPAN — Load pipeline once at startup ──────────────────
# @asynccontextmanager runs code before and after the API starts
# We load all pkl files ONCE when API starts — not on every request
# This makes predictions fast — no disk reads per request

pipeline = None  # global variable to hold the pipeline
drift_monitor = None  # global variable to hold the drift monitor

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code here runs BEFORE the API starts accepting requests
    global pipeline, drift_monitor
    logger.info("Loading prediction pipeline — model version: %s", settings.model_version)
    pipeline = PredictionPipeline(artifacts_path=settings.artifacts_path)

    drift_monitor = DriftMonitor(
        baseline_path=f"{settings.artifacts_path}/baseline_stats.json",
        window_size=settings.drift_window_size,
        z_threshold=settings.drift_z_threshold,
    )
    logger.info("Drift monitor ready — tracking %d features, window size %d",
                len(drift_monitor.baseline), settings.drift_window_size)

    logger.info("API started — pipeline ready. API version %s, model version %s", API_VERSION, settings.model_version)
    yield
    # Code here runs AFTER the API shuts down (cleanup)
    logger.info("API shutting down.")

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

    See **/version** for the live, authoritative values for whichever
    model version is actually currently loaded — the figures above
    describe v1.0.0 and may not reflect a different deployed version.
    """,
    version=API_VERSION,
    lifespan=lifespan
)

# ── ENDPOINT 1 — HEALTH CHECK ─────────────────────────────────
# GET /health → quick check that API is running
# Used by Docker and CI/CD to verify the service is alive
# Returns 200 OK if everything is fine
#
# Deliberately NOT logged on every call: Docker's HEALTHCHECK
# hits this every 30 seconds, and logging each hit would bury
# actually meaningful log entries (predictions, errors) in noise.

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Returns status and model information, sourced live from the
    loaded model's metadata.json — not hardcoded — so this can
    never silently drift out of sync with the actual deployed model.
    Used by Docker healthcheck and CI/CD pipeline.
    """
    return {
        "status"     : "healthy",
        "model"      : pipeline.model_type,
        "model_version": pipeline.model_version,
        "r2_score"   : pipeline.metadata["metrics"]["r2_score"],
        "rmse_dollars": f"${pipeline.metadata['metrics']['rmse_dollars']:,}"
    }

# ── ENDPOINT 2 — VERSION ──────────────────────────────────────
# GET /version → reports exactly what's deployed.
# Useful for confirming CI, Docker, and local environments are
# aligned, and for verifying what build is actually live without
# guessing from a commit history.

@app.get("/version")
def version_info():
    """
    Version endpoint.
    Returns API version, model version and metrics, and runtime info.
    Model details are read live from the loaded metadata.json,
    giving true traceability: this always reflects exactly which
    trained model is actually serving predictions right now, not
    a value that has to be manually kept in sync in source code.
    """
    return {
        "api_version": API_VERSION,
        "model_version": pipeline.model_version,
        "model_type": pipeline.model_type,
        "trained_date": pipeline.metadata.get("trained_date"),
        "metrics": pipeline.metadata["metrics"],
        "python_version": platform.python_version()
    }


@app.get("/drift")
def drift_report():
    """
    Lightweight drift monitoring endpoint. Compares the rolling
    mean of recent /predict inputs against this model version's
    training-time baseline (artifacts/<version>/baseline_stats.json)
    for each tracked numeric feature, and flags features whose
    live mean has moved more than z_threshold baseline standard
    deviations away from the training mean.

    This is an in-memory, single-process signal — see
    app/drift_monitor.py for its deliberate scope and limitations.
    Intended as a lightweight "is something obviously off" check,
    not a replacement for a dedicated drift-monitoring system in
    a real multi-instance production deployment.
    """
    return drift_monitor.get_report()

# ── ENDPOINT 3 — PREDICTION ───────────────────────────────────
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
    # Log a compact summary of the request, not the full payload —
    # enough to trace request volume and rough input shape without
    # flooding logs with every field on every call.
    logger.info(
        "Prediction requested | OverallQual=%s TotalBsmtSF=%s YearBuilt=%s",
        features.OverallQual, features.TotalBsmtSF, features.YearBuilt
    )

    try:
        # Convert Pydantic model to dict
        # .model_dump() extracts all fields as a Python dictionary
        input_data = features.model_dump()

        # Record numeric feature values for drift monitoring. This
        # happens on every VALIDATED request (Pydantic already
        # confirmed the input is well-formed) — recording invalid
        # requests would pollute the drift signal with malformed
        # data rather than reflecting real traffic patterns.
        drift_monitor.record(input_data)

        # Run full prediction pipeline
        # pipeline was loaded once at startup — reused here
        result = pipeline.predict(input_data)

        # Attach API version here, at the serving layer — not inside
        # pipeline.py. The pipeline is pure ML logic and shouldn't need
        # to know what "API version" means; that's a web-layer concern.
        result["api_version"] = API_VERSION

        logger.info("Prediction completed | predicted_price=%s", result["predicted_price"])

        return result

    except Exception as e:
        # logger.exception() captures the full stack trace alongside
        # the message — far more useful for debugging than the bare
        # error string alone.
        logger.exception("Prediction failed")

        # If anything goes wrong in the pipeline
        # return a 500 error with the error message
        # HTTPException is FastAPI's way of returning HTTP errors
        # 500 = Internal Server Error
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# ── ENDPOINT 4 — ROOT ─────────────────────────────────────────
# GET / → welcome message
# Good practice to have a root endpoint that tells users
# what the API does and where to find the docs

@app.get("/")
def root():
    """
    Root endpoint — API information and links.

    Uses relative paths rather than a hardcoded host/port. The API
    doesn't actually know where it's being accessed from — it could
    be localhost, a Docker port mapping, a GitHub Codespaces forwarded
    URL, or a real cloud deployment later. Hardcoding "127.0.0.1:8000"
    would be wrong in every case except one specific local setup.
    Relative paths resolve correctly against whatever host the
    request actually came in on.
    """
    return {
        "message"  : "House Price Prediction API",
        "version"  : API_VERSION,
        "docs"     : "/docs",
        "health"   : "/health",
        "version_info": "/version",
        "drift"    : "/drift",
        "predict"  : "POST /predict"
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
# host/port below come from settings (HOST/PORT env vars or .env),
# defaulting to 0.0.0.0:8000 if unset.
# reload=True     → auto-restart when code changes (development only)
#                   never use reload=True in production/Docker

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )