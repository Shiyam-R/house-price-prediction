# ============================================================
# CONFTEST.PY — SHARED PYTEST FIXTURES
# ============================================================
# pytest automatically discovers this file and makes every
# fixture defined here available to all test files in this
# folder, with no imports needed in the test files themselves.

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pipeline import PredictionPipeline


@pytest.fixture(scope="session")
def client():
    """
    FastAPI TestClient wrapped in a `with` block so the app's
    lifespan (startup/shutdown) events actually run — this is
    what loads the real model artifacts before any test executes,
    exactly as they load when the app starts for real.

    scope="session" — the pipeline loads ONCE for the entire test
    run, not once per test. Artifact loading takes real time
    (loading model.pkl, scaler.pkl, etc.), so re-running it per
    test would make the suite unnecessarily slow for no benefit.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def pipeline():
    """
    A standalone PredictionPipeline instance, used to test
    pipeline logic directly without going through the API layer.
    Also session-scoped for the same reason as `client` above.
    Points at the same versioned artifacts path the API itself
    would load by default (v1.0.0), keeping this fixture in sync
    with real deployment behavior.
    """
    return PredictionPipeline(artifacts_path="artifacts/v1.0.0")


@pytest.fixture
def valid_payload():
    """
    A known-valid house feature payload. This is the same example
    already defined in schemas.py's json_schema_extra and used in
    the README — keeping one canonical example instead of inventing
    a different one here avoids tests, docs, and API examples
    silently drifting out of sync with each other.
    """
    return {
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
        "BsmtHalfBath": 0,
    }