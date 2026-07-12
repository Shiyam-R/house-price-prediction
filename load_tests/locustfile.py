# ============================================================
# LOCUSTFILE.PY — LOAD TEST FOR HOUSE PRICE PREDICTION API
# ============================================================
# Simulates realistic traffic against a RUNNING instance of the
# API (local uvicorn, a Docker container, or a deployed instance —
# this file makes no assumptions about how the app is hosted).
#
# Usage:
#   locust -f load_tests/locustfile.py --host http://localhost:8000
#   Then open http://localhost:8089 for the interactive web UI.
#
# Or headless (no browser, prints a summary and exits):
#   locust -f load_tests/locustfile.py --host http://localhost:8000 \
#       --headless -u 50 -r 10 -t 60s
#   (-u = concurrent users, -r = users spawned per second, -t = duration)

import random
from locust import HttpUser, task, between


def random_valid_payload() -> dict:
    """
    Builds a randomized but SCHEMA-VALID request every call, using
    the same bounds enforced in app/schemas.py. Deliberately varied
    rather than one fixed payload repeated forever — a load test
    that always sends identical input tests an artificially easy
    case and can hide real-world performance characteristics.
    """
    return {
        "ExterQual": random.choice(["Ex", "Gd", "TA", "Fa", "Po"]),
        "KitchenQual": random.choice(["Ex", "Gd", "TA", "Fa", "Po"]),
        "CentralAir": random.choice(["Y", "N"]),
        "GarageCond": random.choice(["Ex", "Gd", "TA", "Fa", "Po", "None"]),
        "Foundation": random.choice(["PConc", "CBlock", "BrkTil", "Slab", "Stone", "Wood"]),
        "BsmtFinType1": random.choice(["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "None"]),
        "MSZoning": random.choice(["A", "C", "FV", "I", "RH", "RL", "RP", "RM"]),
        "TotalBsmtSF": round(random.uniform(0, 3000), 1),
        "FirstFlrSF": round(random.uniform(400, 3000), 1),
        "SecondFlrSF": round(random.uniform(0, 2000), 1),
        "YearBuilt": random.randint(1900, 2010),
        "YrSold": random.randint(2006, 2010),
        "OverallQual": random.randint(1, 10),
        "OverallCond": random.randint(1, 9),
        "GarageCars": random.randint(0, 4),
        "Fireplaces": random.randint(0, 3),
        "KitchenAbvGr": random.randint(1, 2),
        "FullBath": random.randint(0, 3),
        "HalfBath": random.randint(0, 2),
        "BsmtFullBath": random.randint(0, 2),
        "BsmtHalfBath": random.randint(0, 2),
    }


class HousePricePredictionUser(HttpUser):
    # Simulates a real user pausing briefly between actions,
    # rather than firing requests in an unrealistic tight loop.
    wait_time = between(0.5, 2.0)

    @task(10)
    def predict(self):
        """
        The primary, expensive operation — weighted heavily since
        this is what real traffic to this API would mostly consist
        of. Uses a fresh randomized payload on every call.
        """
        payload = random_valid_payload()
        # YrSold must be >= YearBuilt per the schema's cross-field
        # validator — enforce that here too so generated requests
        # don't spuriously fail validation for reasons unrelated to
        # load performance.
        if payload["YrSold"] < payload["YearBuilt"]:
            payload["YrSold"] = payload["YearBuilt"]

        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status {response.status_code}: {response.text[:200]}")

    @task(1)
    def health(self):
        """Infrequent — mirrors how rarely a real orchestrator polls /health relative to actual traffic."""
        self.client.get("/health")

    @task(1)
    def version(self):
        """Infrequent — occasional monitoring/traceability checks, not core traffic."""
        self.client.get("/version")
