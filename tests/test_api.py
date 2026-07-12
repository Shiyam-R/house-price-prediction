# ============================================================
# TEST_API.PY — HTTP ENDPOINT TESTS
# ============================================================
# Exercises the actual FastAPI app through TestClient, which
# sends real HTTP-style requests through the full stack
# (routing, Pydantic validation, the prediction pipeline) rather
# than calling Python functions directly. This is what catches
# issues that only show up at the HTTP layer — wrong status
# codes, wrong methods allowed, response shape not matching the
# declared response_model.


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert "version" in body


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_version"] == "v1.0.0"
    assert "r2_score" in body
    assert "rmse_dollars" in body


def test_version_endpoint(client):
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert "api_version" in body
    assert body["model_version"] == "v1.0.0"
    assert "model_type" in body
    assert "trained_date" in body
    assert "metrics" in body
    assert "python_version" in body


def test_predict_with_valid_payload_returns_200(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_price" in body
    assert "price_range" in body
    assert "confidence" in body
    assert "model_used" in body
    assert "api_version" in body


def test_predict_with_invalid_categorical_returns_422(client, valid_payload):
    """
    Sending a category the model was never trained on (e.g. an
    MSZoning value outside the trained set) should be rejected by
    Pydantic before it ever reaches the model — not silently
    coerced into a wrong prediction.
    """
    bad_payload = valid_payload.copy()
    bad_payload["MSZoning"] = "NotARealZone"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_with_missing_field_returns_422(client, valid_payload):
    bad_payload = valid_payload.copy()
    del bad_payload["OverallQual"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_get_requests(client):
    """
    /predict only accepts POST. A GET request — e.g. from typing
    the URL directly into a browser address bar — must be rejected
    with 405 Method Not Allowed, not silently succeed or 404.
    This directly mirrors real behavior observed when manually
    testing the API.
    """
    response = client.get("/predict")
    assert response.status_code == 405