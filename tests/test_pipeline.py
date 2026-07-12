# ============================================================
# TEST_PIPELINE.PY — PREDICTION PIPELINE TESTS
# ============================================================
# Tests the PredictionPipeline class directly against the real
# trained artifacts (model.pkl, scaler.pkl, etc.) — not mocked.
# This deliberately does NOT re-check individual transformation
# steps (feature engineering formulas, encoding, scaling) since
# those are implementation details already covered by producing
# a correct end-to-end prediction. What matters here is the
# pipeline's observable contract: given valid input, does it
# return a well-formed, sane prediction.

import pytest


def _to_float(dollar_string: str) -> float:
    """Helper: '$145,236' -> 145236.0"""
    return float(dollar_string.replace("$", "").replace(",", ""))


def test_predict_returns_expected_keys(pipeline, valid_payload):
    result = pipeline.predict(valid_payload)

    assert set(result.keys()) == {
        "predicted_price", "price_range", "confidence", "model_used"
    }
    assert set(result["price_range"].keys()) == {"low", "high"}
    assert result["model_used"] == "XGBoost (tuned)"


def test_predicted_price_is_positive(pipeline, valid_payload):
    result = pipeline.predict(valid_payload)
    price = _to_float(result["predicted_price"])
    assert price > 0


def test_price_range_brackets_the_prediction(pipeline, valid_payload):
    """
    low should be below the predicted price, high should be
    above it, and the total width should match 2x the model's
    known RMSE ($28,050) as defined in pipeline.py.
    """
    result = pipeline.predict(valid_payload)
    price = _to_float(result["predicted_price"])
    low = _to_float(result["price_range"]["low"])
    high = _to_float(result["price_range"]["high"])

    assert low < price < high
    assert high - low == pytest.approx(2 * 28050, rel=0.01)


def test_prediction_is_deterministic(pipeline, valid_payload):
    """
    The same input should always produce the same prediction —
    no hidden randomness should affect inference at serving time.
    """
    result_1 = pipeline.predict(valid_payload)
    result_2 = pipeline.predict(valid_payload)
    assert result_1["predicted_price"] == result_2["predicted_price"]


def test_higher_overall_quality_increases_predicted_price(pipeline, valid_payload):
    """
    Sanity check on model direction, not exact values: holding
    everything else constant, a higher OverallQual (material/finish
    quality) should not produce a LOWER predicted price. This is a
    loose directional check, not a claim about magnitude — it exists
    to catch a badly broken pipeline (e.g. wrong column order after
    encoding) rather than to validate the model's precision.
    """
    low_quality = valid_payload.copy()
    low_quality["OverallQual"] = 3

    high_quality = valid_payload.copy()
    high_quality["OverallQual"] = 9

    price_low = _to_float(pipeline.predict(low_quality)["predicted_price"])
    price_high = _to_float(pipeline.predict(high_quality)["predicted_price"])

    assert price_high > price_low
