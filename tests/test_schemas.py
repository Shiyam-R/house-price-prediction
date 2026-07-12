# ============================================================
# TEST_SCHEMAS.PY — INPUT VALIDATION TESTS
# ============================================================
# Verifies that HouseFeatures correctly accepts valid input and
# rejects invalid input BEFORE it ever reaches the model. This is
# the first line of defense against bad predictions — catching
# these errors here is much cheaper than debugging a wrong
# prediction caused by silently-accepted bad data.

import pytest
from pydantic import ValidationError

from app.schemas import HouseFeatures


def test_valid_payload_is_accepted(valid_payload):
    """A known-good payload should construct without error."""
    features = HouseFeatures(**valid_payload)
    assert features.OverallQual == 7
    assert features.ExterQual == "Gd"


def test_invalid_categorical_value_is_rejected(valid_payload):
    """
    Literal fields must reject values outside the trained
    categories. "Great" was never a valid ExterQual label —
    only Ex/Gd/TA/Fa/Po exist in the training data.
    """
    bad_payload = valid_payload.copy()
    bad_payload["ExterQual"] = "Great"
    with pytest.raises(ValidationError):
        HouseFeatures(**bad_payload)


def test_yr_sold_before_year_built_is_rejected(valid_payload):
    """
    Cross-field validator: a house cannot be sold before it
    was built. This exercises the custom @field_validator,
    not just basic type checking.
    """
    bad_payload = valid_payload.copy()
    bad_payload["YearBuilt"] = 2010
    bad_payload["YrSold"] = 2008
    with pytest.raises(ValidationError):
        HouseFeatures(**bad_payload)


def test_yr_sold_equal_to_year_built_is_accepted(valid_payload):
    """
    Boundary case: a house sold the same year it was built
    is valid (validator uses >=, not strictly >).
    """
    payload = valid_payload.copy()
    payload["YearBuilt"] = 2008
    payload["YrSold"] = 2008
    features = HouseFeatures(**payload)
    assert features.YrSold == features.YearBuilt


@pytest.mark.parametrize("field,value", [
    ("OverallQual", 11),   # max allowed is 10
    ("OverallQual", 0),    # min allowed is 1
    ("GarageCars", 6),     # max allowed is 5
    ("BsmtHalfBath", 3),   # max allowed is 2
    ("TotalBsmtSF", -100), # must be >= 0
])
def test_numeric_bounds_are_enforced(valid_payload, field, value):
    """
    Each of these values sits just outside the field's declared
    ge/le bounds in schemas.py and must be rejected.
    """
    bad_payload = valid_payload.copy()
    bad_payload[field] = value
    with pytest.raises(ValidationError):
        HouseFeatures(**bad_payload)


def test_missing_required_field_is_rejected(valid_payload):
    """Omitting a required field entirely must fail validation."""
    bad_payload = valid_payload.copy()
    del bad_payload["OverallQual"]
    with pytest.raises(ValidationError):
        HouseFeatures(**bad_payload)
