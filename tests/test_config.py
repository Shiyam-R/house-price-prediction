# ============================================================
# TEST_CONFIG.PY — SETTINGS / ENVIRONMENT CONFIGURATION TESTS
# ============================================================
# Verifies the 12-factor config pattern actually works: sensible
# defaults with no environment set, and correct overriding when
# environment variables ARE set. _env_file=None is used throughout
# so these tests reflect real environment variables only, not
# whatever .env file might happen to exist on the machine running
# the tests.

import os

from app.config import Settings


def test_defaults_match_existing_behavior(monkeypatch):
    """
    With no environment variables set, Settings should produce
    exactly the values the app used before config.py existed —
    this is what guarantees introducing configuration didn't
    silently change default behavior for anyone not using .env.
    """
    for var in ["MODEL_VERSION", "ARTIFACTS_BASE_PATH", "LOG_LEVEL", "HOST", "PORT"]:
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)

    assert settings.model_version == "v1.0.0"
    assert settings.artifacts_base_path == "artifacts"
    assert settings.artifacts_path == "artifacts/v1.0.0"
    assert settings.log_level == "INFO"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000


def test_model_version_env_var_overrides_default(monkeypatch):
    """
    Setting MODEL_VERSION should change both the reported version
    and the derived artifacts_path together — proving the property
    is actually computed from model_version, not independently
    hardcoded elsewhere.
    """
    monkeypatch.setenv("MODEL_VERSION", "v2.0.0")
    settings = Settings(_env_file=None)

    assert settings.model_version == "v2.0.0"
    assert settings.artifacts_path == "artifacts/v2.0.0"


def test_port_env_var_is_read_as_int(monkeypatch):
    """
    PORT is declared as an int field — pydantic-settings should
    coerce the string environment variable into an actual int,
    not leave it as a string that would break uvicorn.run(port=...).
    """
    monkeypatch.setenv("PORT", "9000")
    settings = Settings(_env_file=None)

    assert settings.port == 9000
    assert isinstance(settings.port, int)