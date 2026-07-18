# ============================================================
# CONFIG.PY — CENTRALIZED APPLICATION SETTINGS
# ============================================================
# 12-factor-app principle: configuration lives in the environment,
# not scattered as hardcoded values or ad-hoc os.environ.get() calls
# throughout the codebase. This file is the single place that
# declares every piece of external configuration the app depends
# on, with typed validation and sensible defaults.
#
# Every setting below has a default matching the app's existing
# behavior — a .env file is entirely OPTIONAL. It only exists to
# let you override a value locally (or in a deployment environment)
# without touching code or rebuilding the image.

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Which trained model version to load, e.g. "v1.0.0", "v1.1.0".
    # This is what makes model versioning actually operable: deploying
    # a new model becomes "set MODEL_VERSION and restart", not
    # "edit source code and rebuild the image".
    model_version: str = "v1.0.0"

    # Base directory containing versioned model artifact folders.
    # The full path used at runtime is artifacts_base_path/model_version,
    # e.g. "artifacts/v1.0.0" — see the artifacts_path property below.
    artifacts_base_path: str = "artifacts"

    # Logging verbosity. INFO is appropriate for normal operation;
    # DEBUG can be set locally (e.g. via .env) for troubleshooting
    # without needing a code change.
    log_level: str = "INFO"

    # Host/port used ONLY by the local-dev fallback path in main.py
    # (`python -m app.main`). When running via Docker, the CMD
    # instruction's own $PORT handling takes precedence instead —
    # see the Dockerfile for that path.
    host: str = "0.0.0.0"
    port: int = 8000

    # Drift monitor tuning. window_size = how many recent requests
    # to keep per tracked feature; z_threshold = how many baseline
    # standard deviations away from the training mean counts as
    # "drifting". Overridable so this can be tuned per-deployment
    # without a code change (e.g. a higher-traffic deployment might
    # want a larger window for a more stable signal).
    drift_window_size: int = 500
    drift_z_threshold: float = 2.0

    # Explicit thread pool size for offloading CPU-bound /predict
    # work, rather than relying on Starlette's hidden default
    # sync-endpoint thread pool cap (~40, undocumented in practice).
    # Default heuristic: 5x CPU count — numpy/pandas/xgboost release
    # the GIL during their actual C-level computation, so more
    # threads than raw core count can still yield real wall-clock
    # throughput up to genuine CPU saturation. Override via
    # PREDICT_THREAD_POOL_SIZE if empirically tuned differently for
    # your actual hardware — see load_tests/LOAD_TEST_RESULTS.md.
    predict_thread_pool_size: int = Field(default_factory=lambda: (os.cpu_count() or 1) * 5)

    # SettingsConfigDict tells pydantic-settings to also read from a
    # .env file if one exists in the working directory, in addition
    # to real environment variables (which always take priority over
    # .env file values if both are set).
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars rather than erroring
    )

    @property
    def artifacts_path(self) -> str:
        """
        The full, version-specific path to load model artifacts from,
        e.g. "artifacts/v1.0.0". Computed from model_version rather
        than stored separately, so the two values can never drift
        out of sync with each other.
        """
        return f"{self.artifacts_base_path}/{self.model_version}"


# A single, module-level instance — imported and reused everywhere
# settings are needed, rather than re-reading the environment
# repeatedly throughout the app.
settings = Settings()