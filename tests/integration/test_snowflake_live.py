from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kg_processor.application.snowflake_access import run_snowflake_access_check
from kg_processor.config.preflight import run_preflight
from kg_processor.config.settings import Settings

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("KG_RUN_SNOWFLAKE_LIVE") != "1",
        reason="Set KG_RUN_SNOWFLAKE_LIVE=1 to run live Snowflake integration checks.",
    ),
]

_CONFIG = Path("configs/snowflake-cortex.yaml")
_REQUIRED_ENV = (
    "KG_JOB_LEASE_OWNER",
    "KG_SNOWFLAKE_ACCOUNT",
    "KG_SNOWFLAKE_DATABASE",
    "KG_SNOWFLAKE_SCHEMA",
    "KG_SNOWFLAKE_STAGE",
    "KG_SNOWFLAKE_BULK_STAGE",
    "KG_SNOWFLAKE_IMAGE_REPOSITORY",
    "KG_SNOWFLAKE_COMPUTE_POOL",
    "KG_SNOWFLAKE_SERVICE_SPEC_STAGE",
    "KG_LLM_MODEL",
    "KG_EMBED_MODEL",
    "KG_EMBED_DIM",
)


def test_live_snowflake_config_preflight_accepts_resolved_environment() -> None:
    settings = _live_settings()

    result = run_preflight(settings)

    assert result.ok, json.dumps(result.model_dump(mode="json"), indent=2)


def test_live_snowflake_access_check_reports_ready_account_objects() -> None:
    settings = _live_settings()
    check_cortex = os.getenv("KG_SNOWFLAKE_LIVE_CHECK_CORTEX", "1") != "0"

    report = run_snowflake_access_check(settings, check_cortex=check_cortex)

    assert report.ok, json.dumps(report.model_dump(mode="json"), indent=2)


def _live_settings() -> Settings:
    missing = [name for name in _REQUIRED_ENV if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        pytest.skip(f"Live Snowflake test env is incomplete. Missing: {joined}")
    return Settings.load(_CONFIG, env=dict(os.environ))
