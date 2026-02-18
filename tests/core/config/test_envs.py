"""Tests for env config fetching."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from diagrid.core.config.envs import EnvConfig, get_env_config


def test_env_config_from_json() -> None:
    """EnvConfig parses from JSON with Go-style field names."""
    data = {
        "apiUrl": "https://api.diagrid.io",
        "authAudience": "admingrid-api",
        "authClientId": "some-client-id",
        "authDomain": "diagrid.us.auth0.com",
    }
    env = EnvConfig.model_validate(data)
    assert env.api_url == "https://api.diagrid.io"
    assert env.auth_domain == "diagrid.us.auth0.com"
    assert env.auth_client_id == "some-client-id"


def test_env_config_populate_by_name() -> None:
    """EnvConfig can be constructed with Python-style names."""
    env = EnvConfig(
        api_url="https://api.example.com",
        auth_audience="aud",
        auth_client_id="cid",
        auth_domain="example.auth0.com",
    )
    assert env.api_url == "https://api.example.com"


@pytest.mark.anyio
async def test_get_env_config() -> None:
    """get_env_config fetches and parses env config."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "apiUrl": "https://api.diagrid.io",
        "authAudience": "aud",
        "authClientId": "cid",
        "authDomain": "domain",
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=mock_response)

    env = await get_env_config(mock_client, "https://api.diagrid.io")
    assert env.api_url == "https://api.diagrid.io"
    assert env.auth_audience == "aud"
