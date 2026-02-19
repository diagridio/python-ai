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


# --- issuerUrl / endpoint property tests ---


def test_env_config_with_issuer_url() -> None:
    """EnvConfig parses issuerUrl from the current API response format."""
    data = {
        "apiUrl": "https://api.r1.diagrid.io",
        "authAudience": "admingrid-api",
        "authClientId": "3zSEje4lkQOPLMWYEYFsgOMNDDo0AHGA",
        "issuerUrl": "https://login.diagrid.io/",
        "authDomain": "login.diagrid.io",
    }
    env = EnvConfig.model_validate(data)
    assert env.issuer_url == "https://login.diagrid.io/"
    assert env.auth_domain == "login.diagrid.io"


def test_env_config_device_authorization_endpoint_from_issuer_url() -> None:
    """device_authorization_endpoint uses issuerUrl when present."""
    env = EnvConfig(
        api_url="https://api.r1.diagrid.io",
        auth_audience="aud",
        auth_client_id="cid",
        issuer_url="https://login.diagrid.io/",
        auth_domain="login.diagrid.io",
    )
    assert (
        env.device_authorization_endpoint
        == "https://login.diagrid.io/oauth/device/code"
    )


def test_env_config_device_authorization_endpoint_fallback_to_auth_domain() -> None:
    """device_authorization_endpoint falls back to authDomain when issuerUrl is absent."""
    env = EnvConfig(
        api_url="https://api.diagrid.io",
        auth_audience="aud",
        auth_client_id="cid",
        auth_domain="diagrid.us.auth0.com",
    )
    assert (
        env.device_authorization_endpoint
        == "https://diagrid.us.auth0.com/oauth/device/code"
    )


def test_env_config_token_endpoint_from_issuer_url() -> None:
    """token_endpoint uses issuerUrl when present."""
    env = EnvConfig(
        api_url="https://api.r1.diagrid.io",
        auth_audience="aud",
        auth_client_id="cid",
        issuer_url="https://login.diagrid.io/",
        auth_domain="login.diagrid.io",
    )
    assert env.token_endpoint == "https://login.diagrid.io/oauth/token"


def test_env_config_token_endpoint_fallback_to_auth_domain() -> None:
    """token_endpoint falls back to authDomain when issuerUrl is absent."""
    env = EnvConfig(
        api_url="https://api.diagrid.io",
        auth_audience="aud",
        auth_client_id="cid",
        auth_domain="diagrid.us.auth0.com",
    )
    assert env.token_endpoint == "https://diagrid.us.auth0.com/oauth/token"


def test_env_config_issuer_url_trailing_slash_stripped() -> None:
    """issuerUrl with trailing slash produces clean endpoint URLs."""
    env = EnvConfig(
        api_url="https://api.r1.diagrid.io",
        auth_audience="aud",
        auth_client_id="cid",
        issuer_url="https://login.diagrid.io/",
    )
    assert (
        env.device_authorization_endpoint
        == "https://login.diagrid.io/oauth/device/code"
    )
    assert env.token_endpoint == "https://login.diagrid.io/oauth/token"
