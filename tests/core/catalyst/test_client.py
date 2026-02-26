"""Tests for Catalyst API client."""

from __future__ import annotations

from diagrid.core.auth.token import AuthContext
from diagrid.core.catalyst.client import CatalystClient


def test_client_headers_with_bearer_token() -> None:
    """Client uses Bearer token + x-diagrid-orgid when access_token is set."""
    ctx = AuthContext(
        api_url="https://api.diagrid.io",
        org_id="org-1",
        access_token="my-token",
    )
    client = CatalystClient(ctx)
    headers = client._headers()
    assert headers["Authorization"] == "Bearer my-token"
    assert headers["x-diagrid-orgid"] == "org-1"


def test_client_headers_with_api_key() -> None:
    """Client uses x-diagrid-api-key when api_key is set."""
    ctx = AuthContext(
        api_url="https://api.diagrid.io",
        org_id="org-1",
        api_key="my-api-key",
    )
    client = CatalystClient(ctx)
    headers = client._headers()
    assert headers["x-diagrid-api-key"] == "my-api-key"
    assert "Authorization" not in headers


def test_client_base_url() -> None:
    """Client builds correct base URL."""
    ctx = AuthContext(
        api_url="https://api.diagrid.io",
        org_id="org-1",
        access_token="token",
    )
    client = CatalystClient(ctx)
    assert client.base_url == "https://api.diagrid.io/apis/cra.diagrid.io/v1beta1"
