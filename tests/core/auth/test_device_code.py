# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for device code auth flow."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest

from diagrid.core.auth.credentials import Credential
from diagrid.core.auth.device_code import DeviceCodeAuth
from diagrid.core.auth.token import AuthContext, TokenResponse
from diagrid.core.config.envs import EnvConfig


def _make_token(org_id: str = "org-123", exp: int = 9999999999) -> str:
    claims = {
        "sub": "auth0|user123",
        f"https://diagrid.io/org_{org_id}/roles": ["cra.diagrid.admin"],
        "exp": exp,
    }
    return pyjwt.encode(
        claims, "test-secret-key-that-is-long-enough", algorithm="HS256"
    )


def test_authenticate_with_api_key(tmp_path: pytest.TempPathFactory) -> None:
    """API key path should extract org and return immediately."""
    token = _make_token("my-org")
    auth = DeviceCodeAuth(api_key_flag=token)
    auth.config_store = MagicMock()
    auth.config_store.get.return_value = MagicMock(current_project_id="proj1")

    ctx = auth.authenticate()
    assert isinstance(ctx, AuthContext)
    assert ctx.org_id == "my-org"
    assert ctx.api_key == token


def test_authenticate_with_cached_credential(tmp_path: pytest.TempPathFactory) -> None:
    """Valid cached credential should be reused."""
    env = EnvConfig(
        apiUrl="https://api.example.com",
        authAudience="aud",
        authClientId="cid",
        issuerUrl="https://example.auth0.com/",
        authDomain="example.auth0.com",
    )
    tkn = TokenResponse(
        access_token="cached-token",
        refresh_token="refresh",
        expires_in=3600,
    )
    cred = Credential(
        subject="user",
        env=env,
        token_response=tkn,
        default_org="org-cached",
        expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )

    auth = DeviceCodeAuth()
    auth.cred_store = MagicMock()
    auth.cred_store.get.return_value = cred
    auth.config_store = MagicMock()
    auth.config_store.get.return_value = MagicMock(
        current_org_id="org-cached", current_project_id=""
    )

    ctx = auth.authenticate()
    assert ctx.access_token == "cached-token"
    assert ctx.org_id == "org-cached"


def test_device_code_uses_issuer_url_endpoint() -> None:
    """_request_device_code uses env.device_authorization_endpoint."""
    env = EnvConfig(
        api_url="https://api.r1.diagrid.io",
        auth_audience="admingrid-api",
        auth_client_id="cid",
        issuer_url="https://login.diagrid.io/",
        auth_domain="login.diagrid.io",
    )
    auth = DeviceCodeAuth()

    captured_url: list[str] = []

    def fake_post(url: str, **kwargs: object) -> MagicMock:
        captured_url.append(url)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "device_code": "dc",
            "user_code": "USER-CODE",
            "verification_uri": "https://login.diagrid.io/activate",
            "verification_uri_complete": "https://login.diagrid.io/activate?user_code=USER-CODE",
            "expires_in": 300,
            "interval": 5,
        }
        return resp

    with patch("diagrid.core.auth.device_code.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = fake_post
        mock_client_cls.return_value = mock_client

        auth._request_device_code(env)

    assert captured_url == ["https://login.diagrid.io/oauth/device/code"]


def test_refresh_token_uses_token_endpoint() -> None:
    """_refresh_token uses cred.env.token_endpoint."""
    env = EnvConfig(
        api_url="https://api.r1.diagrid.io",
        auth_audience="admingrid-api",
        auth_client_id="cid",
        issuer_url="https://login.diagrid.io/",
        auth_domain="login.diagrid.io",
    )
    tkn = TokenResponse(
        access_token=_make_token(),
        refresh_token="old-refresh",
        expires_in=3600,
    )
    from diagrid.core.auth.credentials import Credential
    from datetime import datetime, timezone

    cred = Credential(
        subject="user",
        env=env,
        token_response=tkn,
        default_org="org-1",
        expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
    )

    auth = DeviceCodeAuth()
    auth.cred_store = MagicMock()

    captured_url: list[str] = []

    def fake_post(url: str, **kwargs: object) -> MagicMock:
        captured_url.append(url)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "access_token": _make_token(),
            "expires_in": 3600,
        }
        return resp

    with patch("diagrid.core.auth.device_code.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = fake_post
        mock_client_cls.return_value = mock_client

        auth._refresh_token(cred)

    assert captured_url == ["https://login.diagrid.io/oauth/token"]
