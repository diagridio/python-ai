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
