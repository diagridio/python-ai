"""Tests for credential store."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from diagrid.core.auth.credentials import Credential, FileCredentialStore
from diagrid.core.auth.token import TokenResponse
from diagrid.core.config.envs import EnvConfig


def test_credential_roundtrip(tmp_path: Path) -> None:
    """Credential can be stored and loaded from file."""
    store = FileCredentialStore(tmp_path / "creds")

    env = EnvConfig(
        apiUrl="https://api.diagrid.io",
        authAudience="test-audience",
        authClientId="test-client-id",
        authDomain="diagrid-dev.us.auth0.com",
    )
    tkn = TokenResponse(
        access_token="test-access-token",
        token_type="Bearer",
        refresh_token="test-refresh-token",
        expires_in=3600,
        id_token="test-id-token",
    )
    cred = Credential(
        subject="auth0|123",
        env=env,
        token_response=tkn,
        client_id="test-client-id",
        default_org="org-abc",
        orgs={"org-abc": ["cra.diagrid.admin"]},
        expires_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    store.set(cred)
    loaded = store.get()

    assert loaded.subject == "auth0|123"
    assert loaded.default_org == "org-abc"
    assert loaded.orgs == {"org-abc": ["cra.diagrid.admin"]}
    assert loaded.bearer_token == "test-access-token"


def test_credential_file_is_base64(tmp_path: Path) -> None:
    """Stored credential file must be base64-encoded JSON."""
    store = FileCredentialStore(tmp_path / "creds")
    cred = Credential(subject="test-user")
    store.set(cred)

    raw = (tmp_path / "creds").read_text()
    decoded = base64.b64decode(raw)
    data = json.loads(decoded)
    assert data["subject"] == "test-user"


def test_credential_empty_file(tmp_path: Path) -> None:
    """Empty/missing file returns empty Credential."""
    store = FileCredentialStore(tmp_path / "nonexistent")
    cred = store.get()
    assert cred.subject == ""
    assert cred.bearer_token == ""


def test_credential_file_permissions(tmp_path: Path) -> None:
    """Credential file must have 0600 permissions."""
    store = FileCredentialStore(tmp_path / "creds")
    store.set(Credential())
    mode = (tmp_path / "creds").stat().st_mode & 0o777
    assert mode == 0o600


def test_unset_clears_credential(tmp_path: Path) -> None:
    """Unset should write an empty credential."""
    store = FileCredentialStore(tmp_path / "creds")
    store.set(Credential(subject="will-be-cleared"))
    store.unset()
    cred = store.get()
    assert cred.subject == ""
