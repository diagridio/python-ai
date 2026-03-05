# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for API key parsing."""

from __future__ import annotations

import jwt as pyjwt
import pytest

from diagrid.core.auth.api_key import (
    InvalidAPIKeyError,
    extract_org_id_from_api_key,
    get_api_key,
)


def _make_api_key(org_id: str = "test-org-123") -> str:
    """Create a fake JWT API key with org claims."""
    claims = {
        "sub": "api-key|abc",
        f"https://diagrid.io/org_{org_id}/roles": ["cra.diagrid.admin"],
    }
    return pyjwt.encode(
        claims, "test-secret-key-that-is-long-enough", algorithm="HS256"
    )


def test_extract_org_id_from_api_key() -> None:
    key = _make_api_key("my-org-456")
    org_id = extract_org_id_from_api_key(key)
    assert org_id == "my-org-456"


def test_extract_org_id_invalid_key() -> None:
    with pytest.raises(InvalidAPIKeyError, match="Failed to decode"):
        extract_org_id_from_api_key("not-a-jwt")


def test_extract_org_id_no_claim() -> None:
    key = pyjwt.encode(
        {"sub": "test"}, "test-secret-key-that-is-long-enough", algorithm="HS256"
    )
    with pytest.raises(InvalidAPIKeyError, match="No organization claim"):
        extract_org_id_from_api_key(key)


def test_get_api_key_flag_precedence() -> None:
    assert get_api_key(env_key="env-key", flag_key="flag-key") == "flag-key"


def test_get_api_key_env_fallback() -> None:
    assert get_api_key(env_key="env-key") == "env-key"


def test_get_api_key_none() -> None:
    assert get_api_key() is None
