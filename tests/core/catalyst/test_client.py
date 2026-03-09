# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for Catalyst API client."""

from __future__ import annotations

from diagrid.core.auth.token import AuthContext
from diagrid.core.catalyst.client import CatalystClient, _format_error_body


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


# --- _format_error_body tests ---


def test_format_error_body_single_error() -> None:
    """Extracts title + detail from a single error."""
    body = '{"errors":[{"status":400,"title":"Bad Request","detail":"max number of components reached, current 10 max 10"}]}'
    assert (
        _format_error_body(body)
        == "Bad Request: max number of components reached, current 10 max 10"
    )


def test_format_error_body_multiple_errors() -> None:
    """Joins multiple errors with semicolons."""
    body = '{"errors":[{"title":"Bad Request","detail":"field A invalid"},{"title":"Conflict","detail":"already exists"}]}'
    assert (
        _format_error_body(body)
        == "Bad Request: field A invalid; Conflict: already exists"
    )


def test_format_error_body_detail_only() -> None:
    """Uses detail alone when title is missing."""
    body = '{"errors":[{"detail":"something went wrong"}]}'
    assert _format_error_body(body) == "something went wrong"


def test_format_error_body_title_only() -> None:
    """Uses title alone when detail is missing."""
    body = '{"errors":[{"title":"Internal Server Error"}]}'
    assert _format_error_body(body) == "Internal Server Error"


def test_format_error_body_invalid_json() -> None:
    """Falls back to raw text on invalid JSON."""
    raw = "not json at all"
    assert _format_error_body(raw) == raw


def test_format_error_body_no_errors_key() -> None:
    """Falls back to raw text when 'errors' key is missing."""
    body = '{"message":"oops"}'
    assert _format_error_body(body) == body


def test_format_error_body_empty_errors_list() -> None:
    """Falls back to raw text when errors list is empty."""
    body = '{"errors":[]}'
    assert _format_error_body(body) == body
