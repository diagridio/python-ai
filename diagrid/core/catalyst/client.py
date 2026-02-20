"""Authenticated HTTP client for the Catalyst API."""

from __future__ import annotations

import random
import time
from typing import Any

import httpx

from diagrid.core.auth.token import AuthContext

_MAX_RETRIES = 3
_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


class CatalystAPIError(Exception):
    """Raised on Catalyst API errors."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"Catalyst API error ({status_code}): {message}")


class CatalystClient:
    """Authenticated HTTP client for Diagrid Catalyst API."""

    # Default API group used by most resources.
    _DEFAULT_API_GROUP = "cra.diagrid.io"
    # Dapr resources (configurations, components, …) live under this group.
    DAPR_API_GROUP = "dapr.diagrid.io"

    def __init__(self, auth_ctx: AuthContext) -> None:
        self.auth_ctx = auth_ctx
        self.api_url = auth_ctx.api_url.rstrip("/")
        self.base_url = f"{self.api_url}/apis/{self._DEFAULT_API_GROUP}/v1beta1"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self.auth_ctx.auth_header)
        return headers

    def _base_url_for(self, api_group: str | None) -> str:
        if api_group is None:
            return self.base_url
        return f"{self.api_url}/apis/{api_group}/v1beta1"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_data: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        timeout: float = 30.0,
        api_group: str | None = None,
    ) -> httpx.Response:
        url = f"{self._base_url_for(api_group)}{path}"
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                with httpx.Client() as client:
                    resp = client.request(
                        method,
                        url,
                        headers=self._headers(),
                        json=json_data,
                        params=params,
                        timeout=timeout,
                    )
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_backoff_delay(attempt))
                    continue
                raise CatalystAPIError(
                    504, f"Request timed out after {_MAX_RETRIES + 1} attempts"
                ) from exc

            if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                time.sleep(_backoff_delay(attempt))
                continue

            if resp.status_code >= 400:
                raise CatalystAPIError(resp.status_code, resp.text)
            return resp

        # Should not be reached, but satisfy the type checker.
        raise last_exc  # type: ignore[misc]

    def get(self, path: str, params: dict[str, str] | None = None) -> httpx.Response:
        return self._request("GET", path, params=params)

    def post(
        self,
        path: str,
        json_data: dict[str, Any] | None = None,
        *,
        params: dict[str, str] | None = None,
        timeout: float = 30.0,
        api_group: str | None = None,
    ) -> httpx.Response:
        return self._request(
            "POST",
            path,
            json_data=json_data,
            params=params,
            timeout=timeout,
            api_group=api_group,
        )

    def put(
        self,
        path: str,
        json_data: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        return self._request("PUT", path, json_data=json_data, params=params)

    def delete(self, path: str) -> httpx.Response:
        return self._request("DELETE", path)


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter (matches Go CLI pester behaviour)."""
    return min(2**attempt + random.uniform(0, 1), 10.0)
