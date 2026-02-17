"""Authenticated HTTP client for the Catalyst API."""

from __future__ import annotations

from typing import Any

import httpx

from diagrid.core.auth.token import AuthContext


class CatalystAPIError(Exception):
    """Raised on Catalyst API errors."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"Catalyst API error ({status_code}): {message}")


class CatalystClient:
    """Authenticated HTTP client for Diagrid Catalyst API."""

    def __init__(self, auth_ctx: AuthContext) -> None:
        self.auth_ctx = auth_ctx
        self.base_url = f"{auth_ctx.api_url.rstrip('/')}/apis/cra.diagrid.io/v1beta1"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self.auth_ctx.auth_header)
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_data: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        with httpx.Client() as client:
            resp = client.request(
                method,
                url,
                headers=self._headers(),
                json=json_data,
                params=params,
                timeout=30.0,
            )
        if resp.status_code >= 400:
            raise CatalystAPIError(resp.status_code, resp.text)
        return resp

    def get(self, path: str, params: dict[str, str] | None = None) -> httpx.Response:
        return self._request("GET", path, params=params)

    def post(
        self, path: str, json_data: dict[str, Any] | None = None
    ) -> httpx.Response:
        return self._request("POST", path, json_data=json_data)

    def delete(self, path: str) -> httpx.Response:
        return self._request("DELETE", path)
