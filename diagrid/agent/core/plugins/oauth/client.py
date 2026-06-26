# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Thin httpx client for the sidecar's /v1.0-alpha/auth/verify endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx

AUTH_VERIFY_PATH = "/v1.0-alpha/auth/verify"


@dataclass
class AuthVerifyResponse:
    verified: bool
    claims: Dict[str, Any] = field(default_factory=dict)
    issuer_id: Optional[str] = None
    error: Optional[str] = None
    error_description: Optional[str] = None

    @classmethod
    def from_json(cls, payload: dict) -> "AuthVerifyResponse":
        return cls(
            verified=bool(payload.get("verified", False)),
            claims=payload.get("claims") or {},
            issuer_id=payload.get("issuer_id"),
            error=payload.get("error"),
            error_description=payload.get("error_description"),
        )


class AuthVerifyClient:
    """Forwards a JWT to the local sidecar and returns its verdict."""

    def __init__(self, *, sidecar_url: str, timeout_seconds: float = 5.0):
        self._url = f"{sidecar_url.rstrip('/')}{AUTH_VERIFY_PATH}"
        self._timeout = timeout_seconds
        self._http = httpx.AsyncClient(timeout=timeout_seconds)

    async def verify(
        self, *, jwt: str, expected_audience: Optional[str] = None
    ) -> AuthVerifyResponse:
        body: Dict[str, Any] = {"jwt": jwt}
        if expected_audience:
            body["expected_audience"] = expected_audience
        resp = await self._http.post(self._url, json=body)
        resp.raise_for_status()
        return AuthVerifyResponse.from_json(resp.json())

    async def aclose(self) -> None:
        await self._http.aclose()
