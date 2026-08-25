# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""ASGI middleware for inbound user-token verification."""

from __future__ import annotations

import logging
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from diagrid.identity import OAuthConfig, VerifiedUser
from diagrid.identity.outbound import (
    BEARER_PREFIX,
    USER_TOKEN_HEADER,
    clear_current_token,
    set_current_token,
)
from diagrid.identity.verifier import (
    JWKSVerifier,
    TokenVerificationError,
    VerifierNotReady,
    build_verifier,
)

logger = logging.getLogger(__name__)


class OAuthMiddleware(BaseHTTPMiddleware):
    """Verifies ``X-Diagrid-User-Token`` on every inbound request.

    Usage::

        from fastapi import FastAPI
        from diagrid.identity import OAuthConfig
        from diagrid.identity.asgi import OAuthMiddleware

        app = FastAPI()
        app.add_middleware(OAuthMiddleware, config=OAuthConfig(scopes={"agent.invoke"}))
    """

    def __init__(self, app, config: Optional[OAuthConfig] = None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self._config = config or OAuthConfig()
        self._verifier: Optional[JWKSVerifier] = None

    def _get_verifier(self) -> JWKSVerifier:
        if self._verifier is not None:
            return self._verifier
        self._verifier = build_verifier(
            issuer=self._config.issuer,
            audience=self._config.audience,
            jwks_uri=self._config.jwks_uri,
        )
        return self._verifier

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        raw_header = request.headers.get(USER_TOKEN_HEADER, "")
        token = _trim_bearer(raw_header)

        if not token:
            if self._config.require_auth:
                return _error_response(401, "oauth.missing_token")
            clear_current_token()
            return await call_next(request)

        try:
            verifier = self._get_verifier()
        except RuntimeError:
            logger.warning("identity verifier not configured; rejecting request")
            return _error_response(503, "oauth.not_configured")

        try:
            payload = verifier.verify(token)
        except VerifierNotReady:
            return _error_response(503, "oauth.verifier_unavailable")
        except TokenVerificationError as exc:
            status = 403 if exc.code == "oauth.missing_scope" else 401
            return _error_response(status, exc.code)

        scopes = _extract_scopes(payload)
        missing = self._config.scopes - scopes
        if missing:
            return _error_response(403, "oauth.missing_scope")

        user = VerifiedUser(
            subject=payload.get("sub", ""),
            tenant=payload.get("tid", payload.get("tenant", "")),
            scopes=scopes,
            claims=payload,
            issuer_id=payload.get("iss", ""),
        )
        request.state.user = user
        set_current_token(token)

        try:
            return await call_next(request)
        finally:
            clear_current_token()


def _trim_bearer(value: str) -> str:
    value = value.strip()
    if value.upper().startswith(BEARER_PREFIX.upper()):
        value = value[len(BEARER_PREFIX) :]
    return value.strip()


def _extract_scopes(payload: dict) -> frozenset[str]:
    raw = payload.get("scp") or payload.get("scope") or payload.get("scopes", "")
    if isinstance(raw, list):
        return frozenset(raw)
    if isinstance(raw, str) and raw:
        return frozenset(raw.split())
    return frozenset()


def _error_response(status: int, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": code},
        headers={"Cache-Control": "no-store"},
    )
