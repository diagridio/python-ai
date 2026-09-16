# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""ASGI middleware for inbound user-token verification."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from diagrid.identity import (
    OAuthConfig,
    OAuthErrorCodes,
    TokenVerifier,
    VerifiedUser,
)
from diagrid.identity.outbound import (
    BEARER_PREFIX,
    USER_TOKEN_HEADER,
    clear_current_token,
    reset_current_token,
    set_current_token,
)
from diagrid.identity.verifier import (
    TokenVerificationError,
    VerifierNotReadyError,
    build_verifier,
)

logger = logging.getLogger(__name__)

# ``request.state`` is one namespace shared by every middleware in the stack,
# so the verified caller is namespaced rather than parked on ``state.user``.
_STATE_ATTRIBUTE = "diagrid_user"

__all__ = ["OAuthMiddleware", "verified_user"]


class OAuthMiddleware(BaseHTTPMiddleware):
    """Verifies ``X-Diagrid-User-Token`` on every inbound request.

    The verified caller is read with :func:`verified_user`.  The
    unnamespaced ``request.state.user`` slot is left alone — it belongs to
    the application.

    Usage::

        from fastapi import FastAPI, Request
        from diagrid.identity import OAuthConfig
        from diagrid.identity.asgi import OAuthMiddleware, verified_user

        app = FastAPI()
        app.add_middleware(OAuthMiddleware, config=OAuthConfig(scopes={"agent.invoke"}))

        @app.post("/invoke")
        async def invoke(request: Request):
            user = verified_user(request)
            return {"subject": user.subject}

    Pass *verifier* to supply the verifier yourself — a pre-built
    :class:`~diagrid.identity.verifier.JWKSVerifier`, or any
    :class:`~diagrid.identity.TokenVerifier` — instead of letting the
    middleware discover coordinates and build one on the first request.
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        app,
        config: Optional[OAuthConfig] = None,
        verifier: Optional[TokenVerifier] = None,
    ) -> None:
        super().__init__(app)
        self._config = config or OAuthConfig()
        self._verifier: Optional[TokenVerifier] = verifier

    def _get_verifier(self) -> TokenVerifier:
        if self._verifier is not None:
            return self._verifier
        self._verifier = build_verifier(
            issuer=self._config.issuer,
            audience=self._config.audience,
            jwks_uri=self._config.jwks_uri,
            allow_insecure_jwks=self._config.allow_insecure_jwks,
        )
        return self._verifier

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        raw_header = request.headers.get(USER_TOKEN_HEADER, "")
        token = _trim_bearer(raw_header)

        if not token:
            if self._config.require_auth:
                return _error_response(401, OAuthErrorCodes.MISSING_TOKEN)
            clear_current_token()
            return await call_next(request)

        try:
            verifier = self._get_verifier()
        except RuntimeError:
            logger.warning("identity verifier not configured; rejecting request")
            return _error_response(503, OAuthErrorCodes.NOT_CONFIGURED)

        try:
            payload = verifier.verify(token)
        except VerifierNotReadyError:
            return _error_response(503, OAuthErrorCodes.VERIFIER_UNAVAILABLE)
        except TokenVerificationError as exc:
            status = 403 if exc.code == OAuthErrorCodes.MISSING_SCOPE else 401
            return _error_response(status, exc.code)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Ordered after the specific handlers: first, it would report a
            # correctly-rejected token as a verifier failure.  No caller can be
            # adjudicated, so the answer is 503 rather than a framework 500.
            logger.warning(
                "unexpected error verifying the user token (%s: %s); rejecting request",
                type(exc).__name__,
                exc,
            )
            return _error_response(503, OAuthErrorCodes.VERIFIER_UNAVAILABLE)

        scopes = _extract_scopes(payload)
        missing = self._config.scopes - scopes
        if missing:
            return _error_response(403, OAuthErrorCodes.MISSING_SCOPE)

        user = VerifiedUser(
            subject=payload.get("sub", ""),
            tenant=payload.get("tid", payload.get("tenant", "")),
            scopes=scopes,
            claims=payload,
            issuer_id=payload.get("iss", ""),
        )
        setattr(request.state, _STATE_ATTRIBUTE, user)
        cv_token = set_current_token(token)

        try:
            return await call_next(request)
        finally:
            reset_current_token(cv_token)


def verified_user(request: Request) -> Optional[VerifiedUser]:
    """The verified caller on *request*, or ``None``.

    ``None`` means the request carried no ``X-Diagrid-User-Token`` and the
    config allowed that (``require_auth=False``); a token that was present
    but invalid never reaches a handler.
    """
    user = getattr(request.state, _STATE_ATTRIBUTE, None)
    return user if isinstance(user, VerifiedUser) else None


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
