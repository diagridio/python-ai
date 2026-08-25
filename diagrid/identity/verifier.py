# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""JWKS-backed JWT verification for dp-Sentry-signed tokens."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
import jwt
from jwt import PyJWKClient

logger = logging.getLogger(__name__)

_CLOCK_SKEW_SECONDS = 120
_JWKS_CACHE_LIFETIME = 300  # seconds before a background refresh


class VerifierNotReady(Exception):
    """JWKS key material has not loaded yet."""


class TokenVerificationError(Exception):
    """Signature or claim validation failed."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        super().__init__(message or code)


@dataclass(frozen=True)
class _IdentityCoordinates:
    issuer: str
    jwks_uri: str
    audience: str


class JWKSVerifier:
    """Fetches JWKS from a public HTTPS endpoint, caches keys, and verifies
    dp-Sentry JWTs.

    Thread-safe: the key set is swapped atomically behind a lock.
    """

    def __init__(
        self,
        issuer: str,
        jwks_uri: str,
        audience: str = "",
    ) -> None:
        self._issuer = issuer
        self._jwks_uri = jwks_uri
        self._audience = audience
        self._jwks_client: Optional[PyJWKClient] = None
        self._lock = threading.Lock()
        self._ready = False

    def warm(self) -> None:
        """Eagerly fetch the JWKS so the first verify call does not block."""
        try:
            self._ensure_client()
            self._ready = True
        except Exception:
            logger.warning(
                "JWKS warm-up failed; will retry on first request", exc_info=True
            )

    def _ensure_client(self) -> PyJWKClient:
        if self._jwks_client is not None:
            return self._jwks_client
        with self._lock:
            if self._jwks_client is not None:
                return self._jwks_client
            self._jwks_client = PyJWKClient(
                self._jwks_uri,
                cache_jwk_set=True,
                lifespan=_JWKS_CACHE_LIFETIME,
            )
            self._ready = True
            return self._jwks_client

    def verify(self, raw_token: str) -> Dict[str, Any]:
        """Verify signature and claims, returning the decoded payload.

        Raises:
            VerifierNotReady: key material unavailable.
            TokenVerificationError: any verification failure.
        """
        try:
            client = self._ensure_client()
        except Exception as exc:
            raise VerifierNotReady(str(exc)) from exc

        try:
            signing_key = client.get_signing_key_from_jwt(raw_token)
        except jwt.PyJWKClientError as exc:
            raise VerifierNotReady(str(exc)) from exc

        decode_opts: Dict[str, Any] = {
            "algorithms": ["RS256", "ES256"],
            "leeway": _CLOCK_SKEW_SECONDS,
            "options": {"require": ["exp", "iss", "sub"]},
        }
        if self._issuer:
            decode_opts["issuer"] = self._issuer
        if self._audience:
            decode_opts["audience"] = self._audience
        else:
            decode_opts["options"]["verify_aud"] = False

        try:
            payload = jwt.decode(
                raw_token,
                signing_key.key,
                **decode_opts,
            )
        except jwt.ExpiredSignatureError:
            raise TokenVerificationError("oauth.expired", "token has expired")
        except jwt.InvalidIssuerError:
            raise TokenVerificationError("oauth.invalid_issuer", "issuer mismatch")
        except jwt.InvalidAudienceError:
            raise TokenVerificationError("oauth.invalid_audience", "audience mismatch")
        except jwt.InvalidSignatureError:
            raise TokenVerificationError(
                "oauth.invalid_signature", "signature verification failed"
            )
        except jwt.DecodeError as exc:
            raise TokenVerificationError("oauth.decode_error", str(exc))
        except jwt.InvalidTokenError as exc:
            raise TokenVerificationError("oauth.invalid_token", str(exc))

        return payload


def _discover_from_metadata() -> Optional[_IdentityCoordinates]:
    """Try GET http://127.0.0.1:$PORT/v1.0/metadata for the identity block."""
    port = os.environ.get("CATALYST_DAPR_HTTP_PORT") or os.environ.get("DAPR_HTTP_PORT")
    if not port:
        return None
    url = f"http://127.0.0.1:{port}/v1.0/metadata"
    try:
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.debug("metadata discovery at %s failed", url, exc_info=True)
        return None

    identity = data.get("identity")
    if not identity or not identity.get("issuer"):
        return None
    issuer = identity["issuer"]
    return _IdentityCoordinates(
        issuer=issuer,
        jwks_uri=identity.get("jwks_uri", issuer.rstrip("/") + "/jwks.json"),
        audience=identity.get("audience", ""),
    )


def _discover_from_env() -> Optional[_IdentityCoordinates]:
    """Fall back to env vars."""
    issuer = os.environ.get("DIAGRID_DP_SENTRY_ISSUER", "")
    if not issuer:
        return None
    return _IdentityCoordinates(
        issuer=issuer,
        jwks_uri=issuer.rstrip("/") + "/jwks.json",
        audience=os.environ.get("DIAGRID_DP_SENTRY_AUDIENCE", ""),
    )


def build_verifier(
    issuer: Optional[str] = None,
    audience: Optional[str] = None,
    jwks_uri: Optional[str] = None,
) -> JWKSVerifier:
    """Build a verifier using explicit config, metadata discovery, or env vars.

    Priority: explicit args > /v1.0/metadata > env vars.
    """
    discovered: Optional[_IdentityCoordinates] = None
    if not (issuer and jwks_uri):
        discovered = _discover_from_metadata() or _discover_from_env()

    resolved_issuer = issuer or (discovered.issuer if discovered else "")
    resolved_jwks_uri = jwks_uri
    if not resolved_jwks_uri and resolved_issuer:
        resolved_jwks_uri = resolved_issuer.rstrip("/") + "/jwks.json"
    if not resolved_jwks_uri and discovered:
        resolved_jwks_uri = discovered.jwks_uri
    resolved_audience = audience or (discovered.audience if discovered else "")

    if not resolved_issuer or not resolved_jwks_uri:
        raise RuntimeError(
            "Cannot discover identity coordinates: "
            "set issuer/jwks_uri explicitly, configure the sidecar metadata endpoint, "
            "or set DIAGRID_DP_SENTRY_ISSUER"
        )
    coords = _IdentityCoordinates(
        issuer=resolved_issuer,
        jwks_uri=resolved_jwks_uri,
        audience=resolved_audience,
    )

    v = JWKSVerifier(
        issuer=coords.issuer, jwks_uri=coords.jwks_uri, audience=coords.audience
    )
    v.warm()
    return v
