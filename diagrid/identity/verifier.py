# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""JWKS-backed JWT verification for dp-Sentry-signed tokens."""

from __future__ import annotations

import ipaddress
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

import httpx2
import jwt
from jwt import PyJWKClient

from diagrid.identity import (
    IdentityNotConfiguredError,
    OAuthErrorCodes,
    TokenVerificationError,
    VerifierNotReadyError,
)

logger = logging.getLogger(__name__)

_CLOCK_SKEW_SECONDS = 120
_JWKS_CACHE_LIFETIME = 300  # seconds before a background refresh
_METADATA_PATH = "/v1.0/metadata"
_METADATA_TIMEOUT_SECONDS = 5.0
_API_TOKEN_HEADER = "dapr-api-token"
_HTTP_SCHEME = "http"
_HTTPS_SCHEME = "https"
_LOOPBACK_HOSTNAMES = frozenset({"localhost"})

__all__ = [
    "IdentityNotConfiguredError",
    "JWKSVerifier",
    "TokenVerificationError",
    "VerifierNotReadyError",
    "build_verifier",
]


@dataclass(frozen=True)
class IdentityCoordinates:
    """Where tokens come from and what they must claim.

    Discovery's internal result type: resolved from explicit config, the
    sidecar metadata endpoint, or the environment — in that order.
    """

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

        Claims are checked in one fixed order, so a token with two defects at
        once always reports the same code: the required claims (``exp``,
        ``iss``, ``sub``) first, then ``exp``, then ``iss``, then ``aud``.

        An ``alg`` outside the RS256 / ES256 allowlist — ``none`` above all —
        is ``oauth.invalid_token``, not ``oauth.decode_error``, which is
        reserved for a token that is genuinely unparseable.

        Raises:
            VerifierNotReadyError: key material unavailable.
            TokenVerificationError: any verification failure.
        """
        try:
            client = self._ensure_client()
        except Exception as exc:
            raise VerifierNotReadyError(str(exc)) from exc

        try:
            signing_key = client.get_signing_key_from_jwt(raw_token)
        except jwt.PyJWKClientError as exc:
            raise VerifierNotReadyError(str(exc)) from exc
        except jwt.InvalidTokenError as exc:
            raise TokenVerificationError(
                OAuthErrorCodes.DECODE_ERROR, str(exc)
            ) from exc

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
            raise TokenVerificationError(OAuthErrorCodes.EXPIRED, "token has expired")
        except jwt.InvalidIssuerError:
            raise TokenVerificationError(
                OAuthErrorCodes.INVALID_ISSUER, "issuer mismatch"
            )
        except jwt.InvalidAudienceError:
            raise TokenVerificationError(
                OAuthErrorCodes.INVALID_AUDIENCE, "audience mismatch"
            )
        except jwt.InvalidSignatureError:
            raise TokenVerificationError(
                OAuthErrorCodes.INVALID_SIGNATURE, "signature verification failed"
            )
        except jwt.DecodeError as exc:
            raise TokenVerificationError(OAuthErrorCodes.DECODE_ERROR, str(exc))
        except jwt.InvalidTokenError as exc:
            raise TokenVerificationError(OAuthErrorCodes.INVALID_TOKEN, str(exc))

        return payload


def _coords_from_identity(data: Any) -> Optional[IdentityCoordinates]:
    """Read the identity block out of a /v1.0/metadata response body.

    Called inside the callers' try: a malformed body raises and is treated as
    no discovery.
    """
    identity = data.get("identity")
    if not identity or not identity.get("issuer"):
        return None
    issuer = identity["issuer"]
    return IdentityCoordinates(
        issuer=issuer,
        jwks_uri=identity.get("jwks_uri", issuer.rstrip("/") + "/jwks.json"),
        audience=identity.get("audience", ""),
    )


def _discover_from_metadata() -> Optional[IdentityCoordinates]:
    """Try GET http://127.0.0.1:$PORT/v1.0/metadata for the identity block."""
    port = os.environ.get("CATALYST_DAPR_HTTP_PORT") or os.environ.get("DAPR_HTTP_PORT")
    if not port:
        return None
    url = f"http://127.0.0.1:{port}{_METADATA_PATH}"
    try:
        resp = httpx2.get(url, timeout=_METADATA_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return _coords_from_identity(resp.json())
    except Exception as exc:
        logger.warning(
            "identity discovery via %s failed (%s: %s); trying the next source",
            url,
            type(exc).__name__,
            exc,
        )
        return None


def _discover_from_remote() -> Optional[IdentityCoordinates]:
    """Fall back to the project endpoint when there is no local sidecar port.

    `diagrid dev run` runs the app on the developer's machine against a
    Catalyst-hosted sidecar, so 127.0.0.1 has nothing listening.
    """
    endpoint = os.environ.get("DAPR_HTTP_ENDPOINT", "").rstrip("/")
    if not endpoint:
        return None
    url = f"{endpoint}{_METADATA_PATH}"
    headers: Dict[str, str] = {}
    token = os.environ.get("DAPR_API_TOKEN")
    if token:
        if not endpoint.startswith("https://"):
            # Still sent: a self-hosted sidecar on plain http is a valid setup.
            logger.warning(
                "DAPR_API_TOKEN will be sent in clear text to non-https endpoint %s",
                endpoint,
            )
        headers[_API_TOKEN_HEADER] = token
    try:
        resp = httpx2.get(url, headers=headers, timeout=_METADATA_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return _coords_from_identity(resp.json())
    except Exception as exc:
        logger.warning(
            "identity discovery via %s failed (%s: %s); trying the next source",
            url,
            type(exc).__name__,
            exc,
        )
        return None


def _discover_from_env() -> Optional[IdentityCoordinates]:
    """Fall back to env vars."""
    issuer = os.environ.get("DIAGRID_DP_SENTRY_ISSUER", "")
    if not issuer:
        return None
    return IdentityCoordinates(
        issuer=issuer,
        jwks_uri=issuer.rstrip("/") + "/jwks.json",
        audience=os.environ.get("DIAGRID_DP_SENTRY_AUDIENCE", ""),
    )


def _is_loopback_host(host: str) -> bool:
    """Whether *host* names this machine, the way the local sidecar does."""
    if not host:
        return False
    if host in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _require_secure_jwks_uri(jwks_uri: str, allow_insecure_jwks: bool) -> None:
    """Refuse to fetch the key set over plaintext from a remote host.

    The key set is the entire root of trust: an on-path attacker who rewrites
    a plaintext JWKS response mints tokens this verifier accepts.  Loopback is
    exempt because that is where the local sidecar serves, and
    *allow_insecure_jwks* exists for the rest.

    Both exemptions widen the rule to plain http and to nothing else: a
    ``file://`` URI, or any other scheme, is refused however the flag is set.
    """
    parsed = urlsplit(jwks_uri)
    if parsed.scheme == _HTTPS_SCHEME:
        return
    if parsed.scheme != _HTTP_SCHEME:
        raise IdentityNotConfiguredError(
            f"refusing to fetch JWKS over "
            f"{parsed.scheme or 'an unknown scheme'} from {jwks_uri!r}: the "
            "key set is the root of trust, so it must be served over https. "
            "allow_insecure_jwks relaxes that to plain http, never to another "
            "scheme."
        )
    if allow_insecure_jwks or _is_loopback_host(parsed.hostname or ""):
        return
    raise IdentityNotConfiguredError(
        f"refusing to fetch JWKS over http from {jwks_uri!r}: the key set is "
        "the root of trust, so it must be served over https from a "
        "non-loopback host. Set allow_insecure_jwks=True to opt in."
    )


def build_verifier(
    issuer: Optional[str] = None,
    audience: Optional[str] = None,
    jwks_uri: Optional[str] = None,
    allow_insecure_jwks: bool = False,
) -> JWKSVerifier:
    """Build a verifier using explicit config, metadata discovery, or env vars.

    Priority: explicit args > local /v1.0/metadata > remote /v1.0/metadata >
    env vars. Local comes first so a deployed in-cluster app keeps using the
    loopback call rather than a network round trip.

    Raises:
        IdentityNotConfiguredError: no coordinates could be resolved, or the
            resolved JWKS URI is plaintext and *allow_insecure_jwks* is off.
    """
    discovered: Optional[IdentityCoordinates] = None
    if not (issuer and jwks_uri):
        discovered = (
            _discover_from_metadata() or _discover_from_remote() or _discover_from_env()
        )

    resolved_issuer = issuer or (discovered.issuer if discovered else "")
    # Explicit beats discovered beats derived: a sidecar that publishes a
    # jwks_uri away from its issuer means it, and deriving issuer+/jwks.json
    # ahead of that would point the verifier at an endpoint which need not
    # exist.  The discovered jwks_uri is adopted only when the discovered
    # coordinates describe the issuer actually being verified -- otherwise a
    # token minted by the advertised issuer, claiming the pinned one, would
    # verify against the advertised issuer's keys.
    resolved_jwks_uri = jwks_uri
    if not resolved_jwks_uri and discovered and discovered.issuer == resolved_issuer:
        resolved_jwks_uri = discovered.jwks_uri
    if not resolved_jwks_uri and resolved_issuer:
        resolved_jwks_uri = resolved_issuer.rstrip("/") + "/jwks.json"
    resolved_audience = audience or (discovered.audience if discovered else "")

    if not resolved_issuer or not resolved_jwks_uri:
        raise IdentityNotConfiguredError(
            "Cannot discover identity coordinates: "
            "set issuer/jwks_uri explicitly, configure the sidecar metadata endpoint "
            "(DAPR_HTTP_PORT locally or DAPR_HTTP_ENDPOINT for a remote sidecar), "
            "or set DIAGRID_DP_SENTRY_ISSUER"
        )
    _require_secure_jwks_uri(resolved_jwks_uri, allow_insecure_jwks)
    coords = IdentityCoordinates(
        issuer=resolved_issuer,
        jwks_uri=resolved_jwks_uri,
        audience=resolved_audience,
    )

    v = JWKSVerifier(
        issuer=coords.issuer, jwks_uri=coords.jwks_uri, audience=coords.audience
    )
    v.warm()
    return v
