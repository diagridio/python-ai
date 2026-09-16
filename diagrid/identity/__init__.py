# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""App-side identity surface for Catalyst agents.

Two lines of app code buy verified inbound identity::

    from diagrid.identity import OAuthConfig
    from diagrid.identity.asgi import OAuthMiddleware

    oauth = OAuthConfig(scopes={"agent.invoke"})
    app.add_middleware(OAuthMiddleware, config=oauth)

Handlers read the verified caller with
:func:`diagrid.identity.asgi.verified_user`::

    from diagrid.identity.asgi import verified_user

    user = verified_user(request)

Outbound on-behalf-of calls then cost zero lines beyond the client you
already had to construct — see :mod:`diagrid.identity.http`::

    from diagrid.identity.http import AsyncClient

    client = AsyncClient()

Nothing here imports the optional runtime dependencies; ``asgi``,
``verifier`` and ``http`` each pull their own, so importing this module
works on a bare install.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    runtime_checkable,
)

from diagrid.identity.errors import (
    IdentityNotConfiguredError,
    OAuthErrorCodes,
    TokenVerificationError,
    VerifierNotReadyError,
)

__all__ = [
    "IdentityNotConfiguredError",
    "OAuthConfig",
    "OAuthErrorCodes",
    "TokenVerificationError",
    "TokenVerifier",
    "VerifiedUser",
    "VerifierNotReadyError",
]


class _SortedScopes(FrozenSet[str]):
    """A frozen set of scopes that always iterates in sorted order.

    Set semantics are untouched; only the iteration order is pinned, so a
    handler echoing scopes into a JSON response emits the same order on every
    request and in every Diagrid SDK.
    """

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(super().__iter__()))

    def __repr__(self) -> str:
        # Keeps this private class name out of every ``OAuthConfig`` and
        # ``VerifiedUser`` repr.
        return f"frozenset({sorted(super().__iter__())!r})"


def _sorted_scopes(scopes: Iterable[str]) -> FrozenSet[str]:
    """Normalise any scope iterable to a deterministically ordered set."""
    return _SortedScopes(scopes)


@dataclass(frozen=True)
class OAuthConfig:
    """Policy the middleware enforces on every inbound request.

    Attributes:
        scopes: Required scopes — the middleware returns 403 when the
            verified token lacks any of them.  Iterates in sorted order.
        issuer: Expected ``iss`` claim.  Normally discovered from the
            sidecar ``/v1.0/metadata`` response; set explicitly only
            when the metadata endpoint is unavailable.
        audience: Expected ``aud`` claim.  Same discovery rules as
            *issuer*.
        jwks_uri: JWKS endpoint for signature verification.  Same
            discovery rules.
        require_auth: When ``True`` (default), requests without
            ``X-Diagrid-User-Token`` are rejected with 401.  Set to
            ``False`` to allow unauthenticated routes (health, readiness)
            to share the same app.  A token that *is* present is always
            verified, and an invalid one always rejected, either way.
        allow_insecure_jwks: Opt in to fetching the key set over plaintext
            HTTP from a non-loopback host.  ``False`` by default: the key
            set is the whole root of trust, and an on-path attacker who
            rewrites a plaintext response mints tokens this verifier
            accepts.  Relaxes the rule to plain http only — a ``file://``
            JWKS URI, or any other scheme, stays refused.
    """

    scopes: FrozenSet[str] = field(default_factory=frozenset)
    issuer: Optional[str] = None
    audience: Optional[str] = None
    jwks_uri: Optional[str] = None
    require_auth: bool = True
    allow_insecure_jwks: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "scopes", _sorted_scopes(self.scopes))


@dataclass(frozen=True)
class VerifiedUser:
    """Verified caller identity, read with
    :func:`diagrid.identity.asgi.verified_user`.

    Attributes:
        subject: ``sub`` claim — email, user-id, or agent SPIFFE URI.
        tenant: Tenant / org claim extracted from the token.
        scopes: OAuth scopes carried by the token.  Iterates in sorted
            order.
        claims: Full decoded JWT payload for policies that need richer
            access.
        issuer_id: The ``iss`` value on the verified token.
    """

    subject: str
    tenant: str = ""
    scopes: FrozenSet[str] = field(default_factory=frozenset)
    claims: Dict[str, Any] = field(default_factory=dict)
    issuer_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "scopes", _sorted_scopes(self.scopes))

    def has_scope(self, scope: str) -> bool:
        """Whether the verified token carries *scope*."""
        return scope in self.scopes


@runtime_checkable
class TokenVerifier(Protocol):
    """What the middleware needs from a verifier.

    Structural, so :class:`diagrid.identity.verifier.JWKSVerifier` and a
    test double both satisfy it without inheriting anything.
    """

    def verify(self, raw_token: str) -> Dict[str, Any]:
        """Verify signature and claims, returning the decoded payload.

        Raises:
            VerifierNotReadyError: key material unavailable.
            TokenVerificationError: any verification failure.
        """
        ...
