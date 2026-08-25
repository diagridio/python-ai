# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""App-side identity surface for Catalyst agents.

Two lines of app code buy verified inbound identity; zero lines buy
outbound OBO::

    from diagrid.identity import OAuthConfig, VerifiedUser
    from diagrid.identity.asgi import OAuthMiddleware

    oauth = OAuthConfig(scopes={"agent.invoke"})
    app.add_middleware(OAuthMiddleware, config=oauth)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional


@dataclass(frozen=True)
class OAuthConfig:
    """Policy the middleware enforces on every inbound request.

    Attributes:
        scopes: Required scopes — the middleware returns 403 when the
            verified token lacks any of them.
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
            to share the same app.
    """

    scopes: FrozenSet[str] = field(default_factory=frozenset)
    issuer: Optional[str] = None
    audience: Optional[str] = None
    jwks_uri: Optional[str] = None
    require_auth: bool = True


@dataclass(frozen=True)
class VerifiedUser:
    """Verified caller identity attached to ``request.state.user``.

    Attributes:
        subject: ``sub`` claim — email, user-id, or agent SPIFFE URI.
        tenant: Tenant / org claim extracted from the token.
        scopes: OAuth scopes carried by the token.
        claims: Full decoded JWT payload for policies that need richer
            access.
        issuer_id: The ``iss`` value on the verified token.
    """

    subject: str
    tenant: str = ""
    scopes: FrozenSet[str] = field(default_factory=frozenset)
    claims: Dict[str, Any] = field(default_factory=dict)
    issuer_id: str = ""


__all__ = ["OAuthConfig", "VerifiedUser"]
