# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
OAuthPlugin — verifies inbound caller JWTs at agent ingress.

Thin client of the local Catalyst sidecar's /v1.0-alpha/auth/verify
endpoint (AI-647). The sidecar owns JWKS, multi-issuer config, claim
mapping, and trust context; the plugin just forwards the JWT and
populates ctx.caller with the verified claims.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

try:
    from dapr_agents.hooks import Deny, HookDecision, Proceed
except ModuleNotFoundError:  # pragma: no cover
    # TODO(AI-596): dapr-agents has not yet shipped its hooks module. Fall back
    # to the local definitions so the plugin stays consumable until it does.
    from .._hooks import Deny, HookDecision, Proceed

from ..context import CallerIdentity, LifecycleContext
from ..spi import LifecycleEvent
from .client import AuthVerifyClient

logger = logging.getLogger(__name__)

DEFAULT_SIDECAR_PORT = "3500"


class OAuthPlugin:
    """
    Verifies inbound caller JWTs via the local sidecar /auth/verify
    endpoint and populates ctx.caller with the verified claims.

    Per the "raw access tokens never enter signed workflow history"
    rule, scrubs the raw Authorization header from caller_headers
    after verification — only the verified-claims subset propagates
    into workflow input.
    """

    name = "oauth"
    priority = 10  # Runs first in the chain
    capabilities = frozenset({LifecycleEvent.BEFORE_AGENT_INVOKE})
    failure_mode = "closed"  # Verification failure → Deny

    def __init__(
        self,
        *,
        sidecar_url: Optional[str] = None,
        expected_audience: Optional[str] = None,
        timeout_seconds: float = 5.0,
    ):
        self._client = AuthVerifyClient(
            sidecar_url=sidecar_url or _default_sidecar_url(),
            timeout_seconds=timeout_seconds,
        )
        self._expected_audience = expected_audience

    @classmethod
    def from_catalyst(cls, *, expected_audience: Optional[str] = None) -> "OAuthPlugin":
        """Construct with Catalyst defaults (sidecar URL from env)."""
        return cls(expected_audience=expected_audience)

    def configure(self, agent: object) -> None:
        # Could read agent's SPIFFE ID here to default expected_audience.
        # For v1, expected_audience defaults to the sidecar's own identity
        # if not provided (sidecar handles the fallback).
        pass

    async def on_event(self, ctx: LifecycleContext) -> Optional[HookDecision]:
        if ctx.event != LifecycleEvent.BEFORE_AGENT_INVOKE:
            return None

        caller_headers = getattr(ctx, "caller_headers", None) or {}
        jwt = self._extract_bearer(caller_headers)
        if not jwt:
            return Deny(
                code="oauth.missing_token",
                details={"reason": "no Authorization: Bearer header on inbound call"},
            )

        try:
            response = await self._client.verify(
                jwt=jwt, expected_audience=self._expected_audience
            )
        except Exception as exc:
            logger.error("oauth.verify_call_failed", extra={"error": str(exc)})
            return Deny(
                code="oauth.verify_unavailable",
                details={"error": str(exc)},
            )

        if not response.verified:
            return Deny(
                code=f"oauth.{response.error or 'verification_failed'}",
                details={
                    "error_description": response.error_description,
                },
            )

        # Populate ctx.caller from verified claims.
        ctx.caller = CallerIdentity(
            subject=response.claims.get("sub", ""),
            tenant=response.claims.get("tenant"),
            scopes=frozenset(response.claims.get("scopes", []) or []),
            issuer_id=response.issuer_id,
            claims=response.claims,
        )

        # Scrub raw token from caller_headers — verified claims propagate,
        # raw JWT does NOT enter workflow input or signed history.
        self._scrub_authorization_header(caller_headers)

        return Proceed()

    def _extract_bearer(self, headers: dict) -> Optional[str]:
        for key in ("authorization", "Authorization"):
            value = headers.get(key)
            if value and value.lower().startswith("bearer "):
                return value[7:].strip()
        return None

    def _scrub_authorization_header(self, headers: dict) -> None:
        for key in ("authorization", "Authorization"):
            headers.pop(key, None)


def _default_sidecar_url() -> str:
    """Resolve the local sidecar URL from env (DAPR_HTTP_PORT) or default."""
    port = os.environ.get("DAPR_HTTP_PORT", DEFAULT_SIDECAR_PORT)
    return f"http://127.0.0.1:{port}"
