# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
OAuthPlugin — app-side policy layer.

Reads verified caller claims delivered by the sidecar (via headers or
TriggerAction.caller_claims), populates ctx.caller, and applies per-agent
authorization policy (scope checks, tenant checks, subject allowlist).

Does NOT verify JWTs. Does NOT call the sidecar. The sidecar's inbound
exchange middleware has already verified and exchanged the JWT before the
request reaches the agent app.
"""

from __future__ import annotations

import logging
from typing import FrozenSet, Optional

try:
    from dapr_agents.hooks import Deny, HookDecision, Proceed
except ModuleNotFoundError:  # pragma: no cover
    # TODO: Fall back to local definitions until dapr-agents ships its hooks module.
    from .._hooks import Deny, HookDecision, Proceed

from ..context import CallerIdentity, LifecycleContext
from ..spi import LifecycleEvent

logger = logging.getLogger(__name__)


class OAuthPlugin:
    """
    App-side policy layer for user-identity-based authorization.

    Reads verified claims delivered by the sidecar; populates ctx.caller;
    enforces per-agent policy (scope allowlist, tenant allowlist, subject
    allowlist).
    """

    name = "oauth"
    priority = 10  # Runs first in the chain (policy gate)
    capabilities = frozenset({LifecycleEvent.BEFORE_AGENT_INVOKE})
    failure_mode = "closed"

    def __init__(
        self,
        *,
        required_scopes: Optional[FrozenSet[str]] = None,
        allowed_tenants: Optional[FrozenSet[str]] = None,
        allowed_subjects: Optional[FrozenSet[str]] = None,
    ):
        self._required_scopes = required_scopes or frozenset()
        self._allowed_tenants = allowed_tenants
        self._allowed_subjects = allowed_subjects

    @classmethod
    def from_catalyst(
        cls,
        *,
        required_scopes: Optional[FrozenSet[str]] = None,
        allowed_tenants: Optional[FrozenSet[str]] = None,
        allowed_subjects: Optional[FrozenSet[str]] = None,
    ) -> "OAuthPlugin":
        return cls(
            required_scopes=required_scopes,
            allowed_tenants=allowed_tenants,
            allowed_subjects=allowed_subjects,
        )

    def configure(self, agent: object) -> None:
        pass

    async def on_event(self, ctx: LifecycleContext) -> Optional[HookDecision]:
        if ctx.event != LifecycleEvent.BEFORE_AGENT_INVOKE:
            return None

        claims = self._extract_verified_claims(ctx)
        if claims is None:
            return Deny(
                code="oauth.missing_caller_claims",
                details={"reason": "sidecar did not deliver verified claims"},
            )

        ctx.caller = CallerIdentity(
            subject=claims.get("sub", ""),
            tenant=claims.get("tenant"),
            scopes=frozenset(claims.get("scopes", []) or []),
            issuer_id=claims.get("iss"),
            claims=claims,
        )

        return self._apply_policy(ctx.caller)

    def _extract_verified_claims(self, ctx: LifecycleContext) -> Optional[dict]:
        """Read the verified claims the sidecar delivered on the TriggerAction."""
        trigger = getattr(ctx, "trigger_action", None)
        if trigger and getattr(trigger, "caller_claims", None):
            return trigger.caller_claims
        return None

    def _apply_policy(self, caller: CallerIdentity) -> HookDecision:
        if self._required_scopes and not (self._required_scopes <= caller.scopes):
            missing = self._required_scopes - caller.scopes
            return Deny(
                code="oauth.missing_scope",
                details={
                    "required": sorted(self._required_scopes),
                    "missing": sorted(missing),
                },
            )
        if (
            self._allowed_tenants is not None
            and caller.tenant not in self._allowed_tenants
        ):
            return Deny(
                code="oauth.tenant_not_allowed",
                details={"tenant": caller.tenant},
            )
        if (
            self._allowed_subjects is not None
            and caller.subject not in self._allowed_subjects
        ):
            return Deny(
                code="oauth.subject_not_allowed",
                details={"subject": caller.subject},
            )
        return Proceed()
