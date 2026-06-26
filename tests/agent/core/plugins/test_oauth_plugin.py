# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import pytest
from unittest.mock import AsyncMock

from dapr_agents.hooks import Deny, Proceed

from diagrid.agent.core.plugins.context import LifecycleContext
from diagrid.agent.core.plugins.spi import LifecycleEvent
from diagrid.agent.core.plugins.oauth import OAuthPlugin
from diagrid.agent.core.plugins.oauth.client import AuthVerifyResponse


def make_ctx(headers: dict) -> LifecycleContext:
    ctx = LifecycleContext(event=LifecycleEvent.BEFORE_AGENT_INVOKE)
    ctx.caller_headers = headers
    return ctx


@pytest.mark.asyncio
async def test_valid_token_populates_caller_and_proceeds(monkeypatch):
    plugin = OAuthPlugin(sidecar_url="http://localhost:3500")
    plugin._client.verify = AsyncMock(
        return_value=AuthVerifyResponse(
            verified=True,
            claims={
                "sub": "alice@acme.com",
                "tenant": "acme",
                "scopes": ["agent.invoke"],
                "email": "alice@acme.com",
            },
            issuer_id="catalyst-sentry",
        )
    )
    ctx = make_ctx({"Authorization": "Bearer eyJhbGciOi..."})
    result = await plugin.on_event(ctx)
    assert isinstance(result, Proceed)
    assert ctx.caller.subject == "alice@acme.com"
    assert ctx.caller.tenant == "acme"
    assert "agent.invoke" in ctx.caller.scopes
    assert ctx.caller.issuer_id == "catalyst-sentry"


@pytest.mark.asyncio
async def test_raw_token_scrubbed_after_verification(monkeypatch):
    plugin = OAuthPlugin(sidecar_url="http://localhost:3500")
    plugin._client.verify = AsyncMock(
        return_value=AuthVerifyResponse(
            verified=True,
            claims={"sub": "alice"},
            issuer_id="catalyst-sentry",
        )
    )
    headers = {"Authorization": "Bearer eyJhbG..."}
    ctx = make_ctx(headers)
    await plugin.on_event(ctx)
    assert "Authorization" not in headers
    assert "authorization" not in headers


@pytest.mark.asyncio
async def test_missing_token_denies():
    plugin = OAuthPlugin(sidecar_url="http://localhost:3500")
    ctx = make_ctx({})  # no Authorization header
    result = await plugin.on_event(ctx)
    assert isinstance(result, Deny)
    assert result.code == "oauth.missing_token"


@pytest.mark.asyncio
async def test_invalid_signature_denies(monkeypatch):
    plugin = OAuthPlugin(sidecar_url="http://localhost:3500")
    plugin._client.verify = AsyncMock(
        return_value=AuthVerifyResponse(
            verified=False,
            error="invalid_signature",
            error_description="JWT signature verification failed",
        )
    )
    ctx = make_ctx({"Authorization": "Bearer tampered"})
    result = await plugin.on_event(ctx)
    assert isinstance(result, Deny)
    assert result.code == "oauth.invalid_signature"


@pytest.mark.asyncio
async def test_expired_token_denies(monkeypatch):
    plugin = OAuthPlugin(sidecar_url="http://localhost:3500")
    plugin._client.verify = AsyncMock(
        return_value=AuthVerifyResponse(
            verified=False, error="expired", error_description="token expired"
        )
    )
    ctx = make_ctx({"Authorization": "Bearer eyJold"})
    result = await plugin.on_event(ctx)
    assert isinstance(result, Deny)
    assert result.code == "oauth.expired"


@pytest.mark.asyncio
async def test_invalid_audience_denies(monkeypatch):
    plugin = OAuthPlugin(
        sidecar_url="http://localhost:3500",
        expected_audience="spiffe://.../my-agent",
    )
    plugin._client.verify = AsyncMock(
        return_value=AuthVerifyResponse(
            verified=False,
            error="invalid_audience",
            error_description="audience mismatch",
        )
    )
    ctx = make_ctx({"Authorization": "Bearer wrong-aud"})
    result = await plugin.on_event(ctx)
    assert isinstance(result, Deny)
    assert result.code == "oauth.invalid_audience"


@pytest.mark.asyncio
async def test_sidecar_unavailable_denies_closed(monkeypatch):
    plugin = OAuthPlugin(sidecar_url="http://localhost:3500")
    plugin._client.verify = AsyncMock(side_effect=ConnectionError("sidecar down"))
    ctx = make_ctx({"Authorization": "Bearer x"})
    result = await plugin.on_event(ctx)
    assert isinstance(result, Deny)
    assert result.code == "oauth.verify_unavailable"


@pytest.mark.asyncio
async def test_plugin_ignores_non_inbound_events():
    plugin = OAuthPlugin(sidecar_url="http://localhost:3500")
    ctx = LifecycleContext(event=LifecycleEvent.BEFORE_TOOL_CALL)
    result = await plugin.on_event(ctx)
    assert result is None


def test_plugin_capabilities_only_inbound():
    plugin = OAuthPlugin(sidecar_url="http://localhost:3500")
    assert plugin.capabilities == frozenset({LifecycleEvent.BEFORE_AGENT_INVOKE})


def test_failure_mode_is_closed():
    plugin = OAuthPlugin(sidecar_url="http://localhost:3500")
    assert plugin.failure_mode == "closed"


# TODO(AI-592): Add tests for LifecycleContext sourcing caller_headers from
# TriggerAction once `TriggerAction.caller_headers` lands. These should cover
# the header-propagation path end-to-end (TriggerAction -> LifecycleContext ->
# OAuthPlugin._extract_bearer), which is stubbed here via make_ctx.

# TODO(AI-595): Add e2e tests for the lifecycle dispatcher (AI-596
# `lifecycle_dispatcher` kwarg + BEFORE_AGENT_INVOKE dispatch site) once it is
# wired into DurableAgent, verifying the OAuthPlugin actually fires at ingress.

# TODO(AI-599): Add chain integration tests for the PluginRegistry dispatcher —
# OAuthPlugin (priority=10) running first, short-circuiting the chain on Deny,
# and propagating ctx.caller to downstream plugins on Proceed.
