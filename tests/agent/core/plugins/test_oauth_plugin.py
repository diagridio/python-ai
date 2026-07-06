# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

from types import SimpleNamespace

import pytest

from dapr_agents.hooks import Deny, Proceed

from diagrid.agent.core.plugins.context import LifecycleContext
from diagrid.agent.core.plugins.spi import LifecycleEvent
from diagrid.agent.core.plugins.oauth import OAuthPlugin


def make_ctx(claims: dict | None) -> LifecycleContext:
    """Build a BEFORE_AGENT_INVOKE context with sidecar-delivered claims."""
    trigger = SimpleNamespace(caller_claims=claims)
    ctx = LifecycleContext(event=LifecycleEvent.BEFORE_AGENT_INVOKE)
    ctx.trigger_action = trigger
    return ctx


@pytest.mark.asyncio
async def test_verified_claims_populate_caller_and_proceed():
    plugin = OAuthPlugin()
    ctx = make_ctx(
        {
            "sub": "alice@acme.com",
            "tenant": "acme",
            "scopes": ["agent.invoke"],
            "iss": "catalyst-sentry",
        }
    )
    result = await plugin.on_event(ctx)
    assert isinstance(result, Proceed)
    assert ctx.caller.subject == "alice@acme.com"
    assert ctx.caller.tenant == "acme"
    assert "agent.invoke" in ctx.caller.scopes
    assert ctx.caller.issuer_id == "catalyst-sentry"


@pytest.mark.asyncio
async def test_missing_claims_denies_closed():
    plugin = OAuthPlugin()
    ctx = make_ctx(None)  # sidecar delivered no claims
    result = await plugin.on_event(ctx)
    assert isinstance(result, Deny)
    assert result.code == "oauth.missing_caller_claims"


@pytest.mark.asyncio
async def test_required_scope_satisfied_proceeds():
    plugin = OAuthPlugin(required_scopes=frozenset({"agent.invoke"}))
    ctx = make_ctx({"sub": "alice", "scopes": ["agent.invoke", "agent.read"]})
    result = await plugin.on_event(ctx)
    assert isinstance(result, Proceed)


@pytest.mark.asyncio
async def test_missing_required_scope_denies():
    plugin = OAuthPlugin(required_scopes=frozenset({"agent.invoke", "agent.admin"}))
    ctx = make_ctx({"sub": "alice", "scopes": ["agent.invoke"]})
    result = await plugin.on_event(ctx)
    assert isinstance(result, Deny)
    assert result.code == "oauth.missing_scope"
    assert result.details["missing"] == ["agent.admin"]


@pytest.mark.asyncio
async def test_tenant_not_allowed_denies():
    plugin = OAuthPlugin(allowed_tenants=frozenset({"acme"}))
    ctx = make_ctx({"sub": "alice", "tenant": "evilcorp"})
    result = await plugin.on_event(ctx)
    assert isinstance(result, Deny)
    assert result.code == "oauth.tenant_not_allowed"
    assert result.details["tenant"] == "evilcorp"


@pytest.mark.asyncio
async def test_subject_not_allowed_denies():
    plugin = OAuthPlugin(allowed_subjects=frozenset({"alice@acme.com"}))
    ctx = make_ctx({"sub": "mallory@acme.com"})
    result = await plugin.on_event(ctx)
    assert isinstance(result, Deny)
    assert result.code == "oauth.subject_not_allowed"
    assert result.details["subject"] == "mallory@acme.com"


@pytest.mark.asyncio
async def test_sub_workflow_caller_claims_populate_caller():
    """Sub-workflow path: parent propagates caller_claims on TriggerAction."""
    plugin = OAuthPlugin(required_scopes=frozenset({"agent.invoke"}))
    ctx = make_ctx(
        {"sub": "bob@acme.com", "tenant": "acme", "scopes": ["agent.invoke"]}
    )
    result = await plugin.on_event(ctx)
    assert isinstance(result, Proceed)
    assert ctx.caller.subject == "bob@acme.com"
    assert ctx.caller.tenant == "acme"


@pytest.mark.asyncio
async def test_plugin_ignores_non_inbound_events():
    plugin = OAuthPlugin()
    ctx = LifecycleContext(event=LifecycleEvent.BEFORE_TOOL_CALL)
    result = await plugin.on_event(ctx)
    assert result is None


def test_plugin_capabilities_only_inbound():
    plugin = OAuthPlugin()
    assert plugin.capabilities == frozenset({LifecycleEvent.BEFORE_AGENT_INVOKE})


def test_failure_mode_is_closed():
    plugin = OAuthPlugin()
    assert plugin.failure_mode == "closed"


# TODO: Add tests for LifecycleContext sourcing caller_claims from a real
# TriggerAction once that field lands (delivery path is stubbed via make_ctx).

# TODO: Add e2e tests once the lifecycle dispatcher is wired into DurableAgent,
# verifying the plugin fires at ingress.

# TODO: Add PluginRegistry chain integration tests — plugin runs first,
# short-circuits the chain on Deny, propagates ctx.caller on Proceed.
