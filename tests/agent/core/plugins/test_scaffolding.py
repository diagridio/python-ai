# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Smoke tests for the plugins module scaffolding.

Behavioral tests for the registry land alongside its dispatch
implementation.
"""

import pytest

from diagrid.agent.core.plugins import (
    Plugin,
    LifecycleEvent,
    HookDecision,
    Proceed,
    Skip,
    Mutate,
    RequireApproval,
    Deny,
    LifecycleContext,
    CallerIdentity,
    CallTarget,
    PluginRegistry,
)


def test_public_api_imports():
    """All advertised public names can be imported from the package."""
    assert Plugin is not None
    assert LifecycleEvent is not None
    assert HookDecision is not None
    for decision in (Proceed, Skip, Mutate, RequireApproval, Deny):
        assert decision is not None


def test_lifecycle_event_values():
    """LifecycleEvent enum has the expected event names."""
    expected = {
        "BEFORE_AGENT_INVOKE",
        "AFTER_AGENT_INVOKE",
        "BEFORE_LLM_CALL",
        "AFTER_LLM_CALL",
        "BEFORE_TOOL_CALL",
        "AFTER_TOOL_CALL",
        "BEFORE_MCP_CALL",
        "BEFORE_APPROVAL_DECISION",
    }
    assert {e.value for e in LifecycleEvent} == expected


def test_plugin_protocol_runtime_checkable():
    """A class implementing the full Plugin surface satisfies the Protocol."""

    class FakePlugin:
        name = "fake"
        priority = 100
        capabilities = frozenset({LifecycleEvent.BEFORE_TOOL_CALL})
        failure_mode = "closed"

        def configure(self, agent):
            pass

        async def on_event(self, ctx):
            return Proceed()

    assert isinstance(FakePlugin(), Plugin)


def test_lifecycle_context_construction():
    ctx = LifecycleContext(
        event=LifecycleEvent.BEFORE_AGENT_INVOKE,
        workflow_instance_id="wf-1",
        agent_identity={"spiffe_id": "spiffe://...", "app_id": "agent-svc"},
    )
    assert ctx.event == LifecycleEvent.BEFORE_AGENT_INVOKE
    assert ctx.caller is None
    assert ctx.target is None
    assert ctx.metadata == {}


def test_caller_identity_construction():
    caller = CallerIdentity(
        subject="alice@acme.com",
        tenant="acme",
        scopes=frozenset({"agent.invoke"}),
    )
    assert caller.subject == "alice@acme.com"
    assert caller.is_agent is False


def test_call_target_construction():
    target = CallTarget(kind="mcp", name="weather", source="mcp")
    assert target.kind == "mcp"
    assert target.audience is None
    assert target.resource_indicators == frozenset()


def test_plugin_registry_sorts_by_priority():
    """Plugins are ordered by ascending priority, ties keep insertion order."""

    class StubA:
        name = "a"
        priority = 50
        capabilities = frozenset()
        failure_mode = "open"

        def configure(self, agent):
            pass

        async def on_event(self, ctx):
            return None

    class StubB:
        name = "b"
        priority = 10
        capabilities = frozenset()
        failure_mode = "closed"

        def configure(self, agent):
            pass

        async def on_event(self, ctx):
            return None

    registry = PluginRegistry([StubA(), StubB()])
    assert [p.name for p in registry._plugins] == ["b", "a"]


def test_plugin_registry_dispatch_not_implemented_yet():
    """The scaffold ships without dispatch; it raises NotImplementedError."""
    registry = PluginRegistry()
    with pytest.raises(NotImplementedError):
        registry.dispatch("BEFORE_AGENT_INVOKE", {})
