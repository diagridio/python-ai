# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Edge-case tests for the plugin SPI chain semantics.

Complements test_registry.py with scenarios that exercise less-obvious
dispatch paths: Mutate-then-Deny, key overwrite, decision serialization
branches, and context build defaults.
"""

import asyncio

from dapr_agents.hooks import Deny, Mutate, Proceed, RequireApproval, Skip

from diagrid.agent.core.plugins.registry import PluginRegistry
from diagrid.agent.core.plugins.spi import LifecycleEvent


class FakePlugin:
    def __init__(
        self,
        name="fake",
        priority=100,
        capabilities=None,
        failure_mode="closed",
        on_event_return=None,
    ):
        self.name = name
        self.priority = priority
        self.capabilities = capabilities or frozenset(LifecycleEvent)
        self.failure_mode = failure_mode
        self._return = on_event_return
        self.invocations = 0

    def configure(self, agent):
        pass

    async def on_event(self, ctx):
        self.invocations += 1
        return self._return


# ----- Mutate + short-circuit interaction -----


def test_mutate_then_deny_returns_deny_not_mutate():
    p1 = FakePlugin(
        name="enrich", priority=10, on_event_return=Mutate(payload={"a": 1})
    )
    p2 = FakePlugin(name="guard", priority=20, on_event_return=Deny(code="blocked"))
    p3 = FakePlugin(name="tail", priority=30)
    reg = PluginRegistry([p1, p2, p3])
    result = reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert result["type"] == "deny"
    assert result["code"] == "blocked"
    assert p3.invocations == 0


def test_mutate_key_overwrite():
    p1 = FakePlugin(name="a", priority=10, on_event_return=Mutate(payload={"k": "old"}))
    p2 = FakePlugin(name="b", priority=20, on_event_return=Mutate(payload={"k": "new"}))
    reg = PluginRegistry([p1, p2])
    result = reg.dispatch("BEFORE_LLM_CALL", {})
    assert result["payload"]["k"] == "new"


def test_mutate_with_none_payload():
    p = FakePlugin(name="m", priority=10, on_event_return=Mutate(payload=None))
    reg = PluginRegistry([p])
    result = reg.dispatch("BEFORE_LLM_CALL", {})
    assert result is None


def test_mutate_then_skip_returns_skip():
    p1 = FakePlugin(name="a", priority=10, on_event_return=Mutate(payload={"x": 1}))
    p2 = FakePlugin(name="b", priority=20, on_event_return=Skip(result="cached"))
    reg = PluginRegistry([p1, p2])
    result = reg.dispatch("BEFORE_TOOL_CALL", {})
    assert result["type"] == "skip"
    assert result["result"] == "cached"


# ----- empty chain -----


def test_empty_registry_returns_none():
    reg = PluginRegistry([])
    assert reg.dispatch("BEFORE_AGENT_INVOKE", {}) is None


# ----- _to_decision_dict serialization -----


def test_deny_dict_all_fields():
    p = FakePlugin(
        priority=10,
        on_event_return=Deny(code="auth.fail", reason="expired", details={"ttl": 0}),
    )
    reg = PluginRegistry([p])
    result = reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert result == {
        "type": "deny",
        "code": "auth.fail",
        "reason": "expired",
        "details": {"ttl": 0},
    }


def test_deny_dict_minimal():
    p = FakePlugin(priority=10, on_event_return=Deny())
    reg = PluginRegistry([p])
    result = reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert result == {"type": "deny"}


def test_require_approval_dict_all_fields():
    p = FakePlugin(
        priority=10,
        on_event_return=RequireApproval(
            timeout_seconds=300,
            instructions="please review",
            reason="sensitive",
        ),
    )
    reg = PluginRegistry([p])
    result = reg.dispatch("BEFORE_TOOL_CALL", {})
    assert result["type"] == "require_approval"
    assert result["timeout_seconds"] == 300
    assert result["instructions"] == "please review"
    assert result["reason"] == "sensitive"


def test_skip_dict():
    p = FakePlugin(priority=10, on_event_return=Skip(result={"cached": True}))
    reg = PluginRegistry([p])
    result = reg.dispatch("BEFORE_TOOL_CALL", {})
    assert result == {"type": "skip", "result": {"cached": True}}


# ----- _build_context defaults -----


def test_build_context_defaults_missing_fields():
    reg = PluginRegistry([])
    ctx = reg._build_context("BEFORE_AGENT_INVOKE", {})
    assert ctx.workflow_instance_id == ""
    assert ctx.agent_identity == {}
    assert ctx.caller is None
    assert ctx.metadata == {}
    assert ctx.payload == {}


def test_build_context_ignores_unknown_keys():
    reg = PluginRegistry([])
    ctx = reg._build_context(
        "BEFORE_AGENT_INVOKE", {"unknown_field": 42, "workflow_instance_id": "wf-1"}
    )
    assert ctx.workflow_instance_id == "wf-1"
    assert not hasattr(ctx, "unknown_field")


# ----- capabilities: multi-event selective dispatch -----


def test_multi_capability_selective_dispatch():
    p = FakePlugin(
        name="multi",
        capabilities=frozenset(
            {LifecycleEvent.BEFORE_AGENT_INVOKE, LifecycleEvent.BEFORE_TOOL_CALL}
        ),
        on_event_return=Deny(code="x"),
    )
    reg = PluginRegistry([p])
    assert reg.dispatch("BEFORE_AGENT_INVOKE", {})["type"] == "deny"
    assert reg.dispatch("BEFORE_TOOL_CALL", {})["type"] == "deny"
    assert reg.dispatch("BEFORE_LLM_CALL", {}) is None
    assert reg.dispatch("AFTER_AGENT_INVOKE", {}) is None
    assert p.invocations == 2


# ----- three-plugin chain ordering -----


def test_three_plugin_chain_accumulation():
    seen_by_third = {}

    class Observer(FakePlugin):
        async def on_event(self, ctx):
            self.invocations += 1
            seen_by_third.update(ctx.payload)
            return None

    p1 = FakePlugin(name="a", priority=10, on_event_return=Mutate(payload={"step": 1}))
    p2 = FakePlugin(name="b", priority=20, on_event_return=Mutate(payload={"step": 2}))
    p3 = Observer(name="c", priority=30)
    reg = PluginRegistry([p3, p1, p2])
    result = reg.dispatch("BEFORE_LLM_CALL", {})
    assert seen_by_third == {"step": 2}
    assert result["type"] == "mutate"
    assert result["payload"] == {"step": 2}


# ----- dispatch from async context -----


def test_dispatch_from_nested_async():
    p = FakePlugin(priority=10, on_event_return=Skip(result="ok"))
    reg = PluginRegistry([p])

    async def outer():
        return reg.dispatch("BEFORE_TOOL_CALL", {})

    result = asyncio.run(outer())
    assert result["type"] == "skip"


# ----- explicit Proceed vs None -----


def test_explicit_proceed_same_as_none():
    p1 = FakePlugin(name="a", priority=10, on_event_return=Proceed())
    p2 = FakePlugin(name="b", priority=20, on_event_return=None)
    reg = PluginRegistry([p1, p2])
    assert reg.dispatch("BEFORE_AGENT_INVOKE", {}) is None
    assert p1.invocations == 1
    assert p2.invocations == 1
