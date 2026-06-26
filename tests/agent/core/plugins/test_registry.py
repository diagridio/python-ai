# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

from dapr_agents.hooks import Deny, Mutate, Proceed, RequireApproval, Skip

from diagrid.agent.core.plugins.registry import PluginRegistry
from diagrid.agent.core.plugins.spi import LifecycleEvent


class FakePlugin:
    """Minimal Plugin Protocol implementation for tests."""

    def __init__(
        self,
        name="fake",
        priority=100,
        capabilities=None,
        failure_mode="closed",
        on_event_return=None,
        raises=None,
    ):
        self.name = name
        self.priority = priority
        self.capabilities = capabilities or frozenset(LifecycleEvent)
        self.failure_mode = failure_mode
        self._return = on_event_return
        self._raises = raises
        self.configured_with = None
        self.invocations = 0

    def configure(self, agent):
        self.configured_with = agent

    async def on_event(self, ctx):
        self.invocations += 1
        if self._raises:
            raise self._raises
        return self._return


# ----- Protocol conformance -----


def test_registry_satisfies_lifecycle_dispatcher_protocol():
    from dapr_agents.lifecycle import LifecycleDispatcher

    reg = PluginRegistry()
    assert isinstance(reg, LifecycleDispatcher)


# ----- attach/detach -----


def test_attach_calls_configure_on_every_plugin():
    p1 = FakePlugin(name="p1")
    p2 = FakePlugin(name="p2")
    reg = PluginRegistry([p1, p2])
    agent = object()
    reg.attach(agent)
    assert p1.configured_with is agent
    assert p2.configured_with is agent


def test_detach_clears_agent_ref():
    reg = PluginRegistry([FakePlugin()])
    reg.attach(object())
    reg.detach()
    assert reg._agent is None


# ----- ordering -----


def test_plugins_run_in_priority_order():
    calls = []

    class Recorder(FakePlugin):
        async def on_event(self, ctx):
            calls.append(self.name)
            return None

    p_high = Recorder(name="high", priority=10)
    p_low = Recorder(name="low", priority=100)
    reg = PluginRegistry([p_low, p_high])  # registered out of order
    reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert calls == ["high", "low"]


def test_priority_ties_break_by_registration_order():
    calls = []

    class Recorder(FakePlugin):
        async def on_event(self, ctx):
            calls.append(self.name)
            return None

    p_a = Recorder(name="a", priority=50)
    p_b = Recorder(name="b", priority=50)
    reg = PluginRegistry([p_a, p_b])
    reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert calls == ["a", "b"]


# ----- short-circuit semantics -----


def test_deny_short_circuits():
    p1 = FakePlugin(name="p1", priority=10, on_event_return=Deny(code="x"))
    p2 = FakePlugin(name="p2", priority=20)
    reg = PluginRegistry([p1, p2])
    result = reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert result["type"] == "deny"
    assert result["code"] == "x"
    assert p2.invocations == 0


def test_skip_short_circuits():
    p1 = FakePlugin(name="p1", priority=10, on_event_return=Skip(result="cached"))
    p2 = FakePlugin(name="p2", priority=20)
    reg = PluginRegistry([p1, p2])
    result = reg.dispatch("BEFORE_TOOL_CALL", {})
    assert result["type"] == "skip"
    assert result["result"] == "cached"
    assert p2.invocations == 0


def test_require_approval_short_circuits():
    p1 = FakePlugin(
        name="p1",
        priority=10,
        on_event_return=RequireApproval(timeout_seconds=600),
    )
    p2 = FakePlugin(name="p2", priority=20)
    reg = PluginRegistry([p1, p2])
    result = reg.dispatch("BEFORE_TOOL_CALL", {})
    assert result["type"] == "require_approval"
    assert result["timeout_seconds"] == 600
    assert p2.invocations == 0


# ----- Mutate accumulation -----


def test_mutate_accumulates_through_chain():
    p1 = FakePlugin(name="p1", priority=10, on_event_return=Mutate(payload={"a": 1}))
    p2 = FakePlugin(name="p2", priority=20, on_event_return=Mutate(payload={"b": 2}))
    reg = PluginRegistry([p1, p2])
    result = reg.dispatch("BEFORE_LLM_CALL", {})
    assert result["type"] == "mutate"
    assert result["payload"] == {"a": 1, "b": 2}


def test_mutate_then_proceed_emits_accumulated_mutate():
    p1 = FakePlugin(name="p1", priority=10, on_event_return=Mutate(payload={"x": 1}))
    p2 = FakePlugin(name="p2", priority=20, on_event_return=None)
    reg = PluginRegistry([p1, p2])
    result = reg.dispatch("BEFORE_LLM_CALL", {})
    assert result["type"] == "mutate"
    assert result["payload"] == {"x": 1}


def test_all_proceed_returns_none():
    p1 = FakePlugin(name="p1", priority=10, on_event_return=Proceed())
    p2 = FakePlugin(name="p2", priority=20, on_event_return=None)
    reg = PluginRegistry([p1, p2])
    result = reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert result is None


# ----- failure_mode -----


def test_failure_mode_closed_denies_on_exception():
    p1 = FakePlugin(
        name="oauth",
        priority=10,
        failure_mode="closed",
        raises=RuntimeError("boom"),
    )
    p2 = FakePlugin(name="next", priority=20)
    reg = PluginRegistry([p1, p2])
    result = reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert result["type"] == "deny"
    assert result["code"] == "plugin.oauth.exception"
    assert p2.invocations == 0


def test_failure_mode_open_proceeds_on_exception():
    p1 = FakePlugin(
        name="telemetry",
        priority=10,
        failure_mode="open",
        raises=RuntimeError("boom"),
    )
    p2 = FakePlugin(name="next", priority=20)
    reg = PluginRegistry([p1, p2])
    result = reg.dispatch("BEFORE_AGENT_INVOKE", {})
    # p2 still runs; no decision returned (all Proceed)
    assert result is None
    assert p2.invocations == 1


# ----- capabilities gating -----


def test_plugin_only_invoked_for_declared_capabilities():
    p = FakePlugin(
        name="oauth",
        capabilities=frozenset({LifecycleEvent.BEFORE_AGENT_INVOKE}),
    )
    reg = PluginRegistry([p])
    reg.dispatch("BEFORE_TOOL_CALL", {})
    assert p.invocations == 0
    reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert p.invocations == 1


# ----- sync hook auto-wrap -----


def test_sync_hook_auto_wrapped():
    class SyncPlugin(FakePlugin):
        def on_event(self, ctx):  # sync — not async
            self.invocations += 1
            return None

    p = SyncPlugin(name="sync")
    reg = PluginRegistry([p])
    reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert p.invocations == 1
