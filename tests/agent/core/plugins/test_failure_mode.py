# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Deeper failure-mode tests for the plugin chain.

Covers error details in Deny, mixed chains with both open and closed
plugins, consecutive open failures, and exception message propagation.
"""

import logging

from dapr_agents.hooks import Deny, Proceed

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
        raises=None,
    ):
        self.name = name
        self.priority = priority
        self.capabilities = capabilities or frozenset(LifecycleEvent)
        self.failure_mode = failure_mode
        self._return = on_event_return
        self._raises = raises
        self.invocations = 0

    def configure(self, agent):
        pass

    async def on_event(self, ctx):
        self.invocations += 1
        if self._raises:
            raise self._raises
        return self._return


# ----- closed: Deny contains error detail -----


def test_closed_deny_includes_error_string():
    p = FakePlugin(
        name="oauth", failure_mode="closed", raises=ValueError("token expired")
    )
    reg = PluginRegistry([p])
    result = reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert result["type"] == "deny"
    assert result["code"] == "plugin.oauth.exception"
    assert "token expired" in result["details"]["error"]
    assert result["details"]["plugin"] == "oauth"


def test_closed_deny_code_reflects_plugin_name():
    p = FakePlugin(
        name="my-custom-guard", failure_mode="closed", raises=RuntimeError("nope")
    )
    reg = PluginRegistry([p])
    result = reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert result["code"] == "plugin.my-custom-guard.exception"


# ----- open: downstream still runs -----


def test_open_failure_lets_downstream_deny():
    p_open = FakePlugin(
        name="telemetry", priority=10, failure_mode="open", raises=IOError("flush")
    )
    p_guard = FakePlugin(
        name="guard", priority=20, failure_mode="closed", on_event_return=Deny(code="x")
    )
    reg = PluginRegistry([p_open, p_guard])
    result = reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert result["type"] == "deny"
    assert result["code"] == "x"
    assert p_guard.invocations == 1


def test_open_failure_lets_downstream_proceed():
    p_open = FakePlugin(
        name="telemetry", priority=10, failure_mode="open", raises=IOError("flush")
    )
    p_ok = FakePlugin(name="ok", priority=20, on_event_return=None)
    reg = PluginRegistry([p_open, p_ok])
    result = reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert result is None
    assert p_ok.invocations == 1


# ----- mixed chain: open then closed -----


def test_open_then_closed_both_raise():
    p_open = FakePlugin(
        name="metrics", priority=10, failure_mode="open", raises=RuntimeError("a")
    )
    p_closed = FakePlugin(
        name="auth", priority=20, failure_mode="closed", raises=RuntimeError("b")
    )
    p_tail = FakePlugin(name="tail", priority=30)
    reg = PluginRegistry([p_open, p_closed, p_tail])
    result = reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert result["type"] == "deny"
    assert result["code"] == "plugin.auth.exception"
    assert p_tail.invocations == 0


def test_closed_before_open_short_circuits_immediately():
    p_closed = FakePlugin(
        name="auth", priority=10, failure_mode="closed", raises=RuntimeError("x")
    )
    p_open = FakePlugin(
        name="metrics", priority=20, failure_mode="open", raises=RuntimeError("y")
    )
    reg = PluginRegistry([p_closed, p_open])
    result = reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert result["type"] == "deny"
    assert result["code"] == "plugin.auth.exception"
    assert p_open.invocations == 0


# ----- consecutive open failures -----


def test_multiple_open_failures_all_proceed():
    p1 = FakePlugin(
        name="a", priority=10, failure_mode="open", raises=RuntimeError("1")
    )
    p2 = FakePlugin(
        name="b", priority=20, failure_mode="open", raises=RuntimeError("2")
    )
    p3 = FakePlugin(name="c", priority=30, on_event_return=None)
    reg = PluginRegistry([p1, p2, p3])
    result = reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert result is None
    assert p1.invocations == 1
    assert p2.invocations == 1
    assert p3.invocations == 1


# ----- open failure logs warning -----


def test_open_failure_logs_warning(caplog):
    p = FakePlugin(name="telemetry", failure_mode="open", raises=ValueError("oops"))
    reg = PluginRegistry([p])
    with caplog.at_level(logging.WARNING, logger="diagrid.agent.core.plugins.registry"):
        reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert any(
        "telemetry" in r.message or "telemetry" in str(getattr(r, "plugin", ""))
        for r in caplog.records
    )


def test_closed_failure_logs_error(caplog):
    p = FakePlugin(name="auth", failure_mode="closed", raises=ValueError("bad"))
    reg = PluginRegistry([p])
    with caplog.at_level(logging.ERROR, logger="diagrid.agent.core.plugins.registry"):
        reg.dispatch("BEFORE_AGENT_INVOKE", {})
    assert len(caplog.records) >= 1
