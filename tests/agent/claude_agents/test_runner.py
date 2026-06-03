# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for the claude_agents runner — option resolution and result shape.

These tests stay narrow: they pin the public contract of the runner's
result/output plumbing without spinning up Dapr. The full end-to-end path
is covered by ``examples/test_crash_recovery.py``.
"""

from unittest import mock

from diagrid.agent.claude_agents.models import (
    AgentWorkflowOutput,
    Message,
    MessageRole,
)


def _patched_runner(**overrides):
    """Construct a runner without touching the Dapr workflow runtime.

    The runner's ``__init__`` calls ``super().__init__`` (which opens a
    workflow runtime) and ``_register_agent_metadata`` (which calls Dapr).
    For these tests we only care about option-resolution, so we patch out
    both side-effects and build a thin instance directly.
    """
    from diagrid.agent.claude_agents.runner import DaprWorkflowAgentRunner

    def _fake_base_init(self, name, *, framework, **kwargs):
        # Mirror the attrs ``DaprWorkflowAgentRunner.__init__`` reads off
        # ``self`` after calling super().__init__ — without actually opening
        # a Dapr workflow runtime.
        self._name = name
        self._framework = framework
        self._max_iterations = kwargs.get("max_iterations", 25)
        self._state_store = kwargs.get("state_store")
        self._workflow_runtime = mock.MagicMock()
        self._workflow_client = None
        self._started = False
        self._observability_config = None

    base_init_patch = mock.patch(
        "diagrid.agent.core.workflow.BaseWorkflowRunner.__init__",
        autospec=True,
        side_effect=_fake_base_init,
    )
    register_patch = mock.patch.object(
        DaprWorkflowAgentRunner, "_register_agent_metadata", return_value=None
    )
    components_patch = mock.patch.object(
        DaprWorkflowAgentRunner, "_register_workflow_components", return_value=None
    )
    tools_patch = mock.patch.object(
        DaprWorkflowAgentRunner, "_register_agent_tools", return_value=None
    )

    kwargs = {"name": "test-agent"}
    kwargs.update(overrides)

    with base_init_patch, register_patch, components_patch, tools_patch:
        runner = DaprWorkflowAgentRunner(**kwargs)
    return runner


class TestSystemPromptResolution:
    """Cover the explicit-arg → options → default fallback chain."""

    def test_explicit_system_prompt_wins(self):
        opts = mock.Mock()
        opts.system_prompt = "from-options"
        opts.model = None
        runner = _patched_runner(system_prompt="from-explicit", options=opts)
        assert runner._system_prompt == "from-explicit"

    def test_falls_back_to_options(self):
        opts = mock.Mock()
        opts.system_prompt = "from-options"
        opts.model = None
        runner = _patched_runner(options=opts)
        assert runner._system_prompt == "from-options"

    def test_empty_string_when_unset(self):
        """No explicit prompt, no options → must be the empty string, never None.

        Pinning this lets downstream code skip the ``if system_prompt`` guard
        and pass the value straight to the Anthropic SDK call.
        """
        runner = _patched_runner()
        assert runner._system_prompt == ""
        assert isinstance(runner._system_prompt, str)

    def test_options_with_preset_dict_is_dropped(self):
        """ClaudeAgentOptions.system_prompt accepts a preset dict; we only
        forward plain strings to the Messages API call, so a dict resolves
        to the empty string."""
        opts = mock.Mock()
        opts.system_prompt = {"type": "preset", "preset": "claude_code"}
        opts.model = None
        runner = _patched_runner(options=opts)
        assert runner._system_prompt == ""

    def test_explicit_empty_string_is_respected(self):
        """If a caller deliberately passes ``system_prompt=""`` the runner
        must honor it instead of falling through to options."""
        opts = mock.Mock()
        opts.system_prompt = "ignored-fallback"
        opts.model = None
        runner = _patched_runner(system_prompt="", options=opts)
        assert runner._system_prompt == ""


class TestRunSyncReturnsFullOutput:
    """``run_sync`` must surface the full ``AgentWorkflowOutput``."""

    def test_messages_round_trip_from_workflow_completed_event(self):
        """A ``workflow_completed`` event carrying serialized messages must
        be re-hydrated into ``AgentWorkflowOutput.messages`` (regression test
        for the bug where ``run_sync`` returned ``messages=[]``)."""
        runner = _patched_runner()

        async def fake_run_async(user_message, session_id, *, workflow_id=None):
            yield {
                "type": "workflow_completed",
                "workflow_id": "wf-1",
                "final_response": "all done",
                "messages": [
                    Message(role=MessageRole.USER, content="hello").to_dict(),
                    Message(role=MessageRole.ASSISTANT, content="hi back").to_dict(),
                ],
                "iterations": 1,
                "status": "completed",
                "error": None,
            }

        with (
            mock.patch.object(runner, "run_async", fake_run_async),
            mock.patch.object(
                runner, "_run_sync", side_effect=lambda coro, timeout: _drain(coro)
            ),
        ):
            result = runner.run_sync("hello", "sess-1")

        assert isinstance(result, AgentWorkflowOutput)
        assert result.final_response == "all done"
        assert result.iterations == 1
        assert result.status == "completed"
        assert len(result.messages) == 2
        assert result.messages[0].role == MessageRole.USER
        assert result.messages[0].content == "hello"
        assert result.messages[1].role == MessageRole.ASSISTANT
        assert result.messages[1].content == "hi back"

    def test_error_field_propagates(self):
        """When the workflow completes with a non-success status, the
        ``error`` field must reach the returned ``AgentWorkflowOutput``."""
        runner = _patched_runner()

        async def fake_run_async(user_message, session_id, *, workflow_id=None):
            yield {
                "type": "workflow_completed",
                "workflow_id": "wf-1",
                "final_response": None,
                "messages": [],
                "iterations": 3,
                "status": "max_iterations_reached",
                "error": "Max iterations (3) reached",
            }

        with (
            mock.patch.object(runner, "run_async", fake_run_async),
            mock.patch.object(
                runner, "_run_sync", side_effect=lambda coro, timeout: _drain(coro)
            ),
        ):
            result = runner.run_sync("hello", "sess-1")

        assert result.status == "max_iterations_reached"
        assert result.error == "Max iterations (3) reached"
        assert result.final_response is None


def _drain(coro):
    """Run an async coroutine to completion on a fresh loop.

    Mirrors what the real ``_run_sync`` does, minus the timeout — fine for
    the in-process fake ``run_async`` used in these tests.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
