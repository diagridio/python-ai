# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""DurableAgent keeps conversation history in ``agent.messages``."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("strands")

from diagrid.agent.strands.durable_agent import DurableAgent  # noqa: E402


class _EchoModel:
    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": f"saw {len(messages)}"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


class _BrokenModel:
    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover


def _make_durable(model: Any = None) -> tuple[DurableAgent, dict[str, Any], MagicMock]:
    agent = SimpleNamespace(
        model=model or _EchoModel(),
        system_prompt="",
        tool_registry=SimpleNamespace(registry={}),
        messages=[],
    )
    registered: dict[str, Any] = {}
    runtime = MagicMock()
    runtime.register_workflow.side_effect = lambda fn, name=None: registered.update(
        workflow=fn
    )
    runtime.register_activity.side_effect = lambda fn, name=None: registered.update(
        {name: fn}
    )
    client = MagicMock()
    durable = DurableAgent(agent=agent, name="chat")  # type: ignore[arg-type]

    def _schedule(workflow, input, instance_id):
        ctx = SimpleNamespace(call_activity=lambda fn, input=None: (fn, input))
        gen = workflow(ctx, input)
        try:
            step = next(gen)
            while True:
                fn, activity_input = step
                step = gen.send(fn(None, activity_input))
        except StopIteration as done:
            client.state = SimpleNamespace(
                runtime_status="COMPLETED",
                serialized_output=json.dumps(done.value),
            )

    client.schedule_new_workflow.side_effect = _schedule
    client.get_workflow_state.side_effect = lambda instance_id: client.state

    with (
        patch("dapr.ext.workflow.WorkflowRuntime", return_value=runtime),
        patch("dapr.ext.workflow.DaprWorkflowClient", return_value=client),
        patch("time.sleep"),
    ):
        durable._initialize_dapr()
    return durable, registered, client


def test_second_call_sees_first_turn_and_messages_property_shows_it():
    durable, _, _ = _make_durable()
    assert durable("hi") == "saw 1"
    assert durable("again") == "saw 3"
    assert len(durable.messages) == 4
    assert durable.messages[0]["content"][0]["text"] == "hi"
    assert durable.messages[-1]["content"][0]["text"] == "saw 3"


def test_messages_list_identity_is_preserved():
    durable, _, _ = _make_durable()
    before = durable.messages
    durable("hi")
    assert durable.messages is before


def test_failed_model_call_leaves_messages_untouched():
    durable, _, _ = _make_durable(model=_BrokenModel())
    assert durable("hi").startswith("Error:")
    assert durable.messages == []
