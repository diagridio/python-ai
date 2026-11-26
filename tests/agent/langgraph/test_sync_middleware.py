# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""An agent whose middleware has only a sync ``wrap_model_call`` / ``wrap_tool_call``.

The runner calls a node with an async entry point through ``ainvoke``, where
such a middleware raises ``NotImplementedError``. ``invoke`` works natively,
so the runner falls back to the node's sync entry for that error.
"""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from .log_helpers import log_runner
from .test_runner import _RunnerTestCase


@tool
def double(x: int) -> int:
    """Double a number."""
    return x * 2


class _SyncOnly(AgentMiddleware):
    def __init__(self) -> None:
        super().__init__()
        self.model_calls = 0
        self.tool_calls = 0

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        self.model_calls += 1
        return handler(request)

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        self.tool_calls += 1
        return handler(request)


class _ToolModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def _model() -> Any:
    return _ToolModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "double", "args": {"x": 21}, "id": "c1"}],
                ),
                AIMessage(content="42"),
            ]
        )
    )


class TestSyncOnlyMiddleware(_RunnerTestCase):
    def test_an_agent_with_sync_only_wraps_runs_under_the_runner(self):
        middleware = _SyncOnly()
        agent = create_agent(_model(), tools=[double], middleware=[middleware])
        runner = log_runner(agent, "sync-mw")
        out = runner.invoke(
            {"messages": [{"role": "user", "content": "double 21"}]},
            poll_interval=0.01,
            timeout=20,
        )
        self.assertEqual(out["messages"][-1]["content"], "42")
        self.assertEqual(middleware.model_calls, 2)
        self.assertEqual(middleware.tool_calls, 1)
