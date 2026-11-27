# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``RemainingSteps`` and ``IsLastStep`` are given to nodes and routers.

``create_react_agent`` reads ``remaining_steps`` to answer "Sorry, need more
steps" before it runs into the limit; a node or router that reads them got
``None`` here.
"""

import asyncio
import operator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.prebuilt import create_react_agent
from typing_extensions import Annotated, TypedDict

from .golden_harness import _FakeDaprTest, _runner


class _S(TypedDict, total=False):
    log: Annotated[list, operator.add]
    remaining_steps: RemainingSteps
    last: IsLastStep


def _chain() -> Any:
    def record(name: str) -> Any:
        def node(state: _S) -> dict:
            return {"log": [(name, state.get("remaining_steps"), state.get("last"))]}

        return node

    graph = StateGraph(_S)
    for name in ("a", "b", "c"):
        graph.add_node(name, record(name))
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")

    def route(state: _S) -> str:
        # a router reads them too, at the step of the node it follows: b's
        wanted = (state.get("remaining_steps"), state.get("last"))
        return "c" if wanted[0] is not None and wanted[1] is False else END

    graph.add_conditional_edges("b", route, ["c", END])
    graph.add_edge("c", END)
    return graph.compile()


class TestManagedValues(_FakeDaprTest):
    def _invoke(self, runner: Any, config: Any) -> Any:
        return runner.invoke({"log": []}, config=config, poll_interval=0.01)

    async def test_nodes_read_the_values_langgraph_gives_them(self):
        compiled = _chain()
        for limit in (4, 6, 25):
            with self.subTest(limit=limit):
                self.setUp()
                config = {"recursion_limit": limit}
                native = [list(e) for e in compiled.invoke({"log": []}, config)["log"]]
                self.assertEqual(native[0][1], limit - 1)
                self.assertEqual(len(native), 3)
                runner = _runner(compiled, f"managed-{limit}")
                out = await asyncio.to_thread(self._invoke, runner, config)
                self.assertEqual(out["log"], native)

    async def test_the_last_step_is_flagged(self):
        compiled = _chain()
        native = compiled.invoke({"log": []}, {"recursion_limit": 4})["log"]
        self.assertEqual([step[2] for step in native], [False, False, True])
        runner = _runner(compiled, "managed-last")
        out = await asyncio.to_thread(self._invoke, runner, {"recursion_limit": 4})
        self.assertEqual([step[2] for step in out["log"]], [False, False, True])


@tool
def ping() -> str:
    """Ping."""
    return "pong"


class _AlwaysTools(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "always-tools"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_AlwaysTools":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        call = {"name": "ping", "args": {}, "id": f"c{len(messages)}"}
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="", tool_calls=[call]))
            ]
        )


class TestReactAgentNearTheLimit(_FakeDaprTest):
    async def test_it_answers_that_it_needs_more_steps_as_native_does(self):
        agent = create_react_agent(_AlwaysTools(), [ping])
        config = {"recursion_limit": 6}
        native = agent.invoke({"messages": [("user", "go")]}, config)
        self.assertIn("need more steps", native["messages"][-1].content)
        runner = _runner(agent, "managed-react")
        out = await asyncio.to_thread(
            lambda: runner.invoke(
                {"messages": [{"role": "user", "content": "go"}]},
                config=config,
                poll_interval=0.01,
            )
        )
        self.assertIn("need more steps", out["messages"][-1]["content"])
