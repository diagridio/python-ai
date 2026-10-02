"""E2E: standard LangGraph building blocks under ``DaprWorkflowGraphRunner``.

Runs, against a real Dapr sidecar, graph shapes the runner used to break:

- the prebuilt ``ToolNode`` (its ``runtime`` argument was never injected)
- a node declaring a ``runtime`` parameter
- a conditional edge whose router returns ``path_map`` labels, not node names
- ``async def`` nodes and routers

Deterministic, no LLM required (marked ``integration``, not ``ollama``).
"""

from typing import Any, TypedDict

import pytest

from tests.e2e.conftest import clear_dapr_registration


def _invoke(graph: Any, graph_input: dict, name: str) -> dict:
    from diagrid.agent.langgraph import DaprWorkflowGraphRunner

    clear_dapr_registration()
    runner = DaprWorkflowGraphRunner(graph=graph, name=name, max_steps=10)
    try:
        runner.start()
        return runner.invoke(input=graph_input, timeout=60)
    finally:
        runner.shutdown()
        clear_dapr_registration()


@pytest.mark.integration
def test_tool_node_and_tools_condition() -> None:
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def agent(state: MessagesState) -> dict:
        if any(isinstance(m, ToolMessage) for m in state["messages"]):
            return {"messages": [AIMessage(content="done")]}
        call = {"name": "add", "args": {"a": 2, "b": 3}, "id": "call-1"}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode([add]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    result = _invoke(
        graph.compile(),
        {"messages": [{"role": "user", "content": "2 + 3?"}]},
        "e2e-toolnode",
    )

    messages = result["messages"]
    assert [m["content"] for m in messages if m["type"] == "tool"] == ["5"]
    assert messages[-1]["content"] == "done"


class _State(TypedDict, total=False):
    value: int
    path: str
    seen_runtime: bool


@pytest.mark.integration
def test_node_with_runtime_parameter() -> None:
    from langgraph.graph import END, START, StateGraph
    from langgraph.runtime import Runtime

    def capture(state: _State, runtime: Runtime) -> dict:
        return {"seen_runtime": runtime is not None}

    graph = StateGraph(_State)
    graph.add_node("capture", capture)
    graph.add_edge(START, "capture")
    graph.add_edge("capture", END)

    result = _invoke(graph.compile(), {"value": 1}, "e2e-runtime-param")

    assert result["seen_runtime"] is True


@pytest.mark.integration
def test_router_labels_go_through_path_map() -> None:
    from langgraph.graph import END, START, StateGraph

    def route(state: _State) -> str:
        return "big" if state["value"] > 100 else "small"

    graph = StateGraph(_State)
    graph.add_node("classify", lambda state: {"path": ""})
    graph.add_node("high", lambda state: {"path": "high"})
    graph.add_node("low", lambda state: {"path": "low"})
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route, {"big": "high", "small": "low"})
    graph.add_edge("high", END)
    graph.add_edge("low", END)

    result = _invoke(graph.compile(), {"value": 150}, "e2e-path-map")

    assert result["path"] == "high"


@pytest.mark.integration
def test_async_node_and_router() -> None:
    from langgraph.graph import END, START, StateGraph

    async def double(state: _State) -> dict:
        return {"value": state["value"] * 2}

    async def route(state: _State) -> str:
        return "high" if state["value"] > 100 else "low"

    graph = StateGraph(_State)
    graph.add_node("double", double)
    graph.add_node("high", lambda state: {"path": "high"})
    graph.add_node("low", lambda state: {"path": "low"})
    graph.add_edge(START, "double")
    graph.add_conditional_edges("double", route)
    graph.add_edge("high", END)
    graph.add_edge("low", END)

    result = _invoke(graph.compile(), {"value": 60}, "e2e-async")

    assert result["value"] == 120
    assert result["path"] == "high"
