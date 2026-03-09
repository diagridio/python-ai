# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Diagrid Agent LangGraph - Durable execution of LangGraph graphs using Dapr Workflows.

This extension enables durable execution of LangGraph graphs using Dapr Workflows.
Each node execution becomes a Dapr Workflow activity, providing fault tolerance,
durability, and observability.

Example:
    ```python
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict
    from diagrid.agent.langgraph import DaprWorkflowGraphRunner

    # Define your state and graph
    class State(TypedDict):
        messages: list[str]

    def node_a(state: State) -> dict:
        return {"messages": state["messages"] + ["from A"]}

    graph = StateGraph(State)
    graph.add_node("a", node_a)
    graph.add_edge(START, "a")
    graph.add_edge("a", END)
    compiled = graph.compile()

    # Run with Dapr Workflows
    runner = DaprWorkflowGraphRunner(graph=compiled)
    runner.start()

    result = runner.invoke(
        input={"messages": ["hello"]},
        thread_id="thread-123",
    )
    print(result)  # {"messages": ["hello", "from A"]}

    runner.shutdown()
    ```
"""

from diagrid.agent.langgraph.runner import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.state import DaprMemoryCheckpointer
from diagrid.agent.langgraph.models import (
    ChannelState,
    EdgeConfig,
    ExecuteNodeInput,
    ExecuteNodeOutput,
    EvaluateConditionInput,
    EvaluateConditionOutput,
    GraphConfig,
    GraphWorkflowInput,
    GraphWorkflowOutput,
    NodeConfig,
    NodeWrite,
    WorkflowStatus,
)
from diagrid.agent.langgraph.workflow import (
    agent_workflow,
    execute_node_activity,
    evaluate_condition_activity,
    register_node,
    register_condition,
    register_channel_reducer,
    clear_registries,
)

__all__ = [
    # Main runner
    "DaprWorkflowGraphRunner",
    # State
    "DaprMemoryCheckpointer",
    # Models
    "ChannelState",
    "EdgeConfig",
    "ExecuteNodeInput",
    "ExecuteNodeOutput",
    "EvaluateConditionInput",
    "EvaluateConditionOutput",
    "GraphConfig",
    "GraphWorkflowInput",
    "GraphWorkflowOutput",
    "NodeConfig",
    "NodeWrite",
    "WorkflowStatus",
    # Workflow components (for advanced use)
    "agent_workflow",
    "execute_node_activity",
    "evaluate_condition_activity",
    "register_node",
    "register_condition",
    "register_channel_reducer",
    "clear_registries",
]
