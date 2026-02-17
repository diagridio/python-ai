# Copyright 2025 Diagrid Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    langgraph_workflow,
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
    "langgraph_workflow",
    "execute_node_activity",
    "evaluate_condition_activity",
    "register_node",
    "register_condition",
    "register_channel_reducer",
    "clear_registries",
]
