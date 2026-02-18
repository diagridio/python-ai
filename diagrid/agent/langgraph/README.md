# Diagrid Agent LangGraph

Durable execution of LangGraph graphs using Dapr Workflows.

This package enables durable execution of LangGraph graphs using Dapr Workflows.
Each node execution in a graph runs as a separate Dapr Workflow activity, providing:

- **Fault tolerance**: Graphs automatically resume from the last successful node on failure
- **Durability**: Graph state is persisted and can survive process restarts
- **Observability**: Full visibility into graph execution through Dapr's workflow APIs

## Installation

```bash
pip install diagrid
```

## Quick Start

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from diagrid.agent.langgraph import DaprWorkflowGraphRunner

# Define your state
class State(TypedDict):
    messages: list[str]
    count: int

# Define node functions
def process_node(state: State) -> dict:
    return {
        "messages": state["messages"] + ["processed"],
        "count": state["count"] + 1
    }

def validate_node(state: State) -> dict:
    return {"messages": state["messages"] + ["validated"]}

# Build the graph
graph = StateGraph(State)
graph.add_node("process", process_node)
graph.add_node("validate", validate_node)
graph.add_edge(START, "process")
graph.add_edge("process", "validate")
graph.add_edge("validate", END)
compiled = graph.compile()

# Run with Dapr Workflows
runner = DaprWorkflowGraphRunner(graph=compiled)
runner.start()

try:
    result = runner.invoke(
        input={"messages": ["start"], "count": 0},
        thread_id="my-thread-123",
    )
    print(result)
    # Output: {"messages": ["start", "processed", "validated"], "count": 1}
finally:
    runner.shutdown()
```

## Async Usage

```python
import asyncio

async def main():
    runner = DaprWorkflowGraphRunner(graph=compiled)
    runner.start()

    try:
        async for event in runner.run_async(
            input={"messages": ["start"], "count": 0},
            thread_id="my-thread-123",
        ):
            print(f"Event: {event['type']}")
            if event["type"] == "workflow_completed":
                print(f"Output: {event['output']}")
    finally:
        runner.shutdown()

asyncio.run(main())
```

## Architecture

```text
User's LangGraph
      |
      v
DaprWorkflowGraphRunner
      |
      v
START WORKFLOW: langgraph_workflow
      |
      +--> Activity: execute_node (node_a)
      |         |
      |         v
      +--> Activity: execute_node (node_b)
      |         |
      |         v
      +--> ... (continues until END)
      |
      v
Return final state
```

Each node execution is a durable Dapr Workflow activity with:
- Automatic retries on failure
- State persistence between activities
- Full observability through Dapr APIs

## Requirements

- Python >= 3.10
- Dapr >= 1.16.0
- LangGraph >= 1.0.0
- Running Dapr sidecar

## Links

- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Dapr](https://dapr.io/)
