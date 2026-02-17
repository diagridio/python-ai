"""
Simple LangGraph with Dapr Workflow Example

This example demonstrates how to run a simple LangGraph with durable execution
using Dapr Workflows. Each node execution becomes a separate Dapr activity.

Prerequisites:
    - Redis running on localhost:6379
    - Install dependencies: pip install diagrid langgraph

Run with:
    cd diagrid/agent/langgraph/examples
    dapr run --app-id langgraph-simple --resources-path ./components -- python3 simple_graph.py
"""

from __future__ import annotations

import asyncio
from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END

from diagrid.agent.langgraph import DaprWorkflowGraphRunner


# Define the state schema
class State(TypedDict):
    messages: List[str]
    counter: int


# Define node functions
def process_node(state: State) -> dict:
    """Process the input and add a message."""
    print(f"  [process_node] Received state: {state}", flush=True)
    return {
        "messages": state["messages"] + ["processed by node A"],
        "counter": state["counter"] + 1,
    }


def validate_node(state: State) -> dict:
    """Validate the processed data."""
    print(f"  [validate_node] Received state: {state}", flush=True)
    return {
        "messages": state["messages"] + ["validated by node B"],
        "counter": state["counter"] + 1,
    }


def finalize_node(state: State) -> dict:
    """Finalize the output."""
    print(f"  [finalize_node] Received state: {state}", flush=True)
    return {
        "messages": state["messages"] + ["finalized by node C"],
        "counter": state["counter"] + 1,
    }


def build_graph() -> StateGraph:
    """Build the LangGraph."""
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("process", process_node)
    graph.add_node("validate", validate_node)
    graph.add_node("finalize", finalize_node)

    # Add edges: START -> process -> validate -> finalize -> END
    graph.add_edge(START, "process")
    graph.add_edge("process", "validate")
    graph.add_edge("validate", "finalize")
    graph.add_edge("finalize", END)

    return graph


async def main():
    print("=" * 60, flush=True)
    print("LangGraph with Dapr Workflow - Simple Example", flush=True)
    print("=" * 60, flush=True)

    # Build and compile the graph
    graph = build_graph()
    compiled = graph.compile()

    print("\nGraph structure:", flush=True)
    print("  START -> process -> validate -> finalize -> END", flush=True)
    print("-" * 60, flush=True)

    # Create the Dapr Workflow runner
    runner = DaprWorkflowGraphRunner(
        graph=compiled,
        max_steps=50,
        name="simple_graph",
    )

    # Start the workflow runtime
    print("\nStarting Dapr Workflow runtime...", flush=True)
    runner.start()
    print("Runtime started!", flush=True)

    try:
        # Run the graph
        input_state = {"messages": ["initial message"], "counter": 0}
        print(f"\nInput: {input_state}", flush=True)
        print("-" * 60, flush=True)

        print("\nRunning graph (each node is a durable activity)...\n", flush=True)

        async for event in runner.run_async(
            input=input_state,
            thread_id="simple-example-001",
        ):
            event_type = event["type"]

            if event_type == "workflow_started":
                print(f"[Workflow] Started: {event.get('workflow_id')}", flush=True)
                print(f"[Workflow] Graph: {event.get('graph_name')}", flush=True)

            elif event_type == "workflow_status_changed":
                print(f"[Workflow] Status: {event.get('status')}", flush=True)

            elif event_type == "workflow_completed":
                print("\n[Workflow] Completed!", flush=True)
                print(f"  Steps: {event.get('steps')}", flush=True)
                print(f"  Status: {event.get('status')}", flush=True)
                print(f"\nOutput: {event.get('output')}", flush=True)

            elif event_type == "workflow_failed":
                error = event.get("error")
                print("\n[Workflow] Failed!", flush=True)
                if isinstance(error, dict):
                    print(f"  Message: {error.get('message')}", flush=True)
                else:
                    print(f"  Error: {error}", flush=True)

            elif event_type == "workflow_error":
                print(f"\n[Workflow] Error: {event.get('error')}", flush=True)

        print("\n" + "=" * 60, flush=True)
        print("Workflow execution complete!", flush=True)
        print("=" * 60, flush=True)

    finally:
        # Shutdown the runtime
        print("\nShutting down workflow runtime...", flush=True)
        runner.shutdown()
        print("Done!", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
