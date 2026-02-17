"""
Conditional LangGraph with Dapr Workflow Example

This example demonstrates how to use conditional edges in a LangGraph
with durable execution using Dapr Workflows.

Prerequisites:
    - Redis running on localhost:6379
    - Install dependencies: pip install diagrid langgraph

Run with:
    cd diagrid/agent/langgraph/examples
    dapr run --app-id langgraph-conditional --resources-path ./components -- python3 conditional_graph.py
"""

from __future__ import annotations

import asyncio
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END

from diagrid.agent.langgraph import DaprWorkflowGraphRunner


# Define the state schema
class State(TypedDict):
    input_value: int
    path_taken: str
    result: str


# Define node functions
def classifier_node(state: State) -> dict:
    """Classify the input and determine the path."""
    print(f"  [classifier] Input value: {state['input_value']}", flush=True)

    if state["input_value"] > 100:
        path = "high"
    elif state["input_value"] > 50:
        path = "medium"
    else:
        path = "low"

    print(f"  [classifier] Path determined: {path}", flush=True)
    return {"path_taken": path}


def high_value_processor(state: State) -> dict:
    """Process high value inputs."""
    print(
        f"  [high_processor] Processing high value: {state['input_value']}", flush=True
    )
    return {"result": f"High value processed: {state['input_value'] * 2}"}


def medium_value_processor(state: State) -> dict:
    """Process medium value inputs."""
    print(
        f"  [medium_processor] Processing medium value: {state['input_value']}",
        flush=True,
    )
    return {"result": f"Medium value processed: {state['input_value'] * 1.5}"}


def low_value_processor(state: State) -> dict:
    """Process low value inputs."""
    print(f"  [low_processor] Processing low value: {state['input_value']}", flush=True)
    return {"result": f"Low value processed: {state['input_value']}"}


def finalizer_node(state: State) -> dict:
    """Finalize the processing."""
    print(f"  [finalizer] Finalizing with result: {state['result']}", flush=True)
    return {"result": f"Final: {state['result']} (path: {state['path_taken']})"}


# Routing function for conditional edge
def route_by_value(state: State) -> Literal["high", "medium", "low"]:
    """Route to appropriate processor based on input value."""
    if state["input_value"] > 100:
        return "high"
    elif state["input_value"] > 50:
        return "medium"
    else:
        return "low"


def build_graph() -> StateGraph:
    """Build the LangGraph with conditional routing."""
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("classifier", classifier_node)
    graph.add_node("high", high_value_processor)
    graph.add_node("medium", medium_value_processor)
    graph.add_node("low", low_value_processor)
    graph.add_node("finalizer", finalizer_node)

    # Add edges
    graph.add_edge(START, "classifier")

    # Conditional edge based on classification
    graph.add_conditional_edges(
        "classifier",
        route_by_value,
        {
            "high": "high",
            "medium": "medium",
            "low": "low",
        },
    )

    # All processors lead to finalizer
    graph.add_edge("high", "finalizer")
    graph.add_edge("medium", "finalizer")
    graph.add_edge("low", "finalizer")

    # Finalizer leads to END
    graph.add_edge("finalizer", END)

    return graph


async def run_with_value(
    runner: DaprWorkflowGraphRunner, value: int, thread_suffix: str
):
    """Run the graph with a specific input value."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"Testing with input_value = {value}", flush=True)
    print(f"{'=' * 60}", flush=True)

    input_state = {"input_value": value, "path_taken": "", "result": ""}

    async for event in runner.run_async(
        input=input_state,
        thread_id=f"conditional-{thread_suffix}",
    ):
        event_type = event["type"]

        if event_type == "workflow_started":
            print(f"[Workflow] Started: {event.get('workflow_id')}", flush=True)

        elif event_type == "workflow_completed":
            print(f"[Workflow] Completed!", flush=True)
            output = event.get("output", {})
            print(f"  Path taken: {output.get('path_taken', 'unknown')}", flush=True)
            print(f"  Result: {output.get('result', 'N/A')}", flush=True)

        elif event_type == "workflow_failed":
            print(f"[Workflow] Failed: {event.get('error')}", flush=True)


async def main():
    print("=" * 60, flush=True)
    print("LangGraph with Dapr Workflow - Conditional Routing Example", flush=True)
    print("=" * 60, flush=True)

    # Build and compile the graph
    graph = build_graph()
    compiled = graph.compile()

    print("\nGraph structure:", flush=True)
    print("  START -> classifier -> [high|medium|low] -> finalizer -> END", flush=True)
    print("\nRouting rules:", flush=True)
    print("  value > 100  -> high processor", flush=True)
    print("  value > 50   -> medium processor", flush=True)
    print("  value <= 50  -> low processor", flush=True)
    print("-" * 60, flush=True)

    # Create the Dapr Workflow runner
    runner = DaprWorkflowGraphRunner(
        graph=compiled,
        max_steps=50,
        name="conditional_graph",
    )

    # Start the workflow runtime
    print("\nStarting Dapr Workflow runtime...", flush=True)
    runner.start()
    print("Runtime started!", flush=True)

    try:
        # Test with different values to trigger different paths
        await run_with_value(runner, 150, "high")  # High path
        await run_with_value(runner, 75, "medium")  # Medium path
        await run_with_value(runner, 25, "low")  # Low path

        print("\n" + "=" * 60, flush=True)
        print("All tests complete!", flush=True)
        print("=" * 60, flush=True)

    finally:
        # Shutdown the runtime
        print("\nShutting down workflow runtime...", flush=True)
        runner.shutdown()
        print("Done!", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
